#include "solver/detail/sparse_lu.hpp"

#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace solver::detail {
namespace {

using Index = std::int64_t;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;

class CheckedLu : public Eigen::SparseLU<Matrix, Eigen::NaturalOrdering<Index>> {
 public:
  // Eigen 5.0.0 的部分内存失败路径不刷新 info()，须同时核对因子状态。
  bool factorization_ok() const { return m_factorizationIsOk; }
};

bool all_finite(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

}  // 匿名命名空间

struct SparseLu::Impl {
  Matrix matrix;
  CheckedLu lu;
  std::vector<Index> first;
  std::vector<Index> second;
  bool valid = false;
  Index factor_nnz = 0;

  Impl(std::size_t dimension, std::span<const Index> starts, std::span<const Index> rows) {
    const auto limit = static_cast<std::size_t>(std::min(
        std::numeric_limits<Index>::max(), static_cast<Index>(std::numeric_limits<Eigen::Index>::max())));
    if (dimension >= limit || starts.size() != dimension + 1 || starts.front() != 0 ||
        rows.size() > limit || starts.back() != static_cast<Index>(rows.size())) {
      throw std::invalid_argument("SparseLU 上三角 CSC 维数或列指针无效");
    }
    std::vector<Index> counts(dimension, 0);
    std::size_t full_nnz = rows.size();
    for (std::size_t column = 0; column < dimension; ++column) {
      if (starts[column] < 0 || starts[column] > starts[column + 1] ||
          starts[column + 1] > static_cast<Index>(rows.size())) {
        throw std::invalid_argument("SparseLU 上三角 CSC 列指针越界");
      }
      Index previous = -1;
      for (Index entry = starts[column]; entry < starts[column + 1]; ++entry) {
        const Index row = rows[static_cast<std::size_t>(entry)];
        if (row <= previous || row > static_cast<Index>(column)) {
          throw std::invalid_argument("SparseLU 每列必须为无重复且升序的上三角行索引");
        }
        previous = row;
        ++counts[column];
        if (row != static_cast<Index>(column)) {
          if (full_nnz == limit) throw std::length_error("SparseLU 对称展开的非零数溢出");
          ++full_nnz;
          ++counts[static_cast<std::size_t>(row)];
        }
      }
    }
    matrix.resize(static_cast<Eigen::Index>(dimension), static_cast<Eigen::Index>(dimension));
    matrix.resizeNonZeros(static_cast<Eigen::Index>(full_nnz));
    Index* outer = matrix.outerIndexPtr();
    outer[0] = 0;
    for (std::size_t column = 0; column < dimension; ++column) outer[column + 1] = outer[column] + counts[column];
    std::vector<Index> next(outer, outer + dimension);
    first.resize(rows.size());
    second.assign(rows.size(), -1);
    for (std::size_t column = 0; column < dimension; ++column) {
      for (Index entry = starts[column]; entry < starts[column + 1]; ++entry) {
        const auto source = static_cast<std::size_t>(entry);
        const Index row = rows[source];
        const Index direct = next[column]++;
        matrix.innerIndexPtr()[direct] = row;
        matrix.valuePtr()[direct] = 1.0;
        first[source] = direct;
        if (row != static_cast<Index>(column)) {
          const Index mirror = next[static_cast<std::size_t>(row)]++;
          matrix.innerIndexPtr()[mirror] = static_cast<Index>(column);
          matrix.valuePtr()[mirror] = 1.0;
          second[source] = mirror;
        }
      }
    }
    lu.setPivotThreshold(1.0);
    // 输入严格对称；禁止额外 etree 后序排列，数值行选主元仍保留。
    lu.isSymmetric(true);
    if (dimension != 0) lu.analyzePattern(matrix);
  }
};

SparseLu::SparseLu(std::size_t dimension, std::span<const Index> starts, std::span<const Index> rows)
    : impl_(std::make_unique<Impl>(dimension, starts, rows)) {}

SparseLu::~SparseLu() = default;

bool SparseLu::factor(std::span<const double> values) {
  auto& state = *impl_;
  state.valid = false;
  state.factor_nnz = 0;
  if (values.size() != state.first.size() || !all_finite(values)) return false;
  for (std::size_t entry = 0; entry < values.size(); ++entry) {
    state.matrix.valuePtr()[state.first[entry]] = values[entry];
    if (state.second[entry] >= 0) state.matrix.valuePtr()[state.second[entry]] = values[entry];
  }
  if (state.matrix.rows() != 0) {
    state.lu.factorize(state.matrix);
    if (!state.lu.factorization_ok() || state.lu.info() != Eigen::Success) return false;
    const auto lower = state.lu.nnzL();
    const auto upper = state.lu.nnzU();
    if (lower < 0 || upper < 0 || lower > std::numeric_limits<Index>::max() - upper) {
      throw std::runtime_error("SparseLU 因子非零数无效或溢出");
    }
    state.factor_nnz = static_cast<Index>(lower + upper);
  }
  state.valid = true;
  return true;
}

bool SparseLu::solve(std::vector<double>& rhs) const {
  const auto& state = *impl_;
  if (!state.valid || rhs.size() != static_cast<std::size_t>(state.matrix.rows()) || !all_finite(rhs)) return false;
  if (rhs.empty()) return true;
  const Eigen::Map<const Eigen::VectorXd> input(rhs.data(), static_cast<Eigen::Index>(rhs.size()));
  const Eigen::VectorXd solution = state.lu.solve(input);
  if (!solution.allFinite()) return false;
  std::copy(solution.data(), solution.data() + solution.size(), rhs.begin());
  return true;
}

Index SparseLu::factor_nonzeros() const { return impl_->factor_nnz; }

}  // 命名空间 solver::detail
