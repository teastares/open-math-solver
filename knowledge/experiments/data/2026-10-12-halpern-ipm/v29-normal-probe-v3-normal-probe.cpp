#include <cstdio>
#include "solver/detail/normal_cholesky.hpp"
#include "solver/detail/multifrontal_ldl.hpp"

#include <Eigen/SparseCholesky>
#include <amd.h>
#include <qdldl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace solver::detail {
namespace {

using Index = std::int64_t;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;
using Triplet = Eigen::Triplet<double, Index>;
constexpr Index contribution_budget = 8'000'000;
constexpr Index structural_budget = 16'777'216;

class CheckedCholesky : public Eigen::SimplicialLLT<Matrix, Eigen::Upper, Eigen::NaturalOrdering<Index>> {
 public:
  Index stored_nonzeros() const { return m_matrix.nonZeros(); }
  bool finite_factor(const std::function<void()>& budget_check) const {
    // Eigen 的 info() 不保证捕获分解中间产生的 NaN，独立核对存储因子。
    for (Index col = 0; col < m_matrix.cols(); ++col) {
      if (budget_check) budget_check();
      bool diagonal = false;
      for (Matrix::InnerIterator entry(m_matrix, col); entry; ++entry) {
        if (!std::isfinite(entry.value())) return false;
        if (entry.row() == col) {
          if (!(entry.value() > 0)) return false;
          diagonal = true;
        }
      }
      if (!diagonal) return false;
    }
    return true;
  }
};

bool representable(long double value, double& output) {
  output = static_cast<double>(value);
  return std::isfinite(value) && std::isfinite(output) && (value == 0 || output != 0);
}

void check(const std::function<void()>& budget_check) {
  if (budget_check) budget_check();
}

}  // 匿名命名空间

struct NormalCholesky::Impl {
  struct Entry { Index row; Index source; };
  std::size_t dimension;
  std::size_t input_nonzeros;
  std::vector<Index> primal_coordinates, dual_coordinates;
  std::vector<Index> primal_diagonal, dual_diagonal;
  std::vector<std::vector<Entry>> columns;
  std::vector<Index> inverse, diagonal_slots, product_slots;
  std::vector<double> input_values;
  std::vector<long double> sums;
  Matrix matrix;
  CheckedCholesky cholesky;
  std::unique_ptr<MultifrontalLdl> multifrontal;
  std::vector<double> factor_scaling;
  unsigned long long diagnostic_factor = 0;
  mutable unsigned long long diagnostic_solves = 0;
  bool valid = false;
  Index factor_nnz = 0;
  Index modified_pivots = 0;

  Index slot(Index row, Index col) const {
    if (row > col) std::swap(row, col);
    const Index begin = matrix.outerIndexPtr()[col];
    const Index end = matrix.outerIndexPtr()[col + 1];
    const auto found = std::lower_bound(matrix.innerIndexPtr() + begin, matrix.innerIndexPtr() + end, row);
    if (found == matrix.innerIndexPtr() + end || *found != row) {
      throw std::runtime_error("正规方程固定结构缺少预期项");
    }
    return static_cast<Index>(found - matrix.innerIndexPtr());
  }

  Impl(std::size_t size, std::span<const Index> starts, std::span<const Index> rows,
       std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool use_multifrontal)
      : dimension(size), input_nonzeros(rows.size()) {
    check(budget_check);
    if (size >= static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        starts.size() != size + 1 || signs.size() != size || starts.front() != 0 ||
        rows.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        starts.back() != static_cast<Index>(rows.size())) {
      throw std::invalid_argument("正规方程上三角 CSC 维数或列指针无效");
    }
    if (size > static_cast<std::size_t>(structural_budget) || rows.size() > static_cast<std::size_t>(structural_budget)) {
      throw NormalCholeskyBudgetExceeded("正规方程输入结构超过 16777216 项预算");
    }
    std::vector<Index> local(size), diagonal(size, -1);
    for (std::size_t coordinate = 0; coordinate < size; ++coordinate) {
      if (signs[coordinate] == 1) {
        local[coordinate] = static_cast<Index>(primal_coordinates.size());
        primal_coordinates.push_back(static_cast<Index>(coordinate));
      } else if (signs[coordinate] == -1) {
        local[coordinate] = static_cast<Index>(dual_coordinates.size());
        dual_coordinates.push_back(static_cast<Index>(coordinate));
      } else {
        throw std::invalid_argument("正规方程坐标标签必须为 +1 或 -1");
      }
    }
    columns.resize(primal_coordinates.size());
    for (std::size_t col = 0; col < size; ++col) {
      check(budget_check);
      if (starts[col] < 0 || starts[col] > starts[col + 1] || starts[col + 1] > static_cast<Index>(rows.size())) {
        throw std::invalid_argument("正规方程上三角 CSC 列指针越界");
      }
      Index previous = -1;
      for (Index entry = starts[col]; entry < starts[col + 1]; ++entry) {
        const Index row = rows[static_cast<std::size_t>(entry)];
        if (row <= previous || row > static_cast<Index>(col)) {
          throw std::invalid_argument("正规方程每列必须为无重复且升序的上三角行索引");
        }
        previous = row;
        if (row == static_cast<Index>(col)) {
          diagonal[col] = entry;
        } else {
          const auto other = static_cast<std::size_t>(row);
          if (signs[col] == signs[other]) throw std::invalid_argument("正规方程要求 B 和 D 均为对角块");
          const auto primal = signs[col] == 1 ? col : other;
          const auto dual = signs[col] == -1 ? col : other;
          columns[static_cast<std::size_t>(local[primal])].push_back({local[dual], entry});
        }
      }
      if (diagonal[col] < 0) throw std::invalid_argument("正规方程要求每个坐标有显式对角项");
    }
    for (Index coordinate : primal_coordinates) primal_diagonal.push_back(diagonal[static_cast<std::size_t>(coordinate)]);
    for (Index coordinate : dual_coordinates) dual_diagonal.push_back(diagonal[static_cast<std::size_t>(coordinate)]);
    const Index retained = static_cast<Index>(dual_coordinates.size());
    Index contributions = retained;
    if (contributions > contribution_budget) throw NormalCholeskyBudgetExceeded("正规方程 Gram 展开超过 8000000 项工作预算");
    for (auto& column : columns) {
      check(budget_check);
      std::sort(column.begin(), column.end(), [](const Entry& a, const Entry& b) { return a.row < b.row; });
      const Index degree = static_cast<Index>(column.size());
      // degree 先受输入结构上限约束，乘法在 int64_t 范围内。
      const Index pairs = degree * (degree + 1) / 2;
      if (pairs > contribution_budget - contributions) {
        throw NormalCholeskyBudgetExceeded("正规方程 Gram 展开超过 8000000 项工作预算");
      }
      contributions += pairs;
    }
    matrix.resize(retained, retained);
    if (retained == 0) return;
    {
      std::vector<Triplet> entries;
      entries.reserve(static_cast<std::size_t>(contributions));
      for (Index row = 0; row < retained; ++row) entries.emplace_back(row, row, 1.0);
      for (const auto& column : columns) {
        for (std::size_t right = 0; right < column.size(); ++right) {
          check(budget_check);
          for (std::size_t left = 0; left <= right; ++left) entries.emplace_back(column[left].row, column[right].row, 1.0);
        }
      }
      matrix.setFromTriplets(entries.begin(), entries.end());
    }
    check(budget_check);
    if (matrix.nonZeros() > structural_budget) throw NormalCholeskyBudgetExceeded("正规方程矩阵超过 16777216 个结构项预算");
    std::vector<Index> permutation(static_cast<std::size_t>(retained));
    inverse.resize(static_cast<std::size_t>(retained));
    const auto status = amd_l_order(retained, matrix.outerIndexPtr(), matrix.innerIndexPtr(), permutation.data(), nullptr, nullptr);
    check(budget_check);
    if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) throw std::runtime_error("正规方程 AMD 排序失败");
    for (Index row = 0; row < retained; ++row) inverse[static_cast<std::size_t>(permutation[static_cast<std::size_t>(row)])] = row;
    {
      std::vector<Triplet> entries;
      entries.reserve(static_cast<std::size_t>(matrix.nonZeros()));
      for (Index col = 0; col < retained; ++col) {
        check(budget_check);
        for (Matrix::InnerIterator entry(matrix, col); entry; ++entry) {
          const auto a = inverse[static_cast<std::size_t>(entry.row())];
          const auto b = inverse[static_cast<std::size_t>(col)];
          entries.emplace_back(std::min(a, b), std::max(a, b), 1.0);
        }
      }
      Matrix ordered(retained, retained);
      ordered.setFromTriplets(entries.begin(), entries.end());
      matrix.swap(ordered);
    }
    Index lower_nonzeros;
    {
      // int64_t 与 QDLDL_int 在不同平台可能是不同类型，不能重解释索引指针。
      std::vector<QDLDL_int> starts(matrix.outerIndexPtr(), matrix.outerIndexPtr() + retained + 1);
      std::vector<QDLDL_int> rows(matrix.innerIndexPtr(), matrix.innerIndexPtr() + matrix.nonZeros());
      std::vector<QDLDL_int> work(static_cast<std::size_t>(retained)), counts(work.size()), tree(work.size());
      check(budget_check);
      const auto count = QDLDL_etree(static_cast<QDLDL_int>(retained), starts.data(), rows.data(), work.data(), counts.data(), tree.data());
      check(budget_check);
      if (count < 0) throw std::runtime_error("正规方程符号因子计数失败");
      if (count > structural_budget - retained) throw NormalCholeskyBudgetExceeded("正规方程 Cholesky 因子超过 16777216 个结构项预算");
      lower_nonzeros = static_cast<Index>(count);
    }
    if (use_multifrontal) {
      // 同一 AMD 后的正定 Schur 矩阵；复用已有分块 LDL，坐标符号全部为正。
      // 分块内核可能正则小主元，外层仍必须检查完整未正则化 KKT。
      multifrontal = std::make_unique<MultifrontalLdl>(static_cast<std::size_t>(retained),
          std::span<const Index>(matrix.outerIndexPtr(), static_cast<std::size_t>(retained + 1)),
          std::span<const Index>(matrix.innerIndexPtr(), static_cast<std::size_t>(matrix.nonZeros())),
          std::vector<std::int8_t>(static_cast<std::size_t>(retained), 1), budget_check);
    } else {
      cholesky.analyzePattern(matrix);
      check(budget_check);
      if (cholesky.info() != Eigen::Success || cholesky.stored_nonzeros() != lower_nonzeros + retained) {
        throw std::runtime_error("正规方程 Cholesky 符号结构与预检不一致");
      }
    }
    diagonal_slots.resize(static_cast<std::size_t>(retained));
    for (Index row = 0; row < retained; ++row) diagonal_slots[static_cast<std::size_t>(row)] = slot(inverse[static_cast<std::size_t>(row)], inverse[static_cast<std::size_t>(row)]);
    product_slots.reserve(static_cast<std::size_t>(contributions - retained));
    for (const auto& column : columns) {
      for (std::size_t right = 0; right < column.size(); ++right) {
        check(budget_check);
        for (std::size_t left = 0; left <= right; ++left) {
          product_slots.push_back(slot(inverse[static_cast<std::size_t>(column[left].row)], inverse[static_cast<std::size_t>(column[right].row)]));
        }
      }
    }
    sums.resize(static_cast<std::size_t>(matrix.nonZeros()));
    check(budget_check);
  }
};

NormalCholesky::NormalCholesky(std::size_t size, std::span<const Index> starts, std::span<const Index> rows,
    std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool multifrontal)
    : impl_(std::make_unique<Impl>(size, starts, rows, signs, budget_check, multifrontal)) {}

NormalCholesky::~NormalCholesky() = default;

bool NormalCholesky::factor(std::span<const double> values, const std::function<void()>& budget_check) {
  auto& state = *impl_;
  ++state.diagnostic_factor;
  state.diagnostic_solves = 0;
  state.valid = false;
  state.factor_nnz = 0;
  state.modified_pivots = 0;
  check(budget_check);
  if (values.size() != state.input_nonzeros || !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); })) return false;
  for (Index source : state.primal_diagonal) if (!(values[static_cast<std::size_t>(source)] > 0)) return false;
  for (Index source : state.dual_diagonal) if (!(values[static_cast<std::size_t>(source)] < 0)) return false;
  state.input_values.assign(values.begin(), values.end());
  std::fill(state.sums.begin(), state.sums.end(), 0);
  for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
    state.sums[static_cast<std::size_t>(state.diagonal_slots[row])] = -static_cast<long double>(values[static_cast<std::size_t>(state.dual_diagonal[row])]);
  }
  std::size_t product = 0;
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const long double diagonal = values[static_cast<std::size_t>(state.primal_diagonal[col])];
    const auto& column = state.columns[col];
    for (std::size_t right = 0; right < column.size(); ++right) {
      check(budget_check);
      const double b = values[static_cast<std::size_t>(column[right].source)];
      for (std::size_t left = 0; left <= right; ++left) {
        const double a = values[static_cast<std::size_t>(column[left].source)];
        const long double contribution = static_cast<long double>(a) * b / diagonal;
        if (!std::isfinite(contribution) || (a != 0 && b != 0 && contribution == 0)) return false;
        auto& sum = state.sums[static_cast<std::size_t>(state.product_slots[product++])];
        sum += contribution;
        if (!std::isfinite(sum)) return false;
      }
    }
  }
  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {
    if (!representable(state.sums[entry], state.matrix.valuePtr()[entry])) return false;
  }
  check(budget_check);
  if (!state.dual_coordinates.empty()) {
    if (state.multifrontal) {
      // 用已形成 Schur 矩阵的对角作对称均衡：T S T w = T q，z = T w。
      // 缩放作用于已经舍入到 double 的 S，不改变 Gram 装配；外层仍验完整 KKT。
      const auto retained = state.dual_coordinates.size();
      state.factor_scaling.resize(retained);
      for (std::size_t row = 0; row < retained; ++row) {
        check(budget_check);
        const double diagonal = state.matrix.valuePtr()[state.diagonal_slots[row]];
        if (!(diagonal > 0) || !representable(1 / std::sqrt(static_cast<long double>(diagonal)),
            state.factor_scaling[static_cast<std::size_t>(state.inverse[row])])) return false;
      }
      for (Index col = 0; col < state.matrix.cols(); ++col) {
        check(budget_check);
        for (Matrix::InnerIterator entry(state.matrix, col); entry; ++entry) {
          if (!representable(static_cast<long double>(entry.value()) *
              state.factor_scaling[static_cast<std::size_t>(entry.row())] *
              state.factor_scaling[static_cast<std::size_t>(col)], entry.valueRef())) return false;
        }
      }
      const bool success = state.multifrontal->factor(
          std::span<const double>(state.matrix.valuePtr(), state.sums.size()), budget_check);
      state.modified_pivots = state.multifrontal->regularized_pivots();
      if (!success) return false;
      state.factor_nnz = state.multifrontal->factor_nonzeros();
    } else {
      state.cholesky.factorize(state.matrix);
      check(budget_check);
      if (state.cholesky.info() != Eigen::Success || !state.cholesky.finite_factor(budget_check)) return false;
      state.factor_nnz = state.cholesky.stored_nonzeros();
    }
  }
  check(budget_check);
  state.valid = true;
  return true;
}

bool NormalCholesky::solve(std::vector<double>& rhs, const std::function<void()>& budget_check) const {
  const auto& state = *impl_;
  check(budget_check);
  if (!state.valid || rhs.size() != state.dimension || !std::all_of(rhs.begin(), rhs.end(), [](double x) { return std::isfinite(x); })) return false;
  const auto retained = state.dual_coordinates.size();
  std::vector<long double> reduced_rhs(retained);
  for (std::size_t row = 0; row < retained; ++row) reduced_rhs[row] = -static_cast<long double>(rhs[static_cast<std::size_t>(state.dual_coordinates[row])]);
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const long double value = static_cast<long double>(rhs[static_cast<std::size_t>(state.primal_coordinates[col])]) /
        state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    if (!std::isfinite(value)) return false;
    for (const auto& entry : state.columns[col]) {
      reduced_rhs[static_cast<std::size_t>(entry.row)] += state.input_values[static_cast<std::size_t>(entry.source)] * value;
    }
  }
  Eigen::VectorXd permuted(static_cast<Eigen::Index>(retained));
  for (std::size_t row = 0; row < retained; ++row) {
    const long double value = state.multifrontal ? reduced_rhs[row] *
        state.factor_scaling[static_cast<std::size_t>(state.inverse[row])] : reduced_rhs[row];
    if (!representable(value, permuted[state.inverse[row]])) return false;
  }
  const Eigen::VectorXd diagnostic_factor_rhs = permuted;
  std::vector<double> diagnostic_factor_solution;
  if (retained != 0) {
    check(budget_check);
    if (state.multifrontal) {
      std::vector<double> solution(permuted.data(), permuted.data() + permuted.size());
      if (!state.multifrontal->solve(solution, budget_check)) return false;
      diagnostic_factor_solution = solution;
      for (std::size_t row = 0; row < retained; ++row) {
        if (!representable(static_cast<long double>(solution[row]) * state.factor_scaling[row],
            permuted[static_cast<Eigen::Index>(row)])) return false;
      }
    } else {
      const Eigen::VectorXd solution = state.cholesky.solve(permuted);
      check(budget_check);
      if (!solution.allFinite()) return false;
      diagnostic_factor_solution.assign(solution.data(), solution.data() + solution.size());
      permuted = solution;
    }
  }
  std::vector<double> solution(rhs.size());
  for (std::size_t row = 0; row < retained; ++row) solution[static_cast<std::size_t>(state.dual_coordinates[row])] = permuted[state.inverse[row]];
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
    long double value = rhs[coordinate];
    for (const auto& entry : state.columns[col]) value -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) * permuted[state.inverse[static_cast<std::size_t>(entry.row)]];
    value /= state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    if (!representable(value, solution[coordinate])) return false;
  }
  check(budget_check);

  // 所有额外乘积只用于诊断；不会反馈给因子、方向或接受条件。
  std::vector<long double> formed(retained, 0), rounded(retained, 0), composed(retained, 0);
  std::vector<long double> factored(retained, 0), factor_abs(retained, 0);
  for (Index col = 0; col < state.matrix.cols(); ++col) {
    for (Matrix::InnerIterator entry(state.matrix, col); entry; ++entry) {
      const auto i = static_cast<std::size_t>(entry.row()), j = static_cast<std::size_t>(col);
      const auto k = static_cast<std::size_t>(&entry.value() - state.matrix.valuePtr());
      const long double s = state.sums[k], d = static_cast<double>(s), a = entry.value();
      formed[i] += s * permuted[col];
      rounded[i] += d * permuted[col];
      factored[i] += a * diagnostic_factor_solution[j];
      factor_abs[i] += std::abs(a * diagnostic_factor_solution[j]);
      if (i != j) {
        formed[j] += s * permuted[entry.row()];
        rounded[j] += d * permuted[entry.row()];
        factored[j] += a * diagnostic_factor_solution[i];
        factor_abs[j] += std::abs(a * diagnostic_factor_solution[i]);
      }
    }
  }
  std::vector<long double> kkt(solution.size(), 0);
  long double cancellation = 0;
  for (std::size_t row = 0; row < retained; ++row) {
    const auto coordinate = static_cast<std::size_t>(state.dual_coordinates[row]);
    const long double d = state.input_values[static_cast<std::size_t>(state.dual_diagonal[row])];
    composed[row] = -d * solution[coordinate];
    kkt[coordinate] = d * solution[coordinate];
  }
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
    const long double b = state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    long double dot = 0, absolute = std::abs(static_cast<long double>(rhs[coordinate]));
    kkt[coordinate] = b * solution[coordinate];
    for (const auto& entry : state.columns[col]) {
      const auto dual = static_cast<std::size_t>(state.dual_coordinates[static_cast<std::size_t>(entry.row)]);
      const long double g = state.input_values[static_cast<std::size_t>(entry.source)];
      dot += g * solution[dual];
      absolute += std::abs(g * solution[dual]);
      kkt[coordinate] += g * solution[dual];
      kkt[dual] += g * solution[coordinate];
    }
    const long double denominator = std::abs(static_cast<long double>(rhs[coordinate]) - dot);
    cancellation = std::max(cancellation, absolute / std::max(denominator, std::numeric_limits<long double>::min()));
    for (const auto& entry : state.columns[col]) {
      composed[static_cast<std::size_t>(entry.row)] +=
          state.input_values[static_cast<std::size_t>(entry.source)] * (dot / b);
    }
  }
  long double formed_error = 0, rounded_error = 0, composed_error = 0, assembly_effect = 0, cast_effect = 0;
  long double factor_error = 0, factor_berr = 0, factor_rhs_norm = 0, schur_rhs_norm = 0;
  long double primal_error = 0, dual_error = 0, primal_rhs = 0, dual_rhs = 0;
  for (std::size_t row = 0; row < retained; ++row) {
    const auto p = static_cast<std::size_t>(state.inverse[row]);
    const long double q = reduced_rhs[row], scaled_q = diagnostic_factor_rhs[static_cast<Eigen::Index>(p)];
    formed_error = std::max(formed_error, std::abs(q - formed[p]));
    rounded_error = std::max(rounded_error, std::abs(q - rounded[p]));
    composed_error = std::max(composed_error, std::abs(q - composed[row]));
    assembly_effect = std::max(assembly_effect, std::abs(formed[p] - composed[row]));
    cast_effect = std::max(cast_effect, std::abs(formed[p] - rounded[p]));
    const long double error = std::abs(scaled_q - factored[p]);
    const long double denominator = std::abs(scaled_q) + factor_abs[p];
    factor_error = std::max(factor_error, error);
    factor_berr = std::max(factor_berr, error == 0 ? 0 : error / denominator);
    factor_rhs_norm = std::max(factor_rhs_norm, std::abs(scaled_q));
    schur_rhs_norm = std::max(schur_rhs_norm, std::abs(q));
  }
  for (Index coordinate : state.primal_coordinates) {
    const auto i = static_cast<std::size_t>(coordinate);
    primal_error = std::max(primal_error, std::abs(static_cast<long double>(rhs[i]) - kkt[i]));
    primal_rhs = std::max(primal_rhs, std::abs(static_cast<long double>(rhs[i])));
  }
  for (Index coordinate : state.dual_coordinates) {
    const auto i = static_cast<std::size_t>(coordinate);
    dual_error = std::max(dual_error, std::abs(static_cast<long double>(rhs[i]) - kkt[i]));
    dual_rhs = std::max(dual_rhs, std::abs(static_cast<long double>(rhs[i])));
  }
  std::fprintf(stderr, "NORMAL_RESIDUAL {\"factor\":%llu,\"solve\":%llu,\"multifrontal\":%s,\"modified_pivots\":%lld",
      state.diagnostic_factor, ++state.diagnostic_solves, state.multifrontal ? "true" : "false",
      static_cast<long long>(state.modified_pivots));
  const auto emit = [](const char* name, long double value) {
    std::fprintf(stderr, ",\"%s\":", name);
    const double displayed = static_cast<double>(value);
    if (std::isfinite(value) && std::isfinite(displayed) && (value == 0 || displayed != 0))
      std::fprintf(stderr, "%.17g", displayed);
    else std::fputs("null", stderr);
  };
  // Windows UCRT 的可变参数长双精度格式不可靠；运算仍用 long double，输出显式转 double。
  emit("format_sentinel", 1.25L);
  emit("schur_rhs", schur_rhs_norm); emit("formed_residual", formed_error);
  emit("rounded_residual", rounded_error); emit("composed_residual", composed_error);
  emit("assembly_product_difference", assembly_effect); emit("cast_product_difference", cast_effect);
  emit("factor_rhs", factor_rhs_norm); emit("factor_residual", factor_error); emit("factor_berr", factor_berr);
  emit("kkt_primal_residual", primal_error); emit("kkt_dual_residual", dual_error);
  emit("kkt_primal_rhs", primal_rhs); emit("kkt_dual_rhs", dual_rhs); emit("recovery_cancellation", cancellation);
  std::fputs("}\n", stderr);

  rhs = std::move(solution);
  return true;
}

Index NormalCholesky::factor_nonzeros() const { return impl_->factor_nnz; }
Index NormalCholesky::factor_dimension() const { return static_cast<Index>(impl_->dual_coordinates.size()); }
Index NormalCholesky::regularized_pivots() const {
  return impl_->modified_pivots;
}

}  // 命名空间 solver::detail
