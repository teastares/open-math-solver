#include "solver/detail/normal_cholesky.hpp"
#include "solver/detail/multifrontal_ldl.hpp"

#include <Eigen/SparseCholesky>
#include <amd.h>
#include <cholmod.h>
#include <qdldl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

namespace solver::detail {
namespace {

using Index = std::int64_t;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;
using Triplet = Eigen::Triplet<double, Index>;
using ExtendedMatrix = Eigen::SparseMatrix<long double, Eigen::ColMajor, Index>;
using ExtendedVector = Eigen::Matrix<long double, Eigen::Dynamic, 1>;
constexpr Index contribution_budget = 8'000'000;
constexpr Index structural_budget = 16'777'216;
// 有限 binary64 与正对角的单项约在 2^[-3172,3122]，含舍入的至多八百万项部分和小于 2^3146。
// 仅在上述界位于正常数范围的平台省略逐项重复检查；相消允许为零，最终转换和因子检查仍保留。
constexpr bool bounded_gram_arithmetic =
    std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::radix == 2 &&
    std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::max_exponent == 1024 &&
    std::numeric_limits<double>::min_exponent == -1021 && std::numeric_limits<double>::has_denorm == std::denorm_present &&
    std::numeric_limits<long double>::is_iec559 && std::numeric_limits<long double>::radix == 2 &&
    std::numeric_limits<long double>::digits >= 53 &&
    std::numeric_limits<long double>::max_exponent >= 4096 && std::numeric_limits<long double>::min_exponent <= -4096 &&
    contribution_budget <= 8'000'000;

template <class Scalar>
class CheckedCholesky : public Eigen::SimplicialLLT<Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>,
    Eigen::Upper, Eigen::NaturalOrdering<Index>> {
  using Sparse = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;
 public:
  Index stored_nonzeros() const { return this->m_matrix.nonZeros(); }
  bool finite_factor(const std::function<void()>& budget_check) const {
    // Eigen 的 info() 不保证捕获分解中间产生的 NaN，独立核对存储因子。
    for (Index col = 0; col < this->m_matrix.cols(); ++col) {
      if (budget_check) budget_check();
      bool diagonal = false;
      for (typename Sparse::InnerIterator entry(this->m_matrix, col); entry; ++entry) {
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

// 输入已经经过同一 AMD 排列；本适配器不再排序、均衡或修改主元。
class CholmodFactor {
  struct Resources {
    cholmod_common common{};
    cholmod_factor* factor = nullptr;
    cholmod_dense* solution = nullptr;
    cholmod_dense* work = nullptr;
    cholmod_dense* extra = nullptr;

    Resources() {
      if (!cholmod_l_start(&common)) throw std::runtime_error("CHOLMOD 初始化失败");
      common.print = 0;
      common.error_handler = nullptr;
      common.supernodal = CHOLMOD_SUPERNODAL;
      common.nmethods = 1;
      common.method[0].ordering = CHOLMOD_NATURAL;
      common.postorder = 0;
      common.final_asis = 1;
      common.useGPU = 0;
      common.nthreads_max = 1;
      common.dbound = 0;
      common.sbound = 0;
      common.quick_return_if_not_posdef = 1;
    }
    ~Resources() {
      cholmod_l_free_dense(&solution, &common);
      cholmod_l_free_dense(&work, &common);
      cholmod_l_free_dense(&extra, &common);
      cholmod_l_free_factor(&factor, &common);
      cholmod_l_finish(&common);
    }
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
  };

  mutable Resources resources_;
  cholmod_sparse matrix_{};
  cholmod_sparse lower_matrix_{};
  std::vector<Index> lower_starts_, lower_rows_, lower_sources_;
  std::vector<double> lower_values_;
  bool valid_ = false;

  void cache_lower_pattern(const std::function<void()>& budget_check) {
    check(budget_check);
    const auto size = matrix_.nrow;
    const auto nonzeros = matrix_.nzmax;
    if (size != matrix_.ncol || size >= static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        size == std::numeric_limits<std::size_t>::max() || size + 1 > lower_starts_.max_size() ||
        nonzeros > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        nonzeros > lower_rows_.max_size() || nonzeros > lower_sources_.max_size() ||
        nonzeros > lower_values_.max_size()) {
      throw std::length_error("CHOLMOD 下三角缓存尺寸无法表示");
    }
    const auto* starts = static_cast<const Index*>(matrix_.p);
    const auto* rows = static_cast<const Index*>(matrix_.i);
    const auto count = static_cast<Index>(nonzeros);
    if (!starts || (nonzeros != 0 && !rows) || starts[0] != 0 || starts[size] != count) {
      throw std::runtime_error("CHOLMOD 上三角缓存来源无效");
    }
    lower_starts_.assign(size + 1, 0);
    for (std::size_t col = 0; col < size; ++col) {
      check(budget_check);
      if (starts[col] < 0 || starts[col + 1] < starts[col] || starts[col + 1] > count) {
        throw std::runtime_error("CHOLMOD 上三角列指针越界");
      }
      Index previous = -1;
      for (Index source = starts[col]; source < starts[col + 1]; ++source) {
        if ((source & 4095) == 0) check(budget_check);
        const Index row = rows[source];
        if (row <= previous || row > static_cast<Index>(col)) {
          throw std::runtime_error("CHOLMOD 上三角行索引必须严格有序");
        }
        previous = row;
        auto& entries = lower_starts_[static_cast<std::size_t>(row) + 1];
        if (entries == count) throw std::runtime_error("CHOLMOD 下三角列计数越界");
        ++entries;
      }
      if (previous != static_cast<Index>(col)) {
        throw std::runtime_error("CHOLMOD 上三角缺少显式对角项");
      }
    }
    for (std::size_t col = 0; col < size; ++col) {
      if ((col & 4095) == 0) check(budget_check);
      if (lower_starts_[col + 1] > count - lower_starts_[col]) {
        throw std::runtime_error("CHOLMOD 下三角前缀和越界");
      }
      lower_starts_[col + 1] += lower_starts_[col];
    }
    if (lower_starts_.back() != count) throw std::runtime_error("CHOLMOD 转置结构项数不一致");
    lower_rows_.resize(nonzeros);
    lower_sources_.resize(nonzeros);
    lower_values_.resize(nonzeros);
    std::vector<Index> next(lower_starts_.begin(), lower_starts_.end() - 1);
    // 原上三角列递增遍历，使转置后每列的行索引天然严格递增。
    for (std::size_t col = 0; col < size; ++col) {
      check(budget_check);
      for (Index source = starts[col]; source < starts[col + 1]; ++source) {
        if ((source & 4095) == 0) check(budget_check);
        const auto row = static_cast<std::size_t>(rows[source]);
        const Index destination = next[row];
        if (destination < lower_starts_[row] || destination >= lower_starts_[row + 1]) {
          throw std::runtime_error("CHOLMOD 下三角槽映射越界");
        }
        ++next[row];
        lower_rows_[static_cast<std::size_t>(destination)] = static_cast<Index>(col);
        lower_sources_[static_cast<std::size_t>(destination)] = source;
      }
    }
    for (std::size_t col = 0; col < size; ++col) {
      if ((col & 4095) == 0) check(budget_check);
      if (next[col] != lower_starts_[col + 1]) throw std::runtime_error("CHOLMOD 下三角槽映射不完整");
    }
    lower_matrix_ = matrix_;
    lower_matrix_.p = lower_starts_.data();
    lower_matrix_.i = lower_rows_.data();
    lower_matrix_.x = lower_values_.data();
    lower_matrix_.stype = -1;
    lower_matrix_.xtype = CHOLMOD_REAL;
    check(budget_check);
  }

  void check_status() const {
    if (resources_.common.status == CHOLMOD_OUT_OF_MEMORY) throw std::bad_alloc();
    if (resources_.common.status < CHOLMOD_OK) {
      // 库的尺寸错误、输入错误或分配失败不等同于本项目显式结构预算。
      throw std::runtime_error("CHOLMOD 内部错误，状态=" + std::to_string(resources_.common.status));
    }
  }

  bool finite_factor(const std::function<void()>& budget_check) const {
    const auto& factor = *resources_.factor;
    if (!factor.is_super || !factor.is_ll || factor.itype != CHOLMOD_LONG ||
        factor.xtype != CHOLMOD_REAL || factor.dtype != CHOLMOD_DOUBLE || !factor.x) return false;
    const auto* super = static_cast<const Index*>(factor.super);
    const auto* pointers = static_cast<const Index*>(factor.pi);
    const auto* offsets = static_cast<const Index*>(factor.px);
    const auto* values = static_cast<const double*>(factor.x);
    for (std::size_t node = 0; node < factor.nsuper; ++node) {
      check(budget_check);
      const Index columns = super[node + 1] - super[node];
      const Index rows = pointers[node + 1] - pointers[node];
      const Index offset = offsets[node];
      if (columns <= 0 || rows < columns || offset < 0 ||
          static_cast<std::size_t>(offset) > factor.xsize ||
          static_cast<std::size_t>(columns) > (factor.xsize - static_cast<std::size_t>(offset)) /
              static_cast<std::size_t>(rows)) return false;
      for (Index col = 0; col < columns; ++col) {
        check(budget_check);
        const Index begin = offset + col * rows;
        if (!(values[begin + col] > 0)) return false;
        // 超节点按列存密集块；未使用的上三角占位不参与数值合法性检查。
        for (Index row = col; row < rows; ++row) {
          if ((row & 4095) == 0) check(budget_check);
          if (!std::isfinite(values[begin + row])) return false;
        }
      }
    }
    return true;
  }

 public:
  CholmodFactor(const Matrix& matrix, const std::function<void()>& budget_check, bool rectangular = false) {
    matrix_.nrow = static_cast<std::size_t>(matrix.rows());
    matrix_.ncol = static_cast<std::size_t>(matrix.cols());
    matrix_.nzmax = static_cast<std::size_t>(matrix.nonZeros());
    matrix_.p = const_cast<Index*>(matrix.outerIndexPtr());
    matrix_.i = const_cast<Index*>(matrix.innerIndexPtr());
    matrix_.stype = rectangular ? 0 : 1;
    matrix_.itype = CHOLMOD_LONG;
    matrix_.xtype = CHOLMOD_PATTERN;
    matrix_.dtype = CHOLMOD_DOUBLE;
    matrix_.sorted = matrix_.packed = 1;
    check(budget_check);
    resources_.factor = cholmod_l_analyze(&matrix_, &resources_.common);
    check(budget_check);
    check_status();
    const auto* factor = resources_.factor;
    if (!factor || resources_.common.status != CHOLMOD_OK || !factor->is_super ||
        factor->itype != CHOLMOD_LONG || factor->n != matrix_.nrow || !factor->Perm) {
      throw std::runtime_error("CHOLMOD 超节点符号分析失败");
    }
    // xsize 包含超节点中的占位；在首次分配数值因子之前检查实际存储预算。
    if (factor->xsize > static_cast<std::size_t>(structural_budget)) {
      throw NormalCholeskyBudgetExceeded("CHOLMOD 超节点存储超过 16777216 项预算");
    }
    const auto* permutation = static_cast<const Index*>(factor->Perm);
    for (std::size_t col = 0; col < matrix_.nrow; ++col) {
      if ((col & 4095) == 0) check(budget_check);
      if (permutation[col] != static_cast<Index>(col)) {
        throw std::runtime_error("CHOLMOD 改变了已固定的 AMD 排列");
      }
    }
    check(budget_check);
    // 对称输入缓存原上三角的转置；矩形输入按自己的模式分析，由上游形成乘积。
    if (!rectangular) cache_lower_pattern(budget_check);
  }

  bool factor(std::span<const double> values, const std::function<void()>& budget_check) {
    valid_ = false;
    check(budget_check);
    cholmod_sparse* input = &lower_matrix_;
    if (matrix_.stype == 0) {
      if (values.size() != matrix_.nzmax) return false;
      // 数值由外层每次重建；模式和行排列与本对象实际分析的矩形输入相同。
      matrix_.x = const_cast<double*>(values.data());
      matrix_.xtype = CHOLMOD_REAL;
      input = &matrix_;
    } else {
      if (values.size() != lower_values_.size()) return false;
      for (std::size_t destination = 0; destination < lower_values_.size(); ++destination) {
        if ((destination & 4095) == 0) check(budget_check);
        // 仅复制本次已舍入的 double，不缓存跨次分解的矩阵数值。
        lower_values_[destination] = values[static_cast<std::size_t>(lower_sources_[destination])];
      }
    }
    check(budget_check);
    const int success = cholmod_l_factorize(input, resources_.factor, &resources_.common);
    check(budget_check);
    check_status();
    // 非正定时上游可能返回真；状态、minor 和存储数值都必须独立通过。
    if (!success || resources_.common.status != CHOLMOD_OK ||
        resources_.factor->minor != resources_.factor->n || !finite_factor(budget_check)) return false;
    check(budget_check);
    valid_ = true;
    return true;
  }

  bool solve(Eigen::VectorXd& rhs, const std::function<void()>& budget_check) const {
    if (!valid_) return false;
    cholmod_dense input{};
    input.nrow = input.d = input.nzmax = matrix_.nrow;
    input.ncol = 1;
    input.x = rhs.data();
    input.xtype = CHOLMOD_REAL;
    input.dtype = CHOLMOD_DOUBLE;
    check(budget_check);
    const int success = cholmod_l_solve2(CHOLMOD_A, resources_.factor, &input, nullptr,
        &resources_.solution, nullptr, &resources_.work, &resources_.extra, &resources_.common);
    check(budget_check);
    check_status();
    if (!success || resources_.common.status != CHOLMOD_OK || !resources_.solution) return false;
    const auto* values = static_cast<const double*>(resources_.solution->x);
    for (std::size_t row = 0; row < matrix_.nrow; ++row) {
      if ((row & 4095) == 0) check(budget_check);
      if (!std::isfinite(values[row])) return false;
    }
    check(budget_check);
    std::copy(values, values + matrix_.nrow, rhs.data());
    return true;
  }

  Index stored_nonzeros() const { return static_cast<Index>(resources_.factor->xsize); }
};

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
  Matrix rectangular_matrix;
  std::vector<Index> rectangular_sources;
  CheckedCholesky<double> cholesky;
  ExtendedMatrix extended_matrix;
  CheckedCholesky<long double> extended_cholesky;
  bool extended;
  bool rectangular;
  int max_refinement_steps;
  mutable NormalRefinementStatistics refinement_statistics;
  std::unique_ptr<MultifrontalLdl> multifrontal;
  std::unique_ptr<CholmodFactor> cholmod;
  std::vector<double> factor_scaling;
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

  bool refined_solve(ExtendedVector& rhs, const std::function<void()>& budget_check) const {
    const ExtendedVector original = rhs;
    Eigen::VectorXd rounded(rhs.size());
    for (Eigen::Index row = 0; row < rhs.size(); ++row) {
      if (!representable(rhs[row], rounded[row])) return false;
    }
    check(budget_check);
    const Eigen::VectorXd initial = cholesky.solve(rounded);
    check(budget_check);
    if (!initial.allFinite()) return false;
    rhs = initial.cast<long double>();
    const auto residual = [&](const ExtendedVector& point, ExtendedVector& output) {
      check(budget_check);
      output = original;
      ++refinement_statistics.residual_products;
      // 使用未舍入的既有 Gram 累加结果；结构和向量均在同一 AMD 坐标中。
      for (Index col = 0; col < matrix.cols(); ++col) {
        check(budget_check);
        for (Index entry = matrix.outerIndexPtr()[col]; entry < matrix.outerIndexPtr()[col + 1]; ++entry) {
          const Index row = matrix.innerIndexPtr()[entry];
          const long double value = sums[static_cast<std::size_t>(entry)];
          output[row] -= value * point[col];
          if (row != col) output[col] -= value * point[row];
        }
      }
      long double norm = 0;
      for (Eigen::Index row = 0; row < output.size(); ++row) {
        if (!std::isfinite(output[row])) return std::numeric_limits<long double>::infinity();
        norm = std::max(norm, std::abs(output[row]));
      }
      check(budget_check);
      return norm;
    };
    ExtendedVector defect;
    long double norm = residual(rhs, defect);
    for (int step = 0; step < max_refinement_steps && norm > 0 && std::isfinite(norm); ++step) {
      bool available = true;
      for (Eigen::Index row = 0; row < rhs.size(); ++row) {
        if (!representable(defect[row], rounded[row])) { available = false; break; }
      }
      if (!available) break;
      check(budget_check);
      ++refinement_statistics.refinement_steps;
      const Eigen::VectorXd correction = cholesky.solve(rounded);
      check(budget_check);
      if (!correction.allFinite()) break;
      const ExtendedVector candidate = rhs + correction.cast<long double>();
      if (!candidate.allFinite()) break;
      ExtendedVector next_defect;
      const long double next_norm = residual(candidate, next_defect);
      // 有界搜索只保留实际 Schur 残差严格下降的候选；最终仍由外层验完整 KKT。
      if (!(next_norm < norm)) break;
      rhs = candidate;
      defect = std::move(next_defect);
      norm = next_norm;
      ++refinement_statistics.refinement_acceptances;
    }
    return true;
  }

  Impl(std::size_t size, std::span<const Index> starts, std::span<const Index> rows,
       std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool use_multifrontal,
       bool use_extended, int refinement_budget, NormalCholeskyKernel kernel)
      : dimension(size), input_nonzeros(rows.size()), extended(use_extended),
        rectangular(kernel == NormalCholeskyKernel::cholmod_supernodal_aat), max_refinement_steps(refinement_budget) {
    check(budget_check);
    if (kernel != NormalCholeskyKernel::eigen && kernel != NormalCholeskyKernel::cholmod_supernodal && !rectangular) {
      throw std::invalid_argument("未知的正规方程 Cholesky 内核");
    }
    if ((kernel == NormalCholeskyKernel::cholmod_supernodal || rectangular) &&
        (use_multifrontal || extended || max_refinement_steps != 0)) {
      throw std::invalid_argument("CHOLMOD 与分块、扩展精度及 Schur 校正选项互斥");
    }
    if (use_multifrontal && extended) throw std::invalid_argument("分块与扩展精度正规方程选项不能同时开启");
    if (max_refinement_steps < 0 || max_refinement_steps > 2 ||
        (max_refinement_steps > 0 && (use_multifrontal || extended))) {
      throw std::invalid_argument("Schur 校正预算仅支持 0 至 2，且只能配合 double 标量分解");
    }
    if ((extended || max_refinement_steps > 0) &&
        std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
      throw std::invalid_argument("本平台 long double 未提供比 double 更高的有效精度");
    }
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
    if (rectangular) {
      // 只保留一次 Gram 结构图以复用完全相同的 AMD 及因子预算；不缓存数值展开槽。
      const auto rectangular_columns = primal_coordinates.size() + dual_coordinates.size();
      std::size_t rectangular_nonzeros = dual_coordinates.size();
      for (const auto& column : columns) {
        check(budget_check);
        if (column.size() > static_cast<std::size_t>(structural_budget) - rectangular_nonzeros) {
          throw NormalCholeskyBudgetExceeded("CHOLMOD 矩形输入超过 16777216 个结构项预算");
        }
        rectangular_nonzeros += column.size();
      }
      if (rectangular_columns > static_cast<std::size_t>(structural_budget) ||
          rectangular_columns >= static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
          rectangular_nonzeros > rectangular_sources.max_size()) {
        throw std::length_error("CHOLMOD 矩形输入尺寸无法表示");
      }
      rectangular_matrix.resize(retained, static_cast<Index>(rectangular_columns));
      rectangular_matrix.reserve(static_cast<Index>(rectangular_nonzeros));
      rectangular_sources.reserve(rectangular_nonzeros);
      for (std::size_t col = 0; col < columns.size(); ++col) {
        check(budget_check);
        auto ordered = columns[col];
        std::sort(ordered.begin(), ordered.end(), [&](const Entry& a, const Entry& b) {
          return inverse[static_cast<std::size_t>(a.row)] < inverse[static_cast<std::size_t>(b.row)];
        });
        rectangular_matrix.startVec(static_cast<Index>(col));
        for (std::size_t entry = 0; entry < ordered.size(); ++entry) {
          if ((entry & 4095) == 0) check(budget_check);
          rectangular_matrix.insertBackByOuterInner(static_cast<Index>(col),
              inverse[static_cast<std::size_t>(ordered[entry].row)]) = 1;
          rectangular_sources.push_back(ordered[entry].source);
        }
      }
      for (std::size_t row = 0; row < dual_coordinates.size(); ++row) {
        if ((row & 4095) == 0) check(budget_check);
        const auto col = static_cast<Index>(columns.size() + row);
        rectangular_matrix.startVec(col);
        rectangular_matrix.insertBackByOuterInner(col, inverse[row]) = 1;
        rectangular_sources.push_back(dual_diagonal[row]);
      }
      rectangular_matrix.finalize();
      check(budget_check);
      cholmod = std::make_unique<CholmodFactor>(rectangular_matrix, budget_check, true);
      // 数值分解与回代只需要矩形模式和原 KKT；释放仅用于固定排序的 Gram 图。
      matrix = Matrix{};
      check(budget_check);
      return;
    } else if (kernel == NormalCholeskyKernel::cholmod_supernodal) {
      cholmod = std::make_unique<CholmodFactor>(matrix, budget_check);
    } else if (use_multifrontal) {
      // 同一 AMD 后的正定 Schur 矩阵；复用已有分块 LDL，坐标符号全部为正。
      // 分块内核可能正则小主元，外层仍必须检查完整未正则化 KKT。
      multifrontal = std::make_unique<MultifrontalLdl>(static_cast<std::size_t>(retained),
          std::span<const Index>(matrix.outerIndexPtr(), static_cast<std::size_t>(retained + 1)),
          std::span<const Index>(matrix.innerIndexPtr(), static_cast<std::size_t>(matrix.nonZeros())),
          std::vector<std::int8_t>(static_cast<std::size_t>(retained), 1), budget_check);
    } else if (extended) {
      extended_matrix = matrix.cast<long double>();
      extended_cholesky.analyzePattern(extended_matrix);
      check(budget_check);
      if (extended_cholesky.info() != Eigen::Success ||
          extended_cholesky.stored_nonzeros() != lower_nonzeros + retained) {
        throw std::runtime_error("扩展正规方程 Cholesky 符号结构与预检不一致");
      }
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
    std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool multifrontal, bool extended,
    int max_refinement_steps, NormalCholeskyKernel kernel)
    : impl_(std::make_unique<Impl>(size, starts, rows, signs, budget_check, multifrontal, extended, max_refinement_steps, kernel)) {}

NormalCholesky::~NormalCholesky() = default;

bool NormalCholesky::factor(std::span<const double> values, const std::function<void()>& budget_check) {
  auto& state = *impl_;
  state.valid = false;
  state.factor_nnz = 0;
  state.modified_pivots = 0;
  check(budget_check);
  if (values.size() != state.input_nonzeros || !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); })) return false;
  for (Index source : state.primal_diagonal) if (!(values[static_cast<std::size_t>(source)] > 0)) return false;
  for (Index source : state.dual_diagonal) if (!(values[static_cast<std::size_t>(source)] < 0)) return false;
  state.input_values.assign(values.begin(), values.end());
  if (state.rectangular) {
    if (!state.dual_coordinates.empty()) {
      // C=[G D^(-1/2),sqrt(R)]，R 是逐行正对角，不能替换为统一的标量偏移。
      for (std::size_t col = 0; col < state.columns.size(); ++col) {
        check(budget_check);
        const long double divisor = std::sqrt(static_cast<long double>(
            values[static_cast<std::size_t>(state.primal_diagonal[col])]));
        if (!(divisor > 0) || !std::isfinite(divisor)) return false;
        for (Index entry = state.rectangular_matrix.outerIndexPtr()[col];
             entry < state.rectangular_matrix.outerIndexPtr()[col + 1]; ++entry) {
          if ((entry & 4095) == 0) check(budget_check);
          const double coefficient = values[static_cast<std::size_t>(state.rectangular_sources[static_cast<std::size_t>(entry)])];
          const long double scaled = static_cast<long double>(coefficient) / divisor;
          if ((coefficient != 0 && scaled == 0) ||
              !representable(scaled, state.rectangular_matrix.valuePtr()[entry])) return false;
        }
      }
      for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
        if ((row & 4095) == 0) check(budget_check);
        const auto col = state.columns.size() + row;
        const auto entry = state.rectangular_matrix.outerIndexPtr()[col];
        const long double diagonal = -static_cast<long double>(
            values[static_cast<std::size_t>(state.dual_diagonal[row])]);
        if (!representable(std::sqrt(diagonal), state.rectangular_matrix.valuePtr()[entry]) ||
            !(state.rectangular_matrix.valuePtr()[entry] > 0)) return false;
      }
      check(budget_check);
      if (!state.cholmod->factor(std::span<const double>(state.rectangular_matrix.valuePtr(),
          static_cast<std::size_t>(state.rectangular_matrix.nonZeros())), budget_check)) return false;
      state.factor_nnz = state.cholmod->stored_nonzeros();
    }
    check(budget_check);
    state.valid = true;
    return true;
  }
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
      // 仅在宽指数平台复用商，避免窄精度平台的中间溢出或提前下溢。
      constexpr bool wide_range =
          std::numeric_limits<long double>::max_exponent >= 4 * std::numeric_limits<double>::max_exponent &&
          std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits <=
              4 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits);
      long double scaled_b = 0;
      if constexpr (wide_range) scaled_b = static_cast<long double>(b) / diagonal;
      for (std::size_t left = 0; left <= right; ++left) {
        const double a = values[static_cast<std::size_t>(column[left].source)];
        long double contribution;
        if constexpr (wide_range) contribution = static_cast<long double>(a) * scaled_b;
        else contribution = static_cast<long double>(a) * b / diagonal;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(contribution) || (a != 0 && b != 0 && contribution == 0)) return false;
        }
        auto& sum = state.sums[static_cast<std::size_t>(state.product_slots[product++])];
        sum += contribution;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(sum)) return false;
        }
      }
    }
  }
  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {
    if (state.extended) {
      // 保留已付出成本的扩展精度装配结果，不在分解前舍入回 double。
      state.extended_matrix.valuePtr()[entry] = state.sums[entry];
    } else if (!representable(state.sums[entry], state.matrix.valuePtr()[entry])) return false;
  }
  check(budget_check);
  if (!state.dual_coordinates.empty()) {
    if (state.cholmod) {
      if (!state.cholmod->factor(std::span<const double>(state.matrix.valuePtr(), state.sums.size()), budget_check)) return false;
      state.factor_nnz = state.cholmod->stored_nonzeros();
    } else if (state.multifrontal) {
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
    } else if (state.extended) {
      state.extended_cholesky.factorize(state.extended_matrix);
      check(budget_check);
      if (state.extended_cholesky.info() != Eigen::Success ||
          !state.extended_cholesky.finite_factor(budget_check)) return false;
      state.factor_nnz = state.extended_cholesky.stored_nonzeros();
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
  if (state.extended || state.max_refinement_steps > 0) {
    Eigen::Matrix<long double, Eigen::Dynamic, 1> permuted(static_cast<Eigen::Index>(retained));
    for (std::size_t row = 0; row < retained; ++row) {
      if (!std::isfinite(reduced_rhs[row])) return false;
      permuted[state.inverse[row]] = reduced_rhs[row];
    }
    if (retained != 0) {
      check(budget_check);
      if (state.extended) {
        const ExtendedVector solved = state.extended_cholesky.solve(permuted);
        check(budget_check);
        if (!solved.allFinite()) return false;
        permuted = solved;
      } else if (!state.refined_solve(permuted, budget_check)) return false;
    }
    std::vector<double> solution(rhs.size());
    for (std::size_t row = 0; row < retained; ++row) {
      if (!representable(permuted[state.inverse[row]], solution[static_cast<std::size_t>(state.dual_coordinates[row])])) return false;
    }
    for (std::size_t col = 0; col < state.columns.size(); ++col) {
      check(budget_check);
      const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
      long double value = rhs[coordinate];
      // 恢复也使用未舍入的 Schur 解；仅完整输出转换回外层 double。
      for (const auto& entry : state.columns[col]) {
        value -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) *
            permuted[state.inverse[static_cast<std::size_t>(entry.row)]];
      }
      value /= state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
      if (!representable(value, solution[coordinate])) return false;
    }
    check(budget_check);
    rhs = std::move(solution);
    return true;
  }
  Eigen::VectorXd permuted(static_cast<Eigen::Index>(retained));
  for (std::size_t row = 0; row < retained; ++row) {
    const long double value = state.multifrontal ? reduced_rhs[row] *
        state.factor_scaling[static_cast<std::size_t>(state.inverse[row])] : reduced_rhs[row];
    if (!representable(value, permuted[state.inverse[row]])) return false;
  }
  if (retained != 0) {
    check(budget_check);
    if (state.cholmod) {
      if (!state.cholmod->solve(permuted, budget_check)) return false;
    } else if (state.multifrontal) {
      std::vector<double> solution(permuted.data(), permuted.data() + permuted.size());
      if (!state.multifrontal->solve(solution, budget_check)) return false;
      for (std::size_t row = 0; row < retained; ++row) {
        if (!representable(static_cast<long double>(solution[row]) * state.factor_scaling[row],
            permuted[static_cast<Eigen::Index>(row)])) return false;
      }
    } else {
      const Eigen::VectorXd solution = state.cholesky.solve(permuted);
      check(budget_check);
      if (!solution.allFinite()) return false;
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
  rhs = std::move(solution);
  return true;
}

Index NormalCholesky::factor_nonzeros() const { return impl_->factor_nnz; }
Index NormalCholesky::factor_dimension() const { return static_cast<Index>(impl_->dual_coordinates.size()); }
Index NormalCholesky::regularized_pivots() const {
  return impl_->modified_pivots;
}

NormalRefinementStatistics NormalCholesky::refinement_statistics() const {
  return impl_->refinement_statistics;
}

}  // 命名空间 solver::detail
