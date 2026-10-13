#include "solver/detail/multifrontal_ldl.hpp"

#include "ipm/hipo/factorhighs/FactorHiGHS.h"
#include <qdldl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace solver::detail {
namespace {
void check(const std::function<void()>& budget_check) {
  if (budget_check) budget_check();
}
constexpr std::int64_t structure_limit = 16'777'216;
constexpr double factor_storage_limit = 512.0 * 1024 * 1024;
bool finite_values(std::span<const double> values) {
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}
}  // 匿名命名空间

struct MultifrontalLdl::Impl {
  std::size_t dimension;
  std::vector<hipo::Int> starts, rows, signs;
  std::vector<std::size_t> source, diagonal;
  std::vector<double> values;
  hipo::Symbolic symbolic;
  std::unique_ptr<hipo::FHsolver> factorization = std::make_unique<hipo::FHsolver>();
  bool valid = false;
  bool needs_reset = false;
  std::int64_t modifications = 0;

  Impl(std::size_t n, std::span<const std::int64_t> ap, std::span<const std::int64_t> ai,
       std::span<const std::int8_t> coordinate_signs, const std::function<void()>& budget_check)
      : dimension(n) {
    check(budget_check);
    if (n >= static_cast<std::size_t>(std::numeric_limits<hipo::Int>::max()) || ap.size() != n + 1 ||
        coordinate_signs.size() != n || ap.front() != 0 ||
        ai.size() >= static_cast<std::size_t>(std::numeric_limits<hipo::Int>::max()) ||
        ap.back() != static_cast<std::int64_t>(ai.size())) {
      throw std::invalid_argument("分块 LDL 上三角 CSC 维数或列指针无效");
    }
    if (n > static_cast<std::size_t>(structure_limit) || ai.size() > static_cast<std::size_t>(structure_limit))
      throw std::length_error("分块 LDL 输入结构超过 16777216 项预算");
    starts.assign(n + 1, 0);
    diagonal.resize(n);
    signs.assign(coordinate_signs.begin(), coordinate_signs.end());
    for (std::size_t col = 0; col < n; ++col) {
      check(budget_check);
      if ((signs[col] != 1 && signs[col] != -1) || ap[col] < 0 || ap[col] >= ap[col + 1] ||
          ap[col + 1] > static_cast<std::int64_t>(ai.size()))
        throw std::invalid_argument("分块 LDL 列指针或坐标符号无效");
      std::int64_t previous = -1;
      for (auto k = ap[col]; k < ap[col + 1]; ++k) {
        const auto row = ai[static_cast<std::size_t>(k)];
        if (row <= previous || row < 0 || row > static_cast<std::int64_t>(col))
          throw std::invalid_argument("分块 LDL 必须使用严格递增、无重复的上三角行索引");
        previous = row;
        ++starts[static_cast<std::size_t>(row) + 1];
      }
      if (previous != static_cast<std::int64_t>(col)) throw std::invalid_argument("分块 LDL 缺少对角");
      diagonal[col] = static_cast<std::size_t>(ap[col + 1] - 1);
    }
    std::partial_sum(starts.begin(), starts.end(), starts.begin());
    rows.resize(ai.size()); source.resize(ai.size()); values.resize(ai.size());
    auto next = starts;
    for (std::size_t col = 0; col < n; ++col) {
      check(budget_check);
      for (auto k = ap[col]; k < ap[col + 1]; ++k) {
        const auto slot = static_cast<std::size_t>(next[static_cast<std::size_t>(ai[static_cast<std::size_t>(k)])]++);
        rows[slot] = static_cast<hipo::Int>(col);
        source[slot] = static_cast<std::size_t>(k);
      }
    }
    if (n == 0) return;
    // 在超节点分析分配前，以同一结构的标量 LDL 填充计数约束增长。
    std::vector<QDLDL_int> qap(ap.begin(), ap.end()), qai(ai.begin(), ai.end()), work(n), counts(n), tree(n);
    const auto nnz = QDLDL_etree(static_cast<QDLDL_int>(n), qap.data(), qai.data(), work.data(), counts.data(), tree.data());
    check(budget_check);
    if (nnz < 0) throw std::invalid_argument("分块 LDL 结构预检查失败");
    if (nnz > structure_limit - static_cast<std::int64_t>(n))
      throw std::length_error("分块 LDL 预测填充超过 16777216 项预算");
    std::vector<hipo::Int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), hipo::Int{0});
    symbolic.setParallel(false, false);
    if (factorization->analyse(symbolic, rows, starts, signs, permutation) != 0)
      throw std::runtime_error("分块 LDL 符号分析失败");
    check(budget_check);
    if (symbolic.nz() < 0 || symbolic.nz() > structure_limit || !std::isfinite(symbolic.storage()) ||
        symbolic.storage() > factor_storage_limit)
      throw std::length_error("分块 LDL 超节点填充或工作区超过预算");
  }
};

MultifrontalLdl::MultifrontalLdl(std::size_t dimension, std::span<const std::int64_t> starts,
    std::span<const std::int64_t> rows, std::span<const std::int8_t> signs,
    const std::function<void()>& budget_check)
    : impl_(std::make_unique<Impl>(dimension, starts, rows, signs, budget_check)) {}
MultifrontalLdl::~MultifrontalLdl() = default;

bool MultifrontalLdl::factor(std::span<const double> values, const std::function<void()>& budget_check) {
  auto& state = *impl_;
  state.valid = false;
  state.modifications = 0;
  check(budget_check);
  if (values.size() != state.values.size() || !finite_values(values)) return false;
  for (std::size_t col = 0; col < state.dimension; ++col)
    if (!(state.signs[col] * values[state.diagonal[col]] > 0)) return false;
  if (state.dimension == 0) { state.valid = true; return true; }
  // 上游中途失败可能留下未完成的 clique 栈；重试保留符号结构，重建数值工作区。
  // 预算或分配异常同样保持此标记，不能将半完成状态用于下一次分解。
  if (state.needs_reset) state.factorization = std::make_unique<hipo::FHsolver>();
  for (std::size_t k = 0; k < values.size(); ++k) state.values[k] = values[state.source[k]];
  state.needs_reset = true;
  const auto status = state.factorization->factorise(state.symbolic, state.rows, state.starts, state.values);
  check(budget_check);
  if (status != 0) return false;
  // 上游可能修改小主元；只读取一次，原方程精度仍由自主 IPM 在外部核对。
  std::vector<double> regularization;
  state.factorization->getRegularisation(regularization);
  if (regularization.size() != state.dimension || !finite_values(regularization)) return false;
  state.modifications = std::count_if(regularization.begin(), regularization.end(), [](double value) { return value != 0; });
  state.needs_reset = false;
  state.valid = true;
  return true;
}

bool MultifrontalLdl::solve(std::vector<double>& rhs, const std::function<void()>& budget_check) const {
  auto& state = *impl_;
  check(budget_check);
  if (!state.valid || rhs.size() != state.dimension || !finite_values(rhs)) return false;
  if (state.dimension == 0) return true;
  auto solution = rhs;
  const auto status = state.factorization->solve(solution);
  check(budget_check);
  if (status != 0 || !finite_values(solution)) return false;
  rhs = std::move(solution);
  return true;
}

std::int64_t MultifrontalLdl::factor_nonzeros() const {
  return impl_->valid ? impl_->symbolic.nz() : 0;
}
std::int64_t MultifrontalLdl::regularized_pivots() const { return impl_->modifications; }

}  // 命名空间 solver::detail
