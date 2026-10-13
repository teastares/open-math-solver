#include "solver/lp_optimality.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace solver {
namespace {

bool valid_tolerance(double value) { return std::isfinite(value) && value >= 0.0; }

bool add_product(long double& sum, long double left, long double right) {
  const auto product = left * right;
  if (!std::isfinite(product)) return false;
  sum += product;
  return std::isfinite(sum);
}

bool representable(long double value) {
  return std::isfinite(value) && std::abs(value) <= std::numeric_limits<double>::max();
}

// 对零乘子显式返回零，不计算零乘无穷；替代无限项只能生成估计目标。
bool add_bound_contribution(long double multiplier, double lower, double upper,
                            long double& estimate, bool& finite_direction) {
  if (multiplier == 0.0L) return true;
  const double selected = multiplier > 0.0L ? lower : upper;
  if (std::isfinite(selected)) {
    return add_product(estimate, multiplier, selected);
  }
  finite_direction = false;
  const double alternative = multiplier > 0.0L ? upper : lower;
  return !std::isfinite(alternative) || add_product(estimate, multiplier, alternative);
}

void arithmetic_failure(LpOptimality& result) {
  result.dual_feasible = false;
  result.gap_satisfied = false;
  result.numerically_optimal = false;
  result.dual_objective_estimate.reset();
  result.corrected_dual_bound.reset();
  result.objective_gap.reset();
  result.relative_objective_gap.reset();
  result.errors.emplace_back("计算 LP 对偶残差、目标或间隙时发生溢出或非有限结果");
}

}  // 匿名命名空间

bool project_lp_row_dual_domain(const Model& model, std::span<double> row_dual) {
  if (model.row_lower.size() != model.row_upper.size() ||
      row_dual.size() != model.row_lower.size()) return false;
  // 先完整检查行界，避免后面的错误使输入向量只被修改一部分。
  const double infinity = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < row_dual.size(); ++row) {
    const double lower = model.row_lower[row];
    const double upper = model.row_upper[row];
    if (std::isnan(lower) || std::isnan(upper) || lower == infinity ||
        upper == -infinity || lower > upper) return false;
  }
  for (std::size_t row = 0; row < row_dual.size(); ++row) {
    auto& multiplier = row_dual[row];
    if (!std::isfinite(multiplier)) continue;
    if ((!std::isfinite(model.row_lower[row]) && multiplier > 0.0) ||
        (!std::isfinite(model.row_upper[row]) && multiplier < 0.0)) {
      multiplier = 0.0;
    }
  }
  return true;
}

LpVerification verify_lp_solution(const Model& model,
                                  std::span<const double> primal,
                                  std::span<const double> row_dual,
                                  const Tolerances& primal_tolerances,
                                  const LpOptimalityOptions& options) {
  LpVerification combined;
  combined.primal = verify_solution(model, primal, std::nullopt, primal_tolerances);
  const auto& primal_check = combined.primal;
  auto& result = combined.optimality;
  result.errors = primal_check.errors;
  bool inputs_ok = primal_check.valid_input;
  if (is_mip(model)) {
    inputs_ok = false;
    result.errors.emplace_back("LP 最优性检查不接受包含整数变量的模型");
  }
  if (row_dual.size() != model.row_lower.size()) {
    inputs_ok = false;
    result.errors.emplace_back("行对偶向量长度与约束行数不一致");
  }
  if (!valid_tolerance(options.dual_absolute) || !valid_tolerance(options.dual_relative) ||
      !valid_tolerance(options.gap_absolute) || !valid_tolerance(options.gap_relative)) {
    inputs_ok = false;
    result.errors.emplace_back("LP 对偶与间隙容差必须为非负有限数");
  }
  for (std::size_t row = 0; row < row_dual.size(); ++row) {
    if (!std::isfinite(row_dual[row])) {
      inputs_ok = false;
      result.errors.emplace_back("行对偶乘子必须为有限数：" + std::to_string(row));
    }
  }
  result.valid_input = inputs_ok;
  if (!inputs_ok || !primal_check.objective) return combined;

  const long double sense = model.sense == ObjectiveSense::minimize ? 1.0L : -1.0L;
  long double primal_contribution = 0.0L;
  long double dual_contribution = 0.0L;
  bool all_directions_finite = true;
  bool dual_ok = true;
  for (std::size_t row = 0; row < row_dual.size(); ++row) {
    bool finite_direction = true;
    if (!add_bound_contribution(row_dual[row], model.row_lower[row], model.row_upper[row],
                               dual_contribution, finite_direction)) {
      arithmetic_failure(result);
      return combined;
    }
    if (!finite_direction) {
      // 行乘子的符号约束是严格定义域，不能用数值容差放行。
      dual_ok = false;
      all_directions_finite = false;
      result.errors.emplace_back("行对偶乘子指向无穷行界：" + std::to_string(row));
    }
  }

  for (std::size_t column = 0; column < primal.size(); ++column) {
    const long double cost = sense * static_cast<long double>(model.costs[column]);
    if (!add_product(primal_contribution, cost, primal[column])) {
      arithmetic_failure(result);
      return combined;
    }
    long double transpose_product = 0.0L;
    const auto begin = static_cast<std::size_t>(model.column_start[column]);
    const auto end = static_cast<std::size_t>(model.column_start[column + 1]);
    for (auto position = begin; position < end; ++position) {
      const auto row = static_cast<std::size_t>(model.row_index[position]);
      if (!add_product(transpose_product, model.values[position], row_dual[row])) {
        arithmetic_failure(result);
        return combined;
      }
    }
    const auto reduced_cost = cost - transpose_product;
    if (!std::isfinite(reduced_cost)) {
      arithmetic_failure(result);
      return combined;
    }
    bool finite_direction = true;
    if (!add_bound_contribution(reduced_cost, model.column_lower[column],
                               model.column_upper[column], dual_contribution,
                               finite_direction)) {
      arithmetic_failure(result);
      return combined;
    }
    if (!finite_direction) {
      all_directions_finite = false;
      const auto residual = std::abs(reduced_cost);
      const auto scale = std::max(std::abs(cost), std::abs(transpose_product));
      const auto allowed = static_cast<long double>(options.dual_absolute) +
                           static_cast<long double>(options.dual_relative) * scale;
      if (!representable(residual) || !std::isfinite(allowed)) {
        arithmetic_failure(result);
        return combined;
      }
      result.max_dual_residual = std::max(result.max_dual_residual,
                                         static_cast<double>(residual));
      if (residual > allowed) {
        dual_ok = false;
        result.errors.emplace_back("指向无穷变量界的对偶残差超出容差：" +
                                   std::to_string(column));
      }
    }
  }

  // 常数项同时出现在原始与对偶目标中；间隙直接比较去常数贡献。
  const auto gap = std::abs(primal_contribution - dual_contribution);
  const auto scale = std::max(std::abs(primal_contribution), std::abs(dual_contribution));
  const auto allowed_gap = static_cast<long double>(options.gap_absolute) +
                           static_cast<long double>(options.gap_relative) * scale;
  const auto relative_gap = gap / std::max(1.0L, scale);
  const auto dual_objective = static_cast<long double>(model.objective_offset) +
                              sense * dual_contribution;
  if (!representable(gap) || !representable(relative_gap) || !representable(dual_objective) ||
      !std::isfinite(allowed_gap)) {
    arithmetic_failure(result);
    return combined;
  }

  result.dual_objective_estimate = static_cast<double>(dual_objective);
  if (all_directions_finite) {
    // 该界仍按浮点算术计算，没有区间舍入或精确算术的认证保证。
    result.corrected_dual_bound = static_cast<double>(dual_objective);
  }
  result.objective_gap = static_cast<double>(gap);
  result.relative_objective_gap = static_cast<double>(relative_gap);
  result.dual_feasible = dual_ok;
  result.gap_satisfied = gap <= allowed_gap;
  if (!result.gap_satisfied) {
    result.errors.emplace_back("去除目标常数后的原始与对偶目标间隙超出容差");
  }
  result.numerically_optimal = primal_check.feasible && primal_check.objective_consistent &&
                               result.dual_feasible && result.gap_satisfied;
  return combined;
}

LpOptimality verify_lp_optimality(const Model& model,
                                  std::span<const double> primal,
                                  std::span<const double> row_dual,
                                  const Tolerances& primal_tolerances,
                                  const LpOptimalityOptions& options) {
  return verify_lp_solution(model, primal, row_dual, primal_tolerances, options).optimality;
}

}  // 命名空间 solver
