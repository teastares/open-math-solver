#include "solver/verification.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace solver {
namespace {

bool valid_tolerance(double value) { return std::isfinite(value) && value >= 0.0; }

// 平台上的 long double 可能与 double 等宽，仍逐步检测非有限结果。
bool add_product(long double& sum, double coefficient, double value) {
  const auto product = static_cast<long double>(coefficient) * static_cast<long double>(value);
  if (!std::isfinite(product)) {
    return false;
  }
  sum += product;
  return std::isfinite(sum);
}

bool update_violation(long double violation, double& maximum) {
  if (!std::isfinite(violation) || violation > std::numeric_limits<double>::max()) {
    return false;
  }
  maximum = std::max(maximum, static_cast<double>(violation));
  return true;
}

struct BoundCheck {
  bool arithmetic_ok = true;
  bool satisfied = true;
};

BoundCheck check_bound(long double activity, double bound, bool lower,
                       const Tolerances& tolerances, double& maximum) {
  if (!std::isfinite(bound)) {
    return {};
  }
  const auto edge = static_cast<long double>(bound);
  const auto violation = std::max(0.0L, lower ? edge - activity : activity - edge);
  if (!update_violation(violation, maximum)) {
    return {false, false};
  }
  const auto scale = std::max(std::abs(activity), std::abs(edge));
  const auto allowed = static_cast<long double>(tolerances.feasibility_absolute) +
                       static_cast<long double>(tolerances.feasibility_relative) * scale;
  if (!std::isfinite(allowed)) {
    return {false, false};
  }
  return {true, violation <= allowed};
}

}  // 匿名命名空间

Verification verify_solution(const Model& model, std::span<const double> values,
                             std::optional<double> reported_objective,
                             const Tolerances& tolerances) {
  Verification result;
  result.errors = validate_model(model);
  if (values.size() != model.costs.size()) {
    result.errors.emplace_back("候选解长度与变量数不一致");
  }
  if (!valid_tolerance(tolerances.feasibility_absolute) ||
      !valid_tolerance(tolerances.feasibility_relative) ||
      !valid_tolerance(tolerances.integrality) ||
      !valid_tolerance(tolerances.objective_absolute) ||
      !valid_tolerance(tolerances.objective_relative)) {
    result.errors.emplace_back("验证容差必须为非负有限数");
  }
  for (std::size_t column = 0; column < values.size(); ++column) {
    if (!std::isfinite(values[column])) {
      result.errors.emplace_back("候选解必须为有限数：" + std::to_string(column));
    }
  }
  if (reported_objective && !std::isfinite(*reported_objective)) {
    result.errors.emplace_back("报告的目标值必须为有限数");
  }
  if (!result.errors.empty()) {
    return result;
  }

  result.valid_input = true;
  bool bounds_ok = true;
  bool rows_ok = true;
  bool integrality_ok = true;
  bool arithmetic_ok = true;
  long double objective = model.objective_offset;
  std::vector<long double> activity(model.row_lower.size(), 0.0L);
  for (std::size_t column = 0; column < values.size(); ++column) {
    if (!add_product(objective, model.costs[column], values[column])) {
      result.errors.emplace_back("重算目标时发生算术溢出");
      return result;
    }
    const auto lower = check_bound(values[column], model.column_lower[column], true,
                                   tolerances, result.max_bound_violation);
    const auto upper = check_bound(values[column], model.column_upper[column], false,
                                   tolerances, result.max_bound_violation);
    arithmetic_ok = arithmetic_ok && lower.arithmetic_ok && upper.arithmetic_ok;
    bounds_ok = bounds_ok && lower.satisfied && upper.satisfied;
    if (model.kinds[column] == VariableKind::integer) {
      const auto value = static_cast<long double>(values[column]);
      const auto violation = std::abs(value - std::round(value));
      arithmetic_ok = update_violation(violation, result.max_integrality_violation) &&
                      arithmetic_ok;
      integrality_ok = integrality_ok && violation <= tolerances.integrality;
    }
    const auto begin = static_cast<std::size_t>(model.column_start[column]);
    const auto end = static_cast<std::size_t>(model.column_start[column + 1]);
    for (auto position = begin; position < end; ++position) {
      const auto row = static_cast<std::size_t>(model.row_index[position]);
      if (!add_product(activity[row], model.values[position], values[column])) {
        result.errors.emplace_back("重算行活动值时发生算术溢出：" + std::to_string(row));
        return result;
      }
    }
  }
  for (std::size_t row = 0; row < activity.size(); ++row) {
    const auto lower = check_bound(activity[row], model.row_lower[row], true,
                                   tolerances, result.max_row_violation);
    const auto upper = check_bound(activity[row], model.row_upper[row], false,
                                   tolerances, result.max_row_violation);
    arithmetic_ok = arithmetic_ok && lower.arithmetic_ok && upper.arithmetic_ok;
    rows_ok = rows_ok && lower.satisfied && upper.satisfied;
  }
  if (!arithmetic_ok) {
    result.errors.emplace_back("计算违反量或容差时发生算术溢出");
    return result;
  }
  if (!std::isfinite(objective) || std::abs(objective) > std::numeric_limits<double>::max()) {
    result.errors.emplace_back("重算目标超出有限 double 的表示范围");
    return result;
  }
  result.objective = static_cast<double>(objective);
  result.feasible = bounds_ok && rows_ok && integrality_ok;
  result.objective_consistent = true;
  if (reported_objective) {
    const auto reported = static_cast<long double>(*reported_objective);
    const auto error = std::abs(objective - reported);
    const auto scale = std::max(std::abs(objective), std::abs(reported));
    const auto allowed = static_cast<long double>(tolerances.objective_absolute) +
                         static_cast<long double>(tolerances.objective_relative) * scale;
    if (!std::isfinite(error) || !std::isfinite(allowed)) {
      result.objective_consistent = false;
      result.errors.emplace_back("比较目标值时发生算术溢出");
    } else {
      result.objective_consistent = error <= allowed;
    }
  }
  if (!bounds_ok) {
    result.errors.emplace_back("候选解违反变量界");
  }
  if (!rows_ok) {
    result.errors.emplace_back("候选解违反行约束");
  }
  if (!integrality_ok) {
    result.errors.emplace_back("候选解违反整数性要求");
  }
  if (!result.objective_consistent) {
    result.errors.emplace_back("报告目标与原模型重算目标不一致");
  }
  return result;
}

}  // 命名空间 solver
