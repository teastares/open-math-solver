#pragma once

#include "solver/model.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace solver {

struct Tolerances {
  // 有限边界使用绝对容差加相对容差乘以活动值与边界的最大绝对值。
  double feasibility_absolute = 1e-7;
  double feasibility_relative = 1e-9;
  double integrality = 1e-6;
  double objective_absolute = 1e-7;
  double objective_relative = 1e-9;
};

// 这里只检查候选解的原始可行性，不构成最优性或不可行性证明。
struct Verification {
  bool valid_input = false;
  bool feasible = false;
  // 未传入报告目标且重算成功时为真；为真不代表已证明最优。
  bool objective_consistent = false;
  std::optional<double> objective;
  double max_row_violation = 0.0;
  double max_bound_violation = 0.0;
  double max_integrality_violation = 0.0;
  std::vector<std::string> errors;
};

Verification verify_solution(const Model& model, std::span<const double> values,
                             std::optional<double> reported_objective = std::nullopt,
                             const Tolerances& tolerances = {});

}  // 命名空间 solver
