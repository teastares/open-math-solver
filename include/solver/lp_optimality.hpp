#pragma once

#include "solver/verification.hpp"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace solver {

struct LpOptimalityOptions {
  double dual_absolute = 1e-7;
  double dual_relative = 1e-9;
  double gap_absolute = 1e-7;
  double gap_relative = 1e-9;
};

// 对偶向量总是针对转为最小化的模型，正值对应行下界。
// 浮点残差检查不构成精确算术证明；估计对偶目标不一定是有效界。
struct LpOptimality {
  bool valid_input = false;
  bool dual_feasible = false;
  bool gap_satisfied = false;
  bool numerically_optimal = false;
  double max_dual_residual = 0.0;
  std::optional<double> dual_objective_estimate;
  std::optional<double> corrected_dual_bound;
  std::optional<double> objective_gap;
  std::optional<double> relative_objective_gap;
  std::vector<std::string> errors;
};

// 在原模型上独立重算一次原始检查，同时返回对应的 LP 对偶与间隙检查。
struct LpVerification {
  Verification primal;
  LpOptimality optimality;
};

// 将规范化行对偶投影至合法符号域；非有限乘子保持原样，交给检查器拒绝。
// 行界或维度非法时返回假且不修改乘子；返回真不代表模型或候选已通过验证。
bool project_lp_row_dual_domain(const Model& model, std::span<double> row_dual);

LpVerification verify_lp_solution(const Model& model,
                                  std::span<const double> primal,
                                  std::span<const double> row_dual,
                                  const Tolerances& primal_tolerances = {},
                                  const LpOptimalityOptions& options = {});

LpOptimality verify_lp_optimality(const Model& model,
                                  std::span<const double> primal,
                                  std::span<const double> row_dual,
                                  const Tolerances& primal_tolerances = {},
                                  const LpOptimalityOptions& options = {});

}  // 命名空间 solver
