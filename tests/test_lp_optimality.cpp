#include "solver/lp_optimality.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

int checks = 0;
constexpr auto infinity = std::numeric_limits<double>::infinity();
constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    std::cerr << "验证失败：" << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void near(const std::optional<double>& actual, double expected, const std::string& message) {
  require(actual.has_value() && std::abs(*actual - expected) <= 1e-12, message);
}

solver::Model scalar(double cost, double lower, double upper) {
  solver::Model model;
  model.costs = {cost};
  model.column_lower = {lower};
  model.column_upper = {upper};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  return model;
}

void add_row(solver::Model& model, double coefficient, double lower, double upper) {
  model.row_lower = {lower};
  model.row_upper = {upper};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {coefficient};
}

void analytic_optima_and_bounds() {
  auto model = scalar(2.0, 1.0, 4.0);
  model.objective_offset = 7.0;
  auto result = solver::verify_lp_optimality(model, std::vector<double>{1.0}, {});
  require(result.valid_input && result.numerically_optimal, "有限变量界定义的最小值");
  require(result.dual_feasible && result.gap_satisfied, "有限变量界不要求约化成本为零");
  near(result.dual_objective_estimate, 9.0, "估计对偶目标包含常数");
  near(result.corrected_dual_bound, 9.0, "最小化修正下界包含常数");
  near(result.objective_gap, 0.0, "解析最优点间隙为零");
  require(result.errors.empty(), "解析最优点无错误");
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0}, {});
  require(result.valid_input && result.dual_feasible && !result.numerically_optimal,
          "非最优可行点不通过间隙检查");
  near(result.corrected_dual_bound, 9.0, "非最优点仍可得到独立对偶下界");
  near(result.objective_gap, 2.0, "最优性间隙由目标贡献计算");
  near(result.relative_objective_gap, 0.5, "相对间隙尺度不含目标常数");

  model = scalar(3.0, 0.0, infinity);
  model.objective_offset = -4.0;
  add_row(model, 1.0, 2.0, infinity);
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0},
                                       std::vector<double>{3.0});
  require(result.numerically_optimal, "正行乘子对应有限行下界");
  near(result.corrected_dual_bound, 2.0, "行下界贡献与负常数");

  model = scalar(2.0, 0.0, infinity);
  model.sense = solver::ObjectiveSense::maximize;
  model.objective_offset = 7.0;
  add_row(model, 1.0, -infinity, 3.0);
  result = solver::verify_lp_optimality(model, std::vector<double>{3.0},
                                       std::vector<double>{-2.0});
  require(result.numerically_optimal, "最大化使用规范化后的负行乘子");
  near(result.dual_objective_estimate, 13.0, "最大化估计目标恢复原方向与常数");
  near(result.corrected_dual_bound, 13.0, "最大化修正值为原问题上界");
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0},
                                       std::vector<double>{-2.0});
  require(!result.numerically_optimal && result.dual_feasible, "最大化非最优点被拒绝");
  near(result.objective_gap, 2.0, "最大化间隙保留绝对差值");

  model = scalar(3.0, -infinity, infinity);
  add_row(model, 2.0, 4.0, 4.0);
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0},
                                       std::vector<double>{1.5});
  require(result.numerically_optimal, "等式约束及自由变量的解析解");
  near(result.corrected_dual_bound, 6.0, "自由变量恰好零残差允许有限修正界");
  model.column_start = {0, 2};
  model.row_index = {0, 0};
  model.values = {1.0, 1.0};
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0},
                                       std::vector<double>{1.5});
  require(result.numerically_optimal, "重复 CSC 项参与转置乘积求和");
  near(result.corrected_dual_bound, 6.0, "重复 CSC 项不改变解析界");

  model = scalar(-2.0, -5.0, -2.0);
  result = solver::verify_lp_optimality(model, std::vector<double>{-2.0}, {});
  require(result.numerically_optimal, "负约化成本选择有限变量上界");
  near(result.corrected_dual_bound, 4.0, "负变量上界贡献");
}

void infinite_bounds_and_strict_signs() {
  auto model = scalar(1e-10, -infinity, infinity);
  auto result = solver::verify_lp_optimality(model, std::vector<double>{0.0}, {});
  require(result.dual_feasible && result.gap_satisfied && result.numerically_optimal,
          "近零自由变量残差可通过浮点容差检查");
  require(!result.corrected_dual_bound, "非零残差即使很小也不能产生修正界");
  near(result.dual_objective_estimate, 0.0, "双侧无穷变量的估计项取零");
  require(result.max_dual_residual == 1e-10, "原坐标自由变量残差被记录");
  solver::LpOptimalityOptions strict{0.0, 0.0, 0.0, 0.0};
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0}, {}, {}, strict);
  require(!result.dual_feasible && !result.numerically_optimal,
          "零容差拒绝任何非零自由变量残差");
  model.costs[0] = 0.0;
  result = solver::verify_lp_optimality(model, std::vector<double>{13.0}, {}, {}, strict);
  require(result.numerically_optimal, "恰好零乘子与无穷界不会产生非数");
  near(result.corrected_dual_bound, 0.0, "零乘无穷按零处理");

  model = scalar(1.0, -infinity, 5.0);
  result = solver::verify_lp_optimality(model, std::vector<double>{5.0}, {});
  near(result.dual_objective_estimate, 5.0, "缺少变量下界时估计使用有限上界");
  require(!result.corrected_dual_bound && !result.dual_feasible && result.gap_satisfied,
          "相等的估计目标不能掩盖无效下界及对偶残差");
  require(!result.numerically_optimal, "估计间隙为零不单独构成最优结论");
  model = scalar(-2.0, -3.0, infinity);
  result = solver::verify_lp_optimality(model, std::vector<double>{-3.0}, {});
  near(result.dual_objective_estimate, 6.0, "缺少变量上界时估计使用有限下界");
  require(!result.corrected_dual_bound && !result.dual_feasible,
          "负约化成本指向无穷上界时拒绝有限修正界");

  model = scalar(0.0, 0.0, 1.0);
  add_row(model, 1.0, 0.0, infinity);
  const solver::LpOptimalityOptions loose{10.0, 10.0, 10.0, 10.0};
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                       std::vector<double>{-1e-30}, {}, loose);
  require(result.valid_input && !result.dual_feasible && !result.numerically_optimal,
          "行上界无穷时负乘子必须严格拒绝");
  require(!result.corrected_dual_bound, "行乘子错误符号不能生成修正界");
  model.row_lower[0] = -infinity;
  model.row_upper[0] = 1.0;
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                       std::vector<double>{1e-30}, {}, loose);
  require(!result.dual_feasible, "行下界无穷时正乘子必须严格拒绝");
  model.row_upper[0] = infinity;
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                       std::vector<double>{0.0});
  require(result.numerically_optimal, "无约束行允许零乘子");
}

void primal_and_gap_checks() {
  auto model = scalar(0.0, 0.0, infinity);
  auto result = solver::verify_lp_optimality(model, std::vector<double>{-1.0}, {});
  require(result.valid_input && result.dual_feasible && result.gap_satisfied,
          "原始不可行与输入格式错误分别处理");
  require(!result.numerically_optimal, "原始不可行点不能通过最优性检查");
  model = scalar(1.0, 0.0, 1.0);
  model.objective_offset = 1e30;
  result = solver::verify_lp_optimality(model, std::vector<double>{1.0}, {});
  require(!result.gap_satisfied && !result.numerically_optimal,
          "巨大目标常数不能掩盖单位非最优间隙");
  near(result.objective_gap, 1.0, "巨大目标常数下仍记录真实贡献间隙");
  near(result.relative_objective_gap, 1.0, "相对间隙不按巨大目标常数缩小");
  model.sense = solver::ObjectiveSense::maximize;
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0}, {});
  require(!result.numerically_optimal, "最大化巨大常数同样不能掩盖非最优");
  near(result.objective_gap, 1.0, "最大化去常数间隙");

  model = scalar(1.0, 0.0, 1.0);
  solver::LpOptimalityOptions options{0.0, 0.0, 0.125, 0.0};
  result = solver::verify_lp_optimality(model, std::vector<double>{0.125}, {}, {}, options);
  require(result.numerically_optimal, "目标间隙绝对容差包含等号");
  result = solver::verify_lp_optimality(model,
      std::vector<double>{std::nextafter(0.125, infinity)}, {}, {}, options);
  require(!result.gap_satisfied, "刚超出目标间隙绝对容差被拒绝");

  model.column_lower[0] = 7.0;
  model.column_upper[0] = 10.0;
  options = {0.0, 0.0, 0.0, 0.125};
  result = solver::verify_lp_optimality(model, std::vector<double>{8.0}, {}, {}, options);
  require(result.numerically_optimal, "目标间隙相对容差包含等号");
  result = solver::verify_lp_optimality(model,
      std::vector<double>{std::nextafter(8.0, infinity)}, {}, {}, options);
  require(!result.gap_satisfied, "刚超出目标间隙相对容差被拒绝");

  model = scalar(8.0, -infinity, infinity);
  add_row(model, 1.0, 0.0, 0.0);
  options = {0.0, 0.125, 0.0, 0.0};
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                       std::vector<double>{7.0}, {}, options);
  require(result.dual_feasible, "对偶相对容差采用原成本与转置乘积尺度");
  require(!result.corrected_dual_bound, "相对容差通过不放松修正界有限方向要求");
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
      std::vector<double>{std::nextafter(7.0, -infinity)}, {}, options);
  require(!result.dual_feasible, "刚超出对偶相对容差被拒绝");

  solver::Model empty;
  empty.objective_offset = 12.0;
  result = solver::verify_lp_optimality(empty, {}, {});
  require(result.numerically_optimal, "无变量无行的空模型");
  near(result.corrected_dual_bound, 12.0, "空模型修正界等于常数");
  empty.row_lower = {1.0};
  empty.row_upper = {infinity};
  result = solver::verify_lp_optimality(empty, {}, std::vector<double>{0.0});
  require(result.valid_input && !result.numerically_optimal, "零变量不可行行不得报最优");
}

void invalid_inputs_and_overflow() {
  const auto invalid = [](const solver::Model& model, const std::vector<double>& primal,
                          const std::vector<double>& dual, const std::string& label) {
    const auto result = solver::verify_lp_optimality(model, primal, dual);
    require(!result.valid_input && !result.numerically_optimal && !result.corrected_dual_bound,
            label);
    require(!result.errors.empty(), label + "提供诊断");
  };
  auto model = scalar(1.0, 0.0, 1.0);
  invalid(model, {}, {}, "原始向量维度错误");
  invalid(model, {0.0}, {0.0}, "对偶向量维度错误");
  invalid(model, {nan}, {}, "非数原始向量");
  model.kinds[0] = solver::VariableKind::integer;
  invalid(model, {0.0}, {}, "整数模型不作为连续 LP 认证");
  model.kinds[0] = solver::VariableKind::continuous;
  model.column_start.back() = 2;
  invalid(model, {0.0}, {}, "坏 CSC 在转置乘法前拒绝");
  model = scalar(1.0, 0.0, 1.0);
  add_row(model, 1.0, 0.0, infinity);
  invalid(model, {0.0}, {nan}, "非数行乘子");
  invalid(model, {0.0}, {infinity}, "无穷行乘子");
  for (int field = 0; field < 4; ++field) {
    for (double value : {-1.0, nan, infinity}) {
      solver::LpOptimalityOptions options;
      double* fields[] = {&options.dual_absolute, &options.dual_relative,
                          &options.gap_absolute, &options.gap_relative};
      *fields[field] = value;
      const auto result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                                      std::vector<double>{1.0}, {}, options);
      require(!result.valid_input && !result.numerically_optimal, "非法对偶与间隙容差");
    }
  }
  solver::Tolerances bad_primal;
  bad_primal.feasibility_relative = -1.0;
  auto result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                            std::vector<double>{1.0}, bad_primal);
  require(!result.valid_input, "复用原始验证器的容差校验");

  model = scalar(std::numeric_limits<double>::max(), 0.0, infinity);
  result = solver::verify_lp_optimality(model, std::vector<double>{2.0}, {});
  require(!result.numerically_optimal && !result.corrected_dual_bound && !result.errors.empty(),
          "原始目标溢出不得产生最优结论");
  model = scalar(0.0, -infinity, infinity);
  add_row(model, std::numeric_limits<double>::max(), 0.0, 0.0);
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
      std::vector<double>{std::numeric_limits<double>::max()});
  require(!result.numerically_optimal && !result.corrected_dual_bound && !result.errors.empty(),
          "转置乘积或残差超出表示范围时保守失败");
  model = scalar(0.0, 0.0, 0.0);
  add_row(model, 1.0, -std::numeric_limits<double>::max(), infinity);
  result = solver::verify_lp_optimality(model, std::vector<double>{0.0},
                                       std::vector<double>{2.0});
  require(!result.numerically_optimal && !result.dual_objective_estimate && !result.errors.empty(),
          "对偶目标溢出不能变成无穷有效界");
}

solver::LpVerification compare_entry_points(const solver::Model& model,
                                            const std::vector<double>& primal,
                                            const std::vector<double>& dual,
                                            const std::string& label,
                                            const solver::Tolerances& tolerances = {},
                                            const solver::LpOptimalityOptions& options = {}) {
  auto combined = solver::verify_lp_solution(model, primal, dual, tolerances, options);
  const auto separate_primal = solver::verify_solution(model, primal, std::nullopt, tolerances);
  const auto separate_dual = solver::verify_lp_optimality(model, primal, dual, tolerances, options);
  const auto& checked_primal = combined.primal;
  const auto& checked_dual = combined.optimality;
  require(checked_primal.valid_input == separate_primal.valid_input &&
          checked_primal.feasible == separate_primal.feasible &&
          checked_primal.objective_consistent == separate_primal.objective_consistent &&
          checked_primal.objective == separate_primal.objective &&
          checked_primal.max_row_violation == separate_primal.max_row_violation &&
          checked_primal.max_bound_violation == separate_primal.max_bound_violation &&
          checked_primal.max_integrality_violation == separate_primal.max_integrality_violation &&
          checked_primal.errors == separate_primal.errors,
          label + "：组合原始结果与独立入口完全一致");
  require(checked_dual.valid_input == separate_dual.valid_input &&
          checked_dual.dual_feasible == separate_dual.dual_feasible &&
          checked_dual.gap_satisfied == separate_dual.gap_satisfied &&
          checked_dual.numerically_optimal == separate_dual.numerically_optimal &&
          checked_dual.max_dual_residual == separate_dual.max_dual_residual &&
          checked_dual.dual_objective_estimate == separate_dual.dual_objective_estimate &&
          checked_dual.corrected_dual_bound == separate_dual.corrected_dual_bound &&
          checked_dual.objective_gap == separate_dual.objective_gap &&
          checked_dual.relative_objective_gap == separate_dual.relative_objective_gap &&
          checked_dual.errors == separate_dual.errors,
          label + "：组合最优性结果与兼容入口完全一致");
  return combined;
}

void combined_verification() {
  auto model = scalar(2.0, 0.0, infinity);
  model.sense = solver::ObjectiveSense::maximize;
  model.objective_offset = 7.0;
  add_row(model, 1.0, -infinity, 3.0);
  auto result = compare_entry_points(model, {3.0}, {-2.0}, "带常数最大化解析最优点");
  require(result.primal.feasible && result.optimality.numerically_optimal,
          "组合入口同时返回原始可行与浮点最优结论");
  near(result.primal.objective, 13.0, "组合原始目标恢复最大化常数");
  near(result.optimality.corrected_dual_bound, 13.0, "组合最大化修正界为上界");
  result = compare_entry_points(model, {2.0}, {-2.0}, "原始可行但非最优");
  require(result.primal.feasible && !result.optimality.numerically_optimal,
          "组合入口不混淆原始可行与最优性");
  result = compare_entry_points(model, {4.0}, {-2.0}, "违反行上界");
  require(!result.primal.feasible && !result.optimality.numerically_optimal &&
          result.primal.max_row_violation == 1.0,
          "组合入口保留原始违反量并拒绝最优结论");
  result = compare_entry_points(model, {3.0}, {2.0}, "严格非法行乘子符号");
  require(result.primal.feasible && !result.optimality.dual_feasible &&
          !result.optimality.corrected_dual_bound,
          "对偶失败不改变组合结果中的原始可行性");

  model = scalar(1.0, 0.0, 1.0);
  model.objective_offset = 1e30;
  result = compare_entry_points(model, {1.0}, {}, "巨大目标常数与非最优点");
  near(result.optimality.objective_gap, 1.0, "组合检查保留去常数间隙");
  require(!result.optimality.numerically_optimal, "组合检查不受巨大目标常数掩盖");
  model = scalar(1e-10, -infinity, infinity);
  result = compare_entry_points(model, {0.0}, {}, "容差内自由变量残差");
  require(result.optimality.numerically_optimal && !result.optimality.corrected_dual_bound,
          "组合检查不将估计目标升级为有效界");

  model = scalar(1.0, 0.0, 1.0);
  result = compare_entry_points(model, {}, {}, "原始维度错误");
  require(!result.primal.valid_input && !result.optimality.valid_input,
          "组合入口同时报告原始输入错误");
  result = compare_entry_points(model, {0.0}, {0.0}, "对偶维度错误");
  require(result.primal.valid_input && !result.optimality.valid_input,
          "对偶维度错误保留独立原始检查结果");
  compare_entry_points(model, {nan}, {}, "非有限原始值");
  model.column_start.back() = 1;
  compare_entry_points(model, {0.0}, {}, "非法 CSC");
  model.column_start.back() = 0;
  model.kinds[0] = solver::VariableKind::integer;
  result = compare_entry_points(model, {0.5}, {}, "整数模型与整数违反");
  require(result.primal.valid_input && !result.primal.feasible &&
          result.primal.max_integrality_violation == 0.5 && !result.optimality.valid_input,
          "组合入口保留整数检查但拒绝 LP 最优性认证");

  model = scalar(1.0, 0.0, 1.0);
  solver::LpOptimalityOptions bad_dual;
  bad_dual.gap_absolute = nan;
  compare_entry_points(model, {0.0}, {}, "非法对偶容差", {}, bad_dual);
  solver::Tolerances bad_primal;
  bad_primal.feasibility_relative = -1.0;
  compare_entry_points(model, {0.0}, {}, "非法原始容差", bad_primal);
  add_row(model, 1.0, 0.0, infinity);
  compare_entry_points(model, {0.0}, {infinity}, "非有限对偶值");

  model = scalar(std::numeric_limits<double>::max(), 0.0, infinity);
  result = compare_entry_points(model, {2.0}, {}, "原始目标溢出");
  require(result.primal.valid_input && !result.primal.objective &&
          !result.optimality.numerically_optimal && !result.optimality.corrected_dual_bound,
          "组合入口保留合法输入与算术失败的区别");
  model = scalar(0.0, 0.0, 0.0);
  add_row(model, 1.0, -std::numeric_limits<double>::max(), infinity);
  result = compare_entry_points(model, {0.0}, {2.0}, "对偶目标溢出");
  require(result.primal.feasible && !result.optimality.dual_objective_estimate &&
          !result.optimality.numerically_optimal,
          "对偶算术失败不会破坏已有原始结果");
  compare_entry_points(solver::Model{}, {}, {}, "空模型");

  model = scalar(1e8, 0.0, infinity);
  add_row(model, 1.0, 1.0, infinity);
  result = compare_entry_points(model, {1.0 - 5e-8}, {1e8}, "原始容差被大乘子放大");
  require(result.primal.feasible && result.optimality.dual_feasible &&
          result.optimality.corrected_dual_bound == 1e8 &&
          *result.primal.objective < *result.optimality.corrected_dual_bound &&
          !result.optimality.gap_satisfied && !result.optimality.numerically_optimal,
          "容差可行点可能优于真实最优值，不能截断负间隙后误报最优");
}

void row_dual_domain_projection() {
  solver::Model model;
  model.row_lower = {-1.0, -1.0, 0.0, -infinity, -infinity,
                     0.0, -infinity, -infinity, 0.0};
  model.row_upper = {1.0, 1.0, infinity, 0.0, infinity,
                     infinity, 0.0, infinity, 0.0};
  std::vector<double> dual{2.0, -3.0, -4.0, 5.0, 6.0, 7.0, -8.0, -9.0, 10.0};
  const std::vector<double> expected{2.0, -3.0, 0.0, 0.0, 0.0, 7.0, -8.0, 0.0, 10.0};
  require(solver::project_lp_row_dual_domain(model, dual), "全部合法行域可进行投影");
  require(dual == expected, "双边与等式乘子保留，单边及自由行按符号域投影");
  require(solver::project_lp_row_dual_domain(model, dual) && dual == expected,
          "重复投影不改变已合法的乘子");
  model.sense = solver::ObjectiveSense::maximize;
  require(solver::project_lp_row_dual_domain(model, dual) && dual == expected,
          "最大化模型的乘子仍使用规范化符号约定");
  require(solver::project_lp_row_dual_domain(solver::Model{}, {}), "空行对偶域可投影");

  model = scalar(0.0, 0.0, 1.0);
  add_row(model, 1.0, 0.0, infinity);
  dual = {-1e-30};
  auto check = solver::verify_lp_solution(model, std::vector<double>{0.0}, dual);
  require(!check.optimality.dual_feasible && !check.optimality.numerically_optimal,
          "严格检查器继续拒绝未经投影的极小错误符号");
  require(solver::project_lp_row_dual_domain(model, dual) && dual[0] == 0.0,
          "极小错误符号投影为零且不引入容差");
  check = solver::verify_lp_solution(model, std::vector<double>{0.0}, dual);
  require(check.optimality.numerically_optimal,
          "投影后的解析最优对偶仍必须通过全部原模型检查");
  model.costs[0] = 1.0;
  model.row_lower[0] = 1.0;
  dual = {-1e6};
  require(solver::project_lp_row_dual_domain(model, dual) && dual[0] == 0.0,
          "较大错误符号同样按数学域投影");
  check = solver::verify_lp_solution(model, std::vector<double>{1.0}, dual);
  require(check.primal.feasible && !check.optimality.gap_satisfied &&
          !check.optimality.numerically_optimal,
          "投影成功不授权最优结论，仍需重新验证目标间隙");

  // 即使自由行的合法域只有零，也不能把非有限输入悄悄变成合法乘子。
  const std::vector<double> lower{0.0, -infinity, -1.0, -infinity};
  const std::vector<double> upper{infinity, 0.0, 1.0, infinity};
  for (std::size_t domain = 0; domain < lower.size(); ++domain) {
    model = scalar(0.0, 0.0, 1.0);
    add_row(model, 1.0, lower[domain], upper[domain]);
    for (double value : {nan, infinity, -infinity}) {
      dual = {value};
      require(solver::project_lp_row_dual_domain(model, dual), "非有限乘子不影响行域结构检查");
      require(std::isnan(value) ? std::isnan(dual[0]) : dual[0] == value,
              "所有行域均保留非有限乘子的原值");
      check = solver::verify_lp_solution(model, std::vector<double>{0.0}, dual);
      require(!check.optimality.valid_input && !check.optimality.numerically_optimal,
              "非有限乘子经过投影后仍由严格检查器拒绝");
    }
  }

  model = scalar(0.0, 0.0, 1.0);
  add_row(model, 1.0, 0.0, infinity);
  dual = {-2.0, -3.0};
  require(!solver::project_lp_row_dual_domain(model, dual) &&
          dual == std::vector<double>({-2.0, -3.0}),
          "对偶向量过长时返回失败且不修改任何元素");
  dual.clear();
  require(!solver::project_lp_row_dual_domain(model, dual) && dual.empty(),
          "对偶向量过短时返回失败");
  model.row_lower.push_back(0.0);
  dual = {-2.0, -3.0};
  require(!solver::project_lp_row_dual_domain(model, dual) &&
          dual == std::vector<double>({-2.0, -3.0}),
          "行上下界维度不一致时不进行部分投影");
  model.row_upper.push_back(infinity);
  for (const auto& bounds : std::vector<std::pair<double, double>>{
           {nan, infinity}, {0.0, nan}, {infinity, infinity},
           {-infinity, -infinity}, {2.0, 1.0}}) {
    model.row_lower[1] = bounds.first;
    model.row_upper[1] = bounds.second;
    require(!solver::project_lp_row_dual_domain(model, dual) &&
            dual == std::vector<double>({-2.0, -3.0}),
            "后续行界非法时前面乘子也保持不变");
  }
}

}  // 匿名命名空间

int main() {
  analytic_optima_and_bounds();
  infinite_bounds_and_strict_signs();
  primal_and_gap_checks();
  invalid_inputs_and_overflow();
  combined_verification();
  row_dual_domain_projection();
  std::cout << "LP 原始与对偶检查通过，共 " << checks << " 项检查。\n";
  return EXIT_SUCCESS;
}
