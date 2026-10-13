#include "solver/pdlp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
int checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

solver::Model unique_model() {
  // min x+2y，x+y>=3，0<=x<=2，0<=y<=4；唯一最优点为(2,1)。
  solver::Model model;
  model.costs = {1.0, 2.0};
  model.column_lower = {0.0, 0.0};
  model.column_upper = {2.0, 4.0};
  model.kinds = {solver::VariableKind::continuous, solver::VariableKind::continuous};
  model.row_lower = {3.0, -2.0};
  model.row_upper = {infinity, 3.0};
  model.column_start = {0, 2, 4};
  model.row_index = {0, 1, 0, 1};
  model.values = {1.0, 1.0, 1.0, -1.0};
  return model;
}

void check_returned_pair(const solver::Model& original, const solver::SolveResult& result,
                         const solver::SolveOptions& options) {
  require(result.backend == "pdlp", "自主测试不得借用外部数值后端");
  require(result.model.costs == original.costs && result.model.values == original.values &&
              result.model.row_lower == original.row_lower && result.model.row_upper == original.row_upper &&
              result.model.column_lower == original.column_lower &&
              result.model.column_upper == original.column_upper &&
              result.model.objective_offset == original.objective_offset &&
              result.model.sense == original.sense,
          "返回模型必须保留输入系数、界和目标语义");
  if (!result.has_solution) {
    require(result.solution.empty() && !result.objective,
            "没有已验证候选时不得发布向量或目标");
    require(result.status != "optimal", "没有已验证候选不得报告最优");
    return;
  }
  const auto primal = solver::verify_solution(original, result.solution, result.objective,
                                              options.tolerances);
  require(primal.valid_input && primal.feasible && primal.objective_consistent,
          "任何返回候选都必须在未变换输入上复核通过");
  const auto optimality = solver::verify_lp_optimality(original, result.solution, result.row_dual,
                                                       options.tolerances, options.pdlp.optimality);
  require(result.lp_optimality.has_value(), "自主候选必须有同一对原始/对偶诊断");
  require(result.lp_optimality->numerically_optimal == optimality.numerically_optimal,
          "返回诊断不得混用历史候选与另一个对偶点");
  if (result.best_bound) {
    require(optimality.corrected_dual_bound.has_value(),
            "估计对偶目标不能冒充有限修正界");
    near(*result.best_bound, *optimality.corrected_dual_bound,
         1e-12 * std::max(1.0, std::abs(*result.best_bound)), "返回界属于同一对偶点");
  } else {
    require(!result.absolute_gap && !result.relative_gap, "无有限修正界时不得伪造有效界间隙");
  }
  if (result.status == "optimal") {
    require(optimality.numerically_optimal &&
                result.verification_level == "primal_dual_residuals_checked",
            "最优状态必须由原坐标原始/对偶检查支持");
  }
}

solver::SolveResult solve_checked(const solver::Model& model, const solver::SolveOptions& options,
                                   bool must_converge = true) {
  auto result = solver::solve_pdlp(model, options);
  check_returned_pair(model, result, options);
  if (must_converge) {
    require(result.status == "optimal", options.pdlp.mode + " 简单解析模型应收敛：" +
                result.status + "；" + result.message);
  }
  return result;
}

void check_unique(const solver::SolveResult& result, double x = 2.0, double y = 1.0,
                  double tolerance = 2e-4) {
  require(result.solution.size() == 2, "二维解析模型必须返回两个变量");
  near(result.solution[0], x, tolerance, "第一个变量与独立解析最优点不一致");
  near(result.solution[1], y, tolerance, "第二个变量与独立解析最优点不一致");
}

void transformations(const solver::SolveOptions& options) {
  const auto original = unique_model();
  check_unique(solve_checked(original, options));

  auto model = original;
  const double row_factor[] = {-8.0, 0.125};
  for (std::size_t row = 0; row < model.row_lower.size(); ++row) {
    const double lower = model.row_lower[row] * row_factor[row];
    const double upper = model.row_upper[row] * row_factor[row];
    model.row_lower[row] = std::min(lower, upper);
    model.row_upper[row] = std::max(lower, upper);
  }
  for (std::size_t k = 0; k < model.values.size(); ++k) {
    model.values[k] *= row_factor[static_cast<std::size_t>(model.row_index[k])];
  }
  check_unique(solve_checked(model, options));

  // 输入等价代换 x=8z+1，y=-w/4-2；独立解析点为(z,w)=(1/8,-12)。
  model = original;
  const double column_factor[] = {8.0, -0.25};
  const double shift[] = {1.0, -2.0};
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    model.objective_offset += original.costs[col] * shift[col];
    model.costs[col] *= column_factor[col];
    const double lower = (original.column_lower[col] - shift[col]) / column_factor[col];
    const double upper = (original.column_upper[col] - shift[col]) / column_factor[col];
    model.column_lower[col] = std::min(lower, upper);
    model.column_upper[col] = std::max(lower, upper);
    for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      const auto row = static_cast<std::size_t>(model.row_index[pos]);
      model.row_lower[row] -= original.values[pos] * shift[col];
      model.row_upper[row] -= original.values[pos] * shift[col];
      model.values[pos] *= column_factor[col];
    }
  }
  check_unique(solve_checked(model, options), 0.125, -12.0, 1e-3);

  model = original;
  model.sense = solver::ObjectiveSense::maximize;
  model.objective_offset = 17.0;
  for (double& cost : model.costs) cost = -cost;
  const auto maximum = solve_checked(model, options);
  check_unique(maximum);
  near(*maximum.objective, 13.0, 2e-4, "最大化等价变换必须保留目标方向及常数");

  for (double offset : {-1e30, 1e30}) {
    model = original;
    model.objective_offset = offset;
    check_unique(solve_checked(model, options));
  }
  for (double factor : {0.03125, 32.0}) {
    model = original;
    for (double& cost : model.costs) cost *= factor;
    const auto result = solve_checked(model, options);
    check_unique(result);
    near(*result.objective / factor, 4.0, 2e-4, "正目标缩放不应改变解析最优点");
  }

  // 同时交换两行和两列，避免缓存依赖偶然的存储顺序。
  model = original;
  std::swap(model.costs[0], model.costs[1]);
  std::swap(model.column_lower[0], model.column_lower[1]);
  std::swap(model.column_upper[0], model.column_upper[1]);
  std::swap(model.row_lower[0], model.row_lower[1]);
  std::swap(model.row_upper[0], model.row_upper[1]);
  model.row_index = {1, 0, 1, 0};
  model.values = {1.0, -1.0, 1.0, 1.0};
  check_unique(solve_checked(model, options), 1.0, 2.0);

  // 重复项包含恰好相消的项；有效矩阵仍然等于原模型。
  model = original;
  model.column_start = {0, 4, 8};
  model.row_index = {1, 0, 0, 0, 1, 0, 0, 0};
  model.values = {1.0, 1.0, 4.0, -4.0, -1.0, 1.0, 2.0, -2.0};
  check_unique(solve_checked(model, options));
}

void degeneracy_and_budget(const solver::SolveOptions& options) {
  auto model = unique_model();
  model.costs = {1.0, 1.0};
  model.row_lower = {1.0, -infinity};
  model.row_upper = {infinity, infinity};
  const auto face = solve_checked(model, options);
  near(face.solution[0] + face.solution[1], 1.0, 2e-5,
       "退化最优面只要求满足解析最优条件，不固定坐标选择");

  model.costs = {0.0, 0.0};
  model.row_lower = {1.0, -infinity};
  model.row_upper = {1.0, infinity};
  const auto zero_cost = solve_checked(model, options);
  near(zero_cost.solution[0] + zero_cost.solution[1], 1.0, 2e-5,
       "零目标时仍必须满足等式，不能靠零间隙提前成功");

  auto budget = options;
  budget.pdlp.max_iterations = 0;
  auto result = solve_checked(unique_model(), budget, false);
  require(result.status == "iteration_limit" && result.pdlp_iterations == 0 &&
              !result.has_solution && !result.lp_optimality,
          "零迭代预算不得执行线搜索、初始化认证或发布候选");
  budget = options;
  budget.time_limit = 0.0;
  result = solve_checked(unique_model(), budget, false);
  require(result.status == "time_limit" && result.pdlp_iterations == 0 && !result.has_solution,
          "零时间预算不得执行线搜索或发布候选");
  budget = options;
  budget.time_limit = 1e-12;
  result = solve_checked(unique_model(), budget, false);
  require(result.status == "time_limit" && result.pdlp_iterations == 0,
          "预处理阶段耗尽时间后不得继续进行迭代");

  budget = options;
  budget.pdlp.max_iterations = 17;
  budget.pdlp.check_interval = 50;
  model = unique_model();
  // 同一行表达式同时要求大于等于4和小于等于1，无解且具有有限变量界。
  model.row_lower = {4.0, -infinity};
  model.row_upper = {infinity, 1.0};
  model.values = {1.0, 1.0, 1.0, 1.0};
  result = solve_checked(model, budget, false);
  require(result.status == "iteration_limit" || result.status == "numerical_error" ||
              result.status == "time_limit",
          "未实现证书时，矛盾约束的线搜索停滞只能返回未完成或数值困难");
  require(!result.has_solution && result.pdlp_iterations && *result.pdlp_iterations <= 17,
          "线搜索拒绝或重启不能绕过迭代预算，也不能伪造不可行模型的候选");

  model = {};
  model.costs = {-1.0};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  result = solve_checked(model, budget, false);
  require(result.status != "optimal" && result.status != "infeasible" &&
              result.status != "unbounded" && result.pdlp_iterations &&
              *result.pdlp_iterations <= 17,
          "零交叉能量不能将无界方向当作已收敛或越过迭代预算");
  require(!result.best_bound && !result.absolute_gap && !result.relative_gap,
          "自由无界方向不得输出估计目标形成的伪有限界");
}

}  // 匿名命名空间

int main() {
  try {
    for (const std::string mode : {"baseline", "adaptive", "subspace"}) {
      solver::SolveOptions options;
      options.pdlp.mode = mode;
      options.pdlp.max_iterations = 100000;
      options.pdlp.check_interval = 25;
      options.time_limit = 3.0;
      transformations(options);
      degeneracy_and_budget(options);
    }
    std::cout << "自主 PDHG 变换、退化与预算防错检查通过，共 " << checks << " 项。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "自主 PDHG 防错检查失败：" << error.what() << '\n';
    return 1;
  }
}
