#include "solver/pdlp.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double infinity = std::numeric_limits<double>::infinity();
int checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

solver::Model two_variables() {
  solver::Model model;
  model.costs = {1, 2};
  model.column_lower = {0, 0};
  model.column_upper = {2, 4};
  model.kinds.assign(2, solver::VariableKind::continuous);
  model.row_lower = {3, -2};
  model.row_upper = {infinity, 3};
  model.column_start = {0, 2, 4};
  model.row_index = {0, 1, 0, 1};
  model.values = {1, 1, 1, -1};
  return model;
}

solver::SolveResult solve_checked(const solver::Model& model,
                                  const solver::SolveOptions& options,
                                  bool optimal = true) {
  auto result = solver::solve_pdlp(model, options);
  require(result.backend == "pdlp", "Halpern 不得借用外部数值后端");
  require(result.pdlp_iterations && *result.pdlp_iterations <= options.pdlp.max_iterations,
          "近端映射及反射更新不得超出迭代预算");
  if (*result.pdlp_iterations > 0) {
    require(result.pdlp_acceleration_attempts == 0 && result.pdlp_acceleration_accepted == 0,
            "Halpern 迭代不能冒充历史子空间候选");
    require(result.pdlp_operator_passes &&
                *result.pdlp_operator_passes >= 2 * *result.pdlp_iterations + 1,
            "每次固定点映射的两个稀疏乘积和初始化必须计入算子账本");
  }
  require(std::isfinite(result.total_seconds) && result.total_seconds >= 0 &&
              std::isfinite(result.solve_seconds) && result.solve_seconds >= 0 &&
              std::isfinite(result.verification_seconds) && result.verification_seconds >= 0 &&
              std::abs(result.total_seconds - result.solve_seconds - result.verification_seconds) <= 1e-8,
          "固定步长准备、锚定、缓存刷新及独立验算必须计入时间");
  if (result.has_solution) {
    const auto verification = solver::verify_lp_solution(model, result.solution, result.row_dual,
                                                         options.tolerances,
                                                         options.pdlp.optimality);
    require(verification.primal.valid_input && verification.primal.feasible &&
                verification.primal.objective_consistent &&
                verification.primal.objective == result.objective,
            "返回候选必须通过原模型独立原始检查");
    require(result.lp_optimality && result.lp_optimality->numerically_optimal ==
                verification.optimality.numerically_optimal,
            "返回原始点、对偶点及诊断必须属于同一近端输出");
    if (result.best_bound) {
      require(result.best_bound == verification.optimality.corrected_dual_bound,
              "有效界不得来自另一历史点或反射状态");
    }
    if (result.status == "optimal") {
      require(verification.optimality.numerically_optimal &&
                  result.verification_level == "primal_dual_residuals_checked",
              "固定点残差不能代替原模型最优性检查");
    }
  } else {
    require(result.solution.empty() && !result.objective && result.status != "optimal",
            "没有已验算候选时不得输出反射状态");
  }
  if (optimal) require(result.status == "optimal", options.pdlp.mode + " 应收敛：" +
                         result.status + "；" + result.message);
  return result;
}

void analytic_tests(const solver::SolveOptions& options) {
  auto model = two_variables();
  auto result = solve_checked(model, options);
  require(std::abs(result.solution[0] - 2) < 1e-5 &&
              std::abs(result.solution[1] - 1) < 1e-5,
          "带盒约束的解析最优点应为(2,1)");

  model.objective_offset = 1e30;
  model.sense = solver::ObjectiveSense::maximize;
  for (double& value : model.costs) value = -value;
  result = solve_checked(model, options);
  require(std::abs(result.solution[0] - 2) < 1e-5 &&
              std::abs(result.solution[1] - 1) < 1e-5,
          "大目标常数和最大化不能掩盖未收敛的变量部分");

  model = two_variables();
  model.row_lower = {-infinity, -16};
  model.row_upper = {-0.375, 24};
  model.values = {-0.125, 8, -0.125, -8};
  result = solve_checked(model, options);
  require(std::abs(result.solution[0] - 2) < 1e-5 &&
              std::abs(result.solution[1] - 1) < 1e-5,
          "正负行缩放后的对偶符号和原始恢复必须一致");

  // 近相关等式有唯一解析解 x_j=1，成本交替变化检验固定点度量中的交叉符号。
  model = {};
  constexpr std::size_t dimension = 8;
  model.costs.resize(dimension);
  model.column_lower.assign(dimension, -2);
  model.column_upper.assign(dimension, 2);
  model.row_lower.assign(dimension, dimension + 0.25);
  model.row_upper = model.row_lower;
  model.kinds.assign(dimension, solver::VariableKind::continuous);
  model.column_start = {0};
  for (std::size_t col = 0; col < dimension; ++col) {
    model.costs[col] = (col % 2 == 0 ? 1.0 : -1.0) * (col + 1);
    for (std::size_t row = 0; row < dimension; ++row) {
      model.row_index.push_back(static_cast<std::int64_t>(row));
      model.values.push_back(row == col ? 1.25 : 1);
    }
    model.column_start.push_back(static_cast<std::int64_t>(model.values.size()));
  }
  result = solve_checked(model, options);
  for (double value : result.solution) require(std::abs(value - 1) < 1e-5,
                                               "近相关等式解必须满足各个坐标");

  model = {};
  model.costs = {1};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.row_lower = {2, -infinity};
  model.row_upper = {4, infinity};
  model.column_start = {0, 2};
  model.row_index = {0, 1};
  model.values = {1, 0};
  result = solve_checked(model, options);
  require(std::abs(result.solution[0] - 2) < 1e-6,
          "自由变量、双边约束和自由零行应收敛到解析下界");

  auto unscaled = options;
  unscaled.pdlp.scaling_passes = 0;
  result = solve_checked(two_variables(), unscaled);
  require(std::abs(*result.objective - 4) < 1e-5,
          "无缩放模式的矩阵范数界仍须给出合法固定步长");

  model = two_variables();
  model.costs = {0, 0};
  model.row_lower = {1, 1};
  model.row_upper = model.row_lower;
  model.values = {1, 1, 1, 1};
  result = solve_checked(model, options);
  require(std::abs(result.solution[0] + result.solution[1] - 1) < 2e-7,
          "零成本和秩亏等式仍需满足原始可行性");
}

void budget_and_domain_tests(const solver::SolveOptions& options) {
  auto budget = options;
  budget.pdlp.max_iterations = 0;
  auto result = solve_checked(two_variables(), budget, false);
  require(result.status == "iteration_limit" && !result.has_solution && !result.lp_optimality,
          "零迭代预算不能执行初始化认证或固定点映射");
  budget = options;
  budget.time_limit = 0;
  result = solve_checked(two_variables(), budget, false);
  require(result.status == "time_limit" && !result.has_solution,
          "零时间预算不能发布候选");
  budget = options;
  budget.time_limit = 1e-12;
  result = solve_checked(two_variables(), budget, false);
  require(result.status == "time_limit" && result.pdlp_iterations == 0,
          "初始化阶段耗尽预算不得继续固定点迭代");

  budget = options;
  budget.pdlp.max_iterations = 67;
  auto model = two_variables();
  model.row_lower = {1, 2};
  model.row_upper = model.row_lower;
  model.values = {1, 1, 1, 1};
  result = solve_checked(model, budget, false);
  require(!result.has_solution && result.status != "optimal" && result.status != "infeasible",
          "不一致等式不能由锚定抵消后误报最优或无解证书");

  model = {};
  model.costs = {-1};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  result = solve_checked(model, budget, false);
  require(result.status != "optimal" && result.status != "unbounded" && !result.best_bound,
          "零矩阵的无界移动不能被当作收敛或无界证书");

  model.column_lower = {0};
  model.column_upper = {10};
  budget.pdlp.max_iterations = 1;
  result = solve_checked(model, budget, false);
  require(result.status == "iteration_limit" && result.has_solution &&
              result.solution[0] > 0 && result.solution[0] < 10,
          "预算出口必须返回最后已完成的近端输出及真实未完成状态");

  model.column_lower = {3};
  model.column_upper = {3};
  result = solve_checked(model, options);
  require(result.objective == -3, "固定变量无需通过反射状态进行认证");

  model = {};
  model.column_start = {0};
  model.objective_offset = 7;
  result = solve_checked(model, options);
  require(result.objective == 7, "空矩阵和空变量必须保持目标常数");
}

}  // 匿名命名空间

int main() {
  try {
    for (const std::string mode : {"halpern", "halpern_reflected"}) {
      solver::SolveOptions options;
      options.pdlp.mode = mode;
      options.pdlp.check_interval = 25;
      options.time_limit = 5;
      analytic_tests(options);
      budget_and_domain_tests(options);
    }
    std::cout << "Halpern 锚定与反射独立检查通过，共 " << checks << " 项。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Halpern 锚定与反射检查失败：" << error.what() << '\n';
    return 1;
  }
}
