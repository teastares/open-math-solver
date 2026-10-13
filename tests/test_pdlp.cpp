#include "solver/pdlp.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

solver::Model one_variable(double cost, double lower, double upper,
                            double coefficient, double row_lower, double row_upper) {
  solver::Model model;
  model.costs = {cost};
  model.column_lower = {lower};
  model.column_upper = {upper};
  model.kinds = {solver::VariableKind::continuous};
  model.row_lower = {row_lower};
  model.row_upper = {row_upper};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {coefficient};
  return model;
}

void check_optimal(const solver::SolveResult& result, double expected,
                     double objective_tolerance = 1e-5) {
  require(result.backend == "pdlp" && result.backend_version == solver::pdlp_version(),
          "结果必须明确来自自主算法");
  require(result.status == "optimal", "应达到数值最优：" + result.status + "；" + result.message);
  require(result.has_solution && result.objective &&
              std::abs(*result.objective - expected) < objective_tolerance,
          "自主目标值与解析解不符");
  require(result.verification.feasible && result.lp_optimality &&
              result.lp_optimality->numerically_optimal,
          "最优状态必须同时通过独立原始和对偶检查");
  require(result.verification_level == "primal_dual_residuals_checked",
          "浮点残差检查不能宣称精确证明");
  const auto checked = solver::verify_lp_optimality(result.model, result.solution,
                                                    result.row_dual);
  require(checked.numerically_optimal, "返回原始与对偶向量必须属于同一已验证点");
  require(result.best_bound == result.lp_optimality->corrected_dual_bound,
          "只能以修正对偶界作为有效界");
  if (result.best_bound) {
    const double signed_gap = result.model.sense == solver::ObjectiveSense::minimize
        ? *result.objective - *result.best_bound : *result.best_bound - *result.objective;
    require(signed_gap >= -1e-5, "最大化和最小化的界方向必须正确");
  } else {
    require(!result.absolute_gap && !result.relative_gap,
            "不存在有效对偶界时不能把估计对偶目标充当有效间隙");
  }
}

}  // 匿名命名空间

int main() {
  try {
    // min x + 2y，x+y>=3，x<=2；解析解为 (2,1)，目标为4。
    solver::Model lp;
    lp.costs = {1, 2};
    lp.column_lower = {0, 0};
    lp.column_upper = {2, 4};
    lp.row_lower = {3};
    lp.row_upper = {inf};
    lp.kinds = {solver::VariableKind::continuous, solver::VariableKind::continuous};
    lp.column_start = {0, 1, 2};
    lp.row_index = {0, 0};
    lp.values = {1, 1};
    check_optimal(solver::solve_pdlp(lp), 4);

    auto maximum = lp;
    maximum.sense = solver::ObjectiveSense::maximize;
    maximum.objective_offset = 7;
    maximum.costs = {3, 2};
    maximum.row_lower = {-inf};
    maximum.row_upper = {4};
    maximum.column_upper = {2, 3};
    check_optimal(solver::solve_pdlp(maximum), 17);
    solver::SolveOptions baseline;
    baseline.pdlp.mode = "baseline";
    check_optimal(solver::solve_pdlp(lp, baseline), 4);
    check_optimal(solver::solve_pdlp(maximum, baseline), 17);

    // 近相关稠密等式具有不同曲率方向，检验线搜索与重启不会跳过独立验算。
    solver::Model correlated;
    constexpr std::size_t dimension = 12;
    correlated.costs.resize(dimension);
    correlated.column_lower.assign(dimension, -3);
    correlated.column_upper.assign(dimension, 3);
    correlated.row_lower.assign(dimension, dimension + 0.05);
    correlated.row_upper = correlated.row_lower;
    correlated.kinds.assign(dimension, solver::VariableKind::continuous);
    correlated.column_start = {0};
    double correlated_objective = 0;
    for (std::size_t col = 0; col < dimension; ++col) {
      correlated.costs[col] = (col % 2 == 0 ? 1.0 : -1.0) * (col + 1);
      correlated_objective += correlated.costs[col];
      for (std::size_t row = 0; row < dimension; ++row) {
        correlated.row_index.push_back(static_cast<std::int64_t>(row));
        correlated.values.push_back(row == col ? 1.05 : 1.0);
      }
      correlated.column_start.push_back(static_cast<std::int64_t>(correlated.values.size()));
    }
    check_optimal(solver::solve_pdlp(correlated), correlated_objective);

    // 自由变量、等式行、双边行及两种单边行的对偶符号。
    check_optimal(solver::solve_pdlp(one_variable(1, -inf, inf, 1, 2, 2)), 2);
    check_optimal(solver::solve_pdlp(one_variable(1, -inf, inf, 1, 2, 4)), 2);
    check_optimal(solver::solve_pdlp(one_variable(-1, -inf, inf, 1, 2, 4)), -4);
    check_optimal(solver::solve_pdlp(one_variable(1, -inf, inf, 1, 2, inf)), 2);
    check_optimal(solver::solve_pdlp(one_variable(-1, -inf, inf, 1, -inf, 4)), -4);
    check_optimal(solver::solve_pdlp(one_variable(1, -inf, inf, -1, -4, -2)), 2);

    // 行列缩放与目标归一化还原必须同时保持原始和对偶坐标。
    const auto scaled = one_variable(3, 0, 5, 10000, 20000, inf);
    check_optimal(solver::solve_pdlp(scaled), 6);
    auto large_objective = one_variable(10000, 0, 5, 1, 2, inf);
    large_objective.objective_offset = -13;
    check_optimal(solver::solve_pdlp(large_objective), 19987, 1e-3);
    solver::Model tiny_offset;
    tiny_offset.costs = {1e308};
    tiny_offset.column_lower = {0};
    tiny_offset.column_upper = {1};
    tiny_offset.kinds = {solver::VariableKind::continuous};
    tiny_offset.column_start = {0, 0};
    tiny_offset.objective_offset = 1e-100;
    const auto tiny_offset_result = solver::solve_pdlp(tiny_offset);
    check_optimal(tiny_offset_result, 1e-100);
    require(tiny_offset_result.objective == tiny_offset.objective_offset &&
                tiny_offset_result.model.objective_offset == tiny_offset.objective_offset,
            "目标常数不参与迭代缩放，必须在原坐标精确保留");
    solver::SolveOptions unscaled;
    unscaled.pdlp.scaling_passes = 0;
    check_optimal(solver::solve_pdlp(lp, unscaled), 4);

    solver::Model empty;
    empty.column_start = {0};
    empty.objective_offset = 9;
    check_optimal(solver::solve_pdlp(empty), 9);
    auto zero_matrix = one_variable(-1, 0, 3, 0, -1, 1);
    check_optimal(solver::solve_pdlp(zero_matrix), -3);
    auto zero_objective = one_variable(0, -inf, inf, 0, -inf, inf);
    check_optimal(solver::solve_pdlp(zero_objective), 0);
    auto fixed = one_variable(3, 2, 2, 0, -1, 1);
    check_optimal(solver::solve_pdlp(fixed), 6);

    solver::SolveOptions budget;
    budget.pdlp.max_iterations = 1;
    const auto incomplete = solver::solve_pdlp(one_variable(-1, 0, 10, 0, -1, 1), budget);
    require(incomplete.status == "iteration_limit" && incomplete.has_solution &&
                incomplete.objective && *incomplete.objective > -10 &&
                incomplete.best_bound && *incomplete.best_bound <= -10,
            "预算结束可返回可行非最优解，但不得误报最优");
    require(incomplete.pdlp_iterations == 1 && incomplete.lp_optimality &&
                !incomplete.lp_optimality->numerically_optimal,
            "迭代计数和未收敛检查应保持一致");
    const auto incomplete_check = solver::verify_lp_optimality(
        incomplete.model, incomplete.solution, incomplete.row_dual);
    require(incomplete_check.corrected_dual_bound == incomplete.best_bound,
            "预算结果不能混用历史原始解和另一个对偶点");

    budget.pdlp.max_iterations = 0;
    const auto no_iterations = solver::solve_pdlp(lp, budget);
    require(no_iterations.status == "iteration_limit" && !no_iterations.has_solution &&
                !no_iterations.objective && !no_iterations.best_bound &&
                !no_iterations.lp_optimality, "零迭代预算不能伪造结果");
    budget.time_limit = 0;
    const auto no_time = solver::solve_pdlp(lp, budget);
    require(no_time.status == "time_limit" && !no_time.has_solution &&
                !no_time.objective && !no_time.best_bound, "零时间预算不能伪造结果");
    budget.time_limit = 60;
    budget.pdlp.max_iterations = 200;
    const auto infeasible = solver::solve_pdlp(one_variable(1, 0, 1, 1, 2, inf), budget);
    require(infeasible.status == "iteration_limit" && !infeasible.has_solution,
            "无不可行证书时不能宣称不可行");
    const auto unbounded = solver::solve_pdlp(one_variable(-1, 0, inf, 0, -1, 1), budget);
    require(unbounded.status == "iteration_limit" && unbounded.has_solution,
            "无无界证书时不能宣称无界");
    const auto impossible_zero_row = solver::solve_pdlp(one_variable(0, 0, 1, 0, 1, inf), budget);
    require(impossible_zero_row.status == "iteration_limit" && !impossible_zero_row.has_solution,
            "矛盾零行不能被当作空约束忽略");

    auto mip = lp;
    mip.kinds[0] = solver::VariableKind::integer;
    require(solver::solve_pdlp(mip).status == "error", "首版必须拒绝整数模型");
    solver::SolveOptions threads;
    threads.threads = 2;
    require(solver::solve_pdlp(lp, threads).status == "error", "首版必须明确拒绝多线程设置");
    solver::SolveOptions invalid;
    invalid.pdlp.check_interval = 0;
    require(solver::solve_pdlp(lp, invalid).status == "error", "检查间隔必须为正数");
    invalid = {};
    invalid.pdlp.optimality.gap_absolute = -1;
    require(solver::solve_pdlp(lp, invalid).status == "error", "必须检查最优性容差合法性");
    invalid = {};
    invalid.pdlp.scaling_passes = 21;
    require(solver::solve_pdlp(lp, invalid).status == "error", "缩放轮数不能超过接口约定上限");
    invalid = {};
    invalid.pdlp.mode = "unknown";
    require(solver::solve_pdlp(lp, invalid).status == "error", "必须拒绝未知算法模式");

    auto extreme = one_variable(std::numeric_limits<double>::denorm_min(), 0, 1,
                                1e300, 0, inf);
    require(solver::solve_pdlp(extreme).status == "numerical_error",
            "缩放不能静默丢失有限非零系数");
    std::cout << "自主 PDHG 集成检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "自主 PDHG 集成检查失败：" << error.what() << '\n';
    return 1;
  }
}
