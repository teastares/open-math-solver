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
std::int64_t acceleration_attempts = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

struct AnalyticCase {
  solver::Model model;
  std::vector<double> primal;
  std::vector<double> dual;
};

AnalyticCase planted_kkt() {
  // 先构造满足互补条件的原始/对偶点，再由 c=A^T y+r 得到成本。
  // 最优点不必唯一；该构造无需另一求解器提供预期答案。
  AnalyticCase sample;
  auto& model = sample.model;
  constexpr std::size_t columns = 9;
  constexpr std::size_t rows = 12;
  sample.primal.resize(columns);
  sample.dual = {1, 0.5, -0.75, 0.25, -0.5, 0, 0, -1, 0.5, 0, 0, 0};
  model.costs.assign(columns, 0);
  model.column_lower.resize(columns);
  model.column_upper.resize(columns);
  model.kinds.assign(columns, solver::VariableKind::continuous);
  model.row_lower.assign(rows, 0);
  model.row_upper.assign(rows, 0);
  model.column_start = {0};
  std::vector<double> reduced(columns, 0);
  for (std::size_t col = 0; col < columns; ++col) {
    if (col % 3 == 0) {
      sample.primal[col] = -1;
      model.column_lower[col] = -1;
      model.column_upper[col] = 2;
      reduced[col] = 0.5;
    } else if (col % 3 == 1) {
      sample.primal[col] = 1;
      model.column_lower[col] = -2;
      model.column_upper[col] = 1;
      reduced[col] = -0.75;
    } else {
      sample.primal[col] = 0.25;
      model.column_lower[col] = -infinity;
      model.column_upper[col] = infinity;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      // 三个额外行与前三行相同，刻意保留秩亏和重复方向。
      const auto effective_row = row % columns;
      const double value = effective_row == col ? 2.0 :
          0.0625 * (static_cast<int>((effective_row + 2) * (col + 3) % 7) - 3);
      model.row_index.push_back(static_cast<std::int64_t>(row));
      model.values.push_back(value);
      model.costs[col] += value * sample.dual[row];
      model.row_lower[row] += value * sample.primal[col];
    }
    model.costs[col] += reduced[col];
    model.column_start.push_back(static_cast<std::int64_t>(model.values.size()));
  }
  model.row_upper = model.row_lower;
  model.row_upper[1] = infinity;
  model.row_lower[2] = -infinity;
  model.row_upper[3] += 1;
  model.row_lower[4] -= 1;
  model.row_lower[5] = -infinity;
  model.row_upper[5] = infinity;
  model.row_lower[6] -= 1;
  model.row_upper[6] += 1;
  return sample;
}

solver::SolveResult solve_and_recheck(const solver::Model& model,
                                    const solver::SolveOptions& options,
                                    bool expect_optimal) {
  auto result = solver::solve_pdlp(model, options);
  require(result.backend == "pdlp", "历史加速不得借用外部数值后端");
  require(result.pdlp_iterations && *result.pdlp_iterations >= 0 &&
              *result.pdlp_iterations <= options.pdlp.max_iterations,
          "子空间候选不能绕过迭代预算");
  require(std::isfinite(result.total_seconds) && result.total_seconds >= 0 &&
              std::isfinite(result.solve_seconds) && result.solve_seconds >= 0 &&
              std::isfinite(result.verification_seconds) && result.verification_seconds >= 0,
          "加速、回退和验算必须提供合法计时");
  require(std::abs(result.total_seconds - result.solve_seconds -
                   result.verification_seconds) <= 1e-8,
          "历史拟合、投影和算子重算不能被遗漏到计时之外");
  if (*result.pdlp_iterations > 0) {
    require(result.pdlp_acceleration_attempts && result.pdlp_acceleration_accepted &&
                result.pdlp_operator_passes,
            "已执行的历史加速必须公布候选与稀疏算子工作量");
    require(*result.pdlp_acceleration_attempts >= 0 &&
                *result.pdlp_acceleration_accepted >= 0 &&
                *result.pdlp_acceleration_accepted <= *result.pdlp_acceleration_attempts &&
                *result.pdlp_operator_passes > 0,
            "历史加速计数必须自洽，额外算子调用不得隐藏");
    acceleration_attempts += *result.pdlp_acceleration_attempts;
  }
  if (result.has_solution) {
    const auto checked = solver::verify_lp_solution(model, result.solution, result.row_dual,
                                                    options.tolerances,
                                                    options.pdlp.optimality);
    require(checked.primal.feasible && checked.primal.objective_consistent &&
                checked.primal.objective == result.objective,
            "任何发布的候选都必须在原输入上独立复核");
    require(result.lp_optimality && checked.optimality.numerically_optimal ==
                result.lp_optimality->numerically_optimal,
            "外推点与其对偶、诊断必须成对返回");
    if (result.best_bound) {
      require(result.best_bound == checked.optimality.corrected_dual_bound,
              "外推估计目标不能代替同一对偶点的修正界");
    }
    if (result.status == "optimal") {
      require(checked.optimality.numerically_optimal &&
                  result.verification_level == "primal_dual_residuals_checked",
              "缩放残差改善不能直接授予最优状态");
    }
  } else {
    require(result.solution.empty() && !result.objective && result.status != "optimal",
            "不存在已验算候选时不得发布最优向量");
  }
  if (expect_optimal) {
    require(result.status == "optimal", "解析回归应收敛：" + result.status + "；" + result.message);
  }
  return result;
}

void verify_analytic_case(const AnalyticCase& sample, const solver::SolveOptions& options) {
  const auto exact = solver::verify_lp_solution(sample.model, sample.primal, sample.dual);
  require(exact.primal.feasible && exact.optimality.numerically_optimal,
          "解析测试构造自身必须满足原始/对偶最优条件");
  const auto result = solve_and_recheck(sample.model, options, true);
  // 排除大目标常数后比较变量部分，避免常数舍入掩盖错误。
  long double difference = 0;
  for (std::size_t col = 0; col < sample.primal.size(); ++col) {
    difference += static_cast<long double>(sample.model.costs[col]) *
                  (result.solution[col] - sample.primal[col]);
  }
  require(std::abs(difference) <= 2e-5L, "返回目标变量部分与解析最优值不符");
}

void analytic_regressions(const solver::SolveOptions& options) {
  auto sample = planted_kkt();
  verify_analytic_case(sample, options);

  sample.model.objective_offset = 1e30;
  verify_analytic_case(sample, options);
  sample.model.sense = solver::ObjectiveSense::maximize;
  for (double& cost : sample.model.costs) cost = -cost;
  verify_analytic_case(sample, options);

  // 正负行缩放检验历史残差在当前坐标中的一致性以及单边对偶符号。
  sample = planted_kkt();
  for (std::size_t row = 0; row < sample.dual.size(); ++row) {
    const double factor = row % 2 == 0 ? -4.0 : 0.25;
    const double lower = factor * sample.model.row_lower[row];
    const double upper = factor * sample.model.row_upper[row];
    sample.model.row_lower[row] = std::min(lower, upper);
    sample.model.row_upper[row] = std::max(lower, upper);
    sample.dual[row] /= factor;
  }
  for (std::size_t k = 0; k < sample.model.values.size(); ++k) {
    sample.model.values[k] *= sample.model.row_index[k] % 2 == 0 ? -4.0 : 0.25;
  }
  verify_analytic_case(sample, options);

  // 变量被固定时残差历史缺秩；高相关历史不应造成奇异求解或非有限外推。
  sample = planted_kkt();
  sample.model.column_lower = sample.primal;
  sample.model.column_upper = sample.primal;
  verify_analytic_case(sample, options);
}

void singular_and_budget_regressions(const solver::SolveOptions& options) {
  solver::Model model;
  model.costs = {0, 0};
  model.column_lower = {0, 0};
  model.column_upper = {1, 1};
  model.kinds.assign(2, solver::VariableKind::continuous);
  model.row_lower = {1, 1, 0};
  model.row_upper = {1, 1, 0};
  model.column_start = {0, 3, 6};
  model.row_index = {0, 1, 2, 0, 1, 2};
  model.values = {1, 1, 0, 1, 1, 0};
  const auto degenerate = solve_and_recheck(model, options, true);
  require(std::abs(degenerate.solution[0] + degenerate.solution[1] - 1) <= 2e-7,
          "零目标和重复等式不能使子空间算法提前误报成功");

  auto budget = options;
  budget.pdlp.max_iterations = 0;
  auto result = solve_and_recheck(model, budget, false);
  require(result.status == "iteration_limit" && !result.has_solution && !result.lp_optimality,
          "零迭代预算不得初始化历史或候选");
  budget = options;
  budget.time_limit = 0;
  result = solve_and_recheck(model, budget, false);
  require(result.status == "time_limit" && !result.has_solution && !result.lp_optimality,
          "零时间预算不得初始化历史或候选");

  budget = options;
  budget.pdlp.max_iterations = 67;
  budget.pdlp.check_interval = 2;
  model.row_lower[1] = 2;
  model.row_upper[1] = 2;
  result = solve_and_recheck(model, budget, false);
  require(!result.has_solution && result.status != "optimal" && result.status != "infeasible",
          "无解模型即使历史残差相消，也不能伪造最优或不可行证书");

  model = {};
  model.costs = {-1};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  result = solve_and_recheck(model, budget, false);
  require(result.status != "optimal" && result.status != "unbounded" && !result.best_bound,
          "恒定残差及秩零历史不能生成无界证书或伪有限界");

  model = {};
  model.column_start = {0};
  model.objective_offset = 3;
  result = solve_and_recheck(model, options, true);
  require(result.objective == 3, "零维历史不能触发非法内存访问");
}

void history_budget_regressions(const solver::SolveOptions& options) {
  const auto sample = planted_kkt();
  auto disabled = options;
  disabled.pdlp.history_size = 0;
  // 固定迭代数隔离机器速度差异，零历史应保留原自适应路径的数值行为。
  disabled.pdlp.max_iterations = 67;
  const auto result = solve_and_recheck(sample.model, disabled, false);
  auto adaptive = disabled;
  adaptive.pdlp.mode = "adaptive";
  const auto reference = solver::solve_pdlp(sample.model, adaptive);
  require(result.status == reference.status && result.pdlp_iterations == reference.pdlp_iterations &&
              result.has_solution == reference.has_solution && result.solution == reference.solution &&
              result.row_dual == reference.row_dual && result.objective == reference.objective,
          "history_size=0 必须与无历史自适应算法等价");
  require(result.pdlp_acceleration_attempts == 0 && result.pdlp_acceleration_accepted == 0,
          "关闭历史后不得继续拟合或接受加速点");

  for (const int invalid_size : {-1, 33}) {
    auto invalid = options;
    invalid.pdlp.history_size = invalid_size;
    const auto rejected = solver::solve_pdlp(sample.model, invalid);
    require(rejected.status == "error" && rejected.pdlp_iterations == 0 && !rejected.has_solution,
            "历史预算越界必须在分配或迭代前拒绝");
  }
}

}  // 匿名命名空间

int main() {
  try {
    solver::SolveOptions options;
    options.pdlp.mode = "subspace";
    options.pdlp.check_interval = 5;
    options.pdlp.max_iterations = 100000;
    options.time_limit = 5;
    analytic_regressions(options);
    singular_and_budget_regressions(options);
    history_budget_regressions(options);
    require(acceleration_attempts > 0, "测试必须实际触发历史候选生成，不能仅覆盖基础迭代");
    std::cout << "历史子空间加速独立防错检查通过，共 " << checks << " 项。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "历史子空间加速防错检查失败：" << error.what() << '\n';
    return 1;
  }
}
