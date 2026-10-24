#include "solver/ipm.hpp"
#include "solver/detail/ipm_centrality.hpp"
#include "solver/detail/ipm_full_newton.hpp"
#include "solver/detail/ipm_homogeneous.hpp"
#include "solver/detail/ipm_presolve.hpp"
#include "solver/detail/ipm_recycling.hpp"
#include "solver/detail/ipm_residual.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr double inf = std::numeric_limits<double>::infinity();
int checks = 0;
std::int64_t backward_acceptances = 0;
std::int64_t inexact_acceptances = 0;
std::int64_t prefilter_rejections = 0;
bool prefilter_available = false;
std::int64_t central_attempts = 0, central_acceptances = 0;
std::int64_t recycle_vectors = 0, recycle_projections = 0, recycle_acceptances = 0;
std::int64_t full_newton_test_refinements = 0, full_newton_test_krylov = 0;
int full_newton_test_products = 0, full_newton_test_preconditions = 0;
std::int64_t bounded_newton_test_refinements = 0;
int bounded_newton_test_products = 0, bounded_newton_test_preconditions = 0;
std::int64_t homogeneous_test_constant_solves = 0, homogeneous_test_direction_solves = 0;
int homogeneous_repair_test_attempts = 0, homogeneous_repair_test_successes = 0;

void require(bool condition, const std::string& message) {
  ++checks;
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

solver::SolveResult optimal_mode(const solver::Model& model, double expected, const solver::SolveOptions& options) {
  const auto result = solver::solve_ipm(model, options);
  require(result.backend == "ipm" && result.backend_version == solver::ipm_version(), "结果必须标识自主内点法");
  require(result.status == "optimal", "应达到最优：" + result.status + "；" + result.message +
          "；线性求解器=" + options.ipm.linear_solver);
  require(result.has_solution && result.objective && std::abs(*result.objective - expected) <=
      1e-6 * std::max(1.0, std::abs(expected)), "目标与解析解不符");
  require(result.ipm_iterations && result.ipm_factorizations && result.ipm_refinement_steps,
          "内点法必须报告迭代、分解和迭代改进计数");
  require(result.ipm_homogeneous_constant_solves && result.ipm_homogeneous_direction_solves &&
              *result.ipm_homogeneous_constant_solves >= 0 && *result.ipm_homogeneous_direction_solves >= 0,
          "IPM 必须初始化齐次常数和方向求解启动计数");
  require(result.ipm_homogeneous_constant_repairs && result.ipm_homogeneous_constant_repair_successes &&
              *result.ipm_homogeneous_constant_repairs >= 0 && *result.ipm_homogeneous_constant_repair_successes >= 0 &&
              *result.ipm_homogeneous_constant_repair_successes <= *result.ipm_homogeneous_constant_repairs &&
              *result.ipm_homogeneous_constant_repairs <= *result.ipm_homogeneous_constant_solves,
          "齐次常数恢复次数必须初始化，成功属于尝试，尝试属于真实常数求解");
  if (options.ipm.linear_accuracy != "homogeneous_repair") {
    require(*result.ipm_homogeneous_constant_repairs == 0 &&
                *result.ipm_homogeneous_constant_repair_successes == 0,
            "旧精度模式不能伪造齐次常数恢复工作");
  }
  require(result.ipm_homogeneous_scaled_products && result.ipm_homogeneous_scaled_refinement_steps &&
              result.ipm_homogeneous_scaled_inexact_returns && *result.ipm_homogeneous_scaled_products >= 0 &&
              *result.ipm_homogeneous_scaled_refinement_steps >= 0 &&
              *result.ipm_homogeneous_scaled_inexact_returns >= 0 &&
              *result.ipm_homogeneous_scaled_refinement_steps <= *result.ipm_homogeneous_direction_solves &&
              *result.ipm_homogeneous_scaled_refinement_steps <= *result.ipm_refinement_steps &&
              *result.ipm_homogeneous_scaled_inexact_returns <= *result.ipm_homogeneous_scaled_products,
          "齐次缩放工作须初始化，IR 属于实际方向和总改进，非精确返回须经过真实乘积");
  if (options.ipm.linear_accuracy != "homogeneous_scaled") {
    require(*result.ipm_homogeneous_scaled_products == 0 &&
                *result.ipm_homogeneous_scaled_refinement_steps == 0 &&
                *result.ipm_homogeneous_scaled_inexact_returns == 0,
            "旧精度模式不得产生齐次缩放工作");
  }
  if (options.ipm.method == "homogeneous") {
    require(*result.ipm_homogeneous_constant_solves ==
                *result.ipm_iterations + *result.ipm_homogeneous_constant_repairs &&
                *result.ipm_homogeneous_constant_repairs <= *result.ipm_iterations,
            "成功的齐次完整迭代每次求一个常数右端，并最多追加一次有账本的恢复");
    if (options.ipm.linear_accuracy == "homogeneous_scaled") {
      require(*result.ipm_homogeneous_direction_solves >= 2 * *result.ipm_iterations &&
                  *result.ipm_homogeneous_direction_solves <= 8 * *result.ipm_iterations &&
                  *result.ipm_homogeneous_scaled_products >= 2 * *result.ipm_iterations &&
                  *result.ipm_homogeneous_scaled_products <= *result.ipm_homogeneous_direction_solves &&
                  *result.ipm_homogeneous_scaled_inexact_returns <= 2 * *result.ipm_iterations,
              "缩放预测校正每次至少两次直接解和真乘积，每方向最多增加三次 IR");
    } else {
      require(*result.ipm_homogeneous_direction_solves == 2 * *result.ipm_iterations,
              "严格齐次完整迭代必须每次求两个方向右端");
    }
    require(result.ipm_homogeneous_tau.has_value() == result.ipm_homogeneous_kappa.has_value(),
            "齐次标量报告必须成对出现");
    if (result.ipm_homogeneous_tau) {
      require(*result.ipm_homogeneous_tau > 0 && std::isfinite(*result.ipm_homogeneous_tau) &&
                  *result.ipm_homogeneous_kappa > 0 && std::isfinite(*result.ipm_homogeneous_kappa),
              "已初始化的齐次标量必须为有限正数");
    } else {
      require(*result.ipm_iterations == 0 && *result.ipm_factorizations == 0,
              "仅初始化前的直接原模型验算成功可以不报告齐次标量");
    }
    homogeneous_test_constant_solves += *result.ipm_homogeneous_constant_solves;
    homogeneous_test_direction_solves += *result.ipm_homogeneous_direction_solves;
  } else {
    require(*result.ipm_homogeneous_constant_solves == 0 && *result.ipm_homogeneous_direction_solves == 0 &&
                !result.ipm_homogeneous_tau && !result.ipm_homogeneous_kappa,
            "普通预测校正路径不得生成齐次工作或标量");
  }
  require(result.ipm_presolved_singleton_rows && *result.ipm_presolved_singleton_rows >= 0 &&
              *result.ipm_presolved_singleton_rows <= static_cast<std::int64_t>(model.row_lower.size()),
          "归并计数必须对应原模型行而非锥行或分解维数");
  if (!options.ipm.presolve_singleton_rows) {
    require(*result.ipm_presolved_singleton_rows == 0, "关闭输入归并时计数必须为零");
  }
  require(result.ipm_presolved_fixed_columns && *result.ipm_presolved_fixed_columns >= 0 &&
              *result.ipm_presolved_fixed_columns <= *result.ipm_eliminated_fixed_variables,
          "归并产生的消列数必须属于全部实际消列数");
  if (!options.ipm.eliminate_presolved_fixed_variables) {
    require(*result.ipm_presolved_fixed_columns == 0, "关闭归并固定消元时不能报告新消列");
  }
  require(result.ipm_krylov_recycle_projections && result.ipm_krylov_recycle_acceptances &&
              result.ipm_krylov_recycle_vectors && result.ipm_krylov_recycle_peak_dimension &&
              *result.ipm_krylov_recycle_projections >= *result.ipm_krylov_recycle_acceptances &&
              *result.ipm_krylov_recycle_acceptances >= 0 &&
              *result.ipm_krylov_recycle_vectors >= *result.ipm_krylov_recycle_peak_dimension &&
              *result.ipm_krylov_recycle_peak_dimension >= 0 &&
              *result.ipm_krylov_recycle_peak_dimension <= options.ipm.krylov_history_size,
          "回收尝试、直接接受和入池数必须有明确关系，存储维数不得超过请求预算");
  if (options.ipm.krylov_history_size == 0) {
    require(*result.ipm_krylov_recycle_projections == 0 && *result.ipm_krylov_recycle_acceptances == 0 &&
                *result.ipm_krylov_recycle_vectors == 0 && *result.ipm_krylov_recycle_peak_dimension == 0,
            "关闭回收时所有新工作量计数必须为零");
  } else {
    recycle_vectors += *result.ipm_krylov_recycle_vectors;
    recycle_projections += *result.ipm_krylov_recycle_projections;
    recycle_acceptances += *result.ipm_krylov_recycle_acceptances;
  }
  if (*result.ipm_factorizations > 0) {
    require(result.ipm_primal_regularization && result.ipm_dual_regularization &&
                std::isfinite(*result.ipm_primal_regularization) && *result.ipm_primal_regularization >= 0 &&
                std::isfinite(*result.ipm_dual_regularization) && *result.ipm_dual_regularization > 0,
            "已启动分解须报告最后使用的两个外部正则参数");
    const double ratio = options.ipm.linear_accuracy == "primal_dual_regularization" ?
        1 / std::pow(std::numeric_limits<double>::epsilon(), 0.25) : 1;
    if (*result.ipm_primal_regularization == 0) {
      require(options.ipm.linear_accuracy == "curvature_regularization" &&
                  *result.ipm_dual_regularization == solver::detail::ipm_regularization_at_level(0) &&
                  result.ipm_unshifted_factorizations && *result.ipm_unshifted_factorizations > 0,
              "零原始偏移必须来自有记录的曲率尝试，对偶偏移仍为旧起点");
    } else {
      require(*result.ipm_dual_regularization == *result.ipm_primal_regularization * ratio,
              "外部参数必须与请求的成对规则一致，不能混入内核主元修正");
    }
  } else {
    require(!result.ipm_primal_regularization && !result.ipm_dual_regularization,
            "没有启动分解时不能伪造外部正则参数");
  }
  require(result.ipm_unshifted_factorizations && *result.ipm_unshifted_factorizations >= 0 &&
              *result.ipm_unshifted_factorizations <= *result.ipm_factorizations,
          "曲率尝试须包含在总分解成本中");
  require(result.ipm_direction_retries && result.ipm_direction_retry_successes &&
              *result.ipm_direction_retries >= *result.ipm_direction_retry_successes &&
              *result.ipm_direction_retry_successes >= 0 &&
              *result.ipm_direction_retries <= options.ipm.direction_retries * (*result.ipm_iterations + 1) &&
              *result.ipm_direction_retry_successes <= *result.ipm_iterations + 1,
          "成功解的方向重试计数须受逐组预算限制，恢复组数不超过重试次数");
  if (options.ipm.direction_retries == 0) {
    require(*result.ipm_direction_retries == 0 && *result.ipm_direction_retry_successes == 0,
            "关闭重试时不能增加重分解请求或恢复计数");
  }
  if (options.ipm.linear_accuracy != "curvature_regularization") {
    require(*result.ipm_unshifted_factorizations == 0, "其他模式不能触发曲率尝试");
  } else {
    require(*result.ipm_unshifted_factorizations <= *result.ipm_iterations + 1 &&
                *result.ipm_factorizations <= 9 * (*result.ipm_iterations + 1),
            "初始化及每个外层至多增加一次分解，保留旧八级重试预算");
  }
  require(result.ipm_weighted_krylov_steps && *result.ipm_weighted_krylov_steps >= 0 &&
              *result.ipm_weighted_krylov_steps <= *result.ipm_refinement_steps,
          "加权 Krylov 步数必须是总改进步数的子集");
  if (options.ipm.linear_accuracy != "blockwise_weighted" &&
      options.ipm.linear_accuracy != "blockwise_forcing_weighted" && options.ipm.linear_accuracy != "full_newton") {
    require(*result.ipm_weighted_krylov_steps == 0, "未启用加权模式不能记录加权步骤");
  }
  require(result.ipm_full_newton_preconditioner_solves && result.ipm_full_newton_products &&
              *result.ipm_full_newton_preconditioner_solves >= 0 && *result.ipm_full_newton_products >= 0,
          "IPM 必须初始化完整 Newton 实际预条件与乘积计数");
  if (options.ipm.linear_accuracy != "full_newton" && options.ipm.linear_accuracy != "bounded_newton") {
    require(*result.ipm_full_newton_preconditioner_solves == 0 && *result.ipm_full_newton_products == 0,
            "旧精度模式不得发生完整 Newton 计算");
  } else {
    require(*result.ipm_full_newton_preconditioner_solves >= 2 * *result.ipm_iterations &&
                *result.ipm_full_newton_products >= 2 * *result.ipm_iterations,
            "每个完成的预测校正组至少包含两次实际预条件和真实完整算子验收");
  }
  require(result.ipm_bounded_newton_returns && result.ipm_bounded_newton_inexact_returns &&
              *result.ipm_bounded_newton_returns >= *result.ipm_bounded_newton_inexact_returns &&
              *result.ipm_bounded_newton_inexact_returns >= 0,
          "有界 Newton 非精确返回必须是全部返回的子集");
  if (options.ipm.linear_accuracy == "bounded_newton") {
    require(*result.ipm_bounded_newton_returns == 2 * *result.ipm_iterations &&
                *result.ipm_full_newton_preconditioner_solves <= 4 * *result.ipm_bounded_newton_returns &&
                *result.ipm_full_newton_products <= 4 * *result.ipm_bounded_newton_returns &&
                *result.ipm_weighted_krylov_steps == 0,
            "成功的有界模式每组只返回预测和校正方向，每方向至多初始解加三次 IR");
  } else {
    require(*result.ipm_bounded_newton_returns == 0 && *result.ipm_bounded_newton_inexact_returns == 0,
            "旧模式不得记录有界 Newton 返回");
  }
  require(result.ipm_regularization_searches && result.ipm_regularization_reductions &&
              *result.ipm_regularization_reductions >= 0 &&
              *result.ipm_regularization_reductions <= *result.ipm_regularization_searches &&
              *result.ipm_regularization_searches <= *result.ipm_iterations + 1,
          "降正则计数必须属于搜索次数，且每个外层最多搜索一次");
  if (options.ipm.linear_accuracy != "adaptive_regularization") {
    require(*result.ipm_regularization_searches == 0, "关闭自适应正则模式不得记录搜索");
  }
  require(result.ipm_schur_residual_products && result.ipm_schur_refinement_steps &&
              result.ipm_schur_refinement_acceptances && *result.ipm_schur_refinement_acceptances >= 0 &&
              *result.ipm_schur_refinement_acceptances <= *result.ipm_schur_refinement_steps &&
              *result.ipm_schur_refinement_steps <= 2 * *result.ipm_schur_residual_products,
          "Schur 内部工作必须独立报告且接受次数不超过尝试");
  if (result.ipm_factor_backend != "normal_refined") {
    require(*result.ipm_schur_residual_products == 0 && *result.ipm_schur_refinement_steps == 0 &&
                *result.ipm_schur_refinement_acceptances == 0, "其他内核不得记录 Schur 校正工作");
  }
  require(result.ipm_regularized_pivots && *result.ipm_regularized_pivots >= 0,
          "内点法必须报告局部主元修正计数");
  if (!options.ipm.dynamic_regularization && result.ipm_factor_backend != "factorhighs" && result.ipm_factor_backend != "normal_multifrontal") {
    require(*result.ipm_regularized_pivots == 0, "关闭动态正则时不得记录局部主元修正");
  }
  require(result.ipm_backward_error_acceptances && *result.ipm_backward_error_acceptances >= 0,
          "内点法必须报告后向误差替代接受次数");
  if (options.ipm.linear_accuracy == "strict") {
    require(*result.ipm_backward_error_acceptances == 0, "严格模式不得记录后向误差替代接受");
  } else {
    backward_acceptances += *result.ipm_backward_error_acceptances;
  }
  require(result.ipm_inexact_acceptances && *result.ipm_inexact_acceptances >= 0,
          "内点法必须报告分块非精确方向接受次数");
  if (options.ipm.linear_accuracy != "inexact") {
    require(*result.ipm_inexact_acceptances == 0, "关闭分块 forcing 必须保持零替代接受");
  } else {
    inexact_acceptances += *result.ipm_inexact_acceptances;
    require(*result.ipm_backward_error_acceptances == 0, "分块 forcing 不得混入后向误差验收");
  }
  require(result.ipm_inexact_prefilter_rejections && *result.ipm_inexact_prefilter_rejections == 0,
          "关闭预拒绝不得报告省略 actual helper");
  if (options.ipm.linear_accuracy == "inexact") {
    auto filtered_options = options;
    filtered_options.ipm.inexact_prefilter = true;
    const auto filtered = solver::solve_ipm(model, filtered_options);
    require(filtered.status == result.status && filtered.message == result.message &&
                filtered.solution == result.solution && filtered.row_dual == result.row_dual &&
                filtered.objective == result.objective && filtered.best_bound == result.best_bound &&
                filtered.absolute_gap == result.absolute_gap && filtered.relative_gap == result.relative_gap &&
                filtered.ipm_iterations == result.ipm_iterations &&
                filtered.ipm_factorizations == result.ipm_factorizations &&
                filtered.ipm_refinement_steps == result.ipm_refinement_steps &&
                filtered.ipm_regularized_pivots == result.ipm_regularized_pivots &&
                filtered.ipm_inexact_acceptances == result.ipm_inexact_acceptances &&
                filtered.ipm_backward_error_acceptances == result.ipm_backward_error_acceptances &&
                filtered.ipm_kkt_dimension == result.ipm_kkt_dimension &&
                filtered.ipm_factor_nonzeros == result.ipm_factor_nonzeros &&
                filtered.ipm_eliminated_singletons == result.ipm_eliminated_singletons,
            "预拒绝开关不得改变数值结果、原方向接受或迭代轨迹");
    require(filtered.ipm_inexact_prefilter_rejections && *filtered.ipm_inexact_prefilter_rejections >= 0,
            "预拒绝应单独计数且非负");
    prefilter_rejections += *filtered.ipm_inexact_prefilter_rejections;
  }
  require(result.ipm_central_correction_attempts && result.ipm_central_correction_acceptances &&
              *result.ipm_central_correction_attempts >= *result.ipm_central_correction_acceptances &&
              *result.ipm_central_correction_acceptances >= 0,
          "中心校正必须报告有效的尝试与接受计数");
  require(*result.ipm_central_correction_attempts <= *result.ipm_iterations,
          "每次完成的外层迭代至多求解一次额外中心校正");
  if (options.ipm.central_corrections == 0) {
    require(*result.ipm_central_correction_attempts == 0 && *result.ipm_central_correction_acceptances == 0,
            "关闭中心校正必须保持零尝试和零接受");
  } else {
    central_attempts += *result.ipm_central_correction_attempts;
    central_acceptances += *result.ipm_central_correction_acceptances;
  }
  require(result.verification_level == "primal_dual_residuals_checked", "浮点检查不能宣称精确证明");
  const auto check = solver::verify_lp_solution(model, result.solution, result.row_dual,
                                                options.tolerances, options.pdlp.optimality);
  require(check.primal.feasible && check.optimality.numerically_optimal,
          "返回原始和对偶必须属于通过独立检查的同一个点");
  if (*result.ipm_factorizations > 0) {
    require(result.ipm_factor_backend.has_value() && result.ipm_structural_fallbacks.has_value(),
            "实际分解必须报告选中的内核与结构切换次数");
    if (options.ipm.linear_solver == "normal_cholmod" || options.ipm.linear_solver == "normal_cholmod_aat") {
      require(result.ipm_factor_backend == (*result.ipm_structural_fallbacks > 0 ? "factorhighs" : options.ipm.linear_solver),
              "CHOLMOD 请求必须报告实际内核，仅结构预算切换允许报告增广分解");
    }
    require(result.ipm_kkt_dimension && result.ipm_factor_nonzeros && result.ipm_eliminated_singletons,
            "实际分解必须报告维数、因子非零数和消元行数");
    require(*result.ipm_factor_nonzeros >= *result.ipm_kkt_dimension,
            "因子非零数应包含全部对角条目");
    if (options.ipm.linear_solver == "sparselu") {
      require(*result.ipm_factor_nonzeros >= 2 * *result.ipm_kkt_dimension,
              "LU 逻辑非零和应包含 L 和 U 的各自对角");
    }
    if (!options.ipm.eliminate_singletons) {
      require(*result.ipm_eliminated_singletons == 0, "关闭消元必须保留完整 KKT 行");
    }
  }
  return result;
}

void optimal(const solver::Model& model, double expected, const solver::SolveOptions& options = {}) {
  for (const std::string linear_solver : {"qdldl", "sparselu"}) {
    for (bool elimination : {false, true}) {
      for (bool dynamic : {false, true}) {
        if (linear_solver == "sparselu" && dynamic) continue;
        for (bool equilibration : {false, true}) {
          for (const auto& accuracy : {"strict", "backward"}) {
            for (int central : {0, 1}) {
              auto variant = options;
              variant.ipm.eliminate_singletons = elimination;
              variant.ipm.dynamic_regularization = dynamic;
              variant.ipm.equilibrate_kkt = equilibration;
              variant.ipm.linear_accuracy = accuracy;
              variant.ipm.central_corrections = central;
              variant.ipm.linear_solver = linear_solver;
              optimal_mode(model, expected, variant);
            }
          }
        }
      }
    }
  }
  // 新门槛只增三组有代表性的路径，避免重复扩张所有实验开关的笛卡尔积。
  for (int configuration : {0, 1, 2}) {
    auto variant = options;
    variant.ipm.linear_accuracy = "inexact";
    variant.ipm.equilibrate_kkt = configuration == 1;
    variant.ipm.eliminate_singletons = configuration == 2;
    optimal_mode(model, expected, variant);
  }
  for (bool elimination : {false, true}) {
    auto variant = options;
    variant.ipm.linear_solver = "factorhighs";
    variant.ipm.eliminate_singletons = elimination;
    variant.ipm.equilibrate_kkt = elimination;
    optimal_mode(model, expected, variant);
  }
  // 正规方程只比较严格验收下的两种尺度路径，固定开启已定义的单行消元。
  for (const std::string linear : {"normal", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined"}) {
    if ((linear == "normal_extended" || linear == "normal_refined") && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    for (bool equilibration : {false, true}) {
      auto variant = options;
      variant.ipm.linear_solver = linear;
      variant.ipm.eliminate_singletons = true;
      variant.ipm.equilibrate_kkt = equilibration;
      optimal_mode(model, expected, variant);
    }
  }
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    for (bool elimination : {false, true}) {
      for (bool equilibration : {false, true}) {
        auto variant = options;
        variant.ipm.linear_solver = "qdldl_extended";
        variant.ipm.eliminate_singletons = elimination;
        variant.ipm.equilibrate_kkt = equilibration;
        optimal_mode(model, expected, variant);
      }
    }
  }
}

void check_full_newton_algebra() {
  using namespace solver::detail;
  // 一个不等式加一个等式，等式乘子为零也不能参与互补除法。
  const auto layout = make_ipm_full_newton_layout(2, {true, false});
  const std::vector<std::int64_t> starts = {0, 2, 4}, rows = {0, 1, 0, 1};
  const std::vector<double> values = {1, 3, 2, -1}, slack = {2, 0}, dual = {3, 0};
  const std::vector<double> exact = {-1, 2, 0.5, -1, 3};
  const std::vector<double> rhs = {-2.5, 2, 6, -5, 10};
  require(layout.dimension() == 5 && layout.inequality_rows == std::vector<std::size_t>{0},
          "完整方向仅为不等式增加松弛变量，等式不增加空互补行");
  std::vector<long double> image;
  ipm_full_newton_product(layout, starts, rows, values, slack, dual, exact, image);
  require(image == std::vector<long double>({-2.5L, 2, 6, -5, 10}),
          "真实 J 乘积必须符合独立手算的三块方程");
  const auto reduced_rhs = ipm_full_newton_reduced_rhs(layout, dual, rhs);
  require(reduced_rhs[0] == -2.5 && reduced_rhs[1] == 2 &&
              std::abs(reduced_rhs[2] - 8.0 / 3) < 1e-15 && reduced_rhs[3] == -5,
          "预条件 RHS 使用 qp−qc/z，等式行原样保留");
  const std::vector<double> reduced_direction(exact.begin(), exact.begin() + 4);
  require(ipm_full_newton_recover(layout, slack, dual, rhs, reduced_direction) == exact,
          "恢复已求得的物理 dz 后应得到手算 ds，不得使用正则 H 替代 S/Z");
  const auto tolerances = ipm_full_newton_tolerances(layout, rhs);
  require(tolerances == std::array<double, 3>{1e-12 + 1e-10 * 2.5, 1e-12 + 1e-10 * 6, 1e-12 + 1e-10 * 10},
          "三块门槛来自原始完整右端，而不是互补消元后的大右端");
  const auto equality = make_ipm_full_newton_layout(1, {false});
  const std::vector<double> equality_rhs = {2, -3}, equality_dual = {0};
  require(ipm_full_newton_reduced_rhs(equality, equality_dual, equality_rhs) == equality_rhs &&
              ipm_full_newton_tolerances(equality, equality_rhs)[2] == 1e-12,
          "全部等式的空互补块必须有定义且不发生除零");
  const auto exact_preconditioner = [&](std::span<const double> input) {
    const auto reduced = ipm_full_newton_reduced_rhs(layout, dual, input);
    const double dz0 = (reduced[0] + 3 * reduced[1]) / 7;
    const double dz1 = (2 * reduced[0] - reduced[1]) / 7;
    const double lower0 = reduced[2] + (2.0 / 3) * dz0, lower1 = reduced[3];
    const std::vector<double> solution = {(lower0 + 2 * lower1) / 7,
                                        (3 * lower0 - lower1) / 7, dz0, dz1};
    return ipm_full_newton_recover(layout, slack, dual, input, solution);
  };
  int products = 0, preconditions = 0;
  const auto multiply = [&](std::span<const double> input, std::vector<long double>& output) {
    ++products;
    ipm_full_newton_product(layout, starts, rows, values, slack, dual, input, output);
  };
  IpmFullNewtonWork work;
  const auto solved = solve_ipm_full_newton(layout, rhs, [&](std::span<const double> input) {
    ++preconditions;
    return exact_preconditioner(input);
  }, multiply, work);
  for (std::size_t i = 0; i < exact.size(); ++i)
    require(std::abs(solved[i] - exact[i]) < 1e-12, "精确预条件器必须恢复已知完整方向");
  require(products == 1 && preconditions == 1 && work.refinement_steps == 0 && work.krylov_steps == 0,
          "初始方向真实达标时只需一次预条件和一次 J 验算");
  products = preconditions = 0;
  work = {};
  const auto refined = solve_ipm_full_newton(layout, rhs, [&](std::span<const double> input) {
    ++preconditions;
    auto answer = exact_preconditioner(input);
    for (double& value : answer) value *= 0.5;
    return answer;
  }, multiply, work);
  ipm_full_newton_product(layout, starts, rows, values, slack, dual, refined, image);
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    const std::size_t block = i < 2 ? 0 : i < 4 ? 1 : 2;
    require(std::abs(image[i] - rhs[i]) <= tolerances[block],
            "近似预条件器后的 FGMRES 必须按物理三块真实残差接受");
  }
  require(work.krylov_steps > 0 && work.krylov_steps <= 3 * 4 && work.refinement_steps <= 2 + 3 * 4 &&
              preconditions == 1 + work.refinement_steps && products > preconditions,
          "近似预条件例必须进入 Krylov，且保持旧 KKT 维数的总预算并真实验算候选");
  full_newton_test_refinements = work.refinement_steps;
  full_newton_test_krylov = work.krylov_steps;
  full_newton_test_products = products;
  full_newton_test_preconditions = preconditions;
  bool rejected = false;
  work = {};
  try {
    solve_ipm_full_newton(layout, rhs,
        [](std::span<const double> input) { return std::vector<double>(input.begin(), input.end()); },
        [&](std::span<const double>, std::vector<long double>& output) { output.assign(layout.dimension(), 0); }, work);
  } catch (const std::runtime_error&) { rejected = true; }
  require(rejected && work.krylov_steps > 0 && work.refinement_steps <= 14,
          "不相容零算子发生精确 Arnoldi 崩解时不能凭零 Hessenberg 估计接受错误方向");
  struct TestBudgetExpired {};
  bool expired = false, interrupted = false;
  work = {};
  try {
    solve_ipm_full_newton(layout, rhs, exact_preconditioner,
        [&](std::span<const double> input, std::vector<long double>& output) {
          multiply(input, output);
          expired = true;
        }, work, [&] { if (expired) throw TestBudgetExpired{}; });
  } catch (const TestBudgetExpired&) { interrupted = true; }
  require(interrupted, "即使方向已精确，算子计算期间耗尽预算也必须在验收前传播中断");
  rejected = false;
  try {
    const std::vector<double> zero_dual = {0, 0};
    ipm_full_newton_reduced_rhs(layout, zero_dual, rhs);
  } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "零不等式乘子必须拒绝，不得通过补任意地板定义预条件器");
  rejected = false;
  try {
    auto bad = rhs;
    bad[0] = inf;
    ipm_full_newton_tolerances(layout, bad);
  } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "非有限右端不得形成无限门槛放行错误方向");
}

void check_bounded_newton_algebra() {
  using namespace solver::detail;
  // G=1、s=4、z=2，真实 J(dx,dz,ds)=(dz,dx+ds,4dz+2ds)。
  const auto layout = make_ipm_full_newton_layout(1, {true});
  const std::vector<std::int64_t> starts = {0, 1}, rows = {0};
  const std::vector<double> values = {1}, slack = {4}, dual = {2};
  const std::vector<double> rhs = {2, 3, 6}, exact = {4, 2, -1};
  const auto tolerances = ipm_full_newton_tolerances(layout, rhs);
  const auto inverse = [](std::span<const double> input) {
    const double dz = input[0], ds = (input[2] - 4 * dz) / 2;
    return std::vector<double>{input[1] - ds, dz, ds};
  };
  int products = 0, preconditions = 0;
  const auto multiply = [&](std::span<const double> input, std::vector<long double>& output) {
    ++products;
    ipm_full_newton_product(layout, starts, rows, values, slack, dual, input, output);
  };
  const auto scaled_inverse = [&](std::span<const double> input, double scale) {
    ++preconditions;
    auto answer = inverse(input);
    for (double& value : answer) value *= scale;
    return answer;
  };
  IpmFullNewtonWork work;
  IpmBoundedNewtonInfo info;
  const auto solved = solve_ipm_bounded_newton(layout, rhs,
      [&](std::span<const double> input) { return scaled_inverse(input, 1); }, multiply, work, info);
  require(solved == exact && info.measure == 0 && info.tolerance_satisfied &&
              products == 1 && preconditions == 1 && work.refinement_steps == 0 && work.krylov_steps == 0,
          "有界模式初始精确方向应真实达标且不启动 IR 或 Krylov");
  products = preconditions = 0;
  work = {};
  const auto refined = solve_ipm_bounded_newton(layout, rhs,
      [&](std::span<const double> input) { return scaled_inverse(input, 0.5); }, multiply, work, info);
  long double expected_measure = 0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    require(refined[i] == exact[i] * (15.0 / 16), "三次半精度预条件改进应将初始半解推进到十六分之十五解");
    expected_measure = std::max(expected_measure, std::abs(static_cast<long double>(rhs[i]) / 16) / tolerances[i]);
  }
  require(info.measure == expected_measure && info.measure > 1 && !info.tolerance_satisfied &&
              products == 4 && preconditions == 4 && work.refinement_steps == 3 && work.krylov_steps == 0,
          "有界模式必须真实执行三次改进，报告未达标且不增补 FGMRES");
  bounded_newton_test_refinements = work.refinement_steps;
  bounded_newton_test_products = products;
  bounded_newton_test_preconditions = preconditions;
  products = preconditions = 0;
  work = {};
  const auto retained = solve_ipm_bounded_newton(layout, rhs,
      [&](std::span<const double> input) { return scaled_inverse(input, preconditions < 2 ? 0.5 : -1); },
      multiply, work, info);
  for (std::size_t i = 0; i < exact.size(); ++i)
    require(retained[i] == exact[i] * 0.75, "真实残差变差时必须返回上一最佳方向，不能返回最后试算方向");
  require(products == 3 && preconditions == 3 && work.refinement_steps == 2 &&
              info.measure == 4 * expected_measure && !info.tolerance_satisfied,
          "变差候选只计实际工作，返回 measure 必须属于保留方向");
  products = preconditions = 0;
  work = {};
  const auto stagnant = solve_ipm_bounded_newton(layout, rhs,
      [&](std::span<const double> input) { return scaled_inverse(input, preconditions == 0 ? 0.5 : 0); },
      multiply, work, info);
  require(stagnant[0] == exact[0] * 0.5 && preconditions == 2 && products == 2 && work.refinement_steps == 1,
          "残差相等也须结束改进，不能耗尽剩余预算后才返回");
  for (bool bad_product : {false, true}) {
    products = preconditions = 0;
    work = {};
    const auto kept = solve_ipm_bounded_newton(layout, rhs,
        [&](std::span<const double> input) {
          auto answer = scaled_inverse(input, 0.5);
          if (!bad_product && preconditions > 1) answer[0] = inf;
          return answer;
        }, [&](std::span<const double> input, std::vector<long double>& output) {
          multiply(input, output);
          if (bad_product && products > 1) output[0] = std::numeric_limits<long double>::infinity();
        }, work, info);
    require(kept[0] == exact[0] * 0.5 && info.measure == 8 * expected_measure &&
                !info.tolerance_satisfied && work.refinement_steps == 1 && preconditions == 2 &&
                products == (bad_product ? 2 : 1),
            "可选 IR 正常返回非有限结果只能停止并保留已验算方向，实际启动工作不能消失");
  }
  {
    // 缩小 G 使初始方向及 J 乘积有限，仅候选相加超出 double，且不依赖扩展指数范围。
    const auto equality_layout = make_ipm_full_newton_layout(1, {false});
    const std::vector<double> tiny_values = {std::numeric_limits<double>::min()}, zero = {0}, small_rhs = {1, 1};
    const double largest = std::numeric_limits<double>::max();
    products = preconditions = 0;
    work = {};
    const auto kept = solve_ipm_bounded_newton(equality_layout, small_rhs,
        [&](std::span<const double>) { ++preconditions; return std::vector<double>{largest, largest}; },
        [&](std::span<const double> input, std::vector<long double>& output) {
          ++products;
          ipm_full_newton_product(equality_layout, starts, rows, tiny_values, zero, zero, input, output);
        }, work, info);
    require(kept == std::vector<double>({largest, largest}) && std::isfinite(info.measure) &&
                !info.tolerance_satisfied && preconditions == 2 && products == 1 && work.refinement_steps == 1,
            "候选加法溢出须保留初始有限方向，等式 z=0 不得触发互补除法");
  }
  for (bool bad_product : {false, true}) {
    bool rejected = false;
    work = {};
    try {
      solve_ipm_bounded_newton(layout, rhs,
          [&](std::span<const double> input) {
            auto answer = inverse(input);
            if (!bad_product) answer[0] = inf;
            return answer;
          }, [&](std::span<const double> input, std::vector<long double>& output) {
            multiply(input, output);
            if (bad_product) output[0] = std::numeric_limits<long double>::quiet_NaN();
          }, work, info);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && !info.tolerance_satisfied && !std::isfinite(info.measure),
            "初始方向或真实残差非有限时没有合法保留点，必须失败");
  }
  struct TestBudgetExpired {};
  bool expired = false, interrupted = false;
  products = preconditions = 0;
  work = {};
  try {
    solve_ipm_bounded_newton(layout, rhs,
        [&](std::span<const double> input) { return scaled_inverse(input, 0.5); },
        [&](std::span<const double> input, std::vector<long double>& output) {
          multiply(input, output);
          if (products == 2) expired = true;
        }, work, info, [&] { if (expired) throw TestBudgetExpired{}; });
  } catch (const TestBudgetExpired&) { interrupted = true; }
  require(interrupted && products == 2 && preconditions == 2 && work.refinement_steps == 1 &&
              !std::isfinite(info.measure),
          "可选 IR 期间超预算也必须传播中断，不得借已有有限方向伪装正常返回");
  struct TestCallbackFailure {};
  bool propagated = false;
  preconditions = 0;
  work = {};
  try {
    solve_ipm_bounded_newton(layout, rhs, [&](std::span<const double> input) {
      if (preconditions != 0) throw TestCallbackFailure{};
      return scaled_inverse(input, 0.5);
    }, multiply, work, info);
  } catch (const TestCallbackFailure&) { propagated = true; }
  require(propagated, "预条件回调主动异常必须原样传播，不能误判为可选候选未改善");
}

void check_bounded_newton_solves() {
  for (const std::string linear : {"qdldl", "normal_augmented", "factorhighs"}) {
    solver::SolveOptions options;
    options.ipm.linear_accuracy = "bounded_newton";
    options.ipm.linear_solver = linear;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    optimal_mode(one_variable(-2, -3, 5, 1, -inf, 2), -4, options);
    optimal_mode(one_variable(3, -inf, inf, -2, -4, -4), 6, options);
    auto maximum = one_variable(2, -3, 5, 1, -inf, 2);
    maximum.sense = solver::ObjectiveSense::maximize;
    optimal_mode(maximum, 4, options);
  }
  solver::SolveOptions options;
  options.ipm.linear_accuracy = "bounded_newton";
  const auto model = one_variable(1, 0, inf, 1, 2, inf);
  options.ipm.central_corrections = 1;
  require(solver::solve_ipm(model, options).status == "error", "有界 Newton 暂不接受额外中心校正");
  options.ipm.central_corrections = 0;
  options.ipm.krylov_history_size = 1;
  require(solver::solve_ipm(model, options).status == "error", "有界 Newton 不使用 Krylov 历史");
  options.ipm.krylov_history_size = 0;
  options.ipm.direction_retries = 1;
  require(solver::solve_ipm(model, options).status == "error", "有界 Newton 暂不接受方向重试");
  options.ipm.direction_retries = 0;
  options.time_limit = 0;
  const auto stopped = solver::solve_ipm(model, options);
  require(stopped.status == "time_limit" && *stopped.ipm_full_newton_products == 0 &&
              *stopped.ipm_full_newton_preconditioner_solves == 0 && *stopped.ipm_bounded_newton_returns == 0 &&
              *stopped.ipm_bounded_newton_inexact_returns == 0 && *stopped.ipm_factorizations == 0,
          "零预算不得启动有界 Newton 工作或伪造返回计数");
}

void check_homogeneous_algebra() {
  using solver::detail::ipm_homogeneous_scalar_direction;
  using solver::detail::ipm_homogeneous_step_limit;
  // 一条不等式、一条等式；等式乘子 -3 是合法自由变量。
  // G=(1,1), c=-1, b=(2,3), x=0, s=(1,0), z=(2,-3), tau=kappa=1。
  const std::vector<double> costs = {-1}, bounds = {2, 3}, constant = {3, 2, -1};
  const double rd = -2, rp0 = -1, rp1 = -3, rg = -4;
  const auto inverse = [](double d, double p0, double p1) {
    const double dx = p1, dz0 = 2 * (dx - p0);
    return std::vector<double>{dx, dz0, d - dz0};
  };
  require(inverse(1, 2, 3) == constant, "混合等式常数解应符合独立手算 K0 的三行方程");
  const auto check_five_blocks = [&](double theta, double hs, double ht,
      double expected_tau, double expected_kappa) {
    const auto u = inverse(-theta * rd, -theta * rp0 - hs / 2, -theta * rp1);
    const auto scalar = ipm_homogeneous_scalar_direction(costs, bounds, u, constant, 1, 1, -theta * rg, ht);
    const double dx = u[0] + constant[0] * scalar.tau;
    const double dz0 = u[1] + constant[1] * scalar.tau;
    const double dz1 = u[2] + constant[2] * scalar.tau;
    const double ds = (hs - dz0) / 2;
    const auto close = [&](double value, double target, const std::string& message) {
      require(std::abs(value - target) < 2e-12, message);
    };
    close(scalar.tau, expected_tau, "齐次 tau 方向应符合独立有理数结果");
    close(scalar.kappa, expected_kappa, "齐次 kappa 方向应与已存储 tau 方向配对");
    close(dz0 + dz1 - scalar.tau, -theta * rd, "完整对偶方程必须包含 c*dtau 和 theta");
    close(dx + ds - 2 * scalar.tau, -theta * rp0, "不等式原始方程必须包含松弛和 -b*dtau");
    close(dx - 3 * scalar.tau, -theta * rp1, "等式原始方程不得虚增松弛变量或互补除法");
    close(-dx + 2 * dz0 + 3 * dz1 + scalar.kappa, -theta * rg, "间隙方程须与另外两块同时满足");
    close(2 * ds + dz0, hs, "锥互补方程必须保留原始 S、Z 系数");
    close(scalar.tau + scalar.kappa, ht, "tau*kappa 互补方程不能遗漏");
    return scalar;
  };
  require(2 * rp0 - 3 * rp1 + rg == 3,
          "z^T rp-x^T rd+tau*rg 必须等于 s^Tz+tau*kappa，等式 s=0");
  const auto affine = check_five_blocks(1, -2, -1, -4.0 / 3, 1.0 / 3);
  require(std::abs(ipm_homogeneous_step_limit(1.5, 1, 1, affine) - 0.75) < 1e-15,
          "混合等式仿射步应由 tau 限制到 .75，自由等式乘子不参与锥边界");
  const double sigma = 1.0 / 64, mu = 1.5;
  check_five_blocks(1 - sigma, -2 + sigma * mu - 4.0 / 9,
      -1 + sigma * mu + 4.0 / 9, -1747.0 / 1728, 1655.0 / 3456);
  // 仅等式时没有锥互补项，但 tau*kappa 仍提供一个障碍项，mu=6 而非零。
  const std::vector<double> equality_cost = {3}, equality_bound = {2};
  const std::vector<double> equality_u = {3, -3}, equality_v = {2, -3};
  const auto equality = ipm_homogeneous_scalar_direction(equality_cost, equality_bound,
      equality_u, equality_v, 2, 3, 0.75, -4.5);
  require(equality.tau == 0 && equality.kappa == -2.25 &&
              3 * equality.tau + 2 * equality.kappa == -4.5,
          "纯等式的校正标量方程须使用 mu=tau*kappa 和非零 tau 归一化");
  require(ipm_homogeneous_step_limit(inf, 2, 3, {0, -6}) == 0.5 &&
              ipm_homogeneous_step_limit(0.25, 2, 3, {0, -6}) == 0.25 &&
              ipm_homogeneous_step_limit(inf, 2, 3, {1, 1}) == 1,
          "共同步长必须同时尊重 kappa、原锥与单位上限");
  for (double denominator_case : {1.0, 2.0}) {
    bool rejected = false;
    try {
      const std::vector<double> c = {1}, b, u = {0}, v = {denominator_case};
      ipm_homogeneous_scalar_direction(c, b, u, v, 1, 1, 0, -1);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "零或负标量分母必须失败，不能通过任意正地板继续");
  }
  bool rejected = false;
  try {
    ipm_homogeneous_scalar_direction(costs, bounds, constant, constant, 0, 1, 0, -1);
  } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "提取与标量消元不得接受零 tau");
  rejected = false;
  try {
    ipm_homogeneous_step_limit(inf, 1, 1, {inf, 0});
  } catch (const std::runtime_error&) { rejected = true; }
  require(rejected, "非有限标量方向不得产生貌似合法的共同步长");
  struct TestBudgetExpired {};
  bool interrupted = false;
  int budget_checks = 0;
  try {
    ipm_homogeneous_scalar_direction(costs, bounds, constant, constant, 1, 1, 0, -1,
        [&] { if (++budget_checks == 3) throw TestBudgetExpired{}; });
  } catch (const TestBudgetExpired&) { interrupted = true; }
  require(interrupted, "标量消元末尾超预算必须传播，不能返回已算好的标量方向");
}

void check_homogeneous_repair() {
  using solver::detail::IpmHomogeneousRepairInfo;
  using solver::detail::ipm_homogeneous_scalar_direction;
  using solver::detail::ipm_homogeneous_scalar_direction_repair;
  // G=1、H=1/2 的手算 K0。正确常数 v=(5/2,1)，故分母=3/2。
  // 错误常数 (4,4) 给出有限负分母 -3，回调执行解析逆而非直接复制期望方向。
  const std::vector<double> costs = {-1}, bounds = {2}, u = {1.5, -1}, bad = {4, 4};
  const auto inverse = [](double dual, double primal) {
    return std::vector<double>{primal + 0.5 * dual, dual};
  };
  const auto exact_constant = inverse(1, 2);
  int calls = 0;
  const auto repair = [&] {
    ++calls;
    ++homogeneous_repair_test_attempts;
    return inverse(-costs[0], bounds[0]);
  };
  auto constant = exact_constant;
  IpmHomogeneousRepairInfo info;
  const auto normal = ipm_homogeneous_scalar_direction_repair(
      costs, bounds, u, constant, 1, 1, -5, -1, repair, info);
  const auto reference = ipm_homogeneous_scalar_direction(costs, bounds, u, exact_constant, 1, 1, -5, -1);
  require(normal.tau == reference.tau && normal.kappa == reference.kappa && calls == 0 &&
              !info.attempted && !info.succeeded && constant == exact_constant,
          "正分母不得启动恢复，标量结果与旧函数逐值一致");
  for (const auto& initial : {bad, std::vector<double>{1, 1}}) {
    // 第二组 c^T v_x+b^T v_z=1 恰好令分母为零。
    constant = initial;
    info = {};
    calls = 0;
    const auto scalar = ipm_homogeneous_scalar_direction_repair(
        costs, bounds, u, constant, 1, 1, -5, -1, repair, info);
    require(calls == 1 && info.attempted && info.succeeded && constant == exact_constant &&
                scalar.tau == reference.tau && scalar.kappa == reference.kappa,
            "有限零或负分母须实际触发一次解析恢复，成功后提交共享常数");
    ++homogeneous_repair_test_successes;
    const double dx = u[0] + constant[0] * scalar.tau;
    const double dz = u[1] + constant[1] * scalar.tau;
    const double ds = (-2 - dz) / 2;
    require(std::abs(dz - scalar.tau + 1) < 1e-14 &&
                std::abs(dx + ds - 2 * scalar.tau - 1) < 1e-14 &&
                std::abs(-dx + 2 * dz + scalar.kappa + 5) < 1e-14 &&
                std::abs(2 * ds + dz + 2) < 1e-14 &&
                std::abs(scalar.tau + scalar.kappa + 1) < 1e-14,
            "恢复后方向须同时满足手算三类线性方程与两个互补方程");
    const std::vector<double> corrected_u = {0.25, -0.125};
    const auto corrected = ipm_homogeneous_scalar_direction_repair(
        costs, bounds, corrected_u, constant, 1, 1, -0.75, -0.5, repair, info);
    const auto expected = ipm_homogeneous_scalar_direction(costs, bounds, corrected_u, exact_constant, 1, 1, -0.75, -0.5);
    require(calls == 1 && corrected.tau == expected.tau && corrected.kappa == expected.kappa,
            "同一迭代的校正方向必须复用已恢复常数，不再调用回调");
    constant = bad;
    bool failed = false;
    try { ipm_homogeneous_scalar_direction_repair(costs, bounds, u, constant, 1, 1, -5, -1, repair, info); }
    catch (const std::runtime_error&) { failed = true; }
    require(failed && calls == 1 && constant == bad,
            "同一状态的恢复额度耗尽后不得再次调用，不能形成隐式重试循环");
  }
  for (const auto& candidate : {bad, std::vector<double>{1, 1},
                                std::vector<double>{inf, 1}, std::vector<double>{1}}) {
    constant = bad;
    info = {};
    calls = 0;
    bool failed = false;
    try {
      ipm_homogeneous_scalar_direction_repair(costs, bounds, u, constant, 1, 1, -5, -1,
          [&] { ++calls; return candidate; }, info);
    } catch (const std::exception&) { failed = true; }
    require(failed && calls == 1 && info.attempted && !info.succeeded && constant == bad,
            "回调返回仍非正、非有限或错误维数时必须失败并保留原常数");
  }
  {
    const std::vector<double> one = {1}, empty, zero = {0};
    constant = {2};
    info = {};
    calls = 0;
    bool failed = false;
    try {
      ipm_homogeneous_scalar_direction_repair(one, empty, zero, constant, 1, 1, 0,
          std::numeric_limits<double>::max(), [&] {
            ++calls;
            return std::vector<double>{std::nextafter(1.0, 0.0)};
          }, info);
    } catch (const std::runtime_error&) { failed = true; }
    require(failed && calls == 1 && info.attempted && !info.succeeded && constant == std::vector<double>{2},
            "恢复候选虽有正分母，但标量方向溢出时仍不得提交常数或宣称成功");
  }
  for (int invalid = 0; invalid < 7; ++invalid) {
    constant = bad;
    info = {};
    calls = 0;
    auto input_u = u;
    auto input_costs = costs;
    double tau = 1, kappa = 1;
    long double gap = -5, complementarity = -1;
    if (invalid == 0) tau = 0;
    if (invalid == 1) kappa = inf;
    if (invalid == 2) input_u[0] = inf;
    if (invalid == 3) input_costs[0] = inf;
    if (invalid == 4) gap = std::numeric_limits<long double>::infinity();
    if (invalid == 5) complementarity = std::numeric_limits<long double>::infinity();
    if (invalid == 6) constant[0] = inf;
    const auto original = constant;
    bool failed = false;
    try {
      ipm_homogeneous_scalar_direction_repair(input_costs, bounds, input_u, constant,
          tau, kappa, gap, complementarity, repair, info);
    } catch (const std::runtime_error&) { failed = true; }
    require(failed && calls == 0 && !info.attempted && !info.succeeded && constant == original,
            "其他数值异常不得冒充有限非正分母并启动恢复");
  }
  // 正分母存在，但最终 double 标量溢出；这属于后续方向数值错误，不可恢复常数。
  {
    const std::vector<double> zero = {0}, empty;
    constant = zero;
    info = {};
    calls = 0;
    bool failed = false;
    try {
      ipm_homogeneous_scalar_direction_repair(zero, empty, zero, constant,
          1, std::numeric_limits<double>::denorm_min(), 0, 1, repair, info);
    } catch (const std::runtime_error&) { failed = true; }
    require(failed && calls == 0 && !info.attempted && constant == zero,
            "正分母后的 double 标量溢出不得触发常数恢复");
  }
  struct TestBudgetExpired {};
  for (int failure : {0, 1, 2}) {
    constant = bad;
    info = {};
    calls = 0;
    bool propagated = false;
    try {
      ipm_homogeneous_scalar_direction_repair(costs, bounds, u, constant, 1, 1, -5, -1,
          [&]() -> std::vector<double> {
            ++calls;
            if (failure == 0) throw TestBudgetExpired{};
            if (failure == 1) throw std::bad_alloc{};
            // 从原函数产生真正的其他 NumericalFailure，不能被一次恢复 helper 吞掉。
            ipm_homogeneous_scalar_direction(costs, bounds, u, exact_constant, 0, 1, -5, -1);
            return exact_constant;
          }, info);
    } catch (const TestBudgetExpired&) { propagated = failure == 0; }
      catch (const std::bad_alloc&) { propagated = failure == 1; }
      catch (const std::runtime_error&) { propagated = failure == 2; }
    require(propagated && calls == 1 && info.attempted && !info.succeeded && constant == bad,
            "恢复回调的预算、内存及其他数值异常须原样传播且不提交常数");
  }
  int checks_to_commit = 0;
  constant = bad;
  info = {};
  ipm_homogeneous_scalar_direction_repair(costs, bounds, u, constant, 1, 1, -5, -1,
      [&] { return exact_constant; }, info, [&] { ++checks_to_commit; });
  for (int stop : {1, checks_to_commit}) {
    constant = bad;
    info = {};
    calls = 0;
    int budgets = 0;
    bool interrupted = false;
    try {
      ipm_homogeneous_scalar_direction_repair(costs, bounds, u, constant, 1, 1, -5, -1,
          [&] { ++calls; return exact_constant; }, info,
          [&] { if (++budgets == stop) throw TestBudgetExpired{}; });
    } catch (const TestBudgetExpired&) { interrupted = true; }
    require(interrupted && constant == bad && !info.succeeded &&
                (stop == 1 ? !info.attempted && calls == 0 : info.attempted && calls == 1),
            "首次或提交前最后预算中断必须保留旧常数，成功不能提前记账");
  }
}

void check_homogeneous_scaled_algebra() {
  using namespace solver::detail;
  const auto close = [](long double value, long double target, const std::string& message) {
    require(std::isfinite(value) && std::abs(value - target) <= 5e-13L * std::max(1.0L, std::abs(target)), message);
  };
  // 独立手算：G=(1,1)，一条不等式和一条等式；tau=2、kappa=8，等式乘子为自由负数。
  const std::vector<std::int64_t> starts = {0, 2}, rows = {0, 1};
  const std::vector<double> values = {1, 1}, costs = {-1}, bounds = {2, 3}, slack = {4, 0}, dual = {1, -3};
  const auto scaling = make_ipm_homogeneous_scaling(1, {true, false}, slack, dual, 2, 8);
  require(scaling.dimension() == 6 && scaling.layout.inequality_rows == std::vector<std::size_t>{0} &&
              scaling.weights == std::vector<long double>{2} && scaling.lambda == std::vector<long double>{2} &&
              scaling.d == 2 && scaling.lambda_tau == 4,
          "五块布局只能为不等式分配互补项，非单位 tau 的两个标量尺度须分别形成");
  const auto inverse = [](std::span<const double> rhs) {
    const double dx = rhs[2], dz0 = (dx - rhs[1]) / 4;
    return std::vector<double>{dx, dz0, rhs[0] - dz0};
  };
  const auto constant_rhs = ipm_homogeneous_scaled_constant_rhs(scaling, costs, bounds);
  require(constant_rhs == std::vector<double>({0.5, 1, 1.5}), "常数右端须为 (-c,b)/d");
  const auto constant = inverse(constant_rhs);
  require(constant == std::vector<double>({1.5, 0.125, 0.375}), "常数解必须符合独立三行 K0 方程");
  const auto denominator = ipm_homogeneous_scaled_denominator(scaling, constant);
  require(denominator == 17.0L / 16, "正能量分母只累计不等式乘子的加权平方");
  const std::vector<double> rhs = {3, -2, 3, 1.5, -2, -2};
  const std::vector<double> expected = {36.0 / 17, 3.0 / 17, 43.0 / 17, -80.0 / 17, -5.0 / 17, -48.0 / 17};
  const auto reduced = inverse(ipm_homogeneous_scaled_reduced_rhs(scaling, rhs));
  require(reduced == std::vector<double>({3, 0.25, 2.75}), "互补项消去必须保持原始块右端的符号");
  const auto recovered = ipm_homogeneous_scaled_recover(scaling, costs, bounds, rhs, reduced, constant, denominator);
  for (std::size_t i = 0; i < expected.size(); ++i)
    close(recovered[i], expected[i], "物理 double 方向必须符合独立有理数解析解");
  const long double dx = recovered[0], dz0 = recovered[1], dz1 = recovered[2];
  const long double ds = recovered[3], dt = recovered[4], dk = recovered[5];
  const std::vector<long double> independent = {
      dz0 + dz1 - dt, dx + ds - 2 * dt, dx - 3 * dt,
      (-dx + 2 * dz0 + 3 * dz1 + dk) / 2, (ds + 4 * dz0) / 2, (8 * dt + 2 * dk) / 4};
  std::vector<long double> product;
  ipm_homogeneous_scaled_product(scaling, starts, rows, values, costs, bounds, slack, dual, 2, 8, recovered, product);
  for (std::size_t i = 0; i < independent.size(); ++i) {
    close(independent[i], rhs[i], "真实已舍入方向必须独立满足全部五块方程");
    close(product[i], independent[i], "CSC 乘积须与独立物理方程一致，等式行没有松弛方向");
  }
  const std::vector<double> distinct_rhs = {10, 20, -30, 40, 50, 60};
  const auto tolerance = ipm_homogeneous_scaled_tolerances(scaling, distinct_rhs);
  const std::array<double, 5> norms = {10, 30, 40, 50, 60};
  const std::array<const char*, 5> blocks = {"对偶", "原始", "间隙", "互补", "标量互补"};
  for (std::size_t i = 0; i < norms.size(); ++i) {
    // 用独立长双精度理论值检查公式；数个 double ULP 只吸收常量折叠与运行时舍入差异。
    const long double theoretical = 1e-12L + 1e-10L * static_cast<long double>(norms[i]);
    const double rounded = static_cast<double>(theoretical);
    const double previous_expected = 1e-12 + 1e-10 * norms[i];
    const long double ulp = std::max(
        static_cast<long double>(std::nextafter(rounded, inf)) - rounded,
        static_cast<long double>(rounded) - std::nextafter(rounded, -inf));
    const long double error = std::abs(static_cast<long double>(tolerance[i]) - theoretical);
    std::cout << "五块门槛诊断：块=" << i << "（" << blocks[i] << "），实际=" << std::hexfloat << tolerance[i]
              << "，原 double 期望=" << previous_expected << "，理论=" << theoretical
              << "，ULP=" << ulp << std::defaultfloat << "，误差/ULP=" << error / ulp << '\n';
    require(std::isfinite(tolerance[i]) && tolerance[i] > 0 && error <= 4 * ulp,
            "五块门槛须分别符合固定原始 RHS 的独立理论式，误差不得超过四个 double ULP；块=" +
                std::to_string(i) + "（" + blocks[i] + "）");
  }
  int direct = 0, products = 0;
  IpmBoundedNewtonInfo info;
  const auto solved = solve_ipm_homogeneous_scaled(scaling, rhs,
      [&](std::span<const double> input, bool refinement) {
        ++direct;
        require(!refinement, "精确解析预条件不得启动可选 IR");
        return ipm_homogeneous_scaled_recover(scaling, costs, bounds, input,
            inverse(ipm_homogeneous_scaled_reduced_rhs(scaling, input)), constant, denominator);
      }, [&](std::span<const double> input, std::vector<long double>& output) {
        ++products;
        ipm_homogeneous_scaled_product(scaling, starts, rows, values, costs, bounds, slack, dual, 2, 8, input, output);
      }, info);
  require(solved == recovered && direct == 1 && products == 1 && info.tolerance_satisfied && info.measure <= 1,
          "混合等式精确方向须一次直接解和一次真实乘积达标");
  // 大方向相消后仍按原始 S、Z、tau、kappa 验算，不能以舍入 W^2 替代原系数。
  const std::vector<std::int64_t> empty_starts = {0}, empty_rows;
  const std::vector<double> empty, one = {1}, physical_slack = {3}, physical_dual = {10};
  const auto physical_scaling = make_ipm_homogeneous_scaling(0, {true}, physical_slack, physical_dual, 7, 11);
  const std::vector<double> physical = {1e16, std::nextafter(-3e15, inf), 1e16, std::nextafter(-11e16 / 7, inf)};
  ipm_homogeneous_scaled_product(physical_scaling, empty_starts, empty_rows, empty, empty, one,
      physical_slack, physical_dual, 7, 11, physical, product);
  require(product[2] == (10 * static_cast<long double>(physical[1]) + 3 * static_cast<long double>(physical[0])) /
              std::sqrt(30.0L) &&
              product[3] == (11 * static_cast<long double>(physical[2]) + 7 * static_cast<long double>(physical[3])) /
              std::sqrt(77.0L),
          "物理 double 恢复后的相消残差须使用原始系数，不能通过重构松弛方向消除真实误差");
}

void check_homogeneous_scaled_refinement() {
  using namespace solver::detail;
  // 无 x 的一条不等式：J(z,s,dt,dk)=(s-dt,z+dk,z+s,dt+dk)。
  // 故意把精确常数解 -1 改为 -.5；正分母 1.25 不能保证完整方向正确。
  const std::vector<std::int64_t> starts = {0}, rows;
  const std::vector<double> empty, one = {1}, constant = {-0.5}, rhs = {0, 0, 0, 1};
  const auto scaling = make_ipm_homogeneous_scaling(0, {true}, one, one, 1, 1);
  const auto denominator = ipm_homogeneous_scaled_denominator(scaling, constant);
  require(denominator == 1.25L, "扰动常数例仍须拥有合法正分母");
  int direct = 0, refinements = 0, products = 0;
  const auto precondition = [&](std::span<const double> input, bool refinement) {
    ++direct;
    if (refinement) ++refinements;
    require(refinement == (direct > 1), "精化标记必须区分首解和实际额外直接解");
    auto reduced = ipm_homogeneous_scaled_reduced_rhs(scaling, input);
    reduced[0] = -reduced[0];
    return ipm_homogeneous_scaled_recover(scaling, empty, one, input, reduced, constant, denominator);
  };
  const auto multiply = [&](std::span<const double> input, std::vector<long double>& output) {
    ++products;
    ipm_homogeneous_scaled_product(scaling, starts, rows, empty, empty, one, one, one, 1, 1, input, output);
  };
  const auto close = [](long double value, long double target, const std::string& message) {
    require(std::isfinite(value) && std::abs(value - target) <= 2e-12L * std::max(1.0L, std::abs(target)), message);
  };
  const auto initial = precondition(rhs, false);
  std::vector<long double> product;
  multiply(initial, product);
  close(product[0], -0.4L, "合法正分母不能掩盖扰动常数产生的原始块缺陷");
  close(product[1], -0.2L, "合法正分母不能掩盖扰动常数产生的间隙块缺陷");
  direct = refinements = products = 0;
  IpmBoundedNewtonInfo info;
  const auto refined = solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info);
  const std::vector<double> expected = {-326.0 / 625, 326.0 / 625, 272.0 / 625, 353.0 / 625};
  for (std::size_t i = 0; i < expected.size(); ++i)
    close(refined[i], expected[i], "三次完整 IR 须符合误差每次反向乘 .6 的独立解析结果");
  const long double initial_measure = 0.4L / 1e-12;
  close(info.measure, initial_measure * 0.216L, "返回残差必须属于三次精化后保留的方向");
  require(direct == 4 && refinements == 3 && products == 4 && info.measure > 1 && !info.tolerance_satisfied,
          "缩放方向最多三次 IR，有限未达标如实返回，不得补充隐式 Krylov");
  for (bool stagnate : {false, true}) {
    direct = refinements = products = 0;
    const auto retained = solve_ipm_homogeneous_scaled(scaling, rhs,
        [&](std::span<const double> input, bool refinement) {
          auto answer = precondition(input, refinement);
          if (stagnate && refinement) std::fill(answer.begin(), answer.end(), 0);
          if (!stagnate && direct == 3) for (double& value : answer) value = -value;
          return answer;
        }, multiply, info);
    const std::vector<double> best = stagnate ? initial : std::vector<double>{-0.56, 0.56, 0.32, 0.68};
    for (std::size_t i = 0; i < best.size(); ++i)
      close(retained[i], best[i], "停滞或变差候选须立即停止并保留前一最佳物理方向");
    require(direct == (stagnate ? 2 : 3) && products == direct && refinements == direct - 1 &&
                !info.tolerance_satisfied, "被拒候选仍须计入已启动的直接解和真实乘积");
    close(info.measure, initial_measure * (stagnate ? 1 : 0.6L), "残差报告不能属于被拒绝的最后候选");
  }
  for (bool bad_product : {false, true}) {
    direct = refinements = products = 0;
    const auto retained = solve_ipm_homogeneous_scaled(scaling, rhs,
        [&](std::span<const double> input, bool refinement) {
          auto answer = precondition(input, refinement);
          if (!bad_product && refinement) answer[0] = inf;
          return answer;
        }, [&](std::span<const double> input, std::vector<long double>& output) {
          multiply(input, output);
          if (bad_product && products == 2) output[0] = std::numeric_limits<long double>::quiet_NaN();
        }, info);
    require(retained == initial && direct == 2 && refinements == 1 && products == (bad_product ? 2 : 1) &&
                !info.tolerance_satisfied, "可选候选或乘积非有限只能保留已验算方向并保留实际工作计数");
    close(info.measure, initial_measure, "非有限可选候选不能污染保留方向的残差");
    direct = refinements = products = 0;
    info = {123, true};
    bool rejected = false;
    try {
      solve_ipm_homogeneous_scaled(scaling, rhs,
          [&](std::span<const double> input, bool refinement) {
            auto answer = precondition(input, refinement);
            if (!bad_product) answer[0] = inf;
            return answer;
          }, [&](std::span<const double> input, std::vector<long double>& output) {
            multiply(input, output);
            if (bad_product) output[0] = std::numeric_limits<long double>::infinity();
          }, info);
    } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && info.measure == 123 && info.tolerance_satisfied && direct == 1 &&
                refinements == 0 && products == (bad_product ? 1 : 0),
            "初始方向或乘积非有限必须失败，不能覆盖调用者 info 或抹去已启动工作");
  }
  struct TestBudgetExpired {};
  for (int failure : {0, 1, 2}) {
    for (bool fail_product : {false, true}) {
      direct = refinements = products = 0;
      info = {123, true};
      const auto fail = [&] {
        if (failure == 0) throw std::runtime_error("测试回调异常");
        if (failure == 1) throw std::bad_alloc{};
        throw TestBudgetExpired{};
      };
      bool propagated = false;
      try {
        solve_ipm_homogeneous_scaled(scaling, rhs,
            [&](std::span<const double> input, bool refinement) {
              auto answer = precondition(input, refinement);
              if (!fail_product && refinement) fail();
              return answer;
            }, [&](std::span<const double> input, std::vector<long double>& output) {
              multiply(input, output);
              if (fail_product && products == 2) fail();
            }, info);
      } catch (const std::runtime_error&) { propagated = failure == 0; }
        catch (const std::bad_alloc&) { propagated = failure == 1; }
        catch (const TestBudgetExpired&) { propagated = failure == 2; }
      require(propagated && info.measure == 123 && info.tolerance_satisfied && direct == 2 &&
                  refinements == 1 && products == (fail_product ? 2 : 1),
              "可选精化回调的数值、分配或预算异常必须原样传播，不能伪装为有限方向正常返回");
    }
  }
  direct = refinements = products = 0;
  int budget_checks = 0;
  solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, [&] { ++budget_checks; });
  const int final_check = budget_checks;
  direct = refinements = products = budget_checks = 0;
  info = {123, true};
  bool interrupted = false;
  try {
    solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info,
        [&] { if (++budget_checks == final_check) throw TestBudgetExpired{}; });
  } catch (const TestBudgetExpired&) { interrupted = true; }
  require(interrupted && direct == 4 && refinements == 3 && products == 4 && info.measure == 123 &&
              info.tolerance_satisfied, "末尾预算检查必须在提交 info 和有限方向之前生效");
}

void check_objective_scaling() {
  require(solver::IpmOptions{}.objective_scaling, "目标归一化必须默认开启，保持既有路径");
  // 4x=8，成本为8、目标常数为7；原模型最优值23，最小化行对偶为2，最大化为-2。
  // 保留默认行列缩放，独立检查目标开关下的原坐标与对偶恢复。
  for (const std::string method : {"mehrotra", "homogeneous"}) {
    for (bool objective_scaling : {true, false}) {
      solver::SolveOptions options;
      options.ipm.method = method;
      options.ipm.objective_scaling = objective_scaling;
      if (method == "homogeneous") options.ipm.linear_accuracy = "homogeneous_repair";
      for (bool maximize : {false, true}) {
        auto model = one_variable(8, -inf, inf, 4, 8, 8);
        model.objective_offset = 7;
        model.sense = maximize ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
        const auto result = optimal_mode(model, 23, options);
        require(result.solution.size() == 1 && std::abs(result.solution[0] - 2) < 1e-7 &&
                    result.row_dual.size() == 1 && std::abs(result.row_dual[0] - (maximize ? -2 : 2)) < 1e-7,
                "目标归一化开关不能破坏最小化/最大化的原变量与原行对偶恢复");
        require(result.model.costs == model.costs && result.model.values == model.values &&
                    result.model.objective_offset == 7 && result.model.sense == model.sense,
                "工作目标尺度不得修改用于独立验算的原模型");
      }
    }
  }

  // 两个有限非零成本的比值超出 double 指数范围。on 应保留既有下溢拒绝，
  // off 则不做这次除法；x=0 已是原模型可直接验证的最优点，无须大数值迭代。
  solver::Model extreme;
  extreme.costs = {1e200, 1e-200};
  extreme.column_lower = {0, 0};
  extreme.column_upper = {inf, inf};
  extreme.kinds.assign(2, solver::VariableKind::continuous);
  extreme.column_start = {0, 0, 0};
  solver::SolveOptions options;
  options.ipm.method = "homogeneous";
  const auto normalized = solver::solve_ipm(extreme, options);
  require(normalized.status == "numerical_error" && normalized.message.find("下溢") != std::string::npos &&
              !normalized.has_solution && *normalized.ipm_factorizations == 0,
          "默认目标归一化仍须拒绝非零成本下溢，不能静默丢弃目标项");
  options.ipm.objective_scaling = false;
  const auto unnormalized = optimal_mode(extreme, 0, options);
  require(unnormalized.solution == std::vector<double>({0, 0}) &&
              unnormalized.model.costs == extreme.costs && *unnormalized.ipm_factorizations == 0 &&
              *unnormalized.ipm_iterations == 0,
          "关闭目标归一化须真正避开目标除法，并在原模型上验证有限成本的零最优点");
}

void check_homogeneous_solves(const std::string& accuracy = "strict") {
  for (const std::string linear : {"qdldl", "normal_augmented", "sparselu", "normal_cholmod", "normal_cholmod_aat"}) {
    solver::SolveOptions options;
    options.ipm.method = "homogeneous";
    options.ipm.linear_accuracy = accuracy;
    options.ipm.linear_solver = linear;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    optimal_mode(one_variable(-1, -inf, inf, 1, -inf, 2), -2, options);
    optimal_mode(one_variable(3, -inf, inf, -2, -4, -4), 6, options);
  }
  solver::Model fixed;
  fixed.costs = {7, 4, -1};
  fixed.column_lower = {3, -inf, -inf};
  fixed.column_upper = {3, inf, inf};
  fixed.kinds.assign(3, solver::VariableKind::continuous);
  fixed.row_lower = {5, -inf};
  fixed.row_upper = {5, 7};
  fixed.column_start = {0, 2, 4, 5};
  fixed.row_index = {0, 1, 0, 1, 1};
  fixed.values = {1, 1, 1, 1, 1};
  fixed.objective_offset = 11;
  solver::SolveOptions options;
  options.ipm.method = "homogeneous";
  options.ipm.linear_accuracy = accuracy;
  options.ipm.eliminate_fixed_variables = true;
  options.ipm.presolve_singleton_rows = true;
  options.ipm.eliminate_presolved_fixed_variables = true;
  for (bool maximize : {false, true}) {
    fixed.sense = maximize ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
    fixed.costs[2] = maximize ? 1 : -1;
    const auto result = optimal_mode(fixed, maximize ? 42 : 38, options);
    require(result.ipm_homogeneous_tau && std::abs(*result.ipm_homogeneous_tau - 1) > 1e-6 &&
                result.solution[0] == 3 && result.solution[1] == 2 && std::abs(result.solution[2] - 2) < 1e-6 &&
                *result.ipm_eliminated_fixed_variables == 2 && *result.ipm_presolved_fixed_columns == 1,
            "tau 不等于一时两层固定列仍须恢复原常量，且原行对偶、目标常数和最大化同时通过终验");
  }
  const auto early = solver::solve_ipm(one_variable(3, 2, 2, 1, 2, 2), options);
  require(early.status == "optimal" && *early.ipm_factorizations == 0 &&
              *early.ipm_homogeneous_constant_solves == 0 && *early.ipm_homogeneous_direction_solves == 0 &&
              *early.ipm_homogeneous_constant_repairs == 0 && *early.ipm_homogeneous_constant_repair_successes == 0 &&
              *early.ipm_homogeneous_scaled_products == 0 && *early.ipm_homogeneous_scaled_refinement_steps == 0 &&
              *early.ipm_homogeneous_scaled_inexact_returns == 0 &&
              !early.ipm_homogeneous_tau && !early.ipm_homogeneous_kappa,
          "初始化前直接验算的最优固定点不应伪造齐次标量或求解工作");
  options = {};
  options.ipm.method = "homogeneous";
  const auto model = one_variable(-1, -inf, inf, 1, -inf, 2);
  for (const std::string invalid_accuracy : {"backward", "inexact", "full_newton", "bounded_newton"}) {
    options.ipm.linear_accuracy = invalid_accuracy;
    require(solver::solve_ipm(model, options).status == "error", "齐次方法仅允许已实现的严格或缩放内层规则");
  }
  options.ipm.linear_accuracy = accuracy;
  options.ipm.krylov_history_size = 1;
  require(solver::solve_ipm(model, options).status == "error", "齐次初版不得组合历史回收");
  options.ipm.krylov_history_size = 0;
  options.ipm.direction_retries = 1;
  require(solver::solve_ipm(model, options).status == "error", "齐次初版不得组合方向重试");
  options.ipm.direction_retries = 0;
  options.ipm.central_corrections = 1;
  require(solver::solve_ipm(model, options).status == "error", "齐次初版不得组合额外中心校正");
  options.ipm.central_corrections = 0;
  options.ipm.inexact_prefilter = true;
  require(solver::solve_ipm(model, options).status == "error", "齐次方法不得组合旧非精确方向预拒绝");
  options.ipm.inexact_prefilter = false;
  options.time_limit = 0;
  const auto stopped = solver::solve_ipm(model, options);
  require(stopped.status == "time_limit" && *stopped.ipm_factorizations == 0 &&
              *stopped.ipm_homogeneous_constant_solves == 0 && *stopped.ipm_homogeneous_direction_solves == 0 &&
              *stopped.ipm_homogeneous_constant_repairs == 0 && *stopped.ipm_homogeneous_constant_repair_successes == 0 &&
              *stopped.ipm_homogeneous_scaled_products == 0 && *stopped.ipm_homogeneous_scaled_refinement_steps == 0 &&
              *stopped.ipm_homogeneous_scaled_inexact_returns == 0 &&
              !stopped.ipm_homogeneous_tau && !stopped.ipm_homogeneous_kappa,
          "零预算不得开始齐次初始化或写入其标量");
  options.time_limit = solver::SolveOptions{}.time_limit;
  options.ipm.max_iterations = 5;
  for (const auto& unresolved : {one_variable(1, 0, 1, 1, 2, inf),
                                one_variable(-1, 0, inf, 1, -inf, inf)}) {
    const auto result = solver::solve_ipm(unresolved, options);
    require(result.status != "optimal" && result.status != "infeasible" && result.status != "unbounded",
            "不可行或无界例不能仅依据齐次标量比例宣称证书或最优");
  }
  options.ipm.method = "unknown";
  require(solver::solve_ipm(model, options).status == "error", "未知内点方法必须明确拒绝");
  if (accuracy == "homogeneous_scaled" || accuracy == "homogeneous_repair") {
    options.ipm.method = "mehrotra";
    require(solver::solve_ipm(model, options).status == "error", "齐次专属精度不能静默应用到普通预测校正方法");
  }
}

void check_full_newton_solves() {
  for (const std::string linear : {"qdldl", "normal_augmented", "factorhighs"}) {
    solver::SolveOptions options;
    options.ipm.linear_accuracy = "full_newton";
    options.ipm.linear_solver = linear;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    options.ipm.direction_retries = 8;
    optimal_mode(one_variable(-2, -3, 5, 1, -inf, 2), -4, options);
    optimal_mode(one_variable(3, -inf, inf, -2, -4, -4), 6, options);
    auto maximum = one_variable(2, -3, 5, 1, -inf, 2);
    maximum.sense = solver::ObjectiveSense::maximize;
    optimal_mode(maximum, 4, options);
  }
  solver::SolveOptions options;
  options.ipm.linear_accuracy = "full_newton";
  const auto model = one_variable(1, 0, inf, 1, 2, inf);
  options.ipm.central_corrections = 1;
  require(solver::solve_ipm(model, options).status == "error", "完整 Newton 未定义额外中心校正的组合必须拒绝");
  options.ipm.central_corrections = 0;
  options.ipm.krylov_history_size = 1;
  require(solver::solve_ipm(model, options).status == "error", "K0 历史不能冒充完整 J 的历史");
  options.ipm.krylov_history_size = 0;
  options.time_limit = 0;
  const auto stopped = solver::solve_ipm(model, options);
  require(stopped.status == "time_limit" && *stopped.ipm_full_newton_products == 0 &&
              *stopped.ipm_full_newton_preconditioner_solves == 0 && *stopped.ipm_factorizations == 0,
          "零预算必须在初始化和完整 Newton 工作开始前终止");
}

void check_krylov_history_algebra() {
  using solver::detail::IpmKrylovHistory;
  const auto multiply = [](const std::vector<double>& x) {
    return std::vector<double>{2 * x[0] + x[1], x[0] - 3 * x[1]};
  };
  const auto close = [](const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
      if (!std::isfinite(left[i]) || !std::isfinite(right[i]) ||
          std::abs(left[i] - right[i]) > 1e-11 * std::max({1.0, std::abs(left[i]), std::abs(right[i])})) return false;
    }
    return true;
  };
  const std::vector<double> first{1, 0}, second{1, 1}, residual{4, -7};
  IpmKrylovHistory history(2);
  require(history.size() == 0 && history.capacity() == 2, "回收空间必须从空状态及明确预算开始");
  require(history.add(first, multiply(first)), "独立有限向量应能加入回收空间");
  std::vector<double> correction, projected;
  require(history.project(residual, correction, projected), "已知不定矩阵的像空间投影必须可计算");
  require(close(multiply(correction), {residual[0] - projected[0], residual[1] - projected[1]}) &&
              std::abs(2 * projected[0] + projected[1]) < 1e-11,
          "投影应满足实际算子关系及像空间正交，不依赖正定矩阵假设");
  auto direction = second, image = multiply(second);
  require(history.orthogonalize(direction, image) && close(multiply(direction), image) &&
              std::abs(2 * image[0] + image[1]) < 1e-11,
          "Arnoldi 方向与像必须同步扣除，不能只投影矩阵乘积");
  const auto original_correction = correction, original_projected = projected;
  require(!history.add(std::vector<double>{2, 0}, std::vector<double>{4, 2}) && history.size() == 1,
          "相关方向不能产生虚假维数或污染有效空间");
  require(history.project(residual, correction, projected) &&
              close(correction, original_correction) && close(projected, original_projected),
          "拒绝入池必须保留原投影");
  require(history.add(second, multiply(second)) && history.size() == 2,
          "独立方向应补全二维回收空间");
  require(history.project(residual, correction, projected) && close(projected, {0, 0}) &&
              close(multiply(correction), residual),
          "完备二维空间的投影应解出不定线性系统，而非只减少估计残差");
  for (const auto& invalid : std::vector<std::vector<double>>{{inf, 0},
      {std::numeric_limits<double>::quiet_NaN(), 0}, {0, 0}}) {
    require(!history.add(first, invalid) && history.size() == 2,
            "无效或零像不能破坏已存空间");
  }
  for (const auto& invalid : std::vector<std::vector<double>>{{}, {1}}) {
    bool rejected = false;
    try { history.add(first, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && history.size() == 2, "维数不匹配必须明确拒绝且保留历史");
  }
  direction = {inf, 1}; image = {1, 2};
  require(!history.orthogonalize(direction, image) && direction[0] == inf && image == std::vector<double>({1, 2}),
          "方向投影失败必须保持调用者的两份输入");
  struct BudgetStop {};
  bool propagated = false;
  try { history.project(residual, correction, projected, [] { throw BudgetStop{}; }); }
  catch (const BudgetStop&) { propagated = true; }
  require(propagated && history.size() == 2, "回收不能吞掉原时间预算异常");
  history.clear();
  require(history.size() == 0 && history.capacity() == 2, "算子变化应清空旧空间但保留请求预算");
  const auto last_correction = correction, last_projected = projected;
  require(!history.project(residual, correction, projected) && correction == last_correction &&
              projected == last_projected, "空空间不得提出伪修正或改写输出");
  IpmKrylovHistory single(1);
  require(single.add(first, multiply(first)) && single.add(second, multiply(second)) && single.size() == 1,
          "满空间只在新方向可用时替换，并严格保持内存上限");
  require(close(multiply(single.direction(0)), single.image(0)) &&
              single.project(single.image(0), correction, projected) && close(projected, {0, 0}) &&
              close(correction, single.direction(0)), "FIFO 保留的正交方向及其像仍应满足真实算子关系");
  IpmKrylovHistory disabled(0);
  require(!disabled.add(first, multiply(first)) && disabled.size() == 0,
          "零预算不得保留历史向量");
  IpmKrylovHistory scaled_space(1);
  require(scaled_space.add(std::vector<double>{1e150, 0}, std::vector<double>{1e-150, 0}),
          "有限但尺度差异大的算子对不能误当零空间");
  require(scaled_space.project(std::vector<double>{1e-150, 2}, correction, projected) &&
              close(correction, {1e150, 0}) && close(projected, {0, 2}),
          "像归一化与物理方向尺度必须同步，不能把 U 当作欧氏单位向量");
}

void check_krylov_history_solves() {
  for (const std::string linear : {"qdldl", "normal_augmented", "factorhighs"}) {
    for (const std::string accuracy : {"strict", "blockwise"}) {
      solver::SolveOptions options;
      options.ipm.linear_solver = linear;
      options.ipm.linear_accuracy = accuracy;
      options.ipm.eliminate_singletons = true;
      options.ipm.equilibrate_kkt = true;
      options.ipm.krylov_history_size = 4;
      optimal_mode(one_variable(-2, -3, 5, 1, -inf, 2), -4, options);
      optimal_mode(one_variable(3, -inf, inf, -2, -4, -4), 6, options);
      optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    }
  }
  solver::Model coupled;
  const double epsilon = std::ldexp(1.0, -16);
  coupled.costs = {1, 3}; coupled.column_lower = {0, 0}; coupled.column_upper = {2, 2};
  coupled.kinds.assign(2, solver::VariableKind::continuous);
  coupled.row_lower = {2, 2 + epsilon}; coupled.row_upper = coupled.row_lower;
  coupled.column_start = {0, 2, 4}; coupled.row_index = {0, 1, 0, 1};
  coupled.values = {1, 1, 1, 1 + epsilon};
  for (const std::string linear : {"normal_augmented", "qdldl"}) {
    solver::SolveOptions active;
    active.ipm.linear_solver = linear;
    active.ipm.linear_accuracy = "blockwise";
    active.ipm.eliminate_singletons = true;
    active.ipm.equilibrate_kkt = true;
    active.ipm.direction_retries = 8;
    active.ipm.krylov_history_size = 4;
    const auto result = optimal_mode(coupled, 4, active);
    require(*result.ipm_krylov_recycle_projections > 0,
            "近相关解析约束必须实际执行历史投影，不能只测试未触发的入池路径");
    if (linear == "qdldl") {
      require(*result.ipm_krylov_recycle_acceptances > 0,
              "解析例必须覆盖投影后由原真实残差直接通过的路径");
    }
  }
  solver::SolveOptions options;
  auto model = one_variable(1, 0, inf, 1, 2, inf);
  for (int bad : {-1, 9}) {
    options.ipm.krylov_history_size = bad;
    require(solver::solve_ipm(model, options).status == "error", "回收内存预算不能越过公开范围");
  }
  options.ipm.krylov_history_size = 4;
  for (const std::string accuracy : {"backward", "inexact", "blockwise_weighted", "blockwise_forcing",
      "blockwise_forcing_weighted", "adaptive_regularization", "primal_dual_regularization", "curvature_regularization"}) {
    options.ipm.linear_accuracy = accuracy;
    require(solver::solve_ipm(model, options).status == "error",
            "尚未定义回收度量和验收规则的组合必须明确拒绝");
  }
  options.ipm.linear_accuracy = "blockwise";
  options.time_limit = 0;
  const auto timed = solver::solve_ipm(model, options);
  require(timed.status == "time_limit" && *timed.ipm_krylov_recycle_vectors == 0 &&
              *timed.ipm_krylov_recycle_projections == 0 && *timed.ipm_factorizations == 0,
          "零时间预算不能授权历史搜索、存储或分解");
  require(recycle_vectors > 0, "成功解析例必须实际覆盖已有方向入池");
}

void check_adaptive_regularization() {
  using solver::detail::ipm_min_regularization_level;
  using solver::detail::ipm_regularization_at_level;
  using solver::detail::ipm_regularization_search_level;
  const int minimum = ipm_min_regularization_level();
  const double epsilon = std::numeric_limits<double>::epsilon();
  require(minimum < 0 && ipm_regularization_at_level(minimum) >= epsilon &&
              ipm_regularization_at_level(minimum) / 10 < epsilon,
          "负正则网格必须由机器精度下界导出");
  double legacy = std::pow(epsilon, 0.75);
  for (int level = 0; level < 8; ++level) {
    require(ipm_regularization_at_level(level) == legacy, "原非负层级必须逐位保持旧十倍重试网格");
    legacy *= 10;
  }
  const std::vector<double> rd{1e-9}, dx{1000}, zero{0};
  const int selected = ipm_regularization_search_level(0, 0, 1e-9, rd, dx);
  const long double target = 1e-12L + solver::detail::ipm_inexact_forcing * rd[0];
  require(selected < 0 && ipm_regularization_at_level(selected) * 1000 <= target &&
              ipm_regularization_at_level(selected + 1) * 1000 > target,
          "应搜索到首个达到估计偏移目标的更小格点");
  require(ipm_regularization_search_level(0, 1, 1e-9, rd, dx) == 1,
          "同一矩阵刚失败的较低层级不得重复搜索");
  require(ipm_regularization_search_level(0, 0, 1e-12, rd, dx) == 0,
          "真实缺陷已消除时不能仅靠正则偏移估计触发");
  require(ipm_regularization_search_level(0, 0, 1, rd, zero) == 0,
          "预计偏移不足以解释缺陷时不降低正则量");
  require(ipm_regularization_search_level(0, 0, 1, zero, dx) == minimum &&
              ipm_regularization_search_level(minimum, minimum, 1, zero, dx) == minimum,
          "零对偶残差保留绝对地板，搜索不得越过机器精度格点");
  require(ipm_regularization_search_level(0, 0, 0, {}, {}) == 0, "空系统不得触发搜索");
  for (int level : {minimum - 1, 8}) {
    bool rejected = false;
    try { ipm_regularization_at_level(level); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "非法正则层级必须拒绝");
  }
  for (int bad_case = 0; bad_case < 6; ++bad_case) {
    bool rejected = false;
    try {
      if (bad_case == 0) ipm_regularization_search_level(1, 0, 0, rd, dx);
      if (bad_case == 1) ipm_regularization_search_level(0, 0, 0, rd, {});
      if (bad_case == 2) ipm_regularization_search_level(0, 0, -1, rd, dx);
      if (bad_case == 3) ipm_regularization_search_level(0, 0, inf, rd, dx);
      if (bad_case == 4) ipm_regularization_search_level(0, 0, 1, std::vector<double>{inf}, dx);
      if (bad_case == 5) ipm_regularization_search_level(0, 0, 1, rd,
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "搜索不能接受无效数据或倒置的分解层级");
  }
  for (const std::string linear : {"qdldl", "normal", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined"}) {
    if ((linear == "normal_extended" || linear == "normal_refined") && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.linear_accuracy = "adaptive_regularization";
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
  }
  solver::SolveOptions options;
  options.ipm.linear_solver = "normal_augmented";
  options.ipm.linear_accuracy = "adaptive_regularization";
  options.ipm.eliminate_singletons = true;
  options.ipm.equilibrate_kkt = true;
  bool exercised = false;
  // 自由变量的一维解析族用于确认搜索、重分解与恢复实际经过，不依赖公开实例。
  for (double bound : {1e2, 1e4, 1e6, 1e8}) {
    const auto model = one_variable(1e-4, -inf, inf, 1, bound, inf);
    optimal_mode(model, 1e-4 * bound, options);
    const auto result = solver::solve_ipm(model, options);
    exercised |= *result.ipm_regularization_reductions > 0;
    require(*result.ipm_factorizations >= *result.ipm_iterations + 1 + *result.ipm_regularization_searches,
            "实际重分解成本必须进入分解计数");
  }
  require(exercised, "解析尺度族必须实际覆盖降正则搜索并通过原模型验收");
}

void check_primal_dual_regularization() {
  using solver::detail::ipm_regularization_pair;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const auto initial = ipm_regularization_pair(0, true);
  require(initial[0] == std::pow(epsilon, 0.75) && initial[1] == std::sqrt(epsilon),
          "分离起点必须由同一个机器精度导出");
  for (int level = solver::detail::ipm_min_regularization_level(); level < 8; ++level) {
    const auto coupled = ipm_regularization_pair(level, false);
    const auto separate = ipm_regularization_pair(level, true);
    require(coupled[0] == coupled[1] && coupled[0] == solver::detail::ipm_regularization_at_level(level),
            "默认成对规则必须保留旧网格的逐位值");
    require(separate[0] == coupled[0] && separate[1] > separate[0] &&
                separate[1] / separate[0] == initial[1] / initial[0],
            "两项保持明确的不同量级，不随问题身份变化");
  }
  for (int level : {solver::detail::ipm_min_regularization_level() - 1, 8}) {
    bool rejected = false;
    try { ipm_regularization_pair(level, true); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "分离正则不能越过原分解重试层级预算");
  }
  for (const std::string linear : {"qdldl", "sparselu", "normal", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined"}) {
    if ((linear == "normal_extended" || linear == "normal_refined") &&
        std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.linear_accuracy = "primal_dual_regularization";
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
    optimal_mode(one_variable(-2, -3, 5, 1, -inf, 2), -4, options);
  }
}

void check_curvature_regularization() {
  using solver::detail::ipm_has_positive_curvature;
  require(ipm_has_positive_curvature(std::vector<long double>{1, 1e-20L, 1e20L}),
          "不同量级的已有正曲率可用于零原始偏移尝试");
  for (const auto& diagonal : std::vector<std::vector<long double>>{{}, {0}, {1, 0}, {1, -1},
      {std::numeric_limits<long double>::infinity()}, {std::numeric_limits<long double>::quiet_NaN()},
      {static_cast<long double>(std::numeric_limits<double>::denorm_min()) / 4},
      {static_cast<long double>(std::numeric_limits<double>::max()) * 2}}) {
    require(!ipm_has_positive_curvature(diagonal), "空、零、负、非有限或不能存入 double 的曲率不得触发");
  }
  for (const std::string linear : {"qdldl", "sparselu", "normal", "factorhighs", "normal_augmented",
                                  "normal_multifrontal", "normal_extended", "normal_refined"}) {
    if ((linear == "normal_extended" || linear == "normal_refined") &&
        std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.linear_accuracy = "curvature_regularization";
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    const auto bounded = optimal_mode(one_variable(-2, -3, 5, 1, -inf, 2), -4, options);
    require(*bounded.ipm_unshifted_factorizations > 0 && *bounded.ipm_primal_regularization == 0,
            "有界解析例必须实际使用已有曲率，并通过完整原模型验算");
    const auto free = optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
    require(*free.ipm_unshifted_factorizations == 0, "自由变量等式无单行曲率，须使用原有正则");
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
  }
}

void check_singleton_presolve() {
  using solver::detail::ipm_exact_bound_quotient;
  if constexpr (!solver::detail::ipm_exact_bound_checks_available) {
    require(!ipm_exact_bound_quotient(6, 3), "不支持的浮点表示应保留原始行");
    return;
  }
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double large = std::numeric_limits<double>::max();
  for (double sign_a : {-1.0, 1.0}) {
    for (double sign_b : {-1.0, 1.0}) {
      require(ipm_exact_bound_quotient(sign_b * 6, sign_a * 3) == sign_a * sign_b * 2,
              "精确商须正确覆盖两种系数和端点符号");
    }
    require(ipm_exact_bound_quotient(0, sign_a * 3) == 0 &&
                ipm_exact_bound_quotient(-0.0, sign_a * 3) == 0,
            "正负零端点表示同一数学界");
  }
  require(!ipm_exact_bound_quotient(1, 3) &&
              !ipm_exact_bound_quotient(1, 1 + std::numeric_limits<double>::epsilon()),
          "非精确商和先舍入回乘产生的假相等均不能归并");
  require(ipm_exact_bound_quotient(tiny, tiny) == 1 && ipm_exact_bound_quotient(2 * tiny, tiny) == 2 &&
              ipm_exact_bound_quotient(large, large) == 1,
          "次正规和最大有限值的精确比值不能依赖扩展乘法范围");
  require(!ipm_exact_bound_quotient(tiny, large) && !ipm_exact_bound_quotient(1, tiny),
          "候选商下溢或溢出时必须保留原行");
  for (double invalid : {0.0, -0.0, inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
    require(!ipm_exact_bound_quotient(1, invalid), "无效或零系数必须拒绝转换");
  }
  for (double invalid : {inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
    require(!ipm_exact_bound_quotient(invalid, 1), "有限端点助手不能处理非有限端点");
  }

  for (const std::string linear : {"qdldl", "normal_augmented", "factorhighs"}) {
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    options.ipm.presolve_singleton_rows = true;
    for (double coefficient : {-3.0, 3.0}) {
      for (double cost : {-1.0, 1.0}) {
        const double lower = std::min(2 * coefficient, 4 * coefficient);
        const double upper = std::max(2 * coefficient, 4 * coefficient);
        auto model = one_variable(cost, -inf, inf, coefficient, lower, upper);
        const auto result = optimal_mode(model, cost > 0 ? 2 : -4, options);
        require(*result.ipm_presolved_singleton_rows == 1 &&
                    std::abs(result.row_dual[0] - cost / coefficient) < 1e-6,
                "归并的双边行必须把乘子按原系数恢复到原行");
        model.sense = solver::ObjectiveSense::maximize;
        const auto maximum = optimal_mode(model, cost > 0 ? 4 : -2, options);
        require(*maximum.ipm_presolved_singleton_rows == 1 &&
                    std::abs(maximum.row_dual[0] + cost / coefficient) < 1e-6,
                "最大化行对偶仍遵守转为最小化后的符号约定");
      }
    }
    // 两个不同来源合成固定点，必须按自由乘子符号选择正确的一侧。
    auto fixed = one_variable(3, -inf, inf, 1, 2, inf);
    fixed.row_lower = {2, -inf}; fixed.row_upper = {inf, 2};
    fixed.column_start = {0, 2}; fixed.row_index = {0, 1}; fixed.values = {1, 1};
    for (double cost : {-3.0, 3.0}) {
      fixed.costs[0] = cost;
      const auto result = optimal_mode(fixed, cost * 2, options);
      require(*result.ipm_presolved_singleton_rows == 2 && *result.ipm_eliminated_fixed_variables == 0 &&
                  (cost > 0 ? result.row_dual[0] > 2.999999 && result.row_dual[1] == 0 :
                              result.row_dual[0] == 0 && result.row_dual[1] < -2.999999),
              "新固定点不能假装已消列，单边原行乘子不能分配错误符号");
    }
    auto ties = fixed;
    ties.costs = {1}; ties.row_lower = {2, 2}; ties.row_upper = {inf, inf};
    auto result = optimal_mode(ties, 2, options);
    require(result.row_dual[0] > 0.999999 && result.row_dual[1] == 0,
            "等值行界优先稳定的原行次序，其余冗余行乘子为零");
    ties.column_lower[0] = 2;
    result = optimal_mode(ties, 2, options);
    require(result.row_dual == std::vector<double>({0, 0}), "等值原列界优先，列乘子由原模型重新计算");

    const auto rounded = optimal_mode(one_variable(1, 0, 2, 3, 1, inf), 1.0 / 3, options);
    require(*rounded.ipm_presolved_singleton_rows == 0, "不可精确表示的商不得改变输入界");
    auto duplicate = one_variable(1, 0, 3, 1, 2, inf);
    duplicate.column_start = {0, 2}; duplicate.row_index = {0, 0}; duplicate.values = {2, -1};
    const auto unmerged = optimal_mode(duplicate, 2, options);
    require(*unmerged.ipm_presolved_singleton_rows == 0, "未可靠合并的重复项必须交给原规范化路径");
    auto empty_row = one_variable(1, 2, 3, 0, -1, 1);
    require(*optimal_mode(empty_row, 2, options).ipm_presolved_singleton_rows == 0,
            "空行不能误记为单非零归并");

    auto conflict = one_variable(1, 0, 1, 1, 2, inf);
    options.ipm.max_iterations = 20;
    const auto failed = solver::solve_ipm(conflict, options);
    require(*failed.ipm_presolved_singleton_rows == 0 && failed.status != "optimal" &&
                failed.status != "infeasible" && failed.status != "unbounded",
            "冲突列须保留原约束，不能借归并输出尚未实现的证书");
    options.ipm.max_iterations = 200;
    options.time_limit = 0;
    const auto expired = solver::solve_ipm(ties, options);
    require(expired.status == "time_limit" && *expired.ipm_presolved_singleton_rows == 0 &&
                *expired.ipm_factorizations == 0, "零预算不得执行归并或数值分解");
  }
  solver::Model shifted;
  shifted.costs = {2, 3}; shifted.column_lower = {1, -inf}; shifted.column_upper = {1, inf};
  shifted.kinds.assign(2, solver::VariableKind::continuous);
  shifted.row_lower = {3}; shifted.row_upper = {inf};
  shifted.column_start = {0, 1, 2}; shifted.row_index = {0, 0}; shifted.values = {1, 1};
  solver::SolveOptions options;
  options.ipm.eliminate_fixed_variables = true;
  options.ipm.presolve_singleton_rows = true;
  const auto result = optimal_mode(shifted, 8, options);
  require(*result.ipm_eliminated_fixed_variables == 1 && *result.ipm_presolved_singleton_rows == 1 &&
              result.solution.size() == 2 && result.solution[0] == 1 &&
              std::abs(result.row_dual[0] - 3) < 1e-6,
          "固定变量代入后的单行必须仍恢复原列坐标、原行对偶和原目标");
}

void check_direction_retries() {
  solver::SolveOptions options;
  options.ipm.scaling_passes = 0;
  options.ipm.eliminate_singletons = true;
  options.ipm.equilibrate_kkt = true;
  // 已知最优 x=1；缩放关闭，刻意保留小约束系数与大目标系数的差异。
  const auto bounded = one_variable(1e6, 0, 2, 1e-8, 1e-8, inf);
  bool recovered = false;
  for (const std::string linear : {"qdldl", "normal_augmented"}) {
    options.ipm.linear_solver = linear;
    for (const std::string accuracy : {"strict", "primal_dual_regularization"}) {
      options.ipm.linear_accuracy = accuracy;
      options.ipm.direction_retries = 0;
      const auto baseline = solver::solve_ipm(bounded, options);
      options.ipm.direction_retries = 8;
      const auto result = optimal_mode(bounded, 1e6, options);
      if (*result.ipm_direction_retry_successes > 0) {
        recovered = true;
        require(baseline.status == "numerical_error" && *result.ipm_direction_retries > 0 &&
                    *result.ipm_factorizations >= *result.ipm_iterations + 1 + *result.ipm_direction_retries &&
                    *result.ipm_refinement_steps >= *baseline.ipm_refinement_steps,
                "自然恢复例须保留重试前分解和改进成本，且不能误称预算耗尽为最优");
      }
    }
  }
  require(recovered, "解析尺度构造必须实际覆盖方向失败后恢复，而非仅检查开关");

  // 同一解析例在 strict 可恢复，分块验收仍失败；重试不得暗中退回较宽条件。
  for (const std::string linear : {"qdldl", "normal_augmented"}) {
    options.ipm.linear_solver = linear;
    for (const std::string accuracy : {"blockwise", "blockwise_weighted"}) {
      options.ipm.linear_accuracy = accuracy;
      options.ipm.direction_retries = 0;
      const auto baseline = solver::solve_ipm(bounded, options);
      require(baseline.status == "numerical_error" && *baseline.ipm_iterations > 0,
              "分块反例须实际进入外层后在方向组失败");
      for (int budget : {1, 8}) {
        options.ipm.direction_retries = budget;
        const auto failed = solver::solve_ipm(bounded, options);
        require(failed.status == "numerical_error" && failed.solution == baseline.solution &&
                    failed.ipm_iterations == baseline.ipm_iterations &&
                    *failed.ipm_direction_retries == std::min(budget, 7) &&
                    *failed.ipm_direction_retry_successes == 0 &&
                    *failed.ipm_factorizations > *baseline.ipm_factorizations &&
                    *failed.ipm_refinement_steps > *baseline.ipm_refinement_steps &&
                    *failed.ipm_backward_error_acceptances == 0 && *failed.ipm_inexact_acceptances == 0,
                "分块重试须保留全部失败成本和未更新点，不能借其他验收模式宣称成功");
        require(accuracy == "blockwise_weighted" ?
                    *failed.ipm_weighted_krylov_steps > *baseline.ipm_weighted_krylov_steps :
                    *failed.ipm_weighted_krylov_steps == 0,
                "重试须继续请求的搜索度量，实际加权步骤必须计入成本");
      }
    }
  }

  options.ipm.linear_solver = "qdldl";
  options.ipm.linear_accuracy = "strict";
  const auto equality = one_variable(-1, -inf, inf, 1e-8, 1e-8, 1e-8);
  options.ipm.direction_retries = 0;
  const auto partial = solver::solve_ipm(equality, options);
  require(partial.status == "numerical_error" && *partial.ipm_iterations == 0 &&
              partial.solution.size() == 1 && partial.solution[0] > 0,
          "关闭重试时保留原初始化第二次求解失败的报告行为");
  options.ipm.linear_accuracy = "bounded_newton";
  const auto bounded_initialization = solver::solve_ipm(equality, options);
  require(bounded_initialization.status == partial.status && bounded_initialization.message == partial.message &&
              bounded_initialization.solution == partial.solution &&
              bounded_initialization.ipm_iterations == partial.ipm_iterations &&
              bounded_initialization.ipm_factorizations == partial.ipm_factorizations &&
              bounded_initialization.ipm_refinement_steps == partial.ipm_refinement_steps &&
              *bounded_initialization.ipm_bounded_newton_returns == 0 &&
              *bounded_initialization.ipm_bounded_newton_inexact_returns == 0 &&
              *bounded_initialization.ipm_full_newton_products == 0 &&
              *bounded_initialization.ipm_full_newton_preconditioner_solves == 0,
          "有界方向模式必须保留 strict 初始化失败轨迹，不能提前放行初始化方向");
  options.ipm.method = "homogeneous";
  options.ipm.linear_accuracy = "strict";
  const auto homogeneous_initialization = solver::solve_ipm(equality, options);
  require(homogeneous_initialization.status == partial.status &&
              homogeneous_initialization.message == partial.message &&
              homogeneous_initialization.ipm_iterations == partial.ipm_iterations &&
              homogeneous_initialization.ipm_factorizations == partial.ipm_factorizations &&
              homogeneous_initialization.ipm_refinement_steps == partial.ipm_refinement_steps &&
              *homogeneous_initialization.ipm_homogeneous_constant_solves == 0 &&
              *homogeneous_initialization.ipm_homogeneous_direction_solves == 0 &&
              !homogeneous_initialization.ipm_homogeneous_tau && !homogeneous_initialization.ipm_homogeneous_kappa,
          "齐次初始化仍须严格失败且不记录尚未完成初始化的标量或外层求解");
  options.ipm.linear_accuracy = "homogeneous_scaled";
  const auto scaled_initialization = solver::solve_ipm(equality, options);
  require(scaled_initialization.status == homogeneous_initialization.status &&
              scaled_initialization.message == homogeneous_initialization.message &&
              scaled_initialization.solution == homogeneous_initialization.solution &&
              scaled_initialization.ipm_iterations == homogeneous_initialization.ipm_iterations &&
              scaled_initialization.ipm_factorizations == homogeneous_initialization.ipm_factorizations &&
              scaled_initialization.ipm_refinement_steps == homogeneous_initialization.ipm_refinement_steps &&
              *scaled_initialization.ipm_homogeneous_constant_solves == 0 &&
              *scaled_initialization.ipm_homogeneous_direction_solves == 0 &&
              *scaled_initialization.ipm_homogeneous_scaled_products == 0 &&
              *scaled_initialization.ipm_homogeneous_scaled_refinement_steps == 0 &&
              *scaled_initialization.ipm_homogeneous_scaled_inexact_returns == 0 &&
              !scaled_initialization.ipm_homogeneous_tau && !scaled_initialization.ipm_homogeneous_kappa,
          "缩放策略必须保持严格齐次初始化失败轨迹，不能提前产生缩放方向或齐次标量");
  options.ipm.linear_accuracy = "homogeneous_repair";
  const auto repair_initialization = solver::solve_ipm(equality, options);
  require(repair_initialization.status == homogeneous_initialization.status &&
              repair_initialization.message == homogeneous_initialization.message &&
              repair_initialization.solution == homogeneous_initialization.solution &&
              repair_initialization.ipm_iterations == homogeneous_initialization.ipm_iterations &&
              repair_initialization.ipm_factorizations == homogeneous_initialization.ipm_factorizations &&
              repair_initialization.ipm_refinement_steps == homogeneous_initialization.ipm_refinement_steps &&
              *repair_initialization.ipm_homogeneous_constant_solves == 0 &&
              *repair_initialization.ipm_homogeneous_direction_solves == 0 &&
              *repair_initialization.ipm_homogeneous_constant_repairs == 0 &&
              *repair_initialization.ipm_homogeneous_constant_repair_successes == 0 &&
              !repair_initialization.ipm_homogeneous_tau && !repair_initialization.ipm_homogeneous_kappa,
          "常数恢复模式必须保留 strict 初始化失败轨迹，不能恢复尚未出现的齐次标量故障");
  options.ipm.method = "mehrotra";
  for (int budget : {1, 8}) {
    options.ipm.linear_accuracy = "strict";
    options.ipm.direction_retries = budget;
    const auto failed = solver::solve_ipm(equality, options);
    require(failed.status == "numerical_error" && *failed.ipm_iterations == 0 &&
                failed.solution == std::vector<double>{0} && *failed.ipm_direction_retry_successes == 0 &&
                *failed.ipm_direction_retries == std::min(budget, 7) &&
                *failed.ipm_factorizations == *failed.ipm_direction_retries + 1,
            "初始化双求解未成功时不得提交半成品，达到预算或最高层级须停止");
    for (const std::string accuracy : {"blockwise", "blockwise_weighted", "full_newton"}) {
      options.ipm.linear_accuracy = accuracy;
      const auto block_failed = solver::solve_ipm(equality, options);
      require(block_failed.status == failed.status && block_failed.message == failed.message &&
                  block_failed.solution == failed.solution && block_failed.ipm_iterations == failed.ipm_iterations &&
                  block_failed.ipm_direction_retries == failed.ipm_direction_retries &&
                  block_failed.ipm_direction_retry_successes == failed.ipm_direction_retry_successes &&
                  block_failed.ipm_factorizations == failed.ipm_factorizations &&
                  block_failed.ipm_refinement_steps == failed.ipm_refinement_steps &&
                  block_failed.ipm_primal_regularization == failed.ipm_primal_regularization &&
                  block_failed.ipm_dual_regularization == failed.ipm_dual_regularization &&
                  *block_failed.ipm_weighted_krylov_steps == 0 &&
                  *block_failed.ipm_full_newton_preconditioner_solves == 0 &&
                  *block_failed.ipm_full_newton_products == 0,
              "分块及完整 Newton 模式的初始化重试必须保持 strict 轨迹，不得引入新搜索");
    }
  }
  options.ipm.linear_accuracy = "curvature_regularization";
  const auto exhausted = solver::solve_ipm(one_variable(1, -inf, inf, 1e-8, 1e-8, inf), options);
  require(exhausted.status == "numerical_error" && *exhausted.ipm_iterations == 0 &&
              *exhausted.ipm_unshifted_factorizations == 1 && *exhausted.ipm_direction_retries == 8 &&
              *exhausted.ipm_factorizations == 9 && *exhausted.ipm_direction_retry_successes == 0 &&
              *exhausted.ipm_primal_regularization == solver::detail::ipm_regularization_at_level(7),
          "零偏移之后只进入一次旧网格，不重复已知失败层级，最高格点失败必须返回");
}

void check_blockwise_residual() {
  using solver::detail::ipm_kkt_block_tolerances;
  using solver::detail::ipm_kkt_blocks_satisfied;
  const std::vector<double> rhs{1e-9, -2e-9, 1e6, -3e5};
  const auto tolerances = ipm_kkt_block_tolerances(rhs, 2);
  const std::vector<double> masked{2e-9, 0, 1e-7, 0};
  require(2e-9 < 1e-12 + 1e-10 * 1e6 && !ipm_kkt_blocks_satisfied(masked, 2, tolerances),
          "大原始右端不能掩盖对偶块中比其右端还大的缺陷");
  const auto forcing = ipm_kkt_block_tolerances(rhs, 2, 0.1);
  require(!ipm_kkt_blocks_satisfied(masked, 2, forcing),
          "较宽 forcing 仍须拒绝被全局门槛掩盖的对偶误差");
  const std::vector<double> useful{1e-10, 0, 1e-7, 0};
  require(ipm_kkt_blocks_satisfied(useful, 2, forcing) &&
              !ipm_kkt_blocks_satisfied(useful, 2, tolerances),
          "forcing 允许足以降低块残差但未达到固定高精度的方向");
  require(ipm_kkt_blocks_satisfied(std::vector<double>{forcing[0], 0, forcing[1], 0}, 2, forcing) &&
              !ipm_kkt_blocks_satisfied(std::vector<double>{std::nextafter(forcing[0], inf), 0, 0, 0}, 2, forcing),
          "forcing 仍需按块核对真实含等号边界");
  require(ipm_kkt_block_tolerances(std::vector<double>{0, 0}, 1, 0.1) ==
              std::array<double, 2>{1e-12, 1e-12}, "forcing 零右端不能删除绝对地板");
  for (double bad : {0.0, -0.1, 1.0, inf, std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try { ipm_kkt_block_tolerances(rhs, 2, bad); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "forcing 系数必须有限且严格位于零与一之间");
  }
  require(ipm_kkt_blocks_satisfied(std::vector<double>{tolerances[0], -tolerances[0], tolerances[1], -tolerances[1]},
              2, tolerances), "两块分别在含等号边界内可以接受");
  require(!ipm_kkt_blocks_satisfied(std::vector<double>{0, 0, std::nextafter(tolerances[1], inf), 0},
              2, tolerances), "原始块也必须独立满足精度门槛");
  const auto zero = ipm_kkt_block_tolerances(std::vector<double>{0, 0}, 1);
  require(zero[0] == 1e-12 && zero[1] == 1e-12 &&
              !ipm_kkt_blocks_satisfied(std::vector<double>{2e-12, 0}, 1, zero),
          "零右端沿用既有绝对门槛，不能借用另一块的尺度");
  require(ipm_kkt_blocks_satisfied({}, 0, ipm_kkt_block_tolerances({}, 0)), "两个空块必须安全接受");
  require(ipm_kkt_blocks_satisfied(std::vector<double>{0}, 0, ipm_kkt_block_tolerances(std::vector<double>{1}, 0)) &&
              ipm_kkt_blocks_satisfied(std::vector<double>{0}, 1, ipm_kkt_block_tolerances(std::vector<double>{1}, 1)),
          "单侧空块不能越界");
  for (double bad : {inf, -inf, std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try { ipm_kkt_block_tolerances(std::vector<double>{bad}, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !ipm_kkt_blocks_satisfied(std::vector<double>{bad}, 1, zero),
            "非有限右端和缺陷均不能被接受");
  }
  bool invalid_split = false;
  try { ipm_kkt_block_tolerances({}, 1); }
  catch (const std::invalid_argument&) { invalid_split = true; }
  require(invalid_split, "非法分块维数必须拒绝");
  bool invalid_tolerance = false;
  try { ipm_kkt_blocks_satisfied(std::vector<double>{0}, 1, {inf, 1e-12}); }
  catch (const std::invalid_argument&) { invalid_tolerance = true; }
  require(invalid_tolerance, "不能以非有限门槛绕过检查");
  for (const std::string linear : {"qdldl", "normal", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined"}) {
    if ((linear == "normal_extended" || linear == "normal_refined") && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.linear_accuracy = "blockwise";
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
    options.ipm.linear_accuracy = "blockwise_weighted";
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
    options.ipm.direction_retries = 8;
    for (const std::string accuracy : {"blockwise", "blockwise_weighted"}) {
      options.ipm.linear_accuracy = accuracy;
      options.ipm.central_corrections = 1;
      optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
      optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
      options.time_limit = 0;
      const auto expired = solver::solve_ipm(one_variable(1, 0, 2, 1, 1, inf), options);
      require(expired.status == "time_limit" && *expired.ipm_factorizations == 0 &&
                  *expired.ipm_direction_retries == 0 && *expired.ipm_weighted_krylov_steps == 0,
              "分块重试和中心校正不能绕过零时间预算");
      options.time_limit = solver::SolveOptions{}.time_limit;
    }
    options.ipm.direction_retries = 0;
    options.ipm.central_corrections = 0;
    options.ipm.linear_accuracy = "blockwise_forcing";
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
    options.ipm.linear_accuracy = "blockwise_forcing_weighted";
    optimal_mode(one_variable(1e-4, 0, inf, 1, 1e6, inf), 100, options);
    optimal_mode(one_variable(3, -inf, inf, 1, 2, 2), 6, options);
  }
}

void check_normal_budget_fallback() {
  // 一列连接大量等式，原始矩阵稀疏，但会超过两种 CHOLMOD 路径共同保留的 Gram 结构贡献预算。
  constexpr std::int64_t rows = 4'000;
  const double inf = std::numeric_limits<double>::infinity();
  auto model = one_variable(1, -inf, inf, 1, 1, 1);
  model.row_lower.assign(rows, 1);
  model.row_upper.assign(rows, 1);
  model.column_start = {0, rows};
  model.row_index.resize(rows);
  model.values.assign(rows, 1);
  for (std::int64_t row = 0; row < rows; ++row) model.row_index[static_cast<std::size_t>(row)] = row;
  solver::SolveOptions options;
  options.ipm.linear_solver = "normal";
  options.ipm.eliminate_singletons = true;
  options.ipm.equilibrate_kkt = true;
  const auto explicit_normal = solver::solve_ipm(model, options);
  require(explicit_normal.status == "error" && explicit_normal.ipm_structural_fallbacks == 0 &&
              !explicit_normal.ipm_factor_backend && explicit_normal.ipm_factorizations == 0,
          "显式 normal 必须保留结构超限语义，不能自动切换");
  for (const std::string mode : {"normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined", "normal_cholmod", "normal_cholmod_aat"}) {
    if ((mode == "normal_extended" || mode == "normal_refined") && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    options.ipm.linear_solver = mode;
    const auto fallback = solver::solve_ipm(model, options);
    require(fallback.status == "optimal" && fallback.ipm_structural_fallbacks == 1 &&
                fallback.ipm_factor_backend == "factorhighs" && fallback.ipm_kkt_dimension == rows + 1 &&
                fallback.ipm_factorizations > 0,
            "结构预算模式必须一次切换至增广内核并恢复最优点：" + fallback.status + "；" + fallback.message);
    require(fallback.solution.size() == 1 && std::abs(fallback.solution[0] - 1) < 1e-7 &&
                fallback.row_dual.size() == rows, "增广恢复不能丢失原模型对偶维数");
    const auto check = solver::verify_lp_solution(model, fallback.solution, fallback.row_dual,
        options.tolerances, options.pdlp.optimality);
    require(check.primal.feasible && check.optimality.numerically_optimal,
            "切换后仍必须通过原模型共同验收");
    for (bool zero_time : {false, true}) {
      auto stopped = options;
      if (zero_time) stopped.time_limit = 0;
      else stopped.ipm.max_iterations = 0;
      const auto result = solver::solve_ipm(model, stopped);
      require(result.status == (zero_time ? "time_limit" : "iteration_limit") &&
                  result.ipm_structural_fallbacks == 0 && result.ipm_factorizations == 0 &&
                  !result.ipm_factor_backend, "零预算不得启动结构分析或回退分解");
    }
    options.ipm.scaling_passes = 0;
    const auto numerical = solver::solve_ipm(one_variable(1, -inf, inf, 1e200, 1, inf), options);
    require(numerical.status == "numerical_error" && numerical.ipm_structural_fallbacks == 0 &&
                numerical.ipm_factor_backend == (mode == "normal_augmented" ? "normal" : mode), "非结构数值失败不能触发备用分解");
    if (mode == "normal_cholmod" || mode == "normal_cholmod_aat") {
      auto invalid = options;
      invalid.ipm.eliminate_singletons = false;
      require(solver::solve_ipm(model, invalid).status == "error", "CHOLMOD 正规方程要求显式单行消元");
      invalid.ipm.eliminate_singletons = true;
      invalid.ipm.dynamic_regularization = true;
      require(solver::solve_ipm(model, invalid).status == "error", "CHOLMOD 不允许静默接受未定义的动态正则组合");
    }
  }
}

void check_fixed_variables() {
  solver::Model model;
  model.costs = {4, 1, -2};
  model.objective_offset = 11;
  model.column_lower = {-2, 0, 3};
  model.column_upper = {-2, 10, 3};
  model.kinds.assign(3, solver::VariableKind::continuous);
  model.row_lower = {-3, 1};
  model.row_upper = {inf, 1};
  model.column_start = {0, 2, 3, 5};
  model.row_index = {0, 1, 0, 0, 1};
  model.values = {2, 1, 1, -1, 1};
  solver::SolveOptions options;
  options.ipm.eliminate_fixed_variables = true;
  for (const std::string backend : {"qdldl", "sparselu", "normal", "qdldl_extended", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined", "normal_cholmod", "normal_cholmod_aat"}) {
    if ((backend == "normal_extended" || backend == "normal_refined") && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    if (backend == "qdldl_extended" && std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) continue;
    options.ipm.linear_solver = backend;
    for (bool equilibration : {false, true}) {
      options.ipm.equilibrate_kkt = equilibration;
      options.ipm.eliminate_singletons = true;
      for (const auto sense : {solver::ObjectiveSense::minimize, solver::ObjectiveSense::maximize}) {
        model.sense = sense;
        optimal_mode(model, sense == solver::ObjectiveSense::minimize ? 1 : 7, options);
        const auto result = solver::solve_ipm(model, options);
        require(result.solution.size() == 3 && result.solution[0] == -2 && result.solution[2] == 3 &&
                    result.ipm_eliminated_fixed_variables == 2 && result.row_dual.size() == 2,
                "固定值及非连续列映射必须完整恢复，不能丢失原行");
        require(result.model.costs == model.costs && result.model.objective_offset == 11 &&
                    result.model.values == model.values, "消元不能修改原模型或重复计入固定目标常数");
      }
    }
  }
  options = {};
  options.ipm.eliminate_fixed_variables = true;
  auto all_fixed = one_variable(3, -2, -2, 2, -4, -4);
  all_fixed.objective_offset = 5;
  optimal_mode(all_fixed, -1, options);
  const auto fixed_result = solver::solve_ipm(all_fixed, options);
  require(fixed_result.ipm_eliminated_fixed_variables == 1 && fixed_result.solution == std::vector<double>{-2},
          "所有变量固定且行变空时应恢复实际点并通过原模型检查");
  all_fixed.row_lower = {-3}; all_fixed.row_upper = {-3};
  options.ipm.max_iterations = 5;
  const auto inconsistent = solver::solve_ipm(all_fixed, options);
  require(inconsistent.status != "optimal" && inconsistent.status != "infeasible" &&
              inconsistent.status != "unbounded", "不一致空行不能删去或冒充已获得不可行证书");
  const auto almost_fixed = solver::solve_ipm(one_variable(1, 1, std::nextafter(1.0, 2.0), 0, -inf, inf), options);
  require(almost_fixed.ipm_eliminated_fixed_variables == 0, "仅相近的上下界不能当作固定变量");
  for (double coefficient : {1e100, 1e300}) {
    const auto unsafe = solver::solve_ipm(one_variable(0, 1e100, 1e100, coefficient, 0, 1), options);
    require(unsafe.ipm_eliminated_fixed_variables == 0 && unsafe.status != "optimal",
            "位移溢出或使非零行区间坍缩时应保留原表示，不能饱和成新数据");
  }
  options.time_limit = 0;
  const auto timed = solver::solve_ipm(model, options);
  require(timed.status == "time_limit" && timed.ipm_eliminated_fixed_variables == 0 && !timed.has_solution,
          "零时间预算不能通过固定变量预处理伪造候选");
  options.time_limit = 10; options.ipm.max_iterations = 0;
  const auto limited = solver::solve_ipm(model, options);
  require(limited.status == "iteration_limit" && limited.ipm_eliminated_fixed_variables == 0 && !limited.has_solution,
          "零迭代预算必须保留既有语义");
  model.kinds[0] = solver::VariableKind::integer;
  require(solver::solve_ipm(model, options).status == "error", "固定整数列也不能绕过连续 LP 支持边界");
}

void check_presolved_fixed_variables() {
  if constexpr (!solver::detail::ipm_exact_bound_checks_available) return;
  for (const std::string linear : {"qdldl", "normal_augmented", "factorhighs"}) {
    solver::SolveOptions options;
    options.ipm.linear_solver = linear;
    options.ipm.eliminate_singletons = true;
    options.ipm.equilibrate_kkt = true;
    options.ipm.presolve_singleton_rows = true;
    options.ipm.eliminate_presolved_fixed_variables = true;
    for (const auto sense : {solver::ObjectiveSense::minimize, solver::ObjectiveSense::maximize}) {
      for (double cost : {-3.0, 0.0, 3.0}) {
        for (double coefficient : {-2.0, 2.0}) {
          auto model = one_variable(cost, -inf, inf, coefficient,
                                    coefficient > 0 ? 2 * coefficient : -inf,
                                    coefficient > 0 ? inf : 2 * coefficient);
          model.sense = sense;
          model.objective_offset = 7;
          model.row_lower.push_back(coefficient > 0 ? -inf : 2 * coefficient);
          model.row_upper.push_back(coefficient > 0 ? 2 * coefficient : inf);
          model.column_start = {0, 2}; model.row_index = {0, 1};
          model.values = {coefficient, coefficient};
          const auto result = optimal_mode(model, 7 + cost * 2, options);
          const double signed_cost = sense == solver::ObjectiveSense::minimize ? cost : -cost;
          require(result.solution == std::vector<double>{2} &&
                      *result.ipm_presolved_fixed_columns == 1 &&
                      *result.ipm_eliminated_fixed_variables == 1 &&
                      *result.ipm_presolved_singleton_rows == 2 && *result.ipm_factorizations == 0,
                  "全消列应恢复原点与目标常数，并在零分解下由原模型终验成功");
          require(result.row_dual[signed_cost > 0 ? 0 : 1] == signed_cost / coefficient &&
                      result.row_dual[signed_cost > 0 ? 1 : 0] == 0,
                  "固定点的自由乘子必须按目标方向及原系数分配到正确原行");
        }
      }
    }
    for (double cost : {-3.0, 3.0}) {
      auto lower_column = one_variable(cost, 2, inf, -2, -4, inf);
      auto result = optimal_mode(lower_column, cost * 2, options);
      require(*result.ipm_presolved_fixed_columns == 1 &&
                  result.row_dual[0] == (cost > 0 ? 0 : cost / -2),
              "原列下界优先时，正乘子必须留在原列界而非转给上界原行");
      auto upper_column = one_variable(cost, -inf, 2, -2, -inf, -4);
      result = optimal_mode(upper_column, cost * 2, options);
      require(*result.ipm_presolved_fixed_columns == 1 &&
                  result.row_dual[0] == (cost > 0 ? cost / -2 : 0),
              "原列上界优先时，负乘子必须留在原列界而非转给下界原行");
    }
    // 原固定列在两端，新固定列在中间；保留行的乘子会改变新固定列的待恢复乘子。
    solver::Model coupled;
    coupled.costs = {2, 3, 1, 4};
    coupled.column_lower = {2, -inf, 0, -1};
    coupled.column_upper = {2, inf, 12, -1};
    coupled.kinds.assign(4, solver::VariableKind::continuous);
    coupled.row_lower = {10, 13}; coupled.row_upper = {10, inf};
    coupled.column_start = {0, 1, 3, 4, 5};
    coupled.row_index = {0, 0, 1, 1, 1}; coupled.values = {3, 2, 4, 1, 5};
    coupled.objective_offset = 11;
    options.ipm.eliminate_fixed_variables = true;
    solver::Model scattered;
    scattered.costs = {3, 5, 1, 7, 2};
    scattered.column_lower = {0, 1, -inf, 4, 0};
    scattered.column_upper = {inf, 1, inf, 4, inf};
    scattered.kinds.assign(5, solver::VariableKind::continuous);
    scattered.row_lower = {5, 5}; scattered.row_upper = {5, inf};
    scattered.column_start = {0, 1, 2, 4, 4, 4};
    scattered.row_index = {1, 0, 0, 1}; scattered.values = {1, 1, 2, 1};
    const auto scattered_result = optimal_mode(scattered, 44, options);
    require(*scattered_result.ipm_eliminated_fixed_variables == 3 &&
                *scattered_result.ipm_presolved_fixed_columns == 1 &&
                scattered_result.solution[1] == 1 && scattered_result.solution[2] == 2 &&
                scattered_result.solution[3] == 4 && std::abs(scattered_result.solution[0] - 3) < 1e-6 &&
                std::abs(scattered_result.solution[4]) < 1e-6 &&
                std::abs(scattered_result.row_dual[0] + 1) < 1e-6,
            "两层映射必须保留原坐标不连续的多个活变量，并恢复残差乘子");
    solver::Model all_layers;
    all_layers.costs = {3, 4}; all_layers.column_lower = {1, -inf};
    all_layers.column_upper = {1, inf}; all_layers.kinds.assign(2, solver::VariableKind::continuous);
    all_layers.row_lower = {5}; all_layers.row_upper = {5};
    all_layers.column_start = {0, 1, 2}; all_layers.row_index = {0, 0}; all_layers.values = {1, 2};
    const auto all_layers_result = optimal_mode(all_layers, 11, options);
    require(all_layers_result.solution == std::vector<double>({1, 2}) &&
                *all_layers_result.ipm_eliminated_fixed_variables == 2 &&
                *all_layers_result.ipm_presolved_fixed_columns == 1 &&
                *all_layers_result.ipm_factorizations == 0 && all_layers_result.row_dual[0] == 2,
            "内外层全部消空时，空列映射不能被误认为恒等映射并丢失固定值");
    for (const auto sense : {solver::ObjectiveSense::minimize, solver::ObjectiveSense::maximize}) {
      coupled.sense = sense;
      const bool minimum = sense == solver::ObjectiveSense::minimize;
      const auto result = optimal_mode(coupled, minimum ? 27 : 29, options);
      require(*result.ipm_eliminated_fixed_variables == 3 && *result.ipm_presolved_fixed_columns == 1 &&
                  result.solution.size() == 4 && result.solution[0] == 2 && result.solution[1] == 2 &&
                  result.solution[3] == -1 && std::abs(result.solution[2] - (minimum ? 10 : 12)) < 1e-6,
              "内外两层消列必须组合非连续原列映射，不能覆盖已有固定值");
      require(std::abs(result.row_dual[0] - (minimum ? -0.5 : -1.5)) < 1e-6 &&
                  std::abs(result.row_dual[1] - (minimum ? 1 : 0)) < 1e-6,
              "新固定列对偶恢复必须减去保留原行贡献，并保留原始行编号");
      require(result.model.values == coupled.values && result.model.costs == coupled.costs &&
                  result.model.objective_offset == 11,
              "代入和恢复不得改变用于终验的输入模型");
    }
    options.ipm.eliminate_fixed_variables = false;
    solver::Model chain;
    chain.costs = {1, 1}; chain.column_lower = {-inf, -inf}; chain.column_upper = {inf, inf};
    chain.kinds.assign(2, solver::VariableKind::continuous);
    chain.row_lower = {1, 2}; chain.row_upper = {1, 2};
    chain.column_start = {0, 2, 3}; chain.row_index = {0, 1, 1}; chain.values = {1, 1, 1};
    const auto chain_result = optimal_mode(chain, 2, options);
    require(*chain_result.ipm_presolved_fixed_columns == 1 && *chain_result.ipm_presolved_singleton_rows == 1,
            "单轮消元不能级联归并代入后才出现的单行，并绕过独立来源恢复条件");
    auto already_fixed = one_variable(3, 2, 2, 1, -inf, inf);
    require(*optimal_mode(already_fixed, 6, options).ipm_presolved_fixed_columns == 0,
            "新选项不能擅自消去原本固定但旧固定消元未启用的列");
    auto rounded = one_variable(1, 0, 2, 3, 1, 1);
    require(*optimal_mode(rounded, 1.0 / 3, options).ipm_presolved_fixed_columns == 0,
            "不可精确表示的单行商不能借新固定消元绕过可靠性检查");
    options.ipm.max_iterations = 1;
    auto narrow = one_variable(1, -inf, inf, 1, 1, std::nextafter(1.0, 2.0));
    require(*solver::solve_ipm(narrow, options).ipm_presolved_fixed_columns == 0,
            "仅相近的归并界不得消为固定点");
    auto conflict = one_variable(1, 0, 1, 1, 2, inf);
    const auto failed = solver::solve_ipm(conflict, options);
    require(*failed.ipm_presolved_fixed_columns == 0 && failed.status != "optimal",
            "冲突归并不能产生虚假的固定点或成功状态");
  }
  solver::SolveOptions options;
  options.ipm.presolve_singleton_rows = true;
  options.ipm.eliminate_presolved_fixed_variables = true;
  options.ipm.max_iterations = 1;
  for (double coefficient : {1e100, 1e300}) {
    solver::Model unsafe;
    unsafe.costs = {0, 1}; unsafe.column_lower = {-inf, -inf}; unsafe.column_upper = {inf, inf};
    unsafe.kinds.assign(2, solver::VariableKind::continuous);
    unsafe.row_lower = {1e100, 0}; unsafe.row_upper = {1e100, 1};
    unsafe.column_start = {0, 2, 3}; unsafe.row_index = {0, 1, 1}; unsafe.values = {1, coefficient, 1};
    const auto result = solver::solve_ipm(unsafe, options);
    require(*result.ipm_presolved_fixed_columns == 0 && *result.ipm_eliminated_fixed_variables == 0 &&
                result.status != "optimal",
            "位移使保留行区间坍缩或溢出时，必须撤销新消列并保留原表示");
  }
  auto fixed = one_variable(1, -inf, inf, 1, 2, 2);
  options.time_limit = 0;
  auto result = solver::solve_ipm(fixed, options);
  require(result.status == "time_limit" && *result.ipm_presolved_fixed_columns == 0 && !result.has_solution,
          "零时间预算不能通过新消元绕过既有语义");
  options.time_limit = 10; options.ipm.max_iterations = 0;
  result = solver::solve_ipm(fixed, options);
  require(result.status == "iteration_limit" && *result.ipm_presolved_fixed_columns == 0 && !result.has_solution,
          "零迭代预算不能通过新消元生成未授权候选");
  options.ipm.presolve_singleton_rows = false;
  require(solver::solve_ipm(fixed, options).status == "error",
          "新固定消元必须明确依赖输入单行归并，不能静默忽略无效组合");
}

void check_inexact_direction() {
  const std::vector<std::int64_t> starts = {0, 1}, rows = {0};
  const std::vector<double> values = {1}, zero = {0}, primal = {1}, dual = {0.5};
  const std::vector<double> direction = {-0.9375, -0.46875};
  int budget_checks = 0;
  require(solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, dual,
              zero, direction, [&budget_checks] { ++budget_checks; }),
          "两块实际缺陷均小于各自残差十分之一时可替代接受");
  const std::vector<double> h = {1}, rhs = {-0.5, -0.53125};
  std::vector<double> residual;
  solver::detail::assess_ipm_kkt_residual(starts, rows, values, h, rhs, direction, 2, residual);
  require(std::abs(residual[0]) > 1e-12 + 1e-10 && std::abs(residual[1]) > 1e-12 + 1e-10,
          "解析替代候选必须确实无法通过旧严格线性门槛");
  require(budget_checks >= 3, "额外稀疏扫描须检查时间预算");
  const std::vector<double> bad_dual = {-0.9375, -0.4};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, dual,
              zero, bad_dual), "一块合格不能掩盖另一块缺陷过大");
  const std::vector<double> infeasible_direction = {0, -1e-6};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, zero,
              zero, infeasible_direction),
          "巨大 h/z 右端不能掩盖原始块不收缩或已可行对偶块漂移");
  const std::vector<double> exact_direction = {-1, 0}, zero_pair = {0, 0};
  require(solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, zero,
              zero, exact_direction), "零对偶块实际缺陷为零时仍允许另一块收缩");
  require(solver::detail::accept_ipm_inexact_direction(starts, rows, values, zero, zero,
              zero, zero_pair), "两个零块只接受计算缺陷同为零的方向");

  // H=double(1/3) 的 K0 下块恰为零，实际 ds=-2^52 后仍有 -1/4 缺陷。
  const double magnitude = std::ldexp(1.0, 52);
  const std::vector<double> thirds = {1.0 / 3.0}, negative = {-magnitude};
  const std::vector<double> large_direction = {3 * magnitude, 3 * magnitude};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, thirds, zero, negative,
              negative, large_direction),
          "必须检查实际恢复的 ds，不能用舍入 H 下的虚假零原始缺陷接受");
  const std::vector<double> nonfinite = {inf, 0};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, zero,
              zero, nonfinite), "非有限方向不得通过 max 或比较进入接受分支");
  const std::vector<double> nan_value = {std::numeric_limits<double>::quiet_NaN()};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, nan_value, primal, zero,
              zero, exact_direction), "非有限乘积不得被忽略");
  const std::vector<double> tiny = {std::numeric_limits<double>::denorm_min()}, half = {0.5, 0};
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, tiny, zero, zero,
              zero, half), "不能把非零极小乘积下溢误判为零块精确满足");
  bool expired = false;
  try {
    solver::detail::accept_ipm_inexact_direction(starts, rows, values, primal, dual,
        zero, direction, [] { throw std::runtime_error("测试预算"); });
  } catch (const std::exception&) { expired = true; }
  require(expired, "预算异常必须透传，不能作为普通拒绝吞掉");
}

void check_inexact_prefilter() {
  const std::vector<std::int64_t> starts = {0, 1}, rows = {0};
  const std::vector<double> one = {1}, zero = {0}, one_direction = {0, 1};
  const auto scalar = solver::detail::make_ipm_inexact_prefilter(starts, one);
  prefilter_available = !scalar.column_absolute_sums.empty();
  const std::vector<long double> product = {1};
  const auto rejects = solver::detail::reject_ipm_inexact_dual(scalar, product, one, one);
  require(!prefilter_available || rejects, "明显超出对偶块门槛时应省去完整 helper");
  require(!solver::detail::accept_ipm_inexact_direction(starts, rows, one, zero, one, zero,
              one_direction), "可预拒绝候选的旧 actual helper 必须也拒绝");
  const std::vector<double> negative = {-1};
  require(!solver::detail::reject_ipm_inexact_dual(scalar, product, negative, one),
          "真实零对偶缺陷不能被预拒绝");
  const std::vector<double> boundary_residual = {1}, boundary_direction = {-0.9};
  const std::vector<long double> boundary_product = {static_cast<long double>(boundary_direction[0])};
  require(!solver::detail::reject_ipm_inexact_dual(scalar, boundary_product, boundary_residual,
              boundary_direction), "接近门槛的候选必须由原 helper 判断");

  const std::vector<std::int64_t> cancellation_starts = {0, 3}, cancellation_rows = {0, 1, 2};
  const std::vector<double> cancellation_values = {1, -1, std::numeric_limits<double>::epsilon()};
  const std::vector<double> large_direction = {std::ldexp(1.0, 52), std::ldexp(1.0, 52), 1};
  const auto cancellation = solver::detail::make_ipm_inexact_prefilter(cancellation_starts,
      cancellation_values);
  const std::vector<long double> tiny_product = {std::numeric_limits<double>::epsilon()};
  require(!solver::detail::reject_ipm_inexact_dual(cancellation, tiny_product, zero, large_direction),
          "强相消使误差界含糊时不得直接预拒绝零块");
  const std::vector<long double> nonfinite_product = {std::numeric_limits<long double>::infinity()};
  require(!solver::detail::reject_ipm_inexact_dual(scalar, nonfinite_product, one, one),
          "非有限缓存必须回原验收路径");
  const std::vector<double> nonfinite_values = {inf};
  const auto unreliable = solver::detail::make_ipm_inexact_prefilter(starts, nonfinite_values);
  require(!solver::detail::reject_ipm_inexact_dual(unreliable, product, one, one),
          "无法建立有限上界时不能预拒绝");
  require(!solver::detail::reject_ipm_inexact_dual({}, product, one, one),
          "不支持平台的空预拒绝数据必须回退");
  bool expired = false;
  try {
    solver::detail::reject_ipm_inexact_dual(scalar, product, one, one,
        [] { throw std::runtime_error("测试预算"); });
  } catch (const std::exception&) { expired = true; }
  require(expired, "预拒绝检查不能吞掉预算异常");

  // 固定种子覆盖不同指数、相消、零块和门槛附近；判定对照生产 v12 helper。
  std::mt19937_64 random(0x591a38);
  int definitely_rejected = 0, originally_accepted = 0;
  for (int sample = 0; sample < 512; ++sample) {
    constexpr int n = 3, m = 6;
    std::vector<std::int64_t> random_starts = {0}, random_rows;
    std::vector<double> values;
    std::vector<double> direction(n + m, 0), dual(n), primal(m, 0), slack(m, 0);
    for (int row = 0; row < m; ++row) {
      direction[n + row] = std::ldexp(static_cast<double>(static_cast<int>(random() % 2047) - 1023),
          static_cast<int>(random() % 901) - 450);
    }
    for (int col = 0; col < n; ++col) {
      for (int row = 0; row < m; ++row) {
        random_rows.push_back(row);
        values.push_back(std::ldexp(static_cast<double>(static_cast<int>(random() % 2047) - 1023),
            static_cast<int>(random() % 901) - 450));
      }
      random_starts.push_back(static_cast<std::int64_t>(values.size()));
    }
    std::vector<long double> cached(n, 0);
    for (int col = 0; col < n; ++col) {
      for (auto k = random_starts[col]; k < random_starts[col + 1]; ++k) {
        cached[col] += static_cast<long double>(values[k]) * direction[n + random_rows[k]];
      }
      dual[col] = sample % 3 == 0 ? 0 : static_cast<double>(-cached[col] / (sample % 3 == 1 ? 1 : 1.1L));
    }
    const auto filter = solver::detail::make_ipm_inexact_prefilter(random_starts, values);
    const bool rejected = solver::detail::reject_ipm_inexact_dual(filter, cached, dual,
        std::span<const double>(direction).subspan(n));
    const bool accepted = solver::detail::accept_ipm_inexact_direction(random_starts, random_rows,
        values, primal, dual, slack, direction);
    require(!(rejected && accepted), "随机及门槛相消对照不得出现预拒绝误删原接受方向");
    definitely_rejected += rejected;
    originally_accepted += accepted;
  }
  require(originally_accepted > 0 && (!prefilter_available || definitely_rejected > 0),
          "对照必须同时覆盖原 helper 接受与实际预拒绝");
  for (double magnitude : {std::numeric_limits<double>::denorm_min(),
                            std::numeric_limits<double>::max()}) {
    const std::vector<double> extreme = {magnitude}, full_direction = {0, magnitude};
    const std::vector<long double> extreme_product = {static_cast<long double>(magnitude) * magnitude};
    const auto filter = solver::detail::make_ipm_inexact_prefilter(starts, extreme);
    const bool rejected = solver::detail::reject_ipm_inexact_dual(filter, extreme_product, zero, extreme);
    const bool accepted = solver::detail::accept_ipm_inexact_direction(starts, rows, extreme, zero,
        zero, zero, full_direction);
    require(!(rejected && accepted), "极小与极大乘积对照不得误拒原 helper 接受方向");
  }
}

void check_centrality() {
  const std::vector<double> one = {1}, zero = {0}, descending = {-2}, ascending = {2};
  const std::vector<bool> cone = {true};
  auto trial = solver::detail::ipm_centrality_target(one, one, descending, zero, cone, 0.1);
  require(trial.target.size() == 1 && std::abs(trial.target[0] - 1.01) < 1e-14,
          "试探步长必须沿 HiPO 的 max 规则允许越界，不得截为 min");
  require(trial.primal_step < 0.5 && trial.primal_step > 0.499999999999 && trial.dual_step == 1,
          "内部边界只用机器精度内缩，不提前乘最终步长系数");
  const std::vector<double> primal_correction = {1.01};
  auto weights = solver::detail::ipm_centrality_weights(one, one, descending, zero,
      primal_correction, zero, cone, trial.primal_step, trial.dual_step);
  require(weights.primal_weight == 1 && weights.dual_weight == 0,
          "原始侧改善时只接受原始权重，并覆盖九点搜索的末端");

  const std::vector<double> pair = {1, 1}, pair_zero = {0, 0}, dual_direction = {-2, 2};
  const std::vector<bool> pair_cone = {true, true};
  trial = solver::detail::ipm_centrality_target(pair, pair, pair_zero, dual_direction, pair_cone, 0.1);
  require(std::abs(trial.target[0] - 1.01) < 1e-14 && std::abs(trial.target[1] + 1) < 1e-14,
          "中心校正上侧负调整须截断为负的区间上界");
  const std::vector<double> primal_pair_correction = {0.005, 0.005};
  const std::vector<double> dual_pair_correction = {1.005, -1.005};
  weights = solver::detail::ipm_centrality_weights(pair, pair, pair_zero, dual_direction,
      primal_pair_correction, dual_pair_correction, pair_cone, trial.primal_step, trial.dual_step);
  require(weights.primal_weight == 0 && weights.dual_weight == 1,
          "对偶侧单独改善时不能把原始校正也加入原方向");

  trial = solver::detail::ipm_centrality_target(one, one, zero, descending, cone, 0.1);
  weights = solver::detail::ipm_centrality_weights(one, one, zero, descending,
      primal_correction, zero, cone, trial.primal_step, trial.dual_step);
  require(weights.primal_weight == 0 && weights.dual_weight == 0,
          "纯中心方向未改善任一受阻侧时必须全部拒绝");
  trial = solver::detail::ipm_centrality_target(one, one, descending, ascending, cone, 0);
  require(trial.target.empty(), "中心目标为零时不能造目标下限");
  trial = solver::detail::ipm_centrality_target(one, one, zero, zero, cone, 0.1);
  require(trial.target.empty(), "两侧已允许完整步长时不需要额外校正");
  const std::vector<bool> equality = {false};
  trial = solver::detail::ipm_centrality_target(one, one, descending, descending, equality, 0.1);
  require(trial.target.empty(), "等式行不能成为中心校正的非负锥阻挡项");
  trial = solver::detail::ipm_centrality_target(one, one, descending, descending, cone, 0.1);
  require(trial.target.size() == 1 && trial.target[0] == 0,
          "两侧试探为负但乘积在中心区间内时不得人为改写试探点");
  bool expired = false;
  int calls = 0;
  try {
    solver::detail::ipm_centrality_weights(one, one, descending, zero, primal_correction, zero,
        cone, 0.5, 1, [&calls] { if (++calls == 3) throw std::runtime_error("测试预算"); });
  } catch (const std::exception&) { expired = true; }
  require(expired && calls == 3, "权重搜索必须在候选之间响应预算结束");
  bool rejected = false;
  try {
    const std::vector<double> invalid_correction = {inf};
    solver::detail::ipm_centrality_weights(one, one, descending, zero, invalid_correction, zero,
        cone, 0.5, 1);
  } catch (const std::exception&) { rejected = true; }
  require(rejected, "非有限可选方向必须被拒绝，不能授权合成");
}

void check_backward_residual() {
  const double magnitude = std::ldexp(1.0, 53);
  const std::vector<std::int64_t> starts = {0, 2, 4}, rows = {0, 1, 0, 1};
  const std::vector<double> values = {1, 1, 1, 1 + std::numeric_limits<double>::epsilon()};
  const std::vector<double> h = {0, 0, 0};
  const std::vector<double> point = {magnitude, -magnitude, magnitude, -magnitude, 0};
  const std::vector<double> rhs = {1, -1, 1, -1, 0};
  std::vector<double> residual;
  int budget_checks = 0;
  auto error = solver::detail::assess_ipm_kkt_residual(starts, rows, values, h, rhs, point,
      3, residual, [&budget_checks] { ++budget_checks; });
  require(residual == std::vector<double>({1, 1, 1, 1, 0}),
          "完整 KKT 应保留抵消后的单位残差与可靠的零方程");
  require(error > 0 && error <= solver::detail::ipm_backward_error_tolerance &&
              residual[0] > 1e-12 + 1e-10,
          "分量后向误差应识别单位残差已达到大方向的舍入量级");
  require(budget_checks == 4, "后向误差检查必须覆盖开始、各列和结束预算");
  auto inaccurate_rhs = rhs;
  inaccurate_rhs[0] = 10;
  error = solver::detail::assess_ipm_kkt_residual(starts, rows, values, h, inaccurate_rhs,
      point, 3, residual);
  require(error > solver::detail::ipm_backward_error_tolerance,
          "明显超过舍入量级的分量误差必须拒绝");

  const std::vector<std::int64_t> scalar_starts = {0, 1}, scalar_rows = {0};
  const std::vector<double> scalar_values = {1}, scalar_h = {2}, scalar_point = {5, 3};
  const std::vector<double> scalar_rhs = {3, -1};
  error = solver::detail::assess_ipm_kkt_residual(scalar_starts, scalar_rows, scalar_values,
      scalar_h, scalar_rhs, scalar_point, 2, residual);
  require(error == 0 && residual == std::vector<double>({0, 0}),
          "后向误差必须包括完整未正则化 H 对角");

  const std::vector<double> tiny_values = {std::numeric_limits<double>::min()};
  const std::vector<double> tiny_point = {std::numeric_limits<double>::min(), 0};
  const std::vector<double> zero_pair = {0, 0}, zero_h = {0};
  error = solver::detail::assess_ipm_kkt_residual(scalar_starts, scalar_rows, tiny_values,
      zero_h, zero_pair, tiny_point, 1, residual);
  require(error > solver::detail::ipm_backward_error_tolerance,
          "非零乘积下溢或极小分母不能伪装成可靠零残差");
  error = solver::detail::assess_ipm_kkt_residual(scalar_starts, scalar_rows, tiny_values,
      zero_h, zero_pair, zero_pair, 1, residual);
  require(error == 0, "可靠的零分母零残差必须允许通过");

  const std::vector<double> tiny_rhs = {0, std::numeric_limits<double>::denorm_min()};
  error = solver::detail::assess_ipm_kkt_residual(scalar_starts, scalar_rows, scalar_values,
      zero_h, tiny_rhs, zero_pair, 1, residual);
  require(error > solver::detail::ipm_backward_error_tolerance,
          "最小正分母必须使用 LAPACK 安全常数而不能被舍入忽略");

  if constexpr (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent) {
    const std::vector<double> huge_values(4, 1e308), huge_point = {10, -10, 10, -10};
    const std::vector<double> huge_h(2, 0), huge_rhs(4, 1);
    error = solver::detail::assess_ipm_kkt_residual(starts, rows, huge_values, huge_h,
        huge_rhs, huge_point, 2, residual);
    require(std::isfinite(error) && error > 0 && error < 1e-300L,
            "大于 double 范围的分母须保留扩展精度，不能转成无穷制造零比值");
  }
  bool rejected = false;
  try {
    const std::vector<double> nonfinite_values = {inf};
    solver::detail::assess_ipm_kkt_residual(scalar_starts, scalar_rows, nonfinite_values,
        zero_h, zero_pair, scalar_point, 1, residual);
  } catch (const std::exception&) { rejected = true; }
  require(rejected, "非有限原方程不能获得后向误差授权");
  rejected = false;
  try {
    solver::detail::assess_ipm_kkt_residual(starts, rows, values, h, rhs, point,
        3, residual, [] { throw std::runtime_error("测试预算已耗尽"); });
  } catch (const std::exception&) { rejected = true; }
  require(rejected, "预算结束必须中止检查，不能根据已有部分分量接受");
}

}  // 匿名命名空间

int main() {
  try {
    check_centrality();
    check_full_newton_algebra();
    check_full_newton_solves();
    check_bounded_newton_algebra();
    check_bounded_newton_solves();
    check_homogeneous_algebra();
    check_homogeneous_repair();
    check_homogeneous_scaled_algebra();
    check_homogeneous_scaled_refinement();
    check_objective_scaling();
    check_homogeneous_solves();
    check_homogeneous_solves("homogeneous_scaled");
    check_homogeneous_solves("homogeneous_repair");
    check_backward_residual();
    check_inexact_direction();
    check_fixed_variables();
    check_inexact_prefilter();
    check_normal_budget_fallback();
    check_adaptive_regularization();
    check_primal_dual_regularization();
    check_curvature_regularization();
    check_singleton_presolve();
    check_presolved_fixed_variables();
    check_krylov_history_algebra();
    check_krylov_history_solves();
    check_direction_retries();
    check_blockwise_residual();
    solver::Model model;
    model.costs = {1, 2};
    model.column_lower = {0, 0};
    model.column_upper = {2, 4};
    model.kinds.assign(2, solver::VariableKind::continuous);
    model.row_lower = {3};
    model.row_upper = {inf};
    model.column_start = {0, 1, 2};
    model.row_index = {0, 0};
    model.values = {1, 1};
    optimal(model, 4);
    solver::SolveOptions central_options;
    central_options.ipm.central_corrections = 1;
    const auto central_result = solver::solve_ipm(model, central_options);
    require(central_result.status == "optimal" && *central_result.ipm_central_correction_attempts > 0 &&
                *central_result.ipm_factorizations == *central_result.ipm_iterations + 1,
            "真实额外中心校正必须复用已有因子，每步仅一次分解并含一次初始化分解");
    auto maximum = model;
    maximum.sense = solver::ObjectiveSense::maximize;
    maximum.costs = {3, 2};
    maximum.row_lower = {-inf};
    maximum.row_upper = {4};
    maximum.column_upper = {2, 3};
    maximum.objective_offset = 7;
    optimal(maximum, 17);

    // 双边行、两类单边行、自由变量和等式的符号映射。
    optimal(one_variable(1, -inf, inf, 1, 2, 2), 2);
    optimal(one_variable(1, -inf, inf, 1, 2, 4), 2);
    optimal(one_variable(-1, -inf, inf, 1, 2, 4), -4);
    optimal(one_variable(1, -inf, inf, 1, 2, inf), 2);
    optimal(one_variable(-1, -inf, inf, 1, -inf, 4), -4);
    optimal(one_variable(1, -inf, inf, -1, -4, -2), 2);
    optimal(one_variable(3, 2, 2, 0, -inf, inf), 6);
    optimal(one_variable(-1, 0, 3, 0, -inf, inf), -3);
    optimal(one_variable(0, -inf, inf, 1, 2, 2), 0);
    optimal(one_variable(3, 0, 5, 10000, 20000, inf), 6);

    // 重复条目及线性相关等式不能要求满秩，也不能生成假的严格内点。
    auto duplicate = one_variable(1, -inf, inf, 1, 2, 2);
    duplicate.row_lower = {2, 4};
    duplicate.row_upper = duplicate.row_lower;
    duplicate.column_start = {0, 4};
    duplicate.row_index = {1, 0, 1, 0};
    duplicate.values = {3, 3, -1, -2};
    optimal(duplicate, 2);

    // 从已知原始/对偶 KKT 点构造远离零点、混合行域和量级的通用回归。
    bool forcing_search_exercised = false;
    for (int dimension : {12, 24}) {
      solver::Model distant;
      distant.costs.assign(dimension, 0);
      distant.column_lower.assign(dimension, -inf);
      distant.column_upper.assign(dimension, inf);
      distant.kinds.assign(dimension, solver::VariableKind::continuous);
      distant.row_lower.resize(dimension);
      distant.row_upper.resize(dimension);
      std::vector<double> exact(dimension), row_scale(dimension), column_scale(dimension);
      std::vector<double> rhs(dimension, 0), dual(dimension);
      for (int i = 0; i < dimension; ++i) {
        row_scale[i] = std::pow(10.0, i % 7 - 3);
        column_scale[i] = std::pow(10.0, i % 5 - 2);
        exact[i] = (i % 2 ? -1 : 1) * (100.0 + 17 * i) / column_scale[i];
        dual[i] = (i % 3 == 1 ? -1 : 1) * (1.0 + i) / row_scale[i];
      }
      distant.column_start = {0};
      for (int col = 0; col < dimension; ++col) {
        for (int row = col; row < dimension; ++row) {
          const double value = row_scale[row] * column_scale[col] * (row == col ? 1 : 0.125);
          distant.row_index.push_back(row);
          distant.values.push_back(value);
          rhs[row] += value * exact[col];
          distant.costs[col] += value * dual[row];
        }
        distant.column_start.push_back(static_cast<std::int64_t>(distant.values.size()));
      }
      double expected = 0;
      for (int row = 0; row < dimension; ++row) {
        distant.row_lower[row] = row % 3 == 1 ? -inf : rhs[row];
        distant.row_upper[row] = row % 3 == 0 ? inf : rhs[row];
        expected += distant.costs[row] * exact[row];
      }
      optimal(distant, expected);
      solver::SolveOptions weighted_options;
      weighted_options.ipm.linear_accuracy = "blockwise_weighted";
      weighted_options.ipm.equilibrate_kkt = true;
      optimal_mode(distant, expected, weighted_options);
      const auto weighted_result = solver::solve_ipm(distant, weighted_options);
      require(weighted_result.ipm_weighted_krylov_steps > 0,
              "混合行域和量级的已知 KKT 构造必须实际进入加权 Krylov");
      weighted_options.ipm.linear_accuracy = "blockwise_forcing_weighted";
      optimal_mode(distant, expected, weighted_options);
      const auto forcing_weighted_result = solver::solve_ipm(distant, weighted_options);
      forcing_search_exercised |= *forcing_weighted_result.ipm_weighted_krylov_steps > 0;
      weighted_options.ipm.linear_accuracy = "adaptive_regularization";
      optimal_mode(distant, expected, weighted_options);
      weighted_options.ipm.linear_accuracy = "primal_dual_regularization";
      optimal_mode(distant, expected, weighted_options);
      weighted_options.ipm.linear_accuracy = "curvature_regularization";
      optimal_mode(distant, expected, weighted_options);
      weighted_options.ipm.direction_retries = 8;
      for (const std::string accuracy : {"strict", "blockwise", "blockwise_weighted",
                                       "primal_dual_regularization", "curvature_regularization"}) {
        weighted_options.ipm.linear_accuracy = accuracy;
        const auto retried = optimal_mode(distant, expected, weighted_options);
        if (accuracy == "blockwise_weighted") {
          require(*retried.ipm_weighted_krylov_steps > 0,
                  "启用重试的解析 KKT 族也须实际使用加权搜索并通过原模型验算");
        }
      }
    }
    require(forcing_search_exercised, "解析 KKT 尺寸族必须实际覆盖 forcing 加权搜索并通过原模型验收");

    auto offset = one_variable(-1, 0, 3, 0, -inf, inf);
    offset.objective_offset = 1e30;
    optimal(offset, 1e30);
    const auto offset_result = solver::solve_ipm(offset);
    require(offset_result.solution[0] > 2.99999, "大目标常数不能掩盖未优化的候选");

    solver::SolveOptions unscaled;
    unscaled.ipm.scaling_passes = 0;
    optimal(model, 4, unscaled);
    optimal(one_variable(1, -inf, inf, -3, -12, -6), 2, unscaled);
    // LP 数据不预缩放时，等价单行约束跨越十二个数量级仍须按原模型验算。
    for (double magnitude : {1e-6, 1e6}) {
      optimal(one_variable(1, 0, 5, magnitude, 2 * magnitude, inf), 2, unscaled);
    }
    auto tiny_coupling = model;
    tiny_coupling.column_upper = {inf, inf};
    tiny_coupling.row_lower = {1};
    tiny_coupling.row_upper = {1};
    tiny_coupling.values = {1e300, 1e-200};
    auto equilibrated_unscaled = unscaled;
    equilibrated_unscaled.ipm.equilibrate_kkt = true;
    const auto underflow_result = solver::solve_ipm(tiny_coupling, equilibrated_unscaled);
    require(underflow_result.status == "numerical_error" && !underflow_result.has_solution &&
                underflow_result.message.find("KKT 对称均衡导致非零数据下溢") != std::string::npos,
            "均衡不得静默丢失有限矩阵中的非零耦合");
    equilibrated_unscaled.ipm.direction_retries = 8;
    const auto retry_underflow = solver::solve_ipm(tiny_coupling, equilibrated_unscaled);
    require(retry_underflow.status == underflow_result.status && retry_underflow.message == underflow_result.message &&
                *retry_underflow.ipm_direction_retries == 0,
            "分解装配中的均衡下溢不得被方向重试掩盖");

    // 必须在重复项合并后识别单非零；完全相消的空行仍由原方程保留。
    auto merged = one_variable(1, -inf, inf, 1, 2, inf);
    merged.row_lower = {2, -inf};
    merged.row_upper = {inf, 1};
    merged.column_start = {0, 4};
    merged.row_index = {1, 0, 1, 0};
    merged.values = {2, 4, -2, -3};
    optimal(merged, 2);
    solver::SolveOptions eliminated;
    eliminated.ipm.eliminate_singletons = true;
    const auto merged_result = solver::solve_ipm(merged, eliminated);
    require(merged_result.ipm_kkt_dimension == 2 && merged_result.ipm_eliminated_singletons == 1,
            "合并后的单非零行应消元，空行应保留");
    const auto bounds_result = solver::solve_ipm(one_variable(-1, 0, 3, 0, -inf, inf), eliminated);
    require(bounds_result.status == "optimal" && bounds_result.ipm_kkt_dimension == 1 &&
                bounds_result.ipm_eliminated_singletons == 2 && bounds_result.ipm_factor_nonzeros == 1,
            "纯变量界的两条不等式应精确压缩为一个标量分解");
    auto lu_eliminated = eliminated;
    lu_eliminated.ipm.linear_solver = "sparselu";
    const auto lu_bounds = solver::solve_ipm(one_variable(-1, 0, 3, 0, -inf, inf), lu_eliminated);
    require(lu_bounds.status == "optimal" && lu_bounds.ipm_kkt_dimension == 1 &&
                lu_bounds.ipm_factor_nonzeros == 2 && lu_bounds.ipm_eliminated_singletons == 2,
            "标量 LU 应报告 L/U 两个逻辑对角，且完整恢复已消元方向");
    auto normal_options = eliminated;
    normal_options.ipm.linear_solver = "normal";
    const auto normal_bounds = solver::solve_ipm(one_variable(-1, 0, 3, 0, -inf, inf), normal_options);
    require(normal_bounds.status == "optimal" && normal_bounds.ipm_kkt_dimension == 0 &&
                normal_bounds.ipm_factor_nonzeros == 0 && normal_bounds.ipm_eliminated_singletons == 2,
            "纯界消元后没有保留行，正规方程应报告零分解维数和零因子条目");
    const auto normal_equality = solver::solve_ipm(one_variable(1, -inf, inf, 1, 2, 2), normal_options);
    require(normal_equality.status == "optimal" && normal_equality.ipm_kkt_dimension == 1 &&
                normal_equality.ipm_factor_nonzeros == 1 && normal_equality.ipm_eliminated_singletons == 0,
            "等式仍保留为正规方程的一维系统，不能按不等式单行消去");
    const auto equality_result = solver::solve_ipm(one_variable(1, -inf, inf, 1, 2, 2), eliminated);
    require(equality_result.status == "optimal" && equality_result.ipm_kkt_dimension == 2 &&
                equality_result.ipm_eliminated_singletons == 0,
            "单非零等式不得按正锥行消元");
    const auto central_equality = solver::solve_ipm(one_variable(1, -inf, inf, 1, 2, 2), central_options);
    require(central_equality.status == "optimal" && central_equality.ipm_central_correction_attempts == 0 &&
                central_equality.ipm_central_correction_acceptances == 0,
            "没有不等式的真实求解不能创建中心校正右端");
    auto overflow_options = eliminated;
    overflow_options.ipm.scaling_passes = 0;
    const auto overflow_result = solver::solve_ipm(
        one_variable(1, -inf, inf, 1e200, 1, inf), overflow_options);
    require(overflow_result.status == "numerical_error" && !overflow_result.has_solution,
            "Schur 对角超出有限范围必须报告数值失败，不能静默删去贡献");
    auto zero_eliminated = eliminated;
    zero_eliminated.ipm.max_iterations = 0;
    require(solver::solve_ipm(model, zero_eliminated).status == "iteration_limit",
            "启用消元也必须遵守零迭代预算");
    solver::Model empty;
    empty.column_start = {0};
    empty.objective_offset = 9;
    optimal(empty, 9);

    solver::SolveOptions budget;
    budget.ipm.max_iterations = 0;
    auto result = solver::solve_ipm(model, budget);
    require(result.status == "iteration_limit" && !result.has_solution && !result.objective &&
                !result.best_bound && result.ipm_iterations == 0, "零迭代预算不能生成候选");
    budget.ipm.max_iterations = 200;
    budget.time_limit = 0;
    result = solver::solve_ipm(model, budget);
    require(result.status == "time_limit" && !result.has_solution && !result.objective,
            "零时间预算不能生成候选");
    auto central_budget = central_options;
    central_budget.time_limit = 0;
    const auto central_budget_result = solver::solve_ipm(model, central_budget);
    require(central_budget_result.status == "time_limit" && !central_budget_result.has_solution &&
                central_budget_result.ipm_central_correction_attempts == 0,
            "启用中心校正也必须遵守零时间预算");
    auto inexact_budget = budget;
    inexact_budget.ipm.linear_accuracy = "inexact";
    const auto inexact_budget_result = solver::solve_ipm(model, inexact_budget);
    require(inexact_budget_result.status == "time_limit" &&
                inexact_budget_result.ipm_inexact_acceptances == 0,
            "非精确模式零时间预算不能产生替代接受");
    auto normal_budget = normal_options;
    normal_budget.time_limit = 0;
    const auto normal_budget_result = solver::solve_ipm(model, normal_budget);
    require(normal_budget_result.status == "time_limit" && !normal_budget_result.has_solution &&
                normal_budget_result.ipm_factorizations == 0,
            "正规方程模式零预算不能进入初始化分解");
    budget.time_limit = 10;
    budget.ipm.max_iterations = 1;
    result = solver::solve_ipm(one_variable(-1, 0, 10, 0, -inf, inf), budget);
    require(result.status == "iteration_limit" && result.ipm_iterations == 1 &&
                result.has_solution && result.lp_optimality && !result.lp_optimality->numerically_optimal,
            "提前预算出口不能把非最优候选标成最优");
    budget.ipm.max_iterations = 30;
    for (const auto& impossible : {one_variable(1, 0, 1, 1, 2, inf),
                                   one_variable(-1, -inf, inf, 0, -inf, inf),
                                   one_variable(0, 0, 1, 0, 1, inf)}) {
      result = solver::solve_ipm(impossible, budget);
      require(result.status != "optimal" && result.status != "infeasible" && result.status != "unbounded",
              "未实现齐次嵌入证书时不得宣称最优、不可行或无界");
    }
    auto mip = model;
    mip.kinds[0] = solver::VariableKind::integer;
    require(solver::solve_ipm(mip).status == "error", "必须拒绝整数模型");
    solver::SolveOptions invalid;
    invalid.ipm.linear_solver = "unknown";
    require(solver::solve_ipm(model, invalid).status == "error", "必须拒绝未知线性求解器");
    invalid = {};
    invalid.ipm.linear_solver = "sparselu";
    invalid.ipm.dynamic_regularization = true;
    require(solver::solve_ipm(model, invalid).status == "error", "LU 不得静默忽略 QDLDL 专属动态主元正则");
    invalid.ipm.linear_solver = "qdldl_extended";
    require(solver::solve_ipm(model, invalid).status == "error", "扩展精度路径不得静默忽略动态主元正则");
    invalid.ipm.dynamic_regularization = false;
    if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
      require(solver::solve_ipm(model, invalid).status == "error", "平台无额外精度时不得冒充扩展精度");
    } else {
      invalid.time_limit = 0;
      const auto extended_budget = solver::solve_ipm(model, invalid);
      require(extended_budget.status == "time_limit" && extended_budget.ipm_factorizations == 0,
              "扩展精度路径零预算不能进行分解");
    }
    invalid = {};
    invalid.ipm.linear_solver = "normal";
    require(solver::solve_ipm(model, invalid).status == "error", "正规方程实验必须显式启用单行消元");
    invalid.ipm.eliminate_singletons = true;
    invalid.ipm.dynamic_regularization = true;
    require(solver::solve_ipm(model, invalid).status == "error", "正规方程不得静默忽略动态主元正则");
    invalid = {};
    invalid.threads = 2;
    require(solver::solve_ipm(model, invalid).status == "error", "必须拒绝尚未实现的线程设置");
    invalid = {};
    for (int count : {-1, 2}) {
      invalid.ipm.central_corrections = count;
      require(solver::solve_ipm(model, invalid).status == "error", "必须拒绝超出 0/1 的中心校正配置");
    }
    invalid = {};
    invalid.ipm.linear_accuracy = "loose";
    require(solver::solve_ipm(model, invalid).status == "error", "必须拒绝未知线性精度模式");
    for (int budget : {-1, 9}) {
      invalid = {};
      invalid.ipm.direction_retries = budget;
      require(solver::solve_ipm(model, invalid).status == "error", "直接 API 必须拒绝越界方向重试预算");
    }
    for (const std::string accuracy : {"backward", "inexact", "blockwise_forcing",
                                     "blockwise_forcing_weighted", "adaptive_regularization"}) {
      invalid = {};
      invalid.ipm.direction_retries = 1;
      invalid.ipm.linear_accuracy = accuracy;
      require(solver::solve_ipm(model, invalid).status == "error", "不支持的线性精度组合必须显式拒绝");
    }
    invalid = {};
    invalid.ipm.direction_retries = 8;
    invalid.time_limit = 0;
    const auto expired_retry = solver::solve_ipm(model, invalid);
    require(expired_retry.status == "time_limit" && *expired_retry.ipm_direction_retries == 0 &&
                *expired_retry.ipm_factorizations == 0,
            "耗尽的时间预算不能授权任何重试或分解");
    invalid = {};
    invalid.ipm.linear_accuracy = "inexact";
    invalid.ipm.central_corrections = 1;
    require(solver::solve_ipm(model, invalid).status == "error",
            "不能静默组合尚未定义总缺陷预算的非精确方向与额外中心校正");
    for (const std::string accuracy : {"strict", "backward", "blockwise", "blockwise_weighted", "blockwise_forcing", "blockwise_forcing_weighted"}) {
      invalid = {};
      invalid.ipm.linear_accuracy = accuracy;
      invalid.ipm.inexact_prefilter = true;
      require(solver::solve_ipm(model, invalid).status == "error",
              "非 inexact 模式不能静默忽略预拒绝选项");
    }
    invalid = {};
    invalid.ipm.scaling_passes = 21;
    require(solver::solve_ipm(model, invalid).status == "error", "必须检查缩放轮数");
    invalid = {};
    invalid.pdlp.optimality.dual_relative = -1;
    require(solver::solve_ipm(model, invalid).status == "error", "必须检查共同最优性容差");
    invalid = {};
    invalid.highs.solver = "ipm";
    require(solver::solve_ipm(model, invalid).status == "error", "直接 API 必须拒绝 HiGHS 专属算法选项");
    invalid = {};
    invalid.highs.run_crossover = "off";
    require(solver::solve_ipm(model, invalid).status == "error", "直接 API 必须拒绝 HiGHS 专属交叉选项");
    invalid = {};
    invalid.highs.presolve = "off";
    require(solver::solve_ipm(model, invalid).status == "error", "直接 API 必须拒绝 HiGHS 专属预求解选项");
    require(central_attempts > 0 && central_acceptances > 0, "解析 LP 必须实际覆盖额外校正尝试和接受路径");
    require(inexact_acceptances > 0, "解析 LP 必须实际覆盖 strict 失败后的分块方向接受");
    require(!prefilter_available || prefilter_rejections > 0, "解析 LP 必须实际省去部分完整 helper");
    std::cout << "自主稀疏内点法检查通过，共 " << checks << " 项；后向误差接受 "
              << backward_acceptances << " 次；中心校正尝试 " << central_attempts
              << " 次，接受 " << central_acceptances << " 次；非精确方向接受 "
              << inexact_acceptances << " 次；预拒绝 " << prefilter_rejections
              << " 次；历史入池 " << recycle_vectors << " 次，投影 " << recycle_projections
              << " 次，投影直接通过 " << recycle_acceptances
              << " 次；完整 Newton 解析改进 " << full_newton_test_refinements
              << " 步，其中 Krylov " << full_newton_test_krylov << " 步，预条件 "
              << full_newton_test_preconditions << " 次，乘积 " << full_newton_test_products
              << " 次；有界 Newton 解析改进 " << bounded_newton_test_refinements << " 步，预条件 "
              << bounded_newton_test_preconditions << " 次，乘积 " << bounded_newton_test_products
              << " 次；齐次常数求解 " << homogeneous_test_constant_solves << " 次，方向求解 "
              << homogeneous_test_direction_solves << " 次；齐次恢复解析尝试 " << homogeneous_repair_test_attempts
              << " 次，成功 " << homogeneous_repair_test_successes << " 次。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "自主稀疏内点法检查失败：" << error.what() << '\n';
    return 1;
  }
}
