#include "ipm-presolve.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
namespace candidate = solver_v87_presolve;
const double infinity = std::numeric_limits<double>::infinity();

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool same_model(const solver::Model& left, const solver::Model& right) {
  return std::tie(left.name, left.sense, left.objective_offset, left.costs, left.column_lower,
      left.column_upper, left.row_lower, left.row_upper, left.kinds, left.column_start,
      left.row_index, left.values) ==
      std::tie(right.name, right.sense, right.objective_offset, right.costs, right.column_lower,
      right.column_upper, right.row_lower, right.row_upper, right.kinds, right.column_start,
      right.row_index, right.values);
}

bool same_verification(const solver::Verification& a, const solver::Verification& b) {
  return std::tie(a.valid_input, a.feasible, a.objective_consistent, a.objective,
      a.max_row_violation, a.max_bound_violation, a.max_integrality_violation, a.errors) ==
      std::tie(b.valid_input, b.feasible, b.objective_consistent, b.objective,
      b.max_row_violation, b.max_bound_violation, b.max_integrality_violation, b.errors);
}

bool same_optimality(const solver::LpOptimality& a, const solver::LpOptimality& b) {
  return std::tie(a.valid_input, a.dual_feasible, a.gap_satisfied, a.numerically_optimal,
      a.max_dual_residual, a.dual_objective_estimate, a.corrected_dual_bound,
      a.objective_gap, a.relative_objective_gap, a.errors) ==
      std::tie(b.valid_input, b.dual_feasible, b.gap_satisfied, b.numerically_optimal,
      b.max_dual_residual, b.dual_objective_estimate, b.corrected_dual_bound,
      b.objective_gap, b.relative_objective_gap, b.errors);
}

// 对照原 core 的完整非计时载荷，包括尚未进入迭代的状态与所有 legacy 计数。
void same_non_timing(const solver::SolveResult& a, const solver::SolveResult& b) {
  require(same_model(a.model, b.model) && same_verification(a.verification, b.verification) &&
      a.lp_optimality.has_value() == b.lp_optimality.has_value() &&
      (!a.lp_optimality || same_optimality(*a.lp_optimality, *b.lp_optimality)),
      "关闭候选必须保留原模型与完整质量载荷");
#define V87_SAME(field) require(a.field == b.field, "非计时字段改变：" #field)
  V87_SAME(backend); V87_SAME(status); V87_SAME(backend_status); V87_SAME(backend_version);
  V87_SAME(message); V87_SAME(verification_level); V87_SAME(solution); V87_SAME(has_solution);
  V87_SAME(objective); V87_SAME(best_bound); V87_SAME(absolute_gap); V87_SAME(relative_gap);
  V87_SAME(nodes); V87_SAME(simplex_iterations); V87_SAME(pdlp_iterations);
  V87_SAME(pdlp_acceleration_attempts); V87_SAME(pdlp_acceleration_accepted); V87_SAME(pdlp_operator_passes);
  V87_SAME(ipm_iterations); V87_SAME(ipm_factorizations); V87_SAME(ipm_refinement_steps);
  V87_SAME(ipm_weighted_krylov_steps); V87_SAME(ipm_full_newton_preconditioner_solves);
  V87_SAME(ipm_full_newton_products); V87_SAME(ipm_bounded_newton_returns);
  V87_SAME(ipm_bounded_newton_inexact_returns); V87_SAME(ipm_homogeneous_constant_solves);
  V87_SAME(ipm_homogeneous_constant_repairs); V87_SAME(ipm_homogeneous_constant_repair_successes);
  V87_SAME(ipm_homogeneous_direction_solves); V87_SAME(ipm_homogeneous_scaled_products);
  V87_SAME(ipm_homogeneous_scaled_refinement_steps); V87_SAME(ipm_homogeneous_scaled_inexact_returns);
  V87_SAME(ipm_homogeneous_tau); V87_SAME(ipm_homogeneous_kappa);
  V87_SAME(ipm_regularization_searches); V87_SAME(ipm_regularization_reductions);
  V87_SAME(ipm_unshifted_factorizations); V87_SAME(ipm_direction_retries); V87_SAME(ipm_direction_retry_successes);
  V87_SAME(ipm_krylov_recycle_projections); V87_SAME(ipm_krylov_recycle_acceptances);
  V87_SAME(ipm_krylov_recycle_vectors); V87_SAME(ipm_krylov_recycle_peak_dimension);
  V87_SAME(ipm_primal_regularization); V87_SAME(ipm_dual_regularization);
  V87_SAME(ipm_schur_residual_products); V87_SAME(ipm_schur_refinement_steps);
  V87_SAME(ipm_schur_refinement_acceptances); V87_SAME(ipm_kkt_dimension); V87_SAME(ipm_factor_nonzeros);
  V87_SAME(ipm_factor_backend); V87_SAME(ipm_structural_fallbacks); V87_SAME(ipm_eliminated_singletons);
  V87_SAME(ipm_eliminated_fixed_variables); V87_SAME(ipm_presolved_singleton_rows);
  V87_SAME(ipm_presolved_fixed_columns); V87_SAME(ipm_regularized_pivots);
  V87_SAME(ipm_backward_error_acceptances); V87_SAME(ipm_inexact_acceptances);
  V87_SAME(ipm_inexact_prefilter_rejections); V87_SAME(ipm_central_correction_attempts);
  V87_SAME(ipm_central_correction_acceptances); V87_SAME(row_dual);
#undef V87_SAME
}

solver::SolveOptions policy() {
  solver::SolveOptions options;
  options.backend = "ipm";
  options.time_limit = 5;
  options.threads = 1;
  options.seed = 0;
  options.ipm.method = "homogeneous";
  options.ipm.linear_solver = "normal_cholmod_aat";
  options.ipm.linear_accuracy = "homogeneous_repair";
  options.ipm.objective_scaling = false;
  options.ipm.max_iterations = 200;
  options.ipm.scaling_passes = 5;
  options.ipm.eliminate_singletons = true;
  options.ipm.equilibrate_kkt = true;
  options.ipm.eliminate_fixed_variables = true;
  options.ipm.dynamic_regularization = false;
  options.ipm.presolve_singleton_rows = false;
  options.ipm.eliminate_presolved_fixed_variables = false;
  options.ipm.krylov_history_size = 0;
  options.ipm.central_corrections = 0;
  options.ipm.direction_retries = 0;
  options.ipm.inexact_prefilter = false;
  return options;
}

solver::Model scalar(double cost, double lower, double upper, double offset = 0,
                     bool maximum = false) {
  solver::Model model;
  model.name = "v87_component_scalar";
  model.sense = maximum ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
  model.objective_offset = offset;
  model.costs = {cost};
  model.column_lower = {lower}; model.column_upper = {upper};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  return model;
}

void row(solver::Model& model, double coefficient, double lower, double upper) {
  model.row_lower = {lower}; model.row_upper = {upper};
  model.column_start = {0, 1}; model.row_index = {0}; model.values = {coefficient};
}

solver::Model coupled(bool fixed = false, bool maximum = false) {
  solver::Model model;
  model.name = "v87_component_coupled";
  model.sense = maximum ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
  model.objective_offset = 11;
  const double cost = maximum ? 1.0 : -1.0;
  model.costs = {cost, cost}; model.column_lower = {0, 0};
  model.column_upper = {infinity, infinity};
  model.row_lower = {-infinity, -infinity}; model.row_upper = {4, 4};
  model.column_start = {0, 2, 4}; model.row_index = {0, 1, 0, 1}; model.values = {2, 1, 1, 2};
  if (fixed) {
    model.costs.insert(model.costs.begin(), 7);
    model.column_lower.insert(model.column_lower.begin(), 3);
    model.column_upper.insert(model.column_upper.begin(), 3);
    model.row_upper = {19, -2}; model.column_start = {0, 2, 4, 6};
    model.row_index = {0, 1, 0, 1, 0, 1}; model.values = {5, -2, 2, 1, 1, 2};
  }
  model.kinds.assign(model.costs.size(), solver::VariableKind::continuous);
  return model;
}

bool quality_ok(const solver::LpVerification& q) {
  return q.primal.valid_input && q.primal.feasible && q.primal.objective_consistent &&
      q.optimality.valid_input && q.optimality.dual_feasible && q.optimality.gap_satisfied &&
      q.optimality.numerically_optimal;
}

void original_quality(const solver::Model& model, const solver::SolveResult& result,
                      double objective) {
  if (!(result.status == "optimal" && result.has_solution && same_model(result.model, model))) {
    throw std::runtime_error("候选成功必须持有原模型坐标与原模型身份：status=" + result.status +
        ",has_solution=" + std::to_string(result.has_solution) + ",same_model=" +
        std::to_string(same_model(result.model, model)) + ",message=" + result.message);
  }
  const auto q = solver::verify_lp_solution(model, result.solution, result.row_dual);
  require(quality_ok(q) && result.verification.valid_input && result.verification.feasible &&
      result.verification.objective_consistent && result.lp_optimality && result.lp_optimality->numerically_optimal,
      "所有成功小例均须经共同精度的原模型完整验收");
  require(q.primal.objective && result.objective &&
      std::abs(*q.primal.objective - objective) <= 1e-6 &&
      std::abs(*result.objective - objective) <= 1e-6,
      "原方向目标、offset 与固定列贡献必须一致");
}

struct Mode {
  explicit Mode(bool value) : old(candidate::enabled_override), old_emit(candidate::emit_diagnostics) {
    candidate::enabled_override = value;
    candidate::emit_diagnostics = false;
  }
  ~Mode() { candidate::enabled_override = old; candidate::emit_diagnostics = old_emit; }
  std::optional<bool> old;
  bool old_emit;
};

struct Detail {
  const char* id;
  const char* group;
};
std::vector<Detail> details;
int wrapper_cases = 0, mapping_cases = 0, transaction_cases = 0, budget_cases = 0, exception_cases = 0;
// 组件调用指本 harness 进入真实 solve_ipm/v87_core 的次数，包含零预算入口；
// 它不是内部迭代或 child 调用拦截，synthetic core 回调另行统计。
int component_ipm_calls = 0, synthetic_core_calls = 0;

solver::SolveResult real_wrapper(const solver::Model& model, const solver::SolveOptions& options,
                                 bool enabled) {
  Mode mode(enabled);
  ++component_ipm_calls;
  return solver::solve_ipm(model, options);
}

solver::SolveResult real_core(const solver::Model& model, const solver::SolveOptions& options) {
  ++component_ipm_calls;
  return solver::v87_core(model, options, solver::Clock::now());
}

solver::SolveResult bait(const solver::Model& model, const char* status = "optimal") {
  auto result = candidate::make_shell(model);
  result.status = status; result.backend_status = "child_backend_status";
  result.message = "child_message"; result.verification_level = "child_quality";
  result.has_solution = true; result.solution = {101}; result.row_dual = {202};
  result.objective = 303; result.best_bound = 304; result.absolute_gap = 305; result.relative_gap = 306;
  result.verification.valid_input = true; result.verification.feasible = true;
  result.verification.objective_consistent = true; result.verification.objective = 307;
  result.lp_optimality.emplace(); result.lp_optimality->valid_input = true;
  result.lp_optimality->dual_feasible = true; result.lp_optimality->gap_satisfied = true;
  result.lp_optimality->numerically_optimal = true; result.lp_optimality->objective_gap = 308;
  result.ipm_iterations = 137; result.ipm_factorizations = 11;
  result.ipm_refinement_steps = 19; result.ipm_homogeneous_constant_solves = 139;
  result.ipm_homogeneous_direction_solves = 275;
  result.verification_seconds = 7;
  return result;
}

void cleared_quality(const solver::SolveResult& result) {
  require(!result.has_solution && result.solution.empty() && result.row_dual.empty() &&
      !result.objective && !result.best_bound && !result.absolute_gap && !result.relative_gap &&
      !result.lp_optimality && !result.verification.valid_input && !result.verification.feasible &&
      !result.verification.objective_consistent && !result.verification.objective,
      "child/恢复失败不得遗留约化坐标或旧成功质量");
}

void legacy_bait(const solver::SolveResult& result) {
  require(result.ipm_iterations == 137 && result.ipm_factorizations == 11 &&
      result.ipm_refinement_steps == 19 && result.ipm_homogeneous_constant_solves == 139 &&
      result.ipm_homogeneous_direction_solves == 275,
      "原模型验收与质量清理不得覆盖已经付出的 child legacy 工作");
}

void check_off_wrapper() {
  const auto model = coupled();
  const auto options = policy();
  const auto expected = real_core(model, options);
  const auto actual = real_wrapper(model, options, false);
  same_non_timing(actual, expected);
  original_quality(model, actual, 25.0 / 3.0);
  ++wrapper_cases;
  details.push_back({"off_wrapper_is_original_core", "wrapper"});
}

void mapping(const char* id, const solver::Model& model, double objective, bool empty) {
  const auto options = policy();
  original_quality(model, real_wrapper(model, options, true), objective);
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  const auto actual = candidate::control(model, options, start, stats,
      [&](const solver::Model& child, const solver::SolveOptions& child_options,
          solver::Clock::time_point child_start) {
        require(!empty, "空约化不得进入自主 IPM core");
        require(child_start == start && child_options.time_limit == options.time_limit,
            "约化 child 必须共享原 epoch 和五秒总预算");
        require(child.sense == solver::ObjectiveSense::minimize && child.costs.size() == 2 &&
            child.row_lower.size() == 2, "真实非空 child 必须是已归一化的约化模型");
        ++component_ipm_calls;
        return solver::v87_core(child, child_options, child_start);
      });
  original_quality(model, actual, objective);
  require(stats.accepted && std::string(stats.status) == "accepted" && stats.assessments.completed == 1 &&
      stats.postsolve.completed == 1 && stats.native.basis_valid == false,
      "候选须真实完成无有效基恢复与唯一原模型终验");
  if (empty) {
    require(std::string(stats.route) == "empty" && stats.child.started == 0 && stats.child.completed == 0 &&
        actual.ipm_iterations == 0 && actual.ipm_factorizations == 0,
        "真实空约化必须跳过全部自主 IPM 工作");
  } else {
    require(std::string(stats.route) == "reduced" && stats.native.presolve_status == 3 &&
        stats.child.started == 1 && stats.child.completed == 1,
        "非空覆盖必须真实缩减后执行一次 child，不能用 NotReduced 冒充");
  }
  ++mapping_cases;
  details.push_back({id, "mapping"});
}

void check_not_reduced() {
  const auto model = coupled();
  const auto options = policy();
  const auto expected = real_core(model, options);
  same_non_timing(real_wrapper(model, options, true), expected);
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  const auto actual = candidate::control(model, options, start, stats,
      [&](const solver::Model& input, const solver::SolveOptions& child_options,
          solver::Clock::time_point child_start) {
        require(same_model(input, model) && child_start == start &&
            child_options.time_limit == options.time_limit,
            "NotReduced 必须直接把原模型与原预算交给原 core");
        ++component_ipm_calls;
        return solver::v87_core(input, child_options, child_start);
      });
  same_non_timing(actual, expected);
  require(stats.native.presolve_status == 0 && std::string(stats.route) == "original" &&
      stats.child.started == 1 && stats.child.completed == 1 && stats.postsolve.started == 0 &&
      stats.assessments.started == 0,
      "真实 NotReduced 路由不得恢复或重复原模型终验");
  ++wrapper_cases;
  details.push_back({"not_reduced_is_original_core", "wrapper"});
}

void check_zero_budget(bool iteration) {
  const auto model = coupled(true);
  auto options = policy();
  if (iteration) options.ipm.max_iterations = 0;
  else options.time_limit = 0;
  const auto expected = real_core(model, options);
  const auto actual = real_wrapper(model, options, true);
  same_non_timing(actual, expected);
  require(actual.status == (iteration ? "iteration_limit" : "time_limit") &&
      actual.ipm_iterations == 0 && actual.ipm_factorizations == 0,
      "零原预算必须保持原 core 状态和零迭代");
  ++budget_cases;
  details.push_back({iteration ? "iteration_zero_before_native" : "time_zero_before_native", "budget"});
}

void check_quality_transaction() {
  const auto original = coupled(true);
  const auto child_model = coupled();
  auto child = bait(child_model);
  const auto before = child;
  candidate::clear_coordinate_quality(child);
  cleared_quality(child);
  legacy_bait(child);
  require(same_model(child.model, child_model) && child.verification_seconds == before.verification_seconds &&
      child.backend == before.backend && child.backend_version == before.backend_version,
      "纯质量清理不得更改模型、legacy 或已付验收时间");
  auto outer = candidate::make_shell(original);
  candidate::adopt_child(outer, before);
  cleared_quality(outer);
  legacy_bait(outer);
  require(same_model(outer.model, original) && outer.verification_seconds == before.verification_seconds,
      "child 必须保留原模型和已付工作，并清除约化质量");
  ++transaction_cases;
  details.push_back({"adopt_child_clears_all_coordinate_quality", "transaction"});
}

void check_commit(bool valid) {
  const auto model = coupled(true, true);
  auto result = bait(model, "numerical_error");
  candidate::clear_coordinate_quality(result);
  const std::vector<double> x = valid ? std::vector<double>{3, 4.0 / 3.0, 4.0 / 3.0}
                                       : std::vector<double>{3, 3, 3};
  const std::vector<double> y{-1.0 / 3.0, -1.0 / 3.0};
  auto quality = solver::verify_lp_solution(model, x, y);
  const bool committed = candidate::commit_original_quality(result, x, y, std::move(quality), policy());
  legacy_bait(result);
  require(same_model(result.model, model) && result.verification_seconds == 7,
      "提交适配器不得重复累计原验时间或替换模型");
  if (valid) {
    require(committed, "完整原模型质量应能替换旧 numerical_error 并提交");
    original_quality(model, result, 104.0 / 3.0);
  } else {
    require(!committed && !result.has_solution && result.solution.empty() && !result.objective &&
        result.lp_optimality && !result.lp_optimality->numerically_optimal && !result.verification.feasible,
        "坏原始点不得保留旧诱饵成功解或伪 optimal");
  }
  ++transaction_cases;
  details.push_back({valid ? "commit_valid_original_quality" : "reject_bad_original_primal_quality", "transaction"});
}

void check_mip_bypass() {
  auto model = scalar(1, 0, 2);
  model.kinds[0] = solver::VariableKind::integer;
  const auto options = policy();
  const auto expected = real_core(model, options);
  const auto actual = real_wrapper(model, options, true);
  same_non_timing(actual, expected);
  require(actual.status == "error" && !actual.has_solution && actual.ipm_iterations == 0,
      "不支持的 MIP 必须保持原 core 拒绝，不能借 native 路径绕行");
  ++wrapper_cases;
  details.push_back({"mip_is_original_core_rejection", "wrapper"});
}

struct SyntheticClock {
  solver::Clock::time_point start;
  candidate::Stats& stats;
  double time = 0;
  bool expired = false, expire_commit = false, throw_after_assessment = false, thrown = false;

  solver::Clock::time_point operator()() {
    if (throw_after_assessment && !thrown && std::string(stats.phase) == "verification" &&
        stats.assessments.completed == 1) {
      thrown = true;
      throw std::runtime_error("原验之后的确定性时钟异常");
    }
    if (expired || (expire_commit && std::string(stats.phase) == "commit")) time = 6;
    else time += 0.0001;
    return start + std::chrono::duration_cast<solver::Clock::duration>(std::chrono::duration<double>(time));
  }
};

void timing_contract(const solver::SolveResult& result, const candidate::Stats& stats,
                      bool assessed) {
  require(result.verification_seconds == stats.timings.child_verification + stats.timings.verification &&
      result.total_seconds == stats.timings.total &&
      result.solve_seconds == std::max(0.0, result.total_seconds - result.verification_seconds),
      "child 与最终原验耗时只能各累计一次，总时间不能再叠加 child 前缀");
  if (assessed) require(stats.timings.verification > 0,
                       "完整或异常的最终原验阶段必须保留实际阶段成本");
}

enum class ChildKind { good_nonoptimal, bad_dual, bad_primal, no_candidate, bad_dimension };

void synthetic_child(ChildKind kind) {
  const auto model = coupled(true);
  const auto options = policy();
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats};
  const auto result = candidate::control(model, options, start, stats,
      [&](const solver::Model& input, const solver::SolveOptions& child_options,
          solver::Clock::time_point child_start) {
        ++synthetic_core_calls;
        require(input.costs == std::vector<double>({-1, -1}) && input.row_lower.size() == 2 &&
            input.values == std::vector<double>({2, 1, 1, 2}) && child_start == start &&
            child_options.time_limit == options.time_limit && child_options.ipm.max_iterations == 200,
            "合成 child 必须接到真实约化模型、原 epoch 与原迭代额度");
        auto child = bait(input, kind == ChildKind::good_nonoptimal ? "numerical_error" : "optimal");
        child.verification_seconds = 0.00001;
        child.solution = {4.0 / 3.0, 4.0 / 3.0}; child.row_dual = {-1.0 / 3.0, -1.0 / 3.0};
        if (kind == ChildKind::bad_dual) child.row_dual.assign(2, 0);
        if (kind == ChildKind::bad_primal) child.solution.assign(2, 3);
        if (kind == ChildKind::no_candidate) child.has_solution = false;
        if (kind == ChildKind::bad_dimension) child.row_dual.pop_back();
        return child;
      }, [&] { return clock(); });
  require(same_model(result.model, model) && stats.child.started == 1 && stats.child.completed == 1 &&
      stats.timings.child_verification == 0.00001, "合成 child 账本与原模型必须完整恢复");
  legacy_bait(result);
  const bool assessed = kind != ChildKind::no_candidate && kind != ChildKind::bad_dimension;
  timing_contract(result, stats, assessed);
  if (kind == ChildKind::good_nonoptimal) {
    original_quality(model, result, 88.0 / 3.0);
    require(stats.accepted && stats.child.status == std::string("numerical_error") &&
        stats.assessments.completed == 1, "child 非 optimal 不能阻止独立原终验认可真实最优点");
  } else if (kind == ChildKind::bad_dual) {
    require(!stats.accepted && result.status == "verification_failed" && result.has_solution &&
        result.lp_optimality && !result.lp_optimality->dual_feasible &&
        result.solution.size() == model.costs.size() && stats.assessments.completed == 1,
        "child 假 optimal 的坏对偶须拒原最优，但可保原坐标可行 incumbent");
  } else if (kind == ChildKind::bad_primal) {
    require(!stats.accepted && result.status == "verification_failed" && !result.has_solution &&
        result.solution.empty() && !result.objective && !result.verification.feasible &&
        stats.assessments.completed == 1, "child 假 optimal 的坏原始点不能保成功载荷");
  } else {
    cleared_quality(result);
    require(!stats.accepted && stats.postsolve.started == 0 && stats.assessments.started == 0,
        "无候选或坏维度不应进入恢复/原验，也不得遗留 child 诱饵质量");
  }
  ++transaction_cases;
  const char* id = kind == ChildKind::good_nonoptimal ? "nonoptimal_child_passes_original_quality" :
      kind == ChildKind::bad_dual ? "false_optimal_child_dual_rejected" :
      kind == ChildKind::bad_primal ? "false_optimal_child_primal_rejected" :
      kind == ChildKind::no_candidate ? "missing_candidate_drops_child_bait" : "bad_child_dimension_drops_bait";
  details.push_back({id, "transaction"});
}

void check_presolve_status() {
  auto model = scalar(0, 0, infinity);
  row(model, 1, -infinity, -1);
  Mode mode(true);
  candidate::Stats stats;
  const auto options = policy();
  int calls = 0;
  const auto result = candidate::control(model, options, solver::Clock::now(), stats,
      [&](const solver::Model& input, const solver::SolveOptions&, solver::Clock::time_point) {
        ++calls; ++synthetic_core_calls;
        return bait(input);
      });
  require(calls == 0 && stats.native.presolve_status == 1 && !stats.accepted &&
      std::string(stats.status) == "presolve_rejected" && result.status == "error" &&
      stats.child.started == 0 && stats.postsolve.started == 0 && stats.assessments.started == 0,
      "原生不可行状态只能有界拒绝，不能冒充自主原模型不可行证书");
  cleared_quality(result);
  ++transaction_cases;
  details.push_back({"native_infeasible_is_not_original_certificate", "transaction"});
}

void check_entry_deadline() {
  const auto model = coupled(true);
  const auto options = policy();
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats}; clock.expired = true;
  int calls = 0;
  const auto result = candidate::control(model, options, start, stats,
      [&](const solver::Model& input, const solver::SolveOptions&, solver::Clock::time_point) {
        ++calls; ++synthetic_core_calls; return bait(input);
      }, [&] { return clock(); });
  require(result.status == "time_limit" && !stats.accepted && calls == 0 &&
      stats.pass_model.started == 0 && stats.presolve.started == 0 && stats.child.started == 0 &&
      stats.assessments.started == 0, "入口已过期不能启动任何组件或重新获取预算");
  cleared_quality(result);
  timing_contract(result, stats, false);
  ++budget_cases;
  details.push_back({"entry_deadline_stops_before_native", "budget"});
}

void check_late_boundary(bool expire_child, bool throw_assessment) {
  const auto model = coupled(true);
  const auto options = policy();
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats};
  clock.expire_commit = !expire_child && !throw_assessment;
  clock.throw_after_assessment = throw_assessment;
  const auto result = candidate::control(model, options, start, stats,
      [&](const solver::Model& input, const solver::SolveOptions&, solver::Clock::time_point) {
        ++synthetic_core_calls;
        auto child = bait(input);
        child.solution = {4.0 / 3.0, 4.0 / 3.0}; child.row_dual = {-1.0 / 3.0, -1.0 / 3.0};
        child.verification_seconds = 0.00001;
        if (expire_child) clock.expired = true;
        return child;
      }, [&] { return clock(); });
  require(!stats.accepted && result.status == (throw_assessment ? "error" : "time_limit") &&
      stats.child.completed == 1, "后置预算或时钟异常不能提交候选 optimal");
  legacy_bait(result);
  cleared_quality(result);
  timing_contract(result, stats, !expire_child);
  if (expire_child) {
    require(stats.postsolve.started == 0 && stats.assessments.started == 0,
        "child 后预算失败必须先保工作再停，不能启动额外恢复");
  } else {
    require(stats.assessments.completed == 1 && stats.original_quality &&
        stats.original_quality->numerically_optimal && stats.postsolve.completed == 1,
        "原验完成但提交前失败可保诊断，不得提交原质量");
  }
  if (throw_assessment) {
    require(clock.thrown && std::string(stats.status) == "exception",
        "原验后异常应留独立异常诊断和已付阶段成本");
    ++exception_cases;
    details.push_back({"after_assessment_exception_keeps_cost_only", "exception"});
  } else {
    ++budget_cases;
    details.push_back({expire_child ? "child_deadline_keeps_legacy_only" : "post_verify_deadline_cannot_commit", "budget"});
  }
}

void check_child_exception(bool standard) {
  const auto model = coupled(true);
  const auto options = policy();
  Mode mode(true);
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats};
  const auto result = candidate::control(model, options, start, stats,
      [&](const solver::Model&, const solver::SolveOptions&, solver::Clock::time_point) -> solver::SolveResult {
        ++synthetic_core_calls;
        if (standard) throw std::runtime_error("child 原异常");
        throw 17;
      }, [&] { return clock(); });
  require(!stats.accepted && result.status == "error" && std::string(stats.status) == "exception" &&
      stats.child.started == 1 && stats.child.completed == 0 && stats.timings.child > 0 &&
      stats.postsolve.started == 0 && stats.assessments.started == 0 &&
      same_model(result.model, model), "child 异常必须保真实部分账本、计时和原模型，不能伪完成");
  cleared_quality(result);
  timing_contract(result, stats, false);
  ++exception_cases;
  details.push_back({standard ? "child_standard_exception_partial_cost" : "child_nonstandard_exception_partial_cost", "exception"});
}

enum class ProjectionKind {
  raw_pass, unchanged, dual_lower, dual_upper, primal_lower, primal_upper,
  both, breaks_row, breaks_gap, nonfinite
};

struct ProjectionInput {
  solver::Model model;
  std::vector<double> x, y;
};

ProjectionInput projection_input(ProjectionKind kind) {
  ProjectionInput input{scalar(0, 0, 1), {0}, {}};
  if (kind == ProjectionKind::raw_pass) input.x[0] = -5e-8;
  if (kind == ProjectionKind::unchanged) return {coupled(), {3, 0}, {-1.0 / 3.0, -1.0 / 3.0}};
  if (kind == ProjectionKind::dual_lower || kind == ProjectionKind::both || kind == ProjectionKind::nonfinite) {
    row(input.model, 0, 0, infinity); input.y = {-1e-300};
  }
  if (kind == ProjectionKind::dual_upper) {
    row(input.model, 0, -infinity, 0); input.y = {1e-300};
  }
  if (kind == ProjectionKind::primal_lower) input.model.costs[0] = 1;
  if (kind == ProjectionKind::primal_lower || kind == ProjectionKind::both || kind == ProjectionKind::nonfinite)
    input.x[0] = -2e-7;
  if (kind == ProjectionKind::primal_upper) {
    input.model.costs[0] = -1; input.x[0] = 1 + 2e-7;
  }
  if (kind == ProjectionKind::breaks_row) {
    input.x[0] = -5e-8;
    input.model.row_lower = {-5e-7, -1}; input.model.row_upper = {-5e-7, infinity};
    input.model.column_start = {0, 1}; input.model.row_index = {0}; input.model.values = {10};
    input.y = {0, -1e-300};
  }
  if (kind == ProjectionKind::breaks_gap) {
    input.model.costs[0] = 1; input.x[0] = 0.5;
    row(input.model, 0, -5e9, infinity); input.y = {-1e-10};
  }
  if (kind == ProjectionKind::nonfinite) input.y[0] = std::numeric_limits<double>::quiet_NaN();
  return input;
}

bool same_vector(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (a[k] != b[k] && !(std::isnan(a[k]) && std::isnan(b[k]))) return false;
  return true;
}

void prepare_projection(candidate::Stats& stats, const solver::LpVerification& raw) {
  // helper位于一次完整raw原验之后；新试验只准再增加至多一次原验。
  stats.enabled = true; stats.attempted = true;
  stats.assessments.started = 1; stats.assessments.completed = 1;
  stats.original_quality = candidate::quality(raw.primal, raw.optimality);
}

void check_projection(ProjectionKind kind) {
  auto input = projection_input(kind);
  const auto before_x = input.x, before_y = input.y;
  const auto options = policy();
  const auto raw = solver::verify_lp_solution(input.model, input.x, input.y,
      options.tolerances, options.pdlp.optimality);
  require(quality_ok(raw) == (kind == ProjectionKind::raw_pass),
      "投影夹具必须先按共同精度区分原验已过与未过");
  if (kind == ProjectionKind::breaks_row || kind == ProjectionKind::breaks_gap)
    require(raw.primal.feasible, "拒绝试点时须真正存在可保留的raw可行质量");
  if (kind == ProjectionKind::breaks_gap)
    require(raw.optimality.gap_satisfied && !raw.optimality.dual_feasible,
        "非法符号补偿的近零估计gap不得冒充原最优");
  candidate::Stats stats;
  prepare_projection(stats, raw);
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats};
  candidate::Budget budget(start, options.time_limit, stats, [&] { return clock(); });
  auto trial = candidate::bounded_projection(input.model, options, input.x, input.y, raw, stats, budget);
  require(same_vector(input.x, before_x) && same_vector(input.y, before_y),
      "域投影只能写局部拥有的试点，不能修改raw任一向量");
  const bool projected = kind == ProjectionKind::dual_lower || kind == ProjectionKind::dual_upper ||
      kind == ProjectionKind::primal_lower || kind == ProjectionKind::primal_upper || kind == ProjectionKind::both;
  const bool second_assessment = projected || kind == ProjectionKind::breaks_row || kind == ProjectionKind::breaks_gap;
  require(trial.has_value() == projected && stats.assessments.started == 1 + second_assessment &&
      stats.assessments.completed == 1 + second_assessment,
      "raw已过或坐标未变不能重复原验；改变时最多再验一次");
  const auto& p = stats.projection;
  require(p.raw_quality && p.raw_quality->numerically_optimal == raw.optimality.numerically_optimal,
      "原验诊断不能被试点质量覆盖");
  if (kind == ProjectionKind::raw_pass) {
    require(p.started == 0 && p.completed == 0 && std::string(p.status) == "not_needed" &&
        std::string(p.chosen) == "raw" && p.x_changed == 0 && p.y_changed == 0,
        "共同容差内已通过的raw点不得被无必要地投影");
  } else if (kind == ProjectionKind::unchanged) {
    require(p.started == 1 && p.completed == 1 && std::string(p.status) == "unchanged" &&
        std::string(p.chosen) == "raw" && !p.trial_quality,
        "raw失败但域坐标未变时必须保raw且不做第二原验");
  } else if (kind == ProjectionKind::nonfinite) {
    require(!trial && p.completed == 0 && std::string(p.status) == "exception" &&
        std::string(p.chosen) == "raw" && !p.trial_quality,
        "第二向量非有限时不能泄漏已经投影的第一向量或选择半点");
  } else {
    require(p.started == 1 && p.completed == 1 && p.trial_quality &&
        std::string(p.status) == (projected ? "accepted" : "rejected") &&
        std::string(p.chosen) == (projected ? "projected" : "raw"),
        "只有完整有限试点全过原验后才能选择projected");
    if (kind == ProjectionKind::breaks_row) require(!p.trial_quality->primal_feasible,
        "合法域投影破坏原行的试点必须拒绝");
    if (kind == ProjectionKind::breaks_gap) require(!p.trial_quality->gap_satisfied,
        "投影后gap不合格不能仅凭域合法而接纳");
  }
  if (trial) {
    require(trial->x.size() == input.x.size() && trial->y.size() == input.y.size() &&
        std::all_of(trial->x.begin(), trial->x.end(), [](double value) { return std::isfinite(value); }) &&
        std::all_of(trial->y.begin(), trial->y.end(), [](double value) { return std::isfinite(value); }) &&
        quality_ok(trial->checked), "返回试点必须拥有两份完整有限向量及完整原质量");
    require((input.x.empty() || trial->x.data() != input.x.data()) &&
        (input.y.empty() || trial->y.data() != input.y.data()), "试点不能借用仍待回退的raw存储");
    if (kind == ProjectionKind::both) require(p.x_changed == 1 && p.y_changed == 1,
        "双side反例必须实际改变x与y");
  }
  if (kind != ProjectionKind::nonfinite) {
    auto result = bait(input.model, "numerical_error");
    candidate::clear_coordinate_quality(result);
    const bool committed = trial ? candidate::commit_original_quality(result, std::move(trial->x),
        std::move(trial->y), std::move(trial->checked), options) :
        candidate::commit_original_quality(result, input.x, input.y, raw, options);
    legacy_bait(result);
    require(committed == (projected || kind == ProjectionKind::raw_pass),
        "最终提交只能采用全过原验的raw或projected，不能把域内点当最优");
    if (kind == ProjectionKind::breaks_row || kind == ProjectionKind::breaks_gap) {
      require(result.has_solution && result.status != "optimal" && result.solution == input.x &&
          result.row_dual == input.y && same_verification(result.verification, raw.primal) &&
          result.lp_optimality && same_optimality(*result.lp_optimality, raw.optimality),
          "被拒试点不得覆盖raw可行incumbent及其真实质量");
    }
  }
  ++transaction_cases;
  const char* id = kind == ProjectionKind::raw_pass ? "projection_skips_already_accepted_raw" :
      kind == ProjectionKind::unchanged ? "projection_unchanged_avoids_second_assessment" :
      kind == ProjectionKind::dual_lower ? "projection_removes_tiny_wrong_lower_dual_sign" :
      kind == ProjectionKind::dual_upper ? "projection_removes_tiny_wrong_upper_dual_sign" :
      kind == ProjectionKind::primal_lower ? "projection_repairs_original_lower_bound" :
      kind == ProjectionKind::primal_upper ? "projection_repairs_original_upper_bound" :
      kind == ProjectionKind::both ? "projection_repairs_both_sides_transactionally" :
      kind == ProjectionKind::breaks_row ? "projection_rejects_broken_row_keeps_raw_incumbent" :
      kind == ProjectionKind::breaks_gap ? "projection_rejects_broken_gap_keeps_raw_incumbent" :
      "projection_nonfinite_second_vector_never_selects_partial_point";
  details.push_back({id, "transaction"});
}

enum class ProjectionBoundary { copy, primal, dual_after, scan, trial_assessment, commit, ordinary_then_expired };

struct ProjectionClock {
  solver::Clock::time_point start;
  candidate::Stats& stats;
  ProjectionBoundary boundary;
  double time = 0;
  int dual_checks = 0;
  bool armed = false, thrown = false;

  solver::Clock::time_point operator()() {
    const auto phase = std::string(stats.projection.phase);
    if (boundary == ProjectionBoundary::ordinary_then_expired && !thrown && phase == "copy") {
      thrown = true;
      throw std::runtime_error("投影临时时钟异常后截止");
    }
    bool expired = boundary == ProjectionBoundary::ordinary_then_expired && thrown;
    if (boundary == ProjectionBoundary::copy && phase == "copy" && stats.budget.loop_units >= 256) expired = true;
    if (boundary == ProjectionBoundary::primal && phase == "primal" && stats.projection.x_changed > 0) expired = true;
    if (boundary == ProjectionBoundary::dual_after && phase == "dual" && ++dual_checks >= 2) expired = true;
    if (boundary == ProjectionBoundary::scan && phase == "scan" && stats.projection.y_changed > 0) expired = true;
    if (boundary == ProjectionBoundary::trial_assessment && phase == "verification" && stats.assessments.completed == 2) expired = true;
    if (boundary == ProjectionBoundary::commit && armed) expired = true;
    time = expired ? 6 : time + 0.0001;
    return start + std::chrono::duration_cast<solver::Clock::duration>(std::chrono::duration<double>(time));
  }
};

ProjectionInput long_projection_input(bool dual) {
  ProjectionInput input{scalar(0, 0, 1), {0}, {}};
  if (dual) {
    input.model.row_lower.assign(600, 0); input.model.row_upper.assign(600, infinity);
    input.y.assign(600, 0); input.y[0] = -1e-300;
  } else {
    input.model.costs.assign(600, 0); input.model.column_lower.assign(600, 0);
    input.model.column_upper.assign(600, 1); input.model.kinds.assign(600, solver::VariableKind::continuous);
    input.model.column_start.assign(601, 0); input.x.assign(600, 0); input.x[0] = -2e-7;
  }
  return input;
}

void check_projection_budget(ProjectionBoundary boundary) {
  auto input = boundary == ProjectionBoundary::copy || boundary == ProjectionBoundary::primal ?
      long_projection_input(false) : boundary == ProjectionBoundary::scan ?
      long_projection_input(true) : projection_input(ProjectionKind::both);
  const auto raw = solver::verify_lp_solution(input.model, input.x, input.y);
  require(!quality_ok(raw), "投影预算反例必须真正需要可选trial");
  const auto before_x = input.x, before_y = input.y;
  candidate::Stats stats;
  prepare_projection(stats, raw);
  const auto options = policy();
  const auto start = solver::Clock::now();
  ProjectionClock clock{start, stats, boundary};
  candidate::Budget budget(start, options.time_limit, stats, [&] { return clock(); });
  auto result = bait(input.model, "numerical_error");
  candidate::clear_coordinate_quality(result);
  const auto before_result = result;
  bool expired = false;
  try {
    auto trial = candidate::bounded_projection(input.model, options, input.x, input.y, raw, stats, budget);
    require(boundary == ProjectionBoundary::commit && trial && quality_ok(trial->checked),
        "除最终提交边界外，预算必须在helper交出试点之前中断");
    // 与生产最后提交使用同一个Budget；local已选试点不等于public已提交。
    stats.phase = "commit"; clock.armed = true;
    budget.check();
    candidate::commit_original_quality(result, std::move(trial->x), std::move(trial->y),
        std::move(trial->checked), options);
  } catch (const candidate::Expired&) { expired = true; }
  require(expired && same_vector(input.x, before_x) && same_vector(input.y, before_y),
      "复制、循环或最后边界预算必须保Expired类型及完整raw向量");
  same_non_timing(result, before_result);
  legacy_bait(result);
  require(result.verification_seconds == before_result.verification_seconds,
      "没有提交的本地trial不能偷写公共结果或重复其验收账本");
  if (boundary == ProjectionBoundary::commit) {
    require(stats.projection.trial_quality && stats.projection.trial_quality->numerically_optimal &&
        std::string(stats.projection.chosen) == "projected" && stats.assessments.completed == 2,
        "本地trial已完成而最终提交超时应保本地诊断但不提交公共质量");
  } else {
    require(std::string(stats.projection.status) == "time_limit" && std::string(stats.projection.chosen) == "none",
        "helper内预算失败不能选择raw回退或projected来绕过截止");
    if (boundary == ProjectionBoundary::trial_assessment)
      require(stats.assessments.completed == 2 && stats.projection.trial_quality &&
          stats.projection.trial_quality->numerically_optimal && stats.timings.verification > 0,
          "第二原验完成后预算失败仍须保真实验收成本和完整诊断");
  }
  require(stats.timings.recovery > 0 && stats.budget.checks > 0,
      "失败投影的复制、投影和统计实际阶段成本不能丢失");
  if (boundary == ProjectionBoundary::copy || boundary == ProjectionBoundary::primal || boundary == ProjectionBoundary::scan)
    require(stats.budget.loop_units >= 256, "长向量反例必须真实进入固定256循环检查");
  ++budget_cases;
  const char* id = boundary == ProjectionBoundary::copy ? "projection_copy_loop_deadline" :
      boundary == ProjectionBoundary::primal ? "projection_primal_loop_deadline" :
      boundary == ProjectionBoundary::dual_after ? "projection_after_dual_api_deadline" :
      boundary == ProjectionBoundary::scan ? "projection_scan_loop_deadline" :
      boundary == ProjectionBoundary::trial_assessment ? "projection_post_assessment_deadline" :
      boundary == ProjectionBoundary::commit ? "projection_final_commit_deadline" :
      "projection_exception_cannot_fallback_after_deadline";
  details.push_back({id, "budget"});
}

void check_projection_exception(bool allocation) {
  auto input = projection_input(ProjectionKind::both);
  const auto before_x = input.x, before_y = input.y;
  const auto raw = solver::verify_lp_solution(input.model, input.x, input.y);
  const auto options = policy();
  candidate::Stats stats;
  prepare_projection(stats, raw);
  const auto start = solver::Clock::now();
  double time = 0;
  bool injected = false;
  candidate::Budget budget(start, options.time_limit, stats, [&] {
    if (!injected && std::string(stats.projection.phase) == "copy") {
      injected = true;
      // 精确注入异常类型，模拟局部trial复制分配/时钟失败；不声称耗尽系统内存。
      if (allocation) throw std::bad_alloc();
      throw std::runtime_error("可选trial临时时钟异常");
    }
    time += 0.0001;
    return start + std::chrono::duration_cast<solver::Clock::duration>(std::chrono::duration<double>(time));
  });
  const auto trial = candidate::bounded_projection(input.model, options, input.x, input.y, raw, stats, budget);
  require(injected && !trial && std::string(stats.projection.status) == "exception" &&
      std::string(stats.projection.chosen) == "raw" && stats.assessments.completed == 1 &&
      !stats.projection.trial_quality && same_vector(input.x, before_x) && same_vector(input.y, before_y) &&
      stats.timings.recovery > 0,
      "普通或分配异常只有预算复核通过才回raw，并保局部失败真实成本");
  auto result = bait(input.model, "numerical_error");
  candidate::clear_coordinate_quality(result);
  require(!candidate::commit_original_quality(result, input.x, input.y, raw, options) && result.status != "optimal",
      "异常回退不得把未过原验的raw点伪装成最优");
  legacy_bait(result);
  ++exception_cases;
  details.push_back({allocation ? "projection_allocation_exception_raw_fallback" : "projection_clock_exception_raw_fallback", "exception"});
}

}  // 匿名命名空间

int main() {
  try {
    check_off_wrapper();
    mapping("minimum_column_lower_offset", scalar(2, 1, 4, 7), 9, true);
    mapping("minimum_column_upper_offset", scalar(-2, -5, -2, -3), 1, true);
    mapping("maximum_column_upper_offset", scalar(3, -1, 4, 7, true), 19, true);
    auto model = scalar(3, -infinity, infinity, -4); row(model, 2, 4, infinity);
    mapping("minimum_original_row_lower", model, 2, true);
    model = scalar(2, -infinity, infinity, 7, true); row(model, 1, -infinity, 3);
    mapping("maximum_original_row_upper", model, 13, true);
    model = scalar(3, -5, 8); row(model, 1, 2, 5);
    mapping("ranged_row_lower", model, 6, true);
    model.sense = solver::ObjectiveSense::maximize;
    mapping("ranged_row_upper", model, 15, true);
    model = scalar(7, 3, 3, 11);
    model.costs.push_back(3); model.column_lower.push_back(-infinity); model.column_upper.push_back(infinity);
    model.kinds.push_back(solver::VariableKind::continuous); model.row_lower = {19}; model.row_upper = {19};
    model.column_start = {0, 1, 2}; model.row_index = {0, 0}; model.values = {5, 2};
    mapping("fixed_column_equality_offset", model, 38, true);
    mapping("minimum_real_reduced_child", coupled(true), 88.0 / 3.0, false);
    mapping("maximum_real_reduced_child", coupled(true, true), 104.0 / 3.0, false);
    check_not_reduced(); check_mip_bypass(); check_zero_budget(false); check_zero_budget(true);
    check_quality_transaction(); check_commit(true); check_commit(false);
    for (const auto kind : {ChildKind::good_nonoptimal, ChildKind::bad_dual, ChildKind::bad_primal,
                           ChildKind::no_candidate, ChildKind::bad_dimension}) synthetic_child(kind);
    check_presolve_status(); check_entry_deadline();
    check_late_boundary(true, false); check_late_boundary(false, false); check_late_boundary(false, true);
    check_child_exception(true); check_child_exception(false);
    for (const auto kind : {ProjectionKind::raw_pass, ProjectionKind::unchanged, ProjectionKind::dual_lower,
                           ProjectionKind::dual_upper, ProjectionKind::primal_lower, ProjectionKind::primal_upper,
                           ProjectionKind::both, ProjectionKind::breaks_row, ProjectionKind::breaks_gap,
                           ProjectionKind::nonfinite}) check_projection(kind);
    for (const auto boundary : {ProjectionBoundary::copy, ProjectionBoundary::primal, ProjectionBoundary::dual_after,
                               ProjectionBoundary::scan, ProjectionBoundary::trial_assessment, ProjectionBoundary::commit,
                               ProjectionBoundary::ordinary_then_expired}) check_projection_budget(boundary);
    check_projection_exception(false); check_projection_exception(true);
    require(details.size() == static_cast<std::size_t>(wrapper_cases + mapping_cases + transaction_cases +
        budget_cases + exception_cases) && component_ipm_calls > 0 && synthetic_core_calls > 0,
        "实际小例与合成边界计数必须守恒且不能冒称零全部求解");
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"explicit_highs_lp_solve_calls\":0,"
        "\"scope\":\"仅组件自测；component_ipm_calls计harness进入真实solve_ipm或v87_core的API次数(含零预算入口)，不含内部调用拦截；synthetic_core_calls另计，不是公开LP成绩\","
        "\"cases\":%zu,\"wrapper_cases\":%d,\"mapping_cases\":%d,\"transaction_cases\":%d,"
        "\"budget_cases\":%d,\"exception_cases\":%d,\"component_ipm_calls\":%d,\"synthetic_core_calls\":%d,\"details\":[",
        details.size(), wrapper_cases, mapping_cases, transaction_cases, budget_cases, exception_cases,
        component_ipm_calls, synthetic_core_calls);
    for (std::size_t i = 0; i < details.size(); ++i) {
      if (i) std::printf(",");
      std::printf("{\"id\":\"%s\",\"group\":\"%s\"}", details[i].id, details[i].group);
    }
    std::printf("]}\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "组件自测失败（已完成%zu组，真实入口%d，合成入口%d）：%s\n",
        details.size(), component_ipm_calls, synthetic_core_calls, error.what());
    return 1;
  }
}
