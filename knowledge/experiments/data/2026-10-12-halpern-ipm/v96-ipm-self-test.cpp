#include "ipm-presolve.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
namespace candidate = solver_v96_presolve;
namespace coupled_state = solver_v96_coupled_state;
namespace coupled_kernel = solver::v96_coupled;
namespace energy_state = solver_v96_energy_state;
namespace energy_kernel = solver::v93_energy;
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
#define V94_SAME(field) require(a.field == b.field, "非计时字段改变：" #field)
  V94_SAME(backend); V94_SAME(status); V94_SAME(backend_status); V94_SAME(backend_version);
  V94_SAME(message); V94_SAME(verification_level); V94_SAME(solution); V94_SAME(has_solution);
  V94_SAME(objective); V94_SAME(best_bound); V94_SAME(absolute_gap); V94_SAME(relative_gap);
  V94_SAME(nodes); V94_SAME(simplex_iterations); V94_SAME(pdlp_iterations);
  V94_SAME(pdlp_acceleration_attempts); V94_SAME(pdlp_acceleration_accepted); V94_SAME(pdlp_operator_passes);
  V94_SAME(ipm_iterations); V94_SAME(ipm_factorizations); V94_SAME(ipm_refinement_steps);
  V94_SAME(ipm_weighted_krylov_steps); V94_SAME(ipm_full_newton_preconditioner_solves);
  V94_SAME(ipm_full_newton_products); V94_SAME(ipm_bounded_newton_returns);
  V94_SAME(ipm_bounded_newton_inexact_returns); V94_SAME(ipm_homogeneous_constant_solves);
  V94_SAME(ipm_homogeneous_constant_repairs); V94_SAME(ipm_homogeneous_constant_repair_successes);
  V94_SAME(ipm_homogeneous_direction_solves); V94_SAME(ipm_homogeneous_scaled_products);
  V94_SAME(ipm_homogeneous_scaled_refinement_steps); V94_SAME(ipm_homogeneous_scaled_inexact_returns);
  V94_SAME(ipm_homogeneous_tau); V94_SAME(ipm_homogeneous_kappa);
  V94_SAME(ipm_regularization_searches); V94_SAME(ipm_regularization_reductions);
  V94_SAME(ipm_unshifted_factorizations); V94_SAME(ipm_direction_retries); V94_SAME(ipm_direction_retry_successes);
  V94_SAME(ipm_krylov_recycle_projections); V94_SAME(ipm_krylov_recycle_acceptances);
  V94_SAME(ipm_krylov_recycle_vectors); V94_SAME(ipm_krylov_recycle_peak_dimension);
  V94_SAME(ipm_primal_regularization); V94_SAME(ipm_dual_regularization);
  V94_SAME(ipm_schur_residual_products); V94_SAME(ipm_schur_refinement_steps);
  V94_SAME(ipm_schur_refinement_acceptances); V94_SAME(ipm_kkt_dimension); V94_SAME(ipm_factor_nonzeros);
  V94_SAME(ipm_factor_backend); V94_SAME(ipm_structural_fallbacks); V94_SAME(ipm_eliminated_singletons);
  V94_SAME(ipm_eliminated_fixed_variables); V94_SAME(ipm_presolved_singleton_rows);
  V94_SAME(ipm_presolved_fixed_columns); V94_SAME(ipm_regularized_pivots);
  V94_SAME(ipm_backward_error_acceptances); V94_SAME(ipm_inexact_acceptances);
  V94_SAME(ipm_inexact_prefilter_rejections); V94_SAME(ipm_central_correction_attempts);
  V94_SAME(ipm_central_correction_acceptances); V94_SAME(row_dual);
#undef V94_SAME
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
  model.name = "v94_component_scalar";
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
  model.name = "v94_component_coupled";
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
// 组件调用指本 harness 进入真实 solve_ipm/v96_core 的次数，包含零预算入口；
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
  return solver::v96_core(model, options, solver::Clock::now());
}

solver::SolveResult real_coupled_core(const solver::Model& model, const solver::SolveOptions& options) {
  coupled_state::Log log;
  log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
  ++component_ipm_calls;
  return solver::v96_core(model, options, solver::Clock::now());
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
        return solver::v96_core(child, child_options, child_start);
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
  // NotReduced应对照本轮已选策略的原core，off另有旧core完整比较。
  const auto expected = real_coupled_core(model, options);
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
        coupled_state::Log log;
        log.enabled = true;
        coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
        ++component_ipm_calls;
        return solver::v96_core(input, child_options, child_start);
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
    // oracle的原验属于逐事件frame；外层phase在该调用期间仍为oracle。
    const auto* frame = candidate::active_oracle && candidate::active_oracle->current_frame ?
        candidate::active_oracle->current_frame : &stats;
    if (throw_after_assessment && !thrown && std::string(frame->phase) == "verification" &&
        frame->assessments.completed == 1) {
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
        if (kind != ChildKind::no_candidate && kind != ChildKind::bad_dimension) {
          // 这是合成child错误声称当前已最优的边界测试；真实assess触发另有两组直接覆盖。
          // 必须显式经过oracle原验，不能仅返回带optimal标签的诱饵来冒充hook已执行。
          const auto decision = candidate::consult_current(true, input, child.solution, child.row_dual, 137);
          require(decision == (kind == ChildKind::good_nonoptimal ? candidate::OracleDecision::accept :
              candidate::OracleDecision::reject), "合成当前点必须实际经过原终验防线");
        }
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
        else {
          const auto decision = candidate::consult_current(true, input, child.solution, child.row_dual, 137);
          require(decision == (throw_assessment ? candidate::OracleDecision::error : candidate::OracleDecision::accept),
              "后置失败夹具须先实际进入oracle，不能依赖已删除的尾部恢复");
        }
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
    require(stats.assessments.completed == 1 && stats.postsolve.completed == 1 &&
        stats.oracle.events.size() == 1 && stats.oracle.events.front().original_quality &&
        stats.oracle.events.front().original_quality->numerically_optimal,
        "原验完成但提交前失败可保诊断，不得提交原质量");
  }
  if (throw_assessment) {
    require(clock.thrown && std::string(stats.status) == "native_error" && stats.oracle.errors == 1 &&
        std::string(stats.oracle.events.front().decision) == "error",
        "oracle原验后异常应留独立事件诊断和已付阶段成本，由owner拒绝不完整恢复");
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

// 这里仅调用原生预处理和一参postsolve；不调用HiGHS的任何LP迭代入口。
// 所有引用的owner均为具名成员，oracle比Budget/native先析构。
struct NativeOracleFixture {
  solver::Model original;
  solver::SolveOptions options;
  solver::Clock::time_point start;
  candidate::Stats stats;
  double time = 0;
  bool expire_after_assessment = false, error_after_assessment = false, injected = false;
  candidate::Budget budget;
  Highs native;
  solver::Model reduced;
  std::optional<candidate::OriginalOracleContext> oracle;

  explicit NativeOracleFixture(solver::Model model = coupled(true))
      : original(std::move(model)), options(policy()), start(solver::Clock::now()),
        budget(start, options.time_limit, stats, [this] { return now(); }) {
    stats.enabled = true; stats.attempted = true;
    candidate::configure(native, stats, budget, options);
    auto minimum = candidate::minimum_copy(original, budget);
    budget.check();
    require(native.passModel(std::move(minimum)) == HighsStatus::kOk &&
        native.presolve() == HighsStatus::kOk && native.getModelPresolveStatus() == HighsPresolveStatus::kReduced,
        "oracle数学夹具须真实产生非空约化模型");
    budget.check();
    candidate::observe(native, stats);
    reduced = candidate::own_model(native.getPresolvedLp(), budget);
    require(reduced.costs == std::vector<double>({-1, -1}) && reduced.costs.size() == 2 &&
        reduced.row_lower.size() == 2 && reduced.values == std::vector<double>({2, 1, 1, 2}),
        "oracle数学夹具不得用空约化或NotReduced替代非空恢复栈");
    oracle.emplace(original, options, start, native, stats, budget);
  }

  solver::Clock::time_point now() {
    const auto* frame = oracle && oracle->current_frame ? oracle->current_frame : nullptr;
    if (!injected && frame && std::string(frame->phase) == "verification" && frame->assessments.completed == 1 &&
        (expire_after_assessment || error_after_assessment)) {
      injected = true;
      if (error_after_assessment) throw std::runtime_error("oracle原验后确定性异常");
      time = 6;
    }
    if (!(expire_after_assessment && injected)) time += 0.0001;
    return start + std::chrono::duration_cast<solver::Clock::duration>(std::chrono::duration<double>(time));
  }

  void no_native_iterations() const {
    require(stats.native.basis_valid == false &&
        stats.native.simplex_iterations && *stats.native.simplex_iterations <= 0 &&
        stats.native.ipm_iterations && *stats.native.ipm_iterations <= 0 &&
        stats.native.pdlp_iterations && *stats.native.pdlp_iterations <= 0 &&
        stats.native.crossover_iterations && *stats.native.crossover_iterations <= 0 &&
        stats.native.qp_iterations && *stats.native.qp_iterations <= 0,
        "重复无基恢复不得产生有效基或正原生LP迭代计数；保留原始-1而不伪造0");
  }
};

solver::ConeModel identity_assess_cone() {
  solver::ConeModel cone;
  cone.column_scale = {1, 1}; cone.row_scale = {1, 1};
  cone.original_row = {0, 1}; cone.dual_sign = {-1, -1};
  return cone;
}

solver::Point assess_point(double x) {
  solver::Point point;
  point.x = {x, x}; point.z = {1.0 / 3.0, 1.0 / 3.0};
  return point;
}

void check_old_incumbent_hook() {
  NativeOracleFixture fixture;
  const auto cone = identity_assess_cone();
  auto incumbent = candidate::make_shell(fixture.reduced);
  const std::vector<double> good_x{4.0 / 3.0, 4.0 / 3.0}, good_y{-1.0 / 3.0, -1.0 / 3.0};
  auto checked = solver::verify_lp_solution(fixture.reduced, good_x, good_y);
  require(candidate::commit_original_quality(incumbent, good_x, good_y, std::move(checked), fixture.options),
      "旧incumbent夹具必须先通过约化原检查器");
  const auto bad_point = assess_point(0);
  auto without_oracle = incumbent;
  {
    candidate::ScopedOracle scope(nullptr);
    require(solver::assess(cone, bad_point, fixture.options, fixture.start, without_oracle),
        "无oracle时保留原assess对既有最优incumbent的行为");
  }
  {
    candidate::ScopedOracle scope(&*fixture.oracle);
    require(!solver::assess(cone, bad_point, fixture.options, fixture.start, incumbent) &&
        fixture.stats.oracle.started == 0 && fixture.stats.oracle.events.empty() && incumbent.status != "optimal",
        "当前点未过约化原验时不能因旧incumbent误触发oracle或提前停止");
  }
  require(incumbent.has_solution && incumbent.solution == good_x && incumbent.lp_optimality &&
      incumbent.lp_optimality->numerically_optimal,
      "当前点防护只控制停止，不能破坏已保存的可行incumbent载荷");
  ++transaction_cases;
  details.push_back({"oracle_old_incumbent_cannot_trigger_current", "transaction"});
}

void check_reject_then_accept_hook() {
  auto original = coupled(true);
  // 固定列贡献抵消去常数目标：同一绝对gap可过约化相对容差而不过原模型相对容差。
  // 不改变任何公开容差，先实际核两层夹具质量，再调用真实assess窄hook。
  original.costs[0] = 8.0 / 9.0;
  const auto options = policy();
  candidate::Stats stats;
  const auto start = solver::Clock::now();
  SyntheticClock clock{start, stats};
  const auto result = candidate::control(original, options, start, stats,
      [&](const solver::Model& reduced, const solver::SolveOptions& child_options, solver::Clock::time_point child_start) {
        ++synthetic_core_calls;
        auto child = candidate::make_shell(reduced);
        const auto cone = identity_assess_cone();
        const auto near = assess_point(4.0 / 3.0 - 5.06e-8);
        const auto exact = assess_point(4.0 / 3.0);
        const std::vector<double> dual{-1.0 / 3.0, -1.0 / 3.0};
        const std::vector<double> original_near{3, near.x[0], near.x[1]};
        require(quality_ok(solver::verify_lp_solution(reduced, near.x, dual)) &&
            !quality_ok(solver::verify_lp_solution(original, original_near, dual)),
            "继续迭代夹具须真实过约化原验但未过原模型终验");
        child.ipm_iterations = 7;
        require(!solver::assess(cone, near, child_options, child_start, child) &&
            child.status != "optimal" && stats.oracle.rejected == 1 && stats.oracle.accepted == 0,
            "第一次原验拒绝须使当前同一core继续，不能遗留optimal提前退出");
        child.ipm_iterations = 8;
        require(solver::assess(cone, exact, child_options, child_start, child) &&
            stats.oracle.accepted == 1 && stats.oracle.started == 2,
            "同一core的后继点通过原验才可停止；不得重启core或另加epoch");
        return child;
      }, [&] { return clock(); });
  original_quality(original, result, 11);
  require(stats.oracle.completed == 2 && stats.oracle.rejected == 1 && stats.oracle.accepted == 1 &&
      stats.oracle.events.size() == 2 && stats.oracle.events[0].iteration == 7 && stats.oracle.events[1].iteration == 8 &&
      stats.oracle.selected_event == 1 && stats.postsolve.completed == 2 && stats.assessments.completed == 2 &&
      result.ipm_iterations == 8 && stats.child.started == 1 && stats.child.completed == 1,
      "拒绝与接纳须逐事件保留真实恢复、原验与迭代编号；owner不得尾部重复恢复");
  timing_contract(result, stats, true);
  ++transaction_cases;
  details.push_back({"oracle_reject_then_accept_same_core", "transaction"});
}

void check_oracle_tls() {
  NativeOracleFixture first, second;
  auto* previous = candidate::active_oracle;
  {
    candidate::ScopedOracle outer(&*first.oracle);
    require(candidate::active_oracle == &*first.oracle && candidate::oracle_active(), "外层TLS须指向具名oracle owner");
    original_quality(first.original, real_wrapper(first.original, first.options, false), 88.0 / 3.0);
    const auto unchanged = coupled();
    original_quality(unchanged, real_wrapper(unchanged, first.options, true), 25.0 / 3.0);
    require(candidate::active_oracle == &*first.oracle && first.stats.oracle.started == 0,
        "真实关闭和NotReduced公开入口须隔离外层oracle，并在返回后恢复TLS");
    {
      candidate::ScopedOracle disabled(nullptr);
      require(!candidate::oracle_active(), "关闭与NotReduced嵌套应显式隔离外层oracle");
      { candidate::ScopedOracle inner(&*second.oracle); require(candidate::active_oracle == &*second.oracle, "内层TLS装入"); }
      require(!candidate::oracle_active(), "内层正常退出须恢复nullptr隔离");
    }
    require(candidate::active_oracle == &*first.oracle, "关闭隔离正常退出须恢复外层");
    try { candidate::ScopedOracle inner(&*second.oracle); throw std::runtime_error("TLS作用域异常"); }
    catch (const std::runtime_error&) {}
    require(candidate::active_oracle == &*first.oracle && first.stats.oracle.started == 0 && second.stats.oracle.started == 0,
        "异常析构须恢复TLS，作用域切换本身不得伪造oracle事件");
  }
  require(candidate::active_oracle == previous, "所有owner析构前必须恢复此前TLS指针");
  ++transaction_cases;
  details.push_back({"oracle_tls_scope_restores_previous", "transaction"});
}

void check_owner_without_tail() {
  const auto original = coupled(true);
  const auto options = policy();
  const auto start = solver::Clock::now();
  candidate::Stats stats;
  SyntheticClock clock{start, stats};
  const auto result = candidate::control(original, options, start, stats,
      [&](const solver::Model& reduced, const solver::SolveOptions&, solver::Clock::time_point) {
        ++synthetic_core_calls;
        const std::vector<double> x{4.0 / 3.0, 4.0 / 3.0}, y{-1.0 / 3.0, -1.0 / 3.0};
        require(quality_ok(solver::verify_lp_solution(reduced, x, y)) &&
            candidate::consult_current(true, reduced, x, y, 137) == candidate::OracleDecision::accept,
            "合成core必须显式提交真实通过的当前约化点，不能靠返回status冒充hook");
        auto child = bait(reduced);
        child.solution.assign(2, 999); child.row_dual.assign(2, 999);
        child.verification_seconds = 0.00001;
        return child;
      }, [&] { return clock(); });
  original_quality(original, result, 88.0 / 3.0);
  legacy_bait(result);
  require(stats.oracle.started == 1 && stats.oracle.completed == 1 && stats.oracle.selected_event == 0 &&
      stats.postsolve.started == 1 && stats.postsolve.completed == 1 && stats.assessments.started == 1 &&
      stats.assessments.completed == 1 && result.solution == std::vector<double>({3, 4.0 / 3.0, 4.0 / 3.0}),
      "owner须提交oracle拥有的原坐标点；不能用child诱饵或尾部再恢复覆盖");
  timing_contract(result, stats, true);
  ++transaction_cases;
  details.push_back({"oracle_owner_commits_without_tail_recovery", "transaction"});
}

void check_native_replay() {
  NativeOracleFixture fixture;
  const std::vector<double> a{4.0 / 3.0, 4.0 / 3.0}, b{0, 0}, y{-1.0 / 3.0, -1.0 / 3.0};
  auto& oracle = *fixture.oracle;
  require(oracle.evaluate(fixture.reduced, a, y, 2) == candidate::OracleDecision::accept && oracle.candidate,
      "同一非空恢复栈的首个A点须通过原模型终验");
  const auto owned_a = *oracle.candidate;
  require(oracle.evaluate(fixture.reduced, b, y, 3) == candidate::OracleDecision::reject && oracle.candidate &&
      oracle.candidate->x != owned_a.x && !quality_ok(oracle.candidate->checked),
      "同一Highs实例第二次B恢复须实际改变原坐标点并保真实拒绝质量");
  require(same_vector(owned_a.x, {3, a[0], a[1]}) && quality_ok(owned_a.checked),
      "首个A拥有的向量不能被原生解缓存的后续恢复覆盖");
  require(oracle.evaluate(fixture.reduced, a, y, 4) == candidate::OracleDecision::accept && oracle.candidate &&
      same_vector(oracle.candidate->x, owned_a.x) && same_vector(oracle.candidate->y, owned_a.y) &&
      same_verification(oracle.candidate->checked.primal, owned_a.checked.primal) &&
      same_optimality(oracle.candidate->checked.optimality, owned_a.checked.optimality),
      "同一presolve栈A→B→A须恢复相同数值与完整原质量，不能复用末次缓存冒充");
  require(fixture.stats.oracle.events.size() == 3 && fixture.stats.oracle.started == 3 && fixture.stats.oracle.completed == 3 &&
      fixture.stats.oracle.accepted == 2 && fixture.stats.oracle.rejected == 1 &&
      fixture.stats.postsolve.completed == 3 && fixture.stats.assessments.completed == 3 &&
      fixture.stats.oracle.candidate_event == 2 && !fixture.stats.oracle.selected_event,
      "非空重放须实际发生三次恢复和原验，局部候选不等于owner已提交");
  fixture.no_native_iterations();
  ++transaction_cases;
  details.push_back({"native_postsolve_replays_A_B_A_without_lp", "transaction"});
}

void check_oracle_failure(bool exception) {
  NativeOracleFixture fixture;
  fixture.error_after_assessment = exception;
  fixture.expire_after_assessment = !exception;
  const std::vector<double> x{4.0 / 3.0, 4.0 / 3.0}, y{-1.0 / 3.0, -1.0 / 3.0};
  const auto decision = fixture.oracle->evaluate(fixture.reduced, x, y, 11);
  const auto expected = exception ? candidate::OracleDecision::error : candidate::OracleDecision::budget;
  require(fixture.injected && decision == expected && fixture.oracle->terminal == expected &&
      !fixture.oracle->candidate && fixture.oracle->current_frame == nullptr &&
      fixture.stats.oracle.started == 1 && fixture.stats.oracle.completed == 1 && fixture.stats.oracle.events.size() == 1,
      "oracle内部预算与异常须不抛穿assess，保终止类型并释放局部frame引用");
  const auto& event = fixture.stats.oracle.events.front();
  require(event.postsolve.started == 1 && event.postsolve.completed == 1 && event.assessments.started == 1 &&
      event.assessments.completed == 1 && event.original_quality && event.original_quality->numerically_optimal &&
      event.budget.checks > 0 && event.budget.loop_units > 0 && event.timings.recovery > 0 && event.timings.verification > 0 &&
      fixture.stats.assessments.completed == 1 && !fixture.stats.oracle.selected_event && !fixture.stats.accepted &&
      std::string(event.decision) == (exception ? "error" : "budget") &&
      (exception ? fixture.stats.oracle.errors : fixture.stats.oracle.budget_failures) == 1,
      "原验已完成后失败可留诊断质量，但不得丢已付工作或伪造owner提交");
  fixture.no_native_iterations();
  if (exception) { ++exception_cases; details.push_back({"oracle_error_preserves_event_work", "exception"}); }
  else { ++budget_cases; details.push_back({"oracle_budget_preserves_event_work", "budget"}); }
}

void check_real_core_oracle_failure(bool exception) {
  NativeOracleFixture fixture;
  fixture.error_after_assessment = exception;
  fixture.expire_after_assessment = !exception;
  solver::SolveResult result;
  {
    candidate::ScopedOracle scope(&*fixture.oracle);
    ++component_ipm_calls;
    // 真实HSD迭代与factor均执行；仅oracle的Now在原验完成后注入确定性失败。
    // core自身继续使用同一个真实Clock epoch，不伪造求解计时或更换公开预算。
    result = solver::v96_core(fixture.reduced, fixture.options, fixture.start);
  }
  const auto expected = exception ? candidate::OracleDecision::error : candidate::OracleDecision::budget;
  require(fixture.injected && result.status == (exception ? "numerical_error" : "time_limit") &&
      result.backend_status == result.status && fixture.oracle->terminal == expected &&
      result.ipm_iterations && *result.ipm_iterations > 0 && result.ipm_factorizations && *result.ipm_factorizations > 0 &&
      fixture.stats.oracle.started == 1 && fixture.stats.oracle.completed == 1 && fixture.stats.oracle.events.size() == 1,
      "真实assess的budget/error桥接必须由实际HSD core返回既有失败类型，并保已付迭代与factor");
  const auto& event = fixture.stats.oracle.events.front();
  require(event.iteration == result.ipm_iterations && event.postsolve.completed == 1 &&
      event.assessments.completed == 1 && event.original_quality && event.budget.checks > 0 &&
      event.budget.loop_units > 0 && event.timings.recovery > 0 && event.timings.verification > 0 &&
      !fixture.oracle->candidate && !fixture.stats.oracle.selected_event && !fixture.stats.accepted &&
      (exception ? fixture.stats.oracle.errors : fixture.stats.oracle.budget_failures) == 1,
      "真实core异常收口不能抹去oracle事件的恢复、原验与部分成本，也不得提交原候选");
  require(result.solve_seconds >= 0 && result.verification_seconds >= 0 && result.total_seconds >= result.solve_seconds,
      "真实core保自身时钟测量，不把oracle合成时钟写成其耗时证据");
  fixture.no_native_iterations();
  if (exception) { ++exception_cases; details.push_back({"oracle_real_core_error_bridge", "exception"}); }
  else { ++budget_cases; details.push_back({"oracle_real_core_budget_bridge", "budget"}); }
}

void coupled_strict_log(const coupled_state::Log& log) {
  require(log.enabled && log.last && log.directions_started == log.directions_completed &&
      std::string_view(log.last->work.status) == "accepted" && log.last->work.final_measure &&
      *log.last->work.final_measure <= 1 && log.preconditioner_started == log.preconditioner_completed &&
      log.products_started == log.products_completed && log.true_checks_started == log.true_checks_completed &&
      log.krylov_started == log.krylov_completed,
      "联立成功账本须对应完整真实方向与所有已返回回调");
  for (long double value : log.last->work.block_measures)
    require(std::isfinite(value) && value <= 1 && value >= 0, "五块缺陷须分别严格通过");
}

void coupled_explicit_return_log(const coupled_state::Log& log) {
  require(log.enabled && log.last && log.directions_started == log.directions_completed &&
      log.preconditioner_started == log.preconditioner_completed && log.products_started == log.products_completed &&
      log.krylov_started == log.krylov_completed && log.true_checks_started == log.true_checks_completed &&
      log.initial_forcing_returns <= log.initial_forcing_checks && log.initial_forcing_checks <= log.directions_started &&
      log.initial_forcing_returns <= log.directions_completed,
      "实际新策略返回必须保全部已付调用与显式forcing累计守恒");
  const auto& work = log.last->work;
  if (std::string_view(work.status) == "accepted") { coupled_strict_log(log); return; }
  require(std::string_view(work.status) == "forced_initial" && work.initial_forcing &&
      work.initial_forcing->measure && *work.initial_forcing->measure <= 1 &&
      std::string_view(work.initial_forcing->decision) == "returned" &&
      work.initial_forcing_checks == 1 && work.initial_forcing_returns == 1 && work.final_measure && *work.final_measure > 1 &&
      work.block_measures[3] <= 1 && work.block_measures[4] <= 1 && work.krylov_started == 0 && work.cycles_started == 0 &&
      work.preconditioner_started == 1 && work.products_started == 1 && work.true_checks_started == 1,
      "非strict返回只允许原始初P的明确forcing快照，两个互补块仍必须严格");
}

template<class Scalar, class Explicit>
void coupled_independent_equations(const solver::detail::IpmHomogeneousScaling& scaling,
    const std::vector<double>& rhs, const std::vector<Scalar>& answer, Explicit explicit_product) {
  const auto image = explicit_product(answer);
  const auto tolerance = solver::detail::ipm_homogeneous_scaled_tolerances(scaling, rhs);
  require(image.size() == rhs.size(), "独立物理方程维数一致");
  const coupled_kernel::Layout layout{scaling.layout.columns, scaling.layout.rows, scaling.layout.inequality_rows.size()};
  for (std::size_t i = 0; i < rhs.size(); ++i)
    require(std::isfinite(image[i]) && std::abs(image[i] - rhs[i]) <= tolerance[layout.block(i)],
        "交付完整方向须独立代回显式五块物理方程，不用被测乘积给自身作证");
}

void check_coupled_mixed_equations() {
  using namespace solver::detail;
  const std::vector<std::int64_t> starts{0, 2}, rows{0, 1};
  const std::vector<double> values{1, 1}, costs{-1}, bounds{2, 3}, slack{4, 0}, dual{1, -3};
  const auto scaling = make_ipm_homogeneous_scaling(1, {true, false}, slack, dual, 2, 8);
  const auto inverse = [](std::span<const double> rhs) {
    const double dx = rhs[2], dz0 = (dx - rhs[1]) / 4;
    return std::vector<double>{dx, dz0, rhs[0] - dz0};
  };
  const auto constant = inverse(ipm_homogeneous_scaled_constant_rhs(scaling, costs, bounds));
  const auto denominator = ipm_homogeneous_scaled_denominator(scaling, constant);
  const std::vector<double> rhs{3, -2, 3, 1.5, -2, -2};
  const auto explicit_product = [](const auto& v) {
    return std::vector<long double>{static_cast<long double>(v[1]) + v[2] - v[4],
        static_cast<long double>(v[0]) + v[3] - 2 * v[4], static_cast<long double>(v[0]) - 3 * v[4],
        (-static_cast<long double>(v[0]) + 2 * v[1] + 3 * v[2] + v[5]) / 2,
        (static_cast<long double>(v[3]) + 4 * v[1]) / 2, 2 * static_cast<long double>(v[4]) + 0.5L * v[5]};
  };
  // 每个物理基向量独立核对矩阵列，避免只在一个恰好相同的解上互相作证。
  for (std::size_t col = 0; col < scaling.dimension(); ++col) {
    std::vector<double> basis(scaling.dimension(), 0);
    basis[col] = 1;
    std::vector<long double> observed;
    solver::v94_hsd_product(scaling, starts, rows, values, costs, bounds, slack, dual, 2, 8,
        std::span<const double>(basis), observed, {});
    require(observed == explicit_product(basis), "混合等式和非单位齐次尺度的每个物理矩阵列须符合独立手算");
  }
  const auto p = [&](std::span<const double> input, bool) {
    return ipm_homogeneous_scaled_recover(scaling, costs, bounds, input,
        inverse(ipm_homogeneous_scaled_reduced_rhs(scaling, input)), constant, denominator);
  };
  const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) {
    solver::v94_hsd_product(scaling, starts, rows, values, costs, bounds, slack, dual, 2, 8, input, output, {});
  };
  coupled_state::Log log;
  log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
  const auto answer = solver::v96_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 7);
  coupled_strict_log(log);
  coupled_independent_equations(scaling, rhs, answer, explicit_product);
  const std::vector<double> expected{36.0 / 17, 3.0 / 17, 43.0 / 17, -80.0 / 17, -5.0 / 17, -48.0 / 17};
  for (std::size_t i = 0; i < answer.size(); ++i)
    require(std::abs(answer[i] - expected[i]) < 1e-11, "六维物理方向须符合独立解析解");
  require(log.last->attempt_iteration == 7 && std::string_view(log.last->role) == "affine" &&
      log.preconditioner_started == 1 && log.krylov_started == 0, "精确混合P不增加额外IR或FG列");
  ++transaction_cases;
  details.push_back({"coupled_mixed_physical_equations", "transaction"});
}

struct CoupledScalarSystem {
  solver::ConeModel cone;
  solver::Point point;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  solver::detail::IpmHomogeneousScaling scaling;
  std::unique_ptr<solver::KktFactor> factor;
  CoupledScalarSystem() {
    options.ipm.linear_solver = "qdldl";
    cone.starts = {0}; cone.rhs = {1}; cone.inequality = {true}; cone.barrier_rows = 1;
    point.s = {1}; point.z = {1};
    scaling = solver::detail::make_ipm_homogeneous_scaling(0, {true}, point.s, point.z, 1, 1);
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    factor->factor({1}, options, start, result);
  }
};

void check_coupled_imperfect_constant() {
  using namespace solver::detail;
  const std::vector<std::int64_t> starts{0}, rows;
  const std::vector<double> empty, one{1}, constant{-0.5}, rhs{0, 0, 0, 1};
  const auto scaling = make_ipm_homogeneous_scaling(0, {true}, one, one, 1, 1);
  const auto denominator = ipm_homogeneous_scaled_denominator(scaling, constant);
  require(denominator == 1.25L, "故意不精确常量仍只形成近似正D，不把它宣称精确Schur");
  const auto p = [&](std::span<const double> input, bool) {
    auto reduced = ipm_homogeneous_scaled_reduced_rhs(scaling, input);
    reduced[0] = -reduced[0];
    return ipm_homogeneous_scaled_recover(scaling, empty, one, input, reduced, constant, denominator);
  };
  const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) {
    solver::v94_hsd_product(scaling, starts, rows, empty, empty, one, one, one, 1, 1, input, output, {});
  };
  const auto explicit_product = [](const auto& v) {
    return std::vector<long double>{static_cast<long double>(v[1]) - v[2],
        static_cast<long double>(v[0]) + v[3], static_cast<long double>(v[0]) + v[1],
        static_cast<long double>(v[2]) + v[3]};
  };
  IpmBoundedNewtonInfo old;
  solve_ipm_homogeneous_scaled(scaling, rhs, p, a, old);
  require(!old.tolerance_satisfied && old.measure > 1, "原三IR政策在同一个扰动P下仍应如实inexact");
  coupled_state::Log log;
  log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
  const auto answer = solver::v96_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 1);
  coupled_strict_log(log);
  coupled_independent_equations(scaling, rhs, answer, explicit_product);
  require(log.krylov_started > 0 && log.last->work.initial_measure && *log.last->work.initial_measure > 1,
      "新策略须实际从不精确P进入FG，而非把原inexact值改标签");
  CoupledScalarSystem actual;
  const auto direction = solver::homogeneous_scaled_direction(actual.cone, actual.point, 1, 1,
      {0}, {}, 0, 0, {0}, 1, constant, actual.scaling, denominator,
      *actual.factor, actual.options, actual.start, actual.result, "corrector", 2);
  require(direction.v94_wide && direction.point.z.empty() && direction.point.s.empty(),
      "活动策略必须交付宽方向，不能先窄Point再补LD摘要");
  const std::vector<long double> physical{direction.v94_wide->z[0], direction.v94_wide->s[0],
      direction.v94_wide->tau, direction.v94_wide->kappa};
  coupled_independent_equations(scaling, rhs, physical, explicit_product);
  coupled_strict_log(log);
  require(std::string_view(log.last->role) == "corrector" && log.last->attempt_iteration == 2 &&
      *actual.result.ipm_homogeneous_direction_solves > 1 && *actual.result.ipm_homogeneous_scaled_refinement_steps > 0 &&
      *actual.result.ipm_homogeneous_scaled_inexact_returns == 0 && *actual.result.ipm_weighted_krylov_steps == 0,
      "真实adapter须用literal调用站标签，严格FG工作计现direct，不能误记旧inexact或K0 weighted");
  ++transaction_cases;
  details.push_back({"coupled_imperfect_constant_strict", "transaction"});
}

void check_coupled_empty_blocks() {
  using namespace solver::detail;
  const std::vector<double> zero{0}, negative{-2}, empty;
  for (bool equality : {false, true}) {
    const auto scaling = equality ? make_ipm_homogeneous_scaling(1, {false}, zero, negative, 2, 8) :
        make_ipm_homogeneous_scaling(0, {}, empty, empty, 1, 1);
    const std::vector<double> rhs = equality ? std::vector<double>{-1, -0.5, -2.5, 0.5} : std::vector<double>{1, 2};
    const std::vector<double> costs = equality ? std::vector<double>{2} : std::vector<double>{};
    const std::vector<double> bounds = equality ? std::vector<double>{3} : std::vector<double>{};
    const auto inverse = [&](std::span<const double> input) {
      return equality ? std::vector<double>{input[1], input[0]} : std::vector<double>{};
    };
    const auto constant = inverse(ipm_homogeneous_scaled_constant_rhs(scaling, costs, bounds));
    const auto denominator = ipm_homogeneous_scaled_denominator(scaling, constant);
    const auto p = [&](std::span<const double> input, bool) {
      auto result = ipm_homogeneous_scaled_recover(scaling, costs, bounds, input,
          inverse(ipm_homogeneous_scaled_reduced_rhs(scaling, input)), constant, denominator);
      if (!equality) result[0] *= 0.5;
      return result;
    };
    const auto explicit_product = [&](const auto& v) {
      return equality ? std::vector<long double>{static_cast<long double>(v[1]) + 2 * v[2],
          static_cast<long double>(v[0]) - 3 * v[2], (2 * static_cast<long double>(v[0]) + 3 * v[1] + v[3]) / 2,
          2 * static_cast<long double>(v[2]) + 0.5L * v[3]} :
          std::vector<long double>{v[1], static_cast<long double>(v[0]) + v[1]};
    };
    const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) { output = explicit_product(input); };
    coupled_state::Log log;
    log.enabled = true;
    coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
    const auto answer = solver::v96_solve_coupled(scaling, rhs, p, a, a, {}, "corrector", 3);
    coupled_strict_log(log);
    coupled_independent_equations(scaling, rhs, answer, explicit_product);
    require(log.last->work.tolerances[3] == 1e-12 && (!equality ? log.krylov_started > 0 : true),
        "空互补块保绝对门槛，纯标量系统仍有真实FG空间");
  }
  ++transaction_cases;
  details.push_back({"coupled_equality_scalar_blocks", "transaction"});
}

void check_coupled_real_mapping() {
  Mode quiet(true);
  for (bool maximum : {false, true}) {
    const auto model = coupled(true, maximum);
    const auto options = policy();
    const auto start = solver::Clock::now();
    candidate::Stats stats;
    coupled_state::Log log;
    log.enabled = true;
    coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
    const auto result = candidate::control(model, options, start, stats,
        [&](const solver::Model& reduced, const solver::SolveOptions& child_options, solver::Clock::time_point child_start) {
          require(child_start == start, "新核不得重置预处理与原验共享epoch");
          ++component_ipm_calls;
          return solver::v96_core(reduced, child_options, child_start);
        });
    original_quality(model, result, maximum ? 104.0 / 3.0 : 88.0 / 3.0);
    coupled_explicit_return_log(log);
    require(std::string_view(stats.route) == "reduced" && stats.accepted && stats.oracle.accepted == 1 &&
        stats.native.basis_valid == false && result.solution[0] == 3 && *result.ipm_iterations > 0 &&
        log.constant_calls_started == log.constant_calls_completed &&
        log.constant_calls_started == *result.ipm_homogeneous_constant_solves &&
        log.preconditioner_started == *result.ipm_homogeneous_direction_solves &&
        log.products_started == *result.ipm_homogeneous_scaled_products &&
        log.krylov_started == *result.ipm_homogeneous_scaled_refinement_steps &&
        *result.ipm_refinement_steps >= *result.ipm_homogeneous_scaled_refinement_steps &&
        *result.ipm_homogeneous_scaled_inexact_returns == log.initial_forcing_returns &&
        *result.ipm_homogeneous_constant_repairs == 0 &&
        *result.ipm_weighted_krylov_steps == 0 && log.directions_started == 2 * *result.ipm_iterations,
        "真实预处理/原验与固定列最大化恢复保持，显式初forcing须按实际末交付计公共inexact且不记repair");
  }
  ++mapping_cases;
  details.push_back({"coupled_real_core_mapping", "mapping"});
}

void check_coupled_scope_isolation() {
  const auto model = coupled();
  const auto options = policy();
  const auto expected = real_core(model, options);
  coupled_state::Log outer;
  outer.enabled = true;
  {
    coupled_state::Scope scope(&outer);
    same_non_timing(real_wrapper(model, options, false), expected);
    require(coupled_state::current() == &outer && outer.directions_started == 0 && outer.constant_calls_started == 0,
        "off wrapper必须隔离外层coupled TLS并完整还原，不能继承隐式策略");
    auto stopped_options = options;
    stopped_options.time_limit = 0;
    const auto stopped = real_wrapper(model, stopped_options, true);
    require(stopped.status == "time_limit" && *stopped.ipm_factorizations == 0 &&
        *stopped.ipm_homogeneous_scaled_products == 0 && coupled_state::current() == &outer && outer.directions_started == 0,
        "零预算旁路不得启动联立初始化或污染外层TLS");
    coupled_state::Log disabled;
    {
      coupled_state::Scope inner(&disabled);
      require(!coupled_state::enabled() && coupled_state::current() == nullptr, "关闭局部Log须隔离外层激活状态");
    }
    require(coupled_state::current() == &outer, "具名嵌套Scope必须复原前指针");
  }
  require(coupled_state::context == nullptr, "最外层Scope退出后不得留下悬空Log");
  ++transaction_cases;
  details.push_back({"coupled_disabled_and_scope", "transaction"});
}

void check_coupled_budget_partial() {
  using namespace solver::detail;
  enum class Boundary { before_p_direct, after_p_direct, before_a_count, in_a, final };
  for (const auto boundary : {Boundary::before_p_direct, Boundary::after_p_direct,
                            Boundary::before_a_count, Boundary::in_a, Boundary::final}) {
    CoupledScalarSystem system;
    const std::vector<double> constant{-1}, rhs{0, 0, 0, 1};
    const auto denominator = ipm_homogeneous_scaled_denominator(system.scaling, constant);
    coupled_state::Log log;
    log.enabled = true;
    coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
    bool injected = false;
    const auto budget = [&] {
      if (!log.last) return;
      const auto& work = log.last->work;
      const bool stop = boundary == Boundary::before_p_direct ? work.preconditioner_started > 0 && *system.result.ipm_homogeneous_direction_solves == 0 :
          boundary == Boundary::after_p_direct ? *system.result.ipm_homogeneous_direction_solves > 0 && work.preconditioner_completed == 0 :
          boundary == Boundary::before_a_count ? work.products_started > 0 && *system.result.ipm_homogeneous_scaled_products == 0 :
          boundary == Boundary::in_a ? *system.result.ipm_homogeneous_scaled_products > 0 && work.products_completed == 0 :
          work.final_measure && *work.final_measure <= 1;
      if (stop) { injected = true; throw solver::BudgetExpired{}; }
    };
    const auto p = [&](std::span<const double> input, bool refinement) {
      const auto reduced_rhs = ipm_homogeneous_scaled_reduced_rhs(system.scaling, input, budget);
      const auto reduced = system.factor->solve_homogeneous_direct(reduced_rhs, system.options,
          system.start, system.result, false, refinement);
      return ipm_homogeneous_scaled_recover(system.scaling, {}, system.cone.rhs, input,
          reduced, constant, denominator, budget);
    };
    const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) {
      budget();
      ++*system.result.ipm_homogeneous_scaled_products;
      solver::v94_hsd_product(system.scaling, system.cone.starts, system.cone.rows, system.cone.values,
          system.cone.costs, system.cone.rhs, system.point.s, system.point.z, 1, 1, input, output, budget);
      budget();
    };
    bool original = false;
    std::vector<long double> output{101, 202};
    try { output = solver::v96_solve_coupled(system.scaling, rhs, p, a, a, budget, "affine", 9); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && injected && output == std::vector<long double>({101, 202}) && log.last &&
        std::string_view(log.last->work.status) == "budget" && log.directions_started == 1 && log.directions_completed == 0 &&
        log.preconditioner_started == 1 && log.preconditioner_completed <= 1 &&
        *system.result.ipm_homogeneous_direction_solves == (boundary == Boundary::before_p_direct ? 0 : 1) &&
        *system.result.ipm_homogeneous_scaled_products == (boundary == Boundary::in_a || boundary == Boundary::final ? 1 : 0) &&
        *system.result.ipm_homogeneous_scaled_inexact_returns == 0 && *system.result.ipm_homogeneous_scaled_refinement_steps == 0,
        "caller开始与内部actual direct/product有精确partial前缀差，预算保持原类型且不能提前提交");
    if (boundary == Boundary::final)
      require(log.last->work.final_measure && *log.last->work.final_measure <= 1 &&
          log.true_checks_completed == 1, "末预算中断须保已过严格真验但未交付的快照");
  }
  ++budget_cases;
  details.push_back({"coupled_budget_and_count_partial", "budget"});
}

struct EnergySystem {
  solver::ConeModel cone;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  const std::vector<double> h;
  std::unique_ptr<solver::KktFactor> factor;
  explicit EnergySystem(double first_h = 2) : h{first_h, 0, 3} {
    options.ipm.linear_solver = "qdldl";
    cone.costs = {-1, 2}; cone.rhs = {3, -2, 5};
    cone.starts = {0, 3, 5}; cone.rows = {0, 1, 2, 0, 1}; cone.values = {1, 3, 5, 2, -1};
    cone.inequality = {true, false, true}; cone.barrier_rows = 2;
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    factor->factor(h, options, start, result);
    require(result.ipm_primal_regularization && result.ipm_dual_regularization &&
        *result.ipm_eliminated_singletons == 1 && factor->v93_regularized_h().size() == 3,
        "名义夹具须实际形成r、全行R及一个消元单例行");
  }
};

void energy_near(long double actual, long double expected, const char* message) {
  require(std::abs(actual - expected) <= 3e-14L * std::max(1.0L, std::abs(expected)), message);
}

using EnergyMatrix = std::array<std::array<long double, 5>, 5>;
EnergyMatrix explicit_nominal(const EnergySystem& system) {
  EnergyMatrix matrix{};
  const auto stored = system.factor->v93_regularized_h();
  for (std::size_t column = 0; column < 2; ++column) {
    matrix[column][column] = *system.result.ipm_primal_regularization;
    for (auto k = system.cone.starts[column]; k < system.cone.starts[column + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      const auto row_index = 2 + static_cast<std::size_t>(system.cone.rows[position]);
      matrix[column][row_index] = matrix[row_index][column] = system.cone.values[position];
    }
  }
  for (std::size_t row_index = 0; row_index < 3; ++row_index) matrix[2 + row_index][2 + row_index] = -stored[row_index];
  return matrix;
}

std::vector<double> independent_energy_dense(EnergyMatrix matrix, std::vector<long double> rhs) {
  for (std::size_t column = 0; column < 5; ++column) {
    std::size_t pivot = column;
    for (std::size_t row_index = column + 1; row_index < 5; ++row_index)
      if (std::abs(matrix[row_index][column]) > std::abs(matrix[pivot][column])) pivot = row_index;
    std::swap(matrix[column], matrix[pivot]); std::swap(rhs[column], rhs[pivot]);
    require(matrix[column][column] != 0, "独立完整名义KKT必须非奇异");
    const auto divisor = matrix[column][column];
    for (std::size_t entry = column; entry < 5; ++entry) matrix[column][entry] /= divisor;
    rhs[column] /= divisor;
    for (std::size_t row_index = 0; row_index < 5; ++row_index) if (row_index != column) {
      const auto multiple = matrix[row_index][column];
      for (std::size_t entry = column; entry < 5; ++entry) matrix[row_index][entry] -= multiple * matrix[column][entry];
      rhs[row_index] -= multiple * rhs[column];
    }
  }
  return std::vector<double>(rhs.begin(), rhs.end());
}

void check_energy_nominal_identity() {
  EnergySystem system;
  const auto matrix = explicit_nominal(system);
  const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5};
  const auto dense = independent_energy_dense(matrix, std::vector<long double>(rhs.begin(), rhs.end()));
  const auto direct = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, true);
  require(direct.size() == 5, "生产direct必须已经恢复被消元的单例行，而非凝聚向量");
  for (double value : direct) require(std::isfinite(value), "实际近似常量必须是已恢复的有限完整向量");
  // raw direct只提供近似P；独立dense核精确名义解，实际解保留其真实缺陷eta。
  energy_near(static_cast<long double>(system.factor->v93_regularized_h()[2]) * direct[4],
      5.0L * direct[0] - rhs[4], "被消元行须按完整名义方程真实恢复，不能用凝聚坐标冒充");
  for (int kind : {0, 1, 2}) {
    auto v = kind == 0 ? dense : direct;
    if (kind == 2) v[0] += 0.25;
    energy_kernel::Work work;
    const auto denominator = energy_kernel::compute(2, v, system.factor->v93_regularized_h(),
        *system.result.ipm_primal_regularization, work);
    std::array<long double, 5> error{};
    for (std::size_t i = 0; i < 5; ++i) {
      error[i] = -rhs[i];
      for (std::size_t j = 0; j < 5; ++j) error[i] += matrix[i][j] * v[j];
    }
    const long double eta = v[0] * error[0] + v[1] * error[1] -
        v[2] * error[2] - v[3] * error[3] - v[4] * error[4];
    const long double schur = 1 - (-v[0] + 2.0L * v[1] + 3.0L * v[2] - 2.0L * v[3] + 5.0L * v[4]) / 2;
    energy_near(denominator - schur, eta, "完整名义Kdelta必须满足D-S=vx*ed-vz*ep，不能忽略近似缺陷");
    if (kind == 0) energy_near(eta, 0, "独立dense精确名义解应消去eta");
    if (kind == 2) require(std::abs(eta) > 1e-3L, "实际扰动必须产生非零缺陷，不能只核精确零特例");
    require(work.primal_terms == 2 && work.row_terms == 3, "等式及单例恢复行必须全部计入能量");
  }
  ++transaction_cases;
  details.push_back({"energy_nominal_full_kkt_identity", "transaction"});
}

void check_energy_stored_singleton() {
  static_assert(std::is_same_v<decltype(std::declval<const solver::KktFactor&>().v93_regularized_h()), std::span<const double>>);
  EnergySystem system(1048576);
  const auto stored = system.factor->v93_regularized_h();
  const double r = *system.result.ipm_primal_regularization, delta = *system.result.ipm_dual_regularization;
  std::size_t rounded_row = stored.size();
  for (std::size_t i = 0; i < stored.size(); ++i) {
    const long double recomputed = static_cast<long double>(system.h[i]) + delta;
    require(stored[i] == static_cast<double>(recomputed), "getter必须返回factor实际已舍入R");
    if (static_cast<long double>(stored[i]) != recomputed) rounded_row = i;
  }
  require(rounded_row < stored.size(), "夹具必须实际区分stored double与LD重新计算");
  std::vector<double> v{2, -3, 4, 5, 6};
  v[2 + rounded_row] = 1e16;
  long double primal = 0, rows = 0, recomputed_rows = 0;
  for (std::size_t i = 0; i < 2; ++i) primal += (static_cast<long double>(r) * v[i]) * v[i];
  for (std::size_t i = 0; i < 3; ++i) {
    rows += (static_cast<long double>(stored[i]) * v[2 + i]) * v[2 + i];
    recomputed_rows += ((static_cast<long double>(system.h[i]) + delta) * v[2 + i]) * v[2 + i];
  }
  coupled_state::Log coupled_log; coupled_log.enabled = true;
  coupled_state::Scope coupled_scope(&coupled_log);
  energy_state::Log log; log.enabled = true;
  energy_state::Scope scope(&log);
  const auto factors_before = system.result.ipm_factorizations;
  const auto solves_before = system.result.ipm_homogeneous_constant_solves;
  const auto denominator = solver::v96_energy_denominator(*system.factor, 2, v, system.result, {}, 7);
  require(denominator == (1 + primal) + rows && denominator != (1 + primal) + recomputed_rows &&
      log.started == 1 && log.completed == 1 && log.primal_terms == 2 && log.row_terms == 3 &&
      log.last && log.last->attempt_iteration == 7 && log.last->work.denominator == denominator &&
      system.result.ipm_factorizations == factors_before && system.result.ipm_homogeneous_constant_solves == solves_before,
      "真实adapter只作全向量正归约、使用已有r和storedR，不添加solve/factor");
  // 存储舍入用大向量放大；曲率双计另用普通向量，避免几十的差额淹没在1e32的尾数里。
  const std::vector<double> moderate{2, -3, 4, 5, 6};
  const auto moderate_denominator = solver::v96_energy_denominator(*system.factor, 2, moderate, system.result, {}, 8);
  const long double moderate_expected = (1 + 13.0L * r) +
      ((16.0L * stored[0] + 25.0L * stored[1]) + 36.0L * stored[2]);
  energy_near(moderate_denominator, moderate_expected, "普通向量的真实能量仍须只用原始r和完整三行R");
  const long double double_counted = moderate_expected + (25.0L / stored[2] * moderate[0]) * moderate[0];
  require(double_counted > moderate_denominator + 1, "若再用凝聚r+a²/R会把已恢复单例曲率重复计入");
  ++mapping_cases;
  details.push_back({"energy_stored_diagonal_singleton", "mapping"});
}

void check_energy_scope_budget() {
  const auto model = coupled(); const auto options = policy();
  const auto expected = real_core(model, options);
  energy_state::Log outer; outer.enabled = true;
  {
    energy_state::Scope scope(&outer);
    same_non_timing(real_wrapper(model, options, false), expected);
    require(energy_state::current() == &outer && outer.started == 0, "off须隔离外能量scope并原样恢复");
    auto expired = options; expired.time_limit = 0;
    require(real_wrapper(model, expired, true).status == "time_limit" && outer.started == 0 &&
        energy_state::current() == &outer, "预算bypass不能激活能量或污染外账本");
    energy_state::Log disabled;
    { energy_state::Scope inner(&disabled); require(!energy_state::enabled() && !energy_state::current(), "disabled scope须屏蔽外层"); }
    require(energy_state::current() == &outer, "嵌套能量scope须恢复原指针");
  }
  require(!energy_state::context, "能量TLS退出后不能悬挂");
  CoupledScalarSystem system;
  const std::vector<double> constant{-2};
  coupled_state::Log coupled_log; coupled_log.enabled = true;
  coupled_state::Scope coupled_scope(&coupled_log);
  int checks = 0;
  energy_state::Log complete; complete.enabled = true;
  { energy_state::Scope scope(&complete); solver::v96_energy_denominator(*system.factor, 0, constant,
        system.result, [&] { ++checks; }, 4); }
  require(complete.started == 1 && complete.completed == 1 && checks > 2, "完整adapter路径提供真实预算边界");
  bool formed = false;
  for (int boundary = 1; boundary <= checks; ++boundary) {
    energy_state::Log log; log.enabled = true;
    energy_state::Scope scope(&log);
    int observed = 0; bool original = false; long double output = 123;
    try { output = solver::v96_energy_denominator(*system.factor, 0, constant, system.result,
        [&] { if (++observed == boundary) throw solver::BudgetExpired{}; }, 4); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && output == 123 && log.completed == 0 && log.started == (boundary == 1 ? 0 : 1),
        "adapter任何预算中断须保持原异常、实际started前缀且不交付D");
    if (log.last) {
      require(std::string_view(log.last->work.status) == "budget" &&
          log.row_terms == log.last->work.row_terms && log.budget_checks == log.last->work.budget_checks,
          "预算partial必须累计实际已处理项和内部检查");
      if (log.last->work.denominator) { formed = true; require(boundary == checks && log.row_terms == 1, "末预算保已形成D而非伪完成"); }
    }
  }
  require(formed, "真实adapter必须覆盖D形成后末预算中断");
  auto missing = system.result; missing.ipm_primal_regularization.reset();
  energy_state::Log invalid; invalid.enabled = true;
  {
    energy_state::Scope scope(&invalid); bool rejected = false;
    try { solver::v96_energy_denominator(*system.factor, 0, constant, missing, {}, 4); }
    catch (const solver::NumericalFailure&) { rejected = true; }
    require(rejected && invalid.started == 1 && invalid.completed == 0 && invalid.last &&
        std::string_view(invalid.last->work.status) == "numerical_error", "缺失public r必须拒绝，不得默认为0");
  }
  ++budget_cases;
  details.push_back({"energy_scope_off_budget", "budget"});
}

void check_wide_physical_dense() {
  using namespace solver::detail;
  static_assert(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits);
  const std::vector<std::int64_t> starts{0, 1}, rows{0};
  const std::vector<double> values{1}, zero{0}, one{1}, rhs{1e16, 0, 0, 1, 0};
  const auto scaling = make_ipm_homogeneous_scaling(1, {true}, one, one, 1, 1);
  const EnergyMatrix matrix{{{{0, 1, 0, 0, 0}}, {{1, 0, 1, 0, 0}}, {{0, 0, 0, 0, 1}},
      {{0, 1, 1, 0, 0}}, {{0, 0, 0, 1, 1}}}};
  // 复用独立五阶高斯消元；其输入是完整非对称J，未调用被测五块算子。
  const auto dense_double = independent_energy_dense(matrix, std::vector<long double>(rhs.begin(), rhs.end()));
  require(dense_double[0] == 1e16 && dense_double[2] == -1e16,
      "窄dense副本应确实丢掉单位低位，不能拿该副本作严格方向");
  const std::vector<long double> exact{1e16L - 1, 1e16L, -1e16L + 1, 0, 0};
  const auto explicit_product = [](const auto& v) {
    return std::vector<long double>{v[1], v[0] + v[2], v[4], v[2] + v[1], v[3] + v[4]};
  };
  require(explicit_product(exact) == std::vector<long double>(rhs.begin(), rhs.end()), "独立显式J的LD解析解必须精确对应原double RHS");
  const auto p = [](std::span<const double> input, bool) {
    const double ds = input[3] - input[0];
    return std::vector<double>{input[1] - ds, input[0], ds, input[4] - input[2], input[2]};
  };
  int columns = 0, true_checks = 0; bool saw_low_bit = false;
  std::vector<long double> last_true;
  const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) {
    if constexpr (std::is_same_v<Scalar, long double>) {
      ++true_checks; last_true.assign(input.begin(), input.end());
      for (auto value : input) if (value != static_cast<long double>(static_cast<double>(value))) saw_low_bit = true;
    } else { static_assert(std::is_same_v<Scalar, double>); ++columns; }
    solver::v94_hsd_product(scaling, starts, rows, values, zero, zero, one, one, 1, 1, input, output, {});
    require(output == explicit_product(input), "double列及LD真验都必须符合独立显式J");
  };
  coupled_state::Log log; log.enabled = true; coupled_state::Scope scope(&log);
  const auto answer = solver::v96_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 1);
  static_assert(std::is_same_v<std::remove_cv_t<decltype(answer)>, std::vector<long double>>);
  coupled_strict_log(log);
  coupled_independent_equations(scaling, rhs, answer, explicit_product);
  require(answer == exact && last_true == answer && saw_low_bit && columns > 0 && true_checks > 1 &&
      log.last->work.initial_measure && *log.last->work.initial_measure > 1,
      "初P丢单位位后须真实FG修复，LD真验对象就是原样返回的同一完整方向");
  const auto delivered = solver::v94_deliver_direction(scaling, answer, {});
  require(delivered.v94_wide && delivered.point.x.empty() && delivered.point.s.empty() && delivered.point.z.empty() &&
      delivered.scalars.tau == 0 && delivered.scalars.kappa == 0 && delivered.v94_wide->x[0] == exact[0] &&
      delivered.v94_wide->z[0] == exact[1] && delivered.v94_wide->s[0] == exact[2],
      "交付桥只填Wide，不得在真验之后又窄化到旧Point字段");
  ++transaction_cases;
  details.push_back({"wide_direction_physical_dense", "transaction"});
}

solver::HomogeneousDirection wide_fixture(long double dx, long double ds, long double dz,
    long double dt, long double dk) {
  solver::HomogeneousDirection direction;
  direction.v94_wide.emplace();
  direction.v94_wide->x = {dx}; direction.v94_wide->s = {ds}; direction.v94_wide->z = {dz};
  direction.v94_wide->tau = dt; direction.v94_wide->kappa = dk;
  return direction;
}

void check_wide_consumers() {
  solver::ConeModel cone; cone.inequality = {true};
  solver::Point point; point.x = {-1e16}; point.s = {1e16}; point.z = {1};
  const auto affine = wide_fixture(0, -1e16L + 1, 1, -1e16L + 1, 1);
  std::vector<double> h{0};
  const auto h_tau = solver::v94_corrector_rhs(cone, point, 1e16, 1, affine, 0, 1, h, {});
  require(h[0] == -1 && h_tau == -1, "实际二阶校正两种乘积都须读取LD低位，再窄h而非先窄方向");
  point.z = {1e16};
  const auto corrected = wide_fixture(1e16L + 1, -1e16L + 1, -1e16L + 1, -1e16L + 1, -1e16L + 1);
  const auto updated = solver::v94_update_point(cone, point, 1e16, 1e16, corrected, 1, {});
  require(updated.point.x == std::vector<double>{1} && updated.point.s == std::vector<double>{1} &&
      updated.point.z == std::vector<double>{1} && updated.tau == 1 && updated.kappa == 1,
      "实际点更新的五个方向都必须LD积加后仅末finite，不能遗留double消费者");
  solver::Point unit; unit.x = {0}; unit.s = {1}; unit.z = {1};
  const auto options = policy();
  const long double negative = -(1 + std::ldexp(1.0L, -53));
  for (int coordinate : {0, 1, 2, 3}) {
    auto boundary = wide_fixture(0, 0, 0, 0, 0);
    if (coordinate == 0) boundary.v94_wide->s[0] = negative;
    if (coordinate == 1) boundary.v94_wide->z[0] = negative;
    if (coordinate == 2) boundary.v94_wide->tau = negative;
    if (coordinate == 3) boundary.v94_wide->kappa = negative;
    require(solver::homogeneous_step(cone, unit, 1, 1, boundary, options, solver::Clock::now()) == std::nextafter(1.0, 0.0),
        "s/z/tau/kappa边界都须LD除法后末转double，不可先窄方向");
  }
  ++mapping_cases;
  details.push_back({"wide_direction_consumers", "mapping"});
}

template<class Operation>
void every_wide_boundary(Operation operation) {
  int checks = 0;
  operation([&] { ++checks; });
  require(checks >= 2, "预算边界必须来自实际完成的LD桥或消费者");
  for (int boundary = 1; boundary <= checks; ++boundary) {
    int observed = 0; bool original = false, committed = false;
    try { operation([&] { if (++observed == boundary) throw solver::BudgetExpired{}; }); committed = true; }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && observed == boundary && !committed, "实际LD桥或消费者任一预算边界都不能提前正常交付");
  }
}

void check_wide_scope_budget() {
  const auto model = coupled(); const auto options = policy();
  const auto expected = real_core(model, options);
  coupled_state::Log outer; outer.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  {
    coupled_state::Scope scope(&outer); energy_state::Scope energy_scope(&energy);
    same_non_timing(real_wrapper(model, options, false), expected);
    require(outer.directions_started == 0 && energy.started == 0 && coupled_state::current() == &outer,
        "off须保持原报告并隔离全部外层宽方向作用域");
  }
  original_quality(model, real_wrapper(model, options, true), 25.0 / 3.0);
  solver::ConeModel cone; cone.inequality = {true};
  solver::Point point; point.x = {0}; point.s = {2}; point.z = {3};
  const auto scaling = solver::detail::make_ipm_homogeneous_scaling(1, {true}, point.s, point.z, 1, 1);
  every_wide_boundary([&](const std::function<void()>& budget) {
    const auto direction = solver::v94_deliver_direction(scaling, {0, 0, 0, 0, 0}, budget);
    require(direction.v94_wide && direction.point.x.empty(), "预算化pack正常返回必须仍为wide-only");
  });
  const auto direction = wide_fixture(1, 1, 1, 1, 1);
  every_wide_boundary([&](const std::function<void()>& budget) {
    std::vector<double> h{0}; solver::v94_corrector_rhs(cone, point, 1, 1, direction, 0, 1, h, budget);
  });
  every_wide_boundary([&](const std::function<void()>& budget) {
    const auto before = point;
    const auto update = solver::v94_update_point(cone, point, 1, 1, direction, 0.5, budget);
    require(point.x == before.x && point.s == before.s && point.z == before.z && update.tau > 0,
        "更新helper不修改持久点，只在所有末预算完成后交付完整候选");
  });
  // 跨4096坐标核pack及提交循环内部检查，非只检查首尾。
  const std::size_t size = 4098;
  const auto large = solver::detail::make_ipm_homogeneous_scaling(size, {}, {}, {}, 1, 1);
  every_wide_boundary([&](const std::function<void()>& budget) {
    solver::v94_deliver_direction(large, std::vector<long double>(size + 2, 0), budget);
  });
  solver::ConeModel large_cone; large_cone.inequality.assign(size, true);
  solver::Point large_point; large_point.x.assign(size, 0); large_point.s.assign(size, 1); large_point.z.assign(size, 1);
  solver::HomogeneousDirection large_direction; large_direction.v94_wide.emplace();
  large_direction.v94_wide->x.assign(size, 0); large_direction.v94_wide->s.assign(size, 0); large_direction.v94_wide->z.assign(size, 0);
  every_wide_boundary([&](const std::function<void()>& budget) {
    std::vector<double> large_h(size, 0);
    solver::v94_corrector_rhs(large_cone, large_point, 1, 1, large_direction, 0, 1, large_h, budget);
  });
  every_wide_boundary([&](const std::function<void()>& budget) {
    solver::v94_update_point(large_cone, large_point, 1, 1, large_direction, 0.5, budget);
  });
  bool rejected = false;
  try { solver::v94_deliver_direction(scaling, {0}, {}); } catch (const solver::NumericalFailure&) { rejected = true; }
  require(rejected, "错维宽方向不能在桥中交付");
  ++budget_cases;
  details.push_back({"wide_direction_scope_budget", "budget"});
}

struct RestartHsdFixture {
  const std::vector<std::int64_t> starts{0, 1}, rows{0};
  const std::vector<double> values{1}, zero{0}, one{1}, rhs{1, 0, 0, 0, 0};
  const solver::detail::IpmHomogeneousScaling scaling =
      solver::detail::make_ipm_homogeneous_scaling(1, {true}, one, one, 1, 1);
  const std::array<double, 5> tolerance = solver::detail::ipm_homogeneous_scaled_tolerances(scaling, rhs);
  coupled_state::Log log;
  std::vector<std::int64_t> p_cycles;
  int column_calls = 0, true_calls = 0;
  std::vector<std::vector<long double>> true_inputs;
  RestartHsdFixture() { log.enabled = true; }
  std::vector<double> precondition(std::span<const double> input, bool refinement) {
    p_cycles.push_back(log.last->work.cycles_started);
    if (!refinement) return std::vector<double>(input.size(), 0);
    const double ds = input[3] - input[0];
    return {input[1] - ds, input[0], ds, input[4] - input[2], input[2]};
  }
  template<class Scalar>
  void physical(std::span<const Scalar> input, std::vector<long double>& image) {
    solver::v94_hsd_product(scaling, starts, rows, values, zero, zero, one, one, 1, 1, input, image, {});
    const std::vector<long double> explicit_image{input[1], input[0] + input[2], input[4],
        input[2] + input[1], input[3] + input[4]};
    require(image == explicit_image, "restart真目标必须是独立显式完整HSD五块方程");
  }
  void column(std::span<const double> input, std::vector<long double>& image) {
    ++column_calls;
    physical(input, image);
    // 仅组件人工注入列误差以稳定触发residual-gap；生产双输入算子仍同模板原公式。
    const long double first = image[0];
    image[0] *= 1 + std::ldexp(1.0L, -20);
    image[1] += std::ldexp(1.0L, -40) * tolerance[1] * (first / tolerance[0]);
  }
  void truth(std::span<const long double> input, std::vector<long double>& image) {
    ++true_calls;
    true_inputs.emplace_back(input.begin(), input.end());
    physical(input, image);
  }
  long double measure(const std::vector<long double>& input) {
    std::vector<long double> image;
    physical(std::span<const long double>(input), image);
    const coupled_kernel::Layout layout{1, 1, 1};
    long double result = 0;
    for (std::size_t i = 0; i < image.size(); ++i)
      result = std::max(result, std::abs(image[i] - rhs[i]) / tolerance[layout.block(i)]);
    return result;
  }
  std::vector<long double> solve(const std::function<void()>& budget = {}, bool fail_in_true = false) {
    coupled_state::Scope scope(&log);
    return solver::v96_solve_coupled(scaling, rhs,
        [&](std::span<const double> input, bool refinement) { return precondition(input, refinement); },
        [&](std::span<const double> input, std::vector<long double>& image) { column(input, image); },
        [&](std::span<const long double> input, std::vector<long double>& image) {
          truth(input, image);
          if (fail_in_true && true_calls == 2) throw solver::BudgetExpired{};
        }, budget, "corrector", 5);
  }
};

void check_restart_real_hsd_contract() {
  RestartHsdFixture fixture;
  const auto answer = fixture.solve();
  coupled_strict_log(fixture.log);
  require(fixture.true_inputs.size() >= 3 && fixture.measure(fixture.true_inputs[1]) > 1 &&
      fixture.measure(fixture.true_inputs[1]) < fixture.measure(fixture.true_inputs[0]),
      "人工列误差夹具须真实出现候选改善而严格真验仍未过，不能靠happy breakdown触发");
  require(fixture.p_cycles == std::vector<std::int64_t>({0, 1, 2}) &&
      fixture.column_calls == 2 && fixture.true_calls == 3 && fixture.measure(answer) <= 1 &&
      fixture.log.cycles_started == 2 && fixture.log.krylov_started == 2 &&
      fixture.log.preconditioner_started == 3 && fixture.log.products_started == 5 &&
      fixture.log.true_checks_started == 3 && fixture.true_inputs.back() == answer,
      "首列false convergence后须立即以已付真残差进入cycle2，无额外P/A或验后窄化");
  const auto delivered = solver::v94_deliver_direction(fixture.scaling, answer, {});
  require(delivered.v94_wide && delivered.point.x.empty() && delivered.v94_wide->x[0] == answer[0] &&
      delivered.v94_wide->z[0] == answer[1] && delivered.v94_wide->s[0] == answer[2],
      "重启后的严格完整方向仍须原样通过父LD桥");
  ++transaction_cases;
  details.push_back({"restart_real_hsd_contract", "transaction"});
}

void check_restart_disabled_mapping() {
  const auto model = coupled(); const auto options = policy();
  const auto expected = real_core(model, options);
  coupled_state::Log outer; outer.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  {
    coupled_state::Scope scope(&outer); energy_state::Scope energy_scope(&energy);
    same_non_timing(real_wrapper(model, options, false), expected);
    require(outer.directions_started == 0 && energy.started == 0 && coupled_state::current() == &outer &&
        energy_state::current() == &energy, "关闭重启候选须隔离并恢复外TLS，完整报告保持原core");
  }
  for (bool maximum : {false, true}) {
    const auto mapped_model = coupled(true, maximum);
    original_quality(mapped_model, real_wrapper(mapped_model, options, true), maximum ? 104.0 / 3.0 : 88.0 / 3.0);
  }
  require(coupled_state::context == nullptr && energy_state::context == nullptr,
      "实际min/max固定列恢复之后不能遗留重启或能量上下文");
  ++mapping_cases;
  details.push_back({"restart_disabled_mapping", "mapping"});
}

void check_restart_budget_prefix() {
  for (bool fail_in_true : {true, false}) {
    RestartHsdFixture fixture;
    bool injected = false, original = false;
    std::vector<long double> output{123, 456};
    const auto budget = [&] {
      if (!fail_in_true && fixture.log.last && fixture.log.last->work.cycles_started == 2) {
        injected = true;
        throw solver::BudgetExpired{};
      }
    };
    try { output = fixture.solve(budget, fail_in_true); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && output == std::vector<long double>({123, 456}) && fixture.log.last &&
        std::string_view(fixture.log.last->work.status) == "budget" && fixture.log.directions_started == 1 &&
        fixture.log.directions_completed == 0, "重启中断须保原BudgetExpired和调用者输出，不能把线性快照当交付");
    const auto& work = fixture.log.last->work;
    require(fixture.p_cycles == std::vector<std::int64_t>({0, 1}) && work.preconditioner_started == 2 &&
        work.preconditioner_completed == 2 && work.krylov_started == 1 && work.krylov_completed == 1 &&
        work.products_started == 3 && work.true_checks_started == 2 && work.final_measure && *work.final_measure > 1,
        "cycle2前缀只保初P和第一列P以及原始/列/候选三次A，不加隐藏真验或solve");
    if (fail_in_true) {
      require(work.cycles_started == 1 && work.products_completed == 2 && work.true_checks_completed == 1 &&
          work.final_measure == work.initial_measure,
          "未完成的候选真验不能更新best或提前触发restart");
    } else {
      require(injected && work.cycles_started == 2 && work.products_completed == 3 && work.true_checks_completed == 2 &&
          work.initial_measure && *work.final_measure < *work.initial_measure,
          "完成真验改善后直接复用其残差，cycle2预算失败保已付best且不新增A/P");
    }
  }
  ++budget_cases;
  details.push_back({"restart_budget_prefix", "budget"});
}

struct InitialForcingHsdFixture {
  const std::vector<std::int64_t> starts{0, 2}, rows{0, 1};
  const std::vector<double> values{1, 1}, costs{-1}, bounds{2, 3}, slack{4, 0}, dual{1, -3};
  const std::vector<double> rhs{3, -2, 3, 1.5, -2, -2};
  const solver::detail::IpmHomogeneousScaling scaling =
      solver::detail::make_ipm_homogeneous_scaling(1, {true, false}, slack, dual, 2, 8);
  coupled_state::Log log;
  int initial_calls = 0, refinement_calls = 0, column_calls = 0, true_calls = 0;
  std::vector<const char*> events;
  explicit InitialForcingHsdFixture() { log.enabled = true; }
  static std::vector<double> inverse(std::span<const double> q) {
    const long double dt = (12.0L * q[0] + q[1] - 5.0L * q[2] - 8.0L * q[3] - 2.0L * q[4] + 8.0L * q[5]) / 17;
    const long double dx = q[2] + 3 * dt;
    const long double ds = q[1] - static_cast<long double>(q[2]) - dt;
    const long double dz0 = (2.0L * q[4] - ds) / 4;
    const long double dz1 = q[0] + dt - dz0;
    return {static_cast<double>(dx), static_cast<double>(dz0), static_cast<double>(dz1),
        static_cast<double>(ds), static_cast<double>(dt), static_cast<double>(2.0L * q[5] - 4 * dt)};
  }
  template<class Scalar>
  static std::vector<long double> explicit_product(std::span<const Scalar> v) {
    return {static_cast<long double>(v[1]) + v[2] - v[4],
        static_cast<long double>(v[0]) + v[3] - 2 * v[4], static_cast<long double>(v[0]) - 3 * v[4],
        (-static_cast<long double>(v[0]) + 2 * v[1] + 3 * v[2] + v[5]) / 2,
        (static_cast<long double>(v[3]) + 4 * v[1]) / 2, 2 * static_cast<long double>(v[4]) + 0.5L * v[5]};
  }
  template<class Scalar>
  void product(std::span<const Scalar> input, std::vector<long double>& output) {
    if constexpr (std::is_same_v<Scalar, double>) { ++column_calls; events.push_back("AD"); }
    else { ++true_calls; events.push_back("AW"); }
    solver::v94_hsd_product(scaling, starts, rows, values, costs, bounds, slack, dual, 2, 8, input, output, {});
    const auto explicit_image = explicit_product(input);
    for (std::size_t i = 0; i < output.size(); ++i)
      energy_near(output[i], explicit_image[i], "forcing仍必须用非单位d/lambda的原物理完整J真验");
  }
  std::vector<long double> solve(bool* forced, bool break_complementarity = false,
      const std::function<void()>& budget = {}) {
    coupled_state::Scope scope(&log);
    const auto p = [&](std::span<const double> input, bool refinement) {
      auto value = inverse(input);
      if (refinement) { ++refinement_calls; events.push_back("P1"); }
      else {
        ++initial_calls; events.push_back("P0");
        value[0] += 0.1;
        if (break_complementarity) value[3] += 0.01;
      }
      return value;
    };
    const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) { product(input, output); };
    return solver::v96_solve_coupled(scaling, rhs, p, a, a, budget, "affine", 7, forced);
  }
};

void check_initial_forcing_real_hsd() {
  InitialForcingHsdFixture fixture;
  require(fixture.scaling.d == 2 && fixture.scaling.lambda[0] == 2 && fixture.scaling.lambda_tau == 4,
      "新forcing原RHS夹具不能退化为单位齐次缩放");
  const auto reduced = solver::detail::ipm_homogeneous_scaled_reduced_rhs(fixture.scaling, fixture.rhs);
  require(reduced.size() == 3 && reduced[1] != fixture.rhs[1], "必须实际区分完整原RHS与已经消元的RHS");
  const auto data = solver::v96_tolerances_and_rhs_norms(fixture.scaling, fixture.rhs);
  require(data.rhs_norms == std::array<double, 3>{3, 3, 1.5} && data.tolerances ==
      solver::detail::ipm_homogeneous_scaled_tolerances(fixture.scaling, fixture.rhs),
      "同扫B3须来自完整未消元三块，strict门槛仍逐字原公式");
  auto extreme_rhs = fixture.rhs;
  extreme_rhs = {0, 1e-300, -1e-300, -1e300, 0, 0};
  const auto extreme = solver::v96_tolerances_and_rhs_norms(fixture.scaling, extreme_rhs);
  require(extreme.rhs_norms == std::array<double, 3>{0, 1e-300, 1e300} && extreme.tolerances[1] == 1e-12,
      "零/极小B不能由已舍入strict门槛反推，大范数仍须有限保留");
  bool forced = false;
  const auto answer = fixture.solve(&forced);
  const auto& work = fixture.log.last->work;
  coupled_explicit_return_log(fixture.log);
  require(forced && std::string_view(work.status) == "forced_initial" && work.initial_forcing &&
      work.initial_forcing->rhs_norms == data.rhs_norms && work.initial_forcing->defect_norms &&
      work.initial_forcing->measure && *work.initial_forcing->measure <= 1 && work.final_measure &&
      *work.final_measure > 1 && fixture.initial_calls == 1 && fixture.refinement_calls == 0 &&
      fixture.column_calls == 0 && fixture.true_calls == 1 && fixture.log.initial_forcing_checks == 1 &&
      fixture.log.initial_forcing_returns == 1 && fixture.log.preconditioner_started == 1 &&
      fixture.log.products_started == 1 && fixture.log.true_checks_started == 1 && fixture.log.krylov_started == 0,
      "真实初P仅在三线性块受控而两互补块strict时一次P/A交付，必须明记非strict");
  const auto image = InitialForcingHsdFixture::explicit_product(std::span<const long double>(answer));
  std::array<long double, 5> errors{};
  const coupled_kernel::Layout layout{1, 2, 1};
  for (std::size_t i = 0; i < image.size(); ++i)
    errors[layout.block(i)] = std::max(errors[layout.block(i)], std::abs(image[i] - fixture.rhs[i]));
  for (std::size_t block = 0; block < 5; ++block)
    energy_near(errors[block], (*work.initial_forcing->defect_norms)[block], "独立块缺陷须与同次assess缓存一致，容许LD累加次序舍入");
  require(errors[1] > data.tolerances[1] &&
      errors[1] < 0.1L * data.rhs_norms[1] && errors[3] <= data.tolerances[3] && errors[4] <= data.tolerances[4],
      "E5必须是已完成同一真实乘积的完整块缺陷，不可用后来FG快照或比值反推");
  auto result = candidate::make_shell(solver::Model{});
  require(*result.ipm_homogeneous_scaled_inexact_returns == 0, "kernel初forcing返回还不代表公共方向交付");
  const auto delivered = solver::v96_deliver_direction(fixture.scaling, answer, forced, result);
  require(delivered.v94_wide && delivered.point.x.empty() && delivered.v94_wide->x[0] == answer[0] &&
      delivered.v94_wide->z[1] == answer[2] && delivered.v94_wide->s[0] == answer[3] &&
      delivered.v94_wide->s[1] == 0 && delivered.v94_wide->tau == answer[4] && delivered.v94_wide->kappa == answer[5] &&
      *result.ipm_homogeneous_scaled_inexact_returns == fixture.log.initial_forcing_returns,
      "原LD pack正常末预算后才记公共inexact，完整等式行和两标量不能提前窄化");
  ++transaction_cases;
  details.push_back({"initial_forcing_real_hsd", "transaction"});
}

void check_initial_forcing_strict_fallback() {
  InitialForcingHsdFixture fixture;
  bool forced = true;
  const auto answer = fixture.solve(&forced, true);
  coupled_strict_log(fixture.log);
  const auto& work = fixture.log.last->work;
  require(!forced && work.initial_forcing && std::string_view(work.initial_forcing->decision) == "rejected" &&
      work.initial_forcing->measure && *work.initial_forcing->measure > 1 && work.initial_forcing->defect_norms &&
      (*work.initial_forcing->defect_norms)[3] > work.tolerances[3] &&
      fixture.log.initial_forcing_checks == 1 && fixture.log.initial_forcing_returns == 0 &&
      fixture.initial_calls == 1 && fixture.refinement_calls > 0 && fixture.events.size() >= 3 &&
      std::string_view(fixture.events[0]) == "P0" && std::string_view(fixture.events[1]) == "AW" &&
      std::string_view(fixture.events[2]) == "P1",
      "互补缺陷不得forcing批准，拒绝后复用已付初P/A/residual直接进FG，不能重复初值调用");
  const auto explicit_product = [](const auto& value) {
    return InitialForcingHsdFixture::explicit_product(std::span<const long double>(value));
  };
  coupled_independent_equations(fixture.scaling, fixture.rhs, answer, explicit_product);
  require(fixture.log.preconditioner_started == fixture.initial_calls + fixture.refinement_calls &&
      fixture.log.products_started == fixture.column_calls + fixture.true_calls,
      "严格回退仍必须如实累计同一预条件器及双输入真实乘积");
  auto result = candidate::make_shell(solver::Model{});
  const auto direction = solver::v96_deliver_direction(fixture.scaling, answer, forced, result);
  require(direction.v94_wide && *result.ipm_homogeneous_scaled_inexact_returns == 0,
      "严格FG回退完成原LD映射，不得因曾尝试forcing误记公共inexact");
  ++mapping_cases;
  details.push_back({"initial_forcing_strict_fallback", "mapping"});
}

void check_initial_forcing_delivery_budget() {
  InitialForcingHsdFixture complete;
  bool complete_forced = false;
  const auto complete_answer = complete.solve(&complete_forced);
  auto complete_result = candidate::make_shell(solver::Model{});
  int checks = 0;
  solver::v96_deliver_direction(complete.scaling, complete_answer, complete_forced, complete_result, [&] { ++checks; });
  require(checks >= 2 && *complete_result.ipm_homogeneous_scaled_inexact_returns == 1,
      "交付预算边界必须来自真实正常完成的父pack路径");
  for (int boundary = 1; boundary <= checks; ++boundary) {
    InitialForcingHsdFixture fixture;
    bool forced = false;
    const auto answer = fixture.solve(&forced);
    auto result = candidate::make_shell(solver::Model{});
    solver::HomogeneousDirection output; output.point.x = {123}; output.scalars.tau = 456;
    int observed = 0; bool original = false;
    try { output = solver::v96_deliver_direction(fixture.scaling, answer, forced, result,
        [&] { if (++observed == boundary) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && observed == boundary && !output.v94_wide && output.point.x == std::vector<double>{123} &&
        output.scalars.tau == 456 && *result.ipm_homogeneous_scaled_inexact_returns == 0 &&
        fixture.log.initial_forcing_returns == 1 && fixture.log.directions_completed == 1 && fixture.log.last &&
        std::string_view(fixture.log.last->work.status) == "forced_initial" &&
        std::string_view(fixture.log.last->work.initial_forcing->decision) == "returned" &&
        fixture.log.preconditioner_started == fixture.log.preconditioner_completed &&
        fixture.log.products_started == fixture.log.products_completed,
        "kernel已返回而pack任一预算失败须保private1/public0和旧输出，不能回改已完成forcing快照");
  }
  InitialForcingHsdFixture failure;
  bool forced = false;
  const auto answer = failure.solve(&forced);
  auto result = candidate::make_shell(solver::Model{});
  bool ordinary = false;
  try { solver::v96_deliver_direction(failure.scaling, answer, forced, result,
      [] { throw std::runtime_error("交付组件异常"); }); }
  catch (const std::runtime_error& error) { ordinary = std::string_view(error.what()) == "交付组件异常"; }
  require(ordinary && failure.log.initial_forcing_returns == 1 && *result.ipm_homogeneous_scaled_inexact_returns == 0,
      "普通交付异常也必须保真实private/public末差1，不能称公共交付成功");
  InitialForcingHsdFixture prepare;
  bool stale_flag = true, budget_original = false;
  try { prepare.solve(&stale_flag, false, [] { throw solver::BudgetExpired{}; }); }
  catch (const solver::BudgetExpired&) { budget_original = true; }
  require(budget_original && !stale_flag && !prepare.log.last && prepare.log.directions_started == 0 &&
      prepare.log.initial_forcing_checks == 0 && prepare.log.initial_forcing_returns == 0 && prepare.initial_calls == 0,
      "原RHS扫描预算失败必须先清旧flag，不能冒记方向、初P或forcing尝试");
  ++budget_cases;
  details.push_back({"initial_forcing_delivery_budget", "budget"});
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
    check_old_incumbent_hook(); check_reject_then_accept_hook(); check_oracle_tls();
    check_owner_without_tail(); check_native_replay(); check_oracle_failure(false); check_oracle_failure(true);
    check_real_core_oracle_failure(false); check_real_core_oracle_failure(true);
    check_coupled_mixed_equations(); check_coupled_imperfect_constant(); check_coupled_empty_blocks();
    check_coupled_real_mapping(); check_coupled_scope_isolation(); check_coupled_budget_partial();
    check_energy_nominal_identity(); check_energy_stored_singleton(); check_energy_scope_budget();
    check_wide_physical_dense(); check_wide_consumers(); check_wide_scope_budget();
    check_restart_real_hsd_contract(); check_restart_disabled_mapping(); check_restart_budget_prefix();
    check_initial_forcing_real_hsd(); check_initial_forcing_strict_fallback(); check_initial_forcing_delivery_budget();
    require(details.size() == static_cast<std::size_t>(wrapper_cases + mapping_cases + transaction_cases +
        budget_cases + exception_cases) && component_ipm_calls > 0 && synthetic_core_calls > 0,
        "实际小例与合成边界计数必须守恒且不能冒称零全部求解");
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"explicit_highs_lp_solve_calls\":0,"
        "\"scope\":\"仅组件自测；component_ipm_calls计harness进入真实solve_ipm或v96_core的API次数(含零预算入口)，不含内部调用拦截；synthetic_core_calls另计，不是公开LP成绩\","
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
