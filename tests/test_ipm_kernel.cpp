#include "../src/ipm.cpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
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

// v121 诊断I/O组件新增头开始
#include <atomic>
#include <fcntl.h>
#include <io.h>
#include <thread>
// v121 诊断I/O组件新增头结束

namespace {
namespace candidate = solver::detail::ipm_presolve;
namespace coupled_state = solver::detail::ipm_coupled_state;
namespace coupled_kernel = solver::detail::ipm_coupled;
namespace energy_state = solver::detail::ipm_energy_state;
namespace energy_kernel = solver::detail::ipm_energy;
namespace crossover_state = solver::detail::ipm_crossover;
namespace crossover = solver::detail::ipx_crossover;
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
// 组件调用指本 harness 进入真实 solve_ipm/solve_ipm_core 的次数，包含零预算入口；
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
  return solver::solve_ipm_core(model, options, solver::Clock::now());
}

solver::SolveResult real_coupled_core(const solver::Model& model, const solver::SolveOptions& options) {
  coupled_state::Log log;
  log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy_log;
  energy_log.enabled = true;
  energy_state::Scope energy_scope(&energy_log);
  ++component_ipm_calls;
  return solver::solve_ipm_core(model, options, solver::Clock::now());
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
        return solver::solve_ipm_core(child, child_options, child_start);
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
        return solver::solve_ipm_core(input, child_options, child_start);
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
    result = solver::solve_ipm_core(fixture.reduced, fixture.options, fixture.start);
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
      log.initial_forcing_returns + log.candidate_forcing_returns <= log.directions_completed &&
      log.candidate_forcing_returns <= log.candidate_forcing_checks && log.candidate_forcing_checks <= log.krylov_completed,
      "实际新策略返回必须保全部已付调用与显式forcing累计守恒");
  const auto& work = log.last->work;
  if (std::string_view(work.status) == "accepted") { coupled_strict_log(log); return; }
  if (std::string_view(work.status) == "forced_candidate") {
    require(work.candidate_forcing && work.candidate_forcing->measure && *work.candidate_forcing->measure <= 1 &&
        std::string_view(work.candidate_forcing->decision) == "returned" && work.candidate_forcing->strict_measure &&
        work.final_measure == work.candidate_forcing->strict_measure && *work.final_measure > 1 &&
        work.block_measures[3] <= 1 && work.block_measures[4] <= 1 && work.candidate_forcing_returns == 1 &&
        work.initial_forcing_returns == 0 && work.initial_forcing &&
        std::string_view(work.initial_forcing->decision) == "rejected" && work.krylov_completed > 0 &&
        work.cycles_started > 0 && work.preconditioner_completed >= 2 && work.products_completed >= 3 &&
        work.true_checks_completed >= 2, "候选forcing必须保刚验算并交付的候选快照，不能偷换最佳strict点");
    return;
  }
  require(std::string_view(work.status) == "forced_initial" && work.initial_forcing &&
      work.initial_forcing->measure && *work.initial_forcing->measure <= 1 &&
      std::string_view(work.initial_forcing->decision) == "returned" &&
      work.initial_forcing_checks == 1 && work.initial_forcing_returns == 1 && work.final_measure && *work.final_measure > 1 &&
      work.candidate_forcing_returns == 0 &&
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
  const auto answer = solver::v100_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 7);
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
  const auto answer = solver::v100_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 1);
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
    const auto answer = solver::v100_solve_coupled(scaling, rhs, p, a, a, {}, "corrector", 3);
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
          return solver::solve_ipm_core(reduced, child_options, child_start);
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
        *result.ipm_homogeneous_scaled_inexact_returns == log.initial_forcing_returns + log.candidate_forcing_returns &&
        *result.ipm_homogeneous_constant_repairs == 0 &&
        *result.ipm_weighted_krylov_steps == 0 && log.directions_started == 2 * *result.ipm_iterations,
        "真实预处理/原验与固定列最大化恢复保持，两类forcing按实际末交付合计公共inexact且不记repair");
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
    try { output = solver::v100_solve_coupled(system.scaling, rhs, p, a, a, budget, "affine", 9); }
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
  const auto denominator = solver::v100_energy_denominator(*system.factor, 2, v, system.result, {}, 7);
  require(denominator == (1 + primal) + rows && denominator != (1 + primal) + recomputed_rows &&
      log.started == 1 && log.completed == 1 && log.primal_terms == 2 && log.row_terms == 3 &&
      log.last && log.last->attempt_iteration == 7 && log.last->work.denominator == denominator &&
      system.result.ipm_factorizations == factors_before && system.result.ipm_homogeneous_constant_solves == solves_before,
      "真实adapter只作全向量正归约、使用已有r和storedR，不添加solve/factor");
  // 存储舍入用大向量放大；曲率双计另用普通向量，避免几十的差额淹没在1e32的尾数里。
  const std::vector<double> moderate{2, -3, 4, 5, 6};
  const auto moderate_denominator = solver::v100_energy_denominator(*system.factor, 2, moderate, system.result, {}, 8);
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
  { energy_state::Scope scope(&complete); solver::v100_energy_denominator(*system.factor, 0, constant,
        system.result, [&] { ++checks; }, 4); }
  require(complete.started == 1 && complete.completed == 1 && checks > 2, "完整adapter路径提供真实预算边界");
  bool formed = false;
  for (int boundary = 1; boundary <= checks; ++boundary) {
    energy_state::Log log; log.enabled = true;
    energy_state::Scope scope(&log);
    int observed = 0; bool original = false; long double output = 123;
    try { output = solver::v100_energy_denominator(*system.factor, 0, constant, system.result,
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
    try { solver::v100_energy_denominator(*system.factor, 0, constant, missing, {}, 4); }
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
  const auto answer = solver::v100_solve_coupled(scaling, rhs, p, a, a, {}, "affine", 1);
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
    return solver::v100_solve_coupled(scaling, rhs,
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
    return solver::v100_solve_coupled(scaling, rhs, p, a, a, budget, "affine", 7, forced);
  }
};

void check_initial_forcing_real_hsd() {
  InitialForcingHsdFixture fixture;
  require(fixture.scaling.d == 2 && fixture.scaling.lambda[0] == 2 && fixture.scaling.lambda_tau == 4,
      "新forcing原RHS夹具不能退化为单位齐次缩放");
  const auto reduced = solver::detail::ipm_homogeneous_scaled_reduced_rhs(fixture.scaling, fixture.rhs);
  require(reduced.size() == 3 && reduced[1] != fixture.rhs[1], "必须实际区分完整原RHS与已经消元的RHS");
  const auto data = solver::v100_tolerances_and_rhs_norms(fixture.scaling, fixture.rhs);
  require(data.rhs_norms == std::array<double, 3>{3, 3, 1.5} && data.tolerances ==
      solver::detail::ipm_homogeneous_scaled_tolerances(fixture.scaling, fixture.rhs),
      "同扫B3须来自完整未消元三块，strict门槛仍逐字原公式");
  auto extreme_rhs = fixture.rhs;
  extreme_rhs = {0, 1e-300, -1e-300, -1e300, 0, 0};
  const auto extreme = solver::v100_tolerances_and_rhs_norms(fixture.scaling, extreme_rhs);
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
  const auto delivered = solver::v100_deliver_direction(fixture.scaling, answer, forced, result);
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
  const auto direction = solver::v100_deliver_direction(fixture.scaling, answer, forced, result);
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
  solver::v100_deliver_direction(complete.scaling, complete_answer, complete_forced, complete_result, [&] { ++checks; });
  require(checks >= 2 && *complete_result.ipm_homogeneous_scaled_inexact_returns == 1,
      "交付预算边界必须来自真实正常完成的父pack路径");
  for (int boundary = 1; boundary <= checks; ++boundary) {
    InitialForcingHsdFixture fixture;
    bool forced = false;
    const auto answer = fixture.solve(&forced);
    auto result = candidate::make_shell(solver::Model{});
    solver::HomogeneousDirection output; output.point.x = {123}; output.scalars.tau = 456;
    int observed = 0; bool original = false;
    try { output = solver::v100_deliver_direction(fixture.scaling, answer, forced, result,
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
  try { solver::v100_deliver_direction(failure.scaling, answer, forced, result,
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

struct CandidateForcingHsdFixture : InitialForcingHsdFixture {
  std::vector<double> initial, correction;
  std::vector<long double> initial_residual, first_true, last_true;
  const long double* last_true_data = nullptr;
  CandidateForcingHsdFixture() {
    auto reference = inverse(rhs);
    initial = reference;
    initial[3] += 2e-8;
    auto desired = reference;
    desired[0] += 0.1;
    correction.resize(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) correction[i] = desired[i] - initial[i];
    const auto image = explicit_product(std::span<const double>(initial));
    initial_residual.resize(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) initial_residual[i] = rhs[i] - image[i];
  }
  std::vector<long double> solve_candidate(bool* forced, std::optional<bool> allow = std::nullopt,
      const std::function<void()>& budget = {}, bool accurate_column = false) {
    coupled_state::Scope scope(&log);
    const auto p = [&](std::span<const double> input, bool refinement) {
      if (!refinement) { ++initial_calls; events.push_back("P0"); return initial; }
      ++refinement_calls; events.push_back("P1");
      return accurate_column ? inverse(input) : correction;
    };
    const auto ad = [&](std::span<const double> input, std::vector<long double>& image) {
      if (accurate_column) product(input, image);
      else {
        ++column_calls; events.push_back("AD");
        // 仅组件人工构造列误差，生产的double列与LD真验仍共享原物理J模板。
        image = initial_residual;
      }
    };
    const auto aw = [&](std::span<const long double> input, std::vector<long double>& image) {
      product(input, image);
      if (true_calls == 1) first_true.assign(input.begin(), input.end());
      last_true.assign(input.begin(), input.end());
      last_true_data = input.data();
    };
    if (!allow) return solver::v100_solve_coupled(scaling, rhs, p, ad, aw, budget, "corrector", 9, forced);
    return solver::v100_solve_coupled(scaling, rhs, p, ad, aw, budget, "corrector", 9, forced, *allow);
  }
  std::array<long double, 5> defects(std::span<const long double> value) const {
    const auto image = explicit_product(value);
    const coupled_kernel::Layout layout{1, 2, 1};
    std::array<long double, 5> errors{};
    for (std::size_t i = 0; i < image.size(); ++i)
      errors[layout.block(i)] = std::max(errors[layout.block(i)], std::abs(image[i] - rhs[i]));
    return errors;
  }
};

void check_candidate_forcing_real_hsd() {
  CandidateForcingHsdFixture fixture;
  bool forced = false;
  auto answer = fixture.solve_candidate(&forced, true);
  coupled_explicit_return_log(fixture.log);
  const auto& work = fixture.log.last->work;
  require(forced && std::string_view(work.status) == "forced_candidate" && work.candidate_forcing &&
      work.candidate_forcing->strict_measure && work.candidate_forcing->best_measure_before &&
      work.candidate_forcing->measure && *work.candidate_forcing->measure <= 1 &&
      *work.candidate_forcing->strict_measure > 100 * *work.candidate_forcing->best_measure_before &&
      work.final_measure == work.candidate_forcing->strict_measure && work.initial_measure &&
      *work.final_measure > *work.initial_measure &&
      work.candidate_forcing->best_measure_before == work.initial_measure,
      "候选qF受控即应交付实际候选，即使其strict比best更差；final不能偷换成旧best快照");
  require(answer.data() == fixture.last_true_data && answer == fixture.last_true && answer != fixture.first_true &&
      fixture.initial_calls == 1 && fixture.refinement_calls == 1 && fixture.column_calls == 1 && fixture.true_calls == 2 &&
      fixture.log.preconditioner_started == 2 && fixture.log.products_started == 3 && fixture.log.true_checks_started == 2 &&
      fixture.log.krylov_completed == 1 && fixture.log.cycles_started == 1 &&
      fixture.log.initial_forcing_checks == 1 && fixture.log.initial_forcing_returns == 0 &&
      fixture.log.candidate_forcing_checks == 1 && fixture.log.candidate_forcing_returns == 1,
      "已有候选真验的同一缓冲必须move交付，不重算A/P或复制别的best向量");
  require(work.initial_forcing && work.initial_forcing->defect_norms && work.initial_forcing->measure &&
      *work.initial_forcing->measure > 1 && std::string_view(work.initial_forcing->decision) == "rejected" &&
      std::string_view(work.candidate_forcing->decision) == "returned" && work.candidate_forcing->cycle == 1 &&
      work.candidate_forcing->krylov_step == 1 && work.candidate_forcing->defect_norms,
      "候选证明必须与初始rejected证明分别保留，不能覆盖初P缺陷");
  const auto errors = fixture.defects(answer);
  const auto data = solver::v100_tolerances_and_rhs_norms(fixture.scaling, fixture.rhs);
  long double forcing_measure = 0, strict_measure = 0;
  for (std::size_t block = 0; block < 5; ++block) {
    const long double limit = block < 3 ? std::max(static_cast<long double>(data.tolerances[block]),
        0.1L * data.rhs_norms[block]) : data.tolerances[block];
    forcing_measure = std::max(forcing_measure, errors[block] / limit);
    strict_measure = std::max(strict_measure, errors[block] / data.tolerances[block]);
    energy_near(errors[block], (*work.candidate_forcing->defect_norms)[block], "候选E5必须独立代回同一非单位HSD J");
  }
  energy_near(strict_measure, *work.final_measure, "最终strict量必须属于真实交付候选");
  energy_near(forcing_measure, *work.candidate_forcing->measure, "候选qF只使用原B3与同三线性10%/两互补strict门槛");
  require(forcing_measure <= 1 && errors[3] <= data.tolerances[3] && errors[4] <= data.tolerances[4] &&
      (*work.initial_forcing->defect_norms)[3] > data.tolerances[3],
      "候选必须修好初始互补缺陷，而非仅把初P改名返回");
  auto result = candidate::make_shell(solver::Model{});
  const auto delivered = solver::v100_deliver_direction(fixture.scaling, std::move(answer), forced, result);
  require(delivered.v94_wide && delivered.point.x.empty() && delivered.v94_wide->x[0] == fixture.last_true[0] &&
      delivered.v94_wide->z[1] == fixture.last_true[2] && delivered.v94_wide->s[0] == fixture.last_true[3] &&
      delivered.v94_wide->tau == fixture.last_true[4] && delivered.v94_wide->kappa == fixture.last_true[5] &&
      *result.ipm_homogeneous_scaled_inexact_returns == fixture.log.initial_forcing_returns + fixture.log.candidate_forcing_returns,
      "原LD pack仍须交付该候选五块变量，公共inexact只在末预算正常返回后计两类之和");
  ++transaction_cases;
  details.push_back({"candidate_forcing_real_hsd", "transaction"});
}

void check_candidate_forcing_default_mapping() {
  CandidateForcingHsdFixture compatibility;
  bool forced = true, failed = false;
  std::vector<long double> output{123};
  const auto stop_after_candidate = [&] {
    if (compatibility.log.last && compatibility.log.last->work.true_checks_completed >= 2) throw solver::BudgetExpired{};
  };
  try { output = compatibility.solve_candidate(&forced, std::nullopt, stop_after_candidate); }
  catch (const solver::BudgetExpired&) { failed = true; }
  require(failed && !forced && output == std::vector<long double>{123} && compatibility.log.last &&
      !compatibility.log.last->work.candidate_forcing && compatibility.log.candidate_forcing_checks == 0 &&
      compatibility.log.candidate_forcing_returns == 0 && compatibility.log.initial_forcing_returns == 0 &&
      compatibility.log.krylov_completed == 1 && compatibility.log.true_checks_completed == 2 &&
      compatibility.log.cycles_started <= 3,
      "省略新参数必须保父initial-only合同，已验候选不能暗返，继续原FG前预算中断也不能交付");
  CandidateForcingHsdFixture strict;
  forced = true;
  const auto answer = strict.solve_candidate(&forced, true, {}, true);
  coupled_strict_log(strict.log);
  require(!forced && strict.log.last->work.candidate_forcing && strict.log.candidate_forcing_checks == 0 &&
      strict.log.candidate_forcing_returns == 0 && strict.log.initial_forcing_returns == 0 && strict.refinement_calls > 0,
      "显式开启候选gate仍须严格优先，strict真验成功不能记一次非strict候选批准");
  const auto explicit_product = [](const auto& value) {
    return InitialForcingHsdFixture::explicit_product(std::span<const long double>(value));
  };
  coupled_independent_equations(strict.scaling, strict.rhs, answer, explicit_product);
  auto result = candidate::make_shell(solver::Model{});
  const auto delivered = solver::v100_deliver_direction(strict.scaling, answer, forced, result);
  require(delivered.v94_wide && delivered.v94_wide->s[1] == 0 && *result.ipm_homogeneous_scaled_inexact_returns == 0,
      "严格返回仍按原LD位置恢复等式行，不能因开启gate误记公共inexact");
  ++mapping_cases;
  details.push_back({"candidate_forcing_default_mapping", "mapping"});
}

void check_candidate_forcing_delivery_budget() {
  CandidateForcingHsdFixture eligible;
  bool forced = true, interrupted = false;
  std::vector<long double> output{123, 456};
  const auto budget = [&] {
    if (eligible.log.last && eligible.log.last->work.candidate_forcing &&
        std::string_view(eligible.log.last->work.candidate_forcing->decision) == "eligible") throw solver::BudgetExpired{};
  };
  try { output = eligible.solve_candidate(&forced, true, budget); }
  catch (const solver::BudgetExpired&) { interrupted = true; }
  require(interrupted && !forced && output == std::vector<long double>({123, 456}) && eligible.log.last &&
      std::string_view(eligible.log.last->work.status) == "budget" && eligible.log.directions_completed == 0 &&
      eligible.log.candidate_forcing_checks == 1 && eligible.log.candidate_forcing_returns == 0 &&
      eligible.log.initial_forcing_returns == 0, "eligible末预算失败不能伪造候选返回或设置输出forced标记");
  const auto& partial = eligible.log.last->work;
  require(partial.candidate_forcing && partial.candidate_forcing->strict_measure && partial.final_measure &&
      partial.final_measure == partial.initial_measure && partial.candidate_forcing->best_measure_before == partial.initial_measure &&
      *partial.candidate_forcing->strict_measure > *partial.final_measure &&
      std::string_view(partial.candidate_forcing->decision) == "eligible" && partial.true_checks_completed == 2 &&
      partial.preconditioner_completed == 2 && partial.products_completed == 3,
      "未交付的非best eligible候选只留独立证明，最终摘要仍是已付最佳strict快照");
  CandidateForcingHsdFixture complete;
  forced = false;
  const auto complete_answer = complete.solve_candidate(&forced, true);
  auto complete_result = candidate::make_shell(solver::Model{});
  int checks = 0;
  solver::v100_deliver_direction(complete.scaling, complete_answer, forced, complete_result, [&] { ++checks; });
  require(checks >= 2 && *complete_result.ipm_homogeneous_scaled_inexact_returns == 1,
      "候选返回的实际交付末预算成功后才承认公共inexact");
  for (int boundary = 1; boundary <= checks; ++boundary) {
    CandidateForcingHsdFixture fixture;
    bool selected = false;
    const auto answer = fixture.solve_candidate(&selected, true);
    auto result = candidate::make_shell(solver::Model{});
    solver::HomogeneousDirection delivered; delivered.point.x = {123}; delivered.scalars.tau = 456;
    int observed = 0; bool original = false;
    try { delivered = solver::v100_deliver_direction(fixture.scaling, answer, selected, result,
        [&] { if (++observed == boundary) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && observed == boundary && !delivered.v94_wide && delivered.point.x == std::vector<double>{123} &&
        delivered.scalars.tau == 456 && *result.ipm_homogeneous_scaled_inexact_returns == 0 &&
        fixture.log.initial_forcing_returns == 0 && fixture.log.candidate_forcing_returns == 1 &&
        fixture.log.directions_completed == 1 && fixture.log.last &&
        std::string_view(fixture.log.last->work.status) == "forced_candidate" &&
        std::string_view(fixture.log.last->work.candidate_forcing->decision) == "returned" &&
        fixture.log.preconditioner_started == fixture.log.preconditioner_completed &&
        fixture.log.products_started == fixture.log.products_completed,
        "kernel候选已返而pack任一预算失败须保privatecandidate1/public0、真实末候选及调用者旧输出");
  }
  CandidateForcingHsdFixture ordinary;
  bool selected = false;
  const auto answer = ordinary.solve_candidate(&selected, true);
  auto result = candidate::make_shell(solver::Model{});
  bool original = false;
  try { solver::v100_deliver_direction(ordinary.scaling, answer, selected, result,
      [] { throw std::runtime_error("候选交付组件异常"); }); }
  catch (const std::runtime_error& error) { original = std::string_view(error.what()) == "候选交付组件异常"; }
  require(original && ordinary.log.candidate_forcing_returns == 1 && *result.ipm_homogeneous_scaled_inexact_returns == 0,
      "普通候选交付异常同样保已付private1/public0，不能把核交付当Point提交");
  ++budget_cases;
  details.push_back({"candidate_forcing_delivery_budget", "budget"});
}

struct ForcingWeightHsdFixture : InitialForcingHsdFixture {
  const std::vector<double> weighted_rhs{10, 20, 40, 30, 0, 0};
  std::vector<std::vector<double>> refinement_inputs;
  std::vector<double> first_direction, clean_direction;
  std::vector<long double> first_column_image, last_true;
  const long double* last_true_data = nullptr;

  std::array<long double, 5> limits() const {
    const auto data = solver::v100_tolerances_and_rhs_norms(scaling, weighted_rhs);
    std::array<long double, 5> result{};
    for (std::size_t block = 0; block < 5; ++block) {
      result[block] = data.tolerances[block];
      if (block < 3) result[block] = std::max(result[block], 0.1L * data.rhs_norms[block]);
    }
    return result;
  }

  std::vector<long double> solve_weighted(bool* forced, std::optional<bool> weights = std::nullopt,
      const std::function<void()>& budget = {}, bool artificial_column_error = false) {
    coupled_state::Scope scope(&log);
    const auto p = [&](std::span<const double> input, bool refinement) {
      if (!refinement) {
        ++initial_calls; events.push_back("P0");
        return std::vector<double>(input.size(), 0);
      }
      ++refinement_calls; events.push_back("P1");
      refinement_inputs.emplace_back(input.begin(), input.end());
      std::vector<double> modified(input.begin(), input.end());
      modified[1] *= 1.05; modified[2] *= 1.1;
      auto value = inverse(modified);
      clean_direction = value;
      if (artificial_column_error) value[3] += 0.01;
      if (refinement_calls == 1) first_direction = value;
      return value;
    };
    const auto ad = [&](std::span<const double> input, std::vector<long double>& image) {
      product(input, image);
      if (artificial_column_error) {
        // 只为反例人工隐藏列中的互补误差；LD真验始终使用未改动的真实物理J。
        image = explicit_product(std::span<const double>(clean_direction));
      }
      if (column_calls == 1) first_column_image = image;
    };
    const auto aw = [&](std::span<const long double> input, std::vector<long double>& image) {
      product(input, image);
      last_true.assign(input.begin(), input.end());
      last_true_data = input.data();
    };
    if (!weights) return solver::v100_solve_coupled(scaling, weighted_rhs, p, ad, aw, budget,
        "corrector", 11, forced, true);
    return solver::v100_solve_coupled(scaling, weighted_rhs, p, ad, aw, budget,
        "corrector", 11, forced, true, *weights);
  }

  std::pair<long double, long double> independent_one_column() const {
    const auto scale = limits();
    const coupled_kernel::Layout layout{1, 2, 1};
    require(first_column_image.size() == weighted_rhs.size(), "独立单列关系必须读取已经付出的实际物理列");
    long double beta_squared = 0, rhs_image_dot = 0, image_squared = 0;
    for (std::size_t i = 0; i < weighted_rhs.size(); ++i) {
      const long double r = weighted_rhs[i] / scale[layout.block(i)];
      const long double image = first_column_image[i] / scale[layout.block(i)];
      beta_squared += r * r; rhs_image_dot += r * image; image_squared += image * image;
    }
    // 一列最小二乘直接解标量正规方程，不读取内核的basis/H/Givens或三角系数。
    require(beta_squared > 1 && image_squared > 0, "独立投影必须是非零、尚未通过的初始系统");
    const long double coefficient = rhs_image_dot / image_squared;
    long double residual_squared = 0;
    for (std::size_t i = 0; i < weighted_rhs.size(); ++i) {
      const long double residual = (weighted_rhs[i] - coefficient * first_column_image[i]) / scale[layout.block(i)];
      residual_squared += residual * residual;
    }
    return {coefficient, std::sqrt(residual_squared)};
  }
};

void check_forcing_weights_real_hsd() {
  ForcingWeightHsdFixture fixture;
  const auto data = solver::v100_tolerances_and_rhs_norms(fixture.scaling, fixture.weighted_rhs);
  const auto scale = fixture.limits();
  require(fixture.scaling.d == 2 && fixture.scaling.lambda[0] == 2 && fixture.scaling.lambda_tau == 4 &&
      data.rhs_norms == std::array<double, 3>{10, 40, 30} &&
      scale == std::array<long double, 5>{1, 4, 3, 1e-12, 1e-12},
      "搜索尺度夹具须来自非单位真实HSD和不同完整线性块范数，两个互补块仍用原double门槛提升值");
  bool forced = false;
  auto answer = fixture.solve_weighted(&forced, true);
  coupled_explicit_return_log(fixture.log);
  const auto& work = fixture.log.last->work;
  require(forced && std::string_view(work.status) == "forced_candidate" &&
      work.tolerances == data.tolerances && work.initial_forcing && work.initial_forcing->rhs_norms == data.rhs_norms &&
      std::string_view(work.initial_forcing->decision) == "rejected" && work.candidate_forcing &&
      work.candidate_forcing->measure && *work.candidate_forcing->measure <= 1 &&
      fixture.initial_calls == 1 && fixture.refinement_calls == 1 && fixture.column_calls == 1 && fixture.true_calls == 2 &&
      fixture.log.preconditioner_completed == 2 && fixture.log.products_completed == 3 &&
      fixture.log.krylov_completed == 1 && fixture.log.cycles_started == 1,
      "F尺度只改变搜索，一列既有候选仍须完整真实strict失败及同F批准，不能新增P/A或改tau摘要");
  const long double beta = std::sqrt(325.0L);
  require(fixture.refinement_inputs.size() == 1, "须观察实际唯一的首列物理P右端");
  for (std::size_t i = 0; i < fixture.weighted_rhs.size(); ++i)
    energy_near(fixture.refinement_inputs[0][i], fixture.weighted_rhs[i] / beta,
        "residual/T归一后必须乘回同T再送P；本夹具beta平方独立为325");
  const auto [coefficient, projected] = fixture.independent_one_column();
  require(projected > 0.1L && projected < 1 && answer.data() == fixture.last_true_data && answer == fixture.last_true,
      "实际同T单列投影应在窗口前触发真实候选，交付仍是最后真验的同一LD缓冲");
  for (std::size_t i = 0; i < answer.size(); ++i)
    energy_near(answer[i], coefficient * fixture.first_direction[i],
        "物理Z不改尺度；真实候选必须符合独立image/T单列最小二乘系数");
  const auto image = ForcingWeightHsdFixture::explicit_product(std::span<const long double>(answer));
  const coupled_kernel::Layout layout{1, 2, 1};
  std::array<long double, 5> errors{};
  for (std::size_t i = 0; i < image.size(); ++i)
    errors[layout.block(i)] = std::max(errors[layout.block(i)], std::abs(image[i] - fixture.weighted_rhs[i]));
  long double strict = 0, controlled = 0;
  require(work.candidate_forcing->defect_norms.has_value(), "完整truecheck后才有候选E5");
  for (std::size_t block = 0; block < 5; ++block) {
    strict = std::max(strict, errors[block] / data.tolerances[block]);
    controlled = std::max(controlled, errors[block] / scale[block]);
    energy_near(errors[block], (*work.candidate_forcing->defect_norms)[block], "同F搜索不能替代真实物理J的五块缺陷缓存");
  }
  energy_near(strict, *work.final_measure, "最终strict残差仍按原tau，而非搜索T计算");
  energy_near(controlled, *work.candidate_forcing->measure, "真实gate仍使用原B3及原两互补strict门槛");
  require(strict > 1 && controlled <= 1 && controlled <= projected + 1e-9L &&
      errors[3] <= data.tolerances[3] && errors[4] <= data.tolerances[4],
      "独立投影2范数与真实qF须分清，批准始终来自完整真实残差");
  auto result = candidate::make_shell(solver::Model{});
  const auto direction = solver::v100_deliver_direction(fixture.scaling, std::move(answer), forced, result);
  require(direction.v94_wide && direction.v94_wide->x[0] == fixture.last_true[0] &&
      direction.v94_wide->z[1] == fixture.last_true[2] && direction.v94_wide->s[0] == fixture.last_true[3] &&
      direction.v94_wide->s[1] == 0 && direction.v94_wide->tau == fixture.last_true[4] &&
      direction.v94_wide->kappa == fixture.last_true[5] && *result.ipm_homogeneous_scaled_inexact_returns == 1,
      "F尺度不改变原完整LD坐标映射，正常末pack后才记公共非strict交付");
  ++transaction_cases;
  details.push_back({"forcing_weights_real_hsd", "transaction"});
}

void check_forcing_weights_default_mapping() {
  ForcingWeightHsdFixture omitted, explicit_false;
  for (auto* fixture : {&omitted, &explicit_false}) {
    bool forced = true, stopped = false;
    std::vector<long double> output{123};
    const auto budget = [&] {
      if (fixture->log.last && fixture->log.last->work.krylov_completed >= 1) throw solver::BudgetExpired{};
    };
    try { output = fixture->solve_weighted(&forced, fixture == &omitted ? std::optional<bool>{} : std::optional<bool>{false}, budget); }
    catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && !forced && output == std::vector<long double>{123} && fixture->log.last &&
        std::string_view(fixture->log.last->work.status) == "budget" && fixture->log.directions_completed == 0 &&
        fixture->log.last->work.candidate_forcing &&
        std::string_view(fixture->log.last->work.candidate_forcing->decision) == "not_checked" &&
        fixture->log.krylov_completed == 1 && fixture->log.true_checks_completed == 1 &&
        fixture->log.candidate_forcing_checks == 0 && fixture->log.candidate_forcing_returns == 0 &&
        fixture->initial_calls == 1 && fixture->refinement_calls == 1 && fixture->column_calls == 1,
        "省略forcing_weights或显式false必须保父strict尺度；首列投影尚未触发truecheck，预算失败不能交付");
  }
  require(omitted.refinement_inputs == explicit_false.refinement_inputs &&
      omitted.first_direction == explicit_false.first_direction && omitted.first_column_image == explicit_false.first_column_image &&
      omitted.refinement_inputs.size() == 1 && std::abs(omitted.refinement_inputs[0][0]) < 1e-6 &&
      10 / std::sqrt(325.0L) > 0.1L,
      "默认参数必须实际重现父搜索输入和物理列，同时与F尺度输入明显区分，不能只核开关声明");
  coupled_state::Log outer; outer.enabled = true;
  {
    coupled_state::Scope scope(&outer);
    ForcingWeightHsdFixture inner;
    bool forced = false;
    const auto answer = inner.solve_weighted(&forced, true);
    require(forced && answer == inner.last_true && coupled_state::current() == &outer && outer.directions_started == 0 &&
        outer.preconditioner_started == 0 && !outer.last, "真实权重调用仍须隔离TLS、复原外层Scope且不污染外层账本");
  }
  require(coupled_state::current() == nullptr, "新增私有参数不改变Scope退出复原合同");
  ++mapping_cases;
  details.push_back({"forcing_weights_default_mapping", "mapping"});
}

void check_forcing_weights_budget() {
  ForcingWeightHsdFixture complete;
  bool forced = false;
  int checks = 0;
  const auto answer = complete.solve_weighted(&forced, true, [&] { ++checks; });
  require(forced && checks > 2, "预算前缀须取自真实同F完整求解，不预造callback次数");
  for (int boundary = 1; boundary <= checks; ++boundary) {
    ForcingWeightHsdFixture fixture;
    bool selected = true, stopped = false;
    std::vector<long double> output{123, 456};
    int observed = 0;
    try { output = fixture.solve_weighted(&selected, true,
        [&] { if (++observed == boundary) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && observed == boundary && !selected && output == std::vector<long double>({123, 456}) &&
        fixture.log.directions_completed == 0 && fixture.log.initial_forcing_returns == 0 && fixture.log.candidate_forcing_returns == 0,
        "每一真实预算边界须保原异常类型、调用者旧输出和false标记，不能把搜索估计当返回");
    if (fixture.log.last) {
      const auto& partial = fixture.log.last->work;
      require(std::string_view(partial.status) == "budget" &&
          partial.preconditioner_started - partial.preconditioner_completed >= 0 &&
          partial.preconditioner_started - partial.preconditioner_completed <= 1 &&
          partial.products_started - partial.products_completed >= 0 && partial.products_started - partial.products_completed <= 1 &&
          partial.true_checks_started - partial.true_checks_completed >= 0 && partial.true_checks_started - partial.true_checks_completed <= 1 &&
          partial.candidate_forcing &&
          (std::string_view(partial.candidate_forcing->decision) != "not_checked" || !partial.candidate_forcing->defect_norms),
          "同T三处工作中断保真实P/A/真验partial前缀，半份E5不能提前发布");
      if (std::string_view(partial.candidate_forcing->decision) == "eligible")
        require(partial.candidate_forcing_checks == 1 && partial.candidate_forcing_returns == 0 &&
            partial.candidate_forcing->defect_norms && partial.true_checks_completed == 2 && partial.final_measure && partial.initial_measure &&
            *partial.final_measure <= *partial.initial_measure,
            "eligible末预算失败只保已付最佳strict及完整候选证明，不能伪记返回");
    }
  }
  ForcingWeightHsdFixture rejected;
  bool rejected_flag = true, stopped = false;
  std::vector<long double> rejected_output{789};
  const auto reject_budget = [&] {
    if (rejected.log.last && rejected.log.last->work.candidate_forcing &&
        std::string_view(rejected.log.last->work.candidate_forcing->decision) == "rejected") throw solver::BudgetExpired{};
  };
  try { rejected_output = rejected.solve_weighted(&rejected_flag, true, reject_budget, true); }
  catch (const solver::BudgetExpired&) { stopped = true; }
  const auto [unused_coefficient, projected] = rejected.independent_one_column();
  static_cast<void>(unused_coefficient);
  require(stopped && !rejected_flag && rejected_output == std::vector<long double>{789} && projected < 1 &&
      rejected.log.last && rejected.log.last->work.candidate_forcing &&
      rejected.log.last->work.candidate_forcing->defect_norms && rejected.log.last->work.candidate_forcing->measure &&
      *rejected.log.last->work.candidate_forcing->measure > 1 &&
      (*rejected.log.last->work.candidate_forcing->defect_norms)[3] > rejected.log.last->work.tolerances[3] &&
      rejected.log.candidate_forcing_checks == 1 && rejected.log.candidate_forcing_returns == 0 &&
      rejected.log.true_checks_completed == 2 && rejected.log.krylov_completed == 1,
      "人工列误差下投影估计已过仍须用真实J拒绝互补缺陷；不能把同T估计当forcing验收");
  auto result = candidate::make_shell(solver::Model{});
  int pack_checks = 0;
  solver::v100_deliver_direction(complete.scaling, answer, forced, result, [&] { ++pack_checks; });
  require(*result.ipm_homogeneous_scaled_inexact_returns == 1 && pack_checks >= 2, "同T结果正常末pack才记公共返回");
  auto partial_result = candidate::make_shell(solver::Model{});
  int observed = 0; bool pack_stopped = false;
  solver::HomogeneousDirection delivered; delivered.point.x = {123};
  try { delivered = solver::v100_deliver_direction(complete.scaling, answer, forced, partial_result,
      [&] { if (++observed == pack_checks) throw solver::BudgetExpired{}; }); }
  catch (const solver::BudgetExpired&) { pack_stopped = true; }
  require(pack_stopped && !delivered.v94_wide && delivered.point.x == std::vector<double>{123} &&
      complete.log.candidate_forcing_returns == 1 && complete.log.directions_completed == 1 &&
      *partial_result.ipm_homogeneous_scaled_inexact_returns == 0 &&
      std::string_view(complete.log.last->work.status) == "forced_candidate",
      "同T核已返回但末pack预算失败仍保private1/public0及旧输出，不能回改真实forcing快照");
  ++budget_cases;
  details.push_back({"forcing_weights_budget", "budget"});
}

struct WarmRegularizationSystem {
  solver::ConeModel cone;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  const std::vector<double> h{2, 0, 3};
  std::unique_ptr<solver::KktFactor> factor;

  explicit WarmRegularizationSystem(bool prepare = true) {
    cone.costs = {-1, 2}; cone.rhs = {3, -2, 5};
    cone.starts = {0, 3, 5}; cone.rows = {0, 1, 2, 0, 1}; cone.values = {1, 3, 5, 2, -1};
    cone.inequality = {true, false, true}; cone.barrier_rows = 2;
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    require(factor->regularization_level() == 0 && *result.ipm_factorizations == 0,
        "新实际KKT对象只能从自身初始level0开始，不能继承跨对象历史");
    if (prepare) {
      // 明确选择旧网格最高合法层级，使本组只验证复用合同，不预设冷分解何时首次成功。
      factor->factor(h, options, start, result, 7);
      require(factor->regularization_level() == 7 && *result.ipm_factorizations == 1 &&
          result.ipm_primal_regularization && result.ipm_dual_regularization &&
          result.ipm_factor_backend == "normal_cholmod_aat" && options.ipm.equilibrate_kkt &&
          *result.ipm_eliminated_singletons == 1 && factor->v93_regularized_h().size() == 3,
          "新夹具须真实执行level7的normalAAT、均衡和单例消元，不用假getter或虚构因子");
    }
  }

  EnergyMatrix nominal() const {
    const auto stored = factor->v93_regularized_h();
    require(stored.size() == 3 && result.ipm_primal_regularization.has_value(), "完整名义矩阵须来自当前实际r/R");
    const long double r = *result.ipm_primal_regularization;
    // 显式保留原G三行；最后一行虽被分解凝聚，也必须进入独立完整系统。
    return {{{r, 0, 1, 3, 5}, {0, r, 2, -1, 0}, {1, 2, -stored[0], 0, 0},
        {3, -1, 0, -stored[1], 0}, {5, 0, 0, 0, -stored[2]}}};
  }
};

void check_warm_regularization_seed_scope() {
  WarmRegularizationSystem system;
  const auto snapshot = system.result;
  const auto before_rows = std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end());
  require(coupled_state::current() == nullptr && solver::v100_regularization_seed(*system.factor) == 0,
      "没有活动联立Scope时，即使对象已有level7也必须选择原0");
  coupled_state::Log outer; outer.enabled = true;
  {
    coupled_state::Scope scope(&outer);
    require(solver::v100_regularization_seed(*system.factor) == 7 &&
        solver::v100_regularization_seed(*system.factor) == 7 && !outer.last && outer.directions_started == 0,
        "活动起点只读同一真实因子的level，无方向日志或状态更新");
    coupled_state::Log disabled;
    {
      coupled_state::Scope isolation(&disabled);
      require(solver::v100_regularization_seed(*system.factor) == 0 && coupled_state::current() == nullptr,
          "禁用Scope必须压住外层活动状态，而非读取任意非空TLS指针");
    }
    require(coupled_state::current() == &outer && solver::v100_regularization_seed(*system.factor) == 7,
        "退出禁用Scope须完整恢复原活动层级选择");
    {
      coupled_state::Scope isolation(nullptr);
      require(solver::v100_regularization_seed(*system.factor) == 0, "显式空隔离必须恢复原冷起点");
    }
    WarmRegularizationSystem fresh(false);
    require(solver::v100_regularization_seed(*fresh.factor) == 0 && *fresh.result.ipm_factorizations == 0 &&
        !fresh.result.ipm_primal_regularization && fresh.factor->v93_regularized_h().empty(),
        "活动Scope不向新模型对象传播旧level7或未支付的r/R");
    require(coupled_state::current() == &outer && outer.constant_calls_started == 0 && outer.directions_started == 0,
        "读取起点不得新增常量、方向或预条件工作");
  }
  same_non_timing(system.result, snapshot);
  require(coupled_state::current() == nullptr && system.factor->regularization_level() == 7 &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == before_rows,
      "所有Scope/起点查询都必须无副作用并在退出时复原TLS");
  ++mapping_cases;
  details.push_back({"warm_regularization_seed_scope", "mapping"});
}

void check_warm_regularization_real_factor() {
  WarmRegularizationSystem system;
  const auto old_count = *system.result.ipm_factorizations;
  const auto old_primal = system.result.ipm_primal_regularization, old_dual = system.result.ipm_dual_regularization;
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end());
  coupled_state::Log log; log.enabled = true;
  coupled_state::Scope scope(&log);
  system.factor->factor(system.h, system.options, system.start, system.result,
      solver::v100_regularization_seed(*system.factor));
  require(*system.result.ipm_factorizations == old_count + 1 && system.factor->regularization_level() == 7 &&
      system.result.ipm_primal_regularization == old_primal && system.result.ipm_dual_regularization == old_dual &&
      *old_primal == solver::detail::ipm_regularization_at_level(7) &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows &&
      *system.result.ipm_homogeneous_constant_solves == 0 && *system.result.ipm_homogeneous_direction_solves == 0 &&
      log.constant_calls_started == 0 && log.directions_started == 0,
      "同h真实warm seed7只能支付一次新分解并重建同一实际r/R，不能复用旧因子值或偷偷求方向");
  for (std::size_t row_index = 0; row_index < system.h.size(); ++row_index)
    require(old_rows[row_index] == static_cast<double>(static_cast<long double>(system.h[row_index]) + *old_dual),
        "R必须是当前factor实际存储的double加法舍入值，含等式及被凝聚行");
  const auto matrix = system.nominal();
  const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5};
  const auto dense = independent_energy_dense(matrix, std::vector<long double>(rhs.begin(), rhs.end()));
  const auto direct = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, true);
  require(direct.size() == 5 && *system.result.ipm_homogeneous_constant_solves == 1,
      "真实warm因子必须可供direct使用并恢复完整五维，而非只通过层级查询");
  for (double value : direct) require(std::isfinite(value), "完整恢复的warm direct方向须有限");
  energy_near(static_cast<long double>(old_rows[2]) * direct[4], 5.0L * direct[0] - rhs[4],
      "真实单例行恢复必须使用该次存储R和原完整名义方程");
  for (int kind : {0, 1}) {
    const auto& value = kind == 0 ? dense : direct;
    std::array<long double, 5> residual{};
    for (std::size_t i = 0; i < 5; ++i) {
      residual[i] = -rhs[i];
      for (std::size_t j = 0; j < 5; ++j) residual[i] += matrix[i][j] * value[j];
    }
    const long double eta = value[0] * residual[0] + value[1] * residual[1] -
        value[2] * residual[2] - value[3] * residual[3] - value[4] * residual[4];
    energy_kernel::Work work;
    const auto denominator = energy_kernel::compute(2, value, system.factor->v93_regularized_h(), *old_primal, work);
    const long double schur = 1 - (-value[0] + 2.0L * value[1] + 3.0L * value[2] - 2.0L * value[3] + 5.0L * value[4]) / 2;
    energy_near(denominator - schur, eta, "warm起点不改完整Kdelta的D-S=eta，近似direct保其真实缺陷");
    if (kind == 0) energy_near(eta, 0, "独立dense名义解应消去eta，不要求近似P冒充精确解");
    require(work.primal_terms == 2 && work.row_terms == 3, "warm名义能量仍含全部列/行，不能双计凝聚曲率或遗漏等式行");
  }
  ++transaction_cases;
  details.push_back({"warm_regularization_real_factor", "transaction"});
}

void check_warm_regularization_hsd_budget() {
  const auto model = coupled();
  const auto options = policy();
  const auto preparation_start = solver::Clock::now();
  const auto working = solver::canonicalize(model, options, preparation_start);
  const auto expired_start = solver::Clock::now() - std::chrono::seconds(10);
  auto hsd_result = candidate::make_shell(model);
  const auto hsd_snapshot = hsd_result;
  coupled_state::Log log; log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy; energy.enabled = true;
  energy_state::Scope energy_scope(&energy);
  bool hsd_stopped = false;
  try { solver::run_homogeneous_ipm(working, options, expired_start, hsd_result); }
  catch (const solver::BudgetExpired&) { hsd_stopped = true; }
  same_non_timing(hsd_result, hsd_snapshot);
  require(hsd_stopped && *hsd_result.ipm_factorizations == 0 &&
      *hsd_result.ipm_homogeneous_constant_solves == 0 && *hsd_result.ipm_homogeneous_direction_solves == 0 &&
      *hsd_result.ipm_homogeneous_scaled_products == 0 && !log.last && log.constant_calls_started == 0 &&
      log.directions_started == 0 && energy.started == 0 && coupled_state::current() == &log,
      "真实HSD过期入口须沿原共享epoch抛预算且保持零factor/P/A，不因warm helper启动工作或提交点");
  for (bool prepare : {false, true}) {
    WarmRegularizationSystem system(prepare);
    const auto snapshot = system.result;
    const int previous_level = system.factor->regularization_level();
    const auto stored = std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end());
    bool factor_stopped = false;
    try { system.factor->factor(system.h, system.options, expired_start, system.result,
        solver::v100_regularization_seed(*system.factor)); }
    catch (const solver::BudgetExpired&) { factor_stopped = true; }
    same_non_timing(system.result, snapshot);
    const auto after_rows = std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end());
    // 原factor在trial入口预算前已经resize；fresh对象可留下零行缓存，不能虚称内部完全无写入。
    const auto expected_rows = prepare ? stored : std::vector<double>(system.h.size(), 0);
    require(factor_stopped && system.factor->regularization_level() == previous_level &&
        after_rows == expected_rows &&
        solver::v100_regularization_seed(*system.factor) == previous_level &&
        coupled_state::current() == &log && log.directions_started == 0 && energy.started == 0,
        "实际factor过期入口保原public计数/level及已发生resize前缀，不启动分解；seed不重置时钟或Scope");
  }
  // 此组仅覆盖真实HSD与factor的过期入口；未增加注入钩，也不声称穷尽内部预算位置。
  ++budget_cases;
  details.push_back({"warm_regularization_hsd_budget", "budget"});
}

void check_fma_production_binding() {
  namespace axpy = solver::detail::fma_axpy;
  ForcingWeightHsdFixture fixture;
  bool forced = false;
  const auto answer = fixture.solve_weighted(&forced, true);
  coupled_explicit_return_log(fixture.log);
  require(forced && fixture.log.krylov_completed == 1 && fixture.log.preconditioner_completed == 2 &&
      fixture.log.products_completed == 3 && fixture.log.true_checks_completed == 2,
      "实际生产adapter须进入非平凡MGS和已有真实候选，不能只测初P旁路");
  const auto data = solver::v100_tolerances_and_rhs_norms(fixture.scaling, fixture.weighted_rhs);
  const coupled_kernel::InitialForcing forcing{data.rhs_norms};
  const auto p = [&](std::span<const double> input, bool refinement) {
    if (!refinement) return std::vector<double>(input.size(), 0);
    std::vector<double> modified(input.begin(), input.end());
    modified[1] *= 1.05; modified[2] *= 1.1;
    return InitialForcingHsdFixture::inverse(modified);
  };
  const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& image) {
    solver::v94_hsd_product(fixture.scaling, fixture.starts, fixture.rows, fixture.values, fixture.costs,
        fixture.bounds, fixture.slack, fixture.dual, 2, 8, input, image, {});
  };
  coupled_kernel::Work control;
  const auto legacy = coupled_kernel::solve({1, 2, 1}, fixture.weighted_rhs, data.tolerances, p, a, a,
      control, {}, &forcing, true, true);
  require(std::string_view(control.status) == "forced_candidate" && control.krylov_completed == 1 &&
      control.preconditioner_completed == 2 && control.products_completed == 3,
      "末硬件参数默认false的纯核控制须保父数学与原P/A账本");
  for (std::size_t i = 0; i < answer.size(); ++i)
    energy_near(answer[i], legacy[i], "生产true与父MGS控制均须保持本非单位HSD的独立单列方向关系");
  std::vector<double> value(4, 1), basis(4, 0x1.ffffffffffffep-1);
  bool used = false;
  const auto observed = axpy::subtract_scaled(value, basis, 0x1.0000000000001p0, {}, true, &used);
  const auto state = axpy::read_control_state();
  require(observed.finite && observed.completed == 4 &&
      used == (axpy::hardware_available() && axpy::environment_supported(state.mxcsr, state.x87_control)),
      "实际生产true绑定由源码审计核；同进程probe必须记录真正target选择，不能强制假设FMA存在");
  if (used) require(value == std::vector<double>(4, 0x1p-104), "已选择硬件必须有真实融合取消误差证据");
  ++transaction_cases;
  details.push_back({"fma_production_binding", "transaction"});
}

void check_fma_real_hsd_mapping() {
  Mode quiet(true);
  for (bool maximum : {false, true}) {
    const auto model = coupled(true, maximum);
    const auto options = policy();
    const auto start = solver::Clock::now();
    candidate::Stats stats;
    coupled_state::Log log; log.enabled = true;
    coupled_state::Scope scope(&log);
    energy_state::Log energy; energy.enabled = true;
    energy_state::Scope energy_scope(&energy);
    const auto result = candidate::control(model, options, start, stats,
        [&](const solver::Model& reduced, const solver::SolveOptions& child_options, solver::Clock::time_point child_start) {
          require(child_start == start, "FMA不得为真实HSD重置共享预算起点");
          ++component_ipm_calls;
          return solver::solve_ipm_core(reduced, child_options, child_start);
        });
    original_quality(model, result, maximum ? 104.0 / 3.0 : 88.0 / 3.0);
    coupled_explicit_return_log(log);
    require(stats.accepted && stats.oracle.accepted == 1 && std::string_view(stats.route) == "reduced" &&
        result.solution[0] == 3 && *result.ipm_iterations > 0 && log.directions_started == 2 * *result.ipm_iterations &&
        log.constant_calls_completed == *result.ipm_homogeneous_constant_solves &&
        log.preconditioner_completed == *result.ipm_homogeneous_direction_solves &&
        log.products_completed == *result.ipm_homogeneous_scaled_products &&
        log.krylov_completed == *result.ipm_homogeneous_scaled_refinement_steps &&
        *result.ipm_homogeneous_scaled_inexact_returns == log.initial_forcing_returns + log.candidate_forcing_returns,
        "FMA组合必须真实经过HSD、固定列和最大化恢复及原q验收，不能以硬件资格代替LP成功");
  }
  ++mapping_cases;
  details.push_back({"fma_real_hsd_mapping", "mapping"});
}

struct FmaLedgerFixture : CoupledScalarSystem {
  const std::vector<double> constant{-0.5}, rhs{0, 0, 0, 1};
  const long double denominator = solver::detail::ipm_homogeneous_scaled_denominator(scaling, constant);
  coupled_state::Log log;
  bool forced = false;

  FmaLedgerFixture() { log.enabled = true; }
  std::vector<long double> solve(const std::function<void()>& budget = {}) {
    coupled_state::Scope scope(&log);
    const auto p = [&](std::span<const double> input, bool refinement) {
      const auto reduced_rhs = solver::detail::ipm_homogeneous_scaled_reduced_rhs(scaling, input, budget);
      const auto reduced = factor->solve_homogeneous_direct(reduced_rhs, options, start, result, false, refinement);
      return solver::detail::ipm_homogeneous_scaled_recover(scaling, cone.costs, cone.rhs,
          input, reduced, constant, denominator, budget);
    };
    const auto a = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& output) {
      if (budget) budget();
      ++*result.ipm_homogeneous_scaled_products;
      solver::v94_hsd_product(scaling, cone.starts, cone.rows, cone.values, cone.costs, cone.rhs,
          point.s, point.z, 1, 1, input, output, budget);
      if (budget) budget();
    };
    return solver::v100_solve_coupled(scaling, rhs, p, a, a, budget, "corrector", 13, &forced);
  }
};

void check_fma_hsd_budget_prefix() {
  FmaLedgerFixture complete;
  int checks = 0;
  const auto answer = complete.solve([&] { ++checks; });
  coupled_strict_log(complete.log);
  require(!complete.forced && complete.log.krylov_completed > 0 && checks > 2 &&
      complete.log.preconditioner_completed == *complete.result.ipm_homogeneous_direction_solves &&
      complete.log.products_completed == *complete.result.ipm_homogeneous_scaled_products,
      "真实因子和故意近似常量须实际进入MGS，公共P/A与私有完成数保持原边界");
  const auto explicit_product = [](const auto& v) {
    return std::vector<long double>{static_cast<long double>(v[1]) - v[2], static_cast<long double>(v[0]) + v[3],
        static_cast<long double>(v[0]) + v[1], static_cast<long double>(v[2]) + v[3]};
  };
  coupled_independent_equations(complete.scaling, complete.rhs, answer, explicit_product);
  int in_search = 0;
  for (int cutoff = 1; cutoff <= checks; ++cutoff) {
    FmaLedgerFixture fixture;
    int observed = 0; bool stopped = false;
    std::vector<long double> output{101, 202};
    try { output = fixture.solve([&] { if (++observed == cutoff) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && observed == cutoff && !fixture.forced && output == std::vector<long double>({101, 202}) &&
        fixture.log.directions_completed == 0 && *fixture.result.ipm_homogeneous_scaled_inexact_returns == 0,
        "FMA实际组合的每个预算前缀须保原类型、旧输出和未交付事实");
    if (fixture.log.last) {
      const auto& work = fixture.log.last->work;
      const auto direct = *fixture.result.ipm_homogeneous_direction_solves;
      const auto products = *fixture.result.ipm_homogeneous_scaled_products;
      const auto refinement = *fixture.result.ipm_homogeneous_scaled_refinement_steps;
      require(std::string_view(work.status) == "budget" &&
          work.preconditioner_completed <= direct && direct <= work.preconditioner_started &&
          work.preconditioner_started - direct <= 1 && work.products_completed <= products && products <= work.products_started &&
          work.products_started - products <= 1 && work.krylov_completed <= refinement && refinement <= work.krylov_started &&
          *fixture.result.ipm_refinement_steps >= refinement && work.krylov_started <= 96 && work.cycles_started <= 3,
          "AVX段不得增加P/A或冒记refinement；回调开始与实际direct/product仍只有真实末partial前缀");
      if (work.krylov_started > 0) ++in_search;
    }
  }
  require(in_search > 0, "预算组件必须实际覆盖进入MGS之后的中断，而非只测入口");
  ++budget_cases;
  details.push_back({"fma_hsd_budget_prefix", "budget"});
}

struct NormalInverseData {
  using Index = std::int64_t;
  std::vector<Index> starts, rows;
  std::vector<std::int8_t> signs;
  std::vector<double> values;
  std::vector<std::vector<long double>> dense;
};

NormalInverseData normal_inverse_data(bool changed = false) {
  // 原始/对偶坐标交错，独立完整矩阵显式保留非单位D、不同R及所有G项。
  if (changed) return {{0, 1, 3, 5, 8, 11}, {0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4}, {1, -1, 1, -1, 1},
      {5, -2, -3, 0.75, 2.75, 2.5, -3, -17, 1.5, 4, 13},
      {{5, -2, 0, 2.5, 0}, {-2, -3, 0.75, 0, 1.5}, {0, 0.75, 2.75, -3, 0},
       {2.5, 0, -3, -17, 4}, {0, 1.5, 0, 4, 13}}};
  return {{0, 1, 3, 5, 8, 11}, {0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4}, {1, -1, 1, -1, 1},
      {3, 1.25, -5, -2, 7, -3, 4.5, -13, 0.5, 2, 11},
      {{3, 1.25, 0, -3, 0}, {1.25, -5, -2, 0, 0.5}, {0, -2, 7, 4.5, 0},
       {-3, 0, 4.5, -13, 2}, {0, 0.5, 0, 2, 11}}};
}

std::unique_ptr<solver::detail::NormalCholesky> normal_inverse_object(const NormalInverseData& data,
    solver::detail::NormalCholeskyKernel kernel = solver::detail::NormalCholeskyKernel::cholmod_supernodal_aat) {
  return std::make_unique<solver::detail::NormalCholesky>(data.signs.size(), data.starts, data.rows,
      data.signs, std::function<void()>{}, false, false, 0, kernel);
}

bool normal_inverse_factor(solver::detail::NormalCholesky& factor, std::span<const double> values,
    bool enabled, const std::function<void()>& budget = {}) {
  coupled_state::Log log; log.enabled = enabled;
  coupled_state::Scope scope(&log);
  return factor.factor(values, budget);
}

std::vector<double> normal_inverse_rhs(const NormalInverseData& data, std::span<const double> expected) {
  require(data.dense.size() == expected.size(), "完整Normal夹具的维数须一致");
  std::vector<double> rhs(expected.size());
  for (std::size_t row_index = 0; row_index < rhs.size(); ++row_index) {
    long double value = 0;
    for (std::size_t col = 0; col < rhs.size(); ++col) value += data.dense[row_index][col] * expected[col];
    rhs[row_index] = static_cast<double>(value);
    require(std::isfinite(rhs[row_index]) && (value == 0 || rhs[row_index] != 0),
        "完整KKT测试右端须有限且非零项不因构造而消失");
  }
  return rhs;
}

std::vector<long double> normal_inverse_dense_solve(std::vector<std::vector<long double>> matrix,
    std::span<const double> rhs) {
  // 独立带主元高斯消元不调用Normal、Schur缓存或被测回代公式。
  std::vector<long double> answer(rhs.begin(), rhs.end());
  for (std::size_t col = 0; col < answer.size(); ++col) {
    std::size_t pivot = col;
    for (std::size_t row_index = col + 1; row_index < answer.size(); ++row_index)
      if (std::abs(matrix[row_index][col]) > std::abs(matrix[pivot][col])) pivot = row_index;
    require(matrix[pivot][col] != 0 && std::isfinite(matrix[pivot][col]), "独立完整KKT必须可逆");
    std::swap(matrix[pivot], matrix[col]); std::swap(answer[pivot], answer[col]);
    for (std::size_t row_index = col + 1; row_index < answer.size(); ++row_index) {
      const long double multiplier = matrix[row_index][col] / matrix[col][col];
      for (std::size_t j = col + 1; j < answer.size(); ++j)
        matrix[row_index][j] -= multiplier * matrix[col][j];
      answer[row_index] -= multiplier * answer[col];
      matrix[row_index][col] = 0;
    }
  }
  for (std::size_t row_index = answer.size(); row_index-- > 0;) {
    for (std::size_t col = row_index + 1; col < answer.size(); ++col)
      answer[row_index] -= matrix[row_index][col] * answer[col];
    answer[row_index] /= matrix[row_index][row_index];
    require(std::isfinite(answer[row_index]), "独立完整KKT解须有限");
  }
  return answer;
}

std::vector<double> normal_inverse_check(solver::detail::NormalCholesky& factor,
    const NormalInverseData& data, std::span<const double> expected,
    std::span<const double> coordinate_scale = {}, bool compare_dense = true) {
  const auto original = normal_inverse_rhs(data, expected);
  auto actual = original;
  require(factor.solve(actual), "真实已分解Normal系统必须可完整回代");
  require(coordinate_scale.empty() || coordinate_scale.size() == actual.size(), "Normal验收坐标尺度须匹配");
  const auto dense = compare_dense ? normal_inverse_dense_solve(data.dense, original) : std::vector<long double>{};
  for (std::size_t row_index = 0; row_index < actual.size(); ++row_index) {
    require(std::isfinite(actual[row_index]), "实际Normal解必须有限");
    const long double scale = coordinate_scale.empty() ? 1 : coordinate_scale[row_index];
    const long double target = static_cast<long double>(expected[row_index]) * scale;
    require(std::abs((static_cast<long double>(actual[row_index]) - expected[row_index]) * scale) <=
        1e-11L * std::max(1.0L, std::abs(target)), "实际Normal解必须恢复已知完整KKT解及其原坐标尺度");
    if (compare_dense) require(std::abs(static_cast<long double>(actual[row_index]) - dense[row_index]) <=
        1e-11L * std::max(1.0L, std::abs(dense[row_index])), "实际Normal解须符合独立完整dense求解");
    long double product = 0;
    for (std::size_t col = 0; col < actual.size(); ++col) product += data.dense[row_index][col] * actual[col];
    require(std::isfinite(product) && std::abs(product - original[row_index]) <=
        1e-11L * std::max(1.0L, std::abs(static_cast<long double>(original[row_index]))),
        "所有原始及对偶行必须通过独立完整KKT残差，不能只验约化Schur方程");
  }
  const auto statistics = factor.refinement_statistics();
  require(factor.regularized_pivots() == 0 && statistics.residual_products == 0 &&
      statistics.refinement_steps == 0 && statistics.refinement_acceptances == 0,
      "倒数缓存不能暗加主元正则、Schur残差或迭代精化");
  return actual;
}

void check_normal_inverse_multi_rhs() {
  const auto data = normal_inverse_data();
  auto factor = normal_inverse_object(data);
  require(normal_inverse_factor(*factor, data.values, true), "活动Scope中的非单位D/R矩形系统应分解成功");
  const auto storage = factor->factor_nonzeros();
  require(storage > 0 && factor->factor_dimension() == 2, "实际Normal因子须具有两个保留对偶坐标");
  const std::vector<std::vector<double>> targets{{1, 2, -3, 4, -5}, {-0.25, 3, 0.5, -2, 7},
      {0, 0, 0, 0, 0}, {3.5, -1.25, 2.75, 0.125, -0.75}};
  for (const auto& expected : targets) {
    normal_inverse_check(*factor, data, expected);
    require(factor->factor_nonzeros() == storage, "同一真实因子的多个右端不能触发隐藏重分解或改存储");
  }
  ++transaction_cases;
  details.push_back({"normal_inverse_multi_rhs", "transaction"});
}

void check_normal_inverse_refactor() {
  const auto original = normal_inverse_data(), changed = normal_inverse_data(true);
  auto factor = normal_inverse_object(original);
  const std::vector<double> expected{2, -3, 0.75, 1.5, -4};
  require(normal_inverse_factor(*factor, original.values, true), "原矩形系统应成功建立因子");
  normal_inverse_check(*factor, original, expected);
  require(normal_inverse_factor(*factor, changed.values, true), "同模式的新D/G/R必须重新成功分解");
  normal_inverse_check(*factor, changed, expected);
  require(normal_inverse_factor(*factor, original.values, true), "返回原D/G/R应重新分解而不累乘旧缓存");
  normal_inverse_check(*factor, original, std::vector<double>{-1, 0.5, 2, -3, 4});
  ++mapping_cases;
  details.push_back({"normal_inverse_refactor", "mapping"});
}

void check_normal_inverse_factor_failure() {
  const auto data = normal_inverse_data();
  auto factor = normal_inverse_object(data);
  std::vector<std::vector<double>> invalid;
  invalid.push_back({1});
  for (int kind = 0; kind < 6; ++kind) {
    auto values = data.values;
    if (kind == 0) values[0] = 0;
    if (kind == 1) values[0] = -3;
    if (kind == 2) values[2] = 0;
    if (kind == 3) values[2] = 5;
    if (kind == 4) values[4] = std::numeric_limits<double>::quiet_NaN();
    if (kind == 5) values[5] = infinity;
    invalid.push_back(std::move(values));
  }
  for (const auto& values : invalid) {
    require(normal_inverse_factor(*factor, data.values, true), "每次非法分解前须确有可用的旧因子及缓存");
    normal_inverse_check(*factor, data, std::vector<double>{1, -2, 3, -4, 5});
    require(!normal_inverse_factor(*factor, values, true), "错误长度、非法D/R及非有限输入必须拒绝");
    std::vector<double> rhs{1, 2, 3, 4, 5};
    const auto untouched = rhs;
    require(!factor->solve(rhs) && rhs == untouched && factor->factor_nonzeros() == 0,
        "非法分解必须作废此前成功的因子及可用缓存，不能提交任何旧解");
    require(normal_inverse_factor(*factor, data.values, true), "非法输入后应可显式重建完整因子");
    normal_inverse_check(*factor, data, std::vector<double>{-2, 1, 0.5, -0.75, 4});
  }
  ++transaction_cases;
  details.push_back({"normal_inverse_factor_failure", "transaction"});
}

struct NormalInverseBudgetStop : std::runtime_error { using std::runtime_error::runtime_error; };

void check_normal_inverse_budget_prefix() {
  const auto data = normal_inverse_data(), changed = normal_inverse_data(true);
  auto factor = normal_inverse_object(data);
  int factor_checks = 0;
  require(normal_inverse_factor(*factor, changed.values, true, [&] { ++factor_checks; }) && factor_checks > 1,
      "真实矩形分解预算基准必须包含内部及完成边界");
  for (int cutoff = 1; cutoff <= factor_checks; ++cutoff) {
    require(normal_inverse_factor(*factor, data.values, true), "每个预算截断前须真实发布一份旧因子");
    int observed = 0; bool stopped = false;
    try {
      normal_inverse_factor(*factor, changed.values, true, [&] {
        if (++observed == cutoff) throw NormalInverseBudgetStop("Normal倒数分解预算");
      });
    } catch (const NormalInverseBudgetStop& error) { stopped = std::string_view(error.what()) == "Normal倒数分解预算"; }
    std::vector<double> rhs{1, 2, 3, 4, 5};
    const auto untouched = rhs;
    require(stopped && observed == cutoff && !factor->solve(rhs) && rhs == untouched,
        "实际分解每个预算前缀须原样抛出、作废旧因子并保留完整未提交RHS，包括最后发布前边界");
    require(normal_inverse_factor(*factor, changed.values, true), "预算中断后显式重分解须重建新D的因子");
    normal_inverse_check(*factor, changed, std::vector<double>{-1, 2, -3, 4, -5});
  }
  const std::vector<double> expected{1, -2, 3, -4, 5};
  const auto original = normal_inverse_rhs(changed, expected);
  auto complete = original;
  int solve_checks = 0;
  require(factor->solve(complete, [&] { ++solve_checks; }) && solve_checks > 1,
      "实际回代预算基准必须遍历约化、库求解、恢复及末提交检查");
  for (int cutoff = 1; cutoff <= solve_checks; ++cutoff) {
    auto rhs = original;
    int observed = 0; bool stopped = false;
    try { factor->solve(rhs, [&] { if (++observed == cutoff) throw NormalInverseBudgetStop("Normal倒数回代预算"); }); }
    catch (const NormalInverseBudgetStop& error) { stopped = std::string_view(error.what()) == "Normal倒数回代预算"; }
    require(stopped && observed == cutoff && rhs == original,
        "实际Normal回代任一预算边界不得部分提交RHS或转换原异常类型");
    normal_inverse_check(*factor, changed, expected);
  }
  ++budget_cases;
  details.push_back({"normal_inverse_budget_prefix", "budget"});
}

void check_normal_inverse_scope_and_fallback() {
  const auto original = normal_inverse_data(), changed = normal_inverse_data(true);
  const std::vector<double> expected{1.25, -2.5, 3.75, -4.5, 0.625};
  auto factor = normal_inverse_object(original);
  require(normal_inverse_factor(*factor, original.values, true), "活动factor应成功，退出Scope后仍是同一个实际因子");
  require(!coupled_state::enabled(), "factor辅助Scope退出后须恢复外层关闭状态");
  normal_inverse_check(*factor, original, expected);
  // 同一对象先曾填旧缓存，再在关闭状态用不同D分解；随后开Scope求解以检出错误的TLS晚绑定。
  require(normal_inverse_factor(*factor, changed.values, false), "关闭factor也必须重建不同D/G/R的完整因子");
  coupled_state::Log log; log.enabled = true;
  {
    coupled_state::Scope scope(&log);
    normal_inverse_check(*factor, changed, expected);
    coupled_state::Log disabled;
    {
      coupled_state::Scope isolation(&disabled);
      require(!coupled_state::enabled(), "关闭隔离须压住外层活动Scope");
      normal_inverse_check(*factor, changed, std::vector<double>{-3, 1, 2, -0.5, 4});
    }
    require(coupled_state::current() == &log && log.directions_started == 0 && log.preconditioner_started == 0,
        "Normal公共API的Scope切换不应创建联立方向或预条件器账本");
    for (const auto kernel : {solver::detail::NormalCholeskyKernel::eigen,
                             solver::detail::NormalCholeskyKernel::cholmod_supernodal}) {
      auto non_aat = normal_inverse_object(original, kernel);
      require(non_aat->factor(original.values), "非矩形内核在活动Scope中仍应保留原分解");
      normal_inverse_check(*non_aat, original, expected);
    }
    const NormalInverseData primal{{0, 1, 2}, {0, 1}, {1, 1}, {3, 7}, {{3, 0}, {0, 7}}};
    auto primal_factor = normal_inverse_object(primal);
    require(primal_factor->factor(primal.values) && primal_factor->factor_dimension() == 0 &&
        primal_factor->factor_nonzeros() == 0, "纯原始系统不应建立零维CHOLMOD因子");
    normal_inverse_check(*primal_factor, primal, std::vector<double>{2, -3});
    const NormalInverseData dual{{0, 1, 2}, {0, 1}, {-1, -1}, {-3, -11}, {{-3, 0}, {0, -11}}};
    auto dual_factor = normal_inverse_object(dual);
    require(dual_factor->factor(dual.values), "无原始列的逐行R矩形系统仍须成功");
    normal_inverse_check(*dual_factor, dual, std::vector<double>{-2, 4});
    const NormalInverseData empty{{0}, {}, {}, {}, {}};
    auto empty_factor = normal_inverse_object(empty);
    std::vector<double> rhs;
    require(empty_factor->factor({}) && empty_factor->solve(rhs) && rhs.empty() &&
        empty_factor->factor_nonzeros() == 0, "空矩形系统应安全通过且不触及未发布的倒数");
  }
  require(!coupled_state::enabled(), "所有Normal组件Scope结束后必须恢复关闭状态");
  // 数值正确只证明这些实际调用可用；具体按factor快照选分支另由源码审计核实。
  ++mapping_cases;
  details.push_back({"normal_inverse_scope_and_fallback", "mapping"});
}

void check_normal_inverse_extreme_recovery() {
  constexpr bool wide_range = std::numeric_limits<double>::is_iec559 &&
      std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53 &&
      std::numeric_limits<double>::max_exponent == 1024 && std::numeric_limits<double>::min_exponent == -1021 &&
      std::numeric_limits<double>::has_denorm == std::denorm_present &&
      std::numeric_limits<long double>::is_iec559 && std::numeric_limits<long double>::radix == 2 &&
      std::numeric_limits<long double>::digits >= 53 && std::numeric_limits<long double>::max_exponent >= 4096 &&
      std::numeric_limits<long double>::min_exponent <= -4096;
  const auto power = [](int exponent) { return std::ldexp(1.0, exponent); };
  if constexpr (wide_range) {
    const NormalInverseData tiny{{0, 1, 3}, {0, 0, 1}, {1, -1},
        {power(-1074), power(-37), -power(1000)},
        {{power(-1074), power(-37)}, {power(-37), -power(1000)}}};
    auto factor = normal_inverse_object(tiny);
    require(normal_inverse_factor(*factor, tiny.values, true), "最小正binary64 D的实际矩形系统应可分解");
    const long double inverse = 1.0L / tiny.values[0];
    require(std::isfinite(inverse) && inverse > std::numeric_limits<double>::max(),
        "此夹具须实际要求超出double范围的有限LD倒数，不能窄存缓存");
    normal_inverse_check(*factor, tiny, std::vector<double>{power(537), power(-499)},
        std::vector<double>{power(-537), power(500)}, false);
    const NormalInverseData crossing{{0, 1, 3, 5}, {0, 0, 1, 0, 2}, {1, -1, -1},
        {power(500), power(500), -power(500), power(-700), -power(-1074)},
        {{power(500), power(500), power(-700)}, {power(500), -power(500), 0}, {power(-700), 0, -power(-1074)}}};
    auto crossing_factor = normal_inverse_object(crossing);
    require(normal_inverse_factor(*crossing_factor, crossing.values, true), "跨越double商下界的实际矩形系统应成功");
    normal_inverse_check(*crossing_factor, crossing, std::vector<double>{0, power(-500), power(700)},
        std::vector<double>{power(500), power(500), power(-700)}, false);
    const double largest = std::numeric_limits<double>::max();
    const NormalInverseData large{{0, 1, 3}, {0, 0, 1}, {1, -1}, {largest, 0, -1}, {{largest, 0}, {0, -1}}};
    auto large_factor = normal_inverse_object(large);
    require(normal_inverse_factor(*large_factor, large.values, true), "最大有限D不能使缓存倒数提前归零");
    normal_inverse_check(*large_factor, large, std::vector<double>{power(-1000), 2},
        std::vector<double>{power(1000), 1}, false);
    std::vector<double> underflow_rhs{std::numeric_limits<double>::denorm_min(), 1};
    const auto untouched = underflow_rhs;
    require(!large_factor->solve(underflow_rhs) && underflow_rhs == untouched,
        "完整实际解的非零坐标下溢为double零时仍须拒绝且不提交RHS");
  }
  // C=(3,4)使本行Schur为25；分解不引入无关sqrt(3)舍入来盖住恢复中的小非零量。
  const NormalInverseData cancellation{{0, 1, 3}, {0, 0, 1}, {1, -1}, {4, 6, -16}, {{4, 6}, {6, -16}}};
  auto factor = normal_inverse_object(cancellation);
  require(normal_inverse_factor(*factor, cancellation.values, true), "相消恢复夹具须先真实分解");
  const auto cancelled = normal_inverse_check(*factor, cancellation, std::vector<double>{power(-30), power(10)},
      std::vector<double>{power(30), power(-10)});
  require(cancelled[0] != 0, "完整恢复中的小非零原始方向不能被相消误删");
  const double smallest = std::numeric_limits<double>::min();
  const NormalInverseData overflow{{0, 1, 3}, {0, 0, 1}, {1, -1}, {smallest, 0, -1}, {{smallest, 0}, {0, -1}}};
  auto overflow_factor = normal_inverse_object(overflow);
  require(normal_inverse_factor(*overflow_factor, overflow.values, true), "不可表示解的系统本身仍应可分解");
  std::vector<double> rhs{std::numeric_limits<double>::max(), 1};
  const auto untouched = rhs;
  require(!overflow_factor->solve(rhs) && rhs == untouched, "有限RHS若产生不可表示完整解，必须拒绝且保留原右端");
  rhs = {infinity, 1};
  require(!overflow_factor->solve(rhs) && std::isinf(rhs[0]) && rhs[1] == 1,
      "非有限RHS仍须在回代入口拒绝，不因缓存跳过原输入检查");
  if constexpr (std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53 &&
                std::numeric_limits<double>::has_denorm == std::denorm_present) {
    require(!normal_inverse_factor(*factor, std::vector<double>{power(1000), power(-1074), -1}, true),
        "非零G除sqrt(D)舍入为零时仍必须按原AAT装配规则拒绝");
    rhs = {1, 2};
    require(!factor->solve(rhs) && rhs == std::vector<double>({1, 2}), "装配下溢失败后不能沿用相消夹具的旧因子");
  }
  // 这里只在实际limits允许的宽指数平台运行超binary64中间量，不修改任何FP控制状态。
  ++transaction_cases;
  details.push_back({"normal_inverse_extreme_recovery", "transaction"});
}

struct NormalBudgetStar {
  std::size_t columns, retained;
  std::vector<std::int64_t> starts{0}, rows;
  std::vector<std::int8_t> signs;
  std::vector<double> values;

  NormalBudgetStar(std::size_t primal, std::size_t dual) : columns(primal), retained(dual), signs(primal + dual, -1) {
    require(primal <= 1 || dual <= 1, "预算星形夹具只能线性生成，不展开大稠密KKT");
    for (std::size_t col = 0; col < primal + dual; ++col) {
      if (col < primal) {
        signs[col] = 1;
        rows.push_back(static_cast<std::int64_t>(col)); values.push_back(4);
      } else {
        for (std::size_t row = 0; row < primal; ++row) {
          rows.push_back(static_cast<std::int64_t>(row)); values.push_back(2);
        }
        rows.push_back(static_cast<std::int64_t>(col)); values.push_back(-9);
      }
      starts.push_back(static_cast<std::int64_t>(rows.size()));
    }
    require(values.size() == primal + dual + primal * dual, "预算夹具的存储必须与实际G非零数线性一致");
  }

  std::vector<double> point(double primal, double dual) const {
    std::vector<double> answer(columns + retained, dual);
    std::fill(answer.begin(), answer.begin() + static_cast<std::ptrdiff_t>(columns), primal);
    return answer;
  }
  std::vector<long double> product(std::span<const double> answer) const {
    require(answer.size() == columns + retained, "预算夹具的完整星算子维数须一致");
    std::vector<long double> output(answer.size());
    for (std::size_t col = 0; col < columns; ++col) output[col] = 4.0L * answer[col];
    for (std::size_t row = 0; row < retained; ++row) {
      output[columns + row] = -9.0L * answer[columns + row];
      for (std::size_t col = 0; col < columns; ++col) {
        output[col] += 2.0L * answer[columns + row];
        output[columns + row] += 2.0L * answer[col];
      }
    }
    return output;
  }
  std::vector<double> rhs(std::span<const double> expected) const {
    const auto image = product(expected);
    std::vector<double> answer(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) {
      answer[i] = static_cast<double>(image[i]);
      require(std::isfinite(answer[i]) && (image[i] == 0 || answer[i] != 0), "预算测试的已知完整RHS须有限且可表示");
    }
    return answer;
  }
};

std::unique_ptr<solver::detail::NormalCholesky> normal_budget_object(const NormalBudgetStar& data,
    solver::detail::NormalCholeskyKernel kernel = solver::detail::NormalCholeskyKernel::cholmod_supernodal_aat,
    bool factor_scope = true) {
  auto factor = std::make_unique<solver::detail::NormalCholesky>(data.signs.size(), data.starts, data.rows,
      data.signs, std::function<void()>{}, false, false, 0, kernel);
  require(normal_inverse_factor(*factor, data.values, factor_scope), "预算组件必须先真实分解原星形矩阵");
  return factor;
}

bool normal_budget_bits_equal(std::span<const double> left, std::span<const double> right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (std::bit_cast<std::uint64_t>(left[i]) != std::bit_cast<std::uint64_t>(right[i])) return false;
  return true;
}

struct NormalBudgetRun { std::vector<double> answer; int callbacks = 0; bool success = false; };

NormalBudgetRun normal_budget_run(solver::detail::NormalCholesky& factor, const std::vector<double>& rhs,
    bool enabled) {
  NormalBudgetRun run; run.answer = rhs;
  coupled_state::Log log; log.enabled = enabled;
  coupled_state::Scope scope(&log);
  run.success = factor.solve(run.answer, [&] { ++run.callbacks; });
  return run;
}

void normal_budget_equations(const NormalBudgetStar& data, const std::vector<double>& rhs,
    const std::vector<double>& expected, const NormalBudgetRun& run) {
  require(run.success && run.answer.size() == expected.size(), "预算组织变化后必须真实返回完整原坐标解");
  const auto image = data.product(run.answer);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require(std::isfinite(run.answer[i]) && std::abs(static_cast<long double>(run.answer[i]) - expected[i]) <=
        1e-11L * std::max(1.0L, std::abs(static_cast<long double>(expected[i]))),
        "预算组件须恢复已知原解，不能只检查回调计数");
    require(std::isfinite(image[i]) && std::abs(image[i] - rhs[i]) <=
        1e-11L * std::max(1.0L, std::abs(static_cast<long double>(rhs[i]))),
        "预算组件必须以LD独立原算子核完整Mu=b，不只检查Schur解");
  }
}

void check_normal_budget_short_columns() {
  for (bool factor_scope : {false, true}) for (std::size_t columns : {1U, 63U, 64U, 65U, 129U}) {
    const NormalBudgetStar data(columns, 1);
    auto factor = normal_budget_object(data, solver::detail::NormalCholeskyKernel::cholmod_supernodal_aat, factor_scope);
    const auto expected = data.point(2, 1), rhs = data.rhs(expected);
    const auto off = normal_budget_run(*factor, rhs, false), on = normal_budget_run(*factor, rhs, true);
    normal_budget_equations(data, rhs, expected, off); normal_budget_equations(data, rhs, expected, on);
    require(normal_budget_bits_equal(on.answer, off.answer),
        "同一已factor对象的solve开关只能改预算，所有double位及符号零必须保留");
    require(off.callbacks == 7 + static_cast<int>(2 * columns) &&
        on.callbacks == 7 + static_cast<int>((4 * columns) / 256) && on.callbacks < off.callbacks,
        "短列真实回调须按两遍列头加entry共享批量，固定入口及库检查不能删");
  }
  ++transaction_cases;
  details.push_back({"normal_budget_short_columns", "transaction"});
}

void check_normal_budget_long_boundaries() {
  for (std::size_t degree : {254U, 255U, 256U, 257U, 511U}) {
    const NormalBudgetStar data(1, degree);
    auto factor = normal_budget_object(data);
    const auto expected = data.point(1, 0), rhs = data.rhs(expected);
    const auto off = normal_budget_run(*factor, rhs, false), on = normal_budget_run(*factor, rhs, true);
    normal_budget_equations(data, rhs, expected, off); normal_budget_equations(data, rhs, expected, on);
    require(normal_budget_bits_equal(on.answer, off.answer) && off.callbacks == 9 &&
        on.callbacks == 7 + static_cast<int>((2 * (degree + 1)) / 256),
        "实际长列须新增256项检查，254/255/256/257及511度边界按共享总工作量计数");
    if (degree == 511) require(on.callbacks > off.callbacks,
        "长列可能增加真实预算回调，不能把批量机制一律描述为减少检查");
    if (degree == 255) {
      const auto nonzero = data.point(1024, 1), nonzero_rhs = data.rhs(nonzero);
      const auto result = normal_budget_run(*factor, nonzero_rhs, true);
      normal_budget_equations(data, nonzero_rhs, nonzero, result);
    }
  }
  // 测试可见的是实际回调数与完整结果；触发项尚未算术须另由源码审计证明。
  ++transaction_cases;
  details.push_back({"normal_budget_long_boundaries", "transaction"});
}

void check_normal_budget_shared_passes() {
  for (std::size_t degree : {126U, 127U, 128U}) {
    const NormalBudgetStar data(1, degree);
    auto factor = normal_budget_object(data);
    const auto expected = data.point(2, 0), rhs = data.rhs(expected);
    const auto result = normal_budget_run(*factor, rhs, true);
    normal_budget_equations(data, rhs, expected, result);
    const int expected_checks = degree == 126 ? 7 : 8;
    require(result.callbacks == expected_checks,
        "约化和恢复须共享remaining，127个entry时两遍合计256项，固定库检查不能重置");
  }
  ++mapping_cases;
  details.push_back({"normal_budget_shared_passes", "mapping"});
}

void check_normal_budget_all_cutoffs() {
  for (const auto shape : {std::pair<std::size_t, std::size_t>{129, 1}, {1, 511}}) {
    const NormalBudgetStar data(shape.first, shape.second);
    auto factor = normal_budget_object(data);
    const auto expected = data.point(1, 0), rhs = data.rhs(expected);
    for (bool enabled : {false, true}) {
      const auto completed = normal_budget_run(*factor, rhs, enabled);
      normal_budget_equations(data, rhs, expected, completed);
      require(completed.callbacks > 2, "全部实际截断基准须已完成内部及末提交预算边界");
      for (int cutoff = 1; cutoff <= completed.callbacks; ++cutoff) {
        auto answer = rhs;
        int observed = 0; bool stopped = false;
        {
          coupled_state::Log log; log.enabled = enabled;
          coupled_state::Scope scope(&log);
          try { factor->solve(answer, [&] {
            if (++observed == cutoff) throw NormalInverseBudgetStop("Normal批量回代预算");
          }); } catch (const NormalInverseBudgetStop& error) {
            stopped = std::string_view(error.what()) == "Normal批量回代预算";
          }
        }
        require(stopped && observed == cutoff && normal_budget_bits_equal(answer, rhs),
            "实际每个callback截断须原异常透传且RHS逐位不提交，含新增entry检查与库后末边界");
      }
      const auto restored = normal_budget_run(*factor, rhs, enabled);
      normal_budget_equations(data, rhs, expected, restored);
      require(restored.callbacks == completed.callbacks && normal_budget_bits_equal(restored.answer, completed.answer),
          "所有中断后同一因子仍须可解，不残留本次批量remaining或部分输出");
    }
  }
  ++budget_cases;
  details.push_back({"normal_budget_all_cutoffs", "budget"});
}

void check_normal_budget_scope_snapshot() {
  const NormalBudgetStar data(64, 1);
  auto factor = normal_budget_object(data);
  const auto expected = data.point(2, 1), rhs = data.rhs(expected);
  const auto reference = normal_budget_run(*factor, rhs, false);
  for (bool entry_enabled : {false, true}) {
    NormalBudgetRun run; run.answer = rhs;
    coupled_state::Log entry_log; entry_log.enabled = entry_enabled;
    coupled_state::Log changed_log; changed_log.enabled = !entry_enabled;
    {
      coupled_state::Scope entry_scope(&entry_log);
      std::optional<coupled_state::Scope> persistent_change;
      run.success = factor->solve(run.answer, [&] {
        if (++run.callbacks == 1) persistent_change.emplace(&changed_log);
      });
      require(coupled_state::enabled() == !entry_enabled, "首callback中的Scope须持续到整个solve结束，不能只临时切换");
      persistent_change.reset();
      require(coupled_state::enabled() == entry_enabled, "内层持久Scope解除后必须恢复入口TLS");
    }
    normal_budget_equations(data, rhs, expected, run);
    require(run.callbacks == (entry_enabled ? 8 : 135) && normal_budget_bits_equal(run.answer, reference.answer),
        "策略必须在首entrycheck前快照，callback改变TLS不能重选检查频率或数学路径");
  }
  auto answer = rhs;
  int callbacks = 0; bool stopped = false;
  coupled_state::Log active; active.enabled = true;
  coupled_state::Log disabled;
  try {
    coupled_state::Scope entry_scope(&active);
    std::optional<coupled_state::Scope> persistent_change;
    factor->solve(answer, [&] {
      ++callbacks;
      if (callbacks == 1) persistent_change.emplace(&disabled);
      if (callbacks == 2) throw NormalInverseBudgetStop("Normal持久Scope预算");
    });
  } catch (const NormalInverseBudgetStop& error) { stopped = std::string_view(error.what()) == "Normal持久Scope预算"; }
  require(stopped && callbacks == 2 && normal_budget_bits_equal(answer, rhs) && !coupled_state::enabled(),
      "持久TLS变化后异常也须按RAII恢复原上下文与未提交RHS");
  ++mapping_cases;
  details.push_back({"normal_budget_scope_snapshot", "mapping"});
}

void check_normal_budget_fallback_rejection() {
  const NormalBudgetStar data(64, 1);
  const auto expected = data.point(2, 1), rhs = data.rhs(expected);
  for (const auto kernel : {solver::detail::NormalCholeskyKernel::eigen,
                           solver::detail::NormalCholeskyKernel::cholmod_supernodal}) {
    auto factor = normal_budget_object(data, kernel);
    const auto off = normal_budget_run(*factor, rhs, false), on = normal_budget_run(*factor, rhs, true);
    normal_budget_equations(data, rhs, expected, off); normal_budget_equations(data, rhs, expected, on);
    require(on.callbacks == off.callbacks && normal_budget_bits_equal(on.answer, off.answer),
        "非AAT内核不能因活动Scope改预算或数值");
  }
  for (const auto shape : {std::pair<std::size_t, std::size_t>{0, 0}, {129, 0}, {0, 2}}) {
    const NormalBudgetStar special(shape.first, shape.second);
    auto factor = normal_budget_object(special);
    const auto point = special.point(2, -1), input = special.rhs(point);
    const auto off = normal_budget_run(*factor, input, false), on = normal_budget_run(*factor, input, true);
    normal_budget_equations(special, input, point, off); normal_budget_equations(special, input, point, on);
    require(normal_budget_bits_equal(on.answer, off.answer), "空及纯块系统仍须逐位保持完整回代");
    if (shape.second == 0) require(off.callbacks == 2 + static_cast<int>(2 * shape.first) &&
        on.callbacks == 2 + static_cast<int>((2 * shape.first) / 256),
        "批量启用不能额外要求dual非空，空系统仍保两个固定检查");
  }
  auto factor = normal_budget_object(data);
  auto no_callback = rhs;
  {
    coupled_state::Log log; log.enabled = true;
    coupled_state::Scope scope(&log);
    require(factor->solve(no_callback), "没有callback时仍须保原成功数学路径");
  }
  const auto counted = normal_budget_run(*factor, rhs, false);
  require(normal_budget_bits_equal(no_callback, counted.answer), "空callback不能改变原解，是否跳过tick由源码审计核实");
  for (const auto& invalid : std::vector<std::vector<double>>{{1}, {infinity}, {std::numeric_limits<double>::quiet_NaN()}}) {
    auto input = rhs;
    if (invalid.size() == 1 && std::isfinite(invalid[0])) input = invalid;
    else input[0] = invalid[0];
    for (bool enabled : {false, true}) {
      const auto rejected = normal_budget_run(*factor, input, enabled);
      require(!rejected.success && rejected.callbacks == 1 && normal_budget_bits_equal(rejected.answer, input),
          "非法维数和非有限RHS须在原入口检查后拒绝且逐位保输入，不能为填batch继续运算");
    }
  }
  const NormalInverseData overflow{{0, 1, 3}, {0, 0, 1}, {1, -1},
      {std::numeric_limits<double>::min(), 0, -1}, {{std::numeric_limits<double>::min(), 0}, {0, -1}}};
  auto extreme = normal_inverse_object(overflow);
  require(normal_inverse_factor(*extreme, overflow.values, true), "原数值拒绝夹具本身须可分解");
  const std::vector<double> overflow_rhs{std::numeric_limits<double>::max(), 1};
  for (bool enabled : {false, true}) {
    const auto rejected = normal_budget_run(*extreme, overflow_rhs, enabled);
    require(!rejected.success && normal_budget_bits_equal(rejected.answer, overflow_rhs),
        "不可表示的完整Normal解仍须按原规则拒绝且不提交，不能因批量检查而越过数值守卫");
  }
  require(!normal_inverse_factor(*extreme, std::vector<double>{0, 0, -1}, true), "非法factor仍须作废原因子");
  const auto invalid_factor = normal_budget_run(*extreme, std::vector<double>{1, 2}, true);
  require(!invalid_factor.success && invalid_factor.callbacks == 1 && invalid_factor.answer == std::vector<double>({1, 2}),
      "已失效因子不能支付额外batch或借用旧解");
  ++transaction_cases;
  details.push_back({"normal_budget_fallback_rejection", "transaction"});
}

struct ProductBudgetData {
  std::size_t columns, rows_count;
  std::vector<std::int64_t> starts{0}, rows;
  std::vector<double> values, costs, bounds, slack, dual;
  std::vector<bool> inequality;
  double tau = 2, kappa = 8;
  solver::detail::IpmHomogeneousScaling scaling;

  ProductBudgetData(std::size_t n, std::size_t m, bool mixed = false)
      : columns(n), rows_count(m), costs(n), bounds(m), slack(m, 2), dual(m, 8), inequality(m, true) {
    if (mixed) {
      require(n == 2 && m == 3, "混合五块夹具必须保持显式小矩阵形状");
      starts = {0, 3, 5}; rows = {0, 1, 2, 0, 1}; values = {1, 3, 5, 2, -1};
      costs = {-1, 2}; bounds = {3, -2, 5}; slack = {2, 0, 8}; dual = {8, -3, 0.5};
      inequality = {true, false, true};
    } else {
      require(n <= 1 || m <= 1, "大型五块夹具只生成稀疏星形，不建立稠密矩阵");
      for (std::size_t col = 0; col < n; ++col) {
        costs[col] = static_cast<double>(static_cast<int>(col % 3) - 1) / 2;
        for (std::size_t row = 0; row < m; ++row) { rows.push_back(static_cast<std::int64_t>(row)); values.push_back(2); }
        starts.push_back(static_cast<std::int64_t>(rows.size()));
      }
      for (std::size_t row = 0; row < m; ++row) bounds[row] = static_cast<double>(static_cast<int>(row % 5) - 2) / 4;
    }
    rebuild_scaling();
  }
  void rebuild_scaling() {
    scaling = solver::detail::make_ipm_homogeneous_scaling(columns, inequality, slack, dual, tau, kappa);
  }
  template<class Scalar> std::vector<Scalar> direction() const {
    std::vector<Scalar> value(scaling.dimension());
    for (std::size_t col = 0; col < columns; ++col) value[col] = static_cast<Scalar>(static_cast<int>(col % 5) - 2) / 2;
    for (std::size_t row = 0; row < rows_count; ++row) value[columns + row] = static_cast<Scalar>(static_cast<int>(row % 7) - 3) / 4;
    const auto gap = columns + rows_count;
    for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i)
      value[gap + i] = static_cast<Scalar>(static_cast<int>(i % 3) - 1) / 8;
    value[value.size() - 2] = Scalar{0.5}; value.back() = Scalar{-0.25};
    return value;
  }
  template<class Scalar> std::vector<long double> explicit_product(std::span<const Scalar> value) const {
    const auto gap = columns + rows_count, tau_index = value.size() - 2;
    const long double dt = value[tau_index], dk = value.back();
    std::vector<long double> image(value.size());
    long double gap_value = dk;
    for (std::size_t col = 0; col < columns; ++col) {
      image[col] = static_cast<long double>(costs[col]) * dt;
      gap_value += static_cast<long double>(costs[col]) * value[col];
    }
    for (std::size_t row = 0; row < rows_count; ++row) {
      image[columns + row] = -static_cast<long double>(bounds[row]) * dt;
      gap_value += static_cast<long double>(bounds[row]) * value[columns + row];
    }
    // 独立物理方程散射；不用缩约H、预条件器或被测模板构造期望值。
    for (std::size_t col = 0; col < columns; ++col) for (auto k = starts[col]; k < starts[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k), row = static_cast<std::size_t>(rows[position]);
      image[col] += static_cast<long double>(values[position]) * value[columns + row];
      image[columns + row] += static_cast<long double>(values[position]) * value[col];
    }
    image[gap] = gap_value / scaling.d;
    for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
      const auto row = scaling.layout.inequality_rows[i];
      image[columns + row] += value[gap + i];
      image[gap + 1 + i] = (static_cast<long double>(dual[row]) * value[gap + i] +
          static_cast<long double>(slack[row]) * value[columns + row]) / scaling.lambda[i];
    }
    image.back() = (static_cast<long double>(kappa) * dt + static_cast<long double>(tau) * dk) / scaling.lambda_tau;
    return image;
  }
};

bool product_budget_equal(std::span<const long double> left, std::span<const long double> right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (!std::isfinite(left[i]) || !std::isfinite(right[i]) || left[i] != right[i] ||
        (left[i] == 0 && std::signbit(left[i]) != std::signbit(right[i]))) return false;
  // 无容差比较有效浮点数值及signed zero；不读取LD对象的padding。
  return true;
}

template<class Scalar>
void product_budget_call(const ProductBudgetData& data, const std::vector<Scalar>& value,
    std::vector<long double>& image, const std::function<void()>& check) {
  solver::v94_hsd_product(data.scaling, data.starts, data.rows, data.values, data.costs, data.bounds,
      data.slack, data.dual, data.tau, data.kappa, std::span<const Scalar>(value), image, check);
}

struct ProductBudgetRun { std::vector<long double> image{101, 202, 303}; int callbacks = 0; bool completed = false; };

template<class Scalar>
ProductBudgetRun product_budget_run(const ProductBudgetData& data, const std::vector<Scalar>& value, bool enabled) {
  ProductBudgetRun run;
  coupled_state::Log log; log.enabled = enabled;
  coupled_state::Scope scope(&log);
  product_budget_call(data, value, run.image, [&] { ++run.callbacks; });
  run.completed = true;
  return run;
}

template<class Scalar>
void product_budget_equations(const ProductBudgetData& data, const std::vector<Scalar>& value, const ProductBudgetRun& run) {
  const auto expected = data.explicit_product(std::span<const Scalar>(value));
  require(run.completed && run.image.size() == expected.size(), "五块模板须真实完整返回，不能只检查预算数");
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(std::isfinite(run.image[i]) && std::abs(run.image[i] - expected[i]) <=
        3e-14L * std::max(1.0L, std::abs(expected[i])), "Double及LD模板都须符合原G/S/Z/tau/kappa的独立完整五块");
}

void check_product_budget_double_wide() {
  const ProductBudgetData data(2, 3, true);
  const auto check_type = [&]<class Scalar>() {
    const auto value = data.direction<Scalar>();
    const auto off = product_budget_run(data, value, false), on = product_budget_run(data, value, true);
    product_budget_equations(data, value, off); product_budget_equations(data, value, on);
    require(product_budget_equal(off.image, on.image), "同一Scalar入口开关只能改预算，有效值位及signed zero必须相同");
  };
  check_type.template operator()<double>(); check_type.template operator()<long double>();
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    ProductBudgetData low_bit(1, 1);
    low_bit.values = {1}; low_bit.costs = {0}; low_bit.bounds = {0}; low_bit.slack = {1}; low_bit.dual = {1};
    low_bit.tau = low_bit.kappa = 1; low_bit.rebuild_scaling();
    const std::vector<long double> wide{1e16L - 1, 1e16L, -1e16L + 1, 0, 0};
    const std::vector<double> narrow(wide.begin(), wide.end());
    const auto off = product_budget_run(low_bit, wide, false), on = product_budget_run(low_bit, wide, true);
    const auto double_result = product_budget_run(low_bit, narrow, true);
    product_budget_equations(low_bit, wide, off); product_budget_equations(low_bit, wide, on);
    product_budget_equations(low_bit, narrow, double_result);
    require(product_budget_equal(off.image, on.image) && on.image[3] == 1 && double_result.image[3] == 0,
        "LD实际入口必须保留不可窄化的单位低位，不能先转Double再冒称有效位一致");
  }
  ++transaction_cases;
  details.push_back({"product_budget_double_wide", "transaction"});
}

void check_product_budget_work_boundaries() {
  const auto check_shape = []<class Scalar>(std::size_t n, std::size_t m) {
    const ProductBudgetData data(n, m);
    const auto value = data.direction<Scalar>();
    const auto off = product_budget_run(data, value, false), on = product_budget_run(data, value, true);
    product_budget_equations(data, value, off); product_budget_equations(data, value, on);
    const auto fixed = 2 + (m + 4095) / 4096 + (data.scaling.layout.inequality_rows.size() + 4095) / 4096;
    require(product_budget_equal(off.image, on.image) && off.callbacks == static_cast<int>(fixed + n + (data.values.size() + 4095) / 4096) &&
        on.callbacks == static_cast<int>(fixed + (n + data.values.size()) / 256),
        "真实模板256/4096边界须保固定行/互补检查，active不得叠加原global-k检查");
  };
  for (std::size_t n : {64U, 128U, 129U}) {
    check_shape.template operator()<double>(n, 1); check_shape.template operator()<long double>(n, 1);
  }
  for (std::size_t m : {254U, 255U, 256U, 257U, 4095U, 4096U, 4097U}) {
    check_shape.template operator()<double>(1, m); check_shape.template operator()<long double>(1, m);
  }
  ++transaction_cases;
  details.push_back({"product_budget_work_boundaries", "transaction"});
}

void check_product_budget_shared_columns() {
  const auto check_type = []<class Scalar>() {
    for (std::size_t n : {127U, 128U, 129U}) {
      const ProductBudgetData data(n, 1);
      const auto value = data.direction<Scalar>();
      const auto result = product_budget_run(data, value, true);
      product_budget_equations(data, value, result);
      require(result.callbacks == (n == 127 ? 4 : 5), "列头和entry的remaining必须跨列共享，128列恰计256项");
    }
    const ProductBudgetData empty_columns(256, 0);
    const auto value = empty_columns.direction<Scalar>();
    const auto off = product_budget_run(empty_columns, value, false), on = product_budget_run(empty_columns, value, true);
    product_budget_equations(empty_columns, value, off); product_budget_equations(empty_columns, value, on);
    require(off.callbacks == 258 && on.callbacks == 3 && product_budget_equal(off.image, on.image),
        "没有Gentry的列头仍须计工作，且不能依赖行或互补块非空");
  };
  check_type.template operator()<double>(); check_type.template operator()<long double>();
  ++mapping_cases;
  details.push_back({"product_budget_shared_columns", "mapping"});
}

void check_product_budget_all_cutoffs() {
  const auto check_type = []<class Scalar>() {
    for (const auto shape : {std::pair<std::size_t, std::size_t>{129, 1}, {1, 4097}}) {
      const ProductBudgetData data(shape.first, shape.second);
      const auto value = data.direction<Scalar>();
      for (bool enabled : {false, true}) {
        const auto complete = product_budget_run(data, value, enabled);
        product_budget_equations(data, value, complete);
        bool partial_seen = false;
        for (int cutoff = 1; cutoff <= complete.callbacks; ++cutoff) {
          ProductBudgetRun partial;
          bool stopped = false;
          coupled_state::Log log; log.enabled = enabled;
          coupled_state::Scope scope(&log);
          try {
            product_budget_call(data, value, partial.image, [&] {
              if (++partial.callbacks == cutoff) throw NormalInverseBudgetStop("五块乘积预算");
            });
            partial.completed = true;
          } catch (const NormalInverseBudgetStop& error) { stopped = std::string_view(error.what()) == "五块乘积预算"; }
          require(stopped && !partial.completed && partial.callbacks == cutoff, "模板每个实际callback截断须原异常逃出，不宣布完成");
          if (cutoff == 1) require(partial.image == std::vector<long double>({101, 202, 303}), "原入口预算在assign前，仍须保旧输出");
          else {
            require(partial.image.size() == data.scaling.dimension(), "assign后原算子允许完整维数的partial输出，不强制事务回滚");
            if (!product_budget_equal(partial.image, complete.image)) partial_seen = true;
          }
        }
        require(partial_seen, "必须实际观察已写入但未完成的乘积，不能只测试入口异常");
        const auto replay = product_budget_run(data, value, enabled);
        require(replay.callbacks == complete.callbacks && product_budget_equal(replay.image, complete.image), "中断不得把本次remaining残留到下次模板调用");
      }
    }
  };
  check_type.template operator()<double>(); check_type.template operator()<long double>();
  ++budget_cases;
  details.push_back({"product_budget_all_cutoffs", "budget"});
}

void check_product_budget_scope_snapshot() {
  const ProductBudgetData data(128, 1);
  const auto check_type = [&]<class Scalar>() {
    const auto value = data.direction<Scalar>();
    const auto reference = product_budget_run(data, value, false);
    for (bool enabled : {false, true}) {
      ProductBudgetRun run;
      coupled_state::Log entry; entry.enabled = enabled;
      coupled_state::Log changed; changed.enabled = !enabled;
      {
        coupled_state::Scope entry_scope(&entry);
        std::optional<coupled_state::Scope> persistent;
        product_budget_call(data, value, run.image, [&] { if (++run.callbacks == 1) persistent.emplace(&changed); });
        run.completed = true;
        require(coupled_state::enabled() == !enabled, "callback改变的Scope须持续至乘积返回，不能只在回调内临时生效");
        persistent.reset();
        require(coupled_state::enabled() == enabled, "解除持久Scope须恢复原入口上下文");
      }
      product_budget_equations(data, value, run);
      require(run.callbacks == (enabled ? 5 : 133) && product_budget_equal(run.image, reference.image),
          "首entrycheck前必须已绑定策略，后续TLS改变不能重选频率或改变物理结果");
    }
  };
  check_type.template operator()<double>(); check_type.template operator()<long double>();
  require(!coupled_state::enabled(), "模板Scope测试结束须恢复原关闭上下文");
  ++mapping_cases;
  details.push_back({"product_budget_scope_snapshot", "mapping"});
}

void check_product_budget_kernel_boundaries() {
  FmaLedgerFixture complete;
  int checks = 0, adapter_entry = 0, after_callback = 0;
  std::vector<int> first_product_callbacks;
  const auto answer = complete.solve([&] {
    ++checks;
    if (!complete.log.last) return;
    const auto& work = complete.log.last->work;
    const auto public_products = *complete.result.ipm_homogeneous_scaled_products;
    if (work.products_started == 1 && public_products == 0 && adapter_entry == 0) adapter_entry = checks;
    if (work.products_started == 1 && public_products == 1 && work.products_completed == 0) first_product_callbacks.push_back(checks);
    if (work.products_completed == 1 && work.true_checks_completed == 0 && after_callback == 0) after_callback = checks;
  });
  coupled_strict_log(complete.log);
  // n=0/m=1/I=1的实际模板有4个检查，随后才是adapter尾检查；不把其partial当完整真验。
  require(adapter_entry > 0 && first_product_callbacks.size() == 5 && after_callback > 0,
      "真实kernel基准须实际覆盖adapter入口、模板尾和callback完成后的真验前缀");
  for (int kind = 0; kind < 3; ++kind) {
    const int cutoff = kind == 0 ? adapter_entry : kind == 1 ? first_product_callbacks[3] : after_callback;
    FmaLedgerFixture partial;
    solver::HomogeneousDirection delivered; delivered.point.x = {101};
    int observed = 0; bool stopped = false;
    try {
      const auto solution = partial.solve([&] { if (++observed == cutoff) throw solver::BudgetExpired{}; });
      delivered = solver::v100_deliver_direction(partial.scaling, solution, partial.forced, partial.result);
    } catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && observed == cutoff && !delivered.v94_wide && delivered.point.x == std::vector<double>{101} &&
        partial.log.last && partial.log.directions_completed == 0 && *partial.result.ipm_homogeneous_scaled_inexact_returns == 0,
        "实际kernel乘积前缀中断不能交付宽方向、Point字段或公共inexact完成");
    const auto& work = partial.log.last->work;
    require(std::string_view(work.status) == "budget" && work.products_started == 1 &&
        work.products_completed == (kind == 2 ? 1 : 0) && work.true_checks_started == 1 && work.true_checks_completed == 0 &&
        *partial.result.ipm_homogeneous_scaled_products == (kind == 0 ? 0 : 1) &&
        !work.initial_measure && !work.final_measure && work.initial_forcing && !work.initial_forcing->defect_norms,
        "private/public产品可差1且callback完成不等于truecheck完成，半份E5/measure不得发布");
  }
  const auto delivered = solver::v100_deliver_direction(complete.scaling, answer, complete.forced, complete.result);
  int update_checks = 0;
  const auto updated = solver::v94_update_point(complete.cone, complete.point, 1, 1, delivered, 0.125, [&] { ++update_checks; });
  require(update_checks > 1 && updated.point.s[0] > 0 && updated.point.z[0] > 0, "完整真验后须真实通过LD Point更新入口");
  solver::V94UpdatedPoint pending{complete.point, 1, 1};
  int observed = 0; bool stopped = false;
  try { pending = solver::v94_update_point(complete.cone, complete.point, 1, 1, delivered, 0.125,
      [&] { if (++observed == update_checks) throw solver::BudgetExpired{}; }); }
  catch (const solver::BudgetExpired&) { stopped = true; }
  require(stopped && pending.point.x == complete.point.x && pending.point.s == complete.point.s &&
      pending.point.z == complete.point.z && pending.tau == 1 && pending.kappa == 1,
      "即使kernel真验已完成，原Point末预算失败仍不能提交候选点");
  const ProductBudgetData empty(0, 0), mixed(2, 3, true);
  const auto empty_value = empty.direction<long double>();
  const auto empty_run = product_budget_run(empty, empty_value, true);
  product_budget_equations(empty, empty_value, empty_run);
  require(empty_run.callbacks == 2, "空五块只保原首末检查，tau/kappa方程仍须正确");
  const auto value = mixed.direction<long double>();
  std::vector<long double> no_callback;
  {
    coupled_state::Log log; log.enabled = true; coupled_state::Scope scope(&log);
    product_budget_call(mixed, value, no_callback, {});
  }
  require(product_budget_equal(no_callback, product_budget_run(mixed, value, false).image), "空check仍保原数学，不据输出虚称内部tick执行");
  auto bad_value = value; bad_value.pop_back();
  std::vector<long double> invalid_output{101, 202};
  int invalid_checks = 0; bool invalid = false;
  try { product_budget_call(mixed, bad_value, invalid_output, [&] { ++invalid_checks; }); }
  catch (const std::invalid_argument&) { invalid = true; }
  require(invalid && invalid_checks == 1 && invalid_output == std::vector<long double>({101, 202}),
      "维数拒绝仍在原入口check之后和assign之前，不能混同预算后的partial语义");
  ++budget_cases;
  details.push_back({"product_budget_kernel_boundaries", "budget"});
}

// 以下六组只覆盖冻结的通用交叉衔接；手工构造的近点明确不是HSD更新。
struct CrossoverMode {
  explicit CrossoverMode(bool value) : previous(crossover_state::enabled_override) {
    crossover_state::enabled_override = value;
  }
  ~CrossoverMode() { crossover_state::enabled_override = previous; }
  std::optional<bool> previous;
};

solver::Model crossover_equality() {
  solver::Model model;
  model.name = "v117_真实HSD组件";
  model.costs = {1, 0}; model.column_lower = {0, 0}; model.column_upper = {1, 1};
  model.row_lower = {1}; model.row_upper = {1};
  model.column_start = {0, 1, 2}; model.row_index = {0, 0}; model.values = {1, 1};
  model.kinds.assign(2, solver::VariableKind::continuous);
  return model;
}

void crossover_native_once(const crossover_state::Log& log) {
  require(log.attempted && log.started == 1 && log.completed == 1 && log.last,
      "一次交叉须保留真实开始、完成和当次事件");
  const auto& stats = log.last->stats;
  require(stats.load_started == 1 && stats.load_completed == 1 && stats.crossover_started == 1 &&
      stats.crossover_completed == 1 && stats.get_started == 1 && stats.get_completed == 1 && stats.native_info &&
      stats.native_info->status_ipm == IPX_STATUS_not_run && stats.native_info->iter == 0 &&
      stats.native_info->kktiter1 == 0 && stats.native_info->kktiter2 == 0 && stats.native_info->updates_ipm == 0,
      "成功衔接必须实际Load/crossover/Get；IPM未运行计数只是禁完整Solve的源码审计旁证");
  require(stats.native_threads == 1 && std::isfinite(stats.native_time_limit) && stats.native_time_limit > 0 &&
      stats.total_seconds >= stats.prepare_seconds && stats.total_seconds >= stats.native_scope_seconds,
      "交叉须保持单线程、显式剩余时限和包含准备与清理的总成本");
}

void check_crossover_hsd_actual_handoff() {
  const auto model = crossover_equality();
  auto options = policy(); options.ipm.scaling_passes = 0; options.ipm.eliminate_fixed_variables = false;
  solver::SolveResult ordinary;
  {
    crossover_state::Scope isolated(nullptr);
    ordinary = real_coupled_core(model, options);
  }
  original_quality(model, ordinary, 0);
  crossover_state::Log log; log.enabled = true;
  solver::SolveResult result;
  {
    crossover_state::Scope scope(&log);
    result = real_coupled_core(model, options);
  }
  original_quality(model, result, 0);
  crossover_native_once(log);
  require(log.local_accepted && log.adopted && log.last->status == crossover::Status::accepted &&
      log.last->iteration >= 1 && result.ipm_iterations == log.last->iteration &&
      result.ipm_factorizations && *result.ipm_factorizations > 0 && result.ipm_homogeneous_direction_solves &&
      *result.ipm_homogeneous_direction_solves >= 2 && result.ipm_homogeneous_tau && *result.ipm_homogeneous_tau > 0 &&
      result.ipm_homogeneous_kappa && *result.ipm_homogeneous_kappa > 0,
      "实际HSD须先完成真实预测校正及Point更新，再从同次迭代成功交叉，不以合成iteration标签代替");
  require(!log.last->before_quality.numerically_optimal && log.last->before_quality.dual_feasible &&
      log.last->after_quality && log.last->after_quality->numerically_optimal &&
      result.verification_seconds >= log.last->stats.verification_seconds,
      "交叉前须真实未过局部最优验收，交叉后须过原精度且原验成本已付费");
  ++transaction_cases;
  details.push_back({"crossover_hsd_actual_handoff", "transaction"});
}

void check_crossover_gate_gap_and_domain() {
  auto options = policy();
  auto model = scalar(1, 0, 1); row(model, 0, 0, 0);
  const std::vector<double> far{0.5}, exact{0}, dual{0};
  const auto checked_far = solver::verify_lp_solution(model, far, dual);
  const auto checked_exact = solver::verify_lp_solution(model, exact, dual);
  require(checked_far.primal.feasible && checked_far.optimality.dual_feasible &&
      !checked_far.optimality.gap_satisfied && quality_ok(checked_exact),
      "门测试须使用真实原检查器，不能人工拼质量标签");
  crossover_state::Log log; log.enabled = true;
  int checks = 0;
  {
    crossover_state::Scope scope(&log);
    require(!crossover_state::try_current(model, far, dual, checked_far, options, solver::Clock::now(), 0,
        [&] { ++checks; }) && checks == 0, "零次更新不能启动门扫描或交叉");
    auto mehrotra = options; mehrotra.ipm.method = "mehrotra";
    require(!crossover_state::try_current(model, far, dual, checked_far, mehrotra, solver::Clock::now(), 1,
        [&] { ++checks; }) && checks == 0, "仅HSD路径可进入冻结交叉门");
    require(!crossover_state::try_current(model, exact, dual, checked_exact, options, solver::Clock::now(), 1,
        [&] { ++checks; }) && checks == 0, "当前已最优点须沿既有oracle，不再付交叉成本");
    require(!crossover_state::try_current(model, far, dual, checked_far, options, solver::Clock::now(), 1,
        [&] { ++checks; }) && checks > 0 && log.norms_ready && !log.attempted && log.started == 0,
        "原始和对偶可行但gap大的点必须被真实归一化门拒绝");
    require(log.bounds_inf == 1 && log.cost_inf == 1 && log.norm_items == 2,
        "门只付一次列和行范数扫描，不能暗加矩阵乘积或忽略有限界");
  }
  auto free_row = scalar(0, 0, 1); row(free_row, 0, -infinity, infinity);
  const std::vector<double> x{0.5}, illegal_y{1};
  const auto invalid_domain = solver::verify_lp_solution(free_row, x, illegal_y);
  require(invalid_domain.primal.feasible && invalid_domain.optimality.valid_input &&
      !invalid_domain.optimality.dual_feasible && invalid_domain.optimality.max_dual_residual == 0 &&
      invalid_domain.optimality.relative_objective_gap == 0,
      "自由行非零乘子须真实展示max_dual_residual为0仍不满足对偶定义域");
  crossover_state::Log forbidden; forbidden.enabled = true;
  checks = 0;
  {
    crossover_state::Scope scope(&forbidden);
    require(!crossover_state::try_current(free_row, x, illegal_y, invalid_domain, options, solver::Clock::now(), 1,
        [&] { ++checks; }) && checks == 0 && !forbidden.norms_ready && forbidden.started == 0,
        "必须先保对偶符号域布尔门，不能把零proxy当成完整对偶可行性");
  }
  ++exception_cases;
  details.push_back({"crossover_gate_gap_and_domain", "exception"});
}

void check_crossover_remaining_deadline() {
  const auto options = policy();
  const auto expired_start = solver::Clock::now() - std::chrono::seconds(6);
  const auto expired = crossover_state::make_deadline(expired_start, options.time_limit);
  require(expired && !expired->saturated && expired->value == expired_start + std::chrono::seconds(5) &&
      expired->value < solver::Clock::now(), "交叉绝对deadline须沿原solve起点，不能重授五秒");
  const auto huge = crossover_state::make_deadline(solver::Clock::now(), std::numeric_limits<double>::max());
  require(huge && huge->saturated && huge->value == solver::Clock::time_point::max() &&
      !crossover_state::make_deadline(expired_start, -1) &&
      !crossover_state::make_deadline(expired_start, std::numeric_limits<double>::quiet_NaN()),
      "通用巨大时限须饱和安全转换，负数与NaN不能进入时钟算术");
  auto model = scalar(1, 0, 1); row(model, 0, 0, 0);
  const std::vector<double> x{1e-5}, y{0};
  const auto checked = solver::verify_lp_solution(model, x, y);
  crossover_state::Log expired_log; expired_log.enabled = true;
  {
    crossover_state::Scope scope(&expired_log);
    // 外层回调只计次；adapter必须仍由真实steady_clock拒绝已过期的同一deadline。
    int checks = 0;
    const auto stopped = crossover_state::try_current(model, x, y, checked, options, expired_start, 1, [&] { ++checks; });
    require(stopped && stopped->status == crossover::Status::time_limit && checks > 0 &&
        stopped->x.empty() && stopped->y.empty() && stopped->stats.load_started == 0 &&
        expired_log.started == 1 && expired_log.completed == 1 && !expired_log.adopted,
        "真实截止时间过期须在Load前失败并保完整前缀，不靠模拟native hook");
  }
  constexpr std::size_t count = 4100;
  solver::Model large;
  large.costs.assign(count, 0); large.costs[0] = 1;
  large.column_lower.assign(count, 0); large.column_upper.assign(count, 1);
  large.kinds.assign(count, solver::VariableKind::continuous); large.column_start.assign(count + 1, 0);
  large.row_lower = {0}; large.row_upper = {0};
  std::vector<double> large_x(count, 0); large_x[0] = 1e-5;
  const auto large_checked = solver::verify_lp_solution(large, large_x, y);
  require(large_checked.primal.valid_input && large_checked.optimality.dual_feasible &&
      !large_checked.optimality.numerically_optimal, "预算夹具须先真实通过门输入合同");
  for (bool during_scan : {false, true}) {
    crossover_state::Log partial; partial.enabled = true;
    bool stopped = false;
    {
      crossover_state::Scope scope(&partial);
      try {
        static_cast<void>(crossover_state::try_current(large, large_x, y, large_checked, options,
            solver::Clock::now(), 1, [&] {
              if (!during_scan || partial.norm_items >= 4096) throw solver::BudgetExpired{};
            }));
      } catch (const solver::BudgetExpired&) { stopped = true; }
    }
    require(stopped && !partial.norms_ready && !partial.attempted && partial.started == 0 && !partial.last &&
        (during_scan ? partial.norm_items >= 4096 : partial.norm_items == 0),
        "原预算异常须在入口或范数扫描真实前缀原样传播，不发布半缓存或启动native");
  }
  crossover_state::Log completed; completed.enabled = true;
  std::optional<crossover::Result> delivered;
  bool stopped_after_native = false;
  double paid_verification = 0;
  {
    crossover_state::Scope scope(&completed);
    solver::CrossoverVerificationCollect collect(paid_verification);
    try {
      delivered = crossover_state::try_current(model, x, y, checked, options, solver::Clock::now(), 1, [&] {
        if (completed.completed == 1) throw solver::BudgetExpired{};
      });
    } catch (const solver::BudgetExpired&) { stopped_after_native = true; }
  }
  crossover_native_once(completed);
  require(stopped_after_native && !delivered && completed.local_accepted && !completed.adopted &&
      completed.last->after_quality && completed.last->after_quality->numerically_optimal &&
      paid_verification == completed.last->stats.verification_seconds,
      "真实交叉与原验完成后，原预算末检查仍可拒绝交付；保已付native账本但不能发布候选");
  {
    crossover_state::Scope scope(&completed);
    solver::CrossoverVerificationCollect repeated(paid_verification);
  }
  require(paid_verification == completed.last->stats.verification_seconds,
      "后继assess复用已消费事件时不得重复归入同一笔验算耗时");
  ++budget_cases;
  details.push_back({"crossover_remaining_deadline", "budget"});
}

void check_crossover_single_attempt_scope() {
  const auto model = scalar(1, 0, 1); const auto options = policy();
  const std::vector<double> x{1e-5}, y;
  const auto checked = solver::verify_lp_solution(model, x, y);
  crossover_state::Log outer; outer.enabled = true;
  auto* previous = crossover_state::current();
  {
    crossover_state::Scope scope(&outer);
    const auto failed = crossover_state::try_current(model, x, y, checked, options, solver::Clock::now(), 1, [] {});
    require(failed && failed->status == crossover::Status::unsupported_empty && failed->x.empty() && failed->y.empty() &&
        failed->stats.load_started == 0 && outer.attempted && outer.started == 1 && outer.completed == 1,
        "真实空行模型须明确不支持；失败仍消费唯一一次交叉尝试");
    int checks = 0;
    require(!crossover_state::try_current(model, x, y, checked, options, solver::Clock::now(), 2,
        [&] { ++checks; }) && checks == 0 && outer.started == 1, "失败后不得在下一次评估偷偷重试");
    {
      crossover_state::Scope disabled(nullptr);
      require(!crossover_state::enabled() &&
          !crossover_state::try_current(model, x, y, checked, options, solver::Clock::now(), 3,
              [&] { ++checks; }) && checks == 0, "关闭Scope不得做范数扫描、预算回调或native工作");
    }
    require(crossover_state::current() == &outer, "关闭Scope必须恢复原owner");
    crossover_state::Log inner; inner.enabled = true;
    bool unwound = false;
    try {
      crossover_state::Scope nested(&inner);
      const auto failure = crossover_state::try_current(model, x, y, checked, options, solver::Clock::now(), 1, [] {});
      require(failure && inner.started == 1 && outer.started == 1, "不同solve owner必须分别持有一次尝试预算");
      throw 117;
    } catch (int marker) { require(marker == 117, "仅捕获本组作用域退出标记"); unwound = true; }
    require(unwound && crossover_state::current() == &outer && outer.started == 1, "异常退出须恢复此前交叉owner");
    const auto unchanged = coupled();
    solver::SolveResult baseline;
    {
      CrossoverMode disabled_mode(false);
      crossover_state::Scope disabled(nullptr);
      baseline = real_core(unchanged, options);
    }
    {
      CrossoverMode disabled(false);
      same_non_timing(real_wrapper(unchanged, options, false), baseline);
    }
    require(crossover_state::current() == &outer && outer.started == 1,
        "关闭公开求解入口须隔离外层交叉owner且恢复之，不改变原非计时报告");
  }
  require(crossover_state::current() == previous, "最后一个交叉Scope须恢复原TLS");
  ++transaction_cases;
  details.push_back({"crossover_single_attempt_scope", "transaction"});
}

void check_crossover_original_postsolve() {
  const auto original = coupled(true); const auto options = policy();
  candidate::Stats stats;
  crossover_state::Log log; log.enabled = true;
  const auto start = solver::Clock::now();
  solver::SolveResult result;
  {
    crossover_state::Scope cross_scope(&log);
    result = candidate::control(original, options, start, stats,
        [&](const solver::Model& reduced, const solver::SolveOptions& child_options, solver::Clock::time_point child_start) {
          ++synthetic_core_calls;
          auto child = candidate::make_shell(reduced); child.ipm_iterations = 7;
          const auto cone = identity_assess_cone();
          const auto point = assess_point(4.0 / 3.0 - 1e-5);
          const auto before = point;
          const std::vector<double> dual{-1.0 / 3.0, -1.0 / 3.0};
          const auto raw = solver::verify_lp_solution(reduced, point.x, dual);
          require(raw.primal.feasible && raw.optimality.dual_feasible && !raw.optimality.numerically_optimal,
              "此组是明确合成的非最优近点，不能拿已知最优点冒充交叉输入");
          require(solver::assess(cone, point, child_options, child_start, child) && child.status == "optimal" &&
              point.x == before.x && point.z == before.z && point.s == before.s,
              "实际assess须先交叉和原oracle接受；不得修改持久HSD Point");
          return child;
        });
  }
  original_quality(original, result, 88.0 / 3.0);
  crossover_native_once(log);
  require(log.adopted && std::string_view(log.oracle_decision) == "accept" && stats.oracle.started == 1 &&
      stats.oracle.completed == 1 && stats.oracle.accepted == 1 && stats.postsolve.started == 1 && stats.postsolve.completed == 1 &&
      stats.oracle.selected_event == 0 && stats.child.started == 1 && stats.child.completed == 1 &&
      stats.native.crossover_iterations && *stats.native.crossover_iterations <= 0 && stats.native.basis_valid == false,
      "独立IPX账本与无优化预处理账本必须分开；成功不得再postsolve一次或输出约化坐标");
  ++transaction_cases;
  details.push_back({"crossover_original_postsolve", "transaction"});
}

void check_crossover_gate_current_snapshot() {
  NativeOracleFixture fixture;
  const auto cone = identity_assess_cone();
  const std::vector<double> exact_x{4.0 / 3.0, 4.0 / 3.0}, dual{-1.0 / 3.0, -1.0 / 3.0};
  auto incumbent = candidate::make_shell(fixture.reduced);
  auto exact = solver::verify_lp_solution(fixture.reduced, exact_x, dual);
  require(candidate::commit_original_quality(incumbent, exact_x, dual, std::move(exact), fixture.options),
      "快照夹具先真实建立最优旧incumbent");
  crossover_state::Log log; log.enabled = true;
  {
    crossover_state::Scope cross_scope(&log);
    candidate::ScopedOracle oracle_scope(&*fixture.oracle);
    // 迭代号只是assess的合成调用标签；真实更新由actual_handoff组另行证明。
    incumbent.ipm_iterations = 9;
    require(!solver::assess(cone, assess_point(0), fixture.options, fixture.start, incumbent) &&
        !log.attempted && fixture.stats.oracle.started == 0 && incumbent.solution == exact_x,
        "旧最优incumbent不能让当前远点触发交叉或原oracle");
    const auto near = assess_point(4.0 / 3.0 - 1e-5);
    const auto checked = solver::verify_lp_solution(fixture.reduced, near.x, dual);
    incumbent.ipm_iterations = 10;
    require(solver::assess(cone, near, fixture.options, fixture.start, incumbent),
        "当前合成近点须按自己的质量触发真实交叉，不因旧incumbent已最优而跳过");
    crossover_native_once(log);
    require(log.last->iteration == 10 && !log.last->before_quality.numerically_optimal &&
        log.last->before_quality.objective == checked.primal.objective &&
        log.last->before_quality.objective_gap == checked.optimality.objective_gap &&
        fixture.stats.oracle.started == 1 && fixture.stats.oracle.accepted == 1,
        "交叉事件必须记录同一次assess未move前的真实目标与gap，不能引用旧candidate质量");
  }
  fixture.no_native_iterations();
  ++mapping_cases;
  details.push_back({"crossover_gate_current_snapshot", "mapping"});
}

// v118 行对偶域投影组件新增块开始
bool v118_same_bits(std::span<const double> a, std::span<const double> b) {
  if (a.size() != b.size()) return false;
  for (std::size_t k = 0; k < a.size(); ++k)
    if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k])) return false;
  return true;
}

crossover::Result v118_raw(const ProjectionInput& input, const solver::SolveOptions& options,
    solver::Clock::time_point deadline, bool enabled = true) {
  crossover::Result result;
  result.stats.projection.enabled = enabled;
  crossover::verify_candidate_once(input.model, input.x, input.y, result,
      options.tolerances, options.pdlp.optimality, [&] { crossover::detail::check(deadline); });
  require(result.verification && result.stats.verification_started == 1 && result.stats.verification_completed == 1 &&
      !result.raw_verification, "先通过真实首验取得raw质量，不能手工拼未通过标签");
  return result;
}

void v118_no_native(const crossover::Result& result) {
  require(result.stats.load_started == 0 && result.stats.load_completed == 0 &&
      result.stats.crossover_started == 0 && result.stats.crossover_completed == 0 &&
      result.stats.get_started == 0 && result.stats.get_completed == 0 && !result.stats.native_info &&
      result.x.empty() && result.y.empty(),
      "生产投影helper不启动native、不擅自发布候选；是否接纳仍由调用者完整终验决定");
}

void v118_preserved_raw(const crossover::Result& result, const solver::LpVerification& raw) {
  require(result.raw_verification && same_verification(result.raw_verification->primal, raw.primal) &&
      same_optimality(result.raw_verification->optimality, raw.optimality),
      "第二验前必须保存第一份完整真实报告，不能用投影质量覆盖raw拒绝原因");
}

void check_crossover_dual_projection_unchanged() {
  const auto options = policy();
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  const std::vector<ProjectionInput> inputs{
    projection_input(ProjectionKind::raw_pass),
    projection_input(ProjectionKind::unchanged),
    {scalar(1, 0, 1), {0.5}, {}}};
  for (const auto& input : inputs) {
    const auto x = input.x, y = input.y;
    auto result = v118_raw(input, options, deadline);
    const bool already_good = quality_ok(*result.verification);
    const auto raw = *result.verification;
    int callbacks = 0;
    const auto projected = crossover::project_rejected_row_dual(input.model, input.x, input.y, result,
        options.tolerances, options.pdlp.optimality, [&] { ++callbacks; crossover::detail::check(deadline); });
    const auto& p = result.stats.projection;
    require(!projected && !result.raw_verification && result.stats.verification_started == 1 &&
        result.stats.verification_completed == 1 && p.project_started == 0 && p.project_completed == 0 &&
        !p.project_rc && p.changed_rows == 0 && p.eligible_rows == 0 && p.max_abs_change == 0 &&
        same_verification(result.verification->primal, raw.primal) && same_optimality(result.verification->optimality, raw.optimality) &&
        v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
        "raw已通过或合法域无改变时不能复制/投影/二验，更不能改x或原raw_y");
    require(already_good ? callbacks == 0 && p.scan_started == 0 && p.scan_completed == 0 :
        callbacks > 0 && p.scan_started == 1 && p.scan_completed == 1 && p.scan_items == y.size(),
        "raw全过必须完全跳过扫描；域无改变的拒绝点只付真实扫描成本");
    v118_no_native(result);
  }
  const auto input = projection_input(ProjectionKind::dual_lower);
  auto disabled = v118_raw(input, options, deadline, false);
  int callbacks = 0;
  require(!crossover::project_rejected_row_dual(input.model, input.x, input.y, disabled,
      options.tolerances, options.pdlp.optimality, [&] { ++callbacks; }) && callbacks == 0 &&
      disabled.stats.projection.scan_started == 0 && disabled.stats.verification_completed == 1,
      "新增显式开关默认关闭时不扫描、不二验，保留旧直接adapter合同");
  ++mapping_cases;
  details.push_back({"crossover_dual_projection_unchanged", "mapping"});
}

void check_crossover_dual_projection_accept() {
  const auto options = policy();
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  for (int kind : {0, 1, 2}) {
    ProjectionInput input{scalar(0, 0, 1), {-0.0}, {kind == 0 ? -1e-300 : kind == 1 ? 1e-300 : 1.0}};
    row(input.model, 0, kind == 0 ? 0 : -infinity, kind == 1 ? 0 : infinity);
    const auto x = input.x, y = input.y;
    auto result = v118_raw(input, options, deadline);
    const auto raw = *result.verification;
    require(raw.primal.feasible && !raw.optimality.dual_feasible && !quality_ok(raw),
        "成功反例先由真实checker拒绝非法行对偶符号，不伪造native误差");
    const auto projected = crossover::project_rejected_row_dual(input.model, input.x, input.y, result,
        options.tolerances, options.pdlp.optimality, [&] { crossover::detail::check(deadline); });
    require(projected && projected->size() == 1 && (*projected)[0] == 0 && projected->data() != input.y.data() &&
        result.verification && quality_ok(*result.verification) &&
        quality_ok(solver::verify_lp_solution(input.model, input.x, *projected, options.tolerances, options.pdlp.optimality)) &&
        v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
        "合法投影须返回拥有的y且过独立全原验，x含负零位和原raw_y保持不变");
    const auto& p = result.stats.projection;
    require(p.enabled && p.scan_started == 1 && p.scan_completed == 1 && p.scan_items == 1 && p.eligible_rows == 1 &&
        p.project_started == 1 && p.project_completed == 1 && p.project_rc == true && p.changed_rows == 1 &&
        p.max_abs_change == (kind == 2 ? 1.0 : 1e-300) &&
        result.stats.verification_started == 2 && result.stats.verification_completed == 2,
        "只有真实域变化才能投影并增加第二次完整verify，扫描/比较/返回账本须守恒");
    v118_preserved_raw(result, raw);
    v118_no_native(result);
  }
  // 真实HSD贯通只证明新字段与一次native衔接，不要求native偶然产生非法符号。
  auto hsd_options = options; hsd_options.ipm.scaling_passes = 0; hsd_options.ipm.eliminate_fixed_variables = false;
  const auto hsd_model = crossover_equality();
  crossover_state::Log log; log.enabled = true;
  solver::SolveResult hsd_result;
  {
    CrossoverMode mode(true);
    crossover_state::Scope scope(&log);
    hsd_result = real_coupled_core(hsd_model, hsd_options);
  }
  original_quality(hsd_model, hsd_result, 0);
  crossover_native_once(log);
  require(log.adopted && log.local_accepted && log.last->iteration >= 1 && log.last->unprojected_quality &&
      log.last->after_quality && log.last->after_quality->numerically_optimal && log.last->stats.projection.enabled &&
      hsd_result.backend_version == "ipm-ipx-crossover-cpu-0.4.0" &&
      log.last->stats.verification_started == log.last->stats.verification_completed &&
      (log.last->stats.verification_completed == 1 || log.last->stats.verification_completed == 2),
      "真实HSD须进入schema2适配器并保原始native质量、最后完整质量和一次native账本");
  require(log.last->stats.projection.changed_rows == 0 ? log.last->stats.verification_completed == 1 :
      log.last->stats.verification_completed == 2,
      "贯通计数按实际域变化判定，不能把普通raw成功冒称native符号投影分支已执行");
  require(hsd_result.verification_seconds >= log.last->stats.verification_seconds,
      "两验累计成本只能归入真实公共验算耗时一次");
  ++transaction_cases;
  details.push_back({"crossover_dual_projection_accept", "transaction"});
}

void check_crossover_dual_projection_reject() {
  const auto options = policy();
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  for (const auto kind : {ProjectionKind::both, ProjectionKind::breaks_gap}) {
    const auto input = projection_input(kind);
    const auto x = input.x, y = input.y;
    auto result = v118_raw(input, options, deadline);
    const auto raw = *result.verification;
    require(!quality_ok(raw) && !raw.optimality.dual_feasible, "拒绝反例必须有可投影的真实非法域乘子");
    const auto projected = crossover::project_rejected_row_dual(input.model, input.x, input.y, result,
        options.tolerances, options.pdlp.optimality, [&] { crossover::detail::check(deadline); });
    require(projected && (*projected)[0] == 0 && result.verification && !quality_ok(*result.verification) &&
        result.verification->optimality.dual_feasible && result.stats.verification_started == 2 &&
        result.stats.verification_completed == 2 && result.stats.projection.changed_rows == 1 &&
        v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
        "helper交出投影向量不等于验收成功；修复对偶域不能放宽原始或gap终验");
    const auto independently_checked = solver::verify_lp_solution(input.model, input.x, *projected,
        options.tolerances, options.pdlp.optimality);
    require(same_verification(result.verification->primal, independently_checked.primal) &&
        same_optimality(result.verification->optimality, independently_checked.optimality),
        "拒绝必须来自原Model重新计算的完整质量，不能只用投影成功标志决定");
    if (kind == ProjectionKind::both)
      require(!result.verification->primal.feasible && input.x[0] == -2e-7,
          "只投y必须保留原x违反变量界的反例，不能复用同时截断x的旧投影路径");
    else require(raw.optimality.gap_satisfied && !result.verification->optimality.gap_satisfied &&
        result.verification->optimality.objective_gap == 0.5,
        "非法符号恰好补成raw零gap的反例，在域投影后必须因真实gap恶化拒绝");
    v118_preserved_raw(result, raw);
    v118_no_native(result);
  }
  ++transaction_cases;
  details.push_back({"crossover_dual_projection_reject", "transaction"});
}

void check_crossover_dual_projection_budget() {
  const auto options = policy();
  ProjectionInput input{scalar(0, 0, 1), {-0.0}, std::vector<double>(4100, 0)};
  input.model.row_lower.assign(input.y.size(), 0);
  input.model.row_upper.assign(input.y.size(), infinity);
  input.y.back() = -1e-300;
  const auto x = input.x, y = input.y;
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  auto complete = v118_raw(input, options, deadline);
  std::size_t callbacks = 0;
  const auto accepted_y = crossover::project_rejected_row_dual(input.model, input.x, input.y, complete,
      options.tolerances, options.pdlp.optimality, [&] { ++callbacks; crossover::detail::check(deadline); });
  require(accepted_y && complete.verification && quality_ok(*complete.verification) && callbacks > 2 &&
      complete.stats.projection.scan_items == input.y.size() && complete.stats.projection.changed_rows == 1 &&
      complete.stats.verification_completed == 2,
      "长向量基准必须真正跨4096扫描并完成一次合法域变化和完整第二验");
  bool saw_scan_partial = false, saw_project_before_compare = false, saw_reset_before_second = false, saw_complete_unpublished = false;
  for (std::size_t cutoff = 0; cutoff < callbacks; ++cutoff) {
    auto partial = v118_raw(input, options, deadline);
    const auto raw = *partial.verification;
    std::optional<std::vector<double>> delivered;
    std::size_t reached = 0;
    bool stopped = false;
    try {
      delivered = crossover::project_rejected_row_dual(input.model, input.x, input.y, partial,
          options.tolerances, options.pdlp.optimality, [&] {
            crossover::detail::check(deadline);
            if (reached++ == cutoff) throw solver::BudgetExpired{};
          });
    } catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && !delivered && reached == cutoff + 1 && v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
        "所有真实callback截止点须保原预算异常和原x/raw_y，不能交付半个投影向量");
    const auto& p = partial.stats.projection;
    require(p.scan_started <= 1 && p.scan_completed <= p.scan_started && p.project_started <= 1 &&
        p.project_completed <= p.project_started && p.changed_rows <= 1 &&
        partial.stats.verification_started >= 1 && partial.stats.verification_started <= 2 &&
        partial.stats.verification_completed <= partial.stats.verification_started,
        "中断须保已经支付的扫描/库调用/二验前缀，不要求所有计数同时完成");
    if (partial.raw_verification) v118_preserved_raw(partial, raw);
    else require(partial.verification && same_verification(partial.verification->primal, raw.primal) &&
        same_optimality(partial.verification->optimality, raw.optimality),
        "尚未开始二验时verification必须仍是第一份完整报告");
    if (partial.stats.verification_completed == 2)
      require(partial.verification && quality_ok(*partial.verification),
          "第二验已完成而末预算失败时须保最后完整质量，不能抹掉真实已完成工作");
    saw_scan_partial = saw_scan_partial || (p.scan_started == 1 && p.scan_completed == 0 && p.scan_items >= 4096);
    saw_project_before_compare = saw_project_before_compare || (p.project_completed == 1 && p.changed_rows == 0);
    saw_reset_before_second = saw_reset_before_second || (partial.raw_verification && !partial.verification &&
        partial.stats.verification_started == 1 && partial.stats.verification_completed == 1);
    saw_complete_unpublished = saw_complete_unpublished || partial.stats.verification_completed == 2;
    v118_no_native(partial);
  }
  require(saw_scan_partial && saw_project_before_compare && saw_reset_before_second && saw_complete_unpublished,
      "实际截止点须覆盖跨4096扫描、projector已返、首报告move/reset及二验已完成但未交付四类前缀");
  auto expired = v118_raw(input, options, deadline);
  const auto past = solver::Clock::now() - std::chrono::seconds(1);
  bool expired_at_entry = false;
  try {
    static_cast<void>(crossover::project_rejected_row_dual(input.model, input.x, input.y, expired,
        options.tolerances, options.pdlp.optimality, [&] { crossover::detail::check(past); }));
  } catch (const crossover::Failure& failure) { expired_at_entry = failure.status == crossover::Status::time_limit; }
  require(expired_at_entry && expired.stats.projection.scan_started == 0 && expired.stats.verification_completed == 1 &&
      !expired.raw_verification && v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
      "生产同一绝对deadline已过期须在扫描前拒绝，不重授时间或依赖主机忙等");
  v118_no_native(expired);
  ++budget_cases;
  details.push_back({"crossover_dual_projection_budget", "budget"});
}
// v118 行对偶域投影组件新增块结束
// v120 自主暖基衔接组件新增块开始
namespace v120_warm = solver::detail::warm_cleanup;
struct V120WarmInput {
  solver::Model model;
  std::vector<double> x, y;
  crossover::Prepared prepared;
  std::vector<ipx::Int> variable_status, row_status;
  std::vector<double> seed;
};

V120WarmInput v120_input(int kind = 0) {
  V120WarmInput input;
  input.model.name = "v120_明确合成暖基输入";
  if (kind == 0) {
    input.model = scalar(-1, 0, 2); row(input.model, 1, -infinity, 1);
    input.x = {0}; input.y = {1e-300};
    input.variable_status = {-1}; input.row_status = {0}; input.seed = {0};
  } else if (kind == 1) {
    input.model.costs = {1, 1}; input.model.column_lower = {0, 0}; input.model.column_upper = {infinity, infinity};
    input.model.row_lower = {1}; input.model.row_upper = {1}; input.model.column_start = {0, 1, 2};
    input.model.row_index = {0, 0}; input.model.values = {1, -1};
    input.x = {1, 0}; input.y = {0};
    input.variable_status = {-1, 0}; input.row_status = {-1}; input.seed = {0, 0};
  } else if (kind == 2) {
    input.model.costs = {0, 0}; input.model.column_lower = {0, 0}; input.model.column_upper = {infinity, infinity};
    input.model.row_lower = {1, 1}; input.model.row_upper = {1, 1}; input.model.column_start = {0, 2, 4};
    input.model.row_index = {0, 1, 0, 1}; input.model.values = {1, 1, 2, 2};
    input.x = {1, 0}; input.y = {1, 0};
    input.variable_status = {0, 0}; input.row_status = {-1, -1}; input.seed = {1, 0};
  } else {
    input.model = scalar(-1, 0, infinity); row(input.model, 0, 0, 0);
    input.x = {0}; input.y = {0};
    input.variable_status = {-1}; input.row_status = {0}; input.seed = {0};
  }
  input.model.kinds.assign(input.model.costs.size(), solver::VariableKind::continuous);
  input.prepared = crossover::prepare(input.model, input.x, input.y, solver::Clock::now() + std::chrono::seconds(5));
  return input;
}

crossover::Result v120_rejected(const V120WarmInput& input, const solver::SolveOptions& options,
    solver::Clock::time_point deadline) {
  auto result = v118_raw(ProjectionInput{input.model, input.x, input.y}, options, deadline);
  if (input.model.costs.size() == 1 && input.model.row_upper[0] == 1) {
    const auto y = crossover::project_rejected_row_dual(input.model, input.x, input.y, result,
        options.tolerances, options.pdlp.optimality, [&] { crossover::detail::check(deadline); });
    require(y && result.raw_verification && result.verification && !quality_ok(*result.verification),
        "合成成功夹具须真实经历raw和合法域投影两验均拒绝，而非手工制造native失败");
  }
  require(result.verification && !quality_ok(*result.verification), "合成暖基衔接必须先有真实未过原验的点");
  result.status = crossover::Status::rejected;
  result.message = "明确合成基与真实checker拒绝报告；未调用native";
  result.basis_export.emplace(crossover::BasisExport{input.prepared, input.variable_status, input.row_status, input.seed});
  // 此payload是组件显式构造，不将任何native调用/导出统计伪造为1。
  return result;
}

crossover_state::Event v120_event(const crossover::Result& result) {
  crossover_state::Event event;
  event.status = event.crossover_status = result.status; event.stats = result.stats;
  event.basis_export = result.basis_stats;
  if (result.verification) event.crossover_quality = event.after_quality = crossover_state::quality(*result.verification);
  if (result.raw_verification) event.unprojected_quality = crossover_state::quality(*result.raw_verification);
  else event.unprojected_quality = event.crossover_quality;
  event.verification_seconds_total = result.stats.verification_seconds;
  return event;
}

void v120_synthetic_native_zero(const crossover::Result& result, const crossover_state::Event& event) {
  require(result.stats.load_started == 0 && result.stats.crossover_started == 0 && result.stats.get_started == 0 &&
      !result.stats.native_info && event.stats.load_started == 0 && event.stats.crossover_started == 0 &&
      event.stats.get_started == 0 && !event.stats.native_info,
      "synthetic基测试只计真实BASICLU暖清理，不伪造native或完整求解调用");
}

void check_crossover_basis_export() {
  auto model = scalar(0, -infinity, infinity);
  model.costs.push_back(0); model.column_lower.push_back(-infinity); model.column_upper.push_back(infinity);
  model.kinds.push_back(solver::VariableKind::continuous);
  model.row_lower = {1, 1}; model.row_upper = {1, 2};
  model.column_start = {0, 1, 2}; model.row_index = {0, 1}; model.values = {3, 1};
  const std::vector<double> x{1.0 / 3.0, 1.5}, y{0, 0};
  auto strict = policy().tolerances; strict.feasibility_absolute = 0; strict.feasibility_relative = 0;
  const auto initial = solver::verify_lp_solution(model, x, y, strict);
  require(std::numeric_limits<long double>::digits >= 55 && !initial.primal.feasible &&
      static_cast<long double>(x[0]) * 3 != 1,
      "本组件的严格表示反例须先由原算子确认，3乘任意binary64三分之一不能恰为1");
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  const auto disabled = crossover::run(model, x, y, deadline, strict, {}, true);
  require(disabled.status == crossover::Status::rejected && !disabled.basis_export &&
      !disabled.basis_stats.enabled && !disabled.basis_stats.statuses_requested && !disabled.basis_stats.statuses_received &&
      !disabled.basis_stats.published && disabled.stats.get_started == 1 && disabled.stats.get_completed == 1,
      "旧默认adapter拒绝路径仍只取一次解，不申请或发布新基payload");
  const auto exported = crossover::run(model, x, y, deadline, strict, {}, true, true);
  require(exported.status == crossover::Status::rejected && exported.basis_export && exported.basis_stats.enabled &&
      exported.basis_stats.statuses_requested && exported.basis_stats.statuses_received && exported.basis_stats.published &&
      exported.stats.load_started == 1 && exported.stats.crossover_started == 1 &&
      exported.stats.get_started == 1 && exported.stats.get_completed == 1 && exported.stats.native_info &&
      exported.stats.native_info->status_ipm == IPX_STATUS_not_run && exported.stats.native_info->iter == 0,
      "显式导出须来自同一次真实GetBasicSolution，不追加GetBasis、Load、crossover或IPM");
  const auto& basis = *exported.basis_export;
  require(basis.prepared.original_columns == 2 && basis.prepared.costs.size() == 3 && basis.prepared.rhs.size() == 2 &&
      basis.variable_status.size() == 3 && basis.row_status.size() == 2 && basis.structural_seed.size() == 3 &&
      exported.basis_stats.columns == 3 && exported.basis_stats.rows == 2 &&
      basis.prepared.auxiliary_row == std::vector<std::size_t>{1} &&
      std::count(basis.variable_status.begin(), basis.variable_status.end(), IPX_basic) +
          std::count(basis.row_status.begin(), basis.row_status.end(), IPX_basic) == 2,
      "基状态及seed必须完整包含range的q，基本列总数是m而不是原n");
  require(std::all_of(basis.structural_seed.begin(), basis.structural_seed.end(),
      [](double value) { return std::isfinite(value); }) &&
      std::all_of(basis.variable_status.begin(), basis.variable_status.end(), [](ipx::Int status) {
        return status == IPX_basic || status == IPX_nonbasic_lb || status == IPX_nonbasic_ub || status == IPX_superbasic;
      }) && std::all_of(basis.row_status.begin(), basis.row_status.end(), [](ipx::Int status) {
        return status == IPX_basic || status == IPX_nonbasic;
      }), "同Get导出的完整结构seed须有限，列基和行基状态须符合真实IPX定义；不假定q恰好非基");
  const std::vector<double> actual_x{basis.structural_seed[0], basis.structural_seed[1]};
  require(!solver::verify_lp_solution(model, actual_x, y, strict).primal.feasible &&
      exported.verification && !exported.verification->primal.feasible && exported.x.empty() && exported.y.empty(),
      "实际native点的拒绝必须来自原Model严格表示残差，不靠偶然符号错误且不能泄漏失败候选");
  auto ranged = scalar(-1, -infinity, infinity); row(ranged, 1, 1, 2);
  const auto good = crossover::run(ranged, std::array{2.0}, std::array{-1.0}, deadline, {}, {}, true, true);
  require(good.status == crossover::Status::accepted && good.basis_stats.enabled && good.basis_stats.statuses_requested &&
      good.basis_stats.statuses_received && !good.basis_stats.published && !good.basis_export &&
      good.stats.get_started == 1 && good.stats.get_completed == 1 && good.verification && quality_ok(*good.verification),
      "普通原验接受时只能统计同Get已取基，不能向外保留无必要payload");
  auto upper = scalar(0, -infinity, infinity);
  upper.costs.push_back(-1); upper.column_lower.push_back(-infinity); upper.column_upper.push_back(2);
  upper.kinds.push_back(solver::VariableKind::continuous); upper.row_lower = {1}; upper.row_upper = {1};
  upper.column_start = {0, 1, 1}; upper.row_index = {0}; upper.values = {1};
  const auto upper_result = crossover::run(upper, std::array{1.0, 2.0}, std::array{0.0}, deadline, {}, {}, true, true);
  require(upper_result.status == crossover::Status::accepted && upper_result.basis_stats.statuses_received &&
      !upper_result.basis_export && !upper_result.basis_stats.published &&
      quality_ok(solver::verify_lp_solution(upper, upper_result.x, upper_result.y)),
      "upper-only内部翻号的普通最优例也须正确导出统计并恢复原坐标");
  ++mapping_cases;
  details.push_back({"crossover_basis_export", "mapping"});
}

void check_crossover_warm_accept() {
  const auto options = policy();
  const auto input = v120_input();
  const auto x = input.x, y = input.y;
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  auto result = v120_rejected(input, options, deadline);
  const auto raw = *result.raw_verification;
  const auto rejected = *result.verification;
  crossover_state::Log log; log.enabled = true; log.last = v120_event(result);
  auto& event = *log.last;
  double charged_verification = result.stats.verification_seconds;
  {
    crossover_state::Scope scope(&log);
    solver::CrossoverVerificationCollect collect(charged_verification);
    crossover_state::try_warm_cleanup(input.model, result, event, deadline, options,
        [&] { crossover::detail::check(deadline); });
  }
  require(result.status == crossover::Status::accepted && result.verification && quality_ok(*result.verification) &&
      quality_ok(solver::verify_lp_solution(input.model, result.x, result.y)) && result.x == std::vector<double>{1} &&
      !result.basis_export && event.warm && event.warm->started == 1 && event.warm->completed == 1 &&
      event.warm->status == v120_warm::Status::accepted && event.warm->stats.pivots == 1 &&
      event.warm->stats.update.completed == 1 && event.warm->stats.initial_checks &&
      event.warm->stats.initial_checks->primal_feasible,
      "明确合成可行非最优基须经生产helper和真实BASICLU换基后全过原验，不走假native坏点");
  require(event.crossover_status == crossover::Status::rejected && event.crossover_quality &&
      !event.crossover_quality->numerically_optimal && event.crossover_quality->objective_gap == rejected.optimality.objective_gap &&
      event.after_quality && event.after_quality->numerically_optimal && event.status == crossover::Status::accepted &&
      v118_same_bits(input.x, x) && v118_same_bits(input.y, y) && result.raw_verification &&
      same_verification(result.raw_verification->primal, raw.primal) && same_optimality(result.raw_verification->optimality, raw.optimality),
      "最后hybrid接受不能篡改先前crossover拒绝质量或native raw历史，原HSD输入保持不变");
  require(event.verification_seconds_total == event.stats.verification_seconds + event.warm->stats.verification_seconds,
      "清理验算成本与crossover验算只累计一次，不能加总整个暖基时间两次");
  require(std::abs(charged_verification - event.verification_seconds_total) <=
      2 * std::numeric_limits<double>::epsilon() * event.verification_seconds_total,
      "真实core验算收集器析构须只补暖基checker成本，差值再相加只允许double舍入误差");
  const double charged_before = charged_verification;
  {
    crossover_state::Scope scope(&log);
    solver::CrossoverVerificationCollect collect(charged_verification);
  }
  require(charged_verification == charged_before, "无新helper的第二次收集器不能重复计入已付验算成本");
  v120_synthetic_native_zero(result, event);
  const auto completed = event.warm->stats.update.completed;
  int budget_calls = 0;
  crossover_state::try_warm_cleanup(input.model, result, event, deadline, options, [&] { ++budget_calls; });
  require(budget_calls == 0 && event.warm->stats.update.completed == completed, "已处理或已接受的同一payload不得重跑暖基");
  ++transaction_cases;
  details.push_back({"crossover_warm_accept", "transaction"});
}

void check_crossover_warm_reject() {
  const auto options = policy();
  for (int kind : {1, 2, 3}) {
    const auto input = v120_input(kind);
    const auto x = input.x, y = input.y;
    const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
    auto result = v120_rejected(input, options, deadline);
    const auto original = *result.verification;
    auto event = v120_event(result);
    crossover_state::try_warm_cleanup(input.model, result, event, deadline, options, [&] { crossover::detail::check(deadline); });
    require(result.status == crossover::Status::rejected && result.x.empty() && result.y.empty() && !result.basis_export &&
        event.warm && event.warm->started == 1 && event.warm->completed == 1 && event.warm->stats.steps_completed == 0 &&
        result.verification && same_verification(result.verification->primal, original.primal) &&
        same_optimality(result.verification->optimality, original.optimality) &&
        v118_same_bits(input.x, x) && v118_same_bits(input.y, y),
        "暖基初始不可行、奇异或无有限步时须丢弃试点并保原拒绝质量，不能修改HSD输入或偷跑PhaseI");
    const auto expected = kind == 1 ? v120_warm::Status::primal_infeasible :
        kind == 2 ? v120_warm::Status::singular_basis : v120_warm::Status::no_finite_step;
    require(event.warm->status == expected, "失败原因须由实际内核产生，不能统称成功或用最终原验遮盖初基失败");
    if (kind == 1) require(event.warm->stats.pricing_passes == 0, "初基不可行必须在进入自主定价前拒绝");
    int checks = 0;
    crossover_state::try_warm_cleanup(input.model, result, event, deadline, options, [&] { ++checks; });
    require(checks == 0 && event.warm->started == 1, "失败已消费的导出基不可重试");
    v120_synthetic_native_zero(result, event);
  }
  ++transaction_cases;
  details.push_back({"crossover_warm_reject", "transaction"});
}

void check_crossover_warm_budget() {
  const auto options = policy();
  const auto input = v120_input();
  const auto deadline = solver::Clock::now() + std::chrono::seconds(5);
  auto complete = v120_rejected(input, options, deadline);
  auto complete_event = v120_event(complete);
  std::size_t callbacks = 0;
  crossover_state::try_warm_cleanup(input.model, complete, complete_event, deadline, options, [&] { ++callbacks; });
  require(complete.status == crossover::Status::accepted && complete_event.warm &&
      complete_event.warm->stats.update.completed == 1 && callbacks > 2,
      "预算基准必须真实经过生产桥和一次BASICLU更新");
  bool saw_update_partial = false, saw_accepted_unpublished = false;
  for (std::size_t cutoff = 0; cutoff < callbacks; ++cutoff) {
    auto result = v120_rejected(input, options, deadline);
    auto event = v120_event(result);
    std::size_t reached = 0;
    bool stopped = false;
    try {
      crossover_state::try_warm_cleanup(input.model, result, event, deadline, options, [&] {
        if (reached++ == cutoff) throw solver::BudgetExpired{};
      });
    } catch (const solver::BudgetExpired&) { stopped = true; }
    require(stopped && reached == cutoff + 1 && result.status == crossover::Status::time_limit &&
        result.x.empty() && result.y.empty() && !result.basis_export && event.warm &&
        event.warm->started <= 1 && event.warm->completed <= event.warm->started,
        "所有实际外预算截止须先保暖基partial、清payload和半候选，再原样重抛core BudgetExpired");
    const auto& w = *event.warm;
    require(w.stats.update.completed <= w.stats.update.started && w.stats.steps_completed <= w.stats.steps_started &&
        event.verification_seconds_total == event.stats.verification_seconds + w.stats.verification_seconds,
        "内核库更新、重建与验收可在不同前缀完成，不能为失败回滚或丢掉实际验算成本");
    saw_update_partial = saw_update_partial || (w.stats.update.completed == 1 && w.stats.pivots == 0);
    saw_accepted_unpublished = saw_accepted_unpublished || (w.completed == 1 && w.status == v120_warm::Status::accepted);
    v120_synthetic_native_zero(result, event);
  }
  require(saw_update_partial && saw_accepted_unpublished,
      "真实回调截止点须覆盖库更新已付而新基未记，以及暖基原验已过但桥接末预算拒绝采用");
  auto expired = v120_rejected(input, options, deadline);
  auto expired_event = v120_event(expired);
  crossover_state::try_warm_cleanup(input.model, expired, expired_event,
      solver::Clock::now() - std::chrono::seconds(1), options, [] {});
  require(expired.status == crossover::Status::time_limit && expired.x.empty() && expired.y.empty() &&
      !expired.basis_export && expired_event.warm && expired_event.warm->stats.init.started == 0,
      "真实绝对deadline过期必须在BASICLU初始化前拒绝，不能借暖基重新授时");
  ++budget_cases;
  details.push_back({"crossover_warm_budget", "budget"});
}

void check_crossover_warm_hsd_bridge() {
  const auto model = crossover_equality();
  auto options = policy(); options.ipm.scaling_passes = 0; options.ipm.eliminate_fixed_variables = false;
  crossover_state::Log log; log.enabled = true;
  solver::SolveResult solved;
  {
    CrossoverMode mode(true); crossover_state::Scope scope(&log);
    solved = real_coupled_core(model, options);
  }
  original_quality(model, solved, 0);
  crossover_native_once(log);
  require(log.last->iteration >= 1 && log.adopted && log.last->basis_export.enabled &&
      log.last->basis_export.statuses_requested && log.last->basis_export.statuses_received &&
      log.last->crossover_quality && log.last->after_quality && log.last->after_quality->numerically_optimal &&
      solved.backend_version == "ipm-ipx-crossover-cpu-0.4.0" &&
      solved.verification_seconds >= log.last->verification_seconds_total,
      "真实HSD须贯通schema3独立导出和最终质量，数值总成本仍在同一core中");
  // 原生小例通常raw即通过；此处不把它冒称实际执行了暖基修复。
  if (log.last->crossover_status == crossover::Status::accepted)
    require(!log.last->warm && !log.last->basis_export.published, "raw已过就不能再启动暖清理或发布无必要基");
  NativeOracleFixture fixture;
  const auto deadline = fixture.start + std::chrono::seconds(5);
  V120WarmInput input;
  input.model = fixture.reduced;
  input.x = {4.0 / 3.0 - 1e-5, 4.0 / 3.0 - 1e-5}; input.y = {-1.0 / 3.0, -1.0 / 3.0};
  input.prepared = crossover::prepare(input.model, input.x, input.y, deadline);
  input.variable_status = {0, 0}; input.row_status = {-1, -1}; input.seed = input.x;
  auto trial = v120_rejected(input, fixture.options, deadline);
  auto event = v120_event(trial);
  crossover_state::try_warm_cleanup(input.model, trial, event, deadline, fixture.options,
      [&] { crossover::detail::check(deadline); });
  require(trial.status == crossover::Status::accepted && event.warm && event.warm->stats.factor.completed >= 1 &&
      trial.verification && quality_ok(*trial.verification),
      "明确合成的近点与可行列基须经真实生产清理重建，才能送入现原模型oracle");
  {
    candidate::ScopedOracle oracle_scope(&*fixture.oracle);
    require(candidate::consult_current(true, fixture.reduced, trial.x, trial.y, 7) == candidate::OracleDecision::accept,
        "暖基局部成功仍须真实原生postsolve和原Model完整检查，不能直接宣布全局接受");
  }
  require(fixture.oracle->candidate && fixture.stats.oracle.started == 1 && fixture.stats.oracle.accepted == 1 &&
      fixture.stats.postsolve.started == 1 && fixture.stats.postsolve.completed == 1,
      "暖基候选的原oracle只应恢复一次，不得在最终commit再postsolve");
  auto original = candidate::make_shell(fixture.original);
  auto restored = std::move(*fixture.oracle->candidate);
  require(candidate::commit_original_quality(original, std::move(restored.x), std::move(restored.y),
      std::move(restored.checked), fixture.options), "只有oracle拥有的完整原坐标质量才可最终提交");
  original_quality(fixture.original, original, 88.0 / 3.0);
  fixture.no_native_iterations();
  v120_synthetic_native_zero(trial, event);
  ++transaction_cases;
  details.push_back({"crossover_warm_hsd_bridge", "transaction"});
}
// v120 自主暖基衔接组件新增块结束

// v121 缓冲诊断组件新增块开始
struct V121File {
  FILE* stream;
  explicit V121File(FILE* value = std::tmpfile()) : stream(value) {
    require(stream != nullptr, "诊断组件须成功创建真实临时FILE");
  }
  ~V121File() { std::fclose(stream); }
  V121File(const V121File&) = delete;
  V121File& operator=(const V121File&) = delete;
};

std::string v121_bytes(FILE* stream) {
  require(std::fflush(stream) == 0 && std::fseek(stream, 0, SEEK_SET) == 0,
      "读取诊断原字节前须完成真实刷新与定位");
  std::string bytes;
  std::array<char, 4096> buffer;
  for (;;) {
    const auto count = std::fread(buffer.data(), 1, buffer.size(), stream);
    bytes.append(buffer.data(), count);
    if (count < buffer.size()) break;
  }
  require(!std::ferror(stream), "临时文件原字节读取失败");
  return bytes;
}

struct V121Logs {
  candidate::Stats presolve;
  coupled_state::Log coupled;
  energy_state::Log energy;
  crossover_state::Log crossover;
  explicit V121Logs(bool populated) {
    presolve.enabled = coupled.enabled = energy.enabled = crossover.enabled = true;
    if (!populated) return;
    // 本夹具只构造日志，含正常字符串转义与摘要边界，不调用求解或原生优化。
    presolve.child.status = "中文\"\\\n";
    auto& quality = presolve.original_quality.emplace();
    quality.objective = -0.0; quality.max_row_violation = std::numeric_limits<double>::quiet_NaN();
    quality.max_bound_violation = infinity;
    presolve.oracle.events.resize(2);
    presolve.oracle.events[0].decision = "reject"; presolve.oracle.events[1].decision = "accept";
    presolve.oracle.events[1].original_quality = quality;
    auto& call = coupled.last.emplace(); call.role = "corrector"; call.attempt_iteration = 1;
    call.work.dimension = 8; call.work.tolerances.fill(1e-9); call.work.status = "numerical_error";
    call.work.initial_measure = -0.0L; call.work.final_measure = std::numeric_limits<long double>::max();
    call.work.block_measures = {0, 1, std::numeric_limits<long double>::quiet_NaN(), 2, 3};
    call.work.initial_forcing.emplace().rhs_norms = {1, 2, 3};
    call.work.candidate_forcing.emplace().defect_norms = std::array<long double, 5>{0, 1, 2, 3, 4};
    auto& energy_call = energy.last.emplace(); energy_call.attempt_iteration = 1;
    energy_call.work.columns = 2; energy_call.work.rows = 3;
    energy_call.work.primal_energy = std::numeric_limits<long double>::denorm_min();
    energy_call.work.row_energy = -0.0L; energy_call.work.denominator = std::numeric_limits<long double>::max();
    auto& event = crossover.last.emplace(); event.iteration = 1;
    event.before_quality.objective = -0.0; event.after_quality.emplace().max_dual_residual = infinity;
    event.crossover_quality.emplace(); event.unprojected_quality.emplace();
    event.stats.native_info.emplace();
    auto& cleanup = event.warm.emplace(); cleanup.raw_quality.emplace(); cleanup.after_quality.emplace();
    cleanup.stats.initial_checks.emplace(); cleanup.stats.last_checks.emplace();
    cleanup.stats.stage = "中文\"\\\n";
  }
};

constexpr std::array<const char*, 4> v121_prefixes{
    "V100_PRESOLVE ", "V100_ENERGY ", "V100_COUPLED ", "V117_CROSSOVER "};
void v121_write(FILE* stream, const V121Logs& logs, std::size_t marker) noexcept {
  switch (marker) {
    case 0: candidate::write_diagnostics(stream, logs.presolve); break;
    case 1: energy_state::write(stream, logs.energy); break;
    case 2: coupled_state::write(stream, logs.coupled); break;
    case 3: crossover_state::write(stream, logs.crossover); break;
  }
}

void check_diagnostic_buffered_serializers() {
  for (const bool populated : {false, true}) {
    const V121Logs logs(populated);
    for (std::size_t marker = 0; marker < v121_prefixes.size(); ++marker) {
      V121File direct, buffered;
      unsigned direct_calls = 0, buffered_calls = 0;
      require(candidate::emit_direct(direct.stream, v121_prefixes[marker], [&](FILE* stream) noexcept {
        ++direct_calls; v121_write(stream, logs, marker);
      }) == candidate::EmitStatus::direct &&
          candidate::emit_buffered(buffered.stream, v121_prefixes[marker], [&](FILE* stream) noexcept {
            ++buffered_calls; v121_write(stream, logs, marker);
          }) == candidate::EmitStatus::buffered,
          "四真实serializer须实际走原直接路径和新副本缓冲路径，不以fallback冒称缓冲成功");
      const auto reference = v121_bytes(direct.stream);
      require(reference == v121_bytes(buffered.stream) && direct_calls == 1 && buffered_calls == 1 &&
          reference.starts_with(v121_prefixes[marker]) && reference.ends_with('\n'),
          "全部可选字段、特殊值、UTF8和转义必须逐字节相同，不能重新拼造数学或JSON字段");
      if (populated && marker == 0)
        require(reference.find("中文\\\"\\\\\\u000a") != std::string::npos && reference.find("null") != std::string::npos,
            "独立夹具须确实经过原字符串转义与非有限摘要分支");
    }
  }
  ++mapping_cases;
  details.push_back({"diagnostic_buffered_serializers", "mapping"});
}

void check_diagnostic_buffered_file_modes() {
  std::array<char, 4096> pending_buffer;
  V121File file;
  require(std::setvbuf(file.stream, pending_buffer.data(), _IOFBF, pending_buffer.size()) == 0,
      "只对全新组件FILE配置pending缓冲，不修改全局stderr");
  std::string expected(128, '~');
  require(std::fwrite(expected.data(), 1, expected.size(), file.stream) == expected.size() &&
      std::fflush(file.stream) == 0 && std::fseek(file.stream, 16, SEEK_SET) == 0 &&
      std::fputs("PENDING", file.stream) >= 0, "普通文件须先建立旧内容、中间offset和未刷前缀");
  unsigned calls = 0;
  require(candidate::emit_buffered(file.stream, "TMP ", [&](FILE* stream) noexcept {
    ++calls; std::fputs("{}", stream);
  }) == candidate::EmitStatus::buffered && calls == 1 && std::fputs("TAIL", file.stream) >= 0 &&
      std::fflush(file.stream) == 0, "副本关闭后原FILE必须仍可继续写，且先刷新原pending前缀");
  const std::string middle = "PENDINGTMP {}\nTAIL";
  require(std::ftell(file.stream) == static_cast<long>(16 + middle.size()), "副本写入须推进同一真实文件位置");
  expected.replace(16, middle.size(), middle);
  require(v121_bytes(file.stream) == expected, "fdopen既有描述符不能截断文件或覆盖错误offset，原剩余后缀须完整");
  for (const int mode : {_O_TEXT, _O_BINARY}) {
    V121File direct, buffered;
    require(_setmode(_fileno(direct.stream), mode) != -1 && _setmode(_fileno(buffered.stream), mode) != -1,
        "模式组件须显式建立Windows文本及二进制原FD");
    const auto writer = [](FILE* stream) noexcept { std::fputs("正文\n第二行", stream); };
    require(std::fputs("HEAD\n", direct.stream) >= 0 && std::fputs("HEAD\n", buffered.stream) >= 0 &&
        candidate::emit_direct(direct.stream, "MODE ", writer) == candidate::EmitStatus::direct &&
        candidate::emit_buffered(buffered.stream, "MODE ", writer) == candidate::EmitStatus::buffered &&
        std::fputs("TAIL\n", direct.stream) >= 0 && std::fputs("TAIL\n", buffered.stream) >= 0 &&
        std::fflush(direct.stream) == 0 && std::fflush(buffered.stream) == 0,
        "text/binary必须逐一真实发射，关闭副本后原FD仍可写后续文本");
    // 完成写入后在原FD查询模式；再以binary读原始CR/LF，不让读取翻译掩盖差异。
    require(_setmode(_fileno(direct.stream), _O_BINARY) == mode &&
        _setmode(_fileno(buffered.stream), _O_BINARY) == mode,
        "helper不能改变原FD翻译模式，也不能把文本流暗改成二进制");
    const auto reference = v121_bytes(direct.stream);
    require(reference == v121_bytes(buffered.stream) &&
        ((reference.find("\r\n") != std::string::npos) == (mode == _O_TEXT)),
        "原始捕获须证明文本CRLF与二进制LF各自保持，而非只比较翻译后的读取文本");
  }
  ++transaction_cases;
  details.push_back({"diagnostic_buffered_file_modes", "transaction"});
}

void check_diagnostic_buffered_large_concurrent() {
  const std::array<std::string, 2> payload{std::string(70001, 'A'), std::string(70001, 'B')};
  V121File direct, buffered;
  const auto writer = [&](FILE* stream) noexcept { std::fwrite(payload[0].data(), 1, payload[0].size(), stream); };
  require(candidate::emit_direct(direct.stream, "LARGE ", writer) == candidate::EmitStatus::direct &&
      candidate::emit_buffered(buffered.stream, "LARGE ", writer) == candidate::EmitStatus::buffered &&
      v121_bytes(direct.stream) == v121_bytes(buffered.stream),
      "大于64KiB的真实多次flush必须无损，不能只测始终装得下一块的短日志");
  const std::array<std::string, 2> lines{"THREAD " + payload[0] + '\n', "THREAD " + payload[1] + '\n'};
  for (const bool independent : {false, true}) {
    V121File first, second;
    std::atomic<bool> valid{true};
    const auto worker = [&](std::size_t index) {
      FILE* target = independent && index == 1 ? second.stream : first.stream;
      for (unsigned repeat = 0; repeat < 4; ++repeat)
        if (candidate::emit_buffered(target, "THREAD ", [&](FILE* stream) noexcept {
          std::fwrite(payload[index].data(), 1, payload[index].size(), stream);
        }) != candidate::EmitStatus::buffered) valid.store(false);
    };
    {
      std::jthread left(worker, 0), right(worker, 1);
    }
    require(valid.load(), "两个线程的真实副本流都必须成功，不能把失败或fallback当作并发通过");
    const auto bytes = v121_bytes(first.stream);
    if (independent) {
      std::string a, b;
      for (unsigned repeat = 0; repeat < 4; ++repeat) { a += lines[0]; b += lines[1]; }
      require(bytes == a && v121_bytes(second.stream) == b, "不同目标FILE也不能共享易被覆盖的静态64KiB缓冲");
    } else {
      std::array<unsigned, 2> counts{};
      std::size_t position = 0;
      while (position < bytes.size()) {
        if (bytes.compare(position, lines[0].size(), lines[0]) == 0) { ++counts[0]; position += lines[0].size(); }
        else if (bytes.compare(position, lines[1].size(), lines[1]) == 0) { ++counts[1]; position += lines[1].size(); }
        else require(false, "同一FILE锁必须覆盖完整prefix、正文、newline和flush，不能出现线程间碎行");
      }
      require(counts == std::array<unsigned, 2>{4, 4} && v121_bytes(second.stream).empty(),
          "同目标输出顺序可变，但每种完整长行必须恰好出现四次且不写入其它流");
    }
  }
  ++transaction_cases;
  details.push_back({"diagnostic_buffered_large_concurrent", "transaction"});
}

struct V121StderrCapture {
  V121File file;
  int saved = -1;
  V121StderrCapture() {
    require(std::fflush(stderr) == 0, "重定向前须刷新现stderr，不改变其缓冲配置");
    saved = _dup(_fileno(stderr));
    require(saved >= 0, "须保存原stderr描述符");
    if (_dup2(_fileno(file.stream), _fileno(stderr)) != 0) {
      _close(saved); saved = -1;
      require(false, "诊断组件stderr临时重定向失败");
    }
  }
  void restore() {
    if (saved < 0) return;
    const int flush = std::fflush(stderr);
    const int restored = _dup2(saved, _fileno(stderr));
    const int closed = _close(saved); saved = -1;
    require(flush == 0 && restored == 0 && closed == 0, "必须恢复真实stderr且不关闭其原目标");
  }
  ~V121StderrCapture() {
    if (saved >= 0) { std::fflush(stderr); _dup2(saved, _fileno(stderr)); _close(saved); }
  }
};

void check_diagnostic_buffered_disabled_errors() {
  V121Logs logs(true);
  V121StderrCapture capture;
  {
    struct DisableEmit {
      bool previous = candidate::emit_diagnostics;
      DisableEmit() noexcept { candidate::emit_diagnostics = false; }
      ~DisableEmit() { candidate::emit_diagnostics = previous; }
    } disabled;
    candidate::Log presolve_report(logs.presolve);
    energy_state::Report energy_report(logs.energy);
    coupled_state::Report coupled_report(logs.coupled);
    crossover_state::Report crossover_report(logs.crossover);
  }
  capture.restore();
  require(v121_bytes(capture.file.stream).empty(), "真实四Report关闭时必须零输出，不能仅测未被调用的helper");
  V121File valid;
  unsigned calls = 0;
  const auto writer = [&](FILE* stream) noexcept { ++calls; std::fputs("正文", stream); };
  require(candidate::emit_buffered(nullptr, "BAD ", writer) == candidate::EmitStatus::failed &&
      candidate::emit_direct(nullptr, "BAD ", writer) == candidate::EmitStatus::failed &&
      candidate::emit_buffered(valid.stream, nullptr, writer) == candidate::EmitStatus::failed &&
      candidate::emit_direct(valid.stream, nullptr, writer) == candidate::EmitStatus::failed &&
      calls == 0 && v121_bytes(valid.stream).empty(), "null目标或prefix须明确failed且不进入writer或写半个marker");
  int descriptors[2] = {-1, -1};
  require(_pipe(descriptors, 4096, _O_BINARY) == 0, "须创建真实binary管道，不用伪造IO回调");
  const int read_closed = _close(descriptors[0]);
  FILE* stream = _fdopen(descriptors[1], "wb");
  if (!stream) { _close(descriptors[1]); require(false, "管道写端关联FILE失败"); }
  V121File broken_pipe(stream);
  require(read_closed == 0, "必须关闭全部管道读端，才能真实触发写出后的broken-pipe错误");
  require(candidate::emit_buffered(broken_pipe.stream, "BROKEN ", writer) == candidate::EmitStatus::failed && calls == 1,
      "prefix和writer已进缓冲后的真实flush错误须failed，不能重放而第二次调用writer");
  const int still_open = _dup(_fileno(broken_pipe.stream));
  require(still_open >= 0 && _close(still_open) == 0, "写失败也只能释放副本，不能关闭调用者原FD");
  V121File unicode;
  require(_setmode(_fileno(unicode.stream), _O_U8TEXT) != -1, "须建立只检查资格的Unicode描述符模式");
  calls = 0;
  const auto status = candidate::emit_buffered(unicode.stream, "UNICODE ", writer);
  const int restored = _setmode(_fileno(unicode.stream), _O_BINARY);
  require(status == candidate::EmitStatus::failed && calls == 0 &&
      (restored == _O_U8TEXT || restored == _O_U16TEXT || restored == _O_WTEXT) &&
      v121_bytes(unicode.stream).empty(), "Unicode模式只做整数资格判断后拒绝，不在该模式调用narrow writer");
  ++exception_cases;
  details.push_back({"diagnostic_buffered_disabled_errors", "exception"});
}
// v121 缓冲诊断组件新增块结束

// 生产类型化选项组件开始
std::pair<solver::SolveResult, std::string> public_typed_solve(const solver::Model& model,
    const solver::SolveOptions& options) {
  V121StderrCapture capture;
  ++component_ipm_calls;
  auto result = solver::solve_ipm(model, options);
  capture.restore();
  return {std::move(result), v121_bytes(capture.file.stream)};
}

std::string_view public_marker(std::string_view text, std::string_view prefix) {
  const auto begin = text.find(prefix);
  require(begin != std::string_view::npos && text.find(prefix, begin + prefix.size()) == std::string_view::npos,
      "单次公共求解的指定诊断marker必须恰好出现一次");
  const auto end = text.find('\n', begin);
  require(end != std::string_view::npos, "公共诊断marker必须在返回前输出完整行");
  return text.substr(begin, end - begin);
}

void check_public_typed_presolve_and_crossover() {
  require(!candidate::enabled_override && !crossover_state::enabled_override,
      "公共类型化控制组件不能依赖内部测试override或版本环境变量");
  auto options = policy(); options.ipm.presolve = true; options.ipm.diagnostics = true;
  const auto original = coupled(true);
  auto [presolved, presolve_text] = public_typed_solve(original, options);
  original_quality(original, presolved, 88.0 / 3.0);
  const auto presolve_line = public_marker(presolve_text, "V100_PRESOLVE ");
  require(presolve_line.find("\"enabled\":true,\"attempted\":true") != std::string_view::npos &&
      presolve_line.find("\"route\":\"reduced\"") != std::string_view::npos &&
      public_marker(presolve_text, "V100_COUPLED ").find("\"enabled\":true") != std::string_view::npos &&
      presolve_text.find("V117_CROSSOVER ") == std::string::npos &&
      presolved.backend_version == "ipm-cpu-0.54.0",
      "仅SolveOptions.presolve就须真实进入约化及联立方向，不能靠环境变量或偷偷启用交叉");
  auto hybrid = policy(); hybrid.ipm.crossover = true; hybrid.ipm.diagnostics = true;
  hybrid.ipm.scaling_passes = 0; hybrid.ipm.eliminate_fixed_variables = false;
  const auto model = crossover_equality();
  auto [crossed, crossover_text] = public_typed_solve(model, hybrid);
  original_quality(model, crossed, 0);
  const auto crossover_line = public_marker(crossover_text, "V117_CROSSOVER ");
  require(crossover_line.find("\"schema\":3") != std::string_view::npos &&
      crossover_line.find("\"enabled\":true,\"attempted\":true") != std::string_view::npos &&
      crossover_line.find("\"get_started\":1,\"get_completed\":1") != std::string_view::npos &&
      crossover_line.find("\"adopted\":true") != std::string_view::npos &&
      crossed.ipm_iterations && *crossed.ipm_iterations >= 1 &&
      crossed.backend_version == "ipm-ipx-crossover-cpu-0.4.0" &&
      solver::ipm_version(hybrid) == crossed.backend_version && solver::ipm_version() == "ipm-cpu-0.54.0",
      "类型化交叉须来自真实HSD更新及同次Get，并通过完整原验；选项和版本不得泄漏到下一次调用");
  auto unsupported = hybrid; unsupported.ipm.method = "mehrotra";
  unsupported.ipm.linear_accuracy = "strict"; unsupported.ipm.diagnostics = false;
  auto [invalid, invalid_text] = public_typed_solve(model, unsupported);
  require(invalid.status == "error" && !invalid.has_solution && invalid.ipm_iterations == 0 &&
      invalid.ipm_factorizations == 0 && invalid.message.find("IPX交叉仅支持 homogeneous") != std::string::npos &&
      invalid_text.empty(), "正式交叉选项与非HSD组合必须在数值工作前明确报错，不能静默不执行");
  ++mapping_cases;
  details.push_back({"public_typed_presolve_and_crossover", "mapping"});
}

void check_public_diagnostics_profile_control() {
  require(!candidate::enabled_override && !crossover_state::enabled_override,
      "诊断开关组件通过公共选项而非内部override求解");
  auto options = policy(); options.ipm.presolve = true;
  const auto model = coupled(true);
  auto [silent, silent_text] = public_typed_solve(model, options);
  original_quality(model, silent, 88.0 / 3.0);
  require(silent_text.empty(), "生产诊断和profile默认关闭，公共求解不得输出隐藏版本日志");
  options.ipm.diagnostics = true;
  auto [diagnostic, diagnostic_text] = public_typed_solve(model, options);
  same_non_timing(silent, diagnostic);
  for (const auto* prefix : {"V100_PRESOLVE ", "V100_COUPLED ", "V100_ENERGY "}) public_marker(diagnostic_text, prefix);
  require(diagnostic_text.find("V112_PROFILE ") == std::string::npos &&
      diagnostic_text.find("V117_CROSSOVER ") == std::string::npos,
      "诊断只控制既有数据输出，不默认启动profile或原生交叉");
  options.ipm.diagnostics = false; options.ipm.profile = true;
  auto [profiled, profile_text] = public_typed_solve(model, options);
  same_non_timing(silent, profiled);
  public_marker(profile_text, "V112_PROFILE ");
  for (const auto* prefix : v121_prefixes)
    require(profile_text.find(prefix) == std::string::npos, "profile须能独立开启且不捎带四个数值诊断marker");
  ++transaction_cases;
  details.push_back({"public_diagnostics_profile_control", "transaction"});
}

void check_public_option_context_restoration() {
  namespace profile = solver::detail::ipm_profile;
  require(!candidate::enabled_override && !crossover_state::enabled_override,
      "作用域组件不得借测试override遮盖实际类型化选项");
  const bool previous_presolve = candidate::option_enabled, previous_crossover = crossover_state::option_enabled;
  const bool previous_emit = candidate::emit_diagnostics;
  const auto previous_profile = profile::implementation::current;
  auto* previous_coupled = coupled_state::context;
  auto* previous_energy = energy_state::context;
  auto* previous_cross = crossover_state::context;
  {
    candidate::OptionsScope outer_presolve(true);
    crossover_state::OptionsScope outer_crossover(true);
    candidate::DiagnosticScope outer_emit(true);
    profile::Report outer_profile(true, false);
    const auto outer_context = profile::implementation::current;
    coupled_state::Log coupled_log; coupled_log.enabled = true;
    energy_state::Log energy_log; energy_log.enabled = true;
    crossover_state::Log cross_log; cross_log.enabled = true;
    coupled_state::Scope coupled_scope(&coupled_log);
    energy_state::Scope energy_scope(&energy_log);
    crossover_state::Scope crossover_scope(&cross_log);
    const auto restored = [&] {
      return candidate::enabled() && crossover_state::requested() && candidate::emit_diagnostics &&
          profile::implementation::current == outer_context && coupled_state::context == &coupled_log &&
          energy_state::context == &energy_log && crossover_state::context == &cross_log;
    };
    auto options = policy();
    const auto model = coupled();
    auto [result, text] = public_typed_solve(model, options);
    original_quality(model, result, 25.0 / 3.0);
    require(restored() && text.empty() && result.backend_version == "ipm-cpu-0.54.0" &&
        outer_profile.stats.clock_reads == 0 && !coupled_log.last && !energy_log.last && !cross_log.last,
        "嵌套公共入口必须按本次false选项隔离外层日志/profile，退出后完整恢复而不写入外层账本");
    options.ipm.method = "invalid";
    auto [invalid, invalid_text] = public_typed_solve(model, options);
    require(invalid.status == "error" && invalid_text.empty() && restored(),
        "公共参数错误的真实返回路径也须恢复选项及TLS，不冒称存在公开异常透传");
    struct ScopeUnwind {};
    bool caught = false;
    try {
      candidate::OptionsScope inner_presolve(false);
      crossover_state::OptionsScope inner_crossover(false);
      candidate::DiagnosticScope inner_emit(false);
      profile::Report inner_profile(false, false);
      require(!candidate::enabled() && !crossover_state::requested() && !candidate::emit_diagnostics &&
          profile::implementation::current == nullptr, "实际生产guard必须先装入内层选项再验证异常恢复");
      throw ScopeUnwind{};
    } catch (const ScopeUnwind&) { caught = true; }
    require(caught && restored(), "实际OptionsScope、DiagnosticScope和profile Report在异常析构后须恢复外层");
  }
  require(candidate::option_enabled == previous_presolve && crossover_state::option_enabled == previous_crossover &&
      candidate::emit_diagnostics == previous_emit && profile::implementation::current == previous_profile &&
      coupled_state::context == previous_coupled && energy_state::context == previous_energy &&
      crossover_state::context == previous_cross, "所有外层owner退出后不得残留本组件的请求或TLS指针");
  ++exception_cases;
  details.push_back({"public_option_context_restoration", "exception"});
}
// 生产类型化选项组件结束

// 自主对偶暖基组件开始
namespace warm_dual = solver::detail::warm_cleanup;
solver::Model warm_dual_model(std::vector<double> costs, std::vector<double> lower, std::vector<double> upper,
    const std::vector<std::vector<double>>& matrix, std::vector<double> row_lower, std::vector<double> row_upper) {
  solver::Model model;
  model.name = "自主对偶暖基解析组件";
  model.costs = std::move(costs); model.column_lower = std::move(lower); model.column_upper = std::move(upper);
  model.row_lower = std::move(row_lower); model.row_upper = std::move(row_upper);
  model.kinds.assign(model.costs.size(), solver::VariableKind::continuous);
  model.column_start.assign(1, 0);
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    for (std::size_t row_index = 0; row_index < matrix.size(); ++row_index) {
      require(matrix[row_index].size() == model.costs.size(), "解析小矩阵维数必须明确");
      if (matrix[row_index][col] != 0) {
        model.row_index.push_back(static_cast<std::int64_t>(row_index)); model.values.push_back(matrix[row_index][col]);
      }
    }
    model.column_start.push_back(static_cast<std::int64_t>(model.values.size()));
  }
  return model;
}

V120WarmInput warm_dual_fixture(solver::Model model, std::vector<ipx::Int> variables,
    std::vector<ipx::Int> rows, std::vector<double> seed = {}) {
  V120WarmInput input;
  input.model = std::move(model); input.variable_status = std::move(variables); input.row_status = std::move(rows);
  for (std::size_t col = 0; col < input.model.costs.size(); ++col)
    input.x.push_back(std::min(input.model.column_upper[col], std::max(0.0, input.model.column_lower[col])));
  input.y.assign(input.model.row_lower.size(), 0);
  input.prepared = crossover::prepare(input.model, input.x, input.y, solver::Clock::now() + std::chrono::seconds(5));
  input.seed = seed.empty() ? input.prepared.x : std::move(seed);
  require(input.variable_status.size() == input.prepared.costs.size() && input.row_status.size() == input.prepared.rhs.size() &&
      input.seed.size() == input.prepared.costs.size(), "显式合成列基、逻辑列及seed必须对应真实Prepared");
  return input;
}

warm_dual::Input warm_dual_input(const V120WarmInput& input) {
  return {input.model, input.prepared, input.variable_status, input.row_status, input.seed};
}

warm_dual::Result warm_dual_run(const V120WarmInput& input, std::uint64_t max_steps = 64) {
  return warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
      warm_dual::Limits{max_steps, true});
}

void warm_dual_success(const V120WarmInput& input, const warm_dual::Result& result,
    const std::vector<double>& expected, double objective, std::uint64_t pivots, std::uint64_t factors = 1) {
  const auto& s = result.stats;
  require(result.status == warm_dual::Status::accepted && result.x == expected && result.verification &&
      quality_ok(*result.verification) && quality_ok(solver::verify_lp_solution(input.model, result.x, result.y)),
      "自主对偶修复必须得到解析原坐标点并独立通过原Model全验，不能拿IPX结果自证");
  require(result.verification->primal.objective && *result.verification->primal.objective == objective &&
      s.dual_enabled && s.dual_entered && s.initial_checks && !s.initial_checks->primal_feasible &&
      s.initial_checks->nonbasic_dual_checked && s.initial_checks->nonbasic_dual_feasible &&
      s.dual_entry_checks && s.dual_entry_checks->equation_ratio <= 1 &&
      s.dual_entry_checks->basis_dual_feasible && s.dual_entry_checks->nonbasic_dual_checked &&
      s.dual_entry_checks->nonbasic_dual_feasible &&
      s.last_checks && s.last_checks->primal_feasible && s.last_checks->nonbasic_dual_checked &&
      s.last_checks->nonbasic_dual_feasible,
      "初始真越界、完整非基对偶资格及最终可行必须分别证明，不能只验r_B");
  require(s.pivots == pivots && s.dual_pivots == pivots && s.steps_completed == pivots &&
      s.dual_steps_completed == pivots && s.bound_flips == 0 && s.update.completed == pivots &&
      s.ftran.completed == pivots && s.btran.completed == pivots && s.dual_row_solve.completed == pivots &&
      s.factor.completed == factors && s.refactors + 1 == factors && s.pricing_passes == 0,
      "对偶须用真实行BTRAN、FTRAN、T更新及BASICLU换基，不新增入口因子、不冒用primal flip");
}

V120WarmInput warm_dual_lower() {
  return warm_dual_fixture(warm_dual_model({1, 0}, {0, 0}, {infinity, infinity}, {{1, -1}}, {1}, {1}), {-1, 0}, {-1});
}

V120WarmInput warm_dual_overshoot() {
  return warm_dual_fixture(warm_dual_model({1, 2}, {0, 0}, {1, infinity}, {{1, 1}}, {2}, {infinity}), {-1, -1}, {0});
}

void check_warm_dual_lower_upper() {
  const auto lower = warm_dual_lower();
  const auto disabled = warm_dual::run(warm_dual_input(lower), solver::Clock::now() + std::chrono::seconds(5));
  require(disabled.status == warm_dual::Status::primal_infeasible && !disabled.stats.dual_enabled &&
      !disabled.stats.dual_entered && disabled.stats.dual_checks_started == 0 && disabled.stats.dual_row_solve.started == 0,
      "默认false必须保持旧原始PhaseII拒绝，不额外扫描或分配对偶工作");
  const auto repaired = warm_dual_run(lower);
  warm_dual_success(lower, repaired, {1, 0}, 1, 1);
  require(repaired.stats.last_dual_theta == 1 && repaired.stats.last_dual_primal_step == 1 &&
      repaired.stats.last_pivot == -1, "下界修复须保原始带符号主元并区分两个步长账本");
  const auto upper = warm_dual_fixture(warm_dual_model({1, 0}, {0, 0}, {infinity, 1}, {{1, 1}}, {2}, {2}), {-1, 0}, {-1});
  warm_dual_success(upper, warm_dual_run(upper), {1, 1}, 1, 1);
  for (const double lower_bound : {0.0, -infinity}) {
    const auto incoming_upper = warm_dual_fixture(warm_dual_model({-1, 0}, {lower_bound, 0}, {2, infinity},
        {{1, 1}}, {1}, {1}), {-2, 0}, {-1});
    warm_dual_success(incoming_upper, warm_dual_run(incoming_upper), {1, 0}, -1, 1);
  }
  auto maximum_model = lower.model;
  maximum_model.sense = solver::ObjectiveSense::maximize; maximum_model.costs[0] = -1; maximum_model.objective_offset = 7;
  const auto maximum = warm_dual_fixture(std::move(maximum_model), {-1, 0}, {-1});
  warm_dual_success(maximum, warm_dual_run(maximum), {1, 0}, 6, 1);
  ++transaction_cases;
  details.push_back({"warm_dual_lower_upper", "transaction"});
}

void check_warm_dual_bounded_overshoot() {
  const auto input = warm_dual_overshoot();
  const auto first = warm_dual_run(input, 1);
  require(first.status == warm_dual::Status::step_limit && first.x.empty() && first.y.empty() &&
      first.stats.dual_pivots == 1 && first.stats.bound_flips == 0 && first.stats.last_checks &&
      !first.stats.last_checks->primal_feasible && first.stats.last_checks->nonbasic_dual_feasible &&
      first.stats.last_checks->bound_violation == 1 && first.stats.last_dual_primal_step == 2 &&
      first.stats.last_dual_theta == 1,
      "首入x必须真越过上界至2，不得偷偷BFRT翻界或因重建仍越界立即弃掉可继续的对偶路线");
  const auto complete = warm_dual_run(input, 2);
  warm_dual_success(input, complete, {1, 1}, 3, 2, 2);
  require(complete.stats.last_entering == 1 && complete.stats.last_leaving == 0 &&
      complete.stats.last_dual_primal_step == 1 && complete.stats.last_dual_theta == 1,
      "第二pivot才把x置上界1；m1原周期refactor保留，但不为dual入口另做因子");
  ++transaction_cases;
  details.push_back({"warm_dual_bounded_overshoot", "transaction"});
}

void check_warm_dual_free_superbasic() {
  for (const bool negative : {false, true}) {
    const auto free = warm_dual_fixture(warm_dual_model({0, 0}, {-infinity, 0}, {infinity, infinity},
        {{1, negative ? 1.0 : -1.0}}, {negative ? -1.0 : 1.0}, {negative ? -1.0 : 1.0}),
        {-3, 0}, {-1}, {negative ? 2.0 : -2.0, 0});
    const auto result = warm_dual_run(free);
    warm_dual_success(free, result, {negative ? -1.0 : 1.0, 0}, 0, 1);
    require(result.stats.last_dual_theta == 0 && result.stats.last_dual_primal_step == 3 &&
        result.stats.last_theta == result.stats.last_dual_primal_step && result.stats.zero_steps == 0 &&
        result.stats.dual_zero_steps == 1,
        "自由非零seed须双向使用真实值，dual零价不等于原始位移零或不用换基");
  }
  auto at_lower = warm_dual_lower(); at_lower.variable_status[0] = -3;
  warm_dual_success(at_lower, warm_dual_run(at_lower), {1, 0}, 1, 1);
  const auto at_upper = warm_dual_fixture(warm_dual_model({-1, 0}, {0, 0}, {2, infinity}, {{1, 1}}, {1}, {1}),
      {-3, 0}, {-1}, {2, 0});
  warm_dual_success(at_upper, warm_dual_run(at_upper), {1, 0}, -1, 1);
  const auto interior = warm_dual_fixture(warm_dual_model({0, 0}, {0, 0}, {3, infinity}, {{1, -1}}, {2}, {2}),
      {-3, 0}, {-1}, {1, 0});
  warm_dual_success(interior, warm_dual_run(interior), {2, 0}, 0, 1);
  auto wrong_model = interior.model; wrong_model.costs[0] = 1;
  const auto wrong_interior = warm_dual_fixture(std::move(wrong_model), {-3, 0}, {-1}, {1, 0});
  const auto refused = warm_dual_run(wrong_interior);
  require(refused.status == warm_dual::Status::dual_infeasible && refused.stats.dual_checks_completed == 1 &&
      refused.stats.initial_checks && refused.stats.initial_checks->nonbasic_dual_checked &&
      !refused.stats.initial_checks->nonbasic_dual_feasible && !refused.stats.dual_entered &&
      refused.stats.dual_row_solve.started == 0 && refused.stats.update.started == 0 && refused.x.empty(),
      "有限-3内点的非零r必须完整资格拒绝，不能把-3当成恰下界或任意移动方向");
  const auto wrong_free = warm_dual_fixture(warm_dual_model({1, 0}, {-infinity, 0}, {infinity, infinity},
      {{1, 1}}, {-1}, {-1}), {-3, 0}, {-1}, {2, 0});
  const auto free_refused = warm_dual_run(wrong_free);
  require(free_refused.status == warm_dual::Status::dual_infeasible && !free_refused.stats.dual_entered &&
      free_refused.stats.update.started == 0, "自由非基的非零r同样不具对偶资格，不能只检查有限内点");
  ++mapping_cases;
  details.push_back({"warm_dual_free_superbasic", "mapping"});
}

void check_warm_dual_qualification_fixed() {
  const auto fixed_basic = warm_dual_fixture(warm_dual_model({1, 0}, {0, 0}, {infinity, 0}, {{1, 1}}, {1}, {1}),
      {-1, 0}, {-1});
  warm_dual_success(fixed_basic, warm_dual_run(fixed_basic), {1, 0}, 1, 1);
  const auto redundant = warm_dual_fixture(warm_dual_model({-100, 1, 0}, {1, 0, 0}, {1, infinity, infinity},
      {{1, 1, -1}, {0, 0, 0}}, {2, 0}, {2, 0}), {-1, -1, 0}, {-1, 0});
  warm_dual_success(redundant, warm_dual_run(redundant), {1, 1, 0}, -99, 1);
  const auto range = warm_dual_fixture(warm_dual_model({1}, {0}, {infinity}, {{1}}, {1}, {2}), {-1, 0}, {-1});
  require(range.prepared.auxiliary_row == std::vector<std::size_t>{0}, "range夹具须真实产生q列而非伪造逻辑slack");
  warm_dual_success(range, warm_dual_run(range), {1}, 1, 1);
  const auto wrong_slack = warm_dual_fixture(warm_dual_model({1}, {2}, {infinity}, {{1}}, {-infinity}, {1}), {0}, {-1});
  const auto wrong_q = warm_dual_fixture(warm_dual_model({-1}, {2}, {infinity}, {{1}}, {1}, {2}), {0, -1}, {-1});
  for (const auto* input : {&wrong_slack, &wrong_q}) {
    const auto result = warm_dual_run(*input);
    require(result.status == warm_dual::Status::dual_infeasible && result.stats.dual_checks_completed == 1 &&
        !result.stats.dual_entered && result.stats.update.started == 0,
        "完整非基资格须实际包含非固定逻辑slack和range的q，不能只检查原结构列");
  }
  const auto singular = warm_dual_run(v120_input(2));
  require(singular.status == warm_dual::Status::singular_basis && singular.stats.factor.completed == 1 &&
      singular.stats.dual_checks_started == 0 && singular.stats.update.started == 0 && singular.x.empty(),
      "奇异基必须在资格与对偶行定价前拒绝，不能借新增路径接受BASICLU补单位列");
  for (const bool upper : {false, true}) {
    const auto wrong = warm_dual_fixture(warm_dual_model({upper ? 1.0 : -1.0, 0}, {0, 0}, {2, infinity},
        {{1, upper ? 1.0 : -1.0}}, {1}, {1}), {upper ? -2 : -1, 0}, {-1});
    const auto result = warm_dual_run(wrong);
    require(result.status == warm_dual::Status::dual_infeasible && !result.stats.dual_entered &&
        result.stats.dual_checks_completed == 1 && result.stats.dual_pricing_passes == 0 &&
        result.stats.initial_checks && result.stats.initial_checks->nonbasic_dual_ratio > 1 &&
        result.stats.factor.completed == 1 && result.stats.update.started == 0 && result.x.empty() && result.y.empty(),
        "基本站立条件可合格，但lower负r或upper正r的完整非基资格必须拒绝，不能偷跑PhaseI");
  }
  ++mapping_cases;
  details.push_back({"warm_dual_qualification_fixed", "mapping"});
}

void check_warm_dual_ratio_limits() {
  const auto tie = warm_dual_fixture(warm_dual_model({1, 1, 0}, {0, 0, 0}, {infinity, infinity, infinity},
      {{1, 1, -1}}, {1}, {1}), {-1, -1, 0}, {-1});
  const auto chosen = warm_dual_run(tie);
  warm_dual_success(tie, chosen, {1, 0, 0}, 1, 1);
  require(chosen.stats.last_entering == 0 && chosen.stats.last_leaving == 2, "入基精确比值tie须选最低全局列，不靠遍历偶然不同基");
  const auto blocks = warm_dual_fixture(warm_dual_model({1, 1, 0, 0}, {0, 0, 0, 0},
      {infinity, infinity, infinity, infinity}, {{1, 0, -1, 0}, {0, 1, 0, -1}}, {1, 1}, {1, 1}), {-1, -1, 0, 0}, {-1, -1});
  for (std::uint64_t cap : {0, 1, 2}) {
    const auto result = warm_dual_run(blocks, cap);
    if (cap == 2) warm_dual_success(blocks, result, {1, 1, 0, 0}, 2, 2);
    else require(result.status == warm_dual::Status::step_limit && result.stats.dual_pivots == cap &&
        result.stats.steps_completed == cap && result.stats.factor.completed == 1 && result.x.empty() && result.y.empty(),
        "共享cap0/1必须真实限制已完成pivot，失败不能发布部分修复点");
    if (cap == 1) require(result.stats.last_entering == 0 && result.stats.last_leaving == 2,
        "相同绝对离界量的out tie须选最低全局基本列");
  }
  auto larger_model = blocks.model; larger_model.row_lower[1] = larger_model.row_upper[1] = 2;
  const auto larger = warm_dual_fixture(std::move(larger_model), blocks.variable_status, blocks.row_status, blocks.seed);
  const auto largest = warm_dual_run(larger, 1);
  require(largest.status == warm_dual::Status::step_limit && largest.stats.last_entering == 1 && largest.stats.last_leaving == 3,
      "离基优先最大真实界违反，不能无条件按最低列选行");
  const auto none = warm_dual_fixture(warm_dual_model({0}, {0}, {infinity}, {{1}}, {-1}, {-1}), {0}, {-1});
  const auto unrepaired = warm_dual_run(none);
  require(unrepaired.status == warm_dual::Status::no_finite_step && unrepaired.stats.dual_entered &&
      unrepaired.stats.dual_pricing_passes == 1 && unrepaired.stats.dual_pivots == 0 &&
      unrepaired.stats.verification_started == 0 && unrepaired.x.empty() && unrepaired.y.empty(),
      "没有合法入列只报告未修复，不能发布不可行证书或把初始越界点送最终验收");
  ++transaction_cases;
  details.push_back({"warm_dual_ratio_limits", "transaction"});
}

void check_warm_dual_budget_prefix() {
  const auto input = warm_dual_overshoot();
  const auto model_before = input.model;
  const auto seed_before = input.seed;
  std::size_t callbacks = 0;
  const auto complete = warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
      warm_dual::Limits{64, true}, {}, {}, [&] { ++callbacks; });
  warm_dual_success(input, complete, {1, 1}, 3, 2, 2);
  bool saw_checked_not_entered = false, saw_row_solve = false;
  bool saw_update_partial = false, saw_verification = false;
  for (std::size_t cutoff = 0; cutoff < callbacks; ++cutoff) {
    std::size_t reached = 0;
    const auto result = warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
        warm_dual::Limits{64, true}, {}, {}, [&] {
          if (reached++ == cutoff) throw warm_dual::Failure(warm_dual::Status::time_limit, "对偶暖基实际预算截止");
        });
    const auto& s = result.stats;
    require(reached == cutoff + 1 && result.status == warm_dual::Status::time_limit && result.x.empty() && result.y.empty(),
        "每个实际成功轨迹回调截止都须保真实time_limit而不泄漏半候选，不能伪造内部原生可抢占");
    for (const auto* call : {&s.init, &s.factor, &s.solve_n, &s.solve_t, &s.ftran, &s.btran, &s.update, &s.dual_row_solve})
      require(call->completed <= call->started && call->started - call->completed <= 1, "实际库调用账本不能为预算异常回滚");
    require(s.dual_checks_completed <= s.dual_checks_started && s.dual_steps_completed <= s.dual_steps_started &&
        s.dual_steps_completed <= s.dual_pivots && s.dual_pivots <= s.dual_steps_started &&
        s.verification_completed <= s.verification_started, "资格、换基和验收须各自保真实partial完成边界");
    saw_checked_not_entered = saw_checked_not_entered || (s.dual_checks_completed > 0 && !s.dual_entered);
    saw_row_solve = saw_row_solve || (s.dual_row_solve.completed > 0 && s.dual_pivots == 0);
    saw_update_partial = saw_update_partial || (s.update.completed > s.dual_pivots);
    saw_verification = saw_verification || s.verification_completed > 0;
  }
  // 小夹具没有资格循环内部的4096项检查点，全部实际回调不等于遍历每项算术。
  require(saw_checked_not_entered && saw_row_solve && saw_update_partial && saw_verification,
      "全实际prefix须覆盖完整资格未进入、行BTRAN已付、update已返未记pivot及终验已付的不同阶段");
  warm_dual_success(input, warm_dual_run(input), {1, 1}, 3, 2, 2);
  require(same_model(input.model, model_before) && v118_same_bits(input.seed, seed_before) && complete.x == std::vector<double>({1, 1}),
      "所有失败后连续再调用须从拥有的原输入重建，旧成功输出和seed不能被失败LU污染");
  ++budget_cases;
  details.push_back({"warm_dual_budget_prefix", "budget"});
}
// 自主对偶暖基组件结束

// 对偶Harris两遍比值组件开始
constexpr double harris_tau = 0x1p-24;
solver::LpOptimalityOptions harris_options() {
  solver::LpOptimalityOptions options;
  // 仅组件采用可精确表示的原dual容差；两遍和后验用同一对象，gap门槛保持默认。
  options.dual_absolute = harris_tau; options.dual_relative = 0;
  return options;
}

V120WarmInput harris_pair(double second_cost = 2) {
  return warm_dual_fixture(warm_dual_model({1, second_cost, 0}, {0, 0, 0}, {1, infinity, infinity},
      {{1, 2, -1}}, {1}, {1}), {-1, -1, 0}, {-1});
}

warm_dual::Result harris_run(const V120WarmInput& input, std::uint64_t cap = 64,
    const solver::LpOptimalityOptions& options = harris_options()) {
  return warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
      warm_dual::Limits{cap, true}, {}, options);
}

void harris_success(const V120WarmInput& input, const warm_dual::Result& result,
    const std::vector<double>& expected, double objective, std::uint64_t pivots = 1,
    const solver::LpOptimalityOptions& options = harris_options()) {
  warm_dual_success(input, result, expected, objective, pivots);
  require(quality_ok(solver::verify_lp_solution(input.model, result.x, result.y, {}, options)),
      "Harris候选仍须用本次原dual及gap容差在原Model独立重验，不能依赖更宽的后验");
  const auto& s = result.stats;
  require(s.dual_ratio_passes_started == 2 * pivots && s.dual_ratio_passes == 2 * pivots &&
      s.dual_pricing_passes == pivots && s.dual_band_candidates >= pivots &&
      s.dual_tableau_entries_peak == input.prepared.costs.size() + input.prepared.rhs.size(),
      "每个成功对偶pivot须实际完成两遍，缓存峰值只计本次完整表行而非假定免费扫描");
}

void check_warm_harris_pivot_stability() {
  const auto input = harris_pair();
  const auto result = harris_run(input);
  harris_success(input, result, {0, 0.5, 0}, 1);
  require(result.stats.last_entering == 1 && result.stats.last_leaving == 2 &&
      result.stats.last_dual_theta == 1 && result.stats.last_dual_selected_alpha == 2 &&
      result.stats.last_pivot == -2 && result.stats.dual_band_candidates == 2,
      "相同精确比值必须选择较大的实际表主元，不能继续只按最低全局列入基");
  const auto equal_alpha = warm_dual_fixture(warm_dual_model({1 + harris_tau / 2, 1, 0}, {0, 0, 0},
      {1, 1, infinity}, {{1, 1, -1}}, {1}, {1}), {-1, -1, 0}, {-1});
  const auto tied = harris_run(equal_alpha);
  harris_success(equal_alpha, tied, {1, 0, 0}, 1 + harris_tau / 2);
  require(tied.stats.last_entering == 0 && tied.stats.last_dual_selected_alpha == 1 &&
      tied.stats.last_dual_theta == 1 + static_cast<long double>(harris_tau) / 2,
      "band内主元相同才按最低全局列，不能偷偷再以较小精确比值改变已定义tie规则");
  ++transaction_cases;
  details.push_back({"warm_harris_pivot_stability", "transaction"});
}

void check_warm_harris_tolerance_band() {
  const auto inside = harris_pair(2 + harris_tau);
  const auto selected = harris_run(inside);
  harris_success(inside, selected, {0, 0.5, 0}, 1 + harris_tau / 2);
  require(selected.stats.last_entering == 1 && selected.stats.last_dual_theta == 1 + static_cast<long double>(harris_tau) / 2 &&
      selected.stats.last_dual_relaxed_theta == 1 + static_cast<long double>(harris_tau) &&
      selected.stats.last_dual_selected_alpha == 2 && selected.stats.dual_band_candidates == 2 &&
      selected.stats.last_checks->nonbasic_dual_violation == static_cast<long double>(harris_tau) / 2 &&
      selected.stats.last_checks->nonbasic_dual_ratio == 0.5L,
      "较大精确比值在原tol band内须真正选较强列；重建负r仍只能消耗同一个原tau的一半");
  const auto outside = harris_pair(2 + 4 * harris_tau);
  const auto rejected_column = harris_run(outside);
  harris_success(outside, rejected_column, {1, 0, 0}, 1);
  require(rejected_column.stats.last_entering == 0 && rejected_column.stats.last_dual_theta == 1 &&
      rejected_column.stats.last_dual_selected_alpha == 1 && rejected_column.stats.dual_band_candidates == 1,
      "band外更强列不得入选，不能寄望后验拒绝后再用额外工作掩盖错误选择");
  auto relative = harris_options(); relative.dual_absolute = 0; relative.dual_relative = harris_tau;
  const auto relative_result = harris_run(inside, 64, relative);
  harris_success(inside, relative_result, {0, 0.5, 0}, 1 + harris_tau / 2, 1, relative);
  const long double expected_ratio = 0.5L / (1 + static_cast<long double>(harris_tau) / 2);
  require(relative_result.stats.last_entering == 1 &&
      std::abs(relative_result.stats.last_checks->nonbasic_dual_ratio - expected_ratio) <=
          8 * std::numeric_limits<long double>::epsilon(),
      "相对dual容差须用新pi实际乘积重算，不能复用旧tau或额外扩大验收门槛");
  ++mapping_cases;
  details.push_back({"warm_harris_tolerance_band", "mapping"});
}

void check_warm_harris_tiny_directions() {
  constexpr double tiny = 1e-12;
  const auto necessary = warm_dual_fixture(warm_dual_model({tiny, 0}, {0, 0}, {infinity, infinity},
      {{tiny, -1}}, {1}, {1}), {-1, 0}, {-1});
  const auto small_pivot = harris_run(necessary);
  harris_success(necessary, small_pivot, {1e12, 0}, 1);
  require(small_pivot.stats.last_dual_selected_alpha == static_cast<long double>(tiny) &&
      small_pivot.stats.last_dual_theta == 1 && small_pivot.stats.last_dual_primal_step &&
      *small_pivot.stats.last_dual_primal_step > 1e11L && small_pivot.stats.dual_band_candidates == 1,
      "唯一必要的1e-12主元必须真实换基，不能加人工tiny阈值删除仍可由BASICLU处理的候选");
  const auto upper = warm_dual_fixture(warm_dual_model({-2, 0}, {0, 0}, {1, infinity},
      {{2, 1}}, {1}, {1}), {-2, 0}, {-1});
  const auto reverse = harris_run(upper);
  harris_success(upper, reverse, {0.5, 0}, -1);
  require(reverse.stats.last_dual_selected_alpha == 2 && reverse.stats.last_pivot == 2 &&
      reverse.stats.last_dual_primal_step == 0.5L,
      "上界非基须沿负向进入，但BASICLU更新仍接收原始正主元而非sigma乘过的符号");
  for (const bool negative : {false, true}) {
    const auto free = warm_dual_fixture(warm_dual_model({0, 0}, {-infinity, 0}, {infinity, infinity},
        {{2, negative ? 1.0 : -1.0}}, {negative ? -1.0 : 1.0}, {negative ? -1.0 : 1.0}),
        {-3, 0}, {-1}, {negative ? 2.0 : -2.0, 0});
    const auto result = harris_run(free);
    harris_success(free, result, {negative ? -0.5 : 0.5, 0}, 0);
    require(result.stats.last_dual_selected_alpha == 2 && result.stats.last_dual_theta == 0 &&
        result.stats.last_dual_primal_step == 2.5L && result.stats.zero_steps == 0 && result.stats.dual_zero_steps == 1,
        "两遍选择仍须允许自由变量双向及非零seed，dual价零不表示原始位移或真实pivot为零");
  }
  ++transaction_cases;
  details.push_back({"warm_harris_tiny_directions", "transaction"});
}

void check_warm_harris_limits_no_candidate() {
  const auto blocks = warm_dual_fixture(warm_dual_model({1, 1, 0, 0}, {0, 0, 0, 0},
      {infinity, infinity, infinity, infinity}, {{1, 0, -1, 0}, {0, 1, 0, -1}}, {1, 1}, {1, 1}), {-1, -1, 0, 0}, {-1, -1});
  for (std::uint64_t cap : {0, 1, 2}) {
    const auto result = harris_run(blocks, cap);
    if (cap == 2) harris_success(blocks, result, {1, 1, 0, 0}, 2, 2);
    else require(result.status == warm_dual::Status::step_limit && result.x.empty() && result.y.empty() &&
        result.stats.steps_completed == cap && result.stats.dual_ratio_passes == 2 * cap,
        "两遍定价不能绕过原共享cap0/1，也不能把只有部分修复的点发布");
    if (cap == 1) require(result.stats.last_entering == 0 && result.stats.last_leaving == 2,
        "Harris不改变相同离界量的全局离基tie规则");
  }
  const auto none = warm_dual_fixture(warm_dual_model({0}, {0}, {infinity}, {{1}}, {-1}, {-1}), {0}, {-1});
  const auto result = harris_run(none);
  require(result.status == warm_dual::Status::no_finite_step && result.x.empty() && result.y.empty() &&
      result.stats.dual_ratio_passes_started == 2 && result.stats.dual_ratio_passes == 2 &&
      result.stats.dual_band_candidates == 0 && result.stats.last_dual_relaxed_theta &&
      std::isinf(*result.stats.last_dual_relaxed_theta) && !result.stats.last_dual_selected_alpha &&
      result.stats.update.started == 0 && result.stats.verification_started == 0,
      "没有合法候选时仍诚实计两遍扫描，保无解候选而不宣布不可行证书或制造selected主元");
  ++transaction_cases;
  details.push_back({"warm_harris_limits_no_candidate", "transaction"});
}

void check_warm_harris_budget_prefix() {
  const auto input = harris_pair();
  const auto options = harris_options();
  std::size_t callbacks = 0;
  const auto complete = warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
      warm_dual::Limits{64, true}, {}, options, [&] { ++callbacks; });
  harris_success(input, complete, {0, 0.5, 0}, 1);
  bool saw_cache = false, saw_first = false, saw_second = false, saw_selected = false;
  for (std::size_t cutoff = 0; cutoff < callbacks; ++cutoff) {
    std::size_t reached = 0;
    const auto result = warm_dual::run(warm_dual_input(input), solver::Clock::now() + std::chrono::seconds(5),
        warm_dual::Limits{64, true}, {}, options, [&] {
          if (reached++ == cutoff) throw warm_dual::Failure(warm_dual::Status::time_limit, "Harris真实预算截止");
        });
    const auto& s = result.stats;
    require(reached == cutoff + 1 && result.status == warm_dual::Status::time_limit && result.x.empty() && result.y.empty() &&
        s.dual_ratio_passes <= s.dual_ratio_passes_started && s.dual_ratio_passes_started - s.dual_ratio_passes <= 1 &&
        s.dual_ratio_passes_started <= 2 * s.dual_steps_started && s.dual_tableau_entries_peak <= 4 &&
        s.dual_band_candidates <= 2 && s.update.completed <= s.update.started,
        "每个实际预算前缀须留两遍、缓存及候选的真实部分工作，失败不得发布方向或回滚计数");
    saw_cache = saw_cache || (s.dual_tableau_entries_peak == 4 && s.dual_ratio_passes_started == 0);
    saw_first = saw_first || (s.dual_ratio_passes == 1 && s.last_dual_relaxed_theta && !s.last_dual_selected_alpha);
    saw_second = saw_second || (s.dual_ratio_passes == 2 && s.last_dual_selected_alpha &&
        !s.last_entering && !s.last_dual_theta);
    saw_selected = saw_selected || (s.last_entering == 1 && s.last_dual_theta == 1 && s.ftran.started == 0);
  }
  require(saw_cache && saw_first && saw_second && saw_selected,
      "实际回调须覆盖cache已分配、首遍band已存、二遍alpha已存但旧快照未交付及FTRAN前的各边界");
  // 小例两遍内部没有4096项检查点；只声称遍历真实存在的全部回调。
  harris_success(input, harris_run(input), {0, 0.5, 0}, 1);
  require(complete.x == std::vector<double>({0, 0.5, 0}), "中断和后继调用不能污染已拥有的成功结果");
  ++budget_cases;
  details.push_back({"warm_harris_budget_prefix", "budget"});
}
// 对偶Harris两遍比值组件结束

solver::Model normal_mapping_model(bool all_singletons) {
  solver::Model model;
  model.name = "normal_mapping_component";
  model.costs = {-1, 2}; model.column_lower = {-infinity, -infinity};
  model.column_upper = {infinity, infinity};
  model.kinds.assign(2, solver::VariableKind::continuous);
  if (all_singletons) {
    model.row_lower = {-infinity, -infinity}; model.row_upper = {3, 5};
    model.column_start = {0, 1, 2}; model.row_index = {0, 1}; model.values = {1, -2};
  } else {
    model.row_lower = {-infinity, -2, -infinity}; model.row_upper = {3, -2, 5};
    model.column_start = {0, 3, 5}; model.row_index = {0, 1, 2, 0, 1}; model.values = {1, 3, 5, 2, -1};
  }
  // 用二进制精确缩放控制解析夹具的正规方程消减误差，保留全部映射结构。
  // 此处检验映射与恢复的精度，不给原近似预条件器增加任意病态系统的精确稠密解合同。
  for (double& value : model.values) value *= 0x1p-8;
  return model;
}

void check_normal_mapped_kkt_physical() {
  for (bool all_singletons : {false, true}) for (bool eliminate : {false, true}) for (bool equilibrate : {false, true}) {
    const auto model = normal_mapping_model(all_singletons);
    auto options = policy();
    options.ipm.scaling_passes = 0; options.ipm.eliminate_singletons = eliminate;
    options.ipm.equilibrate_kkt = equilibrate;
    const auto start = solver::Clock::now();
    const auto cone = solver::canonicalize(model, options, start);
    const auto n = cone.costs.size(), m = cone.rhs.size();
    require(n == 2 && m == (all_singletons ? 2U : 3U), "真实canonicalize须保持本夹具的完整变量与行");
    auto result = candidate::make_shell(model);
    solver::KktFactor factor(cone, options, start, result);
    std::vector<double> h(m);
    std::vector<std::size_t> degree(m, 0);
    for (auto index : cone.rows) ++degree[static_cast<std::size_t>(index)];
    std::int64_t eliminated = 0;
    for (std::size_t row_index = 0; row_index < m; ++row_index) {
      h[row_index] = cone.inequality[row_index] ? 2 + static_cast<double>(row_index) : 0;
      if (eliminate && cone.inequality[row_index] && degree[row_index] == 1) ++eliminated;
    }
    for (int epoch = 0; epoch < 2; ++epoch) {
      if (epoch) for (double& value : h) value *= 1.5;
      factor.factor(h, options, start, result, 7);
      require(result.ipm_factor_backend == "normal_cholmod_aat" &&
          *result.ipm_eliminated_singletons == eliminated &&
          *result.ipm_kkt_dimension == static_cast<std::int64_t>(m) - eliminated,
          "实际融合路径须使用Normal、外层AMD及指定单例消元，不能结构回退绕过测试");
      const auto stored = factor.v93_regularized_h();
      const long double primal_regularization = *result.ipm_primal_regularization;
      // 只用完整原G和该次实际存储r/R形成名义算子，不复用被测Schur约化或映射。
      const auto product = [&](const std::vector<double>& point) {
        std::vector<long double> image(n + m, 0);
        for (std::size_t col = 0; col < n; ++col) image[col] = primal_regularization * point[col];
        for (std::size_t row_index = 0; row_index < m; ++row_index)
          image[n + row_index] = -static_cast<long double>(stored[row_index]) * point[n + row_index];
        for (std::size_t col = 0; col < n; ++col) for (auto k = cone.starts[col]; k < cone.starts[col + 1]; ++k) {
          const auto position = static_cast<std::size_t>(k), row_index = static_cast<std::size_t>(cone.rows[position]);
          image[col] += static_cast<long double>(cone.values[position]) * point[n + row_index];
          image[n + row_index] += static_cast<long double>(cone.values[position]) * point[col];
        }
        return image;
      };
      for (int seed : {1, 2}) {
        std::vector<double> expected(n + m);
        for (std::size_t i = 0; i < expected.size(); ++i)
          expected[i] = (i % 2 ? -1.0 : 1.0) * static_cast<double>((i + 1) * seed) / 4;
        const auto wide_rhs = product(expected);
        const std::vector<double> rhs(wide_rhs.begin(), wide_rhs.end()), snapshot = rhs;
        const auto answer = factor.solve_homogeneous_direct(rhs, options, start, result, false);
        require(rhs == snapshot && answer.size() == expected.size(), "KKT融合求解只能返回完整候选，不能写原rhs");
        const auto residual_product = product(answer);
        for (std::size_t i = 0; i < answer.size(); ++i) {
          const long double forward_difference = std::abs(static_cast<long double>(answer[i]) - expected[i]);
          const long double physical_residual = residual_product[i] - rhs[i];
          if (!(std::isfinite(answer[i]) && forward_difference <=
                  1e-10L * std::max(1.0L, std::abs(static_cast<long double>(expected[i])))) ||
              !(std::abs(physical_residual) <= 1e-10L * std::max(1.0L, std::abs(static_cast<long double>(rhs[i]))))) {
            std::fprintf(stderr, "Normal融合夹具诊断：all_singletons=%d eliminate=%d equilibrate=%d epoch=%d seed=%d i=%zu\n",
                static_cast<int>(all_singletons), static_cast<int>(eliminate), static_cast<int>(equilibrate), epoch, seed, i);
            std::fprintf(stderr, "实际=%.17g 解析=%.17g 绝对误差=%.17g 原方程残差=%.17g rhs=%.17g 乘积=%.17g r=%.17g\n",
                answer[i], expected[i], static_cast<double>(forward_difference), static_cast<double>(physical_residual),
                rhs[i], static_cast<double>(residual_product[i]), static_cast<double>(primal_regularization));
          }
          require(std::isfinite(answer[i]) && std::abs(static_cast<long double>(answer[i]) - expected[i]) <=
              1e-10L * std::max(1.0L, std::abs(static_cast<long double>(expected[i]))),
              "混合/全单例及均衡开关的融合结果须恢复独立解析向量");
          require(std::abs(residual_product[i] - rhs[i]) <=
              1e-10L * std::max(1.0L, std::abs(static_cast<long double>(rhs[i]))),
              "融合映射与单例恢复须通过完整名义KKT方程检查");
        }
      }
    }
    require(*result.ipm_factorizations == 2 && *result.ipm_homogeneous_direction_solves == 4,
        "重复RHS和重分解须完整计费，不能误用旧数值映射或静默重试");
  }
  ++mapping_cases;
  details.push_back({"normal_mapped_kkt_physical", "mapping"});
}

void check_normal_mapped_kkt_failure_budget() {
  WarmRegularizationSystem system;
  const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5}, saved = rhs;
  const auto snapshot = system.result;
  auto expired = system.options; expired.time_limit = 0;
  bool stopped = false;
  try { system.factor->solve_homogeneous_direct(rhs, expired, system.start, system.result, false); }
  catch (const solver::BudgetExpired&) { stopped = true; }
  same_non_timing(system.result, snapshot);
  require(stopped && rhs == saved, "融合KKT的过期入口须保持原rhs及未启动的方向账本");
  auto invalid = rhs; invalid[0] = infinity;
  const auto old_count = *system.result.ipm_homogeneous_direction_solves;
  bool rejected = false;
  try { system.factor->solve_homogeneous_direct(invalid, system.options, system.start, system.result, false); }
  catch (const solver::NumericalFailure&) { rejected = true; }
  require(rejected && std::isinf(invalid[0]) && *system.result.ipm_homogeneous_direction_solves == old_count + 1,
      "非有限融合输入不得成为解，已经启动的方向调用必须保留计数");
  const auto recovered = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false);
  require(rhs == saved && recovered.size() == rhs.size() &&
      std::all_of(recovered.begin(), recovered.end(), [](double value) { return std::isfinite(value); }),
      "输入失败后当前成功因子仍须支持新的完整有限回代");
  ++budget_cases;
  details.push_back({"normal_mapped_kkt_failure_budget", "budget"});
}

std::pair<std::vector<double>, long double> factor_reuse_test_constant(const solver::ConeModel& cone,
    const solver::KktFactor& factor, const solver::detail::IpmHomogeneousScaling& scaling,
    const solver::SolveOptions& options, solver::Clock::time_point start, solver::SolveResult& result,
    std::int64_t iteration) {
  auto* log = coupled_state::current();
  require(log != nullptr, "常量探测组件须有真实联立计费作用域");
  const auto rhs = solver::detail::ipm_homogeneous_scaled_constant_rhs(scaling, cone.costs, cone.rhs);
  ++log->constant_calls_started;
  auto constant = factor.solve_homogeneous_direct(rhs, options, start, result, true);
  ++log->constant_calls_completed;
  const auto denominator = solver::v100_energy_denominator(factor, cone.costs.size(), constant, result, {}, iteration);
  return {std::move(constant), denominator};
}

void check_factor_reuse_current_physics() {
  using namespace solver::detail;
  CoupledScalarSystem system;
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  const auto old_regularization = system.result.ipm_primal_regularization;
  coupled_state::Log log; log.enabled = true;
  coupled_state::Scope scope(&log);
  energy_state::Log energy; energy.enabled = true;
  energy_state::Scope energy_scope(&energy);
  const auto old_constant = factor_reuse_test_constant(system.cone, *system.factor, system.scaling,
      system.options, system.start, system.result, 1);
  const std::vector<double> slack{2}, dual{1};
  const auto current = make_ipm_homogeneous_scaling(0, {true}, slack, dual, 2, 8);
  require(system.scaling.d == 1 && current.d == 2 && current.lambda_tau == 4,
      "解析跨点夹具须真实改变d与齐次互补尺度");
  const auto current_constant = factor_reuse_test_constant(system.cone, *system.factor, current,
      system.options, system.start, system.result, 2);
  require(current_constant.first.size() == 1 && current_constant.first[0] == old_constant.first[0] / 2 &&
      current_constant.second != old_constant.second && *system.result.ipm_factorizations == 1 &&
      system.result.ipm_primal_regularization == old_regularization &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows,
      "新d必须重新支付常量解，旧factor/r/R在探测前后保持同一成功epoch");
  const long double v = current_constant.first[0];
  energy_near(current_constant.second, 1 + static_cast<long double>(old_rows[0]) * v * v,
      "旧因子常量的正能量必须使用旧R，不能混入当前s/z");
  require(std::abs(current_constant.second - (1 + 2 * v * v)) > 0.1L,
      "本夹具必须能识别误将当前对角2与旧因子混用的能量");
  const auto explicit_product = [&](const auto& value) {
    return std::vector<long double>{static_cast<long double>(value[1]) - value[2],
        (static_cast<long double>(value[0]) + value[3]) / 2,
        (static_cast<long double>(value[1]) + 2 * value[0]) / current.lambda[0],
        2 * static_cast<long double>(value[2]) + 0.5L * value[3]};
  };
  for (std::size_t coordinate = 0; coordinate < current.dimension(); ++coordinate) {
    std::vector<double> basis(current.dimension(), 0); basis[coordinate] = 1;
    std::vector<long double> product;
    solver::v94_hsd_product(current, system.cone.starts, system.cone.rows, system.cone.values,
        system.cone.costs, system.cone.rhs, slack, dual, 2, 8, std::span<const double>(basis), product, {});
    require(product == explicit_product(basis), "跨点真验的每个物理矩阵列都须来自当前s/z/tau/kappa");
  }
  const std::vector<double> rhs{1, 0, 0, 0};
  const auto solve_current = [&](const auto& constant) {
    const auto precondition = [&](std::span<const double> input, bool refinement) {
      const auto reduced = system.factor->solve_homogeneous_direct(
          ipm_homogeneous_scaled_reduced_rhs(current, input), system.options, system.start, system.result,
          false, refinement);
      return ipm_homogeneous_scaled_recover(current, system.cone.costs, system.cone.rhs,
          input, reduced, constant.first, constant.second);
    };
    const auto product = [&]<class Scalar>(std::span<const Scalar> value, std::vector<long double>& image) {
      ++*system.result.ipm_homogeneous_scaled_products;
      solver::v94_hsd_product(current, system.cone.starts, system.cone.rows, system.cone.values,
          system.cone.costs, system.cone.rhs, slack, dual, 2, 8, value, image, {});
    };
    return solver::v100_solve_coupled(current, rhs, precondition, product, product, {}, "corrector", 2,
        nullptr, true, true, true);
  };
  bool rejected = false;
  try { solve_current(current_constant); }
  catch (const coupled_kernel::InitialRejected&) { rejected = true; }
  require(rejected && log.last && std::string_view(log.last->work.status) == "initial_rejected" &&
      log.last->work.preconditioner_completed == 1 && log.last->work.true_checks_completed == 1 &&
      log.last->work.krylov_started == 0 && log.last->work.initial_forcing &&
      log.last->work.initial_forcing->measure && *log.last->work.initial_forcing->measure > 1 &&
      *system.result.ipm_factorizations == 1,
      "旧K不能通过当前点物理方程时须只付一次P/真验后拒绝，而非用旧方程或启动FG");
  system.factor->factor({2}, system.options, system.start, system.result);
  const auto fresh_constant = factor_reuse_test_constant(system.cone, *system.factor, current,
      system.options, system.start, system.result, 2);
  const auto answer = solve_current(fresh_constant);
  coupled_independent_equations(current, rhs, answer, explicit_product);
  require(log.last && std::string_view(log.last->work.status) == "accepted" &&
      log.last->work.krylov_started == 0 && *system.result.ipm_factorizations == 2,
      "当前对角重新分解后，同一右端须由真实五块strict检查通过且完整计入新分解");

  // 再覆盖真实Normal、外层均衡和凝聚单例；d变化仍不更新其已付数值快照。
  WarmRegularizationSystem normal;
  const auto normal_rows = std::vector<double>(normal.factor->v93_regularized_h().begin(),
      normal.factor->v93_regularized_h().end());
  const auto normal_r = normal.result.ipm_primal_regularization;
  const std::vector<double> normal_slack{7, 0, 11}, normal_dual{1, -3, 2};
  const auto before_scaling = make_ipm_homogeneous_scaling(2, normal.cone.inequality,
      normal_slack, normal_dual, 1, 1);
  const auto after_scaling = make_ipm_homogeneous_scaling(2, normal.cone.inequality,
      normal_slack, normal_dual, 2, 8);
  const auto before = factor_reuse_test_constant(normal.cone, *normal.factor, before_scaling,
      normal.options, normal.start, normal.result, 3);
  const auto after = factor_reuse_test_constant(normal.cone, *normal.factor, after_scaling,
      normal.options, normal.start, normal.result, 4);
  for (std::size_t i = 0; i < after.first.size(); ++i)
    require(after.first[i] == before.first[i] / 2,
        "带均衡和单例的同一旧Normal因子也必须按当前d重算完整常量方向");
  energy_near(after.second - 1, (before.second - 1) / 4,
      "同一旧r/R下常量缩半，名义二次能量应相应缩为四分之一");
  require(*normal.result.ipm_factorizations == 1 && normal.result.ipm_primal_regularization == normal_r &&
      *normal.result.ipm_eliminated_singletons == 1 && normal.options.ipm.equilibrate_kkt &&
      std::vector<double>(normal.factor->v93_regularized_h().begin(), normal.factor->v93_regularized_h().end()) == normal_rows,
      "结构和当前数值不得因只读探测而重分解或混入新点对角");
  auto mixed_epoch = normal.result;
  mixed_epoch.ipm_primal_regularization = 2 * *normal_r;
  bool wrong_owner = false;
  try { solver::v100_energy_denominator(*normal.factor, 2, after.first, mixed_epoch, {}, 4); }
  catch (const std::invalid_argument&) { wrong_owner = true; }
  require(wrong_owner && normal.factor->nominal_primal_regularization() == *normal_r &&
      normal.result.ipm_primal_regularization == normal_r,
      "有限但不同epoch的公共r必须作为输入错误拒绝，不能与旧factor拼成可恢复数值候选");
  ++mapping_cases;
  details.push_back({"factor_reuse_current_physics", "mapping"});
}

struct FactorReusePairFixture {
  CoupledScalarSystem system;
  solver::Point point;
  double tau = 1, kappa = 1;
  solver::detail::IpmHomogeneousScaling scaling;
  std::vector<double> h{-1};
  coupled_state::Log main;
  energy_state::Log energy;
  solver::detail::factor_reuse::Log reuse;
  coupled_state::Scope coupled_scope;
  energy_state::Scope energy_scope;
  std::function<void()> after_affine;
  int affine_calls = 0, corrector_calls = 0;

  explicit FactorReusePairFixture(bool stale)
      : point(system.point), scaling(system.scaling), coupled_scope(&main), energy_scope(&energy) {
    main.enabled = energy.enabled = reuse.enabled = true;
    if (stale) {
      point.s[0] = 2; tau = 2; kappa = 8;
      scaling = solver::detail::make_ipm_homogeneous_scaling(0, {true}, point.s, point.z, tau, kappa);
    }
  }

  solver::HomogeneousDirection pair(bool initial_only) {
    const auto constant = factor_reuse_test_constant(system.cone, *system.factor, scaling,
        system.options, system.start, system.result, 2);
    ++affine_calls;
    // 用两个解析右端隔离整组事务；每次P和真验仍实际执行，不冒称预测器sigma公式测试。
    auto affine = solver::homogeneous_scaled_direction(system.cone, point, tau, kappa, {0}, {}, 0,
        0, {0}, 0, constant.first, scaling, constant.second, *system.factor,
        system.options, system.start, system.result, "affine", 2, initial_only);
    require(affine.v94_wide.has_value(), "首阶段须真实完成物理宽方向交付");
    h[0] = 0;
    if (after_affine) after_affine();
    ++corrector_calls;
    return solver::homogeneous_scaled_direction(system.cone, point, tau, kappa, {-1}, {}, 0,
        1, h, 0, constant.first, scaling, constant.second, *system.factor,
        system.options, system.start, system.result, "corrector", 2, initial_only);
  }

  void selected_pair_is_complete() const {
    require(main.constant_calls_started == 1 && main.constant_calls_completed == 1 &&
        main.directions_started == 2 && main.directions_completed == 2 &&
        main.preconditioner_started == 2 && main.preconditioner_completed == 2 &&
        main.products_started == 2 && main.products_completed == 2 &&
        main.true_checks_started == 2 && main.true_checks_completed == 2 && main.krylov_started == 0 &&
        energy.started == 1 && energy.completed == 1 && main.last &&
        std::string_view(main.last->role) == "corrector",
        "主账本只能包含完整选定pair的一次常量与两次真实方向，不能合并部分探测");
  }
};

void check_factor_reuse_pair_transaction() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (bool stale : {false, true}) {
    FactorReusePairFixture fixture(stale);
    const auto before = reuse_state::counts(fixture.system.result);
    const auto saved_h = fixture.h, saved_s = fixture.point.s, saved_z = fixture.point.z;
    const auto answer = solver::v122_try_reuse_pair(fixture.h,
        [&](bool initial_only) {
          require(initial_only && coupled_state::current() != &fixture.main &&
              energy_state::current() != &fixture.energy,
              "探测整组须在独立联立和能量日志执行，原主账本仍隔离");
          return fixture.pair(initial_only);
        }, fixture.reuse, 2, 1, fixture.system.result);
    require(answer.has_value() != stale && fixture.reuse.events.size() == 1 &&
        coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy &&
        fixture.affine_calls == 1 && fixture.corrector_calls == 1 && fixture.point.s == saved_s &&
        fixture.point.z == saved_z && fixture.tau == (stale ? 2 : 1) && fixture.kappa == (stale ? 8 : 1),
        "只有完整pair能返回；失败与成功都必须恢复外层上下文且不能提前更新当前点");
    const auto& event = fixture.reuse.events[0];
    const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
    require(event.attempt_iteration == 2 && event.factor_epoch == 1 &&
        event.public_delta.constant_solves == 1 && event.public_delta.direction_solves == 2 &&
        event.public_delta.scaled_products == 2 && event.public_delta.factorizations == 0 &&
        paid.constant_solves == event.public_delta.constant_solves && paid.direction_solves == event.public_delta.direction_solves &&
        paid.scaled_products == event.public_delta.scaled_products && paid.factorizations == 0,
        "成功和被拒绝探测都须保真实公共费用，且不得在探测中支付新factor或减回旧费用");
    if (!stale) {
      require(std::string_view(event.status) == "accepted" && fixture.reuse.reused_steps == 1 &&
          fixture.h == std::vector<double>({0}), "整pair成功才保留校正h并计一次复用");
      fixture.selected_pair_is_complete();
    } else {
      require(std::string_view(event.status) == "initial_rejected" && fixture.reuse.reused_steps == 0 &&
          fixture.h == saved_h && fixture.main.constant_calls_started == 0 && fixture.main.directions_started == 0 &&
          fixture.energy.started == 0 && event.coupled.directions_started == 2 &&
          event.coupled.directions_completed == 1 && event.coupled.last &&
          std::string_view(event.coupled.last->role) == "corrector" &&
          std::string_view(event.coupled.last->work.status) == "initial_rejected",
          "corrector拒绝须丢弃整个旧affine/h组，原主日志保持空而失败前缀完整留存");
      ++fixture.reuse.fresh_started;
      fixture.system.factor->factor({2}, fixture.system.options, fixture.system.start, fixture.system.result);
      ++fixture.reuse.fresh_completed;
      require(fixture.h == saved_h, "fresh重做前必须仍是原affine h，不能沿用被拒的corrector右端");
      const auto fresh = fixture.pair(false);
      require(fresh.v94_wide && fixture.affine_calls == 2 && fixture.corrector_calls == 2 &&
          *fixture.system.result.ipm_factorizations == 2 && *fixture.system.result.ipm_homogeneous_constant_solves == 2 &&
          *fixture.system.result.ipm_homogeneous_direction_solves == 4 && *fixture.system.result.ipm_homogeneous_scaled_products == 4,
          "拒绝后须真实新分解并重做两方向，失败探测和fresh所有实际工作都保留");
      fixture.selected_pair_is_complete();
    }
  }
  ++transaction_cases;
  details.push_back({"factor_reuse_pair_transaction", "transaction"});
}

void check_factor_reuse_budget_scope_serializer() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (const auto [iteration, epoch] : std::array<std::array<std::int64_t, 2>, 5>{{{1, 0}, {201, 1}, {2, 0}, {2, 2}, {2, 3}}}) {
    FactorReusePairFixture fixture(false);
    const auto saved_h = fixture.h;
    bool invalid = false;
    try {
      solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
          fixture.reuse, iteration, epoch, fixture.system.result);
    } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid && fixture.reuse.events.empty() && fixture.affine_calls == 0 && fixture.h == saved_h &&
        *fixture.system.result.ipm_homogeneous_constant_solves == 0,
        "非法当前迭代或因子epoch必须在创建探测和任何数值工作前拒绝");
  }
  for (int scope_kind = 0; scope_kind < 3; ++scope_kind) {
    FactorReusePairFixture fixture(false);
    fixture.reuse.enabled = scope_kind != 0;
    if (scope_kind == 1) fixture.main.enabled = false;
    if (scope_kind == 2) fixture.energy.enabled = false;
    bool invalid = false;
    try {
      solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
          fixture.reuse, 2, 1, fixture.system.result);
    } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid && fixture.reuse.events.empty() && fixture.affine_calls == 0 &&
        *fixture.system.result.ipm_homogeneous_constant_solves == 0,
        "关闭探测或缺少任一计费作用域不能进入旧因子尝试");
  }
  int total = 0;
  {
    FactorReusePairFixture fixture(false);
    const auto answer = solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
        fixture.reuse, 2, 1, fixture.system.result, [&] { ++total; });
    require(answer && total > 0, "先遍历整pair助手实际存在的全部预算回调");
  }
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    FactorReusePairFixture fixture(false);
    const auto saved_h = fixture.h;
    int checks = 0;
    bool original = false;
    try {
      solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
          fixture.reuse, 2, 1, fixture.system.result,
          [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; });
    } catch (const solver::BudgetExpired&) { original = true; }
    require(original && checks == cutoff && fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
        fixture.main.directions_started == 0 && fixture.energy.started == 0 &&
        coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy,
        "每个真实助手预算中断都须保原异常、h和未合并的主账本");
    if (fixture.reuse.events.empty()) {
      require(cutoff == 1 && *fixture.system.result.ipm_homogeneous_constant_solves == 0,
          "最早预算失败不得创建未启动探测或虚报公共费用");
    } else {
      const auto& event = fixture.reuse.events.back();
      require(std::string_view(event.status) == "budget" && event.coupled.directions_completed == 2 &&
          event.public_delta.direction_solves == 2 && *fixture.system.result.ipm_homogeneous_direction_solves == 2,
          "末预算失败须保完整已付pair但不合并、不采用或回滚公共工作");
    }
  }
  for (int failure = 0; failure < 5; ++failure) {
    FactorReusePairFixture fixture(false);
    const auto saved_h = fixture.h;
    fixture.after_affine = [&] {
      if (failure == 0) throw solver::BudgetExpired{};
      if (failure == 1) throw std::bad_alloc{};
      if (failure == 2) throw std::invalid_argument("整组输入异常");
      if (failure == 3) throw std::runtime_error("整组原始异常");
      throw solver::NumericalFailure("旧因子数值拒绝");
    };
    bool propagated = false;
    std::optional<solver::HomogeneousDirection> answer;
    try {
      answer = solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
          fixture.reuse, 2, 1, fixture.system.result);
    } catch (const solver::BudgetExpired&) { propagated = failure == 0; }
    catch (const std::bad_alloc&) { propagated = failure == 1; }
    catch (const std::invalid_argument&) { propagated = failure == 2; }
    catch (const std::runtime_error&) { propagated = failure == 3; }
    const auto& event = fixture.reuse.events.back();
    require(propagated == (failure != 4) && !answer && fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
        fixture.main.directions_started == 0 && fixture.energy.started == 0 && event.coupled.directions_completed == 1 &&
        event.public_delta.constant_solves == 1 && event.public_delta.direction_solves == 1 &&
        event.public_delta.scaled_products == 1 && event.public_delta.factorizations == 0 &&
        fixture.affine_calls == 1 && fixture.corrector_calls == 0 &&
        std::string_view(event.status) == (failure == 0 ? "budget" : failure == 4 ? "numerical_error" : "exception"),
        "仅已知数值失败可回拒绝；其它类型传播，所有未接受出口恢复h并保实际前缀");
    require(coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy,
        "探测异常必须恢复两层原日志上下文");
  }
  {
    reuse_state::Log outer; outer.enabled = true;
    require(!reuse_state::current(), "探测作用域组件入口不得继承其它模型的活动对象");
    {
      reuse_state::Scope scope(&outer);
      require(reuse_state::current() == &outer, "外层探测作用域应可读");
      reuse_state::Log disabled;
      { reuse_state::Scope off(&disabled); require(!reuse_state::current(), "关闭作用域须隔离外层探测"); }
      try { reuse_state::Scope off(nullptr); throw std::runtime_error("作用域恢复"); }
      catch (const std::runtime_error&) {}
      require(reuse_state::current() == &outer, "嵌套和异常退出均须恢复原探测上下文");
    }
    require(!reuse_state::current(), "最外作用域退出后不能泄漏旧探测");
  }
  {
    FactorReusePairFixture fixture(true);
    solver::v122_try_reuse_pair(fixture.h, [&](bool single) { return fixture.pair(single); },
        fixture.reuse, 2, 1, fixture.system.result);
    V121File file;
    reuse_state::write(file.stream, fixture.reuse);
    const auto bytes = v121_bytes(file.stream);
    require(bytes.starts_with("{\"schema\":5,\"enabled\":true") && bytes.ends_with("}") &&
        bytes.find("\"max_krylov_steps\":1") != std::string::npos &&
        bytes.find("\"preconditioner\":\"current_diagonal_congruence\"") != std::string::npos &&
        bytes.find("\"energy_model\":\"current_positive_approximation\"") != std::string::npos &&
        bytes.find("\"scalar_model\":\"natural_schur_enclosed\"") != std::string::npos &&
        bytes.find("\"factor_epoch\":1") != std::string::npos &&
        bytes.find("\"status\":\"initial_rejected\"") != std::string::npos &&
        bytes.find("\"public_delta\":{\"constant_solves\":1,\"direction_solves\":2,\"scaled_products\":2") != std::string::npos &&
        bytes.find("\"coupled\":{\"schema\":3") != std::string::npos &&
        bytes.find("\"energy\":{\"schema\":1") != std::string::npos,
        "探测serializer须保失败事件、epoch、实际公共差值及原两类完整子对象");
    for (bool diagnostics : {false, true}) for (bool enabled : {false, true}) {
      fixture.reuse.enabled = enabled;
      candidate::DiagnosticScope emit(diagnostics);
      V121StderrCapture capture;
      { reuse_state::Report report(fixture.reuse); }
      capture.restore();
      const auto output = v121_bytes(capture.file.stream);
      require((enabled && diagnostics) ? output == "V122_FACTOR_REUSE " + bytes + "\n" : output.empty(),
          "第五标记仅在显式模式和诊断都开启时输出，不能让日志开关改变数学或费用");
    }
  }
  ++budget_cases;
  details.push_back({"factor_reuse_budget_scope_serializer", "budget"});
}

void check_factor_reuse_bounded_physics() {
  using namespace solver::detail;
  CoupledScalarSystem system;
  coupled_state::Log main; main.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  coupled_state::Scope scope(&main);
  energy_state::Scope energy_scope(&energy);
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  const auto old_regularization = system.result.ipm_primal_regularization;
  auto point = system.point; point.s = {2};
  const auto current = make_ipm_homogeneous_scaling(0, {true}, point.s, point.z, 2, 8);
  const auto before = factor_reuse::counts(system.result);
  const auto constant = factor_reuse_test_constant(system.cone, *system.factor, current,
      system.options, system.start, system.result, 2);
  const auto answer = solver::homogeneous_scaled_direction(system.cone, point, 2, 8, {-1}, {}, 0,
      1, {0}, 0, constant.first, current, constant.second, *system.factor,
      system.options, system.start, system.result, "corrector", 2, false, factor_reuse::reuse_krylov_budget);
  require(answer.v94_wide.has_value(), "一次FG的真实生产adapter仍须交付完整LD方向");
  const auto& wide = *answer.v94_wide;
  const std::vector<long double> physical{wide.z[0], wide.s[0], wide.tau, wide.kappa};
  const auto explicit_product = [&](const auto& value) {
    return std::vector<long double>{static_cast<long double>(value[1]) - value[2],
        (static_cast<long double>(value[0]) + value[3]) / 2,
        (static_cast<long double>(value[1]) + 2 * value[0]) / current.lambda[0],
        2 * static_cast<long double>(value[2]) + 0.5L * value[3]};
  };
  coupled_independent_equations(current, {1, 0, 0, 0}, physical, explicit_product);
  const std::array<long double, 4> expected{-4.0L / 9, 8.0L / 9, -1.0L / 9, 4.0L / 9};
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(std::abs(physical[i] - expected[i]) < 1e-12L,
        "旧h1与当前h2的真实物理方向须符合独立解析解");
  const auto paid = factor_reuse::difference(factor_reuse::counts(system.result), before);
  require(main.last && std::string_view(main.last->work.status) == "accepted" &&
      main.last->work.window == 1 && main.directions_started == 1 && main.directions_completed == 1 &&
      main.preconditioner_started == 2 && main.preconditioner_completed == 2 && main.krylov_started == 1 &&
      main.krylov_completed == 1 && main.cycles_started == 1 && main.products_completed == 3 &&
      main.true_checks_completed == 2 && main.initial_forcing_checks == 1 && main.initial_forcing_returns == 0 &&
      paid.constant_solves == 1 && paid.direction_solves == 2 && paid.scaled_products == 3 &&
      paid.refinement_steps == 1 && paid.scaled_refinement_steps == 1 && paid.factorizations == 0 &&
      paid.inexact_returns == 0 && energy.started == 1 && energy.completed == 1,
      "真实跨点旧因子须只付一次常量、初P及一个精化P、两次真验和一列乘积，全部公共费用保留");
  const long double v = constant.first[0];
  energy_near(constant.second, 1 + static_cast<long double>(old_rows[0]) * v * v,
      "单步修正也必须用当前d重算常量与旧R正能量");
  require(current.d == 2 && *system.result.ipm_factorizations == 1 &&
      system.result.ipm_primal_regularization == old_regularization &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows,
      "单步方向成功不能偷偷更新旧数值因子或正则化epoch");
  ++mapping_cases;
  details.push_back({"factor_reuse_bounded_physics", "mapping"});
}

struct BoundedFactorReusePairFixture : FactorReusePairFixture {
  std::vector<std::vector<long double>> inverse;
  std::vector<const char*> operations;
  bool reject_corrector;

  explicit BoundedFactorReusePairFixture(bool reject) : FactorReusePairFixture(true), reject_corrector(reject) {
    // 仅组件夹具先独立装配P的逆；探测中的P仍执行真实旧K回代及当前尺度恢复。
    // 显式A=D*P^-1隔离事务：D的一个或两个特征值可解析决定一步FG的可接受性。
    coupled_state::Log setup; setup.enabled = true;
    energy_state::Log setup_energy; setup_energy.enabled = true;
    coupled_state::Scope setup_scope(&setup);
    energy_state::Scope setup_energy_scope(&setup_energy);
    const auto constant = factor_reuse_test_constant(system.cone, *system.factor, scaling,
        system.options, system.start, system.result, 2);
    std::vector<std::vector<long double>> matrix(4, std::vector<long double>(4));
    for (std::size_t column = 0; column < 4; ++column) {
      std::vector<double> basis(4, 0); basis[column] = 1;
      const auto image = precondition(basis, false, constant);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) matrix[row_index][column] = image[row_index];
    }
    inverse.assign(4, std::vector<long double>(4));
    for (std::size_t column = 0; column < 4; ++column) {
      std::vector<double> basis(4, 0); basis[column] = 1;
      const auto image = normal_inverse_dense_solve(matrix, basis);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) inverse[row_index][column] = image[row_index];
    }
  }

  std::vector<double> precondition(std::span<const double> input, bool refinement,
      const std::pair<std::vector<double>, long double>& constant) {
    const auto reduced = system.factor->solve_homogeneous_direct(
        solver::detail::ipm_homogeneous_scaled_reduced_rhs(scaling, input),
        system.options, system.start, system.result, false, refinement);
    return solver::detail::ipm_homogeneous_scaled_recover(scaling, system.cone.costs, system.cone.rhs,
        input, reduced, constant.first, constant.second);
  }

  solver::HomogeneousDirection direction(const char* role, const std::array<double, 4>& diagonal,
      const std::pair<std::vector<double>, long double>& constant, std::size_t cap) {
    const std::vector<double> rhs{1, 1, 0, 0};
    const auto p = [&](std::span<const double> input, bool refinement) {
      operations.push_back(refinement ? "P1" : "P0");
      return precondition(input, refinement, constant);
    };
    const auto multiply = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& image) {
      operations.push_back(std::is_same_v<Scalar, double> ? "A列" : "A真");
      ++*system.result.ipm_homogeneous_scaled_products;
      image.assign(4, 0);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) {
        for (std::size_t column = 0; column < 4; ++column)
          image[row_index] += inverse[row_index][column] * input[column];
        image[row_index] *= diagonal[row_index];
      }
    };
    bool forced = false;
    auto answer = solver::v100_solve_coupled(scaling, rhs, p,
        multiply, multiply, {}, role, 2, &forced, true, true, false, cap);
    const auto explicit_product = [&](const auto& input) {
      std::vector<long double> image(4, 0);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) {
        for (std::size_t column = 0; column < 4; ++column)
          image[row_index] += inverse[row_index][column] * input[column];
        image[row_index] *= diagonal[row_index];
      }
      return image;
    };
    const auto image = explicit_product(answer);
    const auto tolerance = solver::detail::ipm_homogeneous_scaled_tolerances(scaling, rhs);
    const coupled_kernel::Layout layout{scaling.layout.columns, scaling.layout.rows,
        scaling.layout.inequality_rows.size()};
    for (std::size_t i = 0; i < image.size(); ++i) {
      const auto strict = tolerance[layout.block(i)];
      const long double limit = forced && i < 2 ? std::max(0.1L, static_cast<long double>(strict)) : strict;
      require(std::abs(image[i] - rhs[i]) <= limit, "完整pair的每次交付须独立符合其原strict或forcing门槛");
    }
    return solver::v100_deliver_direction(scaling, std::move(answer), forced, system.result);
  }

  solver::HomogeneousDirection bounded_pair(bool reuse_probe) {
    const auto constant = factor_reuse_test_constant(system.cone, *system.factor, scaling,
        system.options, system.start, system.result, 2);
    const auto cap = reuse_probe ? solver::detail::factor_reuse::reuse_krylov_budget :
        coupled_kernel::max_window * coupled_kernel::max_cycles;
    ++affine_calls;
    const auto affine = direction("affine", {2, 1, 1, 1}, constant, cap);
    require(affine.v94_wide.has_value(), "一步事务首方向必须实际完成宽方向交付");
    h[0] = 7;
    ++corrector_calls;
    return direction("corrector", reject_corrector ? std::array<double, 4>{2, 100, 1, 1} :
        std::array<double, 4>{2, 2.1, 1, 1}, constant, cap);
  }
};

void check_factor_reuse_bounded_pair_transaction() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (bool reject : {false, true}) {
    BoundedFactorReusePairFixture fixture(reject);
    const auto saved_h = fixture.h;
    const auto before = reuse_state::counts(fixture.system.result);
    const auto answer = solver::v122_try_reuse_pair(fixture.h,
        [&](bool reuse_probe) { return fixture.bounded_pair(reuse_probe); }, fixture.reuse, 2, 1, fixture.system.result);
    require(answer.has_value() != reject && fixture.reuse.events.size() == 1 &&
        fixture.affine_calls == 1 && fixture.corrector_calls == 1 && fixture.operations.size() == 10,
        "整组须实际运行两次各一列FG，只有corrector也通过才返回完整方向");
    const std::array<const char*, 5> expected{"P0", "A真", "P1", "A列", "A真"};
    for (std::size_t i = 0; i < fixture.operations.size(); ++i)
      require(std::string_view(fixture.operations[i]) == expected[i % expected.size()],
          "每个方向必须复用初P真残差，真实执行一次精化P、一列A和候选真验而不重做初P");
    const auto& event = fixture.reuse.events[0];
    const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
    require(event.public_delta.constant_solves == 1 && event.public_delta.direction_solves == 4 &&
        event.public_delta.scaled_products == 6 && event.public_delta.refinement_steps == 2 &&
        event.public_delta.scaled_refinement_steps == 2 && event.public_delta.factorizations == 0 &&
        paid.constant_solves == 1 && paid.direction_solves == 4 && paid.scaled_products == 6 &&
        paid.refinement_steps == 2 && paid.scaled_refinement_steps == 2 && paid.factorizations == 0 &&
        event.coupled.constant_calls_completed == 1 && event.coupled.directions_started == 2 &&
        event.coupled.directions_completed == (reject ? 1 : 2) && event.coupled.preconditioner_completed == 4 &&
        event.coupled.products_completed == 6 && event.coupled.true_checks_completed == 4 &&
        event.coupled.krylov_started == 2 && event.coupled.krylov_completed == 2 && event.coupled.cycles_started == 2 &&
        event.coupled.initial_forcing_checks == 2 && event.coupled.initial_forcing_returns == 0 &&
        event.coupled.candidate_forcing_checks == 1 && event.coupled.candidate_forcing_returns == (reject ? 0 : 1) &&
        event.energy.started == 1 && event.energy.completed == 1,
        "完整探测须保真实旧K回代与FG费用，候选forcing检查及返回按实际数学结果计费");
    require(coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy,
        "完整pair的成功与typed拒绝都必须恢复外层日志作用域");
    if (!reject) {
      require(std::string_view(event.status) == "accepted" && fixture.h == std::vector<double>({7}) &&
          fixture.reuse.reused_steps == 1 && paid.inexact_returns == 1 && fixture.main.directions_completed == 2 &&
          fixture.main.preconditioner_completed == 4 && fixture.main.krylov_completed == 2 &&
          fixture.main.products_completed == 6 && fixture.main.true_checks_completed == 4 &&
          fixture.main.candidate_forcing_checks == 1 && fixture.main.candidate_forcing_returns == 1 &&
          fixture.energy.completed == 1,
          "完整成功才合并一次常量、两次FG、候选forcing与能量账本并保留选定h");
    } else {
      require(std::string_view(event.status) == "bounded_rejected" && fixture.h == saved_h &&
          fixture.reuse.reused_steps == 0 && paid.inexact_returns == 0 && fixture.main.constant_calls_started == 0 &&
          fixture.main.directions_started == 0 && fixture.main.krylov_started == 0 && fixture.energy.started == 0 &&
          event.coupled.last && std::string_view(event.coupled.last->role) == "corrector" &&
          std::string_view(event.coupled.last->work.status) == "bounded_rejected",
          "corrector一步真验拒绝必须丢弃整组、恢复原h，主账本隔离而公共已付费用不回滚");
      ++fixture.reuse.fresh_started;
      fixture.system.factor->factor({2}, fixture.system.options, fixture.system.start, fixture.system.result);
      ++fixture.reuse.fresh_completed;
      require(fixture.h == saved_h, "fresh重做前不能沿用失败affine更新的h");
      const auto fresh_before = reuse_state::counts(fixture.system.result);
      const auto fresh = fixture.bounded_pair(false);
      const auto fresh_paid = reuse_state::difference(reuse_state::counts(fixture.system.result), fresh_before);
      require(fresh.v94_wide && fixture.affine_calls == 2 && fixture.corrector_calls == 2 &&
          fresh_paid.constant_solves == 1 && fresh_paid.direction_solves >= 2 &&
          fresh_paid.direction_solves == fixture.main.preconditioner_completed &&
          fresh_paid.scaled_products == fixture.main.products_completed &&
          fresh_paid.scaled_refinement_steps == fixture.main.krylov_completed &&
          fresh_paid.refinement_steps == fixture.main.krylov_completed && *fixture.system.result.ipm_factorizations == 2 &&
          fixture.main.constant_calls_completed == 1 && fixture.main.directions_completed == 2 && fixture.energy.completed == 1,
          "typed拒绝后须支付真实新factor并在原96步预算重做整pair，失败探测费用持续保留");
    }
  }
  ++transaction_cases;
  details.push_back({"factor_reuse_bounded_pair_transaction", "transaction"});
}

namespace current_diagonal = solver::detail::current_diagonal;

std::pair<std::vector<double>, long double> current_diagonal_test_constant(
    const solver::ConeModel& cone, const solver::KktFactor& factor,
    const solver::detail::IpmHomogeneousScaling& scaling, const std::vector<double>& rows,
    const current_diagonal::ValidatedView& view, current_diagonal::Work& work,
    const solver::SolveOptions& options, solver::Clock::time_point start,
    solver::SolveResult& result, const std::function<void()>& budget = {}) {
  auto* log = coupled_state::current();
  require(log != nullptr, "当前R常量测试须使用真实联立作用域");
  const auto rhs = solver::detail::ipm_homogeneous_scaled_constant_rhs(scaling, cone.costs, cone.rhs, budget);
  if (budget) budget();
  ++log->constant_calls_started;
  auto constant = solver::v124_apply_current(view, rhs, factor, options, start, result, true, false, work, budget);
  ++log->constant_calls_completed;
  const auto denominator = solver::v124_energy_denominator(factor, cone.costs.size(), constant, rows,
      result, budget, 2);
  return {std::move(constant), denominator};
}

void check_current_diagonal_hand_schur() {
  const std::vector<current_diagonal::Index> starts{0, 1}, indices{0};
  const std::vector<double> values{1}, rows{2}, rhs{3, 4};
  current_diagonal::Work work;
  const current_diagonal::ValidatedView view(1, 1, starts, indices, values, rows, work);
  int calls = 0;
  const auto old_direct = [&](std::span<const double> input) {
    ++calls;
    require(input.size() == 2 && input[1] == 0, "旧完整逆只接收[q;0]，不得再次消去原b");
    return std::vector<double>{input[0] / 2, input[0] / 2};
  };
  const auto answer = current_diagonal::apply(view, rhs, old_direct, work);
  require(answer == std::vector<double>({2.5, -0.75}) && calls == 1 &&
      answer[0] - 2 * answer[1] == 4 && rhs[0] - (answer[0] + answer[1]) == 1.25 &&
      work.validation_started == 1 && work.validation_completed == 1 && work.applications_started == 1 &&
      work.applications_completed == 1 && work.transpose_completed == 1 && work.direct_completed == 1 &&
      work.product_completed == 1 && work.transpose_terms == 1 && work.product_terms == 1 && work.row_divisions == 2,
      "手算符号须给x=2.5、z=-0.75，当前下块精确，上块保留Schur近似缺陷并记录两次稀疏乘积");
  current_diagonal::Work same_work;
  const std::vector<double> same_rows{1};
  const current_diagonal::ValidatedView same(1, 1, starts, indices, values, same_rows, same_work);
  require(current_diagonal::apply(same, rhs, old_direct, same_work) == std::vector<double>({3.5, -0.5}),
      "同R时外消元与旧完整K逆一致，包含非零a和b而非仅零下块右端");
  current_diagonal::Work zero_work;
  const std::vector<current_diagonal::Index> zero_starts{0, 0}, empty_indices;
  const std::vector<double> empty_values;
  const current_diagonal::ValidatedView zero(1, 1, zero_starts, empty_indices, empty_values, rows, zero_work);
  const auto zero_answer = current_diagonal::apply(zero, rhs, [](std::span<const double> input) {
    return std::vector<double>{input[0], 0};
  }, zero_work);
  require(zero_answer == std::vector<double>({3, -2}) && zero_work.transpose_terms == 0 && zero_work.product_terms == 0,
      "G为零时仍须恢复负号正确的z=-b/Rcur，不能沿用旧逆返回的对偶方向");
  ++mapping_cases;
  details.push_back({"current_diagonal_hand_schur", "mapping"});
}

void check_current_diagonal_real_normal() {
  WarmRegularizationSystem system;
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  const auto old_r = system.result.ipm_primal_regularization, old_delta = system.result.ipm_dual_regularization;
  const auto old_factors = system.result.ipm_factorizations;
  solver::detail::factor_reuse::Probe probe;
  const std::vector<double> diagonal{7, 0, 11};
  const auto rows = solver::v124_current_rows(*system.factor, diagonal, system.result, probe, {});
  current_diagonal::ValidatedView view(2, 3, system.cone.starts, system.cone.rows, system.cone.values,
      rows, probe.current_diagonal);
  for (std::size_t i = 0; i < rows.size(); ++i)
    require(rows[i] == static_cast<double>(static_cast<long double>(diagonal[i]) + *old_delta),
        "当前R须按本次diagonal与旧dual正则同一次double舍入，包含等式与凝聚行");
  const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5};
  const auto before = solver::detail::factor_reuse::counts(system.result);
  const auto answer = solver::v124_apply_current(view, rhs, *system.factor, system.options, system.start,
      system.result, false, true, probe.current_diagonal, {});
  const std::array<long double, 3> gx{static_cast<long double>(answer[0]) + 2 * answer[1],
      3 * static_cast<long double>(answer[0]) - answer[1], 5 * static_cast<long double>(answer[0])};
  for (std::size_t i = 0; i < rows.size(); ++i)
    energy_near(gx[i] - static_cast<long double>(rows[i]) * answer[2 + i], rhs[2 + i],
        "反均衡与单例完整恢复后，每个当前名义原始行必须闭合");
  std::vector<long double> reduced(5, 0);
  reduced[0] = rhs[0] + static_cast<long double>(rhs[2]) / rows[0] +
      3 * static_cast<long double>(rhs[3]) / rows[1] + 5 * static_cast<long double>(rhs[4]) / rows[2];
  reduced[1] = rhs[1] + 2 * static_cast<long double>(rhs[2]) / rows[0] -
      static_cast<long double>(rhs[3]) / rows[1];
  std::vector<double> rounded_rhs(5, 0);
  for (std::size_t column = 0; column < 2; ++column) {
    long double q = rhs[column];
    for (auto term = system.cone.starts[column]; term < system.cone.starts[column + 1]; ++term) {
      const auto row_index = static_cast<std::size_t>(system.cone.rows[static_cast<std::size_t>(term)]);
      q += static_cast<long double>(system.cone.values[static_cast<std::size_t>(term)]) *
          (static_cast<long double>(rhs[2 + row_index]) / rows[row_index]);
    }
    rounded_rhs[column] = static_cast<double>(q);
  }
  const auto old_action = system.factor->solve_homogeneous_direct(rounded_rhs, system.options,
      system.start, system.result, false);
  for (std::size_t i = 0; i < 2; ++i)
    require(answer[i] == old_action[i],
        "当前R只改变外消元与恢复，x必须逐值复用同一旧近似P对独立舍入[q;0]的实际作用");
  const auto nominal = system.nominal();
  const auto dense = independent_energy_dense(nominal, reduced);
  // raw direct仅承诺近似P；dense保留独立名义解及缺陷，不能要求两者前向相同。
  for (std::size_t i = 0; i < 5; ++i) {
    long double dense_defect = -reduced[i], actual_defect = -static_cast<long double>(rounded_rhs[i]);
    for (std::size_t j = 0; j < 5; ++j) {
      dense_defect += nominal[i][j] * dense[j];
      actual_defect += nominal[i][j] * old_action[j];
    }
    require(std::isfinite(dense_defect) && std::isfinite(actual_defect) &&
        std::abs(dense_defect) <= 1e-11L * std::max(1.0L, std::abs(reduced[i])),
        "独立dense须满足名义K方程，实际旧P缺陷单独保留且不冒称已消除");
  }
  current_diagonal::Work same_work;
  const current_diagonal::ValidatedView same_view(2, 3, system.cone.starts, system.cone.rows,
      system.cone.values, old_rows, same_work);
  const auto same = solver::v124_apply_current(same_view, rhs, *system.factor, system.options, system.start,
      system.result, false, false, same_work, {});
  const auto direct = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false);
  for (int kind : {0, 1}) {
    const auto& value = kind == 0 ? same : direct;
    const std::array<long double, 3> image{static_cast<long double>(value[0]) + 2 * value[1],
        3 * static_cast<long double>(value[0]) - value[1], 5 * static_cast<long double>(value[0])};
    std::array<long double, 5> residual{};
    for (std::size_t i = 0; i < 5; ++i) {
      residual[i] = rhs[i];
      for (std::size_t j = 0; j < 5; ++j) residual[i] -= nominal[i][j] * value[j];
    }
    if (kind == 0) for (std::size_t i = 0; i < 3; ++i)
      energy_near(image[i] - static_cast<long double>(old_rows[i]) * value[2 + i], rhs[2 + i],
          "同R外消元的实际名义下块仍须逐行闭合，不假设raw P的前向误差消失");
    for (std::size_t column = 0; column < 2; ++column) {
      long double schur_residual = rhs[column] - static_cast<long double>(*old_r) * value[column];
      long double contracted_residual = residual[column];
      for (auto term = system.cone.starts[column]; term < system.cone.starts[column + 1]; ++term) {
        const auto row_index = static_cast<std::size_t>(system.cone.rows[static_cast<std::size_t>(term)]);
        const long double coefficient = system.cone.values[static_cast<std::size_t>(term)];
        schur_residual += coefficient * ((static_cast<long double>(rhs[2 + row_index]) - image[row_index]) / old_rows[row_index]);
        contracted_residual += coefficient * (residual[2 + row_index] / old_rows[row_index]);
      }
      require(std::abs(schur_residual - contracted_residual) <= 1e-11L * std::max(1.0L, std::abs(schur_residual)),
          "同R的完整行与Schur残差必须符合独立消元恒等式，允许旧近似P保留其实际缺陷");
    }
  }
  const auto paid = solver::detail::factor_reuse::difference(solver::detail::factor_reuse::counts(system.result), before);
  require(probe.current_rows_started == 1 && probe.current_rows_completed == 1 && probe.current_rows_terms == 3 &&
      probe.current_diagonal.validation_completed == 1 && probe.current_diagonal.applications_completed == 1 &&
      probe.current_diagonal.transpose_terms == 5 && probe.current_diagonal.product_terms == 5 &&
      probe.current_diagonal.row_divisions == 6 && paid.direction_solves == 4 && paid.refinement_steps == 1 &&
      paid.scaled_refinement_steps == 1 && paid.factorizations == 0 &&
      system.result.ipm_factorizations == old_factors && system.result.ipm_primal_regularization == old_r &&
      system.result.ipm_dual_regularization == old_delta &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows &&
      system.options.ipm.equilibrate_kkt && *system.result.ipm_eliminated_singletons == 1,
      "当前R独立快照须保旧epoch全部数值与公共正则，实际新增Gt/G、旧direct和精化费用守恒");
  ++mapping_cases;
  details.push_back({"current_diagonal_real_normal", "mapping"});
}

void check_current_diagonal_constant_physics() {
  using namespace solver::detail;
  CoupledScalarSystem system;
  coupled_state::Log main; main.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  coupled_state::Scope scope(&main);
  energy_state::Scope energy_scope(&energy);
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  auto point = system.point; point.s = {2};
  const auto scaling = make_ipm_homogeneous_scaling(0, {true}, point.s, point.z, 2, 8);
  factor_reuse::Probe probe;
  const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, probe, {});
  const current_diagonal::ValidatedView view(0, 1, system.cone.starts, system.cone.rows, system.cone.values,
      rows, probe.current_diagonal);
  const auto before = factor_reuse::counts(system.result);
  const auto constant = current_diagonal_test_constant(system.cone, *system.factor, scaling, rows, view,
      probe.current_diagonal, system.options, system.start, system.result);
  energy_near(constant.first[0], -1 / (static_cast<long double>(scaling.d) * rows[0]),
      "无原始列时当前R常量必须按当前d和当前R恢复，而非沿用旧因子对偶结果");
  const long double v = constant.first[0];
  energy_near(constant.second, 1 + static_cast<long double>(rows[0]) * v * v,
      "当前正能量须读取当前R快照，不能走旧factor隐式R入口");
  require(std::abs(constant.second - (1 + static_cast<long double>(old_rows[0]) * v * v)) > 0.01L,
      "解析夹具须实际区分当前正能量与旧R能量");
  const auto direction = solver::homogeneous_scaled_direction(system.cone, point, 2, 8, {-1}, {}, 0,
      1, {0}, 0, constant.first, scaling, constant.second, *system.factor, system.options, system.start,
      system.result, "corrector", 2, false, factor_reuse::reuse_krylov_budget, &view, &probe.current_diagonal);
  require(direction.v94_wide.has_value(), "当前R方向仍须交付同一生产宽方向");
  const auto& wide = *direction.v94_wide;
  const std::vector<long double> physical{wide.z[0], wide.s[0], wide.tau, wide.kappa};
  const auto explicit_product = [&](const auto& value) {
    return std::vector<long double>{static_cast<long double>(value[1]) - value[2],
        (static_cast<long double>(value[0]) + value[3]) / 2,
        (static_cast<long double>(value[1]) + 2 * value[0]) / scaling.lambda[0],
        2 * static_cast<long double>(value[2]) + 0.5L * value[3]};
  };
  coupled_independent_equations(scaling, {1, 0, 0, 0}, physical, explicit_product);
  const auto paid = factor_reuse::difference(factor_reuse::counts(system.result), before);
  require(main.last && std::string_view(main.last->work.status) == "accepted" && main.krylov_started == 0 &&
      main.constant_calls_completed == 1 && main.preconditioner_completed == 1 && main.products_completed == 1 &&
      probe.current_diagonal.applications_completed == 2 && probe.current_diagonal.transpose_completed == 2 &&
      probe.current_diagonal.product_completed == 2 && probe.current_diagonal.transpose_terms == 0 &&
      probe.current_diagonal.product_terms == 0 && probe.current_diagonal.row_divisions == 4 &&
      paid.constant_solves == 1 && paid.direction_solves == 1 && paid.scaled_products == 1 &&
      paid.refinement_steps == 0 && paid.factorizations == 0 && paid.inexact_returns == 0 &&
      energy.completed == 1 && *system.result.ipm_factorizations == 1 &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows,
      "G0真实当前点应由当前R初P通过原strict，常量和方向统一wrapper且不修改旧因子");
  ++mapping_cases;
  details.push_back({"current_diagonal_constant_physics", "mapping"});
}

void check_current_diagonal_indefinite_energy() {
  const std::vector<current_diagonal::Index> starts{0, 1}, indices{0};
  const std::vector<double> values{1}, rows{0.1}, rhs{-8.99, 1};
  current_diagonal::Work work;
  const current_diagonal::ValidatedView view(1, 1, starts, indices, values, rows, work);
  const auto constant = current_diagonal::apply(view, rhs, [](std::span<const double> input) {
    const double x = input[0] / 1.01;
    return std::vector<double>{x, x / 100};
  }, work);
  solver::detail::ipm_energy::Work energy;
  const auto denominator = solver::detail::ipm_energy::compute(1, constant, rows, 1, energy);
  const long double hybrid_top = 1 + 1.0L / 100 - 1 / static_cast<long double>(rows[0]);
  const long double natural = 1 - (8.99L * constant[0] + constant[1]);
  require(hybrid_top < 0 && natural < 0 && denominator >= 1 && denominator > natural + 9,
      "等效hybrid左上块可以为负，正近似能量不得误称其精确齐次Schur分母");
  energy_near(constant[0], 1, "不定反例常量须由真实wrapper计算得到x约1");
  energy_near(constant[1], 0, "不定反例当前行仍须闭合，得到z约0");
  energy_near(denominator, 2, "current正近似能量为2，而自然分母约-7.99");
  ++mapping_cases;
  details.push_back({"current_diagonal_indefinite_energy", "mapping"});
}

struct CurrentDiagonalPairFixture : FactorReusePairFixture {
  std::vector<std::vector<long double>> inverse;
  std::vector<const char*> operations;
  bool reject_corrector;

  explicit CurrentDiagonalPairFixture(bool reject) : FactorReusePairFixture(true), reject_corrector(reject) {
    coupled_state::Log setup; setup.enabled = true;
    energy_state::Log setup_energy; setup_energy.enabled = true;
    coupled_state::Scope setup_scope(&setup);
    energy_state::Scope setup_energy_scope(&setup_energy);
    solver::detail::factor_reuse::Probe preparation;
    const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, preparation, {});
    const current_diagonal::ValidatedView view(0, 1, system.cone.starts, system.cone.rows, system.cone.values,
        rows, preparation.current_diagonal);
    const auto constant = current_diagonal_test_constant(system.cone, *system.factor, scaling, rows, view,
        preparation.current_diagonal, system.options, system.start, system.result);
    std::vector<std::vector<long double>> matrix(4, std::vector<long double>(4));
    for (std::size_t column = 0; column < 4; ++column) {
      std::vector<double> basis(4, 0); basis[column] = 1;
      const auto image = precondition(basis, false, constant, &view, &preparation.current_diagonal, {});
      for (std::size_t row_index = 0; row_index < 4; ++row_index) matrix[row_index][column] = image[row_index];
    }
    inverse.assign(4, std::vector<long double>(4));
    for (std::size_t column = 0; column < 4; ++column) {
      std::vector<double> basis(4, 0); basis[column] = 1;
      const auto image = normal_inverse_dense_solve(matrix, basis);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) inverse[row_index][column] = image[row_index];
    }
  }

  std::vector<double> precondition(std::span<const double> input, bool refinement,
      const std::pair<std::vector<double>, long double>& constant,
      const current_diagonal::ValidatedView* view, current_diagonal::Work* work,
      const std::function<void()>& budget) {
    const auto rhs = solver::detail::ipm_homogeneous_scaled_reduced_rhs(scaling, input, budget);
    const auto reduced = view ? solver::v124_apply_current(*view, rhs, *system.factor, system.options,
        system.start, system.result, false, refinement, *work, budget) :
        system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false, refinement);
    return solver::detail::ipm_homogeneous_scaled_recover(scaling, system.cone.costs, system.cone.rhs,
        input, reduced, constant.first, constant.second, budget);
  }

  solver::HomogeneousDirection direction(const char* role, const std::array<double, 4>& diagonal,
      const std::pair<std::vector<double>, long double>& constant, std::size_t cap,
      const current_diagonal::ValidatedView* view, current_diagonal::Work* work,
      const std::function<void()>& budget) {
    const std::vector<double> rhs{1, 1, 0, 0};
    const auto p = [&](std::span<const double> input, bool refinement) {
      operations.push_back(refinement ? "P1" : "P0");
      return precondition(input, refinement, constant, view, work, budget);
    };
    const auto explicit_product = [&](const auto& input) {
      std::vector<long double> image(4, 0);
      for (std::size_t row_index = 0; row_index < 4; ++row_index) {
        for (std::size_t column = 0; column < 4; ++column)
          image[row_index] += inverse[row_index][column] * input[column];
        image[row_index] *= diagonal[row_index];
      }
      return image;
    };
    const auto product = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& image) {
      operations.push_back(std::is_same_v<Scalar, double> ? "A列" : "A真");
      ++*system.result.ipm_homogeneous_scaled_products;
      image = explicit_product(input);
    };
    bool forced = false;
    auto answer = solver::v100_solve_coupled(scaling, rhs, p, product, product, budget, role, 2,
        &forced, true, true, false, cap);
    const auto image = explicit_product(answer);
    const auto tolerances = solver::detail::ipm_homogeneous_scaled_tolerances(scaling, rhs);
    const coupled_kernel::Layout layout{0, 1, 1};
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      const long double strict = tolerances[layout.block(i)];
      const long double limit = forced && i < 2 ? std::max(strict, 0.1L) : strict;
      require(std::abs(image[i] - rhs[i]) <= limit, "当前R整组每次交付须代回显式算子，保持原五块门槛");
    }
    return solver::v100_deliver_direction(scaling, std::move(answer), forced, system.result, budget);
  }

  solver::HomogeneousDirection pair_current(bool reuse_probe, const std::function<void()>& budget = {}) {
    std::vector<double> rows;
    std::optional<current_diagonal::ValidatedView> view;
    current_diagonal::Work* work = nullptr;
    std::pair<std::vector<double>, long double> constant;
    if (reuse_probe) {
      auto& event = reuse.events.back();
      event.current_diagonal.columns = 0;
      event.current_diagonal.rows = 1;
      event.current_diagonal.nonzeros = 0;
      rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, event, budget);
      work = &event.current_diagonal;
      view.emplace(0, 1, system.cone.starts, system.cone.rows, system.cone.values, rows, *work, budget);
      constant = current_diagonal_test_constant(system.cone, *system.factor, scaling, rows, *view,
          *work, system.options, system.start, system.result, budget);
    } else {
      constant = factor_reuse_test_constant(system.cone, *system.factor, scaling,
          system.options, system.start, system.result, 2);
    }
    const auto cap = reuse_probe ? solver::detail::factor_reuse::reuse_krylov_budget :
        coupled_kernel::max_window * coupled_kernel::max_cycles;
    ++affine_calls;
    const auto affine = direction("affine", {2, 1, 1, 1}, constant, cap, view ? &*view : nullptr, work, budget);
    require(affine.v94_wide.has_value(), "当前R整组须实际完成首方向");
    h[0] = 7;
    ++corrector_calls;
    return direction("corrector", reject_corrector ? std::array<double, 4>{2, 100, 1, 1} :
        std::array<double, 4>{2, 2.1, 1, 1}, constant, cap, view ? &*view : nullptr, work, budget);
  }
};

void check_current_diagonal_pair_transaction() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (bool reject : {false, true}) {
    CurrentDiagonalPairFixture fixture(reject);
    const auto saved_h = fixture.h;
    const auto old_rows = std::vector<double>(fixture.system.factor->v93_regularized_h().begin(),
        fixture.system.factor->v93_regularized_h().end());
    const auto before = reuse_state::counts(fixture.system.result);
    const auto answer = solver::v122_try_reuse_pair(fixture.h,
        [&](bool reuse_probe) { return fixture.pair_current(reuse_probe); },
        fixture.reuse, 2, 1, fixture.system.result);
    require(answer.has_value() != reject && fixture.reuse.events.size() == 1 &&
        fixture.affine_calls == 1 && fixture.corrector_calls == 1 && fixture.operations.size() == 10,
        "当前R整组仍须运行两方向真实FG，只能完整成功后采用");
    const auto& event = fixture.reuse.events[0];
    const auto& work = event.current_diagonal;
    const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
    require(event.current_rows_started == 1 && event.current_rows_completed == 1 && event.current_rows_terms == 1 &&
        work.validation_started == 1 && work.validation_completed == 1 && work.validation_terms == 2 &&
        work.applications_started == 5 && work.applications_completed == 5 && work.transpose_completed == 5 &&
        work.direct_completed == 5 && work.product_completed == 5 && work.row_divisions == 10 &&
        paid.constant_solves == 1 && paid.direction_solves == 4 && paid.refinement_steps == 2 &&
        paid.scaled_refinement_steps == 2 && paid.scaled_products == 6 && paid.factorizations == 0 &&
        event.public_delta.constant_solves == 1 && event.public_delta.direction_solves == 4 &&
        event.public_delta.refinement_steps == 2 && event.public_delta.scaled_refinement_steps == 2 &&
        event.coupled.krylov_completed == 2 && event.coupled.true_checks_completed == 4 &&
        event.coupled.candidate_forcing_checks == 1 && event.coupled.candidate_forcing_returns == (reject ? 0 : 1) &&
        event.energy.completed == 1 &&
        std::vector<double>(fixture.system.factor->v93_regularized_h().begin(), fixture.system.factor->v93_regularized_h().end()) == old_rows,
        "同一当前R验证只付一次，五次应用各付真实旧回代与新Gt/G，原因子和所有公共费用不回滚");
    if (!reject) {
      require(fixture.h == std::vector<double>({7}) && fixture.reuse.reused_steps == 1 &&
          std::string_view(event.status) == "accepted" && fixture.main.directions_completed == 2 &&
          fixture.main.krylov_completed == 2 && fixture.main.candidate_forcing_checks == 1 &&
          fixture.main.candidate_forcing_returns == 1 && fixture.energy.completed == 1 && paid.inexact_returns == 1,
          "当前R完整成功才合并两方向FG、forcing及正近似能量账本");
    } else {
      require(fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
          std::string_view(event.status) == "bounded_rejected" && fixture.main.directions_started == 0 &&
          fixture.main.constant_calls_started == 0 && fixture.energy.started == 0 && paid.inexact_returns == 0,
          "当前R corrector拒绝仍须隔离整pair、恢复h并保留已经支付的常量和五次wrapper");
      fixture.system.factor->factor({2}, fixture.system.options, fixture.system.start, fixture.system.result);
      require(fixture.h == saved_h, "当前R拒绝后新分解重做前h须保持原值");
      const auto fresh_before = reuse_state::counts(fixture.system.result);
      const auto fresh = fixture.pair_current(false);
      const auto fresh_paid = reuse_state::difference(reuse_state::counts(fixture.system.result), fresh_before);
      require(fresh.v94_wide && fixture.affine_calls == 2 && fixture.corrector_calls == 2 &&
          fresh_paid.constant_solves == 1 && fresh_paid.direction_solves == fixture.main.preconditioner_completed &&
          fresh_paid.scaled_refinement_steps == fixture.main.krylov_completed &&
          fixture.main.directions_completed == 2 && *fixture.system.result.ipm_factorizations == 2 &&
          work.applications_completed == 5 && event.current_rows_completed == 1,
          "fresh96须重做完整pair且直接使用新factor，不能继续支付或修改旧探测wrapper账本");
    }
    require(coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy,
        "当前R整组各出口均须恢复原两层作用域");
  }
  ++transaction_cases;
  details.push_back({"current_diagonal_pair_transaction", "transaction"});
}

void current_diagonal_prefix(const current_diagonal::Work& work) {
  const auto prefix = [](auto started, auto completed) {
    require(completed <= started && started - completed <= 1,
        "当前R费用只能保留真实启动/完成前缀，不能发布半份完成");
  };
  prefix(work.validation_started, work.validation_completed);
  prefix(work.applications_started, work.applications_completed);
  prefix(work.transpose_started, work.transpose_completed);
  prefix(work.direct_started, work.direct_completed);
  prefix(work.product_started, work.product_completed);
  require(work.transpose_started <= work.applications_started && work.direct_started <= work.transpose_completed &&
      work.product_started <= work.direct_completed && work.applications_completed <= work.product_completed,
      "当前R的Gt、旧direct、G与结果交付须保持真实依赖次序");
}

void check_current_diagonal_budget_and_errors() {
  namespace reuse_state = solver::detail::factor_reuse;
  WarmRegularizationSystem system;
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  const auto run = [&](reuse_state::Probe& probe, const std::function<void()>& budget) {
    probe.current_diagonal.columns = 2;
    probe.current_diagonal.rows = 3;
    probe.current_diagonal.nonzeros = system.cone.values.size();
    const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{7, 0, 11}, system.result, probe, budget);
    const current_diagonal::ValidatedView view(2, 3, system.cone.starts, system.cone.rows, system.cone.values,
        rows, probe.current_diagonal, budget);
    return solver::v124_apply_current(view, std::vector<double>{0.5, -1, 1.5, -1, 2.5}, *system.factor,
        system.options, system.start, system.result, false, true, probe.current_diagonal, budget);
  };
  int total = 0;
  reuse_state::Probe baseline;
  run(baseline, [&] { ++total; });
  require(total > 0 && baseline.current_diagonal.applications_completed == 1,
      "先遍历真实均衡/单例K的当前R构造、验证、Gt、旧direct和G所有预算边界");
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    reuse_state::Probe probe;
    const auto before = reuse_state::counts(system.result);
    int checks = 0;
    std::vector<double> output{101, 202};
    bool original = false;
    try { output = run(probe, [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    current_diagonal_prefix(probe.current_diagonal);
    const auto paid = reuse_state::difference(reuse_state::counts(system.result), before);
    require(original && checks == cutoff && output == std::vector<double>({101, 202}) &&
        probe.current_rows_completed <= probe.current_rows_started && probe.current_rows_terms <= 3 &&
        probe.current_diagonal.transpose_terms <= 5 && probe.current_diagonal.product_terms <= 5 &&
        probe.current_diagonal.row_divisions <= 6 && paid.direction_solves == probe.current_diagonal.direct_started &&
        paid.refinement_steps == paid.direction_solves && paid.scaled_refinement_steps == paid.direction_solves &&
        paid.constant_solves == 0 && paid.factorizations == 0,
        "每个新预算中断须保原类型、旧结果与实际已付旧K精化，不能转数值拒绝或撤回费用");
  }
  {
    // 长矩形夹具只隔离预算和存储遍历，不把回调单位算子宣称为物理旧K逆。
    constexpr std::size_t count = 4097;
    const std::vector<current_diagonal::Index> starts{0, static_cast<current_diagonal::Index>(count)};
    std::vector<current_diagonal::Index> indices(count);
    for (std::size_t i = 0; i < count; ++i) indices[i] = static_cast<current_diagonal::Index>(i);
    const std::vector<double> values(count, 1), rows(count, 2), rhs(count + 1, 1);
    const auto exercise = [&](current_diagonal::Work& work, int& direct_calls,
        const std::function<void()>& budget) {
      const current_diagonal::ValidatedView view(1, count, starts, indices, values, rows, work, budget);
      return current_diagonal::apply(view, rhs, [&](std::span<const double> input) {
        ++direct_calls;
        return std::vector<double>(input.begin(), input.end());
      }, work, budget);
    };
    current_diagonal::Work full;
    int direct_calls = 0, boundaries = 0;
    exercise(full, direct_calls, [&] { ++boundaries; });
    require(full.validation_terms == 2 + 2 * count && full.transpose_terms == count &&
        full.product_terms == count && full.row_divisions == 2 * count && direct_calls == 1,
        "超过4096的R与非零项均须跨真实分块预算边界且每项只计一次");
    for (int cutoff = 1; cutoff <= boundaries; ++cutoff) {
      current_diagonal::Work work;
      int calls = 0, checks = 0;
      bool original = false;
      try { exercise(work, calls, [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
      catch (const solver::BudgetExpired&) { original = true; }
      current_diagonal_prefix(work);
      require(original && checks == cutoff && calls == work.direct_started &&
          work.transpose_terms <= static_cast<current_diagonal::Index>(count) &&
          work.product_terms <= static_cast<current_diagonal::Index>(count) &&
          work.row_divisions <= static_cast<current_diagonal::Index>(2 * count) && work.applications_completed == 0,
          "长向量与长稀疏列的每个实际回调中断须保4096边界前缀并阻止未完成应用交付");
    }
  }
  for (bool reject : {false, true}) {
    int pair_total = 0;
    {
      CurrentDiagonalPairFixture fixture(reject);
      solver::v122_try_reuse_pair(fixture.h,
          [&](bool probe) { return fixture.pair_current(probe, [&] { ++pair_total; }); },
          fixture.reuse, 2, 1, fixture.system.result, [&] { ++pair_total; });
    }
    for (int cutoff = 1; cutoff <= pair_total; ++cutoff) {
      CurrentDiagonalPairFixture fixture(reject);
      const auto saved_h = fixture.h;
      const auto before = reuse_state::counts(fixture.system.result);
      int checks = 0;
      const auto budget = [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; };
      bool original = false;
      try { solver::v122_try_reuse_pair(fixture.h,
          [&](bool probe) { return fixture.pair_current(probe, budget); },
          fixture.reuse, 2, 1, fixture.system.result, budget); }
      catch (const solver::BudgetExpired&) { original = true; }
      require(original && checks == cutoff && fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
          fixture.main.directions_started == 0 && fixture.main.constant_calls_started == 0 && fixture.energy.started == 0 &&
          coupled_state::current() == &fixture.main && energy_state::current() == &fixture.energy,
          "当前R成功和拒绝两条路径的每个预算中断都须恢复整pair，不能合并部分方向或能量");
      if (!fixture.reuse.events.empty()) {
        const auto& event = fixture.reuse.events.back();
        current_diagonal_prefix(event.current_diagonal);
        const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
        require(std::string_view(event.status) == "budget" && event.current_diagonal.applications_started <= 5 &&
            paid.constant_solves + paid.direction_solves == event.current_diagonal.direct_started &&
            event.public_delta.constant_solves == paid.constant_solves && event.public_delta.direction_solves == paid.direction_solves &&
            event.public_delta.scaled_products == paid.scaled_products &&
            event.public_delta.scaled_refinement_steps == paid.scaled_refinement_steps,
            "新构造和wrapper费用须随失败事件保留，不能漏记初P前或FG内已付的真实工作");
      }
    }
  }
  const std::vector<current_diagonal::Index> starts{0, 1}, indices{0};
  const std::vector<double> values{1}, rows{2}, rhs{3, 4};
  for (int failure = 0; failure < 4; ++failure) {
    current_diagonal::Work work;
    const current_diagonal::ValidatedView view(1, 1, starts, indices, values, rows, work);
    bool original = false;
    try {
      current_diagonal::apply(view, rhs, [&](std::span<const double>) -> std::vector<double> {
        if (failure == 0) throw std::bad_alloc{};
        if (failure == 1) throw std::invalid_argument("当前R旧direct原输入异常");
        if (failure == 2) throw solver::BudgetExpired{};
        throw std::runtime_error("当前R旧direct原始异常");
      }, work);
    } catch (const std::bad_alloc&) { original = failure == 0; }
    catch (const std::invalid_argument&) { original = failure == 1; }
    catch (const solver::BudgetExpired&) { original = failure == 2; }
    catch (const std::runtime_error&) { original = failure == 3; }
    current_diagonal_prefix(work);
    require(original && work.direct_started == 1 && work.direct_completed == 0 && work.transpose_completed == 1 &&
        work.product_started == 0 && work.applications_completed == 0,
        "旧direct的分配、输入、预算及其它异常均须原样传播并停在真实回调前缀");
  }
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto bad_starts = starts, bad_indices = indices;
    auto bad_values = values, bad_rows = rows;
    if (invalid == 0) bad_starts[0] = 1;
    if (invalid == 1) bad_indices[0] = 1;
    if (invalid == 2) bad_rows[0] = 0;
    if (invalid == 3) bad_rows[0] = std::numeric_limits<double>::infinity();
    if (invalid == 4) bad_values[0] = std::numeric_limits<double>::quiet_NaN();
    current_diagonal::Work work;
    bool rejected = false;
    try { current_diagonal::ValidatedView view(1, 1, bad_starts, bad_indices, bad_values, bad_rows, work); }
    catch (const std::invalid_argument&) { rejected = invalid < 2; }
    catch (const current_diagonal::Failure&) { rejected = invalid >= 2; }
    require(rejected && work.validation_started == 1 && work.validation_completed == 0 && work.direct_started == 0,
        "结构非法与数值非法须在direct前按原类型拒绝，不能形成可用的半验证视图");
  }
  for (int invalid = 0; invalid < 5; ++invalid) {
    auto result = system.result;
    std::vector<double> diagonal{7, 0, 11};
    if (invalid == 0) result.ipm_dual_regularization = 2 * *result.ipm_dual_regularization;
    if (invalid == 1) result.ipm_dual_regularization.reset();
    if (invalid == 2) diagonal.pop_back();
    if (invalid == 3) diagonal[0] = -1;
    if (invalid == 4) diagonal[0] = std::numeric_limits<double>::quiet_NaN();
    reuse_state::Probe probe;
    const auto before = reuse_state::counts(result);
    bool rejected = false;
    try { solver::v124_current_rows(*system.factor, diagonal, result, probe, {}); }
    catch (const std::invalid_argument&) { rejected = true; }
    const auto paid = reuse_state::difference(reuse_state::counts(result), before);
    require(rejected && probe.current_rows_completed == 0 && probe.current_rows_started == (invalid < 3 ? 0 : 1) &&
        probe.current_diagonal.direct_started == 0 && paid.factorizations == 0 && paid.direction_solves == 0,
        "当前R必须核对真实旧dual owner与完整行数，非法本次diagonal也不能被当作可恢复候选");
  }
  require(std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows &&
      *system.result.ipm_factorizations == 1, "所有预算与输入错误均不得修改真实旧factor快照或重分解");
  ++budget_cases;
  details.push_back({"current_diagonal_budget_and_errors", "budget"});
}

namespace schur_transport = solver::detail::schur_diagonal_transport;

void check_schur_transport_exact_identity() {
  const std::vector<current_diagonal::Index> starts{0, 1}, indices{0};
  const std::vector<double> values{1}, old_rows{1}, rows{2}, rhs{3, 4};
  current_diagonal::Work current_work;
  const current_diagonal::ValidatedView view(1, 1, starts, indices, values, rows, current_work);
  schur_transport::Cache cache;
  schur_transport::Work work;
  const auto scales = schur_transport::prepare(view, old_rows, 1, 1, &cache, cache, work);
  energy_near(scales[0], std::sqrt(2.0L / 1.5L), "单列T须匹配旧/当前Schur名义对角的平方根比");
  int direct_calls = 0;
  const auto answer = current_diagonal::apply(view, rhs, [&](std::vector<double>& reduced) {
    return schur_transport::apply(scales, reduced, [&](const std::vector<double>& input) {
      ++direct_calls;
      require(input.size() == 2 && input[1] == 0, "传输仍须只调用旧K对[Tq;0]的实际作用");
      return std::vector<double>{input[0] / 2, input[0] / 2};
    }, work);
  }, current_work);
  energy_near(answer[0], 10.0L / 3, "单列传输应精确恢复当前Schur解x=10/3");
  energy_near(answer[1], -1.0L / 3, "单列当前R恢复应给出z=-1/3");
  energy_near(answer[0] + answer[1], rhs[0], "单列组合P须独立满足当前名义上块");
  energy_near(answer[0] - 2 * answer[1], rhs[1], "单列组合P须独立满足当前名义下块");
  require(direct_calls == 1 && cache.valid && cache.old_diagonal.size() == 1 &&
      cache.old_diagonal[0] == 2 && work.prepare_completed == 1 && work.old_build_completed == 1 &&
      work.input_completed == 1 && work.output_completed == 1 && work.input_terms == 1 && work.output_terms == 1,
      "单列prepare只建一次旧对角，每个P真实支付输入和输出两次T乘法");
  current_diagonal::Work same_current;
  const current_diagonal::ValidatedView same(1, 1, starts, indices, values, old_rows, same_current);
  schur_transport::Work same_work;
  const auto same_scales = schur_transport::prepare(same, old_rows, 1, 1, &cache, cache, same_work);
  require(same_scales == std::vector<double>({1}) && same_work.cache_hits == 1 && same_work.old_build_started == 0,
      "同R时T必须为单位，命中同一成功epoch旧对角cache而不重建");
  const std::vector<current_diagonal::Index> zero_starts{0, 0, 0}, empty_indices;
  const std::vector<double> empty_values;
  current_diagonal::Work zero_current;
  const current_diagonal::ValidatedView zero(2, 1, zero_starts, empty_indices, empty_values, rows, zero_current);
  schur_transport::Cache zero_cache;
  schur_transport::Work zero_work;
  const auto zero_scales = schur_transport::prepare(zero, old_rows, 0.25, 1, &zero_cache, zero_cache, zero_work);
  require(zero_scales == std::vector<double>({1, 1}) && zero_cache.old_diagonal == std::vector<long double>({0.25L, 0.25L}),
      "G0且r正时两端Schur对角均为r，不能除零或凭R变化制造虚假T");
  schur_transport::Cache singular_cache;
  schur_transport::Work singular_work;
  bool singular = false;
  try { schur_transport::prepare(zero, old_rows, 0, 1, &singular_cache, singular_cache, singular_work); }
  catch (const schur_transport::Failure&) { singular = true; }
  require(singular && !singular_cache.valid && singular_work.prepare_completed == 0,
      "G0且r0时名义对角为零，不能用单位T掩盖0/0或发布cache");
  ++mapping_cases;
  details.push_back({"schur_transport_exact_identity", "mapping"});
}

void check_schur_transport_duplicates_correlation() {
  {
    const std::vector<current_diagonal::Index> starts{0, 3, 6}, indices{0, 1, 0, 1, 0, 1};
    const std::vector<double> values{1, 3, -1, 2, 0, -2}, old_rows{1, 2}, rows{4, 8};
    current_diagonal::Work current_work;
    const current_diagonal::ValidatedView view(2, 2, starts, indices, values, rows, current_work);
    schur_transport::Cache cache;
    schur_transport::Work work;
    const auto scales = schur_transport::prepare(view, old_rows, 1, 1, &cache, cache, work);
    require(cache.old_diagonal == std::vector<long double>({5.5L, 1}) && scales[1] == 1,
        "同一行的1与-1、2与-2须先代数合并后平方，显式零和非相邻重复项不能增加曲率");
    require(work.merge_terms == 6 && work.unique_terms == 4 && work.old_terms == 4 && work.current_terms == 4 &&
        work.scratch_row_terms == 4 && work.column_terms == 2,
        "重复项、零贡献坐标、scratch初始化和每列归约都须保真实费用，不能将取消项当作未遍历");
    energy_near(scales[0], std::sqrt(5.5L / 2.125L), "重复项合并后的有效G仅剩第二行系数3");
    energy_near(cache.old_diagonal[0] / (static_cast<long double>(scales[0]) * scales[0]), 2.125L,
        "匹配的是B=Jx逆的名义对角，而非把Jx的对角误设为当前Schur对角倒数");
  }
  {
    const std::vector<current_diagonal::Index> starts{0, 2, 4}, indices{0, 1, 0, 1};
    const std::vector<double> values{1, 1, 1, -1}, old_rows{2.0 / 67, 2}, rows{2, 2.0 / 67};
    current_diagonal::Work current_work;
    const current_diagonal::ValidatedView view(2, 2, starts, indices, values, rows, current_work);
    schur_transport::Cache cache;
    schur_transport::Work work;
    const auto scales = schur_transport::prepare(view, old_rows, 0, 1, &cache, cache, work);
    require(scales == std::vector<double>({1, 1}), "相关翻转的两个名义对角相同，T应保持单位");
    const long double diagonal = 1 / static_cast<long double>(old_rows[0]) + 1 / static_cast<long double>(old_rows[1]);
    const long double off_diagonal = 1 / static_cast<long double>(old_rows[0]) - 1 / static_cast<long double>(old_rows[1]);
    const long double eigen_ratio = (diagonal + off_diagonal) / (diagonal - off_diagonal);
    energy_near(eigen_ratio * eigen_ratio, 4489,
        "相关从+33/34翻转为-33/34时，匹配名义对角仍可留下广义条件数4489");
    const long double inverse_diagonal = diagonal / (diagonal * diagonal - off_diagonal * off_diagonal);
    require(std::abs(inverse_diagonal - 1 / diagonal) > 0.1L,
        "非对角旧Schur的逆对角不是原对角倒数，不得误宣称diag(Jx)=1/diagScur");
    for (std::size_t i = 0; i < scales.size(); ++i)
      energy_near(cache.old_diagonal[i] / (static_cast<long double>(scales[i]) * scales[i]), diagonal,
          "B=T逆*Sold*T逆的名义对角必须匹配当前Scur，即使相关结构无法由T修复");
  }
  ++mapping_cases;
  details.push_back({"schur_transport_duplicates_correlation", "mapping"});
}

bool schur_cache_equal(const schur_transport::Cache& left, const schur_transport::Cache& right) {
  return left.valid == right.valid && left.owner == right.owner && left.factor_epoch == right.factor_epoch &&
      left.r_old == right.r_old && left.columns == right.columns && left.rows == right.rows &&
      left.nonzeros == right.nonzeros && left.old_diagonal == right.old_diagonal &&
      left.starts_data == right.starts_data && left.starts_size == right.starts_size &&
      left.row_indices_data == right.row_indices_data && left.row_indices_size == right.row_indices_size &&
      left.values_data == right.values_data && left.values_size == right.values_size &&
      left.old_rows_data == right.old_rows_data && left.old_rows_size == right.old_rows_size;
}

void check_schur_transport_prepare_cache() {
  const std::vector<current_diagonal::Index> starts{0, 1}, indices{0};
  const std::vector<double> values{2}, old_rows{1}, rows{4};
  current_diagonal::Work current_work;
  const current_diagonal::ValidatedView view(1, 1, starts, indices, values, rows, current_work);
  int first_owner = 1, second_owner = 2;
  schur_transport::Cache cache;
  schur_transport::Work first;
  schur_transport::prepare(view, old_rows, 1, 1, &first_owner, cache, first);
  require(cache.valid && cache.owner == &first_owner && cache.factor_epoch == 1 && cache.old_diagonal[0] == 5 &&
      first.old_build_started == 1 && first.old_build_completed == 1 && first.cache_hits == 0,
      "首次成功owner/epoch必须真实构建旧Schur名义对角后完整发布cache");
  const auto saved = cache;
  schur_transport::Work hit;
  schur_transport::prepare(view, old_rows, 1, 1, &first_owner, cache, hit);
  require(hit.cache_hits == 1 && hit.old_build_started == 0 && schur_cache_equal(cache, saved),
      "同owner/epoch复用旧对角，不因本次prepare重新形成旧cache");
  {
    const std::vector<double> another_current{8};
    current_diagonal::Work another_view_work;
    const current_diagonal::ValidatedView another(1, 1, starts, indices, values, another_current, another_view_work);
    schur_transport::Work another_work;
    const auto scales = schur_transport::prepare(another, old_rows, 1, 1, &first_owner, cache, another_work);
    require(another_work.cache_hits == 1 && another_work.old_terms == 0 && schur_cache_equal(cache, saved),
        "当前R改变仍须命中旧epoch cache，只重算本点对角而不把当前R并入旧存储身份");
    energy_near(scales[0], std::sqrt(5.0L / 1.5L), "cache命中不能沿用上一个当前点的T");
  }
  for (int invalid = 0; invalid < 2; ++invalid) {
    auto copied_rows = old_rows;
    schur_transport::Work work;
    bool rejected = false;
    try { schur_transport::prepare(view, invalid == 0 ? std::span<const double>(copied_rows) : std::span<const double>(old_rows),
        invalid == 0 ? 1 : 2, 1, &first_owner, cache, work); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && schur_cache_equal(cache, saved),
        "同一成功owner/epoch不能混入另一个R存储或原始正则，拒绝后原cache保持完整");
  }
  schur_transport::Work owner_change;
  schur_transport::prepare(view, old_rows, 1, 1, &second_owner, cache, owner_change);
  require(cache.owner == &second_owner && owner_change.old_build_completed == 1 && owner_change.cache_hits == 0,
      "新owner即使epoch编号相同也必须重建，不能跨factor借用cache");
  schur_transport::Work epoch_change;
  schur_transport::prepare(view, old_rows, 1, 2, &second_owner, cache, epoch_change);
  require(cache.factor_epoch == 2 && epoch_change.old_build_completed == 1 && epoch_change.cache_hits == 0,
      "成功分解epoch改变必须使旧cache失效并真实重建");
  const auto before = cache;
  int total = 0;
  {
    auto baseline = before;
    schur_transport::Work work;
    schur_transport::prepare(view, old_rows, 1, 3, &second_owner, baseline, work, [&] { ++total; });
  }
  for (bool initially_valid : {false, true}) for (int cutoff = 1; cutoff <= total; ++cutoff) {
    auto candidate = initially_valid ? before : schur_transport::Cache{};
    const auto original_cache = candidate;
    schur_transport::Work work;
    int checks = 0;
    bool original = false;
    try { schur_transport::prepare(view, old_rows, 1, 3, &second_owner, candidate, work,
        [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && checks == cutoff && work.prepare_completed == 0 && schur_cache_equal(candidate, original_cache),
        "prepare每个真实预算中断必须保原异常，未建cache保持空、已有cache保持上一完整epoch");
  }
  for (int failure = 0; failure < 4; ++failure) {
    auto candidate = before;
    schur_transport::Work work;
    bool rejected = false;
    try { schur_transport::prepare(view, old_rows, failure == 0 ? -1 : failure == 1 ?
        std::numeric_limits<double>::quiet_NaN() : 1,
        failure == 2 ? 0 : 3, failure == 3 ? nullptr : &second_owner, candidate, work); }
    catch (const schur_transport::Failure&) { rejected = failure < 2; }
    catch (const std::invalid_argument&) { rejected = failure >= 2; }
    require(rejected && work.prepare_completed == 0 && schur_cache_equal(candidate, before),
        "非法r、owner和epoch不能形成可用T或改写上一完整cache，数值与输入异常保持各自类型");
  }
  {
    constexpr std::size_t count = 4097;
    const std::vector<current_diagonal::Index> long_starts{0, static_cast<current_diagonal::Index>(count)};
    std::vector<current_diagonal::Index> long_indices(count);
    for (std::size_t i = 0; i < count; ++i) long_indices[i] = static_cast<current_diagonal::Index>(i);
    const std::vector<double> long_values(count, 1), long_old_rows(count, 1), long_rows(count, 4);
    current_diagonal::Work long_current;
    const current_diagonal::ValidatedView long_view(1, count, long_starts, long_indices,
        long_values, long_rows, long_current);
    int boundaries = 0;
    schur_transport::Cache complete_cache;
    schur_transport::Work complete;
    schur_transport::prepare(long_view, long_old_rows, 1, 1, &first_owner, complete_cache, complete,
        [&] { ++boundaries; });
    require(complete.old_row_terms == static_cast<schur_transport::Index>(count) &&
        complete.scratch_row_terms == static_cast<schur_transport::Index>(2 * count) &&
        complete.merge_terms == static_cast<schur_transport::Index>(count) &&
        complete.old_terms == static_cast<schur_transport::Index>(count) &&
        complete.current_terms == static_cast<schur_transport::Index>(count),
        "prepare长行、长列及4096块边界的旧R、scratch、合并与两种贡献均须实际遍历");
    for (int cutoff = 1; cutoff <= boundaries; ++cutoff) {
      schur_transport::Cache candidate;
      schur_transport::Work work;
      int checks = 0;
      bool original = false;
      try { schur_transport::prepare(long_view, long_old_rows, 1, 1, &first_owner, candidate, work,
          [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
      catch (const solver::BudgetExpired&) { original = true; }
      require(original && checks == cutoff && !candidate.valid && candidate.old_diagonal.empty() &&
          work.prepare_completed == 0 && work.old_build_completed <= work.old_build_started,
          "长prepare任一分块预算失败不能提交已经部分形成的旧对角cache");
    }
  }
  ++budget_cases;
  details.push_back({"schur_transport_prepare_cache", "budget"});
}

void check_schur_transport_apply_budget() {
  constexpr std::size_t dimension = 4097;
  const std::vector<double> scales(dimension, 2);
  const auto exercise = [&](schur_transport::Work& work, int& direct_calls, const std::function<void()>& budget) {
    std::vector<double> rhs(dimension + 1, 1); rhs.back() = 0;
    return schur_transport::apply(scales, rhs, [&](const std::vector<double>& input) {
      ++direct_calls;
      require(input.back() == 0, "传输只能乘原始列，不得改变旧K的零下块输入");
      return input;
    }, work, budget);
  };
  schur_transport::Work baseline;
  int calls = 0, total = 0;
  const auto answer = exercise(baseline, calls, [&] { ++total; });
  require(calls == 1 && answer.front() == 4 && answer[dimension - 1] == 4 && answer.back() == 0 &&
      baseline.input_terms == static_cast<schur_transport::Index>(dimension) &&
      baseline.output_terms == static_cast<schur_transport::Index>(dimension) &&
      baseline.input_completed == 1 && baseline.direct_completed == 1 && baseline.output_completed == 1,
      "4097列的输入/输出各乘一次T，新增扫描费用与真实direct严格守恒");
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    schur_transport::Work work;
    int direct_calls = 0, checks = 0;
    bool original = false;
    std::vector<double> output{101};
    try { output = exercise(work, direct_calls, [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && checks == cutoff && output == std::vector<double>({101}) &&
        direct_calls == work.direct_started && work.direct_completed <= work.direct_started &&
        work.input_completed <= work.input_started && work.output_completed <= work.output_started &&
        work.input_terms <= static_cast<schur_transport::Index>(dimension) &&
        work.output_terms <= static_cast<schur_transport::Index>(dimension) && work.output_completed == 0,
        "每个乘T分块预算中断须保真实输入/direct/输出前缀，不能交付半缩放方向");
  }
  for (int failure = 0; failure < 4; ++failure) {
    schur_transport::Work work;
    std::vector<double> rhs{3, 0};
    const std::vector<double> one_scale{2};
    bool original = false;
    try { schur_transport::apply(one_scale, rhs, [&](const std::vector<double>&) -> std::vector<double> {
      if (failure == 0) throw std::bad_alloc{};
      if (failure == 1) throw std::invalid_argument("传输旧P原输入异常");
      if (failure == 2) throw solver::BudgetExpired{};
      throw std::runtime_error("传输旧P原始异常");
    }, work); }
    catch (const std::bad_alloc&) { original = failure == 0; }
    catch (const std::invalid_argument&) { original = failure == 1; }
    catch (const solver::BudgetExpired&) { original = failure == 2; }
    catch (const std::runtime_error&) { original = failure == 3; }
    require(original && work.input_completed == 1 && work.direct_started == 1 && work.direct_completed == 0 &&
        work.output_started == 0, "旧P所有原异常必须穿过传输原样传播，不能假装有限搜索拒绝");
  }
  for (int failure = 0; failure < 6; ++failure) {
    schur_transport::Work work;
    std::vector<double> rhs{failure == 2 ? std::numeric_limits<double>::max() :
        failure == 4 ? std::numeric_limits<double>::denorm_min() : 1, 0};
    const std::vector<double> scales_one{failure == 0 ? 0 : failure == 1 ?
        std::numeric_limits<double>::infinity() : failure >= 4 ? std::numeric_limits<double>::denorm_min() : 2};
    bool rejected = false;
    try { schur_transport::apply(scales_one, rhs, [&](const std::vector<double>& input) {
      return std::vector<double>{failure == 3 ? std::numeric_limits<double>::max() :
          failure == 5 ? std::numeric_limits<double>::denorm_min() : input[0], 0};
    }, work); }
    catch (const schur_transport::Failure&) { rejected = true; }
    const bool in_output = failure == 3 || failure == 5;
    require(rejected && work.input_started == 1 && work.input_completed == (in_output ? 1 : 0) &&
        work.direct_started == (in_output ? 1 : 0) && work.direct_completed == (in_output ? 1 : 0) &&
        work.output_started == (in_output ? 1 : 0) && work.output_completed == 0,
        "非正T及两侧乘法溢出/非零下溢必须按实际阶段拒绝，不能静默交付零或无穷方向");
  }
  ++budget_cases;
  details.push_back({"schur_transport_apply_budget", "budget"});
}

void check_schur_transport_real_normal() {
  WarmRegularizationSystem system;
  const auto old_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
      system.factor->v93_regularized_h().end());
  const auto old_r = system.result.ipm_primal_regularization, old_delta = system.result.ipm_dual_regularization;
  solver::detail::factor_reuse::Probe probe;
  const std::vector<double> diagonal{7, 0, 11};
  const auto rows = solver::v124_current_rows(*system.factor, diagonal, system.result, probe);
  const current_diagonal::ValidatedView view(2, 3, system.cone.starts, system.cone.rows, system.cone.values,
      rows, probe.current_diagonal);
  schur_transport::Cache cache;
  const auto scales = schur_transport::prepare(view, system.factor->v93_regularized_h(), *old_r,
      1, system.factor.get(), cache, probe.transport);
  const std::array<long double, 2> old_diagonal{*old_r + 1 / static_cast<long double>(old_rows[0]) +
      9 / static_cast<long double>(old_rows[1]) + 25 / static_cast<long double>(old_rows[2]),
      *old_r + 4 / static_cast<long double>(old_rows[0]) + 1 / static_cast<long double>(old_rows[1])};
  const std::array<long double, 2> current_diagonal{*old_r + 1 / static_cast<long double>(rows[0]) +
      9 / static_cast<long double>(rows[1]) + 25 / static_cast<long double>(rows[2]),
      *old_r + 4 / static_cast<long double>(rows[0]) + 1 / static_cast<long double>(rows[1])};
  for (std::size_t i = 0; i < scales.size(); ++i) {
    energy_near(cache.old_diagonal[i], old_diagonal[i], "旧Schur对角须使用完整未均衡G及实际旧R，单例曲率只计一次");
    energy_near(scales[i], std::sqrt(old_diagonal[i]) / std::sqrt(current_diagonal[i]),
        "真实均衡/单例/等式系统的T须匹配原始坐标名义Schur对角");
  }
  const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5};
  const auto before = solver::detail::factor_reuse::counts(system.result);
  const auto answer = solver::v124_apply_current(view, rhs, *system.factor, system.options, system.start,
      system.result, false, true, probe.current_diagonal, {}, scales, &probe.transport);
  std::vector<double> transported_rhs(5, 0);
  for (std::size_t column = 0; column < 2; ++column) {
    long double q = rhs[column];
    for (auto term = system.cone.starts[column]; term < system.cone.starts[column + 1]; ++term) {
      const auto row_index = static_cast<std::size_t>(system.cone.rows[static_cast<std::size_t>(term)]);
      q += static_cast<long double>(system.cone.values[static_cast<std::size_t>(term)]) *
          (static_cast<long double>(rhs[2 + row_index]) / rows[row_index]);
    }
    const double stored_q = static_cast<double>(q);
    transported_rhs[column] = static_cast<double>(static_cast<long double>(stored_q) * scales[column]);
  }
  const auto actual_old_p = system.factor->solve_homogeneous_direct(transported_rhs, system.options,
      system.start, system.result, false);
  for (std::size_t i = 0; i < 2; ++i)
    require(answer[i] == static_cast<double>(static_cast<long double>(actual_old_p[i]) * scales[i]),
        "新P必须逐值等于同一真实旧近似P的T输入和T输出，不拿dense精确逆作前向合同");
  const std::array<long double, 3> image{static_cast<long double>(answer[0]) + 2 * answer[1],
      3 * static_cast<long double>(answer[0]) - answer[1], 5 * static_cast<long double>(answer[0])};
  for (std::size_t i = 0; i < rows.size(); ++i)
    energy_near(image[i] - static_cast<long double>(rows[i]) * answer[2 + i], rhs[2 + i],
        "最终Tx之后才算Gx和恢复z，当前名义行独立闭合且不沿用旧z");
  const auto paid = solver::detail::factor_reuse::difference(solver::detail::factor_reuse::counts(system.result), before);
  require(paid.direction_solves == 2 && paid.refinement_steps == 1 && paid.scaled_refinement_steps == 1 &&
      paid.factorizations == 0 && probe.transport.input_terms == 2 && probe.transport.output_terms == 2 &&
      probe.transport.direct_completed == 1 && *system.result.ipm_factorizations == 1 &&
      system.result.ipm_primal_regularization == old_r && system.result.ipm_dual_regularization == old_delta &&
      std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == old_rows,
      "真实传输新增两次T扫描和一次旧P，参考回代独立计费而旧数值epoch保持只读");
  schur_transport::Work hit;
  schur_transport::prepare(view, system.factor->v93_regularized_h(), *old_r, 1, system.factor.get(), cache, hit);
  require(hit.cache_hits == 1 && hit.old_build_started == 0, "同一真实成功factor可跨当前点命中旧对角cache");
  cache.valid = false;
  auto expired = system.options; expired.time_limit = 0;
  bool stopped = false;
  try { system.factor->factor(diagonal, expired, system.start, system.result); }
  catch (const solver::BudgetExpired&) { stopped = true; }
  require(stopped && !cache.valid && *system.result.ipm_factorizations == 1,
      "fresh开始先失效旧cache，分解入口预算失败也不能重新声称旧cache属于当前成功epoch");
  system.factor->factor(diagonal, system.options, system.start, system.result, 7);
  schur_transport::Work rebuilt;
  const auto new_scales = schur_transport::prepare(view, system.factor->v93_regularized_h(),
      *system.result.ipm_primal_regularization, 2, system.factor.get(), cache, rebuilt);
  require(cache.valid && cache.factor_epoch == 2 && rebuilt.old_build_completed == 1 &&
      rebuilt.cache_hits == 0 && new_scales == std::vector<double>({1, 1}) && *system.result.ipm_factorizations == 2,
      "真实新分解成功后须在新owner/epoch重建旧对角，此时同R应恢复单位T");
  ++mapping_cases;
  details.push_back({"schur_transport_real_normal", "mapping"});
}

struct TransportScalarSystem {
  solver::ConeModel cone;
  solver::Point point;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  solver::detail::IpmHomogeneousScaling scaling;
  std::unique_ptr<solver::KktFactor> factor;
  TransportScalarSystem() {
    options.ipm.linear_solver = "qdldl";
    cone.costs = {0}; cone.rhs = {1}; cone.starts = {0, 1}; cone.rows = {0}; cone.values = {1};
    cone.inequality = {true}; cone.barrier_rows = 1;
    point.x = {0}; point.s = {2}; point.z = {1};
    scaling = solver::detail::make_ipm_homogeneous_scaling(1, {true}, point.s, point.z, 2, 8);
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    factor->factor({1}, options, start, result);
  }
};

std::pair<std::vector<double>, long double> transport_test_constant(TransportScalarSystem& system,
    const std::vector<double>& rows, const current_diagonal::ValidatedView& view,
    solver::detail::factor_reuse::Probe& probe, std::span<const double> scales,
    const std::function<void()>& budget = {}) {
  const auto rhs = solver::detail::ipm_homogeneous_scaled_constant_rhs(system.scaling,
      system.cone.costs, system.cone.rhs, budget);
  auto* log = coupled_state::current();
  require(log != nullptr, "传输常量必须在真实联立计费作用域执行");
  if (budget) budget();
  ++log->constant_calls_started;
  auto constant = solver::v124_apply_current(view, rhs, *system.factor, system.options, system.start,
      system.result, true, false, probe.current_diagonal, budget, scales, &probe.transport);
  ++log->constant_calls_completed;
  const auto denominator = solver::v124_energy_denominator(*system.factor, 1, constant, rows,
      system.result, budget, 2);
  solver::v125_scalar_diagnostics(system.scaling, system.cone.costs, system.cone.rhs, constant,
      denominator, probe.scalar_consistency, budget);
  return {std::move(constant), denominator};
}

void check_schur_transport_constant_physics() {
  TransportScalarSystem system;
  coupled_state::Log main; main.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  coupled_state::Scope scope(&main);
  energy_state::Scope energy_scope(&energy);
  solver::detail::factor_reuse::Probe probe;
  const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, probe);
  const current_diagonal::ValidatedView view(1, 1, system.cone.starts, system.cone.rows, system.cone.values,
      rows, probe.current_diagonal);
  schur_transport::Cache cache;
  const auto scales = schur_transport::prepare(view, system.factor->v93_regularized_h(),
      *system.result.ipm_primal_regularization, 1, system.factor.get(), cache, probe.transport);
  require(scales.size() == 1 && scales[0] > 1, "真实跨点常量与方向须使用非单位且同一份T");
  const auto constant = transport_test_constant(system, rows, view, probe, scales);
  const long double expected = (1 + static_cast<long double>(*system.result.ipm_primal_regularization) *
      constant.first[0] * constant.first[0]) + static_cast<long double>(rows[0]) * constant.first[1] * constant.first[1];
  energy_near(constant.second, expected, "positive approximation须读取传输后的最终x/z及当前R，不能用中间旧P向量");
  const long double natural = 1 - static_cast<long double>(constant.first[1]) / system.scaling.d;
  require(probe.scalar_consistency.started == 1 && probe.scalar_consistency.completed == 1 &&
      probe.scalar_consistency.terms == 2 && probe.scalar_consistency.natural_denominator &&
      probe.scalar_consistency.positive_denominator && probe.scalar_consistency.relative_difference,
      "常量必须完整记录自然分母与正近似分母观察量，不能据观察改变原验收或另付G产品");
  energy_near(*probe.scalar_consistency.natural_denominator, natural, "自然分母观察只从当前d与常量扫描获得");
  energy_near(*probe.scalar_consistency.positive_denominator, expected, "自然分母诊断不能替换正近似D");
  const auto direction = solver::homogeneous_scaled_direction(system.cone, system.point, 2, 8, {-1}, {-1}, 0,
      1, {0}, 0, constant.first, system.scaling, constant.second, *system.factor, system.options, system.start,
      system.result, "corrector", 2, false, solver::detail::factor_reuse::reuse_krylov_budget,
      &view, &probe.current_diagonal, scales, &probe.transport);
  require(direction.v94_wide.has_value(), "非空T生产adapter须实际交付完整宽方向");
  const auto& wide = *direction.v94_wide;
  const std::vector<long double> physical{wide.x[0], wide.z[0], wide.s[0], wide.tau, wide.kappa};
  const auto explicit_product = [&](const auto& value) {
    return std::vector<long double>{value[1], static_cast<long double>(value[0]) + value[2] - value[3],
        (static_cast<long double>(value[1]) + value[4]) / 2,
        (static_cast<long double>(value[2]) + 2 * value[1]) / system.scaling.lambda[0],
        2 * static_cast<long double>(value[3]) + 0.5L * value[4]};
  };
  coupled_independent_equations(system.scaling, {1, 1, 0, 0, 0}, physical, explicit_product);
  require(main.last && std::string_view(main.last->work.status) == "accepted" &&
      probe.transport.input_completed == probe.current_diagonal.applications_completed &&
      probe.transport.output_completed == probe.current_diagonal.applications_completed &&
      probe.transport.input_terms == probe.current_diagonal.applications_completed &&
      probe.transport.output_terms == probe.current_diagonal.applications_completed &&
      main.constant_calls_completed == 1 && main.directions_completed == 1 && energy.completed == 1 &&
      probe.scalar_consistency.completed == 1 && *system.result.ipm_factorizations == 1,
      "真实常量与方向共用同一T/P，所有左右T费用守恒，诊断只执行一次且保原物理strict");
  solver::detail::factor_reuse::ScalarConsistency disagreement;
  const std::vector<double> observed_costs{20}, observed_bounds{0}, observed_constant{1, 0};
  const auto counts_before = solver::detail::factor_reuse::counts(system.result);
  solver::v125_scalar_diagnostics(system.scaling, observed_costs, observed_bounds, observed_constant, 2, disagreement);
  const auto diagnostic_paid = solver::detail::factor_reuse::difference(
      solver::detail::factor_reuse::counts(system.result), counts_before);
  require(disagreement.completed == 1 && disagreement.natural_denominator == -9 &&
      disagreement.positive_denominator == 2 && disagreement.relative_difference &&
      diagnostic_paid.constant_solves == 0 && diagnostic_paid.direction_solves == 0 && diagnostic_paid.scaled_products == 0,
      "自然分母为负也只记录观察，不能替换正D、增加G产品或从诊断中产生接受决策");
  ++mapping_cases;
  details.push_back({"schur_transport_constant_physics", "mapping"});
}

struct TransportPairFixture {
  TransportScalarSystem system;
  coupled_state::Log main;
  energy_state::Log energy;
  solver::detail::factor_reuse::Log reuse;
  coupled_state::Scope main_scope{&main};
  energy_state::Scope energy_scope{&energy};
  schur_transport::Cache cache;
  std::vector<std::vector<long double>> inverse;
  std::vector<double> h{-1};
  bool reject_corrector;
  int affine_calls = 0, corrector_calls = 0;

  explicit TransportPairFixture(bool reject) : reject_corrector(reject) {
    main.enabled = energy.enabled = reuse.enabled = true;
    coupled_state::Log setup; setup.enabled = true;
    energy_state::Log setup_energy; setup_energy.enabled = true;
    coupled_state::Scope setup_scope(&setup);
    energy_state::Scope setup_energy_scope(&setup_energy);
    solver::detail::factor_reuse::Probe preparation;
    const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, preparation);
    const current_diagonal::ValidatedView view(1, 1, system.cone.starts, system.cone.rows, system.cone.values,
        rows, preparation.current_diagonal);
    const auto scales = schur_transport::prepare(view, system.factor->v93_regularized_h(),
        *system.result.ipm_primal_regularization, 1, system.factor.get(), cache, preparation.transport);
    const auto constant = transport_test_constant(system, rows, view, preparation, scales);
    // 只为隔离事务构造显式A=D*P逆；所有数学准备费用在实际probe快照之前已真实支付。
    std::vector<std::vector<long double>> matrix(5, std::vector<long double>(5));
    for (std::size_t column = 0; column < 5; ++column) {
      std::vector<double> basis(5, 0); basis[column] = 1;
      const auto image = precondition(basis, false, constant, &view, &preparation, scales, {});
      for (std::size_t row_index = 0; row_index < 5; ++row_index) matrix[row_index][column] = image[row_index];
    }
    inverse.assign(5, std::vector<long double>(5));
    for (std::size_t column = 0; column < 5; ++column) {
      std::vector<double> basis(5, 0); basis[column] = 1;
      const auto image = normal_inverse_dense_solve(matrix, basis);
      for (std::size_t row_index = 0; row_index < 5; ++row_index) inverse[row_index][column] = image[row_index];
    }
  }

  std::vector<double> precondition(std::span<const double> input, bool refinement,
      const std::pair<std::vector<double>, long double>& constant,
      const current_diagonal::ValidatedView* view, solver::detail::factor_reuse::Probe* probe,
      std::span<const double> scales, const std::function<void()>& budget) {
    const auto rhs = solver::detail::ipm_homogeneous_scaled_reduced_rhs(system.scaling, input, budget);
    const auto reduced = view ? solver::v124_apply_current(*view, rhs, *system.factor, system.options,
        system.start, system.result, false, refinement, probe->current_diagonal, budget, scales, &probe->transport) :
        system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false, refinement);
    return solver::detail::ipm_homogeneous_scaled_recover(system.scaling, system.cone.costs, system.cone.rhs,
        input, reduced, constant.first, constant.second, budget);
  }

  solver::HomogeneousDirection direction(const char* role, const std::array<double, 5>& diagonal,
      const std::pair<std::vector<double>, long double>& constant, std::size_t cap,
      const current_diagonal::ValidatedView* view, solver::detail::factor_reuse::Probe* probe,
      std::span<const double> scales, const std::function<void()>& budget) {
    const std::vector<double> rhs{1, 1, 0, 0, 0};
    const auto p = [&](std::span<const double> input, bool refinement) {
      return precondition(input, refinement, constant, view, probe, scales, budget);
    };
    const auto explicit_product = [&](const auto& input) {
      std::vector<long double> image(5, 0);
      for (std::size_t row_index = 0; row_index < 5; ++row_index) {
        for (std::size_t column = 0; column < 5; ++column)
          image[row_index] += inverse[row_index][column] * input[column];
        image[row_index] *= diagonal[row_index];
      }
      return image;
    };
    const auto product = [&]<class Scalar>(std::span<const Scalar> input, std::vector<long double>& image) {
      ++*system.result.ipm_homogeneous_scaled_products;
      image = explicit_product(input);
    };
    bool forced = false;
    auto answer = solver::v100_solve_coupled(system.scaling, rhs, p, product, product, budget,
        role, 2, &forced, true, true, false, cap);
    const auto image = explicit_product(answer);
    const auto tolerances = solver::detail::ipm_homogeneous_scaled_tolerances(system.scaling, rhs);
    const coupled_kernel::Layout layout{1, 1, 1};
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      const long double strict = tolerances[layout.block(i)];
      const long double limit = forced && i < 2 ? std::max(strict, 0.1L) : strict;
      require(std::abs(image[i] - rhs[i]) <= limit, "非空T整组的交付须独立满足原五块strict或forcing物理门槛");
    }
    return solver::v100_deliver_direction(system.scaling, std::move(answer), forced, system.result, budget);
  }

  solver::HomogeneousDirection pair(bool reuse_probe, const std::function<void()>& budget = {}) {
    solver::detail::factor_reuse::Probe* probe = nullptr;
    std::vector<double> rows, scales;
    std::optional<current_diagonal::ValidatedView> view;
    std::pair<std::vector<double>, long double> constant;
    if (reuse_probe) {
      probe = &reuse.events.back();
      probe->current_diagonal.columns = 1; probe->current_diagonal.rows = 1; probe->current_diagonal.nonzeros = 1;
      probe->transport.columns = 1; probe->transport.rows = 1; probe->transport.nonzeros = 1;
      rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, *probe, budget);
      view.emplace(1, 1, system.cone.starts, system.cone.rows, system.cone.values, rows, probe->current_diagonal, budget);
      scales = schur_transport::prepare(*view, system.factor->v93_regularized_h(),
          *system.result.ipm_primal_regularization, 1, system.factor.get(), cache, probe->transport, budget);
      constant = transport_test_constant(system, rows, *view, *probe, scales, budget);
    } else {
      constant = factor_reuse_test_constant(system.cone, *system.factor, system.scaling,
          system.options, system.start, system.result, 2);
    }
    const auto cap = reuse_probe ? solver::detail::factor_reuse::reuse_krylov_budget :
        coupled_kernel::max_window * coupled_kernel::max_cycles;
    ++affine_calls;
    const auto affine = direction("affine", {2, 1, 1, 1, 1}, constant, cap,
        view ? &*view : nullptr, probe, scales, budget);
    require(affine.v94_wide.has_value(), "非空T完整pair首方向须实际交付");
    h[0] = 7;
    ++corrector_calls;
    return direction("corrector", reject_corrector ? std::array<double, 5>{2, 100, 1, 1, 1} :
        std::array<double, 5>{2, 2.1, 1, 1, 1}, constant, cap,
        view ? &*view : nullptr, probe, scales, budget);
  }
};

void check_schur_transport_pair_transaction() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (bool reject : {false, true}) {
    TransportPairFixture fixture(reject);
    const auto saved_h = fixture.h;
    const auto before = reuse_state::counts(fixture.system.result);
    const auto answer = solver::v122_try_reuse_pair(fixture.h,
        [&](bool probe) { return fixture.pair(probe); }, fixture.reuse, 2, 1, fixture.system.result);
    const auto& event = fixture.reuse.events.back();
    const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
    require(answer.has_value() != reject && fixture.affine_calls == 1 && fixture.corrector_calls == 1 &&
        event.current_diagonal.applications_completed == 5 && event.transport.prepare_completed == 1 &&
        event.transport.cache_hits == 1 && event.transport.old_build_started == 0 && event.transport.input_completed == 5 &&
        event.transport.output_completed == 5 && event.transport.input_terms == 5 && event.transport.output_terms == 5 &&
        event.transport.direct_completed == 5 && event.scalar_consistency.completed == 1 && event.scalar_consistency.terms == 2 &&
        paid.constant_solves == 1 && paid.direction_solves == 4 && paid.refinement_steps == 2 &&
        paid.scaled_refinement_steps == 2 && paid.scaled_products == 6 && paid.factorizations == 0 &&
        event.coupled.krylov_completed == 2 && event.coupled.candidate_forcing_checks == 1 &&
        event.coupled.candidate_forcing_returns == (reject ? 0 : 1),
        "非空T常量和两方向必须共享本点同一prepare/T/P，五次旧direct及十次T扫描费用真实保留");
    if (!reject) {
      require(fixture.h == std::vector<double>({7}) && fixture.reuse.reused_steps == 1 &&
          fixture.main.directions_completed == 2 && fixture.main.krylov_completed == 2 && fixture.energy.completed == 1 &&
          std::string_view(event.status) == "accepted", "仅非空T整组完整成功才合并主账本和选定h");
    } else {
      require(fixture.h == saved_h && fixture.reuse.reused_steps == 0 && fixture.main.directions_started == 0 &&
          fixture.energy.started == 0 && std::string_view(event.status) == "bounded_rejected",
          "非空T corrector拒绝须恢复原h并丢弃整组，实际新R/T/旧P费用不回滚");
      fixture.cache.valid = false;
      fixture.system.factor->factor({2}, fixture.system.options, fixture.system.start, fixture.system.result);
      require(fixture.h == saved_h && !fixture.cache.valid, "fresh前须保原h且使旧成功epoch的T cache失效");
      const auto fresh = fixture.pair(false);
      require(fresh.v94_wide && fixture.affine_calls == 2 && fixture.corrector_calls == 2 &&
          fixture.main.directions_completed == 2 && !fixture.cache.valid && *fixture.system.result.ipm_factorizations == 2,
          "fresh96真实新分解后须重做整pair，不得继续使用被拒探测的旧T或旧方向");
    }
  }
  for (bool reject : {false, true}) {
    int total = 0;
    {
      TransportPairFixture fixture(reject);
      solver::v122_try_reuse_pair(fixture.h, [&](bool probe) { return fixture.pair(probe, [&] { ++total; }); },
          fixture.reuse, 2, 1, fixture.system.result, [&] { ++total; });
    }
    for (int cutoff = 1; cutoff <= total; ++cutoff) {
      TransportPairFixture fixture(reject);
      const auto saved_h = fixture.h;
      const auto saved_cache = fixture.cache;
      int checks = 0;
      const auto budget = [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; };
      bool original = false;
      try { solver::v122_try_reuse_pair(fixture.h, [&](bool probe) { return fixture.pair(probe, budget); },
          fixture.reuse, 2, 1, fixture.system.result, budget); }
      catch (const solver::BudgetExpired&) { original = true; }
      require(original && checks == cutoff && fixture.h == saved_h && schur_cache_equal(fixture.cache, saved_cache) &&
          fixture.reuse.reused_steps == 0 && fixture.main.directions_started == 0 && fixture.energy.started == 0,
          "非空T两条pair路径全部prepare/诊断/左右T/交付预算中断须保原异常与h，不合并半组或改旧cache");
      if (!fixture.reuse.events.empty()) {
        const auto& event = fixture.reuse.events.back();
        require(std::string_view(event.status) == "budget" && event.transport.direct_started <= 5 &&
            event.transport.input_terms <= 5 && event.transport.output_terms <= 5 &&
            event.public_delta.constant_solves + event.public_delta.direction_solves == event.transport.direct_started,
            "非空T异常事件须保真实新扫描与旧direct公共费用前缀，不能从失败探测中减回");
      }
    }
  }
  ++transaction_cases;
  details.push_back({"schur_transport_pair_transaction", "transaction"});
}

namespace natural_schur = solver::detail::natural_schur;

void natural_schur_reliable(const natural_schur::Work& work, long double value, std::size_t terms) {
  require(work.started == 1 && work.completed == 1 && std::string_view(work.decision) == "reliable" &&
      work.natural && *work.natural == value && work.lower && work.upper &&
      *work.lower <= value && value <= *work.upper && (*work.upper < 0 || *work.lower > 0) &&
      work.abs_products && work.error_bound && *work.error_bound >= 0 && work.margin && *work.margin > 0 &&
      work.terms == static_cast<std::int64_t>(terms) && work.divisions == static_cast<std::int64_t>(terms) &&
      work.products == static_cast<std::int64_t>(terms) && work.accumulations == static_cast<std::int64_t>(terms) &&
      work.abs_accumulations == static_cast<std::int64_t>(terms) && work.final_subtractions == 1 &&
      work.bound_operations == static_cast<std::int64_t>(4 * terms + 4) &&
      work.enclosure_steps == static_cast<std::int64_t>(6 * terms + 4),
      "有符号自然分母必须由完整、严格排零且包含中心的区间证明，所有中心和包围运算按真实项数计费");
}

void check_natural_schur_signed_recover() {
  using namespace solver::detail;
  const std::vector<double> empty, constant{1}, reduced{3}, rhs{1, 4, 5};
  for (double cost : {2.0, 0.5}) {
    const std::vector<double> costs{cost};
    natural_schur::Work work;
    const auto denominator = natural_schur::compute(costs, empty, constant, 1, work);
    natural_schur_reliable(work, denominator, 1);
    require(denominator == (cost == 2 ? -1 : 0.5L), "自然分母允许负数和小于1的正数，不套用正能量下限");
    const auto scaling = make_ipm_homogeneous_scaling(1, {}, empty, empty, 1, 1);
    const auto answer = ipm_homogeneous_scaled_recover(scaling, costs, empty, rhs, reduced, constant,
        denominator, {}, IpmScalarDenominatorPolicy::signed_schur);
    if (cost == 2) require(answer == std::vector<double>({-4, -7, 12}),
        "负N=-1的解析方向须保留符号，得到x=-4、dt=-7、dk=12");
    else require(answer == std::vector<double>({8, 5, 0}), "正N=1/2须得到x=8、dt=5、dk=0");
    energy_near(static_cast<long double>(cost) * answer[0] + answer[2], rhs[1],
        "自然标量消元须独立闭合gap行");
    energy_near(static_cast<long double>(answer[1]) + answer[2], rhs[2],
        "自然标量消元仍须独立闭合tau/kappa行");
    if (cost == 2) {
      bool rejected = false;
      try { ipm_homogeneous_scaled_recover(scaling, costs, empty, rhs, reduced, constant, denominator); }
      catch (const solver::NumericalFailure&) { rejected = true; }
      require(rejected, "默认正能量恢复合同必须保持，不能因新增signed策略放行负分母");
      require(std::abs(static_cast<long double>(cost) * answer[1] - rhs[0]) > 1,
          "gap与tau行闭合不能冒称完整五块合格，负N解析夹具的dual仍明确失败");
    }
  }
  natural_schur::Work scalar;
  const auto denominator = natural_schur::compute(empty, empty, empty, 1, scalar);
  natural_schur_reliable(scalar, denominator, 0);
  const auto empty_scaling = make_ipm_homogeneous_scaling(0, {}, empty, empty, 1, 1);
  const auto answer = ipm_homogeneous_scaled_recover(empty_scaling, empty, empty, std::vector<double>{4, 5},
      empty, empty, denominator, {}, IpmScalarDenominatorPolicy::signed_schur);
  require(denominator == 1 && answer == std::vector<double>({1, 4}), "空维自然分母为1，仍完成两条真实标量方程");
  ++mapping_cases;
  details.push_back({"natural_schur_signed_recover", "mapping"});
}

void check_natural_schur_enclosure_rounding() {
  const std::vector<double> empty;
  const double epsilon = std::ldexp(1.0, -30);
  const std::vector<double> costs{1 + epsilon}, constant{1 - epsilon};
  natural_schur::Work wide;
  const auto natural = natural_schur::compute(costs, empty, constant, 1, wide);
  natural_schur_reliable(wide, natural, 1);
  require(natural == std::ldexp(1.0L, -60), "先LD乘法应保留(1+2^-30)(1-2^-30)的真实非零自然分母");
  volatile double rounded_product = costs[0] * constant[0];
  require(1 - rounded_product == 0, "本夹具必须确实区分double点积的零与LD可认证的非零");
  natural_schur::Work divided;
  const auto third = natural_schur::compute(std::vector<double>{1}, empty, std::vector<double>{1}, 3, divided);
  natural_schur_reliable(divided, third, 1);
  const long double expected = 1 - 1.0L / 3;
  require(third == expected && third != static_cast<long double>(1 - 1.0 / 3),
      "q=c/d及其后的乘积必须保持LD，不能先做double除法再补宽摘要");
  for (int scenario = 0; scenario < 3; ++scenario) {
    const double small = std::ldexp(1.0, -32);
    const std::vector<double> near_costs{scenario == 0 ? 1 : 1 + small};
    const std::vector<double> near_constant{scenario == 0 ?
        static_cast<double>(1 + std::ldexp(1.0L, -54)) : 1 - small};
    natural_schur::Work work;
    bool rejected = false;
    if (scenario < 2) {
      try { natural_schur::compute(near_costs, empty, near_constant, 1, work); }
      catch (const natural_schur::Failure&) { rejected = true; }
    } else {
      const std::vector<double> negative_costs{1 + small, std::ldexp(1.0, -31)};
      const std::vector<double> negative_constant{1 - small, std::ldexp(1.0, -32)};
      try { natural_schur::compute(negative_costs, empty, negative_constant, 1, work); }
      catch (const natural_schur::Failure&) { rejected = true; }
    }
    require(rejected && work.completed == 0 && std::string_view(work.decision) == "uncertain" &&
        work.lower && work.upper && *work.lower <= 0 && *work.upper >= 0,
        "实际double输入的零与正负近消去都须依据区间排零失败拒绝，不能使用舍入前常量或人为epsilon补救");
  }
  ++mapping_cases;
  details.push_back({"natural_schur_enclosure_rounding", "mapping"});
}

void check_natural_schur_budget_inputs() {
  constexpr std::size_t count = 4097;
  const std::vector<double> costs(count, 1), empty, constant(count, 1);
  natural_schur::Work full;
  int total = 0;
  const auto natural = natural_schur::compute(costs, empty, constant, 1, full, [&] { ++total; });
  natural_schur_reliable(full, natural, count);
  require(natural == -4096 && total > 0 && full.environment_checks == full.budget_checks,
      "4097项完整负N必须跨4096分块，成功预算返回后逐次重新验证环境");
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    natural_schur::Work work;
    int checks = 0;
    long double output = 101;
    bool original = false;
    try { output = natural_schur::compute(costs, empty, constant, 1, work,
        [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; }); }
    catch (const solver::BudgetExpired&) { original = true; }
    require(original && checks == cutoff && output == 101 && work.completed == 0 &&
        work.environment_checks + 1 == work.budget_checks && work.terms <= static_cast<std::int64_t>(count),
        "每个实际包围预算中断须传播原异常，不交付分母或虚记未返回回调的环境检查");
    if (work.final_subtractions == 0)
      require(!work.natural && !work.lower && !work.upper && !work.error_bound && !work.margin,
          "完整包围未形成前不能发布半份自然分母证书");
  }
  for (int invalid = 0; invalid < 5; ++invalid) {
    natural_schur::Work work;
    if (invalid == 4) work.started = 1;
    const std::vector<double> short_constant(invalid == 0 ? 0 : 1, 1);
    const double d = invalid == 1 ? 0 : invalid == 2 ? -1 : invalid == 3 ? infinity : 1;
    bool rejected = false;
    try { natural_schur::compute(std::vector<double>{1}, empty, short_constant, d, work); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && work.completed == 0 && work.terms == 0,
        "尺寸、d或重复使用账本非法时须作为输入错误拒绝，不能创建新证书");
    if (invalid >= 1 && invalid <= 3)
      require(work.started == 0 && work.budget_checks == 1 && work.environment_checks == 1 &&
          std::string_view(work.decision) == "not_started",
          "非法d也须先验证真实FP环境，再在数值运算启动前拒绝，不能越过新资格前缀");
    else require(work.budget_checks == 0 && work.environment_checks == 0,
        "尺寸或脏账本错误须在任何预算及FP资格读取之前拒绝");
  }
  ++budget_cases;
  details.push_back({"natural_schur_budget_inputs", "budget"});
}

void natural_test_control(solver::detail::fma_axpy::ControlState control) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __asm__ __volatile__("ldmxcsr %0\n\tfldcw %1" : : "m"(control.mxcsr), "m"(control.x87_control) : "memory");
#else
  (void)control;
#endif
}

struct NaturalControlRestore {
  solver::detail::fma_axpy::ControlState saved = solver::detail::fma_axpy::read_control_state();
  ~NaturalControlRestore() noexcept { natural_test_control(saved); }
};

void check_natural_schur_environment() {
  const std::vector<double> costs{2}, empty, constant{1};
  natural_schur::Work baseline;
  int total = 0;
  natural_schur::compute(costs, empty, constant, 1, baseline, [&] { ++total; });
  const auto initial = solver::detail::fma_axpy::read_control_state();
  for (int mode = 0; mode < 5; ++mode) for (int cutoff = 0; cutoff <= total; ++cutoff) {
    auto changed = initial;
    if (mode == 0) changed.mxcsr |= 0x8000u;
    if (mode == 1) changed.mxcsr |= 0x0040u;
    if (mode == 2) changed.mxcsr = (changed.mxcsr & ~0x6000u) | 0x2000u;
    if (mode == 3) changed.x87_control = static_cast<std::uint16_t>((changed.x87_control & ~0x0300u) | 0x0200u);
    if (mode == 4) changed.x87_control = static_cast<std::uint16_t>((changed.x87_control & ~0x0c00u) | 0x0400u);
    natural_schur::Work work;
    int checks = 0;
    bool rejected = false;
    {
      NaturalControlRestore restore;
      if (cutoff == 0) natural_test_control(changed);
      try { natural_schur::compute(costs, empty, constant, 1, work, [&] {
        if (++checks == cutoff) natural_test_control(changed);
      }); }
      catch (const natural_schur::Failure&) { rejected = true; }
    }
    require(rejected && work.completed == 0 && std::string_view(work.decision) == "unsupported" &&
        work.environment_checks == work.budget_checks && (cutoff == 0 ? checks == 1 : checks == cutoff),
        "原环境或任一成功预算回调改变FTZ/DAZ/舍入/精度后，必须重读状态并拒绝，不能沿用旧资格");
    if (cutoff <= 1) require(work.started == 0 && work.terms == 0,
        "最早环境失败不能启动任何区间运算或发布候选");
  }
  for (bool allocation : {false, true}) {
    natural_schur::Work work;
    bool original = false;
    {
      NaturalControlRestore restore;
      try { natural_schur::compute(costs, empty, constant, 1, work, [&] {
        auto changed = initial; changed.mxcsr |= 0x8000u; natural_test_control(changed);
        if (allocation) throw std::bad_alloc{};
        throw solver::BudgetExpired{};
      }); }
      catch (const std::bad_alloc&) { original = allocation; }
      catch (const solver::BudgetExpired&) { original = !allocation; }
    }
    require(original && work.started == 0 && work.environment_checks == 0 && work.budget_checks == 1,
        "预算回调本身抛异常时应原样传播，不能以随后环境不支持掩盖原始异常");
  }
  const auto restored = solver::detail::fma_axpy::read_control_state();
  require((restored.mxcsr & 0xffc0u) == (initial.mxcsr & 0xffc0u) &&
      (restored.x87_control & 0x0f3fu) == (initial.x87_control & 0x0f3fu),
      "环境测试必须恢复调用方控制态，sticky算术状态不作为资格或错误恢复的替代");
  ++exception_cases;
  details.push_back({"natural_schur_environment", "exception"});
}

struct NaturalNormalSystem {
  solver::ConeModel cone;
  solver::Point point;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  solver::detail::IpmHomogeneousScaling scaling;
  std::unique_ptr<solver::KktFactor> factor;
  NaturalNormalSystem() {
    cone.costs = {66, -66}; cone.rhs = {0, 2}; cone.starts = {0, 2, 4};
    cone.rows = {0, 1, 0, 1}; cone.values = {1, 1, 1, -1}; cone.inequality = {true, true}; cone.barrier_rows = 2;
    point.x = {0, 0}; point.s = {2, 2.0 / 67}; point.z = {1, 1};
    scaling = solver::detail::make_ipm_homogeneous_scaling(2, cone.inequality, point.s, point.z, 1, 1);
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    factor->factor({2.0 / 67, 2}, options, start, result);
  }
};

std::pair<std::vector<double>, long double> natural_test_constant(NaturalNormalSystem& system,
    const std::vector<double>& rows, const current_diagonal::ValidatedView& view,
    std::span<const double> scales, solver::detail::factor_reuse::Probe& probe,
    const std::function<void()>& budget = {}) {
  const auto rhs = solver::detail::ipm_homogeneous_scaled_constant_rhs(system.scaling,
      system.cone.costs, system.cone.rhs, budget);
  auto* log = coupled_state::current();
  require(log != nullptr, "自然分母常量必须保留真实联立费用作用域");
  if (budget) budget();
  ++log->constant_calls_started;
  auto constant = solver::v124_apply_current(view, rhs, *system.factor, system.options, system.start,
      system.result, true, false, probe.current_diagonal, budget, scales, &probe.transport);
  ++log->constant_calls_completed;
  const auto positive = solver::v124_energy_denominator(*system.factor, 2, constant, rows, system.result, budget, 2);
  solver::v125_scalar_diagnostics(system.scaling, system.cone.costs, system.cone.rhs, constant,
      positive, probe.scalar_consistency, budget);
  const auto denominator = natural_schur::compute(system.cone.costs, system.cone.rhs, constant,
      system.scaling.d, probe.natural_schur, budget);
  return {std::move(constant), denominator};
}

void check_natural_schur_real_normal() {
  NaturalNormalSystem system;
  coupled_state::Log main; main.enabled = true;
  energy_state::Log energy; energy.enabled = true;
  coupled_state::Scope scope(&main);
  energy_state::Scope energy_scope(&energy);
  solver::detail::factor_reuse::Probe probe;
  const auto rows = solver::v124_current_rows(*system.factor, system.point.s, system.result, probe);
  const current_diagonal::ValidatedView view(2, 2, system.cone.starts, system.cone.rows, system.cone.values,
      rows, probe.current_diagonal);
  schur_transport::Cache cache;
  const auto scales = schur_transport::prepare(view, system.factor->v93_regularized_h(),
      *system.result.ipm_primal_regularization, 1, system.factor.get(), cache, probe.transport);
  const auto constant = natural_test_constant(system, rows, view, scales, probe);
  natural_schur_reliable(probe.natural_schur, constant.second, 4);
  require(constant.second < -100 && probe.scalar_consistency.positive_denominator &&
      *probe.scalar_consistency.positive_denominator >= 1 && system.result.ipm_factor_backend == "normal_cholmod_aat",
      "真实Normal相关翻转须产生可靠负N并保留独立正能量观察，不拿解析极限替代实际旧P");
  volatile long double dot = 0;
  for (std::size_t i = 0; i < constant.first.size(); ++i) {
    volatile long double quotient = static_cast<long double>(i < 2 ? system.cone.costs[i] : system.cone.rhs[i - 2]) / system.scaling.d;
    volatile long double product = quotient * constant.first[i];
    dot = dot + product;
  }
  require(constant.second == 1 - dot, "N必须来自同一实际T/R/旧P常量，独立逐步LD表达式逐值对照");
  const std::vector<double> rhs{1, 0, 0, 0, 0, 0, 0, 0};
  const auto reduced = solver::v124_apply_current(view,
      solver::detail::ipm_homogeneous_scaled_reduced_rhs(system.scaling, rhs), *system.factor,
      system.options, system.start, system.result, false, false, probe.current_diagonal, {}, scales, &probe.transport);
  const auto answer = solver::detail::ipm_homogeneous_scaled_recover(system.scaling, system.cone.costs,
      system.cone.rhs, rhs, reduced, constant.first, constant.second, {}, solver::detail::IpmScalarDenominatorPolicy::signed_schur);
  const long double gap = (66 * static_cast<long double>(answer[0]) - 66 * answer[1] + 2 * answer[3] + answer[7]) / system.scaling.d;
  const long double tau = static_cast<long double>(answer[6]) + answer[7];
  require(std::abs(gap) <= 1e-10L && std::abs(tau) <= 1e-12L,
      "真实负N恢复须闭合标量关系，恢复到double后的误差仍独立计入物理检查");
  const long double dual0 = static_cast<long double>(answer[2]) + answer[3] + 66 * answer[6];
  const long double dual1 = static_cast<long double>(answer[2]) - answer[3] - 66 * answer[6];
  require(std::max(std::abs(dual0 - 1), std::abs(dual1)) > 0.1L,
      "可靠负N及scalar闭合不代表完整方向合格，真实dual仍必须通过原五块forcing门");
  bool rejected = false;
  try { solver::homogeneous_scaled_direction(system.cone, system.point, 1, 1, {0, 0}, {-1, 0}, 0, 1,
      {0, 0}, 0, constant.first, system.scaling, constant.second, *system.factor, system.options, system.start,
      system.result, "affine", 2, true, 1, &view, &probe.current_diagonal, scales, &probe.transport); }
  catch (const coupled_kernel::InitialRejected&) { rejected = true; }
  require(rejected && main.last && std::string_view(main.last->work.status) == "initial_rejected" &&
      main.last->work.krylov_started == 0 && probe.natural_schur.completed == 1 &&
      main.constant_calls_completed == 1 && energy.completed == 1 && *system.result.ipm_factorizations == 1,
      "生产signed adapter不得把自然分母证书当接受方向证明，原初P物理门与旧费用合同保持");
  ++mapping_cases;
  details.push_back({"natural_schur_real_normal", "mapping"});
}

struct NaturalPairFixture : TransportPairFixture {
  explicit NaturalPairFixture(bool reject) : TransportPairFixture(reject) {}

  solver::HomogeneousDirection pair_natural(bool reuse_probe, const std::function<void()>& budget = {}) {
    if (!reuse_probe) return TransportPairFixture::pair(false, budget);
    auto& probe = reuse.events.back();
    probe.current_diagonal.columns = 1; probe.current_diagonal.rows = 1; probe.current_diagonal.nonzeros = 1;
    probe.transport.columns = 1; probe.transport.rows = 1; probe.transport.nonzeros = 1;
    const auto rows = solver::v124_current_rows(*system.factor, std::vector<double>{2}, system.result, probe, budget);
    const current_diagonal::ValidatedView view(1, 1, system.cone.starts, system.cone.rows, system.cone.values,
        rows, probe.current_diagonal, budget);
    const auto scales = schur_transport::prepare(view, system.factor->v93_regularized_h(),
        *system.result.ipm_primal_regularization, 1, system.factor.get(), cache, probe.transport, budget);
    auto constant = transport_test_constant(system, rows, view, probe, scales, budget);
    constant.second = natural_schur::compute(system.cone.costs, system.cone.rhs, constant.first,
        system.scaling.d, probe.natural_schur, budget);
    require(constant.second > 0, "整pair事务夹具使用良态正N，负N真实物理合同由独立Normal组覆盖");
    ++affine_calls;
    const auto affine = direction("affine", {2, 1, 1, 1, 1}, constant, 1, &view, &probe, scales, budget);
    require(affine.v94_wide.has_value(), "自然分母整pair首方向必须真实交付");
    h[0] = 7;
    ++corrector_calls;
    return direction("corrector", reject_corrector ? std::array<double, 5>{2, 100, 1, 1, 1} :
        std::array<double, 5>{2, 2.1, 1, 1, 1}, constant, 1, &view, &probe, scales, budget);
  }
};

void check_natural_schur_pair_transaction() {
  namespace reuse_state = solver::detail::factor_reuse;
  for (bool reject : {false, true}) {
    NaturalPairFixture fixture(reject);
    const auto saved_h = fixture.h;
    const auto before = reuse_state::counts(fixture.system.result);
    const auto answer = solver::v122_try_reuse_pair(fixture.h,
        [&](bool probe) { return fixture.pair_natural(probe); }, fixture.reuse, 2, 1, fixture.system.result);
    const auto& event = fixture.reuse.events.back();
    const auto paid = reuse_state::difference(reuse_state::counts(fixture.system.result), before);
    require(answer.has_value() != reject && fixture.affine_calls == 1 && fixture.corrector_calls == 1 &&
        event.natural_schur.started == 1 && event.natural_schur.completed == 1 && event.natural_schur.terms == 2 &&
        event.scalar_consistency.completed == 1 && event.energy.completed == 1 && event.transport.prepare_completed == 1 &&
        event.current_diagonal.applications_completed == 5 && event.transport.input_completed == 5 &&
        event.transport.output_completed == 5 && event.coupled.krylov_completed == 2 &&
        paid.constant_solves == 1 && paid.direction_solves == 4 && paid.refinement_steps == 2 &&
        paid.scaled_refinement_steps == 2 && paid.scaled_products == 6 && paid.factorizations == 0,
        "整pair仅构造一次自然证书，两方向共用同一实际常量/N/T/R/P，旧正能量观察和所有新增费用继续保留");
    if (!reject) {
      require(fixture.h == std::vector<double>({7}) && fixture.main.directions_completed == 2 &&
          fixture.reuse.reused_steps == 1 && std::string_view(event.status) == "accepted",
          "自然证书通过后仍须完整pair通过，才合并主账本和保留h");
    } else {
      require(fixture.h == saved_h && fixture.main.directions_started == 0 && fixture.energy.started == 0 &&
          fixture.reuse.reused_steps == 0 && std::string_view(event.status) == "bounded_rejected",
          "corrector的原五块cap1拒绝须恢复h并丢弃整组，不能凭自然证书部分采用");
      fixture.cache.valid = false;
      fixture.system.factor->factor({2}, fixture.system.options, fixture.system.start, fixture.system.result);
      require(fixture.h == saved_h, "自然路径拒绝后fresh96重做之前仍须保持原h");
      const auto fresh = fixture.pair_natural(false);
      require(fresh.v94_wide && fixture.affine_calls == 2 && fixture.corrector_calls == 2 &&
          fixture.main.directions_completed == 2 && *fixture.system.result.ipm_factorizations == 2 &&
          event.natural_schur.completed == 1, "fresh新因子仍按原正能量96步重做整pair，不篡改失败证书费用");
    }
  }
  {
    NaturalPairFixture fixture(false);
    const auto saved_h = fixture.h;
    const std::vector<double> one{1}, empty;
    const auto answer = solver::v122_try_reuse_pair(fixture.h, [&](bool) -> solver::HomogeneousDirection {
      // 组件隔离typed证书失败的事务出口，不宣称此处另有物理常量求解。
      fixture.h[0] = 7;
      natural_schur::compute(one, empty, one, 1, fixture.reuse.events.back().natural_schur);
      throw std::runtime_error("不可靠自然分母不应返回");
    }, fixture.reuse, 2, 1, fixture.system.result);
    const auto& event = fixture.reuse.events.back();
    require(!answer && fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
        std::string_view(event.status) == "numerical_error" && std::string_view(event.natural_schur.decision) == "uncertain" &&
        event.coupled.directions_started == 0 && event.public_delta.direction_solves == 0 && event.public_delta.constant_solves == 0,
        "typed自然证书失败须进入整pair已知数值拒绝出口并恢复h，不能启动方向或伪报常量费用");
  }
  for (bool reject : {false, true}) {
    int total = 0;
    {
      NaturalPairFixture fixture(reject);
      solver::v122_try_reuse_pair(fixture.h, [&](bool probe) { return fixture.pair_natural(probe, [&] { ++total; }); },
          fixture.reuse, 2, 1, fixture.system.result, [&] { ++total; });
    }
    for (int cutoff = 1; cutoff <= total; ++cutoff) {
      NaturalPairFixture fixture(reject);
      const auto saved_h = fixture.h;
      int checks = 0;
      const auto budget = [&] { if (++checks == cutoff) throw solver::BudgetExpired{}; };
      bool original = false;
      try { solver::v122_try_reuse_pair(fixture.h, [&](bool probe) { return fixture.pair_natural(probe, budget); },
          fixture.reuse, 2, 1, fixture.system.result, budget); }
      catch (const solver::BudgetExpired&) { original = true; }
      require(original && checks == cutoff && fixture.h == saved_h && fixture.reuse.reused_steps == 0 &&
          fixture.main.constant_calls_started == 0 && fixture.main.directions_started == 0 && fixture.energy.started == 0,
          "整pair每个新自然计算及方向预算中断都须保原异常和h，不合并半证书或部分方向");
      if (!fixture.reuse.events.empty()) {
        const auto& event = fixture.reuse.events.back();
        require(std::string_view(event.status) == "budget" && event.natural_schur.completed <= 1 &&
            event.natural_schur.terms <= 2 &&
            event.public_delta.constant_solves + event.public_delta.direction_solves == event.transport.direct_started,
            "自然计算失败前后所有真实旧P费用都留在事件中，末证书已形成也不等于正常交付");
      }
    }
  }
  ++transaction_cases;
  details.push_back({"natural_schur_pair_transaction", "transaction"});
}

bool normal_input_same_bits(const std::vector<double>& first, const std::vector<double>& second) {
  if (first.size() != second.size()) return false;
  for (std::size_t i = 0; i < first.size(); ++i)
    if (std::bit_cast<std::uint64_t>(first[i]) != std::bit_cast<std::uint64_t>(second[i])) return false;
  return true;
}

struct NormalInputKktSystem {
  solver::ConeModel cone;
  solver::SolveOptions options = policy();
  solver::Clock::time_point start = solver::Clock::now();
  solver::SolveResult result = candidate::make_shell(solver::Model{});
  std::unique_ptr<solver::KktFactor> factor;
  NormalInputKktSystem(const std::string& backend, bool equilibrate) {
    options.ipm.linear_solver = backend; options.ipm.equilibrate_kkt = equilibrate;
    cone.costs = {-1, 2}; cone.rhs = {3, -2, 5};
    cone.starts = {0, 3, 5}; cone.rows = {0, 1, 2, 0, 1}; cone.values = {1, 3, 5, 2, -1};
    cone.inequality = {true, false, true}; cone.barrier_rows = 2;
    factor = std::make_unique<solver::KktFactor>(cone, options, start, result);
    factor->factor({2, 0, 3}, options, start, result, 7);
  }
};

void check_normal_kkt_input_epochs() {
  std::vector<std::string> backends{"normal", "normal_multifrontal", "normal_cholmod", "normal_cholmod_aat"};
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    backends.push_back("normal_extended"); backends.push_back("normal_refined");
  }
  for (const auto& backend : backends) for (bool equilibrate : {false, true}) {
    coupled_state::Log scope_log; scope_log.enabled = true;
    coupled_state::Scope scope(&scope_log);
    NormalInputKktSystem system(backend, equilibrate);
    const auto original_values = system.cone.values;
    const auto original_rows = std::vector<double>(system.factor->v93_regularized_h().begin(),
        system.factor->v93_regularized_h().end());
    const auto original_r = system.result.ipm_primal_regularization;
    const auto original_delta = system.result.ipm_dual_regularization;
    const std::vector<double> rhs{0.5, -1, 1.5, -1, 2.5};
    const auto initial = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false);
    const auto first_count = *system.result.ipm_factorizations;
    for (const auto& diagonal : std::vector<std::vector<double>>{{7, 0, 11}, {2, 0, 3}, {2, 0, 3}}) {
      system.factor->factor(diagonal, system.options, system.start, system.result, 7);
      const auto answer = system.factor->solve_homogeneous_direct(rhs, system.options, system.start, system.result, false);
      require(answer.size() == rhs.size() && std::all_of(answer.begin(), answer.end(), [](double value) { return std::isfinite(value); }),
          "各Normal组合的实际KKT重分解与完整单例恢复必须形成有限方向");
      const auto rows = system.factor->v93_regularized_h();
      energy_near(static_cast<long double>(rows[2]) * answer[4], 5 * static_cast<long double>(answer[0]) - rhs[4],
          "重分解后单例仍须按本轮实际R恢复，不能使用上一轮的旧数值");
      if (diagonal[0] == 2)
        require(normal_input_same_bits(answer, initial),
            "实际producer A到B再回A及连续A分解的旧P结果须逐位重现，不能累乘均衡或漏写scratch");
    }
    require(*system.result.ipm_factorizations == first_count + 3 && *system.result.ipm_homogeneous_direction_solves == 4 &&
        system.result.ipm_primal_regularization == original_r && system.result.ipm_dual_regularization == original_delta &&
        std::vector<double>(system.factor->v93_regularized_h().begin(), system.factor->v93_regularized_h().end()) == original_rows &&
        system.cone.values == original_values && *system.result.ipm_eliminated_singletons == 1 &&
        *system.result.ipm_structural_fallbacks == 0 && system.factor->regularization_level() == 7,
        "均衡与非均衡生产调用均须完整覆盖每轮输入，保正则/结构/公共费用及未缩放G");
  }
  ++mapping_cases;
  details.push_back({"normal_kkt_input_epochs", "mapping"});
}

void check_normal_kkt_regularization_retry() {
  solver::ConeModel cone;
  cone.costs = {0, 0}; cone.rhs = {1, 1}; cone.starts = {0, 2, 4};
  cone.rows = {0, 1, 0, 1}; cone.values = {1, 1, 1, 1};
  cone.inequality = {false, false}; cone.barrier_rows = 0;
  auto options = policy(); options.ipm.linear_solver = "normal_cholmod";
  const auto start = solver::Clock::now();
  auto result = candidate::make_shell(solver::Model{});
  coupled_state::Log log; log.enabled = true;
  coupled_state::Scope scope(&log);
  solver::KktFactor factor(cone, options, start, result);
  // level0的Gram为2^40*ones，sqrt及消元均精确，末pivot必为0；后续成功层由旧copy实测。
  // 未均衡K的非对角全为1，此正则网格内Ruiz各坐标恰为1，两行等式均不凝聚。
  solver::detail::NormalCholesky copy(4, std::vector<std::int64_t>{0, 1, 2, 5, 8},
      std::vector<std::int64_t>{0, 1, 0, 1, 2, 0, 1, 3}, std::vector<std::int8_t>{1, 1, -1, -1}, {}, false, false, 0,
      solver::detail::NormalCholeskyKernel::cholmod_supernodal);
  int success_level = -1, attempts = 0;
  for (int level = 0; level < 8; ++level) {
    const auto regularization = solver::detail::ipm_regularization_pair(level, false);
    const std::vector<double> values{regularization[0], regularization[0], 1, 1, -regularization[1], 1, 1, -regularization[1]};
    ++attempts;
    const bool success = copy.factor(values);
    if (success) { success_level = level; break; }
  }
  require(success_level > 0 && attempts == success_level + 1,
      "正则重试夹具必须真实包含旧copy数值失败，再由同一原网格成功，不能把单次分解冒称重试");
  factor.factor({0, 0}, options, start, result, 0);
  const auto regularization = solver::detail::ipm_regularization_pair(success_level, false);
  require(factor.regularization_level() == success_level && *result.ipm_factorizations == attempts &&
      result.ipm_primal_regularization == regularization[0] && result.ipm_dual_regularization == regularization[1] &&
      factor.nominal_primal_regularization() == regularization[0] && factor.nominal_dual_regularization() == regularization[1],
      "producer每次重试必须从未均衡ax重新形成正则数值，成功层及次数与独立旧copy完全一致");
  const std::vector<double> rhs{1, 1, 0, 0};
  const auto first = factor.solve_homogeneous_direct(rhs, options, start, result, false);
  const auto first_count = *result.ipm_factorizations;
  factor.factor({0, 0}, options, start, result, 0);
  const auto second = factor.solve_homogeneous_direct(rhs, options, start, result, false);
  require(normal_input_same_bits(first, second) && *result.ipm_factorizations == first_count + attempts &&
      factor.regularization_level() == success_level && factor.v93_regularized_h().size() == 2 &&
      factor.v93_regularized_h()[0] == regularization[1] && factor.v93_regularized_h()[1] == regularization[1] &&
      cone.values == std::vector<double>({1, 1, 1, 1}) && *result.ipm_regularized_pivots == 0 && *result.ipm_structural_fallbacks == 0,
      "成功之后再次从level0全覆盖重试须逐位重现同一实际P，失败输入与旧scratch不能污染下一轮");
  ++transaction_cases;
  details.push_back({"normal_kkt_regularization_retry", "transaction"});
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
    check_candidate_forcing_real_hsd(); check_candidate_forcing_default_mapping(); check_candidate_forcing_delivery_budget();
    check_forcing_weights_real_hsd(); check_forcing_weights_default_mapping(); check_forcing_weights_budget();
    check_warm_regularization_seed_scope(); check_warm_regularization_real_factor(); check_warm_regularization_hsd_budget();
    check_fma_production_binding(); check_fma_real_hsd_mapping(); check_fma_hsd_budget_prefix();
    check_normal_inverse_multi_rhs(); check_normal_inverse_refactor(); check_normal_inverse_factor_failure();
    check_normal_inverse_budget_prefix(); check_normal_inverse_scope_and_fallback(); check_normal_inverse_extreme_recovery();
    check_normal_budget_short_columns(); check_normal_budget_long_boundaries(); check_normal_budget_shared_passes();
    check_normal_budget_all_cutoffs(); check_normal_budget_scope_snapshot(); check_normal_budget_fallback_rejection();
    check_product_budget_double_wide(); check_product_budget_work_boundaries(); check_product_budget_shared_columns();
    check_product_budget_all_cutoffs(); check_product_budget_scope_snapshot(); check_product_budget_kernel_boundaries();
    check_crossover_hsd_actual_handoff(); check_crossover_gate_gap_and_domain(); check_crossover_remaining_deadline();
    check_crossover_single_attempt_scope(); check_crossover_original_postsolve(); check_crossover_gate_current_snapshot();
    // v118 行对偶域投影组件注册开始
    check_crossover_dual_projection_unchanged(); check_crossover_dual_projection_accept();
    check_crossover_dual_projection_reject(); check_crossover_dual_projection_budget();
    // v118 行对偶域投影组件注册结束
    // v120 自主暖基衔接组件注册开始
    check_crossover_basis_export(); check_crossover_warm_accept(); check_crossover_warm_reject();
    check_crossover_warm_budget(); check_crossover_warm_hsd_bridge();
    // v120 自主暖基衔接组件注册结束
    // v121 缓冲诊断组件注册开始
    check_diagnostic_buffered_serializers(); check_diagnostic_buffered_file_modes();
    check_diagnostic_buffered_large_concurrent(); check_diagnostic_buffered_disabled_errors();
    // v121 缓冲诊断组件注册结束
    check_public_typed_presolve_and_crossover(); check_public_diagnostics_profile_control();
    check_public_option_context_restoration();
    check_warm_dual_lower_upper(); check_warm_dual_bounded_overshoot(); check_warm_dual_free_superbasic();
    check_warm_dual_qualification_fixed(); check_warm_dual_ratio_limits(); check_warm_dual_budget_prefix();
    check_warm_harris_pivot_stability(); check_warm_harris_tolerance_band(); check_warm_harris_tiny_directions();
    check_warm_harris_limits_no_candidate(); check_warm_harris_budget_prefix();
    check_normal_mapped_kkt_physical(); check_normal_mapped_kkt_failure_budget();
    check_factor_reuse_current_physics();
    check_factor_reuse_pair_transaction(); check_factor_reuse_budget_scope_serializer();
    check_factor_reuse_bounded_physics(); check_factor_reuse_bounded_pair_transaction();
    check_current_diagonal_hand_schur(); check_current_diagonal_real_normal();
    check_current_diagonal_constant_physics(); check_current_diagonal_indefinite_energy();
    check_current_diagonal_pair_transaction(); check_current_diagonal_budget_and_errors();
    check_schur_transport_exact_identity(); check_schur_transport_duplicates_correlation();
    check_schur_transport_prepare_cache(); check_schur_transport_apply_budget();
    check_schur_transport_real_normal(); check_schur_transport_constant_physics();
    check_schur_transport_pair_transaction();
    check_natural_schur_signed_recover(); check_natural_schur_enclosure_rounding();
    check_natural_schur_budget_inputs(); check_natural_schur_environment();
    check_natural_schur_real_normal(); check_natural_schur_pair_transaction();
    check_normal_kkt_input_epochs();
    check_normal_kkt_regularization_retry();
    require(details.size() == static_cast<std::size_t>(wrapper_cases + mapping_cases + transaction_cases +
        budget_cases + exception_cases) && component_ipm_calls > 0 && synthetic_core_calls > 0,
        "实际小例与合成边界计数必须守恒且不能冒称零全部求解");
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"explicit_highs_lp_solve_calls\":0,"
        "\"scope\":\"仅组件自测；component_ipm_calls计harness进入真实solve_ipm或solve_ipm_core的API次数(含零预算入口)，不含内部调用拦截；synthetic_core_calls另计，不是公开LP成绩\","
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
