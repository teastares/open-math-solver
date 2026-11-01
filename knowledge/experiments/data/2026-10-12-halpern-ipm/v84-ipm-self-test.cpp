#include "ipm-warm.cpp"

#include <cstdio>
#include <iostream>
#include <string>

namespace {
int cases = 0, mapping_cases = 0, isolation_cases = 0;
int verification_accounting_cases = 0, budget_cases = 0, exception_cases = 0;
const double infinity = std::numeric_limits<double>::infinity();

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

solver::SolveOptions policy() {
  solver::SolveOptions options;
  options.time_limit = 60;
  options.ipm.objective_scaling = false;
  options.ipm.eliminate_fixed_variables = true;
  return options;
}

solver::Model mapped_model(bool maximum = false) {
  solver::Model model;
  model.sense = maximum ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
  model.objective_offset = 11;
  model.costs = {7, maximum ? -2.0 : 2.0, 0};
  model.column_lower = {3, 0, -infinity}; model.column_upper = {3, 10, infinity};
  model.row_lower = {16, -infinity, -25}; model.row_upper = {16, 40, infinity};
  model.kinds.assign(3, solver::VariableKind::continuous);
  model.column_start = {0, 0, 1, 3}; model.row_index = {0, 1, 2}; model.values = {8, 10, 5};
  return model;
}

solver::Model merged_model(bool upper, bool fixed = false, bool maximum = false) {
  solver::Model model;
  model.sense = maximum ? solver::ObjectiveSense::maximize : solver::ObjectiveSense::minimize;
  model.costs = {upper ? -3.0 : 3.0};
  model.column_lower = {-infinity}; model.column_upper = {infinity};
  model.row_lower = {8, -infinity}; model.row_upper = {infinity, fixed ? 4.0 : 10.0};
  model.kinds.assign(1, solver::VariableKind::continuous);
  model.column_start = {0, 2}; model.row_index = {0, 1}; model.values = {4, 2};
  return model;
}

solver::SolveResult original(const solver::Model& model) {
  solver::SolveResult result;
  result.model = model;
  result.backend = "ipm"; result.backend_version = "v66-unchanged";
  result.status = "numerical_error"; result.backend_status = "original_status";
  result.message = "原错误不能被恢复失败替换"; result.verification_level = "original_level";
  result.has_solution = true; result.solution = {101}; result.row_dual = {202};
  result.objective = 303; result.best_bound = 304; result.absolute_gap = 305; result.relative_gap = 306;
  result.verification.objective = 307; result.verification.errors = {"原诊断"};
  result.lp_optimality.emplace(); result.lp_optimality->objective_gap = 308;
  result.verification_seconds = 7;
  result.ipm_iterations = 137; result.ipm_factorizations = 11; result.ipm_refinement_steps = 19;
  result.ipm_homogeneous_constant_solves = 139; result.ipm_homogeneous_direction_solves = 275;
  return result;
}

void same_payload(const solver::SolveResult& value, const solver::SolveResult& before) {
  require(value.backend == before.backend && value.backend_version == before.backend_version &&
      value.backend_status == before.backend_status && value.status == before.status && value.message == before.message &&
      value.verification_level == before.verification_level && value.has_solution == before.has_solution &&
      value.solution == before.solution && value.row_dual == before.row_dual && value.objective == before.objective &&
      value.best_bound == before.best_bound && value.absolute_gap == before.absolute_gap && value.relative_gap == before.relative_gap &&
      value.verification.objective == before.verification.objective && value.verification.errors == before.verification.errors &&
      value.verification.valid_input == before.verification.valid_input && value.verification.feasible == before.verification.feasible &&
      value.lp_optimality.has_value() == before.lp_optimality.has_value() &&
      (!value.lp_optimality || value.lp_optimality->objective_gap == before.lp_optimality->objective_gap) &&
      value.model.values == before.model.values && value.model.costs == before.model.costs &&
      value.model.column_lower == before.model.column_lower && value.model.row_lower == before.model.row_lower &&
      value.ipm_iterations == before.ipm_iterations && value.ipm_factorizations == before.ipm_factorizations &&
      value.ipm_refinement_steps == before.ipm_refinement_steps &&
      value.ipm_homogeneous_constant_solves == before.ipm_homogeneous_constant_solves &&
      value.ipm_homogeneous_direction_solves == before.ipm_homogeneous_direction_solves,
      "未提交恢复不得污染原解、质量、状态、模型、后端或legacy计数");
}

solver::Point seed(const solver::ConeModel& cone, const std::vector<double>& x, const std::vector<double>& y,
    double tau = 1) {
  solver::Point point;
  point.x.resize(cone.costs.size()); point.z.assign(cone.rhs.size(), 0);
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const auto original_col = cone.original_columns.empty() ? col : cone.original_columns[col];
    point.x[col] = (x[original_col] / cone.column_scale[col]) * tau;
  }
  for (std::size_t row = 0; row < point.z.size(); ++row) {
    if (cone.original_row[row] < 0) continue;
    const auto original_row = static_cast<std::size_t>(cone.original_row[row]);
    const double z = y[original_row] / (cone.dual_sign[row] * cone.row_scale[row] * cone.objective_scale);
    if (!cone.inequality[row] || z >= 0) point.z[row] = z * tau;
  }
  for (const auto& recovery : cone.presolved_bounds) {
    double multiplier = 0;
    if (recovery.lower.row >= 0) multiplier = y[static_cast<std::size_t>(recovery.lower.row)] * recovery.lower.coefficient;
    if (multiplier == 0 && recovery.upper.row >= 0)
      multiplier = y[static_cast<std::size_t>(recovery.upper.row)] * recovery.upper.coefficient;
    const auto row = static_cast<std::size_t>(recovery.cone_row);
    const double z = multiplier / (recovery.sign * cone.row_scale[row] * cone.objective_scale);
    if (!cone.inequality[row] || z >= 0) point.z[row] = z * tau;
  }
  return point;
}

void mapping(const solver::Model& model, const std::vector<double>& known_x, const std::vector<double>& known_y,
    solver::SolveOptions options, bool use_kernel = true) {
  const auto start = solver::Clock::now();
  const auto cone = solver::canonicalize(model, options, start);
  auto result = original(model);
  const auto before = result;
  const auto model_storage = result.model.values.data();
  solver_v84_warm_state::Log log;
  log.enabled = true;
  bool accepted;
  if (use_kernel) {
    const double tau = std::ldexp(1.0, -40);
    const auto point = seed(cone, known_x, known_y, tau);
    accepted = solver::v84_try_warm(cone, point, tau, 0.125, options, start, result, log);
    require(log.attempted && log.accepted && log.kernel.accepted && log.kernel.stats.steps_started == 0,
        "原模型已验收种子应直接通过，不伪造PDHG迭代");
  } else {
    // 输入归并后全固定的模型会在原HSD主循环前早退；这里只测试同一实际assess适配器的映射。
    const auto point = seed(cone, known_x, known_y);
    solver::V84WarmAssessment assessment(cone, options, start, result, log);
    accepted = assessment(0, point.x, point.z);
    require(accepted && assessment.pending, "全固定映射必须经原assess通过");
    solver::check_budget(options, start);
    assessment.commit();
  }
  require(accepted && result.status == "optimal" && result.has_solution && result.verification.feasible &&
      result.lp_optimality && result.lp_optimality->numerically_optimal && log.seed_quality && log.last_quality &&
      log.seed_quality->completed_steps == 0 && log.last_quality->numerically_optimal == std::optional<bool>(true),
      "原坐标映射和原完整终验必须决定接纳");
  require(result.solution.size() == known_x.size() && result.row_dual.size() == known_y.size(), "恢复必须保留原维数");
  for (std::size_t col = 0; col < known_x.size(); ++col)
    require(std::abs(result.solution[col] - known_x[col]) <= 1e-11 * std::max(1.0, std::abs(known_x[col])),
        "固定列和列scale必须恢复原primal");
  for (std::size_t row = 0; row < known_y.size(); ++row)
    require(std::abs(result.row_dual[row] - known_y[row]) <= 1e-9 * std::max(1.0, std::abs(known_y[row])),
        "行scale、上下界来源和目标方向必须恢复原dual");
  require(result.backend == before.backend && result.backend_version == before.backend_version &&
      result.backend_status == before.backend_status && result.model.values.data() == model_storage &&
      result.ipm_iterations == before.ipm_iterations && result.ipm_factorizations == before.ipm_factorizations &&
      result.ipm_refinement_steps == before.ipm_refinement_steps &&
      result.ipm_homogeneous_constant_solves == before.ipm_homogeneous_constant_solves &&
      result.ipm_homogeneous_direction_solves == before.ipm_homogeneous_direction_solves &&
      result.verification_seconds >= before.verification_seconds, "接纳只搬质量，不覆盖原计数或模型");
  ++mapping_cases; ++cases;
}

void check_current_quality() {
  auto options = policy(); const auto model = mapped_model(); const auto start = solver::Clock::now();
  const auto cone = solver::canonicalize(model, options, start);
  const auto good = seed(cone, {3, 2, 0}, {0.25, 0, 0});
  auto bad = good; bad.x[0] += 1;
  auto result = original(model); const auto before = result;
  solver_v84_warm_state::Log log;
  solver::V84WarmAssessment assessment(cone, options, start, result, log);
  require(assessment(0, good.x, good.z), "第一次原验收应通过");
  require(!assessment(64, bad.x, bad.z) && !assessment.pending && log.seed_quality->primal_feasible &&
      !log.last_quality->primal_feasible && log.last_quality->completed_steps == 64,
      "每次force当前点质量，不能把前一个可行点误记到last_quality");
  same_payload(result, before);
  ++isolation_cases; ++cases;
}

void check_off() {
  auto options = policy(); options.time_limit = 0;
  auto result = original(mapped_model()); const auto before = result;
  solver::ConeModel cone; solver::Point point; solver_v84_warm_state::Log log;
  require(!solver::v84_try_warm(cone, point, 0, 0, options, solver::Clock::now(), result, log) &&
      !log.attempted && !log.accepted && !log.seed_quality && !log.last_quality, "关闭不能启动恢复或改变原失败");
  same_payload(result, before);
  require(result.verification_seconds == before.verification_seconds, "关闭不得伪造验收成本");
  ++isolation_cases; ++cases;
}

void check_entry_budget() {
  auto options = policy(); options.time_limit = 0;
  auto result = original(mapped_model()); const auto before = result;
  solver::ConeModel cone; solver::Point point; solver_v84_warm_state::Log log; log.enabled = true;
  bool interrupted = false;
  try { solver::v84_try_warm(cone, point, 1, 1, options, solver::Clock::now(), result, log); }
  catch (const solver::BudgetExpired&) { interrupted = true; }
  require(interrupted && !log.attempted && !log.accepted && !log.seed_quality, "原截止时间已到时不得抢先记作启动恢复");
  same_payload(result, before);
  require(result.verification_seconds == before.verification_seconds, "未启动assessment不能伪造计时");
  ++budget_cases; ++cases;
}

void check_pending_budget() {
  auto options = policy(); const auto model = mapped_model(); const auto start = solver::Clock::now();
  const auto cone = solver::canonicalize(model, options, start);
  const auto good = seed(cone, {3, 2, 0}, {0.25, 0, 0});
  auto result = original(model); const auto before = result;
  solver_v84_warm_state::Log log;
  solver::V84WarmAssessment assessment(cone, options, start, result, log);
  require(assessment(0, good.x, good.z) && assessment.pending, "先取得真实已核pending候选");
  options.time_limit = 0;
  bool interrupted = false;
  try { solver::check_budget(options, start); assessment.commit(); }
  catch (const solver::BudgetExpired&) { interrupted = true; }
  require(interrupted && !log.accepted && log.last_quality->numerically_optimal == std::optional<bool>(true),
      "已核质量不等于提交，原预算仍可拒绝pending候选");
  same_payload(result, before);
  ++budget_cases; ++cases;
}

void check_assessment_budget() {
  auto options = policy(); options.ipm.presolve_singleton_rows = true;
  const auto model = merged_model(false); const auto start = solver::Clock::now();
  const auto cone = solver::canonicalize(model, options, start);
  require(!cone.presolved_bounds.empty(), "预算fixture必须实际进入原界乘子恢复");
  const auto good = seed(cone, {2}, {0.75, 0});
  auto result = original(model); const auto before = result;
  solver_v84_warm_state::Log log;
  solver::V84WarmAssessment assessment(cone, options, start, result, log);
  options.time_limit = 0;
  bool interrupted = false;
  try { assessment(0, good.x, good.z); }
  catch (const solver::BudgetExpired&) { interrupted = true; }
  require(interrupted && !log.seed_quality && !log.last_quality && !assessment.pending,
      "原assess中途预算须保原类型且不得伪造完整质量");
  same_payload(result, before);
  require(result.verification_seconds >= before.verification_seconds, "抛异常的assessment实际时间也须累计");
  ++budget_cases; ++cases;
}

void check_accounting(bool failure) {
  auto options = policy(); const auto model = mapped_model(); const auto start = solver::Clock::now();
  auto cone = solver::canonicalize(model, options, start);
  const auto good = seed(cone, {3, 2, 0}, {0.25, 0, 0});
  auto result = original(model); const auto before = result;
  if (failure) cone.column_scale[0] = infinity;
  solver_v84_warm_state::Log log;
  solver::V84WarmAssessment assessment(cone, options, start, result, log);
  const auto model_storage = assessment.trial.model.values.data();
  assessment.trial.verification_seconds = 1000;
  const auto timing_begin = solver::Clock::now();
  bool threw = false;
  try { assessment(0, good.x, good.z); }
  catch (const solver::NumericalFailure&) { threw = true; }
  const double total = solver::elapsed(timing_begin);
  require(threw == failure && result.verification_seconds >= before.verification_seconds &&
      result.verification_seconds - before.verification_seconds <= total + 1e-9 &&
      assessment.trial.verification_seconds < 1000 && assessment.trial.model.values.data() == model_storage,
      "assessment只记本次真实调用成本，不累计旧trial计时或重复复制model");
  if (failure) require(!log.seed_quality && !log.last_quality, "未完成原assess时质量必须为null");
  else require(result.verification_seconds - before.verification_seconds + 1e-9 >= assessment.trial.verification_seconds,
      "完整原assess内层验收成本不能漏计");
  same_payload(result, before);
  ++verification_accounting_cases; ++cases;
}

void check_original_exception() {
  auto options = policy(); const auto model = mapped_model(); const auto start = solver::Clock::now();
  auto cone = solver::canonicalize(model, options, start);
  const auto good = seed(cone, {3, 2, 0}, {0.25, 0, 0});
  cone.column_scale[0] = infinity;
  auto result = original(model); const auto before = result;
  solver_v84_warm_state::Log log; log.enabled = true;
  bool original_preserved = false;
  try {
    try { throw solver::NumericalFailure("原主循环数值失败"); }
    catch (const solver::NumericalFailure&) {
      if (solver::v84_try_warm(cone, good, 1, 1, options, start, result, log))
        throw std::runtime_error("异常fixture不得接纳");
      throw;
    }
  } catch (const solver::NumericalFailure& error) {
    original_preserved = std::string(error.what()) == "原主循环数值失败";
  }
  require(original_preserved && log.attempted && !log.accepted && std::string(log.status) == "exception" &&
      !log.seed_quality && !log.last_quality, "warm自己的验收异常不能吞掉或替换原异常");
  same_payload(result, before);
  ++exception_cases; ++cases;
}
}  // 匿名命名空间

int main() {
  try {
    for (const bool maximum : {false, true}) for (const bool scale_objective : {false, true}) {
      auto options = policy(); options.ipm.objective_scaling = scale_objective;
      mapping(mapped_model(maximum), {3, 2, 0}, {0.25, 0, 0}, options);
    }
    for (const bool upper : {false, true}) {
      auto options = policy(); options.ipm.presolve_singleton_rows = true;
      mapping(merged_model(upper), {upper ? 5.0 : 2.0}, upper ? std::vector<double>{0, -1.5} : std::vector<double>{0.75, 0}, options);
    }
    for (const bool maximum : {false, true}) {
      auto options = policy(); options.ipm.presolve_singleton_rows = true; options.ipm.eliminate_presolved_fixed_variables = true;
      mapping(merged_model(false, true, maximum), {2}, maximum ? std::vector<double>{0, -1.5} : std::vector<double>{0.75, 0}, options, false);
    }
    check_current_quality(); check_off(); check_entry_budget(); check_pending_budget(); check_assessment_budget();
    check_accounting(false); check_accounting(true); check_original_exception();
    std::printf("{\"passed\":true,\"solver_runs\":0,\"cases\":%d,\"mapping_cases\":%d,\"isolation_cases\":%d,"
        "\"verification_accounting_cases\":%d,\"budget_cases\":%d,\"exception_cases\":%d}\n",
        cases, mapping_cases, isolation_cases, verification_accounting_cases, budget_cases, exception_cases);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
