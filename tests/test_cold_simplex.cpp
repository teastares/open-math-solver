#include "solver/simplex.hpp"
#include "solver/detail/cold_simplex.hpp"
#include "solver/detail/warm_cleanup.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
namespace cold = solver::detail::cold_simplex;
namespace warm = solver::detail::warm_cleanup;
using solver::Model;
using solver::ObjectiveSense;
using solver::VariableKind;
constexpr double inf = std::numeric_limits<double>::infinity();

void require(bool passed, const char* message) {
  if (!passed) throw std::runtime_error(message);
}

solver::SolveOptions policy(std::uint64_t cap = 64) {
  solver::SolveOptions options;
  options.backend = "simplex";
  options.time_limit = 10;
  options.simplex.max_iterations = cap;
  return options;
}

Model scalar(double cost, double lower, double upper, double row_lower, double row_upper) {
  Model model;
  model.costs = {cost};
  model.kinds = {VariableKind::continuous};
  model.column_lower = {lower};
  model.column_upper = {upper};
  model.row_lower = {row_lower};
  model.row_upper = {row_upper};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {1};
  return model;
}

Model independent_ranges(std::size_t count) {
  Model model;
  model.costs.assign(count, -1);
  model.kinds.assign(count, VariableKind::continuous);
  model.column_lower.assign(count, 0);
  model.column_upper.assign(count, inf);
  model.row_lower.assign(count, 1);
  model.row_upper.assign(count, 2);
  model.column_start = {0};
  for (std::size_t row = 0; row < count; ++row) {
    model.row_index.push_back(static_cast<std::int64_t>(row));
    model.values.push_back(1);
    model.column_start.push_back(static_cast<std::int64_t>(row + 1));
  }
  return model;
}

warm::Result run(const Model& model, const solver::SolveOptions& options = policy()) {
  const auto original_costs = model.costs;
  const auto original_sense = model.sense;
  const auto original_offset = model.objective_offset;
  auto result = cold::run(model, options, warm::Clock::now() + std::chrono::seconds(10));
  require(model.costs == original_costs && model.sense == original_sense &&
      model.objective_offset == original_offset, "可行性阶段不能修改原模型目标");
  return result;
}

void expect_accepted(const Model& model, const warm::Result& result, double objective,
                     const solver::SolveOptions& options = policy()) {
  require(result.status == warm::Status::accepted && result.verification &&
      warm::detail::quality_ok(*result.verification), "内部接口必须返回经过完整原模型验收的解");
  const auto verification = solver::verify_lp_solution(model, result.x, result.y,
      options.tolerances, options.simplex.optimality);
  require(warm::detail::quality_ok(verification) && verification.primal.objective &&
      std::abs(*verification.primal.objective - objective) <= 1e-9 * (1 + std::abs(objective)),
      "独立核对原始解、行对偶和解析目标");
}

void test_initial_branches() {
  const auto primal = scalar(-1, 0, inf, -inf, 2);
  const auto already_primal = run(primal);
  expect_accepted(primal, already_primal, -2);
  require(!already_primal.stats.feasibility_started && !already_primal.stats.objective_restored &&
      already_primal.stats.steps_completed == 1 && already_primal.stats.dual_pivots == 0 &&
      already_primal.stats.feasibility_update_live.attempted == 0,
      "原始已可行时直接以原目标推进");

  const auto dual = scalar(1, 0, inf, 1, inf);
  const auto already_dual = run(dual);
  expect_accepted(dual, already_dual, 1);
  require(!already_dual.stats.feasibility_started && !already_dual.stats.objective_restored &&
      already_dual.stats.dual_pivots == 1 && already_dual.stats.feasibility_update_live.attempted == 0,
      "对偶已可行的原目标路径不执行零工作目标增量");

  const auto both = scalar(-1, 0, inf, 1, 2);
  const auto restored = run(both);
  expect_accepted(both, restored, -2);
  require(restored.stats.feasibility_started && restored.stats.objective_restored &&
      restored.stats.feasibility_steps == 1 && restored.stats.steps_completed == 2 &&
      restored.stats.dual_pivots == 1 && restored.stats.feasibility_update_live.accepted == 1,
      "双失败时真实可行化、恢复目标后推进");
}

void test_bound_shapes_and_objective() {
  Model fixed;
  fixed.costs = {-1, 3};
  fixed.kinds.assign(2, VariableKind::continuous);
  fixed.column_lower = {0, 2};
  fixed.column_upper = {inf, 2};
  fixed.row_lower = {3};
  fixed.row_upper = {4};
  fixed.column_start = {0, 1, 2};
  fixed.row_index = {0, 0};
  fixed.values = {1, 1};
  fixed.objective_offset = 5;

  auto upper = scalar(1, -inf, 0, -2, -1);
  upper.objective_offset = -5;
  auto maximum = scalar(1, 0, inf, 1, 2);
  maximum.sense = ObjectiveSense::maximize;
  maximum.objective_offset = 7;

  const std::vector<std::pair<Model, double>> fixtures{
      {std::move(fixed), 9}, {std::move(upper), -7},
      {scalar(-1, -inf, inf, -2, 3), -3}, {scalar(1, -inf, inf, -2, 3), -2},
      {std::move(maximum), 9}};
  for (const auto& [model, objective] : fixtures) {
    const auto result = run(model);
    expect_accepted(model, result, objective);
    require(result.stats.feasibility_started && result.stats.objective_restored &&
        result.stats.feasibility_steps == 1 && result.stats.feasibility_update_live.accepted == 1,
        "固定列、单侧界、自由列和最大化保留原目标");
  }
}

// 独立范围行的显式压缩表示。
// 每个范围行引入一个有界辅助变量和一个基本等式行。
struct RangedFixture {
  Model model;
  warm::cross::Prepared prepared;
  std::vector<warm::Index> variable_status, row_status;
  std::vector<double> structural_seed;
  solver::SolveOptions options = policy();

  explicit RangedFixture(std::size_t count, double coefficient = 1, bool scaled = false)
      : model(independent_ranges(count)), variable_status(2 * count, IPX_nonbasic_lb),
        row_status(count, IPX_basic), structural_seed(2 * count, 0) {
    prepared.original_columns = prepared.original_rows = count;
    prepared.row_type.assign(count, '=');
    prepared.rhs.assign(count, 0);
    prepared.column_start = {0};
    for (std::size_t row = 0; row < count; ++row) {
      const double scale = scaled ? std::ldexp(1.0, static_cast<int>(row % 3) * 20 - 20) : 1;
      model.values[row] = coefficient * scale;
      model.row_lower[row] = scale;
      model.row_upper[row] = 2 * scale;
      prepared.original_row.push_back(row);
      prepared.auxiliary_row.push_back(row);
      prepared.costs.push_back(-1);
      prepared.lower.push_back(0);
      prepared.upper.push_back(inf);
      prepared.row_index.push_back(static_cast<warm::Index>(row));
      prepared.values.push_back(model.values[row]);
      prepared.column_start.push_back(static_cast<warm::Index>(row + 1));
    }
    for (std::size_t row = 0; row < count; ++row) {
      prepared.costs.push_back(0);
      prepared.lower.push_back(model.row_lower[row]);
      prepared.upper.push_back(model.row_upper[row]);
      prepared.row_index.push_back(static_cast<warm::Index>(row));
      prepared.values.push_back(-1);
      prepared.column_start.push_back(static_cast<warm::Index>(count + row + 1));
      structural_seed[count + row] = model.row_lower[row];
    }
  }

  template<class Budget>
  warm::Result solve(bool incremental, std::uint64_t cap, Budget&& budget) const {
    return warm::run({model, prepared, variable_status, row_status, structural_seed},
        warm::Clock::now() + std::chrono::seconds(10), {cap, true, true, incremental},
        options.tolerances, options.simplex.optimality, std::forward<Budget>(budget));
  }

  warm::Result solve(bool incremental, std::uint64_t cap = 64) const {
    return solve(incremental, cap, [] {});
  }
};

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
      (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

void compare_rebuild_and_incremental(const warm::Result& reference, const warm::Result& incremental,
                                     std::uint64_t expected) {
  require(reference.status == incremental.status && same_bits(reference.x, incremental.x) &&
      same_bits(reference.y, incremental.y), "解析例完整重建与增量返回的状态和解应逐位等价");
  const auto& a = reference.stats;
  const auto& b = incremental.stats;
  require(a.steps_completed == b.steps_completed && a.pivots == b.pivots &&
      a.dual_pivots == b.dual_pivots && a.bound_flips == b.bound_flips &&
      a.feasibility_steps == b.feasibility_steps && a.objective_restored == b.objective_restored,
      "两条路线共享原步骤、主元和目标恢复合同");
  const auto& off = a.feasibility_update_live;
  const auto& live = b.feasibility_update_live;
  require(!off.enabled && off.attempted == 0 && off.verification_entries == 0 &&
      live.enabled && live.attempted == expected && live.predicted == expected &&
      live.verification_completed == expected && live.accepted == expected && live.fallback == 0,
      "解析例增量验收和关闭侧工作账本");
  require(live.refresh_completed <= expected && a.rebuild_completed >= b.rebuild_completed &&
      a.rebuild_completed - b.rebuild_completed == expected - live.refresh_completed &&
      a.solve_n.completed - b.solve_n.completed == expected - live.refresh_completed &&
      a.solve_t.completed - b.solve_t.completed == expected - live.refresh_completed,
      "真实节省的重建和两次求解须扣除阶段刷新");
  if (expected != 0)
    require(live.max_equation_ratio <= 1 && live.verification_columns > 0 && live.verification_entries > 0,
        "增量验收应真实重算方程");
}

void test_shared_limits_and_scaling() {
  for (std::size_t blocks : {30U, 32U, 33U}) {
    const RangedFixture fixture(blocks);
    const auto reference = fixture.solve(false);
    const auto live = fixture.solve(true);
    compare_rebuild_and_incremental(reference, live, blocks);
    require(live.stats.feasibility_steps == blocks && live.stats.objective_restored,
        "长路线上实际完成可行性阶段并恢复目标");
    if (blocks <= 32) expect_accepted(fixture.model, live, -2.0 * static_cast<double>(blocks));
    else require(live.status == warm::Status::step_limit && live.stats.steps_completed == 64 &&
        live.x.empty() && live.y.empty(), "恢复目标不重置共享 64 步预算");
  }
  const RangedFixture unscaled(30);
  for (std::uint64_t cap : {0U, 1U, 29U, 30U}) {
    const auto reference = unscaled.solve(false, cap);
    const auto live = unscaled.solve(true, cap);
    compare_rebuild_and_incremental(reference, live, cap);
    require(live.status == warm::Status::step_limit && live.stats.steps_completed == cap &&
        live.x.empty() && live.y.empty(), "共享预算也约束可行性阶段");
  }

  const RangedFixture scaled(30, 1, true);
  const auto reference = scaled.solve(false);
  const auto live = scaled.solve(true);
  compare_rebuild_and_incremental(reference, live, 30);
  expect_accepted(scaled.model, live, -60);
}

void test_real_rounding_fallback() {
  RangedFixture fixture(1, 3);
  fixture.options.tolerances.feasibility_absolute = 0;
  fixture.options.tolerances.feasibility_relative = 0;
  const auto reference = fixture.solve(false);
  const auto live = fixture.solve(true);
  // 扩展精度验算中，3 * double(1/3) 的残差非零。
  require(reference.status != warm::Status::accepted && live.status == reference.status &&
      live.x.empty() && live.y.empty(), "舍入拒绝后不得交付失效解");
  const auto& stats = live.stats.feasibility_update_live;
  require(stats.attempted == 1 && stats.predicted == 1 && stats.verification_completed == 1 &&
      stats.accepted == 0 && stats.fallback == 1 && stats.fallback_rebuild_started == 1 &&
      stats.max_equation_ratio > 1 && live.stats.rebuild_started == reference.stats.rebuild_started &&
      live.stats.solve_n.completed == reference.stats.solve_n.completed &&
      live.stats.solve_t.completed == reference.stats.solve_t.completed,
      "真实舍入残差须触发同一换基后的完整重建");
}

void test_budget_prefixes() {
  const RangedFixture fixture(2);
  std::size_t total = 0;
  const auto complete = fixture.solve(true, 64, [&] { ++total; });
  expect_accepted(fixture.model, complete, -4);
  require(complete.stats.feasibility_update_live.accepted == 2, "完整路径实际接受两个增量");
  bool attempted = false, accepted = false;
  for (std::size_t cutoff = 1; cutoff <= total; ++cutoff) {
    std::size_t calls = 0;
    const auto stopped = fixture.solve(true, 64, [&] {
      if (++calls == cutoff) throw warm::Failure(warm::Status::time_limit, "组件预算截止");
    });
    require(stopped.status == warm::Status::time_limit && stopped.x.empty() && stopped.y.empty() &&
        calls == cutoff, "任何预算前缀均不得交付候选");
    const auto& live = stopped.stats.feasibility_update_live;
    require(live.predicted <= live.attempted && live.verification_completed <= live.verification_started &&
        live.accepted + live.fallback <= live.attempted &&
        live.fallback_rebuild_completed <= live.fallback_rebuild_started &&
        live.refresh_completed <= live.refresh_started, "中断必须保留实际工作前缀");
    attempted = attempted || live.attempted > 0;
    accepted = accepted || live.accepted > 0;
  }
  require(attempted && accepted, "预算测试实际覆盖增量预测和验收后的边界");

  try {
    (void)cold::run(fixture.model, policy(), warm::Clock::now() - std::chrono::seconds(1));
  } catch (const warm::cross::Failure& error) {
    require(error.status == warm::cross::Status::time_limit, "内部接口须沿用已到期的绝对时限");
    return;
  }
  require(false, "已过绝对时限时内部接口不得返回候选");
}

void test_public_solver() {
  const auto model = independent_ranges(3);
  const auto optimal = solver::solve_simplex(model, policy());
  require(optimal.backend == "simplex" && optimal.status == "optimal" && optimal.has_solution &&
      optimal.solution == std::vector<double>({2, 2, 2}) && optimal.row_dual.size() == 3 &&
      optimal.objective == -6 && optimal.simplex_iterations == 6 &&
      optimal.verification.valid_input && optimal.verification.feasible &&
      optimal.verification.objective_consistent && optimal.lp_optimality &&
      optimal.lp_optimality->valid_input && optimal.lp_optimality->dual_feasible &&
      optimal.lp_optimality->gap_satisfied && optimal.lp_optimality->numerically_optimal,
      "公开最优结果必须提供完整原模型验收、行对偶和共享步数");

  const auto limited = solver::solve_simplex(model, policy(0));
  require(limited.status == "iteration_limit" && limited.simplex_iterations == 0 &&
      !limited.has_solution && limited.solution.empty(), "零步限制必须返回未交付状态");
  auto zero_time = policy();
  zero_time.time_limit = 0;
  const auto timeout = solver::solve_simplex(model, zero_time);
  require(timeout.status == "time_limit" && !timeout.has_solution && timeout.solution.empty(),
      "零时间预算不可提交冷启动候选");
  const auto unresolved = solver::solve_simplex(scalar(-1, 0, inf, -inf, -1), policy());
  require(unresolved.status == "unresolved" && !unresolved.has_solution && !unresolved.objective,
      "未构造证明时保留无法判定状态");

  auto integer = model;
  integer.kinds[0] = VariableKind::integer;
  auto invalid = model;
  invalid.column_upper[0] = -1;
  for (const auto& unsupported : {integer, invalid}) {
    const auto error = solver::solve_simplex(unsupported, policy());
    require(error.status == "error" && !error.has_solution && error.solution.empty(),
        "自主连续单纯形入口拒绝整数和非法模型");
  }
}
}  // namespace

static int production_regression_suite() {
  try {
    test_initial_branches();
    test_bound_shapes_and_objective();
    test_shared_limits_and_scaling();
    test_real_rounding_fallback();
    test_budget_prefixes();
    test_public_solver();
    std::puts("Cold simplex components passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}

namespace {
struct ObjectiveFixture {
  Model model;
  warm::cross::Prepared prepared;
  solver::detail::logical_basis::Result basis;
  solver::SolveOptions options = policy();

  explicit ObjectiveFixture(Model input) : model(std::move(input)) {
    prepared = cold::prepare_structure(model,warm::Clock::now()+std::chrono::seconds(10));
    basis = solver::detail::logical_basis::build(
        {prepared.costs,prepared.lower,prepared.upper,prepared.rhs.size()}, options.simplex.optimality);
    require(basis.status == solver::detail::logical_basis::Status::ready ||
        basis.status == solver::detail::logical_basis::Status::dual_sign_infeasible,
        "构造真实合法初始状态");
  }
  template<class Budget>
  warm::Result solve(bool live, std::uint64_t cap, Budget&& budget) const {
    warm::Limits limits{cap,true,true,true};
    limits.incremental_objective_dual_update = live;
    return warm::run({model,prepared,basis.variable_status,basis.row_status,basis.structural_seed},
        warm::Clock::now()+std::chrono::seconds(10),limits,options.tolerances,options.simplex.optimality,
        std::forward<Budget>(budget));
  }
  warm::Result solve(bool live, std::uint64_t cap=64) const {
    return solve(live,cap,[] {});
  }
};

void compare_objective_paths(const warm::Result& reference, const warm::Result& live) {
  require(reference.status == live.status && same_bits(reference.x,live.x) && same_bits(reference.y,live.y),
      "解析例完整重建与原目标增量路径的最终状态/原坐标解应逐位一致");
  const auto& a=reference.stats; const auto& b=live.stats;
  require(a.steps_started == b.steps_started && a.steps_completed == b.steps_completed &&
      a.pivots == b.pivots && a.dual_pivots == b.dual_pivots && a.bound_flips == b.bound_flips &&
      a.refactors == b.refactors && a.factor.completed == b.factor.completed &&
      a.feasibility_started == b.feasibility_started && a.feasibility_steps == b.feasibility_steps &&
      a.objective_restored == b.objective_restored, "解析例保留换基轨迹和共享预算");
  const auto& off=a.objective_update_live;
  require(!off.enabled && off.attempted == 0 && off.accepted == 0 && off.fallback == 0 &&
      off.verification_entries == 0 && off.refresh_started == 0,
      "关闭原目标增量路径不得增加相应工作");
}

void objective_live_counts(const warm::Result& reference, const warm::Result& result, std::uint64_t expected) {
  const auto& s=result.stats.objective_update_live;
  require(s.enabled && s.attempted == expected && s.predicted == expected &&
      s.verification_started == expected && s.verification_completed == expected && s.accepted == expected &&
      s.fallback == 0 && s.prediction_failures == 0 && s.equation_failures == 0 &&
      s.basic_dual_failures == 0 && s.nonbasic_dual_failures == 0 && s.reduced_failures == 0,
      "每次接受应完整验算原方程和对偶资格");
  require(s.refresh_completed <= expected &&
      reference.stats.rebuild_completed-result.stats.rebuild_completed == expected-s.refresh_completed &&
      reference.stats.solve_n.completed-result.stats.solve_n.completed == expected-s.refresh_completed &&
      reference.stats.solve_t.completed-result.stats.solve_t.completed == expected-s.refresh_completed,
      "真实节省必须扣除原目标刷新及其两次求解");
  require(s.max_equation_ratio <= 1 && s.max_basic_dual_ratio <= 1 && s.max_nonbasic_dual_ratio <= 1 &&
      s.max_reduced_residual_ratio <= 1, "原目标增量验收不放宽原精度");
  if (expected) require(s.verification_columns && s.verification_entries && s.work_items,
      "完整原始与对偶验算必须真实扫描矩阵");
  std::printf("objective_live accepted=%llu fallback=%llu refresh=%llu reference_rebuilds=%llu live_rebuilds=%llu equation_ratio=%.17g reduced_ratio=%.17g\n",
      static_cast<unsigned long long>(s.accepted),static_cast<unsigned long long>(s.fallback),
      static_cast<unsigned long long>(s.refresh_completed),static_cast<unsigned long long>(reference.stats.rebuild_completed),
      static_cast<unsigned long long>(result.stats.rebuild_completed),static_cast<double>(s.max_equation_ratio),
      static_cast<double>(s.max_reduced_residual_ratio));
}

Model coupled_objective_model() {
  Model m;
  m.costs={1,1}; m.kinds.assign(2,VariableKind::continuous);
  m.column_lower={0,0}; m.column_upper={inf,inf};
  m.row_lower={1,1}; m.row_upper={inf,inf};
  m.column_start={0,2,4}; m.row_index={0,1,0,1}; m.values={1,0.25,0.5,1};
  return m;
}

void test_objective_live_shapes() {
  auto negative=scalar(-1,-inf,0,1,inf); negative.values[0]=-1;
  auto rational=scalar(1,0,inf,1,inf); rational.values[0]=3;
  auto maximum=scalar(-1,0,inf,1,2); maximum.sense=ObjectiveSense::maximize; maximum.objective_offset=7;
  const std::vector<std::pair<Model,double>> models{
      {scalar(1,0,inf,1,1),1},{scalar(1,0,inf,1,2),1},{scalar(1,0,inf,1,inf),1},
      {negative,1},{rational,1.0/3.0},{scalar(0,-inf,inf,1,inf),0},{maximum,6}};
  for (const auto& [model,objective]:models) {
    const ObjectiveFixture fixture(model);
    const auto reference=fixture.solve(false), live=fixture.solve(true);
    compare_objective_paths(reference,live); objective_live_counts(reference,live,1);
    expect_accepted(model,live,objective,fixture.options);
    require(!live.stats.feasibility_started && live.stats.dual_pivots == 1,
        "原目标已对偶可行案例不进入零目标阶段");
  }
  const ObjectiveFixture fixture(negative);
  const auto live=fixture.solve(true);
  require(live.stats.last_pivot && *live.stats.last_pivot == -1 && live.x[0] == -1,
      "确实执行负FTRAN主元的原目标对偶更新");
}

void test_objective_live_coupled_and_fixed() {
  const ObjectiveFixture coupled(coupled_objective_model());
  const auto reference=coupled.solve(false), live=coupled.solve(true);
  compare_objective_paths(reference,live); objective_live_counts(reference,live,2);
  expect_accepted(coupled.model,live,10.0/7.0,coupled.options);
  require(std::abs(live.x[0]-4.0/7.0)<1e-12 && std::abs(live.x[1]-6.0/7.0)<1e-12,
      "耦合案例按更新后的非零乘子继续推进");

  Model fixed;
  fixed.costs={1,-7}; fixed.kinds.assign(2,VariableKind::continuous);
  fixed.column_lower={0,2}; fixed.column_upper={inf,2};
  fixed.row_lower={3}; fixed.row_upper={inf};
  fixed.column_start={0,1,2}; fixed.row_index={0,0}; fixed.values={1,1};
  const ObjectiveFixture fixed_fixture(fixed);
  const auto fixed_ref=fixed_fixture.solve(false), fixed_live=fixed_fixture.solve(true);
  compare_objective_paths(fixed_ref,fixed_live); objective_live_counts(fixed_ref,fixed_live,1);
  expect_accepted(fixed,fixed_live,-13,fixed_fixture.options);
}

void test_objective_live_caps() {
  for (std::size_t count:{30U,64U,65U}) {
    auto model=independent_ranges(count); model.costs.assign(count,1);
    const ObjectiveFixture fixture(model);
    const auto reference=fixture.solve(false), live=fixture.solve(true);
    compare_objective_paths(reference,live); objective_live_counts(reference,live,std::min<std::size_t>(count,64));
    if(count<=64) expect_accepted(model,live,static_cast<double>(count),fixture.options);
    else require(live.status == warm::Status::step_limit && live.stats.steps_completed == 64 &&
        live.x.empty() && live.y.empty(), "原目标实际增量也受共享64步约束");
  }
  auto model=independent_ranges(30); model.costs.assign(30,1);
  const ObjectiveFixture fixture(model);
  for(std::uint64_t cap:{0U,1U,29U,30U}) {
    const auto reference=fixture.solve(false,cap), live=fixture.solve(true,cap);
    compare_objective_paths(reference,live); objective_live_counts(reference,live,cap);
    if(cap<30) require(live.status == warm::Status::step_limit && live.stats.steps_completed == cap,
        "不得为原目标刷新重置预算");
    else expect_accepted(model,live,30,fixture.options);
  }
}

void test_objective_live_transitions() {
  const ObjectiveFixture feasibility(scalar(-1,0,inf,1,2));
  const auto reference=feasibility.solve(false), live=feasibility.solve(true);
  compare_objective_paths(reference,live); objective_live_counts(reference,live,0);
  expect_accepted(feasibility.model,live,-2,feasibility.options);
  require(live.stats.feasibility_started && live.stats.objective_restored &&
      live.stats.feasibility_update_live.accepted == 1,
      "零目标真实增量仍工作且恢复原目标后不混淆两个账本");
  for(const Model& model:{scalar(-1,0,inf,-inf,2),scalar(1,0,inf,-inf,-1)}) {
    const ObjectiveFixture fixture(model);
    const auto off=fixture.solve(false), on=fixture.solve(true);
    compare_objective_paths(off,on); objective_live_counts(off,on,0);
  }
  const auto coupled=coupled_objective_model();
  const auto options=policy();
  const auto full=cold::run(coupled,options,warm::Clock::now()+std::chrono::seconds(10),false);
  const auto selected=cold::run(coupled,options,warm::Clock::now()+std::chrono::seconds(10),true);
  compare_objective_paths(full,selected); objective_live_counts(full,selected,2);
  expect_accepted(coupled,selected,10.0/7.0,options);
}

void test_objective_live_real_fallback() {
  auto model=scalar(1,0,inf,1,inf); model.values[0]=3;
  ObjectiveFixture fixture(model);
  fixture.options.tolerances.feasibility_absolute=0;
  fixture.options.tolerances.feasibility_relative=0;
  const auto reference=fixture.solve(false), live=fixture.solve(true);
  require(reference.status != warm::Status::accepted && live.status == reference.status &&
      live.x.empty() && live.y.empty(), "真实舍入失败后不得发布原目标候选");
  const auto& s=live.stats.objective_update_live;
  require(s.attempted == 1 && s.predicted == 1 && s.verification_started == 1 &&
      s.verification_completed == 1 && s.accepted == 0 && s.fallback == 1 && s.equation_failures == 1 &&
      s.fallback_rebuild_started == 1 && s.max_equation_ratio > 1,
      "3*double(1/3)的严格方程检查应真实触发回退");
  require(reference.stats.solve_n.completed == live.stats.solve_n.completed &&
      reference.stats.solve_t.completed == live.stats.solve_t.completed &&
      reference.stats.rebuild_started == live.stats.rebuild_started,
      "拒绝后须实际执行原重建及两次LU求解");
  std::printf("objective_fallback status=%s started=%llu completed=%llu\n",warm::status_name(live.status),
      static_cast<unsigned long long>(s.fallback_rebuild_started),static_cast<unsigned long long>(s.fallback_rebuild_completed));
}

void test_objective_live_budget_prefixes() {
  const ObjectiveFixture fixture(coupled_objective_model());
  std::size_t total=0;
  const auto full=fixture.solve(true,64,[&]{++total;});
  expect_accepted(fixture.model,full,10.0/7.0,fixture.options);
  bool attempted=false,accepted=false,refreshed=false;
  for(std::size_t cutoff=1;cutoff<=total;++cutoff) {
    std::size_t calls=0;
    const auto stopped=fixture.solve(true,64,[&]{
      if(++calls==cutoff) throw warm::Failure(warm::Status::time_limit,"原目标实际更新预算前缀");
    });
    require(stopped.status == warm::Status::time_limit && stopped.x.empty() && stopped.y.empty() && calls==cutoff,
        "任何实际增量预算中断不得交付候选");
    const auto& s=stopped.stats.objective_update_live;
    require(s.predicted<=s.attempted && s.verification_completed<=s.verification_started &&
        s.accepted+s.fallback<=s.attempted && s.fallback_rebuild_completed<=s.fallback_rebuild_started &&
        s.refresh_completed<=s.refresh_started,"预算账本保留真实完成前缀");
    attempted=attempted||s.attempted>0; accepted=accepted||s.accepted>0; refreshed=refreshed||s.refresh_started>0;
  }
  require(attempted&&accepted&&refreshed,"实际预算路径跨越预测、接受和最终精确刷新");
  std::printf("objective_live_budget_prefixes=%zu\n",total);
}
}  // 命名空间

int main() {
  if(production_regression_suite()!=0) return 1;
  try {
    test_objective_live_shapes(); std::puts("PASS objective_live_shapes_and_signed_pivot");
    test_objective_live_coupled_and_fixed(); std::puts("PASS objective_live_coupled_and_fixed");
    test_objective_live_caps(); std::puts("PASS objective_live_30_64_65_and_caps");
    test_objective_live_transitions(); std::puts("PASS objective_live_stage_transitions");
    test_objective_live_real_fallback(); std::puts("PASS objective_live_true_rounding_fallback");
    test_objective_live_budget_prefixes(); std::puts("PASS objective_live_actual_budget_prefixes");
    std::puts("cold_simplex_r152_groups_passed=12");
    return 0;
  } catch(const std::exception& error) {std::fprintf(stderr,"FAIL %s\n",error.what());return 1;}
}

