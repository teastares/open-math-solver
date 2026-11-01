#include "ipm-pair.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

namespace {

int cases = 0, native_cases = 0, fallback_cases = 0;
int finish0_refinement_cases = 0, repair_cases = 0, budget_cases = 0;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

solver::SolveResult work() {
  solver::SolveResult r;
  r.ipm_iterations = 0;
  r.ipm_factorizations = 0;
  r.ipm_unshifted_factorizations = 0;
  r.ipm_regularized_pivots = 0;
  r.ipm_refinement_steps = 0;
  r.ipm_weighted_krylov_steps = 0;
  r.ipm_structural_fallbacks = 0;
  r.ipm_krylov_recycle_projections = 0;
  r.ipm_krylov_recycle_acceptances = 0;
  r.ipm_krylov_recycle_vectors = 0;
  r.ipm_krylov_recycle_peak_dimension = 0;
  r.ipm_backward_error_acceptances = 0;
  r.ipm_inexact_acceptances = 0;
  r.ipm_inexact_prefilter_rejections = 0;
  r.ipm_homogeneous_constant_solves = 0;
  r.ipm_homogeneous_constant_repairs = 0;
  r.ipm_homogeneous_constant_repair_successes = 0;
  r.ipm_homogeneous_direction_solves = 0;
  return r;
}

solver::ConeModel cone() {
  solver::ConeModel model;
  model.costs = {-1, 2};
  model.rhs = {2, -1, 3};
  model.starts = {0, 2, 4};
  model.rows = {0, 1, 1, 2};
  model.values = {1, 2, 3, 1};
  model.inequality = {true, true, true};
  return model;
}

std::vector<double> product(const solver::ConeModel& model, const std::vector<double>& h,
    const std::vector<double>& x) {
  const auto n = model.costs.size();
  std::vector<long double> y(n + model.rhs.size(), 0);
  for (std::size_t row = 0; row < h.size(); ++row) y[n + row] = -static_cast<long double>(h[row]) * x[n + row];
  for (std::size_t col = 0; col < n; ++col) {
    for (auto k = model.starts[col]; k < model.starts[col + 1]; ++k) {
      const auto entry = static_cast<std::size_t>(k), row = static_cast<std::size_t>(model.rows[entry]);
      y[col] += static_cast<long double>(model.values[entry]) * x[n + row];
      y[n + row] += static_cast<long double>(model.values[entry]) * x[col];
    }
  }
  return {y.begin(), y.end()};
}

void check_equation(const solver::ConeModel& model, const std::vector<double>& h,
    const std::vector<double>& rhs, const std::vector<double>& answer, const std::vector<double>& known) {
  const auto actual = product(model, h, answer);
  double scale = 0, defect = 0;
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    scale = std::max(scale, std::abs(rhs[i]));
    defect = std::max(defect, std::abs(actual[i] - rhs[i]));
    require(std::abs(answer[i] - known[i]) <= 1e-8 * std::max(1.0, std::abs(known[i])),
        "KKT已知解前向误差不合格");
  }
  require(defect <= 1e-12 + 1e-10 * scale, "成对求解不能放宽原完整KKT残差门槛");
}

bool zero_stats() {
  const auto& s = solver_v83_pair::stats;
  return s.pairs_started == 0 && s.pairs_completed == 0 && s.normal_calls_started == 0 &&
      s.normal_calls_completed == 0 && s.finish0_started == 0 && s.finish0_completed == 0 &&
      s.finish1_started == 0 && s.finish1_completed == 0;
}

void one_pair() {
  const auto& s = solver_v83_pair::stats;
  require(s.pairs_started == 1 && s.pairs_completed == 1 && s.normal_calls_started == 1 &&
      s.normal_calls_completed == 1 && s.finish0_started == 1 && s.finish0_completed == 1 &&
      s.finish1_started == 1 && s.finish1_completed == 1, "原生成对工作必须真实执行两次独立验收");
}

void check_case(const std::string& backend, bool singleton, bool equilibrate, bool enabled = true) {
  const auto model = cone();
  const std::vector<double> h{1, 2, 3}, x0{1, -2, 0.5, -0.25, 1.5}, x1{2, 0.5, -1, 0.75, -0.5};
  const auto b0 = product(model, h, x0), b1 = product(model, h, x1);
  solver::SolveOptions options;
  options.time_limit = 60;
  options.ipm.linear_solver = backend;
  options.ipm.linear_accuracy = "strict";
  options.ipm.eliminate_singletons = singleton;
  options.ipm.equilibrate_kkt = equilibrate;
  const auto start = solver::Clock::now();
  auto original_work = work(), paired_work = work();
  solver::KktFactor original(model, options, start, original_work), paired(model, options, start, paired_work);
  original.factor(h, options, start, original_work);
  paired.factor(h, options, start, paired_work);
  const auto original0 = original.solve(b0, options, start, original_work);
  const auto original1 = original.solve(b1, options, start, original_work);
  solver_v83_pair::enabled = enabled;
  solver_v83_pair::stats = {};
  const auto answers = paired.solve_pair(b0, b1, options, start, paired_work);
  check_equation(model, h, b0, answers[0], x0);
  check_equation(model, h, b1, answers[1], x1);
  if (enabled && paired.supports_native_pair()) {
    one_pair();
    ++native_cases;
  } else {
    require(zero_stats(), "关闭或不支持路径不得冒称执行原生pair");
    require(answers[0] == original0 && answers[1] == original1 &&
        paired_work.ipm_refinement_steps == original_work.ipm_refinement_steps &&
        paired_work.ipm_weighted_krylov_steps == original_work.ipm_weighted_krylov_steps &&
        paired_work.ipm_schur_residual_products == original_work.ipm_schur_residual_products &&
        paired_work.ipm_schur_refinement_steps == original_work.ipm_schur_refinement_steps &&
        paired_work.ipm_schur_refinement_acceptances == original_work.ipm_schur_refinement_acceptances,
        "回退必须保持完整solve0后完整solve1及原精修账本");
    ++fallback_cases;
  }
  ++cases;
}

void check_first_refinement() {
  const auto model = cone();
  const std::vector<double> h{1, 2, 3}, x0{1, -2, 0.5, -0.25, 1.5}, zero(5, 0);
  const auto b0 = product(model, h, x0);
  solver::SolveOptions options;
  options.time_limit = 60;
  options.ipm.linear_solver = "normal_cholmod_aat";
  const auto start = solver::Clock::now();
  auto r = work();
  solver::KktFactor factor(model, options, start, r);
  // 原正则网格第6级使首右端实际需要精修，第二右端严格为零且无需精修。
  factor.factor(h, options, start, r, 6);
  require(factor.supports_native_pair(), "精修所有权测试须实际进入CHOLMOD成对路径");
  solver_v83_pair::enabled = true;
  solver_v83_pair::stats = {};
  const auto answers = factor.solve_pair(b0, zero, options, start, r);
  require(*r.ipm_refinement_steps > 0, "第0右端须实际经历原IR或Krylov修正");
  check_equation(model, h, b0, answers[0], x0);
  require(answers[1] == zero, "第0右端单RHS精修不得污染仍待验收的第1初解");
  one_pair();
  ++finish0_refinement_cases;
  ++cases;
}

void check_repair_corrector() {
  solver::ConeModel model;
  model.costs = {-1}; model.rhs = {2}; model.starts = {0, 1}; model.rows = {0};
  model.values = {1}; model.inequality = {true};
  solver::Point point;
  point.x = {0}; point.s = {0.5}; point.z = {1};
  solver::SolveOptions options;
  options.time_limit = 60;
  options.ipm.linear_solver = "normal_cholmod_aat";
  options.ipm.linear_accuracy = "homogeneous_repair";
  const auto start = solver::Clock::now();
  auto r = work();
  solver::KktFactor factor(model, options, start, r);
  factor.factor({0.5}, options, start, r);
  const auto u = factor.solve(std::vector<double>{-1, 2}, options, start, r);
  std::vector<double> constant{4, 4};
  solver::detail::IpmHomogeneousRepairInfo info;
  const auto affine = solver::homogeneous_direction_from_solution(model, point, 1, 1, 5, 1,
      std::vector<double>{-0.5}, -1, constant, factor, options, start, r, u, &info);
  require(info.attempted && info.succeeded && *r.ipm_homogeneous_constant_repairs == 1 &&
      *r.ipm_homogeneous_constant_repair_successes == 1, "原有限非正分母必须实际触发一次严格常量修复");
  require(std::abs(constant[0] - 2.5) <= 1e-9 && std::abs(constant[1] - 1) <= 1e-9,
      "修复必须把原lvalue更新为完整KKT常量解");
  const auto check = [](const solver::HomogeneousDirection& direction, double upper, double lower,
      double gap, double complementarity, double h) {
    const auto dx = direction.point.x[0], dz = direction.point.z[0], ds = direction.point.s[0];
    const auto dt = direction.scalars.tau, dk = direction.scalars.kappa;
    // 原间隙为 kappa+c^T*x+b^T*z，方向方程的 d_kappa 系数为正。
    require(std::abs(dz - dt - upper) <= 1e-9 && std::abs(dx - 0.5 * dz - 2 * dt - lower) <= 1e-9 &&
        std::abs(-dx + 2 * dz + dk - gap) <= 1e-9 && std::abs(dt + dk - complementarity) <= 1e-9 &&
        std::abs(ds + 0.5 * dz - h) <= 1e-9, "齐次恢复必须满足原五块方程");
  };
  check(affine, -1, 2, -5, -1, -0.5);
  const auto repaired = constant;
  const auto corrected = solver::homogeneous_direction(model, point, 1, 1, std::vector<double>{-1.5},
      std::vector<double>{1}, 5, 0.4, std::vector<double>{-0.125}, -0.25, constant,
      factor, options, start, r, &info);
  check(corrected, -0.4, 0.725, -2, -0.25, -0.125);
  require(constant == repaired && *r.ipm_homogeneous_constant_repairs == 1,
      "corrector必须复用已修复常量且不能重置一次修复额度");
  ++repair_cases;
  ++cases;
}

void check_budget() {
  const auto model = cone();
  const std::vector<double> h{1, 2, 3}, b0{1, 2, 3, 4, 5}, b1{-2, 3, -4, 5, 1};
  solver::SolveOptions options;
  options.time_limit = 60;
  options.ipm.linear_solver = "normal_cholmod_aat";
  const auto start = solver::Clock::now();
  auto r = work();
  solver::KktFactor factor(model, options, start, r);
  factor.factor(h, options, start, r);
  solver_v83_pair::enabled = true;
  solver_v83_pair::stats = {};
  options.time_limit = 0;
  bool interrupted = false;
  try { factor.solve_pair(b0, b1, options, start, r); }
  catch (const solver::BudgetExpired&) { interrupted = true; }
  require(interrupted && zero_stats(), "入口预算异常须保留原类型且不得伪记启动");
  ++budget_cases;
  ++cases;
}

}  // 匿名命名空间

int main() {
  try {
    for (const auto& backend : {"normal_cholmod", "normal_cholmod_aat"})
      for (const bool singleton : {false, true})
        for (const bool equilibrate : {false, true}) check_case(backend, singleton, equilibrate);
    for (const auto& backend : {"qdldl", "qdldl_extended", "sparselu", "factorhighs", "normal", "normal_multifrontal", "normal_extended", "normal_refined"})
      check_case(backend, true, true);
    check_case("normal_cholmod_aat", true, true, false);
    check_first_refinement();
    check_repair_corrector();
    check_budget();
    std::printf("{\"passed\":true,\"solver_runs\":0,\"cases\":%d,\"native_cases\":%d,\"fallback_cases\":%d,"
        "\"finish0_refinement_cases\":%d,\"repair_cases\":%d,\"budget_cases\":%d}\n",
        cases, native_cases, fallback_cases, finish0_refinement_cases, repair_cases, budget_cases);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
