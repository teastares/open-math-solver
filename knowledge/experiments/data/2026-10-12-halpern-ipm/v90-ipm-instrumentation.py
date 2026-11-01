"""v90：只改变HSD中心参数的有限精度预测反馈，不改方程门槛或终验。"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "c5ef31fe54cdf254c452fe26437faf5ac8da1c785df682a84797022a9e13d842"
PARENT_INSTRUMENTATION_SHA256 = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
ENVIRONMENT = "SOLVER_V90_PRESOLVE"
PREFIX = "V90_PRESOLVE "
COUNT_FIELDS = (
    "ipm_iterations", "ipm_factorizations", "ipm_refinement_steps", "ipm_weighted_krylov_steps",
    "ipm_full_newton_preconditioner_solves", "ipm_full_newton_products", "ipm_bounded_newton_returns",
    "ipm_bounded_newton_inexact_returns", "ipm_homogeneous_constant_solves", "ipm_homogeneous_constant_repairs",
    "ipm_homogeneous_constant_repair_successes", "ipm_homogeneous_direction_solves", "ipm_homogeneous_scaled_products",
    "ipm_homogeneous_scaled_refinement_steps", "ipm_homogeneous_scaled_inexact_returns", "ipm_regularization_searches",
    "ipm_regularization_reductions", "ipm_unshifted_factorizations", "ipm_direction_retries",
    "ipm_direction_retry_successes", "ipm_krylov_recycle_projections", "ipm_krylov_recycle_acceptances",
    "ipm_krylov_recycle_vectors", "ipm_krylov_recycle_peak_dimension", "ipm_schur_residual_products",
    "ipm_schur_refinement_steps", "ipm_schur_refinement_acceptances", "ipm_structural_fallbacks",
    "ipm_eliminated_fixed_variables", "ipm_presolved_singleton_rows", "ipm_presolved_fixed_columns",
    "ipm_regularized_pivots", "ipm_backward_error_acceptances", "ipm_inexact_acceptances",
    "ipm_inexact_prefilter_rejections", "ipm_central_correction_attempts", "ipm_central_correction_acceptances")

INITIALIZATION = '''  SolveResult result;
  result.backend = "ipm";
  result.backend_version = ipm_version();
  result.model = model;
''' + ''.join('  result.' + field + ' = 0;\n' for field in COUNT_FIELDS)

HEADER = r'''#pragma once
#include "Highs.h"
#include "solver/ipm.hpp"
#include "solver/lp_optimality.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace solver_v90_presolve {
using Clock = std::chrono::steady_clock;
using Now = std::function<Clock::time_point()>;
using solver::Model;
using solver::SolveOptions;
using solver::SolveResult;
using solver::ipm_version;
inline thread_local bool emit_diagnostics = true;
inline thread_local std::optional<bool> enabled_override;
inline bool enabled() {
  if (enabled_override) return *enabled_override;
  const char* value = std::getenv("SOLVER_V90_PRESOLVE");
  return value && std::string_view(value) == "1";
}
struct Dimensions { std::int64_t n = 0, m = 0, nnz = 0; };
struct Call { std::int64_t started = 0, completed = 0; std::optional<int> status; };
struct Child {
  std::int64_t started = 0, completed = 0;
  std::optional<std::string> status;
  std::optional<bool> has_solution;
};
struct Assessments { std::int64_t started = 0, completed = 0; };
struct NativeOptions {
  double primal_feasibility_tolerance = 0, dual_feasibility_tolerance = 0, ipm_optimality_tolerance = 0;
  bool lp_presolve_requires_basis_postsolve = true;
};
struct Native {
  std::optional<int> presolve_status, model_status;
  std::optional<bool> info_valid, basis_valid;
  std::optional<std::int64_t> simplex_iterations, ipm_iterations, pdlp_iterations, crossover_iterations, qp_iterations;
  std::optional<double> time_limit;
  std::optional<NativeOptions> options;
};
struct Quality {
  bool primal_valid = false, primal_feasible = false, objective_consistent = false;
  bool dual_feasible = false, gap_satisfied = false, numerically_optimal = false;
  std::optional<double> objective, max_row_violation, max_bound_violation, max_dual_residual, objective_gap, relative_objective_gap;
};
struct Projection {
  std::int64_t started = 0, completed = 0, x_changed = 0, y_changed = 0;
  const char* status = "not_started";
  const char* phase = "not_started";
  const char* chosen = "none";
  std::optional<double> max_x_change, max_y_change;
  std::optional<Quality> raw_quality, trial_quality;
};
struct Timings {
  double preparation = 0, presolve = 0, child = 0, recovery = 0, verification = 0, child_verification = 0, total = 0;
};
struct Work { std::int64_t checks = 0, loop_units = 0; };
struct OracleEvent {
  std::int64_t sequence = 0;
  std::optional<std::int64_t> iteration;
  const char* decision = "pending";
  Call postsolve;
  Assessments assessments;
  Projection projection;
  std::optional<Quality> original_quality;
  struct Times { double total = 0, recovery = 0, verification = 0; } timings;
  Work budget;
};
struct OracleStats {
  std::int64_t started = 0, completed = 0, accepted = 0, rejected = 0, budget_failures = 0, errors = 0;
  std::optional<std::int64_t> candidate_event, selected_event, projection_event;
  double total_seconds = 0;
  std::vector<OracleEvent> events;
};
struct CenterLast {
  std::optional<std::int64_t> iteration;
  const char* status = "running";
  std::optional<double> alpha, current_c, affine_c, ratio, sigma, legacy_sigma;
  std::int64_t negative_predictions = 0, rows_scanned = 0, inequality_terms = 0;
  bool scalar_term_complete = false, sigma_underflow = false;
};
struct CenterDiagnostics {
  bool enabled = false;
  std::int64_t started = 0, completed = 0, rows_scanned = 0, inequality_terms = 0, budget_checks = 0;
  std::optional<CenterLast> last;
};
struct Stats {
  bool enabled = false, attempted = false, accepted = false;
  const char* status = "not_started";
  const char* phase = "not_started";
  const char* route = "none";
  std::optional<Dimensions> original, reduced;
  Call pass_model, presolve, postsolve;
  Native native;
  Child child;
  Assessments assessments;
  std::optional<Quality> original_quality;
  Projection projection;
  Timings timings;
  Work budget;
  OracleStats oracle;
  CenterDiagnostics center_diag;
};
struct Expired {};
struct NativeError : std::runtime_error { using std::runtime_error::runtime_error; };
struct RejectedStatus : std::runtime_error { using std::runtime_error::runtime_error; };
struct CenterError : std::runtime_error { using std::runtime_error::runtime_error; };
inline thread_local CenterDiagnostics* active_center = nullptr;
struct CenterScope {
  explicit CenterScope(CenterDiagnostics* next) noexcept : previous(active_center) { active_center = next; }
  ~CenterScope() noexcept { active_center = previous; }
  CenterScope(const CenterScope&) = delete;
  CenterScope& operator=(const CenterScope&) = delete;
  CenterDiagnostics* previous;
};
inline std::optional<double> center_summary(long double value) noexcept {
  const double stored = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(stored) || (value != 0 && stored == 0)) return std::nullopt;
  return stored;
}

// 精确HSDE仿射方程使此比值等于1-alpha；这里只反馈已存方向的数值偏离。
// Check只调用原epoch预算，任何BudgetExpired或用户回调异常均原样透传。
template<class Check>
double predicted_center(const std::vector<bool>& inequality, const std::vector<double>& s,
    const std::vector<double>& z, const std::vector<double>& ds, const std::vector<double>& dz,
    double tau, double kappa, double dtau, double dkappa, double alpha, long double current_c,
    double legacy_sigma, std::optional<std::int64_t> iteration, CenterDiagnostics& diagnostics, Check&& check) {
  static_assert(std::is_nothrow_default_constructible_v<CenterLast>);
  ++diagnostics.started; diagnostics.last.emplace();
  auto& last = *diagnostics.last;
  last.iteration = iteration; last.alpha = center_summary(alpha); last.current_c = center_summary(current_c);
  last.legacy_sigma = center_summary(legacy_sigma);
  const auto budget = [&] { ++diagnostics.budget_checks; check(); };
  const auto fail = [&](const char* message) -> void { last.status = "arithmetic_error"; throw CenterError(message); };
  budget();
  if (s.size() != inequality.size() || z.size() != s.size() || ds.size() != s.size() || dz.size() != s.size())
    fail("HSD预测中心参数的行维数不一致");
  if (!(current_c > 0) || !std::isfinite(current_c) || !std::isfinite(alpha) || alpha < 0 || alpha > 1 ||
      !std::isfinite(legacy_sigma) || legacy_sigma < 0 || legacy_sigma > 1 ||
      !(tau > 0) || !(kappa > 0) || !std::isfinite(tau) || !std::isfinite(kappa) ||
      !std::isfinite(dtau) || !std::isfinite(dkappa)) fail("HSD预测中心参数的标量输入无效");
  const long double predicted_tau = static_cast<long double>(tau) + static_cast<long double>(alpha) * dtau;
  const long double predicted_kappa = static_cast<long double>(kappa) + static_cast<long double>(alpha) * dkappa;
  if (!std::isfinite(predicted_tau) || !std::isfinite(predicted_kappa)) fail("HSD预测齐次标量非有限");
  last.negative_predictions += predicted_tau < 0; last.negative_predictions += predicted_kappa < 0;
  long double affine_c = predicted_tau * predicted_kappa;
  if (!std::isfinite(affine_c)) fail("HSD预测齐次互补乘积非有限");
  last.scalar_term_complete = true;
  for (std::size_t row = 0; row < inequality.size(); ++row) {
    if ((row & 4095) == 0) budget();
    ++diagnostics.rows_scanned; ++last.rows_scanned;
    if (!inequality[row]) continue;
    if (!(s[row] > 0) || !(z[row] > 0) || !std::isfinite(s[row]) || !std::isfinite(z[row]) ||
        !std::isfinite(ds[row]) || !std::isfinite(dz[row])) fail("HSD预测互补行输入无效");
    const long double predicted_s = static_cast<long double>(s[row]) + static_cast<long double>(alpha) * ds[row];
    const long double predicted_z = static_cast<long double>(z[row]) + static_cast<long double>(alpha) * dz[row];
    if (!std::isfinite(predicted_s) || !std::isfinite(predicted_z)) fail("HSD预测互补坐标非有限");
    last.negative_predictions += predicted_s < 0; last.negative_predictions += predicted_z < 0;
    const long double product = predicted_s * predicted_z;
    if (!std::isfinite(product)) fail("HSD预测互补乘积非有限");
    affine_c += product;
    if (!std::isfinite(affine_c)) fail("HSD预测互补累加非有限");
    ++diagnostics.inequality_terms; ++last.inequality_terms;
  }
  last.affine_c = center_summary(affine_c);
  // 先比较再除法，避免A/C在本就应截到1时溢出；不逐坐标clamp预测点。
  const long double ratio = affine_c <= 0 ? 0 : affine_c >= current_c ? 1 : affine_c / current_c;
  if (!std::isfinite(ratio) || ratio < 0 || ratio > 1) fail("HSD预测互补总比无效");
  const long double cube = ratio * ratio * ratio;
  const double sigma = static_cast<double>(cube);
  if (!std::isfinite(cube) || !std::isfinite(sigma) || sigma < 0 || sigma > 1) fail("HSD预测中心参数非有限");
  last.ratio = center_summary(ratio); last.sigma = sigma;
  last.sigma_underflow = ratio > 0 && sigma == 0;
  budget();
  last.status = "completed"; ++diagnostics.completed;
  return sigma;
}

class Budget {
 public:
  Budget(Clock::time_point start, double limit, Stats& stats, Now now)
      : start_(start), limit_(limit), stats_(stats), now_(std::move(now)) {}
  double elapsed() const { return std::chrono::duration<double>(now_() - start_).count(); }
  void check() {
    ++stats_.budget.checks;
    if (elapsed() >= limit_) throw Expired{};
  }
  void tick() {
    ++stats_.budget.loop_units;
    if (stats_.budget.loop_units % 256 == 0) check();
  }
  double remaining() { check(); const double value = limit_ - elapsed(); if (!(value > 0)) throw Expired{}; return value; }
 private:
  Clock::time_point start_;
  double limit_;
  Stats& stats_;
  Now now_;
};
struct Stage {
  Stage(Stats& stats, const char* phase, double& seconds, Budget& budget)
      : seconds_(seconds), budget_(budget), begin_(budget.elapsed()) { stats.phase = phase; }
  ~Stage() noexcept {
    try { seconds_ += std::max(0.0, budget_.elapsed() - begin_); } catch (...) {}
  }
 private:
  double& seconds_;
  Budget& budget_;
  double begin_;
};
inline double narrow(long double value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result)) throw std::runtime_error("预处理恢复出现非有限数");
  return result;
}
inline std::optional<double> summary(double value) {
  return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
}
inline std::optional<double> summary(const std::optional<double>& value) {
  return value ? summary(*value) : std::nullopt;
}
inline Quality quality(const solver::Verification& primal, const solver::LpOptimality& dual) {
  return {primal.valid_input, primal.feasible, primal.objective_consistent, dual.dual_feasible,
      dual.gap_satisfied, dual.numerically_optimal, summary(primal.objective), summary(primal.max_row_violation),
      summary(primal.max_bound_violation), summary(dual.max_dual_residual), summary(dual.objective_gap), summary(dual.relative_objective_gap)};
}

inline SolveResult make_shell(const Model& model) {
''' + INITIALIZATION + r'''  return result;
}

inline void clear_coordinate_quality(SolveResult& result) {
  result.has_solution = false;
  result.solution.clear(); result.row_dual.clear();
  result.objective.reset(); result.best_bound.reset(); result.absolute_gap.reset(); result.relative_gap.reset();
  result.verification = solver::Verification{};
  result.lp_optimality.reset(); result.verification_level = "none";
  if (result.status == "optimal") result.status = "verification_failed";
}
inline void adopt_child(SolveResult& outer, SolveResult child) {
  static_assert(std::is_nothrow_swappable_v<Model> && std::is_nothrow_move_assignable_v<SolveResult>);
  using std::swap;
  swap(outer.model, child.model);
  outer = std::move(child);
  // 先用无抛move保护全部已支付账本，再做可能分配的质量标签清理。
  clear_coordinate_quality(outer);
}

// 与v66 assess的原始可行性、修正对偶界和sense/gap写回公式保持一致。
inline bool commit_original_quality(SolveResult& result, std::vector<double> x, std::vector<double> y,
                                    solver::LpVerification checks, const SolveOptions& options) {
  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  result.verification = std::move(checks.primal);
  result.lp_optimality = std::move(checks.optimality);
  result.row_dual = std::move(y);
  result.has_solution = feasible;
  result.solution = feasible ? std::move(x) : std::vector<double>{};
  result.objective = feasible ? result.verification.objective : std::nullopt;
  result.best_bound = result.lp_optimality->corrected_dual_bound;
  result.absolute_gap.reset(); result.relative_gap.reset();
  result.verification_level = feasible ? "primal_checked" : "residuals_checked";
  if (result.objective && result.best_bound) {
    const double sense = result.model.sense == solver::ObjectiveSense::minimize ? 1 : -1;
    const long double gap = sense * (static_cast<long double>(*result.objective) - *result.best_bound);
    const long double scale = std::max({1.0L, std::abs(static_cast<long double>(*result.objective)),
                                        std::abs(static_cast<long double>(*result.best_bound))});
    const long double tolerance = options.pdlp.optimality.gap_absolute + options.pdlp.optimality.gap_relative * scale;
    if (!std::isfinite(gap) || gap < -tolerance) result.best_bound.reset();
    else {
      const long double nonnegative = std::max(0.0L, gap);
      if (std::isfinite(static_cast<double>(nonnegative))) result.absolute_gap = static_cast<double>(nonnegative);
      result.relative_gap = narrow(nonnegative / scale);
    }
  }
  if (result.has_solution && result.lp_optimality && result.lp_optimality->numerically_optimal) {
    result.status = "optimal";
    result.verification_level = "primal_dual_residuals_checked";
    return true;
  }
  if (result.status == "optimal") result.status = "verification_failed";
  return false;
}

inline bool complete_quality(const solver::LpVerification& checked) {
  return checked.primal.valid_input && checked.primal.feasible && checked.primal.objective_consistent &&
      checked.primal.objective.has_value() && checked.optimality.valid_input && checked.optimality.dual_feasible &&
      checked.optimality.gap_satisfied && checked.optimality.numerically_optimal;
}
struct ProjectionTrial {
  std::vector<double> x, y;
  solver::LpVerification checked;
};

// 原点和原质量只读；只有完整试点通过原终验及出口预算后才交出其拥有的向量。
inline std::optional<ProjectionTrial> bounded_projection(const Model& original, const SolveOptions& options,
    const std::vector<double>& raw_x, const std::vector<double>& raw_y, const solver::LpVerification& raw_quality,
    Stats& stats, Budget& budget) {
  auto& p = stats.projection;
  p.raw_quality = quality(raw_quality.primal, raw_quality.optimality);
  p.chosen = "raw";
  if (complete_quality(raw_quality)) { p.status = "not_needed"; return std::nullopt; }
  if (p.started != 0) throw std::logic_error("同一恢复点的域投影试验最多一次");
  ++p.started; p.status = "running";
  try {
    ProjectionTrial trial;
    {
      Stage stage(stats, "projection", stats.timings.recovery, budget);
      p.phase = "copy";
      budget.check();
      if (raw_x.size() != original.costs.size() || raw_y.size() != original.row_lower.size() ||
          original.column_lower.size() != raw_x.size() || original.column_upper.size() != raw_x.size())
        throw std::runtime_error("域投影原坐标维数无效");
      trial.x.reserve(raw_x.size()); trial.y.reserve(raw_y.size());
      for (double value : raw_x) { budget.tick(); if (!std::isfinite(value)) throw std::runtime_error("域投影原始点非有限"); trial.x.push_back(value); }
      for (double value : raw_y) { budget.tick(); if (!std::isfinite(value)) throw std::runtime_error("域投影行对偶非有限"); trial.y.push_back(value); }
      p.phase = "primal"; p.max_x_change = 0;
      long double max_x = 0;
      for (std::size_t col = 0; col < trial.x.size(); ++col) {
        budget.tick();
        const double lower = original.column_lower[col], upper = original.column_upper[col];
        if (std::isnan(lower) || std::isnan(upper) || lower == std::numeric_limits<double>::infinity() ||
            upper == -std::numeric_limits<double>::infinity() || lower > upper)
          throw std::runtime_error("域投影列界无效");
        trial.x[col] = std::clamp(trial.x[col], lower, upper);
        if (trial.x[col] != raw_x[col]) ++p.x_changed;
        max_x = std::max(max_x, std::abs(static_cast<long double>(trial.x[col]) - raw_x[col]));
        // 摘要溢出可为null，不用于判断投影或验收是否通过。
        p.max_x_change = summary(static_cast<double>(max_x));
      }
      p.phase = "dual";
      // 既有API内部无预算回调；像其他库调用一样核入口/出口，自有循环仍每256单位检查。
      budget.check();
      if (!solver::project_lp_row_dual_domain(original, trial.y)) throw std::runtime_error("域投影行界或维数无效");
      budget.check();
      p.phase = "scan"; p.max_y_change = 0;
      long double max_y = 0;
      for (std::size_t row = 0; row < trial.y.size(); ++row) {
        budget.tick();
        if (trial.y[row] != raw_y[row]) ++p.y_changed;
        max_y = std::max(max_y, std::abs(static_cast<long double>(trial.y[row]) - raw_y[row]));
        p.max_y_change = summary(static_cast<double>(max_y));
      }
      budget.check();
    }
    if (p.x_changed == 0 && p.y_changed == 0) {
      ++p.completed; p.status = "unchanged"; p.phase = "finished";
      return std::nullopt;
    }
    {
      p.phase = "verification";
      Stage stage(stats, "verification", stats.timings.verification, budget);
      budget.check(); ++stats.assessments.started;
      trial.checked = solver::verify_lp_solution(original, trial.x, trial.y, options.tolerances, options.pdlp.optimality);
      ++stats.assessments.completed; p.trial_quality = quality(trial.checked.primal, trial.checked.optimality);
      budget.check();
    }
    ++p.completed; p.phase = "finished";
    if (complete_quality(trial.checked)) {
      static_assert(std::is_nothrow_move_constructible_v<ProjectionTrial>);
      p.status = "accepted"; p.chosen = "projected";
      return std::optional<ProjectionTrial>(std::move(trial));
    }
    p.status = "rejected";
    return std::nullopt;
  } catch (const Expired&) {
    p.status = "time_limit"; p.chosen = "none";
    throw;
  } catch (...) {
    // 可选试验失败只回到仍拥有的原点；复核预算失败时不得借回退提交结果。
    p.status = "exception"; p.chosen = "raw";
    try { budget.check(); }
    catch (const Expired&) { p.status = "time_limit"; p.chosen = "none"; throw; }
    return std::nullopt;
  }
}

template<class Value> inline void option(Highs& highs, const char* name, Value value, Budget& budget) {
  budget.check(); const auto status = highs.setOptionValue(name, value); budget.check();
  if (status != HighsStatus::kOk) throw NativeError(std::string("无法设置组件参数：") + name);
}
template<class Value> inline void query(const Highs& highs, const char* name, Value& value, Budget& budget) {
  budget.check(); const auto status = highs.getOptionValue(name, value); budget.check();
  if (status != HighsStatus::kOk) throw NativeError(std::string("无法读取组件参数：") + name);
}
inline void observe(Highs& highs, Stats& stats) {
  const auto& info = highs.getInfo();
  stats.native.presolve_status = static_cast<int>(highs.getModelPresolveStatus());
  stats.native.model_status = static_cast<int>(highs.getModelStatus());
  stats.native.info_valid = info.valid; stats.native.basis_valid = highs.getBasis().valid;
  stats.native.simplex_iterations = info.simplex_iteration_count;
  stats.native.ipm_iterations = info.ipm_iteration_count;
  stats.native.pdlp_iterations = info.pdlp_iteration_count;
  stats.native.crossover_iterations = info.crossover_iteration_count;
  stats.native.qp_iterations = info.qp_iteration_count;
  if (info.simplex_iteration_count > 0 || info.ipm_iteration_count > 0 || info.pdlp_iteration_count > 0 ||
      info.crossover_iteration_count > 0 || info.qp_iteration_count > 0 || highs.getBasis().valid)
    throw NativeError("纯组件路径观察到有效基或正求解迭代");
}
inline void configure(Highs& highs, Stats& stats, Budget& budget, const SolveOptions& options) {
  option(highs, "output_flag", false, budget); option(highs, "log_to_console", false, budget);
  option(highs, "threads", static_cast<HighsInt>(options.threads), budget);
  option(highs, "random_seed", static_cast<HighsInt>(options.seed), budget);
  option(highs, "presolve", std::string("choose"), budget);
  option(highs, "allow_unbounded_or_infeasible", false, budget);
  option(highs, "lp_presolve_requires_basis_postsolve", true, budget);
  NativeOptions value;
  query(highs, "primal_feasibility_tolerance", value.primal_feasibility_tolerance, budget);
  query(highs, "dual_feasibility_tolerance", value.dual_feasibility_tolerance, budget);
  query(highs, "ipm_optimality_tolerance", value.ipm_optimality_tolerance, budget);
  query(highs, "lp_presolve_requires_basis_postsolve", value.lp_presolve_requires_basis_postsolve, budget);
  stats.native.options = value;
}

inline HighsLp minimum_copy(const Model& model, Budget& budget) {
  budget.check();
  if (model.costs.size() > static_cast<std::size_t>(std::numeric_limits<HighsInt>::max()) ||
      model.row_lower.size() > static_cast<std::size_t>(std::numeric_limits<HighsInt>::max())) throw NativeError("组件维数不可表示");
  HighsLp lp;
  lp.num_col_ = static_cast<HighsInt>(model.costs.size()); lp.num_row_ = static_cast<HighsInt>(model.row_lower.size());
  lp.model_name_ = model.name; lp.sense_ = ObjSense::kMinimize;
  const double sign = model.sense == solver::ObjectiveSense::minimize ? 1 : -1;
  lp.offset_ = sign * model.objective_offset;
  lp.col_cost_.reserve(model.costs.size());
  for (double value : model.costs) { budget.tick(); lp.col_cost_.push_back(sign * value); }
  const auto copy = [&](const auto& from, auto& to) {
    using Target = typename std::decay_t<decltype(to)>::value_type;
    to.reserve(from.size());
    for (const auto value : from) {
      budget.tick();
      if constexpr (std::is_integral_v<Target> && std::is_integral_v<decltype(value)>) {
        if (value < std::numeric_limits<Target>::min() || value > std::numeric_limits<Target>::max())
          throw NativeError("组件稀疏索引不能无损表示");
      }
      to.push_back(value);
    }
  };
  copy(model.column_lower, lp.col_lower_); copy(model.column_upper, lp.col_upper_);
  copy(model.row_lower, lp.row_lower_); copy(model.row_upper, lp.row_upper_);
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.clear(); copy(model.column_start, lp.a_matrix_.start_);
  copy(model.row_index, lp.a_matrix_.index_); copy(model.values, lp.a_matrix_.value_);
  lp.setMatrixDimensions(); budget.check(); return lp;
}
inline Model own_model(const HighsLp& lp, Budget& budget) {
  budget.check();
  if (lp.sense_ != ObjSense::kMinimize || !lp.a_matrix_.isColwise()) throw NativeError("约化LP方向或矩阵方向无效");
  Model model;
  model.name = lp.model_name_; model.sense = solver::ObjectiveSense::minimize; model.objective_offset = lp.offset_;
  const auto copy = [&](const auto& from, auto& to) {
    to.reserve(from.size()); for (const auto value : from) { budget.tick(); to.push_back(value); }
  };
  copy(lp.col_cost_, model.costs); copy(lp.col_lower_, model.column_lower); copy(lp.col_upper_, model.column_upper);
  copy(lp.row_lower_, model.row_lower); copy(lp.row_upper_, model.row_upper);
  model.column_start.clear();
  copy(lp.a_matrix_.start_, model.column_start); copy(lp.a_matrix_.index_, model.row_index); copy(lp.a_matrix_.value_, model.values);
  model.kinds.reserve(model.costs.size());
  for (std::size_t k = 0; k < model.costs.size(); ++k) { budget.tick(); model.kinds.push_back(solver::VariableKind::continuous); }
  if (model.costs.empty() && model.column_start.empty()) model.column_start.push_back(0);
  budget.check();
  if (!solver::validate_model(model).empty()) throw NativeError("约化LP未通过自主模型表示检查");
  budget.check(); return model;
}
inline Dimensions dimensions(const Model& model) {
  return {static_cast<std::int64_t>(model.costs.size()), static_cast<std::int64_t>(model.row_lower.size()),
      static_cast<std::int64_t>(model.values.size())};
}
inline HighsSolution payload(const Model& reduced, std::vector<double> x, std::vector<double> y, Budget& budget) {
  budget.check();
  if (x.size() != reduced.costs.size() || y.size() != reduced.row_lower.size()) throw NativeError("约化候选维数不完整");
  HighsSolution input;
  input.value_valid = true; input.dual_valid = true;
  input.col_value = std::move(x); input.row_dual = std::move(y);
  for (double value : input.col_value) { budget.tick(); if (!std::isfinite(value)) throw NativeError("约化原始点非有限"); }
  for (double value : input.row_dual) { budget.tick(); if (!std::isfinite(value)) throw NativeError("约化行对偶非有限"); }
  input.col_dual.reserve(reduced.costs.size());
  for (std::size_t col = 0; col < reduced.costs.size(); ++col) {
    budget.tick(); long double value = reduced.costs[col];
    for (std::int64_t k = reduced.column_start[col]; k < reduced.column_start[col + 1]; ++k) {
      budget.tick(); const auto entry = static_cast<std::size_t>(k);
      value -= static_cast<long double>(reduced.values[entry]) * input.row_dual[static_cast<std::size_t>(reduced.row_index[entry])];
      if (!std::isfinite(value)) throw NativeError("约化列对偶累加非有限");
    }
    const double stored = narrow(value);
    if (value != 0 && stored == 0) throw NativeError("约化列对偶转换非零下溢");
    input.col_dual.push_back(stored);
  }
  budget.check(); return input;
}

enum class OracleDecision { accept, reject, budget, error };
struct OwnedOriginalCandidate {
  std::vector<double> x, y;
  solver::LpVerification checked;
  std::int64_t event = 0;
};

// original/options/native/Stats/Budget均由同一controller具名持有并覆盖整个core调用。
// 只保存原坐标的拥有向量，不保存assess局部引用或Highs解缓存引用。
class OriginalOracleContext {
 public:
  OriginalOracleContext(const Model& original, const SolveOptions& options, Clock::time_point start,
      Highs& native, Stats& stats, Budget& budget)
      : original_(original), options_(options), native_(native), stats_(stats), budget_(budget) {
    (void)start;
    budget_.check(); stats_.oracle.events.reserve(204); budget_.check();
  }
  OriginalOracleContext(const OriginalOracleContext&) = delete;
  OriginalOracleContext& operator=(const OriginalOracleContext&) = delete;
  Stats* current_frame = nullptr;
  std::optional<OwnedOriginalCandidate> candidate;
  std::optional<OracleDecision> terminal;

  OracleDecision evaluate(const Model& reduced, const std::vector<double>& x, const std::vector<double>& y,
      std::optional<std::int64_t> iteration) noexcept {
    if (in_call_ || terminal || stats_.oracle.events.size() >= 204) {
      terminal = OracleDecision::error;
      return OracleDecision::error;
    }
    static_assert(std::is_nothrow_move_constructible_v<OracleEvent> && std::is_nothrow_move_constructible_v<OwnedOriginalCandidate> &&
        std::is_nothrow_move_assignable_v<OwnedOriginalCandidate>);
    stats_.oracle.events.emplace_back();
    auto& event = stats_.oracle.events.back();
    event.sequence = stats_.oracle.started++; event.iteration = iteration;
    in_call_ = true;
    Stats frame;
    current_frame = &frame;
    const auto old_phase = stats_.phase; stats_.phase = "oracle";
    const Work before = stats_.budget;
    double begin = 0, end = 0;
    bool have_begin = false, helper_entered = false;
    OracleDecision decision = OracleDecision::error;
    std::optional<OwnedOriginalCandidate> pending;
    try {
      begin = budget_.elapsed(); have_begin = true;
      budget_.check();
      HighsSolution input;
      {
        Stage stage(frame, "candidate", frame.timings.recovery, budget_);
        std::vector<double> copied_x, copied_y;
        copied_x.reserve(x.size()); copied_y.reserve(y.size());
        for (double value : x) { budget_.tick(); copied_x.push_back(value); }
        for (double value : y) { budget_.tick(); copied_y.push_back(value); }
        input = payload(reduced, std::move(copied_x), std::move(copied_y), budget_);
      }
      std::vector<double> original_x, original_y;
      {
        Stage stage(frame, "recovery", frame.timings.recovery, budget_);
        budget_.check(); ++event.postsolve.started; ++stats_.postsolve.started;
        const auto status = native_.postsolve(input);
        ++event.postsolve.completed; ++stats_.postsolve.completed;
        event.postsolve.status = stats_.postsolve.status = static_cast<int>(status);
        observe(native_, stats_); budget_.check();
        if (status == HighsStatus::kError) throw NativeError("原停止oracle无基恢复失败");
        const auto& restored = native_.getSolution();
        if (!restored.value_valid || !restored.dual_valid || restored.col_value.size() != original_.costs.size() ||
            restored.row_dual.size() != original_.row_lower.size()) throw NativeError("原停止oracle恢复维数或有效性无效");
        original_x.reserve(restored.col_value.size()); original_y.reserve(restored.row_dual.size());
        for (double value : restored.col_value) { budget_.tick(); if (!std::isfinite(value)) throw NativeError("oracle原始点非有限"); original_x.push_back(value); }
        for (double value : restored.row_dual) { budget_.tick(); if (!std::isfinite(value)) throw NativeError("oracle原行对偶非有限"); original_y.push_back(value); }
        budget_.check();
      }
      solver::LpVerification checked;
      {
        Stage stage(frame, "verification", frame.timings.verification, budget_);
        budget_.check(); ++frame.assessments.started;
        checked = solver::verify_lp_solution(original_, original_x, original_y, options_.tolerances, options_.pdlp.optimality);
        ++frame.assessments.completed; frame.original_quality = quality(checked.primal, checked.optimality);
        budget_.check();
      }
      helper_entered = true;
      auto trial = bounded_projection(original_, options_, original_x, original_y, checked, frame, budget_);
      if (trial) {
        original_x = std::move(trial->x); original_y = std::move(trial->y); checked = std::move(trial->checked);
        frame.original_quality = frame.projection.trial_quality;
      }
      decision = complete_quality(checked) ? OracleDecision::accept : OracleDecision::reject;
      pending.emplace(OwnedOriginalCandidate{std::move(original_x), std::move(original_y), std::move(checked), event.sequence});
      budget_.check();
      end = budget_.elapsed();
      budget_.check();
    } catch (const Expired&) {
      decision = OracleDecision::budget; terminal = decision;
    } catch (...) {
      decision = OracleDecision::error; terminal = decision;
    }
    // 以下计数/快照均不分配；所有事件保留各阶段partial与已支付成本。
    try { end = budget_.elapsed(); }
    catch (...) { decision = OracleDecision::error; terminal = decision; }
    event.assessments = frame.assessments;
    event.projection = frame.projection;
    event.original_quality = frame.original_quality;
    event.timings.total = have_begin ? std::max(0.0, end - begin) : 0;
    event.timings.recovery = frame.timings.recovery;
    event.timings.verification = frame.timings.verification;
    event.budget = {stats_.budget.checks - before.checks, stats_.budget.loop_units - before.loop_units};
    stats_.assessments.started += frame.assessments.started; stats_.assessments.completed += frame.assessments.completed;
    stats_.timings.recovery += frame.timings.recovery; stats_.timings.verification += frame.timings.verification;
    stats_.oracle.total_seconds += event.timings.total;
    if (helper_entered) { stats_.projection = frame.projection; stats_.oracle.projection_event = event.sequence; }
    if ((decision == OracleDecision::accept || decision == OracleDecision::reject) && pending) {
      candidate = std::move(pending); stats_.oracle.candidate_event = event.sequence;
      stats_.original_quality = frame.original_quality;
    }
    switch (decision) {
      case OracleDecision::accept: event.decision = "accept"; ++stats_.oracle.accepted; break;
      case OracleDecision::reject: event.decision = "reject"; ++stats_.oracle.rejected; break;
      case OracleDecision::budget: event.decision = "budget"; ++stats_.oracle.budget_failures; break;
      case OracleDecision::error: event.decision = "error"; ++stats_.oracle.errors; break;
    }
    ++stats_.oracle.completed;
    current_frame = nullptr; in_call_ = false; stats_.phase = old_phase;
    return decision;
  }
 private:
  const Model& original_;
  const SolveOptions& options_;
  Highs& native_;
  Stats& stats_;
  Budget& budget_;
  bool in_call_ = false;
};
inline thread_local OriginalOracleContext* active_oracle = nullptr;
struct ScopedOracle {
  explicit ScopedOracle(OriginalOracleContext* next) noexcept : previous(active_oracle) { active_oracle = next; }
  ~ScopedOracle() noexcept { active_oracle = previous; }
  ScopedOracle(const ScopedOracle&) = delete;
  ScopedOracle& operator=(const ScopedOracle&) = delete;
  OriginalOracleContext* previous;
};
inline bool oracle_active() noexcept { return active_oracle != nullptr; }
inline OracleDecision consult_current(bool current_local_optimal, const Model& reduced,
    const std::vector<double>& x, const std::vector<double>& y, std::optional<std::int64_t> iteration) noexcept {
  if (!active_oracle || !current_local_optimal) return OracleDecision::reject;
  return active_oracle->evaluate(reduced, x, y, iteration);
}

template<class Core>
SolveResult control(const Model& original, const SolveOptions& options, Clock::time_point start,
                    Stats& stats, Core core, Now now = [] { return Clock::now(); }) {
  stats.enabled = true; stats.attempted = true; stats.status = "running"; stats.phase = "preparation";
  stats.original = dimensions(original);
  Budget budget(start, options.time_limit, stats, std::move(now));
  SolveResult result = [&] {
    Stage stage(stats, "preparation", stats.timings.preparation, budget);
    return make_shell(original);
  }();
  try {
    budget.check();
    Highs native;
    HighsLp minimum;
    {
      Stage stage(stats, "preparation", stats.timings.preparation, budget);
      configure(native, stats, budget, options);
      minimum = minimum_copy(original, budget);
    }
    {
      Stage stage(stats, "pass_model", stats.timings.preparation, budget);
      budget.check(); ++stats.pass_model.started;
      const auto status = native.passModel(std::move(minimum));
      ++stats.pass_model.completed; stats.pass_model.status = static_cast<int>(status); budget.check();
      if (status != HighsStatus::kOk) throw NativeError("最小化副本不能载入原生组件");
    }
    {
      Stage stage(stats, "presolve", stats.timings.presolve, budget);
      const double limit = native.getRunTime() + budget.remaining();
      option(native, "time_limit", limit, budget); stats.native.time_limit = limit;
      budget.check(); ++stats.presolve.started;
      const auto status = native.presolve();
      ++stats.presolve.completed; stats.presolve.status = static_cast<int>(status);
      observe(native, stats); budget.check();
      if (native.getModelPresolveStatus() == HighsPresolveStatus::kTimeout) throw Expired{};
      if (status != HighsStatus::kOk) throw NativeError("原生预处理调用未正常完成");
    }
    const auto presolved = native.getModelPresolveStatus();
    if (presolved == HighsPresolveStatus::kNotReduced) {
      stats.route = "original"; stats.reduced = stats.original;
      ScopedOracle scope(nullptr);
      {
        Stage stage(stats, "child", stats.timings.child, budget);
        budget.check(); ++stats.child.started;
        result = core(original, options, start);
        ++stats.child.completed;
        stats.timings.child_verification = result.verification_seconds;
        stats.child.has_solution = result.has_solution; stats.child.status = result.status;
        budget.check();
      }
      if (result.lp_optimality) stats.original_quality = quality(result.verification, *result.lp_optimality);
      stats.accepted = result.status == "optimal" && result.has_solution && result.lp_optimality && result.lp_optimality->numerically_optimal;
      stats.status = stats.accepted ? "accepted" : "no_candidate";
    } else {
      if (presolved != HighsPresolveStatus::kReduced && presolved != HighsPresolveStatus::kReducedToEmpty)
        throw RejectedStatus("原生预处理状态不用于自主不可行或无界证书");
      Model reduced;
      {
        Stage stage(stats, "preparation", stats.timings.preparation, budget);
        reduced = own_model(native.getPresolvedLp(), budget); stats.reduced = dimensions(reduced);
      }
      OriginalOracleContext oracle(original, options, start, native, stats, budget);
      if (presolved == HighsPresolveStatus::kReducedToEmpty) {
        stats.route = "empty";
        if (!reduced.costs.empty() || !reduced.row_lower.empty()) throw NativeError("空约化状态与实际维数不符");
        oracle.evaluate(reduced, {}, {}, std::nullopt);
      } else {
        stats.route = "reduced";
        SolveResult child;
        {
          // child为包含oracle回调的完整耗时；recovery/verification仅为其中子项。
          Stage stage(stats, "child", stats.timings.child, budget);
          ScopedOracle scope(&oracle);
          budget.check(); ++stats.child.started;
          child = core(reduced, options, start);
          ++stats.child.completed;
          stats.timings.child_verification = child.verification_seconds;
        }
        // 诊断字符串复制前先无抛接管已支付账本，约化坐标绝不作为原坐标输出。
        const bool available = child.has_solution;
        std::string child_status = std::move(child.status); child.status.clear();
        adopt_child(result, std::move(child));
        result.status = std::move(child_status);
        stats.child.has_solution = available; stats.child.status = result.status;
        if (result.status == "optimal") result.status = "verification_failed";
        budget.check();
      }
      if (oracle.terminal == OracleDecision::budget) throw Expired{};
      if (oracle.terminal == OracleDecision::error) throw NativeError("原停止oracle未完整完成");
      if (!oracle.candidate) {
        clear_coordinate_quality(result); stats.status = "no_candidate";
      } else {
        stats.phase = "commit";
        budget.check();
        const auto selected = oracle.candidate->event;
        stats.accepted = commit_original_quality(result, std::move(oracle.candidate->x),
            std::move(oracle.candidate->y), std::move(oracle.candidate->checked), options);
        stats.oracle.selected_event = selected;
        stats.status = stats.accepted ? "accepted" : "checked_not_optimal";
      }
    }
    stats.phase = "finished"; budget.check();
  } catch (const Expired&) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "time_limit"; clear_coordinate_quality(result);
    result.status = "time_limit"; result.message = "预处理与自主内点法的共享时间预算耗尽";
  } catch (const RejectedStatus& error) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "presolve_rejected"; clear_coordinate_quality(result);
    result.status = "error"; result.message = error.what();
  } catch (const NativeError& error) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "native_error"; clear_coordinate_quality(result);
    result.status = "error"; result.message = error.what();
  } catch (const std::exception& error) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "exception"; clear_coordinate_quality(result);
    result.status = "error"; result.message = std::string("预处理组件路径失败：") + error.what();
  } catch (...) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "exception"; clear_coordinate_quality(result);
    result.status = "error"; result.message = "预处理组件路径出现未知异常";
  }
  stats.timings.total = budget.elapsed();
  if (stats.timings.total >= options.time_limit) {
    stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "time_limit"; clear_coordinate_quality(result);
    result.status = "time_limit"; result.message = "预处理组件路径结束时共享预算已耗尽";
  }
  result.backend_status = result.status;
  result.verification_seconds = stats.timings.child_verification + stats.timings.verification;
  result.total_seconds = stats.timings.total;
  result.solve_seconds = std::max(0.0, result.total_seconds - result.verification_seconds);
  return result;
}

inline void quote(std::FILE* stream, std::string_view value) {
  std::fputc('"', stream);
  for (unsigned char ch : value) {
    if (ch == '"' || ch == '\\') { std::fputc('\\', stream); std::fputc(ch, stream); }
    else if (ch < 32) std::fprintf(stream, "\\u%04x", static_cast<unsigned>(ch));
    else std::fputc(ch, stream);
  }
  std::fputc('"', stream);
}
template<class Value> inline void json(std::FILE* stream, const std::optional<Value>& value) {
  if (!value) std::fputs("null", stream);
  else if constexpr (std::is_same_v<Value, bool>) std::fputs(*value ? "true" : "false", stream);
  else if constexpr (std::is_same_v<Value, std::string>) quote(stream, *value);
  else if constexpr (std::is_floating_point_v<Value>) { if (std::isfinite(*value)) std::fprintf(stream, "%.17g", *value); else std::fputs("null", stream); }
  else std::fprintf(stream, "%lld", static_cast<long long>(*value));
}
inline void json_dimensions(std::FILE* stream, const std::optional<Dimensions>& value) {
  if (!value) std::fputs("null", stream);
  else std::fprintf(stream, "{\"n\":%lld,\"m\":%lld,\"nnz\":%lld}", static_cast<long long>(value->n), static_cast<long long>(value->m), static_cast<long long>(value->nnz));
}
inline void json_call(std::FILE* stream, const Call& call) {
  std::fprintf(stream, "{\"started\":%lld,\"completed\":%lld,\"status\":", static_cast<long long>(call.started), static_cast<long long>(call.completed));
  json(stream, call.status); std::fputc('}', stream);
}
inline void json_quality(std::FILE* stream, const std::optional<Quality>& value) {
  if (!value) { std::fputs("null", stream); return; }
  const auto& q = *value;
  std::fprintf(stream, "{\"primal_valid\":%s,\"primal_feasible\":%s,\"objective_consistent\":%s,\"dual_feasible\":%s,\"gap_satisfied\":%s,\"numerically_optimal\":%s,\"objective\":",
      q.primal_valid ? "true" : "false", q.primal_feasible ? "true" : "false", q.objective_consistent ? "true" : "false",
      q.dual_feasible ? "true" : "false", q.gap_satisfied ? "true" : "false", q.numerically_optimal ? "true" : "false");
  json(stream, q.objective); std::fputs(",\"max_row_violation\":", stream); json(stream, q.max_row_violation);
  std::fputs(",\"max_bound_violation\":", stream); json(stream, q.max_bound_violation);
  std::fputs(",\"max_dual_residual\":", stream); json(stream, q.max_dual_residual);
  std::fputs(",\"objective_gap\":", stream); json(stream, q.objective_gap);
  std::fputs(",\"relative_objective_gap\":", stream); json(stream, q.relative_objective_gap); std::fputc('}', stream);
}
inline void json_projection(std::FILE* stream, const Projection& p) {
  std::fprintf(stream, "{\"started\":%lld,\"completed\":%lld,\"status\":", static_cast<long long>(p.started), static_cast<long long>(p.completed));
  quote(stream, p.status); std::fputs(",\"phase\":", stream); quote(stream, p.phase);
  std::fputs(",\"chosen\":", stream); quote(stream, p.chosen);
  std::fprintf(stream, ",\"x_changed\":%lld,\"y_changed\":%lld,\"max_x_change\":", static_cast<long long>(p.x_changed), static_cast<long long>(p.y_changed));
  json(stream, p.max_x_change); std::fputs(",\"max_y_change\":", stream); json(stream, p.max_y_change);
  std::fputs(",\"raw_quality\":", stream); json_quality(stream, p.raw_quality);
  std::fputs(",\"trial_quality\":", stream); json_quality(stream, p.trial_quality); std::fputc('}', stream);
}
inline void json_oracle(std::FILE* stream, const OracleStats& o) {
  std::fprintf(stream, "{\"started\":%lld,\"completed\":%lld,\"accepted\":%lld,\"rejected\":%lld,\"budget_failures\":%lld,\"errors\":%lld,\"candidate_event\":",
      static_cast<long long>(o.started), static_cast<long long>(o.completed), static_cast<long long>(o.accepted),
      static_cast<long long>(o.rejected), static_cast<long long>(o.budget_failures), static_cast<long long>(o.errors));
  json(stream, o.candidate_event); std::fputs(",\"selected_event\":", stream); json(stream, o.selected_event);
  std::fputs(",\"projection_event\":", stream); json(stream, o.projection_event);
  std::fprintf(stream, ",\"total_seconds\":%.17g,\"events\":[", o.total_seconds);
  for (std::size_t index = 0; index < o.events.size(); ++index) {
    if (index) std::fputc(',', stream);
    const auto& e = o.events[index];
    std::fprintf(stream, "{\"sequence\":%lld,\"iteration\":", static_cast<long long>(e.sequence)); json(stream, e.iteration);
    std::fputs(",\"decision\":", stream); quote(stream, e.decision);
    std::fputs(",\"postsolve\":", stream); json_call(stream, e.postsolve);
    std::fprintf(stream, ",\"assessments\":{\"started\":%lld,\"completed\":%lld},\"projection\":",
        static_cast<long long>(e.assessments.started), static_cast<long long>(e.assessments.completed));
    json_projection(stream, e.projection); std::fputs(",\"original_quality\":", stream); json_quality(stream, e.original_quality);
    std::fprintf(stream, ",\"timings\":{\"total\":%.17g,\"recovery\":%.17g,\"verification\":%.17g},\"budget\":{\"checks\":%lld,\"loop_units\":%lld}}",
        e.timings.total, e.timings.recovery, e.timings.verification,
        static_cast<long long>(e.budget.checks), static_cast<long long>(e.budget.loop_units));
  }
  std::fputs("]}", stream);
}
inline void json_center(std::FILE* stream, const CenterDiagnostics& center) {
  std::fprintf(stream, "{\"enabled\":%s,\"started\":%lld,\"completed\":%lld,\"rows_scanned\":%lld,\"inequality_terms\":%lld,\"budget_checks\":%lld,\"last\":",
      center.enabled ? "true" : "false", static_cast<long long>(center.started), static_cast<long long>(center.completed),
      static_cast<long long>(center.rows_scanned), static_cast<long long>(center.inequality_terms), static_cast<long long>(center.budget_checks));
  if (!center.last) { std::fputs("null}", stream); return; }
  const auto& last = *center.last;
  std::fputs("{\"iteration\":", stream); json(stream, last.iteration);
  std::fputs(",\"status\":", stream); quote(stream, last.status);
  std::fputs(",\"alpha\":", stream); json(stream, last.alpha);
  std::fputs(",\"current_c\":", stream); json(stream, last.current_c);
  std::fputs(",\"affine_c\":", stream); json(stream, last.affine_c);
  std::fputs(",\"ratio\":", stream); json(stream, last.ratio);
  std::fputs(",\"sigma\":", stream); json(stream, last.sigma);
  std::fputs(",\"legacy_sigma\":", stream); json(stream, last.legacy_sigma);
  std::fprintf(stream, ",\"negative_predictions\":%lld,\"rows_scanned\":%lld,\"inequality_terms\":%lld,\"scalar_term_complete\":%s,\"sigma_underflow\":%s}}",
      static_cast<long long>(last.negative_predictions), static_cast<long long>(last.rows_scanned), static_cast<long long>(last.inequality_terms),
      last.scalar_term_complete ? "true" : "false", last.sigma_underflow ? "true" : "false");
}
inline void write_diagnostics(std::FILE* stream, const Stats& s) {
  std::fprintf(stream, "{\"schema\":1,\"enabled\":%s,\"attempted\":%s,\"accepted\":%s,\"status\":", s.enabled ? "true" : "false", s.attempted ? "true" : "false", s.accepted ? "true" : "false");
  quote(stream, s.status); std::fputs(",\"phase\":", stream); quote(stream, s.phase);
  std::fputs(",\"route\":", stream); quote(stream, s.route);
  std::fputs(",\"original\":", stream); json_dimensions(stream, s.original);
  std::fputs(",\"reduced\":", stream); json_dimensions(stream, s.reduced);
  std::fputs(",\"pass_model\":", stream); json_call(stream, s.pass_model);
  std::fputs(",\"presolve\":", stream); json_call(stream, s.presolve);
  std::fputs(",\"postsolve\":", stream); json_call(stream, s.postsolve);
  const auto& n = s.native;
  std::fputs(",\"native\":{\"presolve_status\":", stream); json(stream, n.presolve_status);
  std::fputs(",\"model_status\":", stream); json(stream, n.model_status);
  std::fputs(",\"info_valid\":", stream); json(stream, n.info_valid);
  std::fputs(",\"basis_valid\":", stream); json(stream, n.basis_valid);
  std::fputs(",\"simplex_iterations\":", stream); json(stream, n.simplex_iterations);
  std::fputs(",\"ipm_iterations\":", stream); json(stream, n.ipm_iterations);
  std::fputs(",\"pdlp_iterations\":", stream); json(stream, n.pdlp_iterations);
  std::fputs(",\"crossover_iterations\":", stream); json(stream, n.crossover_iterations);
  std::fputs(",\"qp_iterations\":", stream); json(stream, n.qp_iterations);
  std::fputs(",\"time_limit\":", stream); json(stream, n.time_limit);
  std::fputs(",\"options\":", stream);
  if (n.options) std::fprintf(stream, "{\"primal_feasibility_tolerance\":%.17g,\"dual_feasibility_tolerance\":%.17g,\"ipm_optimality_tolerance\":%.17g,\"lp_presolve_requires_basis_postsolve\":%s}",
      n.options->primal_feasibility_tolerance, n.options->dual_feasibility_tolerance, n.options->ipm_optimality_tolerance, n.options->lp_presolve_requires_basis_postsolve ? "true" : "false");
  else std::fputs("null", stream);
  std::fprintf(stream, "},\"child\":{\"started\":%lld,\"completed\":%lld,\"status\":", static_cast<long long>(s.child.started), static_cast<long long>(s.child.completed));
  json(stream, s.child.status); std::fputs(",\"has_solution\":", stream); json(stream, s.child.has_solution);
  std::fprintf(stream, "},\"assessments\":{\"started\":%lld,\"completed\":%lld},\"original_quality\":", static_cast<long long>(s.assessments.started), static_cast<long long>(s.assessments.completed));
  json_quality(stream, s.original_quality);
  std::fputs(",\"projection\":", stream); json_projection(stream, s.projection);
  const auto& t = s.timings;
  std::fprintf(stream, ",\"timings\":{\"preparation\":%.17g,\"presolve\":%.17g,\"child\":%.17g,\"recovery\":%.17g,\"verification\":%.17g,\"child_verification\":%.17g,\"total\":%.17g},\"budget\":{\"checks\":%lld,\"loop_units\":%lld},\"explicit_highs_lp_solve_calls\":0,\"oracle\":",
      t.preparation, t.presolve, t.child, t.recovery, t.verification, t.child_verification, t.total,
      static_cast<long long>(s.budget.checks), static_cast<long long>(s.budget.loop_units));
  json_oracle(stream, s.oracle);
  std::fputs(",\"center_diag\":", stream); json_center(stream, s.center_diag); std::fputc('}', stream);
}
struct Log {
  explicit Log(Stats& value) : stats(value) {}
  ~Log() noexcept { if (emit_diagnostics) { std::fputs("V90_PRESOLVE ", stderr); write_diagnostics(stderr, stats); std::fputc('\n', stderr); } }
  Stats& stats;
};
}  // 命名空间 solver_v90_presolve
'''

WRAPPER = r'''
SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  solver_v90_presolve::Stats stats;
  stats.enabled = solver_v90_presolve::enabled();
  solver_v90_presolve::Log log(stats);
  solver_v90_presolve::ScopedOracle isolation(nullptr);
  solver_v90_presolve::CenterScope center_isolation(nullptr);
  if (!stats.enabled) {
    stats.status = "disabled";
    return v90_core(model, options, start);
  }
  bool bypass = options.time_limit == 0 || options.ipm.max_iterations == 0;
  try { validate_options(options); bypass = bypass || !validate_model(model).empty() || is_mip(model); }
  catch (...) { bypass = true; }
  if (bypass) {
    stats.status = "bypassed"; stats.phase = "bypass";
    return v90_core(model, options, start);
  }
  stats.center_diag.enabled = true;
  solver_v90_presolve::CenterScope center_scope(&stats.center_diag);
  try { return solver_v90_presolve::control(model, options, start, stats, v90_core); }
  catch (...) { stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "exception"; throw; }
}
'''

OPENING = '''SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
'''
CORE_OPENING = '''SolveResult v90_core(const Model& model, const SolveOptions& options, Clock::time_point start) {
'''

CENTER_BEFORE = '''    const double affine_step = homogeneous_step(model, point, tau, kappa, affine, options, start);
    const double remaining = 1 - affine_step;
    const double sigma = remaining * remaining * remaining;
'''
CENTER_AFTER = '''    const double affine_step = homogeneous_step(model, point, tau, kappa, affine, options, start);
    const double remaining = 1 - affine_step;
    double sigma = remaining * remaining * remaining;
    if (solver_v90_presolve::active_center) {
      try {
        sigma = solver_v90_presolve::predicted_center(model.inequality, point.s, point.z, affine.point.s, affine.point.z,
            tau, kappa, affine.scalars.tau, affine.scalars.kappa, affine_step, complementarity, sigma, iteration,
            *solver_v90_presolve::active_center, [&] { check_budget(options, start); });
      } catch (const solver_v90_presolve::CenterError& error) {
        throw NumericalFailure(error.what());
      }
    }
'''

ASSESS_CURRENT_BEFORE = '''  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  bool replace = !result.has_solution || checks.optimality.numerically_optimal;
'''
ASSESS_CURRENT_AFTER = '''  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  // 当前局部质量和坐标在move前咨询；旧incumbent不能触发原模型停止oracle。
  const bool v90_oracle_active = solver_v90_presolve::oracle_active();
  const bool v90_current_local_optimal = feasible && checks.optimality.numerically_optimal;
  const auto v90_oracle_decision = v90_oracle_active
      ? solver_v90_presolve::consult_current(v90_current_local_optimal, result.model, x, y, result.ipm_iterations)
      : solver_v90_presolve::OracleDecision::reject;
  bool replace = !result.has_solution || checks.optimality.numerically_optimal;
'''
ASSESS_STOP_BEFORE = '''  if (result.has_solution && result.lp_optimality && result.lp_optimality->numerically_optimal) {
    result.status = "optimal";
    result.verification_level = "primal_dual_residuals_checked";
    return true;
  }
  return false;
}

Point direction(const ConeModel& model, const Point& point,
'''
ASSESS_STOP_AFTER = '''  if (v90_oracle_active) {
    // 先保留原局部质量和计数，再把回调结果桥接到core已有异常处理。
    if (v90_oracle_decision == solver_v90_presolve::OracleDecision::budget) throw BudgetExpired{};
    if (v90_oracle_decision == solver_v90_presolve::OracleDecision::error) throw NumericalFailure("原模型停止oracle失败");
    if (v90_oracle_decision == solver_v90_presolve::OracleDecision::accept) {
      result.status = "optimal";
      result.verification_level = "primal_dual_residuals_checked";
      return true;
    }
    result.status = "iteration_limit";
    return false;
  }
''' + ASSESS_STOP_BEFORE

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def static_delta_source():
    """静态剥除center增量后恢复整个父v88实现，不执行父模块。"""
    path = Path(__file__).with_name("v88-ipm-instrumentation.py")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_INSTRUMENTATION_SHA256, "冻结v88父模块身份变化")
    tree = ast.parse(raw.decode("utf-8"))
    assignments = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign) and
        len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    require(ast.literal_eval(assignments["COUNT_FIELDS"]) == COUNT_FIELDS, "原37计数合同变化")
    def string(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str): return node.value
        if isinstance(node, ast.Name) and node.id == "INITIALIZATION": return INITIALIZATION
        if isinstance(node, ast.Name) and node.id in assignments: return string(assignments[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add): return string(node.left) + string(node.right)
        raise RuntimeError("父字符串不是允许的静态表达式")
    normalize = lambda value: value.replace("v88", "v90").replace("V88", "V90")
    baseline, wrapper = normalize(string(assignments["HEADER"])), normalize(string(assignments["WRAPPER"]))
    restored, removed = HEADER, []
    for label, first, last, gap in (
        ("center_structs", "struct CenterLast {", "struct Stats {", ""),
        ("center_helper", "struct CenterError :", "class Budget {", "\n"),
        ("center_writer", "inline void json_center(", "inline void write_diagnostics(", ""),
    ):
        require(restored.count(first) == 1 and restored.count(last) == 1, "center逆剥边界不唯一：" + label)
        begin, end = restored.index(first), restored.index(last, restored.index(first))
        restored = restored[:begin] + gap + restored[end:]
        removed.append(label)
    member = "  CenterDiagnostics center_diag;\n"
    require(restored.count(member) == 1, "center成员唯一")
    restored = restored.replace(member, "", 1)
    output = r'''  json_oracle(stream, s.oracle);
  std::fputs(",\"center_diag\":", stream); json_center(stream, s.center_diag); std::fputc('}', stream);
'''
    original_output = "  json_oracle(stream, s.oracle); std::fputc('}', stream);\n"
    require(restored.count(output) == 1, "center输出唯一")
    restored = restored.replace(output, original_output, 1)
    require(restored == baseline, "剥center后HEADER须逐字恢复v88")
    isolation = "  solver_v90_presolve::CenterScope center_isolation(nullptr);\n"
    scope = "  stats.center_diag.enabled = true;\n  solver_v90_presolve::CenterScope center_scope(&stats.center_diag);\n"
    require(WRAPPER.count(isolation) == 1 and WRAPPER.count(scope) == 1 and
        WRAPPER.replace(isolation, "", 1).replace(scope, "", 1) == wrapper,
        "wrapper只增加center激活与隔离，不改父路由")
    require(normalize(string(assignments["ASSESS_CURRENT_AFTER"])) == ASSESS_CURRENT_AFTER and
        normalize(string(assignments["ASSESS_STOP_AFTER"])) == ASSESS_STOP_AFTER, "原停止oracle双hook逐字保留")
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256, removed_blocks=removed,
        parent_header_restored=True, parent_wrapper_restored=True, parent_oracle_hooks_preserved=True)

def transform(source):
    source = source.replace("\r\n", "\n")
    original, changes = source, []
    require("v90-presolve-state.hpp" not in source and "v90_core(" not in source, "拒绝重复插桩")
    def replace(label, before, after):
        nonlocal source
        require(source.count(before) == 1, "锚点缺失或重复：" + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))
    at = source.index(OPENING) + len(OPENING)
    require(source[at:at + len(INITIALIZATION)] == INITIALIZATION, "原初始化计数合同变化")
    replace("private_header", '#include "solver/ipm.hpp"\n', '#include "v90-presolve-state.hpp"\n#include "solver/ipm.hpp"\n')
    replace("shared_start_core", OPENING, CORE_OPENING)
    replace("assess_current_before_move", ASSESS_CURRENT_BEFORE, ASSESS_CURRENT_AFTER)
    replace("assess_current_stop", ASSESS_STOP_BEFORE, ASSESS_STOP_AFTER)
    replace("homogeneous_center_feedback", CENTER_BEFORE, CENTER_AFTER)
    replace("public_wrapper", '\n}  // 命名空间 solver\n', WRAPPER + '\n}  // 命名空间 solver\n')
    restored = source
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆恢复锚点不唯一：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "无法逐字逆恢复原v66")
    return source, changes

def static_check(root):
    source = (Path(root) / "src/ipm.cpp").read_text(encoding="utf-8")
    patched, changes = transform(source)
    require(len(changes) == 6, "变换数量变化")
    for modified in (source.replace(OPENING, "", 1), source + source, patched,
                     source.replace(CENTER_BEFORE, "", 1), source + CENTER_BEFORE):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, "缺失、重复或已变换来源必须拒绝")
    for field in COUNT_FIELDS:
        require(INITIALIZATION.count("result." + field + " = 0;") == 1, "原零计数合同缺失")
    for forbidden in (".run(", ".optimize(", ".crossover("):
        require(forbidden not in HEADER and forbidden not in WRAPPER, "组件禁止调用LP求解入口")
    require(HEADER.count("native_.postsolve(input)") == 1, "只能使用一参恢复")
    require("stats.timings.child_verification + stats.timings.verification" in HEADER and
        "core(reduced, options, start)" in HEADER and "core(original, options, start)" in HEADER, "共享预算与计时合同缺失")
    require(HEADER.count("auto trial = bounded_projection(") == 1 and
        HEADER.count("solver::project_lp_row_dual_domain(original, trial.y)") == 1 and
        HEADER.count("++stats.assessments.started;") == 1 and HEADER.count("++frame.assessments.started;") == 1, "每恢复点仅一次投影和至多两次原验")
    require(ASSESS_CURRENT_AFTER.index("consult_current(") < ASSESS_CURRENT_AFTER.index("bool replace =") and
        "feasible && checks.optimality.numerically_optimal" in ASSESS_CURRENT_AFTER and
        "result.status = \"iteration_limit\";\n    return false;" in ASSESS_STOP_AFTER,
        "必须在move前用当前局部质量咨询，拒绝后不能借旧incumbent停止")
    require("ScopedOracle isolation(nullptr);" in WRAPPER and "ScopedOracle scope(&oracle);" in HEADER and
        "stats.oracle.selected_event.reset();" in HEADER, "TLS隔离或拥有候选事务缺失")
    center = HEADER[HEADER.index("struct CenterError :"):HEADER.index("class Budget {")]
    require(all(fragment not in center for fragment in
        (".solve(", ".multiply(", ".transpose(", ".resize(", ".reserve(", ".push_back(", "Clock::now")),
        "中心反馈不能新增线性求解、矩阵乘积、数组存储或时钟")
    require("double sigma = remaining * remaining * remaining;" in CENTER_AFTER and
        "if (solver_v90_presolve::active_center)" in CENTER_AFTER and
        CENTER_AFTER.count("catch (") == 1 and "catch (const solver_v90_presolve::CenterError& error)" in CENTER_AFTER and
        "(row & 4095) == 0" in center and "const long double ratio = affine_c <= 0 ? 0 : affine_c >= current_c ? 1 : affine_c / current_c;" in center,
        "关闭保原式、只捕CenterError、4096预算与总比截断须明确")
    # 静态只核源码，不编译、不导入运行器、不调用求解或扫描任何模型。
    return dict(passed=True, solver_runs=0, anchors=len(changes), reversible=True,
        source_text_sha256=digest(source), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        count_fields=list(COUNT_FIELDS), shared_start=True, core_body_preserved_except_candidate_hooks=True,
        assess_hook_reversible=True, current_local_saved_before_move=True, centering_hook_reversible=True,
        parent_oracle_hooks_preserved=True, center_only_sigma_change=True, center_no_extra_linear_solve=True,
        center_disabled_preserves_original=True,
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, staticdelta_source=static_delta_source())

if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
