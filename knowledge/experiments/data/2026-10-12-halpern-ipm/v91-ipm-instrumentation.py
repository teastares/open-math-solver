"""v91：组合冻结v88原模型停止oracle与同Cone有界暖PDHG，只用于机制诊断。"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "55ca339ebb9100e626b62080e95e9cd15ce0c2673b23be8727944920b96b68cd"
PARENT_INSTRUMENTATION_SHA256 = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
ENVIRONMENT = "SOLVER_V91_PRESOLVE"
PREFIX = "V91_PRESOLVE "
WARM_ENVIRONMENT = "SOLVER_V91_WARM_PDHG"
WARM_PREFIX = "V91_WARM "
WARM_KERNEL_SHA256 = "38ee5158a5e78c94152d369db0f9612316a11476004f5fb1996d45351fb5ec30"
PRETEST_CLARIFICATION_SHA256 = "b0cf18b7591c124ce5229c5a4671b0028b059ff6a5617218df736b97d9892906"
WARM_QUALITY_FIELDS = ("completed_steps", "primal_valid", "primal_feasible", "objective_consistent",
    "dual_feasible", "gap_satisfied", "numerically_optimal", "objective", "max_row_violation",
    "max_bound_violation", "max_dual_residual", "objective_gap", "relative_objective_gap")
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

namespace solver_v91_presolve {
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
  const char* value = std::getenv("SOLVER_V91_PRESOLVE");
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
};
struct Expired {};
struct NativeError : std::runtime_error { using std::runtime_error::runtime_error; };
struct RejectedStatus : std::runtime_error { using std::runtime_error::runtime_error; };

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
  std::size_t warm_event_count() const noexcept { return stats_.oracle.events.size(); }
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
  json_oracle(stream, s.oracle); std::fputc('}', stream);
}
struct Log {
  explicit Log(Stats& value) : stats(value) {}
  ~Log() noexcept { if (emit_diagnostics) { std::fputs("V91_PRESOLVE ", stderr); write_diagnostics(stderr, stats); std::fputc('\n', stderr); } }
  Stats& stats;
};
}  // 命名空间 solver_v91_presolve
'''

WARM_HEADER = r'''#pragma once
#include "v84-warm-pdhg.hpp"
#include "v91-presolve-state.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <type_traits>

namespace solver_v91_warm_state {
struct Quality {
  std::int64_t completed_steps = 0;
  bool primal_valid = false, primal_feasible = false, objective_consistent = false;
  std::optional<bool> dual_feasible, gap_satisfied, numerically_optimal;
  std::optional<double> objective, max_row_violation, max_bound_violation, max_dual_residual;
  std::optional<double> objective_gap, relative_objective_gap;
};
struct Log {
  bool enabled = false, attempted = false, accepted = false;
  const char* status = "disabled";
  std::optional<double> tau, kappa;
  solver::v84_warm_pdhg::Result kernel;
  std::optional<Quality> seed_quality, last_quality;
};
inline thread_local Log* active_state = nullptr;
inline thread_local std::optional<bool> enabled_override;

inline std::optional<double> summary(double value) noexcept {
  return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
}
inline std::optional<double> summary(const std::optional<double>& value) noexcept {
  return value ? summary(*value) : std::nullopt;
}
inline void number(FILE* stream, const std::optional<double>& value) noexcept {
  if (value && std::isfinite(*value)) std::fprintf(stream, "%.17g", *value);
  else std::fputs("null", stream);
}
inline void boolean(FILE* stream, const std::optional<bool>& value) noexcept {
  std::fputs(value ? (*value ? "true" : "false") : "null", stream);
}
inline void quality(FILE* stream, const std::optional<Quality>& value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  const auto& q = *value;
  std::fprintf(stream, "{\"completed_steps\":%lld,\"primal_valid\":%s,\"primal_feasible\":%s,"
      "\"objective_consistent\":%s,\"dual_feasible\":", static_cast<long long>(q.completed_steps),
      q.primal_valid ? "true" : "false", q.primal_feasible ? "true" : "false", q.objective_consistent ? "true" : "false");
  boolean(stream, q.dual_feasible); std::fputs(",\"gap_satisfied\":", stream); boolean(stream, q.gap_satisfied);
  std::fputs(",\"numerically_optimal\":", stream); boolean(stream, q.numerically_optimal);
  std::fputs(",\"objective\":", stream); number(stream, q.objective);
  std::fputs(",\"max_row_violation\":", stream); number(stream, q.max_row_violation);
  std::fputs(",\"max_bound_violation\":", stream); number(stream, q.max_bound_violation);
  std::fputs(",\"max_dual_residual\":", stream); number(stream, q.max_dual_residual);
  std::fputs(",\"objective_gap\":", stream); number(stream, q.objective_gap);
  std::fputs(",\"relative_objective_gap\":", stream); number(stream, q.relative_objective_gap);
  std::fputc('}', stream);
}
struct Report {
  Log state;
  Log* previous = nullptr;
  Report() noexcept : previous(active_state) {
    const char* value = std::getenv("SOLVER_V91_WARM_PDHG");
    state.enabled = enabled_override ? *enabled_override : value && std::strcmp(value, "1") == 0;
    state.status = state.enabled ? "not_triggered" : "disabled";
    active_state = &state;
  }
  ~Report() noexcept {
    if (solver_v91_presolve::emit_diagnostics) write(stderr);
    active_state = previous;
  }
  void write(std::FILE* stream) const noexcept {
    const auto& s = state;
    std::fprintf(stream, "V91_WARM {\"schema\":1,\"enabled\":%s,\"attempted\":%s,\"accepted\":%s,\"status\":\"%s\",\"tau\":",
        s.enabled ? "true" : "false", s.attempted ? "true" : "false", s.accepted ? "true" : "false", s.status);
    number(stream, s.tau); std::fputs(",\"kappa\":", stream); number(stream, s.kappa);
    std::fprintf(stream, ",\"kernel_status\":\"%s\",\"kernel_stats\":", s.kernel.status);
    solver::v84_warm_pdhg::write_stats_diagnostics(stream, s.kernel.stats);
    std::fputs(",\"seed_quality\":", stream); quality(stream, s.seed_quality);
    std::fputs(",\"last_quality\":", stream); quality(stream, s.last_quality);
    std::fputs("}\n", stream);
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v91_warm_state
'''

CONTROLLER = r'''struct V91WarmOracleLimit {};
struct V91WarmOracleUnavailable {};
// 只管理尚未由暖owner提交的新accept；事件和candidate索引保持原事实，不回滚账本。
struct V91WarmOracleGuard {
  solver_v91_presolve::OriginalOracleContext* context = solver_v91_presolve::active_oracle;
  std::optional<std::int64_t> previous_candidate;
  bool committed = false;
  V91WarmOracleGuard() noexcept {
    if (context && context->candidate) previous_candidate = context->candidate->event;
  }
  bool abort(solver_v91_presolve::OracleDecision reason) noexcept {
    if (!committed && context && context->candidate &&
        (!previous_candidate || context->candidate->event != *previous_candidate) &&
        solver_v91_presolve::complete_quality(context->candidate->checked)) {
      if (reason == solver_v91_presolve::OracleDecision::budget || !context->terminal) context->terminal = reason;
      return true;
    }
    return false;
  }
  ~V91WarmOracleGuard() noexcept { if (!committed) abort(solver_v91_presolve::OracleDecision::error); }
};

// 仅适配既有原模型验收，不把PDHG残差或状态当作接纳条件。
struct V91WarmAssessment {
  const ConeModel& model;
  const SolveOptions& options;
  Clock::time_point start;
  SolveResult& result;
  solver_v91_warm_state::Log& log;
  SolveResult trial;
  Point candidate;
  bool pending = false;

  V91WarmAssessment(const ConeModel& cone, const SolveOptions& policy, Clock::time_point begin,
      SolveResult& original, solver_v91_warm_state::Log& diagnostic)
      : model(cone), options(policy), start(begin), result(original), log(diagnostic) {
    // 整个恢复只复制一次原模型；各次验收复用同一模型和点缓冲。
    trial.model = result.model;
  }

  static solver_v91_warm_state::Quality quality(const SolveResult& value, std::int64_t completed_steps) noexcept {
    using solver_v91_warm_state::summary;
    solver_v91_warm_state::Quality q;
    q.completed_steps = completed_steps;
    q.primal_valid = value.verification.valid_input;
    q.primal_feasible = value.verification.feasible;
    q.objective_consistent = value.verification.objective_consistent;
    q.objective = summary(value.verification.objective);
    if (value.verification.valid_input) {
      q.max_row_violation = summary(value.verification.max_row_violation);
      q.max_bound_violation = summary(value.verification.max_bound_violation);
    }
    if (value.lp_optimality) {
      q.dual_feasible = value.lp_optimality->dual_feasible;
      q.gap_satisfied = value.lp_optimality->gap_satisfied;
      q.numerically_optimal = value.lp_optimality->numerically_optimal;
      if (value.lp_optimality->valid_input) q.max_dual_residual = summary(value.lp_optimality->max_dual_residual);
      q.objective_gap = summary(value.lp_optimality->objective_gap);
      q.relative_objective_gap = summary(value.lp_optimality->relative_objective_gap);
    }
    return q;
  }

  bool operator()(std::int64_t completed_steps, const std::vector<double>& x, const std::vector<double>& z) {
    pending = false;
    auto* oracle = solver_v91_presolve::active_oracle;
    if (oracle && oracle->warm_event_count() >= 204) throw V91WarmOracleLimit{};
    if (oracle && oracle->terminal) throw V91WarmOracleUnavailable{};
    solver_v91_presolve::clear_coordinate_quality(trial);
    trial.ipm_iterations = result.ipm_iterations;
    trial.verification_seconds = 0;
    bool valid = false;
    try {
      candidate.x = x; candidate.z = z;
      // 先完成本地恢复/verify及可抛提交标签；不得把Cone坐标直接传给原生恢复。
      solver_v91_presolve::ScopedOracle isolated(nullptr);
      valid = assess(model, candidate, options, start, trial);
    } catch (...) {
      result.verification_seconds += trial.verification_seconds;
      throw;
    }
    // 此处只计本地verify已记录部分，原oracle的计时由父controller单列。
    result.verification_seconds += trial.verification_seconds;
    const auto q = quality(trial, completed_steps);
    if (completed_steps == 0) log.seed_quality = q;
    log.last_quality = q;
    if (!valid) return false;
    if (oracle) {
      const auto decision = oracle->evaluate(trial.model, trial.solution, trial.row_dual, trial.ipm_iterations);
      if (decision == solver_v91_presolve::OracleDecision::budget) throw BudgetExpired{};
      if (decision == solver_v91_presolve::OracleDecision::error) throw V91WarmOracleUnavailable{};
      pending = decision == solver_v91_presolve::OracleDecision::accept;
    } else pending = true;
    return pending;
  }

  void commit() noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Verification>);
    static_assert(std::is_nothrow_move_assignable_v<std::optional<LpOptimality>>);
    static_assert(std::is_nothrow_swappable_v<std::string> && std::is_nothrow_swappable_v<std::vector<double>>);
    // 提交前由调用者检查原deadline；这些移动不分配，不混入任何legacy计数或模型。
    result.status.swap(trial.status);
    result.verification_level.swap(trial.verification_level);
    result.has_solution = trial.has_solution;
    result.solution.swap(trial.solution);
    result.row_dual.swap(trial.row_dual);
    result.verification = std::move(trial.verification);
    result.lp_optimality = std::move(trial.lp_optimality);
    result.objective = trial.objective;
    result.best_bound = trial.best_bound;
    result.absolute_gap = trial.absolute_gap;
    result.relative_gap = trial.relative_gap;
  }
};
bool v91_run_warm(const ConeModel& model, const Point& point, double tau, double kappa,
    const SolveOptions& options, Clock::time_point start, SolveResult& result,
    solver_v91_warm_state::Log& log, const ::solver::v84_warm_pdhg::Limits& limits,
    const ::solver::v84_warm_pdhg::Budget& budget) {
  if (!log.enabled || log.attempted) return false;
  auto* oracle = solver_v91_presolve::active_oracle;
  if (oracle && oracle->terminal) { log.status = "oracle_unavailable"; return false; }
  if (oracle && oracle->warm_event_count() >= 204) { log.status = "oracle_limit"; return false; }
  if (budget) budget();
  log.attempted = true; log.status = "running";
  log.tau = solver_v91_warm_state::summary(tau); log.kappa = solver_v91_warm_state::summary(kappa);
  V91WarmOracleGuard guard;
  try {
    V91WarmAssessment assessment(model, options, start, result, log);
    const ::solver::v84_warm_pdhg::View view{
        static_cast<::solver::v84_warm_pdhg::Index>(model.costs.size()),
        static_cast<::solver::v84_warm_pdhg::Index>(model.rhs.size()),
        model.starts, model.rows, model.values, model.costs, model.rhs, model.inequality};
    ::solver::v84_warm_pdhg::run(view, limits, point.x, point.z, tau, budget,
        [&](::solver::v84_warm_pdhg::Index step, const std::vector<double>& x, const std::vector<double>& z) {
          return assessment(step, x, z);
        }, log.kernel);
    log.status = log.kernel.status;
    if (!log.kernel.accepted || !assessment.pending) {
      if (guard.abort(solver_v91_presolve::OracleDecision::error)) log.status = "oracle_accept_aborted";
      return false;
    }
    if (budget) budget();
    assessment.commit();
    guard.committed = true;
    log.accepted = true; log.status = "accepted";
    return true;
  } catch (const BudgetExpired&) {
    guard.abort(solver_v91_presolve::OracleDecision::budget);
    log.status = "budget_exception";
    throw;
  } catch (const V91WarmOracleLimit&) {
    log.status = "oracle_limit";
    return false;
  } catch (const V91WarmOracleUnavailable&) {
    log.status = "oracle_unavailable";
    return false;
  } catch (...) {
    log.status = guard.abort(solver_v91_presolve::OracleDecision::error) ? "oracle_accept_aborted" : "exception";
    return false;
  }
}

bool v91_try_warm(const ConeModel& model, const Point& point, double tau, double kappa,
    const SolveOptions& options, Clock::time_point start, SolveResult& result, solver_v91_warm_state::Log& log) {
  if (!log.enabled || log.attempted) return false;
  check_budget(options, start);
  try {
    const ::solver::v84_warm_pdhg::Limits limits;
    const ::solver::v84_warm_pdhg::Budget budget = [&] { check_budget(options, start); };
    return v91_run_warm(model, point, tau, kappa, options, start, result, log, limits, budget);
  } catch (const BudgetExpired&) { throw; }
  catch (...) { log.status = "exception"; return false; }
}

'''

WRAPPER = r'''
SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  solver_v91_presolve::Stats stats;
  stats.enabled = solver_v91_presolve::enabled();
  solver_v91_presolve::Log log(stats);
  solver_v91_presolve::ScopedOracle isolation(nullptr);
  if (!stats.enabled) {
    stats.status = "disabled";
    return v91_core(model, options, start);
  }
  bool bypass = options.time_limit == 0 || options.ipm.max_iterations == 0;
  try { validate_options(options); bypass = bypass || !validate_model(model).empty() || is_mip(model); }
  catch (...) { bypass = true; }
  if (bypass) {
    stats.status = "bypassed"; stats.phase = "bypass";
    return v91_core(model, options, start);
  }
  try { return solver_v91_presolve::control(model, options, start, stats, v91_core); }
  catch (...) { stats.oracle.selected_event.reset(); stats.accepted = false; stats.status = "exception"; throw; }
}
'''

OPENING = '''SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
'''
CORE_OPENING = '''SolveResult v91_core(const Model& model, const SolveOptions& options, Clock::time_point start) {
'''

ASSESS_CURRENT_BEFORE = '''  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  bool replace = !result.has_solution || checks.optimality.numerically_optimal;
'''
ASSESS_CURRENT_AFTER = '''  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  // 当前局部质量和坐标在move前咨询；旧incumbent不能触发原模型停止oracle。
  const bool v91_oracle_active = solver_v91_presolve::oracle_active();
  const bool v91_current_local_optimal = feasible && checks.optimality.numerically_optimal;
  const auto v91_oracle_decision = v91_oracle_active
      ? solver_v91_presolve::consult_current(v91_current_local_optimal, result.model, x, y, result.ipm_iterations)
      : solver_v91_presolve::OracleDecision::reject;
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
ASSESS_STOP_AFTER = '''  if (v91_oracle_active) {
    // 先保留原局部质量和计数，再把回调结果桥接到core已有异常处理。
    if (v91_oracle_decision == solver_v91_presolve::OracleDecision::budget) throw BudgetExpired{};
    if (v91_oracle_decision == solver_v91_presolve::OracleDecision::error) throw NumericalFailure("原模型停止oracle失败");
    if (v91_oracle_decision == solver_v91_presolve::OracleDecision::accept) {
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
    """父HEADER仅增加只读容量方法；父wrapper由独立warm_report锚增量连接。"""
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
    normalize = lambda value: value.replace("v88", "v91").replace("V88", "V91")
    getter = "  std::size_t warm_event_count() const noexcept { return stats_.oracle.events.size(); }\n"
    require(HEADER.count(getter) == 1 and HEADER.replace(getter, "", 1) == normalize(string(assignments["HEADER"])),
        "父HEADER仅增加无抛只读容量方法")
    require(WRAPPER == normalize(string(assignments["WRAPPER"])), "父wrapper字符串必须逐字保留")
    require(ASSESS_CURRENT_AFTER == normalize(string(assignments["ASSESS_CURRENT_AFTER"])) and
        ASSESS_STOP_AFTER == normalize(string(assignments["ASSESS_STOP_AFTER"])), "父停止oracle双hook逐字保留")
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256, parent_header_restored=True,
        parent_wrapper_preserved=True, parent_oracle_hooks_preserved=True, readonly_capacity_method_only=True)

def transform(source):
    source = source.replace("\r\n", "\n")
    original, changes = source, []
    require("v91-presolve-state.hpp" not in source and "v91_core(" not in source, "拒绝重复插桩")
    def replace(label, before, after):
        nonlocal source
        require(source.count(before) == 1, "锚点缺失或重复：" + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))
    at = source.index(OPENING) + len(OPENING)
    require(source[at:at + len(INITIALIZATION)] == INITIALIZATION, "原初始化计数合同变化")
    replace("private_header", '#include "solver/ipm.hpp"\n', '#include "v91-presolve-state.hpp"\n#include "solver/ipm.hpp"\n')
    replace("shared_start_core", OPENING, CORE_OPENING)
    replace("assess_current_before_move", ASSESS_CURRENT_BEFORE, ASSESS_CURRENT_AFTER)
    replace("assess_current_stop", ASSESS_STOP_BEFORE, ASSESS_STOP_AFTER)
    replace("private_warm_header", "namespace solver {\n", '#include "v91-warm-state.hpp"\n\nnamespace solver {\n')
    replace("warm_assessment_controller", "void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n",
        CONTROLLER + "void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n")
    begin = source.index("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n")
    end = source.index("void validate_options(const SolveOptions& options)", begin)
    before = source[begin:end]
    first = before.index("  for (std::int64_t iteration = 1; iteration <= options.ipm.max_iterations; ++iteration) {\n")
    require(before.endswith("  }\n}\n\n"), "主齐次循环末尾结构变化")
    after = before[:first] + "  try {\n" + before[first:-3]
    after += "  } catch (const NumericalFailure&) {\n"
    after += "    if (!solver_v91_warm_state::active_state || !solver_v91_warm_state::active_state->enabled) throw;\n"
    after += "    if (v91_try_warm(model, point, tau, kappa, options, start, result, *solver_v91_warm_state::active_state)) return;\n"
    after += "    throw;\n  }\n}\n\n"
    replace("main_loop_warm_recovery", before, after)
    replace("public_wrapper", '\n}  // 命名空间 solver\n', WRAPPER + '\n}  // 命名空间 solver\n')
    replace("warm_report", "  solver_v91_presolve::Log log(stats);\n",
        "  solver_v91_warm_state::Report warm_report;\n  solver_v91_presolve::Log log(stats);\n")
    restored = source
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆恢复锚点不唯一：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "无法逐字逆恢复原v66")
    return source, changes

def static_check(root):
    source = (Path(root) / "src/ipm.cpp").read_text(encoding="utf-8")
    patched, changes = transform(source)
    require(len(changes) == 9, "变换数量变化")
    for modified in (source.replace(OPENING, "", 1), source + source, patched,
                     source.replace("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n", "", 1)):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, "缺失、重复或已变换来源必须拒绝")
    for field in COUNT_FIELDS:
        require(INITIALIZATION.count("result." + field + " = 0;") == 1, "原零计数合同缺失")
    for forbidden in (".run(", ".optimize(", ".crossover("):
        require(all(forbidden not in text for text in (HEADER, WRAPPER, WARM_HEADER, CONTROLLER)), "组件禁止调用LP求解入口")
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
    wrapped = next(item for item in changes if item["label"] == "main_loop_warm_recovery")
    original_loop = wrapped["before"][wrapped["before"].index("  for (std::int64_t iteration = 1;"):-3]
    require(original_loop in wrapped["after"] and
        wrapped["after"].index("initialize(model, point, factor, options, start, result);") < wrapped["after"].index("  try {\n") and
        wrapped["after"].count("catch (const NumericalFailure&)") == 1 and "catch (const BudgetExpired&)" not in wrapped["after"],
        "只在原主循环外接NumericalFailure，不修改数学或捕获初始化/预算")
    require(CONTROLLER.count("valid = assess(model, candidate, options, start, trial);") == 1 and
        "solver_v91_presolve::ScopedOracle isolated(nullptr);" in CONTROLLER and
        CONTROLLER.count("oracle->evaluate(trial.model, trial.solution, trial.row_dual, trial.ipm_iterations)") == 1 and
        "trial.ipm_iterations = result.ipm_iterations;" in CONTROLLER,
        "暖点只验当前本地坐标，显式原oracle只能咨询一次并沿最后HSD标签")
    require(CONTROLLER.count("result.verification_seconds += trial.verification_seconds;") == 2 and
        "elapsed(" not in CONTROLLER and "Clock::now" not in CONTROLLER and
        "oracle->warm_event_count() >= 204" in CONTROLLER and "const ::solver::v84_warm_pdhg::Limits limits;" in CONTROLLER,
        "暖验收不双计嵌套计时，保oracle容量与固定默认限制")
    require("guard.abort(solver_v91_presolve::OracleDecision::budget);" in CONTROLLER and
        "guard.abort(solver_v91_presolve::OracleDecision::error)" in CONTROLLER and
        CONTROLLER.index("assessment.commit();") < CONTROLLER.index("guard.committed = true;") and
        "if (solver_v91_presolve::emit_diagnostics) write(stderr);" in WARM_HEADER and "active_state = previous;" in WARM_HEADER,
        "接受后未提交失败必须安全收口，嵌套暖scope与日志开关必须保真")
    kernel_path = Path(__file__).with_name("v84-warm-pdhg.hpp")
    require(hashlib.sha256(kernel_path.read_bytes()).hexdigest() == WARM_KERNEL_SHA256, "冻结v84暖核身份变化")
    clarification = Path(__file__).with_name("v91-pretest-clarification.json")
    require(hashlib.sha256(clarification.read_bytes()).hexdigest() == PRETEST_CLARIFICATION_SHA256, "测前事务澄清身份变化")
    # 静态只核源码，不编译、不导入运行器、不调用求解或扫描任何模型。
    return dict(passed=True, solver_runs=0, anchors=len(changes), reversible=True,
        source_text_sha256=digest(source), patched_text_sha256=digest(patched), header_sha256=digest(HEADER), warm_header_sha256=digest(WARM_HEADER), controller_sha256=digest(CONTROLLER),
        count_fields=list(COUNT_FIELDS), shared_start=True, core_body_preserved_except_candidate_hooks=True,
        assess_hook_reversible=True, current_local_saved_before_move=True, parent_oracle_hooks_preserved=True,
        warm_core_loop_preserved=True, warm_initialization_excluded=True, warm_kernel_preserved=True,
        warm_isolated_local_assessment=True, warm_verification_not_double_counted=True, warm_accept_abort_fail_closed=True,
        warm_kernel_sha256=WARM_KERNEL_SHA256, warm_quality_fields=list(WARM_QUALITY_FIELDS),
        pretest_clarification_sha256=PRETEST_CLARIFICATION_SHA256,
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, staticdelta_source=static_delta_source())

if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
