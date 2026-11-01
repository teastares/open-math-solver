"""v87：保留原成功点，原验失败后最多一次合法域投影，并保持同一数值截止。"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "a2036d124eb9fb61d49bee940b63883747eb3575cced4dbbcf787ad5823b4df4"
PARENT_INSTRUMENTATION_SHA256 = "589b8b810785b5dd14cfa8707896e2fd3b90def192cf1cc10504a9fe525491e2"
ENVIRONMENT = "SOLVER_V87_PRESOLVE"
PREFIX = "V87_PRESOLVE "
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

namespace solver_v87_presolve {
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
  const char* value = std::getenv("SOLVER_V87_PRESOLVE");
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
      std::vector<double> x, y;
      bool available = false;
      if (presolved == HighsPresolveStatus::kReducedToEmpty) {
        stats.route = "empty";
        if (!reduced.costs.empty() || !reduced.row_lower.empty()) throw NativeError("空约化状态与实际维数不符");
        available = true;
      } else {
        stats.route = "reduced";
        SolveResult child;
        {
          Stage stage(stats, "child", stats.timings.child, budget);
          budget.check(); ++stats.child.started;
          child = core(reduced, options, start);
          ++stats.child.completed;
          stats.timings.child_verification = child.verification_seconds;
        }
        // 先保全部旧账本并移走约化坐标质量；预算失败也不能丢掉已支付的core工作。
        available = child.has_solution;
        stats.child.has_solution = available;
        x = std::move(child.solution); y = std::move(child.row_dual);
        std::string child_status = std::move(child.status); child.status.clear();
        adopt_child(result, std::move(child));
        result.status = std::move(child_status);
        stats.child.status = result.status;
        if (result.status == "optimal") result.status = "verification_failed";
        budget.check();
      }
      if (!available) {
        clear_coordinate_quality(result); stats.status = "no_candidate";
      } else {
        HighsSolution input;
        {
          Stage stage(stats, "candidate", stats.timings.recovery, budget);
          input = payload(reduced, std::move(x), std::move(y), budget);
        }
        std::vector<double> restored_x, restored_y;
        {
          Stage stage(stats, "recovery", stats.timings.recovery, budget);
          budget.check(); ++stats.postsolve.started;
          // 唯一恢复入口的局部basis确实为空；返回的无效basis数组可以被上游resize。
          const auto status = native.postsolve(input);
          ++stats.postsolve.completed; stats.postsolve.status = static_cast<int>(status);
          observe(native, stats); budget.check();
          if (status == HighsStatus::kError) throw NativeError("纯无基恢复被原生组件拒绝");
          const auto& restored = native.getSolution();
          if (!restored.value_valid || !restored.dual_valid || restored.col_value.size() != original.costs.size() ||
              restored.row_dual.size() != original.row_lower.size()) throw NativeError("原始坐标恢复维数或有效性不完整");
          restored_x.reserve(restored.col_value.size()); restored_y.reserve(restored.row_dual.size());
          for (double value : restored.col_value) { budget.tick(); if (!std::isfinite(value)) throw NativeError("恢复原始点非有限"); restored_x.push_back(value); }
          for (double value : restored.row_dual) { budget.tick(); if (!std::isfinite(value)) throw NativeError("恢复原行对偶非有限"); restored_y.push_back(value); }
          budget.check();
        }
        solver::LpVerification checked;
        {
          Stage stage(stats, "verification", stats.timings.verification, budget);
          budget.check(); ++stats.assessments.started;
          // y已针对最小化副本归一化，原检查器内部使用sense*c，因此不再翻转。
          checked = solver::verify_lp_solution(original, restored_x, restored_y, options.tolerances, options.pdlp.optimality);
          ++stats.assessments.completed; stats.original_quality = quality(checked.primal, checked.optimality);
          budget.check();
        }
        auto trial = bounded_projection(original, options, restored_x, restored_y, checked, stats, budget);
        if (trial) {
          restored_x = std::move(trial->x); restored_y = std::move(trial->y); checked = std::move(trial->checked);
          stats.original_quality = stats.projection.trial_quality;
        }
        stats.phase = "commit";
        budget.check();
        stats.accepted = commit_original_quality(result, std::move(restored_x), std::move(restored_y), std::move(checked), options);
        stats.status = stats.accepted ? "accepted" : "checked_not_optimal";
      }
    }
    stats.phase = "finished"; budget.check();
  } catch (const Expired&) {
    stats.accepted = false; stats.status = "time_limit"; clear_coordinate_quality(result);
    result.status = "time_limit"; result.message = "预处理与自主内点法的共享时间预算耗尽";
  } catch (const RejectedStatus& error) {
    stats.accepted = false; stats.status = "presolve_rejected"; clear_coordinate_quality(result);
    result.status = "error"; result.message = error.what();
  } catch (const NativeError& error) {
    stats.accepted = false; stats.status = "native_error"; clear_coordinate_quality(result);
    result.status = "error"; result.message = error.what();
  } catch (const std::exception& error) {
    stats.accepted = false; stats.status = "exception"; clear_coordinate_quality(result);
    result.status = "error"; result.message = std::string("预处理组件路径失败：") + error.what();
  } catch (...) {
    stats.accepted = false; stats.status = "exception"; clear_coordinate_quality(result);
    result.status = "error"; result.message = "预处理组件路径出现未知异常";
  }
  stats.timings.total = budget.elapsed();
  if (stats.timings.total >= options.time_limit) {
    stats.accepted = false; stats.status = "time_limit"; clear_coordinate_quality(result);
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
  std::fprintf(stream, ",\"timings\":{\"preparation\":%.17g,\"presolve\":%.17g,\"child\":%.17g,\"recovery\":%.17g,\"verification\":%.17g,\"child_verification\":%.17g,\"total\":%.17g},\"budget\":{\"checks\":%lld,\"loop_units\":%lld},\"explicit_highs_lp_solve_calls\":0}",
      t.preparation, t.presolve, t.child, t.recovery, t.verification, t.child_verification, t.total,
      static_cast<long long>(s.budget.checks), static_cast<long long>(s.budget.loop_units));
}
struct Log {
  explicit Log(Stats& value) : stats(value) {}
  ~Log() noexcept { if (emit_diagnostics) { std::fputs("V87_PRESOLVE ", stderr); write_diagnostics(stderr, stats); std::fputc('\n', stderr); } }
  Stats& stats;
};
}  // 命名空间 solver_v87_presolve
'''

WRAPPER = r'''
SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  solver_v87_presolve::Stats stats;
  stats.enabled = solver_v87_presolve::enabled();
  solver_v87_presolve::Log log(stats);
  if (!stats.enabled) {
    stats.status = "disabled";
    return v87_core(model, options, start);
  }
  bool bypass = options.time_limit == 0 || options.ipm.max_iterations == 0;
  try { validate_options(options); bypass = bypass || !validate_model(model).empty() || is_mip(model); }
  catch (...) { bypass = true; }
  if (bypass) {
    stats.status = "bypassed"; stats.phase = "bypass";
    return v87_core(model, options, start);
  }
  try { return solver_v87_presolve::control(model, options, start, stats, v87_core); }
  catch (...) { stats.status = "exception"; throw; }
}
'''

OPENING = '''SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
'''
CORE_OPENING = '''SolveResult v87_core(const Model& model, const SolveOptions& options, Clock::time_point start) {
'''

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def static_delta_source():
    """只折叠冻结父模块的字符串表达式，不导入或执行其Python代码。"""
    path = Path(__file__).with_name("v86-ipm-instrumentation.py")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_INSTRUMENTATION_SHA256, "冻结v86父模块身份变化")
    tree = ast.parse(raw.decode("utf-8"))
    assignments = {node.targets[0].id: node.value for node in tree.body if isinstance(node, ast.Assign) and
        len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
    require(ast.literal_eval(assignments["COUNT_FIELDS"]) == COUNT_FIELDS, "原37计数合同变化")
    def string(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str): return node.value
        if isinstance(node, ast.Name) and node.id == "INITIALIZATION": return INITIALIZATION
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add): return string(node.left) + string(node.right)
        raise RuntimeError("父HEADER不是允许的静态字符串表达式")
    baseline = string(assignments["HEADER"]).replace("v86", "v87").replace("V86", "V87")
    require(string(assignments["WRAPPER"]).replace("v86", "v87").replace("V86", "V87") == WRAPPER,
        "公开wrapper除版本名称外必须保持父实现")
    restored = HEADER
    blocks = (
        ("projection_struct", "struct Projection {", "struct Timings {"),
        ("projection_helper", "inline bool complete_quality(", "template<class Value> inline void option("),
        ("projection_call", "        auto trial = bounded_projection(", '        stats.phase = "commit";'),
        ("projection_writer", "inline void json_projection(", "inline void write_diagnostics("),
    )
    removed = []
    for label, first, last in blocks:
        require(restored.count(first) == 1 and restored.count(last) == 1, "父差异边界不唯一：" + label)
        begin = restored.index(first)
        end = restored.index(last, begin)
        restored = restored[:begin] + restored[end:]
        removed.append(label)
    for label, fragment in (
        ("projection_member", "  Projection projection;\n"),
        ("projection_output", '  std::fputs(",\\\"projection\\\":", stream); json_projection(stream, s.projection);\n'),
    ):
        require(restored.count(fragment) == 1, "父差异片段不唯一：" + label)
        restored = restored.replace(fragment, "", 1)
        removed.append(label)
    require(restored == baseline, "除投影增量外的原生分支或原质量规则发生变化")
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256, removed_blocks=removed,
        parent_header_restored=True, parent_wrapper_preserved=True)

def transform(source):
    source = source.replace("\r\n", "\n")
    original, changes = source, []
    require("v87-presolve-state.hpp" not in source and "v87_core(" not in source, "拒绝重复插桩")
    def replace(label, before, after):
        nonlocal source
        require(source.count(before) == 1, "锚点缺失或重复：" + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))
    at = source.index(OPENING) + len(OPENING)
    require(source[at:at + len(INITIALIZATION)] == INITIALIZATION, "原初始化计数合同变化")
    replace("private_header", '#include "solver/ipm.hpp"\n', '#include "v87-presolve-state.hpp"\n#include "solver/ipm.hpp"\n')
    replace("shared_start_core", OPENING, CORE_OPENING)
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
    require(len(changes) == 3, "变换数量变化")
    for modified in (source.replace(OPENING, "", 1), source + source, patched):
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
    require(HEADER.count("native.postsolve(input)") == 1, "只能使用一参恢复")
    require("stats.timings.child_verification + stats.timings.verification" in HEADER and
        "core(reduced, options, start)" in HEADER and "core(original, options, start)" in HEADER, "共享预算与计时合同缺失")
    require(HEADER.count("auto trial = bounded_projection(") == 1 and
        HEADER.count("solver::project_lp_row_dual_domain(original, trial.y)") == 1 and
        HEADER.count("++stats.assessments.started;") == 2, "每恢复点仅一次投影和至多两次原验")
    # 静态只核源码，不编译、不导入运行器、不调用求解或扫描任何模型。
    return dict(passed=True, solver_runs=0, anchors=len(changes), reversible=True,
        source_text_sha256=digest(source), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        count_fields=list(COUNT_FIELDS), shared_start=True, original_core_body_preserved=True,
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, staticdelta_source=static_delta_source())

if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
