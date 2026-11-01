"""v84：仅齐次主循环数值失败后，用已提交点做共享截止时间的PDHG恢复。"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "4dfec47350f3b208602f0ec36f5fda8a2888d68f630b7c48620f72c5dd86b6aa"
ENVIRONMENT = "SOLVER_V84_WARM_PDHG"
PREFIX = "V84_WARM "
QUALITY_FIELDS = ("completed_steps", "primal_valid", "primal_feasible", "objective_consistent",
    "dual_feasible", "gap_satisfied", "numerically_optimal", "objective", "max_row_violation",
    "max_bound_violation", "max_dual_residual", "objective_gap", "relative_objective_gap")


HEADER = r'''#pragma once
#include "v84-warm-pdhg.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <type_traits>

namespace solver_v84_warm_state {
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
inline thread_local Log state{};

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
  Report() noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Log>);
    state = Log{};
    const char* value = std::getenv("SOLVER_V84_WARM_PDHG");
    state.enabled = value && std::strcmp(value, "1") == 0;
    state.status = state.enabled ? "not_triggered" : "disabled";
  }
  ~Report() noexcept {
    const auto& s = state;
    std::fprintf(stderr, "V84_WARM {\"schema\":1,\"enabled\":%s,\"attempted\":%s,\"accepted\":%s,\"status\":\"%s\",\"tau\":",
        s.enabled ? "true" : "false", s.attempted ? "true" : "false", s.accepted ? "true" : "false", s.status);
    number(stderr, s.tau); std::fputs(",\"kappa\":", stderr); number(stderr, s.kappa);
    std::fprintf(stderr, ",\"kernel_status\":\"%s\",\"kernel_stats\":", s.kernel.status);
    solver::v84_warm_pdhg::write_stats_diagnostics(stderr, s.kernel.stats);
    std::fputs(",\"seed_quality\":", stderr); quality(stderr, s.seed_quality);
    std::fputs(",\"last_quality\":", stderr); quality(stderr, s.last_quality);
    std::fputs("}\n", stderr);
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v84_warm_state
'''


CONTROLLER = r'''// 仅适配既有原模型验收，不把PDHG残差或状态当作接纳条件。
struct V84WarmAssessment {
  const ConeModel& model;
  const SolveOptions& options;
  Clock::time_point start;
  SolveResult& result;
  solver_v84_warm_state::Log& log;
  SolveResult trial;
  Point candidate;
  bool pending = false;

  V84WarmAssessment(const ConeModel& cone, const SolveOptions& policy, Clock::time_point begin,
      SolveResult& original, solver_v84_warm_state::Log& diagnostic)
      : model(cone), options(policy), start(begin), result(original), log(diagnostic) {
    // 整个恢复只复制一次原模型；各次验收复用同一模型和点缓冲。
    trial.model = result.model;
  }

  static solver_v84_warm_state::Quality quality(const SolveResult& value, std::int64_t completed_steps) noexcept {
    using solver_v84_warm_state::summary;
    solver_v84_warm_state::Quality q;
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
    trial.has_solution = false;
    trial.lp_optimality.reset();
    trial.verification_seconds = 0;
    const auto assessment_start = Clock::now();
    bool valid;
    try {
      candidate.x = x;
      candidate.z = z;
      // s不是PDHG状态；原assess只使用x/z恢复与验证原模型。
      valid = assess(model, candidate, options, start, trial);
    } catch (...) {
      // 覆盖原assess内部抛异常而未写回计时的情形，实际调用时间只累计一次。
      result.verification_seconds += elapsed(assessment_start);
      throw;
    }
    result.verification_seconds += elapsed(assessment_start);
    const auto q = quality(trial, completed_steps);
    if (completed_steps == 0) log.seed_quality = q;
    log.last_quality = q;
    pending = valid;
    return valid;
  }

  void commit() noexcept {
    static_assert(std::is_nothrow_move_assignable_v<Verification>);
    static_assert(std::is_nothrow_move_assignable_v<std::optional<LpOptimality>>);
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

bool v84_try_warm(const ConeModel& model, const Point& point, double tau, double kappa,
    const SolveOptions& options, Clock::time_point start, SolveResult& result,
    solver_v84_warm_state::Log& log) {
  if (!log.enabled) return false;
  // 原预算已过时不记作启动恢复，也不抢走外层time_limit语义。
  check_budget(options, start);
  log.attempted = true;
  log.status = "running";
  log.tau = solver_v84_warm_state::summary(tau);
  log.kappa = solver_v84_warm_state::summary(kappa);
  try {
    V84WarmAssessment assessment(model, options, start, result, log);
    const ::solver::v84_warm_pdhg::View view{
        static_cast<::solver::v84_warm_pdhg::Index>(model.costs.size()),
        static_cast<::solver::v84_warm_pdhg::Index>(model.rhs.size()),
        model.starts, model.rows, model.values, model.costs, model.rhs, model.inequality};
    const ::solver::v84_warm_pdhg::Limits limits;
    const auto budget = [&] { check_budget(options, start); };
    ::solver::v84_warm_pdhg::run(view, limits, point.x, point.z, tau, budget,
        [&](::solver::v84_warm_pdhg::Index step, const std::vector<double>& x, const std::vector<double>& z) {
          return assessment(step, x, z);
        }, log.kernel);
    log.status = log.kernel.status;
    if (!log.kernel.accepted) return false;
    check_budget(options, start);
    if (!assessment.pending) { log.status = "exception"; return false; }
    assessment.commit();
    log.accepted = true;
    log.status = "accepted";
    return true;
  } catch (const BudgetExpired&) {
    log.status = "budget_exception";
    throw;
  } catch (...) {
    // 恢复自己的分配/验收/算术异常不能替换触发它的原NumericalFailure。
    log.status = "exception";
    return false;
  }
}

'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(source):
    source = source.replace("\r\n", "\n")
    require("v84_try_warm" not in source and '"v84-warm-state.hpp"' not in source, "拒绝重复变换")
    original, changes = source, []

    def replace(label, before, after):
        nonlocal source
        require(before != after and source.count(before) == 1, label + " 缺失或重复锚点")
        source = source.replace(before, after, 1)
        changes.append({"label": label, "before": before, "after": after})

    replace("private_header", "namespace solver {\n", '#include "v84-warm-state.hpp"\n\nnamespace solver {\n')
    replace("assessment_controller", "void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n",
        CONTROLLER + "void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n")
    begin = source.index("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n")
    end = source.index("void validate_options(const SolveOptions& options)", begin)
    before = source[begin:end]
    first = before.index("  for (std::int64_t iteration = 1; iteration <= options.ipm.max_iterations; ++iteration) {\n")
    require(before.endswith("  }\n}\n\n"), "主齐次循环末尾结构变化")
    after = before[:first] + "  try {\n" + before[first:-3]
    after += "  } catch (const NumericalFailure&) {\n"
    after += "    if (!solver_v84_warm_state::state.enabled) throw;\n"
    after += "    if (v84_try_warm(model, point, tau, kappa, options, start, result, solver_v84_warm_state::state)) return;\n"
    after += "    throw;\n  }\n}\n\n"
    replace("main_loop_recovery", before, after)
    replace("warm_report", "SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n",
        "SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n  solver_v84_warm_state::Report v84_report;\n")
    restored = source
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆变换锚点不唯一：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "不能逐字恢复v66来源")
    return source, changes


def static_check(root):
    source = (Path(root) / "src/ipm.cpp").read_text(encoding="utf-8")
    patched, changes = transform(source)
    require(len(changes) == 4, "变换数量变化")
    for modified in (source.replace("namespace solver {", "namespace removed {", 1), source + source, patched):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, "缺失、重复或已变换来源必须拒绝")
    wrapped = next(item for item in changes if item["label"] == "main_loop_recovery")
    original_loop = wrapped["before"][wrapped["before"].index("  for (std::int64_t iteration = 1;"):-3]
    require(original_loop in wrapped["after"], "原主循环内容不得改变")
    require(wrapped["after"].index("initialize(model, point, factor, options, start, result);") <
        wrapped["after"].index("  try {\n"), "初始化不得进入恢复捕获")
    require(wrapped["after"].count("catch (const NumericalFailure&)") == 1 and
        "catch (const BudgetExpired&)" not in wrapped["after"], "只能接主循环数值异常")
    return {"passed": True, "solver_runs": 0, "anchors": len(changes), "reversible": True,
        "source_text_sha256": digest(source), "patched_text_sha256": digest(patched), "header_sha256": digest(HEADER),
        "original_loop_unchanged": True, "initialization_outside_recovery": True,
        "plan_sha256": PLAN_SHA256, "environment": ENVIRONMENT, "prefix": PREFIX, "quality_fields": list(QUALITY_FIELDS)}


if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False))
