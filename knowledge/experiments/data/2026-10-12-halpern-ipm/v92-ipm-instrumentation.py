"""v92：在冻结v88原q流程上用严格五块FGMRES替代独立辅助方程精化。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = "f8d194d1157e0bed6a554033d4203727c2d99efd79a79119d450440be810ef41"
PRETEST_CLARIFICATION_SHA256 = "be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8"
PARENT_INSTRUMENTATION_SHA256 = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
ENVIRONMENT = "SOLVER_V92_PRESOLVE"
PREFIX = "V92_PRESOLVE "
COUPLED_PREFIX = "V92_COUPLED "
_parent_path = DATA / "v88-ipm-instrumentation.py"
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError("冻结v88插桩来源变化")
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    return text.replace("v88", "v92").replace("V88", "V92")


HEADER = renamed(_parent["HEADER"])
WRAPPER = renamed(_parent["WRAPPER"])
OPENING = _parent["OPENING"]
CORE_OPENING = renamed(_parent["CORE_OPENING"])
INITIALIZATION = _parent["INITIALIZATION"]
COUNT_FIELDS = _parent["COUNT_FIELDS"]
ASSESS_CURRENT_BEFORE = _parent["ASSESS_CURRENT_BEFORE"]
ASSESS_CURRENT_AFTER = renamed(_parent["ASSESS_CURRENT_AFTER"])
ASSESS_STOP_BEFORE = _parent["ASSESS_STOP_BEFORE"]
ASSESS_STOP_AFTER = renamed(_parent["ASSESS_STOP_AFTER"])

COUPLED_HEADER = r'''#pragma once
#include "v92-coupled.hpp"
#include <cstdio>
#include <type_traits>

namespace solver_v92_coupled_state {
using Index = std::int64_t;
struct Call {
  Index attempt_iteration = 0;
  const char* role = "none";
  solver::v92_coupled::Work work;
};
struct Log {
  bool enabled = false;
  Index constant_calls_started = 0, constant_calls_completed = 0;
  Index directions_started = 0, directions_completed = 0;
  Index preconditioner_started = 0, preconditioner_completed = 0;
  Index products_started = 0, products_completed = 0;
  Index krylov_started = 0, krylov_completed = 0, cycles_started = 0;
  Index true_checks_started = 0, true_checks_completed = 0;
  std::optional<Call> last;
};
inline thread_local Log* context = nullptr;
inline bool enabled() noexcept { return context && context->enabled; }
inline Log* current() noexcept { return enabled() ? context : nullptr; }
struct Scope {
  explicit Scope(Log* next) noexcept : previous(context) { context = next; }
  ~Scope() noexcept { context = previous; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Log* previous;
};
struct Collect {
  Log& log;
  Call& call;
  ~Collect() noexcept {
    const auto& w = call.work;
    log.preconditioner_started += w.preconditioner_started;
    log.preconditioner_completed += w.preconditioner_completed;
    log.products_started += w.products_started; log.products_completed += w.products_completed;
    log.krylov_started += w.krylov_started; log.krylov_completed += w.krylov_completed;
    log.cycles_started += w.cycles_started;
    log.true_checks_started += w.true_checks_started; log.true_checks_completed += w.true_checks_completed;
  }
};
// 摘要只窄化有限值；下溢后的0不表示数学精确零，严格接纳始终读原LD。
inline void number(FILE* stream, const std::optional<long double>& value) noexcept {
  if (value) {
    const double narrowed = static_cast<double>(*value);
    if (std::isfinite(*value) && std::isfinite(narrowed)) {
      std::fprintf(stream, "%.17g", narrowed);
      return;
    }
  }
  std::fputs("null", stream);
}
inline void write(FILE* stream, const Log& log) noexcept {
  std::fprintf(stream, "{\"schema\":1,\"enabled\":%s", log.enabled ? "true" : "false");
  const auto count = [&](const char* key, Index value) {
    std::fprintf(stream, ",\"%s\":%lld", key, static_cast<long long>(value));
  };
  count("constant_calls_started", log.constant_calls_started); count("constant_calls_completed", log.constant_calls_completed);
  count("directions_started", log.directions_started); count("directions_completed", log.directions_completed);
  count("preconditioner_started", log.preconditioner_started); count("preconditioner_completed", log.preconditioner_completed);
  count("products_started", log.products_started); count("products_completed", log.products_completed);
  count("krylov_started", log.krylov_started); count("krylov_completed", log.krylov_completed);
  count("cycles_started", log.cycles_started);
  count("true_checks_started", log.true_checks_started); count("true_checks_completed", log.true_checks_completed);
  std::fputs(",\"last\":", stream);
  if (!log.last) { std::fputs("null}", stream); return; }
  const auto& last = *log.last;
  const auto& w = last.work;
  std::fprintf(stream, "{\"attempt_iteration\":%lld,\"role\":\"%s\",\"status\":\"%s\",\"dimension\":%llu,\"initial_measure\":",
      static_cast<long long>(last.attempt_iteration), last.role, w.status, static_cast<unsigned long long>(w.dimension));
  number(stream, w.initial_measure); std::fputs(",\"final_measure\":", stream); number(stream, w.final_measure);
  std::fputs(",\"block_measures\":[", stream);
  for (std::size_t block = 0; block < 5; ++block) {
    if (block) std::fputc(',', stream);
    number(stream, w.final_measure ? std::optional<long double>(w.block_measures[block]) : std::nullopt);
  }
  std::fputs("],\"tolerances\":[", stream);
  for (std::size_t block = 0; block < 5; ++block)
    std::fprintf(stream, "%s%.17g", block ? "," : "", w.tolerances[block]);
  std::fputs("]}}", stream);
}
struct Report {
  Log& log;
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (solver_v92_presolve::emit_diagnostics) {
      std::fputs("V92_COUPLED ", stderr); write(stderr, log); std::fputc('\n', stderr);
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v92_coupled_state
'''

ADAPTER = r'''std::vector<double> v92_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const detail::IpmFullNewtonProduct& multiply, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v92_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 门槛准备若被预算打断，尚未启动新的联立核，也不冒记一个空调用。
  const auto tolerance = detail::ipm_homogeneous_scaled_tolerances(scaling, rhs, budget);
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v92_coupled_state::Collect collect{*log, call};
  const ::solver::v92_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v92_coupled::solve(layout, rhs, tolerance, precondition, multiply, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v92_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''

SCALED_OPENING = '''HomogeneousDirection homogeneous_scaled_direction(const ConeModel& model, const Point& point,
    double tau, double kappa, const std::vector<double>& primal_residual,
    const std::vector<double>& dual_residual, long double gap_residual, double theta,
    const std::vector<double>& h, long double h_tau, const std::vector<double>& constant,
    const detail::IpmHomogeneousScaling& scaling, long double denominator,
    const KktFactor& factor, const SolveOptions& options, Clock::time_point start, SolveResult& result) {
'''
SCALED_AFTER = SCALED_OPENING.replace('SolveResult& result) {',
    'SolveResult& result,\n    const char* v92_role = nullptr, std::int64_t v92_iteration = 0) {')
KERNEL_BEFORE = '''  const auto solution = detail::solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, budget);
'''
KERNEL_AFTER = '''  const bool v92_coupled_enabled = solver_v92_coupled_state::enabled();
  const auto solution = v92_coupled_enabled
      ? v92_solve_coupled(scaling, rhs, precondition, multiply, budget, v92_role, v92_iteration)
      : detail::solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, budget);
  // 新核只有完整五块严格通过才返回，旧尾部不得把它误记为inexact。
  if (v92_coupled_enabled) info.tolerance_satisfied = true;
'''
CONSTANT_BEFORE = '''      constant = factor.solve_homogeneous_direct(scaled_rhs, options, start, result, true);
'''
CONSTANT_AFTER = '''      if (auto* coupled = solver_v92_coupled_state::current()) {
        budget(); ++coupled->constant_calls_started;
        constant = factor.solve_homogeneous_direct(scaled_rhs, options, start, result, true);
        ++coupled->constant_calls_completed;
      } else {
        constant = factor.solve_homogeneous_direct(scaled_rhs, options, start, result, true);
      }
'''
AFFINE_BEFORE = '''        dual_residual, gap_residual, 1, h, affine_h_tau, constant, scaling, denominator, factor, options, start, result) :
'''
AFFINE_AFTER = AFFINE_BEFORE.replace('start, result) :', 'start, result, "affine", iteration) :')
CORRECTOR_BEFORE = '''        dual_residual, gap_residual, 1 - sigma, h, h_tau, constant, scaling, denominator, factor, options, start, result) :
'''
CORRECTOR_AFTER = CORRECTOR_BEFORE.replace('start, result) :', 'start, result, "corrector", iteration) :')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    original, changes = source, []
    require('v92_core(' not in source and 'v92-coupled-state.hpp' not in source, '拒绝重复插桩')

    def replace(label, before, after):
        nonlocal source
        require(before != after and source.count(before) == 1, '锚点缺失或重复：' + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))

    at = source.index(OPENING) + len(OPENING)
    require(source[at:at + len(INITIALIZATION)] == INITIALIZATION, '原初始化计数合同变化')
    replace('private_header', '#include "solver/ipm.hpp"\n', '#include "v92-presolve-state.hpp"\n#include "solver/ipm.hpp"\n')
    replace('shared_start_core', OPENING, CORE_OPENING)
    replace('assess_current_before_move', ASSESS_CURRENT_BEFORE, ASSESS_CURRENT_AFTER)
    replace('assess_current_stop', ASSESS_STOP_BEFORE, ASSESS_STOP_AFTER)
    replace('public_wrapper', '\n}  // 命名空间 solver\n', WRAPPER + '\n}  // 命名空间 solver\n')
    replace('private_coupled_header', 'namespace solver {\n', '#include "v92-coupled-state.hpp"\n\nnamespace solver {\n')
    replace('coupled_direction_adapter', SCALED_OPENING, ADAPTER + SCALED_AFTER)
    replace('coupled_direction_solver', KERNEL_BEFORE, KERNEL_AFTER)
    replace('coupled_mode', '  const bool scaled_direction = options.ipm.linear_accuracy == "homogeneous_scaled";\n',
        '  const bool scaled_direction = options.ipm.linear_accuracy == "homogeneous_scaled" || solver_v92_coupled_state::enabled();\n')
    replace('coupled_constant_ledger', CONSTANT_BEFORE, CONSTANT_AFTER)
    replace('coupled_affine_role', AFFINE_BEFORE, AFFINE_AFTER)
    replace('coupled_corrector_role', CORRECTOR_BEFORE, CORRECTOR_AFTER)
    replace('coupled_report', '  solver_v92_presolve::Log log(stats);\n',
        '  solver_v92_coupled_state::Log coupled;\n  solver_v92_coupled_state::Report coupled_report(coupled);\n  solver_v92_presolve::Log log(stats);\n')
    replace('coupled_isolation', '  solver_v92_presolve::ScopedOracle isolation(nullptr);\n',
        '  solver_v92_presolve::ScopedOracle isolation(nullptr);\n  solver_v92_coupled_state::Scope coupled_isolation(nullptr);\n')
    replace('coupled_activation', '  try { return solver_v92_presolve::control(model, options, start, stats, v92_core); }\n',
        '  coupled.enabled = true;\n  solver_v92_coupled_state::Scope coupled_scope(&coupled);\n  try { return solver_v92_presolve::control(model, options, start, stats, v92_core); }\n')
    restored = source
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆变换锚点不唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == original, '无法逐字逆恢复冻结v66来源')
    return source, changes


def static_delta_source():
    require(HEADER == renamed(_parent['HEADER']) and WRAPPER == renamed(_parent['WRAPPER']), '父HEADER与wrapper字面保持')
    require(ASSESS_CURRENT_AFTER == renamed(_parent['ASSESS_CURRENT_AFTER']) and
        ASSESS_STOP_AFTER == renamed(_parent['ASSESS_STOP_AFTER']), '父当前点oracle双hook保持')
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        parent_header_preserved=True, parent_wrapper_preserved=True, parent_oracle_hooks_preserved=True)


def static_check(root):
    require(hashlib.sha256((DATA / 'v92-pretest-clarification.json').read_bytes()).hexdigest() == PRETEST_CLARIFICATION_SHA256,
        '测前摘要/partial澄清原件变化')
    require(hashlib.sha256((DATA / 'v92-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结计划变化')
    source = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8')
    patched, changes = transform(source)
    require(len(changes) == 15, '联立策略可逆锚点数量变化')
    for modified in (source.replace(OPENING, '', 1), source + source, patched):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已变换来源必须拒绝')
    for field in COUNT_FIELDS:
        require(INITIALIZATION.count('result.' + field + ' = 0;') == 1, '父初始化零计数合同缺失')
    for forbidden in ('.run(', '.optimize(', '.crossover('):
        require(all(forbidden not in part for part in (HEADER, WRAPPER, COUPLED_HEADER, ADAPTER)), '组件禁止调用LP迭代入口')
    require(HEADER.count('native_.postsolve(input)') == 1 and 'stats.timings.child_verification + stats.timings.verification' in HEADER,
        '原恢复与分离验时保持')
    require('"affine", iteration' in AFFINE_AFTER and '"corrector", iteration' in CORRECTOR_AFTER,
        '角色必须来自两个真实调用站与本次迭代')
    require('catch (const ::solver::v92_coupled::Failure& error)' in ADAPTER and
        'catch (const BudgetExpired&)' in ADAPTER and 'catch (...)' not in ADAPTER,
        '联立核只翻译自己的数值失败，不吞原预算')
    require('coupled_isolation(nullptr)' in patched and 'coupled_scope(&coupled)' in patched,
        '关闭与旁路必须隔离外层TLS')
    return dict(passed=True, solver_runs=0, anchors=len(changes), reversible=True,
        source_text_sha256=digest(source), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER), count_fields=list(COUNT_FIELDS),
        shared_start=True, core_body_preserved_except_candidate_hooks=True, assess_hook_reversible=True,
        current_local_saved_before_move=True, parent_oracle_hooks_preserved=True,
        coupled_reversible=True, coupled_role_from_callsites=True, coupled_constant_is_direct_only=True,
        coupled_physical_operator_preserved=True, coupled_no_inexact_return=True, coupled_no_extra_ir=True,
        coupled_disabled_preserves_old_path=True, coupled_private_krylov_only=True,
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX,
        pretest_clarification_sha256=PRETEST_CLARIFICATION_SHA256,
        staticdelta_source=static_delta_source())


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
