"""v93：只使活动联立策略的能量分母与当前名义正则KKT一致。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = "d8f3a26068dbc730230d3ecabd276eaab3d2b442cc06c9e6a0c41df24cb6bbce"
PARENT_INSTRUMENTATION_SHA256 = "9521e91350bf1082cf8ada80a922f3c36e0ef419f91170a2bf1227ad80733762"
PARENT_COUPLED_KERNEL_SHA256 = "601878a7525336108d126e57ad6fd326e1ab7685eb2dc61421e73a6b666c2017"
INHERITED_CLARIFICATION_SHA256 = "be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8"
ENVIRONMENT = "SOLVER_V93_PRESOLVE"
PREFIX = "V93_PRESOLVE "
COUPLED_PREFIX = "V93_COUPLED "
ENERGY_PREFIX = "V93_ENERGY "
_parent_path = DATA / "v92-ipm-instrumentation.py"
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError("冻结v92插桩来源变化")
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    """只重命名私有包装名；v92纯FG核文件与命名空间必须保持。"""
    replacements = (
        ('solver_v92_', 'solver_v93_'),
        ('v92_core', 'v93_core'),
        ('v92_solve_coupled', 'v93_solve_coupled'),
        ('v92_coupled_enabled', 'v93_coupled_enabled'),
        ('v92_role', 'v93_role'),
        ('v92_iteration', 'v93_iteration'),
        ('V92_PRESOLVE', 'V93_PRESOLVE'),
        ('V92_COUPLED', 'V93_COUPLED'),
        ('v92-presolve-state.hpp', 'v93-presolve-state.hpp'),
        ('v92-coupled-state.hpp', 'v93-coupled-state.hpp'),
    )
    for before, after in replacements:
        text = text.replace(before, after)
    return text


HEADER = renamed(_parent['HEADER'])
WRAPPER = renamed(_parent['WRAPPER'])
OPENING = _parent['OPENING']
CORE_OPENING = renamed(_parent['CORE_OPENING'])
INITIALIZATION = _parent['INITIALIZATION']
COUNT_FIELDS = _parent['COUNT_FIELDS']
BASE_COUPLED_HEADER = renamed(_parent['COUPLED_HEADER'])
ADAPTER = renamed(_parent['ADAPTER'])
ASSESS_CURRENT_BEFORE = _parent['ASSESS_CURRENT_BEFORE']
ASSESS_CURRENT_AFTER = renamed(_parent['ASSESS_CURRENT_AFTER'])
ASSESS_STOP_BEFORE = _parent['ASSESS_STOP_BEFORE']
ASSESS_STOP_AFTER = renamed(_parent['ASSESS_STOP_AFTER'])

ENERGY_STATE = r'''
#include "v93-energy.hpp"

namespace solver_v93_energy_state {
using Index = std::int64_t;
struct Call {
  Index attempt_iteration = 0;
  solver::v93_energy::Work work;
};
struct Log {
  bool enabled = false;
  Index started = 0, completed = 0, primal_terms = 0, row_terms = 0, budget_checks = 0;
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
    log.primal_terms += call.work.primal_terms;
    log.row_terms += call.work.row_terms;
    log.budget_checks += call.work.budget_checks;
  }
};
// 有限LD摘要允许下溢到0，不据摘要0断言精确零；溢出double时保null。
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
  std::fprintf(stream, "{\"schema\":1,\"enabled\":%s,\"started\":%lld,\"completed\":%lld,\"primal_terms\":%lld,\"row_terms\":%lld,\"budget_checks\":%lld,\"last\":",
      log.enabled ? "true" : "false", static_cast<long long>(log.started), static_cast<long long>(log.completed),
      static_cast<long long>(log.primal_terms), static_cast<long long>(log.row_terms), static_cast<long long>(log.budget_checks));
  if (!log.last) { std::fputs("null}", stream); return; }
  const auto& call = *log.last;
  const auto& work = call.work;
  std::fprintf(stream, "{\"attempt_iteration\":%lld,\"status\":\"%s\",\"columns\":%llu,\"rows\":%llu,\"primal_regularization\":",
      static_cast<long long>(call.attempt_iteration), work.status,
      static_cast<unsigned long long>(work.columns), static_cast<unsigned long long>(work.rows));
  number(stream, std::optional<long double>(work.primal_regularization));
  std::fprintf(stream, ",\"primal_terms\":%lld,\"row_terms\":%lld,\"primal_energy\":",
      static_cast<long long>(work.primal_terms), static_cast<long long>(work.row_terms));
  number(stream, std::optional<long double>(work.primal_energy));
  std::fputs(",\"row_energy\":", stream); number(stream, std::optional<long double>(work.row_energy));
  std::fputs(",\"denominator\":", stream); number(stream, work.denominator);
  std::fputs("}}", stream);
}
struct Report {
  Log& log;
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (solver_v93_presolve::emit_diagnostics) {
      std::fputs("V93_ENERGY ", stderr); write(stderr, log); std::fputc('\n', stderr);
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v93_energy_state
'''
COUPLED_HEADER = BASE_COUPLED_HEADER + ENERGY_STATE

ENERGY_ADAPTER = r'''long double v93_energy_denominator(const KktFactor& factor, std::size_t columns,
    const std::vector<double>& constant, const SolveResult& result, const std::function<void()>& budget,
    std::int64_t attempt_iteration) {
  auto* log = solver_v93_energy_state::current();
  if (!log || !solver_v93_coupled_state::enabled() || attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("名义正则能量作用域或本次迭代标签无效");
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.attempt_iteration = attempt_iteration;
  ++log->started;
  solver_v93_energy_state::Collect collect{*log, call};
  // 缺失正则是非法内部输入，不能用默认0掩盖缺失的分解状态。
  const double regularization = result.ipm_primal_regularization.value_or(std::numeric_limits<double>::quiet_NaN());
  try {
    const long double denominator = ::solver::v93_energy::compute(columns, constant, factor.v93_regularized_h(),
        regularization, call.work, budget);
    ++log->completed;
    return denominator;
  } catch (const ::solver::v93_energy::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''
GETTER_BEFORE = '  int regularization_level() const { return regularization_level_; }\n'
GETTER_AFTER = '''  std::span<const double> v93_regularized_h() const noexcept { return regularized_h_; }
''' + GETTER_BEFORE
ENERGY_BEFORE = '      denominator = detail::ipm_homogeneous_scaled_denominator(scaling, constant, budget);\n'
ENERGY_AFTER = '''      denominator = solver_v93_coupled_state::enabled()
          ? v93_energy_denominator(factor, n, constant, result, budget, iteration)
          : detail::ipm_homogeneous_scaled_denominator(scaling, constant, budget);
'''


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    original, changes = source, []
    require('v93_core(' not in source and 'v93_regularized_h' not in source, '拒绝重复插桩')

    def replace(label, before, after):
        nonlocal source
        require(before != after and source.count(before) == 1, '锚点缺失或重复：' + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))

    _, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 15, '冻结v92父锚点数量变化')
    for item in parent_changes:
        replace(item['label'], renamed(item['before']), renamed(item['after']))
    parent_patched = source
    replace('energy_readonly_rows', GETTER_BEFORE, GETTER_AFTER)
    adapter_opening = 'std::vector<double> v93_solve_coupled(const detail::IpmHomogeneousScaling& scaling,\n'
    replace('energy_adapter', adapter_opening, ENERGY_ADAPTER + adapter_opening)
    replace('energy_denominator', ENERGY_BEFORE, ENERGY_AFTER)
    replace('energy_report', '  solver_v93_coupled_state::Report coupled_report(coupled);\n',
        '  solver_v93_coupled_state::Report coupled_report(coupled);\n  solver_v93_energy_state::Log energy;\n  solver_v93_energy_state::Report energy_report(energy);\n')
    replace('energy_isolation', '  solver_v93_coupled_state::Scope coupled_isolation(nullptr);\n',
        '  solver_v93_coupled_state::Scope coupled_isolation(nullptr);\n  solver_v93_energy_state::Scope energy_isolation(nullptr);\n')
    replace('energy_activation', '  solver_v93_coupled_state::Scope coupled_scope(&coupled);\n',
        '  solver_v93_coupled_state::Scope coupled_scope(&coupled);\n  energy.enabled = true;\n  solver_v93_energy_state::Scope energy_scope(&energy);\n')
    restored_parent = source
    for item in reversed(changes[15:]):
        require(restored_parent.count(item['after']) == 1, '能量逆恢复锚不唯一：' + item['label'])
        restored_parent = restored_parent.replace(item['after'], item['before'], 1)
    require(restored_parent == parent_patched, '删除六处能量增量后须逐字回到派生v92父全文')
    restored = restored_parent
    for item in reversed(changes[:15]):
        require(restored.count(item['after']) == 1, '父逆恢复锚不唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == original, '不能逐字恢复原冻结核心')
    return source, changes


def static_delta_source():
    require(HEADER == renamed(_parent['HEADER']) and WRAPPER == renamed(_parent['WRAPPER']), '父21头与原wrapper字面保持')
    require(COUPLED_HEADER == renamed(_parent['COUPLED_HEADER']) + ENERGY_STATE, '父16头只附加独立能量namespace')
    require(ADAPTER == renamed(_parent['ADAPTER']) and ASSESS_CURRENT_AFTER == renamed(_parent['ASSESS_CURRENT_AFTER']) and
        ASSESS_STOP_AFTER == renamed(_parent['ASSESS_STOP_AFTER']), '父严格FG adapter及原q双hook保持')
    require('"v92-coupled.hpp"' in BASE_COUPLED_HEADER and 'solver::v92_coupled::Work' in BASE_COUPLED_HEADER and
        '::solver::v92_coupled::solve' in ADAPTER and '::solver::v93_coupled' not in ADAPTER,
        '冻结纯FG核文件与命名空间禁止整体改名')
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        parent_header_preserved=True, parent_coupled_prefix_preserved=True, parent_adapter_preserved=True,
        parent_oracle_hooks_preserved=True, pure_coupled_kernel_name_preserved=True)


def static_check(root):
    require(hashlib.sha256((DATA / 'v93-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v93计划变化')
    require(hashlib.sha256((DATA / 'v92-pretest-clarification.json').read_bytes()).hexdigest() == INHERITED_CLARIFICATION_SHA256,
        '继承的摘要和partial澄清原件变化')
    require(hashlib.sha256((DATA / 'v92-coupled.hpp').read_bytes()).hexdigest() == PARENT_COUPLED_KERNEL_SHA256,
        '纯FG核必须保持父版本原字节')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 15, '冻结v92静态证明必须仍通过')
    source = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8')
    patched, changes = transform(source)
    require(len(changes) == 21, '名义正则能量总锚点数量变化')
    for modified in (source.replace(OPENING, '', 1), source + source, patched):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复和已变换来源必须拒绝')
    require(GETTER_AFTER.count('return regularized_h_;') == 1 and 'std::span<const double>' in GETTER_AFTER,
        '只加只读实际行对角getter，不加类成员或凝聚primal对角访问')
    require('factor.v93_regularized_h()' in ENERGY_ADAPTER and 'result.ipm_primal_regularization.value_or(' in ENERGY_ADAPTER and
        'catch (const BudgetExpired&)' in ENERGY_ADAPTER and 'call.work.status = "budget";\n    throw;' in ENERGY_ADAPTER,
        '本次已存r/R直接传入，预算保持原类型')
    require('ipm_homogeneous_scaled_denominator(scaling, constant, budget)' in ENERGY_AFTER and
        'solver_v93_coupled_state::enabled()' in ENERGY_AFTER and 'energy_isolation(nullptr)' in patched,
        '仅活动联立策略改变D，关闭路径沿原D与TLS隔离')
    for forbidden in ('.run(', '.optimize(', '.crossover('):
        require(forbidden not in ENERGY_STATE and forbidden not in ENERGY_ADAPTER, '能量适配器不得调用求解入口')
    result = dict(inherited)
    result.update(anchors=21, source_text_sha256=digest(source), patched_text_sha256=digest(patched),
        header_sha256=digest(HEADER), coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        energy_only_denominator_change=True, energy_readonly_rows=True, energy_existing_primal_regularization=True,
        energy_no_old_denominator_rescan=True, energy_extra_anchors=6, energy_parent_restorable=True,
        energy_no_extra_linear_work=True, energy_disabled_preserves_old_path=True,
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX,
        energy_prefix=ENERGY_PREFIX, inherited_coupled_clarification_sha256=INHERITED_CLARIFICATION_SHA256,
        coupled_kernel_sha256=PARENT_COUPLED_KERNEL_SHA256,
        staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
