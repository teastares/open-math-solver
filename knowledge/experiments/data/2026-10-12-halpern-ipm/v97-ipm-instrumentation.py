"""v97只在已有候选真验后扩受控forcing，初始合同与LD消费者保持。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = 'd9badff6f967973e5d5b44ceee2581fa4b3b74d3b1cb42442eb833015b9f9048'
PARENT_INSTRUMENTATION_SHA256 = '21c7582bfbe92e90e14846da328978396b51b9c177273e0b872c1b7919ed3c9e'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V97_PRESOLVE', 'V97_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V97_COUPLED ', 'V97_ENERGY '
RENAMES = (
    ('solver_v96_', 'solver_v97_'), ('v96_core', 'v97_core'),
    ('v96_solve_coupled', 'v97_solve_coupled'), ('v96_coupled_enabled', 'v97_coupled_enabled'),
    ('v96_role', 'v97_role'), ('v96_iteration', 'v97_iteration'),
    ('v96_energy_denominator', 'v97_energy_denominator'),
    ('v96_tolerances_and_rhs_norms', 'v97_tolerances_and_rhs_norms'),
    ('v96_deliver_direction', 'v97_deliver_direction'),
    ('V96_PRESOLVE', 'V97_PRESOLVE'), ('V96_COUPLED', 'V97_COUPLED'), ('V96_ENERGY', 'V97_ENERGY'),
    ('v96-presolve-state.hpp', 'v97-presolve-state.hpp'), ('v96-coupled-state.hpp', 'v97-coupled-state.hpp'),
    ('v96-coupled.hpp', 'v97-coupled.hpp'), ('solver::v96_coupled', 'solver::v97_coupled'),
)
_parent_path = DATA / 'v96-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v96插桩来源变化')
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    for before, after in RENAMES:
        text = text.replace(before, after)
    return text


HEADER, WRAPPER = (renamed(_parent[key]) for key in ('HEADER', 'WRAPPER'))
ENERGY_STATE, ENERGY_ADAPTER = (renamed(_parent[key]) for key in ('ENERGY_STATE', 'ENERGY_ADAPTER'))
OPENING, INITIALIZATION, COUNT_FIELDS = (_parent[key] for key in ('OPENING', 'INITIALIZATION', 'COUNT_FIELDS'))
CORE_OPENING = renamed(_parent['CORE_OPENING'])
WIDE_CHANGES = [{key: item[key] if key == 'label' else renamed(item[key]) for key in ('label', 'before', 'after')}
    for item in _parent['WIDE_CHANGES']]

CANDIDATE_CPP_CHANGES = [
    {'label': 'candidate_forcing_adapter', 'before': r'''HomogeneousDirection v97_deliver_direction(const detail::IpmHomogeneousScaling& scaling,
    std::vector<long double> solution, bool forced_initial, SolveResult& result,
    const std::function<void()>& budget = {}) {
  auto direction = v94_deliver_direction(scaling, std::move(solution), budget);
  // 原pack的最后预算已经正常返回，公共计数才承认一次非strict方向交付。
  if (forced_initial) ++*result.ipm_homogeneous_scaled_inexact_returns;
  return direction;
}

std::vector<long double> v97_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v97_coupled::ProductDouble& multiply,
    const ::solver::v97_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration, bool* forced_initial = nullptr) {
  if (forced_initial) *forced_initial = false;
  auto* log = solver_v97_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 原门槛准备仍只扫一遍完整RHS，准备失败不启动新的方向或forcing决策。
  const auto data = v97_tolerances_and_rhs_norms(scaling, rhs, budget);
  const ::solver::v97_coupled::InitialForcing forcing{data.rhs_norms};
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v97_coupled_state::Collect collect{*log, call};
  const ::solver::v97_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v97_coupled::solve(layout, rhs, data.tolerances, precondition,
        multiply, multiply_wide, call.work, budget, &forcing);
    if (forced_initial) *forced_initial = std::string_view(call.work.status) == "forced_initial";
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v97_coupled::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

''', 'after': r'''HomogeneousDirection v97_deliver_direction(const detail::IpmHomogeneousScaling& scaling,
    std::vector<long double> solution, bool forced, SolveResult& result,
    const std::function<void()>& budget = {}) {
  auto direction = v94_deliver_direction(scaling, std::move(solution), budget);
  // 初始或候选forcing均在原pack的最后预算成功后才记公共inexact。
  if (forced) ++*result.ipm_homogeneous_scaled_inexact_returns;
  return direction;
}

std::vector<long double> v97_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v97_coupled::ProductDouble& multiply,
    const ::solver::v97_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration, bool* forced = nullptr, bool allow_candidates = false) {
  if (forced) *forced = false;
  auto* log = solver_v97_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 原门槛准备仍只扫一遍完整RHS，准备失败不启动新的方向或forcing决策。
  const auto data = v97_tolerances_and_rhs_norms(scaling, rhs, budget);
  const ::solver::v97_coupled::InitialForcing forcing{data.rhs_norms};
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v97_coupled_state::Collect collect{*log, call};
  const ::solver::v97_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v97_coupled::solve(layout, rhs, data.tolerances, precondition,
        multiply, multiply_wide, call.work, budget, &forcing, allow_candidates);
    if (forced) *forced = std::string_view(call.work.status) == "forced_initial" ||
        std::string_view(call.work.status) == "forced_candidate";
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v97_coupled::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''},
    {'label': 'candidate_forcing_production_call', 'before': r'''    bool forced_initial = false;
    auto solution = v97_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v97_role, v97_iteration, &forced_initial);
    return v97_deliver_direction(scaling, std::move(solution), forced_initial, result, budget);
''', 'after': r'''    bool forced = false;
    auto solution = v97_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v97_role, v97_iteration, &forced, true);
    return v97_deliver_direction(scaling, std::move(solution), forced, result, budget);
'''},
]

COUPLED_HEADER_CHANGES = [
    {'before': r'''  Index initial_forcing_checks = 0, initial_forcing_returns = 0;
''', 'after': r'''  Index initial_forcing_checks = 0, initial_forcing_returns = 0;
  Index candidate_forcing_checks = 0, candidate_forcing_returns = 0;
'''},
    {'before': r'''    log.initial_forcing_checks += w.initial_forcing_checks; log.initial_forcing_returns += w.initial_forcing_returns;
''', 'after': r'''    log.initial_forcing_checks += w.initial_forcing_checks; log.initial_forcing_returns += w.initial_forcing_returns;
    log.candidate_forcing_checks += w.candidate_forcing_checks; log.candidate_forcing_returns += w.candidate_forcing_returns;
'''},
    {'before': r'''  std::fprintf(stream, "{\"schema\":2,\"enabled\":%s", log.enabled ? "true" : "false");
''', 'after': r'''  std::fprintf(stream, "{\"schema\":3,\"enabled\":%s", log.enabled ? "true" : "false");
'''},
    {'before': r'''  count("initial_forcing_checks", log.initial_forcing_checks); count("initial_forcing_returns", log.initial_forcing_returns);
''', 'after': r'''  count("initial_forcing_checks", log.initial_forcing_checks); count("initial_forcing_returns", log.initial_forcing_returns);
  count("candidate_forcing_checks", log.candidate_forcing_checks); count("candidate_forcing_returns", log.candidate_forcing_returns);
'''},
    {'before': r'''inline void write(FILE* stream, const Log& log) noexcept {
''', 'after': r'''inline void write_candidate_forcing(FILE* stream,
    const std::optional<::solver::v97_coupled::CandidateForcingWork>& value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  const auto& forcing = *value;
  std::fprintf(stream, "{\"cycle\":%lld,\"krylov_step\":%lld,\"defect_norms\":",
      static_cast<long long>(forcing.cycle), static_cast<long long>(forcing.krylov_step));
  if (!forcing.defect_norms) std::fputs("null", stream);
  else {
    std::fputc('[', stream);
    for (std::size_t block = 0; block < 5; ++block) {
      if (block) std::fputc(',', stream);
      number(stream, std::optional<long double>((*forcing.defect_norms)[block]));
    }
    std::fputc(']', stream);
  }
  std::fputs(",\"strict_measure\":", stream); number(stream, forcing.strict_measure);
  std::fputs(",\"best_measure_before\":", stream); number(stream, forcing.best_measure_before);
  std::fputs(",\"measure\":", stream); number(stream, forcing.measure);
  std::fprintf(stream, ",\"decision\":\"%s\"}", forcing.decision);
}
inline void write(FILE* stream, const Log& log) noexcept {
'''},
    {'before': r'''  write_initial_forcing(stream, w.initial_forcing);
  std::fputs("}}", stream);
''', 'after': r'''  write_initial_forcing(stream, w.initial_forcing);
  std::fputs(",\"candidate_forcing\":", stream);
  write_candidate_forcing(stream, w.candidate_forcing);
  std::fputs("}}", stream);
'''},
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


COUPLED_HEADER = renamed(_parent['COUPLED_HEADER'])
for _change in COUPLED_HEADER_CHANGES:
    require(COUPLED_HEADER.count(_change['before']) >= 1, 'coupled schema3头锚缺失')
    COUPLED_HEADER = COUPLED_HEADER.replace(_change['before'], _change['after'], 1)
ADAPTER = renamed(_parent['ADAPTER']).replace(CANDIDATE_CPP_CHANGES[0]['before'], CANDIDATE_CPP_CHANGES[0]['after'], 1)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    require('v97_core(' not in source, '拒绝重复插桩')
    parent_patched, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 31, '冻结v96的31锚变化')
    patched, changes = source, []
    for item in parent_changes:
        changed = dict(label=item['label'], before=renamed(item['before']), after=renamed(item['after']))
        require(patched.count(changed['before']) == 1, '父重绑定锚缺失或重复：' + changed['label'])
        patched = patched.replace(changed['before'], changed['after'], 1)
        changes.append(changed)
    require(patched == renamed(parent_patched), '应用候选政策前必须精确保父v96白名单改名')
    for item in CANDIDATE_CPP_CHANGES:
        require(patched.count(item['before']) == 1, '候选增量锚缺失或重复：' + item['label'])
        patched = patched.replace(item['before'], item['after'], 1)
        changes.append(dict(item))
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆锚必须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部父与候选锚必须逐字逆恢复冻结核心')
    return patched, changes


def static_delta_source():
    return dict(passed=True, immediate_parent_version=96, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        immediate_v96_cpp_preserved_after_rename=False, parent_cpp_restored_after_two_candidate_hooks=True,
        parent_energy_preserved=True, parent_wide_consumers_preserved=True, parent_off_path_preserved=True,
        coupled_schema=3, coupled_no_inexact_return=False, strict_or_explicit_forced_initial_return=False,
        strict_or_explicit_forced_return=True, candidate_adapter_default_false=True, candidate_production_explicit_true=True,
        candidate_public_after_delivery=True, initial_forcing_rhs_same_scan=True, new_cpp_algorithm_anchors=2,
        scope='只说明两类受控forcing的接入与公共交付边界；实际候选移动和eligible快照由纯核独立审计。')


def static_check(root):
    require(hashlib.sha256((DATA / 'v97-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v97计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '纯v93能量核变化')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 31, '最终v96静态父必须通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    require(len(changes) == 33 and len(CANDIDATE_CPP_CHANGES) == 2, '父31加候选两锚必须为33')
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已经变换来源必须拒绝')
    require('bool* forced = nullptr, bool allow_candidates = false' in ADAPTER and
        'call.work, budget, &forcing, allow_candidates)' in ADAPTER and
        'budget, v97_role, v97_iteration, &forced, true);' in patched,
        '直接helper须默认initial-only，实际调用站必须显式开启候选gate')
    require('"forced_initial" ||' in ADAPTER and '"forced_candidate";' in ADAPTER and
        ADAPTER.index('v94_deliver_direction(scaling, std::move(solution), budget)') <
        ADAPTER.index('++*result.ipm_homogeneous_scaled_inexact_returns'), '两类forced均必须先完成原LD交付末预算')
    require('"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER and
        'factor.v93_regularized_h()' in ENERGY_ADAPTER, '纯能量文件/namespace/getter必须保原名')
    result = {key: value for key, value in inherited.items() if key not in
        ('staticdelta_source', 'inherited_v95_static', 'inherited_v96_static')}
    result.update(anchors=33, parent_anchors=31, candidate_forcing_extra_anchors=2, new_cpp_algorithm_anchors=2,
        reversible=True, shared_start=True, coupled_no_inexact_return=False,
        strict_or_explicit_forced_initial_return=False, strict_or_explicit_forced_return=True,
        candidate_adapter_default_false=True, candidate_production_explicit_true=True, candidate_public_after_delivery=True,
        immediate_v96_cpp_preserved_after_rename=False,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v97-coupled.hpp').read_bytes()).hexdigest(),
        inherited_v96_static=inherited, staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
