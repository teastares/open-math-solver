"""v98只切换显式启用的FG搜索尺度，真实strict与forcing验收保持。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = 'a66d1fc30ba2cc8eb2a9a56fe1381801b6f73297fabc9e2cb8bd6ae1e2f219a2'
PARENT_INSTRUMENTATION_SHA256 = '611148324db4fda3c3c33d450094841c889a8ed70643c90e472f5b6f37bbf4be'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V98_PRESOLVE', 'V98_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V98_COUPLED ', 'V98_ENERGY '
RENAMES = (
    ('solver_v97_', 'solver_v98_'), ('v97_core', 'v98_core'),
    ('v97_solve_coupled', 'v98_solve_coupled'), ('v97_coupled_enabled', 'v98_coupled_enabled'),
    ('v97_role', 'v98_role'), ('v97_iteration', 'v98_iteration'),
    ('v97_energy_denominator', 'v98_energy_denominator'),
    ('v97_tolerances_and_rhs_norms', 'v98_tolerances_and_rhs_norms'),
    ('v97_deliver_direction', 'v98_deliver_direction'),
    ('V97_PRESOLVE', 'V98_PRESOLVE'), ('V97_COUPLED', 'V98_COUPLED'), ('V97_ENERGY', 'V98_ENERGY'),
    ('v97-presolve-state.hpp', 'v98-presolve-state.hpp'), ('v97-coupled-state.hpp', 'v98-coupled-state.hpp'),
    ('v97-coupled.hpp', 'v98-coupled.hpp'), ('solver::v97_coupled', 'solver::v98_coupled'),
)
_parent_path = DATA / 'v97-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v97插桩来源变化')
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    for before, after in RENAMES:
        text = text.replace(before, after)
    return text


HEADER, WRAPPER, COUPLED_HEADER = (renamed(_parent[key]) for key in ('HEADER', 'WRAPPER', 'COUPLED_HEADER'))
ENERGY_STATE, ENERGY_ADAPTER = (renamed(_parent[key]) for key in ('ENERGY_STATE', 'ENERGY_ADAPTER'))
OPENING, INITIALIZATION, COUNT_FIELDS = (_parent[key] for key in ('OPENING', 'INITIALIZATION', 'COUNT_FIELDS'))
CORE_OPENING = renamed(_parent['CORE_OPENING'])
WIDE_CHANGES = [{key: item[key] if key == 'label' else renamed(item[key]) for key in ('label', 'before', 'after')}
    for item in _parent['WIDE_CHANGES']]
COUPLED_HEADER_CHANGES = []

WEIGHT_CPP_CHANGES = [
    {'label': 'forcing_weights_adapter', 'before': r'''std::vector<long double> v98_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v98_coupled::ProductDouble& multiply,
    const ::solver::v98_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration, bool* forced = nullptr, bool allow_candidates = false) {
  if (forced) *forced = false;
  auto* log = solver_v98_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 原门槛准备仍只扫一遍完整RHS，准备失败不启动新的方向或forcing决策。
  const auto data = v98_tolerances_and_rhs_norms(scaling, rhs, budget);
  const ::solver::v98_coupled::InitialForcing forcing{data.rhs_norms};
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v98_coupled_state::Collect collect{*log, call};
  const ::solver::v98_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v98_coupled::solve(layout, rhs, data.tolerances, precondition,
        multiply, multiply_wide, call.work, budget, &forcing, allow_candidates);
    if (forced) *forced = std::string_view(call.work.status) == "forced_initial" ||
        std::string_view(call.work.status) == "forced_candidate";
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v98_coupled::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

''', 'after': r'''std::vector<long double> v98_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v98_coupled::ProductDouble& multiply,
    const ::solver::v98_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration, bool* forced = nullptr, bool allow_candidates = false,
    bool forcing_weights = false) {
  if (forced) *forced = false;
  auto* log = solver_v98_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 原门槛准备仍只扫一遍完整RHS，准备失败不启动新的方向或forcing决策。
  const auto data = v98_tolerances_and_rhs_norms(scaling, rhs, budget);
  const ::solver::v98_coupled::InitialForcing forcing{data.rhs_norms};
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v98_coupled_state::Collect collect{*log, call};
  const ::solver::v98_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v98_coupled::solve(layout, rhs, data.tolerances, precondition,
        multiply, multiply_wide, call.work, budget, &forcing, allow_candidates, forcing_weights);
    if (forced) *forced = std::string_view(call.work.status) == "forced_initial" ||
        std::string_view(call.work.status) == "forced_candidate";
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v98_coupled::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''},
    {'label': 'forcing_weights_production_call', 'before': r'''    bool forced = false;
    auto solution = v98_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v98_role, v98_iteration, &forced, true);
    return v98_deliver_direction(scaling, std::move(solution), forced, result, budget);
''', 'after': r'''    bool forced = false;
    auto solution = v98_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v98_role, v98_iteration, &forced, true, true);
    return v98_deliver_direction(scaling, std::move(solution), forced, result, budget);
'''},
]
ADAPTER = renamed(_parent['ADAPTER']).replace(WEIGHT_CPP_CHANGES[0]['before'], WEIGHT_CPP_CHANGES[0]['after'], 1)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    require('v98_core(' not in source, '拒绝重复插桩')
    parent_patched, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 33, '冻结v97的33锚变化')
    patched, changes = source, []
    for item in parent_changes:
        changed = dict(label=item['label'], before=renamed(item['before']), after=renamed(item['after']))
        require(patched.count(changed['before']) == 1, '父重绑定锚缺失或重复：' + changed['label'])
        patched = patched.replace(changed['before'], changed['after'], 1)
        changes.append(changed)
    require(patched == renamed(parent_patched), '应用搜索尺度政策前必须精确保父v97允许改名')
    for item in WEIGHT_CPP_CHANGES:
        require(patched.count(item['before']) == 1, '搜索尺度锚缺失或重复：' + item['label'])
        patched = patched.replace(item['before'], item['after'], 1)
        changes.append(dict(item))
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆锚必须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部35锚须逐字逆恢复冻结核心')
    return patched, changes


def static_delta_source():
    return dict(passed=True, immediate_parent_version=97, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        immediate_v97_cpp_preserved_after_rename=False, parent_cpp_restored_after_two_weight_hooks=True,
        parent_headers_preserved_after_rename=True, parent_energy_preserved=True, parent_wide_consumers_preserved=True,
        parent_off_path_preserved=True, coupled_schema=3, coupled_no_inexact_return=False,
        strict_or_explicit_forced_initial_return=False, strict_or_explicit_forced_return=True,
        forcing_weights_adapter_default_false=True, forcing_weights_production_explicit_true=True,
        candidate_public_after_delivery=True, initial_forcing_rhs_same_scan=True, new_cpp_algorithm_anchors=2,
        scope='只说明搜索尺度私有参数接入；真实门槛、proof和交付合同保持，三处同T数学由纯核及组件独立核验。')


def static_check(root):
    require(hashlib.sha256((DATA / 'v98-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v98计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '纯v93能量核变化')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 33, '最终v97静态父必须通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    require(len(changes) == 35 and len(WEIGHT_CPP_CHANGES) == 2 and COUPLED_HEADER_CHANGES == [], '父33加两锚为35，头无增量')
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已变换来源须拒绝')
    require('bool* forced = nullptr, bool allow_candidates = false,\n    bool forcing_weights = false)' in ADAPTER and
        'call.work, budget, &forcing, allow_candidates, forcing_weights)' in ADAPTER and
        'budget, v98_role, v98_iteration, &forced, true, true);' in patched,
        '直接helper默认保父搜索尺度，production必须显式开启候选及同F搜索尺度')
    require(HEADER == renamed(_parent['HEADER']) and COUPLED_HEADER == renamed(_parent['COUPLED_HEADER']) and
        ENERGY_STATE == renamed(_parent['ENERGY_STATE']) and ENERGY_ADAPTER == renamed(_parent['ENERGY_ADAPTER']),
        'primary、coupled及Energy头只能固定改名，不新增字段或schema')
    require('"forced_initial" ||' in ADAPTER and '"forced_candidate";' in ADAPTER and
        ADAPTER.index('v94_deliver_direction(scaling, std::move(solution), budget)') <
        ADAPTER.index('++*result.ipm_homogeneous_scaled_inexact_returns'), '两类forced继续原LD交付末预算后记公共计数')
    require('"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER and
        'factor.v93_regularized_h()' in ENERGY_ADAPTER, '纯能量文件/namespace/getter保持原名')
    result = {key: value for key, value in inherited.items() if key not in
        ('staticdelta_source', 'inherited_v96_static', 'inherited_v97_static', 'immediate_v96_cpp_preserved_after_rename')}
    result.update(anchors=35, parent_anchors=33, forcing_weights_extra_anchors=2, new_cpp_algorithm_anchors=2,
        reversible=True, shared_start=True, coupled_no_inexact_return=False,
        strict_or_explicit_forced_initial_return=False, strict_or_explicit_forced_return=True,
        forcing_weights_adapter_default_false=True, forcing_weights_production_explicit_true=True,
        parent_headers_preserved_after_rename=True, parent_cpp_restored_after_two_weight_hooks=True,
        candidate_public_after_delivery=True, immediate_v97_cpp_preserved_after_rename=False,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v98-coupled.hpp').read_bytes()).hexdigest(),
        inherited_v97_static=inherited, staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
