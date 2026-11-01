"""v100仅复用同一HSD因子对象已有的成功正则层级作为下轮起点。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = '6332b0c43bbf488ef4171ec3d67cb152db70f074a89a34f2a61ffb0895a57181'
PARENT_INSTRUMENTATION_SHA256 = 'd63a7ff4eb3284ed644b2a766666a47e393b61aa4f1456ae4398ab3cb1f746be'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V100_PRESOLVE', 'V100_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V100_COUPLED ', 'V100_ENERGY '
RENAMES = (
    ('solver_v98_', 'solver_v100_'), ('v98_core', 'v100_core'),
    ('v98_solve_coupled', 'v100_solve_coupled'), ('v98_coupled_enabled', 'v100_coupled_enabled'),
    ('v98_role', 'v100_role'), ('v98_iteration', 'v100_iteration'),
    ('v98_energy_denominator', 'v100_energy_denominator'),
    ('v98_tolerances_and_rhs_norms', 'v100_tolerances_and_rhs_norms'),
    ('v98_deliver_direction', 'v100_deliver_direction'),
    ('V98_PRESOLVE', 'V100_PRESOLVE'), ('V98_COUPLED', 'V100_COUPLED'), ('V98_ENERGY', 'V100_ENERGY'),
    ('v98-presolve-state.hpp', 'v100-presolve-state.hpp'), ('v98-coupled-state.hpp', 'v100-coupled-state.hpp'),
    ('v98-coupled.hpp', 'v100-coupled.hpp'), ('solver::v98_coupled', 'solver::v100_coupled'),
)
_parent_path = DATA / 'v98-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v98插桩来源变化')
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    for before, after in RENAMES:
        text = text.replace(before, after)
    return text


HEADER, WRAPPER, COUPLED_HEADER, ADAPTER = (renamed(_parent[key]) for key in ('HEADER', 'WRAPPER', 'COUPLED_HEADER', 'ADAPTER'))
ENERGY_STATE, ENERGY_ADAPTER = (renamed(_parent[key]) for key in ('ENERGY_STATE', 'ENERGY_ADAPTER'))
OPENING, INITIALIZATION, COUNT_FIELDS = (_parent[key] for key in ('OPENING', 'INITIALIZATION', 'COUNT_FIELDS'))
CORE_OPENING = renamed(_parent['CORE_OPENING'])
WIDE_CHANGES = [{key: item[key] if key == 'label' else renamed(item[key]) for key in ('label', 'before', 'after')}
    for item in _parent['WIDE_CHANGES']]
COUPLED_HEADER_CHANGES = []
REGULARIZATION_CPP_CHANGES = [
    {'label': 'warm_regularization_seed', 'before': r'''bool assess(const ConeModel& model, const Point& point, const SolveOptions& options,
               Clock::time_point solve_start, SolveResult& result) {''', 'after': r'''int v100_regularization_seed(const KktFactor& factor) {
  return solver_v100_coupled_state::enabled() ? factor.regularization_level() : 0;
}

bool assess(const ConeModel& model, const Point& point, const SolveOptions& options,
               Clock::time_point solve_start, SolveResult& result) {'''},
    {'label': 'warm_regularization_hsd_factor', 'before': r'''    const double mu = finite(complementarity / (static_cast<long double>(model.barrier_rows) + 1));
    factor.factor(diagonal, options, start, result);
''', 'after': r'''    const double mu = finite(complementarity / (static_cast<long double>(model.barrier_rows) + 1));
    factor.factor(diagonal, options, start, result, v100_regularization_seed(factor));
'''},
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    require('v100_core(' not in source, '拒绝重复插桩')
    parent_patched, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 35, '冻结v98须保35锚')
    patched, changes = source, []
    for item in parent_changes:
        changed = dict(label=item['label'], before=renamed(item['before']), after=renamed(item['after']))
        require(patched.count(changed['before']) == 1, '父重绑定锚缺失或重复：' + changed['label'])
        patched = patched.replace(changed['before'], changed['after'], 1)
        changes.append(changed)
    require(patched == renamed(parent_patched), '新政策前须逐字保持父35锚允许改名')
    for item in REGULARIZATION_CPP_CHANGES:
        require(patched.count(item['before']) == 1, '正则起点锚缺失或重复：' + item['label'])
        patched = patched.replace(item['before'], item['after'], 1)
        changes.append(dict(item))
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆锚须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部37锚须逐字逆恢复冻结核心')
    return patched, changes


def static_delta_source():
    return dict(passed=True, immediate_parent_version=98, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        immediate_v98_cpp_preserved_after_rename=False, parent_cpp_restored_after_two_regularization_hooks=True,
        parent_headers_preserved_after_rename=True, parent_energy_preserved=True, parent_wide_consumers_preserved=True,
        pure_coupled_parent_preserved_after_rename=True, regularization_seed_scope_only=True,
        regularization_seed_no_state_or_checks=True, regularization_seed_hsd_only=True,
        regularization_seed_off_zero=True, regularization_seed_initialization_kept=True,
        coupled_schema=3, coupled_no_inexact_return=False, strict_or_explicit_forced_return=True,
        new_cpp_algorithm_anchors=2,
        scope='仅活动HSD外循环复用已有level；初始化和其它caller不改，旧成功不是新矩阵成功保证，不证明总成本改善。')


def static_check(root):
    require(hashlib.sha256((DATA / 'v100-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v100计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '纯v93能量核变化')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 35, '最终v98静态父须通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    require(len(changes) == 37 and len(REGULARIZATION_CPP_CHANGES) == 2 and COUPLED_HEADER_CHANGES == [],
        '父35加两个正则起点锚为37，头无增量')
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已变换来源须拒绝')
    helper = 'int v100_regularization_seed(const KktFactor& factor) {\n  return solver_v100_coupled_state::enabled() ? factor.regularization_level() : 0;\n}\n\n'
    require(REGULARIZATION_CPP_CHANGES[0]['after'] == helper + REGULARIZATION_CPP_CHANGES[0]['before'] and
        REGULARIZATION_CPP_CHANGES[1]['after'] == REGULARIZATION_CPP_CHANGES[1]['before'].replace(
            'factor.factor(diagonal, options, start, result);',
            'factor.factor(diagonal, options, start, result, v100_regularization_seed(factor));') and
        patched.count('int v100_regularization_seed(') == 1 and
        patched.count('factor.factor(diagonal, options, start, result, v100_regularization_seed(factor));') == 1,
        '只准无状态薄helper及HSD唯一caller第五参，无额外检查或扫描')
    parent_patched, _ = _parent['transform'](original)
    restored_parent = patched
    for item in reversed(REGULARIZATION_CPP_CHANGES):
        require(restored_parent.count(item['after']) == 1, '新两锚逆恢复须唯一')
        restored_parent = restored_parent.replace(item['after'], item['before'], 1)
    require(restored_parent == renamed(parent_patched), '去除新两锚须恢复完整父CPP，包括初始化和factor内部trial')
    require(HEADER == renamed(_parent['HEADER']) and COUPLED_HEADER == renamed(_parent['COUPLED_HEADER']) and
        ADAPTER == renamed(_parent['ADAPTER']) and ENERGY_STATE == renamed(_parent['ENERGY_STATE']) and
        ENERGY_ADAPTER == renamed(_parent['ENERGY_ADAPTER']), '所有父头及方向适配器仅固定改名')
    parent_kernel = (DATA / 'v98-coupled.hpp').read_text(encoding='utf-8')
    child_kernel = (DATA / 'v100-coupled.hpp').read_text(encoding='utf-8')
    require(child_kernel == parent_kernel.replace('solver::v98_coupled', 'solver::v100_coupled') and
        'solver_v99_profile' not in patched + child_kernel, '纯FG核仅namespace重绑，不混入计时诊断')
    require('"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER and
        'factor.v93_regularized_h()' in ENERGY_ADAPTER, '纯能量文件/namespace/getter保持原名')
    result = {key: value for key, value in inherited.items() if key not in
        ('staticdelta_source', 'inherited_v97_static', 'inherited_v98_static', 'immediate_v97_cpp_preserved_after_rename',
         'core_body_preserved_except_candidate_hooks', 'parent_cpp_restored_after_two_weight_hooks')}
    result.update(anchors=37, parent_anchors=35, regularization_seed_extra_anchors=2, new_cpp_algorithm_anchors=2,
        reversible=True, shared_start=True, regularization_seed_scope_only=True, regularization_seed_no_state_or_checks=True,
        regularization_seed_hsd_only=True, regularization_seed_off_zero=True, regularization_seed_initialization_kept=True,
        parent_cpp_restored_after_two_regularization_hooks=True, parent_headers_preserved_after_rename=True,
        pure_coupled_parent_preserved_after_rename=True, immediate_v98_cpp_preserved_after_rename=False,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v100-coupled.hpp').read_bytes()).hexdigest(),
        inherited_v98_static=inherited, staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
