"""v95仅重绑定冻结v94接入；唯一数值增量位于独立restart核。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = '842e40ac36676ba1d83a0e96a1474d95d55cd83b1fa4dc043d52a0d19769dcf6'
PARENT_INSTRUMENTATION_SHA256 = 'd6f9499fcafcb91e7b4036d2f04d91c193fbbb294d27c0a1aaec469876be0b3f'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V95_PRESOLVE', 'V95_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V95_COUPLED ', 'V95_ENERGY '
# 只改私有包装和所调用纯核，复用v94的宽方向字段及全部消费者helper。
RENAMES = (
    ('solver_v94_', 'solver_v95_'),
    ('v94_core', 'v95_core'),
    ('v94_solve_coupled', 'v95_solve_coupled'),
    ('v94_coupled_enabled', 'v95_coupled_enabled'),
    ('v94_role', 'v95_role'),
    ('v94_iteration', 'v95_iteration'),
    ('v94_energy_denominator', 'v95_energy_denominator'),
    ('V94_PRESOLVE', 'V95_PRESOLVE'),
    ('V94_COUPLED', 'V95_COUPLED'),
    ('V94_ENERGY', 'V95_ENERGY'),
    ('v94-presolve-state.hpp', 'v95-presolve-state.hpp'),
    ('v94-coupled-state.hpp', 'v95-coupled-state.hpp'),
    ('v94-coupled.hpp', 'v95-coupled.hpp'),
    ('solver::v94_coupled', 'solver::v95_coupled'),
)
_parent_path = DATA / 'v94-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v94插桩来源变化')
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    for before, after in RENAMES:
        text = text.replace(before, after)
    return text


HEADER, WRAPPER, COUPLED_HEADER, ADAPTER = (renamed(_parent[key]) for key in ('HEADER', 'WRAPPER', 'COUPLED_HEADER', 'ADAPTER'))
ENERGY_STATE, ENERGY_ADAPTER = (renamed(_parent[key]) for key in ('ENERGY_STATE', 'ENERGY_ADAPTER'))
OPENING, INITIALIZATION, COUNT_FIELDS = (_parent[key] for key in ('OPENING', 'INITIALIZATION', 'COUNT_FIELDS'))
CORE_OPENING = renamed(_parent['CORE_OPENING'])
# 本轮不新增CPP字面量；独立审计从父纯字面表和RENAMES精确重建全部29锚。
WIDE_CHANGES = [{key: item[key] if key == 'label' else renamed(item[key]) for key in ('label', 'before', 'after')}
    for item in _parent['WIDE_CHANGES']]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    require('v95_core(' not in source, '拒绝重复插桩')
    parent_patched, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 29 and len(_parent['WIDE_CHANGES']) == 8, '冻结v94全部29锚及宽桥八锚数量变化')
    patched, changes = source, []
    for item in parent_changes:
        changed = dict(label=item['label'], before=renamed(item['before']), after=renamed(item['after']))
        require(changed['before'] != changed['after'] and patched.count(changed['before']) == 1,
            '父字面重绑定锚缺失或重复：' + changed['label'])
        patched = patched.replace(changed['before'], changed['after'], 1)
        changes.append(changed)
    require(changes[21:] == WIDE_CHANGES and patched == renamed(parent_patched),
        '实际CPP必须恰为父v94白名单改名，不能暗加重启或消费者算法')
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '全部父逆锚必须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部父锚须逐字恢复原冻结核心')
    return patched, changes


def static_delta_source():
    return dict(passed=True, immediate_parent_version=94, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        immediate_v94_cpp_preserved_after_rename=True, immediate_v94_adapter_preserved_after_rename=True,
        immediate_v94_consumers_preserved=True, parent_energy_preserved=True, coupled_schema_preserved=True,
        energy_schema_preserved=True, energy_file_namespace_preserved=True, new_cpp_algorithm_anchors=0,
        scope='只说明v95相对v94的差分；v94相对v93的完整LD桥证明保留在inherited_v94_static。')


def static_check(root):
    require(hashlib.sha256((DATA / 'v95-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v95计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '纯能量核必须逐字保留v93')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 29, '最终v94静态父须仍通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已经变换来源必须拒绝')
    require('"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER and
        'factor.v93_regularized_h()' in ENERGY_ADAPTER and 'v94_wide' in patched and 'v94_update_point(' in patched and
        '::solver::v95_coupled::solve' in ADAPTER, '纯能量、完整LD消费者或新核重绑定不符')
    result = {key: value for key, value in inherited.items() if key not in ('staticdelta_source', 'inherited_v94_static')}
    result.update(anchors=len(changes), reversible=True, shared_start=True, parent_anchors=29,
        new_cpp_algorithm_anchors=0, wide_extra_anchors=0, inherited_wide_anchors=8,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        immediate_v94_cpp_preserved_after_rename=True, immediate_v94_consumers_preserved=True,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v95-coupled.hpp').read_bytes()).hexdigest(),
        inherited_v94_static=inherited, staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
