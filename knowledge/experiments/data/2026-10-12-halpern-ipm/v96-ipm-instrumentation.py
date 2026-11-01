"""v96初P受控forcing：同扫原RHS范数，并在完整LD交付后记公共inexact。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = 'ad60585b037cf6988156c5b039420925e2b984851910f7e5351b2ff766c8c7df'
PARENT_INSTRUMENTATION_SHA256 = 'f06fadb5dbe3260e20608163e77df7ad05b5d0bb64d3194b82948fa760fe7253'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V96_PRESOLVE', 'V96_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V96_COUPLED ', 'V96_ENERGY '
RENAMES = (
    ('solver_v95_', 'solver_v96_'), ('v95_core', 'v96_core'),
    ('v95_solve_coupled', 'v96_solve_coupled'), ('v95_coupled_enabled', 'v96_coupled_enabled'),
    ('v95_role', 'v96_role'), ('v95_iteration', 'v96_iteration'),
    ('v95_energy_denominator', 'v96_energy_denominator'),
    ('V95_PRESOLVE', 'V96_PRESOLVE'), ('V95_COUPLED', 'V96_COUPLED'), ('V95_ENERGY', 'V96_ENERGY'),
    ('v95-presolve-state.hpp', 'v96-presolve-state.hpp'), ('v95-coupled-state.hpp', 'v96-coupled-state.hpp'),
    ('v95-coupled.hpp', 'v96-coupled.hpp'), ('solver::v95_coupled', 'solver::v96_coupled'),
)
_parent_path = DATA / 'v95-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v95插桩来源变化')
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

# 原生CPP只有以下两处增量；全部before/after是供独立AST审计读取的纯字面量。
FORCING_CPP_CHANGES = [
    {'label': 'initial_forcing_adapter', 'before': r'''std::vector<long double> v96_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v96_coupled::ProductDouble& multiply,
    const ::solver::v96_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v96_coupled_state::current();
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
  solver_v96_coupled_state::Collect collect{*log, call};
  const ::solver::v96_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v96_coupled::solve(layout, rhs, tolerance, precondition, multiply, multiply_wide, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v96_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

''', 'after': r'''struct V96ToleranceData {
  std::array<double, 5> tolerances{};
  std::array<double, 3> rhs_norms{};
};
V96ToleranceData v96_tolerances_and_rhs_norms(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const std::function<void()>& budget = {}) {
  if (budget) budget();
  if (rhs.size() != scaling.dimension()) throw std::invalid_argument("齐次五块门槛维数不一致");
  const ::solver::v96_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  V96ToleranceData data;
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    if ((i & 4095) == 0 && budget) budget();
    if (!std::isfinite(rhs[i])) throw NumericalFailure("齐次五块右端非有限");
    const auto block = layout.block(i);
    data.tolerances[block] = std::max(data.tolerances[block], std::abs(rhs[i]));
  }
  // 在原max扫描的结果乘strict系数前保留三项，不能从已舍入门槛反推原范数。
  data.rhs_norms = {data.tolerances[0], data.tolerances[1], data.tolerances[2]};
  for (auto& value : data.tolerances) value = 1e-12 + 1e-10 * value;
  if (budget) budget();
  return data;
}

HomogeneousDirection v96_deliver_direction(const detail::IpmHomogeneousScaling& scaling,
    std::vector<long double> solution, bool forced_initial, SolveResult& result,
    const std::function<void()>& budget = {}) {
  auto direction = v94_deliver_direction(scaling, std::move(solution), budget);
  // 原pack的最后预算已经正常返回，公共计数才承认一次非strict方向交付。
  if (forced_initial) ++*result.ipm_homogeneous_scaled_inexact_returns;
  return direction;
}

std::vector<long double> v96_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v96_coupled::ProductDouble& multiply,
    const ::solver::v96_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration, bool* forced_initial = nullptr) {
  if (forced_initial) *forced_initial = false;
  auto* log = solver_v96_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 原门槛准备仍只扫一遍完整RHS，准备失败不启动新的方向或forcing决策。
  const auto data = v96_tolerances_and_rhs_norms(scaling, rhs, budget);
  const ::solver::v96_coupled::InitialForcing forcing{data.rhs_norms};
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v96_coupled_state::Collect collect{*log, call};
  const ::solver::v96_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v96_coupled::solve(layout, rhs, data.tolerances, precondition,
        multiply, multiply_wide, call.work, budget, &forcing);
    if (forced_initial) *forced_initial = std::string_view(call.work.status) == "forced_initial";
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v96_coupled::Failure& error) {
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''},
    {'label': 'initial_forcing_delivery', 'before': r'''    auto solution = v96_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v96_role, v96_iteration);
    return v94_deliver_direction(scaling, std::move(solution), budget);
''', 'after': r'''    bool forced_initial = false;
    auto solution = v96_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v96_role, v96_iteration, &forced_initial);
    return v96_deliver_direction(scaling, std::move(solution), forced_initial, result, budget);
'''},
]

COUPLED_HEADER_CHANGES = [
    {'before': r'''  Index true_checks_started = 0, true_checks_completed = 0;
''', 'after': r'''  Index true_checks_started = 0, true_checks_completed = 0;
  Index initial_forcing_checks = 0, initial_forcing_returns = 0;
'''},
    {'before': r'''    log.true_checks_started += w.true_checks_started; log.true_checks_completed += w.true_checks_completed;
''', 'after': r'''    log.true_checks_started += w.true_checks_started; log.true_checks_completed += w.true_checks_completed;
    log.initial_forcing_checks += w.initial_forcing_checks; log.initial_forcing_returns += w.initial_forcing_returns;
'''},
    {'before': r'''  std::fprintf(stream, "{\"schema\":1,\"enabled\":%s", log.enabled ? "true" : "false");
''', 'after': r'''  std::fprintf(stream, "{\"schema\":2,\"enabled\":%s", log.enabled ? "true" : "false");
'''},
    {'before': r'''  count("true_checks_started", log.true_checks_started); count("true_checks_completed", log.true_checks_completed);
''', 'after': r'''  count("true_checks_started", log.true_checks_started); count("true_checks_completed", log.true_checks_completed);
  count("initial_forcing_checks", log.initial_forcing_checks); count("initial_forcing_returns", log.initial_forcing_returns);
'''},
    {'before': r'''inline void write(FILE* stream, const Log& log) noexcept {
''', 'after': r'''inline void write_initial_forcing(FILE* stream,
    const std::optional<::solver::v96_coupled::InitialForcingWork>& value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  const auto& forcing = *value;
  std::fputs("{\"rhs_norms\":[", stream);
  for (std::size_t block = 0; block < 3; ++block) {
    if (block) std::fputc(',', stream);
    number(stream, std::optional<long double>(forcing.rhs_norms[block]));
  }
  std::fputs("],\"defect_norms\":", stream);
  if (!forcing.defect_norms) std::fputs("null", stream);
  else {
    std::fputc('[', stream);
    for (std::size_t block = 0; block < 5; ++block) {
      if (block) std::fputc(',', stream);
      number(stream, std::optional<long double>((*forcing.defect_norms)[block]));
    }
    std::fputc(']', stream);
  }
  std::fputs(",\"measure\":", stream); number(stream, forcing.measure);
  std::fprintf(stream, ",\"decision\":\"%s\"}", forcing.decision);
}
inline void write(FILE* stream, const Log& log) noexcept {
'''},
    {'before': r'''  std::fputs("]}}", stream);
''', 'after': r'''  std::fputs("],\"initial_forcing\":", stream);
  write_initial_forcing(stream, w.initial_forcing);
  std::fputs("}}", stream);
'''},
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


COUPLED_HEADER = renamed(_parent['COUPLED_HEADER'])
# write()同名还存在于独立energy namespace；只改第一处coupled实现。
for _change in COUPLED_HEADER_CHANGES:
    require(COUPLED_HEADER.count(_change['before']) >= 1, 'coupled schema2头锚缺失')
    COUPLED_HEADER = COUPLED_HEADER.replace(_change['before'], _change['after'], 1)
ADAPTER = FORCING_CPP_CHANGES[0]['after']


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def transform(source):
    source = source.replace('\r\n', '\n')
    require('v96_core(' not in source, '拒绝重复插桩')
    parent_patched, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 29, '冻结v95的29锚变化')
    patched, changes = source, []
    for item in parent_changes:
        changed = dict(label=item['label'], before=renamed(item['before']), after=renamed(item['after']))
        require(patched.count(changed['before']) == 1, '父重绑定锚缺失或重复：' + changed['label'])
        patched = patched.replace(changed['before'], changed['after'], 1)
        changes.append(changed)
    require(patched == renamed(parent_patched), '应用新政策前必须精确保父v95白名单改名')
    for item in FORCING_CPP_CHANGES:
        require(patched.count(item['before']) == 1, 'forcing增量锚缺失或重复：' + item['label'])
        patched = patched.replace(item['before'], item['after'], 1)
        changes.append(dict(item))
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆锚必须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部父与forcing锚必须逐字逆恢复冻结核心')
    return patched, changes


def static_delta_source():
    return dict(passed=True, immediate_parent_version=95, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        immediate_v95_cpp_preserved_after_rename=False, parent_cpp_restored_after_two_forcing_hooks=True,
        parent_energy_preserved=True, parent_wide_consumers_preserved=True, parent_off_path_preserved=True,
        coupled_schema=2, coupled_no_inexact_return=False, strict_or_explicit_forced_initial_return=True,
        initial_forcing_rhs_same_scan=True, initial_forcing_public_after_delivery=True,
        initial_forcing_no_repeat_initial_pa=True, new_cpp_algorithm_anchors=2,
        scope='只说明初Pforcing合同及账本，不再宣称本轮全部线性返回strict。')


def static_check(root):
    require(hashlib.sha256((DATA / 'v96-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v96计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '纯v93能量核变化')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 29, '最终v95静态父必须通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    require(len(changes) == 31 and len(FORCING_CPP_CHANGES) == 2, '父29加forcing两锚必须为31')
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已经变换来源必须拒绝')
    require('"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER and
        'factor.v93_regularized_h()' in ENERGY_ADAPTER, '纯能量文件/namespace/getter必须保父原名')
    require(ADAPTER.count('for (std::size_t i = 0; i < rhs.size(); ++i)') == 1 and
        'data.rhs_norms = {data.tolerances[0], data.tolerances[1], data.tolerances[2]};' in ADAPTER and
        ADAPTER.index('data.rhs_norms =') < ADAPTER.index('1e-12 + 1e-10 * value'), 'B3必须在原扫描、门槛舍入前保留')
    require(ADAPTER.index('v94_deliver_direction(scaling, std::move(solution), budget)') <
        ADAPTER.index('++*result.ipm_homogeneous_scaled_inexact_returns') and
        'call.work, budget, &forcing)' in ADAPTER, '生产必传真实B3且公共inexact必须在原末pack之后')
    result = {key: value for key, value in inherited.items() if key not in
        ('staticdelta_source', 'inherited_v94_static', 'inherited_v95_static')}
    result.update(anchors=31, parent_anchors=29, initial_forcing_extra_anchors=2, new_cpp_algorithm_anchors=2,
        inherited_wide_anchors=8, reversible=True, shared_start=True, coupled_no_inexact_return=False,
        strict_or_explicit_forced_initial_return=True, initial_forcing_rhs_same_scan=True,
        initial_forcing_public_after_delivery=True, initial_forcing_no_repeat_initial_pa=True,
        immediate_v94_cpp_preserved_after_rename=False, immediate_v95_cpp_preserved_after_rename=False,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v96-coupled.hpp').read_bytes()).hexdigest(),
        inherited_v95_static=inherited, staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
