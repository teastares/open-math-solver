"""第93轮构建与两套数学原件的只读审计；不启动编译器或求解进程。"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'knowledge/experiments/data/2026-10-12-halpern-ipm'
PLAN_SHA = 'ad60585b037cf6988156c5b039420925e2b984851910f7e5351b2ff766c8c7df'
PARENT_INST_SHA = 'f06fadb5dbe3260e20608163e77df7ad05b5d0bb64d3194b82948fa760fe7253'
PARENT_PLAN_SHA = '842e40ac36676ba1d83a0e96a1474d95d55cd83b1fa4dc043d52a0d19769dcf6'
CLARIFICATION_SHA = 'be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8'
MAPS = ('protected_before_and_after', 'generated_inputs_before_and_after', 'compile_inputs_after_and_final',
    'link_explicit_before_and_after', 'test_link_explicit_before_and_after', 'outputs')
GROUPS = ('wrapper', 'mapping', 'transaction', 'budget', 'exception')
REQUIRED_CASES = {
    'initial_forcing_real_hsd': 'transaction',
    'initial_forcing_strict_fallback': 'mapping',
    'initial_forcing_delivery_budget': 'budget',
}
KERNEL_CASES = {'exact_initial_physical', 'fgmres_variable_preconditioner', 'block_tolerances_are_independent',
    'happy_breakdown_requires_true_residual', 'fixed_window_and_cycle_exhaustion', 'invalid_dimension_and_callback_values',
    'budget_boundaries_keep_partial', 'pure_tau_kappa_has_search_space'}
ENERGY_CASES = {'energy_nominal_schur_identity', 'energy_stored_diagonal_contract', 'energy_budget_partial_boundaries'}
WIDE_CASES = {'wide_exact_initial_physical', 'wide_fgmres_variable_preconditioner', 'wide_block_tolerances_are_independent',
    'wide_happy_breakdown_requires_true_residual', 'wide_fixed_window_and_cycle_exhaustion',
    'wide_invalid_dimension_and_callback_values', 'wide_budget_boundaries_keep_partial', 'wide_pure_tau_kappa_has_search_space',
    'wide_direction_late_narrow_bridge', 'wide_double_column_wide_true_product', 'wide_promotion_budget_prefix'}
RESTART_CASES = {'residual_gap_improvement_restarts', 'residual_gap_without_improvement_keeps_basis',
    'residual_gap_restart_budget_and_caps'}
FORCING_CASES = {'initial_forcing_strict_precedence', 'initial_forcing_linear_boundaries',
    'initial_forcing_complementarity_stays_strict', 'initial_forcing_empty_zero_extreme_rhs',
    'initial_forcing_rejection_strict_fallback', 'initial_forcing_only_initial_point',
    'initial_forcing_budget_partial_snapshot', 'initial_forcing_invalid_norms_callbacks'}
LABELS = {'private_header', 'shared_start_core', 'assess_current_before_move', 'assess_current_stop', 'public_wrapper',
    'private_coupled_header', 'coupled_direction_adapter', 'coupled_direction_solver', 'coupled_mode',
    'coupled_constant_ledger', 'coupled_affine_role', 'coupled_corrector_role', 'coupled_report',
    'coupled_isolation', 'coupled_activation', 'energy_readonly_rows', 'energy_adapter', 'energy_denominator',
    'energy_report', 'energy_isolation', 'energy_activation', 'wide_direction_storage', 'wide_coupled_adapter',
    'wide_physical_product', 'wide_direction_delivery', 'wide_legacy_direction_path', 'wide_step_consumers',
    'wide_corrector_consumers', 'wide_point_commit', 'initial_forcing_adapter', 'initial_forcing_delivery'}
SNAPSHOTS = {'v96-plan.json', 'v96-baseline-check.json', 'v96-ipm-instrumentation.py', 'v96-coupled.hpp',
    'v96-ipm-self-test.cpp', 'v96-coupled-self-test.cpp', 'v96-build.py', 'v96-run.py', 'v96-runner-self-check.json',
    'v95-ipm-instrumentation.py', 'v95-coupled-self-test.cpp', 'v95-coupled.hpp', 'v93-energy.hpp', 'v92-coupled.hpp',
    'v92-pretest-clarification.json', 'v66-run.py', 'v47-profile-build.py'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'JSON重复键：' + key)
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(-float('inf') < value < float('inf'), 'JSON浮点溢出')
        return value
    def invalid(text):
        raise ValueError('非有限JSON：' + text)
    return json.loads(text, object_pairs_hook=unique, parse_float=finite, parse_constant=invalid)


def read(path):
    return parse(Path(path).read_text(encoding='utf-8'))


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def verify_files(mapping):
    require(isinstance(mapping, dict) and mapping, '身份映射须非空')
    for name, expected in mapping.items():
        require(Path(name).is_absolute() and sha(name) == expected, '身份变化：' + name)


def successful(path, command, cwd):
    value = read(path)
    require(type(value['returncode']) is int and value['returncode'] == 0 and value['wall_timeout'] is False and
        value['command'] == command and Path(value['cwd']) == cwd and type(value['stdout']) is str and type(value['stderr']) is str,
        '实际命令、退出或原始输出不符：' + str(path))
    return value


def static_archive_member(path, desired):
    """读取原普通ar成员，比较借用对象，不调用外部ar程序。"""
    raw = Path(path).read_bytes()
    require(raw.startswith(b'!<arch>\n'), '原静态库不是普通archive')
    offset, names, found = 8, b'', []
    while offset < len(raw):
        header = raw[offset:offset + 60]
        require(len(header) == 60 and header[-2:] == b'`\n', '原静态库成员头无效')
        size = int(header[48:58].decode('ascii').strip())
        require(size >= 0 and offset + 60 + size <= len(raw), '原静态库成员范围无效')
        body, name = raw[offset + 60:offset + 60 + size], header[:16].decode('ascii').strip()
        if name == '//':
            names = body
        elif name.startswith('/') and name[1:].isdigit():
            begin = int(name[1:])
            require(begin < len(names), '长成员名称越界')
            name = names[begin:].split(b'/\n', 1)[0].split(b'\x00', 1)[0].decode('utf-8')
        elif name.startswith('#1/'):
            count = int(name[3:])
            require(count <= len(body), '内联成员名称越界')
            name, body = body[:count].rstrip(b'\x00').decode('utf-8'), body[count:]
        else:
            name = name.rstrip('/')
        if Path(name).name == desired:
            found.append(body)
        offset += 60 + size + size % 2
    require(offset == len(raw) and len(found) == 1, '借用对象须为原库唯一成员')
    return found[0]


def math_details(value):
    fields = {'passed', 'public_lp_runs', 'explicit_highs_lp_solve_calls', 'scope', 'cases',
        'component_ipm_calls', 'synthetic_core_calls', 'details', *(group + '_cases' for group in GROUPS)}
    require(isinstance(value, dict) and set(value) == fields and value['passed'] is True and
        type(value['public_lp_runs']) is int and value['public_lp_runs'] == 0 and
        type(value['explicit_highs_lp_solve_calls']) is int and value['explicit_highs_lp_solve_calls'] == 0 and
        type(value['scope']) is str and value['scope'] and all(type(value[name]) is int and value[name] > 0
        for name in ('cases', 'component_ipm_calls', 'synthetic_core_calls', *(group + '_cases' for group in GROUPS))),
        '数学组件字段、类型或作用域不符')
    require(isinstance(value['details'], list) and len(value['details']) == value['cases'] ==
        sum(value[group + '_cases'] for group in GROUPS), '数学完成详情与计数不守恒')
    cases, counts = {}, dict.fromkeys(GROUPS, 0)
    for item in value['details']:
        require(isinstance(item, dict) and set(item) == {'id', 'group'} and type(item['id']) is str and
            item['id'] and item['id'].isascii() and item['id'] not in cases and item['group'] in GROUPS,
            '数学覆盖标识须唯一且分组合法')
        cases[item['id']] = item['group']
        counts[item['group']] += 1
    require(all(counts[group] == value[group + '_cases'] for group in GROUPS), '数学五组详情数不符')
    return cases


def math_coverage(build, work, details, plan):
    path = DATA / plan['math_evidence_file']
    require(sha(path) == plan['math_evidence_sha256'], '父数学原件变化')
    previous = read(path)
    old_check = previous['math_self_checks']['ipm']
    require(previous['passed'] is True and previous['plan_sha256'] == PARENT_PLAN_SHA and old_check['passed'] is True and
        type(old_check['public_lp_runs']) is int and old_check['public_lp_runs'] == 0, '父数学须为最终v95已通过原件')
    old = old_check['details']
    old_cases, cases = math_details(old), math_details(details)
    added = [item for item in details['details'] if item['id'] not in old_cases]
    require(old['cases'] == 73 and details['cases'] == 76 and old_cases.items() <= cases.items() and
        {item['id']: item['group'] for item in added} == REQUIRED_CASES == plan['required_coupled_ipm_cases'],
        '父73逐id/group保留，新三组须真实完成且不重复')
    calls = {name: dict(predecessor=old[name], current=details[name], change=details[name] - old[name])
        for name in ('component_ipm_calls', 'synthetic_core_calls')}
    expected = dict(passed=True, public_lp_runs=0, predecessor_evidence_file=str(path),
        predecessor_evidence_sha256=plan['math_evidence_sha256'], preserved_predecessor_cases=73,
        new_cases=added, required_new_cases=[dict(id=name, group=group) for name, group in REQUIRED_CASES.items()], component_calls=calls)
    extension = work / 'ipm-self-extension-check.json'
    require(equal(read(extension), expected), '覆盖extension须由实际父/子stdout重算')
    metadata = dict(file=extension.name, sha256=sha(extension), preserved_predecessor_cases=73, new_cases=3)
    require(equal(build['math_coverage_extension'], metadata), '覆盖extension身份或清单不符')
    source = (DATA / 'v96-ipm-self-test.cpp').read_text(encoding='utf-8')
    require(all(source.count('"' + name + '"') == 1 for name in REQUIRED_CASES) and
        'const auto expected = real_coupled_core(model, options);' in source and
        'solve_ipm_homogeneous_scaled(scaling, rhs, p, a, old);' in source and
        'std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits' in source and
        'struct InitialForcingHsdFixture' in source and 'solver::v94_hsd_product(' in source and
        'solver::v96_tolerances_and_rhs_norms(' in source and 'solver::v96_solve_coupled(' in source and
        'data.rhs_norms == std::array<double, 3>{3, 3, 1.5}' in source and 'reduced[1] != fixture.rhs[1]' in source and
        'errors == *work.initial_forcing->defect_norms' in source and 'fixture.events[2]) == "P1"' in source and
        'solver::v96_deliver_direction(' in source and 'observed == boundary && !output.v94_wide' in source and
        'failure.log.initial_forcing_returns == 1 && *result.ipm_homogeneous_scaled_inexact_returns == 0' in source and
        'Boundary::before_p_direct' in source,
        '三组须连接非单位齐次缩放原RHS同扫B3、完整J初E5、互补严格回退不重P/A及真实pack入口/末budget的private/public差1')
    return metadata, expected


def math_and_build(build, work, runner, helper, legacy, plan):
    """核现有命令、字节、三份.d与两套实际数学输出，0新增求解。"""
    work, cwd = Path(work).resolve(), ROOT / 'build/release'
    require(work.is_relative_to(ROOT) and sha(DATA / 'v96-plan.json') == PLAN_SHA == build['plan_sha256'] and
        len(legacy['files']) == 95 and len(legacy['libraries']) == 17 and build['production_changes'] is False and
        build['experimental_candidate_only'] is True and type(build['public_lp_runs']) is int and build['public_lp_runs'] == 0 and
        build['current_files'] == build['frozen_files'] == legacy['files'] and build['actual_libraries'] == legacy['libraries'],
        '单IPM候选95来源17库合同不符')
    require(sha(DATA / 'ipm-v66-identity.json') == plan['parent_identity_sha256'] == build['parent_identity_sha256'] and
        equal(read(DATA / 'ipm-v66-identity.json'), legacy) and sha(legacy['compiler']) == legacy['compiler_sha256'] and
        sha(DATA / 'ipm-v66-source.zip') == legacy['source_zip_sha256'] == plan['parent_source_sha256'], '冻结父身份变化')
    baseline_path = DATA / plan['baseline_verification_file']
    require(sha(baseline_path) == plan['baseline_verification_sha256'] == build['baseline_verification_sha256'] and
        Path(build['baseline_verification_file']) == baseline_path, '事前基线身份不符')
    baseline = read(baseline_path)
    require(baseline['files'] == legacy['files'] and baseline['libraries'] == legacy['libraries'] and
        baseline['solver_sha256'] == legacy['solver_sha256'], '事前95/17基线不符')
    for key in ('evidence', 'math_evidence', 'counter_evidence'):
        require(sha(DATA / plan[key + '_file']) == plan[key + '_sha256'], '背景、数学父与计数证据各自保护：' + key)
    require(plan['math_evidence_file'] == 'v95-closeout.json' and plan['evidence_file'] == 'v95-closeout.json' and
        sha(DATA / 'v92-pretest-clarification.json') == CLARIFICATION_SHA and
        sha(DATA / 'v95-ipm-instrumentation.py') == PARENT_INST_SHA and
        plan['inherited_coupled_clarification_file'] == 'v92-pretest-clarification.json' and
        plan['inherited_coupled_clarification_sha256'] == CLARIFICATION_SHA,
        '父v95背景/数学角色各自保护，澄清仍继承旧v92原件')
    require(build['readonly_parent_instrumentation'] == dict(path=str(DATA / 'v95-ipm-instrumentation.py'), sha256=PARENT_INST_SHA),
        '构建清单只读算法父必须为v95实际插桩，不能误指v88数学祖')
    algorithm_parent = DATA / plan['algorithm_parent_coupled_kernel_file']
    require(algorithm_parent.name == 'v95-coupled.hpp' and sha(algorithm_parent) == plan['algorithm_parent_coupled_kernel_sha256'] and
        build['readonly_algorithm_parent_coupled_kernel'] == dict(path=str(algorithm_parent), sha256=sha(algorithm_parent)),
        '纯核算法父须为冻结v95原件，仅DATA只读保护不作第八生成源')
    for mapping in ({str(ROOT / name): digest for name, digest in legacy['files'].items()},
        {str(ROOT / name): digest for name, digest in legacy['libraries'].items()},
        {str(work / 'frozen' / name): digest for name, digest in legacy['files'].items()}):
        verify_files(mapping)
    union = {}
    for name in MAPS:
        verify_files(build[name])
        for path, digest in build[name].items():
            require(path not in union or union[path] == digest, '重叠身份映射冲突')
            union[path] = digest
    verify_files(plan['live_highs_sources'])
    require(all(build['protected_before_and_after'][name] == value for name, value in plan['live_highs_sources'].items()),
        '六项原生关键来源构建前后保护')
    require(build['outputs'] == {str(path): sha(path) for path in work.iterdir() if path.is_file() and path.name != 'build-manifest.json'},
        '顶层真实输出必须完整')
    production, reference = ROOT / 'build/release/solver.exe', ROOT / '.tools/solver-ipm-v66.exe'
    require(sha(production) == sha(reference) == legacy['solver_sha256'] == build['production_sha256'] == build['frozen_parent_sha256'] and
        Path(build['executable']) == work / 'solver-presolve.exe' and sha(build['executable']) == build['executable_sha256'],
        '默认/冻结参考与候选程序身份不符')
    commands = build['commands']
    require(Path(commands['cwd']) == cwd and set(commands['compile']) == {'ipm'} and
        set(commands['math_self_tests']) == {'ipm', 'coupled'} and build['implicit_link_inputs_verified'] is False,
        '候选单元或双数学命令角色不符')
    lines = (work / 'ninja-commands.txt').read_text(encoding='utf-8').splitlines()
    originals = {}
    for role, relative in (('ipm', 'src/ipm.cpp'), ('main', 'src/main.cpp'), ('highs', 'src/highs_backend.cpp')):
        found = [helper.split_windows(line) for line in lines if ' -c ' in line and line.replace('\\', '/').endswith('/' + relative)]
        require(len(found) == 1 and helper.resolve(found[0][0], cwd) == Path(legacy['compiler']), '原编译模板不唯一')
        originals[role] = found[0]
    require(originals == commands['original_compile'] and [line for line in lines if ' -o solver.exe ' in line] == [legacy['link_command']],
        'Ninja原始模板不符')
    old_includes = {helper.resolve(value[2:], ROOT) for value in originals['ipm'] if value.startswith('-I')}
    includes = [value for value in originals['highs'] if value.startswith('-I') and
        helper.resolve(value[2:], ROOT) != ROOT / 'include' and helper.resolve(value[2:], ROOT) not in old_includes]
    require(includes and includes == commands['highs_include_arguments'], '只添加实际HiGHS头目录')
    outer = helper.split_windows(legacy['link_command'])
    require(len(outer) == 3 and outer[1].lower() == '/c' and outer[2].startswith('cd . && ') and outer[2].endswith(' && cd .'),
        '原链接包装不符')
    original_link = helper.split_windows(outer[2][len('cd . && '):-len(' && cd .')])
    require(original_link == commands['original_link'] and original_link.count('libsolver_ipm.a') == 1, '原链接顺序或IPM项不符')
    regularized = cwd / 'CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj'
    original_explicit = {str(helper.resolve(value, cwd)) for value in original_link if not value.startswith('-') and
        Path(value).suffix.lower() in ('.a', '.lib', '.obj', '.o')}
    expected_protected = {str(path) for path in (production, reference, DATA / 'ipm-v66-identity.json', DATA / 'ipm-v66-source.zip',
        baseline_path, DATA / plan['evidence_file'], DATA / plan['math_evidence_file'], DATA / plan['counter_evidence_file'],
        DATA / 'v92-pretest-clarification.json', DATA / 'v96-plan.json', DATA / 'v47-profile-build.py', DATA / 'v96-build.py',
        DATA / plan['coupled_kernel_file'], DATA / plan['parent_coupled_kernel_file'], algorithm_parent, DATA / plan['energy_kernel_file'], Path(legacy['compiler']),
        ROOT / '.tools/ninja-1.13.2/ninja.exe', regularized,
        cwd / 'build.ninja', cwd / 'CMakeFiles/rules.ninja', *(DATA / name for name in ('v96-ipm-instrumentation.py',
            'v96-ipm-self-test.cpp', 'v96-coupled-self-test.cpp', 'v96-run.py', 'v96-runner-self-check.json', 'v66-run.py',
            'v95-ipm-instrumentation.py', 'v95-coupled-self-test.cpp')))}
    require(set(build['protected_before_and_after']) == expected_protected | original_explicit | set(plan['live_highs_sources']),
        '全部实际构建输入须精确保护')
    require(len(plan['build_snapshot_files']) == 17 and set(plan['build_snapshot_files']) == SNAPSHOTS and
        {str(DATA / name) for name in SNAPSHOTS} <= set(build['protected_before_and_after']),
        '17份构建前快照来源须精确对应冻结计划并同属实际输入保护')
    desired = []
    for value in original_link:
        if value == 'libsolver_ipm.a':
            desired.extend([str(work / 'ipm-presolve.obj'), str(regularized)])
        elif not value.startswith('-Wl,--out-implib,'):
            desired.append(value)
    desired[desired.index('-o') + 1] = str(work / 'solver-presolve.exe')
    require(desired == commands['link'] and desired.count('libsolver_normal_cholesky.a') == 1,
        '只替IPM库并保原Normal及其余链接顺序')
    require(static_archive_member(cwd / 'libsolver_ipm.a', 'regularized_ldl.cpp.obj') == regularized.read_bytes(),
        '借用regularized对象须为原库成员字节')
    successful(work / 'link.json', desired, cwd)
    generated = {str(work / name) for name in ('ipm-presolve.cpp', 'v96-presolve-state.hpp', 'v96-coupled-state.hpp',
        'v96-coupled.hpp', 'v92-coupled.hpp', 'coupled-self.cpp', 'v93-energy.hpp')}
    require(set(build['generated_inputs_before_and_after']) == generated and
        Path(build['private_state_header']) == work / 'v96-presolve-state.hpp' and
        Path(build['private_coupled_state_header']) == work / 'v96-coupled-state.hpp', '七份生成源及状态头合同不符')
    kernel_path = DATA / plan['coupled_kernel_file']
    require(build['coupled_kernel'] == dict(source=str(kernel_path), source_sha256=sha(kernel_path),
        private_copy=str(work / 'v96-coupled.hpp'), private_copy_sha256=sha(kernel_path), namespace_renamed=False) and
        sha(work / 'v96-coupled.hpp') == sha(kernel_path) and
        (work / 'coupled-self.cpp').read_bytes() == (DATA / 'v96-coupled-self-test.cpp').read_bytes(),
        '新宽方向核和测试副本须逐字一致，不从DATA优先拾取头')
    parent_kernel_path = DATA / plan['parent_coupled_kernel_file']
    require(build['parent_coupled_kernel'] == dict(source=str(parent_kernel_path), source_sha256=sha(parent_kernel_path),
        private_copy=str(work / 'v92-coupled.hpp'), private_copy_sha256=sha(parent_kernel_path)) and
        sha(work / 'v92-coupled.hpp') == sha(parent_kernel_path) == plan['parent_coupled_kernel_sha256'],
        '旧11组须实际编译冻结v92双精度核副本，新LD核不冒称父字节保持')
    energy_path = DATA / plan['energy_kernel_file']
    require(build['energy_kernel'] == dict(source=str(energy_path), source_sha256=sha(energy_path),
        private_copy=str(work / 'v93-energy.hpp'), private_copy_sha256=sha(energy_path)) and
        sha(work / 'v93-energy.hpp') == sha(energy_path) == plan['energy_kernel_sha256'], '父能量header须与实际编译私有副本逐字一致')
    parent_test_path = DATA / plan['parent_coupled_test_file']
    require(parent_test_path.name == 'v95-coupled-self-test.cpp' and sha(parent_test_path) == plan['parent_coupled_test_sha256'],
        '原25组数学源身份冻结')
    parent_prefix = parent_test_path.read_bytes().split(b'int main() {', 1)[0]
    child_source = (DATA / 'v96-coupled-self-test.cpp').read_bytes()
    legacy_prefix = parent_prefix.split(b'#include "v95-coupled.hpp"', 1)[0]
    normalized_parent = parent_prefix.replace(b'v95-coupled.hpp', b'v96-coupled.hpp').replace(b'solver::v95_coupled', b'solver::v96_coupled')
    require(child_source.startswith(legacy_prefix) and child_source.startswith(normalized_parent),
        'legacy11保持原字节，父25严格API只重绑定纯核文件及namespace，原函数和验收不改')
    parent_calls = parent_test_path.read_text(encoding='utf-8').split('int main() {', 1)[1].split('static_assert(coupled::max_window', 1)[0]
    child_calls = child_source.decode('utf-8').replace('\r\n', '\n').split('int main() {', 1)[1]
    require(child_calls.startswith(parent_calls),
        '新main须先原顺序真实调用全部父25组，再运行新增forcing测试')
    dependencies = set()
    for source, obj, command, record in ((work / 'ipm-presolve.cpp', work / 'ipm-presolve.obj', commands['compile']['ipm'], 'ipm-compile.json'),
        (DATA / 'v96-ipm-self-test.cpp', work / 'ipm-self.obj', commands['math_self_tests']['ipm']['compile'], 'ipm-self-build.json')):
        expected = helper.compile_arguments(originals['ipm'], source, obj, work / 'frozen', ROOT)
        expected.insert(1, '-I' + str(work))
        expected[expected.index('-c'):expected.index('-c')] = includes
        require(expected == command, '保持原IPM全部浮点编译选项，仅替实际来源/私有头路径')
        successful(work / record, expected, cwd)
        actual = {str(path) for path in helper.dependency_paths(Path(str(obj) + '.d'), cwd)}
        require({str(source), *(str(work / name) for name in ('ipm-presolve.cpp', 'v96-presolve-state.hpp',
            'v96-coupled-state.hpp', 'v96-coupled.hpp', 'v93-energy.hpp'))} <= actual, '实际IPM .d须包含两状态头及当前两数学核')
        dependencies.update(actual)
    main_object = originals['main'][originals['main'].index('-o') + 1]
    ipm_link = []
    for value in original_link:
        if value == main_object:
            ipm_link.append(str(work / 'ipm-self.obj'))
        elif value == 'libsolver_ipm.a':
            ipm_link.append(str(regularized))
        elif value != '-municode' and not value.startswith('-Wl,--out-implib,'):
            ipm_link.append(value)
    ipm_link[ipm_link.index('-o') + 1] = str(work / 'ipm-self-test.exe')
    require(ipm_link == commands['math_self_tests']['ipm']['link'] and str(work / 'ipm-presolve.obj') not in ipm_link,
        '包含式harness只替main，不得重复链接IPM定义')
    successful(work / 'ipm-self-link.json', ipm_link, cwd)
    kernel_source, kernel_object, kernel_exe = work / 'coupled-self.cpp', work / 'coupled-self.obj', work / 'coupled-self-test.exe'
    kernel_compile = [str(Path(legacy['compiler'])), '-O3', '-DNDEBUG', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic',
        '-MMD', '-MF', str(kernel_object) + '.d', '-I' + str(work), '-c', str(kernel_source), '-o', str(kernel_object)]
    kernel_link = [str(Path(legacy['compiler'])), '-static', str(kernel_object), '-o', str(kernel_exe)]
    require(commands['math_self_tests']['coupled'] == dict(compile=kernel_compile, link=kernel_link, run=[str(kernel_exe)]),
        '独立核使用冻结编译器、标准头与独立对象，不混外部求解库')
    successful(work / 'coupled-self-build.json', kernel_compile, cwd)
    successful(work / 'coupled-self-link.json', kernel_link, cwd)
    kernel_dependencies = {str(path) for path in helper.dependency_paths(Path(str(kernel_object) + '.d'), cwd)}
    require({str(kernel_source), str(work / 'v92-coupled.hpp'), str(work / 'v96-coupled.hpp'), str(work / 'v93-energy.hpp')} <= kernel_dependencies,
        '第三.d同时包含旧双精度FG核、新LD核与父energy核副本')
    dependencies.update(kernel_dependencies)
    require(dependencies == set(build['compile_inputs_after_and_final']), '三份实际.d并集必须精确')
    explicit = lambda command: {str(helper.resolve(value, cwd)) for value in command if not value.startswith('-') and
        Path(value).suffix.lower() in ('.a', '.lib', '.obj', '.o')}
    require(explicit(desired) == set(build['link_explicit_before_and_after']) and
        explicit(ipm_link) | explicit(kernel_link) == set(build['test_link_explicit_before_and_after']), '显式链接输入并集不符')
    math_checks, math_values = {}, {}
    for name, executable in (('ipm', work / 'ipm-self-test.exe'), ('coupled', kernel_exe)):
        raw = successful(work / f'{name}-self-run.json', [str(executable)], ROOT)
        check = read(work / f'{name}-self-check.json')
        require(raw['stderr'] == '' and set(check) == {'passed', 'public_lp_runs', 'details'} and check['passed'] is True and
            type(check['public_lp_runs']) is int and check['public_lp_runs'] == 0 and equal(parse(raw['stdout']), check['details']) and
            commands['math_self_tests'][name]['run'] == raw['command'], '双数学摘要须对应实际命令/退出/原始stdout')
        math_checks[name], math_values[name] = check, check['details']
    math_details(math_values['ipm'])
    kernel_details = math_values['coupled']
    require(set(kernel_details) == {'passed', 'public_lp_runs', 'cases', 'default_limits', 'details'} and kernel_details['passed'] is True and
        type(kernel_details['public_lp_runs']) is int and kernel_details['public_lp_runs'] == 0 and
        type(kernel_details['cases']) is int and kernel_details['cases'] == 33 and type(kernel_details['details']) is list and
        len(kernel_details['details']) == 33 and all(type(item) is str for item in kernel_details['details']) and
        set(kernel_details['details']) == KERNEL_CASES | ENERGY_CASES | WIDE_CASES | RESTART_CASES | FORCING_CASES and
        set(plan['required_energy_standalone_cases']) == ENERGY_CASES and set(plan['required_wide_standalone_cases']) == WIDE_CASES and
        set(plan['required_restart_standalone_cases']) == RESTART_CASES and set(plan['required_initial_forcing_standalone_cases']) == FORCING_CASES,
        '独立核原25组与新forcing8组须有唯一已完成标识')
    old_coupled = read(DATA / plan['math_evidence_file'])['math_self_checks']['coupled']
    require(old_coupled['passed'] is True and type(old_coupled['public_lp_runs']) is int and old_coupled['public_lp_runs'] == 0 and
        old_coupled['details']['cases'] == 25 and set(old_coupled['details']['details']) == KERNEL_CASES | ENERGY_CASES | WIDE_CASES | RESTART_CASES,
        '父25组数学覆盖须来自v95实际已通过stdout对应原件')
    limits = kernel_details['default_limits']
    require(type(limits) is dict and set(limits) == {'max_window', 'max_cycles'} and all(type(limits[key]) is int and
        limits[key] == plan['coupled_schema'][key] for key in limits), '真实核窗口和周期必须绑定冻结计划')
    require(equal(build['math_self_tests'], {name: {**math_values[name], 'run_file': f'{name}-self-run.json',
        'check_file': f'{name}-self-check.json'} for name in ('ipm', 'coupled')}), '双数学清单与真实输出不符')
    extension, coverage = math_coverage(build, work, math_values['ipm'], plan)
    transform = read(work / 'transforms.json')
    original = (work / 'frozen/src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    actual = (work / 'ipm-presolve.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    changes = transform['changes']
    require(len(changes) == 31 and {item['label'] for item in changes} == LABELS and
        Path(transform['source']) == work / 'frozen/src/ipm.cpp' and transform['source_sha256'] == legacy['files']['src/ipm.cpp'] and
        Path(transform['patched']) == work / 'ipm-presolve.cpp' and transform['patched_sha256'] == sha(work / 'ipm-presolve.cpp'),
        '三十一锚与真实编译来源身份不符')
    restored = actual
    for item in reversed(changes):
        require(set(item) == {'label', 'before', 'after'} and item['before'] != item['after'] and restored.count(item['after']) == 1,
            '逆恢复锚须唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == original, '三十一锚须逐字逆恢复冻结原始核心')
    static = read(work / 'static-check.json')
    ipm_static = static['ipm']
    delta = ipm_static['staticdelta_source']
    require(static['passed'] is True and ipm_static['anchors'] == 31 and ipm_static['reversible'] is True and
        ipm_static['parent_anchors'] == 29 and ipm_static['new_cpp_algorithm_anchors'] == 2 and
        ipm_static['initial_forcing_extra_anchors'] == 2 and ipm_static['inherited_wide_anchors'] == 8 and
        ipm_static['coupled_no_inexact_return'] is False and ipm_static['strict_or_explicit_forced_initial_return'] is True and
        ipm_static['immediate_v95_cpp_preserved_after_rename'] is False and
        delta['immediate_parent_version'] == 95 and delta['new_cpp_algorithm_anchors'] == 2 and delta['coupled_schema'] == 2 and
        delta['immediate_v95_cpp_preserved_after_rename'] is False and delta['coupled_no_inexact_return'] is False and
        all(delta[key] is True for key in ('parent_cpp_restored_after_two_forcing_hooks', 'parent_energy_preserved',
            'parent_wide_consumers_preserved', 'parent_off_path_preserved', 'strict_or_explicit_forced_initial_return',
            'initial_forcing_rhs_same_scan', 'initial_forcing_public_after_delivery', 'initial_forcing_no_repeat_initial_pa')) and
        ipm_static['inherited_v95_static']['passed'] is True and ipm_static['inherited_v95_static']['anchors'] == 29 and
        ipm_static['inherited_v95_static']['coupled_no_inexact_return'] is True and
        all(ipm_static[key] is True for key in ('wide_direction_return_preserved', 'wide_double_column_wide_true_product',
            'wide_all_consumers_covered', 'wide_no_direction_narrowing', 'wide_point_late_narrowing',
            'wide_disabled_preserves_old_path', 'wide_parent_restorable', 'wide_no_extra_linear_work')),
        '实际CPP父29加forcing两锚，schema2明确非strict受控返回，父严格返回证明只属历史v95')
    candidates = []
    for path in sorted(DATA.glob('v96-build-self-check-attempt*.json')):
        record = read(path)
        if record['returncode'] == 0 and record['source_sha256'] == sha(DATA / 'v96-build.py'):
            require(hashlib.sha256(record['source'].encode('utf-8')).hexdigest() == record['source_sha256'] and
                equal(parse(record['stdout']), static), '静态原始执行需对应最终构建source及冻结输入')
            candidates.append(dict(file=path.name, sha256=sha(path), elapsed_seconds=record['elapsed_seconds']))
    require(candidates, '必须有真实成功的最终构建器静态执行原件')
    return math_checks, dict(passed=True, candidate_units=1, compiled_including_math_harnesses=3,
        dependencies=len(dependencies), generated_sources=7, candidate_explicit_link_inputs=len(explicit(desired)),
        test_explicit_link_inputs=len(explicit(ipm_link) | explicit(kernel_link)), original_normal_library_retained=True,
        upstream_highs_include_arguments=includes, transformations_reversible=31, parent_math_cases=73, new_math_cases=coverage['new_cases'],
        component_calls=coverage['component_calls'], math_coverage_extension=extension, final_static_attempt=candidates[-1],
        regularized_matches_original_archive_member=True, math_raw_files={name: f'{name}-self-run.json' for name in ('ipm', 'coupled')},
        old_coupled_math_cases=25, new_initial_forcing_standalone_cases=8, parent_coupled_functions_preserved_after_kernel_rebinding=True,
        legacy_standalone_prefix_bytes_preserved=True, parent_standalone_call_sequence_preserved=True,
        coupled_schema=2, initial_forcing_cpp_hooks=2, algorithm_parent_kernel_only_data=True,
        source_semantics_scope='构建审计独立逆恢复/命令/3.d/7生成来源/76+33数学原件；31锚/forcing同扫、schema2与公共交付账本由另一forcing审计交叉核对。',
        public_lp_runs=0)
