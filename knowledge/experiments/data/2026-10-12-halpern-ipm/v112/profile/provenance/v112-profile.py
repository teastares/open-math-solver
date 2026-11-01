"""父106的固定216次分层诊断：复用数值验证，只编排与重解析，不构建或生成C++。"""
import argparse
import contextlib
import copy
import datetime
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'knowledge/experiments/data/2026-10-12-halpern-ipm'
TOOLS = {'build_experimental.py': '2b6c28bf5315e9ecb168a5deca92fcc7da9d3e676d1f424cf52585ccbd36bd03',
    'benchmark_experimental.py': '568dcbba11dc7f297d9e2d5dde63e309e5b69d1bf01a4be93c608690273b2f45'}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
for name, expected in TOOLS.items():
    if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError('稳定纯工具身份变化：' + name)
B, C = importlib.import_module('build_experimental'), importlib.import_module('benchmark_experimental')
require, sha, read, equal = B.require, B.sha, B.read, B.equal
STAGES = ('compatibility-development', 'compatibility-probe', 'profile')
SCHEDULE = dict(zip(STAGES, (22, 12, 182)))
PROFILE_STAGES = ('total', 'canonicalize', 'kkt_setup', 'normal_setup', 'symbolic_analysis', 'factor',
    'normal_factor', 'rectangular_assembly', 'explicit_gram_assembly', 'cholmod_numeric', 'factor_check',
    'hsd_constant_direct', 'hsd_direction_direct', 'direct', 'normal_solve', 'cholmod_solve', 'preconditioner',
    'hsd_column_product', 'hsd_true_product', 'coupled', 'point_update', 'assess_homogeneous', 'verify_local',
    'initialize', 'run_homogeneous', 'corrector_rhs', 'homogeneous_step', 'cone_multiply', 'cone_transpose_multiply',
    'cholmod_factor_adapter', 'solve_factored', 'presolve_control', 'dot', 'norm', 'axpy')
PROFILE_TOP = ('schema', 'stages', 'clock_reads', 'exclusive_sum_ns', 'total_inclusive_ns', 'conserved', 'stack_empty', 'counts_closed')
PROFILE_FIELDS = ('inclusive_ns', 'exclusive_ns', 'started', 'completed', 'unwound')
MARKER_FIELDS = ('presolve_marker', 'presolve_marker_error', 'coupled_marker', 'coupled_marker_error', 'energy_marker', 'energy_marker_error')


def path_in_root(value):
    path = (ROOT / value).resolve(); require(path.is_relative_to(ROOT), '路径不得越出工作区'); return path


def fixed_plan(plan_path, work, output):
    plan = read(plan_path)
    require(plan['version'] == plan['candidate_version'] == 112 and plan['algorithm_parent_version'] == 106 and
        plan['diagnostic_only'] is True and plan['numerical_algorithm_changes'] is False and plan['production_changes'] is False,
        '仅接受父106的纯诊断计划')
    require(plan['new_algorithm_round'] is True and plan['maximum_diagnostic_solver_runs'] == 216 and plan['diagnostic_schedule'] == SCHEDULE,
        '源码体构建模式不代表新增数值算法或性能轮，诊断调度固定216')
    require(tuple(plan['profile_stages']) == PROFILE_STAGES and tuple(plan['profile_top_fields']) == PROFILE_TOP and
        tuple(plan['profile_stage_fields']) == PROFILE_FIELDS and plan['profile_environment'] == 'SOLVER_V112_PROFILE' and plan['profile_marker'] == 'V112_PROFILE', '诊断35阶段及8/5字段必须测前冻结')
    require(path_in_root(plan['work_directory']) == work and path_in_root(plan['output_directory']) == output and work != output,
        'CLI目录必须对应冻结计划且互不重合')
    require(plan['source_count'] == len(plan['sources']) == 12 and plan['verification']['component_cases'] == {'ipm': 106, 'coupled': 57}, '十二源与163组件')
    names = [item['canonical'] for item in plan['sources']]
    for name in names: B.canonical_relative(name)
    require(len(set(names)) == len(names), '源名称唯一')
    B.verify({str(path_in_root(name)): value for name, value in plan['tool_sources'].items()})
    static = path_in_root(plan['source_static_audit_file'])
    require(sha(static) == plan['source_static_audit_sha256'] and read(static)['passed'] is True and read(static)['public_lp_runs'] == 0,
        '测前独立源码审计身份')
    runner, frozen = C.protocol(plan)
    require((frozen['time_limit'], frozen['wall_timeout'], frozen['threads'], frozen['seed']) == (5, 35, 1, 0), '原共同预算与资源')
    return plan, runner, frozen


def valid_profile(value):
    integer = lambda item: type(item) is int and item >= 0
    if type(value) is not dict or set(value) != set(PROFILE_TOP) or type(value['schema']) is not int or value['schema'] != 1:
        return False
    stages = value['stages']
    if type(stages) is not dict or set(stages) != set(PROFILE_STAGES): return False
    for item in stages.values():
        if type(item) is not dict or set(item) != set(PROFILE_FIELDS) or not all(integer(item[key]) for key in PROFILE_FIELDS): return False
        if item['exclusive_ns'] > item['inclusive_ns'] or item['started'] != item['completed'] + item['unwound']: return False
        if item['started'] == 0 and (item['inclusive_ns'] != 0 or item['exclusive_ns'] != 0): return False
    return all(type(value[key]) is bool and value[key] for key in ('conserved', 'stack_empty', 'counts_closed')) and \
        all(integer(value[key]) for key in ('clock_reads', 'exclusive_sum_ns', 'total_inclusive_ns')) and stages['total']['started'] == 1 and \
        value['clock_reads'] == 2 * sum(item['started'] for item in stages.values()) and \
        value['exclusive_sum_ns'] == sum(item['exclusive_ns'] for item in stages.values()) == value['total_inclusive_ns'] == stages['total']['inclusive_ns']


def profile_marker(stderr, enabled, plan):
    prefix = plan['profile_marker'] + ' '
    lines = [line for line in stderr.splitlines() if line.startswith(prefix)]
    if not enabled: return None, 'unexpected_profile_marker' if lines else None
    if len(lines) != 1: return None, 'profile_marker_count'
    try: value = B.parse(lines[0][len(prefix):])
    except (TypeError, ValueError, RuntimeError): return None, 'invalid_profile_json'
    return value, None if valid_profile(value) else 'invalid_profile_schema'


def markers(stderr, enabled, plan, runner):
    # 保留原始stderr；只为冻结的三标记验证器提供不含第四标记的临时视图。
    prefix = plan['profile_marker'] + ' '
    legacy = '\n'.join(line for line in stderr.splitlines() if not line.startswith(prefix))
    return (*runner.inspect_markers(legacy, 'diagnostic_on'), *profile_marker(stderr, enabled, plan))


def valid_process(row, runner):
    return runner.valid_process(row) and row['profile_marker_error'] is None and type(row['profile_enabled']) is bool and \
        (valid_profile(row['profile_marker']) if row['profile_enabled'] else row['profile_marker'] is None)


def variants(stage):
    return ('profile_off', 'profile_on') if stage == 'profile' else ('parent106', 'profile_off')


def specification(stage, runner, frozen):
    models = runner.specification('netlib' if stage == 'profile' else stage, frozen)['models']
    require(len(models) * 2 == SCHEDULE[stage], '固定完整模型集合')
    return models


@contextlib.contextmanager
def solver_environment(plan, enabled, runner):
    # 冻结运行器先清所有SOLVER变量；之后只加入独立profile开关。
    with runner.solver_environment(True):
        os.environ[plan['profile_environment']] = '1' if enabled else '0'
        yield


def context(plan_path, work, output, plan):
    candidate, build = C.verify_work(work, plan_path)
    require(build['source_snapshot'] == {str((work / 'source' / item['canonical']).resolve()): item['canonical_sha256'] for item in plan['sources']}, '实际十二份快照')
    for role, count in plan['verification']['component_cases'].items():
        check = build['math_self_tests'][role]
        require(check['passed'] is True and check['public_lp_runs'] == 0 and check['details']['cases'] == len(check['details']['details']) == count,
            '实际组件通过，fake clock不算新增LP')
    parent_path = path_in_root(plan['source_parent_identity_file'])
    require(sha(parent_path) == plan['source_parent_identity_sha256'], '父106不可变身份')
    parent = read(parent_path); B.verify(parent['protected'])
    require(sha(parent['build_manifest']) == parent['build_manifest_sha256'] and sha(parent['candidate']) == parent['candidate_sha256'], '父106实际程序')
    static, audit_path = path_in_root(plan['source_static_audit_file']), path_in_root(plan['source_build_audit_file'])
    audit = read(audit_path)
    require(audit['passed'] is True and audit['public_lp_runs'] == 0 and audit['plan_sha256'] == sha(plan_path) and
        audit['work'] == str(work) and audit['build_identity_sha256'] == sha(work / 'identity.json') and audit['candidate_sha256'] == candidate['candidate_sha256'], 'LP前ROOT实际构建审计绑定')
    protected = dict(candidate['protected'])
    paths = [plan_path, Path(__file__), Path(B.__file__), Path(C.__file__), parent_path, work / 'identity.json', static, audit_path,
        DATA / plan['protocol_file'], DATA / 'v100-run.py', DATA / 'v88-run.py', DATA / 'v66-run.py']
    protected.update(B.hashes(paths)); protected[parent['candidate']] = parent['candidate_sha256']
    result = dict(schema=1, diagnostic_only=True, plan_sha256=sha(plan_path), work=str(work), build_identity_sha256=sha(work / 'identity.json'),
        candidate=candidate['candidate'], candidate_sha256=candidate['candidate_sha256'], parent=parent['candidate'], parent_sha256=parent['candidate_sha256'],
        source_static_audit_sha256=sha(static), source_build_audit_sha256=sha(audit_path), protected=protected)
    target = output / 'run-identity.json'
    if target.exists(): require(equal(read(target), result), '不得改写本轮运行身份')
    else:
        output.mkdir(parents=True, exist_ok=True); B.save(target, result)
    B.verify(protected)
    return result


def pair_result(left, right, runner, frozen):
    public = C.differences(C.public_view(left['report']), C.public_view(right['report']))
    private = C.differences(C.private_view(left), C.private_view(right))
    process = C.differences({key: left.get(key) for key in ('returncode', 'classification', 'error', 'status')},
        {key: right.get(key) for key in ('returncode', 'classification', 'error', 'status')})
    quality = C.differences({key: left.get(key) for key in ('quality_pass', 'quality_failure')},
        {key: right.get(key) for key in ('quality_pass', 'quality_failure')})
    valid = valid_process(left, runner) and valid_process(right, runner)
    limited = runner.base.limited(left, frozen) or runner.base.limited(right, frozen)
    return dict(valid_processes=valid, budget_limited=limited, public_equal_excluding_timings=not public,
        private_equal_excluding_fixed_clocks=not private, process_equal=not process, quality_equal=not quality,
        public_differences=public, private_strict_differences=private, process_differences=process, quality_differences=quality,
        dynamic_budget_differences=[item for item in private if item['path'] == C.DYNAMIC_BUDGET],
        unexpected_private_differences=[item for item in private if item['path'] != C.DYNAMIC_BUDGET],
        nonbudget_public_difference=not limited and bool(public or process or quality),
        same_path_attribution=valid and not limited and not (public or private or process or quality),
        reference_quality_pass=left['quality_pass'], candidate_quality_pass=right['quality_pass'])


def summary(stage, rows, models, runner, frozen, plan_sha, raw_sha):
    names = variants(stage); pairs = []
    for model in models:
        chosen = {row['requested_variant']: row for row in rows if row['model_index'] == model['model_index']}
        require(set(chosen) == set(names), '完整对子必须恰好两条')
        pairs.append(dict(model_index=model['model_index'], input=model['input'], **pair_result(chosen[names[0]], chosen[names[1]], runner, frozen)))
    valid = sum(valid_process(row, runner) for row in rows)
    passed = valid == len(rows) and (stage == 'profile' or not any(pair['nonbudget_public_difference'] for pair in pairs))
    on = [row for row in rows if row['profile_enabled'] and valid_profile(row['profile_marker'])]
    aggregates = {name: {key: sum(row['profile_marker']['stages'][name][key] for row in on) for key in PROFILE_FIELDS} for name in PROFILE_STAGES}
    return dict(schema=1, diagnostic_only=True, stage=stage, passed=passed, public_lp_runs=len(rows), plan_sha256=plan_sha, raw_sha256=raw_sha,
        valid_process_count=valid, profile_marker_error_count=sum(row['profile_marker_error'] is not None for row in rows),
        profile_marker_count=len(on), variants={name: dict(runs=sum(row['requested_variant'] == name for row in rows),
            quality_pass_count=sum(row['requested_variant'] == name and row['quality_pass'] for row in rows)) for name in names},
        pairs=pairs, same_path_attribution_pairs=sum(pair['same_path_attribution'] for pair in pairs), profile_on_stage_sums=aggregates,
        performance_gate_applied=False, new_highs_runs=0, goal_evaluated=False, goal_achieved=False)


def observed_sha(path):
    try: return sha(path), None
    except OSError as error: return None, str(error)


def run(stage, plan_path, work, output, plan, runner, frozen):
    identity = context(plan_path, work, output, plan)
    if stage == 'profile':
        for name in STAGES[:2]: require(archive(name, plan_path, work, output, plan, runner, frozen)[1]['passed'], '关闭兼容失败，停止完整诊断扩集')
    models, names = specification(stage, runner, frozen), variants(stage)
    for model in models: require(model['hash_error'] is None and sha(model['input']) == model['sha256'], '模型冻结身份')
    directory = output / stage; require(not directory.exists(), '每阶段必须新建目录，不覆盖或补跑替换')
    directory.mkdir(); provenance = directory / 'provenance'; provenance.mkdir()
    for source, name in ((plan_path, 'plan.json'), (Path(__file__), 'v112-profile.py'), (Path(B.__file__), 'build_experimental.py'),
        (Path(C.__file__), 'benchmark_experimental.py'), (DATA / plan['protocol_file'], 'protocol.json'), (DATA / 'v100-run.py', 'protocol-runner.py')):
        (provenance / name).write_bytes(source.read_bytes())
    protected = dict(identity['protected']); protected.update(B.hashes(provenance.iterdir())); protected[str(output / 'run-identity.json')] = sha(output / 'run-identity.json')
    manifest = dict(schema=1, diagnostic_only=True, stage=stage, plan_sha256=sha(plan_path), run_identity_sha256=sha(output / 'run-identity.json'),
        work=str(work), models=models, repeats=1, variants=list(names), runs=SCHEDULE[stage], time_limit=5, wall_timeout=35, threads=1, seed=0,
        algorithm_environment={runner.ENVIRONMENT: '1'}, profile_environment=plan['profile_environment'], protected=protected)
    B.save(directory / 'manifest.json', manifest)
    rows = []; raw_path = directory / 'runs.jsonl.gz'; protocol_stage = 'netlib' if stage == 'profile' else stage
    with gzip.open(raw_path, 'xt', encoding='utf-8') as stream:
        for model, repeat, position, variant in runner.compare_lp.schedule(models, 1, names):
            executable = identity['parent'] if variant == 'parent106' else identity['candidate']
            digest = identity['parent_sha256'] if variant == 'parent106' else identity['candidate_sha256']; enabled = variant == 'profile_on'
            before, before_error = observed_sha(executable)
            with solver_environment(plan, enabled, runner):
                row = runner.benchmark.run_one(runner.options_for(protocol_stage, 'diagnostic_on', Path(executable), frozen), Path(model['input']), 0, len(rows))
            parsed = markers(row['stderr'], enabled, plan, runner)
            row.update(dict(zip((*MARKER_FIELDS, 'profile_marker', 'profile_marker_error'), parsed)))
            after, after_error = observed_sha(executable); model_after, model_error = observed_sha(model['input'])
            row.update(model_index=model['model_index'], repetition=repeat, position_in_pair=position, requested_variant=variant, protocol_variant='diagnostic_on',
                frozen_model_sha256=model['sha256'], model_sha256_after=model_after, solver_sha256_before=before, solver_sha256_after=after,
                frozen_solver_sha256=digest, identity_observation_errors=[value for value in (before_error, after_error, model_error) if value],
                presolve_environment={runner.ENVIRONMENT: '1'}, profile_enabled=enabled, profile_environment={plan['profile_environment']: '1' if enabled else '0'},
                configuration_matches=runner.configuration_matches(row['report'], protocol_stage, 'diagnostic_on', model, frozen))
            row.update(runner.quality_fields(row, model, digest, frozen))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush(); rows.append(row)
            print(json.dumps(dict(stage=stage, run_index=row['run_index'], variant=variant, model=Path(model['input']).stem,
                quality_pass=row['quality_pass'], profile_marker_error=row['profile_marker_error']), ensure_ascii=False), flush=True)
    require(len(rows) == SCHEDULE[stage], '必须保留完整调度，失败不能删除')
    result = summary(stage, rows, models, runner, frozen, sha(plan_path), sha(raw_path)); B.save(directory / 'summary.json', result)
    B.verify(protected)
    archive(stage, plan_path, work, output, plan, runner, frozen)
    return result


def archive(stage, plan_path, work, output, plan, runner, frozen):
    identity = context(plan_path, work, output, plan); directory = output / stage
    manifest = read(directory / 'manifest.json'); models, names = specification(stage, runner, frozen), variants(stage)
    require(manifest['stage'] == stage and manifest['diagnostic_only'] is True and manifest['plan_sha256'] == sha(plan_path) == sha(directory / 'provenance/plan.json') and
        manifest['run_identity_sha256'] == sha(output / 'run-identity.json') and manifest['work'] == str(work) and manifest['models'] == models and
        manifest['variants'] == list(names) and manifest['runs'] == SCHEDULE[stage] and manifest['repeats'] == 1 and
        (manifest['time_limit'], manifest['wall_timeout'], manifest['threads'], manifest['seed']) == (5, 35, 1, 0), '阶段冻结调度身份')
    require(manifest['algorithm_environment'] == {runner.ENVIRONMENT: '1'} and manifest['profile_environment'] == plan['profile_environment'], '两开关独立且算法始终开启')
    B.verify(manifest['protected'])
    raw_path = directory / 'runs.jsonl.gz'
    with gzip.open(raw_path, 'rt', encoding='utf-8') as stream: rows = [B.parse(line) for line in stream]
    schedule = list(runner.compare_lp.schedule(models, 1, names)); require(len(rows) == len(schedule) == SCHEDULE[stage], '原始行必须对应全部唯一调度')
    protocol_stage = 'netlib' if stage == 'profile' else stage
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, schedule)):
        enabled = variant == 'profile_on'; executable = identity['parent'] if variant == 'parent106' else identity['candidate']
        digest = identity['parent_sha256'] if variant == 'parent106' else identity['candidate_sha256']
        require((row['run_index'], row['model_index'], row['repetition'], row['position_in_pair'], row['requested_variant']) == (index, model['model_index'], repeat, position, variant) and
            row['protocol_variant'] == 'diagnostic_on' and row['seed'] == 0 and row['command'][:3] == [executable, 'solve', model['input']] and runner.valid_command(row, 'diagnostic_on', frozen), '实际命令与逐行调度')
        require(not row['identity_observation_errors'] and row['solver_sha256_before'] == row['solver_sha256_after'] == row['frozen_solver_sha256'] == digest == sha(executable) and
            row['model_sha256'] == row['model_sha256_after'] == row['frozen_model_sha256'] == model['sha256'] == sha(model['input']), '程序和模型运行前后身份严格保持')
        require(row['presolve_environment'] == {runner.ENVIRONMENT: '1'} and row['profile_environment'] == {plan['profile_environment']: '1' if enabled else '0'} and row['profile_enabled'] is enabled, '真实模式字段')
        parsed = markers(row['stderr'], enabled, plan, runner)
        require(all(equal(row[key], value) for key, value in zip((*MARKER_FIELDS, 'profile_marker', 'profile_marker_error'), parsed)), '四标记必须从原stderr严格重解析')
        require(row['configuration_matches'] is runner.configuration_matches(row['report'], protocol_stage, 'diagnostic_on', model, frozen), '原配置验证')
        if row['report'] is not None: require(equal(B.parse(row['stdout']), row['report']), '原stdout完整报告')
        require(all(equal(row[key], value) for key, value in runner.quality_fields(row, model, digest, frozen).items()), '原LP质量不篡改')
        require(valid_process(row, runner), '原进程或诊断schema不合法，严格失败')
    result = summary(stage, rows, models, runner, frozen, sha(plan_path), sha(raw_path))
    require(equal(result, read(directory / 'summary.json')), '原summary与完整重算相等')
    return manifest, result, rows


def self_check(plan, runner, frozen):
    blank = dict.fromkeys(PROFILE_FIELDS, 0); entries = {name: dict(blank) for name in PROFILE_STAGES}
    entries['total'] = dict(inclusive_ns=100, exclusive_ns=60, started=1, completed=1, unwound=0)
    entries['factor'] = dict(inclusive_ns=40, exclusive_ns=40, started=1, completed=0, unwound=1)
    good = dict(schema=1, stages=entries, clock_reads=4, exclusive_sum_ns=100, total_inclusive_ns=100, conserved=True, stack_empty=True, counts_closed=True)
    require(valid_profile(good), '合法整数守恒夹具')
    mutations = [('clock_reads', 5), ('exclusive_sum_ns', 99), ('total_inclusive_ns', -1), ('schema', True), ('conserved', False), ('stack_empty', False), ('counts_closed', False)]
    for key, value in mutations:
        bad = copy.deepcopy(good); bad[key] = value; require(not valid_profile(bad), '诊断顶层突变必须拒绝')
    for key in PROFILE_FIELDS:
        bad = copy.deepcopy(good); bad['stages']['factor'][key] = -1; require(not valid_profile(bad), '阶段负值必须拒绝')
    bad = copy.deepcopy(good); bad['stages']['unknown'] = dict(blank); require(not valid_profile(bad), '未知阶段拒绝')
    text = plan['profile_marker'] + ' ' + json.dumps(good)
    require(profile_marker(text, True, plan)[1] is None and profile_marker(text, False, plan)[1] == 'unexpected_profile_marker' and
        profile_marker(text + '\n' + text, True, plan)[1] == 'profile_marker_count' and profile_marker('', True, plan)[1] == 'profile_marker_count', 'marker开关/重复/缺失突变')
    before = dict(os.environ)
    with solver_environment(plan, True, runner):
        require(os.environ[runner.ENVIRONMENT] == '1' and os.environ[plan['profile_environment']] == '1' and os.environ['OMP_NUM_THREADS'] == '1', '独立开启环境')
        with solver_environment(plan, False, runner): require(os.environ[runner.ENVIRONMENT] == '1' and os.environ[plan['profile_environment']] == '0', '嵌套关闭只关profile')
        require(os.environ[plan['profile_environment']] == '1', '嵌套环境恢复')
    require(dict(os.environ) == before, '原环境完整恢复')
    require(sum(len(list(runner.compare_lp.schedule(specification(stage, runner, frozen), 1, variants(stage)))) for stage in STAGES) == 216, '完整216唯一调度')
    return dict(passed=True, diagnostic_only=True, public_lp_runs=0, profile_schema_cases=14, marker_cases=4, environment_cases=4, scheduled_runs=SCHEDULE)


def closeout(plan_path, work, output, plan, runner, frozen):
    target, archive_path, inputs_path = (output.parent / name for name in ('v112-closeout.json', 'v112-candidate-evidence.zip', 'v112-evidence-inputs.json'))
    require(not target.exists() and not archive_path.exists() and not inputs_path.exists(), '结项、输入锁及归档均须CreateNew')
    identity = context(plan_path, work, output, plan); summaries, manifests, rows = {}, {}, {}
    for stage in STAGES: manifests[stage], summaries[stage], rows[stage] = archive(stage, plan_path, work, output, plan, runner, frozen)
    require(all(value['passed'] for value in summaries.values()) and sum(len(value) for value in rows.values()) == 216, '完整固定诊断及关闭兼容门槛')
    require(all(summaries['profile']['profile_on_stage_sums'][name]['started'] > 0 for name in ('normal_setup', 'normal_factor', 'cholmod_numeric', 'normal_solve', 'cholmod_solve')), '真实Normal TU须进入同一跨TU统计')
    wrappers, calls = {}, []
    for path in sorted(output.parent.glob('v112-*.json')):
        value = read(path)
        if not ('command' in value and 'input_sources_before_execution' in value): continue
        require(value['changed_inputs'] == [], '实际调用输入原件不变')
        for item in value['input_sources_before_execution'].values(): require(hashlib.sha256(item['source'].encode()).hexdigest() == item['sha256'], '实际调用前源码快照')
        wrappers[path.name] = dict(sha256=sha(path), command=value['command'], returncode=value['returncode']); calls.append(value)
    stages = [value for value in calls if len(value['command']) >= 2 and Path(value['command'][1]).resolve() == Path(__file__).resolve() and '--stage' in value['command'] and value['command'][value['command'].index('--stage') + 1] in STAGES]
    require(len(stages) == 3 and all(sum(value['command'][value['command'].index('--stage') + 1] == stage and value['returncode'] == 0 for value in stages) == 1 for stage in STAGES), '各诊断stage真实唯一执行，不补跑替换')
    files = {plan_path, Path(__file__), work / 'identity.json', path_in_root(plan['source_static_audit_file']), path_in_root(plan['source_build_audit_file'])}
    files.update(path_in_root(name) for name in plan['tool_sources'])
    files.update(path for path in output.parent.glob('v112-*') if path.is_file() and path.suffix in ('.json', '.py', '.txt', '.gz') and path not in (target, archive_path, inputs_path))
    files.update(path for path in (ROOT / '.tools').glob('v112-*.py') if path.is_file())
    files.update(path for path in output.rglob('*') if path.is_file())
    files.update(path for path in work.iterdir() if path.is_file() and path.suffix in ('.json', '.d', '.exe'))
    for child in ('source', 'provenance'): files.update(path for path in (work / child).rglob('*') if path.is_file())
    files.update(DATA / name for name in ('v100-run.py', 'v88-run.py', 'v66-run.py', plan['protocol_file'], 'ipm-v66-identity.json'))
    members = {}
    for path in sorted(files):
        path = path.resolve(); require(path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(work / 'frozen'), '归档仅含工作区原件，不重复冻结生产树')
        name = path.relative_to(ROOT).as_posix(); require(name not in members and '..' not in Path(name).parts, '归档成员唯一且无越界'); members[name] = sha(path)
    B.save(inputs_path, dict(schema=1, diagnostic_only=True, plan_sha256=sha(plan_path), run_identity_sha256=sha(output / 'run-identity.json'),
        source_static_audit_sha256=identity['source_static_audit_sha256'], source_build_audit_sha256=identity['source_build_audit_sha256'], member_sha256=members))
    members[inputs_path.relative_to(ROOT).as_posix()] = sha(inputs_path)
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive_file:
        for name in members: archive_file.write(ROOT / name, name)
    with zipfile.ZipFile(archive_path) as archive_file:
        require(len(archive_file.namelist()) == len(members) and set(archive_file.namelist()) == set(members), '归档成员完整')
        for name, expected in members.items(): require(hashlib.sha256(archive_file.read(name)).hexdigest() == expected == sha(ROOT / name), '归档逐成员原字节回读')
    result = dict(passed=True, version=112, round=109, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), diagnostic_only=True,
        numerical_algorithm_changes=False, new_performance_candidate=False, algorithm_parent_version=106, plan_sha256=sha(plan_path),
        production_unchanged=True, source_generation=False, total_diagnostic_lp_runs_revalidated=216, additional_public_lp_runs=0, new_highs_runs=0,
        stages=list(STAGES), stage_summaries=summaries, wrappers=wrappers, run_identity_sha256=sha(output / 'run-identity.json'),
        evidence_inputs_sha256=sha(inputs_path), goal=dict(evaluated=False, achieved=False, status='progress'),
        archive=dict(path=str(archive_path), sha256=sha(archive_path), members=len(members), bytes=archive_path.stat().st_size, member_sha256=members))
    B.save(target, result)
    return dict(passed=True, diagnostic_only=True, total_diagnostic_lp_runs_revalidated=216, additional_public_lp_runs=0, goal=result['goal'], archive_members=len(members))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--work', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('self-check', *STAGES, 'closeout'), required=True)
    args = parser.parse_args(); plan_path, work, output = (path_in_root(value) for value in (args.plan, args.work, args.output))
    plan, runner, frozen = fixed_plan(plan_path, work, output)
    result = self_check(plan, runner, frozen) if args.stage == 'self-check' else closeout(plan_path, work, output, plan, runner, frozen) if args.stage == 'closeout' else run(args.stage, plan_path, work, output, plan, runner, frozen)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
