"""第82轮固定91例预处理结构审计；不调用LP迭代，不生成求解成绩。"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = '5a12611f867d4c78fc3399050343a389eee1df883934e7e5917005a54ff451ec'
WORK = ROOT / '.tools/v85-presolve'
MAPS = ('protected_before_and_after', 'generated_inputs_before_and_after',
    'compile_inputs_after_and_final', 'link_explicit_before_and_after', 'outputs')
TIMINGS = ('configuration', 'read', 'copy', 'original_statistics', 'normalization', 'pass_model',
    'presolve', 'reduced_statistics', 'total')
NATIVE_OPTIONS = dict(primal_feasibility_tolerance=1e-7, dual_feasibility_tolerance=1e-7,
    ipm_optimality_tolerance=1e-8, lp_presolve_requires_basis_postsolve=True)
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text):
    def unique(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('重复JSON键：' + key)
            value[key] = item
        return value
    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('JSON浮点溢出')
        return value
    return json.loads(text, object_pairs_hook=unique, parse_float=finite,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('非有限JSON：' + value)))


def read(path):
    return strict_json(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def integer(value, minimum=0, maximum=(1 << 64) - 1):
    return type(value) is int and minimum <= value <= maximum


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_plan():
    require(sha(DATA / 'v85-plan.json') == PLAN_SHA, '冻结计划变化')
    plan = read(DATA / 'v85-plan.json')
    require(plan['maximum_public_lp_runs'] == 0 and plan['expected_structure_processes'] == 91 and
        (plan['structure_seconds'], plan['wall_timeout_seconds'], plan['threads'], plan['seed']) == (5, 35, 1, 0), '诊断预算不符')
    for name, expected in ((plan['baseline_verification_file'], plan['baseline_verification_sha256']),
            (plan['evidence_file'], plan['evidence_sha256']), ('ipm-v66-identity.json', plan['parent_identity_sha256'])):
        require(sha(DATA / name) == expected, '父证据变化：' + name)
    group = plan['structure_set']
    require(group['repeats'] == 1 and len(group['models']) == 91 and sha(group['models_file']) == group['models_file_sha256'], '91例来源变化')
    return plan


def valid_range(value, count, bounds=False):
    require(isinstance(value, dict) and set(value) == {'finite_count', 'zero_count', 'nonfinite_count', 'min_abs', 'max_abs'}, '范围字段')
    require(all(integer(value[key]) for key in ('finite_count', 'zero_count', 'nonfinite_count')) and
        value['zero_count'] <= value['finite_count'] and value['finite_count'] + value['nonfinite_count'] == count, '范围计数')
    samples = value['finite_count'] if bounds else value['finite_count'] - value['zero_count']
    if not samples:
        require(value['min_abs'] is None and value['max_abs'] is None, '空范围必须null')
    else:
        lo, hi = value['min_abs'], value['max_abs']
        require(number(lo) and number(hi) and 0 <= lo <= hi and (bounds or lo > 0), '有限范围端点')
    if not bounds:
        require(value['nonfinite_count'] == 0, '矩阵和成本不可非有限')


def valid_summary(value):
    if value is None:
        return
    require(isinstance(value, dict) and set(value) == {'n', 'm', 'nnz', 'matrix', 'cost', 'bounds'}, '规模字段')
    require(all(integer(value[key]) for key in ('n', 'm', 'nnz')), '规模类型')
    valid_range(value['matrix'], value['nnz'])
    valid_range(value['cost'], value['n'])
    valid_range(value['bounds'], 2 * (value['n'] + value['m']), True)


def inspect_report(value, returncode):
    fields = {'schema', 'mode', 'status', 'phase', 'scan_complete', 'deadline_exceeded', 'explicit_lp_solve_calls',
        'lp_solution_claimed', 'read_fixed_retry', 'input_sense', 'normalization_sign', 'original', 'reduced',
        'native_options', 'read_diagnostics', 'calls', 'native', 'timings_seconds', 'budget', 'error'}
    require(isinstance(value, dict) and set(value) == fields and type(value['schema']) is int and value['schema'] == 1 and
        value['mode'] == 'structure_only', '结构诊断顶层字段')
    require(all(type(value[key]) is bool for key in ('scan_complete', 'deadline_exceeded', 'lp_solution_claimed', 'read_fixed_retry')) and
        value['lp_solution_claimed'] is False and type(value['explicit_lp_solve_calls']) is int and value['explicit_lp_solve_calls'] == 0, '不能包含求解或解声明')
    status = value['status']
    require(type(status) is str and status in {'complete', 'deadline', 'read_error', 'input_error', 'api_error', 'exception', 'presolve_stopped'}, '状态标签')
    require(value['scan_complete'] is (status == 'complete') and value['deadline_exceeded'] is (status == 'deadline') and
        type(returncode) is int and returncode == (0 if status == 'complete' else 4 if status == 'deadline' else 2), '退出与完成/截止状态')
    require(type(value['phase']) is str and value['phase'] in {'arguments', 'finished', *TIMINGS[:-1]} and type(value['error']) is str, '阶段与错误文本')
    require(value['input_sense'] is None or value['input_sense'] in ('minimize', 'maximize'), '目标方向')
    sign = value['normalization_sign']
    require(sign is None or type(sign) is int and sign in (-1, 1), '目标符号类型')
    require((value['input_sense'] is None) is (sign is None) and (sign is None or sign == (1 if value['input_sense'] == 'minimize' else -1)), '方向归一化')
    for key in ('original', 'reduced'):
        valid_summary(value[key])
    require(value['native_options'] is None or same(value['native_options'], NATIVE_OPTIONS), '固定版native数值默认')
    diagnostic = value['read_diagnostics']
    require(isinstance(diagnostic, dict) and set(diagnostic) == {'warning', 'fixed_format_switch', 'other_warning', 'error', 'finite_data_replaced_by_infinity'} and
        all(type(item) is bool for item in diagnostic.values()), '读取诊断')
    calls = value['calls']
    require(isinstance(calls, dict) and set(calls) == {'read', 'pass_model', 'presolve'}, 'API账本字段')
    for role, maximum in (('read', 2), ('pass_model', 1), ('presolve', 1)):
        call = calls[role]
        require(isinstance(call, dict) and set(call) == {'started', 'completed'} and
            integer(call['started'], maximum=maximum) and integer(call['completed'], maximum=maximum) and
            0 <= call['started'] - call['completed'] <= 1, 'API部分完成账本')
    require(calls['read']['started'] != 2 or value['read_fixed_retry'], '第二读取必须fixed重试')
    require(not calls['pass_model']['started'] or calls['read']['completed'] >= 1, '传入前须完成读取')
    require(not calls['presolve']['started'] or calls['pass_model']['completed'] == 1, '预处理前须传模型')
    native = value['native']
    ints = ('read_status', 'pass_status', 'presolve_call_status', 'presolve_status_code', 'model_status_code')
    iterations = ('simplex_iterations', 'ipm_iterations', 'pdlp_iterations', 'crossover_iterations')
    require(isinstance(native, dict) and set(native) == {*ints, *iterations, 'presolve_status', 'model_status', 'info_valid', 'native_time_limit'}, 'native字段')
    require(all(native[key] is None or integer(native[key], -1) for key in ints) and
        all(type(native[key]) is int and native[key] in (-1, 0) for key in iterations), '原始状态与未知迭代不可伪造0')
    require(all(native[key] is None or type(native[key]) is str for key in ('presolve_status', 'model_status')) and
        type(native['info_valid']) is bool and (native['native_time_limit'] is None or number(native['native_time_limit']) and native['native_time_limit'] > 0), 'native标签/预算')
    timings = value['timings_seconds']
    require(isinstance(timings, dict) and set(timings) == set(TIMINGS) and all(number(item) and item >= 0 for item in timings.values()) and
        sum(timings[key] for key in TIMINGS[:-1]) <= timings['total'] + 1e-8, '实际非重叠阶段计时')
    budget = value['budget']
    require(isinstance(budget, dict) and set(budget) == {'external_seconds', 'budget_check_work_interval', 'checks', 'loop_units'} and
        type(budget['external_seconds']) is int and budget['external_seconds'] == 5 and
        type(budget['budget_check_work_interval']) is int and budget['budget_check_work_interval'] == 256 and
        integer(budget['checks']) and integer(budget['loop_units']), '结构预算定义')
    if status == 'complete':
        require(value['phase'] == 'finished' and value['original'] is not None and value['native_options'] is not None and
            native['read_status'] == native['pass_status'] == 0 and native['presolve_call_status'] in (0, 1) and
            calls['read']['started'] == calls['read']['completed'] >= 1 and
            calls['pass_model'] == calls['presolve'] == {'started': 1, 'completed': 1} and
            all(not flag for flag in diagnostic.values()) and timings['total'] < 5 and value['error'] == '', '完整扫描边界')
        require(native['presolve_status_code'] not in (5, 9), 'native停止不可冒充完整扫描')
        if native['presolve_status_code'] in (0, 3, 4):
            require(value['reduced'] is not None, '可用约化模型缺失')
    if status == 'deadline':
        require(timings['total'] >= 5, '外部deadline不应伪造')
    return value


def build_module():
    spec = importlib.util.spec_from_file_location('v85_build_evidence', DATA / 'v85-build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(work, plan):
    path = work / 'build-manifest.json'
    build = read(path)
    require(build['plan_sha256'] == PLAN_SHA and build['production_changes'] is False and build['public_lp_runs'] == 0, '构建身份')
    parent = read(DATA / 'ipm-v66-identity.json')
    require(build['current_files'] == build['frozen_files'] == parent['files'] and build['actual_libraries'] == parent['libraries'] and
        len(parent['files']) == 95 and len(parent['libraries']) == 17, '原95/17身份')
    protected = {str(Path(__file__)): sha(__file__), str(path): sha(path)}
    for group in MAPS:
        require(isinstance(build[group], dict) and build[group], '非空构建身份映射')
        for name, digest in build[group].items():
            require(name not in protected or protected[name] == digest, '身份映射冲突')
            protected[name] = digest
    for group in ('files', 'libraries'):
        protected.update({str(ROOT / name): digest for name, digest in parent[group].items()})
    protected.update({str(work / 'frozen' / name): digest for name, digest in parent['files'].items()})
    for name, expected in protected.items():
        require(sha(name) == expected, '实际构建输入/输出变化：' + name)
    inspect = build_module().inspect_test
    for role in ('probe', 'postsolve'):
        raw, check = read(work / f'{role}-self-run.json'), read(work / f'{role}-self-check.json')
        require(type(raw['returncode']) is int and raw['returncode'] == 0 and raw['wall_timeout'] is False and
            raw['command'] == build['commands']['self_tests'][role] and raw['cwd'] == str(ROOT), '实际数学命令/退出')
        result = inspect(role, raw)
        require(same(result, check['details']) and same(check, build['self_tests'][role]), '真实数学stdout与摘要')
    require(Path(build['executables']['probe']) == work / 'presolve-probe.exe', '诊断程序路径')
    value = dict(plan_sha256=PLAN_SHA, build_manifest=str(path), build_manifest_sha256=sha(path), protected=protected,
        executable=build['executables']['probe'], executable_sha256=build['executable_sha256']['probe'])
    target = DATA / 'v85-run-identity.json'
    if target.exists():
        require(same(read(target), value), '禁止覆盖不同运行身份')
    else:
        save(target, value)
    return value


def environment():
    clean = {key: value for key, value in os.environ.items() if not key.upper().startswith('SOLVER_V') and
        key.upper() not in ('OPENBLAS_CORETYPE', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    clean.update(dict.fromkeys(('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'), '1'))
    return clean


def inspect_raw(raw):
    try:
        require(raw['wall_timeout'] is False and raw['stderr'] == '', '进程超时或stderr污染')
        report = inspect_report(strict_json(raw['stdout']), raw['returncode'])
        return report, None
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return None, str(error)


def summarize(rows):
    reports = [row['report'] for row in rows if row['validation_error'] is None]
    usable = [report for report in reports if report['scan_complete'] and report['native']['presolve_status_code'] in (0, 3, 4) and report['reduced'] is not None]
    totals = {side: {key: sum(report[side][key] for report in usable) for key in ('n', 'm', 'nnz')} for side in ('original', 'reduced')}
    return dict(public_lp_runs=0, structure_processes=len(rows), valid_processes=len(reports),
        scan_complete_count=sum(report['scan_complete'] for report in reports), usable_structure_count=len(usable),
        status_counts=dict(collections.Counter(report['status'] for report in reports)),
        presolve_status_counts=dict(collections.Counter(report['native']['presolve_status'] for report in reports)),
        reduced_to_empty_count=sum(report['native']['presolve_status_code'] == 4 for report in usable),
        totals=totals, timing_totals_seconds={key: sum(report['timings_seconds'][key] for report in reports) for key in TIMINGS},
        timing_medians_seconds={key: statistics.median(report['timings_seconds'][key] for report in reports) if reports else None for key in TIMINGS},
        external_wall_seconds=sum(row['raw']['elapsed_wall_seconds'] for row in rows),
        interpretation='仅预处理结构与成本审计；无LP成绩，不证明缩减带来加速、覆盖或线性扩展。')


def archive():
    plan = fixed_plan()
    manifest = read(DATA / 'v85-structure-manifest.json')
    require(manifest['plan_sha256'] == PLAN_SHA, '扫描清单计划')
    identity = read(DATA / 'v85-run-identity.json')
    require(manifest['run_identity_sha256'] == sha(DATA / 'v85-run-identity.json'), '扫描身份原件')
    for name, expected in identity['protected'].items():
        require(sha(name) == expected, '冻结来源变化：' + name)
    raw_path = DATA / 'v85-structure-runs.jsonl.gz'
    require(sha(raw_path) == manifest['raw_sha256'], '原始行包变化')
    with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
        rows = [strict_json(line) for line in stream]
    require(len(rows) == 91, '91例不得遗漏')
    for index, (row, model) in enumerate(zip(rows, plan['structure_set']['models'])):
        require(row['index'] == index and row['input'] == model['input'] and row['input_sha256'] == model['sha256'] == sha(model['input']), '顺序与输入原件')
        require(row['raw']['command'] == [identity['executable'], model['input']] and row['raw']['cwd'] == str(ROOT), '实际诊断命令')
        report, error = inspect_raw(row['raw'])
        require(same(report, row['report']) and error == row['validation_error'], '原始stdout/stderr复算')
    summary = summarize(rows)
    require(same(summary, read(DATA / 'v85-structure-summary.json')), '结构汇总复算')
    return manifest, summary, rows


def run(work):
    require(not (DATA / 'v85-structure-runs.jsonl.gz').exists(), '禁止重跑或覆盖固定扫描')
    plan = fixed_plan()
    identity = freeze(work, plan)
    rows = []
    raw_path = DATA / 'v85-structure-runs.jsonl.gz'
    with gzip.open(raw_path, 'xt', encoding='utf-8') as stream:
        for index, model in enumerate(plan['structure_set']['models']):
            require(sha(model['input']) == model['sha256'], '输入变化')
            command = [identity['executable'], model['input']]
            begin = time.perf_counter()
            try:
                process = subprocess.run(command, cwd=ROOT, env=environment(), capture_output=True, timeout=plan['wall_timeout_seconds'])
                output, error, returncode, wall_timeout = process.stdout, process.stderr, process.returncode, False
            except subprocess.TimeoutExpired as expired:
                output, error, returncode, wall_timeout = expired.stdout or b'', expired.stderr or b'', None, True
            raw = dict(command=command, cwd=str(ROOT), returncode=returncode, wall_timeout=wall_timeout,
                elapsed_wall_seconds=time.perf_counter()-begin, stdout=output.decode('utf-8', errors='replace'), stderr=error.decode('utf-8', errors='replace'))
            report, validation_error = inspect_raw(raw)
            row = dict(index=index, input=model['input'], input_sha256=model['sha256'], raw=raw, report=report, validation_error=validation_error)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps(dict(index=index, model=Path(model['input']).stem, status=report['status'] if report else None, validation_error=validation_error), ensure_ascii=False), flush=True)
    summary = summarize(rows)
    save(DATA / 'v85-structure-summary.json', summary)
    save(DATA / 'v85-structure-manifest.json', dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA, run_identity_sha256=sha(DATA / 'v85-run-identity.json'), raw_sha256=sha(raw_path),
        structure_processes=len(rows), public_lp_runs=0))
    archive()
    print(json.dumps({key: summary[key] for key in ('structure_processes', 'valid_processes', 'scan_complete_count', 'usable_structure_count', 'totals')}, ensure_ascii=False))


def self_check():
    fixed_plan()
    require(strict_json('{"a":1}') == {'a': 1}, 'JSON基础')
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e10000}'):
        rejected = False
        try:
            strict_json(text)
        except ValueError:
            rejected = True
        require(rejected, '拒重复和非有限JSON')
    require(integer(0) and not integer(True) and not integer(1.0) and not number(True), '严格数值类型')
    empty = dict(finite_count=0, zero_count=0, nonfinite_count=0, min_abs=None, max_abs=None)
    valid_range(empty, 0)
    valid_range(dict(empty, finite_count=2, zero_count=2), 2)
    valid_range(dict(empty, finite_count=2, zero_count=2, min_abs=0, max_abs=0), 2, True)
    mutations = [dict(empty, finite_count=True), dict(empty, min_abs=0), dict(empty, finite_count=1)]
    for changed in mutations:
        rejected = False
        try:
            valid_range(changed, 0)
        except ValueError:
            rejected = True
        require(rejected, '范围计数和null反例')
    require(same(NATIVE_OPTIONS, copy.deepcopy(NATIVE_OPTIONS)) and not same(dict(NATIVE_OPTIONS, lp_presolve_requires_basis_postsolve=1), NATIVE_OPTIONS), 'native布尔不能冒充整数')
    require(not any(key.upper().startswith('SOLVER_V') for key in environment()), '实验开关清理')
    summary = dict(n=2, m=2, nnz=4, matrix=dict(empty, finite_count=4, min_abs=1, max_abs=2),
        cost=dict(empty, finite_count=2, min_abs=1, max_abs=1),
        bounds=dict(empty, finite_count=4, zero_count=2, nonfinite_count=4, min_abs=0, max_abs=4))
    native = dict(read_status=0, pass_status=0, presolve_call_status=0, presolve_status_code=3,
        presolve_status='Reduced', model_status_code=0, model_status='Not Set', info_valid=False,
        simplex_iterations=-1, ipm_iterations=-1, pdlp_iterations=-1, crossover_iterations=-1, native_time_limit=4.9)
    sample = dict(schema=1, mode='structure_only', status='complete', phase='finished', scan_complete=True,
        deadline_exceeded=False, explicit_lp_solve_calls=0, lp_solution_claimed=False, read_fixed_retry=False,
        input_sense='minimize', normalization_sign=1, original=summary, reduced=copy.deepcopy(summary),
        native_options=copy.deepcopy(NATIVE_OPTIONS), read_diagnostics=dict.fromkeys(
            ('warning', 'fixed_format_switch', 'other_warning', 'error', 'finite_data_replaced_by_infinity'), False),
        calls={role: dict(started=1, completed=1) for role in ('read', 'pass_model', 'presolve')},
        native=native, timings_seconds={**dict.fromkeys(TIMINGS, 0), 'total': 0.01},
        budget=dict(external_seconds=5, budget_check_work_interval=256, checks=1, loop_units=0), error='')
    inspect_report(sample, 0)
    stopped = copy.deepcopy(sample)
    stopped.update(status='presolve_stopped', scan_complete=False)
    stopped['native']['presolve_status_code'] = 5
    inspect_report(stopped, 2)
    expired = copy.deepcopy(sample)
    expired.update(status='deadline', scan_complete=False, deadline_exceeded=True, original=None, reduced=None,
        input_sense=None, normalization_sign=None, native_options=None, phase='arguments')
    expired['timings_seconds']['total'] = 5
    expired['calls'] = {role: dict(started=0, completed=0) for role in ('read', 'pass_model', 'presolve')}
    inspect_report(expired, 4)
    retry = copy.deepcopy(sample)
    retry['read_fixed_retry'] = True
    retry['calls']['read'] = dict(started=2, completed=2)
    inspect_report(retry, 0)
    changes = [
        (('schema',), True), (('normalization_sign',), True), (('lp_solution_claimed',), True),
        (('explicit_lp_solve_calls',), True), (('original', 'n'), True),
        (('original', 'matrix', 'finite_count'), 3), (('original', 'cost', 'min_abs'), 0),
        (('native_options', 'lp_presolve_requires_basis_postsolve'), 1),
        (('budget', 'external_seconds'), 5.0), (('budget', 'budget_check_work_interval'), True),
        (('calls', 'presolve', 'completed'), 2), (('calls', 'pass_model', 'completed'), 0),
        (('calls', 'read', 'started'), 2), (('native', 'simplex_iterations'), 1),
        (('native', 'ipm_iterations'), False), (('native', 'presolve_status_code'), 5),
        (('timings_seconds', 'presolve'), 1), (('timings_seconds', 'total'), float('inf')),
        (('scan_complete',), False), (('deadline_exceeded',), True)]
    for keys, wrong in changes:
        changed = copy.deepcopy(sample)
        part = changed
        for key in keys[:-1]:
            part = part[key]
        part[keys[-1]] = wrong
        rejected = False
        try:
            inspect_report(changed, 0)
        except (ValueError, TypeError):
            rejected = True
        require(rejected, '结构反例未拒绝：' + '.'.join(keys))
    rejected = False
    try:
        inspect_report(sample, 4)
    except ValueError:
        rejected = True
    require(rejected, '退出码不可脱离报告状态')
    return dict(passed=True, public_lp_runs=0, strict_json_rejections=3, strict_type_cases=4, range_cases=6,
        report_valid_cases=4, report_mutation_cases=len(changes) + 1,
        interpretation='运行器纯数据检查；不执行结构程序或LP。')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('self-check', 'scan', 'verify'))
    parser.add_argument('--work', type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == 'self-check':
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage == 'scan':
        run(args.work.resolve())
    else:
        _, summary, _ = archive()
        print(json.dumps(dict(passed=True, public_lp_runs=0, structure_processes=summary['structure_processes']), ensure_ascii=False))
