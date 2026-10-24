"""从冻结 v66 与原 v70 实现构建只读基候选捕获程序；生产保持只读。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = 'dd03b4c7cdad5319b1e47fa0f8e67845f7002f85e7b6c33da3cf03b857b55c7b'
BASE_PATH = DATA / 'v47-profile-build.py'
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[k] for k in ('sha', 'require', 'save'))
split_windows, resolve = (BASE[k] for k in ('split_windows', 'resolve'))
file_hashes, require_hashes = (BASE[k] for k in ('file_hashes', 'require_hashes'))
compile_arguments, execute, dependency_paths = (BASE[k] for k in ('compile_arguments', 'execute', 'dependency_paths'))
CASES = ('candidate', 'no_candidate', 'cache_limit', 'budget_interrupted')
ARRAYS = ('g_starts', 'g_rows', 'g_values', 'costs', 'rhs', 'slack', 'dual', 'inequality',
          'candidate_x', 'candidate_s', 'candidate_z', 'selected_rows', 'pivot_columns')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def static_check():
    require(sha(DATA / 'v71-plan.json') == PLAN_SHA, '计划身份变化')
    plan = read(DATA / 'v71-plan.json')
    require(plan['candidate_limits'] == {'max_input_nonzeros': 8000000, 'max_factor_nonzeros': 32000000,
                                        'max_dimension': 1000000, 'max_selection_work': 1000000000},
            '候选限额与固定实现不符')
    instrument = runpy.run_path(str(DATA / 'v71-basis-instrumentation.py'))
    return instrument['self_check']()


def verify_self_capture(output, case, rows):
    require(rows['off']['stdout'] == rows['on']['stdout'], '捕获开关改变合成输出')
    require(not rows['off']['stderr'], '关闭捕获仍输出标记')
    require(json.loads(rows['on']['stdout'])['passed'] is True, '捕获内核自测未通过')
    off = output / ('capture-self-' + case + '-off')
    require(not off.with_suffix('.json').exists() and not off.with_suffix('.bin').exists(), '关闭捕获仍写文件')
    prefix = output / ('capture-self-' + case + '-on')
    metadata = read(prefix.with_suffix('.json'))
    binary = prefix.with_suffix('.bin').read_bytes()
    lines = rows['on']['stderr'].splitlines()
    require(len(lines) == 1 and lines[0].startswith('V71_CAPTURE '), '捕获标记不唯一')
    marker = json.loads(lines[0][len('V71_CAPTURE '):])
    require(marker['write_ok'] is True, '捕获文件写入失败')
    for key in ('schema', 'version', 'status', 'attempts', 'event_count', 'capture_errors', 'truncated', 'binary_bytes'):
        require(marker[key] == metadata[key], '捕获标记与文件不符：' + key)
    require(binary[:8] == b'V71BAS01' and len(binary) == metadata['binary_bytes'], '捕获二进制头或长度错误')
    require(prefix.with_suffix('.json').stat().st_size <= 65536, '元数据长度超限')
    expected = 'captured' if case == 'candidate' else 'no_candidate' if case == 'no_candidate' else 'capture_error'
    require(metadata['status'] == expected, '捕获状态错误')
    require(metadata['event_count'] == len(metadata['events']) == int(case == 'candidate'), '捕获事件数错误')
    require(metadata['attempts'] == int(case != 'no_candidate'), '捕获尝试数错误')
    require(metadata['capture_errors'] == int(case in ('cache_limit', 'budget_interrupted')), '捕获错误计数错误')
    require(metadata['truncated'] == (case == 'cache_limit'), '捕获截断标记错误')
    require(metadata['cache_bytes'] <= metadata['cache_limit_bytes'], '捕获实际缓存超限')
    if case == 'candidate':
        require((metadata['n'], metadata['m'], metadata['nnz']) == (2, 3, 4), '合成捕获维数错误')
        event = metadata['events'][0]
        require(event['phase'] == 'recovered_basis_before_assess', '捕获时点错误')
        require(tuple(field['name'] for field in event['arrays']) == ARRAYS, '捕获数组集合错误')
        expected_values = ((0, 2, 4), (0, 2, 1, 2), (1, 2, 3, 4), (5, 6), (7, 8, 9), (1, 2, 3),
                           (4, 5, 6), (0, 1, 1), (0.5, -0.25), (0, 1.5, 2), (-1, 2, 0), (0, 1), (1, 0))
        offset = 8
        require(metadata['byte_order'] in ('little', 'big'), '捕获端序错误')
        order = '<' if metadata['byte_order'] == 'little' else '>'
        for field, values in zip(event['arrays'], expected_values):
            width, code = {'int64': (8, 'q'), 'float64': (8, 'd'), 'uint8': (1, 'B')}[field['dtype']]
            require(field['offset'] == offset and field['width'] == width and
                    field['count'] == len(values) and field['shape'] == [len(values)], '捕获数组布局错误')
            actual = struct.unpack_from(order + str(len(values)) + code, binary, offset)
            require(actual == values, '捕获数组原值回读错误：' + field['name'])
            offset += width * len(values)
        require(offset == len(binary) == 8 + metadata['payload_bytes'], '捕获数组未完整覆盖二进制')
        require(metadata['cache_bytes'] == metadata['fixed_cache_bytes'] + metadata['payload_bytes'], '捕获缓存账本不符')
    else:
        require(len(binary) == 8, '无完整事件仍输出载荷')
    return {'case': case, 'passed': True, 'status': metadata['status'], 'event_count': metadata['event_count']}


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), '输出必须是工作区内的新目录')
    plan_path = DATA / 'v71-plan.json'
    require(sha(plan_path) == PLAN_SHA, '计划身份变化')
    plan = read(plan_path)
    reference_manifest_path = ROOT / plan['reference_build_manifest_file']
    reference_executable = ROOT / plan['reference_solver_file']
    require(sha(reference_manifest_path) == plan['reference_build_manifest_sha256'], 'v70 参考构建身份变化')
    require(sha(reference_executable) == plan['reference_solver_sha256'], 'v70 参考程序身份变化')
    require(read(reference_manifest_path)['executable_sha256'] == plan['reference_solver_sha256'], '参考构建与程序不符')
    require_hashes({str(DATA / name): value for name, value in plan['frozen_helpers'].items()})
    parent_path = DATA / 'ipm-v66-identity.json'
    baseline_path = DATA / plan['baseline_identity_file']
    evidence_path = DATA / plan['evidence_file']
    verification_path = DATA / plan['baseline_verification_file']
    require(sha(parent_path) == plan['parent_identity_sha256'], '父版身份变化')
    require(sha(baseline_path) == plan['baseline_identity_sha256'], '生产基线身份变化')
    require(sha(evidence_path) == plan['evidence_sha256'], '既有核验证据变化')
    require(sha(verification_path) == plan['baseline_verification_sha256'], '开始状态核验变化')
    parent, baseline = read(parent_path), read(baseline_path)
    source_archive = DATA / 'ipm-v66-source.zip'
    require(sha(source_archive) == parent['source_zip_sha256'] == plan['parent_source_sha256'], '冻结来源包变化')
    require(baseline['files'] == parent['files'] and baseline['libraries'] == parent['libraries'], '生产与冻结父版不一致')
    sources = {str(ROOT / name): value for name, value in baseline['files'].items()}
    libraries = {str(ROOT / name): value for name, value in baseline['libraries'].items()}
    require(len(sources) == 95 and len(libraries) == 17, '来源或库集合变化')
    require_hashes(sources)
    require_hashes(libraries)
    production = ROOT / 'build/release/solver.exe'
    frozen_parent = ROOT / '.tools/solver-ipm-v66.exe'
    require(sha(production) == baseline['solver_sha256'] == plan['parent_solver_sha256'], '生产程序变化')
    require(sha(frozen_parent) == parent['solver_sha256'] == plan['parent_solver_sha256'], '冻结父程序变化')
    compiler = Path(parent['compiler'])
    require(sha(compiler) == parent['compiler_sha256'], '编译器变化')
    build_dir = ROOT / 'build/release'
    eigen = ROOT / plan['eigen_include_directory']
    require((eigen / 'Eigen/SparseLU').is_file(), 'Eigen 稀疏分解头文件缺失')
    ninja = ROOT / '.tools/ninja-1.13.2/ninja.exe'
    command_text = subprocess.check_output([str(ninja), '-C', str(build_dir), '-t', 'commands', 'solver.exe'],
                                           text=True, encoding='utf-8')
    lines = command_text.splitlines()
    units = {'ipm': 'ipm.cpp', 'main': 'main.cpp', 'highs': 'highs_backend.cpp', 'pdlp': 'pdlp.cpp'}
    originals = {}
    for unit, filename in units.items():
        matches = [split_windows(line) for line in lines if ' -c ' in line and
                   line.replace('\\', '/').endswith('/src/' + filename)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, '实际编译命令不唯一：' + unit)
        originals[unit] = matches[0]
    links = [line for line in lines if ' -o solver.exe ' in line]
    require(links == [parent['link_command']], '实际链接命令与父版不同')
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == '/c', '链接包装变化')
    inner = outer[2]
    require(inner.startswith('cd . && ') and inner.endswith(' && cd .'), '链接目录包装变化')
    original_link = split_windows(inner[len('cd . && '):-len(' && cd .')])
    explicit = [resolve(v, build_dir) for v in original_link if not v.startswith('-') and
                Path(v).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
    regularized = build_dir / 'CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj'
    helpers = [DATA / name for name in ('v70-basis-recovery.hpp', 'v70-basis-instrumentation.py', 'v70-run.py',
                                       'v71-basis-capture.hpp', 'v71-basis-instrumentation.py',
                                       'v71-basis-capture-self-test.cpp', 'v71-run.py')]
    protected = file_hashes([production, frozen_parent, compiler, ninja, plan_path, baseline_path, parent_path,
                            source_archive, BASE_PATH, Path(__file__), evidence_path, verification_path, regularized,
                            reference_manifest_path, reference_executable,
                            build_dir / 'build.ninja', build_dir / 'CMakeFiles/rules.ninja', *explicit, *helpers])
    output.mkdir(parents=True)
    frozen = output / 'frozen'
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent['files']), '冻结包成员不一致')
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), '冻结包路径越界')
        archive.extractall(frozen)
    frozen_files = {str(frozen / name): value for name, value in parent['files'].items()}
    require_hashes(frozen_files)
    (output / 'ninja-commands.txt').write_text(command_text, encoding='utf-8')
    instrument = runpy.run_path(str(DATA / 'v71-basis-instrumentation.py'))
    source_path = frozen / 'src/ipm.cpp'
    original = source_path.read_text(encoding='utf-8')
    checks = instrument['self_check'](original)
    transformed, anchors = instrument['transform'](original)
    patched = output / 'ipm-basis.cpp'
    patched.write_text(transformed, encoding='utf-8')
    save(output / 'static-check.json', checks)
    save(output / 'transforms.json', {'source': str(source_path), 'source_sha256': sha(source_path),
                                     'patched': str(patched), 'patched_sha256': sha(patched), 'anchors': anchors})
    commands, objects = {}, {}
    for unit, filename in units.items():
        obj = output / (unit + '.obj')
        objects[unit] = obj
        command = compile_arguments(originals[unit], patched if unit == 'ipm' else frozen / 'src' / filename,
                                    obj, frozen, ROOT)
        command.extend(['-I' + str(DATA), '-isystem', str(eigen), '-DEIGEN_MPL2_ONLY', '-DEIGEN_DONT_PARALLELIZE'])
        commands[unit] = command
    replacements = {'libsolver_ipm.a': [str(objects['ipm']), str(regularized)],
                    'libsolver_highs.a': [str(objects['highs'])], 'libsolver_pdlp.a': [str(objects['pdlp'])]}
    main_original = originals['main'][originals['main'].index('-o') + 1]
    replacements[main_original] = [str(objects['main'])]
    link, counts = [], dict.fromkeys(replacements, 0)
    for value in original_link:
        if value in replacements:
            link.extend(replacements[value])
            counts[value] += 1
        elif not value.startswith('-Wl,--out-implib,'):
            link.append(value)
    require(all(count == 1 for count in counts.values()), '四处独立链接替换必须唯一')
    executable = output / 'solver-diagnostic.exe'
    link[link.index('-o') + 1] = str(executable)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('SOLVER_V') and key != 'OPENBLAS_CORETYPE'}
    environment['PATH'] = str(compiler.parent) + os.pathsep + environment.get('PATH', '')
    compiled_inputs = {}
    for unit in units:
        execute(commands[unit], build_dir, output / (unit + '-compile.json'), environment)
        compiled_inputs.update(file_hashes(dependency_paths(Path(str(objects[unit]) + '.d'), build_dir)))
    final_inputs = [resolve(value, build_dir) for value in link if not value.startswith('-') and
                    Path(value).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
    before_link = file_hashes(final_inputs)
    execute(link, build_dir, output / 'link.json', environment)
    self_exe = output / 'capture-self-test.exe'
    self_command = [str(compiler), '-O3', '-DNDEBUG', '-static', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic',
                    '-MMD', '-MF', str(output / 'capture-self-test.d'), '-I' + str(DATA), '-isystem', str(eigen),
                    '-DEIGEN_MPL2_ONLY', '-DEIGEN_DONT_PARALLELIZE',
                    str(DATA / 'v71-basis-capture-self-test.cpp'), '-o', str(self_exe)]
    execute(self_command, build_dir, output / 'capture-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / 'capture-self-test.d', build_dir)))
    self_checks = []
    for case in CASES:
        rows = {}
        for mode in ('off', 'on'):
            env = environment.copy()
            prefix = 'capture-self-' + case + '-' + mode
            if mode == 'on':
                env[plan['capture_env']] = str(output / prefix)
            rows[mode] = execute([str(self_exe), '--case', case], ROOT, output / (prefix + '-run.json'), env)
        self_checks.append(verify_self_capture(output, case, rows))
    save(output / 'capture-self-check.json', {'passed': True, 'off_on_pairs': len(CASES), 'cases': self_checks, 'lp_runs': 0})
    for group in (sources, libraries, protected, frozen_files, before_link, compiled_inputs):
        require_hashes(group)
    outputs = [executable, self_exe, *objects.values(), patched, output / 'ninja-commands.txt',
               *output.glob('*.json'), *output.glob('*.bin')]
    manifest = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'diagnostic_only': True,
                'experimental_candidate_only': False, 'production_changes': False,
                'plan_sha256': PLAN_SHA, 'baseline_identity_sha256': sha(baseline_path), 'parent_identity_sha256': sha(parent_path),
                'production_executable': str(production), 'production_sha256': sha(production),
                'frozen_parent_executable': str(frozen_parent), 'frozen_parent_sha256': sha(frozen_parent),
                'reference_build_manifest': str(reference_manifest_path), 'reference_build_manifest_sha256': sha(reference_manifest_path),
                'reference_executable': str(reference_executable), 'reference_sha256': sha(reference_executable),
                'executable': str(executable), 'executable_sha256': sha(executable), 'frozen_files': parent['files'],
                'current_files': baseline['files'], 'actual_libraries': baseline['libraries'],
                'protected_before_and_after': protected, 'link_explicit_before_and_after': before_link,
                'compile_inputs_after_and_final': compiled_inputs, 'outputs': file_hashes(outputs),
                'commands': {'cwd': str(build_dir), 'original_compile': originals, 'original_link': original_link,
                             'compile': commands, 'link': link, 'self_test': self_command},
                'implicit_link_inputs_verified': False, 'lp_runs': 0}
    save(output / 'build-manifest.json', manifest)
    print(json.dumps({'passed': True, 'compiled_units': list(units), 'solver_sha256': sha(executable),
                      'capture_self_test_pairs': len(CASES), 'lp_runs': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.tools/v71-diagnostic')
    parser.add_argument('--self-test', action='store_true', help='仅检查冻结源插桩及逆变换，不编译或运行求解器')
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
