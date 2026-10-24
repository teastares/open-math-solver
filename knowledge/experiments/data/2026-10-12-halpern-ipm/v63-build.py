"""从冻结v62独立构建宽方向诊断；当前生产来源和产物只读。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = 'f23be965cc316b4aafce25d435314e3280c5abaf7dce6d9779de5f359b4b738c'
BASE_PATH = DATA / 'v47-profile-build.py'
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[k] for k in ('sha', 'require', 'save'))
split_windows, resolve = (BASE[k] for k in ('split_windows', 'resolve'))
file_hashes, require_hashes = (BASE[k] for k in ('file_hashes', 'require_hashes'))
compile_arguments, execute, dependency_paths = (BASE[k] for k in ('compile_arguments', 'execute', 'dependency_paths'))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), '输出必须是工作区内的新目录')
    plan_path = DATA / 'v63-plan.json'
    require(sha(plan_path) == PLAN_SHA, '计划身份变化')
    plan = read(plan_path)
    parent_path = DATA / 'ipm-v62-identity.json'
    baseline_path = DATA / plan['baseline_identity_file']
    require(sha(parent_path) == plan['parent_identity_sha256'], '父版身份变化')
    require(sha(baseline_path) == plan['baseline_identity_sha256'], '当前基线身份变化')
    require(sha(DATA / plan['evidence_file']) == plan['evidence_sha256'], '上一轮核验证据变化')
    require(sha(DATA / plan['baseline_verification_file']) == plan['baseline_verification_sha256'], '本轮开始状态核验变化')
    runner_check = read(DATA / 'v63-runner-self-check-attempt3.json')
    require(runner_check['returncode'] == 0 and runner_check['runner_sha256'] == sha(DATA / 'v63-run.py'), '实际运行器自检与待冻来源不符')
    parent, baseline = read(parent_path), read(baseline_path)
    source_archive = DATA / 'ipm-v62-source.zip'
    require(sha(source_archive) == parent['source_zip_sha256'] == plan['parent_source_sha256'], '来源包变化')
    sources = {str(ROOT / name): value for name, value in baseline['files'].items()}
    libraries = {str(ROOT / name): value for name, value in baseline['libraries'].items()}
    require(len(sources) == 95 and len(libraries) == 17, '当前来源或库集合变化')
    require_hashes(sources)
    require_hashes(libraries)
    external = [n for n in baseline['libraries'] if not Path(n).name.startswith('libsolver_')]
    require(len(external) == 8 and all(baseline['libraries'][n] == parent['libraries'][n] for n in external), '第三方库不同')
    production = ROOT / 'build/release/solver.exe'
    frozen_parent = ROOT / '.tools/solver-ipm-v62.exe'
    require(sha(production) == baseline['solver_sha256'], '当前程序变化')
    require(sha(frozen_parent) == parent['solver_sha256'] == plan['parent_solver_sha256'], '冻结父程序变化')
    compiler = Path(parent['compiler'])
    require(sha(compiler) == parent['compiler_sha256'], '编译器变化')
    build_dir = ROOT / 'build/release'
    ninja = ROOT / '.tools/ninja-1.13.2/ninja.exe'
    command_text = subprocess.check_output([str(ninja), '-C', str(build_dir), '-t', 'commands', 'solver.exe'], text=True, encoding='utf-8')
    lines = command_text.splitlines()
    units = {'ipm': 'ipm.cpp', 'main': 'main.cpp', 'highs': 'highs_backend.cpp', 'pdlp': 'pdlp.cpp'}
    originals = {}
    for unit, filename in units.items():
        matches = [split_windows(line) for line in lines if ' -c ' in line and line.replace('\\', '/').endswith('/src/' + filename)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, '实际编译命令不唯一：' + unit)
        originals[unit] = matches[0]
    links = [line for line in lines if ' -o solver.exe ' in line]
    require(links == [parent['link_command']], '链接命令与父版不同')
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == '/c', '链接包装变化')
    inner = outer[2]
    require(inner.startswith('cd . && ') and inner.endswith(' && cd .'), '链接目录包装变化')
    original_link = split_windows(inner[len('cd . && '):-len(' && cd .')])
    explicit = [resolve(v, build_dir) for v in original_link if not v.startswith('-') and Path(v).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
    regularized = build_dir / 'CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj'
    helpers = [DATA / n for n in ('v63-hsd-diagnostic.hpp', 'v63-hsd-instrumentation.py', 'v63-hsd-self-test.cpp', 'v63-analyze-hsd.py', 'v63-run.py', 'v61-analyze-hsd.py', 'v53-analyze-hsd.py')]
    protected = file_hashes([production, frozen_parent, compiler, ninja, plan_path, baseline_path, parent_path, source_archive,
        BASE_PATH, Path(__file__), DATA / plan['evidence_file'], DATA / plan['baseline_verification_file'],
        DATA / 'v63-plan-revision.json', DATA / 'v63-runner-self-check-attempt3.json',
        regularized, build_dir / 'build.ninja', build_dir / 'CMakeFiles/rules.ninja', *explicit, *helpers])
    output.mkdir(parents=True)
    frozen = output / 'frozen'
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent['files']), '父包成员不一致')
        require(all((frozen / n).resolve().is_relative_to(frozen.resolve()) for n in archive.namelist()), '父包路径越界')
        archive.extractall(frozen)
    require_hashes({str(frozen / n): value for n, value in parent['files'].items()})
    (output / 'ninja-commands.txt').write_text(command_text, encoding='utf-8')
    instrument = runpy.run_path(str(DATA / 'v63-hsd-instrumentation.py'))
    source_path = frozen / 'src/ipm.cpp'
    original = source_path.read_text(encoding='utf-8')
    checks = instrument['self_check'](original)
    transformed, anchors = instrument['transform'](original)
    patched = output / 'ipm-diagnostic.cpp'
    patched.write_text(transformed, encoding='utf-8')
    save(output / 'static-check.json', checks)
    save(output / 'transforms.json', {'source': str(source_path), 'source_sha256': sha(source_path),
        'patched': str(patched), 'patched_sha256': sha(patched), 'anchors': anchors})
    commands, objects = {}, {}
    for unit, filename in units.items():
        obj = output / (unit + '.obj')
        objects[unit] = obj
        cmd = compile_arguments(originals[unit], patched if unit == 'ipm' else frozen / 'src' / filename, obj, frozen, ROOT)
        cmd.append('-I' + str(DATA))
        commands[unit] = cmd
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
    require(all(n == 1 for n in counts.values()), '共享结果结构的四处链接替换必须唯一')
    executable = output / 'solver-diagnostic.exe'
    link[link.index('-o') + 1] = str(executable)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('SOLVER_V') and k != 'OPENBLAS_CORETYPE'}
    environment['PATH'] = str(compiler.parent) + os.pathsep + environment.get('PATH', '')
    compiled_inputs = {}
    for unit in units:
        execute(commands[unit], build_dir, output / (unit + '-compile.json'), environment)
        compiled_inputs.update(file_hashes(dependency_paths(Path(str(objects[unit]) + '.d'), build_dir)))
    final_inputs = [resolve(v, build_dir) for v in link if not v.startswith('-') and Path(v).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
    before_link = file_hashes(final_inputs)
    execute(link, build_dir, output / 'link.json', environment)
    self_exe = output / 'hsd-self-test.exe'
    self_command = [str(compiler), '-O3', '-DNDEBUG', '-static', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic',
        '-MMD', '-MF', str(output / 'hsd-self-test.d'), '-I' + str(DATA), '-I' + str(frozen / 'include'),
        str(DATA / 'v63-hsd-self-test.cpp'), '-o', str(self_exe)]
    execute(self_command, build_dir, output / 'hsd-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / 'hsd-self-test.d', build_dir)))
    for case in ('success', 'failed_candidate', 'failed_solve', 'no_trigger', 'event_limit', 'cache_limit', 'step_limit'):
        for mode in ('off', 'on'):
            env = environment.copy()
            prefix = 'hsd-self-' + case + '-' + mode
            if mode == 'on':
                env[plan['diagnostic_environment']] = str(output / prefix)
            execute([str(self_exe), '--case', case], ROOT, output / (prefix + '-run.json'), env)
    execute([str(ROOT / '.tools/python-3.14.8/python.exe'), str(DATA / 'v63-analyze-hsd.py'), '--self-test'],
        ROOT, output / 'hsd-math-self-test.json', environment)
    for group in (sources, libraries, protected, before_link, compiled_inputs):
        require_hashes(group)
    outputs = [executable, self_exe, *objects.values(), patched, output / 'ninja-commands.txt',
        *output.glob('*.json'), *output.glob('*.bin')]
    manifest = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'diagnostic_only': True,
        'plan_sha256': PLAN_SHA, 'baseline_identity_sha256': sha(baseline_path), 'parent_identity_sha256': sha(parent_path),
        'production_executable': str(production), 'production_sha256': sha(production),
        'frozen_parent_executable': str(frozen_parent), 'frozen_parent_sha256': sha(frozen_parent),
        'executable': str(executable), 'executable_sha256': sha(executable), 'frozen_files': parent['files'],
        'current_files': baseline['files'], 'actual_libraries': baseline['libraries'],
        'protected_before_and_after': protected, 'link_explicit_before_and_after': before_link,
        'compile_inputs_after_and_final': compiled_inputs, 'outputs': file_hashes(outputs),
        'commands': {'cwd': str(build_dir), 'original_compile': originals, 'original_link': original_link, 'compile': commands, 'link': link, 'self_test': self_command},
        'implicit_link_inputs_verified': False, 'lp_runs': 0}
    save(output / 'build-manifest.json', manifest)
    print(json.dumps({'passed': True, 'compiled_units': list(units), 'solver_sha256': sha(executable), 'lp_runs': 0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.tools/v63-diagnostic')
    build(parser.parse_args().output.resolve())
