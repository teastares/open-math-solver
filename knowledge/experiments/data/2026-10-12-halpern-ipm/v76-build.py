"""从冻结 v66 独立构建受约束近退化转换候选；生产来源、库与程序保持只读。"""
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
PLAN_SHA = 'f3e313b85f17e60060b4af76d4f58c59e1fd553ad8d8eadcd682fa79234ec2e2'
BASE_PATH = DATA / 'v47-profile-build.py'
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[k] for k in ('sha', 'require', 'save'))
split_windows, resolve = (BASE[k] for k in ('split_windows', 'resolve'))
file_hashes, require_hashes = (BASE[k] for k in ('file_hashes', 'require_hashes'))
compile_arguments, execute, dependency_paths = (BASE[k] for k in ('compile_arguments', 'execute', 'dependency_paths'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def static_check():
    require(sha(DATA / 'v76-plan.json') == PLAN_SHA, '计划身份变化')
    plan = read(DATA / 'v76-plan.json')
    require(plan['candidate_limits'] == {'max_input_nonzeros': 8000000, 'max_factor_nonzeros': 32000000,
                                        'max_dimension': 1000000, 'max_selection_work': 1000000000,
                                        'max_repair_pivots': 128, 'max_repair_work': 100000000},
            '候选限额与固定实现不符')
    instrument = runpy.run_path(str(DATA / 'v76-basis-instrumentation.py'))
    checks = instrument['self_check']()
    recovery = (DATA / 'v76-basis-recovery.hpp').read_text(encoding='utf-8')
    require(recovery.count('::solver::v76_basis_repair::repair(') == 1 and
            recovery.index('const Eigen::VectorXd dual = lu.transpose().solve(dual_rhs);') <
            recovery.index('::solver::v76_basis_repair::repair(') <
            recovery.index('const auto& candidate_rows = output.repair.selected_rows;'),
            '修复必须只复用一次初始正反回代后的 LU')
    require('output.z[candidate_rows[i]] = output.repair.dual[i];' in recovery and
            'stats.selected_rows = output.repair' not in recovery,
            '最终乘子映射必须与初始选基账本分离')
    repair = (DATA / 'v76-basis-repair.hpp').read_text(encoding='utf-8')
    require(repair.count('refine_point(input, limits, state, initial_lu, work, stats, stats.refined_guard,') == 1 and
            'if (!state.basis_valid) {' in repair and
            repair.index('evaluate(input, limits, state, work, stats, &stats.initial_guard);') <
            repair.index('refine_point(input, limits, state, initial_lu, work, stats, stats.refined_guard,'),
            '初始修正只能在完整入口检查失败后使用同一 LU 一次')
    correction = repair[repair.index('inline Eigen::VectorXd correct_forward('):repair.index('inline void refine_point(')]
    require(correction.count('lu.solve(rhs)') == 1 and 'transpose()' not in correction and
            'factorize(' not in correction and 'analyzePattern(' not in correction and
            repair.count('correct_forward(') == 3,
            '点和方向必须复用同一个单次正向修正内核，不得增加分解或对偶回代')
    point = repair[repair.index('inline void refine_point('):repair.index('struct Event {')]
    require('if (!next.basis_valid)' in point and
            point.index('work.check();\n  state = std::move(next);') >
            point.index('evaluate(input, limits, next, work, stats, &after);') and
            'next.dual.resize(state.dual.size());' in point and
            'k < state.dual.size()' in point,
            '原始点修正须保留空对偶语义，并在完整守卫和预算通过后提交')
    direction = repair[repair.index('auto& forward = stats.direction_forward;'):
                       repair.index('      Event step;')]
    require(direction.count('correct_forward(') == 1 and
            'return direction_state.change[state.selected[k]] - expected;' in direction and
            direction.index('evaluate_direction(input, limits, state, trial, entering, work, stats, trial_state, forward.after);') <
            direction.index('change[row] = expected;') and
            'work.check();\n        direction = std::move(trial);' in direction,
            '方向修正必须使用未覆盖的原始残差，同一守卫和预算通过后才能覆盖结构性方向')
    require('begin_forward(rebuild, "rebuild", stats.pivots + 1, state.selected[entering], entering);' in repair and
            'refine_point(input, limits, next, *next_lu, work, stats, rebuild.after,' in repair,
            '换基后原始回代必须复用同一有限修正机制及其实际新因子')
    transition = repair[repair.index('inline bool phase1_transition('):repair.index('inline void begin_forward(')]
    require(transition.index('if (next.phi < state.phi)') < transition.index('++stats.near_attempts;') and
            'std::abs(state.slack[blocker]) <= state.roundoff[blocker]' in transition and
            'theta <= state.roundoff[released]' in transition and
            'state.roundoff[row] + next.roundoff[row]' in transition and
            'if (change > tolerance) all_rows_near = false;' in transition and
            'if (!stats.blocking_near)' in transition and 'if (!stats.released_near)' in transition and
            'if (!all_rows_near)' in transition,
            '阶段一须优先严格下降，近退化路径必须完整检查原近界和全行变化限制')
    history = repair[repair.index('struct BasisHistory {'):repair.index('inline bool phase1_transition(')]
    require('std::vector<Index> initial_rows;' in history and 'std::vector<BasisSwap> swaps;' in history and
            'std::vector<std::uint64_t> hashes;' in history and
            'std::unique_ptr<std::uint8_t[]> replay;' in history and
            'swaps.reserve(static_cast<std::size_t>(limits.max_pivots));' in history and
            'hashes.reserve(static_cast<std::size_t>(limits.max_pivots) + 1);' in history and
            history.index('if (hashes[version] != hash) continue;') <
            history.index('replay.reset(new std::uint8_t[static_cast<std::size_t>(m)]);') and
            '(replay[row] != 0) != (candidate.position[row] >= 0)' in history and
            'if (same)' in history,
            '历史须用初始集合和有限交换日志，哈希命中后重放并精确比较集合')
    commit = repair[repair.index('const long double point_bound ='):repair.index('      // 映射复制也计费')]
    require(commit.index('if (point_error > point_bound)') < commit.index('near_transition = phase1_transition(') <
            commit.index('basis_history.contains(') <
            commit.index('work.check();\n      basis_history.commit(') < commit.index('state = std::move(next);') <
            commit.index('++stats.pivots;') < commit.index('++stats.phase1_transition.near_accepted;') and
            'if (!next.primal_feasible)' in commit and 'if (!(difference < 0))' in commit and
            'if (!(step.theta > 0))' in repair,
            '数值与目标检查、共同历史判重及预算须先于基提交和接纳计数，原零步及阶段二数值条件保持')
    checks['repair_after_initial_solves'] = True
    checks['initial_and_final_row_maps_separate'] = True
    checks['initial_refinement_only_after_failed_guard'] = True
    checks['single_forward_refinement_same_factor'] = True
    checks['refinement_guard_and_budget_before_commit'] = True
    checks['raw_direction_before_structural_override'] = True
    checks['rebuild_empty_dual_preserved'] = True
    checks['three_forward_contexts_shared'] = True
    checks['phase1_strict_then_constrained_near'] = True
    checks['history_hash_then_exact_membership'] = True
    checks['history_linear_storage_and_reserved_commit'] = True
    checks['transition_and_history_before_commit'] = True
    return checks


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), '输出必须是工作区内的新目录')
    plan_path = DATA / 'v76-plan.json'
    require(sha(plan_path) == PLAN_SHA, '计划身份变化')
    plan = read(plan_path)
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
    helpers = [DATA / name for name in ('v76-basis-recovery.hpp', 'v76-basis-instrumentation.py',
                                       'v76-basis-self-test.cpp', 'v76-run.py',
                                       'v76-basis-repair.hpp', 'v76-repair-self-test.cpp')]
    protected = file_hashes([production, frozen_parent, compiler, ninja, plan_path, baseline_path, parent_path,
                            source_archive, BASE_PATH, Path(__file__), evidence_path, verification_path, regularized,
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
    instrument = runpy.run_path(str(DATA / 'v76-basis-instrumentation.py'))
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
    self_exe = output / 'basis-self-test.exe'
    self_command = [str(compiler), '-O3', '-DNDEBUG', '-static', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic',
                    '-MMD', '-MF', str(output / 'basis-self-test.d'), '-I' + str(DATA), '-isystem', str(eigen),
                    '-DEIGEN_MPL2_ONLY', '-DEIGEN_DONT_PARALLELIZE',
                    str(DATA / 'v76-basis-self-test.cpp'), '-o', str(self_exe)]
    execute(self_command, build_dir, output / 'basis-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / 'basis-self-test.d', build_dir)))
    self_run = execute([str(self_exe)], ROOT, output / 'basis-self-run.json', environment)
    self_check = json.loads(self_run['stdout'])
    require(self_check['passed'] is True, '基恢复内核自测失败')
    save(output / 'basis-self-check.json', {'passed': True, 'details': self_check, 'lp_runs': 0})
    repair_exe = output / 'repair-self-test.exe'
    repair_command = [str(compiler), '-O3', '-DNDEBUG', '-static', '-std=c++20', '-Wall', '-Wextra', '-Wpedantic',
                      '-MMD', '-MF', str(output / 'repair-self-test.d'), '-I' + str(DATA), '-isystem', str(eigen),
                      '-DEIGEN_MPL2_ONLY', '-DEIGEN_DONT_PARALLELIZE',
                      str(DATA / 'v76-repair-self-test.cpp'), '-o', str(repair_exe)]
    execute(repair_command, build_dir, output / 'repair-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / 'repair-self-test.d', build_dir)))
    repair_run = execute([str(repair_exe)], ROOT, output / 'repair-self-run.json', environment)
    repair_check = json.loads(repair_run['stdout'])
    require(repair_check['passed'] is True, '两阶段基修复内核自测失败')
    save(output / 'repair-self-check.json', {'passed': True, 'details': repair_check, 'lp_runs': 0})
    for group in (sources, libraries, protected, frozen_files, before_link, compiled_inputs):
        require_hashes(group)
    outputs = [executable, self_exe, repair_exe, *objects.values(), patched,
               output / 'ninja-commands.txt', *output.glob('*.json')]
    manifest = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'diagnostic_only': False,
                'experimental_candidate_only': True, 'production_changes': False,
                'plan_sha256': PLAN_SHA, 'baseline_identity_sha256': sha(baseline_path), 'parent_identity_sha256': sha(parent_path),
                'production_executable': str(production), 'production_sha256': sha(production),
                'frozen_parent_executable': str(frozen_parent), 'frozen_parent_sha256': sha(frozen_parent),
                'executable': str(executable), 'executable_sha256': sha(executable), 'frozen_files': parent['files'],
                'current_files': baseline['files'], 'actual_libraries': baseline['libraries'],
                'protected_before_and_after': protected, 'link_explicit_before_and_after': before_link,
                'compile_inputs_after_and_final': compiled_inputs, 'outputs': file_hashes(outputs),
                'commands': {'cwd': str(build_dir), 'original_compile': originals, 'original_link': original_link,
                             'compile': commands, 'link': link, 'self_test': self_command,
                             'repair_self_test': repair_command},
                'implicit_link_inputs_verified': False, 'lp_runs': 0}
    save(output / 'build-manifest.json', manifest)
    print(json.dumps({'passed': True, 'compiled_units': list(units), 'solver_sha256': sha(executable),
                      'kernel_self_tests': self_check, 'repair_self_tests': repair_check,
                      'lp_runs': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.tools/v76-basis')
    parser.add_argument('--self-test', action='store_true', help='仅检查冻结源插桩及逆变换，不编译或运行求解器')
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
