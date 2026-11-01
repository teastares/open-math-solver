"""从冻结 v66 独立构建回代预算批量候选；生产来源、库与程序保持只读。"""
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
PLAN_SHA = 'c57b41bf3bb129e20bda63304f861db2fef958a544a5e7d0a00d24a56df8ab4f'
BASE_PATH = DATA / 'v47-profile-build.py'
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[k] for k in ('sha', 'require', 'save'))
split_windows, resolve = (BASE[k] for k in ('split_windows', 'resolve'))
file_hashes, require_hashes = (BASE[k] for k in ('file_hashes', 'require_hashes'))
compile_arguments, execute, dependency_paths = (BASE[k] for k in ('compile_arguments', 'execute', 'dependency_paths'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def link_arguments(line, build_dir, compiler):
    outer = split_windows(line)
    require(len(outer) == 3 and outer[1].lower() == '/c', '链接包装变化')
    inner = outer[2]
    require(inner.startswith('cd . && ') and inner.endswith(' && cd .'), '链接目录包装变化')
    result = split_windows(inner[len('cd . && '):-len(' && cd .')])
    require(resolve(result[0], build_dir) == compiler, '链接编译器变化')
    return result


def execute_normal(command, cwd, destination, environment):
    # 保存原始字节供双程序比较；JSON日志沿用现有构建记录的字段。
    try:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, timeout=300)
        stdout, stderr = process.stdout, process.stderr
        row = {'command': command, 'cwd': str(cwd), 'returncode': process.returncode,
               'stdout': stdout.decode('utf-8', errors='replace'),
               'stderr': stderr.decode('utf-8', errors='replace'), 'wall_timeout': False}
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b'', error.stderr or b''
        row = {'command': command, 'cwd': str(cwd), 'returncode': None,
               'stdout': stdout.decode('utf-8', errors='replace'),
               'stderr': stderr.decode('utf-8', errors='replace'), 'wall_timeout': True}
    save(destination, row)
    require(row['returncode'] == 0, f'正规方程独立自测失败，已保留 {destination}')
    require(stdout.decode('utf-8') == row['stdout'], '正规方程自测输出必须是有效UTF-8')
    return row, stdout


def static_check():
    require(sha(DATA / 'v81-plan.json') == PLAN_SHA, '计划身份变化')
    plan = read(DATA / 'v81-plan.json')
    require(plan['candidate_limits'] == {'max_input_nonzeros': 8000000, 'max_factor_nonzeros': 32000000,
                                        'max_dimension': 1000000, 'max_selection_work': 1000000000,
                                        'max_repair_pivots': 512, 'max_repair_work': 400000000,
                                        'max_repair_factor_nonzeros': 64000000, 'max_eta_updates': 8},
            '候选限额与固定实现不符')
    instrument = runpy.run_path(str(DATA / 'v81-basis-instrumentation.py'))
    checks = instrument['self_check']()
    recovery = (DATA / 'v81-basis-recovery.hpp').read_text(encoding='utf-8')
    require(recovery.count('::solver::v81_basis_repair::repair(') == 1 and
            recovery.index('const Eigen::VectorXd dual = lu.transpose().solve(dual_rhs);') <
            recovery.index('::solver::v81_basis_repair::repair(') <
            recovery.index('const auto& candidate_rows = output.repair.selected_rows;'),
            '修复必须只复用一次初始正反回代后的 LU')
    require('output.z[candidate_rows[i]] = output.repair.dual[i];' in recovery and
            'stats.selected_rows = output.repair' not in recovery,
            '最终乘子映射必须与初始选基账本分离')
    require('Index max_factor_nonzeros = 32000000;' in recovery and
            'Index max_repair_factor_nonzeros = 64000000;' in recovery and
            'repair_limits.max_factor_nonzeros = limits.max_repair_factor_nonzeros;' in recovery and
            'repair_limits.max_single_factor_nonzeros = limits.max_factor_nonzeros;' in recovery and
            'Index max_eta_updates = 8;' in recovery and
            'repair_limits.max_eta_updates = limits.max_eta_updates;' in recovery,
            '原因子额度须保持，逆更新链上限须独立传入且默认为八')
    repair = (DATA / 'v81-basis-repair.hpp').read_text(encoding='utf-8')
    require('Index max_pivots = 512;' in repair and 'Index max_repair_work = 400000000;' in repair and
            'Index max_factor_nonzeros = 64000000;' in repair and
            'Index max_single_factor_nonzeros = 32000000;' in repair and
            'Index max_eta_updates = 8;' in repair,
            '修复内核统一资源额度必须与本轮计划对应')
    factor_counts = repair[repair.index('const Index lower = static_cast<Index>(pending.new_base->nnzL())'):
                           repair.index('    work_.account(lower + upper);')]
    require(factor_counts.index('stats_.last_factor_nonzeros = lower + upper;') <
            factor_counts.index('stats_.factor_nonzeros += stats_.last_factor_nonzeros;') <
            factor_counts.index('if (stats_.last_factor_nonzeros > limits_.max_single_factor_nonzeros)') <
            factor_counts.index('if (stats_.factor_nonzeros > limits_.max_factor_nonzeros)'),
            '新因子须先保存单次及累计后验计数，再依次核单因子与累计上限')
    require(repair.count('refine_point(input, limits, state, factors, work, stats, stats.refined_guard,') == 1 and
            'if (!state.basis_valid) {' in repair and
            repair.index('evaluate(input, limits, state, work, stats, &stats.initial_guard);') <
            repair.index('refine_point(input, limits, state, factors, work, stats, stats.refined_guard,') and
            'EtaFactor<Lu> factors(initial_lu, limits, work, stats);' in repair,
            '初始修正只能在完整入口检查失败后使用同一 LU 一次')
    correction = repair[repair.index('inline Eigen::VectorXd correct_forward('):repair.index('inline void refine_point(')]
    require(correction.count('lu.solve(rhs)') == 1 and 'transpose()' not in correction and
            'factorize(' not in correction and 'analyzePattern(' not in correction and
            repair.count('correct_forward(') == 3 and 'if (work.budget_failed()) throw;' in correction,
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
    normalized_direction = '\n'.join(line.strip() for line in direction.splitlines())
    require(direction.count('correct_forward(') == 1 and
            'return direction_state.change[state.selected[k]] - expected;' in direction and
            direction.index('evaluate_direction(input, limits, state, trial, entering, work, stats, trial_state, forward.after);') <
            direction.index('change[row] = expected;') and
            'work.check();\ndirection = std::move(trial);' in normalized_direction,
            '方向修正必须使用未覆盖的原始残差，同一守卫和预算通过后才能覆盖结构性方向')
    candidates = repair[repair.index('      std::vector<Index> candidates;'):
                        repair.index('      auto& change = direction_state.change;')]
    trial_loop = candidates[candidates.index('      for (Index ordinal = 0; ordinal < trial_count; ++ordinal) {'):
                            candidates.index('      if (!direction_found) {')]
    require('if (!input.inequality[state.selected[k]] || !(reduced[k] < 0)) continue;' in candidates and
            'std::sort(candidates.begin(), candidates.end(),' in candidates and
            'reduced[first] < reduced[second]' in candidates and
            'state.selected[first] < state.selected[second]' in candidates and
            'transpose().solve' not in trial_loop and
            repair.index('reduced = factors.transpose().solve(rhs);') <
            repair.index('      for (Index ordinal = 0; ordinal < trial_count; ++ordinal) {'),
            '阶段一须在同一基定价一次，按负价与原行次序搜索，不为每候选重新定价')
    require('const Index trial_count = phase1 ? static_cast<Index>(candidates.size()) : 1;' in candidates and
            'begin_search_trial(stats.search, ordinal, state.selected[entering], entering, reduced[entering]);' in trial_loop and
            'stats.last_right_derivative.reset(); stats.last_step.reset();' in trial_loop and
            'for (Index k = 0; k < n; ++k) { work.take(); rhs[k] = k == entering ? -1 : 0; }' in trial_loop and
            'if (!(derivative < 0)) { ++stats.search.non_descent; stats.search.status = "non_descent"; continue; }' in trial_loop and
            'phase1_step(' not in trial_loop and
            'throw Stop{"direction_search_exhausted"};' in candidates and
            repair.count('++stats.iterations;') == 1 and
            repair.index('++stats.iterations;') < repair.index('      for (Index ordinal = 0; ordinal < trial_count; ++ordinal) {'),
            '非下降只能跳过当前候选，每候选工作受预算约束，阶段二和外层迭代计数保持原义')
    require(repair.index('++stats.search.descent;') <
            repair.index('step = phase1_step(input, state, change, derivative, work);') and
            'stats.search.step_started = true;' in repair,
            '真实右导数严格下降后才可开始原阶段一比值检验')
    require('begin_forward(rebuild, "rebuild", stats.pivots + 1, state.selected[entering], entering);' in repair and
            'refine_point(input, limits, next, pending, work, stats, rebuild.after,' in repair and
            'auto pending = factors.prepare(input, state, next, entering, direction);' in repair and
            'next.x = pending.solve(rhs);' in repair,
            '换基点须使用独立pending算子及同一有限修正机制')
    eta_impl = repair[repair.index('class EtaFactor {'):repair.index('inline void begin_forward(')]
    require('entry.u[col] = -direction[col];' in eta_impl and
            'const long double difference = new_value - old_value;' in eta_impl and
            'const long double alpha = 1 + product_sum;' in eta_impl and
            'if (static_cast<Index>(chain_.size()) >= limits_.max_eta_updates)' in eta_impl and
            eta_impl.count('factorize(matrix)') == 1 and eta_impl.count('++stats_.lu_factor_calls;') == 1,
            '行替换必须使用已验方向和原行差，满链才转完整LU')
    require(eta_impl.index('Eigen::VectorXd result = base->solve(rhs);') <
            eta_impl.index('for (const auto& entry : chain_) apply(result, entry, false);') <
            eta_impl.index('if (pending && pending->entry) apply(result, *pending->entry, false);') and
            eta_impl.index('if (pending && pending->entry) apply(transformed, *pending->entry, true);') <
            eta_impl.index('for (std::size_t k = chain_.size(); k > 0; --k) apply(transformed, chain_[k - 1], true);') <
            eta_impl.index('Eigen::VectorXd result = base->transpose().solve(transformed);'),
            '正向须base后正序更新，转置须逆序更新后base转置求解')
    eta_commit = eta_impl[eta_impl.index('void commit(Pending& pending) noexcept {'):eta_impl.index(' private:')]
    require('chain_.reserve(static_cast<std::size_t>(limits.max_eta_updates));' in eta_impl and
            'pending.retired_chain.reserve(static_cast<std::size_t>(limits_.max_eta_updates));' in eta_impl and
            'owned_base_.swap(pending.new_base);' in eta_commit and
            'chain_.swap(pending.retired_chain);' in eta_commit and
            'work_.' not in eta_commit and 'reserve(' not in eta_commit and 'resize(' not in eta_commit and
            'EtaFactor(EtaFactor&&) = delete;' in eta_impl and 'EtaFactor* owner;' in eta_impl,
            'pending须借稳定owner，提交预留槽并退休旧资源，提交段不得分配或检查预算')
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
            commit.index('work.check();\n      basis_history.commit(') < commit.index('factors.commit(pending);') <
            commit.index('state = std::move(next);') <
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
    checks['single_and_cumulative_factor_limits_separate'] = True
    checks['factor_posterior_counts_before_limits'] = True
    checks['phase1_pricing_once_per_basis'] = True
    checks['fixed_negative_price_candidate_order'] = True
    checks['non_descent_skips_current_candidate'] = True
    checks['true_descent_before_phase1_ratio'] = True
    checks['phase2_single_direction_kept'] = True
    checks['eta_row_inverse_update_and_fixed_chain'] = True
    checks['eta_forward_and_transpose_order'] = True
    checks['pending_operator_original_refinement'] = True
    checks['pending_owned_resources_retired_after_atomic_commit'] = True
    checks['eta_budget_exceptions_propagate'] = True
    normal_instrument = runpy.run_path(str(DATA / 'v81-normal-instrumentation.py'))
    checks['normal_budget'] = normal_instrument['self_check']()
    return checks


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), '输出必须是工作区内的新目录')
    plan_path = DATA / 'v81-plan.json'
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
    command_text = subprocess.check_output([str(ninja), '-C', str(build_dir), '-t', 'commands',
                                            'solver.exe', 'test_normal_cholesky.exe'],
                                           text=True, encoding='utf-8')
    lines = command_text.splitlines()
    units = {'ipm': 'ipm.cpp', 'main': 'main.cpp', 'highs': 'highs_backend.cpp', 'pdlp': 'pdlp.cpp',
             'normal': 'normal_cholesky.cpp'}
    originals = {}
    for unit, filename in units.items():
        matches = [split_windows(line) for line in lines if ' -c ' in line and
                   line.replace('\\', '/').endswith('/src/' + filename)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, '实际编译命令不唯一：' + unit)
        originals[unit] = matches[0]
    links = [line for line in lines if ' -o solver.exe ' in line]
    require(links == [parent['link_command']], '实际链接命令与父版不同')
    original_link = link_arguments(links[0], build_dir, compiler)
    normal_test_compiles = [split_windows(line) for line in lines if ' -c ' in line and
                            line.replace('\\', '/').endswith('/tests/test_normal_cholesky.cpp')]
    require(len(normal_test_compiles) == 1 and resolve(normal_test_compiles[0][0], build_dir) == compiler,
            '正规方程自测编译模板不唯一')
    normal_test_links = [line for line in lines if ' -o test_normal_cholesky.exe ' in line]
    require(len(normal_test_links) == 1, '正规方程自测链接模板不唯一')
    normal_original_link = link_arguments(normal_test_links[0], build_dir, compiler)
    explicit = [resolve(v, build_dir) for v in original_link if not v.startswith('-') and
                Path(v).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
    normal_libraries = [resolve(v, build_dir) for v in normal_original_link if not v.startswith('-') and
                        Path(v).suffix.lower() in ('.a', '.lib')]
    regularized = build_dir / 'CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj'
    helpers = [DATA / name for name in ('v81-basis-recovery.hpp', 'v81-basis-instrumentation.py',
                                       'v81-basis-self-test.cpp', 'v81-run.py',
                                       'v81-basis-repair.hpp', 'v81-repair-self-test.cpp',
                                       'v81-normal-instrumentation.py', 'v81-normal-self-test.cpp')]
    protected = file_hashes([production, frozen_parent, compiler, ninja, plan_path, baseline_path, parent_path,
                            source_archive, BASE_PATH, Path(__file__), evidence_path, verification_path, regularized,
                            build_dir / 'build.ninja', build_dir / 'CMakeFiles/rules.ninja',
                            *explicit, *normal_libraries, *helpers])
    output.mkdir(parents=True)
    frozen = output / 'frozen'
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent['files']), '冻结包成员不一致')
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), '冻结包路径越界')
        archive.extractall(frozen)
    frozen_files = {str(frozen / name): value for name, value in parent['files'].items()}
    require_hashes(frozen_files)
    (output / 'ninja-commands.txt').write_text(command_text, encoding='utf-8')
    instrument = runpy.run_path(str(DATA / 'v81-basis-instrumentation.py'))
    source_path = frozen / 'src/ipm.cpp'
    original = source_path.read_text(encoding='utf-8')
    checks = instrument['self_check'](original)
    transformed, anchors = instrument['transform'](original)
    patched = output / 'ipm-basis.cpp'
    patched.write_text(transformed, encoding='utf-8')
    save(output / 'static-check.json', checks)
    save(output / 'transforms.json', {'source': str(source_path), 'source_sha256': sha(source_path),
                                     'patched': str(patched), 'patched_sha256': sha(patched), 'anchors': anchors})
    normal_instrument = runpy.run_path(str(DATA / 'v81-normal-instrumentation.py'))
    normal_source = frozen / 'src/normal_cholesky.cpp'
    normal_original = normal_source.read_text(encoding='utf-8')
    normal_checks = normal_instrument['self_check'](normal_original)
    normal_transformed, normal_anchors = normal_instrument['transform'](normal_original)
    normal_patched = output / 'normal-budget.cpp'
    normal_patched.write_text(normal_transformed, encoding='utf-8')
    save(output / 'normal-static-check.json', normal_checks)
    save(output / 'normal-transforms.json', {'source': str(normal_source), 'source_sha256': sha(normal_source),
                                            'patched': str(normal_patched), 'patched_sha256': sha(normal_patched),
                                            'anchors': normal_anchors})
    commands, objects = {}, {}
    for unit, filename in units.items():
        obj = output / (unit + '.obj')
        objects[unit] = obj
        source = patched if unit == 'ipm' else normal_patched if unit == 'normal' else frozen / 'src' / filename
        command = compile_arguments(originals[unit], source, obj, frozen, ROOT)
        command.extend(['-I' + str(DATA), '-isystem', str(eigen), '-DEIGEN_MPL2_ONLY', '-DEIGEN_DONT_PARALLELIZE'])
        commands[unit] = command
    replacements = {'libsolver_ipm.a': [str(objects['ipm']), str(regularized)],
                    'libsolver_highs.a': [str(objects['highs'])], 'libsolver_pdlp.a': [str(objects['pdlp'])],
                    'libsolver_normal_cholesky.a': [str(objects['normal'])]}
    main_original = originals['main'][originals['main'].index('-o') + 1]
    replacements[main_original] = [str(objects['main'])]
    link, counts = [], dict.fromkeys(replacements, 0)
    for value in original_link:
        if value in replacements:
            link.extend(replacements[value])
            counts[value] += 1
        elif not value.startswith('-Wl,--out-implib,'):
            link.append(value)
    require(all(count == 1 for count in counts.values()), '五处独立链接替换必须唯一')
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
                    str(DATA / 'v81-basis-self-test.cpp'), '-o', str(self_exe)]
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
                      str(DATA / 'v81-repair-self-test.cpp'), '-o', str(repair_exe)]
    execute(repair_command, build_dir, output / 'repair-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / 'repair-self-test.d', build_dir)))
    repair_run = execute([str(repair_exe)], ROOT, output / 'repair-self-run.json', environment)
    repair_check = json.loads(repair_run['stdout'])
    require(repair_check['passed'] is True, '两阶段基修复内核自测失败')
    save(output / 'repair-self-check.json', {'passed': True, 'details': repair_check, 'lp_runs': 0})
    normal_self_obj = output / 'normal-self.obj'
    normal_self_command = compile_arguments(normal_test_compiles[0], DATA / 'v81-normal-self-test.cpp',
                                            normal_self_obj, frozen, ROOT)
    execute(normal_self_command, build_dir, output / 'normal-self-build.json', environment)
    compiled_inputs.update(file_hashes(dependency_paths(Path(str(normal_self_obj) + '.d'), build_dir)))
    normal_test_original_obj = normal_test_compiles[0][normal_test_compiles[0].index('-o') + 1]
    normal_exes, normal_links, normal_runs, normal_stdout = {}, {}, {}, {}
    normal_before_link = {}
    for mode in ('reference', 'candidate'):
        normal_exe = output / ('normal-' + mode + '-self-test.exe')
        normal_exes[mode] = normal_exe
        normal_link, harness_count, normal_count = [], 0, 0
        for value in normal_original_link:
            if value == normal_test_original_obj:
                normal_link.append(str(normal_self_obj)); harness_count += 1
            elif value == 'libsolver_normal_cholesky.a':
                normal_link.append(str(objects['normal']) if mode == 'candidate' else value)
                normal_count += 1
            elif not value.startswith('-Wl,--out-implib,'):
                normal_link.append(value)
        require(harness_count == 1 and normal_count == 1, '双版本自测必须唯一替换同一harness与正规方程实现')
        require(normal_link.count('-o') == 1, '正规方程自测输出选项不唯一')
        normal_link[normal_link.index('-o') + 1] = str(normal_exe)
        normal_links[mode] = normal_link
        normal_inputs = [resolve(value, build_dir) for value in normal_link if not value.startswith('-') and
                         Path(value).suffix.lower() in ('.a', '.lib', '.o', '.obj')]
        for path, value in file_hashes(normal_inputs).items():
            require(path not in normal_before_link or normal_before_link[path] == value,
                    '双版本自测之间显式链接输入变化：' + path)
            normal_before_link[path] = value
        execute(normal_link, build_dir, output / ('normal-' + mode + '-build.json'), environment)
        row, stdout_bytes = execute_normal([str(normal_exe)], ROOT,
                                           output / ('normal-' + mode + '-run.json'), environment)
        normal_runs[mode] = json.loads(row['stdout'])
        normal_stdout[mode] = stdout_bytes
        require(normal_runs[mode]['passed'] is True and type(normal_runs[mode]['cases']) is int and
                normal_runs[mode]['cases'] > 0 and normal_runs[mode]['long_column_interrupted'] is True,
                '正规方程原方程与长列预算事务自测失败')
    require(normal_stdout['reference'] == normal_stdout['candidate'], '原版与批量版正规方程输出字节不同')
    normal_check = {'passed': True, 'lp_runs': 0, 'stdout_identical': True,
                    'reference_run': 'normal-reference-run.json', 'candidate_run': 'normal-candidate-run.json',
                    'reference': normal_runs['reference'], 'candidate': normal_runs['candidate']}
    save(output / 'normal-budget-self-check.json', normal_check)
    for group in (sources, libraries, protected, frozen_files, before_link, normal_before_link, compiled_inputs):
        require_hashes(group)
    outputs = [executable, self_exe, repair_exe, *normal_exes.values(), normal_self_obj,
               *objects.values(), patched, normal_patched,
               output / 'ninja-commands.txt', *output.glob('*.json')]
    manifest = {'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'diagnostic_only': False,
                'experimental_candidate_only': True, 'production_changes': False,
                'plan_sha256': PLAN_SHA, 'baseline_identity_sha256': sha(baseline_path), 'parent_identity_sha256': sha(parent_path),
                'production_executable': str(production), 'production_sha256': sha(production),
                'frozen_parent_executable': str(frozen_parent), 'frozen_parent_sha256': sha(frozen_parent),
                'executable': str(executable), 'executable_sha256': sha(executable), 'frozen_files': parent['files'],
                'current_files': baseline['files'], 'actual_libraries': baseline['libraries'],
                'protected_before_and_after': protected, 'link_explicit_before_and_after': before_link,
                'normal_link_explicit_before_and_after': normal_before_link,
                'compile_inputs_after_and_final': compiled_inputs, 'outputs': file_hashes(outputs),
                'commands': {'cwd': str(build_dir), 'original_compile': originals, 'original_link': original_link,
                             'compile': commands, 'link': link, 'self_test': self_command,
                             'repair_self_test': repair_command,
                             'normal_original_compile': normal_test_compiles[0],
                             'normal_original_link': normal_original_link,
                             'normal_self_test_compile': normal_self_command, 'normal_self_test_link': normal_links},
                'implicit_link_inputs_verified': False, 'lp_runs': 0}
    save(output / 'build-manifest.json', manifest)
    print(json.dumps({'passed': True, 'compiled_units': list(units), 'solver_sha256': sha(executable),
                      'kernel_self_tests': self_check, 'repair_self_tests': repair_check,
                      'normal_self_tests': {'passed': True, 'cases': normal_runs['reference']['cases'],
                                            'stdout_identical': True},
                      'lp_runs': 0}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.tools/v81-basis')
    parser.add_argument('--self-test', action='store_true', help='仅检查冻结源插桩及逆变换，不编译或运行求解器')
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
