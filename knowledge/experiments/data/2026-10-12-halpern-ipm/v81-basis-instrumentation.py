"""在冻结 v66 副本接入有限逆更新链，原守卫及完整候选事务保持。"""
import hashlib
from pathlib import Path
import zipfile


LOG = r'''
// 额外恢复单独记账；原内点分解和迭代计数保持原有定义。
struct V81BasisRunLog {
  bool enabled = false, attempted = false, accepted = false;
  const char* stage = "not_attempted";
  double attempt_seconds = 0, verification_seconds = 0;
  bool evaluated = false, primal_valid = false, primal_feasible = false;
  bool dual_feasible = false, gap_satisfied = false, numerically_optimal = false;
  std::optional<double> max_row_violation, max_bound_violation, max_dual_residual, objective_gap;
  ::solver::v81_basis::Result recovery;
  V81BasisRunLog() {
    const auto* value = std::getenv("SOLVER_V81_BASIS_RECOVERY");
    enabled = value && value[0] == '1' && value[1] == '\0';
  }
  ~V81BasisRunLog() noexcept {
    const auto& s = recovery.stats;
    std::fprintf(stderr, "V81_BASIS {\"schema\":1,\"enabled\":%s,\"attempted\":%s,"
        "\"accepted\":%s,\"stage\":\"%s\",\"candidate_available\":%s,"
        "\"attempt_seconds\":%.17g,\"verification_seconds\":%.17g,"
        "\"recovery_status\":\"%s\",\"stats\":{"
        "\"n\":%lld,\"m\":%lld,\"input_nonzeros\":%lld,\"rank\":%lld,"
        "\"qr_q_nonzeros\":%lld,\"qr_r_nonzeros\":%lld,"
        "\"lu_l_nonzeros\":%lld,\"lu_u_nonzeros\":%lld,\"budget_checks\":%lld,"
        "\"qr_analyze_calls\":%lld,\"qr_factor_calls\":%lld,"
        "\"lu_analyze_calls\":%lld,\"lu_factor_calls\":%lld,"
        "\"primal_solves\":%lld,\"dual_solves\":%lld,\"slack_matvecs\":%lld,"
        "\"selection_candidates\":%lld,\"selection_pivots\":%lld,\"selection_rejected\":%lld,"
        "\"work_units\":%lld,\"l_nonzeros\":%lld,\"pivot_columns_count\":%llu,"
        "\"ordered_rows_count\":%llu,\"qr_columns_permutation_count\":%llu,"
        "\"selected_rows_count\":%llu,\"qr_default_threshold\":",
        enabled ? "true" : "false", attempted ? "true" : "false", accepted ? "true" : "false",
        stage, recovery.candidate_available ? "true" : "false", attempt_seconds, verification_seconds,
        recovery.status.c_str(), static_cast<long long>(s.n), static_cast<long long>(s.m),
        static_cast<long long>(s.input_nonzeros), static_cast<long long>(s.rank),
        static_cast<long long>(s.qr_q_nonzeros), static_cast<long long>(s.qr_r_nonzeros),
        static_cast<long long>(s.lu_l_nonzeros), static_cast<long long>(s.lu_u_nonzeros),
        static_cast<long long>(s.budget_checks), static_cast<long long>(s.qr_analyze_calls),
        static_cast<long long>(s.qr_factor_calls), static_cast<long long>(s.lu_analyze_calls),
        static_cast<long long>(s.lu_factor_calls), static_cast<long long>(s.primal_solves),
        static_cast<long long>(s.dual_solves), static_cast<long long>(s.slack_matvecs),
        static_cast<long long>(s.selection_candidates), static_cast<long long>(s.selection_pivots),
        static_cast<long long>(s.selection_rejected), static_cast<long long>(s.work_units),
        static_cast<long long>(s.l_nonzeros), static_cast<unsigned long long>(s.pivot_columns.size()),
        static_cast<unsigned long long>(s.ordered_rows.size()),
        static_cast<unsigned long long>(s.qr_columns_permutation.size()),
        static_cast<unsigned long long>(s.selected_rows.size()));
    const auto number = [](std::optional<double> value) {
      if (value && std::isfinite(*value)) std::fprintf(stderr, "%.17g", *value);
      else std::fputs("null", stderr);
    };
    number(s.qr_default_threshold);
    std::fputs(",\"min_abs_pivot\":", stderr); number(s.min_abs_pivot);
    std::fputs(",\"selection_threshold\":", stderr); number(s.selection_threshold);
    std::fputs("},\"repair\":{", stderr);
    const auto& repair = recovery.repair;
    const auto& r = repair.stats;
    std::fprintf(stderr, "\"status\":\"%s\",\"candidate_available\":%s,\"stats\":{\"phase\":\"%s\"",
        repair.status.c_str(), repair.candidate_available ? "true" : "false", r.phase.c_str());
    const auto count = [](const char* name, std::int64_t value) {
      std::fprintf(stderr, ",\"%s\":%lld", name, static_cast<long long>(value));
    };
    count("attempts", r.attempts); count("iterations", r.iterations);
    count("phase1_iterations", r.phase1_iterations); count("phase2_iterations", r.phase2_iterations);
    count("pivots", r.pivots); count("zero_steps", r.zero_steps);
    count("work_units", r.work_units); count("budget_checks", r.budget_checks);
    count("lu_analyze_calls", r.lu_analyze_calls); count("lu_factor_calls", r.lu_factor_calls);
    count("factor_nonzeros", r.factor_nonzeros); count("last_factor_nonzeros", r.last_factor_nonzeros);
    count("primal_solves", r.primal_solves);
    count("dual_solves", r.dual_solves); count("direction_solves", r.direction_solves);
    count("slack_matvecs", r.slack_matvecs); count("direction_matvecs", r.direction_matvecs);
    count("pricing_matvecs", r.pricing_matvecs); count("roundoff_rows", r.roundoff_rows);
    count("final_selected_rows_count", static_cast<std::int64_t>(r.final_selected_rows.size()));
    std::fputs(",\"initial_phi\":", stderr); number(r.initial_phi);
    std::fputs(",\"final_phi\":", stderr); number(r.final_phi);
    std::fputs(",\"initial_max_violation\":", stderr); number(r.initial_max_violation);
    std::fputs(",\"final_max_violation\":", stderr); number(r.final_max_violation);
    std::fputs(",\"max_roundoff_bound\":", stderr); number(r.max_roundoff_bound);
    std::fputs(",\"last_right_derivative\":", stderr); number(r.last_right_derivative);
    std::fputs(",\"last_step\":", stderr); number(r.last_step);
    std::fputs(",\"max_selected_residual\":", stderr); number(r.max_selected_residual);
    std::fputs(",\"max_direction_residual\":", stderr); number(r.max_direction_residual);
    std::fputs(",\"last_point_error\":", stderr); number(r.last_point_error);
    std::fputs(",\"last_point_bound\":", stderr); number(r.last_point_bound);
    // 两个守卫快照分别对应原点和临时修正点，逐行尺度不能混用跨行最大值。
    std::fputs(",\"initial_guard\":", stderr);
    ::solver::v81_basis_repair::write_guard_diagnostics(stderr, r.initial_guard);
    std::fputs(",\"refined_guard\":", stderr);
    ::solver::v81_basis_repair::write_guard_diagnostics(stderr, r.refined_guard);
    std::fputs(",\"initial_refinement\":", stderr);
    ::solver::v81_basis_repair::write_refinement_diagnostics(stderr, r.initial_refinement);
    // 累计正向系统工作和最后一次右端的守卫快照由同一内核输出。
    std::fputs(",\"direction_forward\":", stderr);
    ::solver::v81_basis_repair::write_forward_diagnostics(stderr, r.direction_forward);
    std::fputs(",\"rebuild_forward\":", stderr);
    ::solver::v81_basis_repair::write_forward_diagnostics(stderr, r.rebuild_forward);
    std::fputs(",\"phase1_transition\":", stderr);
    ::solver::v81_basis_repair::write_transition_diagnostics(stderr, r.phase1_transition);
    std::fputs(",\"history\":", stderr);
    ::solver::v81_basis_repair::write_history_diagnostics(stderr, r.history);
    std::fputs(",\"search\":", stderr);
    ::solver::v81_basis_repair::write_search_diagnostics(stderr, r.search);
    std::fputs(",\"eta\":", stderr);
    ::solver::v81_basis_repair::write_eta_diagnostics(stderr, r.eta);
    std::fputs("}},\"trial_verification\":", stderr);
    if (!evaluated) std::fputs("null", stderr);
    else {
      std::fprintf(stderr, "{\"primal_valid\":%s,\"primal_feasible\":%s,\"dual_feasible\":%s,"
          "\"gap_satisfied\":%s,\"numerically_optimal\":%s,\"max_row_violation\":",
          primal_valid ? "true" : "false", primal_feasible ? "true" : "false",
          dual_feasible ? "true" : "false", gap_satisfied ? "true" : "false",
          numerically_optimal ? "true" : "false");
      number(max_row_violation);
      std::fputs(",\"max_bound_violation\":", stderr); number(max_bound_violation);
      std::fputs(",\"max_dual_residual\":", stderr); number(max_dual_residual);
      std::fputs(",\"objective_gap\":", stderr); number(objective_gap);
      std::fputs("}", stderr);
    }
    std::fputs(",\"additional_work_in_legacy_ipm_counts\":false,\"work_in_total_seconds\":true,"
        "\"marker_in_total_seconds\":false}\n", stderr);
  }
};

'''


RECOVERY = r'''  } catch (const NumericalFailure&) {
    if (!v81_log.enabled) throw;
    v81_log.attempted = true;
    v81_log.stage = "recovering";
    const auto v81_start = Clock::now();
    try {
      check_budget(options, start);
      ::solver::v81_basis::Limits v81_limits;
      v81_limits.max_input_nonzeros = 8000000;
      v81_limits.max_factor_nonzeros = 32000000;
      v81_limits.max_dimension = 1000000;
      v81_limits.max_selection_work = 1000000000;
      v81_limits.max_repair_pivots = 512;
      v81_limits.max_repair_work = 400000000;
      v81_limits.max_repair_factor_nonzeros = 64000000;
      v81_limits.max_eta_updates = 8;
      const ::solver::v81_basis::Input v81_input{model.starts, model.rows, model.values,
          model.costs, model.rhs, point.s, point.z, model.inequality};
      ::solver::v81_basis::recover(v81_input, v81_limits,
          [&] { check_budget(options, start); }, v81_log.recovery);
      if (v81_log.recovery.candidate_available) {
        v81_log.stage = "verifying";
        Point v81_point{std::move(v81_log.recovery.x), std::move(v81_log.recovery.s),
            std::move(v81_log.recovery.z)};
        // 临时副本隔离失败候选，只把实际发生的终验成本累加回原结果。
        SolveResult v81_trial = result;
        // 强制副本保留本次候选的检查，避免较差候选的诊断仍引用旧点。
        v81_trial.has_solution = false;
        const double v81_previous_verification = result.verification_seconds;
        bool v81_valid = false;
        try {
          v81_valid = assess(model, v81_point, options, start, v81_trial);
          v81_log.evaluated = true;
          v81_log.primal_valid = v81_trial.verification.valid_input;
          v81_log.primal_feasible = v81_trial.verification.feasible;
          v81_log.max_row_violation = v81_trial.verification.max_row_violation;
          v81_log.max_bound_violation = v81_trial.verification.max_bound_violation;
          if (v81_trial.lp_optimality) {
            v81_log.dual_feasible = v81_trial.lp_optimality->dual_feasible;
            v81_log.gap_satisfied = v81_trial.lp_optimality->gap_satisfied;
            v81_log.numerically_optimal = v81_trial.lp_optimality->numerically_optimal;
            v81_log.max_dual_residual = v81_trial.lp_optimality->max_dual_residual;
            v81_log.objective_gap = v81_trial.lp_optimality->objective_gap;
          }
        } catch (...) {
          result.verification_seconds = v81_trial.verification_seconds;
          v81_log.verification_seconds = v81_trial.verification_seconds - v81_previous_verification;
          throw;
        }
        result.verification_seconds = v81_trial.verification_seconds;
        v81_log.verification_seconds = v81_trial.verification_seconds - v81_previous_verification;
        check_budget(options, start);
        if (v81_valid) {
          result = std::move(v81_trial);
          v81_log.accepted = true;
          v81_log.stage = "accepted";
          v81_log.attempt_seconds = elapsed(v81_start);
          return;
        }
        v81_log.stage = "original_verification_rejected";
      } else {
        v81_log.stage = "candidate_unavailable";
      }
    } catch (const BudgetExpired&) {
      v81_log.stage = "time_limit";
      v81_log.attempt_seconds = elapsed(v81_start);
      throw;
    } catch (const std::exception&) {
      v81_log.stage = "recovery_exception";
    }
    v81_log.attempt_seconds = elapsed(v81_start);
    // 恢复没有通过时，保留原数值异常和进入恢复前的最佳点。
    throw;
  }
'''


def changes():
    signature = ('void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n'
                 '    Clock::time_point start, SolveResult& result) {\n')
    loop = ('  const bool scaled_direction = options.ipm.linear_accuracy == "homogeneous_scaled";\n'
            '  for (std::int64_t iteration = 1; iteration <= options.ipm.max_iterations; ++iteration) {\n')
    ending = ('    if (iteration == options.ipm.max_iterations) break;\n'
              '  }\n}\n\nvoid validate_options(const SolveOptions& options) {\n')
    solve = ('SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n'
             '  const auto start = Clock::now();\n')
    return [
        ('#include <vector>\n', '#include <vector>\n#include <cstdio>\n#include <cstdlib>\n'
         '#include "v81-basis-recovery.hpp"\n'),
        (signature, LOG + signature.replace('SolveResult& result)', 'SolveResult& result, V81BasisRunLog& v81_log)')),
        (loop, loop.replace('  for (', '  try {\n  for (', 1)),
        (ending, '    if (iteration == options.ipm.max_iterations) break;\n  }\n' + RECOVERY +
         '}\n\nvoid validate_options(const SolveOptions& options) {\n'),
        (solve, solve + '  V81BasisRunLog v81_log;\n'),
        ('        run_homogeneous_ipm(working, options, start, result);\n',
         '        run_homogeneous_ipm(working, options, start, result, v81_log);\n'),
    ]


def transform(source):
    source = source.replace('\r\n', '\n')
    if 'v81_basis' in source or 'v81-basis-recovery' in source or 'V81BasisRunLog' in source:
        raise ValueError('输入已有 v81 候选插桩')
    original = source
    applied, anchors = [], []
    for number, (anchor, replacement) in enumerate(changes()):
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f'插桩锚点 {number} 必须唯一，实际 {count} 次')
        anchors.append({'index': number, 'source_line': original[:original.index(anchor)].count('\n') + 1,
                        'anchor_sha256': hashlib.sha256(anchor.encode()).hexdigest(),
                        'replacement_sha256': hashlib.sha256(replacement.encode()).hexdigest()})
        source = source.replace(anchor, replacement, 1)
        applied.append((anchor, replacement))
    restored = source
    for anchor, replacement in reversed(applied):
        if restored.count(replacement) != 1:
            raise ValueError('逆变换锚点不唯一')
        restored = restored.replace(replacement, anchor, 1)
    if restored != original:
        raise ValueError('逆变换没有还原原始程序')
    return source, anchors


def self_check(source=None):
    if source is None:
        with zipfile.ZipFile(Path(__file__).with_name('ipm-v66-source.zip')) as archive:
            source = archive.read('src/ipm.cpp').decode('utf-8')
    transformed, anchors = transform(source)
    try:
        transform(transformed)
    except ValueError:
        pass
    else:
        raise AssertionError('重复插桩未拒绝')
    function = transformed[transformed.index('void run_homogeneous_ipm('):transformed.index('void validate_options(')]
    assert function.index('initialize(model, point, factor') < function.index('  try {')
    assert function.count('::solver::v81_basis::recover(') == 1
    assert function.count('catch (const NumericalFailure&)') == 1
    assert function.index('check_budget(options, start);\n        if (v81_valid)') < function.index('result = std::move(v81_trial)')
    selection_fields = ('selection_candidates', 'selection_pivots', 'selection_rejected', 'work_units',
                        'l_nonzeros', 'min_abs_pivot', 'selection_threshold', 'pivot_columns')
    assert all('s.' + field in LOG for field in selection_fields)
    assert 'v81_limits.max_selection_work = 1000000000;' in function
    assert 'v81_limits.max_repair_pivots = 512;' in function
    assert 'v81_limits.max_repair_work = 400000000;' in function
    assert 'v81_limits.max_repair_factor_nonzeros = 64000000;' in function
    assert 'v81_limits.max_eta_updates = 8;' in function
    repair_fields = ('attempts', 'iterations', 'phase1_iterations', 'phase2_iterations', 'pivots',
                     'zero_steps', 'work_units', 'budget_checks', 'lu_analyze_calls', 'lu_factor_calls',
                     'factor_nonzeros', 'last_factor_nonzeros', 'primal_solves', 'dual_solves', 'direction_solves', 'slack_matvecs',
                     'direction_matvecs', 'pricing_matvecs', 'roundoff_rows', 'final_selected_rows',
                     'initial_phi', 'final_phi', 'initial_max_violation', 'final_max_violation',
                     'max_roundoff_bound', 'last_right_derivative', 'last_step',
                     'max_selected_residual', 'max_direction_residual', 'last_point_error', 'last_point_bound')
    assert all('r.' + field in LOG for field in repair_fields)
    guard_fields = ('initial_guard', 'refined_guard', 'initial_refinement')
    assert all('r.' + field in LOG for field in guard_fields)
    assert LOG.count('::solver::v81_basis_repair::write_guard_diagnostics(stderr,') == 2
    assert LOG.count('::solver::v81_basis_repair::write_refinement_diagnostics(stderr,') == 1
    forward_fields = ('direction_forward', 'rebuild_forward')
    assert all('r.' + field in LOG for field in forward_fields)
    assert LOG.count('::solver::v81_basis_repair::write_forward_diagnostics(stderr,') == 2
    transition_fields = ('phase1_transition', 'history')
    assert all('r.' + field in LOG for field in transition_fields)
    assert LOG.count('::solver::v81_basis_repair::write_transition_diagnostics(stderr,') == 1
    assert LOG.count('::solver::v81_basis_repair::write_history_diagnostics(stderr,') == 1
    assert 'r.search' in LOG
    assert LOG.count('::solver::v81_basis_repair::write_search_diagnostics(stderr,') == 1
    assert 'r.eta' in LOG
    assert LOG.count('::solver::v81_basis_repair::write_eta_diagnostics(stderr,') == 1
    return {'passed': True, 'anchor_count': len(anchors), 'reversible': True, 'duplicate_rejected': True,
            'initialization_outside_recovery': True, 'single_recovery_site': True,
            'selection_diagnostic_fields': list(selection_fields), 'selection_work_limit': 1000000000,
            'repair_diagnostic_fields': list(repair_fields), 'repair_pivot_limit': 512,
            'repair_work_limit': 400000000, 'repair_cumulative_factor_limit': 64000000,
            'single_factor_limit': 32000000,
            'eta_chain_limit': 8,
            'guard_diagnostic_objects': list(guard_fields),
            'forward_diagnostic_objects': list(forward_fields),
            'transition_diagnostic_objects': list(transition_fields),
            'search_diagnostic_object': 'search',
            'eta_diagnostic_object': 'eta',
            'final_budget_before_commit': True, 'solver_runs': 0,
            'original_normalized_sha256': hashlib.sha256(source.replace('\r\n', '\n').encode()).hexdigest(),
            'transformed_sha256': hashlib.sha256(transformed.encode()).hexdigest()}
