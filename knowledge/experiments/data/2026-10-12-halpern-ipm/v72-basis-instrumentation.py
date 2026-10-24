"""在冻结 v66 的齐次主循环失败出口接入带单步基精修的恢复候选。"""
import hashlib
from pathlib import Path
import zipfile


LOG = r'''
// 额外恢复单独记账；原内点分解和迭代计数保持原有定义。
struct V72BasisRunLog {
  bool enabled = false, attempted = false, accepted = false;
  const char* stage = "not_attempted";
  double attempt_seconds = 0, verification_seconds = 0;
  bool evaluated = false, primal_valid = false, primal_feasible = false;
  bool dual_feasible = false, gap_satisfied = false, numerically_optimal = false;
  std::optional<double> max_row_violation, max_bound_violation, max_dual_residual, objective_gap;
  ::solver::v72_basis::Result recovery;
  V72BasisRunLog() {
    const auto* value = std::getenv("SOLVER_V72_BASIS_RECOVERY");
    enabled = value && value[0] == '1' && value[1] == '\0';
  }
  ~V72BasisRunLog() noexcept {
    const auto& s = recovery.stats;
    std::fprintf(stderr, "V72_BASIS {\"schema\":1,\"enabled\":%s,\"attempted\":%s,"
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
    std::fprintf(stderr, ",\"refinement_double_digits\":%lld,\"refinement_long_double_digits\":%lld,"
        "\"refinement_supported\":%s", static_cast<long long>(s.refinement_double_digits),
        static_cast<long long>(s.refinement_long_double_digits), s.refinement_supported ? "true" : "false");
    const auto refinement = [&](const char* name, const auto& value) {
      std::fprintf(stderr, ",\"%s\":{\"attempts\":%lld,\"solves\":%lld,\"residual_matvecs\":%lld,"
          "\"accepted\":%lld,\"status\":\"%s\",\"before_max_abs\":", name,
          static_cast<long long>(value.attempts), static_cast<long long>(value.solves),
          static_cast<long long>(value.residual_matvecs), static_cast<long long>(value.accepted), value.status.c_str());
      number(value.before_max_abs);
      std::fputs(",\"after_max_abs\":", stderr); number(value.after_max_abs);
      std::fputs(",\"before_componentwise\":", stderr); number(value.before_componentwise);
      std::fputs(",\"after_componentwise\":", stderr); number(value.after_componentwise);
      std::fputs(",\"correction_max_abs\":", stderr); number(value.correction_max_abs);
      std::fputs("}", stderr);
    };
    refinement("primal_refinement", s.primal_refinement);
    refinement("dual_refinement", s.dual_refinement);
    std::fputs("},\"trial_verification\":", stderr);
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
    if (!v72_log.enabled) throw;
    v72_log.attempted = true;
    v72_log.stage = "recovering";
    const auto v72_start = Clock::now();
    try {
      check_budget(options, start);
      ::solver::v72_basis::Limits v72_limits;
      v72_limits.max_input_nonzeros = 8000000;
      v72_limits.max_factor_nonzeros = 32000000;
      v72_limits.max_dimension = 1000000;
      v72_limits.max_selection_work = 1000000000;
      const ::solver::v72_basis::Input v72_input{model.starts, model.rows, model.values,
          model.costs, model.rhs, point.s, point.z, model.inequality};
      ::solver::v72_basis::recover(v72_input, v72_limits,
          [&] { check_budget(options, start); }, v72_log.recovery);
      if (v72_log.recovery.candidate_available) {
        v72_log.stage = "verifying";
        Point v72_point{std::move(v72_log.recovery.x), std::move(v72_log.recovery.s),
            std::move(v72_log.recovery.z)};
        // 临时副本隔离失败候选，只把实际发生的终验成本累加回原结果。
        SolveResult v72_trial = result;
        // 强制副本保留本次候选的检查，避免较差候选的诊断仍引用旧点。
        v72_trial.has_solution = false;
        const double v72_previous_verification = result.verification_seconds;
        bool v72_valid = false;
        try {
          v72_valid = assess(model, v72_point, options, start, v72_trial);
          v72_log.evaluated = true;
          v72_log.primal_valid = v72_trial.verification.valid_input;
          v72_log.primal_feasible = v72_trial.verification.feasible;
          v72_log.max_row_violation = v72_trial.verification.max_row_violation;
          v72_log.max_bound_violation = v72_trial.verification.max_bound_violation;
          if (v72_trial.lp_optimality) {
            v72_log.dual_feasible = v72_trial.lp_optimality->dual_feasible;
            v72_log.gap_satisfied = v72_trial.lp_optimality->gap_satisfied;
            v72_log.numerically_optimal = v72_trial.lp_optimality->numerically_optimal;
            v72_log.max_dual_residual = v72_trial.lp_optimality->max_dual_residual;
            v72_log.objective_gap = v72_trial.lp_optimality->objective_gap;
          }
        } catch (...) {
          result.verification_seconds = v72_trial.verification_seconds;
          v72_log.verification_seconds = v72_trial.verification_seconds - v72_previous_verification;
          throw;
        }
        result.verification_seconds = v72_trial.verification_seconds;
        v72_log.verification_seconds = v72_trial.verification_seconds - v72_previous_verification;
        check_budget(options, start);
        if (v72_valid) {
          result = std::move(v72_trial);
          v72_log.accepted = true;
          v72_log.stage = "accepted";
          v72_log.attempt_seconds = elapsed(v72_start);
          return;
        }
        v72_log.stage = "original_verification_rejected";
      } else {
        v72_log.stage = "candidate_unavailable";
      }
    } catch (const BudgetExpired&) {
      v72_log.stage = "time_limit";
      v72_log.attempt_seconds = elapsed(v72_start);
      throw;
    } catch (const std::exception&) {
      v72_log.stage = "recovery_exception";
    }
    v72_log.attempt_seconds = elapsed(v72_start);
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
         '#include "v72-basis-recovery.hpp"\n'),
        (signature, LOG + signature.replace('SolveResult& result)', 'SolveResult& result, V72BasisRunLog& v72_log)')),
        (loop, loop.replace('  for (', '  try {\n  for (', 1)),
        (ending, '    if (iteration == options.ipm.max_iterations) break;\n  }\n' + RECOVERY +
         '}\n\nvoid validate_options(const SolveOptions& options) {\n'),
        (solve, solve + '  V72BasisRunLog v72_log;\n'),
        ('        run_homogeneous_ipm(working, options, start, result);\n',
         '        run_homogeneous_ipm(working, options, start, result, v72_log);\n'),
    ]


def transform(source):
    source = source.replace('\r\n', '\n')
    if 'v72_basis' in source or 'v72-basis-recovery' in source or 'V72BasisRunLog' in source:
        raise ValueError('输入已有 v72 候选插桩')
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
    assert function.count('::solver::v72_basis::recover(') == 1
    assert function.count('catch (const NumericalFailure&)') == 1
    assert function.index('check_budget(options, start);\n        if (v72_valid)') < function.index('result = std::move(v72_trial)')
    selection_fields = ('selection_candidates', 'selection_pivots', 'selection_rejected', 'work_units',
                        'l_nonzeros', 'min_abs_pivot', 'selection_threshold', 'pivot_columns')
    assert all('s.' + field in LOG for field in selection_fields)
    assert 'v72_limits.max_selection_work = 1000000000;' in function
    refinement_fields = ('refinement_double_digits', 'refinement_long_double_digits', 'refinement_supported',
                         'primal_refinement', 'dual_refinement')
    assert all('s.' + field in LOG for field in refinement_fields)
    return {'passed': True, 'anchor_count': len(anchors), 'reversible': True, 'duplicate_rejected': True,
            'initialization_outside_recovery': True, 'single_recovery_site': True,
            'selection_diagnostic_fields': list(selection_fields), 'selection_work_limit': 1000000000,
            'refinement_diagnostic_fields': list(refinement_fields),
            'final_budget_before_commit': True, 'solver_runs': 0,
            'original_normalized_sha256': hashlib.sha256(source.replace('\r\n', '\n').encode()).hexdigest(),
            'transformed_sha256': hashlib.sha256(transformed.encode()).hexdigest()}
