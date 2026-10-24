"""向冻结 v66 追加只读观察点，并逐字验证逆变换。"""
import hashlib
from pathlib import Path
import zipfile


def changes():
    """每项由原唯一锚点、前插文本、后插文本组成。"""
    return [
        ('#include <vector>\n', '', '#include "v67-krylov-diagnostic.hpp"\n'),
        ('class KktFactor {\n public:\n', '',
         '  std::span<const double> v67_diagonal() const noexcept { return diagonal_h_; }\n'),
        ('void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n'
         '    Clock::time_point start, SolveResult& result) {\n', '',
         '  ::solver::v67krylov::begin_run();\n'),
        ('      constant = factor.solve(constant_rhs, options, start, result);\n',
         '      {\n      ::solver::v67krylov::Scope v67_scope(model, factor.v67_diagonal(), constant_rhs, iteration, result);\n',
         '      }\n'),
        ('      const long double backward_error = assess_residual(rhs, answer, residual, options, start, cached_product);\n'
         '      const double norm = norm_inf(residual);\n', '',
         '      ::solver::v67krylov::ir_candidate();\n'),
        ('      const auto correction = direct(residual, options, start, result);\n',
         '      ::solver::v67krylov::ir_correction_started();\n',
         '      ::solver::v67krylov::ir_correction_completed();\n'),
        ('    constexpr std::size_t window = 32;\n',
         '    ::solver::v67krylov::krylov_begin();\n', ''),
        ('    for (int restart = 0; restart < 3; ++restart) {\n      check_budget(options, start);\n', '',
         '      ::solver::v67krylov::cycle_begin(restart);\n'),
        ('      for (std::size_t col = 0; col < limit; ++col) {\n        check_budget(options, start);\n', '',
         '        ::solver::v67krylov::column_begin(restart);\n'),
        ('        ++*result.ipm_refinement_steps;\n        std::vector<double> work(dimension_);\n',
         '        ::solver::v67krylov::preconditioner_complete(restart);\n', ''),
        ('        transformed[col] *= cosine[col];\n', '',
         '        ::solver::v67krylov::column_complete(restart, remaining);\n'),
        ('          const long double backward_error = assess_residual(rhs, candidate, residual, options, start, cached_product);\n', '',
         '          ::solver::v67krylov::candidate(restart, col, coefficients);\n'),
        ('      answer = std::move(candidate);\n    }\n    throw NumericalFailure("KKT Krylov 改进未达到原方程精度");\n',
         '      ::solver::v67krylov::cycle_end(restart, answer, preconditioned, candidate, residual, tolerance, block_tolerances);\n', ''),
        ('    throw NumericalFailure("KKT Krylov 改进未达到原方程精度");\n',
         '    ::solver::v67krylov::terminal_exhausted();\n', ''),
    ]


def transform(source):
    source = source.replace('\r\n', '\n')
    if 'v67krylov' in source or 'v67-krylov-diagnostic' in source:
        raise ValueError('输入已有 v67 插桩')
    original = source
    applied, anchors = [], []
    for number, (anchor, before, after) in enumerate(changes()):
        replacement = before + anchor + after
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
    return {'passed': True, 'anchor_count': len(anchors), 'reversible': True, 'duplicate_rejected': True,
            'solver_runs': 0, 'original_normalized_sha256': hashlib.sha256(source.replace('\r\n', '\n').encode()).hexdigest(),
            'transformed_sha256': hashlib.sha256(transformed.encode()).hexdigest()}
