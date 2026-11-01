"""仅批量化正规方程回代的预算回调；浮点语句与遍历顺序保持原样。"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SIGNATURE = 'bool NormalCholesky::solve(std::vector<double>& rhs, const std::function<void()>& budget_check) const {'
END = '\nIndex NormalCholesky::factor_nonzeros() const'
COUNTER = '''  std::size_t v81_budget_remaining = 256;
  const auto v81_budget_tick = [&] {
    if (--v81_budget_remaining == 0) {
      check(budget_check);
      v81_budget_remaining = 256;
    }
  };
'''


def replacements():
    first_head = '''  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const long double value = static_cast<long double>(rhs[static_cast<std::size_t>(state.primal_coordinates[col])]) /
'''
    first_entries = '''    for (const auto& entry : state.columns[col]) {
      reduced_rhs[static_cast<std::size_t>(entry.row)] += state.input_values[static_cast<std::size_t>(entry.source)] * value;
'''
    extended_head = '''    for (std::size_t col = 0; col < state.columns.size(); ++col) {
      check(budget_check);
      const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
'''
    extended_entries = '''      for (const auto& entry : state.columns[col]) {
        value -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) *
'''
    ordinary_head = '''  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
'''
    ordinary_expression = 'value -= static_cast<long double>(state.input_values[static_cast<std::size_t>(entry.source)]) * permuted[state.inverse[static_cast<std::size_t>(entry.row)]];'
    ordinary_entries = '    for (const auto& entry : state.columns[col]) ' + ordinary_expression + '\n'
    return [
        ('shared_counter', '  const auto& state = *impl_;\n  check(budget_check);\n',
         '  const auto& state = *impl_;\n  check(budget_check);\n' + COUNTER),
        ('first_column_head', first_head, first_head.replace('check(budget_check);', 'v81_budget_tick();')),
        ('first_column_entries', first_entries, first_entries.replace(
            'state.columns[col]) {\n', 'state.columns[col]) {\n      v81_budget_tick();\n')),
        ('extended_column_head', extended_head, extended_head.replace('check(budget_check);', 'v81_budget_tick();')),
        ('extended_column_entries', extended_entries, extended_entries.replace(
            'state.columns[col]) {\n', 'state.columns[col]) {\n        v81_budget_tick();\n')),
        ('ordinary_column_head', ordinary_head, ordinary_head.replace('check(budget_check);', 'v81_budget_tick();')),
        ('ordinary_column_entries', ordinary_entries,
         '    for (const auto& entry : state.columns[col]) {\n      v81_budget_tick();\n      ' +
         ordinary_expression + '\n    }\n'),
    ]


def transform(source):
    source = source.replace('\r\n', '\n')
    if 'v81_budget_tick' in source or 'v81_budget_remaining' in source:
        raise ValueError('输入已有 v81 正规方程预算修改')
    if source.count(SIGNATURE) != 1 or source.count(END) != 1:
        raise ValueError('正规方程回代边界不唯一')
    begin = source.index(SIGNATURE)
    end = source.index(END, begin)
    body = source[begin:end]
    original_body = body
    anchors = []
    changes = replacements()
    for label, old, new in changes:
        if body.count(old) != 1:
            raise ValueError('正规方程预算锚点不唯一：' + label)
        body = body.replace(old, new)
        anchors.append({'label': label, 'count': 1,
                        'anchor_sha256': hashlib.sha256(old.encode()).hexdigest(),
                        'replacement_sha256': hashlib.sha256(new.encode()).hexdigest()})
    # 逆变换须恢复完整原函数，防止预算改动夹带浮点表达式或其他流程变化。
    restored = body
    for label, old, new in reversed(changes):
        if restored.count(new) != 1:
            raise ValueError('正规方程预算逆变换不唯一：' + label)
        restored = restored.replace(new, old)
    if restored != original_body:
        raise ValueError('正规方程预算逆变换未恢复原浮点源码')
    return source[:begin] + body + source[end:], anchors


def self_check(source=None):
    source = (ROOT / 'src/normal_cholesky.cpp').read_text(encoding='utf-8') if source is None else source
    source = source.replace('\r\n', '\n')
    transformed, anchors = transform(source)
    begin = transformed.index(SIGNATURE)
    end = transformed.index(END, begin)
    body = transformed[begin:end]
    original_begin = source.index(SIGNATURE)
    original_end = source.index(END, original_begin)
    original_body = source[original_begin:original_end]
    assert transformed[:begin] == source[:original_begin]
    assert transformed[end:] == source[original_end:]
    assert len(anchors) == 7 and body.count(COUNTER) == 1
    assert body.count('v81_budget_tick();') == 6
    assert body.count('check(budget_check);') == original_body.count('check(budget_check);') - 2
    assert body.count('rhs = std::move(solution);') == 2
    assert body.count('check(budget_check);\n    rhs = std::move(solution);') == 1
    assert body.count('check(budget_check);\n  rhs = std::move(solution);') == 1
    assert 'const ExtendedVector solved = state.extended_cholesky.solve(permuted);\n        check(budget_check);' in body
    assert 'const Eigen::VectorXd solution = state.cholesky.solve(permuted);\n      check(budget_check);' in body
    assert 'state.refined_solve(permuted, budget_check)' in body
    assert 'state.cholmod->solve(permuted, budget_check)' in body
    assert 'state.multifrontal->solve(solution, budget_check)' in body
    assert body.index(COUNTER) < body.index('if (!state.valid')
    return {'passed': True, 'public_lp_runs': 0, 'batch_work_units': 256,
            'column_head_ticks': 3, 'entry_loop_ticks': 3, 'shared_counter': True,
            'floating_source_round_trip': True, 'solve_only': True,
            'library_and_commit_checks_preserved': True}
