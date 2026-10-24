"""冻结 v66 先应用原 v70，再仅追加候选原位只读捕获点。"""
import hashlib
from pathlib import Path
import runpy
import zipfile

DATA = Path(__file__).resolve().parent


def changes():
    return [
        ('#include "v70-basis-recovery.hpp"\n', '#include "v70-basis-recovery.hpp"\n'
         '#include "v71-basis-capture.hpp"\n'),
        ('  V70BasisRunLog v70_log;\n', '  V70BasisRunLog v70_log;\n'
         '  ::solver::v71_basis_capture::Report v71_capture;\n'),
        ('      if (v70_log.recovery.candidate_available) {\n',
         '      if (v70_log.recovery.candidate_available) {\n'
         '        ::solver::v71_basis_capture::capture(v70_input, v70_log.recovery,\n'
         '            [&] { check_budget(options, start); });\n'),
    ]


def transform(source):
    source = source.replace('\r\n', '\n')
    if 'v71_basis_capture' in source or 'v71-basis-capture' in source:
        raise ValueError('输入已有 v71 捕获插桩')
    original = source
    parent = runpy.run_path(str(DATA / 'v70-basis-instrumentation.py'))
    source, parent_anchors = parent['transform'](source)
    parent_source = source
    applied, anchors = [], []
    for number, (anchor, replacement) in enumerate(changes()):
        if source.count(anchor) != 1:
            raise ValueError(f'捕获锚点 {number} 必须唯一')
        anchors.append({'index': number, 'parent_source_line': parent_source[:parent_source.index(anchor)].count('\n') + 1,
                        'anchor_sha256': hashlib.sha256(anchor.encode()).hexdigest(),
                        'replacement_sha256': hashlib.sha256(replacement.encode()).hexdigest()})
        source = source.replace(anchor, replacement, 1)
        applied.append((anchor, replacement))
    restored = source
    for anchor, replacement in reversed(applied):
        if restored.count(replacement) != 1:
            raise ValueError('逆变换捕获锚点不唯一')
        restored = restored.replace(replacement, anchor, 1)
    if restored != parent_source:
        raise ValueError('逆变换没有还原冻结 v70 候选')
    parent['self_check'](original)
    return source, {'parent_v70': parent_anchors, 'capture_v71': anchors,
                    'parent_normalized_sha256': hashlib.sha256(parent_source.encode()).hexdigest()}


def self_check(source=None):
    if source is None:
        with zipfile.ZipFile(DATA / 'ipm-v66-source.zip') as archive:
            source = archive.read('src/ipm.cpp').decode('utf-8')
    transformed, anchors = transform(source)
    try:
        transform(transformed)
    except ValueError:
        pass
    else:
        raise AssertionError('重复捕获插桩未拒绝')
    hook = transformed.index('::solver::v71_basis_capture::capture(')
    branch = transformed.index('if (v70_log.recovery.candidate_available)')
    move = transformed.index('std::move(v70_log.recovery.x)')
    assert branch < hook < move
    return {'passed': True, 'capture_anchor_count': len(anchors['capture_v71']),
            'parent_anchor_count': len(anchors['parent_v70']), 'reversible_to_parent': True,
            'duplicate_rejected': True, 'capture_before_move': True, 'solver_runs': 0,
            'parent_normalized_sha256': anchors['parent_normalized_sha256'],
            'transformed_sha256': hashlib.sha256(transformed.encode()).hexdigest()}
