"""只向冻结的正规方程源码追加观测；原数值语句可精确逆变换恢复。"""

from __future__ import annotations

import hashlib


def _edits() -> list[tuple[str, str, str]]:
    return [
        ("header", '#include <string>\n', '#include <string>\n#include "v49-normal-diagnostic.hpp"\n'),
        ("attempt_scope", '  auto& state = *impl_;\n  state.valid = false;',
         '  auto& state = *impl_;\n  v49normal::Scope v49_scope(static_cast<bool>(state.cholmod), state.valid);\n  state.valid = false;'),
        ("gram_stage", '  std::fill(state.sums.begin(), state.sums.end(), 0);',
         '  v49_scope.phase(v49normal::Phase::gram);\n  std::fill(state.sums.begin(), state.sums.end(), 0);'),
        ("cast_stage", '  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {',
         '  v49_scope.phase(v49normal::Phase::conversion);\n  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {'),
        ("numeric_stage", '    if (state.cholmod) {\n      if (!state.cholmod->factor',
         '    if (state.cholmod) {\n      v49_scope.phase(v49normal::Phase::cholmod);\n      if (!state.cholmod->factor'),
        ("numeric_start", '    const int success = cholmod_l_factorize(&matrix_, resources_.factor, &resources_.common);',
         '    v49normal::numeric_start();\n    const int success = cholmod_l_factorize(&matrix_, resources_.factor, &resources_.common);'),
        ("failure", '        resources_.factor->minor != resources_.factor->n || !finite_factor(budget_check)) return false;',
         '        resources_.factor->minor != resources_.factor->n || !finite_factor(budget_check)) {\n'
         '      v49normal::failure(matrix_, *resources_.factor, success, resources_.common.status);\n'
         '      return false;\n    }'),
        ("complete", '  state.valid = true;\n  return true;',
         '  state.valid = true;\n  v49_scope.phase(v49normal::Phase::complete);\n  return true;'),
    ]


def transform(source: str) -> tuple[str, list[dict[str, object]]]:
    newline = "\r\n" if "\r\n" in source else "\n"
    output = source
    anchors = []
    for name, before, after in _edits():
        before, after = before.replace("\n", newline), after.replace("\n", newline)
        count = output.count(before)
        if count != 1:
            raise ValueError(f"normal 锚点 {name} 应唯一，实际 {count}")
        output = output.replace(before, after, 1)
        anchors.append({"name": name, "before": before, "after": after, "count": count})
    restored = output
    for anchor in reversed(anchors):
        after, before = str(anchor["after"]), str(anchor["before"])
        if restored.count(after) != 1:
            raise ValueError(f"normal 逆变换锚点 {anchor['name']} 不唯一")
        restored = restored.replace(after, before, 1)
    if restored != source:
        raise ValueError("normal 逆变换未精确恢复冻结来源")
    return output, anchors


def self_check(source: str) -> dict[str, object]:
    transformed, anchors = transform(source)
    for changed in (source + str(anchors[0]["before"]), source.replace(str(anchors[0]["before"]), "", 1)):
        try:
            transform(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("重复或丢失锚点未拒绝")
    return {"passed": True, "anchor_count": len(anchors), "reversible": True,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "transformed_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
