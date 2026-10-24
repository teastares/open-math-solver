"""v53 冻结副本插桩：唯一字面量锚点、仅追加诊断语句、可逐字逆变换。"""

import hashlib


def changes():
    """原始锚点、前插文本、后插文本；独立核验器可直接审计这些字面量。"""
    return [
        ("#include <vector>\n", "", '#include "v53-hsd-diagnostic.hpp"\n'),
        ("class KktFactor {\n public:\n", "",
         "  std::span<const double> v53_diagonal() const noexcept { return diagonal_h_; }\n"),
        ("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n"
         "    Clock::time_point start, SolveResult& result) {\n", "",
         "  ::solver::v53hsd::RunScope v53_run(model, result.model, options);\n"),
        ("  const auto u = factor.solve(rhs, options, start, result);\n",
         "  ::solver::v53hsd::DirectionScope v53_direction(point, dual_residual, rhs,\n"
         "      factor.v53_diagonal(), tau, kappa, theta, result);\n",
         "  v53_direction.got_u(u);\n"),
        ("  return direction;\n}\n\nHomogeneousDirection homogeneous_scaled_direction(",
         "  v53_direction.returned(direction, constant, result);\n", ""),
        ("    point = std::move(candidate);\n    tau = next_tau;\n", 
         "    ::solver::v53hsd::commit(candidate, next_tau, step, iteration);\n", ""),
        ("  if (replace) {\n    result.verification = std::move(checks.primal);\n",
         "  ::solver::v53hsd::assess(point, x, y, checks, replace, result);\n", ""),
    ]


def transform(source):
    source = source.replace("\r\n", "\n")
    if "v53hsd" in source or "v53-hsd-diagnostic" in source:
        raise ValueError("输入已经包含 v53 HSD 插桩")
    original = source
    applied = []
    anchors = []
    for number, (anchor, before, after) in enumerate(changes()):
        replacement = before + anchor + after
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"HSD 插桩锚点 {number} 必须唯一，实际 {count} 次")
        anchors.append({"index": number, "source_line": original[:original.index(anchor)].count("\n") + 1,
                        "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
                        "replacement_sha256": hashlib.sha256(replacement.encode()).hexdigest()})
        source = source.replace(anchor, replacement, 1)
        applied.append((anchor, replacement))
    restored = source
    for anchor, replacement in reversed(applied):
        if restored.count(replacement) != 1:
            raise ValueError("HSD 逆变换锚点不是唯一")
        restored = restored.replace(replacement, anchor, 1)
    if restored != original:
        raise ValueError("HSD 插桩逆变换没有还原原始程序")
    return source, anchors


def self_check(source):
    transformed, anchors = transform(source)
    try:
        transform(transformed)
    except ValueError:
        pass
    else:
        raise AssertionError("重复插桩没有被拒绝")
    return {"passed": True, "anchor_count": len(anchors), "reversible": True, "duplicate_rejected": True,
            "original_normalized_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
            "transformed_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
