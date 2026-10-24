"""冻结 v48 的齐次失败快照插桩；只追加语句并验证逐字逆变换。"""

import hashlib


def changes():
    """每项为原始唯一锚点、前插文本、后插文本。"""
    return [
        ("#include <vector>\n", "", '#include "v49-hsd-diagnostic.hpp"\n'),
        ("class KktFactor {\n public:\n", "", """  std::span<const double> v49_diagonal() const noexcept { return diagonal_h_; }
  std::span<const double> v49_regularized_diagonal() const noexcept { return regularized_h_; }
  int v49_level() const noexcept { return regularization_level_; }
"""),
        ("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n"
         "    Clock::time_point start, SolveResult& result) {\n", "",
         "  ::solver::v49hsd::begin_run();\n"),
        ("    const long double affine_h_tau = -static_cast<long double>(tau) * kappa;\n", "",
         '    ::solver::v49hsd::phase("affine", iteration);\n'),
        ("    const auto corrected = scaled_direction ? homogeneous_scaled_direction(model, point, tau, kappa, primal_residual,\n",
         '    ::solver::v49hsd::phase("corrector", iteration);\n', ""),
        ("  direction.scalars = detail::ipm_homogeneous_scalar_direction(model.costs, model.rhs, u, constant,\n",
         """  ::solver::v49hsd::Scope v49_scope(model, point, factor.v49_diagonal(),
      factor.v49_regularized_diagonal(), factor.v49_level(), u, constant, tau, kappa, result);
""", ""),
        ("    long double gap_rhs, long double complementarity_rhs, const std::function<void()>& check) {\n",
         "", """  ::solver::v49hsd::ScalarObservation v49_observation(gap_rhs, complementarity_rhs);
  try {
"""),
        ("  const long double denominator = static_cast<long double>(kappa) / tau - dot_v;\n", "",
         "  v49_observation.values(dot_u, dot_v, denominator);\n"),
        ("  return result;\n}\n\ndouble detail::ipm_homogeneous_step_limit(", "", ""),
    ]


def transform(source):
    source = source.replace("\r\n", "\n")
    if "v49hsd" in source or "v49-hsd-diagnostic" in source:
        raise ValueError("输入已经包含 v49 HSD 插桩")
    original = source
    applied = []
    anchors = []
    for number, (anchor, before, after) in enumerate(changes()):
        # 函数结尾是唯一需要在原花括号内追加 catch 的位置。
        if number == len(changes()) - 1:
            anchor = "  return result;\n}\n\ndouble detail::ipm_homogeneous_step_limit("
            replacement = """  return result;
  } catch (const NumericalFailure& v49_failure) {
    v49_observation.failure(v49_failure.what());
    throw;
  }
}

double detail::ipm_homogeneous_step_limit("""
        else:
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
    return {"passed": True, "anchor_count": len(anchors), "reversible": True,
            "duplicate_rejected": True,
            "original_normalized_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
            "transformed_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
