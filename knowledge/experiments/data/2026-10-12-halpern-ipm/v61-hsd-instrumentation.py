"""冻结 v60 的齐次恢复事件插桩；只追加观察语句并验证逐字逆变换。"""

import hashlib
import zipfile
from pathlib import Path


def changes():
    """每项为原始唯一锚点、前插文本、后插文本。"""
    return [
        ("#include <vector>\n", "", '#include "v61-hsd-diagnostic.hpp"\n'),
        ("class KktFactor {\n public:\n", "", """  std::span<const double> v61_diagonal() const noexcept { return diagonal_h_; }
  std::span<const double> v61_regularized_diagonal() const noexcept { return regularized_h_; }
"""),
        ("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n"
         "    Clock::time_point start, SolveResult& result) {\n", "", "  ::solver::v61hsd::begin_run();\n"),
        ("    const long double affine_h_tau = -static_cast<long double>(tau) * kappa;\n", "",
         '    ::solver::v61hsd::phase("affine", iteration);\n'),
        ("    const auto corrected = scaled_direction ? homogeneous_scaled_direction(model, point, tau, kappa, primal_residual,\n",
         '    ::solver::v61hsd::phase("corrector", iteration);\n', ""),
        ("  HomogeneousDirection direction;\n  if (repair_info) {\n", """  ::solver::v61hsd::Scope v61_scope(model, point, factor.v61_diagonal(),
      factor.v61_regularized_diagonal(), factor.regularization_level(), u, rhs, h,
      primal_residual, dual_residual, tau, kappa, theta, result);
""", ""),
        ("  const long double denominator = static_cast<long double>(kappa) / tau - dot_v;\n", "",
         "  ::solver::v61hsd::observe_values<Scalar>(dot_u, dot_v, denominator);\n"),
        ("    info.attempted = true;\n    auto candidate = repair();\n", "", """    ::solver::v61hsd::candidate(candidate);
"""),
        ("    info.attempted = true;\n", "", """    ::solver::v61hsd::begin_repair<Scalar>(u, v, gap_rhs, complementarity_rhs);
    try {
"""),
        ("    info.succeeded = true;\n    return direction;\n", "", """    } catch (const std::exception& v61_failure) {
      ::solver::v61hsd::failure(v61_failure.what());
      throw;
    } catch (...) {
      ::solver::v61hsd::failure("non_std_exception");
      throw;
    }
"""),
        ("    info.succeeded = true;\n", "", "    ::solver::v61hsd::committed();\n"),
    ]


def frozen_source(archive=None):
    archive = Path(archive) if archive else Path(__file__).with_name("ipm-v60-source.zip")
    with zipfile.ZipFile(archive) as source:
        return source.read("src/ipm.cpp").decode("utf-8")


def transform(source):
    source = source.replace("\r\n", "\n")
    if "v61hsd" in source or "v61-hsd-diagnostic" in source:
        raise ValueError("输入已经包含 v61 HSD 插桩")
    original = source
    applied, anchors = [], []
    for number, (anchor, before, after) in enumerate(changes()):
        replacement = before + anchor + after
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"HSD 插桩锚点 {number} 必须唯一，实际 {count} 次")
        anchors.append({"index": number,
                        "source_line": original[:original.index(anchor)].count("\n") + 1,
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


def self_check(source=None):
    source = frozen_source() if source is None else source
    transformed, anchors = transform(source)
    try:
        transform(transformed)
    except ValueError:
        pass
    else:
        raise AssertionError("重复插桩没有被拒绝")
    return {"passed": True, "anchor_count": len(anchors), "reversible": True,
            "duplicate_rejected": True, "solver_runs": 0,
            "original_normalized_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
            "transformed_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
