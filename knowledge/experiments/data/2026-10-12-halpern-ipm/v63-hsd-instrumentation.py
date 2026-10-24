"""冻结 v62 的齐次恢复事件插桩；只追加观察语句并验证逐字逆变换。"""

import hashlib
import zipfile
from pathlib import Path


def changes():
    """每项为原始唯一锚点、前插文本、后插文本。"""
    return [
        ("#include <vector>\n", "", '#include "v63-hsd-diagnostic.hpp"\n'),
        ("class KktFactor {\n public:\n", "", """  std::span<const double> v63_diagonal() const noexcept { return diagonal_h_; }
  std::span<const double> v63_regularized_diagonal() const noexcept { return regularized_h_; }
"""),
        ("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n"
         "    Clock::time_point start, SolveResult& result) {\n", "", "  ::solver::v63hsd::begin_run();\n"),
        ("    const long double affine_h_tau = -static_cast<long double>(tau) * kappa;\n", "",
         '    ::solver::v63hsd::phase("affine", iteration);\n'),
        ("    const auto corrected = scaled_direction ? homogeneous_scaled_direction(model, point, tau, kappa, primal_residual,\n",
         '    ::solver::v63hsd::phase("corrector", iteration);\n', ""),
        ("  HomogeneousDirection direction;\n  if (repair_info) {\n", """  ::solver::v63hsd::Scope v63_scope(model, point, factor.v63_diagonal(),
      factor.v63_regularized_diagonal(), factor.regularization_level(), u, rhs, h,
      primal_residual, dual_residual, tau, kappa, theta, result);
""", ""),
        ("  if (!(denominator > 0) || !std::isfinite(denominator) || !std::isfinite(dot_u)) {\n",
         "  ::solver::v63hsd::observe_values<Scalar>(dot_u, dot_v, denominator);\n", ""),
        ("  full_newton_budget(check);\n  return denominator > 0 && std::isfinite(denominator);\n",
         "  ::solver::v63hsd::application_values(dot_v, denominator);\n", ""),
        ("    info.attempted = true;\n    auto candidate = repair();\n", "", """    ::solver::v63hsd::candidate(candidate);
"""),
        ("    info.attempted = true;\n", "", """    ::solver::v63hsd::begin_repair<Scalar>(u, v, gap_rhs, complementarity_rhs);
    try {
"""),
        ("    info.succeeded = true;\n    return direction;\n", "", """    } catch (const std::exception& v63_failure) {
      ::solver::v63hsd::failure(v63_failure.what());
      throw;
    } catch (...) {
      ::solver::v63hsd::failure("non_std_exception");
      throw;
    }
"""),
        ("    info.succeeded = true;\n", "", "    ::solver::v63hsd::committed();\n"),
        ("    auto answer = direct(rhs, options, start, result);\n"
         "    const double tolerance = 1e-12 + 1e-10 * norm_inf(rhs);\n",
         "    ::solver::v63hsd::linear_begin(rhs, scalar_acceptance != nullptr);\n", ""),
        ("      const long double backward_error = assess_residual(rhs, answer, residual, options, start, cached_product);\n"
         "      const double norm = norm_inf(residual);\n", "", """      ::solver::v63hsd::linear_candidate("ir_candidate", iteration, -1, -1,
          answer, residual, tolerance, block_tolerances ? &*block_tolerances : nullptr, model_.costs.size());
"""),
        ("          accept_homogeneous_scalar(answer, scalar_acceptance, options, start, result)) {\n", "",
         '        ::solver::v63hsd::linear_exit("ir_return");\n'),
        ("    check_budget(options, start);\n    if (!accepted) ++*result.ipm_homogeneous_scalar_rejections;\n",
         "    ::solver::v63hsd::application_return(accepted);\n", ""),
        ("    if (!accepted) ++*result.ipm_homogeneous_scalar_rejections;\n", "",
         "    ::solver::v63hsd::application_accounted();\n"),
        ("    for (int restart = 0; restart < 3; ++restart) {\n      check_budget(options, start);\n", "",
         "      ::solver::v63hsd::cycle_begin(restart);\n"),
        ('      if (!(beta > 0)) throw NumericalFailure("KKT Krylov 初始残差无效");\n',
         '      if (!(beta > 0)) ::solver::v63hsd::linear_exit("invalid_initial_residual");\n', ""),
        ("      for (std::size_t col = 0; col < limit; ++col) {\n        check_budget(options, start);\n", "",
         "        ::solver::v63hsd::column_begin(restart, static_cast<int>(col + 1));\n"),
        ('        if (!(diagonal > 0) || !std::isfinite(diagonal)) throw NumericalFailure("KKT Krylov 子空间退化");\n',
         '        if (!(diagonal > 0) || !std::isfinite(diagonal))\n'
         '          ::solver::v63hsd::linear_exit("degenerate_subspace");\n', ""),
        ("        transformed[col] *= cosine[col];\n", "",
         "        ::solver::v63hsd::column_complete(restart, remaining);\n"),
        ("          const long double backward_error = assess_residual(rhs, candidate, residual, options, start, cached_product);\n", "",
         """          ::solver::v63hsd::linear_candidate("krylov_candidate", -1, restart, static_cast<int>(col + 1),
              candidate, residual, tolerance, block_tolerances, model_.costs.size());
"""),
        ("              accept_homogeneous_scalar(candidate, scalar_acceptance, options, start, result)) {\n", "",
         '            ::solver::v63hsd::linear_exit("krylov_return");\n'),
        ("      answer = std::move(candidate);\n    }\n    throw NumericalFailure(scalar_acceptance ?",
         "      ::solver::v63hsd::cycle_end(restart);\n", ""),
        ('    throw NumericalFailure(scalar_acceptance ? "KKT Krylov 改进未同时满足原方程精度和齐次正分母" :\n',
         '    ::solver::v63hsd::linear_exit("krylov_exhausted");\n', ""),
    ]


def frozen_source(archive=None):
    archive = Path(archive) if archive else Path(__file__).with_name("ipm-v62-source.zip")
    with zipfile.ZipFile(archive) as source:
        return source.read("src/ipm.cpp").decode("utf-8")


def transform(source):
    source = source.replace("\r\n", "\n")
    if "v63hsd" in source or "v63-hsd-diagnostic" in source:
        raise ValueError("输入已经包含 v63 HSD 插桩")
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
