"""v59 冻结副本插桩：只追加采集语句，唯一字面量锚点可逐字逆变换。"""

import hashlib


def changes():
    """原始锚点、前插文本、后插文本；顺序是唯一逆变换的审计合同。"""
    return [
        ("#include <vector>\n", "", '#include "v59-kkt-diagnostic.hpp"\n'),
        ("void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,\n"
         "    Clock::time_point start, SolveResult& result) {\n", "",
         "  ::solver::v59kkt::RunScope v59_run;\n"),
        ("      constant = factor.solve(constant_rhs, options, start, result);\n",
         "      ::solver::v59kkt::ConstantScope v59_constant(iteration);\n", ""),
        ("    auto answer = direct(rhs, options, start, result);\n"
         "    const double tolerance = 1e-12 + 1e-10 * norm_inf(rhs);\n",
         "    ::solver::v59kkt::DirectScope v59_first_direct(rhs, factor_dimension_);\n",
         "    v59_first_direct.complete(answer);\n"),
        ("  void solve_factored(std::vector<double>& rhs, const SolveOptions& options,\n"
         "      Clock::time_point start, SolveResult& result) const {\n", "",
         "    ::solver::v59kkt::FactorScope v59_factor(rhs);\n"),
        ("      const double value = finite(reduced_rhs[i]);\n", "",
         "      ::solver::v59kkt::reduced_rhs(i, value);\n"),
        ('    throw NumericalFailure("KKT Krylov 改进未达到原方程精度");\n',
         "    ::solver::v59kkt::state().failure(model_, diagonal_h_, regularized_h_, ap_, ai_, ax_,\n"
         "        equilibration_.empty() ? ax_ : factor_values_, equilibration_, inverse_, retained_index_,\n"
         "        singleton_column_, singleton_value_, rhs, answer, residual, tolerance, regularization_level_, result);\n", ""),
    ]


def transform(source):
    source = source.replace("\r\n", "\n")
    if "v59kkt" in source or "v59-kkt-diagnostic" in source:
        raise ValueError("输入已经包含 v59 KKT 插桩")
    original = source
    applied = []
    anchors = []
    for number, (anchor, before, after) in enumerate(changes()):
        replacement = before + anchor + after
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"KKT 插桩锚点 {number} 必须唯一，实际 {count} 次")
        anchors.append({"index": number, "source_line": original[:original.index(anchor)].count("\n") + 1,
                        "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
                        "replacement_sha256": hashlib.sha256(replacement.encode()).hexdigest()})
        source = source.replace(anchor, replacement, 1)
        applied.append((anchor, replacement))
    restored = source
    for anchor, replacement in reversed(applied):
        if restored.count(replacement) != 1:
            raise ValueError("KKT 逆变换锚点不是唯一")
        restored = restored.replace(replacement, anchor, 1)
    if restored != original:
        raise ValueError("KKT 插桩逆变换没有还原原始程序")
    return source, anchors


def self_check(source):
    transformed, anchors = transform(source)
    for invalid in (transformed, source + changes()[0][0], source.replace(changes()[0][0], "", 1)):
        try:
            transform(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("重复插桩或非唯一锚点没有被拒绝")
    return {"passed": True, "anchor_count": len(anchors), "reversible": True, "duplicate_rejected": True,
            "missing_anchor_rejected": True, "duplicate_anchor_rejected": True,
            "original_normalized_sha256": hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest(),
            "transformed_sha256": hashlib.sha256(transformed.encode()).hexdigest()}
