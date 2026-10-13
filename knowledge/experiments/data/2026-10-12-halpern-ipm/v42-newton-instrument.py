"""仅对冻结 v41 追加只读钩子；唯一锚点及逆变换校验保持原算法文本。"""


def additions() -> list[tuple[str, str]]:
    return [
        (
            "struct Point {\n  std::vector<double> x;\n  std::vector<double> s;\n  std::vector<double> z;\n};\n",
            '\n#include "v42-newton-diagnostic.hpp"\n',
        ),
        (
            "void run_ipm(const ConeModel& model, const SolveOptions& options,\n"
            "                Clock::time_point start, SolveResult& result) {\n",
            "  v42_run_start(start);\n",
        ),
        (
            "      for (;;) {\n        try {\n",
            '          v42_phase("predictor", iteration);\n',
        ),
        (
            "          // 预测和校正只更新右端项，共享同一次数值分解。\n",
            '          v42_phase("corrector", iteration);\n',
        ),
        (
            "    for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) rhs[n + m + i] = h[layout.inequality_rows[i]];\n",
            "    V42DirectionScope observation(model_, point, layout, rhs, result);\n",
        ),
        (
            "    } catch (...) {\n      collect();\n",
            "      v42_direction_failure(work);\n",
        ),
        (
            "    collect();\n    Point delta;\n",
            "    v42_work(work);\n",
        ),
        (
            "  const auto tolerance = ipm_full_newton_tolerances(layout, rhs);\n  full_newton_budget(check);\n",
            "  v42_helper_begin(tolerance);\n",
        ),
        (
            "  const auto apply_preconditioner = [&](std::span<const double> value) {\n    full_newton_budget(check);\n",
            "    v42_preconditioner_begin();\n",
        ),
        (
            "    auto answer = precondition(value);\n",
            "    v42_preconditioner_end();\n",
        ),
        (
            "  const auto product = [&](const std::vector<double>& value, std::vector<long double>& image) {\n    full_newton_budget(check);\n",
            "    v42_product_begin();\n",
        ),
        (
            "    multiply(value, image);\n",
            "    v42_product_end();\n",
        ),
        (
            "    full_newton_budget(check);\n    return measure;\n",
            "",
        ),
        (
            "  for (int iteration = 0; ; ++iteration) {\n",
            "    v42_iteration_check(iteration, work);\n",
        ),
        (
            "    const auto correction = apply_preconditioner(physical_residual);\n",
            "",
        ),
        (
            "  for (int restart = 0; restart < 3; ++restart) {\n    full_newton_budget(check);\n",
            "    v42_restart(restart, answer, residual, work);\n",
        ),
        (
            "      directions.push_back(apply_preconditioner(physical_rhs));\n",
            "",
        ),
        (
            "      product(directions.back(), image);\n",
            "      v42_arnoldi_image(col, image, work);\n",
        ),
        (
            "      ++work.krylov_steps;\n",
            "      v42_arnoldi_started(work);\n",
        ),
        (
            "      transformed[col] = finite(static_cast<long double>(transformed[col]) * cosine[col]);\n",
            "      v42_arnoldi_end(remaining, transformed[col + 1]);\n",
        ),
        (
            "        if (assess(candidate, residual) <= 1) return candidate;\n",
            "",
        ),
    ]


# 以下钩子必须放在原语句之前；原语句及控制流保持逐字不变。
BEFORE = {
    "    full_newton_budget(check);\n    return measure;\n":
        "    v42_assessed(answer, residual, measure);\n",
    "      directions.push_back(apply_preconditioner(physical_rhs));\n":
        "      v42_arnoldi_begin(col);\n",
    "    const auto correction = apply_preconditioner(physical_residual);\n":
        "    v42_ir_begin(iteration + 1);\n",
    "        if (assess(candidate, residual) <= 1) return candidate;\n":
        "        v42_candidate_prepare(directions, coefficients, transformed[col + 1], remaining, col + 1 == limit);\n",
}


def instrument(source: str) -> str:
    source = source.replace("\r\n", "\n")
    original = source
    if "v42_" in source or "v42-newton-diagnostic" in source:
        raise ValueError("输入已有 v42 插桩，拒绝重复变换")
    changes = additions()
    for anchor, addition in changes:
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"插桩锚点应唯一，实际 {count} 次：{anchor[:100]!r}")
        before = BEFORE.get(anchor, "")
        source = source.replace(anchor, before + anchor + addition, 1)
    restored = source
    for anchor, addition in reversed(changes):
        inserted = BEFORE.get(anchor, "") + anchor + addition
        if restored.count(inserted) != 1:
            raise ValueError("逆变换锚点不唯一")
        restored = restored.replace(inserted, anchor, 1)
    if restored != original:
        raise ValueError("逆变换未还原原始算法")
    return source
