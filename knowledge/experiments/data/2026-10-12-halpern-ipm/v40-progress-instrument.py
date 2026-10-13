"""仅对冻结的 v39 源码插入只读观测钩子，每个锚点必须唯一。"""


def instrument(source: str) -> str:
    source = source.replace("\r\n", "\n")
    changes = [
        (
            "struct Point {\n  std::vector<double> x;\n  std::vector<double> s;\n  std::vector<double> z;\n};\n",
            '\n#include "v40-progress-diagnostic.hpp"\n',
        ),
        (
            "void run_ipm(const ConeModel& model, const SolveOptions& options,\n"
            "                Clock::time_point start, SolveResult& result) {\n",
            "  v40_run_start(options, start);\n",
        ),
        (
            "      const double mu = model.barrier_rows > 0 ? finite(complementarity / model.barrier_rows) : 0;\n",
            "      v40_begin(model, point, primal_residual, dual_residual, mu, iteration, options, result);\n",
        ),
        (
            '      const double dual_step = std::min(1.0, 0.995 * boundary_step(model, point.z, corrected.z));\n',
            "      v40_before_step(model, point, corrected, primal_residual, dual_residual, diagonal, h,\n"
            "          primal_step, dual_step, sigma, mu, iteration, factor.regularization_level(), options, result);\n",
        ),
        (
            "      result.ipm_iterations = iteration;\n",
            "      v40_after_step(model, point, iteration, options, result);\n",
        ),
        (
            "  auto checks = verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality);\n"
            "  result.verification_seconds += elapsed(start);\n",
            "  v40_current_assess(checks, result);\n",
        ),
    ]
    if "v40_" in source or "v40-progress-diagnostic" in source:
        raise ValueError("输入已含 v40 插桩，拒绝重复变换")
    for anchor, addition in changes:
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"插桩锚点应唯一，实际 {count} 次：{anchor[:100]!r}")
        source = source.replace(anchor, anchor + addition, 1)
    return source
