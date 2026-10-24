#!/usr/bin/env python3
"""在冻结模型列表上交替比较 CPU 后端或 PDLP 模式，并要求相同的数值质量。"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import uuid

import benchmark


BACKENDS = ("highs", "pdlp")


def file_digest(path):
    try:
        return benchmark.sha256(path), None
    except OSError as error:
        return None, str(error)


def load_models(path):
    content = path.read_bytes()
    names = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if not names:
        raise ValueError("模型列表不能为空")
    paths = [(path.parent / name).resolve() for name in names]
    if len(set(paths)) != len(paths):
        raise ValueError("模型列表含有重复路径；不能重复加权同一模型")
    models = []
    for index, model in enumerate(paths):
        digest, error = file_digest(model)
        models.append({"model_index": index, "input": str(model), "sha256": digest, "hash_error": error})
    return models, hashlib.sha256(content).hexdigest()


def schedule(models, repeats, variants=BACKENDS):
    for model in models:
        for repetition in range(repeats):
            order = variants if (model["model_index"] + repetition) % 2 == 0 else tuple(reversed(variants))
            for position, backend in enumerate(order):
                yield model, repetition, position, backend


def quality_failure(record, frozen, final_digest, expected_solver, current_solver):
    if frozen["sha256"] is None:
        return "input_unavailable_at_freeze"
    if record.get("model_sha256") != frozen["sha256"] or final_digest != frozen["sha256"]:
        return "input_changed"
    if expected_solver is None or current_solver != expected_solver:
        return "solver_changed_or_unavailable"
    if record.get("classification") != "resolved":
        return record.get("classification", "invalid_record")
    report = record.get("report")
    if not isinstance(report, dict) or report.get("status") != "optimal":
        return "not_optimal"
    if benchmark.classify(record["returncode"], report) != "resolved":
        return "invalid_report"
    if report.get("model", {}).get("is_mip") is not False:
        return "not_continuous_lp"
    if not benchmark.valid_lp_optimality(report):
        return "optimality_quality_failed"
    timings = report.get("timings")
    if not isinstance(timings, dict) or not all(
        benchmark.finite_real(timings.get(field)) and timings[field] >= 0
        for field in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")
    ):
        return "invalid_timings"
    if not benchmark.finite_real(record.get("elapsed_wall_seconds")) or record["elapsed_wall_seconds"] <= 0:
        return "invalid_wall_time"
    return None


def numerical_seconds(record):
    report = record.get("report")
    timings = report.get("timings") if isinstance(report, dict) else None
    if not isinstance(timings, dict) or not all(
        benchmark.finite_real(timings.get(field)) and timings[field] >= 0
        for field in ("solve_seconds", "verification_seconds")
    ):
        return None
    # PDLP 的重复停止检查也属于算法工作；加回复核耗时才能共同比较。
    result = timings["solve_seconds"] + timings["verification_seconds"]
    return result if benchmark.finite_real(result) else None


def shifted_geometric_mean(values):
    return math.expm1(sum(math.log1p(value / 10.0) for value in values) / len(values)) * 10.0


def summarize(records, models, repeats, wall_budget, solve_budget, variants=BACKENDS):
    reference, candidate = variants
    legacy_comparison = tuple(variants) == BACKENDS
    distinct_backends = len({variant.split(":", 1)[0] for variant in variants}) == 2
    per_backend = {}
    for backend in variants:
        runs = [record for record in records if record.get("requested_variant", record["requested_backend"]) == backend]
        quality_counts = collections.Counter(record["model_index"] for record in runs if record["quality_pass"])
        per_backend[backend] = {
            "run_count": len(runs), "quality_pass_count": sum(record["quality_pass"] for record in runs),
            "all_repeats_quality_pass_model_count": sum(quality_counts[frozen["model_index"]] == repeats for frozen in models),
            "quality_failure_counts": dict(collections.Counter(record["quality_failure"] for record in runs if not record["quality_pass"])),
            "status_counts": dict(collections.Counter(record.get("status") or "no_status" for record in runs)),
            "penalized_wall_sgm10_seconds": shifted_geometric_mean([record["quality_penalized_wall_seconds"] for record in runs]),
            "penalized_numerical_sgm10_seconds": shifted_geometric_mean([record["quality_penalized_numerical_seconds"] for record in runs]),
            "numerical_budget_exceeded_count": sum(record.get("numerical_budget_exceeded") is True for record in runs),
            "wall_budget_exceeded_count": sum(record.get("wall_budget_exceeded") is True for record in runs),
        }
    per_model = []
    for frozen in models:
        groups = {backend: [record for record in records if record["model_index"] == frozen["model_index"]
                            and record.get("requested_variant", record["requested_backend"]) == backend] for backend in variants}
        complete = all(len(runs) == repeats and all(record["quality_pass"] for record in runs) for runs in groups.values())
        entry = {"model_index": frozen["model_index"], "input": frozen["input"], "sha256": frozen["sha256"],
                 "quality_pass_counts": {backend: sum(record["quality_pass"] for record in runs) for backend, runs in groups.items()},
                 "all_repeats_quality_pass": complete, "median_wall_seconds": None, "median_numerical_seconds": None,
                 "wall_range_seconds": None, "numerical_range_seconds": None,
                 "wall_speedup_reference_over_candidate": None, "numerical_speedup_reference_over_candidate": None,
                 "wall_speedup_highs_over_pdlp": None, "numerical_speedup_highs_over_pdlp": None, "wall_winner": None}
        if complete:
            wall = {backend: statistics.median(record["elapsed_wall_seconds"] for record in runs) for backend, runs in groups.items()}
            numerical = {backend: statistics.median(numerical_seconds(record) for record in runs) for backend, runs in groups.items()}
            entry.update(median_wall_seconds=wall, median_numerical_seconds=numerical,
                         wall_range_seconds={backend: [min(record["elapsed_wall_seconds"] for record in runs),
                                                       max(record["elapsed_wall_seconds"] for record in runs)]
                                             for backend, runs in groups.items()},
                         numerical_range_seconds={backend: [min(numerical_seconds(record) for record in runs),
                                                            max(numerical_seconds(record) for record in runs)]
                                                  for backend, runs in groups.items()},
                         wall_speedup_reference_over_candidate=wall[reference] / wall[candidate],
                         numerical_speedup_reference_over_candidate=numerical[reference] / numerical[candidate] if numerical[candidate] > 0 else None,
                         wall_winner="tie" if wall[reference] == wall[candidate] else min(wall, key=wall.get))
            if legacy_comparison:
                entry["wall_speedup_highs_over_pdlp"] = entry["wall_speedup_reference_over_candidate"]
                entry["numerical_speedup_highs_over_pdlp"] = entry["numerical_speedup_reference_over_candidate"]
        per_model.append(entry)
    return {"schema_version": 1, "model_count": len(models), "repeats": repeats, "run_count": len(records),
            "quality_pass_count": sum(record["quality_pass"] for record in records),
            "reference_variant": reference, "candidate_variant": candidate,
            "both_variants_all_repeats_pass_count": sum(entry["all_repeats_quality_pass"] for entry in per_model),
            "both_backends_all_repeats_pass_count": sum(entry["all_repeats_quality_pass"] for entry in per_model) if distinct_backends else None,
            "wall_winner_counts": dict(collections.Counter(entry["wall_winner"] for entry in per_model if entry["wall_winner"])),
            "failure_wall_penalty_seconds": wall_budget, "failure_numerical_penalty_seconds": solve_budget,
            "variants": per_backend, "backends": per_backend if distinct_backends else None, "models": per_model,
            "timing_definition": "墙钟截至子进程结束与输出收集完成，不含驱动 JSON 解析；数值阶段为 solve_seconds + verification_seconds，保留全部停止检查和共同质量复核成本。后端预算并非硬中断；完整数值时间超出 time_limit 的结果不计预算内达标。",
            "aggregation_rule": "SGM 对所有预定模型和重复等权计入，失败不删除；逐模型速度比另外使用中位数，不将两种聚合混用。",
            "interpretation": "速度比只来自双方所有重复均达标的模型；失败的预算惩罚是评分，不是精确耗时。wall_winner 仅表示观测中位数更小，不代表统计显著；同时给出重复测量范围。结果仅描述此冻结列表，不代表广泛性能领先。"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--reference-solver", type=Path,
                        help="参考侧独立冻结的可执行文件；省略时双方使用 --solver")
    parser.add_argument("--models-file", required=True, type=Path, help="UTF-8 文本，一行一个模型路径，相对路径以列表目录为基准")
    parser.add_argument("--repeats", type=benchmark.positive_integer, default=3)
    parser.add_argument("--time-limit", type=benchmark.finite_positive, default=60.0)
    parser.add_argument("--wall-timeout", type=benchmark.finite_positive, help="默认求解预算加 30 秒")
    parser.add_argument("--threads", type=benchmark.positive_integer, default=1)
    parser.add_argument("--seed", type=benchmark.nonnegative_integer, default=0)
    parser.add_argument("--candidate-backend", choices=benchmark.BACKENDS, default="pdlp")
    parser.add_argument("--reference-backend", choices=benchmark.BACKENDS,
                        help="默认 highs；显式参考 PDLP 模式可兼容选择 pdlp")
    parser.add_argument("--iteration-limit", type=benchmark.nonnegative_int64,
                        help="待测自主后端的迭代预算；同后端消融默认共用")
    parser.add_argument("--scaling-passes", type=benchmark.scaling_passes,
                        help="待测自主后端的缩放轮数；同后端消融默认共用")
    parser.add_argument("--reference-iteration-limit", type=benchmark.nonnegative_int64,
                        help="参考自主后端的迭代预算；异构对照省略时采用其自身默认值")
    parser.add_argument("--reference-scaling-passes", type=benchmark.scaling_passes)
    parser.add_argument("--ipm-method", choices=benchmark.IPM_METHODS,
                        help="待测 IPM 的 mehrotra 或齐次方法 homogeneous，省略时 mehrotra")
    parser.add_argument("--reference-ipm-method", choices=benchmark.IPM_METHODS,
                        help="参考 IPM 的方法；始终独立于待测配置，省略时 mehrotra")
    parser.add_argument("--ipm-singleton-elimination", choices=("on", "off"),
                        help="待测 IPM 的单非零不等式 KKT 消元；省略时关闭")
    parser.add_argument("--reference-ipm-singleton-elimination", choices=("on", "off"),
                        help="参考 IPM 的消元开关；始终独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-dynamic-regularization", choices=("on", "off"),
                        help="待测 IPM 的动态主元正则化；省略时关闭")
    parser.add_argument("--reference-ipm-dynamic-regularization", choices=("on", "off"),
                        help="参考 IPM 的动态正则开关；始终独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-kkt-equilibration", choices=("on", "off"),
                        help="待测 IPM 的 KKT 均衡化；省略时关闭")
    parser.add_argument("--reference-ipm-kkt-equilibration", choices=("on", "off"),
                        help="参考 IPM 的 KKT 均衡化；始终独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-objective-scaling", choices=("on", "off"),
                        help="待测 IPM 的目标归一化，省略时开启，与行列缩放轮数独立")
    parser.add_argument("--reference-ipm-objective-scaling", choices=("on", "off"),
                        help="参考 IPM 的目标归一化；始终独立于待测配置，省略时开启")
    parser.add_argument("--ipm-linear-accuracy", choices=benchmark.IPM_LINEAR_ACCURACIES,
                        help="待测 IPM 的线性系统精度规则，含 inexact、blockwise、blockwise_weighted、blockwise_forcing、blockwise_forcing_weighted、adaptive_regularization、primal_dual_regularization、curvature_regularization、full_newton、bounded_newton、homogeneous_scaled、homogeneous_repair；省略时 strict；homogeneous_scaled 仅用于 homogeneous，有限精化后保留最佳有限方向，仍须原模型终验；homogeneous_repair 仅用于 homogeneous，在有限非正标量分母时每轮至多按块严格重算一次常数方向")
    parser.add_argument("--reference-ipm-linear-accuracy", choices=benchmark.IPM_LINEAR_ACCURACIES,
                        help="参考 IPM 的线性系统精度规则，含 inexact、blockwise、blockwise_weighted、blockwise_forcing、blockwise_forcing_weighted、adaptive_regularization、primal_dual_regularization、curvature_regularization、full_newton、bounded_newton、homogeneous_scaled、homogeneous_repair；始终独立于待测配置，省略时 strict；homogeneous_scaled 仅用于 homogeneous，有限精化后保留最佳有限方向，仍须原模型终验；homogeneous_repair 仅用于 homogeneous，在有限非正标量分母时每轮至多按块严格重算一次常数方向")
    parser.add_argument("--ipm-central-corrections", type=benchmark.nonnegative_integer, choices=(0, 1),
                        help="待测 IPM 每步中央修正次数；省略时 0")
    parser.add_argument("--reference-ipm-central-corrections", type=benchmark.nonnegative_integer, choices=(0, 1),
                        help="参考 IPM 每步中央修正次数；始终独立于待测配置，省略时 0")
    parser.add_argument("--ipm-direction-retries", type=benchmark.nonnegative_integer, choices=range(9),
                        help="待测 IPM 方向失败后的正则重试预算，0 到 8；省略时 0；正预算支持 strict、primal_dual_regularization、curvature_regularization、blockwise、blockwise_weighted、full_newton")
    parser.add_argument("--reference-ipm-direction-retries", type=benchmark.nonnegative_integer, choices=range(9),
                        help="参考 IPM 的方向重试预算；始终独立于待测配置，省略时 0；正预算支持 strict、primal_dual_regularization、curvature_regularization、blockwise、blockwise_weighted、full_newton")
    parser.add_argument("--ipm-krylov-history-size", type=benchmark.nonnegative_integer, choices=range(9),
                        help="待测侧同一 KKT 的 Krylov 历史维数，0 到 8，默认 0；正值仅支持 strict、blockwise")
    parser.add_argument("--reference-ipm-krylov-history-size", type=benchmark.nonnegative_integer, choices=range(9),
                        help="参考侧独立 Krylov 历史维数，0 到 8，默认 0；正值仅支持 strict、blockwise")
    parser.add_argument("--ipm-linear-solver", choices=benchmark.IPM_LINEAR_SOLVERS,
                        help="待测 IPM 的稀疏分解实现，可选 qdldl、sparselu、normal、qdldl_extended、factorhighs、normal_augmented、normal_multifrontal、normal_extended、normal_refined、normal_cholmod、normal_cholmod_aat；省略时 qdldl；normal_cholmod 保持原 Gram 和 AMD，使用 CHOLMOD 超节点分解；normal_cholmod_aat 保持相同 AMD 和结构预算，使用 CHOLMOD 矩形输入数值乘积")
    parser.add_argument("--reference-ipm-linear-solver", choices=benchmark.IPM_LINEAR_SOLVERS,
                        help="参考 IPM 的稀疏分解实现；始终独立于待测配置，省略时 qdldl")
    parser.add_argument("--ipm-inexact-prefilter", choices=("on", "off"),
                        help="待测 IPM 的非精确方向预筛选；省略时关闭")
    parser.add_argument("--reference-ipm-inexact-prefilter", choices=("on", "off"),
                        help="参考 IPM 的非精确方向预筛选；始终独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-fixed-variable-elimination", choices=("on", "off"),
                        help="待测 IPM 的固定变量代入；省略时关闭")
    parser.add_argument("--reference-ipm-fixed-variable-elimination", choices=("on", "off"),
                        help="参考 IPM 的固定变量代入；始终独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-singleton-presolve", choices=("on", "off"),
                        help="待测 IPM 的单非零输入行归并；省略时关闭")
    parser.add_argument("--reference-ipm-singleton-presolve", choices=("on", "off"),
                        help="参考 IPM 的单非零输入行归并；独立于待测配置，省略时关闭")
    parser.add_argument("--ipm-presolve-fixed-elimination", choices=("on", "off"),
                        help="待测 IPM 消除归并后新固定的变量；默认关闭，开启要求待测输入归并开启")
    parser.add_argument("--reference-ipm-presolve-fixed-elimination", choices=("on", "off"),
                        help="参考 IPM 独立消除归并后新固定的变量；默认关闭，开启要求参考输入归并开启")
    parser.add_argument("--highs-solver", choices=benchmark.HIGHS_SOLVERS)
    parser.add_argument("--highs-crossover", choices=("on", "off"))
    parser.add_argument("--highs-presolve", choices=("choose", "on", "off"))
    parser.add_argument("--reference-highs-solver", choices=benchmark.HIGHS_SOLVERS)
    parser.add_argument("--reference-highs-crossover", choices=("on", "off"))
    parser.add_argument("--reference-highs-presolve", choices=("choose", "on", "off"))
    parser.add_argument("--pdlp-mode", choices=benchmark.PDLP_MODES)
    parser.add_argument("--pdlp-history-size", type=benchmark.history_size,
                        help="待测 PDLP 的历史数量；省略时使用求解器默认值并保留实际报告")
    parser.add_argument("--reference-pdlp-mode", choices=benchmark.PDLP_MODES,
                        help="指定时以此 PDLP 模式替代 HiGHS 作为参考，进行成对消融")
    parser.add_argument("--reference-pdlp-history-size", type=benchmark.history_size,
                        help="参考 PDLP 的历史数量；省略时使用求解器默认值并保留实际报告")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    args.reference_backend = args.reference_backend or ("pdlp" if args.reference_pdlp_mode is not None else "highs")
    if args.candidate_backend != "highs" and any(value is not None for value in (args.highs_solver, args.highs_crossover, args.highs_presolve)):
        parser.error("待测 HiGHS 参数要求 --candidate-backend highs")
    if args.reference_backend != "highs" and any(value is not None for value in (args.reference_highs_solver, args.reference_highs_crossover, args.reference_highs_presolve)):
        parser.error("参考 HiGHS 参数要求 --reference-backend highs")
    if args.candidate_backend != "pdlp" and (args.pdlp_mode is not None or args.pdlp_history_size is not None):
        parser.error("--pdlp-mode 和 --pdlp-history-size 要求 --candidate-backend pdlp")
    if args.reference_backend != "pdlp" and (args.reference_pdlp_mode is not None or args.reference_pdlp_history_size is not None):
        parser.error("参考 PDLP 参数要求 --reference-backend pdlp 或 --reference-pdlp-mode")
    if args.candidate_backend != "ipm" and args.ipm_method is not None:
        parser.error("--ipm-method 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_method is not None:
        parser.error("--reference-ipm-method 要求 --reference-backend ipm")
    benchmark.validate_ipm_method(parser, args)
    benchmark.validate_ipm_method(parser, args, "reference_")
    if args.candidate_backend != "ipm" and args.ipm_singleton_elimination is not None:
        parser.error("--ipm-singleton-elimination 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_singleton_elimination is not None:
        parser.error("--reference-ipm-singleton-elimination 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_dynamic_regularization is not None:
        parser.error("--ipm-dynamic-regularization 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_dynamic_regularization is not None:
        parser.error("--reference-ipm-dynamic-regularization 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_kkt_equilibration is not None:
        parser.error("--ipm-kkt-equilibration 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_kkt_equilibration is not None:
        parser.error("--reference-ipm-kkt-equilibration 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_objective_scaling is not None:
        parser.error("--ipm-objective-scaling 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_objective_scaling is not None:
        parser.error("--reference-ipm-objective-scaling 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_linear_accuracy is not None:
        parser.error("--ipm-linear-accuracy 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_linear_accuracy is not None:
        parser.error("--reference-ipm-linear-accuracy 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_central_corrections is not None:
        parser.error("--ipm-central-corrections 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_central_corrections is not None:
        parser.error("--reference-ipm-central-corrections 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_direction_retries is not None:
        parser.error("--ipm-direction-retries 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_direction_retries is not None:
        parser.error("--reference-ipm-direction-retries 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_krylov_history_size is not None:
        parser.error("--ipm-krylov-history-size 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_krylov_history_size is not None:
        parser.error("--reference-ipm-krylov-history-size 要求 --reference-backend ipm")
    if args.ipm_linear_accuracy in ("full_newton", "bounded_newton") and args.ipm_central_corrections:
        parser.error("待测 full_newton 和 bounded_newton 暂不支持正 --ipm-central-corrections")
    if args.reference_ipm_linear_accuracy in ("full_newton", "bounded_newton") and args.reference_ipm_central_corrections:
        parser.error("参考 full_newton 和 bounded_newton 暂不支持正 --reference-ipm-central-corrections")
    if args.ipm_linear_accuracy == "bounded_newton" and args.ipm_direction_retries:
        parser.error("待测 bounded_newton 暂不支持正 --ipm-direction-retries")
    if args.reference_ipm_linear_accuracy == "bounded_newton" and args.reference_ipm_direction_retries:
        parser.error("参考 bounded_newton 暂不支持正 --reference-ipm-direction-retries")
    if args.ipm_krylov_history_size and (args.ipm_linear_accuracy or "strict") not in ("strict", "blockwise"):
        parser.error("正 --ipm-krylov-history-size 仅支持待测侧 strict 或 blockwise 精度")
    if args.reference_ipm_krylov_history_size and (args.reference_ipm_linear_accuracy or "strict") not in ("strict", "blockwise"):
        parser.error("正 --reference-ipm-krylov-history-size 仅支持参考侧 strict 或 blockwise 精度")
    if args.candidate_backend != "ipm" and args.ipm_linear_solver is not None:
        parser.error("--ipm-linear-solver 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_linear_solver is not None:
        parser.error("--reference-ipm-linear-solver 要求 --reference-backend ipm")
    benchmark.validate_ipm_linear_solver(parser, args)
    benchmark.validate_ipm_linear_solver(parser, args, "reference_")
    if args.candidate_backend != "ipm" and args.ipm_fixed_variable_elimination is not None:
        parser.error("--ipm-fixed-variable-elimination 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_fixed_variable_elimination is not None:
        parser.error("--reference-ipm-fixed-variable-elimination 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_singleton_presolve is not None:
        parser.error("--ipm-singleton-presolve 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_singleton_presolve is not None:
        parser.error("--reference-ipm-singleton-presolve 要求 --reference-backend ipm")
    if args.candidate_backend != "ipm" and args.ipm_presolve_fixed_elimination is not None:
        parser.error("--ipm-presolve-fixed-elimination 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_presolve_fixed_elimination is not None:
        parser.error("--reference-ipm-presolve-fixed-elimination 要求 --reference-backend ipm")
    if args.ipm_presolve_fixed_elimination == "on" and args.ipm_singleton_presolve != "on":
        parser.error("--ipm-presolve-fixed-elimination on 要求 --ipm-singleton-presolve on")
    if args.reference_ipm_presolve_fixed_elimination == "on" and args.reference_ipm_singleton_presolve != "on":
        parser.error("--reference-ipm-presolve-fixed-elimination on 要求 --reference-ipm-singleton-presolve on")
    if args.candidate_backend != "ipm" and args.ipm_inexact_prefilter is not None:
        parser.error("--ipm-inexact-prefilter 要求 --candidate-backend ipm")
    if args.reference_backend != "ipm" and args.reference_ipm_inexact_prefilter is not None:
        parser.error("--reference-ipm-inexact-prefilter 要求 --reference-backend ipm")
    if args.candidate_backend == "highs" and (args.iteration_limit is not None or args.scaling_passes is not None):
        parser.error("待测迭代和缩放参数仅适用于 pdlp 或 ipm")
    if args.reference_backend == "highs" and (args.reference_iteration_limit is not None or args.reference_scaling_passes is not None):
        parser.error("参考迭代和缩放参数仅适用于 pdlp 或 ipm")
    args.pdlp_mode = (args.pdlp_mode or "adaptive") if args.candidate_backend == "pdlp" else None
    args.reference_pdlp_mode = (args.reference_pdlp_mode or "adaptive") if args.reference_backend == "pdlp" else None
    same_backend = args.reference_backend == args.candidate_backend
    if same_backend:
        if args.reference_iteration_limit is None:
            args.reference_iteration_limit = args.iteration_limit
        if args.reference_scaling_passes is None:
            args.reference_scaling_passes = args.scaling_passes
    reference = {"backend": args.reference_backend, "mode": args.reference_pdlp_mode,
                 "ipm_method": args.reference_ipm_method,
                 "history_size": args.reference_pdlp_history_size,
                 "highs_solver": args.reference_highs_solver, "highs_crossover": args.reference_highs_crossover,
                 "highs_presolve": args.reference_highs_presolve,
                 "ipm_singleton_elimination": args.reference_ipm_singleton_elimination,
                 "ipm_dynamic_regularization": args.reference_ipm_dynamic_regularization,
                 "ipm_kkt_equilibration": args.reference_ipm_kkt_equilibration,
                 "ipm_objective_scaling": args.reference_ipm_objective_scaling,
                 "ipm_linear_accuracy": args.reference_ipm_linear_accuracy,
                 "ipm_central_corrections": args.reference_ipm_central_corrections,
                 "ipm_direction_retries": args.reference_ipm_direction_retries,
                 "ipm_krylov_history_size": args.reference_ipm_krylov_history_size,
                 "ipm_linear_solver": args.reference_ipm_linear_solver,
                 "ipm_inexact_prefilter": args.reference_ipm_inexact_prefilter,
                 "ipm_fixed_variable_elimination": args.reference_ipm_fixed_variable_elimination,
                 "ipm_singleton_presolve": args.reference_ipm_singleton_presolve,
                 "ipm_presolve_fixed_elimination": args.reference_ipm_presolve_fixed_elimination,
                 "iteration_limit": args.reference_iteration_limit, "scaling_passes": args.reference_scaling_passes}
    candidate = {"backend": args.candidate_backend, "mode": args.pdlp_mode,
                 "ipm_method": args.ipm_method,
                 "history_size": args.pdlp_history_size,
                 "highs_solver": args.highs_solver, "highs_crossover": args.highs_crossover,
                 "highs_presolve": args.highs_presolve,
                 "ipm_singleton_elimination": args.ipm_singleton_elimination,
                 "ipm_dynamic_regularization": args.ipm_dynamic_regularization,
                 "ipm_kkt_equilibration": args.ipm_kkt_equilibration,
                 "ipm_objective_scaling": args.ipm_objective_scaling,
                 "ipm_linear_accuracy": args.ipm_linear_accuracy,
                 "ipm_central_corrections": args.ipm_central_corrections,
                 "ipm_direction_retries": args.ipm_direction_retries,
                 "ipm_krylov_history_size": args.ipm_krylov_history_size,
                 "ipm_linear_solver": args.ipm_linear_solver,
                 "ipm_inexact_prefilter": args.ipm_inexact_prefilter,
                 "ipm_fixed_variable_elimination": args.ipm_fixed_variable_elimination,
                 "ipm_singleton_presolve": args.ipm_singleton_presolve,
                 "ipm_presolve_fixed_elimination": args.ipm_presolve_fixed_elimination,
                 "iteration_limit": args.iteration_limit, "scaling_passes": args.scaling_passes}
    args.solver = args.solver.resolve()
    args.reference_solver = (args.reference_solver or args.solver).resolve()
    different_binaries = args.reference_solver != args.solver
    if reference == candidate and not different_binaries:
        parser.error("参考和待测配置完全相同，无法形成对照")
    configurations = {}
    for side, configuration in [("reference", reference), ("candidate", candidate)]:
        label = configuration["backend"]
        if label == "highs" and (same_backend or any(configuration[field] is not None for field in ("highs_solver", "highs_crossover", "highs_presolve"))):
            label += ":solver=" + (configuration["highs_solver"] or "choose") + ":crossover=" + (configuration["highs_crossover"] or "on")
            label += ":presolve=" + (configuration["highs_presolve"] or "choose")
        if label == "ipm" and (same_backend or any(configuration[field] is not None
                              for field in ("ipm_singleton_elimination", "ipm_dynamic_regularization",
                                            "ipm_kkt_equilibration", "ipm_linear_accuracy", "ipm_central_corrections",
                                            "ipm_linear_solver", "ipm_inexact_prefilter", "ipm_fixed_variable_elimination"))):
            label += ":singletons=" + (configuration["ipm_singleton_elimination"] or "off")
            label += ":dynamic=" + (configuration["ipm_dynamic_regularization"] or "off")
            label += ":equilibrate=" + (configuration["ipm_kkt_equilibration"] or "off")
            label += ":linear=" + (configuration["ipm_linear_accuracy"] or "strict")
            label += ":corrections=" + str(configuration["ipm_central_corrections"] or 0)
            label += ":solver=" + (configuration["ipm_linear_solver"] or "qdldl")
            label += ":prefilter=" + (configuration["ipm_inexact_prefilter"] or "off")
            label += ":fixed=" + (configuration["ipm_fixed_variable_elimination"] or "off")
        if same_backend:
            if label == "pdlp":
                size = configuration["history_size"]
                label += ":" + configuration["mode"] + ":history=" + ("default" if size is None else str(size))
            if reference["iteration_limit"] != candidate["iteration_limit"] or reference["scaling_passes"] != candidate["scaling_passes"]:
                label += ":iterations=" + str(configuration["iteration_limit"]) + ":scaling=" + str(configuration["scaling_passes"])
        if different_binaries:
            label += ":binary=" + side
        if configuration["backend"] == "ipm" and configuration["ipm_direction_retries"]:
            label += ":retries=" + str(configuration["ipm_direction_retries"])
        if configuration["backend"] == "ipm" and configuration["ipm_singleton_presolve"] == "on":
            label += ":presolve_singletons=on"
        if configuration["backend"] == "ipm" and configuration["ipm_presolve_fixed_elimination"] == "on":
            label += ":presolve_fixed=on"
        if configuration["backend"] == "ipm" and configuration["ipm_krylov_history_size"]:
            label += ":krylov_history=" + str(configuration["ipm_krylov_history_size"])
        if configuration["backend"] == "ipm" and configuration["ipm_method"] == "homogeneous":
            label += ":method=homogeneous"
        if configuration["backend"] == "ipm" and configuration["ipm_objective_scaling"] == "off":
            label += ":objective_scaling=off"
        configurations[label] = configuration
    if len(configurations) != 2:
        parser.error("参考和待测的实际配置相同，无法形成独立对照标签")
    variants = tuple(configurations)
    if args.threads != 1:
        parser.error("当前共同 CPU 对照只支持 --threads 1")
    args.wall_timeout = args.wall_timeout if args.wall_timeout is not None else args.time_limit + 30.0
    if args.wall_timeout < args.time_limit:
        parser.error("--wall-timeout 不能小于 --time-limit")
    args.solver = args.solver.resolve()
    args.models_file = args.models_file.resolve()
    try:
        models, list_digest = load_models(args.models_file)
    except (OSError, ValueError) as error:
        parser.error("无法冻结模型列表：" + str(error))
    if args.output_dir is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = Path("results") / ("comparison-" + stamp + "-" + uuid.uuid4().hex[:8])
    args.output_dir = args.output_dir.resolve()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("不能新建结果目录：" + str(error))
    solver_digest, solver_error = file_digest(args.solver)
    variant_solvers = {variants[0]: args.reference_solver, variants[1]: args.solver}
    frozen_solvers = {}
    for variant, executable in variant_solvers.items():
        digest, error = file_digest(executable)
        frozen_solvers[variant] = {"path": str(executable), "sha256": digest, "hash_error": error}
    manifest = {"schema_version": 1, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "comparison_script_sha256": file_digest(Path(__file__))[0],
                "benchmark_script_sha256": file_digest(Path(benchmark.__file__))[0],
                "solver": str(args.solver), "solver_sha256": solver_digest, "solver_hash_error": solver_error,
                "solvers": frozen_solvers,
                "models_file": str(args.models_file), "models_file_sha256": list_digest, "models": models,
                "repeats": args.repeats, "time_limit": args.time_limit, "wall_timeout": args.wall_timeout,
                "threads": args.threads, "seed": args.seed, "pdlp_mode": args.pdlp_mode,
                "candidate_backend": args.candidate_backend, "reference_backend": args.reference_backend,
                "pdlp_history_size": args.pdlp_history_size,
                "reference_pdlp_mode": args.reference_pdlp_mode,
                "reference_pdlp_history_size": args.reference_pdlp_history_size,
                "reference_variant": variants[0], "candidate_variant": variants[1], "variants": configurations,
                "iteration_limit": args.iteration_limit, "scaling_passes": args.scaling_passes,
                "reference_iteration_limit": args.reference_iteration_limit,
                "reference_scaling_passes": args.reference_scaling_passes,
                "ipm_method": args.ipm_method,
                "reference_ipm_method": args.reference_ipm_method,
                "ipm_singleton_elimination": args.ipm_singleton_elimination,
                "reference_ipm_singleton_elimination": args.reference_ipm_singleton_elimination,
                "ipm_dynamic_regularization": args.ipm_dynamic_regularization,
                "reference_ipm_dynamic_regularization": args.reference_ipm_dynamic_regularization,
                "ipm_kkt_equilibration": args.ipm_kkt_equilibration,
                "reference_ipm_kkt_equilibration": args.reference_ipm_kkt_equilibration,
                "ipm_objective_scaling": args.ipm_objective_scaling,
                "reference_ipm_objective_scaling": args.reference_ipm_objective_scaling,
                "ipm_linear_accuracy": args.ipm_linear_accuracy,
                "reference_ipm_linear_accuracy": args.reference_ipm_linear_accuracy,
                "ipm_central_corrections": args.ipm_central_corrections,
                "reference_ipm_central_corrections": args.reference_ipm_central_corrections,
                "ipm_direction_retries": args.ipm_direction_retries,
                "ipm_krylov_history_size": args.ipm_krylov_history_size,
                "reference_ipm_direction_retries": args.reference_ipm_direction_retries,
                "reference_ipm_krylov_history_size": args.reference_ipm_krylov_history_size,
                "ipm_linear_solver": args.ipm_linear_solver,
                "reference_ipm_linear_solver": args.reference_ipm_linear_solver,
                "ipm_inexact_prefilter": args.ipm_inexact_prefilter,
                "reference_ipm_inexact_prefilter": args.reference_ipm_inexact_prefilter,
                "ipm_fixed_variable_elimination": args.ipm_fixed_variable_elimination,
                "reference_ipm_fixed_variable_elimination": args.reference_ipm_fixed_variable_elimination,
                "ipm_singleton_presolve": args.ipm_singleton_presolve,
                "reference_ipm_singleton_presolve": args.reference_ipm_singleton_presolve,
                "ipm_presolve_fixed_elimination": args.ipm_presolve_fixed_elimination,
                "reference_ipm_presolve_fixed_elimination": args.reference_ipm_presolve_fixed_elimination,
                "platform": platform.platform(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
                "python_version": platform.python_version(), "execution": "本机 CPU，串行执行各次测量；模型哈希读取会预热文件缓存，本实验不衡量磁盘冷启动。",
                "order_rule": "模型序号加重复序号为偶数时先 reference_variant，否则先 candidate_variant；每对使用相同模型、种子与预算。",
                "quality_rule": "仅 optimal、独立候选检查和原模型 LP 最优性四项布尔检查全通过的连续 LP 达标；不放宽任何后端的检查容差。",
                "penalty_rule": "质量失败、超时和运行失败按事先冻结的墙钟/求解预算分别惩罚；达标墙钟和数值阶段耗时按对应预算截断。数值阶段为 solve_seconds + verification_seconds，所有原始耗时仍完整保留。",
                "wall_timing_rule": "只计子进程运行与输出收集，不计驱动 JSON 解码和分类；超过墙钟预算的结果不纳入达标速度比。",
                "solve_budget_rule": "time_limit 由后端实施；数值阶段使用 solve_seconds + verification_seconds 的完整时间，超过预算不计成功。原始耗时和超限次数保留，不以截断评分掩盖超限。"}
    try:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        records = []
        with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as output:
            for frozen, repetition, position, variant in schedule(models, args.repeats, variants):
                configuration = configurations[variant]
                backend = configuration["backend"]
                executable = variant_solvers[variant]
                options = argparse.Namespace(solver=executable, backend=backend, time_limit=args.time_limit,
                                             wall_timeout=args.wall_timeout, threads=args.threads, mip_gap=1e-4,
                                             iteration_limit=configuration["iteration_limit"],
                                             scaling_passes=configuration["scaling_passes"],
                                             pdlp_mode=configuration["mode"],
                                             pdlp_history_size=configuration["history_size"],
                                             ipm_method=configuration["ipm_method"],
                                             ipm_singleton_elimination=configuration["ipm_singleton_elimination"],
                                             ipm_dynamic_regularization=configuration["ipm_dynamic_regularization"],
                                             ipm_kkt_equilibration=configuration["ipm_kkt_equilibration"],
                                             ipm_objective_scaling=configuration["ipm_objective_scaling"],
                                             ipm_linear_accuracy=configuration["ipm_linear_accuracy"],
                                             ipm_central_corrections=configuration["ipm_central_corrections"],
                                             ipm_direction_retries=configuration["ipm_direction_retries"],
                                             ipm_krylov_history_size=configuration["ipm_krylov_history_size"],
                                             ipm_linear_solver=configuration["ipm_linear_solver"],
                                             ipm_inexact_prefilter=configuration["ipm_inexact_prefilter"],
                                             ipm_fixed_variable_elimination=configuration["ipm_fixed_variable_elimination"],
                                             ipm_singleton_presolve=configuration["ipm_singleton_presolve"],
                                             ipm_presolve_fixed_elimination=configuration["ipm_presolve_fixed_elimination"],
                                             highs_solver=configuration["highs_solver"],
                                             highs_crossover=configuration["highs_crossover"],
                                             highs_presolve=configuration["highs_presolve"])
                current_solver, _ = file_digest(executable)
                record = benchmark.run_one(options, Path(frozen["input"]), args.seed, len(records))
                final_digest, _ = file_digest(Path(frozen["input"]))
                final_solver, _ = file_digest(executable)
                stable_solver = current_solver if current_solver == final_solver else None
                failure = quality_failure(record, frozen, final_digest, frozen_solvers[variant]["sha256"], stable_solver)
                wall_exceeded = record["elapsed_wall_seconds"] > args.wall_timeout
                measured_numerical = numerical_seconds(record)
                if failure is None and measured_numerical is None:
                    failure = "invalid_timings"
                if failure is None and wall_exceeded:
                    failure = "wall_budget_exceeded"
                if failure is None and measured_numerical > args.time_limit:
                    failure = "numerical_budget_exceeded"
                record.update(model_index=frozen["model_index"], repetition=repetition, position_in_pair=position,
                              requested_variant=variant,
                              frozen_model_sha256=frozen["sha256"], model_sha256_after=final_digest,
                              solver_sha256_before=current_solver, solver_sha256_after=final_solver,
                              frozen_solver_sha256=frozen_solvers[variant]["sha256"],
                              numerical_seconds=measured_numerical,
                              numerical_budget_exceeded=measured_numerical > args.time_limit if benchmark.finite_real(measured_numerical) else None,
                              wall_budget_exceeded=wall_exceeded,
                              quality_pass=failure is None, quality_failure=failure)
                record["quality_penalized_wall_seconds"] = min(record["elapsed_wall_seconds"], args.wall_timeout) if failure is None else args.wall_timeout
                record["quality_penalized_numerical_seconds"] = min(measured_numerical, args.time_limit) if failure is None else args.time_limit
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                records.append(record)
        summary = summarize(records, models, args.repeats, args.wall_timeout, args.time_limit, variants)
        summary["configurations"] = configurations
        summary["output_dir"] = str(args.output_dir)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0 if summary["quality_pass_count"] == len(records) else 1
    except OSError as error:
        print("写入对照记录失败：" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
