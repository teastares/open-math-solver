#!/usr/bin/env python3
"""按固定预算逐模型、逐种子运行求解器，并保留失败记录。"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid


RESOLVED = {"optimal", "empty", "infeasible", "unbounded"}
BACKENDS = ("highs", "pdlp", "ipm")
PDLP_MODES = ("baseline", "adaptive", "subspace", "halpern", "halpern_reflected")
HIGHS_SOLVERS = ("choose", "simplex", "ipm")
IPM_METHODS = ("mehrotra", "homogeneous")
IPM_LINEAR_ACCURACIES = ("strict", "backward", "inexact", "blockwise", "blockwise_weighted", "blockwise_forcing", "blockwise_forcing_weighted", "adaptive_regularization", "primal_dual_regularization", "curvature_regularization", "full_newton", "bounded_newton", "homogeneous_scaled", "homogeneous_repair")
IPM_LINEAR_SOLVERS = ("qdldl", "sparselu", "normal", "qdldl_extended", "factorhighs", "normal_augmented", "normal_multifrontal", "normal_extended", "normal_refined", "normal_cholmod", "normal_cholmod_aat")


def finite_positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def nonnegative_integer(value):
    number = int(value)
    if number < 0 or number > 2147483647:
        raise argparse.ArgumentTypeError("必须是 0 到 2147483647 之间的整数")
    return number


def positive_integer(value):
    number = nonnegative_integer(value)
    if number == 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def nonnegative_int64(value):
    number = int(value)
    if number < 0 or number > 9223372036854775807:
        raise argparse.ArgumentTypeError("必须是非负 64 位整数")
    return number


def scaling_passes(value):
    number = nonnegative_integer(value)
    if number > 20:
        raise argparse.ArgumentTypeError("缩放轮数必须在 0 到 20 之间")
    return number


def history_size(value):
    number = nonnegative_integer(value)
    if number > 32:
        raise argparse.ArgumentTypeError("历史数量必须在 0 到 32 之间")
    return number


def nonnegative_finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须是有限非负数")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalid_constant(value):
    raise ValueError("JSON 中含有非有限常量：" + value)


def finite_json_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON 中含有超出有限范围的数值：" + value)
    return number


def strict_json(value):
    return json.loads(value, parse_constant=invalid_constant, parse_float=finite_json_float)


def finite_real(value):
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def valid_homogeneous_work(report):
    counts = ("ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves")
    scalars = ("ipm_homogeneous_tau", "ipm_homogeneous_kappa")
    if not all(type(report.get(field)) is int and 0 <= report[field] <= 9223372036854775807
               for field in counts) or not all(field in report for field in scalars):
        return False
    if all(report[field] is None for field in scalars):
        return all(report[field] == 0 for field in counts)
    return all(finite_real(report[field]) and report[field] > 0 for field in scalars)


def valid_homogeneous_scaled_work(report):
    fields = ("ipm_homogeneous_scaled_products", "ipm_homogeneous_scaled_refinement_steps",
              "ipm_homogeneous_scaled_inexact_returns", "ipm_homogeneous_direction_solves")
    if not all(type(report.get(field)) is int and 0 <= report[field] <= 9223372036854775807 for field in fields):
        return False
    products, refinements, inexact_returns, directions = (report[field] for field in fields)
    return refinements <= directions and inexact_returns <= directions and products >= inexact_returns


def valid_homogeneous_repair_work(report):
    fields = ("ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes",
              "ipm_homogeneous_constant_solves")
    if not all(type(report.get(field)) is int and 0 <= report[field] <= 9223372036854775807 for field in fields):
        return False
    repairs, successes, constants = (report[field] for field in fields)
    return successes <= repairs <= constants


def validate_ipm_method(parser, args, prefix=""):
    method = getattr(args, prefix + "ipm_method") or "mehrotra"
    accuracy = getattr(args, prefix + "ipm_linear_accuracy") or "strict"
    if accuracy in ("homogeneous_scaled", "homogeneous_repair") and method != "homogeneous":
        flag_prefix = "--" + prefix.replace("_", "-")
        parser.error(flag_prefix + "ipm-linear-accuracy " + accuracy + " 要求 " + flag_prefix + "ipm-method homogeneous")
    if method != "homogeneous":
        return
    budgets = ("ipm_central_corrections", "ipm_direction_retries", "ipm_krylov_history_size")
    if accuracy not in ("strict", "homogeneous_scaled", "homogeneous_repair") or any(getattr(args, prefix + field) for field in budgets) or (
        getattr(args, prefix + "ipm_inexact_prefilter") == "on"
    ):
        flag = "--" + prefix.replace("_", "-") + "ipm-method homogeneous"
        parser.error(flag + " 仅支持 strict、homogeneous_scaled 或 homogeneous_repair 精度、零中央修正/方向重试/历史预算及关闭非精确预筛选")


def validate_ipm_linear_solver(parser, args, prefix=""):
    linear_solver = getattr(args, prefix + "ipm_linear_solver")
    if linear_solver not in ("normal_cholmod", "normal_cholmod_aat"):
        return
    flag = "--" + prefix.replace("_", "-")
    if getattr(args, prefix + "ipm_singleton_elimination") != "on":
        parser.error(flag + "ipm-linear-solver " + linear_solver + " 要求 " + flag + "ipm-singleton-elimination on")
    if getattr(args, prefix + "ipm_dynamic_regularization") == "on":
        parser.error(flag + "ipm-linear-solver " + linear_solver + " 不支持 " + flag + "ipm-dynamic-regularization on")


def valid_lp_optimality(report):
    optimality = report.get("lp_optimality")
    dual = report.get("row_dual")
    model = report.get("model")
    if not isinstance(optimality, dict) or not all(
        optimality.get(field) is True
        for field in ("valid_input", "dual_feasible", "gap_satisfied", "numerically_optimal")
    ):
        return False
    if not all(finite_real(optimality.get(field)) for field in (
        "max_dual_residual", "dual_objective_estimate", "objective_gap", "relative_objective_gap"
    )):
        return False
    return isinstance(model, dict) and isinstance(dual, list) and (
        type(model.get("rows")) is int and model["rows"] >= 0
        and len(dual) == model["rows"] and all(finite_real(value) for value in dual)
    )
def classify(returncode, report):
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        return "invalid_report"
    status = report.get("status")
    if not isinstance(status, str):
        return "invalid_report"
    if status == "verification_failed":
        return "verification_failed"
    if returncode == 0 and status in RESOLVED:
        if not isinstance(report.get("backend"), str) or report["backend"] not in BACKENDS:
            return "invalid_report"
        if status in {"optimal", "empty"}:
            verification = report.get("verification")
            model = report.get("model")
            solution = report.get("solution")
            if not isinstance(verification, dict) or not (
                report.get("has_solution") is True
                and verification.get("valid_input") is True
                and verification.get("feasible") is True
                and verification.get("objective_consistent") is True
                and finite_real(report.get("objective"))
                and finite_real(verification.get("recomputed_objective"))
            ):
                return "invalid_report"
            if not isinstance(model, dict) or not isinstance(solution, list) or not (
                type(model.get("columns")) is int
                and model["columns"] >= 0
                and len(solution) == model["columns"]
                and all(finite_real(value) for value in solution)
            ):
                return "invalid_report"
            if report["backend"] in {"pdlp", "ipm"}:
                if not valid_lp_optimality(report):
                    return "invalid_report"
        return "resolved"
    if status == "error" and returncode == 2:
        return "solver_error"
    if returncode == 4 and status not in RESOLVED and status != "error":
        return "incomplete"
    return "exit_status_mismatch"


def output_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_one(args, model, seed, run_index):
    command = [str(args.solver), "solve", str(model), "--backend", args.backend,
               "--time-limit", str(args.time_limit),
               "--threads", str(args.threads), "--seed", str(seed),
               "--mip-gap", str(args.mip_gap)]
    if args.iteration_limit is not None:
        command += ["--iteration-limit", str(args.iteration_limit)]
    if args.scaling_passes is not None:
        command += ["--scaling-passes", str(args.scaling_passes)]
    if getattr(args, "pdlp_mode", None) is not None:
        command += ["--pdlp-mode", args.pdlp_mode]
    if getattr(args, "pdlp_history_size", None) is not None:
        command += ["--pdlp-history-size", str(args.pdlp_history_size)]
    for attribute, flag in (("ipm_method", "--ipm-method"),
                            ("ipm_singleton_elimination", "--ipm-singleton-elimination"),
                            ("ipm_dynamic_regularization", "--ipm-dynamic-regularization"),
                            ("ipm_kkt_equilibration", "--ipm-kkt-equilibration"),
                            ("ipm_objective_scaling", "--ipm-objective-scaling"),
                            ("ipm_linear_accuracy", "--ipm-linear-accuracy"),
                            ("ipm_central_corrections", "--ipm-central-corrections"),
                            ("ipm_direction_retries", "--ipm-direction-retries"),
                            ("ipm_krylov_history_size", "--ipm-krylov-history-size"),
                            ("ipm_linear_solver", "--ipm-linear-solver"),
                            ("ipm_inexact_prefilter", "--ipm-inexact-prefilter"),
                            ("ipm_fixed_variable_elimination", "--ipm-fixed-variable-elimination"),
                            ("ipm_singleton_presolve", "--ipm-singleton-presolve"),
                            ("ipm_presolve_fixed_elimination", "--ipm-presolve-fixed-elimination")):
        value = getattr(args, attribute, None)
        if value is not None:
            command += [flag, str(value)]
    for attribute, flag in (("highs_solver", "--highs-solver"), ("highs_crossover", "--highs-crossover"),
                            ("highs_presolve", "--highs-presolve")):
        value = getattr(args, attribute, None)
        if value is not None:
            command += [flag, value]
    record = {"schema_version": 1, "run_index": run_index, "input": str(model),
              "seed": seed, "requested_backend": args.backend, "command": command, "returncode": None,
              "classification": "process_error", "status": None,
              "report": None, "stdout": "", "stderr": "", "error": None,
              "model_sha256": None, "hash_error": None}
    try:
        record["model_sha256"] = sha256(model)
    except OSError as error:
        record["hash_error"] = str(error)
    started = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, timeout=args.wall_timeout, check=False)
        finished = time.perf_counter()
        record.update(returncode=process.returncode, stdout=output_text(process.stdout),
                      stderr=output_text(process.stderr))
        try:
            report = strict_json(process.stdout.decode("utf-8", errors="strict"))
            record["report"] = report
            if isinstance(report, dict) and isinstance(report.get("status"), str):
                record["status"] = report.get("status")
            record["classification"] = classify(process.returncode, report)
            if record["classification"] == "resolved" and report.get("backend") != args.backend:
                record["classification"] = "backend_mismatch"
            if record["classification"] == "resolved" and args.backend == "highs":
                options = report.get("options")
                reported_highs = options.get("highs") if isinstance(options, dict) else None
                for attribute, field in (("highs_solver", "solver"), ("highs_crossover", "run_crossover"),
                                         ("highs_presolve", "presolve")):
                    expected = getattr(args, attribute, None)
                    if expected is not None and (not isinstance(reported_highs, dict) or reported_highs.get(field) != expected):
                        record["classification"] = attribute + "_mismatch"
                        break
            if record["classification"] == "resolved" and args.backend == "pdlp" and getattr(args, "pdlp_mode", None) is not None:
                options = report.get("options")
                reported_pdlp = options.get("pdlp") if isinstance(options, dict) else None
                if not isinstance(reported_pdlp, dict) or reported_pdlp.get("mode") != args.pdlp_mode:
                    record["classification"] = "pdlp_mode_mismatch"
            if record["classification"] == "resolved" and args.backend == "pdlp" and getattr(args, "pdlp_history_size", None) is not None:
                options = report.get("options")
                reported_pdlp = options.get("pdlp") if isinstance(options, dict) else None
                if not isinstance(reported_pdlp, dict) or type(reported_pdlp.get("history_size")) is not int or reported_pdlp["history_size"] != args.pdlp_history_size:
                    record["classification"] = "pdlp_history_size_mismatch"
            if record["classification"] == "resolved" and args.backend in {"pdlp", "ipm"}:
                options = report.get("options")
                reported_options = options.get(args.backend) if isinstance(options, dict) else None
                for argument, field in [("iteration_limit", "max_iterations"), ("scaling_passes", "scaling_passes")]:
                    expected = getattr(args, argument, None)
                    if expected is not None and (not isinstance(reported_options, dict) or
                                                 type(reported_options.get(field)) is not int or
                                                 reported_options[field] != expected):
                        record["classification"] = argument + "_mismatch"
                        break
            if record["classification"] == "resolved" and args.backend == "ipm":
                options = report.get("options")
                reported_ipm = options.get("ipm") if isinstance(options, dict) else None
                for attribute, field in (("ipm_singleton_elimination", "eliminate_singletons"),
                                         ("ipm_dynamic_regularization", "dynamic_regularization"),
                                         ("ipm_kkt_equilibration", "equilibrate_kkt"),
                                         ("ipm_objective_scaling", "objective_scaling"),
                                         ("ipm_inexact_prefilter", "inexact_prefilter"),
                                         ("ipm_fixed_variable_elimination", "eliminate_fixed_variables"),
                                         ("ipm_singleton_presolve", "presolve_singleton_rows"),
                                         ("ipm_presolve_fixed_elimination", "eliminate_presolved_fixed_variables")):
                    requested = getattr(args, attribute, None)
                    if requested is not None and (not isinstance(reported_ipm, dict) or
                                                   reported_ipm.get(field) is not (requested == "on")):
                        record["classification"] = attribute + "_mismatch"
                        break
                for attribute, field in (("ipm_method", "method"),
                                         ("ipm_linear_accuracy", "linear_accuracy"),
                                         ("ipm_linear_solver", "linear_solver")):
                    expected = getattr(args, attribute, None)
                    if record["classification"] == "resolved" and expected is not None and (
                        not isinstance(reported_ipm, dict) or type(reported_ipm.get(field)) is not str
                        or reported_ipm[field] != expected
                    ):
                        record["classification"] = attribute + "_mismatch"
                if record["classification"] == "resolved" and getattr(args, "ipm_method", None) == "homogeneous":
                    if not valid_homogeneous_work(report):
                        record["classification"] = "ipm_homogeneous_work_mismatch"
                if record["classification"] == "resolved" and getattr(args, "ipm_linear_accuracy", None) == "homogeneous_scaled":
                    if not valid_homogeneous_scaled_work(report):
                        record["classification"] = "ipm_homogeneous_scaled_work_mismatch"
                if record["classification"] == "resolved" and getattr(args, "ipm_linear_accuracy", None) == "homogeneous_repair":
                    if not valid_homogeneous_repair_work(report):
                        record["classification"] = "ipm_homogeneous_repair_work_mismatch"
                for attribute, field in (("ipm_central_corrections", "central_corrections"),
                                         ("ipm_direction_retries", "direction_retries"),
                                         ("ipm_krylov_history_size", "krylov_history_size")):
                    expected = getattr(args, attribute, None)
                    if record["classification"] == "resolved" and expected is not None and (
                        not isinstance(reported_ipm, dict) or type(reported_ipm.get(field)) is not int
                        or reported_ipm[field] != expected
                    ):
                        record["classification"] = attribute + "_mismatch"
                if record["classification"] == "resolved" and getattr(args, "ipm_linear_accuracy", None) == "bounded_newton":
                    fields = ("ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns",
                              "ipm_full_newton_preconditioner_solves", "ipm_full_newton_products")
                    if not all(type(report.get(field)) is int and report[field] >= 0 for field in fields) or (
                        report[fields[1]] > report[fields[0]]
                    ):
                        record["classification"] = "ipm_bounded_newton_work_mismatch"
        except (ValueError, TypeError) as error:
            record["classification"] = "invalid_json"
            record["error"] = str(error)
    except subprocess.TimeoutExpired as error:
        finished = time.perf_counter()
        record["classification"] = "wall_timeout"
        record["stdout"] = output_text(error.stdout)
        record["stderr"] = output_text(error.stderr)
        record["error"] = "进程超过固定墙钟预算，已终止"
    except OSError as error:
        finished = time.perf_counter()
        record["error"] = str(error)
    # 进程墙钟截至子进程结束和输出收集完成；驱动端 JSON 解析不计入求解性能。
    record["elapsed_wall_seconds"] = finished - started
    # 原始耗时始终保留；汇总指标将失败按统一上限计入，避免只统计成功样本。
    record["penalized_capped_seconds"] = (
        min(record["elapsed_wall_seconds"], args.time_limit)
        if record["classification"] == "resolved" else args.time_limit
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="待求解的 MPS/LP 文件")
    parser.add_argument("--solver", required=True, type=Path, help="求解器可执行文件路径")
    parser.add_argument("--backend", choices=BACKENDS, default="highs")
    parser.add_argument("--time-limit", type=finite_positive, default=60.0, help="每次求解的时间预算，单位秒")
    parser.add_argument("--wall-timeout", type=finite_positive, help="进程墙钟上限，默认求解预算加 30 秒")
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--seeds", type=nonnegative_integer, nargs="+", default=[0])
    parser.add_argument("--mip-gap", type=nonnegative_finite, default=1e-4)
    parser.add_argument("--iteration-limit", type=nonnegative_int64, help="用于 PDLP/IPM；省略时采用对应后端默认值")
    parser.add_argument("--scaling-passes", type=scaling_passes, help="用于 PDLP/IPM；省略时采用对应后端默认值")
    parser.add_argument("--pdlp-mode", choices=PDLP_MODES, help="仅用于 PDLP；省略时采用求解器默认模式")
    parser.add_argument("--pdlp-history-size", type=history_size, help="仅用于 PDLP；历史数量 0 到 32，省略时采用求解器默认值")
    parser.add_argument("--ipm-method", choices=IPM_METHODS,
                        help="仅用于 IPM；mehrotra 或齐次方法 homogeneous，省略时 mehrotra")
    parser.add_argument("--ipm-singleton-elimination", choices=("on", "off"),
                        help="仅用于 IPM；是否消去单非零不等式的 KKT 变量，默认关闭")
    parser.add_argument("--ipm-dynamic-regularization", choices=("on", "off"),
                        help="仅用于 IPM；是否启用动态主元正则化，默认关闭")
    parser.add_argument("--ipm-kkt-equilibration", choices=("on", "off"),
                        help="仅用于 IPM；是否均衡化 KKT 系统，默认关闭")
    parser.add_argument("--ipm-objective-scaling", choices=("on", "off"),
                        help="仅用于 IPM；是否独立归一化目标系数，默认开启，与行列缩放轮数独立")
    parser.add_argument("--ipm-linear-accuracy", choices=IPM_LINEAR_ACCURACIES,
                        help="仅用于 IPM；线性系统精度规则，可选 strict、backward、inexact、blockwise、blockwise_weighted、blockwise_forcing、blockwise_forcing_weighted、adaptive_regularization、primal_dual_regularization、curvature_regularization、full_newton、bounded_newton、homogeneous_scaled、homogeneous_repair，默认 strict；homogeneous_scaled 仅用于 homogeneous，有限精化后保留最佳有限方向，仍须原模型终验；homogeneous_repair 仅用于 homogeneous，在有限非正标量分母时每轮至多按块严格重算一次常数方向")
    parser.add_argument("--ipm-central-corrections", type=nonnegative_integer, choices=(0, 1),
                        help="仅用于 IPM；每步中央修正次数，默认 0")
    parser.add_argument("--ipm-direction-retries", type=nonnegative_integer, choices=range(9),
                        help="仅用于 IPM；方向失败后的正则重试预算，0 到 8，默认 0；正预算支持 strict、primal_dual_regularization、curvature_regularization、blockwise、blockwise_weighted、full_newton")
    parser.add_argument("--ipm-krylov-history-size", type=nonnegative_integer, choices=range(9),
                        help="仅用于 IPM；同一 KKT 的 Krylov 历史维数，0 到 8，默认 0；正值仅支持 strict、blockwise")
    parser.add_argument("--ipm-linear-solver", choices=IPM_LINEAR_SOLVERS,
                        help="仅用于 IPM；稀疏线性系统分解，默认 qdldl；normal_cholmod 保持原 Gram 和 AMD，使用 CHOLMOD 超节点分解；normal_cholmod_aat 保持相同 AMD 和结构预算，使用 CHOLMOD 矩形输入数值乘积")
    parser.add_argument("--ipm-inexact-prefilter", choices=("on", "off"),
                        help="仅用于 IPM；复用残差的非精确方向预筛选，默认关闭")
    parser.add_argument("--ipm-fixed-variable-elimination", choices=("on", "off"),
                        help="仅用于 IPM；代入上下界完全相等的有限固定变量，默认关闭")
    parser.add_argument("--ipm-singleton-presolve", choices=("on", "off"),
                        help="仅用于 IPM；将单非零输入行归并到变量界，默认关闭")
    parser.add_argument("--ipm-presolve-fixed-elimination", choices=("on", "off"),
                        help="仅用于 IPM；消除归并后新固定的变量，默认关闭；开启要求 --ipm-singleton-presolve on")
    parser.add_argument("--highs-solver", choices=HIGHS_SOLVERS, help="仅用于 HiGHS 连续 LP 的固定算法对照")
    parser.add_argument("--highs-crossover", choices=("on", "off"), help="HiGHS 的最优基恢复；点解对照可显式关闭")
    parser.add_argument("--highs-presolve", choices=("choose", "on", "off"))
    parser.add_argument("--output-dir", type=Path, help="新建结果目录；已存在时拒绝覆盖")
    args = parser.parse_args()
    if args.backend == "highs" and (args.iteration_limit is not None or args.scaling_passes is not None):
        parser.error("--iteration-limit 和 --scaling-passes 仅适用于 pdlp 或 ipm 后端")
    if args.backend != "pdlp" and (args.pdlp_mode is not None or args.pdlp_history_size is not None):
        parser.error("--pdlp-mode 和 --pdlp-history-size 仅适用于 pdlp 后端")
    if args.backend != "ipm" and args.ipm_method is not None:
        parser.error("--ipm-method 仅适用于 ipm 后端")
    validate_ipm_method(parser, args)
    if args.backend != "ipm" and args.ipm_singleton_elimination is not None:
        parser.error("--ipm-singleton-elimination 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_dynamic_regularization is not None:
        parser.error("--ipm-dynamic-regularization 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_kkt_equilibration is not None:
        parser.error("--ipm-kkt-equilibration 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_objective_scaling is not None:
        parser.error("--ipm-objective-scaling 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_linear_accuracy is not None:
        parser.error("--ipm-linear-accuracy 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_central_corrections is not None:
        parser.error("--ipm-central-corrections 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_direction_retries is not None:
        parser.error("--ipm-direction-retries 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_krylov_history_size is not None:
        parser.error("--ipm-krylov-history-size 仅适用于 ipm 后端")
    if args.ipm_linear_accuracy in ("full_newton", "bounded_newton") and args.ipm_central_corrections:
        parser.error("full_newton 和 bounded_newton 暂不支持正 --ipm-central-corrections")
    if args.ipm_linear_accuracy == "bounded_newton" and args.ipm_direction_retries:
        parser.error("bounded_newton 暂不支持正 --ipm-direction-retries")
    if args.ipm_krylov_history_size and (args.ipm_linear_accuracy or "strict") not in ("strict", "blockwise"):
        parser.error("正 --ipm-krylov-history-size 仅支持 strict 或 blockwise 精度")
    if args.backend != "ipm" and args.ipm_linear_solver is not None:
        parser.error("--ipm-linear-solver 仅适用于 ipm 后端")
    validate_ipm_linear_solver(parser, args)
    if args.backend != "ipm" and args.ipm_fixed_variable_elimination is not None:
        parser.error("--ipm-fixed-variable-elimination 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_singleton_presolve is not None:
        parser.error("--ipm-singleton-presolve 仅适用于 ipm 后端")
    if args.backend != "ipm" and args.ipm_presolve_fixed_elimination is not None:
        parser.error("--ipm-presolve-fixed-elimination 仅适用于 ipm 后端")
    if args.ipm_presolve_fixed_elimination == "on" and args.ipm_singleton_presolve != "on":
        parser.error("--ipm-presolve-fixed-elimination on 要求 --ipm-singleton-presolve on")
    if args.backend != "ipm" and args.ipm_inexact_prefilter is not None:
        parser.error("--ipm-inexact-prefilter 仅适用于 ipm 后端")
    if args.backend != "highs" and any(value is not None for value in (args.highs_solver, args.highs_crossover, args.highs_presolve)):
        parser.error("HiGHS 算法、crossover 与 presolve 参数仅适用于 highs 后端")
    if args.backend != "highs" and args.threads != 1:
        parser.error("当前 pdlp 和 ipm 后端仅支持 --threads 1")
    args.solver = args.solver.resolve()
    args.models = [model.resolve() for model in args.models]
    args.wall_timeout = args.wall_timeout if args.wall_timeout is not None else args.time_limit + 30.0
    if args.wall_timeout < args.time_limit:
        parser.error("--wall-timeout 不能小于 --time-limit")
    if args.output_dir is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = Path("results") / (stamp + "-" + uuid.uuid4().hex[:8])
    args.output_dir = args.output_dir.resolve()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error("不能新建结果目录：" + str(error))
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "solver": str(args.solver), "solver_sha256": None,
        "backend": args.backend,
        "models": [str(model) for model in args.models], "seeds": args.seeds,
        "time_limit": args.time_limit, "wall_timeout": args.wall_timeout,
        "threads": args.threads, "mip_relative_gap": args.mip_gap,
        "pdlp_requested_options": {"max_iterations": args.iteration_limit, "scaling_passes": args.scaling_passes,
                                   "mode": args.pdlp_mode, "history_size": args.pdlp_history_size}
                                  if args.backend == "pdlp" else None,
        "ipm_requested_options": {"method": args.ipm_method,
                                  "max_iterations": args.iteration_limit, "scaling_passes": args.scaling_passes,
                                  "eliminate_singletons": None if args.ipm_singleton_elimination is None
                                  else args.ipm_singleton_elimination == "on",
                                  "dynamic_regularization": None if args.ipm_dynamic_regularization is None
                                  else args.ipm_dynamic_regularization == "on",
                                  "equilibrate_kkt": None if args.ipm_kkt_equilibration is None
                                  else args.ipm_kkt_equilibration == "on",
                                  "objective_scaling": None if args.ipm_objective_scaling is None
                                  else args.ipm_objective_scaling == "on",
                                  "linear_accuracy": args.ipm_linear_accuracy,
                                  "central_corrections": args.ipm_central_corrections,
                                  "direction_retries": args.ipm_direction_retries,
                                  "krylov_history_size": args.ipm_krylov_history_size,
                                  "linear_solver": args.ipm_linear_solver,
                                  "eliminate_fixed_variables": None if args.ipm_fixed_variable_elimination is None
                                  else args.ipm_fixed_variable_elimination == "on",
                                  "presolve_singleton_rows": None if args.ipm_singleton_presolve is None
                                  else args.ipm_singleton_presolve == "on",
                                  "eliminate_presolved_fixed_variables": None if args.ipm_presolve_fixed_elimination is None
                                  else args.ipm_presolve_fixed_elimination == "on",
                                  "inexact_prefilter": None if args.ipm_inexact_prefilter is None
                                  else args.ipm_inexact_prefilter == "on"}
                                 if args.backend == "ipm" else None,
        "highs_requested_options": {"solver": args.highs_solver, "run_crossover": args.highs_crossover,
                                    "presolve": args.highs_presolve}
                                   if args.backend == "highs" else None,
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(), "python_version": platform.python_version(),
        "metric_definition": "SGM10 对所有模型和种子计入：已结束样本使用 min(端到端墙钟耗时, time_limit)，失败和未完成样本使用 time_limit；后端最优、不可行、无界、空模型分别计数。",
        "verification_scope": "HiGHS 结论来自后端并独立复核候选；自主 PDLP/IPM 最优还要求原模型上的对偶残差与间隙检查通过，属于浮点数值最优。估计对偶目标不作为有效界使用。",
    }
    try:
        manifest["solver_sha256"] = sha256(args.solver)
    except OSError as error:
        manifest["solver_hash_error"] = str(error)
    try:
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        records = []
        with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as output:
            for model in args.models:
                for seed in args.seeds:
                    record = run_one(args, model, seed, len(records))
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    records.append(record)
        classifications = dict(collections.Counter(record["classification"] for record in records))
        statuses = dict(collections.Counter(record["status"] or "no_status" for record in records))
        summary = {
            "schema_version": 1, "run_count": len(records),
            "resolved_count": classifications.get("resolved", 0),
            "unresolved_count": len(records) - classifications.get("resolved", 0),
            "optimal_count": sum(record["classification"] == "resolved" and record["status"] == "optimal" for record in records),
            "infeasible_count": sum(record["classification"] == "resolved" and record["status"] == "infeasible" for record in records),
            "unbounded_count": sum(record["classification"] == "resolved" and record["status"] == "unbounded" for record in records),
            "empty_count": sum(record["classification"] == "resolved" and record["status"] == "empty" for record in records),
            "classification_counts": classifications, "status_counts": statuses,
            "penalized_capped_wall_sgm10_seconds": math.expm1(
                sum(math.log1p(record["penalized_capped_seconds"] / 10.0) for record in records) / len(records)
            ) * 10.0,
            "failure_penalty_seconds": args.time_limit,
            "output_dir": str(args.output_dir),
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
        return 0 if summary["unresolved_count"] == 0 else 1
    except OSError as error:
        print("写入基准报告失败：" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    # 管道中的结构化结果固定使用 UTF-8，不依赖主机区域设置。
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
