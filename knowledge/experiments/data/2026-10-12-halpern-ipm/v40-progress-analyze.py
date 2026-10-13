"""只读归档中的 v40 事件，固定位置取样并统计完整区间，不调用求解器。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


ANCHORS = (1, 10, 25, 40, 60, 100, 150)
INTERVALS = ((1, 10), (11, 24), (25, 39), (40, 59), (60, 99), (100, 149), (150, 200))
FIELDS = {
    "begin": (
        "rp_ld_inf", "rd_ld_inf", "rp_algorithm_inf", "rd_algorithm_inf",
        "rp_algorithm_ld_difference", "rd_algorithm_ld_difference", "mu_algorithm",
        "x_inf", "s_inf", "z_inf", "working_gap_raw", "working_gap_dual_term",
        "working_gap_primal_term", "working_gap_complementarity", "working_gap_identity_error",
    ),
    "before_step": (
        "ep_algorithm_inf", "ed_algorithm_inf", "ep_ld_inf", "ed_ld_inf",
        "h_complementarity_defect_inf", "h_inf", "diagonal_h_inf", "dx_inf", "ds_inf", "dz_inf",
        "alpha_primal", "alpha_dual", "sigma", "mu_algorithm", "regularization_level",
        "primal_regularization", "dual_regularization", "kkt_rhs_upper_inf", "kkt_rhs_lower_inf",
        "kkt_upper_ld_inf", "kkt_lower_ld_inf", "kkt_global_tolerance",
        "kkt_strict_upper_tolerance", "kkt_strict_lower_tolerance", "strict_block_tolerances_active",
        "h_matches_direction", "predicted_rp_ld_inf", "predicted_rd_ld_inf",
        "factorizations", "refinement_steps", "direction_retries",
    ),
    "after_step": (
        "rp_ld_inf", "rd_ld_inf", "rp_update_prediction_drift", "rd_update_prediction_drift",
        "x_inf", "s_inf", "z_inf", "mu_ld", "working_gap_raw", "working_gap_dual_term",
        "working_gap_primal_term", "working_gap_complementarity", "working_gap_identity_error",
        "objective_scale",
    ),
    "current_assess": (
        "original_primal_feasible", "original_dual_feasible", "original_gap_satisfied",
        "original_numerically_optimal", "original_max_row_violation", "original_max_bound_violation",
        "original_max_dual_residual", "original_objective", "original_dual_estimate",
        "original_corrected_dual_bound", "original_gap", "original_relative_gap",
    ),
}


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    # 零分母或不可表示的比值保持缺失，不添加人为地板。
    if numerator is None or denominator is None or denominator == 0:
        return None
    answer = numerator / denominator
    return answer if math.isfinite(answer) else None


def project(event: dict | None, fields: tuple[str, ...]) -> dict | None:
    if event is None:
        return None
    return {key: event.get(key) for key in ("event_id", "iteration", "completed_iterations", "phase") + fields}


def metrics(events: dict) -> dict:
    begin, step, after = (events.get(name, {}) for name in ("begin", "before_step", "after_step"))
    answer = {key: step.get(key) for key in (
        "alpha_primal", "alpha_dual", "ep_algorithm_inf", "ed_algorithm_inf", "ep_ld_inf", "ed_ld_inf",
        "kkt_global_tolerance", "kkt_upper_ld_inf", "kkt_lower_ld_inf", "h_complementarity_defect_inf")}
    for prefix, event, fields in (
        ("begin", begin, ("rp_ld_inf", "rd_ld_inf", "mu_algorithm", "rp_algorithm_ld_difference", "rd_algorithm_ld_difference")),
        ("after", after, ("rp_ld_inf", "rd_ld_inf", "mu_ld", "rp_update_prediction_drift", "rd_update_prediction_drift",
                          "working_gap_raw", "working_gap_dual_term", "working_gap_primal_term", "working_gap_complementarity")),
    ):
        answer.update({prefix + "_" + key: event.get(key) for key in fields})
    answer.update({
        "ep_algorithm_over_rp_algorithm": ratio(step.get("ep_algorithm_inf"), begin.get("rp_algorithm_inf")),
        "ed_algorithm_over_rd_algorithm": ratio(step.get("ed_algorithm_inf"), begin.get("rd_algorithm_inf")),
        "ep_ld_over_rp_ld": ratio(step.get("ep_ld_inf"), begin.get("rp_ld_inf")),
        "ed_ld_over_rd_ld": ratio(step.get("ed_ld_inf"), begin.get("rd_ld_inf")),
        "kkt_global_tolerance_over_rd_algorithm": ratio(step.get("kkt_global_tolerance"), begin.get("rd_algorithm_inf")),
        "kkt_upper_over_global_tolerance": ratio(step.get("kkt_upper_ld_inf"), step.get("kkt_global_tolerance")),
        "kkt_upper_over_strict_block_tolerance": ratio(step.get("kkt_upper_ld_inf"), step.get("kkt_strict_upper_tolerance")),
        "h_defect_over_h": ratio(step.get("h_complementarity_defect_inf"), step.get("h_inf")),
        "actual_rp_contraction": ratio(after.get("rp_ld_inf"), begin.get("rp_ld_inf")),
        "actual_rd_contraction": ratio(after.get("rd_ld_inf"), begin.get("rd_ld_inf")),
        "rp_algorithm_difference_over_rp_ld": ratio(begin.get("rp_algorithm_ld_difference"), begin.get("rp_ld_inf")),
        "rd_algorithm_difference_over_rd_ld": ratio(begin.get("rd_algorithm_ld_difference"), begin.get("rd_ld_inf")),
        "rp_update_drift_over_actual_rp": ratio(after.get("rp_update_prediction_drift"), after.get("rp_ld_inf")),
        "rd_update_drift_over_actual_rd": ratio(after.get("rd_update_prediction_drift"), after.get("rd_ld_inf")),
    })
    for field in ("factorizations", "refinement_steps", "direction_retries"):
        left, right = begin.get(field), step.get(field)
        answer["step_" + field] = right - left if left is not None and right is not None else None
    return answer


def summarize_metrics(values: list[dict]) -> dict:
    if not values:
        return {}
    answer = {}
    for key in values[0]:
        samples = [value[key] for value in values if value.get(key) is not None]
        if samples:
            answer[key] = {"count": len(samples), "minimum": min(samples),
                           "median": statistics.median(samples), "maximum": max(samples)}
    return answer


def analyze_record(row: dict) -> dict:
    report = row["report"]
    by_iteration = {}
    for event in row["events"]:
        if event["event"] in FIELDS:
            by_iteration.setdefault(event["iteration"], {})[event["event"]] = event
    complete = {i: events for i, events in by_iteration.items()
                if i > 0 and all(key in events for key in ("begin", "before_step", "after_step"))}
    iterations = sorted(complete)
    selected = set(ANCHORS)
    if iterations:
        selected.add(iterations[-1])
    anchors = []
    for iteration in sorted(selected & complete.keys()):
        events = complete[iteration]
        anchors.append({"iteration": iteration, "last_completed": iteration == iterations[-1],
                        **{kind: project(events.get(kind), fields) for kind, fields in FIELDS.items()},
                        "derived": metrics(events)})
    intervals = []
    for lower, upper in INTERVALS:
        samples = [metrics(complete[i]) for i in iterations if lower <= i <= upper]
        if samples:
            intervals.append({"requested_interval": [lower, upper], "complete_steps": len(samples),
                              "metrics": summarize_metrics(samples)})
    last_assess = next((event for event in reversed(row["events"]) if event["event"] == "current_assess"), None)
    return {
        "model": row["model"], "configuration": row["configuration"], "development": row["development"],
        "status": report["status"], "message": report["message"], "iterations": report["ipm_iterations"],
        "factorizations": report["ipm_factorizations"], "refinement_steps": report["ipm_refinement_steps"],
        "direction_retries": report["ipm_direction_retries"],
        "retained_best_report": {"has_solution": report["has_solution"], "objective": report["objective"],
                                 "verification": report["verification"], "lp_optimality": report["lp_optimality"]},
        "last_current_assess": project(last_assess, FIELDS["current_assess"]),
        "event_count": len(row["events"]), "complete_step_count": len(complete),
        "incomplete_attempt_iterations": [i for i, events in by_iteration.items() if i > 0 and "begin" in events and i not in complete],
        "nonfinite_events": [{"event_id": event.get("event_id"), "iteration": event.get("iteration"), "event": event["event"]}
                             for event in row["events"] if event.get("finite_fields") is False],
        "anchors": anchors, "intervals": intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path(__file__).with_name("v40-progress-runs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("v40-progress-analysis.json"))
    args = parser.parse_args()
    path = args.runs
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if path.suffix == ".gz" else raw.decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    enabled = [row for row in rows if row["variant"] == "diagnostic_on"]
    if any(row["diagnostic_parse_errors"] for row in rows):
        raise ValueError("归档有诊断解析错误，拒绝生成不完整分析")
    if any(row["report"] is None for row in enabled):
        raise ValueError("启用诊断的记录缺少报告")
    result = {
        "schema": 1, "diagnostic_only": True,
        "runs_file": path.name, "runs_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "record_count": len(rows), "enabled_record_count": len(enabled),
        "event_count": sum(len(row["events"]) for row in rows),
        "fixed_anchors": list(ANCHORS), "additional_anchor": "每条记录的最后一个完整迭代",
        "fixed_intervals": list(INTERVALS),
        "limitations": [
            "只反映已发生的方向、步长与残差；未改变策略，不能单凭相关性证明替代算法的因果收益。",
            "诊断增加时间和内存；超时轨迹不能用于性能比较。",
            "working_gap 是工作锥的有符号恒等式，original_gap 是本次原模型验算的绝对间隙，retained_best_report 可能保留旧点。",
            "在读取本轮轨迹后的事后分析中，对全部 39 个配置统一使用这些位置与区间；不是预注册的独立统计检验。",
            "比值只在非零分母且结果有限时记录；范数比不包含误差向量的方向信息。",
            "每区间只统计 begin/before_step/after_step 齐全的迭代；失败前没有完整方向的尝试另外列出。",
        ],
        "records": [analyze_record(row) for row in enabled],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(enabled), "events": result["event_count"],
                      "anchors": sum(len(row["anchors"]) for row in result["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
