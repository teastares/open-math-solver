"""按测量前固定的方向选择规则分析 v42 只读事件，不调用求解器。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


ANALYSIS_RULES = {
    "schema": 1,
    "fixed_before_measurement": True,
    "grouping": "每次运行按 direction_id 分组，原始事件逐条保留；不同运行不合并方向编号。",
    "selections": [
        "最后一个 direction_failure，或运行有超时证据且诊断未禁用的缺终点方向；其他缺失结束事件单列，不推断求解失败。",
        "在该方向之前最近一个明确返回的方向；不是任意历史上残差最小方向。",
        "首次实际执行 Arnoldi 的方向；候选残差大小不影响选择。",
    ],
    "overlap": "同一方向可以承担多个选择角色，完整事件只保存一组并引用其编号。",
    "absence": "未发生失败、没有此前通过方向、没有 Arnoldi，分别报告缺失原因；不替换为其他取样。",
    "comparison": "只比较同一方向固定三块权重下的 L2 与三块无穷范数；初始、IR、Arnoldi 候选分别保留。",
    "precision_stages": "区分生产逐次舍入候选、已有图像重用的残差、long double 一次累加候选及转回 double 后候选。",
    "cost": "记录额外扫描、已有图像复制/组合、方向重建与诊断计时；重用图像不等于零成本。",
    "invalid": "缺报告、解析错误、缺事件或无效数值显式列入 issues，不静默删除运行或方向。",
}

LIMITATIONS = [
    "本轮仅诊断冻结实现，诊断开销消耗预算，不作为公开性能成绩。",
    "范数或范数比相同不代表向量相同，差的范数也不能用两个范数相减代替。",
    "Givens 与重用图像残差均含浮点近似；记录差距不自动证明哪个算子或哪个步骤造成失败。",
    "一次 long double 累加或转回 double 的候选只是在固定已有方向和系数上的诊断，不构成最优舍入搜索。",
    "当前候选未通过不能证明不存在可表示的 double 解；误差上界不是不可表示性的下界。",
    "累计工作量不能定位最后方向；方向内观测计数、生产累计计数端点差分别报告，缺端点时保持缺失。",
    "observed_preconditioner 是整个预条件回调请求/返回；production_preconditioner 是内部 direct 实际启动，二者不必相等。",
    "diagnostic_cache_peak_elements 是全运行缓存高水位，端点不代表本方向的缓存峰值。",
    "diagnostic_seconds_completed_before_event 仅含事件输出前已完成的观测时间；端点差不含最后事件的完整输出成本。",
    "candidate accepted 仅说明生产残差门槛通过，后续预算或恢复仍可失败；方向通过以 direction_return 为准。",
    "candidate_pending 或 candidates_started 大于 completed 表示部分尝试，不能算成一次真实残差拒绝。",
]


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def ratio(numerator, denominator):
    # 只对已有有限非零分母计算比值，不添加人为地板。
    if not finite_number(numerator) or not finite_number(denominator) or denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


EVENT_KINDS = {"direction_begin", "initial_check", "ir_check", "restart", "arnoldi_start", "arnoldi",
               "candidate", "direction_return", "direction_failure", "diagnostic_error"}
CHECK_KINDS = {"initial_check": "initial", "ir_check": "ir", "candidate": "arnoldi_candidate"}
PREFIXES = ("true", "image", "ld_sum", "once_cast")
NORM_SUFFIXES = ("weighted_l2", "dual_scaled_inf", "primal_scaled_inf",
                 "complementarity_scaled_inf", "scaled_inf", "dual_inf", "primal_inf", "complementarity_inf")
DIFFERENCE_PREFIXES = ("image_minus_true", "ld_sum_minus_true", "once_cast_minus_true",
                       "once_cast_minus_ld_sum", "image_minus_ld_sum")
DIRECTION_DIFFERENCE_PREFIXES = ("current_minus_ld_sum", "once_cast_minus_ld_sum", "current_minus_once_cast")
DIRECTION_COUNTERS = ("observed_preconditioner_started", "observed_preconditioner_completed",
                      "observed_products_started", "observed_products_completed", "direction_refinement_steps",
                      "direction_krylov_steps", "candidates_started", "candidates_completed")
DIAGNOSTIC_COUNTERS = ("diagnostic_j_started", "diagnostic_j_completed", "diagnostic_csc_started",
                      "diagnostic_csc_completed", "diagnostic_nnz_visits", "diagnostic_metric_elements",
                      "diagnostic_copied_elements", "diagnostic_cached_images", "diagnostic_reconstruction_terms")
PRODUCTION_COUNTERS = ("production_preconditioner_started", "production_products_started",
                       "reported_refinement_steps", "reported_krylov_steps", "factorizations")


def valid_metric(event, name):
    value = event.get(name)
    return value if event.get(name + "_valid") is True and finite_number(value) else None


def inspect_check(event, position):
    candidate = event.get("event") == "candidate"
    expected = ["true_" + suffix for suffix in NORM_SUFFIXES] + ["production_measure"]
    if candidate:
        expected += [prefix + "_" + suffix for prefix in PREFIXES[1:] for suffix in NORM_SUFFIXES]
        expected += [prefix + "_" + suffix for prefix in DIFFERENCE_PREFIXES for suffix in NORM_SUFFIXES]
        expected += [prefix + "_" + suffix for prefix in DIRECTION_DIFFERENCE_PREFIXES for suffix in ("l2", "inf")]
        expected += ["givens_weighted_l2", "remaining"]
    metrics = {name: valid_metric(event, name) for name in expected}
    missing = [name for name in expected if name not in event or name + "_valid" not in event]
    invalid = [name for name in expected if name not in missing and metrics[name] is None]
    ratios = {}
    if candidate:
        for numerator, denominator in (("true", "givens"), ("true", "image"),
                                       ("image", "givens"), ("once_cast", "ld_sum"),
                                       ("true", "once_cast")):
            ratios[numerator + "_over_" + denominator + "_weighted_l2"] = ratio(
                metrics.get(numerator + "_weighted_l2"), metrics.get(denominator + "_weighted_l2"))
    return {"event_position": position, "event_id": event.get("event_id"),
            "kind": CHECK_KINDS[event["event"]], "stage": event.get("stage"),
            "restart": event.get("restart"), "column": event.get("column"),
            "candidate_id": event.get("candidate_id"), "ir_iteration": event.get("ir_iteration"),
            "accepted_by_production_measure": event.get("accepted"),
            "reconstruction": {key: event.get(key) for key in ("ld_sum_finite", "once_cast_finite", "once_cast_underflows")}
                if candidate else None,
            "metrics": metrics, "ratios": ratios,
            "trigger_estimate": event.get("trigger_estimate"), "trigger_window": event.get("trigger_window"),
            "trigger_remaining_zero": event.get("trigger_remaining_zero"),
            "missing_metrics": missing, "invalid_metrics": invalid}


def counter_delta(begin, end, names, optional=False):
    # 高水位和计时单列，不把它们混入可累加的整数工作账本。
    values, issues = {}, []
    for key in names:
        left, right = (begin or {}).get(key), (end or {}).get(key)
        valid = all(type(value) is int and value >= 0 for value in (left, right))
        if optional:
            valid = valid and all((event or {}).get(key + "_valid") is True for event in (begin, end))
        if valid and right >= left:
            values[key] = right - left
        else:
            values[key] = None
            issues.append({"field": key, "reason": "缺少完整累计端点、类型无效或计数回退"})
    return values, issues


def direction_costs(begin, terminal, events):
    diagnostic, issues = counter_delta(begin, terminal, DIAGNOSTIC_COUNTERS)
    production, production_issues = counter_delta(begin, terminal, PRODUCTION_COUNTERS, optional=True)
    issues.extend(production_issues)
    # 中断后只输出最后可观察值，避免把缺失结束时的观测当作完整总数。
    observed = {}
    for key in DIRECTION_COUNTERS:
        observations = [(event.get("event_id"), event[key]) for event in events
                        if type(event.get(key)) is int and event[key] >= 0]
        observed[key] = {"last_observed": observations[-1][1] if observations else None,
                         "last_event_id": observations[-1][0] if observations else None,
                         "terminal_value": (terminal or {}).get(key)}
    start_time = valid_metric(begin or {}, "diagnostic_seconds_completed_before_event")
    end_time = valid_metric(terminal or {}, "diagnostic_seconds_completed_before_event")
    return {"diagnostic_counter_delta": diagnostic, "production_counter_delta": production,
            "direction_counters": observed,
            "global_cache_peak_endpoints": {"begin": (begin or {}).get("diagnostic_cache_peak_elements"),
                                             "terminal": (terminal or {}).get("diagnostic_cache_peak_elements")},
            "completed_observation_seconds_endpoints": {"begin": start_time, "terminal": end_time,
                "difference_excluding_terminal_event_completion": end_time - start_time
                if start_time is not None and end_time is not None and end_time >= start_time else None}}, issues


def analyze_direction(direction_id, positioned):
    events = [event for _, event in positioned]
    begins = [event for event in events if event.get("event") == "direction_begin"]
    terminals = [event for event in events if event.get("event") in ("direction_return", "direction_failure")]
    issues = []
    if len(begins) != 1:
        issues.append({"kind": "begin_count", "count": len(begins)})
    if len(terminals) != 1:
        issues.append({"kind": "terminal_count", "count": len(terminals)})
    begin = begins[0] if begins else None
    terminal = terminals[-1] if terminals else None
    if begin is not None and events[0] is not begin:
        issues.append({"kind": "begin_not_first"})
    if terminal is not None and events[-1] is not terminal:
        issues.append({"kind": "events_after_terminal"})
    if terminal is None:
        outcome = "missing_terminal"
    elif terminal["event"] == "direction_return":
        outcome = "returned"
    else:
        outcome = "failed"
    context_fields = ("outer_iteration", "phase")
    for key in context_fields:
        values = {event.get(key) for event in events}
        if len(values) != 1 or None in values:
            issues.append({"kind": "inconsistent_context", "field": key, "values": sorted(values, key=str)})
    checks = [inspect_check(event, position) for position, event in positioned if event.get("event") in CHECK_KINDS]
    for check in checks:
        if check["missing_metrics"] or check["invalid_metrics"]:
            issues.append({"kind": "check_metrics", "event_id": check["event_id"],
                           "missing": check["missing_metrics"], "invalid": check["invalid_metrics"]})
    for event in events:
        if event.get("event") not in EVENT_KINDS:
            issues.append({"kind": "unknown_event", "event": event.get("event"), "event_id": event.get("event_id")})
        if event.get("event") == "diagnostic_error":
            issues.append({"kind": "diagnostic_disabled", "event_id": event.get("event_id")})
        if event.get("finite_fields") is False:
            issues.append({"kind": "nonfinite_fields", "event_id": event.get("event_id")})
    costs, cost_issues = direction_costs(begin, terminal, events)
    triggered_krylov = any(event.get("event") in ("arnoldi_start", "arnoldi")
        or (type(event.get("direction_krylov_steps")) is int and event["direction_krylov_steps"] > 0) for event in events)
    last = terminal or events[-1]
    started, completed = last.get("candidates_started"), last.get("candidates_completed")
    partial_candidates = started - completed if type(started) is int and type(completed) is int and started >= completed else None
    if terminal and outcome == "returned":
        termination = "returned"
    elif terminal:
        termination = terminal.get("failure_kind") or "unspecified_failure"
    else:
        termination = "diagnostic_disabled" if any(event.get("event") == "diagnostic_error" for event in events) else "missing_terminal"
    issues.extend({"kind": "diagnostic_cost_delta", **issue} for issue in cost_issues)
    return {"direction_id": direction_id, "outer_iteration": (begin or events[0]).get("outer_iteration"),
            "phase": (begin or events[0]).get("phase"), "first_event_position": positioned[0][0],
            "last_event_position": positioned[-1][0], "outcome": outcome,
            "failure_kind": (terminal or {}).get("failure_kind"),
            "failure_message": (terminal or {}).get("message"),
            "termination": termination, "triggered_krylov": triggered_krylov,
            "partial_candidate_state": {"candidate_pending": last.get("candidate_pending"),
                                        "started_minus_completed": partial_candidates,
                                        "endpoint_is_terminal": terminal is not None},
            "event_counts": {kind: sum(event.get("event") == kind for event in events)
                             for kind in sorted({event.get("event") for event in events}, key=str)},
            "begin": begin, "terminal": terminal, "checks": checks,
            "costs": costs, "issues": issues, "events": events}


def selections(directions, time_limited=False):
    failed = [direction for direction in directions if direction["outcome"] == "failed"
              or (direction["outcome"] == "missing_terminal" and time_limited
                  and direction["termination"] != "diagnostic_disabled")]
    last_failed = failed[-1] if failed else None
    prior = [direction for direction in directions if direction["outcome"] == "returned" and last_failed is not None
             and direction["last_event_position"] < last_failed["first_event_position"]]
    first_krylov = next((direction for direction in directions if direction["triggered_krylov"]), None)
    return {
        "last_failed_or_interrupted": {"direction_id": last_failed["direction_id"] if last_failed else None,
                                       "absence_reason": None if last_failed else "没有明确失败或有运行超时证据的未禁用诊断部分方向"},
        "preceding_returned": {"direction_id": prior[-1]["direction_id"] if prior else None,
                               "absence_reason": None if prior else "无目标失败方向或其前没有明确通过方向"},
        "first_krylov": {"direction_id": first_krylov["direction_id"] if first_krylov else None,
                         "absence_reason": None if first_krylov else "未观察到 Arnoldi 开始事件或正的方向 Krylov 工作计数"},
    }


def analyze_record(row):
    events = row.get("events")
    issues = []
    if not isinstance(events, list):
        issues.append({"kind": "missing_events_array"})
        events = []
    if not row.get("report"):
        issues.append({"kind": "missing_report", "error": row.get("report_parse_error")})
    if row.get("diagnostic_parse_errors"):
        issues.append({"kind": "diagnostic_parse_errors", "errors": row["diagnostic_parse_errors"]})
    if not events:
        issues.append({"kind": "no_events"})
    groups, other_events = {}, []
    previous_event_id = None
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            issues.append({"kind": "invalid_event", "position": position})
            other_events.append(event)
            continue
        event_id = event.get("event_id")
        if type(event_id) is not int or (previous_event_id is not None and event_id <= previous_event_id):
            issues.append({"kind": "event_id_order", "position": position, "event_id": event_id})
        if type(event_id) is int:
            previous_event_id = event_id
        direction_id = event.get("direction_id")
        if type(direction_id) is not int or direction_id <= 0:
            other_events.append(event)
            issues.append({"kind": "ungrouped_event", "position": position, "event": event.get("event")})
            continue
        groups.setdefault(direction_id, []).append((position, event))
    directions = [analyze_direction(direction_id, positioned) for direction_id, positioned in groups.items()]
    report = row.get("report") or {}
    chosen = selections(directions, row.get("wall_timeout") is True or report.get("status") == "time_limit")
    return {"model": row.get("model"), "input": row.get("input"), "variant": row.get("variant"),
            "returncode": row.get("returncode"), "status": report.get("status"), "message": report.get("message"),
            "wall_timeout": row.get("wall_timeout"), "diagnostic_parse_errors": row.get("diagnostic_parse_errors"),
            "event_count": len(events), "direction_count": len(directions), "selections": chosen,
            "partial_direction_ids": [direction["direction_id"] for direction in directions
                                      if direction["outcome"] == "missing_terminal"],
            "outcome_counts": {kind: sum(direction["outcome"] == kind for direction in directions)
                               for kind in ("returned", "failed", "missing_terminal")},
            "termination_counts": {kind: sum(direction["termination"] == kind for direction in directions)
                                   for kind in sorted({direction["termination"] for direction in directions})},
            "analysis_complete": not issues and all(not direction["issues"] for direction in directions),
            "issues": issues, "directions": directions, "ungrouped_events": other_events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="只输出测量前分析规则，不读取任何运行记录")
    parser.add_argument("--runs", type=Path, default=Path(__file__).with_name("v42-newton-runs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("v42-newton-analysis.json"))
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({"analysis_rules": ANALYSIS_RULES, "limitations": LIMITATIONS}, ensure_ascii=False, indent=2))
        return
    path = args.runs
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    raw = path.read_bytes()
    uncompressed = gzip.decompress(raw) if path.suffix == ".gz" else raw
    rows = [json.loads(line) for line in uncompressed.decode("utf-8").splitlines() if line.strip()]
    enabled = [row for row in rows if row.get("variant") == "diagnostic_on"]
    records = [analyze_record(row) for row in enabled]
    result = {"schema": 1, "diagnostic_only": True, "runs_file": path.name,
              "runs_sha256": hashlib.sha256(raw).hexdigest(),
              "uncompressed_runs_sha256": hashlib.sha256(uncompressed).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "analysis_rules": ANALYSIS_RULES, "limitations": LIMITATIONS,
              "record_count": len(rows), "enabled_record_count": len(enabled),
              "enabled_event_count": sum(record["event_count"] for record in records),
              "direction_count": sum(record["direction_count"] for record in records),
              "records_with_issues": sum(not record["analysis_complete"] for record in records),
              "non_enabled_event_counts": [{"model": row.get("model"), "variant": row.get("variant"),
                                             "events": len(row.get("events") or [])}
                                            for row in rows if row.get("variant") != "diagnostic_on"],
              "records": records}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("record_count", "enabled_record_count", "enabled_event_count",
                                                 "direction_count", "records_with_issues")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
