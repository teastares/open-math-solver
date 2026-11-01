"""第85轮：原模型停止oracle的多次恢复、关闭兼容与固定扩集门槛。"""
import argparse
import contextlib
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "8a47239e7f9cf412b2f64eb604eda9ab25c8143a11e4075335b1011ccde3964f"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
PARENT_INST_SHA = "56b815b680ac9d2869794439ced61de783449cf26a672b71c5a883da2a935afd"
WORK = ROOT / ".tools/v88-presolve"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V88_PRESOLVE "
ENVIRONMENT = "SOLVER_V88_PRESOLVE"
QUALITY_BOOLS = ("primal_valid", "primal_feasible", "objective_consistent", "dual_feasible", "gap_satisfied", "numerically_optimal")
QUALITY_NUMBERS = ("objective", "max_row_violation", "max_bound_violation", "max_dual_residual", "objective_gap", "relative_objective_gap")
NATIVE_ITERATIONS = ("simplex_iterations", "ipm_iterations", "pdlp_iterations", "crossover_iterations", "qp_iterations")
TIMINGS = ("preparation", "presolve", "child", "recovery", "verification", "child_verification", "total")
STATUSES = {"disabled", "bypassed", "running", "accepted", "checked_not_optimal", "no_candidate", "presolve_rejected", "native_error", "time_limit", "exception", "not_started"}
PHASES = {"not_started", "bypass", "preparation", "pass_model", "presolve", "child", "oracle", "candidate", "recovery", "projection", "verification", "commit", "finished"}
PROJECTION_STATUSES = {"not_started", "not_needed", "running", "unchanged", "rejected", "accepted", "time_limit", "exception"}
PROJECTION_PHASES = {"not_started", "copy", "primal", "dual", "scan", "verification", "finished"}
NATIVE_OPTIONS = dict(primal_feasibility_tolerance=1e-7, dual_feasibility_tolerance=1e-7,
    ipm_optimality_tolerance=1e-8, lp_presolve_requires_basis_postsolve=True)
MAX_ORACLE_CALLS = 204
MAX_ORIGINAL_ASSESSMENTS = 408
REQUIRED_ORACLE_CASES = {
    "oracle_old_incumbent_cannot_trigger_current": "transaction",
    "oracle_reject_then_accept_same_core": "transaction",
    "oracle_tls_scope_restores_previous": "transaction",
    "oracle_owner_commits_without_tail_recovery": "transaction",
    "native_postsolve_replays_A_B_A_without_lp": "transaction",
    "oracle_budget_preserves_event_work": "budget",
    "oracle_error_preserves_event_work": "exception",
}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("JSON键重复：" + key)
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique, parse_constant=benchmark.invalid_constant,
        parse_float=benchmark.finite_json_float)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_once(path, value):
    if path.exists():
        assert base.typed_equal(read(path), value), "禁止覆盖不一致证据"
    else:
        save(path, value)


assert sha(DATA / "v66-run.py") == HELPER_SHA
_spec = importlib.util.spec_from_file_location("v88_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def fixed_plan():
    assert sha(DATA / "v88-plan.json") == PLAN_SHA
    plan = read(DATA / "v88-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert type(plan["candidate_version"]) is int and plan["candidate_version"] == 88 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V88_PRESOLVE"
    assert plan["diagnostic_marker"] == "V88_PRESOLVE"
    for name, expected in (("max_projection_trials_per_point", 1), ("max_mapped_original_assessments_per_point", 2),
        ("max_original_oracle_calls", 204), ("max_original_oracle_assessments", 408)):
        assert type(plan[name]) is int and plan[name] == expected
    assert plan["parent_instrumentation_file"] == "v87-ipm-instrumentation.py"
    assert plan["parent_instrumentation_sha256"] == PARENT_INST_SHA
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"]) == (5, 35, 200, 1, 0)
    for side in ("candidate", "reference", "compatibility"):
        assert plan[side + "_accuracy"] == "homogeneous_repair"
        assert plan[side + "_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert plan[side + "_backend_version"] == "ipm-cpu-0.53.0"
        assert plan[side + "_objective_scaling"] == "off"
    assert plan["method"] == "homogeneous" and plan["stage_order"] == list(STAGES)
    assert plan["maximum_solver_runs"] == 500
    original = read(DATA / "v66-plan.json")
    assert plan["sets"] == original["sets"]
    return plan


def specification(stage, plan):
    return base.specification(stage, plan)


def variants(stage):
    return ("highs" if stage == "highs" else "frozen_v66", "diagnostic_off" if stage in COMPAT else "diagnostic_on")


def options_for(stage, variant, executable, plan):
    backend = "highs" if variant == "highs" else "ipm"
    values = dict(solver=executable, backend=backend, time_limit=float(plan["time_limit"]),
        wall_timeout=plan["wall_timeout"], threads=plan["threads"], mip_gap=0.0001,
        iteration_limit=None, scaling_passes=None)
    if backend == "ipm":
        values.update(iteration_limit=plan["max_iterations"], scaling_passes=plan["scaling_passes"],
            ipm_method=plan["method"], ipm_linear_accuracy="homogeneous_repair",
            ipm_linear_solver="normal_cholmod_aat", ipm_objective_scaling="off", ipm_direction_retries=0)
        for key, value in plan["shared_ipm"].items():
            values["ipm_" + key.replace("-", "_")] = int(value) if key in ("central-corrections", "krylov-history-size") else value
    else:
        values.update({"highs_" + key.replace("-", "_"): value for key, value in plan["highs_reference"].items()})
    return argparse.Namespace(**values)


def valid_command(row, variant, plan):
    args = row["command"]
    if len(args[3:]) % 2:
        return False
    flags = dict(zip(args[3::2], args[4::2]))
    if len(flags) * 2 != len(args[3:]):
        return False
    backend = "highs" if variant == "highs" else "ipm"
    expected = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
    if backend == "ipm":
        expected.update({"--iteration-limit": "200", "--scaling-passes": "5", "--ipm-method": "homogeneous",
            "--ipm-linear-accuracy": "homogeneous_repair", "--ipm-linear-solver": "normal_cholmod_aat",
            "--ipm-direction-retries": "0", "--ipm-objective-scaling": "off"})
        expected.update({"--ipm-" + key: value for key, value in plan["shared_ipm"].items()})
    else:
        expected.update({"--highs-" + key: value for key, value in plan["highs_reference"].items()})
    return flags == expected


@contextlib.contextmanager
def solver_environment(enabled):
    """单进程串行作用域：每次调用重建环境，结束后逐字恢复。"""
    prior = dict(os.environ)
    clean = clean_environment(prior, enabled)
    try:
        os.environ.clear()
        os.environ.update(clean)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def clean_environment(prior, enabled):
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    clean = {key: value for key, value in prior.items() if not key.upper().startswith("SOLVER_V")
        and key.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    clean.update(dict.fromkeys(threads, "1"))
    if enabled:
        clean[ENVIRONMENT] = "1"
    return clean


def inspect_marker(stderr, variant):
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if variant in ("frozen_v66", "highs"):
        return (None, None) if not lines else (None, "unexpected_marker")
    if len(lines) != 1:
        return None, "marker_count"
    try:
        value = strict_json(lines[0][len(MARKER):])
    except (TypeError, ValueError):
        return None, "invalid_marker_json"
    if not valid_marker(value, variant == "diagnostic_on"):
        return value, "invalid_marker_schema"
    return value, None


def integer(value, maximum=(1 << 63) - 1):
    return type(value) is int and 0 <= value <= maximum


def nullable_number(value):
    return value is None or benchmark.finite_real(value)


def valid_dimensions(value):
    return value is None or isinstance(value, dict) and set(value) == {"n", "m", "nnz"} and all(integer(item) for item in value.values())


def valid_quality(value):
    if value is None: return True
    return isinstance(value, dict) and set(value) == {*QUALITY_BOOLS, *QUALITY_NUMBERS} and \
        all(type(value[key]) is bool for key in QUALITY_BOOLS) and all(nullable_number(value[key]) for key in QUALITY_NUMBERS) and \
        all(value[key] is None or value[key] >= 0 for key in ("max_row_violation", "max_bound_violation", "max_dual_residual"))


def valid_call(value, with_status=True, maximum=None, cumulative=False):
    if not isinstance(value, dict) or set(value) != ({"started", "completed", "status"} if with_status else {"started", "completed"}):
        return False
    if maximum is None:
        maximum = 1 if with_status else 2
    difference = value["started"] - value["completed"] if type(value["started"]) is int and type(value["completed"]) is int else -1
    if not integer(value["started"], maximum) or not integer(value["completed"], maximum) or difference < 0 or not cumulative and difference > 1:
        return False
    if with_status:
        return (value["status"] is None) is (value["completed"] == 0) and (value["status"] is None or type(value["status"]) is int and value["status"] in (-1, 0, 1))
    return True


def valid_native(value):
    if not isinstance(value, dict) or set(value) != {"presolve_status", "model_status", "info_valid", "basis_valid", "time_limit", "options", *NATIVE_ITERATIONS}:
        return False
    if any(value[key] is not None and (type(value[key]) is not int or not -1 <= value[key] <= (1 << 63) - 1) for key in
        ("presolve_status", "model_status", *NATIVE_ITERATIONS)):
        return False
    if any(value[key] is not None and type(value[key]) is not bool for key in ("info_valid", "basis_valid")):
        return False
    if value["time_limit"] is not None and (not benchmark.finite_real(value["time_limit"]) or value["time_limit"] <= 0):
        return False
    options = value["options"]
    return options is None or isinstance(options, dict) and set(options) == set(NATIVE_OPTIONS) and \
        type(options["lp_presolve_requires_basis_postsolve"]) is bool and options["lp_presolve_requires_basis_postsolve"] is True and \
        all(benchmark.finite_real(options[key]) and options[key] == NATIVE_OPTIONS[key] for key in NATIVE_OPTIONS if key != "lp_presolve_requires_basis_postsolve")


def empty_projection():
    return dict(started=0, completed=0, status="not_started", phase="not_started", chosen="none", x_changed=0, y_changed=0,
        max_x_change=None, max_y_change=None, raw_quality=None, trial_quality=None)


def valid_projection(value, original):
    if not isinstance(value, dict) or set(value) != set(empty_projection()) or not integer(value["started"], 1) or \
        not integer(value["completed"], 1) or value["completed"] > value["started"]:
        return False
    if type(value["status"]) is not str or value["status"] not in PROJECTION_STATUSES or type(value["phase"]) is not str or value["phase"] not in PROJECTION_PHASES or \
        type(value["chosen"]) is not str or value["chosen"] not in {"none", "raw", "projected"}:
        return False
    if any(not integer(value[key]) for key in ("x_changed", "y_changed")) or \
        any(not nullable_number(value[key]) or value[key] is not None and value[key] < 0 for key in ("max_x_change", "max_y_change")) or \
        not valid_quality(value["raw_quality"]) or not valid_quality(value["trial_quality"]):
        return False
    if original is not None and (value["x_changed"] > original["n"] or value["y_changed"] > original["m"]):
        return False
    if value["status"] == "not_started":
        return base.typed_equal(value, empty_projection())
    if value["raw_quality"] is None:
        return False
    if value["started"] == 0 and (value["completed"] or value["x_changed"] or value["y_changed"] or value["trial_quality"] is not None):
        return False
    if value["status"] == "not_needed" and (value["started"] or value["chosen"] != "raw" or not all(value["raw_quality"][key] is True for key in QUALITY_BOOLS)):
        return False
    if value["status"] == "not_needed" and value["phase"] != "not_started":
        return False
    if value["status"] in {"unchanged", "rejected", "accepted"} and (value["started"] != 1 or value["completed"] != 1 or value["phase"] != "finished"):
        return False
    if value["status"] == "time_limit" and (value["completed"] != 0 or value["chosen"] != "none"):
        return False
    if value["status"] == "exception" and (value["completed"] != 0 or value["chosen"] != "raw"):
        return False
    if value["status"] in {"unchanged", "rejected"} and value["chosen"] != "raw":
        return False
    if value["status"] == "unchanged" and (value["x_changed"] or value["y_changed"] or value["trial_quality"] is not None):
        return False
    if value["status"] in {"rejected", "accepted"} and (value["trial_quality"] is None or value["x_changed"] + value["y_changed"] == 0):
        return False
    if value["chosen"] == "projected" and (value["started"] != 1 or value["trial_quality"] is None or not all(value["trial_quality"][key] is True for key in QUALITY_BOOLS)):
        return False
    return value["status"] != "accepted" or value["chosen"] == "projected"


def empty_oracle():
    return dict(started=0, completed=0, accepted=0, rejected=0, budget_failures=0, errors=0,
        candidate_event=None, selected_event=None, projection_event=None, total_seconds=0, events=[])


def valid_oracle_event(value, original):
    fields = {"sequence", "iteration", "decision", "postsolve", "assessments", "projection", "original_quality", "timings", "budget"}
    if not isinstance(value, dict) or set(value) != fields or not integer(value["sequence"], MAX_ORACLE_CALLS):
        return False
    if value["iteration"] is not None and not integer(value["iteration"], 200) or \
        type(value["decision"]) is not str or value["decision"] not in {"pending", "accept", "reject", "budget", "error"}:
        return False
    if not valid_call(value["postsolve"]) or not valid_call(value["assessments"], False) or \
        not valid_projection(value["projection"], original) or not valid_quality(value["original_quality"]):
        return False
    timings, budget = value["timings"], value["budget"]
    if not isinstance(timings, dict) or set(timings) != {"total", "recovery", "verification"} or \
        any(not benchmark.finite_real(item) or item < 0 for item in timings.values()):
        return False
    if not isinstance(budget, dict) or set(budget) != {"checks", "loop_units"} or \
        any(not integer(item) for item in budget.values()) or budget["checks"] < budget["loop_units"] // 256:
        return False
    return True


def close_time(left, right):
    return abs(left - right) <= 1e-8


def complete_quality(value):
    return value is not None and all(value[key] is True for key in QUALITY_BOOLS)


def valid_oracle(value, original):
    if not isinstance(value, dict) or set(value) != set(empty_oracle()):
        return False
    counts = ("started", "completed", "accepted", "rejected", "budget_failures", "errors")
    if any(not integer(value[key], MAX_ORACLE_CALLS) for key in counts) or \
        not 0 <= value["started"] - value["completed"] <= 1 or \
        sum(value[key] for key in counts[2:]) != value["completed"] or \
        not benchmark.finite_real(value["total_seconds"]) or value["total_seconds"] < 0:
        return False
    events = value["events"]
    if not isinstance(events, list) or len(events) != value["started"] or \
        any(not valid_oracle_event(event, original) for event in events):
        return False
    decisions = dict.fromkeys(("pending", "accept", "reject", "budget", "error"), 0)
    candidates, projections = [], []
    for index, event in enumerate(events):
        decision, projection, assessed = event["decision"], event["projection"], event["assessments"]
        if event["sequence"] != index or index + 1 < len(events) and decision != "reject":
            return False
        decisions[decision] += 1
        if event["original_quality"] is not None and event["postsolve"]["completed"] != 1 or \
            (assessed["completed"] > 0) is not (event["original_quality"] is not None):
            return False
        if projection["status"] != "not_started":
            if assessed["completed"] < 1:
                return False
            projections.append(index)
        if assessed["started"] > 1 and projection["started"] != 1 or \
            projection["trial_quality"] is not None and assessed["completed"] != 2:
            return False
        if projection["chosen"] == "projected" and assessed["completed"] != 2:
            return False
        if projection["status"] != "not_started":
            chosen = projection["trial_quality"] if projection["chosen"] == "projected" else projection["raw_quality"]
            if not base.typed_equal(chosen, event["original_quality"]):
                return False
        if decision in {"accept", "reject"}:
            if event["original_quality"] is None or projection["status"] == "not_started" or \
                complete_quality(event["original_quality"]) is not (decision == "accept"):
                return False
            candidates.append(index)
        # 错误时钟可能只留下真实子阶段而无完整total；不补造0成本或合成总时长。
        if decision != "error" and event["timings"]["recovery"] + event["timings"]["verification"] > event["timings"]["total"] + 1e-8:
            return False
    if decisions["pending"] != value["started"] - value["completed"] or \
        any(decisions[name] != value[key] for name, key in (("accept", "accepted"), ("reject", "rejected"),
            ("budget", "budget_failures"), ("error", "errors"))):
        return False
    if not close_time(math.fsum(event["timings"]["total"] for event in events), value["total_seconds"]):
        return False
    for key in ("candidate_event", "selected_event", "projection_event"):
        if value[key] is not None and not integer(value[key], len(events) - 1):
            return False
    if value["candidate_event"] != (candidates[-1] if candidates else None) or \
        value["projection_event"] != (projections[-1] if projections else None):
        return False
    if value["selected_event"] is not None and value["selected_event"] != value["candidate_event"]:
        return False
    return True


def valid_marker(value, enabled):
    fields = {"schema", "enabled", "attempted", "accepted", "status", "phase", "route", "original", "reduced", "pass_model", "presolve", "postsolve",
        "native", "child", "assessments", "original_quality", "timings", "budget", "explicit_highs_lp_solve_calls", "projection", "oracle"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema"]) is not int or value["schema"] != 1 or \
        type(value["enabled"]) is not bool or value["enabled"] is not enabled or any(type(value[key]) is not bool for key in ("attempted", "accepted")) or \
        type(value["explicit_highs_lp_solve_calls"]) is not int or value["explicit_highs_lp_solve_calls"] != 0:
        return False
    if type(value["status"]) is not str or value["status"] not in STATUSES or type(value["phase"]) is not str or value["phase"] not in PHASES or \
        type(value["route"]) is not str or value["route"] not in {"none", "original", "reduced", "empty"}:
        return False
    if not valid_dimensions(value["original"]) or not valid_dimensions(value["reduced"]) or not valid_native(value["native"]) or \
        not valid_quality(value["original_quality"]) or not valid_projection(value["projection"], value["original"]) or \
        not valid_oracle(value["oracle"], value["original"]):
        return False
    if any(not valid_call(value[key]) for key in ("pass_model", "presolve")) or \
        not valid_call(value["postsolve"], maximum=MAX_ORACLE_CALLS, cumulative=True) or \
        not valid_call(value["assessments"], False, MAX_ORIGINAL_ASSESSMENTS, cumulative=True):
        return False
    child = value["child"]
    if not isinstance(child, dict) or set(child) != {"started", "completed", "status", "has_solution"} or \
        not integer(child["started"], 1) or not integer(child["completed"], 1) or child["completed"] > child["started"] or \
        child["status"] is not None and (type(child["status"]) is not str or not child["status"]) or \
        child["has_solution"] is not None and type(child["has_solution"]) is not bool:
        return False
    if child["completed"] == 0 and (child["status"] is not None or child["has_solution"] is not None):
        return False
    timings, budget = value["timings"], value["budget"]
    if not isinstance(timings, dict) or set(timings) != set(TIMINGS) or any(not benchmark.finite_real(item) or item < 0 for item in timings.values()) or \
        not isinstance(budget, dict) or set(budget) != {"checks", "loop_units"} or any(not integer(item) for item in budget.values()) or \
        budget["checks"] < budget["loop_units"] // 256:
        return False
    if child["completed"] == 0 and timings["child_verification"] != 0:
        return False
    oracle, events = value["oracle"], value["oracle"]["events"]
    for key in ("postsolve", "assessments"):
        if any(value[key][field] != sum(event[key][field] for event in events) for field in ("started", "completed")):
            return False
    final_post_status = next((event["postsolve"]["status"] for event in reversed(events) if event["postsolve"]["completed"]), None)
    if value["postsolve"]["status"] != final_post_status or \
        any(sum(event["budget"][key] for event in events) > budget[key] for key in budget):
        return False
    if any(not close_time(timings[key], math.fsum(event["timings"][key] for event in events)) for key in ("recovery", "verification")):
        return False
    if not value["attempted"]:
        if value["accepted"] or value["route"] != "none" or value["original"] is not None or value["reduced"] is not None or value["original_quality"] is not None or \
            not base.typed_equal(value["projection"], empty_projection()) or not base.typed_equal(oracle, empty_oracle()):
            return False
        if any(value[key]["started"] or value[key]["completed"] for key in ("pass_model", "presolve", "postsolve", "child", "assessments")) or \
            any(item is not None for item in value["native"].values()) or any(timings.values()) or any(budget.values()):
            return False
        return value["phase"] == ("not_started" if not enabled or value["status"] == "exception" else "bypass") and \
            value["status"] in ({"bypassed", "exception"} if enabled else {"disabled"})
    if not enabled or value["original"] is None or value["status"] in {"disabled", "bypassed", "not_started"}:
        return False
    passed, pre = value["pass_model"], value["presolve"]
    if pre["started"] and passed["completed"] != 1 or oracle["started"] and pre["completed"] != 1:
        return False
    if passed["started"] and value["native"]["options"] is None or pre["started"] and value["native"]["time_limit"] is None:
        return False
    observed = ("presolve_status", "model_status", "info_valid", "basis_valid", *NATIVE_ITERATIONS)
    if pre["completed"] and any(value["native"][key] is None for key in observed):
        return False
    if child["started"] and (pre["completed"] != 1 or value["route"] not in {"original", "reduced"} or value["reduced"] is None):
        return False
    if value["route"] in {"none", "original"} and not base.typed_equal(oracle, empty_oracle()):
        return False
    if value["route"] == "original":
        if not base.typed_equal(value["original"], value["reduced"]) or not base.typed_equal(value["projection"], empty_projection()) or \
            value["original_quality"] is not None and child["completed"] != 1:
            return False
    if value["route"] == "reduced":
        if value["reduced"] is None or oracle["started"] and child["started"] != 1 or any(event["iteration"] is None for event in events):
            return False
        iterations = [event["iteration"] for event in events]
        if iterations != sorted(iterations):
            return False
    if value["route"] == "empty" and (value["reduced"] != dict(n=0, m=0, nnz=0) or child["started"] or \
        oracle["started"] > 1 or any(event["iteration"] is not None for event in events)):
        return False
    if oracle["started"] and value["route"] not in {"reduced", "empty"}:
        return False
    if value["route"] != "original":
        candidate = oracle["candidate_event"]
        expected_quality = events[candidate]["original_quality"] if candidate is not None else None
        if not base.typed_equal(expected_quality, value["original_quality"]):
            return False
        projection = oracle["projection_event"]
        expected_projection = events[projection]["projection"] if projection is not None else empty_projection()
        if not base.typed_equal(expected_projection, value["projection"]):
            return False
    selected = oracle["selected_event"]
    if selected is not None and (value["status"] not in {"accepted", "checked_not_optimal"} or value["phase"] != "finished" or \
        value["route"] == "reduced" and child["completed"] != 1):
        return False
    if selected is not None and complete_quality(value["original_quality"]) is not value["accepted"]:
        return False
    if value["route"] in {"reduced", "empty"} and value["status"] in {"accepted", "checked_not_optimal"} and selected is None:
        return False
    # child是包含oracle的实测区间；回调recovery/verification不能再作为兄弟阶段相加。
    shell_exception = value["status"] == "exception" and value["phase"] == "preparation" and timings["total"] == 0 and \
        value["route"] == "none" and not events and all(value[key]["started"] == 0 for key in ("pass_model", "presolve", "child"))
    unknown_clock = not value["accepted"] and (oracle["errors"] > 0 or value["status"] == "exception" and timings["total"] == 0)
    if not unknown_clock and (timings["child_verification"] > timings["child"] + 1e-8 or \
        not shell_exception and timings["preparation"] + timings["presolve"] + timings["child"] + \
        (oracle["total_seconds"] if value["route"] == "empty" else 0) > timings["total"] + 1e-8):
        return False
    if value["route"] == "reduced" and not unknown_clock and oracle["total_seconds"] > timings["child"] + 1e-8:
        return False
    native = value["native"]
    if value["route"] != "none" and (pre["completed"] != 1 or native["presolve_status"] != {"original": 0, "reduced": 3, "empty": 4}[value["route"]]):
        return False
    if any(native[key] is not None and native[key] > 0 for key in NATIVE_ITERATIONS) or native["basis_valid"] is True:
        if value["status"] not in {"native_error", "time_limit", "exception"} or value["accepted"]:
            return False
    if value["accepted"]:
        if value["status"] != "accepted" or value["phase"] != "finished" or not complete_quality(value["original_quality"]) or timings["total"] >= 5:
            return False
        return (child["completed"] == 1 and child["status"] == "optimal" and child["has_solution"] is True) if value["route"] == "original" else \
            selected is not None and events[selected]["decision"] == "accept"
    return value["status"] != "accepted"

def valid_process(row):
    if row.get("presolve_marker_error") is not None or row.get("configuration_matches") is not True:
        return False
    if not base.valid_process(row):
        return False
    report = row["report"]
    return report["backend"] != "ipm" or benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report) and benchmark.valid_homogeneous_repair_work(report)


def configuration_matches(report, stage, variant, model, plan):
    side = "reference" if variant == variants(stage)[0] else "candidate"
    backend = "highs" if variant == "highs" else "ipm"
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == backend \
        and report.get("backend_version") == ("1.15.1" if backend == "highs" else "ipm-cpu-0.53.0") \
        and base.typed_equal(report.get("options"), base.expected_options(stage, side, plan))


def pair_result(a, b, plan):
    answer = base.pair_result(a, b, plan)
    if not valid_process(a) or not valid_process(b):
        answer.update(valid_processes=False, category="invalid_process")
    return answer


def quality_fields(row, model, expected_solver, plan):
    numerical = compare_lp.numerical_seconds(row)
    failure = compare_lp.quality_failure(row, model, row["model_sha256_after"], expected_solver,
        row["solver_sha256_after"] if row["solver_sha256_before"] == row["solver_sha256_after"] else None)
    wall_exceeded = row["elapsed_wall_seconds"] > plan["wall_timeout"]
    numerical_exceeded = numerical > plan["time_limit"] if benchmark.finite_real(numerical) else None
    probe = dict(row, numerical_seconds=numerical)
    if not valid_process(probe):
        failure = "invalid_process"
    elif failure is None and wall_exceeded:
        failure = "wall_budget_exceeded"
    elif failure is None and numerical_exceeded:
        failure = "numerical_budget_exceeded"
    return dict(numerical_seconds=numerical, numerical_budget_exceeded=numerical_exceeded,
        wall_budget_exceeded=wall_exceeded, quality_failure=failure, quality_pass=failure is None,
        quality_penalized_wall_seconds=min(row["elapsed_wall_seconds"], plan["wall_timeout"]) if failure is None else plan["wall_timeout"],
        quality_penalized_numerical_seconds=min(numerical, plan["time_limit"]) if failure is None else plan["time_limit"])


def strict_same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def valid_math_details(value):
    groups = ("wrapper", "mapping", "transaction", "budget", "exception")
    fields = {"passed", "public_lp_runs", "explicit_highs_lp_solve_calls", "scope", "cases", "component_ipm_calls",
        "synthetic_core_calls", "details", *(group + "_cases" for group in groups)}
    if not isinstance(value, dict) or set(value) != fields or value["passed"] is not True or \
        type(value["public_lp_runs"]) is not int or value["public_lp_runs"] != 0 or \
        type(value["explicit_highs_lp_solve_calls"]) is not int or value["explicit_highs_lp_solve_calls"] != 0:
        return False
    if not isinstance(value["scope"], str) or not value["scope"] or any(type(value[key]) is not int or value[key] <= 0 for key in
        ("cases", "component_ipm_calls", "synthetic_core_calls", *(group + "_cases" for group in groups))):
        return False
    if not isinstance(value["details"], list) or len(value["details"]) != value["cases"] or sum(value[group + "_cases"] for group in groups) != value["cases"]:
        return False
    ids, counts = set(), dict.fromkeys(groups, 0)
    for item in value["details"]:
        if not isinstance(item, dict) or set(item) != {"id", "group"} or not isinstance(item["id"], str) or \
            not item["id"] or not item["id"].isascii() or item["id"] in ids or type(item["group"]) is not str or item["group"] not in groups:
            return False
        ids.add(item["id"]); counts[item["group"]] += 1
    return all(counts[group] == value[group + "_cases"] for group in groups)


def valid_math_evidence(role, checks, raw, work):
    if role != "ipm" or not isinstance(checks, dict) or set(checks) != {"passed", "public_lp_runs", "details"} or checks.get("passed") is not True or \
        type(checks.get("public_lp_runs")) is not int or checks["public_lp_runs"] != 0 or not isinstance(raw, dict):
        return False
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("wall_timeout") is not False or \
        raw.get("command") != [str(work / "ipm-self-test.exe")] or raw.get("cwd") != str(ROOT) or raw.get("stderr") != "":
        return False
    if not isinstance(raw.get("stdout"), str): return False
    try:
        value = strict_json(raw["stdout"])
        matching = strict_same(value, checks.get("details"))
    except (TypeError, ValueError):
        return False
    return matching and valid_math_details(value)


def math_evidence_self_check():
    groups = ("wrapper", "mapping", "transaction", "budget", "exception")
    value = dict(passed=True, public_lp_runs=0, explicit_highs_lp_solve_calls=0, component_ipm_calls=1, synthetic_core_calls=1,
        scope="真实组件调用与注入核心分开记账", cases=5, details=[dict(id=group + "_one", group=group) for group in groups],
        **{group + "_cases": 1 for group in groups})
    checks = dict(passed=True, public_lp_runs=0, details=value)
    raw = dict(command=[str(WORK / "ipm-self-test.exe")], cwd=str(ROOT), returncode=0, wall_timeout=False,
        stdout=json.dumps(value), stderr="")
    assert valid_math_evidence("ipm", checks, raw, WORK)
    for key, invalid in (("returncode", False), ("returncode", 1), ("wall_timeout", True), ("stderr", "污染"),
        ("command", []), ("cwd", "wrong"), ("stdout", "{}"), ("stdout", '{"passed":true,"passed":true}')):
        changed = dict(raw); changed[key] = invalid
        assert not valid_math_evidence("ipm", checks, changed, WORK)
    for key, invalid in (("public_lp_runs", False), ("public_lp_runs", 1), ("explicit_highs_lp_solve_calls", True),
        ("component_ipm_calls", 0), ("component_ipm_calls", True), ("synthetic_core_calls", True), ("wrapper_cases", 1.0),
        ("cases", 5.0), ("cases", True), ("details", []), ("scope", True)):
        changed = copy.deepcopy(value); changed[key] = invalid
        assert not valid_math_evidence("ipm", dict(checks, details=changed), dict(raw, stdout=json.dumps(changed)), WORK)
    wrong = copy.deepcopy(checks); wrong["details"]["cases"] = True
    assert not valid_math_evidence("ipm", wrong, raw, WORK)
    for key, invalid in (("id", ""), ("id", "非ASCII"), ("group", "wrong"), ("group", True)):
        changed = copy.deepcopy(value); changed["details"][0][key] = invalid
        assert not valid_math_evidence("ipm", dict(checks, details=changed), dict(raw, stdout=json.dumps(changed)), WORK)
    changed = copy.deepcopy(value); changed["details"][1]["id"] = changed["details"][0]["id"]
    assert not valid_math_details(changed)

def freeze(work, plan):
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    identity_path = DATA / "ipm-v66-identity.json"
    assert sha(identity_path) == plan["parent_identity_sha256"]
    parent = read(identity_path)
    assert build["current_files"] == build["frozen_files"] == parent["files"]
    assert build["actual_libraries"] == parent["libraries"]
    assert len(parent["files"]) == 95 and len(parent["libraries"]) == 17
    assert build["baseline_verification_sha256"] == plan["baseline_verification_sha256"]
    assert build["production_changes"] is False
    assert Path(build["frozen_parent_executable"]) == PARENT
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-presolve.exe"
    assert set(build["commands"]["compile"]) == {"ipm"}
    assert build["commands"]["link"].count(str(work / "ipm-presolve.obj")) == 1
    assert all("normal-pair" not in str(arg) and "libsolver_ipm.a" not in str(arg) for arg in build["commands"]["link"])
    assert any(str(arg).endswith("libsolver_normal_cholesky.a") for arg in build["commands"]["link"])
    protected = {str(Path(__file__)): sha(__file__), str(DATA / "v66-run.py"): HELPER_SHA,
        str(build_path): sha(build_path), str(PARENT): plan["parent_solver_sha256"],
        str(candidate): build["executable_sha256"],
        str(ROOT / "scripts/benchmark.py"): sha(ROOT / "scripts/benchmark.py"),
        str(ROOT / "scripts/compare_lp.py"): sha(ROOT / "scripts/compare_lp.py")}
    for group in ("protected_before_and_after", "generated_inputs_before_and_after",
        "compile_inputs_after_and_final", "link_explicit_before_and_after",
        "test_link_explicit_before_and_after", "outputs"):
        for path, digest in build[group].items():
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for group in ("files", "libraries"):
        for name, digest in parent[group].items():
            path = str(ROOT / name)
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for name, digest in parent["files"].items():
        path = str(work / "frozen" / name)
        assert path not in protected or protected[path] == digest
        protected[path] = digest
    for name in plan["new_code_and_plan_files"]:
        path = str(DATA / name)
        assert build["protected_before_and_after"][path] == sha(path)
        protected[path] = sha(path)
    for name in ("static-check.json", "ipm-self-check.json"):
        path = str(work / name)
        assert path in protected and read(path)["passed"] is True
    for role in ("ipm",):
        checks = read(work / (role + "-self-check.json"))
        assert valid_math_evidence(role, checks, read(work / (role + "-self-run.json")), work), "数学摘要与实际执行原件不同"
        assert strict_same(build["math_self_tests"][role], {**checks["details"], "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"})
    extension_meta = build["math_coverage_extension"]
    assert set(extension_meta) == {"file", "sha256", "preserved_predecessor_cases", "new_cases"}
    assert extension_meta["file"] == "ipm-self-extension-check.json"
    extension_path = work / extension_meta["file"]
    assert protected[str(extension_path)] == build["outputs"][str(extension_path)] == extension_meta["sha256"] == sha(extension_path)
    extension = read(extension_path)
    assert set(extension) == {"passed", "public_lp_runs", "predecessor_evidence_file", "predecessor_evidence_sha256",
        "preserved_predecessor_cases", "new_cases", "required_new_cases", "component_calls"}
    assert extension["passed"] is True and type(extension["public_lp_runs"]) is int and extension["public_lp_runs"] == 0
    predecessor = read(DATA / plan["evidence_file"])["math_self_checks"]
    assert predecessor["passed"] is True and valid_math_details(predecessor["details"])
    old, current = predecessor["details"], checks["details"]
    assert extension["predecessor_evidence_file"] == str(DATA / plan["evidence_file"]) and \
        extension["predecessor_evidence_sha256"] == plan["evidence_sha256"]
    old_cases = {item["id"]: item["group"] for item in old["details"]}
    current_cases = {item["id"]: item["group"] for item in current["details"]}
    assert old_cases.items() <= current_cases.items() and REQUIRED_ORACLE_CASES.items() <= current_cases.items()
    assert not (REQUIRED_ORACLE_CASES.keys() & old_cases.keys())
    added = [item for item in current["details"] if item["id"] not in old_cases]
    assert added and strict_same(extension["new_cases"], added)
    assert strict_same(extension["required_new_cases"], [{"id": key, "group": group} for key, group in REQUIRED_ORACLE_CASES.items()])
    assert type(extension["preserved_predecessor_cases"]) is int and extension["preserved_predecessor_cases"] == old["cases"] == extension_meta["preserved_predecessor_cases"]
    assert type(extension_meta["new_cases"]) is int and extension_meta["new_cases"] == len(added) and current["cases"] == old["cases"] + len(added)
    assert set(extension["component_calls"]) == {"component_ipm_calls", "synthetic_core_calls"}
    for name, pair in extension["component_calls"].items():
        assert set(pair) == {"predecessor", "current", "change"} and all(type(item) is int for item in pair.values())
        assert pair == {"predecessor": old[name], "current": current[name], "change": current[name] - old[name]}
    path = str(work / "v88-presolve-state.hpp")
    assert build["private_state_header"] == path and path in build["generated_inputs_before_and_after"] and path in build["compile_inputs_after_and_final"]
    assert set(build["generated_inputs_before_and_after"]) == {str(work / "ipm-presolve.cpp"), path}
    assert set(build["commands"]["original_compile"]) == {"ipm", "highs", "main"}
    assert build["commands"]["highs_include_arguments"] and all(arg.startswith("-I") for arg in build["commands"]["highs_include_arguments"])
    parent_inst = str(DATA / "v87-ipm-instrumentation.py")
    assert build["readonly_parent_instrumentation"] == {"path": parent_inst, "sha256": PARENT_INST_SHA}
    assert protected[parent_inst] == PARENT_INST_SHA
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v88-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v88-{stage}-manifest.json")
    summary = read(DATA / f"v88-{stage}-summary.json")
    raw = DATA / f"v88-{stage}-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v88-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v88-run-identity.json")
    assert manifest["plan_sha256"] == PLAN_SHA and manifest["models"] == expected["models"]
    assert manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert manifest["stage"] == stage and manifest["repeats"] == expected["repeats"]
    assert manifest["variants"] == list(variants(stage))
    for field in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[field] == plan[field]
    scheduled = list(compare_lp.schedule(expected["models"], expected["repeats"], variants(stage)))
    assert len(rows) == len(scheduled)
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, scheduled)):
        executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
        digest = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
        assert (row["run_index"], row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"]) == (index, model["model_index"], repeat, position, variant)
        assert row["input"] == model["input"] and row["seed"] == plan["seed"]
        assert row["command"][:3] == [str(executable), "solve", model["input"]]
        assert valid_command(row, variant, plan)
        assert row["presolve_environment"] == ({"SOLVER_V88_PRESOLVE": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert base.typed_equal(row["presolve_marker"], marker) and row["presolve_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert base.typed_equal(row["report"], strict_json(row["stdout"]))
        for field, value in quality_fields(row, model, digest, plan).items():
            assert base.typed_equal(row[field], value), (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows))
    assert base.typed_equal(computed, summary)
    return manifest, summary, rows


def compatibility():
    plan = fixed_plan()
    pairs = []
    for stage in COMPAT:
        _, _, rows = archive(stage)
        for model in specification(stage, plan)["models"]:
            selected = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
            a, b = (selected[variant] for variant in variants(stage))
            pairs.append({"stage": stage, "input": model["input"], **pair_result(a, b, plan)})
    assert len(pairs) == 17
    return {"passed": all(pair["category"] == "equal" for pair in pairs), "runs": 34, "pairs": pairs,
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "plan_sha256": PLAN_SHA}


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    result = base.gate_decision(development, probe)
    result["plan_sha256"] = PLAN_SHA
    result["valid_processes"] = all(item["valid_process_count"] == item["run_count"] for item in (development, probe))
    result["eligible_for_netlib"] = result["eligible_for_netlib"] and result["valid_processes"]
    return result


def run(stage, work):
    plan = fixed_plan()
    identity = freeze(work, plan)
    if stage not in COMPAT:
        result = compatibility()
        save_once(DATA / "v88-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v88-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v88-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v88-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V88_PRESOLVE=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；成批求解计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v88-{stage}-manifest.json", manifest)
    raw = DATA / f"v88-{stage}-runs.jsonl.gz"
    rows = []
    with gzip.open(raw, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in compare_lp.schedule(models, expected["repeats"], variants(stage)):
            executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
            expected_solver = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
            before = sha(executable)
            options = options_for(stage, variant, executable, plan)
            with solver_environment(variant == "diagnostic_on"):
                row = benchmark.run_one(options, Path(model["input"]), plan["seed"], len(rows))
            after, model_after = sha(executable), sha(model["input"])
            marker, marker_error = inspect_marker(row["stderr"], variant)
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position,
                requested_variant=variant, frozen_model_sha256=model["sha256"], model_sha256_after=model_after,
                solver_sha256_before=before, solver_sha256_after=after, frozen_solver_sha256=expected_solver,
                presolve_environment={"SOLVER_V88_PRESOLVE": "1"} if variant == "diagnostic_on" else {},
                presolve_marker=marker, presolve_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "presolve_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows))
    save(DATA / f"v88-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"], "valid_process_count": summary["valid_process_count"]}, ensure_ascii=False), flush=True)


def self_check():
    plan = fixed_plan()
    inherited = base.validation_self_check()
    math_evidence_self_check()
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    before = dict(os.environ)
    with solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V88_PRESOLVE": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V88_PRESOLVE"] == "1"
    assert dict(os.environ) == before
    polluted = {"solver_v1": "1", "SoLvEr_V88_PrEsOlVe": "2", "openblas_coretype": "x",
        "omp_num_threads": "99", "OpenBlas_Num_Threads": "88", "mkl_NUM_threads": "77", "PATH": "保留"}
    assert clean_environment(polluted, False) == {"PATH": "保留", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    assert clean_environment(polluted, True) == dict(clean_environment(polluted, False), **{ENVIRONMENT: "1"})
    for stage in STAGES:
        for variant in variants(stage):
            value = options_for(stage, variant, PARENT, plan)
            assert value.time_limit == 5.0 and value.wall_timeout == 35 and value.threads == 1
            if variant == "highs":
                assert value.backend == "highs" and value.highs_solver == "choose"
            else:
                assert value.ipm_linear_solver == "normal_cholmod_aat" and value.ipm_linear_accuracy == "homogeneous_repair"
                assert value.ipm_objective_scaling == "off" and value.ipm_direction_retries == 0
    return {"passed": True, "public_lp_runs": 0, "inherited_count_and_budget_checks": inherited,
        "environment_cases": 6, "configuration_cases": 12, "math_evidence_roles": 1, "marker_cases": marker_self_check()}


def empty_marker(enabled):
    call = dict(started=0, completed=0, status=None)
    native = {key: None for key in ("presolve_status", "model_status", "info_valid", "basis_valid", "time_limit", "options", *NATIVE_ITERATIONS)}
    return dict(schema=1, enabled=enabled, attempted=False, accepted=False, status="bypassed" if enabled else "disabled",
        phase="bypass" if enabled else "not_started", route="none", original=None, reduced=None,
        pass_model=copy.deepcopy(call), presolve=copy.deepcopy(call), postsolve=copy.deepcopy(call), native=native,
        child=dict(started=0, completed=0, status=None, has_solution=None), assessments=dict(started=0, completed=0), original_quality=None,
        timings=dict.fromkeys(TIMINGS, 0), budget=dict(checks=0, loop_units=0), explicit_highs_lp_solve_calls=0, projection=empty_projection(), oracle=empty_oracle())


def marker_self_check():
    good = dict(**dict.fromkeys(QUALITY_BOOLS, True), **dict.fromkeys(QUALITY_NUMBERS, 0.0))
    def quality(sequence):
        value = copy.deepcopy(good)
        value.update(dual_feasible=False, numerically_optimal=False, objective=sequence + 1.0, max_dual_residual=1.0)
        return value
    def prepared(route="reduced"):
        value = empty_marker(True)
        value.update(attempted=True, status="no_candidate", phase="finished", route=route,
            original=dict(n=2, m=2, nnz=4), reduced=dict(n=1, m=1, nnz=1))
        value["pass_model"].update(started=1, completed=1, status=0)
        value["presolve"].update(started=1, completed=1, status=0)
        value["native"].update(presolve_status=3, model_status=0, info_valid=False, basis_valid=False,
            time_limit=4.9, options=copy.deepcopy(NATIVE_OPTIONS), **dict.fromkeys(NATIVE_ITERATIONS, -1))
        value["child"].update(started=1, completed=1, status="iteration_limit", has_solution=False)
        value["timings"].update(preparation=0.1, presolve=0.1, child=0.6, child_verification=0.03, total=1.0)
        value["budget"].update(checks=10)
        if route == "original":
            value["reduced"] = copy.deepcopy(value["original"])
            value["native"]["presolve_status"] = 0
        if route == "empty":
            value["reduced"] = dict(n=0, m=0, nnz=0)
            value["native"]["presolve_status"] = 4
            value["child"] = dict(started=0, completed=0, status=None, has_solution=None)
            value["timings"].update(child=0, child_verification=0)
        return value
    def event(sequence, decision="reject", projected=False):
        q = copy.deepcopy(good) if decision == "accept" else quality(sequence)
        projection = dict(empty_projection(), status="not_needed", chosen="raw", raw_quality=copy.deepcopy(q)) if decision == "accept" else \
            dict(empty_projection(), started=1, completed=1, status="unchanged", phase="finished", chosen="raw",
                max_x_change=0, max_y_change=0, raw_quality=copy.deepcopy(q))
        assessed = dict(started=1, completed=1)
        if projected:
            projection.update(started=1, completed=1, status="accepted", phase="finished", chosen="projected",
                x_changed=1, max_x_change=1e-7, max_y_change=0, raw_quality=quality(sequence), trial_quality=copy.deepcopy(good))
            assessed.update(started=2, completed=2)
        return dict(sequence=sequence, iteration=sequence, decision=decision, postsolve=dict(started=1, completed=1, status=0),
            assessments=assessed, projection=projection, original_quality=q,
            timings=dict(total=0.03, recovery=0.01, verification=0.01), budget=dict(checks=2, loop_units=0))
    def attach(value, events, selected=None):
        value = copy.deepcopy(value)
        value["oracle"] = oracle = empty_oracle()
        oracle.update(started=len(events), completed=sum(row["decision"] != "pending" for row in events), events=copy.deepcopy(events),
            total_seconds=math.fsum(row["timings"]["total"] for row in events))
        for decision, field in (("accept", "accepted"), ("reject", "rejected"), ("budget", "budget_failures"), ("error", "errors")):
            oracle[field] = sum(row["decision"] == decision for row in events)
        candidates = [row["sequence"] for row in events if row["decision"] in {"accept", "reject"}]
        helpers = [row["sequence"] for row in events if row["projection"]["status"] != "not_started"]
        oracle.update(candidate_event=candidates[-1] if candidates else None, selected_event=selected,
            projection_event=helpers[-1] if helpers else None)
        value["original_quality"] = copy.deepcopy(events[candidates[-1]]["original_quality"]) if candidates else None
        value["projection"] = copy.deepcopy(events[helpers[-1]]["projection"]) if helpers else empty_projection()
        for key in ("postsolve", "assessments"):
            for count in ("started", "completed"):
                value[key][count] = sum(row[key][count] for row in events)
        value["postsolve"]["status"] = next((row["postsolve"]["status"] for row in reversed(events) if row["postsolve"]["completed"]), None)
        for key in ("recovery", "verification"):
            value["timings"][key] = math.fsum(row["timings"][key] for row in events)
        for key in ("checks", "loop_units"):
            value["budget"][key] = sum(row["budget"][key] for row in events) + (10 if key == "checks" else 0)
        return value

    empty, bypass = empty_marker(False), empty_marker(True)
    early = copy.deepcopy(bypass); early.update(status="exception", phase="not_started")
    shell = copy.deepcopy(bypass)
    shell.update(attempted=True, status="exception", phase="preparation", original=dict(n=2, m=2, nnz=4))
    shell["timings"]["preparation"] = 0.01
    pass_partial = copy.deepcopy(shell)
    pass_partial.update(status="time_limit", phase="pass_model")
    pass_partial["timings"]["total"] = 5.1
    pass_partial["pass_model"]["started"] = 1
    pass_partial["native"]["options"] = copy.deepcopy(NATIVE_OPTIONS)
    pre_partial = copy.deepcopy(pass_partial)
    pre_partial.update(phase="presolve")
    pre_partial["pass_model"].update(completed=1, status=0)
    pre_partial["presolve"]["started"] = 1
    pre_partial["native"]["time_limit"] = 4.9
    no_oracle = prepared()
    original = prepared("original")
    original.update(status="accepted", accepted=True, original_quality=copy.deepcopy(good))
    original["child"].update(status="optimal", has_solution=True)
    rejected = attach(prepared(), [event(0)], 0)
    rejected.update(status="checked_not_optimal")
    accepted = attach(prepared(), [event(0), event(1, "accept", True)], 1)
    accepted.update(status="accepted", accepted=True)
    accepted["child"].update(status="optimal", has_solution=True)
    empty_accepted = event(0, "accept"); empty_accepted["iteration"] = None
    empty_accepted = attach(prepared("empty"), [empty_accepted], 0)
    empty_accepted.update(status="accepted", accepted=True)
    owner_budget = copy.deepcopy(accepted)
    owner_budget.update(status="time_limit", accepted=False, phase="commit")
    owner_budget["oracle"]["selected_event"] = None
    owner_budget["timings"]["total"] = 5.1
    owner_clock = copy.deepcopy(owner_budget)
    owner_clock.update(status="exception", phase="finished")
    owner_clock["timings"]["total"] = 0
    # 末事件有完整诊断但budget/error不能覆盖已经拥有的前一候选。
    last_budget = event(1)
    last_budget.update(decision="budget", projection=empty_projection())
    stale_candidate = attach(prepared(), [event(0), last_budget])
    stale_candidate.update(status="time_limit", phase="child")
    stale_candidate["timings"]["total"] = 5.1
    helper_budget = event(1)
    helper_budget.update(decision="budget", assessments=dict(started=2, completed=2))
    helper_budget["projection"].update(completed=0, status="time_limit", phase="verification", chosen="none",
        x_changed=1, max_x_change=1e-7, trial_quality=copy.deepcopy(good))
    partial_projection = attach(prepared(), [event(0), helper_budget])
    partial_projection.update(status="time_limit", phase="child")
    partial_projection["timings"]["total"] = 5.1
    last_error = event(1)
    last_error["decision"] = "error"
    last_error["timings"]["total"] = 0
    error_partial = attach(prepared(), [event(0), last_error])
    error_partial.update(status="native_error", phase="child")
    error_clock = copy.deepcopy(error_partial)
    error_clock.update(status="exception")
    error_clock["timings"]["total"] = 0
    pending = event(1)
    pending.update(decision="pending", projection=empty_projection(), original_quality=None,
        assessments=dict(started=0, completed=0), postsolve=dict(started=1, completed=0, status=None))
    pending["timings"]["verification"] = 0
    pending_marker = attach(prepared(), [event(0), pending])
    pending_marker.update(status="exception", phase="oracle")
    pending_marker["child"].update(completed=0, status=None, has_solution=None)
    pending_marker["timings"]["child_verification"] = 0
    # 两次可选trial异常都退回raw并继续，累计started-completed可大于一。
    aborted_events = [event(0), event(1)]
    for row in aborted_events:
        row["assessments"].update(started=2, completed=1)
        row["projection"].update(completed=0, status="exception", phase="verification", x_changed=1, max_x_change=None)
    cumulative_partial = attach(prepared(), aborted_events, 1)
    cumulative_partial.update(status="checked_not_optimal")
    # 高计数仅检验上界与索引守恒；不把理论计数当新增实际预算。
    many = [event(index) for index in range(MAX_ORACLE_CALLS)]
    for row in many:
        row["timings"] = dict.fromkeys(("total", "recovery", "verification"), 0)
        row["iteration"] = min(200, row["iteration"])
        row["assessments"].update(started=2, completed=2)
        row["projection"].update(status="rejected", x_changed=1, max_x_change=1e-7,
            trial_quality=copy.deepcopy(row["original_quality"]))
    maximum = attach(prepared(), many, MAX_ORACLE_CALLS - 1)
    maximum.update(status="checked_not_optimal")
    native_failure = copy.deepcopy(no_oracle)
    native_failure.update(status="native_error", phase="presolve")
    native_failure["native"]["simplex_iterations"] = 1
    valid = [empty, bypass, early, shell, pass_partial, pre_partial, no_oracle, original, rejected, accepted,
        empty_accepted, owner_budget, owner_clock, stale_candidate, partial_projection, error_partial,
        error_clock, pending_marker, cumulative_partial, maximum, native_failure]
    for row in valid:
        assert valid_marker(row, row["enabled"]), row

    mutations = []
    def changed(path, wrong, source=accepted):
        row = copy.deepcopy(source); part = row
        for key in path[:-1]: part = part[key]
        part[path[-1]] = wrong
        mutations.append(row)
    for key in accepted:
        row = copy.deepcopy(accepted); row.pop(key); mutations.append(row)
    for key in empty_oracle():
        row = copy.deepcopy(accepted); row["oracle"].pop(key); mutations.append(row)
    for key in accepted["oracle"]["events"][0]:
        row = copy.deepcopy(accepted); row["oracle"]["events"][0].pop(key); mutations.append(row)
    for key in ("schema", "enabled", "attempted", "accepted", "explicit_highs_lp_solve_calls"):
        changed((key,), True if key in ("schema", "explicit_highs_lp_solve_calls") else 1)
    for key in ("status", "phase", "route"): changed((key,), "wrong")
    for key in ("started", "completed", "accepted", "rejected", "budget_failures", "errors"):
        for wrong in (True, -1, 1.0, MAX_ORACLE_CALLS + 1): changed(("oracle", key), wrong)
    for key in ("candidate_event", "selected_event", "projection_event"):
        for wrong in (True, -1, 0.0, MAX_ORACLE_CALLS): changed(("oracle", key), wrong)
    changed(("oracle", "candidate_event"), 0)
    changed(("oracle", "selected_event"), 0)
    changed(("oracle", "projection_event"), 0)
    changed(("oracle", "total_seconds"), 9)
    changed(("oracle", "events", 0, "sequence"), 1)
    changed(("oracle", "events", 0, "iteration"), True)
    changed(("oracle", "events", 0, "iteration"), 201)
    changed(("oracle", "events", 0, "decision"), "accept")
    changed(("oracle", "events", 1, "iteration"), None)
    changed(("oracle", "events", 1, "original_quality"), None)
    changed(("oracle", "events", 1, "projection", "chosen"), "raw")
    changed(("oracle", "events", 1, "timings", "total"), 0)
    changed(("oracle", "events", 1, "timings", "verification"), -1)
    changed(("oracle", "events", 1, "budget", "checks"), True)
    for key in ("postsolve", "assessments"):
        for field in ("started", "completed"):
            for wrong in (True, -1, 1.0, MAX_ORIGINAL_ASSESSMENTS + 1): changed((key, field), wrong)
    changed(("postsolve", "started"), 3)
    changed(("postsolve", "status"), None)
    changed(("assessments", "started"), 2)
    changed(("original_quality", "objective"), 123)
    changed(("projection", "x_changed"), 0)
    changed(("timings", "child"), 0.01)
    changed(("timings", "total"), 0.1)
    changed(("timings", "verification"), 0)
    changed(("timings", "child_verification"), 1)
    changed(("timings", "total"), 5)
    changed(("budget", "checks"), 0)
    for key in ("n", "m", "nnz"):
        for wrong in (True, -1, 1.0): changed(("original", key), wrong)
    for key in QUALITY_BOOLS:
        for wrong in (1, None): changed(("original_quality", key), wrong)
    for key in QUALITY_NUMBERS:
        for wrong in (True, float("nan"), float("inf")): changed(("original_quality", key), wrong)
    for key in NATIVE_ITERATIONS:
        changed(("native", key), True); changed(("native", key), 1)
    changed(("native", "basis_valid"), True)
    changed(("native", "options", "lp_presolve_requires_basis_postsolve"), 1)
    changed(("oracle", "selected_event"), 1, owner_budget)
    changed(("original_quality", "objective"), 2, stale_candidate)
    changed(("oracle", "candidate_event"), 1, stale_candidate)
    changed(("oracle", "projection_event"), 0, partial_projection)
    changed(("assessments", "completed"), 3, cumulative_partial)
    for row in mutations:
        assert not valid_marker(row, True), row
    for row in valid:
        variant = "diagnostic_on" if row["enabled"] else "diagnostic_off"
        parsed, error = inspect_marker(MARKER + json.dumps(row), variant)
        assert strict_same(parsed, row) and error is None
    assert inspect_marker("", "frozen_v66") == (None, None)
    duplicate = json.dumps(bypass).replace('"schema": 1', '"schema": 1, "schema": 1')
    for text in ("", MARKER + "{}", MARKER + duplicate, MARKER + json.dumps(bypass) + "\n" + MARKER + json.dumps(bypass)):
        assert inspect_marker(text, "diagnostic_on")[1] is not None
    assert inspect_marker(MARKER + json.dumps(bypass), "highs")[1] == "unexpected_marker"
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "presolve_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    assert pair_result(dict(row, presolve_marker_error="marker_count"), row, plan)["category"] == "invalid_process"
    changed_report = copy.deepcopy(row); changed_report["report"]["ipm_factorizations"] = True
    assert not valid_process(changed_report)
    return dict(valid_and_parse_cases=len(valid), mutation_cases=len(mutations), sidecar_cases=len(valid) + 6,
        multi_event_cases=12, cumulative_partial_cases=1, owner_selection_cases=5, inclusive_timing_cases=5,
        full_report_and_budget_cases=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v88-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
