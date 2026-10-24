"""第70轮：原始单纯形基修复候选的关闭兼容、开启性能及固定扩集门槛。"""
import argparse
import contextlib
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "d0c4d461e92ab9045c7f51419f9f16393a6440ed1d7e3e8620e43cf440e17587"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
WORK = ROOT / ".tools/v73-basis"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V73_BASIS "
MARKER_COUNTS = ("n", "m", "input_nonzeros", "rank", "qr_q_nonzeros", "qr_r_nonzeros", "lu_l_nonzeros", "lu_u_nonzeros",
    "budget_checks", "qr_analyze_calls", "qr_factor_calls", "lu_analyze_calls", "lu_factor_calls", "primal_solves", "dual_solves",
    "slack_matvecs", "ordered_rows_count", "qr_columns_permutation_count", "selected_rows_count",
    "selection_candidates", "selection_pivots", "selection_rejected", "work_units", "l_nonzeros", "pivot_columns_count")
UNUSED_QR_COUNTS = ("qr_q_nonzeros", "qr_r_nonzeros", "qr_analyze_calls", "qr_factor_calls", "qr_columns_permutation_count")
REPAIR_COUNTS = ("attempts", "iterations", "phase1_iterations", "phase2_iterations", "pivots", "zero_steps",
    "work_units", "budget_checks", "lu_analyze_calls", "lu_factor_calls", "factor_nonzeros", "primal_solves", "dual_solves",
    "direction_solves", "slack_matvecs", "direction_matvecs", "pricing_matvecs", "roundoff_rows", "final_selected_rows_count")
REPAIR_NUMBERS = ("initial_phi", "final_phi", "initial_max_violation", "final_max_violation", "max_roundoff_bound",
    "last_right_derivative", "last_step", "max_selected_residual", "max_direction_residual")
REPAIR_PHASES = {"not_started", "validation", "initial_residual", "phase1", "phase2", "complete"}
REPAIR_STATUSES = {"not_started", "budget_exception", "exception", "invalid_input", "work_limit", "factor_limit", "arithmetic_failure",
    "basis_residual_failure", "no_descent", "no_blocker", "no_entering", "pivot_limit", "solve_failure",
    "direction_residual_failure", "zero_step_stall", "weak_pivot", "lu_failure", "lu_nonfinite",
    "point_consistency_failure", "objective_not_improved", "primal_guard_failure", "candidate"}
RECOVERY_STATUSES = {"not_started", "budget_exception", "invalid_input", "input_limit", "invalid_interior_point",
    "candidate", "structural_rank_deficient", "selection_work_limit", "selection_norm_nonfinite", "selection_threshold_nonfinite",
    "selection_threshold_underflow", "selection_product_nonfinite", "selection_product_underflow", "selection_update_nonfinite",
    "selection_update_underflow", "selection_factor_limit", "selection_division_nonfinite", "selection_division_underflow",
    "selection_rank_deficient", "lu_failure", "lu_factor_limit", "lu_nonfinite",
    "primal_solve_failure", "dual_solve_failure", "slack_nonfinite", "repair_unavailable", "repair_invalid_candidate"}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_once(path, value):
    if path.exists():
        assert read(path) == value, "禁止覆盖不一致证据"
    else:
        save(path, value)


assert sha(DATA / "v66-run.py") == HELPER_SHA
_spec = importlib.util.spec_from_file_location("v73_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def fixed_plan():
    assert sha(DATA / "v73-plan.json") == PLAN_SHA
    plan = read(DATA / "v73-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_identity_file"], plan["baseline_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert plan["candidate_version"] == 73 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V73_BASIS_RECOVERY"
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"]) == (5, 35, 200, 1, 0)
    for side in ("candidate", "reference", "compatibility"):
        assert plan[side + "_accuracy"] == "homogeneous_repair"
        assert plan[side + "_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert plan[side + "_backend_version"] == "ipm-cpu-0.53.0"
        assert plan[side + "_objective_scaling"] == "off"
    assert plan["method"] == "homogeneous" and plan["stage_order"] == list(STAGES)
    assert plan["maximum_solver_runs"] == 500
    assert plan["candidate_limits"] == {"max_input_nonzeros": 8000000, "max_factor_nonzeros": 32000000,
        "max_dimension": 1000000, "max_selection_work": 1000000000,
        "max_repair_work": 100000000, "max_repair_pivots": 128}
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
    clean = {key: value for key, value in prior.items() if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
    clean.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    if enabled:
        clean["SOLVER_V73_BASIS_RECOVERY"] = "1"
    try:
        os.environ.clear()
        os.environ.update(clean)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def inspect_marker(stderr, variant):
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if variant in ("frozen_v66", "highs"):
        return (None, None) if not lines else (None, "unexpected_marker")
    if len(lines) != 1:
        return None, "marker_count"
    try:
        value = benchmark.strict_json(lines[0][len(MARKER):])
    except (TypeError, ValueError):
        return None, "invalid_marker_json"
    if not valid_marker(value, variant == "diagnostic_on"):
        return value, "invalid_marker_schema"
    return value, None


def valid_repair(value, initial):
    if not isinstance(value, dict) or value.keys() != {"status", "candidate_available", "stats"} \
        or type(value["candidate_available"]) is not bool or not isinstance(value["status"], str) \
        or value["status"] not in REPAIR_STATUSES:
        return False
    stats = value["stats"]
    if not isinstance(stats, dict) or stats.keys() != set(REPAIR_COUNTS + REPAIR_NUMBERS) | {"phase"}:
        return False
    if any(type(stats[key]) is not int or not 0 <= stats[key] <= 9223372036854775807 for key in REPAIR_COUNTS):
        return False
    if any(stats[key] is not None and (not benchmark.finite_real(stats[key])
        or (key != "last_right_derivative" and stats[key] < 0)) for key in REPAIR_NUMBERS):
        return False
    if not isinstance(stats["phase"], str) or stats["phase"] not in REPAIR_PHASES:
        return False
    if stats["attempts"] > 1 or stats["pivots"] > 128 or stats["work_units"] > 100000000 \
        or stats["final_selected_rows_count"] > initial["n"]:
        return False
    if stats["iterations"] != stats["phase1_iterations"] + stats["phase2_iterations"] \
        or not (0 <= stats["iterations"] - stats["pivots"] <= 1 and stats["iterations"] <= 128) \
        or not (stats["primal_solves"] <= stats["lu_factor_calls"] <= stats["lu_analyze_calls"]
            <= stats["direction_solves"] <= stats["iterations"]) \
        or stats["direction_matvecs"] > stats["direction_solves"] or stats["zero_steps"] > 1:
        return False
    if value["status"] == "not_started":
        return not value["candidate_available"] and stats["phase"] == "not_started" \
            and all(stats[key] == 0 for key in REPAIR_COUNTS) and all(stats[key] is None for key in REPAIR_NUMBERS)
    if stats["attempts"] != 1 or not (initial["primal_solves"] == initial["dual_solves"] == 1):
        return False
    if value["candidate_available"]:
        return value["status"] == "candidate" and stats["phase"] == "complete" \
            and stats["final_selected_rows_count"] == initial["n"] \
            and stats["iterations"] == stats["pivots"] \
            and initial["lu_l_nonzeros"] + initial["lu_u_nonzeros"] <= stats["factor_nonzeros"] <= 32000000
    return value["status"] != "candidate"


def valid_marker(value, enabled):
    fields = {"schema", "enabled", "attempted", "accepted", "stage", "candidate_available", "attempt_seconds",
        "verification_seconds", "recovery_status", "stats", "additional_work_in_legacy_ipm_counts", "work_in_total_seconds",
        "marker_in_total_seconds", "trial_verification", "repair"}
    if not isinstance(value, dict) or value.keys() != fields or type(value["schema"]) is not int or value["schema"] != 1:
        return False
    if any(type(value[key]) is not bool for key in ("enabled", "attempted", "accepted", "candidate_available")) \
        or value["enabled"] is not enabled or value["additional_work_in_legacy_ipm_counts"] is not False \
        or value["work_in_total_seconds"] is not True or value["marker_in_total_seconds"] is not False:
        return False
    if not all(benchmark.finite_real(value[key]) and value[key] >= 0 for key in ("attempt_seconds", "verification_seconds")):
        return False
    if not isinstance(value["recovery_status"], str) or value["recovery_status"] not in RECOVERY_STATUSES:
        return False
    if not isinstance(value["stage"], str):
        return False
    stats = value["stats"]
    if not isinstance(stats, dict) or stats.keys() != set(MARKER_COUNTS) | {"qr_default_threshold", "min_abs_pivot", "selection_threshold"}:
        return False
    if any(type(stats[key]) is not int or not 0 <= stats[key] <= 9223372036854775807 for key in MARKER_COUNTS):
        return False
    threshold = stats["qr_default_threshold"]
    if not benchmark.finite_real(threshold) or threshold != 0 or any(stats[key] != 0 for key in UNUSED_QR_COUNTS):
        return False
    if any(not benchmark.finite_real(stats[key]) or stats[key] < 0
        for key in ("min_abs_pivot", "selection_threshold")):
        return False
    if stats["rank"] > min(stats["n"], stats["m"]) or stats["selected_rows_count"] > stats["n"] \
        or stats["ordered_rows_count"] > stats["m"] or stats["qr_columns_permutation_count"] > stats["m"]:
        return False
    if not (stats["rank"] == stats["selection_pivots"] == stats["selected_rows_count"] == stats["pivot_columns_count"]
        and stats["selection_candidates"] <= stats["m"]
        and 0 <= stats["selection_candidates"] - stats["selection_pivots"] - stats["selection_rejected"] <= 1
        and stats["work_units"] <= 1000000000):
        return False
    if (stats["selection_pivots"] == 0) is not (stats["min_abs_pivot"] == 0):
        return False
    if not (stats["lu_factor_calls"] <= stats["lu_analyze_calls"] <= 1
        and stats["slack_matvecs"] <= stats["dual_solves"] <= stats["primal_solves"] <= stats["lu_factor_calls"]):
        return False
    if not valid_repair(value["repair"], stats):
        return False
    trial = value["trial_verification"]
    bools = ("primal_valid", "primal_feasible", "dual_feasible", "gap_satisfied", "numerically_optimal")
    numbers = ("max_row_violation", "max_bound_violation", "max_dual_residual", "objective_gap")
    if trial is not None and (not isinstance(trial, dict) or trial.keys() != set(bools + numbers)
        or any(type(trial[key]) is not bool for key in bools)
        or any(trial[key] is not None and not benchmark.finite_real(trial[key]) for key in numbers)):
        return False
    if not value["attempted"]:
        return value["stage"] == "not_attempted" and value["recovery_status"] == "not_started" \
            and not value["accepted"] and not value["candidate_available"] and trial is None \
            and value["attempt_seconds"] == value["verification_seconds"] == 0 \
            and all(stats[key] == 0 for key in MARKER_COUNTS) and threshold == 0 \
            and stats["min_abs_pivot"] == stats["selection_threshold"] == 0 and value["repair"]["status"] == "not_started"
    if not enabled or value["stage"] not in {"accepted", "original_verification_rejected", "candidate_unavailable", "time_limit", "recovery_exception"}:
        return False
    if value["accepted"] is not (value["stage"] == "accepted"):
        return False
    if value["candidate_available"]:
        if value["recovery_status"] != "candidate":
            return False
        if (stats["n"] > 0 and not value["repair"]["candidate_available"]) \
            or (stats["n"] == 0 and value["repair"]["status"] != "not_started"):
            return False
        if stats["n"] and not (stats["rank"] == stats["n"] == stats["selected_rows_count"]
            and stats["ordered_rows_count"] == stats["m"]
            and stats["selection_candidates"] == stats["selection_pivots"] + stats["selection_rejected"]
            and benchmark.finite_real(stats["min_abs_pivot"]) and stats["min_abs_pivot"] > 0
            and benchmark.finite_real(stats["selection_threshold"]) and stats["selection_threshold"] > 0
            and all(stats[key] == 1 for key in ("lu_analyze_calls", "lu_factor_calls",
                "primal_solves", "dual_solves", "slack_matvecs"))):
            return False
    elif value["recovery_status"] == "candidate" or trial is not None:
        return False
    if value["stage"] in ("accepted", "original_verification_rejected") and (not value["candidate_available"] or trial is None):
        return False
    if value["stage"] == "candidate_unavailable" and value["candidate_available"]:
        return False
    if value["accepted"] and (any(trial[key] is not True for key in bools) or any(trial[key] is None for key in numbers)):
        return False
    return True


def valid_process(row):
    if row.get("basis_marker_error") is not None or row.get("configuration_matches") is not True:
        return False
    if not base.valid_process(row):
        return False
    report = row["report"]
    return report["backend"] != "ipm" or benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report)


def configuration_matches(report, stage, variant, model, plan):
    side = "reference" if variant == variants(stage)[0] else "candidate"
    backend = "highs" if variant == "highs" else "ipm"
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == backend \
        and report.get("backend_version") == ("1.15.1" if backend == "highs" else "ipm-cpu-0.53.0") \
        and base.typed_equal(report.get("options"), base.expected_options(stage, side, plan))


def basis_sidecar_compatible(a, b):
    left, right = a.get("basis_marker"), b.get("basis_marker")
    if left is None and right is None:
        return True, None
    if left is None or right is None:
        # v66没有侧账本；只可与关闭候选的规范零事件账本对应。
        return valid_marker(right if left is None else left, False), None
    if not isinstance(left, dict) or not isinstance(right, dict) \
        or not valid_marker(left, left.get("enabled")) or not valid_marker(right, right.get("enabled")):
        return False, False
    excluded = {"attempt_seconds", "verification_seconds"}
    equal = base.typed_equal({key: item for key, item in left.items() if key not in excluded},
        {key: item for key, item in right.items() if key not in excluded})
    return equal, equal


def pair_result(a, b, plan):
    answer = base.pair_result(a, b, plan)
    compatible, equal = basis_sidecar_compatible(a, b)
    answer.update(report_and_exit_equal=answer["equal"], equal=answer["equal"] and compatible,
        basis_sidecar_compatible=compatible, basis_sidecar_equal_excluding_durations=equal)
    if not valid_process(a) or not valid_process(b):
        answer.update(valid_processes=False, category="invalid_process")
    elif answer["category"] == "equal" and not compatible:
        answer["category"] = "different"
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


def freeze(work, plan):
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["parent_identity_sha256"] == plan["parent_identity_sha256"]
    assert build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    assert Path(build["frozen_parent_executable"]) == PARENT
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-diagnostic.exe"
    protected = {str(Path(__file__)): sha(__file__), str(DATA / "v66-run.py"): HELPER_SHA,
        str(build_path): sha(build_path), str(PARENT): plan["parent_solver_sha256"],
        str(candidate): build["executable_sha256"],
        str(ROOT / "scripts/benchmark.py"): sha(ROOT / "scripts/benchmark.py"),
        str(ROOT / "scripts/compare_lp.py"): sha(ROOT / "scripts/compare_lp.py")}
    for name in ("v73-basis-recovery.hpp", "v73-basis-repair.hpp", "v73-basis-instrumentation.py",
        "v73-basis-self-test.cpp", "v73-repair-self-test.cpp", "v73-build.py", "v73-run.py"):
        path = str(DATA / name)
        digest = build["protected_before_and_after"][path]
        assert path not in protected or protected[path] == digest
        protected[path] = digest
    for name in ("static-check.json", "basis-self-check.json", "repair-self-check.json"):
        path = str(work / name)
        protected[path] = build["outputs"][path]
        assert read(path)["passed"] is True
    # 构建已冻结库；本入口只核执行来源及工具，完整库身份由独立终验统一核对。
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v73-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v73-{stage}-manifest.json")
    summary = read(DATA / f"v73-{stage}-summary.json")
    raw = DATA / f"v73-{stage}-runs.jsonl.gz"
    rows = [benchmark.strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v73-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v73-run-identity.json")
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
        assert row["basis_environment"] == ({"SOLVER_V73_BASIS_RECOVERY": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert row["basis_marker"] == marker and row["basis_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert row["report"] == benchmark.strict_json(row["stdout"])
        for field, value in quality_fields(row, model, digest, plan).items():
            assert row[field] == value, (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), basis_marker_error_count=sum(row["basis_marker_error"] is not None for row in rows))
    assert computed == summary
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
        save_once(DATA / "v73-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v73-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v73-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v73-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V73_BASIS_RECOVERY=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；额外恢复计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v73-{stage}-manifest.json", manifest)
    raw = DATA / f"v73-{stage}-runs.jsonl.gz"
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
                basis_environment={"SOLVER_V73_BASIS_RECOVERY": "1"} if variant == "diagnostic_on" else {},
                basis_marker=marker, basis_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "basis_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), basis_marker_error_count=sum(row["basis_marker_error"] is not None for row in rows))
    save(DATA / f"v73-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"], "valid_process_count": summary["valid_process_count"]}, ensure_ascii=False), flush=True)


def self_check():
    plan = fixed_plan()
    inherited = base.validation_self_check()
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    before = dict(os.environ)
    with solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V73_BASIS_RECOVERY": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V73_BASIS_RECOVERY"] == "1"
    assert dict(os.environ) == before
    for stage in STAGES:
        for variant in variants(stage):
            value = options_for(stage, variant, PARENT, plan)
            assert value.time_limit == 5.0 and value.wall_timeout == 35 and value.threads == 1
            if variant == "highs":
                assert value.backend == "highs" and value.highs_solver == "choose"
            else:
                assert value.ipm_linear_solver == "normal_cholmod_aat" and value.ipm_linear_accuracy == "homogeneous_repair"
                assert value.ipm_objective_scaling == "off" and value.ipm_direction_retries == 0
    return {"passed": True, "solver_runs": 0, "inherited_count_and_budget_checks": inherited,
        "environment_cases": 4, "configuration_cases": 12, "marker_cases": marker_self_check()}


def marker_self_check():
    idle_repair = {"status": "not_started", "candidate_available": False,
        "stats": {**dict.fromkeys(REPAIR_COUNTS, 0), **dict.fromkeys(REPAIR_NUMBERS), "phase": "not_started"}}
    empty = {"schema": 1, "enabled": False, "attempted": False, "accepted": False, "stage": "not_attempted",
        "candidate_available": False, "attempt_seconds": 0, "verification_seconds": 0, "recovery_status": "not_started",
        "stats": {**dict.fromkeys(MARKER_COUNTS, 0), "qr_default_threshold": 0, "min_abs_pivot": 0, "selection_threshold": 0}, "trial_verification": None,
        "repair": copy.deepcopy(idle_repair),
        "additional_work_in_legacy_ipm_counts": False, "work_in_total_seconds": True, "marker_in_total_seconds": False}
    assert valid_marker(empty, False)
    enabled = dict(empty, enabled=True)
    assert valid_marker(enabled, True)
    assert inspect_marker("", "frozen_v66") == (None, None)
    assert inspect_marker(MARKER + json.dumps(empty), "diagnostic_off") == (empty, None)
    assert inspect_marker(MARKER + json.dumps(enabled), "diagnostic_on") == (enabled, None)
    assert inspect_marker(MARKER + json.dumps(empty), "frozen_v66")[1] == "unexpected_marker"
    assert inspect_marker("", "diagnostic_on")[1] == "marker_count"
    assert inspect_marker(MARKER + "{}\n" + MARKER + "{}", "diagnostic_on")[1] == "marker_count"
    assert inspect_marker(MARKER + "{", "diagnostic_on")[1] == "invalid_marker_json"
    candidate = copy.deepcopy(enabled)
    candidate.update(attempted=True, accepted=True, stage="accepted", candidate_available=True, recovery_status="candidate",
        attempt_seconds=0.2, verification_seconds=0.01, trial_verification={"primal_valid": True, "primal_feasible": True,
            "dual_feasible": True, "gap_satisfied": True, "numerically_optimal": True, "max_row_violation": 0,
            "max_bound_violation": 0, "max_dual_residual": 0, "objective_gap": 0})
    candidate["stats"].update(n=1, m=2, input_nonzeros=2, rank=1,
        lu_l_nonzeros=1, lu_u_nonzeros=1, budget_checks=9,
        lu_analyze_calls=1, lu_factor_calls=1, primal_solves=1, dual_solves=1, slack_matvecs=1,
        ordered_rows_count=2, selected_rows_count=1, selection_candidates=1, selection_pivots=1,
        pivot_columns_count=1, work_units=5, l_nonzeros=1, min_abs_pivot=1, selection_threshold=1e-12)
    candidate["repair"].update(status="candidate", candidate_available=True)
    candidate["repair"]["stats"].update(phase="complete", attempts=1, slack_matvecs=1,
        budget_checks=4, factor_nonzeros=2, final_selected_rows_count=1,
        initial_phi=0, final_phi=0, initial_max_violation=0, final_max_violation=0,
        max_roundoff_bound=1e-12, max_selected_residual=0)
    assert valid_marker(candidate, True)
    interrupted_repair = copy.deepcopy(candidate)
    interrupted_repair.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    interrupted_repair["stats"]["slack_matvecs"] = 0
    interrupted_repair["repair"] = copy.deepcopy(idle_repair)
    interrupted_repair["repair"].update(status="budget_exception")
    interrupted_repair["repair"]["stats"].update(phase="validation", attempts=1, budget_checks=1)
    assert valid_marker(interrupted_repair, True)
    limited_repair = copy.deepcopy(interrupted_repair)
    limited_repair.update(stage="candidate_unavailable", recovery_status="repair_unavailable")
    limited_repair["repair"]["status"] = "work_limit"
    limited_repair["repair"]["stats"].update(phase="phase1", work_units=100000000,
        pricing_matvecs=1, slack_matvecs=1, factor_nonzeros=2, final_selected_rows_count=1,
        initial_phi=1, final_phi=1, initial_max_violation=1, final_max_violation=1)
    assert valid_marker(limited_repair, True)
    factor_repair = copy.deepcopy(limited_repair)
    factor_repair["repair"]["status"] = "factor_limit"
    factor_repair["repair"]["stats"].update(work_units=100, iterations=1, phase1_iterations=1,
        dual_solves=1, direction_solves=1, direction_matvecs=1, lu_analyze_calls=1, lu_factor_calls=1,
        factor_nonzeros=32000001, last_right_derivative=-1, last_step=1)
    assert valid_marker(factor_repair, True)
    committed_repair = copy.deepcopy(factor_repair)
    committed_repair.update(stage="time_limit", recovery_status="budget_exception")
    committed_repair["repair"]["status"] = "budget_exception"
    committed_repair["repair"]["stats"].update(pivots=1, primal_solves=1,
        slack_matvecs=2, factor_nonzeros=4, final_phi=0, final_max_violation=0)
    assert valid_marker(committed_repair, True)
    exception_repair = copy.deepcopy(interrupted_repair)
    exception_repair.update(stage="recovery_exception", recovery_status="invalid_input")
    exception_repair["repair"]["status"] = "exception"
    assert valid_marker(exception_repair, True)
    rejected = copy.deepcopy(candidate)
    rejected.update(accepted=False, stage="original_verification_rejected")
    rejected["trial_verification"]["dual_feasible"] = False
    assert valid_marker(rejected, True)
    expired = copy.deepcopy(enabled)
    expired.update(attempted=True, stage="time_limit", attempt_seconds=0.1)
    assert valid_marker(expired, True)
    partial = copy.deepcopy(expired)
    partial.update(recovery_status="budget_exception")
    partial["stats"].update(n=3, m=4, input_nonzeros=8, budget_checks=2, rank=1, selection_candidates=2,
        selection_pivots=1, selected_rows_count=1, pivot_columns_count=1, ordered_rows_count=4,
        work_units=1000000000, l_nonzeros=5, min_abs_pivot=1, selection_threshold=1e-12)
    assert valid_marker(partial, True)
    rejected_column = copy.deepcopy(partial)
    rejected_column["stats"].update(selection_candidates=3, selection_rejected=1)
    assert valid_marker(rejected_column, True)
    work_limit = copy.deepcopy(partial)
    work_limit.update(stage="candidate_unavailable", recovery_status="selection_work_limit")
    assert valid_marker(work_limit, True)
    arithmetic_failure = copy.deepcopy(partial)
    arithmetic_failure.update(stage="candidate_unavailable", recovery_status="selection_update_underflow")
    assert valid_marker(arithmetic_failure, True)
    mutations = [(("enabled",), False), (("accepted",), False), (("stage",), "verifying"),
        (("stage",), []), (("recovery_status",), []),
        (("candidate_available",), False), (("schema",), True), (("attempt_seconds",), -1),
        (("verification_seconds",), float("nan")), (("work_in_total_seconds",), False),
        (("stats", "qr_factor_calls"), 2), (("stats", "rank"), 2), (("stats", "selected_rows_count"), 0),
        (("trial_verification", "primal_feasible"), False), (("trial_verification", "objective_gap"), None),
        (("stats", "work_units"), 1000000001), (("stats", "pivot_columns_count"), 0),
        (("stats", "selection_candidates"), 2), (("stats", "selection_rejected"), 1),
        (("stats", "min_abs_pivot"), 0), (("stats", "min_abs_pivot"), None),
        (("stats", "selection_threshold"), 0), (("stats", "selection_threshold"), None),
        (("stats", "selection_threshold"), float("nan")),
        (("repair", "status"), "unknown"), (("repair", "candidate_available"), False),
        (("repair", "candidate_available"), 1), (("repair", "stats", "phase"), "phase2"),
        (("repair", "stats", "attempts"), 2), (("repair", "stats", "work_units"), 100000001),
        (("repair", "stats", "pivots"), 129), (("repair", "stats", "phase1_iterations"), 1),
        (("repair", "stats", "lu_factor_calls"), 1), (("repair", "stats", "final_selected_rows_count"), 0),
        (("repair", "stats", "factor_nonzeros"), 32000001), (("repair", "stats", "zero_steps"), 2),
        (("repair", "stats", "final_phi"), -1), (("repair", "stats", "last_right_derivative"), float("nan"))]
    for key in UNUSED_QR_COUNTS:
        mutations.append((("stats", key), 1))
    mutations.append((("stats", "qr_default_threshold"), 1e-12))
    for key in MARKER_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            mutations.append((("stats", key), bad))
    for key in REPAIR_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            mutations.append((("repair", "stats", key), bad))
    for path, bad in mutations:
        value = copy.deepcopy(candidate)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        assert not valid_marker(value, True), path
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "basis_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    off_row = dict(row, basis_marker=empty)
    assert pair_result(row, off_row, plan)["category"] == "equal" and pair_result(row, off_row, plan)["basis_sidecar_compatible"]
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["stats"]["budget_checks"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["attempted"] = True
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["budget_checks"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    active_row = dict(row, basis_marker=candidate)
    duration_changed = copy.deepcopy(active_row)
    duration_changed["basis_marker"]["attempt_seconds"] += 0.1
    assert pair_result(active_row, duration_changed, plan)["category"] == "equal"
    assert pair_result(active_row, dict(row, basis_marker=rejected), plan)["category"] == "different"
    repair_changed = copy.deepcopy(active_row)
    repair_changed["basis_marker"]["repair"]["stats"]["work_units"] += 1
    assert pair_result(active_row, repair_changed, plan)["category"] == "different"
    assert pair_result(active_row, repair_changed, plan)["equal"] is False
    changed = copy.deepcopy(row)
    changed["report"]["backend_version"] = "ipm-cpu-other"
    assert pair_result(row, changed, plan)["category"] == "different"
    budget = copy.deepcopy(row)
    budget["status"] = budget["report"]["status"] = "time_limit"
    invalid = dict(row, basis_marker_error="marker_count")
    assert pair_result(invalid, budget, plan)["category"] == "invalid_process"
    for status, code, classification in (("numerical_error", 4, "incomplete"), ("time_limit", 4, "incomplete"),
        ("iteration_limit", 4, "incomplete"), ("verification_failed", 3, "verification_failed")):
        altered = copy.deepcopy(row)
        altered.update(status=status, returncode=code, classification=classification)
        altered["report"]["status"] = status
        assert valid_process(altered)
        altered["report"]["ipm_factorizations"] = True
        assert pair_result(altered, budget, plan)["category"] == "invalid_process"
    return {"valid_and_parse_cases": 21, "mutation_cases": len(mutations), "full_report_and_budget_cases": 7, "sidecar_cases": 7}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v73-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
