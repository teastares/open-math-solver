"""第81轮：已提交点PDHG恢复候选的关闭兼容、开启性能及固定扩集门槛。"""
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
PLAN_SHA = "4dfec47350f3b208602f0ec36f5fda8a2888d68f630b7c48620f72c5dd86b6aa"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
WORK = ROOT / ".tools/v84-warm"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V84_WARM "
ENVIRONMENT = "SOLVER_V84_WARM_PDHG"
WARM_LIMITS = dict(max_iterations=20000, max_work_units=500000000, check_interval=64,
    budget_check_work_interval=256, step_safety=0.9)
STATS_COUNTS = ("setup_started", "setup_completed", "steps_started", "steps_completed",
    "transpose_started", "transpose_completed", "multiply_started", "multiply_completed",
    "budget_checks", "assessments_started", "assessments_completed", "work_units")
QUALITY_BOOLS = ("primal_valid", "primal_feasible", "objective_consistent",
    "dual_feasible", "gap_satisfied", "numerically_optimal")
QUALITY_NUMBERS = ("objective", "max_row_violation", "max_bound_violation", "max_dual_residual",
    "objective_gap", "relative_objective_gap")
KERNEL_STATUSES = {"not_started", "running", "invalid_input", "arithmetic_failure", "work_limit",
    "iteration_limit", "accepted", "budget_exception", "assessment_exception", "exception"}
PHASES = {"not_started", "validation", "seed", "assessment", "setup", "transpose", "primal",
    "multiply", "dual", "commit", "finished"}
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
_spec = importlib.util.spec_from_file_location("v84_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def strict_limits(value):
    return isinstance(value, dict) and value.keys() == WARM_LIMITS.keys() and all(
        type(value[key]) is type(expected) and value[key] == expected for key, expected in WARM_LIMITS.items())


def fixed_plan():
    assert sha(DATA / "v84-plan.json") == PLAN_SHA
    plan = read(DATA / "v84-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_identity_file"], plan["baseline_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert type(plan["candidate_version"]) is int and plan["candidate_version"] == 84 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V84_WARM_PDHG"
    assert plan["diagnostic_marker"] == "V84_WARM"
    assert strict_limits(plan["warm_limits"])
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


def valid_quality(value, steps, seed=False):
    if value is None:
        return True
    if not isinstance(value, dict) or set(value) != {"completed_steps", *QUALITY_BOOLS, *QUALITY_NUMBERS}:
        return False
    if not integer(value["completed_steps"], steps) or seed and value["completed_steps"] != 0:
        return False
    if any(type(value[key]) is not bool for key in QUALITY_BOOLS[:3]):
        return False
    if any(value[key] is not None and type(value[key]) is not bool for key in QUALITY_BOOLS[3:]):
        return False
    if any(not nullable_number(value[key]) for key in QUALITY_NUMBERS):
        return False
    return all(value[key] is None or value[key] >= 0 for key in
        ("max_row_violation", "max_bound_violation", "max_dual_residual"))


def valid_kernel_stats(stats):
    if not isinstance(stats, dict) or set(stats) != {"phase", "seed_available", "last_assessed_step", *STATS_COUNTS}:
        return False
    if type(stats["phase"]) is not str or stats["phase"] not in PHASES or type(stats["seed_available"]) is not bool:
        return False
    if any(not integer(stats[key]) for key in STATS_COUNTS):
        return False
    if stats["work_units"] > WARM_LIMITS["max_work_units"] or stats["setup_started"] > 1 or stats["steps_started"] > WARM_LIMITS["max_iterations"]:
        return False
    for stem in ("setup", "steps", "transpose", "multiply", "assessments"):
        if not 0 <= stats[stem + "_started"] - stats[stem + "_completed"] <= 1:
            return False
    chain = ("steps_completed", "multiply_completed", "multiply_started", "transpose_completed", "transpose_started", "steps_started")
    if any(stats[a] > stats[b] for a, b in zip(chain, chain[1:])):
        return False
    assessed = stats["last_assessed_step"]
    if stats["assessments_completed"] == 0:
        if assessed is not None:
            return False
    elif not integer(assessed, stats["steps_completed"]):
        return False
    if not stats["seed_available"] and any(stats[key] != 0 for key in STATS_COUNTS if key not in ("budget_checks", "work_units")):
        return False
    if stats["steps_started"] and stats["setup_completed"] != 1:
        return False
    return not stats["setup_started"] or stats["seed_available"] and stats["assessments_completed"] > 0


def valid_marker(value, enabled):
    fields = {"schema", "enabled", "attempted", "accepted", "status", "tau", "kappa", "kernel_status",
        "kernel_stats", "seed_quality", "last_quality"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema"]) is not int or value["schema"] != 1:
        return False
    if type(value["enabled"]) is not bool or value["enabled"] is not enabled or any(type(value[key]) is not bool for key in ("attempted", "accepted")):
        return False
    if type(value["kernel_status"]) is not str or value["kernel_status"] not in KERNEL_STATUSES or \
        type(value["status"]) is not str or value["status"] not in KERNEL_STATUSES | {"disabled", "not_triggered"}:
        return False
    if not all(nullable_number(value[key]) for key in ("tau", "kappa")) or not valid_kernel_stats(value["kernel_stats"]):
        return False
    stats = value["kernel_stats"]
    if not valid_quality(value["seed_quality"], stats["steps_completed"], True) or not valid_quality(value["last_quality"], stats["steps_completed"]):
        return False
    if not value["attempted"]:
        return value["accepted"] is False and value["status"] == ("not_triggered" if enabled else "disabled") and \
            value["kernel_status"] == "not_started" and value["tau"] is None and value["kappa"] is None and \
            value["seed_quality"] is None and value["last_quality"] is None and \
            stats["phase"] == "not_started" and not stats["seed_available"] and all(stats[key] == 0 for key in STATS_COUNTS)
    if not enabled or value["status"] in {"disabled", "not_triggered", "not_started"}:
        return False
    if value["kernel_status"] == "not_started" and (stats["phase"] != "not_started" or stats["seed_available"] or any(stats[key] for key in STATS_COUNTS)):
        return False
    if not stats["seed_available"] and (value["seed_quality"] is not None or value["last_quality"] is not None):
        return False
    if (value["seed_quality"] is None) != (value["last_quality"] is None):
        return False
    if stats["assessments_completed"] and value["last_quality"] is None:
        return False
    # 回调可完成原assess并记录质量后，才在内核post-budget处抛出；不能据此误判已提交。
    if value["last_quality"] is not None and (not stats["seed_available"] or stats["assessments_started"] == 0 or
        stats["last_assessed_step"] is not None and value["last_quality"]["completed_steps"] < stats["last_assessed_step"]):
        return False
    if value["accepted"]:
        return value["status"] == value["kernel_status"] == "accepted" and stats["seed_available"] and \
            stats["phase"] == "finished" and stats["assessments_completed"] > 0 and \
            stats["last_assessed_step"] == stats["steps_completed"] and value["last_quality"] is not None and \
            value["last_quality"]["completed_steps"] == stats["steps_completed"] and \
            all(value["last_quality"][key] is True for key in QUALITY_BOOLS)
    return value["status"] != "accepted" and (value["kernel_status"] != "accepted" or value["status"] in {"budget_exception", "exception"})


def valid_process(row):
    if row.get("warm_marker_error") is not None or row.get("configuration_matches") is not True:
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


def valid_math_evidence(role, checks, raw, work):
    if role not in ("warm", "ipm") or not isinstance(checks, dict) or checks.get("passed") is not True or \
        type(checks.get("lp_runs")) is not int or checks["lp_runs"] != 0 or not isinstance(raw, dict):
        return False
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("wall_timeout") is not False \
        or raw.get("command") != [str(work / (role + "-self-test.exe"))] or raw.get("cwd") != str(ROOT):
        return False
    if not isinstance(raw.get("stdout"), str) or raw.get("stderr") != "":
        return False
    try:
        value = strict_json(raw["stdout"])
    except (ValueError, TypeError):
        return False
    if not isinstance(value, dict) or value.get("passed") is not True or not base.typed_equal(value, checks.get("details")):
        return False
    if type(value.get("cases")) is not int or value["cases"] <= 0:
        return False
    if role == "warm":
        return type(value.get("public_lp_runs")) is int and value["public_lp_runs"] == 0 and \
            isinstance(value.get("details"), list) and len(value["details"]) == value["cases"] and \
            all(isinstance(label, str) and label for label in value["details"]) and \
            strict_limits(value.get("default_limits"))
    return raw["stderr"] == "" and type(value.get("solver_runs")) is int and value["solver_runs"] == 0 and \
        all(type(value.get(key)) is int and value[key] > 0 for key in
            ("mapping_cases", "isolation_cases", "verification_accounting_cases", "budget_cases", "exception_cases"))


def math_evidence_self_check():
    for role in ("warm", "ipm"):
        value = {"passed": True, "cases": 1}
        if role == "warm":
            value.update(public_lp_runs=0, details=["原方程"], default_limits=copy.deepcopy(WARM_LIMITS))
        else:
            value.update(solver_runs=0, mapping_cases=1, isolation_cases=1,
                verification_accounting_cases=1, budget_cases=1, exception_cases=1)
        checks = {"passed": True, "lp_runs": 0, "details": value}
        raw = {"command": [str(WORK / (role + "-self-test.exe"))], "cwd": str(ROOT),
            "returncode": 0, "wall_timeout": False, "stdout": json.dumps(value), "stderr": ""}
        assert valid_math_evidence(role, checks, raw, WORK)
        for key, invalid in (("returncode", False), ("returncode", 4), ("wall_timeout", True),
            ("stdout", "{}"), ("stdout", '{"passed":true,"passed":true}'), ("command", []), ("cwd", "wrong"), ("stderr", "污染")):
            changed = dict(raw); changed[key] = invalid
            assert not valid_math_evidence(role, checks, changed, WORK)
        changed = copy.deepcopy(checks); changed["details"]["cases"] = True
        assert not valid_math_evidence(role, changed, raw, WORK)
        bad = copy.deepcopy(value); bad["public_lp_runs" if role == "warm" else "budget_cases"] = 1 if role == "warm" else 0
        assert not valid_math_evidence(role, dict(checks, details=bad), dict(raw, stdout=json.dumps(bad)), WORK)
        if role == "warm":
            for key, invalid in (("max_iterations", True), ("max_iterations", 20000.0), ("step_safety", True), ("max_work_units", 0)):
                bad = copy.deepcopy(value); bad["default_limits"][key] = invalid
                assert not valid_math_evidence(role, dict(checks, details=bad), dict(raw, stdout=json.dumps(bad)), WORK)


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
    assert build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    assert build["production_changes"] is False
    assert Path(build["frozen_parent_executable"]) == PARENT
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-warm.exe"
    assert set(build["commands"]["compile"]) == {"ipm"}
    assert build["commands"]["link"].count(str(work / "ipm-warm.obj")) == 1
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
    for name in ("static-check.json", "warm-self-check.json", "ipm-self-check.json"):
        path = str(work / name)
        assert path in protected and read(path)["passed"] is True
    for role in ("warm", "ipm"):
        assert valid_math_evidence(role, read(work / (role + "-self-check.json")),
            read(work / (role + "-self-run.json")), work), "数学摘要与实际执行原件不同"
    for key, filename in (("private_state_header", "v84-warm-state.hpp"), ("private_kernel_header", "v84-warm-pdhg.hpp")):
        path = str(work / filename)
        assert build[key] == path and path in build["generated_inputs_before_and_after"] and path in build["compile_inputs_after_and_final"]
    assert sha(work / "v84-warm-pdhg.hpp") == sha(DATA / "v84-warm-pdhg.hpp")
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v84-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v84-{stage}-manifest.json")
    summary = read(DATA / f"v84-{stage}-summary.json")
    raw = DATA / f"v84-{stage}-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v84-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v84-run-identity.json")
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
        assert row["warm_environment"] == ({"SOLVER_V84_WARM_PDHG": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert base.typed_equal(row["warm_marker"], marker) and row["warm_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert base.typed_equal(row["report"], strict_json(row["stdout"]))
        for field, value in quality_fields(row, model, digest, plan).items():
            assert base.typed_equal(row[field], value), (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), warm_marker_error_count=sum(row["warm_marker_error"] is not None for row in rows))
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
        save_once(DATA / "v84-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v84-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v84-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v84-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V84_WARM_PDHG=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；成批求解计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v84-{stage}-manifest.json", manifest)
    raw = DATA / f"v84-{stage}-runs.jsonl.gz"
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
                warm_environment={"SOLVER_V84_WARM_PDHG": "1"} if variant == "diagnostic_on" else {},
                warm_marker=marker, warm_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "warm_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), warm_marker_error_count=sum(row["warm_marker_error"] is not None for row in rows))
    save(DATA / f"v84-{stage}-summary.json", summary)
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
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V84_WARM_PDHG": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V84_WARM_PDHG"] == "1"
    assert dict(os.environ) == before
    polluted = {"solver_v1": "1", "SoLvEr_V84_WaRm_PdHg": "2", "openblas_coretype": "x",
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
    return {"passed": True, "solver_runs": 0, "inherited_count_and_budget_checks": inherited,
        "environment_cases": 6, "configuration_cases": 12, "math_evidence_roles": 2, "marker_cases": marker_self_check()}


def marker_self_check():
    stats = {"phase": "not_started", "seed_available": False, "last_assessed_step": None, **dict.fromkeys(STATS_COUNTS, 0)}
    empty = dict(schema=1, enabled=False, attempted=False, accepted=False, status="disabled", tau=None, kappa=None,
        kernel_status="not_started", kernel_stats=stats, seed_quality=None, last_quality=None)
    assert valid_marker(empty, False)
    on = dict(empty, enabled=True, status="not_triggered")
    assert valid_marker(on, True)
    copied = copy.deepcopy(on); copied.update(attempted=True, status="exception", tau=1.0, kappa=1.0)
    validation = copy.deepcopy(copied); validation.update(status="work_limit", kernel_status="work_limit")
    validation["kernel_stats"].update(phase="validation", budget_checks=1, work_units=16)
    seed = copy.deepcopy(validation); seed.update(status="budget_exception", kernel_status="budget_exception")
    seed["kernel_stats"].update(phase="assessment", seed_available=True)
    quality = dict(completed_steps=0, **dict.fromkeys(QUALITY_BOOLS, False), **dict.fromkeys(QUALITY_NUMBERS, None))
    seeded = copy.deepcopy(seed); seeded.update(status="work_limit", kernel_status="work_limit",
        seed_quality=copy.deepcopy(quality), last_quality=copy.deepcopy(quality))
    seeded["kernel_stats"].update(assessments_started=1, assessments_completed=1, last_assessed_step=0)
    setup = copy.deepcopy(seeded); setup["kernel_stats"].update(phase="setup", setup_started=1)
    completed = copy.deepcopy(seeded)
    completed["kernel_stats"].update(phase="finished", setup_started=1, setup_completed=1,
        steps_started=2, steps_completed=2, transpose_started=2, transpose_completed=2, multiply_started=2, multiply_completed=2)
    valid = [empty, on, copied, validation, seed, seeded, setup, completed]
    # 一个trial在任意原子提交前边界中断，旧point仍是唯一已提交点。
    for stage in ("transpose", "primal", "multiply", "dual", "commit"):
        row = copy.deepcopy(completed); s = row["kernel_stats"]
        s.update(phase=stage, steps_started=3, transpose_started=3)
        if stage != "transpose": s["transpose_completed"] = 3
        if stage in ("multiply", "dual", "commit"): s["multiply_started"] = 3
        if stage in ("dual", "commit"): s["multiply_completed"] = 3
        valid.append(row)
    pending_assessment = copy.deepcopy(completed)
    pending_assessment.update(status="assessment_exception", kernel_status="assessment_exception")
    pending_assessment["kernel_stats"].update(phase="assessment", assessments_started=2)
    valid.append(pending_assessment)
    latest_quality = copy.deepcopy(pending_assessment)
    latest_quality["last_quality"]["completed_steps"] = 2
    valid.append(latest_quality)
    accepted = copy.deepcopy(seeded)
    accepted.update(accepted=True, status="accepted", kernel_status="accepted")
    accepted["kernel_stats"]["phase"] = "finished"
    accepted["last_quality"] = dict(quality, **dict.fromkeys(QUALITY_BOOLS, True))
    valid.append(accepted)
    accepted_steps = copy.deepcopy(completed)
    accepted_steps.update(accepted=True, status="accepted", kernel_status="accepted")
    accepted_steps["kernel_stats"].update(assessments_started=2, assessments_completed=2, last_assessed_step=2)
    accepted_steps["last_quality"] = dict(quality, completed_steps=2, **dict.fromkeys(QUALITY_BOOLS, True))
    valid.append(accepted_steps)
    pending_accepted = copy.deepcopy(accepted_steps)
    pending_accepted["kernel_stats"].update(steps_started=3, transpose_started=3)
    valid.append(pending_accepted)
    post_budget = copy.deepcopy(accepted_steps); post_budget.update(accepted=False, status="budget_exception")
    valid.append(post_budget)
    unsubmitted_quality = copy.deepcopy(latest_quality)
    unsubmitted_quality["last_quality"].update(dict.fromkeys(QUALITY_BOOLS, True))
    valid.append(unsubmitted_quality)
    for row in valid:
        assert valid_marker(row, row["enabled"]), row
    mutations = []
    for key in STATS_COUNTS:
        for wrong in (True, -1, 0.5, None):
            row = copy.deepcopy(completed); row["kernel_stats"][key] = wrong; mutations.append(row)
        row = copy.deepcopy(completed); row["kernel_stats"].pop(key); mutations.append(row)
    for key in ("schema", "enabled", "attempted", "accepted", "status", "tau", "kappa", "kernel_status", "kernel_stats", "seed_quality", "last_quality"):
        row = copy.deepcopy(completed); row.pop(key); mutations.append(row)
    for key, wrong in (("steps_started", 20001), ("work_units", 500000001), ("setup_started", 2),
        ("steps_completed", 4), ("multiply_completed", 3), ("transpose_started", 4), ("assessments_completed", 3),
        ("last_assessed_step", None), ("last_assessed_step", 3), ("last_assessed_step", True), ("seed_available", False), ("phase", "wrong")):
        row = copy.deepcopy(completed); row["kernel_stats"][key] = wrong; mutations.append(row)
    for key in QUALITY_BOOLS:
        row = copy.deepcopy(accepted_steps); row["last_quality"][key] = False; mutations.append(row)
        row = copy.deepcopy(completed); row["last_quality"][key] = 1; mutations.append(row)
    for key in QUALITY_NUMBERS:
        for wrong in (True, float("nan"), float("inf")):
            row = copy.deepcopy(completed); row["last_quality"][key] = wrong; mutations.append(row)
    for key in ("max_row_violation", "max_bound_violation", "max_dual_residual"):
        row = copy.deepcopy(completed); row["last_quality"][key] = -1; mutations.append(row)
    for wrong in (True, -1, 3):
        row = copy.deepcopy(completed); row["last_quality"]["completed_steps"] = wrong; mutations.append(row)
    for key, wrong in (("schema", True), ("schema", 2), ("enabled", 1), ("attempted", 1), ("accepted", 1),
        ("tau", True), ("kappa", float("inf")), ("status", "wrong"), ("kernel_status", "wrong"), ("unexpected", 0)):
        row = copy.deepcopy(completed); row[key] = wrong; mutations.append(row)
    row = copy.deepcopy(accepted_steps); row["kernel_stats"]["last_assessed_step"] = 1; mutations.append(row)
    row = copy.deepcopy(accepted_steps); row["status"] = "budget_exception"; mutations.append(row)
    row = copy.deepcopy(seed); row["seed_quality"] = quality; mutations.append(row)
    row = copy.deepcopy(seeded); row["seed_quality"]["completed_steps"] = 1; mutations.append(row)
    for row in mutations:
        assert not valid_marker(row, True)
    signed = copy.deepcopy(completed)
    signed["last_quality"].update(objective=-1.0, objective_gap=-2.0, relative_objective_gap=-0.1,
        dual_feasible=None, gap_satisfied=None, numerically_optimal=None)
    assert valid_marker(signed, True)
    assert not valid_marker(on, False) and not valid_marker(empty, True)
    for row in valid:
        variant = "diagnostic_on" if row["enabled"] else "diagnostic_off"
        parsed, error = inspect_marker(MARKER + json.dumps(row), variant)
        assert base.typed_equal(parsed, row) and error is None
    assert inspect_marker("", "frozen_v66") == (None, None)
    assert inspect_marker(MARKER + json.dumps(on), "highs")[1] == "unexpected_marker"
    for text in ("", MARKER + "{}", MARKER + '{"schema":1,"schema":1}',
                 MARKER + json.dumps(on) + "\n" + MARKER + json.dumps(on)):
        assert inspect_marker(text, "diagnostic_on")[1] is not None
    duplicate = json.dumps(on).replace('"schema": 1', '"schema": 1, "schema": 1')
    assert inspect_marker(MARKER + duplicate, "diagnostic_on")[1] == "invalid_marker_json"
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "warm_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    changed = copy.deepcopy(row); changed["report"]["backend_version"] = "other"
    assert pair_result(row, changed, plan)["category"] == "different"
    budget = copy.deepcopy(row); budget["status"] = budget["report"]["status"] = "time_limit"
    assert pair_result(dict(row, warm_marker_error="marker_count"), budget, plan)["category"] == "invalid_process"
    for status, code, classification in (("numerical_error", 4, "incomplete"), ("time_limit", 4, "incomplete"),
        ("iteration_limit", 4, "incomplete"), ("verification_failed", 3, "verification_failed")):
        altered = copy.deepcopy(row)
        altered.update(status=status, returncode=code, classification=classification)
        altered["report"]["status"] = status
        assert valid_process(altered)
        altered["report"]["ipm_factorizations"] = True
        assert pair_result(altered, budget, plan)["category"] == "invalid_process"
    return {"valid_and_parse_cases": len(valid), "mutation_cases": len(mutations),
        "sidecar_cases": len(valid) + 6, "full_report_and_budget_cases": 7}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v84-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
