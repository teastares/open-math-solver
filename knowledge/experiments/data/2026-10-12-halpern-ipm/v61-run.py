"""串行比较冻结 v60 与独立诊断开关，完整保存失败及宽方向采集证据。"""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v61-diagnostic"
PLAN_SHA = "d9c0281868acbb9b93243af0d5608572263dbc24aab4511e3a521377f5a10b8b"
RAW, MANIFEST, SUMMARY = (DATA / name for name in ("v61-runs.jsonl.gz", "v61-run-manifest.json", "v61-summary.json"))
VARIANTS = ("frozen_v60", "diagnostic_off", "diagnostic_on")
EXTENDED_COUNTS = ("ipm_extended_krylov_attempts", "ipm_extended_krylov_successes")
MARKER_PREFIX = "V61_HSD "
MAX_INT64 = 9223372036854775807
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def typed_equal(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(typed_equal(a[key], b[key]) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    if type(a) is bool or type(b) is bool:
        return type(a) is type(b) and a == b
    return a == b


def count(value):
    return type(value) is int and 0 <= value <= MAX_INT64


def valid_counts(report):
    if not isinstance(report, dict) or not all(count(report.get(name)) for name in EXTENDED_COUNTS):
        return False
    attempts, successes = (report[name] for name in EXTENDED_COUNTS)
    return successes <= attempts and benchmark.valid_homogeneous_work(report) \
        and benchmark.valid_homogeneous_scaled_work(report) and benchmark.valid_homogeneous_repair_work(report)


def fixed_plan():
    assert sha(DATA / "v61-plan.json") == PLAN_SHA
    plan = read(DATA / "v61-plan.json")
    assert sha(DATA / plan["baseline_identity_file"]) == plan["baseline_identity_sha256"]
    assert sha(DATA / "ipm-v60-identity.json") == plan["parent_identity_sha256"]
    assert sha(DATA / plan["evidence_file"]) == plan["evidence_sha256"]
    assert plan["variants"] == list(VARIANTS) and plan["repeats"] == 2
    assert plan["runs"] == len(plan["models"]) * plan["repeats"] * len(VARIANTS) == 18
    assert plan["linear_accuracy"] == "homogeneous_extended" and plan["method"] == "homogeneous"
    assert all(model["backends"] == ["normal_cholmod"] for model in plan["models"])
    return plan


def configurations(plan):
    return [{"case_index": i * plan["repeats"] + repetition, "model_index": i,
             "repetition": repetition, "model": model, "linear_solver": "normal_cholmod"}
            for i, model in enumerate(plan["models"]) for repetition in range(plan["repeats"])]


def schedule(plan):
    for case in configurations(plan):
        offset = case["case_index"] % len(VARIANTS)
        for variant in VARIANTS[offset:] + VARIANTS[:offset]:
            yield case, variant


def command_for(executable, case, plan):
    command = [str(executable), "solve", case["model"]["input"], "--backend", "ipm",
        "--time-limit", str(plan["time_limit"]), "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
        "--iteration-limit", str(plan["max_iterations"]), "--scaling-passes", str(plan["scaling_passes"]),
        "--ipm-method", plan["method"], "--ipm-linear-accuracy", plan["linear_accuracy"],
        "--ipm-linear-solver", case["linear_solver"]]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def configuration_matches(report, model, plan):
    return isinstance(report, dict) and report.get("backend") == "ipm" \
        and report.get("backend_version") == "ipm-cpu-0.49.0" and report.get("input") == model["input"] \
        and typed_equal(report.get("options"), plan["expected_report_options"]) \
        and report.get("ipm_factor_backend") == "normal_cholmod" \
        and type(report.get("ipm_structural_fallbacks")) is int and report["ipm_structural_fallbacks"] == 0


def valid_process(row):
    report = row.get("report")
    if row.get("error") is not None or not isinstance(report, dict) or type(row.get("returncode")) is not int:
        return False
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1 \
        or row.get("configuration_matches") is not True or not valid_counts(report):
        return False
    known = {"optimal": (0, "resolved"), "empty": (0, "resolved"), "infeasible": (0, "resolved"),
        "unbounded": (0, "resolved"), "verification_failed": (3, "verification_failed"),
        "iteration_limit": (4, "incomplete"), "time_limit": (4, "incomplete"), "numerical_error": (4, "incomplete")}
    status = report.get("status")
    if not isinstance(status, str) or status not in known or (row["returncode"], row.get("classification")) != known[status]:
        return False
    if benchmark.classify(row["returncode"], report) != row["classification"]:
        return False
    timings = report.get("timings")
    if not isinstance(timings, dict) or not all(benchmark.finite_real(timings.get(name)) and timings[name] >= 0
        for name in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")):
        return False
    numerical, wall = row.get("numerical_seconds"), row.get("elapsed_wall_seconds")
    return benchmark.finite_real(numerical) and numerical >= 0 and benchmark.finite_real(wall) and wall > 0


def limited(row, plan):
    return row.get("error") == "wall_timeout" or (row.get("report") or {}).get("status") == "time_limit" \
        or benchmark.finite_real(row.get("elapsed_wall_seconds")) and row["elapsed_wall_seconds"] > plan["wall_timeout"] \
        or benchmark.finite_real(row.get("numerical_seconds")) and row["numerical_seconds"] > plan["time_limit"]


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row.get("report")) for row in (a, b)]
    for report in reports:
        if isinstance(report, dict):
            report.pop("timings", None)
    equal = typed_equal(*reports) and typed_equal(a.get("returncode"), b.get("returncode")) \
        and typed_equal(a.get("error"), b.get("error"))
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, plan) or limited(b, plan)
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def capture_valid(metadata, marker, binary_bytes, plan):
    if not isinstance(metadata, dict) or metadata.get("schema") != "v61_hsd_1" or metadata.get("enabled") is not True:
        return False
    if metadata.get("status") not in ("no_repair_events", "captured"):
        return False
    names = ("event_count", "trigger_count", "dropped_events", "payload_bytes", "cache_bytes",
             "extra_lp_products", "extra_lp_solves", "long_double_digits")
    if not all(count(metadata.get(name)) for name in names):
        return False
    events = metadata.get("events")
    if not isinstance(events, list) or len(events) != metadata["event_count"] \
        or metadata["event_count"] != metadata["trigger_count"] or metadata["dropped_events"] != 0:
        return False
    if metadata["extra_lp_products"] != 0 or metadata["extra_lp_solves"] != 0 or metadata["long_double_digits"] <= 53:
        return False
    limits = metadata.get("limits")
    expected = {"max_events": plan["maximum_repair_events"], "payload_limit_bytes": plan["cache_limit_bytes"],
                "metadata_reserve_bytes": 65536}
    if not typed_equal(limits, expected) or len(events) > limits["max_events"] \
        or metadata["cache_bytes"] > limits["payload_limit_bytes"] - limits["metadata_reserve_bytes"] \
        or metadata["payload_bytes"] > metadata["cache_bytes"] or binary_bytes != 8 + metadata["payload_bytes"]:
        return False
    if (metadata["status"] == "no_repair_events") != (len(events) == 0):
        return False
    if not isinstance(marker, dict) or marker.get("write_error") is not False:
        return False
    return marker.get("status") == metadata["status"] \
        and type(marker.get("event_count")) is int and marker["event_count"] == len(events)


def inspect_capture(prefix, enabled, stderr, plan, report):
    paths = sorted(prefix.parent.glob(prefix.name + ".*"))
    files = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths if path.is_file()}
    markers = [line[len(MARKER_PREFIX):] for line in stderr.splitlines() if line.startswith(MARKER_PREFIX)]
    if not enabled:
        return files, "diagnostic_output_when_disabled" if paths or markers else None, None
    try:
        assert set(files) == {str(prefix) + ".hsd.json", str(prefix) + ".hsd.bin"}
        assert len(markers) == 1
        marker = benchmark.strict_json(markers[0])
        metadata = read(Path(str(prefix) + ".hsd.json"))
        binary = Path(str(prefix) + ".hsd.bin")
        with binary.open("rb") as stream:
            assert stream.read(8) == b"V61HSD1\0"
        assert capture_valid(metadata, marker, binary.stat().st_size, plan)
        if isinstance(report, dict):
            assert type(report.get("ipm_homogeneous_constant_repairs")) is int
            assert metadata["trigger_count"] == report["ipm_homogeneous_constant_repairs"]
        compact = {name: metadata[name] for name in ("schema", "status", "event_count", "trigger_count",
            "dropped_events", "payload_bytes", "cache_bytes", "extra_lp_products", "extra_lp_solves", "long_double_digits")}
        return files, None, compact
    except (AssertionError, ValueError, TypeError, OSError, KeyError):
        return files, "invalid_diagnostic_capture", None


def summarize():
    plan = fixed_plan()
    rows = [benchmark.strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    scheduled = list(schedule(plan))
    assert len(rows) == len(scheduled) == plan["runs"]
    for index, (row, (case, variant)) in enumerate(zip(rows, scheduled)):
        assert row["index"] == index and row["case_index"] == case["case_index"] and row["variant"] == variant
        assert row["model_index"] == case["model_index"] and row["repetition"] == case["repetition"]
        assert row["model"] == case["model"]["name"] and row["input"] == case["model"]["input"]
        assert row["configuration_matches"] is configuration_matches(row["report"], case["model"], plan)
        assert row["numerical_seconds"] == compare_lp.numerical_seconds(row)
        assert row["command"] == command_for(Path(row["command"][0]), case, plan)
    pairs, cases = [], []
    for case in configurations(plan):
        selected_rows = [row for row in rows if row["case_index"] == case["case_index"]]
        selected = {row["variant"]: row for row in selected_rows}
        assert len(selected_rows) == len(selected) == 3 and set(selected) == set(VARIANTS)
        for left, right in zip(VARIANTS, VARIANTS[1:]):
            pairs.append({"case_index": case["case_index"], "model": case["model"]["name"], "repetition": case["repetition"],
                "left": left, "right": right, **pair_result(selected[left], selected[right], plan)})
        cases.append({"case_index": case["case_index"], "model": case["model"]["name"], "repetition": case["repetition"],
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"],
            "capture_prefix": selected["diagnostic_on"]["capture_prefix"],
            "capture_summary": selected["diagnostic_on"]["capture_summary"]})
    summary = {"diagnostic_only": True, "runs": len(rows), "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs),
        "capture_errors": sum(row["capture_error"] is not None for row in rows),
        "raw_sha256": sha(RAW), "scope": "采集成本计入实际预算；预算分叉不视为轨迹兼容；不生成公开性能成绩。"}
    summary["diagnostic_trajectory_verified"] = summary["non_budget_equal"] == 12 and summary["capture_errors"] == 0
    save_new(SUMMARY, summary)
    print(json.dumps({key: summary[key] for key in ("runs", "non_budget_pairs", "non_budget_equal", "budget_pairs",
        "invalid_pairs", "different_pairs", "capture_errors", "diagnostic_trajectory_verified")}), flush=True)
    assert summary["invalid_pairs"] == summary["different_pairs"] == summary["capture_errors"] == 0, "核验失败，全部原始结果已保留"


def run(work):
    plan = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY))
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    assert build["parent_identity_sha256"] == plan["parent_identity_sha256"]
    executables = {"frozen_v60": Path(build["frozen_parent_executable"]),
        "diagnostic_off": Path(build["executable"]), "diagnostic_on": Path(build["executable"])}
    assert executables["frozen_v60"] == ROOT / ".tools/solver-ipm-v60.exe"
    assert executables["diagnostic_on"] == work / "solver-diagnostic.exe"
    identities = {str(executables["frozen_v60"]): build["frozen_parent_sha256"],
        str(executables["diagnostic_on"]): build["executable_sha256"]}
    baseline = read(DATA / plan["baseline_identity_file"])
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    assert build["production_sha256"] == baseline["solver_sha256"]
    protected = {str(Path(__file__)): sha(__file__), str(build_path): sha(build_path), **identities}
    for group in (build["protected_before_and_after"], build["link_explicit_before_and_after"],
                  build["compile_inputs_after_and_final"], build["outputs"]):
        for path, expected in group.items():
            assert path not in protected or protected[path] == expected, path
            protected[path] = expected
    for group in ("files", "libraries"):
        for name, expected in baseline[group].items():
            path = str(ROOT / name)
            assert path not in protected or protected[path] == expected, path
            protected[path] = expected
    for path, expected in protected.items():
        assert sha(path) == expected, path
    for model in plan["models"]:
        assert sha(model["input"]) == model["sha256"]
    capture_root = work / "captures"
    capture_root.mkdir()
    environment = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "build_manifest": str(build_path), "protected": protected, "environment": environment,
        "removed_environment": "所有 SOLVER_V 开头变量及 OPENBLAS_CORETYPE；on 只设置 SOLVER_V61_DIAGNOSTICS",
        "capture_root": str(capture_root), "work": str(work),
        "schedule": "冻结模型顺序，每模型两次重复；三 variant 向左轮转 case_index mod 3，全部串行；生产 v57 只保护不执行。"})
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for index, (case, variant) in enumerate(schedule(plan)):
            executable, enabled = executables[variant], variant == "diagnostic_on"
            prefix = capture_root / (str(case["case_index"]) + "-" + variant)
            command = command_for(executable, case, plan)
            env = {key: value for key, value in os.environ.items() if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
            env.update(environment)
            if enabled:
                env[plan["diagnostic_environment"]] = str(prefix)
            model = case["model"]
            before = {"input": sha(model["input"]), "solver": sha(executable)}
            start, error = time.perf_counter(), None
            try:
                process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=plan["wall_timeout"], env=env)
                stdout, stderr, code = process.stdout, process.stderr, process.returncode
            except subprocess.TimeoutExpired as failure:
                stdout, stderr, code, error = failure.stdout or b"", failure.stderr or b"", None, "wall_timeout"
            except OSError as failure:
                stdout, stderr, code, error = b"", str(failure).encode("utf-8"), None, "launch_error"
            elapsed = time.perf_counter() - start
            after = {"input": sha(model["input"]), "solver": sha(executable)}
            stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
            try:
                report = benchmark.strict_json(stdout)
                if not isinstance(report, dict):
                    report, error = None, error or "invalid_report"
            except (ValueError, TypeError):
                report, error = None, error or "invalid_json"
            files, capture_error, capture_summary = inspect_capture(prefix, enabled, stderr, plan, report)
            row = {"index": index, "case_index": case["case_index"], "model_index": case["model_index"],
                "repetition": case["repetition"], "input": model["input"], "model": model["name"],
                "linear_solver": case["linear_solver"], "variant": variant, "command": command, "returncode": code,
                "report": report, "stdout": stdout, "stderr": stderr, "error": error, "elapsed_wall_seconds": elapsed,
                "hashes_before": before, "hashes_after": after, "model_sha256": before["input"],
                "classification": benchmark.classify(code, report) if report is not None else error,
                "capture_prefix": str(prefix), "capture_files": files, "capture_error": capture_error,
                "capture_summary": capture_summary, "configuration_matches": configuration_matches(report, model, plan)}
            row["numerical_seconds"] = compare_lp.numerical_seconds(row)
            failure = compare_lp.quality_failure(row, model, after["input"], identities[str(executable)], after["solver"])
            if not valid_process(row):
                failure = "invalid_process"
            elif failure is None and limited(row, plan):
                failure = "budget_exceeded"
            row.update(quality_pass=failure is None, quality_failure=failure)
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            assert before == after and before["input"] == model["sha256"] and before["solver"] == identities[str(executable)]
            print(json.dumps({"index": index, "model": model["name"], "repetition": case["repetition"], "variant": variant,
                "status": (report or {}).get("status"), "captures": len(files), "valid_process": valid_process(row),
                "capture_error": capture_error}), flush=True)
    for path, expected in protected.items():
        assert sha(path) == expected, path
    summarize()


def self_test():
    """仅合成内存对象，不读计划、哈希库或启动任何子进程。"""
    plan = {"time_limit": 5, "wall_timeout": 35, "maximum_repair_events": 8, "cache_limit_bytes": 134217728}
    report = {"schema_version": 1, "status": "numerical_error", "backend_version": "ipm-cpu-0.49.0",
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1},
        "ipm_homogeneous_tau": 1.0, "ipm_homogeneous_kappa": 1.0,
        "ipm_homogeneous_constant_solves": 1, "ipm_homogeneous_direction_solves": 1,
        "ipm_homogeneous_scaled_products": 0, "ipm_homogeneous_scaled_refinement_steps": 0,
        "ipm_homogeneous_scaled_inexact_returns": 0, "ipm_homogeneous_constant_repairs": 0,
        "ipm_homogeneous_constant_repair_successes": 0,
        "ipm_extended_krylov_attempts": 2, "ipm_extended_krylov_successes": 1}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "configuration_matches": True,
        "report": report, "numerical_seconds": 1, "elapsed_wall_seconds": 2}
    assert pair_result(row, row, plan)["category"] == "equal"
    time_row = copy.deepcopy(row)
    time_row["report"]["status"] = "time_limit"
    assert pair_result(row, time_row, plan)["category"] == "budget_limited"
    different = copy.deepcopy(row)
    different["report"]["backend_version"] = "changed"
    assert pair_result(row, different, plan)["category"] == "different"
    different = copy.deepcopy(row)
    different["report"]["timings"]["total_seconds"] = 2
    assert pair_result(row, different, plan)["category"] == "equal"
    different = copy.deepcopy(row)
    different["report"]["additional_flag"] = True
    different2 = copy.deepcopy(different)
    different2["report"]["additional_flag"] = 1
    assert pair_result(different, different2, plan)["category"] == "different"
    assert not typed_equal({"a": [True]}, {"a": [1]}) and typed_equal({"a": [1.0]}, {"a": [1]})
    broken_rows = []
    for key, value in (("returncode", -1073741819), ("returncode", False), ("returncode", 0),
        ("error", "launch_error"), ("configuration_matches", False), ("numerical_seconds", None),
        ("numerical_seconds", float("nan")), ("elapsed_wall_seconds", -1)):
        broken = copy.deepcopy(row)
        broken[key] = value
        broken_rows.append(broken)
    for status in ("bogus", None, "error"):
        broken = copy.deepcopy(row)
        broken["report"]["status"] = status
        broken_rows.append(broken)
    broken = copy.deepcopy(row)
    broken["report"]["schema_version"] = True
    broken_rows.append(broken)
    for broken in broken_rows:
        assert not valid_process(broken) and pair_result(broken, time_row, plan)["category"] == "invalid_process"
    count_checks = 0
    for field in EXTENDED_COUNTS:
        for value in (None, True, False, -1, 1.5, MAX_INT64 + 1):
            broken = copy.deepcopy(row)
            broken["report"][field] = value
            assert not valid_process(broken) and pair_result(broken, time_row, plan)["category"] == "invalid_process"
            count_checks += 1
        broken = copy.deepcopy(row)
        del broken["report"][field]
        assert not valid_process(broken)
        count_checks += 1
    broken = copy.deepcopy(row)
    broken["report"][EXTENDED_COUNTS[1]] = 3
    assert not valid_process(broken)
    count_checks += 1
    for status in ("numerical_error", "iteration_limit", "time_limit", "verification_failed", "infeasible", "unbounded", "optimal", "empty"):
        broken = copy.deepcopy(row)
        broken["report"]["status"] = status
        broken["returncode"], broken["classification"] = (3, "verification_failed") if status == "verification_failed" else \
            (0, "resolved") if status in ("infeasible", "unbounded", "optimal", "empty") else (4, "incomplete")
        broken["report"]["backend"] = "ipm"
        if status in ("optimal", "empty"):
            broken["report"].update(has_solution=True, objective=0, solution=[], row_dual=[],
                model={"columns": 0, "rows": 0, "is_mip": False},
                verification={"valid_input": True, "feasible": True, "objective_consistent": True, "recomputed_objective": 0},
                lp_optimality={"valid_input": True, "dual_feasible": True, "gap_satisfied": True, "numerically_optimal": True,
                    "max_dual_residual": 0, "dual_objective_estimate": 0, "objective_gap": 0, "relative_objective_gap": 0})
        assert valid_process(broken)
        broken["report"][EXTENDED_COUNTS[0]] = True
        assert not valid_process(broken)
        count_checks += 1
    capture = {"schema": "v61_hsd_1", "enabled": True, "status": "no_repair_events", "event_count": 0,
        "trigger_count": 0, "dropped_events": 0, "payload_bytes": 0, "cache_bytes": 0,
        "extra_lp_products": 0, "extra_lp_solves": 0, "long_double_digits": 64, "events": [],
        "limits": {"max_events": 8, "payload_limit_bytes": 134217728, "metadata_reserve_bytes": 65536}}
    marker = {"write_error": False, "event_count": 0, "status": "no_repair_events"}
    assert capture_valid(capture, marker, 8, plan)
    captured = {**capture, "status": "captured", "event_count": 1, "trigger_count": 1,
        "payload_bytes": 8, "cache_bytes": 56, "events": [{"index": 0}]}
    assert capture_valid(captured, {**marker, "status": "captured", "event_count": 1}, 16, plan)
    capture_checks = 2
    for field, value in (("event_count", True), ("trigger_count", 1), ("dropped_events", 1),
        ("extra_lp_products", 1), ("extra_lp_solves", 1), ("status", "event_limit"),
        ("status", "cache_limit"), ("enabled", 1), ("long_double_digits", 53), ("payload_bytes", 1)):
        bad = copy.deepcopy(capture)
        bad[field] = value
        assert not capture_valid(bad, marker, 8, plan)
        capture_checks += 1
    for bad_marker in ({**marker, "write_error": True}, {**marker, "event_count": False}, {**marker, "status": "captured"}):
        assert not capture_valid(capture, bad_marker, 8, plan)
        capture_checks += 1
    mock = {**plan, "models": [{"name": "m" + str(i), "input": "m" + str(i) + ".mps"} for i in range(3)],
        "repeats": 2, "threads": 1, "seed": 0, "max_iterations": 200, "scaling_passes": 5,
        "method": "homogeneous", "linear_accuracy": "homogeneous_extended", "shared_ipm": {"objective-scaling": "off"}}
    cases, scheduled = configurations(mock), list(schedule(mock))
    assert len(cases) == 6 and len(scheduled) == 18 and [case["case_index"] for case in cases] == list(range(6))
    assert all([variant for case, variant in scheduled if case["case_index"] == i] == list(VARIANTS[i % 3:] + VARIANTS[:i % 3]) for i in range(6))
    for case in cases:
        command = command_for(Path("test.exe"), case, mock)
        assert command.count("--ipm-objective-scaling") == 1 and command[command.index("--ipm-objective-scaling") + 1] == "off"
        assert command[command.index("--ipm-linear-accuracy") + 1] == "homogeneous_extended"
        flags = [value for value in command if value.startswith("--")]
        assert len(flags) == len(set(flags))
    print(json.dumps({"passed": True, "trajectory_checks": 5 + len(broken_rows), "typed_checks": 2,
        "count_checks": count_checks, "capture_checks": capture_checks, "commands_checked": len(cases),
        "schedule_rows": len(scheduled), "lp_runs": 0, "filesystem_hashes": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run", "summary", "self-test"))
    parser.add_argument("--self-test", action="store_true", dest="self_test_flag")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.self_test_flag:
        if args.action not in (None, "self-test"):
            parser.error("--self-test 不能与实际运行或摘要动作同时使用")
        self_test()
    elif args.action == "run":
        run(args.work.resolve())
    elif args.action == "summary":
        summarize()
    elif args.action == "self-test":
        self_test()
    else:
        parser.error("必须指定 run、summary 或 --self-test")
