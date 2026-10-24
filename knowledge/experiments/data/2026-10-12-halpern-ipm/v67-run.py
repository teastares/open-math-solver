"""串行比较冻结 v66 与独立诊断开关，完整保存失败及普通常数方向失败采集证据。"""
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
WORK = ROOT / ".tools/v67-diagnostic"
PLAN_SHA = "0dafab3cc269cec22abe22bb2dbff014dd7a8d2d45255acd54ac3bfad0f42c8b"
RAW, MANIFEST, SUMMARY = (DATA / name for name in ("v67-runs.jsonl.gz", "v67-run-manifest.json", "v67-summary.json"))
VARIANTS = ("frozen_v66", "diagnostic_off", "diagnostic_on")
COUNT_FIELDS = ('ipm_iterations', 'ipm_factorizations', 'ipm_refinement_steps', 'ipm_weighted_krylov_steps', 'ipm_krylov_recycle_projections', 'ipm_krylov_recycle_acceptances', 'ipm_krylov_recycle_vectors', 'ipm_krylov_recycle_peak_dimension', 'ipm_regularization_searches', 'ipm_regularization_reductions', 'ipm_unshifted_factorizations', 'ipm_full_newton_preconditioner_solves', 'ipm_full_newton_products', 'ipm_bounded_newton_returns', 'ipm_bounded_newton_inexact_returns', 'ipm_homogeneous_constant_solves', 'ipm_homogeneous_constant_repairs', 'ipm_homogeneous_constant_repair_successes', 'ipm_homogeneous_direction_solves', 'ipm_homogeneous_scaled_products', 'ipm_homogeneous_scaled_refinement_steps', 'ipm_homogeneous_scaled_inexact_returns', 'ipm_direction_retries', 'ipm_direction_retry_successes', 'ipm_schur_residual_products', 'ipm_schur_refinement_steps', 'ipm_schur_refinement_acceptances', 'ipm_kkt_dimension', 'ipm_factor_nonzeros', 'ipm_structural_fallbacks', 'ipm_eliminated_singletons', 'ipm_regularized_pivots', 'ipm_backward_error_acceptances', 'ipm_inexact_acceptances', 'ipm_inexact_prefilter_rejections', 'ipm_eliminated_fixed_variables', 'ipm_presolved_singleton_rows', 'ipm_presolved_fixed_columns', 'ipm_central_correction_attempts', 'ipm_central_correction_acceptances')
OPTIONAL_COUNT_FIELDS = ("ipm_kkt_dimension", "ipm_factor_nonzeros", "ipm_eliminated_singletons")
ABSENT_FIELDS = ("ipm_extended_krylov_attempts", "ipm_extended_krylov_successes",
                 "ipm_homogeneous_scalar_checks", "ipm_homogeneous_scalar_rejections")
MARKER_PREFIX = "V67_KRYLOV "
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
    if not isinstance(report, dict) or any(name in report for name in ABSENT_FIELDS) or not all(name in report for name in COUNT_FIELDS):
        return False
    for name in COUNT_FIELDS:
        value = report[name]
        if value is None and name in OPTIONAL_COUNT_FIELDS:
            continue
        if not count(value):
            return False
    return report["ipm_weighted_krylov_steps"] == 0 and report["ipm_homogeneous_constant_repair_successes"] <= \
        report["ipm_homogeneous_constant_repairs"] <= report["ipm_homogeneous_constant_solves"] and \
        benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report)


def fixed_plan():
    assert sha(DATA / "v67-plan.json") == PLAN_SHA
    plan = read(DATA / "v67-plan.json")
    for path, digest in ((plan["baseline_identity_file"], plan["baseline_identity_sha256"]),
        ("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"])):
        assert sha(DATA / path) == digest
    assert plan["variants"] == list(VARIANTS) and plan["repeats"] == 2
    assert plan["runs"] == len(plan["models"]) * plan["repeats"] * len(VARIANTS) == 18
    assert [model["name"] for model in plan["models"]] == ["greenbea", "forplan", "greenbeb"]
    assert (plan["linear_accuracy"], plan["method"], plan["linear_solver"]) == ("homogeneous_repair", "homogeneous", "normal_cholmod_aat")
    assert (plan["time_limit"],plan["wall_timeout"],plan["max_iterations"],plan["threads"],plan["seed"]) == (5,35,200,1,0)
    assert plan["diagnostic_environment"] == "SOLVER_V67_CAPTURE" and plan["capture_magic"] == "V67KRY01"
    assert (plan["cache_limit_bytes"],plan["metadata_limit_bytes"],plan["maximum_events"],plan["maximum_columns"]) == (134217728,65536,1,32)
    assert plan["expected_report_options"]["ipm"]["objective_scaling"] is False
    assert "objective-scaling" not in plan["shared_ipm"] and "direction-retries" not in plan["shared_ipm"]
    return plan


def configurations(plan):
    return [{"case_index": i * plan["repeats"] + repetition, "model_index": i,
             "repetition": repetition, "model": model, "linear_solver": plan["linear_solver"]}
            for i, model in enumerate(plan["models"]) for repetition in range(plan["repeats"])]


def schedule(plan):
    for case in configurations(plan):
        offset = case["case_index"] % len(VARIANTS)
        for variant in VARIANTS[offset:] + VARIANTS[:offset]:
            yield case, variant


def command_for(executable, case, plan):
    command = [str(executable), "solve", case["model"]["input"], "--backend", "ipm",
        "--time-limit", str(plan["time_limit"]), "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
        "--mip-gap", "0.0001",
        "--iteration-limit", str(plan["max_iterations"]), "--scaling-passes", str(plan["scaling_passes"]),
        "--ipm-method", plan["method"], "--ipm-linear-accuracy", plan["linear_accuracy"],
        "--ipm-linear-solver", case["linear_solver"], "--ipm-objective-scaling", "off", "--ipm-direction-retries", "0"]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def configuration_matches(report, model, plan):
    if not isinstance(report, dict) or report.get("backend") != "ipm" or report.get("backend_version") != "ipm-cpu-0.53.0" \
        or report.get("input") != model["input"] or not typed_equal(report.get("options"), plan["expected_report_options"]):
        return False
    actual, fallback = report.get("ipm_factor_backend"), report.get("ipm_structural_fallbacks")
    return count(fallback) and (actual is None or actual == "normal_cholmod_aat" and fallback == 0 or actual == "factorhighs" and fallback > 0)


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


CAPTURE_COUNTS = ("completed_iterations", "factorizations", "refinements", "constant_solves", "direction_solves", "constant_repairs", "constant_repair_successes")
CAPTURE_TO_REPORT = dict(zip(CAPTURE_COUNTS, ("ipm_iterations", "ipm_factorizations", "ipm_refinement_steps",
    "ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves", "ipm_homogeneous_constant_repairs",
    "ipm_homogeneous_constant_repair_successes")))


def work_valid(work):
    if not isinstance(work, dict) or not all(count(work.get(key)) for key in
        ("ir_candidates", "ir_corrections_started", "ir_corrections_completed", "krylov_calls")):
        return False
    if not (work["ir_candidates"] <= 3 and work["ir_corrections_completed"] <= work["ir_corrections_started"] <= 2 and work["krylov_calls"] <= 1):
        return False
    cycles = work.get("cycles")
    if not isinstance(cycles,list) or len(cycles) != 3:
        return False
    for index, cycle in enumerate(cycles):
        if not isinstance(cycle,dict) or type(cycle.get("restart_index")) is not int or cycle["restart_index"] != index or \
            any(type(cycle.get(key)) is not bool for key in ("started","ended")) or \
            any(not count(cycle.get(key)) for key in ("columns_started","preconditioner_solves_completed","columns_completed","candidates","remaining_zero_count")):
            return False
        if not (cycle["candidates"] <= cycle["columns_completed"] <= cycle["preconditioner_solves_completed"] <= cycle["columns_started"] <= 32 and
                cycle["remaining_zero_count"] <= cycle["columns_completed"]):
            return False
        if (cycle["ended"] and not cycle["started"]) or (not cycle["started"] and cycle["columns_started"]):
            return False
    return True


def capture_valid(metadata, marker, binary_bytes, metadata_bytes, plan):
    try:
        assert type(metadata["schema"]) is int and metadata["schema"] == 1 and metadata["run_started"] is True
        assert metadata["status"] in ("captured","no_target_failure") and metadata["error"] == ""
        assert metadata["capture_errors"] == 0 and type(metadata["capture_errors"]) is int and metadata["truncated"] is False
        assert type(metadata["terminal"]) is bool
        for key in ("capture_started","terminal_failures","event_count","ordinary_scopes_started","ordinary_scopes_completed",
                    "ordinary_scopes_unwound","fixed_cache_bytes","binary_bytes"):
            assert count(metadata[key])
        assert metadata["ordinary_scopes_started"] == metadata["ordinary_scopes_completed"] + metadata["ordinary_scopes_unwound"]
        assert metadata["last_scope_exit"] in ("not_started","returned","nonterminal_exception","krylov_exhausted")
        assert work_valid(metadata["last_scope_work"])
        assert all(count(metadata["last_scope_after_counts"][key]) for key in CAPTURE_COUNTS)
        assert metadata["byte_order"] in ("little","big")
        assert metadata["floating_point"]["double"]["bytes"] == 8 and metadata["floating_point"]["double"]["digits"] == 53
        assert metadata["floating_point"]["double"]["is_iec559"] is True
        assert count(metadata["floating_point"]["long_double"]["digits"]) and metadata["floating_point"]["long_double"]["digits"] >= 53
        assert type(metadata["rounding_mode"]["value"]) is int and type(metadata["rounding_mode"]["is_to_nearest"]) is bool
        limits = {key:plan[key] for key in ("maximum_events","maximum_columns","cache_limit_bytes","metadata_limit_bytes")}
        assert all(type(metadata["limits"][key]) is int for key in limits)
        assert typed_equal(metadata["limits"],limits) and metadata["binary_bytes"] == binary_bytes
        assert 0 < metadata_bytes <= limits["metadata_limit_bytes"] and 0 < metadata["fixed_cache_bytes"] <= limits["cache_limit_bytes"]
        events = metadata["events"]
        assert isinstance(events,list) and len(events) == metadata["event_count"] <= 1
        assert type(marker["schema"]) is int and marker["schema"] == 1 and marker["write_ok"] is True
        assert all(typed_equal(marker[key],metadata[key]) for key in ("status","event_count","capture_errors","truncated","terminal_failures"))
        if not events:
            assert metadata["status"] == "no_target_failure" and metadata["terminal"] is False and metadata["terminal_failures"] == 0
            assert binary_bytes == 8
            return True
        assert metadata["status"] == "captured" and metadata["terminal"] is True and metadata["terminal_failures"] == 1
        assert metadata["capture_started"] >= 1 and metadata["last_scope_exit"] == "krylov_exhausted"
        event = events[0]
        assert type(event["index"]) is int and event["index"] == 0 and event["terminal"] is True
        assert event["phase"] == "ordinary_constant" and event["linear_exit"] == "krylov_exhausted"
        assert event["candidate_submitted"] is False and event["residual_sign"] == "rhs_minus_K0_candidate"
        assert all(event[key] is True for key in ("base_available","z_available","coefficients_available","last_candidate_available","last_residual_available","linear_rhs_available"))
        assert all(count(event[key]) for key in ("iteration","n","m","dimension","nnz","generated_columns","coefficient_count","payload_bytes","cache_bytes","restart_index","column_index"))
        n,m,d,nnz,cols,coeffs = (event[key] for key in ("n","m","dimension","nnz","generated_columns","coefficient_count"))
        assert d == n+m and d > 0 and 1 <= coeffs <= cols <= 32 and event["restart_index"] == 2 and event["column_index"] == coeffs-1
        assert event["cache_bytes"] == metadata["fixed_cache_bytes"] + event["payload_bytes"] <= limits["cache_limit_bytes"]
        assert binary_bytes == 8+event["payload_bytes"]
        assert isinstance(event["global_tolerance"],str) and 0 < float.fromhex(event["global_tolerance"]) < float("inf")
        assert event["block_tolerances"] is None or (isinstance(event["block_tolerances"],list) and len(event["block_tolerances"]) == 2 and
            all(isinstance(value,str) and 0 <= float.fromhex(value) < float("inf") for value in event["block_tolerances"]))
        assert all(count(event[side][key]) for side in ("before_counts","after_counts") for key in CAPTURE_COUNTS)
        assert all(event["before_counts"][key] <= event["after_counts"][key] for key in CAPTURE_COUNTS)
        assert typed_equal(event["after_counts"],metadata["last_scope_after_counts"]) and typed_equal(event["work"],metadata["last_scope_work"])
        assert work_valid(event["work"]) and event["work"]["krylov_calls"] == 1
        assert all(cycle["started"] and cycle["ended"] for cycle in event["work"]["cycles"])
        assert event["after_counts"]["refinements"] - event["before_counts"]["refinements"] == \
            event["work"]["ir_corrections_completed"] + sum(cycle["preconditioner_solves_completed"] for cycle in event["work"]["cycles"])
        assert event["work"]["cycles"][2]["columns_completed"] == cols
        expected = [("g_starts","int64",[n+1]),("g_rows","int64",[nnz]),("g_values","float64",[nnz]),
            ("h","float64",[m]),("linear_rhs","float64",[d]),("base","float64",[d]),("z","float64",[cols,d]),
            ("coefficients","float64",[coeffs]),("last_candidate","float64",[d]),("last_residual","float64",[d])]
        arrays=event["arrays"];assert isinstance(arrays,list) and len(arrays)==len(expected)
        offset=8
        for array,(name,dtype,shape) in zip(arrays,expected):
            total=shape[0] if len(shape)==1 else shape[0]*shape[1]
            assert all(count(array[key]) for key in ("count","width","offset")) and all(count(value) for value in array["shape"])
            assert typed_equal(array,{"name":name,"dtype":dtype,"count":total,"width":8,"offset":offset,"shape":shape})
            offset += total*8
        assert offset == binary_bytes
        return True
    except (AssertionError,KeyError,TypeError,ValueError,OverflowError):
        return False


def inspect_capture(prefix, enabled, stderr, plan, report):
    paths = sorted(prefix.parent.glob(prefix.name + ".*"))
    files = {str(path): {"sha256":sha(path),"bytes":path.stat().st_size} for path in paths if path.is_file()}
    markers = [line[len(MARKER_PREFIX):] for line in stderr.splitlines() if line.startswith(MARKER_PREFIX)]
    if not enabled:
        return files, "diagnostic_output_when_disabled" if paths or markers else None, None
    try:
        json_path, binary_path = Path(str(prefix)+".json"), Path(str(prefix)+".bin")
        assert set(files)=={str(json_path),str(binary_path)} and len(markers)==1
        metadata,marker=read(json_path),benchmark.strict_json(markers[0])
        with binary_path.open("rb") as stream:
            assert stream.read(8)==plan["capture_magic"].encode("ascii")
        assert capture_valid(metadata,marker,binary_path.stat().st_size,json_path.stat().st_size,plan)
        if metadata["event_count"]:
            assert isinstance(report,dict)
            assert all(count(report.get(field)) and metadata["events"][0]["after_counts"][key] <= report[field]
                       for key,field in CAPTURE_TO_REPORT.items())
        compact={key:metadata[key] for key in ("schema","status","run_started","capture_started","terminal_failures",
            "event_count","capture_errors","truncated","ordinary_scopes_started","ordinary_scopes_completed","ordinary_scopes_unwound",
            "last_scope_exit","fixed_cache_bytes","binary_bytes")}
        return files,None,compact
    except (AssertionError,KeyError,TypeError,ValueError,OSError):
        return files,"invalid_diagnostic_capture",None


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
    executables = {"frozen_v66": Path(build["frozen_parent_executable"]),
        "diagnostic_off": Path(build["executable"]), "diagnostic_on": Path(build["executable"])}
    assert executables["frozen_v66"] == ROOT / ".tools/solver-ipm-v66.exe"
    assert executables["diagnostic_on"] == work / "solver-diagnostic.exe"
    identities = {str(executables["frozen_v66"]): build["frozen_parent_sha256"],
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
        "removed_environment": "所有 SOLVER_V 开头变量及 OPENBLAS_CORETYPE；on 只设置 SOLVER_V67_CAPTURE",
        "capture_root": str(capture_root), "work": str(work),
        "schedule": "冻结模型顺序，每模型两次重复；三 variant 向左轮转 case_index mod 3，全部串行；生产 v66 只保护，真实参考执行冻结副本。"})
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


def capture_self_test():
    """仅用内存元数据覆盖正常控制、完整事件和预算边界。"""
    plan = {"maximum_events":1,"maximum_columns":32,"cache_limit_bytes":134217728,"metadata_limit_bytes":65536}
    empty_work = {"ir_candidates":0,"ir_corrections_started":0,"ir_corrections_completed":0,"krylov_calls":0,
        "cycles":[{"restart_index":i,"started":False,"ended":False,"columns_started":0,
            "preconditioner_solves_completed":0,"columns_completed":0,"candidates":0,"remaining_zero_count":0} for i in range(3)]}
    control = {"schema":1,"status":"no_target_failure","run_started":True,"capture_started":0,
        "capture_errors":0,"truncated":False,"terminal":False,"terminal_failures":0,"event_count":0,"error":"",
        "ordinary_scopes_started":0,"ordinary_scopes_completed":0,"ordinary_scopes_unwound":0,
        "last_scope_exit":"not_started","last_scope_work":copy.deepcopy(empty_work),
        "last_scope_after_counts":dict.fromkeys(CAPTURE_COUNTS,0),"fixed_cache_bytes":1024,"binary_bytes":8,
        "byte_order":"little","limits":copy.deepcopy(plan),"rounding_mode":{"value":0,"is_to_nearest":True},
        "floating_point":{"double":{"bytes":8,"digits":53,"is_iec559":True},
            "long_double":{"bytes":16,"digits":64},"flt_eval_method":0},"events":[]}
    def marker(value):
        return {"schema":1,"write_ok":True,**{key:value[key] for key in
            ("status","event_count","capture_errors","truncated","terminal_failures")}}
    checks = 0
    def check(value, expected, **overrides):
        nonlocal checks
        arguments = {"metadata":value,"marker":marker(value),"binary_bytes":value["binary_bytes"],"metadata_bytes":1024,"plan":plan}
        arguments.update(overrides)
        assert capture_valid(**arguments) is expected
        checks += 1
    check(control,True)
    boundary = copy.deepcopy(control)
    boundary.update(capture_started=1,ordinary_scopes_started=1,ordinary_scopes_unwound=1,last_scope_exit="nonterminal_exception")
    check(boundary,True)
    work = copy.deepcopy(empty_work)
    work.update(ir_candidates=3,ir_corrections_started=2,ir_corrections_completed=2,krylov_calls=1)
    for cycle in work["cycles"]:
        cycle.update(started=True,ended=True,columns_started=1,preconditioner_solves_completed=1,columns_completed=1,candidates=1)
    before = dict.fromkeys(CAPTURE_COUNTS,0);before["constant_solves"]=1
    after = dict(before,refinements=5)
    event = {"index":0,"phase":"ordinary_constant","linear_exit":"krylov_exhausted","terminal":True,
        "iteration":0,"n":1,"m":1,"dimension":2,"nnz":1,"restart_index":2,"column_index":0,
        "generated_columns":1,"coefficient_count":1,"candidate_submitted":False,
        "residual_sign":"rhs_minus_K0_candidate","global_tolerance":"0x1p-20","block_tolerances":None,
        "before_counts":before,"after_counts":after,"work":work,
        **dict.fromkeys(("base_available","z_available","coefficients_available","last_candidate_available","last_residual_available","linear_rhs_available"),True)}
    event["arrays"] = []
    offset = 8
    for name,dtype,shape in (("g_starts","int64",[2]),("g_rows","int64",[1]),("g_values","float64",[1]),
        ("h","float64",[1]),("linear_rhs","float64",[2]),("base","float64",[2]),("z","float64",[1,2]),
        ("coefficients","float64",[1]),("last_candidate","float64",[2]),("last_residual","float64",[2])):
        length=shape[0] if len(shape)==1 else shape[0]*shape[1]
        event["arrays"].append({"name":name,"dtype":dtype,"count":length,"width":8,"offset":offset,"shape":shape})
        offset += length*8
    event.update(payload_bytes=offset-8,cache_bytes=1024+offset-8)
    captured=copy.deepcopy(control)
    captured.update(status="captured",capture_started=1,terminal=True,terminal_failures=1,event_count=1,
        ordinary_scopes_started=1,ordinary_scopes_unwound=1,last_scope_exit="krylov_exhausted",
        last_scope_work=copy.deepcopy(work),last_scope_after_counts=copy.deepcopy(after),events=[event],binary_bytes=offset)
    check(captured,True)
    blocks=copy.deepcopy(captured);blocks["events"][0]["block_tolerances"]=["0x0p+0","0x1p-10"]
    check(blocks,True)
    mutations = [(("schema",),True),(("run_started",),False),(("status",),"capture_error"),
        (("capture_errors",),True),(("truncated",),True),(("terminal",),1),(("event_count",),True),
        (("terminal_failures",),0),(("capture_started",),0),(("ordinary_scopes_started",),2),
        (("last_scope_exit",),"active"),(("limits","maximum_events"),1.0),(("fixed_cache_bytes",),0),
        (("rounding_mode","is_to_nearest"),1),(("events",0,"candidate_submitted"),True),
        (("events",0,"restart_index"),1),(("events",0,"column_index"),1),(("events",0,"dimension"),3),
        (("events",0,"coefficient_count"),0),(("events",0,"global_tolerance"),"nan"),
        (("events",0,"block_tolerances"),["-0x1p0","0x1p0"]),(("events",0,"cache_bytes"),134217729),
        (("events",0,"before_counts","refinements"),6),(("events",0,"after_counts","refinements"),6),
        (("events",0,"arrays",6,"shape"),[2,1]),(("events",0,"arrays",0,"offset"),9),
        (("events",0,"arrays",0,"width"),8.0),(("events",0,"arrays",0,"count"),True),
        (("events",0,"last_residual_available"),False)]
    for path,value in mutations:
        bad=copy.deepcopy(captured);target=bad
        for key in path[:-1]: target=target[key]
        target[path[-1]]=value
        check(bad,False)
    for field,value in (("ir_candidates",4),("ir_corrections_completed",3),("krylov_calls",2)):
        bad=copy.deepcopy(captured)
        for target in (bad["last_scope_work"],bad["events"][0]["work"]): target[field]=value
        check(bad,False)
    for field,value in (("preconditioner_solves_completed",0),("columns_started",33),("candidates",2),
        ("remaining_zero_count",2),("started",1),("ended",False)):
        bad=copy.deepcopy(captured)
        for target in (bad["last_scope_work"],bad["events"][0]["work"]): target["cycles"][2][field]=value
        check(bad,False)
    bad=copy.deepcopy(captured)
    for target in (bad["last_scope_after_counts"],bad["events"][0]["after_counts"]): target["refinements"]=6
    check(bad,False)
    bad_marker=marker(captured);bad_marker["write_ok"]=False
    check(captured,False,marker=bad_marker)
    bad_marker=marker(captured);bad_marker["event_count"]=0
    check(captured,False,marker=bad_marker)
    check(captured,False,binary_bytes=offset+8)
    check(captured,False,metadata_bytes=65537)
    bad=copy.deepcopy(control);bad["terminal"]=True
    check(bad,False)
    bad=copy.deepcopy(control);bad["capture_errors"]=1
    check(bad,False)
    return checks


def self_test():
    """仅合成内存对象，不读计划、哈希库或启动任何子进程。"""
    plan = {"time_limit":5,"wall_timeout":35,"models":[{"name":"m"+str(i),"input":"m"+str(i)+".mps"} for i in range(3)],
        "repeats":2,"threads":1,"seed":0,"max_iterations":200,"scaling_passes":5,"method":"homogeneous",
        "linear_accuracy":"homogeneous_repair","linear_solver":"normal_cholmod_aat",
        "shared_ipm":{"singleton-elimination":"on","dynamic-regularization":"off"}}
    report = {"schema_version":1,"status":"numerical_error","backend":"ipm","backend_version":"ipm-cpu-0.53.0",
        "ipm_homogeneous_tau":1.0,"ipm_homogeneous_kappa":1.0,
        "timings":{"read_seconds":0,"solve_seconds":1,"verification_seconds":0,"total_seconds":1},
        **dict.fromkeys(COUNT_FIELDS,0)}
    row = {"error":None,"classification":"incomplete","returncode":4,"configuration_matches":True,
        "report":report,"numerical_seconds":1,"elapsed_wall_seconds":2}
    budget = copy.deepcopy(row); budget["report"]["status"]="time_limit"
    assert pair_result(row,row,plan)["category"]=="equal"
    assert pair_result(row,budget,plan)["category"]=="budget_limited"
    changed = copy.deepcopy(row); changed["report"]["backend_version"]="changed"
    assert pair_result(row,changed,plan)["category"]=="different"
    changed = copy.deepcopy(row); changed["report"]["timings"]["total_seconds"]=2
    assert pair_result(row,changed,plan)["category"]=="equal"
    changed = copy.deepcopy(row); changed["report"]["new_flag"]=True
    other = copy.deepcopy(changed);other["report"]["new_flag"]=1
    assert pair_result(changed,other,plan)["category"]=="different"
    assert not typed_equal({"x":[True]},{"x":[1]}) and typed_equal({"x":[1.0]},{"x":[1]})
    bad_processes=[]
    for key,value in (("returncode",False),("returncode",0),("returncode",-1073741819),("error","wall_timeout"),
        ("error","launch_error"),("configuration_matches",False),("numerical_seconds",None),
        ("numerical_seconds",float("nan")),("elapsed_wall_seconds",0)):
        item=copy.deepcopy(row);item[key]=value;bad_processes.append(item)
    for value in ("bogus",None,"error"):
        item=copy.deepcopy(row);item["report"]["status"]=value;bad_processes.append(item)
    item=copy.deepcopy(row);item["report"]["schema_version"]=True;bad_processes.append(item)
    for item in bad_processes:
        assert pair_result(item,budget,plan)["category"]=="invalid_process"
    count_cases=0
    for field in COUNT_FIELDS:
        for value in (True,False,-1,1.0,MAX_INT64+1):
            item=copy.deepcopy(row);item["report"][field]=value
            assert not valid_process(item);count_cases+=1
        item=copy.deepcopy(row);item["report"].pop(field)
        assert not valid_process(item);count_cases+=1
        item=copy.deepcopy(row);item["report"][field]=None
        assert valid_process(item) is (field in OPTIONAL_COUNT_FIELDS);count_cases+=1
    for field in ABSENT_FIELDS:
        item=copy.deepcopy(row);item["report"][field]=0
        assert not valid_process(item);count_cases+=1
    for values,expected in (((0,0,0),True),((2,1,3),True),((2,3,3),False),((3,0,2),False)):
        item=copy.deepcopy(row)
        item["report"].update(dict(zip(("ipm_homogeneous_constant_repairs","ipm_homogeneous_constant_repair_successes","ipm_homogeneous_constant_solves"),values)))
        assert valid_process(item) is expected;count_cases+=1
    item=copy.deepcopy(row);item["report"].update(ipm_weighted_krylov_steps=1,ipm_refinement_steps=1)
    assert not valid_process(item);count_cases+=1
    status_cases=0
    for status,code,classification in (("optimal",0,"resolved"),("empty",0,"resolved"),("infeasible",0,"resolved"),
        ("unbounded",0,"resolved"),("verification_failed",3,"verification_failed"),("iteration_limit",4,"incomplete"),
        ("time_limit",4,"incomplete"),("numerical_error",4,"incomplete")):
        positive=copy.deepcopy(row);positive.update(returncode=code,classification=classification)
        positive["report"]["status"]=status
        if status in ("optimal","empty"):
            positive["report"].update(has_solution=True,objective=0,solution=[],row_dual=[],
                model={"columns":0,"rows":0,"is_mip":False},
                verification={"valid_input":True,"feasible":True,"objective_consistent":True,"recomputed_objective":0},
                lp_optimality={"valid_input":True,"dual_feasible":True,"gap_satisfied":True,"numerically_optimal":True,
                    "max_dual_residual":0,"dual_objective_estimate":0,"objective_gap":0,"relative_objective_gap":0})
        assert valid_process(positive)
        for field in COUNT_FIELDS:
            item=copy.deepcopy(positive);item.update(returncode=code,classification=classification)
            item["report"]["status"]=status;item["report"][field]=True
            assert not valid_counts(item["report"]) and pair_result(item,budget,plan)["category"]=="invalid_process";status_cases+=1
    cases,scheduled=configurations(plan),list(schedule(plan))
    assert len(cases)==6 and len(scheduled)==18
    for case in cases:
        i=case["case_index"]
        assert [variant for value,variant in scheduled if value["case_index"]==i]==list(VARIANTS[i%3:]+VARIANTS[:i%3])
        cmd=command_for(Path("test.exe"),case,plan)
        flags=dict(zip(cmd[3::2],cmd[4::2]));assert len(flags)*2==len(cmd[3:])
        assert flags["--ipm-linear-solver"]=="normal_cholmod_aat" and flags["--ipm-linear-accuracy"]=="homogeneous_repair"
        assert flags["--ipm-objective-scaling"]=="off" and flags["--ipm-direction-retries"]=="0"
    capture_checks = capture_self_test()
    print(json.dumps({"passed":True,"trajectory_checks":5+len(bad_processes),"typed_checks":2,
        "count_checks":count_cases,"all_status_invalid_budget_checks":status_cases,"capture_checks":capture_checks,
        "commands_checked":len(cases),"schedule_rows":len(scheduled),"lp_runs":0,"filesystem_hashes":0}),flush=True)


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
