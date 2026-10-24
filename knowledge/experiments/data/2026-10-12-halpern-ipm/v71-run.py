"""第68轮：固定18次基恢复采集诊断及12个完整报告对子，不生成性能扩集结论。"""
import argparse
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
WORK = ROOT / ".tools/v71-diagnostic"
PLAN_SHA = "dd03b4c7cdad5319b1e47fa0f8e67845f7002f85e7b6c33da3cf03b857b55c7b"
HELPER_SHA = "cdc10668c1b75c1c021bdb0f029813971e25b6ba56c3b1fce306be3518d5b016"
VARIANTS = ("frozen_v70_on", "diagnostic_capture_off", "diagnostic_capture_on")
MARKER = "V71_CAPTURE "
RAW = DATA / "v71-runs.jsonl.gz"
MANIFEST = DATA / "v71-run-manifest.json"
SUMMARY = DATA / "v71-summary.json"
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(DATA / "v70-run.py") == HELPER_SHA
_spec = importlib.util.spec_from_file_location("v71_frozen_helpers", DATA / "v70-run.py")
previous = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(previous)
benchmark, compare_lp = previous.benchmark, previous.compare_lp
read, save, typed_equal = previous.read, previous.save, previous.base.typed_equal


def fixed_plan():
    assert sha(DATA / "v71-plan.json") == PLAN_SHA
    plan = read(DATA / "v71-plan.json")
    for field in ("parent_run_identity", "parent_plan", "baseline_identity", "baseline_verification"):
        assert sha(DATA / plan[field + "_file"]) == plan[field + "_sha256"]
    parent = read(DATA / plan["parent_plan_file"])
    assert plan["variants"] == list(VARIANTS) and plan["repeats"] == 2 and plan["runs"] == 18
    assert len(plan["models"]) == 3 and [item["name"] for item in plan["models"]] == ["greenbea", "forplan", "greenbeb"]
    assert plan["production_changes"] is False
    assert plan["kind"] == "diagnostic_only" and plan["maximum_solver_runs"] == 18
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["recovery_environment"] == "SOLVER_V70_BASIS_RECOVERY" and plan["recovery_marker"] == "V70_BASIS"
    assert plan["capture_env"] == "SOLVER_V71_CAPTURE" and plan["capture_magic"] == "V71BAS01"
    assert (plan["cache_limit_bytes"], plan["metadata_limit_bytes"], plan["maximum_events"]) == (134217728, 65536, 1)
    for key in ("time_limit", "wall_timeout", "threads", "seed", "max_iterations", "scaling_passes", "method",
        "shared_ipm", "expected_report_options"):
        assert typed_equal(plan[key], parent[key]), key
    assert (plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"]) == (5, 35, 1, 0)
    return plan, parent


def schedule(plan):
    for model_index, model in enumerate(plan["models"]):
        for repetition in range(plan["repeats"]):
            case_index = model_index * plan["repeats"] + repetition
            offset = case_index % len(VARIANTS)
            for variant in VARIANTS[offset:] + VARIANTS[:offset]:
                yield {"case_index": case_index, "model_index": model_index, "model": model,
                    "repetition": repetition}, variant


def capture_valid(metadata, marker, binary, metadata_bytes, plan):
    metadata_keys = {"schema", "version", "status", "attempts", "event_count", "capture_errors", "truncated", "error",
        "n", "m", "nnz", "fixed_cache_bytes", "payload_bytes", "cache_bytes", "cache_limit_bytes", "metadata_limit_bytes",
        "binary_bytes", "byte_order", "events"}
    marker_keys = {"schema", "version", "status", "attempts", "event_count", "capture_errors", "truncated", "binary_bytes", "write_ok"}
    if not isinstance(metadata, dict) or metadata.keys() != metadata_keys or not isinstance(marker, dict) or marker.keys() != marker_keys:
        return False
    counts = ("schema", "attempts", "event_count", "capture_errors", "n", "m", "nnz", "fixed_cache_bytes", "payload_bytes",
        "cache_bytes", "cache_limit_bytes", "metadata_limit_bytes", "binary_bytes")
    if any(type(metadata[key]) is not int or not 0 <= metadata[key] <= 9223372036854775807 for key in counts):
        return False
    if any(type(marker[key]) is not int for key in ("schema", "attempts", "event_count", "capture_errors", "binary_bytes")):
        return False
    if metadata["schema"] != 1 or metadata["version"] != "v71" or metadata["truncated"] is not False \
        or metadata["capture_errors"] != 0 or metadata["error"] != "none" or marker["write_ok"] is not True \
        or marker["truncated"] is not False or metadata["byte_order"] not in ("little", "big"):
        return False
    if any(not typed_equal(marker[key], metadata[key]) for key in marker_keys - {"write_ok"}):
        return False
    if metadata["cache_limit_bytes"] != plan["cache_limit_bytes"] or metadata["metadata_limit_bytes"] != plan["metadata_limit_bytes"] \
        or not 0 < metadata["fixed_cache_bytes"] <= metadata["cache_bytes"] <= plan["cache_limit_bytes"] \
        or not 0 < metadata_bytes <= plan["metadata_limit_bytes"] \
        or metadata["binary_bytes"] != len(binary) or binary[:8] != plan["capture_magic"].encode("ascii"):
        return False
    if metadata["status"] == "no_candidate":
        return all(metadata[key] == 0 for key in ("attempts", "event_count", "n", "m", "nnz", "payload_bytes")) \
            and metadata["cache_bytes"] == metadata["fixed_cache_bytes"] and len(binary) == 8 and metadata["events"] == []
    if metadata["status"] != "captured" or metadata["attempts"] != 1 or metadata["event_count"] != 1 \
        or not isinstance(metadata["events"], list) or len(metadata["events"]) != 1:
        return False
    event = metadata["events"][0]
    if not isinstance(event, dict) or event.keys() != {"index", "phase", "arrays"} or type(event["index"]) is not int \
        or event["index"] != 0 or event["phase"] != "recovered_basis_before_assess" or not isinstance(event["arrays"], list):
        return False
    n, m, nnz = (metadata[key] for key in ("n", "m", "nnz"))
    specification = (("g_starts", "int64", n + 1), ("g_rows", "int64", nnz), ("g_values", "float64", nnz),
        ("costs", "float64", n), ("rhs", "float64", m), ("slack", "float64", m), ("dual", "float64", m),
        ("inequality", "uint8", m), ("candidate_x", "float64", n), ("candidate_s", "float64", m),
        ("candidate_z", "float64", m), ("selected_rows", "int64", n), ("pivot_columns", "int64", n))
    if len(event["arrays"]) != len(specification):
        return False
    offset = 8
    for field, (name, dtype, count) in zip(event["arrays"], specification):
        width = 1 if dtype == "uint8" else 8
        if not isinstance(field, dict) or field.keys() != {"name", "dtype", "count", "width", "offset", "shape"} \
            or any(type(field[key]) is not int for key in ("count", "width", "offset")) \
            or not isinstance(field["shape"], list) or len(field["shape"]) != 1 or type(field["shape"][0]) is not int \
            or field != {"name": name, "dtype": dtype, "count": count, "width": width, "offset": offset, "shape": [count]}:
            return False
        offset += count * width
    return offset == len(binary) and metadata["payload_bytes"] == offset - 8 \
        and metadata["cache_bytes"] == metadata["fixed_cache_bytes"] + metadata["payload_bytes"]


def inspect_capture(prefix, enabled, stderr, plan):
    metadata_path, binary_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".bin")
    value = {"enabled": enabled, "prefix": str(prefix), "metadata_path": str(metadata_path) if enabled else None,
        "binary_path": str(binary_path) if enabled else None, "metadata_sha256": None, "binary_sha256": None,
        "metadata_bytes": None, "binary_bytes": None, "metadata": None, "marker": None}
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if not enabled:
        error = "unexpected_capture" if lines or metadata_path.exists() or binary_path.exists() else None
        return value, error
    errors = []
    if len(lines) != 1:
        errors.append("capture_marker_count")
    else:
        try:
            value["marker"] = benchmark.strict_json(lines[0][len(MARKER):])
        except (ValueError, TypeError):
            errors.append("capture_marker_json")
    for label, path, limit in (("metadata", metadata_path, plan["metadata_limit_bytes"]),
        ("binary", binary_path, plan["cache_limit_bytes"] + 8)):
        if not path.is_file():
            errors.append("capture_" + label + "_missing")
            continue
        value[label + "_sha256"], value[label + "_bytes"] = sha(path), path.stat().st_size
        if value[label + "_bytes"] > limit:
            errors.append("capture_" + label + "_limit")
        elif label == "metadata":
            try:
                value["metadata"] = read(path)
            except (ValueError, TypeError):
                errors.append("capture_metadata_json")
    if not errors and not capture_valid(value["metadata"], value["marker"], binary_path.read_bytes(), value["metadata_bytes"], plan):
        errors.append("capture_invalid_or_incomplete")
    return value, ";".join(errors) if errors else None


def configuration_matches(report, model, plan):
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == "ipm" \
        and report.get("backend_version") == "ipm-cpu-0.53.0" \
        and typed_equal(report.get("options"), plan["expected_report_options"])


def capture_relation_error(capture, basis_marker, error):
    if error is not None or not capture["enabled"]:
        return error
    if not isinstance(basis_marker, dict):
        return "capture_basis_marker_missing"
    metadata = capture["metadata"]
    available = basis_marker.get("candidate_available")
    if metadata["status"] == "no_candidate":
        return None if available is False else "capture_no_candidate_mismatch"
    stats = basis_marker.get("stats")
    if available is not True or not isinstance(stats, dict) or any(metadata[key] != stats.get(field)
        for key, field in (("n", "n"), ("m", "m"), ("nnz", "input_nonzeros"))):
        return "capture_basis_dimensions_mismatch"
    return None


def pair_result(a, b, plan):
    value = previous.pair_result(a, b, {**plan, "compatibility_excluded_report_fields": ["timings"]})
    value["capture_valid"] = a["capture_error"] is None and b["capture_error"] is None
    if value["category"] == "equal" and not value["capture_valid"]:
        value["category"] = "capture_error"
    basis = [copy.deepcopy(row["basis_marker"]) for row in (a, b)]
    for marker in basis:
        if isinstance(marker, dict):
            marker.pop("attempt_seconds", None)
            marker.pop("verification_seconds", None)
    value["basis_sidecar_equal_excluding_durations"] = typed_equal(*basis)
    return value


def initial_identity(work, plan):
    parent_path = DATA / plan["parent_run_identity_file"]
    parent = read(parent_path)
    assert Path(parent["candidate"]) == ROOT / ".tools/v70-basis/solver-diagnostic.exe"
    assert sha(parent["build_manifest"]) == parent["build_manifest_sha256"]
    assert parent["build_manifest_sha256"] == plan["reference_build_manifest_sha256"]
    assert parent["candidate_sha256"] == plan["reference_solver_sha256"]
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    assert build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    assert build["reference_executable"] == parent["candidate"] and build["reference_sha256"] == parent["candidate_sha256"]
    assert build["reference_build_manifest"] == parent["build_manifest"]
    assert build["reference_build_manifest_sha256"] == parent["build_manifest_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-diagnostic.exe"
    protected = {str(DATA / "v70-run.py"): HELPER_SHA, str(Path(__file__)): sha(__file__),
        str(parent_path): sha(parent_path), str(build_path): sha(build_path),
        parent["build_manifest"]: parent["build_manifest_sha256"], parent["candidate"]: parent["candidate_sha256"],
        str(candidate): build["executable_sha256"]}
    for name in ("benchmark.py", "compare_lp.py"):
        path = str(ROOT / "scripts" / name)
        protected[path] = parent["protected"][path]
    for path, digest in build["protected_before_and_after"].items():
        if Path(path).parent == DATA and Path(path).suffix in (".py", ".hpp", ".cpp"):
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for path, digest in protected.items():
        assert sha(path) == digest, path
    return {"reference": parent["candidate"], "reference_sha256": parent["candidate_sha256"],
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path), "protected": protected}


def archive():
    plan, parent = fixed_plan()
    manifest = read(MANIFEST)
    identity = initial_identity(Path(manifest["build_manifest"]).parent, plan)
    assert all(manifest[key] == value for key, value in identity.items())
    assert manifest["plan_sha256"] == PLAN_SHA and manifest["models"] == plan["models"]
    assert manifest["variants"] == list(VARIANTS) and manifest["repeats"] == plan["repeats"]
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[key] == plan[key]
    rows = [benchmark.strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    ordered = list(schedule(plan))
    assert len(rows) == len(ordered) == 18
    for index, (row, (case, variant)) in enumerate(zip(rows, ordered)):
        model = case["model"]
        side = "reference" if variant == "frozen_v70_on" else "candidate"
        executable, digest = manifest[side], manifest[side + "_sha256"]
        assert (row["run_index"], row["case_index"], row["model_index"], row["repetition"], row["requested_variant"]) == \
            (index, case["case_index"], case["model_index"], case["repetition"], variant)
        assert row["input"] == model["input"] and row["seed"] == plan["seed"] and row["requested_backend"] == "ipm"
        assert row["command"][:3] == [executable, "solve", model["input"]]
        assert previous.valid_command(row, "diagnostic_on", parent)
        assert all(row[key] == digest for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[key] == model["sha256"] for key in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        enabled = variant == "diagnostic_capture_on"
        wanted_environment = {"SOLVER_V70_BASIS_RECOVERY": "1"}
        if enabled:
            wanted_environment[plan["capture_env"]] = row["capture"]["prefix"]
        assert row["diagnostic_environment"] == wanted_environment
        marker, error = previous.inspect_marker(row["stderr"], "diagnostic_on")
        assert row["basis_marker"] == marker and row["basis_marker_error"] == error
        capture, error = inspect_capture(Path(row["capture"]["prefix"]), enabled, row["stderr"], plan)
        error = capture_relation_error(capture, marker, error)
        assert row["capture"] == capture and row["capture_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], model, plan)
        if row["report"] is not None:
            assert row["report"] == benchmark.strict_json(row["stdout"])
        for key, value in previous.quality_fields(row, model, digest, plan).items():
            assert row[key] == value, (index, key)
    pairs = []
    for case_index in range(6):
        selected = {row["requested_variant"]: row for row in rows if row["case_index"] == case_index}
        assert set(selected) == set(VARIANTS)
        for a, b in ((VARIANTS[0], VARIANTS[1]), (VARIANTS[1], VARIANTS[2])):
            pairs.append({"case_index": case_index, "model": selected[a]["model"], "repetition": selected[a]["repetition"],
                "reference": a, "candidate": b, **pair_result(selected[a], selected[b], plan)})
    summary = {"diagnostic_only": True, "plan_sha256": PLAN_SHA, "run_count": len(rows), "pair_count": len(pairs),
        "pairs": pairs, "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["budget_limited"] for pair in pairs),
        "invalid_process_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs),
        "capture_error_count": sum(row["capture_error"] is not None for row in rows),
        "quality_pass_count": sum(row["quality_pass"] for row in rows), "raw_sha256": sha(RAW),
        "capture_files": [row["capture"] for row in rows if row["capture"]["enabled"]],
        "scope": "全部18次均启用原v70基恢复；采集复制计入5秒预算，写出计入35秒墙钟；不扩集、不运行oracle、不产生性能接纳结论。"}
    summary["diagnostic_trajectory_verified"] = summary["non_budget_equal"] == 12 and summary["capture_error_count"] == 0
    return summary, rows


def run(work):
    plan, parent = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY)), "禁止覆盖本轮既有记录"
    identity = initial_identity(work, plan)
    capture_root = work / "captures"
    capture_root.mkdir()
    for model in plan["models"]:
        assert sha(model["input"]) == model["sha256"]
    manifest = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        **identity, "models": plan["models"], "variants": list(VARIANTS), "repeats": plan["repeats"],
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")}, "capture_root": str(capture_root),
        "environment_rule": "每次先清除所有SOLVER_V前缀与OPENBLAS_CORETYPE；三状态均设置V70恢复=1，仅采集on另设置V71前缀路径；单线程，调用后恢复原环境。",
        "order_rule": "每模型两重复，三状态按case_index模3向左轮转，全部串行。"}
    save(MANIFEST, manifest)
    with gzip.open(RAW, "xt", encoding="utf-8") as stream:
        for index, (case, variant) in enumerate(schedule(plan)):
            model = case["model"]
            side = "reference" if variant == "frozen_v70_on" else "candidate"
            executable, digest = Path(identity[side]), identity[side + "_sha256"]
            prefix = capture_root / (str(case["case_index"]) + "-" + variant)
            enabled = variant == "diagnostic_capture_on"
            environment = {"SOLVER_V70_BASIS_RECOVERY": "1"}
            if enabled:
                environment[plan["capture_env"]] = str(prefix)
            before = sha(executable)
            with previous.solver_environment(True):
                os.environ.update(environment)
                row = benchmark.run_one(previous.options_for("probe", "diagnostic_on", executable, parent),
                    Path(model["input"]), plan["seed"], index)
            marker, error = previous.inspect_marker(row["stderr"], "diagnostic_on")
            capture, capture_error = inspect_capture(prefix, enabled, row["stderr"], plan)
            capture_error = capture_relation_error(capture, marker, capture_error)
            row.update(case_index=case["case_index"], model_index=case["model_index"], model=model["name"],
                repetition=case["repetition"], requested_variant=variant, diagnostic_environment=environment,
                frozen_model_sha256=model["sha256"], model_sha256_after=sha(model["input"]),
                solver_sha256_before=before, solver_sha256_after=sha(executable), frozen_solver_sha256=digest,
                basis_marker=marker, basis_marker_error=error, capture=capture, capture_error=capture_error,
                configuration_matches=configuration_matches(row["report"], model, plan))
            row.update(previous.quality_fields(row, model, digest, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"index": index, "model": model["name"], "variant": variant, "status": row["status"],
                "capture_error": capture_error, "quality_pass": row["quality_pass"]}, ensure_ascii=False), flush=True)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    summary, _ = archive()
    save(SUMMARY, summary)
    print(json.dumps({key: summary[key] for key in ("run_count", "pair_count", "non_budget_equal", "budget_pairs",
        "invalid_process_pairs", "different_pairs", "capture_error_count", "diagnostic_trajectory_verified")}, ensure_ascii=False), flush=True)


def self_check():
    plan, _ = fixed_plan()
    ordered = list(schedule(plan))
    assert len(ordered) == 18
    for case_index in range(6):
        assert len({variant for case, variant in ordered if case["case_index"] == case_index}) == 3
    control = {"schema": 1, "version": "v71", "status": "no_candidate", "attempts": 0, "event_count": 0,
        "capture_errors": 0, "truncated": False, "error": "none", "n": 0, "m": 0, "nnz": 0,
        "fixed_cache_bytes": 512, "payload_bytes": 0, "cache_bytes": 512,
        "cache_limit_bytes": 134217728, "metadata_limit_bytes": 65536, "binary_bytes": 8, "byte_order": "little", "events": []}
    def marker(metadata):
        return {**{key: metadata[key] for key in ("schema", "version", "status", "attempts", "event_count", "capture_errors",
            "truncated", "binary_bytes")}, "write_ok": True}
    magic = b"V71BAS01"
    assert capture_valid(control, marker(control), magic, 512, plan)
    captured = copy.deepcopy(control)
    captured.update(status="captured", attempts=1, event_count=1, n=1, m=1, nnz=1)
    event = {"index": 0, "phase": "recovered_basis_before_assess", "arrays": []}
    offset = 8
    for name, dtype, count in (("g_starts", "int64", 2), ("g_rows", "int64", 1), ("g_values", "float64", 1),
        ("costs", "float64", 1), ("rhs", "float64", 1), ("slack", "float64", 1), ("dual", "float64", 1),
        ("inequality", "uint8", 1), ("candidate_x", "float64", 1), ("candidate_s", "float64", 1),
        ("candidate_z", "float64", 1), ("selected_rows", "int64", 1), ("pivot_columns", "int64", 1)):
        width = 1 if dtype == "uint8" else 8
        event["arrays"].append({"name": name, "dtype": dtype, "count": count, "width": width, "offset": offset, "shape": [count]})
        offset += count * width
    captured.update(events=[event], payload_bytes=offset - 8, binary_bytes=offset, cache_bytes=512 + offset - 8)
    binary = magic + bytes(offset - 8)
    assert capture_valid(captured, marker(captured), binary, 4096, plan)
    mutations = [(("schema",), True), (("truncated",), True), (("capture_errors",), 1), (("status",), "capture_error"),
        (("cache_bytes",), 134217729), (("payload_bytes",), 0), (("events", 0, "arrays", 0, "offset"), 9),
        (("events", 0, "arrays", 0, "width"), 4), (("events", 0, "arrays", 0, "shape"), [2.0]),
        (("events", 0, "arrays", 7, "dtype"), "float64"), (("events", 0, "arrays", 12, "count"), True)]
    for path, value in mutations:
        broken = copy.deepcopy(captured)
        current = broken
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = value
        assert not capture_valid(broken, marker(broken), binary, 4096, plan), path
    assert not capture_valid(captured, {**marker(captured), "write_ok": False}, binary, 4096, plan)
    assert not capture_valid(captured, marker(captured), binary[:-1], 4096, plan)
    assert not capture_valid(control, marker(control), b"WRONG000", 512, plan)
    assert capture_relation_error({"enabled": True, "metadata": control}, {"candidate_available": False}, None) is None
    assert capture_relation_error({"enabled": True, "metadata": control}, {"candidate_available": True}, None) is not None
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(previous.base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True,
        "basis_marker_error": None, "basis_marker": {}, "capture_error": None}
    assert pair_result(row, row, plan)["category"] == "equal"
    changed = copy.deepcopy(row)
    changed["report"]["backend_version"] = "different"
    assert pair_result(row, changed, plan)["category"] == "different"
    missing = dict(row, capture_error="capture_marker_count")
    assert pair_result(row, missing, plan)["category"] == "capture_error"
    expired = copy.deepcopy(row)
    expired["status"] = expired["report"]["status"] = "time_limit"
    assert pair_result(row, expired, plan)["category"] == "budget_limited"
    invalid = copy.deepcopy(row)
    invalid["report"]["ipm_factorizations"] = True
    assert pair_result(invalid, expired, plan)["category"] == "invalid_process"
    prior = dict(os.environ)
    with previous.solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V70_BASIS_RECOVERY": "1"}
        os.environ[plan["capture_env"]] = "self-check-only"
        assert os.environ[plan["capture_env"]] == "self-check-only"
    assert dict(os.environ) == prior
    return {"passed": True, "solver_runs": 0, "scheduled_runs_checked": 18, "capture_cases": 2 + len(mutations) + 3,
        "capture_relation_cases": 2, "compatibility_cases": 5, "environment_cases": 2,
        "reused_validation_helper_sha256": HELPER_SHA}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "run", "archive"))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage == "archive":
        result, _ = archive()
        previous.save_once(SUMMARY, result)
        print(json.dumps({key: result[key] for key in ("run_count", "non_budget_equal", "diagnostic_trajectory_verified")}, ensure_ascii=False))
    else:
        run(args.work.resolve())
