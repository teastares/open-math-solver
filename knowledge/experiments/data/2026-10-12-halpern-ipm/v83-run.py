"""第80轮：同因子双右端候选的关闭兼容、开启性能及固定扩集门槛。"""
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
PLAN_SHA = "46e074b8cab1b8f02772cb00b737d9c91d49dd7f90ddc14c99d87a913922308b"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
WORK = ROOT / ".tools/v83-pair"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V83_PAIR "
MARKER_COUNTS = ("pairs_started", "pairs_completed", "normal_calls_started", "normal_calls_completed",
    "finish0_started", "finish0_completed", "finish1_started", "finish1_completed")
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
_spec = importlib.util.spec_from_file_location("v83_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def fixed_plan():
    assert sha(DATA / "v83-plan.json") == PLAN_SHA
    plan = read(DATA / "v83-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_identity_file"], plan["baseline_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert plan["candidate_version"] == 83 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V83_PAIR_RHS"
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
    clean = {key: value for key, value in prior.items() if not key.upper().startswith("SOLVER_V") and key.upper() != "OPENBLAS_CORETYPE"}
    clean.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    if enabled:
        clean["SOLVER_V83_PAIR_RHS"] = "1"
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


def valid_marker(value, enabled):
    fields = {"schema", "enabled", *MARKER_COUNTS}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema"]) is not int or value["schema"] != 1:
        return False
    if type(value["enabled"]) is not bool or value["enabled"] is not enabled:
        return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= 200 for key in MARKER_COUNTS):
        return False
    if not enabled:
        return all(value[key] == 0 for key in MARKER_COUNTS)
    order = ("pairs_started", "normal_calls_started", "normal_calls_completed", "finish0_started",
        "finish0_completed", "finish1_started", "finish1_completed", "pairs_completed")
    if any(value[left] < value[right] for left, right in zip(order, order[1:])):
        return False
    return value["pairs_started"] - value["pairs_completed"] <= 1 and \
        value["finish0_completed"] == value["finish1_started"] and \
        value["finish1_completed"] == value["pairs_completed"]


def valid_process(row):
    if row.get("pair_marker_error") is not None or row.get("configuration_matches") is not True:
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
    if role not in ("normal", "ipm") or not isinstance(checks, dict) or checks.get("passed") is not True or \
        type(checks.get("lp_runs")) is not int or checks["lp_runs"] != 0 or not isinstance(raw, dict):
        return False
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("wall_timeout") is not False \
        or raw.get("command") != [str(work / (role + "-self-test.exe"))] or raw.get("cwd") != str(ROOT):
        return False
    if not isinstance(raw.get("stdout"), str) or not isinstance(raw.get("stderr"), str):
        return False
    try:
        value = benchmark.strict_json(raw["stdout"])
    except (ValueError, TypeError):
        return False
    if not isinstance(value, dict) or value.get("passed") is not True or not base.typed_equal(value, checks.get("details")):
        return False
    if type(value.get("cases")) is not int or value["cases"] <= 0:
        return False
    if role == "normal":
        return type(value.get("pair_cases")) is int and value["pair_cases"] > 0 and \
            isinstance(value.get("pair_details"), list) and len(value["pair_details"]) == value["pair_cases"] and \
            all(isinstance(label, str) and label for label in value["pair_details"]) and \
            value.get("long_column_interrupted") is True and isinstance(value.get("snapshots"), list) and \
            len(value["snapshots"]) == value["cases"]
    return raw["stderr"] == "" and type(value.get("solver_runs")) is int and value["solver_runs"] == 0 and \
        all(type(value.get(key)) is int and value[key] > 0 for key in ("native_cases", "fallback_cases")) and \
        all(type(value.get(key)) is int and value[key] == 1 for key in
            ("finish0_refinement_cases", "repair_cases", "budget_cases"))


def math_evidence_self_check():
    for role in ("normal", "ipm"):
        value = {"passed": True, "cases": 1}
        if role == "normal":
            value.update(pair_cases=1, pair_details=["原方程"], long_column_interrupted=True, snapshots=[{}])
        else:
            value.update(solver_runs=0, native_cases=1, fallback_cases=1,
                finish0_refinement_cases=1, repair_cases=1, budget_cases=1)
        checks = {"passed": True, "lp_runs": 0, "details": value}
        raw = {"command": [str(WORK / (role + "-self-test.exe"))], "cwd": str(ROOT),
            "returncode": 0, "wall_timeout": False, "stdout": json.dumps(value), "stderr": ""}
        assert valid_math_evidence(role, checks, raw, WORK)
        for key, invalid in (("returncode", False), ("returncode", 4), ("wall_timeout", True),
            ("stdout", "{}"), ("stdout", '{"passed":true,"passed":true}'), ("command", []), ("cwd", "wrong")):
            changed = dict(raw); changed[key] = invalid
            assert not valid_math_evidence(role, checks, changed, WORK)
        changed = copy.deepcopy(checks); changed["details"]["cases"] = True
        assert not valid_math_evidence(role, changed, raw, WORK)
        bad = copy.deepcopy(value); bad["pair_cases" if role == "normal" else "repair_cases"] = 0
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
    assert candidate == work / "solver-pair.exe"
    assert set(build["commands"]["compile"]) == {"ipm", "normal"}
    assert build["commands"]["link"].count(str(work / "ipm-pair.obj")) == 1
    assert build["commands"]["link"].count(str(work / "normal-pair.obj")) == 1
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
    for name in ("static-check.json", "normal-self-check.json", "ipm-self-check.json"):
        path = str(work / name)
        assert path in protected and read(path)["passed"] is True
    for role in ("normal", "ipm"):
        assert valid_math_evidence(role, read(work / (role + "-self-check.json")),
            read(work / (role + "-self-run.json")), work), "数学摘要与实际执行原件不同"
    override = str(work / "override/include/solver/detail/normal_cholesky.hpp")
    assert override in build["generated_inputs_before_and_after"] and override in build["compile_inputs_after_and_final"]
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v83-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v83-{stage}-manifest.json")
    summary = read(DATA / f"v83-{stage}-summary.json")
    raw = DATA / f"v83-{stage}-runs.jsonl.gz"
    rows = [benchmark.strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v83-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v83-run-identity.json")
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
        assert row["pair_environment"] == ({"SOLVER_V83_PAIR_RHS": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert row["pair_marker"] == marker and row["pair_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert row["report"] == benchmark.strict_json(row["stdout"])
        for field, value in quality_fields(row, model, digest, plan).items():
            assert row[field] == value, (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), pair_marker_error_count=sum(row["pair_marker_error"] is not None for row in rows))
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
        save_once(DATA / "v83-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v83-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v83-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v83-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V83_PAIR_RHS=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；成批求解计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v83-{stage}-manifest.json", manifest)
    raw = DATA / f"v83-{stage}-runs.jsonl.gz"
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
                pair_environment={"SOLVER_V83_PAIR_RHS": "1"} if variant == "diagnostic_on" else {},
                pair_marker=marker, pair_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "pair_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), pair_marker_error_count=sum(row["pair_marker_error"] is not None for row in rows))
    save(DATA / f"v83-{stage}-summary.json", summary)
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
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V83_PAIR_RHS": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V83_PAIR_RHS"] == "1"
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
    empty = {"schema": 1, "enabled": False, **dict.fromkeys(MARKER_COUNTS, 0)}
    assert valid_marker(empty, False)
    on = dict(empty, enabled=True)
    assert valid_marker(on, True)
    completed = {"schema": 1, "enabled": True, **dict.fromkeys(MARKER_COUNTS, 3)}
    assert valid_marker(completed, True)
    valid = [empty, on, completed]
    order = ("pairs_started", "normal_calls_started", "normal_calls_completed", "finish0_started",
        "finish0_completed", "finish1_started", "finish1_completed", "pairs_completed")
    for length in (1, 2, 3, 4, 6):
        row = dict(on)
        for key in order[:length]:
            row[key] = 1
        assert valid_marker(row, True)
        valid.append(row)
    mutations = []
    for key in ("schema", *MARKER_COUNTS):
        for wrong in (True, -1, 201, 0.5, None):
            row = dict(completed); row[key] = wrong; mutations.append(row)
    for key in ("enabled", *MARKER_COUNTS):
        row = dict(completed); row.pop(key); mutations.append(row)
    for key in MARKER_COUNTS:
        row = dict(on); row[key] = 1
        if not valid_marker(row, True): mutations.append(row)
    mutations += [dict(completed, enabled=1), dict(completed, unexpected=0),
        dict(completed, pairs_started=5), dict(completed, finish1_started=2),
        dict(completed, pairs_completed=2)]
    for row in mutations:
        assert not valid_marker(row, True)
    assert not valid_marker(dict(empty, pairs_started=1), False)
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
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "pair_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    changed = copy.deepcopy(row); changed["report"]["backend_version"] = "other"
    assert pair_result(row, changed, plan)["category"] == "different"
    budget = copy.deepcopy(row); budget["status"] = budget["report"]["status"] = "time_limit"
    assert pair_result(dict(row, pair_marker_error="marker_count"), budget, plan)["category"] == "invalid_process"
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
        save_once(DATA / ("v83-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
