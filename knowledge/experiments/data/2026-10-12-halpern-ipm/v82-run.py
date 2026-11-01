"""第79轮v66全LP阶段诊断：17例五模式轮转，保留失败和完整报告，不优化算法。"""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
import re
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v82-diagnostic"
PLAN_SHA = "b5df77438c3905566b0a8ddb02b84c2c2f65484facfe26bf559ec2ae085ec44b"
RAW = DATA / "v82-runs.jsonl.gz"
MANIFEST = DATA / "v82-run-manifest.json"
SUMMARY = DATA / "v82-summary.json"
MODES = ("production", "off", "on1", "on2", "on3")
FIELDS = ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")
ENVIRONMENT = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            assert key not in result, "JSON对象含重复字段：" + key
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=pairs, parse_constant=benchmark.invalid_constant,
        parse_float=benchmark.finite_json_float)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def fixed_plan():
    assert sha(DATA / "v82-plan.json") == PLAN_SHA
    plan = read(DATA / "v82-plan.json")
    assert sha(DATA / "ipm-v66-identity.json") == plan["parent_identity_sha256"]
    assert plan["diagnostic_only"] is True
    assert (plan["round"], plan["version"], plan["parent_version"]) == (79, "v82", 66)
    assert (plan["method"], plan["primary_linear_solver"], plan["linear_accuracy"], plan["objective_scaling"],
        plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
        plan["max_iterations"], plan["scaling_passes"]) == (
        "homogeneous", "normal_cholmod_aat", "homogeneous_repair", "off", 5, 35, 1, 0, 200, 5)
    assert plan["modes"] == list(MODES) and type(plan["runs"]) is int and plan["runs"] == 85
    assert len(plan["models"]) == 17 and [item["model_index"] for item in plan["models"]] == list(range(17))
    assert len({item["input"] for item in plan["models"]}) == 17
    assert all(type(item["model_index"]) is int and item["hash_error"] is None and
        re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in plan["models"])
    assert (plan["profile_environment"], plan["profile_enabled_value"], plan["profile_prefix"]) == (
        "SOLVER_V47_PROFILE", "1", "V47_PROFILE ")
    assert len(plan["profile_stages"]) == len(set(plan["profile_stages"])) == 35
    assert plan["profile_stages"][-3:] == ["normal_ordering", "aat_scaling", "cholmod_aat_numeric"]
    assert plan["transformed_units"] == {"ipm": "src/ipm.cpp", "normal": "src/normal_cholesky.cpp"}
    assert plan["shared_ipm"] == {"singleton-elimination": "on", "dynamic-regularization": "off",
        "kkt-equilibration": "on", "fixed-variable-elimination": "on", "singleton-presolve": "off",
        "presolve-fixed-elimination": "off", "krylov-history-size": "0", "central-corrections": "0",
        "inexact-prefilter": "off"}
    return plan


def command_for(executable, model, plan):
    command = [str(executable), "solve", model["input"], "--backend", "ipm",
        "--time-limit", str(plan["time_limit"]), "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
        "--iteration-limit", str(plan["max_iterations"]), "--scaling-passes", str(plan["scaling_passes"]),
        "--ipm-method", plan["method"], "--ipm-linear-accuracy", plan["linear_accuracy"],
        "--ipm-linear-solver", plan["primary_linear_solver"], "--ipm-direction-retries", "0",
        "--ipm-objective-scaling", plan["objective_scaling"]]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def valid_process(row):
    expected = {"resolved": 0, "incomplete": 4, "verification_failed": 3}
    if row.get("error") is not None or type(row.get("returncode")) is not int or not isinstance(row.get("report"), dict):
        return False
    actual = benchmark.classify(row["returncode"], row["report"])
    timings = row["report"].get("timings")
    return actual in expected and row.get("classification") == actual and row["returncode"] == expected[actual] and (
        benchmark.finite_real(row.get("elapsed_wall_seconds")) and row["elapsed_wall_seconds"] >= 0 and
        isinstance(timings, dict) and all(benchmark.finite_real(timings.get(name)) and timings[name] >= 0
            for name in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")) and
        benchmark.finite_real(row.get("numerical_seconds")) and
        row["numerical_seconds"] == timings["solve_seconds"] + timings["verification_seconds"])


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row["report"]) for row in (a, b)]
    for report in reports:
        if report is not None:
            report.pop("timings", None)
    # JSON序列化保留bool与整数等类型差异；仅排除timings，版本和全部数值字段仍须一致。
    equal = json.dumps(reports[0], sort_keys=True, allow_nan=False) == json.dumps(reports[1], sort_keys=True, allow_nan=False) and (
        a["returncode"] == b["returncode"] and a["error"] == b["error"])
    valid = valid_process(a) and valid_process(b)
    budget = any(row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or
        row["elapsed_wall_seconds"] > plan["wall_timeout"] or row["numerical_seconds"] is not None and
        row["numerical_seconds"] > plan["time_limit"] for row in (a, b))
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def parse_profile(stderr, enabled, plan):
    assert isinstance(stderr, str) and type(enabled) is bool
    lines = [line for line in stderr.splitlines() if line.startswith(plan["profile_prefix"])]
    assert len(lines) == int(enabled), "计时报告数量不符"
    if not enabled:
        return None
    payload = strict_json(lines[0][len(plan["profile_prefix"]):])
    assert isinstance(payload, dict) and set(payload) == {"schema", "stages", "clock_reads", "exclusive_sum_ns",
        "total_inclusive_ns", "conserved", "stack_empty", "counts_closed"}
    assert type(payload["schema"]) is int and payload["schema"] == 1
    assert all(payload[key] is True for key in ("conserved", "stack_empty", "counts_closed"))
    stages = payload["stages"]
    assert isinstance(stages, dict) and list(stages) == plan["profile_stages"]
    for stage in stages.values():
        assert set(stage) == {"inclusive_ns", "exclusive_ns", "started", "completed", "unwound"}
        assert all(type(value) is int and 0 <= value < 2**63 for value in stage.values())
        assert stage["started"] == stage["completed"] + stage["unwound"]
        assert stage["exclusive_ns"] <= stage["inclusive_ns"]
        if stage["started"] == 0:
            assert stage["inclusive_ns"] == stage["exclusive_ns"] == 0
    assert stages["total"]["started"] == 1
    assert all(type(payload[key]) is int and 0 <= payload[key] < 2**63 for key in ("clock_reads", "exclusive_sum_ns", "total_inclusive_ns"))
    assert payload["clock_reads"] == 2 * sum(item["started"] for item in stages.values())
    assert sum(item["exclusive_ns"] for item in stages.values()) == stages["total"]["inclusive_ns"]
    assert payload["exclusive_sum_ns"] == payload["total_inclusive_ns"] == stages["total"]["inclusive_ns"]
    return payload


def run_environment(source, enabled, plan):
    # Windows环境名不区分大小写；每次调用重新清除继承开关，避免on污染后续off。
    env = {key: value for key, value in source.items()
        if not key.upper().startswith("SOLVER_V") and key.upper() != "OPENBLAS_CORETYPE"}
    env.update(ENVIRONMENT)
    if enabled:
        env[plan["profile_environment"]] = plan["profile_enabled_value"]
    return env


def validate_timer_run(raw, enabled, work, plan):
    assert raw["command"] == [str(work / "profile-self-test.exe")] and Path(raw["cwd"]) == ROOT
    assert type(raw["returncode"]) is int and raw["returncode"] == 0 and raw["wall_timeout"] is False
    assert isinstance(raw["stdout"], str)
    assert json.dumps(strict_json(raw["stdout"]), sort_keys=True) == json.dumps(
        {"passed": True, "enabled": enabled}, sort_keys=True)
    payload = parse_profile(raw["stderr"], enabled, plan)
    if enabled:
        # 锁定原双翻译单元假时钟：正常返回、异常展开和父子独占账本均已实际覆盖。
        assert payload["clock_reads"] == 12 and payload["total_inclusive_ns"] == 110
        for name, values in (("total", (110, 20, 1, 1, 0)),
            ("canonicalize", (110, 50, 3, 2, 1)), ("normal_setup", (60, 40, 2, 1, 1))):
            assert payload["stages"][name] == dict(zip(FIELDS, values))
        assert all(entry["started"] == 0 for name, entry in payload["stages"].items()
            if name not in ("total", "canonicalize", "normal_setup"))
    else:
        assert raw["stderr"] == ""


def validate_timer_evidence(work, plan):
    proof = read(work / "timer-self-check.json")
    expected = {"passed": True, "solver_runs": 0, "off": {"passed": True, "enabled": False},
        "on": {"passed": True, "enabled": True}}
    assert json.dumps(proof, sort_keys=True) == json.dumps(expected, sort_keys=True)
    for mode, enabled in (("off", False), ("on", True)):
        validate_timer_run(read(work / ("self-" + mode + "-run.json")), enabled, work, plan)
    static = read(work / "static-check.json")
    assert static["passed"] is True and static["stages"] == plan["profile_stages"]
    header = (work / "v47-profile.hpp").read_text(encoding="utf-8")
    assert static["header_sha256"] == hashlib.sha256(header.encode("utf-8")).hexdigest()


def freeze_build(work, plan, build_path, build):
    identity_path = DATA / "ipm-v66-identity.json"
    identity = read(identity_path)
    production, diagnostic = Path(build["production_executable"]), Path(build["executable"])
    parent = ROOT / ".tools/solver-ipm-v66.exe"
    assert production == ROOT / "build/release/solver.exe" and diagnostic == work / "solver-profile.exe"
    assert build["diagnostic_only"] is True and build["production_changes"] is False
    assert build["plan_sha256"] == PLAN_SHA
    assert build["production_sha256"] == identity["solver_sha256"] == plan["parent_solver_sha256"]
    assert Path(build["frozen_parent_executable"]) == parent
    assert build["frozen_parent_sha256"] == identity["solver_sha256"]
    assert build["frozen_files"] == identity["files"]
    assert build["current_files"] == identity["files"] and build["actual_libraries"] == identity["libraries"]
    assert identity["source_zip_sha256"] == plan["parent_source_sha256"]
    assert set(build["commands"]["compile"]) == set(build["commands"]["original_compile"]) == {"ipm", "normal"}
    protected = {}

    def protect(path, expected):
        path = Path(path).resolve()
        assert isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
        key = str(path)
        assert key not in protected or protected[key] == expected, "冻结来源哈希冲突：" + key
        assert sha(path) == expected, "冻结来源已变化：" + key
        protected[key] = expected

    for name in ("protected_before_and_after", "link_explicit_before_and_after", "compile_inputs_after_and_final",
        "generated_inputs_before_and_after", "outputs"):
        values = build[name]
        assert isinstance(values, dict) and values
        for path, expected in values.items():
            protect(path, expected)
    for group in ("files", "libraries"):
        for name, expected in identity[group].items():
            protect(ROOT / name, expected)
    for name, expected in identity["files"].items():
        protect(work / "frozen" / name, expected)
    for path, expected in ((Path(__file__), sha(__file__)), (build_path, sha(build_path)),
        (production, build["production_sha256"]), (diagnostic, build["executable_sha256"]),
        (parent, identity["solver_sha256"]), (identity_path, plan["parent_identity_sha256"]),
        (DATA / "ipm-v66-source.zip", identity["source_zip_sha256"])):
        protect(path, expected)
    assert build["protected_before_and_after"][str(Path(__file__).resolve())] == protected[str(Path(__file__).resolve())]
    required = ("v47-profile.hpp", "ipm-profile.cpp", "normal-profile.cpp", "ipm-profile.obj", "normal-profile.obj",
        "static-check.json", "transforms.json", "commands.json", "self-main.cpp", "self-cross.cpp",
        "self-main.obj", "self-cross.obj", "profile-self-test.exe", "self-off-run.json", "self-on-run.json",
        "timer-self-check.json", "solver-profile.exe")
    assert all(str(work / name) in build["outputs"] for name in required)
    # build后编写的独立核验器不参与编译或运行，不能反向成为尚不存在的输入。
    for name in (name for name in plan["new_code_and_plan_files"] if name != "v82-verify.py"):
        path = DATA / name
        assert str(path) in build["protected_before_and_after"]
        protect(path, build["protected_before_and_after"][str(path)])
    for prefix in ("baseline_identity", "baseline_verification"):
        path, expected = DATA / plan[prefix + "_file"], plan[prefix + "_sha256"]
        assert Path(build[prefix + "_file"]) == path and build[prefix + "_sha256"] == expected
        protect(path, expected)
    for name, expected in ((plan["evidence_file"], plan["evidence_sha256"]),
        ("v47-profile-build.py", plan["helper_sha256"])):
        protect(DATA / name, expected)
    for item in plan["local_references"]:
        protect(ROOT / item["file"], item["sha256"])
    for model in plan["models"]:
        protect(model["input"], model["sha256"])
    validate_timer_evidence(work, plan)
    return protected


def summarize_rows(rows, plan):
    assert len(rows) == plan["runs"]
    expected = [(model["model_index"], variant) for model in plan["models"]
        for variant in plan["modes"][model["model_index"] % 5:] + plan["modes"][:model["model_index"] % 5]]
    assert [row["index"] for row in rows] == list(range(plan["runs"]))
    assert [(row["case_index"], row["variant"]) for row in rows] == expected
    pairs, cases = [], []
    for model in plan["models"]:
        group = [row for row in rows if row["case_index"] == model["model_index"]]
        selected = {row["variant"]: row for row in group}
        assert len(group) == len(selected) == len(plan["modes"]) and set(selected) == set(plan["modes"])
        for left, right in (("production", "off"), ("off", "on1"), ("off", "on2"), ("off", "on3")):
            pairs.append({"case_index": model["model_index"], "model": Path(model["input"]).stem,
                "left": left, "right": right, **pair_result(selected[left], selected[right], plan)})
        enabled = [selected[name] for name in ("on1", "on2", "on3")]
        valid_profiles = all(row["profile_error"] is None and row["profile"] is not None for row in enabled)
        medians = {name: {key: statistics.median(row["profile"]["stages"][name][key] for row in enabled)
            for key in ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")}
            for name in plan["profile_stages"]} if valid_profiles else None
        cases.append({"case_index": model["model_index"], "model": Path(model["input"]).stem,
            "status": {name: (row["report"] or {}).get("status") for name, row in selected.items()},
            "numerical_seconds": {name: row["numerical_seconds"] for name, row in selected.items()},
            "stage_medians": medians})
    return {"diagnostic_only": True, "runs": len(rows), "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs),
        "profile_errors": sum(row["profile_error"] is not None for row in rows),
        "scope": "仅诊断成本；阶段中位数分别计算不承诺加总守恒，不作为公开性能或新覆盖。"}


def summarize():
    plan = fixed_plan()
    rows = [strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    summary = summarize_rows(rows, plan)
    summary["raw_sha256"] = sha(RAW)
    save_new(SUMMARY, summary)
    assert summary["invalid_pairs"] == summary["different_pairs"] == summary["profile_errors"] == 0, "诊断合同不成立，全部记录已保留"
    print(json.dumps({key: summary[key] for key in ("runs", "non_budget_pairs", "non_budget_equal", "budget_pairs", "profile_errors")}), flush=True)


def run(work):
    plan = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY))
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    production, diagnostic = Path(build["production_executable"]), Path(build["executable"])
    identities = {str(production): build["production_sha256"], str(diagnostic): build["executable_sha256"]}
    protected = freeze_build(work, plan, build_path, build)
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "build_manifest": str(build_path), "protected": protected, "environment": ENVIRONMENT,
        "removed_environment": "清除所有SOLVER_V开头变量及OPENBLAS_CORETYPE；仅on设置冻结计时开关",
        "work": str(work), "schedule": plan["order"]})
    index = 0
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for model in plan["models"]:
            offset = model["model_index"] % len(plan["modes"])
            for variant in plan["modes"][offset:] + plan["modes"][:offset]:
                executable = production if variant == "production" else diagnostic
                enabled = variant.startswith("on")
                command = command_for(executable, model, plan)
                env = run_environment(os.environ, enabled, plan)
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
                    report = strict_json(stdout)
                    if not isinstance(report, dict):
                        report, error = None, error or "invalid_report"
                except (AssertionError, ValueError, TypeError):
                    report, error = None, error or "invalid_json"
                profile, profile_error = None, None
                try:
                    profile = parse_profile(stderr, enabled, plan)
                except (AssertionError, ValueError, KeyError, TypeError) as failure:
                    profile_error = type(failure).__name__ + ": " + str(failure)
                row = {"index": index, "case_index": model["model_index"], "input": model["input"], "model": Path(model["input"]).stem,
                    "variant": variant, "command": command, "returncode": code, "report": report, "stdout": stdout,
                    "stderr": stderr, "error": error, "elapsed_wall_seconds": elapsed, "hashes_before": before,
                    "hashes_after": after, "model_sha256": before["input"], "profile": profile, "profile_error": profile_error,
                    "classification": benchmark.classify(code, report) if report is not None else error}
                numerical = compare_lp.numerical_seconds(row)
                failure = compare_lp.quality_failure(row, model, after["input"], identities[str(executable)], after["solver"])
                if failure is None and (numerical > plan["time_limit"] or elapsed > plan["wall_timeout"]):
                    failure = "budget_exceeded"
                row.update(numerical_seconds=numerical, quality_pass=failure is None, quality_failure=failure)
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                assert before == after and before["input"] == model["sha256"] and before["solver"] == identities[str(executable)]
                print(json.dumps({"index": index, "model": row["model"], "variant": variant,
                    "status": (report or {}).get("status"), "profile_error": profile_error, "valid_process": valid_process(row)}), flush=True)
                index += 1
    for path, expected in protected.items():
        assert sha(path) == expected
    summarize()


def self_test():
    plan = fixed_plan()
    row = {"error": None, "classification": "verification_failed", "returncode": 3,
        "report": {"schema_version": 1, "status": "verification_failed", "backend_version": "ipm-cpu-0.53.0",
            "options": {"ipm": {"objective_scaling": False}},
            "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}},
        "numerical_seconds": 1, "elapsed_wall_seconds": 2}
    assert pair_result(row, row, plan)["category"] == "equal"
    limited = copy.deepcopy(row)
    limited["report"]["status"] = "time_limit"
    limited.update(classification="incomplete", returncode=4)
    assert pair_result(row, limited, plan)["category"] == "budget_limited"
    broken = copy.deepcopy(row)
    broken["returncode"] = -1073741819
    assert pair_result(broken, limited, plan)["category"] == "invalid_process"
    broken.update(classification="resolved", returncode=False)
    assert not valid_process(broken)
    changed = copy.deepcopy(row)
    changed["report"]["iterations"] = 1
    assert pair_result(row, changed, plan)["category"] == "different"
    changed = copy.deepcopy(row)
    changed["report"]["backend_version"] = "other"
    assert pair_result(row, changed, plan)["category"] == "different"
    changed = copy.deepcopy(row)
    changed["report"]["options"]["ipm"]["objective_scaling"] = 0
    assert pair_result(row, changed, plan)["category"] == "different"
    for mutate in (lambda item: item["report"].update(schema_version=True),
        lambda item: item["report"]["timings"].update(solve_seconds=True)):
        changed = copy.deepcopy(row)
        mutate(changed)
        assert pair_result(changed, limited, plan)["category"] == "invalid_process"
    zero = {"inclusive_ns": 0, "exclusive_ns": 0, "started": 0, "completed": 0, "unwound": 0}
    stages = {name: zero.copy() for name in plan["profile_stages"]}
    stages["total"].update(inclusive_ns=10, exclusive_ns=10, started=1, completed=1)
    payload = {"schema": 1, "stages": stages, "clock_reads": 2, "exclusive_sum_ns": 10,
        "total_inclusive_ns": 10, "conserved": True, "stack_empty": True, "counts_closed": True}
    encode = lambda value: plan["profile_prefix"] + json.dumps(value)
    assert parse_profile(encode(payload), True, plan) == payload
    assert parse_profile("", False, plan) is None
    bad = []
    for field in ("clock_reads", "exclusive_sum_ns", "total_inclusive_ns", "schema"):
        item = copy.deepcopy(payload)
        item[field] = True
        bad.append(encode(item))
    item = copy.deepcopy(payload)
    item["stages"]["total"]["exclusive_ns"] = 9
    bad.append(encode(item))
    item = copy.deepcopy(payload)
    item["stages"]["total"]["completed"] = 0
    bad.append(encode(item))
    item = copy.deepcopy(payload)
    item["stages"].pop(plan["profile_stages"][-1])
    bad.append(encode(item))
    bad += [encode(payload) + "\n" + encode(payload), ""]
    item = copy.deepcopy(payload)
    item["stages"]["total"].update(inclusive_ns=2**63, exclusive_ns=2**63)
    bad.append(encode(item))
    item = copy.deepcopy(payload)
    item["stages"]["total"]["started"] = 1.0
    bad.append(encode(item))
    item = copy.deepcopy(payload)
    item["unexpected"] = 1
    bad.append(encode(item))
    bad.append(encode(payload).replace('"schema": 1', '"schema": 1, "schema": 1'))
    for stderr in bad:
        try:
            parse_profile(stderr, True, plan)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("无效计时报告未被拒绝")
    try:
        parse_profile(encode(payload), False, plan)
    except AssertionError:
        pass
    else:
        raise AssertionError("关闭侧计时污染未被拒绝")
    contaminated = {"SOLVER_V47_PROFILE": "1", "solver_v81_basis": "1", "OPENBLAS_CORETYPE": "bad",
        "PATH": "kept", "OMP_NUM_THREADS": "9"}
    off = run_environment(contaminated, False, plan)
    on = run_environment(contaminated, True, plan)
    assert off == {"PATH": "kept", **ENVIRONMENT}
    assert on == {"PATH": "kept", **ENVIRONMENT, plan["profile_environment"]: "1"}
    assert run_environment(on, False, plan) == off
    golden = copy.deepcopy(payload)
    for name, values in (("total", (110, 20, 1, 1, 0)),
        ("canonicalize", (110, 50, 3, 2, 1)), ("normal_setup", (60, 40, 2, 1, 1))):
        golden["stages"][name] = dict(zip(FIELDS, values))
    golden.update(clock_reads=12, exclusive_sum_ns=110, total_inclusive_ns=110)
    raw = {"command": [str(WORK / "profile-self-test.exe")], "cwd": str(ROOT), "returncode": 0,
        "stdout": '{"passed":true,"enabled":true}\n', "stderr": encode(golden), "wall_timeout": False}
    validate_timer_run(raw, True, WORK, plan)
    raw_off = copy.deepcopy(raw)
    raw_off.update(stdout='{"passed":true,"enabled":false}\n', stderr="")
    validate_timer_run(raw_off, False, WORK, plan)
    timer_negative = 0
    for mutate in (lambda item: item.update(returncode=False), lambda item: item.update(returncode=1),
        lambda item: item.update(wall_timeout=True), lambda item: item.update(stdout='{"passed":true,"enabled":false}'),
        lambda item: item.update(stdout='{"passed":1,"enabled":true}'), lambda item: item.update(stderr=""),
        lambda item: item.update(command=["other.exe"])):
        item = copy.deepcopy(raw)
        mutate(item)
        try:
            validate_timer_run(item, True, WORK, plan)
        except AssertionError:
            timer_negative += 1
        else:
            raise AssertionError("计时器自测原件污染未被拒绝")
    raw_off["stderr"] = "unexpected"
    try:
        validate_timer_run(raw_off, False, WORK, plan)
    except AssertionError:
        timer_negative += 1
    else:
        raise AssertionError("计时器关闭侧原件污染未被拒绝")
    return {"passed": True, "trajectory_cases": 9, "profile_positive_cases": 2,
        "profile_negative_cases": len(bad) + 1, "environment_cases": 3,
        "timer_positive_cases": 2, "timer_negative_cases": timer_negative, "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary", "self-test"))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.action == "run":
        run(args.work.resolve())
    elif args.action == "summary":
        summarize()
    else:
        print(json.dumps(self_test(), ensure_ascii=False))
