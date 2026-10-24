"""固定输入五项轮转的阶段诊断；保留失败、计时账本与完整数值报告。"""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v55-diagnostic"
PLAN_SHA = "7bb52355663442a361ff799fdddf61b96e908c3e5302eb2615e8e4cf7e58a6a9"
RAW = DATA / "v55-runs.jsonl.gz"
MANIFEST = DATA / "v55-run-manifest.json"
SUMMARY = DATA / "v55-summary.json"
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def fixed_plan():
    assert sha(DATA / "v55-plan.json") == PLAN_SHA
    plan = read(DATA / "v55-plan.json")
    assert sha(DATA / "ipm-v54-identity.json") == plan["parent_identity_sha256"]
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
    return row["error"] is None and row["classification"] in expected and type(row["returncode"]) is int and row["returncode"] == expected[row["classification"]] and isinstance(row["report"], dict)


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row["report"]) for row in (a, b)]
    for report in reports:
        if report is not None:
            report.pop("timings", None)
    equal = reports[0] == reports[1] and a["returncode"] == b["returncode"] and a["error"] == b["error"]
    valid = valid_process(a) and valid_process(b)
    budget = any(row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or
        row["elapsed_wall_seconds"] > plan["wall_timeout"] or row["numerical_seconds"] is not None and
        row["numerical_seconds"] > plan["time_limit"] for row in (a, b))
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def parse_profile(stderr, enabled, plan):
    lines = [line for line in stderr.splitlines() if line.startswith(plan["profile_prefix"])]
    assert len(lines) == int(enabled), "计时报告数量不符"
    if not enabled:
        return None
    payload = benchmark.strict_json(lines[0][len(plan["profile_prefix"]):])
    assert type(payload["schema"]) is int and payload["schema"] == 1
    assert all(payload[key] is True for key in ("conserved", "stack_empty", "counts_closed"))
    stages = payload["stages"]
    assert list(stages) == plan["profile_stages"]
    for stage in stages.values():
        assert set(stage) == {"inclusive_ns", "exclusive_ns", "started", "completed", "unwound"}
        assert all(type(value) is int and value >= 0 for value in stage.values())
        assert stage["started"] == stage["completed"] + stage["unwound"]
        assert stage["exclusive_ns"] <= stage["inclusive_ns"]
        if stage["started"] == 0:
            assert stage["inclusive_ns"] == stage["exclusive_ns"] == 0
    assert stages["total"]["started"] == 1
    assert all(type(payload[key]) is int and payload[key] >= 0 for key in ("clock_reads", "exclusive_sum_ns", "total_inclusive_ns"))
    assert payload["clock_reads"] == 2 * sum(item["started"] for item in stages.values())
    assert sum(item["exclusive_ns"] for item in stages.values()) == stages["total"]["inclusive_ns"]
    assert payload["exclusive_sum_ns"] == payload["total_inclusive_ns"] == stages["total"]["inclusive_ns"]
    return payload


def summarize_rows(rows, plan):
    assert len(rows) == plan["runs"]
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
    rows = [benchmark.strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
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
    protected = {str(Path(__file__)): sha(__file__), str(build_path): sha(build_path), **identities}
    assert build["protected_before_and_after"][str(Path(__file__))] == sha(__file__)
    identity = read(DATA / "ipm-v54-identity.json")
    for group in ("files", "libraries"):
        for name, expected in identity[group].items():
            assert sha(ROOT / name) == expected
            protected[str(ROOT / name)] = expected
    for path, digest in identities.items():
        assert sha(path) == digest
    for model in plan["models"]:
        assert sha(model["input"]) == model["sha256"]
    environment = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "build_manifest": str(build_path), "protected": protected, "environment": environment,
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
                env = {key: value for key, value in os.environ.items() if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
                env.update(environment)
                if enabled:
                    env[plan["profile_environment"]] = plan["profile_enabled_value"]
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
        "report": {"status": "verification_failed", "timings": {"a": 1}}, "numerical_seconds": 1, "elapsed_wall_seconds": 2}
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
    for stderr in bad:
        try:
            parse_profile(stderr, True, plan)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("无效计时报告未被拒绝")
    return {"passed": True, "trajectory_cases": 5, "profile_positive_cases": 2, "profile_negative_cases": len(bad), "solver_runs": 0}


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
