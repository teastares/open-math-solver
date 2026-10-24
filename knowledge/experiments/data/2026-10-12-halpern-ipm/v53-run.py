"""串行采集第50轮HSD方向诊断；预算分叉与无效进程分别留存。"""
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
WORK = ROOT / ".tools/v53-diagnostic"
PLAN_SHA = "87fd87be5bc80deb0ee7689d5ad16635969b447e836f88fc8bb9019479f5cdf0"
RAW = DATA / "v53-runs.jsonl.gz"
MANIFEST = DATA / "v53-run-manifest.json"
SUMMARY = DATA / "v53-summary.json"
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def fixed_plan():
    assert sha(DATA / "v53-plan.json") == PLAN_SHA
    plan = read(DATA / "v53-plan.json")
    assert sha(DATA / "ipm-v52-identity.json") == plan["parent_identity_sha256"]
    return plan


def configurations(plan):
    return [{"case_index": i, "model": model, "linear_solver": "normal_cholmod"} for i, model in enumerate(plan["models"])]


def command_for(executable, case, plan):
    command = [str(executable), "solve", case["model"]["input"], "--backend", "ipm",
        "--time-limit", str(plan["time_limit"]), "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
        "--iteration-limit", str(plan["max_iterations"]), "--scaling-passes", str(plan["scaling_passes"]),
        "--ipm-method", plan["method"], "--ipm-linear-accuracy", plan["linear_accuracy"],
        "--ipm-linear-solver", case["linear_solver"]]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def valid_process(row):
    expected = {"resolved": 0, "incomplete": 4, "verification_failed": 3}
    return row["error"] is None and row["classification"] in expected and type(row["returncode"]) is int and row["returncode"] == expected[row["classification"]] and isinstance(row["report"], dict)


def limited(row, plan):
    return row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or row["elapsed_wall_seconds"] > plan["wall_timeout"] or row["numerical_seconds"] is not None and row["numerical_seconds"] > plan["time_limit"]


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row["report"]) for row in (a, b)]
    for report in reports:
        if report is not None:
            report.pop("timings", None)
    equal = reports[0] == reports[1] and a["returncode"] == b["returncode"] and a["error"] == b["error"]
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, plan) or limited(b, plan)
    # 墙钟超时属于采集未完成，不能宣称已验证轨迹等价。
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def summarize():
    plan = fixed_plan()
    rows = [json.loads(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    assert len(rows) == plan["runs"] == 12
    pairs, cases = [], []
    for case in configurations(plan):
        selected = {row["variant"]: row for row in rows if row["case_index"] == case["case_index"]}
        assert set(selected) == set(plan["variants"]) and len(selected) == 3
        for left, right in (("production_v52", "diagnostic_off"), ("diagnostic_off", "diagnostic_on")):
            pairs.append({"case_index": case["case_index"], "model": case["model"]["name"], "linear_solver": case["linear_solver"],
                "left": left, "right": right, **pair_result(selected[left], selected[right], plan)})
        cases.append({"case_index": case["case_index"], "model": case["model"]["name"], "linear_solver": case["linear_solver"],
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"], "capture_prefix": selected["diagnostic_on"]["capture_prefix"]})
    summary = {"diagnostic_only": True, "runs": len(rows), "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs),
        "raw_sha256": sha(RAW), "scope": "诊断复制成本计入实际预算；不计算性能加速比或新的公开覆盖成绩。"}
    save_new(SUMMARY, summary)
    assert summary["invalid_pairs"] == summary["different_pairs"] == 0, "轨迹核验未通过，保留全部原始结果"
    print(json.dumps({key: summary[key] for key in ("runs", "non_budget_pairs", "non_budget_equal", "budget_pairs", "invalid_pairs")}), flush=True)


def run(work):
    plan = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY))
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    production, diagnostic = Path(build["production_executable"]), Path(build["executable"])
    identities = {str(production): build["production_sha256"], str(diagnostic): build["executable_sha256"]}
    protected = {str(Path(__file__)): sha(__file__), str(build_path): sha(build_path), **identities}
    identity = read(DATA / "ipm-v52-identity.json")
    for group in ("files", "libraries"):
        for name, expected in identity[group].items():
            assert sha(ROOT / name) == expected
            protected[str(ROOT / name)] = expected
    for name in ("benchmark.py", "compare_lp.py"):
        protected[str(ROOT / "scripts" / name)] = sha(ROOT / "scripts" / name)
    for model in plan["models"]:
        assert sha(Path(model["input"])) == model["sha256"]
    capture_root = work / "captures"
    capture_root.mkdir()
    environment = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "build_manifest": str(build_path), "protected": protected, "environment": environment,
        "removed_environment": "所有SOLVER_V开头诊断变量及OPENBLAS_CORETYPE；on只设置SOLVER_V53_DIAGNOSTICS",
        "capture_root": str(capture_root), "work": str(work),
        "schedule": "冻结模型顺序，每配置三variant向左轮转case_index mod 3，全部串行。"})
    index = 0
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for case in configurations(plan):
            offset = case["case_index"] % 3
            for variant in plan["variants"][offset:] + plan["variants"][:offset]:
                executable = production if variant == "production_v52" else diagnostic
                enabled = variant == "diagnostic_on"
                prefix = capture_root / (str(case["case_index"]) + "-" + variant)
                command = command_for(executable, case, plan)
                env = {key: value for key, value in os.environ.items() if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
                env.update(environment)
                if enabled:
                    env["SOLVER_V53_DIAGNOSTICS"] = str(prefix)
                model = case["model"]
                before = {"input": sha(Path(model["input"])), "solver": sha(executable)}
                start, error = time.perf_counter(), None
                try:
                    process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=plan["wall_timeout"], env=env)
                    stdout, stderr, code = process.stdout, process.stderr, process.returncode
                except subprocess.TimeoutExpired as failure:
                    stdout, stderr, code, error = failure.stdout or b"", failure.stderr or b"", None, "wall_timeout"
                except OSError as failure:
                    stdout, stderr, code, error = b"", str(failure).encode("utf-8"), None, "launch_error"
                elapsed = time.perf_counter() - start
                after = {"input": sha(Path(model["input"])), "solver": sha(executable)}
                stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
                try:
                    report = benchmark.strict_json(stdout)
                    if not isinstance(report, dict):
                        report, error = None, error or "invalid_report"
                except (ValueError, TypeError):
                    report, error = None, error or "invalid_json"
                files = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(capture_root.glob(prefix.name + "-*"))}
                row = {"index": index, "case_index": case["case_index"], "input": model["input"], "model": model["name"],
                    "linear_solver": case["linear_solver"], "variant": variant, "command": command, "returncode": code,
                    "report": report, "stdout": stdout, "stderr": stderr, "error": error, "elapsed_wall_seconds": elapsed,
                    "hashes_before": before, "hashes_after": after, "model_sha256": before["input"],
                    "classification": benchmark.classify(code, report) if report is not None else error,
                    "capture_prefix": str(prefix), "capture_files": files}
                numerical = compare_lp.numerical_seconds(row)
                failure = compare_lp.quality_failure(row, model, after["input"], identities[str(executable)], after["solver"])
                if failure is None and (numerical > plan["time_limit"] or elapsed > plan["wall_timeout"]):
                    failure = "budget_exceeded"
                row.update(numerical_seconds=numerical, quality_pass=failure is None, quality_failure=failure)
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                assert before == after and before["input"] == model["sha256"] and before["solver"] == identities[str(executable)]
                if not enabled:
                    assert not files and "V53_HSD" not in stderr
                print(json.dumps({"index": index, "model": model["name"], "variant": variant,
                    "status": (report or {}).get("status"), "captures": len(files), "valid_process": valid_process(row)}), flush=True)
                index += 1
    for path, expected in protected.items():
        assert sha(Path(path)) == expected
    summarize()


def self_test():
    plan = {"time_limit": 5, "wall_timeout": 35}
    row = {"error": None, "classification": "verification_failed", "returncode": 3,
        "report": {"status": "verification_failed", "backend_version": "ipm-cpu-0.44.0", "timings": {"a": 1}},
        "numerical_seconds": 1, "elapsed_wall_seconds": 2}
    assert pair_result(row, row, plan)["category"] == "equal"
    time_row = copy.deepcopy(row)
    time_row["report"]["status"] = "time_limit"
    time_row.update(classification="incomplete", returncode=4)
    assert pair_result(row, time_row, plan)["category"] == "budget_limited"
    broken = copy.deepcopy(row)
    broken["returncode"] = -1073741819
    assert pair_result(broken, time_row, plan)["category"] == "invalid_process"
    different = copy.deepcopy(row)
    different["report"]["backend_version"] = "changed"
    assert pair_result(row, different, plan)["category"] == "different"
    different = copy.deepcopy(row)
    different["report"]["timings"]["a"] = 2
    assert pair_result(row, different, plan)["category"] == "equal"
    broken.update(classification="resolved", returncode=False)
    assert not valid_process(broken)
    print(json.dumps({"passed": True, "checks": 6}))


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
        self_test()
