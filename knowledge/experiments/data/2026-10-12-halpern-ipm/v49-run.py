"""串行采集冻结的失败诊断；保留全部报告与预算分叉，不生成性能胜负。"""
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
WORK = ROOT / ".tools/v49-diagnostic"
PLAN_SHA = "f3b1871230ae2062b2a4ccd1bea4bcc4ab4d158caf05f5fc28a2bd7dab9becc7"
CLARIFICATION_SHA = "71f12b6e76f1d12e4e8554f5e6310927145e5933c1f8e6b8a1fc8ffe4b59b054"
RAW = DATA / "v49-runs.jsonl.gz"
MANIFEST = DATA / "v49-run-manifest.json"
SUMMARY = DATA / "v49-summary.json"
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fixed_plan():
    assert sha(DATA / "v49-plan.json") == PLAN_SHA
    assert sha(DATA / "v49-diagnostic-clarification.json") == CLARIFICATION_SHA
    plan = read(DATA / "v49-plan.json")
    assert sha(DATA / "ipm-v48-identity.json") == plan["parent_identity_sha256"]
    return plan


def configurations(plan):
    result = []
    for model in plan["models"]:
        for backend in model["backends"]:
            result.append({"case_index": len(result), "model": model, "linear_solver": backend})
    return result


def command_for(executable, case, plan):
    command = [str(executable), "solve", case["model"]["input"], "--backend", "ipm",
        "--time-limit", str(plan["time_limit"]), "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
        "--iteration-limit", str(plan["max_iterations"]), "--scaling-passes", str(plan["scaling_passes"]),
        "--ipm-method", plan["method"], "--ipm-linear-accuracy", plan["linear_accuracy"],
        "--ipm-linear-solver", case["linear_solver"]]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def limited(row, plan):
    return row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or row["elapsed_wall_seconds"] > plan["wall_timeout"] or row["numerical_seconds"] is not None and row["numerical_seconds"] > plan["time_limit"]


def summarize():
    plan = fixed_plan()
    rows = [json.loads(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    assert len(rows) == plan["runs"] == 21
    for row in rows:
        assert row["error"] in (None, "wall_timeout"), "保留了无效报告，但不能把它判为轨迹通过"
        if row["error"] is None:
            assert row["classification"] in ("resolved", "incomplete", "verification_failed")
            assert row["report"]["backend"] == "ipm"
            assert row["report"]["options"]["ipm"]["method"] == plan["method"]
            assert row["report"]["options"]["ipm"]["linear_solver"] == row["linear_solver"]
    pairs, cases = [], []
    for case in configurations(plan):
        selected = {row["variant"]: row for row in rows if row["case_index"] == case["case_index"]}
        assert set(selected) == set(plan["variants"])
        for left, right in (("production_v48", "diagnostic_off"), ("diagnostic_off", "diagnostic_on")):
            a, b = selected[left], selected[right]
            normalized = []
            for row in (a, b):
                report = copy.deepcopy(row["report"])
                if report is not None:
                    report.pop("timings", None)
                    report.pop("backend_version", None)
                normalized.append(report)
            equal = normalized[0] == normalized[1] and a["returncode"] == b["returncode"] and a["error"] == b["error"]
            budget = limited(a, plan) or limited(b, plan)
            pairs.append({"case_index": case["case_index"], "model": case["model"]["name"], "linear_solver": case["linear_solver"],
                "left": left, "right": right, "equal": equal, "budget_limited": budget})
        cases.append({"case_index": case["case_index"], "model": case["model"]["name"], "linear_solver": case["linear_solver"],
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"], "capture_prefix": selected["diagnostic_on"]["capture_prefix"]})
    summary = {"diagnostic_only": True, "runs": len(rows), "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(not pair["budget_limited"] and pair["equal"] for pair in pairs),
        "budget_pairs": sum(pair["budget_limited"] for pair in pairs),
        "raw_sha256": sha(RAW), "scope": "诊断开销在实际预算内；这里不计算性能加速比或新的公开覆盖成绩。"}
    save_new(SUMMARY, summary)
    assert all(pair["equal"] or pair["budget_limited"] for pair in pairs), "非预算轨迹发生变化，保留全部原始结果"
    print(json.dumps({key: summary[key] for key in ("runs", "non_budget_pairs", "non_budget_equal", "budget_pairs")}), flush=True)


def run(work):
    plan = fixed_plan()
    assert not RAW.exists() and not MANIFEST.exists() and not SUMMARY.exists()
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["clarification_sha256"] == CLARIFICATION_SHA
    production = Path(build["production_executable"])
    diagnostic = Path(build["executable"])
    identities = {str(production): build["production_sha256"], str(diagnostic): build["executable_sha256"]}
    protected = {str(Path(__file__)): sha(Path(__file__)), str(build_path): sha(build_path), **identities}
    for name, expected in read(DATA / "ipm-v48-identity.json")["files"].items():
        assert sha(ROOT / name) == expected
    for name in ("benchmark.py", "compare_lp.py"):
        protected[str(ROOT / "scripts" / name)] = sha(ROOT / "scripts" / name)
    for model in plan["models"]:
        assert sha(Path(model["input"])) == model["sha256"]
    capture_root = work / "captures"
    assert not capture_root.exists()
    capture_root.mkdir()
    environment = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "clarification_sha256": CLARIFICATION_SHA, "build_manifest": str(build_path), "protected": protected,
        "environment": environment, "capture_root": str(capture_root), "work": str(work),
        "schedule": "按冻结模型及其后端顺序；每配置将三variant向左轮转case_index mod 3，全部串行。"})
    index = 0
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for case in configurations(plan):
            offset = case["case_index"] % 3
            order = plan["variants"][offset:] + plan["variants"][:offset]
            for variant in order:
                executable = production if variant == "production_v48" else diagnostic
                enabled = variant == "diagnostic_on"
                prefix = capture_root / (str(case["case_index"]) + "-" + variant)
                command = command_for(executable, case, plan)
                env = os.environ.copy()
                env.update(environment)
                env.pop("SOLVER_V49_DIAGNOSTICS", None)
                if enabled:
                    env["SOLVER_V49_DIAGNOSTICS"] = str(prefix)
                model = case["model"]
                before = {"input": sha(Path(model["input"])), "solver": sha(executable)}
                start = time.perf_counter()
                error = None
                try:
                    process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=plan["wall_timeout"], env=env)
                    stdout, stderr, code = process.stdout, process.stderr, process.returncode
                except subprocess.TimeoutExpired as failure:
                    stdout, stderr, code = failure.stdout or b"", failure.stderr or b"", None
                    error = "wall_timeout"
                except OSError as failure:
                    stdout, stderr, code = b"", str(failure).encode("utf-8"), None
                    error = "launch_error"
                elapsed = time.perf_counter() - start
                after = {"input": sha(Path(model["input"])), "solver": sha(executable)}
                stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
                try:
                    report = benchmark.strict_json(stdout)
                    if not isinstance(report, dict):
                        report, error = None, error or "invalid_report"
                except (ValueError, TypeError):
                    report, error = None, error or "invalid_json"
                files = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(capture_root.glob(prefix.name + ".*"))}
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
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                target.flush()
                assert before == after and before["input"] == model["sha256"] and before["solver"] == identities[str(executable)]
                if not enabled:
                    assert not files and "V49_NORMAL" not in stderr and "V49_HSD" not in stderr
                elif error != "wall_timeout":
                    assert Path(str(prefix) + ".normal.json").is_file() and Path(str(prefix) + ".hsd.json").is_file()
                print(json.dumps({"index": index, "model": model["name"], "linear_solver": case["linear_solver"], "variant": variant,
                    "status": (report or {}).get("status"), "captures": len(files)}), flush=True)
                index += 1
    for path, expected in protected.items():
        assert sha(Path(path)) == expected
    summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summary"))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    run(args.work.resolve()) if args.action == "run" else summarize()
