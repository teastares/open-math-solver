"""串行采集冻结 v57、重建基线及诊断开关；完整报告比较不作性能推断。"""
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
WORK = ROOT / ".tools/v59-diagnostic"
PLAN_SHA = "1773534738374fca2c1b655ec22b96f0a93970cb02d0e6900c37a25ce7ef7ef1"
RAW, MANIFEST, SUMMARY = (DATA / name for name in ("v59-runs.jsonl.gz", "v59-run-manifest.json", "v59-summary.json"))
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def read(path):
    return benchmark.strict_json(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def fixed_plan():
    assert sha(DATA / "v59-plan.json") == PLAN_SHA
    plan = read(DATA / "v59-plan.json")
    assert sha(DATA / "v59-baseline.json") == plan["baseline_identity_sha256"]
    assert sha(DATA / "ipm-v57-identity.json") == plan["parent_identity_sha256"]
    assert plan["variants"] == ["frozen_v57", "production_baseline", "diagnostic_off", "diagnostic_on"]
    assert plan["runs"] == len(plan["models"]) * plan["repeats"] * len(plan["variants"]) == 16
    return plan


def configurations(plan):
    return [{"case_index": i * plan["repeats"] + repetition, "model_index": i,
        "repetition": repetition, "model": model, "linear_solver": "normal_cholmod"}
        for i, model in enumerate(plan["models"]) for repetition in range(plan["repeats"])]


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
    return isinstance(report, dict) and report.get("backend") == "ipm" and report.get("backend_version") == "ipm-cpu-0.47.0" \
        and report.get("input") == model["input"] and report.get("options") == plan["expected_report_options"] \
        and report.get("ipm_factor_backend") == "normal_cholmod" and report.get("ipm_structural_fallbacks") == 0


def valid_process(row):
    report = row.get("report")
    if row.get("error") is not None or not isinstance(report, dict) or type(row.get("returncode")) is not int:
        return False
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1 or row.get("configuration_matches") is not True:
        return False
    status = report.get("status")
    known = {"optimal": (0, "resolved"), "empty": (0, "resolved"), "infeasible": (0, "resolved"),
        "unbounded": (0, "resolved"), "verification_failed": (3, "verification_failed"),
        "iteration_limit": (4, "incomplete"), "time_limit": (4, "incomplete"), "numerical_error": (4, "incomplete")}
    if not isinstance(status, str) or status not in known or (row["returncode"], row.get("classification")) != known[status]:
        return False
    if benchmark.classify(row["returncode"], report) != row["classification"]:
        return False
    timings = report.get("timings")
    if not isinstance(timings, dict) or not all(benchmark.finite_real(timings.get(name)) and timings[name] >= 0
        for name in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")):
        return False
    numerical = row.get("numerical_seconds")
    wall = row.get("elapsed_wall_seconds")
    return benchmark.finite_real(numerical) and numerical >= 0 and benchmark.finite_real(wall) and wall > 0


def limited(row, plan):
    return row.get("error") == "wall_timeout" or (row.get("report") or {}).get("status") == "time_limit" \
        or row["elapsed_wall_seconds"] > plan["wall_timeout"] \
        or row["numerical_seconds"] is not None and row["numerical_seconds"] > plan["time_limit"]


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row["report"]) for row in (a, b)]
    for report in reports:
        if report is not None:
            report.pop("timings", None)
    equal = reports[0] == reports[1] and a["returncode"] == b["returncode"] and a["error"] == b["error"]
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, plan) or limited(b, plan)
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def summarize():
    plan = fixed_plan()
    rows = [benchmark.strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    assert len(rows) == plan["runs"] and [row["index"] for row in rows] == list(range(plan["runs"]))
    pairs, cases = [], []
    for case in configurations(plan):
        selected_rows = [row for row in rows if row["case_index"] == case["case_index"]]
        selected = {row["variant"]: row for row in selected_rows}
        assert len(selected_rows) == len(selected) == 4 and set(selected) == set(plan["variants"])
        for left, right in zip(plan["variants"], plan["variants"][1:]):
            pairs.append({"case_index": case["case_index"], "model": case["model"]["name"], "repetition": case["repetition"],
                "left": left, "right": right, **pair_result(selected[left], selected[right], plan)})
        cases.append({"case_index": case["case_index"], "model": case["model"]["name"], "repetition": case["repetition"],
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"], "capture_prefix": selected["diagnostic_on"]["capture_prefix"]})
    summary = {"diagnostic_only": True, "runs": len(rows), "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs),
        "capture_errors": sum(row["capture_error"] is not None for row in rows),
        "raw_sha256": sha(RAW), "scope": "诊断复制成本计入实际预算；不计算性能加速比或新的公开覆盖成绩。"}
    save_new(SUMMARY, summary)
    assert summary["invalid_pairs"] == summary["different_pairs"] == summary["capture_errors"] == 0, "核验未通过，保留全部原始结果"
    print(json.dumps({key: summary[key] for key in ("runs", "non_budget_pairs", "non_budget_equal", "budget_pairs", "invalid_pairs", "capture_errors")}), flush=True)


def run(work):
    plan = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY))
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    executables = {"frozen_v57": Path(build["frozen_parent_executable"]), "production_baseline": Path(build["production_executable"]),
        "diagnostic_off": Path(build["executable"]), "diagnostic_on": Path(build["executable"])}
    identities = {str(executables["frozen_v57"]): build["frozen_parent_sha256"], str(executables["production_baseline"]): build["production_sha256"],
        str(executables["diagnostic_on"]): build["executable_sha256"]}
    identity = read(DATA / "v59-baseline.json")
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"] and build["production_sha256"] == identity["solver_sha256"]
    protected = {str(Path(__file__)): sha(__file__), str(build_path): sha(build_path), **identities,
        **build["protected_before_and_after"], **build["link_explicit_before_and_after"], **build["compile_inputs_after_and_final"], **build["outputs"]}
    for group in ("files", "libraries"):
        protected.update({str(ROOT / name): expected for name, expected in identity[group].items()})
    for name in ("benchmark.py", "compare_lp.py"):
        protected[str(ROOT / "scripts" / name)] = identity["files"]["scripts/" + name]
    for path, expected in protected.items():
        assert sha(path) == expected, path
    for model in plan["models"]:
        assert sha(model["input"]) == model["sha256"]
    capture_root = work / "captures"
    capture_root.mkdir()
    environment = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    save_new(MANIFEST, {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan_sha256": PLAN_SHA,
        "build_manifest": str(build_path), "protected": protected, "environment": environment,
        "removed_environment": "所有 SOLVER_V 开头诊断变量及 OPENBLAS_CORETYPE；on 只设置 SOLVER_V59_DIAGNOSTICS",
        "capture_root": str(capture_root), "work": str(work),
        "schedule": "冻结模型顺序，各模型两次重复；每配置四 variant 向左轮转 case_index mod 4，全部串行。"})
    index = 0
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for case in configurations(plan):
            offset = case["case_index"] % len(plan["variants"])
            for variant in plan["variants"][offset:] + plan["variants"][:offset]:
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
                files = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(capture_root.glob(prefix.name + "-*"))}
                markers = [line[len("V59_KKT "):] for line in stderr.splitlines() if line.startswith("V59_KKT ")]
                capture_error = None
                if enabled:
                    try:
                        marker = benchmark.strict_json(markers[0]) if len(markers) == 1 else None
                        expected_files = {str(prefix) + "-kkt.json", str(prefix) + "-kkt.bin"}
                        if not isinstance(marker, dict) or marker.get("write_error") is not False \
                            or type(marker.get("captured")) is not bool or set(files) != expected_files:
                            capture_error = "invalid_diagnostic_marker"
                    except (ValueError, TypeError):
                        capture_error = "invalid_diagnostic_marker"
                elif files or markers:
                    capture_error = "diagnostic_output_when_disabled"
                row = {"index": index, "case_index": case["case_index"], "model_index": case["model_index"],
                    "repetition": case["repetition"], "input": model["input"], "model": model["name"],
                    "linear_solver": case["linear_solver"], "variant": variant, "command": command, "returncode": code,
                    "report": report, "stdout": stdout, "stderr": stderr, "error": error, "elapsed_wall_seconds": elapsed,
                    "hashes_before": before, "hashes_after": after, "model_sha256": before["input"],
                    "classification": benchmark.classify(code, report) if report is not None else error,
                    "capture_prefix": str(prefix), "capture_files": files, "capture_error": capture_error,
                    "configuration_matches": configuration_matches(report, model, plan)}
                numerical = compare_lp.numerical_seconds(row)
                row["numerical_seconds"] = numerical
                failure = compare_lp.quality_failure(row, model, after["input"], identities[str(executable)], after["solver"])
                if not valid_process(row):
                    failure = "invalid_process"
                elif failure is None and (numerical > plan["time_limit"] or elapsed > plan["wall_timeout"]):
                    failure = "budget_exceeded"
                row.update(quality_pass=failure is None, quality_failure=failure)
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                assert before == after and before["input"] == model["sha256"] and before["solver"] == identities[str(executable)]
                print(json.dumps({"index": index, "model": model["name"], "repetition": case["repetition"], "variant": variant,
                    "status": (report or {}).get("status"), "captures": len(files), "valid_process": valid_process(row), "capture_error": capture_error}), flush=True)
                index += 1
    for path, expected in protected.items():
        assert sha(path) == expected, path
    summarize()


def self_test():
    plan = {"time_limit": 5, "wall_timeout": 35}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "configuration_matches": True,
        "report": {"schema_version": 1, "status": "numerical_error", "backend_version": "ipm-cpu-0.47.0",
            "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}},
        "numerical_seconds": 1, "elapsed_wall_seconds": 2}
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
    broken_rows = []
    for key, value in (("returncode", -1073741819), ("returncode", False), ("returncode", 0), ("error", "launch_error"),
        ("configuration_matches", False), ("numerical_seconds", None), ("numerical_seconds", float("nan")), ("elapsed_wall_seconds", -1)):
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
    frozen = fixed_plan()
    cases = configurations(frozen)
    assert len(cases) == 4 and [case["case_index"] for case in cases] == [0, 1, 2, 3]
    for case in cases:
        command = command_for(Path("test.exe"), case, frozen)
        assert command.count("--ipm-objective-scaling") == 1 and command[command.index("--ipm-objective-scaling") + 1] == "off"
        assert len([value for value in command if value.startswith("--")]) == len(set(value for value in command if value.startswith("--")))
    print(json.dumps({"passed": True, "trajectory_checks": 4 + len(broken_rows), "commands_checked": len(cases), "lp_runs": 0}))


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
