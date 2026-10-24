"""按第47轮冻结协议串行比较失败触发的常数方向恢复，保留全部失败。"""
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN = DATA / "v50-plan.json"
PLAN_SHA = "477a5efec5cdfd73f8330ff3a114179065a3e287014160149a46c7cd9c896ea9"
EXE = ROOT / ".tools/solver-ipm-v50.exe"
FIELDS = ("ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes")
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def plan():
    assert sha(PLAN) == PLAN_SHA
    value = read(PLAN)
    identity = DATA / "ipm-v50-identity.json"
    if identity.exists():
        frozen = read(identity)
        assert frozen["runner_sha256"] == sha(__file__) and frozen["plan_sha256"] == PLAN_SHA
    return value


def environment():
    result = os.environ.copy()
    result.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    result.pop("SOLVER_V49_DIAGNOSTICS", None)
    return result


def check_solver(version=50):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    identity = read(DATA / f"ipm-v{version}-identity.json")
    assert sha(executable) == identity["solver_sha256"]
    return executable


def specification(stage, frozen):
    if stage in ("netlib", "highs"):
        return frozen["sets"]["netlib"], frozen["primary_linear_solver"]
    item, = (item for item in frozen["initial_stages"] if item["name"] == stage)
    return frozen["sets"][item["set"]], item["linear_solver"]


def command_for(stage, frozen, output):
    expected, linear = specification(stage, frozen)
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--models-file", expected["models_file"], "--repeats", str(expected["repeats"]),
               "--time-limit", str(frozen["time_limit"]), "--wall-timeout", str(frozen["wall_timeout"]),
               "--threads", str(frozen["threads"]), "--seed", str(frozen["seed"]),
               "--candidate-backend", "ipm", "--reference-backend", "highs" if stage == "highs" else "ipm",
               "--output-dir", str(output)]
    for prefix, accuracy in (("--", frozen["candidate_accuracy"]), ("--reference-", frozen["reference_accuracy"])):
        if stage == "highs" and prefix == "--reference-":
            for name, value in frozen["highs_reference"].items():
                command += [prefix + "highs-" + name, value]
            continue
        command += [prefix + "iteration-limit", str(frozen["max_iterations"]),
                    prefix + "scaling-passes", str(frozen["scaling_passes"]),
                    prefix + "ipm-method", frozen["method"], prefix + "ipm-linear-accuracy", accuracy,
                    prefix + "ipm-linear-solver", linear, prefix + "ipm-direction-retries", "0"]
        for key, value in frozen["shared_ipm"].items():
            command += [prefix + "ipm-" + key, value]
    return command


def archive(stage):
    frozen = plan()
    expected, linear = specification(stage, frozen)
    manifest = read(DATA / f"v50-{stage}-manifest.json")
    summary = read(DATA / f"v50-{stage}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v50-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
    assert manifest["models"] == expected["models"] and manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[key] == frozen[key]
    assert manifest["repeats"] == expected["repeats"]
    assert manifest["ipm_linear_accuracy"] == frozen["candidate_accuracy"]
    assert manifest["ipm_linear_solver"] == linear and manifest["ipm_method"] == frozen["method"]
    if stage != "highs":
        assert manifest["reference_ipm_linear_accuracy"] == frozen["reference_accuracy"]
        assert manifest["reference_ipm_linear_solver"] == linear and manifest["reference_ipm_method"] == frozen["method"]
    reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v50-{stage}"))
    assert recomputed == summary
    return manifest, summary, rows


def gate_decision(development, probe, actual_successes):
    def changes(summary, relation):
        r, c = summary["reference_variant"], summary["candidate_variant"]
        return [Path(item["input"]).stem for item in summary["models"]
                if relation(item["quality_pass_counts"][c], item["quality_pass_counts"][r])]
    lost_dev = changes(development, lambda c, r: c < r)
    lost_probe = changes(probe, lambda c, r: c < r)
    gain = changes(probe, lambda c, r: c > r)
    return {"development_lost": lost_dev, "probe_lost": lost_probe, "probe_gain": gain,
            "actual_repair_successes": actual_successes,
            "eligible_for_netlib": not lost_dev and not lost_probe and bool(gain) and actual_successes > 0,
            "plan_sha256": PLAN_SHA}


def gate():
    ma, development, a = archive("development")
    mb, probe, b = archive("probe")
    successes = 0
    for manifest, rows in ((ma, a), (mb, b)):
        for row in rows:
            if row["requested_variant"] != manifest["candidate_variant"] or row.get("report") is None:
                continue
            report = row["report"]
            assert report["options"]["ipm"]["linear_accuracy"] == "homogeneous_repair"
            assert benchmark.valid_homogeneous_repair_work(report)
            successes += report[FIELDS[1]]
    return gate_decision(development, probe, successes)


def run(stage):
    frozen = plan()
    check_solver()
    if stage in ("netlib", "highs"):
        decision = gate()
        gate_path = DATA / "v50-gate.json"
        if gate_path.exists():
            assert read(gate_path) == decision
        else:
            save(gate_path, decision)
        assert decision["eligible_for_netlib"], "主配置未通过事前门槛"
        if stage == "highs":
            archive("netlib")
    expected, _ = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v50-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v50-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v50-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    with (DATA / f"v50-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v50-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v50-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v50-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}})
    check_solver()
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def valid_old_report(report, code, linear, accuracy, method):
    if not isinstance(report, dict) or report.get("backend") != "ipm":
        return False
    if benchmark.classify(code, report) not in ("resolved", "incomplete", "verification_failed"):
        return False
    options = report.get("options", {}).get("ipm", {})
    if any(options.get(key) != value for key, value in (("linear_solver", linear), ("linear_accuracy", accuracy), ("method", method))):
        return False
    timings = report.get("timings", {})
    return all(benchmark.finite_real(timings.get(key)) and timings[key] >= 0
               for key in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds"))


def trajectories():
    frozen = plan()
    destination = DATA / "v50-old-mode-trajectory-runs.json.gz"
    assert not destination.exists()
    executables = [check_solver(version) for version in (48, 50)]
    records = []
    for linear, accuracy, method in frozen["old_trajectory"]["configurations"]:
        for model in frozen["old_trajectory"]["models"]:
            row = {"model": model["input"], "model_sha256": model["sha256"], "commands": [], "reports": [],
                   "returncodes": [], "outputs": [], "hashes": [], "non_time_equal": False, "budget_limited": False,
                   "fields_valid": False, "protocol_valid": [], "linear_solver": linear, "linear_accuracy": accuracy, "method": method}
            records.append(row)
            for executable in executables:
                command = [str(executable), "solve", model["input"], "--backend", "ipm", "--time-limit", "5",
                           "--threads", "1", "--seed", "0", "--iteration-limit", "200", "--scaling-passes", "5",
                           "--ipm-method", method, "--ipm-linear-solver", linear, "--ipm-linear-accuracy", accuracy,
                           "--ipm-direction-retries", "0"]
                for key, value in frozen["shared_ipm"].items():
                    if linear == "qdldl" and key in ("singleton-elimination", "fixed-variable-elimination"):
                        value = "off"
                    command += ["--ipm-" + key, value]
                before = {"solver": sha(executable), "input": sha(model["input"])}
                row["commands"].append(command)
                try:
                    process = subprocess.run(command, cwd=ROOT, env=environment(), capture_output=True, timeout=35)
                    code, stdout, stderr, error = process.returncode, process.stdout, process.stderr, None
                except subprocess.TimeoutExpired as failure:
                    code, stdout, stderr, error = None, failure.stdout or b"", failure.stderr or b"", "wall_timeout"
                stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
                row["returncodes"].append(code)
                row["outputs"].append({"stdout": stdout, "stderr": stderr, "error": error})
                row["hashes"].append({"before": before, "after": {"solver": sha(executable), "input": sha(model["input"])}})
                try:
                    report = benchmark.strict_json(stdout)
                except ValueError:
                    report = None
                row["reports"].append(report)
                row["protocol_valid"].append(valid_old_report(report, code, linear, accuracy, method) if error is None else error == "wall_timeout")
                destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
            old, new = row["reports"]
            row["fields_valid"] = isinstance(new, dict) and all(type(new.get(key)) is int and new[key] == 0 for key in FIELDS)
            row["budget_limited"] = any(output["error"] == "wall_timeout" or isinstance(r, dict) and
                (r.get("status") == "time_limit" or r["timings"]["solve_seconds"] + r["timings"]["verification_seconds"] > 5)
                for r, output in zip((old, new), row["outputs"]))
            normalized = copy.deepcopy([old, new])
            for report in normalized:
                if isinstance(report, dict):
                    for key in frozen["old_trajectory"]["excluded_fields"]:
                        report.pop(key, None)
            row["non_time_equal"] = isinstance(old, dict) and isinstance(new, dict) and normalized[0] == normalized[1] and row["returncodes"][0] == row["returncodes"][1]
            destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
    assert len(records) == frozen["old_trajectory"]["pairs"]
    assert all(all(row["protocol_valid"]) for row in records), "无效报告不能借用预算分叉被接受"
    assert all(isinstance(report, dict) or output["error"] == "wall_timeout" for row in records
               for report, output in zip(row["reports"], row["outputs"]))
    assert all(row["fields_valid"] and (row["non_time_equal"] or row["budget_limited"]) for row in records)
    assert all(item["before"] == item["after"] and item["before"]["input"] == row["model_sha256"] for row in records for item in row["hashes"])
    for version in (48, 50):
        check_solver(version)
    print(json.dumps({"pairs": len(records), "equal": sum(r["non_time_equal"] for r in records),
                      "budget_limited": sum(r["budget_limited"] for r in records)}), flush=True)


def self_check():
    frozen = plan()
    for stage in ("development", "probe", "secondary-probe", "netlib", "highs"):
        command = command_for(stage, frozen, ROOT / "unused-v50-self-check")
        pairs = command[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert len(flags) * 2 == len(pairs)
        assert flags["--ipm-linear-accuracy"] == "homogeneous_repair"
        assert flags["--ipm-linear-solver"] == specification(stage, frozen)[1]
        if stage == "highs":
            assert not any(k.startswith("--reference-ipm-") for k in flags)
        else:
            assert flags["--reference-ipm-linear-accuracy"] == "strict"
    def sample(a, b):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": a, "c": b}}]}
    cases = [(3, 3, 0, 3, 1, True), (3, 2, 0, 3, 1, False), (3, 3, 3, 2, 1, False),
             (3, 3, 0, 0, 1, False), (3, 3, 0, 3, 0, False), (3, 3, 1, 2, 1, True)]
    for a, b, c, d, count, expected in cases:
        assert gate_decision(sample(a, b), sample(c, d), count)["eligible_for_netlib"] is expected
    return {"passed": True, "command_cases": 5, "gate_cases": len(cases), "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "trajectories", "development", "probe", "secondary-probe", "netlib", "highs", "gate"))
    stage = parser.parse_args().stage
    if stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif stage == "trajectories":
        trajectories()
    elif stage == "gate":
        decision = gate()
        save(DATA / "v50-gate.json", decision)
        print(json.dumps(decision, ensure_ascii=False))
    else:
        run(stage)
