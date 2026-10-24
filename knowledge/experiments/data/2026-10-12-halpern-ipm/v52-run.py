"""第49轮只缓存固定下三角结构的两程序串行对照；失败保留，不覆盖已有结果。"""
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
PLAN = DATA / "v52-plan.json"
PLAN_SHA = "d1a027a5eb7b045b9e138da205bfb884323984af1242b1b4d7e22cd09d338648"
EXE = ROOT / ".tools/solver-ipm-v52.exe"
PARENT = ROOT / ".tools/solver-ipm-v51.exe"
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
    assert sha(DATA / "ipm-v51-identity.json") == value["parent_identity_sha256"]
    identity = DATA / "ipm-v52-identity.json"
    if identity.exists():
        frozen = read(identity)
        assert frozen["runner_sha256"] == sha(__file__) and frozen["plan_sha256"] == PLAN_SHA
    return value


def environment():
    result = os.environ.copy()
    result.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    result.pop("SOLVER_V49_DIAGNOSTICS", None)
    result.pop("OPENBLAS_CORETYPE", None)
    return result


def check_solver(version):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    assert sha(executable) == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    return executable


def specification(stage, frozen):
    return frozen["sets"]["netlib" if stage in ("netlib", "highs") else stage]


def command_for(stage, frozen, output):
    expected = specification(stage, frozen)
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--reference-solver", str(EXE if stage == "highs" else PARENT),
               "--models-file", expected["models_file"], "--repeats", str(expected["repeats"]),
               "--time-limit", str(frozen["time_limit"]), "--wall-timeout", str(frozen["wall_timeout"]),
               "--threads", str(frozen["threads"]), "--seed", str(frozen["seed"]),
               "--candidate-backend", "ipm", "--reference-backend", "highs" if stage == "highs" else "ipm",
               "--output-dir", str(output)]
    for prefix in ("--", "--reference-"):
        if stage == "highs" and prefix == "--reference-":
            for name, value in frozen["highs_reference"].items():
                command += [prefix + "highs-" + name, value]
            continue
        command += [prefix + "iteration-limit", str(frozen["max_iterations"]),
                    prefix + "scaling-passes", str(frozen["scaling_passes"]),
                    prefix + "ipm-method", frozen["method"],
                    prefix + "ipm-linear-accuracy", frozen["candidate_accuracy"],
                    prefix + "ipm-linear-solver", frozen["primary_linear_solver"],
                    prefix + "ipm-direction-retries", "0"]
        for key, value in frozen["shared_ipm"].items():
            command += [prefix + "ipm-" + key, value]
    return command


def archive(stage):
    frozen = plan()
    expected = specification(stage, frozen)
    manifest = read(DATA / f"v52-{stage}-manifest.json")
    summary = read(DATA / f"v52-{stage}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v52-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
    assert manifest["models"] == expected["models"] and manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[key] == frozen[key]
    assert manifest["repeats"] == expected["repeats"]
    assert manifest["ipm_linear_accuracy"] == frozen["candidate_accuracy"]
    assert manifest["ipm_linear_solver"] == frozen["primary_linear_solver"] and manifest["ipm_method"] == frozen["method"]
    if stage != "highs":
        assert manifest["reference_ipm_linear_accuracy"] == frozen["reference_accuracy"]
        assert manifest["reference_ipm_linear_solver"] == frozen["primary_linear_solver"]
        assert manifest["reference_ipm_method"] == frozen["method"]
    reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
    for variant, version in ((reference, 52 if stage == "highs" else 51), (candidate, 52)):
        solver = manifest["solvers"][variant]
        assert Path(solver["path"]) == check_solver(version)
        assert solver["sha256"] == sha(solver["path"]) and solver["hash_error"] is None
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v52-{stage}"))
    assert recomputed == summary
    return manifest, summary, rows


def gate_decision(development, probe, trajectories):
    def lost(summary):
        r, c = summary["reference_variant"], summary["candidate_variant"]
        return [Path(item["input"]).stem for item in summary["models"]
                if item["quality_pass_counts"][c] < item["quality_pass_counts"][r]]
    dev, pro = lost(development), lost(probe)
    r, c = development["reference_variant"], development["candidate_variant"]
    a, b = development["variants"][r], development["variants"][c]
    speed = b["penalized_numerical_sgm10_seconds"] < a["penalized_numerical_sgm10_seconds"] and b["penalized_wall_sgm10_seconds"] <= a["penalized_wall_sgm10_seconds"]
    pr, pc = probe["reference_variant"], probe["candidate_variant"]
    gain = [Path(item["input"]).stem for item in probe["models"] if item["quality_pass_counts"][pc] > item["quality_pass_counts"][pr]]
    return {"development_lost": dev, "probe_lost": pro, "probe_gain": gain,
            "development_speed_gate": speed, "trajectories": trajectories,
            "eligible_for_netlib": not dev and not pro and trajectories["passed"] and (speed or bool(gain)),
            "plan_sha256": PLAN_SHA}


def trajectory_audit(manifest, rows):
    pairs = {}
    for row in rows:
        key = (row["input"], row["repetition"])
        sides = pairs.setdefault(key, {})
        assert row["requested_variant"] not in sides
        sides[row["requested_variant"]] = row
    result = {"pairs": len(pairs), "equal": 0, "budget_limited": 0, "invalid": 0, "different": 0}
    r, c = manifest["reference_variant"], manifest["candidate_variant"]
    for sides in pairs.values():
        assert set(sides) == {r, c}
        a, b = sides[r], sides[c]
        valid = True
        for row in (a, b):
            report = row.get("report")
            if not isinstance(report, dict):
                valid = False
                continue
            classification = benchmark.classify(row["returncode"], report)
            if (classification, row["returncode"]) not in (("resolved", 0), ("incomplete", 4), ("verification_failed", 3)):
                valid = False
        if not valid:
            result["invalid"] += 1
            continue
        budget = any(row["report"]["status"] == "time_limit" or
            row["report"]["timings"]["solve_seconds"] + row["report"]["timings"]["verification_seconds"] > 5 for row in (a, b))
        if budget:
            result["budget_limited"] += 1
            continue
        normalized = copy.deepcopy([a["report"], b["report"]])
        for report in normalized:
            report.pop("timings", None)
            report.pop("backend_version", None)
        same = normalized[0] == normalized[1] and a["returncode"] == b["returncode"]
        result["equal" if same else "different"] += 1
    result["passed"] = result["invalid"] == result["different"] == 0
    return result


def gate():
    ma, development, a = archive("development")
    mb, probe, b = archive("probe")
    audits = {"development": trajectory_audit(ma, a), "probe": trajectory_audit(mb, b)}
    audits["passed"] = all(value["passed"] for value in audits.values())
    return gate_decision(development, probe, audits)


def run(stage):
    frozen = plan()
    check_solver(51)
    check_solver(52)
    if stage in ("netlib", "highs"):
        decision = gate()
        destination = DATA / "v52-gate.json"
        if destination.exists():
            assert read(destination) == decision
        else:
            save(destination, decision)
        assert decision["eligible_for_netlib"], "轨迹、覆盖或开发成本未通过事前门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v52-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v52-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v52-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    with (DATA / f"v52-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v52-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v52-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v52-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "reference_solver_sha256_after": sha(EXE if stage == "highs" else PARENT),
        "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "removed_environment": ["OPENBLAS_CORETYPE", "SOLVER_V49_DIAGNOSTICS"]})
    check_solver(51)
    check_solver(52)
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def self_check():
    frozen = plan()
    for stage in ("development", "probe", "netlib", "highs"):
        command = command_for(stage, frozen, ROOT / "unused-v52-self-check")
        pairs = command[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert len(flags) * 2 == len(pairs)
        assert flags["--ipm-linear-accuracy"] == "homogeneous_repair"
        assert flags["--reference-solver"] == str(EXE if stage == "highs" else PARENT)
        if stage == "highs":
            assert not any(k.startswith("--reference-ipm-") for k in flags)
        else:
            assert flags["--reference-ipm-linear-accuracy"] == "homogeneous_repair"
    def sample(a, b, speed=True):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": a, "c": b}}],
            "variants": {"r": {"penalized_numerical_sgm10_seconds": 2, "penalized_wall_sgm10_seconds": 2},
                "c": {"penalized_numerical_sgm10_seconds": 1 if speed else 3, "penalized_wall_sgm10_seconds": 2}}}
    cases = [(3, 3, 0, 0, True, True, True), (3, 2, 0, 3, True, True, False),
             (3, 3, 3, 2, True, True, False), (3, 3, 0, 3, False, True, True),
             (3, 3, 0, 3, True, False, False), (3, 3, 0, 0, False, True, False)]
    for a, b, c, d, speed, same, expected in cases:
        assert gate_decision(sample(a, b, speed), sample(c, d), {"passed": same})["eligible_for_netlib"] is expected
    # 合成轨迹只检验完整报告与退出码合同；不把缺失报告算作预算分叉。
    rows = [{"input": "x", "repetition": 0, "requested_variant": side, "report": None, "returncode": 3221225477} for side in ("r", "c")]
    audit = trajectory_audit({"reference_variant": "r", "candidate_variant": "c"}, rows)
    assert audit["invalid"] == 1 and not audit["passed"]
    for row in rows:
        row["report"] = {"schema_version": 1, "status": "numerical_error", "backend_version": row["requested_variant"],
            "timings": {"solve_seconds": 1, "verification_seconds": 0}, "iterations": 2}
        row["returncode"] = 4
    manifest = {"reference_variant": "r", "candidate_variant": "c"}
    assert trajectory_audit(manifest, rows)["equal"] == 1
    rows[1]["report"]["iterations"] = 3
    assert trajectory_audit(manifest, rows)["different"] == 1
    rows[1]["report"]["timings"]["solve_seconds"] = 6
    assert trajectory_audit(manifest, rows)["budget_limited"] == 1
    rows[1]["report"]["status"] = "verification_failed"
    rows[1]["returncode"] = 3221225477
    assert trajectory_audit(manifest, rows)["invalid"] == 1
    return {"passed": True, "command_cases": 4, "gate_cases": len(cases), "trajectory_cases": 5, "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "development", "probe", "netlib", "highs", "gate"))
    stage = parser.parse_args().stage
    if stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif stage == "gate":
        result = gate()
        save(DATA / "v52-gate.json", result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(stage)
