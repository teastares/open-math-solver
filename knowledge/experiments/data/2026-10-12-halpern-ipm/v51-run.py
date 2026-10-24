"""第48轮只改变BLAS修复的两程序串行对照；失败保留，不覆盖已有结果。"""
import argparse
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
PLAN = DATA / "v51-plan.json"
PLAN_SHA = "1ceb01fda1ac152d6d2753dfa7f38bc16d8a939b6d2602f4e567a9d1cbf25219"
EXE = ROOT / ".tools/solver-ipm-v51.exe"
PARENT = ROOT / ".tools/solver-ipm-v50.exe"
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
    assert sha(DATA / "ipm-v50-identity.json") == value["parent_identity_sha256"]
    identity = DATA / "ipm-v51-identity.json"
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
    manifest = read(DATA / f"v51-{stage}-manifest.json")
    summary = read(DATA / f"v51-{stage}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v51-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
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
    for variant, version in ((reference, 51 if stage == "highs" else 50), (candidate, 51)):
        solver = manifest["solvers"][variant]
        assert Path(solver["path"]) == check_solver(version)
        assert solver["sha256"] == sha(solver["path"]) and solver["hash_error"] is None
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v51-{stage}"))
    assert recomputed == summary
    return manifest, summary, rows


def gate_decision(development, probe, correctness, stability):
    def lost(summary):
        r, c = summary["reference_variant"], summary["candidate_variant"]
        return [Path(item["input"]).stem for item in summary["models"]
                if item["quality_pass_counts"][c] < item["quality_pass_counts"][r]]
    dev, pro = lost(development), lost(probe)
    return {"development_lost": dev, "probe_lost": pro, "correctness_passed": correctness,
            "stability_passed": stability, "eligible_for_netlib": not dev and not pro and correctness and stability,
            "plan_sha256": PLAN_SHA}


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    validation = read(DATA / "v51-validation.json")
    assert validation["solver_sha256"] == sha(EXE)
    assert validation["plan_sha256"] == PLAN_SHA
    stability = read(DATA / "v51-stability.json")
    assert stability["solver_sha256"] == sha(EXE) and stability["plan_sha256"] == PLAN_SHA
    return gate_decision(development, probe, validation["passed"], stability["passed"])


def run(stage):
    frozen = plan()
    check_solver(50)
    check_solver(51)
    if stage in ("netlib", "highs"):
        decision = gate()
        destination = DATA / "v51-gate.json"
        if destination.exists():
            assert read(destination) == decision
        else:
            save(destination, decision)
        assert decision["eligible_for_netlib"], "正确性、稳定性或初始覆盖未通过门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v51-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v51-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v51-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    with (DATA / f"v51-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v51-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v51-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v51-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "reference_solver_sha256_after": sha(EXE if stage == "highs" else PARENT),
        "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "removed_environment": ["OPENBLAS_CORETYPE", "SOLVER_V49_DIAGNOSTICS"]})
    check_solver(50)
    check_solver(51)
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def self_check():
    frozen = plan()
    for stage in ("development", "probe", "netlib", "highs"):
        command = command_for(stage, frozen, ROOT / "unused-v51-self-check")
        pairs = command[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert len(flags) * 2 == len(pairs)
        assert flags["--ipm-linear-accuracy"] == "homogeneous_repair"
        assert flags["--reference-solver"] == str(EXE if stage == "highs" else PARENT)
        if stage == "highs":
            assert not any(k.startswith("--reference-ipm-") for k in flags)
        else:
            assert flags["--reference-ipm-linear-accuracy"] == "homogeneous_repair"
    def sample(a, b):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": a, "c": b}}]}
    cases = [(3, 3, 0, 0, True, True, True), (3, 2, 0, 3, True, True, False),
             (3, 3, 3, 2, True, True, False), (3, 3, 0, 3, False, True, False),
             (3, 3, 0, 3, True, False, False), (3, 3, 0, 3, True, True, True)]
    for a, b, c, d, correct, stable, expected in cases:
        assert gate_decision(sample(a, b), sample(c, d), correct, stable)["eligible_for_netlib"] is expected
    return {"passed": True, "command_cases": 4, "gate_cases": len(cases), "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "development", "probe", "netlib", "highs", "gate"))
    stage = parser.parse_args().stage
    if stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif stage == "gate":
        result = gate()
        save(DATA / "v51-gate.json", result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(stage)
