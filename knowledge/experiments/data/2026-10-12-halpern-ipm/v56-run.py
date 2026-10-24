"""第53轮Gram除法外提消融：旧v54与新v56均关闭目标归一化，按事前门槛扩集。"""
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
PLAN = DATA / "v56-plan.json"
PLAN_SHA = "5b5f38473b95d3b5c7d3d2728aaa0f444501763c54c5e1b3a2d09637a2227729"
EXE = ROOT / ".tools/solver-ipm-v56.exe"
PARENT = ROOT / ".tools/solver-ipm-v54.exe"
STAGES = ("development", "probe", "netlib", "highs")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def plan():
    assert sha(PLAN) == PLAN_SHA
    value = read(PLAN)
    assert sha(DATA / "ipm-v54-identity.json") == value["parent_identity_sha256"]
    assert sha(DATA / "v55-verification.json") == value["evidence_sha256"]
    assert (value["parent_version"], value["reference_version"], value["candidate_version"],
            value["reference_objective_scaling"], value["candidate_objective_scaling"]) == (54, 54, 56, "off", "off")
    identity = DATA / "ipm-v56-identity.json"
    if identity.exists():
        frozen = read(identity)
        assert frozen["runner_sha256"] == sha(__file__) and frozen["plan_sha256"] == PLAN_SHA
    return value


def environment():
    result = os.environ.copy()
    result.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    for key in tuple(result):
        if key.startswith("SOLVER_V"):
            result.pop(key)
    result.pop("OPENBLAS_CORETYPE", None)
    return result


def check_solver(version):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    assert sha(executable) == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    return executable


def specification(stage, frozen):
    assert stage in STAGES
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
    for prefix, side in (("--", "candidate"), ("--reference-", "reference")):
        if stage == "highs" and side == "reference":
            for name, value in frozen["highs_reference"].items():
                command += [prefix + "highs-" + name, value]
            continue
        command += [prefix + "iteration-limit", str(frozen["max_iterations"]),
                    prefix + "scaling-passes", str(frozen["scaling_passes"]),
                    prefix + "ipm-method", frozen["method"],
                    prefix + "ipm-linear-accuracy", frozen[side + "_accuracy"],
                    prefix + "ipm-linear-solver", frozen["primary_linear_solver"],
                    prefix + "ipm-direction-retries", "0"]
        for key, value in frozen["shared_ipm"].items():
            command += [prefix + "ipm-" + key, value]
    command += ["--ipm-objective-scaling", frozen["candidate_objective_scaling"]]
    if stage != "highs":
        command += ["--reference-ipm-objective-scaling", frozen["reference_objective_scaling"]]
    return command


def archive(stage):
    frozen = plan()
    expected = specification(stage, frozen)
    manifest = read(DATA / f"v56-{stage}-manifest.json")
    summary = read(DATA / f"v56-{stage}-summary.json")
    rows = [benchmark.strict_json(line) for line in gzip.decompress((DATA / f"v56-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
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
    assert manifest["ipm_objective_scaling"] == frozen["candidate_objective_scaling"]
    assert manifest["reference_ipm_objective_scaling"] == (None if stage == "highs" else frozen["reference_objective_scaling"])
    for variant, version in ((reference, 56 if stage == "highs" else 54), (candidate, 56)):
        solver = manifest["solvers"][variant]
        assert Path(solver["path"]) == check_solver(version)
        assert solver["sha256"] == sha(solver["path"]) and solver["hash_error"] is None
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v56-{stage}"))
    assert recomputed == summary
    return manifest, summary, rows


def gate_decision(development, probe):
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
            "development_speed_gate": speed, "eligible_for_netlib": not dev and not pro and (speed or bool(gain)),
            "plan_sha256": PLAN_SHA}


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    return gate_decision(development, probe)


def save_gate(decision):
    destination = DATA / "v56-gate.json"
    if destination.exists():
        assert read(destination) == decision, "已有门槛证据不一致，禁止覆盖"
    else:
        save(destination, decision)


def run(stage):
    frozen = plan()
    check_solver(54)
    check_solver(56)
    if stage in ("netlib", "highs"):
        decision = gate()
        save_gate(decision)
        assert decision["eligible_for_netlib"], "逐例覆盖或开发成本未通过事前门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v56-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v56-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v56-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    identity = read(DATA / "ipm-v56-identity.json")
    assert all(digest == identity["files"]["scripts/" + Path(path).name] for path, digest in protected.items())
    with (DATA / f"v56-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v56-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v56-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v56-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "reference_solver_sha256_after": sha(EXE if stage == "highs" else PARENT),
        "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "removed_environment": ["OPENBLAS_CORETYPE", "所有SOLVER_V前缀变量"]})
    check_solver(54)
    check_solver(56)
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def self_check():
    frozen = plan()
    for stage in STAGES:
        command = command_for(stage, frozen, ROOT / "unused-v56-self-check")
        pairs = command[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert len(flags) * 2 == len(pairs)
        assert flags["--solver"] == str(EXE) and flags["--reference-solver"] == str(EXE if stage == "highs" else PARENT)
        assert flags["--ipm-linear-accuracy"] == "homogeneous_repair" and flags["--ipm-objective-scaling"] == "off"
        assert flags["--candidate-backend"] == "ipm" and flags["--reference-backend"] == ("highs" if stage == "highs" else "ipm")
        if stage == "highs":
            assert not any(key.startswith("--reference-ipm-") for key in flags)
            assert {key: flags["--reference-highs-" + key] for key in frozen["highs_reference"]} == frozen["highs_reference"]
        else:
            assert flags["--reference-ipm-linear-accuracy"] == "homogeneous_repair" and flags["--reference-ipm-objective-scaling"] == "off"
    configuration_cases = 0
    for reference, candidate in (("strict", "homogeneous_repair"), ("homogeneous_repair", "strict")):
        altered = copy.deepcopy(frozen)
        altered.update(reference_accuracy=reference, candidate_accuracy=candidate, reference_objective_scaling="on", candidate_objective_scaling="off")
        pairs = command_for("development", altered, ROOT / "unused-v56-self-check")[2:]
        flags = dict(zip(pairs[::2], pairs[1::2]))
        assert flags["--reference-ipm-linear-accuracy"] == reference and flags["--ipm-linear-accuracy"] == candidate
        assert flags["--reference-ipm-objective-scaling"] == "on" and flags["--ipm-objective-scaling"] == "off"
        configuration_cases += 1
    def sample(a, b, numerical=1, wall=2):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": a, "c": b}}],
            "variants": {"r": {"penalized_numerical_sgm10_seconds": 2, "penalized_wall_sgm10_seconds": 2},
                "c": {"penalized_numerical_sgm10_seconds": numerical, "penalized_wall_sgm10_seconds": wall}}}
    cases = [(3, 3, 0, 0, 1, 2, True), (3, 2, 0, 3, 1, 1, False), (3, 3, 3, 2, 1, 1, False),
             (3, 3, 0, 3, 3, 3, True), (3, 3, 0, 0, 3, 1, False), (3, 3, 0, 0, 1, 3, False),
             (3, 3, 0, 0, 2, 2, False), (3, 3, 0, 0, 1, 1, True), (0, 0, 0, 1, 2, 2, True),
             (3, 3, 0, 0, 1, 2.000000001, False)]
    for a, b, c, d, numerical, wall, expected in cases:
        assert gate_decision(sample(a, b, numerical, wall), sample(c, d))["eligible_for_netlib"] is expected
    return {"passed": True, "command_cases": len(STAGES), "configuration_cases": configuration_cases,
            "gate_cases": len(cases), "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", *STAGES, "gate"))
    stage = parser.parse_args().stage
    if stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif stage == "gate":
        result = gate()
        save_gate(result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(stage)
