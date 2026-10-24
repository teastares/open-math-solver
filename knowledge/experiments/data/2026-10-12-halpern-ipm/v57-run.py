"""第54轮Gram冗余检查消融：旧v56与新v57均关闭目标归一化，按质量、成本与完整轨迹门槛扩集。"""
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
PLAN = DATA / "v57-plan.json"
PLAN_SHA = "c28c0ad28ca89dd04887b5033cc07289966afe0e7a9447737b0339317ad831f3"
EXE = ROOT / ".tools/solver-ipm-v57.exe"
PARENT = ROOT / ".tools/solver-ipm-v56.exe"
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
    assert sha(DATA / "ipm-v56-identity.json") == value["parent_identity_sha256"]
    assert sha(DATA / "v56-verification.json") == value["evidence_sha256"]
    assert (value["parent_version"], value["reference_version"], value["candidate_version"],
            value["reference_objective_scaling"], value["candidate_objective_scaling"]) == (56, 56, 57, "off", "off")
    identity = DATA / "ipm-v57-identity.json"
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
    manifest = read(DATA / f"v57-{stage}-manifest.json")
    summary = read(DATA / f"v57-{stage}-summary.json")
    rows = [benchmark.strict_json(line) for line in gzip.decompress((DATA / f"v57-{stage}-runs.jsonl.gz").read_bytes()).splitlines()]
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
    for variant, version in ((reference, 57 if stage == "highs" else 56), (candidate, 57)):
        solver = manifest["solvers"][variant]
        assert Path(solver["path"]) == check_solver(version)
        assert solver["sha256"] == sha(solver["path"]) and solver["hash_error"] is None
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
        frozen["wall_timeout"], frozen["time_limit"], (reference, candidate))
    recomputed.update(configurations=manifest["variants"], output_dir=str(ROOT / f"results/round-ipm-v57-{stage}"))
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


def requested_ipm_options(configuration):
    booleans = {"ipm_singleton_elimination": "eliminate_singletons",
        "ipm_dynamic_regularization": "dynamic_regularization", "ipm_kkt_equilibration": "equilibrate_kkt",
        "ipm_fixed_variable_elimination": "eliminate_fixed_variables", "ipm_singleton_presolve": "presolve_singleton_rows",
        "ipm_presolve_fixed_elimination": "eliminate_presolved_fixed_variables", "ipm_inexact_prefilter": "inexact_prefilter",
        "ipm_objective_scaling": "objective_scaling"}
    result = {field: configuration[key] == "on" for key, field in booleans.items()}
    result.update({field: configuration[key] for key, field in {
        "ipm_method": "method", "ipm_linear_solver": "linear_solver", "ipm_linear_accuracy": "linear_accuracy",
        "iteration_limit": "max_iterations", "scaling_passes": "scaling_passes",
        "ipm_direction_retries": "direction_retries", "ipm_central_corrections": "central_corrections",
        "ipm_krylov_history_size": "krylov_history_size"}.items()})
    return result


def trajectory_audit(manifest, rows):
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    assert len(set(sides)) == 2
    pairs = {}
    for row in rows:
        assert row["requested_variant"] in sides
        key = (row["input"], row["repetition"])
        pair = pairs.setdefault(key, {})
        assert row["requested_variant"] not in pair
        pair[row["requested_variant"]] = row
    result = {"pairs": len(pairs), "equal": 0, "budget_limited": 0, "invalid": 0, "different": 0}
    exits = {"resolved": 0, "incomplete": 4, "verification_failed": 3}
    for pair in pairs.values():
        assert set(pair) == set(sides)
        a, b = (pair[side] for side in sides)
        valid = True
        for index, row in enumerate((a, b)):
            report = row.get("report")
            if (row.get("error") is not None or type(row.get("returncode")) is not int or
                    not isinstance(report, dict)):
                valid = False
                continue
            classification = benchmark.classify(row["returncode"], report)
            timings, options = report.get("timings"), report.get("options")
            if (classification not in exits or row["returncode"] != exits[classification] or
                    not isinstance(timings, dict) or not all(benchmark.finite_real(timings.get(key)) and timings[key] >= 0
                    for key in ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")) or
                    not benchmark.finite_real(row.get("elapsed_wall_seconds")) or row["elapsed_wall_seconds"] <= 0 or
                    report.get("backend") != "ipm" or report.get("input") != row["input"] or
                    report.get("backend_version") != ("ipm-cpu-0.46.0" if index == 0 else "ipm-cpu-0.47.0") or
                    not isinstance(report.get("model"), dict) or report["model"].get("is_mip") is not False or
                    not isinstance(options, dict)):
                valid = False
                continue
            configuration = manifest["variants"][sides[index]]
            requested = requested_ipm_options(configuration)
            if (configuration["backend"] != "ipm" or options.get("backend") != "ipm" or
                    any(options.get(key) != manifest[key] for key in ("time_limit", "threads", "seed")) or
                    type(options.get("threads")) is not int or type(options.get("seed")) is not int or
                    not benchmark.finite_real(options.get("time_limit")) or
                    json.dumps(options.get("ipm"), sort_keys=True, allow_nan=False) != json.dumps(requested, sort_keys=True, allow_nan=False)):
                valid = False
        # 缺报告、异常退出和错误请求先判无效，不能借另一侧超预算获得豁免。
        if not valid:
            result["invalid"] += 1
            continue
        budget = any(row["report"]["status"] == "time_limit" or
            row["report"]["timings"]["solve_seconds"] + row["report"]["timings"]["verification_seconds"] > manifest["time_limit"] or
            row["elapsed_wall_seconds"] > manifest["wall_timeout"] for row in (a, b))
        if budget:
            result["budget_limited"] += 1
            continue
        normalized = copy.deepcopy([a["report"], b["report"]])
        for report in normalized:
            del report["timings"]
            del report["backend_version"]
        same = (json.dumps(normalized[0], sort_keys=True, allow_nan=False) ==
                json.dumps(normalized[1], sort_keys=True, allow_nan=False) and a["returncode"] == b["returncode"])
        result["equal" if same else "different"] += 1
    result["passed"] = result["invalid"] == result["different"] == 0
    return result


def gate():
    ma, development, a = archive("development")
    mb, probe, b = archive("probe")
    audits = {"development": trajectory_audit(ma, a), "probe": trajectory_audit(mb, b)}
    audits["passed"] = all(value["passed"] for value in audits.values())
    return gate_decision(development, probe, audits)


def netlib_trajectories():
    manifest, _, rows = archive("netlib")
    audit = trajectory_audit(manifest, rows)
    result = {"plan_sha256": PLAN_SHA, "stage": "netlib", **audit}
    destination = DATA / "v57-netlib-trajectories.json"
    if destination.exists():
        assert read(destination) == result, "已有完整轨迹证据不一致，禁止覆盖"
    else:
        save(destination, result)
    return result


def save_gate(decision):
    destination = DATA / "v57-gate.json"
    if destination.exists():
        assert read(destination) == decision, "已有门槛证据不一致，禁止覆盖"
    else:
        save(destination, decision)


def run(stage):
    frozen = plan()
    check_solver(56)
    check_solver(57)
    if stage in ("netlib", "highs"):
        decision = gate()
        save_gate(decision)
        assert decision["eligible_for_netlib"], "逐例覆盖或开发成本未通过事前门槛"
        if stage == "highs":
            assert netlib_trajectories()["passed"], "完整旧新非预算轨迹不一致，不得进入HiGHS对照"
    expected = specification(stage, frozen)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / f"results/round-ipm-v57-{stage}"
    assert not output.exists() and not list(DATA.glob(f"v57-{stage}-*")), "禁止覆盖结果"
    command = command_for(stage, frozen, output)
    save(DATA / f"v57-{stage}-command.json", command)
    protected = {str(ROOT / "scripts" / name): sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")}
    identity = read(DATA / "ipm-v57-identity.json")
    assert all(digest == identity["files"]["scripts/" + Path(path).name] for path, digest in protected.items())
    with (DATA / f"v57-{stage}-run.log").open("x", encoding="utf-8") as target:
        process = subprocess.run(command, cwd=ROOT, env=environment(), stdout=target, stderr=subprocess.STDOUT)
    for name in ("manifest.json", "summary.json"):
        if (output / name).exists():
            shutil.copy2(output / name, DATA / f"v57-{stage}-{name}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v57-{stage}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v57-{stage}-process.json", {"returncode": process.returncode, "plan_sha256": PLAN_SHA,
        "solver_sha256_after": sha(EXE), "reference_solver_sha256_after": sha(EXE if stage == "highs" else PARENT),
        "protected_scripts": protected,
        "environment": {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "removed_environment": ["OPENBLAS_CORETYPE", "所有SOLVER_V前缀变量"]})
    check_solver(56)
    check_solver(57)
    assert all(sha(path) == value for path, value in protected.items())
    assert process.returncode in (0, 1)
    _, summary, _ = archive(stage)
    if stage == "netlib":
        assert netlib_trajectories()["passed"], "完整旧新轨迹失败，已保留全部原始记录与审计"
    print(json.dumps({"stage": stage, "variants": summary["variants"]}), flush=True)


def self_check():
    frozen = plan()
    for stage in STAGES:
        command = command_for(stage, frozen, ROOT / "unused-v57-self-check")
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
        pairs = command_for("development", altered, ROOT / "unused-v57-self-check")[2:]
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
        assert gate_decision(sample(a, b, numerical, wall), sample(c, d), {"passed": True})["eligible_for_netlib"] is expected
    assert gate_decision(sample(3, 3), sample(0, 3), {"passed": False})["eligible_for_netlib"] is False
    configuration = {"backend": "ipm", "ipm_method": "homogeneous", "ipm_linear_solver": "normal_cholmod",
        "ipm_linear_accuracy": "homogeneous_repair", "iteration_limit": 200, "scaling_passes": 5,
        "ipm_direction_retries": 0, "ipm_central_corrections": 0, "ipm_krylov_history_size": 0,
        "ipm_singleton_elimination": "on", "ipm_dynamic_regularization": "off", "ipm_kkt_equilibration": "on",
        "ipm_fixed_variable_elimination": "on", "ipm_singleton_presolve": "off",
        "ipm_presolve_fixed_elimination": "off", "ipm_inexact_prefilter": "off", "ipm_objective_scaling": "off"}
    manifest = {"reference_variant": "r", "candidate_variant": "c", "variants": {"r": configuration, "c": configuration},
        "time_limit": 5, "wall_timeout": 35, "threads": 1, "seed": 0}
    original = [{"input": "synthetic.mps", "repetition": 0, "requested_variant": side,
        "error": None, "returncode": 4, "elapsed_wall_seconds": 2,
        "report": {"schema_version": 1, "status": "numerical_error", "input": "synthetic.mps", "backend": "ipm",
            "backend_version": "ipm-cpu-0.46.0" if side == "r" else "ipm-cpu-0.47.0", "iterations": 2,
            "model": {"is_mip": False}, "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1},
            "options": {"backend": "ipm", "time_limit": 5, "threads": 1, "seed": 0,
                "ipm": requested_ipm_options(configuration)}}} for side in ("r", "c")]
    trajectory_cases = 0
    def audit(rows, category):
        nonlocal trajectory_cases
        result = trajectory_audit(manifest, rows)
        assert result[category] == 1 and result["passed"] is (category in ("equal", "budget_limited"))
        trajectory_cases += 1
    audit(original, "equal")
    rows = copy.deepcopy(original)
    rows[1]["report"]["iterations"] = 3
    audit(rows, "different")
    rows[1]["report"]["timings"]["solve_seconds"] = 6
    audit(rows, "budget_limited")
    rows = copy.deepcopy(original)
    rows[1]["report"]["status"] = "time_limit"
    audit(rows, "budget_limited")
    rows = copy.deepcopy(original)
    rows[1]["elapsed_wall_seconds"] = 36
    audit(rows, "budget_limited")
    # 所有污染反例均让另一侧超时，确保先拒无效再判断预算。
    mutations = [
        lambda row: row.update(report=None),
        lambda row: row.update(returncode=3221225477),
        lambda row: row.update(returncode=True),
        lambda row: row.update(returncode=3),
        lambda row: row.update(error="wall_timeout"),
        lambda row: row.update(elapsed_wall_seconds=0),
        lambda row: row.update(elapsed_wall_seconds=True),
        lambda row: row["report"].update(backend_version="ipm-cpu-0.46.0"),
        lambda row: row["report"].update(backend="highs"),
        lambda row: row["report"].update(input="other.mps"),
        lambda row: row["report"].update(options=None),
        lambda row: row["report"]["timings"].update(solve_seconds=float("nan")),
        lambda row: row["report"]["timings"].update(solve_seconds=-1),
        lambda row: row["report"]["timings"].update(solve_seconds=True),
        lambda row: row["report"]["timings"].pop("total_seconds"),
        lambda row: row["report"]["options"]["ipm"].update(objective_scaling=True),
        lambda row: row["report"]["options"]["ipm"].update(objective_scaling=0),
        lambda row: row["report"]["options"]["ipm"].update(max_iterations=201),
        lambda row: row["report"]["options"].update(threads=True),
    ]
    for mutate in mutations:
        rows = copy.deepcopy(original)
        rows[0]["report"]["status"] = "time_limit"
        mutate(rows[1])
        audit(rows, "invalid")
    rows = copy.deepcopy(original)
    for row in rows:
        row["report"]["status"] = "verification_failed"
        row["returncode"] = 3
    audit(rows, "equal")
    rows[1]["returncode"] = 3221225477
    rows[0]["report"]["timings"]["solve_seconds"] = 6
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[0]["report"]["extra"] = False
    rows[1]["report"]["extra"] = 0
    audit(rows, "different")
    rows = copy.deepcopy(original)
    rows[1]["report"]["extra"] = "保留完整报告字段"
    audit(rows, "different")
    return {"passed": True, "command_cases": len(STAGES), "configuration_cases": configuration_cases,
            "gate_cases": len(cases) + 1, "trajectory_cases": trajectory_cases, "solver_runs": 0}


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
