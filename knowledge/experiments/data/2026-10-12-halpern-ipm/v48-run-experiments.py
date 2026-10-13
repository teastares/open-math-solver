"""按冻结计划串行运行第45轮齐次方法的超节点Cholesky消融，保留全部失败并按门槛控制完整集扩展。"""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
EXE = ROOT / ".tools/solver-ipm-v48.exe"
PLAN = DATA / "v48-plan.json"
FIELDS = ()
SCALARS = ()
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(stage, retries):
    return f"{stage}-homogeneous-normal_cholmod-retries{retries}"


def frozen_plan():
    plan = read(PLAN)
    assert sha(PLAN) == "160ad4060e3135a1146e9b034e0d38103288cd2c377c94e4c28f0145837fb179"
    plan["candidate_linear_solver"] = plan["candidate"]["linear_solver"]
    plan["reference_linear_solver"] = plan["reference"]["linear_solver"]
    identity = DATA / "ipm-v48-identity.json"
    if identity.exists():
        assert read(identity)["runner_sha256"] == sha(Path(__file__))
    return plan


def check_solver(version=48):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    identity = read(DATA / f"ipm-v{version}-identity.json")
    assert sha(executable) == identity["solver_sha256"]
    if version == 48:
        assert identity["plan_sha256"] == sha(PLAN)
    return executable


def archive(stage, retries):
    name = tag(stage, retries)
    manifest = read(DATA / f"v48-{name}-manifest.json")
    summary = read(DATA / f"v48-{name}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v48-{name}-runs.jsonl.gz").read_bytes()).splitlines()]
    plan = frozen_plan()
    expected = plan["sets"][stage]
    assert manifest["models"] == expected["models"] and manifest["repeats"] == expected["repeats"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key, value in (("time_limit", plan["time_limit"]), ("wall_timeout", plan["wall_timeout"]),
                       ("threads", plan["threads"]), ("seed", plan["seed"]),
                       ("ipm_linear_accuracy", "strict"), ("reference_ipm_linear_accuracy", "strict"),
                       ("ipm_method", "homogeneous"), ("reference_ipm_method", "homogeneous"),
                       ("ipm_linear_solver", "normal_cholmod"), ("reference_ipm_linear_solver", "normal_augmented"),
                       ("ipm_direction_retries", retries), ("reference_ipm_direction_retries", retries)):
        assert manifest[key] == value, (key, manifest[key], value)
    for key, value in plan["shared_ipm"].items():
        for prefix in ("ipm_", "reference_ipm_"):
            attribute = prefix + key.replace("-", "_")
            assert manifest[attribute] == (int(value) if key in ("krylov-history-size", "central-corrections") else value)
    reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
    recomputed = compare_lp.summarize(rows, expected["models"], expected["repeats"],
                                      plan["wall_timeout"], plan["time_limit"], (reference, candidate))
    # 比较工具在聚合结果之外还保存配置与原结果目录，完整重建后再核对。
    recomputed["configurations"] = manifest["variants"]
    recomputed["output_dir"] = str(ROOT / f"results/round-ipm-v48-{name}")
    assert summary == recomputed, "汇总与逐次记录不一致"
    return manifest, summary, rows


def gate_decision(development, probe, retries, plan_hash):
    reference, candidate = development["reference_variant"], development["candidate_variant"]
    development_lost = [Path(item["input"]).stem for item in development["models"]
                        if item["quality_pass_counts"][candidate] < item["quality_pass_counts"][reference]]
    before, after = development["variants"][reference], development["variants"][candidate]
    faster = (after["penalized_numerical_sgm10_seconds"] < before["penalized_numerical_sgm10_seconds"] and
              after["penalized_wall_sgm10_seconds"] <= before["penalized_wall_sgm10_seconds"])
    reference, candidate = probe["reference_variant"], probe["candidate_variant"]
    probe_gain = [Path(item["input"]).stem for item in probe["models"]
                  if item["quality_pass_counts"][candidate] > item["quality_pass_counts"][reference]]
    probe_lost = [Path(item["input"]).stem for item in probe["models"]
                  if item["quality_pass_counts"][candidate] < item["quality_pass_counts"][reference]]
    return {"retries": retries, "development_lost": development_lost, "development_time_improved": faster,
            "probe_gain": probe_gain, "probe_lost": probe_lost,
            "eligible_for_netlib": not development_lost and (faster or (bool(probe_gain) and not probe_lost)),
            "plan_sha256": plan_hash}


def gate(retries):
    _, development, _ = archive("development", retries)
    _, probe, _ = archive("probe", retries)
    return gate_decision(development, probe, retries, sha(PLAN))


def comparison_command(stage, retries, plan, output):
    expected = plan["sets"][stage]
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--models-file", expected["models_file"], "--repeats", str(expected["repeats"]),
               "--time-limit", str(plan["time_limit"]), "--wall-timeout", str(plan["wall_timeout"]),
               "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
               "--candidate-backend", "ipm", "--reference-backend", "ipm", "--output-dir", str(output),
               "--iteration-limit", "200", "--reference-iteration-limit", "200",
               "--scaling-passes", "5", "--reference-scaling-passes", "5",
               "--ipm-linear-accuracy", "strict", "--reference-ipm-linear-accuracy", "strict",
               "--ipm-method", "homogeneous", "--reference-ipm-method", "homogeneous",
               "--ipm-linear-solver", plan["candidate_linear_solver"],
               "--reference-ipm-linear-solver", plan["reference_linear_solver"],
               "--ipm-direction-retries", str(retries), "--reference-ipm-direction-retries", str(retries)]
    for key, value in plan["shared_ipm"].items():
        for prefix in ("--ipm-", "--reference-ipm-"):
            command += [prefix + key, value]
    return command


def benchmark(stage, retries):
    plan = frozen_plan()
    check_solver()
    expected = plan["sets"][stage]
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    if stage == "netlib":
        decision = gate(retries)
        save(DATA / f"v48-retries{retries}-gate.json", decision)
        assert decision["eligible_for_netlib"], "开发门槛未通过，禁止扩大完整集"
    name = tag(stage, retries)
    output = ROOT / f"results/round-ipm-v48-{name}"
    assert not output.exists(), "不能覆盖已测结果"
    assert not list(DATA.glob(f"v48-{name}-*")), "不能覆盖已有归档"
    command = comparison_command(stage, retries, plan, output)
    save(DATA / f"v48-{name}-command.json", command)
    with (DATA / f"v48-{name}-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    # 先保存已有产物和退出码，再检查完整性，避免断言丢失失败证据。
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / f"v48-{name}-{filename}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v48-{name}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v48-{name}-process.json", {"returncode": process.returncode, "plan_sha256": sha(PLAN),
                                          "solver_sha256_after": sha(EXE)})
    check_solver()
    assert process.returncode in (0, 1), (name, process.returncode)
    _, summary, _ = archive(stage, retries)
    print(json.dumps({"tag": name, "variants": summary["variants"]}), flush=True)


def highs_command(plan, output):
    expected = plan["sets"]["netlib"]
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--models-file", expected["models_file"], "--repeats", "1", "--time-limit", "5",
               "--wall-timeout", "35", "--threads", "1", "--seed", "0", "--candidate-backend", "ipm",
               "--reference-backend", "highs", "--reference-highs-solver", "choose",
               "--reference-highs-presolve", "choose", "--reference-highs-crossover", "on",
               "--iteration-limit", "200", "--scaling-passes", "5", "--ipm-direction-retries", "0",
               "--ipm-linear-accuracy", "strict", "--ipm-method", "homogeneous",
               "--ipm-linear-solver", plan["candidate_linear_solver"], "--output-dir", str(output)]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def highs_benchmark():
    plan = frozen_plan()
    assert gate(0)["eligible_for_netlib"], "开发门槛未通过"
    archive("netlib", 0)
    check_solver()
    expected = plan["sets"]["netlib"]
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / "results/round-ipm-v48-highs"
    assert not output.exists() and not list(DATA.glob("v48-highs-*")), "不能覆盖已有结果"
    command = highs_command(plan, output)
    save(DATA / "v48-highs-command.json", command)
    with (DATA / "v48-highs-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / ("v48-highs-" + filename))
    if (output / "runs.jsonl").exists():
        (DATA / "v48-highs-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / "v48-highs-process.json", {"returncode": process.returncode,
         "plan_sha256": sha(PLAN), "solver_sha256_after": sha(EXE)})
    check_solver()
    assert process.returncode in (0, 1)
    manifest, summary = read(output / "manifest.json"), read(output / "summary.json")
    rows = [json.loads(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["models"] == expected["models"] and len(rows) == 182
    recomputed = compare_lp.summarize(rows, models, 1, 35, 5,
                                      (manifest["reference_variant"], manifest["candidate_variant"]))
    recomputed.update(configurations=manifest["variants"], output_dir=str(output))
    assert summary == recomputed
    print(json.dumps({"stage": "highs", "variants": summary["variants"]}), flush=True)


def trajectories():
    frozen_plan()
    executables = [check_solver(version) for version in (46, 48)]
    models = read(PLAN)["sets"]["development"]["models"][:8]
    records = []
    destination = DATA / "v48-old-mode-trajectory-runs.json.gz"
    assert not destination.exists(), "不能覆盖旧轨迹归档"
    for linear, accuracy, retries, presolve, method in (("qdldl", "strict", 0, "off", "mehrotra"),
            ("normal_augmented", "strict", 0, "off", "mehrotra"),
            ("normal_augmented", "blockwise", 8, "on", "mehrotra"),
            ("normal_augmented", "full_newton", 0, "off", "mehrotra"),
            ("normal_augmented", "bounded_newton", 0, "off", "mehrotra"),
            ("normal_augmented", "strict", 0, "off", "homogeneous"),
            ("normal_multifrontal", "strict", 0, "off", "homogeneous")):
        for model in models:
            reports, commands, codes, hashes, outputs = [], [], [], [], []
            row = {"model": model["input"], "model_sha256": model["sha256"], "commands": commands,
                   "returncodes": codes, "reports": reports, "outputs": outputs, "hashes": hashes,
                   "method_preserved": False, "non_time_equal": False, "budget_limited": False}
            records.append(row)
            for executable in executables:
                command = [str(executable), "solve", model["input"], "--backend", "ipm", "--time-limit", "5",
                           "--threads", "1", "--seed", "0", "--ipm-linear-solver", linear,
                           "--ipm-method", method, "--ipm-linear-accuracy", accuracy, "--ipm-direction-retries", str(retries),
                           "--ipm-krylov-history-size", "0", "--ipm-kkt-equilibration", "on",
                           "--ipm-singleton-elimination", "off" if linear == "qdldl" else "on",
                           "--ipm-fixed-variable-elimination", "off" if linear == "qdldl" else "on",
                           "--ipm-singleton-presolve", presolve, "--ipm-presolve-fixed-elimination", presolve]
                before = {"solver": sha(executable), "input": sha(Path(model["input"]))}
                commands.append(command)
                try:
                    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=35)
                    outputs.append({"stdout": process.stdout, "stderr": process.stderr})
                    codes.append(process.returncode)
                except subprocess.TimeoutExpired as error:
                    # 超时异常也先保存可用原始输出，不将未完成对子改写为一致。
                    def decoded(value):
                        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
                    outputs.append({"stdout": decoded(error.stdout), "stderr": decoded(error.stderr), "timeout": True})
                    codes.append(None)
                    destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
                    raise
                hashes.append({"before": before, "after": {"solver": sha(executable), "input": sha(Path(model["input"]))}})
                destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
                reports.append(json.loads(process.stdout))
            normalized = copy.deepcopy(reports)
            fields_valid = all(type(reports[1].get(field)) is int and reports[1][field] == 0 for field in FIELDS)
            fields_valid = fields_valid and all(reports[1].get(field) is None for field in SCALARS)
            fields_valid = fields_valid and reports[1]["options"]["ipm"]["method"] == method
            for report in normalized:
                for key in ("timings", "backend_version") + FIELDS + SCALARS:
                    report.pop(key, None)
            row["method_preserved"] = fields_valid
            row["non_time_equal"] = normalized[0] == normalized[1] and codes[0] == codes[1]
            row["budget_limited"] = any(report.get("status") == "time_limit" or
                report["timings"]["solve_seconds"] + report["timings"]["verification_seconds"] > 5 for report in reports)
            destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
    for version in (46, 48):
        check_solver(version)
    assert len(records) == 56
    assert all((row["non_time_equal"] or row["budget_limited"]) and row["method_preserved"] for row in records)
    assert all(item["before"] == item["after"] and item["before"]["input"] == row["model_sha256"]
               for row in records for item in row["hashes"])
    print(json.dumps({"pairs": len(records), "old_trajectories_equal": sum(row["non_time_equal"] for row in records),
        "budget_limited_pairs": sum(row["budget_limited"] for row in records)}), flush=True)



def details():
    groups = {"development": {}, "probe": {}, "coverage": {}}
    for stage in ("development", "probe", "netlib"):
        for retries in (0,):
            name = tag(stage, retries)
            if not (DATA / f"v48-{name}-summary.json").exists():
                continue
            manifest, summary, rows = archive(stage, retries)
            reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
            passed = {side: {Path(item["input"]).stem for item in summary["models"]
                             if item["quality_pass_counts"][side] == manifest["repeats"]}
                      for side in (reference, candidate)}
            entries = []
            for row in rows:
                report = row.get("report") or {}
                entries.append({"model": Path(row["input"]).stem, "candidate": row["requested_variant"] == candidate,
                                "repetition": row.get("repetition"), "quality_pass": row["quality_pass"],
                                "status": row["status"], "quality_failure": row["quality_failure"],
                                "numerical_seconds": row["numerical_seconds"], "message": report.get("message"),
                                "work": {k: v for k, v in report.items() if k.startswith("ipm_")},
                                "optimality": report.get("lp_optimality")})
            groups["coverage" if stage == "netlib" else stage][name] = {
                "summary": summary["variants"], "gain": sorted(passed[candidate] - passed[reference]),
                "lost": sorted(passed[reference] - passed[candidate]), "models": entries}
    for key, value in groups.items():
        save(DATA / f"v48-{key}-detail.json", value)
    for retries in (0,):
        if all((DATA / f"v48-{tag(stage, retries)}-summary.json").exists() for stage in ("development", "probe")):
            save(DATA / f"v48-retries{retries}-gate.json", gate(retries))


def self_check():
    # 仅使用已有计划作命令输入和内存聚合量作门槛输入，不启动进程或写入结果文件。
    plan = read(DATA / "v48-plan.json")
    plan["candidate_linear_solver"] = "normal_cholmod"
    plan["reference_linear_solver"] = "normal_augmented"
    plan["shared_ipm"].pop("linear-solver", None)
    def command_flags(command):
        pairs = command[2:]
        assert len(pairs) % 2 == 0
        result = dict(zip(pairs[::2], pairs[1::2]))
        assert len(result) * 2 == len(pairs)
        return result
    for stage in ("development", "probe", "netlib"):
        flags = command_flags(comparison_command(stage, 0, plan, ROOT / "unused-self-check-output"))
        assert flags["--solver"] == str(EXE)
        assert flags["--ipm-method"] == flags["--reference-ipm-method"] == "homogeneous"
        assert flags["--ipm-linear-accuracy"] == flags["--reference-ipm-linear-accuracy"] == "strict"
        assert flags["--ipm-linear-solver"] == "normal_cholmod" and flags["--reference-ipm-linear-solver"] == "normal_augmented"
        assert flags["--iteration-limit"] == flags["--reference-iteration-limit"] == "200"
        assert (flags["--time-limit"], flags["--wall-timeout"], flags["--threads"], flags["--seed"]) == ("5", "35", "1", "0")
        assert all(flags["--ipm-" + key] == flags["--reference-ipm-" + key] == value for key, value in plan["shared_ipm"].items())
    flags = command_flags(highs_command(plan, ROOT / "unused-self-check-output"))
    assert (flags["--reference-backend"], flags["--reference-highs-solver"], flags["--reference-highs-presolve"], flags["--reference-highs-crossover"]) == ("highs", "choose", "choose", "on")
    assert flags["--ipm-linear-solver"] == "normal_cholmod" and flags["--ipm-method"] == "homogeneous"
    assert not any(key.startswith("--reference-ipm-") for key in flags)
    def summary(before=3, after=3, numerical=(1., 1.), wall=(1., 1.)):
        return {"reference_variant": "r", "candidate_variant": "c",
                "models": [{"input": "synthetic.mps", "quality_pass_counts": {"r": before, "c": after}}],
                "variants": {"r": {"penalized_numerical_sgm10_seconds": numerical[0], "penalized_wall_sgm10_seconds": wall[0]},
                             "c": {"penalized_numerical_sgm10_seconds": numerical[1], "penalized_wall_sgm10_seconds": wall[1]}}}
    cases = [(summary(numerical=(1, .9), wall=(1, .9)), summary(0, 0), True),
             (summary(), summary(0, 0), False),
             (summary(3, 2, numerical=(1, .5), wall=(1, .5)), summary(0, 1), False),
             (summary(numerical=(1, .9), wall=(1, 1.1)), summary(0, 0), False),
             (summary(numerical=(1, 1.1)), summary(0, 1), True),
             (summary(), summary(1, 0), False)]
    mixed = summary(0, 1)
    mixed["models"].append({"input": "synthetic-loss.mps", "quality_pass_counts": {"r": 1, "c": 0}})
    cases.append((summary(), mixed, False))
    for development, probe, expected in cases:
        assert gate_decision(development, probe, 0, "synthetic-plan")["eligible_for_netlib"] is expected
    return {"passed": True, "command_configurations": 4, "gate_scenarios": 7, "checks": 11, "solver_runs": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("trajectories", "self-check", "development", "probe", "netlib", "highs", "details", "gate"))
    parser.add_argument("--retries", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if args.stage == "trajectories":
        trajectories()
    elif args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    elif args.stage == "highs":
        highs_benchmark()
    elif args.stage == "details":
        details()
    elif args.stage == "gate":
        print(json.dumps(gate(args.retries), ensure_ascii=False), flush=True)
    else:
        benchmark(args.stage, args.retries)
