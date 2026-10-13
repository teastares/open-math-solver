"""按冻结计划串行运行第40轮有界 Newton 消融，保留全部失败并按门槛控制完整集扩展。"""
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
EXE = ROOT / ".tools/solver-ipm-v43.exe"
PLAN = DATA / "v43-plan.json"
FIELDS = ("ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns")
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(stage, retries):
    return f"{stage}-normal_augmented-bounded_newton-retries{retries}"


def prepare_plan():
    assert not PLAN.exists(), "不能覆盖已经冻结的计划"
    probe = DATA / "v43-known-failures.txt"
    assert not probe.exists(), "不能覆盖已经冻结的探针输入"
    probe.write_text("\n".join("../../../../.deps/netlib/" + name + ".mps" for name in
                              ("dfl001", "finnis", "greenbea", "bnl2", "grow22")) + "\n", encoding="utf-8")
    sets = {}
    for stage, path, repeats, expected in (("development", ROOT / "benchmarks/development.txt", 3, 8),
                                          ("probe", probe, 1, 5),
                                          ("netlib", ROOT / "benchmarks/netlib-all.txt", 1, 91)):
        models, digest = compare_lp.load_models(path)
        assert len(models) == expected and all(model["sha256"] and not model["hash_error"] for model in models)
        sets[stage] = {"models_file": str(path), "models_file_sha256": digest, "repeats": repeats, "models": models}
    save(PLAN, {"schema": 1, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "runner_sha256": sha(Path(__file__)), "sets": sets, "retries": [0],
                "candidate_accuracy": "bounded_newton", "reference_accuracy": "strict",
                "hypothesis": "直接借鉴 HiPO 与 Clarabel 有限精化后保留最佳有限方向的政策；本实现仍衡量真实未正则 J0 和原三块 scaled infinity measure，不复制 HiPO 的 Jreg/omega，也不放宽最终 LP 验算。",
                "source_audit": {"highs_version": "1.15.1", "highs_hipo_built": False,
                                 "clarabel_version": "0.11.1",
                                 "clarabel_source": "https://github.com/oxfordcontrol/Clarabel.rs/blob/v0.11.1/src/solver/core/kktsolvers/direct/quasidef/directldlkktsolver.rs",
                                 "industrial_reference": "https://docs.mosek.com/latest/capi/solving-linear.html"},
                "highs_stage": "开发门槛通过并完成91例自主消融后，才允许同预算直接对照 HiGHS choose/presolve choose/crossover on；仍只属探索。",
                "time_limit": 5, "wall_timeout": 35, "threads": 1, "seed": 0,
                "max_iterations": 200, "scaling_passes": 5,
                "shared_ipm": {"linear-solver": "normal_augmented", "singleton-elimination": "on",
                               "dynamic-regularization": "off", "kkt-equilibration": "on",
                               "fixed-variable-elimination": "on", "singleton-presolve": "off",
                               "presolve-fixed-elimination": "off", "krylov-history-size": "0",
                               "central-corrections": "0", "inexact-prefilter": "off"},
                "scope": "八开发输入三重复及五个已知失败单次探针均用于开发；探针不是新验证集，不按模型名称路由求解。",
                "gate": "只测试 retries=0：开发逐模型达标次数不得下降，且开发罚分数值SGM10下降且墙钟不升，或五例探针至少恢复一例且无新增丢失；否则不得扩大该候选到91例。完整集单次仍属探索，不能宣称独立泛化领先。",
                "failure_policy": "不删除失败，不改预算后重跑替换原结果；保存比较工具退出码及已有全部原始输出。"})
    print(json.dumps({"plan": str(PLAN), "plan_sha256": sha(PLAN), "sets": {k: len(v["models"]) for k, v in sets.items()}}), flush=True)


def frozen_plan():
    plan = read(PLAN)
    assert plan["runner_sha256"] == sha(Path(__file__)), "运行器已偏离冻结计划"
    return plan


def check_solver(version=43):
    executable = ROOT / f".tools/solver-ipm-v{version}.exe"
    assert sha(executable) == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    return executable


def archive(stage, retries):
    name = tag(stage, retries)
    manifest = read(DATA / f"v43-{name}-manifest.json")
    summary = read(DATA / f"v43-{name}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v43-{name}-runs.jsonl.gz").read_bytes()).splitlines()]
    plan = frozen_plan()
    expected = plan["sets"][stage]
    assert manifest["models"] == expected["models"] and manifest["repeats"] == expected["repeats"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key, value in (("time_limit", plan["time_limit"]), ("wall_timeout", plan["wall_timeout"]),
                       ("threads", plan["threads"]), ("seed", plan["seed"]),
                       ("ipm_linear_accuracy", "bounded_newton"), ("reference_ipm_linear_accuracy", "strict"),
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
    recomputed["output_dir"] = str(ROOT / f"results/round-ipm-v43-{name}")
    assert summary == recomputed, "汇总与逐次记录不一致"
    return manifest, summary, rows


def gate(retries):
    _, development, _ = archive("development", retries)
    _, probe, _ = archive("probe", retries)
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
            "plan_sha256": sha(PLAN)}


def benchmark(stage, retries):
    plan = frozen_plan()
    check_solver()
    expected = plan["sets"][stage]
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    if stage == "netlib":
        decision = gate(retries)
        save(DATA / f"v43-retries{retries}-gate.json", decision)
        assert decision["eligible_for_netlib"], "开发门槛未通过，禁止扩大完整集"
    name = tag(stage, retries)
    output = ROOT / f"results/round-ipm-v43-{name}"
    assert not output.exists(), "不能覆盖已测结果"
    assert not list(DATA.glob(f"v43-{name}-*")), "不能覆盖已有归档"
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--models-file", expected["models_file"], "--repeats", str(expected["repeats"]),
               "--time-limit", str(plan["time_limit"]), "--wall-timeout", str(plan["wall_timeout"]),
               "--threads", str(plan["threads"]), "--seed", str(plan["seed"]),
               "--candidate-backend", "ipm", "--reference-backend", "ipm", "--output-dir", str(output),
               "--iteration-limit", "200", "--reference-iteration-limit", "200",
               "--scaling-passes", "5", "--reference-scaling-passes", "5",
               "--ipm-linear-accuracy", "bounded_newton", "--reference-ipm-linear-accuracy", "strict",
               "--ipm-direction-retries", str(retries), "--reference-ipm-direction-retries", str(retries)]
    for key, value in plan["shared_ipm"].items():
        for prefix in ("--ipm-", "--reference-ipm-"):
            command += [prefix + key, value]
    save(DATA / f"v43-{name}-command.json", command)
    with (DATA / f"v43-{name}-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    # 先保存已有产物和退出码，再检查完整性，避免断言丢失失败证据。
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / f"v43-{name}-{filename}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v43-{name}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v43-{name}-process.json", {"returncode": process.returncode, "plan_sha256": sha(PLAN),
                                          "solver_sha256_after": sha(EXE)})
    check_solver()
    assert process.returncode in (0, 1), (name, process.returncode)
    _, summary, _ = archive(stage, retries)
    print(json.dumps({"tag": name, "variants": summary["variants"]}), flush=True)


def highs_benchmark():
    plan = frozen_plan()
    assert gate(0)["eligible_for_netlib"], "开发门槛未通过"
    archive("netlib", 0)
    check_solver()
    expected = plan["sets"]["netlib"]
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    output = ROOT / "results/round-ipm-v43-highs"
    assert not output.exists() and not list(DATA.glob("v43-highs-*")), "不能覆盖已有结果"
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
               "--models-file", expected["models_file"], "--repeats", "1", "--time-limit", "5",
               "--wall-timeout", "35", "--threads", "1", "--seed", "0", "--candidate-backend", "ipm",
               "--reference-backend", "highs", "--reference-highs-solver", "choose",
               "--reference-highs-presolve", "choose", "--reference-highs-crossover", "on",
               "--iteration-limit", "200", "--scaling-passes", "5", "--ipm-direction-retries", "0",
               "--ipm-linear-accuracy", "bounded_newton", "--output-dir", str(output)]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    save(DATA / "v43-highs-command.json", command)
    with (DATA / "v43-highs-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / ("v43-highs-" + filename))
    if (output / "runs.jsonl").exists():
        (DATA / "v43-highs-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / "v43-highs-process.json", {"returncode": process.returncode,
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
    executables = [check_solver(version) for version in (41, 43)]
    models = read(PLAN)["sets"]["development"]["models"]
    records = []
    destination = DATA / "v43-old-mode-trajectory-runs.json.gz"
    assert not destination.exists(), "不能覆盖旧轨迹归档"
    for linear, accuracy, retries, presolve in (("qdldl", "strict", 0, "off"),
            ("normal_augmented", "strict", 0, "off"), ("normal_augmented", "blockwise", 8, "on"),
            ("normal_augmented", "full_newton", 0, "off")):
        for model in models:
            reports, commands, codes, hashes, outputs = [], [], [], [], []
            row = {"model": model["input"], "model_sha256": model["sha256"], "commands": commands,
                   "returncodes": codes, "reports": reports, "outputs": outputs, "hashes": hashes,
                   "new_fields_integer_zero": False, "non_time_equal": False}
            records.append(row)
            for executable in executables:
                command = [str(executable), "solve", model["input"], "--backend", "ipm", "--time-limit", "5",
                           "--threads", "1", "--seed", "0", "--ipm-linear-solver", linear,
                           "--ipm-linear-accuracy", accuracy, "--ipm-direction-retries", str(retries),
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
            for report in normalized:
                for key in ("timings", "backend_version") + FIELDS:
                    report.pop(key, None)
            row["new_fields_integer_zero"] = fields_valid
            row["non_time_equal"] = normalized[0] == normalized[1] and codes[0] == codes[1]
            destination.write_bytes(gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
    for version in (41, 43):
        check_solver(version)
    assert len(records) == 32
    assert all(row["non_time_equal"] and row["new_fields_integer_zero"] for row in records)
    assert all(item["before"] == item["after"] and item["before"]["input"] == row["model_sha256"]
               for row in records for item in row["hashes"])
    print(json.dumps({"old_trajectories_equal": len(records)}), flush=True)


def details():
    groups = {"development": {}, "probe": {}, "coverage": {}}
    for stage in ("development", "probe", "netlib"):
        for retries in (0,):
            name = tag(stage, retries)
            if not (DATA / f"v43-{name}-summary.json").exists():
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
        save(DATA / f"v43-{key}-detail.json", value)
    for retries in (0,):
        if all((DATA / f"v43-{tag(stage, retries)}-summary.json").exists() for stage in ("development", "probe")):
            save(DATA / f"v43-retries{retries}-gate.json", gate(retries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "trajectories", "development", "probe", "netlib", "highs", "details", "gate"))
    parser.add_argument("--retries", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if args.stage == "plan":
        prepare_plan()
    elif args.stage == "trajectories":
        trajectories()
    elif args.stage == "highs":
        highs_benchmark()
    elif args.stage == "details":
        details()
    elif args.stage == "gate":
        print(json.dumps(gate(args.retries), ensure_ascii=False), flush=True)
    else:
        benchmark(args.stage, args.retries)
