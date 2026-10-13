"""按冻结计划串行运行第42轮齐次方法的既有多波前路线消融，保留全部失败并按门槛控制完整集扩展。"""
import argparse
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
EXE = ROOT / ".tools/solver-ipm-v44.exe"
PLAN = DATA / "v45-plan.json"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(stage, retries):
    return f"{stage}-homogeneous-normal_multifrontal-retries{retries}"


def prepare_plan():
    assert not PLAN.exists(), "不能覆盖已经冻结的计划"
    probe = DATA / "v44-known-failures.txt"
    sets = {}
    for stage, path, repeats, expected in (("development", ROOT / "benchmarks/development.txt", 3, 8),
                                          ("probe", probe, 1, 5),
                                          ("netlib", ROOT / "benchmarks/netlib-all.txt", 1, 91)):
        models, digest = compare_lp.load_models(path)
        assert len(models) == expected and all(model["sha256"] and not model["hash_error"] for model in models)
        sets[stage] = {"models_file": str(path), "models_file_sha256": digest, "repeats": repeats, "models": models}
    save(PLAN, {"schema": 1, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "runner_sha256": sha(Path(__file__)), "self_check": self_check(), "sets": sets, "retries": [0],
                "candidate_accuracy": "strict", "reference_accuracy": "strict",
                "candidate_method": "homogeneous", "reference_method": "homogeneous",
                "candidate_linear_solver": "normal_multifrontal", "reference_linear_solver": "normal_augmented",
                "production_identity": "ipm-v44-identity.json",
                "production_identity_sha256": sha(DATA / "ipm-v44-identity.json"),
                "solver": str(EXE), "solver_sha256": sha(EXE), "new_production_build": False,
                "prior_verification": "v44-verification.json",
                "prior_verification_sha256": sha(DATA / "v44-verification.json"),
                "prior_trajectories": "v44-old-mode-trajectory-runs.json.gz",
                "prior_trajectories_sha256": sha(DATA / "v44-old-mode-trajectory-runs.json.gz"),
                "prior_evidence_scope": "90来源、17项CTest和40对旧轨迹均引用v44既有归档，本轮不构建、不重复执行、不记作新测量。",
                "hypothesis": "HSD改变外层轨迹及常数右端工作后，既有分块Schur路线可能改变实际分解成本和可达精度；只消融完整既有路线，保持同一HSD、strict、输入表示及原模型终验。",
                "source_audit": {"highs_version": "1.15.1", "highs_hipo_built": False,
                                 "route_sources": ["src/ipm.cpp", "src/normal_cholesky.cpp", "src/multifrontal_ldl.cpp",
                                                   "third_party/factorhighs/build.cmake", "dependencies.lock.json"],
                                 "scope": "核对v44冻结源码的实际分支和已有v28/v29归档；本轮未新增线性代数实现，也不重述上游完整LP算法为自身能力。"},
                "historical_boundaries": {
                    "v28": "旧Mehrotra下，四组开发中的strict/fixedon、blockwise_weighted/fixedon及blockwise_weighted/fixedoff均新增80bau3b失败；唯一开发全过的strict/fixedoff完整集同配置85到75。不能从较快提前失败声称加速，也不能与fixedon历史最好88直接混比。",
                    "v29": "只给既有分块Schur加对称对角均衡，strict/fixedoff从75到78，恢复4例而丢ship08s，仍13例数值失败。并未检验新HSD外层下的交互，亦未证明分块内核普遍更快。",
                    "current_route": "normal_multifrontal包含已有Schur正对角均衡、FactorHiGHS分块LDL及其主元修改；normal_augmented通常为标量Cholesky。两侧只在相同结构预算异常时回退factorhighs，必须记录实际ipm_factor_backend和回退/主元/精化工作，不将全部差异归因BLAS。",
                    "no_algorithm_change": "复用v44冻结程序、单线程OpenBLAS及已有实现；不新增依赖、默认策略、正则参数、缩放规则或按模型身份分派。",
                    "evidence": {name: sha(DATA / name) for name in (
                        "v28-development-strict-fon-summary.json", "v28-netlib-ipm-summary.json",
                        "v29-netlib-ipm-summary.json", "v29-coverage-detail.json")}},
                "highs_stage": "开发门槛通过并完成91例自主消融后，才允许同预算直接对照 HiGHS choose/presolve choose/crossover on；仍只属探索。",
                "homogeneous_policy": {"centering": "(1-alpha_affine)^3", "corrector_residual_factor": "1-sigma", "affine_cross_product_factor": 1, "fraction_to_boundary": 0.995, "rescale": False, "scalar_floor": False},
                "time_limit": 5, "wall_timeout": 35, "threads": 1, "seed": 0,
                "max_iterations": 200, "scaling_passes": 5,
                "shared_ipm": {"singleton-elimination": "on",
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
    assert sha(DATA / plan["production_identity"]) == plan["production_identity_sha256"], "v44生产锚点身份发生变化"
    return plan


def check_solver():
    identity = read(DATA / "ipm-v45-identity.json")
    assert identity["solver"] == str(EXE)
    assert sha(EXE) == identity["solver_sha256"] == read(DATA / "ipm-v44-identity.json")["solver_sha256"]
    assert identity["plan_sha256"] == sha(PLAN)
    return EXE


def archive(stage, retries):
    name = tag(stage, retries)
    manifest = read(DATA / f"v45-{name}-manifest.json")
    summary = read(DATA / f"v45-{name}-summary.json")
    rows = [json.loads(line) for line in gzip.decompress((DATA / f"v45-{name}-runs.jsonl.gz").read_bytes()).splitlines()]
    plan = frozen_plan()
    expected = plan["sets"][stage]
    assert manifest["models"] == expected["models"] and manifest["repeats"] == expected["repeats"]
    assert len(rows) == len(expected["models"]) * expected["repeats"] * 2
    for key, value in (("time_limit", plan["time_limit"]), ("wall_timeout", plan["wall_timeout"]),
                       ("threads", plan["threads"]), ("seed", plan["seed"]),
                       ("ipm_linear_accuracy", "strict"), ("reference_ipm_linear_accuracy", "strict"),
                       ("ipm_method", "homogeneous"), ("reference_ipm_method", "homogeneous"),
                       ("ipm_linear_solver", "normal_multifrontal"), ("reference_ipm_linear_solver", "normal_augmented"),
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
    recomputed["output_dir"] = str(ROOT / f"results/round-ipm-v45-{name}")
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
        save(DATA / f"v45-retries{retries}-gate.json", decision)
        assert decision["eligible_for_netlib"], "开发门槛未通过，禁止扩大完整集"
    name = tag(stage, retries)
    output = ROOT / f"results/round-ipm-v45-{name}"
    assert not output.exists(), "不能覆盖已测结果"
    assert not list(DATA.glob(f"v45-{name}-*")), "不能覆盖已有归档"
    command = comparison_command(stage, retries, plan, output)
    save(DATA / f"v45-{name}-command.json", command)
    with (DATA / f"v45-{name}-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    # 先保存已有产物和退出码，再检查完整性，避免断言丢失失败证据。
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / f"v45-{name}-{filename}")
    if (output / "runs.jsonl").exists():
        (DATA / f"v45-{name}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / f"v45-{name}-process.json", {"returncode": process.returncode, "plan_sha256": sha(PLAN),
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
    output = ROOT / "results/round-ipm-v45-highs"
    assert not output.exists() and not list(DATA.glob("v45-highs-*")), "不能覆盖已有结果"
    command = highs_command(plan, output)
    save(DATA / "v45-highs-command.json", command)
    with (DATA / "v45-highs-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for filename in ("manifest.json", "summary.json"):
        if (output / filename).exists():
            shutil.copy2(output / filename, DATA / ("v45-highs-" + filename))
    if (output / "runs.jsonl").exists():
        (DATA / "v45-highs-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    save(DATA / "v45-highs-process.json", {"returncode": process.returncode,
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


def details():
    groups = {"development": {}, "probe": {}, "coverage": {}}
    for stage in ("development", "probe", "netlib"):
        for retries in (0,):
            name = tag(stage, retries)
            if not (DATA / f"v45-{name}-summary.json").exists():
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
        save(DATA / f"v45-{key}-detail.json", value)
    for retries in (0,):
        if all((DATA / f"v45-{tag(stage, retries)}-summary.json").exists() for stage in ("development", "probe")):
            save(DATA / f"v45-retries{retries}-gate.json", gate(retries))


def self_check():
    # 仅使用已有计划作命令输入和内存聚合量作门槛输入，不启动进程或写入结果文件。
    plan = read(DATA / "v44-plan.json")
    plan["candidate_linear_solver"] = "normal_multifrontal"
    plan["reference_linear_solver"] = "normal_augmented"
    plan["shared_ipm"].pop("linear-solver")
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
        assert flags["--ipm-linear-solver"] == "normal_multifrontal" and flags["--reference-ipm-linear-solver"] == "normal_augmented"
        assert flags["--iteration-limit"] == flags["--reference-iteration-limit"] == "200"
        assert (flags["--time-limit"], flags["--wall-timeout"], flags["--threads"], flags["--seed"]) == ("5", "35", "1", "0")
        assert all(flags["--ipm-" + key] == flags["--reference-ipm-" + key] == value for key, value in plan["shared_ipm"].items())
    flags = command_flags(highs_command(plan, ROOT / "unused-self-check-output"))
    assert (flags["--reference-backend"], flags["--reference-highs-solver"], flags["--reference-highs-presolve"], flags["--reference-highs-crossover"]) == ("highs", "choose", "choose", "on")
    assert flags["--ipm-linear-solver"] == "normal_multifrontal" and flags["--ipm-method"] == "homogeneous"
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
    parser.add_argument("stage", choices=("plan", "self-check", "development", "probe", "netlib", "highs", "details", "gate"))
    parser.add_argument("--retries", type=int, choices=(0,), default=0)
    args = parser.parse_args()
    if args.stage == "plan":
        prepare_plan()
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
