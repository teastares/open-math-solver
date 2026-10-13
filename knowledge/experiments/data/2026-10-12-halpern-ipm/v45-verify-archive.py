"""独立核验第42轮既有分块路线消融的历史来源、原始质量、门槛与成本审计；不编译或求解。"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import types
import runpy
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
EXPECTED_SOLVER = "58291db80a9dc7faaa17d037060d6cc53fe26db06de3bf3e67107e6401a7c24c"
EXPECTED_PLAN = "6ee7cf6de90a43da905b379461c2e71e98b071db4bea7d5623660ffae5f193ea"
PRIOR_SOLVER = "ef6e30f5fb386f4d77f0dd491dba0c5811ea773ed510e86cad82ee3d9775bc4e"
FIELDS = ("ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves")
SCALARS = ("ipm_homogeneous_tau", "ipm_homogeneous_kappa")
BOUNDED_FIELDS = ("ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns")
PRODUCT_FIELDS = ("ipm_full_newton_preconditioner_solves", "ipm_full_newton_products")
BOOLEAN_FIELDS = {
    "singleton-elimination": "eliminate_singletons", "dynamic-regularization": "dynamic_regularization",
    "kkt-equilibration": "equilibrate_kkt", "fixed-variable-elimination": "eliminate_fixed_variables",
    "singleton-presolve": "presolve_singleton_rows", "presolve-fixed-elimination": "eliminate_presolved_fixed_variables",
    "inexact-prefilter": "inexact_prefilter",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def close(left, right, detail):
    require(math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-13), detail)


def flags(command, start):
    require((len(command) - start) % 2 == 0, "命令选项必须成对")
    result = {}
    for index in range(start, len(command), 2):
        key = command[index]
        require(key.startswith("--") and key not in result, "命令选项重复或缺少名称")
        result[key] = command[index + 1]
    return result


def tag(stage, retries):
    return f"{stage}-homogeneous-normal_multifrontal-retries{retries}"


def verify_sources():
    identity = read(DATA / "ipm-v45-identity.json")
    require(identity["solver_sha256"] == EXPECTED_SOLVER and identity["plan_sha256"] == EXPECTED_PLAN,
            "本轮冻结程序与测前计划身份")
    require(identity["new_production_build"] is False and identity["production_version"] == "ipm-cpu-0.39.0",
            "本轮仅复用既有程序")
    require(identity["production_identity"] == "ipm-v44-identity.json" and
            sha(DATA / identity["production_identity"]) == identity["production_identity_sha256"], "生产身份锚点")
    previous = read(DATA / identity["production_identity"])
    for key in ("files", "solver_sha256", "source_zip_sha256", "compiler", "compiler_sha256", "libraries", "logs"):
        require(identity[key] == previous[key], "复用生产身份字段：" + key)
    require(len(identity["files"]) == 90 and len(identity["libraries"]) == 15, "90来源与15份链接库身份记录")
    archive_path = DATA / identity["source_archive"]
    require(identity["source_archive"] == "ipm-v44-source.zip" and sha(archive_path) == identity["source_zip_sha256"], "历史源码包整体身份")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 90, "历史源码归档数量或重复项")
        require({name: hashlib.sha256(archive.read(name)).hexdigest() for name in names} == identity["files"], "历史源码包逐文件身份")
    require(Path(identity["solver"]) == ROOT / ".tools/solver-ipm-v44.exe" and
            sha(Path(identity["solver"])) == EXPECTED_SOLVER, "复用的归档程序身份")
    for name, digest in identity["logs"].items():
        require(sha(DATA / name) == digest, "既有归档日志：" + name)
    for filename, key in (("v45-plan.json", "plan_sha256"), ("v45-run-experiments.py", "runner_sha256"),
                          ("v45-freeze.py", "freeze_script_sha256")):
        require(sha(DATA / filename) == identity[key], "本轮冻结实验来源：" + filename)
    for name in ("prior_verification", "prior_trajectories"):
        require(sha(DATA / identity[name]) == identity[name + "_sha256"], "既有证据锚点：" + name)
    prior = read(DATA / identity["prior_verification"])
    require(prior["passed"] is True and prior["final"] is True and prior["solver_sha256"] == EXPECTED_SOLVER and
            prior["source_files_verified"] == 90 and prior["ctest_passed"] == 17 and
            prior["old_mode_trajectories_verified"]["count"] == 40, "已完成的v44最终核验")
    # 当前源码、库及构建日志可以进入后续开发；本轮以冻结时证明和历史归档为锚点。
    return identity


def load_frozen_protocol(identity):
    loaded = []
    with zipfile.ZipFile(DATA / identity["source_archive"]) as archive:
        for name in ("benchmark", "compare_lp"):
            path = "scripts/" + name + ".py"
            source = archive.read(path)
            require(hashlib.sha256(source).hexdigest() == identity["files"][path], "冻结Python来源：" + path)
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / path)
            sys.modules[name] = module
            exec(compile(source, identity["source_archive"] + "/" + path, "exec"), module.__dict__)
            loaded.append(module)
    return loaded[1], loaded[0]


def verify_plan(identity, compare):
    plan = read(DATA / "v45-plan.json")
    require(plan["runner_sha256"] == identity["runner_sha256"], "计划运行器身份")
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <=
            datetime.datetime.fromisoformat(identity["created_at_utc"]), "测量前计划与生产冻结时间顺序")
    require(plan["retries"] == [0] and plan["candidate_accuracy"] == "strict" and plan["candidate_method"] == "homogeneous" and
            plan["reference_method"] == "homogeneous" and
            plan["reference_accuracy"] == "strict", "预声明消融维度")
    require((plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
             plan["max_iterations"], plan["scaling_passes"]) == (5, 35, 1, 0, 200, 5), "固定共同预算")
    require(plan["homogeneous_policy"] == {"centering": "(1-alpha_affine)^3", "corrector_residual_factor": "1-sigma",
            "affine_cross_product_factor": 1, "fraction_to_boundary": 0.995, "rescale": False, "scalar_floor": False},
            "冻结齐次数学约定")
    expected_shared = {"singleton-elimination": "on",
                       "dynamic-regularization": "off", "kkt-equilibration": "on",
                       "fixed-variable-elimination": "on", "singleton-presolve": "off",
                       "presolve-fixed-elimination": "off", "krylov-history-size": "0",
                       "central-corrections": "0", "inexact-prefilter": "off"}
    require(plan["shared_ipm"] == expected_shared, "预声明共同算法配置")
    require(set(plan["sets"]) == {"development", "probe", "netlib"}, "输入阶段")
    for stage, count, repeats in (("development", 8, 3), ("probe", 5, 1), ("netlib", 91, 1)):
        entry = plan["sets"][stage]
        models, digest = compare.load_models(Path(entry["models_file"]))
        require(len(models) == count and entry["repeats"] == repeats and entry["models"] == models
                and entry["models_file_sha256"] == digest, stage + " 冻结输入列表及当前输入身份")
        require(all(model["sha256"] and model["hash_error"] is None for model in models), stage + " 输入可读性")
    require(plan["candidate_linear_solver"] == "normal_multifrontal" and
            plan["reference_linear_solver"] == "normal_augmented", "只消融既有完整分解路线")
    require(plan["new_production_build"] is False and plan["solver_sha256"] == EXPECTED_SOLVER and
            plan["production_identity_sha256"] == identity["production_identity_sha256"], "测前复用生产身份")
    for key in ("prior_verification", "prior_trajectories"):
        require(plan[key] == identity[key] and plan[key + "_sha256"] == identity[key + "_sha256"], "计划引用既有证据")
    for name, digest in plan["historical_boundaries"]["evidence"].items():
        require(sha(DATA / name) == digest, "历史失败证据：" + name)
    namespace = runpy.run_path(str(DATA / "v45-run-experiments.py"), run_name="v45_frozen_runner_self_check")
    require(namespace["self_check"]() == plan["self_check"] ==
            {"passed": True, "command_configurations": 4, "gate_scenarios": 7, "checks": 11, "solver_runs": 0},
            "仅内存重跑命令/门槛自检")
    return plan


def verify_counts(report, context):
    counters = (*FIELDS, *PRODUCT_FIELDS, *BOUNDED_FIELDS, "ipm_weighted_krylov_steps", "ipm_refinement_steps", "ipm_iterations")
    if report["backend"] != "ipm":
        require(all(key in report and report[key] is None for key in (*counters, *SCALARS)), context + " 非IPM字段必须为null")
        return
    require(all(type(report.get(key)) is int and 0 <= report[key] < 2**63 for key in counters), context + " 非负int64计数")
    noncounters = {"ipm_factor_backend", "ipm_primal_regularization", "ipm_dual_regularization", *SCALARS}
    require(all(type(value) is int and 0 <= value < 2**63 for key, value in report.items()
                if key.startswith("ipm_") and key not in noncounters), context + " 全部IPM工作计数必须为非负整数")
    ipm = report["options"]["ipm"]
    constant, directions = (report[key] for key in FIELDS)
    iterations = report["ipm_iterations"]
    require(all(key in report for key in SCALARS), context + " 齐次标量字段必须存在")
    if ipm["method"] == "mehrotra":
        require(constant == directions == 0 and all(report[key] is None for key in SCALARS), context + " 普通方法无齐次工作或标量")
    else:
        require(ipm["method"] == "homogeneous" and ipm["linear_accuracy"] == "strict" and
                all(ipm[key] == 0 for key in ("central_corrections", "direction_retries", "krylov_history_size")) and
                ipm["inexact_prefilter"] is False, context + " 齐次方法合法配置")
        require(iterations <= constant <= iterations + 1 and 2 * iterations <= directions <= 2 * constant,
                context + " 齐次实际启动工作与完整外层步数关系")
        if all(report[key] is None for key in SCALARS):
            require(constant == directions == iterations == 0, context + " 未初始化齐次标量不得出现工作")
        else:
            require(all(type(report[key]) in (int, float) and math.isfinite(report[key]) and report[key] > 0 for key in SCALARS),
                    context + " 最近有效齐次点的标量必须同时有限严格正")
    accuracy = ipm["linear_accuracy"]
    returns, inexact = (report[key] for key in BOUNDED_FIELDS)
    require(inexact <= returns, context + " 非精确返回是实际方向返回子集")
    if accuracy != "bounded_newton":
        require(returns == inexact == 0, context + " 非有界模式返回计数为零")
    else:
        require(report["ipm_weighted_krylov_steps"] == 0 and ipm["direction_retries"] == 0,
                context + " 有界模式不得启动Krylov或方向重试")
        require(2 * iterations <= returns <= 2 * iterations + 2, context + " 有界返回与完整步数")
        require(returns <= report[PRODUCT_FIELDS[0]] <= 4 * (returns + 1) and
                returns <= report[PRODUCT_FIELDS[1]] <= report[PRODUCT_FIELDS[0]], context + " 有界工作范围")
    if accuracy not in ("bounded_newton", "full_newton"):
        require(all(report[key] == 0 for key in PRODUCT_FIELDS), context + " 旧KKT模式无完整J工作")
    if accuracy == "full_newton":
        require(report["ipm_weighted_krylov_steps"] <= report[PRODUCT_FIELDS[0]], context + " Arnoldi预条件工作")
    require(report["ipm_weighted_krylov_steps"] <= report["ipm_refinement_steps"], context + " 改进与Arnoldi计数")
    require(all(report["ipm_krylov_recycle_" + key] == 0 for key in
                ("projections", "acceptances", "vectors", "peak_dimension")), context + " 历史关闭计数")


def verify_group(manifest_path, identity, plan, compare, benchmark, input_hashes):
    name = manifest_path.name[len("v45-"):-len("-manifest.json")]
    choices = {tag(stage, retry): (stage, retry) for stage in plan["sets"] for retry in plan["retries"]}
    choices["highs"] = ("highs", 0)
    require(name in choices, "非计划实验组：" + name)
    stage, retries = choices[name]
    declared = plan["sets"]["netlib" if stage == "highs" else stage]
    original = ROOT / "results" / ("round-ipm-v45-" + name)
    summary_path, raw_path = (DATA / ("v45-" + name + suffix) for suffix in ("-summary.json", "-runs.jsonl.gz"))
    manifest, summary = read(manifest_path), read(summary_path)
    require(datetime.datetime.fromisoformat(identity["created_at_utc"]) <=
            datetime.datetime.fromisoformat(manifest["started_at_utc"]), name + " 实验开始早于测前冻结")
    raw = gzip.decompress(raw_path.read_bytes())
    require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), name + " 清单字节")
    require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), name + " 汇总字节")
    require(raw == (original / "runs.jsonl").read_bytes(), name + " 原始记录字节")
    require(read(DATA / ("v45-" + name + "-run.log")) == summary, name + " 比较进程原始输出")
    process = read(DATA / ("v45-" + name + "-process.json"))
    require(process["plan_sha256"] == identity["plan_sha256"] and
            process["solver_sha256_after"] == identity["solver_sha256"], name + " 比较进程冻结身份")
    rows = [benchmark.strict_json(line) for line in raw.splitlines()]
    require(manifest["models"] == declared["models"] and manifest["repeats"] == declared["repeats"] and
            manifest["models_file"] == declared["models_file"] and
            manifest["models_file_sha256"] == declared["models_file_sha256"], name + " 预声明输入及重复")
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        require(manifest[key] == plan[key], name + " 共同资源：" + key)
    expected = identity["solver_sha256"]
    require(manifest["solver_sha256"] == expected and manifest["solver_hash_error"] is None, name + " 程序身份")
    require(manifest["comparison_script_sha256"] == identity["files"]["scripts/compare_lp.py"] and
            manifest["benchmark_script_sha256"] == identity["files"]["scripts/benchmark.py"], name + " 协议身份")
    variants = (manifest["reference_variant"], manifest["candidate_variant"])
    configs = manifest["variants"]
    require(len(set(variants)) == 2 and set(configs) == set(variants), name + " 独立双侧标签")
    command_path = DATA / ("v45-" + name + "-command.json")
    command = read(command_path)
    top = flags(command, 2)
    require(Path(command[1]) == ROOT / "scripts/compare_lp.py" and
            Path(top["--solver"]) == ROOT / ".tools/solver-ipm-v44.exe" and
            Path(top["--output-dir"]) == original and top["--models-file"] == declared["models_file"], name + " 比较入口")
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        require(float(top["--" + key.replace("_", "-")]) == plan[key], name + " 比较资源")
    require(int(top["--repeats"]) == declared["repeats"], name + " 比较重复")
    for side, variant in enumerate(variants):
        prefix = "--reference-" if side == 0 else "--"
        config = configs[variant]
        require(manifest["solvers"][variant]["sha256"] == expected and
                manifest["solvers"][variant]["hash_error"] is None, name + " 两侧程序身份")
        if stage == "highs" and side == 0:
            require(config["backend"] == top["--reference-backend"] == "highs", name + " HiGHS参考后端")
            for key, value in (("solver", "choose"), ("presolve", "choose"), ("crossover", "on")):
                require(config["highs_" + key] == top["--reference-highs-" + key] == value,
                        name + " HiGHS参考配置（清单保存在variants中）")
            require(config["iteration_limit"] is None and config["scaling_passes"] is None and
                    all(value is None for key, value in config.items() if key.startswith("ipm_")), name + " HiGHS无IPM专属参数")
            continue
        accuracy = "strict"
        method = "homogeneous"
        require(config["backend"] == "ipm" and config["ipm_linear_accuracy"] == accuracy and config["ipm_method"] == method and
                config["ipm_direction_retries"] == retries, name + " 算法消融配置")
        require(config["iteration_limit"] == 200 and config["scaling_passes"] == 5, name + " 配置迭代与缩放预算")
        require(manifest["solvers"][variant]["sha256"] == expected and
                manifest["solvers"][variant]["hash_error"] is None, name + " 两侧程序身份")
        require(top["--candidate-backend" if side else "--reference-backend"] == "ipm", name + " 比较后端")
        require(top[prefix + "iteration-limit"] == "200" and top[prefix + "scaling-passes"] == "5", name + " 迭代与缩放预算")
        for key, value in plan["shared_ipm"].items():
            attribute = "ipm_" + key.replace("-", "_")
            requested = int(value) if key in ("krylov-history-size", "central-corrections") else value
            require(config[attribute] == requested and top[prefix + "ipm-" + key] == value, name + " 共同配置：" + key)
            require(manifest[("reference_" if side == 0 else "") + attribute] == requested, name + " 清单共同配置：" + key)
        require(config["ipm_linear_solver"] == plan["candidate_linear_solver" if side else "reference_linear_solver"],
                name + " 预声明分解路线")
        for key in ("method", "linear_accuracy", "direction_retries", "linear_solver"):
            require(top[prefix + "ipm-" + key.replace("_", "-")] == str(config["ipm_" + key]), name + " 顶层精度配置")
            require(manifest[("reference_" if side == 0 else "") + "ipm_" + key] == config["ipm_" + key], name + " 清单精度配置")
    if stage != "highs":
        require({key for key in configs[variants[0]] if configs[variants[0]][key] != configs[variants[1]][key]} ==
                {"ipm_linear_solver"}, name + " 存在非预声明配置差异")
    schedule = list(compare.schedule(manifest["models"], manifest["repeats"], variants))
    require(len(rows) == len(schedule) == summary["run_count"], name + " 原始记录完整性")
    for index, (row, scheduled) in enumerate(zip(rows, schedule)):
        model, repetition, position, variant = scheduled
        require((row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"]) ==
                (model["model_index"], repetition, position, variant), name + " 交替顺序或重复")
        require(row["run_index"] == index and row["seed"] == 0 and row["input"] == model["input"], name + " 原始索引")
        if model["input"] not in input_hashes:
            input_hashes[model["input"]] = sha(Path(model["input"]))
        require(input_hashes[model["input"]] == model["sha256"] and
                all(row[key] == model["sha256"] for key in ("model_sha256", "frozen_model_sha256", "model_sha256_after")),
                name + " 输入前后身份")
        require(all(row[key] == expected for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")),
                name + " 程序前后身份")
        solve_command = row["command"]
        require(Path(solve_command[0]) == ROOT / ".tools/solver-ipm-v44.exe" and
                solve_command[1:3] == ["solve", row["input"]], name + " 求解入口")
        actual = flags(solve_command, 3)
        config = configs[variant]
        require(actual["--backend"] == config["backend"] and float(actual["--time-limit"]) == 5 and
                actual["--threads"] == "1" and actual["--seed"] == "0", name + " 求解共同预算")
        if config["backend"] == "ipm":
            require(actual["--iteration-limit"] == "200" and actual["--scaling-passes"] == "5", name + " IPM迭代预算")
            for key in (*plan["shared_ipm"], "method", "linear-accuracy", "direction-retries", "linear-solver"):
                require(actual["--ipm-" + key] == str(config["ipm_" + key.replace("-", "_")]), name + " 求解配置：" + key)
        else:
            for key in ("solver", "presolve", "crossover"):
                require(actual["--highs-" + key] == config["highs_" + key], name + " HiGHS实际命令")
            require(not any(key.startswith("--ipm-") for key in actual), name + " HiGHS不带自主参数")
        report = row["report"]
        if report is not None:
            require(benchmark.strict_json(row["stdout"]) == report, name + " stdout与报告")
            require(report["backend"] == config["backend"], name + " 报告后端")
            require(report["backend_version"] == ("ipm-cpu-0.39.0" if config["backend"] == "ipm" else "1.15.1"), name + " 报告版本")
            require(row["status"] == report["status"] and benchmark.classify(row["returncode"], report) == row["classification"], name + " 分类与退出码")
            options = report["options"]
            require((options["time_limit"], options["threads"], options["seed"]) == (5, 1, 0), name + " 报告共同预算")
            if config["backend"] == "ipm":
                ipm = options["ipm"]
                require((ipm["max_iterations"], ipm["scaling_passes"]) == (200, 5), name + " 报告IPM预算")
                for key, field in BOOLEAN_FIELDS.items():
                    require(ipm[field] is (plan["shared_ipm"][key] == "on"), name + " 报告布尔配置")
                for key in ("method", "linear_solver", "linear_accuracy", "direction_retries", "krylov_history_size", "central_corrections"):
                    require(ipm[key] == config["ipm_" + key], name + " 报告精度配置")
                require(report["ipm_presolved_singleton_rows"] == report["ipm_presolved_fixed_columns"] == 0,
                        name + " 本轮输入归并关闭，相关实际计数必须为零")
            else:
                require(options["ipm"] is None and options["highs"] ==
                        {"solver": "choose", "presolve": "choose", "run_crossover": "on"}, name + " HiGHS实际报告配置")
            verify_counts(report, name)
            if config["backend"] == "ipm":
                requested_backend = "normal_multifrontal" if config["ipm_linear_solver"] == "normal_multifrontal" else "normal"
                require(report["ipm_factor_backend"] in (requested_backend, "factorhighs"), name + " 实际分解路线")
                require((report["ipm_structural_fallbacks"] > 0) is (report["ipm_factor_backend"] == "factorhighs"), name + " 实际结构回退账本")
                for key in ("ipm_structural_fallbacks", "ipm_factorizations", "ipm_regularized_pivots"):
                    require(type(report.get(key)) is int and 0 <= report[key] < 2**63, name + " 实际分解非负计数")
        else:
            require(row["classification"] in ("wall_timeout", "invalid_json", "process_error", "launch_error"), name + " 缺报告失败分类")
        numerical = compare.numerical_seconds(row)
        require(row["numerical_seconds"] == numerical, name + " 完整数值耗时")
        require(math.isfinite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] > 0, name + " 墙钟耗时")
        failure = compare.quality_failure(row, model, model["sha256"], expected, expected)
        if failure is None and numerical is None:
            failure = "invalid_timings"
        if failure is None and row["elapsed_wall_seconds"] > 35:
            failure = "wall_budget_exceeded"
        if failure is None and numerical > 5:
            failure = "numerical_budget_exceeded"
        require(row["quality_pass"] is (failure is None) and row["quality_failure"] == failure, name + " 共同质量")
        require(row["wall_budget_exceeded"] is (row["elapsed_wall_seconds"] > 35) and
                row["numerical_budget_exceeded"] is (numerical > 5 if numerical is not None else None), name + " 超时标志")
        close(row["quality_penalized_wall_seconds"], min(row["elapsed_wall_seconds"], 35) if failure is None else 35, name + " 墙钟罚分")
        close(row["quality_penalized_numerical_seconds"], min(numerical, 5) if failure is None else 5, name + " 数值罚分")
        close(row["penalized_capped_seconds"], min(row["elapsed_wall_seconds"], 5) if row["classification"] == "resolved" else 5, name + " 原始罚分")
    recomputed = compare.summarize(rows, manifest["models"], manifest["repeats"], 35, 5, variants)
    recomputed.update(configurations=configs, output_dir=str(original))
    require(recomputed == summary, name + " 完整汇总重算")
    require(process["returncode"] == (0 if summary["quality_pass_count"] == len(rows) else 1), name + " 真实比较退出码")
    return {"tag": name, "stage": stage, "retries": retries, "raw_records": len(rows),
            "quality_pass_count": summary["quality_pass_count"], "process_returncode": process["returncode"],
            "manifest_sha256": sha(manifest_path), "summary_sha256": sha(summary_path), "raw_sha256": sha(raw_path),
            "command_sha256": sha(command_path), "process_sha256": sha(DATA / ("v45-" + name + "-process.json")),
            "missing_reports": sum(row["report"] is None for row in rows),
            "actual_factor_backends": {side: dict(collections.Counter((row["report"] or {}).get("ipm_factor_backend")
                for row in rows if row["requested_variant"] == side)) for side in variants},
            "homogeneous_work": {side: {key: sum((row["report"] or {}).get(key) or 0 for row in rows if row["requested_variant"] == side)
                                             for key in (*FIELDS, *PRODUCT_FIELDS)} for side in variants},
            "actual_homogeneous_direction_solves": sum((row["report"] or {}).get(FIELDS[1], 0) or 0 for row in rows),
            "actual_homogeneous_work_reports": sum(((row["report"] or {}).get(FIELDS[1]) or 0) > 0 for row in rows)}, (manifest, summary, rows)


def verify_trajectories(benchmark):
    path = DATA / "v44-old-mode-trajectory-runs.json.gz"
    items = json.loads(gzip.decompress(path.read_bytes()))
    require(len(items) == 40, "40组旧轨迹")
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (43, 44)}
    require(identities[43]["solver_sha256"] == PRIOR_SOLVER and identities[44]["solver_sha256"] == EXPECTED_SOLVER,
            "轨迹旧新程序锁定身份")
    expected_models = {model["input"] for model in read(DATA / "v44-plan.json")["sets"]["development"]["models"]}
    expected_configs = {("qdldl", "strict", 0, False), ("normal_augmented", "strict", 0, False),
                        ("normal_augmented", "blockwise", 8, True), ("normal_augmented", "full_newton", 0, False),
                        ("normal_augmented", "bounded_newton", 0, False)}
    seen = set()
    for item in items:
        require(item["model"] in expected_models and sha(Path(item["model"])) == item["model_sha256"], "轨迹输入身份")
        require(all(len(item[key]) == 2 for key in ("reports", "commands", "returncodes", "hashes", "outputs")), "轨迹完整成对记录")
        normalized = copy.deepcopy(item["reports"])
        for index, version in enumerate((43, 44)):
            report = item["reports"][index]
            require(benchmark.strict_json(item["outputs"][index]["stdout"]) == report, "轨迹原始stdout")
            require(report["backend"] == "ipm" and report["backend_version"] ==
                    ("ipm-cpu-0.38.0" if version == 43 else "ipm-cpu-0.39.0"), "轨迹版本")
            expected_hashes = {"solver": identities[version]["solver_sha256"], "input": item["model_sha256"]}
            require(item["hashes"][index] == {"before": expected_hashes, "after": expected_hashes}, "轨迹程序和输入前后hash")
            command = item["commands"][index]
            require(Path(command[0]) == ROOT / f".tools/solver-ipm-v{version}.exe" and command[1:3] == ["solve", item["model"]], "轨迹入口")
            actual = flags(command, 3)
            require(actual["--backend"] == "ipm" and actual["--time-limit"] == "5" and actual["--threads"] == "1" and
                    actual["--seed"] == "0" and actual["--ipm-krylov-history-size"] == "0", "轨迹共同预算")
            ipm = report["options"]["ipm"]
            require((report["options"]["time_limit"], report["options"]["threads"], report["options"]["seed"],
                     ipm["max_iterations"], ipm["scaling_passes"], ipm["krylov_history_size"]) == (5, 1, 0, 200, 5, 0), "轨迹报告预算")
            for field in ("linear_solver", "linear_accuracy", "direction_retries"):
                require(actual["--ipm-" + field.replace("_", "-")] == str(ipm[field]), "轨迹精度配置")
            for flag in ("singleton-elimination", "fixed-variable-elimination", "singleton-presolve", "presolve-fixed-elimination", "kkt-equilibration"):
                require(ipm[BOOLEAN_FIELDS[flag]] is (actual["--ipm-" + flag] == "on"), "轨迹布尔配置")
            require(benchmark.classify(item["returncodes"][index], report) in ("resolved", "incomplete"), "轨迹退出码")
            if version == 44:
                verify_counts(report, "旧轨迹")
                require(all(type(report.get(key)) is int and report[key] == 0 for key in FIELDS), "新计数必须为整数零")
                require(all(key in report and report[key] is None for key in SCALARS), "普通方法齐次标量必须明确null")
                require(ipm["method"] == "mehrotra", "新增method字段默认值")
            else:
                require(all(key not in report for key in (*FIELDS, *SCALARS)) and "method" not in ipm, "旧版本不应出现新增字段")
            normalized[index]["options"]["ipm"].pop("method", None)
            for key in ("timings", "backend_version", *FIELDS, *SCALARS):
                normalized[index].pop(key, None)
        ipm = item["reports"][1]["options"]["ipm"]
        config = (ipm["linear_solver"], ipm["linear_accuracy"], ipm["direction_retries"], ipm["presolve_singleton_rows"])
        require(config in expected_configs and (item["model"], config) not in seen, "轨迹配置或重复项")
        require(ipm["eliminate_presolved_fixed_variables"] is config[3] and ipm["equilibrate_kkt"] is True and
                ipm["eliminate_singletons"] is (config[0] != "qdldl") and
                ipm["eliminate_fixed_variables"] is (config[0] != "qdldl"), "轨迹消元组合")
        seen.add((item["model"], config))
        require(normalized[0] == normalized[1] and item["returncodes"][0] == item["returncodes"][1] and
                item["non_time_equal"] is True and item["new_fields_integer_zero"] is True, "旧轨迹非时间内容或退出码差异")
    require(seen == {(model, config) for model in expected_models for config in expected_configs}, "40轨迹覆盖")
    return {"new_trajectory_runs": 0, "scope": "引用并重核v44既有40对轨迹，不是本轮重新运行。", "count": len(items), "sha256": sha(path), "two_new_counts_integer_zero": True, "two_new_scalars_null": True, "new_method_default_mehrotra": True,
            "program_and_input_pre_post_hashes_verified": True,
            "exit_code_pairs": dict(collections.Counter(str(item["returncodes"]) for item in items))}


def verify_derived(groups, plan_hash, final):
    checked, seen, gates = [], set(), []
    for stage in ("development", "probe", "coverage"):
        path = DATA / f"v45-{stage}-detail.json"
        if not path.exists() and not final:
            continue
        details = read(path)
        for name, item in details.items():
            require(name in groups and name not in seen and name.startswith("netlib" if stage == "coverage" else stage), "详情组名")
            seen.add(name)
            manifest, summary, rows = groups[name]
            expected = []
            for row in rows:
                report = row["report"] or {}
                expected.append({"model": Path(row["input"]).stem, "candidate": row["requested_variant"] == manifest["candidate_variant"],
                                 "repetition": row["repetition"], "quality_pass": row["quality_pass"], "quality_failure": row["quality_failure"],
                                 "status": row["status"], "message": report.get("message"), "numerical_seconds": row["numerical_seconds"],
                                 "work": {key: value for key, value in report.items() if key.startswith("ipm_")},
                                 "optimality": report.get("lp_optimality")})
            require(item["models"] == expected and item["summary"] == summary["variants"], "派生逐次详情")
            passed = {side: {Path(model["input"]).stem for model in summary["models"] if model["quality_pass_counts"][side] == manifest["repeats"]}
                      for side in (manifest["reference_variant"], manifest["candidate_variant"])}
            before, after = passed[manifest["reference_variant"]], passed[manifest["candidate_variant"]]
            require(item["gain"] == sorted(after - before) and item["lost"] == sorted(before - after), "全部重复通过口径得失")
        checked.append({"path": path.name, "sha256": sha(path), "groups": len(details)})
    require(not final or seen == set(groups) - {"highs"}, "派生详情缺组")
    for retries in (0,):
        needed = [tag(stage, retries) for stage in ("development", "probe")]
        if any(name not in groups for name in needed):
            require(not final, "缺少开发或已知失败探针")
            continue
        development, probe = (groups[name][1] for name in needed)
        reference, candidate = development["reference_variant"], development["candidate_variant"]
        lost = [Path(model["input"]).stem for model in development["models"] if model["quality_pass_counts"][candidate] < model["quality_pass_counts"][reference]]
        before, after = development["variants"][reference], development["variants"][candidate]
        faster = after["penalized_numerical_sgm10_seconds"] < before["penalized_numerical_sgm10_seconds"] and after["penalized_wall_sgm10_seconds"] <= before["penalized_wall_sgm10_seconds"]
        reference, candidate = probe["reference_variant"], probe["candidate_variant"]
        gain = [Path(model["input"]).stem for model in probe["models"] if model["quality_pass_counts"][candidate] > model["quality_pass_counts"][reference]]
        probe_lost = [Path(model["input"]).stem for model in probe["models"] if model["quality_pass_counts"][candidate] < model["quality_pass_counts"][reference]]
        gate = {"retries": retries, "development_lost": lost, "development_time_improved": faster,
                "probe_gain": gain, "probe_lost": probe_lost, "eligible_for_netlib": not lost and (faster or (bool(gain) and not probe_lost)),
                "plan_sha256": plan_hash}
        path = DATA / f"v45-retries{retries}-gate.json"
        if path.exists() or final:
            require(read(path) == gate, "扩展门槛独立重算")
        require(tag("netlib", retries) not in groups or gate["eligible_for_netlib"], "未经预声明门槛扩大完整集")
        require("highs" not in groups or (gate["eligible_for_netlib"] and tag("netlib", 0) in groups),
                "HiGHS直接比较必须通过门槛并完成自主完整集")
        gates.append(gate)
    return checked, gates


def verify_documents(output):
    documents = list(ROOT.glob("*.md"))
    for directory in ("knowledge", "changelog", "benchmarks"):
        documents.extend((ROOT / directory).rglob("*.md"))
    links, broken, self_links = 0, [], 0
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            links += 1
            local = (path.parent / unquote(target.split("#")[0].strip("<>"))).resolve()
            self_links += local == output.resolve()
            if not local.exists():
                broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    require(not broken and self_links >= 2, "文档断链或实验/changelog缺少最终报告链接：" + str(broken))
    return {"markdown_files_checked": len(documents), "local_links_checked": links,
            "verification_report_links": self_links, "broken_links": broken}


def verify_counter_guards(groups):
    reports = [row["report"] for _, _, rows in groups.values() for row in rows
               if ((row.get("report") or {}).get(FIELDS[1]) or 0) > 0]
    if not reports:
        return {"checked": False, "reason": "尚无实际齐次方向报告"}
    report = reports[0]
    verify_counts(report, "计数校验正例")
    cases = [(FIELDS[0], None), (FIELDS[0], True), (FIELDS[0], -1), (FIELDS[0], 2**63),
             (FIELDS[1], 2 * report[FIELDS[0]] + 1), (SCALARS[0], None), (SCALARS[0], False),
             (SCALARS[0], 0), (SCALARS[1], -1), (SCALARS[1], float("inf"))]
    rejected = 0
    for key, value in cases:
        candidate = copy.deepcopy(report)
        candidate[key] = value
        try:
            verify_counts(candidate, "齐次统计纯内存反例")
        except AssertionError:
            rejected += 1
    require(rejected == len(cases), "齐次计数及标量校验必须拒绝全部反例")
    return {"checked": True, "valid_report_passed": True, "invalid_counter_or_scalar_cases_rejected": rejected,
            "scope": "只在内存复制既有报告进行反例验证，不修改归档或启动求解。"}


def audit_prior_common_cost(compare, benchmark, identity):
    prior = read(DATA / identity["prior_verification"])
    evidence = next(group for group in prior["archive_groups_verified"] if group["tag"] == "highs")
    paths = {kind: DATA / ("v44-highs-" + suffix) for kind, suffix in
             (("manifest", "manifest.json"), ("summary", "summary.json"), ("raw", "runs.jsonl.gz"))}
    for kind, path in paths.items():
        require(sha(path) == evidence[kind + "_sha256"], "v44成本审计的既有已核归档：" + kind)
    manifest, summary = read(paths["manifest"]), read(paths["summary"])
    rows = [benchmark.strict_json(line) for line in gzip.decompress(paths["raw"].read_bytes()).splitlines()]
    require(len(rows) == 182, "既有HiGHS比较182记录")
    variants = (manifest["reference_variant"], manifest["candidate_variant"])
    models = {model["input"]: model for model in manifest["models"]}
    pairs = {}
    for row in rows:
        model = models[row["input"]]
        require(benchmark.strict_json(row["stdout"]) == row["report"], "既有成本报告与stdout一致")
        require(row["numerical_seconds"] == compare.numerical_seconds(row), "既有完整数值成本")
        failure = compare.quality_failure(row, model, model["sha256"], EXPECTED_SOLVER, EXPECTED_SOLVER)
        if failure is None and row["numerical_seconds"] is None:
            failure = "invalid_timings"
        if failure is None and row["elapsed_wall_seconds"] > 35:
            failure = "wall_budget_exceeded"
        if failure is None and row["numerical_seconds"] > 5:
            failure = "numerical_budget_exceeded"
        require(row["quality_pass"] is (failure is None) and row["quality_failure"] == failure, "既有共同质量重核")
        group = pairs.setdefault(row["input"], {})
        require(row["requested_variant"] not in group, "既有成本配对唯一")
        group[row["requested_variant"]] = row
    recomputed = compare.summarize(rows, manifest["models"], 1, 35, 5, variants)
    recomputed.update(configurations=manifest["variants"], output_dir=summary["output_dir"])
    require(recomputed == summary, "既有HiGHS完整摘要重建")
    require(len(pairs) == 91 and all(set(pair) == set(variants) for pair in pairs.values()), "共同成本配对覆盖")
    common = {path: pair for path, pair in pairs.items() if all(row["quality_pass"] for row in pair.values())}
    require(len(common) == 89, "共同通过集必须89例，不能替代91例罚分成绩")
    def sgm(values):
        return math.exp(math.fsum(math.log(10 + value) for value in values) / len(values)) - 10
    def costs(selected):
        return {side: {"numerical_sgm10_seconds": sgm([pair[variant]["numerical_seconds"] for pair in selected]),
                       "wall_sgm10_seconds": sgm([pair[variant]["elapsed_wall_seconds"] for pair in selected])}
                for side, variant in zip(("highs", "homogeneous"), variants)}
    buckets = []
    bounds = ((0, 1000), (1000, 10000), (10000, 100000), (100000, None))
    expected = ((13, .000962, .002322, .010018, .011593), (55, .007098, .046608, .019069, .059175),
                (19, .169783, .442787, .194138, .468775), (2, .189195, 1.192380, .251916, 1.257844))
    for (lower, upper), check in zip(bounds, expected):
        selected = [pair for pair in common.values() if lower <= pair[variants[1]]["report"]["model"]["nonzeros"]
                    and (upper is None or pair[variants[1]]["report"]["model"]["nonzeros"] < upper)]
        item = {"nonzeros_min": lower, "nonzeros_max_exclusive": upper, "models": len(selected), **costs(selected)}
        actual = (item["models"], *(round(item[side][kind + "_sgm10_seconds"], 6)
                  for kind in ("numerical", "wall") for side in ("highs", "homogeneous")))
        require(actual == check, "正文成本分桶与独立重算一致")
        buckets.append(item)
    total = costs(list(common.values()))
    require(tuple(round(total[side][kind + "_sgm10_seconds"], 6)
                  for kind in ("numerical", "wall") for side in ("highs", "homogeneous")) ==
            (.044779, .147967, .060054, .163908), "正文共同总体SGM")
    numerical_sum = math.fsum(pair[variants[1]]["numerical_seconds"] for pair in common.values())
    verification_sum = math.fsum(pair[variants[1]]["report"]["timings"]["verification_seconds"] for pair in common.values())
    differences = {Path(path).stem: pair[variants[1]]["numerical_seconds"] - pair[variants[0]]["numerical_seconds"]
                   for path, pair in common.items()}
    net_difference = math.fsum(differences.values())
    selected_difference = differences["pilot87"] + differences["maros-r7"]
    fraction = selected_difference / net_difference
    close(numerical_sum, 14.2164735, "自主数值累计")
    close(verification_sum, .4781067, "已记录原模型验算累计")
    require(round(100 * verification_sum / numerical_sum, 2) == 3.36 and round(100 * fraction, 2) == 48.73,
            "正文验算占比及两模型净差占比")
    backend_counts = dict(collections.Counter(pair[variants[1]]["report"]["ipm_factor_backend"] for pair in common.values()))
    require(backend_counts.get("normal") == 88, "共同集88例实际normal路线")
    fit = next(pair[variants[1]]["report"] for path, pair in common.items() if Path(path).stem == "fit1p")
    require(fit["ipm_kkt_dimension"] == 627 and fit["ipm_factor_nonzeros"] == 627 * 628 // 2, "fit1p满三角存储")
    return {"source": paths["raw"].name, "raw_sha256": sha(paths["raw"]), "prior_records_recomputed": 182,
            "new_solver_runs": 0, "common_successes": 89, "excluded_failed_models": sorted(Path(path).stem for path in pairs if path not in common),
            "nonzero_buckets": buckets, "common_total": total, "homogeneous_numerical_sum_seconds": numerical_sum,
            "recorded_verification_sum_seconds": verification_sum, "recorded_verification_fraction": verification_sum / numerical_sum,
            "net_numerical_difference_seconds": net_difference, "pilot87_and_maros_r7_difference_seconds": selected_difference,
            "pilot87_and_maros_r7_net_difference_fraction": fraction,
            "absolute_difference_fraction_for_distinction": selected_difference / math.fsum(abs(value) for value in differences.values()),
            "actual_homogeneous_factor_backends": backend_counts,
            "fit1p_factor_dimension": fit["ipm_kkt_dimension"], "fit1p_factor_nonzeros": fit["ipm_factor_nonzeros"],
            "limitations": "事后只读成本审计；累计差按自主减HiGHS的有符号净差。原坐标提取不在verification计时内，未细分形成/分解/回代；共同集不能替代91例总成绩。"}


def failure_details(groups):
    manifest, _, rows = groups[tag("probe", 0)]
    fields = ("ipm_iterations", "ipm_factorizations", "ipm_refinement_steps", "ipm_regularized_pivots",
              "ipm_factor_backend", "ipm_structural_fallbacks", *FIELDS, *SCALARS)
    return [{"model": Path(row["input"]).stem,
             "side": "candidate" if row["requested_variant"] == manifest["candidate_variant"] else "reference",
             "quality_failure": row["quality_failure"], "status": row["status"],
             "numerical_seconds": row["numerical_seconds"], "message": row["report"]["message"],
             "work": {key: row["report"][key] for key in fields}}
            for row in rows if not row["quality_pass"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="预检使用独立文件，不要求正文的最终链接完成")
    args = parser.parse_args()
    output = DATA / ("v45-verification-preflight.json" if args.preflight else "v45-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False,
              "final": not args.preflight, "verification_script_sha256": sha(Path(__file__))}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        identity = verify_sources()
        compare, benchmark = load_frozen_protocol(identity)
        plan = verify_plan(identity, compare)
        groups, metrics, inputs = {}, [], {}
        for path in sorted(DATA.glob("v45-*-manifest.json")):
            metric, group = verify_group(path, identity, plan, compare, benchmark, inputs)
            metrics.append(metric)
            groups[metric["tag"]] = group
        commands = {path.name[len("v45-"):-len("-command.json")] for path in DATA.glob("v45-*-command.json")}
        directories = {path.name[len("round-ipm-v45-"):] for path in (ROOT / "results").glob("round-ipm-v45-*") if path.is_dir()}
        require(commands == directories == set(groups), "命令、目录及归档完整组必须一致")
        require({tag(stage, 0) for stage in ("development", "probe")} <= set(groups), "开发与探针均须完整")
        trajectories = verify_trajectories(benchmark)
        details, gates = verify_derived(groups, identity["plan_sha256"], True)
        require(len(gates) == 1, "独立重算唯一开发门槛")
        if gates[0]["eligible_for_netlib"]:
            require(tag("netlib", 0) in groups and "highs" in groups, "通过门槛后的计划扩展须完成")
        actual_homogeneous = sum(item["actual_homogeneous_direction_solves"] for item in metrics)
        require(actual_homogeneous > 0, "实际报告须进入齐次路径")
        result.update(solver_sha256=identity["solver_sha256"], source_files_verified=90,
                      source_zip_sha256=identity["source_zip_sha256"], production_identity_sha256=identity["production_identity_sha256"],
                      compiler_identity_record=identity["compiler_sha256"], library_identity_records_verified=15,
                      historical_test_logs_verified=identity["logs"], referenced_prior_ctest_passed=17,
                      new_production_build=False, current_production_sources_or_build_libraries_required_unchanged=False,
                      referenced_prior_trajectories_verified=trajectories, new_trajectory_runs=0,
                      actual_homogeneous_direction_solves_verified=actual_homogeneous,
                      actual_homogeneous_work_reports_verified=sum(item["actual_homogeneous_work_reports"] for item in metrics),
                      counter_validator_self_check=verify_counter_guards(groups), runner_self_check_recomputed=plan["self_check"],
                      plan_sha256=identity["plan_sha256"], runner_sha256=identity["runner_sha256"],
                      archive_groups_verified=metrics, raw_records_verified=sum(item["raw_records"] for item in metrics),
                      independent_input_files_verified=len(inputs), pending_groups=[],
                      derived_details_verified=details, gates_recomputed=gates, probe_failures=failure_details(groups),
                      prior_v44_cost_audit=audit_prior_common_cost(compare, benchmark, identity),
                      runtime_scope="仅读归档，不编译、不求解、不重新测量。质量/摘要用v44源码zip内冻结Python执行，门槛/成本聚合独立实现。当前生产可以继续开发。",
                      all_raw_and_archived_bytes_equal=True, all_model_and_program_pre_post_hashes_verified=True,
                      per_record_quality_and_penalties_recomputed=True, full_summaries_recomputed=True,
                      actual_compare_process_exit_codes_verified=True)
        if not args.preflight:
            result.update(verify_documents(output))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "final", "source_files_verified", "raw_records_verified", "pending_groups")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    main()


