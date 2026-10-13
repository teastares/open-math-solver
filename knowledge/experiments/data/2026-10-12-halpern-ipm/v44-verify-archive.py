"""独立核验第41轮齐次自对偶 来源、原始质量、旧轨迹与条件扩展；不编译或求解。"""
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
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
EXPECTED_SOLVER = "58291db80a9dc7faaa17d037060d6cc53fe26db06de3bf3e67107e6401a7c24c"
EXPECTED_PLAN = "497c54a5a664c47b2f56c1180451ff6283cf65badb406ba3fdd3dc0a876bfb81"
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
    return f"{stage}-normal_augmented-homogeneous-retries{retries}"


def verify_sources():
    identity = read(DATA / "ipm-v44-identity.json")
    require(identity["solver_sha256"] == EXPECTED_SOLVER and identity["plan_sha256"] == EXPECTED_PLAN,
            "正式冻结程序与测量前计划身份")
    previous = read(DATA / "ipm-v43-identity.json")
    require(set(identity["files"]) == set(previous["files"]) | {"include/solver/detail/ipm_homogeneous.hpp"}, "90份生产来源仅新增齐次标量helper")
    require(len(identity["files"]) == 90, "生产来源数量")
    archive_path = DATA / "ipm-v44-source.zip"
    require(sha(archive_path) == identity["source_zip_sha256"], "源码包整体身份")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 90, "源码归档路径数量或重复项")
        require({name: hashlib.sha256(archive.read(name)).hexdigest() for name in names} == identity["files"],
                "源码包逐文件身份")
    require({name: sha(ROOT / name) for name in identity["files"]} == identity["files"], "当前生产来源发生改变")
    for path in (ROOT / ".tools/solver-ipm-v44.exe", ROOT / "build/release/solver.exe"):
        require(sha(path) == identity["solver_sha256"], "冻结及当前程序身份：" + str(path))
    require(sha(Path(identity["compiler"])) == identity["compiler_sha256"], "编译器身份")
    require(len(identity["libraries"]) == 15 and
            {name: sha(ROOT / name) for name in identity["libraries"]} == identity["libraries"], "15份链接库身份")
    for name, digest in identity["logs"].items():
        require(sha(DATA / name) == digest, "冻结测试日志身份：" + name)
    for name, live in (("iteration-ipm-v44-build.log", ROOT / ".tools/iteration-ipm-v44-build.log"),
                       ("iteration-ipm-v44-ctest-detail.log", ROOT / "build/release/Testing/Temporary/LastTest.log")):
        require((DATA / name).read_bytes() == live.read_bytes(), "测试日志原样归档：" + name)
    build = (DATA / "iteration-ipm-v44-build.log").read_text(encoding="utf-8", errors="replace")
    tests = (DATA / "iteration-ipm-v44-ctest-detail.log").read_text(encoding="utf-8", errors="replace")
    require("100% tests passed out of 17" in build and tests.count("Test Passed.") == 17, "17/17测试结果")
    require(
            "完整 Newton 解析改进 3 步，其中 Krylov 1 步，预条件 4 次，乘积 5 次" in tests,
            "正式测试实际进入2IR及1Arnoldi，并完成独立J验收")
    require("有界 Newton 解析改进 3 步，预条件 4 次，乘积 4 次" in tests,
            "新解析测试实际完成三次IR及真实J，不把分支未触发当覆盖")
    duration = re.search(r"Total Test time \(real\) =\s+([0-9.]+) sec", build)
    checks = re.search(r"自主稀疏内点法检查通过，共 (\d+) 项", tests)
    homogeneous = re.search(r"齐次常数求解 (\d+) 次，方向求解 (\d+) 次", tests)
    require(duration is not None and checks is not None and homogeneous is not None, "实际日志包含时间、检查数及齐次路径工作")
    require(float(duration[1]) == 13.68 and int(checks[1]) == 78614, "正式测试时间和检查数")
    require((int(homogeneous[1]), int(homogeneous[2])) == (20, 40), "解析集成实际触发齐次完整迭代")
    identity["verified_ctest_seconds"] = float(duration[1])
    identity["verified_ipm_checks"] = int(checks[1])
    identity["verified_homogeneous_work"] = {"constant_solves": int(homogeneous[1]), "direction_solves": int(homogeneous[2])}
    for filename, key in (("v44-plan.json", "plan_sha256"), ("v44-run-experiments.py", "runner_sha256"),
                          ("v44-freeze.py", "freeze_script_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验来源：" + filename)
    return identity


def verify_plan(identity, compare):
    plan = read(DATA / "v44-plan.json")
    require(plan["runner_sha256"] == identity["runner_sha256"], "计划运行器身份")
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <=
            datetime.datetime.fromisoformat(identity["created_at_utc"]), "测量前计划与生产冻结时间顺序")
    require(plan["retries"] == [0] and plan["candidate_accuracy"] == "strict" and plan["candidate_method"] == "homogeneous" and
            plan["reference_method"] == "mehrotra" and
            plan["reference_accuracy"] == "strict", "预声明消融维度")
    require((plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
             plan["max_iterations"], plan["scaling_passes"]) == (5, 35, 1, 0, 200, 5), "固定共同预算")
    require(plan["homogeneous_policy"] == {"centering": "(1-alpha_affine)^3", "corrector_residual_factor": "1-sigma",
            "affine_cross_product_factor": 1, "fraction_to_boundary": 0.995, "rescale": False, "scalar_floor": False},
            "冻结齐次数学约定")
    expected_shared = {"linear-solver": "normal_augmented", "singleton-elimination": "on",
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
    return plan


def verify_counts(report, context):
    counters = (*FIELDS, *PRODUCT_FIELDS, *BOUNDED_FIELDS, "ipm_weighted_krylov_steps", "ipm_refinement_steps", "ipm_iterations")
    if report["backend"] != "ipm":
        require(all(key in report and report[key] is None for key in (*counters, *SCALARS)), context + " 非IPM字段必须为null")
        return
    require(all(type(report.get(key)) is int and 0 <= report[key] < 2**63 for key in counters), context + " 非负int64计数")
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
    name = manifest_path.name[len("v44-"):-len("-manifest.json")]
    choices = {tag(stage, retry): (stage, retry) for stage in plan["sets"] for retry in plan["retries"]}
    choices["highs"] = ("highs", 0)
    require(name in choices, "非计划实验组：" + name)
    stage, retries = choices[name]
    declared = plan["sets"]["netlib" if stage == "highs" else stage]
    original = ROOT / "results" / ("round-ipm-v44-" + name)
    summary_path, raw_path = (DATA / ("v44-" + name + suffix) for suffix in ("-summary.json", "-runs.jsonl.gz"))
    manifest, summary = read(manifest_path), read(summary_path)
    raw = gzip.decompress(raw_path.read_bytes())
    require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), name + " 清单字节")
    require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), name + " 汇总字节")
    require(raw == (original / "runs.jsonl").read_bytes(), name + " 原始记录字节")
    require(read(DATA / ("v44-" + name + "-run.log")) == summary, name + " 比较进程原始输出")
    process = read(DATA / ("v44-" + name + "-process.json"))
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
    command_path = DATA / ("v44-" + name + "-command.json")
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
        method = "homogeneous" if side else "mehrotra"
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
        for key in ("method", "linear_accuracy", "direction_retries"):
            require(top[prefix + "ipm-" + key.replace("_", "-")] == str(config["ipm_" + key]), name + " 顶层精度配置")
            require(manifest[("reference_" if side == 0 else "") + "ipm_" + key] == config["ipm_" + key], name + " 清单精度配置")
    if stage != "highs":
        require({key for key in configs[variants[0]] if configs[variants[0]][key] != configs[variants[1]][key]} ==
                {"ipm_method"}, name + " 存在非预声明配置差异")
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
            for key in (*plan["shared_ipm"], "method", "linear-accuracy", "direction-retries"):
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
            "command_sha256": sha(command_path), "process_sha256": sha(DATA / ("v44-" + name + "-process.json")),
            "missing_reports": sum(row["report"] is None for row in rows),
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
    for version, identity in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == identity["solver_sha256"], "轨迹冻结程序当前身份")
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
    return {"count": len(items), "sha256": sha(path), "two_new_counts_integer_zero": True, "two_new_scalars_null": True, "new_method_default_mehrotra": True,
            "program_and_input_pre_post_hashes_verified": True,
            "exit_code_pairs": dict(collections.Counter(str(item["returncodes"]) for item in items))}


def verify_derived(groups, plan_hash, final):
    checked, seen, gates = [], set(), []
    for stage in ("development", "probe", "coverage"):
        path = DATA / f"v44-{stage}-detail.json"
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
        path = DATA / f"v44-retries{retries}-gate.json"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="允许尚未完成的计划组；不检查未完成的文档链接")
    args = parser.parse_args()
    output = DATA / ("v44-verification-preflight.json" if args.preflight else "v44-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False,
              "final": not args.preflight, "verification_script_sha256": sha(Path(__file__))}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        identity = verify_sources()
        sys.path.insert(0, str(ROOT / "scripts"))
        import compare_lp
        import benchmark
        plan = verify_plan(identity, compare_lp)
        groups, metrics, inputs = {}, [], {}
        for path in sorted(DATA.glob("v44-*-manifest.json")):
            name = path.name[len("v44-"):-len("-manifest.json")]
            if args.preflight and not all((DATA / ("v44-" + name + suffix)).exists() for suffix in ("-summary.json", "-runs.jsonl.gz", "-process.json")):
                continue
            metric, group = verify_group(path, identity, plan, compare_lp, benchmark, inputs)
            metrics.append(metric)
            groups[name] = group
        commands = {path.name[len("v44-"):-len("-command.json")] for path in DATA.glob("v44-*-command.json")}
        directories = {path.name[len("round-ipm-v44-"):] for path in (ROOT / "results").glob("round-ipm-v44-*") if path.is_dir()}
        require(commands == directories, "计划命令与原始结果目录不一致")
        declared = {tag(stage, 0) for stage in ("development", "probe")}
        pending = sorted((declared | commands) - set(groups))
        require(set(groups) <= commands and (args.preflight or not pending), "未完成或未计划归档组")
        trajectory_path = DATA / "v44-old-mode-trajectory-runs.json.gz"
        trajectories = verify_trajectories(benchmark) if trajectory_path.exists() else None
        require(args.preflight or trajectories is not None, "最终缺少40组旧轨迹")
        details, gates = verify_derived(groups, identity["plan_sha256"], not args.preflight)
        actual_homogeneous = sum(item["actual_homogeneous_direction_solves"] for item in metrics)
        counter_guards = verify_counter_guards(groups)
        if not args.preflight:
            require(actual_homogeneous > 0, "至少一份真实LP报告必须进入齐次方向求解")
            require(len(gates) == 1, "最终必须独立重算开发门槛")
            if gates[0]["eligible_for_netlib"]:
                require(tag("netlib", 0) in groups and "highs" in groups, "通过门槛后计划的完整集与HiGHS对照尚未完成")
        previous = read(DATA / "ipm-v43-identity.json")
        result.update(solver_sha256=identity["solver_sha256"], source_files_verified=90,
                      source_zip_sha256=identity["source_zip_sha256"], compiler_sha256=identity["compiler_sha256"],
                      libraries_verified=len(identity["libraries"]), test_logs_verified=identity["logs"], ctest_passed=17,
                      ctest_seconds=identity["verified_ctest_seconds"], ipm_checks=identity["verified_ipm_checks"],
                      analytic_homogeneous_work=identity["verified_homogeneous_work"],
                      analytic_full_newton_branch={"refinement_steps": 3, "krylov_steps": 1, "preconditioner_solves": 4, "products": 5},
                      analytic_bounded_newton_branch={"refinement_steps": 3, "krylov_steps": 0, "preconditioner_solves": 4, "products": 4},
                      actual_homogeneous_direction_solves_verified=actual_homogeneous,
                      actual_homogeneous_work_reports_verified=sum(item["actual_homogeneous_work_reports"] for item in metrics),
                      counter_validator_self_check=counter_guards,
                      plan_sha256=identity["plan_sha256"], runner_sha256=identity["runner_sha256"],
                      changed_production_files=[name for name, digest in identity["files"].items() if previous["files"].get(name) != digest],
                      archive_groups_verified=metrics, raw_records_verified=sum(item["raw_records"] for item in metrics),
                      independent_input_files_verified=len(inputs), pending_groups=pending,
                      old_mode_trajectories_verified=trajectories, derived_details_verified=details, gates_recomputed=gates,
                      trajectory_exclusions=["timings", "backend_version", "新增且强制整数零的两项齐次求解启动计数", "新增且强制null的tau/kappa", "新增且强制mehrotra的method"],
                      runtime_scope="仅读归档，不编译、不求解、不重新测量。冻结Python协议负责质量与摘要复算；门槛另行实现核对。",
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
