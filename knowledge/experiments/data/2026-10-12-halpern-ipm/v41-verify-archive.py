"""独立核验第38轮冻结身份、完整 Newton 消融、旧轨迹及扩展门槛；不重新求解。"""
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
EXPECTED_SOLVER = "5a283adb07ed7e15dd3575b021c6a42e1643558a75f4ab09d8cae2a27ff06aa7"
EXPECTED_PLAN = "1c74c36d5dbb115454d81e7ed4a03c8f27ffd8c379915500ff79df20db3f43fb"
PRIOR_PLAN = "7350ccb7de4bb2bddd7be15688fbb3c804577ff6c363b72faf27c52a7ea4d247"
FIELDS = ("ipm_full_newton_preconditioner_solves", "ipm_full_newton_products")
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
    return f"{stage}-normal_augmented-full_newton-retries{retries}"


def verify_sources():
    identity = read(DATA / "ipm-v41-identity.json")
    require(identity["solver_sha256"] == EXPECTED_SOLVER and identity["plan_sha256"] == EXPECTED_PLAN,
            "正式冻结程序与修订2计划身份")
    previous = read(DATA / "ipm-v39-identity.json")
    require(set(identity["files"]) == set(previous["files"]) | {"include/solver/detail/ipm_full_newton.hpp"},
            "89份生产来源范围与上一正式版本加新头文件一致")
    require(len(identity["files"]) == 89, "生产来源数量")
    archive_path = DATA / "ipm-v41-source.zip"
    require(sha(archive_path) == identity["source_zip_sha256"], "源码包整体身份")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 89, "源码归档路径数量或重复项")
        require({name: hashlib.sha256(archive.read(name)).hexdigest() for name in names} == identity["files"],
                "源码包逐文件身份")
    require({name: sha(ROOT / name) for name in identity["files"]} == identity["files"], "当前生产来源发生改变")
    for path in (ROOT / ".tools/solver-ipm-v41.exe", ROOT / "build/release/solver.exe"):
        require(sha(path) == identity["solver_sha256"], "冻结及当前程序身份：" + str(path))
    require(sha(Path(identity["compiler"])) == identity["compiler_sha256"], "编译器身份")
    require(len(identity["libraries"]) == 15 and
            {name: sha(ROOT / name) for name in identity["libraries"]} == identity["libraries"], "15份链接库身份")
    for name, digest in identity["logs"].items():
        require(sha(DATA / name) == digest, "冻结测试日志身份：" + name)
    for name, live in (("iteration-ipm-v41-build.log", ROOT / ".tools/iteration-ipm-v41-build.log"),
                       ("iteration-ipm-v41-ctest-detail.log", ROOT / "build/release/Testing/Temporary/LastTest.log")):
        require((DATA / name).read_bytes() == live.read_bytes(), "测试日志原样归档：" + name)
    build = (DATA / "iteration-ipm-v41-build.log").read_text(encoding="utf-8", errors="replace")
    tests = (DATA / "iteration-ipm-v41-ctest-detail.log").read_text(encoding="utf-8", errors="replace")
    require("100% tests passed out of 17" in build and tests.count("Test Passed.") == 17, "17/17测试结果")
    require("11.24 sec" in build and "共 70500 项" in tests and
            "完整 Newton 解析改进 3 步，其中 Krylov 1 步，预条件 4 次，乘积 5 次" in tests,
            "正式测试实际进入2IR及1Arnoldi，并完成独立J验收")
    for filename, key in (("v41-plan.json", "plan_sha256"), ("v41-run-experiments.py", "runner_sha256"),
                          ("v41-freeze.py", "freeze_script_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验来源：" + filename)
    return identity


def verify_plan(identity, compare):
    plan = read(DATA / "v41-plan.json")
    require(plan["runner_sha256"] == identity["runner_sha256"], "计划运行器身份")
    require(plan["revision"] == 2 and plan["prior_plan_sha256"] == PRIOR_PLAN and
            plan["prior_plan_archive"] == "v41-before-summary-fix-plan.json" and
            plan["prior_runner_archive"] == "v41-before-summary-fix-run-experiments.py", "修订计划历史链")
    prior = read(DATA / plan["prior_plan_archive"])
    require(sha(DATA / plan["prior_plan_archive"]) == PRIOR_PLAN and
            sha(DATA / plan["prior_runner_archive"]) == prior["runner_sha256"], "修复前历史来源逐字节身份")
    current_content = {key: value for key, value in plan.items() if key not in
                       ("created_at_utc", "runner_sha256", "revision", "prior_plan_sha256", "revision_reason", "prior_plan_archive", "prior_runner_archive")}
    require(current_content == {key: value for key, value in prior.items() if key not in ("created_at_utc", "runner_sha256")},
            "修订只修汇总器，输入、预算、配置和扩展门槛原样")
    require(datetime.datetime.fromisoformat(prior["created_at_utc"]) <= datetime.datetime.fromisoformat(plan["created_at_utc"]) <=
            datetime.datetime.fromisoformat(identity["created_at_utc"]), "初始计划、修订计划和生产冻结时间顺序")
    require(plan["retries"] == [0, 8] and plan["candidate_accuracy"] == "full_newton" and
            plan["reference_accuracy"] == "strict", "预声明消融维度")
    require((plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
             plan["max_iterations"], plan["scaling_passes"]) == (5, 35, 1, 0, 200, 5), "固定共同预算")
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
    require(all(type(report.get(key)) is int and report[key] >= 0 for key in FIELDS), context + " 两项非负整数计数")
    if report["options"]["ipm"]["linear_accuracy"] != "full_newton":
        require(all(report[key] == 0 for key in FIELDS), context + " 旧模式新增计数必须为零")
    else:
        require(report["ipm_weighted_krylov_steps"] <= report[FIELDS[0]], context + " 每列Arnoldi必须实际调用预条件器")
    require(report["ipm_weighted_krylov_steps"] <= report["ipm_refinement_steps"], context + " 改进与Arnoldi计数")
    require(all(report["ipm_krylov_recycle_" + key] == 0 for key in
                ("projections", "acceptances", "vectors", "peak_dimension")), context + " 历史关闭计数")


def verify_group(manifest_path, identity, plan, compare, benchmark, input_hashes):
    name = manifest_path.name[len("v41-"):-len("-manifest.json")]
    choices = {tag(stage, retry): (stage, retry) for stage in plan["sets"] for retry in plan["retries"]}
    require(name in choices, "非计划实验组：" + name)
    stage, retries = choices[name]
    declared = plan["sets"][stage]
    original = ROOT / "results" / ("round-ipm-v41-" + name)
    summary_path, raw_path = (DATA / ("v41-" + name + suffix) for suffix in ("-summary.json", "-runs.jsonl.gz"))
    manifest, summary = read(manifest_path), read(summary_path)
    raw = gzip.decompress(raw_path.read_bytes())
    require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), name + " 清单字节")
    require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), name + " 汇总字节")
    require(raw == (original / "runs.jsonl").read_bytes(), name + " 原始记录字节")
    require(read(DATA / ("v41-" + name + "-run.log")) == summary, name + " 比较进程原始输出")
    process = read(DATA / ("v41-" + name + "-process.json"))
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
    command_path = DATA / ("v41-" + name + "-command.json")
    command = read(command_path)
    top = flags(command, 2)
    require(Path(command[1]) == ROOT / "scripts/compare_lp.py" and
            Path(top["--solver"]) == ROOT / ".tools/solver-ipm-v41.exe" and
            Path(top["--output-dir"]) == original and top["--models-file"] == declared["models_file"], name + " 比较入口")
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        require(float(top["--" + key.replace("_", "-")]) == plan[key], name + " 比较资源")
    require(int(top["--repeats"]) == declared["repeats"], name + " 比较重复")
    for side, variant in enumerate(variants):
        prefix = "--reference-" if side == 0 else "--"
        accuracy = "full_newton" if side else "strict"
        config = configs[variant]
        require(config["backend"] == "ipm" and config["ipm_linear_accuracy"] == accuracy and
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
        for key in ("linear_accuracy", "direction_retries"):
            require(top[prefix + "ipm-" + key.replace("_", "-")] == str(config["ipm_" + key]), name + " 顶层精度配置")
            require(manifest[("reference_" if side == 0 else "") + "ipm_" + key] == config["ipm_" + key], name + " 清单精度配置")
    require({key for key in configs[variants[0]] if configs[variants[0]][key] != configs[variants[1]][key]} ==
            {"ipm_linear_accuracy"}, name + " 存在非预声明配置差异")
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
        require(Path(solve_command[0]) == ROOT / ".tools/solver-ipm-v41.exe" and
                solve_command[1:3] == ["solve", row["input"]], name + " 求解入口")
        actual = flags(solve_command, 3)
        require(actual["--backend"] == "ipm" and float(actual["--time-limit"]) == 5 and
                actual["--threads"] == "1" and actual["--seed"] == "0" and
                actual["--iteration-limit"] == "200" and actual["--scaling-passes"] == "5", name + " 求解预算")
        config = configs[variant]
        for key in (*plan["shared_ipm"], "linear-accuracy", "direction-retries"):
            require(actual["--ipm-" + key] == str(config["ipm_" + key.replace("-", "_")]), name + " 求解配置：" + key)
        report = row["report"]
        if report is not None:
            require(benchmark.strict_json(row["stdout"]) == report, name + " stdout与报告")
            require(report["backend"] == "ipm" and report["backend_version"] == "ipm-cpu-0.37.0", name + " 报告版本")
            require(row["status"] == report["status"] and benchmark.classify(row["returncode"], report) == row["classification"], name + " 分类与退出码")
            options, ipm = report["options"], report["options"]["ipm"]
            require((options["time_limit"], options["threads"], options["seed"], ipm["max_iterations"], ipm["scaling_passes"]) ==
                    (5, 1, 0, 200, 5), name + " 报告预算")
            for key, field in BOOLEAN_FIELDS.items():
                require(ipm[field] is (plan["shared_ipm"][key] == "on"), name + " 报告布尔配置")
            for key in ("linear_solver", "linear_accuracy", "direction_retries", "krylov_history_size", "central_corrections"):
                require(ipm[key] == config["ipm_" + key], name + " 报告精度配置")
            verify_counts(report, name)
            require(report["ipm_presolved_singleton_rows"] == report["ipm_presolved_fixed_columns"] == 0,
                    name + " 本轮输入归并关闭，相关实际计数必须为零")
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
            "command_sha256": sha(command_path), "process_sha256": sha(DATA / ("v41-" + name + "-process.json")),
            "missing_reports": sum(row["report"] is None for row in rows),
            "full_newton_work": {side: {key: sum((row["report"] or {}).get(key, 0) for row in rows if row["requested_variant"] == side)
                                             for key in FIELDS} for side in variants}}, (manifest, summary, rows)


def verify_trajectories(benchmark):
    path = DATA / "v41-old-mode-trajectory-runs.json.gz"
    items = json.loads(gzip.decompress(path.read_bytes()))
    require(len(items) == 32, "32组旧轨迹")
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (39, 41)}
    for version, identity in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == identity["solver_sha256"], "轨迹冻结程序当前身份")
    expected_models = {model["input"] for model in read(DATA / "v41-plan.json")["sets"]["development"]["models"]}
    expected_configs = {("qdldl", "strict", 0, False), ("normal_augmented", "strict", 0, False),
                        ("normal_augmented", "blockwise", 8, True), ("normal_augmented", "blockwise_weighted", 8, False)}
    seen = set()
    for item in items:
        require(item["model"] in expected_models and sha(Path(item["model"])) == item["model_sha256"], "轨迹输入身份")
        require(all(len(item[key]) == 2 for key in ("reports", "commands", "returncodes", "hashes", "outputs")), "轨迹完整成对记录")
        normalized = copy.deepcopy(item["reports"])
        for index, version in enumerate((39, 41)):
            report = item["reports"][index]
            require(benchmark.strict_json(item["outputs"][index]["stdout"]) == report, "轨迹原始stdout")
            require(report["backend"] == "ipm" and report["backend_version"] ==
                    ("ipm-cpu-0.36.0" if version == 39 else "ipm-cpu-0.37.0"), "轨迹版本")
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
            if version == 41:
                verify_counts(report, "旧轨迹")
            else:
                require(all(key not in report for key in FIELDS), "旧版本不应出现新增字段")
            for key in ("timings", "backend_version", *FIELDS):
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
    require(seen == {(model, config) for model in expected_models for config in expected_configs}, "32轨迹覆盖")
    return {"count": len(items), "sha256": sha(path), "two_new_counts_integer_zero": True,
            "program_and_input_pre_post_hashes_verified": True,
            "exit_code_pairs": dict(collections.Counter(str(item["returncodes"]) for item in items))}


def verify_derived(groups, plan_hash, final):
    checked, seen, gates = [], set(), []
    for stage in ("development", "probe", "coverage"):
        path = DATA / f"v41-{stage}-detail.json"
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
    require(not final or seen == set(groups), "派生详情缺组")
    for retries in (0, 8):
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
        path = DATA / f"v41-retries{retries}-gate.json"
        if path.exists() or final:
            require(read(path) == gate, "扩展门槛独立重算")
        require(tag("netlib", retries) not in groups or gate["eligible_for_netlib"], "未经预声明门槛扩大完整集")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="允许尚未完成的计划组；不检查未完成的文档链接")
    args = parser.parse_args()
    output = DATA / ("v41-verification-preflight.json" if args.preflight else "v41-verification.json")
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
        for path in sorted(DATA.glob("v41-*-manifest.json")):
            name = path.name[len("v41-"):-len("-manifest.json")]
            if args.preflight and not all((DATA / ("v41-" + name + suffix)).exists() for suffix in ("-summary.json", "-runs.jsonl.gz", "-process.json")):
                continue
            metric, group = verify_group(path, identity, plan, compare_lp, benchmark, inputs)
            metrics.append(metric)
            groups[name] = group
        commands = {path.name[len("v41-"):-len("-command.json")] for path in DATA.glob("v41-*-command.json")}
        directories = {path.name[len("round-ipm-v41-"):] for path in (ROOT / "results").glob("round-ipm-v41-*") if path.is_dir()}
        require(commands == directories, "计划命令与原始结果目录不一致")
        declared = {tag(stage, retry) for stage in ("development", "probe") for retry in (0, 8)}
        pending = sorted((declared | commands) - set(groups))
        require(set(groups) <= commands and (args.preflight or not pending), "未完成或未计划归档组")
        trajectories = verify_trajectories(benchmark)
        details, gates = verify_derived(groups, identity["plan_sha256"], not args.preflight)
        previous = read(DATA / "ipm-v39-identity.json")
        result.update(solver_sha256=identity["solver_sha256"], source_files_verified=89,
                      source_zip_sha256=identity["source_zip_sha256"], compiler_sha256=identity["compiler_sha256"],
                      libraries_verified=len(identity["libraries"]), test_logs_verified=identity["logs"], ctest_passed=17,
                      ctest_seconds=11.24, ipm_checks=70500,
                      analytic_full_newton_branch={"refinement_steps": 3, "krylov_steps": 1, "preconditioner_solves": 4, "products": 5},
                      plan_sha256=identity["plan_sha256"], runner_sha256=identity["runner_sha256"],
                      plan_revision=2, prior_plan_sha256=PRIOR_PLAN,
                      changed_production_files=[name for name, digest in identity["files"].items() if previous["files"].get(name) != digest],
                      archive_groups_verified=metrics, raw_records_verified=sum(item["raw_records"] for item in metrics),
                      independent_input_files_verified=len(inputs), pending_groups=pending,
                      old_mode_trajectories_verified=trajectories, derived_details_verified=details, gates_recomputed=gates,
                      trajectory_exclusions=["timings", "backend_version", "新增且强制整数零的两项完整Newton工作计数"],
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
