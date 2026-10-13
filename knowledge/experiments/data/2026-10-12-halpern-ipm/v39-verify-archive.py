"""独立核验 v39 冻结来源、原始结果、关闭轨迹及文档链接，不重新运行求解。"""
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
EXPECTED_SOLVER = "75c20fc0b39f31125f745540fc2a46f218ccf28c713cb30c341b4705b86ac7bf"
HISTORY_FIELDS = tuple("ipm_krylov_recycle_" + suffix for suffix in
                       ("projections", "acceptances", "vectors", "peak_dimension"))
DECLARED_DEVELOPMENT = {
    "development-normal_augmented-blockwise-presolve_off-ipm",
    "development-normal_augmented-blockwise-presolve_on-ipm",
    "development-factorhighs-blockwise-presolve_off-ipm",
    "development-normal_augmented-strict-presolve_off-ipm",
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


def verify_sources():
    identity = read(DATA / "ipm-v39-identity.json")
    expected = identity["solver_sha256"]
    require(expected == EXPECTED_SOLVER, "本轮正式冻结程序身份")
    require(len(identity["files"]) == 88, "冻结来源数量不符")
    with zipfile.ZipFile(DATA / "ipm-v39-source.zip") as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        require(len(names) == len(set(names)), "源码归档含重复路径")
        zipped = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    require(zipped == identity["files"], "源码包与身份清单不符")
    require({name: sha(ROOT / name) for name in identity["files"]} == identity["files"],
            "当前生产源码与冻结版本不符")
    for path in (ROOT / ".tools/solver-ipm-v39.exe", ROOT / "build/release/solver.exe"):
        require(sha(path) == expected, "冻结与当前程序不符：" + str(path))
    for archived, live in (
        ("iteration-ipm-v39-build.log", ROOT / ".tools/iteration-ipm-v39-build.log"),
        ("iteration-ipm-v39-ctest-detail.log", ROOT / "build/release/Testing/Temporary/LastTest.log"),
    ):
        require((DATA / archived).read_bytes() == live.read_bytes(), "构建测试日志原样归档：" + archived)
    build = (DATA / "iteration-ipm-v39-build.log").read_text(encoding="utf-8", errors="replace")
    tests = (DATA / "iteration-ipm-v39-ctest-detail.log").read_text(encoding="utf-8", errors="replace")
    require("100% tests passed out of 17" in build and "10.91 sec" in build, "CTest 总结不符")
    require(tests.count("Test Passed.") == 17 and "66512" in tests, "CTest 详细检查数不符")
    require("历史入池 166 次，投影 10 次，投影直接通过 1 次" in tests, "CTest 回收分支覆盖计数")
    return identity


def verify_history_counts(report, requested, context):
    require(type(requested) is int and 0 <= requested <= 8, context + " 历史空间配置范围")
    require(all(key in report and type(report[key]) is int and report[key] >= 0
                for key in HISTORY_FIELDS), context + " 回收四计数字段必须完整且为非负整数")
    projections, acceptances, vectors, peak = (report[key] for key in HISTORY_FIELDS)
    require(acceptances <= projections and peak <= requested and peak <= vectors,
            context + " 回收计数边界")
    require(bool(vectors) == bool(peak), context + " 历史入池与峰值一致性")
    require(requested != 0 or all(report[key] == 0 for key in HISTORY_FIELDS), context + " 关闭历史必须全零")


def verify_group(manifest_path, identity, compare, benchmark, input_hashes):
    tag = manifest_path.name[len("v39-"):-len("-manifest.json")]
    original = ROOT / "results" / ("round-ipm-v39-" + tag)
    manifest = read(manifest_path)
    summary_path = DATA / ("v39-" + tag + "-summary.json")
    summary = read(summary_path)
    raw_path = DATA / ("v39-" + tag + "-runs.jsonl.gz")
    raw = gzip.decompress(raw_path.read_bytes())
    require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), tag + " 清单字节")
    require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), tag + " 汇总字节")
    require(raw == (original / "runs.jsonl").read_bytes(), tag + " 原始记录字节")
    require(read(DATA / ("v39-" + tag + "-run.log")) == summary, tag + " 比较进程输出总结")
    rows = [benchmark.strict_json(line) for line in raw.splitlines()]
    development = tag.startswith("development-")
    require(tag.startswith(("development-", "netlib-")), tag + " 不明阶段")
    require(manifest["repeats"] == (3 if development else 1), tag + " 重复次数")
    require(len(manifest["models"]) == (8 if development else 91), tag + " 预定模型数量")
    require(manifest["time_limit"] == 5 and manifest["wall_timeout"] == 35
            and manifest["threads"] == 1 and manifest["seed"] == 0, tag + " 共同资源协议")
    models_list = ROOT / "benchmarks" / ("development.txt" if development else "netlib-all.txt")
    loaded, digest = compare.load_models(models_list)
    require(Path(manifest["models_file"]) == models_list
            and manifest["models_file_sha256"] == digest and manifest["models"] == loaded,
            tag + " 模型列表及当前输入")
    require(manifest["comparison_script_sha256"] == identity["files"]["scripts/compare_lp.py"]
            and manifest["benchmark_script_sha256"] == identity["files"]["scripts/benchmark.py"],
            tag + " 协议脚本身份")
    expected = identity["solver_sha256"]
    require(manifest["solver_sha256"] == expected and manifest["solver_hash_error"] is None, tag + " 程序身份")
    variants = (manifest["reference_variant"], manifest["candidate_variant"])
    require(len(set(variants)) == 2 and set(manifest["variants"]) == set(variants), tag + " 两侧独立标签")
    configs = manifest["variants"]
    for side, variant in enumerate(variants):
        config = configs[variant]
        require(config["backend"] == "ipm" and config["ipm_linear_solver"] in ("normal_augmented", "factorhighs"),
                tag + " 后端选择")
        require(config["ipm_linear_accuracy"] in ("strict", "blockwise")
                and config["ipm_direction_retries"] == (0 if config["ipm_linear_accuracy"] == "strict" else 8),
                tag + " 精度模式及方向预算")
        require(config["ipm_krylov_history_size"] == (4 if side else 0), tag + " 历史预算消融")
        presolve = "on" if "-presolve_on-" in tag else "off"
        require(config["ipm_presolve_fixed_elimination"] == config["ipm_singleton_presolve"] == presolve,
                tag + " 两侧输入预处理一致")
        require(config["ipm_singleton_elimination"] == config["ipm_kkt_equilibration"] ==
                config["ipm_fixed_variable_elimination"] == "on", tag + " 共同消元及平衡配置")
        require(manifest["solvers"][variant]["sha256"] == expected
                and manifest["solvers"][variant]["hash_error"] is None, tag + " 两侧程序身份")
    differing = {key for key in configs[variants[0]] if configs[variants[0]][key] != configs[variants[1]][key]}
    require(differing == {"ipm_krylov_history_size"}, tag + " 存在非预定配置差异")
    command_path = DATA / ("v39-" + tag + "-command.json")
    command = read(command_path)
    top = flags(command, 2)
    require(Path(command[1]) == ROOT / "scripts/compare_lp.py", tag + " 比较入口")
    require(Path(top["--solver"]) == ROOT / ".tools/solver-ipm-v39.exe"
            and Path(top["--models-file"]) == models_list and Path(top["--output-dir"]) == original,
            tag + " 比较命令路径")
    require(float(top["--time-limit"]) == 5 and float(top["--wall-timeout"]) == 35
            and int(top["--threads"]) == 1 and int(top["--seed"]) == 0
            and int(top["--repeats"]) == manifest["repeats"], tag + " 比较命令预算")
    require(top["--candidate-backend"] == top["--reference-backend"] == "ipm", tag + " 比较命令后端")
    for side, variant in enumerate(variants):
        prefix = "--reference-" if side == 0 else "--"
        for name, value in configs[variant].items():
            if name.startswith("ipm_") and value is not None:
                require(top.get(prefix + name.replace("_", "-")) == str(value), tag + " 顶层选项：" + name)
    schedule = list(compare.schedule(manifest["models"], manifest["repeats"], variants))
    require(len(rows) == len(schedule) == summary["run_count"], tag + " 原始记录完整性")
    boolean_fields = {
        "ipm_singleton_elimination": "eliminate_singletons",
        "ipm_dynamic_regularization": "dynamic_regularization",
        "ipm_kkt_equilibration": "equilibrate_kkt",
        "ipm_inexact_prefilter": "inexact_prefilter",
        "ipm_fixed_variable_elimination": "eliminate_fixed_variables",
        "ipm_singleton_presolve": "presolve_singleton_rows",
        "ipm_presolve_fixed_elimination": "eliminate_presolved_fixed_variables",
    }
    for index, (row, scheduled) in enumerate(zip(rows, schedule)):
        model, repetition, position, variant = scheduled
        require((row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"])
                == (model["model_index"], repetition, position, variant), tag + " 交替顺序或重复测量")
        require(row["run_index"] == index and row["seed"] == 0 and row["input"] == model["input"], tag + " 原始索引")
        if model["input"] not in input_hashes:
            input_hashes[model["input"]] = sha(Path(model["input"]))
        require(input_hashes[model["input"]] == model["sha256"], tag + " 当前输入身份")
        require(all(row[key] == model["sha256"] for key in (
            "model_sha256", "frozen_model_sha256", "model_sha256_after")), tag + " 输入前后身份")
        require(all(row[key] == expected for key in (
            "solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")), tag + " 程序前后身份")
        report = row["report"]
        require(benchmark.strict_json(row["stdout"]) == report, tag + " 原始stdout与解析报告")
        require(report["backend_version"] == "ipm-cpu-0.36.0" and report["backend"] == "ipm", tag + " 报告版本")
        require(row["status"] == report["status"]
                and benchmark.classify(row["returncode"], report) == row["classification"], tag + " 状态与退出码")
        cfg = configs[variant]
        ipm = report["options"]["ipm"]
        require(report["options"]["time_limit"] == 5 and report["options"]["threads"] == 1
                and report["options"]["seed"] == 0 and ipm["max_iterations"] == 200
                and ipm["scaling_passes"] == 5, tag + " 报告共同预算")
        require(ipm["dynamic_regularization"] is False and ipm["inexact_prefilter"] is False
                and ipm["central_corrections"] == 0, tag + " 隐式默认选项")
        solve_command = row["command"]
        require(Path(solve_command[0]) == ROOT / ".tools/solver-ipm-v39.exe"
                and solve_command[1:3] == ["solve", row["input"]], tag + " solve入口")
        actual_flags = flags(solve_command, 3)
        require(actual_flags["--backend"] == "ipm" and float(actual_flags["--time-limit"]) == 5
                and int(actual_flags["--threads"]) == 1 and int(actual_flags["--seed"]) == 0, tag + " solve资源")
        for attribute, field in boolean_fields.items():
            requested = cfg[attribute]
            if requested is not None:
                require(ipm[field] is (requested == "on")
                        and actual_flags["--" + attribute.replace("_", "-")] == requested, tag + " 布尔选项：" + attribute)
        for field in ("linear_solver", "linear_accuracy", "direction_retries", "krylov_history_size"):
            attribute = "ipm_" + field
            require(ipm[field] == cfg[attribute]
                     and actual_flags["--" + attribute.replace("_", "-")] == str(cfg[attribute]), tag + " 数值选项")
        verify_history_counts(report, ipm["krylov_history_size"], tag)
        merged, fixed, total_fixed = (report[key] for key in (
            "ipm_presolved_singleton_rows", "ipm_presolved_fixed_columns", "ipm_eliminated_fixed_variables"))
        require(all(type(value) is int and value >= 0 for value in (merged, fixed, total_fixed)), tag + " 计数类型")
        require(merged <= report["model"]["rows"] and fixed <= total_fixed <= report["model"]["columns"]
                and fixed <= merged, tag + " 计数边界")
        require(ipm["presolve_singleton_rows"] or merged == 0, tag + " 关闭归并计数")
        require(ipm["eliminate_presolved_fixed_variables"] or fixed == 0, tag + " 关闭新消列计数")
        numerical = report["timings"]["solve_seconds"] + report["timings"]["verification_seconds"]
        require(math.isfinite(numerical) and numerical >= 0 and math.isfinite(row["elapsed_wall_seconds"])
                and row["elapsed_wall_seconds"] > 0, tag + " 有限耗时")
        close(row["numerical_seconds"], numerical, tag + " 完整数值耗时")
        failure = compare.quality_failure(row, model, model["sha256"], expected, expected)
        if failure is None and row["elapsed_wall_seconds"] > 35:
            failure = "wall_budget_exceeded"
        if failure is None and numerical > 5:
            failure = "numerical_budget_exceeded"
        require(row["quality_pass"] is (failure is None) and row["quality_failure"] == failure, tag + " 共同质量")
        require(row["wall_budget_exceeded"] is (row["elapsed_wall_seconds"] > 35)
                and row["numerical_budget_exceeded"] is (numerical > 5), tag + " 超时标志")
        close(row["quality_penalized_wall_seconds"], min(row["elapsed_wall_seconds"], 35) if failure is None else 35,
              tag + " 墙钟失败罚分")
        close(row["quality_penalized_numerical_seconds"], min(numerical, 5) if failure is None else 5,
              tag + " 数值失败罚分")
        close(row["penalized_capped_seconds"], min(row["elapsed_wall_seconds"], 5) if row["classification"] == "resolved" else 5,
              tag + " 基准基础罚分")
    recomputed = compare.summarize(rows, manifest["models"], manifest["repeats"], 35, 5, variants)
    recomputed["configurations"] = configs
    recomputed["output_dir"] = str(original)
    require(recomputed == summary, tag + " 完整摘要重算不一致")
    metrics = {
        "tag": tag, "models": len(manifest["models"]), "repeats": manifest["repeats"], "raw_records": len(rows),
        "manifest_sha256": sha(manifest_path), "summary_sha256": sha(summary_path),
        "compressed_raw_sha256": sha(raw_path), "command_sha256": sha(command_path),
        "quality_pass_count": summary["quality_pass_count"],
        "quality_failure_count": len(rows) - summary["quality_pass_count"],
        "compare_exit_code_implied_by_complete_results": 0 if summary["quality_pass_count"] == len(rows) else 1,
        "recycle_counts": {
            variant: {
                **{key: sum(row["report"][key] for row in rows if row["requested_variant"] == variant)
                   for key in HISTORY_FIELDS[:-1]},
                "peak_dimension": max(row["report"][HISTORY_FIELDS[-1]] for row in rows
                                      if row["requested_variant"] == variant),
                "projection_runs": sum(row["report"][HISTORY_FIELDS[0]] > 0 for row in rows
                                       if row["requested_variant"] == variant),
            } for variant in variants
        },
    }
    return metrics, (manifest, summary, rows)


def verify_trajectories(benchmark):
    path = DATA / "v39-history-zero-trajectory-runs.json.gz"
    items = json.loads(gzip.decompress(path.read_bytes()))
    require(len(items) == 32, "关闭轨迹数量")
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (38, 39)}
    for version, identity in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == identity["solver_sha256"], "轨迹冻结程序身份")
    models_file = ROOT / "benchmarks/development.txt"
    expected_models = {
        str((models_file.parent / line.strip()).resolve())
        for line in models_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    }
    require(len(expected_models) == 8, "轨迹开发输入数量")
    expected_configs = {
        ("qdldl", "strict", 0, False, False),
        ("normal_augmented", "strict", 0, False, False),
        ("normal_augmented", "blockwise", 8, True, True),
        ("normal_augmented", "blockwise_weighted", 8, False, False),
    }
    seen, configurations = set(), collections.Counter()
    for item in items:
        require(item["model"] in expected_models and sha(Path(item["model"])) == item["model_sha256"],
                "轨迹输入身份与开发列表")
        require(len(item["reports"]) == len(item["commands"]) == len(item["returncodes"]) == 2, "轨迹成对记录")
        reports = copy.deepcopy(item["reports"])
        require(type(reports[1]["options"]["ipm"]["krylov_history_size"]) is int and
                reports[1]["options"]["ipm"]["krylov_history_size"] == 0, "新版本轨迹必须明确关闭历史")
        verify_history_counts(reports[1], 0, "旧轨迹")
        require("krylov_history_size" not in reports[0]["options"]["ipm"] and
                all(key not in reports[0] for key in HISTORY_FIELDS), "v38轨迹不应有新增字段")
        for index, version in enumerate((38, 39)):
            report = reports[index]
            command = item["commands"][index]
            require(Path(command[0]) == ROOT / f".tools/solver-ipm-v{version}.exe"
                    and command[1:3] == ["solve", item["model"]], "轨迹命令路径")
            actual = flags(command, 3)
            require(actual["--backend"] == "ipm" and float(actual["--time-limit"]) == 5
                    and int(actual["--threads"]) == 1 and int(actual["--seed"]) == 0, "轨迹预算")
            require(report["backend_version"] == ("ipm-cpu-0.35.0" if version == 38 else "ipm-cpu-0.36.0"),
                    "轨迹运行版本")
            require(report["backend"] == "ipm" and report["options"]["time_limit"] == 5
                    and report["options"]["threads"] == 1 and report["options"]["seed"] == 0, "轨迹报告预算")
            reported_ipm = report["options"]["ipm"]
            for name in ("linear_solver", "linear_accuracy", "direction_retries"):
                require(actual["--ipm-" + name.replace("_", "-")] == str(reported_ipm[name]),
                        "轨迹命令与精度配置")
            for flag, field in (
                ("singleton-presolve", "presolve_singleton_rows"),
                ("singleton-elimination", "eliminate_singletons"),
                ("fixed-variable-elimination", "eliminate_fixed_variables"),
                ("presolve-fixed-elimination", "eliminate_presolved_fixed_variables"),
                ("kkt-equilibration", "equilibrate_kkt"),
            ):
                require(reported_ipm[field] is (actual["--ipm-" + flag] == "on"),
                        "轨迹命令与布尔配置")
            if version == 39:
                require(actual["--ipm-krylov-history-size"] == "0", "轨迹新开关命令")
            else:
                require("--ipm-krylov-history-size" not in actual, "旧程序收到新开关")
            require(benchmark.classify(item["returncodes"][index], report) in ("resolved", "incomplete"),
                    "轨迹退出码与报告状态冲突")
        ipm = reports[1]["options"]["ipm"]
        config = (ipm["linear_solver"], ipm["linear_accuracy"], ipm["direction_retries"],
                  ipm["presolve_singleton_rows"], ipm["eliminate_presolved_fixed_variables"])
        require(config in expected_configs, "旧轨迹配置")
        require(ipm["eliminate_singletons"] is (config[0] != "qdldl")
                and ipm["eliminate_fixed_variables"] is (config[0] != "qdldl"), "旧轨迹消元选项")
        require((item["model"], config) not in seen, "重复旧轨迹")
        seen.add((item["model"], config))
        configurations[config] += 1
        for report in reports:
            for key in ("timings", "backend_version", *HISTORY_FIELDS):
                report.pop(key, None)
            report["options"]["ipm"].pop("krylov_history_size", None)
        require(reports[0] == reports[1] and item["returncodes"][0] == item["returncodes"][1]
                and item["non_time_equal"] is True, "旧非时间轨迹或退出码不同")
    require(set(configurations) == expected_configs and set(configurations.values()) == {8}, "轨迹模型覆盖")
    return {"count": len(items), "compressed_sha256": sha(path),
            "solver_current_hashes_verified": {str(version): identity["solver_sha256"]
                                               for version, identity in identities.items()},
            "four_new_counts_present_and_integer_zero": True,
            "exit_code_pairs": dict(collections.Counter(str(item["returncodes"]) for item in items))}


def verify_probes(identity, benchmark):
    metrics = []
    for filename, final_cache in (("v39-analytic-probe-runs.json.gz", True),
                                  ("v39-analytic-probe-pre-cache-runs.json.gz", False)):
        path = DATA / filename
        rows = json.loads(gzip.decompress(path.read_bytes()))
        require(len(rows) == 56, "解析诊断必须保留全部56条：" + filename)
        expected_hash = identity["solver_sha256"] if final_cache else \
            "7f7a69d47f182f6768dad014018ff03a80f3f215d57f837997fe1ec4e7136236"
        seen, models, statuses = set(), set(), collections.Counter()
        for row in rows:
            require(row["solver_sha256"] == expected_hash, "解析诊断版本混入：" + filename)
            require(Path(row["input"]).read_text(encoding="utf-8") == row["input_content"] and
                    sha(Path(row["input"])) == row["input_sha256"], "解析诊断输入与内嵌内容")
            require(row["report"] is not None and row["parse_error"] is None, "解析诊断缺少报告")
            report = benchmark.strict_json(row["stdout"])
            require(report == row["report"] and report["backend"] == "ipm" and
                    report["backend_version"] == "ipm-cpu-0.36.0", "解析诊断stdout与报告")
            require(benchmark.classify(row["returncode"], report) in ("resolved", "incomplete"), "解析诊断退出码")
            command = row["command"]
            require(Path(command[0]) == ROOT / "build/release/solver.exe" and
                    command[1:3] == ["solve", row["input"]], "解析诊断命令入口")
            actual = flags(command, 3)
            require(actual["--backend"] == "ipm" and float(actual["--time-limit"]) == 2 and
                    actual["--iteration-limit"] == "80" and actual["--threads"] == "1" and
                    actual["--seed"] == "0", "解析诊断预算")
            ipm = report["options"]["ipm"]
            for name, expected in (("direction_retries", 8), ("krylov_history_size", 4)):
                require(ipm[name] == expected and actual["--ipm-" + name.replace("_", "-")] == str(expected),
                        "解析诊断回收及重试预算")
            require(report["options"]["time_limit"] == 2 and ipm["max_iterations"] == 80, "解析诊断报告预算")
            require(ipm["linear_solver"] in ("normal_augmented", "qdldl") and
                    ipm["linear_accuracy"] in ("strict", "blockwise"), "解析诊断配置")
            for name in ("linear_solver", "linear_accuracy"):
                require(actual["--ipm-" + name.replace("_", "-")] == ipm[name], "解析诊断命令配置")
            verify_history_counts(report, 4, "解析诊断")
            key = (row["model"], ipm["linear_solver"], ipm["linear_accuracy"])
            require(key not in seen, "重复解析诊断记录")
            seen.add(key)
            models.add(row["model"])
            statuses[report["status"]] += 1
        require(len(models) == 14, "解析诊断模型数量")
        projected = sum(row["report"][HISTORY_FIELDS[0]] > 0 for row in rows)
        require(not final_cache or projected == 22, "正式解析诊断投影覆盖数量")
        metrics.append({"path": filename, "compressed_sha256": sha(path), "records": len(rows),
                        "recorded_solver_sha256": expected_hash, "matches_final_frozen_solver": final_cache,
                        "projection_runs": projected, "missing_reports": 0, "statuses": dict(statuses),
                        "scope": "仅解析构造的分支诊断，全部保留，不计入公开性能或覆盖成绩。" if final_cache else
                                 "缓存改造前的旧版本诊断；仅核对已记录身份与数据一致性，不声称现存旧程序或旧源码已核验。"})
    return metrics


def verify_details(groups, final):
    checked, seen = [], set()
    for filename in ("v39-development-detail.json", "v39-coverage-detail.json"):
        path = DATA / filename
        if not path.exists() and not final:
            continue
        detail = read(path)
        for tag, item in detail.items():
            require(tag in groups and tag not in seen, "派生详情组名或重复条目")
            seen.add(tag)
            manifest, summary, rows = groups[tag]
            require(item["summary"] == summary["variants"], tag + " 派生摘要")
            expected = []
            for row in rows:
                report = row["report"]
                expected.append({
                    "model": Path(row["input"]).stem,
                    "candidate": row["requested_variant"] == manifest["candidate_variant"],
                    "repetition": row["repetition"], "quality_pass": row["quality_pass"],
                    "quality_failure": row["quality_failure"], "status": row["status"],
                    "message": report["message"], "numerical_seconds": row["numerical_seconds"],
                    "work": {key: value for key, value in report.items() if key.startswith("ipm_")},
                    "optimality": report["lp_optimality"],
                })
            require(item["models"] == expected, tag + " 派生逐次细节")
            gain, lost = [], []
            for model in summary["models"]:
                before = model["quality_pass_counts"][manifest["reference_variant"]] == manifest["repeats"]
                after = model["quality_pass_counts"][manifest["candidate_variant"]] == manifest["repeats"]
                if after and not before:
                    gain.append(Path(model["input"]).stem)
                if before and not after:
                    lost.append(Path(model["input"]).stem)
            require(sorted(item["gain"]) == sorted(gain) and sorted(item["lost"]) == sorted(lost), tag + " 派生得失")
        checked.append({"path": filename, "sha256": sha(path), "groups": len(detail)})
    missing = sorted(set(groups) - seen)
    require(not final or not missing, "派生详情缺少组：" + str(missing))
    return checked, missing


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
            if local == output.resolve():
                self_links += 1
            if not local.exists():
                broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    require(not broken, "文档断链：" + str(broken))
    require(self_links >= 2, "实验正文和changelog应链接本轮验证报告")
    return {"markdown_files_checked": len(documents), "local_links_checked": links,
            "verification_report_links": self_links, "broken_links": broken,
            "markdown_scope": "项目维护文档，不含build第三方依赖、.tools与results。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="只核对已归档数据，文档尚未完成时不写最终报告")
    args = parser.parse_args()
    output = DATA / ("v39-verification-preflight.json" if args.preflight else "v39-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "passed": False, "final": not args.preflight, "verification_script_sha256": sha(Path(__file__))}
    # 先写明确未通过状态，使最终报告自身的文档链接可核验；失败时不保留过期成功状态。
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        identity = verify_sources()
        sys.path.insert(0, str(ROOT / "scripts"))
        import compare_lp
        import benchmark
        groups, metrics, inputs = {}, [], {}
        manifests = sorted(DATA.glob("v39-*-manifest.json"))
        for path in manifests:
            tag = path.name[len("v39-"):-len("-manifest.json")]
            complete = all((DATA / ("v39-" + tag + suffix)).exists()
                           for suffix in ("-summary.json", "-runs.jsonl.gz"))
            if args.preflight and not complete:
                continue
            metric, group = verify_group(path, identity, compare_lp, benchmark, inputs)
            metrics.append(metric)
            groups[metric["tag"]] = group
        planned = {path.name[len("v39-"):-len("-command.json")] for path in DATA.glob("v39-*-command.json")}
        result_directories = {path.name[len("round-ipm-v39-"):] for path in (ROOT / "results").glob("round-ipm-v39-*") if path.is_dir()}
        require(planned == result_directories, "预定命令与原始结果目录不一致")
        pending = sorted((planned | DECLARED_DEVELOPMENT) - set(groups))
        require(set(groups) <= planned and (args.preflight or not pending), "存在未完成或未计划归档组")
        trajectory = verify_trajectories(benchmark)
        probes = verify_probes(identity, benchmark)
        details, missing_details = verify_details(groups, not args.preflight)
        previous = read(DATA / "ipm-v38-identity.json")
        result.update(
            solver_sha256=identity["solver_sha256"], source_files_verified=len(identity["files"]),
            source_current_scope="当前88份生产文件、冻结源码包、身份清单以及build/.tools程序逐一核对。",
            source_zip_sha256=sha(DATA / "ipm-v39-source.zip"),
            changed_production_files=[name for name, digest in identity["files"].items() if previous["files"].get(name) != digest],
            archive_groups_verified=metrics, raw_records_verified=sum(item["raw_records"] for item in metrics),
            model_and_solver_pre_post_hash_records_verified=sum(item["raw_records"] for item in metrics),
            independent_input_files_verified=len(inputs), pending_groups=pending,
            all_archived_stdout_reports_match=True, complete_summaries_recomputed=True,
            per_record_quality_and_penalties_recomputed=True, alternating_schedule_verified=True,
            history_zero_trajectories_verified=trajectory,
            trajectory_exclusions=["timings", "backend_version", "新增且为零的krylov_history_size及四项回收计数"],
            analytic_diagnostics_verified=probes,
            derived_details_verified=details, missing_derived_detail_groups=missing_details,
            ctest={"passed": 17, "total": 17, "seconds": 10.91, "ipm_checks": 66512,
                   "recycle_vectors": 166, "recycle_projections": 10, "recycle_direct_acceptances": 1,
                   "build_log_sha256": sha(DATA / "iteration-ipm-v39-build.log"),
                   "detail_log_sha256": sha(DATA / "iteration-ipm-v39-ctest-detail.log")},
            runner_sha256=sha(DATA / "v39-run-experiments.py"),
            runtime_scope="本核验仅读取归档，不运行编译、求解或性能。质量与摘要按身份已核对的冻结Python协议重新计算。",
            compare_exit_scope="比较工具退出码由完整结果按协议推导；原始逐次求解退出码另逐条核验。",
        )
        if not args.preflight:
            result.update(verify_documents(output))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "passed", "final", "source_files_verified", "raw_records_verified", "pending_groups",
        "history_zero_trajectories_verified")}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    main()
