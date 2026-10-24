"""独立核验第45轮后端消融的来源、完整原始报告与扩展门槛；不编译或求解。"""
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
import runpy
import statistics
import sys
import types
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "160ad4060e3135a1146e9b034e0d38103288cd2c377c94e4c28f0145837fb179"
PRIOR_SHA = "e3d9f917f96fe3332b84faf8c160b1377d3ee5c1640ec3b34114c009bbe74c98"
SOLVER_SHA = "38b90b7034120b57405cfcdff5e4569cca8622a4ed2aa7a7a7eb580217e7484c"
EXE = ROOT / ".tools/solver-ipm-v48.exe"
BOOLEAN_FIELDS = {
    "singleton-elimination": "eliminate_singletons", "dynamic-regularization": "dynamic_regularization",
    "kkt-equilibration": "equilibrate_kkt", "fixed-variable-elimination": "eliminate_fixed_variables",
    "singleton-presolve": "presolve_singleton_rows", "presolve-fixed-elimination": "eliminate_presolved_fixed_variables",
    "inexact-prefilter": "inexact_prefilter",
}
SCALARS = ("ipm_homogeneous_tau", "ipm_homogeneous_kappa")
ZERO_FIELDS = ("ipm_full_newton_preconditioner_solves", "ipm_full_newton_products",
    "ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns",
    "ipm_homogeneous_scaled_products", "ipm_homogeneous_scaled_refinement_steps",
    "ipm_homogeneous_scaled_inexact_returns")
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right, message):
    require(math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-13), message)


def flags(command, start):
    require((len(command) - start) % 2 == 0, "命令参数必须成对")
    output = {}
    for index in range(start, len(command), 2):
        key = command[index]
        require(key.startswith("--") and key not in output, "重复或无效选项")
        output[key] = command[index + 1]
    return output


def tag(stage, retries=0):
    require(retries == 0, "本轮只允许零方向重试")
    return stage + "-homogeneous-normal_cholmod-retries0"


def sources():
    identity = read(DATA / "ipm-v48-identity.json")
    require(sha(DATA / "v48-plan.json") == identity["plan_sha256"] == PLAN_SHA, "测前计划身份")
    require(sha(EXE) == sha(ROOT / "build/release/solver.exe") == identity["solver_sha256"] == SOLVER_SHA, "实际程序身份")
    require(len(identity["files"]) == 92 and all(sha(ROOT / name) == digest for name, digest in identity["files"].items()),
            "92份冻结来源与当前来源")
    archive_path = DATA / "ipm-v48-source.zip"
    require(sha(archive_path) == identity["source_zip_sha256"], "冻结源码包")
    loaded = []
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) == 92 and
                {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()} == identity["files"],
                "冻结源码包逐文件身份")
        for name in ("benchmark", "compare_lp"):
            source = archive.read("scripts/" + name + ".py")
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / "scripts" / (name + ".py"))
            sys.modules[name] = module
            exec(compile(source, str(archive_path) + "/scripts/" + name + ".py", "exec"), module.__dict__)
            loaded.append(module)
    require(sha(Path(identity["compiler"])) == identity["compiler_sha256"], "实际编译器")
    for group, base in (("libraries", ROOT), ("logs", DATA)):
        require(identity[group], "空证据集合：" + group)
        for name, digest in identity[group].items():
            require(sha(base / name) == digest, "实际依赖或日志身份：" + name)
    library_paths = {(ROOT / name).resolve() for name in identity["libraries"]}
    require(len(library_paths) == 17, "17份不同的实际链接库")
    # -Wl,--out-implib,...dll.a 是输出选项；不通过文件是否存在来猜测输入。
    parsed = [value for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', identity["link_command"])
              if not (value := quoted or plain).startswith("-")]
    require({(ROOT / "build/release" / name).resolve() for name in parsed} == library_paths,
            "实际链接命令与冻结库集合一致")
    require(sum(path.name == "libopenblas.a" for path in library_paths) == 1 and
            {"libcholmod.a", "libcolamd.a", "libamd.a"}.issubset({path.name for path in library_paths}),
            "唯一OpenBLAS与CHOLMOD依赖")
    require("100% tests passed out of 17" in (DATA / "iteration-ipm-v48-build.log").read_text(encoding="utf-8"),
            "正式17项CTest全过")
    for filename, key in (("v48-run-experiments.py", "runner_sha256"), ("v48-freeze.py", "freeze_script_sha256"),
                          ("v48-runner-self-check.json", "runner_self_check_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验辅助来源：" + filename)
    return identity, loaded[0], loaded[1]


def plan_and_inputs(identity, compare):
    plan = read(DATA / "v48-plan.json")
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <= datetime.datetime.fromisoformat(identity["created_at_utc"]),
            "计划在生产冻结前")
    require((plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
             plan["max_iterations"], plan["scaling_passes"], plan["direction_retries"], plan["initial_runs"]) ==
            (5, 35, 1, 0, 200, 5, 0, 76), "共同预算及初始样本数")
    require(plan["reference"] == {"method": "homogeneous", "linear_solver": "normal_augmented", "linear_accuracy": "strict"} and
            plan["candidate"] == {"method": "homogeneous", "linear_solver": "normal_cholmod", "linear_accuracy": "strict"},
            "仅消融分解后端")
    require(set(plan["sets"]) == {"development", "probe", "netlib"}, "预声明输入组")
    for stage, count, repeats in (("development", 11, 3), ("probe", 5, 1), ("netlib", 91, 1)):
        declared = plan["sets"][stage]
        models, digest = compare.load_models(Path(declared["models_file"]))
        require(models == declared["models"] and len(models) == count and declared["repeats"] == repeats and
                digest == declared["models_file_sha256"] and all(item["hash_error"] is None for item in models),
                stage + "输入与哈希")
    return plan


def build_attempt_evidence():
    names = ("v48-freeze-attempt1.py", "v48-blas-build-attempt1.log", "v48-blas-build-attempt2.log")
    digests = {name: sha(DATA / name) for name in names}
    original = (DATA / names[0]).read_text(encoding="utf-8")
    require(original.index("assert len(paths)") < original.index("shutil.copy2("), "原冻结器的库集合断言在复制前")
    return {"files": digests, "original_guard_precedes_copy": True,
            "scope": "保留初次依赖配置失败、随后依赖构建及原冻结器来源；源码顺序不代替实际构建日志。"}


def counts(report, strict_hsd=False):
    exceptions = {"ipm_factor_backend", "ipm_primal_regularization", "ipm_dual_regularization", *SCALARS}
    if report["backend"] != "ipm":
        require(all(value is None for key, value in report.items() if key.startswith("ipm_")), "非IPM统计必须为null")
        return
    require(all(type(value) is int and 0 <= value < 2**63 for key, value in report.items()
                if key.startswith("ipm_") and key not in exceptions), "所有工作计数为非负int64")
    require(report["ipm_weighted_krylov_steps"] <= report["ipm_refinement_steps"], "Krylov是改进子集")
    if not strict_hsd:
        return
    iterations, constant, directions = (report[key] for key in
        ("ipm_iterations", "ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves"))
    require(iterations <= constant <= iterations + 1 and 2 * iterations <= directions <= 2 * constant, "HSD实际工作与迭代关系")
    require(all(report.get(key) == 0 and type(report.get(key)) is int for key in ZERO_FIELDS), "strict HSD无其他方向政策工作")
    require(all(key in report for key in SCALARS), "齐次标量存在")
    if all(report[key] is None for key in SCALARS):
        require(iterations == constant == directions == 0, "未初始化不能有齐次工作")
    else:
        require(all(type(report[key]) in (int, float) and math.isfinite(report[key]) and report[key] > 0 for key in SCALARS),
                "齐次标量同时有限严格正")
    if report["ipm_factorizations"] > 0:
        requested = report["options"]["ipm"]["linear_solver"]
        expected = "factorhighs" if report["ipm_structural_fallbacks"] else ("normal_cholmod" if requested == "normal_cholmod" else "normal")
        require(report["ipm_factor_backend"] == expected, "实际后端与显式结构切换一致")


def trajectories(identity, benchmark):
    path = DATA / "v48-old-mode-trajectory-runs.json.gz"
    records = json.loads(gzip.decompress(path.read_bytes()))
    models = read(DATA / "v48-plan.json")["sets"]["development"]["models"][:8]
    configurations = (("qdldl", "strict", 0, "off", "mehrotra"), ("normal_augmented", "strict", 0, "off", "mehrotra"),
        ("normal_augmented", "blockwise", 8, "on", "mehrotra"), ("normal_augmented", "full_newton", 0, "off", "mehrotra"),
        ("normal_augmented", "bounded_newton", 0, "off", "mehrotra"), ("normal_augmented", "strict", 0, "off", "homogeneous"),
        ("normal_multifrontal", "strict", 0, "off", "homogeneous"))
    require(len(records) == 56 and sha(ROOT / ".tools/solver-ipm-v46.exe") == PRIOR_SHA, "56旧对子与旧程序")
    limited_pairs = []
    for index, row in enumerate(records):
        model = models[index % 8]
        linear, accuracy, retries, presolve, method = configurations[index // 8]
        require(row["model"] == model["input"] and row["model_sha256"] == model["sha256"] and
                all(len(row[key]) == 2 for key in ("commands", "reports", "returncodes", "hashes", "outputs")), "旧对子完整性")
        for side, version, digest in ((0, 46, PRIOR_SHA), (1, 48, identity["solver_sha256"])):
            command, report = row["commands"][side], row["reports"][side]
            require(command[:3] == [str(ROOT / f".tools/solver-ipm-v{version}.exe"), "solve", model["input"]], "旧轨迹实际程序")
            actual = flags(command, 3)
            expected = {"--backend": "ipm", "--time-limit": "5", "--threads": "1", "--seed": "0",
                "--ipm-linear-solver": linear, "--ipm-method": method, "--ipm-linear-accuracy": accuracy,
                "--ipm-direction-retries": str(retries), "--ipm-krylov-history-size": "0", "--ipm-kkt-equilibration": "on",
                "--ipm-singleton-elimination": "off" if linear == "qdldl" else "on",
                "--ipm-fixed-variable-elimination": "off" if linear == "qdldl" else "on",
                "--ipm-singleton-presolve": presolve, "--ipm-presolve-fixed-elimination": presolve}
            require(actual == expected, "旧轨迹预声明配置")
            require(row["hashes"][side] == {key: {"solver": digest, "input": model["sha256"]} for key in ("before", "after")}, "旧轨迹前后身份")
            require(benchmark.strict_json(row["outputs"][side]["stdout"]) == report and
                    report["options"]["ipm"]["method"] == method and report["backend_version"] == f"ipm-cpu-0.{40 if version == 46 else 41}.0",
                    "旧轨迹真实报告")
            require(benchmark.classify(row["returncodes"][side], report) in ("resolved", "incomplete", "verification_failed"), "旧轨迹有效报告/退出码")
            counts(report)
        clean = [{key: value for key, value in report.items() if key not in ("timings", "backend_version")} for report in row["reports"]]
        equal = clean[0] == clean[1] and row["returncodes"][0] == row["returncodes"][1]
        limited = any(report.get("status") == "time_limit" or
            report["timings"]["solve_seconds"] + report["timings"]["verification_seconds"] > 5 for report in row["reports"])
        require(row["non_time_equal"] is equal and row["budget_limited"] is limited and row["method_preserved"] is True and (equal or limited),
                "旧轨迹仅排除版本和时间，预算受限差异单列")
        if limited:
            limited_pairs.append({"index": index, "model": Path(model["input"]).stem, "linear": linear,
                                  "method": method, "accuracy": accuracy, "equal": equal})
    return {"pairs": 56, "budget_limited_pairs": limited_pairs, "non_budget_pairs_equal": 56 - len(limited_pairs), "sha256": sha(path)}


def verify_group(manifest_path, identity, plan, compare, benchmark, input_hashes):
    name = manifest_path.name[len("v48-"):-len("-manifest.json")]
    choices = {tag(stage, retry): (stage, retry) for stage in plan["sets"] for retry in (0,)}
    choices["highs"] = ("highs", 0)
    require(name in choices, "非计划实验组：" + name)
    stage, retries = choices[name]
    declared = plan["sets"]["netlib" if stage == "highs" else stage]
    original = ROOT / "results" / ("round-ipm-v48-" + name)
    summary_path, raw_path = (DATA / ("v48-" + name + suffix) for suffix in ("-summary.json", "-runs.jsonl.gz"))
    manifest, summary = read(manifest_path), read(summary_path)
    require(datetime.datetime.fromisoformat(identity["created_at_utc"]) <=
            datetime.datetime.fromisoformat(manifest["started_at_utc"]), name + " 测量发生于冻结之后")
    raw = gzip.decompress(raw_path.read_bytes())
    require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), name + " 清单字节")
    require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), name + " 汇总字节")
    require(raw == (original / "runs.jsonl").read_bytes(), name + " 原始记录字节")
    require(read(DATA / ("v48-" + name + "-run.log")) == summary, name + " 比较进程原始输出")
    process = read(DATA / ("v48-" + name + "-process.json"))
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
    command_path = DATA / ("v48-" + name + "-command.json")
    command = read(command_path)
    top = flags(command, 2)
    require(Path(command[1]) == ROOT / "scripts/compare_lp.py" and
            Path(top["--solver"]) == ROOT / ".tools/solver-ipm-v48.exe" and
            Path(top["--output-dir"]) == original and top["--models-file"] == declared["models_file"], name + " 比较入口")
    for key in ("time_limit", "wall_timeout", "threads", "seed"):
        require(float(top["--" + key.replace("_", "-")]) == plan[key], name + " 比较资源")
    require(int(top.get("--repeats", "1")) == declared["repeats"], name + " 比较重复")
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
        require(config["ipm_linear_solver"] == ("normal_cholmod" if side else "normal_augmented"), name + " 后端消融配置")
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
        for key in ("method", "linear_solver", "linear_accuracy", "direction_retries"):
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
        require(Path(solve_command[0]) == ROOT / ".tools/solver-ipm-v48.exe" and
                solve_command[1:3] == ["solve", row["input"]], name + " 求解入口")
        actual = flags(solve_command, 3)
        config = configs[variant]
        require(actual["--backend"] == config["backend"] and float(actual["--time-limit"]) == 5 and
                actual["--threads"] == "1" and actual["--seed"] == "0", name + " 求解共同预算")
        if config["backend"] == "ipm":
            require(actual["--iteration-limit"] == "200" and actual["--scaling-passes"] == "5", name + " IPM迭代预算")
            for key in (*plan["shared_ipm"], "method", "linear-solver", "linear-accuracy", "direction-retries"):
                require(actual["--ipm-" + key] == str(config["ipm_" + key.replace("-", "_")]), name + " 求解配置：" + key)
        else:
            for key in ("solver", "presolve", "crossover"):
                require(actual["--highs-" + key] == config["highs_" + key], name + " HiGHS实际命令")
            require(not any(key.startswith("--ipm-") for key in actual), name + " HiGHS不带自主参数")
        report = row["report"]
        if report is not None:
            require(benchmark.strict_json(row["stdout"]) == report, name + " stdout与报告")
            require(report["backend"] == config["backend"], name + " 报告后端")
            require(report["backend_version"] == ("ipm-cpu-0.41.0" if config["backend"] == "ipm" else "1.15.1"), name + " 报告版本")
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
            counts(report, strict_hsd=True)
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
    return {"tag": name, "stage": stage, "raw_records": len(rows),
            "quality_pass_count": summary["quality_pass_count"], "process_returncode": process["returncode"],
            "manifest_sha256": sha(manifest_path), "summary_sha256": sha(summary_path), "raw_sha256": sha(raw_path),
            "missing_reports": sum(row["report"] is None for row in rows),
            "status_counts": dict(collections.Counter(row["status"] for row in rows)),
            "actual_factor_backends": dict(collections.Counter((row["report"] or {}).get("ipm_factor_backend") for row in rows))}, (manifest, summary, rows)


def decision(development, probe):
    reference, candidate = development["reference_variant"], development["candidate_variant"]
    lost = [Path(model["input"]).stem for model in development["models"]
            if model["quality_pass_counts"][candidate] < model["quality_pass_counts"][reference]]
    before, after = (development["variants"][side] for side in (reference, candidate))
    faster = (after["penalized_numerical_sgm10_seconds"] < before["penalized_numerical_sgm10_seconds"] and
              after["penalized_wall_sgm10_seconds"] <= before["penalized_wall_sgm10_seconds"])
    reference, candidate = probe["reference_variant"], probe["candidate_variant"]
    gain = [Path(model["input"]).stem for model in probe["models"]
            if model["quality_pass_counts"][candidate] > model["quality_pass_counts"][reference]]
    probe_lost = [Path(model["input"]).stem for model in probe["models"]
                  if model["quality_pass_counts"][candidate] < model["quality_pass_counts"][reference]]
    return {"retries": 0, "development_lost": lost, "development_time_improved": faster,
            "probe_gain": gain, "probe_lost": probe_lost, "eligible_for_netlib": not lost and (faster or (bool(gain) and not probe_lost)),
            "plan_sha256": PLAN_SHA}


def derived(groups, final):
    require({tag("development"), tag("probe")}.issubset(groups), "初始开发和探针不能遗漏")
    gate = decision(groups[tag("development")][1], groups[tag("probe")][1])
    require(read(DATA / "v48-retries0-gate.json") == gate, "独立重建扩展门槛")
    require(tag("netlib") not in groups or gate["eligible_for_netlib"], "门槛未过不得扩大91")
    require("highs" not in groups or (gate["eligible_for_netlib"] and tag("netlib") in groups), "HiGHS比较前完成自主完整集")
    require(not final or not gate["eligible_for_netlib"] or {tag("netlib"), "highs"}.issubset(groups), "最终核验不能遗漏通过门槛后的计划组")
    require(sum(len(groups[tag(stage)][2]) for stage in ("development", "probe")) == 76, "初始76记录完整")
    verified = []
    for stage, prefix in (("development", "development"), ("probe", "probe"), ("coverage", "netlib")):
        path = DATA / f"v48-{stage}-detail.json"
        if not final and not path.exists():
            continue
        expected = {}
        for name, (manifest, summary, rows) in groups.items():
            if not name.startswith(prefix + "-"):
                continue
            reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
            passed = {side: {Path(model["input"]).stem for model in summary["models"]
                            if model["quality_pass_counts"][side] == manifest["repeats"]} for side in (reference, candidate)}
            entries = []
            for row in rows:
                report = row["report"] or {}
                entries.append({"model": Path(row["input"]).stem, "candidate": row["requested_variant"] == candidate,
                    "repetition": row["repetition"], "quality_pass": row["quality_pass"], "status": row["status"],
                    "quality_failure": row["quality_failure"], "numerical_seconds": row["numerical_seconds"], "message": report.get("message"),
                    "work": {key: value for key, value in report.items() if key.startswith("ipm_")}, "optimality": report.get("lp_optimality")})
            expected[name] = {"summary": summary["variants"], "gain": sorted(passed[candidate] - passed[reference]),
                              "lost": sorted(passed[reference] - passed[candidate]), "models": entries}
        require(read(path) == expected, "派生明细全部重建：" + stage)
        verified.append({"path": path.name, "sha256": sha(path)})
    return {"gate": gate, "details": verified, "expanded_netlib": tag("netlib") in groups, "direct_highs": "highs" in groups}


def cost_audit(groups, plan):
    def paired(group, allowed=None):
        manifest, _, rows = group
        sides = (manifest["reference_variant"], manifest["candidate_variant"])
        repeated = collections.defaultdict(lambda: {side: [] for side in sides})
        for row in rows:
            if allowed is None or Path(row["input"]).stem in allowed:
                repeated[row["input"]][row["requested_variant"]].append(row)
        successful = {name: by_side for name, by_side in repeated.items()
                      if all(len(by_side[side]) == manifest["repeats"] and all(row["quality_pass"] for row in by_side[side]) for side in sides)}
        if not successful:
            return {"models": 0, "reference": sides[0], "candidate": sides[1]}
        aggregate, details = {}, []
        for side in sides:
            selected = [row for by_side in successful.values() for row in by_side[side]]
            values = {"numerical": [row["numerical_seconds"] for row in selected], "wall": [row["elapsed_wall_seconds"] for row in selected]}
            aggregate[side] = {"runs": len(selected), **{measure + "_sum_seconds": sum(times) for measure, times in values.items()},
                **{measure + "_sgm10_seconds": math.expm1(sum(math.log1p(value / 10.) for value in times) / len(times)) * 10.
                   for measure, times in values.items()}}
        for name, by_side in successful.items():
            entry = {"model": Path(name).stem, "model_dimensions": by_side[sides[0]][0]["report"]["model"], "sides": {}}
            for side in sides:
                selected = by_side[side]
                entry["sides"][side] = {"numerical_median_seconds": statistics.median(row["numerical_seconds"] for row in selected),
                    "wall_median_seconds": statistics.median(row["elapsed_wall_seconds"] for row in selected),
                    "iterations": [row["report"]["ipm_iterations"] for row in selected],
                    "factor_nonzeros": [row["report"]["ipm_factor_nonzeros"] for row in selected],
                    "factor_dimension": [row["report"]["ipm_kkt_dimension"] for row in selected]}
            entry["numerical_median_seconds_saved"] = entry["sides"][sides[0]]["numerical_median_seconds"] - entry["sides"][sides[1]]["numerical_median_seconds"]
            details.append(entry)
        return {"models": len(successful), "reference": sides[0], "candidate": sides[1], "sides": aggregate,
                "candidate_faster_numerical_models": sum(item["numerical_median_seconds_saved"] > 0 for item in details),
                "candidate_faster_wall_models": sum(item["sides"][sides[0]]["wall_median_seconds"] > item["sides"][sides[1]]["wall_median_seconds"] for item in details),
                "details_by_numerical_saving": sorted(details, key=lambda item: item["numerical_median_seconds_saved"], reverse=True)}
    development = plan["sets"]["development"]["models"]
    audit = {"development_original_eight": paired(groups[tag("development")], {Path(model["input"]).stem for model in development[:8]}),
             "development_added_three": paired(groups[tag("development")], {Path(model["input"]).stem for model in development[8:]}),
             "failure_records": [],
             "limitations": "仅共同全部重复达标模型用于成功耗时比较，不把早退当加速；分组来自测前11开发清单，不增加事后规模阈值。耗时包括完整数值流程，外层步数可变化，不能全归因于单次分解。CHOLMOD计数为含占位xsize，旧计数为有效存储，不用于直接计算填充倍率。"}
    for name in (tag("netlib"), "highs"):
        if name in groups:
            audit[name + "_common_success"] = paired(groups[name])
    for name, (manifest, _, rows) in groups.items():
        for row in rows:
            model = Path(row["input"]).stem
            if model not in ("forplan", "dfl001", "greenbea"):
                continue
            report = row["report"] or {}
            stage = None
            iterations = report.get("ipm_iterations")
            if (report.get("message") == "正则化正规方程 Cholesky 分解重试未成功" and iterations == 0 and
                report.get("ipm_homogeneous_constant_solves") == report.get("ipm_homogeneous_direction_solves") == 0 and
                report.get("ipm_homogeneous_tau") == report.get("ipm_homogeneous_kappa") == 1):
                stage = "初始化已完成；首次齐次迭代的正规方程分解重试耗尽，尚未启动常数方向"
            if (report.get("message") == "齐次标量消元的分母不是有限正数" and type(iterations) is int and
                report.get("ipm_homogeneous_constant_solves") == iterations + 1 and
                report.get("ipm_homogeneous_direction_solves") == 2 * iterations + 1):
                stage = "下一轮仿射标量消元失败；按源码调用顺序，该轮常数方向与u方向已通过原严格KKT验收"
            audit["failure_records"].append({"group": name, "model": model,
                "candidate": row["requested_variant"] == manifest["candidate_variant"], "status": row["status"],
                "quality_pass": row["quality_pass"], "quality_failure": row["quality_failure"], "message": report.get("message"),
                "numerical_seconds": row["numerical_seconds"], "wall_seconds": row["elapsed_wall_seconds"],
                "has_solution": report.get("has_solution"), "lp_optimality": report.get("lp_optimality"),
                "source_based_stage_inference": stage,
                "work": {key: value for key, value in report.items() if key.startswith("ipm_")}})
    audit["failure_stage_limits"] = "阶段由冻结源码调用顺序、消息和工作计数共同推断。归档未记录CHOLMOD status/minor或每次factor=false的具体分支，不能把分解路径失败进一步统称为POTRF失败；严格KKT验收也不证明后续标量Schur补稳定。"
    return audit


def self_check():
    report = {"backend": "ipm", "options": {"ipm": {"linear_solver": "normal_cholmod"}},
        "ipm_iterations": 1, "ipm_homogeneous_constant_solves": 1, "ipm_homogeneous_direction_solves": 2,
        "ipm_homogeneous_tau": 1., "ipm_homogeneous_kappa": 1., "ipm_factorizations": 2,
        "ipm_factor_backend": "normal_cholmod", "ipm_structural_fallbacks": 0,
        "ipm_factor_nonzeros": 4, "ipm_refinement_steps": 1, "ipm_weighted_krylov_steps": 1,
        **dict.fromkeys(ZERO_FIELDS, 0)}
    counts(report, True)
    rejected = 0
    mutations = [({"ipm_factor_nonzeros": value}) for value in (-1, True, 1.5, 2**63)] + [
        {"ipm_homogeneous_constant_solves": 3}, {"ipm_homogeneous_direction_solves": 3},
        {"ipm_homogeneous_direction_solves": 1}, {"ipm_homogeneous_tau": 0},
        {"ipm_homogeneous_kappa": float("inf")}, {"ipm_homogeneous_tau": None},
        {"ipm_homogeneous_scaled_products": 1}, {"ipm_factor_backend": "normal"},
        {"ipm_structural_fallbacks": 1}, {"ipm_weighted_krylov_steps": 2}]
    for mutation in mutations:
        try:
            counts({**report, **mutation}, True)
        except (AssertionError, KeyError):
            rejected += 1
    require(rejected == len(mutations), "计数和后端反例必须拒绝")
    for command in (["--x", "1", "--x", "2"], ["x", "1"], ["--x"]):
        try:
            flags(command, 0)
        except AssertionError:
            rejected += 1
    require(rejected == len(mutations) + 3, "非法命令反例必须拒绝")
    counts({**report, "ipm_factor_backend": "factorhighs", "ipm_structural_fallbacks": 1}, True)
    def summary(before=3, after=3, numerical=(1., 1.), wall=(1., 1.)):
        return {"reference_variant": "r", "candidate_variant": "c", "models": [
                {"input": "synthetic.mps", "quality_pass_counts": {"r": before, "c": after}}],
                "variants": {side: {"penalized_numerical_sgm10_seconds": numerical[index],
                                     "penalized_wall_sgm10_seconds": wall[index]} for index, side in enumerate(("r", "c"))}}
    cases = [(summary(numerical=(1, .9)), summary(), True), (summary(), summary(), False),
        (summary(3, 2, numerical=(1, .5), wall=(1, .5)), summary(0, 1), False),
        (summary(numerical=(1, .9), wall=(1, 1.1)), summary(), False),
        (summary(numerical=(1, 2)), summary(0, 1), True), (summary(), summary(1, 0), False)]
    mixed = summary(0, 1)
    mixed["models"].append({"input": "lost.mps", "quality_pass_counts": {"r": 1, "c": 0}})
    cases.append((summary(), mixed, False))
    for development, probe, expected in cases:
        require(decision(development, probe)["eligible_for_netlib"] is expected, "扩展门槛反例")
    return {"passed": True, "invalid_cases_rejected": rejected, "gate_cases": len(cases), "valid_counter_cases": 2, "solver_runs": 0}


def documents(output):
    paths = list(ROOT.glob("*.md"))
    for directory in ("knowledge", "changelog", "benchmarks"):
        paths.extend((ROOT / directory).rglob("*.md"))
    links, own, broken = 0, 0, []
    for path in paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            links += 1
            local = (path.parent / unquote(target.split("#")[0].strip("<>"))).resolve()
            own += local == output.resolve()
            if not local.exists():
                broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    require(not broken and own >= 2, "本地断链或缺少最终报告链接：" + str(broken))
    return {"markdown_files_checked": len(paths), "local_links_checked": links, "verification_report_links": own, "broken_links": broken}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_check(), ensure_ascii=False))
        return
    output = DATA / ("v48-verification-preflight.json" if args.preflight else "v48-verification.json")
    result = {"passed": False, "final": not args.preflight, "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "script_sha256": sha(Path(__file__)), "scope": "仅重核既有来源和数据，不编译、不求解、不重新测量。"}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        result["self_test"] = self_check()
        identity, benchmark, compare = sources()
        plan = plan_and_inputs(identity, compare)
        runner = runpy.run_path(str(DATA / "v48-run-experiments.py"), run_name="v48_runner_for_readonly_self_check")
        require(runner["self_check"]() == read(DATA / "v48-runner-self-check.json"), "冻结运行器纯内存自检")
        result.update(solver_sha256=identity["solver_sha256"], source_files_verified=len(identity["files"]),
                      libraries_verified=len(identity["libraries"]), source_zip_sha256=identity["source_zip_sha256"], plan_sha256=PLAN_SHA,
                      build_attempt_evidence=build_attempt_evidence(),
                      trajectories=trajectories(identity, benchmark))
        groups, reports, input_hashes = {}, [], {}
        for path in sorted(DATA.glob("v48-*-manifest.json")):
            report, raw = verify_group(path, identity, plan, compare, benchmark, input_hashes)
            reports.append(report)
            groups[report["tag"]] = raw
        result.update(groups=reports, raw_records=sum(group["raw_records"] for group in reports),
                      derived=derived(groups, not args.preflight), cost_audit=cost_audit(groups, plan))
        candidates = [row["report"] for manifest, _, rows in groups.values() for row in rows
                      if row["requested_variant"] == manifest["candidate_variant"] and row["report"] is not None]
        actual = [report for report in candidates if report["ipm_factor_backend"] == "normal_cholmod" and report["ipm_factorizations"] > 0]
        require(actual, "真实候选必须实际进入CHOLMOD，不能全部使用结构回退")
        result["candidate_backend_usage"] = {"reports": len(candidates), "actual_cholmod_reports": len(actual),
            "factor_attempts_in_cholmod_reports": sum(report["ipm_factorizations"] for report in actual),
            "structural_fallback_reports": sum(report["ipm_factor_backend"] == "factorhighs" and report["ipm_structural_fallbacks"] > 0 for report in candidates),
            "largest_reported_xsize": max(report["ipm_factor_nonzeros"] for report in actual)}
        if not args.preflight:
            result.update(documents(output))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "final": result["final"], "groups": len(result["groups"]), "raw_records": result["raw_records"]}))


if __name__ == "__main__":
    main()
