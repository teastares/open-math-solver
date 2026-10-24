"""独立核验有限齐次常数方向恢复的来源、轨迹、工作账本与扩集门槛。"""

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
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "477a5efec5cdfd73f8330ff3a114179065a3e287014160149a46c7cd9c896ea9"
PARENT_IDENTITY_SHA = "988d97e2c29b6d3d6edb918e4bdcac0e05e51e380cf939023b71457b6e7e65bd"
PARENT_SOLVER_SHA = "38b90b7034120b57405cfcdff5e4569cca8622a4ed2aa7a7a7eb580217e7484c"
NEW_FIELDS = ("ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes")
SCALARS = ("ipm_homogeneous_tau", "ipm_homogeneous_kappa")
ZERO_FIELDS = ("ipm_full_newton_preconditioner_solves", "ipm_full_newton_products",
    "ipm_bounded_newton_returns", "ipm_bounded_newton_inexact_returns",
    "ipm_homogeneous_scaled_products", "ipm_homogeneous_scaled_refinement_steps", "ipm_homogeneous_scaled_inexact_returns")
BOOLEAN_FIELDS = {"singleton-elimination": "eliminate_singletons", "dynamic-regularization": "dynamic_regularization",
    "kkt-equilibration": "equilibrate_kkt", "fixed-variable-elimination": "eliminate_fixed_variables",
    "singleton-presolve": "presolve_singleton_rows", "presolve-fixed-elimination": "eliminate_presolved_fixed_variables",
    "inexact-prefilter": "inexact_prefilter"}
STAGES = ("development", "probe", "secondary-probe", "netlib", "highs")
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strict_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, "重复 JSON 键：" + key)
            result[key] = item
        return result
    def invalid(value):
        raise AssertionError("非有限 JSON 字面量：" + value)
    def decimal(value):
        result = float(value)
        require(math.isfinite(result), "JSON 小数超出有限范围")
        return result
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid, parse_float=decimal)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value, label):
    require(type(value) is int and 0 <= value < 2**63, "非负 int64 无效：" + label)
    return value


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def close(left, right, label):
    require(finite(left) and finite(right) and math.isclose(left, right, rel_tol=1e-13, abs_tol=1e-13), label)


def flags(command, offset):
    require(isinstance(command, list) and len(command) >= offset and (len(command)-offset) % 2 == 0, "命令参数成对")
    result = {}
    for key, value in zip(command[offset::2], command[offset+1::2]):
        require(type(key) is str and key.startswith("--") and key not in result and type(value) is str, "重复或非法命令选项")
        result[key] = value
    return result


def frozen_plan(check_inputs=True):
    require(sha(DATA / "v50-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v50-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA and
            sha(DATA / "ipm-v48-identity.json") == PARENT_IDENTITY_SHA, "父来源身份")
    require(sha(DATA / "v49-verification.json") == plan["evidence_sha256"], "第46轮独立证据身份")
    require((plan["primary_linear_solver"], plan["secondary_linear_solver"], plan["reference_accuracy"],
             plan["candidate_accuracy"], plan["method"]) ==
            ("normal_cholmod", "normal_augmented", "strict", "homogeneous_repair", "homogeneous"), "单机制消融")
    require((plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            (5, 35, 1, 0, 200, 5), "同预算配置")
    require(plan["repair_contract"]["maximum_recovery_solves_per_iteration"] == 1 and
            plan["repair_contract"]["recovery_accuracy"] == "blockwise" and
            plan["repair_contract"]["counts"] == list(NEW_FIELDS) and plan["repair_contract"]["defaults_unchanged"] is True,
            "恢复次数、策略和计数合同")
    require(plan["shared_ipm"] == {"singleton-elimination": "on", "dynamic-regularization": "off", "kkt-equilibration": "on",
        "fixed-variable-elimination": "on", "singleton-presolve": "off", "presolve-fixed-elimination": "off",
        "krylov-history-size": "0", "central-corrections": "0", "inexact-prefilter": "off"}, "共同 IPM 选项")
    require(plan["initial_stages"] == [{"name": "development", "set": "development", "linear_solver": "normal_cholmod"},
        {"name": "probe", "set": "probe", "linear_solver": "normal_cholmod"},
        {"name": "secondary-probe", "set": "probe", "linear_solver": "normal_augmented"}] and
        plan["full_stages"] == ["netlib", "highs"], "固定初始组和扩集")
    for name, count, repeats in (("development", 11, 3), ("probe", 6, 3), ("netlib", 91, 1)):
        group = plan["sets"][name]
        require(group["repeats"] == repeats and len(group["models"]) == count, "固定输入规模：" + name)
        require([item["model_index"] for item in group["models"]] == list(range(count)) and
                len({item["input"] for item in group["models"]}) == count, "输入唯一且顺序固定")
        path = Path(group["models_file"])
        require(sha(path) == group["models_file_sha256"], "模型列表身份")
        actual = [str((path.parent / line.strip()).resolve()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(actual == [item["input"] for item in group["models"]], "模型列表路径")
        if check_inputs:
            for item in group["models"]:
                require(item["hash_error"] is None and sha(item["input"]) == item["sha256"], "输入身份：" + item["input"])
    old = plan["old_trajectory"]
    require(old["models"] == plan["sets"]["development"]["models"][:8] and old["pairs"] == 32 and
            old["configurations"] == [["qdldl", "strict", "mehrotra"], ["normal_augmented", "strict", "homogeneous"],
              ["normal_cholmod", "strict", "homogeneous"], ["normal_augmented", "homogeneous_scaled", "homogeneous"]] and
            old["excluded_fields"] == ["timings", "backend_version", *NEW_FIELDS] and old["time_limited_pairs_separate"] is True,
            "旧轨迹集合及唯一可排除字段")
    require(plan["highs_reference"] == {"solver": "choose", "presolve": "choose", "crossover": "on"}, "HiGHS 参考分支")
    return plan


def counts(report, legacy=False):
    backend = report["backend"]
    if backend != "ipm":
        require(all(value is None for key, value in report.items() if key.startswith("ipm_")), "非 IPM 统计必须为 null")
        if not legacy:
            require(all(key in report for key in NEW_FIELDS), "当前非 IPM 缺少新 null 统计")
        return
    special = {"ipm_factor_backend", "ipm_primal_regularization", "ipm_dual_regularization", *SCALARS}
    for key, value in report.items():
        if key.startswith("ipm_") and key not in special:
            integer(value, key)
    options = report["options"]["ipm"]
    mode = options["linear_accuracy"]
    if not legacy or mode == "homogeneous_repair":
        require(all(key in report for key in NEW_FIELDS), "当前报告必须包含恢复统计")
    attempts, successes = (report.get(key, 0) for key in NEW_FIELDS)
    require(successes <= attempts <= report["ipm_homogeneous_constant_solves"], "恢复成功/启动/常数求解关系")
    if mode != "homogeneous_repair":
        require(attempts == successes == 0, "其他线性政策不能包含恢复工作")
    else:
        require(options["method"] == "homogeneous" and options["direction_retries"] == options["central_corrections"] ==
                options["krylov_history_size"] == 0 and options["inexact_prefilter"] is False, "恢复模式适用范围")
    require(report["ipm_weighted_krylov_steps"] <= report["ipm_refinement_steps"], "Krylov 工作是精化子集")
    if options["method"] != "homogeneous":
        require(all(report[key] is None for key in SCALARS), "非齐次模式不得有齐次标量")
        return
    iterations, constants, directions = (report[key] for key in
        ("ipm_iterations", "ipm_homogeneous_constant_solves", "ipm_homogeneous_direction_solves"))
    if mode in ("strict", "homogeneous_repair"):
        base = constants-attempts
        require(iterations <= base <= iterations+1 and 2*iterations <= directions <= 2*base and attempts <= base,
                "每轮一次基础常数方向、最多一次恢复、两次方向右端")
        require(all(report.get(key) == 0 and type(report.get(key)) is int for key in ZERO_FIELDS), "strict/repair 不得混入其他方向政策")
    if all(report[key] is None for key in SCALARS):
        require(iterations == constants == directions == attempts == successes == 0, "未初始化不得记录齐次工作")
    else:
        require(all(finite(report[key]) and report[key] > 0 for key in SCALARS), "齐次标量成对有限正数")


def decision(development_rows, development_sides, probe_rows, probe_sides):
    """直接从重新验过质量的原始记录计数，不调用 runner 的门槛函数。"""
    def differences(rows, sides):
        grouped = collections.defaultdict(lambda: [0, 0])
        for row in rows:
            require(row["requested_variant"] in sides, "门槛出现未知侧")
            grouped[row["input"]][sides.index(row["requested_variant"]) ] += int(row["quality_pass"])
        lost = [Path(path).stem for path, (before, after) in grouped.items() if after < before]
        gain = [Path(path).stem for path, (before, after) in grouped.items() if after > before]
        return lost, gain
    development_lost, _ = differences(development_rows, development_sides)
    probe_lost, probe_gain = differences(probe_rows, probe_sides)
    repairs = sum((row["report"] or {}).get(NEW_FIELDS[1], 0) for rows, sides in
        ((development_rows, development_sides), (probe_rows, probe_sides)) for row in rows if row["requested_variant"] == sides[1])
    return {"development_lost": development_lost, "probe_lost": probe_lost, "probe_gain": probe_gain,
            "actual_repair_successes": repairs, "eligible_for_netlib": not development_lost and not probe_lost and bool(probe_gain) and repairs > 0,
            "plan_sha256": PLAN_SHA}


def sources(plan):
    identity = read(DATA / "ipm-v50-identity.json")
    require(identity["plan_sha256"] == PLAN_SHA and
            datetime.datetime.fromisoformat(plan["created_at_utc"]) <= datetime.datetime.fromisoformat(identity["created_at_utc"]), "冻结时间和计划身份")
    solver = ROOT / ".tools/solver-ipm-v50.exe"
    require(sha(solver) == sha(ROOT / "build/release/solver.exe") == identity["solver_sha256"], "当前与冻结程序身份")
    require(sha(ROOT / ".tools/solver-ipm-v48.exe") == PARENT_SOLVER_SHA, "旧程序身份")
    for name, expected in identity["files"].items():
        require(sha(ROOT/name) == expected, "当前来源改变：" + name)
    archive = DATA / "ipm-v50-source.zip"
    require(sha(archive) == identity["source_zip_sha256"], "当前冻结源码包身份")
    loaded = []
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == len(set(bundle.namelist())) and set(bundle.namelist()) == set(identity["files"]), "来源包集合")
        for name, expected in identity["files"].items():
            require(hashlib.sha256(bundle.read(name)).hexdigest() == expected, "冻结源码成员身份：" + name)
        for name in ("benchmark", "compare_lp"):
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / "scripts" / (name+".py"))
            sys.modules[name] = module
            exec(compile(bundle.read("scripts/"+name+".py"), str(archive)+"/"+name, "exec"), module.__dict__)
            loaded.append(module)
    require(sha(identity["compiler"]) == identity["compiler_sha256"], "编译器身份")
    for group, base in (("libraries", ROOT), ("logs", DATA)):
        require(identity[group], "身份集合为空："+group)
        for name, expected in identity[group].items():
            require(sha(base/name) == expected, "依赖或日志身份："+name)
    libraries = {(ROOT/name).resolve() for name in identity["libraries"]}
    require(len(libraries) == 17, "实际链接库数量")
    parsed = [value for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', identity["link_command"])
              if not (value := quoted or plain).startswith("-")]
    require({(ROOT/"build/release"/name).resolve() for name in parsed} == libraries, "链接命令对应同17个库")
    require(sum(path.name == "libopenblas.a" for path in libraries) == 1 and
            {"libcholmod.a", "libamd.a", "libcolamd.a"} <= {path.name for path in libraries}, "唯一 BLAS 与 CHOLMOD 库")
    require(len(identity["files"]) == 92, "冻结源码集合必须为92份")
    for filename, key in (("v50-run.py", "runner_sha256"), ("v50-freeze.py", "freeze_script_sha256"),
                          ("v50-runner-self-check.json", "runner_self_check_sha256")):
        require(sha(DATA/filename) == identity[key], "冻结实验工具："+filename)
    check = read(DATA/"v50-runner-self-check.json")
    require(check == {"passed": True, "command_cases": 5, "gate_cases": 6, "solver_runs": 0}, "测前 runner 合成自检")
    require("100% tests passed out of 17" in (DATA/"iteration-ipm-v50-build.log").read_text(encoding="utf-8"), "正式17项CTest通过")
    # Windows CTest 的日期头可能采用本地编码；测试程序的 UTF-8 输出按原始字节核对。
    require("齐次恢复解析尝试 2 次，成功 2 次".encode("utf-8") in
            (DATA/"iteration-ipm-v50-ctest-detail.log").read_bytes(), "恢复解析边界测试实际执行")
    return identity, loaded[0], loaded[1]


def spec(stage, plan):
    return (plan["sets"]["development" if stage == "development" else "netlib" if stage in ("netlib", "highs") else "probe"],
            "normal_augmented" if stage == "secondary-probe" else "normal_cholmod")


def expected_ipm_flags(plan, linear, accuracy, prefix="--"):
    return {prefix+"iteration-limit": "200", prefix+"scaling-passes": "5", prefix+"ipm-method": "homogeneous",
        prefix+"ipm-linear-solver": linear, prefix+"ipm-linear-accuracy": accuracy, prefix+"ipm-direction-retries": "0",
        **{prefix+"ipm-"+key: value for key, value in plan["shared_ipm"].items()}}


def verify_options(report, plan, linear, accuracy, backend):
    require(report["backend"] == backend, "真实报告后端")
    options = report["options"]
    require((options["time_limit"], options["threads"], options["seed"]) == (5, 1, 0), "真实报告共同资源")
    require(options["backend"] == backend and options["mip_relative_gap"] == 1e-4 and options["pdlp"] is None and
            options["tolerances"] == {"feasibility_absolute": 1e-7, "feasibility_relative": 1e-9,
                "integrality": 1e-6, "objective_absolute": 1e-7, "objective_relative": 1e-9}, "共同原模型验收容差")
    if backend == "highs":
        require(options["ipm"] is None and options["highs"] == {"solver": "choose", "presolve": "choose", "run_crossover": "on"}, "HiGHS choose 分支")
        return
    expected = {"method": "homogeneous", "max_iterations": 200, "scaling_passes": 5, "linear_solver": linear,
        "linear_accuracy": accuracy, "direction_retries": 0, "central_corrections": 0, "krylov_history_size": 0,
        **{field: plan["shared_ipm"][flag] == "on" for flag, field in BOOLEAN_FIELDS.items()}}
    require(options["ipm"] == expected, "真实报告完整 IPM 配置")
    require(options["highs"] is None and report["backend_version"] == "ipm-cpu-0.42.0", "自主当前版本及专属选项")
    actual = report["ipm_factor_backend"]
    expected_factor = "normal" if linear == "normal_augmented" else linear
    require(actual in (None, expected_factor, "factorhighs"), "实际分解器仅允许原结构回退")
    if actual is not None:
        require((actual == "factorhighs") is (report["ipm_structural_fallbacks"] > 0), "结构回退计数与实际后端对应")


def verify_group(stage, identity, plan, benchmark, compare):
    require(stage in STAGES, "未知阶段")
    declared, linear = spec(stage, plan)
    stem = "v50-"+stage
    manifest = read(DATA/(stem+"-manifest.json"))
    summary = read(DATA/(stem+"-summary.json"))
    original = ROOT/"results"/("round-ipm-"+stem)
    raw_path = DATA/(stem+"-runs.jsonl.gz")
    with gzip.open(raw_path, "rb") as source:
        raw = source.read()
    rows = [strict_json(line) for line in raw.splitlines()]
    require(raw == (original/"runs.jsonl").read_bytes(), "归档原始字节："+stage)
    for name in ("manifest", "summary"):
        require((DATA/(stem+"-"+name+".json")).read_bytes() == (original/(name+".json")).read_bytes(), "归档清单/汇总字节")
    require(read(DATA/(stem+"-run.log")) == summary, "比较进程 stdout 汇总")
    require(datetime.datetime.fromisoformat(manifest["started_at_utc"]) >= datetime.datetime.fromisoformat(identity["created_at_utc"]), "运行必须发生于冻结之后")
    require(manifest["models"] == declared["models"] and manifest["models_file"] == declared["models_file"] and
            manifest["models_file_sha256"] == declared["models_file_sha256"] and manifest["repeats"] == declared["repeats"], "冻结输入与重复")
    require(all(manifest[key] == plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")), "清单共同预算")
    require(manifest["comparison_script_sha256"] == identity["files"]["scripts/compare_lp.py"] and
            manifest["benchmark_script_sha256"] == identity["files"]["scripts/benchmark.py"], "报告协议身份")
    solver = str(ROOT/".tools/solver-ipm-v50.exe")
    require(manifest["solver"] == solver and manifest["solver_sha256"] == identity["solver_sha256"] and manifest["solver_hash_error"] is None,
            "清单程序身份")
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    require(len(set(sides)) == 2 and set(manifest["variants"]) == set(sides) and set(manifest["solvers"]) == set(sides), "独立双侧标签")
    command = read(DATA/(stem+"-command.json"))
    require(Path(command[1]) == ROOT/"scripts/compare_lp.py", "比较入口")
    top = {"--solver": solver, "--models-file": declared["models_file"], "--repeats": str(declared["repeats"]),
        "--time-limit": "5", "--wall-timeout": "35", "--threads": "1", "--seed": "0", "--candidate-backend": "ipm",
        "--reference-backend": "highs" if stage == "highs" else "ipm", "--output-dir": str(original)}
    for side, variant in enumerate(sides):
        prefix = "--reference-" if side == 0 else "--"
        config = manifest["variants"][variant]
        backend = "highs" if stage == "highs" and side == 0 else "ipm"
        accuracy = "strict" if side == 0 else "homogeneous_repair"
        require(config["backend"] == backend and manifest["solvers"][variant] ==
                {"path": solver, "sha256": identity["solver_sha256"], "hash_error": None}, "双侧真实后端和程序")
        if backend == "highs":
            require(all(value is None for key, value in config.items() if key.startswith("ipm_")) and
                    config["iteration_limit"] is None and config["scaling_passes"] is None, "HiGHS 无自主专属配置")
            for key, value in plan["highs_reference"].items():
                require(config["highs_"+key] == value, "HiGHS 独立参考配置")
                top[prefix+"highs-"+key] = value
        else:
            expected = expected_ipm_flags(plan, linear, accuracy, prefix)
            top.update(expected)
            require(config["ipm_method"] == "homogeneous" and config["ipm_linear_solver"] == linear and
                    config["ipm_linear_accuracy"] == accuracy and config["iteration_limit"] == 200 and config["scaling_passes"] == 5,
                    "清单算法消融")
            for key, value in {**plan["shared_ipm"], "direction-retries": "0"}.items():
                field = "ipm_"+key.replace("-", "_")
                expected_value = int(value) if key in ("direction-retries", "krylov-history-size", "central-corrections") else value
                require(config[field] == expected_value, "清单 IPM 共同配置："+key)
    require(flags(command, 2) == top, "顶层比较命令不得有隐含偏离")
    process = read(DATA/(stem+"-process.json"))
    require(process["plan_sha256"] == PLAN_SHA and process["solver_sha256_after"] == identity["solver_sha256"] and
            process["environment"] == {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "比较进程身份/环境")
    require(process["protected_scripts"] == {str(ROOT/"scripts"/name): identity["files"]["scripts/"+name]
            for name in ("benchmark.py", "compare_lp.py")}, "比较脚本前后身份")
    schedule = [(model, repetition, position, variant) for model in declared["models"] for repetition in range(declared["repeats"])
        for position, variant in enumerate(sides if (model["model_index"]+repetition) % 2 == 0 else sides[::-1])]
    require(len(rows) == len(schedule), "全部预定运行必须保留")
    for index, (row, (model, repetition, position, variant)) in enumerate(zip(rows, schedule)):
        require((row["run_index"], row["model_index"], row["input"], row["repetition"], row["position_in_pair"], row["requested_variant"]) ==
                (index, model["model_index"], model["input"], repetition, position, variant), "实际运行顺序")
        side = sides.index(variant)
        config = manifest["variants"][variant]
        backend = config["backend"]
        accuracy = config["ipm_linear_accuracy"]
        expected_flags = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
        if backend == "ipm":
            expected_flags.update(expected_ipm_flags(plan, linear, accuracy))
        else:
            expected_flags.update({"--highs-"+key: value for key, value in plan["highs_reference"].items()})
        require(row["command"][:3] == [solver, "solve", model["input"]] and flags(row["command"], 3) == expected_flags,
                "实际子进程命令")
        require(row["requested_backend"] == backend and row["hash_error"] is None and
                all(row[key] == model["sha256"] for key in ("model_sha256", "model_sha256_after", "frozen_model_sha256")) and
                all(row[key] == identity["solver_sha256"] for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")),
                "运行前后模型与程序身份")
        report = row["report"]
        if report is not None:
            require(strict_json(row["stdout"]) == report and benchmark.classify(row["returncode"], report) == row["classification"] and
                    row["status"] == report["status"], "完整报告和退出分类")
            verify_options(report, plan, linear, accuracy, backend)
            counts(report)
        else:
            require(row["classification"] in ("wall_timeout", "invalid_json", "process_error", "launch_error"), "缺报告失败须原样保留")
        require(finite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] > 0, "真实墙钟时间")
        numerical = compare.numerical_seconds(row)
        require(row["numerical_seconds"] == numerical, "完整数值计时")
        failure = compare.quality_failure(row, model, model["sha256"], identity["solver_sha256"], identity["solver_sha256"])
        if failure is None and numerical is None:
            failure = "invalid_timings"
        if failure is None and row["elapsed_wall_seconds"] > 35:
            failure = "wall_budget_exceeded"
        if failure is None and numerical > 5:
            failure = "numerical_budget_exceeded"
        require(row["quality_failure"] == failure and row["quality_pass"] is (failure is None), "原模型质量与共同预算重新验收")
        require(row["wall_budget_exceeded"] is (row["elapsed_wall_seconds"] > 35) and
                row["numerical_budget_exceeded"] is (numerical > 5 if numerical is not None else None), "超预算标志")
        close(row["quality_penalized_wall_seconds"], min(row["elapsed_wall_seconds"],35) if failure is None else 35, "失败墙钟罚分")
        close(row["quality_penalized_numerical_seconds"], min(numerical,5) if failure is None else 5, "失败数值罚分")
        close(row["penalized_capped_seconds"], min(row["elapsed_wall_seconds"],5) if row["classification"] == "resolved" else 5, "旧报告墙钟评分")
    recomputed = compare.summarize(rows, declared["models"], declared["repeats"],35,5,sides)
    recomputed.update(configurations=manifest["variants"], output_dir=str(original))
    require(recomputed == summary, "冻结报告协议的完整汇总重建")
    # 关键聚合另从原始质量和完整罚分自行计算，避免仅调用同一汇总函数。
    for side in sides:
        selected = [row for row in rows if row["requested_variant"] == side]
        values = summary["variants"][side]
        require(values["run_count"] == len(selected) and values["quality_pass_count"] == sum(row["quality_pass"] for row in selected), "独立达标计数")
        for key, field in (("wall", "quality_penalized_wall_seconds"), ("numerical", "quality_penalized_numerical_seconds")):
            value = math.expm1(sum(math.log1p(row[field]/10) for row in selected)/len(selected))*10
            close(values["penalized_"+key+"_sgm10_seconds"], value, "独立含失败 SGM10")
    require(process["returncode"] == (0 if all(row["quality_pass"] for row in rows) else 1), "比较进程实际退出码")
    return manifest, summary, rows


def trajectories(identity, plan, benchmark):
    path = DATA/"v50-old-mode-trajectory-runs.json.gz"
    with gzip.open(path,"rb") as source:
        records = strict_json(source.read())
    require(len(records) == 32, "32个旧配置对子")
    limited_count = equal_count = 0
    for index, row in enumerate(records):
        model = plan["old_trajectory"]["models"][index % 8]
        linear, accuracy, method = plan["old_trajectory"]["configurations"][index//8]
        require((row["model"],row["model_sha256"],row["linear_solver"],row["linear_accuracy"],row["method"]) ==
                (model["input"],model["sha256"],linear,accuracy,method), "旧对子顺序与声明")
        require(all(len(row[key]) == 2 for key in ("commands","reports","returncodes","outputs","hashes","protocol_valid")), "旧对子两侧完整")
        for side, version, digest in ((0,48,PARENT_SOLVER_SHA),(1,50,identity["solver_sha256"])):
            command, report, output = row["commands"][side], row["reports"][side], row["outputs"][side]
            expected = expected_ipm_flags(plan, linear, accuracy)
            expected["--ipm-method"] = method
            if linear == "qdldl":
                expected["--ipm-singleton-elimination"] = expected["--ipm-fixed-variable-elimination"] = "off"
            expected.update({"--backend":"ipm","--time-limit":"5","--threads":"1","--seed":"0"})
            require(command[:3] == [str(ROOT/f".tools/solver-ipm-v{version}.exe"),"solve",model["input"]] and flags(command,3) == expected,
                    "旧轨迹预声明命令")
            require(row["hashes"][side] == {key:{"solver":digest,"input":model["sha256"]} for key in ("before","after")}, "旧轨迹前后身份")
            if report is None:
                require(output["error"] == "wall_timeout" and row["protocol_valid"][side] is True,
                        "旧轨迹缺报告仅可记墙钟耗尽")
                continue
            require(strict_json(output["stdout"]) == report and benchmark.classify(row["returncodes"][side],report) in
                    ("resolved","incomplete","verification_failed"), "旧轨迹报告协议")
            require(output["error"] is None and row["protocol_valid"][side] is True and report["backend"] == "ipm" and
                    report["backend_version"] == ("ipm-cpu-0.41.0" if side == 0 else "ipm-cpu-0.42.0"), "旧轨迹后端、版本及协议标志")
            require(all(finite(report["timings"].get(key)) and report["timings"][key] >= 0 for key in
                        ("read_seconds","solve_seconds","verification_seconds","total_seconds")), "旧轨迹完整有限计时")
            require(report["options"]["ipm"]["method"] == method and report["options"]["ipm"]["linear_accuracy"] == accuracy and
                    report["options"]["ipm"]["linear_solver"] == linear, "旧轨迹实际方法")
            counts(report,legacy=side == 0)
            if side == 1:
                require(all(report[key] == 0 and type(report[key]) is int for key in NEW_FIELDS), "旧模式新增统计必须整数零")
        normalized = [{key:value for key,value in report.items() if key not in plan["old_trajectory"]["excluded_fields"]}
                      if isinstance(report,dict) else None for report in row["reports"]]
        equal = all(isinstance(report,dict) for report in row["reports"]) and normalized[0] == normalized[1] and row["returncodes"][0] == row["returncodes"][1]
        limited = any(output["error"] == "wall_timeout" or isinstance(report,dict) and (report["status"] == "time_limit" or
            report["timings"]["solve_seconds"]+report["timings"]["verification_seconds"] > 5)
            for report,output in zip(row["reports"],row["outputs"]))
        valid = isinstance(row["reports"][1],dict) and all(type(row["reports"][1].get(key)) is int and row["reports"][1][key] == 0 for key in NEW_FIELDS)
        require(row["non_time_equal"] is equal and row["budget_limited"] is limited and row["fields_valid"] is valid and valid and (equal or limited),
                "旧模式完整轨迹，预算分叉单列")
        equal_count += equal
        limited_count += limited
    return {"pairs":32,"equal":equal_count,"budget_limited":limited_count,"sha256":sha(path)}


def budget_limited(row):
    report = row.get("report")
    return (row["classification"] == "wall_timeout" or row["elapsed_wall_seconds"] > 35 or
            row["numerical_seconds"] is not None and row["numerical_seconds"] > 5 or
            isinstance(report, dict) and report["status"] == "time_limit")


def trigger_free_report(report):
    """同一程序的无触发消融仅去除计时与预声明的精度选项差异。"""
    normalized = copy.deepcopy(report)
    normalized.pop("timings")
    normalized["options"]["ipm"].pop("linear_accuracy")
    return normalized


def work_audit(manifest, rows):
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    result = {"trigger_free_pairs_equal": 0, "budget_limited_pairs": 0, "missing_report_pairs": 0,
              "triggered_pairs": 0, "missing_reports": [], "candidate_work": []}
    per_model = {}
    for row in rows:
        if not isinstance(row["report"], dict):
            result["missing_reports"].append({key: row[key] for key in
                ("run_index", "input", "repetition", "requested_variant", "returncode", "classification", "quality_failure")})
        if row["requested_variant"] == sides[1]:
            current = per_model.setdefault(row["input"], {"input": row["input"], "valid_reports": 0,
                "quality_passes": 0, "repairs": 0, "repair_successes": 0, "iterations": 0,
                "constant_solves": 0, "direction_solves": 0, "refinement_steps": 0, "weighted_krylov_steps": 0})
            current["quality_passes"] += int(row["quality_pass"])
            report = row["report"]
            if isinstance(report, dict):
                current["valid_reports"] += 1
                for target, source in (("repairs", NEW_FIELDS[0]), ("repair_successes", NEW_FIELDS[1]),
                    ("iterations", "ipm_iterations"), ("constant_solves", "ipm_homogeneous_constant_solves"),
                    ("direction_solves", "ipm_homogeneous_direction_solves"), ("refinement_steps", "ipm_refinement_steps"),
                    ("weighted_krylov_steps", "ipm_weighted_krylov_steps")):
                    current[target] += report[source]
    result["candidate_work"] = list(per_model.values())
    if manifest["variants"][sides[0]]["backend"] != "ipm":
        result["trigger_free_pairs_applicable"] = False
        return result
    result["trigger_free_pairs_applicable"] = True
    for offset in range(0, len(rows), 2):
        pair = {row["requested_variant"]: row for row in rows[offset:offset+2]}
        require(set(pair) == set(sides), "每个原始对子必须保留独立两侧")
        before, after = (pair[side] for side in sides)
        require((before["input"], before["repetition"]) == (after["input"], after["repetition"]), "正常路径对子对应")
        # 无报告故障与预算分叉均保留，但没有足够证据声称逐字段轨迹相同。
        if any(not isinstance(row["report"], dict) for row in (before, after)):
            result["missing_report_pairs"] += 1
        elif any(budget_limited(row) for row in (before, after)):
            result["budget_limited_pairs"] += 1
        elif after["report"][NEW_FIELDS[0]]:
            result["triggered_pairs"] += 1
        else:
            require(trigger_free_report(before["report"]) == trigger_free_report(after["report"]) and
                    before["returncode"] == after["returncode"] and before["error"] == after["error"],
                    "未触发恢复的非预算完整轨迹改变："+after["input"])
            result["trigger_free_pairs_equal"] += 1
    return result


def verify(initial_only=False):
    plan = frozen_plan()
    identity, benchmark, compare = sources(plan)
    old = trajectories(identity, plan, benchmark)
    groups = {}
    for stage in STAGES:
        if (DATA/("v50-"+stage+"-manifest.json")).exists():
            groups[stage] = verify_group(stage, identity, plan, benchmark, compare)
    require(set(STAGES[:3]) <= set(groups), "三个预声明初始阶段必须全部完成")
    require(sum(len(groups[stage][2]) for stage in STAGES[:3]) == 138, "138次初始运行必须全部保留")
    dev, probe = groups["development"], groups["probe"]
    gate = decision(dev[2], (dev[0]["reference_variant"], dev[0]["candidate_variant"]),
                    probe[2], (probe[0]["reference_variant"], probe[0]["candidate_variant"]))
    require(read(DATA/"v50-gate.json") == gate, "门槛须从 primary 原始质量和真实恢复独立重算")
    if not gate["eligible_for_netlib"]:
        require(set(groups) == set(STAGES[:3]), "未通过 primary 门槛不得扩集")
    elif not initial_only:
        require(set(groups) == set(STAGES), "通过门槛的最终核验需要两项完整扩集")
    result = {"passed": True, "scope": "初始阶段核验" if initial_only else "完整冻结实验核验",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA/"ipm-v50-identity.json"),
        "solver_sha256": identity["solver_sha256"], "source_files": len(identity["files"]),
        "linked_libraries": len(identity["libraries"]), "old_trajectories": old, "gate": gate,
        "run_count": sum(len(group[2]) for group in groups.values()), "stages": {},
        "self_test": self_test(),
        "limits": ["分解和常数恢复计数不能单独证明每次分母触发；触发范围依据冻结源码静态审查及解析边界测试。",
                   "恢复成功表示常数方向通过原检查并提交，不表示 LP 达到共同原模型精度。",
                   "无报告、崩溃和预算失败完整保留并按统一预算罚分；不将其当作恢复成功或相同轨迹证据。"]}
    for stage, (manifest, summary, rows) in groups.items():
        result["stages"][stage] = {"runs": len(rows), "reference_variant": manifest["reference_variant"],
            "candidate_variant": manifest["candidate_variant"], "variants": summary["variants"],
            "both_variants_all_repeats_pass_count": summary["both_variants_all_repeats_pass_count"],
            "work": work_audit(manifest, rows), "artifacts": {suffix: sha(DATA/("v50-"+stage+suffix)) for suffix in
                ("-manifest.json", "-summary.json", "-runs.jsonl.gz", "-command.json", "-process.json", "-run.log")}}
    return result


def self_test():
    rejected = 0
    def fails(operation):
        nonlocal rejected
        try:
            operation()
        except (AssertionError, ValueError, TypeError):
            rejected += 1
        else:
            raise AssertionError("非法合成输入没有被拒绝")
    for value in (True, -1, 1.5, 2**63, None):
        fails(lambda value=value: integer(value, "合成计数"))
    fails(lambda: strict_json('{"x":1,"x":2}'))
    fails(lambda: strict_json('{"x":NaN}'))
    fails(lambda: strict_json('{"x":1e999}'))
    fails(lambda: flags(["solver", "solve", "a", "--threads", "1", "--threads", "2"], 3))
    report = {"backend": "ipm", "options": {"ipm": {"method": "homogeneous", "linear_accuracy": "homogeneous_repair",
        "direction_retries": 0, "central_corrections": 0, "krylov_history_size": 0, "inexact_prefilter": False}},
        "ipm_iterations": 2, "ipm_homogeneous_constant_solves": 3, "ipm_homogeneous_direction_solves": 4,
        "ipm_refinement_steps": 5, "ipm_weighted_krylov_steps": 3, NEW_FIELDS[0]: 1, NEW_FIELDS[1]: 1,
        SCALARS[0]: 0.5, SCALARS[1]: 0.1, **dict.fromkeys(ZERO_FIELDS, 0)}
    counts(report)
    for key, value in ((NEW_FIELDS[0], True), (NEW_FIELDS[1], 2), (NEW_FIELDS[0], 4), (SCALARS[0], None),
                       ("ipm_homogeneous_direction_solves", 7), (ZERO_FIELDS[0], 1)):
        fails(lambda key=key, value=value: counts(dict(report, **{key: value})))
    invalid = copy.deepcopy(report)
    invalid["options"]["ipm"]["linear_accuracy"] = "strict"
    fails(lambda: counts(invalid))
    def row(path, side, passed, successes=0):
        return {"input": path, "requested_variant": side, "quality_pass": passed, "report": {NEW_FIELDS[1]: successes}}
    sides = ("reference", "candidate")
    good = [row("control", side, True) for side in sides]
    gain = [row("failure", "reference", False), row("failure", "candidate", True, 1)]
    require(decision(good, sides, gain, sides)["eligible_for_netlib"], "有真实恢复且逐例不降的 gain 应过门槛")
    require(not decision(good, sides, [row("failure", "reference", False), row("failure", "candidate", True)], sides)["eligible_for_netlib"], "gain 无实际修复不得扩集")
    require(not decision([row("control", "reference", True), row("control", "candidate", False)], sides, gain, sides)["eligible_for_netlib"], "开发丢失不得以探针 gain 抵销")
    require(not decision(good, sides, [row("failure", "reference", False), row("failure", "candidate", False, 1)], sides)["eligible_for_netlib"], "恢复计数成功不能代替 LP gain")
    missing = row("failure", "candidate", False)
    missing["report"] = None
    require(not decision(good, sides, [row("failure", "reference", False), missing], sides)["eligible_for_netlib"], "无报告不得贡献真实恢复成功")
    base = {"options": {"ipm": {"linear_accuracy": "strict"}}, "timings": {"solve_seconds": 1}, "ipm_iterations": 2}
    changed = copy.deepcopy(base)
    changed["options"]["ipm"]["linear_accuracy"] = "homogeneous_repair"
    changed["timings"]["solve_seconds"] = 2
    require(trigger_free_report(base) == trigger_free_report(changed), "正常路径仅去除允许的两处差异")
    changed["ipm_iterations"] = 3
    require(trigger_free_report(base) != trigger_free_report(changed), "不能剔除真实算法工作差异")
    missing_pair = [{"run_index": index, "input": "synthetic.mps", "repetition": 0, "requested_variant": side,
        "report": None if index == 0 else report, "returncode": 3221225477 if index == 0 else 4,
        "classification": "invalid_json" if index == 0 else "incomplete", "quality_pass": False,
        "quality_failure": "invalid_json" if index == 0 else "incomplete"} for index, side in enumerate(sides)]
    audit = work_audit({"reference_variant": sides[0], "candidate_variant": sides[1],
        "variants": {side: {"backend": "ipm"} for side in sides}}, missing_pair)
    require(audit["missing_report_pairs"] == 1 and len(audit["missing_reports"]) == 1 and
            audit["trigger_free_pairs_equal"] == 0 and audit["candidate_work"][0]["repair_successes"] == 1,
            "崩溃原样保留，候选常数修复成功与 LP 失败分别记录")
    return {"passed": True, "negative_cases": rejected, "gate_cases": 5, "trajectory_cases": 3,
            "scope": "纯合成协议和门槛，未编译或求解"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--initial-only", action="store_true", help="仅验已完成初始阶段，不要求有资格但尚未运行的扩集")
    parser.add_argument("--brief", action="store_true", help="仅向终端打印短摘要，完整结果写入输出文件")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.preflight:
        plan = frozen_plan(check_inputs=False)
        result = {"passed": True, "plan_sha256": PLAN_SHA, "initial_runs": 138, "old_pairs": plan["old_trajectory"]["pairs"],
                  "self_test": self_test(), "scope": "只核计划与合成协议，不代表正式实验已验"}
    else:
        result = verify(initial_only=args.initial_only)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(text)
    if args.brief:
        print(json.dumps({key:result[key] for key in ("passed","scope","run_count","gate") if key in result}, ensure_ascii=False))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
