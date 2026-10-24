"""独立核验第51轮目标归一化兼容性与同程序on/off实验，不运行求解器。"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
from pathlib import Path
import re
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "7475487f6d1cb8a6e63f04384292b0887df8cee8c143de482d1e7bf97091c8bf"
PARENT_IDENTITY_SHA = "2e9cfcded7e59b37ec1af19273b15db75e4d3adcc27026b94e41f2ab69591cda"
HELPER_SHA = "323e5ece25007e2207d5028bd43cc65a1e7b4db86ca89746768e8b5afa6eebe3"
STAGES = ("compatibility", "development", "probe", "netlib", "highs")
sys.dont_write_bytecode = True
helper_path = DATA / "v52-verify.py"
content = helper_path.read_bytes()
if hashlib.sha256(content).hexdigest() != HELPER_SHA:
    raise AssertionError("锁定复用核验器身份改变")
shared = types.ModuleType("v54_shared_audit")
shared.__file__ = str(helper_path)
exec(compile(content, str(helper_path), "exec"), shared.__dict__)
base = shared.base
require, read, sha, flags = shared.require, shared.read, shared.sha, shared.flags
strict_json, finite, close = shared.strict_json, shared.finite, shared.close
json, math = shared.json, shared.math
score, valid_process_report = shared.score, shared.valid_process_report


def frozen_plan(check_inputs=True):
    require(sha(DATA / "v54-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v54-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA == sha(DATA / "ipm-v52-identity.json") and
            plan["evidence_sha256"] == sha(DATA / "v53-verification.json"), "父程序与诊断证据身份")
    require((plan["parent_version"], plan["reference_version"], plan["candidate_version"], plan["reference_objective_scaling"],
             plan["candidate_objective_scaling"], plan["default_objective_scaling"]) == (52, 54, 54, "on", "off", True), "同新程序on/off与默认on")
    require((plan["method"], plan["primary_linear_solver"], plan["reference_accuracy"], plan["candidate_accuracy"], plan["time_limit"],
             plan["wall_timeout"], plan["threads"], plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            ("homogeneous", "normal_cholmod", "homogeneous_repair", "homogeneous_repair", 5, 35, 1, 0, 200, 5), "不混改数值政策或资源")
    parent = read(DATA / "v52-plan.json")
    require(all(plan[key] == parent[key] for key in ("shared_ipm", "highs_reference")) and
            all(plan["sets"][name] == parent["sets"][name] for name in ("development", "probe", "netlib")), "原输入/参数不改变")
    require((plan["compatibility_runs"], plan["initial_performance_runs"], plan["conditional_full_performance_runs"], plan["gate"]["ctest"]) ==
            (8, 102, 364, 18) and plan["no_model_specific_routing"] is True and
            plan["compatibility_exclusion"] == ["timings", "backend_version", "options.ipm.objective_scaling"], "运行数量与兼容唯一排除字段")
    require(plan["changed_source_files"] == ["include/solver/ipm_options.hpp", "src/ipm.cpp", "src/main.cpp", "scripts/benchmark.py",
            "scripts/compare_lp.py", "tests/test_ipm.cpp", "tests/test_cli.py", "tests/test_compare_lp.py"], "只允许事前8文件变更")
    for name, count, repeats in (("compatibility", 4, 1), ("development", 11, 3), ("probe", 6, 3), ("netlib", 91, 1)):
        group = plan["sets"][name]
        require(group["repeats"] == repeats and [item["model_index"] for item in group["models"]] == list(range(count)), "固定输入数量/顺序")
        path = Path(group["models_file"])
        require(sha(path) == group["models_file_sha256"], "固定模型列表身份")
        inputs = [str((path.parent / line.strip()).resolve()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(inputs == [item["input"] for item in group["models"]] and len(set(inputs)) == count, "模型列表完整且无重复")
        if name == "compatibility":
            require([Path(value).stem for value in inputs] == ["dfl001", "greenbea", "forplan", "afiro"], "事前四例兼容对照")
        if check_inputs:
            for item in group["models"]:
                require(item["hash_error"] is None and sha(item["input"]) == item["sha256"], "每个实际输入来源")
    return plan


def sources(plan):
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (52, 54)}
    parent, identity = identities[52], identities[54]
    require(identity["plan_sha256"] == PLAN_SHA and datetime.datetime.fromisoformat(identity["created_at_utc"]) >=
            datetime.datetime.fromisoformat(plan["created_at_utc"]), "计划先于候选冻结")
    for version, item in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == item["solver_sha256"], "冻结旧新程序真实身份")
    require(sha(ROOT / "build/release/solver.exe") == identity["solver_sha256"] != parent["solver_sha256"], "正式构建对应候选")
    require(set(identity["files"]) == set(parent["files"]) and len(identity["files"]) == 95, "95份生产来源集合不变")
    snapshots = {}
    for version, item in identities.items():
        path = DATA / f"ipm-v{version}-source.zip"
        require(sha(path) == item["source_zip_sha256"], "两冻结源码包身份")
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(item["files"]), "源码包唯一成员集合")
            snapshots[version] = {name: archive.read(name) for name in archive.namelist()}
        require(all(hashlib.sha256(value).hexdigest() == item["files"][name] for name, value in snapshots[version].items()), "每个源码包成员身份")
    require({name for name in identity["files"] if identity["files"][name] != parent["files"][name]} == set(plan["changed_source_files"]), "实际源码变化恰为冻结8文件")
    for name, digest in identity["files"].items():
        require(sha(ROOT / name) == digest, "新候选当前来源身份：" + name)
    require(identity["compiler"] == parent["compiler"] and sha(identity["compiler"]) == identity["compiler_sha256"] == parent["compiler_sha256"], "同一编译器")
    for group, directory in (("libraries", ROOT), ("logs", DATA)):
        for name, digest in identity[group].items():
            require(sha(directory / name) == digest, "新候选库/日志真实身份")
    parsed = [value for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', identity["link_command"])
              if not (value := quoted or plain).startswith("-")]
    require(len(identity["libraries"]) == 17 and set(identity["libraries"]) == set(parent["libraries"]) and
            {(ROOT / "build/release" / value).resolve() for value in parsed} == {(ROOT / name).resolve() for name in identity["libraries"]}, "真实链接17库集合")
    external = {name for name in parent["libraries"] if not Path(name).name.startswith("libsolver_")}
    require(len(external) == 8 and all(identity["libraries"][name] == parent["libraries"][name] for name in external), "8个第三方库与父版完全相同")
    require(identity["libraries"]["build/release/libsolver_ipm.a"] != parent["libraries"]["build/release/libsolver_ipm.a"], "新增目标归一化分支必须实际编译")
    patch, = strict_json(snapshots[54]["dependencies.lock.json"])["openblas"]["local_patches"]
    require(sha(ROOT / "build/openblas/openblas-0.3.33/src/solver_openblas_build" / patch["source_file"]) ==
            identity["patched_openblas_source_sha256"] == parent["patched_openblas_source_sha256"] == patch["patched_sha256"], "BLAS修补来源保持不变")
    for filename, key in (("v54-run.py", "runner_sha256"), ("v54-freeze.py", "freeze_script_sha256"), ("v54-runner-self-check.json", "runner_self_check_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验工具实际身份")
    require(read(DATA / "v54-runner-self-check.json") == {"passed": True, "command_cases": 5, "gate_cases": 6,
            "trajectory_cases": 10, "solver_runs": 0}, "实际测前runner自检")
    require(b"100% tests passed out of 18" in (DATA / "iteration-ipm-v54-build.log").read_bytes(), "正式18项CTest通过")
    loaded = []
    for name in ("benchmark", "compare_lp"):
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / "scripts" / (name + ".py"))
        sys.modules[name] = module
        exec(compile(snapshots[54]["scripts/" + name + ".py"], str(DATA / "ipm-v54-source.zip") + "/" + name, "exec"), module.__dict__)
        loaded.append(module)
    return identities, loaded[0], loaded[1]


def objective_mode(stage, reference):
    if stage == "compatibility":
        return None if reference else "on"
    if stage == "highs" and reference:
        return None
    return "on" if reference else "off"


def verify_options(report, plan, backend, version, mode):
    normalized = copy.deepcopy(report)
    if backend == "ipm":
        require(report["backend_version"] == f"ipm-cpu-0.{44 if version == 52 else 45}.0", "实际执行本侧版本")
        ipm = normalized["options"]["ipm"]
        if version == 54:
            require(type(ipm["objective_scaling"]) is bool and ipm.pop("objective_scaling") is (mode == "on"), "新选项必须为实际布尔开关")
        else:
            require("objective_scaling" not in ipm and mode is None, "旧程序报告不能虚构新开关")
        normalized["backend_version"] = "ipm-cpu-0.42.0"
    base.verify_options(normalized, plan, "normal_cholmod", "homogeneous_repair", backend)
    base.counts(report)
    require(report["model"]["is_mip"] is False, "连续LP")


def verify_group(stage, identities, plan, benchmark, compare):
    declared = plan["sets"]["netlib" if stage in ("netlib", "highs") else stage]
    identity = identities[54]
    stem = "v54-" + stage
    manifest, summary = (read(DATA / (stem + suffix)) for suffix in ("-manifest.json", "-summary.json"))
    original = ROOT / "results" / ("round-ipm-" + stem)
    raw = gzip.decompress((DATA / (stem + "-runs.jsonl.gz")).read_bytes())
    rows = [strict_json(line) for line in raw.splitlines()]
    require(raw == (original / "runs.jsonl").read_bytes(), "归档原始字节完整")
    for name in ("manifest", "summary"):
        require((DATA / (stem + "-" + name + ".json")).read_bytes() == (original / (name + ".json")).read_bytes(), "清单与摘要未改写")
    require(read(DATA / (stem + "-run.log")) == summary, "比较进程stdout与摘要相同")
    require(datetime.datetime.fromisoformat(manifest["started_at_utc"]) >= datetime.datetime.fromisoformat(identity["created_at_utc"]), "冻结后执行")
    require(manifest["models"] == declared["models"] and manifest["models_file"] == declared["models_file"] and
            manifest["models_file_sha256"] == declared["models_file_sha256"] and manifest["repeats"] == declared["repeats"] and
            all(manifest[key] == plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")), "冻结输入和预算")
    require(manifest["comparison_script_sha256"] == identity["files"]["scripts/compare_lp.py"] and
            manifest["benchmark_script_sha256"] == identity["files"]["scripts/benchmark.py"], "比较工具身份")
    candidate = str(ROOT / ".tools/solver-ipm-v54.exe")
    reference_version = 52 if stage == "compatibility" else 54
    reference = str(ROOT / f".tools/solver-ipm-v{reference_version}.exe")
    require(manifest["solver"] == candidate and manifest["solver_sha256"] == identity["solver_sha256"] and manifest["solver_hash_error"] is None, "候选清单身份")
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    require(len(set(sides)) == 2 and set(manifest["variants"]) == set(manifest["solvers"]) == set(sides), "真实独立双侧")
    require(manifest["ipm_objective_scaling"] == objective_mode(stage, False) and
            manifest["reference_ipm_objective_scaling"] == objective_mode(stage, True), "顶层清单开关独立请求")
    require(":objective_scaling=" not in sides[0] and
            (":objective_scaling=" not in sides[1] if stage == "compatibility" else
             sides[1].endswith(":objective_scaling=off") and sides[1].count(":objective_scaling=") == 1),
            "默认/on标签等价，只有off追加唯一末尾标识")
    command = read(DATA / (stem + "-command.json"))
    require(Path(command[0]) == ROOT / ".tools/python-3.14.8/python.exe" and Path(command[1]) == ROOT / "scripts/compare_lp.py", "实际比较入口")
    top = {"--solver": candidate, "--reference-solver": reference, "--models-file": declared["models_file"],
        "--repeats": str(declared["repeats"]), "--time-limit": "5", "--wall-timeout": "35", "--threads": "1", "--seed": "0",
        "--candidate-backend": "ipm", "--reference-backend": "highs" if stage == "highs" else "ipm", "--output-dir": str(original)}
    for index, side in enumerate(sides):
        version = reference_version if index == 0 else 54
        executable = reference if index == 0 else candidate
        backend = "highs" if stage == "highs" and index == 0 else "ipm"
        prefix = "--reference-" if index == 0 else "--"
        config = manifest["variants"][side]
        require(config["backend"] == backend and manifest["solvers"][side] ==
                {"path": executable, "sha256": identities[version]["solver_sha256"], "hash_error": None}, "每侧真实程序和后端")
        if backend == "highs":
            require(all(value is None for key, value in config.items() if key.startswith("ipm_")) and
                    config["iteration_limit"] is None and config["scaling_passes"] is None, "HiGHS无自主专属选项")
            for key, value in plan["highs_reference"].items():
                require(config["highs_" + key] == value, "HiGHS默认参考政策")
                top[prefix + "highs-" + key] = value
        else:
            top.update(base.expected_ipm_flags(plan, "normal_cholmod", "homogeneous_repair", prefix))
            mode = objective_mode(stage, index == 0)
            require(config["ipm_objective_scaling"] == mode, "清单目标归一化请求")
            if version == 54:
                top[prefix + "ipm-objective-scaling"] = mode
            require((config["ipm_method"], config["ipm_linear_solver"], config["ipm_linear_accuracy"], config["iteration_limit"], config["scaling_passes"]) ==
                    ("homogeneous", "normal_cholmod", "homogeneous_repair", 200, 5), "清单两侧相同LP政策")
            for key, value in {**plan["shared_ipm"], "direction-retries": "0"}.items():
                require(config["ipm_" + key.replace("-", "_")] == (int(value) if value == "0" else value), "共同IPM配置")
    require(flags(command, 2) == top, "顶层命令完整匹配，不能忽略reference-solver")
    process = read(DATA / (stem + "-process.json"))
    require(process["plan_sha256"] == PLAN_SHA and process["solver_sha256_after"] == identity["solver_sha256"] and
            process["reference_solver_sha256_after"] == identities[reference_version]["solver_sha256"] and
            process["environment"] == {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} and
            process["removed_environment"] == ["OPENBLAS_CORETYPE", "所有SOLVER_V前缀变量"], "比较进程、线程与自动内核选择")
    require(process["protected_scripts"] == {str(ROOT / "scripts" / name): identity["files"]["scripts/" + name]
            for name in ("benchmark.py", "compare_lp.py")}, "脚本执行前后身份")
    schedule = [(model, repetition, position, side) for model in declared["models"] for repetition in range(declared["repeats"])
        for position, side in enumerate(sides if (model["model_index"] + repetition) % 2 == 0 else sides[::-1])]
    require(len(rows) == len(schedule), "所有事前运行必须完整保留")
    for index, (row, (model, repetition, position, side)) in enumerate(zip(rows, schedule)):
        require((row["run_index"], row["model_index"], row["input"], row["repetition"], row["position_in_pair"], row["requested_variant"], row["seed"]) ==
                (index, model["model_index"], model["input"], repetition, position, side, 0), "真实交替顺序")
        version = reference_version if side == sides[0] else 54
        executable = reference if side == sides[0] else candidate
        backend = manifest["variants"][side]["backend"]
        solver_sha = identities[version]["solver_sha256"]
        expected = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
        expected.update(base.expected_ipm_flags(plan, "normal_cholmod", "homogeneous_repair") if backend == "ipm" else
                        {"--highs-" + key: value for key, value in plan["highs_reference"].items()})
        if backend == "ipm" and version == 54:
            expected["--ipm-objective-scaling"] = objective_mode(stage, side == sides[0])
        require(row["command"][:3] == [executable, "solve", model["input"]] and flags(row["command"], 3) == expected, "真实子命令对应本侧程序")
        require(row["requested_backend"] == backend and row["hash_error"] is None and
                all(row[key] == model["sha256"] for key in ("model_sha256", "model_sha256_after", "frozen_model_sha256")) and
                all(row[key] == solver_sha for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")), "每次前后身份")
        report = row["report"]
        if report is not None:
            require(strict_json(row["stdout"]) == report and benchmark.classify(row["returncode"], report) == row["classification"] and
                    row["status"] == report["status"] and report["input"] == model["input"], "报告、原始输出、状态与输入一致")
            verify_options(report, plan, backend, version, objective_mode(stage, side == sides[0]))
        else:
            require(row["classification"] in ("wall_timeout", "invalid_json", "process_error", "launch_error") and row["status"] is None,
                    "无报告保留为原始进程失败")
            if row["classification"] == "invalid_json":
                try:
                    strict_json(row["stdout"])
                except (ValueError, AssertionError):
                    pass
                else:
                    raise AssertionError("invalid_json不能隐藏一个可解析报告")
        failure = compare.quality_failure(row, model, model["sha256"], solver_sha, solver_sha)
        expected_score = score(row, failure)
        for key, value in expected_score.items():
            require(type(row[key]) is type(value) or type(row[key]) in (int, float) and type(value) in (int, float), "评分字段类型")
            require(row[key] == value, "质量、完整计时或失败罚分不一致：" + key)
        require(compare.numerical_seconds(row) == expected_score["numerical_seconds"], "独立重算数值耗时")
        close(row["penalized_capped_seconds"], min(row["elapsed_wall_seconds"], 5) if row["classification"] == "resolved" else 5, "旧字段罚分")
    recomputed = compare.summarize(rows, declared["models"], declared["repeats"], 35, 5, sides)
    recomputed.update(configurations=manifest["variants"], output_dir=str(original))
    require(recomputed == summary, "从全部重新验收的原始记录重建完整摘要")
    for side in sides:
        selected = [row for row in rows if row["requested_variant"] == side]
        actual = summary["variants"][side]
        passes = collections.Counter(row["model_index"] for row in selected if row["quality_pass"])
        require(actual["run_count"] == len(selected) and actual["quality_pass_count"] == sum(passes.values()) and
                actual["all_repeats_quality_pass_model_count"] == sum(passes[item["model_index"]] == declared["repeats"] for item in declared["models"]), "独立逐例次数聚合")
        for kind in ("wall", "numerical"):
            value = math.expm1(sum(math.log1p(row["quality_penalized_" + kind + "_seconds"] / 10) for row in selected) / len(selected)) * 10
            close(actual["penalized_" + kind + "_sgm10_seconds"], value, "独立含失败SGM10")
    require(process["returncode"] == (0 if all(row["quality_pass"] for row in rows) else 1), "比较进程退出码")
    return manifest, summary, rows


def compatibility_audit(manifest, rows, benchmark):
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    pairs = collections.defaultdict(dict)
    for row in rows:
        key, side = (row["input"], row["repetition"]), row["requested_variant"]
        require(side in sides and side not in pairs[key], "兼容对子不能重复或混入第三侧")
        pairs[key][side] = row
    result = {"pairs": len(pairs), "equal": 0, "budget_limited": 0, "invalid": 0, "different": 0}
    for pair in pairs.values():
        require(set(pair) == set(sides), "兼容对子不能缺侧")
        a, b = (pair[side] for side in sides)
        valid = all(valid_process_report(row, benchmark.classify(row["returncode"], row["report"])) for row in (a, b))
        if valid:
            valid = all(isinstance(row["report"].get("options", {}).get("ipm"), dict) and
                        isinstance(row["report"].get("timings"), dict) and
                        all(finite(row["report"]["timings"].get(key)) and row["report"]["timings"][key] >= 0 for key in
                            ("solve_seconds", "verification_seconds")) for row in (a, b))
        if valid:
            valid = "objective_scaling" not in a["report"]["options"]["ipm"] and b["report"]["options"]["ipm"].get("objective_scaling") is True
        if not valid:
            result["invalid"] += 1
            continue
        budget = any(row["report"]["status"] == "time_limit" or row["report"]["timings"]["solve_seconds"] +
                     row["report"]["timings"]["verification_seconds"] > 5 or row["elapsed_wall_seconds"] > 35 for row in (a, b))
        if budget:
            result["budget_limited"] += 1
            continue
        normalized = copy.deepcopy([a["report"], b["report"]])
        for report in normalized:
            del report["timings"]
            report.pop("backend_version", None)
        del normalized[1]["options"]["ipm"]["objective_scaling"]
        same = (json.dumps(normalized[0], sort_keys=True, allow_nan=False) == json.dumps(normalized[1], sort_keys=True, allow_nan=False) and
                a["returncode"] == b["returncode"])
        result["equal" if same else "different"] += 1
    result["passed"] = result["invalid"] == result["different"] == 0
    return result


def decision(groups, compatibility):
    # 锁定辅助实现直接从raw计算逐例覆盖和SGM，不使用runner摘要作门槛输入。
    result = shared.decision(groups, compatibility)
    result["plan_sha256"] = PLAN_SHA
    return result


def self_test():
    previous = shared.self_test()
    benchmark = types.ModuleType("v54_synthetic_benchmark")
    benchmark.__file__ = str(ROOT / "scripts/benchmark.py")
    exec(compile((ROOT / "scripts/benchmark.py").read_bytes(), benchmark.__file__, "exec"), benchmark.__dict__)
    manifest = {"reference_variant": "r", "candidate_variant": "c"}
    original = [{"input": "x", "repetition": 0, "requested_variant": side, "error": None, "returncode": 4, "elapsed_wall_seconds": 2,
                 "report": {"schema_version": 1, "status": "numerical_error", "backend_version": side,
                            "timings": {"solve_seconds": 1, "verification_seconds": 0}, "iterations": 2,
                            "options": {"ipm": {} if side == "r" else {"objective_scaling": True}}}} for side in ("r", "c")]
    checked = 0
    def audit(rows, field):
        nonlocal checked
        result = compatibility_audit(manifest, rows, benchmark)
        require(result[field] == 1 and result["passed"] is (field in ("equal", "budget_limited")), "兼容轨迹合成反例")
        checked += 1
    audit(original, "equal")
    rows = copy.deepcopy(original)
    rows[1]["report"]["iterations"] = 3
    audit(rows, "different")
    rows[1]["report"]["timings"]["solve_seconds"] = 6
    audit(rows, "budget_limited")
    rows[1]["report"]["status"] = "verification_failed"
    rows[1]["returncode"] = 3221225477
    audit(rows, "invalid")
    rows[1]["report"] = None
    audit(rows, "invalid")
    for value in (False, 1, "on", None):
        rows = copy.deepcopy(original)
        rows[1]["report"]["options"]["ipm"]["objective_scaling"] = value
        audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[0]["report"]["options"]["ipm"]["objective_scaling"] = True
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[1]["report"]["options"]["ipm"]["extra"] = 1
    audit(rows, "different")
    rows = copy.deepcopy(original)
    rows[1]["elapsed_wall_seconds"] = 36
    audit(rows, "budget_limited")
    rows[1]["error"] = "wall_timeout"
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[1]["returncode"] = False
    audit(rows, "invalid")
    return {"passed": True, "compatibility_cases": checked, "shared_checks": previous, "solver_runs": 0}


def verify(initial_only=False):
    plan = frozen_plan()
    identities, benchmark, compare = sources(plan)
    groups = {"compatibility": verify_group("compatibility", identities, plan, benchmark, compare)}
    compatibility = compatibility_audit(groups["compatibility"][0], groups["compatibility"][2], benchmark)
    gate = None
    if compatibility["passed"]:
        groups.update({stage: verify_group(stage, identities, plan, benchmark, compare) for stage in ("development", "probe")})
        gate = decision(groups, compatibility)
        require(read(DATA / "v54-gate.json") == gate, "独立重算兼容、逐例覆盖及速度门槛")
        if gate["eligible_for_netlib"] and not initial_only:
            groups.update({stage: verify_group(stage, identities, plan, benchmark, compare) for stage in ("netlib", "highs")})
        forbidden = () if gate["eligible_for_netlib"] else ("netlib", "highs")
    else:
        forbidden = ("development", "probe", "netlib", "highs")
    for stage in forbidden:
        require(not any(DATA.glob("v54-" + stage + "-*")) and not (ROOT / ("results/round-ipm-v54-" + stage)).exists(), "前置/扩集门槛失败时不能继续或丢弃后续证据")
    count = sum(len(group[2]) for group in groups.values())
    expected = 8 if not compatibility["passed"] else 474 if gate["eligible_for_netlib"] and not initial_only else 110
    require(count == expected, "固定兼容与性能数量完整保留")
    result = {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": f"{count}条冻结记录核验，兼容8条单列", "plan_sha256": PLAN_SHA, "helper_sha256": HELPER_SHA,
              "verifier_sha256": sha(__file__), "identity_sha256": sha(DATA / "ipm-v54-identity.json"),
              "solver_sha256": identities[54]["solver_sha256"], "parent_solver_sha256": identities[52]["solver_sha256"],
              "source_files": 95, "linked_libraries": 17, "ctest_passed": 18,
              "changed_source_files": plan["changed_source_files"],
              "changed_libraries": sorted(name for name in identities[54]["libraries"] if identities[54]["libraries"][name] != identities[52]["libraries"][name]),
              "compatibility": compatibility, "gate": gate, "run_count": count, "compatibility_runs": 8, "performance_runs": count - 8,
              "stages": {}, "self_test": self_test(),
              "limits": ["质量复核重新验既有原模型报告和共同预算，不重新求解LP或构造原始矩阵。",
                         "只有旧v52与新on前置对照要求非预算轨迹相等；on/off性能对照允许不同数值轨迹，所有失败照常罚分。",
                         "兼容8条不计入性能SGM；门槛停止时保留全部初始记录，不生成未执行的完整公开对照成绩。",
                         "passed表示证据完整及冻结合同执行正确，不代表已经打平HiGHS。"]}
    for stage, (manifest, summary, rows) in groups.items():
        result["stages"][stage] = {"runs": len(rows), "reference_variant": manifest["reference_variant"], "candidate_variant": manifest["candidate_variant"],
            "variants": summary["variants"], "both_variants_all_repeats_pass_count": summary["both_variants_all_repeats_pass_count"],
            "missing_reports": [{key: row[key] for key in ("run_index", "input", "requested_variant", "returncode", "classification", "quality_failure")}
                                for row in rows if row["report"] is None],
            "invalid_process_reports": [{key: row[key] for key in ("run_index", "input", "requested_variant", "returncode", "classification", "quality_failure")}
                for row in rows if not valid_process_report(row, benchmark.classify(row["returncode"], row["report"]))],
            "artifacts": {suffix: sha(DATA / ("v54-" + stage + suffix)) for suffix in
                          ("-manifest.json", "-summary.json", "-runs.jsonl.gz", "-command.json", "-process.json", "-run.log")}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.preflight:
        frozen_plan(check_inputs=False)
        result = {"passed": True, "scope": "只核冻结计划与合成协议，不读取性能raw或大库", "self_test": self_test()}
    else:
        result = verify(args.initial_only)
    destination = args.output
    if destination is None and not (args.preflight or args.self_test):
        destination = DATA / ("v54-initial-verification.json" if args.initial_only else "v54-verification.json")
    if destination is not None:
        with destination.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "run_count", "compatibility", "gate", "self_test") if key in result}, ensure_ascii=False))
