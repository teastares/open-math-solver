"""独立核验第49轮固定下三角缓存的双程序来源、精确轨迹与完整LP评分。"""

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
PLAN_SHA = "d1a027a5eb7b045b9e138da205bfb884323984af1242b1b4d7e22cd09d338648"
PARENT_IDENTITY_SHA = "2f979abe7625cc4859746d314e1091e6527949b368aabee3948d6e7ab34e4c38"
HELPER_SHA = "21d851bc6d3c7ccc3dc98f14010cb46e73d510a97a31725da7a43e3fb47c29ac"
STAGES = ("development", "probe", "netlib", "highs")
sys.dont_write_bytecode = True

# 只复用锁定的无副作用检查，不调用旧轮来源、稳定性或BLAS前后核验。
helper_path = DATA / "v51-verify.py"
helper_bytes = helper_path.read_bytes()
if hashlib.sha256(helper_bytes).hexdigest() != HELPER_SHA:
    raise AssertionError("复用核验器身份改变")
shared = types.ModuleType("v52_shared_audit")
shared.__file__ = str(helper_path)
exec(compile(helper_bytes, str(helper_path), "exec"), shared.__dict__)
base = shared.base
require, read, sha, flags = shared.require, shared.read, shared.sha, shared.flags
strict_json, finite, close = shared.strict_json, shared.finite, shared.close
json, math = shared.json, shared.math
score, valid_process_report = shared.score, shared.valid_process_report


def frozen_plan(check_inputs=True):
    require(sha(DATA / "v52-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v52-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA == sha(DATA / "ipm-v51-identity.json") and
            plan["evidence_sha256"] == sha(DATA / "v51-verification.json"), "父程序与上轮完整证据身份")
    require((plan["reference_version"], plan["candidate_version"], plan["reference_accuracy"], plan["candidate_accuracy"]) ==
            (51, 52, "homogeneous_repair", "homogeneous_repair"), "同政策旧51与新52")
    require((plan["primary_linear_solver"], plan["method"], plan["time_limit"], plan["wall_timeout"], plan["threads"],
             plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            ("normal_cholmod", "homogeneous", 5, 35, 1, 0, 200, 5), "冻结算法、精度与预算")
    parent = read(DATA / "v51-plan.json")
    require(all(plan[key] == parent[key] for key in ("sets", "shared_ipm", "highs_reference", "initial_stages")) and
            (plan["initial_performance_runs"], plan["conditional_full_performance_runs"], plan["gate"]["ctest"]) == (102, 364, 18) and
            plan["no_model_specific_routing"] is True, "固定输入、资源与公平对照")
    for name, count, repeats in (("development", 11, 3), ("probe", 6, 3), ("netlib", 91, 1)):
        group = plan["sets"][name]
        require(group["repeats"] == repeats and len(group["models"]) == count and
                [item["model_index"] for item in group["models"]] == list(range(count)), "输入数量与顺序")
        path = Path(group["models_file"])
        require(sha(path) == group["models_file_sha256"], "列表身份")
        actual = [str((path.parent / line.strip()).resolve()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(actual == [item["input"] for item in group["models"]] and len(set(actual)) == count, "输入不重复、不删改")
        if check_inputs:
            for item in group["models"]:
                require(item["hash_error"] is None and sha(item["input"]) == item["sha256"], "模型身份")
    return plan


def sources(plan):
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (51, 52)}
    parent, identity = identities[51], identities[52]
    require(identity["plan_sha256"] == PLAN_SHA and datetime.datetime.fromisoformat(identity["created_at_utc"]) >=
            datetime.datetime.fromisoformat(plan["created_at_utc"]), "计划先于候选冻结")
    for version, item in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == item["solver_sha256"], "两真实冻结程序身份")
    require(sha(ROOT / "build/release/solver.exe") == identity["solver_sha256"] != parent["solver_sha256"], "正式构建候选与父程序不同")
    require(set(identity["files"]) == set(parent["files"]) and len(identity["files"]) == 95, "95份源码集合不变")
    snapshots = {}
    for version, item in identities.items():
        path = DATA / f"ipm-v{version}-source.zip"
        require(sha(path) == item["source_zip_sha256"], "两源码包身份")
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(item["files"]), "真实源码包集合")
            snapshot = {name: archive.read(name) for name in archive.namelist()}
        require(all(hashlib.sha256(value).hexdigest() == item["files"][name] for name, value in snapshot.items()), "每份源码包成员身份")
        snapshots[version] = snapshot
    changed = {name for name in parent["files"] if parent["files"][name] != identity["files"][name]}
    require(changed == {"src/normal_cholesky.cpp", "tests/test_normal_cholesky.cpp", "src/ipm.cpp"}, "只允许缓存、相关回归与版本标识变化")
    old_ipm, new_ipm = snapshots[51]["src/ipm.cpp"], snapshots[52]["src/ipm.cpp"]
    old_line, = re.findall(rb'std::string ipm_version\(\) \{ return "ipm-cpu-0\.43\.0"; \}\r?\n', old_ipm)
    new_line, = re.findall(rb'std::string ipm_version\(\) \{ return "ipm-cpu-0\.44\.0"; \}\r?\n', new_ipm)
    require(new_ipm.replace(new_line, old_line) == old_ipm, "除版本所在行外LP主流程逐字节相同")
    for name, digest in identity["files"].items():
        require(sha(ROOT / name) == digest, "当前候选来源身份：" + name)
    require(identity["compiler"] == parent["compiler"] and sha(identity["compiler"]) == identity["compiler_sha256"] == parent["compiler_sha256"], "同一编译器")
    for group, directory in (("libraries", ROOT), ("logs", DATA)):
        for name, digest in identity[group].items():
            require(sha(directory / name) == digest, "当前候选库和日志身份")
    libraries = {(ROOT / name).resolve() for name in identity["libraries"]}
    parsed = [value for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', identity["link_command"])
              if not (value := quoted or plain).startswith("-")]
    require(len(libraries) == 17 and {(ROOT / "build/release" / name).resolve() for name in parsed} == libraries and
            sum(path.name == "libopenblas.a" for path in libraries) == 1, "真实链接17库与唯一BLAS")
    changed_libraries = {"build/release/libsolver_normal_cholesky.a", "build/release/libsolver_ipm.a"}
    unchanged = set(parent["libraries"]) - changed_libraries
    require(set(identity["libraries"]) == set(parent["libraries"]) and
            all(identity["libraries"][name] == parent["libraries"][name] for name in unchanged), "未涉及的外部及数值依赖库不得改变")
    require({name for name in identity["libraries"] if identity["libraries"][name] != parent["libraries"][name]} == changed_libraries,
            "缓存实现和版本库必须实际重新编译，不能仅替换源码包")
    require(identity["libraries"]["build/openblas/install/lib/libopenblas.a"] == parent["libraries"]["build/openblas/install/lib/libopenblas.a"], "本轮BLAS未变")
    patch, = strict_json(snapshots[52]["dependencies.lock.json"])["openblas"]["local_patches"]
    require(identity["files"][patch["path"]] == patch["sha256"] and
            sha(ROOT / "build/openblas/openblas-0.3.33/src/solver_openblas_build" / patch["source_file"]) ==
            identity["patched_openblas_source_sha256"] == parent["patched_openblas_source_sha256"] == patch["patched_sha256"], "沿用同一已修BLAS源码")
    for filename, key in (("v52-run.py", "runner_sha256"), ("v52-freeze.py", "freeze_script_sha256"),
                          ("v52-runner-self-check.json", "runner_self_check_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验工具身份")
    require(read(DATA / "v52-runner-self-check.json") == {"passed": True, "command_cases": 4, "gate_cases": 6,
            "trajectory_cases": 5, "solver_runs": 0}, "测前runner检查")
    require(b"100% tests passed out of 18" in (DATA / "iteration-ipm-v52-build.log").read_bytes(), "正式18项CTest通过")
    detail = (DATA / "iteration-ipm-v52-ctest-detail.log").read_bytes()
    require("正规方程 Cholesky 隔离检查通过".encode("utf-8") in detail and
            "公共 BLAS 数值与存储边界检查通过：19 组".encode("utf-8") in detail, "缓存和BLAS回归实际执行")
    loaded = []
    for name in ("benchmark", "compare_lp"):
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / "scripts" / (name + ".py"))
        sys.modules[name] = module
        exec(compile(snapshots[52]["scripts/" + name + ".py"], str(DATA / "ipm-v52-source.zip") + "/" + name, "exec"), module.__dict__)
        loaded.append(module)
    return identities, loaded[0], loaded[1]


def verify_options(report, plan, backend, version, accuracy="homogeneous_repair"):
    if backend == "ipm":
        require(report["backend_version"] == f"ipm-cpu-0.{43 if version == 51 else 44}.0", "真实执行对应本侧版本")
    options_copy = dict(report)
    if backend == "ipm":
        options_copy["backend_version"] = "ipm-cpu-0.42.0"
    base.verify_options(options_copy, plan, "normal_cholmod", accuracy, backend)
    base.counts(report)
    require(report["model"]["is_mip"] is False, "连续LP")

def verify_group(stage, identities, plan, benchmark, compare):
    declared = plan["sets"]["netlib" if stage in ("netlib", "highs") else stage]
    identity = identities[52]
    stem = "v52-" + stage
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
    candidate = str(ROOT / ".tools/solver-ipm-v52.exe")
    reference_version = 52 if stage == "highs" else 51
    reference = str(ROOT / f".tools/solver-ipm-v{reference_version}.exe")
    require(manifest["solver"] == candidate and manifest["solver_sha256"] == identity["solver_sha256"] and manifest["solver_hash_error"] is None, "候选清单身份")
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    require(len(set(sides)) == 2 and set(manifest["variants"]) == set(manifest["solvers"]) == set(sides), "真实独立双侧")
    command = read(DATA / (stem + "-command.json"))
    require(Path(command[0]) == ROOT / ".tools/python-3.14.8/python.exe" and Path(command[1]) == ROOT / "scripts/compare_lp.py", "实际比较入口")
    top = {"--solver": candidate, "--reference-solver": reference, "--models-file": declared["models_file"],
        "--repeats": str(declared["repeats"]), "--time-limit": "5", "--wall-timeout": "35", "--threads": "1", "--seed": "0",
        "--candidate-backend": "ipm", "--reference-backend": "highs" if stage == "highs" else "ipm", "--output-dir": str(original)}
    for index, side in enumerate(sides):
        version = reference_version if index == 0 else 52
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
            require((config["ipm_method"], config["ipm_linear_solver"], config["ipm_linear_accuracy"], config["iteration_limit"], config["scaling_passes"]) ==
                    ("homogeneous", "normal_cholmod", "homogeneous_repair", 200, 5), "清单两侧相同LP政策")
            for key, value in {**plan["shared_ipm"], "direction-retries": "0"}.items():
                require(config["ipm_" + key.replace("-", "_")] == (int(value) if value == "0" else value), "共同IPM配置")
    require(flags(command, 2) == top, "顶层命令完整匹配，不能忽略reference-solver")
    process = read(DATA / (stem + "-process.json"))
    require(process["plan_sha256"] == PLAN_SHA and process["solver_sha256_after"] == identity["solver_sha256"] and
            process["reference_solver_sha256_after"] == identities[reference_version]["solver_sha256"] and
            process["environment"] == {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} and
            process["removed_environment"] == ["OPENBLAS_CORETYPE", "SOLVER_V49_DIAGNOSTICS"], "比较进程、线程与自动内核选择")
    require(process["protected_scripts"] == {str(ROOT / "scripts" / name): identity["files"]["scripts/" + name]
            for name in ("benchmark.py", "compare_lp.py")}, "脚本执行前后身份")
    schedule = [(model, repetition, position, side) for model in declared["models"] for repetition in range(declared["repeats"])
        for position, side in enumerate(sides if (model["model_index"] + repetition) % 2 == 0 else sides[::-1])]
    require(len(rows) == len(schedule), "所有事前运行必须完整保留")
    for index, (row, (model, repetition, position, side)) in enumerate(zip(rows, schedule)):
        require((row["run_index"], row["model_index"], row["input"], row["repetition"], row["position_in_pair"], row["requested_variant"], row["seed"]) ==
                (index, model["model_index"], model["input"], repetition, position, side, 0), "真实交替顺序")
        version = reference_version if side == sides[0] else 52
        executable = reference if side == sides[0] else candidate
        backend = manifest["variants"][side]["backend"]
        solver_sha = identities[version]["solver_sha256"]
        expected = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
        expected.update(base.expected_ipm_flags(plan, "normal_cholmod", "homogeneous_repair") if backend == "ipm" else
                        {"--highs-" + key: value for key, value in plan["highs_reference"].items()})
        require(row["command"][:3] == [executable, "solve", model["input"]] and flags(row["command"], 3) == expected, "真实子命令对应本侧程序")
        require(row["requested_backend"] == backend and row["hash_error"] is None and
                all(row[key] == model["sha256"] for key in ("model_sha256", "model_sha256_after", "frozen_model_sha256")) and
                all(row[key] == solver_sha for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")), "每次前后身份")
        report = row["report"]
        if report is not None:
            require(strict_json(row["stdout"]) == report and benchmark.classify(row["returncode"], report) == row["classification"] and
                    row["status"] == report["status"] and report["input"] == model["input"], "报告、原始输出、状态与输入一致")
            verify_options(report, plan, backend, version)
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


def trajectory_audit(manifest, rows, benchmark):
    sides = (manifest["reference_variant"], manifest["candidate_variant"])
    pairs = collections.defaultdict(dict)
    for row in rows:
        key, side = (row["input"], row["repetition"]), row["requested_variant"]
        require(side in sides and side not in pairs[key], "轨迹对子不能重复或混入第三侧")
        pairs[key][side] = row
    result = {"pairs": len(pairs), "equal": 0, "budget_limited": 0, "invalid": 0, "different": 0}
    for pair in pairs.values():
        require(set(pair) == set(sides), "轨迹对子不得缺侧")
        a, b = (pair[side] for side in sides)
        valid = all(valid_process_report(row, benchmark.classify(row["returncode"], row["report"])) for row in (a, b))
        if valid:
            valid = all(isinstance(row["report"].get("timings"), dict) and
                        all(finite(row["report"]["timings"].get(key)) and row["report"]["timings"][key] >= 0
                            for key in ("solve_seconds", "verification_seconds")) for row in (a, b))
        if not valid:
            result["invalid"] += 1
            continue
        # 只有合法完整报告的数值预算受限才豁免，不能用缺报告、崩溃或墙钟异常伪装。
        if any(row["report"]["status"] == "time_limit" or
               row["report"]["timings"]["solve_seconds"] + row["report"]["timings"]["verification_seconds"] > 5
               for row in (a, b)):
            result["budget_limited"] += 1
            continue
        normalized = [{key: value for key, value in row["report"].items() if key not in ("timings", "backend_version")}
                      for row in (a, b)]
        # JSON类型也必须保持一致，不能把true和1视为同一报告。
        encoded = [json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False) for report in normalized]
        same = encoded[0] == encoded[1] and a["returncode"] == b["returncode"]
        result["equal" if same else "different"] += 1
    result["passed"] = result["invalid"] == result["different"] == 0
    return result


def decision(groups, trajectories):
    counts, costs = {}, {}
    for stage in STAGES[:2]:
        manifest, _, rows = groups[stage]
        sides = (manifest["reference_variant"], manifest["candidate_variant"])
        counts[stage] = collections.defaultdict(lambda: [0, 0])
        for row in rows:
            counts[stage][row["input"]][sides.index(row["requested_variant"])] += int(row["quality_pass"])
        costs[stage] = {}
        for side in sides:
            selected = [row for row in rows if row["requested_variant"] == side]
            costs[stage][side] = {kind: math.expm1(sum(math.log1p(row["quality_penalized_" + kind + "_seconds"] / 10)
                for row in selected) / len(selected)) * 10 for kind in ("wall", "numerical")}
    lost = {stage: [Path(path).stem for path, (before, after) in counts[stage].items() if after < before] for stage in STAGES[:2]}
    gain = [Path(path).stem for path, (before, after) in counts["probe"].items() if after > before]
    manifest = groups["development"][0]
    a, b = (costs["development"][manifest[key]] for key in ("reference_variant", "candidate_variant"))
    speed = b["numerical"] < a["numerical"] and b["wall"] <= a["wall"]
    return {"development_lost": lost["development"], "probe_lost": lost["probe"], "probe_gain": gain,
            "development_speed_gate": speed, "trajectories": trajectories,
            "eligible_for_netlib": not lost["development"] and not lost["probe"] and trajectories["passed"] and (speed or bool(gain)),
            "plan_sha256": PLAN_SHA}


def self_test():
    previous = shared.self_test()
    benchmark = types.ModuleType("v52_synthetic_benchmark")
    benchmark.__file__ = str(ROOT / "scripts/benchmark.py")
    exec(compile((ROOT / "scripts/benchmark.py").read_bytes(), benchmark.__file__, "exec"), benchmark.__dict__)
    manifest = {"reference_variant": "r", "candidate_variant": "c"}
    original = [{"input": "x", "repetition": 0, "requested_variant": side, "error": None, "returncode": 4,
        "report": {"schema_version": 1, "status": "numerical_error", "backend_version": side, "iterations": 2,
                   "timings": {"solve_seconds": 1, "verification_seconds": 0}}} for side in ("r", "c")]
    cases = 0
    def audit(rows, field):
        nonlocal cases
        result = trajectory_audit(manifest, rows, benchmark)
        require(result[field] == 1 and result["passed"] is (field in ("equal", "budget_limited")), "独立轨迹合成反例")
        cases += 1
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
    rows = copy.deepcopy(original)
    rows[1]["report"]["status"] = "time_limit"
    audit(rows, "budget_limited")
    rows[1]["error"] = "wall_timeout"
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[1]["returncode"] = False
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[1]["report"]["timings"]["solve_seconds"] = True
    audit(rows, "invalid")
    rows = copy.deepcopy(original)
    rows[0]["report"]["extra"] = True
    rows[1]["report"]["extra"] = 1
    audit(rows, "different")
    def group(before, after, numerical=1, wall=2):
        rows = [{"input": "synthetic.mps", "requested_variant": side, "quality_pass": index < count,
                 "quality_penalized_numerical_seconds": 2 if side == "r" else numerical,
                 "quality_penalized_wall_seconds": 2 if side == "r" else wall}
                for side, count in (("r", before), ("c", after)) for index in range(3)]
        return manifest, {}, rows
    gates = [(3, 3, 0, 0, 1, 2, True, True), (3, 2, 0, 3, 1, 2, True, False),
             (3, 3, 3, 2, 1, 2, True, False), (3, 3, 0, 3, 3, 3, True, True),
             (3, 3, 0, 3, 1, 2, False, False), (3, 3, 0, 0, 2, 2, True, False),
             (3, 3, 0, 0, 1, 3, True, False), (3, 3, 0, 0, 3, 1, True, False)]
    for a, b, c, d, numerical, wall, same, expected in gates:
        result = decision({"development": group(a, b, numerical, wall), "probe": group(c, d)}, {"passed": same})
        require(result["eligible_for_netlib"] is expected, "速度、覆盖、探针增益与轨迹联合门槛")
    rejected = 0
    for rows in (original + [original[0]], original[:1]):
        try:
            trajectory_audit(manifest, rows, benchmark)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError("重复或缺失侧未被拒绝")
    return {"passed": True, "trajectory_cases": cases, "pair_rejection_cases": rejected, "gate_cases": len(gates),
            "shared_checks": previous, "solver_runs": 0}


def verify(initial_only=False):
    plan = frozen_plan()
    identities, benchmark, compare = sources(plan)
    groups = {stage: verify_group(stage, identities, plan, benchmark, compare) for stage in STAGES[:2]}
    audits = {stage: trajectory_audit(group[0], group[2], benchmark) for stage, group in groups.items()}
    audits["passed"] = all(audit["passed"] for audit in audits.values())
    gate = decision(groups, audits)
    require(read(DATA / "v52-gate.json") == gate, "门槛由完整原始报告与逐例质量独立重算")
    if gate["eligible_for_netlib"] and not initial_only:
        groups.update({stage: verify_group(stage, identities, plan, benchmark, compare) for stage in STAGES[2:]})
    elif not gate["eligible_for_netlib"]:
        require(not any(DATA.glob("v52-netlib-*")) and not any(DATA.glob("v52-highs-*")) and
                not (ROOT / "results/round-ipm-v52-netlib").exists() and not (ROOT / "results/round-ipm-v52-highs").exists(),
                "门槛失败时不得扩集或丢弃扩集证据")
    count = sum(len(group[2]) for group in groups.values())
    require(count == (466 if gate["eligible_for_netlib"] and not initial_only else 102), "完整固定运行数量")
    result = {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "初始102条核验" if initial_only else ("完整466条冻结实验核验" if count == 466 else "门槛停止的完整102条实验核验"),
              "plan_sha256": PLAN_SHA, "helper_sha256": HELPER_SHA, "verifier_sha256": sha(__file__),
              "identity_sha256": sha(DATA / "ipm-v52-identity.json"), "solver_sha256": identities[52]["solver_sha256"],
              "parent_solver_sha256": identities[51]["solver_sha256"], "source_files": 95, "linked_libraries": 17,
              "changed_source_files": sorted(name for name in identities[52]["files"] if identities[52]["files"][name] != identities[51]["files"][name]),
              "ctest_passed": 18, "gate": gate, "run_count": count, "stages": {}, "self_test": self_test(),
              "limits": ["公开LP验证复算既存原模型检查报告、共同容差与预算，不重新求解或重建原始矩阵。",
                         "初始非预算完整报告仅排除timings/backend_version，其余字段及退出码逐对完全相等；预算报告仍计入失败与成本。",
                         "本轮沿用同一已修BLAS库，不重新进行旧库负对照或固定20次稳定性试验。",
                         "passed表示证据完整且门槛按冻结合同执行，并不表示相对HiGHS性能目标已达到。"]}
    for stage, (manifest, summary, rows) in groups.items():
        result["stages"][stage] = {"runs": len(rows), "reference_variant": manifest["reference_variant"],
            "candidate_variant": manifest["candidate_variant"], "variants": summary["variants"],
            "both_variants_all_repeats_pass_count": summary["both_variants_all_repeats_pass_count"],
            "missing_reports": [{key: row[key] for key in ("run_index", "input", "requested_variant", "returncode", "classification", "quality_failure")}
                                for row in rows if row["report"] is None],
            "invalid_process_reports": [{key: row[key] for key in ("run_index", "input", "requested_variant", "returncode", "classification", "quality_failure")}
                for row in rows if not valid_process_report(row, benchmark.classify(row["returncode"], row["report"]))],
            "artifacts": {suffix: sha(DATA / ("v52-" + stage + suffix)) for suffix in
                          ("-manifest.json", "-summary.json", "-runs.jsonl.gz", "-command.json", "-process.json", "-run.log")}}
    if "netlib" in groups:
        manifest, _, rows = groups["netlib"]
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[(row["input"], row["repetition"])].append(row)
        diagnostic = {"diagnostic_only": True, "audit": trajectory_audit(manifest, rows, benchmark), "non_equal_pairs": []}
        for (path, repetition), pair in grouped.items():
            audit = trajectory_audit(manifest, pair, benchmark)
            if audit["equal"]:
                continue
            category, = [key for key in ("budget_limited", "invalid", "different") if audit[key]]
            item = {"input": path, "repetition": repetition, "classification": category,
                    "raw_run_indices": [row["run_index"] for row in pair]}
            if category == "different":
                a, b = (row["report"] for row in pair)
                item["different_top_level_fields"] = sorted(key for key in set(a) | set(b) if key not in ("timings", "backend_version") and
                    json.dumps(a.get(key), sort_keys=True, allow_nan=False) != json.dumps(b.get(key), sort_keys=True, allow_nan=False))
                item["returncodes"] = [row["returncode"] for row in pair]
            diagnostic["non_equal_pairs"].append(item)
        result["netlib_trajectory_diagnostic"] = diagnostic
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.preflight:
        frozen_plan(check_inputs=False)
        result = {"passed": True, "plan_sha256": PLAN_SHA, "self_test": self_test(), "scope": "只核计划与合成协议，不读取性能raw"}
    else:
        result = verify(args.initial_only)
    destination = args.output
    if destination is None and not (args.self_test or args.preflight):
        destination = DATA / ("v52-initial-verification.json" if args.initial_only else "v52-verification.json")
    if destination is not None:
        with destination.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "run_count", "gate", "self_test") if key in result}, ensure_ascii=False))
