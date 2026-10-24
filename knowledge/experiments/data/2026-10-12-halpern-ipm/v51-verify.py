"""独立核验第48轮 BLAS 修复的双程序来源、正确性、稳定性和完整 LP 评分。"""

import argparse
import collections
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
PLAN_SHA = "1ceb01fda1ac152d6d2753dfa7f38bc16d8a939b6d2602f4e567a9d1cbf25219"
PARENT_IDENTITY_SHA = "f37d8020a57c93130c213eb46b3c88f233e0a280bc4d47a33d8ff8cdf055f521"
HELPER_SHA = "f46fc6dbd8d5aab9e6537a05958796a79c600e19ec835d1a48da5c98333fb9b1"
STAGES = ("development", "probe", "netlib", "highs")
sys.dont_write_bytecode = True

# 只复用锁定版本的 JSON、计数、参数等无副作用检查；不调用旧轮来源或轨迹核验。
helper_path = DATA / "v50-verify.py"
helper_bytes = helper_path.read_bytes()
if hashlib.sha256(helper_bytes).hexdigest() != HELPER_SHA:
    raise AssertionError("复用核验工具的冻结身份改变")
base = types.ModuleType("v51_shared_audit")
base.__file__ = str(helper_path)
exec(compile(helper_bytes, str(helper_path), "exec"), base.__dict__)
require, read, sha, flags = base.require, base.read, base.sha, base.flags
strict_json, finite, close = base.strict_json, base.finite, base.close
json, math = base.json, base.math


def frozen_plan(check_inputs=True):
    require(sha(DATA / "v51-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v51-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA == sha(DATA / "ipm-v50-identity.json"), "父程序来源身份")
    require(sha(DATA / "v51-blas-kernel-probe-result.json") == plan["evidence_sha256"], "测前内核错值证据身份")
    require((plan["reference_version"], plan["candidate_version"], plan["reference_accuracy"], plan["candidate_accuracy"]) ==
            (50, 51, "homogeneous_repair", "homogeneous_repair"), "两冻结程序使用相同 LP 政策")
    require((plan["primary_linear_solver"], plan["method"], plan["time_limit"], plan["wall_timeout"], plan["threads"],
             plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            ("normal_cholmod", "homogeneous", 5, 35, 1, 0, 200, 5), "固定算法与共同预算")
    require(plan["shared_ipm"] == {"singleton-elimination": "on", "dynamic-regularization": "off", "kkt-equilibration": "on",
        "fixed-variable-elimination": "on", "singleton-presolve": "off", "presolve-fixed-elimination": "off",
        "krylov-history-size": "0", "central-corrections": "0", "inexact-prefilter": "off"}, "共同 IPM 参数")
    require(plan["highs_reference"] == {"solver": "choose", "presolve": "choose", "crossover": "on"} and
            (plan["initial_performance_runs"], plan["conditional_full_performance_runs"], plan["diagnostic_runs"], plan["gate"]["ctest"]) ==
            (102, 364, 20, 18) and plan["no_model_specific_routing"] is True, "测量数量、门槛与 HiGHS 配置")
    require(plan["initial_stages"] == [{"name": stage, "set": stage, "linear_solver": "normal_cholmod"}
                                      for stage in STAGES[:2]], "预声明初始阶段")
    parent_plan = read(DATA / "v50-plan.json")
    require(plan["sets"] == parent_plan["sets"], "沿用冻结输入集合，不能删减失败模型")
    for name, count, repeats in (("development", 11, 3), ("probe", 6, 3), ("netlib", 91, 1)):
        group = plan["sets"][name]
        require(group["repeats"] == repeats and len(group["models"]) == count and
                [item["model_index"] for item in group["models"]] == list(range(count)), "集合规模与顺序")
        path = Path(group["models_file"])
        require(sha(path) == group["models_file_sha256"], "模型列表身份")
        actual = [str((path.parent / line.strip()).resolve()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(actual == [item["input"] for item in group["models"]] and len(set(actual)) == count, "输入唯一性")
        if check_inputs:
            for item in group["models"]:
                require(item["hash_error"] is None and sha(item["input"]) == item["sha256"], "输入身份")
    return plan


def sources(plan):
    identities = {version: read(DATA / f"ipm-v{version}-identity.json") for version in (50, 51)}
    identity, parent = identities[51], identities[50]
    require(identity["plan_sha256"] == PLAN_SHA and datetime.datetime.fromisoformat(identity["created_at_utc"]) >=
            datetime.datetime.fromisoformat(plan["created_at_utc"]), "新程序冻结在事前计划之后")
    for version, item in identities.items():
        require(sha(ROOT / f".tools/solver-ipm-v{version}.exe") == item["solver_sha256"], "实际冻结程序身份")
    require(sha(ROOT / "build/release/solver.exe") == identity["solver_sha256"], "正式构建与候选相同")
    require(identities[50]["solver_sha256"] != identities[51]["solver_sha256"], "两个程序必须不同")
    require(set(identity["files"]) == set(parent["files"]) | {"tests/test_openblas.cpp", "third_party/openblas/NOTICE.txt",
            "third_party/openblas/patch_dgemm_stack.cmake"} and len(identity["files"]) == 95, "完整95份来源")
    snapshots = {}
    for version, item in identities.items():
        path = DATA / f"ipm-v{version}-source.zip"
        require(sha(path) == item["source_zip_sha256"], "新旧源码包身份")
        with zipfile.ZipFile(path) as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(item["files"]), "源码包真实集合")
            snapshot = {name: archive.read(name) for name in archive.namelist()}
        require(all(hashlib.sha256(content).hexdigest() == item["files"][name] for name, content in snapshot.items()), "源码成员身份")
        snapshots[version] = snapshot
    changed = {name for name in parent["files"] if parent["files"][name] != identity["files"][name]}
    require(changed == {"CMakeLists.txt", "dependencies.lock.json", "src/ipm.cpp", "third_party/openblas/CMakeLists.txt"}, "仅允许版本、补丁与回归测试构建变化")
    old_version = b'std::string ipm_version() { return "ipm-cpu-0.42.0"; }\n'
    new_version = b'std::string ipm_version() { return "ipm-cpu-0.43.0"; }\r\n'
    require(snapshots[50]["src/ipm.cpp"].count(old_version) == snapshots[51]["src/ipm.cpp"].count(new_version) == 1 and
            snapshots[51]["src/ipm.cpp"].replace(new_version, old_version) == snapshots[50]["src/ipm.cpp"],
            "LP 实现只变化一行版本字符串及该行LF到CRLF，不能放宽其他源码差异")
    for name, digest in identity["files"].items():
        require(sha(ROOT / name) == digest, "当前新来源身份：" + name)
    require(identity["compiler"] == parent["compiler"] and sha(identity["compiler"]) == identity["compiler_sha256"] == parent["compiler_sha256"], "同一编译器")
    for group, directory in (("libraries", ROOT), ("logs", DATA)):
        for name, digest in identity[group].items():
            require(sha(directory / name) == digest, "候选依赖或日志身份")
    libraries = {(ROOT / name).resolve() for name in identity["libraries"]}
    parsed = [value for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', identity["link_command"])
              if not (value := quoted or plain).startswith("-")]
    require(len(libraries) == 17 and {(ROOT / "build/release" / name).resolve() for name in parsed} == libraries and
            sum(path.name == "libopenblas.a" for path in libraries) == 1, "真实链接17库且唯一 BLAS")
    library_name = "build/openblas/install/lib/libopenblas.a"
    require(sha(ROOT / ".tools/v51-parent-libopenblas.a") == identity["parent_openblas_library_sha256"] == parent["libraries"][library_name] and
            identity["libraries"][library_name] != parent["libraries"][library_name], "保存旧库并独立冻结新库")
    lock = strict_json(snapshots[51]["dependencies.lock.json"])["openblas"]
    patch, = lock["local_patches"]
    require(lock["version"] == "0.3.33" and patch["source_sha256"] == "b64d4c34defbdd3e5cb62d259d35ee8fa488c01d90af13cb3c4c7367e5fc7e26" and
            patch["patched_sha256"] == "7ad33091ab73c53684d1b4302335ac80969ccc4e4371d4895ca5564424a84e9b" and
            identity["files"][patch["path"]] == patch["sha256"], "依赖补丁及前后源码锁")
    require(sha(ROOT / "build/openblas/openblas-0.3.33/src/solver_openblas_build" / patch["source_file"]) ==
            identity["patched_openblas_source_sha256"] == patch["patched_sha256"], "实际参与构建的补丁源身份")
    for filename, key in (("v51-run.py", "runner_sha256"), ("v51-freeze.py", "freeze_script_sha256"),
                          ("v51-check.py", "check_script_sha256"), ("v51-runner-self-check.json", "runner_self_check_sha256")):
        require(sha(DATA / filename) == identity[key], "冻结实验脚本身份")
    require(read(DATA / "v51-runner-self-check.json") == {"passed": True, "command_cases": 4, "gate_cases": 6, "solver_runs": 0}, "测前 runner 检查")
    require(b"100% tests passed out of 18" in (DATA / "iteration-ipm-v51-build.log").read_bytes(), "正式18项CTest")
    detail = (DATA / "iteration-ipm-v51-ctest-detail.log").read_bytes()
    require("公共 BLAS 数值与存储边界检查通过：19 组".encode("utf-8") in detail and
            "齐次恢复解析尝试 2 次，成功 2 次".encode("utf-8") in detail, "BLAS回归及原恢复边界测试实际执行")
    loaded = []
    for name in ("benchmark", "compare_lp"):
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / "scripts" / (name + ".py"))
        sys.modules[name] = module
        exec(compile(snapshots[51]["scripts/" + name + ".py"], str(DATA / "ipm-v51-source.zip") + "/" + name, "exec"), module.__dict__)
        loaded.append(module)
    return identities, loaded[0], loaded[1]


def verify_options(report, plan, backend, version, accuracy="homogeneous_repair"):
    if backend == "ipm":
        require(report["backend_version"] == f"ipm-cpu-0.{42 if version == 50 else 43}.0", "实际执行对应冻结程序版本")
    # 旧助手仅把版本固定为0.42；先独立核新旧版本，再对副本复用未变的完整选项检查。
    options_copy = dict(report)
    if backend == "ipm":
        options_copy["backend_version"] = "ipm-cpu-0.42.0"
    base.verify_options(options_copy, plan, "normal_cholmod", accuracy, backend)
    base.counts(report)
    require(report["model"]["is_mip"] is False, "所有输入是连续 LP")


def numerical_seconds(row):
    timing = (row.get("report") or {}).get("timings")
    if not isinstance(timing, dict) or not all(finite(timing.get(key)) and timing[key] >= 0 for key in ("solve_seconds", "verification_seconds")):
        return None
    value = timing["solve_seconds"] + timing["verification_seconds"]
    return value if finite(value) else None


def score(row, failure):
    numerical = numerical_seconds(row)
    wall = row["elapsed_wall_seconds"]
    require(finite(wall) and wall > 0, "真实有限墙钟")
    if failure is None and numerical is None:
        failure = "invalid_timings"
    if failure is None and wall > 35:
        failure = "wall_budget_exceeded"
    if failure is None and numerical > 5:
        failure = "numerical_budget_exceeded"
    return {"quality_failure": failure, "quality_pass": failure is None, "numerical_seconds": numerical,
            "wall_budget_exceeded": wall > 35, "numerical_budget_exceeded": numerical > 5 if numerical is not None else None,
            "quality_penalized_wall_seconds": min(wall, 35) if failure is None else 35,
            "quality_penalized_numerical_seconds": min(numerical, 5) if failure is None else 5}


def verify_group(stage, identities, plan, benchmark, compare):
    declared = plan["sets"]["netlib" if stage in ("netlib", "highs") else stage]
    identity = identities[51]
    stem = "v51-" + stage
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
    candidate = str(ROOT / ".tools/solver-ipm-v51.exe")
    reference_version = 51 if stage == "highs" else 50
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
        version = reference_version if index == 0 else 51
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
        version = reference_version if side == sides[0] else 51
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


def validation(identities):
    identity = identities[51]
    records = {}
    for stage in ("before", "after"):
        item = read(DATA / f"v51-public-blas-{stage}.json")
        source = ROOT / "tests/test_openblas.cpp"
        library = ROOT / (".tools/v51-parent-libopenblas.a" if stage == "before" else "build/openblas/install/lib/libopenblas.a")
        executable = ROOT / f".tools/v51-public-blas-{stage}.exe"
        command = [identity["compiler"], "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-static",
            "-I" + str(ROOT / "build/openblas/install/include/openblas"), str(source), str(library), "-o", str(executable)]
        require(item["diagnostic_only"] is True and item["plan_sha256"] == PLAN_SHA and
                item["script_sha256"] == identity["check_script_sha256"] and item["build_command"] == command and
                item["build"]["returncode"] == 0, "公共BLAS同构建参数且确实构建成功")
        require(item["source_sha256"] == item["source_sha256_after"] == identity["files"]["tests/test_openblas.cpp"] and
                item["compiler_sha256"] == identity["compiler_sha256"] and
                item["library_sha256"] == item["library_sha256_after"] == sha(library) and
                item["executable_sha256"] == sha(executable), "公共回归程序及输入前后身份")
        require("error" not in item["run"] and item["run"]["returncode"] == (1 if stage == "before" else 0), "旧库是错值失败，新库成功，超时或崩溃不能代替负对照")
        records[stage] = item
    before, after = records["before"], records["after"]
    require(before["library_sha256"] == identities[50]["libraries"]["build/openblas/install/lib/libopenblas.a"] and
            after["library_sha256"] == identity["libraries"]["build/openblas/install/lib/libopenblas.a"] and
            before["library_sha256"] != after["library_sha256"], "唯一替换库来源")
    require("M=513，N=6，K=342" in before["run"]["stderr"] and "实际=6129，预期=6132" in before["run"]["stderr"] and
            "BLAS 乘积错误或输出填充区被修改" in before["run"]["stderr"], "原库可重复公共接口错值证据")
    require("公共 BLAS 数值与存储边界检查通过：19 组" in after["run"]["stdout"] and after["run"]["stderr"] == "", "19组数值及存储回归全通过")
    require(before["run"]["stdout"].splitlines()[0] == after["run"]["stdout"].splitlines()[0], "公共回归使用相同BLAS配置与自动CPU内核")
    result = read(DATA / "v51-validation.json")
    require(result["passed"] is True and result["plan_sha256"] == PLAN_SHA and result["solver_sha256"] == identity["solver_sha256"] and
            result["public_before_sha256"] == sha(DATA / "v51-public-blas-before.json") and
            result["public_after_sha256"] == sha(DATA / "v51-public-blas-after.json") and
            result["build_log_sha256"] == identity["logs"]["iteration-ipm-v51-build.log"] and
            result["ctest_passed"] == 18 and result["script_sha256"] == identity["check_script_sha256"], "正确性门槛证据不可仅信布尔值")
    return result


def valid_process_report(row, classification):
    # 合法失败也必须遵守 CLI 退出合同，不能用 verification_failed 掩盖进程崩溃。
    expected_exit = {"resolved": 0, "incomplete": 4, "verification_failed": 3}
    return (row["error"] is None and isinstance(row["report"], dict) and
            type(row["returncode"]) is int and classification in expected_exit and
            row["returncode"] == expected_exit[classification])


def stability(identity, plan, benchmark):
    result = read(DATA / "v51-stability.json")
    path = DATA / "v51-stability-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    model, = [item for item in plan["sets"]["probe"]["models"] if Path(item["input"]).stem == "dfl001"]
    expected_flags = {"--backend": "ipm", "--time-limit": "5", "--threads": "1", "--seed": "0",
                      **base.expected_ipm_flags(plan, "normal_cholmod", "strict")}
    expected_hashes = {"solver": identity["solver_sha256"], "input": model["sha256"]}
    require(len(rows) == 20 and [row["run_index"] for row in rows] == list(range(20)), "固定20次稳定性完整保留")
    valid_count, violations = 0, 0
    for row in rows:
        require(row["command"][:3] == [str(ROOT / ".tools/solver-ipm-v51.exe"), "solve", model["input"]] and
                flags(row["command"], 3) == expected_flags and row["before"] == row["after"] == expected_hashes, "固定strict诊断命令及来源")
        report = row["report"]
        if report is not None:
            require(strict_json(row["stdout"]) == report and report["input"] == model["input"], "稳定性原始报告")
            verify_options(report, plan, "ipm", 51, "strict")
        classification = benchmark.classify(row["returncode"], report) if row["error"] is None else row["error"]
        valid = valid_process_report(row, classification)
        require(row["classification"] == classification and row["valid_report"] is valid, "合法数值失败与无效进程分别分类")
        valid_count += int(valid)
        violations += int(row["returncode"] == 3221225477)
    require(result["diagnostic_only"] is True and result["plan_sha256"] == PLAN_SHA and result["solver_sha256"] == identity["solver_sha256"] and
            result["script_sha256"] == identity["check_script_sha256"] and result["raw_sha256"] == sha(path) and
            result["run_count"] == 20 and result["valid_report_count"] == valid_count and result["access_violations"] == violations and
            result["passed"] is (valid_count == 20 and violations == 0), "稳定性门槛独立重算")
    return {**result, "status_counts": dict(collections.Counter((row["report"] or {}).get("status", "no_status") for row in rows)),
            "strict_exit_contract_audited": True,
            "runner_protocol_gap": "冻结check沿用的classify对verification_failed未核退出码；本核验独立要求合法分类/退出组合、无error且零访问违规。原记录不修改，本次真实数据重新满足严格合同。"}


def decision(groups, correctness, stable):
    lost = {}
    for stage in STAGES[:2]:
        manifest, _, rows = groups[stage]
        sides = (manifest["reference_variant"], manifest["candidate_variant"])
        grouped = collections.defaultdict(lambda: [0, 0])
        for row in rows:
            grouped[row["input"]][sides.index(row["requested_variant"])] += int(row["quality_pass"])
        lost[stage] = [Path(path).stem for path, (before, after) in grouped.items() if after < before]
    return {"development_lost": lost["development"], "probe_lost": lost["probe"], "correctness_passed": correctness,
            "stability_passed": stable, "eligible_for_netlib": not lost["development"] and not lost["probe"] and correctness and stable,
            "plan_sha256": PLAN_SHA}


def self_test():
    rejected = 0
    def fails(operation):
        nonlocal rejected
        try:
            operation()
        except (AssertionError, ValueError):
            rejected += 1
        else:
            raise AssertionError("应拒绝的合成记录被接受")
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        fails(lambda text=text: strict_json(text))
    fails(lambda: flags(["solver", "solve", "x", "--threads", "1", "--threads", "2"], 3))
    shared = base.self_test()
    require(shared["passed"], "复用的严格计数与旧解析自检")
    def row(wall, solve=None, verify=0):
        return {"elapsed_wall_seconds": wall, "report": None if solve is None else {"timings": {"solve_seconds": solve, "verification_seconds": verify}}}
    require(score(row(1), "invalid_json")["quality_penalized_wall_seconds"] == 35 and
            score(row(1), "invalid_json")["quality_penalized_numerical_seconds"] == 5, "无报告失败不能被删除或按短耗时计分")
    require(score(row(36, 1), None)["quality_failure"] == "wall_budget_exceeded" and
            score(row(1, 4.9, 0.2), None)["quality_failure"] == "numerical_budget_exceeded" and
            score(row(1, True), None)["quality_failure"] == "invalid_timings", "完整预算与非bool计时")
    for classification, code in (("resolved", 0), ("incomplete", 4), ("verification_failed", 3)):
        valid = {"error": None, "returncode": code, "report": {}}
        require(valid_process_report(valid, classification), "合法退出合同")
        require(not valid_process_report(dict(valid, returncode=3221225477), classification), "访问违规不能被报告状态掩盖")
    require(not valid_process_report({"error": "wall_timeout", "returncode": 3, "report": {}}, "verification_failed"), "超时不能算稳定")
    require(not valid_process_report({"error": None, "returncode": False, "report": {}}, "resolved"), "bool不能作为退出码")
    def group(before, after):
        manifest = {"reference_variant": "r", "candidate_variant": "c"}
        rows = [{"input": "synthetic.mps", "requested_variant": side, "quality_pass": index < count}
                for side, count in (("r", before), ("c", after)) for index in range(3)]
        return manifest, {}, rows
    cases = [(3, 3, 0, 0, True, True, True), (3, 2, 0, 3, True, True, False),
             (3, 3, 3, 2, True, True, False), (3, 3, 0, 3, False, True, False),
             (3, 3, 0, 3, True, False, False), (3, 3, 0, 3, True, True, True)]
    for a, b, c, d, correct, stable, expected in cases:
        require(decision({"development": group(a, b), "probe": group(c, d)}, correct, stable)["eligible_for_netlib"] is expected, "独立门槛合成反例")
    return {"passed": True, "negative_cases": rejected, "budget_cases": 4, "process_contract_cases": 8,
            "gate_cases": len(cases), "shared_checks": shared, "solver_runs": 0}


def verify(initial_only=False):
    plan = frozen_plan()
    identities, benchmark, compare = sources(plan)
    correct = validation(identities)
    stable = stability(identities[51], plan, benchmark)
    stages = STAGES[:2] if initial_only else STAGES
    groups = {stage: verify_group(stage, identities, plan, benchmark, compare) for stage in stages}
    gate = decision(groups, correct["passed"], stable["passed"])
    require(read(DATA / "v51-gate.json") == gate, "门槛由原始逐例质量与正确性/稳定性证据重算")
    if not initial_only:
        require(gate["eligible_for_netlib"] and sum(len(group[2]) for group in groups.values()) == 466, "完整扩集需要门槛和466条记录")
    result = {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "初始102条核验" if initial_only else "完整466条冻结实验核验", "plan_sha256": PLAN_SHA,
              "helper_sha256": HELPER_SHA, "verifier_sha256": sha(__file__), "identity_sha256": sha(DATA / "ipm-v51-identity.json"),
              "solver_sha256": identities[51]["solver_sha256"], "parent_solver_sha256": identities[50]["solver_sha256"],
              "source_files": 95, "linked_libraries": 17, "validation": correct, "stability": stable, "gate": gate,
              "run_count": sum(len(group[2]) for group in groups.values()), "stages": {}, "self_test": self_test(),
              "limits": ["公开LP验证复算既存原模型检查报告、共同容差与预算，不重新求解或重建原始矩阵。",
                         "旧库已有静默错值，不将修补前后轨迹相等设为条件；两程序实际命令仍采用相同LP算法政策。",
                         "20次稳定性是固定有限复现，合法数值失败仍是LP失败，不证明永不崩溃。"]}
    for stage, (manifest, summary, rows) in groups.items():
        result["stages"][stage] = {"runs": len(rows), "reference_variant": manifest["reference_variant"],
            "candidate_variant": manifest["candidate_variant"], "variants": summary["variants"],
            "both_variants_all_repeats_pass_count": summary["both_variants_all_repeats_pass_count"],
            "missing_reports": [{key: row[key] for key in ("run_index", "input", "requested_variant", "returncode", "classification", "quality_failure")}
                                for row in rows if row["report"] is None],
            "artifacts": {suffix: sha(DATA / ("v51-" + stage + suffix)) for suffix in
                          ("-manifest.json", "-summary.json", "-runs.jsonl.gz", "-command.json", "-process.json", "-run.log")}}
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
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "run_count", "gate", "self_test") if key in result}, ensure_ascii=False))
