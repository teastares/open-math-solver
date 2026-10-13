"""独立核验阶段诊断的来源、计时守恒、65次记录与52对轨迹；不编译或求解。"""
import argparse
import collections
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
WORK = ROOT / ".tools/v47-profile-fixed"
PRODUCTION_SHA = "e3d9f917f96fe3332b84faf8c160b1377d3ee5c1640ec3b34114c009bbe74c98"
DIAGNOSTIC_SHA = "751368be43679154f1a3c35b8f93af6404a6c4645ca3e7426343d9982e2b165e"
PLAN_SHA = "a618c8d629da0bd630154aa8c278eeec77159b0cbd56b7f3e08acbfe6428ca24"
ARTIFACTS_SHA = "2e141b1cad02c16b6a265ee0d631870355f24a82f0e7f9f721a72f9ff554f5da"
MODES = ("production", "off", "on0", "on1", "on2")
PREFIX = "V47_PROFILE "
FIELDS = ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(stderr, enabled, stages, benchmark, partial=False):
    lines = [line for line in stderr.splitlines() if line.startswith(PREFIX)]
    require(len(lines) == int(enabled) or (partial and not lines), "开启/关闭计时报告数量")
    if not lines:
        return None
    payload = benchmark.strict_json(lines[0][len(PREFIX):])
    require(type(payload["schema"]) is int and payload["schema"] == 1 and
            all(payload[key] is True for key in ("conserved", "stack_empty", "counts_closed")), "计时报告完整闭合")
    actual = payload["stages"]
    require(set(actual) == set(stages), "计时阶段覆盖")
    for name, entry in actual.items():
        require(set(entry) == set(FIELDS) and all(type(entry[key]) is int and 0 <= entry[key] < 2**63 for key in FIELDS),
                "计时纳秒和计数必须为非负整数：" + name)
        require(entry["started"] == entry["completed"] + entry["unwound"] and
                entry["exclusive_ns"] <= entry["inclusive_ns"], "逐阶段退出/时间守恒：" + name)
    exclusive = sum(entry["exclusive_ns"] for entry in actual.values())
    starts = sum(entry["started"] for entry in actual.values())
    require(actual["total"]["started"] == 1 and exclusive == actual["total"]["inclusive_ns"], "全运行独占时间守恒")
    require(type(payload["clock_reads"]) is int and payload["clock_reads"] == 2 * starts and
            type(payload["exclusive_sum_ns"]) is int and type(payload["total_inclusive_ns"]) is int and
            payload["exclusive_sum_ns"] == payload["total_inclusive_ns"] == exclusive, "时钟调用与报告总数独立复算")
    return actual


def sources():
    plan, manifest = read(DATA / "v47-plan.json"), read(DATA / "v47-profile-manifest.json")
    build = read(WORK / "build-manifest.json")
    locked = read(DATA / "ipm-v46-identity.json")
    correction = read(DATA / "v47-build-correction.json")
    require(sha(DATA / "v47-plan.json") == correction["plan_sha256"] == PLAN_SHA and
            type(correction["lp_runs_before_correction"]) is int and correction["lp_runs_before_correction"] == 0,
            "原科学计划不变，构建修订发生在首次LP运行前")
    require(plan["production_sha256"] == locked["solver_sha256"] == PRODUCTION_SHA, "生产程序身份")
    require(sha(DATA / "ipm-v46-identity.json") == plan["v46_identity_sha256"] == build["v46_identity"]["sha256"], "历史身份锚点")
    require(sha(DATA / "v47-run-profile-attempt1.py") == plan["runner_sha256"] and
            sha(DATA / "v47-profile-build-attempt1.py") == plan["builder_sha256"] and
            sha(DATA / "v47-run-profile.py") == correction["runner_sha256"] and
            sha(DATA / "v47-profile-build.py") == correction["builder_sha256"] == build["builder_sha256"], "原始及修订诊断脚本身份")
    require(sha(DATA / "v47-build-attempt1.log") == correction["first_log_sha256"] and
            sha(Path(correction["first_link_log"])) == correction["first_link_log_sha256"], "首次失败证据保留")
    failed_link = read(Path(correction["first_link_log"]))
    require(failed_link["returncode"] != 0 and failed_link["wall_timeout"] is False and
            "-Wl,-t" in failed_link["command"] and "unknown argument: -t" in failed_link["stderr"], "平台链接失败原因")
    require(manifest["plan_sha256"] == sha(DATA / "v47-plan.json") and
            manifest["build_correction_sha256"] == sha(DATA / "v47-build-correction.json") and
            manifest["build_manifest_sha256"] == sha(WORK / "build-manifest.json"), "运行清单冻结计划与构建")
    require(build["diagnostic_only"] is True and build["production_solver_sha256"] == PRODUCTION_SHA and
            build["frozen_files"] == locked["files"] and len(locked["files"]) == 90, "诊断范围与90份历史来源")
    archive_path = DATA / "ipm-v46-source.zip"
    require(sha(archive_path) == locked["source_zip_sha256"] == build["frozen_archive"]["sha256"], "源码包身份")
    protocols = []
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == 90 and {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()} == locked["files"], "源码包逐文件身份")
        for name in ("benchmark", "compare_lp"):
            source = archive.read("scripts/" + name + ".py")
            require(plan["quality_scripts"][name + ".py"] == hashlib.sha256(source).hexdigest(), "质量协议使用冻结源码")
            module = types.ModuleType(name)
            module.__file__ = str(ROOT / "scripts" / (name + ".py"))
            sys.modules[name] = module
            exec(compile(source, str(archive_path) + "/scripts/" + name + ".py", "exec"), module.__dict__)
            protocols.append(module)
    benchmark, compare = protocols
    require(build["implicit_link_inputs_verified"] is False and "link_trace_after_and_final" not in build,
            "不将显式依赖身份扩张为隐式运行库身份保证")
    artifacts = DATA / "v47-profile-build-artifacts.zip"
    require(sha(artifacts) == ARTIFACTS_SHA, "诊断辅助归档包身份")
    with zipfile.ZipFile(artifacts) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 72 and all(Path(name).suffix in (".json", ".cpp", ".hpp", ".txt", ".d")
                and not Path(name).is_absolute() and ".." not in Path(name).parts for name in names), "72份辅助来源与日志范围")
        require(all(archive.read(name) == (WORK / name).read_bytes() for name in names), "辅助归档逐文件等同实际构建记录")
    for group in ("outputs", "protected_before_and_after", "link_explicit_before_and_after", "compile_inputs_after_and_final"):
        require(build[group], "构建证据集合不可为空：" + group)
        for name, digest in build[group].items():
            require(sha(Path(name)) == digest, "构建证据身份：" + name)
    require(build["executable_sha256"] == DIAGNOSTIC_SHA and
            manifest["executables"] == {"production": PRODUCTION_SHA, "diagnostic": DIAGNOSTIC_SHA} and
            sha(Path(manifest["production"])) == PRODUCTION_SHA and sha(Path(manifest["diagnostic"])) == build["executable_sha256"],
            "实际运行双程序身份")
    require(Path(manifest["production"]) == ROOT / ".tools/solver-ipm-v46.exe" and
            Path(manifest["diagnostic"]) == WORK / "solver-profile.exe", "运行程序路径")
    builder = runpy.run_path(str(DATA / "v47-profile-build.py"), run_name="v47_frozen_builder_for_verification")
    prior_builder = runpy.run_path(str(DATA / "v47-profile-build-attempt1.py"), run_name="v47_original_builder_for_verification")
    require(all(builder[key] == prior_builder[key] for key in ("HEADER", "SELF_MAIN", "SELF_CROSS", "STAGES")),
            "平台修订未更改计时器与双TU自测")
    require((WORK / "v47-profile.hpp").read_text(encoding="utf-8") == builder["HEADER"] and
            build["stages"] == list(builder["STAGES"]), "共享计时header与阶段")
    transformations = read(WORK / "transforms.json")
    with zipfile.ZipFile(archive_path) as archive:
        for unit, filename in (("ipm", "ipm.cpp"), ("normal", "normal_cholesky.cpp")):
            source = archive.read("src/" + filename).decode("utf-8").replace("\r\n", "\n")
            patched, changes = builder["transform"](source, unit)
            require(prior_builder["transform"](source, unit) == (patched, changes), "平台修订未更改插桩")
            saved = transformations[unit]
            require(changes == saved["changes"] and saved["reversible"] is True and
                    (WORK / (unit + "-profile.cpp")).read_text(encoding="utf-8") == patched and
                    sha(Path(saved["patched"])) == saved["patched_sha256"], "插桩锚点可重复生成：" + unit)
    for filename, source in (("self-main.cpp", builder["SELF_MAIN"]), ("self-cross.cpp", builder["SELF_CROSS"])):
        require((WORK / filename).read_text(encoding="utf-8") == source, "双TU自测来源")
    self_tests = {}
    for enabled in (False, True):
        row = read(WORK / ("self-test-" + str(int(enabled)) + ".json"))
        require(row["returncode"] == 0 and row["wall_timeout"] is False and
                benchmark.strict_json(row["stdout"]) == {"passed": True, "enabled": enabled}, "计时器实际自测")
        stages = profile(row["stderr"], enabled, build["stages"], benchmark)
        if not enabled:
            require(row["stderr"] == "", "关闭计时的双TU自测无诊断输出")
        if enabled:
            require(stages["total"] == dict(zip(FIELDS, (110, 20, 1, 1, 0))) and
                    stages["canonicalize"] == dict(zip(FIELDS, (110, 50, 3, 2, 1))) and
                    stages["normal_setup"] == dict(zip(FIELDS, (60, 40, 2, 1, 1))), "独立假钟嵌套/异常黄金结果")
            require(all(entry == dict.fromkeys(FIELDS, 0) for key, entry in stages.items()
                        if key not in ("total", "canonicalize", "normal_setup")), "假钟其他阶段为零")
        self_tests[str(enabled)] = {"raw_sha256": sha(WORK / ("self-test-" + str(int(enabled)) + ".json")), "passed": True}
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <= datetime.datetime.fromisoformat(correction["created_at_utc"])
            <= datetime.datetime.fromisoformat(build["created_at_utc"])
            <= datetime.datetime.fromisoformat(manifest["started_at_utc"]), "计划、构建、测量冻结顺序")
    return plan, manifest, build, benchmark, compare, self_tests


def runs(plan, manifest, build, benchmark, compare):
    require(plan["runs"] == 65 and plan["modes"] == list(MODES) and
            (plan["time_limit"], plan["wall_timeout"]) == (5, 35), "固定运行数量与预算")
    development, digest = compare.load_models(ROOT / "benchmarks/development.txt")
    require(plan["development_list_sha256"] == digest and plan["models"][:8] == development, "八开发输入")
    extras = ("pilot87", "maros-r7", "fit1p", "dfl001", "greenbea")
    require(len(plan["models"]) == 13 and tuple(Path(model["input"]).stem for model in plan["models"][8:]) == extras, "五个预声明额外输入")
    require([model["model_index"] for model in plan["models"]] == list(range(13)), "独立输入索引")
    for model in plan["models"]:
        require(model["hash_error"] is None and sha(Path(model["input"])) == model["sha256"], "输入身份")
    expected_options = ["--backend", "ipm", "--time-limit", "5", "--threads", "1", "--seed", "0",
                        "--iteration-limit", "200", "--scaling-passes", "5", "--ipm-method", "homogeneous",
                        "--ipm-linear-accuracy", "strict", "--ipm-direction-retries", "0"]
    for key, value in read(DATA / "v46-plan.json")["shared_ipm"].items():
        expected_options += ["--ipm-" + key, value]
    require(plan["options"] == expected_options, "冻结HSD strict共同命令")
    require(manifest["environment"] == dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"), "线程环境")
    rows = [benchmark.strict_json(line) for line in gzip.decompress((DATA / "v47-profile-runs.jsonl.gz").read_bytes()).splitlines()]
    schedule = [(model, mode) for model in plan["models"] for mode in
                MODES[model["model_index"] % 5:] + MODES[:model["model_index"] % 5]]
    require(len(rows) == len(schedule) == 65, "原始记录必须完整65条")
    profiles, by_model = {}, {}
    for index, (row, (model, mode)) in enumerate(zip(rows, schedule)):
        require((row["index"], row["input"], row["mode"]) == (index, model["input"], mode), "交替串行顺序")
        executable = manifest["production"] if mode == "production" else manifest["diagnostic"]
        expected_hash = manifest["executables"]["production" if mode == "production" else "diagnostic"]
        require(row["command"] == [executable, "solve", model["input"], *plan["options"]], "逐条实际命令")
        require(row["model_sha256"] == row["model_sha256_after"] == model["sha256"] and
                row["solver_sha256_before"] == row["solver_sha256_after"] == expected_hash and
                row["digest_errors"] == [None] * 4, "逐次真实模型/程序前后哈希")
        report = row["report"]
        wall_timeout = row["error"] == "wall_timeout"
        if not wall_timeout:
            require(row["error"] is None and isinstance(report, dict) and
                    benchmark.strict_json(row["stdout"]) == report and
                    benchmark.classify(row["returncode"], report) == row["classification"] and
                    row["classification"] in ("resolved", "incomplete", "verification_failed"), "完整有效原始报告")
            require(report["backend"] == "ipm" and report["backend_version"] == "ipm-cpu-0.40.0", "诊断保留生产版本")
            options = report["options"]
            require((options["time_limit"], options["threads"], options["seed"]) == (5, 1, 0), "报告共同预算")
            require(options["ipm"] == {
                "method": "homogeneous", "max_iterations": 200, "scaling_passes": 5,
                "eliminate_singletons": True, "dynamic_regularization": False, "equilibrate_kkt": True,
                "linear_accuracy": "strict", "central_corrections": 0, "direction_retries": 0,
                "krylov_history_size": 0, "linear_solver": "normal_augmented", "eliminate_fixed_variables": True,
                "presolve_singleton_rows": False, "eliminate_presolved_fixed_variables": False, "inexact_prefilter": False}, "报告所有IPM选项")
        else:
            require(row["returncode"] is None, "墙钟终止保留空退出码")
        numerical = compare.numerical_seconds(row)
        failure = compare.quality_failure(row, model, row["model_sha256_after"], expected_hash, row["solver_sha256_after"])
        if failure is None and (numerical > 5 or row["elapsed_wall_seconds"] > 35):
            failure = "budget_exceeded"
        require(row["numerical_seconds"] == numerical and row["quality_failure"] == failure and
                row["quality_pass"] is (failure is None), "逐条质量与预算重算")
        require(type(row["elapsed_wall_seconds"]) in (int, float) and math.isfinite(row["elapsed_wall_seconds"]) and
                row["elapsed_wall_seconds"] > 0, "实际墙钟有效")
        parsed = profile(row["stderr"], mode.startswith("on"), build["stages"], benchmark, partial=wall_timeout)
        profiles[index] = None if wall_timeout else parsed
        by_model.setdefault(row["input"], {})[mode] = row
    def limited(row):
        return (row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or
                row["elapsed_wall_seconds"] > 35 or row["numerical_seconds"] is not None and row["numerical_seconds"] > 5)
    expected_models, totals, pairs = [], {}, []
    for model in plan["models"]:
        group = by_model[model["input"]]
        require(set(group) == set(MODES), "每输入完整五模式")
        comparisons = {}
        for mode in ("off", "on0", "on1", "on2"):
            row = group[mode]
            reference_mode = "production" if mode == "off" else "off"
            reference = group[reference_mode]
            left = {key: value for key, value in (reference["report"] or {}).items() if key != "timings"}
            right = {key: value for key, value in (row["report"] or {}).items() if key != "timings"}
            equal = left == right and row["returncode"] == reference["returncode"]
            constrained = limited(reference) or limited(row)
            require(equal or constrained, "非预算受限的完整非时间报告/退出码改变")
            comparisons[mode] = {"equal": equal, "budget_limited": constrained}
            pairs.append({"model": Path(model["input"]).stem, "reference": reference_mode, "candidate": mode,
                          "equal": equal, "budget_limited": constrained})
        active = [profiles[group[mode]["index"]] for mode in ("on0", "on1", "on2")]
        stages = {}
        if all(value is not None for value in active):
            for name in build["stages"]:
                stages[name] = {key: statistics.median(value[name][key] for value in active) for key in FIELDS}
                calls = stages[name]["started"]
                stages[name]["ratio_of_medians_seconds_per_started_call"] = stages[name]["inclusive_ns"] / 1e9 / calls if calls else None
                total = totals.setdefault(name, {"inclusive_ns": 0, "exclusive_ns": 0})
                for key in total:
                    total[key] += stages[name][key]
        expected_models.append({"input": model["input"], "comparisons": comparisons, "median_stages": stages,
            "runs": {mode: {"status": (row["report"] or {}).get("status"), "quality_pass": row["quality_pass"],
                     "numerical_seconds": row["numerical_seconds"], "wall_seconds": row["elapsed_wall_seconds"],
                     "iterations": (row["report"] or {}).get("ipm_iterations"), "budget_limited": limited(row)}
                     for mode, row in group.items()}})
    summary = read(DATA / "v47-profile-summary.json")
    require(summary == {"schema": 1, "plan_sha256": sha(DATA / "v47-plan.json"),
            "raw_sha256": sha(DATA / "v47-profile-runs.jsonl.gz"), "runs": 65, "models": expected_models,
            "sum_of_model_stage_medians": totals,
            "qualification": "包含计时互相嵌套；逐阶段中位数不保证加总守恒，守恒仅逐次原始记录验证。异常展开表示作用域被异常退出，不等同于整个LP失败。诊断程序耗时不计入公开性能成绩。"}, "全部汇总独立重建")
    require(len(pairs) == 52, "13输入各四对轨迹")
    cohorts = {}
    for name, members in (("development", expected_models[:8]),
                          ("all_five_modes_quality_pass", [model for model in expected_models
                            if all(row["quality_pass"] for row in model["runs"].values())]),
                          ("all", expected_models)):
        stage_totals = {stage: sum(model["median_stages"][stage]["inclusive_ns"] for model in members)
                        for stage in build["stages"]}
        cohorts[name] = {"models": len(members), "sum_of_model_inclusive_ns_medians": stage_totals,
                         "numeric_factor_share": stage_totals["numeric_factor"] / stage_totals["total"],
                         "gram_assembly_share": stage_totals["gram_assembly"] / stage_totals["total"]}
    return {"records": 65, "pairs": 52, "non_budget_limited_pairs_equal": sum(not pair["budget_limited"] for pair in pairs),
            "budget_limited_pairs": [pair for pair in pairs if pair["budget_limited"]],
            "profiles_checked": sum(value is not None for value in profiles.values()),
            "quality_pass_counts": dict(collections.Counter(row["mode"] for row in rows if row["quality_pass"])),
            "status_counts": dict(collections.Counter((row["report"] or {}).get("status") for row in rows)),
            "cohort_costs": cohorts,
            "raw_sha256": sha(DATA / "v47-profile-runs.jsonl.gz"), "summary_sha256": sha(DATA / "v47-profile-summary.json"),
            "complete_summary_recomputed": True, "timing_scope": "逐次纳秒守恒；中位数聚合不强制守恒；不作为性能成绩。"}


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
    require(not broken and own >= 2, "本地断链或缺少实验/变更记录最终链接：" + str(broken))
    return {"markdown_files_checked": len(paths), "local_links_checked": links, "verification_report_links": own, "broken_links": broken}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = DATA / ("v47-profile-verification-preflight.json" if args.preflight else "v47-profile-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False,
              "final": not args.preflight, "script_sha256": sha(Path(__file__))}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        plan, manifest, build, benchmark, compare, self_tests = sources()
        result.update(production_sha256=PRODUCTION_SHA, diagnostic_sha256=build["executable_sha256"],
                      plan_sha256=sha(DATA / "v47-plan.json"), build_manifest_sha256=sha(WORK / "build-manifest.json"),
                      source_files_verified=90, stages_verified=len(build["stages"]), self_tests=self_tests,
                      build_artifacts_sha256=ARTIFACTS_SHA, build_artifact_files_verified=72,
                      build_correction_sha256=sha(DATA / "v47-build-correction.json"),
                      implicit_link_inputs_verified=False,
                      build_evidence_counts={key: len(build[key]) for key in ("outputs", "protected_before_and_after", "link_explicit_before_and_after", "compile_inputs_after_and_final")},
                      runs=runs(plan, manifest, build, benchmark, compare),
                      runtime_scope="只读重核既有来源、计时器自测和结果，不编译、不调用LP、不重新测量。")
        if not args.preflight:
            result.update(documents(output))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "final": result["final"], "runs": result["runs"]["records"],
                      "pairs": result["runs"]["pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
