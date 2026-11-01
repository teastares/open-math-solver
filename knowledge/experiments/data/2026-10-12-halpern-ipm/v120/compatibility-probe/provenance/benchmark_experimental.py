#!/usr/bin/env python3
"""评测直接维护的实验C++：机械迁移验证或冻结共同资源的条件500次性能协议。"""
import argparse
import copy
import datetime
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import build_experimental as build_tool

ROOT, DATA = build_tool.ROOT, build_tool.DATA
COMPAT = ("compatibility-development", "compatibility-probe")
PERFORMANCE_STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
STAGES = (*PERFORMANCE_STAGES, "migration")
RUNNER_SHA = "ecd1f8eff16d24a74474817864d8fb86499d9174a82339beb3ea5ada134f8ccc"
PRIVATE_CLOCKS = ["presolve_marker/timings", "presolve_marker/oracle/total_seconds", "presolve_marker/oracle/events/*/timings"]
DYNAMIC_BUDGET = "presolve_marker/native/time_limit"
sys.dont_write_bytecode = True
sha, read, save, require, equal = (getattr(build_tool, name) for name in ("sha", "read", "save", "require", "equal"))


def protocol(plan):
    path = DATA / plan["protocol_file"]
    require(sha(path) == plan["protocol_sha256"] and sha(DATA / "v100-run.py") == RUNNER_SHA, "冻结数值协议或纯验证运行器变化")
    spec = importlib.util.spec_from_file_location("experimental_readonly_v100_protocol", DATA / "v100-run.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    frozen = read(path)
    if plan["new_algorithm_round"]:
        for field in ("sets", "time_limit", "wall_timeout", "threads", "seed", "max_iterations", "scaling_passes",
            "shared_ipm", "method", "candidate_accuracy", "candidate_linear_solver", "candidate_objective_scaling",
            "candidate_backend_version", "reference_accuracy", "reference_linear_solver", "reference_objective_scaling",
            "reference_backend_version", "compatibility_accuracy", "compatibility_linear_solver", "highs_reference",
            "expected_report_options", "compatibility_excluded_report_fields", "stage_order", "maximum_solver_runs"):
            require(equal(plan[field], frozen[field]), "性能共同协议不能改变冻结字段：" + field)
        require(plan["stage_order"] == list(PERFORMANCE_STAGES) and plan["maximum_solver_runs"] == 500 and
            plan["compatibility_excluded_report_fields"] == ["timings"], "固定500次阶段和完整报告比较边界不符")
        require(sha(DATA / "v66-run.py") == module.HELPER_SHA, "实际coverage/SGM扩集门槛函数来源变化")
    else:
        require(plan["verification"]["public_report_excluded_fields"] == ["timings"] and
            plan["verification"]["private_excluded_paths"] == PRIVATE_CLOCKS and
            plan["verification"]["dynamic_private_differences_record_only"] == [DYNAMIC_BUDGET], "预先冻结的报告比较边界变化")
    return module, frozen


def verify_work(work, plan_path):
    identity = read(work / "identity.json")
    require(Path(identity["build_manifest"]) == work / "build-manifest.json" and
        sha(identity["build_manifest"]) == identity["build_manifest_sha256"], "独立构建清单身份变化")
    build_tool.verify(identity["protected"])
    manifest = read(identity["build_manifest"])
    require(manifest["source_generation"] is False and manifest["production_changes"] is False and
        manifest["plan_sha256"] == sha(plan_path) and sha(identity["candidate"]) == identity["candidate_sha256"] == manifest["executable_sha256"], "直接源码构建或候选程序身份不符")
    # 活动canonical仅是构建时来源记录；此处只复核不可变快照及真实依赖。
    build_tool.verify(manifest["source_snapshot"])
    require(not any(str((ROOT / name).resolve()) in identity["protected"] for name in manifest["source_origins_at_build"]), "历史身份不得要求活动canonical永远不变")
    return identity, manifest


def public_view(report):
    result = copy.deepcopy(report)
    if type(result) is dict:
        result.pop("timings", None)
    return result


def private_view(row):
    result = {key: copy.deepcopy(row.get(key)) for key in ("presolve_marker", "coupled_marker", "energy_marker")}
    primary = result["presolve_marker"]
    if type(primary) is dict:
        primary.pop("timings", None)
        oracle = primary.get("oracle")
        if type(oracle) is dict:
            oracle.pop("total_seconds", None)
            for event in oracle.get("events", []):
                if type(event) is dict:
                    event.pop("timings", None)
    return result


def differences(left, right, prefix=""):
    if equal(left, right):
        return []
    if type(left) is dict and type(right) is dict:
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = prefix + "/" + key if prefix else key
            if key not in left or key not in right:
                result.append(dict(path=path, left_present=key in left, right_present=key in right, left=left.get(key), right=right.get(key)))
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    if type(left) is list and type(right) is list and len(left) == len(right):
        return [item for index, (a, b) in enumerate(zip(left, right)) for item in differences(a, b, prefix + "/" + str(index))]
    return [dict(path=prefix, left=left, right=right)]


def pair_result(a, b, migration, runner):
    public = differences(public_view(a["report"]), public_view(b["report"]))
    process_equal = all(equal(a.get(key), b.get(key)) for key in ("returncode", "classification", "error", "status"))
    valid = runner.valid_process(a) and runner.valid_process(b)
    private = differences(private_view(a), private_view(b)) if migration else []
    dynamic = [item for item in private if item["path"] == DYNAMIC_BUDGET]
    unexpected = [item for item in private if item["path"] != DYNAMIC_BUDGET]
    quality_equal = all(equal(a.get(key), b.get(key)) for key in ("quality_pass", "quality_failure"))
    return dict(passed=valid and process_equal and not public and not unexpected and quality_equal,
        valid_processes=valid, public_equal_excluding_timings=not public, process_equal=process_equal,
        quality_equal=quality_equal, reference_quality_pass=a["quality_pass"], candidate_quality_pass=b["quality_pass"],
        private_comparison_applicable=migration, private_equal_excluding_clocks=not private if migration else None,
        public_differences=public, private_strict_differences=private, dynamic_budget_differences=dynamic,
        unexpected_private_differences=unexpected)


def run_migration(work, output, stage, plan_path):
    require(work.is_relative_to(ROOT) and output.is_relative_to(ROOT), "构建和证据目录必须解析到工作区")
    plan = read(plan_path)
    require(plan["schema"] == 1 and plan["new_algorithm_round"] is False and plan["production_changes"] is False, "本入口只验证机械迁移，不接受新增算法性能轮")
    runner, frozen = protocol(plan)
    identity, build = verify_work(work, plan_path)
    settings = [item for item in plan["verification"]["stages"] if item["stage"] == stage]
    require(len(settings) == 1, "评测阶段不在冻结计划中")
    settings = settings[0]
    migration = stage == "migration"
    protocol_stage = "netlib" if migration else stage
    models = runner.specification(protocol_stage, frozen)["models"]
    variants = ("v100_on", "experimental_on") if migration else ("frozen_v66", "diagnostic_off")
    require(len(models) == settings["pairs"] and settings["runs"] == 2 * len(models), "固定模型对数不符")
    reference = ROOT / (plan["algorithm_parent_executable"] if migration else ".tools/solver-ipm-v66.exe")
    reference_sha = plan["algorithm_parent_executable_sha256"] if migration else plan["production_sha256"]
    require(sha(reference) == reference_sha, "冻结参考程序变化")
    stage_dir = output / stage
    require(not stage_dir.exists(), "每个阶段必须使用尚不存在的新证据子目录")
    stage_dir.mkdir(parents=True)
    provenance = stage_dir / "provenance"; provenance.mkdir()
    for source, name in ((Path(__file__), "benchmark_experimental.py"), (Path(build_tool.__file__), "build_experimental.py"),
        (plan_path, "plan.json"), (DATA / plan["protocol_file"], "protocol.json"), (DATA / "v100-run.py", "protocol-runner.py")):
        (provenance / name).write_bytes(source.read_bytes())
    protected = dict(identity["protected"])
    protected[str(work / "identity.json")] = sha(work / "identity.json")
    protected[str(reference)] = reference_sha
    protected.update(build_tool.hashes(provenance.iterdir()))
    for model in models:
        require(model["hash_error"] is None and sha(model["input"]) == model["sha256"], "冻结模型身份变化")
    manifest = dict(schema=1, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), stage=stage,
        purpose="机械迁移等价验证，非新增算法性能轮", plan_sha256=sha(plan_path), work=str(work), build_identity_sha256=sha(work / "identity.json"),
        models=models, repeats=1, variants=list(variants), runs=settings["runs"], reference=str(reference), reference_sha256=reference_sha,
        candidate=identity["candidate"], candidate_sha256=identity["candidate_sha256"], time_limit=5, wall_timeout=35, threads=1, seed=0,
        public_report_excluded_fields=["timings"], private_excluded_paths=PRIVATE_CLOCKS,
        dynamic_private_differences_record_only=[DYNAMIC_BUDGET], protected=protected)
    save(stage_dir / "manifest.json", manifest)
    rows = []
    raw_path = stage_dir / "runs.jsonl.gz"
    with gzip.open(raw_path, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in runner.compare_lp.schedule(models, 1, variants):
            candidate = variant == variants[1]
            executable = Path(identity["candidate"]) if candidate else reference
            digest = identity["candidate_sha256"] if candidate else reference_sha
            protocol_variant = "diagnostic_on" if migration else "diagnostic_off" if candidate else "frozen_v66"
            before = sha(executable)
            with runner.solver_environment(migration):
                row = runner.benchmark.run_one(runner.options_for(protocol_stage, protocol_variant, executable, frozen),
                    Path(model["input"]), 0, len(rows))
            parsed = runner.inspect_markers(row["stderr"], protocol_variant)
            row.update(dict(zip(("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error", "energy_marker", "energy_marker_error"), parsed)))
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position, requested_variant=variant,
                protocol_variant=protocol_variant, frozen_model_sha256=model["sha256"], model_sha256_after=sha(model["input"]),
                solver_sha256_before=before, solver_sha256_after=sha(executable), frozen_solver_sha256=digest,
                presolve_environment={runner.ENVIRONMENT: "1"} if migration else {},
                configuration_matches=runner.configuration_matches(row["report"], protocol_stage, protocol_variant, model, frozen))
            row.update(runner.quality_fields(row, model, digest, frozen))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
            rows.append(row)
            print(json.dumps(dict(stage=stage, index=row["run_index"], model=Path(model["input"]).stem, variant=variant,
                status=row["status"], quality_pass=row["quality_pass"]), ensure_ascii=False), flush=True)
    require(len(rows) == settings["runs"], "实际运行数不符；不能删失败或补跑替换")
    pairs = []
    for model in models:
        chosen = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
        require(set(chosen) == set(variants), "固定模型缺失一侧原始记录")
        pairs.append(dict(model_index=model["model_index"], input=model["input"],
            **pair_result(chosen[variants[0]], chosen[variants[1]], migration, runner)))
    summary = dict(schema=1, passed=all(pair["passed"] for pair in pairs), stage=stage, public_lp_runs=len(rows),
        plan_sha256=sha(plan_path), raw_sha256=sha(raw_path), pairs=pairs,
        public_equal_pairs=sum(pair["public_equal_excluding_timings"] for pair in pairs),
        unexpected_private_difference_pairs=sum(bool(pair["unexpected_private_differences"]) for pair in pairs),
        dynamic_budget_difference_pairs=sum(bool(pair["dynamic_budget_differences"]) for pair in pairs),
        valid_process_count=sum(runner.valid_process(row) for row in rows),
        variants={variant: dict(runs=sum(row["requested_variant"] == variant for row in rows),
            quality_pass_count=sum(row["requested_variant"] == variant and row["quality_pass"] for row in rows)) for variant in variants},
        performance_gate_applied=False, highs_runs=0, new_algorithm_round=False)
    save(stage_dir / "summary.json", summary)
    build_tool.verify(protected)
    require(sha(Path(__file__)) == sha(provenance / "benchmark_experimental.py") and
        sha(build_tool.__file__) == sha(provenance / "build_experimental.py"), "评测期间脚本被并发修改")
    print(json.dumps(dict(passed=summary["passed"], stage=stage, public_lp_runs=len(rows),
        public_equal_pairs=summary["public_equal_pairs"], unexpected_private_difference_pairs=summary["unexpected_private_difference_pairs"],
        summary=str(stage_dir / "summary.json")), ensure_ascii=False))
    return summary


def performance_summary(stage, rows, models, repeats, variants, runner, frozen, plan_sha, raw_sha):
    summary = runner.compare_lp.summarize(rows, models, repeats, frozen["wall_timeout"], frozen["time_limit"], variants)
    summary.update(schema=1, stage=stage, new_algorithm_round=True, plan_sha256=plan_sha, raw_sha256=raw_sha,
        valid_process_count=sum(runner.valid_process(row) for row in rows),
        presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows),
        coupled_marker_error_count=sum(row["coupled_marker_error"] is not None for row in rows),
        energy_marker_error_count=sum(row["energy_marker_error"] is not None for row in rows))
    return summary


def migration_summary(stage, rows, models, variants, runner, plan_sha, raw_sha):
    pairs = []
    for model in models:
        chosen = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
        require(set(chosen) == set(variants), "固定模型缺失一侧原始记录")
        pairs.append(dict(model_index=model["model_index"], input=model["input"],
            **pair_result(chosen[variants[0]], chosen[variants[1]], stage == "migration", runner)))
    return dict(schema=1, passed=all(pair["passed"] for pair in pairs), stage=stage, public_lp_runs=len(rows),
        plan_sha256=plan_sha, raw_sha256=raw_sha, pairs=pairs,
        public_equal_pairs=sum(pair["public_equal_excluding_timings"] for pair in pairs),
        unexpected_private_difference_pairs=sum(bool(pair["unexpected_private_differences"]) for pair in pairs),
        dynamic_budget_difference_pairs=sum(bool(pair["dynamic_budget_differences"]) for pair in pairs),
        valid_process_count=sum(runner.valid_process(row) for row in rows),
        variants={variant: dict(runs=sum(row["requested_variant"] == variant for row in rows),
            quality_pass_count=sum(row["requested_variant"] == variant and row["quality_pass"] for row in rows)) for variant in variants},
        performance_gate_applied=False, highs_runs=0, new_algorithm_round=False)


def archive(stage, output, plan_path):
    """只重解析已有原始记录，不启动编译器或求解程序，也不读取活动canonical。"""
    plan = read(plan_path)
    runner, frozen = protocol(plan)
    stage_dir = output / stage
    manifest = read(stage_dir / "manifest.json")
    require(manifest["stage"] == stage and manifest["plan_sha256"] == sha(plan_path) == sha(stage_dir / "provenance/plan.json"), "阶段计划身份变化")
    identity, _ = verify_work(Path(manifest["work"]), plan_path)
    require(manifest["build_identity_sha256"] == sha(Path(manifest["work"]) / "identity.json"), "阶段构建身份变化")
    build_tool.verify(manifest["protected"])
    performance = plan["new_algorithm_round"]
    selected = runner.specification(stage if performance else "netlib" if stage == "migration" else stage, frozen)
    models, repeats = selected["models"], selected["repeats"] if performance else 1
    variants = runner.variants(stage) if performance else ("v100_on", "experimental_on") if stage == "migration" else ("frozen_v66", "diagnostic_off")
    require(equal(manifest["models"], models) and manifest["repeats"] == repeats and manifest["variants"] == list(variants) and
        all(manifest[key] == frozen[key] for key in ("time_limit", "wall_timeout", "threads", "seed")), "阶段集合/参数/次数不符")
    raw_path = stage_dir / "runs.jsonl.gz"
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        rows = [build_tool.parse(line) for line in stream]
    schedule = list(runner.compare_lp.schedule(models, repeats, variants))
    require(len(rows) == len(schedule) == manifest["runs"], "阶段必须保留全部固定运行，不能补跑替换或删失败")
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, schedule)):
        candidate = variant == variants[1]
        executable = identity["candidate"] if candidate else manifest["reference"]
        digest = identity["candidate_sha256"] if candidate else manifest["reference_sha256"]
        mode = variant if performance else "diagnostic_on" if stage == "migration" else "diagnostic_off" if candidate else "frozen_v66"
        require((row["run_index"], row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"]) ==
            (index, model["model_index"], repeat, position, variant) and row["protocol_variant"] == mode and row["seed"] == frozen["seed"], "原始调度身份不符")
        require(row["command"][:3] == [executable, "solve", model["input"]] and runner.valid_command(row, mode, frozen), "实际求解命令不符共同参数")
        require(row["solver_sha256_before"] == row["solver_sha256_after"] == row["frozen_solver_sha256"] == digest == sha(executable), "程序身份变化")
        require(row["model_sha256"] == row["model_sha256_after"] == row["frozen_model_sha256"] == model["sha256"] == sha(model["input"]), "模型身份变化")
        require(row["presolve_environment"] == ({runner.ENVIRONMENT: "1"} if mode == "diagnostic_on" else {}), "活动环境不符")
        parsed = runner.inspect_markers(row["stderr"], mode)
        require(all(equal(row[key], value) for key, value in zip(("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error", "energy_marker", "energy_marker_error"), parsed)), "三标记原件或partial解释不符")
        protocol_stage = stage if performance else "netlib" if stage == "migration" else stage
        require(row["configuration_matches"] is runner.configuration_matches(row["report"], protocol_stage, mode, model, frozen), "原配置验证结果不符")
        if row["report"] is not None:
            require(equal(build_tool.parse(row["stdout"]), row["report"]), "原stdout与报告不符")
        require(all(equal(row[key], value) for key, value in runner.quality_fields(row, model, digest, frozen).items()), "原质量/预算/罚分未对应完整原件")
    computed = performance_summary(stage, rows, models, repeats, variants, runner, frozen, sha(plan_path), sha(raw_path)) if performance else \
        migration_summary(stage, rows, models, variants, runner, sha(plan_path), sha(raw_path))
    require(equal(read(stage_dir / "summary.json"), computed), "已有summary与只读重算不一致")
    return manifest, computed, rows


def save_same(path, value):
    if path.exists():
        require(equal(read(path), value), "禁止覆盖不一致的已有证据")
    else:
        save(path, value)


def compatibility(output, plan_path):
    plan = read(plan_path)
    require(plan["new_algorithm_round"] is True, "500协议兼容聚合只适用于性能计划")
    runner, frozen = protocol(plan)
    pairs = []
    for stage in COMPAT:
        _, _, rows = archive(stage, output, plan_path)
        for model in runner.specification(stage, frozen)["models"]:
            chosen = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
            pairs.append(dict(stage=stage, input=model["input"], **runner.pair_result(chosen["frozen_v66"], chosen["diagnostic_off"], frozen)))
    require(len(pairs) == 17, "必须有固定17个完整报告兼容对子")
    result = dict(passed=all(pair["category"] == "equal" for pair in pairs), plan_sha256=sha(plan_path), runs=34, pairs=pairs,
        non_budget_equal=sum(pair["category"] == "equal" for pair in pairs), budget_pairs=sum(pair["category"] == "budget_limited" for pair in pairs),
        invalid_pairs=sum(pair["category"] == "invalid_process" for pair in pairs), different_pairs=sum(pair["category"] == "different" for pair in pairs))
    save_same(output / "compatibility.json", result)
    return result


def gate(output, plan_path):
    plan = read(plan_path)
    require(plan["new_algorithm_round"] is True, "性能门槛只适用于固定500计划")
    runner, _ = protocol(plan)
    _, development, _ = archive("development", output, plan_path)
    _, probe, _ = archive("probe", output, plan_path)
    result = runner.base.gate_decision(development, probe)
    result["plan_sha256"] = sha(plan_path)
    result["valid_processes"] = all(summary["valid_process_count"] == summary["run_count"] for summary in (development, probe))
    result["eligible_for_netlib"] = result["eligible_for_netlib"] and result["valid_processes"]
    save_same(output / "gate.json", result)
    return result


def run_performance(work, output, stage, plan_path):
    require(stage in PERFORMANCE_STAGES and work.is_relative_to(ROOT) and output.is_relative_to(ROOT), "性能阶段或独立输出路径无效")
    plan = read(plan_path)
    require(plan["new_algorithm_round"] is True and plan["production_changes"] is False, "性能计划身份不符")
    runner, frozen = protocol(plan)
    identity, _ = verify_work(work, plan_path)
    if stage not in COMPAT:
        require(compatibility(output, plan_path)["passed"], "固定17个非预算完整报告兼容对子尚未全部通过")
    if stage in ("netlib", "highs"):
        require(gate(output, plan_path)["eligible_for_netlib"], "未通过冻结的开发/探针扩集门槛，必须止于136次")
        if stage == "highs":
            archive("netlib", output, plan_path)
    selected, variants = runner.specification(stage, frozen), runner.variants(stage)
    models, repeats = selected["models"], selected["repeats"]
    reference = ROOT / ".tools/solver-ipm-v66.exe"
    reference_sha = plan["production_sha256"]
    require(sha(reference) == reference_sha, "冻结共同资源参考程序变化")
    stage_dir = output / stage
    require(not stage_dir.exists(), "阶段必须是尚不存在的新证据目录")
    stage_dir.mkdir(parents=True)
    provenance = stage_dir / "provenance"; provenance.mkdir()
    for source, name in ((Path(__file__), "benchmark_experimental.py"), (Path(build_tool.__file__), "build_experimental.py"),
        (plan_path, "plan.json"), (DATA / plan["protocol_file"], "protocol.json"), (DATA / "v100-run.py", "protocol-runner.py")):
        (provenance / name).write_bytes(source.read_bytes())
    protected = dict(identity["protected"])
    protected.update(build_tool.hashes(provenance.iterdir()))
    protected.update({str(work / "identity.json"): sha(work / "identity.json"), str(reference): reference_sha})
    require(sha(selected["models_file"]) == selected["models_file_sha256"], "固定模型列表变化")
    for model in models:
        require(model["hash_error"] is None and sha(model["input"]) == model["sha256"], "固定模型身份变化")
    manifest = dict(schema=1, stage=stage, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), new_algorithm_round=True,
        plan_sha256=sha(plan_path), work=str(work), build_identity_sha256=sha(work / "identity.json"), models=models,
        models_file=selected["models_file"], models_file_sha256=selected["models_file_sha256"], repeats=repeats, variants=list(variants),
        runs=2 * repeats * len(models), reference=str(reference), reference_sha256=reference_sha,
        candidate=identity["candidate"], candidate_sha256=identity["candidate_sha256"], time_limit=frozen["time_limit"],
        wall_timeout=frozen["wall_timeout"], threads=frozen["threads"], seed=frozen["seed"], protected=protected)
    save(stage_dir / "manifest.json", manifest)
    rows = []
    raw_path = stage_dir / "runs.jsonl.gz"
    with gzip.open(raw_path, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in runner.compare_lp.schedule(models, repeats, variants):
            candidate = variant == variants[1]
            executable = Path(identity["candidate"]) if candidate else reference
            digest = identity["candidate_sha256"] if candidate else reference_sha
            before = sha(executable)
            with runner.solver_environment(variant == "diagnostic_on"):
                row = runner.benchmark.run_one(runner.options_for(stage, variant, executable, frozen), Path(model["input"]), frozen["seed"], len(rows))
            row.update(dict(zip(("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error", "energy_marker", "energy_marker_error"),
                runner.inspect_markers(row["stderr"], variant))))
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position, requested_variant=variant,
                protocol_variant=variant, frozen_model_sha256=model["sha256"], model_sha256_after=sha(model["input"]), solver_sha256_before=before,
                solver_sha256_after=sha(executable), frozen_solver_sha256=digest, presolve_environment={runner.ENVIRONMENT: "1"} if variant == "diagnostic_on" else {},
                configuration_matches=runner.configuration_matches(row["report"], stage, variant, model, frozen))
            row.update(runner.quality_fields(row, model, digest, frozen))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); stream.flush()
            rows.append(row)
            print(json.dumps(dict(stage=stage, index=row["run_index"], model=Path(model["input"]).stem, variant=variant,
                status=row["status"], quality_pass=row["quality_pass"]), ensure_ascii=False), flush=True)
    require(len(rows) == manifest["runs"], "实际阶段运行数不符")
    summary = performance_summary(stage, rows, models, repeats, variants, runner, frozen, sha(plan_path), sha(raw_path))
    save(stage_dir / "summary.json", summary)
    build_tool.verify(protected)
    require(sha(Path(__file__)) == sha(provenance / "benchmark_experimental.py") and sha(build_tool.__file__) == sha(provenance / "build_experimental.py"), "阶段执行期间工具被并发修改")
    archive(stage, output, plan_path)
    return summary


def self_check(plan_path):
    plan = read(plan_path)
    runner, frozen = protocol(plan)
    # 这里只检查参数、调度与既有验证器反例，不启动任何求解进程。
    inherited = dict(coupled=runner.coupled_self_check(), energy=runner.energy_self_check(),
        initial_forcing=runner.forcing_self_check(), candidate_forcing=runner.candidate_self_check())
    for stage in PERFORMANCE_STAGES:
        for variant in runner.variants(stage):
            options = runner.options_for(stage, variant, ROOT / "placeholder.exe", frozen)
            require((options.time_limit, options.wall_timeout, options.threads) == (5.0, 35, 1), "共同资源参数不符")
            require(options.backend == ("highs" if variant == "highs" else "ipm"), "后端参数不符")
    counts = {stage: len(list(runner.compare_lp.schedule(runner.specification(stage, frozen)["models"],
        runner.specification(stage, frozen)["repeats"], runner.variants(stage)))) for stage in PERFORMANCE_STAGES}
    require(list(counts.values()) == [22, 12, 66, 36, 182, 182] and sum(counts.values()) == 500, "固定136/500调度不符")
    a = dict(presolve_marker=dict(timings=dict(total=1), native=dict(time_limit=4), oracle=dict(total_seconds=1, events=[dict(timings=dict(total=1), decision="保留")])),
        coupled_marker=dict(schema=3), energy_marker=dict(schema=1))
    b = copy.deepcopy(a); b["presolve_marker"]["timings"]["total"] = 2
    require(not differences(private_view(a), private_view(b)), "固定时间字段应排除")
    b["presolve_marker"]["native"]["time_limit"] = 3
    require([item["path"] for item in differences(private_view(a), private_view(b))] == [DYNAMIC_BUDGET], "动态剩余预算必须原样单列")
    b["coupled_marker"]["schema"] = 4
    require(any(item["path"] == "coupled_marker/schema" for item in differences(private_view(a), private_view(b))), "非时间差异不能被排除")
    sample = dict(reference_variant="r", candidate_variant="c", models=[dict(input="fixture.mps", quality_pass_counts=dict(r=1, c=1))],
        variants=dict(r=dict(penalized_numerical_sgm10_seconds=1, penalized_wall_sgm10_seconds=1),
            c=dict(penalized_numerical_sgm10_seconds=0.9, penalized_wall_sgm10_seconds=1)))
    probe = copy.deepcopy(sample)
    require(runner.base.gate_decision(sample, probe)["eligible_for_netlib"], "数值严格改善且墙钟不退时应通过原门槛")
    unchanged = copy.deepcopy(sample); unchanged["variants"]["c"]["penalized_numerical_sgm10_seconds"] = 1
    require(not runner.base.gate_decision(unchanged, probe)["eligible_for_netlib"], "数值持平不能伪称严格改善")
    slower = copy.deepcopy(sample); slower["variants"]["c"]["penalized_wall_sgm10_seconds"] = 1.1
    require(not runner.base.gate_decision(slower, probe)["eligible_for_netlib"], "墙钟退化不能过速度门槛")
    gain = copy.deepcopy(probe); gain["models"][0]["quality_pass_counts"]["r"] = 0
    require(runner.base.gate_decision(unchanged, gain)["eligible_for_netlib"], "无覆盖损失且探针增益应沿原政策通过")
    lost = copy.deepcopy(probe); lost["models"][0]["quality_pass_counts"]["c"] = 0
    require(not runner.base.gate_decision(sample, lost)["eligible_for_netlib"] and
        not runner.base.gate_decision(lost, gain)["eligible_for_netlib"], "任一集合覆盖损失都必须阻止扩集")
    return dict(passed=True, public_lp_runs=0, scheduled_runs=counts, maximum_solver_runs=500,
        conditional_stop_runs=136, configuration_cases=12, difference_cases=3, gate_cases=6, inherited_schema_checks=inherited,
        migration_protocol_preserved=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", required=True, choices=(*STAGES, "self-check", "compatibility", "gate"))
    parser.add_argument("--plan", type=Path, default=DATA / "experimental-source-migration-plan.json")
    parser.add_argument("--archive-only", action="store_true", help="只重解析既有阶段原始记录，不启动新LP")
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve()
    if args.stage == "self-check":
        result = self_check(plan_path)
    else:
        require(args.output is not None, "评测或只读结项必须指定--output证据目录")
        output = (ROOT / args.output).resolve()
        require(output.is_relative_to(ROOT), "证据目录必须位于工作区")
        if args.stage in ("compatibility", "gate"):
            result = compatibility(output, plan_path) if args.stage == "compatibility" else gate(output, plan_path)
        elif args.archive_only:
            _, result, _ = archive(args.stage, output, plan_path)
        else:
            require(args.work is not None, "运行新LP必须指定--work独立构建目录")
            work = (ROOT / args.work).resolve()
            result = run_performance(work, output, args.stage, plan_path) if read(plan_path)["new_algorithm_round"] else run_migration(work, output, args.stage, plan_path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result.get("passed", True) else 1)
