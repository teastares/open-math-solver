#!/usr/bin/env python3
"""验证直接维护的C++与冻结算法等价；保存216次迁移对照，不执行性能门槛。"""
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
STAGES = ("compatibility-development", "compatibility-probe", "migration")
RUNNER_SHA = "ecd1f8eff16d24a74474817864d8fb86499d9174a82339beb3ea5ada134f8ccc"
PRIVATE_CLOCKS = ["presolve_marker/timings", "presolve_marker/oracle/total_seconds", "presolve_marker/oracle/events/*/timings"]
DYNAMIC_BUDGET = "presolve_marker/native/time_limit"
sys.dont_write_bytecode = True
sha, read, save, require, equal = (getattr(build_tool, name) for name in ("sha", "read", "save", "require", "equal"))


def protocol(plan):
    path = DATA / plan["protocol_file"]
    require(sha(path) == plan["protocol_sha256"] and sha(DATA / "v100-run.py") == RUNNER_SHA, "冻结数值协议或纯验证运行器变化")
    require(plan["verification"]["public_report_excluded_fields"] == ["timings"] and
        plan["verification"]["private_excluded_paths"] == PRIVATE_CLOCKS and
        plan["verification"]["dynamic_private_differences_record_only"] == [DYNAMIC_BUDGET], "预先冻结的报告比较边界变化")
    spec = importlib.util.spec_from_file_location("experimental_readonly_v100_protocol", DATA / "v100-run.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, read(path)


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


def run(work, output, stage, plan_path):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--plan", type=Path, default=DATA / "experimental-source-migration-plan.json")
    args = parser.parse_args()
    result = run((ROOT / args.work).resolve(), (ROOT / args.output).resolve(), args.stage, (ROOT / args.plan).resolve())
    raise SystemExit(0 if result["passed"] else 1)
