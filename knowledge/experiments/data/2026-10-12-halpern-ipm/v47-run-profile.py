"""冻结并串行采集当前齐次内点法的阶段成本；诊断计时不作为公开性能成绩。"""
import argparse
import datetime
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v47-profile-fixed"
PLAN = DATA / "v47-plan.json"
RAW = DATA / "v47-profile-runs.jsonl.gz"
MANIFEST = DATA / "v47-profile-manifest.json"
SUMMARY = DATA / "v47-profile-summary.json"
PRODUCTION = ROOT / ".tools/solver-ipm-v46.exe"
DIAGNOSTIC = WORK / "solver-profile.exe"
MODES = ("production", "off", "on0", "on1", "on2")
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return benchmark.sha256(path)


def safe_digest(path):
    try:
        return sha(path), None
    except OSError as error:
        return None, str(error)


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def common_options():
    previous = read(DATA / "v46-plan.json")
    options = ["--backend", "ipm", "--time-limit", "5", "--threads", "1", "--seed", "0",
               "--iteration-limit", "200", "--scaling-passes", "5", "--ipm-method", "homogeneous",
               "--ipm-linear-accuracy", "strict", "--ipm-direction-retries", "0"]
    for key, value in previous["shared_ipm"].items():
        options.extend(["--ipm-" + key, value])
    return options


def prepare():
    models, digest = compare_lp.load_models(ROOT / "benchmarks/development.txt")
    assert len(models) == 8
    for name in ("pilot87", "maros-r7", "fit1p", "dfl001", "greenbea"):
        path = ROOT / ".deps/netlib" / (name + ".mps")
        models.append({"model_index": len(models), "input": str(path), "sha256": sha(path), "hash_error": None})
    assert len({item["input"] for item in models}) == 13
    for item in models:
        assert sha(Path(item["input"])) == item["sha256"]
    identity = read(DATA / "ipm-v46-identity.json")
    assert sha(PRODUCTION) == identity["solver_sha256"]
    save_new(PLAN, {"schema": 1, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "将当前HSD strict的规范化、结构准备、Gram装配、数值分解、回代与验算分开，决定下一轮减少哪部分工作量；不是算法候选或正式加速比。",
        "production_sha256": sha(PRODUCTION), "v46_identity_sha256": sha(DATA / "ipm-v46-identity.json"),
        "runner_sha256": sha(Path(__file__)), "builder_sha256": sha(DATA / "v47-profile-build.py"),
        "quality_scripts": {name: sha(ROOT / "scripts" / name) for name in ("benchmark.py", "compare_lp.py")},
        "development_list_sha256": digest, "models": models, "options": common_options(),
        "time_limit": 5, "wall_timeout": 35, "modes": list(MODES), "runs": 65,
        "schedule": "每模型将production/off/on0/on1/on2列表向左轮转model_index mod 5位，全部串行；每个输入有生产1次、同诊断exe关闭1次、开启3次。",
        "profile_environment": "SOLVER_V47_PROFILE", "profile_prefix": "V47_PROFILE ",
        "trajectory": "比较production/off及off/on三个对子，覆盖全部非timings顶层字段及退出码；一对任一方time_limit/墙钟超时或实际数值预算超限时单列，不要求迭代数相同；其他差异不得忽略。",
        "aggregation": "每模型先取三个开启重复的各阶段整数纳秒中位数，再按阶段累加；每调用成本为阶段包含时间中位数除以启动次数中位数，不是各次平均的中位数。保留全部原始时间及失败。包含时间有嵌套，不可直接加总；每次运行独占纳秒之和须等于total包含纳秒。",
        "failure_policy": "保留失败与超时，不改预算或重跑替换。模型/程序/脚本变化、结构或计时错误均失败。",
        "research_sources": ["https://ergo-code.github.io/HiGHS/dev/solvers/",
            "https://github.com/ERGO-Code/HiGHS/blob/v1.15.1/highs/ipm/ipx/normal_matrix.cc",
            "https://github.com/ERGO-Code/HiGHS/blob/v1.15.1/highs/ipm/hipo/ipm/KktMatrix.cpp",
            "https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/v7.14.1/CHOLMOD/Supernodal/cholmod_super_numeric.c",
            "https://docs.mosek.com/slides/ismp2012/dense.pdf"],
        "source_caveat": "当前HiGHS构建HIPO关闭，choose基线为单纯形；IPX/HiPO/CHOLMOD只作为技术素材，不冒称本轮比较了其LP算法。旧v19的factor_kernel含装配，不能当作当前纯分解占比。"})
    print(json.dumps({"plan_sha256": sha(PLAN), "models": len(models), "runs": 65}), flush=True)


def frozen_plan():
    plan = read(PLAN)
    correction = read(DATA / "v47-build-correction.json")
    assert correction["plan_sha256"] == sha(PLAN) and correction["lp_runs_before_correction"] == 0
    assert sha(DATA / "v47-run-profile-attempt1.py") == plan["runner_sha256"]
    assert sha(DATA / "v47-profile-build-attempt1.py") == plan["builder_sha256"]
    assert sha(Path(__file__)) == correction["runner_sha256"]
    assert sha(DATA / "v47-profile-build.py") == correction["builder_sha256"]
    assert sha(DATA / "ipm-v46-identity.json") == plan["v46_identity_sha256"]
    assert sha(PRODUCTION) == plan["production_sha256"]
    for name, digest in plan["quality_scripts"].items():
        assert sha(ROOT / "scripts" / name) == digest
    return plan


def schedule(plan):
    for model in plan["models"]:
        offset = model["model_index"] % len(MODES)
        for mode in MODES[offset:] + MODES[:offset]:
            yield model, mode


def parse_profile(stderr, enabled):
    lines = [line for line in stderr.splitlines() if line.startswith("V47_PROFILE ")]
    assert len(lines) == int(enabled), "诊断输出数量错误"
    if not enabled:
        return None
    payload = benchmark.strict_json(lines[0][len("V47_PROFILE "):])
    assert payload["schema"] == 1
    assert all(payload[key] is True for key in ("conserved", "stack_empty", "counts_closed"))
    stages = payload["stages"]
    assert isinstance(stages, dict) and "total" in stages
    for name, stage in stages.items():
        for key in ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound"):
            assert type(stage[key]) is int and stage[key] >= 0, (name, key)
        assert stage["started"] == stage["completed"] + stage["unwound"], name
        assert stage["exclusive_ns"] <= stage["inclusive_ns"], name
    assert stages["total"]["started"] == 1
    assert payload["clock_reads"] == 2 * sum(item["started"] for item in stages.values())
    assert sum(item["exclusive_ns"] for item in stages.values()) == stages["total"]["inclusive_ns"]
    assert payload["exclusive_sum_ns"] == payload["total_inclusive_ns"] == stages["total"]["inclusive_ns"]
    return stages


def run():
    plan = frozen_plan()
    assert not RAW.exists() and not MANIFEST.exists() and not SUMMARY.exists()
    build_manifest = WORK / "build-manifest.json"
    assert build_manifest.exists() and DIAGNOSTIC.exists()
    identities = {"production": sha(PRODUCTION), "diagnostic": sha(DIAGNOSTIC)}
    for model in plan["models"]:
        assert sha(Path(model["input"])) == model["sha256"]
    save_new(MANIFEST, {"plan_sha256": sha(PLAN), "build_manifest_sha256": sha(build_manifest),
        "build_correction_sha256": sha(DATA / "v47-build-correction.json"),
        "executables": identities, "diagnostic": str(DIAGNOSTIC), "production": str(PRODUCTION),
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    with gzip.open(RAW, "xt", encoding="utf-8") as target:
        for index, (model, mode) in enumerate(schedule(plan)):
            exe = PRODUCTION if mode == "production" else DIAGNOSTIC
            enabled = mode.startswith("on")
            env = os.environ.copy()
            env.update(read(MANIFEST)["environment"])
            env["SOLVER_V47_PROFILE"] = "1" if enabled else "0"
            command = [str(exe), "solve", model["input"], *plan["options"]]
            model_before, model_before_error = safe_digest(Path(model["input"]))
            exe_before, exe_before_error = safe_digest(exe)
            started = time.perf_counter()
            error = None
            error_detail = None
            try:
                process = subprocess.run(command, capture_output=True, timeout=plan["wall_timeout"], env=env, cwd=ROOT)
                stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
            except subprocess.TimeoutExpired as exception:
                stdout, stderr, returncode = exception.stdout or b"", exception.stderr or b"", None
                error = "wall_timeout"
            except OSError as exception:
                stdout, stderr, returncode = b"", b"", None
                error, error_detail = "launch_error", str(exception)
            elapsed = time.perf_counter() - started
            model_after, model_after_error = safe_digest(Path(model["input"]))
            exe_after, exe_after_error = safe_digest(exe)
            stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
            report = None
            try:
                report = benchmark.strict_json(stdout)
                if not isinstance(report, dict):
                    report, error = None, error or "invalid_report"
            except (ValueError, TypeError):
                error = error or "invalid_json"
            row = {"index": index, "input": model["input"], "mode": mode, "command": command,
                "returncode": returncode, "stdout": stdout, "stderr": stderr, "report": report,
                "elapsed_wall_seconds": elapsed, "error": error, "error_detail": error_detail,
                "model_sha256": model_before, "model_sha256_after": model_after,
                "solver_sha256_before": exe_before, "solver_sha256_after": exe_after,
                "digest_errors": [model_before_error, exe_before_error, model_after_error, exe_after_error],
                "classification": benchmark.classify(returncode, report) if report is not None else error}
            numerical = compare_lp.numerical_seconds(row)
            expected_exe = identities["production" if mode == "production" else "diagnostic"]
            failure = compare_lp.quality_failure(row, model, model_after, expected_exe, exe_after if exe_before == expected_exe else None)
            if failure is None and (numerical > plan["time_limit"] or elapsed > plan["wall_timeout"]):
                failure = "budget_exceeded"
            row.update(numerical_seconds=numerical, quality_pass=failure is None, quality_failure=failure)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            target.flush()
            assert model_before == model_after == model["sha256"] and exe_before == exe_after == expected_exe, "实测过程中来源变化"
            print(json.dumps({"index": index, "model": Path(model["input"]).stem, "mode": mode,
                "status": (report or {}).get("status"), "numerical_seconds": numerical}), flush=True)
    assert sha(PRODUCTION) == identities["production"] and sha(DIAGNOSTIC) == identities["diagnostic"]
    for model in plan["models"]:
        assert sha(Path(model["input"])) == model["sha256"]
    summarize()


def limited(row, plan):
    return (row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit"
            or row["elapsed_wall_seconds"] > plan["wall_timeout"]
            or row["numerical_seconds"] is not None and row["numerical_seconds"] > plan["time_limit"])


def summarize():
    plan = frozen_plan()
    rows = [json.loads(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    assert len(rows) == plan["runs"]
    models, totals = [], {}
    for model in plan["models"]:
        runs = {row["mode"]: row for row in rows if row["input"] == model["input"]}
        assert set(runs) == set(MODES)
        comparisons = {}
        for mode, row in runs.items():
            if row["error"] == "wall_timeout":
                row["profile"] = None
            else:
                assert row["error"] is None and isinstance(row["report"], dict), (model["input"], mode, "报告缺失或无效")
                assert row["classification"] in ("resolved", "incomplete", "verification_failed"), (model["input"], mode, "报告协议无效")
                row["profile"] = parse_profile(row["stderr"], mode.startswith("on"))
            if mode == "production":
                continue
            reference = runs["production" if mode == "off" else "off"]
            comparable = lambda entry: {key: value for key, value in (entry["report"] or {}).items() if key != "timings"}
            same = comparable(row) == comparable(reference) and row["returncode"] == reference["returncode"]
            comparisons[mode] = {"equal": same, "budget_limited": limited(reference, plan) or limited(row, plan)}
            assert same or comparisons[mode]["budget_limited"], (model["input"], mode, "非时间轨迹改变")
        profiles = [runs[mode]["profile"] for mode in MODES if mode.startswith("on")]
        stages = {}
        if all(profile is not None for profile in profiles):
            assert all(set(profile) == set(profiles[0]) for profile in profiles)
            for name in profiles[0]:
                stages[name] = {key: statistics.median(profile[name][key] for profile in profiles)
                    for key in ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")}
                calls = stages[name]["started"]
                stages[name]["ratio_of_medians_seconds_per_started_call"] = stages[name]["inclusive_ns"] / 1e9 / calls if calls else None
                aggregate = totals.setdefault(name, {"inclusive_ns": 0, "exclusive_ns": 0})
                for key in aggregate:
                    aggregate[key] += stages[name][key]
        models.append({"input": model["input"], "comparisons": comparisons, "median_stages": stages,
            "runs": {mode: {"status": (row["report"] or {}).get("status"), "quality_pass": row["quality_pass"],
                "numerical_seconds": row["numerical_seconds"], "wall_seconds": row["elapsed_wall_seconds"],
                "iterations": (row["report"] or {}).get("ipm_iterations"), "budget_limited": limited(row, plan)} for mode, row in runs.items()}})
    result = {"schema": 1, "plan_sha256": sha(PLAN), "raw_sha256": sha(RAW), "runs": len(rows),
        "models": models, "sum_of_model_stage_medians": totals,
        "qualification": "包含计时互相嵌套；逐阶段中位数不保证加总守恒，守恒仅逐次原始记录验证。异常展开表示作用域被异常退出，不等同于整个LP失败。诊断程序耗时不计入公开性能成绩。"}
    if SUMMARY.exists():
        assert read(SUMMARY) == result, "已归档汇总与重新计算不符"
    else:
        save_new(SUMMARY, result)
    print(json.dumps({"runs": len(rows), "summary": str(SUMMARY), "stages": totals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    globals()[parser.parse_args().action]()
