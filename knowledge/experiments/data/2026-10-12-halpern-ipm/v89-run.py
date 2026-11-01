"""第86轮：三模型、两次重复、三路轮转的最后完整assess质量诊断。"""
import argparse
import contextlib
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v89-presolve"
PLAN_SHA = "98ff5dfcbbc0ea7f24cd3e6894e7c2e2a95be5c223da98a9980779412c7d54a8"
PARENT_RUN_SHA = "6080b97380f79d64b3aafd79a265554c29910fbbd767477ed0ee5b0cacfc2d04"
PARENT_INST_SHA = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
VARIANTS = ("frozen_v88", "diagnostic_off", "diagnostic_on")
RAW = DATA / "v89-diagnostic-runs.jsonl.gz"
MANIFEST = DATA / "v89-diagnostic-manifest.json"
SUMMARY = DATA / "v89-diagnostic-summary.json"
IDENTITY = DATA / "v89-run-identity.json"
COMPATIBILITY = DATA / "v89-compatibility.json"
MAPS = ("protected_before_and_after", "generated_inputs_before_and_after", "compile_inputs_after_and_final",
        "link_explicit_before_and_after", "test_link_explicit_before_and_after", "outputs")
REQUIRED_TRACE_CASES = {
    "trace_nonoptimal_current_overrides_old_incumbent": "transaction",
    "trace_incomplete_attempt_keeps_last_complete": "exception",
    "trace_tls_scope_restores_previous": "transaction",
    "trace_disabled_has_no_work": "transaction",
    "trace_records_before_oracle_failure": "exception",
}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(DATA / "v88-run.py") == PARENT_RUN_SHA
_spec = importlib.util.spec_from_file_location("v89_frozen_v88_protocol", DATA / "v88-run.py")
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
strict_json, read, save, typed_equal = parent.strict_json, parent.read, parent.save, parent.base.typed_equal


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def save_once(path, value):
    if path.exists():
        assert same(read(path), value), "禁止覆盖不一致的诊断证据"
    else:
        save(path, value)


def fixed_plan():
    assert sha(DATA / "v89-plan.json") == PLAN_SHA
    plan = read(DATA / "v89-plan.json")
    assert (plan["parent_version"], plan["reference_version"], plan["candidate_version"]) == (66, 88, 89)
    assert plan["diagnostic_only"] is True and plan["production_changes"] is False
    assert plan["variants"] == list(VARIANTS) and plan["stage_order"] == ["diagnostic"]
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"],
            plan["scaling_passes"], plan["repeats"], plan["maximum_solver_runs"], plan["maximum_trace_assessments"]) == (5, 35, 200, 1, 0, 5, 2, 18, 204)
    assert plan["method"] == "homogeneous" and plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["new_report_fields"] == [] and plan["no_model_specific_routing"] is True
    assert plan["diagnostic_environment"] == "SOLVER_V89_PRESOLVE" and plan["trace_environment"] == "SOLVER_V89_ASSESS_TRACE"
    assert plan["reference_environment"] == "SOLVER_V88_PRESOLVE"
    assert plan["diagnostic_marker"] == "V89_PRESOLVE" and plan["reference_marker"] == "V88_PRESOLVE"
    assert plan["primary_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert (plan[side + "_linear_solver"], plan[side + "_accuracy"], plan[side + "_objective_scaling"],
                plan[side + "_backend_version"]) == ("normal_cholmod_aat", "homogeneous_repair", "off", "ipm-cpu-0.53.0")
    assert len(plan["models"]) == 3 and [Path(item["input"]).stem for item in plan["models"]] == ["greenbea", "forplan", "greenbeb"]
    assert [item["model_index"] for item in plan["models"]] == [0, 1, 2]
    assert all(item["hash_error"] is None for item in plan["models"])
    assert plan["reference_runner_file"] == "v88-run.py" and plan["reference_runner_sha256"] == PARENT_RUN_SHA
    assert plan["parent_instrumentation_file"] == "v88-ipm-instrumentation.py" and plan["parent_instrumentation_sha256"] == PARENT_INST_SHA
    for path, digest in ((DATA / plan["evidence_file"], plan["evidence_sha256"]),
        (DATA / plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (DATA / "ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (DATA / plan["parent_instrumentation_file"], PARENT_INST_SHA),
        (Path(plan["reference_manifest"]), plan["reference_manifest_sha256"]),
        (Path(plan["reference_executable"]), plan["reference_executable_sha256"])):
        assert sha(path) == digest, path
    return plan


def schedule(plan):
    for model in plan["models"]:
        for repetition in range(plan["repeats"]):
            case_index = model["model_index"] * plan["repeats"] + repetition
            offset = case_index % len(VARIANTS)
            for order, variant in enumerate(VARIANTS[offset:] + VARIANTS[:offset]):
                yield model, repetition, case_index, order, variant


def expected_options(plan):
    value = copy.deepcopy(plan["expected_report_options"])
    # 冻结模板来自v66；本轮明确采用AAT，按计划的实际求解器字段覆盖。
    value["ipm"].update(linear_solver=plan["primary_linear_solver"], linear_accuracy=plan["candidate_accuracy"], objective_scaling=False)
    return value


def configuration_matches(report, model, plan):
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == "ipm" \
        and report.get("backend_version") == "ipm-cpu-0.53.0" and typed_equal(report.get("options"), expected_options(plan))


def variant_environment(variant):
    return {"SOLVER_V88_PRESOLVE": "1"} if variant == "frozen_v88" else \
        {"SOLVER_V89_PRESOLVE": "1", **({"SOLVER_V89_ASSESS_TRACE": "1"} if variant == "diagnostic_on" else {})}


def clean_environment(prior, variant):
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    value = {key: item for key, item in prior.items() if not key.upper().startswith("SOLVER_V")
             and key.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    value.update(dict.fromkeys(threads, "1"))
    value.update(variant_environment(variant))
    return value


@contextlib.contextmanager
def solver_environment(variant):
    before, cwd = dict(os.environ), Path.cwd()
    try:
        os.environ.clear()
        os.environ.update(clean_environment(before, variant))
        os.chdir(ROOT)
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
        os.chdir(cwd)


def valid_attempt(value):
    return isinstance(value, dict) and set(value) == {"sequence", "iteration"} and parent.integer(value["sequence"], 203) \
        and (value["iteration"] is None or parent.integer(value["iteration"], 200))


def valid_trace(value, enabled):
    if not isinstance(value, dict) or set(value) != {"enabled", "started", "completed", "last_attempt", "last_complete"} \
            or type(value["enabled"]) is not bool or value["enabled"] is not enabled:
        return False
    if not all(parent.integer(value[key], 204) for key in ("started", "completed")) or value["completed"] > value["started"]:
        return False
    if not enabled:
        return same(value, dict(enabled=False, started=0, completed=0, last_attempt=None, last_complete=None))
    attempt, complete = value["last_attempt"], value["last_complete"]
    if (attempt is None) != (value["started"] == 0) or (complete is None) != (value["completed"] == 0):
        return False
    if attempt is not None and (not valid_attempt(attempt) or attempt["sequence"] != value["started"] - 1):
        return False
    if complete is not None:
        if not isinstance(complete, dict) or set(complete) != {"sequence", "iteration", "quality"} \
                or not valid_attempt({key: complete[key] for key in ("sequence", "iteration")}) \
                or complete["quality"] is None or not parent.valid_quality(complete["quality"]):
            return False
        if not value["completed"] - 1 <= complete["sequence"] <= attempt["sequence"]:
            return False
        if complete["sequence"] == attempt["sequence"] and complete["iteration"] != attempt["iteration"]:
            return False
    return True


def old_marker(value):
    if not isinstance(value, dict):
        return value
    return {key: copy.deepcopy(item) for key, item in value.items() if key != "assess_trace"}


def non_timing_marker(value):
    def visit(item, path=()):
        if isinstance(item, dict):
            return {key: visit(child, path + (key,)) for key, child in item.items() if key != "timings"
                    and not (path == ("oracle",) and key == "total_seconds") and not (path == () and key == "assess_trace")}
        if isinstance(item, list):
            return [visit(child, path + (index,)) for index, child in enumerate(item)]
        if isinstance(item, str):
            return item.replace("v89", "v88").replace("V89", "V88")
        return item
    return visit(value)


def inspect_marker(stderr, variant):
    prefix = ("V88_PRESOLVE " if variant == "frozen_v88" else "V89_PRESOLVE ")
    lines = [line for line in stderr.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith(prefix):
        return None, "marker_count", None, None
    try:
        value = strict_json(lines[0][len(prefix):])
    except (TypeError, ValueError):
        return None, "invalid_marker_json", None, None
    if not isinstance(value, dict):
        return value, "invalid_marker_schema", None, None
    trace = value.get("assess_trace")
    trace_error = None
    if variant == "frozen_v88":
        if "assess_trace" in value:
            return value, "unexpected_trace", trace, None
    elif "assess_trace" not in value or not valid_trace(trace, variant == "diagnostic_on"):
        trace_error = "invalid_trace_schema"
    error = "invalid_marker_schema" if not parent.valid_marker(old_marker(value), True) else trace_error
    return value, error, trace, trace_error


def valid_process(row):
    return row.get("trace_error") is None and parent.valid_process(row)


def limited(row, plan):
    return row.get("classification") == "wall_timeout" or parent.base.limited(row, plan)


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row.get("report")) for row in (a, b)]
    for report in reports:
        if isinstance(report, dict):
            report.pop("timings", None)
    report_equal = typed_equal(*reports) and typed_equal(a.get("returncode"), b.get("returncode")) \
        and typed_equal(a.get("error"), b.get("error"))
    marker_equal = typed_equal(non_timing_marker(a.get("presolve_marker")), non_timing_marker(b.get("presolve_marker")))
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, plan) or limited(b, plan)
    equal = report_equal and marker_equal
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return dict(equal=equal, report_equal=report_equal, old_diagnostics_equal=marker_equal,
                valid_processes=valid, budget_limited=budget, category=category)


def freeze(work, plan):
    build_path = work / "build-manifest.json"
    build = read(build_path)
    legacy = read(DATA / "ipm-v66-identity.json")
    assert build["plan_sha256"] == PLAN_SHA and build["production_changes"] is False
    assert build["current_files"] == build["frozen_files"] == legacy["files"] and len(legacy["files"]) == 95
    assert build["actual_libraries"] == legacy["libraries"] and len(legacy["libraries"]) == 17
    assert build["parent_identity_sha256"] == plan["parent_identity_sha256"]
    assert build["production_sha256"] == build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    assert build["baseline_verification_sha256"] == plan["baseline_verification_sha256"]
    assert set(build["commands"]["compile"]) == {"ipm"}
    assert build["commands"]["link"].count(str(work / "ipm-presolve.obj")) == 1
    assert not any("libsolver_ipm.a" in str(arg) or "normal-pair" in str(arg) for arg in build["commands"]["link"])
    assert any(str(arg).endswith("libsolver_normal_cholesky.a") for arg in build["commands"]["link"])
    candidate, reference = Path(build["executable"]), Path(plan["reference_executable"])
    assert candidate == work / "solver-presolve.exe"
    frozen_reference = read(plan["reference_manifest"])
    assert frozen_reference["executable"] == str(reference) and frozen_reference["executable_sha256"] == plan["reference_executable_sha256"]
    protected = {}
    def add(path, digest):
        path = str(Path(path))
        assert path not in protected or protected[path] == digest, path
        protected[path] = digest
    for group in MAPS:
        for path, digest in build[group].items():
            add(path, digest)
    for group in ("files", "libraries"):
        for name, digest in legacy[group].items():
            add(ROOT / name, digest)
    for name, digest in legacy["files"].items():
        add(work / "frozen" / name, digest)
    for name in plan["new_code_and_plan_files"]:
        path = str(DATA / name)
        assert build["protected_before_and_after"][path] == sha(path)
    for path, digest in ((build_path, sha(build_path)), (candidate, build["executable_sha256"]),
        (reference, plan["reference_executable_sha256"]), (plan["reference_manifest"], plan["reference_manifest_sha256"]),
        (DATA / "v88-run.py", PARENT_RUN_SHA), (DATA / "v88-ipm-instrumentation.py", PARENT_INST_SHA),
        (DATA / plan["evidence_file"], plan["evidence_sha256"]),
        (DATA / "ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (ROOT / "scripts/benchmark.py", sha(ROOT / "scripts/benchmark.py")),
        (ROOT / "scripts/compare_lp.py", sha(ROOT / "scripts/compare_lp.py"))):
        add(path, digest)
    add(DATA / "v66-run.py", parent.HELPER_SHA)
    checks = read(work / "ipm-self-check.json")
    raw = read(work / "ipm-self-run.json")
    assert parent.valid_math_evidence("ipm", checks, raw, work)
    assert same(build["math_self_tests"]["ipm"], {**checks["details"], "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"})
    meta = build["math_coverage_extension"]
    assert set(meta) == {"file", "sha256", "preserved_predecessor_cases", "new_cases"}
    extension_path = work / meta["file"]
    assert meta["file"] == "ipm-self-extension-check.json" and meta["sha256"] == protected[str(extension_path)] == sha(extension_path)
    extension = read(extension_path)
    assert set(extension) == {"passed", "public_lp_runs", "predecessor_evidence_file", "predecessor_evidence_sha256",
        "preserved_predecessor_cases", "new_cases", "required_new_cases", "component_calls"}
    assert extension["passed"] is True and type(extension["public_lp_runs"]) is int and extension["public_lp_runs"] == 0
    assert extension["predecessor_evidence_file"] == str(DATA / plan["evidence_file"]) and extension["predecessor_evidence_sha256"] == plan["evidence_sha256"]
    old = read(DATA / plan["evidence_file"])["math_self_checks"]["details"]
    current = checks["details"]
    assert parent.valid_math_details(old)
    old_ids = {item["id"]: item["group"] for item in old["details"]}
    new_ids = {item["id"]: item["group"] for item in current["details"]}
    assert old_ids.items() <= new_ids.items() and REQUIRED_TRACE_CASES.items() <= new_ids.items()
    assert not (old_ids.keys() & REQUIRED_TRACE_CASES.keys())
    added = [item for item in current["details"] if item["id"] not in old_ids]
    assert same(extension["new_cases"], added) and same(extension["required_new_cases"], [{"id": key, "group": group} for key, group in REQUIRED_TRACE_CASES.items()])
    assert type(meta["new_cases"]) is int and meta["new_cases"] == len(added)
    assert type(meta["preserved_predecessor_cases"]) is int and meta["preserved_predecessor_cases"] == extension["preserved_predecessor_cases"] == old["cases"]
    assert current["cases"] == old["cases"] + len(added)
    assert set(extension["component_calls"]) == {"component_ipm_calls", "synthetic_core_calls"}
    for name, pair in extension["component_calls"].items():
        assert all(type(item) is int for item in pair.values()) and pair == {"predecessor": old[name], "current": current[name], "change": current[name] - old[name]}
    header = str(work / "v89-presolve-state.hpp")
    assert build["private_state_header"] == header and header in build["compile_inputs_after_and_final"]
    assert set(build["generated_inputs_before_and_after"]) == {header, str(work / "ipm-presolve.cpp")}
    assert build["readonly_parent_instrumentation"] == {"path": str(DATA / "v88-ipm-instrumentation.py"), "sha256": PARENT_INST_SHA}
    assert read(work / "static-check.json")["passed"] is True
    for path, digest in protected.items():
        assert sha(path) == digest, path
    for model in plan["models"]:
        add(model["input"], model["sha256"])
        assert sha(model["input"]) == model["sha256"]
    result = dict(plan_sha256=PLAN_SHA, build_manifest=str(build_path), build_manifest_sha256=sha(build_path),
        candidate=str(candidate), candidate_sha256=build["executable_sha256"], reference=str(reference),
        reference_sha256=plan["reference_executable_sha256"], protected=protected)
    save_once(IDENTITY, result)
    return result


def analyze(rows, plan):
    pairs = []
    for case_index in range(6):
        chosen = {row["variant"]: row for row in rows if row["case_index"] == case_index}
        assert set(chosen) == set(VARIANTS)
        for left, right in zip(VARIANTS, VARIANTS[1:]):
            pairs.append(dict(case_index=case_index, model_index=chosen[left]["model_index"], input=chosen[left]["input"],
                repetition=chosen[left]["repetition"], reference=left, candidate=right, **pair_result(chosen[left], chosen[right], plan)))
    compatibility = dict(passed=all(pair["category"] == "equal" for pair in pairs), plan_sha256=PLAN_SHA, runs=len(rows), pairs=pairs,
        non_budget_equal=sum(pair["category"] == "equal" for pair in pairs), budget_pairs=sum(pair["category"] == "budget_limited" for pair in pairs),
        invalid_pairs=sum(pair["category"] == "invalid_process" for pair in pairs), different_pairs=sum(pair["category"] == "different" for pair in pairs))
    summary = dict(schema=1, diagnostic_only=True, plan_sha256=PLAN_SHA, runs=len(rows), pairs=len(pairs), raw_sha256=sha(RAW),
        valid_process_count=sum(valid_process(row) for row in rows), marker_errors=sum(row["presolve_marker_error"] is not None for row in rows),
        trace_errors=sum(row["trace_error"] is not None for row in rows), compatibility_passed=compatibility["passed"],
        categories={key: compatibility[key] for key in ("non_budget_equal", "budget_pairs", "invalid_pairs", "different_pairs")},
        variants={variant: dict(runs=sum(row["variant"] == variant for row in rows),
            statuses={status: sum(row["variant"] == variant and str(row.get("status")) == status for row in rows)
                      for status in sorted({str(row.get("status")) for row in rows})}) for variant in VARIANTS},
        trace_observations=[dict(input=row["input"], repetition=row["repetition"], status=row.get("status"),
            route=(row["presolve_marker"] or {}).get("route"), assess_trace=row["assess_trace"]) for row in rows if row["variant"] == "diagnostic_on"],
        scope="只诊断18次调用；Reduced质量属于约化core.model，Original属于原模型；last_complete不是末次已发布点或保存的incumbent；不生成性能胜利或覆盖提升结论。")
    return compatibility, summary


def archive(stage=None):
    assert stage in (None, "diagnostic")
    plan = fixed_plan()
    manifest, identity = read(MANIFEST), read(IDENTITY)
    assert manifest["plan_sha256"] == identity["plan_sha256"] == PLAN_SHA
    assert manifest["identity_sha256"] == sha(IDENTITY) and manifest["variants"] == list(VARIANTS)
    assert manifest["models"] == plan["models"] and manifest["repeats"] == plan["repeats"]
    rows = [strict_json(line) for line in gzip.decompress(RAW.read_bytes()).splitlines()]
    expected = list(schedule(plan))
    assert len(rows) == len(expected) == 18
    for index, (row, (model, repetition, case_index, order, variant)) in enumerate(zip(rows, expected)):
        assert (row["run_index"], row["case_index"], row["model_index"], row["repetition"], row["order"], row["variant"],
                row["requested_variant"]) == (index, case_index, model["model_index"], repetition, order, variant, variant)
        assert row["input"] == model["input"] and row["seed"] == 0 and row["cwd"] == str(ROOT)
        side = "reference" if variant == "frozen_v88" else "candidate"
        assert row["command"][:3] == [identity[side], "solve", model["input"]]
        assert parent.valid_command(row, "diagnostic_on", plan)
        assert row["environment"] == variant_environment(variant)
        assert all(row[key] == identity[side + "_sha256"] for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[key] == model["sha256"] for key in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, marker_error, trace, trace_error = inspect_marker(row["stderr"], variant)
        assert same(row["presolve_marker"], marker) and row["presolve_marker_error"] == marker_error
        assert same(row["assess_trace"], trace) and row["trace_error"] == trace_error
        if row["report"] is not None:
            assert same(row["report"], strict_json(row["stdout"]))
            assert row["status"] == row["report"].get("status")
            assert row["classification"] == benchmark.classify(row["returncode"], row["report"])
        assert row["configuration_matches"] is configuration_matches(row["report"], model, plan)
        assert typed_equal(row["numerical_seconds"], compare_lp.numerical_seconds(row))
        assert row["wall_budget_exceeded"] is (row["elapsed_wall_seconds"] > 35)
        num = row["numerical_seconds"]
        assert row["numerical_budget_exceeded"] is (num > 5 if benchmark.finite_real(num) else None)
    compatibility, summary = analyze(rows, plan)
    assert same(read(COMPATIBILITY), compatibility) and same(read(SUMMARY), summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    return manifest, summary, rows


def run(work):
    plan = fixed_plan()
    assert not any(path.exists() for path in (RAW, MANIFEST, SUMMARY, COMPATIBILITY)), "禁止覆盖已有18次调用原件"
    identity = freeze(work, plan)
    save(MANIFEST, dict(schema=1, diagnostic_only=True, started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=PLAN_SHA, identity_sha256=sha(IDENTITY), work=str(work), models=plan["models"], repeats=2, variants=list(VARIANTS),
        time_limit=5, wall_timeout=35, max_iterations=200, threads=1, seed=0,
        environment_rule="每次大小写无关清除全部SOLVER_V变量及OPENBLAS_CORETYPE，三线程变量置1；三路pipeline始终on，仅diagnostic_on启用新trace；结束恢复原环境。",
        schedule="固定三模型各两重复，case_index模3轮转三variant，串行共18，不重跑失败。"))
    rows = []
    with gzip.open(RAW, "xt", encoding="utf-8") as stream:
        for model, repetition, case_index, order, variant in schedule(plan):
            side = "reference" if variant == "frozen_v88" else "candidate"
            executable = Path(identity[side])
            before = sha(executable)
            options = parent.options_for("development", "diagnostic_on", executable, plan)
            with solver_environment(variant):
                row = benchmark.run_one(options, Path(model["input"]), 0, len(rows))
            after, input_after = sha(executable), sha(model["input"])
            marker, marker_error, trace, trace_error = inspect_marker(row["stderr"], variant)
            row.update(case_index=case_index, model_index=model["model_index"], repetition=repetition, order=order,
                variant=variant, requested_variant=variant, cwd=str(ROOT), environment=variant_environment(variant),
                solver_sha256_before=before, solver_sha256_after=after, frozen_solver_sha256=identity[side + "_sha256"],
                model_sha256_after=input_after, frozen_model_sha256=model["sha256"], presolve_marker=marker,
                presolve_marker_error=marker_error, assess_trace=trace, trace_error=trace_error,
                configuration_matches=configuration_matches(row["report"], model, plan))
            numerical = compare_lp.numerical_seconds(row)
            row.update(numerical_seconds=numerical, numerical_budget_exceeded=numerical > 5 if benchmark.finite_real(numerical) else None,
                wall_budget_exceeded=row["elapsed_wall_seconds"] > 35)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            assert before == after == identity[side + "_sha256"] and row["model_sha256"] == input_after == model["sha256"]
            print(json.dumps(dict(index=row["run_index"], model=Path(model["input"]).stem, variant=variant,
                repetition=repetition, status=row["status"], valid_process=valid_process(row), marker_error=marker_error), ensure_ascii=False), flush=True)
    compatibility, summary = analyze(rows, plan)
    save(COMPATIBILITY, compatibility)
    save(SUMMARY, summary)
    archive()
    print(json.dumps(dict(runs=18, pairs=12, compatibility_passed=compatibility["passed"], categories=summary["categories"]), ensure_ascii=False), flush=True)
    assert compatibility["invalid_pairs"] == compatibility["different_pairs"] == 0, "诊断不兼容，全部18次原件已保存"


def self_check():
    plan = fixed_plan()
    zero = dict(enabled=False, started=0, completed=0, last_attempt=None, last_complete=None)
    quality = dict.fromkeys(parent.QUALITY_BOOLS, False)
    quality.update(dict.fromkeys(parent.QUALITY_NUMBERS, None))
    complete = dict(sequence=0, iteration=0, quality=quality)
    valid = [zero, dict(zero, enabled=True), dict(enabled=True, started=1, completed=1,
        last_attempt=dict(sequence=0, iteration=0), last_complete=complete),
        dict(enabled=True, started=5, completed=2, last_attempt=dict(sequence=4, iteration=6),
             last_complete=dict(complete, sequence=2, iteration=3)),
        dict(enabled=True, started=3, completed=0, last_attempt=dict(sequence=2, iteration=None), last_complete=None)]
    for value in valid:
        assert valid_trace(value, value["enabled"])
    mutations = 0
    for key, bad in (("started", True), ("started", -1), ("started", 205), ("completed", 1.0),
                     ("enabled", 1), ("last_attempt", {}), ("last_complete", {})):
        changed = dict(zero); changed[key] = bad
        assert not valid_trace(changed, False), (key, bad)
        mutations += 1
    active = valid[2]
    for path, bad in (("sequence", True), ("sequence", 1), ("sequence", -1), ("iteration", True), ("iteration", 201)):
        changed = copy.deepcopy(active); changed["last_complete"][path] = bad
        assert not valid_trace(changed, True)
        mutations += 1
    for bad in (True, float("inf"), float("nan")):
        changed = copy.deepcopy(active); changed["last_complete"]["quality"]["objective"] = bad
        assert not valid_trace(changed, True)
        mutations += 1
    changed = copy.deepcopy(active); changed["last_complete"]["quality"]["max_row_violation"] = -1
    assert not valid_trace(changed, True)
    mutations += 1
    assert len(list(schedule(plan))) == 18 and len({case_index for _, _, case_index, _, _ in schedule(plan)}) == 6
    assert [variant for *_, variant in schedule(plan)][:3] == list(VARIANTS)
    polluted = {"solver_v88_presolve": "9", "SOLVER_V89_ASSESS_TRACE": "9", "OpenBlas_Coretype": "x", "mkl_num_threads": "8", "PATH": "保留"}
    for variant in VARIANTS:
        cleaned = clean_environment(polluted, variant)
        assert cleaned == dict(PATH="保留", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **variant_environment(variant))
    before = dict(os.environ)
    with solver_environment("diagnostic_on"):
        with solver_environment("diagnostic_off"):
            assert "SOLVER_V89_ASSESS_TRACE" not in os.environ and os.environ["SOLVER_V89_PRESOLVE"] == "1"
        assert os.environ["SOLVER_V89_ASSESS_TRACE"] == "1"
    assert dict(os.environ) == before
    value = dict(status="accepted", timings=dict(total=9), oracle=dict(total_seconds=4, events=[dict(timings=dict(total=3), budget=dict(checks=8))]), assess_trace=active)
    changed = copy.deepcopy(value); changed["timings"]["total"] = 2; changed["oracle"]["total_seconds"] = 1
    changed["oracle"]["events"][0]["timings"]["total"] = 1
    assert same(non_timing_marker(value), non_timing_marker(changed))
    changed["oracle"]["events"][0]["budget"]["checks"] += 1
    assert not same(non_timing_marker(value), non_timing_marker(changed))
    assert non_timing_marker(dict(version="V89_PRESOLVE v89")) == dict(version="V88_PRESOLVE v88")
    invalid_budget = dict(report=None, error="wall_timeout", classification="wall_timeout", elapsed_wall_seconds=36,
                          numerical_seconds=None, presolve_marker=None)
    assert pair_result(invalid_budget, invalid_budget, plan)["category"] == "invalid_process"
    # 真实marker解析覆盖旧21键与新22键，不通过简化占位对象替代父协议。
    marker = parent.empty_marker(True)
    for variant in VARIANTS:
        trace = dict(zero, enabled=variant == "diagnostic_on")
        payload = dict(marker, assess_trace=trace) if variant != "frozen_v88" else marker
        prefix = "V88_PRESOLVE " if variant == "frozen_v88" else "V89_PRESOLVE "
        parsed, error, observed, trace_error = inspect_marker(prefix + json.dumps(payload), variant)
        assert error is None and trace_error is None and same(parsed, payload)
        assert same(observed, None if variant == "frozen_v88" else trace)
        assert inspect_marker(prefix + json.dumps(payload) + "\n污染", variant)[1] == "marker_count"
    assert expected_options(plan)["ipm"]["linear_solver"] == "normal_cholmod_aat"
    print(json.dumps(dict(passed=True, public_lp_runs=0, trace_valid_fixtures=len(valid), trace_mutations=mutations,
        scheduled_runs=18, comparison_pairs=12, environment_isolated=True, non_timing_policy_checked=True), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-test", "self-check", "run", "verify"))
    parser.add_argument("--work", type=Path, default=WORK)
    arguments = parser.parse_args()
    if arguments.action in ("self-test", "self-check"):
        self_check()
    elif arguments.action == "run":
        run(arguments.work.resolve())
    else:
        manifest, summary, rows = archive()
        print(json.dumps(dict(passed=True, public_lp_runs=0, runs=len(rows), pairs=summary["pairs"], compatibility_passed=summary["compatibility_passed"]), ensure_ascii=False))
