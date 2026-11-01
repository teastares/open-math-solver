"""第88轮：预处理后同Cone暖启动的三模型、两重复、三路诊断。"""
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
WORK = ROOT / ".tools/v91-presolve"
PLAN_SHA = "55ca339ebb9100e626b62080e95e9cd15ce0c2673b23be8727944920b96b68cd"
PARENT_RUN_SHA = "6080b97380f79d64b3aafd79a265554c29910fbbd767477ed0ee5b0cacfc2d04"
PARENT_INST_SHA = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
WARM_RUN_SHA = "e30c862694b39012149ad624d01c8511dca606f1ac1c852330da636adc533c9e"
CLARIFICATION_SHA = "b0cf18b7591c124ce5229c5a4671b0028b059ff6a5617218df736b97d9892906"
VARIANTS = ("frozen_v88", "diagnostic_off", "diagnostic_on")
RAW = DATA / "v91-diagnostic-runs.jsonl.gz"
MANIFEST = DATA / "v91-diagnostic-manifest.json"
SUMMARY = DATA / "v91-diagnostic-summary.json"
IDENTITY = DATA / "v91-run-identity.json"
COMPATIBILITY = DATA / "v91-compatibility.json"
MAPS = ("protected_before_and_after", "generated_inputs_before_and_after", "compile_inputs_after_and_final",
        "link_explicit_before_and_after", "test_link_explicit_before_and_after", "outputs")
REQUIRED_WARM_CASES = {
    "warm_reduced_original_oracle": "transaction",
    "warm_oracle_capacity_stop": "budget",
    "warm_accept_abort_is_fail_closed": "exception",
    "warm_local_verification_counted_once": "transaction",
    "warm_not_reduced_and_disabled_scope": "transaction",
}
WARM_LIMITS = dict(max_iterations=20000, max_work_units=500000000, check_interval=64,
    budget_check_work_interval=256, step_safety=0.9)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(DATA / "v88-run.py") == PARENT_RUN_SHA
_spec = importlib.util.spec_from_file_location("v91_frozen_v88_protocol", DATA / "v88-run.py")
parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent)
strict_json, read, save, typed_equal = parent.strict_json, parent.read, parent.save, parent.base.typed_equal
assert sha(DATA / "v84-run.py") == WARM_RUN_SHA
_warm_spec = importlib.util.spec_from_file_location("v91_frozen_v84_protocol", DATA / "v84-run.py")
warm_parent = importlib.util.module_from_spec(_warm_spec)
_warm_spec.loader.exec_module(warm_parent)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def save_once(path, value):
    if path.exists():
        assert same(read(path), value), "禁止覆盖不一致的诊断证据"
    else:
        save(path, value)


def fixed_plan():
    assert sha(DATA / "v91-plan.json") == PLAN_SHA
    assert sha(DATA / "v91-pretest-clarification.json") == CLARIFICATION_SHA
    plan = read(DATA / "v91-plan.json")
    assert (plan["parent_version"], plan["reference_version"], plan["candidate_version"]) == (66, 88, 91)
    assert plan["diagnostic_only"] is True and plan["production_changes"] is False
    assert plan["variants"] == list(VARIANTS) and plan["stage_order"] == ["diagnostic"]
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"],
            plan["scaling_passes"], plan["repeats"], plan["maximum_solver_runs"], plan["maximum_warm_assessments"]) == (5, 35, 200, 1, 0, 5, 2, 18, 314)
    assert plan["method"] == "homogeneous" and plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["new_report_fields"] == [] and plan["no_model_specific_routing"] is True
    assert plan["diagnostic_environment"] == "SOLVER_V91_PRESOLVE" and plan["warm_environment"] == "SOLVER_V91_WARM_PDHG"
    assert plan["reference_environment"] == "SOLVER_V88_PRESOLVE"
    assert plan["diagnostic_marker"] == "V91_PRESOLVE" and plan["reference_marker"] == "V88_PRESOLVE" and plan["warm_marker"] == "V91_WARM"
    assert warm_parent.strict_limits(plan["warm_limits"]) and plan["max_warm_attempts"] == 1
    assert (plan["max_original_oracle_calls"], plan["max_original_oracle_assessments"]) == (204, 408)
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
        (DATA / plan["math_evidence_file"], plan["math_evidence_sha256"]),
        (DATA / plan["warm_evidence_file"], plan["warm_evidence_sha256"]),
        (DATA / plan["warm_kernel_file"], plan["warm_kernel_sha256"]),
        (DATA / plan["warm_source_test_file"], plan["warm_source_test_sha256"]),
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
        {"SOLVER_V91_PRESOLVE": "1", **({"SOLVER_V91_WARM_PDHG": "1"} if variant == "diagnostic_on" else {})}


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


def empty_warm(enabled=False):
    stats = dict.fromkeys(warm_parent.STATS_COUNTS, 0)
    stats.update(phase="not_started", seed_available=False, last_assessed_step=None)
    return dict(schema=1, enabled=enabled, attempted=False, accepted=False, status="not_triggered" if enabled else "disabled",
        tau=None, kappa=None, kernel_status="not_started", kernel_stats=stats, seed_quality=None, last_quality=None)


def valid_warm(value, enabled):
    if not isinstance(value, dict):
        return False
    checked = copy.deepcopy(value)
    status = value.get("status")
    if type(status) is not str:
        return False
    if status in {"oracle_limit", "oracle_unavailable", "oracle_accept_aborted"}:
        if enabled is not True or value.get("accepted") is not False:
            return False
        if status == "oracle_accept_aborted" and value.get("attempted") is not True:
            return False
        if value.get("attempted") is False:
            checked["status"] = "not_triggered"
        else:
            if status in {"oracle_limit", "oracle_unavailable"} and value.get("kernel_status") != "assessment_exception":
                return False
            checked["status"] = "exception"
    elif status == "exception" and value.get("attempted") is False and enabled:
        # callback包装分配可在真正启动暖核前失败，仍保原异常和零kernel账本。
        checked["status"] = "not_triggered"
    return warm_parent.valid_marker(checked, enabled) and value["kernel_stats"]["assessments_started"] <= 314


def quality_fields(row, model, digest, plan):
    numerical = compare_lp.numerical_seconds(row)
    failure = compare_lp.quality_failure(row, model, row["model_sha256_after"], digest,
        row["solver_sha256_after"] if row["solver_sha256_before"] == row["solver_sha256_after"] else None)
    if not valid_process(dict(row, numerical_seconds=numerical)):
        failure = "invalid_process"
    elif failure is None and limited(dict(row, numerical_seconds=numerical), plan):
        failure = "budget_exceeded"
    return dict(quality_pass=failure is None, quality_failure=failure, numerical_seconds=numerical,
        numerical_budget_exceeded=numerical > 5 if benchmark.finite_real(numerical) else None,
        wall_budget_exceeded=row["elapsed_wall_seconds"] > 35)


def inspect_marker(stderr, variant):
    prefix = "V88_PRESOLVE " if variant == "frozen_v88" else "V91_PRESOLVE "
    lines = [line for line in stderr.splitlines() if line.strip()]
    primary_lines = [line for line in lines if line.startswith(prefix)]
    warm_lines = [line for line in lines if line.startswith("V91_WARM ")]
    marker, error, warm, warm_error = None, None, None, None
    expected_count = 1 if variant == "frozen_v88" else 2
    if len(lines) != expected_count or len(primary_lines) != 1 or len(warm_lines) != expected_count - 1:
        return None, "marker_count", None, "marker_count" if variant != "frozen_v88" else None
    try:
        marker = strict_json(primary_lines[0][len(prefix):])
    except (TypeError, ValueError):
        error = "invalid_marker_json"
    if error is None and not parent.valid_marker(marker, True):
        error = "invalid_marker_schema"
    if variant != "frozen_v88":
        try:
            warm = strict_json(warm_lines[0][len("V91_WARM "):])
        except (TypeError, ValueError):
            warm_error = "invalid_warm_json"
        if warm_error is None and not valid_warm(warm, variant == "diagnostic_on"):
            warm_error = "invalid_warm_schema"
    return marker, error, warm, warm_error


def valid_process(row):
    return row.get("warm_marker_error") is None and parent.valid_process(row)


def limited(row, plan):
    return row.get("classification") == "wall_timeout" or parent.base.limited(row, plan)


def pair_result(a, b, plan):
    reports = [copy.deepcopy(row.get("report")) for row in (a, b)]
    for report in reports:
        if isinstance(report, dict):
            report.pop("timings", None)
    report_equal = typed_equal(*reports) and typed_equal(a.get("returncode"), b.get("returncode")) \
        and typed_equal(a.get("error"), b.get("error"))
    valid = valid_process(a) and valid_process(b)
    budget = limited(a, plan) or limited(b, plan)
    equal = report_equal
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return dict(equal=equal, report_equal=report_equal, valid_processes=valid, budget_limited=budget, category=category)


def valid_warm_math(checks, raw, work):
    return isinstance(checks, dict) and set(checks) == {"passed", "public_lp_runs", "details"} and checks["passed"] is True \
        and type(checks["public_lp_runs"]) is int and checks["public_lp_runs"] == 0 and \
        warm_parent.valid_math_evidence("warm", dict(passed=True, lp_runs=0, details=checks["details"]), raw, work)


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
        (DATA / plan["math_evidence_file"], plan["math_evidence_sha256"]),
        (DATA / plan["warm_evidence_file"], plan["warm_evidence_sha256"]),
        (DATA / "v84-run.py", WARM_RUN_SHA),
        (DATA / "v91-pretest-clarification.json", CLARIFICATION_SHA),
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
    assert extension["predecessor_evidence_file"] == str(DATA / plan["math_evidence_file"]) and extension["predecessor_evidence_sha256"] == plan["math_evidence_sha256"]
    old = read(DATA / plan["math_evidence_file"])["math_self_checks"]["details"]
    current = checks["details"]
    assert parent.valid_math_details(old)
    old_ids = {item["id"]: item["group"] for item in old["details"]}
    new_ids = {item["id"]: item["group"] for item in current["details"]}
    assert old_ids.items() <= new_ids.items() and REQUIRED_WARM_CASES.items() <= new_ids.items()
    assert not (old_ids.keys() & REQUIRED_WARM_CASES.keys())
    added = [item for item in current["details"] if item["id"] not in old_ids]
    required = extension["required_new_cases"]
    assert isinstance(required, list) and len(required) == len(REQUIRED_WARM_CASES) and \
        all(isinstance(item, dict) and set(item) == {"id", "group"} and type(item["id"]) is str for item in required)
    assert len({item["id"] for item in required}) == len(required) and {item["id"]: item["group"] for item in required} == REQUIRED_WARM_CASES
    assert same(extension["new_cases"], added)
    assert type(meta["new_cases"]) is int and meta["new_cases"] == len(added)
    assert type(meta["preserved_predecessor_cases"]) is int and meta["preserved_predecessor_cases"] == extension["preserved_predecessor_cases"] == old["cases"]
    assert current["cases"] == old["cases"] + len(added)
    assert set(extension["component_calls"]) == {"component_ipm_calls", "synthetic_core_calls"}
    for name, pair in extension["component_calls"].items():
        assert all(type(item) is int for item in pair.values()) and pair == {"predecessor": old[name], "current": current[name], "change": current[name] - old[name]}
    header = str(work / "v91-presolve-state.hpp")
    assert build["private_state_header"] == header and header in build["compile_inputs_after_and_final"]
    assert set(build["generated_inputs_before_and_after"]) == {header, str(work / "v91-warm-state.hpp"), str(work / "ipm-presolve.cpp"),
        str(work / "v84-warm-pdhg.hpp"), str(work / "warm-self.cpp")}
    assert build["private_warm_state_header"] == str(work / "v91-warm-state.hpp")
    assert str(work / "v91-warm-state.hpp") in build["compile_inputs_after_and_final"]
    assert build["readonly_parent_instrumentation"] == {"path": str(DATA / "v88-ipm-instrumentation.py"), "sha256": PARENT_INST_SHA}
    assert protected[str(work / "v84-warm-pdhg.hpp")] == plan["warm_kernel_sha256"]
    warm_checks, warm_raw = read(work / "warm-self-check.json"), read(work / "warm-self-run.json")
    assert valid_warm_math(warm_checks, warm_raw, work)
    assert same(build["math_self_tests"]["warm"], {**warm_checks["details"], "run_file": "warm-self-run.json", "check_file": "warm-self-check.json"})
    assert set(build["math_self_tests"]) == {"ipm", "warm"}
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
        for left, right in (("frozen_v88", "diagnostic_off"),):
            pairs.append(dict(case_index=case_index, model_index=chosen[left]["model_index"], input=chosen[left]["input"],
                repetition=chosen[left]["repetition"], reference=left, candidate=right, **pair_result(chosen[left], chosen[right], plan)))
    compatibility = dict(passed=all(pair["category"] == "equal" for pair in pairs), plan_sha256=PLAN_SHA, runs=len(rows), pairs=pairs,
        non_budget_equal=sum(pair["category"] == "equal" for pair in pairs), budget_pairs=sum(pair["category"] == "budget_limited" for pair in pairs),
        invalid_pairs=sum(pair["category"] == "invalid_process" for pair in pairs), different_pairs=sum(pair["category"] == "different" for pair in pairs))
    summary = dict(schema=1, diagnostic_only=True, plan_sha256=PLAN_SHA, runs=len(rows), pairs=len(pairs), raw_sha256=sha(RAW),
        valid_process_count=sum(valid_process(row) for row in rows), marker_errors=sum(row["presolve_marker_error"] is not None for row in rows),
        warm_marker_errors=sum(row["warm_marker_error"] is not None for row in rows), compatibility_passed=compatibility["passed"],
        categories={key: compatibility[key] for key in ("non_budget_equal", "budget_pairs", "invalid_pairs", "different_pairs")},
        variants={variant: dict(runs=sum(row["variant"] == variant for row in rows),
            statuses={status: sum(row["variant"] == variant and str(row.get("status")) == status for row in rows)
                      for status in sorted({str(row.get("status")) for row in rows})}) for variant in VARIANTS},
        warm_observations=[dict(input=row["input"], repetition=row["repetition"], status=row.get("status"), quality_pass=row["quality_pass"],
            quality_failure=row["quality_failure"], route=(row["presolve_marker"] or {}).get("route"), warm_marker=row["warm_marker"],
            original_quality=(row["presolve_marker"] or {}).get("original_quality")) for row in rows if row["variant"] == "diagnostic_on"],
        untriggered_controls=[], algorithm_observations=[],
        scope="仅18条机制诊断；六frozen→off对子只排公共timings。on为算法变化，保全部质量/工作/失败。暖质量属于core.model，Reduced为约化模型；原质量另在父oracle记录。不生成公开性能或覆盖结论，不自动扩集。")
    for case_index in range(6):
        chosen = {row["variant"]: row for row in rows if row["case_index"] == case_index}
        off, on = chosen["diagnostic_off"], chosen["diagnostic_on"]
        warm = on["warm_marker"]
        changes = {key: dict(off=(off["report"] or {}).get(key), on=(on["report"] or {}).get(key)) for key in parent.base.COUNT_FIELDS
                   if not typed_equal((off["report"] or {}).get(key), (on["report"] or {}).get(key))}
        fact = dict(case_index=case_index, input=on["input"], repetition=on["repetition"], off_status=off["status"], on_status=on["status"],
            off_quality_pass=off["quality_pass"], on_quality_pass=on["quality_pass"], legacy_work_changes=changes)
        if isinstance(warm, dict) and warm.get("attempted") is False:
            summary["untriggered_controls"].append(dict(fact, **pair_result(off, on, plan)))
        else:
            summary["algorithm_observations"].append(fact)
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
        marker, marker_error, trace, warm_marker_error = inspect_marker(row["stderr"], variant)
        assert same(row["presolve_marker"], marker) and row["presolve_marker_error"] == marker_error
        assert same(row["warm_marker"], trace) and row["warm_marker_error"] == warm_marker_error
        if row["report"] is not None:
            assert same(row["report"], strict_json(row["stdout"]))
            assert row["status"] == row["report"].get("status")
            assert row["classification"] == benchmark.classify(row["returncode"], row["report"])
        assert row["configuration_matches"] is configuration_matches(row["report"], model, plan)
        for key, value in quality_fields(row, model, identity[side + "_sha256"], plan).items():
            assert typed_equal(row[key], value), (index, key)
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
        environment_rule="每次大小写无关清除全部SOLVER_V变量及OPENBLAS_CORETYPE，三线程变量置1；三路pipeline始终on，仅diagnostic_on启用warm；结束恢复原环境。",
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
            marker, marker_error, trace, warm_marker_error = inspect_marker(row["stderr"], variant)
            row.update(case_index=case_index, model_index=model["model_index"], repetition=repetition, order=order,
                variant=variant, requested_variant=variant, cwd=str(ROOT), environment=variant_environment(variant),
                solver_sha256_before=before, solver_sha256_after=after, frozen_solver_sha256=identity[side + "_sha256"],
                model_sha256_after=input_after, frozen_model_sha256=model["sha256"], presolve_marker=marker,
                presolve_marker_error=marker_error, warm_marker=trace, warm_marker_error=warm_marker_error,
                configuration_matches=configuration_matches(row["report"], model, plan))
            row.update(quality_fields(row, model, identity[side + "_sha256"], plan))
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
    print(json.dumps(dict(runs=18, pairs=6, compatibility_passed=compatibility["passed"], categories=summary["categories"]), ensure_ascii=False), flush=True)
    assert compatibility["invalid_pairs"] == compatibility["different_pairs"] == 0, "诊断不兼容，全部18次原件已保存"
    assert summary["valid_process_count"] == 18 and summary["marker_errors"] == summary["warm_marker_errors"] == 0, "含on失败的18条记录都须符合原质量/工作/状态合同"
    assert all(item["category"] == "equal" for item in summary["untriggered_controls"]), "未触发暖恢复的控制例必须保持完整公共报告"


def self_check():
    plan = fixed_plan()
    inherited = warm_parent.marker_self_check()
    off, on = empty_warm(False), empty_warm(True)
    allocated = copy.deepcopy(on)
    allocated.update(attempted=True, tau=1.0, kappa=1.0, status="exception")
    capped = copy.deepcopy(allocated)
    capped.update(status="oracle_limit", kernel_status="assessment_exception")
    capped["kernel_stats"].update(seed_available=True, phase="assessment", assessments_started=1, budget_checks=1, work_units=1)
    capped_before = dict(on, status="oracle_limit")
    unavailable = dict(on, status="oracle_unavailable")
    q = dict(completed_steps=0, **dict.fromkeys(warm_parent.QUALITY_BOOLS, True), **dict.fromkeys(warm_parent.QUALITY_NUMBERS, 0.0))
    accepted_budget = copy.deepcopy(capped)
    accepted_budget.update(status="budget_exception", kernel_status="accepted", seed_quality=copy.deepcopy(q), last_quality=copy.deepcopy(q))
    accepted_budget["kernel_stats"].update(phase="finished", assessments_completed=1, last_assessed_step=0)
    accepted_error = copy.deepcopy(accepted_budget)
    accepted_error["status"] = "oracle_accept_aborted"
    valid = [off, on, allocated, capped, capped_before, unavailable, dict(capped, status="oracle_unavailable"),
        dict(on, status="exception"), accepted_budget, accepted_error]
    for value in valid:
        assert valid_warm(value, value["enabled"]), value
    mutations = []
    for key in on:
        item = copy.deepcopy(on); item.pop(key); mutations.append(item)
    for original, key, bad in ((capped_before, "accepted", True), (unavailable, "attempted", True),
        (accepted_error, "attempted", False), (capped, "kernel_status", "accepted"), (on, "enabled", 1), (on, "status", [])):
        item = copy.deepcopy(original); item[key] = bad; mutations.append(item)
    item = copy.deepcopy(capped); item["kernel_stats"]["assessments_started"] = 315; mutations.append(item)
    for item in mutations:
        assert not valid_warm(item, item.get("enabled") is True), item
    primary = parent.empty_marker(True)
    for variant in VARIANTS:
        prefix = "V88_PRESOLVE " if variant == "frozen_v88" else "V91_PRESOLVE "
        text = prefix + json.dumps(primary) + "\n"
        warm = None if variant == "frozen_v88" else empty_warm(variant == "diagnostic_on")
        if warm is not None:
            text += "V91_WARM " + json.dumps(warm) + "\n"
        actual, error, observed, warm_error = inspect_marker(text, variant)
        assert error is None and warm_error is None and same(actual, primary) and same(observed, warm)
        assert inspect_marker(text + "污染\n", variant)[1] == "marker_count"
        if warm is not None:
            assert inspect_marker("V91_WARM " + json.dumps(warm) + "\n" + prefix + json.dumps(primary), variant)[1] is None
            assert inspect_marker(text + "V91_WARM {}\n", variant)[1] == "marker_count"
    polluted = {"solver_v88_presolve": "9", "SOLVER_V91_WARM_PDHG": "9", "OpenBlas_Coretype": "x", "mkl_num_threads": "8", "PATH": "保留"}
    for variant in VARIANTS:
        assert clean_environment(polluted, variant) == dict(PATH="保留", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", **variant_environment(variant))
    before = dict(os.environ)
    with solver_environment("diagnostic_on"):
        with solver_environment("diagnostic_off"):
            assert "SOLVER_V91_WARM_PDHG" not in os.environ and os.environ["SOLVER_V91_PRESOLVE"] == "1"
        assert os.environ["SOLVER_V91_WARM_PDHG"] == "1"
    assert dict(os.environ) == before
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(parent.base.COUNT_FIELDS, 0), "ipm_factor_backend": None, "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = dict(error=None, classification="incomplete", returncode=4, status="numerical_error", report=report,
        numerical_seconds=1, elapsed_wall_seconds=2, configuration_matches=True, presolve_marker_error=None, warm_marker_error=None,
        presolve_marker=dict(native=dict(time_limit=4.9)))
    changed = copy.deepcopy(row); changed["presolve_marker"]["native"]["time_limit"] = 4.8
    assert pair_result(row, changed, plan)["category"] == "equal"
    changed["report"]["timings"]["solve_seconds"] = 2
    assert pair_result(row, changed, plan)["category"] == "equal"
    invalid = dict(row, warm_marker_error="invalid_warm_schema", elapsed_wall_seconds=36)
    assert pair_result(invalid, row, plan)["category"] == "invalid_process"
    details = dict(passed=True, public_lp_runs=0, cases=1, default_limits=copy.deepcopy(WARM_LIMITS), details=["原核结构"])
    checks = dict(passed=True, public_lp_runs=0, details=details)
    raw = dict(command=[str(WORK / "warm-self-test.exe")], cwd=str(ROOT), returncode=0, wall_timeout=False, stdout=json.dumps(details), stderr="")
    assert valid_warm_math(checks, raw, WORK)
    for key, bad in (("returncode", True), ("stderr", "污染"), ("command", []), ("stdout", "{}")):
        assert not valid_warm_math(checks, dict(raw, **{key: bad}), WORK)
    assert len(list(schedule(plan))) == 18 and expected_options(plan)["ipm"]["linear_solver"] == "normal_cholmod_aat"
    print(json.dumps(dict(passed=True, public_lp_runs=0, inherited_warm_cases=inherited, warm_valid_cases=len(valid),
        warm_mutation_cases=len(mutations), scheduled_runs=18, off_comparison_pairs=6, environment_isolated=True,
        dynamic_native_time_not_compared=True, warm_math_evidence_cases=5), ensure_ascii=False))


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
