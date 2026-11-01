"""第91轮：完整物理方向LD桥与原固定500条条件扩集协议。"""
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
WORK = ROOT / ".tools/v94-presolve"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
PLAN_SHA = "0d9939a8a51a4ce13514e466c78301ad958f329c2a005d0cbb81cbabc0b9b02f"
CLARIFICATION_SHA = "be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8"
PARENT_RUN_SHA = "6080b97380f79d64b3aafd79a265554c29910fbbd767477ed0ee5b0cacfc2d04"
PARENT_INST_SHA = "8d058f0c6c1eaaefe5bf92662775bb55e642e3e48e38c20fcc608586f6eac97e"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER, COUPLED_MARKER = "V94_PRESOLVE ", "V94_COUPLED "
ENERGY_MARKER = "V94_ENERGY "
ENVIRONMENT = "SOLVER_V94_PRESOLVE"
COUPLED_COUNTS = ("constant_calls_started", "constant_calls_completed", "directions_started", "directions_completed",
    "preconditioner_started", "preconditioner_completed", "products_started", "products_completed",
    "krylov_started", "krylov_completed", "cycles_started", "true_checks_started", "true_checks_completed")
COUPLED_FIELDS = ("schema", "enabled", *COUPLED_COUNTS, "last")
LAST_FIELDS = ("attempt_iteration", "role", "status", "dimension", "initial_measure", "final_measure", "block_measures", "tolerances")
COUPLED_STATUSES = ("running", "accepted", "exhausted", "breakdown", "numerical_error", "exception", "budget")
REQUIRED_COUPLED_CASES = {
    "wide_direction_physical_dense": "transaction",
    "wide_direction_consumers": "mapping",
    "wide_direction_scope_budget": "budget",
}
ENERGY_COUNTS = ("started", "completed", "primal_terms", "row_terms", "budget_checks")
ENERGY_FIELDS = ("schema", "enabled", *ENERGY_COUNTS, "last")
ENERGY_LAST_FIELDS = ("attempt_iteration", "status", "columns", "rows", "primal_regularization", "primal_terms",
    "row_terms", "primal_energy", "row_energy", "denominator")
ENERGY_STATUSES = ("running", "completed", "numerical_error", "exception", "budget")
ENERGY_STANDALONE_CASES = ("energy_nominal_schur_identity", "energy_stored_diagonal_contract", "energy_budget_partial_boundaries")
WIDE_STANDALONE_CASES = ("wide_exact_initial_physical", "wide_fgmres_variable_preconditioner",
    "wide_block_tolerances_are_independent", "wide_happy_breakdown_requires_true_residual",
    "wide_fixed_window_and_cycle_exhaustion", "wide_invalid_dimension_and_callback_values",
    "wide_budget_boundaries_keep_partial", "wide_pure_tau_kappa_has_search_space",
    "wide_direction_late_narrow_bridge", "wide_double_column_wide_true_product", "wide_promotion_budget_prefix")
MAPS = ("protected_before_and_after", "generated_inputs_before_and_after", "compile_inputs_after_and_final",
        "link_explicit_before_and_after", "test_link_explicit_before_and_after", "outputs")
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert sha(DATA / "v88-run.py") == PARENT_RUN_SHA
_spec = importlib.util.spec_from_file_location("v94_frozen_v88_protocol", DATA / "v88-run.py")
predecessor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predecessor)
base, benchmark, compare_lp = predecessor.base, predecessor.benchmark, predecessor.compare_lp
strict_json, read, save = predecessor.strict_json, predecessor.read, predecessor.save
integer, nullable_number = predecessor.integer, predecessor.nullable_number
valid_marker, valid_math_details = predecessor.valid_marker, predecessor.valid_math_details
specification, variants = predecessor.specification, predecessor.variants
options_for, valid_command = predecessor.options_for, predecessor.valid_command


def strict_same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def save_once(path, value):
    if path.exists():
        assert base.typed_equal(read(path), value), "禁止覆盖不一致证据"
    else:
        save(path, value)


def fixed_plan():
    assert sha(DATA / "v94-plan.json") == PLAN_SHA
    assert sha(DATA / "v92-pretest-clarification.json") == CLARIFICATION_SHA
    plan = read(DATA / "v94-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"]), (plan["math_evidence_file"], plan["math_evidence_sha256"]),
        (plan["counter_evidence_file"], plan["counter_evidence_sha256"])):
        assert sha(DATA / name) == digest, name
    assert plan["parent_version"] == plan["reference_version"] == 66 and plan["algorithm_parent_version"] == 93
    assert type(plan["candidate_version"]) is int and plan["candidate_version"] == 94 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"] and plan["new_report_fields"] == []
    assert plan["diagnostic_environment"] == ENVIRONMENT and plan["diagnostic_marker"] == MARKER.strip()
    assert plan["coupled_marker"] == COUPLED_MARKER.strip() and plan["coupled_kernel_file"] == "v94-coupled.hpp"
    assert plan["parent_instrumentation_file"] == "v93-ipm-instrumentation.py" and plan["parent_instrumentation_sha256"] == PARENT_INST_SHA
    assert plan["evidence_file"] == plan["math_evidence_file"] == "v93-closeout.json"
    assert plan["energy_marker"] == ENERGY_MARKER.strip() and plan["energy_kernel_file"] == "v93-energy.hpp"
    assert plan["parent_coupled_kernel_file"] == "v92-coupled.hpp" and plan["parent_coupled_kernel_sha256"] == sha(DATA / "v92-coupled.hpp")
    assert plan["energy_kernel_sha256"] == sha(DATA / "v93-energy.hpp")
    assert plan["inherited_coupled_clarification_file"] == "v92-pretest-clarification.json" and \
        plan["inherited_coupled_clarification_sha256"] == CLARIFICATION_SHA
    assert plan["parent_coupled_test_file"] == "v93-coupled-self-test.cpp" and sha(DATA / plan["parent_coupled_test_file"]) == plan["parent_coupled_test_sha256"]
    assert plan["parent_required_coupled_ipm_cases"] == read(DATA / "v93-plan.json")["required_coupled_ipm_cases"]
    assert plan["required_energy_standalone_cases"] == list(ENERGY_STANDALONE_CASES)
    assert plan["required_wide_standalone_cases"] == list(WIDE_STANDALONE_CASES)
    assert plan["expected_math_and_build"] == dict(ipm_cases=70, standalone_cases=22, legacy_standalone_cases=11,
        new_wide_standalone_cases=11, compiled_units=3, dependency_files=3, generated_sources=7)
    assert plan["required_coupled_ipm_cases"] == REQUIRED_COUPLED_CASES
    schema = plan["coupled_schema"]
    assert schema["schema"] == 1 and schema["top_fields"] == list(COUPLED_FIELDS) and schema["last_fields"] == list(LAST_FIELDS)
    assert schema["statuses"] == list(COUPLED_STATUSES) and schema["roles"] == ["affine", "corrector"]
    energy_schema = plan["energy_schema"]
    assert energy_schema["schema"] == 1 and energy_schema["top_fields"] == list(ENERGY_FIELDS) and energy_schema["last_fields"] == list(ENERGY_LAST_FIELDS)
    assert energy_schema["statuses"] == list(ENERGY_STATUSES) and type(energy_schema["budget_batch"]) is int and energy_schema["budget_batch"] == 4096
    assert type(energy_schema["max_calls_per_solve"]) is int and energy_schema["max_calls_per_solve"] == 200
    for name, expected in (("max_window", 32), ("max_cycles", 3), ("max_directions_per_solve", 400),
        ("max_preconditioners_per_direction", 97), ("max_products_per_direction", 193), ("max_true_checks_per_direction", 97)):
        assert type(schema[name]) is int and schema[name] == expected
    for name, expected in (("max_projection_trials_per_point", 1), ("max_mapped_original_assessments_per_point", 2),
        ("max_original_oracle_calls", 204), ("max_original_oracle_assessments", 408)):
        assert type(plan[name]) is int and plan[name] == expected
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"], plan["scaling_passes"]) == (5, 35, 200, 1, 0, 5)
    for side in ("candidate", "reference", "compatibility"):
        assert plan[side + "_accuracy"] == "homogeneous_repair" and plan[side + "_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert plan[side + "_backend_version"] == "ipm-cpu-0.53.0" and plan[side + "_objective_scaling"] == "off"
    assert plan["method"] == "homogeneous" and plan["stage_order"] == list(STAGES)
    assert plan["maximum_solver_runs"] == 500 and plan["sets"] == read(DATA / "v66-plan.json")["sets"]
    assert plan["gate"] == read(DATA / "v88-plan.json")["gate"]
    return plan


def clean_environment(prior, enabled):
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    clean = {key: value for key, value in prior.items() if not key.upper().startswith("SOLVER_V")
        and key.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    clean.update(dict.fromkeys(threads, "1"))
    if enabled:
        clean[ENVIRONMENT] = "1"
    return clean


@contextlib.contextmanager
def solver_environment(enabled):
    prior = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(clean_environment(prior, enabled))
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def empty_coupled(enabled=False):
    return dict(schema=1, enabled=enabled, **dict.fromkeys(COUPLED_COUNTS, 0), last=None)


def nonnegative(value):
    return nullable_number(value) and (value is None or value >= 0)


def valid_coupled(value, enabled):
    if not isinstance(value, dict) or set(value) != set(COUPLED_FIELDS) or type(value["schema"]) is not int or value["schema"] != 1 or \
        type(value["enabled"]) is not bool or value["enabled"] is not enabled or any(not integer(value[name]) for name in COUPLED_COUNTS):
        return False
    if not enabled:
        return strict_same(value, empty_coupled())
    for name in ("constant_calls", "directions", "preconditioner", "products", "true_checks"):
        started, completed = value[name + "_started"], value[name + "_completed"]
        if not completed <= started <= completed + 1:
            return False
    if value["krylov_completed"] > value["krylov_started"] or value["constant_calls_started"] > 200 or value["directions_started"] > 400:
        return False
    directions = value["directions_started"]
    if directions > 2 * value["constant_calls_completed"] or value["preconditioner_started"] > 97 * directions or \
        value["products_started"] > 193 * directions or value["true_checks_started"] > 97 * directions or \
        value["krylov_started"] > 96 * directions or value["cycles_started"] > 3 * directions:
        return False
    if value["krylov_started"] > 32 * value["cycles_started"] or value["krylov_started"] < value["cycles_started"] - 1 or \
        value["true_checks_started"] > value["krylov_started"] + directions:
        return False
    if not 0 <= value["krylov_started"] + directions - value["preconditioner_started"] <= 1 or \
        not 0 <= value["krylov_started"] + value["true_checks_started"] - value["products_started"] <= 1:
        return False
    if value["directions_completed"] > value["true_checks_completed"]:
        return False
    last = value["last"]
    if (last is None) != (directions == 0):
        return False
    if last is None:
        return all(value[name] == 0 for name in COUPLED_COUNTS if not name.startswith("constant_calls"))
    if not isinstance(last, dict) or set(last) != set(LAST_FIELDS) or \
        not integer(last["attempt_iteration"], 200) or last["attempt_iteration"] == 0 or \
        type(last["role"]) is not str or last["role"] not in {"affine", "corrector"} or \
        type(last["status"]) is not str or last["status"] not in COUPLED_STATUSES or not integer(last["dimension"]) or last["dimension"] < 2:
        return False
    if any(not nonnegative(last[name]) for name in ("initial_measure", "final_measure")) or \
        not isinstance(last["block_measures"], list) or len(last["block_measures"]) != 5 or any(not nonnegative(item) for item in last["block_measures"]) or \
        not isinstance(last["tolerances"], list) or len(last["tolerances"]) != 5 or \
        any(not benchmark.finite_real(item) or item <= 0 for item in last["tolerances"]):
        return False
    if last["initial_measure"] is not None and last["final_measure"] is not None and last["final_measure"] > last["initial_measure"]:
        return False
    if all(item is not None for item in last["block_measures"]) and last["final_measure"] != max(last["block_measures"]):
        return False
    if last["status"] == "accepted":
        return value["directions_completed"] == directions and last["final_measure"] is not None and last["final_measure"] <= 1 and \
            all(item is not None and item <= 1 for item in last["block_measures"])
    return value["directions_completed"] + 1 == directions


def empty_energy(enabled=False):
    return dict(schema=1, enabled=enabled, **dict.fromkeys(ENERGY_COUNTS, 0), last=None)


def valid_energy(value, enabled):
    if not isinstance(value, dict) or set(value) != set(ENERGY_FIELDS) or type(value["schema"]) is not int or value["schema"] != 1 or \
        type(value["enabled"]) is not bool or value["enabled"] is not enabled or any(not integer(value[name]) for name in ENERGY_COUNTS):
        return False
    if not enabled:
        return strict_same(value, empty_energy())
    if not value["completed"] <= value["started"] <= value["completed"] + 1 or value["started"] > 200:
        return False
    last = value["last"]
    if (last is None) != (value["started"] == 0):
        return False
    if last is None:
        return all(value[name] == 0 for name in ENERGY_COUNTS)
    if not isinstance(last, dict) or set(last) != set(ENERGY_LAST_FIELDS) or \
        not integer(last["attempt_iteration"], 200) or last["attempt_iteration"] == 0 or \
        type(last["status"]) is not str or last["status"] not in ENERGY_STATUSES or \
        any(not integer(last[name]) for name in ("columns", "rows", "primal_terms", "row_terms")):
        return False
    n, m = last["columns"], last["rows"]
    if last["primal_terms"] > n or last["row_terms"] > m or not nullable_number(last["primal_regularization"]) or \
        any(not nonnegative(last[name]) for name in ("primal_energy", "row_energy", "denominator")):
        return False
    complete = last["status"] == "completed"
    if complete is not (value["started"] == value["completed"]):
        return False
    if complete and (last["primal_terms"] != n or last["row_terms"] != m or \
        last["primal_regularization"] is None or last["primal_regularization"] < 0):
        return False
    if last["denominator"] is not None and (last["denominator"] < 1 or last["primal_terms"] != n or last["row_terms"] != m):
        return False
    # 同一wrapper只运行一个固定Cone；旧完整调用和最后partial项必须精确守恒。
    if value["primal_terms"] != value["completed"] * n + (0 if complete else last["primal_terms"]) or \
        value["row_terms"] != value["completed"] * m + (0 if complete else last["row_terms"]):
        return False
    checks = 2 + (n + 4095) // 4096 + (m + 4095) // 4096
    lower = value["completed"] * checks
    if not lower <= value["budget_checks"] <= lower + (0 if complete else checks):
        return False
    return True


def valid_energy_public(energy, coupled, report):
    if energy["enabled"] is not coupled["enabled"]:
        return False
    if not energy["enabled"]:
        return True
    if not 0 <= coupled["constant_calls_completed"] - energy["started"] <= 1 or \
        not 0 <= coupled["constant_calls_started"] - energy["completed"] <= 1 or coupled["directions_started"] > 2 * energy["completed"]:
        return False
    last = energy["last"]
    return last is None or last["attempt_iteration"] <= report["ipm_iterations"] + 1


def inspect_markers(stderr, variant):
    parsed, errors = [], []
    for prefix in (MARKER, COUPLED_MARKER, ENERGY_MARKER):
        lines = [line for line in stderr.splitlines() if line.startswith(prefix)]
        if variant in ("frozen_v66", "highs"):
            parsed.append(None)
            errors.append(None if not lines else "unexpected_marker")
            continue
        if len(lines) != 1:
            parsed.append(None); errors.append("marker_count"); continue
        try:
            parsed.append(strict_json(lines[0][len(prefix):])); errors.append(None)
        except (TypeError, ValueError):
            parsed.append(None); errors.append("invalid_marker_json")
    marker, coupled, energy = parsed
    if variant not in ("frozen_v66", "highs"):
        enabled = variant == "diagnostic_on"
        if errors[0] is None and not valid_marker(marker, enabled):
            errors[0] = "invalid_marker_schema"
        scope = enabled and isinstance(marker, dict) and marker.get("status") not in {"disabled", "bypassed"}
        if errors[1] is None and not valid_coupled(coupled, scope):
            errors[1] = "invalid_coupled_schema"
        if errors[2] is None and not valid_energy(energy, scope):
            errors[2] = "invalid_energy_schema"
    return marker, errors[0], coupled, errors[1], energy, errors[2]


def inspect_marker(stderr, variant):
    marker, error, _, _, _, _ = inspect_markers(stderr, variant)
    return marker, error


def valid_coupled_public(coupled, report):
    if not coupled["enabled"]:
        return True
    if any(report[name] != 0 for name in ("ipm_weighted_krylov_steps", "ipm_homogeneous_constant_repairs",
        "ipm_homogeneous_constant_repair_successes", "ipm_homogeneous_scaled_inexact_returns")):
        return False
    for private, public in (("constant_calls", "ipm_homogeneous_constant_solves"),
        ("preconditioner", "ipm_homogeneous_direction_solves"), ("products", "ipm_homogeneous_scaled_products")):
        if not coupled[private + "_completed"] <= report[public] <= coupled[private + "_started"] or \
            coupled[private + "_started"] - report[public] > 1:
            return False
    refinements = report["ipm_homogeneous_scaled_refinement_steps"]
    initial_calls = report["ipm_homogeneous_direction_solves"] - refinements
    if not coupled["krylov_completed"] <= refinements <= coupled["krylov_started"] or coupled["krylov_started"] - refinements > 1 or \
        not coupled["directions_completed"] <= initial_calls <= coupled["directions_started"] or \
        report["ipm_refinement_steps"] < refinements:
        return False
    if coupled["constant_calls_started"] > report["ipm_iterations"] + 1:
        return False
    last = coupled["last"]
    return last is None or last["attempt_iteration"] <= report["ipm_iterations"] + 1


def valid_process(row):
    if row.get("presolve_marker_error") is not None or row.get("coupled_marker_error") is not None or \
        row.get("energy_marker_error") is not None or row.get("configuration_matches") is not True or not base.valid_process(row):
        return False
    report = row["report"]
    if report["backend"] != "ipm":
        return True
    if not (benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report) and benchmark.valid_homogeneous_repair_work(report)):
        return False
    coupled = row.get("coupled_marker")
    energy = row.get("energy_marker")
    return (coupled is None or valid_coupled_public(coupled, report)) and \
        (energy is None or coupled is not None and valid_energy_public(energy, coupled, report))


def configuration_matches(report, stage, variant, model, plan):
    return predecessor.configuration_matches(report, stage, variant, model, plan)


def pair_result(a, b, plan):
    answer = base.pair_result(a, b, plan)
    if not valid_process(a) or not valid_process(b):
        answer.update(valid_processes=False, category="invalid_process")
    return answer


def quality_fields(row, model, expected_solver, plan):
    numerical = compare_lp.numerical_seconds(row)
    failure = compare_lp.quality_failure(row, model, row["model_sha256_after"], expected_solver,
        row["solver_sha256_after"] if row["solver_sha256_before"] == row["solver_sha256_after"] else None)
    wall_exceeded = row["elapsed_wall_seconds"] > plan["wall_timeout"]
    numerical_exceeded = numerical > plan["time_limit"] if benchmark.finite_real(numerical) else None
    if not valid_process(dict(row, numerical_seconds=numerical)):
        failure = "invalid_process"
    elif failure is None and wall_exceeded:
        failure = "wall_budget_exceeded"
    elif failure is None and numerical_exceeded:
        failure = "numerical_budget_exceeded"
    return dict(numerical_seconds=numerical, numerical_budget_exceeded=numerical_exceeded, wall_budget_exceeded=wall_exceeded,
        quality_failure=failure, quality_pass=failure is None,
        quality_penalized_wall_seconds=min(row["elapsed_wall_seconds"], plan["wall_timeout"]) if failure is None else plan["wall_timeout"],
        quality_penalized_numerical_seconds=min(numerical, plan["time_limit"]) if failure is None else plan["time_limit"])


def valid_kernel_details(value):
    return isinstance(value, dict) and set(value) == {"passed", "public_lp_runs", "cases", "default_limits", "details"} and \
        value["passed"] is True and type(value["public_lp_runs"]) is int and value["public_lp_runs"] == 0 and \
        type(value["cases"]) is int and value["cases"] == 22 and isinstance(value["details"], list) and \
        all(type(item) is str and item for item in value["details"]) and \
        len(value["details"]) == len(set(value["details"])) == value["cases"] and \
        isinstance(value["default_limits"], dict) and set(value["default_limits"]) == {"max_window", "max_cycles"} and \
        type(value["default_limits"]["max_window"]) is int and value["default_limits"]["max_window"] == 32 and \
        type(value["default_limits"]["max_cycles"]) is int and value["default_limits"]["max_cycles"] == 3


def valid_math_evidence(role, checks, raw, work):
    if role == "ipm":
        return predecessor.valid_math_evidence(role, checks, raw, work)
    if role != "coupled" or not isinstance(checks, dict) or set(checks) != {"passed", "public_lp_runs", "details"} or \
        checks["passed"] is not True or type(checks["public_lp_runs"]) is not int or checks["public_lp_runs"] != 0 or not isinstance(raw, dict):
        return False
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("wall_timeout") is not False or \
        raw.get("command") != [str(work / "coupled-self-test.exe")] or raw.get("cwd") != str(ROOT) or raw.get("stderr") != "":
        return False
    try:
        value = strict_json(raw["stdout"])
        return strict_same(value, checks["details"]) and valid_kernel_details(value)
    except (KeyError, TypeError, ValueError):
        return False


def freeze(work, plan):
    work = work.resolve()
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    identity_path = DATA / "ipm-v66-identity.json"
    assert sha(identity_path) == plan["parent_identity_sha256"]
    parent = read(identity_path)
    assert build["current_files"] == build["frozen_files"] == parent["files"] and build["actual_libraries"] == parent["libraries"]
    assert len(parent["files"]) == 95 and len(parent["libraries"]) == 17 and build["production_changes"] is False
    assert build["baseline_verification_sha256"] == plan["baseline_verification_sha256"]
    assert Path(build["frozen_parent_executable"]) == PARENT and build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-presolve.exe" and set(build["commands"]["compile"]) == {"ipm"}
    assert build["commands"]["link"].count(str(work / "ipm-presolve.obj")) == 1 and \
        all("normal-pair" not in str(arg) and "libsolver_ipm.a" not in str(arg) for arg in build["commands"]["link"])
    assert any(str(arg).endswith("libsolver_normal_cholesky.a") for arg in build["commands"]["link"])
    protected = {str(Path(__file__)): sha(__file__), str(DATA / "v66-run.py"): HELPER_SHA,
        str(DATA / "v88-run.py"): PARENT_RUN_SHA, str(build_path): sha(build_path), str(PARENT): plan["parent_solver_sha256"],
        str(candidate): build["executable_sha256"], str(ROOT / "scripts/benchmark.py"): sha(ROOT / "scripts/benchmark.py"),
        str(ROOT / "scripts/compare_lp.py"): sha(ROOT / "scripts/compare_lp.py")}
    for group in MAPS:
        for path, digest in build[group].items():
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for group in ("files", "libraries"):
        for name, digest in parent[group].items():
            path = str(ROOT / name)
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for name, digest in parent["files"].items():
        protected[str(work / "frozen" / name)] = digest
    for name in plan["new_code_and_plan_files"]:
        path = str(DATA / name)
        assert build["protected_before_and_after"][path] == sha(path)
    assert build["protected_before_and_after"][str(DATA / "v92-pretest-clarification.json")] == CLARIFICATION_SHA
    for name in ("static-check.json", "ipm-self-check.json", "coupled-self-check.json"):
        assert str(work / name) in protected and read(work / name)["passed"] is True
    assert set(build["math_self_tests"]) == {"ipm", "coupled"}
    for role in ("ipm", "coupled"):
        checks = read(work / (role + "-self-check.json"))
        assert valid_math_evidence(role, checks, read(work / (role + "-self-run.json")), work), "数学摘要与真实执行不符"
        assert strict_same(build["math_self_tests"][role], {**checks["details"], "run_file": role + "-self-run.json", "check_file": role + "-self-check.json"})
    meta = build["math_coverage_extension"]
    assert set(meta) == {"file", "sha256", "preserved_predecessor_cases", "new_cases"} and meta["file"] == "ipm-self-extension-check.json"
    extension_path = work / meta["file"]
    assert protected[str(extension_path)] == build["outputs"][str(extension_path)] == meta["sha256"] == sha(extension_path)
    extension = read(extension_path)
    assert set(extension) == {"passed", "public_lp_runs", "predecessor_evidence_file", "predecessor_evidence_sha256",
        "preserved_predecessor_cases", "new_cases", "required_new_cases", "component_calls"}
    assert extension["passed"] is True and type(extension["public_lp_runs"]) is int and extension["public_lp_runs"] == 0
    previous_math = read(DATA / plan["math_evidence_file"])["math_self_checks"]
    assert set(previous_math) == {"ipm", "coupled"}
    old_checks = previous_math["ipm"]
    assert old_checks["passed"] is True and valid_math_details(old_checks["details"])
    old, current = old_checks["details"], read(work / "ipm-self-check.json")["details"]
    old_cases = {item["id"]: item["group"] for item in old["details"]}
    current_cases = {item["id"]: item["group"] for item in current["details"]}
    assert len(old_cases) == old["cases"] == 67 and old_cases.items() <= current_cases.items() and REQUIRED_COUPLED_CASES.items() <= current_cases.items()
    assert not (REQUIRED_COUPLED_CASES.keys() & old_cases.keys())
    added = [item for item in current["details"] if item["id"] not in old_cases]
    assert strict_same(extension["new_cases"], added) and current["cases"] == old["cases"] + len(added) == 70
    required = extension["required_new_cases"]
    assert isinstance(required, list) and len(required) == len(REQUIRED_COUPLED_CASES) and \
        all(isinstance(item, dict) and set(item) == {"id", "group"} for item in required) and \
        {item["id"]: item["group"] for item in required} == REQUIRED_COUPLED_CASES
    assert extension["predecessor_evidence_file"] == str(DATA / plan["math_evidence_file"]) and extension["predecessor_evidence_sha256"] == plan["math_evidence_sha256"]
    assert extension["preserved_predecessor_cases"] == meta["preserved_predecessor_cases"] == old["cases"] and meta["new_cases"] == len(added)
    assert set(extension["component_calls"]) == {"component_ipm_calls", "synthetic_core_calls"}
    for name, pair in extension["component_calls"].items():
        assert pair == {"predecessor": old[name], "current": current[name], "change": current[name] - old[name]} and \
            all(type(value) is int for value in pair.values())
    parent_kernel_details = previous_math["coupled"]["details"]
    assert previous_math["coupled"]["passed"] is True and parent_kernel_details["cases"] == 11
    current_kernel_details = read(work / "coupled-self-check.json")["details"]
    assert set(parent_kernel_details["details"]) <= set(current_kernel_details["details"]) and \
        set(current_kernel_details["details"]) - set(parent_kernel_details["details"]) == set(WIDE_STANDALONE_CASES)
    assert current_kernel_details["cases"] == parent_kernel_details["cases"] + len(WIDE_STANDALONE_CASES)
    expected_generated = {str(work / name) for name in ("ipm-presolve.cpp", "v94-presolve-state.hpp", "v94-coupled-state.hpp", "v94-coupled.hpp", "v92-coupled.hpp", "v93-energy.hpp", "coupled-self.cpp")}
    assert set(build["generated_inputs_before_and_after"]) == expected_generated
    for key, name in (("private_state_header", "v94-presolve-state.hpp"), ("private_coupled_state_header", "v94-coupled-state.hpp")):
        path = str(work / name)
        assert build[key] == path and path in build["compile_inputs_after_and_final"]
    assert set(build["commands"]["original_compile"]) == {"ipm", "highs", "main"}
    assert build["commands"]["highs_include_arguments"] and all(arg.startswith("-I") for arg in build["commands"]["highs_include_arguments"])
    parent_inst = str(DATA / "v93-ipm-instrumentation.py")
    assert build["readonly_parent_instrumentation"] == {"path": parent_inst, "sha256": PARENT_INST_SHA} and protected[parent_inst] == PARENT_INST_SHA
    assert protected[str(DATA / "v92-coupled.hpp")] == plan["parent_coupled_kernel_sha256"] == sha(work / "v92-coupled.hpp")
    assert protected[str(DATA / "v94-coupled.hpp")] == sha(work / "v94-coupled.hpp")
    assert protected[str(DATA / "v93-energy.hpp")] == plan["energy_kernel_sha256"] == sha(work / "v93-energy.hpp")
    assert protected[str(DATA / plan["parent_coupled_test_file"])] == plan["parent_coupled_test_sha256"]
    for name, digest in (("v88-run.py", PARENT_RUN_SHA), (plan["evidence_file"], plan["evidence_sha256"]),
        (plan["math_evidence_file"], plan["math_evidence_sha256"]), (plan["counter_evidence_file"], plan["counter_evidence_sha256"])):
        path = str(DATA / name)
        assert path not in protected or protected[path] == digest
        protected[path] = digest
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = dict(plan_sha256=PLAN_SHA, build_manifest=str(build_path), build_manifest_sha256=sha(build_path),
        candidate=str(candidate), candidate_sha256=build["executable_sha256"], reference=str(PARENT),
        reference_sha256=plan["parent_solver_sha256"], protected=protected)
    save_once(DATA / "v94-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v94-{stage}-manifest.json")
    summary = read(DATA / f"v94-{stage}-summary.json")
    raw = DATA / f"v94-{stage}-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v94-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v94-run-identity.json") and manifest["plan_sha256"] == PLAN_SHA
    assert manifest["models"] == expected["models"] and manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert manifest["stage"] == stage and manifest["repeats"] == expected["repeats"] and manifest["variants"] == list(variants(stage))
    for field in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[field] == plan[field]
    scheduled = list(compare_lp.schedule(expected["models"], expected["repeats"], variants(stage)))
    assert len(rows) == len(scheduled)
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, scheduled)):
        executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
        digest = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
        assert (row["run_index"], row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"]) == (index, model["model_index"], repeat, position, variant)
        assert row["input"] == model["input"] and row["seed"] == plan["seed"] and row["command"][:3] == [str(executable), "solve", model["input"]]
        assert valid_command(row, variant, plan)
        assert row["presolve_environment"] == ({ENVIRONMENT: "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256")) and row["hash_error"] is None
        marker, error, coupled, coupled_error, energy, energy_error = inspect_markers(row["stderr"], variant)
        assert base.typed_equal(row["presolve_marker"], marker) and row["presolve_marker_error"] == error
        assert base.typed_equal(row["coupled_marker"], coupled) and row["coupled_marker_error"] == coupled_error
        assert base.typed_equal(row["energy_marker"], energy) and row["energy_marker_error"] == energy_error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert base.typed_equal(row["report"], strict_json(row["stdout"]))
        for field, value in quality_fields(row, model, digest, plan).items():
            assert base.typed_equal(row[field], value), (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw), valid_process_count=sum(valid_process(row) for row in rows),
        presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows),
        coupled_marker_error_count=sum(row["coupled_marker_error"] is not None for row in rows),
        energy_marker_error_count=sum(row["energy_marker_error"] is not None for row in rows))
    assert base.typed_equal(computed, summary)
    return manifest, summary, rows


def compatibility():
    plan = fixed_plan()
    pairs = []
    for stage in COMPAT:
        _, _, rows = archive(stage)
        for model in specification(stage, plan)["models"]:
            chosen = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
            a, b = (chosen[variant] for variant in variants(stage))
            pairs.append(dict(stage=stage, input=model["input"], **pair_result(a, b, plan)))
    assert len(pairs) == 17
    return dict(passed=all(pair["category"] == "equal" for pair in pairs), runs=34, pairs=pairs,
        non_budget_equal=sum(pair["category"] == "equal" for pair in pairs), budget_pairs=sum(pair["category"] == "budget_limited" for pair in pairs),
        invalid_pairs=sum(pair["category"] == "invalid_process" for pair in pairs), different_pairs=sum(pair["category"] == "different" for pair in pairs), plan_sha256=PLAN_SHA)


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    result = base.gate_decision(development, probe)
    result["plan_sha256"] = PLAN_SHA
    result["valid_processes"] = all(value["valid_process_count"] == value["run_count"] for value in (development, probe))
    result["eligible_for_netlib"] = result["eligible_for_netlib"] and result["valid_processes"]
    return result


def run(stage, work):
    plan = fixed_plan()
    identity = freeze(work, plan)
    if stage not in COMPAT:
        result = compatibility()
        save_once(DATA / "v94-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v94-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models = expected["models"]
    for item in models:
        assert item["hash_error"] is None and sha(item["input"]) == item["sha256"]
    manifest = dict(schema=1, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), plan_sha256=PLAN_SHA,
        stage=stage, identity_sha256=sha(DATA / "v94-run-identity.json"), models_file=expected["models_file"],
        models_file_sha256=expected["models_file_sha256"], models=models, repeats=expected["repeats"], variants=list(variants(stage)),
        reference_variant=variants(stage)[0], candidate_variant=variants(stage)[1],
        time_limit=plan["time_limit"], wall_timeout=plan["wall_timeout"], threads=plan["threads"], seed=plan["seed"],
        execution="串行轮转；复用benchmark.run_one的35秒进程边界，数值solve+verification超过5秒按失败计罚。")
    save(DATA / f"v94-{stage}-manifest.json", manifest)
    rows = []
    raw = DATA / f"v94-{stage}-runs.jsonl.gz"
    with gzip.open(raw, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in compare_lp.schedule(models, expected["repeats"], variants(stage)):
            executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
            expected_solver = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
            before = sha(executable)
            with solver_environment(variant == "diagnostic_on"):
                row = benchmark.run_one(options_for(stage, variant, executable, plan), Path(model["input"]), plan["seed"], len(rows))
            after, model_after = sha(executable), sha(model["input"])
            marker, marker_error, coupled, coupled_error, energy, energy_error = inspect_markers(row["stderr"], variant)
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position,
                requested_variant=variant, frozen_model_sha256=model["sha256"], model_sha256_after=model_after,
                solver_sha256_before=before, solver_sha256_after=after, frozen_solver_sha256=expected_solver,
                presolve_environment={ENVIRONMENT: "1"} if variant == "diagnostic_on" else {},
                presolve_marker=marker, presolve_marker_error=marker_error, coupled_marker=coupled, coupled_marker_error=coupled_error,
                energy_marker=energy, energy_marker_error=energy_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps(dict(stage=stage, index=row["run_index"], model=Path(model["input"]).stem, variant=variant,
                status=row["status"], quality_pass=row["quality_pass"], presolve_marker_error=marker_error,
                coupled_marker_error=coupled_error, energy_marker_error=energy_error), ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw), valid_process_count=sum(valid_process(row) for row in rows),
        presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows),
        coupled_marker_error_count=sum(row["coupled_marker_error"] is not None for row in rows),
        energy_marker_error_count=sum(row["energy_marker_error"] is not None for row in rows))
    save(DATA / f"v94-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps(dict(stage=stage, variants=summary["variants"], valid_process_count=summary["valid_process_count"]), ensure_ascii=False), flush=True)


def coupled_self_check():
    last = dict(attempt_iteration=1, role="affine", status="accepted", dimension=2, initial_measure=0.0,
        final_measure=0.0, block_measures=[0.0] * 5, tolerances=[1e-12] * 5)
    initial = dict(empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
        directions_started=1, directions_completed=1, preconditioner_started=1, preconditioner_completed=1,
        products_started=1, products_completed=1, true_checks_started=1, true_checks_completed=1, last=copy.deepcopy(last))
    krylov = copy.deepcopy(initial)
    krylov.update(preconditioner_started=2, preconditioner_completed=2, products_started=3, products_completed=3,
        krylov_started=1, krylov_completed=1, cycles_started=1, true_checks_started=2, true_checks_completed=2)
    krylov["last"].update(initial_measure=2.0, final_measure=0.5, block_measures=[0.5] * 5)
    entry = dict(empty_coupled(True), constant_calls_started=1)
    direction_entry = copy.deepcopy(initial)
    direction_entry.update(directions_completed=0, preconditioner_started=0, preconditioner_completed=0,
        products_started=0, products_completed=0, true_checks_started=0, true_checks_completed=0)
    direction_entry["last"].update(status="budget", initial_measure=None, final_measure=None, block_measures=[None] * 5)
    candidate_budget = copy.deepcopy(krylov)
    candidate_budget["directions_completed"] = 0
    candidate_budget["last"]["status"] = "budget"
    failed = copy.deepcopy(krylov)
    failed["directions_completed"] = 0
    failed["last"].update(status="exhausted", final_measure=2.0, block_measures=[2.0] * 5)
    unknown_summary = copy.deepcopy(failed)
    unknown_summary["last"].update(initial_measure=None, final_measure=None, block_measures=[None, 0.0, 0.0, 0.0, 0.0])
    valid = [empty_coupled(), empty_coupled(True), entry, initial, krylov, direction_entry, candidate_budget, failed, unknown_summary]
    for value in valid:
        assert valid_coupled(value, value["enabled"]), value
    mutations = []
    def changed(path, value, original=krylov):
        item = copy.deepcopy(original)
        cursor = item
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        mutations.append(item)
    for key in COUPLED_FIELDS:
        item = copy.deepcopy(krylov); item.pop(key); mutations.append(item)
    for key in LAST_FIELDS:
        item = copy.deepcopy(krylov); item["last"].pop(key); mutations.append(item)
    for key in COUPLED_COUNTS:
        for wrong in (True, -1, 1.0):
            changed((key,), wrong)
    for key, wrong in (("schema", True), ("schema", 2), ("enabled", 1), ("directions_started", 401),
        ("constant_calls_started", 201), ("preconditioner_started", 98), ("products_started", 194),
        ("true_checks_started", 98), ("cycles_started", 4), ("krylov_started", 97), ("directions_completed", 2)):
        changed((key,), wrong)
    for key, wrong in (("attempt_iteration", True), ("attempt_iteration", 201), ("role", "constant"), ("role", True),
        ("status", "arithmetic_error"), ("status", True), ("dimension", True), ("dimension", 1),
        ("initial_measure", -1), ("final_measure", 2.1), ("block_measures", []), ("block_measures", [True] * 5),
        ("block_measures", [None] * 5), ("tolerances", [0] * 5), ("tolerances", [True] * 5), ("tolerances", [1e-12] * 4)):
        changed(("last", key), wrong)
    for value in mutations:
        assert not valid_coupled(value, True), value
    bypass = predecessor.empty_marker(True)
    disabled = predecessor.empty_marker(False)
    for marker, coupled, variant in ((bypass, empty_coupled(), "diagnostic_on"),
        (disabled, empty_coupled(), "diagnostic_off")):
        text = MARKER + json.dumps(marker) + "\n" + COUPLED_MARKER + json.dumps(coupled) + "\n" + ENERGY_MARKER + json.dumps(empty_energy())
        observed, error, other, other_error, energy, energy_error = inspect_markers(text, variant)
        assert error is None and other_error is None and energy_error is None and \
            strict_same(observed, marker) and strict_same(other, coupled) and strict_same(energy, empty_energy())
        assert inspect_markers(text + "\n" + COUPLED_MARKER + json.dumps(coupled), variant)[3] == "marker_count"
        assert inspect_markers(text, "highs")[1] == "unexpected_marker"
    assert inspect_markers("", "highs") == (None, None, None, None, None, None)
    duplicate = COUPLED_MARKER + '{"schema":1,"schema":1}'
    assert inspect_markers(duplicate, "diagnostic_off")[3] == "invalid_marker_json"
    report = dict.fromkeys(base.COUNT_FIELDS, 0)
    report.update(ipm_iterations=0, ipm_homogeneous_constant_solves=1, ipm_homogeneous_direction_solves=1, ipm_homogeneous_scaled_products=1)
    assert valid_coupled_public(initial, report)
    for key in ("ipm_weighted_krylov_steps", "ipm_homogeneous_constant_repairs", "ipm_homogeneous_constant_repair_successes", "ipm_homogeneous_scaled_inexact_returns"):
        changed_report = dict(report); changed_report[key] = 1
        assert not valid_coupled_public(initial, changed_report)
    # 最后活动callback可能在内部预算入口前停止，caller不能伪装actual工作。
    assert valid_coupled_public(direction_entry, dict.fromkeys(base.COUNT_FIELDS, 0)) is False
    partial_report = dict.fromkeys(base.COUNT_FIELDS, 0)
    partial_report["ipm_homogeneous_constant_solves"] = 1
    assert valid_coupled_public(direction_entry, partial_report)
    kernel = dict(passed=True, public_lp_runs=0, cases=22, default_limits=dict(max_window=32, max_cycles=3), details=[f"case{i}" for i in range(22)])
    assert valid_kernel_details(kernel)
    for key, wrong in (("cases", True), ("cases", 22.0), ("details", ["same"] * 22),
        ("details", [{}] * 22), ("details", [None] * 22), ("default_limits", dict(max_window=32.0, max_cycles=3))):
        changed_kernel = dict(kernel); changed_kernel[key] = wrong
        assert not valid_kernel_details(changed_kernel)
    return dict(valid_cases=len(valid), mutation_cases=len(mutations), combined_marker_cases=8, public_caller_cases=7, kernel_math_cases=7)


def energy_self_check():
    last = dict(attempt_iteration=1, status="completed", columns=1, rows=1, primal_regularization=0.01,
        primal_terms=1, row_terms=1, primal_energy=2.0, row_energy=3.0, denominator=6.0)
    complete = dict(empty_energy(True), started=1, completed=1, primal_terms=1, row_terms=1, budget_checks=4, last=copy.deepcopy(last))
    zero = dict(empty_energy(True), started=1, completed=1, budget_checks=2,
        last=dict(last, columns=0, rows=0, primal_regularization=0.0, primal_terms=0, row_terms=0,
            primal_energy=0.0, row_energy=0.0, denominator=1.0))
    partial = dict(empty_energy(True), started=1, primal_terms=4096, budget_checks=3,
        last=dict(last, columns=4097, rows=1, primal_terms=4096, row_terms=0, status="budget",
            primal_energy=2.0, row_energy=0.0, denominator=None))
    final_budget = dict(complete, completed=0, last=dict(last, status="budget"))
    illegal_r = dict(empty_energy(True), started=1, budget_checks=1,
        last=dict(last, status="numerical_error", primal_regularization=-1.0, primal_terms=0, row_terms=0,
            primal_energy=0.0, row_energy=0.0, denominator=None))
    wide = dict(complete, last=dict(last, primal_energy=None, denominator=None))
    previous_and_partial = dict(partial, started=2, completed=1, primal_terms=8193, row_terms=1, budget_checks=8)
    valid = [empty_energy(), empty_energy(True), complete, zero, partial, final_budget, illegal_r, wide, previous_and_partial]
    for value in valid:
        assert valid_energy(value, value["enabled"]), value
    mutations = []
    def changed(path, value, original=complete):
        item = copy.deepcopy(original)
        cursor = item
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        mutations.append(item)
    for key in ENERGY_FIELDS:
        item = copy.deepcopy(complete); item.pop(key); mutations.append(item)
    for key in ENERGY_LAST_FIELDS:
        item = copy.deepcopy(complete); item["last"].pop(key); mutations.append(item)
    for key in ENERGY_COUNTS:
        for wrong in (True, -1, 1.0):
            changed((key,), wrong)
    for key, wrong in (("schema", True), ("schema", 2), ("enabled", 1), ("started", 201), ("completed", 2),
        ("primal_terms", 0), ("row_terms", 0), ("budget_checks", 3), ("budget_checks", 5)):
        changed((key,), wrong)
    for key, wrong in (("attempt_iteration", True), ("attempt_iteration", 0), ("attempt_iteration", 201),
        ("status", "accepted"), ("columns", True), ("rows", -1), ("primal_regularization", -1.0),
        ("primal_regularization", None), ("primal_terms", 0), ("row_terms", 0), ("primal_energy", -1.0),
        ("row_energy", True), ("denominator", 0.5)):
        changed(("last", key), wrong)
    for value in mutations:
        assert not valid_energy(value, True), value
    coupled = dict(empty_coupled(True), constant_calls_started=1, constant_calls_completed=1)
    report = dict.fromkeys(base.COUNT_FIELDS, 0)
    assert valid_energy_public(complete, coupled, report)
    assert valid_energy_public(empty_energy(True), coupled, report)
    assert not valid_energy_public(complete, dict(coupled, constant_calls_completed=0), report)
    assert not valid_energy_public(complete, dict(coupled, directions_started=3), report)
    assert not valid_energy_public(complete, empty_coupled(), report)
    disabled = MARKER + json.dumps(predecessor.empty_marker(False)) + "\n" + COUPLED_MARKER + json.dumps(empty_coupled())
    text = disabled + "\n" + ENERGY_MARKER + json.dumps(empty_energy())
    assert inspect_markers(text, "diagnostic_off")[5] is None
    assert inspect_markers(disabled, "diagnostic_off")[5] == "marker_count"
    assert inspect_markers(text + "\n" + ENERGY_MARKER + json.dumps(empty_energy()), "diagnostic_off")[5] == "marker_count"
    assert inspect_markers(disabled + "\n" + ENERGY_MARKER + '{"schema":1,"schema":1}', "diagnostic_off")[5] == "invalid_marker_json"
    assert inspect_markers(text, "frozen_v66")[5] == "unexpected_marker"
    assert inspect_markers(disabled + "\n" + ENERGY_MARKER + json.dumps(empty_energy(True)), "diagnostic_off")[5] == "invalid_energy_schema"
    return dict(valid_cases=len(valid), mutation_cases=len(mutations), public_prefix_cases=5,
        marker_cases=6, nominal_completed_ld_overflow_summary_allowed=True)


def self_check():
    plan = fixed_plan()
    inherited = predecessor.self_check()
    checks = coupled_self_check()
    energy_checks = energy_self_check()
    prior = dict(os.environ)
    with solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.upper().startswith("SOLVER_V")} == {ENVIRONMENT: "1"}
        with solver_environment(False):
            assert not any(key.upper().startswith("SOLVER_V") for key in os.environ)
        assert os.environ[ENVIRONMENT] == "1"
    assert dict(os.environ) == prior
    polluted = {"solver_v1": "1", "SoLvEr_V94_PrEsOlVe": "2", "openblas_coretype": "x", "PATH": "保留", "omp_num_threads": "99"}
    assert clean_environment(polluted, False) == {"PATH": "保留", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    assert clean_environment(polluted, True) == dict(clean_environment(polluted, False), **{ENVIRONMENT: "1"})
    for stage in STAGES:
        for variant in variants(stage):
            value = options_for(stage, variant, PARENT, plan)
            assert (value.time_limit, value.wall_timeout, value.threads) == (5.0, 35, 1)
            assert value.backend == ("highs" if variant == "highs" else "ipm")
    return dict(passed=True, public_lp_runs=0, inherited_v88_checks=inherited, coupled_checks=checks, energy_checks=energy_checks,
        environment_cases=6, configuration_cases=12, off_pairs=17, maximum_solver_runs=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v94-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
