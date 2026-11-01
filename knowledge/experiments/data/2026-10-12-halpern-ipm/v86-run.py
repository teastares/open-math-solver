"""第83轮：共享预算预处理恢复候选的关闭兼容、开启性能及固定扩集门槛。"""
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
PLAN_SHA = "63aac4070053778773e1a368595c0b8f393731b217649bfd052dcd5b46e85933"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
WORK = ROOT / ".tools/v86-presolve"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V86_PRESOLVE "
ENVIRONMENT = "SOLVER_V86_PRESOLVE"
QUALITY_BOOLS = ("primal_valid", "primal_feasible", "objective_consistent", "dual_feasible", "gap_satisfied", "numerically_optimal")
QUALITY_NUMBERS = ("objective", "max_row_violation", "max_bound_violation", "max_dual_residual", "objective_gap", "relative_objective_gap")
NATIVE_ITERATIONS = ("simplex_iterations", "ipm_iterations", "pdlp_iterations", "crossover_iterations", "qp_iterations")
TIMINGS = ("preparation", "presolve", "child", "recovery", "verification", "child_verification", "total")
STATUSES = {"disabled", "bypassed", "running", "accepted", "checked_not_optimal", "no_candidate", "presolve_rejected", "native_error", "time_limit", "exception", "not_started"}
PHASES = {"not_started", "bypass", "preparation", "pass_model", "presolve", "child", "candidate", "recovery", "verification", "commit", "finished"}
NATIVE_OPTIONS = dict(primal_feasibility_tolerance=1e-7, dual_feasibility_tolerance=1e-7,
    ipm_optimality_tolerance=1e-8, lp_presolve_requires_basis_postsolve=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("JSON键重复：" + key)
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique, parse_constant=benchmark.invalid_constant,
        parse_float=benchmark.finite_json_float)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_once(path, value):
    if path.exists():
        assert base.typed_equal(read(path), value), "禁止覆盖不一致证据"
    else:
        save(path, value)


assert sha(DATA / "v66-run.py") == HELPER_SHA
_spec = importlib.util.spec_from_file_location("v86_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def fixed_plan():
    assert sha(DATA / "v86-plan.json") == PLAN_SHA
    plan = read(DATA / "v86-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert type(plan["candidate_version"]) is int and plan["candidate_version"] == 86 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V86_PRESOLVE"
    assert plan["diagnostic_marker"] == "V86_PRESOLVE"
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"]) == (5, 35, 200, 1, 0)
    for side in ("candidate", "reference", "compatibility"):
        assert plan[side + "_accuracy"] == "homogeneous_repair"
        assert plan[side + "_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert plan[side + "_backend_version"] == "ipm-cpu-0.53.0"
        assert plan[side + "_objective_scaling"] == "off"
    assert plan["method"] == "homogeneous" and plan["stage_order"] == list(STAGES)
    assert plan["maximum_solver_runs"] == 500
    original = read(DATA / "v66-plan.json")
    assert plan["sets"] == original["sets"]
    return plan


def specification(stage, plan):
    return base.specification(stage, plan)


def variants(stage):
    return ("highs" if stage == "highs" else "frozen_v66", "diagnostic_off" if stage in COMPAT else "diagnostic_on")


def options_for(stage, variant, executable, plan):
    backend = "highs" if variant == "highs" else "ipm"
    values = dict(solver=executable, backend=backend, time_limit=float(plan["time_limit"]),
        wall_timeout=plan["wall_timeout"], threads=plan["threads"], mip_gap=0.0001,
        iteration_limit=None, scaling_passes=None)
    if backend == "ipm":
        values.update(iteration_limit=plan["max_iterations"], scaling_passes=plan["scaling_passes"],
            ipm_method=plan["method"], ipm_linear_accuracy="homogeneous_repair",
            ipm_linear_solver="normal_cholmod_aat", ipm_objective_scaling="off", ipm_direction_retries=0)
        for key, value in plan["shared_ipm"].items():
            values["ipm_" + key.replace("-", "_")] = int(value) if key in ("central-corrections", "krylov-history-size") else value
    else:
        values.update({"highs_" + key.replace("-", "_"): value for key, value in plan["highs_reference"].items()})
    return argparse.Namespace(**values)


def valid_command(row, variant, plan):
    args = row["command"]
    if len(args[3:]) % 2:
        return False
    flags = dict(zip(args[3::2], args[4::2]))
    if len(flags) * 2 != len(args[3:]):
        return False
    backend = "highs" if variant == "highs" else "ipm"
    expected = {"--backend": backend, "--time-limit": "5.0", "--threads": "1", "--seed": "0", "--mip-gap": "0.0001"}
    if backend == "ipm":
        expected.update({"--iteration-limit": "200", "--scaling-passes": "5", "--ipm-method": "homogeneous",
            "--ipm-linear-accuracy": "homogeneous_repair", "--ipm-linear-solver": "normal_cholmod_aat",
            "--ipm-direction-retries": "0", "--ipm-objective-scaling": "off"})
        expected.update({"--ipm-" + key: value for key, value in plan["shared_ipm"].items()})
    else:
        expected.update({"--highs-" + key: value for key, value in plan["highs_reference"].items()})
    return flags == expected


@contextlib.contextmanager
def solver_environment(enabled):
    """单进程串行作用域：每次调用重建环境，结束后逐字恢复。"""
    prior = dict(os.environ)
    clean = clean_environment(prior, enabled)
    try:
        os.environ.clear()
        os.environ.update(clean)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def clean_environment(prior, enabled):
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    clean = {key: value for key, value in prior.items() if not key.upper().startswith("SOLVER_V")
        and key.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    clean.update(dict.fromkeys(threads, "1"))
    if enabled:
        clean[ENVIRONMENT] = "1"
    return clean


def inspect_marker(stderr, variant):
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if variant in ("frozen_v66", "highs"):
        return (None, None) if not lines else (None, "unexpected_marker")
    if len(lines) != 1:
        return None, "marker_count"
    try:
        value = strict_json(lines[0][len(MARKER):])
    except (TypeError, ValueError):
        return None, "invalid_marker_json"
    if not valid_marker(value, variant == "diagnostic_on"):
        return value, "invalid_marker_schema"
    return value, None


def integer(value, maximum=(1 << 63) - 1):
    return type(value) is int and 0 <= value <= maximum


def nullable_number(value):
    return value is None or benchmark.finite_real(value)


def valid_dimensions(value):
    return value is None or isinstance(value, dict) and set(value) == {"n", "m", "nnz"} and all(integer(item) for item in value.values())


def valid_quality(value):
    if value is None: return True
    return isinstance(value, dict) and set(value) == {*QUALITY_BOOLS, *QUALITY_NUMBERS} and \
        all(type(value[key]) is bool for key in QUALITY_BOOLS) and all(nullable_number(value[key]) for key in QUALITY_NUMBERS) and \
        all(value[key] is None or value[key] >= 0 for key in ("max_row_violation", "max_bound_violation", "max_dual_residual"))


def valid_call(value, with_status=True):
    if not isinstance(value, dict) or set(value) != ({"started", "completed", "status"} if with_status else {"started", "completed"}):
        return False
    if not integer(value["started"], 1) or not integer(value["completed"], 1) or value["completed"] > value["started"]:
        return False
    if with_status:
        return (value["status"] is None) is (value["completed"] == 0) and (value["status"] is None or type(value["status"]) is int and value["status"] in (-1, 0, 1))
    return True


def valid_native(value):
    if not isinstance(value, dict) or set(value) != {"presolve_status", "model_status", "info_valid", "basis_valid", "time_limit", "options", *NATIVE_ITERATIONS}:
        return False
    if any(value[key] is not None and (type(value[key]) is not int or not -1 <= value[key] <= (1 << 63) - 1) for key in
        ("presolve_status", "model_status", *NATIVE_ITERATIONS)):
        return False
    if any(value[key] is not None and type(value[key]) is not bool for key in ("info_valid", "basis_valid")):
        return False
    if value["time_limit"] is not None and (not benchmark.finite_real(value["time_limit"]) or value["time_limit"] <= 0):
        return False
    options = value["options"]
    return options is None or isinstance(options, dict) and set(options) == set(NATIVE_OPTIONS) and \
        type(options["lp_presolve_requires_basis_postsolve"]) is bool and options["lp_presolve_requires_basis_postsolve"] is True and \
        all(benchmark.finite_real(options[key]) and options[key] == NATIVE_OPTIONS[key] for key in NATIVE_OPTIONS if key != "lp_presolve_requires_basis_postsolve")


def valid_marker(value, enabled):
    fields = {"schema", "enabled", "attempted", "accepted", "status", "phase", "route", "original", "reduced", "pass_model", "presolve", "postsolve",
        "native", "child", "assessments", "original_quality", "timings", "budget", "explicit_highs_lp_solve_calls"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema"]) is not int or value["schema"] != 1 or \
        type(value["enabled"]) is not bool or value["enabled"] is not enabled or any(type(value[key]) is not bool for key in ("attempted", "accepted")) or \
        type(value["explicit_highs_lp_solve_calls"]) is not int or value["explicit_highs_lp_solve_calls"] != 0:
        return False
    if type(value["status"]) is not str or value["status"] not in STATUSES or type(value["phase"]) is not str or value["phase"] not in PHASES or \
        type(value["route"]) is not str or value["route"] not in {"none", "original", "reduced", "empty"}:
        return False
    if not valid_dimensions(value["original"]) or not valid_dimensions(value["reduced"]) or not valid_native(value["native"]) or not valid_quality(value["original_quality"]):
        return False
    if any(not valid_call(value[key]) for key in ("pass_model", "presolve", "postsolve")) or not valid_call(value["assessments"], False):
        return False
    child = value["child"]
    if not isinstance(child, dict) or set(child) != {"started", "completed", "status", "has_solution"} or \
        not integer(child["started"], 1) or not integer(child["completed"], 1) or child["completed"] > child["started"] or \
        child["status"] is not None and (type(child["status"]) is not str or not child["status"]) or \
        child["has_solution"] is not None and type(child["has_solution"]) is not bool:
        return False
    if child["completed"] == 0 and (child["status"] is not None or child["has_solution"] is not None):
        return False
    timings, budget = value["timings"], value["budget"]
    if not isinstance(timings, dict) or set(timings) != set(TIMINGS) or any(not benchmark.finite_real(item) or item < 0 for item in timings.values()) or \
        not isinstance(budget, dict) or set(budget) != {"checks", "loop_units"} or any(not integer(item) for item in budget.values()) or \
        budget["checks"] < budget["loop_units"] // 256:
        return False
    # child_verification是child的子项，不参与阶段求和；只比较真实非重叠阶段。
    shell_exception = value["attempted"] and value["status"] == "exception" and value["phase"] == "preparation" and timings["total"] == 0 and \
        value["route"] == "none" and all(value[key]["started"] == 0 for key in ("pass_model", "presolve", "postsolve", "child", "assessments")) and \
        all(item is None for item in value["native"].values()) and all(timings[key] == 0 for key in TIMINGS if key != "preparation")
    if timings["child_verification"] > timings["child"] + 1e-8 or not shell_exception and sum(timings[key] for key in TIMINGS if key not in ("child_verification", "total")) > timings["total"] + 1e-8:
        return False
    if not value["attempted"]:
        if value["accepted"] or value["route"] != "none" or value["original"] is not None or value["reduced"] is not None or value["original_quality"] is not None:
            return False
        if any(value[key]["started"] or value[key]["completed"] for key in ("pass_model", "presolve", "postsolve", "child", "assessments")) or \
            any(item is not None for item in value["native"].values()) or any(timings.values()) or any(budget.values()):
            return False
        return value["phase"] == ("not_started" if not enabled or value["status"] == "exception" else "bypass") and \
            value["status"] in ({"bypassed", "exception"} if enabled else {"disabled"})
    if not enabled or value["original"] is None or value["status"] in {"disabled", "bypassed", "not_started"}:
        return False
    passed, pre, post, assessed = (value[key] for key in ("pass_model", "presolve", "postsolve", "assessments"))
    if pre["started"] and passed["completed"] != 1 or post["started"] and pre["completed"] != 1 or assessed["started"] and post["completed"] != 1:
        return False
    if child["started"] and (pre["completed"] != 1 or value["route"] not in {"original", "reduced"} or value["reduced"] is None):
        return False
    if value["route"] == "original":
        if not base.typed_equal(value["original"], value["reduced"]) or post["started"] or assessed["started"]:
            return False
        if value["original_quality"] is not None and child["completed"] != 1:
            return False
    if value["route"] == "empty" and (value["reduced"] != dict(n=0, m=0, nnz=0) or child["started"]):
        return False
    if post["started"] and (value["route"] not in {"reduced", "empty"} or value["reduced"] is None or \
        value["route"] == "reduced" and (child["completed"] != 1 or child["has_solution"] is not True)):
        return False
    if value["route"] != "original" and (assessed["completed"] == 1) is not (value["original_quality"] is not None):
        return False
    native = value["native"]
    if value["route"] != "none" and (pre["completed"] != 1 or native["presolve_status"] != {"original": 0, "reduced": 3, "empty": 4}[value["route"]]):
        return False
    if any(native[key] is not None and native[key] > 0 for key in NATIVE_ITERATIONS) or native["basis_valid"] is True:
        if value["status"] not in {"native_error", "time_limit", "exception"} or value["accepted"]:
            return False
    if value["accepted"]:
        return value["status"] == "accepted" and value["phase"] == "finished" and value["route"] in {"original", "reduced", "empty"} and \
            value["original_quality"] is not None and all(value["original_quality"][key] is True for key in QUALITY_BOOLS) and \
            (child["completed"] == 1 and child["status"] == "optimal" and child["has_solution"] is True if value["route"] == "original" else assessed["completed"] == 1)
    return value["status"] != "accepted"

def valid_process(row):
    if row.get("presolve_marker_error") is not None or row.get("configuration_matches") is not True:
        return False
    if not base.valid_process(row):
        return False
    report = row["report"]
    return report["backend"] != "ipm" or benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report) and benchmark.valid_homogeneous_repair_work(report)


def configuration_matches(report, stage, variant, model, plan):
    side = "reference" if variant == variants(stage)[0] else "candidate"
    backend = "highs" if variant == "highs" else "ipm"
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == backend \
        and report.get("backend_version") == ("1.15.1" if backend == "highs" else "ipm-cpu-0.53.0") \
        and base.typed_equal(report.get("options"), base.expected_options(stage, side, plan))


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
    probe = dict(row, numerical_seconds=numerical)
    if not valid_process(probe):
        failure = "invalid_process"
    elif failure is None and wall_exceeded:
        failure = "wall_budget_exceeded"
    elif failure is None and numerical_exceeded:
        failure = "numerical_budget_exceeded"
    return dict(numerical_seconds=numerical, numerical_budget_exceeded=numerical_exceeded,
        wall_budget_exceeded=wall_exceeded, quality_failure=failure, quality_pass=failure is None,
        quality_penalized_wall_seconds=min(row["elapsed_wall_seconds"], plan["wall_timeout"]) if failure is None else plan["wall_timeout"],
        quality_penalized_numerical_seconds=min(numerical, plan["time_limit"]) if failure is None else plan["time_limit"])


def strict_same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def valid_math_details(value):
    groups = ("wrapper", "mapping", "transaction", "budget", "exception")
    fields = {"passed", "public_lp_runs", "explicit_highs_lp_solve_calls", "scope", "cases", "component_ipm_calls",
        "synthetic_core_calls", "details", *(group + "_cases" for group in groups)}
    if not isinstance(value, dict) or set(value) != fields or value["passed"] is not True or \
        type(value["public_lp_runs"]) is not int or value["public_lp_runs"] != 0 or \
        type(value["explicit_highs_lp_solve_calls"]) is not int or value["explicit_highs_lp_solve_calls"] != 0:
        return False
    if not isinstance(value["scope"], str) or not value["scope"] or any(type(value[key]) is not int or value[key] <= 0 for key in
        ("cases", "component_ipm_calls", "synthetic_core_calls", *(group + "_cases" for group in groups))):
        return False
    if not isinstance(value["details"], list) or len(value["details"]) != value["cases"] or sum(value[group + "_cases"] for group in groups) != value["cases"]:
        return False
    ids, counts = set(), dict.fromkeys(groups, 0)
    for item in value["details"]:
        if not isinstance(item, dict) or set(item) != {"id", "group"} or not isinstance(item["id"], str) or \
            not item["id"] or not item["id"].isascii() or item["id"] in ids or type(item["group"]) is not str or item["group"] not in groups:
            return False
        ids.add(item["id"]); counts[item["group"]] += 1
    return all(counts[group] == value[group + "_cases"] for group in groups)


def valid_math_evidence(role, checks, raw, work):
    if role != "ipm" or not isinstance(checks, dict) or set(checks) != {"passed", "public_lp_runs", "details"} or checks.get("passed") is not True or \
        type(checks.get("public_lp_runs")) is not int or checks["public_lp_runs"] != 0 or not isinstance(raw, dict):
        return False
    if type(raw.get("returncode")) is not int or raw["returncode"] != 0 or raw.get("wall_timeout") is not False or \
        raw.get("command") != [str(work / "ipm-self-test.exe")] or raw.get("cwd") != str(ROOT) or raw.get("stderr") != "":
        return False
    if not isinstance(raw.get("stdout"), str): return False
    try:
        value = strict_json(raw["stdout"])
        matching = strict_same(value, checks.get("details"))
    except (TypeError, ValueError):
        return False
    return matching and valid_math_details(value)


def math_evidence_self_check():
    groups = ("wrapper", "mapping", "transaction", "budget", "exception")
    value = dict(passed=True, public_lp_runs=0, explicit_highs_lp_solve_calls=0, component_ipm_calls=1, synthetic_core_calls=1,
        scope="真实组件调用与注入核心分开记账", cases=5, details=[dict(id=group + "_one", group=group) for group in groups],
        **{group + "_cases": 1 for group in groups})
    checks = dict(passed=True, public_lp_runs=0, details=value)
    raw = dict(command=[str(WORK / "ipm-self-test.exe")], cwd=str(ROOT), returncode=0, wall_timeout=False,
        stdout=json.dumps(value), stderr="")
    assert valid_math_evidence("ipm", checks, raw, WORK)
    for key, invalid in (("returncode", False), ("returncode", 1), ("wall_timeout", True), ("stderr", "污染"),
        ("command", []), ("cwd", "wrong"), ("stdout", "{}"), ("stdout", '{"passed":true,"passed":true}')):
        changed = dict(raw); changed[key] = invalid
        assert not valid_math_evidence("ipm", checks, changed, WORK)
    for key, invalid in (("public_lp_runs", False), ("public_lp_runs", 1), ("explicit_highs_lp_solve_calls", True),
        ("component_ipm_calls", 0), ("component_ipm_calls", True), ("synthetic_core_calls", True), ("wrapper_cases", 1.0),
        ("cases", 5.0), ("cases", True), ("details", []), ("scope", True)):
        changed = copy.deepcopy(value); changed[key] = invalid
        assert not valid_math_evidence("ipm", dict(checks, details=changed), dict(raw, stdout=json.dumps(changed)), WORK)
    wrong = copy.deepcopy(checks); wrong["details"]["cases"] = True
    assert not valid_math_evidence("ipm", wrong, raw, WORK)
    for key, invalid in (("id", ""), ("id", "非ASCII"), ("group", "wrong"), ("group", True)):
        changed = copy.deepcopy(value); changed["details"][0][key] = invalid
        assert not valid_math_evidence("ipm", dict(checks, details=changed), dict(raw, stdout=json.dumps(changed)), WORK)
    changed = copy.deepcopy(value); changed["details"][1]["id"] = changed["details"][0]["id"]
    assert not valid_math_details(changed)

def freeze(work, plan):
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA
    identity_path = DATA / "ipm-v66-identity.json"
    assert sha(identity_path) == plan["parent_identity_sha256"]
    parent = read(identity_path)
    assert build["current_files"] == build["frozen_files"] == parent["files"]
    assert build["actual_libraries"] == parent["libraries"]
    assert len(parent["files"]) == 95 and len(parent["libraries"]) == 17
    assert build["baseline_verification_sha256"] == plan["baseline_verification_sha256"]
    assert build["production_changes"] is False
    assert Path(build["frozen_parent_executable"]) == PARENT
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-presolve.exe"
    assert set(build["commands"]["compile"]) == {"ipm"}
    assert build["commands"]["link"].count(str(work / "ipm-presolve.obj")) == 1
    assert all("normal-pair" not in str(arg) and "libsolver_ipm.a" not in str(arg) for arg in build["commands"]["link"])
    assert any(str(arg).endswith("libsolver_normal_cholesky.a") for arg in build["commands"]["link"])
    protected = {str(Path(__file__)): sha(__file__), str(DATA / "v66-run.py"): HELPER_SHA,
        str(build_path): sha(build_path), str(PARENT): plan["parent_solver_sha256"],
        str(candidate): build["executable_sha256"],
        str(ROOT / "scripts/benchmark.py"): sha(ROOT / "scripts/benchmark.py"),
        str(ROOT / "scripts/compare_lp.py"): sha(ROOT / "scripts/compare_lp.py")}
    for group in ("protected_before_and_after", "generated_inputs_before_and_after",
        "compile_inputs_after_and_final", "link_explicit_before_and_after",
        "test_link_explicit_before_and_after", "outputs"):
        for path, digest in build[group].items():
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for group in ("files", "libraries"):
        for name, digest in parent[group].items():
            path = str(ROOT / name)
            assert path not in protected or protected[path] == digest
            protected[path] = digest
    for name, digest in parent["files"].items():
        path = str(work / "frozen" / name)
        assert path not in protected or protected[path] == digest
        protected[path] = digest
    for name in plan["new_code_and_plan_files"]:
        path = str(DATA / name)
        assert build["protected_before_and_after"][path] == sha(path)
        protected[path] = sha(path)
    for name in ("static-check.json", "ipm-self-check.json"):
        path = str(work / name)
        assert path in protected and read(path)["passed"] is True
    for role in ("ipm",):
        checks = read(work / (role + "-self-check.json"))
        assert valid_math_evidence(role, checks, read(work / (role + "-self-run.json")), work), "数学摘要与实际执行原件不同"
        assert strict_same(build["math_self_tests"][role], {**checks["details"], "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"})
    path = str(work / "v86-presolve-state.hpp")
    assert build["private_state_header"] == path and path in build["generated_inputs_before_and_after"] and path in build["compile_inputs_after_and_final"]
    assert set(build["generated_inputs_before_and_after"]) == {str(work / "ipm-presolve.cpp"), path}
    assert set(build["commands"]["original_compile"]) == {"ipm", "highs", "main"}
    assert build["commands"]["highs_include_arguments"] and all(arg.startswith("-I") for arg in build["commands"]["highs_include_arguments"])
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v86-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v86-{stage}-manifest.json")
    summary = read(DATA / f"v86-{stage}-summary.json")
    raw = DATA / f"v86-{stage}-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v86-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v86-run-identity.json")
    assert manifest["plan_sha256"] == PLAN_SHA and manifest["models"] == expected["models"]
    assert manifest["models_file_sha256"] == expected["models_file_sha256"]
    assert manifest["stage"] == stage and manifest["repeats"] == expected["repeats"]
    assert manifest["variants"] == list(variants(stage))
    for field in ("time_limit", "wall_timeout", "threads", "seed"):
        assert manifest[field] == plan[field]
    scheduled = list(compare_lp.schedule(expected["models"], expected["repeats"], variants(stage)))
    assert len(rows) == len(scheduled)
    for index, (row, (model, repeat, position, variant)) in enumerate(zip(rows, scheduled)):
        executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
        digest = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
        assert (row["run_index"], row["model_index"], row["repetition"], row["position_in_pair"], row["requested_variant"]) == (index, model["model_index"], repeat, position, variant)
        assert row["input"] == model["input"] and row["seed"] == plan["seed"]
        assert row["command"][:3] == [str(executable), "solve", model["input"]]
        assert valid_command(row, variant, plan)
        assert row["presolve_environment"] == ({"SOLVER_V86_PRESOLVE": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert base.typed_equal(row["presolve_marker"], marker) and row["presolve_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert base.typed_equal(row["report"], strict_json(row["stdout"]))
        for field, value in quality_fields(row, model, digest, plan).items():
            assert base.typed_equal(row[field], value), (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows))
    assert base.typed_equal(computed, summary)
    return manifest, summary, rows


def compatibility():
    plan = fixed_plan()
    pairs = []
    for stage in COMPAT:
        _, _, rows = archive(stage)
        for model in specification(stage, plan)["models"]:
            selected = {row["requested_variant"]: row for row in rows if row["model_index"] == model["model_index"]}
            a, b = (selected[variant] for variant in variants(stage))
            pairs.append({"stage": stage, "input": model["input"], **pair_result(a, b, plan)})
    assert len(pairs) == 17
    return {"passed": all(pair["category"] == "equal" for pair in pairs), "runs": 34, "pairs": pairs,
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "plan_sha256": PLAN_SHA}


def gate():
    _, development, _ = archive("development")
    _, probe, _ = archive("probe")
    result = base.gate_decision(development, probe)
    result["plan_sha256"] = PLAN_SHA
    result["valid_processes"] = all(item["valid_process_count"] == item["run_count"] for item in (development, probe))
    result["eligible_for_netlib"] = result["eligible_for_netlib"] and result["valid_processes"]
    return result


def run(stage, work):
    plan = fixed_plan()
    identity = freeze(work, plan)
    if stage not in COMPAT:
        result = compatibility()
        save_once(DATA / "v86-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v86-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v86-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v86-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V86_PRESOLVE=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；成批求解计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v86-{stage}-manifest.json", manifest)
    raw = DATA / f"v86-{stage}-runs.jsonl.gz"
    rows = []
    with gzip.open(raw, "xt", encoding="utf-8") as stream:
        for model, repeat, position, variant in compare_lp.schedule(models, expected["repeats"], variants(stage)):
            executable = Path(identity["candidate"] if variant.startswith("diagnostic_") else identity["reference"])
            expected_solver = identity["candidate_sha256"] if variant.startswith("diagnostic_") else identity["reference_sha256"]
            before = sha(executable)
            options = options_for(stage, variant, executable, plan)
            with solver_environment(variant == "diagnostic_on"):
                row = benchmark.run_one(options, Path(model["input"]), plan["seed"], len(rows))
            after, model_after = sha(executable), sha(model["input"])
            marker, marker_error = inspect_marker(row["stderr"], variant)
            row.update(model_index=model["model_index"], repetition=repeat, position_in_pair=position,
                requested_variant=variant, frozen_model_sha256=model["sha256"], model_sha256_after=model_after,
                solver_sha256_before=before, solver_sha256_after=after, frozen_solver_sha256=expected_solver,
                presolve_environment={"SOLVER_V86_PRESOLVE": "1"} if variant == "diagnostic_on" else {},
                presolve_marker=marker, presolve_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "presolve_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), presolve_marker_error_count=sum(row["presolve_marker_error"] is not None for row in rows))
    save(DATA / f"v86-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"], "valid_process_count": summary["valid_process_count"]}, ensure_ascii=False), flush=True)


def self_check():
    plan = fixed_plan()
    inherited = base.validation_self_check()
    math_evidence_self_check()
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    before = dict(os.environ)
    with solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V86_PRESOLVE": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V86_PRESOLVE"] == "1"
    assert dict(os.environ) == before
    polluted = {"solver_v1": "1", "SoLvEr_V86_PrEsOlVe": "2", "openblas_coretype": "x",
        "omp_num_threads": "99", "OpenBlas_Num_Threads": "88", "mkl_NUM_threads": "77", "PATH": "保留"}
    assert clean_environment(polluted, False) == {"PATH": "保留", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    assert clean_environment(polluted, True) == dict(clean_environment(polluted, False), **{ENVIRONMENT: "1"})
    for stage in STAGES:
        for variant in variants(stage):
            value = options_for(stage, variant, PARENT, plan)
            assert value.time_limit == 5.0 and value.wall_timeout == 35 and value.threads == 1
            if variant == "highs":
                assert value.backend == "highs" and value.highs_solver == "choose"
            else:
                assert value.ipm_linear_solver == "normal_cholmod_aat" and value.ipm_linear_accuracy == "homogeneous_repair"
                assert value.ipm_objective_scaling == "off" and value.ipm_direction_retries == 0
    return {"passed": True, "public_lp_runs": 0, "inherited_count_and_budget_checks": inherited,
        "environment_cases": 6, "configuration_cases": 12, "math_evidence_roles": 1, "marker_cases": marker_self_check()}


def empty_marker(enabled):
    call = dict(started=0, completed=0, status=None)
    native = {key: None for key in ("presolve_status", "model_status", "info_valid", "basis_valid", "time_limit", "options", *NATIVE_ITERATIONS)}
    return dict(schema=1, enabled=enabled, attempted=False, accepted=False, status="bypassed" if enabled else "disabled",
        phase="bypass" if enabled else "not_started", route="none", original=None, reduced=None,
        pass_model=copy.deepcopy(call), presolve=copy.deepcopy(call), postsolve=copy.deepcopy(call), native=native,
        child=dict(started=0, completed=0, status=None, has_solution=None), assessments=dict(started=0, completed=0), original_quality=None,
        timings=dict.fromkeys(TIMINGS, 0), budget=dict(checks=0, loop_units=0), explicit_highs_lp_solve_calls=0)


def marker_self_check():
    empty, bypass = empty_marker(False), empty_marker(True)
    early = copy.deepcopy(bypass); early.update(status="exception", phase="not_started")
    attempt = copy.deepcopy(bypass)
    attempt.update(attempted=True, status="exception", phase="preparation", original=dict(n=2, m=2, nnz=4))
    prepared = copy.deepcopy(attempt)
    prepared.update(status="time_limit", phase="pass_model")
    prepared["native"]["options"] = copy.deepcopy(NATIVE_OPTIONS)
    prepared["budget"].update(checks=1)
    pass_partial = copy.deepcopy(prepared); pass_partial["pass_model"]["started"] = 1
    pre_partial = copy.deepcopy(pass_partial)
    pre_partial.update(phase="presolve")
    pre_partial["pass_model"].update(completed=1, status=0)
    pre_partial["presolve"]["started"] = 1
    pre_partial["native"]["time_limit"] = 4.9
    observed = copy.deepcopy(pre_partial)
    observed["presolve"].update(completed=1, status=0)
    observed["native"].update(presolve_status=3, model_status=0, info_valid=False, basis_valid=False,
        **dict.fromkeys(NATIVE_ITERATIONS, -1))
    rejected = copy.deepcopy(observed)
    rejected.update(status="presolve_rejected")
    rejected["native"]["presolve_status"] = 1
    timeout = copy.deepcopy(observed); timeout["native"]["presolve_status"] = 5
    original = copy.deepcopy(observed)
    original.update(route="original", reduced=copy.deepcopy(original["original"]), phase="child")
    original["native"]["presolve_status"] = 0
    original_partial = copy.deepcopy(original); original_partial["child"]["started"] = 1
    original_done = copy.deepcopy(original_partial)
    original_done["child"].update(completed=1, status="optimal", has_solution=True)
    original_done.update(status="accepted", phase="finished", accepted=True)
    quality = dict(**dict.fromkeys(QUALITY_BOOLS, True), **dict.fromkeys(QUALITY_NUMBERS, 0.0))
    original_done["original_quality"] = copy.deepcopy(quality)
    no_original = copy.deepcopy(original_done)
    no_original.update(status="no_candidate", accepted=False)
    no_original["child"].update(status="numerical_error", has_solution=False)
    no_original["original_quality"] = None
    reduced = copy.deepcopy(observed)
    reduced.update(route="reduced", reduced=dict(n=1, m=1, nnz=1), phase="child")
    reduced_partial = copy.deepcopy(reduced); reduced_partial["child"]["started"] = 1
    reduced_done = copy.deepcopy(reduced_partial)
    reduced_done["child"].update(completed=1, status="optimal", has_solution=True)
    reduced_done.update(phase="candidate")
    no_candidate = copy.deepcopy(reduced_done)
    no_candidate.update(status="no_candidate", phase="finished")
    no_candidate["child"].update(status="numerical_error", has_solution=False)
    post_partial = copy.deepcopy(reduced_done)
    post_partial.update(phase="recovery"); post_partial["postsolve"]["started"] = 1
    recovered = copy.deepcopy(post_partial); recovered["postsolve"].update(completed=1, status=0)
    assessment_partial = copy.deepcopy(recovered)
    assessment_partial.update(phase="verification"); assessment_partial["assessments"]["started"] = 1
    checked = copy.deepcopy(assessment_partial)
    checked.update(phase="finished", status="checked_not_optimal")
    checked["assessments"]["completed"] = 1
    checked["original_quality"] = dict(**dict.fromkeys(QUALITY_BOOLS, False), **dict.fromkeys(QUALITY_NUMBERS, None))
    accepted = copy.deepcopy(checked)
    accepted.update(status="accepted", accepted=True)
    accepted["original_quality"] = copy.deepcopy(quality)
    accepted["timings"].update(preparation=0.1, presolve=0.1, child=0.2, recovery=0.05, verification=0.01, child_verification=0.03, total=0.5)
    accepted["budget"].update(checks=5, loop_units=512)
    empty_accepted = copy.deepcopy(accepted)
    empty_accepted.update(route="empty", reduced=dict(n=0, m=0, nnz=0))
    empty_accepted["native"]["presolve_status"] = 4
    empty_accepted["child"] = dict(started=0, completed=0, status=None, has_solution=None)
    empty_accepted["timings"].update(child=0, child_verification=0)
    after_quality = copy.deepcopy(accepted)
    after_quality.update(accepted=False, status="time_limit", phase="commit")
    after_quality["timings"]["total"] = 5.1
    native_failure = copy.deepcopy(observed)
    native_failure.update(status="native_error")
    native_failure["native"]["simplex_iterations"] = 1
    basis_failure = copy.deepcopy(observed); basis_failure.update(status="native_error"); basis_failure["native"]["basis_valid"] = True
    child_metadata_partial = copy.deepcopy(reduced_done)
    child_metadata_partial["child"].update(status=None, has_solution=None)
    shell_partial = copy.deepcopy(attempt); shell_partial["timings"]["preparation"] = 0.01
    valid = [empty, bypass, early, attempt, shell_partial, prepared, pass_partial, pre_partial, observed, rejected, timeout,
        original, original_partial, original_done, no_original, reduced, reduced_partial, reduced_done, no_candidate,
        post_partial, recovered, assessment_partial, checked, accepted, empty_accepted, after_quality, native_failure,
        basis_failure, child_metadata_partial]
    for row in valid:
        assert valid_marker(row, row["enabled"]), row
    mutations = []
    def changed(path, wrong, source=accepted):
        row = copy.deepcopy(source); part = row
        for key in path[:-1]: part = part[key]
        part[path[-1]] = wrong
        mutations.append(row)
    for key in accepted:
        row = copy.deepcopy(accepted); row.pop(key); mutations.append(row)
    for key, wrongs in (("schema", (True, 2, None)), ("enabled", (1, None)), ("attempted", (1, False)), ("accepted", (1, None)),
        ("status", ("wrong", 0)), ("phase", ("wrong", 0)), ("route", ("wrong", 0)), ("explicit_highs_lp_solve_calls", (True, 1)), ("original", (None, False))):
        for wrong in wrongs: changed((key,), wrong)
    for key in ("pass_model", "presolve", "postsolve", "child", "assessments"):
        for count in ("started", "completed"):
            for wrong in (True, -1, 2, 1.0, None): changed((key, count), wrong)
    for key in ("pass_model", "presolve", "postsolve"):
        for wrong in (True, 0.0, 2, None): changed((key, "status"), wrong)
    for key in ("n", "m", "nnz"):
        for wrong in (True, -1, 1.0): changed(("original", key), wrong)
    for key in QUALITY_BOOLS:
        for wrong in (False, 1, None): changed(("original_quality", key), wrong)
    for key in QUALITY_NUMBERS:
        for wrong in (True, float("nan"), float("inf")): changed(("original_quality", key), wrong)
    for key in ("max_row_violation", "max_bound_violation", "max_dual_residual"): changed(("original_quality", key), -1)
    for key in ("presolve_status", "model_status", *NATIVE_ITERATIONS):
        for wrong in (True, 0.0, -2): changed(("native", key), wrong)
    for key in ("info_valid", "basis_valid"): changed(("native", key), 0)
    changed(("native", "basis_valid"), True)
    changed(("native", "ipm_iterations"), 1)
    for wrong in (True, 0, float("inf")): changed(("native", "time_limit"), wrong)
    for key in NATIVE_OPTIONS:
        changed(("native", "options", key), False if key == "lp_presolve_requires_basis_postsolve" else True)
    for key in TIMINGS:
        for wrong in (True, -1, float("inf")): changed(("timings", key), wrong)
    changed(("timings", "child_verification"), 0.3)
    changed(("timings", "total"), 0.01)
    for key in ("checks", "loop_units"):
        for wrong in (True, -1, 0.5): changed(("budget", key), wrong)
    changed(("budget", "checks"), 0)
    changed(("presolve", "started"), 1, attempt)
    changed(("postsolve", "started"), 1, original_done)
    changed(("assessments", "started"), 1, reduced_done)
    changed(("reduced", "n"), 3, original_done)
    changed(("child", "started"), 1, empty_accepted)
    changed(("child", "has_solution"), False)
    changed(("child", "status"), 1)
    changed(("original_quality",), None)
    changed(("phase",), "commit")
    row = copy.deepcopy(accepted); row["unexpected"] = 0; mutations.append(row)
    for row in mutations:
        assert not valid_marker(row, True), row
    signed = copy.deepcopy(checked)
    signed["original_quality"].update(objective=-1, objective_gap=-2, relative_objective_gap=-0.5)
    assert valid_marker(signed, True)
    assert not valid_marker(empty, True) and not valid_marker(bypass, False)
    for row in valid:
        variant = "diagnostic_on" if row["enabled"] else "diagnostic_off"
        parsed, error = inspect_marker(MARKER + json.dumps(row), variant)
        assert strict_same(parsed, row) and error is None
    assert inspect_marker("", "frozen_v66") == (None, None)
    assert inspect_marker(MARKER + json.dumps(bypass), "highs")[1] == "unexpected_marker"
    duplicate = json.dumps(bypass).replace('"schema": 1', '"schema": 1, "schema": 1')
    for text in ("", MARKER + "{}", MARKER + duplicate, MARKER + json.dumps(bypass) + "\n" + MARKER + json.dumps(bypass)):
        assert inspect_marker(text, "diagnostic_on")[1] is not None
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "presolve_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    altered = copy.deepcopy(row); altered["report"]["backend_version"] = "other"
    assert pair_result(row, altered, plan)["category"] == "different"
    budget = copy.deepcopy(row); budget["status"] = budget["report"]["status"] = "time_limit"
    assert pair_result(dict(row, presolve_marker_error="marker_count"), budget, plan)["category"] == "invalid_process"
    for status, code, classification in (("numerical_error", 4, "incomplete"), ("time_limit", 4, "incomplete"),
        ("iteration_limit", 4, "incomplete"), ("verification_failed", 3, "verification_failed")):
        altered = copy.deepcopy(row)
        altered.update(status=status, returncode=code, classification=classification)
        altered["report"]["status"] = status
        assert valid_process(altered)
        altered["report"]["ipm_factorizations"] = True
        assert pair_result(altered, budget, plan)["category"] == "invalid_process"
    return dict(valid_and_parse_cases=len(valid), mutation_cases=len(mutations), sidecar_cases=len(valid) + 6, full_report_and_budget_cases=7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v86-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
