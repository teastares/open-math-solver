"""第77轮：原始单纯形基修复候选的关闭兼容、开启性能及固定扩集门槛。"""
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
PLAN_SHA = "c21a8c33b7ed10f704208d5312530754d6fd3e254d21c28219a29d0a51f44133"
HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
WORK = ROOT / ".tools/v80-basis"
PARENT = ROOT / ".tools/solver-ipm-v66.exe"
COMPAT = ("compatibility-development", "compatibility-probe")
STAGES = (*COMPAT, "development", "probe", "netlib", "highs")
MARKER = "V80_BASIS "
MARKER_COUNTS = ("n", "m", "input_nonzeros", "rank", "qr_q_nonzeros", "qr_r_nonzeros", "lu_l_nonzeros", "lu_u_nonzeros",
    "budget_checks", "qr_analyze_calls", "qr_factor_calls", "lu_analyze_calls", "lu_factor_calls", "primal_solves", "dual_solves",
    "slack_matvecs", "ordered_rows_count", "qr_columns_permutation_count", "selected_rows_count",
    "selection_candidates", "selection_pivots", "selection_rejected", "work_units", "l_nonzeros", "pivot_columns_count")
UNUSED_QR_COUNTS = ("qr_q_nonzeros", "qr_r_nonzeros", "qr_analyze_calls", "qr_factor_calls", "qr_columns_permutation_count")
REPAIR_COUNTS = ("attempts", "iterations", "phase1_iterations", "phase2_iterations", "pivots", "zero_steps",
    "work_units", "budget_checks", "lu_analyze_calls", "lu_factor_calls", "factor_nonzeros", "last_factor_nonzeros", "primal_solves", "dual_solves",
    "direction_solves", "slack_matvecs", "direction_matvecs", "pricing_matvecs", "roundoff_rows", "final_selected_rows_count")
REPAIR_NUMBERS = ("initial_phi", "final_phi", "initial_max_violation", "final_max_violation", "max_roundoff_bound",
    "last_right_derivative", "last_step", "max_selected_residual", "max_direction_residual",
    "last_point_error", "last_point_bound")
GUARD_COUNTS = ("checked_rows", "violations")
GUARD_COORDINATES = ("failure_row", "failure_basis_position")
GUARD_NUMBERS = ("failure_abs_residual", "failure_threshold", "failure_ratio", "max_ratio", "max_abs_residual")
INITIAL_REFINEMENT_COUNTS = ("attempts", "solves", "accepted", "double_digits", "long_double_digits")
INITIAL_REFINEMENT_STATUSES = {"not_started", "not_needed", "preparing_rhs", "unsupported_precision",
    "rhs_representation_failure", "solving", "solve_failure", "updating", "correction_nonfinite",
    "evaluating", "guard_rejected", "accepted", "budget_exception", "exception"}
FORWARD_COUNTS = ("systems", "guard_failures", "attempts", "solves", "accepted", "pivot_sequence")
FORWARD_COORDINATES = ("entering_row", "entering_basis_position")
TRANSITION_COUNTS = ("trials", "strict_accepted", "near_attempts", "near_accepted", "checked_rows")
TRANSITION_COORDINATES = ("blocking_row", "released_row", "max_slack_change_row")
TRANSITION_NUMBERS = ("old_phi", "new_phi", "delta_phi", "blocking_slack", "blocking_tau",
    "released_step", "released_tau", "max_slack_change_ratio")
TRANSITION_BOOLS = ("blocking_near", "released_near", "all_rows_near")
TRANSITION_STATUSES = {"not_started", "strict_candidate", "near_check", "blocking_not_near",
    "released_not_near", "row_change_not_near", "near_candidate", "strict_accepted", "near_accepted", "basis_repeated"}
HISTORY_COUNTS = ("states", "swaps", "candidate_checks", "hash_comparisons", "hash_matches",
    "exact_checks", "replay_swaps", "membership_comparisons", "repeats")
HISTORY_STATUSES = {"not_started", "initializing", "ready", "checking", "unique", "repeated"}
SEARCH_COUNTS = ("bases", "candidates", "trials", "non_descent", "descent", "exhausted", "basis_pivots",
    "candidate_count", "zero_ineq_rows", "zero_eq_rows")
SEARCH_COORDINATES = ("ordinal", "entering_row", "entering_basis_position")
SEARCH_NUMBERS = ("priced_rc", "linear_slope", "zero_ineq", "zero_eq", "true_derivative", "linear_minus_price")
SEARCH_BOOLS = ("derivative_complete", "step_started")
SEARCH_STATUSES = {"not_started", "preparing", "sorting", "searching", "direction", "evaluating",
    "non_descent", "descent", "step_started", "pivot_committed", "exhausted"}
ETA_COUNTS = ("prepares", "eta_trials", "full_refactor_trials", "eta_commits", "full_refactor_commits",
    "chain_length", "max_chain_length", "fallbacks", "forward_calls", "transpose_calls", "base_forward_solves",
    "base_transpose_solves", "forward_applications", "transpose_applications", "work_units", "stored_entries",
    "peak_stored_entries")
ETA_STATUSES = {"not_started", "ready", "prepare", "eta_trial", "full_trial", "commit_eta", "commit_full",
    "forward", "transpose", "apply_failure"}
ETA_MODES = {"none", "eta", "full"}
ETA_FALLBACKS = {"none", "disabled", "chain_full", "alpha_zero", "nonfinite", "unrepresentable"}
REPAIR_PHASES = {"not_started", "validation", "initial_residual", "initial_refinement", "phase1", "phase2", "complete"}
REPAIR_STATUSES = {"not_started", "budget_exception", "exception", "invalid_input", "work_limit", "factor_limit", "single_factor_limit", "arithmetic_failure",
    "basis_residual_failure", "no_descent", "no_blocker", "no_entering", "pivot_limit", "solve_failure",
    "direction_residual_failure", "zero_step_stall", "weak_pivot", "lu_failure", "lu_nonfinite",
    "point_consistency_failure", "objective_not_improved", "primal_guard_failure", "basis_repeated",
    "direction_search_exhausted", "candidate"}
RECOVERY_STATUSES = {"not_started", "budget_exception", "invalid_input", "input_limit", "invalid_interior_point",
    "candidate", "structural_rank_deficient", "selection_work_limit", "selection_norm_nonfinite", "selection_threshold_nonfinite",
    "selection_threshold_underflow", "selection_product_nonfinite", "selection_product_underflow", "selection_update_nonfinite",
    "selection_update_underflow", "selection_factor_limit", "selection_division_nonfinite", "selection_division_underflow",
    "selection_rank_deficient", "lu_failure", "lu_factor_limit", "lu_nonfinite",
    "primal_solve_failure", "dual_solve_failure", "slack_nonfinite", "repair_unavailable", "repair_invalid_candidate"}
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark
import compare_lp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_once(path, value):
    if path.exists():
        assert read(path) == value, "禁止覆盖不一致证据"
    else:
        save(path, value)


assert sha(DATA / "v66-run.py") == HELPER_SHA
_spec = importlib.util.spec_from_file_location("v80_frozen_protocol", DATA / "v66-run.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def fixed_plan():
    assert sha(DATA / "v80-plan.json") == PLAN_SHA
    plan = read(DATA / "v80-plan.json")
    for name, digest in (("ipm-v66-identity.json", plan["parent_identity_sha256"]),
        (plan["baseline_identity_file"], plan["baseline_identity_sha256"]),
        (plan["baseline_verification_file"], plan["baseline_verification_sha256"]),
        (plan["evidence_file"], plan["evidence_sha256"])):
        assert sha(DATA / name) == digest
    assert plan["parent_version"] == plan["reference_version"] == 66
    assert plan["candidate_version"] == 80 and plan["production_changes"] is False
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    assert plan["diagnostic_environment"] == "SOLVER_V80_BASIS_RECOVERY"
    assert (plan["time_limit"], plan["wall_timeout"], plan["max_iterations"], plan["threads"], plan["seed"]) == (5, 35, 200, 1, 0)
    for side in ("candidate", "reference", "compatibility"):
        assert plan[side + "_accuracy"] == "homogeneous_repair"
        assert plan[side + "_linear_solver"] == "normal_cholmod_aat"
    for side in ("candidate", "reference"):
        assert plan[side + "_backend_version"] == "ipm-cpu-0.53.0"
        assert plan[side + "_objective_scaling"] == "off"
    assert plan["method"] == "homogeneous" and plan["stage_order"] == list(STAGES)
    assert plan["maximum_solver_runs"] == 500
    assert plan["candidate_limits"] == {"max_input_nonzeros": 8000000, "max_factor_nonzeros": 32000000,
        "max_dimension": 1000000, "max_selection_work": 1000000000,
        "max_repair_work": 400000000, "max_repair_pivots": 512, "max_repair_factor_nonzeros": 64000000,
        "max_eta_updates": 8}
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
    clean = {key: value for key, value in prior.items() if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
    clean.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    if enabled:
        clean["SOLVER_V80_BASIS_RECOVERY"] = "1"
    try:
        os.environ.clear()
        os.environ.update(clean)
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def inspect_marker(stderr, variant):
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if variant in ("frozen_v66", "highs"):
        return (None, None) if not lines else (None, "unexpected_marker")
    if len(lines) != 1:
        return None, "marker_count"
    try:
        value = benchmark.strict_json(lines[0][len(MARKER):])
    except (TypeError, ValueError):
        return None, "invalid_marker_json"
    if not valid_marker(value, variant == "diagnostic_on"):
        return value, "invalid_marker_schema"
    return value, None


def valid_guard(value, initial):
    fields = set(GUARD_COUNTS + GUARD_COORDINATES + GUARD_NUMBERS) | {"complete", "passed"}
    if not isinstance(value, dict) or value.keys() != fields \
        or any(type(value[key]) is not bool for key in ("complete", "passed")):
        return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= initial["n"] for key in GUARD_COUNTS) \
        or value["violations"] > value["checked_rows"]:
        return False
    for key, bound in (("failure_row", initial["m"]), ("failure_basis_position", initial["n"])):
        if value[key] is not None and (type(value[key]) is not int or not 0 <= value[key] < bound):
            return False
    if any(value[key] is not None and (not benchmark.finite_real(value[key]) or value[key] < 0)
        for key in GUARD_NUMBERS):
        return False
    if value["complete"] and value["checked_rows"] != initial["n"]:
        return False
    if value["passed"] is not (value["complete"] and value["violations"] == 0):
        return False
    if value["violations"]:
        if any(value[key] is None for key in GUARD_COORDINATES):
            return False
        # 窄化摘要可能相等、溢出或下溢为null，不能据展示值重判long double守卫。
        if any(value[key] is not None and value[key] <= 0
            for key in ("failure_abs_residual", "failure_threshold", "failure_ratio")):
            return False
    elif any(value[key] is not None for key in GUARD_COORDINATES + GUARD_NUMBERS[:3]):
        return False
    if value["checked_rows"] == 0 and any(value[key] is not None for key in GUARD_NUMBERS):
        return False
    return True


def valid_initial_refinement(value, initial_guard, refined_guard):
    fields = set(INITIAL_REFINEMENT_COUNTS) | {"status", "correction_max_abs"}
    if not isinstance(value, dict) or value.keys() != fields \
        or not isinstance(value["status"], str) or value["status"] not in INITIAL_REFINEMENT_STATUSES:
        return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= 9223372036854775807
        for key in INITIAL_REFINEMENT_COUNTS):
        return False
    if not (value["accepted"] <= value["solves"] <= value["attempts"] <= 1):
        return False
    if value["correction_max_abs"] is not None and (not benchmark.finite_real(value["correction_max_abs"])
        or value["correction_max_abs"] < 0):
        return False
    status = value["status"]
    if status in ("not_started", "not_needed"):
        if any(value[key] != 0 for key in INITIAL_REFINEMENT_COUNTS) or value["correction_max_abs"] is not None:
            return False
        if status == "not_needed" and not initial_guard["passed"]:
            return False
    else:
        if value["attempts"] != 1 or not initial_guard["complete"] or initial_guard["passed"] \
            or value["double_digits"] <= 0 or value["long_double_digits"] <= 0:
            return False
        if status == "unsupported_precision":
            if value["long_double_digits"] > value["double_digits"]:
                return False
        elif status not in ("budget_exception", "exception") \
            and value["long_double_digits"] <= value["double_digits"]:
            return False
    if status in ("preparing_rhs", "unsupported_precision", "rhs_representation_failure") and value["solves"] != 0:
        return False
    if status in ("solve_failure", "updating", "correction_nonfinite", "evaluating", "guard_rejected", "accepted") \
        and value["solves"] != 1:
        return False
    if (value["accepted"] == 1) is not (status == "accepted"):
        return False
    if refined_guard["checked_rows"] and value["solves"] != 1:
        return False
    if status == "guard_rejected" and (not refined_guard["complete"] or refined_guard["passed"]):
        return False
    if status == "accepted" and not refined_guard["passed"]:
        return False
    return True


def valid_forward(value, initial, repair, kind):
    fields = set(FORWARD_COUNTS + FORWARD_COORDINATES) | {"tag", "before", "after", "refinement"}
    if not isinstance(value, dict) or value.keys() != fields:
        return False
    # 这是依据每个outer基至多枚举n个方向得出的账本界，不新增实际搜索额度。
    systems_bound = initial["n"] * repair["phase1_iterations"] + repair["phase2_iterations"] if kind == "direction" else 512
    for key in FORWARD_COUNTS:
        bound = 512 if key == "pivot_sequence" else systems_bound
        if type(value[key]) is not int or not 0 <= value[key] <= bound:
            return False
    if not (value["accepted"] <= value["solves"] <= value["attempts"]
        <= value["guard_failures"] <= value["systems"]):
        return False
    if any(not valid_guard(value[key], initial) for key in ("before", "after")) \
        or not valid_initial_refinement(value["refinement"], value["before"], value["after"]):
        return False
    if any(value["refinement"][key] > value[key] for key in ("attempts", "solves", "accepted")):
        return False
    if value["systems"] == 0:
        return value["tag"] == "not_started" and all(value[key] == 0 for key in FORWARD_COUNTS) \
            and all(value[key] is None for key in FORWARD_COORDINATES) \
            and all(not value[key]["complete"] and value[key]["checked_rows"] == 0 for key in ("before", "after")) \
            and value["refinement"]["status"] == "not_started"
    if value["tag"] != kind:
        return False
    for key, bound in (("entering_row", initial["m"]), ("entering_basis_position", initial["n"])):
        if type(value[key]) is not int or not 0 <= value[key] < bound:
            return False
    if kind == "direction":
        if value["systems"] > repair["direction_solves"] \
            or value["pivot_sequence"] >= 512 \
            or not 0 <= repair["pivots"] - value["pivot_sequence"] <= 1:
            return False
    elif value["systems"] > repair["primal_solves"] \
        or value["pivot_sequence"] == 0 or not 0 <= value["pivot_sequence"] - repair["pivots"] <= 1:
        return False
    return True


def valid_transition(value, initial, repair):
    fields = set(TRANSITION_COUNTS + TRANSITION_COORDINATES + TRANSITION_NUMBERS + TRANSITION_BOOLS) | {"status"}
    if not isinstance(value, dict) or value.keys() != fields \
        or not isinstance(value["status"], str) or value["status"] not in TRANSITION_STATUSES:
        return False
    for key in TRANSITION_COUNTS:
        bound = initial["m"] if key == "checked_rows" else 512
        if type(value[key]) is not int or not 0 <= value[key] <= bound:
            return False
    if value["trials"] > repair["phase1_iterations"] or not \
        (value["near_accepted"] <= value["near_attempts"] <= value["trials"]
        and value["strict_accepted"] <= value["trials"] - value["near_attempts"]
        and value["strict_accepted"] + value["near_accepted"] <= repair["pivots"]):
        return False
    if any(type(value[key]) is not bool for key in TRANSITION_BOOLS):
        return False
    if any(value[key] is not None and (type(value[key]) is not int or not 0 <= value[key] < initial["m"])
        for key in TRANSITION_COORDINATES):
        return False
    signed = {"delta_phi", "blocking_slack"}
    if any(value[key] is not None and (not benchmark.finite_real(value[key]) or (key not in signed and value[key] < 0))
        for key in TRANSITION_NUMBERS):
        return False
    if value["status"] == "not_started":
        return all(value[key] == 0 for key in TRANSITION_COUNTS) \
            and all(value[key] is None for key in TRANSITION_COORDINATES + TRANSITION_NUMBERS) \
            and all(not value[key] for key in TRANSITION_BOOLS)
    if value["trials"] == 0 or value["blocking_row"] is None or value["released_row"] is None:
        return False
    if value["checked_rows"] == 0:
        if value["max_slack_change_row"] is not None or value["max_slack_change_ratio"] is not None:
            return False
    elif value["max_slack_change_row"] is None:
        return False
    if value["status"] in ("strict_candidate", "strict_accepted") and value["checked_rows"] != 0:
        return False
    if value["status"] == "strict_accepted" and value["strict_accepted"] == 0:
        return False
    if value["status"] in ("near_candidate", "near_accepted") and not \
        (value["checked_rows"] == initial["m"] and all(value[key] for key in TRANSITION_BOOLS)):
        return False
    if value["status"] == "near_accepted" and value["near_accepted"] == 0:
        return False
    if value["status"] in ("blocking_not_near", "released_not_near", "row_change_not_near") \
        and value["checked_rows"] != initial["m"]:
        return False
    if value["status"] == "blocking_not_near" and value["blocking_near"]:
        return False
    if value["status"] == "released_not_near" and (not value["blocking_near"] or value["released_near"]):
        return False
    if value["status"] == "row_change_not_near" and not \
        (value["blocking_near"] and value["released_near"] and not value["all_rows_near"]):
        return False
    return True


def valid_history(value, initial, repair):
    fields = set(HISTORY_COUNTS) | {"status", "initialized", "repeated_state"}
    if not isinstance(value, dict) or value.keys() != fields \
        or not isinstance(value["status"], str) or value["status"] not in HISTORY_STATUSES \
        or type(value["initialized"]) is not bool:
        return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= 9223372036854775807 for key in HISTORY_COUNTS):
        return False
    if not (value["states"] <= 513 and value["swaps"] <= 512 and value["candidate_checks"] <= repair["iterations"]
        and 0 <= value["candidate_checks"] - repair["pivots"] <= 1
        and value["exact_checks"] == value["hash_matches"] <= value["hash_comparisons"]
        and value["hash_comparisons"] >= repair["pivots"]
        and value["hash_comparisons"] <= value["candidate_checks"] * value["states"]
        and value["repeats"] <= min(value["exact_checks"], 1)
        and value["membership_comparisons"] <= value["exact_checks"] * initial["m"]
        and value["replay_swaps"] <= value["exact_checks"] * value["swaps"]):
        return False
    if value["initialized"]:
        if value["states"] != repair["pivots"] + 1 or value["swaps"] != repair["pivots"] \
            or value["status"] in ("not_started", "initializing"):
            return False
    elif any(value[key] != 0 for key in HISTORY_COUNTS) or value["status"] not in ("not_started", "initializing"):
        return False
    repeated = value["repeated_state"]
    if value["repeats"]:
        return value["status"] == "repeated" and type(repeated) is int and 0 <= repeated < value["states"]
    return value["status"] != "repeated" and repeated is None


def valid_search(value, initial, repair):
    fields = set(SEARCH_COUNTS + SEARCH_COORDINATES + SEARCH_NUMBERS + SEARCH_BOOLS) | {"status"}
    if not isinstance(value, dict) or value.keys() != fields \
        or not isinstance(value["status"], str) or value["status"] not in SEARCH_STATUSES:
        return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= 9223372036854775807 for key in SEARCH_COUNTS) \
        or any(type(value[key]) is not bool for key in SEARCH_BOOLS):
        return False
    if not (0 <= value["bases"] - repair["phase1_iterations"] <= 1
        and 0 <= value["bases"] - repair["pricing_matvecs"] <= 1
        and value["candidates"] <= initial["n"] * value["bases"]
        and value["trials"] <= min(value["candidates"], initial["n"] * repair["phase1_iterations"])
        and 0 <= value["trials"] - value["non_descent"] - value["descent"] <= 1
        and value["descent"] <= repair["phase1_iterations"]
        and value["trials"] <= repair["direction_solves"] + 1
        and repair["direction_solves"] <= value["trials"] + repair["phase2_iterations"]
        and value["exhausted"] <= 1 and value["basis_pivots"] <= repair["pivots"]
        and value["candidate_count"] <= initial["n"]
        and value["zero_ineq_rows"] + value["zero_eq_rows"] <= initial["m"]):
        return False
    for key, bound in (("ordinal", value["candidate_count"]), ("entering_row", initial["m"]),
        ("entering_basis_position", initial["n"])):
        if value[key] is not None and (type(value[key]) is not int or not 0 <= value[key] < bound):
            return False
    if any(value[key] is not None and (not benchmark.finite_real(value[key])
        or (key in ("zero_ineq", "zero_eq") and value[key] < 0)) for key in SEARCH_NUMBERS):
        return False
    if value["status"] == "not_started":
        return all(value[key] == 0 for key in SEARCH_COUNTS) \
            and all(value[key] is None for key in SEARCH_COORDINATES + SEARCH_NUMBERS) \
            and all(not value[key] for key in SEARCH_BOOLS)
    if value["bases"] == 0 or (value["step_started"] and not value["derivative_complete"]):
        return False
    if (value["exhausted"] == 1) is not (value["status"] == "exhausted"):
        return False
    if value["status"] in ("non_descent", "descent", "step_started", "pivot_committed"):
        if not value["derivative_complete"] or any(value[key] is None for key in SEARCH_COORDINATES):
            return False
        if value["status"] == "non_descent" and value["non_descent"] == 0:
            return False
        if value["status"] != "non_descent" and value["descent"] == 0:
            return False
        if value["step_started"] is not (value["status"] in ("step_started", "pivot_committed")):
            return False
    return True


def valid_eta(value, initial, repair):
    fields = set(ETA_COUNTS) | {"status", "last_mode", "last_fallback", "last_alpha"}
    if not isinstance(value, dict) or value.keys() != fields:
        return False
    for key, choices in (("status", ETA_STATUSES), ("last_mode", ETA_MODES), ("last_fallback", ETA_FALLBACKS)):
        if not isinstance(value[key], str) or value[key] not in choices:
            return False
    if any(type(value[key]) is not int or not 0 <= value[key] <= 9223372036854775807 for key in ETA_COUNTS) \
        or (value["last_alpha"] is not None and not benchmark.finite_real(value["last_alpha"])):
        return False
    modes = value["eta_trials"] + value["full_refactor_trials"]
    if not (0 <= value["prepares"] - modes <= 1 and value["prepares"] <= repair["iterations"]
        and value["eta_commits"] <= value["eta_trials"] and value["full_refactor_commits"] <= value["full_refactor_trials"]
        and value["eta_commits"] + value["full_refactor_commits"] == repair["pivots"]
        and value["full_refactor_commits"] <= repair["lu_factor_calls"] <= value["full_refactor_trials"]
        and repair["lu_analyze_calls"] <= value["full_refactor_trials"]
        and repair["primal_solves"] <= modes
        and value["chain_length"] <= value["max_chain_length"] <= 8
        and value["chain_length"] <= value["eta_commits"]
        and value["stored_entries"] == 2 * initial["n"] * value["chain_length"]
        and value["stored_entries"] <= value["peak_stored_entries"] <= 16 * initial["n"]
        and value["peak_stored_entries"] == 2 * initial["n"] * value["max_chain_length"]
        and value["fallbacks"] <= value["full_refactor_trials"]
        and value["base_forward_solves"] <= value["forward_calls"]
        and value["base_transpose_solves"] <= value["transpose_calls"]
        and value["forward_calls"] == repair["initial_refinement"]["solves"] + repair["direction_solves"]
            + repair["primal_solves"] + repair["direction_forward"]["solves"] + repair["rebuild_forward"]["solves"]
        and value["transpose_calls"] == repair["dual_solves"]
        and value["forward_applications"] <= 8 * value["forward_calls"]
        and value["transpose_applications"] <= 8 * value["transpose_calls"]
        and value["work_units"] <= repair["work_units"]):
        return False
    if value["last_mode"] == "eta" and (value["eta_trials"] == 0 or value["last_alpha"] is None or value["last_alpha"] == 0):
        return False
    if value["last_mode"] == "full" and value["full_refactor_trials"] == 0:
        return False
    if value["status"] == "not_started":
        return all(value[key] == 0 for key in ETA_COUNTS) and value["last_mode"] == value["last_fallback"] == "none" \
            and value["last_alpha"] is None
    return True


def valid_repair(value, initial):
    if not isinstance(value, dict) or value.keys() != {"status", "candidate_available", "stats"} \
        or type(value["candidate_available"]) is not bool or not isinstance(value["status"], str) \
        or value["status"] not in REPAIR_STATUSES:
        return False
    stats = value["stats"]
    if not isinstance(stats, dict) or stats.keys() != set(REPAIR_COUNTS + REPAIR_NUMBERS) \
        | {"phase", "initial_guard", "refined_guard", "initial_refinement", "direction_forward", "rebuild_forward",
            "phase1_transition", "history", "search", "eta"}:
        return False
    if any(not valid_guard(stats[key], initial) for key in ("initial_guard", "refined_guard")) \
        or not valid_initial_refinement(stats["initial_refinement"], stats["initial_guard"], stats["refined_guard"]):
        return False
    if any(type(stats[key]) is not int or not 0 <= stats[key] <= 9223372036854775807 for key in REPAIR_COUNTS):
        return False
    if not valid_transition(stats["phase1_transition"], initial, stats) or not valid_history(stats["history"], initial, stats):
        return False
    if not valid_search(stats["search"], initial, stats):
        return False
    if not valid_forward(stats["direction_forward"], initial, stats, "direction") \
        or not valid_forward(stats["rebuild_forward"], initial, stats, "rebuild") \
        or any(stats[key]["systems"] < stats["pivots"] for key in ("direction_forward", "rebuild_forward")):
        return False
    if not valid_eta(stats["eta"], initial, stats):
        return False
    if any(stats[key] is not None and (not benchmark.finite_real(stats[key])
        or (key != "last_right_derivative" and stats[key] < 0)) for key in REPAIR_NUMBERS):
        return False
    if not isinstance(stats["phase"], str) or stats["phase"] not in REPAIR_PHASES:
        return False
    if stats["attempts"] > 1 or stats["pivots"] > 512 or stats["work_units"] > 400000000 \
        or stats["final_selected_rows_count"] > initial["n"] or stats["last_factor_nonzeros"] > stats["factor_nonzeros"]:
        return False
    if stats["iterations"] != stats["phase1_iterations"] + stats["phase2_iterations"] \
        or not (0 <= stats["iterations"] - stats["pivots"] <= 1 and stats["iterations"] <= 512) \
        or not (stats["primal_solves"] <= stats["iterations"]
            and stats["lu_factor_calls"] <= stats["lu_analyze_calls"] <= stats["iterations"]
            and stats["lu_analyze_calls"] <= stats["direction_solves"]
            <= initial["n"] * stats["phase1_iterations"] + stats["phase2_iterations"]) \
        or stats["direction_matvecs"] > stats["direction_solves"] + stats["direction_forward"]["solves"] \
        or stats["zero_steps"] > 1:
        return False
    if value["status"] == "not_started":
        return not value["candidate_available"] and stats["phase"] == "not_started" \
            and all(stats[key] == 0 for key in REPAIR_COUNTS) and all(stats[key] is None for key in REPAIR_NUMBERS) \
            and all(not stats[key]["complete"] and stats[key]["checked_rows"] == 0 for key in ("initial_guard", "refined_guard")) \
            and stats["initial_refinement"]["status"] == "not_started" \
            and all(stats[key]["systems"] == 0 for key in ("direction_forward", "rebuild_forward")) \
            and stats["phase1_transition"]["status"] == stats["history"]["status"] == stats["search"]["status"] \
            == stats["eta"]["status"] == "not_started"
    if stats["attempts"] != 1 or not (initial["primal_solves"] == initial["dual_solves"] == 1):
        return False
    if value["status"] in ("no_entering", "direction_search_exhausted"):
        if stats["search"]["status"] != "exhausted" or stats["search"]["exhausted"] != 1 \
            or ((stats["search"]["candidate_count"] == 0) is not (value["status"] == "no_entering")):
            return False
    if stats["phase"] in ("phase1", "phase2", "complete") and not \
        ((stats["initial_guard"]["passed"] or stats["initial_refinement"]["accepted"] == 1)
            and stats["history"]["initialized"]):
        return False
    if value["candidate_available"]:
        return value["status"] == "candidate" and stats["phase"] == "complete" \
            and stats["history"]["initialized"] \
            and stats["initial_guard"]["complete"] \
            and (stats["initial_guard"]["passed"] or stats["initial_refinement"]["accepted"] == 1) \
            and all(stats[key]["systems"] == 0 or stats[key]["before"]["passed"]
                or stats[key]["refinement"]["accepted"] == 1 for key in ("direction_forward", "rebuild_forward")) \
            and (stats["direction_forward"]["systems"] == 0
                or stats["direction_forward"]["pivot_sequence"] + 1 == stats["pivots"]) \
            and (stats["rebuild_forward"]["systems"] == 0
                or stats["rebuild_forward"]["pivot_sequence"] == stats["pivots"]) \
            and stats["final_selected_rows_count"] == initial["n"] \
            and stats["iterations"] == stats["pivots"] \
            and initial["lu_l_nonzeros"] + initial["lu_u_nonzeros"] <= stats["factor_nonzeros"] <= 64000000 \
            and 0 < stats["last_factor_nonzeros"] <= 32000000
    return value["status"] != "candidate"


def valid_marker(value, enabled):
    fields = {"schema", "enabled", "attempted", "accepted", "stage", "candidate_available", "attempt_seconds",
        "verification_seconds", "recovery_status", "stats", "additional_work_in_legacy_ipm_counts", "work_in_total_seconds",
        "marker_in_total_seconds", "trial_verification", "repair"}
    if not isinstance(value, dict) or value.keys() != fields or type(value["schema"]) is not int or value["schema"] != 1:
        return False
    if any(type(value[key]) is not bool for key in ("enabled", "attempted", "accepted", "candidate_available")) \
        or value["enabled"] is not enabled or value["additional_work_in_legacy_ipm_counts"] is not False \
        or value["work_in_total_seconds"] is not True or value["marker_in_total_seconds"] is not False:
        return False
    if not all(benchmark.finite_real(value[key]) and value[key] >= 0 for key in ("attempt_seconds", "verification_seconds")):
        return False
    if not isinstance(value["recovery_status"], str) or value["recovery_status"] not in RECOVERY_STATUSES:
        return False
    if not isinstance(value["stage"], str):
        return False
    stats = value["stats"]
    if not isinstance(stats, dict) or stats.keys() != set(MARKER_COUNTS) | {"qr_default_threshold", "min_abs_pivot", "selection_threshold"}:
        return False
    if any(type(stats[key]) is not int or not 0 <= stats[key] <= 9223372036854775807 for key in MARKER_COUNTS):
        return False
    threshold = stats["qr_default_threshold"]
    if not benchmark.finite_real(threshold) or threshold != 0 or any(stats[key] != 0 for key in UNUSED_QR_COUNTS):
        return False
    if any(not benchmark.finite_real(stats[key]) or stats[key] < 0
        for key in ("min_abs_pivot", "selection_threshold")):
        return False
    if stats["rank"] > min(stats["n"], stats["m"]) or stats["selected_rows_count"] > stats["n"] \
        or stats["ordered_rows_count"] > stats["m"] or stats["qr_columns_permutation_count"] > stats["m"]:
        return False
    if not (stats["rank"] == stats["selection_pivots"] == stats["selected_rows_count"] == stats["pivot_columns_count"]
        and stats["selection_candidates"] <= stats["m"]
        and 0 <= stats["selection_candidates"] - stats["selection_pivots"] - stats["selection_rejected"] <= 1
        and stats["work_units"] <= 1000000000):
        return False
    if (stats["selection_pivots"] == 0) is not (stats["min_abs_pivot"] == 0):
        return False
    if not (stats["lu_factor_calls"] <= stats["lu_analyze_calls"] <= 1
        and stats["slack_matvecs"] <= stats["dual_solves"] <= stats["primal_solves"] <= stats["lu_factor_calls"]):
        return False
    if not valid_repair(value["repair"], stats):
        return False
    trial = value["trial_verification"]
    bools = ("primal_valid", "primal_feasible", "dual_feasible", "gap_satisfied", "numerically_optimal")
    numbers = ("max_row_violation", "max_bound_violation", "max_dual_residual", "objective_gap")
    if trial is not None and (not isinstance(trial, dict) or trial.keys() != set(bools + numbers)
        or any(type(trial[key]) is not bool for key in bools)
        or any(trial[key] is not None and not benchmark.finite_real(trial[key]) for key in numbers)):
        return False
    if not value["attempted"]:
        return value["stage"] == "not_attempted" and value["recovery_status"] == "not_started" \
            and not value["accepted"] and not value["candidate_available"] and trial is None \
            and value["attempt_seconds"] == value["verification_seconds"] == 0 \
            and all(stats[key] == 0 for key in MARKER_COUNTS) and threshold == 0 \
            and stats["min_abs_pivot"] == stats["selection_threshold"] == 0 and value["repair"]["status"] == "not_started"
    if not enabled or value["stage"] not in {"accepted", "original_verification_rejected", "candidate_unavailable", "time_limit", "recovery_exception"}:
        return False
    if value["accepted"] is not (value["stage"] == "accepted"):
        return False
    if value["candidate_available"]:
        if value["recovery_status"] != "candidate":
            return False
        if (stats["n"] > 0 and not value["repair"]["candidate_available"]) \
            or (stats["n"] == 0 and value["repair"]["status"] != "not_started"):
            return False
        if stats["n"] and not (stats["rank"] == stats["n"] == stats["selected_rows_count"]
            and stats["ordered_rows_count"] == stats["m"]
            and stats["selection_candidates"] == stats["selection_pivots"] + stats["selection_rejected"]
            and benchmark.finite_real(stats["min_abs_pivot"]) and stats["min_abs_pivot"] > 0
            and benchmark.finite_real(stats["selection_threshold"]) and stats["selection_threshold"] > 0
            and all(stats[key] == 1 for key in ("lu_analyze_calls", "lu_factor_calls",
                "primal_solves", "dual_solves", "slack_matvecs"))):
            return False
    elif value["recovery_status"] == "candidate" or trial is not None:
        return False
    if value["stage"] in ("accepted", "original_verification_rejected") and (not value["candidate_available"] or trial is None):
        return False
    if value["stage"] == "candidate_unavailable" and value["candidate_available"]:
        return False
    if value["accepted"] and (any(trial[key] is not True for key in bools) or any(trial[key] is None for key in numbers)):
        return False
    return True


def valid_process(row):
    if row.get("basis_marker_error") is not None or row.get("configuration_matches") is not True:
        return False
    if not base.valid_process(row):
        return False
    report = row["report"]
    return report["backend"] != "ipm" or benchmark.valid_homogeneous_work(report) and benchmark.valid_homogeneous_scaled_work(report)


def configuration_matches(report, stage, variant, model, plan):
    side = "reference" if variant == variants(stage)[0] else "candidate"
    backend = "highs" if variant == "highs" else "ipm"
    return isinstance(report, dict) and report.get("input") == model["input"] and report.get("backend") == backend \
        and report.get("backend_version") == ("1.15.1" if backend == "highs" else "ipm-cpu-0.53.0") \
        and base.typed_equal(report.get("options"), base.expected_options(stage, side, plan))


def basis_sidecar_compatible(a, b):
    left, right = a.get("basis_marker"), b.get("basis_marker")
    if left is None and right is None:
        return True, None
    if left is None or right is None:
        # v66没有侧账本；只可与关闭候选的规范零事件账本对应。
        return valid_marker(right if left is None else left, False), None
    if not isinstance(left, dict) or not isinstance(right, dict) \
        or not valid_marker(left, left.get("enabled")) or not valid_marker(right, right.get("enabled")):
        return False, False
    excluded = {"attempt_seconds", "verification_seconds"}
    equal = base.typed_equal({key: item for key, item in left.items() if key not in excluded},
        {key: item for key, item in right.items() if key not in excluded})
    return equal, equal


def pair_result(a, b, plan):
    answer = base.pair_result(a, b, plan)
    compatible, equal = basis_sidecar_compatible(a, b)
    answer.update(report_and_exit_equal=answer["equal"], equal=answer["equal"] and compatible,
        basis_sidecar_compatible=compatible, basis_sidecar_equal_excluding_durations=equal)
    if not valid_process(a) or not valid_process(b):
        answer.update(valid_processes=False, category="invalid_process")
    elif answer["category"] == "equal" and not compatible:
        answer["category"] = "different"
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


def freeze(work, plan):
    build_path = work / "build-manifest.json"
    build = read(build_path)
    assert build["plan_sha256"] == PLAN_SHA and build["parent_identity_sha256"] == plan["parent_identity_sha256"]
    assert build["baseline_identity_sha256"] == plan["baseline_identity_sha256"]
    assert Path(build["frozen_parent_executable"]) == PARENT
    assert build["frozen_parent_sha256"] == plan["parent_solver_sha256"]
    candidate = Path(build["executable"])
    assert candidate == work / "solver-diagnostic.exe"
    protected = {str(Path(__file__)): sha(__file__), str(DATA / "v66-run.py"): HELPER_SHA,
        str(build_path): sha(build_path), str(PARENT): plan["parent_solver_sha256"],
        str(candidate): build["executable_sha256"],
        str(ROOT / "scripts/benchmark.py"): sha(ROOT / "scripts/benchmark.py"),
        str(ROOT / "scripts/compare_lp.py"): sha(ROOT / "scripts/compare_lp.py")}
    for name in ("v80-basis-recovery.hpp", "v80-basis-repair.hpp", "v80-basis-instrumentation.py",
        "v80-basis-self-test.cpp", "v80-repair-self-test.cpp", "v80-build.py", "v80-run.py"):
        path = str(DATA / name)
        digest = build["protected_before_and_after"][path]
        assert path not in protected or protected[path] == digest
        protected[path] = digest
    for name in ("static-check.json", "basis-self-check.json", "repair-self-check.json"):
        path = str(work / name)
        protected[path] = build["outputs"][path]
        assert read(path)["passed"] is True
    # 构建已冻结库；本入口只核执行来源及工具，完整库身份由独立终验统一核对。
    for path, digest in protected.items():
        assert sha(path) == digest, path
    value = {"plan_sha256": PLAN_SHA, "build_manifest": str(build_path), "build_manifest_sha256": sha(build_path),
        "candidate": str(candidate), "candidate_sha256": build["executable_sha256"],
        "reference": str(PARENT), "reference_sha256": plan["parent_solver_sha256"], "protected": protected}
    save_once(DATA / "v80-run-identity.json", value)
    return value


def archive(stage):
    plan = fixed_plan()
    expected = specification(stage, plan)
    manifest = read(DATA / f"v80-{stage}-manifest.json")
    summary = read(DATA / f"v80-{stage}-summary.json")
    raw = DATA / f"v80-{stage}-runs.jsonl.gz"
    rows = [benchmark.strict_json(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    identity = read(DATA / "v80-run-identity.json")
    assert manifest["identity_sha256"] == sha(DATA / "v80-run-identity.json")
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
        assert row["basis_environment"] == ({"SOLVER_V80_BASIS_RECOVERY": "1"} if variant == "diagnostic_on" else {})
        assert all(row[field] == digest for field in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256"))
        assert all(row[field] == model["sha256"] for field in ("model_sha256", "model_sha256_after", "frozen_model_sha256"))
        assert row["hash_error"] is None
        marker, error = inspect_marker(row["stderr"], variant)
        assert row["basis_marker"] == marker and row["basis_marker_error"] == error
        assert row["configuration_matches"] is configuration_matches(row["report"], stage, variant, model, plan)
        if row["report"] is not None:
            assert row["report"] == benchmark.strict_json(row["stdout"])
        for field, value in quality_fields(row, model, digest, plan).items():
            assert row[field] == value, (index, field)
    computed = compare_lp.summarize(rows, expected["models"], expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    computed.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), basis_marker_error_count=sum(row["basis_marker_error"] is not None for row in rows))
    assert computed == summary
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
        save_once(DATA / "v80-compatibility.json", result)
        assert result["passed"], "17个非预算完整报告对子未全部兼容"
    if stage in ("netlib", "highs"):
        result = gate()
        save_once(DATA / "v80-gate.json", result)
        assert result["eligible_for_netlib"], "未通过预先固定扩集门槛"
        if stage == "highs":
            archive("netlib")
    expected = specification(stage, plan)
    models, digest = compare_lp.load_models(Path(expected["models_file"]))
    assert models == expected["models"] and digest == expected["models_file_sha256"]
    assert not list(DATA.glob(f"v80-{stage}-*")), "禁止覆盖既有阶段证据"
    manifest = {"schema": 1, "stage": stage, "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA, "identity_sha256": sha(DATA / "v80-run-identity.json"),
        "models": models, "models_file_sha256": digest, "repeats": expected["repeats"], "variants": list(variants(stage)),
        **{key: plan[key] for key in ("time_limit", "wall_timeout", "threads", "seed")},
        "environment": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "environment_rule": "每次清除所有SOLVER_V前缀及OPENBLAS_CORETYPE；仅diagnostic_on设置SOLVER_V80_BASIS_RECOVERY=1；结束恢复原环境。",
        "execution": "复用benchmark.run_one子进程35秒硬终止；串行轮转；额外恢复计入solve_seconds，5秒超限按失败计罚。"}
    save(DATA / f"v80-{stage}-manifest.json", manifest)
    raw = DATA / f"v80-{stage}-runs.jsonl.gz"
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
                basis_environment={"SOLVER_V80_BASIS_RECOVERY": "1"} if variant == "diagnostic_on" else {},
                basis_marker=marker, basis_marker_error=marker_error,
                configuration_matches=configuration_matches(row["report"], stage, variant, model, plan))
            row.update(quality_fields(row, model, expected_solver, plan))
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            print(json.dumps({"stage": stage, "index": row["run_index"], "model": Path(model["input"]).stem,
                "variant": variant, "status": row["status"], "quality_pass": row["quality_pass"],
                "basis_marker_error": marker_error}, ensure_ascii=False), flush=True)
    summary = compare_lp.summarize(rows, models, expected["repeats"], plan["wall_timeout"], plan["time_limit"], variants(stage))
    summary.update(plan_sha256=PLAN_SHA, stage=stage, raw_sha256=sha(raw),
        valid_process_count=sum(valid_process(row) for row in rows), basis_marker_error_count=sum(row["basis_marker_error"] is not None for row in rows))
    save(DATA / f"v80-{stage}-summary.json", summary)
    for path, digest in identity["protected"].items():
        assert sha(path) == digest, path
    archive(stage)
    print(json.dumps({"stage": stage, "variants": summary["variants"], "valid_process_count": summary["valid_process_count"]}, ensure_ascii=False), flush=True)


def self_check():
    plan = fixed_plan()
    inherited = base.validation_self_check()
    assert plan["compatibility_excluded_report_fields"] == ["timings"]
    before = dict(os.environ)
    with solver_environment(True):
        assert {key: value for key, value in os.environ.items() if key.startswith("SOLVER_V")} == {"SOLVER_V80_BASIS_RECOVERY": "1"}
        assert "OPENBLAS_CORETYPE" not in os.environ
        with solver_environment(False):
            assert not any(key.startswith("SOLVER_V") for key in os.environ)
        assert os.environ["SOLVER_V80_BASIS_RECOVERY"] == "1"
    assert dict(os.environ) == before
    for stage in STAGES:
        for variant in variants(stage):
            value = options_for(stage, variant, PARENT, plan)
            assert value.time_limit == 5.0 and value.wall_timeout == 35 and value.threads == 1
            if variant == "highs":
                assert value.backend == "highs" and value.highs_solver == "choose"
            else:
                assert value.ipm_linear_solver == "normal_cholmod_aat" and value.ipm_linear_accuracy == "homogeneous_repair"
                assert value.ipm_objective_scaling == "off" and value.ipm_direction_retries == 0
    return {"passed": True, "solver_runs": 0, "inherited_count_and_budget_checks": inherited,
        "environment_cases": 4, "configuration_cases": 12, "marker_cases": marker_self_check()}


def marker_self_check():
    idle_guard = {"complete": False, "passed": False, "checked_rows": 0, "violations": 0,
        **dict.fromkeys(GUARD_COORDINATES + GUARD_NUMBERS)}
    idle_refinement = {"status": "not_started", **dict.fromkeys(INITIAL_REFINEMENT_COUNTS, 0),
        "correction_max_abs": None}
    idle_forward = {**dict.fromkeys(FORWARD_COUNTS, 0), **dict.fromkeys(FORWARD_COORDINATES),
        "tag": "not_started", "before": copy.deepcopy(idle_guard), "after": copy.deepcopy(idle_guard),
        "refinement": copy.deepcopy(idle_refinement)}
    idle_transition = {"status": "not_started", **dict.fromkeys(TRANSITION_COUNTS, 0),
        **dict.fromkeys(TRANSITION_COORDINATES + TRANSITION_NUMBERS), **dict.fromkeys(TRANSITION_BOOLS, False)}
    idle_history = {"status": "not_started", "initialized": False, **dict.fromkeys(HISTORY_COUNTS, 0),
        "repeated_state": None}
    idle_search = {"status": "not_started", **dict.fromkeys(SEARCH_COUNTS, 0),
        **dict.fromkeys(SEARCH_COORDINATES + SEARCH_NUMBERS), **dict.fromkeys(SEARCH_BOOLS, False)}
    idle_eta = {"status": "not_started", "last_mode": "none", "last_fallback": "none",
        **dict.fromkeys(ETA_COUNTS, 0), "last_alpha": None}
    def fixture_marker(value, enabled):
        # 为既有字段测试补全与调用者计数一致的完整base调用快照。
        stats = value["repair"]["stats"]
        eta = stats["eta"]
        eta["forward_calls"] = stats["initial_refinement"]["solves"] + stats["direction_solves"] \
            + stats["primal_solves"] + stats["direction_forward"]["solves"] + stats["rebuild_forward"]["solves"]
        eta["transpose_calls"] = stats["dual_solves"]
        eta["base_forward_solves"] = eta["forward_calls"]
        eta["base_transpose_solves"] = eta["transpose_calls"]
        if eta["prepares"] == 0 and (eta["forward_calls"] or eta["transpose_calls"]):
            eta["status"] = "forward" if eta["forward_calls"] else "transpose"
        elif eta["prepares"] == 0 and stats["initial_guard"]["complete"]:
            eta["status"] = "ready"
        return valid_marker(value, enabled)
    idle_repair = {"status": "not_started", "candidate_available": False,
        "stats": {**dict.fromkeys(REPAIR_COUNTS, 0), **dict.fromkeys(REPAIR_NUMBERS), "phase": "not_started",
            "initial_guard": copy.deepcopy(idle_guard), "refined_guard": copy.deepcopy(idle_guard),
            "initial_refinement": copy.deepcopy(idle_refinement), "direction_forward": copy.deepcopy(idle_forward),
            "rebuild_forward": copy.deepcopy(idle_forward), "phase1_transition": copy.deepcopy(idle_transition),
            "history": copy.deepcopy(idle_history), "search": copy.deepcopy(idle_search), "eta": copy.deepcopy(idle_eta)}}
    empty = {"schema": 1, "enabled": False, "attempted": False, "accepted": False, "stage": "not_attempted",
        "candidate_available": False, "attempt_seconds": 0, "verification_seconds": 0, "recovery_status": "not_started",
        "stats": {**dict.fromkeys(MARKER_COUNTS, 0), "qr_default_threshold": 0, "min_abs_pivot": 0, "selection_threshold": 0}, "trial_verification": None,
        "repair": copy.deepcopy(idle_repair),
        "additional_work_in_legacy_ipm_counts": False, "work_in_total_seconds": True, "marker_in_total_seconds": False}
    assert valid_marker(empty, False)
    enabled = dict(empty, enabled=True)
    assert fixture_marker(enabled, True)
    assert inspect_marker("", "frozen_v66") == (None, None)
    assert inspect_marker(MARKER + json.dumps(empty), "diagnostic_off") == (empty, None)
    assert inspect_marker(MARKER + json.dumps(enabled), "diagnostic_on") == (enabled, None)
    assert inspect_marker(MARKER + json.dumps(empty), "frozen_v66")[1] == "unexpected_marker"
    assert inspect_marker("", "diagnostic_on")[1] == "marker_count"
    assert inspect_marker(MARKER + "{}\n" + MARKER + "{}", "diagnostic_on")[1] == "marker_count"
    assert inspect_marker(MARKER + "{", "diagnostic_on")[1] == "invalid_marker_json"
    candidate = copy.deepcopy(enabled)
    candidate.update(attempted=True, accepted=True, stage="accepted", candidate_available=True, recovery_status="candidate",
        attempt_seconds=0.2, verification_seconds=0.01, trial_verification={"primal_valid": True, "primal_feasible": True,
            "dual_feasible": True, "gap_satisfied": True, "numerically_optimal": True, "max_row_violation": 0,
            "max_bound_violation": 0, "max_dual_residual": 0, "objective_gap": 0})
    candidate["stats"].update(n=1, m=2, input_nonzeros=2, rank=1,
        lu_l_nonzeros=1, lu_u_nonzeros=1, budget_checks=9,
        lu_analyze_calls=1, lu_factor_calls=1, primal_solves=1, dual_solves=1, slack_matvecs=1,
        ordered_rows_count=2, selected_rows_count=1, selection_candidates=1, selection_pivots=1,
        pivot_columns_count=1, work_units=5, l_nonzeros=1, min_abs_pivot=1, selection_threshold=1e-12)
    candidate["repair"].update(status="candidate", candidate_available=True)
    candidate["repair"]["stats"].update(phase="complete", attempts=1, slack_matvecs=1,
        budget_checks=4, factor_nonzeros=2, last_factor_nonzeros=2, final_selected_rows_count=1,
        initial_phi=0, final_phi=0, initial_max_violation=0, final_max_violation=0,
        max_roundoff_bound=1e-12, max_selected_residual=0)
    candidate["repair"]["stats"]["initial_guard"].update(complete=True, passed=True, checked_rows=1,
        max_ratio=0, max_abs_residual=0)
    candidate["repair"]["stats"]["initial_refinement"]["status"] = "not_needed"
    def committed_history(count):
        result = copy.deepcopy(idle_history)
        result.update(status="unique" if count else "ready", initialized=True, states=count + 1,
            swaps=count, candidate_checks=count, hash_comparisons=count * (count + 1) // 2)
        return result
    def strict_transition(count):
        result = copy.deepcopy(idle_transition)
        result.update(status="strict_accepted", trials=count, strict_accepted=count,
            blocking_row=1, released_row=0, old_phi=1, new_phi=0, delta_phi=-1,
            blocking_slack=0, blocking_tau=1e-12, released_step=1, released_tau=1e-12)
        return result
    def search_snapshot(status, bases, trials, descent, basis_pivots):
        result = copy.deepcopy(idle_search)
        result.update(status=status, bases=bases, candidates=bases, trials=trials, descent=descent,
            basis_pivots=basis_pivots)
        if status == "preparing":
            result["candidates"] = max(bases - 1, 0)
        elif trials:
            result.update(candidate_count=1, ordinal=0, entering_row=0, entering_basis_position=0, priced_rc=-1)
        if status in ("descent", "step_started", "pivot_committed"):
            result.update(derivative_complete=True, linear_slope=-1, zero_ineq=0, zero_eq=0,
                true_derivative=-1, linear_minus_price=0, step_started=status != "descent")
        return result
    def full_eta(commits, trials):
        result = copy.deepcopy(idle_eta)
        result.update(status="ready")
        if trials:
            result.update(status="commit_full" if commits == trials else "full_trial", last_mode="full",
                last_fallback="unrepresentable", prepares=trials, full_refactor_trials=trials,
                full_refactor_commits=commits, fallbacks=trials)
        return result
    candidate["repair"]["stats"]["history"] = committed_history(0)
    candidate["repair"]["stats"]["eta"] = full_eta(0, 0)
    def completed_forward(kind, sequence):
        result = copy.deepcopy(idle_forward)
        result.update(systems=1, tag=kind, pivot_sequence=sequence, entering_row=0, entering_basis_position=0,
            before=copy.deepcopy(candidate["repair"]["stats"]["initial_guard"]))
        result["refinement"]["status"] = "not_needed"
        return result
    assert fixture_marker(candidate, True)
    interrupted_repair = copy.deepcopy(candidate)
    interrupted_repair.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    interrupted_repair["stats"]["slack_matvecs"] = 0
    interrupted_repair["repair"] = copy.deepcopy(idle_repair)
    interrupted_repair["repair"].update(status="budget_exception")
    interrupted_repair["repair"]["stats"].update(phase="validation", attempts=1, budget_checks=1)
    assert fixture_marker(interrupted_repair, True)
    limited_repair = copy.deepcopy(interrupted_repair)
    limited_repair.update(stage="candidate_unavailable", recovery_status="repair_unavailable")
    limited_repair["repair"]["status"] = "work_limit"
    limited_repair["repair"]["stats"].update(phase="phase1", work_units=400000000,
        pricing_matvecs=1, slack_matvecs=1, factor_nonzeros=2, last_factor_nonzeros=2, final_selected_rows_count=1,
        initial_phi=1, final_phi=1, initial_max_violation=1, final_max_violation=1)
    limited_repair["repair"]["stats"]["initial_guard"] = copy.deepcopy(candidate["repair"]["stats"]["initial_guard"])
    limited_repair["repair"]["stats"]["initial_refinement"]["status"] = "not_needed"
    limited_repair["repair"]["stats"]["history"] = committed_history(0)
    limited_repair["repair"]["stats"]["search"] = search_snapshot("preparing", 1, 0, 0, 0)
    assert fixture_marker(limited_repair, True)
    factor_repair = copy.deepcopy(limited_repair)
    factor_repair["repair"]["status"] = "factor_limit"
    factor_repair["repair"]["stats"].update(work_units=100, iterations=2, phase1_iterations=2, pivots=1,
        dual_solves=2, pricing_matvecs=2, direction_solves=2, direction_matvecs=2, lu_analyze_calls=2, lu_factor_calls=2,
        primal_solves=1, slack_matvecs=2, factor_nonzeros=64000002, last_factor_nonzeros=32000000,
        last_right_derivative=-1, last_step=1)
    factor_repair["repair"]["stats"]["direction_forward"] = completed_forward("direction", 1)
    factor_repair["repair"]["stats"]["direction_forward"]["systems"] = 2
    factor_repair["repair"]["stats"]["rebuild_forward"] = completed_forward("rebuild", 1)
    factor_repair["repair"]["stats"]["history"] = committed_history(1)
    factor_repair["repair"]["stats"]["phase1_transition"] = strict_transition(1)
    factor_repair["repair"]["stats"]["search"] = search_snapshot("step_started", 2, 2, 2, 1)
    factor_repair["repair"]["stats"]["eta"] = full_eta(1, 2)
    assert fixture_marker(factor_repair, True)
    committed_repair = copy.deepcopy(factor_repair)
    committed_repair.update(stage="time_limit", recovery_status="budget_exception")
    committed_repair["repair"]["status"] = "budget_exception"
    committed_repair["repair"]["stats"].update(pivots=1, primal_solves=1,
        slack_matvecs=2, factor_nonzeros=4, last_factor_nonzeros=2, final_phi=0, final_max_violation=0)
    committed_repair["repair"]["stats"]["history"] = committed_history(1)
    committed_repair["repair"]["stats"]["phase1_transition"] = strict_transition(1)
    assert fixture_marker(committed_repair, True)
    single_factor_repair = copy.deepcopy(limited_repair)
    single_factor_repair["repair"]["status"] = "single_factor_limit"
    single_factor_repair["repair"]["stats"].update(work_units=100, iterations=1, phase1_iterations=1,
        dual_solves=1, direction_solves=1, direction_matvecs=1, lu_analyze_calls=1, lu_factor_calls=1,
        factor_nonzeros=32000003, last_factor_nonzeros=32000001, last_right_derivative=-1, last_step=1)
    single_factor_repair["repair"]["stats"]["direction_forward"] = completed_forward("direction", 0)
    single_factor_repair["repair"]["stats"]["search"] = search_snapshot("step_started", 1, 1, 1, 0)
    single_factor_repair["repair"]["stats"]["eta"] = full_eta(0, 1)
    assert fixture_marker(single_factor_repair, True)
    unread_factor_budget = copy.deepcopy(single_factor_repair)
    unread_factor_budget.update(stage="time_limit", recovery_status="budget_exception")
    unread_factor_budget["repair"]["status"] = "budget_exception"
    unread_factor_budget["repair"]["stats"].update(factor_nonzeros=2, last_factor_nonzeros=2)
    assert fixture_marker(unread_factor_budget, True)
    exception_repair = copy.deepcopy(interrupted_repair)
    exception_repair.update(stage="recovery_exception", recovery_status="invalid_input")
    exception_repair["repair"]["status"] = "exception"
    assert fixture_marker(exception_repair, True)
    refined_candidate = copy.deepcopy(candidate)
    refined_stats = refined_candidate["repair"]["stats"]
    refined_stats["initial_guard"].update(passed=False, violations=1,
        failure_row=0, failure_basis_position=0, failure_abs_residual=2e-12,
        failure_threshold=1e-12, failure_ratio=2, max_ratio=2, max_abs_residual=2e-12)
    refined_stats["refined_guard"] = copy.deepcopy(candidate["repair"]["stats"]["initial_guard"])
    refined_stats["initial_refinement"].update(status="accepted", attempts=1, solves=1, accepted=1,
        double_digits=53, long_double_digits=64, correction_max_abs=1e-12)
    assert fixture_marker(refined_candidate, True)
    guard_rejected = copy.deepcopy(refined_candidate)
    guard_rejected.update(accepted=False, candidate_available=False, stage="candidate_unavailable",
        recovery_status="repair_unavailable", trial_verification=None)
    guard_rejected["stats"]["slack_matvecs"] = 0
    guard_rejected["repair"].update(status="basis_residual_failure", candidate_available=False)
    guard_rejected["repair"]["stats"].update(phase="initial_refinement", final_selected_rows_count=0)
    guard_rejected["repair"]["stats"]["history"] = copy.deepcopy(idle_history)
    guard_rejected["repair"]["stats"]["refined_guard"] = copy.deepcopy(refined_stats["initial_guard"])
    guard_rejected["repair"]["stats"]["initial_refinement"].update(status="guard_rejected", accepted=0)
    assert fixture_marker(guard_rejected, True)
    partial_guard = copy.deepcopy(guard_rejected)
    partial_guard.update(stage="time_limit", recovery_status="budget_exception")
    partial_guard["repair"]["status"] = "budget_exception"
    partial_guard["repair"]["stats"]["refined_guard"].update(complete=False)
    partial_guard["repair"]["stats"]["initial_refinement"]["status"] = "budget_exception"
    assert fixture_marker(partial_guard, True)
    narrow_guard = copy.deepcopy(guard_rejected)
    narrow_guard["repair"]["stats"]["refined_guard"].update(failure_abs_residual=1e-12,
        failure_threshold=1e-12, failure_ratio=1, max_ratio=1)
    assert fixture_marker(narrow_guard, True)
    unsupported = copy.deepcopy(guard_rejected)
    unsupported["repair"]["stats"]["refined_guard"] = copy.deepcopy(idle_guard)
    unsupported["repair"]["stats"]["initial_refinement"].update(status="unsupported_precision",
        solves=0, long_double_digits=53, correction_max_abs=None)
    assert fixture_marker(unsupported, True)
    null_guard = copy.deepcopy(guard_rejected)
    for key in GUARD_NUMBERS:
        null_guard["repair"]["stats"]["refined_guard"][key] = None
    assert fixture_marker(null_guard, True)
    accepted_refinement_budget = copy.deepcopy(refined_candidate)
    accepted_refinement_budget.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    accepted_refinement_budget["stats"]["slack_matvecs"] = 0
    accepted_refinement_budget["repair"].update(status="budget_exception", candidate_available=False)
    accepted_refinement_budget["repair"]["stats"].update(phase="initial_refinement", final_selected_rows_count=0)
    assert fixture_marker(accepted_refinement_budget, True)
    forward_candidate = copy.deepcopy(candidate)
    forward_stats = forward_candidate["repair"]["stats"]
    forward_stats.update(iterations=1, phase1_iterations=1, pivots=1, direction_solves=1,
        direction_matvecs=1, pricing_matvecs=1, dual_solves=1, lu_analyze_calls=1, lu_factor_calls=1, primal_solves=1,
        slack_matvecs=2, factor_nonzeros=4)
    forward_stats["direction_forward"] = completed_forward("direction", 0)
    forward_stats["rebuild_forward"] = completed_forward("rebuild", 1)
    forward_stats["history"] = committed_history(1)
    forward_stats["phase1_transition"] = strict_transition(1)
    forward_stats["search"] = search_snapshot("pivot_committed", 1, 1, 1, 0)
    forward_stats["eta"] = full_eta(1, 1)
    assert fixture_marker(forward_candidate, True)
    larger_cumulative_candidate = copy.deepcopy(forward_candidate)
    larger_cumulative_candidate["repair"]["stats"].update(factor_nonzeros=32000002, last_factor_nonzeros=32000000)
    assert fixture_marker(larger_cumulative_candidate, True)
    excessive_single_candidate = copy.deepcopy(larger_cumulative_candidate)
    excessive_single_candidate["repair"]["stats"].update(factor_nonzeros=32000003, last_factor_nonzeros=32000001)
    assert not valid_marker(excessive_single_candidate, True)
    def refined_forward(kind, sequence):
        result = completed_forward(kind, sequence)
        result.update(guard_failures=1, attempts=1, solves=1, accepted=1,
            before=copy.deepcopy(refined_stats["initial_guard"]),
            after=copy.deepcopy(refined_stats["refined_guard"]),
            refinement=copy.deepcopy(refined_stats["initial_refinement"]))
        return result
    direction_refined = copy.deepcopy(forward_candidate)
    direction_refined["repair"]["stats"]["direction_forward"] = refined_forward("direction", 0)
    direction_refined["repair"]["stats"]["direction_matvecs"] = 2
    assert fixture_marker(direction_refined, True)
    rebuild_refined = copy.deepcopy(forward_candidate)
    rebuild_refined["repair"]["stats"]["rebuild_forward"] = refined_forward("rebuild", 1)
    assert fixture_marker(rebuild_refined, True)
    forward_budget = copy.deepcopy(direction_refined)
    forward_budget.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    forward_budget["stats"]["slack_matvecs"] = 0
    forward_budget["repair"].update(status="budget_exception", candidate_available=False)
    forward_budget["repair"]["stats"].update(phase="phase1", pivots=0, lu_analyze_calls=0,
        lu_factor_calls=0, primal_solves=0, slack_matvecs=1, factor_nonzeros=2, final_selected_rows_count=0)
    forward_budget["repair"]["stats"]["rebuild_forward"] = copy.deepcopy(idle_forward)
    forward_budget["repair"]["stats"]["history"] = committed_history(0)
    forward_budget["repair"]["stats"]["phase1_transition"] = copy.deepcopy(idle_transition)
    forward_budget["repair"]["stats"]["search"] = search_snapshot("evaluating", 1, 1, 0, 0)
    forward_budget["repair"]["stats"]["eta"] = full_eta(0, 0)
    assert fixture_marker(forward_budget, True)
    partial_forward = copy.deepcopy(forward_budget)
    partial_direction = partial_forward["repair"]["stats"]["direction_forward"]
    partial_direction.update(guard_failures=0, attempts=0, solves=0, accepted=0,
        after=copy.deepcopy(idle_guard), refinement=copy.deepcopy(idle_refinement))
    partial_direction["before"]["complete"] = False
    partial_forward["repair"]["stats"]["direction_matvecs"] = 1
    partial_forward["repair"]["stats"]["search"]["status"] = "direction"
    assert fixture_marker(partial_forward, True)
    direction_rejected = copy.deepcopy(forward_budget)
    direction_rejected.update(stage="candidate_unavailable", recovery_status="repair_unavailable")
    direction_rejected["repair"]["status"] = "direction_residual_failure"
    direction_rejected_stats = direction_rejected["repair"]["stats"]["direction_forward"]
    direction_rejected_stats.update(accepted=0, after=copy.deepcopy(refined_stats["initial_guard"]))
    direction_rejected_stats["refinement"].update(status="guard_rejected", accepted=0)
    direction_rejected["repair"]["stats"]["search"]["status"] = "direction"
    assert fixture_marker(direction_rejected, True)
    rebuild_rejected = copy.deepcopy(rebuild_refined)
    rebuild_rejected.update(accepted=False, candidate_available=False, stage="candidate_unavailable",
        recovery_status="repair_unavailable", trial_verification=None)
    rebuild_rejected["stats"]["slack_matvecs"] = 0
    rebuild_rejected["repair"].update(status="basis_residual_failure", candidate_available=False)
    rebuild_rejected["repair"]["stats"].update(phase="phase1", pivots=0, final_selected_rows_count=0)
    rebuild_rejected["repair"]["stats"]["history"] = committed_history(0)
    rebuild_rejected["repair"]["stats"]["phase1_transition"] = copy.deepcopy(idle_transition)
    rebuild_rejected["repair"]["stats"]["search"]["status"] = "step_started"
    rebuild_rejected["repair"]["stats"]["eta"] = full_eta(0, 1)
    rebuild_rejected_stats = rebuild_rejected["repair"]["stats"]["rebuild_forward"]
    rebuild_rejected_stats.update(accepted=0, after=copy.deepcopy(refined_stats["initial_guard"]))
    rebuild_rejected_stats["refinement"].update(status="guard_rejected", accepted=0)
    assert fixture_marker(rebuild_rejected, True)
    rebuild_budget = copy.deepcopy(rebuild_refined)
    rebuild_budget.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    rebuild_budget["stats"]["slack_matvecs"] = 0
    rebuild_budget["repair"].update(status="budget_exception", candidate_available=False)
    rebuild_budget["repair"]["stats"].update(phase="phase1", pivots=0, final_selected_rows_count=0)
    rebuild_budget["repair"]["stats"]["history"] = committed_history(0)
    rebuild_budget["repair"]["stats"]["phase1_transition"] = copy.deepcopy(idle_transition)
    rebuild_budget["repair"]["stats"]["search"]["status"] = "step_started"
    rebuild_budget["repair"]["stats"]["eta"] = full_eta(0, 1)
    assert fixture_marker(rebuild_budget, True)
    partial_rebuild = copy.deepcopy(rebuild_budget)
    partial_rebuild_stats = partial_rebuild["repair"]["stats"]["rebuild_forward"]
    partial_rebuild_stats.update(accepted=0, after=copy.deepcopy(refined_stats["initial_guard"]))
    partial_rebuild_stats["after"]["complete"] = False
    partial_rebuild_stats["refinement"].update(status="budget_exception", accepted=0)
    assert fixture_marker(partial_rebuild, True)
    multiple_forward = copy.deepcopy(direction_refined)
    multiple_stats = multiple_forward["repair"]["stats"]
    multiple_stats.update(iterations=2, phase1_iterations=2, pivots=2, direction_solves=2,
        direction_matvecs=4, pricing_matvecs=2, dual_solves=2, lu_analyze_calls=2, lu_factor_calls=2,
        primal_solves=2, slack_matvecs=3, factor_nonzeros=6)
    multiple_stats["direction_forward"].update(systems=2, guard_failures=2, attempts=2, solves=2, accepted=2, pivot_sequence=1)
    multiple_stats["rebuild_forward"].update(systems=2, pivot_sequence=2)
    multiple_stats["history"] = committed_history(2)
    multiple_stats["phase1_transition"] = strict_transition(2)
    multiple_stats["search"] = search_snapshot("pivot_committed", 2, 2, 2, 1)
    multiple_stats["eta"] = full_eta(2, 2)
    assert fixture_marker(multiple_forward, True)
    near_committed = copy.deepcopy(forward_candidate)
    near_transition = near_committed["repair"]["stats"]["phase1_transition"]
    near_transition.update(status="near_accepted", strict_accepted=0, near_attempts=1, near_accepted=1,
        checked_rows=2, max_slack_change_row=0, old_phi=1, new_phi=1, delta_phi=0,
        blocking_slack=-1e-15, released_step=1e-15, max_slack_change_ratio=0.5,
        blocking_near=True, released_near=True, all_rows_near=True)
    assert fixture_marker(near_committed, True)
    near_pending = copy.deepcopy(near_committed)
    near_pending.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    near_pending["stats"]["slack_matvecs"] = 0
    near_pending["repair"].update(status="budget_exception", candidate_available=False)
    near_pending["repair"]["stats"].update(phase="phase1", pivots=0, final_selected_rows_count=0)
    near_pending["repair"]["stats"]["phase1_transition"].update(status="near_candidate", near_accepted=0, delta_phi=1e-16)
    near_pending["repair"]["stats"]["history"] = committed_history(0)
    near_pending["repair"]["stats"]["search"]["status"] = "step_started"
    near_pending["repair"]["stats"]["eta"] = full_eta(0, 1)
    assert fixture_marker(near_pending, True)
    near_partial = copy.deepcopy(near_pending)
    near_partial["repair"]["stats"]["phase1_transition"].update(status="near_check", checked_rows=1,
        all_rows_near=False, max_slack_change_ratio=0.2)
    assert fixture_marker(near_partial, True)
    near_block_rejected = copy.deepcopy(near_pending)
    near_block_rejected.update(stage="candidate_unavailable", recovery_status="repair_unavailable")
    near_block_rejected["repair"]["status"] = "objective_not_improved"
    near_block_rejected["repair"]["stats"]["phase1_transition"].update(status="blocking_not_near",
        blocking_near=False, blocking_slack=1e-5)
    assert fixture_marker(near_block_rejected, True)
    near_row_rejected = copy.deepcopy(near_block_rejected)
    near_row_rejected["repair"]["stats"]["phase1_transition"].update(status="row_change_not_near",
        blocking_near=True, all_rows_near=False, blocking_slack=-1e-15, max_slack_change_ratio=2)
    assert fixture_marker(near_row_rejected, True)
    hash_collision = copy.deepcopy(forward_candidate)
    hash_collision["repair"]["stats"]["history"].update(hash_matches=1, exact_checks=1, membership_comparisons=2)
    assert fixture_marker(hash_collision, True)
    repeated_basis = copy.deepcopy(multiple_forward)
    repeated_basis.update(accepted=False, candidate_available=False, stage="candidate_unavailable",
        recovery_status="repair_unavailable", trial_verification=None)
    repeated_basis["stats"]["slack_matvecs"] = 0
    repeated_basis["repair"].update(status="basis_repeated", candidate_available=False)
    repeated_stats = repeated_basis["repair"]["stats"]
    repeated_stats.update(phase="phase1", pivots=1, final_selected_rows_count=0)
    repeated_stats["history"].update(status="repeated", states=2, swaps=1, candidate_checks=2,
        hash_comparisons=2, hash_matches=1, exact_checks=1, membership_comparisons=2, repeats=1, repeated_state=0)
    repeated_stats["phase1_transition"] = copy.deepcopy(near_pending["repair"]["stats"]["phase1_transition"])
    repeated_stats["phase1_transition"].update(status="basis_repeated", trials=2, strict_accepted=1)
    repeated_stats["search"]["status"] = "step_started"
    repeated_stats["eta"] = full_eta(1, 2)
    assert fixture_marker(repeated_basis, True)
    history_partial = copy.deepcopy(repeated_basis)
    history_partial.update(stage="time_limit", recovery_status="budget_exception")
    history_partial["repair"]["status"] = "budget_exception"
    history_partial["repair"]["stats"]["history"].update(status="checking", hash_comparisons=3,
        replay_swaps=1, membership_comparisons=1, repeats=0, repeated_state=None)
    history_partial["repair"]["stats"]["phase1_transition"]["status"] = "near_candidate"
    assert fixture_marker(history_partial, True)
    unique_pending = copy.deepcopy(near_pending)
    unique_pending["repair"]["stats"]["history"].update(status="unique", candidate_checks=1, hash_comparisons=1)
    assert fixture_marker(unique_pending, True)
    nullable_transition = copy.deepcopy(near_committed)
    for key in TRANSITION_NUMBERS:
        nullable_transition["repair"]["stats"]["phase1_transition"][key] = None
    assert fixture_marker(nullable_transition, True)
    history_initializing = copy.deepcopy(candidate)
    history_initializing.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    history_initializing["stats"]["slack_matvecs"] = 0
    history_initializing["repair"].update(status="budget_exception", candidate_available=False)
    history_initializing["repair"]["stats"].update(phase="initial_residual", final_selected_rows_count=0)
    history_initializing["repair"]["stats"]["history"] = copy.deepcopy(idle_history)
    history_initializing["repair"]["stats"]["history"]["status"] = "initializing"
    assert fixture_marker(history_initializing, True)
    phase2_after_search = copy.deepcopy(multiple_forward)
    phase2_stats = phase2_after_search["repair"]["stats"]
    phase2_stats.update(phase1_iterations=1, phase2_iterations=1, pricing_matvecs=1)
    phase2_stats["phase1_transition"] = strict_transition(1)
    phase2_stats["search"] = search_snapshot("pivot_committed", 1, 1, 1, 0)
    phase2_stats["direction_forward"]["entering_row"] = 1
    phase2_stats["rebuild_forward"]["entering_row"] = 1
    assert fixture_marker(phase2_after_search, True)
    multiple_candidates = copy.deepcopy(direction_refined)
    multiple_candidates["stats"].update(n=3, m=4, input_nonzeros=6, rank=3, lu_l_nonzeros=3, lu_u_nonzeros=3,
        ordered_rows_count=4, selected_rows_count=3, selection_candidates=3, selection_pivots=3, pivot_columns_count=3)
    trial_stats = multiple_candidates["repair"]["stats"]
    trial_stats.update(direction_solves=2, direction_matvecs=3, factor_nonzeros=12, last_factor_nonzeros=6,
        final_selected_rows_count=3)
    for guard in (trial_stats["initial_guard"], trial_stats["refined_guard"],
        trial_stats["direction_forward"]["before"], trial_stats["direction_forward"]["after"],
        trial_stats["rebuild_forward"]["before"], trial_stats["rebuild_forward"]["after"]):
        if guard["complete"]:
            guard["checked_rows"] = 3
    trial_stats["direction_forward"].update(systems=2, entering_row=1, entering_basis_position=1)
    trial_stats["rebuild_forward"].update(entering_row=1, entering_basis_position=1)
    trial_stats["search"].update(candidates=2, trials=2, non_descent=1, candidate_count=2, ordinal=1,
        entering_row=1, entering_basis_position=1, priced_rc=-2, linear_slope=-2, zero_ineq=1,
        zero_ineq_rows=1, true_derivative=-1)
    assert fixture_marker(multiple_candidates, True)
    exhausted_search = copy.deepcopy(multiple_candidates)
    exhausted_search.update(accepted=False, candidate_available=False, stage="candidate_unavailable",
        recovery_status="repair_unavailable", trial_verification=None)
    exhausted_search["stats"]["slack_matvecs"] = 0
    exhausted_search["repair"].update(status="direction_search_exhausted", candidate_available=False)
    exhausted_stats = exhausted_search["repair"]["stats"]
    exhausted_stats.update(phase="phase1", pivots=0, direction_solves=3, direction_matvecs=4,
        lu_analyze_calls=0, lu_factor_calls=0, primal_solves=0, slack_matvecs=1,
        factor_nonzeros=6, last_factor_nonzeros=6, final_selected_rows_count=0)
    exhausted_stats["direction_forward"].update(systems=3, entering_row=2, entering_basis_position=2)
    exhausted_stats["rebuild_forward"] = copy.deepcopy(idle_forward)
    exhausted_stats["history"] = committed_history(0)
    exhausted_stats["phase1_transition"] = copy.deepcopy(idle_transition)
    exhausted_stats["search"].update(status="exhausted", candidates=3, trials=3, non_descent=3, descent=0,
        exhausted=1, candidate_count=3, ordinal=2, entering_row=2, entering_basis_position=2,
        priced_rc=-1, linear_slope=0, true_derivative=1, linear_minus_price=1, step_started=False)
    exhausted_stats["eta"] = full_eta(0, 0)
    assert fixture_marker(exhausted_search, True)
    empty_search = copy.deepcopy(limited_repair)
    empty_search["repair"]["status"] = "no_entering"
    empty_search["repair"]["stats"]["work_units"] = 100
    empty_search["repair"]["stats"]["search"].update(status="exhausted", exhausted=1)
    assert fixture_marker(empty_search, True)
    pending_rhs = copy.deepcopy(multiple_candidates)
    pending_rhs.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    pending_rhs["stats"]["slack_matvecs"] = 0
    pending_rhs["repair"].update(status="budget_exception", candidate_available=False)
    pending_stats = pending_rhs["repair"]["stats"]
    pending_stats.update(phase="phase1", pivots=0, direction_solves=1, direction_matvecs=2,
        lu_analyze_calls=0, lu_factor_calls=0, primal_solves=0, slack_matvecs=1,
        factor_nonzeros=6, last_factor_nonzeros=6, final_selected_rows_count=0)
    pending_stats["direction_forward"].update(systems=1, entering_row=0, entering_basis_position=0)
    pending_stats["rebuild_forward"] = copy.deepcopy(idle_forward)
    pending_stats["history"] = committed_history(0)
    pending_stats["phase1_transition"] = copy.deepcopy(idle_transition)
    pending_stats["search"].update(status="direction", descent=0, zero_ineq_rows=0,
        linear_slope=None, zero_ineq=None, zero_eq=None, true_derivative=None, linear_minus_price=None,
        derivative_complete=False, step_started=False)
    pending_stats["eta"] = full_eta(0, 0)
    assert fixture_marker(pending_rhs, True)
    repricing_budget = copy.deepcopy(multiple_candidates)
    repricing_budget.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    repricing_budget["stats"]["slack_matvecs"] = 0
    repricing_budget["repair"].update(status="budget_exception", candidate_available=False)
    repricing_stats = repricing_budget["repair"]["stats"]
    repricing_stats.update(phase="phase1", pricing_matvecs=2, last_right_derivative=-1, last_step=1)
    repricing_stats["search"].update(status="preparing", bases=2, basis_pivots=1, candidate_count=0,
        zero_ineq_rows=0, zero_eq_rows=0, derivative_complete=False, step_started=False)
    for key in SEARCH_COORDINATES + SEARCH_NUMBERS:
        repricing_stats["search"][key] = None
    assert fixture_marker(repricing_budget, True)
    def eta_chain_marker(commits, full_count):
        value = copy.deepcopy(forward_candidate)
        value["stats"].update(m=12, input_nonzeros=12, ordered_rows_count=12)
        stats = value["repair"]["stats"]
        stats.update(iterations=commits, phase1_iterations=commits, pivots=commits,
            primal_solves=commits, direction_solves=commits, direction_matvecs=commits,
            dual_solves=commits + 1, pricing_matvecs=commits, slack_matvecs=commits + 1,
            lu_analyze_calls=full_count, lu_factor_calls=full_count,
            factor_nonzeros=2 + 2 * full_count, last_factor_nonzeros=2, work_units=4096)
        stats["direction_forward"] = completed_forward("direction", commits - 1)
        stats["direction_forward"]["systems"] = commits
        stats["rebuild_forward"] = completed_forward("rebuild", commits)
        stats["rebuild_forward"]["systems"] = commits
        stats["history"] = committed_history(commits)
        stats["phase1_transition"] = strict_transition(commits)
        stats["search"] = search_snapshot("pivot_committed", commits, commits, commits, commits - 1)
        eta_commits = commits - full_count
        chain = eta_commits if not full_count else 0
        stats["eta"] = {**copy.deepcopy(idle_eta), "status": "transpose", "last_mode": "full" if full_count else "eta",
            "last_fallback": "chain_full" if full_count else "none", "prepares": commits,
            "eta_trials": eta_commits, "full_refactor_trials": full_count, "eta_commits": eta_commits,
            "full_refactor_commits": full_count, "chain_length": chain, "max_chain_length": 8,
            "fallbacks": 0, "forward_calls": 2 * commits,
            "transpose_calls": commits + 1, "base_forward_solves": 2 * commits,
            "base_transpose_solves": commits + 1,
            "forward_applications": 64 if not full_count else 72, "transpose_applications": 36,
            "work_units": 1024, "stored_entries": 2 * chain, "peak_stored_entries": 16,
            "last_alpha": None if full_count else 1}
        return value
    eta_eight = eta_chain_marker(8, 0)
    assert valid_marker(eta_eight, True)
    eta_then_lu = eta_chain_marker(9, 1)
    assert valid_marker(eta_then_lu, True)
    eta_pending = copy.deepcopy(eta_eight)
    eta_pending.update(accepted=False, candidate_available=False, stage="time_limit",
        recovery_status="budget_exception", trial_verification=None)
    eta_pending["stats"]["slack_matvecs"] = 0
    eta_pending["repair"].update(status="budget_exception", candidate_available=False)
    eta_pending_stats = eta_pending["repair"]["stats"]
    eta_pending_stats.update(phase="phase1", pivots=7, dual_solves=8, final_selected_rows_count=1)
    eta_pending_stats["eta"].update(status="forward", eta_commits=7, chain_length=7, max_chain_length=7,
        stored_entries=14, peak_stored_entries=14, transpose_calls=8, base_transpose_solves=8,
        transpose_applications=28)
    eta_pending_stats["history"].update(states=8, swaps=7)
    eta_pending_stats["phase1_transition"].update(status="strict_candidate", strict_accepted=7)
    eta_pending_stats["search"]["status"] = "step_started"
    assert valid_marker(eta_pending, True)
    eta_apply_failure = copy.deepcopy(eta_pending)
    eta_apply_failure.update(stage="candidate_unavailable", recovery_status="repair_unavailable")
    eta_apply_failure["repair"]["status"] = "arithmetic_failure"
    eta_failure_stats = eta_apply_failure["repair"]["stats"]
    eta_failure_stats.update(slack_matvecs=8)
    eta_failure_stats["eta"].update(status="apply_failure", forward_applications=63)
    eta_failure_stats["rebuild_forward"].update(systems=7, pivot_sequence=7)
    eta_failure_stats["history"].update(status="unique", candidate_checks=7, hash_comparisons=28)
    eta_failure_stats["phase1_transition"] = strict_transition(7)
    assert valid_marker(eta_apply_failure, True)
    eta_before_base_budget = copy.deepcopy(eta_apply_failure)
    eta_before_base_budget.update(stage="time_limit", recovery_status="budget_exception")
    eta_before_base_budget["repair"]["status"] = "budget_exception"
    eta_before_base_budget["repair"]["stats"]["eta"].update(status="forward", base_forward_solves=15,
        forward_applications=56)
    assert valid_marker(eta_before_base_budget, True)
    eta_full_pending = copy.deepcopy(eta_pending)
    eta_full_pending_stats = eta_full_pending["repair"]["stats"]
    eta_full_pending_stats.update(lu_analyze_calls=1, lu_factor_calls=1, factor_nonzeros=4)
    eta_full_pending_stats["eta"].update(status="forward", last_mode="full", last_fallback="alpha_zero",
        eta_trials=7, full_refactor_trials=1, fallbacks=1, forward_applications=56,
        transpose_calls=8, base_transpose_solves=8, transpose_applications=28, last_alpha=0)
    assert valid_marker(eta_full_pending, True)
    rejected = copy.deepcopy(candidate)
    rejected.update(accepted=False, stage="original_verification_rejected")
    rejected["trial_verification"]["dual_feasible"] = False
    assert valid_marker(rejected, True)
    expired = copy.deepcopy(enabled)
    expired.update(attempted=True, stage="time_limit", attempt_seconds=0.1)
    assert valid_marker(expired, True)
    partial = copy.deepcopy(expired)
    partial.update(recovery_status="budget_exception")
    partial["stats"].update(n=3, m=4, input_nonzeros=8, budget_checks=2, rank=1, selection_candidates=2,
        selection_pivots=1, selected_rows_count=1, pivot_columns_count=1, ordered_rows_count=4,
        work_units=1000000000, l_nonzeros=5, min_abs_pivot=1, selection_threshold=1e-12)
    assert valid_marker(partial, True)
    rejected_column = copy.deepcopy(partial)
    rejected_column["stats"].update(selection_candidates=3, selection_rejected=1)
    assert valid_marker(rejected_column, True)
    work_limit = copy.deepcopy(partial)
    work_limit.update(stage="candidate_unavailable", recovery_status="selection_work_limit")
    assert valid_marker(work_limit, True)
    arithmetic_failure = copy.deepcopy(partial)
    arithmetic_failure.update(stage="candidate_unavailable", recovery_status="selection_update_underflow")
    assert valid_marker(arithmetic_failure, True)
    mutations = [(("enabled",), False), (("accepted",), False), (("stage",), "verifying"),
        (("stage",), []), (("recovery_status",), []),
        (("candidate_available",), False), (("schema",), True), (("attempt_seconds",), -1),
        (("verification_seconds",), float("nan")), (("work_in_total_seconds",), False),
        (("stats", "qr_factor_calls"), 2), (("stats", "rank"), 2), (("stats", "selected_rows_count"), 0),
        (("trial_verification", "primal_feasible"), False), (("trial_verification", "objective_gap"), None),
        (("stats", "work_units"), 1000000001), (("stats", "pivot_columns_count"), 0),
        (("stats", "selection_candidates"), 2), (("stats", "selection_rejected"), 1),
        (("stats", "min_abs_pivot"), 0), (("stats", "min_abs_pivot"), None),
        (("stats", "selection_threshold"), 0), (("stats", "selection_threshold"), None),
        (("stats", "selection_threshold"), float("nan")),
        (("repair", "status"), "unknown"), (("repair", "candidate_available"), False),
        (("repair", "candidate_available"), 1), (("repair", "stats", "phase"), "phase2"),
        (("repair", "stats", "attempts"), 2), (("repair", "stats", "work_units"), 400000001),
        (("repair", "stats", "pivots"), 513), (("repair", "stats", "phase1_iterations"), 1),
        (("repair", "stats", "lu_factor_calls"), 1), (("repair", "stats", "final_selected_rows_count"), 0),
        (("repair", "stats", "factor_nonzeros"), 64000001), (("repair", "stats", "last_factor_nonzeros"), 0),
        (("repair", "stats", "last_factor_nonzeros"), 3), (("repair", "stats", "zero_steps"), 2),
        (("repair", "stats", "final_phi"), -1), (("repair", "stats", "last_right_derivative"), float("nan")),
        (("repair", "stats", "initial_guard", "complete"), False),
        (("repair", "stats", "initial_guard", "checked_rows"), 2),
        (("repair", "stats", "initial_guard", "violations"), 1),
        (("repair", "stats", "initial_refinement", "status"), "accepted"),
        (("repair", "stats", "initial_refinement", "double_digits"), 53)]
    for key in UNUSED_QR_COUNTS:
        mutations.append((("stats", key), 1))
    mutations.append((("stats", "qr_default_threshold"), 1e-12))
    for key in MARKER_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            mutations.append((("stats", key), bad))
    for key in REPAIR_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            mutations.append((("repair", "stats", key), bad))
    for path, bad in mutations:
        value = copy.deepcopy(candidate)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        assert not valid_marker(value, True), path
    guard_mutations = [("initial_guard", "passed", True), ("initial_guard", "checked_rows", 0),
        ("initial_guard", "violations", 0), ("initial_guard", "failure_row", None),
        ("initial_guard", "failure_basis_position", None), ("initial_guard", "failure_row", 2),
        ("initial_guard", "failure_basis_position", 1), ("initial_guard", "failure_threshold", 0),
        ("refined_guard", "complete", False), ("refined_guard", "passed", False),
        ("initial_refinement", "status", "guard_rejected"), ("initial_refinement", "accepted", 0),
        ("initial_refinement", "solves", 0), ("initial_refinement", "attempts", 0),
        ("initial_refinement", "long_double_digits", 53), ("initial_refinement", "status", "unknown")]
    for field in ("complete", "passed"):
        for bad in (0, 1, None):
            guard_mutations.append(("initial_guard", field, bad))
    for field in GUARD_COUNTS + GUARD_COORDINATES:
        for bad in (True, -1, 1.0, 1 << 63):
            guard_mutations.append(("initial_guard", field, bad))
    for field in GUARD_NUMBERS:
        for bad in (float("nan"), -1):
            guard_mutations.append(("initial_guard", field, bad))
    for field in INITIAL_REFINEMENT_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            guard_mutations.append(("initial_refinement", field, bad))
    for bad in (float("inf"), -1):
        guard_mutations.append(("initial_refinement", "correction_max_abs", bad))
    for group, key, bad in guard_mutations:
        value = copy.deepcopy(refined_candidate)
        value["repair"]["stats"][group][key] = bad
        assert not valid_marker(value, True), (group, key, bad)
    forward_mutations = [(("direction_forward", "systems"), 0), (("direction_forward", "systems"), 2),
        (("direction_forward", "guard_failures"), 0), (("direction_forward", "attempts"), 0),
        (("direction_forward", "solves"), 0), (("direction_forward", "accepted"), 0),
        (("direction_forward", "pivot_sequence"), 1), (("direction_forward", "tag"), "rebuild"),
        (("direction_forward", "entering_row"), None), (("direction_forward", "entering_row"), 2),
        (("direction_forward", "entering_basis_position"), 1),
        (("direction_forward", "before", "complete"), False),
        (("direction_forward", "after", "passed"), False),
        (("direction_forward", "refinement", "accepted"), 0),
        (("rebuild_forward", "systems"), 0), (("rebuild_forward", "systems"), 2),
        (("rebuild_forward", "pivot_sequence"), 0), (("rebuild_forward", "pivot_sequence"), 2)]
    for group in ("direction_forward", "rebuild_forward"):
        for key in FORWARD_COUNTS:
            for bad in (True, -1, 1.0, 513):
                forward_mutations.append(((group, key), bad))
        for key in FORWARD_COORDINATES:
            for bad in (True, -1, 0.0):
                forward_mutations.append(((group, key), bad))
    for path, bad in forward_mutations:
        value = copy.deepcopy(direction_refined)
        target = value["repair"]["stats"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        assert not valid_marker(value, True), (path, bad)
    transition_history_mutations = [("phase1_transition", "status", "unknown"),
        ("phase1_transition", "trials", 0), ("phase1_transition", "strict_accepted", 1),
        ("phase1_transition", "near_attempts", 0), ("phase1_transition", "near_accepted", 2),
        ("phase1_transition", "checked_rows", 1), ("phase1_transition", "blocking_row", None),
        ("phase1_transition", "released_row", None), ("phase1_transition", "max_slack_change_row", None),
        ("phase1_transition", "blocking_near", False), ("phase1_transition", "released_near", False),
        ("phase1_transition", "all_rows_near", False),
        ("history", "status", "unknown"), ("history", "initialized", False),
        ("history", "states", 1), ("history", "states", 3), ("history", "swaps", 0),
        ("history", "candidate_checks", 0), ("history", "candidate_checks", 3),
        ("history", "hash_comparisons", 0), ("history", "hash_comparisons", 3),
        ("history", "hash_matches", 1), ("history", "exact_checks", 1),
        ("history", "replay_swaps", 1), ("history", "membership_comparisons", 1),
        ("history", "repeats", 1), ("history", "repeated_state", 0)]
    for key in TRANSITION_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            transition_history_mutations.append(("phase1_transition", key, bad))
    for key in TRANSITION_COORDINATES:
        for bad in (True, -1, 0.0, 2):
            transition_history_mutations.append(("phase1_transition", key, bad))
    for key in TRANSITION_NUMBERS:
        transition_history_mutations.append(("phase1_transition", key, float("nan")))
        if key not in ("delta_phi", "blocking_slack"):
            transition_history_mutations.append(("phase1_transition", key, -1))
    for key in TRANSITION_BOOLS:
        transition_history_mutations.append(("phase1_transition", key, 1))
    for key in HISTORY_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            transition_history_mutations.append(("history", key, bad))
    for group, key, bad in transition_history_mutations:
        value = copy.deepcopy(near_committed)
        value["repair"]["stats"][group][key] = bad
        assert not valid_marker(value, True), (group, key, bad)
    empty_history_in_phase1 = copy.deepcopy(near_pending)
    empty_history_in_phase1["repair"]["stats"]["history"] = copy.deepcopy(idle_history)
    assert not valid_marker(empty_history_in_phase1, True)
    search_mutations = [(("search", "status"), "unknown"), (("search", "bases"), 0),
        (("search", "bases"), 3), (("search", "candidates"), 4), (("search", "candidates"), 1),
        (("search", "trials"), 4), (("search", "trials"), 1), (("search", "non_descent"), 3),
        (("search", "descent"), 2), (("search", "descent"), 0), (("search", "exhausted"), 1),
        (("search", "basis_pivots"), 2), (("search", "candidate_count"), 4),
        (("search", "zero_ineq_rows"), 5), (("search", "zero_eq_rows"), 5),
        (("search", "ordinal"), 2), (("search", "entering_row"), 4),
        (("search", "entering_basis_position"), 3), (("search", "derivative_complete"), False),
        (("search", "step_started"), False), (("direction_solves",), 4),
        (("direction_forward", "systems"), 4)]
    for key in SEARCH_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            search_mutations.append((("search", key), bad))
    for key in SEARCH_COORDINATES:
        for bad in (True, -1, 0.0, 1 << 63):
            search_mutations.append((("search", key), bad))
    for key in SEARCH_NUMBERS:
        search_mutations.append((("search", key), float("nan")))
    for key in ("zero_ineq", "zero_eq"):
        search_mutations.append((("search", key), -1))
    for key in SEARCH_BOOLS:
        for bad in (0, 1, None):
            search_mutations.append((("search", key), bad))
    for path, bad in search_mutations:
        value = copy.deepcopy(multiple_candidates)
        target = value["repair"]["stats"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        assert not valid_marker(value, True), (path, bad)
    eta_mutations = [("status", "unknown"), ("last_mode", "unknown"), ("last_fallback", "unknown"),
        ("prepares", 8), ("eta_trials", 7), ("full_refactor_trials", 0), ("eta_commits", 9),
        ("full_refactor_commits", 0), ("chain_length", 1), ("max_chain_length", 7),
        ("max_chain_length", 9), ("fallbacks", 10), ("fallbacks", 2), ("forward_calls", 19),
        ("transpose_calls", 11), ("base_forward_solves", 19),
        ("base_transpose_solves", 11), ("forward_applications", 145), ("transpose_applications", 81),
        ("work_units", 4097), ("stored_entries", 1), ("peak_stored_entries", 17),
        ("last_alpha", float("nan")), ("last_alpha", True)]
    for key in ETA_COUNTS:
        for bad in (True, -1, 1.0, 1 << 63):
            eta_mutations.append((key, bad))
    for key in ("status", "last_mode", "last_fallback"):
        for bad in (None, []):
            eta_mutations.append((key, bad))
    for key, bad in eta_mutations:
        value = copy.deepcopy(eta_then_lu)
        value["repair"]["stats"]["eta"][key] = bad
        assert not valid_marker(value, True), (key, bad)
    eta_mode_mutations = [("last_alpha", 0), ("last_alpha", None), ("last_mode", "full")]
    for key, bad in eta_mode_mutations:
        value = copy.deepcopy(eta_eight)
        value["repair"]["stats"]["eta"][key] = bad
        assert not valid_marker(value, True), (key, bad)
    report = {"schema_version": 1, "backend": "ipm", "status": "numerical_error", "backend_version": "ipm-cpu-0.53.0",
        "options": {"ipm": {"linear_accuracy": "homogeneous_repair", "linear_solver": "normal_cholmod_aat"}},
        **dict.fromkeys(base.COUNT_FIELDS, 0), "ipm_factor_backend": None,
        "ipm_homogeneous_tau": None, "ipm_homogeneous_kappa": None,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1}}
    row = {"error": None, "classification": "incomplete", "returncode": 4, "status": "numerical_error", "report": report,
        "numerical_seconds": 1, "elapsed_wall_seconds": 2, "configuration_matches": True, "basis_marker_error": None}
    plan = {"time_limit": 5, "wall_timeout": 35, "compatibility_excluded_report_fields": ["timings"]}
    assert pair_result(row, row, plan)["category"] == "equal"
    off_row = dict(row, basis_marker=empty)
    assert pair_result(row, off_row, plan)["category"] == "equal" and pair_result(row, off_row, plan)["basis_sidecar_compatible"]
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["stats"]["budget_checks"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["attempted"] = True
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["budget_checks"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["initial_guard"]["checked_rows"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["initial_refinement"]["double_digits"] = 53
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["direction_forward"]["systems"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["rebuild_forward"]["tag"] = "rebuild"
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["phase1_transition"]["trials"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["history"]["status"] = "initializing"
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["search"]["bases"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["search"]["derivative_complete"] = True
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["eta"]["prepares"] = 1
    assert pair_result(row, bad_off, plan)["category"] == "different"
    bad_off = copy.deepcopy(off_row)
    bad_off["basis_marker"]["repair"]["stats"]["eta"]["status"] = "ready"
    assert pair_result(row, bad_off, plan)["category"] == "different"
    active_row = dict(row, basis_marker=candidate)
    duration_changed = copy.deepcopy(active_row)
    duration_changed["basis_marker"]["attempt_seconds"] += 0.1
    assert pair_result(active_row, duration_changed, plan)["category"] == "equal"
    assert pair_result(active_row, dict(row, basis_marker=rejected), plan)["category"] == "different"
    repair_changed = copy.deepcopy(active_row)
    repair_changed["basis_marker"]["repair"]["stats"]["work_units"] += 1
    assert pair_result(active_row, repair_changed, plan)["category"] == "different"
    assert pair_result(active_row, repair_changed, plan)["equal"] is False
    changed = copy.deepcopy(row)
    changed["report"]["backend_version"] = "ipm-cpu-other"
    assert pair_result(row, changed, plan)["category"] == "different"
    budget = copy.deepcopy(row)
    budget["status"] = budget["report"]["status"] = "time_limit"
    invalid = dict(row, basis_marker_error="marker_count")
    assert pair_result(invalid, budget, plan)["category"] == "invalid_process"
    for status, code, classification in (("numerical_error", 4, "incomplete"), ("time_limit", 4, "incomplete"),
        ("iteration_limit", 4, "incomplete"), ("verification_failed", 3, "verification_failed")):
        altered = copy.deepcopy(row)
        altered.update(status=status, returncode=code, classification=classification)
        altered["report"]["status"] = status
        assert valid_process(altered)
        altered["report"]["ipm_factorizations"] = True
        assert pair_result(altered, budget, plan)["category"] == "invalid_process"
    return {"valid_and_parse_cases": 64, "mutation_cases": len(mutations), "single_factor_cases": 4,
        "guard_refinement_valid_cases": 7, "guard_refinement_mutation_cases": len(guard_mutations),
        "forward_valid_cases": 10, "forward_mutation_cases": len(forward_mutations),
        "transition_history_valid_cases": 11,
        "transition_history_mutation_cases": len(transition_history_mutations) + 1,
        "search_valid_cases": 6, "search_mutation_cases": len(search_mutations),
        "eta_valid_cases": 6, "eta_mutation_cases": len(eta_mutations) + len(eta_mode_mutations),
        "full_report_and_budget_cases": 7, "sidecar_cases": 17}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("self-check", "compatibility", "gate", *STAGES))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    if args.stage == "self-check":
        print(json.dumps(self_check(), ensure_ascii=False))
    elif args.stage in ("compatibility", "gate"):
        result = compatibility() if args.stage == "compatibility" else gate()
        save_once(DATA / ("v80-" + args.stage + ".json"), result)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run(args.stage, args.work.resolve())
