"""复用冻结118评测运输层，只增加原生导出基和自主暖基阶段的严格账本。"""
import copy
import importlib.util
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import build_experimental as B

PARENT_FILE = ROOT / ".tools/v118-benchmark.py"
PARENT_SHA = "4edf1c46c0acfd4f9c0725e074cbba757771448cdecc84c563bd5a7d9bd9aa99"
B.require(B.sha(PARENT_FILE) == PARENT_SHA, "冻结118运输层源码变化")
spec = importlib.util.spec_from_file_location("v120_frozen_v118_transport", PARENT_FILE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
NATIVE_VALIDATOR, PARENT_SELF_CHECK = P.valid_crossover, P.self_check
NEW_VERSION = "ipm-ipx-crossover-cpu-0.3.0"
EXTRA_FIELDS = "crossover_status crossover_quality basis_export warm scope_seconds verification_seconds_total".split()
CALLS = "init factor solve_n solve_t ftran btran update".split()
COUNTS = "steps_started steps_completed pivots bound_flips zero_steps refactors rebuild_started rebuild_completed pricing_passes work_items verification_started verification_completed".split()
TIMES = "pricing_seconds rebuild_seconds verification_seconds total_seconds".split()
LAST_VALUES = "last_entering last_leaving last_theta last_pivot".split()
CHECK_NUMBERS = "equation_residual bound_violation basic_dual_residual equation_ratio bound_ratio basic_dual_ratio".split()
WARM_STATUSES = "accepted rejected invalid_input unsupported_empty singular_basis primal_infeasible numerical_error step_limit no_finite_step time_limit exception".split()
STAGES = "entry validation initialize factor rebuild solve_n solve_t pricing ftran ratio btran update verification projection cleanup finished".split()


def number(value, nonnegative=False, nullable=False):
    return P.number(value, nonnegative, nullable) and (value is None or abs(value) <= sys.float_info.max)


def contract(plan):
    value = plan["hybrid_protocol"]
    B.require(set(value) == {"validator_file", "validator_sha256", "environment", "marker", "schema",
        "on_backend_version", "off_backend_version", "marker_on_only"} and plan["candidate_version"] == 120 and
        plan["new_algorithm_round"] is True and plan["production_changes"] is False and
        plan["candidate_backend_version"] == value["on_backend_version"] == NEW_VERSION and
        value["off_backend_version"] == P.OLD_VERSION and value["environment"] == P.ENVIRONMENT and
        value["marker"] == P.MARKER and type(value["schema"]) is int and value["schema"] == 3 and
        value["marker_on_only"] is True, "混合120唯一版本/schema桥接不符")
    own = Path(__file__).resolve()
    B.require((ROOT / value["validator_file"]).resolve() == own and B.sha(own) == value["validator_sha256"], "120验证器身份变化")
    tools = {str((ROOT / name).resolve()): digest for name, digest in plan["tool_sources"].items()}
    B.require(tools.get(str(own)) == value["validator_sha256"] and tools.get(str(PARENT_FILE)) == PARENT_SHA,
        "计划必须显式锁定120及所复用的118运输层")
    B.verify(tools)
    return value


def close_time(left, right):
    return abs(left - right) <= 4 * math.ulp(max(left, right))


def checks(value):
    if value is None:
        return True
    if not P.fields(value, ["valid", "primal_feasible", "basis_dual_feasible"] + CHECK_NUMBERS) or \
        value["valid"] is not True or any(type(value[key]) is not bool for key in ("primal_feasible", "basis_dual_feasible")) or \
        not all(number(value[key], True, True) for key in CHECK_NUMBERS):
        return False
    return (not value["primal_feasible"] or all(value[key] is not None and value[key] <= 1 for key in ("equation_ratio", "bound_ratio"))) and \
        (not value["basis_dual_feasible"] or value["basic_dual_ratio"] is not None and value["basic_dual_ratio"] <= 1)


def quality_pair(raw, after, stats):
    started, completed = stats["verification_started"], stats["verification_completed"]
    if not 0 <= completed <= started <= 2 or started - completed > 1 or \
        any(value is not None and not P.quality(value) for value in (raw, after)) or \
        (raw is None) != (completed == 0) or (after is None) != (completed == 0) or \
        completed < 2 and not B.equal(raw, after):
        return False
    projection = stats["projection"]
    if started and not projection["enabled"] or projection["scan_started"] and (completed < 1 or P.complete_quality(raw)):
        return False
    if started == 2 and (projection["project_completed"] != 1 or projection["project_rc"] is not True or
        projection["changed_rows"] == 0 or P.complete_quality(raw)):
        return False
    return completed != 2 or projection["changed_rows"] == projection["eligible_rows"] and all(B.equal(raw[key], after[key]) for key in
        ("primal_valid", "primal_feasible", "objective_consistent", "objective", "max_row_violation", "max_bound_violation"))


def valid_warm(value):
    if not P.fields(value, "started completed max_steps status raw_quality after_quality stats".split()) or \
        not P.integer(value["started"], 1) or not P.integer(value["completed"], 1) or value["completed"] > value["started"] or \
        type(value["max_steps"]) is not int or value["max_steps"] != 64 or value["status"] not in WARM_STATUSES:
        return False
    stats = value["stats"]
    if not P.fields(stats, CALLS + COUNTS + TIMES + LAST_VALUES + ["initial_checks", "last_checks", "projection", "stage"]) or \
        not all(P.integer(stats[key]) for key in COUNTS) or not all(number(stats[key], True) for key in TIMES) or \
        stats["stage"] not in STAGES or not all(checks(stats[key]) for key in ("initial_checks", "last_checks")) or \
        not P.valid_projection(stats["projection"]):
        return False
    for name in CALLS:
        call = stats[name]
        if not P.fields(call, ["started", "completed", "rc", "seconds"]) or not all(P.integer(call[key]) for key in ("started", "completed")) or \
            not 0 <= call["started"] - call["completed"] <= 1 or not number(call["seconds"], True) or \
            (call["rc"] is None) != (call["completed"] == 0) or call["rc"] is not None and \
            (type(call["rc"]) is not int or not -(1 << 31) <= call["rc"] < (1 << 31)):
            return False
    if not all(stats[key] is None or P.integer(stats[key], (1 << 31) - 1) for key in ("last_entering", "last_leaving")) or \
        not number(stats["last_theta"], True, True) or not number(stats["last_pivot"], False, True):
        return False
    pivots = stats["pivots"] + stats["bound_flips"]
    if not 0 <= stats["steps_completed"] <= pivots <= stats["steps_started"] <= value["max_steps"] or \
        stats["steps_started"] - stats["steps_completed"] > 1 or stats["zero_steps"] > pivots or \
        not 0 <= stats["rebuild_started"] - stats["rebuild_completed"] <= 1 or stats["init"]["started"] > 1 or \
        stats["ftran"]["started"] > stats["steps_started"] or stats["btran"]["started"] > stats["ftran"]["completed"] or \
        stats["update"]["started"] > stats["btran"]["completed"] or stats["pivots"] > stats["update"]["completed"] or \
        not quality_pair(value["raw_quality"], value["after_quality"], stats):
        return False
    if value["completed"] == 0:
        return value["status"] in ("time_limit", "exception") and value["raw_quality"] is None and value["after_quality"] is None
    if value["status"] == "accepted":
        return all(stats[key]["started"] == stats[key]["completed"] and
            (stats[key]["completed"] == 0 or stats[key]["rc"] == 0) for key in CALLS) and \
            stats["steps_started"] == stats["steps_completed"] == pivots and \
            stats["rebuild_started"] == stats["rebuild_completed"] and \
            P.complete_quality(value["after_quality"]) and all(value["after_quality"][key] for key in P.QUALITY_BOOLS) and \
            stats["verification_started"] == stats["verification_completed"] >= 1 and all(stats[key]["completed"] >= 1 for key in
            ("init", "factor", "solve_n", "solve_t")) and all(stats[key] is not None and stats[key]["primal_feasible"] and
            stats[key]["basis_dual_feasible"] for key in ("initial_checks", "last_checks"))
    return True


def valid_crossover(value):
    if not P.fields(value, P.TOP_FIELDS) or type(value["schema"]) is not int or value["schema"] != 3 or \
        any(type(value[key]) is not bool for key in ("local_accepted", "adopted")) or \
        value["oracle_decision"] not in ("not_consulted", "not_active", "accept", "reject", "budget", "error"):
        return False
    event = value["last"]
    if event is not None and not P.fields(event, P.LAST_FIELDS + EXTRA_FIELDS):
        return False
    native = copy.deepcopy(value)
    native["schema"] = 2
    native["adopted"], native["oracle_decision"] = False, "not_consulted"
    if event is not None:
        native["last"] = {key: copy.deepcopy(event[key]) for key in P.LAST_FIELDS}
        native["last"].update(status=event["crossover_status"], after_quality=event["crossover_quality"])
        native["local_accepted"] = value["completed"] == 1 and event["crossover_status"] == "accepted"
    if not NATIVE_VALIDATOR(native):
        return False
    if event is None:
        return value["local_accepted"] is False and value["adopted"] is False and value["oracle_decision"] == "not_consulted"
    export, warm = event["basis_export"], event["warm"]
    if not P.fields(export, ["enabled", "statuses_requested", "statuses_received", "published", "columns", "rows"]) or \
        not all(type(export[key]) is bool for key in ("enabled", "statuses_requested", "statuses_received", "published")) or \
        not all(P.integer(export[key], (1 << 31) - 1) for key in ("columns", "rows")) or \
        any(not number(event[key], True) for key in ("scope_seconds", "verification_seconds_total")) or \
        event["status"] not in P.STATUSES or event["after_quality"] is not None and not P.quality(event["after_quality"]):
        return False
    if value["completed"] == 0:
        return event["status"] in ("exception", "time_limit") and warm is None and not any(export[key] for key in export) and event["after_quality"] is None and \
            value["local_accepted"] is False and value["adopted"] is False and value["oracle_decision"] == "not_consulted"
    stats = event["stats"]
    if any(not number(stats[key], True) for key in ("verification_seconds", "total_seconds")):
        return False
    if not export["enabled"] or export["statuses_requested"] != (stats["get_started"] == 1) or \
        export["statuses_received"] != (stats["get_completed"] == 1 and stats["get_rc"] == 0) or \
        export["published"] and (not export["statuses_received"] or event["crossover_status"] != "rejected"):
        return False
    if export["statuses_requested"] and (not export["columns"] or not export["rows"]):
        return False
    if warm is not None and (event["crossover_status"] != "rejected" or not export["published"] or not valid_warm(warm)):
        return False
    warm_accepted = warm is not None and warm["completed"] == 1 and warm["status"] == "accepted"
    selected_quality = warm["after_quality"] if warm_accepted else event["crossover_quality"]
    if not B.equal(event["after_quality"], selected_quality) or value["local_accepted"] is not (event["status"] == "accepted") or \
        event["status"] == "accepted" and not (event["crossover_status"] == "accepted" or warm_accepted) or \
        value["adopted"] and (not value["local_accepted"] or value["oracle_decision"] not in ("accept", "not_active")) or \
        value["oracle_decision"] != "not_consulted" and not value["local_accepted"]:
        return False
    verification = stats["verification_seconds"] + (warm["stats"]["verification_seconds"] if warm else 0)
    phases = stats["total_seconds"] + (warm["stats"]["total_seconds"] if warm else 0)
    return number(verification, True) and number(phases, True) and close_time(event["verification_seconds_total"], verification) and \
        phases <= event["scope_seconds"] + 4 * math.ulp(max(phases, event["scope_seconds"]))


def provenance(output, plan, create):
    for source, digest in ((Path(__file__).resolve(), plan["hybrid_protocol"]["validator_sha256"]), (PARENT_FILE, PARENT_SHA)):
        target = output / "provenance" / source.name
        if create and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(source.read_bytes())
        B.require(target.is_file() and B.sha(target) == digest, "混合120及冻结运输层快照缺失或变化")


def self_check(plan_path):
    previous = P.valid_crossover
    P.valid_crossover = NATIVE_VALIDATOR
    try:
        inherited = PARENT_SELF_CHECK(plan_path)
    finally:
        P.valid_crossover = previous
    # 新fixture只验证序列化和阶段事务，不执行任何数学内核或公开LP。
    good = dict.fromkeys(P.QUALITY_BOOLS, True)
    good.update(dict.fromkeys(P.QUALITY_NUMBERS, 0))
    before = dict(good, primal_feasible=False, gap_satisfied=False, numerically_optimal=False,
        max_row_violation=1e-5, objective_gap=1e-5, relative_objective_gap=1e-5)
    projection = dict.fromkeys(P.PROJECTION_COUNTS + ["max_abs_change", "seconds"], 0)
    projection.update(enabled=False, project_rc=None)
    prep = dict.fromkeys(P.PREP_COUNTS, 0)
    prep.update(max_aggregation_rounding=0, raw=dict.fromkeys(P.RESIDUAL_FIELDS, 0), dropped=dict.fromkeys(P.RESIDUAL_FIELDS, 0))
    native_stats = dict.fromkeys(P.STAT_COUNTS + P.STAT_TIMES + ["native_threads", "native_time_limit"], 0)
    native_stats.update(load_rc=None, crossover_rc=None, get_rc=None, preparation=prep, native_info=None, projection=projection)
    event = dict(iteration=1, bounds_inf=0, cost_inf=0, primal_ratio=1e-5, dual_ratio=0, relative_gap=1e-5,
        deadline_saturated=False, status="accepted", before_quality=before, unprojected_quality=good, after_quality=good,
        stats=copy.deepcopy(native_stats), crossover_status="accepted", crossover_quality=good,
        basis_export=dict(enabled=True, statuses_requested=True, statuses_received=True, published=False, columns=2, rows=1),
        warm=None, scope_seconds=0, verification_seconds_total=0)
    event["stats"].update(native_threads=1, native_time_limit=4, verification_started=1, verification_completed=1,
        native_info=dict.fromkeys(P.INFO_INTS + P.INFO_NUMBERS, 0))
    event["stats"]["native_info"].update(status_crossover=1, num_var=2, num_constr=1)
    event["stats"]["projection"]["enabled"] = True
    for role in ("load", "crossover", "get"):
        event["stats"].update({role + "_started": 1, role + "_completed": 1, role + "_rc": 0})
    native_success = dict(schema=3, enabled=True, attempted=True, started=1, completed=1, local_accepted=True, adopted=False,
        oracle_decision="not_consulted", norms_ready=True, norm_items=3, bounds_inf=0, cost_inf=0, gate_seconds=0,
        norm_seconds=0, gate_checks=1, last_gate_iteration=1, last_gate_measures=[1e-5, 0, 1e-5], last=event)
    rejected = copy.deepcopy(native_success)
    rejected["local_accepted"] = False
    bad_quality = dict(good, gap_satisfied=False, numerically_optimal=False, objective_gap=1e-4, relative_objective_gap=1e-4)
    rejected["last"].update(status="rejected", crossover_status="rejected", after_quality=bad_quality,
        crossover_quality=bad_quality, unprojected_quality=bad_quality)
    rejected["last"]["basis_export"]["published"] = True
    rejected["last"]["stats"]["projection"].update(scan_started=1, scan_completed=1, scan_items=1)
    warm_stats = dict.fromkeys(COUNTS + TIMES, 0)
    warm_stats.update({key: dict(started=0, completed=0, rc=None, seconds=0) for key in CALLS})
    warm_stats.update(dict.fromkeys(LAST_VALUES))
    warm_stats.update(initial_checks=None, last_checks=None, projection=copy.deepcopy(projection), stage="entry")
    wm = dict(started=0, completed=0, max_steps=64, status="time_limit", raw_quality=None, after_quality=None, stats=warm_stats)
    outer_entry = copy.deepcopy(rejected)
    outer_entry["last"].update(status="time_limit", warm=copy.deepcopy(wm))
    inner_budget = copy.deepcopy(outer_entry)
    inner_budget["last"]["warm"].update(started=1, completed=1)
    inner_budget["last"]["warm"]["stats"]["stage"] = "validation"
    warm_success = copy.deepcopy(rejected)
    warm_success["local_accepted"] = True
    warm_quality = dict(good, objective=-1)
    accepted_warm = copy.deepcopy(wm)
    accepted_warm.update(started=1, completed=1, status="accepted", raw_quality=warm_quality, after_quality=warm_quality)
    ws = accepted_warm["stats"]
    for role in CALLS:
        count = 2 if role in ("solve_n", "solve_t") else 1
        ws[role].update(started=count, completed=count, rc=0)
    checked = dict.fromkeys(CHECK_NUMBERS, 0)
    checked.update(valid=True, primal_feasible=True, basis_dual_feasible=True)
    ws.update(steps_started=1, steps_completed=1, pivots=1, rebuild_started=2, rebuild_completed=2,
        pricing_passes=2, work_items=1, verification_started=1, verification_completed=1, last_entering=0,
        last_leaving=1, last_theta=1, last_pivot=1, initial_checks=checked, last_checks=checked, stage="finished")
    ws["projection"]["enabled"] = True
    warm_success["last"].update(status="accepted", warm=accepted_warm, after_quality=warm_quality)
    final_budget = copy.deepcopy(warm_success)
    final_budget["local_accepted"] = False
    final_budget["last"]["status"] = "time_limit"
    update_partial = copy.deepcopy(inner_budget)
    partial_stats = copy.deepcopy(ws)
    partial_stats.update(steps_completed=0, pivots=0, rebuild_started=1, rebuild_completed=1,
        pricing_passes=1, verification_started=0, verification_completed=0, stage="update", projection=copy.deepcopy(projection))
    for role in ("solve_n", "solve_t"):
        partial_stats[role].update(started=1, completed=1)
    update_partial["last"]["warm"]["stats"] = partial_stats
    native_partial = copy.deepcopy(native_success)
    native_partial.update(completed=0, local_accepted=False)
    native_partial["last"].update(status="exception", crossover_status="exception", unprojected_quality=None,
        crossover_quality=None, after_quality=None, stats=native_stats,
        basis_export=dict(enabled=False, statuses_requested=False, statuses_received=False, published=False, columns=0, rows=0))
    native_budget = copy.deepcopy(native_success)
    native_budget["local_accepted"] = False
    native_budget["last"]["status"] = "time_limit"
    valid = (native_success, outer_entry, inner_budget, warm_success, final_budget, update_partial, native_partial, native_budget)
    for value in valid:
        B.require(valid_crossover(value), "新的真实阶段partial或成功合同不应误拒")
    invalid = []
    for role in ("init", "factor"):
        value = copy.deepcopy(warm_success); value["last"]["warm"]["stats"][role]["rc"] = 1; invalid.append(value)
    for key, wrong in (("adopted", 0), ("local_accepted", 1), ("oracle_decision", "bogus")):
        value = copy.deepcopy(warm_success); value[key] = wrong; invalid.append(value)
    value = copy.deepcopy(warm_success); value["last"]["basis_export"]["published"] = False; invalid.append(value)
    value = copy.deepcopy(warm_success); value["last"]["after_quality"] = value["last"]["crossover_quality"]; invalid.append(value)
    value = copy.deepcopy(warm_success); value["last"]["warm"]["stats"]["steps_completed"] = 0; invalid.append(value)
    for value in invalid:
        B.require(not valid_crossover(value), "错误库rc、类型或最终阶段选择必须拒绝")
    return dict(passed=True, public_lp_runs=0, inherited=inherited, stage_valid_cases=len(valid), stage_invalid_cases=len(invalid))


P.NEW_VERSION = NEW_VERSION
P.contract, P.valid_crossover = contract, valid_crossover
P.provenance, P.self_check = provenance, self_check

if __name__ == "__main__":
    P.main()
