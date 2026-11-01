"""混合HSD/IPX评测薄包装；复用冻结调度、质量、门槛和只读归档。"""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_experimental as C

B = C.build_tool
ENVIRONMENT, MARKER = "SOLVER_V117_CROSSOVER", "V117_CROSSOVER "
OLD_VERSION, NEW_VERSION = "ipm-cpu-0.53.0", "ipm-ipx-crossover-cpu-0.1.0"
ORIGINAL_PROTOCOL, ORIGINAL_SUMMARY = C.protocol, C.performance_summary
TOP_FIELDS = "schema enabled attempted started completed local_accepted adopted oracle_decision norms_ready norm_items bounds_inf cost_inf gate_seconds norm_seconds gate_checks last_gate_iteration last_gate_measures last".split()
LAST_FIELDS = "iteration bounds_inf cost_inf primal_ratio dual_ratio relative_gap deadline_saturated status before_quality after_quality stats".split()
QUALITY_BOOLS = "primal_valid primal_feasible objective_consistent dual_valid dual_feasible gap_satisfied numerically_optimal".split()
QUALITY_NUMBERS = "objective max_row_violation max_bound_violation max_dual_residual objective_gap relative_objective_gap".split()
STAT_COUNTS = "load_started load_completed crossover_started crossover_completed get_started get_completed callback_checks callback_interrupts".split()
STAT_TIMES = "prepare_seconds load_seconds crossover_seconds get_seconds native_scope_seconds verification_seconds total_seconds".split()
PREP_COUNTS = "duplicates_merged rounded_aggregates removed_free_rows free_row_duals_zeroed primal_changes dual_changes".split()
RESIDUAL_FIELDS = "primal dual bounds complementarity dual_sign".split()
INFO_INTS = "status status_ipm status_crossover errflag num_var num_constr num_entries num_rows_solver num_cols_solver num_entries_solver dualized dense_cols centring_tried centring_success dependent_rows dependent_cols rows_inconsistent cols_inconsistent primal_dropped dual_dropped iter kktiter1 kktiter2 basis_repairs updates_start updates_ipm updates_crossover maxvol_updates maxvol_skipped maxvol_passes tbl_nnz".split()
INFO_NUMBERS = "abs_presidual abs_dresidual rel_presidual rel_dresidual pobjval dobjval rel_objgap complementarity normx normy normz objval primal_infeas dual_infeas time_total time_ipm1 time_ipm2 time_starting_basis time_crossover time_kkt_factorize time_kkt_solve time_maxvol time_cr1 time_cr1_AAt time_cr1_pre time_cr2 time_cr2_NNt time_cr2_B time_cr2_Bt ftran_sparse btran_sparse time_ftran time_btran time_lu_invert time_lu_update mean_fill max_fill time_symb_invert tbl_max frobnorm_squared lambdamax volume_increase".split()
STATUSES = "accepted rejected invalid_input numerical_error time_limit native_error unsupported_empty unsupported_integer unsupported_threads exception".split()


def contract(plan):
    value = plan["hybrid_protocol"]
    B.require(set(value) == {"validator_file", "validator_sha256", "environment", "marker", "schema",
        "on_backend_version", "off_backend_version", "marker_on_only"}, "混合协议字段不符")
    B.require(plan["candidate_version"] == 117 and plan["new_algorithm_round"] is True and
        plan["production_changes"] is False and plan["candidate_backend_version"] == NEW_VERSION and
        value["environment"] == ENVIRONMENT and value["marker"] == MARKER and
        type(value["schema"]) is int and value["schema"] == 1 and value["marker_on_only"] is True and
        value["on_backend_version"] == NEW_VERSION and value["off_backend_version"] == OLD_VERSION,
        "只允许测前冻结的混合版本与独立开关")
    path = (ROOT / value["validator_file"]).resolve()
    B.require(path == Path(__file__).resolve() and B.sha(path) == value["validator_sha256"], "混合验证器源码身份变化")
    if "tool_sources" in plan:
        B.verify({str(ROOT / name): digest for name, digest in plan["tool_sources"].items()})
    return value


def fields(value, keys):
    return type(value) is dict and set(value) == set(keys)


def integer(value, maximum=(1 << 64) - 1):
    return type(value) is int and 0 <= value <= maximum


def number(value, nonnegative=False, nullable=False):
    return value is None if nullable and value is None else type(value) in (int, float) and \
        -float("inf") < value < float("inf") and (not nonnegative or value >= 0)


def quality(value):
    return fields(value, QUALITY_BOOLS + QUALITY_NUMBERS) and all(type(value[key]) is bool for key in QUALITY_BOOLS) and \
        all(number(value[key], key.startswith("max_"), True) for key in QUALITY_NUMBERS)


def valid_stats(value):
    keys = STAT_COUNTS + STAT_TIMES + ["load_rc", "crossover_rc", "get_rc", "native_threads", "native_time_limit", "preparation", "native_info"]
    if not fields(value, keys) or not all(integer(value[key]) for key in STAT_COUNTS + ["native_threads"]) or \
        not all(number(value[key], True) for key in STAT_TIMES + ["native_time_limit"]):
        return False
    if value["callback_interrupts"] > value["callback_checks"] or value["native_time_limit"] > 5:
        return False
    for role in ("load", "crossover", "get"):
        started, completed, rc = (value[role + suffix] for suffix in ("_started", "_completed", "_rc"))
        if not 0 <= completed <= started <= 1 or (rc is None) != (completed == 0) or \
            rc is not None and (type(rc) is not int or not -(1 << 31) <= rc < (1 << 31)):
            return False
    if value["crossover_started"] > value["load_completed"] or value["get_started"] > value["crossover_completed"] or \
        value["crossover_started"] and value["load_rc"] != 0 or value["get_started"] and value["crossover_rc"] != 0 or \
        value["load_started"] and (value["native_threads"] != 1 or value["native_time_limit"] <= 0):
        return False
    prep = value["preparation"]
    if not fields(prep, PREP_COUNTS + ["max_aggregation_rounding", "raw", "dropped"]) or \
        not all(integer(prep[key]) for key in PREP_COUNTS) or not number(prep["max_aggregation_rounding"], True, True) or \
        any(not fields(prep[key], RESIDUAL_FIELDS) or not all(number(item, True, True) for item in prep[key].values()) for key in ("raw", "dropped")):
        return False
    info = value["native_info"]
    if info is not None and (not fields(info, INFO_INTS + INFO_NUMBERS) or
        not all(type(info[key]) is int and -(1 << 31) <= info[key] < (1 << 31) for key in INFO_INTS) or
        not all(number(info[key], key.startswith("time_"), True) for key in INFO_NUMBERS) or
        any(info[key] != 0 for key in ("status_ipm", "iter", "kktiter1", "kktiter2", "updates_ipm"))):
        return False
    return (value["load_completed"] == 0 or info is not None) and (value["get_started"] == 0 or info["status_crossover"] in (1, 2))


def valid_crossover(value):
    if not fields(value, TOP_FIELDS) or type(value["schema"]) is not int or value["schema"] != 1 or value["enabled"] is not True or \
        not all(type(value[key]) is bool for key in ("attempted", "local_accepted", "adopted", "norms_ready")) or \
        not all(integer(value[key]) for key in ("started", "completed", "norm_items", "gate_checks")) or \
        not all(number(value[key], True) for key in ("bounds_inf", "cost_inf", "gate_seconds", "norm_seconds")) or \
        value["oracle_decision"] not in ("not_consulted", "not_active", "accept", "reject", "budget", "error"):
        return False
    if not 0 <= value["completed"] <= value["started"] <= 1 or value["attempted"] != (value["started"] == 1):
        return False
    iteration, measures = value["last_gate_iteration"], value["last_gate_measures"]
    if (iteration is None) != (value["gate_checks"] == 0) or iteration is not None and (not integer(iteration) or iteration < 1) or \
        measures is not None and (type(measures) is not list or len(measures) != 3 or not all(number(item, True) for item in measures)):
        return False
    if value["gate_checks"] == 0 and (measures is not None or value["norms_ready"] or value["norm_items"] != 0 or
        value["gate_seconds"] != 0 or value["norm_seconds"] != 0):
        return False
    if not value["norms_ready"] and (value["bounds_inf"] != 0 or value["cost_inf"] != 0) or \
        value["local_accepted"] and not value["completed"] or value["adopted"] and (not value["local_accepted"] or value["oracle_decision"] not in ("accept", "not_active")) or \
        value["oracle_decision"] != "not_consulted" and not value["local_accepted"]:
        return False
    last = value["last"]
    if not value["started"]:
        return last is None and not value["local_accepted"] and not value["adopted"] and value["oracle_decision"] == "not_consulted"
    if not fields(last, LAST_FIELDS) or not value["norms_ready"] or iteration != last["iteration"] or \
        measures is None or not integer(last["iteration"]) or last["iteration"] < 1 or \
        not all(number(last[key], True) for key in ("bounds_inf", "cost_inf", "primal_ratio", "dual_ratio", "relative_gap")) or \
        any(last[key] > 1e-4 for key in ("primal_ratio", "dual_ratio", "relative_gap")) or \
        measures != [last[key] for key in ("primal_ratio", "dual_ratio", "relative_gap")] or \
        any(last[key] != value[key] for key in ("bounds_inf", "cost_inf")) or type(last["deadline_saturated"]) is not bool or \
        last["status"] not in STATUSES or not quality(last["before_quality"]) or \
        last["after_quality"] is not None and not quality(last["after_quality"]) or not valid_stats(last["stats"]):
        return False
    before, after, stats = last["before_quality"], last["after_quality"], last["stats"]
    if not all(before[key] for key in ("primal_valid", "objective_consistent", "dual_valid", "dual_feasible")) or \
        before["objective"] is None or before["relative_objective_gap"] is None or before["numerically_optimal"] or \
        any(before[key] is None for key in ("max_row_violation", "max_bound_violation", "max_dual_residual")) or \
        value["local_accepted"] != (value["completed"] == 1 and last["status"] == "accepted"):
        return False
    if value["completed"] == 0:
        return last["status"] == "exception" and after is None and value["oracle_decision"] == "not_consulted"
    return not value["local_accepted"] or after is not None and all(after[key] for key in QUALITY_BOOLS) and after["objective"] is not None and \
        all(stats[role + "_completed"] == 1 and stats[role + "_rc"] == 0 for role in ("load", "crossover", "get"))


def inspect_crossover(stderr, enabled):
    lines = [line for line in stderr.splitlines() if line.startswith(MARKER)]
    if not enabled:
        return None, "unexpected_marker" if lines else None
    if len(lines) != 1:
        return None, "marker_count"
    try:
        value = B.parse(lines[0][len(MARKER):])
    except (TypeError, ValueError, RuntimeError):
        return None, "invalid_marker_json"
    return value, None if valid_crossover(value) else "invalid_marker_schema"


def common_protocol(plan):
    contract(plan)
    common = copy.deepcopy(plan)
    common["candidate_backend_version"] = OLD_VERSION
    return ORIGINAL_PROTOCOL(common)


def protocol(plan):
    runner, frozen = common_protocol(plan)
    old_config, old_valid, old_quality = runner.configuration_matches, runner.valid_process, runner.quality_fields
    old_environment, old_run_one = runner.solver_environment, runner.benchmark.run_one

    def configuration_matches(report, stage, variant, model, options):
        if not isinstance(report, dict):
            return False
        expected = NEW_VERSION if variant == "diagnostic_on" else "1.15.1" if variant == "highs" else OLD_VERSION
        if report.get("backend_version") != expected:
            return False
        probe = dict(report)
        if variant == "diagnostic_on":
            probe["backend_version"] = OLD_VERSION
        return old_config(probe, stage, variant, model, options)

    @contextlib.contextmanager
    def solver_environment(enabled):
        with old_environment(enabled):
            if enabled:
                os.environ[ENVIRONMENT] = "1"
            yield

    def run_one(*args, **kwargs):
        actual_environment = {ENVIRONMENT: os.environ[ENVIRONMENT]} if ENVIRONMENT in os.environ else {}
        row = old_run_one(*args, **kwargs)
        row["crossover_environment"] = actual_environment
        return row

    def valid_process(row):
        if not old_valid(row):
            return False
        enabled = row.get("requested_variant") == "diagnostic_on"
        if row.get("crossover_environment") != ({ENVIRONMENT: "1"} if enabled else {}):
            return False
        value, error = inspect_crossover(row["stderr"], enabled)
        return error is None and row.get("crossover_marker_error") is None and B.equal(row.get("crossover_marker"), value)

    def quality_fields(row, model, expected_solver, options):
        value, error = inspect_crossover(row["stderr"], row.get("requested_variant") == "diagnostic_on")
        probe = dict(row, crossover_marker=value, crossover_marker_error=error)
        return dict(old_quality(probe, model, expected_solver, options), crossover_marker=value, crossover_marker_error=error)

    runner.configuration_matches, runner.solver_environment = configuration_matches, solver_environment
    # 原benchmark模块由多层冻结运行器共享，只包装本次返回模块的视图。
    runner.benchmark = SimpleNamespace(**vars(runner.benchmark))
    runner.benchmark.run_one = run_one
    runner.valid_process, runner.quality_fields = valid_process, quality_fields
    return runner, frozen


def performance_summary(stage, rows, models, repeats, variants, runner, frozen, plan_sha, raw_sha):
    value = ORIGINAL_SUMMARY(stage, rows, models, repeats, variants, runner, frozen, plan_sha, raw_sha)
    value.update(crossover_marker_error_count=sum(row["crossover_marker_error"] is not None for row in rows),
        crossover_marker_runs=sum(row["crossover_marker"] is not None for row in rows))
    return value


def provenance(output, plan, create):
    target = output / "provenance" / "v117-benchmark.py"
    expected = plan["hybrid_protocol"]["validator_sha256"]
    if not target.exists() and create:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
    B.require(target.is_file() and B.sha(target) == expected, "逐字评测包装快照缺失或变化")


def self_check(plan_path):
    current = C.protocol
    C.protocol = common_protocol
    try:
        inherited = C.self_check(plan_path)
    finally:
        C.protocol = current
    # 这里只构造序列化合同，不运行HSD、IPX或公开LP。
    empty = dict(schema=1, enabled=True, attempted=False, started=0, completed=0, local_accepted=False, adopted=False,
        oracle_decision="not_consulted", norms_ready=False, norm_items=0, bounds_inf=0, cost_inf=0, gate_seconds=0,
        norm_seconds=0, gate_checks=0, last_gate_iteration=None, last_gate_measures=None, last=None)
    prepared = dict.fromkeys(PREP_COUNTS, 0)
    prepared.update(max_aggregation_rounding=0, raw=dict.fromkeys(RESIDUAL_FIELDS, 0), dropped=dict.fromkeys(RESIDUAL_FIELDS, 0))
    stats = dict.fromkeys(STAT_COUNTS + STAT_TIMES + ["native_threads", "native_time_limit"], 0)
    stats.update(load_rc=None, crossover_rc=None, get_rc=None, preparation=prepared, native_info=None)
    before = dict.fromkeys(QUALITY_BOOLS, False)
    before.update(dict.fromkeys(QUALITY_NUMBERS, 0))
    before.update(primal_valid=True, objective_consistent=True, dual_valid=True, dual_feasible=True)
    partial = dict(empty, attempted=True, started=1, norms_ready=True, norm_items=2, gate_checks=1,
        last_gate_iteration=1, last_gate_measures=[0, 0, 0], last=dict(iteration=1, bounds_inf=0, cost_inf=0,
        primal_ratio=0, dual_ratio=0, relative_gap=0, deadline_saturated=False, status="exception", before_quality=before,
        after_quality=None, stats=stats))
    returned = copy.deepcopy(partial)
    returned.update(completed=1, local_accepted=True)
    returned["last"].update(status="accepted", after_quality=dict(before, **dict.fromkeys(QUALITY_BOOLS, True)))
    returned["last"]["stats"].update(native_threads=1, native_time_limit=4,
        native_info=dict.fromkeys(INFO_INTS + INFO_NUMBERS, 0))
    returned["last"]["stats"]["native_info"].update(status_crossover=1, basis_repairs=-1)
    for role in ("load", "crossover", "get"):
        returned["last"]["stats"].update({role + "_started": 1, role + "_completed": 1, role + "_rc": 0})
    for value in (empty, partial, returned):
        B.require(inspect_crossover(MARKER + json.dumps(value), True)[1] is None, "真实允许的未触发/partial/未采用返回必须合法")
    B.require(inspect_crossover("", True)[1] == "marker_count", "开启时漏标记必须拒绝")
    B.require(inspect_crossover(MARKER + "{}\n" + MARKER + "{}", True)[1] == "marker_count", "重复标记必须拒绝")
    B.require(inspect_crossover(MARKER + "{}", False)[1] == "unexpected_marker", "关闭侧不得出现新标记")
    B.require(inspect_crossover(MARKER + '{"schema":1,"schema":1}', True)[1] == "invalid_marker_json", "重复JSON键必须拒绝")
    invalid = [dict(empty, started=True), dict(partial, adopted=True), dict(empty, unexpected=0)]
    bad_native = copy.deepcopy(returned); bad_native["last"]["stats"]["native_info"]["iter"] = 1; invalid.append(bad_native)
    for value in invalid:
        B.require(inspect_crossover(MARKER + json.dumps(value), True)[1] == "invalid_marker_schema", "非法类型/提交/原生IPM工作必须拒绝")
    runner, frozen = protocol(B.read(plan_path))
    model = runner.specification("development", frozen)["models"][0]
    report = dict(input=model["input"], backend="ipm", backend_version=NEW_VERSION,
        options=runner.base.expected_options("development", "candidate", frozen))
    B.require(runner.configuration_matches(report, "development", "diagnostic_on", model, frozen) and
        not runner.configuration_matches(dict(report, backend_version=OLD_VERSION), "development", "diagnostic_on", model, frozen),
        "混合开启侧必须显式使用新版本身份")
    return dict(passed=True, public_lp_runs=0, inherited=inherited, crossover_valid_cases=3, crossover_marker_cases=4,
        crossover_invalid_cases=len(invalid), configuration_cases=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", required=True, choices=(*C.PERFORMANCE_STAGES, "self-check", "compatibility", "gate"))
    parser.add_argument("--archive-only", action="store_true", help="只重解析原始记录，不启动新LP")
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve()
    B.require(plan_path.is_relative_to(ROOT), "正式计划必须位于工作区")
    plan = B.read(plan_path)
    plan_sha = B.sha(plan_path)
    contract(plan)
    C.protocol, C.performance_summary = protocol, performance_summary
    try:
        if args.stage == "self-check":
            B.require(not args.archive_only, "self-check本身不启动LP")
            result = self_check(plan_path)
        else:
            B.require(args.output is not None, "阶段需要--output证据目录")
            output = (ROOT / args.output).resolve()
            B.require(output.is_relative_to(ROOT), "证据目录必须位于工作区")
            read_only = args.archive_only or args.stage in ("compatibility", "gate")
            provenance(output, plan, not read_only)
            if args.stage in ("compatibility", "gate"):
                result = C.compatibility(output, plan_path) if args.stage == "compatibility" else C.gate(output, plan_path)
            elif args.archive_only:
                _, result, _ = C.archive(args.stage, output, plan_path)
            else:
                B.require(args.work is not None, "新LP必须指定--work独立构建目录")
                work = (ROOT / args.work).resolve()
                result = C.run_performance(work, output, args.stage, plan_path)
            provenance(output, plan, False)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        passed = result.get("passed", True) and result.get("crossover_marker_error_count", 0) == 0
        if "valid_process_count" in result:
            passed = passed and result["valid_process_count"] == result["run_count"]
    finally:
        B.require(B.sha(plan_path) == plan_sha, "执行期间正式计划身份变化")
        contract(plan)
        C.protocol, C.performance_summary = ORIGINAL_PROTOCOL, ORIGINAL_SUMMARY
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
