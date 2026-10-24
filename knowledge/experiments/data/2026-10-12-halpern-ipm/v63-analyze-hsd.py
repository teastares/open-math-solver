"""只读定位 v63 宽常数恢复的实际搜索阶段，并精确复算最后候选。"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import struct
import tempfile
import time
from fractions import Fraction
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
HELPER_HASHES = {
    "v61-analyze-hsd.py": "160aa12f65cb8eb4ab9e7d595df4a072123a83d6465aa38025abd9ffd3d41a24",
    "v53-analyze-hsd.py": "d5a2203c6ec8799f8cb15723c29635fce06aa1495b4c1482d6f70614d4b9aace",
}
for _name, _expected in HELPER_HASHES.items():
    if hashlib.sha256((DIRECTORY / _name).read_bytes()).hexdigest() != _expected:
        raise ValueError(f"锁定的精确辅助来源改变：{_name}")
SPEC = importlib.util.spec_from_file_location("v63_locked_hsd_helpers", DIRECTORY / "v61-analyze-hsd.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
ZERO = Fraction(0)
CAP, RESERVE, MAX_EVENTS, MAX_STEPS = 128 * 1024 * 1024, 64 * 1024, 8, 256
MAGIC = b"V63HSD1\0"
COUNT_NAMES = BASE.COUNT_NAMES + ("scalar_checks", "scalar_rejections")
integer, boolean = BASE.integer, BASE.boolean
packed, exact_vector, vector_summary = BASE.packed, BASE.exact_vector, BASE.vector_summary


def finite_hex(value):
    return BASE.hex_fraction(value)


def scalar_value(value):
    """允许捕获到的非有限标量，但不将其强转为可参与精确算术的数。"""
    if value in ("nan", "inf", "-inf"):
        return {"kind": value, "finite": False, "exact": None}
    return {"kind": "finite", "finite": True, "exact": packed(finite_hex(value))}


def validate_counts(counts):
    if not isinstance(counts, dict) or set(counts) != set(COUNT_NAMES):
        raise ValueError("v63 工作计数字段缺失或新增")
    for name in COUNT_NAMES:
        integer(counts[name])
    for successes, attempts in (("constant_repair_successes", "constant_repairs"),
                               ("extended_krylov_successes", "extended_krylov_attempts"),
                               ("scalar_rejections", "scalar_checks")):
        if counts[successes] > counts[attempts]:
            raise ValueError("v63 成功或拒绝计数超过尝试计数")


def compatibility_event(event):
    """仅投影已声明的 v61 公共字段，保留其输入检查与精确数学合同。"""
    value = dict(event)
    value["wide"] = {name: event["wide"][name] for name in ("u", "before_v", "after_v")}
    for phase in ("before_counts", "after_counts"):
        value[phase] = {name: event[phase][name] for name in BASE.COUNT_NAMES}
    return value


def residual_classification(global_pass, blocks_pass, scalar_available, scalar_accepted):
    if not global_pass and not blocks_pass:
        return "global_and_block_residual_failed"
    if not global_pass:
        return "global_residual_failed"
    if not blocks_pass:
        return "block_residual_failed"
    if not scalar_available:
        return "residual_passed_scalar_result_unavailable"
    return "residual_and_scalar_passed" if scalar_accepted else "residual_passed_scalar_rejected"


def summarize_step(step):
    if not isinstance(step.get("block_tolerances"), list):
        raise ValueError("v63 宽常数恢复缺少严格分块门槛")
    norm = finite_hex(step["global_norm"])
    block_norms = [finite_hex(x) for x in step["block_norms"]]
    tolerance = finite_hex(step["global_tolerance"])
    block_tolerances = [finite_hex(x) for x in step["block_tolerances"]]
    if len(block_norms) != 2 or len(block_tolerances) != 2 or norm < 0 or \
            any(x < 0 for x in block_norms) or tolerance <= 0 or any(x <= 0 for x in block_tolerances):
        raise ValueError("v63 残差范数或门槛无效")
    if norm != max(block_norms):
        raise ValueError("v63 全局范数与分块范数不一致")
    global_pass, block_pass = norm <= tolerance, all(a <= b for a, b in zip(block_norms, block_tolerances))
    values_available = boolean(step["scalar_values_available"])
    returned = boolean(step["scalar_return_available"])
    accepted = boolean(step["scalar_accepted"]) if returned else None
    if not returned and step["scalar_accepted"] is not None:
        raise ValueError("v63 应用谓词未返回却保存接受结果")
    accounted = boolean(step["scalar_accounted"])
    if (values_available or returned or accounted) and not (global_pass and block_pass):
        raise ValueError("v63 原残差未达标却执行分母检查")
    if returned and not values_available or accepted and not returned or accounted and not returned:
        raise ValueError("v63 分母值、返回值与记账阶段不一致")
    raw = None
    if values_available:
        raw = {name: scalar_value(step[name]) for name in ("dot_v", "denominator")}
        if returned:
            expected = raw["denominator"]["finite"] and finite_hex(step["denominator"]) > 0
            if accepted != expected:
                raise ValueError("v63 实际分母与应用谓词返回不一致")
    elif step["dot_v"] is not None or step["denominator"] is not None:
        raise ValueError("v63 应用谓词未计算却保存分母")
    result = {key: step[key] for key in ("kind", "ir_iteration", "restart", "column")}
    result.update(global_ratio=packed(norm / tolerance),
                  block_ratios=[packed(a / b) for a, b in zip(block_norms, block_tolerances)],
                  global_passed=global_pass, blocks_passed=block_pass,
                  scalar_values=raw, scalar_return_available=returned,
                  scalar_accepted=accepted if returned else None, scalar_accounted=accounted,
                  classification=residual_classification(global_pass, block_pass, returned, accepted))
    return result


def validate_search(event, dimension):
    if type(event.get("step_capacity")) is not int or event["step_capacity"] != MAX_STEPS:
        raise ValueError("v63 阶段记录槽数不符")
    steps = event.get("small_events")
    if not isinstance(steps, list) or len(steps) > MAX_STEPS or integer(event.get("step_count")) != len(steps):
        raise ValueError("v63 阶段记录数不符")
    total, dropped = integer(event.get("step_total")), integer(event.get("step_dropped"))
    if total != len(steps) + dropped:
        raise ValueError("v63 阶段记录与丢弃账本不符")
    available = boolean(event.get("last_linear_available"))
    rhs_available = boolean(event.get("linear_rhs_available"))
    last = integer(event.get("last_candidate_step_index"), -1)
    if available != (total > 0) or (available and (not rhs_available or last != total - 1)) or (not available and last != -1):
        raise ValueError("v63 最后候选索引或可用状态不符")
    for key, value in (("last_linear_candidate", event["wide"].get("last_linear_candidate")),
                       ("last_linear_residual", event.get("last_linear_residual")),
                       ("linear_rhs", event.get("linear_rhs"))):
        expected = rhs_available if key == "linear_rhs" else available
        if expected:
            if not isinstance(value, list) or len(value) != dimension:
                raise ValueError("v63 最后线性工作数组尺寸不符")
            for x in value:
                finite_hex(x)
        elif value is not None:
            raise ValueError("v63 未发生的线性工作却附有候选数组")
    if event.get("linear_exit") not in {"not_started", "searching", "ir_return", "krylov_return",
            "krylov_exhausted", "invalid_initial_residual", "degenerate_subspace", "exception", "exception_after_acceptance"}:
        raise ValueError("v63 线性退出阶段未知")
    position = event.get("linear_position")
    if not isinstance(position, dict) or set(position) != {"ir_iteration", "restart", "column"}:
        raise ValueError("v63 最后线性位置缺字段")
    for key, maximum in (("ir_iteration", 2), ("restart", 2), ("column", min(32, dimension))):
        if integer(position[key], -1) > maximum:
            raise ValueError("v63 最后线性位置越过原预算")
    seen_krylov = False
    ir_positions, previous_position = [], None
    for step in steps:
        if step.get("kind") == "ir_candidate":
            if seen_krylov or integer(step.get("restart"), -1) != -1 or integer(step.get("column"), -1) != -1:
                raise ValueError("v63 IR 阶段顺序或位置不符")
            iteration = integer(step.get("ir_iteration"))
            if iteration != len(ir_positions) or iteration > 2:
                raise ValueError("v63 IR 候选不是原0、1、2顺序")
            ir_positions.append(iteration)
        elif step.get("kind") == "krylov_candidate":
            seen_krylov = True
            restart, column = integer(step.get("restart")), integer(step.get("column"), 1)
            if integer(step.get("ir_iteration"), -1) != -1 or restart >= 3 or column > min(32, dimension):
                raise ValueError("v63 Krylov 候选越过原3×32预算")
            if previous_position is not None and (restart, column) <= previous_position:
                raise ValueError("v63 Krylov 候选位置没有前进")
            previous_position = restart, column
        else:
            raise ValueError("v63 候选阶段非法")
        summarize_step(step)
    cycles = event.get("cycles")
    if not isinstance(cycles, list) or len(cycles) != 3:
        raise ValueError("v63 Krylov 周期数不符")
    for i, cycle in enumerate(cycles):
        started, ended = boolean(cycle["started"]), boolean(cycle["ended"])
        starts, completed, zeroes = (integer(cycle[key]) for key in
                                   ("columns_started", "columns_completed", "remaining_zero_count"))
        if not started and (ended or starts or completed or zeroes) or \
                starts > min(32, dimension) or completed > starts or zeroes > completed:
            raise ValueError("v63 周期实际列数或退化次数不符")
        if started and i and not cycles[i - 1]["ended"]:
            raise ValueError("v63 后一周期启动但前一周期没有结束")
        if any(step["restart"] == i and step["column"] > completed for step in steps if step["kind"] == "krylov_candidate"):
            raise ValueError("v63 记录的候选超出实际完成列数")


def load_snapshot(path):
    path = Path(path)
    if path.stat().st_size > 8 * CAP:
        raise ValueError("v63 JSON 超过受控解析上限")
    meta = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=BASE.unique_object,
                      parse_constant=BASE.reject_constant)
    if not isinstance(meta, dict) or meta.get("schema") != "v63_hsd_1":
        raise ValueError("v63 捕获 schema 不符")
    expected_limits = {"max_events": MAX_EVENTS, "payload_limit_bytes": CAP,
                       "metadata_reserve_bytes": RESERVE, "max_steps_per_event": MAX_STEPS}
    if meta.get("limits") != expected_limits or any(type(meta["limits"][key]) is not int for key in expected_limits):
        raise ValueError("v63 捕获容量合同不符")
    for key in ("extra_lp_products", "extra_lp_solves"):
        if type(meta.get(key)) is not int or meta[key] != 0:
            raise ValueError("v63 捕获增加了在线乘积或求解")
    boolean(meta.get("enabled"))
    if not isinstance(meta.get("status"), str):
        raise ValueError("v63 捕获状态类型无效")
    digits, width = integer(meta.get("long_double_digits"), 54), integer(meta.get("long_double_bytes"), 8)
    step_bytes = integer(meta.get("step_bytes"), 1)
    if digits > 256 or width > 32 or step_bytes > 4096:
        raise ValueError("v63 标量或阶段记录尺寸超出分析合同")
    events = meta.get("events")
    if not isinstance(events, list) or len(events) > MAX_EVENTS or integer(meta.get("event_count")) != len(events):
        raise ValueError("v63 repair 事件数不符")
    triggers, dropped = integer(meta.get("trigger_count")), integer(meta.get("dropped_events"))
    if triggers != len(events) + dropped or not meta["enabled"] and triggers:
        raise ValueError("v63 repair 触发与丢弃账本不符")
    payload, cache = integer(meta.get("payload_bytes")), integer(meta.get("cache_bytes"))
    if cache < payload or cache > CAP - RESERVE:
        raise ValueError("v63 捕获缓存账本少计或超限")
    binary_path = path.with_suffix(".bin")
    if binary_path.stat().st_size > CAP + 8:
        raise ValueError("v63 二进制超过受控解析上限")
    binary = binary_path.read_bytes()
    if binary[:8] != MAGIC or len(binary) != payload + 8:
        raise ValueError("v63 二进制魔数或长度不符")
    cursor, loaded, allocated = 8, [], 0
    previous_ordinal = 0
    for index, event in enumerate(events):
        if integer(event.get("index")) != index:
            raise ValueError("v63 repair 事件索引不是连续顺序")
        ordinal = integer(event.get("repair_ordinal"), 1)
        if ordinal <= previous_ordinal or ordinal > triggers:
            raise ValueError("v63 repair 序号不单调或超出触发数")
        previous_ordinal = ordinal
        integer(event.get("iteration")); integer(event.get("regularization_level"), -1)
        if event.get("phase") not in ("affine", "corrector") or not isinstance(event.get("failure"), str):
            raise ValueError("v63 外层方向阶段或异常字段无效")
        for key in ("committed", "candidate_available", "after_scalar_available"):
            boolean(event.get(key))
        if event["after_scalar_available"] and not event["candidate_available"] or event["committed"] and not event["after_scalar_available"]:
            raise ValueError("v63 返回、标量验算与事务提交关系错误")
        for key in ("before_counts", "after_counts"):
            validate_counts(event.get(key))
        if any(event["after_counts"][key] < event["before_counts"][key] for key in COUNT_NAMES):
            raise ValueError("v63 repair 工作计数倒退")
        descriptors = event.get("arrays")
        if not isinstance(descriptors, list) or tuple(item.get("name") for item in descriptors) != BASE.NAMES:
            raise ValueError("v63 binary 数组名或顺序不符")
        arrays = {}
        for field in descriptors:
            name, dtype = field["name"], field.get("dtype")
            count, offset = integer(field.get("count")), integer(field.get("offset"), 8)
            expected = "i64_le" if name in BASE.INT_NAMES else "u8" if name == "inequality" else "f64_le"
            if dtype != expected or offset != cursor:
                raise ValueError("v63 binary 数组格式或连续布局错误")
            code, item_width = BASE.FORMATS[dtype]
            end = cursor + count * item_width
            if end > len(binary):
                raise ValueError("v63 binary 数组越界")
            arrays[name] = list(struct.unpack_from(f"<{count}{code}", binary, cursor))
            if dtype == "f64_le" and any(not math.isfinite(x) for x in arrays[name]):
                raise ValueError("v63 binary 数组含非有限数")
            cursor = end
        if not isinstance(event.get("wide"), dict) or set(event["wide"]) != {"u", "before_v", "after_v", "last_linear_candidate"}:
            raise ValueError("v63 宽方向字段缺失或新增")
        wide, scalars = BASE.validate_event(compatibility_event(event), arrays)
        dimension = len(arrays["costs"]) + len(arrays["rhs"])
        validate_search(event, dimension)
        allocated += dimension * (4 * width + 16) + MAX_STEPS * step_bytes
        loaded.append((event, arrays, wide, scalars))
    if cursor != len(binary) or cache != payload + allocated:
        raise ValueError("v63 binary 尾部或宽方向/阶段缓存账本不符")
    return meta, loaded, binary_path


def exact_direction(arrays, f, point, scalars, raw=None):
    """复用锁定的稀疏精确乘积；本轮不再重算 H 与 s/z 的额外总界。"""
    n, m = len(f["costs"]), len(f["rhs"])
    rhs = [-x for x in f["costs"]] + f["rhs"]
    upper, lower = BASE.residual_vectors(arrays, f, point, rhs)
    dot_v = sum((a * b for a, b in zip(f["costs"] + f["rhs"], point)), ZERO)
    ratio = scalars["kappa"] / scalars["tau"]
    energy = sum((f["diagonal_h"][i] * point[n + i] ** 2 for i in range(m)), ZERO)
    upper_projection = sum((a * b for a, b in zip(point[:n], upper)), ZERO)
    lower_projection = -sum((a * b for a, b in zip(point[n:], lower)), ZERO)
    defect = upper_projection + lower_projection
    positive_energy, denominator = ratio + energy, ratio - dot_v
    error = denominator - (positive_energy - defect)
    if error:
        raise AssertionError("v63 D=E−χ 精确恒等式失败")
    values = dict(ratio=ratio, dot_v=dot_v, denominator=denominator, energy=energy,
                  positive_energy=positive_energy, defect=defect, upper_projection=upper_projection,
                  lower_projection=lower_projection, identity_error=error,
                  defect_over_positive_energy=defect / positive_energy,
                  denominator_over_positive_energy=denominator / positive_energy)
    classification = "actual_scalar_unavailable"
    if raw is not None:
        actual_dot, actual_denominator = finite_hex(raw["dot_v"]), finite_hex(raw["denominator"])
        dot_error = actual_dot - dot_v
        remainder = actual_denominator - (ratio - actual_dot)
        total = actual_denominator - denominator
        if total != remainder - dot_error:
            raise AssertionError("v63 标量求值误差分解失败")
        values.update(raw_dot_v=actual_dot, raw_denominator=actual_denominator,
                      dot_v_rounding=dot_error, ratio_and_subtraction_rounding=remainder,
                      total_scalar_rounding=total)
        classification = "positive_actual_and_exact_denominator" if actual_denominator > 0 and denominator > 0 else \
            "nonpositive_actual_and_exact_denominator" if actual_denominator <= 0 and denominator <= 0 else \
            "actual_and_exact_denominator_signs_differ"
    return {"identity_exact": True, "classification": classification,
            "scalars": {name: packed(value) for name, value in values.items()},
            "residual": BASE.residual_summary(upper, lower, BASE.recomputed_tolerances(
                [-x for x in arrays["costs"]] + arrays["rhs"], n)),
            "largest_signed_projection_terms": BASE.largest_projection_terms(upper, lower, point)}


def last_linear_analysis(event, arrays, wide, scalars):
    if not event["last_linear_available"]:
        return None
    n, m = len(arrays["costs"]), len(arrays["rhs"])
    point = [finite_hex(x) for x in event["wide"]["last_linear_candidate"]]
    stored = [finite_hex(x) for x in event["last_linear_residual"]]
    linear_rhs = [finite_hex(x) for x in event["linear_rhs"]]
    if len(point) != n + m or len(stored) != n + m or len(linear_rhs) != n + m:
        raise ValueError("v63 最后线性候选、残差或右端维数错误")
    expected_rhs = [-Fraction.from_float(x) for x in arrays["costs"]] + exact_vector(arrays["rhs"])
    if linear_rhs != expected_rhs:
        raise ValueError("v63 最后线性求解不是捕获工作点的常数右端")
    for value in stored + linear_rhs:
        if Fraction.from_float(float(value)) != value:
            raise ValueError("v63 生产残差或右端不是精确提升的 binary64")
    f = {name: exact_vector(arrays[name]) for name in ("values", "costs", "rhs", "diagonal_h", "direction_rhs")}
    upper, lower = BASE.residual_vectors(arrays, f, point, linear_rhs)
    exact_residual = [-x for x in upper + lower]
    error = [observed - actual for observed, actual in zip(stored, exact_residual)]
    tolerances = BASE.recomputed_tolerances([float(x) for x in linear_rhs], n)
    step_index = event["last_candidate_step_index"]
    step = event["small_events"][step_index] if 0 <= step_index < len(event["small_events"]) else None
    raw = None
    if step is not None and step["scalar_values_available"] and \
            scalar_value(step["dot_v"])["finite"] and scalar_value(step["denominator"])["finite"]:
        raw = {"dot_u": event["before"]["dot_u"], "dot_v": step["dot_v"], "denominator": step["denominator"]}
    direction = exact_direction(arrays, f, point, scalars, raw)
    saved_summary = BASE.residual_summary(stored[:n], stored[n:], tolerances)
    selected = sorted(range(n + m), key=lambda i: abs(error[i]), reverse=True)[:10]
    result = {"role": "最后完成真实残差验算的线性候选；未据此声明repair返回或事务提交。",
              "production_residual_definition": "rhs-K0*v；精确能量恒等式沿用r=K0*v-rhs，比较时已翻转符号。",
              "candidate_step_index": step_index, "direction": direction,
              "production_residual": saved_summary,
              "residual_evaluation_error": {"whole": vector_summary(error),
                  "upper": vector_summary(error[:n]), "lower": vector_summary(error[n:]),
                  "largest_entries": [{"index": i, "block": "upper" if i < n else "lower",
                                       "production": packed(stored[i]), "exact": packed(exact_residual[i]),
                                       "error": packed(error[i])} for i in selected]},
              "last_step": summarize_step(step) if step is not None else None,
              "actual_scalar_evaluated": bool(step is not None and step["scalar_values_available"]),
              "repair_candidate_available": event["candidate_available"],
              "committed": event["committed"]}
    if step is not None:
        actual_tolerances = [finite_hex(step["global_tolerance"])] + [finite_hex(x) for x in step["block_tolerances"]]
        if actual_tolerances != list(tolerances):
            raise ValueError("v63 捕获门槛与原 binary64 公式不一致")
        expected_norms = [max(map(abs, stored), default=ZERO), max(map(abs, stored[:n]), default=ZERO),
                          max(map(abs, stored[n:]), default=ZERO)]
        actual_norms = [finite_hex(step["global_norm"])] + [finite_hex(x) for x in step["block_norms"]]
        if expected_norms != actual_norms:
            raise ValueError("v63 最后生产残差向量与捕获范数不一致")
        result["captured_norms_and_thresholds_match"] = True
    return result


def analyze_event(event, arrays, wide, scalars):
    f = {name: exact_vector(arrays[name]) for name in ("values", "costs", "rhs", "diagonal_h", "direction_rhs")}
    result = {key: event[key] for key in ("index", "iteration", "phase", "regularization_level",
              "committed", "candidate_available", "after_scalar_available", "failure")}
    result["dimensions"] = {"n": len(arrays["costs"]), "m": len(arrays["rhs"]), "nnz": len(arrays["values"])}
    result["before"] = exact_direction(arrays, f, wide["before_v"], scalars, event["before"])
    result["after"] = exact_direction(arrays, f, wide["after_v"], scalars, event["after"]) if wide["after_v"] is not None else None
    result["before_counts"] = event["before_counts"]
    result["after_counts"] = event["after_counts"]
    delta = {key: event["after_counts"][key] - event["before_counts"][key] for key in COUNT_NAMES}
    result["work_delta"] = delta
    steps = [summarize_step(step) for step in event["small_events"]]
    ir_completed = max((step["ir_iteration"] for step in steps if step["kind"] == "ir_candidate"), default=None)
    columns_started = sum(cycle["columns_started"] for cycle in event["cycles"])
    columns_completed = sum(cycle["columns_completed"] for cycle in event["cycles"])
    expected_refinements = None if ir_completed is None else ir_completed + columns_completed
    complete_accounting = event["step_dropped"] == 0 and event["linear_exit"] in (
        "ir_return", "krylov_return", "krylov_exhausted", "invalid_initial_residual")
    counted_checks = sum(step["scalar_return_available"] for step in steps)
    counted_rejections = sum(step["scalar_accounted"] and step["scalar_accepted"] is False for step in steps)
    counted_values = sum(step["scalar_values"] is not None for step in steps)
    if counted_values > delta["scalar_checks"] or counted_checks > delta["scalar_checks"] or \
            counted_rejections > delta["scalar_rejections"]:
        raise ValueError("v63 捕获阶段检查数超过生产计数")
    if complete_accounting and (expected_refinements != delta["refinements"] or
            counted_checks != delta["scalar_checks"] or counted_rejections != delta["scalar_rejections"]):
        raise ValueError("v63 完整搜索阶段与真实工作计数不闭合")
    last = last_linear_analysis(event, arrays, wide, scalars)
    result.update(repair_ordinal=event["repair_ordinal"], linear_exit=event["linear_exit"],
                  linear_position=event["linear_position"], stage_records=steps,
                  stage_record_count=event["step_count"], stage_record_total=event["step_total"],
                  stage_record_dropped=event["step_dropped"], cycles=event["cycles"], last_linear=last,
                  work_accounting={"ir_corrections_before_last_recorded_ir_candidate": ir_completed,
                      "arnoldi_columns_started": columns_started, "arnoldi_columns_completed": columns_completed,
                      "sum_recorded_ir_and_completed_columns": expected_refinements,
                      "actual_refinement_delta": delta["refinements"],
                      "complete_search_accounting": complete_accounting,
                      "refinement_delta_matches": expected_refinements == delta["refinements"],
                      "recorded_scalar_returns": counted_checks, "recorded_accounted_rejections": counted_rejections,
                      "scalar_counter_delta_matches": counted_checks == delta["scalar_checks"] and
                          counted_rejections == delta["scalar_rejections"],
                      "scope": "只在记录完整且明确正常返回/额度耗尽/无初始残差时要求精确闭合；异常或截断保留未解释差额。"})
    rejected = [dict(step_index=index, **step) for index, step in enumerate(steps)
                if step["scalar_return_available"] and step["scalar_accepted"] is False]
    result["rejected_candidates"] = rejected
    if event["committed"]:
        classification = "repair_committed_after_application_rejection" if rejected else "repair_committed_without_recorded_rejection"
    elif event["linear_exit"] == "krylov_exhausted" and last and last["last_step"]:
        classification = "krylov_exhausted_last_candidate_" + last["last_step"]["classification"]
    else:
        classification = "not_committed_" + event["linear_exit"]
    result["stage_classification"] = classification
    return result


def analyze(path):
    path = Path(path)
    meta, loaded, binary_path = load_snapshot(path)
    records = [analyze_event(*item) for item in loaded]
    return {"metadata_path": str(path.resolve()), "metadata_sha256": BASE.digest(path),
            "binary_path": str(binary_path.resolve()), "binary_sha256": BASE.digest(binary_path),
            "capture_status": meta["status"], "event_count": len(loaded),
            "trigger_count": meta["trigger_count"], "dropped_events": meta["dropped_events"],
            "capture_bytes": {"binary_payload": meta["payload_bytes"], "owned_cache": meta["cache_bytes"]},
            "analysis_status": "analyzed" if loaded else "no_captured_event", "events": records,
            "rejection_timeline": [{"repair_ordinal": event["repair_ordinal"], "iteration": event["iteration"],
                "phase": event["phase"], "linear_exit": event["linear_exit"], "committed": event["committed"],
                "rejected_candidate": rejected} for event in records for rejected in event["rejected_candidates"]]}


def synthetic_snapshot():
    event, arrays = BASE.synthetic_event()
    for phase in ("before_counts", "after_counts"):
        event[phase].update(scalar_checks=0, scalar_rejections=0)
    event["after_counts"]["scalar_checks"] = 1
    step = {"kind": "ir_candidate", "ir_iteration": 0, "restart": -1, "column": -1,
            "global_norm": "0x0p0", "block_norms": ["0x0p0", "0x0p0"],
            "global_tolerance": (1e-12 + 1e-10).hex(),
            "block_tolerances": [(1e-12 + 1e-10).hex(), (1e-12).hex()],
            "scalar_values_available": True,
            "scalar_return_available": True, "scalar_accepted": True, "scalar_accounted": True,
            "dot_v": "-0x1p0", "denominator": "0x1p1"}
    event.update(repair_ordinal=1, step_capacity=MAX_STEPS, step_count=1, step_total=1, step_dropped=0,
                 last_candidate_step_index=0, last_linear_available=True, linear_rhs_available=True,
                 last_linear_residual=["0x0p0", "0x0p0"], linear_rhs=["-0x1p0", "0x0p0"],
                 linear_exit="ir_return", linear_position={"ir_iteration": 0, "restart": -1, "column": -1},
                 small_events=[step], cycles=[dict(started=False, ended=False, columns_started=0,
                    columns_completed=0, remaining_zero_count=0) for _ in range(3)])
    event["wide"]["last_linear_candidate"] = event["wide"]["after_v"].copy()
    meta, binary = BASE.synthetic_snapshot(compatibility_event(event), arrays)
    event["arrays"] = meta["events"][0]["arrays"]
    meta.update(schema="v63_hsd_1", events=[event], step_bytes=128)
    meta["limits"]["max_steps_per_event"] = MAX_STEPS
    meta["cache_bytes"] = meta["payload_bytes"] + 2 * (4 * meta["long_double_bytes"] + 16) + MAX_STEPS * meta["step_bytes"]
    return meta, MAGIC + binary[8:]


def self_test():
    checks = 0
    def check(condition):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(f"v63 纯合成自检失败：第{checks}项")
    meta, binary = synthetic_snapshot()
    with tempfile.TemporaryDirectory(prefix="v63-math-self-test-") as directory:
        path = Path(directory) / "sample.hsd.json"
        def load(value, blob=binary):
            path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            path.with_suffix(".bin").write_bytes(blob)
            return load_snapshot(path)
        _, events, _ = load(meta)
        result = analyze_event(*events[0])
        check(result["last_linear"]["direction"]["scalars"]["denominator"]["numerator"] == "2")
        check(result["last_linear"]["direction"]["scalars"]["identity_error"]["numerator"] == "0")
        check(result["last_linear"]["residual_evaluation_error"]["whole"]["infinity"]["numerator"] == "0")
        check(result["work_accounting"]["complete_search_accounting"])
        check(result["work_accounting"]["refinement_delta_matches"])
        check(result["work_accounting"]["scalar_counter_delta_matches"])
        check(result["stage_classification"] == "repair_committed_without_recorded_rejection")
        check(result["last_linear"]["captured_norms_and_thresholds_match"])
        # 保存残差与精确矩阵乘积的差分应被独立显露，而非将生产拒绝复制成数学结论。
        changed = copy.deepcopy(meta)
        e = changed["events"][0]
        e.update(committed=False, candidate_available=False, after_scalar_available=False, after=None,
                 linear_exit="exception", failure="合成验算中断")
        e["wide"]["after_v"] = None
        e["after_counts"].update(constant_repair_successes=0, scalar_checks=0)
        e["last_linear_residual"][0] = (2e-10).hex()
        s = e["small_events"][0]
        s.update(global_norm=(2e-10).hex(), block_norms=[(2e-10).hex(), "0x0p0"],
                 scalar_values_available=False, scalar_return_available=False, scalar_accepted=None,
                 scalar_accounted=False, dot_v=None, denominator=None)
        _, changed_events, _ = load(changed)
        answer = analyze_event(*changed_events[0])
        check(answer["last_linear"]["direction"]["residual"]["exact_global_satisfied"])
        check(not answer["last_linear"]["production_residual"]["exact_global_satisfied"])
        check(answer["last_linear"]["residual_evaluation_error"]["upper"]["infinity"]["sign"] == 1)
        check(answer["last_linear"]["last_step"]["classification"] == "global_and_block_residual_failed")
        check(answer["last_linear"]["actual_scalar_evaluated"] is False)
        # 小记录只保存实际检查结果；非有限D明确归入拒绝，不能进入Fraction计算。
        for value in ("nan", "inf", "-inf", "-0x1p0", "0x0p0"):
            test_step = copy.deepcopy(meta["events"][0]["small_events"][0])
            test_step.update(denominator=value, scalar_accepted=False)
            summary = summarize_step(test_step)
            check(summary["classification"] == "residual_passed_scalar_rejected")
            check(summary["scalar_values"]["denominator"]["finite"] == value.startswith(("0x", "-0x")))
        check(finite_hex("0xf.fffffffffffffffp-4") == 1 - Fraction(1, 1 << 64))
        check(finite_hex("0x1.000000000000001p0") > Fraction.from_float(float.fromhex("0x1.000000000000001p0")))
        # 实际阶段列数闭合：一次IR、两列Arnoldi，第一列D拒绝，第二列返回并提交。
        recovered = copy.deepcopy(meta)
        re = recovered["events"][0]
        ir0 = copy.deepcopy(changed["events"][0]["small_events"][0])
        ir1 = dict(ir0, ir_iteration=1)
        rejected = dict(meta["events"][0]["small_events"][0], kind="krylov_candidate", ir_iteration=-1,
                        restart=0, column=1, dot_v="0x1p1", denominator="-0x1p0", scalar_accepted=False)
        accepted = dict(meta["events"][0]["small_events"][0], kind="krylov_candidate", ir_iteration=-1,
                        restart=0, column=2)
        re.update(small_events=[ir0, ir1, rejected, accepted], step_count=4, step_total=4,
                  last_candidate_step_index=3, linear_exit="krylov_return",
                  linear_position={"ir_iteration": -1, "restart": 0, "column": 2})
        re["cycles"][0].update(started=True, columns_started=2, columns_completed=2)
        re["after_counts"].update(refinements=3, extended_krylov_attempts=1, extended_krylov_successes=1,
                                   scalar_checks=2, scalar_rejections=1)
        _, recovered_events, _ = load(recovered)
        recovered_analysis = analyze_event(*recovered_events[0])
        check(recovered_analysis["stage_classification"] == "repair_committed_after_application_rejection")
        check(recovered_analysis["rejected_candidates"][0]["step_index"] == 2)
        check(recovered_analysis["rejected_candidates"][0]["column"] == 1)
        check(recovered_analysis["work_accounting"]["sum_recorded_ir_and_completed_columns"] == 3)
        check(recovered_analysis["work_accounting"]["scalar_counter_delta_matches"])
        # 三周期额度耗尽：最后K0仍失败，即使离线D也非正，不能说生产检查过D。
        exhausted = copy.deepcopy(meta)
        ee = exhausted["events"][0]
        failed_step = dict(ir0, global_norm="0x1.8p1", block_norms=["0x0p0", "0x1.8p1"])
        ee.update(committed=False, candidate_available=False, after_scalar_available=False, after=None,
                  linear_exit="krylov_exhausted", failure="合成三周期耗尽",
                  last_linear_residual=["0x0p0", "-0x1.8p1"],
                  small_events=[dict(failed_step, ir_iteration=i) for i in range(3)] +
                      [dict(failed_step, kind="krylov_candidate", ir_iteration=-1, restart=i, column=2) for i in range(3)],
                  step_count=6, step_total=6, last_candidate_step_index=5,
                  linear_position={"ir_iteration": -1, "restart": 2, "column": 2},
                  cycles=[dict(started=True, ended=True, columns_started=2, columns_completed=2,
                               remaining_zero_count=0) for _ in range(3)])
        ee["wide"].update(after_v=None, last_linear_candidate=["0x1p1", "-0x1p0"])
        ee["after_counts"].update(refinements=8, extended_krylov_attempts=1, extended_krylov_successes=0,
                                   scalar_checks=0, scalar_rejections=0, constant_repair_successes=0)
        _, exhausted_events, _ = load(exhausted)
        exhausted_analysis = analyze_event(*exhausted_events[0])
        check(exhausted_analysis["work_accounting"]["arnoldi_columns_completed"] == 6)
        check(exhausted_analysis["work_accounting"]["sum_recorded_ir_and_completed_columns"] == 8)
        check(exhausted_analysis["last_linear"]["direction"]["scalars"]["denominator"]["sign"] == -1)
        check(exhausted_analysis["last_linear"]["actual_scalar_evaluated"] is False)
        check(exhausted_analysis["rejected_candidates"] == [])
        check(exhausted_analysis["stage_classification"] == "krylov_exhausted_last_candidate_global_and_block_residual_failed")
        check(exhausted_analysis["last_linear"]["residual_evaluation_error"]["whole"]["infinity"]["sign"] == 0)
        interrupted = copy.deepcopy(meta)
        ie = interrupted["events"][0]
        ie.update(linear_exit="exception_after_acceptance", candidate_available=False,
                  after_scalar_available=False, after=None, committed=False)
        ie["wide"]["after_v"] = None
        ie["after_counts"]["constant_repair_successes"] = 0
        _, interrupted_events, _ = load(interrupted)
        interrupted_analysis = analyze_event(*interrupted_events[0])
        check(interrupted_analysis["after"] is None)
        check(interrupted_analysis["last_linear"]["last_step"]["scalar_accepted"])
        check(not interrupted_analysis["work_accounting"]["complete_search_accounting"])
        mutations = []
        def reject(change):
            candidate = copy.deepcopy(meta)
            change(candidate)
            mutations.append(candidate)
        reject(lambda x: x.update(schema="v61_hsd_1"))
        reject(lambda x: x.update(event_count=True))
        reject(lambda x: x.update(cache_bytes=x["cache_bytes"] + 1))
        reject(lambda x: x.update(extra_lp_products=1))
        reject(lambda x: x["events"][0].update(repair_ordinal=0))
        reject(lambda x: x["events"][0].update(step_total=2))
        reject(lambda x: x["events"][0].update(last_candidate_step_index=-1))
        reject(lambda x: x["events"][0].update(last_linear_available=False))
        reject(lambda x: x["events"][0].update(linear_exit="unrecorded"))
        reject(lambda x: x["events"][0]["before_counts"].update(scalar_checks=True))
        reject(lambda x: x["events"][0]["after_counts"].update(scalar_rejections=2))
        reject(lambda x: x["events"][0]["wide"]["last_linear_candidate"].__setitem__(0, "nan"))
        reject(lambda x: x["events"][0]["small_events"][0].update(kind="invented_stage"))
        reject(lambda x: x["events"][0]["small_events"][0].update(ir_iteration=3))
        reject(lambda x: x["events"][0]["small_events"][0].update(block_tolerances=None))
        reject(lambda x: x["events"][0]["small_events"][0].update(denominator="nan"))
        reject(lambda x: x["events"][0]["small_events"][0].update(scalar_return_available=False))
        reject(lambda x: x["events"][0]["small_events"][0].update(global_norm="0x1p0"))
        reject(lambda x: x["events"][0]["cycles"][0].update(columns_completed=1))
        reject(lambda x: x["events"][0]["cycles"][1].update(started=True))
        reject(lambda x: x["events"][0]["arrays"][0].update(offset=9))
        reject(lambda x: x["events"][0]["scalars"].update(tau="0x0p0"))
        for candidate in mutations:
            try:
                load(candidate)
            except (ValueError, KeyError, TypeError):
                check(True)
            else:
                check(False)
        for blob in (b"BADMAGIC" + binary[8:], binary[:-1], binary + b"\0"):
            try:
                load(meta, blob)
            except ValueError:
                check(True)
            else:
                check(False)
        empty = copy.deepcopy(meta)
        empty.update(events=[], event_count=0, trigger_count=0, dropped_events=0, payload_bytes=0,
                     cache_bytes=0, status="no_repair_events")
        check(load(empty, MAGIC)[1] == [])
        # 解析成功不代表账本成功；额外计数必须在阶段核验中被拒绝。
        invalid_balance = copy.deepcopy(meta)
        invalid_balance["events"][0]["after_counts"]["refinements"] = 1
        _, invalid_events, _ = load(invalid_balance)
        try:
            analyze_event(*invalid_events[0])
        except ValueError:
            check(True)
        else:
            check(False)
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        try:
            json.loads(text, object_pairs_hook=BASE.unique_object, parse_constant=BASE.reject_constant)
        except ValueError:
            check(True)
        else:
            check(False)
    return {"passed": True, "checks": checks, "kind": "pure_synthetic_no_solver", "solver_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.inputs and not args.self_test:
        parser.error("至少提供捕获文件，或明确执行 --self-test")
    if args.output:
        protected = {Path(__file__).resolve()} | {(DIRECTORY / name).resolve() for name in HELPER_HASHES}
        protected.update(Path(path).resolve() for path in args.inputs)
        protected.update(Path(path).with_suffix(".bin").resolve() for path in args.inputs)
        if args.output.exists() or args.output.resolve() in protected:
            raise ValueError("分析输出不能覆盖源码、输入或已有证据")
    started = time.perf_counter()
    result = {"schema": "v63_hsd_analysis_1", "analyzer_sha256": BASE.digest(__file__),
              "helper_sha256": HELPER_HASHES, "self_test": self_test(), "records": [], "solver_runs": 0,
              "scope": [
                  "按实际阶段与实际已完成列数定位失败；累计工作量不能单独证明哪个恢复曾被D拒绝。",
                  "r=K0*v-rhs用于D=κ/τ+vz^T*H*vz−χ；生产残差为rhs-K0*v，差分显式翻转符号。",
                  "last_linear_candidate只是最后完成真实残差验算的向量，不自动等于repair返回或提交的after_v。",
                  "未调用应用谓词时只报告离线精确D，不捏造生产D或已执行的分母判断。",
                  "原门槛独立重算binary64公式；原残差求值误差不单独归责于稀疏因子。",
                  "没有采集Arnoldi空间、没有搜索替代方向、没有运行原模型LP或更改数值验收。",
                  "缓存截断或异常明确保留不完整阶段与账本差额，不把未知视为成功。"]}
    for path in args.inputs:
        try:
            result["records"].append(analyze(path))
        except Exception as error:
            result["records"].append({"metadata_path": str(Path(path).resolve()), "analysis_status": "error",
                                      "error": f"{type(error).__name__}: {error}"})
    result["passed"] = all(record["analysis_status"] != "error" for record in result["records"])
    result["analysis_seconds"] = time.perf_counter() - started
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
