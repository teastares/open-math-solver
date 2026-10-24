"""只读精确分解 v61 齐次分母恢复快照；不运行求解器或改写捕获数据。"""

import argparse
import copy
import importlib.util
import json
import math
import struct
import tempfile
import time
from fractions import Fraction
from pathlib import Path


HELPER = Path(__file__).with_name("v53-analyze-hsd.py")
SPEC = importlib.util.spec_from_file_location("v61_exact_helpers", HELPER)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
digest = BASE.digest
packed = BASE.packed
vector_summary = BASE.vector_summary
exact_vector = BASE.exact_vector
validate_csc = BASE.validate_csc
ZERO = Fraction(0)
CAP = 128 * 1024 * 1024
RESERVE = 64 * 1024
MAX_EVENTS = 8
MAGIC = b"V61HSD1\0"
NAMES = ("starts", "rows", "values", "costs", "rhs", "x", "s", "z", "diagonal_h",
         "regularized_h", "inequality", "direction_rhs", "h", "primal_residual", "dual_residual")
INT_NAMES = {"starts", "rows"}
FORMATS = {"i64_le": ("q", 8), "f64_le": ("d", 8), "u8": ("B", 1)}
COUNT_NAMES = ("completed_iterations", "factorizations", "refinements", "constant_solves",
               "direction_solves", "constant_repairs", "constant_repair_successes",
               "extended_krylov_attempts", "extended_krylov_successes")
SCALAR_NAMES = ("tau", "kappa", "primal_regularization", "dual_regularization", "gap_rhs", "complementarity_rhs", "theta")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("整数元数据的类型或下界无效")
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError("布尔元数据不能用整数代替")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 对象含重复字段")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"JSON 含非有限常数：{value}")


def hex_fraction(value):
    """复用 v53 整数尾数解析，禁止先转 binary64 丢失扩展精度低位。"""
    if not isinstance(value, str):
        raise ValueError("要求有限十六进制浮点字符串")
    return BASE.hex_fraction(value)


def load_snapshot(path):
    path = Path(path)
    if path.stat().st_size > 8 * CAP:
        raise ValueError("快照 JSON 超过受控解析上限")
    meta = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)
    if not isinstance(meta, dict) or meta.get("schema") != "v61_hsd_1":
        raise ValueError("HSD 快照 schema 不符")
    limits = {"max_events": MAX_EVENTS, "payload_limit_bytes": CAP, "metadata_reserve_bytes": RESERVE}
    if meta.get("limits") != limits or any(type(meta["limits"].get(key)) is not int for key in limits):
        raise ValueError("HSD 捕获预算合同不符")
    for key in ("extra_lp_products", "extra_lp_solves"):
        if type(meta.get(key)) is not int or meta[key] != 0:
            raise ValueError("HSD 捕获增加了在线乘积或求解")
    boolean(meta.get("enabled"))
    if not isinstance(meta.get("status"), str):
        raise ValueError("HSD 捕获状态类型错误")
    digits = integer(meta.get("long_double_digits"), 54)
    if digits > 256:
        raise ValueError("扩展浮点有效位数超出分析合同")
    events = meta.get("events")
    if not isinstance(events, list) or len(events) > MAX_EVENTS:
        raise ValueError("HSD 事件列表无效")
    if integer(meta.get("event_count")) != len(events):
        raise ValueError("HSD 捕获事件计数不符")
    trigger_count, dropped = (integer(meta.get(key)) for key in ("trigger_count", "dropped_events"))
    if trigger_count != len(events) + dropped:
        raise ValueError("HSD 触发、保存和丢弃计数不一致")
    if not meta["enabled"] and trigger_count:
        raise ValueError("关闭捕获时出现触发记录")
    payload_bytes, cache_bytes = (integer(meta.get(key)) for key in ("payload_bytes", "cache_bytes"))
    if cache_bytes > CAP - RESERVE or cache_bytes < payload_bytes:
        raise ValueError("HSD 受控缓存账本超限或少计")
    binary_path = path.with_suffix(".bin")
    if binary_path.stat().st_size > CAP + 8:
        raise ValueError("HSD 二进制文件超过受控解析上限")
    binary = binary_path.read_bytes()
    if binary[:8] != MAGIC or len(binary) != 8 + payload_bytes:
        raise ValueError("HSD 二进制魔数或长度不符")
    cursor, loaded = 8, []
    total_wide_elements = 0
    for index, event in enumerate(events):
        if integer(event.get("index")) != index:
            raise ValueError("HSD 事件索引必须连续")
        integer(event.get("iteration"))
        integer(event.get("regularization_level"), -1)
        if event.get("phase") not in ("affine", "corrector"):
            raise ValueError("HSD 事件阶段无效")
        for key in ("committed", "candidate_available", "after_scalar_available"):
            boolean(event.get(key))
        if event["after_scalar_available"] and not event["candidate_available"]:
            raise ValueError("HSD 无候选却含候选标量")
        if event["committed"] and not event["after_scalar_available"]:
            raise ValueError("HSD 已提交候选缺少实际标量")
        if not isinstance(event.get("failure"), str):
            raise ValueError("HSD 失败信息类型错误")
        for key in ("before_counts", "after_counts"):
            counts = event.get(key)
            if not isinstance(counts, dict) or set(counts) != set(COUNT_NAMES):
                raise ValueError("HSD 工作计数字段缺失或新增")
            for name in COUNT_NAMES:
                integer(counts[name])
            if counts["constant_repair_successes"] > counts["constant_repairs"] or \
                    counts["extended_krylov_successes"] > counts["extended_krylov_attempts"]:
                raise ValueError("HSD 成功工作数超过尝试数")
        if any(event["after_counts"][key] < event["before_counts"][key] for key in COUNT_NAMES):
            raise ValueError("HSD 恢复期间工作计数倒退")
        arrays = {}
        descriptors = event.get("arrays")
        if not isinstance(descriptors, list) or tuple(item.get("name") for item in descriptors) != NAMES:
            raise ValueError("HSD 数组名或排列不符")
        for field in descriptors:
            name, dtype = field["name"], field.get("dtype")
            count, offset = integer(field.get("count")), integer(field.get("offset"), 8)
            expected = "i64_le" if name in INT_NAMES else "u8" if name == "inequality" else "f64_le"
            if dtype != expected or offset != cursor:
                raise ValueError("HSD 数组格式不符或存在空洞重叠")
            code, width = FORMATS[dtype]
            end = cursor + count * width
            if end > len(binary):
                raise ValueError("HSD 数组越过二进制边界")
            arrays[name] = list(struct.unpack_from(f"<{count}{code}", binary, cursor))
            if dtype == "f64_le" and any(not math.isfinite(value) for value in arrays[name]):
                raise ValueError("HSD 数组含非有限数")
            cursor = end
        wide, scalars = validate_event(event, arrays)
        total_wide_elements += 3 * (len(arrays["costs"]) + len(arrays["rhs"]))
        loaded.append((event, arrays, wide, scalars))
    if cursor != len(binary):
        raise ValueError("HSD 二进制文件含未声明尾部")
    width = integer(meta.get("long_double_bytes"), 8)
    if width > 32 or cache_bytes != payload_bytes + total_wide_elements * width:
        raise ValueError("HSD 宽方向预留缓存账本不一致")
    return meta, loaded, binary_path


def validate_event(event, arrays):
    n, m = len(arrays["costs"]), len(arrays["rhs"])
    if n + m == 0:
        raise ValueError("HSD 工作系统不能为空")
    validate_csc(arrays["starts"], arrays["rows"], arrays["values"], n, m, True)
    lengths = {"x": n, "s": m, "z": m, "diagonal_h": m, "regularized_h": m,
               "inequality": m, "direction_rhs": n + m, "h": m,
               "primal_residual": m, "dual_residual": n}
    if any(len(arrays[name]) != count for name, count in lengths.items()):
        raise ValueError("HSD 工作向量尺寸不符")
    if any(value not in (0, 1) for value in arrays["inequality"]):
        raise ValueError("HSD 行类型掩码无效")
    for row, h in enumerate(arrays["diagonal_h"]):
        if h < 0 or arrays["regularized_h"][row] < h:
            raise ValueError("HSD 对角或有效正则必须非负")
        if arrays["inequality"][row]:
            if arrays["s"][row] <= 0 or arrays["z"][row] <= 0:
                raise ValueError("HSD 不等式的工作松弛或对偶必须严格正")
        elif h != 0:
            raise ValueError("HSD 等式行 H 必须为零")
    if not isinstance(event.get("scalars"), dict) or set(event["scalars"]) != set(SCALAR_NAMES):
        raise ValueError("HSD 标量字段缺失或新增")
    scalars = {name: hex_fraction(event["scalars"][name]) for name in SCALAR_NAMES}
    if scalars["tau"] <= 0 or scalars["kappa"] <= 0 or \
            scalars["primal_regularization"] < 0 or scalars["dual_regularization"] < 0:
        raise ValueError("HSD 齐次标量或正则符号无效")
    if not ZERO <= scalars["theta"] <= 1:
        raise ValueError("HSD 实际残差比例 theta 超出范围")
    wide = {}
    if not isinstance(event.get("wide"), dict) or set(event["wide"]) != {"u", "before_v", "after_v"}:
        raise ValueError("HSD 宽方向字段缺失或新增")
    for name in ("u", "before_v", "after_v"):
        values = event["wide"][name]
        if name == "after_v" and not event["candidate_available"]:
            if values is not None:
                raise ValueError("HSD 未返回候选却保存 after_v")
            wide[name] = None
        else:
            if not isinstance(values, list) or len(values) != n + m:
                raise ValueError("HSD 宽方向尺寸无效")
            wide[name] = [hex_fraction(value) for value in values]
    for name in ("before", "after"):
        values = event.get(name)
        if name == "after" and not event["after_scalar_available"]:
            if values is not None:
                raise ValueError("HSD 未计算候选标量却保存结果")
            continue
        if not isinstance(values, dict) or set(values) != {"dot_u", "dot_v", "denominator"}:
            raise ValueError("HSD 实际点积与分母字段无效")
        for key, value in values.items():
            hex_fraction(value)
    if hex_fraction(event["before"]["denominator"]) > 0:
        raise ValueError("HSD 首次恢复触发分母必须有限非正")
    if event["committed"] and hex_fraction(event["after"]["denominator"]) <= 0:
        raise ValueError("HSD 已提交候选的实际分母必须正")
    return wide, scalars


def residual_vectors(arrays, f, point, rhs):
    n, m = len(f["costs"]), len(f["rhs"])
    upper = [-value for value in rhs[:n]]
    lower = [-f["diagonal_h"][row] * point[n + row] - rhs[n + row] for row in range(m)]
    for col in range(n):
        for offset in range(arrays["starts"][col], arrays["starts"][col + 1]):
            row, coefficient = arrays["rows"][offset], f["values"][offset]
            upper[col] += coefficient * point[n + row]
            lower[row] += coefficient * point[col]
    return upper, lower


def recomputed_tolerances(rhs, n):
    """采用原 binary64 两步表达式重算；这是离线参照，未抄录生产的验收值。"""
    def tolerance(values):
        scale = max(map(abs, values), default=0.0)
        result = 1e-12 + 1e-10 * scale
        if not math.isfinite(result) or result <= 0:
            raise ValueError("离线门槛重算非有限或非正")
        return Fraction.from_float(result)
    return tolerance(rhs), tolerance(rhs[:n]), tolerance(rhs[n:])


def residual_summary(upper, lower, tolerances):
    global_tolerance, upper_tolerance, lower_tolerance = tolerances
    upper_norm = max(map(abs, upper), default=ZERO)
    lower_norm = max(map(abs, lower), default=ZERO)
    global_norm = max(upper_norm, lower_norm)
    return {"upper": vector_summary(upper), "lower": vector_summary(lower),
            "global_norm": packed(global_norm), "recomputed_global_tolerance": packed(global_tolerance),
            "recomputed_upper_tolerance": packed(upper_tolerance),
            "recomputed_lower_tolerance": packed(lower_tolerance),
            "global_ratio": packed(global_norm / global_tolerance),
            "upper_ratio": packed(upper_norm / upper_tolerance),
            "lower_ratio": packed(lower_norm / lower_tolerance),
            "exact_global_satisfied": global_norm <= global_tolerance,
            "exact_blocks_satisfied": upper_norm <= upper_tolerance and lower_norm <= lower_tolerance,
            "threshold_source": "离线按1e-12+1e-10*对应rhs最大绝对值重算binary64；不是捕获的生产门槛或生产残差。"}


def largest_projection_terms(upper, lower, point, count=10):
    n = len(upper)
    terms = [("upper", i, point[i] * value) for i, value in enumerate(upper)]
    terms += [("lower", i, -point[n + i] * value) for i, value in enumerate(lower)]
    selected = sorted(terms, key=lambda item: abs(item[2]), reverse=True)[:count]
    return [{"block": block, "index": index, "value": packed(value)} for block, index, value in selected]


def power_of_two(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def dyadic_sum_bounds(values, precision=160):
    """逐项精确向外取整，公共二进制网格避免合并大量互素 s/z 分母。"""
    largest = max(map(abs, values), default=ZERO)
    if not largest:
        return ZERO, ZERO, 0
    exponent = largest.numerator.bit_length() - largest.denominator.bit_length()
    if largest < power_of_two(exponent):
        exponent -= 1
    grid_exponent = exponent - precision
    lower, upper = 0, 0
    for value in values:
        numerator, denominator = value.numerator, value.denominator
        if grid_exponent >= 0:
            denominator <<= grid_exponent
        else:
            numerator <<= -grid_exponent
        floor, remainder = divmod(numerator, denominator)
        lower += floor
        upper += floor + bool(remainder)
    grid = power_of_two(grid_exponent)
    return lower * grid, upper * grid, grid_exponent


def h_rounding_analysis(arrays, f, vz, positive_energy):
    """只诊断物理系数差；主恒等式仍采用求解器实际保存的 H。"""
    terms, differences = [], []
    for row, saved_h in enumerate(f["diagonal_h"]):
        physical_h = Fraction.from_float(arrays["s"][row]) / Fraction.from_float(arrays["z"][row]) \
            if arrays["inequality"][row] else ZERO
        difference = saved_h - physical_h
        differences.append(difference)
        terms.append(difference * vz[row] ** 2)
    lower, upper, grid_exponent = dyadic_sum_bounds(terms)
    selected = sorted(range(len(terms)), key=lambda row: abs(terms[row]), reverse=True)[:10]
    sign = "positive" if lower > 0 else "negative" if upper < 0 else "exact_zero" if lower == upper == 0 else "undetermined"
    return {"definition": "sum((保存的H_i-精确s_i/z_i)*v_z_i^2)，等式行取零；这是物理系数差，不能单独称为分解误差。",
            "sum_kind": "逐项Fraction向公共dyadic网格外取整所得严格界；未合并所有s/z分母为一个精确总和。",
            "grid_exponent": grid_exponent, "guard_bits_below_largest_term": 160,
            "lower": packed(lower), "upper": packed(upper), "interval_width": packed(upper - lower),
            "sign_certification": sign, "lower_over_positive_energy": packed(lower / positive_energy),
            "upper_over_positive_energy": packed(upper / positive_energy),
            "coefficient_difference": vector_summary(differences),
            "largest_exact_signed_terms": [{"row": row, "coefficient_difference": packed(differences[row]),
                                            "energy_contribution": packed(terms[row])} for row in selected]}


def exact_direction(arrays, f, point, u, scalars, raw):
    n, m = len(f["costs"]), len(f["rhs"])
    vx, vz = point[:n], point[n:]
    const_rhs = [-value for value in f["costs"]] + f["rhs"]
    upper, lower = residual_vectors(arrays, f, point, const_rhs)
    dot_v = sum((coefficient * value for coefficient, value in zip(f["costs"] + f["rhs"], point)), ZERO)
    dot_u = sum((coefficient * value for coefficient, value in zip(f["costs"] + f["rhs"], u)), ZERO)
    ratio = scalars["kappa"] / scalars["tau"]
    energy = sum((f["diagonal_h"][row] * vz[row] ** 2 for row in range(m)), ZERO)
    upper_projection = sum((x * value for x, value in zip(vx, upper)), ZERO)
    lower_projection = -sum((z * value for z, value in zip(vz, lower)), ZERO)
    defect = upper_projection + lower_projection
    positive_energy = ratio + energy
    denominator = ratio - dot_v
    identity_error = denominator - (positive_energy - defect)
    if identity_error or defect != dot_v + energy:
        raise AssertionError("齐次分母精确能量恒等式失败")
    upper_bound = sum((abs(x * value) for x, value in zip(vx, upper)), ZERO)
    lower_bound = sum((abs(z * value) for z, value in zip(vz, lower)), ZERO)
    values = {"ratio": ratio, "dot_v": dot_v, "dot_u": dot_u, "denominator": denominator,
              "energy": energy, "positive_energy": positive_energy, "defect": defect,
              "upper_projection": upper_projection, "lower_projection": lower_projection,
              "absolute_projection_bound": upper_bound + lower_bound, "identity_error": identity_error,
              "defect_over_positive_energy": defect / positive_energy,
              "denominator_over_positive_energy": denominator / positive_energy}
    classification = "actual_scalar_unavailable"
    if raw is not None:
        production = {key: hex_fraction(raw[key]) for key in ("dot_u", "dot_v", "denominator")}
        dot_error = production["dot_v"] - dot_v
        ratio_subtraction_error = production["denominator"] - (ratio - production["dot_v"])
        total_error = production["denominator"] - denominator
        if total_error != ratio_subtraction_error - dot_error:
            raise AssertionError("齐次分母标量算术误差恒等式失败")
        values.update(raw_dot_u=production["dot_u"], raw_dot_v=production["dot_v"],
                      raw_denominator=production["denominator"], dot_u_rounding=production["dot_u"] - dot_u,
                      dot_v_rounding=dot_error, ratio_and_subtraction_rounding=ratio_subtraction_error,
                      total_scalar_rounding=total_error)
        if denominator > 0 and production["denominator"] <= 0:
            classification = "scalar_rounding_changes_positive_saved_direction_to_nonpositive"
        elif denominator <= 0 and production["denominator"] > 0:
            classification = "scalar_rounding_changes_nonpositive_saved_direction_to_positive"
        elif denominator <= 0:
            classification = "saved_direction_has_nonpositive_exact_denominator"
        else:
            classification = "saved_direction_and_actual_scalar_have_positive_denominator"
    double_rhs = [-value for value in arrays["costs"]] + arrays["rhs"]
    summary = residual_summary(upper, lower, recomputed_tolerances(double_rhs, n))
    return {"identity_exact": True, "classification": classification,
            "scalars": {key: packed(value) for key, value in values.items()},
            "residual": summary, "direction": {"upper": vector_summary(vx), "lower": vector_summary(vz)},
            "largest_signed_projection_terms": largest_projection_terms(upper, lower, point),
            "h_rounding": h_rounding_analysis(arrays, f, vz, positive_energy)}, values


def analyze_event(event, arrays, wide, scalars):
    f = {name: exact_vector(arrays[name]) for name in ("values", "costs", "rhs", "diagonal_h", "direction_rhs")}
    n, m = len(f["costs"]), len(f["rhs"])
    before, before_values = exact_direction(arrays, f, wide["before_v"], wide["u"], scalars, event["before"])
    result = {"index": event["index"], "iteration": event["iteration"], "phase": event["phase"],
              "regularization_level": event["regularization_level"], "dimensions": {"n": n, "m": m, "nnz": len(f["values"])},
              "before_counts": event["before_counts"], "after_counts": event["after_counts"],
              "work_delta": {key: event["after_counts"][key] - event["before_counts"][key] for key in COUNT_NAMES},
              "committed": event["committed"], "candidate_available": event["candidate_available"],
              "after_scalar_available": event["after_scalar_available"], "failure": event["failure"],
              "before": before, "after": None, "recovery_difference": None}
    upper_u, lower_u = residual_vectors(arrays, f, wide["u"], f["direction_rhs"])
    result["u_residual"] = residual_summary(upper_u, lower_u, recomputed_tolerances(arrays["direction_rhs"], n))
    if wide["after_v"] is not None:
        after, after_values = exact_direction(arrays, f, wide["after_v"], wide["u"], scalars, event["after"])
        result["after"] = after
        delta = [new - old for new, old in zip(wide["after_v"], wide["before_v"])]
        projected_delta = sum((coefficient * value for coefficient, value in zip(f["costs"] + f["rhs"], delta)), ZERO)
        delta_d = after_values["denominator"] - before_values["denominator"]
        delta_e = after_values["energy"] - before_values["energy"]
        delta_chi = after_values["defect"] - before_values["defect"]
        if delta_d != -projected_delta or delta_d != delta_e - delta_chi:
            raise AssertionError("齐次恢复前后分母差的精确恒等式失败")
        result["recovery_difference"] = {"identity_exact": True,
            "scalars": {key: packed(value) for key, value in {"denominator_delta": delta_d,
                "energy_delta": delta_e, "defect_delta": delta_chi, "q_dot_direction_delta": projected_delta}.items()},
            "direction_delta": vector_summary(delta)}
    return result


def analyze(path):
    path = Path(path)
    meta, loaded, binary = load_snapshot(path)
    result = {"metadata_path": str(path.resolve()), "metadata_sha256": digest(path),
              "binary_path": str(binary.resolve()), "binary_sha256": digest(binary),
              "capture_status": meta["status"], "event_count": len(loaded),
              "trigger_count": meta["trigger_count"], "dropped_events": meta["dropped_events"],
              "capture_bytes": {"binary_payload": meta["payload_bytes"], "owned_cache": meta["cache_bytes"]},
              "analysis_status": "analyzed" if loaded else "no_captured_event", "events": []}
    for event, arrays, wide, scalars in loaded:
        result["events"].append(analyze_event(event, arrays, wide, scalars))
    return result


def synthetic_event():
    """构造一维 G=1,H=1 的非正触发与精确可恢复候选，不调用 LP。"""
    arrays = {"starts": [0, 1], "rows": [0], "values": [1.0], "costs": [1.0], "rhs": [0.0],
              "x": [1.0], "s": [1.0], "z": [1.0], "diagonal_h": [1.0], "regularized_h": [1.25],
              "inequality": [1], "direction_rhs": [0.0, 0.0], "h": [0.0],
              "primal_residual": [0.0], "dual_residual": [0.0]}
    counters = dict.fromkeys(COUNT_NAMES, 0)
    after_counts = dict(counters, constant_solves=1, constant_repairs=1, constant_repair_successes=1)
    event = {"index": 0, "iteration": 1, "phase": "affine", "regularization_level": 0,
             "before_counts": counters, "after_counts": after_counts, "committed": True,
             "candidate_available": True, "after_scalar_available": True, "failure": "",
             "scalars": {key: value.hex() for key, value in {"tau": 1.0, "kappa": 1.0,
                 "primal_regularization": 0.25, "dual_regularization": 0.25,
                 "gap_rhs": 0.0, "complementarity_rhs": 0.0, "theta": 1.0}.items()},
             "before": {"dot_u": "0x0p0", "dot_v": "0x1p1", "denominator": "-0x1p0"},
             "after": {"dot_u": "0x0p0", "dot_v": "-0x1p0", "denominator": "0x1p1"},
             "wide": {"u": ["0x0p0", "0x0p0"], "before_v": ["0x1p1", "-0x1p0"],
                      "after_v": ["-0x1p0", "-0x1p0"]}}
    return event, arrays


def synthetic_snapshot(event, arrays):
    event = copy.deepcopy(event)
    binary = bytearray(MAGIC)
    event["arrays"] = []
    for name in NAMES:
        dtype = "i64_le" if name in INT_NAMES else "u8" if name == "inequality" else "f64_le"
        code, _ = FORMATS[dtype]
        values = arrays[name]
        event["arrays"].append({"name": name, "dtype": dtype, "count": len(values), "offset": len(binary)})
        binary.extend(struct.pack(f"<{len(values)}{code}", *values))
    size = len(binary) - 8
    meta = {"schema": "v61_hsd_1", "enabled": True, "status": "captured", "event_count": 1,
            "trigger_count": 1, "dropped_events": 0, "payload_bytes": size, "cache_bytes": size + 3 * 2 * 16,
            "limits": {"max_events": MAX_EVENTS, "payload_limit_bytes": CAP, "metadata_reserve_bytes": RESERVE},
            "long_double_digits": 64, "long_double_bytes": 16, "long_double_max_exponent": 16384,
            "extra_lp_products": 0, "extra_lp_solves": 0, "events": [event]}
    return meta, bytes(binary)


def self_test():
    checks = 0
    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("v61 离线合成自测失败")
        checks += 1
    check(hex_fraction("0x8.000000000000001p-3") == 1 + Fraction(1, 1 << 63))
    check(hex_fraction("-0x1p-1074") == -Fraction(1, 1 << 1074))
    check(hex_fraction("-0x0p0") == 0)
    for value in (None, True, 1, "nan", "inf", "-inf", "1.0", "--0x1p0", "0x1p999999", "0x1p0extra"):
        try:
            hex_fraction(value)
        except ValueError:
            check(True)
        else:
            check(False)
    event, arrays = synthetic_event()
    wide, scalars = validate_event(event, arrays)
    result = analyze_event(event, arrays, wide, scalars)
    check(result["before"]["scalars"]["denominator"]["numerator"] == "-1")
    check(result["after"]["scalars"]["denominator"]["numerator"] == "2")
    check(result["before"]["scalars"]["energy"]["numerator"] == "1")
    check(result["before"]["scalars"]["defect"]["numerator"] == "3")
    check(result["after"]["scalars"]["defect"]["numerator"] == "0")
    check(result["recovery_difference"]["scalars"]["denominator_delta"]["numerator"] == "3")
    check(result["after"]["residual"]["exact_global_satisfied"])
    check(result["after"]["residual"]["exact_blocks_satisfied"])
    check(result["after"]["h_rounding"]["sign_certification"] == "exact_zero")
    rounded_h = copy.deepcopy(arrays)
    rounded_h.update(s=[1.0], z=[3.0], diagonal_h=[1.0 / 3.0])
    rw, rs = validate_event(event, rounded_h)
    rounded_result = analyze_event(event, rounded_h, rw, rs)
    check(rounded_result["before"]["h_rounding"]["sign_certification"] == "negative")
    exact_h_term = Fraction.from_float(1.0 / 3.0) - Fraction(1, 3)
    h_interval = rounded_result["before"]["h_rounding"]
    lo = Fraction(int(h_interval["lower"]["numerator"]), int(h_interval["lower"]["denominator"]))
    hi = Fraction(int(h_interval["upper"]["numerator"]), int(h_interval["upper"]["denominator"]))
    check(lo <= exact_h_term <= hi)
    for terms in ([Fraction(1, 3), Fraction(-1, 7)], [Fraction(-1, 3), Fraction(1, 7)],
                  [Fraction(1, 3), Fraction(-1, 3)], [ZERO], []):
        lo, hi, _ = dyadic_sum_bounds(terms)
        check(lo <= sum(terms, ZERO) <= hi)
    equality = copy.deepcopy(arrays)
    equality.update(inequality=[0], diagonal_h=[0.0], s=[0.0], z=[-1.0])
    equality_event = copy.deepcopy(event)
    equality_event["wide"]["after_v"] = ["0x0p0", "-0x1p0"]
    equality_event["after"] = {"dot_u": "0x0p0", "dot_v": "0x0p0", "denominator": "0x1p0"}
    ew, es = validate_event(equality_event, equality)
    equality_result = analyze_event(equality_event, equality, ew, es)
    check(equality_result["after"]["scalars"]["energy"]["numerator"] == "0")
    check(equality_result["after"]["scalars"]["denominator"]["numerator"] == "1")
    check(equality_result["after"]["residual"]["exact_blocks_satisfied"])
    # 宽低位使精确分母为正；刻意记录为零的原标量用于检验符号分类。
    lowbits = copy.deepcopy(event)
    lowbits.update(committed=False, candidate_available=False, after_scalar_available=False, after=None)
    lowbits["wide"]["before_v"] = ["0xf.fffffffffffffffp-4", "-0x1p0"]
    lowbits["wide"]["after_v"] = None
    lowbits["before"] = {"dot_u": "0x0p0", "dot_v": "0x1p0", "denominator": "0x0p0"}
    lw, ls = validate_event(lowbits, arrays)
    low_result = analyze_event(lowbits, arrays, lw, ls)
    check(low_result["before"]["scalars"]["denominator"]["denominator"] == str(1 << 64))
    check(low_result["before"]["classification"] == "scalar_rounding_changes_positive_saved_direction_to_nonpositive")
    check(low_result["after"] is None)
    zero_d = copy.deepcopy(lowbits)
    zero_d["wide"]["before_v"][0] = "0x1p0"
    zw, zs = validate_event(zero_d, arrays)
    check(analyze_event(zero_d, arrays, zw, zs)["before"]["scalars"]["denominator"]["sign"] == 0)
    meta, binary = synthetic_snapshot(event, arrays)
    with tempfile.TemporaryDirectory(prefix="v61-hsd-self-test-") as directory:
        path = Path(directory) / "sample.hsd.json"
        def write_and_load(value, blob=binary):
            path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
            path.with_suffix(".bin").write_bytes(blob)
            return load_snapshot(path)
        loaded = write_and_load(meta)
        check(len(loaded[1]) == 1)
        check(loaded[1][0][2]["after_v"] == [Fraction(-1), Fraction(-1)])
        mutations = []
        def reject(mutator):
            value = copy.deepcopy(meta)
            mutator(value)
            mutations.append(value)
        reject(lambda x: x.update(schema="v53_hsd_1"))
        reject(lambda x: x.update(event_count=True))
        reject(lambda x: x.update(trigger_count=2))
        reject(lambda x: x.update(cache_bytes=CAP))
        reject(lambda x: x.update(long_double_digits=53))
        reject(lambda x: x.update(extra_lp_products=1))
        reject(lambda x: x["events"][0]["wide"]["before_v"].__setitem__(0, "nan"))
        reject(lambda x: x["events"][0]["before"].update(denominator="inf"))
        reject(lambda x: x["events"][0]["before_counts"].update(factorizations=False))
        reject(lambda x: x["events"][0]["arrays"][0].update(offset=9))
        reject(lambda x: x["events"][0]["arrays"][1].update(count=-1))
        reject(lambda x: x["events"][0].update(candidate_available=False))
        reject(lambda x: x["events"][0]["scalars"].update(tau="0x0p0"))
        reject(lambda x: x["events"][0]["after"].update(denominator="0x0p0"))
        reject(lambda x: x["events"][0]["wide"]["u"].append("0x0p0"))
        for value in mutations:
            try:
                write_and_load(value)
            except (ValueError, KeyError, TypeError):
                check(True)
            else:
                check(False)
        for blob in (b"BADMAGIC" + binary[8:], binary + b"\0", binary[:-1]):
            try:
                write_and_load(meta, blob)
            except ValueError:
                check(True)
            else:
                check(False)
        corrupted = bytearray(binary)
        value_descriptor = next(x for x in meta["events"][0]["arrays"] if x["name"] == "values")
        struct.pack_into("<d", corrupted, value_descriptor["offset"], math.inf)
        try:
            write_and_load(meta, corrupted)
        except ValueError:
            check(True)
        else:
            check(False)
        empty = copy.deepcopy(meta)
        empty.update(status="no_target", event_count=0, trigger_count=0, payload_bytes=0, cache_bytes=0, events=[])
        check(write_and_load(empty, MAGIC)[1] == [])
    for text in ('{"x":1,"x":2}', '{"x":NaN}'):
        try:
            json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
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
        parser.error("至少提供一个捕获文件，或明确执行 --self-test")
    if args.output:
        protected = {Path(__file__).resolve(), HELPER.resolve()}
        protected.update(Path(path).resolve() for path in args.inputs)
        protected.update(Path(path).with_suffix(".bin").resolve() for path in args.inputs)
        if args.output.resolve() in protected or args.output.exists():
            raise ValueError("分析输出不能覆盖源码、输入或已有证据")
    start = time.perf_counter()
    result = {"schema": "v61_hsd_analysis_1", "analyzer_sha256": digest(__file__), "helper_sha256": digest(HELPER),
              "self_test": self_test(), "records": [], "solver_runs": 0,
              "scope": [
                  "全部精确分数解释已保存的binary64矩阵和完整hex宽方向；不重新求解LP或重建未知理想方向。",
                  "r_x=G^T*v_z+c；r_z=G*v_x-H*v_z-b；chi=v_x^T*r_x-v_z^T*r_z。",
                  "D=kappa/tau-q^T*v=kappa/tau+v_z^T*H*v_z-chi，q=[c;b]；等式由Fraction逐事件验证。",
                  "rawD与exactD之差拆为点积舍入和比例/减法合并舍入；不能据此单独归责于稀疏因子。",
                  "保存H与精确s/z的物理系数差以逐行精确贡献及总和严格dyadic界报告，不改变采用实际H的主恒等式。",
                  "离线分块范数与重算原binary64门槛比较；不覆盖生产残差验收，不把分块通过当作分母为正的保证。",
                  "after_v是repair实际返回候选，committed=false时不得称为后续迭代采用的常数方向。",
                  "工作点、方向和计数来自实际失败/恢复事件；终报的保留最佳候选不能代替这些工作点。",
                  "受控捕获缓存不等于进程RSS；JSON流开销与离线精确分析均不是免费计算。"]}
    for path in args.inputs:
        try:
            result["records"].append(analyze(path))
        except Exception as error:
            result["records"].append({"metadata_path": str(Path(path).resolve()), "analysis_status": "error",
                                      "error": f"{type(error).__name__}: {error}"})
    result["passed"] = all(record["analysis_status"] != "error" for record in result["records"])
    result["analysis_seconds"] = time.perf_counter() - start
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
