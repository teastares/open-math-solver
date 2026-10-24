"""只读分解 v53 已保存的 HSD 方向、更新和原坐标对偶误差；不调用求解器。"""

import argparse
import hashlib
import json
import math
import re
import struct
import tempfile
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


CAP = 128 * 1024 * 1024
FORMATS = {"f64_le": ("d", 8), "i64_le": ("q", 8), "u64_le": ("Q", 8), "u8": ("B", 1)}
STATIC_NAMES = ("g_starts", "g_rows", "g_values", "costs", "b", "inequality", "original_row",
                "dual_sign", "row_scale", "column_scale", "original_columns", "fixed_values",
                "a_starts", "a_rows", "a_values", "original_costs", "column_lower", "column_upper",
                "row_lower", "row_upper")
STEP_NAMES = ("old_x", "old_z", "new_x", "new_z", "stored_rd", "rhs_u", "diagonal_h",
              "u", "v", "dz", "extracted_z", "assess_x", "assess_y")
INT_NAMES = {"g_starts", "g_rows", "original_row", "a_starts", "a_rows"}
BOUND_NAMES = {"column_lower", "column_upper", "row_lower", "row_upper"}
ZERO = Fraction(0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hex_fraction(value):
    """不经过 binary64，保留 C++ long double 十六进制文本的全部有效位。"""
    match = re.fullmatch(r"([+-]?)0[xX]([0-9a-fA-F]+)(?:\.([0-9a-fA-F]*))?[pP]([+-]?\d+)", value or "")
    if not match or len(value) > 256:
        raise ValueError("要求有限的十六进制浮点文本")
    sign, whole, tail, exponent = match.groups()
    tail = tail or ""
    shift = int(exponent) - 4 * len(tail)
    if abs(shift) > 65536:
        raise ValueError("十六进制浮点指数超出诊断范围")
    integer = int(whole + tail, 16) * (-1 if sign == "-" else 1)
    return Fraction(integer << shift) if shift >= 0 else Fraction(integer, 1 << -shift)


def dec(value, precision=80):
    with localcontext() as context:
        context.prec = precision
        return Decimal(value.numerator) / Decimal(value.denominator)


def packed(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal80": str(dec(value)), "sign": (value > 0) - (value < 0)}


def vector_summary(values):
    index = max(range(len(values)), key=lambda i: abs(values[i])) if values else None
    signed = values[index] if index is not None else ZERO
    return {"infinity": packed(abs(signed)), "argmax": index, "signed_at_argmax": packed(signed)}


def exact_vector(values):
    return [Fraction.from_float(value) for value in values]


def selected_columns(n):
    return sorted({0, n // 2, n - 1}) if n else []


def validate_csc(starts, rows, values, n, m, sorted_unique):
    if len(starts) != n + 1 or starts[0] != 0 or starts[-1] != len(values) or len(rows) != len(values):
        raise ValueError("CSC 维数或首尾位置无效")
    if any(x < 0 or x > y for x, y in zip(starts, starts[1:])):
        raise ValueError("CSC 指针不单调")
    if any(row < 0 or row >= m for row in rows):
        raise ValueError("CSC 行号越界")
    if sorted_unique:
        for col in range(n):
            if any(rows[k] >= rows[k + 1] for k in range(starts[col], starts[col + 1] - 1)):
                raise ValueError("工作 CSC 列内行号不是严格递增")


def validate_static(a, scalars):
    n, m = len(a["costs"]), len(a["b"])
    on, om = len(a["original_costs"]), len(a["row_lower"])
    validate_csc(a["g_starts"], a["g_rows"], a["g_values"], n, m, True)
    validate_csc(a["a_starts"], a["a_rows"], a["a_values"], on, om, False)
    for name in ("inequality", "original_row", "dual_sign", "row_scale"):
        if len(a[name]) != m:
            raise ValueError(f"静态行向量 {name} 长度错误")
    if len(a["column_scale"]) != n or len(a["column_lower"]) != on or len(a["column_upper"]) != on or len(a["row_upper"]) != om:
        raise ValueError("原模型界或缩放尺寸不符")
    if any(x not in (0, 1) for x in a["inequality"]) or any(x not in (-1, 1) for x in a["dual_sign"]):
        raise ValueError("不等式掩码或原行对偶符号无效")
    if any(x < -1 or x >= om for x in a["original_row"]):
        raise ValueError("原行映射越界")
    columns = a["original_columns"] or list(range(n))
    if len(columns) != n or len(set(columns)) != n or any(x < 0 or x >= on for x in columns):
        raise ValueError("原列映射无效")
    if (not a["original_columns"] and n != on and n != 0) or (a["fixed_values"] and len(a["fixed_values"]) != on):
        raise ValueError("恒等列映射或固定值尺寸无效")
    if any(x <= 0 for x in a["row_scale"] + a["column_scale"]) or hex_fraction(scalars["objective_scale"]) <= 0:
        raise ValueError("缩放必须有限且严格为正")
    if hex_fraction(scalars["sense"]) not in (-1, 1):
        raise ValueError("目标方向必须为正负一")
    for prefix in ("column", "row"):
        for lower, upper in zip(a[prefix + "_lower"], a[prefix + "_upper"]):
            if lower == math.inf or upper == -math.inf or lower > upper:
                raise ValueError("原模型界无效")
    for name in ("dual_absolute", "dual_relative"):
        if hex_fraction(scalars[name]) < 0:
            raise ValueError("对偶容差无效")
    return n, m, on, om


def validate_step(a, scalars, dimensions):
    n, m, on, om = dimensions
    expected = {"old_x": n, "new_x": n, "stored_rd": n, "old_z": m, "new_z": m,
                "diagonal_h": m, "dz": m, "extracted_z": m, "rhs_u": n + m,
                "u": n + m, "v": n + m, "assess_x": on, "assess_y": om}
    if any(len(a[key]) != count for key, count in expected.items()):
        raise ValueError("完成步的数组维数不符")
    if any(x < 0 for x in a["diagonal_h"]):
        raise ValueError("K0 对角不能为负")
    if any(hex_fraction(scalars[key]) <= 0 for key in ("tau", "next_tau")):
        raise ValueError("完成步的 tau 必须严格为正")
    if not ZERO < hex_fraction(scalars["step"]) <= 1 or not ZERO <= hex_fraction(scalars["theta"]) <= 1:
        raise ValueError("共同步长或 theta 超出合同")


def load_snapshot(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != "v53_hsd_1":
        raise ValueError("快照 schema 不符")
    if meta.get("cache_limit") != CAP or meta.get("metadata_reserve") != 65536 or meta.get("allocated_step_slots") != 4:
        raise ValueError("快照内存合同不符")
    if any(meta.get(key) != 0 for key in ("extra_lp_products", "extra_lp_solves", "diagnostic_clock_reads")):
        raise ValueError("捕获增加了未授权的在线计算")
    if not meta.get("binary_written"):
        raise ValueError("捕获的二进制写出未成功")
    binary_path = path.with_suffix(".bin")
    if not meta.get("static", {}).get("arrays"):
        return meta, None, [], None
    binary = binary_path.read_bytes()
    if len(binary) > CAP + 8 or binary[:8] != b"V53HSD1\0" or meta.get("binary_bytes") != len(binary):
        raise ValueError("快照魔数或受控数据长度无效")
    intervals = []

    def arrays(section, names, complete=True):
        result = {}
        for item in section.get("arrays", []):
            name, dtype, count, offset = (item[key] for key in ("name", "dtype", "count", "offset"))
            expected = "i64_le" if name in INT_NAMES else "u64_le" if name == "original_columns" else "u8" if name == "inequality" else "f64_le"
            if name not in names or name in result or dtype != expected or type(count) is not int or count < 0 or type(offset) is not int or offset < 8:
                raise ValueError("数组描述无效")
            code, width = FORMATS[dtype]
            end = offset + count * width
            if end > len(binary):
                raise ValueError("数组越过快照边界")
            result[name] = list(struct.unpack_from(f"<{count}{code}", binary, offset))
            intervals.append((offset, end))
            if dtype == "f64_le" and any(math.isnan(x) or (name not in BOUND_NAMES and not math.isfinite(x)) for x in result[name]):
                raise ValueError("数组含不允许的非有限数")
        if complete and set(result) != set(names):
            raise ValueError("数组缺失")
        return result

    static = arrays(meta["static"], STATIC_NAMES)
    dimensions = validate_static(static, meta["static"]["scalars"])
    steps = []
    for step in meta.get("steps", []):
        values = arrays(step, STEP_NAMES)
        if not all(step.get("flags", {}).get(key) for key in ("have_input", "have_u", "direction_returned", "step_complete", "assessed")):
            raise ValueError("steps 中包含未完成或未验算步骤")
        if step.get("phase") != "corrector" or not step.get("assessment", {}).get("available"):
            raise ValueError("完整校正步缺少实际验算记录")
        validate_step(values, step["scalars"], dimensions)
        steps.append((step, values))
    iterations = [step[0]["iteration"] for step in steps]
    if len(steps) > 3 or iterations != sorted(set(iterations)):
        raise ValueError("首步和末两步的排序或去重不符")
    pending = meta.get("last_attempt")
    if pending and pending.get("arrays"):
        arrays(pending, STEP_NAMES, complete=False)
    declared_cache = 65536 + meta.get("static_payload_bytes", -1) + 4 * meta.get("step_slot_bytes", -1)
    if meta.get("cache_bytes") != declared_cache or declared_cache > CAP:
        raise ValueError("四槽受控内存账本无效")
    cursor = 8
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("数组不是连续且无重叠的完整 payload")
        cursor = end
    if cursor != len(binary):
        raise ValueError("存在未声明二进制尾部")
    return meta, static, steps, binary_path


def exact_working(a, step, scalars):
    n, m = len(a["costs"]), len(a["b"])
    f = {key: exact_vector(a[key]) for key in ("g_values", "costs", "b")}
    s = {key: exact_vector(step[key]) for key in STEP_NAMES}
    tau, next_tau, theta, alpha, dtau = (hex_fraction(scalars[key]) for key in ("tau", "next_tau", "theta", "step", "dtau"))
    c, b = f["costs"], f["b"]
    uz, vz = s["u"][n:], s["v"][n:]
    rho_reconstruction = [s["dz"][i] - uz[i] - vz[i] * dtau for i in range(m)]
    rho_update_z = [s["new_z"][i] - s["old_z"][i] - alpha * s["dz"][i] for i in range(m)]
    rho_update_tau = next_tau - tau - alpha * dtau
    v = {name: [] for name in ("rd_true", "rd_new", "e_u", "e_v", "dtau_e_v", "rho_rhs",
         "transpose_rho_reconstruction", "theta_residual_rounding", "E_d", "transpose_rho_update_z",
         "cost_rho_update_tau", "predicted_rd_new", "rd_normalized", "predicted_rd_normalized",
         "initial_residual_carry", "direction_contribution", "update_contribution")}
    lower_u = [-s["diagonal_h"][i] * uz[i] - s["rhs_u"][n + i] for i in range(m)]
    lower_v = [-s["diagonal_h"][i] * vz[i] - b[i] for i in range(m)]
    for col in range(n):
        rd, rdnew, eu, ev = c[col] * tau, c[col] * next_tau, -s["rhs_u"][col], c[col]
        ed, recon, update = c[col] * dtau, ZERO, ZERO
        for k in range(a["g_starts"][col], a["g_starts"][col + 1]):
            row, g = a["g_rows"][k], f["g_values"][k]
            rd += g * s["old_z"][row]
            rdnew += g * s["new_z"][row]
            eu += g * uz[row]
            ev += g * vz[row]
            ed += g * s["dz"][row]
            recon += g * rho_reconstruction[row]
            update += g * rho_update_z[row]
            lower_u[row] += g * s["u"][col]
            lower_v[row] += g * s["v"][col]
        ed += theta * rd
        rho_rhs = s["rhs_u"][col] + theta * s["stored_rd"][col]
        residual_rounding = theta * (rd - s["stored_rd"][col])
        if ed != eu + dtau * ev + rho_rhs + recon + residual_rounding:
            raise AssertionError("Ed 五项精确恒等式失败")
        predicted = (1 - alpha * theta) * rd + alpha * ed + update + c[col] * rho_update_tau
        if predicted != rdnew:
            raise AssertionError("共同步长更新精确恒等式失败")
        values = (rd, rdnew, eu, ev, dtau * ev, rho_rhs, recon, residual_rounding, ed, update,
                  c[col] * rho_update_tau, predicted, rdnew / next_tau, predicted / next_tau,
                  (1 - alpha * theta) * rd / next_tau, alpha * ed / next_tau,
                  (update + c[col] * rho_update_tau) / next_tau)
        for key, value in zip(v, values):
            v[key].append(value)
    return v, lower_u, lower_v, {"rho_update_tau": rho_update_tau, "rho_reconstruction": rho_reconstruction,
                                 "rho_update_z": rho_update_z, "dtau": dtau, "next_tau": next_tau,
                                 "theta": theta, "alpha": alpha}


def original_coordinates(a, static_scalars, step, scalars, working):
    n, on, om = len(a["costs"]), len(a["original_costs"]), len(a["row_lower"])
    columns = a["original_columns"] or list(range(n))
    original_to_work = {original: col for col, original in enumerate(columns)}
    ratio = hex_fraction(static_scalars["objective_scale"])
    sense = hex_fraction(static_scalars["sense"])
    tau = hex_fraction(scalars["next_tau"])
    scale = exact_vector(a["column_scale"])
    row_scale = exact_vector(a["row_scale"])
    z, extracted, actual_y = (exact_vector(step[key]) for key in ("new_z", "extracted_z", "assess_y"))
    ideal_y, extracted_y = [ZERO] * om, [ZERO] * om
    for row, original in enumerate(a["original_row"]):
        if original >= 0:
            coefficient = Fraction(a["dual_sign"][row]) * row_scale[row] * ratio
            ideal_y[original] += coefficient * z[row] / tau
            extracted_y[original] += coefficient * extracted[row]
    bound_term = [ZERO] * n
    for col in range(n):
        for k in range(a["g_starts"][col], a["g_starts"][col + 1]):
            row = a["g_rows"][k]
            if a["original_row"][row] < 0:
                bound_term[col] += Fraction.from_float(a["g_values"][k]) * z[row] / tau
    out = {key: [ZERO] * n for key in ("work_residual_unscaled", "column_bound_contribution", "base_without_column_bounds",
           "data_relation_difference", "tau_division_rounding", "original_y_recovery_rounding", "original_reduced_cost")}
    rc_all, transpose_all = [], []
    for original in range(on):
        cost = sense * Fraction.from_float(a["original_costs"][original])
        transpose, ideal_transpose, div_error, recovery_error = ZERO, ZERO, ZERO, ZERO
        for k in range(a["a_starts"][original], a["a_starts"][original + 1]):
            row, value = a["a_rows"][k], Fraction.from_float(a["a_values"][k])
            transpose += value * actual_y[row]
            ideal_transpose += value * ideal_y[row]
            div_error -= value * (extracted_y[row] - ideal_y[row])
            recovery_error -= value * (actual_y[row] - extracted_y[row])
        rc = cost - transpose
        rc_all.append(rc)
        transpose_all.append(transpose)
        if original not in original_to_work:
            continue
        col = original_to_work[original]
        unscaled = ratio / scale[col] * working["rd_normalized"][col]
        bound = ratio / scale[col] * bound_term[col]
        base = unscaled - bound
        data_error = cost - ideal_transpose - base
        if rc != base + data_error + div_error + recovery_error:
            raise AssertionError("原坐标恢复精确恒等式失败")
        for name, value in zip(out, (unscaled, bound, base, data_error, div_error, recovery_error, rc)):
            out[name][col] = value
    absolute = hex_fraction(static_scalars["dual_absolute"])
    relative = hex_fraction(static_scalars["dual_relative"])
    invalid_columns, exceed_columns, details = [], [], []
    for original, rc in enumerate(rc_all):
        lower, upper = a["column_lower"][original], a["column_upper"][original]
        selected_infinite = (rc > 0 and not math.isfinite(lower)) or (rc < 0 and not math.isfinite(upper))
        if not selected_infinite:
            continue
        invalid_columns.append(original)
        cost = sense * Fraction.from_float(a["original_costs"][original])
        allowed = absolute + relative * max(abs(cost), abs(transpose_all[original]))
        exceeded = abs(rc) > allowed
        if exceeded:
            exceed_columns.append(original)
        domain = "free" if not math.isfinite(lower) and not math.isfinite(upper) else "lower_only" if math.isfinite(lower) else "upper_only"
        details.append((original, rc, allowed, exceeded, domain))
    invalid_rows = [row for row, value in enumerate(actual_y) if
                    (value > 0 and not math.isfinite(a["row_lower"][row])) or
                    (value < 0 and not math.isfinite(a["row_upper"][row]))]
    details.sort(key=lambda item: (-abs(item[1]), item[0]))
    domain = {"definition": "原保存 x/y 数据的精确参考；列指向无穷界时才计残差，行乘子域无容差。",
              "column_domain_indices": invalid_columns, "column_exceeds_tolerance_indices": exceed_columns,
              "row_domain_indices": invalid_rows,
              "max_dual_residual": packed(max((abs(rc_all[i]) for i in invalid_columns), default=ZERO)),
              "dual_feasible_reference": not invalid_rows and not exceed_columns,
              "all_selected_bounds_finite": not invalid_rows and not invalid_columns,
              "largest_violations": [{"original_column": i, "working_column": original_to_work.get(i),
                  "domain": kind, "reduced_cost": packed(rc), "allowed": packed(allowed),
                  "exceeds_tolerance": exceeded} for i, rc, allowed, exceeded, kind in details[:16]],
              "largest_violations_omitted": max(0, len(details) - 16),
              "eliminated_columns_without_working_residual": [i for i in range(on) if i not in original_to_work]}
    return out, domain


def decimal_coordinates(a, step, scalars, exact, precision):
    """独立从保存的 double 转 Decimal；不使用 Fraction 主计算的中间结果。"""
    with localcontext() as context:
        context.prec = precision
        d = Decimal.from_float
        scalar = {key: dec(hex_fraction(value), precision) for key, value in scalars.items()}
        tau, nt, theta, alpha, dt = (scalar[key] for key in ("tau", "next_tau", "theta", "step", "dtau"))
        n = len(a["costs"])
        entries = []
        for col in selected_columns(n):
            c = d(a["costs"][col])
            rd, rn, eu, ev, ed, recon, update = c * tau, c * nt, -d(step["rhs_u"][col]), c, c * dt, Decimal(0), Decimal(0)
            for k in range(a["g_starts"][col], a["g_starts"][col + 1]):
                row, g = a["g_rows"][k], d(a["g_values"][k])
                oz, nz, uz, vz, dz = (d(step[key][index]) for key, index in
                                     (("old_z", row), ("new_z", row), ("u", n + row), ("v", n + row), ("dz", row)))
                rd += g * oz
                rn += g * nz
                eu += g * uz
                ev += g * vz
                ed += g * dz
                recon += g * (dz - uz - vz * dt)
                update += g * (nz - oz - alpha * dz)
            ed += theta * rd
            rhs_round = d(step["rhs_u"][col]) + theta * d(step["stored_rd"][col])
            residual_round = theta * (rd - d(step["stored_rd"][col]))
            cost_round = c * (nt - tau - alpha * dt)
            predicted = (1 - alpha * theta) * rd + alpha * ed + update + cost_round
            values = dict(rd_true=rd, rd_new=rn, e_u=eu, e_v=ev, dtau_e_v=dt * ev, rho_rhs=rhs_round,
                          transpose_rho_reconstruction=recon, theta_residual_rounding=residual_round, E_d=ed,
                          transpose_rho_update_z=update, cost_rho_update_tau=cost_round, predicted_rd_new=predicted)
            discrepancies = {}
            passed = True
            for name, value in values.items():
                reference = dec(exact[name][col], precision)
                difference = abs(value - reference)
                allowance = Decimal(10) ** (-(precision - 12)) * max(Decimal(1), abs(reference), abs(rd), abs(rn))
                passed = passed and difference <= allowance
                discrepancies[name] = str(difference)
            entries.append({"column": col, "values": {key: str(value) for key, value in values.items()},
                            "absolute_discrepancies": discrepancies, "passed": passed})
        return {"precision": precision, "coordinates": entries, "passed": all(x["passed"] for x in entries),
                "meaning": "有限精度交叉稳定性检查；精确恒等式和符号以 Fraction 为准。"}


def kkt_reference(upper, lower, rhs):
    rhs_norm = max(map(abs, rhs), default=0.0)
    # 原表达式的 binary64 参考值；未冒称重新执行生产 long double 残差累加。
    threshold = Fraction.from_float(1e-12 + 1e-10 * rhs_norm)
    residual = max(map(abs, upper + lower), default=ZERO)
    return {"upper": vector_summary(upper), "lower": vector_summary(lower), "global_residual": packed(residual),
            "binary64_threshold_reference": packed(threshold), "exact_residual_passes_reference": residual <= threshold,
            "note": "这是保存矩阵与返回向量的精确参考，不等同于生产残差的逐次舍入重放。"}


def analyze_step(a, static_scalars, step_meta, step):
    start = time.perf_counter()
    working, lower_u, lower_v, intermediate = exact_working(a, step, step_meta["scalars"])
    original, domain = original_coordinates(a, static_scalars, step, step_meta["scalars"], working)
    n = len(a["costs"])
    fixed = selected_columns(n)
    explanatory = sorted(set(fixed + [max(range(n), key=lambda i: abs(working[key][i])) for key in ("E_d", "rd_normalized")] )) if n else []
    coordinates = lambda indexes: [{"column": col, "original_column": (a["original_columns"] or list(range(n)))[col],
        "exact": {name: packed(values[col]) for name, values in working.items()},
        "original_exact": {name: packed(values[col]) for name, values in original.items()}} for col in indexes]
    norm = lambda name: max(map(abs, working[name]), default=ZERO)
    components = ("e_u", "dtau_e_v", "rho_rhs", "transpose_rho_reconstruction", "theta_residual_rounding")
    component_sum = sum((norm(name) for name in components), ZERO)
    ednorm = norm("E_d")
    cross = [decimal_coordinates(a, step, step_meta["scalars"], working, precision) for precision in (80, 120)]
    if not all(record["passed"] for record in cross):
        raise AssertionError("固定坐标的 Decimal 交叉检查未通过")
    assessment = step_meta.get("assessment", {})
    return {"iteration": step_meta["iteration"], "assessment_id": assessment.get("assessment_id"),
            "replace": assessment.get("replace"), "production_checks": assessment.get("checks"),
            "scalars": step_meta["scalars"], "exact_identity_all_coordinates": True,
            "vectors": {key: vector_summary(values) for key, values in working.items()},
            "original_vectors": {key: vector_summary(values) for key, values in original.items()},
            "kkt_u": kkt_reference(working["e_u"], lower_u, step["rhs_u"]),
            "kkt_v": kkt_reference(working["e_v"], lower_v, [-x for x in a["costs"]] + a["b"]),
            "rho_update_tau": packed(intermediate["rho_update_tau"]),
            "direction_reconstruction": vector_summary(intermediate["rho_reconstruction"]),
            "update_z_rounding": vector_summary(intermediate["rho_update_z"]),
            "amplification": {"dtau_absolute": packed(abs(intermediate["dtau"])),
                "sum_component_infinity_bounds": packed(component_sum),
                "component_bound_over_Ed": packed(component_sum / ednorm) if ednorm else None,
                "dtau_ev_norm_over_Ed": packed(norm("dtau_e_v") / ednorm) if ednorm else None,
                "note": "范数比可因抵消超过一，不是可相加的因果百分比。"},
            "fixed_coordinates": coordinates(fixed), "explanatory_coordinates": coordinates(explanatory),
            "decimal_crosschecks": cross, "original_domain_reference": domain,
            "analysis_seconds": time.perf_counter() - start}


def analyze(path):
    path = Path(path)
    meta, a, steps, binary = load_snapshot(path)
    result = {"metadata_path": str(path.resolve()), "metadata_sha256": digest(path),
              "capture_status": meta.get("status"), "retained_assessment": meta.get("retained_assessment"),
              "latest_assessment": meta.get("latest_assessment"),
              "last_attempt": {key: value for key, value in (meta.get("last_attempt") or {}).items() if key != "arrays"}}
    if a is None:
        return dict(result, analysis_status="not_captured", steps=[])
    result.update(binary_sha256=digest(binary), dimensions={"n": len(a["costs"]), "m": len(a["b"]),
                  "nnz": len(a["g_values"]), "original_n": len(a["original_costs"]), "original_m": len(a["row_lower"])})
    result["steps"] = [analyze_step(a, meta["static"]["scalars"], step_meta, values) for step_meta, values in steps]
    retained = meta.get("retained_assessment") or {}
    for step in result["steps"]:
        step["is_final_retained_assessment"] = bool(retained.get("available") and
            step["assessment_id"] == retained.get("assessment_id"))
    result["analysis_status"] = "analyzed" if steps else "no_completed_step"
    result["scope"] = [
        "只解释已经保存的完整 corrector；未完成末次尝试不补造方向或推断其失败原因。",
        "全部 Fraction 等式针对保存的矩阵和向量，不是未知理想解或完整求解器的精确正确性证明。",
        "生产 checks、replace 和最终 retained_assessment 原样并列；当前点不能替代保留候选。",
        "原坐标数据差异含规范化与多轮缩放舍入；列界锥贡献单列，不能把所有约化成本要求为零。",
        "原 checks 的 long double 累加和精确参考可能在符号或门槛边界不同；不覆盖生产状态。",
        "恒等式说明误差传递，不独立证明某个后端、分解步骤或未来精度政策的因果收益。",
    ]
    return result


def self_test():
    checks = 0
    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("v53 离线合成自测失败")
        checks += 1
    check(hex_fraction("0x8.000000000000001p-3") == 1 + Fraction(1, 1 << 63))
    check(hex_fraction("-0x1p-1074") == -Fraction(1, 1 << 1074))
    check(hex_fraction("-0x0p0") == 0)
    for bad in ("nan", "inf", "1.0", "--0x1p0", "0x1p999999", "0x1p0extra"):
        try:
            hex_fraction(bad)
        except ValueError:
            check(True)
        else:
            check(False)
    a = {"g_starts": [0, 2, 4], "g_rows": [0, 1, 0, 2], "g_values": [-2.0, -4.0, 2.0, 16.0],
         "costs": [0.75, -2.5], "b": [-1.0, 0.0, 12.0], "inequality": [1, 1, 1],
         "original_row": [0, -1, -1], "dual_sign": [1.0, 1.0, -1.0], "row_scale": [0.5, 2.0, 4.0],
         "column_scale": [2.0, 4.0], "original_columns": [], "fixed_values": [], "a_starts": [0, 1, 2],
         "a_rows": [0, 0], "a_values": [2.0, -1.0], "original_costs": [3.0, -5.0],
         "column_lower": [0.0, -math.inf], "column_upper": [math.inf, 3.0], "row_lower": [2.0], "row_upper": [math.inf]}
    static = {key: value.hex() for key, value in dict(objective_scale=8.0, sense=1.0, objective_offset=0.0,
                                                    dual_absolute=1e-7, dual_relative=1e-9).items()}
    s = {"old_x": [1.0, 2.0], "new_x": [2.0, 3.0], "old_z": [2.0, 3.0, 4.0],
         "new_z": [1.75, 3.5, 4.25], "stored_rd": [-14.5, 63.0], "rhs_u": [10.875, -47.25, 0.0, 1.0, 2.0],
         "diagonal_h": [1.0, 2.0, 3.0], "u": [1.0, 2.0, -1.0, 1.5, 0.0],
         "v": [3.0, 4.0, 1.0, -1.0, 1.0], "dz": [-0.5, 1.0, 0.5],
         "extracted_z": [1.75 / 2.25, 3.5 / 2.25, 4.25 / 2.25], "assess_x": [2.0, 3.0],
         "assess_y": [4.0 * (1.75 / 2.25)]}
    scalars = {key: value.hex() for key, value in dict(tau=2.0, next_tau=2.25, theta=0.75, step=0.5, dtau=0.5).items()}
    validate_step(s, scalars, validate_static(a, static))
    w, lu, lv, extra = exact_working(a, s, scalars)
    check(all(w["rd_new"][i] == w["predicted_rd_new"][i] for i in range(2)))
    check(extra["rho_update_tau"] == 0 and not any(extra["rho_reconstruction"]))
    orig, domain = original_coordinates(a, static, s, scalars, w)
    check(not any(orig["data_relation_difference"]))
    check(domain["column_domain_indices"] == [0] and domain["column_exceeds_tolerance_indices"] == [0])
    check(domain["largest_violations"][0]["domain"] == "lower_only")
    check(not domain["row_domain_indices"])
    check(any(orig["tau_division_rounding"]))
    for precision in (80, 120):
        check(decimal_coordinates(a, s, scalars, w, precision)["passed"])
    changed = dict(s, dz=[math.nextafter(-0.5, 0.0), 1.0, 0.5],
                   new_z=[math.nextafter(1.75, math.inf), 3.5, 4.25], stored_rd=[-14.0, 63.0])
    cw, _, _, ce = exact_working(a, changed, scalars)
    check(any(ce["rho_reconstruction"]) and any(ce["rho_update_z"]))
    check(any(cw["theta_residual_rounding"]) and any(cw["rho_rhs"]))
    data_changed = dict(a, g_values=[math.nextafter(-2.0, 0.0), -4.0, 2.0, 16.0])
    dw, _, _, _ = exact_working(data_changed, s, scalars)
    dor, _ = original_coordinates(data_changed, static, s, scalars, dw)
    check(any(dor["data_relation_difference"]))
    actual_changed = dict(s, assess_y=[math.nextafter(s["assess_y"][0], math.inf)])
    yor, _ = original_coordinates(a, static, actual_changed, scalars, w)
    check(any(yor["original_y_recovery_rounding"]))
    both_finite = dict(a, column_lower=[0.0, -1.0], column_upper=[10.0, 3.0])
    _, finite_domain = original_coordinates(both_finite, static, s, scalars, w)
    check(not finite_domain["column_domain_indices"])
    free = dict(a, column_lower=[-math.inf, -math.inf], column_upper=[math.inf, math.inf])
    _, free_domain = original_coordinates(free, static, s, scalars, w)
    check(free_domain["column_domain_indices"] == [0, 1])
    bad_y = dict(s, assess_y=[-1e-300])
    _, row_domain = original_coordinates(a, static, bad_y, scalars, w)
    check(row_domain["row_domain_indices"] == [0])
    check(selected_columns(0) == [] and selected_columns(1) == [0] and selected_columns(2) == [0, 1])
    return {"passed": True, "checks": checks, "kind": "pure_synthetic_no_solver"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    result = {"schema": "v53_hsd_analysis_1", "analyzer_sha256": digest(__file__), "self_test": self_test(), "records": []}
    for path in args.inputs:
        try:
            result["records"].append(analyze(path))
        except Exception as error:
            result["records"].append({"metadata_path": str(Path(path).resolve()), "analysis_status": "error",
                                      "error": f"{type(error).__name__}: {error}"})
    result["passed"] = all(record["analysis_status"] != "error" for record in result["records"])
    result["analysis_seconds"] = time.perf_counter() - start
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() in {Path(path).resolve() for path in args.inputs}:
            raise ValueError("分析输出不能覆盖输入元数据")
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
