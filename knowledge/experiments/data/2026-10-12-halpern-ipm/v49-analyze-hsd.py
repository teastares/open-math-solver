"""只读分析 v49 HSD 快照；不调用求解器，不把有限精度复算当精确符号证明。"""

import argparse
import hashlib
import json
import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

NAMES = ("starts", "rows", "values", "costs", "rhs", "s", "z", "diagonal_h", "regularized_h", "u", "v", "inequality")
FORMATS = {"i64_le": ("q", 8), "f64_le": ("d", 8), "u8": ("B", 1)}
CAP = 128 * 1024 * 1024


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hex_fraction(value):
    """按文本的全部二进制位解码 C/C++ hexfloat，不经过 Python double。"""
    if not isinstance(value, str) or value.lower() in {"nan", "inf", "+inf", "-inf"}:
        raise ValueError("要求有限的十六进制浮点文本")
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-").lower()
    if not value.startswith("0x") or "p" not in value:
        raise ValueError("十六进制浮点文本格式无效")
    mantissa, exponent = value[2:].split("p")
    whole, _, fractional = mantissa.partition(".")
    integer = int(whole + fractional, 16) * sign
    shift = int(exponent) - 4 * len(fractional)
    return Fraction(integer << shift, 1) if shift >= 0 else Fraction(integer, 1 << -shift)


def decimal_value(value, precision=80):
    with localcontext() as ctx:
        ctx.prec = precision
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def packed(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "decimal80": decimal_value(value), "sign": (value > 0) - (value < 0)}


def load_snapshot(path):
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != "v49_hsd_1":
        raise ValueError("HSD 快照 schema 不符")
    if meta.get("component_cache_limit") != CAP or meta.get("metadata_reserve") != 65536:
        raise ValueError("HSD 缓存额度不符")
    if meta.get("extra_lp_products") != 0 or meta.get("extra_lp_solves") != 0:
        raise ValueError("HSD 捕获不应增加 LP 乘积或求解")
    if not meta.get("captured"):
        return meta, None, None
    binary_path = path.with_suffix(".bin")
    binary = binary_path.read_bytes()
    if binary[:8] != b"V49HSD1\0" or len(binary) != 8 + meta["payload_bytes"]:
        raise ValueError("HSD binary 长度或魔数错误")
    if meta["payload_bytes"] + meta["metadata_reserve"] > CAP:
        raise ValueError("HSD 捕获超过额度")
    arrays, cursor = {}, 8
    if tuple(f["name"] for f in meta["arrays"]) != NAMES:
        raise ValueError("HSD 数组顺序或名称不符")
    for field in meta["arrays"]:
        name, dtype, count, offset = field["name"], field["dtype"], field["count"], field["offset"]
        expected = "i64_le" if name in {"starts", "rows"} else "u8" if name == "inequality" else "f64_le"
        if dtype != expected or type(count) is not int or count < 0 or offset != cursor:
            raise ValueError("HSD 数组索引不符")
        code, width = FORMATS[dtype]
        stop = cursor + width * count
        if stop > len(binary):
            raise ValueError("HSD 数组越界")
        arrays[name] = list(struct.unpack_from(f"<{count}{code}", binary, cursor))
        cursor = stop
    if cursor != len(binary):
        raise ValueError("HSD binary 含未声明尾部")
    validate_arrays(arrays)
    return meta, arrays, binary_path


def validate_arrays(a):
    n, m = len(a["costs"]), len(a["rhs"])
    if len(a["starts"]) != n + 1 or not a["starts"] or a["starts"][0] != 0:
        raise ValueError("HSD CSC 列指针无效")
    if any(x > y for x, y in zip(a["starts"], a["starts"][1:])):
        raise ValueError("HSD CSC 列指针不单调")
    if a["starts"][-1] != len(a["values"]) or len(a["rows"]) != len(a["values"]):
        raise ValueError("HSD CSC nnz 不符")
    if any(r < 0 or r >= m for r in a["rows"]):
        raise ValueError("HSD CSC 行号越界")
    if any(len(a[k]) != m for k in ("s", "z", "diagonal_h", "regularized_h", "inequality")):
        raise ValueError("HSD 行向量维数不符")
    if any(len(a[k]) != n + m for k in ("u", "v")) or any(x not in (0, 1) for x in a["inequality"]):
        raise ValueError("HSD 方向或不等式掩码维数不符")
    for name in NAMES:
        if name in {"starts", "rows", "inequality"}:
            continue
        if any(not math.isfinite(x) for x in a[name]):
            raise ValueError(f"HSD 数组 {name} 含非有限数，无法做有理数恒等式")


def exact_analysis(a, scalars):
    f = {key: [Fraction.from_float(x) for x in a[key]] for key in
         ("values", "costs", "rhs", "diagonal_h", "regularized_h", "u", "v")}
    n, m = len(f["costs"]), len(f["rhs"])
    vx, vz = f["v"][:n], f["v"][n:]
    c, b, h, hr = (f[key] for key in ("costs", "rhs", "diagonal_h", "regularized_h"))
    tau, kappa, dp = (hex_fraction(scalars[k]) for k in ("tau", "kappa", "primal_regularization"))
    if tau <= 0 or kappa <= 0:
        raise ValueError("齐次标量不是严格正数")
    ed = c.copy()
    ep = [-h[i] * vz[i] - b[i] for i in range(m)]
    for col in range(n):
        for k in range(a["starts"][col], a["starts"][col + 1]):
            row, g = a["rows"][k], f["values"][k]
            ed[col] += g * vz[row]
            ep[row] += g * vx[col]
    zero = Fraction(0)
    dot_v = sum((c[i] * vx[i] for i in range(n)), zero) + sum((b[i] * vz[i] for i in range(m)), zero)
    dot_u = sum((x * y for x, y in zip(c + b, f["u"])), zero)
    q = kappa / tau
    denominator = q - dot_v
    energy = sum((h[i] * vz[i] ** 2 for i in range(m)), zero)
    defect = sum((vx[i] * ed[i] for i in range(n)), zero) - sum((vz[i] * ep[i] for i in range(m)), zero)
    rho = [hr[i] - h[i] for i in range(m)]
    regularization_energy = dp * sum((x * x for x in vx), zero) + sum((rho[i] * vz[i] ** 2 for i in range(m)), zero)
    erd = [ed[i] + dp * vx[i] for i in range(n)]
    erp = [ep[i] - rho[i] * vz[i] for i in range(m)]
    regularized_defect = sum((vx[i] * erd[i] for i in range(n)), zero) - sum((vz[i] * erp[i] for i in range(m)), zero)
    identity_error = denominator - (q + energy - defect)
    regularized_identity_error = denominator - (q + energy + regularization_energy - regularized_defect)
    if identity_error or regularized_identity_error or defect != regularized_defect - regularization_energy:
        raise AssertionError("精确齐次能量恒等式不成立")
    residual_inf = max(map(abs, ed + ep), default=zero)
    regularized_residual_inf = max(map(abs, erd + erp), default=zero)
    projection_bound = sum((abs(x) for x in vx + vz), zero) * residual_inf
    values = dict(q=q, dot_v=dot_v, dot_u=dot_u, denominator=denominator, energy=energy,
                  positive_energy=q + energy, defect=defect, regularization_energy=regularization_energy,
                  regularized_defect=regularized_defect, residual_inf=residual_inf,
                  regularized_residual_inf=regularized_residual_inf, projection_bound=projection_bound,
                  identity_error=identity_error, regularized_identity_error=regularized_identity_error,
                  min_effective_dual_shift=min(rho, default=zero))
    if scalars.get("scalar_values_available", True):
        try:
            production_dot = hex_fraction(scalars["dot_v"])
            production_denominator = hex_fraction(scalars["denominator"])
            dot_rounding = production_dot - dot_v
            ratio_subtraction_rounding = production_denominator - (q - production_dot)
            total_rounding = production_denominator - denominator
            if total_rounding != ratio_subtraction_rounding - dot_rounding:
                raise AssertionError("精确标量舍入分解不成立")
            values.update(production_dot_v=production_dot, production_denominator=production_denominator,
                          dot_rounding=dot_rounding, ratio_subtraction_rounding=ratio_subtraction_rounding,
                          total_scalar_rounding=total_rounding)
        except ValueError:
            # 非有限原值本身就是证据，不能以 Decimal/Fraction 伪造有限结果。
            pass
    return {"scalars": {key: packed(value) for key, value in values.items()},
            "identity_exact": True, "regularized_identity_exact": True,
            "nonnegative_h_and_regularization": dp >= 0 and all(x >= 0 for x in h + rho),
            "residual_definition": "e0=K0*v-[-c;b]；与生产 rhs-K0*v 的残差向量反号",
            "regularized_residual_definition": "eR=e0+diag(delta_p,-(regularized_h-h))*v；包含返回方向及消元/舍入影响，不等于单独分解误差"}


def decimal_analysis(a, scalars, precision):
    with localcontext() as ctx:
        ctx.prec = precision
        f = {key: [Decimal.from_float(x) for x in a[key]] for key in
             ("values", "costs", "rhs", "diagonal_h", "regularized_h", "v")}
        n, m = len(f["costs"]), len(f["rhs"])
        vx, vz = f["v"][:n], f["v"][n:]
        c, b, h = (f[key] for key in ("costs", "rhs", "diagonal_h"))
        qf = hex_fraction(scalars["kappa"]) / hex_fraction(scalars["tau"])
        q = Decimal(qf.numerator) / Decimal(qf.denominator)
        ed, ep = c.copy(), [-h[i] * vz[i] - b[i] for i in range(m)]
        for col in range(n):
            for k in range(a["starts"][col], a["starts"][col + 1]):
                row, g = a["rows"][k], f["values"][k]
                ed[col] += g * vz[row]
                ep[row] += g * vx[col]
        dot = sum((x * y for x, y in zip(c + b, vx + vz)), Decimal(0))
        energy = sum((h[i] * vz[i] ** 2 for i in range(m)), Decimal(0))
        defect = sum((x * y for x, y in zip(vx, ed)), Decimal(0)) - sum((x * y for x, y in zip(vz, ep)), Decimal(0))
        return {"precision": precision, "dot_v": str(dot), "denominator": str(q - dot),
                "energy": str(energy), "defect": str(defect), "energy_denominator": str(q + energy - defect),
                "identity_discrepancy": str((q - dot) - (q + energy - defect)),
                "sign": ((q - dot) > 0) - ((q - dot) < 0)}


def analyze(path):
    path = Path(path)
    meta, arrays, binary = load_snapshot(path)
    result = {"metadata_path": str(path.resolve()), "metadata_sha256": digest(path), "captured": meta["captured"],
              "capture_status": meta["status"], "analysis_scope": "只针对实际保存方向和矩阵，不推断理想精确解或所有后端行为"}
    if arrays is None:
        result["analysis_status"] = "not_captured"
        return result
    result["binary_sha256"] = digest(binary)
    result["dimensions"] = {"n": len(arrays["costs"]), "m": len(arrays["rhs"]), "nnz": len(arrays["values"])}
    result["event"] = {k: meta[k] for k in ("phase", "iteration", "completed_iterations", "factorizations", "refinements",
                                             "constant_solves", "direction_solves", "regularization_level", "failure")}
    scalars = dict(meta["scalars"], scalar_values_available=meta["scalar_values_available"])
    result["exact"] = exact_analysis(arrays, scalars)
    result["decimal"] = [decimal_analysis(arrays, scalars, p) for p in (80, 120)]
    values = result["exact"]["scalars"]
    exact_sign = values["denominator"]["sign"]
    production = values.get("production_denominator")
    if production is None:
        classification = "production_scalar_unavailable_or_nonfinite"
    elif exact_sign > 0 and production["sign"] <= 0:
        classification = "scalar_rounding_changes_sign_for_saved_direction"
    elif exact_sign <= 0:
        classification = "saved_direction_has_nonpositive_exact_denominator"
    else:
        classification = "positive_exact_denominator_other_scalar_failure"
    result.update(analysis_status="analyzed", classification=classification,
                  decimal_signs_match_exact=all(x["sign"] == exact_sign for x in result["decimal"]))
    result["limitations"] = [
        "精确分母是保存 binary64 数据的有理数结果；不是未知理想方向的分母，也不是所有 double 方向的不可行证明。",
        "原 H 使用实际 diagonal_h；有效对偶正则为 regularized_h-h，未以重新计算 s/z 取代原系数。",
        "返回 v 经原 K0 验收/可能改进；eR 不能单独归于正则、POTRF 或 Gram 装配。",
        "Decimal80/120 是近似交叉检查；符号和恒等式以 Fraction 为准。",
        "本分析不求解新 LP，也不把快照写入或离线精确计算称为免费工作。",
    ]
    return result


def self_test():
    checks = 0
    def check(value):
        nonlocal checks
        if not value:
            raise AssertionError("HSD 数学自测失败")
        checks += 1
    check(hex_fraction("0x8.000000000000001p-3") == Fraction(1) + Fraction(1, 1 << 63))
    check(hex_fraction("-0x1p-1074") == -Fraction(1, 1 << 1074))
    check(hex_fraction("-0x0p+0") == 0)
    for bad in ("nan", "inf", "1.0"):
        try:
            hex_fraction(bad)
        except ValueError:
            check(True)
        else:
            check(False)
    base = {"starts": [0, 1], "rows": [0], "values": [1.0], "costs": [-1.0], "rhs": [2.0],
            "s": [1.0], "z": [2.0], "diagonal_h": [0.5], "regularized_h": [0.75],
            "u": [1.5, -1.0], "v": [2.5, 1.0], "inequality": [1]}
    scalar = {"tau": "0x1p0", "kappa": "0x1p0", "primal_regularization": "0x1p-2",
              "dual_regularization": "0x1p-2", "dot_v": "-0x1p-1", "denominator": "0x1.8p0"}
    validate_arrays(base)
    r = exact_analysis(base, scalar)
    check(r["scalars"]["denominator"]["numerator"] == "3" and r["scalars"]["denominator"]["denominator"] == "2")
    check(r["scalars"]["defect"]["sign"] == 0)
    check(r["identity_exact"] and r["regularized_identity_exact"])
    check(r["nonnegative_h_and_regularization"])
    for p in (80, 120):
        d = decimal_analysis(base, scalar, p)
        check(Decimal(d["denominator"]) == Decimal("1.5") and Decimal(d["identity_discrepancy"]) == 0)
    # 保存方向本身错误，不能靠更高精度点积把负分母改成正值。
    bad = dict(base, v=[4.0, 4.0])
    r = exact_analysis(bad, dict(scalar, dot_v="0x1p2", denominator="-0x1.8p1"))
    check(r["scalars"]["denominator"]["sign"] == -1)
    check(r["scalars"]["defect"]["numerator"] == "12")
    # 精确正则系统解：K_R=[[1,1],[1,-1]]，v=(1,0)，c=-1，b=1。
    regularized = dict(base, costs=[-1.0], rhs=[1.0], diagonal_h=[0.0], regularized_h=[1.0], v=[1.0, 0.0])
    r = exact_analysis(regularized, dict(scalar, primal_regularization="0x1p0", dot_v="-0x1p0", denominator="0x1p1"))
    check(r["scalars"]["regularized_defect"]["sign"] == 0)
    check(r["scalars"]["defect"]["sign"] == -1 and r["scalars"]["denominator"]["sign"] == 1)
    # 解析构造把保存数据的正分母与生产舍入变号严格分开。
    rounded = exact_analysis(base, dict(scalar, dot_v="0x1p1", denominator="-0x1p0"))
    check(rounded["scalars"]["total_scalar_rounding"]["numerator"] == "-5")
    check(rounded["scalars"]["dot_rounding"]["numerator"] == "5")
    # 等式 H=0 且无不等式也满足同一恒等式。
    equality = dict(base, diagonal_h=[0.0], regularized_h=[0.25], v=[2.0, 1.0], inequality=[0])
    r = exact_analysis(equality, dict(scalar, dot_v="0x0p0", denominator="0x1p0"))
    check(r["scalars"]["energy"]["sign"] == 0 and r["scalars"]["defect"]["sign"] == 0)
    # 实际 regularized_h 可吞掉很小偏移，不能用请求的 delta_d 代替有效 rho。
    swallowed = dict(base, regularized_h=[0.5])
    r = exact_analysis(swallowed, scalar)
    check(r["scalars"]["min_effective_dual_shift"]["sign"] == 0)
    return {"passed": True, "checks": checks, "kind": "pure_mathematical_no_solver"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    result = {"analyzer_sha256": digest(__file__), "self_test": self_test(), "records": []}
    for path in args.inputs:
        try:
            result["records"].append(analyze(path))
        except Exception as error:
            result["records"].append({"metadata_path": str(Path(path).resolve()), "analysis_status": "error",
                                      "error": f"{type(error).__name__}: {error}"})
    result["passed"] = all(r["analysis_status"] != "error" for r in result["records"])
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
