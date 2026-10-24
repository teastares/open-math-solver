"""在捕获的固定空间中作离线无穷范数搜索，并精确复核原 K0；不修改生产求解器。"""

import argparse
from array import array
import datetime
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time


DATA = Path(__file__).resolve().parent
ROOT = DATA.parents[3]
PLAN_SHA = "0dafab3cc269cec22abe22bb2dbff014dd7a8d2d45255acd54ac3bfad0f42c8b"
HELPER_SHA = "d5a2203c6ec8799f8cb15723c29635fce06aa1495b4c1482d6f70614d4b9aace"
MAGIC = b"V67KRY01"
CAP = 128 * 1024 * 1024
NAMES = ("g_starts", "g_rows", "g_values", "h", "linear_rhs", "base", "z",
         "coefficients", "last_candidate", "last_residual")
AVAILABLE = ("base_available", "z_available", "coefficients_available", "last_candidate_available",
             "last_residual_available", "linear_rhs_available")
ZERO = Fraction(0)
helper_path = DATA / "v53-analyze-hsd.py"
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError("冻结的 v53 精确解析辅助来源改变")
spec = importlib.util.spec_from_file_location("v67_exact_helpers", helper_path)
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)
packed, hex_fraction = BASE.packed, BASE.hex_fraction


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=(1 << 63) - 1):
    require(type(value) is int and low <= value <= high, "整数范围或类型无效")
    return value


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, "JSON 重复键：" + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("非有限 JSON：" + value)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def load_snapshot(path):
    path = Path(path).resolve()
    metadata = read_json(path)
    require(type(metadata.get("schema")) is int and metadata["schema"] == 1, "快照 schema 不匹配")
    require(metadata.get("status") in ("captured", "no_target_failure", "capture_error", "not_started"), "未知采集状态")
    binary_path = path.with_suffix(".bin")
    blob = binary_path.read_bytes()
    require(len(blob) >= 8 and len(blob) <= CAP + 8 and blob[:8] == MAGIC, "快照二进制头或尺寸无效")
    require(integer(metadata.get("binary_bytes")) == len(blob), "二进制长度声明不符")
    require(metadata.get("byte_order") == "little", "本轮捕获要求小端字节序")
    require(type(metadata.get("truncated")) is bool, "截断标记类型无效")
    errors = integer(metadata.get("capture_errors"))
    require(isinstance(metadata.get("error"), str), "采集错误描述类型无效")
    count = integer(metadata.get("event_count"), 0, 1)
    require(isinstance(metadata.get("events"), list) and len(metadata["events"]) == count, "事件数不匹配")
    identity = {"metadata": str(path), "metadata_sha256": digest(path), "binary": str(binary_path),
                "binary_sha256": hashlib.sha256(blob).hexdigest(), "status": metadata["status"]}
    if metadata["status"] != "captured":
        require(count == 0 and len(blob) == 8, "非 captured 不能携带完整事件")
        return identity, metadata, None
    require(count == 1 and not metadata["truncated"] and errors == 0 and metadata["error"] == "",
            "不完整或有采集错误的事件不能进入离线搜索")
    event = metadata["events"][0]
    require(isinstance(event, dict) and event.get("phase") == "ordinary_constant", "只分析普通常数方向")
    require(event.get("linear_exit") == "krylov_exhausted" and event.get("terminal") is True and
            event.get("candidate_submitted") is False and event.get("residual_sign") == "rhs_minus_K0_candidate",
            "只分析最终耗尽且未提交的原残差候选")
    require(all(event.get(key) is True for key in AVAILABLE), "捕获缺少必要数组")
    n, m = integer(event.get("n")), integer(event.get("m"))
    dimension = integer(event.get("dimension"), 1)
    require(dimension == n + m, "K0 维数不匹配")
    nnz = integer(event.get("nnz"))
    k = integer(event.get("generated_columns"), 1, 32)
    cc = integer(event.get("coefficient_count"), 1, k)
    require(integer(event.get("restart_index"), 0, 2) == 2 and
            integer(event.get("column_index"), 0, 31) == cc - 1, "不是最后周期实际候选")
    integer(event.get("iteration"))
    descriptors = event.get("arrays")
    require(isinstance(descriptors, list) and all(isinstance(x, dict) for x in descriptors) and
            [x.get("name") for x in descriptors] == list(NAMES), "数组顺序不匹配")
    counts = (n + 1, nnz, nnz, m, dimension, dimension, k * dimension, cc, dimension, dimension)
    arrays, cursor = {}, 8
    for index, (name, expected, descriptor) in enumerate(zip(NAMES, counts, descriptors)):
        dtype = "int64" if index < 2 else "float64"
        shape = [k, dimension] if name == "z" else [expected]
        require(descriptor.get("dtype") == dtype and integer(descriptor.get("width")) == 8 and
                integer(descriptor.get("count")) == expected and integer(descriptor.get("offset")) == cursor and
                isinstance(descriptor.get("shape"), list) and
                [integer(x) for x in descriptor["shape"]] == shape, "数组描述不匹配：" + name)
        end = cursor + 8 * expected
        require(end <= len(blob), "数组越界：" + name)
        values = array("q" if index < 2 else "d")
        require(values.itemsize == 8, "本平台数组元素不是八字节")
        values.frombytes(memoryview(blob)[cursor:end])
        if sys.byteorder != "little":
            values.byteswap()
        if index >= 2:
            require(all(math.isfinite(x) for x in values), "非有限捕获数组：" + name)
        arrays[name] = values
        cursor = end
    require(cursor == len(blob) and integer(event.get("payload_bytes")) == len(blob) - 8, "快照尾部或负载不匹配")
    fixed = integer(metadata.get("fixed_cache_bytes"))
    require(integer(event.get("cache_bytes")) == fixed + len(blob) - 8 and fixed + len(blob) - 8 <= CAP,
            "采集缓存账本不符")
    BASE.validate_csc(arrays["g_starts"], arrays["g_rows"], arrays["g_values"], n, m, True)
    require(all(x >= 0 for x in arrays["h"]), "实际 H 不能为负")
    tolerance = hex_fraction(event.get("global_tolerance"))
    require(tolerance > 0 and Fraction.from_float(float(tolerance)) == tolerance, "原 global 容差必须是有限正 double")
    blocks = event.get("block_tolerances")
    if blocks is not None:
        require(isinstance(blocks, list) and len(blocks) == 2, "块容差维数不匹配")
        blocks = [hex_fraction(x) for x in blocks]
        require(all(x > 0 and Fraction.from_float(float(x)) == x for x in blocks), "块容差必须为有限正 double")
    fp = metadata.get("floating_point")
    require(isinstance(fp, dict) and isinstance(fp.get("double"), dict) and isinstance(fp.get("long_double"), dict),
            "缺少浮点格式元数据")
    double, wide = fp["double"], fp["long_double"]
    require(all(integer(double.get(key), -32768, 32768) == value for key, value in
                (("bytes", 8), ("digits", 53), ("min_exponent", -1021), ("max_exponent", 1024))), "要求 binary64")
    require(double.get("is_iec559") is True and wide.get("is_iec559") is True, "要求 IEC559 舍入格式")
    precision = integer(wide.get("digits"), 53, 113)
    integer(wide.get("bytes"), 8, 32)
    minimum = integer(wide.get("min_exponent"), -32768, -1021)
    maximum = integer(wide.get("max_exponent"), 1024, 32768)
    require(isinstance(metadata.get("rounding_mode"), dict) and metadata["rounding_mode"].get("is_to_nearest") is True,
            "本轮逐步重放要求最近偶数舍入")
    integer(metadata["rounding_mode"].get("value"), -1, 65535)
    require(integer(fp.get("flt_eval_method"), -1, 2) == 0, "要求表达式按声明精度计算，不接受额外求值精度")
    allowed = [min(tolerance, blocks[0] if blocks and i < n else blocks[1] if blocks else tolerance)
               for i in range(dimension)]
    numerical_hash = hashlib.sha256(blob)
    numerical_hash.update(json.dumps({"n": n, "m": m, "k": k, "cc": cc,
        "global": str(tolerance), "blocks": [str(x) for x in blocks] if blocks else None,
        "wide": [precision, minimum, maximum]}, sort_keys=True).encode())
    return identity, metadata, {"event": event, "arrays": arrays, "n": n, "m": m,
        "dimension": dimension, "k": k, "cc": cc, "allowed": allowed,
        "wide_format": (precision, minimum, maximum), "numerical_sha256": numerical_hash.hexdigest()}


def power2(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def floor_log2(value):
    require(value > 0, "二进制指数要求正数")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    return exponent


def rounded_binary(value, precision=53, minimum=-1021, maximum=1024):
    """把精确有理数舍入到指定二进制格式，包含次正规和最近偶数边界。"""
    if not value:
        return ZERO
    negative, magnitude = value < 0, abs(value)
    exponent = floor_log2(magnitude)
    if exponent > maximum:
        raise OverflowError("逐步浮点运算溢出")
    quantum = power2(max(exponent - precision + 1, minimum - precision))
    scaled = magnitude / quantum
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or twice == scaled.denominator and quotient % 2:
        quotient += 1
    answer = quotient * quantum
    if answer >= power2(maximum):
        raise OverflowError("逐步浮点运算溢出")
    return -answer if negative else answer


def cast64(value, negative_zero=False):
    rounded = rounded_binary(value)
    if not rounded:
        return -0.0 if value < 0 or negative_zero else 0.0
    return float(rounded)


def negative(value):
    return math.copysign(1.0, value) < 0


def production_combination(base, z, coefficients, wide_format):
    """复现原 j 外层、i 内层的 double 乘积、宽加法和 double 写回。"""
    candidate = list(base)
    dimension = len(base)
    for j, coefficient in enumerate(coefficients):
        for i in range(dimension):
            direction = z[j * dimension + i]
            product = rounded_binary(Fraction.from_float(coefficient) * Fraction.from_float(direction))
            product_negative_zero = not product and negative(coefficient) != negative(direction)
            total = Fraction.from_float(candidate[i]) + product
            summed = rounded_binary(total, *wide_format)
            zero_sign = not total and candidate[i] == 0 and not product and negative(candidate[i]) and product_negative_zero
            candidate[i] = cast64(summed, negative_zero=zero_sign or total < 0 and not summed)
    return candidate


def continuous_combination(base, z, coefficients):
    result = [Fraction.from_float(x) for x in base]
    dimension = len(base)
    for j, coefficient in enumerate(coefficients):
        for i in range(dimension):
            result[i] += coefficient * Fraction.from_float(z[j * dimension + i])
    return result


def exact_operator(data, point):
    a, n, m = data["arrays"], data["n"], data["m"]
    result = [ZERO] * n + [-Fraction.from_float(a["h"][r]) * point[n + r] for r in range(m)]
    for col in range(n):
        for entry in range(a["g_starts"][col], a["g_starts"][col + 1]):
            row = n + a["g_rows"][entry]
            coefficient = Fraction.from_float(a["g_values"][entry])
            result[col] += coefficient * point[row]
            result[row] += coefficient * point[col]
    return result


def exact_residual(data, point):
    return [Fraction.from_float(rhs) - product for rhs, product in
            zip(data["arrays"]["linear_rhs"], exact_operator(data, point))]


def residual_summary(data, residual):
    ratio = [abs(value) / allowed for value, allowed in zip(residual, data["allowed"])]
    worst = max(range(len(ratio)), key=ratio.__getitem__) if ratio else None
    return {"satisfies_original_tolerances": all(x <= 1 for x in ratio),
        "maximum_normalized_residual": packed(ratio[worst] if worst is not None else ZERO),
        "worst_coordinate": worst, "signed_residual_at_worst": packed(residual[worst] if worst is not None else ZERO),
        "upper_infinity": BASE.vector_summary(residual[:data["n"]]),
        "lower_infinity": BASE.vector_summary(residual[data["n"]:])}


def model_number(value, stats, category):
    converted = cast64(value)
    require(not value or converted != 0, "诊断 LP 非零系数转 double 下溢")
    error = abs(Fraction.from_float(converted) - value)
    stats[category + "_max_absolute_rounding_error"] = max(stats[category + "_max_absolute_rounding_error"], error)
    stats[category + "_inexact_count"] += int(error != 0)
    if value and abs(converted) <= 1e-9:
        stats["nonzero_at_or_below_highs_default_small_matrix_value"] += int(category == "coefficient")
    return format(converted, ".17g")


def write_model(data, center_residual, destination):
    """两侧约束全保留，按列流式计算精确 K0Z；只对变量作二次幂缩放。"""
    start = time.perf_counter()
    projection_seconds = 0.0
    nrows, k = data["dimension"], data["k"]
    stats = {"coefficient_max_absolute_rounding_error": ZERO, "rhs_max_absolute_rounding_error": ZERO,
        "coefficient_inexact_count": 0, "rhs_inexact_count": 0,
        "nonzero_at_or_below_highs_default_small_matrix_value": 0,
        "written_matrix_nonzeros": 2 * nrows, "zero_projected_columns": 0}
    exponents = []
    with Path(destination).open("x", encoding="ascii", newline="\n") as target:
        target.write("NAME V67_DIAGNOSTIC\nROWS\n N OBJ\n")
        for i in range(nrows):
            target.write(f" L P{i}\n L N{i}\n")
        target.write("COLUMNS\n")
        for j in range(k):
            projected_start = time.perf_counter()
            point = [Fraction.from_float(x) for x in data["arrays"]["z"][j * nrows:(j + 1) * nrows]]
            image = exact_operator(data, point)
            normalized = [value / allowed for value, allowed in zip(image, data["allowed"])]
            peak = max(map(abs, normalized), default=ZERO)
            exponent = -floor_log2(peak) if peak else 0
            exponents.append(exponent)
            scale = power2(exponent)
            projection_seconds += time.perf_counter() - projected_start
            target.write(f" X{j} OBJ 0\n")
            if not peak:
                stats["zero_projected_columns"] += 1
            for i, value in enumerate(normalized):
                if value:
                    positive = model_number(value * scale, stats, "coefficient")
                    negative_value = model_number(-value * scale, stats, "coefficient")
                    target.write(f" X{j} P{i} {positive} N{i} {negative_value}\n")
                    stats["written_matrix_nonzeros"] += 2
        target.write(" T OBJ 1\n")
        for i in range(nrows):
            target.write(f" T P{i} -1 N{i} -1\n")
        target.write("RHS\n")
        for i, residual in enumerate(center_residual):
            value = residual / data["allowed"][i]
            target.write(f" RHS1 P{i} {model_number(value, stats, 'rhs')} N{i} {model_number(-value, stats, 'rhs')}\n")
        target.write("BOUNDS\n")
        for j in range(k):
            target.write(f" FR BND X{j}\n")
        target.write(" LO BND T 0\nENDATA\n")
    elapsed = time.perf_counter() - start
    for key in ("coefficient_max_absolute_rounding_error", "rhs_max_absolute_rounding_error"):
        stats[key] = packed(stats[key])
    return {"path": str(destination), "sha256": digest(destination), "bytes": Path(destination).stat().st_size,
        "rows": 2 * nrows, "columns": k + 1, "column_scale_exponents": exponents,
        "projection_seconds": projection_seconds, "model_generation_seconds_including_projection": elapsed,
        "model_write_and_conversion_seconds": max(0.0, elapsed - projection_seconds), "rounding": stats,
        "constraint_convention": "+A_scaled*delta-t<=p_scaled；-A_scaled*delta-t<=-p_scaled；t>=0",
        "small_matrix_value_boundary": "全部原行与非零转换值均写出；HiGHS内部可能处理极小项，oracle只供候选搜索，原方程另精确复核。"}


def run_oracle(model, directory, oracle, plan, ledger):
    report_path = directory / "oracle-report.json"
    command = [str(oracle), "solve", str(model), "--backend", "highs", "--highs-solver", "simplex",
               "--highs-presolve", "choose", "--threads", "1", "--seed", "0",
               "--time-limit", str(plan["offline_analysis"]["oracle_time_limit_seconds"]), "--output", str(report_path)]
    record = {"started_at_utc": utc(), "command": command, "cwd": str(ROOT),
        "oracle_sha256": digest(oracle), "wall_timeout_seconds": plan["offline_analysis"]["oracle_wall_timeout_seconds"],
        "numerical_time_limit_seconds": plan["offline_analysis"]["oracle_time_limit_seconds"],
        "scope": "离线低列数诊断 LP，不属于原公开 LP 的自主求解性能"}
    require(len(ledger) < plan["offline_analysis"]["maximum_oracle_runs"], "oracle调用预算已耗尽")
    ledger.append(record)
    stdout, stderr = b"", b""
    start = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=record["wall_timeout_seconds"])
        stdout, stderr = process.stdout, process.stderr
        record.update(returncode=process.returncode, wall_timeout=False)
    except subprocess.TimeoutExpired as failure:
        stdout, stderr = failure.stdout or b"", failure.stderr or b""
        record.update(returncode=None, wall_timeout=True)
    except OSError as failure:
        record.update(returncode=None, wall_timeout=False, launch_error=str(failure))
    record["wall_seconds"] = time.perf_counter() - start
    record["completed_at_utc"] = utc()
    (directory / "oracle-stdout.bin").write_bytes(stdout)
    (directory / "oracle-stderr.bin").write_bytes(stderr)
    record.update(stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        stdout_path=str(directory / "oracle-stdout.bin"), stderr_path=str(directory / "oracle-stderr.bin"),
        report_path=str(report_path), report_sha256=digest(report_path) if report_path.exists() else None)
    report = None
    if report_path.exists():
        try:
            report = read_json(report_path)
            require(report.get("backend") == "highs", "oracle 报告后端不是 HiGHS")
            record["status"] = report.get("status")
            timings = report.get("timings", {})
            record["reported_timings"] = timings
            require(isinstance(timings, dict), "oracle耗时字段类型无效")
            for key in ("read_seconds", "solve_seconds", "verification_seconds"):
                require(type(timings.get(key)) in (int, float) and math.isfinite(timings[key]) and timings[key] >= 0,
                        "oracle耗时无效：" + key)
            record["reported_numerical_seconds"] = timings["solve_seconds"] + timings["verification_seconds"]
            record["reported_numerical_budget_exceeded"] = record["reported_numerical_seconds"] > record["numerical_time_limit_seconds"]
        except Exception as failure:
            record["report_error"] = str(failure)
            report = None
    write_json(directory / "oracle-process.json", record)
    return record, report


def analyze_capture(identity, data, directory, oracle, plan, ledger, result):
    a = data["arrays"]
    result.update({"numerical_sha256": data["numerical_sha256"], "event": {
        key: data["event"][key] for key in ("phase", "iteration", "restart_index", "column_index", "generated_columns", "coefficient_count")},
        "dimension": data["dimension"], "capture": identity, "oracle_runs": 0,
        "original_space": "base+Z*a；不是以已舍入last_candidate为新基点"})
    start = time.perf_counter()
    reproduced = production_combination(a["base"], a["z"], a["coefficients"], data["wide_format"])
    mismatches = [i for i, (x, y) in enumerate(zip(reproduced, a["last_candidate"]))
                  if struct.pack("<d", x) != struct.pack("<d", y)]
    result["production_replay"] = {"bitwise_equal": not mismatches, "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:8], "seconds": time.perf_counter() - start,
        "arithmetic": "每项先round53(coeff*Z)，再roundLDBL_MANT_DIG(candidate+product)，再round53写回；j顺序不变"}
    if mismatches:
        result.update(classification="production_replay_mismatch_no_oracle", conclusion="无法复现捕获候选，停止可信空间推断")
        return result
    start = time.perf_counter()
    a0 = [Fraction.from_float(x) for x in a["coefficients"]] + [ZERO] * (data["k"] - data["cc"])
    center = continuous_combination(a["base"], a["z"], a0)
    center_residual = exact_residual(data, center)
    captured_residual = exact_residual(data, [Fraction.from_float(x) for x in a["last_candidate"]])
    result["baseline"] = {"coefficients": [packed(x) for x in a0],
        "exact_continuous": residual_summary(data, center_residual),
        "production_candidate_exact": residual_summary(data, captured_residual),
        "production_stored_residual": residual_summary(data, [Fraction.from_float(x) for x in a["last_residual"]]),
        "stored_residual_evaluation_error": BASE.vector_summary(
            [Fraction.from_float(x) - y for x, y in zip(a["last_residual"], captured_residual)]),
        "exact_center_and_baseline_seconds": time.perf_counter() - start}
    directory.mkdir(parents=True, exist_ok=False)
    model = write_model(data, center_residual, directory / "minimax.mps")
    result["diagnostic_model"] = model
    process, report = run_oracle(directory / "minimax.mps", directory, oracle, plan, ledger)
    result.update(oracle_runs=1, oracle=process)
    solution = report.get("solution") if isinstance(report, dict) else None
    if not isinstance(solution, list) or len(solution) != data["k"] + 1 or not all(type(x) in (int, float) and math.isfinite(x) for x in solution):
        result.update(classification="no_finite_oracle_candidate", conclusion="未找到可独立复核的见证；不证明该空间无解")
        return result
    start = time.perf_counter()
    a_exact = [base + power2(exponent) * Fraction.from_float(float(delta)) for base, exponent, delta in
               zip(a0, model["column_scale_exponents"], solution[:-1])]
    result["oracle_solution"] = {"delta_binary64_hex": [float(x).hex() for x in solution[:-1]],
        "reported_t": solution[-1], "a_exact": [packed(x) for x in a_exact]}
    exact_point = continuous_combination(a["base"], a["z"], a_exact)
    result["oracle_map_continuous"] = residual_summary(data, exact_residual(data, exact_point))
    try:
        a_double = [cast64(x) for x in a_exact]
        result["oracle_solution"]["a_double_hex"] = [x.hex() for x in a_double]
        double_point = continuous_combination(a["base"], a["z"], [Fraction.from_float(x) for x in a_double])
        result["double_coefficients_continuous"] = residual_summary(data, exact_residual(data, double_point))
        production = production_combination(a["base"], a["z"], a_double, data["wide_format"])
        result["production_combination_exact"] = residual_summary(data, exact_residual(data, [Fraction.from_float(x) for x in production]))
        result["production_candidate_sha256"] = hashlib.sha256(b"".join(struct.pack("<d", x) for x in production)).hexdigest()
        result["classification"] = "production_point_exact_witness_found" if result["production_combination_exact"]["satisfies_original_tolerances"] else "oracle_candidate_does_not_provide_production_witness"
    except OverflowError as failure:
        result.update(classification="oracle_coefficients_not_representable_in_production", production_error=str(failure))
    result["witness_verification_seconds"] = time.perf_counter() - start
    result["conclusion"] = "仅给出该捕获空间的原方程精确残差见证；未重放候选的新浮点残差接受检查，未作不可行或最优性证明，不证明HSD方向接受、LP达标或端到端更快"
    return result


def self_test():
    checks = 0

    def check(condition):
        nonlocal checks
        require(condition, "v67 精确数学自检失败")
        checks += 1

    def rejects(function):
        nonlocal checks
        try:
            function()
        except (ValueError, OverflowError):
            checks += 1
        else:
            raise ValueError("自检输入本应被拒绝")

    for value in (ZERO, Fraction(1), Fraction(-3, 2), power2(-1074), -power2(-1074), power2(1000)):
        check(Fraction.from_float(cast64(value)) == value)
    check(rounded_binary(Fraction(1) + power2(-53)) == 1)
    check(rounded_binary(Fraction(1) + 3 * power2(-53)) == Fraction(1) + power2(-51))
    check(rounded_binary(power2(-1075)) == 0)
    check(rounded_binary(3 * power2(-1075)) == power2(-1073))
    try:
        rounded_binary(power2(1024))
    except OverflowError:
        checks += 1
    else:
        raise ValueError("应拒绝 binary64 溢出")
    wide = (64, -16381, 16384)
    increment = float(power2(-53) + power2(-65))
    check(production_combination([1.0], [1.0], [increment], wide) == [1.0])
    check(cast64(Fraction(1) + Fraction.from_float(increment)) == 1.0 + 2.0 ** -52)
    check(negative(production_combination([-0.0], [1.0], [-0.0], wide)[0]))
    check(not negative(production_combination([-0.0], [1.0], [0.0], wide)[0]))
    check(production_combination([-1.0], [1.0 - 2.0 ** -27], [1.0 + 2.0 ** -27], wide) == [0.0])
    check(negative(cast64(-power2(-1075))))
    rejects(lambda: model_number(power2(-1075), {}, "coefficient"))
    data = {"n": 1, "m": 1, "dimension": 2, "k": 1, "cc": 1, "wide_format": wide,
        "allowed": [Fraction.from_float(0.35)] * 2,
        "arrays": {"g_starts": [0, 1], "g_rows": [0], "g_values": [1.0], "h": [0.0],
                   "linear_rhs": [0.0, 1.0], "base": [0.0, 0.0], "z": [2.0, 1.0]}}
    point = continuous_combination(data["arrays"]["base"], data["arrays"]["z"], [Fraction(1, 3)])
    check(exact_residual(data, point) == [Fraction(-1, 3), Fraction(1, 3)])
    check(residual_summary(data, exact_residual(data, point))["satisfies_original_tolerances"])
    a0 = Fraction.from_float(0.4)
    center = continuous_combination([0.0, 0.0], [2.0, 1.0], [a0])
    residual = exact_residual(data, center)
    check(not residual_summary(data, residual)["satisfies_original_tolerances"])
    image = exact_operator(data, [Fraction(2), Fraction(1)])
    scale = power2(-floor_log2(max(abs(x) / data["allowed"][i] for i, x in enumerate(image))))
    delta = (Fraction(1, 3) - a0) / scale
    check([r - image[i] * scale * delta for i, r in enumerate(residual)] == exact_residual(data, point))
    t = max(abs(x) / data["allowed"][i] for i, x in enumerate(exact_residual(data, point)))
    for i in range(2):
        lhs, rhs = image[i] * scale * delta / data["allowed"][i], residual[i] / data["allowed"][i]
        check(lhs - t <= rhs and -lhs - t <= -rhs)
    with tempfile.TemporaryDirectory(prefix="v67-math-") as temporary:
        model = write_model(data, residual, Path(temporary) / "fixture.mps")
        check(model["rows"] == 4 and model["columns"] == 2)
        check(model["column_scale_exponents"] == [-floor_log2(max(abs(x) / data["allowed"][i] for i, x in enumerate(image)))])
        text = Path(model["path"]).read_text(encoding="ascii")
        check(" FR BND X0\n" in text and " LO BND T 0\n" in text and " T P1 -1 N1 -1\n" in text)
        arrays = dict(data["arrays"], coefficients=[0.4], last_candidate=[0.8, 0.4], last_residual=[-0.4, 0.19999999999999996])
        binary, descriptors = bytearray(MAGIC), []
        for index, name in enumerate(NAMES):
            values = arrays[name]
            descriptors.append({"name": name, "dtype": "int64" if index < 2 else "float64", "count": len(values),
                "width": 8, "offset": len(binary), "shape": [1, 2] if name == "z" else [len(values)]})
            binary.extend(struct.pack("<" + ("q" if index < 2 else "d") * len(values), *values))
        event = {"phase": "ordinary_constant", "linear_exit": "krylov_exhausted", "terminal": True,
            "candidate_submitted": False, "residual_sign": "rhs_minus_K0_candidate", "iteration": 0,
            "n": 1, "m": 1, "dimension": 2, "nnz": 1, "generated_columns": 1, "coefficient_count": 1,
            "restart_index": 2, "column_index": 0, "arrays": descriptors, "payload_bytes": len(binary) - 8,
            "cache_bytes": 65536 + len(binary) - 8, "global_tolerance": (0.35).hex(), "block_tolerances": None}
        event.update({key: True for key in AVAILABLE})
        metadata = {"schema": 1, "status": "captured", "byte_order": "little", "binary_bytes": len(binary),
            "fixed_cache_bytes": 65536, "truncated": False, "capture_errors": 0, "error": "", "event_count": 1,
            "events": [event], "floating_point": {
                "double": {"bytes": 8, "digits": 53, "min_exponent": -1021, "max_exponent": 1024, "is_iec559": True},
                "long_double": {"bytes": 16, "digits": 64, "min_exponent": -16381, "max_exponent": 16384, "is_iec559": True},
                "flt_eval_method": 0}, "rounding_mode": {"value": 0, "is_to_nearest": True}}
        fixture = Path(temporary) / "fixture.json"
        fixture.with_suffix(".bin").write_bytes(binary)

        def load_fixture():
            fixture.write_text(json.dumps(metadata), encoding="utf-8")
            return load_snapshot(fixture)

        loaded = load_fixture()[2]
        check(list(loaded["arrays"]["z"]) == [2.0, 1.0] and loaded["allowed"] == data["allowed"])
        check(production_combination(loaded["arrays"]["base"], loaded["arrays"]["z"],
              loaded["arrays"]["coefficients"], loaded["wide_format"]) == arrays["last_candidate"])
        event["block_tolerances"] = [(0.25).hex(), (0.5).hex()]
        check(load_fixture()[2]["allowed"] == [Fraction(1, 4), Fraction.from_float(0.35)])
        event["candidate_submitted"] = True
        rejects(load_fixture)
        event["candidate_submitted"] = False
        descriptors[0]["shape"] = [True]
        rejects(load_fixture)
        descriptors[0]["shape"] = [2]
        metadata["floating_point"]["flt_eval_method"] = 2
        rejects(load_fixture)
        metadata["floating_point"]["flt_eval_method"] = 0
        event["cache_bytes"] += 1
        rejects(load_fixture)
        event["cache_bytes"] -= 1
        metadata.update(status="no_target_failure", binary_bytes=8, events=[], event_count=0)
        fixture.with_suffix(".bin").write_bytes(MAGIC)
        check(load_fixture()[2] is None)
    return {"passed": True, "mathematical_checks": checks, "oracle_runs": 0,
        "scope": "精确符号、中心化、缩放、舍入及纯写模；不调用HiGHS或原LP"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(not args.snapshots and args.output is None and args.work_dir is None, "自检不接受正式输入")
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    require(args.output is not None and args.work_dir is not None and 1 <= len(args.snapshots) <= 6,
            "要求快照、输出及独立工作目录")
    args.output, args.work_dir = args.output.resolve(), args.work_dir.resolve()
    require(not args.output.exists() and not args.work_dir.exists(), "不得覆盖已有分析或oracle证据")
    start, started_at = time.perf_counter(), utc()
    plan_path = DATA / "v67-plan.json"
    require(digest(plan_path) == PLAN_SHA, "v67 冻结计划改变")
    plan = read_json(plan_path)
    require(plan["offline_analysis"]["maximum_oracle_runs"] == 2 and
            plan["offline_analysis"]["oracle_time_limit_seconds"] == 60 and
            plan["offline_analysis"]["oracle_wall_timeout_seconds"] == 90, "oracle预算改变")
    oracle = ROOT / ".tools/solver-ipm-v66.exe"
    require(digest(oracle) == plan["parent_solver_sha256"], "冻结oracle身份改变")
    loaded = [load_snapshot(path) for path in args.snapshots]
    require(all(metadata["status"] in ("captured", "no_target_failure") and metadata["capture_errors"] == 0 and
                metadata["truncated"] is False for _, metadata, _ in loaded), "不完整采集不能启动可信空间诊断")
    unique = {data["numerical_sha256"] for _, _, data in loaded if data is not None}
    require(len(unique) <= 2, "唯一数值捕获超过两份oracle预算")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    output = {"schema": 1, "started_at_utc": started_at, "preflight_seconds": time.perf_counter() - start, "plan_sha256": PLAN_SHA,
        "analyzer_sha256": digest(Path(__file__)), "exact_helper_sha256": HELPER_SHA,
        "oracle_sha256": digest(oracle), "captures": [], "analyses": [], "oracle_runs": 0, "oracle_calls": [],
        "unique_numerical_captures": len(unique), "original_lp_runs": 0,
        "limits": ["oracle只解捕获空间的诊断LP，不计入自主性能。", "全部原约束保留，double写模及上游极小项处理可能改变近似oracle问题。",
                   "精确连续映射、double系数连续组合与生产逐次舍入点分别验原K0。",
                   "没有可靠不可行证书，只能报告是否找到见证；不证明该空间无解或已经LP达标。"]}
    reuse = {}
    try:
        for identity, metadata, data in loaded:
            record = dict(identity)
            if data is None:
                record.update(analysis_index=None, reason=metadata["status"], capture_error=metadata.get("error"))
            elif data["numerical_sha256"] in reuse:
                record.update(analysis_index=reuse[data["numerical_sha256"]], reused_identical_numerical_capture=True)
            else:
                index = len(output["analyses"])
                reuse[data["numerical_sha256"]] = index
                result = {}
                output["analyses"].append(result)
                analyze_capture(identity, data, args.work_dir / f"oracle-{index}", oracle, plan, output["oracle_calls"], result)
                record.update(analysis_index=index, reused_identical_numerical_capture=False)
            output["captures"].append(record)
        require(digest(oracle) == output["oracle_sha256"], "oracle程序在分析中改变")
        output["completed"] = True
    except Exception as failure:
        output.update(completed=False, error_type=type(failure).__name__, error=str(failure))
        raise
    finally:
        output["oracle_runs"] = len(output["oracle_calls"])
        output.update(completed_at_utc=utc(), elapsed_wall_seconds=time.perf_counter() - start)
        write_json(args.output, output)
    print(json.dumps({"completed": output["completed"], "captures": len(output["captures"]),
        "analyses": len(output["analyses"]), "oracle_runs": output["oracle_runs"],
        "output_sha256": digest(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
