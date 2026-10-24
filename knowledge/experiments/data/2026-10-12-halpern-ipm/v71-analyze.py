"""精确分析已捕获的基候选；不求解、不改点、不使用已知最优值。"""

import argparse
from array import array
import datetime
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile


DATA = Path(__file__).resolve().parent
PLAN_SHA = "dd03b4c7cdad5319b1e47fa0f8e67845f7002f85e7b6c33da3cf03b857b55c7b"
MAGIC = b"V71BAS01"
CAP = 128 * 1024 * 1024
ZERO = Fraction(0)
NAMES = ("g_starts", "g_rows", "g_values", "costs", "rhs", "slack", "dual",
         "inequality", "candidate_x", "candidate_s", "candidate_z", "selected_rows", "pivot_columns")
INTEGER_NAMES = {"g_starts", "g_rows", "selected_rows", "pivot_columns"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, low=0, high=(1 << 63) - 1):
    require(type(value) is int and low <= value <= high, "整数类型或范围无效")
    return value


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, "JSON 重复键：" + key)
        result[key] = value
    return result


def parse_json(blob):
    return json.loads(blob, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("非有限 JSON：" + x)))


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def packed(value):
    """分子分母是精确值；近似浮点仅供阅读，溢出时不写非标准 JSON。"""
    require(isinstance(value, Fraction), "精确输出要求 Fraction")
    try:
        approximate = float(value)
    except OverflowError:
        approximate = None
    return {"numerator": str(value.numerator), "denominator": str(value.denominator),
            "approx": approximate, "approx_overflow": approximate is None}


def max_summary(values, indices=None):
    values = list(values)
    indices = list(range(len(values))) if indices is None else list(indices)
    require(len(values) == len(indices), "诊断索引数量不匹配")
    best, best_index, signed = ZERO, None, ZERO
    for index, value in zip(indices, values):
        if best_index is None or abs(value) > best:
            best, best_index, signed = abs(value), index, value
    return {"count": len(values), "nonzero_count": sum(value != 0 for value in values),
            "max_abs": packed(best), "max_abs_index": best_index, "signed_at_max_abs": packed(signed)}


def ratio(numerator, denominator):
    require(numerator >= 0 and denominator >= 0, "后向误差需要非负分子分母")
    if denominator:
        return numerator / denominator
    return ZERO if numerator == 0 else None


def backward_summary(residuals, denominators, indices):
    best, best_index, infinite = ZERO, None, []
    for index, residual, denominator in zip(indices, residuals, denominators):
        value = ratio(abs(residual), denominator)
        if value is None:
            infinite.append(index)
        elif best_index is None or value > best:
            best, best_index = value, index
    return {"zero_over_zero": "zero", "positive_over_zero": "infinite",
            "infinite_count": len(infinite), "first_infinite_index": infinite[0] if infinite else None,
            "max_finite": packed(best), "max_finite_index": best_index,
            "maximum_is_infinite": bool(infinite)}


def negative_summary(values, indices):
    pairs = [(index, -value) for index, value in zip(indices, values) if value < 0]
    result = max_summary((value for _, value in pairs), (index for index, _ in pairs))
    result["negative_count"] = len(pairs)
    result["sum_negative_magnitudes"] = packed(sum((value for _, value in pairs), ZERO))
    return result


def validate_arrays(arrays, n, m, nnz):
    counts = (n + 1, nnz, nnz, n, m, m, m, m, n, m, m, n, n)
    require(set(arrays) == set(NAMES), "捕获数组集合不匹配")
    for name, count in zip(NAMES, counts):
        require(len(arrays[name]) == count, "捕获数组长度不匹配：" + name)
        if name not in INTEGER_NAMES and name != "inequality":
            require(all(math.isfinite(x) for x in arrays[name]), "非有限捕获数组：" + name)
    starts, rows = arrays["g_starts"], arrays["g_rows"]
    require(starts[0] == 0 and starts[-1] == nnz, "CSC 首尾偏移不匹配")
    for column in range(n):
        begin, end = starts[column], starts[column + 1]
        require(0 <= begin <= end <= nnz, "CSC 列偏移无效")
        previous = -1
        for offset in range(begin, end):
            row = rows[offset]
            require(previous < row < m, "CSC 行号未严格升序或越界")
            previous = row
    selected, pivots = arrays["selected_rows"], arrays["pivot_columns"]
    require(len(set(selected)) == n and all(0 <= x < m for x in selected), "基原行映射重复或越界")
    require(len(set(pivots)) == n and all(0 <= x < n for x in pivots), "主元原变量映射不是排列")
    require(all(x in (0, 1) for x in arrays["inequality"]), "不等式位不是 0 或 1")
    require(all(not flag or arrays["slack"][i] > 0 and arrays["dual"][i] > 0
                for i, flag in enumerate(arrays["inequality"])), "原内点不满足恢复入口正值条件")


def load_capture(path):
    path = Path(path).resolve()
    require(path.stat().st_size <= 65536, "捕获元数据超出固定容量")
    metadata_blob = path.read_bytes()
    metadata = parse_json(metadata_blob)
    require(metadata.get("schema") == 1 and type(metadata.get("schema")) is int and
            metadata.get("version") == "v71", "捕获版本或 schema 不匹配")
    require(metadata.get("status") in ("captured", "no_candidate", "capture_error"), "未知捕获状态")
    binary_path = path.with_suffix(".bin")
    require(8 <= binary_path.stat().st_size <= CAP + 8, "捕获文件超出固定容量")
    blob = binary_path.read_bytes()
    require(blob[:8] == MAGIC and integer(metadata.get("binary_bytes")) == len(blob), "二进制头或长度不匹配")
    byte_order = metadata.get("byte_order")
    require(byte_order in ("little", "big"), "捕获未声明有效字节序")
    require(integer(metadata.get("cache_limit_bytes")) == CAP and
            integer(metadata.get("metadata_limit_bytes")) == 65536, "捕获资源上限与计划不符")
    require(type(metadata.get("truncated")) is bool, "捕获截断标记类型无效")
    count = integer(metadata.get("event_count"), 0, 1)
    integer(metadata.get("attempts"), 0, 1)
    errors = integer(metadata.get("capture_errors"))
    events = metadata.get("events")
    require(isinstance(events, list) and len(events) == count, "捕获事件数量不匹配")
    identity = {"metadata_path": str(path), "binary_path": str(binary_path),
                "metadata_sha256": digest(metadata_blob), "binary_sha256": digest(blob),
                "metadata_bytes": len(metadata_blob), "binary_bytes": len(blob), "byte_order": byte_order}
    if metadata["status"] != "captured":
        require(count == 0 and len(blob) == 8, "非完整捕获不能携带事件负载")
        if metadata["status"] == "no_candidate":
            require(metadata["attempts"] == 0 and errors == 0 and metadata.get("error") == "none" and
                    not metadata["truncated"], "无候选记录不能隐藏采集错误")
        return identity, metadata, None
    require(count == 1 and metadata["attempts"] == 1 and not metadata["truncated"] and errors == 0 and
            metadata.get("error") == "none",
            "只分析无截断且无捕获错误的单事件")
    n, m = integer(metadata.get("n"), 0, 1000000), integer(metadata.get("m"), 0, 1000000)
    nnz = integer(metadata.get("nnz"), 0, 8000000)
    event = events[0]
    require(isinstance(event, dict) and event.get("phase") == "recovered_basis_before_assess" and
            integer(event.get("index")) == 0, "捕获阶段或事件序号不匹配")
    descriptors = event.get("arrays")
    require(isinstance(descriptors, list) and all(isinstance(x, dict) for x in descriptors) and
            [x.get("name") for x in descriptors] == list(NAMES), "捕获数组顺序不匹配")
    counts = (n + 1, nnz, nnz, n, m, m, m, m, n, m, m, n, n)
    arrays, cursor = {}, 8
    for name, count, descriptor in zip(NAMES, counts, descriptors):
        dtype, code, width = (("int64", "q", 8) if name in INTEGER_NAMES else
                              ("uint8", "B", 1) if name == "inequality" else ("float64", "d", 8))
        require(descriptor.get("dtype") == dtype and integer(descriptor.get("width")) == width and
                integer(descriptor.get("count")) == count and integer(descriptor.get("offset")) == cursor and
                isinstance(descriptor.get("shape"), list) and
                [integer(x) for x in descriptor["shape"]] == [count], "捕获数组描述不匹配：" + name)
        end = cursor + width * count
        require(end <= len(blob), "捕获数组越界：" + name)
        values = array(code)
        require(values.itemsize == width, "本平台数组元素宽度不匹配")
        values.frombytes(memoryview(blob)[cursor:end])
        if sys.byteorder != byte_order and width > 1:
            values.byteswap()
        # 先核对原字节，包括正负零；Fraction 转换只服务数学诊断，不替代原捕获。
        repacked = array(code, values)
        if sys.byteorder != byte_order and width > 1:
            repacked.byteswap()
        require(repacked.tobytes() == blob[cursor:end], "解码未保留原始点字节：" + name)
        arrays[name] = values
        cursor = end
    require(cursor == len(blob), "捕获二进制包含未声明尾部")
    require(integer(metadata.get("payload_bytes")) == len(blob) - 8, "捕获负载长度不匹配")
    fixed = integer(metadata.get("fixed_cache_bytes"), 0, CAP)
    require(integer(metadata.get("cache_bytes"), 0, CAP) == fixed + len(blob) - 8, "捕获缓存账本不匹配")
    validate_arrays(arrays, n, m, nnz)
    identity["array_sha256"] = {item["name"]: digest(blob[item["offset"]:
        item["offset"] + item["count"] * item["width"]]) for item in descriptors}
    return identity, metadata, arrays


def analyze_arrays(arrays):
    """矩阵只按 CSC 遍历一次；有理数运算次数为 O(nnz+n+m)，不构建稠密基。"""
    n, m, nnz = len(arrays["costs"]), len(arrays["rhs"]), len(arrays["g_values"])
    validate_arrays(arrays, n, m, nnz)
    selected = list(arrays["selected_rows"])
    positions = {row: position for position, row in enumerate(selected)}
    q = lambda name: [Fraction.from_float(float(x)) for x in arrays[name]]
    x, z, s, b, c = (q(name) for name in ("candidate_x", "candidate_z", "candidate_s", "rhs", "costs"))
    activity, activity_abs = [ZERO] * m, [ZERO] * m
    rd, dual_abs = list(c), [abs(value) for value in c]
    products = 0
    for column in range(n):
        for offset in range(arrays["g_starts"][column], arrays["g_starts"][column + 1]):
            row = arrays["g_rows"][offset]
            value = Fraction.from_float(float(arrays["g_values"][offset]))
            product = value * x[column]
            activity[row] += product
            activity_abs[row] += abs(product)
            products += 1
            if row in positions:
                product = value * z[row]
                rd[column] += product
                dual_abs[column] += abs(product)
                products += 1
    rp = [activity[i] - b[i] for i in range(m)]
    exact_slack = [-value for value in rp]
    primal_den = [abs(b[i]) + activity_abs[i] for i in range(m)]
    groups = {}
    for in_basis in (True, False):
        for inequality in (False, True):
            rows = [i for i in range(m) if (i in positions) == in_basis and
                    bool(arrays["inequality"][i]) == inequality]
            key = ("selected_" if in_basis else "unselected_") + ("inequalities" if inequality else "equalities")
            groups[key] = {"row_count": len(rows),
                "row_residual": max_summary((rp[i] for i in rows), rows),
                "componentwise_backward_error": backward_summary((rp[i] for i in rows),
                    (primal_den[i] for i in rows), rows)}
            if inequality:
                groups[key]["negative_exact_slack"] = negative_summary((exact_slack[i] for i in rows), rows)
                groups[key]["negative_stored_slack"] = negative_summary((s[i] for i in rows), rows)
    inequalities = [i for i in range(m) if arrays["inequality"][i]]
    equalities = [i for i in range(m) if not arrays["inequality"][i]]
    nonbasis = [i for i in range(m) if i not in positions]
    primal_objective = sum((c[j] * x[j] for j in range(n)), ZERO)
    btz_basis = sum((b[i] * z[i] for i in selected), ZERO)
    btz_full = sum((b[i] * z[i] for i in range(m)), ZERO)
    xrd_terms = [x[j] * rd[j] for j in range(n)]
    zrp_terms = [z[i] * rp[i] for i in selected]
    xrd, zrp = sum(xrd_terms, ZERO), sum(zrp_terms, ZERO)
    raw_gap = primal_objective + btz_basis
    identity_error = raw_gap - (xrd - zrp)
    require(identity_error == 0, "精确基目标差恒等式内部校验失败")
    return {"dimensions": {"n": n, "m": m, "nnz": nnz}, "basis_size": len(selected),
        "computation": {"matrix_passes": 1, "matrix_scalar_products": products,
                        "dense_basis_created": False, "factorizations": 0, "linear_solves": 0},
        "basis_primal": {"residual": max_summary((rp[i] for i in selected), selected),
            "componentwise_backward_error": backward_summary((rp[i] for i in selected),
                (primal_den[i] for i in selected), selected)},
        "basis_dual": {"residual": max_summary(rd),
            "componentwise_backward_error": backward_summary(rd, dual_abs, range(n))},
        "row_groups": groups,
        "inequality_negative_dual": negative_summary((z[i] for i in inequalities), inequalities),
        "equality_negative_dual": {"negative_values_allowed": True,
            **negative_summary((z[i] for i in equalities), equalities)},
        "nonbasis_dual": {"exactly_zero": all(z[i] == 0 for i in nonbasis),
            **max_summary((z[i] for i in nonbasis), nonbasis)},
        "candidate_slack": {
            "inequality_representation_error": max_summary((s[i] - exact_slack[i] for i in inequalities), inequalities),
            "inequality_representation_backward_error": backward_summary(
                (s[i] - exact_slack[i] for i in inequalities),
                (abs(s[i]) + primal_den[i] for i in inequalities), inequalities),
            "equality_stored_slack_exactly_zero": all(s[i] == 0 for i in equalities),
            "equality_stored_slack": max_summary((s[i] for i in equalities), equalities),
            "note": "不等式 s 由宽精度累加后转回 double；精确表示误差不要求为零。等式 s 按实现恒置零，等式残差另计。"},
        "magnitudes": {"x": max_summary(x), "z": max_summary(z),
                       "z_B": max_summary((z[i] for i in selected), selected),
                       "b_B": max_summary((b[i] for i in selected), selected), "c": max_summary(c)},
        "objectives": {"primal_cTx": packed(primal_objective), "dual_minus_bBTzB": packed(-btz_basis),
            "raw_basis_gap_cTx_plus_bBTzB": packed(raw_gap),
            "raw_full_point_gap_cTx_plus_bTz": packed(primal_objective + btz_full),
            "nonbasis_bNTzN": packed(btz_full - btz_basis),
            "xTrd": packed(xrd), "zBTrp": packed(zrp), "xTrd_minus_zBTrp": packed(xrd - zrp),
            "sum_abs_xj_rdj": packed(sum((abs(value) for value in xrd_terms), ZERO)),
            "sum_abs_zBi_rpi": packed(sum((abs(value) for value in zrp_terms), ZERO)),
            "identity_exact": identity_error == 0, "identity_error": packed(identity_error)},
        "interpretation": [
            "数值均针对捕获的原 binary64 数据，以精确有理数计算；不是精确解或最优解证书。",
            "后向误差不等于前向误差或条件数；小后向误差仍可能被巨大 x 或 z 放大为目标差。",
            "基方程准确与原始/对偶符号可行是不同检查；此报告不自动选择精修、换基或 phase-I。",
            "原模型 assess gap 经缩放反变换和乘子构造，应单独列示，不能替代本报告 raw cone gap。"]}


def load_run_record(path, index):
    path = Path(path).resolve()
    blob = path.read_bytes()
    lines = gzip.decompress(blob).splitlines() if path.suffix == ".gz" else blob.splitlines()
    integer(index, 0, len(lines) - 1)
    record = parse_json(lines[index])
    require(record.get("run_index") == index and type(record.get("run_index")) is int, "运行记录序号不匹配")
    return {"path": str(path), "sha256": digest(blob), "record_index": index}, record


def associate_record(identity, record):
    require(record.get("requested_variant") == "diagnostic_capture_on", "必须关联开启捕获的运行")
    capture = record.get("capture")
    require(isinstance(capture, dict) and capture.get("enabled") is True, "运行记录没有捕获元数据")
    require(not record.get("capture_error"), "运行记录包含捕获错误")
    for key in ("metadata_path", "binary_path"):
        require(isinstance(capture.get(key), str) and Path(capture[key]).resolve() == Path(identity[key]),
                "运行记录捕获路径不匹配：" + key)
    for key in ("metadata_sha256", "binary_sha256", "metadata_bytes", "binary_bytes"):
        require(capture.get(key) == identity[key], "运行记录捕获身份不匹配：" + key)
    marker = record.get("basis_marker")
    require(isinstance(marker, dict), "缺少本次原 V70 marker")
    trial = marker.get("trial_verification")
    require(trial is None or isinstance(trial, dict), "原模型验收字段无效")
    return {"marker_type": "V70_BASIS", "stage": marker.get("stage"),
            "recovery_status": marker.get("recovery_status"), "trial_verification": trial,
            "note": "此项来自关联运行记录的原模型 assess；不参与 raw cone 恒等式，也不作为已知最优值。"}


def analyze_capture(path, record):
    identity, metadata, arrays = load_capture(path)
    original = associate_record(identity, record)
    return {"schema": 1, "version": "v71", "diagnostic_only": True,
            "candidate_modified": False, "oracle_used": False, "status": metadata["status"],
            "capture": identity, "original_model_assessment": original,
            "analysis": analyze_arrays(arrays) if arrays is not None else None}


def analyze_runs(path):
    """捕获内容相同才复用精确计算；每次真实运行的身份与验收仍分别保存。"""
    path = Path(path).resolve()
    blob = path.read_bytes()
    lines = gzip.decompress(blob).splitlines() if path.suffix == ".gz" else blob.splitlines()
    require(len(lines) == 18, "完整诊断要求冻结的 18 条运行记录")
    associations, unique = [], {}
    for index, line in enumerate(lines):
        record = parse_json(line)
        require(type(record.get("run_index")) is int and record["run_index"] == index, "运行序号不连续")
        variant = record.get("requested_variant")
        require(variant in ("frozen_v70_on", "diagnostic_capture_off", "diagnostic_capture_on"), "未知运行状态")
        marker = record.get("basis_marker")
        require(isinstance(marker, dict), "关联运行缺少 V70 账本")
        item = {"run_index": index, "case_index": record.get("case_index"),
                "model_index": record.get("model_index"), "repetition": record.get("repetition"),
                "requested_variant": variant, "stage": marker.get("stage"),
                "original_model_trial_verification": marker.get("trial_verification"),
                "capture": None, "analysis_key": None}
        if variant == "diagnostic_capture_on":
            capture = record.get("capture")
            require(isinstance(capture, dict) and isinstance(capture.get("metadata_path"), str), "开启采集但缺少路径")
            identity, metadata, arrays = load_capture(capture["metadata_path"])
            item["original_model_assessment"] = associate_record(identity, record)
            item["capture"] = {**identity, "status": metadata["status"]}
            if arrays is not None:
                # 包含数组字节及解释这些字节所需的维度、布局；不按模型名去重或路由。
                key = digest(json.dumps({"binary_sha256": identity["binary_sha256"],
                    "byte_order": metadata["byte_order"],
                    "n": metadata["n"], "m": metadata["m"], "nnz": metadata["nnz"],
                    "arrays": metadata["events"][0]["arrays"]}, sort_keys=True).encode("utf-8"))
                item["analysis_key"] = key
                if key not in unique:
                    unique[key] = {"first_run_index": index, "run_indices": [], "analysis": analyze_arrays(arrays)}
                unique[key]["run_indices"].append(index)
        associations.append(item)
    return {"schema": 1, "version": "v71", "diagnostic_only": True, "candidate_modified": False,
            "oracle_used": False, "status": "analyzed", "run_count": len(associations),
            "capture_run_count": sum(row["capture"] is not None for row in associations),
            "candidate_capture_count": sum(row["analysis_key"] is not None for row in associations),
            "unique_numerical_captures": len(unique),
            "run_record": {"path": str(path), "sha256": digest(blob)},
            "runs": associations, "analyses": unique,
            "note": "运行器另行核对 12 个完整轨迹对子；此离线分析不将重复捕获解释为全部中间状态一致。"}


def fixture(starts, rows, values, costs, rhs, inequality, x, s, z, selected, pivots):
    m = len(rhs)
    return {"g_starts": starts, "g_rows": rows, "g_values": values, "costs": costs, "rhs": rhs,
            "slack": [1.0] * m, "dual": [1.0] * m, "inequality": inequality,
            "candidate_x": x, "candidate_s": s, "candidate_z": z,
            "selected_rows": selected, "pivot_columns": pivots}


def unpacked(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def self_test():
    cases = []
    a = fixture([0, 2], [0, 1], [1.0, 2.0], [-1.0], [3.0, 6.0], [0, 0],
                [3.0], [0.0, 0.0], [1.0, 0.0], [0], [0])
    result = analyze_arrays(a)
    require(result["row_groups"]["unselected_equalities"]["row_residual"]["nonzero_count"] == 0 and
            result["objectives"]["identity_exact"], "比例依赖原行精确检查失败")
    cases.append("比例依赖原行无需稠密分解")
    a = fixture([0, 2, 3], [0, 2, 1], [1.0, 1.0, 1.0], [1.0, 2.0], [0.0, 0.0, -1.0], [0, 1, 1],
                [0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, -2.0, 0.0], [1, 0], [1, 0])
    result = analyze_arrays(a)
    require(result["inequality_negative_dual"]["negative_count"] == 1 and
            result["equality_negative_dual"]["negative_count"] == 1 and
            result["equality_negative_dual"]["negative_values_allowed"] and
            result["basis_dual"]["residual"]["nonzero_count"] == 0 and
            result["row_groups"]["unselected_inequalities"]["negative_exact_slack"]["negative_count"] == 1,
            "等式与不等式乘子符号、基外负 slack 或映射失败")
    cases.append("完整原行映射及等式负乘子允许")
    a = fixture([0, 1], [0], [1.0], [-1.0 + 2.0 ** -52], [2.0 ** 100], [1],
                [2.0 ** 100], [0.0], [1.0], [0], [0])
    result = analyze_arrays(a)
    require(unpacked(result["basis_dual"]["residual"]["max_abs"]) == Fraction(1, 2 ** 52) and
            unpacked(result["objectives"]["raw_basis_gap_cTx_plus_bBTzB"]) == 2 ** 48 and
            result["objectives"]["identity_exact"], "微小基残差放大目标差失败")
    cases.append("微小残差放大及精确目标差恒等式")
    require(ratio(ZERO, ZERO) == 0 and ratio(Fraction(1), ZERO) is None and
            backward_summary([ZERO], [ZERO], [0])["maximum_is_infinite"] is False,
            "零分母约定失败")
    cases.append("后向误差零分母语义")
    a = fixture([0, 2], [0, 1], [1.0, 3.0], [-1.0], [0.1, 1.0], [0, 1],
                [0.1], [0.0, 0.7], [1.0, 0.0], [0], [0])
    result = analyze_arrays(a)
    require(result["candidate_slack"]["inequality_representation_error"]["nonzero_count"] == 1 and
            result["candidate_slack"]["equality_stored_slack_exactly_zero"] and
            result["nonbasis_dual"]["exactly_zero"], "slack 表示误差与等式约定失败")
    cases.append("slack 表示误差不误判为捕获字节改变")
    a = fixture([0, 1], [0], [1.0], [-0.0], [0.0, 1.0], [0, 1], [-0.0],
                [0.0, math.nextafter(1.0, 2.0)], [float.fromhex("0x0.0000000000001p-1022"), -0.0], [0], [0])
    with tempfile.TemporaryDirectory(prefix="v71-exact-") as temporary:
        for byte_order, prefix in (("little", "<"), ("big", ">")):
            payload, descriptors = bytearray(MAGIC), []
            for name in NAMES:
                dtype, code, width = (("int64", "q", 8) if name in INTEGER_NAMES else
                                      ("uint8", "B", 1) if name == "inequality" else ("float64", "d", 8))
                values = a[name]
                descriptors.append({"name": name, "dtype": dtype, "width": width, "count": len(values),
                                    "shape": [len(values)], "offset": len(payload)})
                payload.extend(struct.pack(prefix + str(len(values)) + code, *values))
            metadata = {"schema": 1, "version": "v71", "status": "captured", "n": 1, "m": 2, "nnz": 1,
                        "event_count": 1, "attempts": 1, "capture_errors": 0, "truncated": False,
                        "error": "none",
                        "binary_bytes": len(payload), "byte_order": byte_order, "fixed_cache_bytes": 0,
                        "cache_limit_bytes": CAP, "metadata_limit_bytes": 65536, "payload_bytes": len(payload) - 8,
                        "cache_bytes": len(payload) - 8,
                        "events": [{"phase": "recovered_basis_before_assess", "index": 0, "arrays": descriptors}]}
            path = Path(temporary) / ("capture-" + byte_order + ".json")
            path.write_text(json.dumps(metadata), encoding="utf-8")
            path.with_suffix(".bin").write_bytes(payload)
            identity, _, loaded = load_capture(path)
            before = {name: values.tobytes() for name, values in loaded.items()}
            analyze_arrays(loaded)
            require(identity["binary_sha256"] == digest(payload) and identity["byte_order"] == byte_order and
                    all(values.tobytes() == before[name] for name, values in loaded.items()) and
                    all(list(loaded[name]) == list(a[name]) for name in NAMES) and
                    struct.pack("<d", loaded["candidate_x"][0]) == struct.pack("<d", -0.0) and
                    Fraction.from_float(loaded["candidate_z"][0]) == Fraction(1, 2 ** 1074),
                    "两端序原值、原点字节、负零或次正规数未保留")
    cases.append("两端序捕获回读及分析保持点原字节")
    return {"passed": True, "cases": len(cases), "checks": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--record-index", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        require(args.capture is None and args.runs is None and args.record_index is None,
                "自检不能同时分析运行数据")
        result = self_test()
    else:
        require(args.runs is not None and ((args.capture is None and args.record_index is None) or
                (args.capture is not None and args.record_index is not None)),
                "分析需要 --runs；单个捕获还需要同时提供 --capture 和 --record-index")
        require(digest((DATA / "v71-plan.json").read_bytes()) == PLAN_SHA, "冻结计划身份不匹配")
        if args.capture is None:
            result = analyze_runs(args.runs)
        else:
            source, record = load_run_record(args.runs, args.record_index)
            result = analyze_capture(args.capture, record)
            result["run_record"] = source
        result.update({"plan_sha256": PLAN_SHA,
                       "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps(result if args.output is None else {"output": str(args.output),
                     "passed": result.get("passed"), "status": result.get("status")},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
