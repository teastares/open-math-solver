"""只读分解 v59 失败常数 KKT 快照；全部恒等式使用精确分数，不运行求解器。"""

import argparse
import copy
import importlib.util
import json
import math
import struct
import time
from fractions import Fraction
from pathlib import Path


HELPER = Path(__file__).with_name("v53-analyze-hsd.py")
SPEC = importlib.util.spec_from_file_location("v59_exact_helpers", HELPER)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
digest = BASE.digest
hex_fraction = BASE.hex_fraction
packed = BASE.packed
vector_summary = BASE.vector_summary
exact_vector = BASE.exact_vector
validate_csc = BASE.validate_csc
ZERO = Fraction(0)
CAP = 128 * 1024 * 1024
RESERVE = 64 * 1024
MAGIC = b"V59KKT1\0"
FORMATS = {"f64_le": ("d", 8), "i64_le": ("q", 8), "u64_le": ("Q", 8)}
NAMES = ("g_starts", "g_rows", "g_values", "costs", "b", "diagonal_h", "regularized_h",
         "ap", "ai", "ax", "factor_values", "equilibration", "inverse", "retained_index",
         "singleton_column", "singleton_value", "rhs", "first_factor_rhs",
         "first_factor_solution", "first_reduced_rhs", "first_answer", "terminal_answer",
         "stored_residual")
INT_NAMES = {"g_starts", "g_rows", "ap", "ai", "retained_index", "singleton_column"}


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("整数元数据的类型或下界无效")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 对象含重复字段")
        result[key] = value
    return result


def add(*vectors):
    if not vectors:
        return []
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("参与精确向量加法的维数不同")
    return [sum(items, ZERO) for items in zip(*vectors)]


def subtract(left, right):
    return add(left, [-x for x in right])


def diagonal_product(diagonal, values):
    if len(diagonal) != len(values):
        raise ValueError("对角乘法维数不同")
    return [a * b for a, b in zip(diagonal, values)]


def exact_identity(left, right, label):
    defect = subtract(left, right)
    if any(defect):
        raise AssertionError(f"{label} 的逐坐标精确恒等式失败")
    return {"passed": True, "coordinates": len(defect), "defect": vector_summary(defect)}


def signed_vector(values):
    """稀疏保存完整有符号分数；省略项明确为零，范数只用作摘要。"""
    return {"length": len(values), "zero_omitted": True,
            "nonzero": [[i, str(value.numerator), str(value.denominator)]
                        for i, value in enumerate(values) if value],
            **vector_summary(values)}


def upper_product(starts, rows, values, point):
    answer = [ZERO] * len(point)
    for col in range(len(point)):
        for offset in range(starts[col], starts[col + 1]):
            row, value = rows[offset], values[offset]
            answer[row] += value * point[col]
            if row != col:
                answer[col] += value * point[row]
    return answer


def validate_arrays(a, s):
    n, m, dim = (integer(s[key]) for key in ("n", "m", "factor_dimension"))
    if n + m == 0 or not n <= dim <= n + m:
        raise ValueError("KKT 或消元后维数无效")
    integer(s["iteration"])
    integer(s["regularization_level"])
    expected = {"costs": n, "b": m, "diagonal_h": m, "regularized_h": m,
                "inverse": dim, "retained_index": m, "singleton_column": m,
                "singleton_value": m, "rhs": n + m, "first_factor_rhs": dim,
                "first_factor_solution": dim, "first_reduced_rhs": dim,
                "first_answer": n + m, "terminal_answer": n + m, "stored_residual": n + m}
    if set(a) != set(NAMES) or any(len(a[key]) != length for key, length in expected.items()):
        raise ValueError("快照数组缺失或维数不符")
    validate_csc(a["g_starts"], a["g_rows"], a["g_values"], n, m, True)
    validate_csc(a["ap"], a["ai"], a["ax"], dim, dim, True)
    if len(a["factor_values"]) != len(a["ax"]):
        raise ValueError("实际分解矩阵与原上三角模式维数不同")
    if len(a["equilibration"]) not in (0, dim) or any(x <= 0 for x in a["equilibration"]):
        raise ValueError("均衡向量维数或正性无效")
    if sorted(a["inverse"]) != list(range(dim)):
        raise ValueError("逆置换不是完整双射")
    if any(h < 0 or rh <= 0 or rh < h for h, rh in zip(a["diagonal_h"], a["regularized_h"])):
        raise ValueError("原对角或有效正则对角无效")
    if any(a["rhs"][col] != -a["costs"][col] for col in range(n)) or a["rhs"][n:] != a["b"]:
        raise ValueError("快照不是固定常数右端 [-c;b]")
    if any(hex_fraction(s[key]) < 0 for key in ("primal_regularization", "dual_regularization")):
        raise ValueError("报告正则值不能为负")
    if hex_fraction(s["tolerance"]) <= 0 or not isinstance(s["backend"], str) or not s["backend"]:
        raise ValueError("验收阈值或后端标识无效")
    row_entries = [[] for _ in range(m)]
    for col in range(n):
        for offset in range(a["g_starts"][col], a["g_starts"][col + 1]):
            row_entries[a["g_rows"][offset]].append((col, a["g_values"][offset]))
    retained = n
    expected_pattern = {(position, position) for position in range(dim)}
    for row in range(m):
        col, position = a["singleton_column"][row], a["retained_index"][row]
        if col >= 0:
            if col >= n or position != -1 or row_entries[row] != [(col, a["singleton_value"][row])]:
                raise ValueError("单行消元映射与完整 G 不符")
        else:
            if col != -1 or position != retained or a["singleton_value"][row] != 0:
                raise ValueError("保留行映射与实际构造顺序不符")
            retained += 1
            for column, _ in row_entries[row]:
                left, right = a["inverse"][column], a["inverse"][position]
                expected_pattern.add((min(left, right), max(left, right)))
    actual_pattern = set()
    for col in range(dim):
        for offset in range(a["ap"][col], a["ap"][col + 1]):
            row = a["ai"][offset]
            if row > col:
                raise ValueError("实际矩阵不是上三角 CSC")
            actual_pattern.add((row, col))
    if retained != dim or actual_pattern != expected_pattern:
        raise ValueError("消元后维数或实际矩阵模式与 G 不符")
    if not a["equilibration"] and a["factor_values"] != a["ax"]:
        raise ValueError("无均衡时实际分解矩阵应逐值等于 ax")
    return n, m, dim


def parse_snapshot(meta, binary):
    if meta.get("schema") != "v59_kkt_1" or meta.get("enabled") is not True:
        raise ValueError("快照 schema 或开启状态无效")
    if meta.get("cache_limit") != CAP or meta.get("metadata_reserve") != RESERVE:
        raise ValueError("受控缓存合同不符")
    for key in ("extra_lp_products", "extra_lp_solves", "diagnostic_clock_reads"):
        if type(meta.get(key)) is not int or meta[key] != 0:
            raise ValueError("捕获增加了未授权的在线计算")
    for key in ("cache_bytes", "pending_payload_bytes", "terminal_payload_bytes", "copied_bytes",
                "constant_calls", "target_failures", "capture_count", "binary_bytes"):
        integer(meta.get(key))
    if meta["cache_bytes"] > CAP or meta["capture_count"] not in (0, 1):
        raise ValueError("缓存或捕获次数越过合同")
    if meta["pending_payload_bytes"] + meta["terminal_payload_bytes"] + RESERVE != meta["cache_bytes"]:
        raise ValueError("缓存分项账本不守恒")
    if (meta.get("binary_written") is not True or meta["binary_bytes"] != len(binary)
            or len(binary) > CAP + len(MAGIC) or binary[:len(MAGIC)] != MAGIC):
        raise ValueError("二进制写出、魔数或长度无效")
    if meta.get("captured") is False:
        if (meta.get("status") not in ("no_target_failure", "not_entered") or meta["capture_count"] != 0
                or meta.get("arrays") != [] or len(binary) != len(MAGIC) or meta["target_failures"] != 0):
            raise ValueError("未捕获状态不是合法无触发控制")
        return None
    if (meta.get("captured") is not True or meta.get("status") != "captured" or meta.get("ready") is not True
            or meta.get("run_started") is not True or meta["capture_count"] != 1
            or meta["target_failures"] != 1 or meta["constant_calls"] == 0):
        raise ValueError("完整捕获状态不符")
    arrays, intervals = {}, []
    for item in meta.get("arrays", []):
        name, dtype, count, offset = (item[key] for key in ("name", "dtype", "count", "offset"))
        expected = "i64_le" if name in INT_NAMES else "u64_le" if name == "inverse" else "f64_le"
        if name not in NAMES or name in arrays or dtype != expected:
            raise ValueError("数组名称、重复字段或类型无效")
        integer(count)
        integer(offset, len(MAGIC))
        code, width = FORMATS[dtype]
        end = offset + count * width
        if end > len(binary):
            raise ValueError("数组越过二进制边界")
        arrays[name] = list(struct.unpack_from(f"<{count}{code}", binary, offset))
        if dtype == "f64_le" and any(not math.isfinite(x) for x in arrays[name]):
            raise ValueError("捕获数组含非有限数")
        intervals.append((offset, end))
    cursor = len(MAGIC)
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("数组存在空洞、重叠或重复 payload")
        cursor = end
    if cursor != len(binary):
        raise ValueError("二进制存在未声明尾部")
    if len(binary) - len(MAGIC) != meta["pending_payload_bytes"] + meta["terminal_payload_bytes"]:
        raise ValueError("二进制 payload 与缓存账本不符")
    validate_arrays(arrays, meta["scalars"])
    return arrays


class ExactKkt:
    """保存矩阵的精确有理数解释，不重新模拟任何浮点求解。"""

    def __init__(self, arrays, scalars):
        self.a, self.s = arrays, scalars
        self.n, self.m, self.dim = validate_arrays(arrays, scalars)
        self.f = {key: exact_vector(value) for key, value in arrays.items()
                  if key not in INT_NAMES and key != "inverse"}
        self.delta = hex_fraction(scalars["primal_regularization"])
        self.d = self.f["equilibration"] or [Fraction(1)] * self.dim
        self.reduced_diagonal = [self.delta] * self.n
        self.reduced_rhs = self.f["rhs"][:self.n] + [ZERO] * (self.dim - self.n)
        for row in range(self.m):
            col, position = arrays["singleton_column"][row], arrays["retained_index"][row]
            if col >= 0:
                value, denominator = self.f["singleton_value"][row], self.f["regularized_h"][row]
                self.reduced_diagonal[col] += value * value / denominator
                self.reduced_rhs[col] += value * self.f["rhs"][self.n + row] / denominator
            else:
                self.reduced_rhs[position] = self.f["rhs"][self.n + row]

    def permute(self, values):
        result = [ZERO] * self.dim
        for i, position in enumerate(self.a["inverse"]):
            result[position] = values[i]
        return result

    def unpermute(self, values):
        return [values[position] for position in self.a["inverse"]]

    def reduced_product(self, values):
        result = diagonal_product(self.reduced_diagonal, values[:self.n]) + [ZERO] * (self.dim - self.n)
        for row, position in enumerate(self.a["retained_index"]):
            if position >= 0:
                result[position] = -self.f["regularized_h"][row] * values[position]
        for col in range(self.n):
            for offset in range(self.a["g_starts"][col], self.a["g_starts"][col + 1]):
                position = self.a["retained_index"][self.a["g_rows"][offset]]
                if position >= 0:
                    value = self.f["g_values"][offset]
                    result[col] += value * values[position]
                    result[position] += value * values[col]
        return result

    def full_product(self, values, regularized):
        diagonal = self.f["regularized_h" if regularized else "diagonal_h"]
        result = [self.delta * value if regularized else ZERO for value in values[:self.n]]
        result += [-diagonal[row] * values[self.n + row] for row in range(self.m)]
        for col in range(self.n):
            for offset in range(self.a["g_starts"][col], self.a["g_starts"][col + 1]):
                row, value = self.a["g_rows"][offset], self.f["g_values"][offset]
                result[col] += value * values[self.n + row]
                result[self.n + row] += value * values[col]
        return result

    def retained(self, full):
        reduced = full[:self.n] + [ZERO] * (self.dim - self.n)
        for row, position in enumerate(self.a["retained_index"]):
            if position >= 0:
                reduced[position] = full[self.n + row]
        return reduced

    def lift(self, reduced):
        full = reduced[:self.n] + [ZERO] * self.m
        for row, position in enumerate(self.a["retained_index"]):
            if position >= 0:
                full[self.n + row] = reduced[position]
        return full

    def residuals(self, point):
        r0 = subtract(self.f["rhs"], self.full_product(point, False))
        rdelta = subtract(self.f["rhs"], self.full_product(point, True))
        regularization = [self.delta * value for value in point[:self.n]]
        regularization += [-(self.f["regularized_h"][row] - self.f["diagonal_h"][row]) * point[self.n + row]
                           for row in range(self.m)]
        identity = exact_identity(r0, add(rdelta, regularization), "原方程与有效正则方程")
        return {"r0": r0, "rdelta_effective": rdelta, "effective_regularization_contribution": regularization}, identity

    def first_direct(self):
        y, rhs_factor = self.f["first_factor_solution"], self.f["first_factor_rhs"]
        q = self.permute(self.reduced_rhs)
        qhat = self.permute(self.f["first_reduced_rhs"])
        dy = diagonal_product(self.d, y)
        ax_dy = upper_product(self.a["ap"], self.a["ai"], self.f["ax"], dy)
        factor_y = upper_product(self.a["ap"], self.a["ai"], self.f["factor_values"], y)
        ideal_dy_product = self.permute(self.reduced_product(self.unpermute(dy)))
        uhat = self.permute(self.retained(self.f["first_answer"]))
        recovery = subtract(uhat, dy)
        divide = lambda values: [value / scale for value, scale in zip(values, self.d)]
        terms_permuted = {
            "adapter_solve_defect": divide(subtract(rhs_factor, factor_y)),
            "singleton_matrix_formation": subtract(ax_dy, ideal_dy_product),
            "singleton_rhs_formation": subtract(q, qhat),
            "equilibration_matrix_conversion": divide(subtract(factor_y, diagonal_product(self.d, ax_dy))),
            "equilibration_rhs_conversion": divide(subtract(diagonal_product(self.d, qhat), rhs_factor)),
            "retained_recovery_conversion": [-x for x in self.permute(self.reduced_product(self.unpermute(recovery)))],
        }
        reduced_residual = subtract(q, self.permute(self.reduced_product(self.unpermute(uhat))))
        reduced_identity = exact_identity(reduced_residual, add(*terms_permuted.values()), "消元与均衡后的六项误差链")
        terms = {key: self.lift(self.unpermute(values)) for key, values in terms_permuted.items()}
        singleton_recovery = [ZERO] * (self.n + self.m)
        for row, col in enumerate(self.a["singleton_column"]):
            if col >= 0:
                value, denominator = self.f["singleton_value"][row], self.f["regularized_h"][row]
                ideal = (value * self.f["first_answer"][col] - self.f["rhs"][self.n + row]) / denominator
                error = self.f["first_answer"][self.n + row] - ideal
                singleton_recovery[col] -= value * error
                singleton_recovery[self.n + row] = denominator * error
        terms["singleton_recovery_conversion"] = singleton_recovery
        residuals, regularization_identity = self.residuals(self.f["first_answer"])
        full_identity = exact_identity(residuals["rdelta_effective"], add(*terms.values()), "完整有效正则方程的七项误差链")
        full_original_identity = exact_identity(residuals["r0"], add(*terms.values(), residuals["effective_regularization_contribution"]),
                                                "完整原方程的八项误差链")
        auxiliary = {"factor_input_residual": subtract(rhs_factor, factor_y),
                     "retained_recovery_error_permuted": recovery,
                     "reduced_residual_permuted": reduced_residual}
        return residuals, terms, auxiliary, {"reduced_chain": reduced_identity, "full_regularized_chain": full_identity,
                                            "full_original_chain": full_original_identity,
                                            "regularization": regularization_identity}


def selected_coordinates(vectors):
    positions = sorted({max(range(len(values)), key=lambda i: abs(values[i]))
                        for values in vectors.values() if values and any(values)})
    return [{"coordinate": i, "values": {name: packed(values[i]) for name, values in vectors.items()}}
            for i in positions]


def analyze_arrays(arrays, scalars):
    start = time.perf_counter()
    kkt = ExactKkt(arrays, scalars)
    first, chain, auxiliary, identities = kkt.first_direct()
    terminal, terminal_identity = kkt.residuals(kkt.f["terminal_answer"])
    terminal["stored_residual"] = kkt.f["stored_residual"]
    terminal["residual_evaluation_difference"] = subtract(terminal["stored_residual"], terminal["r0"])
    evaluation_identity = exact_identity(terminal["stored_residual"],
                                        add(terminal["r0"], terminal["residual_evaluation_difference"]), "生产残差形成差异")
    tolerance = hex_fraction(scalars["tolerance"])
    effective = subtract(kkt.f["regularized_h"], kkt.f["diagonal_h"])
    nominal = hex_fraction(scalars["dual_regularization"])
    first_vectors = dict(first, **chain)
    return {"analysis_status": "analyzed", "dimensions": {"n": kkt.n, "m": kkt.m,
            "factor_dimension": kkt.dim, "g_nonzeros": len(arrays["g_values"]), "upper_nonzeros": len(arrays["ax"])},
            "scalars": scalars, "sign_convention": "r = rhs - K*w",
            "effective_dual_regularization": {"nominal": packed(nominal), "vector": signed_vector(effective),
                "zero_count": sum(value == 0 for value in effective),
                "different_from_nominal_count": sum(value != nominal for value in effective)},
            "first_direct": {"identities": identities,
                "signed_vectors": {name: signed_vector(values) for name, values in first_vectors.items()},
                "auxiliary_signed_vectors": {name: signed_vector(values) for name, values in auxiliary.items()},
                "explanatory_coordinates": selected_coordinates(first_vectors),
                "original_residual_over_tolerance": packed(max(map(abs, first["r0"]), default=ZERO) / tolerance)},
            "terminal": {"identities": {"regularization": terminal_identity, "residual_evaluation": evaluation_identity},
                "signed_vectors": {name: signed_vector(values) for name, values in terminal.items()},
                "explanatory_coordinates": selected_coordinates(terminal),
                "exact_residual_over_tolerance": packed(max(map(abs, terminal["r0"]), default=ZERO) / tolerance),
                "stored_residual_over_tolerance": packed(max(map(abs, terminal["stored_residual"]), default=ZERO) / tolerance)},
            "analysis_seconds": time.perf_counter() - start}


def analyze(path):
    path = Path(path)
    binary_path = path.with_suffix(".bin")
    if path.stat().st_size > RESERVE or binary_path.stat().st_size > CAP + len(MAGIC):
        raise ValueError("输入文件超过冻结缓存合同")
    meta = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    binary = binary_path.read_bytes()
    arrays = parse_snapshot(meta, binary)
    result = {"metadata_path": str(path.resolve()), "metadata_sha256": digest(path),
              "binary_path": str(binary_path.resolve()), "binary_sha256": digest(binary_path),
              "capture_status": meta["status"], "capture_metadata": {key: value for key, value in meta.items() if key != "arrays"}}
    return dict(result, **analyze_arrays(arrays, meta["scalars"])) if arrays is not None else dict(result, analysis_status="not_captured")


def synthetic_case(eliminate=True, equilibrate=True, perturb=True):
    """只构造已知向量；适配器返回值人为给定，不解线性方程。"""
    n, m = 2, 3
    a = {"g_starts": [0, 2, 4], "g_rows": [0, 2, 1, 2], "g_values": [2.0, -1.0, 3.0, 4.0],
         "costs": [-3.0, 7.0], "b": [1.0, 2.0, 5.0], "diagonal_h": [0.0, 2.0**55, 3.0],
         "regularized_h": [1.0, 2.0**55, 4.0], "singleton_column": [0, 1, -1] if eliminate else [-1] * m,
         "singleton_value": [2.0, 3.0, 0.0] if eliminate else [0.0] * m,
         "retained_index": [-1, -1, 2] if eliminate else [2, 3, 4], "rhs": [3.0, -7.0, 1.0, 2.0, 5.0]}
    dim = 3 if eliminate else 5
    a["inverse"] = [2, 0, 1] if eliminate else [3, 1, 4, 0, 2]
    a["equilibration"] = [2.0**(i - 1) for i in range(dim)] if equilibrate else []
    delta, hreg = Fraction(1, 8), exact_vector(a["regularized_h"])
    diag, q = [delta] * n, exact_vector(a["rhs"][:n]) + [ZERO] * (dim - n)
    entries = {}
    for row in range(m):
        col, pos = a["singleton_column"][row], a["retained_index"][row]
        if col >= 0:
            value = Fraction(a["singleton_value"][row])
            diag[col] += value * value / hreg[row]
            q[col] += value * Fraction(a["rhs"][n + row]) / hreg[row]
        else:
            position = a["inverse"][pos]
            entries[position, position] = -hreg[row]
            q[pos] = Fraction(a["rhs"][n + row])
    for col in range(n):
        position = a["inverse"][col]
        entries[position, position] = diag[col]
        for offset in range(a["g_starts"][col], a["g_starts"][col + 1]):
            row = a["g_rows"][offset]
            if a["retained_index"][row] >= 0:
                other = a["inverse"][a["retained_index"][row]]
                entries[min(position, other), max(position, other)] = Fraction(a["g_values"][offset])
    a.update(ap=[0], ai=[], ax=[])
    for col in range(dim):
        for (row, column), value in sorted(entries.items()):
            if column == col:
                a["ai"].append(row)
                a["ax"].append(float(value))
        a["ap"].append(len(a["ai"]))
    if perturb and eliminate:
        a["ax"][a["ap"][a["inverse"][0] + 1] - 1] = math.nextafter(4.125, math.inf)
    d = a["equilibration"] or [1.0] * dim
    a["factor_values"] = [value * d[a["ai"][offset]] * d[col]
                          for col in range(dim) for offset, value in enumerate(a["ax"])
                          if a["ap"][col] <= offset < a["ap"][col + 1]]
    if perturb and equilibrate:
        a["factor_values"][0] = math.nextafter(a["factor_values"][0], math.inf)
    a["first_reduced_rhs"] = [float(value) for value in q]
    if perturb and eliminate:
        a["first_reduced_rhs"][0] = math.nextafter(a["first_reduced_rhs"][0], math.inf)
    a["first_factor_rhs"] = [0.0] * dim
    for i, pos in enumerate(a["inverse"]):
        a["first_factor_rhs"][pos] = a["first_reduced_rhs"][i] * d[pos]
    if perturb and equilibrate:
        a["first_factor_rhs"][0] = math.nextafter(a["first_factor_rhs"][0], math.inf)
    a["first_factor_solution"] = [1.25, -0.25, 2.0, -0.5, 0.75][:dim]
    u = [a["first_factor_solution"][pos] * d[pos] for pos in a["inverse"]]
    if perturb and equilibrate:
        u[0] = math.nextafter(u[0], math.inf)
    a["first_answer"] = u[:n] + [0.0] * m
    for row, col in enumerate(a["singleton_column"]):
        if col >= 0:
            a["first_answer"][n + row] = float((Fraction(a["singleton_value"][row]) * Fraction(u[col])
                                              - Fraction(a["rhs"][n + row])) / hreg[row])
        else:
            a["first_answer"][n + row] = u[a["retained_index"][row]]
    if perturb and eliminate:
        a["first_answer"][n] = math.nextafter(a["first_answer"][n], math.inf)
    a["terminal_answer"] = [math.nextafter(value, math.inf) for value in a["first_answer"]]
    a["stored_residual"] = [0.0] * (n + m)
    s = {"n": n, "m": m, "factor_dimension": dim, "iteration": 75, "regularization_level": 0,
         "primal_regularization": float(delta).hex(), "dual_regularization": 1.0.hex(),
         "tolerance": 1e-10.hex(), "backend": "normal_cholmod"}
    kkt = ExactKkt(a, s)
    terminal, _ = kkt.residuals(kkt.f["terminal_answer"])
    a["stored_residual"] = [float(value) for value in terminal["r0"]]
    return a, s


def synthetic_snapshot(arrays, scalars):
    binary, descriptors = bytearray(MAGIC), []
    for name in NAMES:
        dtype = "i64_le" if name in INT_NAMES else "u64_le" if name == "inverse" else "f64_le"
        values = arrays[name]
        descriptors.append({"name": name, "dtype": dtype, "count": len(values), "offset": len(binary)})
        binary += struct.pack(f"<{len(values)}{FORMATS[dtype][0]}", *values)
    payload = len(binary) - len(MAGIC)
    meta = {"schema": "v59_kkt_1", "enabled": True, "run_started": True, "run_unwound": True,
            "ready": True, "captured": True, "status": "captured", "cache_limit": CAP,
            "metadata_reserve": RESERVE, "cache_bytes": payload + RESERVE,
            "pending_payload_bytes": 0, "terminal_payload_bytes": payload, "copied_bytes": payload,
            "constant_calls": 76, "target_failures": 1, "capture_count": 1,
            "extra_lp_products": 0, "extra_lp_solves": 0, "diagnostic_clock_reads": 0,
            "binary_bytes": len(binary), "binary_written": True, "scalars": scalars, "arrays": descriptors}
    return meta, bytes(binary)


def self_test():
    checks = 0
    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("v59 纯合成自检失败")
        checks += 1
    for eliminate in (False, True):
        for equilibrate in (False, True):
            for perturb in (False, True):
                arrays, scalars = synthetic_case(eliminate, equilibrate, perturb)
                meta, binary = synthetic_snapshot(arrays, scalars)
                check(parse_snapshot(meta, binary) == arrays)
                result = analyze_arrays(arrays, scalars)
                check(all(item["passed"] for item in result["first_direct"]["identities"].values()))
                check(all(item["passed"] for item in result["terminal"]["identities"].values()))
                check(result["effective_dual_regularization"]["zero_count"] == 1)
                check(result["effective_dual_regularization"]["different_from_nominal_count"] == 1)
                if not eliminate:
                    check(not result["first_direct"]["signed_vectors"]["singleton_recovery_conversion"]["nonzero"])
                if not equilibrate:
                    check(not result["first_direct"]["signed_vectors"]["equilibration_matrix_conversion"]["nonzero"])
                    check(not result["first_direct"]["signed_vectors"]["equilibration_rhs_conversion"]["nonzero"])
    arrays, scalars = synthetic_case()
    result = analyze_arrays(arrays, scalars)
    for name in ("adapter_solve_defect", "singleton_matrix_formation", "singleton_rhs_formation",
                 "equilibration_matrix_conversion", "equilibration_rhs_conversion", "retained_recovery_conversion",
                 "singleton_recovery_conversion", "effective_regularization_contribution"):
        check(bool(result["first_direct"]["signed_vectors"][name]["nonzero"]))
    meta, binary = synthetic_snapshot(arrays, scalars)
    corruptions = []
    bad = copy.deepcopy(meta); bad["arrays"].append(copy.deepcopy(bad["arrays"][0])); corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["arrays"][1]["name"] = "unknown"; corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["arrays"][0]["count"] = True; corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["arrays"][1]["offset"] = 8; corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["scalars"]["n"] += 1; corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["cache_bytes"] = CAP + 1; corruptions.append((bad, binary))
    bad = copy.deepcopy(meta); bad["extra_lp_products"] = True; corruptions.append((bad, binary))
    corruptions.append((meta, binary[:-1]))
    corruptions.append((meta, binary + b"x"))
    nonfinite = bytearray(binary)
    value_offset = next(item["offset"] for item in meta["arrays"] if item["name"] == "g_values")
    struct.pack_into("<d", nonfinite, value_offset, math.nan); corruptions.append((meta, bytes(nonfinite)))
    for bad, bad_binary in corruptions:
        try:
            parse_snapshot(bad, bad_binary)
        except (ValueError, KeyError, TypeError, struct.error):
            check(True)
        else:
            check(False)
    for status in ("no_target_failure", "not_entered"):
        control = dict(meta, captured=False, status=status, capture_count=0, target_failures=0,
                       arrays=[], binary_bytes=len(MAGIC), terminal_payload_bytes=0, cache_bytes=RESERVE)
        check(parse_snapshot(control, MAGIC) is None)
    try:
        json.loads('{"same":1,"same":2}', object_pairs_hook=unique_object)
    except ValueError:
        check(True)
    else:
        check(False)
    for key, replacement in (("inverse", [0, 0, 1]), ("retained_index", [-1, -1, 1]),
                             ("singleton_value", [1.0, 3.0, 0.0]), ("rhs", [0.0] * 5)):
        changed = dict(arrays, **{key: replacement})
        try:
            validate_arrays(changed, scalars)
        except ValueError:
            check(True)
        else:
            check(False)
    try:
        exact_identity([Fraction(1)], [Fraction(-1)], "符号污染反例")
    except AssertionError:
        check(True)
    else:
        check(False)
    check(hex_fraction("0x8.000000000000001p-3") == 1 + Fraction(1, 1 << 63))
    return {"passed": True, "checks": checks, "kind": "pure_synthetic_no_solver",
            "arithmetic_cases": 8, "parser_rejections": len(corruptions), "solver_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    result = {"schema": "v59_kkt_analysis_1", "analyzer_sha256": digest(__file__), "helper_sha256": digest(HELPER),
              "self_test": self_test(), "records": [], "scope": [
                  "全部精确分数解释已保存的有限 binary64 数据，不重建未知理想解或重跑 LP。",
                  "首次 direct 的适配器项针对实际输入增广矩阵，包含适配器内部变换、分解和求解的合成缺陷，不能单独命名为因子误差。",
                  "消元 RHS 项合并扩展精度累加与转为 binary64 的舍入；撤均衡项只包含已保存输出与精确 D*y 的差异。",
                  "终端 Krylov 针对 K0，终端有效正则残差不能称为分解误差；各项范数不相加冒充向量恒等式。",
                  "有符号全量向量及最坏坐标用于展示相消关系，某项范数较大不能单独证明其对最终残差的因果占比或修改收益。",
                  "离线精确残差不覆盖或放宽生产验收，未触发控制不补造失败方程。"]}
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
        protected = {Path(__file__).resolve(), HELPER.resolve()}
        protected.update(Path(path).resolve() for path in args.inputs)
        protected.update(Path(path).with_suffix(".bin").resolve() for path in args.inputs)
        if args.output.resolve() in protected or args.output.exists():
            raise ValueError("分析输出不能覆盖源码、输入或已有证据")
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
