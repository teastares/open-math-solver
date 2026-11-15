"""用正常生产程序重放已冻结的 LP 协议，复用公共质量检查和统计。"""

import argparse
import copy
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import benchmark
import compare_lp


def read(path):
    return benchmark.strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stripped(report):
    result = copy.deepcopy(report)
    result.pop("timings", None)
    result.pop("backend_version", None)
    for field in ("presolve", "crossover", "diagnostics", "profile", "warm_dual_cleanup", "factor_reuse"):
        result.get("options", {}).get("ipm", {}).pop(field, None)
    return result


def typed_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(typed_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(typed_equal(x, y) for x, y in zip(a, b))
    return a == b


def gate(development, probe):
    def changes(summary, comparison):
        r, c = summary["reference_variant"], summary["candidate_variant"]
        return [item["input"] for item in summary["models"]
                if comparison(item["quality_pass_counts"][c], item["quality_pass_counts"][r])]
    dev = changes(development, lambda c, r: c < r)
    pro = changes(probe, lambda c, r: c < r)
    gain = changes(probe, lambda c, r: c > r)
    r, c = development["reference_variant"], development["candidate_variant"]
    a, b = development["variants"][r], development["variants"][c]
    speed = (b["penalized_numerical_sgm10_seconds"] < a["penalized_numerical_sgm10_seconds"]
             and b["penalized_wall_sgm10_seconds"] <= a["penalized_wall_sgm10_seconds"])
    return dict(development_lost=dev, probe_lost=pro, probe_gain=gain,
                development_speed_gate=speed, eligible_for_netlib=not dev and not pro and (speed or bool(gain)))


def check_protected(plan):
    for path, digest in plan["protected"].items():
        if benchmark.sha256(Path(path)) != digest:
            raise ValueError("冻结输入改变：" + path)


def digest(path):
    try:
        return benchmark.sha256(Path(path)), None
    except OSError as error:
        return None, str(error)


def load_validator(plan):
    spec = importlib.util.spec_from_file_location("frozen_lp_validation", plan["validation_protocol"])
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    return validator


def check_command(item):
    command = item["command"]
    if len(command) < 3 or command[1] != "solve" or (len(command) - 3) % 2:
        raise ValueError("冻结命令必须由独立的开关和值组成")
    options = dict(zip(command[3::2], command[4::2]))
    if len(options) != (len(command) - 3) // 2 or any(not key.startswith("--") or " " in key for key in options):
        raise ValueError("冻结命令包含重复开关或合并参数")
    ipm = item["expected_options"].get("ipm")
    for name in ("presolve", "crossover", "diagnostics", "profile", "warm_dual_cleanup", "factor_reuse"):
        if isinstance(ipm, dict) and name in ipm:
            if type(ipm[name]) is not bool or options.get("--ipm-" + name.replace("_", "-")) != ("on" if ipm[name] else "off"):
                raise ValueError("正式选项与冻结请求不符")
    if isinstance(ipm, dict) and ipm.get("factor_reuse") is True and (
            item["variant"] != "diagnostic_on" or ipm.get("diagnostics") is not True):
        raise ValueError("复用候选评测须启用完整诊断账本")
    if isinstance(ipm, dict) and ipm.get("profile") is True:
        raise ValueError("本固定公开协议保持profile关闭，不接受额外性能标记")


REUSE_MARKER = "V122_FACTOR_REUSE "
REUSE_PUBLIC = dict(constant_solves="ipm_homogeneous_constant_solves",
    direction_solves="ipm_homogeneous_direction_solves", scaled_products="ipm_homogeneous_scaled_products",
    inexact_returns="ipm_homogeneous_scaled_inexact_returns", refinement_steps="ipm_refinement_steps",
    scaled_refinement_steps="ipm_homogeneous_scaled_refinement_steps", factorizations="ipm_factorizations")
REUSE_STATUSES = {"running", "accepted", "initial_rejected", "numerical_error", "budget", "exception"}
CURRENT_DIAGONAL_FIELDS = (
    "columns", "rows", "nonzeros", "current_rows_started", "current_rows_completed", "current_rows_terms",
    "validation_started", "validation_completed", "validation_terms", "applications_started", "applications_completed",
    "transpose_started", "transpose_completed", "direct_started", "direct_completed", "product_started",
    "product_completed", "transpose_terms", "product_terms", "row_divisions", "budget_checks")
TRANSPORT_FIELDS = (
    "columns", "rows", "nonzeros", "cache_epoch", "prepare_started", "prepare_completed", "cache_hits",
    "old_build_started", "old_build_completed", "old_row_terms", "scratch_row_terms", "merge_terms", "unique_terms",
    "old_terms", "current_terms", "column_terms", "input_started", "input_completed", "input_terms",
    "direct_started", "direct_completed", "output_started", "output_completed", "output_terms", "budget_checks",
    "old_min", "old_max", "current_min", "current_max", "scale_min", "scale_max")
TRANSPORT_RANGES = ("old_min", "old_max", "current_min", "current_max", "scale_min", "scale_max")
NATURAL_SCHUR_COUNTS = (
    "started", "completed", "terms", "budget_checks", "environment_checks", "divisions", "products",
    "accumulations", "abs_accumulations", "final_subtractions", "bound_operations", "enclosure_steps")
NATURAL_SCHUR_NUMBERS = ("natural", "abs_products", "lower", "upper", "error_bound", "margin")


def check_diagonal_transport(value, current, coupled, energy, delta, status, epoch, previous):
    """只认证已声明的对角合同准备、缓存世代及两次缩放的费用前缀。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def integer(number, limit=9223372036854775807):
        return type(number) is int and 0 <= number <= limit
    require(isinstance(value, dict) and set(value) == set(TRANSPORT_FIELDS) and
        all(integer(value[key], 18446744073709551615 if key in {"columns", "rows", "nonzeros"} else
            9223372036854775807) for key in TRANSPORT_FIELDS if key not in TRANSPORT_RANGES) and
        all(value[key] is None or benchmark.finite_real(value[key]) and
            value[key] >= 0 for key in TRANSPORT_RANGES),
        "对角合同账本字段或类型无效")
    n, m, nonzeros = (current[key] for key in ("columns", "rows", "nonzeros"))
    require(all(value[key] == current[key] for key in ("columns", "rows", "nonzeros")) and
        value["cache_epoch"] in (0, epoch) and
        (value["prepare_started"] == 0 or value["cache_epoch"] == epoch), "对角合同维数或成功因子世代无效")
    for name in ("prepare", "old_build", "input", "direct", "output"):
        require(0 <= value[name + "_started"] - value[name + "_completed"] <= 1,
            "对角合同阶段开始/完成前缀无效")
    require(value["prepare_started"] <= current["validation_completed"] and
        value["prepare_started"] <= 1 and value["cache_hits"] + value["old_build_started"] <= value["prepare_started"] and
        value["column_terms"] <= n * value["prepare_started"] and
        value["old_row_terms"] <= m * value["old_build_started"] and
        value["scratch_row_terms"] <= 2 * m * value["prepare_started"] and
        value["merge_terms"] <= nonzeros * value["prepare_started"] and
        value["current_terms"] <= value["unique_terms"] <= value["merge_terms"] and
        value["old_terms"] <= value["unique_terms"], "对角合同准备或合并费用前缀无效")
    expected_hit = previous is not None and previous["status"] == "accepted"
    if value["cache_hits"] or value["old_build_started"]:
        require(bool(value["cache_hits"]) == expected_hit, "对角合同缓存命中与fresh/成功世代推进不符")
    if value["cache_hits"]:
        require(value["old_build_started"] == value["old_build_completed"] ==
            value["old_row_terms"] == value["old_terms"] == 0, "对角合同缓存命中仍重复计旧构造")
    if value["old_build_completed"]:
        require(value["old_row_terms"] == m and value["old_terms"] == value["unique_terms"] and
            value["column_terms"] == n, "旧Schur对角完整构造缺少实际行、项或列费用")
    if value["prepare_completed"]:
        require(value["cache_hits"] + value["old_build_completed"] == 1 and
            value["column_terms"] == n and value["scratch_row_terms"] == 2 * m and
            value["merge_terms"] == nonzeros and value["current_terms"] == value["unique_terms"],
            "完整对角合同准备缺少缓存决策或精确费用")
    require(coupled["constant_calls_started"] == 0 or value["prepare_completed"] == 1,
        "对角合同未准备完便开始常量或方向")
    require(value["input_started"] <= 5 and (value["input_started"] == 0 or value["prepare_completed"] == 1) and
        0 <= current["direct_started"] - value["input_started"] <= 1 and
        0 <= value["input_completed"] - value["direct_started"] <= 1 and
        0 <= value["direct_completed"] - value["output_started"] <= 1 and
        current["direct_completed"] == value["output_completed"], "对角合同缩放与旧直接解的执行顺序无效")
    for name in ("input", "output"):
        require(n * value[name + "_completed"] <= value[name + "_terms"] <= n * value[name + "_started"],
            "对角合同两次缩放的列费用不守恒")
    calls = delta["constant_solves"] + delta["direction_solves"]
    require(value["direct_completed"] <= calls <= value["direct_started"],
        "对角合同旧直接解与实际公共调用不守恒")
    minimum_budget_checks = (3 * value["prepare_completed"] + value["prepare_started"] - value["prepare_completed"] +
        5 * value["output_completed"] + value["input_started"] - value["output_completed"])
    require(value["budget_checks"] >= minimum_budget_checks, "对角合同准备或两次缩放少计预算检查")
    for prefix in ("old", "current", "scale"):
        lower, upper = value[prefix + "_min"], value[prefix + "_max"]
        require(lower is None or upper is None or lower <= upper, "对角合同范围摘要次序无效")
        if prefix == "scale":
            require(all(item is None or item > 0 for item in (lower, upper)), "对角合同已存缩放范围必须有限正")
    if n == 0:
        require(all(value[key] is None for key in TRANSPORT_RANGES), "空原始列不能生成对角合同范围")
    elif value["prepare_completed"]:
        require(value["scale_min"] is not None and value["scale_max"] is not None,
            "完整对角合同缺少已存有限缩放范围")
    if status in {"accepted", "bounded_rejected"}:
        applications = current["applications_completed"]
        require(value["prepare_completed"] == 1 and all(value[key] == applications for key in (
            "input_started", "input_completed", "direct_started", "direct_completed", "output_started", "output_completed")),
            "完整对角合同探测存在未完成的输入、旧直接解或输出缩放")


def check_scalar_consistency(value, current, coupled, energy, status):
    """标量差异只是观察，不据窄化摘要反推完整Schur精度或修改验收。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    counts = ("started", "completed", "terms", "budget_checks")
    numbers = ("natural_denominator", "positive_denominator", "relative_difference")
    require(isinstance(value, dict) and set(value) == set(counts + numbers) and
        all(type(value[key]) is int and 0 <= value[key] <= 9223372036854775807 for key in counts) and
        all(value[key] is None or benchmark.finite_real(value[key]) for key in numbers),
        "标量一致性观察的字段或类型无效")
    terms = current["columns"] + current["rows"]
    require(0 <= value["completed"] <= value["started"] <= 1 and
        terms * value["completed"] <= value["terms"] <= terms * value["started"] and
        value["budget_checks"] >= 2 * value["completed"] + value["started"] - value["completed"] and
        (value["started"] == 0 or energy["completed"] == 1) and
        (coupled["directions_started"] == 0 or value["completed"] == 1),
        "标量一致性观察的能量、方向顺序或预算费用无效")
    relative = value["relative_difference"]
    require(relative is None or -2 <= relative <= 2, "标量相对差异超出归一化范围")
    require(all(value[key] is None for key in numbers) or
        value["started"] == 1 and value["terms"] == terms,
        "标量观察在完整扫描前发布了半份摘要")
    if value["completed"]:
        require(energy["last"] is not None and
            typed_equal(value["positive_denominator"], energy["last"]["denominator"]),
            "标量观察的正分母与同组能量摘要不一致")
    if status in {"accepted", "bounded_rejected"}:
        require(value["completed"] == 1, "完整探测缺少已完成的标量一致性观察")


def check_natural_schur(value, current, coupled, scalar, status):
    """认证有符号自然分母的完整控制流和费用，不把double摘要当作LD证书。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(isinstance(value, dict) and set(value) == set(NATURAL_SCHUR_COUNTS + NATURAL_SCHUR_NUMBERS + ("decision",)) and
        all(type(value[key]) is int and 0 <= value[key] <= 9223372036854775807 for key in NATURAL_SCHUR_COUNTS) and
        all(value[key] is None or benchmark.finite_real(value[key]) for key in NATURAL_SCHUR_NUMBERS) and
        type(value["decision"]) is str and value["decision"] in {
            "not_started", "running", "reliable", "uncertain", "nonfinite", "unsupported"},
        "自然Schur包围账本的字段或类型无效")
    k = current["columns"] + current["rows"]
    started, completed, terms = (value[key] for key in ("started", "completed", "terms"))
    budget, environment = value["budget_checks"], value["environment_checks"]
    decision = value["decision"]
    require(0 <= completed <= started <= 1 and terms <= k * started and
        (budget == 0 or scalar["completed"] == 1) and coupled["directions_started"] <= 2 * completed and
        0 <= budget - environment <= 1 and environment >= started,
        "自然Schur的旧观察、方向顺序或环境检查前缀无效")
    counts = [value[key] for key in ("divisions", "products", "accumulations", "abs_accumulations")]
    require(max(0, terms - 1) <= counts[-1] <= counts[-2] <= counts[-3] <= counts[0] <= terms and
        value["final_subtractions"] <= 1 and (value["final_subtractions"] == 0 or
            started == 1 and terms == k and all(count == k for count in counts)),
        "自然Schur中心除法、乘法、累加或末减费用前缀无效")
    bound, enclosure = value["bound_operations"], value["enclosure_steps"]
    if value["final_subtractions"]:
        extra_bound, extra_enclosure = bound - 4 * k, enclosure - 6 * k
        require(0 <= extra_bound <= 4 and max(0, extra_bound - 1) <= extra_enclosure <= extra_bound,
            "自然Schur末区间或误差距离的费用前缀无效")
    elif terms == 0:
        require(bound == enclosure == 0, "自然Schur尚无坐标却产生区间运算")
    else:
        extra_bound, extra_enclosure = bound - 4 * (terms - 1), enclosure - 6 * (terms - 1)
        require(0 <= extra_bound <= 4 and (0 <= extra_enclosure <= 2 if extra_bound == 0 else
            extra_bound + 1 <= extra_enclosure <= extra_bound + 2) and
            (counts[-1] == terms or extra_bound == extra_enclosure == 0),
            "自然Schur逐项外包围与中心算术顺序不符")
    full = (started == 1 and terms == k and all(count == k for count in counts) and
        value["final_subtractions"] == 1 and bound == 4 * k + 4 and enclosure == 6 * k + 4)
    minimum_budget = started + (terms + 4095) // 4096
    additional_budget = int(started == 1 and (terms < k and terms % 4096 == 0 or full))
    require(minimum_budget <= environment <= budget <= minimum_budget + additional_budget if started else budget <= 1,
        "自然Schur预算检查与已进入坐标或末检查不符")
    published = any(value[key] is not None for key in NATURAL_SCHUR_NUMBERS)
    require(not published or full and decision in {"reliable", "uncertain", "unsupported"},
        "自然Schur在完整运算前发布了部分摘要")
    require(decision != "not_started" or started == 0 and not published and
        (environment == 0 or status == "exception" and environment == budget == 1),
        "未开始的自然Schur账本已经进入算术或发布摘要")
    require(decision != "running" or started == 1 and completed == 0 and not published,
        "运行中的自然Schur账本含交付或已发布摘要")
    require(decision not in {"reliable", "uncertain"} or full,
        "自然Schur决定缺少完整外包围费用")
    require(decision != "nonfinite" or started == 1 and completed == 0 and not published,
        "自然Schur非有限拒绝的实际前缀无效")
    require(decision != "unsupported" or completed == 0 and environment == budget and environment >= 1,
        "自然Schur控制环境拒绝缺少真实检查")
    require(completed == 0 or decision == "reliable", "自然Schur把未可靠分母标成已完成")
    final_budget = 2 + (k + 4095) // 4096
    if decision in {"reliable", "uncertain"}:
        require(budget == final_budget and (decision != "reliable" or completed == 1 or environment == budget - 1),
            "自然Schur决定缺少末预算或伪造未交付可靠前缀")
    if completed:
        require(budget == environment == final_budget, "自然Schur交付少计完整预算或环境检查")
    for key in ("abs_products", "error_bound", "margin"):
        require(value[key] is None or value[key] >= 0, "自然Schur非负观察或包围摘要为负")
    lower, upper, natural = (value[key] for key in ("lower", "upper", "natural"))
    require(lower is None or upper is None or lower <= upper, "自然Schur包围端点次序无效")
    if decision == "reliable":
        require((lower is None or natural is None or lower <= natural) and
            (upper is None or natural is None or natural <= upper) and
            not (lower is not None and upper is not None and lower < 0 < upper),
            "自然Schur可靠摘要与有序包围或同号端点矛盾")
    if decision == "uncertain":
        require(value["margin"] is None or value["margin"] == 0,
            "自然Schur未可靠时伪造了正的非零下界")
    if status in {"accepted", "bounded_rejected"}:
        require(completed == 1, "完整方向组缺少已交付的自然Schur证书")


def check_current_diagonal(value, coupled, energy, delta, status):
    """验证当前R构造、一次CSC验证及每次结构预条件应用的真实前缀。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def integer(number, limit=9223372036854775807):
        return type(number) is int and 0 <= number <= limit
    require(isinstance(value, dict) and set(value) == set(CURRENT_DIAGONAL_FIELDS) and
        all(integer(value[key], 18446744073709551615 if key in {"columns", "rows", "nonzeros"} else
            9223372036854775807) for key in CURRENT_DIAGONAL_FIELDS), "当前R结构账本字段或类型无效")
    n, m, nonzeros = (value[key] for key in ("columns", "rows", "nonzeros"))
    for name in ("current_rows", "validation", "applications", "transpose", "direct", "product"):
        require(0 <= value[name + "_started"] - value[name + "_completed"] <= 1,
            "当前R结构阶段的开始/完成前缀无效")
    require(value["current_rows_started"] <= 1 and value["validation_started"] <= 1 and
        value["validation_started"] <= value["current_rows_completed"], "当前R构造或CSC验证重复、顺序无效")
    for name, terms in (("current_rows", m), ("validation", n + 1 + m + nonzeros),
                        ("transpose", nonzeros), ("product", nonzeros)):
        require(value[name + "_completed"] * terms <= value[name + "_terms"] <= value[name + "_started"] * terms,
            "当前R构造、验证或矩阵乘积项数不守恒")
    started, completed = value["applications_started"], value["applications_completed"]
    minimum_budget_checks = (3 * value["validation_completed"] +
        value["validation_started"] - value["validation_completed"] +
        8 * completed + started - completed)
    require(value["budget_checks"] >= minimum_budget_checks,
        "当前R验证或结构应用少计了已完成及已开始的预算检查")
    require(started <= 5 and (started == 0 or
        value["current_rows_completed"] == value["validation_completed"] == 1),
        "当前R未完成验证便应用，或应用超过五次")
    require(all(0 <= value[left] - value[right] <= 1 for left, right in (
        ("applications_started", "transpose_started"), ("transpose_completed", "direct_started"),
        ("direct_completed", "product_started"), ("product_completed", "applications_completed"))),
        "当前R应用未按转置、旧直接解、正向乘积、恢复的顺序推进")
    calls_started = coupled["constant_calls_started"] + coupled["preconditioner_started"]
    calls_completed = coupled["constant_calls_completed"] + coupled["preconditioner_completed"]
    require(0 <= calls_started - started <= 1 and 0 <= completed - calls_completed <= 1,
        "当前R应用与常量/方向回调的前缀不符")
    direct_calls = delta["constant_solves"] + delta["direction_solves"]
    require(value["direct_completed"] <= direct_calls <= value["direct_started"],
        "当前R旧直接解与实际公共调用不守恒")
    require(2 * m * completed <= value["row_divisions"] <= 2 * m * started,
        "当前R每次应用的行除法费用不守恒")
    if energy["last"] is not None:
        require(energy["last"]["columns"] == n and energy["last"]["rows"] == m,
            "当前R正近似能量的维数与同组结构账本不符")
    if status in {"accepted", "bounded_rejected"}:
        require(value["current_rows_started"] == value["current_rows_completed"] ==
            value["validation_started"] == value["validation_completed"] == 1 and
            started == completed == calls_completed and calls_started == calls_completed and
            all(value[key] == completed for key in ("transpose_started", "transpose_completed", "direct_started",
                "direct_completed", "product_started", "product_completed")) and
            value["row_divisions"] == 2 * m * completed,
            "完整当前R探测未完整支付构造、验证或每次结构预条件费用")


def inspect_reuse_marker(stderr, enabled):
    lines = [line for line in stderr.splitlines() if line.startswith("V122_FACTOR_REUSE")]
    if not enabled:
        if lines:
            raise ValueError("关闭路径出现因子复用标记")
        return None
    if len(lines) != 1 or not lines[0].startswith(REUSE_MARKER):
        raise ValueError("因子复用标记必须恰好出现一次且前缀合法")
    return benchmark.strict_json(lines[0][len(REUSE_MARKER):])


def unknown_marker(stderr):
    prefixes = ("V100_PRESOLVE ", "V100_COUPLED ", "V100_ENERGY ", "V117_CROSSOVER ", REUSE_MARKER)
    return any(re.match(r"^V\d+_[A-Z0-9_]+(?:\s|$)", line) and not line.startswith(prefixes)
               for line in stderr.splitlines())


def reuse_selected_report(value, row, validator):
    """验证实际总费用及独立探测，再仅为旧合同生成主路径计数副本。"""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def integer(number, limit=9223372036854775807):
        return type(number) is int and 0 <= number <= limit
    report = row["report"]
    require(isinstance(value, dict) and type(value.get("schema")) is int and value["schema"] in (1, 2, 3, 4, 5),
        "复用schema类型或版本无效")
    bounded, structural, transported = value["schema"] >= 2, value["schema"] >= 3, value["schema"] >= 4
    natural = value["schema"] >= 5
    fields = {"schema", "enabled", "fresh_started", "fresh_completed", "reused_steps", "events"}
    if bounded:
        fields.add("max_krylov_steps")
    if structural:
        fields.update(("preconditioner", "energy_model"))
    if natural:
        fields.add("scalar_model")
    require(set(value) == fields and value["enabled"] is True and (not bounded or
        type(value["max_krylov_steps"]) is int and value["max_krylov_steps"] == 1),
        "复用顶层字段或类型无效")
    require(not structural or value["preconditioner"] == (
        "current_diagonal_congruence" if transported else "current_diagonal_schur") and
        value["energy_model"] == "current_positive_approximation", "当前R预条件器或正近似能量身份无效")
    require(not natural or value["scalar_model"] == "natural_schur_enclosed", "自然Schur标量模型身份无效")
    require(isinstance(report, dict) and integer(report.get("ipm_iterations"), 200), "复用公开迭代数无效")
    iterations = report["ipm_iterations"]
    maximum = report.get("options", {}).get("ipm", {}).get("max_iterations")
    require(integer(maximum, 200), "复用迭代预算无效")
    outer_limit = min(maximum, iterations + 1)
    fresh, complete, reused = (value[key] for key in ("fresh_started", "fresh_completed", "reused_steps"))
    require(all(integer(item, 200) for item in (fresh, complete, reused)) and complete <= fresh <= complete + 1,
        "复用fresh调用前缀无效")
    events = value["events"]
    require(isinstance(events, list) and len(events) <= 199 and
        fresh + len(events) <= max(0, 2 * outer_limit - 1), "复用总尝试超过原外层有界合同")
    require(iterations <= fresh + reused <= iterations + 1 and
        iterations <= complete + reused <= iterations + 1, "复用采用数与真实外层进度不符")
    discarded = dict.fromkeys(REUSE_PUBLIC, 0)
    accepted_counts = dict.fromkeys(validator.COUPLED_COUNTS, 0)
    accepted_energy = dict.fromkeys(validator.ENERGY_COUNTS, 0)
    accepted, rejected, previous = 0, 0, None
    for index, event in enumerate(events):
        event_fields = {"attempt_iteration", "factor_epoch", "status", "coupled", "energy", "public_delta"}
        if structural:
            event_fields.add("current_diagonal")
        if transported:
            event_fields.update(("transport", "scalar_consistency"))
        if natural:
            event_fields.add("natural_schur")
        require(isinstance(event, dict) and set(event) == event_fields, "复用事件字段无效")
        attempt, epoch, status = (event[key] for key in ("attempt_iteration", "factor_epoch", "status"))
        statuses = (REUSE_STATUSES - {"initial_rejected"}) | {"bounded_rejected"} if bounded else REUSE_STATUSES
        require(integer(attempt, outer_limit) and attempt == index + 2 and integer(epoch, 199) and
            1 <= epoch < attempt and type(status) is str and status in statuses, "复用世代或调用顺序无效")
        expected_epoch = 1 if previous is None else (
            previous["factor_epoch"] if previous["status"] == "accepted" else previous["attempt_iteration"])
        require(epoch == expected_epoch and (previous is None or previous["status"] in (
            {"accepted", "bounded_rejected", "numerical_error"} if bounded else
            {"accepted", "initial_rejected", "numerical_error"})), "复用世代混用或在终止后继续探测")
        if status in {"running", "budget", "exception"}:
            require(index == len(events) - 1, "未完成探测只能是最后前缀")
            require(report.get("status") == "time_limit" if status == "budget" else
                report.get("status") in {"error", "numerical_error", "time_limit"},
                "未完成探测不能与成功报告共存，预算前缀须报告time_limit")
        coupled, energy, delta = (event[key] for key in ("coupled", "energy", "public_delta"))
        require(isinstance(delta, dict) and set(delta) == set(REUSE_PUBLIC) and
            all(integer(item) for item in delta.values()), "复用公共增量类型无效")
        require(delta["factorizations"] == 0 and (bounded or
            delta["refinement_steps"] == delta["scaled_refinement_steps"] == 0),
            "旧因子探测进行了未声明的分解或FG改进")
        normalized = copy.deepcopy(coupled)
        if isinstance(normalized, dict) and isinstance(normalized.get("last"), dict) and normalized["last"].get("status") == "initial_rejected":
            require(not bounded, "单步生产路径不接受初P专用拒绝状态")
            normalized["last"]["status"] = "numerical_error"
        if isinstance(normalized, dict) and isinstance(normalized.get("last"), dict) and normalized["last"].get("status") == "bounded_rejected":
            require(bounded, "零步生产路径不接受单步拒绝状态")
            normalized["last"]["status"] = "exhausted"
        require(validator.valid_coupled(normalized, True) and validator.valid_energy(energy, True),
            "复用探测的原联立或能量合同无效")
        require(coupled["constant_calls_started"] <= 1 and coupled["directions_started"] <= 2 and
            coupled["directions_started"] <= 2 * coupled["constant_calls_completed"], "复用探测超过整对方向合同")
        if bounded:
            directions, krylov = coupled["directions_started"], coupled["krylov_started"]
            rejected_initials = coupled["initial_forcing_checks"] - coupled["initial_forcing_returns"]
            require(krylov <= directions and 0 <= coupled["cycles_started"] - krylov <= 1 and
                0 <= krylov - coupled["krylov_completed"] <= 1 and
                coupled["cycles_started"] <= directions and
                krylov <= rejected_initials and coupled["cycles_started"] <= rejected_initials and
                rejected_initials <= coupled["krylov_completed"] + directions - coupled["directions_completed"] and
                coupled["preconditioner_started"] <= 2 * directions and
                coupled["products_started"] <= 3 * directions and
                coupled["true_checks_started"] <= 2 * directions,
                "复用探测超过每方向一个窗口和一个FG步")
            refinements = delta["scaled_refinement_steps"]
            initial_calls = delta["direction_solves"] - refinements
            require(delta["refinement_steps"] == refinements and
                coupled["krylov_completed"] <= refinements <= krylov and
                coupled["directions_completed"] <= initial_calls <= directions,
                "复用初P与FG精化的公共费用不守恒")
            proof = coupled["last"]["candidate_forcing"] if coupled["last"] is not None else None
            require(proof is None or proof["cycle"] <= 1 and proof["krylov_step"] <= 1,
                "复用末方向证明超过一个窗口和一个FG步")
        else:
            require(all(coupled[key] == 0 for key in ("krylov_started", "krylov_completed", "cycles_started",
                    "candidate_forcing_checks", "candidate_forcing_returns")) and
                all(coupled[key] <= coupled["directions_started"] for key in (
                    "preconditioner_started", "products_started", "true_checks_started")), "复用探测超过一次初P合同")
        require(validator.valid_energy_public(energy, normalized, dict(ipm_iterations=attempt)), "复用常量与能量费用不守恒")
        for private, public in (("constant_calls", "constant_solves"), ("preconditioner", "direction_solves"),
                                ("products", "scaled_products")):
            require(coupled[private + "_completed"] <= delta[public] <= coupled[private + "_started"],
                "复用实际公共调用与私有前缀不符")
        if not bounded:
            require(coupled["directions_completed"] <= delta["direction_solves"] <= coupled["directions_started"],
                "复用Newton方向与一次P调用不符")
        last = coupled["last"]
        if bounded:
            require((status == "bounded_rejected") == (last is not None and last["status"] == "bounded_rejected") and
                (last is None or last["status"] != "exhausted"),
                "单步有限拒绝类型与原工作状态不符")
        if last is not None:
            require(last["attempt_iteration"] == attempt and last["role"] == (
                "affine" if coupled["directions_started"] == 1 else "corrector"), "复用方向角色或迭代号无效")
        if energy["last"] is not None:
            require(energy["last"]["attempt_iteration"] == attempt, "复用能量来自其它点")
        if structural:
            check_current_diagonal(event["current_diagonal"], coupled, energy, delta, status)
        if transported:
            check_diagonal_transport(event["transport"], event["current_diagonal"], coupled, energy, delta,
                status, epoch, previous)
            check_scalar_consistency(event["scalar_consistency"], event["current_diagonal"], coupled, energy, status)
        if natural:
            check_natural_schur(event["natural_schur"], event["current_diagonal"], coupled,
                event["scalar_consistency"], status)
        missing_delivery = (coupled["initial_forcing_returns"] +
            (coupled["candidate_forcing_returns"] if bounded else 0) - delta["inexact_returns"])
        require(missing_delivery in (0, 1), "复用forcing交付费用不符")
        if missing_delivery:
            require(status in {"numerical_error", "budget", "exception"} and last is not None and
                last["status"] in ({"forced_initial", "forced_candidate"} if bounded else {"forced_initial"}) and
                coupled["directions_started"] == coupled["directions_completed"],
                "未完成宽方向交付缺少真实前缀")
            if bounded:
                require(all(coupled[name + "_started"] == coupled[name + "_completed"] for name in (
                    "constant_calls", "directions", "preconditioner", "products", "krylov", "true_checks")) and
                    energy["started"] == energy["completed"] == coupled["constant_calls_completed"] and
                    energy["last"] is not None and energy["last"]["attempt_iteration"] == attempt,
                    "单步forcing未交付缺少完整真实异常前缀")
        if status == "accepted":
            require(coupled["constant_calls_completed"] == energy["completed"] == 1 and
                coupled["directions_started"] == coupled["directions_completed"] == 2 and
                last["status"] in ({"accepted", "forced_initial", "forced_candidate"} if bounded else
                    {"accepted", "forced_initial"}) and missing_delivery == 0,
                "整组复用并未完整验收两个方向")
            if bounded:
                steps = coupled["krylov_completed"]
                require(coupled["krylov_started"] == coupled["cycles_started"] == steps and
                    rejected_initials == steps and
                    delta["scaled_refinement_steps"] == steps and
                    all(coupled[key] == 2 + steps for key in ("preconditioner_started", "preconditioner_completed",
                        "true_checks_started", "true_checks_completed")) and
                    all(coupled[key] == 2 + 2 * steps for key in ("products_started", "products_completed")) and
                    delta["constant_solves"] == 1 and delta["direction_solves"] == 2 + steps and
                    delta["scaled_products"] == 2 + 2 * steps,
                    "单步整组复用的完整FG费用不守恒")
            else:
                require(all(coupled[key] == 2 for key in ("preconditioner_started", "preconditioner_completed",
                    "products_started", "products_completed", "true_checks_started", "true_checks_completed")),
                    "整组复用并未完整验收两个方向")
            accepted += 1
            for key in accepted_counts:
                accepted_counts[key] += coupled[key]
            for key in accepted_energy:
                accepted_energy[key] += energy[key]
        else:
            if status == "initial_rejected":
                require(last is not None and last["status"] == "initial_rejected" and
                    coupled["directions_started"] == coupled["directions_completed"] + 1 and
                    coupled["true_checks_completed"] == coupled["directions_started"] and
                    last["initial_forcing"]["decision"] == "rejected" and
                    (last["initial_measure"] is None or last["initial_measure"] >= 1) and
                    (last["initial_forcing"]["measure"] is None or last["initial_forcing"]["measure"] >= 1),
                    "初P拒绝没有完整原门槛证明")
            if status == "bounded_rejected":
                steps, directions = coupled["krylov_completed"], coupled["directions_started"]
                require(last is not None and last["status"] == "bounded_rejected" and
                    directions == coupled["directions_completed"] + 1 and steps >= 1 and
                    coupled["krylov_started"] == coupled["cycles_started"] == steps and
                    rejected_initials == steps and
                    last["initial_forcing"]["decision"] == last["candidate_forcing"]["decision"] == "rejected" and
                    last["candidate_forcing"]["cycle"] == last["candidate_forcing"]["krylov_step"] == 1 and
                    all(last[key] is None or last[key] >= 1 for key in ("initial_measure", "final_measure")) and
                    all(proof["measure"] is None or proof["measure"] >= 1 for proof in (
                        last["initial_forcing"], last["candidate_forcing"])) and
                    all(coupled[key] == directions + steps for key in ("preconditioner_started", "preconditioner_completed",
                        "true_checks_started", "true_checks_completed")) and
                    all(coupled[key] == directions + 2 * steps for key in ("products_started", "products_completed")) and
                    delta["constant_solves"] == 1 and delta["direction_solves"] == directions + steps and
                    delta["scaled_products"] == directions + 2 * steps and
                    delta["scaled_refinement_steps"] == steps and missing_delivery == 0,
                    "单步拒绝没有完整原五块真验和已付费用证明")
            if status in {"initial_rejected", "bounded_rejected", "numerical_error"}:
                rejected += 1
            for key in discarded:
                discarded[key] += delta[key]
        previous = event
    require(accepted == reused, "复用采用数与事件不符")
    if events:
        # 当前生产控制流在返回拒绝后、任何下一次回调之前先记录 fresh_started。
        require(complete >= 1 and fresh == rejected + 1, "fresh回退次数与被拒探测不符")
    else:
        require(fresh <= 1 and reused == 0, "没有探测却出现多个外层fresh")
    main, main_energy = row["coupled_marker"], row["energy_marker"]
    require(all(main[key] >= count for key, count in accepted_counts.items()) and
        all(main_energy[key] >= count for key, count in accepted_energy.items()), "接受探测未完整合入主账本")
    main_fresh_constants = main["constant_calls_started"] - accepted_counts["constant_calls_started"]
    require(0 <= complete - main_fresh_constants <= 1, "主常量重复计入或遗漏fresh工作")
    for event in events:
        if event["status"] == "accepted" and isinstance(main["last"], dict) and main["last"]["attempt_iteration"] == event["attempt_iteration"]:
            require(typed_equal(main["last"], event["coupled"]["last"]), "接受探测的末次证明被改写")
    selected = copy.deepcopy(report)
    for key, field in REUSE_PUBLIC.items():
        require(integer(report.get(field)) and discarded[key] <= report[field], "公共总费用小于已丢弃的实际费用")
        selected[field] -= discarded[key]
    require(report["ipm_factorizations"] >= complete, "fresh方法成功数超过实际分解数")
    return selected, dict(passed=True, discarded_public_counts=discarded, accepted_probes=accepted,
        discarded_probes=len(events) - accepted, total_attempts=fresh + len(events), outer_limit=outer_limit,
        partition="仅校验副本按已验证弃用探测分区；原报告、评分和实际墙钟保持全量")


def crossover_valid(value):
    if not isinstance(value, dict) or value.get("schema") != 3 or value.get("enabled") is not True:
        return False
    required = "attempted started completed local_accepted adopted oracle_decision norms_ready norm_items bounds_inf cost_inf gate_seconds norm_seconds gate_checks last_gate_iteration last_gate_measures last".split()
    if not all(key in value for key in required) or not all(type(value[key]) is bool for key in
            ("enabled", "attempted", "local_accepted", "adopted", "norms_ready")):
        return False
    if not all(type(value[key]) is int and value[key] >= 0 for key in ("started", "completed", "norm_items", "gate_checks")):
        return False
    if not 0 <= value["completed"] <= value["started"] <= 1 or value["attempted"] != (value["started"] == 1):
        return False
    if (value["last"] is None) != (value["started"] == 0):
        return False
    def calls(node):
        if isinstance(node, dict):
            if {"started", "completed", "rc", "seconds"} <= node.keys():
                a, b = node["started"], node["completed"]
                if type(a) is not int or type(b) is not int or not 0 <= a - b <= 1:
                    return False
                if not benchmark.finite_real(node["seconds"]) or node["seconds"] < 0:
                    return False
            return all(calls(child) for child in node.values())
        if isinstance(node, list):
            return all(calls(child) for child in node)
        return True
    if not calls(value):
        return False
    event = value.get("last")
    if event is None:
        return not value["local_accepted"] and not value["adopted"]
    if not isinstance(event, dict) or not all(key in event for key in
            "iteration status crossover_status before_quality after_quality crossover_quality basis_export warm stats verification_seconds_total scope_seconds".split()):
        return False
    native = event["stats"]
    if not isinstance(native, dict):
        return False
    for name in ("load", "crossover", "get"):
        a, b = native.get(name + "_started"), native.get(name + "_completed")
        if type(a) is not int or type(b) is not int or not 0 <= b <= a <= 1:
            return False
    if native["get_started"] > native["crossover_completed"] or native["crossover_started"] > native["load_completed"]:
        return False
    if value["local_accepted"] != (event["status"] == "accepted") or value["adopted"] and (
            not value["local_accepted"] or value["oracle_decision"] not in ("accept", "not_active")):
        return False
    warm = event.get("warm") if isinstance(event, dict) else None
    if warm is not None:
        stats = warm["stats"]
        if not 0 <= stats["steps_completed"] <= stats["pivots"] + stats["bound_flips"] <= stats["steps_started"] <= warm["max_steps"]:
            return False
        if not 0 <= stats["dual_steps_completed"] <= stats["dual_pivots"] <= stats["dual_steps_started"] <= stats["steps_started"]:
            return False
        if warm["status"] == "accepted" and stats["dual_entered"]:
            for key in ("dual_entry_checks", "last_checks"):
                check = stats[key]
                if not isinstance(check, dict) or not check["basis_dual_feasible"] or not check["nonbasic_dual_checked"] or not check["nonbasic_dual_feasible"]:
                    return False
            if not stats["last_checks"]["primal_feasible"] or stats["dual_checks_started"] != stats["dual_checks_completed"]:
                return False
    return True


def execute(item, plan, index, validator):
    command = item["command"]
    expected = plan["protected"][command[0]]
    before, before_error = digest(command[0])
    row = dict(schema_version=1, run_index=index, input=item["input"], seed=0,
               requested_backend=item["backend"], requested_variant=item["variant"],
               model_index=item["model_index"], repetition=item["repetition"],
               position_in_pair=item["position_in_pair"], command=command,
               returncode=None, report=None, status=None, error=None, stdout="", stderr="",
               model_sha256=digest(item["input"])[0], solver_sha256_before=before,
               classification="process_error")
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("SOLVER_V")
                   and k.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    environment.update(dict.fromkeys(threads, "1"))
    begin = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, timeout=plan["wall_timeout"], env=environment)
        end = time.perf_counter()
        row.update(returncode=process.returncode, stdout=benchmark.output_text(process.stdout),
                   stderr=benchmark.output_text(process.stderr))
        try:
            row["report"] = benchmark.strict_json(process.stdout.decode("utf-8", errors="strict"))
            row["status"] = row["report"].get("status")
            row["classification"] = benchmark.classify(process.returncode, row["report"])
        except (ValueError, TypeError, AttributeError, UnicodeError) as error:
            row.update(classification="invalid_report", error=str(error))
    except subprocess.TimeoutExpired as error:
        end = time.perf_counter()
        row.update(classification="wall_timeout", stdout=benchmark.output_text(error.stdout),
                   stderr=benchmark.output_text(error.stderr), error="超过冻结墙钟预算，已终止")
    except OSError as error:
        end = time.perf_counter()
        row["error"] = str(error)
    after, after_error = digest(command[0])
    model_after, model_error = digest(item["input"])
    row.update(elapsed_wall_seconds=end - begin, solver_sha256_after=after,
               model_sha256_after=model_after, hash_error=before_error or after_error or model_error)
    parsed = validator.inspect_markers(row["stderr"], "diagnostic_on" if item["variant"] == "diagnostic_on" else "frozen_v66")
    for field, value in zip(("presolve_marker", "presolve_marker_error", "coupled_marker", "coupled_marker_error",
                             "energy_marker", "energy_marker_error"), parsed):
        row[field] = value
    report = row["report"]
    row["configuration_matches"] = (isinstance(report, dict)
        and report.get("backend") == item["backend"]
        and report.get("backend_version") == item["backend_version"]
        and isinstance(report.get("input"), str) and Path(report["input"]).resolve() == Path(item["input"]).resolve()
        and typed_equal(report.get("options"), item["expected_options"]))
    row["numerical_seconds"] = compare_lp.numerical_seconds(row)
    row["crossover_marker"], row["crossover_marker_error"] = None, None
    crossover_lines = [line for line in row["stderr"].splitlines() if line.startswith("V117_CROSSOVER")]
    if item["variant"] == "diagnostic_on":
        try:
            if len(crossover_lines) != 1 or not crossover_lines[0].startswith("V117_CROSSOVER "):
                raise ValueError("交叉标记必须恰好出现一次")
            row["crossover_marker"] = benchmark.strict_json(crossover_lines[0][len("V117_CROSSOVER "):])
            if not crossover_valid(row["crossover_marker"]):
                raise ValueError("交叉或对偶账本无效")
        except (KeyError, TypeError, ValueError) as error:
            row["crossover_marker_error"] = str(error)
    elif crossover_lines:
        row["crossover_marker_error"] = "关闭路径出现交叉标记"
    row["factor_reuse_marker"], row["factor_reuse_marker_error"] = None, None
    row["reuse_validation"] = None
    selected_row = row
    try:
        if unknown_marker(row["stderr"]):
            raise ValueError("出现未知或非法结构化标记")
        expected_ipm = item["expected_options"].get("ipm")
        reuse_enabled = isinstance(expected_ipm, dict) and expected_ipm.get("factor_reuse") is True
        row["factor_reuse_marker"] = inspect_reuse_marker(row["stderr"], reuse_enabled)
        if reuse_enabled:
            selected_report, row["reuse_validation"] = reuse_selected_report(row["factor_reuse_marker"], row, validator)
            selected_row = dict(row, report=selected_report)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        row["factor_reuse_marker_error"] = str(error)
    row["valid_process"] = (row["hash_error"] is None and row["crossover_marker_error"] is None
                            and row["factor_reuse_marker_error"] is None and validator.valid_process(selected_row))
    frozen = dict(sha256=item["sha256"], model_index=item["model_index"], input=item["input"])
    failure = compare_lp.quality_failure(row, frozen, row["model_sha256_after"], expected,
        row["solver_sha256_after"] if before == row["solver_sha256_after"] else None)
    numerical = compare_lp.numerical_seconds(row)
    numerical_exceeded = numerical > plan["time_limit"] if benchmark.finite_real(numerical) else None
    wall_exceeded = row["elapsed_wall_seconds"] > plan["wall_timeout"]
    if not row["valid_process"]:
        failure = "invalid_process"
    elif failure is None and wall_exceeded:
        failure = "wall_budget_exceeded"
    elif failure is None and numerical_exceeded:
        failure = "numerical_budget_exceeded"
    row.update(numerical_seconds=numerical, quality_failure=failure, quality_pass=failure is None,
               numerical_budget_exceeded=numerical_exceeded, wall_budget_exceeded=wall_exceeded,
               quality_penalized_wall_seconds=min(row["elapsed_wall_seconds"], plan["wall_timeout"])
                   if failure is None else plan["wall_timeout"],
               quality_penalized_numerical_seconds=min(numerical, plan["time_limit"])
                   if failure is None else plan["time_limit"])
    return row


def run(plan_path, output, stage):
    plan = read(plan_path)
    check_protected(plan)
    for item in plan["stages"][stage]["runs"]:
        check_command(item)
    plan_sha = benchmark.sha256(plan_path)
    validator = load_validator(plan)
    def prior(name):
        value = read(output / name / "summary.json")
        if value["plan_sha256"] != plan_sha or value["raw_sha256"] != benchmark.sha256(output / name / "runs.jsonl"):
            raise ValueError("前置阶段证据身份不符")
        if value.get("valid_processes") is not True:
            raise ValueError("前置阶段进程或配置检查未通过")
        return value
    if not stage.startswith("compatibility-"):
        if not all(prior(name).get("compatibility_passed") for name in ("compatibility-development", "compatibility-probe")):
            raise ValueError("关闭路径兼容阶段未通过")
    if stage in ("netlib", "highs"):
        decision = gate(prior("development"), prior("probe"))
        if not decision["eligible_for_netlib"]:
            raise ValueError("未通过原扩集门槛")
    settings = plan["stages"][stage]
    destination = output / stage
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "manifest.json", dict(plan_sha256=plan_sha, stage=stage, **settings))
    rows = []
    with (destination / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for index, item in enumerate(settings["runs"]):
            row = execute(item, plan, index, validator)
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(dict(stage=stage, run=index, variant=item["variant"], status=row["status"],
                                  quality=row["quality_pass"])), flush=True)
    check_protected(plan)
    if benchmark.sha256(plan_path) != plan_sha:
        raise ValueError("执行期间计划发生改变")
    summary = compare_lp.summarize(rows, settings["models"], settings["repeats"],
                                  plan["wall_timeout"], plan["time_limit"], tuple(settings["variants"]))
    summary["valid_processes"] = all(row["valid_process"] for row in rows)
    summary.update(plan_sha256=plan_sha, raw_sha256=benchmark.sha256(destination / "runs.jsonl"))
    if stage.startswith("compatibility-"):
        pairs = [rows[index:index + 2] for index in range(0, len(rows), 2)]
        summary["compatibility_pairs"] = [dict(input=a["input"],
            passed=a["returncode"] == b["returncode"] and typed_equal(stripped(a["report"]), stripped(b["report"])))
            for a, b in pairs]
        summary["compatibility_passed"] = all(pair["passed"] for pair in summary["compatibility_pairs"])
    save(destination / "summary.json", summary)
    if not summary["valid_processes"] or summary.get("compatibility_passed") is False:
        raise ValueError("进程、配置或关闭路径兼容检查失败；原始记录已全部保留")


def self_check():
    # 使用冻结协议形状检查原门槛的关键边界，不启动求解器。
    def summary(c, r, wall=1, numerical=1):
        return dict(reference_variant="r", candidate_variant="c",
                    models=[dict(input="a", quality_pass_counts=dict(c=c, r=r))],
                    variants=dict(r=dict(penalized_wall_sgm10_seconds=1, penalized_numerical_sgm10_seconds=1),
                                  c=dict(penalized_wall_sgm10_seconds=wall, penalized_numerical_sgm10_seconds=numerical)))
    assert gate(summary(3, 3, 0.9, 0.9), summary(1, 1))["eligible_for_netlib"]
    assert gate(summary(3, 3, 1.1, 1.1), summary(1, 0))["eligible_for_netlib"]
    assert not gate(summary(2, 3, 0.9, 0.9), summary(1, 0))["eligible_for_netlib"]
    assert not gate(summary(3, 3, 0.9, 0.9), summary(0, 1))["eligible_for_netlib"]
    assert not gate(summary(3, 3), summary(1, 1))["eligible_for_netlib"]
    assert benchmark.classify(0, dict(schema_version=1, status="optimal")) == "invalid_report"
    assert benchmark.classify(4, dict(schema_version=1, status="numerical_error")) == "incomplete"
    assert benchmark.classify(0, dict(schema_version=1, status="numerical_error")) == "exit_status_mismatch"
    assert math.isclose(compare_lp.shifted_geometric_mean([35]), 35, rel_tol=1e-14)
    assert typed_equal({"x": 1.0}, {"x": 1}) and not typed_equal({"x": True}, {"x": 1})
    assert not crossover_valid(dict(schema=3, enabled=True))
    return dict(passed=True, cases=11, public_lp_runs=0)


def reuse_self_check(validator):
    """只构造合成账本，覆盖分区、上界、门槛和真实前缀；不调用求解器。"""
    def coupled(iteration=2, directions=2):
        last = dict(attempt_iteration=iteration, role="corrector" if directions == 2 else "affine",
            status="accepted", dimension=2, initial_measure=0.0, final_measure=0.0,
            block_measures=[0.0] * 5, tolerances=[1e-12] * 5,
            initial_forcing=dict(rhs_norms=[0.0] * 3, defect_norms=[0.0] * 5,
                measure=None, decision="strict_initial"), candidate_forcing=validator.empty_candidate_forcing())
        return dict(validator.empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
            directions_started=directions, directions_completed=directions,
            preconditioner_started=directions, preconditioner_completed=directions,
            products_started=directions, products_completed=directions,
            true_checks_started=directions, true_checks_completed=directions, last=last)
    def energy(iteration=2):
        return dict(validator.empty_energy(True), started=1, completed=1, budget_checks=2,
            last=dict(attempt_iteration=iteration, status="completed", columns=0, rows=0,
                primal_regularization=1e-12, primal_terms=0, row_terms=0, primal_energy=0.0,
                row_energy=0.0, denominator=1.0))
    def fixture(accepted=True):
        event = dict(attempt_iteration=2, factor_epoch=1, status="accepted", coupled=coupled(),
            energy=energy(), public_delta=dict.fromkeys(REUSE_PUBLIC, 0))
        event["public_delta"].update(constant_solves=1, direction_solves=2, scaled_products=2)
        main = coupled()
        for key in validator.COUPLED_COUNTS:
            main[key] *= 2
        main_energy = energy()
        for key in validator.ENERGY_COUNTS:
            main_energy[key] *= 2
        value = dict(schema=1, enabled=True, fresh_started=1, fresh_completed=1, reused_steps=1, events=[event])
        report = dict(options=dict(ipm=dict(max_iterations=200)), ipm_iterations=2, status="optimal")
        for key, field in REUSE_PUBLIC.items():
            report[field] = 0
        report.update(ipm_homogeneous_constant_solves=2, ipm_homogeneous_direction_solves=4,
            ipm_homogeneous_scaled_products=4, ipm_factorizations=2)
        if not accepted:
            value.update(fresh_started=2, fresh_completed=2, reused_steps=0)
            event["status"] = "initial_rejected"
            event["coupled"] = coupled(directions=1)
            event["coupled"]["directions_completed"] = 0
            event["coupled"]["initial_forcing_checks"] = 1
            event["coupled"]["last"].update(status="initial_rejected", initial_measure=2.0,
                final_measure=2.0, block_measures=[2.0] * 5)
            event["coupled"]["last"]["initial_forcing"].update(
                defect_norms=[2e-12] * 5, measure=2.0, decision="rejected")
            event["public_delta"].update(direction_solves=1, scaled_products=1)
            report.update(ipm_homogeneous_constant_solves=3, ipm_homogeneous_direction_solves=5,
                ipm_homogeneous_scaled_products=5, ipm_factorizations=3)
        return value, dict(report=report, coupled_marker=main, energy_marker=main_energy)
    accepted, row = fixture()
    original = copy.deepcopy(row)
    selected, audit = reuse_selected_report(accepted, row, validator)
    assert typed_equal(row, original) and typed_equal(selected, row["report"]) and audit["discarded_probes"] == 0
    rejected, rejected_row = fixture(False)
    selected, audit = reuse_selected_report(rejected, rejected_row, validator)
    assert selected["ipm_homogeneous_direction_solves"] == 4 and audit["discarded_public_counts"]["direction_solves"] == 1
    assert rejected_row["report"]["ipm_homogeneous_direction_solves"] == 5
    forced = copy.deepcopy(accepted)
    forced_row = copy.deepcopy(row)
    forced_last = forced["events"][0]["coupled"]["last"]
    forced_last.update(status="forced_initial", initial_measure=2.0, final_measure=2.0,
        block_measures=[2.0, 0.0, 0.0, 0.0, 0.0], dimension=3,
        tolerances=[1.01e-10, 1e-12, 1e-12, 1e-12, 1e-12])
    forced_last["initial_forcing"].update(rhs_norms=[1.0, 0.0, 0.0],
        defect_norms=[2.02e-10, 0.0, 0.0, 0.0, 0.0], measure=2.02e-9, decision="returned")
    forced["events"][0]["coupled"].update(initial_forcing_checks=1, initial_forcing_returns=1)
    forced["events"][0]["public_delta"]["inexact_returns"] = 1
    forced_row["coupled_marker"].update(initial_forcing_checks=1, initial_forcing_returns=1,
        last=copy.deepcopy(forced_last))
    forced_row["report"]["ipm_homogeneous_scaled_inexact_returns"] = 1
    forced["events"][0]["energy"].update(primal_terms=1, budget_checks=3)
    forced["events"][0]["energy"]["last"].update(columns=1, primal_terms=1)
    forced_row["energy_marker"].update(primal_terms=2, budget_checks=6)
    forced_row["energy_marker"]["last"].update(columns=1, primal_terms=1)
    reuse_selected_report(forced, forced_row, validator)
    numerical, numerical_row = fixture(False)
    numerical_event = numerical["events"][0]
    numerical_event.update(status="numerical_error", coupled=validator.empty_coupled(True),
        energy=validator.empty_energy(True))
    numerical_event["coupled"]["constant_calls_started"] = 1
    numerical_event["public_delta"].update(direction_solves=0, scaled_products=0)
    numerical_row["report"].update(ipm_homogeneous_direction_solves=4, ipm_homogeneous_scaled_products=4)
    reuse_selected_report(numerical, numerical_row, validator)
    budget_value, budget_row = fixture(False)
    budget_value.update(fresh_started=1, fresh_completed=1)
    budget_event = budget_value["events"][0]
    budget_event["status"] = "budget"
    budget_event["coupled"].update(preconditioner_started=0, preconditioner_completed=0,
        products_started=0, products_completed=0, true_checks_started=0, true_checks_completed=0,
        initial_forcing_checks=0)
    budget_event["coupled"]["last"].update(status="budget", initial_measure=None,
        final_measure=None, block_measures=[None] * 5)
    budget_event["coupled"]["last"]["initial_forcing"].update(
        defect_norms=None, measure=None, decision="not_checked")
    budget_event["public_delta"].update(direction_solves=0, scaled_products=0)
    budget_row["coupled_marker"] = coupled(iteration=1)
    budget_row["energy_marker"] = energy(iteration=1)
    budget_row["report"].update(ipm_iterations=1, ipm_homogeneous_constant_solves=2,
        ipm_homogeneous_direction_solves=2, ipm_homogeneous_scaled_products=2, ipm_factorizations=2,
        status="time_limit")
    reuse_selected_report(budget_value, budget_row, validator)
    sequence, sequence_row = fixture()
    second_event = copy.deepcopy(sequence["events"][0])
    second_event["attempt_iteration"] = 3
    second_event["coupled"]["last"]["attempt_iteration"] = 3
    second_event["energy"]["last"]["attempt_iteration"] = 3
    sequence["events"].append(second_event)
    sequence["reused_steps"] = 2
    sequence_row["report"].update(ipm_iterations=3, ipm_homogeneous_constant_solves=3,
        ipm_homogeneous_direction_solves=6, ipm_homogeneous_scaled_products=6)
    for key in validator.COUPLED_COUNTS:
        sequence_row["coupled_marker"][key] = 3 * coupled()[key]
    sequence_row["coupled_marker"]["last"] = copy.deepcopy(second_event["coupled"]["last"])
    for key in validator.ENERGY_COUNTS:
        sequence_row["energy_marker"][key] = 3 * energy()[key]
    sequence_row["energy_marker"]["last"] = copy.deepcopy(second_event["energy"]["last"])
    reuse_selected_report(sequence, sequence_row, validator)
    cases = 6
    mutations = []
    def changed(path, wrong, source=accepted, source_row=row):
        value, modified_row = copy.deepcopy(source), copy.deepcopy(source_row)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = wrong
        mutations.append((value, modified_row))
    changed(("schema",), True)
    changed(("enabled",), 1)
    changed(("fresh_started",), True)
    changed(("fresh_completed",), 0)
    changed(("reused_steps",), 2)
    changed(("events", 0, "factor_epoch"), 2)
    changed(("events", 0, "attempt_iteration"), 201)
    changed(("events", 0, "status"), "unknown")
    changed(("events", 0, "public_delta", "factorizations"), 1)
    changed(("events", 0, "public_delta", "refinement_steps"), 1)
    changed(("events", 0, "public_delta", "direction_solves"), 1)
    changed(("events", 0, "coupled", "directions_started"), 3)
    changed(("events", 0, "coupled", "krylov_started"), 1)
    changed(("events", 0, "coupled", "candidate_forcing_checks"), 1)
    changed(("events", 0, "public_delta", "inexact_returns"), 1)
    changed(("events", 0, "public_delta", "constant_solves"), True)
    changed(("events", 0, "coupled", "last", "block_measures"), [0.0, 0.0, 0.0, 2.0, 0.0])
    changed(("events", 0, "coupled", "last", "initial_measure"), 0.5, rejected, rejected_row)
    changed(("events", 0, "coupled", "last", "initial_forcing", "measure"), 0.5, rejected, rejected_row)
    changed(("events", 0, "energy", "last", "attempt_iteration"), 1)
    changed(("events", 1, "factor_epoch"), 2, sequence, sequence_row)
    changed(("events", 1, "attempt_iteration"), 4, sequence, sequence_row)
    bad_row = copy.deepcopy(row)
    bad_row["coupled_marker"]["constant_calls_started"] += 1
    mutations.append((copy.deepcopy(accepted), bad_row))
    bad_row = copy.deepcopy(rejected_row)
    bad_row["report"]["ipm_homogeneous_direction_solves"] = 0
    mutations.append((copy.deepcopy(rejected), bad_row))
    for event_status, report_status in (("budget", "optimal"), ("budget", "error"),
                                        ("exception", "optimal"), ("running", "optimal")):
        value, modified_row = copy.deepcopy(budget_value), copy.deepcopy(budget_row)
        value["events"][0]["status"] = event_status
        modified_row["report"]["status"] = report_status
        mutations.append((value, modified_row))
    for value, modified_row in mutations:
        try:
            reuse_selected_report(value, modified_row, validator)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass
        else:
            raise AssertionError("畸形复用账本被接受")
        cases += 1
    for stderr, enabled in ((REUSE_MARKER + "{}\n" + REUSE_MARKER + "{}", True),
                            (REUSE_MARKER + "{}", False), ("V122_FACTOR_REUSE{}", True)):
        try:
            inspect_reuse_marker(stderr, enabled)
        except ValueError:
            pass
        else:
            raise AssertionError("非法复用标记被接受")
        cases += 1
    assert unknown_marker("V999_UNKNOWN {}") and not unknown_marker(REUSE_MARKER + "{}")
    cases += 1
    try:
        check_command(dict(command=["solver", "solve", "x", "--ipm-profile", "on"],
            expected_options=dict(ipm=dict(profile=True)), variant="diagnostic_on"))
    except ValueError:
        pass
    else:
        raise AssertionError("固定协议接受了profile开启")
    cases += 1
    return cases


def reuse_bounded_self_check(validator):
    """保留零步历史自检，另外覆盖单步账本及预算回调的实际前缀。"""
    def direction(directions=2, steps=1, status="accepted"):
        last = dict(attempt_iteration=2, role="corrector" if directions == 2 else "affine",
            status=status, dimension=2, initial_measure=2.0, final_measure=0.5,
            block_measures=[0.5, 0.0, 0.0, 0.0, 0.0], tolerances=[1e-12] * 5,
            initial_forcing=dict(rhs_norms=[0.0] * 3, defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0],
                measure=2.0, decision="rejected"), candidate_forcing=validator.empty_candidate_forcing())
        return dict(validator.empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
            directions_started=directions, directions_completed=directions,
            preconditioner_started=directions + steps, preconditioner_completed=directions + steps,
            products_started=directions + 2 * steps, products_completed=directions + 2 * steps,
            true_checks_started=directions + steps, true_checks_completed=directions + steps,
            krylov_started=steps, krylov_completed=steps, cycles_started=steps,
            initial_forcing_checks=steps, last=last)

    def energy():
        return dict(validator.empty_energy(True), started=1, completed=1, budget_checks=2,
            last=dict(attempt_iteration=2, status="completed", columns=0, rows=0,
                primal_regularization=1e-12, primal_terms=0, row_terms=0, primal_energy=0.0,
                row_energy=0.0, denominator=1.0))

    def fixture(mode="strict", steps=1):
        probe = direction(steps=steps)
        status, fresh, reused = "accepted", 1, 1
        if mode == "forced":
            probe.update(candidate_forcing_checks=1, candidate_forcing_returns=1)
            probe["last"].update(status="forced_candidate", dimension=3, initial_measure=3e9,
                final_measure=2.0, block_measures=[2.0, 0.0, 0.0, 0.0, 0.0],
                tolerances=[1e-10, 1e-12, 1e-12, 1e-12, 1e-12])
            probe["last"]["initial_forcing"].update(rhs_norms=[1.0, 0.0, 0.0],
                defect_norms=[0.3, 0.0, 0.0, 0.0, 0.0], measure=3.0)
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-10, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=3e9, measure=2e-9, decision="returned")
        elif mode == "rejected":
            status, fresh, reused = "bounded_rejected", 2, 0
            probe = direction(directions=1, status=status)
            probe.update(directions_completed=0, candidate_forcing_checks=1)
            probe["last"].update(final_measure=2.0, block_measures=[2.0, 0.0, 0.0, 0.0, 0.0])
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=2.0, measure=2.0, decision="rejected")
        elif mode == "initial":
            probe = direction(steps=0)
            probe["last"].update(initial_measure=0.5)
            probe["last"]["initial_forcing"].update(defect_norms=[0.5e-12, 0.0, 0.0, 0.0, 0.0],
                measure=None, decision="strict_initial")
        main = direction(steps=0)
        main["last"].update(attempt_iteration=1, initial_measure=0.5)
        main["last"]["initial_forcing"].update(defect_norms=[0.5e-12, 0.0, 0.0, 0.0, 0.0],
            measure=None, decision="strict_initial")
        main_energy = energy()
        main_energy["last"]["attempt_iteration"] = 1
        # 接受时合入已完成探测；拒绝时主账本只含两个fresh整组。
        extra = probe if reused else copy.deepcopy(main)
        for key in validator.COUPLED_COUNTS:
            main[key] += extra[key]
        for key in validator.ENERGY_COUNTS:
            main_energy[key] *= 2
        if reused:
            main["last"] = copy.deepcopy(probe["last"])
            main_energy["last"]["attempt_iteration"] = 2
        delta = dict.fromkeys(REUSE_PUBLIC, 0)
        delta.update(constant_solves=1, direction_solves=probe["preconditioner_completed"],
            scaled_products=probe["products_completed"], inexact_returns=probe["candidate_forcing_returns"],
            refinement_steps=probe["krylov_completed"], scaled_refinement_steps=probe["krylov_completed"])
        event = dict(attempt_iteration=2, factor_epoch=1, status=status, coupled=probe,
            energy=energy(), public_delta=delta)
        value = dict(schema=2, enabled=True, max_krylov_steps=1, fresh_started=fresh,
            fresh_completed=fresh, reused_steps=reused, events=[event])
        report = dict(options=dict(ipm=dict(max_iterations=200)), ipm_iterations=2, status="optimal")
        public_main = dict(constant_solves=2, direction_solves=4, scaled_products=4,
            inexact_returns=0, refinement_steps=0, scaled_refinement_steps=0, factorizations=2 + (not reused))
        for key, field in REUSE_PUBLIC.items():
            report[field] = public_main[key] + (delta[key] - (dict(constant_solves=1,
                direction_solves=2, scaled_products=2).get(key, 0)) if reused else delta[key])
        return value, dict(report=report, coupled_marker=main, energy_marker=main_energy)

    strict, strict_row = fixture()
    original = copy.deepcopy(strict_row)
    selected, audit = reuse_selected_report(strict, strict_row, validator)
    assert typed_equal(original, strict_row) and typed_equal(selected, strict_row["report"])
    assert audit["accepted_probes"] == 1 and audit["discarded_probes"] == 0
    initial, initial_row = fixture("initial")
    reuse_selected_report(initial, initial_row, validator)
    both, both_row = fixture(steps=2)
    reuse_selected_report(both, both_row, validator)
    forced, forced_row = fixture("forced")
    reuse_selected_report(forced, forced_row, validator)
    rejected, rejected_row = fixture("rejected")
    selected, audit = reuse_selected_report(rejected, rejected_row, validator)
    assert selected["ipm_homogeneous_direction_solves"] == 4
    assert selected["ipm_homogeneous_scaled_products"] == 4
    assert selected["ipm_refinement_steps"] == selected["ipm_homogeneous_scaled_refinement_steps"] == 0
    assert rejected_row["report"]["ipm_homogeneous_direction_solves"] == 6
    assert audit["discarded_public_counts"]["scaled_refinement_steps"] == 1
    narrowed, narrowed_row = copy.deepcopy(rejected), copy.deepcopy(rejected_row)
    narrowed_last = narrowed["events"][0]["coupled"]["last"]
    narrowed_last.update(initial_measure=1.0, final_measure=1.0, block_measures=[1.0] * 5)
    narrowed_last["initial_forcing"]["measure"] = 1.0
    narrowed_last["candidate_forcing"].update(strict_measure=1.0, best_measure_before=1.0, measure=1.0)
    reuse_selected_report(narrowed, narrowed_row, validator)
    unrepresented, unrepresented_row = copy.deepcopy(rejected), copy.deepcopy(rejected_row)
    unrepresented_last = unrepresented["events"][0]["coupled"]["last"]
    unrepresented_last.update(initial_measure=None, final_measure=None, block_measures=[None] * 5)
    unrepresented_last["initial_forcing"].update(defect_norms=[None] * 5, measure=None)
    unrepresented_last["candidate_forcing"].update(defect_norms=[None] * 5,
        strict_measure=None, best_measure_before=None, measure=None)
    reuse_selected_report(unrepresented, unrepresented_row, validator)
    cases = 7

    def budget_fixture(stage):
        value, row = fixture("rejected")
        value.update(fresh_started=1, fresh_completed=1)
        event = value["events"][0]
        event["status"] = "budget"
        probe, delta = event["coupled"], event["public_delta"]
        probe.update(preconditioner_started=1, preconditioner_completed=1, products_started=1,
            products_completed=1, true_checks_started=1, true_checks_completed=1,
            krylov_started=1, krylov_completed=0, cycles_started=1, candidate_forcing_checks=0)
        probe["last"].update(status="budget", candidate_forcing=validator.empty_candidate_forcing())
        delta.update(direction_solves=1, scaled_products=1, refinement_steps=0, scaled_refinement_steps=0)
        if stage == "callback":
            probe["preconditioner_started"] = 2
            delta.update(direction_solves=2, refinement_steps=1, scaled_refinement_steps=1)
        elif stage == "column":
            probe.update(preconditioner_started=2, preconditioner_completed=2,
                products_started=2, products_completed=2)
            delta.update(direction_solves=2, scaled_products=2, refinement_steps=1, scaled_refinement_steps=1)
        elif stage == "eligible":
            force_probe = fixture("forced")[0]["events"][0]["coupled"]
            probe.update(preconditioner_started=2, preconditioner_completed=2, products_started=3,
                products_completed=3, true_checks_started=2, true_checks_completed=2,
                krylov_completed=1, candidate_forcing_checks=1, last=copy.deepcopy(force_probe["last"]))
            probe["last"].update(role="affine", status="budget")
            probe["last"]["candidate_forcing"]["decision"] = "eligible"
            delta.update(direction_solves=2, scaled_products=3, refinement_steps=1, scaled_refinement_steps=1)
        main = direction(steps=0)
        main["last"].update(attempt_iteration=1, initial_measure=0.5)
        main["last"]["initial_forcing"].update(defect_norms=[0.5e-12, 0.0, 0.0, 0.0, 0.0],
            measure=None, decision="strict_initial")
        row["coupled_marker"] = main
        row["energy_marker"] = energy()
        row["energy_marker"]["last"]["attempt_iteration"] = 1
        row["report"].update(ipm_iterations=1, status="time_limit")
        base = dict(constant_solves=1, direction_solves=2, scaled_products=2,
            inexact_returns=0, refinement_steps=0, scaled_refinement_steps=0, factorizations=1)
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] = base[key] + delta[key]
        return value, row

    prefixes = []
    for stage in ("started", "callback", "column", "eligible"):
        value, row = budget_fixture(stage)
        original = copy.deepcopy(row)
        reuse_selected_report(value, row, validator)
        assert typed_equal(original, row)
        prefixes.append((value, row))
        cases += 1
    delivery, delivery_row = copy.deepcopy(forced), copy.deepcopy(forced_row)
    delivery.update(reused_steps=0)
    delivery["events"][0]["status"] = "budget"
    delivery["events"][0]["public_delta"]["inexact_returns"] = 0
    delivery_row["coupled_marker"] = copy.deepcopy(prefixes[0][1]["coupled_marker"])
    delivery_row["energy_marker"] = copy.deepcopy(prefixes[0][1]["energy_marker"])
    delivery_row["report"].update(ipm_iterations=1, status="time_limit",
        ipm_homogeneous_scaled_inexact_returns=0)
    reuse_selected_report(delivery, delivery_row, validator)
    cases += 1

    mutations = []
    def changed(path, wrong, source=strict, source_row=strict_row):
        value, row = copy.deepcopy(source), copy.deepcopy(source_row)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = wrong
        mutations.append((value, row))
    changed(("schema",), True)
    changed(("max_krylov_steps",), True)
    changed(("max_krylov_steps",), 0)
    changed(("max_krylov_steps",), 2)
    missing, missing_row = copy.deepcopy(strict), copy.deepcopy(strict_row)
    del missing["max_krylov_steps"]
    mutations.append((missing, missing_row))
    changed(("events", 0, "status"), "initial_rejected")
    changed(("events", 0, "coupled", "last", "status"), "initial_rejected")
    changed(("events", 0, "public_delta", "factorizations"), 1)
    changed(("events", 0, "public_delta", "refinement_steps"), 0)
    changed(("events", 0, "public_delta", "scaled_refinement_steps"), 0)
    changed(("events", 0, "public_delta", "direction_solves"), 2)
    changed(("events", 0, "public_delta", "scaled_products"), 3)
    changed(("events", 0, "public_delta", "inexact_returns"), 1)
    changed(("events", 0, "coupled", "cycles_started"), 2)
    changed(("events", 0, "coupled", "krylov_started"), 3)
    changed(("events", 0, "coupled", "initial_forcing_checks"), 1, both, both_row)
    changed(("events", 0, "coupled", "initial_forcing_checks"), 2)
    changed(("events", 0, "coupled", "true_checks_started"), 2)
    changed(("events", 0, "coupled", "candidate_forcing_checks"), 2, forced, forced_row)
    changed(("events", 0, "coupled", "candidate_forcing_returns"), 2, forced, forced_row)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "cycle"), 2, forced, forced_row)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "krylov_step"), 2, forced, forced_row)
    changed(("events", 0, "coupled", "last", "block_measures"), [2.0] * 5)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "measure"), 2.0, forced, forced_row)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "decision"), "eligible", forced, forced_row)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "decision"), "not_checked", rejected, rejected_row)
    changed(("events", 0, "coupled", "last", "candidate_forcing", "measure"), 0.5, rejected, rejected_row)
    changed(("events", 0, "coupled", "last", "initial_forcing", "measure"), 0.5, rejected, rejected_row)
    changed(("events", 0, "coupled", "last", "final_measure"), 0.5, rejected, rejected_row)
    changed(("events", 0, "status"), "numerical_error", rejected, rejected_row)
    changed(("events", 0, "coupled", "last", "status"), "exhausted", rejected, rejected_row)
    changed(("events", 0, "coupled", "candidate_forcing_checks"), 0, rejected, rejected_row)
    changed(("events", 0, "coupled", "products_completed"), 2, rejected, rejected_row)
    changed(("events", 0, "coupled", "true_checks_completed"), 1, rejected, rejected_row)
    changed(("events", 0, "public_delta", "scaled_refinement_steps"), 2, prefixes[1][0], prefixes[1][1])
    changed(("events", 0, "status"), "bounded_rejected", prefixes[0][0], prefixes[0][1])
    changed(("events", 0, "public_delta", "inexact_returns"), 1, prefixes[3][0], prefixes[3][1])
    changed(("events", 0, "coupled", "preconditioner_started"), 4, delivery, delivery_row)
    wrong_report, wrong_report_row = copy.deepcopy(prefixes[0][0]), copy.deepcopy(prefixes[0][1])
    wrong_report_row["report"]["status"] = "optimal"
    mutations.append((wrong_report, wrong_report_row))
    # 两个假前缀的公开P/A与精化费用均自洽，但先前已完成方向留下了未完成FG。
    for started in (1, 2):
        value, row = copy.deepcopy(prefixes[2][0]), copy.deepcopy(prefixes[2][1])
        probe = value["events"][0]["coupled"]
        probe.update(directions_started=2, directions_completed=1, initial_forcing_checks=2,
            krylov_started=started, krylov_completed=0, cycles_started=started,
            preconditioner_started=2 + started, preconditioner_completed=3,
            products_started=2 + started, products_completed=3,
            true_checks_started=2, true_checks_completed=2)
        probe["last"]["role"] = "corrector"
        value["events"][0]["public_delta"].update(direction_solves=3, scaled_products=3)
        row["report"].update(ipm_homogeneous_direction_solves=5, ipm_homogeneous_scaled_products=5)
        mutations.append((value, row))
    # 末方向完整拒绝也不能替前一个已完成方向补上缺失的FG步。
    value, row = copy.deepcopy(rejected), copy.deepcopy(rejected_row)
    probe = value["events"][0]["coupled"]
    probe.update(directions_started=2, directions_completed=1, initial_forcing_checks=2,
        preconditioner_started=3, preconditioner_completed=3,
        products_started=4, products_completed=4, true_checks_started=3, true_checks_completed=3)
    probe["last"]["role"] = "corrector"
    value["events"][0]["public_delta"].update(direction_solves=3, scaled_products=4)
    row["report"].update(ipm_homogeneous_direction_solves=7, ipm_homogeneous_scaled_products=8)
    mutations.append((value, row))
    for value, row in mutations:
        try:
            reuse_selected_report(value, row, validator)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass
        else:
            raise AssertionError("畸形单步复用账本被接受")
        cases += 1
    return cases


def reuse_current_diagonal_self_check(validator):
    """独立验证schema3；历史零步、单步自检及冻结数学门槛保持原样。"""
    def energy(iteration=2, n=2, m=3, calls=1):
        checks = 2 + (n + 4095) // 4096 + (m + 4095) // 4096
        return dict(validator.empty_energy(True), started=calls, completed=calls,
            primal_terms=calls * n, row_terms=calls * m, budget_checks=calls * checks,
            last=dict(attempt_iteration=iteration, status="completed", columns=n, rows=m,
                primal_regularization=1e-12, primal_terms=n, row_terms=m,
                primal_energy=0.0, row_energy=0.0, denominator=1.0))

    def coupled(iteration=2, directions=2, steps=0, n=2, m=3):
        last = dict(attempt_iteration=iteration, role="corrector" if directions == 2 else "affine",
            status="accepted", dimension=n + m + 2, initial_measure=0.0, final_measure=0.0,
            block_measures=[0.0] * 5, tolerances=[1e-12] * 5,
            initial_forcing=dict(rhs_norms=[0.0] * 3, defect_norms=[0.0] * 5,
                measure=None, decision="strict_initial"), candidate_forcing=validator.empty_candidate_forcing())
        if steps:
            last.update(initial_measure=2.0, final_measure=0.5, block_measures=[0.5, 0.0, 0.0, 0.0, 0.0])
            last["initial_forcing"].update(defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0],
                measure=2.0, decision="rejected")
        return dict(validator.empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
            directions_started=directions, directions_completed=directions,
            preconditioner_started=directions + steps, preconditioner_completed=directions + steps,
            products_started=directions + 2 * steps, products_completed=directions + 2 * steps,
            true_checks_started=directions + steps, true_checks_completed=directions + steps,
            krylov_started=steps, krylov_completed=steps, cycles_started=steps,
            initial_forcing_checks=steps, last=last)

    def current(n=2, m=3, nonzeros=4, applications=3):
        return dict(columns=n, rows=m, nonzeros=nonzeros,
            current_rows_started=1, current_rows_completed=1, current_rows_terms=m,
            validation_started=1, validation_completed=1, validation_terms=n + 1 + m + nonzeros,
            applications_started=applications, applications_completed=applications,
            transpose_started=applications, transpose_completed=applications,
            direct_started=applications, direct_completed=applications,
            product_started=applications, product_completed=applications,
            transpose_terms=applications * nonzeros, product_terms=applications * nonzeros,
            row_divisions=2 * m * applications, budget_checks=1 + 10 * applications)

    def fixture(mode="initial", n=2, m=3, nonzeros=4):
        rejected, forced = mode == "rejected", mode == "forced"
        steps, directions = 2 if mode == "both" else int(mode != "initial"), 1 if rejected else 2
        probe = coupled(directions=directions, steps=steps, n=n, m=m)
        if forced:
            probe.update(candidate_forcing_checks=1, candidate_forcing_returns=1)
            probe["last"].update(status="forced_candidate", initial_measure=3e9, final_measure=2.0,
                block_measures=[2.0, 0.0, 0.0, 0.0, 0.0], tolerances=[1e-10, 1e-12, 1e-12, 1e-12, 1e-12])
            probe["last"]["initial_forcing"].update(rhs_norms=[1.0, 0.0, 0.0],
                defect_norms=[0.3, 0.0, 0.0, 0.0, 0.0], measure=3.0)
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-10, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=3e9, measure=2e-9, decision="returned")
        if rejected:
            probe.update(directions_completed=0, candidate_forcing_checks=1)
            probe["last"].update(status="bounded_rejected", final_measure=2.0,
                block_measures=[2.0, 0.0, 0.0, 0.0, 0.0])
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=2.0, measure=2.0, decision="rejected")
        delta = dict.fromkeys(REUSE_PUBLIC, 0)
        delta.update(constant_solves=1, direction_solves=directions + steps,
            scaled_products=directions + 2 * steps, refinement_steps=steps,
            scaled_refinement_steps=steps, inexact_returns=int(forced))
        event = dict(attempt_iteration=2, factor_epoch=1, status="bounded_rejected" if rejected else "accepted",
            coupled=probe, energy=energy(n=n, m=m), public_delta=delta,
            current_diagonal=current(n, m, nonzeros, 1 + directions + steps))
        value = dict(schema=3, enabled=True, max_krylov_steps=1, preconditioner="current_diagonal_schur",
            energy_model="current_positive_approximation", fresh_started=2 if rejected else 1,
            fresh_completed=2 if rejected else 1, reused_steps=0 if rejected else 1, events=[event])
        main = coupled(iteration=1, n=n, m=m)
        extra = coupled(n=n, m=m) if rejected else probe
        for key in validator.COUPLED_COUNTS:
            main[key] += extra[key]
        main["last"] = copy.deepcopy(extra["last"])
        report = dict(options=dict(ipm=dict(max_iterations=200)), ipm_iterations=2, status="optimal")
        base = dict(constant_solves=2 if rejected else 1, direction_solves=4 if rejected else 2,
            scaled_products=4 if rejected else 2, refinement_steps=0, scaled_refinement_steps=0,
            inexact_returns=0, factorizations=3 if rejected else 2)
        for key, field in REUSE_PUBLIC.items():
            report[field] = base[key] + delta[key]
        return value, dict(report=report, coupled_marker=main, energy_marker=energy(n=n, m=m, calls=2))

    fixtures = [fixture(mode) for mode in ("initial", "strict", "forced", "rejected", "both")]
    fixtures.append(fixture(n=0, m=0, nonzeros=0))
    for value, row in fixtures:
        original = copy.deepcopy(row)
        selected, audit = reuse_selected_report(value, row, validator)
        assert typed_equal(original, row) and audit["passed"]
        if value["events"][0]["status"] == "bounded_rejected":
            assert selected["ipm_homogeneous_direction_solves"] == 4
            assert selected["ipm_homogeneous_scaled_products"] == 4
            assert selected["ipm_refinement_steps"] == 0
    cases = len(fixtures)

    def budget_fixture(stage):
        value, row = fixture()
        value["reused_steps"] = 0
        event = value["events"][0]
        event["status"] = "budget"
        probe = coupled(directions=2)
        probe.update(directions_completed=1, preconditioner_completed=1,
            products_started=1, products_completed=1, true_checks_started=1, true_checks_completed=1)
        probe["last"].update(status="budget", initial_measure=None, final_measure=None, block_measures=[None] * 5)
        probe["last"]["initial_forcing"].update(defect_norms=None, decision="not_checked")
        event["coupled"] = probe
        work = current(applications=2)
        event["current_diagonal"] = work
        event["public_delta"].update(direction_solves=1, scaled_products=1)
        phases = ("before_application", "before_transpose", "transpose_partial", "before_direct", "direct_before_body",
            "direct_entered", "after_direct", "product_partial", "before_completion", "after_completion")
        phase = phases.index(stage)
        if phase >= 1:
            work.update(applications_started=3, row_divisions=15)
        if phase >= 2:
            work.update(transpose_started=3, transpose_terms=10)
        if phase >= 3:
            work.update(transpose_completed=3, transpose_terms=12)
        if phase >= 4:
            work["direct_started"] = 3
        if phase >= 5:
            event["public_delta"]["direction_solves"] = 2
        if phase >= 6:
            work["direct_completed"] = 3
        if phase >= 7:
            work.update(product_started=3, product_terms=10)
        if phase >= 8:
            work.update(product_completed=3, product_terms=12, row_divisions=18)
        if phase >= 9:
            work["applications_completed"] = 3
            work["budget_checks"] = 31
        row["coupled_marker"] = coupled(iteration=1)
        row["energy_marker"] = energy(iteration=1)
        row["report"].update(ipm_iterations=1, status="time_limit")
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] = dict(constant_solves=1, direction_solves=2, scaled_products=2,
                factorizations=2).get(key, 0) + event["public_delta"][key]
        return value, row

    prefixes = [budget_fixture(stage) for stage in ("before_application", "before_transpose", "transpose_partial",
        "before_direct", "direct_before_body", "direct_entered", "after_direct", "product_partial",
        "before_completion", "after_completion")]
    for value, row in prefixes:
        reuse_selected_report(value, row, validator)
        cases += 1
    constructors = []
    for phase in range(6):
        value, row = budget_fixture("before_application")
        event = value["events"][0]
        event.update(coupled=validator.empty_coupled(True), energy=validator.empty_energy(True),
            public_delta=dict.fromkeys(REUSE_PUBLIC, 0))
        work = dict.fromkeys(CURRENT_DIAGONAL_FIELDS, 0)
        work.update(columns=2, rows=3, nonzeros=4)
        if phase >= 1:
            work.update(current_rows_started=1, current_rows_terms=1)
        if phase >= 2:
            work.update(current_rows_completed=1, current_rows_terms=3)
        if phase >= 3:
            work.update(validation_started=1, validation_terms=4, budget_checks=2)
        if phase >= 4:
            work["validation_terms"] = 10
        if phase >= 5:
            work["validation_completed"] = 1
            work["budget_checks"] = 3
        event["current_diagonal"] = work
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] = dict(constant_solves=1, direction_solves=2, scaled_products=2,
                factorizations=2).get(key, 0)
        reuse_selected_report(value, row, validator)
        constructors.append((value, row))
        cases += 1

    for entered in (False, True):
        value, row = copy.deepcopy(constructors[-1][0]), copy.deepcopy(constructors[-1][1])
        event = value["events"][0]
        event["coupled"]["constant_calls_started"] = 1
        event["public_delta"]["constant_solves"] = int(entered)
        event["current_diagonal"].update(applications_started=1, transpose_started=1, transpose_completed=1,
            direct_started=1, transpose_terms=4, row_divisions=3, budget_checks=4)
        row["report"]["ipm_homogeneous_constant_solves"] += int(entered)
        reuse_selected_report(value, row, validator)
        cases += 1

    accepted, accepted_row = fixtures[0]
    rejected, rejected_row = fixtures[3]
    mutations = []
    def changed(path, wrong, source=accepted, source_row=accepted_row):
        value, row = copy.deepcopy(source), copy.deepcopy(source_row)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = wrong
        mutations.append((value, row))
    changed(("schema",), True)
    changed(("preconditioner",), "old_full_kkt")
    changed(("energy_model",), "exact_schur_inverse")
    changed(("max_krylov_steps",), 2)
    missing, missing_row = copy.deepcopy(accepted), copy.deepcopy(accepted_row)
    del missing["events"][0]["current_diagonal"]
    mutations.append((missing, missing_row))
    extra, extra_row = copy.deepcopy(accepted), copy.deepcopy(accepted_row)
    extra["events"][0]["current_diagonal"]["unknown"] = 0
    mutations.append((extra, extra_row))
    for field in CURRENT_DIAGONAL_FIELDS:
        changed(("events", 0, "current_diagonal", field), True)
    for field, wrong in (("current_rows_started", 2), ("current_rows_completed", 0), ("current_rows_terms", 2),
            ("validation_started", 2), ("validation_completed", 0), ("validation_terms", 9),
            ("applications_started", 6), ("applications_completed", 2), ("transpose_started", 4),
            ("transpose_completed", 2), ("direct_started", 2), ("direct_completed", 2),
            ("product_started", 2), ("product_completed", 2), ("transpose_terms", 11),
            ("product_terms", 11), ("row_divisions", 17), ("budget_checks", -1)):
        changed(("events", 0, "current_diagonal", field), wrong)
    changed(("events", 0, "energy", "last", "columns"), 1)
    changed(("events", 0, "current_diagonal", "budget_checks"), 0)
    changed(("events", 0, "current_diagonal", "budget_checks"), 26)
    changed(("events", 0, "current_diagonal", "budget_checks"), 26, rejected, rejected_row)
    changed(("events", 0, "current_diagonal", "budget_checks"), 19, prefixes[1][0], prefixes[1][1])
    changed(("events", 0, "energy", "last", "rows"), 2)
    changed(("events", 0, "public_delta", "factorizations"), 1)
    changed(("events", 0, "public_delta", "refinement_steps"), 1)
    changed(("events", 0, "coupled", "last", "block_measures"), [2.0] * 5)
    changed(("events", 0, "current_diagonal", "row_divisions"), 11, prefixes[0][0], prefixes[0][1])
    changed(("events", 0, "current_diagonal", "applications_started"), 4, prefixes[1][0], prefixes[1][1])
    changed(("events", 0, "current_diagonal", "transpose_terms"), 13, prefixes[2][0], prefixes[2][1])
    changed(("events", 0, "current_diagonal", "direct_started"), 3, prefixes[2][0], prefixes[2][1])
    changed(("events", 0, "current_diagonal", "direct_completed"), 3, prefixes[4][0], prefixes[4][1])
    changed(("events", 0, "current_diagonal", "product_started"), 3, prefixes[5][0], prefixes[5][1])
    changed(("events", 0, "current_diagonal", "product_terms"), 13, prefixes[7][0], prefixes[7][1])
    changed(("events", 0, "current_diagonal", "applications_completed"), 3, prefixes[7][0], prefixes[7][1])
    changed(("events", 0, "current_diagonal", "row_divisions"), 19, prefixes[8][0], prefixes[8][1])
    changed(("events", 0, "current_diagonal", "validation_started"), 1, constructors[1][0], constructors[1][1])
    changed(("events", 0, "current_diagonal", "current_rows_terms"), 4, constructors[1][0], constructors[1][1])
    changed(("events", 0, "current_diagonal", "validation_terms"), 11, constructors[3][0], constructors[3][1])
    changed(("events", 0, "current_diagonal", "applications_completed"), 2, rejected, rejected_row)
    wrong_status, wrong_status_row = copy.deepcopy(prefixes[4][0]), copy.deepcopy(prefixes[4][1])
    wrong_status_row["report"]["status"] = "optimal"
    mutations.append((wrong_status, wrong_status_row))
    for value, row in mutations:
        try:
            reuse_selected_report(value, row, validator)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass
        else:
            raise AssertionError("畸形当前R结构账本被接受")
        cases += 1
    return cases


def reuse_transport_self_check(validator):
    """schema4独立合成账本；缓存、双侧缩放和标量观察不改变原数学自检。"""
    def coupled(iteration=2, directions=2, steps=0, n=2, m=3):
        last = dict(attempt_iteration=iteration, role="corrector" if directions == 2 else "affine",
            status="accepted", dimension=n + m + 2, initial_measure=0.0, final_measure=0.0,
            block_measures=[0.0] * 5, tolerances=[1e-12] * 5,
            initial_forcing=dict(rhs_norms=[0.0] * 3, defect_norms=[0.0] * 5,
                measure=None, decision="strict_initial"), candidate_forcing=validator.empty_candidate_forcing())
        if steps:
            last.update(initial_measure=2.0, final_measure=0.5, block_measures=[0.5, 0.0, 0.0, 0.0, 0.0])
            last["initial_forcing"].update(defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0],
                measure=2.0, decision="rejected")
        return dict(validator.empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
            directions_started=directions, directions_completed=directions,
            preconditioner_started=directions + steps, preconditioner_completed=directions + steps,
            products_started=directions + 2 * steps, products_completed=directions + 2 * steps,
            true_checks_started=directions + steps, true_checks_completed=directions + steps,
            krylov_started=steps, krylov_completed=steps, cycles_started=steps, initial_forcing_checks=steps, last=last)

    def energy(iteration=2, n=2, m=3, calls=1):
        return dict(validator.empty_energy(True), started=calls, completed=calls,
            primal_terms=calls * n, row_terms=calls * m,
            budget_checks=calls * (2 + (n + 4095) // 4096 + (m + 4095) // 4096),
            last=dict(attempt_iteration=iteration, status="completed", columns=n, rows=m,
                primal_regularization=1e-12, primal_terms=n, row_terms=m,
                primal_energy=0.0, row_energy=0.0, denominator=1.0))

    def current(n=2, m=3, nonzeros=4, applications=3):
        return dict(columns=n, rows=m, nonzeros=nonzeros,
            current_rows_started=1, current_rows_completed=1, current_rows_terms=m,
            validation_started=1, validation_completed=1, validation_terms=n + 1 + m + nonzeros,
            applications_started=applications, applications_completed=applications,
            transpose_started=applications, transpose_completed=applications,
            direct_started=applications, direct_completed=applications,
            product_started=applications, product_completed=applications,
            transpose_terms=applications * nonzeros, product_terms=applications * nonzeros,
            row_divisions=2 * m * applications, budget_checks=100)

    def transport(n=2, m=3, nonzeros=4, applications=3, hit=False, epoch=1):
        value = dict.fromkeys(TRANSPORT_FIELDS, 0)
        value.update(columns=n, rows=m, nonzeros=nonzeros, cache_epoch=epoch,
            prepare_started=1, prepare_completed=1, cache_hits=int(hit),
            old_build_started=int(not hit), old_build_completed=int(not hit),
            old_row_terms=0 if hit else m, scratch_row_terms=2 * m, merge_terms=nonzeros,
            unique_terms=nonzeros, old_terms=0 if hit else nonzeros, current_terms=nonzeros, column_terms=n,
            input_started=applications, input_completed=applications, input_terms=n * applications,
            direct_started=applications, direct_completed=applications,
            output_started=applications, output_completed=applications, output_terms=n * applications,
            budget_checks=10 + 8 * applications)
        for key in TRANSPORT_RANGES:
            value[key] = None
        if n:
            value.update(old_min=1.0, old_max=2.0, current_min=0.5, current_max=4.0, scale_min=0.5, scale_max=2.0)
        return value

    def scalar(n=2, m=3, complete=True):
        return dict(started=int(complete), completed=int(complete), terms=n + m if complete else 0,
            budget_checks=(3 if n + m else 2) if complete else 0,
            natural_denominator=(0.5 if n + m else 1.0) if complete else None,
            positive_denominator=1.0 if complete else None,
            relative_difference=(-0.5 if n + m else 0.0) if complete else None)

    def fixture(mode="initial", n=2, m=3, nonzeros=4):
        rejected, forced = mode == "rejected", mode == "forced"
        steps = 2 if mode == "both" else int(mode != "initial")
        directions = 1 if rejected else 2
        probe = coupled(directions=directions, steps=steps, n=n, m=m)
        if rejected:
            probe.update(directions_completed=0, candidate_forcing_checks=1)
            probe["last"].update(status="bounded_rejected", final_measure=2.0, block_measures=[2.0, 0.0, 0.0, 0.0, 0.0])
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=2.0, measure=2.0, decision="rejected")
        if forced:
            probe.update(candidate_forcing_checks=1, candidate_forcing_returns=1)
            probe["last"].update(status="forced_candidate", initial_measure=3e9, final_measure=2.0,
                block_measures=[2.0, 0.0, 0.0, 0.0, 0.0], tolerances=[1e-10, 1e-12, 1e-12, 1e-12, 1e-12])
            probe["last"]["initial_forcing"].update(rhs_norms=[1.0, 0.0, 0.0],
                defect_norms=[0.3, 0.0, 0.0, 0.0, 0.0], measure=3.0)
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-10, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=3e9, measure=2e-9, decision="returned")
        delta = dict.fromkeys(REUSE_PUBLIC, 0)
        delta.update(constant_solves=1, direction_solves=directions + steps, scaled_products=directions + 2 * steps,
            refinement_steps=steps, scaled_refinement_steps=steps, inexact_returns=int(forced))
        applications = 1 + directions + steps
        event = dict(attempt_iteration=2, factor_epoch=1, status="bounded_rejected" if rejected else "accepted",
            coupled=probe, energy=energy(n=n, m=m), public_delta=delta,
            current_diagonal=current(n, m, nonzeros, applications), transport=transport(n, m, nonzeros, applications),
            scalar_consistency=scalar(n, m))
        value = dict(schema=4, enabled=True, max_krylov_steps=1, preconditioner="current_diagonal_congruence",
            energy_model="current_positive_approximation", fresh_started=2 if rejected else 1,
            fresh_completed=2 if rejected else 1, reused_steps=0 if rejected else 1, events=[event])
        main = coupled(iteration=1, n=n, m=m)
        extra = coupled(n=n, m=m) if rejected else probe
        for key in validator.COUPLED_COUNTS:
            main[key] += extra[key]
        main["last"] = copy.deepcopy(extra["last"])
        report = dict(options=dict(ipm=dict(max_iterations=200)), ipm_iterations=2, status="optimal")
        for key, field in REUSE_PUBLIC.items():
            report[field] = dict(constant_solves=2 if rejected else 1, direction_solves=4 if rejected else 2,
                scaled_products=4 if rejected else 2, factorizations=3 if rejected else 2).get(key, 0) + delta[key]
        return value, dict(report=report, coupled_marker=main, energy_marker=energy(n=n, m=m, calls=2))

    fixtures = [fixture(mode) for mode in ("initial", "strict", "forced", "rejected", "both")]
    fixtures.append(fixture(n=0, m=0, nonzeros=0))
    for value, row in fixtures:
        original = copy.deepcopy(row)
        reuse_selected_report(value, row, validator)
        assert typed_equal(original, row)
    cases = len(fixtures)

    def stopped():
        value, row = fixture()
        value["reused_steps"] = 0
        event = value["events"][0]
        event.update(status="budget", coupled=validator.empty_coupled(True), energy=validator.empty_energy(True),
            public_delta=dict.fromkeys(REUSE_PUBLIC, 0), current_diagonal=current(applications=0),
            transport=transport(applications=0), scalar_consistency=scalar(complete=False))
        row.update(coupled_marker=coupled(iteration=1), energy_marker=energy(iteration=1))
        row["report"].update(ipm_iterations=1, status="time_limit")
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] = dict(constant_solves=1, direction_solves=2, scaled_products=2, factorizations=2).get(key, 0)
        return value, row

    preparations = []
    for phase in range(9):
        value, row = stopped()
        work = dict.fromkeys(TRANSPORT_FIELDS, 0)
        work.update(columns=2, rows=3, nonzeros=4)
        for key in TRANSPORT_RANGES:
            work[key] = None
        if phase >= 1:
            work.update(cache_epoch=1, budget_checks=1)
        if phase >= 2:
            work["prepare_started"] = 1
        if phase >= 3:
            work.update(old_build_started=1, old_row_terms=1, budget_checks=2)
        if phase >= 4:
            work.update(old_row_terms=3, scratch_row_terms=2, budget_checks=3)
        if phase >= 5:
            work.update(scratch_row_terms=6, merge_terms=2, budget_checks=4)
        if phase >= 6:
            work.update(unique_terms=2, current_terms=2, old_terms=2, column_terms=1,
                old_min=1.0, old_max=2.0, current_min=0.5, current_max=4.0, scale_min=0.5, scale_max=2.0)
        if phase >= 7:
            work = transport(applications=0)
            work["prepare_completed"] = 0
        if phase >= 8:
            work["prepare_completed"] = 1
        value["events"][0]["transport"] = work
        reuse_selected_report(value, row, validator)
        preparations.append((value, row))
        cases += 1

    applications = []
    for phase in range(10):
        value, row = stopped()
        event = value["events"][0]
        event["coupled"]["constant_calls_started"] = 1
        event["current_diagonal"].update(applications_started=1, transpose_started=1, transpose_completed=1,
            transpose_terms=4, direct_started=1, row_divisions=3)
        work = event["transport"]
        if phase >= 1:
            work.update(input_started=1, input_terms=1, budget_checks=30)
        if phase >= 2:
            work.update(input_completed=1, input_terms=2)
        if phase >= 3:
            work["direct_started"] = 1
        if phase >= 4:
            event["public_delta"]["constant_solves"] = 1
            row["report"]["ipm_homogeneous_constant_solves"] += 1
        if phase >= 5:
            work["direct_completed"] = 1
        if phase >= 6:
            work.update(output_started=1, output_terms=1)
        if phase >= 7:
            work.update(output_completed=1, output_terms=2)
            event["current_diagonal"]["direct_completed"] = 1
        if phase >= 8:
            event["current_diagonal"].update(product_started=1, product_completed=1, product_terms=4, row_divisions=6)
        if phase >= 9:
            event["current_diagonal"]["applications_completed"] = 1
        reuse_selected_report(value, row, validator)
        applications.append((value, row))
        cases += 1

    scalars = []
    for phase in range(7):
        value, row = stopped()
        event = value["events"][0]
        event["coupled"].update(constant_calls_started=1, constant_calls_completed=1)
        event.update(current_diagonal=current(applications=1), transport=transport(applications=1), energy=energy())
        event["public_delta"]["constant_solves"] = 1
        row["report"]["ipm_homogeneous_constant_solves"] += 1
        work = event["scalar_consistency"]
        if phase >= 1:
            work["budget_checks"] = 1
        if phase >= 2:
            work["started"] = 1
        if phase >= 3:
            work.update(terms=2, budget_checks=2)
        if phase >= 4:
            work["terms"] = 5
        if phase >= 5:
            work.update(natural_denominator=0.5, positive_denominator=1.0, relative_difference=-0.5, budget_checks=3)
        if phase >= 6:
            work["completed"] = 1
        reuse_selected_report(value, row, validator)
        scalars.append((value, row))
        cases += 1

    sequences = []
    for first_rejected in (False, True):
        value, row = fixture("rejected" if first_rejected else "initial")
        following = copy.deepcopy(fixture()[0]["events"][0])
        following.update(attempt_iteration=3, factor_epoch=2 if first_rejected else 1)
        following["coupled"]["last"]["attempt_iteration"] = 3
        following["energy"]["last"]["attempt_iteration"] = 3
        following["transport"] = transport(hit=not first_rejected, epoch=following["factor_epoch"])
        value["events"].append(following)
        value["reused_steps"] += 1
        row["report"]["ipm_iterations"] = 3
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] += following["public_delta"][key]
        for key in validator.COUPLED_COUNTS:
            row["coupled_marker"][key] += following["coupled"][key]
        row["coupled_marker"]["last"] = copy.deepcopy(following["coupled"]["last"])
        row["energy_marker"] = energy(iteration=3, calls=3)
        reuse_selected_report(value, row, validator)
        sequences.append((value, row))
        cases += 1

    narrowed, narrowed_row = copy.deepcopy(fixtures[0][0]), copy.deepcopy(fixtures[0][1])
    narrowed_event = narrowed["events"][0]
    for key in ("old_min", "old_max", "current_min", "current_max"):
        narrowed_event["transport"][key] = None
    narrowed_event["energy"]["last"]["denominator"] = None
    narrowed_row["energy_marker"]["last"]["denominator"] = None
    narrowed_event["scalar_consistency"].update(natural_denominator=None,
        positive_denominator=None, relative_difference=None)
    reuse_selected_report(narrowed, narrowed_row, validator)
    cases += 1
    unavailable, unavailable_row = copy.deepcopy(preparations[6][0]), copy.deepcopy(preparations[6][1])
    unavailable.update(fresh_started=2)
    unavailable["events"][0]["status"] = "numerical_error"
    unavailable["events"][0]["transport"].update(old_min=1e308, old_max=1e308,
        current_min=0.0, current_max=0.0, scale_min=None, scale_max=None)
    reuse_selected_report(unavailable, unavailable_row, validator)
    cases += 1
    for natural, relative in ((-1.0, -2.0), (None, 1.0)):
        value, row = copy.deepcopy(fixtures[0][0]), copy.deepcopy(fixtures[0][1])
        value["events"][0]["scalar_consistency"].update(natural_denominator=natural, relative_difference=relative)
        reuse_selected_report(value, row, validator)
        cases += 1

    accepted, accepted_row = fixtures[0]
    mutations = []
    def changed(path, wrong, source=accepted, source_row=accepted_row):
        value, row = copy.deepcopy(source), copy.deepcopy(source_row)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = wrong
        mutations.append((value, row))
    changed(("schema",), True)
    changed(("preconditioner",), "current_diagonal_schur")
    changed(("energy_model",), "exact_schur_inverse")
    for field in ("transport", "scalar_consistency"):
        value, row = copy.deepcopy(accepted), copy.deepcopy(accepted_row)
        del value["events"][0][field]
        mutations.append((value, row))
    for field in TRANSPORT_FIELDS:
        changed(("events", 0, "transport", field), True)
    for field in ("started", "completed", "terms", "budget_checks", "natural_denominator", "positive_denominator", "relative_difference"):
        changed(("events", 0, "scalar_consistency", field), True)
    for field, wrong in (("columns", 3), ("cache_epoch", 2), ("prepare_started", 2), ("prepare_completed", 0),
            ("cache_hits", 1), ("old_build_started", 0), ("old_build_completed", 0), ("old_row_terms", 2),
            ("scratch_row_terms", 5), ("merge_terms", 3), ("unique_terms", 5), ("old_terms", 3),
            ("current_terms", 3), ("column_terms", 1), ("input_started", 4), ("input_completed", 2),
            ("input_terms", 5), ("direct_started", 2), ("direct_completed", 2), ("output_started", 2),
            ("output_completed", 2), ("output_terms", 5), ("budget_checks", 17),
            ("old_min", -1.0), ("scale_min", 0.0), ("scale_max", 0.25)):
        changed(("events", 0, "transport", field), wrong)
    for field, wrong in (("started", 0), ("completed", 0), ("terms", 4), ("budget_checks", 1),
            ("positive_denominator", 2.0), ("relative_difference", -2.01)):
        changed(("events", 0, "scalar_consistency", field), wrong)
    changed(("events", 1, "transport", "cache_hits"), 0, sequences[0][0], sequences[0][1])
    changed(("events", 1, "transport", "old_build_started"), 1, sequences[0][0], sequences[0][1])
    changed(("events", 1, "transport", "cache_hits"), 1, sequences[1][0], sequences[1][1])
    changed(("events", 0, "transport", "cache_hits"), 1, preparations[2][0], preparations[2][1])
    changed(("events", 0, "transport", "scratch_row_terms"), 7, preparations[4][0], preparations[4][1])
    changed(("events", 0, "transport", "input_started"), 1, preparations[7][0], preparations[7][1])
    changed(("events", 0, "transport", "direct_started"), 1, applications[1][0], applications[1][1])
    changed(("events", 0, "transport", "direct_completed"), 1, applications[3][0], applications[3][1])
    changed(("events", 0, "transport", "output_started"), 1, applications[4][0], applications[4][1])
    changed(("events", 0, "transport", "output_completed"), 1, applications[6][0], applications[6][1])
    changed(("events", 0, "scalar_consistency", "started"), 1, applications[9][0], applications[9][1])
    changed(("events", 0, "scalar_consistency", "natural_denominator"), 1.0, scalars[3][0], scalars[3][1])
    changed(("events", 0, "scalar_consistency", "terms"), 6, scalars[3][0], scalars[3][1])
    for value, row in mutations:
        try:
            reuse_selected_report(value, row, validator)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass
        else:
            raise AssertionError("畸形对角合同或标量观察账本被接受")
        cases += 1
    return cases


def reuse_natural_schur_self_check(validator):
    """schema5独立检查向外包围、费用和有符号恢复的控制流，不改历史自检。"""
    def coupled(iteration=2, directions=2, steps=0, n=2, m=3):
        last = dict(attempt_iteration=iteration, role="corrector" if directions == 2 else "affine",
            status="accepted", dimension=n + m + 2, initial_measure=0.0, final_measure=0.0,
            block_measures=[0.0] * 5, tolerances=[1e-12] * 5,
            initial_forcing=dict(rhs_norms=[0.0] * 3, defect_norms=[0.0] * 5,
                measure=None, decision="strict_initial"), candidate_forcing=validator.empty_candidate_forcing())
        if steps:
            last.update(initial_measure=2.0, final_measure=0.5, block_measures=[0.5, 0.0, 0.0, 0.0, 0.0])
            last["initial_forcing"].update(defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0],
                measure=2.0, decision="rejected")
        return dict(validator.empty_coupled(True), constant_calls_started=1, constant_calls_completed=1,
            directions_started=directions, directions_completed=directions,
            preconditioner_started=directions + steps, preconditioner_completed=directions + steps,
            products_started=directions + 2 * steps, products_completed=directions + 2 * steps,
            true_checks_started=directions + steps, true_checks_completed=directions + steps,
            krylov_started=steps, krylov_completed=steps, cycles_started=steps, initial_forcing_checks=steps, last=last)

    def energy(iteration=2, n=2, m=3, calls=1):
        return dict(validator.empty_energy(True), started=calls, completed=calls,
            primal_terms=calls * n, row_terms=calls * m,
            budget_checks=calls * (2 + (n + 4095) // 4096 + (m + 4095) // 4096),
            last=dict(attempt_iteration=iteration, status="completed", columns=n, rows=m,
                primal_regularization=1e-12, primal_terms=n, row_terms=m,
                primal_energy=0.0, row_energy=0.0, denominator=1.0))

    def current(n=2, m=3, nonzeros=4, applications=3):
        return dict(columns=n, rows=m, nonzeros=nonzeros,
            current_rows_started=1, current_rows_completed=1, current_rows_terms=m,
            validation_started=1, validation_completed=1, validation_terms=n + 1 + m + nonzeros,
            applications_started=applications, applications_completed=applications,
            transpose_started=applications, transpose_completed=applications,
            direct_started=applications, direct_completed=applications,
            product_started=applications, product_completed=applications,
            transpose_terms=applications * nonzeros, product_terms=applications * nonzeros,
            row_divisions=2 * m * applications, budget_checks=100)

    def transport(n=2, m=3, nonzeros=4, applications=3, hit=False, epoch=1):
        result = dict.fromkeys(TRANSPORT_FIELDS, 0)
        result.update(columns=n, rows=m, nonzeros=nonzeros, cache_epoch=epoch,
            prepare_started=1, prepare_completed=1, cache_hits=int(hit),
            old_build_started=int(not hit), old_build_completed=int(not hit), old_row_terms=0 if hit else m,
            scratch_row_terms=2 * m, merge_terms=nonzeros, unique_terms=nonzeros,
            old_terms=0 if hit else nonzeros, current_terms=nonzeros, column_terms=n,
            input_started=applications, input_completed=applications, input_terms=n * applications,
            direct_started=applications, direct_completed=applications,
            output_started=applications, output_completed=applications, output_terms=n * applications,
            budget_checks=10 + 8 * applications)
        for key in TRANSPORT_RANGES:
            result[key] = None
        if n:
            result.update(old_min=1.0, old_max=2.0, current_min=0.5, current_max=4.0, scale_min=0.5, scale_max=2.0)
        return result

    def scalar(n=2, m=3):
        return dict(started=1, completed=1, terms=n + m, budget_checks=2 + (n + m + 4095) // 4096,
            natural_denominator=0.5 if n + m else 1.0, positive_denominator=1.0,
            relative_difference=-0.5 if n + m else 0.0)

    def natural(k=5, sign=1, complete=True):
        result = dict.fromkeys(NATURAL_SCHUR_COUNTS, 0)
        result.update(decision="not_started", **dict.fromkeys(NATURAL_SCHUR_NUMBERS, None))
        if complete:
            result.update(started=1, completed=1, terms=k, budget_checks=2 + (k + 4095) // 4096,
                environment_checks=2 + (k + 4095) // 4096, divisions=k, products=k, accumulations=k,
                abs_accumulations=k, final_subtractions=1, bound_operations=4 * k + 4,
                enclosure_steps=6 * k + 4, decision="reliable", natural=float(sign), abs_products=2.0,
                lower=0.9 if sign > 0 else -1.1, upper=1.1 if sign > 0 else -0.9,
                error_bound=0.1, margin=0.9)
            if k == 0:
                result.update(natural=1.0, abs_products=0.0, lower=1.0, upper=1.0,
                    error_bound=2.0 ** -63, margin=1.0)
        return result

    def fixture(mode="initial", n=2, m=3, nonzeros=4, sign=1):
        rejected, forced = mode == "rejected", mode == "forced"
        steps = 2 if mode == "both" else int(mode != "initial")
        directions = 1 if rejected else 2
        probe = coupled(directions=directions, steps=steps, n=n, m=m)
        if rejected:
            probe.update(directions_completed=0, candidate_forcing_checks=1)
            probe["last"].update(status="bounded_rejected", final_measure=2.0, block_measures=[2.0, 0.0, 0.0, 0.0, 0.0])
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-12, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=2.0, measure=2.0, decision="rejected")
        if forced:
            probe.update(candidate_forcing_checks=1, candidate_forcing_returns=1)
            probe["last"].update(status="forced_candidate", initial_measure=3e9, final_measure=2.0,
                block_measures=[2.0, 0.0, 0.0, 0.0, 0.0], tolerances=[1e-10, 1e-12, 1e-12, 1e-12, 1e-12])
            probe["last"]["initial_forcing"].update(rhs_norms=[1.0, 0.0, 0.0],
                defect_norms=[0.3, 0.0, 0.0, 0.0, 0.0], measure=3.0)
            probe["last"]["candidate_forcing"] = dict(cycle=1, krylov_step=1,
                defect_norms=[2e-10, 0.0, 0.0, 0.0, 0.0], strict_measure=2.0,
                best_measure_before=3e9, measure=2e-9, decision="returned")
        delta = dict.fromkeys(REUSE_PUBLIC, 0)
        delta.update(constant_solves=1, direction_solves=directions + steps, scaled_products=directions + 2 * steps,
            refinement_steps=steps, scaled_refinement_steps=steps, inexact_returns=int(forced))
        applications = 1 + directions + steps
        event = dict(attempt_iteration=2, factor_epoch=1, status="bounded_rejected" if rejected else "accepted",
            coupled=probe, energy=energy(n=n, m=m), public_delta=delta,
            current_diagonal=current(n, m, nonzeros, applications), transport=transport(n, m, nonzeros, applications),
            scalar_consistency=scalar(n, m), natural_schur=natural(n + m, sign))
        value = dict(schema=5, enabled=True, max_krylov_steps=1, preconditioner="current_diagonal_congruence",
            energy_model="current_positive_approximation", scalar_model="natural_schur_enclosed",
            fresh_started=2 if rejected else 1, fresh_completed=2 if rejected else 1,
            reused_steps=0 if rejected else 1, events=[event])
        main = coupled(iteration=1, n=n, m=m)
        extra = coupled(n=n, m=m) if rejected else probe
        for key in validator.COUPLED_COUNTS:
            main[key] += extra[key]
        main["last"] = copy.deepcopy(extra["last"])
        report = dict(options=dict(ipm=dict(max_iterations=200)), ipm_iterations=2, status="optimal")
        for key, field in REUSE_PUBLIC.items():
            report[field] = dict(constant_solves=2 if rejected else 1, direction_solves=4 if rejected else 2,
                scaled_products=4 if rejected else 2, factorizations=3 if rejected else 2).get(key, 0) + delta[key]
        return value, dict(report=report, coupled_marker=main, energy_marker=energy(n=n, m=m, calls=2))

    def stopped(status="budget", n=2, m=3, nonzeros=4):
        value, row = fixture(n=n, m=m, nonzeros=nonzeros)
        value.update(reused_steps=0, fresh_started=2 if status == "numerical_error" else 1)
        event = value["events"][0]
        event.update(status=status, coupled=validator.empty_coupled(True),
            public_delta=dict.fromkeys(REUSE_PUBLIC, 0), current_diagonal=current(n, m, nonzeros, 1),
            transport=transport(n, m, nonzeros, 1), natural_schur=natural(n + m, complete=False))
        event["coupled"].update(constant_calls_started=1, constant_calls_completed=1)
        event["public_delta"]["constant_solves"] = 1
        row.update(coupled_marker=coupled(iteration=1, n=n, m=m), energy_marker=energy(iteration=1, n=n, m=m))
        row["report"].update(ipm_iterations=1, status="time_limit")
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] = dict(constant_solves=2, direction_solves=2, scaled_products=2, factorizations=2).get(key, 0)
        return value, row

    fixtures = [fixture(mode) for mode in ("initial", "strict", "forced", "rejected", "both")]
    fixtures.extend((fixture(sign=-1), fixture(n=0, m=0, nonzeros=0)))
    cases = 0
    for value, row in fixtures:
        original = copy.deepcopy(row)
        reuse_selected_report(value, row, validator)
        assert typed_equal(original, row)
        cases += 1

    def good(value, row):
        nonlocal cases
        original = copy.deepcopy(row)
        reuse_selected_report(value, row, validator)
        assert typed_equal(original, row)
        cases += 1

    # 摘要可能窄化为null或零，旧观察也不要求与新求值顺序逐值一致。
    for narrowed in (None, 0.0):
        value, row = fixture()
        value["events"][0]["natural_schur"].update(dict.fromkeys(NATURAL_SCHUR_NUMBERS, narrowed))
        if narrowed == 0.0:
            value["events"][0]["natural_schur"]["abs_products"] = 1.0
        good(value, row)

    sequences = []
    for first_rejected in (False, True):
        value, row = fixture("rejected" if first_rejected else "initial")
        following = copy.deepcopy(fixture(sign=-1)[0]["events"][0])
        following.update(attempt_iteration=3, factor_epoch=2 if first_rejected else 1)
        following["coupled"]["last"]["attempt_iteration"] = 3
        following["energy"]["last"]["attempt_iteration"] = 3
        following["transport"] = transport(hit=not first_rejected, epoch=following["factor_epoch"])
        value["events"].append(following)
        value["reused_steps"] += 1
        row["report"]["ipm_iterations"] = 3
        for key, field in REUSE_PUBLIC.items():
            row["report"][field] += following["public_delta"][key]
        for key in validator.COUPLED_COUNTS:
            row["coupled_marker"][key] += following["coupled"][key]
        row["coupled_marker"]["last"] = copy.deepcopy(following["coupled"]["last"])
        row["energy_marker"] = energy(iteration=3, calls=3)
        good(value, row)
        sequences.append((value, row))

    prefixes = []
    def prefix(work, status="budget", n=2, m=3, nonzeros=4):
        value, row = stopped(status, n, m, nonzeros)
        value["events"][0]["natural_schur"] = work
        good(value, row)
        prefixes.append((value, row))

    before, before_row = stopped()
    before_event = before["events"][0]
    before_event.update(coupled=validator.empty_coupled(True), energy=validator.empty_energy(True),
        current_diagonal=current(applications=0), transport=transport(applications=0),
        scalar_consistency=dict(started=0, completed=0, terms=0, budget_checks=0,
            natural_denominator=None, positive_denominator=None, relative_difference=None))
    before_event["public_delta"]["constant_solves"] = 0
    before_row["report"]["ipm_homogeneous_constant_solves"] -= 1
    good(before, before_row)
    observing, observing_row = stopped()
    observing["events"][0]["scalar_consistency"].update(completed=0, terms=2, budget_checks=2,
        natural_denominator=None, positive_denominator=None, relative_difference=None)
    good(observing, observing_row)

    # 首检查、无效d、下一批检查可各停在预算和环境检查之间。
    for phase in range(7):
        work = natural(complete=False)
        status = "budget"
        if phase >= 1:
            work["budget_checks"] = 1
        if phase == 2:
            work.update(environment_checks=1, decision="unsupported")
            status = "numerical_error"
        if phase == 3:
            work["environment_checks"] = 1
            status = "exception"
        if phase >= 4:
            work.update(started=1, environment_checks=1, decision="running")
        if phase >= 5:
            work["budget_checks"] = 2
        if phase >= 6:
            work.update(environment_checks=2, decision="unsupported")
            status = "numerical_error"
        prefix(work, status)

    # 第二项可以在中心算术和六次向外舍入的任一有限性检查处数值拒绝。
    for stage in range(5):
        work = natural(complete=False)
        work.update(started=1, terms=2, budget_checks=2, environment_checks=2, decision="nonfinite",
            divisions=1 + int(stage >= 1), products=1 + int(stage >= 2),
            accumulations=1 + int(stage >= 3), abs_accumulations=1 + int(stage >= 4),
            bound_operations=4, enclosure_steps=6)
        prefix(work, "numerical_error")
    for bound, enclosure in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
            (2, 4), (3, 4), (3, 5), (4, 5), (4, 6)):
        work = natural(complete=False)
        work.update(started=1, terms=2, budget_checks=2, environment_checks=2, decision="nonfinite",
            divisions=2, products=2, accumulations=2, abs_accumulations=2,
            bound_operations=4 + bound, enclosure_steps=6 + enclosure)
        prefix(work, "numerical_error")

    for final, bound, enclosure in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 2, 1),
            (1, 2, 2), (1, 3, 2), (1, 3, 3), (1, 4, 3), (1, 4, 4)):
        work = natural(complete=False)
        work.update(started=1, terms=5, budget_checks=2, environment_checks=2, decision="nonfinite",
            divisions=5, products=5, accumulations=5, abs_accumulations=5,
            final_subtractions=final, bound_operations=20 + bound, enclosure_steps=30 + enclosure)
        prefix(work, "numerical_error")

    deliveries = []
    for decision, environment, status in (("reliable", 2, "budget"), ("uncertain", 2, "budget"),
            ("uncertain", 3, "numerical_error"), ("unsupported", 3, "numerical_error"),
            ("reliable", 3, "budget")):
        work = natural()
        work.update(completed=int(decision == "reliable" and environment == 3),
            decision=decision, environment_checks=environment)
        if decision == "uncertain":
            work.update(natural=0.0, lower=-0.1, upper=0.1, margin=0.0)
        prefix(work, status)
        deliveries.append(prefixes[-1])

    for budget, environment, decision, status in ((2, 2, "running", "budget"),
            (3, 2, "running", "budget"), (3, 3, "unsupported", "numerical_error")):
        work = natural(complete=False)
        work.update(started=1, terms=4096, budget_checks=budget, environment_checks=environment,
            decision=decision, divisions=4096, products=4096, accumulations=4096, abs_accumulations=4096,
            bound_operations=4 * 4096, enclosure_steps=6 * 4096)
        prefix(work, status, n=4094)
    empty_work = natural(k=0)
    empty_work.update(completed=0, environment_checks=1)
    prefix(empty_work, n=0, m=0, nonzeros=0)

    mutations = []
    accepted, accepted_row = fixtures[0]
    def changed(path, wrong, source=accepted, source_row=accepted_row):
        value, row = copy.deepcopy(source), copy.deepcopy(source_row)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = wrong
        mutations.append((value, row))

    changed(("scalar_model",), "current_positive_approximation")
    changed(("preconditioner",), "current_diagonal_schur")
    changed(("energy_model",), "exact_schur_inverse")
    value, row = fixture()
    del value["events"][0]["natural_schur"]
    mutations.append((value, row))
    for field in NATURAL_SCHUR_COUNTS + NATURAL_SCHUR_NUMBERS + ("decision",):
        changed(("events", 0, "natural_schur", field), True)
    for field, wrong in (("started", 0), ("completed", 0), ("terms", 4), ("budget_checks", 2),
            ("environment_checks", 2), ("divisions", 4), ("products", 6), ("accumulations", 4),
            ("abs_accumulations", 6), ("final_subtractions", 0), ("bound_operations", 23),
            ("enclosure_steps", 33), ("decision", "uncertain"), ("decision", "unsupported"),
            ("decision", "nonfinite"), ("abs_products", -1.0), ("error_bound", -1.0),
            ("margin", -1.0), ("lower", 2.0), ("upper", 0.5)):
        changed(("events", 0, "natural_schur", field), wrong)
    changed(("max_krylov_steps",), 2)
    changed(("scalar_model",), True)
    value, row = fixture()
    del value["scalar_model"]
    mutations.append((value, row))
    value, row = fixture()
    value["events"][0]["natural_schur"]["unpaid_bound"] = 1
    mutations.append((value, row))
    for field, wrong in (("budget_checks", 4), ("environment_checks", 4), ("divisions", 6),
            ("accumulations", 6), ("abs_accumulations", 4), ("final_subtractions", 2),
            ("bound_operations", 25), ("enclosure_steps", 35), ("decision", "converged"),
            ("natural", -1.0), ("lower", -1.0), ("terms", -1), ("budget_checks", -1),
            ("products", None), ("abs_products", float("inf")), ("error_bound", float("nan"))):
        changed(("events", 0, "natural_schur", field), wrong)
    changed(("events", 0, "natural_schur", "environment_checks"), 0, *prefixes[2])
    wrong_status, wrong_status_row = copy.deepcopy(prefixes[3][0]), copy.deepcopy(prefixes[3][1])
    wrong_status.update(fresh_started=2)
    wrong_status["events"][0]["status"] = "numerical_error"
    mutations.append((wrong_status, wrong_status_row))
    for field, wrong in (("natural", 1.0), ("products", 2), ("enclosure_steps", 7),
            ("bound_operations", 5), ("budget_checks", 3)):
        changed(("events", 0, "natural_schur", field), wrong, *prefixes[7])
    changed(("events", 0, "natural_schur", "margin"), 1.0, *deliveries[2])
    changed(("events", 0, "natural_schur", "completed"), 1, *deliveries[2])
    changed(("events", 0, "natural_schur", "environment_checks"), 3, *deliveries[0])
    changed(("events", 0, "natural_schur", "budget_checks"), 1, before, before_row)
    changed(("events", 0, "natural_schur", "completed"), 1, *prefixes[-1])
    for value, row in mutations:
        try:
            reuse_selected_report(value, row, validator)
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
            pass
        else:
            raise AssertionError("畸形自然Schur包围或费用账本被接受")
        cases += 1
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        result = self_check()
        if args.plan:
            plan = read(args.plan)
            check_protected(plan)
            validator = load_validator(plan)
            result["reuse_cases"] = reuse_self_check(validator)
            result["reuse_bounded_cases"] = reuse_bounded_self_check(validator)
            result["reuse_current_diagonal_cases"] = reuse_current_diagonal_self_check(validator)
            result["reuse_transport_cases"] = reuse_transport_self_check(validator)
            result["reuse_natural_schur_cases"] = reuse_natural_schur_self_check(validator)
            for settings in plan["stages"].values():
                for item in settings["runs"]:
                    check_command(item)
            historical = 0
            for path in plan["frozen_runs"]:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        if not validator.valid_process(benchmark.strict_json(line)):
                            raise ValueError("既有冻结记录未通过继承校验")
                        historical += 1
            if historical != plan["maximum_solver_runs"]:
                raise ValueError("冻结重放总数不符")
            result["historical_records_validated"] = historical
        print(json.dumps(result, ensure_ascii=False))
    else:
        if not args.plan or not args.output or not args.stage:
            parser.error("重放需要计划、输出目录和阶段")
        run(args.plan.resolve(), args.output.resolve(), args.stage)


if __name__ == "__main__":
    main()
