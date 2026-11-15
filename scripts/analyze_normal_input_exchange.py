"""只读核验第127轮固定Normal微测；负门槛结果也完整保存。"""

import argparse
import json
import math
from pathlib import Path
import random
import re
import sys


STRUCTURES = ("ring_band", "block_arrow_16")
SIZES = (64, 4096, 65536)
WIDTHS = (3, 9)
LIFECYCLES = ("cold", "warm")
CELLS = tuple((structure, size, width) for structure in STRUCTURES for size in SIZES for width in WIDTHS)
INTEGER_FIELDS = (
    "sequence", "cell", "size", "width", "block", "position", "epochs", "warmup_epochs", "rhs_per_factor",
    "dimension", "input_nonzeros", "gram_contributions", "elapsed_ns", "setup_ns", "warmup_ns",
    "destruction_ns", "total_ns", "factor_calls", "solve_calls", "total_factor_calls", "total_solve_calls",
    "budget_calls", "constructors", "copied_entries", "exchanged_entries", "warmup_copied_entries",
    "warmup_exchanged_entries", "copy_payload_bytes", "copy_logical_read_write_bytes", "factor_nnz_sum")
DIGEST_FIELDS = ("result_digest", "input_digest", "budget_digest", "factor_digest")
ERROR_FIELDS = ("max_forward_error", "max_residual_error")
SAMPLE_FIELDS = set(INTEGER_FIELDS + DIGEST_FIELDS + ERROR_FIELDS + (
    "type", "schema", "structure", "lifecycle", "mode", "passed", "error", "transfer_ledger_complete"))
METADATA = dict(type="metadata", schema=1, round=127, cells=12, samples=432,
    kernel="cholmod_supernodal_aat", coupled_context_enabled=True, threads=1,
    digest="xor_multiply_u64", digest_is_proof=False, public_lp_runs=0)


def typed_equal(first, second):
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(typed_equal(first[key], second[key]) for key in first)
    return first == second


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("JSON对象含重复字段：" + key)
        result[key] = value
    return result


def bad_constant(value):
    raise ValueError("JSON包含非有限常量：" + value)


def finite(number):
    return type(number) in (int, float) and math.isfinite(number)


def contribution_count(structure, size, width):
    if structure == "ring_band":
        return size * width * (width + 1) // 2
    rest = 16 - (width - 1)
    quotient, remainder = divmod(16, rest)
    other = ((rest - remainder) * quotient * (quotient + 1) // 2 +
        remainder * (quotient + 1) * (quotient + 2) // 2)
    return size // 16 * ((width - 1) * 16 * 17 // 2 + other)


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def ratios(pairs, lifecycle):
    groups = []
    details = []
    for cell, (structure, size, width) in enumerate(CELLS):
        logs = [math.log(pairs[cell, lifecycle, block]["exchange"]["elapsed_ns"] /
            pairs[cell, lifecycle, block]["copy"]["elapsed_ns"]) for block in range(9)]
        groups.append(logs)
        details.append(dict(cell=cell, structure=structure, size=size, width=width,
            paired_log_ratios=logs, exchange_over_copy_ratio=math.exp(math.fsum(logs) / 9)))
    mean_log = math.fsum(math.fsum(group) / 9 for group in groups) / 12
    # 每个生命周期独立使用seed0，按12格分层，固定10000次且不重挑样本。
    generator = random.Random(0)
    estimates = []
    for _ in range(10000):
        estimate = math.fsum(math.fsum(group[generator.randrange(9)] for _ in range(9)) / 9
            for group in groups) / 12
        estimates.append(estimate)
    estimates.sort()
    return dict(exchange_over_copy_ratio=math.exp(mean_log),
        confidence_interval_95=[math.exp(percentile(estimates, 0.025)), math.exp(percentile(estimates, 0.975))],
        bootstrap_seed=0, bootstrap_repetitions=10000,
        weighting="12结构格等权；每格9个成对log；按格有放回重采样；百分位线性插值",
        cells=details)


def analyze(path):
    errors, samples, metadata, summaries = [], [], [], []
    line_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_count, line in enumerate(stream, 1):
            try:
                value = json.loads(line, object_pairs_hook=unique_object, parse_constant=bad_constant)
                if not isinstance(value, dict) or value.get("type") not in {"sample", "metadata", "summary"}:
                    raise ValueError("未知记录类型或非对象记录")
            except (ValueError, TypeError) as error:
                errors.append(dict(line=line_count, error=str(error)))
                continue
            if value["type"] == "sample":
                samples.append(value)
            elif value["type"] == "metadata":
                metadata.append((line_count, value))
            else:
                summaries.append((line_count, value))

    def require(condition, message, sequence=None):
        if not condition:
            errors.append(dict(sequence=sequence, error=message))

    require(len(metadata) == 1 and metadata[0][0] == 1 and typed_equal(metadata[0][1], METADATA),
        "微测元数据、身份、顺序或固定配置无效")
    require(len(samples) == 432, "原始样本必须恰为432条，不能删格、删失败或补跑择优")
    require(len(summaries) == 1 and summaries[0][0] == line_count, "必须恰有一个末尾summary")
    expected = []
    for cell in range(12):
        for life, lifecycle in enumerate(LIFECYCLES):
            for block in range(9):
                first = "exchange" if (cell + life + block) % 2 else "copy"
                for position, mode in enumerate((first, "copy" if first == "exchange" else "exchange")):
                    expected.append((cell, lifecycle, block, position, mode))
    pairs, seen = {}, set()
    failures = 0
    for sequence, value in enumerate(samples):
        shape = (set(value) == SAMPLE_FIELDS and type(value.get("schema")) is int and value["schema"] == 1 and
            all(type(value.get(key)) is int and 0 <= value[key] <= 18446744073709551615 for key in INTEGER_FIELDS) and
            all(type(value.get(key)) is str and re.fullmatch(r"[0-9a-f]{16}", value[key]) for key in DIGEST_FIELDS) and
            type(value.get("passed")) is bool and type(value.get("transfer_ledger_complete")) is bool and
            type(value.get("error")) is str and
            all(value.get(key) is None or finite(value[key]) and value[key] >= 0 for key in ERROR_FIELDS))
        require(shape, "样本字段、严格类型或有限性无效", sequence)
        if not shape:
            continue
        failures += not value["passed"]
        require(value["passed"] and value["transfer_ledger_complete"] and value["error"] == "" and all(
            value[key] is not None and value[key] <= 1e-11 for key in ERROR_FIELDS),
            "原程序失败或未通过已知解/独立真实残差1e-11门槛", sequence)
        if sequence >= len(expected):
            continue
        cell, lifecycle, block, position, mode = expected[sequence]
        structure, size, width = CELLS[cell]
        warm = lifecycle == "warm"
        config = dict(sequence=sequence, cell=cell, structure=structure, size=size, width=width,
            lifecycle=lifecycle, block=block, position=position, mode=mode, epochs=8,
            warmup_epochs=3 if warm else 0, rhs_per_factor=3, dimension=2 * size,
            input_nonzeros=size * (width + 2), gram_contributions=contribution_count(structure, size, width))
        require(all(typed_equal(value.get(key), wanted) for key, wanted in config.items()),
            "样本配置、固定epoch或AB/BA顺序改变", sequence)
        key = (value["cell"], value.get("lifecycle"), value["block"], value.get("mode"))
        require(key not in seen, "重复样本或同一组择优替换", sequence)
        seen.add(key)
        nonzeros = size * (width + 2)
        exchanged = mode == "exchange" and warm
        wanted_counts = dict(factor_calls=8, solve_calls=24, total_factor_calls=11 if warm else 8,
            total_solve_calls=33 if warm else 24, constructors=1 if warm else 8,
            copied_entries=0 if exchanged else 8 * nonzeros, exchanged_entries=8 * nonzeros if exchanged else 0,
            warmup_copied_entries=(nonzeros if mode == "exchange" else 3 * nonzeros) if warm else 0,
            warmup_exchanged_entries=2 * nonzeros if exchanged else 0)
        require(all(value[key] == wanted for key, wanted in wanted_counts.items()) and value["budget_calls"] > 0 and
            value["copy_payload_bytes"] == 8 * value["copied_entries"] and
            value["copy_logical_read_write_bytes"] == 16 * value["copied_entries"] and
            value["gram_contributions"] < 8000000 and 8 * size <= value["factor_nnz_sum"] <= 8 * 16777216,
            "实际copy/swap、调用、结构或逻辑字节费用不守恒", sequence)
        require(value["elapsed_ns"] > 0 and value["total_ns"] >= value["elapsed_ns"] and
            (value["total_ns"] >= sum(value[key] for key in (
                "setup_ns", "warmup_ns", "elapsed_ns", "destruction_ns")) if warm else
                value["warmup_ns"] == 0 and value["setup_ns"] + value["destruction_ns"] <= value["elapsed_ns"]),
            "完整生命周期计时遗漏或把预热/销毁当作免费", sequence)
        pairs.setdefault((cell, lifecycle, block), {})[mode] = value
    require(len(pairs) == 216 and all(set(pair) == {"copy", "exchange"} for pair in pairs.values()),
        "必须完整包含216组唯一成对样本")
    compare = DIGEST_FIELDS + ERROR_FIELDS + ("factor_calls", "solve_calls", "total_factor_calls", "total_solve_calls",
        "budget_calls", "constructors", "factor_nnz_sum", "input_nonzeros", "dimension", "gram_contributions")
    pair_errors_before = len(errors)
    for key, pair in pairs.items():
        if set(pair) != {"copy", "exchange"}:
            continue
        require(all(typed_equal(pair["copy"][field], pair["exchange"][field]) for field in compare),
            "成对结果/预算序列/因子统计摘要或真实质量不同：" + str(key))
    paired_consistency_valid = len(errors) == pair_errors_before
    if len(summaries) == 1:
        summary = summaries[0][1]
        require(typed_equal(summary, dict(type="summary", schema=1, samples=len(samples), failures=failures,
            passed=failures == 0 and len(samples) == 432)), "原末尾summary与全部原始样本不符")
    validation_passed = not errors
    cold = ratios(pairs, "cold") if validation_passed else None
    warm = ratios(pairs, "warm") if validation_passed else None
    removed = validation_passed and all(pair["exchange"]["copied_entries"] == 0 and
        pair["exchange"]["exchanged_entries"] == 8 * pair["exchange"]["input_nonzeros"]
        for key, pair in pairs.items() if key[1] == "warm")
    warm_gate = warm is not None and warm["confidence_interval_95"][1] < 1
    cold_gate = cold is not None and cold["confidence_interval_95"][0] <= 1
    gate = validation_passed and removed and warm_gate and cold_gate
    return dict(schema=1, round=127, input=str(path.resolve()), passed=validation_passed,
        raw_line_count=line_count, observed_samples=len(samples), failed_samples=failures,
        paired_consistency_valid=paired_consistency_valid, validation_errors=errors,
        digest_contract="64位字滚动摘要仅作一致性证据，非数学证明；全vector逐位等价由独立C++组件验证",
        transfer_ledger_contract="异常/false时账本只含可证前缀，未记录copy不表示未付费用；失败样本不进入性能门",
        cold=cold, warm=warm, gate=dict(passed=gate, all_records_valid=validation_passed,
            warm_copy_removed=removed, warm_95_upper_below_one=warm_gate, cold_95_lower_at_most_one=cold_gate),
        public_lp_runs=0, goal_achieved=False,
        decision="微测满足事前门槛，可另行冻结原公开500；不代表LP收益或目标达成" if gate else
            "微测未满足事前门槛，本轮不追加公开性能运行；原始失败和全部结构格保留")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出已存在，拒绝覆盖")
    try:
        result = analyze(args.input)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        result = dict(schema=1, round=127, passed=False, input=str(args.input.resolve()),
            validation_errors=[dict(error=str(error))], public_lp_runs=0, goal_achieved=False,
            gate=dict(passed=False), decision="输入读取或校验失败，保留原件且不启动公开性能运行")
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except OSError as error:
        print("分析输出写入失败：" + str(error), file=sys.stderr)
        return 2
    print(json.dumps(dict(passed=result["passed"], gate_passed=result["gate"]["passed"],
        output=str(args.output.resolve())), ensure_ascii=False, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
