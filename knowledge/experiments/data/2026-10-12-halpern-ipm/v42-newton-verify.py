"""独立核验 v42 方向诊断的来源、解析公式、事件账本和选择规则，不编译或求解。"""
import argparse
import collections
import contextlib
import copy
import datetime
import difflib
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v42-newton"
sys.dont_write_bytecode = True
PRODUCTION_SHA = "5a283adb07ed7e15dd3575b021c6a42e1643558a75f4ab09d8cae2a27ff06aa7"
DIAGNOSTIC_SHA = "9ecee9d1d75505ebd44682b9cefd533581609cab33a63fffc6ce7445daf74469"
PLAN_SHA = "31985d25aa6f75bd872dd644eaab9c714fe516b6a77aaeb959d7432bb3300540"
RAW_SHA = "b17e22389f1c67df5a75aba782266724667c0bc9871d98059e06d50da5ae4e38"
SOURCE_CHECKER_SHA = "37490f9b9dc02dc983783f80f579bd993bb8eaccf2f81d1a7e3312a2b9d6dbe6"
PREFIX = "V42_NEWTON "
EXTRA_FIELDS = ("j_started", "j_completed", "csc_started", "csc_completed", "nnz_visits",
                "metric_elements", "copied_elements", "cached_images", "cache_peak_elements", "reconstruction_terms")
METRIC_SUFFIXES = ("weighted_l2", "scaled_inf", "dual_scaled_inf", "dual_inf", "primal_scaled_inf",
                   "primal_inf", "complementarity_scaled_inf", "complementarity_inf")
METRIC_PREFIXES = ("true", "image", "image_minus_true", "ld_sum", "once_cast", "ld_sum_minus_true",
                   "once_cast_minus_true", "once_cast_minus_ld_sum", "image_minus_ld_sum")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(left, right, message):
    require(finite(left) and finite(right) and math.isclose(left, right, rel_tol=5e-13, abs_tol=1e-300), message)


def flags(command):
    require(len(command) >= 3 and (len(command) - 3) % 2 == 0, "求解命令结构")
    result = dict(zip(command[3::2], command[4::2]))
    require(len(result) == (len(command) - 3) // 2, "重复求解参数")
    return result


def parse_events(stderr):
    events, errors = [], []
    for line_number, line in enumerate(stderr.splitlines(), 1):
        if not line.startswith(PREFIX):
            continue
        try:
            events.append(json.loads(line[len(PREFIX):]))
        except Exception as error:
            errors.append({"line": line_number, "error": str(error), "raw": line})
    return events, errors


def verify_sources():
    # 复用已交付且逐字节锁定的 v41 来源检查；不调用其 main、不重写历史证明。
    source_checker = DATA / "v41-verify-archive.py"
    require(sha(source_checker) == SOURCE_CHECKER_SHA, "复用的生产来源核验器身份")
    common = load(source_checker, "v42_checked_source_verifier")
    production = common.verify_sources()
    identity = read(DATA / "v42-newton-identity.json")
    require(identity["executable_sha256"] == DIAGNOSTIC_SHA and identity["plan_sha256"] == PLAN_SHA,
            "正式诊断程序及测量前计划身份")
    require(identity["diagnostic_only"] is True and identity["production_solver_sha256"] ==
            production["solver_sha256"] == PRODUCTION_SHA, "v41生产身份及纯诊断范围")
    require(identity["frozen_files"] == production["files"] and
            identity["frozen_source_zip_sha256"] == production["source_zip_sha256"], "冻结来源与源码包身份")
    require({name: sha(WORK / "frozen" / name) for name in production["files"]} == production["files"], "89份展开副本")
    require(identity["libraries"] == production["libraries"] and
            identity["compiler_sha256"] == production["compiler_sha256"] and identity["compiler"] == production["compiler"],
            "诊断与生产编译器及15份库身份")
    for path, digest in ((WORK / "v42-newton.exe", identity["executable_sha256"]),
                         (WORK / "v42-newton-self-test.exe", identity["self_test_executable_sha256"])):
        require(sha(path) == digest, "诊断程序身份：" + str(path))
    require(Path(identity["executable"]) == WORK / "v42-newton.exe", "诊断程序路径")
    require(set(identity["diagnostic_sources"]) == {"v42-newton-" + name for name in
            ("diagnostic.hpp", "driver.cpp", "instrument.py", "build-run.py", "self-test.cpp")}, "五份诊断来源")
    for name, digest in identity["diagnostic_sources"].items():
        require(sha(DATA / name) == digest, "诊断源身份：" + name)
        if name.endswith((".hpp", ".cpp")):
            require(sha(WORK / name) == digest, "构建使用的诊断副本：" + name)
    for name, key in (("plan.json", "plan_sha256"), ("command.json", "command_sha256"),
                      ("self-test.json", "self_test_result_sha256"), ("instrumented.cpp.gz", "instrumented_archive_sha256")):
        require(sha(DATA / ("v42-newton-" + name)) == identity[key], "诊断附属归档：" + name)
    patched = gzip.decompress((DATA / "v42-newton-instrumented.cpp.gz").read_bytes())
    require(patched == (WORK / "instrumented.cpp").read_bytes() and
            hashlib.sha256(patched).hexdigest() == identity["instrumented_sha256"], "插桩副本与归档")
    original = (WORK / "frozen/src/ipm.cpp").read_text(encoding="utf-8")
    instrument = load(DATA / "v42-newton-instrument.py", "v42_checked_instrument")
    reconstructed = instrument.instrument(original)
    require(reconstructed == patched.decode("utf-8").replace("\r\n", "\n"), "唯一锚点重建插桩")
    differences = difflib.SequenceMatcher(None, original.splitlines(), reconstructed.splitlines(), autojunk=False).get_opcodes()
    require(all(kind in ("equal", "insert") for kind, *_ in differences), "只能追加钩子，不得替换生产算式")
    commands = read(DATA / "v42-newton-command.json")
    for key, source, exe in (("build", "driver.cpp", "v42-newton.exe"),
                             ("self_test_build", "self-test.cpp", "v42-newton-self-test.exe")):
        command = commands[key]
        require(Path(command[0]) == Path(identity["compiler"]) and str(WORK / ("v42-newton-" + source)) in command and
                command[command.index("-o") + 1] == str(WORK / exe), "独立构建入口")
        require(all(str(ROOT / name) in command for name in identity["libraries"]), "构建库身份完整")
        require(all(flag in command for flag in ("-O3", "-DNDEBUG", "-static", "-std=c++20")) and
                ("-municode" in command) is (key == "build"), "完整CLI与解析自测编译参数")
    plan = read(DATA / "v42-newton-plan.json")
    previous = read(DATA / "v41-plan.json")
    expected_models = [{**model, "development": stage == "development"} for stage in ("development", "probe")
                       for model in previous["sets"][stage]["models"]]
    require(plan["models"] == expected_models and len(plan["models"]) == 13, "既有八开发及五已知失败输入")
    require(plan["variants"] == ["production", "diagnostic_off", "diagnostic_on"] and plan["diagnostic_only"] is True,
            "39次诊断协议")
    require((plan["time_limit"], plan["wall_timeout"], plan["iterations"], plan["scaling_passes"], plan["threads"], plan["seed"],
             plan["linear_solver"], plan["accuracy"], plan["direction_retries"]) == (15, 45, 200, 5, 1, 0, "normal_augmented", "full_newton", 0),
            "诊断预算及模式")
    require(plan["shared_ipm"] == previous["shared_ipm"], "其余开关保持v41协议")
    require(plan["analysis_script_sha256"] == identity["analysis_script_sha256"] == sha(DATA / "v42-newton-analyze.py"),
            "分析脚本在测量前固定")
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <= datetime.datetime.fromisoformat(identity["created_at_utc"]),
            "计划先于诊断冻结")
    return common, identity, plan, len([item for item in differences if item[0] == "insert"])


def verify_self_test(identity):
    fixture = read(DATA / "v42-newton-self-test.json")
    require(fixture["command"] == [str(WORK / "v42-newton-self-test.exe")] and fixture["returncode"] == 0,
            "解析自测入口及退出码")
    result = json.loads(fixture["stdout"])
    require(result["passed"] is True and result["long_double_digits"] == 64 and result["checks"] >= 24, "解析自测实际通过")
    events, errors = parse_events(fixture["stderr"])
    require(events == fixture["events"] and errors == fixture["diagnostic_parse_errors"] == [], "自测原始stderr与事件")
    # 用有理数独立计算混合算子及二进制精确的抵消例，不调用诊断数学helper。
    q = [Fraction(-5, 2), Fraction(2), Fraction(6), Fraction(-5), Fraction(10)]
    g = ((1, 2), (3, -1))
    dx, dz, ds = (Fraction(-1), Fraction(2)), (Fraction(1, 2), Fraction(-1)), Fraction(3)
    independent = [sum(Fraction(g[row][col]) * dz[row] for row in range(2)) for col in range(2)]
    independent += [sum(Fraction(g[row][col]) * dx[col] for col in range(2)) + (ds if row == 0 else 0) for row in range(2)]
    independent += [3 * ds + 2 * dz[0]]
    require(independent == q and sum((value / scale) ** 2 for value, scale in zip(q, (Fraction(1, 2), Fraction(1, 2), 2, 2, 5))) == Fraction(241, 4),
            "独立有理数J及weightedL2平方")
    large = 2 ** 53
    sequential = float(float(large + 1) - large)
    exact_sum = Fraction(large) + 1 - large
    require(sequential == 0 and exact_sum == 1 and float(exact_sum) == 1, "独立逐项与一次舍入反例")
    candidate = [event for event in events if event["event"] == "candidate"]
    require(len(candidate) == 1, "开启候选接线实际触发一次")
    candidate = candidate[0]
    require(candidate["phase"] == "self_test" and candidate["true_scaled_inf"] == candidate["production_measure"] == 2
            and candidate["accepted"] is False, "接线例真实拒绝门槛")
    close(candidate["true_weighted_l2"], math.sqrt(8), "接线例真残差加权L2")
    for prefix in ("image", "ld_sum", "once_cast"):
        require(candidate[prefix + "_weighted_l2"] == candidate[prefix + "_scaled_inf"] == 0, "接线例另一候选精确残差")
    for prefix in ("image_minus_true", "ld_sum_minus_true", "once_cast_minus_true"):
        close(candidate[prefix + "_weighted_l2"], math.sqrt(8), "接线例实际差向量")
    require(candidate["once_cast_minus_ld_sum_weighted_l2"] == 0 and
            candidate["current_minus_ld_sum_l2"] == candidate["current_minus_once_cast_l2"] == 1, "未知量差与残差差分开")
    require(all(candidate["diagnostic_" + name] == 2 for name in ("j_started", "j_completed", "csc_started", "csc_completed")) and
            candidate["diagnostic_nnz_visits"] == 2, "接线例恰有两个额外CSC")
    require(candidate["production_preconditioner_started"] == 11 and candidate["production_products_started"] == 13,
            "接线例不写回生产计数")
    range_event = next(event for event in events if event["event"] == "range_test")
    for name in ("overflow", "underflow", "infinite"):
        require(range_event[name] is None and range_event[name + "_valid"] is False, "JSON范围保护")
    require(range_event["zero"] == 0 and range_event["zero_valid"] is True and range_event["finite_fields"] is False,
            "零值和范围失效标记")
    require(events[-1]["event"] == "diagnostic_error" and events[-1]["finite_fields"] is False, "故障关闭诊断事件")
    return {"checks": result["checks"], "events": len(events), "sha256": identity["self_test_result_sha256"],
            "independent_rational_formulas_verified": True, "enabled_candidate_wiring_verified": True,
            "extra_csc_scans": 2, "production_counts_unchanged": True}


def verify_metrics(event, prefix, dimension, tolerance):
    keys = [prefix + "_" + suffix for suffix in METRIC_SUFFIXES]
    if not all(event.get(key + "_valid") is True for key in keys):
        return False
    values = {suffix: event[prefix + "_" + suffix] for suffix in METRIC_SUFFIXES}
    require(all(finite(value) and value >= 0 for value in values.values()), "非负有限范数")
    blocks = [values[name + "_scaled_inf"] for name in ("dual", "primal", "complementarity")]
    close(values["scaled_inf"], max(blocks), "全局无穷范数等于三块最大值")
    for name, scale in zip(("dual", "primal", "complementarity"), tolerance):
        close(values[name + "_scaled_inf"], values[name + "_inf"] / scale, "物理块与缩放块范数关系")
    lower, upper = values["scaled_inf"], math.sqrt(dimension) * values["scaled_inf"]
    require(values["weighted_l2"] >= lower * (1 - 5e-13) and values["weighted_l2"] <= upper * (1 + 5e-13),
            "同权重L2与无穷范数界")
    return True


def verify_events(row):
    events = row["events"]
    if row["variant"] != "diagnostic_on":
        require(not events, "关闭或正式程序不能产生诊断事件")
        return {"events": 0, "directions": 0, "candidates": 0, "extra_csc_scans": 0}
    require(all(type(event.get("event_id")) is int and event["event_id"] == i + 1 for i, event in enumerate(events)), "事件ID连续")
    require(not any(event["event"] == "diagnostic_error" for event in events), "真实运行出现诊断自身错误，需单列审计")
    by_direction = collections.OrderedDict()
    previous_global = {"diagnostic_" + key: 0 for key in EXTRA_FIELDS}
    previous_time = -1
    for event in events:
        require(event["schema"] == 1 and type(event["direction_id"]) is int and event["direction_id"] > 0, "事件schema或方向ID")
        require(event["phase"] in ("predictor", "corrector") and type(event["outer_iteration"]) is int, "方向阶段")
        require(event["elapsed_seconds"] >= previous_time, "事件时间单调")
        previous_time = event["elapsed_seconds"]
        for key, value in event.items():
            if key.endswith("_valid"):
                require(type(value) is bool, "数值有效性必须布尔")
                observed = event[key[:-6]]
                require((value and finite(observed)) or (not value and observed is None), "数值与有效性标记")
            if type(value) is float:
                require(math.isfinite(value), "JSON含非有限浮点")
        for key in previous_global:
            require(type(event[key]) is int and event[key] >= previous_global[key], "额外工作累计计数单调")
            previous_global[key] = event[key]
        by_direction.setdefault(event["direction_id"], []).append(event)
    require(list(by_direction) == list(range(1, len(by_direction) + 1)), "方向ID连续")
    candidate_total = 0
    for direction_id, trace in by_direction.items():
        begin = next((event for event in trace if event["event"] == "direction_begin"), None)
        terminal = trace[-1] if trace[-1]["event"] in ("direction_return", "direction_failure") else None
        require(terminal is not None or row["wall_timeout"], "未超时方向缺失结束事件")
        require(len({(event["outer_iteration"], event["phase"]) for event in trace}) == 1, "同方向上下文一致")
        if begin is None:
            require(terminal is not None and terminal["event"] == "direction_failure", "缺少起始事件必须明确失败")
            continue
        require(trace[0] is begin, "方向起始位置")
        dimension = begin["columns"] + begin["rows"] + begin["inequalities"]
        nonzeros = begin["nonzeros"]
        tolerance = [begin[name + "_tolerance"] for name in ("dual", "primal", "complementarity")]
        require(all(finite(value) and value > 0 for value in tolerance), "固定三块正容差")
        previous = {key: 0 for key in ("observed_preconditioner_started", "observed_preconditioner_completed", "observed_products_started",
                                      "observed_products_completed", "direction_refinement_steps", "direction_krylov_steps",
                                      "candidates_started", "candidates_completed")}
        candidates = []
        starts = []
        for event in trace:
            for key in previous:
                require(type(event[key]) is int and event[key] >= previous[key], "方向局部计数单调")
                previous[key] = event[key]
            require(event["observed_preconditioner_completed"] <= event["observed_preconditioner_started"] and
                    event["observed_products_completed"] <= event["observed_products_started"] and
                    event["direction_krylov_steps"] <= event["direction_refinement_steps"], "局部工作账本范围")
            require(event["candidates_completed"] <= event["candidates_started"], "候选完成次数")
            if "true_weighted_l2" in event:
                verify_metrics(event, "true", dimension, tolerance)
            if event["event"] in ("initial_check", "ir_check", "candidate"):
                close(event["production_measure"], event["true_scaled_inf"], "原验收度量与独立范数")
                # LD边界值经JSON读成binary64可舍入到1；生产布尔值保留为验收权威。
                require(type(event["accepted"]) is bool and
                        (event["production_measure"] <= 1 + 5e-13 if event["accepted"] else event["production_measure"] >= 1 - 5e-13),
                        "验收布尔边界")
            if event["event"] == "arnoldi_start":
                starts.append(event)
            if event["event"] == "candidate":
                candidates.append(event)
                for prefix in METRIC_PREFIXES[1:]:
                    verify_metrics(event, prefix, dimension, tolerance)
                require(event["trigger_estimate"] or event["trigger_window"] or event["trigger_remaining_zero"], "候选必须有原触发原因")
                require(event["trigger_remaining_zero"] is (event["remaining"] == 0), "精确breakdown触发")
        require([event["candidate_id"] for event in candidates] == list(range(1, len(candidates) + 1)), "候选ID连续")
        require(len(starts) == previous["direction_krylov_steps"], "实际Arnoldi启动事件与局部计数")
        require(previous["direction_refinement_steps"] - previous["direction_krylov_steps"] <= 2 and
                previous["direction_krylov_steps"] <= 3 * min(32, begin["columns"] + begin["rows"]), "原IR和Krylov上限")
        candidate_total += len(candidates)
        if terminal is not None:
            if terminal["event"] == "direction_return":
                require(terminal["true_scaled_inf"] <= 1 + 5e-13, "正常返回方向已通过真实门槛")
            require(terminal["candidate_pending"] is (terminal["candidates_started"] > terminal["candidates_completed"]), "中断候选不冒充已完成")
            extra = {key: terminal["diagnostic_" + key] - begin["diagnostic_" + key] for key in EXTRA_FIELDS}
            require(all(extra[key] == 2 * len(candidates) for key in ("j_started", "j_completed", "csc_started", "csc_completed")), "每个候选恰有2次额外CSC")
            require(extra["nnz_visits"] == extra["csc_completed"] * nonzeros, "额外非零访问账本")
            restarts = sum(event["event"] == "restart" for event in trace)
            require(extra["copied_elements"] == dimension * (2 * restarts + extra["cached_images"] + 3 * len(candidates)), "复制及图像缓存元素账本")
            require(extra["reconstruction_terms"] == sum(2 * dimension * (event["column"] + 1) for event in candidates), "图像组合与方向重建项账本")
            checks = sum(event["event"] in ("initial_check", "ir_check") for event in trace)
            require(extra["metric_elements"] == dimension * (checks + restarts + 12 * len(candidates)), "额外范数扫描元素账本")
            for global_key, local_key in (("reported_refinement_steps", "direction_refinement_steps"),
                                          ("reported_krylov_steps", "direction_krylov_steps")):
                require(terminal[global_key] - begin[global_key] == terminal[local_key], "终点已汇总的生产与局部改进账本")
            for global_key, local_key in (("production_preconditioner_started", "observed_preconditioner_started"),
                                          ("production_products_started", "observed_products_started")):
                require(0 <= terminal[global_key] - begin[global_key] <= terminal[local_key], "生产内部启动与整个回调启动关系")
    if events:
        last = events[-1]
        require(last["production_preconditioner_started"] == row["report"]["ipm_full_newton_preconditioner_solves"] and
                last["production_products_started"] == row["report"]["ipm_full_newton_products"], "最终生产计数对应标准报告")
    return {"events": len(events), "directions": len(by_direction), "candidates": candidate_total,
            "extra_csc_scans": previous_global["diagnostic_csc_completed"], "extra_nnz_visits": previous_global["diagnostic_nnz_visits"]}


def verify_runs(identity, plan):
    raw_path = DATA / "v42-newton-runs.jsonl"
    raw = raw_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RAW_SHA, "已完成39条诊断原始身份")
    require(gzip.decompress((DATA / "v42-newton-runs.jsonl.gz").read_bytes()) == raw, "原始/压缩逐字节一致")
    sys.path.insert(0, str(ROOT / "scripts"))
    import benchmark
    rows = [benchmark.strict_json(line) for line in raw.splitlines()]
    expected = [(model, variant) for model in plan["models"] for variant in plan["variants"]]
    require(len(rows) == len(expected) == 39, "39条预定运行完整保留")
    metrics = []
    for row, (model, variant) in zip(rows, expected):
        require(row["input"] == model["input"] and row["model"] == Path(model["input"]).stem and row["variant"] == variant,
                "预定模型及三侧顺序")
        program = ROOT / ".tools/solver-ipm-v41.exe" if variant == "production" else WORK / "v42-newton.exe"
        digest = PRODUCTION_SHA if variant == "production" else identity["executable_sha256"]
        expected_hashes = {"solver": digest, "input": model["sha256"]}
        require(row["hashes_before"] == row["hashes_after"] == expected_hashes and
                sha(Path(row["input"])) == row["model_sha256"] == model["sha256"], "输入和程序前后身份")
        require(row["diagnostic_environment"] == ("1" if variant == "diagnostic_on" else "0"), "诊断开关环境")
        command = row["command"]
        require(Path(command[0]) == program and command[1:3] == ["solve", row["input"]], "完整CLI入口")
        actual = flags(command)
        desired = {"--backend": "ipm", "--time-limit": "15", "--threads": "1", "--seed": "0",
                   "--iteration-limit": "200", "--scaling-passes": "5", "--ipm-linear-accuracy": "full_newton", "--ipm-direction-retries": "0"}
        desired.update({"--ipm-" + key: value for key, value in plan["shared_ipm"].items()})
        require(actual == desired, "诊断全部命令参数")
        require(row["report_parse_error"] is None and benchmark.strict_json(row["stdout"]) == row["report"], "原始标准输出报告")
        report = row["report"]
        require(report["backend_version"] == "ipm-cpu-0.37.0" and report["backend"] == "ipm" and
                benchmark.classify(row["returncode"], report) in ("resolved", "incomplete"), "冻结版本及退出码")
        ipm = report["options"]["ipm"]
        require((report["options"]["time_limit"], report["options"]["threads"], report["options"]["seed"], ipm["max_iterations"],
                 ipm["scaling_passes"], ipm["linear_solver"], ipm["linear_accuracy"], ipm["direction_retries"], ipm["krylov_history_size"]) ==
                (15, 1, 0, 200, 5, "normal_augmented", "full_newton", 0, 0), "报告预算与方向配置")
        events, errors = parse_events(row["stderr"])
        require(events == row["events"] and errors == row["diagnostic_parse_errors"] == [], "原始stderr与完整事件")
        metric = verify_events(row)
        metrics.append({"model": row["model"], "variant": variant, **metric})
    pairs = []
    for model in plan["models"]:
        variants = {row["variant"]: row for row in rows if row["input"] == model["input"]}
        for reference, candidate in (("production", "diagnostic_off"), ("diagnostic_off", "diagnostic_on")):
            before, after = variants[reference], variants[candidate]
            timed = any(row["wall_timeout"] or row["report"]["status"] == "time_limit" for row in (before, after))
            reports = [copy.deepcopy(row["report"]) for row in (before, after)]
            for report in reports:
                report.pop("timings", None)
            equal = reports[0] == reports[1] and before["returncode"] == after["returncode"]
            require(timed or equal, "非超时成对轨迹不一致")
            pairs.append({"model": Path(model["input"]).stem, "reference": reference, "candidate": candidate,
                          "time_limited": timed, "non_time_equal": equal})
    expected_summary = {"diagnostic_only": True, "runs": 39, "raw_sha256": sha(raw_path), "missing_reports": 0,
                        "diagnostic_parse_errors": 0, "events": sum(len(row["events"]) for row in rows), "comparisons": pairs}
    require(read(DATA / "v42-newton-summary.json") == expected_summary, "26比较及完整摘要独立重算")
    return rows, {"records": 39, "events": expected_summary["events"], "comparisons": 26,
                  "time_limited_comparisons": sum(pair["time_limited"] for pair in pairs),
                  "non_time_limited_equal_comparisons": sum(not pair["time_limited"] and pair["non_time_equal"] for pair in pairs),
                  "raw_sha256": sha(raw_path), "groups": metrics,
                  "extra_csc_scans": sum(metric["extra_csc_scans"] for metric in metrics),
                  "candidate_events": sum(metric["candidates"] for metric in metrics)}


def verify_analysis(rows, identity, plan):
    analyzer = load(DATA / "v42-newton-analyze.py", "v42_checked_analyzer")
    result = read(DATA / "v42-newton-analysis.json")
    require(analyzer.ANALYSIS_RULES == plan["analysis_rules"] == result["analysis_rules"] and
            result["script_sha256"] == identity["analysis_script_sha256"] and result["limitations"] == analyzer.LIMITATIONS,
            "测量前分析规则、局限及实现身份")
    require(result["runs_file"] in ("v42-newton-runs.jsonl", "v42-newton-runs.jsonl.gz"), "分析原始文件范围")
    require(result["runs_sha256"] == sha(DATA / result["runs_file"]) and result["uncompressed_runs_sha256"] == RAW_SHA,
            "分析输入与冻结原始记录身份")
    enabled = [row for row in rows if row["variant"] == "diagnostic_on"]
    require(result["records"] == [analyzer.analyze_record(row) for row in enabled], "冻结分析函数重新计算完整事件与派生字段")
    checks = 0
    failed_direction_summaries = []
    for row, analyzed in zip(enabled, result["records"]):
        groups = collections.OrderedDict()
        for position, event in enumerate(row["events"]):
            groups.setdefault(event["direction_id"], []).append((position, event))
        require(len(groups) == analyzed["direction_count"], "独立方向分组")
        failures, returned, triggered = [], [], []
        for direction in analyzed["directions"]:
            positioned = groups[direction["direction_id"]]
            trace = [event for _, event in positioned]
            require(direction["events"] == trace, "全部事件保留")
            terminal = next((event for event in reversed(trace) if event["event"] in ("direction_return", "direction_failure")), None)
            if terminal and terminal["event"] == "direction_failure":
                failures.append(direction)
            if terminal and terminal["event"] == "direction_return":
                returned.append(direction)
            actual_krylov = any(event["event"] == "arnoldi_start" or event.get("direction_krylov_steps", 0) > 0 for event in trace)
            require(direction["triggered_krylov"] is actual_krylov, "首次Arnoldi依据实际工作而非候选出现")
            if actual_krylov:
                triggered.append(direction)
            for check in direction["checks"]:
                event = row["events"][check["event_position"]]
                require(event["event_id"] == check["event_id"], "分析检查点对应原事件")
                for key, value in check["metrics"].items():
                    expected = event.get(key) if event.get(key + "_valid") is True and finite(event.get(key)) else None
                    require(value == expected, "派生指标不替换原值")
                for key, value in check["ratios"].items():
                    numerator, denominator = key.removesuffix("_weighted_l2").split("_over_")
                    num, den = check["metrics"].get(numerator + "_weighted_l2"), check["metrics"].get(denominator + "_weighted_l2")
                    expected = num / den if finite(num) and finite(den) and den != 0 else None
                    if expected is not None and not finite(expected):
                        expected = None
                    require(value == expected, "同weightedL2比值独立重算")
                checks += 1
        failed = failures[-1] if failures else None
        prior = [direction for direction in returned if failed and direction["last_event_position"] < failed["first_event_position"]]
        expected_ids = {"last_failed_or_interrupted": failed["direction_id"] if failed else None,
                        "preceding_returned": prior[-1]["direction_id"] if prior else None,
                        "first_krylov": triggered[0]["direction_id"] if triggered else None}
        require({key: value["direction_id"] for key, value in analyzed["selections"].items()} == expected_ids,
                "三种预声明方向选择独立重算")
        if failed:
            trace = [event for _, event in groups[failed["direction_id"]]]
            candidates = [event for event in trace if event["event"] == "candidate"]
            last = candidates[-1] if candidates else {}
            terminal = trace[-1]
            failed_direction_summaries.append({
                "model": row["model"], "direction_id": failed["direction_id"], "outer_iteration": terminal["outer_iteration"],
                "phase": terminal["phase"], "failure_kind": terminal["failure_kind"],
                "arnoldi_steps": terminal["direction_krylov_steps"], "restarts": sum(event["event"] == "restart" for event in trace),
                "candidate_count": len(candidates),
                "candidate_residual_gate_counts": {
                    "production_accepted": sum(event["accepted"] for event in candidates),
                    **{prefix: sum(event[prefix + "_scaled_inf"] <= 1 for event in candidates
                                   if event[prefix + "_scaled_inf_valid"])
                       for prefix in ("ld_sum", "once_cast")}},
                "last_candidate": {key: last.get(key) for key in ("givens_weighted_l2", "image_weighted_l2", "true_weighted_l2",
                    "ld_sum_weighted_l2", "once_cast_weighted_l2", "true_scaled_inf", "ld_sum_scaled_inf", "once_cast_scaled_inf",
                    "image_minus_true_weighted_l2", "once_cast_minus_ld_sum_weighted_l2")},
            })
    require(result["record_count"] == 39 and result["enabled_record_count"] == 13 and
            result["enabled_event_count"] == sum(len(row["events"]) for row in enabled), "分析总体完整性")
    require(result["direction_count"] == sum(record["direction_count"] for record in result["records"]) and
            result["records_with_issues"] == sum(not record["analysis_complete"] for record in result["records"]), "分析方向和问题总数")
    require(result["non_enabled_event_counts"] == [{"model": row["model"], "variant": row["variant"], "events": len(row["events"])}
            for row in rows if row["variant"] != "diagnostic_on"], "关闭运行的完整事件统计")
    return {"sha256": sha(DATA / "v42-newton-analysis.json"), "script_sha256": identity["analysis_script_sha256"],
            "directions": result["direction_count"], "checks_independently_verified": checks,
            "records_with_issues": result["records_with_issues"], "selection_rules_independently_recomputed": True,
            "last_failed_directions_independently_recomputed": failed_direction_summaries}


def verify_analyzer_self_test(identity):
    path = DATA / "v42-newton-analyze-self-test.py"
    output_path = DATA / "v42-newton-analyze-self-test.json"
    helper = load(path, "v42_checked_analyzer_self_test")
    require(helper.EXPECTED_SHA256 == identity["analysis_script_sha256"], "合成测试锁定冻结分析器")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        helper.main()
    actual = json.loads(captured.getvalue())
    require(actual == read(output_path) == {
        "passed": True, "checks": 15, "analyzer_sha256": identity["analysis_script_sha256"]
    }, "15项纯合成规则检查实际复跑并与归档输出一致")
    return {**actual, "script_sha256": sha(path), "archived_stdout_sha256": sha(output_path),
            "scope": "测量后保存并复跑冻结前的纯合成规则断言；不读取实测，不运行求解。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="数据齐后先核数值与归档，暂不检查文档")
    args = parser.parse_args()
    output = DATA / ("v42-newton-verification-preflight.json" if args.preflight else "v42-newton-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False, "final": not args.preflight,
              "verification_script_sha256": sha(Path(__file__)), "reused_source_checker_sha256": SOURCE_CHECKER_SHA}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        common, identity, plan, insertions = verify_sources()
        fixture = verify_self_test(identity)
        rows, runs = verify_runs(identity, plan)
        analysis = verify_analysis(rows, identity, plan)
        analyzer_self_test = verify_analyzer_self_test(identity)
        result.update(production_solver_sha256=PRODUCTION_SHA, diagnostic_solver_sha256=identity["executable_sha256"],
                      plan_sha256=PLAN_SHA, diagnostic_sources=identity["diagnostic_sources"],
                      production_source_files=89, diagnostic_source_files=5, libraries=15, insertion_blocks=insertions,
                      self_test=fixture, analyzer_self_test=analyzer_self_test, runs=runs, analysis=analysis,
                      runtime_scope="读取既有归档并复跑15项纯合成分析规则断言，不编译、不运行求解。公式用有理数独立核对；实际事件只存标量，不能据此重建全部向量或证明不可表示。",
                      source_scope="复用逐字节锁定的v41核验器检查生产身份，不运行其main或改写历史证明；本轮独立检查诊断源码、插桩、事件、账本和选择。")
        if not args.preflight:
            result.update(common.verify_documents(output))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "final": result["final"], "records": runs["records"],
                      "events": runs["events"], "extra_csc_scans": runs["extra_csc_scans"]}))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    main()
