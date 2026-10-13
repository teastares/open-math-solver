"""独立核验 v40 只读进展诊断；不编译、不求解、不重新采集性能。"""
import argparse
import collections
import copy
import datetime
from decimal import Decimal, localcontext
import difflib
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v40-progress"
sys.dont_write_bytecode = True
PRODUCTION_SHA = "75c20fc0b39f31125f745540fc2a46f218ccf28c713cb30c341b4705b86ac7bf"
DIAGNOSTIC_SHA = "617621a44e1b4f1bd1423efa035bb1e880e890b21fc93c6008a02a8d38759079"
CONFIGS = [{"accuracy": "strict", "retries": 0},
           {"accuracy": "curvature_regularization", "retries": 8},
           {"accuracy": "blockwise", "retries": 8}]
WORK_FIELDS = ("completed_iterations", "factorizations", "refinement_steps", "direction_retries",
               "direction_retry_successes", "schur_residual_products", "schur_refinement_steps")


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def flags(command):
    require(len(command) >= 3 and (len(command) - 3) % 2 == 0, "solve 命令结构")
    result = dict(zip(command[3::2], command[4::2]))
    require(len(result) == (len(command) - 3) // 2, "solve 命令不能重复选项")
    return result


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=2e-14, abs_tol=1e-300), name)


def parse_stderr(stderr):
    events, others, errors = [], [], []
    for line in stderr.splitlines():
        if not line.startswith("V40_PROGRESS "):
            others.append(line)
            continue
        try:
            events.append(json.loads(line[len("V40_PROGRESS "):]))
        except json.JSONDecodeError as error:
            errors.append({"line": line, "error": str(error)})
    return events, others, errors


def verify_identity():
    identity = read(DATA / "v40-progress-identity.json")
    old = read(DATA / "ipm-v39-identity.json")
    require(identity["diagnostic_only"] is True and identity["production_solver_sha256"] ==
            old["solver_sha256"] == PRODUCTION_SHA, "正式生产版本身份")
    require(identity["executable_sha256"] == DIAGNOSTIC_SHA and
            Path(identity["executable"]) == WORK / "v40-progress.exe", "诊断程序身份")
    require(identity["frozen_files"] == old["files"] and len(old["files"]) == 88, "88份冻结来源")
    archive_path = DATA / "ipm-v39-source.zip"
    require(sha(archive_path) == identity["frozen_source_zip_sha256"], "冻结源码包身份")
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) == 88, "源码包条目")
        for name, digest in old["files"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest and
                    sha(ROOT / name) == digest and sha(WORK / "frozen" / name) == digest,
                    "生产/展开副本/包内容：" + name)
        original = archive.read("src/ipm.cpp").decode("utf-8").replace("\r\n", "\n")
    for path, digest in ((ROOT / ".tools/solver-ipm-v39.exe", PRODUCTION_SHA),
                         (ROOT / "build/release/solver.exe", PRODUCTION_SHA),
                         (WORK / "v40-progress.exe", DIAGNOSTIC_SHA),
                         (WORK / "v40-progress-self-test.exe", identity["self_test_executable_sha256"]),
                         (Path(identity["compiler"]), identity["compiler_sha256"])):
        require(sha(path) == digest, "当前程序或编译器：" + str(path))
    for name, digest in identity["libraries"].items():
        require(sha(ROOT / name) == digest, "链接库身份：" + name)
    require(len(identity["libraries"]) == 15 and len(identity["diagnostic_sources"]) == 5, "诊断构建来源数量")
    for name, digest in identity["diagnostic_sources"].items():
        require(sha(DATA / name) == digest, "冻结诊断源：" + name)
        if name.endswith((".cpp", ".hpp")):
            require(sha(WORK / name) == digest, "构建诊断源副本：" + name)
    for name, key in (("v40-progress-plan.json", "plan_sha256"),
                      ("v40-progress-command.json", "command_sha256"),
                      ("v40-progress-self-test.json", "self_test_result_sha256"),
                      ("v40-progress-instrumented.cpp.gz", "instrumented_archive_sha256")):
        require(sha(DATA / name) == identity[key], "构建附属归档：" + name)
    patched = gzip.decompress((DATA / "v40-progress-instrumented.cpp.gz").read_bytes())
    require(patched == (WORK / "instrumented.cpp").read_bytes() and
            hashlib.sha256(patched).hexdigest() == identity["instrumented_sha256"], "插桩完整源码归档")
    instrument = module(DATA / "v40-progress-instrument.py", "v40_checked_instrument")
    reconstructed = instrument.instrument(original)
    require(reconstructed == patched.decode("utf-8").replace("\r\n", "\n"), "从冻结源码重建插桩结果")
    changes = difflib.SequenceMatcher(None, original.splitlines(), reconstructed.splitlines(), autojunk=False)
    require(all(tag in ("equal", "insert") for tag, *_ in changes.get_opcodes()), "插桩不得改写或删除原算式")
    commands = read(DATA / "v40-progress-command.json")
    for key, source, exe in (("build", "v40-progress-driver.cpp", "v40-progress.exe"),
                             ("self_test_build", "v40-progress-self-test.cpp", "v40-progress-self-test.exe")):
        command = commands[key]
        require(Path(command[0]) == Path(identity["compiler"]) and str(WORK / source) in command and
                command[command.index("-o") + 1] == str(WORK / exe), "诊断构建入口")
        require(all(str(ROOT / name) in command for name in identity["libraries"]), "构建所用库必须全部有身份")
        require(all(flag in command for flag in ("-O3", "-DNDEBUG", "-static", "-std=c++20")), "诊断构建选项")
        require(("-municode" in command) is (key == "build"), "完整 Windows CLI 与独立自检入口")
    return identity


def verify_fields(event):
    require(finite_tree(event), "事件不能含 JSON 消费端非有限数")
    for name, value in event.items():
        if name.endswith("_valid"):
            field = name[:-6]
            require(type(value) is bool and field in event, "有效标记必须对应字段")
            if value:
                require(type(event[field]) in (int, float) and math.isfinite(event[field]), "有效字段必须为有限数")
            else:
                require(event[field] is None, "无效字段必须显式为null")
        elif type(value) in (int, float) and name not in ("schema", "event_id", "iteration"):
            require(event.get(name + "_valid") is True, "数值字段必须有有效标记")
    require(type(event.get("finite_fields")) is bool, "事件整体数值标记")


def verify_self_test():
    data = read(DATA / "v40-progress-self-test.json")
    events, others, errors = parse_stderr(data["stderr"])
    require(data["command"] == [str(WORK / "v40-progress-self-test.exe")] and data["returncode"] == 0,
            "自检命令与退出码")
    require(events == data["events"] and not others and not errors and len(events) == 5,
            "自检原始stderr完整解析")
    require(json.loads(data["stdout"]) == {"disabled_no_scans": True}, "关闭事件与扫描的自检")
    for event in events:
        verify_fields(event)
    require([event["event_id"] for event in events] == list(range(1, 6)), "自检事件编号")
    by_name = {event["event"]: event for event in events}
    require(set(by_name) == {"run_start", "begin", "before_step", "after_step", "range_test"}, "自检阶段")
    # 用精确有理数独立重算固定解析构造，不采用归档expected作为数值真值。
    matrix = [[F(1), F(2)], [F(3), F(-1)]]
    costs, rhs = [F(1), F(-1)], [F(5), F(0)]
    x, slack, dual = [F(4), F(1)], [F(2), F(0)], [F(3), F(-2)]
    dx, ds, dz = [F(-1), F(2)], [F(3), F(0)], [F(1, 2), F(-1)]
    ap, ad = F(1, 4), F(1, 2)
    multiply = lambda point: [sum(a * b for a, b in zip(row, point)) for row in matrix]
    transpose = lambda point: [sum(matrix[row][col] * point[row] for row in range(2)) for col in range(2)]
    dot = lambda left, right: sum(a * b for a, b in zip(left, right))
    norm = lambda point: max(abs(item) for item in point)
    def point_fields(px, ps, pz):
        rp = [a + b - c for a, b, c in zip(multiply(px), ps, rhs)]
        rd = [a + b for a, b in zip(transpose(pz), costs)]
        raw = dot(costs, px) + dot(rhs, pz)
        dt, pt, comp = dot(px, rd), -dot(pz, rp), dot(ps, pz)
        return rp, rd, {"rp_ld_inf": norm(rp), "rd_ld_inf": norm(rd), "working_gap_raw": raw,
                       "working_gap_dual_term": dt, "working_gap_primal_term": pt,
                       "working_gap_complementarity": comp, "working_gap_identity_error": raw - dt - pt - comp}
    rp, rd, begin = point_fields(x, slack, dual)
    ep = [a + b + c for a, b, c in zip(rp, multiply(dx), ds)]
    ed = [a + b for a, b in zip(rd, transpose(dz))]
    predicted_p = [(1 - ap) * a + ap * b for a, b in zip(rp, ep)]
    predicted_d = [(1 - ad) * a + ad * b for a, b in zip(rd, ed)]
    next_x, next_s, next_z = ([a + alpha * b for a, b in zip(point, delta)]
                             for point, delta, alpha in ((x, dx, ap), (slack, ds, ap), (dual, dz, ad)))
    actual_p, actual_d, after = point_fields(next_x, next_s, next_z)
    expected = {"begin": begin, "before_step": {"ep_ld_inf": norm(ep), "ed_ld_inf": norm(ed),
                "h_complementarity_defect_inf": abs(dual[0] * ds[0] + slack[0] * dz[0] - 1),
                "predicted_rp_ld_inf": norm(predicted_p), "predicted_rd_ld_inf": norm(predicted_d)},
                "after_step": {**after, "rp_update_prediction_drift": norm([a-b for a,b in zip(actual_p,predicted_p)]),
                               "rd_update_prediction_drift": norm([a-b for a,b in zip(actual_d,predicted_d)])}}
    for phase, fields in expected.items():
        for name, value in fields.items():
            require(by_name[phase][name] == float(value) and by_name[phase][name + "_valid"] is True,
                    "有理数自检恒等式：" + phase + "/" + name)
    for phase, scans, products in (("begin", 1, 2), ("before_step", 2, 5), ("after_step", 3, 7)):
        require(by_name[phase]["diagnostic_csc_scans"] == scans and
                by_name[phase]["diagnostic_vector_products"] == products, "自检额外工作量")
    for field in ("overflow", "underflow", "infinite"):
        require(by_name["range_test"][field] is None and by_name["range_test"][field + "_valid"] is False,
                "跨语言数值范围保护")
    require(by_name["range_test"]["finite_fields"] is False and by_name["run_start"]["long_double_digits"] > 53,
            "长双精度与无效事件标记")
    return {"events": 5, "exact_rational_formula_checks": sum(map(len, expected.values())),
            "disabled_no_scans_or_events": True, "range_guards": 3, "sha256": sha(DATA / "v40-progress-self-test.json"),
            "limitation": "已冻结runner在解析自检stderr后才保存；本次成功且原始输出完整，未修改已测来源，异常归档顺序留待后续模板改进。"}


def verify_events(events, report, accuracy):
    phase, iteration, scans, products = "initial", 0, 0, 0
    work, clocks = {}, {}
    counts = collections.Counter()
    invalid, current_checks, identities, kkt_over_budget = 0, 0, 0, 0
    for index, event in enumerate(events):
        require(event["schema"] == 1, "事件协议版本")
        name = event["event"]
        counts[name] += 1
        verify_fields(event)
        if name == "diagnostic_error":
            require(index == len(events) - 1 and event["finite_fields"] is False, "诊断异常之后必须停止观测")
            continue
        require(event["event_id"] == index + 1, "事件编号必须连续")
        if name == "run_start":
            require(index == 0 and event["iteration"] == 0 and event["linear_accuracy"] == accuracy
                    and event["time_limit"] == 10 and event["long_double_digits"] > 53, "诊断启动配置")
        elif name == "begin":
            iteration += 1
            phase = "begin"
            scans += 1
            products += 2
            require(event["completed_iterations"] == iteration - 1, "begin不能冒充已完成迭代")
        elif name == "before_step":
            require(phase == "begin", "方向事件必须位于begin之后")
            phase = "before_step"
            scans += 1
            products += 3
            require(event["completed_iterations"] == iteration - 1, "before_step完成步数")
            require(event["strict_block_tolerances_active"] is (accuracy == "blockwise")
                    and event["h_matches_direction"] is True, "方向右端与验收语义")
            for field in ("alpha_primal", "alpha_dual", "sigma"):
                if event[field + "_valid"]:
                    require(0 <= event[field] <= 1, "步长和中心参数范围")
            if event["kkt_rhs_upper_inf_valid"] and event["kkt_rhs_lower_inf_valid"]:
                upper, lower = event["kkt_rhs_upper_inf"], event["kkt_rhs_lower_inf"]
                close(event["kkt_global_tolerance"], 1e-12 + 1e-10 * max(upper, lower), "原KKT全局门槛")
                close(event["kkt_strict_upper_tolerance"], 1e-12 + 1e-10 * upper, "上块严格门槛")
                close(event["kkt_strict_lower_tolerance"], 1e-12 + 1e-10 * lower, "下块严格门槛")
                for block in ("upper", "lower"):
                    if event["kkt_" + block + "_ld_inf_valid"]:
                        allowed = event["kkt_global_tolerance"]
                        if accuracy == "blockwise":
                            allowed = min(allowed, event["kkt_strict_" + block + "_tolerance"])
                        kkt_over_budget += event["kkt_" + block + "_ld_inf"] > allowed * (1 + 1e-10)
        elif name == "after_step":
            require(phase == "before_step", "更新后事件必须有同轮方向")
            phase = "after_step"
            scans += 1
            products += 2
            require(event["completed_iterations"] == iteration, "after_step完成步数")
        elif name == "current_assess":
            current_checks += 1
            for field in ("original_valid_input", "original_primal_feasible", "original_dual_feasible",
                          "original_gap_satisfied", "original_numerically_optimal"):
                require(type(event[field]) is bool, "当前原模型验算布尔字段")
            require(not event["original_numerically_optimal"] or all(event[field] for field in
                    ("original_valid_input", "original_primal_feasible", "original_dual_feasible", "original_gap_satisfied")),
                    "当前原模型最优必须满足各独立条件")
        else:
            raise AssertionError("未知诊断事件：" + str(name))
        require(event["iteration"] == iteration and event["phase"] == phase, "事件迭代及最近阶段")
        require(event["diagnostic_csc_scans"] == scans and event["diagnostic_vector_products"] == products,
                "额外CSC扫描和逻辑乘积账本")
        for field in WORK_FIELDS:
            if event.get(field + "_valid"):
                value = event[field]
                require(type(value) is int and value >= work.get(field, 0), "工作计数类型与单调性：" + field)
                work[field] = value
        for field in ("elapsed_seconds", "diagnostic_seconds_completed_before_event"):
            if event[field + "_valid"]:
                require(event[field] >= clocks.get(field, 0), "诊断时钟单调性")
                clocks[field] = event[field]
        require(event["event_compute_seconds"] is None or event["event_compute_seconds"] >= 0, "诊断事件耗时")
        if name in ("begin", "after_step"):
            gap_fields = ("working_gap_raw", "working_gap_dual_term", "working_gap_primal_term",
                          "working_gap_complementarity", "working_gap_identity_error")
            if all(event[field + "_valid"] for field in gap_fields):
                # 只核标量恒等式与JSON舍入的一致性，不假称从摘要恢复完整点或稀疏乘积。
                with localcontext() as context:
                    context.prec = 80
                    raw, dt, pt, comp, error = (Decimal(str(event[field])) for field in gap_fields)
                    discrepancy = abs(raw - dt - pt - comp - error)
                    scale = max(Decimal(1), abs(raw) + abs(dt) + abs(pt) + abs(comp))
                    require(discrepancy <= Decimal(str(16 * sys.float_info.epsilon)) * scale,
                            "工作间隙四项与所记录LD恒等式误差")
                identities += 1
        invalid += event["finite_fields"] is False
    if report:
        mapping = {"completed_iterations": "ipm_iterations", **{field: "ipm_" + field for field in WORK_FIELDS[1:]}}
        for field, value in work.items():
            require(report[mapping[field]] >= value, "事件计数不能超过最终报告")
    return {"event_counts": dict(counts), "events": len(events), "diagnostic_csc_scans": scans,
            "diagnostic_vector_products": products, "invalid_numeric_events": invalid,
            "current_assess_events": current_checks, "scalar_gap_identity_checks": identities,
            "recorded_kkt_blocks_over_active_tolerance": kkt_over_budget}


def verify_plan(compare):
    plan = read(DATA / "v40-progress-plan.json")
    development = compare.load_models(ROOT / "benchmarks/development.txt")[0]
    netlib = compare.load_models(ROOT / "benchmarks/netlib-all.txt")[0]
    extras = {"dfl001", "finnis", "greenbea", "bnl2", "grow22"}
    models = [{**item, "development": True} for item in development]
    models += [{**item, "development": False} for item in netlib if Path(item["input"]).stem in extras]
    require(plan["models"] == models and len(models) == 13 and len(development) == 8, "预声明13例及当前输入身份")
    require(plan["diagnostic_only"] is True and plan["configurations"] == CONFIGS, "三配置诊断协议")
    for field, value in {"time_limit": 10, "wall_timeout": 45, "iterations": 200,
                          "linear_solver": "normal_augmented", "history": 0, "threads": 1, "seed": 0,
                          "singleton_elimination": True, "fixed_variable_elimination": True,
                          "kkt_equilibration": True, "singleton_presolve": False, "presolve_fixed_elimination": False}.items():
        require(plan[field] == value, "预声明共同配置：" + field)
    return plan


def verify_runs(identity, plan, benchmark):
    path = DATA / "v40-progress-runs.jsonl"
    raw = path.read_bytes()
    require(gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes()) == raw, "原始102条压缩归档字节")
    rows = [json.loads(line) for line in raw.splitlines()]
    schedule = [(config, model, variant) for config in CONFIGS for model in plan["models"]
                for variant in (["diagnostic_off", "diagnostic_on", "production"] if model["development"]
                                else ["diagnostic_off", "diagnostic_on"])]
    require(len(rows) == len(schedule) == 102, "诊断102条完整记录")
    event_metrics, groups = [], {}
    for row, (config, model, variant) in zip(rows, schedule):
        require(row["configuration"] == config and row["variant"] == variant and row["input"] == model["input"]
                and row["development"] is model["development"] and row["model"] == Path(model["input"]).stem,
                "预声明顺序/模型/配置")
        require(row["input_sha256"] == model["sha256"] == sha(Path(model["input"])), "输入归档及当前身份")
        exe = ROOT / ".tools/solver-ipm-v39.exe" if variant == "production" else WORK / "v40-progress.exe"
        digest = PRODUCTION_SHA if variant == "production" else DIAGNOSTIC_SHA
        require(row["solver_sha256"] == digest and Path(row["command"][0]) == exe and
                row["command"][1:3] == ["solve", model["input"]], "逐条诊断或生产程序身份")
        enabled = "1" if variant == "diagnostic_on" else "0"
        require(row["diagnostic_environment"] == {"SOLVER_V40_PROGRESS": enabled}, "诊断开关环境")
        expected_flags = {"--backend": "ipm", "--time-limit": "10", "--iteration-limit": "200",
                          "--threads": "1", "--seed": "0", "--ipm-linear-solver": "normal_augmented",
                          "--ipm-linear-accuracy": config["accuracy"], "--ipm-direction-retries": str(config["retries"]),
                          "--ipm-singleton-elimination": "on", "--ipm-fixed-variable-elimination": "on",
                          "--ipm-kkt-equilibration": "on", "--ipm-singleton-presolve": "off",
                          "--ipm-presolve-fixed-elimination": "off", "--ipm-krylov-history-size": "0"}
        require(flags(row["command"]) == expected_flags, "完整CLI诊断参数")
        try:
            report = json.loads(row["stdout"])
        except json.JSONDecodeError:
            report = None
        require(report == row["report"] and finite_tree(report), "原始stdout与最终报告")
        if report:
            require(report["backend"] == "ipm" and report["backend_version"] == "ipm-cpu-0.36.0", "原CLI版本")
            options, ipm = report["options"], report["options"]["ipm"]
            require(options["time_limit"] == 10 and options["threads"] == 1 and options["seed"] == 0,
                    "报告资源预算")
            expected_ipm = {"max_iterations": 200, "scaling_passes": 5, "eliminate_singletons": True,
                            "dynamic_regularization": False, "equilibrate_kkt": True,
                            "linear_accuracy": config["accuracy"], "central_corrections": 0,
                            "direction_retries": config["retries"], "krylov_history_size": 0,
                            "linear_solver": "normal_augmented", "eliminate_fixed_variables": True,
                            "presolve_singleton_rows": False, "eliminate_presolved_fixed_variables": False,
                            "inexact_prefilter": False}
            require(ipm == expected_ipm, "报告全部IPM配置")
            for suffix in ("projections", "acceptances", "vectors", "peak_dimension"):
                value = report["ipm_krylov_recycle_" + suffix]
                require(type(value) is int and value == 0, "回收关闭四计数")
            if not row["wall_timeout"]:
                require(benchmark.classify(row["returncode"], report) in ("resolved", "incomplete"), "报告与退出码")
        require(type(row["wall_timeout"]) is bool and
                ((row["returncode"] is None) if row["wall_timeout"] else type(row["returncode"]) is int), "墙钟中断记录")
        events, others, errors = parse_stderr(row["stderr"])
        require(events == row["events"] and others == row["other_stderr_lines"] and
                errors == row["diagnostic_parse_errors"], "原始stderr事件/截断/其他输出归档")
        if variant == "diagnostic_on":
            require(bool(events) or row["wall_timeout"], "开启诊断必须有事件或明确墙钟中断")
            event_metrics.append({"model": row["model"], "accuracy": config["accuracy"],
                                  **verify_events(events, report, config["accuracy"])})
        else:
            require(not events and not errors, "关闭观测/生产程序不得发出诊断事件")
        groups.setdefault((config["accuracy"], model["input"]), {})[variant] = row
    require(len(groups) == 39 and len(event_metrics) == 39, "39组诊断开关对照")
    comparisons, statuses = [], {}
    for (accuracy, input_path), group in groups.items():
        for variant, row in group.items():
            status = (row["report"] or {}).get("status", "missing_report")
            counter = statuses.setdefault(accuracy, {}).setdefault(variant, {})
            counter[status] = counter.get(status, 0) + 1
        for reference, candidate in (("diagnostic_off", "diagnostic_on"), ("production", "diagnostic_off")):
            if reference not in group:
                continue
            left, right = group[reference], group[candidate]
            limited = any(row["wall_timeout"] or (row["report"] or {}).get("status") == "time_limit" for row in (left, right))
            normalized = []
            for row in (left, right):
                report = copy.deepcopy(row["report"])
                if report is not None:
                    report.pop("timings", None)
                normalized.append(report)
            equal = left["returncode"] == right["returncode"] and normalized[0] is not None and normalized[0] == normalized[1]
            comparisons.append({"accuracy": accuracy, "model": Path(input_path).stem,
                                "reference": reference, "candidate": candidate, "time_limited": limited,
                                "non_time_equal": equal, "reference_status": (left["report"] or {}).get("status"),
                                "candidate_status": (right["report"] or {}).get("status")})
    summary = {"diagnostic_only": True, "raw_records": len(rows), "groups": len(groups),
               "comparisons": comparisons, "status_counts": statuses,
               "non_time_mismatches_excluding_time_limits": [item for item in comparisons if not item["time_limited"] and not item["non_time_equal"]],
               "missing_reports": sum(row["report"] is None for row in rows),
               "diagnostic_parse_errors": sum(len(row["diagnostic_parse_errors"]) for row in rows),
               "events": sum(len(row["events"]) for row in rows), "runs_sha256": sha(path)}
    require(summary == read(DATA / "v40-progress-summary.json"), "完整102条摘要独立重算")
    require(len(comparisons) == 63 and not summary["non_time_mismatches_excluding_time_limits"],
            "非超时开关/独立编译轨迹应相等；超时截断另列")
    return {"raw_records": len(rows), "raw_sha256": sha(path), "compressed_sha256": sha(path.with_suffix(".jsonl.gz")),
            "summary_sha256": sha(DATA / "v40-progress-summary.json"), "summary_recomputed": True,
            "on_off_pairs": 39, "production_off_pairs": 24, "non_time_comparisons_equal": sum(not item["time_limited"] for item in comparisons),
            "time_limited_comparisons": [item for item in comparisons if item["time_limited"]],
            "missing_reports": summary["missing_reports"], "diagnostic_parse_errors": summary["diagnostic_parse_errors"],
            "events": summary["events"], "event_audits": event_metrics,
            "hash_scope": "核对冻结身份、当前文件及每条已记录身份；运行器前后assert成功不等同于另有逐次前后hash字段。"}


def verify_analysis():
    path = DATA / "v40-progress-analysis.json"
    result = read(path)
    script = DATA / "v40-progress-analyze.py"
    require(result["script_sha256"] == sha(script) and
            result["runs_sha256"] == sha(DATA / "v40-progress-runs.jsonl"), "事后分析来源身份")
    schema = module(script, "v40_checked_analysis_schema")
    anchors = [1, 10, 25, 40, 60, 100, 150]
    intervals = [[1, 10], [11, 24], [25, 39], [40, 59], [60, 99], [100, 149], [150, 200]]
    require(result["fixed_anchors"] == anchors and result["fixed_intervals"] == intervals and
            result["schema"] == 1 and result["diagnostic_only"] is True, "统一分析位置与区间")
    require(any("事后分析" in line and "不是预注册" in line for line in result["limitations"]), "事后分析限制不能遗漏")
    rows = [json.loads(line) for line in (DATA / "v40-progress-runs.jsonl").read_bytes().splitlines()]
    selected = [row for row in rows if row["variant"] == "diagnostic_on"]
    require((result["record_count"], result["enabled_record_count"], result["event_count"]) ==
            (len(rows), len(selected), sum(len(row["events"]) for row in rows)), "分析覆盖数量")
    require(len(result["records"]) == len(selected) == 39, "逐条事后分析覆盖")
    def project(event, kind):
        if event is None:
            return None
        names = ("event_id", "iteration", "completed_iterations", "phase") + schema.FIELDS[kind]
        return {name: event.get(name) for name in names}
    def derived(events):
        begin, step, after = (events.get(name, {}) for name in ("begin", "before_step", "after_step"))
        values = {name: step.get(name) for name in ("alpha_primal", "alpha_dual", "ep_algorithm_inf", "ed_algorithm_inf",
                  "ep_ld_inf", "ed_ld_inf", "kkt_global_tolerance", "kkt_upper_ld_inf", "kkt_lower_ld_inf",
                  "h_complementarity_defect_inf")}
        for prefix, source, names in (
            ("begin", begin, ("rp_ld_inf", "rd_ld_inf", "mu_algorithm", "rp_algorithm_ld_difference", "rd_algorithm_ld_difference")),
            ("after", after, ("rp_ld_inf", "rd_ld_inf", "mu_ld", "rp_update_prediction_drift", "rd_update_prediction_drift",
                              "working_gap_raw", "working_gap_dual_term", "working_gap_primal_term", "working_gap_complementarity"))):
            values.update({prefix + "_" + name: source.get(name) for name in names})
        pairs = {
            "ep_algorithm_over_rp_algorithm": (step.get("ep_algorithm_inf"), begin.get("rp_algorithm_inf")),
            "ed_algorithm_over_rd_algorithm": (step.get("ed_algorithm_inf"), begin.get("rd_algorithm_inf")),
            "ep_ld_over_rp_ld": (step.get("ep_ld_inf"), begin.get("rp_ld_inf")),
            "ed_ld_over_rd_ld": (step.get("ed_ld_inf"), begin.get("rd_ld_inf")),
            "kkt_global_tolerance_over_rd_algorithm": (step.get("kkt_global_tolerance"), begin.get("rd_algorithm_inf")),
            "kkt_upper_over_global_tolerance": (step.get("kkt_upper_ld_inf"), step.get("kkt_global_tolerance")),
            "kkt_upper_over_strict_block_tolerance": (step.get("kkt_upper_ld_inf"), step.get("kkt_strict_upper_tolerance")),
            "h_defect_over_h": (step.get("h_complementarity_defect_inf"), step.get("h_inf")),
            "actual_rp_contraction": (after.get("rp_ld_inf"), begin.get("rp_ld_inf")),
            "actual_rd_contraction": (after.get("rd_ld_inf"), begin.get("rd_ld_inf")),
            "rp_algorithm_difference_over_rp_ld": (begin.get("rp_algorithm_ld_difference"), begin.get("rp_ld_inf")),
            "rd_algorithm_difference_over_rd_ld": (begin.get("rd_algorithm_ld_difference"), begin.get("rd_ld_inf")),
            "rp_update_drift_over_actual_rp": (after.get("rp_update_prediction_drift"), after.get("rp_ld_inf")),
            "rd_update_drift_over_actual_rd": (after.get("rd_update_prediction_drift"), after.get("rd_ld_inf")),
        }
        for name, (numerator, denominator) in pairs.items():
            value = None if numerator is None or denominator is None or denominator == 0 else numerator / denominator
            values[name] = value if value is not None and math.isfinite(value) else None
        for name in ("factorizations", "refinement_steps", "direction_retries"):
            start, end = begin.get(name), step.get(name)
            values["step_" + name] = None if start is None or end is None else end - start
        return values
    total_anchors, total_intervals, complete_count = 0, 0, 0
    for row, item in zip(selected, result["records"]):
        grouped = {}
        for event in row["events"]:
            if event["event"] in schema.FIELDS:
                grouped.setdefault(event["iteration"], {})[event["event"]] = event
        complete = {number: stages for number, stages in grouped.items() if number > 0 and
                    all(name in stages for name in ("begin", "before_step", "after_step"))}
        numbers = sorted(complete)
        final_assess = next((event for event in reversed(row["events"]) if event["event"] == "current_assess"), None)
        report = row["report"]
        expected = {"model": row["model"], "configuration": row["configuration"], "development": row["development"],
                    "status": report["status"], "message": report["message"], "iterations": report["ipm_iterations"],
                    "factorizations": report["ipm_factorizations"], "refinement_steps": report["ipm_refinement_steps"],
                    "direction_retries": report["ipm_direction_retries"],
                    "retained_best_report": {name: report[name] for name in ("has_solution", "objective", "verification", "lp_optimality")},
                    "last_current_assess": project(final_assess, "current_assess"), "event_count": len(row["events"]),
                    "complete_step_count": len(numbers),
                    "incomplete_attempt_iterations": [number for number, stages in grouped.items()
                                                      if number > 0 and "begin" in stages and number not in complete],
                    "nonfinite_events": [{"event_id": event.get("event_id"), "iteration": event.get("iteration"), "event": event["event"]}
                                         for event in row["events"] if event.get("finite_fields") is False],
                    "anchors": [], "intervals": []}
        wanted = set(anchors) | ({numbers[-1]} if numbers else set())
        for number in sorted(wanted & complete.keys()):
            stages = complete[number]
            expected["anchors"].append({"iteration": number, "last_completed": number == numbers[-1],
                                        **{kind: project(stages.get(kind), kind) for kind in schema.FIELDS},
                                        "derived": derived(stages)})
        for lower, upper in intervals:
            values = [derived(complete[number]) for number in numbers if lower <= number <= upper]
            if not values:
                continue
            statistics_by_name = {}
            for name in values[0]:
                samples = [value[name] for value in values if value[name] is not None]
                if samples:
                    statistics_by_name[name] = {"count": len(samples), "minimum": min(samples),
                                                "median": statistics.median(samples), "maximum": max(samples)}
            expected["intervals"].append({"requested_interval": [lower, upper], "complete_steps": len(values),
                                           "metrics": statistics_by_name})
        require(item == expected, "事后锚点/比值/完整区间独立复算：" + row["model"] + "/" + row["configuration"]["accuracy"])
        total_anchors += len(expected["anchors"])
        total_intervals += len(expected["intervals"])
        complete_count += len(numbers)
    return {"records": len(selected), "anchors": total_anchors, "nonempty_intervals": total_intervals,
            "complete_steps": complete_count, "analysis_sha256": sha(path), "analysis_script_sha256": sha(script),
            "independent_numeric_recomputation": True, "current_and_retained_separate": True,
            "scope": "采用已核对分析脚本的字段名单作投影schema；锚点选择、比值、区间统计及当前/保留报告另行从原始事件重算。统一事后分析不是预注册或因果实验。"}


def verify_documents(output):
    documents = list(ROOT.glob("*.md"))
    for name in ("knowledge", "changelog", "benchmarks"):
        documents.extend((ROOT / name).rglob("*.md"))
    count, self_links, broken = 0, 0, []
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            count += 1
            local = (path.parent / unquote(target.split("#")[0].strip("<>"))).resolve()
            self_links += local == output.resolve()
            if not local.exists():
                broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    require(not broken and self_links >= 2, "本地链接或最终报告索引：" + str(broken))
    return {"markdown_files_checked": len(documents), "local_links_checked": count,
            "verification_report_links": self_links, "broken_links": broken}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="数据预检，文档完成后另运行最终核验")
    args = parser.parse_args()
    output = DATA / ("v40-progress-verification-preflight.json" if args.preflight else "v40-progress-verification.json")
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": False,
              "final": not args.preflight, "diagnostic_only": True, "verification_script_sha256": sha(Path(__file__))}
    def save():
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    try:
        identity = verify_identity()
        sys.path.insert(0, str(ROOT / "scripts"))
        import compare_lp
        import benchmark
        result.update(production_solver_sha256=PRODUCTION_SHA, diagnostic_solver_sha256=DIAGNOSTIC_SHA,
                      production_source_files_verified=88, diagnostic_sources_verified=5,
                      compiler_and_library_hashes_verified=True, instrumented_source_reconstructed=True,
                      identity_sha256=sha(DATA / "v40-progress-identity.json"), self_test=verify_self_test())
        plan = verify_plan(compare_lp)
        result["runs"] = verify_runs(identity, plan, benchmark)
        result["posthoc_analysis"] = verify_analysis()
        if not args.preflight:
            result.update(verify_documents(output))
        result.update(passed=True,
                      verification_scope="仅读取及重算归档；未调用编译器、求解器或任何性能采集。诊断时间不作性能成绩。",
                      numeric_scope="事件只存范数与标量：可核其恒等式、门槛、阶段和账本，不能从摘要重新形成完整矩阵向量乘积；独立解析自检使用精确有理数。",
                      optional_validity_scope="可选值缺失时value=null/valid=false不等同于数值异常；finite_fields跟踪实际输出的数值范围异常。")
    except Exception as error:
        result["error"] = str(error)
        save()
        raise
    save()
    print(json.dumps({"passed": result["passed"], "final": result["final"],
                      "raw_records": result["runs"]["raw_records"], "events": result["runs"]["events"],
                      "non_time_equal": result["runs"]["non_time_comparisons_equal"],
                      "time_limited_pairs": len(result["runs"]["time_limited_comparisons"]),
                      "anchors": result["posthoc_analysis"]["anchors"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    main()
