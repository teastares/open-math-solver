"""独立核验第79轮阶段计时诊断；不编译、链接或运行求解器。"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import types
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "b5df77438c3905566b0a8ddb02b84c2c2f65484facfe26bf559ec2ae085ec44b"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
PARENT_SHA = "65666c21a39398fef278735be14c79f064579a231d19f014eabaa9c7f106cb5b"
MODES = ("production", "off", "on1", "on2", "on3")
ADDED_STAGES = ("cone_multiply", "cone_transpose", "cholmod_symbolic", "cholmod_numeric", "cholmod_backsolve", "cholmod_values_copy",
                "normal_ordering", "aat_scaling", "cholmod_aat_numeric")
FIELDS = ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")
PREFIX = "V47_PROFILE "
WORK = ROOT / ".tools/v82-diagnostic"
OPTION_HELPER_SHA = "d7265080bc6ff915fdfa7bc4792f651ae929432a1778eb88704b54a135c2a347"
sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "JSON对象存在重复字段：" + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("JSON非有限常量：" + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read(path):
    return strict_json(Path(path).read_bytes())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def typed_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def load_module(path, name, digest):
    require(sha(path) == digest, "锁定复用来源改变：" + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def fixed_plan(check_inputs=False):
    require(sha(DATA / "v82-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v82-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_SHA == sha(DATA / "ipm-v66-identity.json"), "父程序身份锚点")
    identity = read(DATA / "ipm-v66-identity.json")
    require(plan["parent_solver_sha256"] == identity["solver_sha256"] and
            plan["parent_source_sha256"] == identity["source_zip_sha256"], "父程序及源码包冻结值")
    require(plan["evidence_file"] == "v81-closeout.json" and
            plan["evidence_sha256"] == sha(DATA / plan["evidence_file"]) and
            plan["production_precheck_sha256"] == plan["baseline_verification_sha256"] ==
            sha(DATA / plan["baseline_verification_file"]), "上轮核验与本轮生产预检来源")
    require(sha(DATA / plan["baseline_identity_file"]) == plan["baseline_identity_sha256"], "生产基线身份")
    legacy = load_module(DATA / "v47-profile-build.py", "v82_legacy_profile_builder", HELPER_SHA)
    require(plan["helper_sha256"] == HELPER_SHA and len(legacy.STAGES) == 26 and
            plan["profile_stages"] == list(legacy.STAGES + ADDED_STAGES), "只在原26阶段末尾追加9阶段")
    require(plan["diagnostic_only"] is True and (plan["round"], plan["version"], plan["parent_version"]) == (79, "v82", 66), "独立诊断范围")
    require(plan["modes"] == list(MODES) and type(plan["runs"]) is int and plan["runs"] == 85, "完整85次五模式")
    require((plan["method"], plan["primary_linear_solver"], plan["linear_accuracy"], plan["objective_scaling"],
             plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            ("homogeneous", "normal_cholmod_aat", "homogeneous_repair", "off", 5, 35, 1, 0, 200, 5), "固定共同数值政策及预算")
    require(plan["shared_ipm"] == read(DATA / "v66-plan.json")["shared_ipm"], "保持上轮其他IPM参数")
    require((plan["profile_environment"], plan["profile_enabled_value"], plan["profile_prefix"]) ==
            ("SOLVER_V47_PROFILE", "1", PREFIX), "复用旧计时环境及输出协议")
    require(plan["transformed_units"] == {"ipm": "src/ipm.cpp", "normal": "src/normal_cholesky.cpp"}, "仅插桩两翻译单元")
    previous_plan = read(DATA / "v81-plan.json")
    require(sha(DATA / "v81-plan.json") == read(DATA / "v81-closeout.json")["plan_sha256"], "上一轮模型来源计划")
    expected_models = []
    for group in ("development", "probe"):
        for model in previous_plan["sets"][group]["models"]:
            expected_models.append({**model, "model_index": len(expected_models)})
    require(typed_equal(plan["models"], expected_models) and len(expected_models) == 17 and
            len({model["input"] for model in expected_models}) == 17, "既有11开发与6探针的完整固定顺序")
    for model in plan["models"]:
        require(type(model["model_index"]) is int and model["hash_error"] is None and
                Path(model["input"]).is_absolute(), "输入类型和绝对位置")
        if check_inputs:
            require(sha(model["input"]) == model["sha256"], "实际输入身份")
    return plan, identity, legacy


def process_valid(row, benchmark):
    if row.get("error") is not None or type(row.get("returncode")) is not int or not isinstance(row.get("report"), dict):
        return False
    report = row["report"]
    timings = report.get("timings")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1 or not isinstance(timings, dict) or not all(
            finite(timings.get(key)) and timings[key] >= 0 for key in
            ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")):
        return False
    classification = benchmark.classify(row["returncode"], report)
    return classification in ("resolved", "incomplete", "verification_failed") and row["returncode"] == {
        "resolved": 0, "incomplete": 4, "verification_failed": 3}[classification]


def pair_result(a, b, benchmark):
    """先判实际退出和完整有限计时；预算只能豁免合法报告的轨迹差异。"""
    valid = all(process_valid(row, benchmark) and finite(row.get("elapsed_wall_seconds")) and row["elapsed_wall_seconds"] >= 0 and
                isinstance(row["report"].get("timings"), dict) and all(finite(row["report"]["timings"].get(key)) and
                row["report"]["timings"][key] >= 0 for key in ("solve_seconds", "verification_seconds")) for row in (a, b))
    reports = copy.deepcopy([a.get("report"), b.get("report")])
    for report in reports:
        if isinstance(report, dict):
            report.pop("timings", None)
    equal = (json.dumps(reports[0], sort_keys=True, allow_nan=False) == json.dumps(reports[1], sort_keys=True, allow_nan=False) and
             a.get("returncode") == b.get("returncode") and a.get("error") == b.get("error"))
    budget = any(budget_limited(row) for row in (a, b))
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def budget_limited(row):
    report = row.get("report") if isinstance(row.get("report"), dict) else {}
    timings = report.get("timings") if isinstance(report.get("timings"), dict) else {}
    return row.get("error") == "wall_timeout" or report.get("status") == "time_limit" or (
        finite(row.get("elapsed_wall_seconds")) and row["elapsed_wall_seconds"] > 35) or (
        all(finite(timings.get(key)) for key in ("solve_seconds", "verification_seconds")) and
        timings["solve_seconds"] + timings["verification_seconds"] > 5)


def profile(stderr, enabled, stages):
    lines = [line for line in stderr.splitlines() if line.startswith(PREFIX)]
    require(len(lines) == int(enabled), "开启侧恰一条计时报告，关闭侧不得有计时报告")
    if not lines:
        return None
    payload = strict_json(lines[0][len(PREFIX):])
    require(isinstance(payload, dict) and set(payload) == {"schema", "stages", "clock_reads", "exclusive_sum_ns", "total_inclusive_ns",
            "conserved", "stack_empty", "counts_closed"} and type(payload["schema"]) is int and payload["schema"] == 1, "计时报告完整字段")
    require(all(payload[key] is True for key in ("conserved", "stack_empty", "counts_closed")), "计时器自身报告闭合")
    actual = payload["stages"]
    require(isinstance(actual, dict) and set(actual) == set(stages) and len(actual) == 35, "全部35计时阶段")
    for name, entry in actual.items():
        require(isinstance(entry, dict) and set(entry) == set(FIELDS), "计时条目完整字段：" + name)
        require(all(type(entry[key]) is int and 0 <= entry[key] < 2 ** (63 if key.endswith("_ns") else 64) for key in FIELDS),
                "纳秒/次数必须为非负有界整数：" + name)
        require(entry["started"] == entry["completed"] + entry["unwound"] and entry["exclusive_ns"] <= entry["inclusive_ns"],
                "逐阶段正常/异常退出及包含关系：" + name)
        require(entry["started"] > 0 or entry["inclusive_ns"] == entry["exclusive_ns"] == 0, "未调用阶段不得有耗时：" + name)
    exclusive = sum(entry["exclusive_ns"] for entry in actual.values())
    starts = sum(entry["started"] for entry in actual.values())
    require(actual["total"]["started"] == 1 and exclusive == actual["total"]["inclusive_ns"], "独立复算全程独占时间守恒")
    require(type(payload["clock_reads"]) is int and payload["clock_reads"] == 2 * starts and
            type(payload["exclusive_sum_ns"]) is int and type(payload["total_inclusive_ns"]) is int and
            payload["exclusive_sum_ns"] == payload["total_inclusive_ns"] == exclusive, "独立复算时钟次数和声明总时间")
    return payload


def self_test(plan, identity):
    benchmark = load_module(ROOT / "scripts/benchmark.py", "v82_synthetic_benchmark", identity["files"]["scripts/benchmark.py"])
    original = {"error": None, "returncode": 4, "elapsed_wall_seconds": 2, "report": {
        "schema_version": 1, "status": "numerical_error", "backend_version": "ipm-cpu-0.45.0", "iterations": 2,
        "timings": {"read_seconds": 0, "solve_seconds": 1, "verification_seconds": 0, "total_seconds": 1},
        "options": {"ipm": {"objective_scaling": False}}}}
    trajectory_cases = 0

    def trajectory(left, right, category):
        nonlocal trajectory_cases
        require(pair_result(left, right, benchmark)["category"] == category, "合成轨迹或严格退出反例")
        trajectory_cases += 1

    trajectory(original, original, "equal")
    for field, value in (("iterations", 3), ("backend_version", "other")):
        row = copy.deepcopy(original)
        row["report"][field] = value
        trajectory(original, row, "different")
    row = copy.deepcopy(original)
    row["report"]["options"]["ipm"]["objective_scaling"] = 0
    trajectory(original, row, "different")
    for solve, verify, wall, status in ((2, 0, 2, "numerical_error"), (5, 0, 35, "numerical_error"),
                                      (5, 0.1, 2, "numerical_error"), (1, 0, 36, "numerical_error"), (1, 0, 2, "time_limit")):
        row = copy.deepcopy(original)
        row["elapsed_wall_seconds"] = wall
        row["report"]["status"] = status
        row["report"]["timings"].update(solve_seconds=solve, verification_seconds=verify, total_seconds=solve + verify)
        trajectory(original, row, "budget_limited" if status == "time_limit" or solve + verify > 5 or wall > 35 else "equal")
    for mutate in (
        lambda row: row.update(error="wall_timeout", returncode=None, report=None),
        lambda row: row.update(returncode=False),
        lambda row: row.update(returncode=3221225477),
        lambda row: row["report"].update(status="verification_failed"),
        lambda row: row["report"].update(schema_version=True),
        lambda row: row["report"]["timings"].update(solve_seconds=True),
        lambda row: row["report"].pop("timings"),
    ):
        row = copy.deepcopy(original)
        mutate(row)
        budget = copy.deepcopy(original)
        budget["report"]["timings"]["solve_seconds"] = 6
        trajectory(budget, row, "invalid_process")
    row = copy.deepcopy(original)
    row["report"]["status"], row["returncode"] = "verification_failed", 3
    trajectory(row, row, "equal")
    entries = {name: dict.fromkeys(FIELDS, 0) for name in plan["profile_stages"]}
    for name, values in (("total", (110, 20, 1, 1, 0)), ("canonicalize", (110, 50, 3, 2, 1)), ("normal_setup", (60, 40, 2, 1, 1))):
        entries[name] = dict(zip(FIELDS, values))
    golden = {"schema": 1, "stages": entries, "clock_reads": 12, "exclusive_sum_ns": 110, "total_inclusive_ns": 110,
              "conserved": True, "stack_empty": True, "counts_closed": True}
    encode = lambda value: PREFIX + json.dumps(value, allow_nan=False)
    require(profile(encode(golden), True, plan["profile_stages"]) == golden and profile("", False, plan["profile_stages"]) is None,
            "假时钟跨TU嵌套、异常及关闭黄金账本")
    rejected = 0
    for mutate in (
        lambda p: p.update(schema=True), lambda p: p.update(clock_reads=True), lambda p: p.update(clock_reads=11),
        lambda p: p.update(exclusive_sum_ns=109), lambda p: p.update(total_inclusive_ns=109),
        lambda p: p.update(conserved=False), lambda p: p.update(stack_empty=False), lambda p: p.update(counts_closed=False),
        lambda p: p["stages"].pop("cholmod_values_copy"), lambda p: p["stages"].update(unexpected=dict.fromkeys(FIELDS, 0)),
        lambda p: p["stages"]["total"].update(started=True), lambda p: p["stages"]["total"].update(exclusive_ns=-1),
        lambda p: p["stages"]["total"].update(inclusive_ns=2**63), lambda p: p["stages"]["total"].update(exclusive_ns=111),
        lambda p: p["stages"]["canonicalize"].update(unwound=0), lambda p: p["stages"]["cone_multiply"].update(inclusive_ns=1),
    ):
        candidate = copy.deepcopy(golden)
        mutate(candidate)
        try:
            profile(encode(candidate), True, plan["profile_stages"])
        except (AssertionError, ValueError):
            rejected += 1
        else:
            raise AssertionError("必须拒绝污染的计时账本")
    for stderr, enabled in (("", True), (encode(golden), False), (encode(golden) + "\n" + encode(golden), True),
                            (encode(golden).replace('"schema": 1', '"schema": 1, "schema": 1'), True)):
        try:
            profile(stderr, enabled, plan["profile_stages"])
        except (AssertionError, ValueError):
            rejected += 1
        else:
            raise AssertionError("必须拒绝缺失、重复或关闭侧计时报告")
    return {"passed": True, "trajectory_cases": trajectory_cases, "accounting_positive_cases": 2,
            "accounting_rejection_cases": rejected, "solver_runs": 0}


def audit_changes(original, changes, stages, unit):
    """不只信任可逆性，独立限制每个after为指定的计时包装。"""
    scopes = {name: name for name in (("total", "canonicalize", "kkt_setup", "factor", "solve", "krylov", "direct",
        "solve_factored", "kkt_product", "assess_homogeneous", "assess", "homogeneous_direction", "initialize",
        "run_homogeneous", "cone_multiply", "cone_transpose") if unit == "ipm" else ("normal_setup", "normal_factor", "normal_solve"))}
    expressions = ({"verify_original": ("verify_original", None), "kkt_ordering": ("ordering", None),
        "fallback_factor": ("fallback_factor_adapter", None), "fallback_solve": ("fallback_solve_adapter", None)} if unit == "ipm" else {
        "normal_ordering": ("normal_ordering", None), "symbolic_etree": ("symbolic_analysis", None),
        "symbolic_eigen": ("symbolic_analysis", None), "symbolic_extended": ("symbolic_analysis", None),
        "numeric_double": ("numeric_factor", None), "numeric_extended": ("numeric_factor", None),
        "triangular_double": ("triangular_solve", "Eigen::VectorXd"), "triangular_extended": ("triangular_solve", "ExtendedVector"),
        "cholmod_symbolic": ("cholmod_symbolic", None),
        "cholmod_backsolve": ("cholmod_backsolve", None)})
    expected = {"shared_header", *scopes, *expressions}
    if unit == "normal":
        expected |= {"factor_check_eigen", "factor_check_cholmod", "gram_assembly", "cholmod_values_copy", "aat_scaling", "cholmod_numeric"}
    require(len(changes) == len(expected) and {item["label"] for item in changes} == expected, "插桩边界完整且无重复")
    rebuilt = original
    for item in changes:
        require(set(item) == {"label", "before", "after"}, "插桩记录仅有完整三元组")
        label, before, after = (item[key] for key in ("label", "before", "after"))
        if label == "shared_header":
            require(before == ("namespace solver {" if unit == "ipm" else "namespace solver::detail {"), "头文件唯一命名空间锚点")
            desired = '#include "v47-profile.hpp"\n\n' + before
        elif label in scopes:
            patterns = {"total": r"SolveResult solve_ipm\([^{}]+\) \{",
                "canonicalize": r"ConeModel canonicalize\([^{}]+\) \{",
                "kkt_setup": r"KktFactor\([^{}]+krylov_history_\([^{}]+\) \{",
                "factor": r"void factor\([^{}]+\) \{", "solve": r"std::vector<double> solve\([^{}]+\) const \{",
                "krylov": r"std::vector<double> krylov_refine\([^{}]+\) const \{",
                "direct": r"std::vector<double> direct\([^{}]+\) const \{",
                "solve_factored": r"void solve_factored\([^{}]+\) const \{",
                "kkt_product": r"void unregularized_residual\([^{}]+\) const \{",
                "assess_homogeneous": r"bool assess_homogeneous\([^{}]+\) \{", "assess": r"bool assess\([^{}]+\) \{",
                "homogeneous_direction": r"HomogeneousDirection homogeneous_direction\([^{}]+\) \{",
                "initialize": r"void initialize\([^{}]+\) \{", "run_homogeneous": r"void run_homogeneous_ipm\([^{}]+\) \{",
                "cone_multiply": r"void multiply\([^{}]+\) const \{", "cone_transpose": r"void transpose\([^{}]+\) const \{",
                "normal_setup": r"Impl\(std::size_t size,[^{}]+max_refinement_steps\(refinement_budget\) \{",
                "normal_factor": r"bool NormalCholesky::factor\([^{}]+\) \{",
                "normal_solve": r"bool NormalCholesky::solve\([^{}]+\) const \{"}
            require(re.fullmatch(patterns[label], before) is not None, "函数作用域只能紧随指定入口：" + label)
            desired = before + ("\n  solver_v47_profile::Report v47_report;" if label == "total" else "") + \
                f"\n  solver_v47_profile::Scope v47_scope({stages.index(scopes[label])});"
        elif label in expressions:
            stage, value = expressions[label]
            function = "call" if value is None else "materialize<" + value + ">"
            desired = f"solver_v47_profile::{function}({stages.index(stage)}, [&] {{ return {before}; }})"
        elif label in ("factor_check_eigen", "factor_check_cholmod"):
            declaration = "  bool finite_factor(const std::function<void()>& budget_check) const {\n"
            require(before.startswith(declaration), "因子检查入口")
            desired = declaration + f"    solver_v47_profile::Scope v47_scope({stages.index('factor_check')});\n" + before[len(declaration):]
        elif label == "gram_assembly":
            require(before.startswith("  std::fill(state.sums.begin(), state.sums.end(), 0);") and
                    "state.rectangular" not in before and "state.input_values.assign" not in before,
                    "非矩形Gram边界不能包含AAt或输入复制")
            desired = f"  {{\n    solver_v47_profile::Scope v47_gram_scope({stages.index('gram_assembly')});\n" + before + "  }\n"
        elif label == "aat_scaling":
            require(before.startswith("      // C=[G D^(-1/2),sqrt(R)]") and
                    "state.rectangular_matrix.valuePtr()[entry]" in before and
                    "std::sqrt(diagonal)" in before and "state.cholmod->factor" not in before,
                    "AAt缩放计时仅包两组原缩放循环")
            desired = f"      {{\n        solver_v47_profile::Scope v47_aat_scope({stages.index('aat_scaling')});\n" + before + "      }\n"
        elif label == "cholmod_numeric":
            require(before == "cholmod_l_factorize(input, resources_.factor, &resources_.common)", "同步CHOLMOD原调用参数")
            desired = (f"solver_v47_profile::call(matrix_.stype == 0 ? {stages.index('cholmod_aat_numeric')} : "
                       f"{stages.index('cholmod_numeric')}, [&] {{ return {before}; }})")
        else:
            require(label == "cholmod_values_copy" and "lower_values_[destination] = values[static_cast<std::size_t>(lower_sources_[destination])];" in before and
                    "if ((destination & 4095) == 0) check(budget_check);" in before, "下三角复制与原预算检查均保留")
            desired = f"      {{\n        solver_v47_profile::Scope v47_copy_scope({stages.index(label)});\n" + before + "      }\n"
        require(after == desired and rebuilt.count(before) == 1, "独立核对只添加计时包装：" + label)
        rebuilt = rebuilt.replace(before, after, 1)
    restored = rebuilt
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆变换唯一")
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "独立逆变换完整恢复生产源码")
    return rebuilt


def verify_build_attempt(build):
    path = DATA / "v82-build-attempt1.json"
    record = read(path)
    require(type(record["returncode"]) is int and record["returncode"] == 0 and finite(record["elapsed_seconds"]) and
            record["elapsed_seconds"] > 0 and Path(record["command"][1]) == DATA / "v82-build.py" and
            "--self-test" not in record["command"], "实际构建外层执行原件")
    require(hashlib.sha256(record["source"].encode("utf-8")).hexdigest() == record["source_sha256"] == sha(DATA / "v82-build.py") and
            hashlib.sha256(record["wrapper_source"].encode("utf-8")).hexdigest() == record["wrapper_sha256"], "执行的构建器与wrapper源码原件")
    snapshots = record["build_input_sources_before_execution"]
    names = {"v82-plan.json", "v82-baseline-check.json", "v82-instrumentation.py", "v82-build.py",
             "v82-run.py", "v82-runner-self-check.json", "v47-profile-build.py"}
    require(set(snapshots) == names, "实际构建前恰七份来源快照")
    for name, item in snapshots.items():
        require(set(item) == {"source", "sha256"} and isinstance(item["source"], str) and
                item["source"].encode("utf-8") == (DATA / name).read_bytes() and
                hashlib.sha256(item["source"].encode("utf-8")).hexdigest() == item["sha256"] ==
                build["protected_before_and_after"][str(DATA / name)], "构建前来源文本、字节与保护身份：" + name)
    require(strict_json(record["stdout"])["executable_sha256"] == build["executable_sha256"] and
            datetime.datetime.fromisoformat(record["started_at_utc"]) <= datetime.datetime.fromisoformat(build["created_at_utc"]) <=
            datetime.datetime.fromisoformat(record["completed_at_utc"]), "构建输出对应实际执行时间窗口")
    return snapshots


def static_archive_member(path, desired):
    """只读解析普通ar成员，不调用归档器或执行工具。"""
    raw = Path(path).read_bytes()
    require(raw.startswith(b"!<arch>\n"), "原库必须是普通静态archive")
    offset, names, found = 8, b"", []
    while offset < len(raw):
        header = raw[offset:offset + 60]
        require(len(header) == 60 and header[-2:] == b"`\n", "静态archive成员头")
        size = int(header[48:58].decode("ascii").strip())
        require(size >= 0 and offset + 60 + size <= len(raw), "静态archive成员范围")
        body = raw[offset + 60:offset + 60 + size]
        name = header[:16].decode("ascii").strip()
        if name == "//":
            names = body
        elif name.startswith("/") and name[1:].isdigit():
            begin = int(name[1:]); require(begin < len(names), "静态archive长名称范围")
            name = names[begin:].split(b"/\n", 1)[0].split(b"\x00", 1)[0].decode("utf-8")
        elif name.startswith("#1/"):
            count = int(name[3:]); require(count <= len(body), "静态archive内联长名称范围")
            name, body = body[:count].rstrip(b"\x00").decode("utf-8"), body[count:]
        else:
            name = name.rstrip("/")
        if Path(name).name == desired:
            found.append(body)
        offset += 60 + size + size % 2
    require(offset == len(raw) and len(found) == 1, "冻结原库成员必须唯一：" + desired)
    return found[0]


def build_sources(plan, identity, legacy):
    manifest_path = WORK / "build-manifest.json"
    build = read(manifest_path)
    require(build["diagnostic_only"] is True and build["production_changes"] is False and
            build["plan_sha256"] == PLAN_SHA and build["frozen_files"] == build["current_files"] == identity["files"] and
            build["actual_libraries"] == identity["libraries"] and
            len(identity["files"]) == 95 and len(identity["libraries"]) == 17, "两TU诊断保持全部生产来源与库")
    for group in ("files", "libraries"):
        for name, digest in identity[group].items():
            require(sha(ROOT / name) == digest, "生产来源或库改变：" + name)
    production = ROOT / "build/release/solver.exe"
    parent = ROOT / ".tools/solver-ipm-v66.exe"
    diagnostic = WORK / "solver-profile.exe"
    require(Path(build["production_executable"]) == production and Path(build["executable"]) == diagnostic and
            Path(build["frozen_parent_executable"]) == parent and
            sha(production) == sha(parent) == build["production_sha256"] == build["frozen_parent_sha256"] == identity["solver_sha256"] and
            sha(diagnostic) == build["executable_sha256"] != build["production_sha256"], "实际双程序身份")
    require(sha(identity["compiler"]) == identity["compiler_sha256"] and build["implicit_link_inputs_verified"] is False, "编译器与隐式库保证边界")
    archive_path = DATA / "ipm-v66-source.zip"
    require(sha(archive_path) == identity["source_zip_sha256"], "冻结源码包身份")
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) == 95 and set(archive.namelist()) == set(identity["files"]), "源码包唯一成员集合")
        sources = {name: archive.read(name) for name in archive.namelist()}
    for name, content in sources.items():
        require(hashlib.sha256(content).hexdigest() == identity["files"][name] == sha(WORK / "frozen" / name), "生产/独立冻结成员身份")
    snapshots = verify_build_attempt(build)
    for group in ("protected_before_and_after", "outputs", "link_explicit_before_and_after", "compile_inputs_after_and_final",
                  "generated_inputs_before_and_after"):
        require(isinstance(build[group], dict) and build[group], "构建证据集合非空")
        for name, digest in build[group].items():
            require(Path(name).is_absolute() and sha(name) == digest, "构建证据实际身份：" + name)
    require(set(build["outputs"]) == {str(path) for path in WORK.iterdir() if path.is_file() and path != manifest_path}, "所有直接构建产物完整冻结，清单不自包含")
    protected = build["protected_before_and_after"]
    for path in (DATA / "v82-build.py", DATA / "v82-run.py", DATA / "v82-runner-self-check.json", DATA / "v82-instrumentation.py", DATA / "v82-plan.json",
                 DATA / plan["baseline_identity_file"], DATA / plan["baseline_verification_file"] , DATA / plan["evidence_file"],
                 DATA / "v47-profile-build.py", ROOT / "build/release/build.ninja",
                 ROOT / "build/release/CMakeFiles/rules.ninja", production, parent, Path(identity["compiler"])):
        require(protected.get(str(path)) == sha(path), "关键构建输入前后冻结")
    instrument_sha = snapshots["v82-instrumentation.py"]["sha256"]
    require(protected[str(DATA / "v82-instrumentation.py")] == instrument_sha, "独立静态复审的插桩来源")
    helper = load_module(DATA / "v82-instrumentation.py", "v82_instrumentation_audit", instrument_sha)
    stages = plan["profile_stages"]
    expected_header = legacy.HEADER.replace(
        "inline constexpr const char* names[] = {" + ",".join(json.dumps(name) for name in legacy.STAGES) + "};",
        "inline constexpr const char* names[] = {" + ",".join(json.dumps(name) for name in stages) + "};", 1).replace(
        "inline thread_local std::array<Entry, 26> entries{};", "inline thread_local std::array<Entry, 35> entries{};", 1)
    require(helper.HEADER == expected_header == (WORK / "v47-profile.hpp").read_text(encoding="utf-8") and
            helper.SELF_MAIN == legacy.SELF_MAIN == (WORK / "self-main.cpp").read_text(encoding="utf-8") and
            helper.SELF_CROSS == legacy.SELF_CROSS == (WORK / "self-cross.cpp").read_text(encoding="utf-8"), "计时器仅名称和容量变化，双TU测试逐字复用")
    changes = read(WORK / "transforms.json")
    require(set(changes) == {"ipm", "normal"}, "恰两个插桩翻译单元")
    for unit, relative in plan["transformed_units"].items():
        original = sources[relative].decode("utf-8").replace("\r\n", "\n")
        actual = changes[unit]
        patched = audit_changes(original, actual["changes"], stages, unit)
        require(helper.transform(original, unit) == (patched, actual["changes"]) and actual["reversible"] is True and
                Path(actual["source"]) == WORK / "frozen" / relative and Path(actual["patched"]) == WORK / (unit + "-profile.cpp") and
                actual["source_sha256"] == identity["files"][relative] and actual["patched_sha256"] == sha(actual["patched"]) and
                Path(actual["patched"]).read_text(encoding="utf-8") == patched, "变换产物及锚点可重复生成")
    require(read(WORK / "static-check.json") == helper.static_check(WORK / "frozen"), "静态正反例与变换证据一致")
    commands = build["commands"]
    require(commands == read(WORK / "commands.json") and Path(commands["cwd"]) == ROOT / "build/release", "实际编译/链接记录")
    cwd = Path(commands["cwd"])
    lines = (WORK / "ninja-commands.txt").read_text(encoding="utf-8").splitlines()
    original = {}
    for unit, relative in plan["transformed_units"].items():
        matched = [legacy.split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matched) == 1 and legacy.resolve(matched[0][0], cwd) == Path(identity["compiler"]), "原编译模板唯一且编译器一致")
        original[unit] = matched[0]
    require(commands["original_compile"] == original and [line for line in lines if " -o solver.exe " in line] ==
            [identity["link_command"]], "原始编译及链接参数对应冻结Ninja记录")
    outer = legacy.split_windows(identity["link_command"])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "真实链接包装")
    original_link = legacy.split_windows(outer[2][len("cd . && "):-len(" && cd .")])
    require(commands["original_link"] == original_link, "冻结实际链接序列")
    regularized = cwd / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    require(protected[str(regularized)] == sha(regularized) and
            static_archive_member(ROOT / "build/release/libsolver_ipm.a", "regularized_ldl.cpp.obj") == regularized.read_bytes(),
            "借用的regularized对象必须逐字节等于冻结原库成员")
    desired, replacements = [], collections.Counter()
    for value in original_link:
        if value == "libsolver_ipm.a":
            desired.extend([str(WORK / "ipm-profile.obj"), str(regularized)])
            replacements["ipm"] += 1
        elif value == "libsolver_normal_cholesky.a":
            desired.append(str(WORK / "normal-profile.obj"))
            replacements["normal"] += 1
        elif not value.startswith("-Wl,--out-implib,"):
            desired.append(value)
    desired[desired.index("-o") + 1] = str(diagnostic)
    require(replacements == {"ipm": 1, "normal": 1} and commands["link"] == desired, "只按原序替换两库与输出地址")
    explicit = {str(legacy.resolve(value, cwd)) for value in desired if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")}
    require(set(build["link_explicit_before_and_after"]) == explicit and len(explicit) == 19, "最终19个显式链接输入完整")
    dependencies = set()

    def successful(path, command, work):
        row = read(path)
        require(row["command"] == command and Path(row["cwd"]) == work and type(row["returncode"]) is int and
                row["returncode"] == 0 and row["wall_timeout"] is False, "实际构建/自测退出和命令")
        return row

    for unit in ("ipm", "normal"):
        obj = WORK / (unit + "-profile.obj")
        expected = legacy.compile_arguments(original[unit], WORK / (unit + "-profile.cpp"), obj, WORK / "frozen", ROOT)
        require(commands["compile"][unit] == expected, "独立源码使用真实参数，仅替换源码/对象/依赖/项目头路径")
        successful(WORK / (unit + "-compile.json"), expected, cwd)
        dependencies.update(str(path) for path in legacy.dependency_paths(Path(str(obj) + ".d"), cwd))
    successful(WORK / "link.json", desired, cwd)
    self_objects = []
    for index, name in enumerate(("main", "cross")):
        obj = WORK / ("self-" + name + ".obj")
        expected = [identity["compiler"], "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
            "-DSOLVER_V47_PROFILE_SELFTEST", "-MMD", "-MF", str(obj) + ".d", "-c", str(WORK / ("self-" + name + ".cpp")), "-o", str(obj)]
        require(build["self_test_compile"][index] == expected, "假时钟两个独立编译单元")
        successful(WORK / ("self-" + name + "-build.json"), expected, cwd)
        dependencies.update(str(path) for path in legacy.dependency_paths(Path(str(obj) + ".d"), cwd))
        self_objects.append(str(obj))
    self_exe = WORK / "profile-self-test.exe"
    self_link = [identity["compiler"], "-static", *self_objects, "-o", str(self_exe)]
    require(build["self_test_link"] == self_link and set(build["compile_inputs_after_and_final"]) == dependencies, "双TU链接及完整编译依赖集合")
    successful(WORK / "self-link.json", self_link, cwd)
    for mode in ("off", "on"):
        row = successful(WORK / ("self-" + mode + "-run.json"), [str(self_exe)], ROOT)
        enabled = mode == "on"
        require(typed_equal(strict_json(row["stdout"]), {"passed": True, "enabled": enabled}), "双TU假时钟自测实际结果")
        payload = profile(row["stderr"], enabled, stages)
        if enabled:
            for name, values in (("total", (110, 20, 1, 1, 0)), ("canonicalize", (110, 50, 3, 2, 1)), ("normal_setup", (60, 40, 2, 1, 1))):
                require(payload["stages"][name] == dict(zip(FIELDS, values)), "独立假时钟黄金计数与纳秒")
            require(all(entry == dict.fromkeys(FIELDS, 0) for name, entry in payload["stages"].items() if name not in ("total", "canonicalize", "normal_setup")), "其他假时钟阶段均零")
        else:
            require(row["stderr"] == "", "关闭自测无诊断输出")
    require(typed_equal(read(DATA / "v82-runner-self-check.json"), {"passed": True, "trajectory_cases": 9, "profile_positive_cases": 2,
            "profile_negative_cases": 14, "environment_cases": 3, "timer_positive_cases": 2, "timer_negative_cases": 8,
            "solver_runs": 0}), "运行器实际纯合成自检")
    require(typed_equal(read(WORK / "timer-self-check.json"), {"passed": True, "solver_runs": 0,
            "off": {"passed": True, "enabled": False}, "on": {"passed": True, "enabled": True}}), "计时器自测摘要和实际原件一致")
    generated = {str(WORK / name) for name in ("v47-profile.hpp", "ipm-profile.cpp", "normal-profile.cpp", "self-main.cpp", "self-cross.cpp")}
    require(set(build["generated_inputs_before_and_after"]) == generated and generated <= set(build["compile_inputs_after_and_final"]),
            "全部生成源码及共享头在编译前后及依赖链中保持身份")
    for prefix in ("baseline_identity", "baseline_verification"):
        require(Path(build[prefix + "_file"]) == DATA / plan[prefix + "_file"] and
                build[prefix + "_sha256"] == plan[prefix + "_sha256"], "构建基线字段对应计划")
    require(datetime.datetime.fromisoformat(plan["created_at_utc"]) <= datetime.datetime.fromisoformat(build["created_at_utc"]), "先冻结再构建")
    protocols = []
    for name in ("benchmark", "compare_lp"):
        module = types.ModuleType(name)
        module.__file__ = str(ROOT / "scripts" / (name + ".py"))
        sys.modules[name] = module
        exec(compile(sources["scripts/" + name + ".py"], module.__file__, "exec"), module.__dict__)
        protocols.append(module)
    return build, protocols[0], protocols[1], {"source_files": 95, "linked_libraries": 17,
        "explicit_link_inputs": len(explicit), "compile_dependencies": len(dependencies),
        "anchors": {unit: len(value["changes"]) for unit, value in changes.items()}, "stages": len(stages),
        "header_core_unchanged": True, "self_test_two_translation_units": True,
        "runner_self_check_protected_at_build": str(DATA / "v82-runner-self-check.json") in protected,
        "build_input_snapshots": 7, "instrumentation_sha256": instrument_sha}


def verify_runs(plan, identity, build, benchmark, compare):
    manifest = read(DATA / "v82-run-manifest.json")
    build_path = WORK / "build-manifest.json"
    protected = {}
    def protect(path, digest):
        key = str(Path(path).resolve())
        require(key not in protected or protected[key] == digest, "运行保护来源冲突")
        require(sha(key) == digest, "运行保护实际身份")
        protected[key] = digest
    for group in ("protected_before_and_after", "outputs", "link_explicit_before_and_after",
                  "compile_inputs_after_and_final", "generated_inputs_before_and_after"):
        for path, digest in build[group].items(): protect(path, digest)
    for group in ("files", "libraries"):
        for name, digest in identity[group].items(): protect(ROOT / name, digest)
    for name, digest in identity["files"].items(): protect(WORK / "frozen" / name, digest)
    for path, digest in ((DATA / "v82-run.py", sha(DATA / "v82-run.py")), (build_path, sha(build_path)),
        (build["production_executable"], build["production_sha256"]), (build["executable"], build["executable_sha256"]),
        (ROOT / ".tools/solver-ipm-v66.exe", identity["solver_sha256"]),
        (DATA / "ipm-v66-identity.json", PARENT_SHA), (DATA / "ipm-v66-source.zip", identity["source_zip_sha256"])):
        protect(path, digest)
    for prefix in ("baseline_identity", "baseline_verification"):
        protect(DATA / plan[prefix + "_file"], plan[prefix + "_sha256"])
    protect(DATA / plan["evidence_file"], plan["evidence_sha256"])
    protect(DATA / "v47-profile-build.py", HELPER_SHA)
    for item in plan["local_references"]: protect(ROOT / item["file"], item["sha256"])
    for model in plan["models"]: protect(model["input"], model["sha256"])
    require(manifest["plan_sha256"] == PLAN_SHA and Path(manifest["build_manifest"]) == build_path and Path(manifest["work"]) == WORK and
            manifest["protected"] == protected and manifest["schedule"] == plan["order"], "执行清单的计划、构建、双程序与生产来源链")
    require(manifest["environment"] == dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1") and
            manifest["removed_environment"] == "清除所有SOLVER_V开头变量及OPENBLAS_CORETYPE；仅on设置冻结计时开关", "共同单线程和唯一诊断开关")
    require(datetime.datetime.fromisoformat(build["created_at_utc"]) <= datetime.datetime.fromisoformat(manifest["started_at_utc"]), "先完成独立构建再运行")
    require(all(sha(path) == digest for path, digest in protected.items()), "运行保护集合当前仍保持身份")
    options = load_module(DATA / "v66-run.py", "v82_frozen_options_audit", OPTION_HELPER_SHA)
    expected_options = options.expected_options("highs", "candidate", options.plan())
    raw_path = DATA / "v82-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw_path.read_bytes()).splitlines()]
    schedule = [(model, mode) for model in plan["models"] for mode in
                MODES[model["model_index"] % 5:] + MODES[:model["model_index"] % 5]]
    require(len(rows) == len(schedule) == 85, "事前85条全部保留")
    invalid, missing, profile_failures = [], [], []
    by_model = collections.defaultdict(dict)
    for index, (row, (model, mode)) in enumerate(zip(rows, schedule)):
        require(type(row["index"]) is int and type(row["case_index"]) is int and
                (row["index"], row["case_index"], row["input"], row["model"], row["variant"]) ==
                (index, model["model_index"], model["input"], Path(model["input"]).stem, mode), "逐输入左旋五模式真实顺序")
        executable = build["production_executable"] if mode == "production" else build["executable"]
        digest = build["production_sha256"] if mode == "production" else build["executable_sha256"]
        expected = [executable, "solve", model["input"], "--backend", "ipm", "--time-limit", "5", "--threads", "1", "--seed", "0",
            "--iteration-limit", "200", "--scaling-passes", "5", "--ipm-method", "homogeneous", "--ipm-linear-accuracy", "homogeneous_repair",
            "--ipm-linear-solver", "normal_cholmod_aat", "--ipm-direction-retries", "0", "--ipm-objective-scaling", "off"]
        for key, value in plan["shared_ipm"].items():
            expected.extend(["--ipm-" + key, value])
        require(row["command"] == expected, "逐次真实命令与原数值政策一致")
        require(row["hashes_before"] == row["hashes_after"] == {"input": model["sha256"], "solver": digest} and
                row["model_sha256"] == model["sha256"], "逐次输入与本侧程序前后身份")
        require(finite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] > 0, "真实墙钟为有限正数")
        report = row["report"]
        if report is not None:
            require(isinstance(report, dict) and typed_equal(strict_json(row["stdout"]), report) and report["input"] == model["input"] and
                    row["classification"] == benchmark.classify(row["returncode"], report), "原始JSON报告、输入及真实分类")
            require(typed_equal(report["options"], expected_options) and report["backend_version"] == "ipm-cpu-0.53.0" and
                    report["backend"] == "ipm" and report["model"]["is_mip"] is False and
                    options.valid_counts(report, 66, "ipm", "homogeneous_repair") and options.valid_factor_path(report, "ipm"),
                    "冻结v66实际配置、完整计数与因子路径")
            require(isinstance(report["timings"], dict) and all(finite(report["timings"].get(key)) and report["timings"][key] >= 0 for key in
                ("read_seconds", "solve_seconds", "verification_seconds", "total_seconds")), "成功失败报告均有完整有限计时")
            numerical = report["timings"]["solve_seconds"] + report["timings"]["verification_seconds"]
            require(finite(numerical), "数值耗时总和有限")
        else:
            missing.append({"index": index, "model": row["model"], "variant": mode, "error": row["error"]})
            require(row["classification"] == row["error"] and row["error"] in ("wall_timeout", "launch_error", "invalid_report", "invalid_json"), "缺报告失败仍原样保留")
            numerical = None
        if not process_valid(row, benchmark):
            invalid.append({"index": index, "model": row["model"], "variant": mode, "returncode": row["returncode"], "error": row["error"]})
        failure = compare.quality_failure(row, model, model["sha256"], digest, digest)
        if failure is None and (numerical > 5 or row["elapsed_wall_seconds"] > 35):
            failure = "budget_exceeded"
        require((numerical is None and row["numerical_seconds"] is None or finite(row["numerical_seconds"]) and
                row["numerical_seconds"] == numerical == compare.numerical_seconds(row)) and
                row["quality_failure"] == failure and row["quality_pass"] is (failure is None), "逐条原模型质量及完整预算重算")
        try:
            parsed = profile(row["stderr"], mode.startswith("on"), plan["profile_stages"])
        except (AssertionError, ValueError, KeyError, TypeError) as error:
            require(row["profile"] is None and isinstance(row["profile_error"], str) and row["profile_error"], "采集失败不能伪造完整账本")
            profile_failures.append({"index": index, "model": row["model"], "variant": mode, "reason": str(error)})
        else:
            require(typed_equal(row["profile"], parsed) and row["profile_error"] is None, "独立解析stderr并重算全部计时账本")
        require(mode not in by_model[model["model_index"]], "同例模式不能重复")
        by_model[model["model_index"]][mode] = row
    pairs, cases = [], []
    for model in plan["models"]:
        selected = by_model[model["model_index"]]
        require(set(selected) == set(MODES), "每输入完整五模式")
        for left, right in (("production", "off"), ("off", "on1"), ("off", "on2"), ("off", "on3")):
            pairs.append({"case_index": model["model_index"], "model": Path(model["input"]).stem,
                "left": left, "right": right, **pair_result(selected[left], selected[right], benchmark)})
        active = [selected[mode] for mode in ("on1", "on2", "on3")]
        medians = {name: {key: statistics.median(row["profile"]["stages"][name][key] for row in active) for key in FIELDS}
            for name in plan["profile_stages"]} if all(row["profile_error"] is None and row["profile"] is not None for row in active) else None
        cases.append({"case_index": model["model_index"], "model": Path(model["input"]).stem,
            "status": {name: (row["report"] or {}).get("status") for name, row in selected.items()},
            "numerical_seconds": {name: row["numerical_seconds"] for name, row in selected.items()}, "stage_medians": medians})
    summary = {"diagnostic_only": True, "runs": 85, "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "profile_errors": len(profile_failures),
        "scope": "仅诊断成本；阶段中位数分别计算不承诺加总守恒，不作为公开性能或新覆盖。", "raw_sha256": sha(raw_path)}
    require(json.dumps(summary, sort_keys=True, allow_nan=False) == json.dumps(read(DATA / "v82-summary.json"), sort_keys=True, allow_nan=False),
            "从全部raw独立重建完整摘要和每例三次中位数，严格区分bool/int")
    return summary, {"missing_reports": missing, "invalid_process_reports": invalid, "profile_failures": profile_failures,
        "enabled_profiles": sum(row["profile"] is not None for row in rows),
        "quality_pass_by_variant": {mode: sum(row["quality_pass"] for row in rows if row["variant"] == mode) for mode in MODES}}, rows


def verify_common_success(compare):
    analysis = read(DATA / "v82-common-success-analysis.json")
    require(analysis["passed"] is True and type(analysis["new_solver_runs"]) is int and analysis["new_solver_runs"] == 0,
            "共同成功分析不得新增LP")
    expected_sources = {DATA / name for name in ("v82-plan.json", "v81-closeout.json", "v81-run.py", "v66-run.py",
        "v81-highs-manifest.json", "v81-highs-runs.jsonl.gz", "v81-highs-summary.json")}
    expected_sources |= {ROOT / "scripts/benchmark.py", ROOT / "scripts/compare_lp.py"}
    require(set(analysis["source_sha256"]) == {str(path) for path in expected_sources} and
            all(sha(path) == digest for path, digest in analysis["source_sha256"].items()), "同期182原件分析的完整来源身份")
    require(hashlib.sha256(analysis["driver_source"].encode("utf-8")).hexdigest() == analysis["driver_sha256"], "共同成功分析执行源码原件")
    previous = load_module(DATA / "v81-run.py", "v82_common_success_frozen_protocol",
        "d3b0e4e0c67e93766dec8cf05ccb3e0943fa90df62dafc6f17ea404052d95bec")
    _, summary, rows = previous.archive("highs")
    require(len(rows) == analysis["original_public_runs"] == 182 and analysis["original_public_models"] == 91 and
            typed_equal(summary, read(DATA / "v81-closeout.json")["stage_summaries"]["highs"]), "仅复核前轮同期182原始运行")
    variants = ("highs", "diagnostic_on")
    grouped = {variant: {} for variant in variants}
    for row in rows:
        side = row["requested_variant"]
        key = (row["input"], row["repetition"])
        require(side in grouped and key not in grouped[side] and previous.valid_process(row), "同期运行必须有合法真实进程且无重复")
        require(typed_equal(strict_json(row["stdout"]), row["report"]) and
                finite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] >= 0 and
                finite(row["numerical_seconds"]) and row["numerical_seconds"] == compare.numerical_seconds(row), "同期原始报告和两种实际计时")
        grouped[side][key] = row
    require(set(grouped["highs"]) == set(grouped["diagnostic_on"]) and len(grouped["highs"]) == 91, "完整91模型双侧集合")
    common = sorted(key for key in grouped["highs"] if all(grouped[side][key]["quality_pass"] for side in variants))
    excluded = sorted(set(grouped["highs"]) - set(common))
    def distribution(values):
        ordered = sorted(values)
        require(ordered and all(finite(value) and value >= 0 for value in ordered), "共同成功计时分布必须有限")
        return {"count": len(ordered), "sum": sum(ordered), "minimum": ordered[0], "median": statistics.median(ordered),
            "p90_nearest_rank": ordered[max(0, math.ceil(.9 * len(ordered)) - 1)],
            "p95_nearest_rank": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], "maximum": ordered[-1],
            "sgm10_seconds": compare.shifted_geometric_mean(ordered)}
    full, distributions, details = {}, {}, []
    for side in variants:
        selected = list(grouped[side].values())
        full[side] = {"run_count": len(selected), "quality_pass_count": sum(row["quality_pass"] for row in selected),
            "penalized_wall_sgm10_seconds": compare.shifted_geometric_mean([row["quality_penalized_wall_seconds"] for row in selected]),
            "penalized_numerical_sgm10_seconds": compare.shifted_geometric_mean([row["quality_penalized_numerical_seconds"] for row in selected])}
        require(all(full[side][key] == summary["variants"][side][key] for key in full[side]), "原含失败主指标不得被共同成功子集替换")
        distributions[side] = {"wall": distribution([grouped[side][key]["elapsed_wall_seconds"] for key in common]),
            "numerical": distribution([grouped[side][key]["numerical_seconds"] for key in common]),
            "reported_read": distribution([grouped[side][key]["report"]["timings"]["read_seconds"] for key in common])}
    for key in sorted(grouped["highs"]):
        sides = {}
        for side in variants:
            row = grouped[side][key]
            sides[side] = {field: row[field] for field in ("quality_pass", "quality_failure", "status", "returncode", "error",
                "elapsed_wall_seconds", "numerical_seconds", "quality_penalized_wall_seconds", "quality_penalized_numerical_seconds")}
            sides[side]["reported_timings"] = row["report"]["timings"]
        details.append({"input": key[0], "repetition": key[1], "in_common_success": key in common, "variants": sides})
    for field, value in (("full_public_metrics_unchanged", full), ("common_success_distributions", distributions),
        ("all_public_model_details", details), ("excluded_from_common_success_only", [{"input": key[0], "repetition": key[1]} for key in excluded]),
        ("common_success_count", len(common))):
        require(typed_equal(analysis[field], value), "独立重建共同成功分析：" + field)
    return {"passed": True, "new_solver_runs": 0, "original_public_runs": 182, "common_success_count": len(common),
            "analysis_sha256": sha(DATA / "v82-common-success-analysis.json")}


def verify_cost_analysis(plan, summary, rows):
    analysis = read(DATA / "v82-cost-analysis.json")
    require(analysis["passed"] is True and type(analysis["new_solver_runs"]) is int and analysis["new_solver_runs"] == 0 and
            analysis["actual_profile_runs"] == 85 and analysis["actual_on_profiles"] == 51, "本轮成本分析不新增LP")
    for path, digest in analysis["source_sha256"].items(): require(sha(path) == digest, "成本分析原始来源身份")
    require(set(analysis["source_sha256"]) == {str(DATA / ("v82-" + name)) for name in
        ("plan.json", "run-manifest.json", "runs.jsonl.gz", "summary.json")} and
        hashlib.sha256(analysis["driver_source"].encode("utf-8")).hexdigest() == analysis["driver_sha256"], "成本分析执行源码及完整输入")
    on = [row for row in rows if row["variant"].startswith("on")]
    aggregates = {}
    for name, selected in (("all_on", on), ("quality_pass_on", [row for row in on if row["quality_pass"]])):
        total = sum(row["profile"]["total_inclusive_ns"] for row in selected)
        amounts = {stage: {key: sum(row["profile"]["stages"][stage][key] for row in selected) for key in
            ("exclusive_ns", "inclusive_ns", "started")} for stage in plan["profile_stages"]}
        require(sum(item["exclusive_ns"] for item in amounts.values()) == total, "实际观测纳秒聚合独占守恒")
        ranking = sorted(plan["profile_stages"], key=lambda stage: amounts[stage]["exclusive_ns"], reverse=True)
        aggregates[name] = {"profiles": len(selected), "total_inclusive_ns": total, "exclusive_sum_ns": total, "conserved": True,
            "stages_by_exclusive": [{"stage": stage, "exclusive_ns": amounts[stage]["exclusive_ns"],
                "exclusive_fraction": amounts[stage]["exclusive_ns"] / total, "inclusive_ns": amounts[stage]["inclusive_ns"],
                "started": amounts[stage]["started"]} for stage in ranking]}
    cases = []
    for case in summary["cases"]:
        medians = case["stage_medians"]
        cases.append({"model": case["model"], "case_index": case["case_index"], "median_total_ns": medians["total"]["inclusive_ns"],
            "top_exclusive_medians": [{"stage": stage, **medians[stage]} for stage in sorted(medians,
                key=lambda stage: medians[stage]["exclusive_ns"], reverse=True)[:5]],
            "status": case["status"], "numerical_seconds": case["numerical_seconds"]})
    require(typed_equal(analysis["aggregate_observed_time"], aggregates) and typed_equal(analysis["per_model_medians"], cases),
            "独立重建实际纳秒聚合，不能把各阶段中位数之和当总时间")
    return {"passed": True, "new_solver_runs": 0, "all_on": len(on), "quality_pass_on": aggregates["quality_pass_on"]["profiles"],
            "analysis_sha256": sha(DATA / "v82-cost-analysis.json")}


def package_evidence():
    destination = DATA / "v82-verification-evidence.zip"
    require(not destination.exists(), "核验证据包不可覆盖")
    files = {path for path in DATA.glob("v82-*") if path.is_file() and
             path.name != destination.name and path.suffix in (".py", ".json", ".gz")}
    files.update(path for path in (ROOT / ".tools").glob("v82*.py") if path.is_file())
    files.update(path for path in WORK.iterdir() if path.is_file() and path.suffix in (".cpp", ".hpp", ".json", ".d", ".txt"))
    identity = read(DATA / "ipm-v66-identity.json")
    files.update(WORK / "frozen" / name for name in identity["files"])
    for name in ("v47-profile-build.py", "v66-run.py", "v66-plan.json", "ipm-v66-identity.json", "ipm-v66-source.zip",
                 "v66-accepted-state.json", "v81-run.py", "v81-plan.json", "v81-run-identity.json", "v81-closeout.json",
                 "v81-highs-manifest.json", "v81-highs-runs.jsonl.gz", "v81-highs-summary.json"):
        files.add(DATA / name)
    originals = {}
    for path in sorted(files):
        resolved = path.resolve()
        require(resolved.is_relative_to(ROOT) and resolved.is_file() and resolved.suffix.lower() not in (".exe", ".a", ".lib", ".obj", ".o"),
                "证据封包只能收工作区来源和执行记录，不收程序或库")
        originals[resolved.relative_to(ROOT).as_posix()] = resolved.read_bytes()
    expected = {name: hashlib.sha256(raw).hexdigest() for name, raw in originals.items()}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in originals.items(): archive.writestr(name, raw)
    with zipfile.ZipFile(destination) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) == len(expected) and set(archive.namelist()) == set(expected),
                "核验证据包成员完整且唯一")
        for name, digest in expected.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, "封包后逐成员原字节SHA回读：" + name)
    return {"file": str(destination), "sha256": sha(destination), "members": len(expected), "member_sha256": expected,
        "source_and_execution_originals_read_back": True, "excluded_binaries_and_libraries": True,
        "late_artifacts_not_included": ["本次核验返回后生成的outer attempt", "本次verification JSON自身", "后续文档检查记录"]}


def verify():
    plan, identity, legacy = fixed_plan(check_inputs=True)
    build, benchmark, compare, sources = build_sources(plan, identity, legacy)
    summary, records, rows = verify_runs(plan, identity, build, benchmark, compare)
    common = verify_common_success(compare)
    cost = verify_cost_analysis(plan, summary, rows)
    passed = summary["invalid_pairs"] == summary["different_pairs"] == summary["profile_errors"] == 0 and not records["missing_reports"] and not records["invalid_process_reports"]
    result = {"passed": passed, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "new_solver_runs": 0, "scope": "85条冻结诊断、68对完整轨迹、51条on整数账本、17组阶段中位数及前轮182条共同成功分析的独立核验",
        "plan_sha256": PLAN_SHA, "verifier_sha256": sha(__file__), "helper_sha256": HELPER_SHA,
        "instrumentation_sha256": sources["instrumentation_sha256"], "option_verifier_sha256": OPTION_HELPER_SHA,
        "parent_identity_sha256": PARENT_SHA, "production_sha256": identity["solver_sha256"], "diagnostic_sha256": build["executable_sha256"],
        "source_checks": sources, "run_count": 85, "pairs": {key: summary[key] for key in
            ("non_budget_pairs", "non_budget_equal", "budget_pairs", "invalid_pairs", "different_pairs", "profile_errors")},
        "record_checks": records, "cases": summary["cases"], "self_test": self_test(plan, identity),
        "common_success_analysis": common, "cost_analysis": cost,
        "artifacts": {name: sha(DATA / name) for name in ("v82-build.py", "v82-run.py", "v82-run-manifest.json", "v82-runs.jsonl.gz",
            "v82-summary.json", "v82-runner-self-check.json", "v82-common-success-analysis.json", "v82-cost-analysis.json")},
        "build_manifest_sha256": sha(WORK / "build-manifest.json"),
        "limits": ["不运行LP或重构原矩阵；重新检查冻结报告、真实命令与共同质量合同。",
            "仅剔timings比较轨迹；合法预算分叉单列，缺报告和异常退出不获豁免。",
            "每条原始计时账本严格守恒；分别取各阶段中位数后的合计不要求守恒，包含时间与子阶段不能重复相加。",
            "复用v47计时核心；AAt C API包含库内部矩形乘积与分解，不能全部称纯Cholesky；Normal独占也不全是稀疏遍历。",
            "计时覆盖solve_ipm作用域及局部析构，不含输入、JSON、Report初始化与诊断输出，不能要求与生产timings逐纳秒相等。",
            "仅核显式链接输入与依赖文件列举的编译依赖；隐式运行库未逐项核验。",
            "本轮为17个已观察输入的成本诊断，不新增公开性能或覆盖成绩，不把单次off与production差异称为统计显著开销。",
            "共同成功子集仅用于归因，91例含失败主指标完整保留；含失败wall/numerical的SGM不能相减推阶段耗时。"]}
    result["evidence_package"] = package_evidence()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output
    if destination is None and not (args.preflight or args.self_test):
        destination = DATA / "v82-verification.json"
    if destination is not None:
        require(not destination.exists(), "核验输出不可覆盖")
    if args.preflight or args.self_test:
        plan, identity, legacy = fixed_plan()
        result = {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "plan_sha256": PLAN_SHA, "helper_sha256": HELPER_SHA, "verifier_sha256": sha(__file__),
                  "scope": "只核冻结计划、纯合成轨迹及计时账本，不读取性能raw或大库", "self_test": self_test(plan, identity)}
    else:
        result = verify()
    if destination is not None:
        with destination.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "run_count", "pairs", "record_checks", "source_checks", "self_test") if key in result}, ensure_ascii=False, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)
