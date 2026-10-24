"""独立核验第52轮阶段计时诊断；不编译、链接或运行求解器。"""
import argparse
import collections
import copy
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import types
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "7bb52355663442a361ff799fdddf61b96e908c3e5302eb2615e8e4cf7e58a6a9"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
PARENT_SHA = "43289afc82b6c73892b5fa91ae5363dcd4e0a27baad56c127350c9bff42c0c73"
MODES = ("production", "off", "on1", "on2", "on3")
ADDED_STAGES = ("cone_multiply", "cone_transpose", "cholmod_symbolic", "cholmod_numeric", "cholmod_backsolve", "cholmod_values_copy")
FIELDS = ("inclusive_ns", "exclusive_ns", "started", "completed", "unwound")
PREFIX = "V47_PROFILE "
WORK = ROOT / ".tools/v55-diagnostic"
OPTION_HELPER_SHA = "547b7d9a9c4c37120f8dbb0fa0b8d0b8b768ff1613d8004869a863a89cd118c3"
INSTRUMENTATION_SHA = "660276e8ef25e37ea9f28abdfbd4604e410bbbde5ace0ae31db860180a3fb94d"
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


def load_module(path, name, digest):
    require(sha(path) == digest, "锁定复用来源改变：" + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def fixed_plan(check_inputs=False):
    require(sha(DATA / "v55-plan.json") == PLAN_SHA, "冻结计划身份")
    plan = read(DATA / "v55-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_SHA == sha(DATA / "ipm-v54-identity.json"), "父程序身份锚点")
    identity = read(DATA / "ipm-v54-identity.json")
    require(plan["parent_solver_sha256"] == identity["solver_sha256"] and
            plan["parent_source_sha256"] == identity["source_zip_sha256"], "父程序及源码包冻结值")
    require(plan["evidence_sha256"] == sha(DATA / "v54-verification.json") and
            plan["production_precheck_sha256"] == sha(DATA / "v55-production-precheck.json"), "上轮核验与本轮生产预检来源")
    legacy = load_module(DATA / "v47-profile-build.py", "v55_legacy_profile_builder", HELPER_SHA)
    require(plan["helper_sha256"] == HELPER_SHA and len(legacy.STAGES) == 26 and
            plan["profile_stages"] == list(legacy.STAGES + ADDED_STAGES), "只在原26阶段末尾追加6阶段")
    require(plan["diagnostic_only"] is True and (plan["round"], plan["version"], plan["parent_version"]) == (52, "v55", 54), "独立诊断范围")
    require(plan["modes"] == list(MODES) and type(plan["runs"]) is int and plan["runs"] == 40, "完整40次五模式")
    require((plan["method"], plan["primary_linear_solver"], plan["linear_accuracy"], plan["objective_scaling"],
             plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"], plan["max_iterations"], plan["scaling_passes"]) ==
            ("homogeneous", "normal_cholmod", "homogeneous_repair", "off", 5, 35, 1, 0, 200, 5), "固定共同数值政策及预算")
    require(plan["shared_ipm"] == read(DATA / "v54-plan.json")["shared_ipm"], "保持上轮其他IPM参数")
    require((plan["profile_environment"], plan["profile_enabled_value"], plan["profile_prefix"]) ==
            ("SOLVER_V47_PROFILE", "1", PREFIX), "复用旧计时环境及输出协议")
    require(plan["transformed_units"] == {"ipm": "src/ipm.cpp", "normal": "src/normal_cholesky.cpp"}, "仅插桩两翻译单元")
    names = ("maros-r7", "greenbeb", "fit2d", "pilot87", "fit1p", "greenbea", "dfl001", "afiro")
    require(tuple(Path(model["input"]).stem for model in plan["models"]) == names and
            [model["model_index"] for model in plan["models"]] == list(range(8)), "事前8例输入与顺序")
    for model in plan["models"]:
        require(type(model["model_index"]) is int and model["hash_error"] is None and
                Path(model["input"]) == ROOT / ".deps/netlib" / (names[model["model_index"]] + ".mps"), "输入类型和位置")
        if check_inputs:
            require(sha(model["input"]) == model["sha256"], "实际输入身份")
    return plan, identity, legacy


def process_valid(row, benchmark):
    if row.get("error") is not None or type(row.get("returncode")) is not int or not isinstance(row.get("report"), dict):
        return False
    classification = benchmark.classify(row["returncode"], row["report"])
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
    require(isinstance(actual, dict) and set(actual) == set(stages) and len(actual) == 32, "全部32计时阶段")
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
    benchmark = load_module(ROOT / "scripts/benchmark.py", "v55_synthetic_benchmark", identity["files"]["scripts/benchmark.py"])
    original = {"error": None, "returncode": 4, "elapsed_wall_seconds": 2, "report": {
        "schema_version": 1, "status": "numerical_error", "backend_version": "ipm-cpu-0.45.0", "iterations": 2,
        "timings": {"solve_seconds": 1, "verification_seconds": 0}, "options": {"ipm": {"objective_scaling": False}}}}
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
        row["report"].update(status=status, timings={"solve_seconds": solve, "verification_seconds": verify})
        trajectory(original, row, "budget_limited" if status == "time_limit" or solve + verify > 5 or wall > 35 else "equal")
    for mutate in (
        lambda row: row.update(error="wall_timeout", returncode=None, report=None),
        lambda row: row.update(returncode=False),
        lambda row: row.update(returncode=3221225477),
        lambda row: row["report"].update(status="verification_failed"),
        lambda row: row["report"].update(schema_version=True),
        lambda row: row["report"].update(timings={"solve_seconds": True, "verification_seconds": 0}),
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
        "normal_ordering": ("ordering", None), "symbolic_etree": ("symbolic_analysis", None),
        "symbolic_eigen": ("symbolic_analysis", None), "symbolic_extended": ("symbolic_analysis", None),
        "numeric_double": ("numeric_factor", None), "numeric_extended": ("numeric_factor", None),
        "triangular_double": ("triangular_solve", "Eigen::VectorXd"), "triangular_extended": ("triangular_solve", "ExtendedVector"),
        "cholmod_symbolic": ("cholmod_symbolic", None), "cholmod_numeric": ("cholmod_numeric", None),
        "cholmod_backsolve": ("cholmod_backsolve", None)})
    expected = {"shared_header", *scopes, *expressions}
    if unit == "normal":
        expected |= {"factor_check_eigen", "factor_check_cholmod", "gram_assembly", "cholmod_values_copy"}
    require(len(changes) == len(expected) and {item["label"] for item in changes} == expected, "插桩边界完整且无重复")
    rebuilt = original
    for item in changes:
        require(set(item) == {"label", "before", "after"}, "插桩记录仅有完整三元组")
        label, before, after = (item[key] for key in ("label", "before", "after"))
        if label == "shared_header":
            require(before == ("namespace solver {" if unit == "ipm" else "namespace solver::detail {"), "头文件唯一命名空间锚点")
            desired = '#include "v47-profile.hpp"\n\n' + before
        elif label in scopes:
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
            desired = f"  {{\n    solver_v47_profile::Scope v47_gram_scope({stages.index('gram_assembly')});\n" + before + "  }\n"
        else:
            require(label == "cholmod_values_copy" and "lower_values_[destination] = values[static_cast<std::size_t>(lower_sources_[destination])];" in before and
                    "if ((destination & 4095) == 0) check(budget_check);" in before, "下三角复制与原预算检查均保留")
            desired = f"    {{\n    solver_v47_profile::Scope v47_copy_scope({stages.index(label)});\n" + before + "    }\n"
        require(after == desired and rebuilt.count(before) == 1, "独立核对只添加计时包装：" + label)
        rebuilt = rebuilt.replace(before, after, 1)
    restored = rebuilt
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆变换唯一")
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "独立逆变换完整恢复生产源码")
    return rebuilt


def build_sources(plan, identity, legacy):
    manifest_path = WORK / "build-manifest.json"
    build = read(manifest_path)
    require(build["diagnostic_only"] is True and build["plan_sha256"] == PLAN_SHA and build["frozen_files"] == identity["files"] and
            len(identity["files"]) == 95 and len(identity["libraries"]) == 17, "两TU诊断保持全部生产来源与库")
    for group in ("files", "libraries"):
        for name, digest in identity[group].items():
            require(sha(ROOT / name) == digest, "生产来源或库改变：" + name)
    production = ROOT / ".tools/solver-ipm-v54.exe"
    diagnostic = WORK / "solver-profile.exe"
    require(Path(build["production_executable"]) == production and Path(build["executable"]) == diagnostic and
            sha(production) == sha(ROOT / "build/release/solver.exe") == build["production_sha256"] == identity["solver_sha256"] and
            sha(diagnostic) == build["executable_sha256"] != build["production_sha256"], "实际双程序身份")
    require(sha(identity["compiler"]) == identity["compiler_sha256"] and build["implicit_link_inputs_verified"] is False, "编译器与隐式库保证边界")
    archive_path = DATA / "ipm-v54-source.zip"
    require(sha(archive_path) == identity["source_zip_sha256"], "冻结源码包身份")
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())) == 95 and set(archive.namelist()) == set(identity["files"]), "源码包唯一成员集合")
        sources = {name: archive.read(name) for name in archive.namelist()}
    for name, content in sources.items():
        require(hashlib.sha256(content).hexdigest() == identity["files"][name] == sha(WORK / "frozen" / name), "生产/独立冻结成员身份")
    for group in ("protected_before_and_after", "outputs", "link_explicit_before_and_after", "compile_inputs_after_and_final"):
        require(isinstance(build[group], dict) and build[group], "构建证据集合非空")
        for name, digest in build[group].items():
            require(Path(name).is_absolute() and sha(name) == digest, "构建证据实际身份：" + name)
    require(set(build["outputs"]) == {str(path) for path in WORK.iterdir() if path.is_file() and path != manifest_path}, "所有直接构建产物完整冻结，清单不自包含")
    protected = build["protected_before_and_after"]
    for path in (DATA / "v55-build.py", DATA / "v55-run.py", DATA / "v55-runner-self-check.json", DATA / "v55-instrumentation.py", DATA / "v55-plan.json",
                 DATA / "v55-production-precheck.json", DATA / "v47-profile-build.py", ROOT / "build/release/build.ninja",
                 ROOT / "build/release/CMakeFiles/rules.ninja", production, ROOT / "build/release/solver.exe", Path(identity["compiler"])):
        require(protected.get(str(path)) == sha(path), "关键构建输入前后冻结")
    require(protected[str(DATA / "v55-instrumentation.py")] == INSTRUMENTATION_SHA, "独立静态复审的插桩来源")
    helper = load_module(DATA / "v55-instrumentation.py", "v55_instrumentation_audit", INSTRUMENTATION_SHA)
    stages = plan["profile_stages"]
    expected_header = legacy.HEADER.replace(
        "inline constexpr const char* names[] = {" + ",".join(json.dumps(name) for name in legacy.STAGES) + "};",
        "inline constexpr const char* names[] = {" + ",".join(json.dumps(name) for name in stages) + "};", 1).replace(
        "inline thread_local std::array<Entry, 26> entries{};", "inline thread_local std::array<Entry, 32> entries{};", 1)
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
    precheck = read(DATA / "v55-production-precheck.json")
    original = {unit: legacy.split_windows(precheck["original_compile_commands"]["normal_cholesky" if unit == "normal" else unit][0])
                for unit in ("ipm", "normal")}
    require(commands["original_compile"] == original and hashlib.sha256((WORK / "ninja-commands.txt").read_text(encoding="utf-8").encode()).hexdigest() ==
            precheck["ninja_commands_sha256"], "原始编译参数对应事前真实Ninja来源")
    outer = legacy.split_windows(identity["link_command"])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "真实链接包装")
    original_link = legacy.split_windows(outer[2][len("cd . && "):-len(" && cd .")])
    require(commands["original_link"] == original_link, "冻结实际链接序列")
    regularized = cwd / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    require(sha(regularized) == precheck["object_identity"]["regularized_ldl.cpp.obj"]["sha256"] and
            protected[str(regularized)] == sha(regularized), "保留原regularized对象")
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
        require(strict_json(row["stdout"]) == {"passed": True, "enabled": enabled}, "双TU假时钟自测实际结果")
        payload = profile(row["stderr"], enabled, stages)
        if enabled:
            for name, values in (("total", (110, 20, 1, 1, 0)), ("canonicalize", (110, 50, 3, 2, 1)), ("normal_setup", (60, 40, 2, 1, 1))):
                require(payload["stages"][name] == dict(zip(FIELDS, values)), "独立假时钟黄金计数与纳秒")
            require(all(entry == dict.fromkeys(FIELDS, 0) for name, entry in payload["stages"].items() if name not in ("total", "canonicalize", "normal_setup")), "其他假时钟阶段均零")
        else:
            require(row["stderr"] == "", "关闭自测无诊断输出")
    require(read(DATA / "v55-runner-self-check.json") == {"passed": True, "trajectory_cases": 5, "profile_positive_cases": 2,
            "profile_negative_cases": 9, "solver_runs": 0}, "运行器实际纯合成自检")
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
        "runner_self_check_protected_at_build": str(DATA / "v55-runner-self-check.json") in protected}


def verify_runs(plan, identity, build, benchmark, compare):
    manifest = read(DATA / "v55-run-manifest.json")
    build_path = WORK / "build-manifest.json"
    protected = {str(DATA / "v55-run.py"): sha(DATA / "v55-run.py"), str(build_path): sha(build_path),
        build["production_executable"]: build["production_sha256"], build["executable"]: build["executable_sha256"]}
    protected.update({str(ROOT / name): digest for group in ("files", "libraries") for name, digest in identity[group].items()})
    require(manifest["plan_sha256"] == PLAN_SHA and Path(manifest["build_manifest"]) == build_path and Path(manifest["work"]) == WORK and
            manifest["protected"] == protected and manifest["schedule"] == plan["order"], "执行清单的计划、构建、双程序与生产来源链")
    require(manifest["environment"] == dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1") and
            manifest["removed_environment"] == "清除所有SOLVER_V开头变量及OPENBLAS_CORETYPE；仅on设置冻结计时开关", "共同单线程和唯一诊断开关")
    require(datetime.datetime.fromisoformat(build["created_at_utc"]) <= datetime.datetime.fromisoformat(manifest["started_at_utc"]), "先完成独立构建再运行")
    require(all(sha(path) == digest for path, digest in protected.items()), "运行保护集合当前仍保持身份")
    options = load_module(DATA / "v54-verify.py", "v55_frozen_options_audit", OPTION_HELPER_SHA)
    raw_path = DATA / "v55-runs.jsonl.gz"
    rows = [strict_json(line) for line in gzip.decompress(raw_path.read_bytes()).splitlines()]
    schedule = [(model, mode) for model in plan["models"] for mode in
                MODES[model["model_index"] % 5:] + MODES[:model["model_index"] % 5]]
    require(len(rows) == len(schedule) == 40, "事前40条全部保留")
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
            "--ipm-linear-solver", "normal_cholmod", "--ipm-direction-retries", "0", "--ipm-objective-scaling", "off"]
        for key, value in plan["shared_ipm"].items():
            expected.extend(["--ipm-" + key, value])
        require(row["command"] == expected, "逐次真实命令与原数值政策一致")
        require(row["hashes_before"] == row["hashes_after"] == {"input": model["sha256"], "solver": digest} and
                row["model_sha256"] == model["sha256"], "逐次输入与本侧程序前后身份")
        require(finite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] > 0, "真实墙钟为有限正数")
        report = row["report"]
        if report is not None:
            require(isinstance(report, dict) and strict_json(row["stdout"]) == report and report["input"] == model["input"] and
                    row["classification"] == benchmark.classify(row["returncode"], report), "原始JSON报告、输入及真实分类")
            options.verify_options(report, plan, "ipm", 54, "off")
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
        require(row["numerical_seconds"] == numerical == compare.numerical_seconds(row) and
                row["quality_failure"] == failure and row["quality_pass"] is (failure is None), "逐条原模型质量及完整预算重算")
        try:
            parsed = profile(row["stderr"], mode.startswith("on"), plan["profile_stages"])
        except (AssertionError, ValueError, KeyError, TypeError) as error:
            require(row["profile"] is None and isinstance(row["profile_error"], str) and row["profile_error"], "采集失败不能伪造完整账本")
            profile_failures.append({"index": index, "model": row["model"], "variant": mode, "reason": str(error)})
        else:
            require(row["profile"] == parsed and row["profile_error"] is None, "独立解析stderr并重算全部计时账本")
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
    summary = {"diagnostic_only": True, "runs": 40, "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "profile_errors": len(profile_failures),
        "scope": "仅诊断成本；阶段中位数分别计算不承诺加总守恒，不作为公开性能或新覆盖。", "raw_sha256": sha(raw_path)}
    require(json.dumps(summary, sort_keys=True, allow_nan=False) == json.dumps(read(DATA / "v55-summary.json"), sort_keys=True, allow_nan=False),
            "从全部raw独立重建完整摘要和每例三次中位数，严格区分bool/int")
    return summary, {"missing_reports": missing, "invalid_process_reports": invalid, "profile_failures": profile_failures,
        "enabled_profiles": sum(row["profile"] is not None for row in rows),
        "quality_pass_by_variant": {mode: sum(row["quality_pass"] for row in rows if row["variant"] == mode) for mode in MODES}}


def verify():
    plan, identity, legacy = fixed_plan(check_inputs=True)
    build, benchmark, compare, sources = build_sources(plan, identity, legacy)
    summary, records = verify_runs(plan, identity, build, benchmark, compare)
    passed = summary["invalid_pairs"] == summary["different_pairs"] == summary["profile_errors"] == 0 and not records["missing_reports"] and not records["invalid_process_reports"]
    return {"passed": passed, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "scope": "40条冻结运行、32对完整轨迹、24条on计时账本与8组阶段中位数的独立核验",
        "plan_sha256": PLAN_SHA, "verifier_sha256": sha(__file__), "helper_sha256": HELPER_SHA,
        "instrumentation_sha256": INSTRUMENTATION_SHA, "option_verifier_sha256": OPTION_HELPER_SHA,
        "parent_identity_sha256": PARENT_SHA, "production_sha256": identity["solver_sha256"], "diagnostic_sha256": build["executable_sha256"],
        "source_checks": sources, "run_count": 40, "pairs": {key: summary[key] for key in
            ("non_budget_pairs", "non_budget_equal", "budget_pairs", "invalid_pairs", "different_pairs", "profile_errors")},
        "record_checks": records, "cases": summary["cases"], "self_test": self_test(plan, identity),
        "artifacts": {name: sha(DATA / name) for name in ("v55-build.py", "v55-run.py", "v55-run-manifest.json", "v55-runs.jsonl.gz",
            "v55-summary.json", "v55-runner-self-check.json")}, "build_manifest_sha256": sha(WORK / "build-manifest.json"),
        "limits": ["不运行LP或重构原矩阵；重新检查冻结报告、真实命令与共同质量合同。",
            "仅剔timings比较轨迹；合法预算分叉单列，缺报告和异常退出不获豁免。",
            "每条原始计时账本严格守恒；分别取各阶段中位数后的合计不要求守恒，包含时间与子阶段不能重复相加。",
            "复用v47计时核心；当前CHOLMOD纯C调用使用新增标签，旧numeric/triangular标签为零不代表未计量。",
            "计时覆盖solve_ipm作用域及局部析构，不含输入、JSON、Report初始化与诊断输出，不能要求与生产timings逐纳秒相等。",
            "仅核显式链接输入与依赖文件列举的编译依赖；隐式运行库未逐项核验。",
            "本轮为8个已观察输入的成本诊断，不新增公开性能或覆盖成绩，不把单次off与production差异称为统计显著开销。"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight or args.self_test:
        plan, identity, legacy = fixed_plan()
        result = {"passed": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "plan_sha256": PLAN_SHA, "helper_sha256": HELPER_SHA, "verifier_sha256": sha(__file__),
                  "scope": "只核冻结计划、纯合成轨迹及计时账本，不读取性能raw或大库", "self_test": self_test(plan, identity)}
    else:
        result = verify()
    destination = args.output
    if destination is None and not (args.preflight or args.self_test):
        destination = DATA / "v55-verification.json"
    if destination is not None:
        with destination.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "run_count", "pairs", "record_checks", "source_checks", "self_test") if key in result}, ensure_ascii=False, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)
