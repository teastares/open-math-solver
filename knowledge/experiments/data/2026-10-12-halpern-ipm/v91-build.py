"""从冻结v66独立构建预处理与同Cone warm组合诊断，仅替换IPM翻译单元。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "55ca339ebb9100e626b62080e95e9cd15ce0c2673b23be8727944920b96b68cd"
BASE_PATH = DATA / "v47-profile-build.py"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
PARENT_INST_SHA = "c2d9c678a219c7d1737900ab1aa334405b6977c1f440833d739cdb1d2a3d2019"
PARENT_PLAN_SHA = "8a47239e7f9cf412b2f64eb604eda9ab25c8143a11e4075335b1011ccde3964f"
CLARIFICATION_SHA = "b0cf18b7591c124ce5229c5a4671b0028b059ff6a5617218df736b97d9892906"
REQUIRED_COMBINATION_CASES = {
    "warm_reduced_original_oracle": "transaction",
    "warm_oracle_capacity_stop": "budget",
    "warm_accept_abort_is_fail_closed": "exception",
    "warm_local_verification_counted_once": "transaction",
    "warm_not_reduced_and_disabled_scope": "transaction",
}
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == HELPER_SHA
BASE = runpy.run_path(str(BASE_PATH))
sha, require = (BASE[key] for key in ("sha", "require"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "JSON重复键：" + key)
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(-float("inf") < value < float("inf"), "JSON浮点溢出")
        return value
    def invalid(text):
        raise ValueError("JSON非有限数：" + text)
    return json.loads(text, object_pairs_hook=unique, parse_float=finite, parse_constant=invalid)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def helper():
    return runpy.run_path(str(DATA / "v91-ipm-instrumentation.py"))


def verified_transform(instrument, source):
    patched, changes = instrument["transform"](source)
    require(isinstance(changes, list) and changes and len({item["label"] for item in changes}) == len(changes), "窄hook变换须有唯一标识")
    restored = patched
    for item in reversed(changes):
        require(set(item) == {"label", "before", "after"} and item["before"] != item["after"] and restored.count(item["after"]) == 1,
            "静态逆变换锚点缺失或重复：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == source.replace("\r\n", "\n"), "所有新hook须可逐字逆恢复冻结原始来源")
    return patched, changes


def static_check(root=ROOT):
    require(sha(DATA / "v91-plan.json") == PLAN_SHA, "冻结计划变化")
    require(sha(DATA / "v91-pretest-clarification.json") == CLARIFICATION_SHA, "测前组合事务澄清变化")
    require(sha(DATA / "v88-ipm-instrumentation.py") == PARENT_INST_SHA, "只读父插桩来源变化")
    plan = read(DATA / "v91-plan.json")
    require(plan["warm_kernel_file"] == "v84-warm-pdhg.hpp" and
        sha(DATA / plan["warm_kernel_file"]) == plan["warm_kernel_sha256"] and
        sha(DATA / plan["warm_source_test_file"]) == plan["warm_source_test_sha256"] and
        (DATA / "v91-warm-self-test.cpp").read_bytes() == (DATA / plan["warm_source_test_file"]).read_bytes(),
        "warm内核及原七组隔离测试须逐字复用冻结v84来源")
    instrument = helper()
    checks = instrument["static_check"](root)
    require(checks["passed"] is True and checks.get("reversible") is True and checks.get("shared_start") is True,
        "预处理插桩静态检查与共享原始时钟合同失败")
    for field in ("core_body_preserved_except_candidate_hooks", "assess_hook_reversible", "current_local_saved_before_move",
        "parent_oracle_hooks_preserved", "warm_core_loop_preserved", "warm_initialization_excluded", "warm_kernel_preserved",
        "warm_isolated_local_assessment", "warm_verification_not_double_counted", "warm_accept_abort_fail_closed"):
        require(checks.get(field) is True, "预处理与warm组合静态证明缺失：" + field)
    require(checks.get("original_core_body_preserved") is not True and checks.get("core_body_preserved_except_assess_hook") is not True,
        "新增warm hook后不能声称只改assess或整个核心逐字无变")
    _, changes = verified_transform(instrument, (root / "src/ipm.cpp").read_text(encoding="utf-8"))
    require(checks["anchors"] == len(changes), "静态报告锚点数须等于实际可逆变换")
    return {"passed": True, "public_lp_runs": 0, "ipm": checks}


def private_compile(original, source, obj, frozen, output, highs_includes):
    command = compile_arguments(original, source, obj, frozen, ROOT)
    command.insert(1, "-I" + str(output))
    # 仅搬入实际HiGHS头搜索路径；保持原IPM浮点编译选项与宏。
    command[command.index("-c"):command.index("-c")] = highs_includes
    return command


def dependencies(obj, build_dir, required, collected):
    paths = dependency_paths(Path(str(obj) + ".d"), build_dir)
    require({path.resolve() for path in required} <= {path.resolve() for path in paths}, "实际.d未覆盖私有来源/头")
    for path, digest in file_hashes(paths).items():
        require(path not in collected or collected[path] == digest, "编译期间依赖变化：" + path)
        collected[path] = digest


def explicit_inputs(command, cwd):
    return file_hashes([resolve(value, cwd) for value in command if not value.startswith("-") and
                       Path(value).suffix.lower() in (".a", ".lib", ".obj", ".o")])


def inspect_math(raw):
    require(raw["stderr"] == "", "预处理组件数学自测不得有未解释stderr")
    details = strict_json(raw["stdout"])
    require(isinstance(details, dict) and details.get("passed") is True and
        type(details.get("public_lp_runs")) is int and details["public_lp_runs"] == 0 and
        type(details.get("explicit_highs_lp_solve_calls")) is int and details["explicit_highs_lp_solve_calls"] == 0 and
        type(details.get("component_ipm_calls")) is int and details["component_ipm_calls"] > 0 and
        type(details.get("cases")) is int and details["cases"] > 0 and
        isinstance(details.get("details"), list) and len(details["details"]) == details["cases"],
        "预处理/自主核心/原模型恢复组件数学自测摘要无效")
    groups = ("wrapper", "mapping", "transaction", "budget", "exception")
    require(set(details) == {"passed", "public_lp_runs", "explicit_highs_lp_solve_calls", "scope", "cases", "component_ipm_calls",
        "synthetic_core_calls", "details", *(group + "_cases" for group in groups)} and
        isinstance(details["scope"], str) and details["scope"] and type(details["synthetic_core_calls"]) is int and details["synthetic_core_calls"] > 0 and
        all(type(details[group + "_cases"]) is int and details[group + "_cases"] > 0 for group in groups) and
        sum(details[group + "_cases"] for group in groups) == details["cases"], "数学组件覆盖字段与分组计数不守恒")
    ids, counts = set(), dict.fromkeys(groups, 0)
    for item in details["details"]:
        require(isinstance(item, dict) and set(item) == {"id", "group"} and isinstance(item["id"], str) and item["id"] and
            item["id"].isascii() and item["id"] not in ids and type(item["group"]) is str and item["group"] in groups, "数学组件详情标识或分组无效")
        ids.add(item["id"])
        counts[item["group"]] += 1
    require(all(counts[group] == details[group + "_cases"] for group in groups), "数学组件详情须与实际分项完成计数一致")
    return details


def strict_warm_limits(value, expected):
    return isinstance(value, dict) and set(value) == set(expected) and all(
        type(value[key]) is type(expected[key]) and value[key] == expected[key] for key in expected)


def warm_math_tests(context):
    output, compiler, build_dir, env = (context[key] for key in ("output", "compiler", "build_dir", "environment"))
    source = output / "warm-self.cpp"
    # 引号include先搜索源码目录；私有源码副本确保实际编译私有kernel，而不是DATA的原header。
    source.write_bytes((DATA / "v91-warm-self-test.cpp").read_bytes())
    context["generated"].update(file_hashes([source]))
    obj, executable = output / "warm-self.obj", output / "warm-self-test.exe"
    command = [str(compiler), "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
        "-MMD", "-MF", str(obj) + ".d", "-I" + str(output), "-c", str(source), "-o", str(obj)]
    execute(command, build_dir, output / "warm-self-build.json", env)
    dependencies(obj, build_dir, [source, output / "v84-warm-pdhg.hpp"], context["compiled"])
    link = [str(compiler), "-static", str(obj), "-o", str(executable)]
    context["test_inputs"].update(explicit_inputs(link, build_dir))
    execute(link, build_dir, output / "warm-self-link.json", env)
    raw = execute([str(executable)], ROOT, output / "warm-self-run.json", env)
    details = strict_json(raw["stdout"])
    require(raw["stderr"] == "" and isinstance(details, dict) and
        set(details) == {"passed", "public_lp_runs", "cases", "default_limits", "details"} and details["passed"] is True and
        type(details["public_lp_runs"]) is int and details["public_lp_runs"] == 0 and
        type(details["cases"]) is int and details["cases"] == 7 and isinstance(details["details"], list) and
        len(details["details"]) == details["cases"] and all(type(item) is str and item for item in details["details"]) and
        len(set(details["details"])) == details["cases"] and strict_warm_limits(details["default_limits"], context["plan"]["warm_limits"]),
        "原warm七组、唯一完成详情或真实默认限制不符合冻结计划")
    save(output / "warm-self-check.json", {"passed": True, "public_lp_runs": 0, "details": details})
    context["test_commands"]["warm"] = {"compile": command, "link": link, "run": [str(executable)]}
    return {**details, "run_file": "warm-self-run.json", "check_file": "warm-self-check.json"}


def math_extension(details, plan, required_new_cases):
    predecessor_path = DATA / plan["math_evidence_file"]
    require(sha(predecessor_path) == plan["math_evidence_sha256"], "数学覆盖父证据变化")
    predecessor = read(predecessor_path)
    checks = predecessor["math_self_checks"]
    require(predecessor["passed"] is True and predecessor["plan_sha256"] == PARENT_PLAN_SHA and checks["passed"] is True and
        type(checks["public_lp_runs"]) is int and checks["public_lp_runs"] == 0,
        "父数学组件原件须已通过且不含公开LP调用")
    old = checks["details"]
    old_cases = {item["id"]: item["group"] for item in old["details"]}
    new_cases = {item["id"]: item["group"] for item in details["details"]}
    require(type(old["cases"]) is int and old["cases"] == 58 == len(old["details"]) == len(old_cases) and
        old_cases.items() <= new_cases.items(), "须保留最终v88的58组数学测试，不能误取v87父证据")
    require(isinstance(required_new_cases, dict) and required_new_cases and
        not (required_new_cases.keys() & old_cases.keys()) and
        required_new_cases.items() <= new_cases.items(), "新增组合事务测试须真实完成且标识独立")
    call_counts = {}
    for name in ("component_ipm_calls", "synthetic_core_calls"):
        require(type(old[name]) is int and old[name] > 0, "父组件入口计数无效：" + name)
        # 新停止路由可能改变真实组件调用数；保留差值供归因，不凭计数大小冒充数学覆盖。
        call_counts[name] = {"predecessor": old[name], "current": details[name], "change": details[name] - old[name]}
    added = [item for item in details["details"] if item["id"] not in old_cases]
    require(added and details["cases"] == old["cases"] + len(added), "新增数学覆盖计数须按真实唯一详情守恒")
    return {"passed": True, "public_lp_runs": 0,
        "predecessor_evidence_file": str(predecessor_path), "predecessor_evidence_sha256": plan["math_evidence_sha256"],
        "preserved_predecessor_cases": old["cases"], "new_cases": added,
        "required_new_cases": [{"id": name, "group": group} for name, group in required_new_cases.items()],
        "component_calls": call_counts}


def math_tests(context):
    output, build_dir, env = (context[key] for key in ("output", "build_dir", "environment"))
    warm_details = warm_math_tests(context)
    ipm_source, ipm_obj, ipm_exe = DATA / "v91-ipm-self-test.cpp", output / "ipm-self.obj", output / "ipm-self-test.exe"
    ipm_command = private_compile(context["originals"]["ipm"], ipm_source, ipm_obj, context["frozen"], output, context["highs_includes"])
    execute(ipm_command, build_dir, output / "ipm-self-build.json", env)
    dependencies(ipm_obj, build_dir, [output / "ipm-presolve.cpp", output / "v91-presolve-state.hpp",
        output / "v91-warm-state.hpp", output / "v84-warm-pdhg.hpp",
        build_dir / "_deps/highs-src/highs/Highs.h"], context["compiled"])
    original_main = context["originals"]["main"]
    main_obj = original_main[original_main.index("-o") + 1]
    ipm_link, replacements = [], {"harness": 0, "ipm": 0}
    for value in context["original_link"]:
        if value == main_obj:
            ipm_link.append(str(ipm_obj)); replacements["harness"] += 1
        elif value == "libsolver_ipm.a":
            ipm_link.append(str(context["regularized"])); replacements["ipm"] += 1
        elif value != "-municode" and not value.startswith("-Wl,--out-implib,"):
            ipm_link.append(value)
    require(replacements == {"harness": 1, "ipm": 1} and str(output / "ipm-presolve.obj") not in ipm_link and
        "libsolver_ipm.a" not in ipm_link, "包含式IPM测试不能重复链接候选定义")
    ipm_link[ipm_link.index("-o") + 1] = str(ipm_exe)
    context["test_inputs"].update(explicit_inputs(ipm_link, build_dir))
    execute(ipm_link, build_dir, output / "ipm-self-link.json", env)
    raw = execute([str(ipm_exe)], ROOT, output / "ipm-self-run.json", env)
    details = inspect_math(raw)
    save(output / "ipm-self-check.json", {"passed": True, "public_lp_runs": 0, "details": details})
    extension = math_extension(details, context["plan"], REQUIRED_COMBINATION_CASES)
    extension_path = output / "ipm-self-extension-check.json"
    save(extension_path, extension)
    context["math_coverage_extension"] = {"file": extension_path.name, "sha256": sha(extension_path),
        "preserved_predecessor_cases": extension["preserved_predecessor_cases"], "new_cases": len(extension["new_cases"])}
    context["test_commands"]["ipm"] = {"compile": ipm_command, "link": ipm_link, "run": [str(ipm_exe)]}
    return {"warm": warm_details, "ipm": {**details, "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"}}

def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立输出须位于工作区且尚不存在")
    plan_path = DATA / "v91-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    parent_path = DATA / "ipm-v66-identity.json"
    require(sha(parent_path) == plan["parent_identity_sha256"], "冻结v66身份变化")
    parent = read(parent_path)
    verification_path, evidence_path = (DATA / plan[key] for key in ("baseline_verification_file", "evidence_file"))
    math_evidence_path, warm_evidence_path = (DATA / plan[key] for key in ("math_evidence_file", "warm_evidence_file"))
    warm_kernel_path, warm_source_test_path = (DATA / plan[key] for key in ("warm_kernel_file", "warm_source_test_file"))
    require(sha(verification_path) == plan["baseline_verification_sha256"] and
            sha(evidence_path) == plan["evidence_sha256"] and sha(math_evidence_path) == plan["math_evidence_sha256"] and
            sha(warm_evidence_path) == plan["warm_evidence_sha256"] and sha(warm_kernel_path) == plan["warm_kernel_sha256"] and
            sha(warm_source_test_path) == plan["warm_source_test_sha256"], "基线、诊断、数学父或原warm来源发生变化")
    baseline = read(verification_path)
    require(baseline["files"] == parent["files"] and baseline["libraries"] == parent["libraries"] and
            len(parent["files"]) == 95 and len(parent["libraries"]) == 17, "只冻结原v66的95来源和17库")
    production, reference = ROOT / "build/release/solver.exe", ROOT / ".tools/solver-ipm-v66.exe"
    source_archive = DATA / "ipm-v66-source.zip"
    require(sha(production) == sha(reference) == baseline["solver_sha256"] == parent["solver_sha256"] == plan["parent_solver_sha256"] and
            sha(source_archive) == parent["source_zip_sha256"] == plan["parent_source_sha256"], "生产/冻结程序及原源码包变化")
    require(sha(BASE_PATH) == HELPER_SHA, "通用构建辅助器变化")
    runner_check = read(DATA / "v91-runner-self-check.json")
    runner_runs = runner_check.get("public_lp_runs", runner_check.get("solver_runs", runner_check.get("lp_runs")))
    require(runner_check.get("passed") is True and type(runner_runs) is int and runner_runs == 0, "运行器纯数据自检尚未通过")
    sources = {str(ROOT / name): digest for name, digest in parent["files"].items()}
    libraries = {str(ROOT / name): digest for name, digest in parent["libraries"].items()}
    require_hashes(sources); require_hashes(libraries)
    compiler, build_dir = Path(parent["compiler"]), ROOT / "build/release"
    require(sha(compiler) == parent["compiler_sha256"], "编译器变化")
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"], text=True, encoding="utf-8")
    lines, originals = command_text.splitlines(), {}
    for role, relative in (("ipm", "src/ipm.cpp"), ("main", "src/main.cpp"), ("highs", "src/highs_backend.cpp")):
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "原始编译模板不唯一")
        originals[role] = matches[0]
    original_includes = {resolve(value[2:], ROOT) for value in originals["ipm"] if value.startswith("-I")}
    highs_includes = [value for value in originals["highs"] if value.startswith("-I") and
        resolve(value[2:], ROOT) != ROOT / "include" and resolve(value[2:], ROOT) not in original_includes]
    require(highs_includes and len(highs_includes) == len(set(highs_includes)), "上游HiGHS头路径须存在且唯一")
    links = [line for line in lines if " -o solver.exe " in line]
    require(links == [parent["link_command"]], "原链接序列变化")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "原链接包装变化")
    original_link = split_windows(outer[2][len("cd . && "):-len(" && cd .")])
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    explicit = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".obj", ".o")]
    new_files = [DATA / name for name in ("v91-ipm-instrumentation.py", "v91-ipm-self-test.cpp", "v91-warm-self-test.cpp",
        "v91-run.py", "v91-runner-self-check.json", "v66-run.py", "v88-ipm-instrumentation.py")]
    protected = file_hashes([production, reference, parent_path, source_archive, verification_path, evidence_path,
        math_evidence_path, warm_evidence_path, warm_kernel_path, warm_source_test_path,
        DATA / "v91-pretest-clarification.json",
        plan_path, BASE_PATH, Path(__file__), compiler, ninja, regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja",
        *explicit, *new_files])
    require_hashes(plan["live_highs_sources"])
    protected.update(plan["live_highs_sources"])
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent["files"]), "原源码包成员变化")
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), "原源码包路径越界")
        archive.extractall(frozen)
    frozen_files = {str(frozen / name): digest for name, digest in parent["files"].items()}
    require_hashes(frozen_files)
    instrument = helper()
    save(output / "static-check.json", static_check(frozen))
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    patched, changes = verified_transform(instrument, (frozen / "src/ipm.cpp").read_text(encoding="utf-8"))
    source = output / "ipm-presolve.cpp"
    source.write_text(patched, encoding="utf-8")
    private_header = output / "v91-presolve-state.hpp"
    private_header.write_text(instrument["HEADER"], encoding="utf-8")
    private_warm_header = output / "v91-warm-state.hpp"
    private_warm_header.write_text(instrument["WARM_HEADER"], encoding="utf-8")
    kernel_copy = output / "v84-warm-pdhg.hpp"
    kernel_copy.write_bytes(warm_kernel_path.read_bytes())
    require(sha(kernel_copy) == plan["warm_kernel_sha256"], "私有warm内核副本须保持原字节与命名空间")
    generated = file_hashes([source, private_header, private_warm_header, kernel_copy])
    save(output / "transforms.json", {"source": str(frozen / "src/ipm.cpp"), "source_sha256": sha(frozen / "src/ipm.cpp"),
        "patched": str(source), "patched_sha256": sha(source), "changes": changes})
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    environment.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    obj, compiled = output / "ipm-presolve.obj", {}
    command = private_compile(originals["ipm"], source, obj, frozen, output, highs_includes)
    execute(command, build_dir, output / "ipm-compile.json", environment)
    dependencies(obj, build_dir, [source, private_header, private_warm_header, kernel_copy,
        build_dir / "_deps/highs-src/highs/Highs.h"], compiled)
    link, replacements = [], 0
    for value in original_link:
        if value == "libsolver_ipm.a":
            link.extend([str(obj), str(regularized)]); replacements += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replacements == 1 and link.count("libsolver_normal_cholesky.a") == original_link.count("libsolver_normal_cholesky.a") == 1,
            "只唯一替换IPM库，Normal及其他库保持原序")
    executable = output / "solver-presolve.exe"
    link[link.index("-o") + 1] = str(executable)
    link_inputs = explicit_inputs(link, build_dir)
    execute(link, build_dir, output / "link.json", environment)
    context = dict(output=output, frozen=frozen, compiler=compiler, build_dir=build_dir, environment=environment,
        originals=originals, original_link=original_link, regularized=regularized, generated=generated,
        compiled=compiled, test_inputs={}, test_commands={}, highs_includes=highs_includes, plan=plan)
    tests = math_tests(context)
    for mapping in (protected, sources, libraries, frozen_files, generated, compiled, link_inputs, context["test_inputs"]): require_hashes(mapping)
    outputs = file_hashes([path for path in output.iterdir() if path.is_file()])
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "production_changes": False,
        "experimental_candidate_only": True, "plan_sha256": PLAN_SHA, "parent_identity_sha256": plan["parent_identity_sha256"],
        "production_executable": str(production), "production_sha256": sha(production),
        "frozen_parent_executable": str(reference), "frozen_parent_sha256": sha(reference),
        "executable": str(executable), "executable_sha256": sha(executable),
        "baseline_verification_file": str(verification_path), "baseline_verification_sha256": sha(verification_path),
        "frozen_files": parent["files"], "current_files": baseline["files"], "actual_libraries": baseline["libraries"],
        "private_state_header": str(private_header),
        "private_warm_state_header": str(private_warm_header),
        "warm_kernel": {"source": str(warm_kernel_path), "source_sha256": sha(warm_kernel_path),
            "private_copy": str(kernel_copy), "private_copy_sha256": sha(kernel_copy), "namespace_renamed": False},
        "readonly_parent_instrumentation": {"path": str(DATA / "v88-ipm-instrumentation.py"), "sha256": PARENT_INST_SHA},
        "protected_before_and_after": protected, "generated_inputs_before_and_after": generated,
        "compile_inputs_after_and_final": compiled, "link_explicit_before_and_after": link_inputs,
        "test_link_explicit_before_and_after": context["test_inputs"], "outputs": outputs, "math_self_tests": tests,
        "math_coverage_extension": context["math_coverage_extension"],
        "commands": {"cwd": str(build_dir), "original_compile": originals, "original_link": original_link,
                     "highs_include_arguments": highs_includes, "compile": {"ipm": command}, "link": link,
                     "math_self_tests": context["test_commands"]},
        "implicit_link_inputs_verified": False, "public_lp_runs": 0}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"passed": True, "compiled_units": ["ipm"], "solver_sha256": manifest["executable_sha256"],
                      "math_self_tests": tests, "public_lp_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v91-presolve")
    parser.add_argument("--self-test", action="store_true", help="仅锁定计划和插桩静态边界，不编译或运行LP")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build((ROOT / arguments.output).resolve())
