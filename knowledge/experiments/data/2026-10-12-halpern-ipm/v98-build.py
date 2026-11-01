"""从冻结v66构建forcing一致搜索尺度HSD候选与两套组件，仅替换IPM翻译单元。"""
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
PLAN_SHA = "a66d1fc30ba2cc8eb2a9a56fe1381801b6f73297fabc9e2cb8bd6ae1e2f219a2"
BASE_PATH = DATA / "v47-profile-build.py"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
PARENT_INST_SHA = "611148324db4fda3c3c33d450094841c889a8ed70643c90e472f5b6f37bbf4be"
PARENT_PLAN_SHA = "d9badff6f967973e5d5b44ceee2581fa4b3b74d3b1cb42442eb833015b9f9048"
CLARIFICATION_SHA = "be8dc02408f8d3b5415ec6795276a8feca186cbed742dacd9b58e62eded3b1b8"
REQUIRED_COMBINATION_CASES = {
    "forcing_weights_real_hsd": "transaction",
    "forcing_weights_default_mapping": "mapping",
    "forcing_weights_budget": "budget",
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
    return runpy.run_path(str(DATA / "v98-ipm-instrumentation.py"))


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
    require(sha(DATA / "v98-plan.json") == PLAN_SHA, "冻结计划变化")
    require(sha(DATA / "v92-pretest-clarification.json") == CLARIFICATION_SHA, "测前联立残差与状态澄清变化")
    require(sha(DATA / "v97-ipm-instrumentation.py") == PARENT_INST_SHA, "只读父插桩来源变化")
    plan = read(DATA / "v98-plan.json")
    require(plan["coupled_kernel_file"] == "v98-coupled.hpp" and
        plan["parent_coupled_kernel_file"] == "v92-coupled.hpp" and
        sha(DATA / plan["parent_coupled_kernel_file"]) == plan["parent_coupled_kernel_sha256"] and
        plan["algorithm_parent_coupled_kernel_file"] == "v97-coupled.hpp" and
        sha(DATA / plan["algorithm_parent_coupled_kernel_file"]) == plan["algorithm_parent_coupled_kernel_sha256"] and
        plan["energy_kernel_file"] == "v93-energy.hpp" and
        sha(DATA / plan["energy_kernel_file"]) == plan["energy_kernel_sha256"] and
        plan["inherited_coupled_clarification_file"] == "v92-pretest-clarification.json" and
        plan["inherited_coupled_clarification_sha256"] == CLARIFICATION_SHA and
        plan["required_coupled_ipm_cases"] == REQUIRED_COMBINATION_CASES,
        "新宽方向核、原双精度测试核、逐字能量核和继承澄清须与冻结计划一致")
    instrument = helper()
    checks = instrument["static_check"](root)
    require(checks["passed"] is True and checks.get("reversible") is True and checks.get("shared_start") is True,
        "预处理插桩静态检查与共享原始时钟合同失败")
    for field in ("core_body_preserved_except_candidate_hooks", "assess_hook_reversible", "current_local_saved_before_move",
        "parent_oracle_hooks_preserved", "coupled_reversible", "coupled_role_from_callsites", "coupled_constant_is_direct_only",
        "coupled_physical_operator_preserved", "coupled_no_extra_ir",
        "coupled_disabled_preserves_old_path", "coupled_private_krylov_only",
        "energy_readonly_rows", "energy_existing_primal_regularization",
        "energy_no_old_denominator_rescan", "energy_parent_restorable", "energy_no_extra_linear_work",
        "energy_disabled_preserves_old_path", "wide_direction_return_preserved",
        "wide_double_column_wide_true_product", "wide_all_consumers_covered", "wide_no_direction_narrowing",
        "wide_point_late_narrowing", "wide_disabled_preserves_old_path", "wide_parent_restorable", "wide_no_extra_linear_work",
        "strict_or_explicit_forced_return", "initial_forcing_rhs_same_scan", "initial_forcing_public_after_delivery",
        "initial_forcing_no_repeat_initial_pa", "candidate_adapter_default_false", "candidate_production_explicit_true",
        "candidate_public_after_delivery", "forcing_weights_adapter_default_false", "forcing_weights_production_explicit_true",
        "parent_headers_preserved_after_rename", "parent_cpp_restored_after_two_weight_hooks"):
        require(checks.get(field) is True, "预处理与严格联立静态证明缺失：" + field)
    require(checks.get("original_core_body_preserved") is not True and checks.get("core_body_preserved_except_assess_hook") is not True,
        "新增联立hook后不能声称只改assess或整个核心逐字无变")
    _, changes = verified_transform(instrument, (root / "src/ipm.cpp").read_text(encoding="utf-8"))
    require(checks["anchors"] == len(changes) == 35 and checks["parent_anchors"] == 33 and
        checks["new_cpp_algorithm_anchors"] == checks["forcing_weights_extra_anchors"] == 2 and checks["inherited_wide_anchors"] == 8 and
        checks["coupled_no_inexact_return"] is False and checks["strict_or_explicit_forced_initial_return"] is False and
        checks["immediate_v97_cpp_preserved_after_rename"] is False,
        "实际CPP须为父33加搜索尺度两锚，不改三类线性返回政策")
    delta = checks["staticdelta_source"]
    require(delta["immediate_parent_version"] == 97 and delta["new_cpp_algorithm_anchors"] == 2 and delta["coupled_schema"] == 3 and
        delta["immediate_v97_cpp_preserved_after_rename"] is False and delta["coupled_no_inexact_return"] is False and
        delta["strict_or_explicit_forced_initial_return"] is False and
        all(delta[key] is True for key in ("parent_cpp_restored_after_two_weight_hooks", "parent_headers_preserved_after_rename", "parent_energy_preserved",
            "parent_wide_consumers_preserved", "parent_off_path_preserved", "strict_or_explicit_forced_return",
            "initial_forcing_rhs_same_scan", "forcing_weights_adapter_default_false", "forcing_weights_production_explicit_true", "candidate_public_after_delivery")) and
        checks["inherited_v97_static"]["passed"] is True and checks["inherited_v97_static"]["anchors"] == 33 and
        checks["inherited_v97_static"]["strict_or_explicit_forced_return"] is True,
        "父schema3/真实门槛与交付证明保持，当前只接入默认关闭的私有搜索尺度")
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


def strict_coupled_limits(value, expected):
    return isinstance(value, dict) and set(value) == set(expected) and all(
        type(value[key]) is type(expected[key]) and value[key] == expected[key] for key in expected)


def coupled_math_tests(context):
    output, compiler, build_dir, env = (context[key] for key in ("output", "compiler", "build_dir", "environment"))
    source = output / "coupled-self.cpp"
    # 引号include先搜索源码目录；私有源码副本确保实际编译私有kernel，而不是DATA的原header。
    source.write_bytes((DATA / "v98-coupled-self-test.cpp").read_bytes())
    context["generated"].update(file_hashes([source]))
    obj, executable = output / "coupled-self.obj", output / "coupled-self-test.exe"
    command = [str(compiler), "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
        "-MMD", "-MF", str(obj) + ".d", "-I" + str(output), "-c", str(source), "-o", str(obj)]
    execute(command, build_dir, output / "coupled-self-build.json", env)
    dependencies(obj, build_dir, [source, output / "v92-coupled.hpp", output / "v98-coupled.hpp", output / "v93-energy.hpp"], context["compiled"])
    link = [str(compiler), "-static", str(obj), "-o", str(executable)]
    context["test_inputs"].update(explicit_inputs(link, build_dir))
    execute(link, build_dir, output / "coupled-self-link.json", env)
    raw = execute([str(executable)], ROOT, output / "coupled-self-run.json", env)
    details = strict_json(raw["stdout"])
    expected_cases = {"exact_initial_physical", "fgmres_variable_preconditioner", "block_tolerances_are_independent",
        "happy_breakdown_requires_true_residual", "fixed_window_and_cycle_exhaustion", "invalid_dimension_and_callback_values",
        "budget_boundaries_keep_partial", "pure_tau_kappa_has_search_space",
        *context["plan"]["required_energy_standalone_cases"], *context["plan"]["required_wide_standalone_cases"],
        *context["plan"]["required_restart_standalone_cases"], *context["plan"]["required_initial_forcing_standalone_cases"],
        *context["plan"]["required_candidate_forcing_standalone_cases"], *context["plan"]["required_forcing_weight_standalone_cases"]}
    limits = {key: context["plan"]["coupled_schema"][key] for key in ("max_window", "max_cycles")}
    require(raw["stderr"] == "" and isinstance(details, dict) and
        set(details) == {"passed", "public_lp_runs", "cases", "default_limits", "details"} and details["passed"] is True and
        type(details["public_lp_runs"]) is int and details["public_lp_runs"] == 0 and
        type(details["cases"]) is int and details["cases"] == 43 and isinstance(details["details"], list) and
        len(details["details"]) == details["cases"] and all(type(item) is str and item for item in details["details"]) and
        set(details["details"]) == expected_cases and strict_coupled_limits(details["default_limits"], limits),
        "父39组默认策略加搜索尺度4组、唯一完成详情或真实上限不符合冻结计划")
    parent_checks = read(DATA / context["plan"]["math_evidence_file"])["math_self_checks"]["coupled"]
    require(parent_checks["passed"] is True and parent_checks["public_lp_runs"] == 0 and
        parent_checks["details"]["cases"] == 39 and
        set(details["details"]) - set(parent_checks["details"]["details"]) == set(context["plan"]["required_forcing_weight_standalone_cases"]),
        "须保留父v97实际完成的39组并精确新增搜索尺度4组")
    save(output / "coupled-self-check.json", {"passed": True, "public_lp_runs": 0, "details": details})
    context["test_commands"]["coupled"] = {"compile": command, "link": link, "run": [str(executable)]}
    return {**details, "run_file": "coupled-self-run.json", "check_file": "coupled-self-check.json"}


def math_extension(details, plan, required_new_cases):
    predecessor_path = DATA / plan["math_evidence_file"]
    require(sha(predecessor_path) == plan["math_evidence_sha256"], "数学覆盖父证据变化")
    predecessor = read(predecessor_path)
    checks = predecessor["math_self_checks"]["ipm"]
    require(predecessor["passed"] is True and predecessor["plan_sha256"] == PARENT_PLAN_SHA and checks["passed"] is True and
        type(checks["public_lp_runs"]) is int and checks["public_lp_runs"] == 0,
        "父v97数学组件原件须已通过且不含公开LP调用")
    old = checks["details"]
    old_cases = {item["id"]: item["group"] for item in old["details"]}
    new_cases = {item["id"]: item["group"] for item in details["details"]}
    require(type(old["cases"]) is int and old["cases"] == 79 == len(old["details"]) == len(old_cases) and
        old_cases.items() <= new_cases.items(), "须保留最终v97的79组实际数学测试")
    require(isinstance(required_new_cases, dict) and required_new_cases and
        not (required_new_cases.keys() & old_cases.keys()) and
        required_new_cases.items() <= new_cases.items(), "新增组合事务测试须真实完成且标识独立")
    call_counts = {}
    for name in ("component_ipm_calls", "synthetic_core_calls"):
        require(type(old[name]) is int and old[name] > 0, "父组件入口计数无效：" + name)
        # 新停止路由可能改变真实组件调用数；保留差值供归因，不凭计数大小冒充数学覆盖。
        call_counts[name] = {"predecessor": old[name], "current": details[name], "change": details[name] - old[name]}
    added = [item for item in details["details"] if item["id"] not in old_cases]
    require(details["cases"] == 82 and len(added) == 3 and {item["id"]: item["group"] for item in added} == required_new_cases,
        "新增数学覆盖须为冻结三组，父79加三组共82且无重复")
    return {"passed": True, "public_lp_runs": 0,
        "predecessor_evidence_file": str(predecessor_path), "predecessor_evidence_sha256": plan["math_evidence_sha256"],
        "preserved_predecessor_cases": old["cases"], "new_cases": added,
        "required_new_cases": [{"id": name, "group": group} for name, group in required_new_cases.items()],
        "component_calls": call_counts}


def math_tests(context):
    output, build_dir, env = (context[key] for key in ("output", "build_dir", "environment"))
    coupled_details = coupled_math_tests(context)
    ipm_source, ipm_obj, ipm_exe = DATA / "v98-ipm-self-test.cpp", output / "ipm-self.obj", output / "ipm-self-test.exe"
    ipm_command = private_compile(context["originals"]["ipm"], ipm_source, ipm_obj, context["frozen"], output, context["highs_includes"])
    execute(ipm_command, build_dir, output / "ipm-self-build.json", env)
    dependencies(ipm_obj, build_dir, [output / "ipm-presolve.cpp", output / "v98-presolve-state.hpp",
        output / "v98-coupled-state.hpp", output / "v98-coupled.hpp", output / "v93-energy.hpp",
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
    return {"coupled": coupled_details, "ipm": {**details, "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"}}

def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立输出须位于工作区且尚不存在")
    plan_path = DATA / "v98-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    parent_path = DATA / "ipm-v66-identity.json"
    require(sha(parent_path) == plan["parent_identity_sha256"], "冻结v66身份变化")
    parent = read(parent_path)
    verification_path, evidence_path = (DATA / plan[key] for key in ("baseline_verification_file", "evidence_file"))
    math_evidence_path, counter_evidence_path = (DATA / plan[key] for key in ("math_evidence_file", "counter_evidence_file"))
    coupled_kernel_path = DATA / plan["coupled_kernel_file"]
    parent_kernel_path = DATA / plan["parent_coupled_kernel_file"]
    algorithm_parent_kernel_path = DATA / plan["algorithm_parent_coupled_kernel_file"]
    energy_kernel_path = DATA / plan["energy_kernel_file"]
    require(sha(parent_kernel_path) == plan["parent_coupled_kernel_sha256"] and
        sha(algorithm_parent_kernel_path) == plan["algorithm_parent_coupled_kernel_sha256"] and
        sha(energy_kernel_path) == plan["energy_kernel_sha256"] and
        sha(DATA / plan["parent_coupled_test_file"]) == plan["parent_coupled_test_sha256"],
        "原双精度核、名义能量核及父11组数学来源变化")
    require(sha(verification_path) == plan["baseline_verification_sha256"] and
            sha(evidence_path) == plan["evidence_sha256"] and sha(math_evidence_path) == plan["math_evidence_sha256"] and
            sha(counter_evidence_path) == plan["counter_evidence_sha256"], "基线、前轮负结果、数学父或计数来源发生变化")
    baseline = read(verification_path)
    require(baseline["files"] == parent["files"] and baseline["libraries"] == parent["libraries"] and
            len(parent["files"]) == 95 and len(parent["libraries"]) == 17, "只冻结原v66的95来源和17库")
    production, reference = ROOT / "build/release/solver.exe", ROOT / ".tools/solver-ipm-v66.exe"
    source_archive = DATA / "ipm-v66-source.zip"
    require(sha(production) == sha(reference) == baseline["solver_sha256"] == parent["solver_sha256"] == plan["parent_solver_sha256"] and
            sha(source_archive) == parent["source_zip_sha256"] == plan["parent_source_sha256"], "生产/冻结程序及原源码包变化")
    require(sha(BASE_PATH) == HELPER_SHA, "通用构建辅助器变化")
    runner_check = read(DATA / "v98-runner-self-check.json")
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
    new_files = [DATA / name for name in ("v98-ipm-instrumentation.py", "v98-ipm-self-test.cpp", "v98-coupled-self-test.cpp",
        "v98-run.py", "v98-runner-self-check.json", "v66-run.py", "v97-ipm-instrumentation.py",
        "v97-coupled-self-test.cpp")]
    protected = file_hashes([production, reference, parent_path, source_archive, verification_path, evidence_path,
        math_evidence_path, counter_evidence_path, coupled_kernel_path, parent_kernel_path, algorithm_parent_kernel_path, energy_kernel_path,
        DATA / "v92-pretest-clarification.json",
        plan_path, BASE_PATH, Path(__file__), compiler, ninja, regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja",
        *explicit, *new_files])
    require(len(plan["build_snapshot_files"]) == len(set(plan["build_snapshot_files"])) == 17 and
        all(str(DATA / name) in protected for name in plan["build_snapshot_files"]),
        "全部17份测前快照来源须同属实际构建保护集合")
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
    private_header = output / "v98-presolve-state.hpp"
    private_header.write_text(instrument["HEADER"], encoding="utf-8")
    private_coupled_header = output / "v98-coupled-state.hpp"
    private_coupled_header.write_text(instrument["COUPLED_HEADER"], encoding="utf-8")
    kernel_copy = output / "v98-coupled.hpp"
    kernel_copy.write_bytes(coupled_kernel_path.read_bytes())
    require(sha(kernel_copy) == sha(coupled_kernel_path), "私有联立内核须逐字复制实际冻结来源")
    parent_kernel_copy = output / "v92-coupled.hpp"
    parent_kernel_copy.write_bytes(parent_kernel_path.read_bytes())
    require(sha(parent_kernel_copy) == plan["parent_coupled_kernel_sha256"], "旧11组测试须实际使用冻结双精度核")
    energy_copy = output / "v93-energy.hpp"
    energy_copy.write_bytes(energy_kernel_path.read_bytes())
    require(sha(energy_copy) == sha(energy_kernel_path), "私有能量组件须逐字复制实际来源")
    generated = file_hashes([source, private_header, private_coupled_header, kernel_copy, parent_kernel_copy, energy_copy])
    save(output / "transforms.json", {"source": str(frozen / "src/ipm.cpp"), "source_sha256": sha(frozen / "src/ipm.cpp"),
        "patched": str(source), "patched_sha256": sha(source), "changes": changes})
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    environment.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    obj, compiled = output / "ipm-presolve.obj", {}
    command = private_compile(originals["ipm"], source, obj, frozen, output, highs_includes)
    execute(command, build_dir, output / "ipm-compile.json", environment)
    dependencies(obj, build_dir, [source, private_header, private_coupled_header, kernel_copy, energy_copy,
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
        "private_coupled_state_header": str(private_coupled_header),
        "coupled_kernel": {"source": str(coupled_kernel_path), "source_sha256": sha(coupled_kernel_path),
            "private_copy": str(kernel_copy), "private_copy_sha256": sha(kernel_copy), "namespace_renamed": False},
        "parent_coupled_kernel": {"source": str(parent_kernel_path), "source_sha256": sha(parent_kernel_path),
            "private_copy": str(parent_kernel_copy), "private_copy_sha256": sha(parent_kernel_copy)},
        "energy_kernel": {"source": str(energy_kernel_path), "source_sha256": sha(energy_kernel_path),
            "private_copy": str(energy_copy), "private_copy_sha256": sha(energy_copy)},
        "readonly_parent_instrumentation": {"path": str(DATA / "v97-ipm-instrumentation.py"), "sha256": PARENT_INST_SHA},
        "readonly_algorithm_parent_coupled_kernel": {"path": str(algorithm_parent_kernel_path),
            "sha256": plan["algorithm_parent_coupled_kernel_sha256"]},
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
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v98-presolve")
    parser.add_argument("--self-test", action="store_true", help="仅锁定计划和插桩静态边界，不编译或运行LP")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build((ROOT / arguments.output).resolve())
