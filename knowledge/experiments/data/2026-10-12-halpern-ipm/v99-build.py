"""构建可关闭的v98阶段计时诊断；生产来源、数值算法和数学测试均保持。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "523115acd8348c1d21c7f3bc9ce302fedcd2a5b630ec65d96ee1235c8226de15"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
GENERATED = ("ipm-presolve.cpp", "v98-presolve-state.hpp", "v98-coupled-state.hpp", "v98-coupled.hpp",
    "v92-coupled.hpp", "v93-energy.hpp", "coupled-self.cpp", "normal-profile.cpp", "v99-profile.hpp",
    "profile-self.cpp", "profile-cross.cpp")
UNITS = {"ipm": "ipm-presolve.cpp", "normal": "normal-profile.cpp", "coupled": "v98-coupled.hpp"}
sys.dont_write_bytecode = True
assert hashlib.sha256((DATA / "v47-profile-build.py").read_bytes()).hexdigest() == HELPER_SHA
BASE = runpy.run_path(str(DATA / "v47-profile-build.py"))
sha, require = (BASE[key] for key in ("sha", "require"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


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


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def locked_plan():
    require(sha(DATA / "v99-plan.json") == PLAN_SHA, "冻结计划变化")
    plan = read(DATA / "v99-plan.json")
    require(plan["diagnostic_only"] is True and plan["algorithm_parent_version"] == 98 and
        plan["maximum_solver_runs"] == 216 and len(plan["build_snapshot_files"]) == len(set(plan["build_snapshot_files"])) == 17,
        "诊断边界或17份快照合同变化")
    for field, digest in (("parent_closeout_file", "parent_closeout_sha256"), ("parent_plan_file", "parent_plan_sha256"),
        ("parent_identity_file", "algorithm_parent_identity_sha256"), ("baseline_verification_file", "baseline_verification_sha256")):
        require(sha(DATA / plan[field]) == plan[digest], "冻结父证据变化：" + field)
    require(sha(plan["parent_build_manifest"]) == plan["parent_build_manifest_sha256"] and
        sha(plan["parent_reference"]) == plan["parent_reference_sha256"], "冻结父构建或程序变化")
    require_hashes(plan["protected_parent_inputs"])
    return plan


def modules():
    return (runpy.run_path(str(DATA / "v99-profile-instrumentation.py")),
        runpy.run_path(str(DATA / "v99-profile-kernel.py")))


def verified_transform(instrument, source, unit):
    patched, changes = instrument["transform"](source, unit)
    require(type(changes) is list and changes and len({item["label"] for item in changes}) == len(changes), "计时锚标识必须唯一")
    restored = patched
    for item in reversed(changes):
        require(set(item) == {"label", "before", "after"} and item["before"] != item["after"] and restored.count(item["after"]) == 1,
            "计时逆恢复锚不唯一：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == source.replace("\r\n", "\n"), "计时变换不能改变任何原算法字面")
    return patched, changes


def static_check(root=ROOT):
    plan = locked_plan()
    instrument, kernel = modules()
    source_check, kernel_check = instrument["static_check"](root), kernel["static_check"](root)
    require(source_check["passed"] is True and kernel_check["passed"] is True, "阶段计时源或复用计时核静态证明失败")
    require(tuple(kernel["STAGES"]) == tuple(plan["stages"]) and kernel["HEADER_NAME"] == "v99-profile.hpp" and
        kernel["SELFTEST_MACRO"] == "SOLVER_V99_PROFILE_SELFTEST", "计时阶段或双TU绑定变化")
    require(set(source_check["units"]) == set(UNITS) and all(source_check[key] is True for key in
        ("root_report_after_original_start", "root_total_after_isolation", "root_total_excludes_old_markers",
            "only_timer_changes", "no_new_budget_checks", "no_namespace_rename")), "三单元计时边界证明不完整")
    return dict(passed=True, public_lp_runs=0, diagnostic_only=True, instrumentation=source_check, profile=kernel_check)


def private_compile(original, source, obj, frozen, output, includes=()):
    command = compile_arguments(original, source, obj, frozen, ROOT)
    command.insert(1, "-I" + str(output))
    command[command.index("-c"):command.index("-c")] = includes
    return command


def explicit_inputs(command, cwd):
    return file_hashes([resolve(value, cwd) for value in command if not value.startswith("-") and
        Path(value).suffix.lower() in (".a", ".lib", ".obj", ".o")])


def collect_dependencies(obj, cwd, required, collected):
    paths = dependency_paths(Path(str(obj) + ".d"), cwd)
    require({Path(path).resolve() for path in required} <= {path.resolve() for path in paths}, "实际.d未包含应有私有源")
    for path, digest in file_hashes(paths).items():
        require(path not in collected or collected[path] == digest, "编译期间依赖变化：" + path)
        collected[path] = digest


def replacement_link(original, output, regularized, main_object=None):
    result, counts = [], dict(ipm=0, normal=0, main=0)
    for value in original:
        if value == "libsolver_ipm.a":
            result.extend(([str(output / "ipm-presolve.obj"), str(regularized)] if main_object is None else [str(regularized)]))
            counts["ipm"] += 1
        elif value == "libsolver_normal_cholesky.a":
            result.append(str(output / "normal-profile.obj")); counts["normal"] += 1
        elif main_object is not None and value == main_object:
            result.append(str(output / "ipm-self.obj")); counts["main"] += 1
        elif not value.startswith("-Wl,--out-implib,") and not (main_object is not None and value == "-municode"):
            result.append(value)
    require(counts == dict(ipm=1, normal=1, main=0 if main_object is None else 1), "链接替换次数不符")
    result[result.index("-o") + 1] = str(output / ("solver-profile.exe" if main_object is None else "ipm-self-test.exe"))
    return result


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "新诊断输出必须在工作区且目录尚不存在")
    plan = locked_plan()
    parent = read(DATA / "ipm-v66-identity.json")
    require(sha(DATA / "ipm-v66-identity.json") == plan["parent_identity_sha256"], "冻结v66身份变化")
    baseline = read(DATA / plan["baseline_verification_file"])
    current_files = {name: sha(ROOT / name) for name in parent["files"]}
    actual_libraries = {name: sha(ROOT / name) for name in parent["libraries"]}
    require(baseline["passed"] is True and type(baseline["public_lp_runs"]) is int and baseline["public_lp_runs"] == 0 and
        type(baseline["production_sources"]) is int and baseline["production_sources"] == len(current_files) == 95 and
        type(baseline["production_libraries"]) is int and baseline["production_libraries"] == len(actual_libraries) == 17 and
        current_files == parent["files"] and actual_libraries == parent["libraries"] and
        baseline["production_solver_sha256"] == parent["solver_sha256"] and
        baseline["parent_closeout_sha256"] == plan["parent_closeout_sha256"], "真实生产来源95/库17或诊断基线计数/身份变化")
    production, reference = ROOT / "build/release/solver.exe", ROOT / ".tools/solver-ipm-v66.exe"
    require(sha(production) == sha(reference) == plan["production_solver_sha256"] == parent["solver_sha256"], "生产程序变化")
    source_archive = DATA / "ipm-v66-source.zip"
    require(sha(source_archive) == parent["source_zip_sha256"], "原源码包变化")
    parent_identity = read(DATA / plan["parent_identity_file"])
    require_hashes(parent_identity["protected"])
    parent_work = Path(plan["parent_build_manifest"]).parent
    parent_build = read(plan["parent_build_manifest"])
    compiler, cwd = Path(parent["compiler"]), ROOT / "build/release"
    require(sha(compiler) == parent["compiler_sha256"], "冻结编译器变化")
    command_text = (parent_work / "ninja-commands.txt").read_text(encoding="utf-8")
    originals = {}
    for role, relative in (("ipm", "src/ipm.cpp"), ("normal", "src/normal_cholesky.cpp"),
        ("main", "src/main.cpp"), ("highs", "src/highs_backend.cpp")):
        matches = [split_windows(line) for line in command_text.splitlines() if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], cwd) == compiler, "原编译模板不唯一：" + role)
        originals[role] = matches[0]
    require({key: originals[key] for key in ("ipm", "main", "highs")} == parent_build["commands"]["original_compile"], "父实际编译模板不符")
    original_link = parent_build["commands"]["original_link"]
    require([line for line in command_text.splitlines() if " -o solver.exe " in line] == [parent["link_command"]], "原链接模板变化")
    old_includes = {resolve(value[2:], ROOT) for value in originals["ipm"] if value.startswith("-I")}
    includes = [value for value in originals["highs"] if value.startswith("-I") and resolve(value[2:], ROOT) != ROOT / "include" and resolve(value[2:], ROOT) not in old_includes]
    require(includes == parent_build["commands"]["highs_include_arguments"], "HiGHS实际头目录变化")
    regularized = cwd / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    sources = {str(ROOT / name): digest for name, digest in parent["files"].items()}
    libraries = {str(ROOT / name): digest for name, digest in parent["libraries"].items()}
    require_hashes(sources); require_hashes(libraries)
    protected = dict(parent_identity["protected"])
    protected.update(plan["protected_parent_inputs"])
    protected.update(file_hashes([DATA / name for name in plan["build_snapshot_files"]]))
    protected.update(file_hashes([DATA / plan["parent_identity_file"], DATA / plan["parent_closeout_file"], source_archive,
        production, reference, compiler, regularized, ROOT / "scripts/benchmark.py", ROOT / "scripts/compare_lp.py"]))
    protected.update(explicit_inputs(original_link, cwd))
    require_hashes(protected)
    check = read(DATA / "v99-runner-self-check.json")
    require(check["passed"] is True and type(check["public_lp_runs"]) is int and check["public_lp_runs"] == 0, "运行器纯数据自检尚未通过")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent["files"]), "原源码包成员变化")
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), "源码包路径越界")
        archive.extractall(frozen)
    frozen_hashes = {str(frozen / name): digest for name, digest in parent["files"].items()}
    require_hashes(frozen_hashes)
    instrument, kernel = modules()
    save(output / "static-check.json", static_check(ROOT))
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    transforms = {}
    parents = {"ipm": parent_work / "ipm-presolve.cpp", "normal": parent_work / "frozen/src/normal_cholesky.cpp", "coupled": parent_work / "v98-coupled.hpp"}
    for unit, name in UNITS.items():
        patched, changes = verified_transform(instrument, parents[unit].read_text(encoding="utf-8"), unit)
        target = output / name
        target.write_text(patched, encoding="utf-8")
        transforms[unit] = dict(source=str(parents[unit]), source_sha256=sha(parents[unit]), patched=str(target),
            patched_sha256=sha(target), changes=changes)
    for name in ("v98-presolve-state.hpp", "v98-coupled-state.hpp", "v92-coupled.hpp", "v93-energy.hpp"):
        (output / name).write_bytes((parent_work / name).read_bytes())
    (output / "coupled-self.cpp").write_bytes((DATA / "v98-coupled-self-test.cpp").read_bytes())
    for name, content in (("v99-profile.hpp", kernel["HEADER"]), ("profile-self.cpp", kernel["SELF_MAIN"]), ("profile-cross.cpp", kernel["SELF_CROSS"])):
        (output / name).write_text(content, encoding="utf-8")
    generated = file_hashes([output / name for name in GENERATED])
    save(output / "transforms.json", transforms)
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    environment.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    environment.update(SOLVER_V98_PRESOLVE="1", SOLVER_V99_PROFILE="0")
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    compiled, compile_commands = {}, {}
    specs = (("ipm", output / "ipm-presolve.cpp", "ipm-presolve.obj", "ipm", includes, "ipm-compile.json"),
        ("normal", output / "normal-profile.cpp", "normal-profile.obj", "normal", (), "normal-compile.json"),
        ("ipm_self", DATA / "v98-ipm-self-test.cpp", "ipm-self.obj", "ipm", includes, "ipm-self-build.json"))
    for role, source, obj_name, template, extra, record in specs:
        obj = output / obj_name
        command = private_compile(originals[template], source, obj, frozen, output, extra)
        execute(command, cwd, output / record, environment)
        required = [source, output / "v99-profile.hpp"]
        if template == "ipm":
            required += [output / name for name in ("ipm-presolve.cpp", "v98-presolve-state.hpp", "v98-coupled-state.hpp", "v98-coupled.hpp", "v93-energy.hpp")]
        collect_dependencies(obj, cwd, required, compiled)
        compile_commands[role] = command
    for role, source_name, obj_name in (("coupled_self", "coupled-self.cpp", "coupled-self.obj"),
        ("profile_self", "profile-self.cpp", "profile-self.obj"), ("profile_cross", "profile-cross.cpp", "profile-cross.obj")):
        source, obj = output / source_name, output / obj_name
        command = [str(compiler), "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
            "-MMD", "-MF", str(obj) + ".d", "-I" + str(output)]
        if role != "coupled_self":
            command.append("-D" + kernel["SELFTEST_MACRO"])
        command += ["-c", str(source), "-o", str(obj)]
        execute(command, cwd, output / (role.replace("_", "-") + "-build.json"), environment)
        required = [source, output / "v99-profile.hpp"]
        if role == "coupled_self":
            required += [output / name for name in ("v92-coupled.hpp", "v98-coupled.hpp", "v93-energy.hpp")]
        collect_dependencies(obj, cwd, required, compiled)
        compile_commands[role] = command
    link = replacement_link(original_link, output, regularized)
    link_inputs = explicit_inputs(link, cwd)
    execute(link, cwd, output / "link.json", environment)
    main_obj = originals["main"][originals["main"].index("-o") + 1]
    ipm_link = replacement_link(original_link, output, regularized, main_obj)
    coupled_link = [str(compiler), "-static", str(output / "coupled-self.obj"), "-o", str(output / "coupled-self-test.exe")]
    profile_link = [str(compiler), "-static", str(output / "profile-self.obj"), str(output / "profile-cross.obj"), "-o", str(output / "profile-self-test.exe")]
    test_inputs = {}
    for name, command in (("ipm", ipm_link), ("coupled", coupled_link), ("profile", profile_link)):
        test_inputs.update(explicit_inputs(command, cwd))
        execute(command, cwd, output / (name + "-self-link.json"), environment)
    parent_evidence = read(DATA / plan["parent_closeout_file"])["math_self_checks"]
    math, math_commands = {}, {}
    for name in ("ipm", "coupled"):
        exe = output / (name + "-self-test.exe")
        raw = execute([str(exe)], ROOT, output / (name + "-self-run.json"), environment)
        details = strict_json(raw["stdout"])
        require(raw["stderr"] == "" and equal(details, parent_evidence[name]["details"]) and details["passed"] is True and
            type(details["public_lp_runs"]) is int and details["public_lp_runs"] == 0 and details["cases"] == (82 if name == "ipm" else 43),
            "计时关闭的父数学源码须真实重跑并完整保留原验收：" + name)
        wrapper = dict(passed=True, public_lp_runs=0, details=details)
        save(output / (name + "-self-check.json"), wrapper)
        math[name] = dict(details, run_file=name + "-self-run.json", check_file=name + "-self-check.json")
        math_commands[name] = dict(compile=compile_commands[name + "_self"], link=ipm_link if name == "ipm" else coupled_link, run=[str(exe)])
    profile_tests, profile_commands = {}, {}
    runner = runpy.run_path(str(DATA / "v99-run.py"))
    for enabled in (False, True):
        name = "on" if enabled else "off"
        env = dict(environment, SOLVER_V99_PROFILE="1" if enabled else "0")
        command = [str(output / "profile-self-test.exe")]
        raw = execute(command, ROOT, output / ("profile-self-" + name + "-run.json"), env)
        details = strict_json(raw["stdout"])
        require(equal(details, dict(passed=True, enabled=enabled)), "双TU fakeclock自测真实输出不符")
        lines = raw["stderr"].splitlines()
        require(len(lines) == (1 if enabled else 0), "双TU fakeclock模式标记数不符")
        marker = strict_json(lines[0][len("V99_PROFILE "):]) if enabled and lines[0].startswith("V99_PROFILE ") else None
        require(not enabled or runner["valid_profile"](marker), "双TU fakeclock真实计时守恒不成立")
        wrapper = dict(passed=True, public_lp_runs=0, details=details, profile_marker=marker,
            profile_environment={"SOLVER_V99_PROFILE": "1" if enabled else "0"})
        save(output / ("profile-self-" + name + "-check.json"), wrapper)
        profile_tests[name] = dict(wrapper, run_file="profile-self-" + name + "-run.json", check_file="profile-self-" + name + "-check.json")
        profile_commands[name] = dict(run=command, profile_environment=wrapper["profile_environment"])
    for mapping in (protected, sources, libraries, frozen_hashes, generated, compiled, link_inputs, test_inputs):
        require_hashes(mapping)
    outputs = file_hashes([path for path in output.iterdir() if path.is_file()])
    executable = output / "solver-profile.exe"
    manifest = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), production_changes=False,
        diagnostic_only=True, experimental_candidate_only=False, algorithm_parent_version=98, plan_sha256=PLAN_SHA,
        parent_identity_sha256=plan["parent_identity_sha256"], production_executable=str(production), production_sha256=sha(production),
        frozen_parent_executable=str(reference), frozen_parent_sha256=sha(reference), executable=str(executable), executable_sha256=sha(executable),
        algorithm_parent_executable=plan["parent_reference"], algorithm_parent_sha256=plan["parent_reference_sha256"],
        parent_build_manifest=plan["parent_build_manifest"], parent_build_manifest_sha256=plan["parent_build_manifest_sha256"],
        baseline_verification_file=str(DATA / plan["baseline_verification_file"]), baseline_verification_sha256=plan["baseline_verification_sha256"],
        frozen_files=parent["files"], current_files=current_files, actual_libraries=actual_libraries,
        private_state_header=str(output / "v98-presolve-state.hpp"), private_coupled_state_header=str(output / "v98-coupled-state.hpp"),
        protected_before_and_after=protected, generated_inputs_before_and_after=generated, compile_inputs_after_and_final=compiled,
        link_explicit_before_and_after=link_inputs, test_link_explicit_before_and_after=test_inputs, outputs=outputs,
        math_self_tests=math, profile_self_tests=profile_tests,
        commands=dict(cwd=str(cwd), original_compile=originals, original_link=original_link, highs_include_arguments=includes,
            compile=compile_commands, link=link, math_self_tests=math_commands,
            profile_self_tests=dict(compile=[compile_commands["profile_self"], compile_commands["profile_cross"]], link=profile_link, variants=profile_commands)),
        implicit_link_inputs_verified=False, public_lp_runs=0,
        scope="仅216次诊断；两个候选CPP加四个自测CPP/六份.d/四个程序；不构成新的性能成绩或数学覆盖。")
    save(output / "build-manifest.json", manifest)
    print(json.dumps(dict(passed=True, diagnostic_only=True, compiled_units=6, generated_sources=11,
        solver_sha256=manifest["executable_sha256"], math_self_tests=math, profile_self_tests=profile_tests, public_lp_runs=0), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v99-profile")
    parser.add_argument("--self-test", action="store_true", help="仅静态核对冻结计划及可逆计时锚，不编译、不运行LP")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build((ROOT / args.output).resolve())
