"""从冻结v66独立构建双右端候选，生产来源、库与程序保持只读。"""
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
PLAN_SHA = "46e074b8cab1b8f02772cb00b737d9c91d49dd7f90ddc14c99d87a913922308b"
BASE_PATH = DATA / "v47-profile-build.py"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == HELPER_SHA
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[key] for key in ("sha", "require", "save"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def helpers():
    return (runpy.run_path(str(DATA / "v83-normal-instrumentation.py")),
            runpy.run_path(str(DATA / "v83-ipm-instrumentation.py")))


def static_check(root=ROOT):
    require(sha(DATA / "v83-plan.json") == PLAN_SHA, "冻结计划变化")
    normal, ipm = helpers()
    checks = {"passed": True, "solver_runs": 0,
              "normal": normal["static_check"](root), "ipm": ipm["static_check"](root)}
    require(checks["normal"]["passed"] is True and checks["ipm"]["passed"] is True,
            "双右端插桩静态检查未通过")
    return checks


def link_arguments(line, build_dir, compiler):
    outer = split_windows(line)
    require(len(outer) == 3 and outer[1].lower() == "/c", "链接包装变化")
    inner = outer[2]
    require(inner.startswith("cd . && ") and inner.endswith(" && cd ."), "链接目录包装变化")
    result = split_windows(inner[len("cd . && "):-len(" && cd .")])
    require(resolve(result[0], build_dir) == compiler, "链接编译器变化")
    return result


def with_override(original, source, obj, frozen, output, override):
    command = compile_arguments(original, source, obj, frozen, ROOT)
    # 覆盖头必须先于所有原项目头；输出目录用于私有头及包含式数学harness。
    command[1:1] = ["-I" + str(override), "-I" + str(output)]
    return command


def collect_dependencies(obj, build_dir, required_header, compiled_inputs):
    paths = dependency_paths(Path(str(obj) + ".d"), build_dir)
    require(required_header.resolve() in {path.resolve() for path in paths}, "实际编译依赖未使用覆盖声明头")
    for path, value in file_hashes(paths).items():
        require(path not in compiled_inputs or compiled_inputs[path] == value, "编译期间依赖变化：" + path)
        compiled_inputs[path] = value
    return paths


def build_math_tests(context):
    output, frozen, override = (context[key] for key in ("output", "frozen", "override"))
    build_dir, environment = context["build_dir"], context["environment"]
    original = context["originals"]["normal_test"]
    source = DATA / "v83-normal-self-test.cpp"
    obj = output / "normal-self.obj"
    command = with_override(original, source, obj, frozen, output, override)
    execute(command, build_dir, output / "normal-self-build.json", environment)
    collect_dependencies(obj, build_dir, context["override_header"], context["compiled_inputs"])
    original_obj = original[original.index("-o") + 1]
    executable = output / "normal-self-test.exe"
    link, replaced = [], {"harness": 0, "normal": 0}
    for value in context["normal_original_link"]:
        if value == original_obj:
            link.append(str(obj)); replaced["harness"] += 1
        elif value == "libsolver_normal_cholesky.a":
            link.append(str(context["objects"]["normal"])); replaced["normal"] += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replaced == {"harness": 1, "normal": 1}, "Normal数学测试须唯一替换harness和候选实现")
    link[link.index("-o") + 1] = str(executable)
    inputs = file_hashes([resolve(value, build_dir) for value in link if not value.startswith("-") and
                         Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")])
    context["test_link_inputs"].update(inputs)
    execute(link, build_dir, output / "normal-self-link.json", environment)
    row = execute([str(executable)], ROOT, output / "normal-self-run.json", environment)
    details = json.loads(row["stdout"])
    require(details["passed"] is True and type(details["cases"]) is int and details["cases"] > 0 and
            type(details["pair_cases"]) is int and details["pair_cases"] > 0 and
            details["long_column_interrupted"] is True and isinstance(details["pair_details"], list) and
            len(details["pair_details"]) == details["pair_cases"] and
            all(isinstance(item, str) for item in details["pair_details"]) and
            isinstance(details["snapshots"], list) and len(details["snapshots"]) == details["cases"],
            "Normal完整原方程、pair与预算事务自测失败")
    save(output / "normal-self-check.json", {"passed": True, "lp_runs": 0, "details": details})
    context["test_commands"]["normal"] = {"original_compile": original, "compile": command, "link": link}
    normal_summary = {"passed": True, "cases": details["cases"], "pair_cases": details["pair_cases"],
                      "run_file": "normal-self-run.json", "check_file": "normal-self-check.json"}
    ipm_source = DATA / "v83-ipm-self-test.cpp"
    ipm_obj = output / "ipm-self.obj"
    ipm_command = with_override(context["originals"]["ipm"], ipm_source, ipm_obj, frozen, output, override)
    execute(ipm_command, build_dir, output / "ipm-self-build.json", environment)
    paths = collect_dependencies(ipm_obj, build_dir, context["override_header"], context["compiled_inputs"])
    require({(output / "ipm-pair.cpp").resolve(), (output / "v83-pair-state.hpp").resolve()} <=
            {path.resolve() for path in paths}, "实际KKT数学harness必须包含生成IPM和私有pair状态头")
    main_original = context["originals"]["main"]
    main_obj = main_original[main_original.index("-o") + 1]
    ipm_executable = output / "ipm-self-test.exe"
    ipm_link, replaced = [], {"harness": 0, "ipm": 0, "normal": 0}
    for value in context["original_link"]:
        if value == main_obj:
            ipm_link.append(str(ipm_obj)); replaced["harness"] += 1
        elif value == "libsolver_ipm.a":
            ipm_link.append(str(context["regularized"])); replaced["ipm"] += 1
        elif value == "libsolver_normal_cholesky.a":
            ipm_link.append(str(context["objects"]["normal"])); replaced["normal"] += 1
        elif value != "-municode" and not value.startswith("-Wl,--out-implib,"):
            ipm_link.append(value)
    require(replaced == {"harness": 1, "ipm": 1, "normal": 1} and
            str(context["objects"]["ipm"]) not in ipm_link and "libsolver_ipm.a" not in ipm_link,
            "包含式IPM测试仅链接一次IPM定义，保留原regularized和其他库顺序")
    ipm_link[ipm_link.index("-o") + 1] = str(ipm_executable)
    for path, value in file_hashes([resolve(value, build_dir) for value in ipm_link if not value.startswith("-") and
            Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]).items():
        require(path not in context["test_link_inputs"] or context["test_link_inputs"][path] == value,
                "数学测试间显式链接输入变化：" + path)
        context["test_link_inputs"][path] = value
    execute(ipm_link, build_dir, output / "ipm-self-link.json", environment)
    ipm_row = execute([str(ipm_executable)], ROOT, output / "ipm-self-run.json", environment)
    ipm_details = json.loads(ipm_row["stdout"])
    require(ipm_row["stderr"] == "" and ipm_details["passed"] is True and
            type(ipm_details["solver_runs"]) is int and ipm_details["solver_runs"] == 0 and
            all(type(ipm_details[key]) is int and ipm_details[key] > 0 for key in ("cases", "native_cases", "fallback_cases")) and
            all(type(ipm_details[key]) is int and ipm_details[key] == 1 for key in
                ("finish0_refinement_cases", "repair_cases", "budget_cases")),
            "实际KKT双右端、单列精修、标量修复及预算事务数学自测失败")
    save(output / "ipm-self-check.json", {"passed": True, "lp_runs": 0, "details": ipm_details})
    context["test_commands"]["ipm"] = {"original_compile": context["originals"]["ipm"], "compile": ipm_command,
                                       "original_main_compile": main_original, "link": ipm_link}
    return {"normal": normal_summary, "ipm": {**ipm_details, "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"}}


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立输出须位于工作区且尚不存在")
    plan_path = DATA / "v83-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    identity_path = DATA / "ipm-v66-identity.json"
    identity = read(identity_path)
    baseline_path = DATA / plan["baseline_identity_file"]
    verification_path = DATA / plan["baseline_verification_file"]
    evidence_path = DATA / plan["evidence_file"]
    require(sha(identity_path) == plan["parent_identity_sha256"], "冻结v66身份变化")
    require(sha(baseline_path) == plan["baseline_identity_sha256"] and
            sha(verification_path) == plan["baseline_verification_sha256"] and
            sha(evidence_path) == plan["evidence_sha256"], "起点与前轮证据变化")
    baseline = read(baseline_path)
    require(baseline["files"] == identity["files"] and baseline["libraries"] == identity["libraries"] and
            len(identity["files"]) == 95 and len(identity["libraries"]) == 17, "生产来源与库须保持冻结v66")
    source_archive = DATA / "ipm-v66-source.zip"
    production = ROOT / "build/release/solver.exe"
    parent = ROOT / ".tools/solver-ipm-v66.exe"
    require(sha(source_archive) == identity["source_zip_sha256"] == plan["parent_source_sha256"] and
            sha(production) == sha(parent) == baseline["solver_sha256"] == identity["solver_sha256"] ==
            plan["parent_solver_sha256"], "生产与冻结父程序或源码包变化")
    require(sha(BASE_PATH) == HELPER_SHA, "通用构建辅助器身份变化")
    runner_check = read(DATA / "v83-runner-self-check.json")
    runner_runs = runner_check.get("solver_runs", runner_check.get("lp_runs"))
    require(runner_check.get("passed") is True and type(runner_runs) is int and runner_runs == 0,
            "运行器纯数据自检尚未通过")
    sources = {str(ROOT / name): value for name, value in identity["files"].items()}
    libraries = {str(ROOT / name): value for name, value in identity["libraries"].items()}
    require_hashes(sources); require_hashes(libraries)
    compiler = Path(identity["compiler"])
    require(sha(compiler) == identity["compiler_sha256"], "编译器变化")
    build_dir = ROOT / "build/release"
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands",
                                            "solver.exe", "test_normal_cholesky.exe"], text=True, encoding="utf-8")
    lines = command_text.splitlines()
    originals = {}
    for unit, relative in (("ipm", "src/ipm.cpp"), ("normal", "src/normal_cholesky.cpp"), ("main", "src/main.cpp"),
                           ("normal_test", "tests/test_normal_cholesky.cpp")):
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "原编译模板不唯一：" + unit)
        originals[unit] = matches[0]
    solver_links = [line for line in lines if " -o solver.exe " in line]
    require(solver_links == [identity["link_command"]], "生产链接序列变化")
    original_link = link_arguments(solver_links[0], build_dir, compiler)
    normal_links = [line for line in lines if " -o test_normal_cholesky.exe " in line]
    require(len(normal_links) == 1, "Normal数学harness链接模板不唯一")
    normal_original_link = link_arguments(normal_links[0], build_dir, compiler)
    explicit = [resolve(value, build_dir) for value in original_link + normal_original_link if not value.startswith("-") and
                Path(value).suffix.lower() in (".a", ".lib")]
    explicit.extend(resolve(value, build_dir) for value in original_link if not value.startswith("-") and
                    Path(value).suffix.lower() in (".o", ".obj"))
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    helper_files = [DATA / name for name in ("v83-normal-instrumentation.py", "v83-ipm-instrumentation.py",
                                            "v83-normal-self-test.cpp", "v83-ipm-self-test.cpp", "v83-run.py",
                                            "v83-runner-self-check.json", "v66-run.py")]
    protected = file_hashes([production, parent, compiler, ninja, identity_path, source_archive,
                            baseline_path, verification_path, evidence_path, plan_path, BASE_PATH, Path(__file__),
                            build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", regularized,
                            *explicit, *helper_files])
    normal_helper, ipm_helper = helpers()
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(source_archive) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(identity["files"]), "冻结来源包成员变化")
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), "冻结来源包路径越界")
        archive.extractall(frozen)
    frozen_files = {str(frozen / name): value for name, value in identity["files"].items()}
    require_hashes(frozen_files)
    save(output / "static-check.json", static_check(frozen))
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    override = output / "override/include"
    override_header = override / "solver/detail/normal_cholesky.hpp"
    override_header.parent.mkdir(parents=True)
    transforms, generated = {}, {}
    for unit, source, destination, helper, arguments in (
        ("ipm", frozen / "src/ipm.cpp", output / "ipm-pair.cpp", ipm_helper, ()),
        ("normal", frozen / "src/normal_cholesky.cpp", output / "normal-pair.cpp", normal_helper, ("normal",)),
        ("normal_header", frozen / "include/solver/detail/normal_cholesky.hpp", override_header, normal_helper, ("normal_header",))):
        patched, changes = helper["transform"](source.read_text(encoding="utf-8"), *arguments)
        destination.write_text(patched, encoding="utf-8")
        generated.update(file_hashes([destination]))
        transforms[unit] = {"source": str(source), "source_sha256": sha(source), "patched": str(destination),
                            "patched_sha256": sha(destination), "changes": changes}
    private_header = output / "v83-pair-state.hpp"
    private_header.write_text(ipm_helper["HEADER"], encoding="utf-8")
    generated.update(file_hashes([private_header]))
    save(output / "transforms.json", transforms)
    objects, commands, compiled_inputs = {}, {}, {}
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and key.upper() != "OPENBLAS_CORETYPE"}
    environment.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    for unit in ("ipm", "normal"):
        obj = output / (unit + "-pair.obj")
        objects[unit] = obj
        commands[unit] = with_override(originals[unit], output / (unit + "-pair.cpp"), obj, frozen, output, override)
        execute(commands[unit], build_dir, output / (unit + "-compile.json"), environment)
        paths = collect_dependencies(obj, build_dir, override_header, compiled_inputs)
        if unit == "ipm": require(private_header.resolve() in {path.resolve() for path in paths}, "IPM实际依赖必须包含私有pair状态头")
    link, replaced = [], {"ipm": 0, "normal": 0}
    for value in original_link:
        if value == "libsolver_ipm.a":
            link.extend([str(objects["ipm"]), str(regularized)]); replaced["ipm"] += 1
        elif value == "libsolver_normal_cholesky.a":
            link.append(str(objects["normal"])); replaced["normal"] += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replaced == {"ipm": 1, "normal": 1}, "候选只能唯一替换两份库")
    executable = output / "solver-pair.exe"
    link[link.index("-o") + 1] = str(executable)
    link_inputs = file_hashes([resolve(value, build_dir) for value in link if not value.startswith("-") and
                              Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")])
    execute(link, build_dir, output / "link.json", environment)
    context = dict(output=output, frozen=frozen, override=override, override_header=override_header,
                   compiler=compiler, build_dir=build_dir, originals=originals, original_link=original_link,
                   normal_original_link=normal_original_link, regularized=regularized, objects=objects,
                   environment=environment, compiled_inputs=compiled_inputs, protected=protected, generated=generated,
                   test_link_inputs={}, test_commands={})
    tests = build_math_tests(context)
    for values in (protected, sources, libraries, frozen_files, generated, compiled_inputs, link_inputs,
                   context["test_link_inputs"]): require_hashes(values)
    outputs = file_hashes([path for path in output.rglob("*") if path.is_file() and not path.is_relative_to(frozen)])
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "production_changes": False,
        "experimental_candidate_only": True, "plan_sha256": PLAN_SHA,
        "production_executable": str(production), "production_sha256": sha(production),
        "frozen_parent_executable": str(parent), "frozen_parent_sha256": sha(parent),
        "executable": str(executable), "executable_sha256": sha(executable),
        "baseline_identity_file": str(baseline_path), "baseline_identity_sha256": sha(baseline_path),
        "baseline_verification_file": str(verification_path), "baseline_verification_sha256": sha(verification_path),
        "frozen_files": identity["files"], "current_files": baseline["files"], "actual_libraries": baseline["libraries"],
        "override_include_directory": str(override), "override_header": str(override_header),
        "private_pair_header": str(private_header), "override_header_used_by_all_compiled_units": True,
        "protected_before_and_after": protected, "generated_inputs_before_and_after": generated,
        "compile_inputs_after_and_final": compiled_inputs, "link_explicit_before_and_after": link_inputs,
        "test_link_explicit_before_and_after": context["test_link_inputs"],
        "outputs": outputs, "math_self_tests": tests,
        "commands": {"cwd": str(build_dir), "original_compile": {unit: originals[unit] for unit in ("ipm", "normal")},
                     "original_link": original_link, "normal_original_link": normal_original_link,
                     "compile": commands, "link": link, "math_self_tests": context["test_commands"]},
        "implicit_link_inputs_verified": False, "lp_runs": 0}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"passed": True, "compiled_units": ["ipm", "normal"], "solver_sha256": manifest["executable_sha256"],
                      "math_self_tests": tests, "lp_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v83-pair")
    parser.add_argument("--self-test", action="store_true", help="仅锁计划和插桩静态边界，不编译或运行LP")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
