"""从冻结v66独立构建有界warm候选，仅替换IPM翻译单元。"""
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
PLAN_SHA = "4dfec47350f3b208602f0ec36f5fda8a2888d68f630b7c48620f72c5dd86b6aa"
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


def helper():
    return runpy.run_path(str(DATA / "v84-ipm-instrumentation.py"))


def static_check(root=ROOT):
    require(sha(DATA / "v84-plan.json") == PLAN_SHA, "冻结计划变化")
    checks = helper()["static_check"](root)
    require(checks["passed"] is True, "warm插桩静态检查失败")
    return {"passed": True, "solver_runs": 0, "ipm": checks}


def private_compile(original, source, obj, frozen, output):
    command = compile_arguments(original, source, obj, frozen, ROOT)
    command.insert(1, "-I" + str(output))
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


def math_tests(context):
    output, compiler, build_dir, env = (context[key] for key in ("output", "compiler", "build_dir", "environment"))
    warm_source = output / "warm-self.cpp"
    warm_source.write_bytes((DATA / "v84-warm-self-test.cpp").read_bytes())
    context["generated"].update(file_hashes([warm_source]))
    warm_obj, warm_exe = output / "warm-self.obj", output / "warm-self-test.exe"
    warm_command = [str(compiler), "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
                    "-MMD", "-MF", str(warm_obj) + ".d", "-I" + str(output), "-c", str(warm_source), "-o", str(warm_obj)]
    execute(warm_command, build_dir, output / "warm-self-build.json", env)
    dependencies(warm_obj, build_dir, [warm_source, output / "v84-warm-pdhg.hpp"], context["compiled"])
    warm_link = [str(compiler), "-static", str(warm_obj), "-o", str(warm_exe)]
    context["test_inputs"].update(explicit_inputs(warm_link, build_dir))
    execute(warm_link, build_dir, output / "warm-self-link.json", env)
    warm_run = execute([str(warm_exe)], ROOT, output / "warm-self-run.json", env)
    warm_details = json.loads(warm_run["stdout"])
    require(warm_run["stderr"] == "" and warm_details["passed"] is True and
            type(warm_details["public_lp_runs"]) is int and warm_details["public_lp_runs"] == 0 and
            type(warm_details["cases"]) is int and warm_details["cases"] > 0 and
            isinstance(warm_details["details"], list) and len(warm_details["details"]) == warm_details["cases"] and
            all(isinstance(item, str) and item for item in warm_details["details"]), "独立warm内核数学测试失败")
    save(output / "warm-self-check.json", {"passed": True, "lp_runs": 0, "details": warm_details})
    ipm_source, ipm_obj, ipm_exe = DATA / "v84-ipm-self-test.cpp", output / "ipm-self.obj", output / "ipm-self-test.exe"
    ipm_command = private_compile(context["originals"]["ipm"], ipm_source, ipm_obj, context["frozen"], output)
    execute(ipm_command, build_dir, output / "ipm-self-build.json", env)
    dependencies(ipm_obj, build_dir, [output / "ipm-warm.cpp", output / "v84-warm-pdhg.hpp",
                 output / "v84-warm-state.hpp"], context["compiled"])
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
    require(replacements == {"harness": 1, "ipm": 1} and str(output / "ipm-warm.obj") not in ipm_link and
            "libsolver_ipm.a" not in ipm_link, "包含式IPM数学harness不能重复链接IPM定义")
    ipm_link[ipm_link.index("-o") + 1] = str(ipm_exe)
    for path, digest in explicit_inputs(ipm_link, build_dir).items():
        require(path not in context["test_inputs"] or context["test_inputs"][path] == digest, "数学测试间链接输入变化")
        context["test_inputs"][path] = digest
    execute(ipm_link, build_dir, output / "ipm-self-link.json", env)
    ipm_run = execute([str(ipm_exe)], ROOT, output / "ipm-self-run.json", env)
    ipm_details = json.loads(ipm_run["stdout"])
    require(ipm_run["stderr"] == "" and ipm_details["passed"] is True and
            type(ipm_details["solver_runs"]) is int and ipm_details["solver_runs"] == 0 and
            all(type(ipm_details[key]) is int and ipm_details[key] > 0 for key in
                ("cases", "mapping_cases", "isolation_cases", "verification_accounting_cases", "budget_cases", "exception_cases")),
            "IPM warm映射/隔离/验收计账/预算/异常数学测试失败")
    save(output / "ipm-self-check.json", {"passed": True, "lp_runs": 0, "details": ipm_details})
    context["test_commands"].update(warm={"compile": warm_command, "link": warm_link},
        ipm={"original_compile": context["originals"]["ipm"], "original_main_compile": original_main,
             "compile": ipm_command, "link": ipm_link})
    return {"warm": {"passed": True, "cases": warm_details["cases"], "run_file": "warm-self-run.json", "check_file": "warm-self-check.json"},
            "ipm": {**ipm_details, "run_file": "ipm-self-run.json", "check_file": "ipm-self-check.json"}}


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立输出须位于工作区且尚不存在")
    plan_path = DATA / "v84-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    parent_path = DATA / "ipm-v66-identity.json"
    require(sha(parent_path) == plan["parent_identity_sha256"], "冻结v66身份变化")
    parent = read(parent_path)
    baseline_path, verification_path, evidence_path = (DATA / plan[key] for key in
        ("baseline_identity_file", "baseline_verification_file", "evidence_file"))
    require(sha(baseline_path) == plan["baseline_identity_sha256"] and sha(verification_path) == plan["baseline_verification_sha256"] and
            sha(evidence_path) == plan["evidence_sha256"], "事前基线或前轮证据变化")
    baseline = read(baseline_path)
    require(baseline["files"] == parent["files"] and baseline["libraries"] == parent["libraries"] and
            len(parent["files"]) == 95 and len(parent["libraries"]) == 17, "只冻结原v66的95来源和17库")
    production, reference = ROOT / "build/release/solver.exe", ROOT / ".tools/solver-ipm-v66.exe"
    source_archive = DATA / "ipm-v66-source.zip"
    require(sha(production) == sha(reference) == baseline["solver_sha256"] == parent["solver_sha256"] == plan["parent_solver_sha256"] and
            sha(source_archive) == parent["source_zip_sha256"] == plan["parent_source_sha256"], "生产/冻结程序及原源码包变化")
    require(sha(BASE_PATH) == HELPER_SHA, "通用构建辅助器变化")
    runner_check = read(DATA / "v84-runner-self-check.json")
    runner_runs = runner_check.get("solver_runs", runner_check.get("lp_runs"))
    require(runner_check.get("passed") is True and type(runner_runs) is int and runner_runs == 0, "运行器纯数据自检尚未通过")
    sources = {str(ROOT / name): digest for name, digest in parent["files"].items()}
    libraries = {str(ROOT / name): digest for name, digest in parent["libraries"].items()}
    require_hashes(sources); require_hashes(libraries)
    compiler, build_dir = Path(parent["compiler"]), ROOT / "build/release"
    require(sha(compiler) == parent["compiler_sha256"], "编译器变化")
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"], text=True, encoding="utf-8")
    lines, originals = command_text.splitlines(), {}
    for role, relative in (("ipm", "src/ipm.cpp"), ("main", "src/main.cpp")):
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "原始编译模板不唯一")
        originals[role] = matches[0]
    links = [line for line in lines if " -o solver.exe " in line]
    require(links == [parent["link_command"]], "原链接序列变化")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "原链接包装变化")
    original_link = split_windows(outer[2][len("cd . && "):-len(" && cd .")])
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    explicit = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".obj", ".o")]
    new_files = [DATA / name for name in ("v84-ipm-instrumentation.py", "v84-warm-pdhg.hpp", "v84-warm-self-test.cpp",
        "v84-ipm-self-test.cpp", "v84-run.py", "v84-runner-self-check.json", "v66-run.py")]
    protected = file_hashes([production, reference, parent_path, source_archive, baseline_path, verification_path, evidence_path,
        plan_path, BASE_PATH, Path(__file__), compiler, ninja, regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja",
        *explicit, *new_files])
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
    patched, changes = instrument["transform"]((frozen / "src/ipm.cpp").read_text(encoding="utf-8"))
    source = output / "ipm-warm.cpp"
    source.write_text(patched, encoding="utf-8")
    private_header, kernel_copy = output / "v84-warm-state.hpp", output / "v84-warm-pdhg.hpp"
    private_header.write_text(instrument["HEADER"], encoding="utf-8")
    kernel_copy.write_bytes((DATA / "v84-warm-pdhg.hpp").read_bytes())
    require(sha(kernel_copy) == sha(DATA / "v84-warm-pdhg.hpp"), "私有kernel副本必须保持原字节")
    generated = file_hashes([source, private_header, kernel_copy])
    save(output / "transforms.json", {"source": str(frozen / "src/ipm.cpp"), "source_sha256": sha(frozen / "src/ipm.cpp"),
        "patched": str(source), "patched_sha256": sha(source), "changes": changes})
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and key.upper() != "OPENBLAS_CORETYPE"}
    environment.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    obj, compiled = output / "ipm-warm.obj", {}
    command = private_compile(originals["ipm"], source, obj, frozen, output)
    execute(command, build_dir, output / "ipm-compile.json", environment)
    dependencies(obj, build_dir, [source, private_header, kernel_copy], compiled)
    link, replacements = [], 0
    for value in original_link:
        if value == "libsolver_ipm.a":
            link.extend([str(obj), str(regularized)]); replacements += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replacements == 1 and link.count("libsolver_normal_cholesky.a") == original_link.count("libsolver_normal_cholesky.a") == 1,
            "只唯一替换IPM库，Normal及其他库保持原序")
    executable = output / "solver-warm.exe"
    link[link.index("-o") + 1] = str(executable)
    link_inputs = explicit_inputs(link, build_dir)
    execute(link, build_dir, output / "link.json", environment)
    context = dict(output=output, frozen=frozen, compiler=compiler, build_dir=build_dir, environment=environment,
        originals=originals, original_link=original_link, regularized=regularized, generated=generated,
        compiled=compiled, test_inputs={}, test_commands={})
    tests = math_tests(context)
    for mapping in (protected, sources, libraries, frozen_files, generated, compiled, link_inputs, context["test_inputs"]): require_hashes(mapping)
    outputs = file_hashes([path for path in output.iterdir() if path.is_file()])
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "production_changes": False,
        "experimental_candidate_only": True, "plan_sha256": PLAN_SHA, "parent_identity_sha256": plan["parent_identity_sha256"],
        "production_executable": str(production), "production_sha256": sha(production),
        "frozen_parent_executable": str(reference), "frozen_parent_sha256": sha(reference),
        "executable": str(executable), "executable_sha256": sha(executable),
        "baseline_identity_file": str(baseline_path), "baseline_identity_sha256": sha(baseline_path),
        "baseline_verification_file": str(verification_path), "baseline_verification_sha256": sha(verification_path),
        "frozen_files": parent["files"], "current_files": baseline["files"], "actual_libraries": baseline["libraries"],
        "private_state_header": str(private_header), "private_kernel_header": str(kernel_copy),
        "kernel_source_copy": {"source": str(DATA / "v84-warm-pdhg.hpp"), "copy": str(kernel_copy), "sha256": sha(kernel_copy)},
        "protected_before_and_after": protected, "generated_inputs_before_and_after": generated,
        "compile_inputs_after_and_final": compiled, "link_explicit_before_and_after": link_inputs,
        "test_link_explicit_before_and_after": context["test_inputs"], "outputs": outputs, "math_self_tests": tests,
        "commands": {"cwd": str(build_dir), "original_compile": {"ipm": originals["ipm"]}, "original_link": original_link,
                     "compile": {"ipm": command}, "link": link, "math_self_tests": context["test_commands"]},
        "implicit_link_inputs_verified": False, "lp_runs": 0}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"passed": True, "compiled_units": ["ipm"], "solver_sha256": manifest["executable_sha256"],
                      "math_self_tests": tests, "lp_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v84-warm")
    parser.add_argument("--self-test", action="store_true", help="仅锁定计划和插桩静态边界，不编译或运行LP")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
