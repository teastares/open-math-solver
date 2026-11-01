"""从冻结v66构建两翻译单元阶段诊断，复用已验证计时器，不改生产产物。"""
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
PLAN_SHA = "b5df77438c3905566b0a8ddb02b84c2c2f65484facfe26bf559ec2ae085ec44b"
BASE_PATH = DATA / "v47-profile-build.py"
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[key] for key in ("sha", "require", "save"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def static_check():
    plan_path = DATA / "v82-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    helper = runpy.run_path(str(DATA / "v82-instrumentation.py"))
    require(list(helper["STAGES"]) == plan["profile_stages"], "阶段集合与计划不符")
    result = helper["static_check"](ROOT)
    require(result["passed"] is True, "当前AAt插桩静态检查未通过")
    return result


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立输出须位于工作区且尚不存在")
    plan_path = DATA / "v82-plan.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan = read(plan_path)
    require(plan["transformed_units"] == {"ipm": "src/ipm.cpp", "normal": "src/normal_cholesky.cpp"},
            "阶段诊断只能转换两份冻结来源")
    require(plan["primary_linear_solver"] == "normal_cholmod_aat" and
            plan["linear_accuracy"] == "homogeneous_repair", "当前AAt诊断配置变化")
    identity_path = DATA / "ipm-v66-identity.json"
    identity = read(identity_path)
    require(sha(identity_path) == plan["parent_identity_sha256"], "父身份变化")
    archive = DATA / "ipm-v66-source.zip"
    baseline_path = DATA / plan["baseline_identity_file"]
    verification_path = DATA / plan["baseline_verification_file"]
    evidence_path = DATA / plan["evidence_file"]
    require(sha(baseline_path) == plan["baseline_identity_sha256"], "生产基线身份变化")
    require(sha(verification_path) == plan["baseline_verification_sha256"], "起点核对记录变化")
    require(sha(evidence_path) == plan["evidence_sha256"], "前轮证据变化")
    baseline = read(baseline_path)
    production = ROOT / "build/release/solver.exe"
    frozen_parent = ROOT / ".tools/solver-ipm-v66.exe"
    require(sha(archive) == identity["source_zip_sha256"] == plan["parent_source_sha256"], "父源码包变化")
    require(sha(production) == sha(frozen_parent) == baseline["solver_sha256"] ==
            identity["solver_sha256"] == plan["parent_solver_sha256"], "生产与冻结父程序变化")
    require(baseline["files"] == identity["files"] and baseline["libraries"] == identity["libraries"],
            "生产来源和库必须与冻结v66一致")
    require(sha(BASE_PATH) == plan["helper_sha256"], "复用计时工具变化")
    runner_check_path = DATA / "v82-runner-self-check.json"
    runner_check = read(runner_check_path)
    require(runner_check.get("passed") is True and type(runner_check.get("solver_runs")) is int and
            runner_check["solver_runs"] == 0, "运行器纯数据自检尚未通过")
    sources = {str(ROOT / name): value for name, value in baseline["files"].items()}
    libraries = {str(ROOT / name): value for name, value in baseline["libraries"].items()}
    require(len(sources) == 95 and len(libraries) == 17, "生产来源或显式库集合变化")
    require_hashes(sources)
    require_hashes(libraries)
    compiler = Path(identity["compiler"])
    require(sha(compiler) == identity["compiler_sha256"], "编译器变化")
    build_dir = ROOT / "build/release"
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"], text=True, encoding="utf-8")
    lines = command_text.splitlines()
    original_compile = {}
    for unit, relative in plan["transformed_units"].items():
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "两翻译单元实际编译命令不唯一")
        original_compile[unit] = matches[0]
    links = [line for line in lines if " -o solver.exe " in line]
    require(links == [identity["link_command"]], "实际链接命令变化")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c", "链接包装变化")
    inner = outer[2]
    require(inner.startswith("cd . && ") and inner.endswith(" && cd ."), "链接目录包装变化")
    original_link = split_windows(inner[len("cd . && "):-len(" && cd .")])
    explicit = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    helper_path = DATA / "v82-instrumentation.py"
    protected = file_hashes([production, frozen_parent, compiler, ninja, identity_path, archive, BASE_PATH,
        Path(__file__), helper_path, plan_path, DATA / "v82-run.py", baseline_path, verification_path,
        evidence_path, runner_check_path, regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", *explicit])
    helper = runpy.run_path(str(helper_path))
    require(list(helper["STAGES"]) == plan["profile_stages"], "阶段集合与计划不符")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == 95 and set(bundle.namelist()) == set(identity["files"]),
                "冻结来源包成员变化")
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in bundle.namelist()), "源码包路径越界")
        bundle.extractall(frozen)
    frozen_files = {str(frozen / name): value for name, value in identity["files"].items()}
    require_hashes(frozen_files)
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    (output / "v47-profile.hpp").write_text(helper["HEADER"], encoding="utf-8")
    static_check = helper["static_check"](frozen)
    require(static_check["passed"] is True, "当前AAt插桩静态检查未通过")
    save(output / "static-check.json", static_check)
    generated_inputs = file_hashes([output / "v47-profile.hpp"])
    transforms, compile_commands, objects = {}, {}, {}
    for unit, relative in plan["transformed_units"].items():
        source = frozen / relative
        transformed, changes = helper["transform"](source.read_text(encoding="utf-8"), unit)
        destination = output / (unit + "-profile.cpp")
        destination.write_text(transformed, encoding="utf-8")
        generated_inputs.update(file_hashes([destination]))
        obj = output / (unit + "-profile.obj")
        objects[unit] = obj
        compile_commands[unit] = compile_arguments(original_compile[unit], destination, obj, frozen, ROOT)
        transforms[unit] = {"source": str(source), "source_sha256": sha(source), "patched": str(destination),
            "patched_sha256": sha(destination), "changes": changes, "reversible": True}
    save(output / "transforms.json", transforms)
    executable = output / "solver-profile.exe"
    link, replaced = [], {"ipm": 0, "normal": 0}
    for value in original_link:
        if value == "libsolver_ipm.a":
            link.extend([str(objects["ipm"]), str(regularized)])
            replaced["ipm"] += 1
        elif value == "libsolver_normal_cholesky.a":
            link.append(str(objects["normal"]))
            replaced["normal"] += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replaced == {"ipm": 1, "normal": 1}, "两库替换不唯一")
    link[link.index("-o") + 1] = str(executable)
    commands = {"cwd": str(build_dir), "original_compile": original_compile, "original_link": original_link,
        "compile": compile_commands, "link": link}
    save(output / "commands.json", commands)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("SOLVER_V") and key != "OPENBLAS_CORETYPE"}
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    compiled_inputs = {}
    for unit, command in compile_commands.items():
        execute(command, build_dir, output / (unit + "-compile.json"), environment)
        compiled_inputs.update(file_hashes(dependency_paths(Path(str(objects[unit]) + ".d"), build_dir)))
    final_inputs = [resolve(value, build_dir) for value in link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    before = file_hashes(final_inputs)
    execute(link, build_dir, output / "link.json", environment)
    self_commands, self_objects = [], []
    for name, text in (("main", helper["SELF_MAIN"]), ("cross", helper["SELF_CROSS"])):
        source = output / ("self-" + name + ".cpp")
        source.write_text(text, encoding="utf-8")
        generated_inputs.update(file_hashes([source]))
        obj = output / ("self-" + name + ".obj")
        command = [str(compiler), "-O3", "-DNDEBUG", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
            "-DSOLVER_V47_PROFILE_SELFTEST", "-MMD", "-MF", str(obj) + ".d", "-c", str(source), "-o", str(obj)]
        execute(command, build_dir, output / ("self-" + name + "-build.json"), environment)
        compiled_inputs.update(file_hashes(dependency_paths(Path(str(obj) + ".d"), build_dir)))
        self_commands.append(command)
        self_objects.append(str(obj))
    self_executable = output / "profile-self-test.exe"
    self_link = [str(compiler), "-static", *self_objects, "-o", str(self_executable)]
    execute(self_link, build_dir, output / "self-link.json", environment)
    timer_results = {}
    for mode in ("off", "on"):
        env = environment.copy()
        if mode == "on":
            env[plan["profile_environment"]] = plan["profile_enabled_value"]
        row = execute([str(self_executable)], ROOT, output / ("self-" + mode + "-run.json"), env)
        timer_results[mode] = json.loads(row["stdout"])
        require(timer_results[mode] == {"passed": True, "enabled": mode == "on"}, "双翻译单元计时器自测失败")
        require((row["stderr"] == "") if mode == "off" else
                row["stderr"].count(plan["profile_prefix"]) == 1, "计时器开关输出边界错误")
    save(output / "timer-self-check.json", {"passed": True, "solver_runs": 0,
                                            "off": timer_results["off"], "on": timer_results["on"]})
    for values in (protected, sources, libraries, frozen_files, before, generated_inputs, compiled_inputs):
        require_hashes(values)
    outputs = [path for path in output.iterdir() if path.is_file()]
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "production_changes": False,
        "plan_sha256": PLAN_SHA, "production_executable": str(production), "production_sha256": sha(production),
        "frozen_parent_executable": str(frozen_parent), "frozen_parent_sha256": sha(frozen_parent),
        "baseline_identity_file": str(baseline_path), "baseline_identity_sha256": sha(baseline_path),
        "baseline_verification_file": str(verification_path), "baseline_verification_sha256": sha(verification_path),
        "executable": str(executable), "executable_sha256": sha(executable), "self_test_compile": self_commands,
        "self_test_link": self_link, "frozen_files": identity["files"], "protected_before_and_after": protected,
        "current_files": baseline["files"], "actual_libraries": baseline["libraries"],
        "link_explicit_before_and_after": before, "compile_inputs_after_and_final": compiled_inputs,
        "generated_inputs_before_and_after": generated_inputs,
        "implicit_link_inputs_verified": False, "outputs": file_hashes(outputs), "commands": commands,
        "identity_limit": "显式链接输入核前后身份；编译依赖生成后哈希并末尾复核；未逐项核验隐式运行库。"}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"executable_sha256": manifest["executable_sha256"], "sources": len(sources),
        "libraries": len(libraries), "explicit_link_inputs": len(before), "compile_dependencies": len(compiled_inputs)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v82-diagnostic")
    parser.add_argument("--self-test", action="store_true", help="仅核计划身份及插桩静态边界，不编译或运行LP")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
