"""从冻结 v57 源码和当前重建基线生成独立 KKT 失败诊断，不修改生产产物。"""
import argparse
import datetime
import json
import os
from pathlib import Path
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "1773534738374fca2c1b655ec22b96f0a93970cb02d0e6900c37a25ce7ef7ef1"
BASE_PATH = DATA / "v47-profile-build.py"
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[key] for key in ("sha", "require", "save"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立构建目录须在工作区且尚不存在")
    plan_path, baseline_path = DATA / "v59-plan.json", DATA / "v59-baseline.json"
    require(sha(plan_path) == PLAN_SHA, "冻结计划变化")
    plan, identity = read(plan_path), read(baseline_path)
    require(sha(baseline_path) == plan["baseline_identity_sha256"], "当前基线身份变化")
    parent_path = DATA / "ipm-v57-identity.json"
    parent = read(parent_path)
    require(sha(parent_path) == plan["parent_identity_sha256"] == identity["parent_identity_sha256"], "父身份变化")
    require(identity["files"] == parent["files"] and len(identity["files"]) == 95, "当前源码不等于冻结 v57")
    archive = DATA / identity["parent_source_archive"]
    require(sha(archive) == identity["source_zip_sha256"] == parent["source_zip_sha256"], "父源码包变化")
    production, frozen_parent = Path(identity["solver_path"]), ROOT / ".tools/solver-ipm-v57.exe"
    require(str(production) == plan["baseline_program"] and sha(production) == identity["solver_sha256"], "基线程序变化")
    require(sha(frozen_parent) == plan["parent_solver_sha256"] == parent["solver_sha256"], "冻结 v57 程序变化")
    sources = {str(ROOT / name): value for name, value in identity["files"].items()}
    libraries = {str(ROOT / name): value for name, value in identity["libraries"].items()}
    require(len(libraries) == 17, "实际库数量变化")
    third_party = {name: value for name, value in identity["libraries"].items() if not Path(name).name.startswith("libsolver_")}
    require(len(third_party) == 8 and all(value == parent["libraries"][name] for name, value in third_party.items()), "第三方库变化")
    require_hashes(sources)
    require_hashes(libraries)
    compiler = Path(identity["compiler"])
    require(str(compiler) == parent["compiler"] and sha(compiler) == identity["compiler_sha256"] == parent["compiler_sha256"], "编译器变化")
    build_dir = ROOT / "build/release"
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"], text=True, encoding="utf-8")
    lines = command_text.splitlines()
    matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/src/ipm.cpp")]
    require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "实际 ipm 编译命令不唯一")
    original_compile = {"ipm": matches[0]}
    links = [line for line in lines if " -o solver.exe " in line]
    require(links == [identity["link_command"]], "实际链接命令变化")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c", "链接包装变化")
    inner = outer[2]
    require(inner.startswith("cd . && ") and inner.endswith(" && cd ."), "链接目录包装变化")
    original_link = split_windows(inner[len("cd . && "):-len(" && cd .")])
    explicit = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    helpers = [DATA / name for name in ("v59-kkt-instrumentation.py", "v59-kkt-diagnostic.hpp", "v59-kkt-self-test.cpp", "v59-analyze-kkt.py", "v53-analyze-hsd.py", "v59-run.py")]
    protected = file_hashes([production, frozen_parent, compiler, ninja, baseline_path, parent_path, archive,
        BASE_PATH, Path(__file__), plan_path, regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", *explicit, *helpers])
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(archive) as bundle:
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in bundle.namelist()), "源码包路径越界")
        bundle.extractall(frozen)
    require_hashes({str(frozen / name): value for name, value in identity["files"].items()})
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    helper = runpy.run_path(str(DATA / "v59-kkt-instrumentation.py"))
    source_path = frozen / "src/ipm.cpp"
    source = source_path.read_text(encoding="utf-8")
    checks = {"ipm": helper["self_check"](source)}
    transformed, anchors = helper["transform"](source)
    destination = output / "ipm-diagnostic.cpp"
    destination.write_text(transformed, encoding="utf-8")
    obj = output / "ipm-diagnostic.obj"
    compile_command = compile_arguments(original_compile["ipm"], destination, obj, frozen, ROOT)
    compile_command.append("-I" + str(DATA))
    transforms = {"ipm": {"source": str(source_path), "source_sha256": sha(source_path), "patched": str(destination),
        "patched_sha256": sha(destination), "anchors": anchors}}
    save(output / "transforms.json", transforms)
    save(output / "static-check.json", checks)
    executable = output / "solver-diagnostic.exe"
    link, replaced = [], 0
    for value in original_link:
        if value == "libsolver_ipm.a":
            link.extend([str(obj), str(regularized)])
            replaced += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(replaced == 1, "ipm 对象替换不唯一")
    link[link.index("-o") + 1] = str(executable)
    commands = {"cwd": str(build_dir), "original_compile": original_compile, "original_link": original_link,
        "compile": {"ipm": compile_command}, "link": link}
    save(output / "commands.json", commands)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SOLVER_V")}
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    execute(compile_command, build_dir, output / "ipm-compile.json", environment)
    compiled_inputs = file_hashes(dependency_paths(Path(str(obj) + ".d"), build_dir))
    final_inputs = [resolve(value, build_dir) for value in link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    before = file_hashes(final_inputs)
    execute(link, build_dir, output / "link.json", environment)
    self_executable = output / "kkt-self-test.exe"
    self_command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
        "-MMD", "-MF", str(output / "kkt-self-test.d"), "-I" + str(DATA), "-I" + str(frozen / "include"),
        str(DATA / "v59-kkt-self-test.cpp"), "-o", str(self_executable)]
    execute(self_command, build_dir, output / "kkt-self-build.json", environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / "kkt-self-test.d", build_dir)))
    for mode in ("off", "on"):
        env = environment.copy()
        if mode == "on":
            env[plan["diagnostic_environment"]] = str(output / "kkt-self-on")
        execute([str(self_executable)], ROOT, output / f"kkt-self-{mode}-run.json", env)
    execute([str(ROOT / ".tools/python-3.14.8/python.exe"), str(DATA / "v59-analyze-kkt.py"), "--self-test",
        "--output", str(output / "kkt-math-self-test-result.json")], ROOT, output / "kkt-math-self-test.json", environment)
    for values in (protected, sources, libraries, before, compiled_inputs):
        require_hashes(values)
    outputs = [executable, obj, self_executable, *output.glob("*.json"), *output.glob("*.bin"), destination, output / "ninja-commands.txt"]
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "plan_sha256": PLAN_SHA, "baseline_identity_sha256": sha(baseline_path),
        "production_executable": str(production), "production_sha256": sha(production),
        "frozen_parent_executable": str(frozen_parent), "frozen_parent_sha256": sha(frozen_parent),
        "executable": str(executable), "executable_sha256": sha(executable), "self_test_compile": self_command,
        "frozen_files": identity["files"], "actual_libraries": identity["libraries"],
        "protected_before_and_after": protected, "link_explicit_before_and_after": before,
        "compile_inputs_after_and_final": compiled_inputs, "implicit_link_inputs_verified": False,
        "outputs": file_hashes(outputs), "commands": commands,
        "identity_limit": "显式输入核对前后身份；编译依赖在依赖文件产生后哈希并末尾复核；未逐项核验隐式运行库。"}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"executable_sha256": manifest["executable_sha256"], "sources": len(sources),
        "libraries": len(libraries), "third_party_unchanged": len(third_party), "explicit_link_inputs": len(before), "lp_runs": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v59-diagnostic")
    build(parser.parse_args().output.resolve())
