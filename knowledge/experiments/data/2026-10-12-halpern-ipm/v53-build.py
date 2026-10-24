"""从冻结v52源码构建单翻译单元HSD方向诊断，不改生产构建。"""
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
PLAN_SHA = "87fd87be5bc80deb0ee7689d5ad16635969b447e836f88fc8bb9019479f5cdf0"
BASE_PATH = DATA / "v47-profile-build.py"
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[key] for key in ("sha", "require", "save"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立构建目录须在工作区且尚不存在")
    require(sha(DATA / "v53-plan.json") == PLAN_SHA, "冻结计划变化")
    identity_path = DATA / "ipm-v52-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    plan = json.loads((DATA / "v53-plan.json").read_text(encoding="utf-8"))
    require(sha(identity_path) == plan["parent_identity_sha256"], "父身份变化")
    archive = DATA / "ipm-v52-source.zip"
    require(sha(archive) == identity["source_zip_sha256"], "父源码包变化")
    production = ROOT / ".tools/solver-ipm-v52.exe"
    require(sha(production) == plan["parent_solver_sha256"] == identity["solver_sha256"], "父程序变化")
    sources = {str(ROOT / name): value for name, value in identity["files"].items()}
    libraries = {str(ROOT / name): value for name, value in identity["libraries"].items()}
    require_hashes(sources)
    require_hashes(libraries)
    compiler = Path(identity["compiler"])
    require(sha(compiler) == identity["compiler_sha256"], "编译器变化")
    build_dir = ROOT / "build/release"
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    command_text = subprocess.check_output([str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"], text=True, encoding="utf-8")
    lines = command_text.splitlines()
    matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/src/ipm.cpp")]
    require(len(matches) == 1 and resolve(matches[0][0], build_dir) == compiler, "实际ipm编译器或命令不唯一")
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
    helpers = [DATA / name for name in ("v53-hsd-instrumentation.py", "v53-hsd-diagnostic.hpp", "v53-hsd-self-test.cpp", "v53-analyze-hsd.py")]
    protected = file_hashes([production, compiler, ninja, identity_path, archive, BASE_PATH, Path(__file__),
        DATA / "v53-plan.json", regularized, build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", *explicit, *helpers])
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(archive) as bundle:
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in bundle.namelist()), "源码包路径越界")
        bundle.extractall(frozen)
    require_hashes({str(frozen / name): value for name, value in identity["files"].items()})
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    helper = runpy.run_path(str(DATA / "v53-hsd-instrumentation.py"))
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
    require(replaced == 1, "ipm对象替换不唯一")
    link[link.index("-o") + 1] = str(executable)
    commands = {"cwd": str(build_dir), "original_compile": original_compile, "original_link": original_link,
        "compile": {"ipm": compile_command}, "link": link}
    save(output / "commands.json", commands)
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    for key in tuple(environment):
        if key.startswith("SOLVER_V"):
            environment.pop(key)
    execute(compile_command, build_dir, output / "ipm-compile.json", environment)
    compiled_inputs = file_hashes(dependency_paths(Path(str(obj) + ".d"), build_dir))
    final_inputs = [resolve(value, build_dir) for value in link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    before = file_hashes(final_inputs)
    execute(link, build_dir, output / "link.json", environment)
    self_executable = output / "hsd-self-test.exe"
    self_command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
        "-MMD", "-MF", str(output / "hsd-self-test.d"), "-I" + str(DATA), "-I" + str(frozen / "include"),
        str(DATA / "v53-hsd-self-test.cpp"), "-o", str(self_executable)]
    execute(self_command, build_dir, output / "hsd-self-build.json", environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / "hsd-self-test.d", build_dir)))
    for mode in ("off", "on"):
        env = environment.copy()
        if mode == "on":
            env["SOLVER_V53_DIAGNOSTICS"] = str(output / "hsd-self-on")
        execute([str(self_executable)], ROOT, output / f"hsd-self-{mode}-run.json", env)
    execute([str(ROOT / ".tools/python-3.14.8/python.exe"), str(DATA / "v53-analyze-hsd.py"), "--self-test"], ROOT, output / "hsd-math-self-test.json", environment)
    for values in (protected, sources, libraries, before, compiled_inputs):
        require_hashes(values)
    outputs = [executable, obj, self_executable, *output.glob("*.json"), *output.glob("*.bin"), destination, output / "ninja-commands.txt"]
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "plan_sha256": PLAN_SHA, "production_executable": str(production), "production_sha256": sha(production),
        "executable": str(executable), "executable_sha256": sha(executable), "self_test_compile": self_command,
        "frozen_files": identity["files"], "protected_before_and_after": protected, "link_explicit_before_and_after": before,
        "compile_inputs_after_and_final": compiled_inputs, "implicit_link_inputs_verified": False,
        "outputs": file_hashes(outputs), "commands": commands,
        "identity_limit": "显式输入核对前后身份；编译依赖在依赖文件产生后哈希并末尾复核；未逐项核验隐式运行库。"}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"executable_sha256": manifest["executable_sha256"], "sources": len(sources), "libraries": len(libraries), "explicit_link_inputs": len(before)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v53-diagnostic")
    build(parser.parse_args().output.resolve())
