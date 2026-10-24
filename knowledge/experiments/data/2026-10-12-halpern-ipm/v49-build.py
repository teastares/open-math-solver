"""从冻结 v48 源码构建独立失败诊断和矩阵回放程序，不修改生产构建。"""
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
PLAN_SHA = "f3b1871230ae2062b2a4ccd1bea4bcc4ab4d158caf05f5fc28a2bd7dab9becc7"
CLARIFICATION_SHA = "71f12b6e76f1d12e4e8554f5e6310927145e5933c1f8e6b8a1fc8ffe4b59b054"
# 只复用已验过的 Windows 参数解析、编译参数替换、执行与依赖文件解析。
BASE_PATH = DATA / "v47-profile-build.py"
BASE = runpy.run_path(str(BASE_PATH))
sha, require, save = (BASE[key] for key in ("sha", "require", "save"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, execute, dependency_paths = (BASE[key] for key in ("compile_arguments", "execute", "dependency_paths"))


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "独立构建目录必须位于工作区且尚不存在")
    require(sha(DATA / "v49-plan.json") == PLAN_SHA, "冻结计划发生变化")
    require(sha(DATA / "v49-diagnostic-clarification.json") == CLARIFICATION_SHA, "诊断澄清发生变化")
    identity_path = DATA / "ipm-v48-identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    plan = json.loads((DATA / "v49-plan.json").read_text(encoding="utf-8"))
    require(sha(identity_path) == plan["parent_identity_sha256"], "父身份变化")
    archive = DATA / "ipm-v48-source.zip"
    require(sha(archive) == identity["source_zip_sha256"], "父源码包变化")
    production = ROOT / ".tools/solver-ipm-v48.exe"
    require(sha(production) == identity["solver_sha256"], "父程序变化")
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
    original_compile = {}
    for unit, filename in (("ipm", "ipm.cpp"), ("normal", "normal_cholesky.cpp")):
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/src/" + filename)]
        require(len(matches) == 1, "编译命令不唯一：" + filename)
        original_compile[unit] = matches[0]
        require(resolve(matches[0][0], build_dir) == compiler, "编译器与冻结身份不同")
    links = [line for line in lines if " -o solver.exe " in line]
    require(links == [identity["link_command"]], "实际链接命令与冻结身份不同")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c", "链接包装改变")
    inner = outer[2]
    require(inner.startswith("cd . && ") and inner.endswith(" && cd ."), "链接目录包装改变")
    original_link = split_windows(inner[len("cd . && "):-len(" && cd .")])
    explicit = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    helpers = [DATA / name for name in ("v49-normal-instrumentation.py", "v49-normal-diagnostic.hpp", "v49-factor-replay.cpp", "v49-hsd-instrumentation.py", "v49-hsd-diagnostic.hpp", "v49-analyze-hsd.py", "v49-hsd-self-test.cpp")]
    protected = file_hashes([production, compiler, ninja, identity_path, archive, BASE_PATH, Path(__file__),
        DATA / "v49-plan.json", DATA / "v49-diagnostic-clarification.json", regularized,
        build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", *explicit, *helpers])
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(archive) as bundle:
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in bundle.namelist()), "源码包路径越界")
        bundle.extractall(frozen)
    require_hashes({str(frozen / name): value for name, value in identity["files"].items()})
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    objects, transforms, commands, checks = {}, {}, {}, {}
    for unit, filename, module in (("ipm", "ipm.cpp", "hsd"), ("normal", "normal_cholesky.cpp", "normal")):
        helper = runpy.run_path(str(DATA / f"v49-{module}-instrumentation.py"))
        source_path = frozen / "src" / filename
        source = source_path.read_text(encoding="utf-8")
        checks[unit] = helper["self_check"](source)
        transformed, anchors = helper["transform"](source)
        destination = output / f"{unit}-diagnostic.cpp"
        destination.write_text(transformed, encoding="utf-8")
        objects[unit] = output / f"{unit}-diagnostic.obj"
        commands[unit] = compile_arguments(original_compile[unit], destination, objects[unit], frozen, ROOT)
        commands[unit].append("-I" + str(DATA))
        transforms[unit] = {"source": str(source_path), "source_sha256": sha(source_path), "patched": str(destination),
            "patched_sha256": sha(destination), "anchors": anchors}
    save(output / "transforms.json", transforms)
    save(output / "static-check.json", checks)
    executable = output / "solver-diagnostic.exe"
    replacements = {"libsolver_ipm.a": [str(objects["ipm"]), str(regularized)], "libsolver_normal_cholesky.a": [str(objects["normal"])]}
    replaced = dict.fromkeys(replacements, 0)
    link = []
    for value in original_link:
        if value in replacements:
            link.extend(replacements[value])
            replaced[value] += 1
        elif not value.startswith("-Wl,--out-implib,"):
            link.append(value)
    require(all(count == 1 for count in replaced.values()), "独立对象替换不唯一")
    link[link.index("-o") + 1] = str(executable)
    replay_object = output / "factor-replay.obj"
    replay_executable = output / "factor-replay.exe"
    replay_compile = compile_arguments(original_compile["normal"], DATA / "v49-factor-replay.cpp", replay_object, frozen, ROOT)
    replay_compile.append("-I" + str(DATA))
    replay_link = []
    for value in original_link:
        if value.startswith("-Wl,--out-implib,") or value == "-municode":
            continue
        if value.endswith("CMakeFiles/solver.dir/src/main.cpp.obj"):
            replay_link.append(str(replay_object))
        else:
            replay_link.append(value)
    require(str(replay_object) in replay_link, "没有替换回放程序入口")
    replay_link[replay_link.index("-o") + 1] = str(replay_executable)
    command_manifest = {"cwd": str(build_dir), "original_compile": original_compile, "original_link": original_link,
        "compile": commands, "link": link, "replay_compile": replay_compile, "replay_link": replay_link}
    save(output / "commands.json", command_manifest)
    environment = os.environ.copy()
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    inputs = []
    for unit, command in [*commands.items(), ("replay", replay_compile)]:
        execute(command, build_dir, output / f"{unit}-compile.json", environment)
        obj = replay_object if unit == "replay" else objects[unit]
        inputs.extend(dependency_paths(Path(str(obj) + ".d"), build_dir))
    compiled_inputs = file_hashes(inputs)
    explicit_final = [resolve(value, build_dir) for value in link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    before = file_hashes(explicit_final)
    execute(link, build_dir, output / "link.json", environment)
    execute(replay_link, build_dir, output / "replay-link.json", environment)
    self_executable = output / "hsd-self-test.exe"
    self_command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
        "-MMD", "-MF", str(output / "hsd-self-test.d"), "-I" + str(DATA), str(DATA / "v49-hsd-self-test.cpp"), "-o", str(self_executable)]
    execute(self_command, build_dir, output / "hsd-self-build.json", environment)
    compiled_inputs.update(file_hashes(dependency_paths(output / "hsd-self-test.d", build_dir)))
    for mode in ("off", "on"):
        env = environment.copy()
        env.pop("SOLVER_V49_DIAGNOSTICS", None)
        if mode == "on":
            env["SOLVER_V49_DIAGNOSTICS"] = str(output / "hsd-self-on")
        execute([str(self_executable)], ROOT, output / f"hsd-self-{mode}-run.json", env)
    execute([str(replay_executable), "--self-test"], ROOT, output / "factor-self-test.json", environment)
    execute([str(ROOT / ".tools/python-3.14.8/python.exe"), str(DATA / "v49-analyze-hsd.py"), "--self-test"], ROOT, output / "hsd-math-self-test.json", environment)
    require_hashes(protected)
    require_hashes(sources)
    require_hashes(libraries)
    require_hashes(before)
    require_hashes(compiled_inputs)
    outputs = [executable, replay_executable, replay_object, self_executable, *objects.values(), *output.glob("*.json"), *output.glob("*.bin"), *output.glob("*-diagnostic.cpp"), output / "ninja-commands.txt"]
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "plan_sha256": PLAN_SHA, "clarification_sha256": CLARIFICATION_SHA, "production_executable": str(production),
        "production_sha256": sha(production), "executable": str(executable), "executable_sha256": sha(executable),
        "replay_executable": str(replay_executable), "replay_executable_sha256": sha(replay_executable),
        "self_test_compile": self_command,
        "frozen_files": identity["files"], "protected_before_and_after": protected, "link_explicit_before_and_after": before,
        "compile_inputs_after_and_final": compiled_inputs, "implicit_link_inputs_verified": False,
        "outputs": file_hashes(outputs), "commands": command_manifest,
        "identity_limit": "显式输入核对前后身份；编译依赖在依赖文件产生后哈希与末尾复核；未逐项核验隐式运行库。"}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"executable_sha256": manifest["executable_sha256"], "replay_sha256": manifest["replay_executable_sha256"], "sources": len(sources), "explicit_link_inputs": len(before)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v49-diagnostic")
    build(parser.parse_args().output.resolve())
