#!/usr/bin/env python3
"""逐字快照并编译直接维护的实验C++；不生成或补丁数学源码。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "knowledge/experiments/data/2026-10-12-halpern-ipm"
TEMPLATE = ROOT / ".tools/v100-presolve/build-manifest.json"
TEMPLATE_SHA = "c227a1d45337788325db93ac6be02a811d6cb8d1f7550696378bb6cc72717d16"
CANONICAL = ("src/experimental/ipm.cpp", "src/experimental/presolve_state.hpp", "src/experimental/coupled_state.hpp",
    "include/solver/experimental/coupled.hpp", "include/solver/experimental/energy.hpp", "include/solver/experimental/legacy_coupled.hpp",
    "tests/experimental/ipm_self_test.cpp", "tests/experimental/coupled_self_test.cpp")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(text):
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
    return parse(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def hashes(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def verify(mapping):
    for path, digest in mapping.items():
        require(sha(path) == digest, "身份变化：" + str(path))


def equal(a, b):
    return type(a) is type(b) and (a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a) if type(a) is dict else
        len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b)) if type(a) is list else a == b)


def environment():
    result = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    result.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"))
    return result


def canonical_relative(name):
    require(type(name) is str and name and Path(name).parts and not Path(name).is_absolute() and not Path(name).drive and
        not Path(name).root and ".." not in Path(name).parts,
        "canonical必须是无驱动器、根和父目录跳转的相对路径")
    return Path(name)


def static_check(plan_path=None):
    require(sha(TEMPLATE) == TEMPLATE_SHA, "冻结编译命令模板变化")
    template, legacy = read(TEMPLATE), read(DATA / "ipm-v66-identity.json")
    require(len(legacy["files"]) == 95 and len(legacy["libraries"]) == 17, "冻结生产95来源/17库不符")
    production = {str(ROOT / name): digest for group in ("files", "libraries") for name, digest in legacy[group].items()}
    production.update({str(ROOT / "build/release/solver.exe"): legacy["solver_sha256"],
        str(ROOT / ".tools/solver-ipm-v66.exe"): legacy["solver_sha256"], legacy["compiler"]: legacy["compiler_sha256"]})
    verify(production)
    require(sha(DATA / "ipm-v66-source.zip") == legacy["source_zip_sha256"], "冻结基础源码包变化")
    plan = read(plan_path) if plan_path else None
    names = tuple(item["canonical"] for item in plan["sources"]) if plan is not None else CANONICAL
    for name in names:
        canonical_relative(name)
    require(len(names) == len(set(names)) and all((ROOT / name).resolve().is_relative_to(ROOT) for name in names) and
        {CANONICAL[0], CANONICAL[6], CANONICAL[7]} <= set(names), "源清单路径、唯一性或三个真实CPP不符")
    origins = {name: sha(ROOT / name) for name in names}
    if plan is not None:
        require(plan["schema"] == 1 and plan["production_changes"] is False and type(plan["new_algorithm_round"]) is bool and
            plan["source_count"] == len(names) and
            sha(DATA / "ipm-v66-identity.json") == plan["parent_identity_sha256"], "迁移计划来源/身份不符")
        for item in plan["sources"]:
            require(origins[item["canonical"]] == item["canonical_sha256"], "活动源不符合本次冻结计划")
            if plan["new_algorithm_round"]:
                continue
            require(item["otherwise_byte_identical"] is True, "机械迁移须明确原数学字节保持")
            old = ROOT / item["source"]
            require(sha(old) == item["source_sha256"], "冻结迁移父源码变化")
            expected = old.read_bytes()
            for before, after in item["include_rebindings"]:
                before, after = ('#include "' + before + '"').encode(), ('#include "' + after + '"').encode()
                require(expected.count(before) == 1, "迁移include绑定必须唯一")
                expected = expected.replace(before, after, 1)
            require(expected == (ROOT / item["canonical"]).read_bytes(), "迁移只能改变声明的include，不改数学字节")
    require(set(template["commands"]["compile"]) == {"ipm"} and set(template["commands"]["math_self_tests"]) == {"ipm", "coupled"}, "真实命令模板角色不符")
    return dict(passed=True, public_lp_runs=0, source_count=len(names), source_origins_at_build=origins,
        command_template_sha256=TEMPLATE_SHA, production_sources=95, production_libraries=17), template, legacy, production, plan


def adapt_compile(command, source, obj, output, template):
    command = list(command)
    old_work = Path(template["executable"]).parent
    for option, value in (("-c", source), ("-o", obj), ("-MF", Path(str(obj) + ".d")), ("-MT", obj)):
        if option in command:
            command[command.index(option) + 1] = str(value)
    old_include = "-I" + str(old_work)
    require(command.count(old_include) == 1, "私有include模板必须唯一")
    command[command.index(old_include)] = "-I" + str(output / "source/include")
    frozen_include = "-I" + str(old_work / "frozen/include")
    if frozen_include in command:
        command[command.index(frozen_include)] = "-I" + str(output / "frozen/include")
        command.insert(1, "-I" + str(output / "source/src/experimental"))
    return command


def dependencies(path, cwd):
    text = re.sub(r"\\\r?\n", "", path.read_text(encoding="utf-8"))
    require(": " in text, "编译依赖文件格式无效")
    tokens = re.findall(r"(?:\\.|[^\s])+", text.split(": ", 1)[1])
    return {((cwd / re.sub(r"\\([ #:\\])", r"\1", token)).resolve()) for token in tokens}


def execute(command, cwd, target, env):
    started = time.perf_counter()
    raw = dict(command=command, cwd=str(cwd), returncode=None, stdout="", stderr="", wall_timeout=False)
    try:
        process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=300, check=False)
        raw.update(returncode=process.returncode, stdout=process.stdout.decode("utf-8", errors="replace"), stderr=process.stderr.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as error:
        raw.update(wall_timeout=True, stdout=(error.stdout or b"").decode("utf-8", errors="replace"), stderr=(error.stderr or b"").decode("utf-8", errors="replace"))
    raw["elapsed_seconds"] = time.perf_counter() - started
    save(target, raw)
    require(raw["returncode"] == 0 and not raw["wall_timeout"], "实际命令失败，原始记录已保留：" + str(target))
    return raw


def build(output, plan_path=None):
    require(output.is_relative_to(ROOT) and not output.exists(), "--output必须解析为工作区中尚不存在的新目录")
    check, template, legacy, protected, plan = static_check(plan_path)
    output.mkdir(parents=True)
    provenance = output / "provenance"; provenance.mkdir()
    for source, name in ((Path(__file__), "build_experimental.py"), (TEMPLATE, "command-template.json"),
        (DATA / "ipm-v66-identity.json", "production-identity.json")):
        (provenance / name).write_bytes(source.read_bytes())
    if plan_path:
        (provenance / "plan.json").write_bytes(plan_path.read_bytes())
    names = tuple(check["source_origins_at_build"])
    for name in names:
        target = output / "source" / canonical_relative(name)
        require(target.resolve().is_relative_to((output / "source").resolve()), "快照目标越出独立source目录")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    snapshot = hashes(output / "source" / name for name in names)
    require(all(snapshot[str((output / "source" / name).resolve())] == check["source_origins_at_build"][name] for name in names), "源快照前后不一致")
    archive_path, frozen = DATA / "ipm-v66-source.zip", output / "frozen"
    with zipfile.ZipFile(archive_path) as archive:
        require(set(archive.namelist()) == set(legacy["files"]) and len(archive.namelist()) == 95 and
            all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), "冻结源码包成员或路径不符")
        archive.extractall(frozen)
    protected.update({str(frozen / name): digest for name, digest in legacy["files"].items()})
    protected.update(snapshot); protected.update(hashes(provenance.iterdir()))
    cwd, compiler = Path(template["commands"]["cwd"]), Path(legacy["compiler"])
    commands = {"compile": {}, "link": {}, "test": {}}
    old_work = Path(template["executable"]).parent
    sources = {"ipm": output / "source/src/experimental/ipm.cpp", "ipm_self": output / "source/tests/experimental/ipm_self_test.cpp",
        "coupled_self": output / "source/tests/experimental/coupled_self_test.cpp"}
    for role, source in sources.items():
        original = template["commands"]["compile"]["ipm"] if role == "ipm" else template["commands"]["math_self_tests"][role.removesuffix("_self")]["compile"]
        commands["compile"][role] = adapt_compile(original, source, output / (role + ".obj"), output, template)
    for role, original in (("solver", template["commands"]["link"]), ("ipm", template["commands"]["math_self_tests"]["ipm"]["link"]),
        ("coupled", template["commands"]["math_self_tests"]["coupled"]["link"])):
        replaced = {str(old_work / "ipm-presolve.obj"): str(output / "ipm.obj"), str(old_work / "ipm-self.obj"): str(output / "ipm_self.obj"),
            str(old_work / "coupled-self.obj"): str(output / "coupled_self.obj")}
        command = [replaced.get(value, value) for value in original]
        command[command.index("-o") + 1] = str(output / ("solver-experimental.exe" if role == "solver" else role + "-self-test.exe"))
        commands["link"][role] = command
    commands["test"] = {role: [str(output / (role + "-self-test.exe"))] for role in ("ipm", "coupled")}
    for command in commands["link"].values():
        for value in command:
            path = (cwd / value).resolve()
            if not value.startswith("-") and path.suffix.lower() in (".a", ".lib", ".o", ".obj") and not path.is_relative_to(output):
                protected[str(path)] = sha(path)
    verify(protected)
    save(output / "build-before.json", dict(schema=1, created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **check, source_snapshot=snapshot, protected_before=protected, commands=commands,
        plan_sha256=sha(plan_path) if plan_path else None, builder_sha256=sha(__file__)))
    env = environment(); env["PATH"] = str(compiler.parent) + os.pathsep + env.get("PATH", "")
    compiled, dependency_records = {}, {}
    expected = {"ipm": set(CANONICAL[:5]), "ipm_self": set(CANONICAL[:5]) | {CANONICAL[6]},
        "coupled_self": set(CANONICAL[3:6]) | {CANONICAL[7]}}
    for role, command in commands["compile"].items():
        execute(command, cwd, output / (role + "-compile.json"), env)
        deps = dependencies(output / (role + ".obj.d"), cwd)
        actual = {path for path in deps if path.is_relative_to(output / "source")}
        allowed = {(output / "source" / name).resolve() for name in names}
        require(actual <= allowed and sources[role].resolve() in actual, "实际.d含未快照的实验依赖或缺真实CPP")
        if plan is not None and not plan["new_algorithm_round"]:
            require(actual == {(output / "source" / name).resolve() for name in expected[role]}, "迁移.d的实验源依赖集合不精确：" + role)
        if role == "ipm_self":
            require(sources["ipm"].resolve() in actual, "IPM自测必须包含实际快照CPP")
        require(not any(path.is_relative_to(DATA) or path.is_relative_to(old_work) or path in {(ROOT / name).resolve() for name in names} or
            path == (ROOT / "src/ipm.cpp").resolve() for path in deps), "实际编译混入旧生成源或活动canonical")
        require(not any(path.is_relative_to(ROOT / "include") for path in deps), "生产公共头必须从冻结include读取")
        current = hashes(deps)
        require(all(path not in compiled or compiled[path] == digest for path, digest in current.items()), "编译期间依赖变化")
        compiled.update(current); dependency_records[role] = dict(file=role + ".obj.d", sha256=sha(output / (role + ".obj.d")), paths=sorted(current))
    for role, command in commands["link"].items():
        execute(command, cwd, output / (role + "-link.json"), env)
    math = {}
    for role in ("ipm", "coupled"):
        command = [str(output / (role + "-self-test.exe"))]; commands["test"][role] = command
        raw = execute(command, ROOT, output / (role + "-self-run.json"), env)
        details = parse(raw["stdout"])
        require(raw["stderr"] == "" and details["passed"] is True and type(details["public_lp_runs"]) is int and details["public_lp_runs"] == 0 and
            type(details["cases"]) is int and details["cases"] == len(details["details"]) > 0, "组件实际输出无效")
        if plan is not None:
            parent = read(old_work / (role + "-self-check.json"))["details"]
            require(details["cases"] == plan["verification"]["component_cases"][role], "实际组件组数不符冻结计划")
            if not plan["new_algorithm_round"]:
                require(equal(details, parent), "机械迁移须完整保持父85+43组件详情")
        math[role] = dict(passed=True, public_lp_runs=0, details=details)
        save(output / (role + "-self-check.json"), math[role])
    verify(protected); verify(compiled)
    require(check["source_origins_at_build"] == {name: sha(ROOT / name) for name in names}, "本次构建期间活动源被并发修改")
    executables = hashes(output / name for name in ("solver-experimental.exe", "ipm-self-test.exe", "coupled-self-test.exe"))
    manifest = dict(schema=1, production_changes=False, source_generation=False, public_lp_runs=0,
        command_template=dict(path=str(provenance / "command-template.json"), sha256=TEMPLATE_SHA),
        source_origins_at_build=check["source_origins_at_build"], source_snapshot=snapshot, protected=protected,
        compile_dependencies=compiled, dependency_records=dependency_records, commands=commands, math_self_tests=math,
        executable=str(output / "solver-experimental.exe"), executable_sha256=sha(output / "solver-experimental.exe"), executables=executables,
        plan_file=str(provenance / "plan.json") if plan else None, plan_sha256=sha(plan_path) if plan_path else None,
        production_sources=legacy["files"], production_libraries=legacy["libraries"], production_solver_sha256=legacy["solver_sha256"],
        outputs=hashes(path for path in output.iterdir() if path.is_file()))
    save(output / "build-manifest.json", manifest)
    immutable = dict(protected); immutable.update(compiled); immutable.update(executables); immutable.update(manifest["outputs"])
    immutable[str(output / "build-manifest.json")] = sha(output / "build-manifest.json")
    save(output / "identity.json", dict(schema=1, candidate=manifest["executable"], candidate_sha256=manifest["executable_sha256"],
        build_manifest=str(output / "build-manifest.json"), build_manifest_sha256=sha(output / "build-manifest.json"), protected=immutable))
    print(json.dumps(dict(passed=True, public_lp_runs=0, source_count=len(names), compiled_units=3, component_cases={key: value["details"]["cases"] for key, value in math.items()},
        executable=manifest["executable"], executable_sha256=manifest["executable_sha256"]), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", type=Path, help="可选冻结迁移计划；后续直接维护源码的构建可不传")
    parser.add_argument("--self-check", action="store_true", help="只检查来源、include绑定和冻结依赖，不编译或运行LP")
    args = parser.parse_args()
    plan_path = (ROOT / args.plan).resolve() if args.plan else None
    if args.self_check:
        print(json.dumps(static_check(plan_path)[0], ensure_ascii=False))
    else:
        require(args.output is not None, "构建必须显式指定--output新目录")
        build((ROOT / args.output).resolve(), plan_path)
