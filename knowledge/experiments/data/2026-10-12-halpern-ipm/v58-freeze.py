"""正式18项测试通过后冻结第55轮来源；核实仅五个源文件变化，自有库重编须含IPM且第三方库不变。"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "dc1959b18e9c5fdaf5a61b2a3487d8ce3588c7fb5236ff06a1e78e3929cf1e7b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    executable = ROOT / ".tools/solver-ipm-v58.exe"
    identity_path = DATA / "ipm-v58-identity.json"
    archive_path = DATA / "ipm-v58-source.zip"
    logs = {
        "iteration-ipm-v58-build.log": ROOT / ".tools/iteration-ipm-v58-build.log",
        "iteration-ipm-v58-ctest-detail.log": ROOT / "build/release/Testing/Temporary/LastTest.log",
    }
    for path in (executable, identity_path, archive_path, *(DATA / name for name in logs)):
        assert not path.exists(), "不能覆盖已经冻结的来源：" + str(path)
    build = logs["iteration-ipm-v58-build.log"].read_text(encoding="utf-8")
    assert "100% tests passed out of 18" in build, "正式构建测试尚未全部通过"
    assert sha(DATA / "v58-plan.json") == PLAN_SHA
    plan = read(DATA / "v58-plan.json")
    assert sha(DATA / "ipm-v57-identity.json") == plan["parent_identity_sha256"]
    assert sha(DATA / "v57-verification.json") == plan["evidence_sha256"]
    parent = read(DATA / "ipm-v57-identity.json")
    assert sha(ROOT / ".tools/solver-ipm-v57.exe") == parent["solver_sha256"]
    names = sorted(parent["files"])
    assert len(names) == len(set(names)) == 95
    files = {name: sha(ROOT / name) for name in names}
    changed = sorted(name for name in names if files[name] != parent["files"][name])
    assert changed == sorted(plan["changed_source_files"]) == sorted(("include/solver/backend.hpp", "src/ipm.cpp", "src/main.cpp", "tests/test_ipm.cpp", "tests/test_cli.py")), "源码变化必须恰为冻结五文件"
    compiler = Path(parent["compiler"])
    assert sha(compiler) == parent["compiler_sha256"], "编译器不得改变"
    self_check = read(DATA / "v58-runner-self-check.json")
    assert self_check == {"passed": True, "command_cases": 4, "configuration_cases": 2, "gate_cases": 10, "solver_runs": 0}, "实际测前运行器自检不完整"
    commands = subprocess.check_output([str(ROOT / ".tools/ninja-1.13.2/ninja.exe"), "-C", str(ROOT / "build/release"), "-t", "commands", "solver.exe"], text=True, encoding="utf-8").splitlines()
    links = [line for line in commands if " -o solver.exe " in line]
    assert len(links) == 1
    # 仅提取真实静态库输入，排除链接器输出选项。
    names_in_link = [quoted or plain for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', links[0])]
    paths = sorted({(ROOT / "build/release" / name).resolve() for name in names_in_link if not name.startswith("-")}, key=str)
    assert len(paths) == 17 and all(path.is_file() and path.is_relative_to(ROOT) for path in paths)
    libraries = [path.relative_to(ROOT).as_posix() for path in paths]
    current_libraries = {name: sha(ROOT / name) for name in libraries}
    assert set(current_libraries) == set(parent["libraries"])
    changed_libraries = {name for name in libraries if current_libraries[name] != parent["libraries"][name]}
    external = {name for name in libraries if not Path(name).name.startswith("libsolver_")}
    assert len(external) == 8 and all(current_libraries[name] == parent["libraries"][name] for name in external), "8个第三方库不得变化"
    assert "build/release/libsolver_ipm.a" in changed_libraries and not (changed_libraries & external), "实际IPM实现必须重编，变化只能来自自有库"
    patched_source = ROOT / "build/openblas/openblas-0.3.33/src/solver_openblas_build/kernel/x86_64/dgemm_kernel_16x2_skylakex.S"
    assert sha(patched_source) == parent["patched_openblas_source_sha256"], "BLAS修补来源不得改变"
    solver_sha = sha(ROOT / "build/release/solver.exe")
    assert solver_sha != parent["solver_sha256"], "新生产实现必须进入实际程序"
    shutil.copy2(ROOT / "build/release/solver.exe", executable)
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    with zipfile.ZipFile(archive_path) as archive:
        assert {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()} == files
    for name, source in logs.items():
        shutil.copy2(source, DATA / name)
    identity = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "solver_sha256": solver_sha, "files": files, "source_zip_sha256": sha(archive_path),
        "compiler": str(compiler), "compiler_sha256": sha(compiler), "libraries": current_libraries,
        "changed_source_files": changed, "changed_libraries": sorted(changed_libraries),
        "logs": {name: sha(DATA / name) for name in logs}, "plan_sha256": PLAN_SHA,
        "runner_sha256": sha(DATA / "v58-run.py"), "link_command": links[0],
        "runner_self_check_sha256": sha(DATA / "v58-runner-self-check.json"),
        "freeze_script_sha256": sha(Path(__file__)), "patched_openblas_source_sha256": sha(patched_source),
    }
    assert files == {name: sha(ROOT / name) for name in names}, "冻结期间来源发生改变"
    assert current_libraries == {name: sha(ROOT / name) for name in libraries}, "冻结期间链接库发生改变"
    assert sha(executable) == sha(ROOT / "build/release/solver.exe") == solver_sha
    with identity_path.open("x", encoding="utf-8") as target:
        json.dump(identity, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"source_files": len(files), "solver_sha256": identity["solver_sha256"]}), flush=True)


if __name__ == "__main__":
    freeze()
