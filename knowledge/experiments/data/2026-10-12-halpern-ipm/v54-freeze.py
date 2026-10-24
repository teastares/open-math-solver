"""仅在正式构建与测试通过后冻结第51轮来源，不覆盖任何已有实验身份。"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    executable = ROOT / ".tools/solver-ipm-v54.exe"
    identity_path = DATA / "ipm-v54-identity.json"
    archive_path = DATA / "ipm-v54-source.zip"
    logs = {
        "iteration-ipm-v54-build.log": ROOT / ".tools/iteration-ipm-v54-build.log",
        "iteration-ipm-v54-ctest-detail.log": ROOT / "build/release/Testing/Temporary/LastTest.log",
    }
    for path in (executable, identity_path, archive_path, *(DATA / name for name in logs)):
        assert not path.exists(), "不能覆盖已经冻结的来源：" + str(path)
    build = logs["iteration-ipm-v54-build.log"].read_text(encoding="utf-8")
    assert "100% tests passed out of 18" in build, "正式构建测试尚未全部通过"
    names = sorted(read(DATA / "ipm-v52-identity.json")["files"])
    assert len(names) == len(set(names)) == 95
    files = {name: sha(ROOT / name) for name in names}
    plan = read(DATA / "v54-plan.json")
    assert sha(DATA / "v54-plan.json") == "7475487f6d1cb8a6e63f04384292b0887df8cee8c143de482d1e7bf97091c8bf"
    assert sha(DATA / "ipm-v52-identity.json") == plan["parent_identity_sha256"]
    parent = read(DATA / "ipm-v52-identity.json")
    assert sorted(name for name in names if files[name] != parent["files"][name]) == sorted(plan["changed_source_files"])
    compiler = Path(read(DATA / "ipm-v52-identity.json")["compiler"])
    commands = subprocess.check_output([str(ROOT / ".tools/ninja-1.13.2/ninja.exe"), "-C", str(ROOT / "build/release"), "-t", "commands", "solver.exe"], text=True, encoding="utf-8").splitlines()
    links = [line for line in commands if " -o solver.exe " in line]
    assert len(links) == 1
    # 当前Ninja使用带引号的路径或无空白参数；只取真实链接命令中的静态库。
    names_in_link = [quoted or plain for quoted, plain in re.findall(r'"([^"\n]+\.a)"|([^\s";]+\.a)(?=\s|$)', links[0])]
    # 排除 -Wl,--out-implib,*.a 等输出选项；真实输入若缺失仍由后续断言拒绝。
    paths = sorted({(ROOT / "build/release" / name).resolve() for name in names_in_link if not name.startswith("-")}, key=str)
    assert len(paths) >= 17 and all(path.is_file() and path.is_relative_to(ROOT) for path in paths)
    libraries = [path.relative_to(ROOT).as_posix() for path in paths]
    assert sum(path.name == "libopenblas.a" for path in paths) == 1
    assert any(path.name == "libcholmod.a" for path in paths)
    assert any(path.name == "libcolamd.a" for path in paths)
    current_libraries = {name: sha(ROOT / name) for name in libraries}
    assert set(current_libraries) == set(parent["libraries"])
    for name in libraries:
        if not Path(name).name.startswith("libsolver_"):
            assert current_libraries[name] == parent["libraries"][name], "第三方库发生变化：" + name
    shutil.copy2(ROOT / "build/release/solver.exe", executable)
    with zipfile.ZipFile(archive_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    for name, source in logs.items():
        shutil.copy2(source, DATA / name)
    identity = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "solver_sha256": sha(executable), "files": files,
        "source_zip_sha256": sha(archive_path),
        "compiler": str(compiler), "compiler_sha256": sha(compiler),
        "libraries": current_libraries,
        "logs": {name: sha(DATA / name) for name in logs},
        "plan_sha256": sha(DATA / "v54-plan.json"),
        "runner_sha256": sha(DATA / "v54-run.py"),
        "link_command": links[0], "runner_self_check_sha256": sha(DATA / "v54-runner-self-check.json"),
        "freeze_script_sha256": sha(Path(__file__)),
        "patched_openblas_source_sha256": sha(ROOT / "build/openblas/openblas-0.3.33/src/solver_openblas_build/kernel/x86_64/dgemm_kernel_16x2_skylakex.S"),
    }
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert files == {name: sha(ROOT / name) for name in names}, "冻结期间来源发生改变"
    assert sha(executable) == sha(ROOT / "build/release/solver.exe")
    print(json.dumps({"source_files": len(files), "solver_sha256": identity["solver_sha256"]}), flush=True)


if __name__ == "__main__":
    freeze()
