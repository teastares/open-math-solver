"""仅在正式构建与测试通过后冻结第41轮来源，不覆盖任何已有实验身份。"""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    executable = ROOT / ".tools/solver-ipm-v44.exe"
    identity_path = DATA / "ipm-v44-identity.json"
    archive_path = DATA / "ipm-v44-source.zip"
    logs = {
        "iteration-ipm-v44-build.log": ROOT / ".tools/iteration-ipm-v44-build.log",
        "iteration-ipm-v44-ctest-detail.log": ROOT / "build/release/Testing/Temporary/LastTest.log",
    }
    for path in (executable, identity_path, archive_path, *(DATA / name for name in logs)):
        assert not path.exists(), "不能覆盖已经冻结的来源：" + str(path)
    build = logs["iteration-ipm-v44-build.log"].read_text(encoding="utf-8")
    assert "100% tests passed out of 17" in build, "正式构建测试尚未全部通过"
    names = sorted([*read(DATA / "ipm-v43-identity.json")["files"], "include/solver/detail/ipm_homogeneous.hpp"])
    assert len(names) == len(set(names)) == 90
    files = {name: sha(ROOT / name) for name in names}
    plan = read(DATA / "v44-plan.json")
    assert plan["runner_sha256"] == sha(DATA / "v44-run-experiments.py")
    compiler = Path(read(DATA / "v40-progress-identity.json")["compiler"])
    libraries = read(DATA / "v40-progress-identity.json")["libraries"]
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
        "libraries": {name: sha(ROOT / name) for name in libraries},
        "logs": {name: sha(DATA / name) for name in logs},
        "plan_sha256": sha(DATA / "v44-plan.json"),
        "runner_sha256": sha(DATA / "v44-run-experiments.py"),
        "freeze_script_sha256": sha(Path(__file__)),
    }
    identity_path.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert files == {name: sha(ROOT / name) for name in names}, "冻结期间来源发生改变"
    assert sha(executable) == sha(ROOT / "build/release/solver.exe")
    print(json.dumps({"source_files": len(files), "solver_sha256": identity["solver_sha256"]}), flush=True)


if __name__ == "__main__":
    freeze()
