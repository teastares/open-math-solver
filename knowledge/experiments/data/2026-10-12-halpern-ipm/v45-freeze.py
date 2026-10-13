"""冻结第42轮实验身份：复用v44生产程序和来源，不构建、不复制成新版本、不运行求解。"""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
EXPECTED_SOLVER = "58291db80a9dc7faaa17d037060d6cc53fe26db06de3bf3e67107e6401a7c24c"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    destination = DATA / "ipm-v45-identity.json"
    assert not destination.exists(), "不能覆盖已经冻结的实验身份"
    plan_path = DATA / "v45-plan.json"
    plan = read(plan_path)
    production_path = DATA / "ipm-v44-identity.json"
    production = read(production_path)
    executable = ROOT / ".tools/solver-ipm-v44.exe"
    assert plan["production_identity"] == production_path.name
    assert plan["production_identity_sha256"] == sha(production_path)
    assert plan["solver"] == str(executable)
    assert production["solver_sha256"] == plan["solver_sha256"] == EXPECTED_SOLVER
    assert sha(executable) == sha(ROOT / "build/release/solver.exe") == EXPECTED_SOLVER
    assert plan["runner_sha256"] == sha(DATA / "v45-run-experiments.py")
    assert plan["new_production_build"] is False
    files = production["files"]
    assert len(files) == 90 and files == {name: sha(ROOT / name) for name in files}
    archive_path = DATA / "ipm-v44-source.zip"
    assert sha(archive_path) == production["source_zip_sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 90
        assert {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names} == files
    assert sha(Path(production["compiler"])) == production["compiler_sha256"]
    assert {name: sha(ROOT / name) for name in production["libraries"]} == production["libraries"]
    assert {name: sha(DATA / name) for name in production["logs"]} == production["logs"]
    prior = read(DATA / plan["prior_verification"])
    assert prior["passed"] is True and prior["final"] is True
    assert prior["solver_sha256"] == EXPECTED_SOLVER
    for field in ("prior_verification", "prior_trajectories"):
        assert sha(DATA / plan[field]) == plan[field + "_sha256"]
    assert all(sha(DATA / name) == digest for name, digest in plan["historical_boundaries"]["evidence"].items())
    identity = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment": "v45", "production_version": "ipm-cpu-0.39.0", "new_production_build": False,
        "solver": str(executable), "solver_sha256": EXPECTED_SOLVER,
        "production_identity": production_path.name, "production_identity_sha256": sha(production_path),
        "source_archive": archive_path.name, "source_zip_sha256": production["source_zip_sha256"],
        "files": files, "compiler": production["compiler"], "compiler_sha256": production["compiler_sha256"],
        "libraries": production["libraries"], "logs": production["logs"],
        "prior_verification": plan["prior_verification"], "prior_verification_sha256": plan["prior_verification_sha256"],
        "prior_trajectories": plan["prior_trajectories"], "prior_trajectories_sha256": plan["prior_trajectories_sha256"],
        "plan_sha256": sha(plan_path), "runner_sha256": sha(DATA / "v45-run-experiments.py"),
        "freeze_script_sha256": sha(Path(__file__)),
        "scope": "仅本轮实验身份冻结；90来源、17项正式测试日志与40对旧轨迹引用v44已有证据，不是v45重新执行的构建或回归。",
    }
    destination.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert files == {name: sha(ROOT / name) for name in files}
    assert sha(executable) == EXPECTED_SOLVER
    print(json.dumps({"experiment": "v45", "production": "v44", "source_files": len(files),
                      "solver_sha256": EXPECTED_SOLVER, "plan_sha256": identity["plan_sha256"]}), flush=True)


if __name__ == "__main__":
    freeze()
