"""冻结第35轮程序，并按既有预算运行显式消融；各阶段由调用者顺序执行。"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
EXE = ROOT / ".tools/solver-ipm-v38.exe"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    assert not EXE.exists(), "不覆盖已有冻结程序"
    previous = read(DATA / "ipm-v37-identity.json")
    shutil.copy2(ROOT / "build/release/solver.exe", EXE)
    identity = {"solver_sha256": sha(EXE), "files": {name: sha(ROOT / name) for name in previous["files"]}}
    with zipfile.ZipFile(DATA / "ipm-v38-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in identity["files"]:
            archive.write(ROOT / name, name)
    save(DATA / "ipm-v38-identity.json", identity)
    shutil.copy2(ROOT / ".tools/iteration-ipm-v38-build.log", DATA / "iteration-ipm-v38-build.log")
    shutil.copy2(ROOT / "build/release/Testing/Temporary/LastTest.log", DATA / "iteration-ipm-v38-ctest-detail.log")
    print(json.dumps({"frozen": len(identity["files"]), "sha256": identity["solver_sha256"]}), flush=True)


def trajectories():
    for version in (37, 38):
        assert sha(ROOT / f".tools/solver-ipm-v{version}.exe") == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    models = compare_lp.load_models(ROOT / "benchmarks/development.txt")[0]
    records = []
    # 既覆盖输入归并关闭，又覆盖开启但新消列关闭的两种旧路径。
    for linear, accuracy, retries, presolve in (("qdldl", "strict", 0, "off"),
            ("normal_augmented", "strict", 0, "on"),
            ("normal_augmented", "curvature_regularization", 8, "on")):
        for model in models:
            reports = []
            commands = []
            returncodes = []
            for version in (37, 38):
                command = [str(ROOT / f".tools/solver-ipm-v{version}.exe"), "solve", model["input"],
                    "--backend", "ipm", "--time-limit", "5", "--threads", "1", "--seed", "0",
                    "--ipm-linear-solver", linear, "--ipm-linear-accuracy", accuracy,
                    "--ipm-direction-retries", str(retries), "--ipm-kkt-equilibration", "on",
                    "--ipm-singleton-elimination", "off" if linear == "qdldl" else "on",
                    "--ipm-fixed-variable-elimination", "off" if linear == "qdldl" else "on",
                    "--ipm-singleton-presolve", presolve]
                if version == 38:
                    command += ["--ipm-presolve-fixed-elimination", "off"]
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=35)
                reports.append(json.loads(process.stdout))
                commands.append(command)
                returncodes.append(process.returncode)
            assert reports[1]["options"]["ipm"]["eliminate_presolved_fixed_variables"] is False
            assert reports[1]["ipm_presolved_fixed_columns"] == 0
            normalized = copy.deepcopy(reports)
            for report in normalized:
                for key in ("timings", "backend_version", "ipm_presolved_fixed_columns"):
                    report.pop(key, None)
                report["options"]["ipm"].pop("eliminate_presolved_fixed_variables", None)
            equal = normalized[0] == normalized[1] and returncodes[0] == returncodes[1]
            records.append({"model": model["input"], "model_sha256": model["sha256"],
                            "commands": commands, "returncodes": returncodes,
                            "reports": reports, "non_time_equal": equal})
            assert equal, (linear, accuracy, model["input"])
    (DATA / "v38-presolve-fixed-off-trajectory-runs.json.gz").write_bytes(
        gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
    print(json.dumps({"old_trajectories_equal": len(records)}), flush=True)


def benchmark(stage, linear, accuracy, reference):
    assert sha(EXE) == read(DATA / "ipm-v38-identity.json")["solver_sha256"]
    development = stage == "development"
    tag = f"{stage}-{linear}-{accuracy}-{reference}"
    output = ROOT / f"results/round-ipm-v38-{tag}"
    assert not output.exists(), "不覆盖已有实验"
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
        "--models-file", str(ROOT / ("benchmarks/development.txt" if development else "benchmarks/netlib-all.txt")),
        "--repeats", "3" if development else "1", "--time-limit", "5", "--wall-timeout", "35",
        "--threads", "1", "--seed", "0", "--candidate-backend", "ipm",
        "--reference-backend", "highs" if reference == "highs" else "ipm", "--output-dir", str(output)]
    settings = {"linear-solver": linear, "linear-accuracy": accuracy,
        "singleton-elimination": "on", "kkt-equilibration": "on", "fixed-variable-elimination": "on",
        "direction-retries": "0" if accuracy == "strict" else "8", "singleton-presolve": "on"}
    for key, value in settings.items():
        command += ["--ipm-" + key, value]
        if reference != "highs":
            command += ["--reference-ipm-" + key,
                        "off" if reference == "original" and key == "singleton-presolve" else value]
    command += ["--ipm-presolve-fixed-elimination", "on"]
    if reference != "highs":
        command += ["--reference-ipm-presolve-fixed-elimination", "off"]
    save(DATA / f"v38-{tag}-command.json", command)
    with (DATA / f"v38-{tag}-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    # 比较工具以 1 表示完整运行中存在质量失败，必须归档这些失败；2 才是调用或记录错误。
    assert process.returncode in (0, 1), (tag, process.returncode)
    for filename in ("manifest.json", "summary.json"):
        shutil.copy2(output / filename, DATA / f"v38-{tag}-{filename}")
    (DATA / f"v38-{tag}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    summary = read(output / "summary.json")
    print(json.dumps({"tag": tag, "variants": summary["variants"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "trajectories", "development", "netlib"))
    parser.add_argument("--linear", default="normal_augmented")
    parser.add_argument("--accuracy", default="strict")
    parser.add_argument("--reference", choices=("merged", "original", "highs"), default="merged")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "trajectories":
        trajectories()
    else:
        benchmark(args.stage, args.linear, args.accuracy, args.reference)
