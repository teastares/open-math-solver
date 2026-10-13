"""冻结第36轮并串行运行有界Krylov回收消融，所有质量失败均完整归档。"""
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
EXE = ROOT / ".tools/solver-ipm-v39.exe"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    assert not EXE.exists(), "不能覆盖已有冻结程序"
    names = list(read(DATA / "ipm-v38-identity.json")["files"])
    names.append("include/solver/detail/ipm_recycling.hpp")
    shutil.copy2(ROOT / "build/release/solver.exe", EXE)
    identity = {"solver_sha256": sha(EXE), "files": {name: sha(ROOT / name) for name in sorted(names)}}
    with zipfile.ZipFile(DATA / "ipm-v39-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in identity["files"]:
            archive.write(ROOT / name, name)
    save(DATA / "ipm-v39-identity.json", identity)
    shutil.copy2(ROOT / ".tools/iteration-ipm-v39-build.log", DATA / "iteration-ipm-v39-build.log")
    shutil.copy2(ROOT / "build/release/Testing/Temporary/LastTest.log", DATA / "iteration-ipm-v39-ctest-detail.log")
    print(json.dumps({"source_files": len(names), "sha256": identity["solver_sha256"]}), flush=True)


def trajectories():
    for version in (38, 39):
        assert sha(ROOT / f".tools/solver-ipm-v{version}.exe") == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    models = compare_lp.load_models(ROOT / "benchmarks/development.txt")[0]
    records = []
    for linear, accuracy, retries, presolve, fixed in (
            ("qdldl", "strict", 0, "off", "off"),
            ("normal_augmented", "strict", 0, "off", "off"),
            ("normal_augmented", "blockwise", 8, "on", "on"),
            ("normal_augmented", "blockwise_weighted", 8, "off", "off")):
        for model in models:
            reports, commands, returncodes = [], [], []
            for version in (38, 39):
                command = [str(ROOT / f".tools/solver-ipm-v{version}.exe"), "solve", model["input"],
                    "--backend", "ipm", "--time-limit", "5", "--threads", "1", "--seed", "0",
                    "--ipm-linear-solver", linear, "--ipm-linear-accuracy", accuracy,
                    "--ipm-direction-retries", str(retries), "--ipm-kkt-equilibration", "on",
                    "--ipm-singleton-elimination", "off" if linear == "qdldl" else "on",
                    "--ipm-fixed-variable-elimination", "off" if linear == "qdldl" else "on",
                    "--ipm-singleton-presolve", presolve, "--ipm-presolve-fixed-elimination", fixed]
                if version == 39:
                    command += ["--ipm-krylov-history-size", "0"]
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=35)
                reports.append(json.loads(process.stdout))
                commands.append(command)
                returncodes.append(process.returncode)
            assert reports[1]["options"]["ipm"]["krylov_history_size"] == 0
            for suffix in ("projections", "acceptances", "vectors", "peak_dimension"):
                assert type(reports[1]["ipm_krylov_recycle_" + suffix]) is int
                assert reports[1]["ipm_krylov_recycle_" + suffix] == 0
            normalized = copy.deepcopy(reports)
            for report in normalized:
                for key in ("timings", "backend_version"):
                    report.pop(key, None)
                for suffix in ("projections", "acceptances", "vectors", "peak_dimension"):
                    field = "ipm_krylov_recycle_" + suffix
                    if field in report:
                        assert report.pop(field) == 0
                report["options"]["ipm"].pop("krylov_history_size", None)
            equal = normalized[0] == normalized[1] and returncodes[0] == returncodes[1]
            records.append({"model": model["input"], "model_sha256": model["sha256"], "commands": commands,
                            "returncodes": returncodes, "reports": reports, "non_time_equal": equal})
            assert equal, (linear, accuracy, model["input"])
    for version in (38, 39):
        assert sha(ROOT / f".tools/solver-ipm-v{version}.exe") == read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"]
    (DATA / "v39-history-zero-trajectory-runs.json.gz").write_bytes(
        gzip.compress(json.dumps(records, ensure_ascii=False).encode("utf-8"), mtime=0))
    print(json.dumps({"old_trajectories_equal": len(records)}), flush=True)


def benchmark(stage, linear, accuracy, presolve, reference):
    assert sha(EXE) == read(DATA / "ipm-v39-identity.json")["solver_sha256"]
    development = stage == "development"
    tag = f"{stage}-{linear}-{accuracy}-presolve_{presolve}-{reference}"
    output = ROOT / f"results/round-ipm-v39-{tag}"
    assert not output.exists(), "不能覆盖已测结果"
    command = [sys.executable, str(ROOT / "scripts/compare_lp.py"), "--solver", str(EXE),
        "--models-file", str(ROOT / ("benchmarks/development.txt" if development else "benchmarks/netlib-all.txt")),
        "--repeats", "3" if development else "1", "--time-limit", "5", "--wall-timeout", "35",
        "--threads", "1", "--seed", "0", "--candidate-backend", "ipm", "--reference-backend", reference,
        "--output-dir", str(output), "--ipm-krylov-history-size", "4"]
    settings = {"linear-solver": linear, "linear-accuracy": accuracy, "singleton-elimination": "on",
        "kkt-equilibration": "on", "fixed-variable-elimination": "on", "singleton-presolve": presolve,
        "presolve-fixed-elimination": presolve, "direction-retries": "0" if accuracy == "strict" else "8"}
    for key, value in settings.items():
        command += ["--ipm-" + key, value]
        if reference == "ipm":
            command += ["--reference-ipm-" + key, value]
    if reference == "ipm":
        command += ["--reference-ipm-krylov-history-size", "0"]
    save(DATA / f"v39-{tag}-command.json", command)
    with (DATA / f"v39-{tag}-run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert process.returncode in (0, 1), (tag, process.returncode)
    for name in ("manifest.json", "summary.json"):
        shutil.copy2(output / name, DATA / f"v39-{tag}-{name}")
    (DATA / f"v39-{tag}-runs.jsonl.gz").write_bytes(gzip.compress((output / "runs.jsonl").read_bytes(), mtime=0))
    print(json.dumps({"tag": tag, "variants": read(output / "summary.json")["variants"]}), flush=True)


def details():
    groups = {"development": {}, "coverage": {}}
    for manifest_path in sorted(DATA.glob("v39-*-manifest.json")):
        tag = manifest_path.name[4:-14]
        manifest = read(manifest_path)
        rows = [json.loads(row) for row in gzip.decompress((DATA / f"v39-{tag}-runs.jsonl.gz").read_bytes()).splitlines()]
        reference, candidate = manifest["reference_variant"], manifest["candidate_variant"]
        inputs = {row["input"] for row in rows}
        passed = {side: {Path(path).stem for path in inputs
                  if sum(row["quality_pass"] for row in rows
                         if row["requested_variant"] == side and row["input"] == path) == manifest["repeats"]}
                  for side in (reference, candidate)}
        entries = []
        for row in rows:
            report = row["report"]
            entries.append({"model": Path(row["input"]).stem, "candidate": row["requested_variant"] == candidate,
                "repetition": row.get("repetition"), "quality_pass": row["quality_pass"], "status": row["status"],
                "quality_failure": row["quality_failure"], "numerical_seconds": row["numerical_seconds"],
                "message": report.get("message"), "work": {key: value for key, value in report.items() if key.startswith("ipm_")},
                "optimality": report.get("lp_optimality")})
        groups["development" if tag.startswith("development-") else "coverage"][tag] = {
            "summary": read(DATA / f"v39-{tag}-summary.json")["variants"], "gain": sorted(passed[candidate] - passed[reference]),
            "lost": sorted(passed[reference] - passed[candidate]), "models": entries}
    for name, value in groups.items():
        save(DATA / f"v39-{name}-detail.json", value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "trajectories", "development", "netlib", "details"))
    parser.add_argument("--linear", default="normal_augmented")
    parser.add_argument("--accuracy", default="blockwise")
    parser.add_argument("--presolve", choices=("off", "on"), default="off")
    parser.add_argument("--reference", choices=("ipm", "highs"), default="ipm")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "trajectories":
        trajectories()
    elif args.stage == "details":
        details()
    else:
        benchmark(args.stage, args.linear, args.accuracy, args.presolve, args.reference)
