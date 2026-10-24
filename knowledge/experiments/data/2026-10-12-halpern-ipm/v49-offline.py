"""仅对已保存矩阵与齐次状态串行复算，不启动LP求解或修改捕获数据。"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def execute(command, timeout):
    start = time.perf_counter()
    environment = os.environ.copy()
    environment.pop("SOLVER_V49_DIAGNOSTICS", None)
    environment.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout, env=environment)
        code, out, err, failure = result.returncode, result.stdout, result.stderr, None
    except subprocess.TimeoutExpired as result:
        code, out, err, failure = None, result.stdout or b"", result.stderr or b"", "wall_timeout"
    return {"command": command, "returncode": code, "error": failure, "stdout": out.decode("utf-8", errors="replace"),
        "stderr": err.decode("utf-8", errors="replace"), "wall_seconds": time.perf_counter() - start, "wall_limit": timeout}


def run():
    raw_path = DATA / "v49-replay-runs.jsonl.gz"
    manifest_path = DATA / "v49-offline-manifest.json"
    analysis_path = DATA / "v49-hsd-analysis.json"
    log_path = DATA / "v49-hsd-analysis-process.json"
    assert not any(path.exists() for path in (raw_path, manifest_path, analysis_path, log_path))
    summary = read(DATA / "v49-summary.json")
    run_manifest = read(DATA / "v49-run-manifest.json")
    work = Path(run_manifest["work"])
    build_path = Path(run_manifest["build_manifest"])
    build = read(build_path)
    exe = Path(build["replay_executable"])
    assert sha(exe) == build["replay_executable_sha256"]
    assert all(sha(Path(path)) == expected for path, expected in run_manifest["protected"].items())
    protected = {str(Path(__file__)): sha(Path(__file__)), str(exe): sha(exe), str(build_path): sha(build_path),
        str(DATA / "v49-analyze-hsd.py"): sha(DATA / "v49-analyze-hsd.py"),
        str(DATA / "v49-summary.json"): sha(DATA / "v49-summary.json")}
    hsd_inputs = [work / "hsd-self-on.hsd.json"]
    replays = []
    for case in summary["cases"]:
        prefix = Path(case["capture_prefix"])
        for path, identity in case["capture_files"].items():
            assert sha(Path(path)) == identity["sha256"]
            protected[path] = identity["sha256"]
        hsd = Path(str(prefix) + ".hsd.json")
        if hsd.exists():
            hsd_inputs.append(hsd)
        matrix = Path(str(prefix) + ".normal.bin")
        if matrix.exists():
            replays.append((case, prefix, matrix))
    for path in work.glob("hsd-self-on.hsd.*"):
        protected[str(path)] = sha(path)
    manifest = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "protected": protected, "replay_count": len(replays), "hsd_metadata_count": len(hsd_inputs),
        "wall_limit_per_replay": 120, "scope": "同一捕获数据的离线算术核验；全部额外工作不构成LP性能成绩。"}
    save_new(manifest_path, manifest)
    invalid_replays = []
    with gzip.open(raw_path, "xt", encoding="utf-8") as target:
        for case, prefix, matrix in replays:
            before = sha(matrix)
            assert before == protected[str(matrix)]
            row = execute([str(exe), str(prefix)], 120)
            row.update(case_index=case["case_index"], model=case["model"], linear_solver=case["linear_solver"],
                prefix=str(prefix), matrix_sha256_before=before)
            try:
                row["report"] = json.loads(row["stdout"])
            except (TypeError, ValueError):
                row["report"] = None
            row["matrix_sha256_after"] = sha(matrix)
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            target.flush()
            if row["error"] != "wall_timeout" and (row["returncode"] != 0 or not isinstance(row["report"], dict)):
                invalid_replays.append(case["case_index"])
            assert row["matrix_sha256_before"] == row["matrix_sha256_after"] == protected[str(matrix)]
            print(json.dumps({"model": case["model"], "linear_solver": case["linear_solver"], "returncode": row["returncode"], "error": row["error"]}), flush=True)
    command = [sys.executable, str(DATA / "v49-analyze-hsd.py"), *map(str, hsd_inputs), "--output", str(analysis_path)]
    result = execute(command, 120)
    save_new(log_path, result)
    for path, expected in protected.items():
        assert sha(Path(path)) == expected, "离线分析期间来源改变：" + path
    assert result["returncode"] == 0 and read(analysis_path)["passed"], "HSD离线分析失败，日志与结果已保留"
    assert not invalid_replays, "矩阵回放进程或报告无效，原始日志已保留：" + str(invalid_replays)
    print(json.dumps({"replays": len(replays), "hsd_metadata": len(hsd_inputs), "hsd_analysis_passed": True}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
