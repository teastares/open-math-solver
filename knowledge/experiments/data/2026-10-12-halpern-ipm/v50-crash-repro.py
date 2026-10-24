"""独立复现一次访问违规；保留原成绩，固定两冻结版本各10次，不作性能评分。"""
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
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def other_failure(row):
    return row["returncode"] != 3221225477 and (row["error"] is not None or
        not isinstance(row["report"], dict) or benchmark.classify(row["returncode"], row["report"]) not in
        ("resolved", "incomplete", "verification_failed"))


def review():
    raw = DATA / "v50-crash-repro-runs.jsonl.gz"
    rows = [json.loads(line) for line in gzip.decompress(raw.read_bytes()).splitlines()]
    original = read(DATA / "v50-crash-repro-manifest.json")
    assert sha(DATA / "v50-crash-repro-attempt1.py") == original["script_sha256"]
    assert len(rows) == 20
    result = {"diagnostic_only": True, "raw_sha256": sha(raw), "original_script": "v50-crash-repro-attempt1.py",
        "review_script_sha256": sha(__file__), "runs": len(rows),
        "correction": "首次摘要将有合法numerical_error报告的退出码4误计为other_process_failures；本摘要按报告协议修正，原脚本/摘要/运行全部保留。",
        "by_version": {str(v): {"reports": sum(isinstance(r["report"], dict) for r in rows if r["version"] == v),
            "access_violations": sum(r["returncode"] == 3221225477 for r in rows if r["version"] == v),
            "other_process_failures": sum(other_failure(r) for r in rows if r["version"] == v)} for v in (48, 50)}}
    with (DATA / "v50-crash-repro-reviewed.json").open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(result, ensure_ascii=False))


def main():
    raw = DATA / "v50-crash-repro-runs.jsonl.gz"
    manifest_path = DATA / "v50-crash-repro-manifest.json"
    summary_path = DATA / "v50-crash-repro-summary.json"
    assert not any(p.exists() for p in (raw, manifest_path, summary_path))
    rows = [json.loads(line) for line in gzip.decompress((DATA / "v50-probe-runs.jsonl.gz").read_bytes()).splitlines()]
    origin, = (row for row in rows if row["returncode"] == 3221225477)
    assert origin["report"] is None and origin["classification"] == "invalid_json"
    identities = {version: read(DATA / f"ipm-v{version}-identity.json")["solver_sha256"] for version in (48, 50)}
    commands = {}
    for version in identities:
        commands[version] = [str(ROOT / f".tools/solver-ipm-v{version}.exe"), *origin["command"][1:]]
        assert sha(commands[version][0]) == identities[version]
    assert sha(origin["input"]) == origin["frozen_model_sha256"]
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "repeats": 10,
        "solver_runs": 20, "diagnostic_only": True, "commands": commands, "solver_sha256": identities,
        "origin_raw_sha256": sha(DATA / "v50-probe-runs.jsonl.gz"), "origin_run_index": origin["run_index"],
        "input_sha256": origin["frozen_model_sha256"], "script_sha256": sha(__file__), "wall_timeout": 35,
        "scope": "每重复交替v48/v50先后；仅检查进程可重复性，不替换原失败或改变性能门槛。"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env.pop("SOLVER_V49_DIAGNOSTICS", None)
    results = []
    with gzip.open(raw, "xt", encoding="utf-8") as output:
        for repeat in range(10):
            for version in ((48, 50) if repeat % 2 == 0 else (50, 48)):
                command = commands[version]
                before = {"solver": sha(command[0]), "input": sha(origin["input"])}
                start = time.perf_counter()
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=35)
                    code, stdout, stderr, error = process.returncode, process.stdout, process.stderr, None
                except subprocess.TimeoutExpired as failure:
                    code, stdout, stderr, error = None, failure.stdout or b"", failure.stderr or b"", "wall_timeout"
                row = {"version": version, "repetition": repeat, "returncode": code, "error": error,
                       "stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace"),
                       "wall_seconds": time.perf_counter() - start, "before": before,
                       "after": {"solver": sha(command[0]), "input": sha(origin["input"])}}
                try:
                    row["report"] = json.loads(row["stdout"])
                except ValueError:
                    row["report"] = None
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                results.append(row)
                assert before == row["after"] == {"solver": identities[version], "input": manifest["input_sha256"]}
                print(json.dumps({"version": version, "repetition": repeat, "returncode": code,
                                  "status": (row["report"] or {}).get("status")}), flush=True)
    summary = {"diagnostic_only": True, "runs": len(results), "raw_sha256": sha(raw),
        "by_version": {str(version): {"reports": sum(isinstance(r["report"], dict) for r in results if r["version"] == version),
            "access_violations": sum(r["returncode"] == 3221225477 for r in results if r["version"] == version),
            "other_process_failures": sum(other_failure(r) for r in results if r["version"] == version)}
            for version in identities}, "limit": "有限次数复现不能证明没有间歇性错误；原始访问违规保留。"}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert sha(__file__) == manifest["script_sha256"]
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--summarize"]:
        review()
    else:
        assert len(sys.argv) == 1
        main()
