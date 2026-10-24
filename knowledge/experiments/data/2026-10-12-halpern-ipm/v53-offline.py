"""在独立120秒墙钟预算内分析本轮四份快照，保留全部输出和失败。"""
import datetime
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


def main():
    run = json.loads((DATA / "v53-run-manifest.json").read_text(encoding="utf-8"))
    work = Path(run["work"])
    build = json.loads((work / "build-manifest.json").read_text(encoding="utf-8"))
    analyzer = DATA / "v53-analyze-hsd.py"
    assert sha(analyzer) == build["protected_before_and_after"][str(analyzer)]
    inputs = [work / "captures" / f"{index}-diagnostic_on-hsd.json" for index in range(4)]
    protected = {str(path): sha(path) for path in [Path(__file__), analyzer, DATA / "v53-plan.json",
        DATA / "v53-run-manifest.json", work / "build-manifest.json", *inputs, *[path.with_suffix(".bin") for path in inputs]]}
    output, destination = DATA / "v53-hsd-analysis.json", DATA / "v53-hsd-analysis-process.json"
    assert not output.exists() and not destination.exists()
    command = [sys.executable, str(analyzer), *map(str, inputs), "--output", str(output)]
    record = {"started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command,
        "cwd": str(ROOT), "wall_timeout_seconds": 120, "protected_before": protected}
    start = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), capture_output=True, timeout=120)
        record.update(returncode=process.returncode, stdout=process.stdout.decode("utf-8", errors="replace"),
            stderr=process.stderr.decode("utf-8", errors="replace"), wall_timeout=False)
    except subprocess.TimeoutExpired as failure:
        record.update(returncode=None, stdout=(failure.stdout or b"").decode("utf-8", errors="replace"),
            stderr=(failure.stderr or b"").decode("utf-8", errors="replace"), wall_timeout=True)
    record["elapsed_wall_seconds"] = time.perf_counter() - start
    record["protected_after"] = {path: sha(Path(path)) for path in protected}
    record["output"] = {"path": str(output), "sha256": sha(output) if output.exists() else None}
    with destination.open("x", encoding="utf-8") as target:
        json.dump(record, target, ensure_ascii=False, indent=2)
        target.write("\n")
    assert record["protected_before"] == record["protected_after"], "离线输入变化"
    assert record["returncode"] == 0 and not record["wall_timeout"], "离线分析失败，原日志保留"
    print(json.dumps({"returncode": record["returncode"], "wall_seconds": record["elapsed_wall_seconds"], "output_sha256": record["output"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
