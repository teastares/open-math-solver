"""记录一次实验调用的源码、退出状态和原始输出，拒绝覆盖已有记录。"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("需要指定 Python 脚本及其参数")
    target = args.record.resolve()
    script = Path(command[0]).resolve()
    if not target.is_relative_to(ROOT) or target.exists() or not target.parent.is_dir():
        parser.error("记录须位于工作区已有目录，且不得覆盖")
    if not script.is_relative_to(ROOT) or not script.is_file():
        parser.error("脚本须为工作区内现有文件")
    inputs = dict.fromkeys([Path(__file__).resolve(), script, *(p.resolve() for p in args.input)])
    snapshots = {}
    for path in inputs:
        raw = path.read_bytes()
        snapshots[str(path)] = {"sha256": digest(raw), "source": raw.decode("utf-8")}
    invocation = [sys.executable, str(script), *command[1:]]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    begin = time.perf_counter()
    process = subprocess.run(invocation, cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1"), capture_output=True)
    elapsed = time.perf_counter() - begin
    changed = [str(path) for path in inputs if not path.is_file() or digest(path.read_bytes()) != snapshots[str(path)]["sha256"]]
    record = {
        "started_at_utc": started,
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": invocation, "returncode": process.returncode, "elapsed_seconds": elapsed,
        "input_sources_before_execution": snapshots, "changed_inputs": changed,
        "stdout": process.stdout.decode("utf-8", errors="replace"),
        "stderr": process.stderr.decode("utf-8", errors="replace"),
    }
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"record": str(target), "returncode": process.returncode,
                      "elapsed_seconds": elapsed, "changed_inputs": changed}, ensure_ascii=False))
    print(record["stdout"][-3500:])
    if record["stderr"]:
        print(record["stderr"][-3500:])
    return process.returncode or (1 if changed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
