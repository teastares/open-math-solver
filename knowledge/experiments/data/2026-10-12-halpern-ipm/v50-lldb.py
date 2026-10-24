"""最多十次调试旧冻结程序，首次捕获访问违规即停止；不计入性能结果。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    manifest_path = DATA / "v50-lldb-manifest.json"
    result_path = DATA / "v50-lldb-result.json"
    assert not manifest_path.exists() and not result_path.exists()
    identity = read(DATA / "ipm-v48-identity.json")
    previous = read(DATA / "v50-crash-repro-manifest.json")
    target = previous["commands"]["48"]
    debugger = Path(identity["compiler"]).parent / "lldb.exe"
    manifest = {"debugger": str(debugger), "debugger_sha256": sha(debugger), "target": target,
        "solver_sha256": identity["solver_sha256"], "input_sha256": previous["input_sha256"],
        "maximum_attempts": 10, "wall_timeout": 60, "stop_on": "调试器日志包含访问违规0xc0000005",
        "script_sha256": sha(__file__), "diagnostic_only": True}
    assert sha(target[0]) == identity["solver_sha256"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    env.pop("SOLVER_V49_DIAGNOSTICS", None)
    records = []
    for index in range(2, 12):
        log = DATA / f"v50-lldb-attempt{index}.log"
        stdout = DATA / f"v50-lldb-attempt{index}-target.stdout"
        stderr = DATA / f"v50-lldb-attempt{index}-target.stderr"
        command_path = DATA / f"v50-lldb-attempt{index}-command.json"
        assert not any(path.exists() for path in (log, stdout, stderr, command_path))
        launch = f'process launch --stdout "{stdout.as_posix()}" --stderr "{stderr.as_posix()}"'
        command = [str(debugger), "--no-lldbinit", "--batch", "-o", launch,
                   "-o", "thread backtrace all", "-o", "register read", "--", *target]
        command_path.write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        error = None
        with log.open("xb") as destination:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=destination, stderr=subprocess.STDOUT, timeout=60)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code, error = None, "wall_timeout"
        content = log.read_text(encoding="utf-8", errors="replace")
        caught = "0xc0000005" in content.lower()
        records.append({"attempt": index, "returncode": code, "error": error, "access_violation_caught": caught,
            "files": {str(path): sha(path) for path in (log, stdout, stderr, command_path) if path.exists()}})
        assert sha(target[0]) == identity["solver_sha256"] and sha(target[2]) == previous["input_sha256"]
        print(json.dumps({"attempt": index, "returncode": code, "access_violation_caught": caught}), flush=True)
        if caught or error:
            break
    result_path.write_text(json.dumps({"diagnostic_only": True, "records": records,
        "manifest_sha256": sha(manifest_path)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert sha(__file__) == manifest["script_sha256"]


if __name__ == "__main__":
    main()
