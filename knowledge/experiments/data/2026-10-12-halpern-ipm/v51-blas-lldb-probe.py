"""至多一次 LP 调试启动：捕获大 K 内核入口及两个临时缓冲区的重叠写入。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    prefix = DATA / "v51-blas-lldb-probe"
    manifest_path = Path(str(prefix) + "-manifest.json")
    result_path = Path(str(prefix) + "-result.json")
    assert not manifest_path.exists() and not result_path.exists()
    identity = json.loads((DATA / "ipm-v48-identity.json").read_text(encoding="utf-8"))
    old = json.loads((DATA / "v50-crash-repro-manifest.json").read_text(encoding="utf-8"))
    target = old["commands"]["48"]
    debugger = str(Path(identity["compiler"]).parent / "lldb.exe")
    stdout = Path(str(prefix) + "-target.stdout")
    stderr = Path(str(prefix) + "-target.stderr")
    commands = [
        'breakpoint set --name dgemm_kernel_COOPERLAKE --skip-prologue false --condition "$r8 > 341 && $rdx >= 6"',
        f'process launch --stdout "{stdout.as_posix()}" --stderr "{stderr.as_posix()}"',
        'register read rcx rdx r8 rsp rip',
        'thread backtrace',
        'expression -- unsigned long long $entry_rsp = (unsigned long long)$rsp',
        'expression -- unsigned long long $scratch_rsp = ($entry_rsp - 256 - 16512) & ~4095ULL',
        'expression -- unsigned long long $overlap = $scratch_rsp + 8320',
        'expression -- $scratch_rsp + 17536',
        'memory read -fx -s8 -c4 $entry_rsp',
        'watchpoint set expression -w write -s 8 -- $overlap',
        'breakpoint disable 1',
        'continue',
        'register read r12 rsi rsp rip rbx',
        'disassemble --start-address "$rip-32" --count 12',
        'memory read -fx -s8 -c4 $overlap',
        'continue',
        'register read r12 rsi rsp rip rbx',
        'disassemble --start-address "$rip-32" --count 12',
        'memory read -fx -s8 -c4 $overlap',
        'process kill',
    ]
    command = [debugger, "--no-lldbinit", "--batch"]
    for item in commands:
        command.extend(["-o", item])
    command.extend(["-k", "thread backtrace all", "-k", "register read", "--", *target])
    manifest = {"diagnostic_only": True, "maximum_attempts": 1, "timeout_seconds": 60,
                "command": command, "commands": commands, "script_sha256": sha(__file__),
                "debugger_sha256": sha(debugger), "solver_sha256": sha(target[0]), "input_sha256": sha(target[2])}
    assert manifest["solver_sha256"] == identity["solver_sha256"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env.pop("SOLVER_V49_DIAGNOSTICS", None)
    log = Path(str(prefix) + ".log")
    error = None
    with log.open("xb") as stream:
        try:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            code = run.returncode
        except subprocess.TimeoutExpired:
            code, error = None, "wall_timeout"
    result_path.write_text(json.dumps({"diagnostic_only": True, "returncode": code, "error": error,
        "manifest_sha256": sha(manifest_path), "files": {str(p): sha(p) for p in (log, stdout, stderr) if p.exists()}},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert sha(__file__) == manifest["script_sha256"] and sha(target[0]) == manifest["solver_sha256"]
    print(json.dumps({"returncode": code, "error": error}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
