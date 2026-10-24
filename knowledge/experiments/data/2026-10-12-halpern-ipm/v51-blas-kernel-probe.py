"""固定六个边界输入，逐个独立进程调用已安装的旧 OpenBLAS 内核。"""
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
    manifest_path = DATA / "v51-blas-kernel-probe-manifest.json"
    result_path = DATA / "v51-blas-kernel-probe-result.json"
    assert not manifest_path.exists() and not result_path.exists()
    identity = json.loads((DATA / "ipm-v48-identity.json").read_text(encoding="utf-8"))
    compiler = identity["compiler"]
    source = DATA / "v51-blas-kernel-probe.cpp"
    library = ROOT / "build/openblas/install/lib/libopenblas.a"
    executable = ROOT / ".tools/v51-blas-kernel-probe.exe"
    command = [compiler, "-std=c++20", "-O2", "-g", str(source), str(library), "-o", str(executable)]
    cases = [128, 256, 340, 341, 342, 384]
    manifest = {"diagnostic_only": True, "script_sha256": sha(__file__), "source_sha256": sha(source),
                "library_sha256": sha(library), "compiler_sha256": sha(compiler), "build_command": command,
                "k_cases": cases, "m": 16, "n": 6, "timeout_seconds": 10,
                "expected_rule": "C[i,j] = K*(i+1)*(j+1)，输入与结果均可精确表示"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
    (DATA / "v51-blas-kernel-probe-build.log").write_bytes(build.stdout + build.stderr)
    assert build.returncode == 0, build.stderr.decode(errors="replace")
    records = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    for k in cases:
        cmd = [str(executable), str(k)]
        run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, timeout=10)
        stdout = DATA / f"v51-blas-kernel-probe-k{k}.stdout"
        stderr = DATA / f"v51-blas-kernel-probe-k{k}.stderr"
        stdout.write_bytes(run.stdout); stderr.write_bytes(run.stderr)
        record = {"k": k, "command": cmd, "returncode": run.returncode,
                  "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr),
                  "stdout": run.stdout.decode(errors="replace"), "stderr": run.stderr.decode(errors="replace")}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    result_path.write_text(json.dumps({"diagnostic_only": True, "manifest_sha256": sha(manifest_path),
        "executable_sha256": sha(executable), "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assert sha(library) == manifest["library_sha256"] and sha(source) == manifest["source_sha256"]
    assert sha(__file__) == manifest["script_sha256"]


if __name__ == "__main__":
    main()
