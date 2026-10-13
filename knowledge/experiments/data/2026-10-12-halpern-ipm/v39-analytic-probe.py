"""用预声明解析族检查回收分支覆盖；保留所有结果，不用作公开性能成绩。"""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v39-analytic-inputs"
EXE = ROOT / "build/release/solver.exe"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(EXE.read_bytes()).hexdigest()
    models = []
    for exponent in (8, 16, 24, 32):
        epsilon = 2.0 ** -exponent
        for cost in (1.0, 1e6):
            name = f"coupled-{exponent}-{int(cost)}"
            content = (f"Minimize\n obj: {cost:.17g} x + {3 * cost:.17g} y\nSubject To\n"
                       f" r1: x + y = 2\n r2: x + {1 + epsilon:.17g} y = {2 + epsilon:.17g}\n"
                       "Bounds\n 0 <= x <= 2\n 0 <= y <= 2\nEnd\n")
            models.append((name, content, [1, 1], 4 * cost))
    for coefficient in (1e-8, 1.0):
        for cost in (1e-4, 1.0, 1e6):
            name = f"scalar-{coefficient}-{cost}"
            content = (f"Minimize\n obj: {cost:.17g} x\nSubject To\n"
                       f" r1: {coefficient:.17g} x >= {3 * coefficient:.17g}\nBounds\n 0 <= x <= 5\nEnd\n")
            models.append((name, content, [3], 3 * cost))
    rows = []
    for name, content, primal, objective in models:
        path = WORK / (name + ".lp")
        path.write_text(content, encoding="utf-8")
        for linear in ("normal_augmented", "qdldl"):
            for accuracy in ("strict", "blockwise"):
                command = [str(EXE), "solve", str(path), "--backend", "ipm", "--threads", "1",
                    "--seed", "0", "--time-limit", "2", "--iteration-limit", "80",
                    "--ipm-linear-solver", linear, "--ipm-linear-accuracy", accuracy,
                    "--ipm-singleton-elimination", "on", "--ipm-kkt-equilibration", "on",
                    "--ipm-direction-retries", "8", "--ipm-krylov-history-size", "4"]
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=32)
                parse_error = None
                try:
                    report = json.loads(process.stdout)
                except json.JSONDecodeError as error:
                    report = None
                    parse_error = str(error)
                row = {"model": name, "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "input_content": content, "analytic_primal": primal, "analytic_objective": objective,
                    "solver_sha256": digest, "command": command, "returncode": process.returncode,
                    "stderr": process.stderr, "stdout": process.stdout, "parse_error": parse_error, "report": report}
                rows.append(row)
                if report and report.get("ipm_krylov_recycle_projections", 0):
                    print(json.dumps({"model": name, "linear": linear, "accuracy": accuracy,
                        "status": report["status"], "projections": report["ipm_krylov_recycle_projections"],
                        "acceptances": report["ipm_krylov_recycle_acceptances"]}), flush=True)
    assert hashlib.sha256(EXE.read_bytes()).hexdigest() == digest
    (DATA / "v39-analytic-probe-runs.json.gz").write_bytes(
        gzip.compress(json.dumps(rows, ensure_ascii=False).encode("utf-8"), mtime=0))
    print(json.dumps({"runs": len(rows), "missing_reports": sum(r["report"] is None for r in rows),
        "projection_runs": sum(bool((r["report"] or {}).get("ipm_krylov_recycle_projections")) for r in rows)}))


if __name__ == "__main__":
    main()
