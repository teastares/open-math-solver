"""使用已冻结的 Accelerate CLI 运行大规模 LP，逐次保存报告和进程资源数据。"""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import benchmark

CANDIDATE = [
    "--backend", "ipm", "--ipm-method", "homogeneous",
    "--ipm-singleton-elimination", "on", "--ipm-kkt-equilibration", "on",
    "--ipm-objective-scaling", "off", "--ipm-linear-accuracy", "homogeneous_repair",
    "--ipm-linear-solver", "normal_cholmod_aat", "--ipm-fixed-variable-elimination", "on",
    "--ipm-presolve", "on", "--ipm-crossover", "on",
]
REFERENCE = ["--backend", "highs", "--highs-solver", "choose",
             "--highs-presolve", "choose", "--highs-crossover", "on"]

FLAGS = {"pdlp": ["--backend", "pdlp", "--pdlp-mode", "adaptive"],
         "ipm": CANDIDATE + ["--ipm-profile", "on"], "highs": REFERENCE}


def trial(solver, model, backend, budget, folder, iteration_limit, pdlp_mode):
    case = model.stem
    record = folder / f"{case}-{backend}-{budget}.json.gz"
    if record.exists():
        raise FileExistsError(record)
    command = [str(solver), "solve", str(model), "--threads", "1",
               "--seed", "0", "--time-limit", str(budget), *FLAGS[backend]]
    if backend == "pdlp":
        command[command.index("--pdlp-mode") + 1] = pdlp_mode
    if iteration_limit is not None:
        command.extend(["--iteration-limit", str(iteration_limit)])
    env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.monotonic()
        proc = subprocess.Popen(["/usr/bin/time", "-lp", *command], stdout=out, stderr=err,
                                env=env, start_new_session=True)
        try:
            proc.wait(timeout=budget + 45)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            timed_out = True
        wall = time.monotonic() - start
        out.seek(0)
        err.seek(0)
        stderr = err.read().decode(errors="replace")
        try:
            report = json.load(out)
        except (ValueError, UnicodeError):
            report = {"status": "wall_timeout" if timed_out else "process_failed"}
    tmp = record.with_suffix(".tmp")
    with gzip.open(tmp, "wt", compresslevel=3) as f:
        json.dump(dict(command=command, code=proc.returncode, wall_s=wall, stderr=stderr, report=report), f)
    tmp.replace(record)
    print(case, backend, budget, report.get("status"), round(wall, 2), flush=True)


def summarize(folder):
    rows = []
    for path in sorted(folder.glob("*.json.gz")):
        with gzip.open(path, "rt") as f:
            raw = json.load(f)
        j = raw["report"]
        cmd = raw["command"]
        budget = int(cmd[cmd.index("--time-limit")+1])
        timings = j.get("timings") or {}
        primal = j.get("verification") or {}
        dual = j.get("lp_optimality") or {}
        model = j.get("model") or {}
        mem = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size", raw["stderr"])
        verified = (j.get("status") == "optimal" and benchmark.classify(raw["code"], j) == "resolved"
                    and benchmark.valid_lp_optimality(j) and model.get("is_mip") is False
                    and timings["solve_seconds"] + timings["verification_seconds"] <= budget
                    and raw["wall_s"] <= budget + 45)
        row = dict(case=Path(cmd[2]).stem, backend=cmd[cmd.index("--backend")+1], budget=budget,
                   status=j["status"], verified=int(verified), code=raw["code"],
                   wall_s=raw["wall_s"], rss_MiB=int(mem[1])/2**20 if mem else None,
                   rows=model.get("rows"), cols=model.get("columns"), nnz=model.get("nonzeros"),
                   read_s=timings.get("read_seconds"), solve_s=timings.get("solve_seconds"),
                   check_s=timings.get("verification_seconds"), ipm_iter=j.get("ipm_iterations"),
                   factorizations=j.get("ipm_factorizations"), simplex_iter=j.get("simplex_iterations"),
                   pdlp_iter=j.get("pdlp_iterations"), passes=j.get("pdlp_operator_passes"),
                   has_solution=j.get("has_solution"), valid_primal=primal.get("valid_input"),
                   feasible=primal.get("feasible"), dual_feasible=dual.get("dual_feasible"),
                   row_err=primal.get("max_row_violation") if primal.get("valid_input") else None,
                   bound_err=primal.get("max_bound_violation") if primal.get("valid_input") else None,
                   dual_err=dual.get("max_dual_residual"), rel_gap=dual.get("relative_objective_gap"),
                   gap_pass=dual.get("gap_satisfied"), objective=j.get("objective"),
                   message=j.get("message"))
        rows.append(row)
    if rows:
        with (folder/"results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("backend", choices=FLAGS)
    p.add_argument("budgets", nargs="+", type=int)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--iteration-limit", type=int)
    p.add_argument("--pdlp-mode", choices=["adaptive", "halpern"], default="adaptive")
    args = p.parse_args()
    if sys.platform != "darwin":
        p.error("本脚本使用 macOS /usr/bin/time -lp")
    if args.jobs < 1 or any(budget < 1 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        p.error("并发数与预算须为正整数，且预算不能重复")
    if args.backend != "pdlp" and (args.iteration_limit is not None or args.pdlp_mode != "adaptive"):
        p.error("迭代上限覆盖与一阶模式选项仅用于 PDLP")
    solver = args.solver.resolve(strict=True)
    if not solver.is_file() or not os.access(solver, os.X_OK):
        p.error("求解器必须是可执行文件")
    lock = json.loads((ROOT / "benchmarks/modern-lp-smoke.lock.json").read_text())
    locked = {Path(entry["name"]).stem: entry for entry in lock["models"]}
    cases = args.cases if args.cases else sorted(locked)
    if len(set(cases)) != len(cases) or set(cases) - locked.keys():
        p.error("算例名须来自锁文件且不能重复")
    models = [(args.models / locked[case]["name"]).resolve(strict=True) for case in cases]
    files = [solver, *models]
    hashes = {}
    for file in files:
        with file.open("rb") as stream:
            hashes[str(file)] = hashlib.file_digest(stream, "sha256").hexdigest()
    for case, model in zip(cases, models):
        if model.stat().st_size != locked[case]["bytes"] or hashes[str(model)] != locked[case]["sha256"]:
            p.error(f"算例大小或 SHA256 校验失败：{case}")
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(options=vars(args), sha256=hashes)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    for budget in args.budgets:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(trial, solver, model, args.backend, budget, folder,
                                   args.iteration_limit, args.pdlp_mode) for model in models]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        summarize(folder)


if __name__ == "__main__":
    main()
