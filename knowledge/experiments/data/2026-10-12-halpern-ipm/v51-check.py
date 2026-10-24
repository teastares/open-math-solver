"""第48轮公共BLAS前后数值检查与固定20次稳定性复核；独立于性能评分。"""
import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v51_runner", DATA / "v51-run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
ROOT = runner.ROOT


def public_blas(stage):
    destination = DATA / f"v51-public-blas-{stage}.json"
    assert not destination.exists()
    source = ROOT / "tests/test_openblas.cpp"
    library = ROOT / (".tools/v51-parent-libopenblas.a" if stage == "before" else "build/openblas/install/lib/libopenblas.a")
    compiler = Path(runner.read(DATA / "ipm-v50-identity.json")["compiler"])
    executable = ROOT / f".tools/v51-public-blas-{stage}.exe"
    assert not executable.exists()
    command = [str(compiler), "-std=c++20", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-static",
        "-I" + str(ROOT / "build/openblas/install/include/openblas"), str(source), str(library), "-o", str(executable)]
    record = {"diagnostic_only": True, "plan_sha256": runner.PLAN_SHA,
        "script_sha256": runner.sha(__file__), "source_sha256": runner.sha(source),
        "library_sha256": runner.sha(library), "compiler_sha256": runner.sha(compiler), "build_command": command}
    build = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace")
    record["build"] = {"returncode": build.returncode, "stdout": build.stdout, "stderr": build.stderr}
    if build.returncode == 0:
        record["executable_sha256"] = runner.sha(executable)
        try:
            result = subprocess.run([str(executable)], cwd=ROOT, env=runner.environment(), capture_output=True,
                encoding="utf-8", errors="replace", timeout=120)
            record["run"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired as error:
            record["run"] = {"returncode": None, "error": "wall_timeout",
                "stdout": (error.stdout or b"").decode("utf-8", errors="replace"),
                "stderr": (error.stderr or b"").decode("utf-8", errors="replace")}
    record["source_sha256_after"] = runner.sha(source)
    record["library_sha256_after"] = runner.sha(library)
    runner.save(destination, record)
    assert build.returncode == 0, "公共测试构建失败，证据已保留"
    assert record["source_sha256"] == record["source_sha256_after"] and record["library_sha256"] == record["library_sha256_after"]
    print(json.dumps(record["run"], ensure_ascii=False), flush=True)


def stability():
    frozen = runner.plan()
    destination = DATA / "v51-stability.json"
    raw = DATA / "v51-stability-runs.jsonl.gz"
    assert not destination.exists() and not raw.exists()
    executable = runner.check_solver(51)
    model, = (item for item in frozen["sets"]["probe"]["models"] if Path(item["input"]).stem == "dfl001")
    command = [str(executable), "solve", model["input"], "--backend", "ipm", "--time-limit", "5",
        "--threads", "1", "--seed", "0", "--iteration-limit", "200", "--scaling-passes", "5",
        "--ipm-method", "homogeneous", "--ipm-linear-solver", "normal_cholmod", "--ipm-linear-accuracy", "strict",
        "--ipm-direction-retries", "0"]
    for key, value in frozen["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    records = []
    for index in range(frozen["diagnostic_runs"]):
        before = {"solver": runner.sha(executable), "input": runner.sha(model["input"])}
        assert before["input"] == model["sha256"]
        try:
            process = subprocess.run(command, cwd=ROOT, env=runner.environment(), capture_output=True, timeout=35)
            code, stdout, stderr, error = process.returncode, process.stdout, process.stderr, None
        except subprocess.TimeoutExpired as failure:
            code, stdout, stderr, error = None, failure.stdout or b"", failure.stderr or b"", "wall_timeout"
        stdout, stderr = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
        try:
            report = runner.benchmark.strict_json(stdout)
        except ValueError:
            report = None
        classification = runner.benchmark.classify(code, report) if error is None else error
        valid = isinstance(report, dict) and classification in ("resolved", "incomplete", "verification_failed")
        records.append({"run_index": index, "command": command, "returncode": code, "stdout": stdout,
            "stderr": stderr, "error": error, "report": report, "classification": classification,
            "valid_report": valid, "before": before,
            "after": {"solver": runner.sha(executable), "input": runner.sha(model["input"])}})
        raw.write_bytes(gzip.compress(b"\n".join(json.dumps(item, ensure_ascii=False).encode("utf-8") for item in records)+b"\n", mtime=0))
        print(json.dumps({"run": index + 1, "classification": classification, "returncode": code}), flush=True)
    result = {"diagnostic_only": True, "plan_sha256": runner.PLAN_SHA, "solver_sha256": runner.sha(executable),
        "script_sha256": runner.sha(__file__), "raw_sha256": runner.sha(raw), "run_count": len(records),
        "valid_report_count": sum(item["valid_report"] for item in records),
        "access_violations": sum(item["returncode"] == 3221225477 for item in records),
        "passed": len(records) == 20 and all(item["valid_report"] and item["before"] == item["after"] for item in records),
        "interpretation": "固定有限重复的进程稳定性检查；合法数值失败仍是LP失败，不证明永不崩溃或达到最优。"}
    runner.save(destination, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def validate():
    before = runner.read(DATA / "v51-public-blas-before.json")
    after = runner.read(DATA / "v51-public-blas-after.json")
    identity = runner.read(DATA / "ipm-v51-identity.json")
    build_log = DATA / "iteration-ipm-v51-build.log"
    assert "100% tests passed out of 18" in build_log.read_text(encoding="utf-8")
    assert before["source_sha256"] == after["source_sha256"] == runner.sha(ROOT / "tests/test_openblas.cpp")
    assert before["compiler_sha256"] == after["compiler_sha256"]
    assert before["build"]["returncode"] == after["build"]["returncode"] == 0
    assert before["run"]["returncode"] != 0 and after["run"]["returncode"] == 0
    assert before["library_sha256"] != after["library_sha256"]
    assert after["library_sha256"] == identity["libraries"]["build/openblas/install/lib/libopenblas.a"]
    result = {"passed": True, "plan_sha256": runner.PLAN_SHA, "solver_sha256": runner.sha(runner.check_solver(51)),
        "public_before_sha256": runner.sha(DATA / "v51-public-blas-before.json"),
        "public_after_sha256": runner.sha(DATA / "v51-public-blas-after.json"),
        "build_log_sha256": runner.sha(build_log), "ctest_passed": 18,
        "script_sha256": runner.sha(__file__), "interpretation": "同一公共BLAS测试修复前失败、修复后通过；LP成绩另评。"}
    runner.save(DATA / "v51-validation.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("before", "after", "stability", "validate"))
    stage = parser.parse_args().stage
    runner.plan()
    if stage in ("before", "after"):
        public_blas(stage)
    elif stage == "stability":
        stability()
    else:
        validate()
