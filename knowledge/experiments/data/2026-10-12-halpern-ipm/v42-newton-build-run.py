"""冻结 v41 副本上的单方向误差观测；诊断成本计入时限，不生成性能分数。"""
import argparse
import copy
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v42-newton"
BUILD = ROOT / "build/release"
EXE = WORK / "v42-newton.exe"
PRODUCTION = ROOT / ".tools/solver-ipm-v41.exe"
PREFIX = "V42_NEWTON "
SOURCES = ["v42-newton-diagnostic.hpp", "v42-newton-driver.cpp", "v42-newton-instrument.py",
           "v42-newton-build-run.py", "v42-newton-self-test.cpp"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def production_identity():
    identity = read(DATA / "ipm-v41-identity.json")
    assert sha(PRODUCTION) == sha(BUILD / "solver.exe") == identity["solver_sha256"]
    assert {name: sha(ROOT / name) for name in identity["files"]} == identity["files"]
    assert {name: sha(ROOT / name) for name in identity["libraries"]} == identity["libraries"]
    return identity


def plan():
    destination = DATA / "v42-newton-plan.json"
    assert not destination.exists(), "计划不能覆盖"
    previous = read(DATA / "v41-plan.json")
    analysis = load_module("v42_analysis_rules", DATA / "v42-newton-analyze.py")
    models = [{**model, "development": stage == "development"}
              for stage in ("development", "probe") for model in previous["sets"][stage]["models"]]
    assert len(models) == 13
    for model in models:
        assert sha(Path(model["input"])) == model["sha256"]
    save(destination, {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_only": True, "models": models,
        "variants": ["production", "diagnostic_off", "diagnostic_on"],
        "time_limit": 15, "wall_timeout": 45, "iterations": 200, "scaling_passes": 5,
        "threads": 1, "seed": 0, "linear_solver": "normal_augmented", "accuracy": "full_newton",
        "direction_retries": 0, "shared_ipm": previous["shared_ipm"],
        "analysis_rules": analysis.ANALYSIS_RULES,
        "analysis_script_sha256": sha(DATA / "v42-newton-analyze.py"),
        "scope": "既有八开发及五失败输入，各正式程序/诊断关闭/开启一次；39次仅作机制诊断。15秒诊断不能混入公开5秒评分。",
        "comparison": "逐模型比较正式与关闭、关闭与开启的非时间完整报告及退出码；时间或墙钟耗尽单列，不以截断差异冒充数学变化。",
        "limitations": "记录标量及方向事件，不能由范数重建所有向量，也不能证明不存在可表示的double方向。",
    })
    print(json.dumps({"models": len(models), "plan_sha256": sha(destination)}), flush=True)


def events_from(stderr):
    events, errors = [], []
    for number, line in enumerate(stderr.splitlines(), 1):
        if line.startswith(PREFIX):
            try:
                events.append(json.loads(line[len(PREFIX):]))
            except Exception as error:
                errors.append({"line": number, "error": str(error), "raw": line})
    return events, errors


def build():
    assert not EXE.exists() and not (DATA / "v42-newton-identity.json").exists(), "不能覆盖已冻结诊断"
    declared = read(DATA / "v42-newton-plan.json")
    assert sha(DATA / "v42-newton-analyze.py") == declared["analysis_script_sha256"]
    locked = production_identity()
    WORK.mkdir(parents=True, exist_ok=True)
    frozen = WORK / "frozen"
    archive = DATA / "ipm-v41-source.zip"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            assert (frozen / name).resolve().is_relative_to(frozen.resolve())
        source.extractall(frozen)
    assert {name: sha(frozen / name) for name in locked["files"]} == locked["files"]
    instrument = load_module("v42_newton_instrument", DATA / "v42-newton-instrument.py")
    patched = instrument.instrument((frozen / "src/ipm.cpp").read_text(encoding="utf-8"))
    patched_path = WORK / "instrumented.cpp"
    patched_path.write_text(patched, encoding="utf-8")
    for name in ("v42-newton-driver.cpp", "v42-newton-diagnostic.hpp", "v42-newton-self-test.cpp"):
        shutil.copy2(DATA / name, WORK / name)
    compiler = Path(locked["compiler"])
    includes = [frozen / "include", WORK, BUILD / "_deps/qdldl-src/include",
                BUILD / "_deps/qdldl-build/include", BUILD / "_deps/suitesparse-src/AMD/Include",
                BUILD / "_deps/suitesparse-src/SuiteSparse_config"]
    version = re.search(r"project\(solver VERSION ([\d.]+)", (frozen / "CMakeLists.txt").read_text(encoding="utf-8"))[1]
    libraries = [ROOT / name for name in locked["libraries"]]
    blas = next(path for path in libraries if path.name == "libopenblas.a")
    command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-municode", "-Wall", "-Wextra",
               "-Wpedantic", f'-DSOLVER_VERSION="{version}"', *[f"-I{path}" for path in includes],
               str(WORK / "v42-newton-driver.cpp"), "-o", str(EXE)]
    command += [str(path) for path in libraries if path != blas]
    command += ["-lpthread", str(blas), "-lkernel32", "-luser32", "-lgdi32", "-lwinspool", "-lshell32",
                "-lole32", "-loleaut32", "-luuid", "-lcomdlg32", "-ladvapi32"]
    fixture = WORK / "v42-newton-self-test.exe"
    fixture_command = command.copy()
    fixture_command.remove("-municode")
    fixture_command[fixture_command.index(str(WORK / "v42-newton-driver.cpp"))] = str(WORK / "v42-newton-self-test.cpp")
    fixture_command[fixture_command.index("-o") + 1] = str(fixture)
    save(DATA / "v42-newton-command.json", {"build": command, "self_test_build": fixture_command, "cwd": str(ROOT)})
    for label, cmd in (("build", command), ("self-test-build", fixture_command)):
        process = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (DATA / f"v42-newton-{label}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
        assert process.returncode == 0, process.stderr[-3000:]
    # 先存原始输出，再解析断言，解析失败也不丢失自测证据。
    try:
        process = subprocess.run([str(fixture)], cwd=ROOT, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30)
        self_test = {"command": [str(fixture)], "returncode": process.returncode,
                     "stdout": process.stdout, "stderr": process.stderr, "wall_timeout": False}
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        self_test = {"command": [str(fixture)], "returncode": None,
                     "stdout": decode(error.stdout), "stderr": decode(error.stderr), "wall_timeout": True}
    save(DATA / "v42-newton-self-test.json", self_test)
    events, errors = events_from(self_test["stderr"])
    self_test.update(events=events, diagnostic_parse_errors=errors)
    save(DATA / "v42-newton-self-test.json", self_test)
    assert self_test["returncode"] == 0 and not errors and json.loads(self_test["stdout"])["passed"] is True
    patched_archive = DATA / "v42-newton-instrumented.cpp.gz"
    patched_archive.write_bytes(gzip.compress(patched_path.read_bytes(), mtime=0))
    assert production_identity() == locked
    save(DATA / "v42-newton-identity.json", {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "diagnostic_only": True,
        "production_solver_sha256": locked["solver_sha256"], "frozen_source_zip_sha256": sha(archive),
        "frozen_files": locked["files"], "diagnostic_sources": {name: sha(DATA / name) for name in SOURCES},
        "instrumented_sha256": sha(patched_path), "instrumented_archive_sha256": sha(patched_archive),
        "compiler": str(compiler), "compiler_sha256": sha(compiler), "libraries": locked["libraries"],
        "executable": str(EXE), "executable_sha256": sha(EXE), "self_test_executable_sha256": sha(fixture),
        "self_test_result_sha256": sha(DATA / "v42-newton-self-test.json"),
        "plan_sha256": sha(DATA / "v42-newton-plan.json"), "command_sha256": sha(DATA / "v42-newton-command.json"),
        "analysis_script_sha256": declared["analysis_script_sha256"],
    })
    print(json.dumps({"diagnostic_exe_sha256": sha(EXE), "self_test_passed": True}), flush=True)


def command_for(executable, model, plan):
    command = [str(executable), "solve", model["input"], "--backend", "ipm", "--time-limit", "15",
               "--threads", "1", "--seed", "0", "--iteration-limit", "200", "--scaling-passes", "5",
               "--ipm-linear-accuracy", "full_newton", "--ipm-direction-retries", "0"]
    for key, value in plan["shared_ipm"].items():
        command += ["--ipm-" + key, value]
    return command


def run():
    destination = DATA / "v42-newton-runs.jsonl"
    assert not destination.exists(), "不能覆盖正式诊断记录"
    identity = read(DATA / "v42-newton-identity.json")
    plan = read(DATA / "v42-newton-plan.json")
    assert sha(DATA / "v42-newton-plan.json") == identity["plan_sha256"]
    assert {name: sha(DATA / name) for name in identity["diagnostic_sources"]} == identity["diagnostic_sources"]
    production_identity()
    with destination.open("x", encoding="utf-8") as out:
        for model in plan["models"]:
            for variant in plan["variants"]:
                executable = PRODUCTION if variant == "production" else EXE
                before = {"solver": sha(executable), "input": sha(Path(model["input"]))}
                expected = identity["production_solver_sha256" if variant == "production" else "executable_sha256"]
                assert before == {"solver": expected, "input": model["sha256"]}
                env = os.environ.copy()
                env["SOLVER_V42_NEWTON"] = "1" if variant == "diagnostic_on" else "0"
                command = command_for(executable, model, plan)
                record = {"model": Path(model["input"]).stem, "input": model["input"], "model_sha256": model["sha256"],
                          "variant": variant, "command": command, "hashes_before": before,
                          "diagnostic_environment": env["SOLVER_V42_NEWTON"]}
                start = time.perf_counter()
                try:
                    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                             encoding="utf-8", errors="replace", timeout=plan["wall_timeout"])
                    record.update(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr, wall_timeout=False)
                except subprocess.TimeoutExpired as error:
                    def decode(value):
                        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                    record.update(returncode=None, stdout=decode(error.stdout), stderr=decode(error.stderr), wall_timeout=True)
                record["elapsed_wall_seconds"] = time.perf_counter() - start
                record["hashes_after"] = {"solver": sha(executable), "input": sha(Path(model["input"]))}
                record["report"] = None
                record["report_parse_error"] = None
                try:
                    record["report"] = json.loads(record["stdout"])
                except Exception as error:
                    record["report_parse_error"] = str(error)
                record["events"], record["diagnostic_parse_errors"] = events_from(record["stderr"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                assert record["hashes_before"] == record["hashes_after"]
                print(json.dumps({"model": record["model"], "variant": variant,
                                  "status": (record["report"] or {}).get("status"), "events": len(record["events"])}), flush=True)
    (DATA / "v42-newton-runs.jsonl.gz").write_bytes(gzip.compress(destination.read_bytes(), mtime=0))
    production_identity()
    summary()


def summary():
    path = DATA / "v42-newton-runs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    pairs = []
    for model in read(DATA / "v42-newton-plan.json")["models"]:
        by_variant = {row["variant"]: row for row in rows if row["input"] == model["input"]}
        for ref, cand in (("production", "diagnostic_off"), ("diagnostic_off", "diagnostic_on")):
            before, after = by_variant[ref], by_variant[cand]
            reports = [copy.deepcopy(row["report"]) for row in (before, after)]
            time_limited = any(row["wall_timeout"] or (row["report"] or {}).get("status") == "time_limit"
                               for row in (before, after))
            for report in reports:
                if report is not None:
                    report.pop("timings", None)
            pairs.append({"model": Path(model["input"]).stem, "reference": ref, "candidate": cand,
                          "time_limited": time_limited, "non_time_equal": reports[0] == reports[1]
                          and reports[0] is not None and before["returncode"] == after["returncode"]})
    result = {"diagnostic_only": True, "runs": len(rows), "raw_sha256": sha(path),
              "missing_reports": sum(row["report"] is None for row in rows),
              "diagnostic_parse_errors": sum(len(row["diagnostic_parse_errors"]) for row in rows),
              "events": sum(len(row["events"]) for row in rows), "comparisons": pairs}
    save(DATA / "v42-newton-summary.json", result)
    assert len(rows) == 39 and len(pairs) == 26 and not result["missing_reports"] and not result["diagnostic_parse_errors"]
    assert all(pair["time_limited"] or pair["non_time_equal"] for pair in pairs)
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "build", "run", "summary"))
    globals()[parser.parse_args().stage]()
