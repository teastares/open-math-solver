"""在冻结 v39 副本中观测外层进展；只作机制诊断，不产生公开性能评分。"""
from __future__ import annotations

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
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
WORK = ROOT / ".tools/v40-progress"
BUILD = ROOT / "build/release"
EXE = WORK / "v40-progress.exe"
PRODUCTION = ROOT / ".tools/solver-ipm-v39.exe"
sys.path.insert(0, str(ROOT / "scripts"))
import compare_lp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def libraries():
    names = [
        "libsolver_highs.a", "libsolver_pdlp.a", "libsolver_ipm.a", "libsolver_core.a",
        "libsolver_qdldl_extended.a", "libsolver_sparse_lu.a", "libsolver_normal_cholesky.a",
        "_deps/suitesparse-build/AMD/libamd.a",
        "_deps/suitesparse-build/SuiteSparse_config/libsuitesparseconfig.a",
        "libsolver_multifrontal_ldl.a", "_deps/qdldl-build/out/libqdldl.a",
        "libsolver_factorhighs_kernel.a", "bin/libhighs.a", "bin/libhighs_extras.a",
    ]
    return [BUILD / name for name in names]


def plan():
    development = compare_lp.load_models(ROOT / "benchmarks/development.txt")[0]
    netlib = compare_lp.load_models(ROOT / "benchmarks/netlib-all.txt")[0]
    extra_names = {"dfl001", "finnis", "greenbea", "bnl2", "grow22"}
    extra = [model for model in netlib if Path(model["input"]).stem in extra_names]
    assert len(development) == 8 and len(extra) == 5
    models = [{**model, "development": True} for model in development]
    models += [{**model, "development": False} for model in extra]
    return {
        "diagnostic_only": True,
        "models": models,
        "configurations": [
            {"accuracy": "strict", "retries": 0},
            {"accuracy": "curvature_regularization", "retries": 8},
            {"accuracy": "blockwise", "retries": 8},
        ],
        "time_limit": 10, "wall_timeout": 45, "iterations": 200,
        "linear_solver": "normal_augmented", "history": 0,
        "threads": 1, "seed": 0, "singleton_elimination": True,
        "fixed_variable_elimination": True, "kkt_equilibration": True,
        "singleton_presolve": False, "presolve_fixed_elimination": False,
        "scope": "8个冻结开发输入及5个既有失败输入用于定位机制；不是独立评估集。",
        "comparison": "每个模型/配置诊断关闭和开启各一次；另对8个开发输入运行正式v39以核对独立编译。时间耗尽的截断单列，不要求非时间报告相同。",
    }


def build():
    assert not EXE.exists(), "不能覆盖已有诊断程序"
    WORK.mkdir(parents=True, exist_ok=True)
    locked = read(DATA / "ipm-v39-identity.json")
    assert sha(PRODUCTION) == sha(BUILD / "solver.exe") == locked["solver_sha256"]
    for name, expected in locked["files"].items():
        assert sha(ROOT / name) == expected, name
    archive = DATA / "ipm-v39-source.zip"
    frozen = WORK / "frozen"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            assert (frozen / name).resolve().is_relative_to(frozen.resolve()), name
        source.extractall(frozen)
    for name, expected in locked["files"].items():
        assert sha(frozen / name) == expected, name
    helper_path = DATA / "v40-progress-instrument.py"
    spec = importlib.util.spec_from_file_location("progress_instrument", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    patched = helper.instrument((frozen / "src/ipm.cpp").read_text(encoding="utf-8"))
    patched_path = WORK / "instrumented.cpp"
    patched_path.write_text(patched, encoding="utf-8")
    for name in ("v40-progress-driver.cpp", "v40-progress-diagnostic.hpp", "v40-progress-self-test.cpp"):
        shutil.copy2(DATA / name, WORK / name)
    version_match = re.search(r"project\(solver VERSION ([\d.]+)",
                              (frozen / "CMakeLists.txt").read_text(encoding="utf-8"))
    assert version_match
    compiler = ROOT / ".tools/llvm-mingw-20261006-ucrt-x86_64/bin/clang++.exe"
    includes = [frozen / "include", WORK, BUILD / "_deps/qdldl-src/include",
                BUILD / "_deps/qdldl-build/include", BUILD / "_deps/suitesparse-src/AMD/Include",
                BUILD / "_deps/suitesparse-src/SuiteSparse_config"]
    blas = BUILD / "_deps/solver-openblas-install/lib/libopenblas.a"
    command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-municode",
               "-Wall", "-Wextra", "-Wpedantic", f'-DSOLVER_VERSION="{version_match[1]}"']
    command += [f"-I{path}" for path in includes]
    command += [str(WORK / "v40-progress-driver.cpp"), "-o", str(EXE)]
    command += [str(path) for path in libraries()]
    command += ["-lpthread", str(blas), "-lkernel32", "-luser32", "-lgdi32", "-lwinspool",
                "-lshell32", "-lole32", "-loleaut32", "-luuid", "-lcomdlg32", "-ladvapi32"]
    declared = plan()
    save(DATA / "v40-progress-plan.json", declared)
    fixture = WORK / "v40-progress-self-test.exe"
    fixture_command = command.copy()
    fixture_command.remove("-municode")
    fixture_command[fixture_command.index(str(WORK / "v40-progress-driver.cpp"))] = str(WORK / "v40-progress-self-test.cpp")
    fixture_command[fixture_command.index("-o") + 1] = str(fixture)
    save(DATA / "v40-progress-command.json", {"build": command, "self_test_build": fixture_command, "cwd": str(ROOT)})
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    (DATA / "v40-progress-build.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stderr[-3000:]
    process = subprocess.run(fixture_command, cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    (DATA / "v40-progress-self-test-build.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stderr[-3000:]
    process = subprocess.run([str(fixture)], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30)
    events = [json.loads(line[len("V40_PROGRESS "):]) for line in process.stderr.splitlines()
              if line.startswith("V40_PROGRESS ")]
    tests = {event["event"]: event for event in events}
    expected = {
        "begin": {"rp_ld_inf": 11, "rd_ld_inf": 7, "working_gap_raw": 18,
                  "working_gap_complementarity": 6, "working_gap_dual_term": -1,
                  "working_gap_primal_term": 13, "working_gap_identity_error": 0,
                  "rp_algorithm_ld_difference": 0, "rd_algorithm_ld_difference": 0},
        "before_step": {"ep_ld_inf": 9, "ed_ld_inf": 9, "ep_algorithm_inf": 9,
                        "ed_algorithm_inf": 9, "h_complementarity_defect_inf": 9,
                        "predicted_rp_ld_inf": 9.75, "predicted_rd_ld_inf": 8},
        "after_step": {"rp_ld_inf": 9.75, "rd_ld_inf": 8, "working_gap_raw": 18.5,
                       "working_gap_complementarity": 8.9375, "working_gap_dual_term": -.1875,
                       "working_gap_primal_term": 9.75, "working_gap_identity_error": 0,
                       "rp_update_prediction_drift": 0, "rd_update_prediction_drift": 0,
                       "diagnostic_csc_scans": 3, "diagnostic_vector_products": 7},
    }
    save(DATA / "v40-progress-self-test.json", {"command": [str(fixture)], "returncode": process.returncode,
        "stdout": process.stdout, "stderr": process.stderr, "events": events, "expected": expected})
    assert process.returncode == 0 and json.loads(process.stdout)["disabled_no_scans"] is True
    assert len(events) == 5 and tests["run_start"]["long_double_digits"] > 53
    for event, fields in expected.items():
        for field, value in fields.items():
            assert tests[event][field] == value and tests[event][field + "_valid"] is True, (event, field)
    for field in ("overflow", "underflow", "infinite"):
        assert tests["range_test"][field] is None and tests["range_test"][field + "_valid"] is False
    archive_patched = DATA / "v40-progress-instrumented.cpp.gz"
    archive_patched.write_bytes(gzip.compress(patched_path.read_bytes(), mtime=0))
    assert sha(PRODUCTION) == sha(BUILD / "solver.exe") == locked["solver_sha256"]
    sources = ["v40-progress-driver.cpp", "v40-progress-diagnostic.hpp", "v40-progress-instrument.py",
               "v40-progress-build-run.py", "v40-progress-self-test.cpp"]
    save(DATA / "v40-progress-identity.json", {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_only": True, "production_solver_sha256": locked["solver_sha256"],
        "frozen_source_zip_sha256": sha(archive), "frozen_files": locked["files"],
        "diagnostic_sources": {name: sha(DATA / name) for name in sources},
        "instrumented_sha256": sha(patched_path), "instrumented_archive_sha256": sha(archive_patched),
        "compiler": str(compiler), "compiler_sha256": sha(compiler),
        "libraries": {str(path.relative_to(ROOT)): sha(path) for path in libraries() + [blas]},
        "executable": str(EXE), "executable_sha256": sha(EXE),
        "self_test_executable_sha256": sha(fixture),
        "self_test_result_sha256": sha(DATA / "v40-progress-self-test.json"),
        "plan_sha256": sha(DATA / "v40-progress-plan.json"),
        "command_sha256": sha(DATA / "v40-progress-command.json"),
    })
    print(json.dumps({"built": str(EXE), "sha256": sha(EXE)}, ensure_ascii=False), flush=True)


def arguments(executable, model, configuration, declared):
    return [str(executable), "solve", model["input"], "--backend", "ipm",
            "--time-limit", str(declared["time_limit"]), "--iteration-limit", str(declared["iterations"]),
            "--threads", "1", "--seed", "0", "--ipm-linear-solver", "normal_augmented",
            "--ipm-linear-accuracy", configuration["accuracy"], "--ipm-direction-retries", str(configuration["retries"]),
            "--ipm-singleton-elimination", "on", "--ipm-fixed-variable-elimination", "on",
            "--ipm-kkt-equilibration", "on", "--ipm-singleton-presolve", "off",
            "--ipm-presolve-fixed-elimination", "off", "--ipm-krylov-history-size", "0"]


def assert_identity(identity):
    assert sha(EXE) == identity["executable_sha256"]
    assert sha(PRODUCTION) == sha(BUILD / "solver.exe") == identity["production_solver_sha256"]
    for name, expected in identity["diagnostic_sources"].items():
        assert sha(DATA / name) == expected, name
    for name, expected in identity["frozen_files"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in identity["libraries"].items():
        assert sha(ROOT / name) == expected, name
    assert sha(DATA / "v40-progress-plan.json") == identity["plan_sha256"]


def run():
    path = DATA / "v40-progress-runs.jsonl"
    assert not path.exists(), "不能覆盖已有诊断记录"
    identity = read(DATA / "v40-progress-identity.json")
    assert_identity(identity)
    declared = read(DATA / "v40-progress-plan.json")
    completed = 0
    with path.open("w", encoding="utf-8") as output:
        for configuration in declared["configurations"]:
            for model in declared["models"]:
                variants = [("diagnostic_off", EXE, "0"), ("diagnostic_on", EXE, "1")]
                if model["development"]:
                    variants.append(("production", PRODUCTION, "0"))
                for variant, executable, enabled in variants:
                    assert sha(Path(model["input"])) == model["sha256"]
                    command = arguments(executable, model, configuration, declared)
                    environment = os.environ.copy()
                    environment["SOLVER_V40_PROGRESS"] = enabled
                    try:
                        process = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                                 text=True, encoding="utf-8", errors="replace",
                                                 timeout=declared["wall_timeout"])
                        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
                        timeout = False
                    except subprocess.TimeoutExpired as error:
                        def decoded(value):
                            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
                        stdout, stderr, returncode = decoded(error.stdout), decoded(error.stderr), None
                        timeout = True
                    try:
                        report = json.loads(stdout)
                    except json.JSONDecodeError:
                        report = None
                    events, other_lines, parse_errors = [], [], []
                    for line in stderr.splitlines():
                        if line.startswith("V40_PROGRESS "):
                            try:
                                events.append(json.loads(line[len("V40_PROGRESS "):]))
                            except json.JSONDecodeError as error:
                                # 墙钟中断可能留下半行；保留原文和错误，不能丢掉本次记录。
                                parse_errors.append({"line": line, "error": str(error)})
                        else:
                            other_lines.append(line)
                    assert sha(Path(model["input"])) == model["sha256"]
                    assert sha(executable) == identity["executable_sha256" if executable == EXE else "production_solver_sha256"]
                    record = {
                        "model": Path(model["input"]).stem, "input": model["input"], "input_sha256": model["sha256"],
                        "development": model["development"], "configuration": configuration, "variant": variant,
                        "command": command, "diagnostic_environment": {"SOLVER_V40_PROGRESS": enabled},
                        "solver_sha256": sha(executable), "returncode": returncode, "wall_timeout": timeout,
                        "stdout": stdout, "stderr": stderr, "report": report,
                        "events": events, "other_stderr_lines": other_lines,
                        "diagnostic_parse_errors": parse_errors,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    completed += 1
                print(json.dumps({"completed": completed, "accuracy": configuration["accuracy"],
                                  "model": Path(model["input"]).stem}, ensure_ascii=False), flush=True)
    assert_identity(identity)
    path.with_suffix(".jsonl.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    print(json.dumps({"records": completed, "runs_sha256": sha(path)}, ensure_ascii=False), flush=True)


def normalize(report):
    result = copy.deepcopy(report)
    result.pop("timings", None)
    return result


def summarize():
    path = DATA / "v40-progress-runs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 102
    groups = {}
    for row in rows:
        key = (row["configuration"]["accuracy"], row["input"])
        assert row["variant"] not in groups.setdefault(key, {})
        groups[key][row["variant"]] = row
    comparisons, statuses = [], {}
    for (accuracy, input_path), group in groups.items():
        for variant, row in group.items():
            status = (row.get("report") or {}).get("status", "missing_report")
            statuses.setdefault(accuracy, {}).setdefault(variant, {}).setdefault(status, 0)
            statuses[accuracy][variant][status] += 1
        for reference, candidate in (("diagnostic_off", "diagnostic_on"), ("production", "diagnostic_off")):
            if reference not in group:
                continue
            left, right = group[reference], group[candidate]
            time_limited = any(row["wall_timeout"] or (row["report"] or {}).get("status") == "time_limit"
                               for row in (left, right))
            equal = left["returncode"] == right["returncode"] and left["report"] is not None and right["report"] is not None \
                and normalize(left["report"]) == normalize(right["report"])
            comparisons.append({"accuracy": accuracy, "model": Path(input_path).stem,
                                "reference": reference, "candidate": candidate,
                                "time_limited": time_limited, "non_time_equal": equal,
                                "reference_status": (left["report"] or {}).get("status"),
                                "candidate_status": (right["report"] or {}).get("status")})
    result = {"diagnostic_only": True, "raw_records": len(rows), "groups": len(groups),
              "comparisons": comparisons, "status_counts": statuses,
              "non_time_mismatches_excluding_time_limits": [item for item in comparisons
                    if not item["time_limited"] and not item["non_time_equal"]],
              "missing_reports": sum(row["report"] is None for row in rows),
              "diagnostic_parse_errors": sum(len(row["diagnostic_parse_errors"]) for row in rows),
              "events": sum(len(row["events"]) for row in rows),
              "runs_sha256": sha(path)}
    save(DATA / "v40-progress-summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "run", "summarize"))
    args = parser.parse_args()
    {"build": build, "run": run, "summarize": summarize}[args.stage]()
