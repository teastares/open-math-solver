"""只解析和变换的冻结 v37 结构诊断；不运行 LP 迭代，不修改正式程序。"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[4]
DATA = ROOT / "knowledge/experiments/data/2026-10-12-halpern-ipm"
WORK = ROOT / ".tools/v37-structural"
BUILD = ROOT / "build/release"

def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name: str, value: object) -> None:
    (DATA / f"v37-structural-{name}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    frozen = WORK / "frozen"
    archive = DATA / "ipm-v37-source.zip"
    locked = json.loads((DATA / "ipm-v37-identity.json").read_text(encoding="utf-8"))
    production = ROOT / "build/release/solver.exe"
    production_before = sha(production)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            target = (frozen / item.filename).resolve()
            if not target.is_relative_to(frozen.resolve()):
                raise RuntimeError("源码归档路径超出诊断目录")
        source.extractall(frozen)
    frozen_hashes = {}
    for name, expected in locked["files"].items():
        path = frozen / name
        if path.is_file():
            actual = sha(path)
            if actual != expected:
                raise RuntimeError(f"冻结源码身份不符：{name}")
            frozen_hashes[name] = actual
    driver = WORK / "v37-structural-driver.cpp"
    shutil.copyfile(DATA / driver.name, driver)
    executable = WORK / "v37-structural.exe"
    compiler = ROOT / ".tools/llvm-mingw-20261006-ucrt-x86_64/bin/clang++.exe"
    libraries = [
        "libsolver_highs.a", "libsolver_pdlp.a", "libsolver_ipm.a", "libsolver_core.a",
        "libsolver_qdldl_extended.a", "libsolver_sparse_lu.a", "libsolver_normal_cholesky.a",
        "_deps/suitesparse-build/AMD/libamd.a",
        "_deps/suitesparse-build/SuiteSparse_config/libsuitesparseconfig.a",
        "libsolver_multifrontal_ldl.a", "_deps/qdldl-build/out/libqdldl.a",
        "libsolver_factorhighs_kernel.a", "bin/libhighs.a", "bin/libhighs_extras.a",
    ]
    openblas = BUILD / "_deps/solver-openblas-install/lib/libopenblas.a"
    command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic"]
    includes = [frozen / "include", BUILD / "_deps/qdldl-src/include", BUILD / "_deps/qdldl-build/include",
                BUILD / "_deps/suitesparse-src/AMD/Include", BUILD / "_deps/suitesparse-src/SuiteSparse_config"]
    command += [f"-I{path}" for path in includes]
    command += [str(driver), "-o", str(executable)]
    command += [str(BUILD / name) for name in libraries]
    command += ["-lpthread", str(openblas), "-lkernel32", "-luser32", "-lgdi32", "-lwinspool",
                "-lshell32", "-lole32", "-loleaut32", "-luuid", "-lcomdlg32", "-ladvapi32"]
    save("commands", {"build": command, "cwd": str(ROOT), "runtime_template": [str(executable), "INPUT"]})
    build = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (DATA / "v37-structural-build.log").write_text(build.stdout + build.stderr, encoding="utf-8")
    if build.returncode:
        raise RuntimeError(f"诊断编译失败：{build.returncode}")
    manifest = json.loads((DATA / "v37-netlib-normal_augmented-strict-ipm-manifest.json").read_text(encoding="utf-8"))
    contexts = {}
    context_sources = []
    for mode in ("strict", "curvature_regularization"):
        source = DATA / f"v37-netlib-normal_augmented-{mode}-ipm-runs.jsonl.gz"
        context_sources.append({"path": str(source.relative_to(ROOT)), "sha256": sha(source)})
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line in stream:
                run = json.loads(line)
                report = run.get("report") or {}
                enabled = report.get("options", {}).get("ipm", {}).get("presolve_singleton_rows")
                key = pathlib.Path(run["input"]).name
                entry = contexts.setdefault(key, {}).setdefault(mode, {})
                entry["on" if enabled else "off"] = {
                    "status": run.get("status"), "classification": run.get("classification"),
                    "message": report.get("message"),
                    "presolved_singleton_rows": report.get("ipm_presolved_singleton_rows"),
                }
    rows = []
    with (DATA / "v37-structural-runs.jsonl").open("w", encoding="utf-8") as output:
        for item in manifest["models"]:
            path = pathlib.Path(item["input"])
            actual = sha(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"输入身份不符：{path}")
            run = subprocess.run([str(executable), str(path)], cwd=ROOT, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=120)
            if run.returncode:
                raise RuntimeError(f"结构诊断失败：{path.name}：{run.stderr}")
            structure = json.loads(run.stdout)
            for mode, context in contexts[path.name].items():
                if context["off"]["presolved_singleton_rows"] != 0:
                    raise RuntimeError(f"关闭归并的性能记录计数异常：{path.name}/{mode}")
                if context["on"]["presolved_singleton_rows"] != structure["presolved_singleton_rows"]:
                    raise RuntimeError(f"归并计数与实际性能记录不符：{path.name}/{mode}")
            row = {"model_index": item["model_index"], "input": str(path), "input_sha256": actual,
                   "structure": structure, "performance_context": contexts[path.name], "stderr": run.stderr}
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
    regressions = {}
    for mode in ("strict", "curvature_regularization"):
        regressions[mode] = [pathlib.Path(row["input"]).stem for row in rows
                             if row["performance_context"][mode]["off"]["classification"] == "resolved"
                             and row["performance_context"][mode]["on"]["classification"] != "resolved"]
    summary = {
        "diagnostic_only": True, "models": len(rows), "iterations": 0, "factorizations": 0,
        "models_with_merged_rows": sum(row["structure"]["presolved_singleton_rows"] > 0 for row in rows),
        "models_with_new_fixed_bounds": sum(row["structure"]["new_fixed_bound_columns"] > 0 for row in rows),
        "models_losing_singleton_inequality_support": sum(row["structure"]["lost_singleton_inequality_support_columns"] > 0 for row in rows),
        "regression_models_from_archived_performance": regressions,
        "totals": {key: sum(row["structure"][key] for row in rows) for key in (
            "presolved_singleton_rows", "new_fixed_bound_columns",
            "lost_singleton_inequality_support_columns", "gained_singleton_inequality_support_columns")},
        "support_loss_performance_context": {
            mode: {
                "loss" if loss else "no_loss": {
                    f"{before}_to_{after}": sum(
                        bool(row["structure"]["lost_singleton_inequality_support_columns"]) == loss
                        and row["performance_context"][mode]["off"]["classification"] == before
                        and row["performance_context"][mode]["on"]["classification"] == after
                        for row in rows)
                    for before in ("resolved", "incomplete") for after in ("resolved", "incomplete")
                } for loss in (False, True)
            } for mode in ("strict", "curvature_regularization")
        },
        "checks": {
            "unique_models": len({row["input"] for row in rows}),
            "presolve_counts_match_both_archived_performance_modes": True,
            "fixed_variable_mappings_match_off_on": True,
            "zero_budget_parser_iterations_and_factorizations": True,
        },
        "metric_definitions": {
            "off_and_on": "同一解析模型，先沿用原固定变量消除，再按五遍原缩放构造锥模型；仅归并开关不同。",
            "new_fixed_bound_columns": "归并后出现原列界来源的单非零等式，归并前无此列界等式，且原输入列并非有限相等界；包含此前已被单非零输入等式隐含固定的变量，不代表新增数学约束。",
            "lost_singleton_inequality_support_columns": "归并前至少有一个仅含该坐标的非零不等式行，归并后完全没有；按实际 canonicalize 合并重复项后的矩阵计数，不等价于完整曲率矩阵失秩。",
            "column_indices": "零基原模型列序号，经过现有固定变量消除映射还原。",
        },
        "scope": "冻结实现只解析和 canonicalize；性能上下文来自已有归档，不是本诊断的性能成绩或因果证明。",
    }
    save("summary", summary)
    production_after = sha(production)
    if production_before != production_after:
        raise RuntimeError("正式求解器在诊断期间发生变化")
    save("identity", {
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_only": True, "source_zip_sha256": sha(archive), "frozen_files": frozen_hashes,
        "driver_sha256": sha(driver), "build_run_script_sha256": sha(pathlib.Path(__file__)),
        "compiler_sha256": sha(compiler), "executable": str(executable), "executable_sha256": sha(executable),
        "linked_libraries": {str((BUILD / name).relative_to(ROOT)): sha(BUILD / name) for name in libraries}
                           | {str(openblas.relative_to(ROOT)): sha(openblas)},
        "production_solver_sha256_before": production_before, "production_solver_sha256_after": production_after,
        "performance_context_sources": context_sources,
        "runs_sha256": sha(DATA / "v37-structural-runs.jsonl"),
        "options": {"eliminate_fixed_variables": True, "scaling_passes": 5, "presolve_singleton_rows": [False, True]},
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        name = pathlib.Path(row["input"]).stem
        if any(name in names for names in regressions.values()):
            structure = row["structure"]
            print(name, "merged", structure["presolved_singleton_rows"], "new_fixed", structure["new_fixed_bound_columns"],
                  "lost", structure["lost_singleton_inequality_support_columns"],
                  "barrier", structure["off"]["barrier_rows"], structure["on"]["barrier_rows"],
                  "eq", structure["off"]["equality_rows"], structure["on"]["equality_rows"])

if __name__ == "__main__":
    main()
