"""核验 v37 冻结来源、公开结果与独立结构审计，不重新运行求解。"""
import collections
import copy
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def require(condition, detail):
    if not condition:
        raise AssertionError(detail)

def main():
    identity = read(DATA / "ipm-v37-identity.json")
    snapshot = read(DATA / "v37-pre-v38-source-verification.json")
    expected = identity["solver_sha256"]
    require(snapshot["passed"] and not snapshot["errors"], "修改前来源快照未通过")
    require(snapshot["identity_sha256"] == sha(DATA / "ipm-v37-identity.json"), "来源身份变化")
    require(snapshot["source_hashes"] == identity["files"], "修改前当前来源与身份不符")
    require(len(identity["files"]) == 87, "来源数量不符")
    require(all(value == expected for value in snapshot["executables"].values()), "修改前两个程序不符")
    require(sha(ROOT / ".tools/solver-ipm-v37.exe") == expected, "冻结程序变化")
    require(snapshot["source_zip_sha256"] == sha(DATA / "ipm-v37-source.zip"), "源码压缩包变化")
    with zipfile.ZipFile(DATA / "ipm-v37-source.zip") as archive:
        zip_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                      for name in archive.namelist() if not name.endswith("/")}
    require(zip_hashes == identity["files"], "源码压缩包内容不符")
    previous = read(DATA / "ipm-v36-identity.json")
    groups, totals, performance = [], 0, {}
    manifests = sorted(DATA.glob("v37-*-manifest.json"))
    require(len(manifests) == 6, "六组对照不完整")
    inputs = {}
    for manifest_path in manifests:
        tag = manifest_path.name[len("v37-"):-len("-manifest.json")]
        original = ROOT / "results" / ("round-ipm-v37-" + tag)
        manifest = read(manifest_path)
        summary_path = DATA / ("v37-" + tag + "-summary.json")
        summary = read(summary_path)
        require(manifest_path.read_bytes() == (original / "manifest.json").read_bytes(), tag + " 清单字节不符")
        require(summary_path.read_bytes() == (original / "summary.json").read_bytes(), tag + " 汇总字节不符")
        raw_path = DATA / ("v37-" + tag + "-runs.jsonl.gz")
        raw = gzip.decompress(raw_path.read_bytes())
        require(raw == (original / "runs.jsonl").read_bytes(), tag + " 原始字节不符")
        rows = [json.loads(line) for line in raw.splitlines()]
        models = {item["model_index"]: item for item in manifest["models"]}
        require(sha(Path(manifest["models_file"])) == manifest["models_file_sha256"], tag + " 模型列表变化")
        require(manifest["comparison_script_sha256"] == identity["files"]["scripts/compare_lp.py"], tag + " 比较脚本身份")
        require(manifest["benchmark_script_sha256"] == identity["files"]["scripts/benchmark.py"], tag + " 基准脚本身份")
        require(manifest["time_limit"] == 5 and manifest["wall_timeout"] == 35 and manifest["threads"] == 1 and manifest["seed"] == 0, tag + " 资源协议")
        require(len(rows) == len(models) * manifest["repeats"] * 2 == summary["run_count"], tag + " 记录数量")
        seen = set()
        for row in rows:
            model = models[row["model_index"]]
            inputs.setdefault(model["input"], sha(Path(model["input"])))
            require(inputs[model["input"]] == model["sha256"], tag + " 当前输入哈希")
            require(row["input"] == model["input"], tag + " 输入路径")
            require(all(row[key] == model["sha256"] for key in ("model_sha256", "frozen_model_sha256", "model_sha256_after")), tag + " 输入前后哈希")
            variant = row["requested_variant"]
            require(manifest["solvers"][variant]["sha256"] == expected, tag + " 清单程序哈希")
            require(all(row[key] == expected for key in ("solver_sha256_before", "solver_sha256_after", "frozen_solver_sha256")), tag + " 程序前后哈希")
            key = (row["model_index"], row["repetition"], variant)
            require(key not in seen, tag + " 重复测量记录")
            seen.add(key)
            report = row["report"]
            require(report["backend_version"] == "ipm-cpu-0.34.0", tag + " 运行版本")
            configured = manifest["variants"][variant]["ipm_singleton_presolve"] == "on"
            require(report["options"]["ipm"]["presolve_singleton_rows"] is configured, tag + " 归并开关")
            count = report["ipm_presolved_singleton_rows"]
            require(type(count) is int and 0 <= count <= report["model"]["rows"], tag + " 归并行计数")
            require(configured or count == 0, tag + " 关闭归并计数")
            if tag.startswith("netlib-"):
                mode = manifest["variants"][variant]["ipm_linear_accuracy"]
                performance[(row["input"], mode, "on" if configured else "off")] = row
        for variant, metrics in summary["variants"].items():
            selected = [row for row in rows if row["requested_variant"] == variant]
            require(len(selected) == metrics["run_count"], tag + " 分组数量")
            require(sum(row["quality_pass"] for row in selected) == metrics["quality_pass_count"], tag + " 达标数量")
            require(dict(collections.Counter(row.get("status") or "no_status" for row in selected)) == metrics["status_counts"], tag + " 状态统计")
            for kind in ("numerical", "wall"):
                value = math.expm1(sum(math.log1p(row["quality_penalized_" + kind + "_seconds"] / 10) for row in selected) / len(selected)) * 10
                require(math.isclose(value, metrics["penalized_" + kind + "_sgm10_seconds"], rel_tol=1e-13, abs_tol=1e-13), tag + " 罚分重算")
        totals += len(rows)
        groups.append({"tag": tag, "models": len(models), "repeats": manifest["repeats"], "raw_records": len(rows),
                       "manifest_sha256": sha(manifest_path), "summary_sha256": sha(summary_path), "compressed_raw_sha256": sha(raw_path)})
    trajectory_path = DATA / "v37-presolve-off-trajectory-runs.json.gz"
    trajectories = json.loads(gzip.decompress(trajectory_path.read_bytes()))
    require(len(trajectories) == 24, "旧轨迹数量")
    for item in trajectories:
        reports = copy.deepcopy(item["reports"])
        require(len(reports) == 2 and item["non_time_equal"], "旧轨迹结构")
        require(reports[1]["options"]["ipm"]["presolve_singleton_rows"] is False and reports[1]["ipm_presolved_singleton_rows"] == 0, "旧轨迹新选项")
        for report in reports:
            for key in ("timings", "backend_version", "ipm_presolved_singleton_rows"):
                report.pop(key, None)
            report["options"]["ipm"].pop("presolve_singleton_rows", None)
        require(reports[0] == reports[1], "旧非时间轨迹不同：" + item["model"])
    audit_identity = read(DATA / "v37-structural-identity.json")
    audit_summary = read(DATA / "v37-structural-summary.json")
    audit_rows = [json.loads(line) for line in (DATA / "v37-structural-runs.jsonl").read_bytes().splitlines()]
    require(len(audit_rows) == len({row["input"] for row in audit_rows}) == 91, "结构审计输入数量")
    require(audit_identity["frozen_files"] == identity["files"] and audit_identity["source_zip_sha256"] == snapshot["source_zip_sha256"], "结构审计源码来源")
    for filename, key in (("driver.cpp", "driver_sha256"), ("build-run.py", "build_run_script_sha256"), ("runs.jsonl", "runs_sha256")):
        require(sha(DATA / ("v37-structural-" + filename)) == audit_identity[key], "结构审计文件哈希：" + filename)
    require(sha(Path(audit_identity["executable"])) == audit_identity["executable_sha256"], "结构审计程序哈希")
    require(audit_identity["production_solver_sha256_before"] == audit_identity["production_solver_sha256_after"] == expected, "结构审计期间生产程序身份")
    for source in audit_identity["performance_context_sources"]:
        require(sha(ROOT / source["path"]) == source["sha256"], "结构审计性能上下文身份")
    counter_checks = 0
    for row in audit_rows:
        require(sha(Path(row["input"])) == row["input_sha256"], "结构审计输入身份")
        structure = row["structure"]
        require(structure["iterations"] == structure["factorizations"] == 0 and structure["diagnostic_only"] is True, "结构审计必须无数值迭代")
        for key, indices in (("new_fixed_bound_columns", "new_fixed_original_column_indices"),
                             ("lost_singleton_inequality_support_columns", "lost_support_original_column_indices"),
                             ("gained_singleton_inequality_support_columns", "gained_support_original_column_indices")):
            require(structure[key] == len(structure[indices]) == len(set(structure[indices])), "结构列计数")
            require(all(0 <= index < structure["original_columns"] for index in structure[indices]), "结构列索引")
        require(set(structure["lost_support_original_column_indices"]).issubset(structure["new_fixed_original_column_indices"]), "丢失支撑与等值界关系")
        for mode in ("strict", "curvature_regularization"):
            for side in ("off", "on"):
                raw_row = performance[(row["input"], mode, side)]
                context = row["performance_context"][mode][side]
                require(context["classification"] == raw_row["classification"] and context["status"] == raw_row["status"], "结构审计性能分类")
                actual_count = raw_row["report"]["ipm_presolved_singleton_rows"]
                require(actual_count == context["presolved_singleton_rows"] == (structure["presolved_singleton_rows"] if side == "on" else 0), "结构审计归并计数")
                counter_checks += 1
    for key, value in audit_summary["totals"].items():
        require(sum(row["structure"][key] for row in audit_rows) == value, "结构审计总计：" + key)
    for summary_key, row_key in (("models_with_merged_rows", "presolved_singleton_rows"),
                                 ("models_with_new_fixed_bounds", "new_fixed_bound_columns"),
                                 ("models_losing_singleton_inequality_support", "lost_singleton_inequality_support_columns")):
        require(sum(row["structure"][row_key] > 0 for row in audit_rows) == audit_summary[summary_key], "结构审计模型计数")
    for mode in ("strict", "curvature_regularization"):
        for group, changes in audit_summary["support_loss_performance_context"][mode].items():
            for transition, value in changes.items():
                before, after = transition.split("_to_")
                actual = sum(bool(row["structure"]["lost_singleton_inequality_support_columns"]) == (group == "loss")
                             and row["performance_context"][mode]["off"]["classification"] == before
                             and row["performance_context"][mode]["on"]["classification"] == after for row in audit_rows)
                require(actual == value, "结构审计交叉计数")
    documents = list(ROOT.glob("*.md"))
    for directory in ("knowledge", "changelog", "benchmarks"):
        documents.extend((ROOT / directory).rglob("*.md"))
    links, broken = 0, []
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            links += 1
            local = unquote(target.split("#")[0].strip("<>"))
            if not (path.parent / local).exists():
                broken.append({"file": str(path.relative_to(ROOT)), "target": target})
    require(not broken, "文档断链：" + str(broken))
    result = {"verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "solver_sha256": expected, "source_files_verified": len(identity["files"]),
              "source_current_scope": "当前生产文件与build程序在v38修改前快照核对；本次核验继续检查冻结包与v37程序，不把后续生产改动当作v37偏差。",
              "pre_v38_source_snapshot": "v37-pre-v38-source-verification.json",
              "pre_v38_source_snapshot_sha256": sha(DATA / "v37-pre-v38-source-verification.json"),
              "changed_production_files": [name for name, digest in identity["files"].items() if previous["files"].get(name) != digest],
              "archive_groups_verified": groups, "raw_records_verified": totals,
              "model_and_solver_pre_post_hash_records_verified": totals,
              "presolve_off_trajectories_verified": len(trajectories),
              "trajectory_exclusions": ["timings", "backend_version", "新增且关闭的presolve_singleton_rows与其零计数"],
              "structural_audit_inputs_verified": len(audit_rows), "structural_performance_count_matches": counter_checks,
              "structural_driver_identity_verified": True, "structural_summary_recomputed": True,
              "structural_scope": "0次迭代和分解，仅解析及变换；相关性不构成因果证明。",
              "markdown_files_checked": len(documents), "local_links_checked": links, "broken_links": broken,
              "markdown_scope": "项目维护文档，不含build第三方依赖、.tools与results。",
              "verification_script_sha256": sha(Path(__file__)), "passed": True}
    (DATA / "v37-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("archive_groups_verified", "changed_production_files")}, ensure_ascii=False))

if __name__ == "__main__":
    main()
