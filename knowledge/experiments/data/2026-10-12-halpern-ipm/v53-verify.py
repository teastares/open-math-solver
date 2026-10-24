"""独立核验第50轮诊断构建、运行轨迹、缓存与精选精确恒等式，不运行LP。"""

import argparse
import collections
import datetime
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "87fd87be5bc80deb0ee7689d5ad16635969b447e836f88fc8bb9019479f5cdf0"
PARENT_IDENTITY_SHA = "2e9cfcded7e59b37ec1af19273b15db75e4d3adcc27026b94e41f2ab69591cda"
PRODUCTION_SHA = "7dd9ce7c9b67389329cbd09db1753cc4c3b068bbaf1a5802a4b63edeffcc5963"
VARIANTS = ("production_v52", "diagnostic_off", "diagnostic_on")
CAPACITY = 128 * 1024**2
sys.dont_write_bytecode = True


def load_helper(name, expected):
    path = DATA / name
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise AssertionError("锁定辅助核验器身份改变：" + name)
    module = types.ModuleType("v53_" + name.replace("-", "_").replace(".", "_"))
    module.__file__ = str(path)
    exec(compile(content, str(path), "exec"), module.__dict__)
    return module


legacy = load_helper("v49-verify.py", "fa4bc92c70de8355a0f5f921c00ead9427b98b8c4a68d4b002ed4e6bdb188efa")
recent = load_helper("v52-verify.py", "323e5ece25007e2207d5028bd43cc65a1e7b4db86ca89746768e8b5afa6eebe3")
require, sha, read = recent.require, legacy.sha, recent.read
strict_json, finite = recent.strict_json, recent.finite
BinaryArrays, integer = legacy.BinaryArrays, legacy.integer
legacy.DTYPES = {**legacy.DTYPES, "u64_le": (8, "Q")}


def frozen_plan():
    require(sha(DATA / "v53-plan.json") == PLAN_SHA, "诊断计划冻结身份")
    plan = read(DATA / "v53-plan.json")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA == sha(DATA / "ipm-v52-identity.json") and
            plan["parent_solver_sha256"] == PRODUCTION_SHA, "固定v52生产来源")
    require(plan["kind"] == "diagnostic_only" and plan["production_changes"] is False and
            plan["variants"] == list(VARIANTS), "纯诊断与三模式合同")
    require([(item["name"], item["backends"]) for item in plan["models"]] ==
            [(name, ["normal_cholmod"]) for name in ("dfl001", "greenbea", "forplan", "afiro")], "固定输入及单后端顺序")
    require((plan["runs"], plan["repeats"], plan["time_limit"], plan["wall_timeout"], plan["threads"], plan["seed"],
             plan["max_iterations"], plan["scaling_passes"], plan["method"], plan["linear_accuracy"]) ==
            (12, 1, 5, 35, 1, 0, 200, 5, "homogeneous", "homogeneous_repair"), "运行数量及同政策预算")
    parent = read(DATA / "v52-plan.json")
    require(plan["shared_ipm"] == {**parent["shared_ipm"], "direction-retries": "0"}, "共同LP选项不变")
    require(len({item["input"] for item in plan["models"]}) == 4, "输入不能重复")
    return plan


def verify_production():
    identity = read(DATA / "ipm-v52-identity.json")
    require(len(identity["files"]) == 95 and len(identity["libraries"]) == 17 and
            identity["solver_sha256"] == PRODUCTION_SHA, "冻结来源集合")
    for group in ("files", "libraries"):
        for name, expected in identity[group].items():
            require(sha(ROOT / name) == expected, "生产来源不得变化：" + name)
    for name in (".tools/solver-ipm-v52.exe", "build/release/solver.exe"):
        require(sha(ROOT / name) == PRODUCTION_SHA, "两个生产程序不得改变")
    require(sha(identity["compiler"]) == identity["compiler_sha256"], "同一编译器")
    archive = DATA / "ipm-v52-source.zip"
    require(sha(archive) == identity["source_zip_sha256"], "冻结源码包身份")
    with zipfile.ZipFile(archive) as bundle:
        require(len(bundle.namelist()) == len(set(bundle.namelist())) and set(bundle.namelist()) == set(identity["files"]), "源码包唯一成员集合")
        for name, expected in identity["files"].items():
            require(hashlib.sha256(bundle.read(name)).hexdigest() == expected, "冻结源码包成员身份")
    return identity


def verify_build(work, plan, identity):
    build = read(work / "build-manifest.json")
    require(build["diagnostic_only"] is True and build["implicit_link_inputs_verified"] is False and
            build["plan_sha256"] == PLAN_SHA and build["frozen_files"] == identity["files"] and
            build["production_sha256"] == PRODUCTION_SHA and
            datetime.datetime.fromisoformat(build["created_at_utc"]) >= datetime.datetime.fromisoformat(plan["created_at_utc"]),
            "独立构建合同与测前冻结时间")
    for key in ("protected_before_and_after", "link_explicit_before_and_after", "compile_inputs_after_and_final", "outputs"):
        legacy.hashes(build[key])
    require(Path(build["production_executable"]) == ROOT / ".tools/solver-ipm-v52.exe" and
            Path(build["executable"]) == work / "solver-diagnostic.exe" and
            sha(build["executable"]) == build["executable_sha256"] != PRODUCTION_SHA, "生产与插桩程序身份")
    frozen = work / "frozen"
    legacy.hashes({str(frozen / name): expected for name, expected in identity["files"].items()})
    transforms = read(work / "transforms.json")
    require(set(transforms) == {"ipm"}, "只有ipm翻译单元可插桩")
    record = transforms["ipm"]
    source, patched = frozen / "src/ipm.cpp", work / "ipm-diagnostic.cpp"
    require(Path(record["source"]) == source and Path(record["patched"]) == patched and
            sha(source) == record["source_sha256"] and sha(patched) == record["patched_sha256"], "插桩源路径和身份")
    original = source.read_text(encoding="utf-8")
    changed, anchors, replacements = original, [], []
    for index, (anchor, before, after) in enumerate(legacy.literal_edits(DATA / "v53-hsd-instrumentation.py", "changes")):
        replacement = before + anchor + after
        require(original.count(anchor) == changed.count(anchor) == 1, "独立插桩锚点必须唯一")
        anchors.append({"index": index, "source_line": original[:original.index(anchor)].count("\n") + 1,
                        "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
                        "replacement_sha256": hashlib.sha256(replacement.encode()).hexdigest()})
        changed = changed.replace(anchor, replacement, 1)
        replacements.append((anchor, replacement))
    require(changed == patched.read_text(encoding="utf-8") and record["anchors"] == anchors, "不能信任变换脚本自报结果")
    for before, after in reversed(replacements):
        require(changed.count(after) == 1, "独立逆变换锚点必须唯一")
        changed = changed.replace(after, before, 1)
    require(changed == original, "插桩逆变换须还原冻结来源")
    command = read(work / "commands.json")
    require(command == build["commands"] and Path(command["cwd"]) == ROOT / "build/release", "实际构建命令清单")
    cwd = Path(command["cwd"])
    text = (work / "ninja-commands.txt").read_text(encoding="utf-8").splitlines()
    require([line for line in text if " -o solver.exe " in line] == [identity["link_command"]], "冻结链接命令文本")
    outer = legacy.windows_arguments(identity["link_command"])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "链接包装")
    original_link = legacy.windows_arguments(outer[2][len("cd . && "):-len(" && cd .")])
    require(command["original_link"] == original_link, "独立解码链接命令")
    original_compile, = [legacy.windows_arguments(line) for line in text if " -c " in line and line.replace("\\", "/").endswith("/src/ipm.cpp")]
    require(command["original_compile"] == {"ipm": original_compile} and Path(original_compile[0]) == Path(identity["compiler"]), "独立解码编译命令")
    expected_compile = list(original_compile)
    obj = work / "ipm-diagnostic.obj"
    for option, value in (("-c", patched), ("-o", obj), ("-MF", str(obj) + ".d"), ("-MT", obj)):
        require(expected_compile.count(option) == 1, "编译选项唯一")
        expected_compile[expected_compile.index(option) + 1] = str(value)
    positions = [index for index, value in enumerate(expected_compile) if value.startswith("-I") and legacy.path_equal(value[2:], ROOT / "include")]
    require(len(positions) == 1, "冻结include替换唯一")
    expected_compile[positions[0]] = "-I" + str(frozen / "include")
    expected_compile.append("-I" + str(DATA))
    require(command["compile"] == {"ipm": expected_compile}, "编译参数只能替换来源/输出和冻结头文件")
    expected_link, replaced = [], 0
    regularized = cwd / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    for value in original_link:
        if value == "libsolver_ipm.a":
            expected_link.extend([str(obj), str(regularized)])
            replaced += 1
        elif not value.startswith("-Wl,--out-implib,"):
            expected_link.append(value)
    require(replaced == 1 and expected_link.count("-o") == 1, "唯一替换ipm对象")
    expected_link[expected_link.index("-o") + 1] = str(work / "solver-diagnostic.exe")
    require(command["link"] == expected_link, "链接不能替换其他库或编译参数")
    explicit = {(cwd / value).resolve() for value in expected_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")}
    require({Path(value).resolve() for value in build["link_explicit_before_and_after"]} == explicit, "真实显式链接输入集合")
    protected = {Path(path).resolve() for path in build["protected_before_and_after"]}
    required = {DATA / name for name in ("v53-plan.json", "v53-build.py", "v53-hsd-instrumentation.py", "v53-hsd-diagnostic.hpp",
                "v53-hsd-self-test.cpp", "v53-analyze-hsd.py", "v47-profile-build.py", "ipm-v52-identity.json", "ipm-v52-source.zip")}
    required.update((ROOT / name).resolve() for name in identity["libraries"])
    required.update((regularized, Path(identity["compiler"]), ROOT / ".tools/solver-ipm-v52.exe"))
    require(required <= protected, "构建必需保护项完整")
    for name, expected in (("ipm-compile", expected_compile), ("link", expected_link), ("hsd-self-build", build["self_test_compile"])):
        log = read(work / (name + ".json"))
        require(log["command"] == expected and log["returncode"] == 0 and log["wall_timeout"] is False, "构建实际成功")
    for name in ("hsd-self-off-run", "hsd-self-on-run", "hsd-math-self-test"):
        log = read(work / (name + ".json"))
        require(log["returncode"] == 0 and log["wall_timeout"] is False and strict_json(log["stdout"])["passed"] is True, "合成自检实际成功")
    require(not list(work.glob("hsd-self-off-hsd.*")), "关闭开关的合成测试不能产生快照")
    return build, {"sources": 95, "libraries": 17, "reversible_anchors": len(anchors),
                   "explicit_link_inputs": len(explicit), "compile_inputs": len(build["compile_inputs_after_and_final"]),
                   "implicit_link_inputs_verified": False}


def selected_columns(n):
    require(type(n) is int and n > 0, "精选精确交叉要求非空列集合")
    return sorted({0, n // 2, n - 1})


def exact_column(arrays, col, tau, tau_new, theta, alpha, dtau):
    """只读取一个CSC列，按保存binary64值精确重算有符号分解。"""
    n = len(arrays["costs"])
    f = Fraction
    cost = f(arrays["costs"][col])
    z, z_new, dz = arrays["old_z"], arrays["new_z"], arrays["direction_z"]
    u, v = arrays["u"], arrays["v"]
    tau, tau_new, theta, alpha, dtau = map(f, (tau, tau_new, theta, alpha, dtau))
    products = collections.defaultdict(Fraction)
    for offset in range(arrays["starts"][col], arrays["starts"][col + 1]):
        row, value = arrays["rows"][offset], f(arrays["values"][offset])
        old, new, direction, uu, vv = map(f, (z[row], z_new[row], dz[row], u[n + row], v[n + row]))
        for key, item in (("old", old), ("new", new), ("direction", direction), ("u", uu), ("v", vv),
                          ("reconstruction", direction - uu - vv * dtau), ("update", new - old - alpha * direction)):
            products[key] += value * item
    stored = f(arrays["dual_residual"][col])
    rhs = f(arrays["direction_rhs"][col])
    old = products["old"] + cost * tau
    new = products["new"] + cost * tau_new
    eu, ev = products["u"] - rhs, products["v"] + cost
    rhs_rounding = rhs + theta * stored
    residual_rounding = theta * (old - stored)
    defect = products["direction"] + cost * dtau + theta * old
    separated = eu + dtau * ev + rhs_rounding + products["reconstruction"] + residual_rounding
    update_tau = cost * (tau_new - tau - alpha * dtau)
    predicted = (1 - alpha * theta) * old + alpha * defect + products["update"] + update_tau
    require(defect == separated and new == predicted, "保存方向与更新的精确恒等式")
    return {"rd_true": old, "rd_new": new, "e_u": eu, "e_v": ev,
            "dtau_e_v": dtau * ev, "rho_rhs": rhs_rounding, "transpose_rho_reconstruction": products["reconstruction"],
            "theta_residual_rounding": residual_rounding, "E_d": defect,
            "transpose_rho_update_z": products["update"], "cost_rho_update_tau": update_tau,
            "predicted_rd_new": predicted}


def verify_decimal(value, expected, precision, scale):
    require(isinstance(value, str), "高精度结果必须以文本保存")
    observed = Decimal(value)
    require(observed.is_finite(), "高精度结果必须有限")
    with localcontext() as context:
        context.prec = precision + 40
        exact = Decimal(expected.numerator) / Decimal(expected.denominator)
        tolerance = Decimal(scale.numerator) / Decimal(scale.denominator) * Decimal(10) ** (-precision + 12)
        require(abs(observed - exact) <= tolerance, "高精度结果与精选Fraction独立结果不符")


def process_state(row, benchmark):
    if row["error"] == "wall_timeout":
        require(row["returncode"] is None, "墙钟超时必须无正常退出码")
        return "wall_timeout"
    classification = benchmark.classify(row["returncode"], row["report"])
    return "valid" if recent.valid_process_report(row, classification) else "invalid"


def pair_result(a, b, benchmark):
    normalized = [{key: value for key, value in row["report"].items() if key != "timings"}
                  if isinstance(row["report"], dict) else None for row in (a, b)]
    equal = (json.dumps(normalized[0], sort_keys=True, allow_nan=False) == json.dumps(normalized[1], sort_keys=True, allow_nan=False) and
             (a["returncode"], a["error"]) == (b["returncode"], b["error"]))
    valid = all(process_state(row, benchmark) == "valid" for row in (a, b))
    budget = any(row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or
                 row["elapsed_wall_seconds"] > 35 or row["numerical_seconds"] is not None and row["numerical_seconds"] > 5 for row in (a, b))
    category = "invalid_process" if not valid else "budget_limited" if budget else "equal" if equal else "different"
    return {"equal": equal, "budget_limited": budget, "valid_processes": valid, "category": category}


def verify_runs(plan, identity, build, work):
    manifest = read(DATA / "v53-run-manifest.json")
    require(manifest["plan_sha256"] == PLAN_SHA and Path(manifest["build_manifest"]) == work / "build-manifest.json" and
            Path(manifest["work"]) == work and Path(manifest["capture_root"]) == work / "captures" and
            datetime.datetime.fromisoformat(manifest["started_at_utc"]) >= datetime.datetime.fromisoformat(build["created_at_utc"]),
            "构建后执行冻结诊断")
    require(manifest["environment"] == {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")} and
            manifest["removed_environment"] == "所有SOLVER_V开头诊断变量及OPENBLAS_CORETYPE；on只设置SOLVER_V53_DIAGNOSTICS", "线程与诊断开关隔离")
    legacy.hashes(manifest["protected"])
    protected = {Path(path).resolve() for path in manifest["protected"]}
    required = {DATA / "v53-run.py", work / "build-manifest.json", Path(build["production_executable"]), Path(build["executable"])}
    required.update((ROOT / name).resolve() for group in ("files", "libraries") for name in identity[group])
    require(required <= protected, "运行前后完整保护来源")
    benchmark, compare = legacy.frozen_quality_modules()
    rows = [strict_json(line) for line in gzip.decompress((DATA / "v53-runs.jsonl.gz").read_bytes()).splitlines()]
    schedule = [(index, model, variant) for index, model in enumerate(plan["models"])
                for variant in VARIANTS[index % 3:] + VARIANTS[:index % 3]]
    require(len(rows) == len(schedule) == 12, "所有12次运行按冻结轮转顺序保留")
    expected_flags = {"--backend": "ipm", "--time-limit": "5", "--threads": "1", "--seed": "0",
        "--iteration-limit": "200", "--scaling-passes": "5", "--ipm-method": "homogeneous",
        "--ipm-linear-accuracy": "homogeneous_repair", "--ipm-linear-solver": "normal_cholmod",
        **{"--ipm-" + key: value for key, value in plan["shared_ipm"].items()}}
    for index, (row, (case_index, model, variant)) in enumerate(zip(rows, schedule)):
        require((row["index"], row["case_index"], row["model"], row["input"], row["variant"], row["linear_solver"]) ==
                (index, case_index, model["name"], model["input"], variant, "normal_cholmod"), "真实运行配置和顺序")
        exe = build["production_executable"] if variant == "production_v52" else build["executable"]
        digest = PRODUCTION_SHA if variant == "production_v52" else build["executable_sha256"]
        require(row["command"][:3] == [exe, "solve", model["input"]] and legacy.flags(row["command"]) == expected_flags,
                "真实子命令必须指向对应冻结程序")
        require(row["hashes_before"] == row["hashes_after"] == {"input": model["sha256"], "solver": digest} and
                row["model_sha256"] == model["sha256"] == sha(model["input"]), "每次前后输入与程序身份")
        require(finite(row["elapsed_wall_seconds"]) and row["elapsed_wall_seconds"] > 0, "墙钟计时合法")
        if row["report"] is not None:
            require(strict_json(row["stdout"]) == row["report"] and row["report"]["input"] == model["input"], "原始输出与报告相同")
            recent.verify_options(row["report"], plan, "ipm", 52)
        else:
            require(row["error"] in ("wall_timeout", "launch_error", "invalid_json", "invalid_report"), "缺报告必须保留明确原因")
        classification = benchmark.classify(row["returncode"], row["report"]) if row["report"] is not None else row["error"]
        require(row["classification"] == classification, "不能信任自报进程分类")
        numerical = compare.numerical_seconds(row)
        failure = compare.quality_failure(row, model, model["sha256"], digest, digest)
        if failure is None and (numerical > 5 or row["elapsed_wall_seconds"] > 35):
            failure = "budget_exceeded"
        require(row["numerical_seconds"] == numerical and row["quality_failure"] == failure and row["quality_pass"] is (failure is None), "独立原模型质量与完整预算")
        prefix = work / "captures" / (str(case_index) + "-" + variant)
        require(Path(row["capture_prefix"]) == prefix, "快照前缀不能更换")
        observed = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(prefix.parent.glob(prefix.name + "-*"))}
        require(observed == row["capture_files"] and set(observed) <= {str(prefix) + "-hsd.json", str(prefix) + "-hsd.bin"}, "全部快照来源及文件集合")
        if variant != "diagnostic_on":
            require(not observed and "V53_HSD" not in row["stderr"], "关闭诊断不得产出快照")
    pairs, cases = [], []
    for case_index, model in enumerate(plan["models"]):
        selected = {row["variant"]: row for row in rows if row["case_index"] == case_index}
        require(set(selected) == set(VARIANTS), "每配置三模式齐全")
        for left, right in zip(VARIANTS, VARIANTS[1:]):
            pairs.append({"case_index": case_index, "model": model["name"], "linear_solver": "normal_cholmod",
                          "left": left, "right": right, **pair_result(selected[left], selected[right], benchmark)})
        cases.append({"case_index": case_index, "model": model["name"], "linear_solver": "normal_cholmod",
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"], "capture_prefix": selected["diagnostic_on"]["capture_prefix"]})
    expected = {"diagnostic_only": True, "runs": 12, "pairs": pairs, "cases": cases,
        "non_budget_pairs": sum(pair["valid_processes"] and not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(pair["category"] == "equal" for pair in pairs),
        "budget_pairs": sum(pair["category"] == "budget_limited" for pair in pairs),
        "invalid_pairs": sum(pair["category"] == "invalid_process" for pair in pairs),
        "different_pairs": sum(pair["category"] == "different" for pair in pairs), "raw_sha256": sha(DATA / "v53-runs.jsonl.gz")}
    summary = read(DATA / "v53-summary.json")
    require({key: summary[key] for key in expected} == expected, "8对轨迹和预算必须由全部原始记录重建")
    require(expected["invalid_pairs"] == expected["different_pairs"] == 0, "异常进程和非预算差异不能被预算豁免")
    return rows, summary


STATIC_FIELDS = ("g_starts", "g_rows", "g_values", "costs", "b", "inequality", "original_row", "dual_sign", "row_scale",
                 "column_scale", "original_columns", "fixed_values", "a_starts", "a_rows", "a_values", "original_costs",
                 "column_lower", "column_upper", "row_lower", "row_upper")
STEP_FIELDS = ("old_x", "old_z", "new_x", "new_z", "stored_rd", "rhs_u", "diagonal_h", "u", "v", "dz",
               "extracted_z", "assess_x", "assess_y")


def exact_scalar(text):
    value = legacy.exact_hex(text)
    require(math.isfinite(float(value)) and Fraction(float(value)) == value, "实际标量必须可由binary64精确表示")
    return value


def verify_csc(starts, rows, values, columns, height, sorted_rows=False):
    require(len(starts) == columns + 1 and starts[0] == 0 and starts[-1] == len(rows) == len(values), "CSC维度与完整末指针")
    for col in range(columns):
        require(0 <= starts[col] <= starts[col + 1] <= len(values), "CSC列指针范围")
        previous = -1
        for offset in range(starts[col], starts[col + 1]):
            row = rows[offset]
            require(0 <= row < height and (not sorted_rows or row > previous) and math.isfinite(values[offset]), "CSC行索引/有序性/数值")
            previous = row


def assessment_matches_report(assessment, report):
    require(assessment["available"] is True and assessment["replace"] is True, "终报来源必须是实际保留的验算")
    for section, target, rename in (("primal", report["verification"], {"objective": "recomputed_objective"}),
                                     ("optimality", report["lp_optimality"], {})):
        require(isinstance(target, dict), "终报应包含保留候选的验算摘要")
        for key, value in assessment["checks"][section].items():
            actual = target[rename.get(key, key)]
            if key == "errors":
                require(value["count"] == len(actual) and len(value["sample"]) <= len(actual) and
                        all(expected.encode("utf-8").startswith(sample.encode("utf-8")) for sample, expected in zip(value["sample"], actual)),
                        "错误样本及截断不能改换来源")
            elif isinstance(value, str):
                observed = float.fromhex(value)
                require(actual == observed if math.isfinite(observed) else actual is None, "终报保留验算数值不一致")
            else:
                require(type(actual) is type(value) and actual == value, "终报保留验算字段不一致")


def verify_capture(row):
    prefix = row["capture_prefix"]
    path, binary = Path(prefix + "-hsd.json"), Path(prefix + "-hsd.bin")
    if not path.exists():
        reason = "wall_timeout" if row["error"] == "wall_timeout" else "write_error" if '"write_error":true' in row["stderr"] else "missing_metadata_unexplained"
        return {"capture_state": reason, "complete": False, "exact": []}
    meta = read(path)
    require(meta["schema"] == "v53_hsd_1" and meta["enabled"] is True and meta["cache_limit"] == CAPACITY and
            meta["metadata_reserve"] == 65536 and meta["allocated_step_slots"] == 4, "冻结快照格式与总缓存上限")
    for key in ("cache_bytes", "static_payload_bytes", "step_slot_bytes", "copied_bytes", "completed_count", "assessment_count", "binary_bytes"):
        integer(meta[key], key)
    require(meta["cache_bytes"] <= CAPACITY and all(meta[key] == 0 and type(meta[key]) is int for key in
            ("extra_lp_products", "extra_lp_solves", "diagnostic_clock_reads")), "缓存计费与禁止额外LP工作")
    require(type(meta["run_unwound"]) is bool and type(meta["run_started"]) is bool and type(meta["ready"]) is bool, "运行状态类型")
    if not meta["binary_written"] or '"write_error":true' in row["stderr"]:
        return {"capture_state": "write_error", "complete": False, "exact": []}
    require(binary.is_file() and binary.stat().st_size == meta["binary_bytes"], "二进制写出状态与真实长度")
    if not meta["ready"]:
        require(not meta["static"]["arrays"] and not meta["steps"] and meta["last_attempt"] is None and
                meta["binary_bytes"] == 8 and binary.read_bytes() == b"V53HSD1\0", "未就绪不能暴露无效数组")
        require(meta["status"] in ("not_entered", "cache_limit", "memory_allocation_failed", "capture_exception", "unsupported_binary_format", "prefix_too_long",
                "unsupported_presolve_recovery", "dimension_mismatch", "direction_sequence_mismatch", "multiple_runs_unsupported"), "缺快照原因必须明确")
        return {"capture_state": meta["status"], "complete": False, "exact": []}
    require(meta["status"] == "captured" and meta["run_started"] is True, "就绪来源状态")
    descriptors, groups = [], {}
    def append(label, fields):
        require(len(fields) == len({item["name"] for item in fields}), "数组名不能重复")
        for item in fields:
            name = item["name"]
            dtype = "i64_le" if label == "static" and name in ("g_starts", "g_rows", "original_row", "a_starts", "a_rows") else \
                    "u64_le" if label == "static" and name == "original_columns" else "u8" if label == "static" and name == "inequality" else "f64_le"
            require(item["dtype"] == dtype, "每个数组类型与实际生产位宽一致")
        groups[label] = {item["name"]: label + ":" + item["name"] for item in fields}
        descriptors.extend(dict(item, name=label + ":" + item["name"]) for item in fields)
    append("static", meta["static"]["arrays"])
    require(tuple(groups["static"]) == STATIC_FIELDS, "全部静态数组顺序")
    steps = meta["steps"]
    count = meta["completed_count"]
    expected_iterations = sorted({1, max(1, count - 1), count}) if count else []
    require([step["iteration"] for step in steps] == expected_iterations and len(steps) <= 3, "首完整步及最后两步去重环形缓存")
    slots = list(enumerate(steps))
    if meta["last_attempt"] is not None:
        slots.append(("last", meta["last_attempt"]))
    for label, step in slots:
        append(str(label), step["arrays"])
    with BinaryArrays(binary, descriptors, b"V53HSD1\0", max_bytes=CAPACITY + 8) as values:
        static = {name: values[key] for name, key in groups["static"].items()}
        n, m, original_n, original_m = (len(static[name]) for name in ("costs", "b", "original_costs", "row_lower"))
        require(n > 0 and m > 0, "本轮诊断工作矩阵非空")
        verify_csc(static["g_starts"], static["g_rows"], static["g_values"], n, m, True)
        verify_csc(static["a_starts"], static["a_rows"], static["a_values"], original_n, original_m)
        counts = {"inequality": m, "original_row": m, "dual_sign": m, "row_scale": m, "column_scale": n,
                  "column_lower": original_n, "column_upper": original_n, "row_upper": original_m}
        require(all(len(static[name]) == size for name, size in counts.items()) and len(static["original_columns"]) in (0, n) and
                len(static["fixed_values"]) in (0, original_n), "静态映射与原模型维度")
        require(all(value in (0, 1) for value in static["inequality"]) and all(-1 <= value < original_m for value in static["original_row"]), "约束身份范围")
        require(all(0 <= value < original_n for value in static["original_columns"]) and len(set(static["original_columns"])) == len(static["original_columns"]), "原列映射唯一有效")
        for name in STATIC_FIELDS:
            if name in ("column_lower", "column_upper", "row_lower", "row_upper"):
                require(all(not math.isnan(value) for value in static[name]), "原边界允许无穷但禁止NaN")
            elif name not in ("g_starts", "g_rows", "original_row", "original_columns", "a_starts", "a_rows", "inequality"):
                require(all(math.isfinite(value) for value in static[name]), "静态double值有限")
        require(all(value > 0 for name in ("row_scale", "column_scale") for value in static[name]), "缩放系数严格为正")
        static_scalars = {key: exact_scalar(value) for key, value in meta["static"]["scalars"].items()}
        require(static_scalars["sense"] in (-1, 1) and static_scalars["objective_scale"] > 0, "原目标方向和尺度")
        static_bytes = sum(item["count"] * legacy.DTYPES[item["dtype"]][0] for item in meta["static"]["arrays"])
        step_bytes = 8 * (6 * n + 8 * m + original_n + original_m)
        require(meta["static_payload_bytes"] == static_bytes and meta["step_slot_bytes"] == step_bytes and
                meta["cache_bytes"] == 65536 + static_bytes + 4 * step_bytes and meta["copied_bytes"] >= static_bytes, "总缓存包含静态、四槽和元数据")
        exact_records = []
        for label, step in slots:
            flags = step["flags"]
            require(set(flags) == {"have_input", "have_u", "direction_returned", "step_complete", "assessed", "unwound"} and
                    all(type(value) is bool for value in flags.values()) and flags["have_input"] and
                    (not flags["direction_returned"] or flags["have_u"]) and (not flags["assessed"] or flags["step_complete"]), "阶段标志必须遵守先后关系")
            require(step["ordinal"] in (1, 2) and step["phase"] == ("affine" if step["ordinal"] == 1 else "corrector"), "方向阶段分类")
            if label == "last":
                require(not flags["step_complete"] and not flags["assessed"] and step["iteration"] == count + 1, "末尝试不得伪装完成步")
            else:
                require(flags["step_complete"] and flags["direction_returned"] and not flags["unwound"] and step["ordinal"] == 2, "只保留完成corrector")
            expected = [name for index, name in enumerate(STEP_FIELDS) if
                        (flags["step_complete"] if index in (2, 3) else flags["have_u"] if index == 7 else
                         flags["direction_returned"] if index in (8, 9) else flags["assessed"] if index >= 10 else flags["have_input"])]
            if label == "last":
                expected = []
            require(list(groups[str(label)]) == expected, "只写阶段有效数组，不暴露未初始化缓存")
            dims = (n, m, n, m, n, n + m, m, n + m, n + m, m, m, original_n, original_m)
            slot = {name: values[key] for name, key in groups[str(label)].items()}
            require(all(len(slot[name]) == dims[STEP_FIELDS.index(name)] and all(math.isfinite(value) for value in slot[name]) for name in slot), "动态数组维度及有限性")
            scalars = {name: exact_scalar(value) for name, value in step["scalars"].items()}
            require(scalars["tau"] > 0 and scalars["kappa"] > 0 and 0 <= scalars["theta"] <= 1, "实际HSD标量")
            require(all(type(value) is int and value >= 0 for value in step["counters"].values()) and
                    step["counters"]["successful_repairs"] <= step["counters"]["constant_repairs"] <= step["counters"]["constant_solves"], "快照计数合同")
            for key, report_key in {"factorizations": "ipm_factorizations", "refinements": "ipm_refinement_steps",
                    "constant_solves": "ipm_homogeneous_constant_solves", "direction_solves": "ipm_homogeneous_direction_solves",
                    "constant_repairs": "ipm_homogeneous_constant_repairs", "successful_repairs": "ipm_homogeneous_constant_repair_successes"}.items():
                require(step["counters"][key] <= row["report"][report_key], "快照工作量不能超出真实终报累计")
            if flags["step_complete"]:
                require(scalars["next_tau"] > 0 and 0 < scalars["step"] <= 1, "完成步标量合法")
                aliases = {"starts": static["g_starts"], "rows": static["g_rows"], "values": static["g_values"], "costs": static["costs"],
                           **slot, "direction_z": slot["dz"], "dual_residual": slot["stored_rd"], "direction_rhs": slot["rhs_u"]}
                exact_records.append({"iteration": step["iteration"], "coordinates": [{"column": column,
                    "values": exact_column(aliases, column, scalars["tau"], scalars["next_tau"], scalars["theta"], scalars["step"], scalars["dtau"])}
                    for column in selected_columns(n)]})
        report = row["report"]
        require(count == report["ipm_iterations"], "完成快照计数对应真实提交步数")
        latest, retained = meta["latest_assessment"], meta["retained_assessment"]
        require(latest["assessment_id"] == meta["assessment_count"] and retained["assessment_id"] <= latest["assessment_id"], "当前与保留候选来源有序")
        if retained["available"]:
            assessment_matches_report(retained, report)
        for step in steps:
            assessment = step["assessment"]
            require(assessment["available"] is step["flags"]["assessed"], "步验算可用标志")
            if assessment["available"]:
                require(assessment["iteration"] == step["iteration"] and 0 < assessment["assessment_id"] <= meta["assessment_count"], "步验算来源")
                if assessment["assessment_id"] == retained["assessment_id"]:
                    require(assessment == retained, "保留候选恰来自对应已保存步")
    return {"capture_state": "captured", "complete": True, "cache_bytes": meta["cache_bytes"], "completed_count": count,
            "saved_iterations": expected_iterations, "last_attempt_flags": None if meta["last_attempt"] is None else meta["last_attempt"]["flags"],
            "retained_assessment_id": meta["retained_assessment"]["assessment_id"], "latest_assessment_id": meta["latest_assessment"]["assessment_id"],
            "metadata_sha256": sha(path), "binary_sha256": sha(binary), "exact": exact_records}


def verify_packed(item, expected):
    require(item["numerator"] == str(expected.numerator) and item["denominator"] == str(expected.denominator) and
            type(item["sign"]) is int and item["sign"] == (expected > 0) - (expected < 0), "精确有理数及符号独立一致")
    verify_decimal(item["decimal80"], expected, 80, max(Fraction(1), abs(expected)))


def verify_analysis_records(analysis, rows, captures):
    require(analysis["schema"] == "v53_hsd_analysis_1" and analysis["passed"] is True and
            analysis["analyzer_sha256"] == sha(DATA / "v53-analyze-hsd.py") and
            analysis["self_test"] == {"passed": True, "checks": 26, "kind": "pure_synthetic_no_solver"}, "离线分析来源和实际纯自检")
    expected_paths = [Path(row["capture_prefix"] + "-hsd.json") for row in rows if row["variant"] == "diagnostic_on"]
    require([Path(record["metadata_path"]) for record in analysis["records"]] == expected_paths, "离线分析固定全部4输入，无事后筛选")
    checked = 0
    for record, path, audit in zip(analysis["records"], expected_paths, captures):
        meta = read(path)
        require(record["metadata_sha256"] == sha(path) and record["capture_status"] == meta["status"] and
                record["retained_assessment"] == meta["retained_assessment"] and record["latest_assessment"] == meta["latest_assessment"] and
                record["last_attempt"] == {key: value for key, value in (meta["last_attempt"] or {}).items() if key != "arrays"}, "离线来源与末尝试/保留候选逐字段一致")
        if not audit["complete"]:
            require(record["analysis_status"] == "not_captured" and not record["steps"], "缺快照不能声称数学归因")
            continue
        require(record["binary_sha256"] == audit["binary_sha256"] and
                record["analysis_status"] == ("analyzed" if audit["exact"] else "no_completed_step"), "离线二进制来源与完整步集合")
        require([step["iteration"] for step in record["steps"]] == audit["saved_iterations"], "每个保存完成步必须完整分析")
        for step, source, exact in zip(record["steps"], meta["steps"], audit["exact"]):
            assessment = source["assessment"]
            require(step["scalars"] == source["scalars"] and step["assessment_id"] == assessment["assessment_id"] and
                    step["replace"] == assessment["replace"] and step["production_checks"] == assessment["checks"] and
                    step["is_final_retained_assessment"] is (meta["retained_assessment"]["available"] and
                       assessment["assessment_id"] == meta["retained_assessment"]["assessment_id"]), "当前验算不能替代终报保留候选")
            columns = [item["column"] for item in exact["coordinates"]]
            require([item["column"] for item in step["fixed_coordinates"]] == columns and
                    [item["precision"] for item in step["decimal_crosschecks"]] == [80, 120], "固定精选列与两档精度")
            for actual, expected in zip(step["fixed_coordinates"], exact["coordinates"]):
                for name, value in expected["values"].items():
                    verify_packed(actual["exact"][name], value)
                    checked += 1
            for decimal in step["decimal_crosschecks"]:
                require(decimal["passed"] is True and [item["column"] for item in decimal["coordinates"]] == columns, "高精度独立交叉集合")
                for actual, expected in zip(decimal["coordinates"], exact["coordinates"]):
                    require(actual["passed"] is True, "逐坐标交叉须通过")
                    for name, value in expected["values"].items():
                        scale = max(Fraction(1), abs(value), abs(expected["values"]["rd_true"]), abs(expected["values"]["rd_new"]))
                        verify_decimal(actual["values"][name], value, decimal["precision"], scale)
            audit.setdefault("exact_columns_checked", []).append({"iteration": exact["iteration"], "columns": columns})
        # 完整有理数已保存在离线报告；独立核验只保留核对集合与数量，避免重复大分数。
        audit.pop("exact")
    return {"records": len(analysis["records"]), "fraction_components_independently_checked": checked,
            "decimal_precisions": [80, 120], "all_coordinate_analysis_recomputed_by_verifier": False,
            "limit": "核验器独立重算测前固定精选列的12项有符号分解；全坐标范数和原坐标归因由来源已锁定的离线分析器计算，未在本核验重复全部乘积。"}


def verify_offline(rows, captures, work, build):
    process = read(DATA / "v53-hsd-analysis-process.json")
    output = DATA / "v53-hsd-analysis.json"
    inputs = [work / "captures" / f"{index}-diagnostic_on-hsd.json" for index in range(4)]
    expected = [str(ROOT / ".tools/python-3.14.8/python.exe"), str(DATA / "v53-analyze-hsd.py"), *map(str, inputs), "--output", str(output)]
    require(process["command"] == expected and Path(process["cwd"]) == ROOT and process["wall_timeout_seconds"] == 120 and
            process["returncode"] == 0 and process["wall_timeout"] is False and finite(process["elapsed_wall_seconds"]) and
            process["elapsed_wall_seconds"] > 0, "独立离线命令及120秒墙钟合同")
    protected = process["protected_before"]
    require(protected == process["protected_after"], "离线前后来源一致")
    legacy.hashes(protected)
    required = {DATA / name for name in ("v53-offline.py", "v53-analyze-hsd.py", "v53-plan.json", "v53-run-manifest.json")}
    required.update([work / "build-manifest.json", *inputs, *[path.with_suffix(".bin") for path in inputs]])
    require({Path(path).resolve() for path in protected} == required and
            protected[str(DATA / "v53-analyze-hsd.py")] == build["protected_before_and_after"][str(DATA / "v53-analyze-hsd.py")], "离线保护集合与实际构建时分析器相同")
    require(process["output"] == {"path": str(output), "sha256": sha(output)}, "离线结果真实来源")
    result = verify_analysis_records(read(output), rows, captures)
    result.update(analysis_sha256=sha(output), process_sha256=sha(DATA / "v53-hsd-analysis-process.json"),
                  elapsed_wall_seconds=process["elapsed_wall_seconds"])
    return result


def self_test():
    benchmark = types.ModuleType("v53_synthetic_benchmark")
    benchmark.__file__ = str(ROOT / "scripts/benchmark.py")
    exec(compile((ROOT / "scripts/benchmark.py").read_bytes(), benchmark.__file__, "exec"), benchmark.__dict__)
    row = {"error": None, "returncode": 4, "classification": "incomplete", "numerical_seconds": 1,
           "elapsed_wall_seconds": 2, "report": {"schema_version": 1, "status": "numerical_error", "backend_version": "v52", "timings": {"solve_seconds": 1}}}
    import copy
    checked = 0
    def pair(other, expected):
        nonlocal checked
        require(pair_result(row, other, benchmark)["category"] == expected, "独立三模式轨迹反例")
        checked += 1
    pair(row, "equal")
    other = copy.deepcopy(row)
    other["report"]["timings"]["solve_seconds"] = 2
    pair(other, "equal")
    other["report"]["backend_version"] = "changed"
    pair(other, "different")
    other["report"]["status"] = "time_limit"
    pair(other, "budget_limited")
    other["returncode"] = 3221225477
    other["report"]["status"] = "verification_failed"
    pair(other, "invalid_process")
    other["report"] = None
    pair(other, "invalid_process")
    other.update(returncode=None, error="wall_timeout")
    pair(other, "invalid_process")
    other = copy.deepcopy(row)
    other["returncode"] = False
    pair(other, "invalid_process")
    arrays = {"starts": [0, 1], "rows": [0], "values": [2.0], "costs": [3.0], "old_z": [5.0], "new_z": [4.5],
              "direction_z": [-1.0], "u": [0.0, 1.0], "v": [0.0, -4.0], "dual_residual": [22.0], "direction_rhs": [-5.5]}
    exact = exact_column(arrays, 0, 4.0, 4.25, 0.25, 0.5, 0.5)
    require(exact["rd_true"] == 22 and exact["rd_new"] == Fraction(87, 4) and exact["E_d"] == 5 and
            exact["e_u"] == Fraction(15, 2) and exact["dtau_e_v"] == Fraction(-5, 2), "解析一列系统符号与恒等式")
    changed = dict(arrays, new_z=[math.nextafter(4.5, math.inf)], dual_residual=[21.0], direction_z=[math.nextafter(-1.0, 0.0)])
    perturbed = exact_column(changed, 0, 4.0, math.nextafter(4.25, math.inf), 0.25, 0.5, 0.5)
    require(all(perturbed[key] != 0 for key in ("rho_rhs", "transpose_rho_reconstruction", "theta_residual_rounding",
            "transpose_rho_update_z", "cost_rho_update_tau")), "各类实际舍入项不能被合并丢弃")
    require(selected_columns(1) == [0] and selected_columns(2) == [0, 1] and selected_columns(9) == [0, 4, 8], "固定坐标去重")
    verify_decimal("21.75", exact["rd_new"], 80, Fraction(22))
    rejected = 0
    for operation in (lambda: strict_json('{"x":1,"x":2}'), lambda: strict_json('{"x":NaN}'),
                      lambda: exact_scalar("0x1.0000000000000001p+0"),
                      lambda: verify_decimal("22.75", exact["rd_new"], 80, Fraction(22)),
                      lambda: verify_csc([0, 2], [0], [1.0], 1, 1),
                      lambda: legacy.flags(["x", "solve", "m", "--threads", "1", "--threads", "2"])):
        try:
            operation()
        except (AssertionError, ValueError):
            rejected += 1
        else:
            raise AssertionError("非法合成证据没有被拒绝")
    return {"passed": True, "trajectory_cases": checked, "exact_math_cases": 4, "negative_cases": rejected, "solver_runs": 0}


def verify(work):
    plan = frozen_plan()
    identity = verify_production()
    build, build_result = verify_build(work, plan, identity)
    require(read(DATA / "v53-runner-self-check.json") == {"passed": True, "checks": 6}, "测前runner合成自检")
    rows, summary = verify_runs(plan, identity, build, work)
    captures = [verify_capture(row) for row in rows if row["variant"] == "diagnostic_on"]
    offline = verify_offline(rows, captures, work, build)
    require(all(item["complete"] for item in captures), "本轮数学归因要求四份完整快照，缺失原因已分类但不能充当成立证据")
    return {"passed": True, "diagnostic_only": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "12次诊断、8对轨迹及4份缓存离线证据独立核验", "verifier_sha256": sha(__file__), "plan_sha256": PLAN_SHA,
            "parent_identity_sha256": PARENT_IDENTITY_SHA, "production_sha256": PRODUCTION_SHA,
            "diagnostic_sha256": build["executable_sha256"], "build_manifest_sha256": sha(work / "build-manifest.json"),
            "raw_sha256": sha(DATA / "v53-runs.jsonl.gz"), "summary_sha256": sha(DATA / "v53-summary.json"),
            "build": build_result, "runs": 12, "pairs": len(summary["pairs"]),
            **{key: summary[key] for key in ("non_budget_pairs", "non_budget_equal", "budget_pairs", "invalid_pairs", "different_pairs")},
            "captures": captures, "offline": offline, "self_test": self_test(),
            "limits": ["纯诊断证据，不产生新的LP覆盖或加速成绩。",
                       "逐报告仅排除timings；预算分叉单列，崩溃和异常退出不能被另一侧预算豁免。",
                       "只有保存的完整更新进入精确恒等式；末次未完成尝试只保留阶段、标量与标志，不推测未保存方向。",
                       "完整更新可能在后续assess中预算中断；冻结分析器要求assessed，本次未触发该边界，不声称其已覆盖。",
                       "128MiB是受控快照缓存上限，包含静态数据、四个动态槽和64KiB元数据，不是进程RSS上限。",
                       "仅核显式构建输入与已保存依赖；隐式编译运行库未逐项追溯。", offline["limit"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--work", type=Path, default=ROOT / ".tools/v53-diagnostic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.preflight:
        frozen_plan()
        result = {"passed": True, "scope": "只核冻结计划与合成协议，不读raw/库/快照", "self_test": self_test()}
    else:
        work = args.work.resolve()
        require(work.is_relative_to(ROOT), "独立构建目录须在工作区内")
        result = verify(work)
    destination = args.output
    if destination is None and not (args.self_test or args.preflight):
        destination = DATA / "v53-verification.json"
    if destination is not None:
        with destination.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({key: result[key] for key in ("passed", "scope", "runs", "pairs", "non_budget_equal", "budget_pairs", "self_test") if key in result}, ensure_ascii=False))

