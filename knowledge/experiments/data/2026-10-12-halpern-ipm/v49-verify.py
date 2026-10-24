"""独立核验 v49 诊断来源、轨迹与离线证据；不编译、不启动求解器。"""

import argparse
import ast
import collections
import ctypes
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import struct
import sys
import tempfile
import types
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "f3b1871230ae2062b2a4ccd1bea4bcc4ab4d158caf05f5fc28a2bd7dab9becc7"
CLARIFICATION_SHA = "71f12b6e76f1d12e4e8554f5e6310927145e5933c1f8e6b8a1fc8ffe4b59b054"
PARENT_IDENTITY_SHA = "988d97e2c29b6d3d6edb918e4bdcac0e05e51e380cf939023b71457b6e7e65bd"
PRODUCTION_SHA = "38b90b7034120b57405cfcdff5e4569cca8622a4ed2aa7a7a7eb580217e7484c"
VARIANTS = ("production_v48", "diagnostic_off", "diagnostic_on")
DTYPES = {"i64_le": (8, "q"), "f64_le": (8, "d"), "u8": (1, "B")}
sys.dont_write_bytecode = True


def hashes(mapping):
    require(isinstance(mapping, dict) and mapping, "身份映射不能为空")
    for path, expected in mapping.items():
        require(sha(path) == expected, "文件身份改变：" + path)


def path_equal(left, right):
    return Path(left).resolve() == Path(right).resolve()


def windows_arguments(command):
    """只调用系统参数解码，不启动命令或 shell。"""
    argc = ctypes.c_int()
    decode = ctypes.windll.shell32.CommandLineToArgvW
    decode.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))
    decode.restype = ctypes.POINTER(ctypes.c_wchar_p)
    pointer = decode(command, ctypes.byref(argc))
    require(bool(pointer), "Windows 命令文本解码失败")
    try:
        return [pointer[index] for index in range(argc.value)]
    finally:
        release = ctypes.windll.kernel32.LocalFree
        release.argtypes = (ctypes.c_void_p,)
        release.restype = ctypes.c_void_p
        release(pointer)


def literal_edits(path, function):
    """仅解析静态字符串，不执行被核验的变换脚本。"""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
    require(len(functions) == 1, "变换列表函数必须唯一")
    returned = [node for node in functions[0].body if isinstance(node, ast.Return)]
    require(len(returned) == 1, "变换列表必须是单个字面量返回")
    return ast.literal_eval(returned[0].value)


def verify_transform(unit, record, frozen, work):
    filename = "ipm.cpp" if unit == "ipm" else "normal_cholesky.cpp"
    original = frozen / "src" / filename
    patched = work / (unit + "-diagnostic.cpp")
    require(path_equal(record["source"], original) and path_equal(record["patched"], patched), "插桩路径越界")
    require(sha(original) == record["source_sha256"] and sha(patched) == record["patched_sha256"], "插桩来源身份")
    source = original.read_text(encoding="utf-8")
    transformed = source
    replacements, anchors = [], []
    if unit == "normal":
        for name, before, after in literal_edits(DATA / "v49-normal-instrumentation.py", "_edits"):
            replacements.append((before, after))
            anchors.append({"name": name, "before": before, "after": after, "count": 1})
    else:
        entries = literal_edits(DATA / "v49-hsd-instrumentation.py", "changes")
        for index, (anchor, before, after) in enumerate(entries):
            replacement = before + anchor + after
            if index == len(entries) - 1:
                replacement = ("  return result;\n  } catch (const NumericalFailure& v49_failure) {\n"
                               "    v49_observation.failure(v49_failure.what());\n    throw;\n  }\n}\n\n"
                               "double detail::ipm_homogeneous_step_limit(")
            replacements.append((anchor, replacement))
            require(source.count(anchor) == 1, "HSD 原始锚点不唯一")
            anchors.append({"index": index, "source_line": source[:source.index(anchor)].count("\n") + 1,
                            "anchor_sha256": hashlib.sha256(anchor.encode()).hexdigest(),
                            "replacement_sha256": hashlib.sha256(replacement.encode()).hexdigest()})
    for before, after in replacements:
        require(transformed.count(before) == 1, "正向插桩锚点必须唯一")
        transformed = transformed.replace(before, after, 1)
    require(transformed == patched.read_text(encoding="utf-8") and record["anchors"] == anchors,
            "副本存在未记录改动或锚点记录不一致")
    for before, after in reversed(replacements):
        require(transformed.count(after) == 1, "逆向插桩锚点必须唯一")
        transformed = transformed.replace(after, before, 1)
    require(transformed == source, "逆变换没有恢复冻结源码")
    return len(replacements)


def verify_build(work, plan):
    identity_path = DATA / "ipm-v48-identity.json"
    require(sha(identity_path) == PARENT_IDENTITY_SHA, "父身份文件")
    identity = read(identity_path)
    archive = DATA / "ipm-v48-source.zip"
    require(sha(archive) == identity["source_zip_sha256"], "父来源包")
    require(len(identity["files"]) == 92 and len(identity["libraries"]) == 17, "父源码和库数量")
    hashes({str(ROOT / key): value for key, value in identity["files"].items()})
    hashes({str(ROOT / key): value for key, value in identity["libraries"].items()})
    hashes({identity["compiler"]: identity["compiler_sha256"]})
    with zipfile.ZipFile(archive) as bundle:
        names = [item.filename for item in bundle.infolist() if not item.is_dir()]
        require(len(names) == len(set(names)) and set(names) == set(identity["files"]), "来源包内容集合")
        for name in names:
            require(hashlib.sha256(bundle.read(name)).hexdigest() == identity["files"][name], "来源包成员身份：" + name)
    build = read(work / "build-manifest.json")
    require(build["diagnostic_only"] is True and build["implicit_link_inputs_verified"] is False and
            build["plan_sha256"] == PLAN_SHA and build["clarification_sha256"] == CLARIFICATION_SHA and
            build["frozen_files"] == identity["files"] and build["production_sha256"] == PRODUCTION_SHA,
            "构建清单与来源合同")
    for key in ("protected_before_and_after", "link_explicit_before_and_after", "compile_inputs_after_and_final", "outputs"):
        hashes(build[key])
    for key, expected in (("production_executable", PRODUCTION_SHA), ("executable", build["executable_sha256"]),
                          ("replay_executable", build["replay_executable_sha256"])):
        require(sha(build[key]) == expected, "构建可执行文件身份：" + key)
    frozen = work / "frozen"
    hashes({str(frozen / key): value for key, value in identity["files"].items()})
    transforms = read(work / "transforms.json")
    require(set(transforms) == {"ipm", "normal"}, "只能替换两个诊断副本")
    counts = {unit: verify_transform(unit, transforms[unit], frozen, work) for unit in transforms}
    command = read(work / "commands.json")
    require(command == build["commands"] and path_equal(command["cwd"], ROOT / "build/release"), "构建命令清单")
    cwd = Path(command["cwd"])
    lines = (work / "ninja-commands.txt").read_text(encoding="utf-8").splitlines()
    require([line for line in lines if " -o solver.exe " in line] == [identity["link_command"]], "冻结链接文本")
    outer = windows_arguments(identity["link_command"])
    require(outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "链接包装")
    original_link = windows_arguments(outer[2][len("cd . && "):-len(" && cd .")])
    require(original_link == command["original_link"], "原链接参数解码")
    for unit, filename in (("ipm", "ipm.cpp"), ("normal", "normal_cholesky.cpp")):
        matches = [windows_arguments(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/src/" + filename)]
        require(matches == [command["original_compile"][unit]], "原编译参数")
    def expected_compile(original, source, obj):
        result = list(original)
        require(path_equal(result[0], identity["compiler"]), "编译器参数身份")
        for option, value in (("-c", source), ("-o", obj), ("-MF", str(obj) + ".d"), ("-MT", obj)):
            require(result.count(option) == 1, "编译选项必须唯一：" + option)
            result[result.index(option) + 1] = str(value)
        includes = [index for index, value in enumerate(result) if value.startswith("-I") and path_equal(value[2:], ROOT / "include")]
        require(len(includes) == 1, "项目 include 路径必须唯一")
        result[includes[0]] = "-I" + str(frozen / "include")
        return result + ["-I" + str(DATA)]
    for unit in ("ipm", "normal"):
        require(command["compile"][unit] == expected_compile(command["original_compile"][unit],
                work / (unit + "-diagnostic.cpp"), work / (unit + "-diagnostic.obj")), "诊断编译参数超出允许替换")
    require(command["replay_compile"] == expected_compile(command["original_compile"]["normal"],
            DATA / "v49-factor-replay.cpp", work / "factor-replay.obj"), "回放编译参数")
    diag, replay = [], []
    replaced = collections.Counter()
    for value in original_link:
        if value.startswith("-Wl,--out-implib,"):
            continue
        if value == "libsolver_ipm.a":
            diag += [str(work / "ipm-diagnostic.obj"), str(cwd / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj")]
            replaced[value] += 1
        elif value == "libsolver_normal_cholesky.a":
            diag.append(str(work / "normal-diagnostic.obj"))
            replaced[value] += 1
        else:
            diag.append(value)
        if value == "-municode":
            continue
        if value.endswith("CMakeFiles/solver.dir/src/main.cpp.obj"):
            replay.append(str(work / "factor-replay.obj"))
            replaced["main"] += 1
        else:
            replay.append(value)
    require(replaced == {"libsolver_ipm.a": 1, "libsolver_normal_cholesky.a": 1, "main": 1}, "链接替换次数")
    diag[diag.index("-o") + 1] = str(work / "solver-diagnostic.exe")
    replay[replay.index("-o") + 1] = str(work / "factor-replay.exe")
    require(diag == command["link"] and replay == command["replay_link"], "链接参数超出允许替换")
    jobs = {"ipm-compile": command["compile"]["ipm"], "normal-compile": command["compile"]["normal"],
            "replay-compile": command["replay_compile"], "link": diag, "replay-link": replay,
            "hsd-self-build": build["self_test_compile"]}
    for name, expected in jobs.items():
        record = read(work / (name + ".json"))
        require(record["command"] == expected and record["returncode"] == 0 and not record["wall_timeout"], "编译日志失败或命令不一致：" + name)
    for name in ("hsd-self-off-run", "hsd-self-on-run", "factor-self-test", "hsd-math-self-test"):
        record = read(work / (name + ".json"))
        require(record["returncode"] == 0 and not record["wall_timeout"], "已有合成自测未成功：" + name)
        require(strict_json(record["stdout"])["passed"] is True, "已有合成自测报告：" + name)
    require(not list(work.glob("hsd-self-off.*")), "关闭诊断合成自测不应输出快照")
    return build, {"sources": 92, "libraries": 17, "reversible_anchors": counts,
                   "explicit_compile_inputs": len(build["compile_inputs_after_and_final"]),
                   "implicit_link_inputs_verified": False}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "JSON 重复字段：" + key)
        result[key] = value
    return result


def strict_json(text):
    def reject(value):
        raise AssertionError("JSON 非有限字面量：" + value)
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def integer(value, label, maximum=2**63 - 1):
    require(type(value) is int and 0 <= value <= maximum, "非负整数无效：" + label)
    return value


def expected_runs(plan):
    expected = [(model["name"], backend, variant) for model in plan["models"]
                for backend in model["backends"] for variant in VARIANTS]
    require(len(expected) == len(set(expected)) == plan["runs"] == 21,
            "冻结运行集合必须是 7 个模型/后端组合各 3 个程序模式")
    return expected


def frozen_plan():
    require(sha(DATA / "v49-plan.json") == PLAN_SHA, "冻结计划身份")
    require(sha(DATA / "v49-diagnostic-clarification.json") == CLARIFICATION_SHA, "测前澄清身份")
    plan = read(DATA / "v49-plan.json")
    clarification = read(DATA / "v49-diagnostic-clarification.json")
    require(plan["kind"] == "diagnostic_only" and plan["production_changes"] is False and
            clarification["production_changes"] is False and
            clarification["diagnostic_solver_runs_before_clarification"] == 0 and
            clarification["plan_sha256"] == PLAN_SHA, "只读诊断及测前澄清合同")
    require(plan["parent_identity_sha256"] == PARENT_IDENTITY_SHA and
            plan["parent_solver_sha256"] == PRODUCTION_SHA, "生产来源锚点")
    require(plan["variants"] == list(VARIANTS) and
            (plan["repeats"], plan["time_limit"], plan["wall_timeout"], plan["max_iterations"],
             plan["scaling_passes"], plan["threads"], plan["seed"]) == (1, 5, 35, 200, 5, 1, 0),
            "诊断预算与模式")
    require((plan["method"], plan["linear_accuracy"]) == ("homogeneous", "strict"), "齐次严格模式")
    require(plan["shared_ipm"] == {
        "singleton-elimination": "on", "dynamic-regularization": "off", "kkt-equilibration": "on",
        "fixed-variable-elimination": "on", "singleton-presolve": "off", "presolve-fixed-elimination": "off",
        "krylov-history-size": "0", "direction-retries": "0", "central-corrections": "0", "inexact-prefilter": "off"},
        "共同 IPM 参数")
    require([(item["name"], item["backends"]) for item in plan["models"]] == [
        ("dfl001", ["normal_augmented", "normal_cholmod"]),
        ("forplan", ["normal_augmented", "normal_cholmod"]),
        ("greenbea", ["normal_augmented", "normal_cholmod"]), ("afiro", ["normal_cholmod"])], "固定输入组")
    expected_runs(plan)
    return plan, clarification


def non_timing_report(report):
    require(isinstance(report, dict), "完整报告必须是对象")
    return {key: value for key, value in report.items() if key not in ("timings", "backend_version")}


def flags(command):
    require(len(command) >= 3 and command[1] == "solve" and (len(command) - 3) % 2 == 0, "求解命令结构")
    result = {}
    for key, value in zip(command[3::2], command[4::2]):
        require(key.startswith("--") and key not in result and type(value) is str, "命令重复或无效选项")
        result[key] = value
    return result


def frozen_quality_modules():
    """只复用冻结的报告协议函数；排程、命令、配对和恒等式由本文件重算。"""
    modules = {}
    for name in ("benchmark", "compare_lp"):
        path = ROOT / "scripts" / (name + ".py")
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules["benchmark"], modules["compare_lp"]


def verify_runs(plan, build, work):
    manifest = read(DATA / "v49-run-manifest.json")
    require(manifest["plan_sha256"] == PLAN_SHA and manifest["clarification_sha256"] == CLARIFICATION_SHA and
            path_equal(manifest["build_manifest"], work / "build-manifest.json") and path_equal(manifest["work"], work),
            "运行清单来源")
    require(manifest["environment"] == {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "运行线程环境")
    hashes(manifest["protected"])
    protected = {Path(path).resolve() for path in manifest["protected"]}
    require({DATA / "v49-run.py", work / "build-manifest.json", Path(build["production_executable"]), Path(build["executable"]),
             ROOT / "scripts/benchmark.py", ROOT / "scripts/compare_lp.py"} <= protected, "运行清单缺少必需身份")
    capture_root = work / "captures"
    require(path_equal(manifest["capture_root"], capture_root), "快照目录")
    benchmark, compare = frozen_quality_modules()
    with gzip.open(DATA / "v49-runs.jsonl.gz", "rt", encoding="utf-8") as source:
        rows = [strict_json(line) for line in source]
    cases = [(model, backend) for model in plan["models"] for backend in model["backends"]]
    schedule = [(index, model, backend, variant) for index, (model, backend) in enumerate(cases)
                for variant in VARIANTS[index % 3:] + VARIANTS[:index % 3]]
    require(len(rows) == len(schedule) == 21, "原始运行数")
    for index, (row, (case_index, model, backend, variant)) in enumerate(zip(rows, schedule)):
        require((row["index"], row["case_index"], row["model"], row["input"], row["linear_solver"], row["variant"]) ==
                (index, case_index, model["name"], model["input"], backend, variant), "运行顺序、输入或模式不符")
        executable = build["production_executable"] if variant == "production_v48" else build["executable"]
        expected_solver = PRODUCTION_SHA if variant == "production_v48" else build["executable_sha256"]
        expected_options = {"--backend": "ipm", "--time-limit": "5", "--threads": "1", "--seed": "0",
            "--iteration-limit": "200", "--scaling-passes": "5", "--ipm-method": "homogeneous",
            "--ipm-linear-accuracy": "strict", "--ipm-linear-solver": backend,
            **{"--ipm-" + key: value for key, value in plan["shared_ipm"].items()}}
        require(path_equal(row["command"][0], executable) and row["command"][2] == model["input"] and
                flags(row["command"]) == expected_options, "运行命令偏离冻结配置")
        require(row["hashes_before"] == row["hashes_after"] == {"input": model["sha256"], "solver": expected_solver} and
                row["model_sha256"] == model["sha256"] and sha(model["input"]) == model["sha256"], "运行前后输入或程序身份")
        require(row["error"] in (None, "wall_timeout"), "启动或报告无效，不能判为轨迹通过")
        require(type(row["elapsed_wall_seconds"]) in (int, float) and math.isfinite(row["elapsed_wall_seconds"]) and
                row["elapsed_wall_seconds"] > 0, "墙钟记录")
        report = row["report"]
        if report is not None:
            require(strict_json(row["stdout"]) == report, "stdout 与完整报告不符")
            require(report["backend"] == "ipm", "报告后端")
            ipm = report["options"]["ipm"]
            require(ipm == {"method": "homogeneous", "max_iterations": 200, "scaling_passes": 5,
                "eliminate_singletons": True, "dynamic_regularization": False, "equilibrate_kkt": True,
                "linear_accuracy": "strict", "central_corrections": 0, "direction_retries": 0,
                "krylov_history_size": 0, "linear_solver": backend, "eliminate_fixed_variables": True,
                "presolve_singleton_rows": False, "eliminate_presolved_fixed_variables": False,
                "inexact_prefilter": False}, "实际报告 IPM 配置")
        else:
            require(row["error"] == "wall_timeout", "正常返回必须有完整报告")
        classification = benchmark.classify(row["returncode"], report) if report is not None else row["error"]
        require(row["classification"] == classification, "报告分类重算")
        numerical = compare.numerical_seconds(row)
        failure = compare.quality_failure(row, model, model["sha256"], expected_solver, expected_solver)
        if failure is None and (numerical > 5 or row["elapsed_wall_seconds"] > 35):
            failure = "budget_exceeded"
        require(row["numerical_seconds"] == numerical and row["quality_failure"] == failure and
                row["quality_pass"] is (failure is None), "数值计时和质量重新分类")
        prefix = capture_root / (str(case_index) + "-" + variant)
        require(path_equal(row["capture_prefix"], prefix), "快照前缀")
        observed = {str(path): {"sha256": sha(path), "bytes": path.stat().st_size}
                    for path in sorted(capture_root.glob(prefix.name + ".*"))}
        require(observed == row["capture_files"], "快照文件集合、身份及长度")
        if variant != "diagnostic_on":
            require(not observed and "V49_NORMAL" not in row["stderr"] and "V49_HSD" not in row["stderr"], "关闭开关必须无诊断产物")
        elif row["error"] != "wall_timeout":
            require(str(prefix) + ".normal.json" in observed and str(prefix) + ".hsd.json" in observed,
                    "开启快照元数据缺失")
    summary = read(DATA / "v49-summary.json")
    pairs, case_reports = [], []
    def limited(row):
        return row["error"] == "wall_timeout" or (row["report"] or {}).get("status") == "time_limit" or \
            row["elapsed_wall_seconds"] > 35 or row["numerical_seconds"] is not None and row["numerical_seconds"] > 5
    for index, (model, backend) in enumerate(cases):
        selected = {row["variant"]: row for row in rows if row["case_index"] == index}
        require(set(selected) == set(VARIANTS), "每配置必须三模式且无重复")
        for left, right in zip(VARIANTS, VARIANTS[1:]):
            a, b = selected[left], selected[right]
            ar = non_timing_report(a["report"]) if a["report"] is not None else None
            br = non_timing_report(b["report"]) if b["report"] is not None else None
            equal = ar == br and (a["returncode"], a["error"]) == (b["returncode"], b["error"])
            budget = bool(limited(a) or limited(b))
            pairs.append({"case_index": index, "model": model["name"], "linear_solver": backend,
                          "left": left, "right": right, "equal": equal, "budget_limited": budget})
            require(equal or budget, "非预算完整轨迹不一致")
        case_reports.append({"case_index": index, "model": model["name"], "linear_solver": backend,
            "status": {key: (row["report"] or {}).get("status") for key, row in selected.items()},
            "capture_files": selected["diagnostic_on"]["capture_files"], "capture_prefix": selected["diagnostic_on"]["capture_prefix"]})
    expected_summary = {"diagnostic_only": True, "runs": 21, "pairs": pairs, "cases": case_reports,
        "non_budget_pairs": sum(not pair["budget_limited"] for pair in pairs),
        "non_budget_equal": sum(not pair["budget_limited"] and pair["equal"] for pair in pairs),
        "budget_pairs": sum(pair["budget_limited"] for pair in pairs), "raw_sha256": sha(DATA / "v49-runs.jsonl.gz")}
    require({key: summary[key] for key in expected_summary} == expected_summary, "14 对轨迹与预算分列汇总不符")
    return rows, summary


def exact_hex(text):
    """不经 binary64 中转，直接把 C++ hexfloat 解码为精确有理数。"""
    require(type(text) is str, "十六进制标量必须是字符串")
    matched = re.fullmatch(r"([+-]?)0x([0-9a-fA-F]+)(?:\.([0-9a-fA-F]*))?[pP]([+-]?\d+)", text)
    require(matched is not None, "无效或非有限 hexfloat：" + text)
    sign, whole, fractional, exponent = matched.groups()
    fractional = fractional or ""
    result = Fraction(int(whole + fractional, 16), 16 ** len(fractional))
    power = int(exponent)
    require(abs(power) <= 20000, "hexfloat 指数超出本轮浮点格式范围")
    result = result * 2**power if power >= 0 else result / 2**(-power)
    return -result if sign == "-" else result


class BinaryArrays:
    """通过只读映射访问紧凑数组；不构造逐非零项字典。"""
    def __init__(self, path, descriptors, magic, max_bytes=128 * 1024**2):
        self.path = Path(path)
        self.file = None
        self.mapping = None
        self.views = {}
        try:
            require(sys.byteorder == "little", "当前核验器仅接受小端平台")
            size = self.path.stat().st_size
            require(len(magic) <= size <= max_bytes, "快照字节数不合法或超过分组件上限")
            self.file = self.path.open("rb")
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            require(self.mapping[:len(magic)] == magic, "快照魔数")
            position = len(magic)
            for item in descriptors:
                name, dtype = item["name"], item["dtype"]
                require(type(name) is str and name not in self.views and dtype in DTYPES, "重复数组名或未知类型")
                count = integer(item["count"], name + ".count")
                offset = integer(item["offset"], name + ".offset")
                width, code = DTYPES[dtype]
                require(offset == position and count <= (size - position) // width, "数组布局必须连续且无越界")
                position += width * count
                self.views[name] = memoryview(self.mapping)[offset:position].cast(code)
            require(position == size, "快照存在未声明尾部或不完整数据")
        except BaseException:
            self.close()
            raise

    def close(self):
        for view in self.views.values():
            view.release()
        self.views.clear()
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self.views

    def __exit__(self, *args):
        self.close()


def fraction_hsd(arrays, tau, kappa, primal_regularization):
    """从稀疏乘积独立形成实际 H 对应的 K0 残差和能量分解。"""
    n, m = len(arrays["costs"]), len(arrays["rhs"])
    require(len(arrays["starts"]) == n + 1 and len(arrays["v"]) == len(arrays["u"]) == n + m and
            len(arrays["diagonal_h"]) == len(arrays["regularized_h"]) == m, "HSD 数组维数")
    starts, rows, values = arrays["starts"], arrays["rows"], arrays["values"]
    require(starts[0] == 0 and starts[-1] == len(rows) == len(values), "HSD CSC 首尾指针")
    require(all(0 <= starts[col] <= starts[col + 1] <= len(values) for col in range(n)), "HSD CSC 列指针")
    require(tau > 0 and kappa > 0 and primal_regularization >= 0, "HSD 正标量和非负原始正则")
    v = [Fraction.from_float(value) for value in arrays["v"]]
    ed = [Fraction.from_float(value) for value in arrays["costs"]]
    ep = [-Fraction.from_float(value) for value in arrays["rhs"]]
    energy = Fraction(0)
    dual_regularization_energy = Fraction(0)
    dot_v = Fraction(0)
    dot_u = Fraction(0)
    for col in range(n):
        coefficient = Fraction.from_float(arrays["costs"][col])
        dot_v += coefficient * v[col]
        dot_u += coefficient * Fraction.from_float(arrays["u"][col])
        for index in range(starts[col], starts[col + 1]):
            row = rows[index]
            require(0 <= row < m, "HSD CSC 行越界")
            coefficient = Fraction.from_float(values[index])
            ed[col] += coefficient * v[n + row]
            ep[row] += coefficient * v[col]
    for row in range(m):
        vz = v[n + row]
        h = Fraction.from_float(arrays["diagonal_h"][row])
        regularized_h = Fraction.from_float(arrays["regularized_h"][row])
        require(h >= 0 and regularized_h >= h, "实际 H 及正则 H 顺序")
        ep[row] -= h * vz
        energy += h * vz * vz
        dual_regularization_energy += (regularized_h - h) * vz * vz
        coefficient = Fraction.from_float(arrays["rhs"][row])
        dot_v += coefficient * vz
        dot_u += coefficient * Fraction.from_float(arrays["u"][n + row])
    chi = sum((v[col] * ed[col] for col in range(n)), Fraction(0))
    chi -= sum((v[n + row] * ep[row] for row in range(m)), Fraction(0))
    # 正则残差单独代回，不直接把无正则 χ 加上能量当作独立验证。
    chi_regularized = sum((v[col] * (ed[col] + primal_regularization * v[col])
                           for col in range(n)), Fraction(0))
    chi_regularized -= sum((v[n + row] * (ep[row] -
        (Fraction.from_float(arrays["regularized_h"][row]) - Fraction.from_float(arrays["diagonal_h"][row])) * v[n + row])
        for row in range(m)), Fraction(0))
    primal_energy = primal_regularization * sum((value * value for value in v[:n]), Fraction(0))
    q = kappa / tau
    denominator = q - dot_v
    require(denominator == q + energy - chi, "无正则 K0 能量恒等式")
    require(denominator == q + energy + primal_energy + dual_regularization_energy - chi_regularized,
            "实际正则 KKT 能量恒等式")
    return {"dot_u": dot_u, "dot_v": dot_v, "kappa_over_tau": q, "denominator": denominator,
            "energy": energy, "chi": chi, "chi_regularized": chi_regularized,
            "primal_regularization_energy": primal_energy, "dual_regularization_energy": dual_regularization_energy,
            "dual_residual_inf": max(map(abs, ed), default=Fraction(0)),
            "primal_residual_inf": max(map(abs, ep), default=Fraction(0))}


def verify_normal(path, expected_backend):
    meta = read(path)
    require(meta["schema"] == 1 and meta["component"] == "normal" and meta["byte_order"] == "little" and
            meta["cache_limit_bytes"] == 128 * 1024**2, "normal 元数据合同")
    for name in ("snapshot_attempted", "backend_selected", "captured", "snapshot_oversize", "snapshot_invalid",
                 "diagnostic_failed", "event_limit_reached", "write_failed", "binary_written"):
        require(type(meta[name]) is bool, "normal 布尔值：" + name)
    for name in ("event_storage_bytes", "snapshot_bytes", "n", "nnz", "snapshot_attempt", "attempts", "numeric_calls"):
        integer(meta[name], "normal." + name)
    require(meta["event_storage_bytes"] + meta["snapshot_bytes"] <= meta["cache_limit_bytes"], "normal 缓存额度")
    require(len(meta["events"]) <= 2048 and meta["numeric_calls"] <= meta["attempts"], "normal 事件/调用计数")
    if expected_backend == "normal_augmented":
        require(not meta["backend_selected"] and not meta["events"] and meta["attempts"] == meta["numeric_calls"] == 0,
                "非 CHOLMOD 路线仅记录空组件元数据")
    failures = []
    previous_attempt = previous_numeric = 0
    for event in meta["events"]:
        require(event["kind"] in ("cholmod_failure", "attempt_exit") and
                event["phase"] in ("input", "gram_accumulation", "double_conversion", "cholmod_numeric", "complete") and
                event["outcome"] in ("success", "false", "unwound"), "normal 事件类型或阶段")
        attempt, numeric = integer(event["attempt"], "event.attempt"), integer(event["numeric_calls"], "event.numeric_calls")
        require(previous_attempt <= attempt <= meta["attempts"] and previous_numeric <= numeric <= meta["numeric_calls"], "normal 事件顺序")
        previous_attempt, previous_numeric = attempt, numeric
        if event["kind"] == "cholmod_failure":
            require(event["numeric_started"] is True and event["phase"] == "cholmod_numeric" and event["outcome"] == "false",
                    "CHOLMOD 失败必须在数值调用后记录")
            failures.append(event)
    binary = Path(path).with_suffix(".bin")
    if not meta["captured"]:
        require(not meta["binary_written"] and not binary.exists(), "未捕获不应出现二进制")
        return {"captured": False, "attempts": meta["attempts"], "numeric_calls": meta["numeric_calls"]}, meta
    require(meta["snapshot_attempted"] and failures and meta["snapshot_attempt"] == failures[0]["attempt"], "必须保存首份数值失败")
    require(meta["binary_written"] and not meta["write_failed"] and binary.is_file(), "已捕获快照写盘失败")
    with binary.open("rb") as source:
        header = source.read(32)
    require(header[:8] == b"V49GRAM1", "Gram 魔数")
    n, nnz, attempt = struct.unpack("<QQQ", header[8:])
    require((n, nnz, attempt) == (meta["n"], meta["nnz"], meta["snapshot_attempt"]) and n > 0 and
            binary.stat().st_size == meta["snapshot_bytes"] == 40 + 8*n + 16*nnz, "Gram 尺寸或身份")
    descriptors = [{"name": "starts", "dtype": "i64_le", "count": n+1, "offset": 32},
        {"name": "rows", "dtype": "i64_le", "count": nnz, "offset": 40+8*n},
        {"name": "values", "dtype": "f64_le", "count": nnz, "offset": 40+8*n+8*nnz}]
    with BinaryArrays(binary, descriptors, header) as arrays:
        starts, rows, values = arrays["starts"], arrays["rows"], arrays["values"]
        require(starts[0] == 0 and starts[-1] == nnz, "Gram CSC 首尾")
        sums, diagonal = [Fraction(0) for _ in range(n)], [None] * n
        for col in range(n):
            require(0 <= starts[col] <= starts[col+1] <= nnz, "Gram CSC 列指针")
            previous = -1
            for index in range(starts[col], starts[col+1]):
                row = rows[index]
                require(previous < row <= col and math.isfinite(values[index]), "Gram 必须是有序有限上三角")
                previous = row
                value = Fraction.from_float(values[index])
                sums[row] += abs(value)
                if row != col:
                    sums[col] += abs(value)
                else:
                    diagonal[col] = value
            require(diagonal[col] is not None, "Gram 缺少显式对角")
        metrics = {"inf_norm": max(sums), "diagonal_min": min(diagonal), "diagonal_max": max(diagonal),
                   "gershgorin_lower": min(diagonal[index] - sums[index] + abs(diagonal[index]) for index in range(n))}
    return {"captured": True, "n": n, "nnz": nnz, "attempt": attempt, "matrix_sha256": sha(binary),
            "failure": failures[0], "metrics": metrics}, meta


def sign(value):
    return (value > 0) - (value < 0)


def verify_fraction_value(record, exact, label):
    require(record["numerator"] == str(exact.numerator) and record["denominator"] == str(exact.denominator) and
            record["sign"] == sign(exact), "Fraction 独立重算不符：" + label)
    with localcontext() as context:
        context.prec = 80
        require(Decimal(record["decimal80"]) == Decimal(exact.numerator) / Decimal(exact.denominator), "Fraction 十进制表示：" + label)


def verify_hsd(path, analysis):
    meta = read(path)
    require(meta["schema"] == "v49_hsd_1" and meta["enabled"] is True and
            meta["component_cache_limit"] == 128 * 1024**2 and meta["metadata_reserve"] == 65536 and
            meta["extra_lp_products"] == meta["extra_lp_solves"] == meta["event_count"] == 0, "HSD 捕获合同")
    require(analysis["metadata_sha256"] == sha(path) and path_equal(analysis["metadata_path"], path) and
            analysis["captured"] is meta["captured"] and analysis["capture_status"] == meta["status"], "HSD 分析来源")
    if not meta["captured"]:
        require(analysis["analysis_status"] == "not_captured", "未捕获 HSD 不能声称完成分析")
        return {"captured": False, "status": meta["status"]}
    require(meta["attempted"] is True and meta["status"] == "captured" and analysis["analysis_status"] == "analyzed", "HSD 捕获/分析状态")
    names = ("starts", "rows", "values", "costs", "rhs", "s", "z", "diagonal_h", "regularized_h", "u", "v", "inequality")
    require(tuple(item["name"] for item in meta["arrays"]) == names, "HSD 数组顺序")
    for item in meta["arrays"]:
        expected = "i64_le" if item["name"] in ("starts", "rows") else "u8" if item["name"] == "inequality" else "f64_le"
        require(item["dtype"] == expected, "HSD 数组数据类型")
    binary = Path(path).with_suffix(".bin")
    require(binary.stat().st_size == 8 + meta["payload_bytes"] and
            meta["payload_bytes"] + 65536 <= 128*1024**2 and analysis["binary_sha256"] == sha(binary), "HSD 二进制身份/缓存")
    event_keys = ("phase", "iteration", "completed_iterations", "factorizations", "refinements", "constant_solves",
                  "direction_solves", "regularization_level", "failure")
    require(analysis["event"] == {key: meta[key] for key in event_keys}, "HSD 首次失败事件")
    scalars = meta["scalars"]
    with BinaryArrays(binary, meta["arrays"], b"V49HSD1\0") as arrays:
        n, m = len(arrays["costs"]), len(arrays["rhs"])
        require(analysis["dimensions"] == {"n": n, "m": m, "nnz": len(arrays["values"])}, "HSD 维数")
        require(all(len(arrays[key]) == m for key in ("s", "z", "inequality")) and
                all(value in (0, 1) for value in arrays["inequality"]), "HSD 点/掩码维数")
        for name in names:
            if name not in ("starts", "rows", "inequality"):
                require(all(math.isfinite(value) for value in arrays[name]), "非有限快照不能声称有限精确重算")
        exact = fraction_hsd(arrays, exact_hex(scalars["tau"]), exact_hex(scalars["kappa"]), exact_hex(scalars["primal_regularization"]))
    mapped = {"q": exact["kappa_over_tau"], "dot_v": exact["dot_v"], "dot_u": exact["dot_u"],
        "denominator": exact["denominator"], "energy": exact["energy"], "positive_energy": exact["kappa_over_tau"]+exact["energy"],
        "defect": exact["chi"], "regularized_defect": exact["chi_regularized"],
        "regularization_energy": exact["primal_regularization_energy"]+exact["dual_regularization_energy"],
        "identity_error": Fraction(0), "regularized_identity_error": Fraction(0),
        "residual_inf": max(exact["dual_residual_inf"], exact["primal_residual_inf"])}
    values = analysis["exact"]["scalars"]
    require(analysis["exact"]["identity_exact"] is True and analysis["exact"]["regularized_identity_exact"] is True and
            analysis["exact"]["nonnegative_h_and_regularization"] is True, "HSD 恒等式和实际 H 正则条件")
    finite_production = meta["scalar_values_available"] and all(
        re.fullmatch(r"[+-]?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?\d+", scalars[key]) for key in ("dot_v", "denominator"))
    if finite_production:
        pd, pv = exact_hex(scalars["denominator"]), exact_hex(scalars["dot_v"])
        mapped.update(production_dot_v=pv, production_denominator=pd, dot_rounding=pv-exact["dot_v"],
            ratio_subtraction_rounding=pd-exact["kappa_over_tau"]+pv, total_scalar_rounding=pd-exact["denominator"])
    else:
        require("production_denominator" not in values, "非有限或未完成标量不得声称有限生产分母")
    for label, value in mapped.items():
        verify_fraction_value(values[label], value, label)
    require([entry["precision"] for entry in analysis["decimal"]] == [80, 120], "两档 Decimal 精度")
    for entry in analysis["decimal"]:
        with localcontext() as context:
            context.prec = 160
            scale = max(Fraction(1), abs(exact["dot_v"]), abs(exact["chi"]), abs(exact["energy"]), abs(exact["kappa_over_tau"]))
            tolerance = Decimal(scale.numerator)/Decimal(scale.denominator) * Decimal(10)**(-entry["precision"]+12)
            for label, key in (("dot_v", "dot_v"), ("denominator", "denominator"), ("energy", "energy"), ("defect", "chi"), ("energy_denominator", "denominator")):
                target = Decimal(exact[key].numerator)/Decimal(exact[key].denominator)
                require(abs(Decimal(entry[label])-target) <= tolerance, "Decimal 与精确值不一致：" + label)
            require(entry["sign"] == sign(Decimal(entry["denominator"])) and abs(Decimal(entry["identity_discrepancy"])) <= 2*tolerance,
                    "Decimal 符号/恒等式近似误差")
    signs_match = all(entry["sign"] == sign(exact["denominator"]) for entry in analysis["decimal"])
    require(analysis["decimal_signs_match_exact"] is signs_match, "Decimal 符号交叉检查记录")
    classification = ("production_scalar_unavailable_or_nonfinite" if not finite_production else
        "scalar_rounding_changes_sign_for_saved_direction" if exact["denominator"] > 0 and pd <= 0 else
        "saved_direction_has_nonpositive_exact_denominator" if exact["denominator"] <= 0 else
        "positive_exact_denominator_other_scalar_failure")
    require(analysis["classification"] == classification, "HSD 分母归因分类")
    return {"captured": True, "classification": classification, "exact_denominator_sign": sign(exact["denominator"]),
            "decimal_signs_match_exact": signs_match, "exact_metrics_checked": len(mapped)}


def decimal_metric(item):
    require(type(item["valid"]) is bool, "LD 指标有效性必须是布尔值")
    if not item["valid"]:
        require(item["decimal"] is None, "无效 LD 指标必须为 null")
        return None
    require(type(item["decimal"]) is str, "LD 指标须保存十进制文本")
    value = Decimal(item["decimal"])
    require(value.is_finite(), "有效 LD 指标不得是 NaN 或无穷")
    return Fraction(value)


def verify_replay(report, normal):
    require(report["schema"] == 1 and (report["n"], report["nnz"], report["snapshot_attempt"]) ==
            (normal["n"], normal["nnz"], normal["attempt"]), "回放矩阵维数和失败代号")
    precision = integer(report["long_double_digits"], "LD digits")
    require(53 <= precision <= 113, "已支持 LD 位数范围")
    epsilon = Fraction(1, 2**(precision-1))
    scale = max(Fraction(1), abs(normal["metrics"]["inf_norm"]))
    matrix_tolerance = scale * (2*normal["nnz"]+8) * epsilon
    for key, expected in normal["metrics"].items():
        value = decimal_metric(report["matrix_metrics"][key])
        require(value is not None and abs(value-expected) <= matrix_tolerance, "Gram 独立矩阵指标：" + key)
    require([item["name"] for item in report["backends"]] == ["cholmod_supernodal", "eigen_double", "eigen_long_double"], "三个固定回放内核")
    summary = []
    for backend in report["backends"]:
        for key in ("success", "finite_factor", "permutation_identity"):
            require(type(backend[key]) is bool, "回放布尔字段：" + key)
        for key in ("stored_entries", "nsuper", "factor_started", "solve_started", "residual_products"):
            integer(backend[key], "replay." + key)
        require(backend["factor_started"] <= 1 and backend["solve_started"] <= 2 and len(backend["solves"]) <= 2,
                "回放只允许一次分解与两个固定 RHS")
        if backend["success"]:
            minimum, maximum = (decimal_metric(backend[key]) for key in ("factor_diagonal_min", "factor_diagonal_max"))
            require(backend["finite_factor"] and backend["permutation_identity"] and backend["factor_started"] == 1 and
                    minimum is not None and maximum is not None and 0 < minimum <= maximum, "成功因子须为有限正对角且不重排")
            if backend["error"] is None:
                require(backend["solve_started"] == len(backend["solves"]) == 2, "成功因子须回放两组 RHS")
        rhs_names = [item["rhs"] for item in backend["solves"]]
        require(rhs_names == ["ones", "alternating"][:len(rhs_names)], "回放 RHS 顺序")
        required_products = 0
        for solve in backend["solves"]:
            require(solve["success"] is solve["physical"]["finite"], "solve.success 仅表示真实残差有限")
            residuals = [solve["physical"]]
            if backend["name"] == "eigen_long_double":
                require(isinstance(solve["cast_double"], dict), "LD 必须记录 double 恢复结果")
                residuals.append(solve["cast_double"]["residual"])
            else:
                require(solve["cast_double"] is None, "直接 double 路线不能伪装 LD 转换")
            for residual in residuals:
                values = {key: decimal_metric(residual[key]) for key in ("residual_inf", "backward_error", "solution_inf")}
                if residual["finite"]:
                    required_products += 1
                    require(values["residual_inf"] is not None and values["solution_inf"] is not None and
                            values["residual_inf"] >= 0 and values["solution_inf"] >= 0, "有限真实残差与解范数")
                    if values["backward_error"] is not None:
                        norm = decimal_metric(report["matrix_metrics"]["inf_norm"])
                        expected = values["residual_inf"] / (norm*values["solution_inf"]+1)
                        require(abs(values["backward_error"]-expected) <= max(abs(expected), Fraction(1, 2**precision))*32*epsilon,
                                "后向误差与保存范数关系")
        require(required_products <= backend["residual_products"] <= 2*backend["solve_started"], "回放残差计数必要条件")
        summary.append({"name": backend["name"], "factor_success": backend["success"],
                        "rhs_finite": [item["success"] for item in backend["solves"]]})
    actual = report["backends"][0]
    previous = normal["failure"]
    reproduced = (actual["status"], actual["minor"]) == (previous["status"], previous["minor"])
    return {"cholmod_status_minor_reproduced": reproduced, "backends": summary,
            "residual_vectors_independently_recomputed": False,
            "residual_limit": "回放不保存完整解向量；这里核对计数、有限性和后向误差公式，未独立重做逐项残差。"}


def verify_offline(build, work, rows, summary):
    manifest = read(DATA / "v49-offline-manifest.json")
    hashes(manifest["protected"])
    require(manifest["diagnostic_only"] is True and manifest["wall_limit_per_replay"] == 120, "离线合同与预算")
    protected = {Path(path).resolve() for path in manifest["protected"]}
    require({DATA / "v49-offline.py", DATA / "v49-analyze-hsd.py", DATA / "v49-summary.json",
             Path(build["replay_executable"]), work / "build-manifest.json"} <= protected, "离线必需身份缺失")
    normals, normal_metadata, hsd_paths = {}, {}, [work / "hsd-self-on.hsd.json"]
    for case in summary["cases"]:
        prefix = Path(case["capture_prefix"])
        normal_path, hsd_path = Path(str(prefix)+".normal.json"), Path(str(prefix)+".hsd.json")
        if normal_path.exists():
            normals[case["case_index"]], normal_metadata[case["case_index"]] = verify_normal(normal_path, case["linear_solver"])
        if hsd_path.exists():
            hsd_paths.append(hsd_path)
        require({Path(path).resolve() for path in case["capture_files"]} <= protected, "离线快照身份覆盖")
    expected_cases = [case for case in summary["cases"] if normals.get(case["case_index"], {}).get("captured")]
    with gzip.open(DATA / "v49-replay-runs.jsonl.gz", "rt", encoding="utf-8") as source:
        replay_rows = [strict_json(line) for line in source]
    require(len(replay_rows) == manifest["replay_count"] == len(expected_cases), "保存矩阵必须各回放一次")
    replay_results = []
    for row, case in zip(replay_rows, expected_cases):
        require((row["case_index"], row["model"], row["linear_solver"], row["prefix"]) ==
                (case["case_index"], case["model"], case["linear_solver"], case["capture_prefix"]) and
                row["command"] == [build["replay_executable"], case["capture_prefix"]] and row["wall_limit"] == 120,
                "回放输入、顺序或命令")
        normal = normals[case["case_index"]]
        require(row["matrix_sha256_before"] == row["matrix_sha256_after"] == normal["matrix_sha256"], "回放矩阵前后身份")
        if row["error"] == "wall_timeout":
            require(row["returncode"] is None, "回放超时退出码")
            replay_results.append({"case_index": case["case_index"], "wall_timeout": True})
            continue
        require(row["error"] is None and row["returncode"] == 0 and strict_json(row["stdout"]) == row["report"], "回放正常报告")
        replay_results.append({"case_index": case["case_index"], **verify_replay(row["report"], normal)})
    analysis = read(DATA / "v49-hsd-analysis.json")
    process = read(DATA / "v49-hsd-analysis-process.json")
    require(process["returncode"] == 0 and process["error"] is None and process["wall_limit"] == 120 and
            analysis["passed"] is True and analysis["self_test"]["passed"] is True and
            analysis["analyzer_sha256"] == sha(DATA / "v49-analyze-hsd.py"), "HSD 离线分析来源和退出状态")
    require(len(hsd_paths) == manifest["hsd_metadata_count"] == len(analysis["records"]), "HSD 元数据分析集合")
    require(process["command"][1:] == [str(DATA / "v49-analyze-hsd.py"), *map(str, hsd_paths), "--output", str(DATA / "v49-hsd-analysis.json")], "HSD 离线命令")
    hsd_results = [verify_hsd(path, record) for path, record in zip(hsd_paths, analysis["records"])]
    synthetic_meta = read(hsd_paths[0])
    synthetic_run = strict_json(read(work / "hsd-self-on-run.json")["stdout"])
    digits = integer(synthetic_meta["long_double_digits"], "HSD LD digits")
    target = Fraction(1) + Fraction(2)**(1-digits)
    require(exact_hex(synthetic_meta["scalars"]["gap_rhs"]) == exact_hex(synthetic_run["target_ld"]) == target and
            synthetic_meta["failure"] == "synthetic_first" and synthetic_run["lp_runs"] == 0,
            "LD 非 double 位、首份快照和合成零 LP 边界")
    return {"normal_metadata": len(normals), "normal_captured": len(expected_cases), "replays": replay_results,
            "hsd_metadata": len(hsd_results), "hsd": hsd_results}


def verify_archive(manifest_path):
    """归档成员逐项与来源复核，不信任 ZIP 内路径，不展开文件。"""
    manifest = read(manifest_path)
    archive = Path(manifest["archive"])
    require(sha(archive) == manifest["zip_sha256"], "归档 ZIP 身份")
    members = manifest["members"]
    require(isinstance(members, dict) and members, "归档成员不能为空")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)) and set(names) == set(members), "归档成员集合或重复成员")
        for name, record in members.items():
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts and not name.startswith(("/", "\\")), "归档路径越界")
            require(relative.suffix.lower() not in (".exe", ".obj") and "frozen" not in relative.parts, "本归档不包含 exe/obj/冻结源码重复副本")
            source = Path(record["source"])
            require(source.stat().st_size == integer(record["bytes"], "archive.bytes") and sha(source) == record["sha256"], "归档来源身份")
            require(bundle.getinfo(name).file_size == record["bytes"], "归档成员长度")
            checksum = hashlib.sha256()
            with bundle.open(name) as stream:
                for data in iter(lambda: stream.read(1024*1024), b""):
                    checksum.update(data)
            require(checksum.hexdigest() == record["sha256"], "归档成员内容")
    return {"archive": str(archive), "zip_sha256": manifest["zip_sha256"], "members": len(members)}


def self_test():
    checks = 0
    def rejects(operation):
        nonlocal checks
        try:
            operation()
        except (AssertionError, ValueError, OverflowError):
            checks += 1
        else:
            raise AssertionError("合成非法输入没有被拒绝")
    rejects(lambda: strict_json('{"x":1,"x":2}'))
    rejects(lambda: strict_json('{"x":NaN}'))
    rejects(lambda: integer(True, "布尔值"))
    rejects(lambda: flags(["solver", "solve", "a.lp", "--threads", "1", "--threads", "2"]))
    require(non_timing_report({"timings": 1, "backend_version": "v1", "status": "x"}) == {"status": "x"},
            "只剔除两个已冻结的顶层字段")
    require(exact_hex("0x1.0000000000000001p+0") == 1 + Fraction(1, 2**64), "LD 位不得被 binary64 吞掉")
    require(exact_hex("-0x8p-3") == -1 and exact_hex("0x0p+0") == 0, "hexfloat 符号和指数")
    rejects(lambda: exact_hex("nan"))
    rejects(lambda: decimal_metric({"valid": True, "decimal": "Infinity"}))
    rejects(lambda: decimal_metric({"valid": False, "decimal": "0"}))
    rejects(lambda: verify_fraction_value({"numerator": "1", "denominator": "2", "decimal80": "0.5", "sign": 1}, Fraction(2, 3), "合成异值"))
    require(decimal_metric({"valid": True, "decimal": "1e400"}) == 10**400, "合法 LD 文本不应降为 double 无穷")
    require(windows_arguments('"C:\\some path\\clang++.exe" -c "a b.cpp" -o c.obj') ==
            ["C:\\some path\\clang++.exe", "-c", "a b.cpp", "-o", "c.obj"], "Windows 编译参数独立解码")
    arrays = {"starts": [0, 1], "rows": [0], "values": [1.0], "costs": [-1.0], "rhs": [1.0],
              "v": [2.0, 1.0], "u": [0.5, 0.25], "diagonal_h": [1.0], "regularized_h": [1.5]}
    answer = fraction_hsd(arrays, Fraction(1), Fraction(1), Fraction(1, 4))
    require(answer["denominator"] == 2 and answer["chi"] == 0 and answer["chi_regularized"] == Fraction(3, 2),
            "解析常数解与实际正则残差")
    arrays["v"] = [0.0, 2.0]
    answer = fraction_hsd(arrays, Fraction(1), Fraction(1), Fraction(1, 4))
    require(answer["denominator"] == -1 and answer["energy"] == 4 and answer["chi"] == 6,
            "精确负分母必须保留，不截取正值")
    with tempfile.TemporaryDirectory(prefix="v49-verify-self-") as directory:
        path = Path(directory) / "synthetic.bin"
        path.write_bytes(b"V49HSD1\0" + struct.pack("<2d", 1.0, -2.0))
        descriptors = [{"name": "v", "dtype": "f64_le", "count": 2, "offset": 8}]
        with BinaryArrays(path, descriptors, b"V49HSD1\0") as values:
            require(list(values["v"]) == [1.0, -2.0], "只读数组映射")
        rejects(lambda: BinaryArrays(path, [dict(descriptors[0], count=3)], b"V49HSD1\0"))
        rejects(lambda: BinaryArrays(path, [dict(descriptors[0], offset=9)], b"V49HSD1\0"))
        rejects(lambda: BinaryArrays(path, [dict(descriptors[0], count=1)], b"V49HSD1\0"))
        archive = Path(directory) / "synthetic.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.write(path, "synthetic.bin")
        manifest = Path(directory) / "manifest.json"
        manifest.write_text(json.dumps({"archive": str(archive), "zip_sha256": sha(archive),
            "members": {"synthetic.bin": {"source": str(path), "sha256": sha(path), "bytes": path.stat().st_size}}}), encoding="utf-8")
        require(verify_archive(manifest)["members"] == 1, "合成归档来源和成员逐项核验")
        changed = read(manifest)
        changed["members"]["synthetic.bin"]["bytes"] += 1
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        rejects(lambda: verify_archive(manifest))
    return {"passed": True, "scope": "仅合成 JSON、命令、二进制数组和 Fraction 代数；未运行 LP", "negative_cases": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--work", type=Path, default=ROOT / ".tools/v49-diagnostic")
    parser.add_argument("--archive-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return 0
    if args.preflight:
        plan, _ = frozen_plan()
        print(json.dumps({"passed": True, "scope": "仅测前计划与合成自测，不代表正式归档核验",
                          "plan_sha256": PLAN_SHA, "clarification_sha256": CLARIFICATION_SHA,
                          "expected_runs": len(expected_runs(plan)), "self_test": self_test()}, ensure_ascii=False, indent=2))
        return 0
    plan, _ = frozen_plan()
    work = args.work.resolve()
    require(work.is_relative_to(ROOT), "核验工作目录须位于项目内")
    build, build_result = verify_build(work, plan)
    rows, summary = verify_runs(plan, build, work)
    offline = verify_offline(build, work, rows, summary)
    result = {"passed": True, "diagnostic_only": True, "verifier_sha256": sha(__file__), "plan_sha256": PLAN_SHA,
        "clarification_sha256": CLARIFICATION_SHA, "build_manifest_sha256": sha(work / "build-manifest.json"),
        "raw_sha256": sha(DATA / "v49-runs.jsonl.gz"), "self_test": self_test(), "build": build_result,
        "runs": len(rows), "pairs": len(summary["pairs"]), "non_budget_pairs": summary["non_budget_pairs"],
        "non_budget_equal": summary["non_budget_equal"], "budget_pairs": summary["budget_pairs"], "offline": offline,
        "archive": verify_archive(args.archive_manifest) if args.archive_manifest else None,
        "limits": ["诊断证据核验，不是性能扩集、LP 达标或加速结论。",
                   "仅核验显式构建输入及记录的编译依赖；不声称逐项核验隐式运行库。",
                   "轨迹仅剔除顶层 timings/backend_version；预算受限分叉单列。",
                   "齐次精确恒等式针对保存方向与实际 H，不推断理想精确方向。"]}
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
