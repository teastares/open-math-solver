"""从冻结v66环境构建presolve结构探针与postsolve小型测试，不替换生产求解器。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA = "5a12611f867d4c78fc3399050343a389eee1df883934e7e5917005a54ff451ec"
HELPER_SHA = "157f157a6de6f6e5cec8935ccd8977e9e13b7f30f674d4e76701bd1c25aed6b5"
BASE_PATH = DATA / "v47-profile-build.py"
assert hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() == HELPER_SHA
BASE = runpy.run_path(str(BASE_PATH))
sha, require = (BASE[key] for key in ("sha", "require"))
split_windows, resolve = (BASE[key] for key in ("split_windows", "resolve"))
file_hashes, require_hashes = (BASE[key] for key in ("file_hashes", "require_hashes"))
compile_arguments, dependency_paths = (BASE[key] for key in ("compile_arguments", "dependency_paths"))
ROLES = {"probe": ("v85-presolve-probe.cpp", "presolve-probe.cpp", "presolve-probe.exe"),
    "postsolve": ("v85-postsolve-self-test.cpp", "postsolve-self-test.cpp", "postsolve-self-test.exe")}


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "JSON键重复：" + key)
            result[key] = value
        return result
    def finite(text):
        value = float(text)
        require(-float("inf") < value < float("inf"), "JSON浮点溢出")
        return value
    def constant(text):
        raise ValueError("非有限JSON：" + text)
    return json.loads(text, object_pairs_hook=unique, parse_float=finite, parse_constant=constant)


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute(command, cwd, path, environment):
    try:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, timeout=300)
        row = dict(command=command, cwd=str(cwd), returncode=process.returncode, wall_timeout=False,
            stdout=process.stdout.decode("utf-8", errors="replace"), stderr=process.stderr.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        row = dict(command=command, cwd=str(cwd), returncode=None, wall_timeout=True,
            stdout=decode(error.stdout), stderr=decode(error.stderr))
    save(path, row)
    require(type(row["returncode"]) is int and row["returncode"] == 0 and row["wall_timeout"] is False,
        "构建或小型测试失败，原件已保留：" + str(path))
    return row


def source_check(role, path):
    text = path.read_text(encoding="utf-8")
    # 词法检查只约束本轮显式调用；内部行为仍须独立源码审查和真实计数证据。
    scrubbed = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', " ", text)
    require(len(re.findall(r"\bint\s+main\s*\(", scrubbed)) == 1, "独立诊断程序须有唯一标准main")
    require(not re.search(r"(?:\.|->|::)\s*(?:run|crossover|solve)\s*\(", scrubbed) and
        not re.search(r"\bHighs_(?:run|crossover)\s*\(", scrubbed), "诊断不得显式调用LP迭代或crossover入口")
    if role == "probe":
        require(not re.search(r"(?:\.|->|::)\s*postsolve\s*\(", scrubbed), "结构探针不得执行postsolve")
    return dict(source=str(path), source_sha256=sha(path), standard_main=True,
        explicit_lp_iteration_calls=0, explicit_crossover_calls=0,
        scope="仅检查本轮源码的显式调用，不宣称二进制hook或完整内部调用证明。")


def fixed_plan():
    require(sha(DATA / "v85-plan.json") == PLAN_SHA, "冻结计划变化或尚未填写SHA")
    plan = read(DATA / "v85-plan.json")
    require(sha(DATA / "ipm-v66-identity.json") == plan["parent_identity_sha256"], "冻结v66身份变化")
    for key in ("baseline_verification", "evidence"):
        require(sha(DATA / plan[key + "_file"]) == plan[key + "_sha256"], "计划来源变化：" + key)
    require(plan["production_changes"] is False and type(plan["maximum_public_lp_runs"]) is int and plan["maximum_public_lp_runs"] == 0 and
        plan["build_helper_sha256"] == HELPER_SHA and plan["structure_seconds"] == 5 and plan["threads"] == 1, "结构诊断与固定构建约束变化")
    require_hashes(plan["live_highs_sources"])
    return plan


def static_check():
    fixed_plan()
    checks = {role: source_check(role, DATA / names[0]) for role, names in ROLES.items()}
    return dict(passed=True, public_lp_runs=0, explicit_lp_solve_calls=0,
        plan_sha256=PLAN_SHA, units=checks)


def explicit_inputs(command, cwd):
    return file_hashes([resolve(value, cwd) for value in command if not value.startswith("-") and
        Path(value).suffix.lower() in (".a", ".lib", ".obj", ".o")])


def inspect_test(role, raw):
    require(raw["stderr"] == "", "小型诊断测试不得有未解释stderr：" + role)
    value = strict_json(raw["stdout"])
    require(isinstance(value, dict) and value.get("passed") is True and
        type(value.get("public_lp_runs")) is int and value["public_lp_runs"] == 0 and
        type(value.get("explicit_lp_solve_calls")) is int and value["explicit_lp_solve_calls"] == 0 and
        type(value.get("cases")) is int and value["cases"] > 0 and isinstance(value.get("details"), list) and
        len(value["details"]) == value["cases"], "小型诊断测试摘要无效：" + role)
    if role == "probe":
        require(set(value) == {"passed", "public_lp_runs", "explicit_lp_solve_calls", "cases", "default_limits", "details"}, "探针自测摘要字段变化")
        require(all(isinstance(item, str) and item for item in value["details"]), "探针自测名称必须完整")
        limits = value.get("default_limits")
        require(isinstance(limits, dict) and set(limits) == {"external_seconds", "budget_check_work_interval"} and
            type(limits["external_seconds"]) is int and limits["external_seconds"] == 5 and
            type(limits["budget_check_work_interval"]) is int and limits["budget_check_work_interval"] == 256,
            "实际探针默认预算值和JSON类型必须与冻结限制一致")
    else:
        require(type(value.get("explicit_lp_solve_calls")) is int and value["explicit_lp_solve_calls"] == 0,
            "postsolve小型测试显式LP求解调用必须为零")
        groups = ("recovery_cases", "rejection_cases", "budget_cases", "contract_cases")
        coverage = ("min_cases", "max_cases", "empty_reduced_cases", "nonempty_reduced_cases")
        require(set(value) == {"passed", "public_lp_runs", "explicit_lp_solve_calls", "cases", "details", *groups, *coverage} and
            all(type(value.get(key)) is int and value[key] > 0 for key in (*groups, *coverage)) and
            sum(value[key] for key in groups) == value["cases"] and
            value["min_cases"] + value["max_cases"] == value["empty_reduced_cases"] + value["nonempty_reduced_cases"] == value["recovery_cases"],
            "postsolve测试组、方向与真实空/非空恢复覆盖必须守恒")
        fields = {"id", "kind", "highs_observed", "original_columns", "original_rows", "reduced_columns", "reduced_rows",
            "presolve_status", "postsolve_status", "expected_accept", "accepted", "original_verified", "original_objective", "objective_gap",
            "basis_valid", "basis_statuses_empty", "simplex_iterations", "ipm_iterations", "crossover_iterations", "positive_iterations_observed"}
        ids = set()
        for detail in value["details"]:
            require(isinstance(detail, dict) and set(detail) == fields and isinstance(detail["id"], str) and detail["id"] and
                detail["id"] not in ids and isinstance(detail["kind"], str), "postsolve详细原件字段或标识无效")
            ids.add(detail["id"])
            require(all(type(detail[key]) is bool for key in ("highs_observed", "expected_accept", "accepted", "original_verified", "positive_iterations_observed")) and
                detail["positive_iterations_observed"] is False and all(type(detail[key]) is int and detail[key] >= 0 for key in
                    ("original_columns", "original_rows", "reduced_columns", "reduced_rows")), "postsolve详细布尔、维数与无正迭代证据")
            for key in ("presolve_status", "postsolve_status"):
                require(detail[key] is None or type(detail[key]) is int, "原生状态只能是原始整数或null")
            for key in ("original_objective", "objective_gap"):
                require(detail[key] is None or type(detail[key]) in (int, float) and -float("inf") < detail[key] < float("inf"), "原目标质量数值必须有限或null")
            if detail["highs_observed"]:
                require(detail["basis_valid"] is False and type(detail["basis_statuses_empty"]) is bool and all(
                    type(detail[key]) is int and detail[key] in (-1, 0) for key in ("simplex_iterations", "ipm_iterations", "crossover_iterations")),
                    "真实恢复须无效basis并保留返回数组观察，原计数-1/0不得伪作零")
            else:
                require(all(detail[key] is None for key in ("basis_valid", "basis_statuses_empty", "simplex_iterations", "ipm_iterations", "crossover_iterations",
                    "presolve_status", "postsolve_status")), "纯合同/预算例不得伪造原生观察")
            require(detail["accepted"] is detail["expected_accept"] and
                (not detail["accepted"] or detail["original_verified"]), "原模型完整验收与接纳合同不符")
    return value


def build(output):
    require(output.is_relative_to(ROOT) and not output.exists(), "输出须位于工作区且不得覆盖既有原件")
    plan = fixed_plan()
    checks = static_check()
    parent_path, archive_path = DATA / "ipm-v66-identity.json", DATA / "ipm-v66-source.zip"
    parent = read(parent_path)
    baseline = read(DATA / plan["baseline_verification_file"])
    require(baseline["passed"] is True and baseline["files"] == parent["files"] and baseline["libraries"] == parent["libraries"] and
        len(parent["files"]) == 95 and len(parent["libraries"]) == 17, "只冻结v66原95来源和17库")
    production, reference = ROOT / "build/release/solver.exe", ROOT / ".tools/solver-ipm-v66.exe"
    require(sha(production) == sha(reference) == baseline["solver_sha256"] == parent["solver_sha256"] == plan["parent_solver_sha256"] and
        sha(archive_path) == parent["source_zip_sha256"] == plan["parent_source_sha256"], "原程序或来源包变化")
    sources = {str(ROOT / name): digest for name, digest in parent["files"].items()}
    libraries = {str(ROOT / name): digest for name, digest in parent["libraries"].items()}
    require_hashes(sources); require_hashes(libraries)
    compiler, cwd = Path(parent["compiler"]), ROOT / "build/release"
    require(sha(compiler) == parent["compiler_sha256"], "编译器变化")
    ninja = ROOT / ".tools/ninja-1.13.2/ninja.exe"
    query = [str(ninja), "-C", str(cwd), "-t", "commands", "solver.exe"]
    command_text = subprocess.check_output(query, text=True, encoding="utf-8")
    lines, original_compile = command_text.splitlines(), {}
    for role, relative in (("highs", "src/highs_backend.cpp"), ("main", "src/main.cpp")):
        matches = [split_windows(line) for line in lines if " -c " in line and line.replace("\\", "/").endswith("/" + relative)]
        require(len(matches) == 1 and resolve(matches[0][0], cwd) == compiler, "原始编译模板不唯一：" + role)
        original_compile[role] = matches[0]
    require([line for line in lines if " -o solver.exe " in line] == [parent["link_command"]], "原链接模板变化")
    outer = split_windows(parent["link_command"])
    require(len(outer) == 3 and outer[1].lower() == "/c" and outer[2].startswith("cd . && ") and outer[2].endswith(" && cd ."), "原链接包装变化")
    original_link = split_windows(outer[2][len("cd . && "):-len(" && cd .")])
    require(resolve(original_link[0], cwd) == compiler, "链接编译器变化")
    main_obj = original_compile["main"][original_compile["main"].index("-o") + 1]
    code_files = [DATA / name for name in ("v85-presolve-probe.cpp", "v85-postsolve-self-test.cpp", "v85-build.py", "v85-run.py", "v85-runner-self-check.json")]
    protected = file_hashes([production, reference, parent_path, archive_path, DATA / "v85-plan.json", BASE_PATH,
        DATA / plan["baseline_verification_file"], DATA / plan["evidence_file"],
        compiler, ninja, cwd / "build.ninja", cwd / "CMakeFiles/rules.ninja", *code_files,
        *map(Path, explicit_inputs(original_link, cwd))])
    protected.update(plan["live_highs_sources"])
    runner_check = read(DATA / "v85-runner-self-check.json")
    require(runner_check.get("passed") is True and type(runner_check.get("public_lp_runs")) is int and
        runner_check["public_lp_runs"] == 0, "运行器纯数据自检尚未通过")
    output.mkdir(parents=True)
    frozen = output / "frozen"
    with zipfile.ZipFile(archive_path) as archive:
        require(len(archive.namelist()) == 95 and set(archive.namelist()) == set(parent["files"]) and
            all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in archive.namelist()), "冻结来源包成员或路径变化")
        archive.extractall(frozen)
    frozen_map = {str(frozen / name): digest for name, digest in parent["files"].items()}
    require_hashes(frozen_map)
    save(output / "static-check.json", checks)
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("SOLVER_V") and
        key.upper() not in {"OPENBLAS_CORETYPE", *threads}}
    environment.update(dict.fromkeys(threads, "1"))
    environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    generated, dependencies, links, executables, executable_sha256, copies, self_tests = {}, {}, {}, {}, {}, {}, {}
    compile_commands, link_commands, test_commands = {}, {}, {}
    for role, (source_name, copy_name, exe_name) in ROLES.items():
        source, obj, executable = output / copy_name, output / (role + ".obj"), output / exe_name
        source.write_bytes((DATA / source_name).read_bytes())
        generated.update(file_hashes([source]))
        copies[role] = dict(source=str(DATA / source_name), copy=str(source), sha256=sha(source))
        command = compile_arguments(original_compile["highs"], source, obj, frozen, ROOT)
        compile_commands[role] = command
        execute(command, cwd, output / (role + "-compile.json"), environment)
        paths = dependency_paths(Path(str(obj) + ".d"), cwd)
        require(source.resolve() in {path.resolve() for path in paths}, "实际.d缺独立CPP来源")
        for path, digest in file_hashes(paths).items():
            require(path not in dependencies or dependencies[path] == digest, "两个单元编译期间依赖变化")
            dependencies[path] = digest
        link, replacements = [], 0
        for value in original_link:
            if value == main_obj: link.append(str(obj)); replacements += 1
            elif value != "-municode" and not value.startswith("-Wl,--out-implib,"): link.append(value)
        require(replacements == 1 and link.count("-o") == 1, "标准main链接替换必须唯一")
        link[link.index("-o") + 1] = str(executable)
        link_commands[role] = link
        for path, digest in explicit_inputs(link, cwd).items():
            require(path not in links or links[path] == digest, "两个程序链接期间输入变化")
            links[path] = digest
        execute(link, cwd, output / (role + "-link.json"), environment)
        test_command = [str(executable), "--self-test"] if role == "probe" else [str(executable)]
        test_commands[role] = test_command
        raw = execute(test_command, ROOT, output / (role + "-self-run.json"), environment)
        details = inspect_test(role, raw)
        check = dict(passed=True, public_lp_runs=0, details=details)
        save(output / (role + "-self-check.json"), check)
        self_tests[role] = check
        executables[role], executable_sha256[role] = str(executable), sha(executable)
    for mapping in (protected, sources, libraries, frozen_map, generated, dependencies, links): require_hashes(mapping)
    outputs = file_hashes([path for path in output.iterdir() if path.is_file()])
    manifest = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), production_changes=False,
        experimental_diagnostic_only=True, plan_sha256=PLAN_SHA, parent_identity_sha256=plan["parent_identity_sha256"],
        production_executable=str(production), production_sha256=sha(production), frozen_parent_executable=str(reference),
        frozen_parent_sha256=sha(reference), executables=executables, executable_sha256=executable_sha256,
        baseline_verification_file=str(DATA / plan["baseline_verification_file"]), baseline_verification_sha256=plan["baseline_verification_sha256"],
        frozen_files=parent["files"], current_files=baseline["files"], actual_libraries=baseline["libraries"], source_copies=copies,
        protected_before_and_after=protected, generated_inputs_before_and_after=generated,
        compile_inputs_after_and_final=dependencies, link_explicit_before_and_after=links,
        outputs=outputs, self_tests=self_tests,
        commands=dict(cwd=str(cwd), query=query, original_compile=original_compile, original_link=original_link,
            compile=compile_commands, link=link_commands, self_tests=test_commands),
        original_all_libraries_retained=True, implicit_link_inputs_verified=False, public_lp_runs=0, explicit_lp_solve_calls=0)
    save(output / "build-manifest.json", manifest)
    print(json.dumps(dict(passed=True, compiled_units=list(ROLES), public_lp_runs=0,
        executable_sha256=executable_sha256, self_test_cases={role: value["details"]["cases"] for role, value in self_tests.items()}), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".tools/v85-presolve")
    parser.add_argument("--self-test", action="store_true", help="仅锁定计划与显式源码调用，不编译或运行程序")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(static_check(), ensure_ascii=False))
    else:
        build(arguments.output.resolve())
