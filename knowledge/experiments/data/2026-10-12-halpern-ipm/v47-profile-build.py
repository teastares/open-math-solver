"""生成两翻译单元的独立阶段剖析程序；不修改生产源码或生产构建产物。"""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PREFIX = "V47_PROFILE "
ENVIRONMENT = "SOLVER_V47_PROFILE"
STAGES = (
    "total", "canonicalize", "kkt_setup", "normal_setup", "ordering", "symbolic_analysis",
    "factor", "normal_factor", "gram_assembly", "numeric_factor", "factor_check",
    "solve", "krylov", "direct", "solve_factored", "normal_solve", "triangular_solve",
    "kkt_product", "assess_homogeneous", "assess", "verify_original",
    "homogeneous_direction", "initialize", "run_homogeneous",
    "fallback_factor_adapter", "fallback_solve_adapter",
)

HEADER = r'''#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace solver_v47_profile {
struct Entry {
  std::int64_t inclusive_ns = 0, exclusive_ns = 0;
  std::uint64_t started = 0, completed = 0, unwound = 0;
};
inline constexpr const char* names[] = {@NAMES@};
inline thread_local std::array<Entry, @COUNT@> entries{};
inline thread_local bool enabled = false;
inline thread_local std::uint64_t clock_reads = 0;
struct Scope;
inline thread_local Scope* parent = nullptr;
#ifdef SOLVER_V47_PROFILE_SELFTEST
inline thread_local std::int64_t (*test_clock)() noexcept = nullptr;
#endif
inline std::int64_t now() noexcept {
  ++clock_reads;
#ifdef SOLVER_V47_PROFILE_SELFTEST
  if (test_clock) return test_clock();
#endif
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Scope {
  bool active = false;
  std::size_t index = 0;
  Scope* previous = nullptr;
  std::int64_t start = 0, children = 0;
  int exceptions = 0;
  explicit Scope(std::size_t value) noexcept {
    // 关闭时不取时间、不访问作用域链，也不分配内存。
    if (!enabled) return;
    active = true;
    index = value;
    previous = parent;
    exceptions = std::uncaught_exceptions();
    start = now();
    ++entries[index].started;
    parent = this;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept {
    if (!active) return;
    const auto duration = now() - start;
    auto& entry = entries[index];
    entry.inclusive_ns += duration;
    entry.exclusive_ns += duration - children;
    if (std::uncaught_exceptions() > exceptions) ++entry.unwound;
    else ++entry.completed;
    parent = previous;
    if (previous) previous->children += duration;
  }
};
template<class Function> decltype(auto) call(std::size_t index, Function&& function) {
  Scope scope(index);
  return function();
}
template<class Value, class Function> Value materialize(std::size_t index, Function&& function) {
  Scope scope(index);
  // Eigen 惰性表达式必须在计时作用域内物化，不能只计表达式构造。
  return Value(function());
}
inline std::int64_t exclusive_sum() noexcept {
  std::int64_t result = 0;
  for (const auto& entry : entries) result += entry.exclusive_ns;
  return result;
}
inline bool counts_closed() noexcept {
  std::uint64_t started = 0;
  for (const auto& entry : entries) {
    if (entry.started != entry.completed + entry.unwound) return false;
    started += entry.started;
  }
  return clock_reads == 2 * started;
}
struct Report {
  bool active = false;
  Report() noexcept {
    const auto* value = std::getenv("SOLVER_V47_PROFILE");
    enabled = value && value[0] == '1' && value[1] == '\0';
    if (!enabled) return;
    active = true;
    entries = {};
    clock_reads = 0;
    parent = nullptr;
  }
  ~Report() noexcept {
    if (!active) return;
    const auto sum = exclusive_sum();
    std::fputs("V47_PROFILE {\"schema\":1,\"stages\":{", stderr);
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const auto& entry = entries[i];
      std::fprintf(stderr, "%s\"%s\":{\"inclusive_ns\":%lld,\"exclusive_ns\":%lld,"
          "\"started\":%llu,\"completed\":%llu,\"unwound\":%llu}", i ? "," : "", names[i],
          static_cast<long long>(entry.inclusive_ns), static_cast<long long>(entry.exclusive_ns),
          static_cast<unsigned long long>(entry.started), static_cast<unsigned long long>(entry.completed),
          static_cast<unsigned long long>(entry.unwound));
    }
    std::fprintf(stderr, "},\"clock_reads\":%llu,\"exclusive_sum_ns\":%lld,"
        "\"total_inclusive_ns\":%lld,\"conserved\":%s,\"stack_empty\":%s,\"counts_closed\":%s}\n",
        static_cast<unsigned long long>(clock_reads), static_cast<long long>(sum),
        static_cast<long long>(entries[0].inclusive_ns), sum == entries[0].inclusive_ns ? "true" : "false",
        parent == nullptr ? "true" : "false", counts_closed() ? "true" : "false");
    enabled = false;
  }
};
}  // 命名空间 solver_v47_profile
'''.replace("@NAMES@", ",".join(json.dumps(name) for name in STAGES)).replace("@COUNT@", str(len(STAGES)))


SELF_MAIN = r'''#include "v47-profile.hpp"
void profile_cross_tu(bool fail);
namespace {
std::int64_t tick = 0;
std::int64_t fake_clock() noexcept { const auto result = tick; tick += 10; return result; }
}
int main() {
  using namespace solver_v47_profile;
  test_clock = fake_clock;
  Report report;
  const bool active = enabled;
  {
    Scope total(0);
    Scope outer(1);
    profile_cross_tu(false);
    try { profile_cross_tu(true); } catch (int value) { if (value != 7) return 3; }
  }
  bool passed = parent == nullptr && counts_closed();
  if (active) {
    passed = passed && clock_reads == 12 && entries[0].inclusive_ns == 110 &&
        entries[0].exclusive_ns == 20 && entries[1].inclusive_ns == 110 &&
        entries[1].exclusive_ns == 50 && entries[3].inclusive_ns == 60 &&
        entries[3].exclusive_ns == 40 && entries[1].started == 3 &&
        entries[1].completed == 2 && entries[1].unwound == 1 &&
        entries[3].started == 2 && entries[3].completed == 1 && entries[3].unwound == 1 &&
        exclusive_sum() == entries[0].inclusive_ns;
  } else {
    passed = passed && clock_reads == 0 && tick == 0;
    for (const auto& entry : entries) passed = passed && entry.started == 0 && entry.inclusive_ns == 0;
  }
  std::printf("{\"passed\":%s,\"enabled\":%s}\n", passed ? "true" : "false", active ? "true" : "false");
  return passed ? 0 : 2;
}
'''
SELF_CROSS = r'''#include "v47-profile.hpp"
void profile_cross_tu(bool fail) {
  solver_v47_profile::Scope outer(3);
  solver_v47_profile::Scope inner(1);
  if (fail) throw 7;
}
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def transform(source, unit):
    """每项替换必须唯一，逐项逆变换必须精确恢复规范化换行后的原文。"""
    changes = []

    def replace(label, before, after):
        nonlocal source
        require(source.count(before) == 1, f"{unit}/{label} 锚点不唯一")
        require(before != after, "插桩变换不得为空")
        source = source.replace(before, after, 1)
        changes.append({"label": label, "before": before, "after": after})

    def scope(label, pattern, stage):
        matches = list(re.finditer(pattern, source))
        require(len(matches) == 1, f"{unit}/{label} 函数锚点数量为 {len(matches)}")
        before = matches[0].group()
        addition = f"\n  solver_v47_profile::Scope v47_scope({STAGES.index(stage)});"
        if stage == "total":
            addition = "\n  solver_v47_profile::Report v47_report;" + addition
        replace(label, before, before + addition)

    def expression(label, expression, stage, value_type=None):
        name = "call" if value_type is None else f"materialize<{value_type}>"
        replace(label, expression, f"solver_v47_profile::{name}({STAGES.index(stage)}, [&] {{ return {expression}; }})")

    original = source
    namespace = "namespace solver {" if unit == "ipm" else "namespace solver::detail {"
    replace("shared_header", namespace, '#include "v47-profile.hpp"\n\n' + namespace)
    if unit == "ipm":
        for label, pattern, stage in (
            ("total", r"SolveResult solve_ipm\([^{}]+\) \{", "total"),
            ("canonicalize", r"ConeModel canonicalize\([^{}]+\) \{", "canonicalize"),
            ("kkt_setup", r"KktFactor\([^{}]+krylov_history_\([^{}]+\) \{", "kkt_setup"),
            ("factor", r"void factor\([^{}]+\) \{", "factor"),
            ("solve", r"std::vector<double> solve\([^{}]+\) const \{", "solve"),
            ("krylov", r"std::vector<double> krylov_refine\([^{}]+\) const \{", "krylov"),
            ("direct", r"std::vector<double> direct\([^{}]+\) const \{", "direct"),
            ("solve_factored", r"void solve_factored\([^{}]+\) const \{", "solve_factored"),
            ("kkt_product", r"void unregularized_residual\([^{}]+\) const \{", "kkt_product"),
            ("assess_homogeneous", r"bool assess_homogeneous\([^{}]+\) \{", "assess_homogeneous"),
            ("assess", r"bool assess\([^{}]+\) \{", "assess"),
            ("homogeneous_direction", r"HomogeneousDirection homogeneous_direction\([^{}]+\) \{", "homogeneous_direction"),
            ("initialize", r"void initialize\([^{}]+\) \{", "initialize"),
            ("run_homogeneous", r"void run_homogeneous_ipm\([^{}]+\) \{", "run_homogeneous"),
        ):
            scope(label, pattern, stage)
        expression("verify_original", "verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality)", "verify_original")
        expression("kkt_ordering", "amd_l_order(static_cast<Index>(factor_dimension_), starts.data(), rows.data(),\n                                      permutation.data(), nullptr, nullptr)", "ordering")
        # 结构回退的适配器仍含内部检查，不把它命名为纯数值内核。
        expression("fallback_factor", "multifrontal_ldl_->factor(std::span<const double>(factor_values, ax_.size()),\n            [&options, &start] { check_budget(options, start); })", "fallback_factor_adapter")
        expression("fallback_solve", "multifrontal_ldl_->solve(rhs,\n          [&options, &start] { check_budget(options, start); })", "fallback_solve_adapter")
    elif unit == "normal":
        scope("normal_setup", r"Impl\(std::size_t size,[^{}]+max_refinement_steps\(refinement_budget\) \{", "normal_setup")
        scope("normal_factor", r"bool NormalCholesky::factor\([^{}]+\) \{", "normal_factor")
        scope("normal_solve", r"bool NormalCholesky::solve\([^{}]+\) const \{", "normal_solve")
        scope("factor_check", r"bool finite_factor\([^{}]+\) const \{", "factor_check")
        expression("normal_ordering", "amd_l_order(retained, matrix.outerIndexPtr(), matrix.innerIndexPtr(), permutation.data(), nullptr, nullptr)", "ordering")
        expression("symbolic_etree", "QDLDL_etree(static_cast<QDLDL_int>(retained), starts.data(), rows.data(), work.data(), counts.data(), tree.data())", "symbolic_analysis")
        expression("symbolic_eigen", "cholesky.analyzePattern(matrix)", "symbolic_analysis")
        expression("symbolic_extended", "extended_cholesky.analyzePattern(extended_matrix)", "symbolic_analysis")
        pattern = r"  state.input_values.assign\(values.begin\(\), values.end\(\)\);[\s\S]+?  check\(budget_check\);\n(?=  if \(!state.dual_coordinates.empty\(\)\) \{)"
        matches = list(re.finditer(pattern, source))
        require(len(matches) == 1, "Gram 装配锚点不唯一")
        body = matches[0].group()
        replace("gram_assembly", body, "  {\n    solver_v47_profile::Scope v47_gram_scope(8);\n" + body + "  }\n")
        expression("numeric_double", "state.cholesky.factorize(state.matrix)", "numeric_factor")
        expression("numeric_extended", "state.extended_cholesky.factorize(state.extended_matrix)", "numeric_factor")
        for label, expr, value_type in (
            ("triangular_double", "state.cholesky.solve(permuted)", "Eigen::VectorXd"),
            ("triangular_extended", "state.extended_cholesky.solve(permuted)", "ExtendedVector"),
        ):
            expression(label, expr, "triangular_solve", value_type)
    else:
        raise ValueError("未知翻译单元")
    rebuilt = source
    for change in reversed(changes):
        require(rebuilt.count(change["after"]) == 1, f"逆锚点不唯一：{change['label']}")
        rebuilt = rebuilt.replace(change["after"], change["before"], 1)
    require(rebuilt == original, f"{unit} 插桩不可逆")
    return source, changes


def static_check(root):
    records = {}
    for unit, relative in (("ipm", "src/ipm.cpp"), ("normal", "src/normal_cholesky.cpp")):
        original = (root / relative).read_text(encoding="utf-8")
        patched, changes = transform(original, unit)
        records[unit] = {"anchors": len(changes), "source_text_sha256": digest(original),
                         "instrumented_sha256": digest(patched), "reversible": True}
        # 重复和缺失的真实锚点必须拒绝，不能宽松地选择其中一个。
        for modified in (original.replace("namespace solver", "namespace removed", 1), original + original):
            rejected = False
            try:
                transform(modified, unit)
            except RuntimeError:
                rejected = True
            require(rejected, "插桩必须拒绝缺失或重复锚点")
    return {"passed": True, "units": records, "stages": list(STAGES),
            "header_sha256": digest(HEADER), "scope": "仅静态变换与反例，不编译、不运行 LP"}


def split_windows(command):
    require(os.name == "nt", "本构建器复用 Windows Ninja 命令")
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command, ctypes.byref(count))
    require(bool(pointer), "无法解析 Windows 参数")
    try:
        return [pointer[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(pointer)


def resolve(path, cwd):
    value = Path(path)
    return (value if value.is_absolute() else cwd / value).resolve()


def file_hashes(paths):
    return {str(path): sha(path) for path in sorted(set(paths), key=str)}


def require_hashes(mapping):
    for path, expected in mapping.items():
        require(sha(path) == expected, f"身份变化：{path}")


def compile_arguments(original, source, obj, frozen, root):
    result = list(original)
    for option, value in (("-o", obj), ("-c", source), ("-MF", Path(str(obj) + ".d")), ("-MT", obj)):
        require(result.count(option) == 1, f"编译选项 {option} 不唯一")
        result[result.index(option) + 1] = str(value)
    old_include = root / "include"
    changed = 0
    for index, value in enumerate(result):
        if value.startswith("-I") and resolve(value[2:], root) == old_include:
            result[index] = "-I" + str(frozen / "include")
            changed += 1
    require(changed == 1, "未唯一替换冻结头文件路径")
    return result


def execute(command, cwd, destination, environment=None):
    try:
        process = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, timeout=300)
        row = {"command": command, "cwd": str(cwd), "returncode": process.returncode,
               "stdout": process.stdout.decode("utf-8", errors="replace"),
               "stderr": process.stderr.decode("utf-8", errors="replace"), "wall_timeout": False}
    except subprocess.TimeoutExpired as error:
        decode = lambda value: value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        row = {"command": command, "cwd": str(cwd), "returncode": None, "stdout": decode(error.stdout),
               "stderr": decode(error.stderr), "wall_timeout": True}
    save(destination, row)
    require(row["returncode"] == 0, f"独立构建或自测失败，已保留 {destination}")
    return row


def dependency_paths(path, cwd):
    text = re.sub(r"\\\r?\n", "", path.read_text(encoding="utf-8"))
    require(": " in text, "编译依赖文件格式无效")
    body = text.split(": ", 1)[1]
    tokens = re.findall(r"(?:\\.|[^\s])+", body)
    return [resolve(re.sub(r"\\([ #:\\])", r"\1", token), cwd) for token in tokens]


def build(root, output):
    data = root / DATA.relative_to(ROOT)
    build_dir = root / "build/release"
    require(output.is_relative_to(root) and not output.exists(), "输出必须是工作区中尚不存在的独立目录")
    checks = static_check(root)
    identity_path = data / "ipm-v46-identity.json"
    archive = data / "ipm-v46-source.zip"
    locked = json.loads(identity_path.read_text(encoding="utf-8"))
    require(sha(archive) == locked["source_zip_sha256"], "v46 源码归档身份变化")
    production = build_dir / "solver.exe"
    require(sha(production) == locked["solver_sha256"], "正式程序不对应冻结 v46")
    current_sources = {str(root / name): value for name, value in locked["files"].items()}
    current_libraries = {str(root / name): value for name, value in locked["libraries"].items()}
    require_hashes(current_sources)
    require_hashes(current_libraries)
    compiler = Path(locked["compiler"])
    require(sha(compiler) == locked["compiler_sha256"], "编译器身份变化")
    ninja = root / ".tools/ninja-1.13.2/ninja.exe"
    query = [str(ninja), "-C", str(build_dir), "-t", "commands", "solver.exe"]
    command_text = subprocess.check_output(query, text=True, encoding="utf-8")
    commands = command_text.splitlines()
    compile_source = {}
    for unit, filename in (("ipm", "ipm.cpp"), ("normal", "normal_cholesky.cpp")):
        matches = [split_windows(line) for line in commands if " -c " in line and line.replace("\\", "/").endswith("/src/" + filename)]
        require(len(matches) == 1, f"{filename} 编译命令不唯一")
        compile_source[unit] = matches[0]
        require(resolve(matches[0][0], build_dir) == compiler, "翻译单元的编译器不同于冻结编译器")
    links = [line for line in commands if " -o solver.exe " in line]
    require(len(links) == 1, "正式链接命令不唯一")
    outer = split_windows(links[0])
    require(len(outer) == 3 and outer[1].lower() == "/c" and Path(outer[0]).name.lower() == "cmd.exe", "链接包装命令格式改变")
    inner = outer[2]
    require(inner.startswith("cd . && ") and inner.endswith(" && cd ."), "链接工作目录包装不符")
    original_link = split_windows(inner[len("cd . && "):-len(" && cd .")])
    require(resolve(original_link[0], build_dir) == compiler, "链接编译器与冻结身份不符")
    explicit_original = [resolve(value, build_dir) for value in original_link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    regularized = build_dir / "CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj"
    protected = file_hashes([production, compiler, ninja, identity_path, archive, Path(__file__).resolve(),
                            build_dir / "build.ninja", build_dir / "CMakeFiles/rules.ninja", regularized,
                            *explicit_original, *[resolve(cmd[cmd.index("-o") + 1], build_dir) for cmd in compile_source.values()]])
    output.mkdir(parents=True)
    save(output / "static-check.json", checks)
    (output / "ninja-commands.txt").write_text(command_text, encoding="utf-8")
    frozen = output / "frozen"
    with zipfile.ZipFile(archive) as source:
        require(all((frozen / name).resolve().is_relative_to(frozen.resolve()) for name in source.namelist()), "源码归档包含越界路径")
        source.extractall(frozen)
    require_hashes({str(frozen / name): value for name, value in locked["files"].items()})
    (output / "v47-profile.hpp").write_text(HEADER, encoding="utf-8")
    transforms, compile_commands, objects = {}, {}, {}
    for unit, filename in (("ipm", "ipm.cpp"), ("normal", "normal_cholesky.cpp")):
        original = frozen / "src" / filename
        patched, changes = transform(original.read_text(encoding="utf-8"), unit)
        destination = output / (unit + "-profile.cpp")
        destination.write_text(patched, encoding="utf-8")
        objects[unit] = output / (unit + "-profile.obj")
        compile_commands[unit] = compile_arguments(compile_source[unit], destination, objects[unit], frozen, root)
        transforms[unit] = {"source": str(original), "source_sha256": sha(original), "patched": str(destination),
                            "patched_sha256": sha(destination), "changes": changes, "reversible": True}
    save(output / "transforms.json", transforms)
    executable = output / "solver-profile.exe"
    link = []
    replacements = {"libsolver_ipm.a": [str(objects["ipm"]), str(regularized)],
                    "libsolver_normal_cholesky.a": [str(objects["normal"])]}
    replaced = dict.fromkeys(replacements, 0)
    for value in original_link:
        if value in replacements:
            link.extend(replacements[value])
            replaced[value] += 1
        elif value.startswith("-Wl,--out-implib,"):
            continue
        else:
            link.append(value)
    require(all(value == 1 for value in replaced.values()), "独立对象链接替换不唯一")
    require(link.count("-o") == 1, "链接输出选项不唯一")
    link[link.index("-o") + 1] = str(executable)
    # 当前 Windows PE 链接器不支持 GNU 的 -t；仅核验显式对象与库输入。
    plan = {"query": query, "original_compile": compile_source, "original_link": original_link,
            "compile": compile_commands, "link": link, "cwd": str(build_dir)}
    save(output / "commands.json", plan)
    compile_inputs = []
    for unit, command in compile_commands.items():
        execute(command, build_dir, output / (unit + "-compile.json"))
        compile_inputs.extend(dependency_paths(Path(str(objects[unit]) + ".d"), build_dir))
    compile_hashes = file_hashes(compile_inputs)
    explicit_final = [resolve(value, build_dir) for value in link if not value.startswith("-") and Path(value).suffix.lower() in (".a", ".lib", ".o", ".obj")]
    explicit_before = file_hashes(explicit_final)
    execute(link, build_dir, output / "link.json")
    # 双翻译单元小程序只检验计时，不调用任何 LP 求解代码。
    for filename, text in (("self-main.cpp", SELF_MAIN), ("self-cross.cpp", SELF_CROSS)):
        (output / filename).write_text(text, encoding="utf-8")
    self_executable = output / "profile-self-test.exe"
    self_command = [str(compiler), "-O3", "-DNDEBUG", "-static", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic",
                    "-DSOLVER_V47_PROFILE_SELFTEST", str(output / "self-main.cpp"), str(output / "self-cross.cpp"),
                    "-o", str(self_executable)]
    execute(self_command, build_dir, output / "self-build.json")
    for value in ("0", "1"):
        environment = os.environ.copy()
        environment[ENVIRONMENT] = value
        row = execute([str(self_executable)], root, output / ("self-test-" + value + ".json"), environment)
        require(json.loads(row["stdout"]) == {"passed": True, "enabled": value == "1"}, "计时自测失败")
        reports = [json.loads(line[len(PREFIX):]) for line in row["stderr"].splitlines() if line.startswith(PREFIX)]
        require(len(reports) == int(value), "关闭诊断仍输出报告或开启报告数量不对")
        if reports:
            require(reports[0]["conserved"] and reports[0]["stack_empty"] and reports[0]["counts_closed"], "跨翻译单元计时不守恒")
    require_hashes(protected)
    require_hashes(current_sources)
    require_hashes(current_libraries)
    require_hashes(explicit_before)
    require_hashes(compile_hashes)
    outputs = [executable, self_executable, *objects.values(), output / "v47-profile.hpp", output / "transforms.json",
               output / "commands.json", output / "static-check.json", output / "self-main.cpp", output / "self-cross.cpp",
               *output.glob("*-profile.cpp"), *output.glob("*-compile.json"), output / "link.json", output / "self-build.json",
               output / "self-test-0.json", output / "self-test-1.json"]
    manifest = {"schema": 1, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "diagnostic_only": True, "prefix": PREFIX, "environment": ENVIRONMENT, "stages": list(STAGES),
                "builder": str(Path(__file__).resolve()), "builder_sha256": sha(__file__),
                "production_executable": str(production), "production_solver_sha256": sha(production),
                "executable": str(executable), "executable_sha256": sha(executable),
                "v46_identity": {"path": str(identity_path), "sha256": sha(identity_path)},
                "frozen_archive": {"path": str(archive), "sha256": sha(archive)}, "frozen_files": locked["files"],
                "protected_before_and_after": protected, "link_explicit_before_and_after": explicit_before,
                "implicit_link_inputs_verified": False, "compile_inputs_after_and_final": compile_hashes,
                "outputs": file_hashes(outputs), "commands": plan, "self_test_command": self_command,
                "scope": "仅阶段成本诊断；completed 表示正常退出，可能为 false 或失败状态；unwound 表示异常传播退出。",
                "identity_limit": "显式链接输入核对链接前后；Windows PE链接器不支持GNU -t，本轮未逐项核验隐式运行库。编译依赖在生成依赖文件后读取哈希并于结束复核，不声称做过其事前哈希。",
                "timing_limit": "包含插桩成本；normal_setup含结构构造，fallback适配器含内部检查；不作为性能验收。",
                "clock_contract": "所有作用域完成析构后 clock_reads=2*sum(started)，关闭时为0且无报告；报告自身不读时钟。"}
    save(output / "build-manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "build-manifest.json"), "executable": str(executable),
                      "executable_sha256": sha(executable), "self_test_passed": True}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-check", "build"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "self-check":
        print(json.dumps(static_check(root), ensure_ascii=False))
    else:
        build(root, (args.output or root / ".tools/v47-profile").resolve())


if __name__ == "__main__":
    main()
