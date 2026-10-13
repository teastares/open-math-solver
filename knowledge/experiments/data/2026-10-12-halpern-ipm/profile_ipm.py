"""从现有 Ninja 构建生成独立插桩程序；不修改生产源码、对象或可执行文件。"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


PROFILER = r'''
#include <array>
#include <cstdio>
namespace solver_ipm_profile {
using Clock = std::chrono::steady_clock;
struct Entry { double inclusive = 0, exclusive = 0; unsigned long long calls = 0; };
constexpr const char* names[] = {"total", "canonicalize", "symbolic", "factor",
  "solve", "krylov", "direct", "solve_factored", "kkt_product", "assess",
  "direction", "initialize", "run_ipm", "inexact_check", "factor_kernel"};
struct Scope;
inline thread_local std::array<Entry, 15> entries{};
inline thread_local Scope* parent = nullptr;
struct Scope {
  int index;
  Scope* previous;
  Clock::time_point start;
  double children = 0;
  explicit Scope(int value) : index(value), previous(parent), start(Clock::now()) { parent = this; }
  ~Scope() {
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    entries[index].inclusive += seconds;
    entries[index].exclusive += seconds - children;
    ++entries[index].calls;
    parent = previous;
    if (previous) previous->children += seconds;
  }
};
template<class Function> auto factor_call(Function&& function) {
  Scope scope(14);
  return function();
}
struct Report {
  Report() { entries = {}; }
  ~Report() {
    std::fputs("IPM_PROFILE {", stderr);
    for (int i = 0; i < 15; ++i) {
      std::fprintf(stderr, "%s\"%s\":{\"inclusive\":%.17g,\"exclusive\":%.17g,\"calls\":%llu}",
          i ? "," : "", names[i], entries[i].inclusive, entries[i].exclusive, entries[i].calls);
    }
    std::fputs("}\n", stderr);
  }
};
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def insert_scope(source, pattern, index):
    matches = list(re.finditer(pattern, source))
    if len(matches) != 1:
        raise RuntimeError(f"插桩锚点数量异常：{pattern}：{len(matches)}")
    position = matches[0].end()
    addition = f"\n  solver_ipm_profile::Scope profile_scope({index});"
    if index == 0:
        addition = "\n  solver_ipm_profile::Report profile_report;" + addition
    return source[:position] + addition + source[position:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="同时输出已有逐迭代诊断；打印成本不用于性能结论")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise RuntimeError("诊断输出必须是工作区中尚不存在的目录")
    source_path = root / "src/ipm.cpp"
    production = root / "build/release/solver.exe"
    identity = {str(p.relative_to(root)): sha(p) for p in (source_path, production)}
    source = source_path.read_text(encoding="utf-8")
    source = source.replace("namespace solver {", PROFILER + "\nnamespace solver {", 1)
    patterns = [
        r"SolveResult solve_ipm\([^{}]+\) \{",
        r"ConeModel canonicalize\([^{}]+\) \{",
        r"KktFactor\([^{}]+factor_dimension_\(model.costs.size\(\)\) \{",
        r"void factor\([^{}]+\) \{",
        r"std::vector<double> solve\([^{}]+\) const \{",
        r"std::vector<double> krylov_refine\([^{}]+\) const \{",
        r"std::vector<double> direct\([^{}]+\) const \{",
        r"void solve_factored\([^{}]+\) const \{",
        r"void unregularized_residual\([^{}]+\) const \{",
        r"bool assess\([^{}]+\) \{",
        r"Point direction\([^{}]+\) \{",
        r"void initialize\([^{}]+\) \{",
        r"void run_ipm\([^{}]+\) \{",
        r"bool accept_inexact_direction\([^{}]+\) const \{",
    ]
    for index, pattern in enumerate(patterns):
        source = insert_scope(source, pattern, index)
    for function in ("QDLDL_factor", "detail::solver_qdldl_extended_factor",
                     "detail::factor_regularized_ldl", "normal_cholesky_->factor",
                     "multifrontal_ldl_->factor", "sparse_lu_->factor"):
        start = source.index(function + "(")
        position = start + len(function)
        level = 0
        while position < len(source):
            if source[position] == "(":
                level += 1
            elif source[position] == ")":
                level -= 1
                if level == 0:
                    position += 1
                    break
            position += 1
        expression = source[start:position]
        source = source[:start] + "solver_ipm_profile::factor_call([&] { return " + expression + "; })" + source[position:]
    if args.trace:
        anchor = "const double norm = norm_inf(residual);"
        if source.count(anchor) != 1:
            raise RuntimeError("分块残差诊断锚点不唯一")
        source = source.replace(anchor, anchor + r'''
      const auto trace_norm = [](const std::vector<double>& values, std::size_t begin, std::size_t end) {
        double maximum = 0;
        for (std::size_t i = begin; i < end; ++i) maximum = std::max(maximum, std::abs(values[i]));
        return maximum;
      };
      const auto trace_n = model_.costs.size();
      std::cerr << "残差分块 " << *result.ipm_iterations << ' ' << iteration << ' '
                << trace_norm(rhs, 0, trace_n) << ' ' << trace_norm(rhs, trace_n, rhs.size()) << ' '
                << trace_norm(residual, 0, trace_n) << ' ' << trace_norm(residual, trace_n, residual.size())
                << ' ' << tolerance << '\n';
''')
    output.mkdir(parents=True)
    instrumented = output / "ipm-profile.cpp"
    instrumented.write_text(source, encoding="utf-8")
    build = root / "build/release"
    commands = subprocess.check_output(
        [root / ".tools/ninja-1.13.2/ninja.exe", "-C", build, "-t", "commands", "solver.exe"],
        text=True, encoding="utf-8").splitlines()
    compile_commands = [c for c in commands if " -c " in c and c.endswith("/src/ipm.cpp")]
    link_commands = [c for c in commands if " -o solver.exe " in c]
    if len(compile_commands) != 1 or len(link_commands) != 1:
        raise RuntimeError("Ninja 编译或链接命令不唯一")
    object_path = output / "ipm-profile.obj"
    executable = output / "solver-profile.exe"
    compile_command = compile_commands[0]
    compile_command = compile_command[:compile_command.index(" -MD ")] + (
        f' -o "{object_path}" -c "{instrumented}"')
    if args.trace:
        compile_command += " -DSOLVER_IPM_TRACE"
    link_command = link_commands[0].split("cd . && ", 1)[1].rsplit(" && cd .", 1)[0]
    link_command = link_command.replace(" -o solver.exe ", f' -o "{executable}" ')
    link_command = link_command.replace("-Wl,--out-implib,libsolver.dll.a ", "")
    link_command = link_command.replace("libsolver_ipm.a", (
        f'"{object_path}" CMakeFiles/solver_ipm.dir/src/regularized_ldl.cpp.obj'))
    # Windows 原生命令行交给 CreateProcess；不经过命令解释器，也不执行删除或改写生产产物。
    for command in (compile_command, link_command):
        subprocess.run(command, cwd=build, check=True)
    for path, expected in identity.items():
        if sha(root / path) != expected:
            raise RuntimeError(f"生产文件意外变化：{path}")
    manifest = {"production": identity, "script_sha256": sha(Path(__file__)),
                "instrumented_sha256": sha(instrumented), "solver_sha256": sha(executable),
                "commands": [compile_command, link_command],
                "trace": args.trace,
                "scope": "阶段包含/独占墙钟诊断，含插桩成本，不用于性能验收"}
    (output / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(executable)


if __name__ == "__main__":
    main()
