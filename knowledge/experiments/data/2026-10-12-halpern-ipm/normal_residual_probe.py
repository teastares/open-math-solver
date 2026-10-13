"""构建独立 Schur/完整 KKT 残差诊断程序，不改生产源代码或求解接受规则。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


DIRECT_PROBE = r'''
  void diagnostic_direct(const std::vector<double>& rhs, const std::vector<double>& point) const {
    const auto n = model_.costs.size();
    std::vector<long double> product(dimension_, 0);
    for (std::size_t col = 0; col < n; ++col) {
      for (auto k = model_.starts[col]; k < model_.starts[col + 1]; ++k) {
        const auto p = static_cast<std::size_t>(k), row = static_cast<std::size_t>(model_.rows[p]);
        const long double a = model_.values[p];
        product[col] += a * point[n + row];
        product[n + row] += a * point[col];
      }
    }
    std::array<long double, 2> rhs_norm{}, zero_error{}, regularized_error{}, shift{}, point_norm{};
    for (std::size_t i = 0; i < dimension_; ++i) {
      const auto block = i < n ? 0u : 1u;
      const long double unregularized = i < n ? product[i] :
          product[i] - static_cast<long double>(diagonal_h_[i - n]) * point[i];
      const long double change = i < n ? static_cast<long double>(diagnostic_regularization_) * point[i] :
          -(static_cast<long double>(regularized_h_[i - n]) - diagonal_h_[i - n]) * point[i];
      rhs_norm[block] = std::max(rhs_norm[block], std::abs(static_cast<long double>(rhs[i])));
      zero_error[block] = std::max(zero_error[block], std::abs(static_cast<long double>(rhs[i]) - unregularized));
      regularized_error[block] = std::max(regularized_error[block], std::abs(static_cast<long double>(rhs[i]) - unregularized - change));
      shift[block] = std::max(shift[block], std::abs(change));
      point_norm[block] = std::max(point_norm[block], std::abs(static_cast<long double>(point[i])));
    }
    std::fprintf(stderr, "DIRECT_REGULARIZED {\"format_sentinel\":1.25,\"regularization\":%.17g", diagnostic_regularization_);
    const auto emit = [](const char* name, long double value) {
      const double shown = static_cast<double>(value);
      std::fprintf(stderr, ",\"%s\":", name);
      if (std::isfinite(shown) && (value == 0 || shown != 0)) std::fprintf(stderr, "%.17g", shown);
      else std::fputs("null", stderr);
    };
    emit("rhs_dual", rhs_norm[0]); emit("rhs_primal", rhs_norm[1]);
    emit("k0_dual", zero_error[0]); emit("k0_primal", zero_error[1]);
    emit("kreg_dual", regularized_error[0]); emit("kreg_primal", regularized_error[1]);
    emit("shift_dual", shift[0]); emit("shift_primal", shift[1]);
    emit("dx", point_norm[0]); emit("dz", point_norm[1]);
    std::fputs("}\n", stderr);
  }
'''


PROBE = r'''
  // 所有额外乘积只用于诊断；不会反馈给因子、方向或接受条件。
  std::vector<long double> formed(retained, 0), rounded(retained, 0), composed(retained, 0);
  std::vector<long double> factored(retained, 0), factor_abs(retained, 0);
  for (Index col = 0; col < state.matrix.cols(); ++col) {
    for (Matrix::InnerIterator entry(state.matrix, col); entry; ++entry) {
      const auto i = static_cast<std::size_t>(entry.row()), j = static_cast<std::size_t>(col);
      const auto k = static_cast<std::size_t>(&entry.value() - state.matrix.valuePtr());
      const long double s = state.sums[k], d = static_cast<double>(s), a = entry.value();
      formed[i] += s * permuted[col];
      rounded[i] += d * permuted[col];
      factored[i] += a * diagnostic_factor_solution[j];
      factor_abs[i] += std::abs(a * diagnostic_factor_solution[j]);
      if (i != j) {
        formed[j] += s * permuted[entry.row()];
        rounded[j] += d * permuted[entry.row()];
        factored[j] += a * diagnostic_factor_solution[i];
        factor_abs[j] += std::abs(a * diagnostic_factor_solution[i]);
      }
    }
  }
  std::vector<long double> kkt(solution.size(), 0);
  long double cancellation = 0;
  for (std::size_t row = 0; row < retained; ++row) {
    const auto coordinate = static_cast<std::size_t>(state.dual_coordinates[row]);
    const long double d = state.input_values[static_cast<std::size_t>(state.dual_diagonal[row])];
    composed[row] = -d * solution[coordinate];
    kkt[coordinate] = d * solution[coordinate];
  }
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    const auto coordinate = static_cast<std::size_t>(state.primal_coordinates[col]);
    const long double b = state.input_values[static_cast<std::size_t>(state.primal_diagonal[col])];
    long double dot = 0, absolute = std::abs(static_cast<long double>(rhs[coordinate]));
    kkt[coordinate] = b * solution[coordinate];
    for (const auto& entry : state.columns[col]) {
      const auto dual = static_cast<std::size_t>(state.dual_coordinates[static_cast<std::size_t>(entry.row)]);
      const long double g = state.input_values[static_cast<std::size_t>(entry.source)];
      dot += g * solution[dual];
      absolute += std::abs(g * solution[dual]);
      kkt[coordinate] += g * solution[dual];
      kkt[dual] += g * solution[coordinate];
    }
    const long double denominator = std::abs(static_cast<long double>(rhs[coordinate]) - dot);
    cancellation = std::max(cancellation, absolute / std::max(denominator, std::numeric_limits<long double>::min()));
    for (const auto& entry : state.columns[col]) {
      composed[static_cast<std::size_t>(entry.row)] +=
          state.input_values[static_cast<std::size_t>(entry.source)] * (dot / b);
    }
  }
  long double formed_error = 0, rounded_error = 0, composed_error = 0, assembly_effect = 0, cast_effect = 0;
  long double factor_error = 0, factor_berr = 0, factor_rhs_norm = 0, schur_rhs_norm = 0;
  long double primal_error = 0, dual_error = 0, primal_rhs = 0, dual_rhs = 0;
  for (std::size_t row = 0; row < retained; ++row) {
    const auto p = static_cast<std::size_t>(state.inverse[row]);
    const long double q = reduced_rhs[row], scaled_q = diagnostic_factor_rhs[static_cast<Eigen::Index>(p)];
    formed_error = std::max(formed_error, std::abs(q - formed[p]));
    rounded_error = std::max(rounded_error, std::abs(q - rounded[p]));
    composed_error = std::max(composed_error, std::abs(q - composed[row]));
    assembly_effect = std::max(assembly_effect, std::abs(formed[p] - composed[row]));
    cast_effect = std::max(cast_effect, std::abs(formed[p] - rounded[p]));
    const long double error = std::abs(scaled_q - factored[p]);
    const long double denominator = std::abs(scaled_q) + factor_abs[p];
    factor_error = std::max(factor_error, error);
    factor_berr = std::max(factor_berr, error == 0 ? 0 : error / denominator);
    factor_rhs_norm = std::max(factor_rhs_norm, std::abs(scaled_q));
    schur_rhs_norm = std::max(schur_rhs_norm, std::abs(q));
  }
  for (Index coordinate : state.primal_coordinates) {
    const auto i = static_cast<std::size_t>(coordinate);
    primal_error = std::max(primal_error, std::abs(static_cast<long double>(rhs[i]) - kkt[i]));
    primal_rhs = std::max(primal_rhs, std::abs(static_cast<long double>(rhs[i])));
  }
  for (Index coordinate : state.dual_coordinates) {
    const auto i = static_cast<std::size_t>(coordinate);
    dual_error = std::max(dual_error, std::abs(static_cast<long double>(rhs[i]) - kkt[i]));
    dual_rhs = std::max(dual_rhs, std::abs(static_cast<long double>(rhs[i])));
  }
  std::fprintf(stderr, "NORMAL_RESIDUAL {\"factor\":%llu,\"solve\":%llu,\"multifrontal\":%s,\"modified_pivots\":%lld",
      state.diagnostic_factor, ++state.diagnostic_solves, state.multifrontal ? "true" : "false",
      static_cast<long long>(state.modified_pivots));
  const auto emit = [](const char* name, long double value) {
    std::fprintf(stderr, ",\"%s\":", name);
    const double displayed = static_cast<double>(value);
    if (std::isfinite(value) && std::isfinite(displayed) && (value == 0 || displayed != 0))
      std::fprintf(stderr, "%.17g", displayed);
    else std::fputs("null", stderr);
  };
  // Windows UCRT 的可变参数长双精度格式不可靠；运算仍用 long double，输出显式转 double。
  emit("format_sentinel", 1.25L);
  emit("schur_rhs", schur_rhs_norm); emit("formed_residual", formed_error);
  emit("rounded_residual", rounded_error); emit("composed_residual", composed_error);
  emit("assembly_product_difference", assembly_effect); emit("cast_product_difference", cast_effect);
  emit("factor_rhs", factor_rhs_norm); emit("factor_residual", factor_error); emit("factor_berr", factor_berr);
  emit("kkt_primal_residual", primal_error); emit("kkt_dual_residual", dual_error);
  emit("kkt_primal_rhs", primal_rhs); emit("kkt_dual_rhs", dual_rhs); emit("recovery_cancellation", cancellation);
  std::fputs("}\n", stderr);
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"诊断锚点数量异常：{old}：{text.count(old)}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise RuntimeError("诊断输出必须是工作区中尚不存在的目录")
    paths = [root / "src/ipm.cpp", root / "src/normal_cholesky.cpp", root / "build/release/solver.exe"]
    identity = {str(p.relative_to(root)): sha(p) for p in paths}
    output.mkdir(parents=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("profile_ipm.py")),
                    "--root", str(root), "--output", str(output / "ipm"), "--trace"], check=True)
    source = paths[1].read_text(encoding="utf-8")
    source = "#include <cstdio>\n" + source
    source = replace_once(source, "  bool valid = false;", "  unsigned long long diagnostic_factor = 0;\n  mutable unsigned long long diagnostic_solves = 0;\n  bool valid = false;")
    source = replace_once(source, "  auto& state = *impl_;", "  auto& state = *impl_;\n  ++state.diagnostic_factor;\n  state.diagnostic_solves = 0;")
    start = source.index("bool NormalCholesky::solve(")
    before, solve = source[:start], source[start:]
    solve = replace_once(solve, "  if (retained != 0) {", "  const Eigen::VectorXd diagnostic_factor_rhs = permuted;\n  std::vector<double> diagnostic_factor_solution;\n  if (retained != 0) {")
    anchor = "      if (!state.multifrontal->solve(solution, budget_check)) return false;"
    solve = replace_once(solve, anchor, anchor + "\n      diagnostic_factor_solution = solution;")
    anchor = "      permuted = solution;"
    solve = replace_once(solve, anchor, "      diagnostic_factor_solution.assign(solution.data(), solution.data() + solution.size());\n" + anchor)
    solve = replace_once(solve, "  rhs = std::move(solution);", PROBE + "\n  rhs = std::move(solution);")
    instrumented = output / "normal-probe.cpp"
    instrumented.write_text(before + solve, encoding="utf-8")
    build = root / "build/release"
    commands = subprocess.check_output([root / ".tools/ninja-1.13.2/ninja.exe", "-C", build, "-t", "commands", "solver.exe"], text=True, encoding="utf-8").splitlines()
    candidates = [c for c in commands if " -c " in c and c.endswith("/src/normal_cholesky.cpp")]
    if len(candidates) != 1:
        raise RuntimeError("正规方程编译命令不唯一")
    obj, executable = output / "normal-probe.obj", output / "solver-probe.exe"
    compile_command = candidates[0].split(" -MD ", 1)[0] + f' -o "{obj}" -c "{instrumented}"'
    base = json.loads((output / "ipm/build-manifest.json").read_text(encoding="utf-8"))
    ipm_source = (output / "ipm/ipm-profile.cpp").read_text(encoding="utf-8")
    ipm_source = replace_once(ipm_source, "  const ConeModel& model_;", "  double diagnostic_regularization_ = 0;\n  const ConeModel& model_;")
    anchor = "      std::vector<long double> primal_diagonal(model_.costs.size(), regularization);"
    ipm_source = replace_once(ipm_source, anchor, "      diagnostic_regularization_ = regularization;\n" + anchor)
    begin, end = ipm_source.index("  std::vector<double> direct("), ipm_source.index("  void unregularized_residual(")
    direct = ipm_source[begin:end]
    if direct.count("return answer;") != 2:
        raise RuntimeError("完整直接方向的返回锚点数量异常")
    direct = direct.replace("return answer;", "diagnostic_direct(rhs, answer);\n    return answer;")
    ipm_source = ipm_source[:begin] + direct + DIRECT_PROBE + ipm_source[end:]
    ipm_instrumented, ipm_obj = output / "ipm-direct-probe.cpp", output / "ipm-direct-probe.obj"
    ipm_instrumented.write_text(ipm_source, encoding="utf-8")
    compile_ipm = base["commands"][0].replace(str(output / "ipm/ipm-profile.cpp"), str(ipm_instrumented))
    compile_ipm = compile_ipm.replace(str(output / "ipm/ipm-profile.obj"), str(ipm_obj))
    link_command = base["commands"][-1]
    link_command = replace_once(link_command, str(output / "ipm/ipm-profile.obj"), str(ipm_obj))
    link_command = replace_once(link_command, str(output / "ipm/solver-profile.exe"), str(executable))
    link_command = replace_once(link_command, "libsolver_normal_cholesky.a", f'"{obj}"')
    for command in (compile_ipm, compile_command, link_command):
        subprocess.run(command, cwd=build, check=True)
    for path, expected in identity.items():
        if sha(root / path) != expected:
            raise RuntimeError(f"生产文件意外变化：{path}")
    manifest = {"production": identity, "script_sha256": sha(Path(__file__)),
                "instrumented_sha256": sha(instrumented), "solver_sha256": sha(executable),
                "ipm_direct_sha256": sha(ipm_instrumented),
                "commands": [compile_ipm, compile_command, link_command], "ipm_build": base,
                "scope": "逐次 Schur 与实际正则约化 KKT 残差诊断；包含额外乘积/复制/打印，不用于性能结论"}
    (output / "build-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(executable)


if __name__ == "__main__":
    main()
