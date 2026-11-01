"""v83：仅在独立候选中配对齐次常量与预测的首次回代，保留原逐右端验收。"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENT = "SOLVER_V83_PAIR_RHS"
PREFIX = "V83_PAIR "
COUNTERS = ("pairs_started", "pairs_completed", "normal_calls_started", "normal_calls_completed",
    "finish0_started", "finish0_completed", "finish1_started", "finish1_completed")


HEADER = r'''#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace solver_v83_pair {
struct Stats {
  std::int64_t pairs_started = 0, pairs_completed = 0;
  std::int64_t normal_calls_started = 0, normal_calls_completed = 0;
  std::int64_t finish0_started = 0, finish0_completed = 0;
  std::int64_t finish1_started = 0, finish1_completed = 0;
};
inline thread_local bool enabled = false;
inline thread_local Stats stats{};

// pair在入口预算检查后启动，只有两次原验收都返回才完成。
// normal仅记成对Normal API；正常返回false仍算完成，抛异常不算完成。
// finish按常量、预测依次启动；完成仅表示各自原残差/IR/Krylov正常返回。
struct Report {
  Report() noexcept {
    const char* value = std::getenv("SOLVER_V83_PAIR_RHS");
    enabled = value && std::strcmp(value, "1") == 0;
    stats = {};
  }
  ~Report() noexcept {
    const auto& s = stats;
    std::fprintf(stderr,
        "V83_PAIR {\"schema\":1,\"enabled\":%s,\"pairs_started\":%lld,\"pairs_completed\":%lld,"
        "\"normal_calls_started\":%lld,\"normal_calls_completed\":%lld,"
        "\"finish0_started\":%lld,\"finish0_completed\":%lld,"
        "\"finish1_started\":%lld,\"finish1_completed\":%lld}\n",
        enabled ? "true" : "false", static_cast<long long>(s.pairs_started),
        static_cast<long long>(s.pairs_completed), static_cast<long long>(s.normal_calls_started),
        static_cast<long long>(s.normal_calls_completed), static_cast<long long>(s.finish0_started),
        static_cast<long long>(s.finish0_completed), static_cast<long long>(s.finish1_started),
        static_cast<long long>(s.finish1_completed));
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v83_pair
'''


PAIR_PUBLIC = r'''  bool supports_native_pair() const {
    return normal_cholesky_ && normal_cholesky_->supports_native_pair();
  }

  std::array<std::vector<double>, 2> solve_pair(const std::vector<double>& left,
      const std::vector<double>& right, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    if (!solver_v83_pair::enabled || options.ipm.linear_accuracy == "homogeneous_scaled" ||
        !supports_native_pair()) {
      // 非原生路径完整保留第0右端的验收和精修，再启动第1右端。
      auto first = solve(left, options, start, result);
      auto second = solve(right, options, start, result);
      return {std::move(first), std::move(second)};
    }
    if (left.size() != dimension_ || right.size() != dimension_)
      throw std::invalid_argument("成对KKT右端维数不匹配");
    check_budget(options, start);
    ++solver_v83_pair::stats.pairs_started;
    auto answers = direct_pair(left, right, options, start, result);
    ++solver_v83_pair::stats.finish0_started;
    answers[0] = finish(left, std::move(answers[0]), options, start, result);
    ++solver_v83_pair::stats.finish0_completed;
    ++solver_v83_pair::stats.finish1_started;
    answers[1] = finish(right, std::move(answers[1]), options, start, result);
    ++solver_v83_pair::stats.finish1_completed;
    ++solver_v83_pair::stats.pairs_completed;
    return answers;
  }

'''


PAIR_DIRECT = r'''  std::array<std::vector<double>, 2> direct_pair(const std::vector<double>& left,
      const std::vector<double>& right, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    const std::vector<double>* rhs[2] = {&left, &right};
    std::array<std::vector<double>, 2> permuted;
    for (auto& column : permuted) column.resize(factor_dimension_);
    if (factor_dimension_ == dimension_) {
      for (std::size_t i = 0; i < dimension_; ++i) {
        const auto position = inverse_[i];
        for (std::size_t q = 0; q < 2; ++q)
          permuted[q][position] = equilibration_.empty() ? (*rhs[q])[i] : scaled((*rhs[q])[i], equilibration_[position]);
      }
    } else {
      const auto n = model_.costs.size();
      std::array<std::vector<long double>, 2> reduced_rhs;
      for (auto& column : reduced_rhs) column.assign(factor_dimension_, 0);
      for (std::size_t col = 0; col < n; ++col)
        for (std::size_t q = 0; q < 2; ++q) reduced_rhs[q][col] = (*rhs[q])[col];
      for (std::size_t row = 0; row < model_.rhs.size(); ++row) {
        if (singleton_column_[row] >= 0) {
          const auto col = static_cast<std::size_t>(singleton_column_[row]);
          for (std::size_t q = 0; q < 2; ++q)
            reduced_rhs[q][col] += static_cast<long double>(singleton_value_[row]) * (*rhs[q])[n + row] / regularized_h_[row];
        } else {
          for (std::size_t q = 0; q < 2; ++q)
            reduced_rhs[q][static_cast<std::size_t>(retained_index_[row])] = (*rhs[q])[n + row];
        }
      }
      for (std::size_t i = 0; i < factor_dimension_; ++i) {
        const auto position = inverse_[i];
        for (std::size_t q = 0; q < 2; ++q) {
          const double value = finite(reduced_rhs[q][i]);
          permuted[q][position] = equilibration_.empty() ? value : scaled(value, equilibration_[position]);
        }
      }
    }
    const auto collect = [&] {
      const auto statistics = normal_cholesky_->refinement_statistics();
      result.ipm_schur_residual_products = statistics.residual_products;
      result.ipm_schur_refinement_steps = statistics.refinement_steps;
      result.ipm_schur_refinement_acceptances = statistics.refinement_acceptances;
    };
    check_budget(options, start);
    ++solver_v83_pair::stats.normal_calls_started;
    bool success;
    try {
      success = normal_cholesky_->solve_pair(permuted, [&options, &start] { check_budget(options, start); });
      ++solver_v83_pair::stats.normal_calls_completed;
    } catch (...) {
      collect();
      throw;
    }
    collect();
    check_budget(options, start);
    if (!success) throw NumericalFailure("正规方程 Cholesky 成对方向恢复失败");
    // 两列都由自己的vector拥有；第0列精修再用单RHS缓存也不能覆盖第1列。
    std::array<std::vector<double>, 2> answers;
    for (auto& column : answers) column.resize(dimension_);
    if (factor_dimension_ == dimension_) {
      for (std::size_t i = 0; i < dimension_; ++i) {
        const auto position = inverse_[i];
        for (std::size_t q = 0; q < 2; ++q)
          answers[q][i] = equilibration_.empty() ? permuted[q][position] : scaled(permuted[q][position], equilibration_[position]);
      }
      return answers;
    }
    if (!equilibration_.empty()) {
      for (std::size_t i = 0; i < factor_dimension_; ++i)
        for (std::size_t q = 0; q < 2; ++q) permuted[q][i] = scaled(permuted[q][i], equilibration_[i]);
    }
    const auto n = model_.costs.size();
    for (std::size_t col = 0; col < n; ++col)
      for (std::size_t q = 0; q < 2; ++q) answers[q][col] = finite(permuted[q][inverse_[col]]);
    for (std::size_t row = 0; row < model_.rhs.size(); ++row) {
      if (singleton_column_[row] >= 0) {
        const auto col = static_cast<std::size_t>(singleton_column_[row]);
        for (std::size_t q = 0; q < 2; ++q)
          answers[q][n + row] = finite((static_cast<long double>(singleton_value_[row]) * answers[q][col] -
              (*rhs[q])[n + row]) / regularized_h_[row]);
      } else {
        for (std::size_t q = 0; q < 2; ++q)
          answers[q][n + row] = finite(permuted[q][inverse_[static_cast<std::size_t>(retained_index_[row])]]);
      }
    }
    return answers;
  }

'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(source):
    """唯一原文锚点逐项变换，并逐项逆放回原始v66文本。"""
    source = source.replace("\r\n", "\n")
    require("solver_v83_pair::" not in source and '"v83-pair-state.hpp"' not in source, "拒绝重复变换")
    original, changes = source, []

    def replace(label, before, after):
        nonlocal source
        require(before != after and source.count(before) == 1, label + " 缺失或重复锚点")
        source = source.replace(before, after, 1)
        changes.append({"label": label, "before": before, "after": after})

    replace("private_header", "namespace solver {\n", '#include "v83-pair-state.hpp"\n\nnamespace solver {\n')
    begin = source.index("  std::vector<double> solve(const std::vector<double>& rhs,")
    end = source.index("  Point solve_full_newton(", begin)
    before = source[begin:end]
    split = "    const double tolerance = 1e-12 + 1e-10 * norm_inf(rhs);\n"
    require(before.count(split) == 1, "原solve验收入口不唯一")
    initial, rest = before.split(split, 1)
    require(initial.endswith("    auto answer = direct(rhs, options, start, result);\n"), "原solve首次回代顺序变化")
    after = initial + "    return finish(rhs, std::move(answer), options, start, result, inexact, initialization);\n  }\n\n"
    after += "  std::vector<double> finish(const std::vector<double>& rhs, std::vector<double> answer,\n"
    after += "      const SolveOptions& options, Clock::time_point start, SolveResult& result,\n"
    after += "      const InexactDirectionContext* inexact = nullptr, bool initialization = false) const {\n"
    after += split + rest
    replace("finish_extraction", before, after)
    replace("pair_public", "  Point solve_full_newton(", PAIR_PUBLIC + "  Point solve_full_newton(")
    replace("pair_direct", "  void unregularized_residual(", PAIR_DIRECT + "  void unregularized_residual(")

    begin = source.index("HomogeneousDirection homogeneous_direction(const ConeModel& model,")
    end = source.index("HomogeneousDirection homogeneous_scaled_direction(", begin)
    before = source[begin:end]
    body_begin = before.index("  check_budget(options, start);\n")
    counter_begin = before.index("  ++*result.ipm_homogeneous_direction_solves;\n")
    tail_begin = before.index("  const auto budget = [&] { check_budget(options, start); };\n", counter_begin)
    require(before[counter_begin:tail_begin] == "  ++*result.ipm_homogeneous_direction_solves;\n"
        "  const auto u = factor.solve(rhs, options, start, result);\n", "齐次原首次求解语句变化")
    rhs_body, tail = before[body_begin:counter_begin], before[tail_begin:]
    after = "std::vector<double> homogeneous_direction_rhs(const ConeModel& model, const Point& point,\n"
    after += "    const std::vector<double>& primal_residual, const std::vector<double>& dual_residual,\n"
    after += "    double theta, const std::vector<double>& h, const SolveOptions& options, Clock::time_point start) {\n"
    after += rhs_body + "  return rhs;\n}\n\n"
    after += "HomogeneousDirection homogeneous_direction_from_solution(const ConeModel& model, const Point& point,\n"
    after += "    double tau, double kappa, long double gap_residual, double theta, const std::vector<double>& h,\n"
    after += "    long double h_tau, std::vector<double>& constant, const KktFactor& factor,\n"
    after += "    const SolveOptions& options, Clock::time_point start, SolveResult& result,\n"
    after += "    const std::vector<double>& u, detail::IpmHomogeneousRepairInfo* repair_info = nullptr) {\n"
    after += "  const auto n = point.x.size(), m = point.s.size();\n" + tail
    after += before[:body_begin]
    after += "  const auto rhs = homogeneous_direction_rhs(model, point, primal_residual, dual_residual, theta, h, options, start);\n"
    after += "  ++*result.ipm_homogeneous_direction_solves;\n  const auto u = factor.solve(rhs, options, start, result);\n"
    after += "  return homogeneous_direction_from_solution(model, point, tau, kappa, gap_residual, theta, h, h_tau,\n"
    after += "      constant, factor, options, start, result, u, repair_info);\n}\n\n"
    replace("homogeneous_helpers", before, after)

    begin = source.index("    std::vector<double> constant;\n    detail::IpmHomogeneousScaling scaling;")
    end = source.index("    const double affine_step = homogeneous_step(", begin)
    before = source[begin:end]
    after = before.replace("    std::vector<double> constant;\n", "    std::vector<double> constant;\n"
        "    const bool paired = solver_v83_pair::enabled && !scaled_direction && factor.supports_native_pair();\n", 1)
    require(after.count("    } else {\n      ++*result.ipm_homogeneous_constant_solves;") == 1, "主齐次常量分支不唯一")
    after = after.replace("    } else {\n      ++*result.ipm_homogeneous_constant_solves;",
        "    } else if (!paired) {\n      ++*result.ipm_homogeneous_constant_solves;", 1)
    affine_begin = after.index("    const auto affine = scaled_direction ?")
    affine = after[affine_begin:].replace("    const auto affine =", "      affine =", 1)
    after = after[:affine_begin] + "    HomogeneousDirection affine;\n    if (paired) {\n"
    after += "      const auto affine_rhs = homogeneous_direction_rhs(model, point, primal_residual, dual_residual, 1, h, options, start);\n"
    after += "      check_budget(options, start);\n      ++*result.ipm_homogeneous_constant_solves;\n"
    after += "      ++*result.ipm_homogeneous_direction_solves;\n"
    after += "      auto answers = factor.solve_pair(constant_rhs, affine_rhs, options, start, result);\n"
    after += "      constant = std::move(answers[0]);\n"
    after += "      affine = homogeneous_direction_from_solution(model, point, tau, kappa, gap_residual, 1, h,\n"
    after += "          affine_h_tau, constant, factor, options, start, result, answers[1], repair);\n"
    after += "    } else {\n" + affine + "    }\n"
    replace("main_constant_affine", before, after)
    replace("pair_report", "SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n",
        "SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n  solver_v83_pair::Report v83_report;\n")
    restored = source
    for item in reversed(changes):
        require(restored.count(item["after"]) == 1, "逆变换锚点不唯一：" + item["label"])
        restored = restored.replace(item["after"], item["before"], 1)
    require(restored == original, "变换不能逐字恢复v66来源")
    return source, changes


def static_check(root):
    """仅静态核对机械提取、唯一锚点与逆变换；不编译、不执行求解器。"""
    source = (Path(root) / "src/ipm.cpp").read_text(encoding="utf-8")
    patched, changes = transform(source)
    require(len(changes) == 7, "独立变换数量变化")
    for modified in (source.replace("namespace solver {", "namespace removed {", 1), source + source, patched):
        rejected = False
        try:
            transform(modified)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, "缺失、重复或已变换来源必须拒绝")
    require(patched.count("factor.solve_pair(constant_rhs, affine_rhs, options, start, result)") == 1,
        "主齐次流程必须实际调用成对回代")
    finish = next(item for item in changes if item["label"] == "finish_extraction")
    old_tail = finish["before"].split("    const double tolerance =", 1)[1]
    require(finish["after"].split("    const double tolerance =", 1)[1] == old_tail, "原验收循环必须逐字保留")
    sequence = ("++solver_v83_pair::stats.pairs_started;", "auto answers = direct_pair(",
        "++solver_v83_pair::stats.finish0_started;", "answers[0] = finish(left,",
        "++solver_v83_pair::stats.finish0_completed;", "++solver_v83_pair::stats.finish1_started;",
        "answers[1] = finish(right,", "++solver_v83_pair::stats.finish1_completed;",
        "++solver_v83_pair::stats.pairs_completed;")
    positions = [PAIR_PUBLIC.index(item) for item in sequence]
    require(positions == sorted(positions) and len(set(positions)) == len(positions), "两次原验收的次序必须固定")
    require(PAIR_PUBLIC.index("auto first = solve(left,") < PAIR_PUBLIC.index("auto second = solve(right,"),
        "非原生路径必须先完整第0右端再第1右端")
    return {"passed": True, "solver_runs": 0, "anchors": len(changes), "reversible": True,
        "source_text_sha256": digest(source), "patched_text_sha256": digest(patched),
        "header_sha256": digest(HEADER), "finish_loop_unchanged": True, "ordered_finishes": True,
        "environment": ENVIRONMENT, "prefix": PREFIX, "counter_fields": list(COUNTERS)}


if __name__ == "__main__":
    print(json.dumps(static_check(ROOT), ensure_ascii=False))
