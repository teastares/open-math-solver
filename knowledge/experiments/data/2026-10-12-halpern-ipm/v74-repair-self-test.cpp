#include "v70-basis-recovery.hpp"
#include "v74-basis-repair.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace repair = solver::v74_basis_repair;
using Index = repair::Index;
using Lu = solver::v70_basis::implementation::CheckedLu;

class ControlledLu : public Lu {
 public:
  enum class Mode { normal, half, fail, nonfinite, overflow };
  Mode mode = Mode::normal;
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
    if (mode == Mode::fail) throw std::runtime_error("解析求解失败");
    if (mode == Mode::nonfinite)
      return Eigen::VectorXd::Constant(rhs.size(), std::numeric_limits<double>::infinity());
    if (mode == Mode::overflow)
      return Eigen::VectorXd::Constant(rhs.size(), std::numeric_limits<double>::max());
    Eigen::VectorXd result = Lu::solve(rhs);
    if (mode == Mode::half) result *= 0.5;
    return result;
  }
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <=
      1e-12 * std::max(1.0, std::abs(expected)), message);
}

struct Fixture {
  std::vector<Index> starts, rows;
  std::vector<double> values, costs, rhs;
  std::vector<bool> inequality;
  repair::Input input() const { return {starts, rows, values, costs, rhs, inequality}; }
};

Fixture fixture(const std::vector<std::vector<double>>& matrix,
                std::vector<double> costs, std::vector<double> rhs,
                std::vector<bool> inequality) {
  Fixture result;
  result.costs = std::move(costs); result.rhs = std::move(rhs); result.inequality = std::move(inequality);
  result.starts.push_back(0);
  for (std::size_t col = 0; col < result.costs.size(); ++col) {
    for (std::size_t row = 0; row < matrix.size(); ++row) {
      if (matrix[row][col] == 0) continue;
      result.rows.push_back(static_cast<Index>(row)); result.values.push_back(matrix[row][col]);
    }
    result.starts.push_back(static_cast<Index>(result.values.size()));
  }
  return result;
}

template<class Factor>
struct BasicSession {
  const Fixture& f;
  std::vector<Index> selected;
  repair::Matrix basis;
  Factor lu;
  Eigen::VectorXd x, dual;
  BasicSession(const Fixture& input, std::vector<Index> rows)
      : f(input), selected(std::move(rows)), basis(input.costs.size(), input.costs.size()) {
    std::vector<Eigen::Triplet<double, Index>> entries;
    for (Index col = 0; col < static_cast<Index>(f.costs.size()); ++col) {
      for (Index k = f.starts[col]; k < f.starts[col + 1]; ++k) {
        for (Index position = 0; position < static_cast<Index>(selected.size()); ++position) {
          if (selected[position] == f.rows[k]) entries.emplace_back(position, col, f.values[k]);
        }
      }
    }
    basis.setFromTriplets(entries.begin(), entries.end()); basis.makeCompressed();
    lu.analyzePattern(basis); lu.factorize(basis);
    require(lu.factorization_ok() && lu.info() == Eigen::Success, "解析例初始基应可分解");
    Eigen::VectorXd rhs(f.costs.size());
    for (Index k = 0; k < static_cast<Index>(selected.size()); ++k) rhs[k] = f.rhs[selected[k]];
    x = lu.solve(rhs);
    for (Index k = 0; k < static_cast<Index>(selected.size()); ++k) rhs[k] = -f.costs[k];
    dual = lu.transpose().solve(rhs);
  }
  repair::Start start() const {
    return {selected, std::span<const double>(x.data(), static_cast<std::size_t>(x.size())),
        std::span<const double>(dual.data(), static_cast<std::size_t>(dual.size())),
        static_cast<Index>(lu.nnzL() + lu.nnzU())};
  }
  repair::Result run(const repair::Limits& limits = {}) {
    repair::Result result;
    repair::repair(f.input(), limits, start(), lu, {}, result);
    return result;
  }
};

using Session = BasicSession<Lu>;
using ControlledSession = BasicSession<ControlledLu>;

void check_candidate(const Fixture& f, const repair::Result& result) {
  require(result.candidate_available && result.status == "candidate" && result.stats.phase == "complete",
      "修复结果只标记待原模型验收的候选");
  std::vector<long double> activity(f.rhs.size(), 0);
  std::vector<long double> dual(f.rhs.size(), 0);
  for (std::size_t k = 0; k < result.selected_rows.size(); ++k) {
    dual[result.selected_rows[k]] = result.dual[k];
    if (f.inequality[result.selected_rows[k]]) require(result.dual[k] >= 0, "不等式乘子必须非负");
  }
  for (std::size_t col = 0; col < f.costs.size(); ++col) {
    long double transpose = 0;
    for (Index k = f.starts[col]; k < f.starts[col + 1]; ++k) {
      activity[f.rows[k]] += static_cast<long double>(f.values[k]) * result.x[col];
      transpose += static_cast<long double>(f.values[k]) * dual[f.rows[k]];
    }
    close(static_cast<double>(transpose), -f.costs[col], "恢复原成本后的对偶方程应成立");
  }
  for (std::size_t row = 0; row < f.rhs.size(); ++row) {
    const long double slack = static_cast<long double>(f.rhs[row]) - activity[row];
    if (f.inequality[row]) require(slack >= -1e-12L, "解析候选满足原不等式");
    else close(static_cast<double>(slack), 0, "解析候选满足全部原等式");
  }
  require(result.stats.iterations == result.stats.phase1_iterations + result.stats.phase2_iterations,
      "阶段迭代账本应相加一致");
  require(result.selected_rows == result.stats.final_selected_rows, "成功结果映射与账本一致");
}

Fixture two_phase_fixture() {
  return fixture({{1}, {1}, {-1}}, {1}, {3, 2, -1}, {true, true, true});
}

}  // 匿名命名空间

int main() {
  try {
    std::vector<std::string> names;
    {
      const auto f = fixture({{1}, {1}, {-1}}, {-1}, {2, 1, 0}, {true, true, true});
      Session session(f, {0}); const auto result = session.run(); check_candidate(f, result);
      close(result.x[0], 1, "原始阶段应从宽界移到可行紧界");
      require(result.stats.phase1_iterations == 1 && result.stats.phase2_iterations == 0 &&
          result.stats.pivots == 1, "一次原始阶段换基足够");
      names.emplace_back("不等式可行性恢复");
    }
    {
      const auto f = fixture({{2, 1}, {3, 4}, {1, 0}, {-1, 0}}, {-2, -1},
          {5, 15, 0, 1}, {true, true, true, true});
      Session session(f, {0, 1}); const auto result = session.run(); check_candidate(f, result);
      close(result.x[0], 0, "非对称基修复第一坐标"); close(result.x[1], 3.75, "非对称基修复第二坐标");
      require(result.selected_rows == std::vector<Index>({2, 1}), "行替换不能误用原变量置换");
      names.emplace_back("非对称基方向与原行映射");
    }
    {
      const auto f = fixture({{1}, {1}}, {1}, {2, 1}, {true, false});
      Session session(f, {0}); const auto result = session.run(); check_candidate(f, result);
      close(result.x[0], 1, "负等式slack应由绝对值目标修复"); close(result.dual[0], -1, "等式允许负乘子");
      require(result.selected_rows == std::vector<Index>({1}), "基外等式可作为离基slack对应的新选中行");
      names.emplace_back("基外等式负残差与负乘子");
    }
    {
      const auto f = fixture({{-1}, {1}}, {1}, {0, 1}, {true, false});
      Session session(f, {0}); const auto result = session.run(); check_candidate(f, result);
      close(result.x[0], 1, "正等式slack同样必须进入绝对值目标");
      names.emplace_back("基外等式正残差");
    }
    {
      const auto f = fixture({{1}, {1}}, {-1}, {1, 0}, {false, true});
      Session session(f, {0}); const auto initial = std::bit_cast<std::uint64_t>(session.x[0]);
      const auto result = session.run();
      require(!result.candidate_available && result.status == "no_entering" && result.stats.pivots == 0,
          "选中等式的固定slack不可进入，不能据此宣称原LP不可行");
      require(std::bit_cast<std::uint64_t>(session.x[0]) == initial, "拒绝修复不能改写初始点");
      names.emplace_back("选中等式进入限制及失败事务");
    }
    {
      const auto f = fixture({{1}, {1}, {-1}}, {1}, {0, -1, 0}, {true, true, true});
      Session session(f, {0}); const auto result = session.run();
      require(!result.candidate_available && result.status == "no_descent" &&
          result.stats.last_right_derivative == std::optional<double>(0), "零不等式处应补真实右导数");
      names.emplace_back("零不等式断点拒绝伪下降");
    }
    {
      const auto f = fixture({{1}, {1}, {1}}, {1}, {0, -1, 0}, {true, true, false});
      Session session(f, {0}); const auto result = session.run();
      require(!result.candidate_available && result.status == "no_descent" &&
          result.stats.last_right_derivative == std::optional<double>(0), "零等式处应使用方向绝对值");
      names.emplace_back("零等式断点拒绝伪下降");
    }
    {
      const auto f = fixture({{1}, {1}, {2}, {-1}}, {-1}, {2, 1, 2, 0}, {true, true, true, true});
      Session session(f, {0}); const auto result = session.run(); check_candidate(f, result);
      require(result.selected_rows == std::vector<Index>({2}) && result.stats.last_step == std::optional<double>(1),
          "同点断点应完整分组并选择较大数值主元");
      names.emplace_back("同点断点合并与离基选择");
    }
    {
      const auto f = two_phase_fixture(); Session session(f, {0}); const auto result = session.run();
      check_candidate(f, result); close(result.x[0], 1, "恢复原成本后应从上界继续走到下界");
      require(result.stats.phase1_iterations == 1 && result.stats.phase2_iterations == 1 &&
          result.stats.pivots == 2, "阶段切换不得保留辅助成本");
      names.emplace_back("阶段切换恢复原成本");
    }
    {
      const auto f = fixture({{1}, {-1}}, {1}, {0, -1e-16}, {true, true});
      Session session(f, {0}); const auto result = session.run();
      require(!result.candidate_available && result.status == "zero_step_stall" &&
          result.stats.roundoff_rows == 1 && result.stats.zero_steps == 1,
          "舍入守卫容纳的负slack仍须在第二阶段保守阻挡");
      require(result.stats.initial_phi && *result.stats.initial_phi > 0, "原始罚目标不能被舍入守卫清零");
      names.emplace_back("微小负slack的第二阶段零步");
    }
    {
      const auto f = fixture({{-1}}, {-1}, {0}, {true}); Session session(f, {0}); const auto result = session.run();
      require(!result.candidate_available && result.status == "no_blocker", "没有阻挡行只退出，不发布无界证书");
      names.emplace_back("无阻挡方向保守退出");
    }
    {
      const auto f = two_phase_fixture(); Session session(f, {0}); repair::Limits limits; limits.max_pivots = 1;
      const auto result = session.run(limits);
      require(!result.candidate_available && result.status == "pivot_limit" && result.stats.pivots == 1 &&
          result.x.empty() && session.x[0] == 3, "中间已换基仍不得提交不完整候选");
      names.emplace_back("换基上限与中间状态回退");
    }
    {
      const auto f = two_phase_fixture(); Session session(f, {0}); repair::Result result; bool interrupted = false;
      try {
        repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
          if (result.stats.pivots == 1) throw std::runtime_error("解析预算中断");
        }, result);
      } catch (const std::runtime_error&) { interrupted = true; }
      require(interrupted && result.status == "budget_exception" && !result.candidate_available &&
          result.stats.pivots == 1 && session.x[0] == 3, "预算异常保留真实部分账本及初始点");
      names.emplace_back("换基后预算中断");
    }
    {
      const auto f = fixture({{1}, {1}, {-1}}, {-1}, {2, 1, 0}, {true, true, true});
      Session session(f, {0}); const auto full = session.run(); check_candidate(f, full);
      repair::Limits exact; exact.max_repair_work = full.stats.work_units;
      const auto allowed = session.run(exact); check_candidate(f, allowed);
      --exact.max_repair_work; const auto stopped = session.run(exact);
      require(!stopped.candidate_available && stopped.status == "work_limit" &&
          stopped.stats.work_units <= exact.max_repair_work, "独立工作额度必须在访问前核验");
      names.emplace_back("工作额度精确边界");
    }
    {
      const auto f = fixture({{1}, {1}}, {-1}, {2, 1}, {true, true}); Session session(f, {0});
      repair::Limits limits; limits.max_factor_nonzeros = session.start().factor_nonzeros;
      const auto before = session.run(limits);
      require(before.status == "factor_limit" && before.stats.lu_factor_calls == 0, "累计因子额度含初始因子");
      ++limits.max_factor_nonzeros; const auto after = session.run(limits);
      require(!after.candidate_available && after.status == "factor_limit" && after.stats.lu_factor_calls == 1 &&
          after.stats.factor_nonzeros > limits.max_factor_nonzeros, "因子填充后验计数超额不得隐藏");
      names.emplace_back("累计因子填充边界");
    }
    {
      const auto f = fixture({{1}}, {-1}, {2}, {true}); Session session(f, {0}); session.x[0] += 1;
      const auto result = session.run();
      require(result.stats.initial_guard.complete && !result.stats.initial_guard.passed &&
          result.stats.initial_guard.failure_row == std::optional<Index>(0) &&
          result.stats.initial_guard.failure_basis_position == std::optional<Index>(0) &&
          result.stats.initial_guard.failure_abs_residual == std::optional<double>(1) &&
          result.stats.initial_guard.checked_rows == 1 && result.stats.initial_guard.violations == 1,
          "完整初始守卫应保留对应行与残差");
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        check_candidate(f, result);
        require(result.stats.initial_refinement.accepted == 1 && result.stats.initial_refinement.solves == 1 &&
            result.stats.refined_guard.complete && result.stats.refined_guard.passed &&
            result.stats.refined_guard.max_ratio == std::optional<double>(0), "同LU修正必须通过原逐行守卫");
        require(result.stats.primal_solves == 0 && result.stats.lu_factor_calls == 0 && session.x[0] == 3,
            "原始侧修正单独计数且不修改外部初始点");
      } else require(result.stats.initial_refinement.status == "unsupported_precision" &&
          !result.candidate_available, "无更宽精度时应明确拒绝修正");
      names.emplace_back("初始完整守卫与同因子修正");
    }
    {
      const auto f = fixture({{1}}, {1}, {2}, {false}); Session session(f, {0}); const auto result = session.run();
      check_candidate(f, result);
      require(result.stats.pivots == 0 && result.stats.lu_factor_calls == 0 && result.stats.dual_solves == 0 &&
          std::bit_cast<std::uint64_t>(result.x[0]) == std::bit_cast<std::uint64_t>(session.x[0]),
          "已满足原目标的初始基应复用正反回代结果");
      require(result.stats.initial_guard.complete && result.stats.initial_guard.passed &&
          result.stats.initial_refinement.status == "not_needed" && result.stats.initial_refinement.attempts == 0 &&
          result.stats.initial_refinement.double_digits == 0 && result.stats.initial_refinement.long_double_digits == 0 &&
          !result.stats.refined_guard.complete, "无需精修时不产生修正工作或平台探测账本");
      names.emplace_back("无需修复时复用初始因子和点");
    }
    {
      const auto f = fixture({{1}}, {-1}, {2}, {true}); Session session(f, {0}); repair::Result result;
      auto start = session.start(); std::vector<Index> invalid{-1}; start.selected_rows = invalid;
      repair::repair(f.input(), repair::Limits{}, start, session.lu, {}, result);
      require(!result.candidate_available && result.status == "invalid_input", "独立接口应拒绝错误行映射");
      names.emplace_back("坏输入与映射验证");
    }
    {
      const auto f = fixture({{1, 0}, {0, 1}, {0, 0}}, {-1, -1}, {2, 3, 0}, {true, true, false});
      Session session(f, {1, 0}); session.x[0] = 3; session.x[1] = 5;
      const auto result = session.run(); const auto& before = result.stats.initial_guard;
      require(before.complete && !before.passed && before.checked_rows == 2 && before.violations == 2 &&
          before.failure_row == std::optional<Index>(0) && before.failure_basis_position == std::optional<Index>(1) &&
          before.failure_abs_residual == std::optional<double>(1) && before.max_abs_residual == std::optional<double>(2) &&
          before.max_ratio > before.failure_ratio, "首次失败后仍须扫描全部选中行，坐标与比例不能混用");
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits)
        check_candidate(f, result);
      names.emplace_back("完整选中行扫描与首次失败坐标");
    }
    {
      const auto f = two_phase_fixture(); Session session(f, {0}); session.x[0] = 4;
      const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        check_candidate(f, result);
        require(result.stats.initial_refinement.accepted == 1 && result.stats.phase1_iterations == 1 &&
            result.stats.phase2_iterations == 1 && result.stats.pivots == 2,
            "初始修正通过后必须真正进入两阶段换基");
      } else require(result.stats.initial_refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("修正通过后进入阶段I与阶段II");
    }
    {
      const auto f = fixture({{1}, {0}}, {-1}, {2, 0}, {true, false}); ControlledSession session(f, {0});
      session.x[0] = 3; session.lu.mode = ControlledLu::Mode::half; const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        require(!result.candidate_available && result.status == "basis_residual_failure" &&
            result.stats.initial_refinement.status == "guard_rejected" && result.stats.initial_refinement.accepted == 0 &&
            result.stats.refined_guard.max_abs_residual < result.stats.initial_guard.max_abs_residual &&
            result.stats.refined_guard.complete && !result.stats.refined_guard.passed,
            "最大绝对残差下降不能替代逐行守卫");
        require(result.stats.final_phi == result.stats.initial_phi &&
            result.stats.max_selected_residual == std::optional<double>(1) && session.x[0] == 3,
            "拒绝临时点后平面摘要和外部初始点必须保持");
      } else require(result.stats.initial_refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("残差下降但原守卫仍拒绝");
    }
    {
      const auto f = fixture({{1}}, {-1}, {2}, {true}); ControlledSession session(f, {0}); session.x[0] = 3;
      session.lu.mode = ControlledLu::Mode::fail; const auto failed = session.run();
      session.lu.mode = ControlledLu::Mode::nonfinite; const auto nonfinite = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        require(!failed.candidate_available && failed.stats.initial_refinement.status == "solve_failure" &&
            failed.stats.initial_refinement.solves == 1 && !failed.stats.refined_guard.complete,
            "精修求解失败必须明确拒绝且不伪造后验守卫");
        require(!nonfinite.candidate_available && nonfinite.stats.initial_refinement.status == "correction_nonfinite" &&
            nonfinite.stats.initial_refinement.accepted == 0 && session.x[0] == 3,
            "非有限修正必须保留初始点");
      } else require(failed.stats.initial_refinement.status == "unsupported_precision" &&
          nonfinite.stats.initial_refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("精修回代失败与非有限修正");
    }
    {
      const auto f = fixture({{1e308}}, {0}, {-1e308}, {true}); Session session(f, {0}); session.x[0] = 1e308;
      const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        require(!result.candidate_available && result.stats.initial_refinement.status == "rhs_representation_failure" &&
            result.stats.initial_refinement.solves == 0 && result.stats.initial_guard.complete,
            "无法表示的修正右端必须在回代前拒绝");
      } else require(!result.candidate_available, "窄累加平台也应拒绝不可表示算术");
      names.emplace_back("修正右端表示失败");
    }
    {
      const auto f = fixture({{1}}, {-1}, {2}, {true}); ControlledSession session(f, {0});
      session.x[0] = std::numeric_limits<double>::max(); session.lu.mode = ControlledLu::Mode::overflow;
      const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        require(!result.candidate_available && result.stats.initial_refinement.status == "correction_nonfinite" &&
            result.stats.initial_refinement.solves == 1 && result.stats.initial_refinement.accepted == 0,
            "有限修正导致候选溢出也必须拒绝");
      } else require(result.stats.initial_refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("精修候选更新溢出");
    }
    {
      const auto f = fixture({{1}}, {-1}, {2}, {true}); Session session(f, {0}); session.x[0] = 3;
      repair::Result result; bool interrupted = false;
      try {
        repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
          if (result.stats.refined_guard.complete) throw std::runtime_error("精修提交前预算中断");
        }, result);
      } catch (const std::runtime_error&) { interrupted = true; }
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        require(interrupted && result.status == "budget_exception" &&
            result.stats.initial_refinement.status == "budget_exception" && result.stats.refined_guard.passed &&
            result.stats.initial_refinement.accepted == 0 && !result.candidate_available && session.x[0] == 3,
            "精修完整守卫后仍须预算通过才能提交");
        repair::Result committed; bool stopped_after_commit = false;
        try {
          repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
            if (committed.stats.initial_refinement.accepted == 1) throw std::runtime_error("精修提交后预算中断");
          }, committed);
        } catch (const std::runtime_error&) { stopped_after_commit = true; }
        require(stopped_after_commit && committed.status == "budget_exception" &&
            committed.stats.initial_refinement.status == "accepted" &&
            committed.stats.initial_refinement.accepted == 1 && !committed.candidate_available && session.x[0] == 3,
            "精修已提交的局部账本不应被后续恢复中断改写");
      } else require(result.stats.initial_refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("精修守卫通过后的预算事务");
    }
    std::cout << "{\"passed\":true,\"public_lp_runs\":0,\"cases\":" << names.size() << ",\"details\":[";
    for (std::size_t k = 0; k < names.size(); ++k) {
      if (k) std::cout << ',';
      std::cout << '"' << names[k] << '"';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
