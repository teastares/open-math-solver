#include "v70-basis-recovery.hpp"
#include "v76-basis-repair.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace repair = solver::v76_basis_repair;
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

class AlternatingLu : public Lu {
 public:
  inline static bool enabled = false;
  inline static bool boundary_bias = false;
  mutable Index forward_calls = 0;
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const {
    Eigen::VectorXd result = Lu::solve(rhs);
    // 每个原始 forward 右端故意带可见误差，紧随其后的一次修正使用真实因子。
    if (boundary_bias && (rhs.array() == 0).all()) result.array() += 1e-25;
    else if (enabled && (++forward_calls & 1) != 0) result.array() += 0.25;
    return result;
  }
};

struct AlternatingScope {
  AlternatingScope() { AlternatingLu::enabled = true; }
  ~AlternatingScope() { AlternatingLu::enabled = false; }
};

struct BoundaryBiasScope {
  BoundaryBiasScope() { AlternatingLu::boundary_bias = true; }
  ~BoundaryBiasScope() { AlternatingLu::boundary_bias = false; }
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
using AlternatingSession = BasicSession<AlternatingLu>;

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

Fixture near_transition_fixture() {
  // 第一方向的严格改善小于L1累加分辨率，但它释放新的基表示供第二方向继续下降。
  return fixture({{1, 0}, {0, 1}, {1, 0}, {0, 1}}, {0, 0}, {0, 0, -1e-25, -1},
      {true, true, true, true});
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
    {
      const auto f = two_phase_fixture(); AlternatingSession session(f, {0}); AlternatingScope scope;
      const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        check_candidate(f, result);
        const auto& direction = result.stats.direction_forward; const auto& rebuild = result.stats.rebuild_forward;
        require(result.stats.pivots == 2 && result.stats.phase1_iterations == 1 && result.stats.phase2_iterations == 1 &&
            direction.systems == 2 && direction.guard_failures == 2 && direction.attempts == 2 &&
            direction.solves == 2 && direction.accepted == 2 && rebuild.systems == 2 &&
            rebuild.guard_failures == 2 && rebuild.attempts == 2 && rebuild.solves == 2 && rebuild.accepted == 2,
            "多个不同右端各最多一次修正，方向与重建分开计账");
        require(direction.refinement.attempts == 1 && direction.refinement.solves == 1 &&
            rebuild.refinement.attempts == 1 && rebuild.refinement.solves == 1 &&
            direction.before.complete && !direction.before.passed && direction.after.passed &&
            rebuild.before.complete && !rebuild.before.passed && rebuild.after.passed,
            "最后右端快照不能冒充跨右端累计量");
        require(direction.pivot_sequence == 1 && rebuild.pivot_sequence == 2 &&
            direction.entering_row == std::optional<Index>(1) && rebuild.entering_row == std::optional<Index>(1) &&
            result.stats.last_step == std::optional<double>(1), "修正方向的符号、步长和释放原行必须正确");
        require(result.stats.direction_solves == 2 && result.stats.primal_solves == 2 &&
            result.stats.lu_factor_calls == 2 && session.x[0] == 3,
            "修正回代不混入初次回代计数，也不改外部点");
        require(result.stats.last_point_error && result.stats.last_point_bound &&
            *result.stats.last_point_error <= *result.stats.last_point_bound,
            "已有点预测一致性计算应记录对应误差与界");
      } else require(result.stats.direction_forward.refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("多右端方向与空对偶重建点统一修正");
    }
    {
      const auto f = two_phase_fixture(); ControlledSession session(f, {0}); session.lu.mode = ControlledLu::Mode::half;
      const auto result = session.run();
      if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
        const auto& forward = result.stats.direction_forward;
        require(!result.candidate_available && result.status == "direction_residual_failure" &&
            forward.systems == 1 && forward.guard_failures == 1 && forward.attempts == 1 &&
            forward.solves == 1 && forward.accepted == 0 && forward.refinement.status == "guard_rejected" &&
            forward.after.max_abs_residual < forward.before.max_abs_residual && result.stats.pivots == 0,
            "一次方向修正后即使残差下降仍必须通过同一个守卫，不得继续修正");
      } else require(result.stats.direction_forward.refinement.status == "unsupported_precision", "无宽精度明确退出");
      names.emplace_back("方向修正一次上限与原守卫拒绝");
    }
    {
      for (bool after_commit : {false, true}) {
        const auto f = two_phase_fixture(); AlternatingSession session(f, {0}); AlternatingScope scope;
        repair::Result result; bool interrupted = false;
        try {
          repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
            const auto& forward = result.stats.direction_forward;
            if (after_commit ? forward.refinement.accepted == 1 :
                forward.after.complete && forward.refinement.accepted == 0)
              throw std::runtime_error("方向修正预算中断");
          }, result);
        } catch (const std::runtime_error&) { interrupted = true; }
        if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
          const auto& forward = result.stats.direction_forward;
          require(interrupted && result.status == "budget_exception" && result.stats.pivots == 0 &&
              forward.after.complete && forward.after.passed && forward.accepted == (after_commit ? 1 : 0) &&
              forward.refinement.status == (after_commit ? "accepted" : "budget_exception") &&
              !result.candidate_available && session.x[0] == 3,
              "方向局部修正提交前后都不能污染原始点或伪造换基");
        }
      }
      names.emplace_back("方向局部提交前后预算事务");
    }
    {
      for (bool after_commit : {false, true}) {
        const auto f = two_phase_fixture(); AlternatingSession session(f, {0}); AlternatingScope scope;
        repair::Result result; bool interrupted = false;
        try {
          repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
            const auto& forward = result.stats.rebuild_forward;
            if (after_commit ? forward.refinement.accepted == 1 :
                forward.after.complete && forward.refinement.accepted == 0)
              throw std::runtime_error("重建点修正预算中断");
          }, result);
        } catch (const std::runtime_error&) { interrupted = true; }
        if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
          const auto& forward = result.stats.rebuild_forward;
          require(interrupted && result.status == "budget_exception" && result.stats.pivots == 0 &&
              forward.pivot_sequence == 1 && forward.after.passed && forward.accepted == (after_commit ? 1 : 0) &&
              forward.refinement.status == (after_commit ? "accepted" : "budget_exception") &&
              result.stats.final_phi == result.stats.initial_phi && session.x[0] == 3 && !result.candidate_available,
              "空对偶临时点的局部修正成功不代表换基提交，平面摘要须保持旧状态");
        }
      }
      names.emplace_back("重建点局部提交与换基事务分离");
    }
    {
      const auto f = near_transition_fixture(); Session session(f, {0, 1});
      repair::Limits one_pivot; one_pivot.max_pivots = 1;
      const auto first = session.run(one_pivot);
      require(!first.candidate_available && first.status == "pivot_limit" && first.stats.pivots == 1 &&
          first.stats.phase1_transition.status == "near_accepted" &&
          first.stats.phase1_transition.near_accepted == 1 && first.stats.phase1_transition.strict_accepted == 0 &&
          first.stats.phase1_transition.delta_phi == std::optional<double>(0) &&
          first.stats.phase1_transition.checked_rows == 4 && first.stats.phase1_transition.blocking_near &&
          first.stats.phase1_transition.released_near && first.stats.phase1_transition.all_rows_near,
          "正但极小的步长可改变基而不伪造可分辨L1改善");
      require(first.stats.history.initialized && first.stats.history.states == 2 && first.stats.history.swaps == 1 &&
          first.stats.history.candidate_checks == 1, "近退化接纳与实际换基历史必须同时提交");
      const auto result = session.run(); check_candidate(f, result);
      require(result.stats.pivots == 2 && result.stats.phase1_transition.near_accepted == 1 &&
          result.stats.phase1_transition.strict_accepted == 1 && result.stats.history.states == 3 &&
          result.stats.history.swaps == 2 && result.stats.history.candidate_checks == 2,
          "近退化基转换之后应继续获得严格下降和完整候选");
      close(result.x[0], -1e-25, "第一近界变量"); close(result.x[1], -1, "随后严格改善的变量");
      names.emplace_back("真实近退化接纳后继续严格下降");
    }
    {
      using namespace repair::implementation;
      State old, next; old.slack = {0, 0, 0, -1}; old.roundoff.assign(4, 1e-14L); old.phi = 1;
      next.slack = old.slack; next.roundoff = old.roundoff; next.phi = 1;
      repair::Result ledger; repair::Limits limits; Work work(limits, {}, ledger);
      const auto reject = [&](const char* expected) {
        bool rejected = false;
        try { phase1_transition(old, next, 0, 1, 1e-25L, work, ledger.stats.phase1_transition); }
        catch (const Stop& stop) { rejected = std::string(stop.status) == "objective_not_improved"; }
        require(rejected && ledger.stats.phase1_transition.status == expected &&
            ledger.stats.phase1_transition.checked_rows == 4 && ledger.stats.phase1_transition.near_accepted == 0,
            "近退化条件失败须完整记账并拒绝");
      };
      next.slack[2] = 3e-14L; reject("row_change_not_near");
      next.slack = old.slack; old.slack[0] = 3e-14L; next.slack = old.slack; reject("blocking_not_near");
      old.slack[0] = 0; next.slack = old.slack;
      bool rejected = false;
      try { phase1_transition(old, next, 0, 1, 3e-14L, work, ledger.stats.phase1_transition); }
      catch (const Stop& stop) { rejected = std::string(stop.status) == "objective_not_improved"; }
      require(rejected && ledger.stats.phase1_transition.status == "released_not_near" &&
          ledger.stats.phase1_transition.checked_rows == 4, "释放步长条件不能由其他近界条件替代");
      names.emplace_back("近退化三项必要条件逐项拒绝");
    }
    {
      using namespace repair::implementation;
      repair::Result ledger; repair::Limits limits; Work work(limits, {}, ledger);
      State initial, candidate; initial.selected = {0, 1}; candidate.selected = {2, 0};
      positions(initial, 3, work); positions(candidate, 3, work);
      BasisHistory history; history.initialize(initial, limits, work, ledger.stats.history);
      const auto hash = history.candidate_hash(1, 2);
      history.hashes[0] = hash;  // 人为制造碰撞，只检验hash不是相等证明。
      require(!history.contains(candidate, hash, work, ledger.stats.history) &&
          ledger.stats.history.hash_matches == 1 && ledger.stats.history.exact_checks == 1 &&
          ledger.stats.history.repeats == 0, "不同成员集合不能因hash碰撞被判为重复");
      history.commit(1, 2, hash, ledger.stats.history);
      candidate.selected = {0, 2}; positions(candidate, 3, work);
      require(history.contains(candidate, hash, work, ledger.stats.history) &&
          ledger.stats.history.repeated_state == std::optional<Index>(1) &&
          ledger.stats.history.replay_swaps == 1 && history.initial_rows.size() == 2 &&
          history.swaps.size() == 1 && history.hashes.size() == 2,
          "从初始集合重放swap应确认同一集合，与行位置排列无关");
      names.emplace_back("碰撞预筛与精确集合重放判重");
    }
    {
      const auto f = fixture({{1}, {1}, {0}}, {0}, {0, 0, 1}, {true, true, false});
      AlternatingSession session(f, {0}); session.x[0] = 1e-25; BoundaryBiasScope scope;
      const auto result = session.run();
      require(!result.candidate_available && result.status == "basis_repeated" && result.stats.pivots == 1 &&
          result.stats.phase1_transition.trials == 2 && result.stats.phase1_transition.near_accepted == 1 &&
          result.stats.phase1_transition.status == "basis_repeated" && result.stats.history.states == 2 &&
          result.stats.history.swaps == 1 && result.stats.history.candidate_checks == 2 &&
          result.stats.history.repeats == 1 && result.stats.history.repeated_state == std::optional<Index>(0),
          "守卫允许的微小回代误差不能导致同一基反复提交，也不发布不可行证明");
      names.emplace_back("真实近退化循环在重复基提交前停止");
    }
    {
      for (bool after_commit : {false, true}) {
        const auto f = near_transition_fixture(); Session session(f, {0, 1}); repair::Result result;
        bool interrupted = false;
        try {
          repair::repair(f.input(), repair::Limits{}, session.start(), session.lu, [&] {
            const auto& transition = result.stats.phase1_transition;
            if (after_commit ? transition.near_accepted == 1 : transition.status == "near_candidate")
              throw std::runtime_error("近退化基提交预算中断");
          }, result);
        } catch (const std::runtime_error&) { interrupted = true; }
        require(interrupted && result.status == "budget_exception" && !result.candidate_available &&
            result.stats.pivots == (after_commit ? 1 : 0) && result.stats.history.swaps == result.stats.pivots &&
            result.stats.history.states == result.stats.pivots + 1 &&
            result.stats.phase1_transition.near_accepted == result.stats.pivots &&
            session.x[0] == 0 && session.x[1] == 0,
            "历史与基状态必须同事务提交，外部起点不受中断影响");
      }
      names.emplace_back("近退化与历史提交前后预算事务");
    }
    {
      using namespace repair::implementation;
      repair::Result ledger; repair::Limits limits;
      Work work(limits, [&] {
        if (ledger.stats.history.exact_checks == 1 && ledger.stats.history.membership_comparisons > 0)
          throw std::runtime_error("精确历史成员比较预算中断");
      }, ledger);
      State initial; initial.selected = {0}; positions(initial, 600, work);
      BasisHistory history; history.initialize(initial, limits, work, ledger.stats.history);
      bool interrupted = false;
      try { history.contains(initial, history.current_hash, work, ledger.stats.history); }
      catch (const std::runtime_error&) { interrupted = true; }
      require(interrupted && ledger.status == "budget_exception" && ledger.stats.history.status == "checking" &&
          ledger.stats.history.exact_checks == 1 && ledger.stats.history.membership_comparisons > 0 &&
          ledger.stats.history.membership_comparisons < 600 && ledger.stats.history.repeats == 0 &&
          ledger.stats.history.states == 1 && ledger.stats.history.swaps == 0,
          "精确比较未完成不能伪造重复结论或历史提交");
      names.emplace_back("精确历史回放中的部分预算账本");
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
