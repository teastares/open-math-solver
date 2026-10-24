#include "v69-basis-recovery.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using solver::v69_basis::Index;
using solver::v69_basis::Input;
using solver::v69_basis::Limits;
using solver::v69_basis::Result;

struct Fixture {
  std::vector<Index> starts, rows;
  std::vector<double> values, costs, rhs, slack, dual;
  std::vector<bool> inequality;
  Input input() const { return {starts, rows, values, costs, rhs, slack, dual, inequality}; }
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const std::string& message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= 1e-12 * std::max(1.0, std::abs(expected)), message);
}

void check_equations(const Fixture& fixture, const Result& result) {
  require(result.candidate_available && result.status == "candidate", "应生成候选，仍需原模型验收");
  std::vector<long double> activity(fixture.rhs.size(), 0);
  for (std::size_t col = 0; col < fixture.costs.size(); ++col) {
    long double transpose = 0;
    for (Index k = fixture.starts[col]; k < fixture.starts[col + 1]; ++k) {
      activity[fixture.rows[k]] += static_cast<long double>(fixture.values[k]) * result.x[col];
      transpose += static_cast<long double>(fixture.values[k]) * result.z[fixture.rows[k]];
    }
    close(static_cast<double>(transpose), -fixture.costs[col], "完整原矩阵对偶方程应成立");
  }
  for (std::size_t row = 0; row < fixture.rhs.size(); ++row)
    close(static_cast<double>(activity[row] + result.s[row]), fixture.rhs[row], "完整原矩阵原始方程应成立");
  if (!fixture.costs.empty()) {
    require(result.stats.qr_analyze_calls == 0 && result.stats.qr_factor_calls == 0 &&
        result.stats.lu_analyze_calls == 1 && result.stats.lu_factor_calls == 1 &&
        result.stats.primal_solves == 1 && result.stats.dual_solves == 1 &&
        result.stats.slack_matvecs == 1, "一次选基仅用一个LU及正反各一次回代，不执行QR");
    require(result.stats.selection_pivots == static_cast<Index>(fixture.costs.size()) &&
        result.stats.rank == result.stats.selection_pivots && result.stats.qr_columns_permutation.empty() &&
        result.stats.pivot_columns.size() == fixture.costs.size(), "选择计数及主元映射完整，rank不冒称QR秩");
  }
}

Fixture dependent_priority() {
  return {{0, 2, 5}, {0, 1, 0, 1, 2}, {1, 2, 1, 2, 1}, {-1, -2}, {5, 10, 3},
      {1, 1, 1}, {1, 1, 1}, {true, true, true}};
}

Fixture identity_fixture(Index dimension) {
  Fixture f;
  f.starts.push_back(0);
  for (Index i = 0; i < dimension; ++i) {
    f.starts.push_back(i + 1); f.rows.push_back(i); f.values.push_back(1);
    f.costs.push_back(-1); f.rhs.push_back(static_cast<double>(i + 1));
    f.slack.push_back(1); f.dual.push_back(1); f.inequality.push_back(true);
  }
  return f;
}

}  // 匿名命名空间

int main() {
  try {
    int cases = 0;
    const Limits limits;
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {2, 3, 1, 4}, {4, -2}, {5, 15},
          {0, 0}, {0, 0}, {false, false}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      close(r.x[0], 1, "非对称基的第一原变量"); close(r.x[1], 3, "非对称基的第二原变量");
      close(r.z[0], -4.4, "等式允许负乘子"); close(r.z[1], 1.6, "等式对偶乘子");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 2, 3}, {1, -1, 1, -1}, {-2, -1}, {3, 0, 4, 0},
          {1, 1, 1, 1}, {20, 1, 10, 1}, {true, true, true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.selected_rows == std::vector<Index>({0, 2}), "上下界按已有比值选取");
      close(r.x[0], 3, "第一上界"); close(r.x[1], 4, "第二上界");
      close(r.s[1], 3, "第一未选下界松弛"); close(r.s[3], 4, "第二未选下界松弛");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.ordered_rows == std::vector<Index>({0, 1, 2}) &&
          r.stats.pivot_columns == std::vector<Index>({0, 1}) &&
          r.stats.selected_rows == std::vector<Index>({0, 2}) && r.stats.selection_rejected == 1 &&
          r.stats.selection_candidates == 3, "结构可匹配的比例依赖行须跳过，随后独立原行才能入基");
      close(r.z[1], 0, "未选行乘子应为零");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 3}, {1, 2, 0}, {1, 1, 1}, {-2, -3}, {5, 4, 4},
          {1e-300, 0, 1}, {1e300, 0, 1}, {true, false, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.ordered_rows == std::vector<Index>({1, 0, 2}) &&
          r.stats.selected_rows == std::vector<Index>({1, 0}), "等式优先且巨大比值无需直接形成溢出的double商");
      close(r.x[0], 4, "等式恢复第一变量"); close(r.x[1], 5, "界恢复第二变量");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {1, 2, 2, 4}, {-1, -2}, {3, 6},
          {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "selection_rank_deficient" &&
          r.stats.rank == 1 && r.stats.lu_factor_calls == 0, "数值秩不足应在LU前拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 1, 1}, {0}, {1}, {1, 0}, {2, 3}, {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "structural_rank_deficient" &&
          r.stats.qr_factor_calls == 0, "结构空列应在Eigen前拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 1}, {0}, {1}, {1}, {2}, {1}, {1}, {true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.z[0] < 0, "负乘子候选可以生成，但回代成功不等于LP最优");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.starts = {0, 3, 2};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "坏CSC必须在访问前拒绝");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.rows[1] = f.rows[0];
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "重复CSC行违反规范化输入契约");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.values[0] = std::numeric_limits<double>::infinity();
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "非有限输入必须拒绝");
      ++cases;
    }
    {
      auto f = dependent_priority(); f.slack[0] = 0;
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_interior_point", "不能从非内部点形成排序比值");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      Limits small = limits; small.max_input_nonzeros = 2;
      const auto r = solver::v69_basis::recover(f.input(), small, {});
      require(!r.candidate_available && r.status == "input_limit" && r.stats.qr_factor_calls == 0,
          "输入上限须在分解前生效");
      ++cases;
    }
    {
      const Fixture f{{0, 1, 2, 3}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {3, 0, 0},
          {1, 1, 1}, {1, 1, 1}, {true, true, true}};
      Limits small = limits; small.max_factor_nonzeros = 1;
      const auto r = solver::v69_basis::recover(f.input(), small, {});
      require(!r.candidate_available && r.status == "selection_factor_limit" && r.stats.l_nonzeros == 0 &&
          r.stats.selection_pivots == 0 && r.stats.lu_factor_calls == 0,
          "新增L列超过填充上限时必须在写任何该列填充前拒绝");
      ++cases;
    }
    {
      const auto f = identity_fixture(32);
      Result r;
      struct Expired {};
      int checks = 0;
      bool threw = false;
      try { solver::v69_basis::recover(f.input(), limits, [&] { if (++checks == 3) throw Expired{}; }, r); }
      catch (const Expired&) { threw = true; }
      require(threw && !r.candidate_available && r.status == "budget_exception" &&
          r.stats.budget_checks == 3 && r.stats.qr_factor_calls == 0, "预算异常须透传且保留未提交状态和账本");
      ++cases;
    }
    {
      const auto f = identity_fixture(32);
      Result r;
      struct Expired {};
      bool threw = false;
      try { solver::v69_basis::recover(f.input(), limits, [&] {
        if (r.stats.selection_pivots >= 1) throw Expired{};
      }, r); }
      catch (const Expired&) { threw = true; }
      require(threw && !r.candidate_available && r.status == "budget_exception" &&
          r.stats.lu_factor_calls == 0 && r.stats.selection_pivots > 0 &&
          r.stats.selection_pivots < static_cast<Index>(f.costs.size()) &&
          static_cast<Index>(r.stats.selected_rows.size()) == r.stats.selection_pivots &&
          static_cast<Index>(r.stats.pivot_columns.size()) == r.stats.selection_pivots,
          "选基预算中断必须阻止LU且保留已提交映射");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      const auto full = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, full);
      Limits boundary = limits; boundary.max_selection_work = full.stats.work_units;
      const auto exact = solver::v69_basis::recover(f.input(), boundary, {});
      check_equations(f, exact);
      --boundary.max_selection_work;
      const auto short_result = solver::v69_basis::recover(f.input(), boundary, {});
      require(!short_result.candidate_available && short_result.status == "selection_work_limit" &&
          short_result.stats.work_units == boundary.max_selection_work && short_result.stats.lu_factor_calls == 0,
          "工作额度恰好够时可完成，少一个单位必须在操作前拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 1, 2, 3}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {3, 0, 0},
          {1, 1, 1}, {1, 1, 1}, {true, true, true}};
      Limits boundary = limits; boundary.max_factor_nonzeros = 2;
      const auto r = solver::v69_basis::recover(f.input(), boundary, {});
      require(!r.candidate_available && r.status == "selection_rank_deficient" &&
          r.stats.l_nonzeros == 2 && r.stats.selection_pivots == 1,
          "填充额度恰好等于一列所需非零数时不得提前拒绝或丢弃非零");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {1e308, -1e308, 1e308, 1e308}, {0, 0}, {0, 0},
          {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "selection_update_nonfinite" &&
          r.stats.selection_pivots == 1 && r.stats.lu_factor_calls == 0 &&
          std::isfinite(r.stats.selection_threshold), "原数据和范数有限，但消元差溢出必须拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {1, 1e-200, 1e-200, 1}, {0, 0}, {0, 0},
          {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "selection_product_underflow" &&
          r.stats.selection_pivots == 1 && r.stats.lu_factor_calls == 0,
          "非零消元乘积归零不得静默视为零填充");
      ++cases;
    }
    {
      const double tiny = std::numeric_limits<double>::denorm_min();
      const Fixture f{{0, 1, 3}, {0, 0, 1}, {1, tiny, 1}, {-1, -1}, {1, 1},
          {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.l_nonzeros == 1, "可表示的非零subnormal填充应保留");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4, 6}, {0, 1, 0, 2, 1, 2}, {1, 1, 1, 1, 1, 1}, {-2, -2, -2},
          {3, 4, 5}, {1, 1, 1}, {1, 1, 1}, {true, true, true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.pivot_columns == std::vector<Index>({0, 1, 2}) &&
          r.stats.selected_rows == std::vector<Index>({0, 1, 2}) && r.stats.l_nonzeros == 2,
          "新填充必须参与主元选择，同幅值按原变量号选择而非touch先后");
      close(r.x[0], 1, "完整填充恢复第一变量"); close(r.x[1], 2, "完整填充恢复第二变量");
      close(r.x[2], 3, "完整填充恢复第三变量");
      ++cases;
    }
    {
      const Fixture f{{0}, {}, {}, {}, {2}, {1}, {1}, {true}};
      const auto r = solver::v69_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.x.empty() && r.stats.qr_factor_calls == 0 && r.stats.lu_factor_calls == 0,
          "零变量模型具有不分解的明确候选语义");
      std::ostringstream serialized;
      solver::v69_basis::write_diagnostics(serialized, r);
      require(serialized.str().find("\"candidate_available\":true") != std::string::npos &&
          serialized.str().find("\"selected_rows\":[]") != std::string::npos, "诊断必须记录候选与行映射");
      ++cases;
    }
    std::cout << "{\"passed\":true,\"cases\":" << cases << ",\"public_lp_runs\":0}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "基恢复隔离自测失败：" << error.what() << '\n';
    return 1;
  }
}
