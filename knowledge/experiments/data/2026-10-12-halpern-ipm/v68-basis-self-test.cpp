#include "v68-basis-recovery.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using solver::v68_basis::Index;
using solver::v68_basis::Input;
using solver::v68_basis::Limits;
using solver::v68_basis::Result;

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
    require(result.stats.qr_analyze_calls == 1 && result.stats.qr_factor_calls == 1 &&
        result.stats.lu_analyze_calls == 1 && result.stats.lu_factor_calls == 1 &&
        result.stats.primal_solves == 1 && result.stats.dual_solves == 1 &&
        result.stats.slack_matvecs == 1, "一次选基须仅用一个QR、一个LU及正反各一次回代");
  }
}

Fixture dependent_priority() {
  return {{0, 2, 3}, {0, 1, 2}, {1, 2, 1}, {-1, -2}, {2, 4, 3},
      {1, 1, 1}, {1, 1, 1}, {true, true, true}};
}

}  // 匿名命名空间

int main() {
  try {
    int cases = 0;
    const Limits limits;
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {2, 3, 1, 4}, {4, -2}, {5, 15},
          {0, 0}, {0, 0}, {false, false}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      close(r.x[0], 1, "非对称基的第一原变量"); close(r.x[1], 3, "非对称基的第二原变量");
      close(r.z[0], -4.4, "等式允许负乘子"); close(r.z[1], 1.6, "等式对偶乘子");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 2, 3}, {1, -1, 1, -1}, {-2, -1}, {3, 0, 4, 0},
          {1, 1, 1, 1}, {20, 1, 10, 1}, {true, true, true, true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.selected_rows == std::vector<Index>({0, 2}), "上下界按已有比值选取");
      close(r.x[0], 3, "第一上界"); close(r.x[1], 4, "第二上界");
      close(r.s[1], 3, "第一未选下界松弛"); close(r.s[3], 4, "第二未选下界松弛");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.ordered_rows == std::vector<Index>({0, 1, 2}) &&
          r.stats.qr_columns_permutation == std::vector<Index>({0, 2, 1}) &&
          r.stats.selected_rows == std::vector<Index>({0, 2}), "秩亏前列须通过完整列置换恢复原行号");
      close(r.z[1], 0, "未选行乘子应为零");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 3}, {1, 2, 0}, {1, 1, 1}, {-2, -3}, {5, 4, 4},
          {1e-300, 0, 1}, {1e300, 0, 1}, {true, false, true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.stats.ordered_rows == std::vector<Index>({1, 0, 2}) &&
          r.stats.selected_rows == std::vector<Index>({1, 0}), "等式优先且巨大比值无需直接形成溢出的double商");
      close(r.x[0], 4, "等式恢复第一变量"); close(r.x[1], 5, "界恢复第二变量");
      ++cases;
    }
    {
      const Fixture f{{0, 2, 4}, {0, 1, 0, 1}, {1, 2, 2, 4}, {-1, -2}, {3, 6},
          {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "numerical_rank_deficient" &&
          r.stats.rank == 1 && r.stats.lu_factor_calls == 0, "数值秩不足应在LU前拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 1, 1}, {0}, {1}, {1, 0}, {2, 3}, {1, 1}, {1, 1}, {true, true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "structural_rank_deficient" &&
          r.stats.qr_factor_calls == 0, "结构空列应在Eigen前拒绝");
      ++cases;
    }
    {
      const Fixture f{{0, 1}, {0}, {1}, {1}, {2}, {1}, {1}, {true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.z[0] < 0, "负乘子候选可以生成，但回代成功不等于LP最优");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.starts = {0, 3, 2};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "坏CSC必须在访问前拒绝");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.rows[1] = f.rows[0];
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "重复CSC行违反规范化输入契约");
      ++cases;
    }
    {
      auto f = dependent_priority();
      f.values[0] = std::numeric_limits<double>::infinity();
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_input", "非有限输入必须拒绝");
      ++cases;
    }
    {
      auto f = dependent_priority(); f.slack[0] = 0;
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      require(!r.candidate_available && r.status == "invalid_interior_point", "不能从非内部点形成排序比值");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      Limits small = limits; small.max_input_nonzeros = 2;
      const auto r = solver::v68_basis::recover(f.input(), small, {});
      require(!r.candidate_available && r.status == "input_limit" && r.stats.qr_factor_calls == 0,
          "输入上限须在分解前生效");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      Limits small = limits; small.max_factor_nonzeros = 1;
      const auto r = solver::v68_basis::recover(f.input(), small, {});
      require(!r.candidate_available && r.status == "qr_factor_limit" && r.stats.qr_factor_calls == 1 &&
          r.stats.lu_factor_calls == 0, "因子上限是分解后验拒绝，不冒称内部峰值上限");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      Result r;
      struct Expired {};
      int checks = 0;
      bool threw = false;
      try { solver::v68_basis::recover(f.input(), limits, [&] { if (++checks == 3) throw Expired{}; }, r); }
      catch (const Expired&) { threw = true; }
      require(threw && !r.candidate_available && r.status == "budget_exception" &&
          r.stats.budget_checks == 3 && r.stats.qr_factor_calls == 0, "预算异常须透传且保留未提交状态和账本");
      ++cases;
    }
    {
      const auto f = dependent_priority();
      Result r;
      struct Expired {};
      bool threw = false;
      try { solver::v68_basis::recover(f.input(), limits, [&] {
        if (r.stats.qr_factor_calls == 1) throw Expired{};
      }, r); }
      catch (const Expired&) { threw = true; }
      require(threw && !r.candidate_available && r.status == "budget_exception" &&
          r.stats.lu_factor_calls == 0, "QR不可内部中断，但返回后的预算检查必须阻止后续LU");
      ++cases;
    }
    {
      const Fixture f{{0}, {}, {}, {}, {2}, {1}, {1}, {true}};
      const auto r = solver::v68_basis::recover(f.input(), limits, {});
      check_equations(f, r);
      require(r.x.empty() && r.stats.qr_factor_calls == 0 && r.stats.lu_factor_calls == 0,
          "零变量模型具有不分解的明确候选语义");
      std::ostringstream serialized;
      solver::v68_basis::write_diagnostics(serialized, r);
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
