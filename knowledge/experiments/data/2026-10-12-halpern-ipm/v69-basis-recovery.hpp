#pragma once

#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace solver::v69_basis {

using Index = std::int64_t;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;

struct Input {
  std::span<const Index> starts, rows;
  std::span<const double> values, costs, rhs, slack, dual;
  const std::vector<bool>& inequality;
};

struct Limits {
  Index max_input_nonzeros = 8000000;
  Index max_factor_nonzeros = 32000000;
  Index max_dimension = 1000000;
  Index max_selection_work = 100000000;
};

struct Stats {
  Index n = 0, m = 0, input_nonzeros = 0, rank = 0;
  Index qr_q_nonzeros = 0, qr_r_nonzeros = 0, lu_l_nonzeros = 0, lu_u_nonzeros = 0;
  Index budget_checks = 0, qr_analyze_calls = 0, qr_factor_calls = 0;
  Index lu_analyze_calls = 0, lu_factor_calls = 0;
  Index primal_solves = 0, dual_solves = 0, slack_matvecs = 0;
  double qr_default_threshold = 0;
  std::vector<Index> ordered_rows, qr_columns_permutation, selected_rows;
  Index selection_candidates = 0, selection_pivots = 0, selection_rejected = 0;
  Index work_units = 0, l_nonzeros = 0;
  double min_abs_pivot = 0, selection_threshold = 0;
  std::vector<Index> pivot_columns;
};

struct Result {
  std::string status = "not_started";
  bool candidate_available = false;
  std::vector<double> x, s, z;
  Stats stats;
};

namespace implementation {

struct SelectionWorkExceeded {};

struct LowerEntry {
  Index coordinate;
  double value;
};

struct LowerColumn {
  Index pivot;
  std::vector<LowerEntry> entries;
};

class CheckedLu : public Eigen::SparseLU<Matrix, Eigen::COLAMDOrdering<Index>> {
 public:
  // Eigen 部分内存失败分支不更新 info，另核对内部的因子有效标志。
  bool factorization_ok() const { return m_factorizationIsOk; }
  bool finite_factors(const std::function<void()>& check) const {
    Index visited = 0;
    for (Eigen::Index col = 0; col < cols(); ++col) {
      check();
      for (SCMatrix::InnerIterator entry(m_Lstore, col); entry; ++entry) {
        if ((visited++ & 255) == 0) check();
        if (!std::isfinite(entry.value())) return false;
      }
      for (decltype(m_Ustore)::InnerIterator entry(m_Ustore, col); entry; ++entry) {
        if ((visited++ & 255) == 0) check();
        if (!std::isfinite(entry.value())) return false;
      }
    }
    return true;
  }
};

struct Score {
  int exponent;
  long double mantissa;
};

inline Score ratio_score(double numerator, double denominator) {
  int numerator_exponent = 0, denominator_exponent = 0;
  const long double numerator_mantissa = std::frexp(numerator, &numerator_exponent);
  const long double denominator_mantissa = std::frexp(denominator, &denominator_exponent);
  long double mantissa = numerator_mantissa / denominator_mantissa;
  int exponent = numerator_exponent - denominator_exponent;
  // 分离指数，避免 long double 等同 double 的平台也出现中间商溢出。
  if (mantissa >= 1) { mantissa *= 0.5L; ++exponent; }
  return {exponent, mantissa};
}

inline bool exceeds_sum(Index first, Index second, Index limit) {
  return first < 0 || second < 0 || first > limit || second > limit - first;
}

}  // 实现细节命名空间

// 仅生成候选，不以基方程回代成功替代原模型的可行性、符号及对偶间隙验收。
inline void recover(const Input& input, const Limits& limits,
                    const std::function<void()>& budget, Result& output) {
  output = Result{};
  auto& stats = output.stats;
  const auto check = [&] {
    ++stats.budget_checks;
    if (budget) {
      try { budget(); }
      catch (...) { output.status = "budget_exception"; throw; }
    }
  };
  check();
  output.status = "invalid_input";
  const auto index_limit = static_cast<std::size_t>(std::min<Index>(
      std::numeric_limits<Index>::max(), std::numeric_limits<Eigen::Index>::max()));
  if (input.costs.size() >= index_limit || input.rhs.size() >= index_limit ||
      input.values.size() >= index_limit) return;
  stats.n = static_cast<Index>(input.costs.size());
  stats.m = static_cast<Index>(input.rhs.size());
  stats.input_nonzeros = static_cast<Index>(input.values.size());
  const Index n = stats.n, m = stats.m;
  if (limits.max_dimension < 0 || limits.max_input_nonzeros < 0 ||
      limits.max_factor_nonzeros < 0 || limits.max_selection_work < 0 ||
      input.starts.size() != input.costs.size() + 1 ||
      input.starts.empty() || input.starts.front() != 0 ||
      input.starts.back() != stats.input_nonzeros || input.rows.size() != input.values.size() ||
      input.inequality.size() != input.rhs.size() || input.slack.size() != input.rhs.size() ||
      input.dual.size() != input.rhs.size()) return;
  if (n > limits.max_dimension || m > limits.max_dimension ||
      stats.input_nonzeros > limits.max_input_nonzeros) {
    output.status = "input_limit";
    return;
  }
  std::vector<Index> row_counts(static_cast<std::size_t>(m), 0);
  bool empty_column = false;
  for (Index col = 0; col < n; ++col) {
    check();
    if (!std::isfinite(input.costs[col]) || input.starts[col] < 0 ||
        input.starts[col] > input.starts[col + 1] || input.starts[col + 1] > stats.input_nonzeros) return;
    Index previous = -1;
    bool nonzero = false;
    for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
      if ((k & 255) == 0) check();
      const Index row = input.rows[k];
      // ConeModel 已合并重复项并按行升序；独立接口拒绝违反该契约的数据。
      if (row <= previous || row >= m || !std::isfinite(input.values[k])) return;
      previous = row;
      nonzero = nonzero || input.values[k] != 0;
      ++row_counts[row];
    }
    empty_column = empty_column || !nonzero;
  }
  std::vector<implementation::Score> scores(static_cast<std::size_t>(m));
  for (Index row = 0; row < m; ++row) {
    check();
    if (!std::isfinite(input.rhs[row]) || !std::isfinite(input.slack[row]) ||
        !std::isfinite(input.dual[row])) return;
    if (input.inequality[row]) {
      if (!(input.slack[row] > 0) || !(input.dual[row] > 0)) {
        output.status = "invalid_interior_point";
        return;
      }
      scores[row] = implementation::ratio_score(input.dual[row], input.slack[row]);
    }
  }
  if (n == 0) {
    output.s.resize(static_cast<std::size_t>(m));
    output.z.assign(static_cast<std::size_t>(m), 0);
    for (Index row = 0; row < m; ++row) {
      check();
      output.s[row] = input.inequality[row] ? input.rhs[row] : 0;
    }
    output.status = "candidate";
    output.candidate_available = true;
    return;
  }
  if (m < n || empty_column) { output.status = "structural_rank_deficient"; return; }
  // 一个工作单位是一项自有元素循环访问或固定规模的比较/提交操作，
  // 包括行构造、范数、排序、初始化、装载、主元访问、消元、扫描及存储。
  // 输入合法性检查和后续 LU 不计入此账本，仍受同一总时间预算约束。
  const auto consume = [&] {
    if (stats.work_units >= limits.max_selection_work) {
      output.status = "selection_work_limit";
      throw implementation::SelectionWorkExceeded{};
    }
    if ((stats.work_units & 255) == 0) check();
    ++stats.work_units;
  };
  try {
    // 选择工作区在此作用域内释放，之后才分配最终原基及其 LU 因子。
    std::unique_ptr<Index[]> row_starts(new Index[static_cast<std::size_t>(m) + 1]);
    std::unique_ptr<Index[]> next(new Index[static_cast<std::size_t>(m)]);
    std::unique_ptr<Index[]> coordinates(new Index[static_cast<std::size_t>(stats.input_nonzeros)]);
    std::unique_ptr<double[]> row_values(new double[static_cast<std::size_t>(stats.input_nonzeros)]);
    std::unique_ptr<double[]> row_norms(new double[static_cast<std::size_t>(m)]);
    consume(); row_starts[0] = 0;
    for (Index row = 0; row < m; ++row) {
      consume();
      row_starts[row + 1] = row_starts[row] + row_counts[row];
      next[row] = row_starts[row];
      row_norms[row] = 0;
    }
    for (Index col = 0; col < n; ++col) {
      consume();
      for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
        consume();
        const Index row = input.rows[k], target = next[row]++;
        coordinates[target] = col;
        row_values[target] = input.values[k];
        // 逐原行用 hypot 累积二范数，避免先平方造成中间上溢/下溢。
        row_norms[row] = std::hypot(row_norms[row], input.values[k]);
        if (!std::isfinite(row_norms[row])) { output.status = "selection_norm_nonfinite"; return; }
      }
    }
    double max_norm = 0;
    for (Index row = 0; row < m; ++row) {
      consume();
      max_norm = std::max(max_norm, row_norms[row]);
    }
    if (max_norm == 0) max_norm = 1;
    consume();
    // 数学尺度沿用 20*(n+m)*max_row_2norm*eps；先乘 eps 再乘范数，
    // 防止公式中间溢出。此阈值作用于消元残量，不是 QR 的秩判定。
    const double scale = 20.0 * static_cast<double>(n + m) * std::numeric_limits<double>::epsilon();
    const double threshold = scale * max_norm;
    if (!std::isfinite(threshold)) { output.status = "selection_threshold_nonfinite"; return; }
    if (threshold == 0) { output.status = "selection_threshold_underflow"; return; }
    stats.selection_threshold = threshold;
    stats.ordered_rows.resize(static_cast<std::size_t>(m));
    for (Index row = 0; row < m; ++row) { consume(); stats.ordered_rows[row] = row; }
    std::sort(stats.ordered_rows.begin(), stats.ordered_rows.end(), [&](Index a, Index b) {
      consume();
      if (input.inequality[a] != input.inequality[b]) return !input.inequality[a];
      if (input.inequality[a]) {
        if (scores[a].exponent != scores[b].exponent) return scores[a].exponent > scores[b].exponent;
        if (scores[a].mantissa != scores[b].mantissa) return scores[a].mantissa > scores[b].mantissa;
      }
      return a < b;
    });
    std::unique_ptr<double[]> work(new double[static_cast<std::size_t>(n)]);
    std::unique_ptr<Index[]> generation(new Index[static_cast<std::size_t>(n)]);
    std::unique_ptr<bool[]> used(new bool[static_cast<std::size_t>(n)]);
    for (Index col = 0; col < n; ++col) {
      consume(); generation[col] = 0; used[col] = false;
    }
    std::vector<Index> touched;
    touched.reserve(static_cast<std::size_t>(n));
    std::vector<implementation::LowerColumn> lower;
    lower.reserve(static_cast<std::size_t>(n));
    stats.selected_rows.reserve(static_cast<std::size_t>(n));
    stats.pivot_columns.reserve(static_cast<std::size_t>(n));
    for (const Index row : stats.ordered_rows) {
      consume();
      ++stats.selection_candidates;
      touched.clear();
      const Index current_generation = stats.selection_candidates;
      const auto touch = [&](Index coordinate) {
        if (generation[coordinate] != current_generation) {
          consume();
          generation[coordinate] = current_generation;
          work[coordinate] = 0;
          touched.push_back(coordinate);
        }
      };
      for (Index k = row_starts[row]; k < row_starts[row + 1]; ++k) {
        consume();
        const Index coordinate = coordinates[k];
        touch(coordinate);
        work[coordinate] = row_values[k];
      }
      // 每个既有主元按接纳顺序访问；不构造整矩阵 QR，也不丢弃非零填充。
      for (const auto& column : lower) {
        consume();
        if (generation[column.pivot] != current_generation) continue;
        const double multiplier = work[column.pivot];
        if (multiplier == 0) continue;
        for (const auto& entry : column.entries) {
          consume();
          const double product = multiplier * entry.value;
          if (!std::isfinite(product)) { output.status = "selection_product_nonfinite"; return; }
          if (product == 0 && entry.value != 0) { output.status = "selection_product_underflow"; return; }
          touch(entry.coordinate);
          const double previous = work[entry.coordinate];
          const double updated = previous - product;
          if (!std::isfinite(updated)) { output.status = "selection_update_nonfinite"; return; }
          // 相等 double 的精确抵消允许为零；非零 subnormal 仍完整保存。
          if (updated == 0 && previous != product) { output.status = "selection_update_underflow"; return; }
          work[entry.coordinate] = updated;
        }
      }
      Index pivot = -1, nonzeros = 0;
      double maximum = 0;
      for (const Index coordinate : touched) {
        consume();
        if (used[coordinate]) continue;
        const double magnitude = std::abs(work[coordinate]);
        if (magnitude != 0) ++nonzeros;
        if (magnitude > maximum || (magnitude == maximum && magnitude > 0 && coordinate < pivot)) {
          maximum = magnitude; pivot = coordinate;
        }
      }
      if (maximum <= stats.selection_threshold) {
        consume(); ++stats.selection_rejected;
        continue;
      }
      consume();
      const Index new_nonzeros = nonzeros - 1;
      if (implementation::exceeds_sum(stats.l_nonzeros, new_nonzeros, limits.max_factor_nonzeros)) {
        output.status = "selection_factor_limit";
        return;
      }
      lower.push_back({pivot, {}});
      auto& entries = lower.back().entries;
      entries.reserve(static_cast<std::size_t>(new_nonzeros));
      for (const Index coordinate : touched) {
        consume();
        if (used[coordinate] || coordinate == pivot || work[coordinate] == 0) continue;
        const double coefficient = work[coordinate] / work[pivot];
        if (!std::isfinite(coefficient)) { output.status = "selection_division_nonfinite"; return; }
        if (coefficient == 0) { output.status = "selection_division_underflow"; return; }
        consume();
        entries.push_back({coordinate, coefficient});
        ++stats.l_nonzeros;
      }
      consume();
      // 一次固定规模提交不中插预算检查，使异常账本始终保持映射与计数一致。
      used[pivot] = true;
      stats.selected_rows.push_back(row);
      stats.pivot_columns.push_back(pivot);
      ++stats.selection_pivots;
      stats.rank = stats.selection_pivots;
      stats.min_abs_pivot = stats.selection_pivots == 1 ? maximum : std::min(stats.min_abs_pivot, maximum);
      if (stats.selection_pivots == n) break;
    }
    check();
    if (stats.selection_pivots != n) { output.status = "selection_rank_deficient"; return; }
  } catch (const implementation::SelectionWorkExceeded&) {
    return;
  }
  std::vector<Index> basis_position(static_cast<std::size_t>(m), -1);
  for (Index position = 0; position < n; ++position) {
    check(); basis_position[stats.selected_rows[position]] = position;
  }
  std::vector<Eigen::Triplet<double, Index>> entries;
  entries.reserve(static_cast<std::size_t>(stats.input_nonzeros));
  for (Index col = 0; col < n; ++col) {
    check();
    for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
      if ((k & 255) == 0) check();
      const Index row = basis_position[input.rows[k]];
      if (row >= 0) entries.emplace_back(row, col, input.values[k]);
    }
  }
  Matrix basis(n, n);
  check();
  basis.setFromTriplets(entries.begin(), entries.end());
  check();
  basis.makeCompressed();
  check();
  implementation::CheckedLu lu;
  ++stats.lu_analyze_calls;
  lu.analyzePattern(basis);
  check();
  ++stats.lu_factor_calls;
  lu.factorize(basis);
  check();
  if (!lu.factorization_ok() || lu.info() != Eigen::Success) { output.status = "lu_failure"; return; }
  stats.lu_l_nonzeros = static_cast<Index>(lu.nnzL());
  stats.lu_u_nonzeros = static_cast<Index>(lu.nnzU());
  if (implementation::exceeds_sum(stats.lu_l_nonzeros, stats.lu_u_nonzeros, limits.max_factor_nonzeros)) {
    output.status = "lu_factor_limit";
    return;
  }
  if (!lu.finite_factors(check)) { output.status = "lu_nonfinite"; return; }
  Eigen::VectorXd primal_rhs(n), dual_rhs(n);
  for (Index i = 0; i < n; ++i) {
    check();
    primal_rhs[i] = input.rhs[stats.selected_rows[i]];
    dual_rhs[i] = -input.costs[i];
  }
  check();
  ++stats.primal_solves;
  const Eigen::VectorXd primal = lu.solve(primal_rhs);
  check();
  if (!primal.allFinite() || lu.info() != Eigen::Success) { output.status = "primal_solve_failure"; return; }
  ++stats.dual_solves;
  const Eigen::VectorXd dual = lu.transpose().solve(dual_rhs);
  check();
  if (!dual.allFinite() || lu.info() != Eigen::Success) { output.status = "dual_solve_failure"; return; }
  output.x.assign(primal.data(), primal.data() + n);
  output.s.assign(static_cast<std::size_t>(m), 0);
  output.z.assign(static_cast<std::size_t>(m), 0);
  for (Index i = 0; i < n; ++i) {
    check();
    output.z[stats.selected_rows[i]] = dual[i];
  }
  std::vector<long double> activity(static_cast<std::size_t>(m), 0);
  ++stats.slack_matvecs;
  for (Index col = 0; col < n; ++col) {
    check();
    for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
      if ((k & 255) == 0) check();
      activity[input.rows[k]] += static_cast<long double>(input.values[k]) * output.x[col];
    }
  }
  for (Index row = 0; row < m; ++row) {
    check();
    if (!std::isfinite(activity[row])) { output.status = "slack_nonfinite"; return; }
    if (input.inequality[row]) {
      const long double slack = static_cast<long double>(input.rhs[row]) - activity[row];
      output.s[row] = static_cast<double>(slack);
      if (!std::isfinite(slack) || !std::isfinite(output.s[row])) {
        output.status = "slack_nonfinite";
        return;
      }
    }
  }
  check();
  output.status = "candidate";
  output.candidate_available = true;
}

inline Result recover(const Input& input, const Limits& limits, const std::function<void()>& budget) {
  Result result;
  recover(input, limits, budget, result);
  return result;
}

inline void write_diagnostics(std::ostream& stream, const Result& result) {
  const auto& s = result.stats;
  stream << "{\"status\":\"" << result.status << "\",\"candidate_available\":"
      << (result.candidate_available ? "true" : "false");
  const auto scalar = [&](const char* name, auto value) { stream << ",\"" << name << "\":" << value; };
  scalar("n", s.n); scalar("m", s.m); scalar("input_nonzeros", s.input_nonzeros); scalar("rank", s.rank);
  scalar("qr_q_nonzeros", s.qr_q_nonzeros); scalar("qr_r_nonzeros", s.qr_r_nonzeros);
  scalar("lu_l_nonzeros", s.lu_l_nonzeros); scalar("lu_u_nonzeros", s.lu_u_nonzeros);
  scalar("budget_checks", s.budget_checks); scalar("qr_analyze_calls", s.qr_analyze_calls);
  scalar("qr_factor_calls", s.qr_factor_calls); scalar("lu_analyze_calls", s.lu_analyze_calls);
  scalar("lu_factor_calls", s.lu_factor_calls); scalar("primal_solves", s.primal_solves);
  scalar("dual_solves", s.dual_solves); scalar("slack_matvecs", s.slack_matvecs);
  scalar("selection_candidates", s.selection_candidates); scalar("selection_pivots", s.selection_pivots);
  scalar("selection_rejected", s.selection_rejected); scalar("work_units", s.work_units);
  scalar("l_nonzeros", s.l_nonzeros);
  stream << std::setprecision(17);
  scalar("min_abs_pivot", s.min_abs_pivot); scalar("selection_threshold", s.selection_threshold);
  stream << ",\"qr_default_threshold\":";
  if (std::isfinite(s.qr_default_threshold)) stream << std::setprecision(17) << s.qr_default_threshold;
  else stream << "null";
  const auto array = [&](const char* name, const std::vector<Index>& values) {
    stream << ",\"" << name << "\":[";
    bool first = true;
    for (const auto value : values) { if (!first) stream << ','; stream << value; first = false; }
    stream << ']';
  };
  array("ordered_rows", s.ordered_rows);
  array("qr_columns_permutation", s.qr_columns_permutation);
  array("selected_rows", s.selected_rows);
  array("pivot_columns", s.pivot_columns);
  stream << '}';
}

}  // 命名空间 solver::v69_basis
