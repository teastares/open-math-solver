#pragma once

#include <Eigen/SparseLU>
#include <Eigen/SparseQR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <ostream>
#include <span>
#include <string>
#include <vector>

namespace solver::v68_basis {

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
};

struct Stats {
  Index n = 0, m = 0, input_nonzeros = 0, rank = 0;
  Index qr_q_nonzeros = 0, qr_r_nonzeros = 0, lu_l_nonzeros = 0, lu_u_nonzeros = 0;
  Index budget_checks = 0, qr_analyze_calls = 0, qr_factor_calls = 0;
  Index lu_analyze_calls = 0, lu_factor_calls = 0;
  Index primal_solves = 0, dual_solves = 0, slack_matvecs = 0;
  double qr_default_threshold = 0;
  std::vector<Index> ordered_rows, qr_columns_permutation, selected_rows;
};

struct Result {
  std::string status = "not_started";
  bool candidate_available = false;
  std::vector<double> x, s, z;
  Stats stats;
};

namespace implementation {

class CheckedQr : public Eigen::SparseQR<Matrix, Eigen::NaturalOrdering<Index>> {
 public:
  Index reflector_nonzeros() const { return static_cast<Index>(m_Q.nonZeros()); }
  bool finite_factors(const std::function<void()>& check) const {
    for (const auto* matrix : {&m_Q, &m_R}) {
      for (Eigen::Index col = 0; col < matrix->outerSize(); ++col) {
        check();
        for (Matrix::InnerIterator entry(*matrix, col); entry; ++entry)
          if (!std::isfinite(entry.value())) return false;
      }
    }
    for (Eigen::Index i = 0; i < m_hcoeffs.size(); ++i) {
      if ((i & 255) == 0) check();
      if (!std::isfinite(m_hcoeffs[i])) return false;
    }
    return true;
  }
};

class CheckedLu : public Eigen::SparseLU<Matrix, Eigen::COLAMDOrdering<Index>> {
 public:
  // Eigen 部分内存失败分支不更新 info，另核对内部的因子有效标志。
  bool factorization_ok() const { return m_factorizationIsOk; }
  bool finite_factors(const std::function<void()>& check) const {
    for (Eigen::Index col = 0; col < cols(); ++col) {
      check();
      for (SCMatrix::InnerIterator entry(m_Lstore, col); entry; ++entry)
        if (!std::isfinite(entry.value())) return false;
      for (decltype(m_Ustore)::InnerIterator entry(m_Ustore, col); entry; ++entry)
        if (!std::isfinite(entry.value())) return false;
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
      limits.max_factor_nonzeros < 0 || input.starts.size() != input.costs.size() + 1 ||
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
  stats.ordered_rows.resize(static_cast<std::size_t>(m));
  std::iota(stats.ordered_rows.begin(), stats.ordered_rows.end(), Index{0});
  // 比较器包含原行号，因此同分顺序固定，不依赖排序实现的稳定性。
  std::sort(stats.ordered_rows.begin(), stats.ordered_rows.end(), [&](Index a, Index b) {
    check();
    if (input.inequality[a] != input.inequality[b]) return !input.inequality[a];
    if (input.inequality[a]) {
      if (scores[a].exponent != scores[b].exponent) return scores[a].exponent > scores[b].exponent;
      if (scores[a].mantissa != scores[b].mantissa) return scores[a].mantissa > scores[b].mantissa;
    }
    return a < b;
  });
  std::vector<Index> inverse_order(static_cast<std::size_t>(m));
  Matrix ordered_transpose(n, m);
  ordered_transpose.resizeNonZeros(stats.input_nonzeros);
  ordered_transpose.outerIndexPtr()[0] = 0;
  for (Index position = 0; position < m; ++position) {
    check();
    const Index row = stats.ordered_rows[position];
    inverse_order[row] = position;
    ordered_transpose.outerIndexPtr()[position + 1] =
        ordered_transpose.outerIndexPtr()[position] + row_counts[row];
  }
  std::vector<Index> next(ordered_transpose.outerIndexPtr(), ordered_transpose.outerIndexPtr() + m);
  for (Index col = 0; col < n; ++col) {
    check();
    for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
      if ((k & 255) == 0) check();
      const Index target = next[inverse_order[input.rows[k]]]++;
      ordered_transpose.innerIndexPtr()[target] = col;
      ordered_transpose.valuePtr()[target] = input.values[k];
    }
  }
  double max_norm = 0;
  for (Index col = 0; col < m; ++col) {
    check();
    const double norm = ordered_transpose.col(col).norm();
    if (!std::isfinite(norm)) { output.status = "qr_threshold_nonfinite"; return; }
    max_norm = std::max(max_norm, norm);
  }
  if (max_norm == 0) max_norm = 1;
  stats.qr_default_threshold = 20.0 * static_cast<double>(n + m) * max_norm *
      std::numeric_limits<double>::epsilon();
  if (!std::isfinite(stats.qr_default_threshold)) { output.status = "qr_threshold_nonfinite"; return; }
  implementation::CheckedQr qr;
  // 不设置额外阈值；以上仅记录与 Eigen 官方默认公式一致的诊断值。
  check();
  ++stats.qr_analyze_calls;
  qr.analyzePattern(ordered_transpose);
  check();
  ++stats.qr_factor_calls;
  qr.factorize(ordered_transpose);
  check();
  if (qr.info() != Eigen::Success) { output.status = "qr_failure"; return; }
  stats.rank = static_cast<Index>(qr.rank());
  stats.qr_q_nonzeros = qr.reflector_nonzeros();
  stats.qr_r_nonzeros = static_cast<Index>(qr.matrixR().nonZeros());
  if (implementation::exceeds_sum(stats.qr_q_nonzeros, stats.qr_r_nonzeros, limits.max_factor_nonzeros)) {
    output.status = "qr_factor_limit";
    return;
  }
  if (!qr.finite_factors(check)) { output.status = "qr_nonfinite"; return; }
  if (stats.rank != n) { output.status = "numerical_rank_deficient"; return; }
  const auto& permutation = qr.colsPermutation().indices();
  if (permutation.size() != m) { output.status = "invalid_qr_permutation"; return; }
  std::vector<bool> seen(static_cast<std::size_t>(m), false);
  stats.qr_columns_permutation.resize(static_cast<std::size_t>(m));
  stats.selected_rows.resize(static_cast<std::size_t>(n));
  std::vector<Index> basis_position(static_cast<std::size_t>(m), -1);
  for (Index position = 0; position < m; ++position) {
    check();
    const Index source = permutation[position];
    if (source < 0 || source >= m || seen[source]) { output.status = "invalid_qr_permutation"; return; }
    seen[source] = true;
    stats.qr_columns_permutation[position] = source;
    if (position < n) {
      // A*P 的第 position 列来自 A 的 P.indices()[position] 列。
      const Index row = stats.ordered_rows[source];
      stats.selected_rows[position] = row;
      basis_position[row] = position;
    }
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
  stream << '}';
}

}  // 命名空间 solver::v68_basis
