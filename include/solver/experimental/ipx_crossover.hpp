#pragma once

// 本适配器复用 HiGHS 1.15.1 的 IPX crossover 与 BASICLU，不是自研交叉算法。
// 贴界/置零规则参考锁定 ipx/iterate.cc 的 DropToComplementarity；此处仅有净对偶，
// 将其拆成正负部分，并不重建原内点状态的两侧界乘子。IPX 与 BASICLU 的归属保持。
// 来源：https://github.com/ERGO-Code/HiGHS/tree/v1.15.1/highs/ipm
//
// MIT License
// Copyright (c) 2026 HiGHS
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "solver/lp_optimality.hpp"
#include "ipm/ipx/lp_solver.h"
#include "parallel/HighsParallel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace solver::experimental::ipx_crossover {
using Clock = std::chrono::steady_clock;

enum class Status {
  accepted, rejected, invalid_input, numerical_error, time_limit, native_error,
  unsupported_empty, unsupported_integer, unsupported_threads, exception
};

inline const char* status_name(Status status) noexcept {
  switch (status) {
    case Status::accepted: return "accepted";
    case Status::rejected: return "rejected";
    case Status::invalid_input: return "invalid_input";
    case Status::numerical_error: return "numerical_error";
    case Status::time_limit: return "time_limit";
    case Status::native_error: return "native_error";
    case Status::unsupported_empty: return "unsupported_empty";
    case Status::unsupported_integer: return "unsupported_integer";
    case Status::unsupported_threads: return "unsupported_threads";
    case Status::exception: return "exception";
  }
  return "exception";
}

struct Failure : std::runtime_error {
  Status status;
  Failure(Status value, const char* message) : std::runtime_error(message), status(value) {}
};

struct Residuals {
  long double primal = 0, dual = 0, bounds = 0, complementarity = 0, dual_sign = 0;
};
struct PreparationStats {
  std::size_t duplicates_merged = 0, rounded_aggregates = 0;
  std::size_t removed_free_rows = 0, free_row_duals_zeroed = 0;
  std::size_t primal_changes = 0, dual_changes = 0;
  long double max_aggregation_rounding = 0;
  Residuals raw, dropped;
};
struct Prepared {
  std::size_t original_columns = 0, original_rows = 0;
  std::vector<ipx::Int> column_start, row_index;
  std::vector<double> values, costs, lower, upper, rhs;
  std::vector<char> row_type;
  double objective_offset = 0;
  std::vector<std::size_t> original_row, auxiliary_row;
  std::vector<double> x, slack, y, z;
  PreparationStats stats;
};
struct BasisExport {
  Prepared prepared;
  std::vector<ipx::Int> variable_status, row_status;
  std::vector<double> structural_seed;
};
struct BasisExportStats {
  bool enabled = false, statuses_requested = false, statuses_received = false, published = false;
  std::uint64_t columns = 0, rows = 0;
};
struct ProjectionStats {
  bool enabled = false;
  std::uint64_t scan_started = 0, scan_completed = 0, scan_items = 0, eligible_rows = 0;
  std::uint64_t project_started = 0, project_completed = 0;
  std::optional<bool> project_rc;
  std::uint64_t changed_rows = 0;
  double max_abs_change = 0, seconds = 0;
};
struct Stats {
  PreparationStats preparation;
  ProjectionStats projection;
  std::uint64_t load_started = 0, load_completed = 0;
  std::uint64_t crossover_started = 0, crossover_completed = 0;
  std::uint64_t get_started = 0, get_completed = 0;
  std::uint64_t verification_started = 0, verification_completed = 0;
  std::uint64_t callback_checks = 0, callback_interrupts = 0;
  std::optional<ipx::Int> load_rc, crossover_rc, get_rc;
  std::optional<ipx::Info> native_info;
  int native_threads = 0;
  double native_time_limit = 0;
  double prepare_seconds = 0, load_seconds = 0, crossover_seconds = 0;
  double get_seconds = 0, native_scope_seconds = 0, verification_seconds = 0, total_seconds = 0;
};
struct Result {
  Status status = Status::exception;
  std::string message;
  std::vector<double> x, y;
  std::optional<LpVerification> verification;
  // 仅开始第二次验收前保存第一报告；尚无第二次验收时，verification就是原报告。
  std::optional<LpVerification> raw_verification;
  Stats stats;
  std::optional<BasisExport> basis_export;
  BasisExportStats basis_stats;
};

namespace detail {
inline void check(Clock::time_point deadline) {
  if (Clock::now() >= deadline) throw Failure(Status::time_limit, "交叉组件已达到共享截止时点");
}
inline void tick(std::size_t index, Clock::time_point deadline) {
  if ((index & 4095U) == 0) check(deadline);
}
inline double narrow(long double value) {
  const double stored = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(stored) || (value != 0 && stored == 0))
    throw Failure(Status::numerical_error, "交叉映射不能有限且非零保持地窄化");
  return stored;
}
inline ipx::Int index(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<ipx::Int>::max()))
    throw Failure(Status::invalid_input, "交叉映射索引超出锁定IPX整数范围");
  return static_cast<ipx::Int>(value);
}
inline void finite_add(long double& target, long double term) {
  if (!std::isfinite(term)) throw Failure(Status::numerical_error, "交叉映射乘积非有限");
  target += term;
  if (!std::isfinite(target)) throw Failure(Status::numerical_error, "交叉映射累加非有限");
}

// 以净对偶的最小正负分拆选择贴界或置零，所有输入均来自同一未修改快照。
inline std::pair<double, double> drop_pair(double value, double lower, double upper, double dual) {
  const double clipped = std::clamp(value, lower, upper);
  if (lower == upper) return {lower, dual};
  if (!std::isfinite(lower) && !std::isfinite(upper)) return {clipped, 0.0};
  const double positive = std::max(dual, 0.0), negative = std::max(-dual, 0.0);
  if (std::isfinite(lower) && positive > 0) {
    if (static_cast<long double>(positive) >= static_cast<long double>(clipped) - lower)
      return {lower, positive};
  }
  if (std::isfinite(upper) && negative > 0) {
    if (static_cast<long double>(negative) >= static_cast<long double>(upper) - clipped)
      return {upper, -negative};
  }
  return {clipped, 0.0};
}

inline void pair_residual(Residuals& result, double x, double z, double lower, double upper) {
  if (std::isfinite(lower)) result.bounds = std::max(result.bounds, static_cast<long double>(lower) - x);
  if (std::isfinite(upper)) result.bounds = std::max(result.bounds, static_cast<long double>(x) - upper);
  if (x != lower && z > 0) result.dual_sign = std::max(result.dual_sign, static_cast<long double>(z));
  if (x != upper && z < 0) result.dual_sign = std::max(result.dual_sign, -static_cast<long double>(z));
  if (z > 0 && std::isfinite(lower))
    result.complementarity = std::max(result.complementarity, std::abs((static_cast<long double>(x) - lower) * z));
  if (z < 0 && std::isfinite(upper))
    result.complementarity = std::max(result.complementarity, std::abs((static_cast<long double>(upper) - x) * z));
  if (!std::isfinite(result.complementarity)) throw Failure(Status::numerical_error, "交叉互补残差非有限");
}

inline Residuals residuals(const Prepared& p, Clock::time_point deadline) {
  Residuals result;
  std::vector<long double> row(p.rhs.size());
  for (std::size_t i = 0; i < row.size(); ++i) {
    tick(i, deadline);
    row[i] = static_cast<long double>(p.slack[i]) - p.rhs[i];
    const double inf = std::numeric_limits<double>::infinity();
    const double lower = p.row_type[i] == '>' ? -inf : 0;
    const double upper = p.row_type[i] == '<' ? inf : 0;
    pair_residual(result, p.slack[i], -p.y[i], lower, upper);
  }
  for (std::size_t j = 0; j < p.costs.size(); ++j) {
    tick(j, deadline);
    long double dual = static_cast<long double>(p.z[j]) - p.costs[j];
    pair_residual(result, p.x[j], p.z[j], p.lower[j], p.upper[j]);
    for (ipx::Int k = p.column_start[j]; k < p.column_start[j + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k), i = static_cast<std::size_t>(p.row_index[pos]);
      tick(pos, deadline);
      finite_add(row[i], static_cast<long double>(p.values[pos]) * p.x[j]);
      finite_add(dual, static_cast<long double>(p.values[pos]) * p.y[i]);
    }
    result.dual = std::max(result.dual, std::abs(dual));
  }
  for (std::size_t i = 0; i < row.size(); ++i) {
    tick(i, deadline);
    result.primal = std::max(result.primal, std::abs(row[i]));
  }
  check(deadline);
  return result;
}

struct Timed {
  double& seconds;
  Clock::time_point start = Clock::now();
  explicit Timed(double& destination) : seconds(destination) {}
  ~Timed() { seconds += std::chrono::duration<double>(Clock::now() - start).count(); }
};
inline void validate_tolerances(const Tolerances& t, const LpOptimalityOptions& o) {
  for (double value : {t.feasibility_absolute, t.feasibility_relative, t.integrality,
      t.objective_absolute, t.objective_relative, o.dual_absolute, o.dual_relative,
      o.gap_absolute, o.gap_relative})
    if (!std::isfinite(value) || value < 0)
      throw Failure(Status::invalid_input, "交叉组件容差必须有限且非负");
}
inline bool complete_quality(const LpVerification& checked) noexcept {
  return checked.primal.valid_input && checked.primal.feasible && checked.primal.objective_consistent &&
      checked.primal.objective.has_value() && checked.optimality.valid_input && checked.optimality.numerically_optimal;
}
}  // 命名空间 detail

// 两次验收使用同一生产预算；检查器内部不支持抢占，出口检查仍可拒绝已经算出的点。
template<class Budget>
inline void verify_candidate_once(const Model& model, std::span<const double> x,
    std::span<const double> y, Result& result, const Tolerances& tolerances,
    const LpOptimalityOptions& optimality_options, Budget&& budget) {
  detail::Timed timer(result.stats.verification_seconds);
  budget();
  if (result.stats.verification_started >= 2)
    throw Failure(Status::invalid_input, "交叉候选最多允许两次完整验收");
  ++result.stats.verification_started;
  result.verification = verify_lp_solution(model, x, y, tolerances, optimality_options);
  ++result.stats.verification_completed;
  budget();
}

// 返回拥有的试点向量并不代表通过验收；调用者必须检查最后的完整质量和总截止时间。
template<class Budget>
inline std::optional<std::vector<double>> project_rejected_row_dual(const Model& model,
    std::span<const double> x, std::span<const double> raw_y, Result& result,
    const Tolerances& tolerances, const LpOptimalityOptions& optimality_options, Budget&& budget) {
  auto& stats = result.stats.projection;
  if (!stats.enabled) return std::nullopt;
  if (!result.verification)
    throw Failure(Status::invalid_input, "行对偶域投影缺少第一次完整验收报告");
  if (detail::complete_quality(*result.verification)) return std::nullopt;
  std::vector<double> projected_y;
  {
    detail::Timed timer(stats.seconds);
    budget();
    if (stats.scan_started != 0 || stats.project_started != 0 || result.raw_verification ||
        result.stats.verification_started != 1 || result.stats.verification_completed != 1)
      throw Failure(Status::invalid_input, "同一交叉候选的行对偶域投影最多一次");
    ++stats.scan_started;
    if (raw_y.size() != model.row_lower.size() || model.row_lower.size() != model.row_upper.size() ||
        x.size() != model.costs.size())
      throw Failure(Status::invalid_input, "行对偶域投影原坐标维数不匹配");
    for (std::size_t row = 0; row < raw_y.size(); ++row) {
      if ((row & 4095U) == 0) budget();
      ++stats.scan_items;
      if (!std::isfinite(raw_y[row]))
        throw Failure(Status::numerical_error, "行对偶域投影候选非有限");
      if ((!std::isfinite(model.row_lower[row]) && raw_y[row] > 0.0) ||
          (!std::isfinite(model.row_upper[row]) && raw_y[row] < 0.0))
        ++stats.eligible_rows;
    }
    ++stats.scan_completed;
    budget();
    if (stats.eligible_rows == 0) return std::nullopt;
    budget();
    projected_y.assign(raw_y.begin(), raw_y.end());
    budget();
    ++stats.project_started;
    stats.project_rc = project_lp_row_dual_domain(model, projected_y);
    ++stats.project_completed;
    budget();
    if (!*stats.project_rc)
      throw Failure(Status::invalid_input, "行对偶域投影的行界或维数无效");
    for (std::size_t row = 0; row < raw_y.size(); ++row) {
      if ((row & 4095U) == 0) budget();
      if (projected_y[row] != raw_y[row]) {
        ++stats.changed_rows;
        // 既有投影只将有限乘子置零，绝对变化不会超出double范围。
        stats.max_abs_change = std::max(stats.max_abs_change, std::abs(raw_y[row]));
      }
    }
    budget();
    if (stats.changed_rows == 0) return std::nullopt;
  }
  budget();
  result.raw_verification = std::move(result.verification);
  result.verification.reset();
  verify_candidate_once(model, x, projected_y, result, tolerances, optimality_options, budget);
  budget();
  return std::optional<std::vector<double>>(std::move(projected_y));
}

inline Prepared prepare(const Model& model, std::span<const double> primal,
    std::span<const double> normalized_row_dual, Clock::time_point absolute_deadline) {
  detail::check(absolute_deadline);
  const auto errors = validate_model(model);
  detail::check(absolute_deadline);
  if (!errors.empty()) throw Failure(Status::invalid_input, "交叉组件原模型非法");
  if (is_mip(model)) throw Failure(Status::unsupported_integer, "独立交叉组件仅接受连续LP");
  const auto n = model.costs.size(), m = model.row_lower.size();
  if (primal.size() != n || normalized_row_dual.size() != m)
    throw Failure(Status::invalid_input, "交叉中间点维数不匹配");
  for (std::size_t j = 0; j < n; ++j) {
    detail::tick(j, absolute_deadline);
    if (!std::isfinite(primal[j])) throw Failure(Status::invalid_input, "交叉原始点非有限");
  }
  for (std::size_t i = 0; i < m; ++i) {
    detail::tick(i, absolute_deadline);
    if (!std::isfinite(normalized_row_dual[i])) throw Failure(Status::invalid_input, "交叉行对偶非有限");
  }
  detail::index(n); detail::index(m); detail::index(model.values.size());
  Prepared p;
  p.original_columns = n; p.original_rows = m;
  std::vector<ipx::Int> row_map(m, -1);
  for (std::size_t i = 0; i < m; ++i) {
    detail::tick(i, absolute_deadline);
    const double lower = model.row_lower[i], upper = model.row_upper[i];
    if (!std::isfinite(lower) && !std::isfinite(upper)) {
      ++p.stats.removed_free_rows;
      if (normalized_row_dual[i] != 0) ++p.stats.free_row_duals_zeroed;
      continue;
    }
    row_map[i] = detail::index(p.original_row.size());
    p.original_row.push_back(i);
    if (lower == upper) { p.row_type.push_back('='); p.rhs.push_back(lower); }
    else if (std::isfinite(lower) && std::isfinite(upper)) {
      p.row_type.push_back('='); p.rhs.push_back(0); p.auxiliary_row.push_back(i);
    } else if (std::isfinite(lower)) { p.row_type.push_back('>'); p.rhs.push_back(lower); }
    else { p.row_type.push_back('<'); p.rhs.push_back(upper); }
  }
  if (n == 0 || p.original_row.empty())
    throw Failure(Status::unsupported_empty, "映射后零行或零列不进入锁定IPX基组件");
  detail::index(n + p.auxiliary_row.size());
  detail::index(n + p.auxiliary_row.size() + p.original_row.size());
  const double sense = model.sense == ObjectiveSense::minimize ? 1 : -1;
  p.objective_offset = sense * model.objective_offset;
  p.costs.reserve(n + p.auxiliary_row.size());
  for (double value : model.costs) p.costs.push_back(sense * value);
  p.lower = model.column_lower; p.upper = model.column_upper;
  p.x.assign(primal.begin(), primal.end()); p.z.resize(n);
  std::vector<long double> activity(m, 0);
  p.column_start.push_back(0);
  std::vector<std::pair<ipx::Int, double>> column;
  for (std::size_t j = 0; j < n; ++j) {
    detail::tick(j, absolute_deadline);
    long double reduced_cost = p.costs[j];
    column.clear();
    for (std::int64_t k = model.column_start[j]; k < model.column_start[j + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k), i = static_cast<std::size_t>(model.row_index[pos]);
      detail::tick(pos, absolute_deadline);
      detail::finite_add(activity[i], static_cast<long double>(model.values[pos]) * primal[j]);
      detail::finite_add(reduced_cost, -static_cast<long double>(model.values[pos]) * normalized_row_dual[i]);
      if (row_map[i] >= 0) column.emplace_back(row_map[i], model.values[pos]);
    }
    p.z[j] = detail::narrow(reduced_cost);
    // 同行项保持原存储顺序；排序与库分解一样，只能在操作前后检查截止。
    detail::check(absolute_deadline);
    std::stable_sort(column.begin(), column.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    detail::check(absolute_deadline);
    for (std::size_t begin = 0; begin < column.size();) {
      std::size_t end = begin;
      long double sum = 0;
      while (end < column.size() && column[end].first == column[begin].first) {
        detail::tick(end, absolute_deadline);
        detail::finite_add(sum, column[end].second); ++end;
      }
      p.stats.duplicates_merged += end - begin - 1;
      const double stored = detail::narrow(sum);
      const long double rounding = std::abs(sum - static_cast<long double>(stored));
      if (rounding != 0) ++p.stats.rounded_aggregates;
      p.stats.max_aggregation_rounding = std::max(p.stats.max_aggregation_rounding, rounding);
      if (stored != 0) { p.row_index.push_back(column[begin].first); p.values.push_back(stored); }
      begin = end;
    }
    p.column_start.push_back(detail::index(p.values.size()));
  }
  for (std::size_t k = 0; k < p.auxiliary_row.size(); ++k) {
    detail::tick(k, absolute_deadline);
    const auto original = p.auxiliary_row[k];
    p.costs.push_back(0); p.lower.push_back(model.row_lower[original]); p.upper.push_back(model.row_upper[original]);
    p.x.push_back(detail::narrow(activity[original]));
    p.z.push_back(normalized_row_dual[original]);  // Ax-q=0 的 q 约化成本是 +y。
    p.row_index.push_back(row_map[original]); p.values.push_back(-1);
    p.column_start.push_back(detail::index(p.values.size()));
  }
  detail::index(p.values.size() + p.original_row.size());
  p.y.resize(p.original_row.size()); p.slack.resize(p.original_row.size());
  for (std::size_t i = 0; i < p.original_row.size(); ++i) {
    detail::tick(i, absolute_deadline);
    const auto original = p.original_row[i];
    p.y[i] = normalized_row_dual[original];
    p.slack[i] = p.row_type[i] == '=' ? 0 : detail::narrow(static_cast<long double>(p.rhs[i]) - activity[original]);
  }
  p.stats.raw = detail::residuals(p, absolute_deadline);
  for (std::size_t j = 0; j < p.costs.size(); ++j) {
    detail::tick(j, absolute_deadline);
    const auto dropped = detail::drop_pair(p.x[j], p.lower[j], p.upper[j], p.z[j]);
    p.stats.primal_changes += p.x[j] != dropped.first;
    p.stats.dual_changes += p.z[j] != dropped.second;
    p.x[j] = dropped.first; p.z[j] = dropped.second;
  }
  for (std::size_t i = 0; i < p.original_row.size(); ++i) {
    detail::tick(i, absolute_deadline);
    if (p.row_type[i] == '=') continue;
    const double inf = std::numeric_limits<double>::infinity();
    const auto dropped = detail::drop_pair(p.slack[i], p.row_type[i] == '>' ? -inf : 0,
        p.row_type[i] == '<' ? inf : 0, -p.y[i]);
    p.stats.primal_changes += p.slack[i] != dropped.first;
    p.stats.dual_changes += -p.y[i] != dropped.second;
    p.slack[i] = dropped.first; p.y[i] = -dropped.second;
  }
  for (std::size_t k = 0; k < p.auxiliary_row.size(); ++k) {
    detail::tick(k, absolute_deadline);
    p.y[static_cast<std::size_t>(row_map[p.auxiliary_row[k]])] = p.z[n + k];
  }
  p.stats.dropped = detail::residuals(p, absolute_deadline);
  if (p.stats.dropped.bounds != 0 || p.stats.dropped.complementarity != 0 || p.stats.dropped.dual_sign != 0)
    throw Failure(Status::numerical_error, "交叉起点不满足精确界与互补域");
  detail::check(absolute_deadline);
  return p;
}

inline Result run(const Model& model, std::span<const double> primal,
    std::span<const double> normalized_row_dual, Clock::time_point absolute_deadline,
    const Tolerances& tolerances = {}, const LpOptimalityOptions& optimality_options = {},
    bool enable_row_dual_projection = false, bool export_rejected_basis = false) {
  const auto start = Clock::now();
  Result result;
  result.stats.projection.enabled = enable_row_dual_projection;
  result.basis_stats.enabled = export_rejected_basis;
  try {
    detail::check(absolute_deadline);
    detail::validate_tolerances(tolerances, optimality_options);
    std::vector<double> candidate_x, candidate_y;
    std::optional<BasisExport> pending_basis;
    {
      Prepared p;
      {
        detail::Timed timer(result.stats.prepare_seconds);
        p = prepare(model, primal, normalized_row_dual, absolute_deadline);
      }
      result.stats.preparation = p.stats;
      detail::check(absolute_deadline);
      {
        detail::Timed scope_timer(result.stats.native_scope_seconds);
        highs::parallel::initialize_scheduler(1);
        result.stats.native_threads = highs::parallel::num_threads();
        detail::check(absolute_deadline);
        if (result.stats.native_threads != 1)
          throw Failure(Status::unsupported_threads, "现有共享scheduler不是单线程，交叉组件未执行");
        HighsCallback callback(nullptr);
        callback.active[kCallbackIpmInterrupt] = true;
        callback.user_callback = [&stats = result.stats, absolute_deadline](int, const std::string&,
            const HighsCallbackOutput*, HighsCallbackInput* input, void*) noexcept {
          ++stats.callback_checks;
          const bool expired = Clock::now() >= absolute_deadline;
          if (expired) ++stats.callback_interrupts;
          input->user_interrupt = expired;
        };
        // callback 先构造，保证其生命周期覆盖 IPX 对象及其析构。
        ipx::LpSolver native;
        detail::check(absolute_deadline);
        result.stats.native_time_limit = std::chrono::duration<double>(absolute_deadline - Clock::now()).count();
        if (!(result.stats.native_time_limit > 0) || !std::isfinite(result.stats.native_time_limit))
          throw Failure(Status::time_limit, "交叉组件没有有限正剩余预算");
        ipx::Parameters parameters;
        parameters.dualize = 0; parameters.scale = 0; parameters.display = 0;
        parameters.debug = 0; parameters.highs_logging = false; parameters.logfile = nullptr;
        parameters.crash_basis = 1; parameters.run_crossover = 1; parameters.ipm_maxiter = 0;
        parameters.time_limit = result.stats.native_time_limit;
        parameters.pfeasibility_tol = tolerances.feasibility_absolute;
        parameters.dfeasibility_tol = optimality_options.dual_absolute;
        native.SetParameters(parameters);
        native.SetCallback(&callback);
        const ipx::Int dummy_index = 0;
        const double dummy_value = 0;
        {
          detail::Timed timer(result.stats.load_seconds);
          detail::check(absolute_deadline); ++result.stats.load_started;
          result.stats.load_rc = native.LoadModel(detail::index(p.costs.size()), p.objective_offset,
              p.costs.data(), p.lower.data(), p.upper.data(), detail::index(p.rhs.size()), p.column_start.data(),
              p.row_index.empty() ? &dummy_index : p.row_index.data(),
              p.values.empty() ? &dummy_value : p.values.data(), p.rhs.data(), p.row_type.data());
          ++result.stats.load_completed;
          result.stats.native_info = native.GetInfo();
          detail::check(absolute_deadline);
        }
        if (*result.stats.load_rc != 0) throw Failure(Status::native_error, "IPX模型装载失败");
        {
          detail::Timed timer(result.stats.crossover_seconds);
          detail::check(absolute_deadline); ++result.stats.crossover_started;
          try {
            result.stats.crossover_rc = native.CrossoverFromStartingPoint(p.x.data(), p.slack.data(), p.y.data(), p.z.data());
            ++result.stats.crossover_completed;
          } catch (...) {
            result.stats.native_info = native.GetInfo();
            throw;
          }
          result.stats.native_info = native.GetInfo();
          detail::check(absolute_deadline);
        }
        const auto& info = *result.stats.native_info;
        if (info.status_ipm != IPX_STATUS_not_run || info.iter != 0 || info.kktiter1 != 0 ||
            info.kktiter2 != 0 || info.updates_ipm != 0)
          throw Failure(Status::native_error, "独立交叉入口观察到不应发生的IPM工作");
        if (*result.stats.crossover_rc != 0)
          throw Failure(Status::native_error, "IPX交叉起点被拒绝");
        if (info.status_crossover == IPX_STATUS_time_limit || info.status_crossover == IPX_STATUS_user_interrupt)
          throw Failure(Status::time_limit, "IPX交叉收到共享时限中断");
        if (info.status_crossover != IPX_STATUS_optimal && info.status_crossover != IPX_STATUS_imprecise)
          throw Failure(Status::native_error, "IPX交叉没有可取回的完整基解");
        std::vector<double> native_x(p.costs.size()), native_y(p.rhs.size());
        std::vector<ipx::Int> variable_status, row_status;
        {
          detail::Timed timer(result.stats.get_seconds);
          detail::check(absolute_deadline);
          if (export_rejected_basis) {
            variable_status.resize(p.costs.size());
            result.basis_stats.columns = static_cast<std::uint64_t>(variable_status.size());
            detail::check(absolute_deadline);
            row_status.resize(p.rhs.size());
            result.basis_stats.rows = static_cast<std::uint64_t>(row_status.size());
            detail::check(absolute_deadline);
          }
          ++result.stats.get_started;
          if (export_rejected_basis) result.basis_stats.statuses_requested = true;
          result.stats.get_rc = native.GetBasicSolution(native_x.data(), nullptr, native_y.data(), nullptr,
              export_rejected_basis ? row_status.data() : nullptr,
              export_rejected_basis ? variable_status.data() : nullptr);
          ++result.stats.get_completed;
          if (export_rejected_basis && *result.stats.get_rc == 0) result.basis_stats.statuses_received = true;
          detail::check(absolute_deadline);
        }
        if (*result.stats.get_rc != 0) throw Failure(Status::native_error, "IPX完整基解取回失败");
        candidate_x.resize(p.original_columns); candidate_y.assign(p.original_rows, 0);
        for (std::size_t j = 0; j < p.original_columns; ++j) {
          detail::tick(j, absolute_deadline);
          if (!std::isfinite(native_x[j])) throw Failure(Status::numerical_error, "交叉返回原始点非有限");
          candidate_x[j] = native_x[j];
        }
        for (std::size_t i = 0; i < p.original_row.size(); ++i) {
          detail::tick(i, absolute_deadline);
          if (!std::isfinite(native_y[i])) throw Failure(Status::numerical_error, "交叉返回行对偶非有限");
          candidate_y[p.original_row[i]] = native_y[i];
        }
        detail::check(absolute_deadline);
        if (export_rejected_basis) {
          auto& pending = pending_basis.emplace();
          pending.variable_status = std::move(variable_status);
          pending.row_status = std::move(row_status);
          // 必须保留同一次Get返回的全部结构点，包括范围辅助变量；p.x是Drop起点。
          pending.structural_seed = std::move(native_x);
          detail::check(absolute_deadline);
        }
      }
      detail::check(absolute_deadline);
      if (export_rejected_basis) {
        pending_basis->prepared = std::move(p);
        detail::check(absolute_deadline);
      }
    }
    detail::check(absolute_deadline);
    const auto budget = [absolute_deadline] { detail::check(absolute_deadline); };
    verify_candidate_once(model, candidate_x, candidate_y, result, tolerances, optimality_options, budget);
    if (enable_row_dual_projection) {
      if (auto projected_y = project_rejected_row_dual(model, candidate_x, candidate_y, result,
          tolerances, optimality_options, budget))
        candidate_y = std::move(*projected_y);
    }
    const auto& checked = *result.verification;
    if (detail::complete_quality(checked)) {
      result.x = std::move(candidate_x); result.y = std::move(candidate_y);
      result.status = Status::accepted;
      result.message = "独立交叉候选通过原模型完整验收";
    } else {
      result.status = Status::rejected;
      result.message = "交叉返回候选未通过原模型完整验收";
      if (export_rejected_basis && pending_basis) {
        detail::check(absolute_deadline);
        result.basis_export = std::move(pending_basis);
        pending_basis.reset();
        detail::check(absolute_deadline);
      }
    }
  } catch (const Failure& failure) {
    result.status = failure.status; result.message = failure.what();
  } catch (const std::exception& failure) {
    result.status = Status::exception; result.message = failure.what();
  } catch (...) {
    result.status = Status::exception; result.message = "交叉组件出现未知异常";
  }
  // 原生对象和未发布的局部点、映射已析构；导出快照的后续生命周期由返回值持有。
  if (result.status != Status::rejected) result.basis_export.reset();
  auto finish = Clock::now();
  if (finish >= absolute_deadline) {
    result.status = Status::time_limit;
    result.message = "交叉组件含清理与原验的总成本达到共享截止时点";
  }
  if (result.status != Status::accepted) {
    std::vector<double>().swap(result.x); std::vector<double>().swap(result.y);
    if (result.status != Status::rejected) result.basis_export.reset();
    finish = Clock::now();
  }
  if (export_rejected_basis && finish >= absolute_deadline) {
    result.status = Status::time_limit;
    result.message = "交叉基快照含出口清理的总成本达到共享截止时点";
    std::vector<double>().swap(result.x); std::vector<double>().swap(result.y);
    result.basis_export.reset();
    finish = Clock::now();
  }
  result.basis_stats.published = result.status == Status::rejected && result.basis_export.has_value();
  result.stats.total_seconds = std::chrono::duration<double>(finish - start).count();
  return result;
}
}  // 命名空间 solver::experimental::ipx_crossover
