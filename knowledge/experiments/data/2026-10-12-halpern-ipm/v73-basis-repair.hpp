#pragma once

#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <iomanip>
#include <span>
#include <string>
#include <vector>

namespace solver::v73_basis_repair {

using Index = std::int64_t;
using Matrix = Eigen::SparseMatrix<double, Eigen::ColMajor, Index>;

struct Input {
  std::span<const Index> starts, rows;
  std::span<const double> values, costs, rhs;
  const std::vector<bool>& inequality;
};

struct Start {
  std::span<const Index> selected_rows;
  std::span<const double> x, dual;
  Index factor_nonzeros = 0;
};

struct Limits {
  Index max_pivots = 128;
  Index max_repair_work = 100000000;
  Index max_factor_nonzeros = 32000000;
  double roundoff_multiplier = 64;
};

struct Stats {
  std::string phase = "not_started";
  Index attempts = 0, iterations = 0, phase1_iterations = 0, phase2_iterations = 0;
  Index pivots = 0, zero_steps = 0, work_units = 0, budget_checks = 0;
  Index lu_analyze_calls = 0, lu_factor_calls = 0, factor_nonzeros = 0;
  Index primal_solves = 0, dual_solves = 0, direction_solves = 0;
  Index slack_matvecs = 0, direction_matvecs = 0, pricing_matvecs = 0;
  Index roundoff_rows = 0;
  std::optional<double> initial_phi, final_phi;
  std::optional<double> initial_max_violation, final_max_violation;
  std::optional<double> max_roundoff_bound, last_right_derivative, last_step;
  std::optional<double> max_selected_residual, max_direction_residual;
  std::vector<Index> final_selected_rows;
};

struct Result {
  std::string status = "not_started";
  bool candidate_available = false;
  std::vector<double> x, dual;
  std::vector<Index> selected_rows;
  Stats stats;
};

namespace implementation {

struct Stop { const char* status; };

inline std::optional<double> summary(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) return {};
  const double converted = static_cast<double>(value);
  if (value != 0 && converted == 0) return {};
  return converted;
}

class Work {
 public:
  Work(const Limits& limits, const std::function<void()>& budget, Result& result)
      : limits_(limits), budget_(budget), result_(result) {}
  void check() {
    ++result_.stats.budget_checks;
    if (budget_) {
      try { budget_(); }
      catch (...) { result_.status = "budget_exception"; throw; }
    }
  }
  void take() {
    if (result_.stats.work_units >= limits_.max_repair_work) throw Stop{"work_limit"};
    if ((result_.stats.work_units & 255) == 0) check();
    ++result_.stats.work_units;
  }
  void account(Index units) {
    // 因子遍历的元素数可在进入前取得，先按至多 256 项分批计费。
    while (units > 0) {
      check();
      const Index batch = std::min<Index>(units, 256);
      if (batch > limits_.max_repair_work - result_.stats.work_units) throw Stop{"work_limit"};
      result_.stats.work_units += batch;
      units -= batch;
    }
  }
  double narrow(long double value) {
    take();
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
      throw Stop{"arithmetic_failure"};
    const double converted = static_cast<double>(value);
    if (value != 0 && converted == 0) throw Stop{"arithmetic_failure"};
    return converted;
  }
 private:
  const Limits& limits_;
  const std::function<void()>& budget_;
  Result& result_;
};

struct State {
  std::vector<Index> selected, position;
  Eigen::VectorXd x, dual;
  std::vector<long double> slack, scale, roundoff;
  long double phi = 0, max_violation = 0, max_bound = 0, max_selected = 0;
  Index roundoff_rows = 0;
  bool primal_feasible = false, dual_valid = false;
};

inline void finite(long double value) {
  if (!std::isfinite(value)) throw Stop{"arithmetic_failure"};
}

inline void positions(State& state, Index m, Work& work) {
  work.check();
  state.position.resize(static_cast<std::size_t>(m));
  for (Index row = 0; row < m; ++row) { work.take(); state.position[row] = -1; }
  for (Index k = 0; k < static_cast<Index>(state.selected.size()); ++k) {
    work.take();
    const Index row = state.selected[k];
    if (row < 0 || row >= m || state.position[row] >= 0) throw Stop{"invalid_input"};
    state.position[row] = k;
  }
}

// 所有行使用真实矩阵重算；等式的基本 slack 也计入绝对值罚目标。
inline void evaluate(const Input& input, const Limits& limits, State& state,
                     Work& work, Stats& stats) {
  const Index n = static_cast<Index>(input.costs.size()), m = static_cast<Index>(input.rhs.size());
  work.check();
  state.slack.resize(static_cast<std::size_t>(m));
  state.scale.resize(static_cast<std::size_t>(m));
  state.roundoff.resize(static_cast<std::size_t>(m));
  for (Index row = 0; row < m; ++row) {
    work.take(); state.slack[row] = input.rhs[row]; state.scale[row] = std::abs(input.rhs[row]);
  }
  ++stats.slack_matvecs;
  for (Index col = 0; col < n; ++col) {
    work.take();
    if (!std::isfinite(state.x[col])) throw Stop{"arithmetic_failure"};
    for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
      work.take();
      const Index row = input.rows[k];
      const long double product = static_cast<long double>(input.values[k]) * state.x[col];
      finite(product);
      if (input.values[k] != 0 && state.x[col] != 0 && product == 0) throw Stop{"arithmetic_failure"};
      state.slack[row] -= product;
      state.scale[row] += std::abs(product);
      finite(state.slack[row]); finite(state.scale[row]);
    }
  }
  state.phi = state.max_violation = state.max_bound = state.max_selected = 0;
  state.roundoff_rows = 0;
  state.primal_feasible = true;
  const long double factor = static_cast<long double>(limits.roundoff_multiplier) *
      std::numeric_limits<double>::epsilon();
  for (Index row = 0; row < m; ++row) {
    work.take();
    const long double violation = input.inequality[row] ? std::max(-state.slack[row], 0.0L)
                                                       : std::abs(state.slack[row]);
    state.phi += violation;
    state.max_violation = std::max(state.max_violation, violation);
    state.roundoff[row] = factor * (state.scale[row] + 1);
    finite(state.phi); finite(state.roundoff[row]);
    state.max_bound = std::max(state.max_bound, state.roundoff[row]);
    if (violation > state.roundoff[row]) state.primal_feasible = false;
    else if (violation > 0) ++state.roundoff_rows;
    if (state.position[row] >= 0) {
      state.max_selected = std::max(state.max_selected, std::abs(state.slack[row]));
      if (std::abs(state.slack[row]) > state.roundoff[row]) {
        stats.max_selected_residual = summary(state.max_selected);
        stats.max_roundoff_bound = summary(state.max_bound);
        throw Stop{"basis_residual_failure"};
      }
    }
  }
  stats.max_selected_residual = summary(state.max_selected);
}

inline void record(const State& state, Stats& stats) {
  stats.final_phi = summary(state.phi);
  stats.final_max_violation = summary(state.max_violation);
  stats.max_roundoff_bound = summary(state.max_bound);
  stats.roundoff_rows = state.roundoff_rows;
  stats.max_selected_residual = summary(state.max_selected);
}

struct Event { long double theta, jump, magnitude; Index row; };

// 零处按真实右导数计价；不把 q=0 误当作所有方向的导数均为零。
inline long double right_derivative(const Input& input, const State& state,
                                   const std::vector<long double>& direction, Work& work) {
  long double derivative = 0;
  for (Index row = 0; row < static_cast<Index>(input.rhs.size()); ++row) {
    work.take();
    if (state.position[row] >= 0) continue;
    const long double slack = state.slack[row], change = direction[row];
    if (!input.inequality[row]) {
      derivative += slack < 0 ? -change : (slack > 0 ? change : std::abs(change));
    } else if (slack < 0) derivative -= change;
    else if (slack == 0) derivative += std::max(-change, 0.0L);
    finite(derivative);
  }
  return derivative;
}

inline Event phase1_step(const Input& input, const State& state,
                         const std::vector<long double>& direction,
                         long double derivative, Work& work) {
  if (!(derivative < 0)) throw Stop{"no_descent"};
  std::vector<Event> events;
  work.check();
  events.reserve(input.rhs.size());
  for (Index row = 0; row < static_cast<Index>(input.rhs.size()); ++row) {
    work.take();
    if (state.position[row] >= 0) continue;
    const long double slack = state.slack[row], change = direction[row];
    // 零断点的斜率变化已在右导数中计入；其余只收集正向穿零事件。
    if (!((slack < 0 && change > 0) || (slack > 0 && change < 0))) continue;
    const long double theta = -slack / change;
    const long double magnitude = std::abs(change);
    const long double jump = (input.inequality[row] ? 1.0L : 2.0L) * magnitude;
    finite(theta); finite(jump);
    events.push_back({theta, jump, magnitude, row});
  }
  std::sort(events.begin(), events.end(), [&](const Event& a, const Event& b) {
    work.take(); return a.theta != b.theta ? a.theta < b.theta : a.row < b.row;
  });
  for (std::size_t begin = 0; begin < events.size();) {
    work.take();
    std::size_t end = begin;
    Event chosen = events[begin];
    // 同一断点完整合并，再决定是否达到最小点；不依赖排序中的先后次序。
    while (end < events.size() && events[end].theta == events[begin].theta) {
      work.take();
      derivative += events[end].jump;
      finite(derivative);
      if (events[end].magnitude > chosen.magnitude ||
          (events[end].magnitude == chosen.magnitude && events[end].row < chosen.row)) chosen = events[end];
      ++end;
    }
    if (derivative >= 0) return chosen;
    begin = end;
  }
  throw Stop{"no_blocker"};
}

inline Event phase2_step(const Input& input, const State& state,
                         const std::vector<long double>& direction, Work& work) {
  Event chosen{0, 0, 0, -1};
  for (Index row = 0; row < static_cast<Index>(input.rhs.size()); ++row) {
    work.take();
    if (state.position[row] >= 0) continue;
    const long double change = direction[row], slack = state.slack[row];
    if (change == 0 || (input.inequality[row] && change > 0)) continue;
    // 微小负 slack 不能被平移成可行界；它对继续越界的方向保守地给零步。
    const long double theta = input.inequality[row]
        ? std::max(slack, 0.0L) / (-change) : std::max(-slack / change, 0.0L);
    finite(theta);
    const long double magnitude = std::abs(change);
    if (chosen.row < 0 || theta < chosen.theta ||
        (theta == chosen.theta && (magnitude > chosen.magnitude ||
         (magnitude == chosen.magnitude && row < chosen.row)))) chosen = {theta, 0, magnitude, row};
  }
  if (chosen.row < 0) throw Stop{"no_blocker"};
  return chosen;
}

}  // 实现细节命名空间

// 仅生成待原模型验收的候选；所有修复状态均不是不可行或无界证书。
// Lu 复用已有 CheckedLu 接口，初始因子借用，换基后用独立因子事务性替换。
template<class Lu>
void repair(const Input& input, const Limits& limits, const Start& start,
            Lu& initial_lu, const std::function<void()>& budget, Result& output) {
  using namespace implementation;
  output = Result{};
  auto& stats = output.stats;
  Work work(limits, budget, output);
  stats.attempts = 1;
  stats.phase = "validation";
  try {
    work.check();
    const auto largest = static_cast<std::size_t>(std::min<Index>(
        std::numeric_limits<Index>::max(), std::numeric_limits<Eigen::Index>::max()));
    if (input.costs.size() >= largest || input.rhs.size() >= largest || input.values.size() >= largest ||
        limits.max_pivots < 0 || limits.max_repair_work < 0 || limits.max_factor_nonzeros < 0 ||
        !(limits.roundoff_multiplier > 0) || !std::isfinite(limits.roundoff_multiplier)) throw Stop{"invalid_input"};
    const Index n = static_cast<Index>(input.costs.size()), m = static_cast<Index>(input.rhs.size());
    if (n == 0 || m < n || input.starts.size() != input.costs.size() + 1 ||
        input.starts.front() != 0 || input.starts.back() != static_cast<Index>(input.values.size()) ||
        input.rows.size() != input.values.size() || input.inequality.size() != input.rhs.size() ||
        start.selected_rows.size() != input.costs.size() || start.x.size() != input.costs.size() ||
        start.dual.size() != input.costs.size() || start.factor_nonzeros < 0 ||
        initial_lu.rows() != n || initial_lu.cols() != n || !initial_lu.factorization_ok()) throw Stop{"invalid_input"};
    stats.factor_nonzeros = start.factor_nonzeros;
    if (stats.factor_nonzeros > limits.max_factor_nonzeros) throw Stop{"factor_limit"};
    for (Index col = 0; col < n; ++col) {
      work.take();
      if (!std::isfinite(input.costs[col]) || input.starts[col] < 0 ||
          input.starts[col] > input.starts[col + 1] || input.starts[col + 1] > input.starts.back())
        throw Stop{"invalid_input"};
      Index previous = -1;
      for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
        work.take();
        if (input.rows[k] <= previous || input.rows[k] >= m || !std::isfinite(input.values[k])) throw Stop{"invalid_input"};
        previous = input.rows[k];
      }
    }
    for (Index row = 0; row < m; ++row) { work.take(); if (!std::isfinite(input.rhs[row])) throw Stop{"invalid_input"}; }
    State state;
    work.check();
    state.selected.resize(static_cast<std::size_t>(n)); state.x.resize(n); state.dual.resize(n);
    for (Index k = 0; k < n; ++k) {
      work.take();
      state.selected[k] = start.selected_rows[k]; state.x[k] = start.x[k]; state.dual[k] = start.dual[k];
      if (!std::isfinite(state.x[k]) || !std::isfinite(state.dual[k])) throw Stop{"invalid_input"};
    }
    state.dual_valid = true;
    positions(state, m, work);
    stats.phase = "initial_residual";
    evaluate(input, limits, state, work, stats);
    stats.initial_phi = summary(state.phi); stats.initial_max_violation = summary(state.max_violation);
    record(state, stats);
    stats.final_selected_rows.resize(static_cast<std::size_t>(n));
    for (Index k = 0; k < n; ++k) { work.take(); stats.final_selected_rows[k] = state.selected[k]; }
    Lu* lu = &initial_lu;
    std::unique_ptr<Lu> owned_lu;
    const long double epsilon_factor = static_cast<long double>(limits.roundoff_multiplier) *
        std::numeric_limits<double>::epsilon();
    for (;;) {
      work.check();
      const bool phase1 = !state.primal_feasible;
      stats.phase = phase1 ? "phase1" : "phase2";
      Eigen::VectorXd reduced;
      if (phase1) {
        ++stats.pricing_matvecs;
        Eigen::VectorXd rhs(n);
        for (Index col = 0; col < n; ++col) {
          work.take(); long double sum = 0;
          for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
            work.take();
            const Index row = input.rows[k];
            if (state.position[row] >= 0) continue;
            const long double slack = state.slack[row];
            const int q = slack < 0 ? -1 : ((!input.inequality[row] && slack > 0) ? 1 : 0);
            sum += static_cast<long double>(input.values[k]) * q; finite(sum);
          }
          rhs[col] = work.narrow(sum);
        }
        work.check(); ++stats.dual_solves;
        try { reduced = lu->transpose().solve(rhs); } catch (const std::exception&) { throw Stop{"solve_failure"}; }
        work.check();
        if (lu->info() != Eigen::Success) throw Stop{"solve_failure"};
      } else {
        if (!state.dual_valid) {
          Eigen::VectorXd rhs(n);
          for (Index col = 0; col < n; ++col) { work.take(); rhs[col] = -input.costs[col]; }
          work.check(); ++stats.dual_solves;
          try { state.dual = lu->transpose().solve(rhs); } catch (const std::exception&) { throw Stop{"solve_failure"}; }
          work.check();
          if (lu->info() != Eigen::Success) throw Stop{"solve_failure"};
          state.dual_valid = true;
        }
        work.check(); reduced = state.dual; work.check();
      }
      Index entering = -1;
      for (Index k = 0; k < n; ++k) {
        work.take();
        if (!std::isfinite(reduced[k])) throw Stop{"solve_failure"};
        if (!input.inequality[state.selected[k]] || !(reduced[k] < 0)) continue;
        if (entering < 0 || reduced[k] < reduced[entering] ||
            (reduced[k] == reduced[entering] && state.selected[k] < state.selected[entering])) entering = k;
      }
      if (entering < 0) {
        if (phase1) throw Stop{"no_entering"};
        work.check();
        output.x.resize(static_cast<std::size_t>(n)); output.dual.resize(static_cast<std::size_t>(n));
        output.selected_rows.resize(static_cast<std::size_t>(n));
        for (Index k = 0; k < n; ++k) {
          work.take(); output.x[k] = state.x[k]; output.dual[k] = state.dual[k]; output.selected_rows[k] = state.selected[k];
        }
        work.check(); stats.phase = "complete"; output.status = "candidate"; output.candidate_available = true;
        return;
      }
      if (stats.pivots >= limits.max_pivots) throw Stop{"pivot_limit"};
      ++stats.iterations;
      if (phase1) ++stats.phase1_iterations; else ++stats.phase2_iterations;
      Eigen::VectorXd rhs(n), direction;
      for (Index k = 0; k < n; ++k) { work.take(); rhs[k] = k == entering ? -1 : 0; }
      work.check(); ++stats.direction_solves;
      try { direction = lu->solve(rhs); } catch (const std::exception&) { throw Stop{"solve_failure"}; }
      work.check();
      if (lu->info() != Eigen::Success) throw Stop{"solve_failure"};
      std::vector<long double> change(static_cast<std::size_t>(m)), change_scale(static_cast<std::size_t>(m));
      for (Index row = 0; row < m; ++row) { work.take(); change[row] = change_scale[row] = 0; }
      ++stats.direction_matvecs;
      for (Index col = 0; col < n; ++col) {
        work.take();
        if (!std::isfinite(direction[col])) throw Stop{"solve_failure"};
        for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
          work.take();
          const Index row = input.rows[k];
          const long double product = static_cast<long double>(input.values[k]) * direction[col];
          finite(product);
          if (input.values[k] != 0 && direction[col] != 0 && product == 0) throw Stop{"arithmetic_failure"};
          change[row] -= product; change_scale[row] += std::abs(product);
          finite(change[row]); finite(change_scale[row]);
        }
      }
      long double maximum_direction_residual = 0, maximum_change = 1;
      for (Index row = 0; row < m; ++row) {
        work.take();
        if (state.position[row] >= 0) {
          const long double expected = state.position[row] == entering ? 1 : 0;
          const long double residual = std::abs(change[row] - expected);
          maximum_direction_residual = std::max(maximum_direction_residual, residual);
          if (residual > epsilon_factor * (change_scale[row] + 1)) {
            stats.max_direction_residual = summary(maximum_direction_residual);
            throw Stop{"direction_residual_failure"};
          }
          // 选中行在基表示中保持结构性零，进入 slack 为 t，不能制造虚假等式断点。
          change[row] = expected;
        }
        maximum_change = std::max(maximum_change, std::abs(change[row]));
      }
      stats.max_direction_residual = summary(maximum_direction_residual);
      Event step;
      if (phase1) {
        const long double derivative = right_derivative(input, state, change, work);
        stats.last_right_derivative = summary(derivative);
        step = phase1_step(input, state, change, derivative, work);
      } else {
        stats.last_right_derivative = summary(static_cast<long double>(reduced[entering]));
        step = phase2_step(input, state, change, work);
      }
      stats.last_step = summary(step.theta);
      if (!(step.theta > 0)) { ++stats.zero_steps; throw Stop{"zero_step_stall"}; }
      // 仅用统一舍入尺度拒绝相对过弱的行替换主元，不修改原矩阵或原界。
      if (!(step.magnitude > epsilon_factor * maximum_change)) throw Stop{"weak_pivot"};
      State next;
      work.check(); next.selected.resize(static_cast<std::size_t>(n));
      for (Index k = 0; k < n; ++k) { work.take(); next.selected[k] = k == entering ? step.row : state.selected[k]; }
      positions(next, m, work);
      std::vector<Eigen::Triplet<double, Index>> entries;
      work.check(); entries.reserve(input.values.size());
      for (Index col = 0; col < n; ++col) {
        work.take();
        for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
          work.take();
          const Index row = next.position[input.rows[k]];
          if (row >= 0) entries.emplace_back(row, col, input.values[k]);
        }
      }
      work.check(); Matrix matrix(n, n);
      matrix.setFromTriplets(entries.begin(), entries.end()); work.check();
      matrix.makeCompressed(); work.check();
      if (stats.factor_nonzeros >= limits.max_factor_nonzeros) throw Stop{"factor_limit"};
      auto next_lu = std::make_unique<Lu>();
      work.check(); ++stats.lu_analyze_calls;
      try { next_lu->analyzePattern(matrix); } catch (const std::exception&) { throw Stop{"lu_failure"}; }
      work.check(); ++stats.lu_factor_calls;
      try { next_lu->factorize(matrix); } catch (const std::exception&) { throw Stop{"lu_failure"}; }
      work.check();
      if (!next_lu->factorization_ok() || next_lu->info() != Eigen::Success) throw Stop{"lu_failure"};
      const Index lower = static_cast<Index>(next_lu->nnzL()), upper = static_cast<Index>(next_lu->nnzU());
      if (lower < 0 || upper < 0 || lower > std::numeric_limits<Index>::max() - upper ||
          stats.factor_nonzeros > std::numeric_limits<Index>::max() - lower - upper) throw Stop{"factor_limit"};
      stats.factor_nonzeros += lower + upper;
      if (stats.factor_nonzeros > limits.max_factor_nonzeros) throw Stop{"factor_limit"};
      work.account(lower + upper);
      // CheckedLu 自己每至多 256 个元素调用此检查，不能再次稀释检查频率。
      if (!next_lu->finite_factors([&] { work.check(); })) throw Stop{"lu_nonfinite"};
      for (Index k = 0; k < n; ++k) { work.take(); rhs[k] = input.rhs[next.selected[k]]; }
      work.check(); ++stats.primal_solves;
      try { next.x = next_lu->solve(rhs); } catch (const std::exception&) { throw Stop{"solve_failure"}; }
      work.check();
      if (next_lu->info() != Eigen::Success) throw Stop{"solve_failure"};
      evaluate(input, limits, next, work, stats);
      // 重分解后核对真正到达的点，避免以线性预测代替已求得的新基本解。
      long double point_scale = 1, point_error = 0;
      for (Index col = 0; col < n; ++col) {
        work.take();
        const long double predicted = static_cast<long double>(state.x[col]) + step.theta * direction[col];
        finite(predicted);
        point_scale = std::max(point_scale, std::abs(predicted) + std::abs(static_cast<long double>(state.x[col])));
        point_error = std::max(point_error, std::abs(static_cast<long double>(next.x[col]) - predicted));
      }
      if (point_error > epsilon_factor * point_scale) throw Stop{"point_consistency_failure"};
      if (phase1) {
        if (!(next.phi < state.phi)) throw Stop{"objective_not_improved"};
      } else {
        if (!next.primal_feasible) throw Stop{"primal_guard_failure"};
        long double difference = 0;
        for (Index col = 0; col < n; ++col) {
          work.take(); difference += static_cast<long double>(input.costs[col]) *
              (static_cast<long double>(next.x[col]) - state.x[col]); finite(difference);
        }
        if (!(difference < 0)) throw Stop{"objective_not_improved"};
      }
      work.check();
      state = std::move(next); owned_lu = std::move(next_lu); lu = owned_lu.get();
      ++stats.pivots;
      record(state, stats);
      // 映射复制也计费；中断时向量可部分更新，但已提交 pivot 计数保持真实。
      stats.final_selected_rows.resize(static_cast<std::size_t>(n));
      for (Index k = 0; k < n; ++k) { work.take(); stats.final_selected_rows[k] = state.selected[k]; }
    }
  } catch (const Stop& stop) {
    if (output.status == "budget_exception") throw;
    output.status = stop.status;
    output.candidate_available = false;
    output.x.clear(); output.dual.clear(); output.selected_rows.clear();
  } catch (...) {
    if (output.status != "budget_exception") output.status = "exception";
    throw;
  }
}

inline void write_diagnostics(std::ostream& stream, const Result& result) {
  const auto& stats = result.stats;
  stream << std::setprecision(17) << "{\"status\":\"" << result.status
      << "\",\"candidate_available\":" << (result.candidate_available ? "true" : "false")
      << ",\"stats\":{\"phase\":\"" << stats.phase << '"';
  const auto integer = [&](const char* name, Index value) { stream << ",\"" << name << "\":" << value; };
  const auto number = [&](const char* name, const std::optional<double>& value) {
    stream << ",\"" << name << "\":";
    if (value) stream << *value; else stream << "null";
  };
  integer("attempts", stats.attempts); integer("iterations", stats.iterations);
  integer("phase1_iterations", stats.phase1_iterations); integer("phase2_iterations", stats.phase2_iterations);
  integer("pivots", stats.pivots); integer("zero_steps", stats.zero_steps);
  integer("work_units", stats.work_units); integer("budget_checks", stats.budget_checks);
  integer("lu_analyze_calls", stats.lu_analyze_calls); integer("lu_factor_calls", stats.lu_factor_calls);
  integer("factor_nonzeros", stats.factor_nonzeros); integer("primal_solves", stats.primal_solves);
  integer("dual_solves", stats.dual_solves); integer("direction_solves", stats.direction_solves);
  integer("slack_matvecs", stats.slack_matvecs); integer("direction_matvecs", stats.direction_matvecs);
  integer("pricing_matvecs", stats.pricing_matvecs); integer("roundoff_rows", stats.roundoff_rows);
  integer("final_selected_rows_count", static_cast<Index>(stats.final_selected_rows.size()));
  number("initial_phi", stats.initial_phi); number("final_phi", stats.final_phi);
  number("initial_max_violation", stats.initial_max_violation); number("final_max_violation", stats.final_max_violation);
  number("max_roundoff_bound", stats.max_roundoff_bound); number("last_right_derivative", stats.last_right_derivative);
  number("last_step", stats.last_step); number("max_selected_residual", stats.max_selected_residual);
  number("max_direction_residual", stats.max_direction_residual);
  stream << "}}";
}

}  // solver::v73_basis_repair
