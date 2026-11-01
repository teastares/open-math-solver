#pragma once

#include <Eigen/SparseLU>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

namespace solver::v81_basis_repair {

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
  Index max_pivots = 512;
  Index max_repair_work = 400000000;
  Index max_factor_nonzeros = 64000000;
  Index max_single_factor_nonzeros = 32000000;
  Index max_eta_updates = 8;
  double roundoff_multiplier = 64;
};

struct GuardStats {
  bool complete = false, passed = false;
  Index checked_rows = 0, violations = 0;
  std::optional<Index> failure_row, failure_basis_position;
  std::optional<double> failure_abs_residual, failure_threshold, failure_ratio;
  std::optional<double> max_ratio, max_abs_residual;
};

struct InitialRefinementStats {
  std::string status = "not_started";
  Index attempts = 0, solves = 0, accepted = 0;
  int double_digits = 0, long_double_digits = 0;
  std::optional<double> correction_max_abs;
};

struct ForwardStats {
  Index systems = 0, guard_failures = 0, attempts = 0, solves = 0, accepted = 0;
  std::string tag = "not_started";
  Index pivot_sequence = 0;
  std::optional<Index> entering_row, entering_basis_position;
  GuardStats before, after;
  InitialRefinementStats refinement;
};

struct Phase1TransitionStats {
  std::string status = "not_started";
  Index trials = 0, strict_accepted = 0, near_attempts = 0, near_accepted = 0, checked_rows = 0;
  std::optional<Index> blocking_row, released_row, max_slack_change_row;
  std::optional<double> old_phi, new_phi, delta_phi, blocking_slack, blocking_tau;
  std::optional<double> released_step, released_tau, max_slack_change_ratio;
  bool blocking_near = false, released_near = false, all_rows_near = false;
};

struct HistoryStats {
  std::string status = "not_started";
  bool initialized = false;
  Index states = 0, swaps = 0, candidate_checks = 0, hash_comparisons = 0;
  Index hash_matches = 0, exact_checks = 0, replay_swaps = 0, membership_comparisons = 0, repeats = 0;
  std::optional<Index> repeated_state;
};

struct SearchStats {
  std::string status = "not_started";
  Index bases = 0, candidates = 0, trials = 0, non_descent = 0, descent = 0, exhausted = 0;
  Index basis_pivots = 0, candidate_count = 0, zero_ineq_rows = 0, zero_eq_rows = 0;
  std::optional<Index> ordinal, entering_row, entering_basis_position;
  std::optional<double> priced_rc, linear_slope, zero_ineq, zero_eq, true_derivative, linear_minus_price;
  bool derivative_complete = false, step_started = false;
};

struct EtaStats {
  const char* status = "not_started";
  const char* last_mode = "none";
  const char* last_fallback = "none";
  Index prepares = 0, eta_trials = 0, full_refactor_trials = 0, eta_commits = 0, full_refactor_commits = 0;
  Index chain_length = 0, max_chain_length = 0, fallbacks = 0;
  Index forward_calls = 0, transpose_calls = 0, base_forward_solves = 0, base_transpose_solves = 0;
  Index forward_applications = 0, transpose_applications = 0, work_units = 0;
  Index stored_entries = 0, peak_stored_entries = 0;
  std::optional<double> last_alpha;
};

struct Stats {
  std::string phase = "not_started";
  Index attempts = 0, iterations = 0, phase1_iterations = 0, phase2_iterations = 0;
  Index pivots = 0, zero_steps = 0, work_units = 0, budget_checks = 0;
  Index lu_analyze_calls = 0, lu_factor_calls = 0, factor_nonzeros = 0, last_factor_nonzeros = 0;
  Index primal_solves = 0, dual_solves = 0, direction_solves = 0;
  Index slack_matvecs = 0, direction_matvecs = 0, pricing_matvecs = 0;
  Index roundoff_rows = 0;
  std::optional<double> initial_phi, final_phi;
  std::optional<double> initial_max_violation, final_max_violation;
  std::optional<double> max_roundoff_bound, last_right_derivative, last_step;
  std::optional<double> max_selected_residual, max_direction_residual;
  std::optional<double> last_point_error, last_point_bound;
  GuardStats initial_guard, refined_guard;
  InitialRefinementStats initial_refinement;
  ForwardStats direction_forward, rebuild_forward;
  Phase1TransitionStats phase1_transition;
  HistoryStats history;
  SearchStats search;
  EtaStats eta;
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
  bool budget_failed() const { return result_.status == "budget_exception"; }
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
  std::function<void()> budget_;
  Result& result_;
};

struct State {
  std::vector<Index> selected, position;
  Eigen::VectorXd x, dual;
  std::vector<long double> slack, scale, roundoff;
  long double phi = 0, max_violation = 0, max_bound = 0, max_selected = 0;
  Index roundoff_rows = 0;
  bool primal_feasible = false, dual_valid = false, basis_valid = false;
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
                     Work& work, Stats& stats, GuardStats* guard = nullptr) {
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
  state.basis_valid = true;
  if (guard) *guard = GuardStats{};
  long double maximum_ratio = 0;
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
      const long double residual = std::abs(state.slack[row]);
      const long double ratio = residual / state.roundoff[row];
      finite(ratio);
      state.max_selected = std::max(state.max_selected, residual);
      maximum_ratio = std::max(maximum_ratio, ratio);
      if (guard) {
        ++guard->checked_rows;
        guard->max_abs_residual = summary(state.max_selected);
        guard->max_ratio = summary(maximum_ratio);
      }
      if (residual > state.roundoff[row]) {
        state.basis_valid = false;
        if (guard) {
          ++guard->violations;
          if (!guard->failure_row) {
            guard->failure_row = row;
            guard->failure_basis_position = state.position[row];
            guard->failure_abs_residual = summary(residual);
            guard->failure_threshold = summary(state.roundoff[row]);
            guard->failure_ratio = summary(ratio);
          }
        }
      }
    }
  }
  stats.max_selected_residual = summary(state.max_selected);
  if (guard) { guard->complete = true; guard->passed = state.basis_valid; }
}

inline void record(const State& state, Stats& stats) {
  stats.final_phi = summary(state.phi);
  stats.final_max_violation = summary(state.max_violation);
  stats.max_roundoff_bound = summary(state.max_bound);
  stats.roundoff_rows = state.roundoff_rows;
  stats.max_selected_residual = summary(state.max_selected);
}

struct BasisSwap { Index removed, inserted; };

struct BasisHistory {
  std::vector<Index> initial_rows;
  std::vector<BasisSwap> swaps;
  std::vector<std::uint64_t> hashes;
  std::uint64_t current_hash = 0;
  std::unique_ptr<std::uint8_t[]> replay;

  static std::uint64_t row_hash(Index row) {
    std::uint64_t value = static_cast<std::uint64_t>(row) + UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }

  void initialize(const State& state, const Limits& limits, Work& work, HistoryStats& stats) {
    stats.status = "initializing";
    work.check();
    if (limits.max_pivots == std::numeric_limits<Index>::max()) throw Stop{"invalid_input"};
    initial_rows.resize(state.selected.size());
    // 小日志预留全部允许的换基容量，使真正提交阶段不再分配或检查预算。
    swaps.reserve(static_cast<std::size_t>(limits.max_pivots));
    hashes.reserve(static_cast<std::size_t>(limits.max_pivots) + 1);
    for (std::size_t k = 0; k < state.selected.size(); ++k) {
      work.take(); initial_rows[k] = state.selected[k]; current_hash ^= row_hash(state.selected[k]);
    }
    work.check();
    hashes.push_back(current_hash);
    stats.initialized = true; stats.states = 1; stats.status = "ready";
  }

  std::uint64_t candidate_hash(Index removed, Index inserted) const {
    return current_hash ^ row_hash(removed) ^ row_hash(inserted);
  }

  bool contains(const State& candidate, std::uint64_t hash, Work& work, HistoryStats& stats) {
    ++stats.candidate_checks; stats.status = "checking"; stats.repeated_state.reset();
    work.check();
    const Index m = static_cast<Index>(candidate.position.size());
    for (std::size_t version = 0; version < hashes.size(); ++version) {
      work.take(); ++stats.hash_comparisons;
      if (hashes[version] != hash) continue;
      ++stats.hash_matches; ++stats.exact_checks;
      if (!replay) { work.check(); replay.reset(new std::uint8_t[static_cast<std::size_t>(m)]); work.check(); }
      for (Index row = 0; row < m; ++row) { work.take(); replay[row] = 0; }
      for (const Index row : initial_rows) { work.take(); replay[row] = 1; }
      for (std::size_t step = 0; step < version; ++step) {
        work.take();
        const auto& change = swaps[step];
        if (!replay[change.removed] || replay[change.inserted]) throw Stop{"invalid_input"};
        replay[change.removed] = 0; replay[change.inserted] = 1; ++stats.replay_swaps;
      }
      bool same = true;
      for (Index row = 0; row < m; ++row) {
        work.take(); ++stats.membership_comparisons;
        if ((replay[row] != 0) != (candidate.position[row] >= 0)) same = false;
      }
      if (same) {
        ++stats.repeats; stats.repeated_state = static_cast<Index>(version); stats.status = "repeated";
        return true;
      }
    }
    stats.status = "unique";
    return false;
  }

  // 调用前完成预算与判重检查；这组常数操作与外层基状态一起提交。
  void commit(Index removed, Index inserted, std::uint64_t hash, HistoryStats& stats) {
    swaps.push_back({removed, inserted}); hashes.push_back(hash); current_hash = hash;
    ++stats.swaps; ++stats.states;
  }
};

inline bool phase1_transition(const State& state, const State& next, Index blocker, Index released,
                              long double theta, Work& work, Phase1TransitionStats& stats) {
  ++stats.trials;
  stats.checked_rows = 0; stats.max_slack_change_ratio.reset(); stats.max_slack_change_row.reset();
  stats.blocking_row = blocker; stats.released_row = released;
  stats.old_phi = summary(state.phi); stats.new_phi = summary(next.phi);
  stats.delta_phi = summary(next.phi - state.phi);
  stats.blocking_slack = summary(state.slack[blocker]); stats.blocking_tau = summary(state.roundoff[blocker]);
  stats.released_step = summary(theta); stats.released_tau = summary(state.roundoff[released]);
  stats.blocking_near = false; stats.released_near = false; stats.all_rows_near = false;
  if (next.phi < state.phi) { stats.status = "strict_candidate"; return false; }
  ++stats.near_attempts; stats.status = "near_check";
  stats.blocking_near = std::abs(state.slack[blocker]) <= state.roundoff[blocker];
  stats.released_near = theta <= state.roundoff[released];
  bool all_rows_near = true;
  long double maximum_ratio = 0;
  for (Index row = 0; row < static_cast<Index>(state.slack.size()); ++row) {
    work.take();
    const long double change = std::abs(next.slack[row] - state.slack[row]);
    const long double tolerance = state.roundoff[row] + next.roundoff[row];
    const long double ratio = change / tolerance;
    finite(change); finite(tolerance); finite(ratio);
    if (!stats.max_slack_change_row || ratio > maximum_ratio) {
      maximum_ratio = ratio; stats.max_slack_change_row = row;
    }
    ++stats.checked_rows; stats.max_slack_change_ratio = summary(maximum_ratio);
    if (change > tolerance) all_rows_near = false;
  }
  stats.all_rows_near = all_rows_near;
  if (!stats.blocking_near) { stats.status = "blocking_not_near"; throw Stop{"objective_not_improved"}; }
  if (!stats.released_near) { stats.status = "released_not_near"; throw Stop{"objective_not_improved"}; }
  if (!all_rows_near) { stats.status = "row_change_not_near"; throw Stop{"objective_not_improved"}; }
  stats.status = "near_candidate";
  return true;
}

template<class Lu>
class EtaFactor {
 public:
  struct Entry { std::vector<double> u, v; double alpha = 1; };

  struct Pending {
    EtaFactor* owner;
    std::optional<Entry> entry;
    std::unique_ptr<Lu> new_base;
    std::vector<Entry> retired_chain;
    bool full = false, consumed = false;
    explicit Pending(EtaFactor& factor) : owner(&factor) {}
    Pending(Pending&&) noexcept = default;
    Pending& operator=(Pending&&) noexcept = default;
    Pending(const Pending&) = delete;
    Pending& operator=(const Pending&) = delete;
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const { return owner->solve_view(rhs, this, false); }
    struct Transpose {
      const Pending* pending;
      Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const { return pending->owner->solve_view(rhs, pending, true); }
    };
    Transpose transpose() const { return {this}; }
    Eigen::ComputationInfo info() const { return consumed ? owner->info() : (full ? new_base->info() : owner->info()); }
  };

  EtaFactor(Lu& initial, const Limits& limits, Work& work, Stats& stats)
      : base_(&initial), limits_(limits), work_(work), stats_(stats), n_(static_cast<Index>(initial.rows())) {
    if (limits.max_eta_updates < 0 || limits.max_eta_updates > 8) throw Stop{"invalid_input"};
    work_.check(); chain_.reserve(static_cast<std::size_t>(limits.max_eta_updates)); work_.check();
    stats_.eta.status = "ready";
  }
  EtaFactor(const EtaFactor&) = delete;
  EtaFactor& operator=(const EtaFactor&) = delete;
  EtaFactor(EtaFactor&&) = delete;
  EtaFactor& operator=(EtaFactor&&) = delete;
  Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const { return solve_view(rhs, nullptr, false); }
  struct Transpose {
    const EtaFactor* factor;
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const { return factor->solve_view(rhs, nullptr, true); }
  };
  Transpose transpose() const { return {this}; }
  Eigen::ComputationInfo info() const { return base_->info(); }

  Pending prepare(const Input& input, const State& state, const State& next,
                  Index position, const Eigen::VectorXd& direction) {
    auto& eta = stats_.eta;
    ++eta.prepares; eta.status = "prepare"; eta.last_mode = "none";
    eta.last_fallback = "none"; eta.last_alpha.reset();
    work_.check();
    if (position < 0 || position >= n_ || direction.size() != n_ ||
        state.selected.size() != static_cast<std::size_t>(n_) || next.selected.size() != static_cast<std::size_t>(n_))
      throw Stop{"invalid_input"};
    Pending pending(*this);
    if (limits_.max_eta_updates == 0) {
      eta.last_fallback = "disabled"; prepare_full(pending, input, next); return pending;
    }
    if (static_cast<Index>(chain_.size()) >= limits_.max_eta_updates) {
      eta.last_fallback = "chain_full"; prepare_full(pending, input, next); return pending;
    }
    Entry entry;
    work_.check(); entry.u.resize(static_cast<std::size_t>(n_)); entry.v.resize(static_cast<std::size_t>(n_)); work_.check();
    const Index removed = state.selected[position], inserted = next.selected[position];
    const char* rejection = nullptr;
    for (Index col = 0; col < n_; ++col) {
      eta_take();
      entry.u[col] = -direction[col];
      if (!std::isfinite(entry.u[col])) { rejection = "nonfinite"; break; }
      long double old_value = 0, new_value = 0;
      for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
        eta_take();
        if (input.rows[k] == removed) old_value = input.values[k];
        if (input.rows[k] == inserted) new_value = input.values[k];
      }
      eta_take();
      const long double difference = new_value - old_value;
      const double converted = static_cast<double>(difference);
      if (!std::isfinite(difference)) { rejection = "nonfinite"; break; }
      if (!std::isfinite(converted) || (difference != 0 && converted == 0)) { rejection = "unrepresentable"; break; }
      entry.v[col] = converted;
    }
    long double product_sum = 0;
    if (!rejection) {
      for (Index col = 0; col < n_; ++col) {
        eta_take();
        const long double product = static_cast<long double>(entry.v[col]) * entry.u[col];
        if (!std::isfinite(product)) { rejection = "nonfinite"; break; }
        if (entry.v[col] != 0 && entry.u[col] != 0 && product == 0) { rejection = "unrepresentable"; break; }
        product_sum += product;
        if (!std::isfinite(product_sum)) { rejection = "nonfinite"; break; }
      }
    }
    if (!rejection) {
      const long double alpha = 1 + product_sum;
      eta.last_alpha = summary(alpha);
      const double converted = static_cast<double>(alpha);
      if (!std::isfinite(alpha)) rejection = "nonfinite";
      else if (alpha == 0) rejection = "alpha_zero";
      else if (!std::isfinite(converted) || converted == 0) rejection = "unrepresentable";
      else entry.alpha = converted;
    }
    if (rejection) {
      ++eta.fallbacks; eta.last_fallback = rejection;
      prepare_full(pending, input, next); return pending;
    }
    pending.entry = std::move(entry);
    ++eta.eta_trials; eta.last_mode = "eta"; eta.status = "eta_trial";
    return pending;
  }

  // 调用者已完成全部预算和原模型数值守卫；这里只移动预留槽和所有权。
  void commit(Pending& pending) noexcept {
    auto& eta = stats_.eta;
    if (pending.full) {
      owned_base_.swap(pending.new_base);
      base_ = owned_base_.get();
      chain_.swap(pending.retired_chain);
      ++eta.full_refactor_commits; eta.status = "commit_full";
    } else {
      chain_.push_back(std::move(*pending.entry));
      ++eta.eta_commits; eta.status = "commit_eta";
    }
    pending.consumed = true;
    eta.chain_length = static_cast<Index>(chain_.size());
    eta.max_chain_length = std::max(eta.max_chain_length, eta.chain_length);
    eta.stored_entries = 2 * n_ * eta.chain_length;
    eta.peak_stored_entries = std::max(eta.peak_stored_entries, eta.stored_entries);
  }

 private:
  void eta_take() const { work_.take(); ++stats_.eta.work_units; }
  [[noreturn]] void application_failure() const { stats_.eta.status = "apply_failure"; throw Stop{"arithmetic_failure"}; }
  double narrow_application(long double value) const {
    const double converted = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0 && converted == 0)) application_failure();
    return converted;
  }
  void apply(Eigen::VectorXd& vector, const Entry& entry, bool transposed) const {
    if (transposed) ++stats_.eta.transpose_applications; else ++stats_.eta.forward_applications;
    work_.check();
    long double dot = 0;
    for (Index col = 0; col < n_; ++col) {
      eta_take();
      const double coefficient = transposed ? entry.u[col] : entry.v[col];
      const long double product = static_cast<long double>(coefficient) * vector[col];
      if (!std::isfinite(product) || (coefficient != 0 && vector[col] != 0 && product == 0)) application_failure();
      dot += product;
      if (!std::isfinite(dot)) application_failure();
    }
    const long double multiplier = dot / entry.alpha;
    if (!std::isfinite(multiplier)) application_failure();
    for (Index col = 0; col < n_; ++col) {
      eta_take();
      const double coefficient = transposed ? entry.v[col] : entry.u[col];
      const long double product = static_cast<long double>(coefficient) * multiplier;
      if (!std::isfinite(product) || (coefficient != 0 && multiplier != 0 && product == 0)) application_failure();
      vector[col] = narrow_application(static_cast<long double>(vector[col]) - product);
    }
    work_.check();
  }
  Eigen::VectorXd solve_view(const Eigen::VectorXd& rhs, const Pending* pending, bool transposed) const {
    if (pending && pending->consumed) throw Stop{"invalid_input"};
    auto& eta = stats_.eta;
    if (transposed) { ++eta.transpose_calls; eta.status = "transpose"; }
    else { ++eta.forward_calls; eta.status = "forward"; }
    work_.check();
    Lu* base = pending && pending->full ? pending->new_base.get() : base_;
    if (!transposed) {
      ++eta.base_forward_solves;
      Eigen::VectorXd result = base->solve(rhs);
      work_.check();
      if (base->info() != Eigen::Success) return result;
      if (!(pending && pending->full)) {
        for (const auto& entry : chain_) apply(result, entry, false);
        if (pending && pending->entry) apply(result, *pending->entry, false);
      }
      return result;
    }
    if ((pending && pending->full) || (chain_.empty() && !(pending && pending->entry))) {
      ++eta.base_transpose_solves;
      Eigen::VectorXd result = base->transpose().solve(rhs); work_.check(); return result;
    }
    work_.check(); Eigen::VectorXd transformed(n_);
    for (Index col = 0; col < n_; ++col) { eta_take(); transformed[col] = rhs[col]; }
    if (pending && pending->entry) apply(transformed, *pending->entry, true);
    for (std::size_t k = chain_.size(); k > 0; --k) apply(transformed, chain_[k - 1], true);
    work_.check(); ++eta.base_transpose_solves;
    Eigen::VectorXd result = base->transpose().solve(transformed); work_.check(); return result;
  }
  void prepare_full(Pending& pending, const Input& input, const State& next) {
    auto& eta = stats_.eta;
    pending.full = true; ++eta.full_refactor_trials; eta.last_mode = "full"; eta.status = "full_trial";
    work_.check(); pending.retired_chain.reserve(static_cast<std::size_t>(limits_.max_eta_updates));
    std::vector<Eigen::Triplet<double, Index>> entries;
    work_.check(); entries.reserve(input.values.size());
    for (Index col = 0; col < n_; ++col) {
      work_.take();
      for (Index k = input.starts[col]; k < input.starts[col + 1]; ++k) {
        work_.take();
        const Index row = next.position[input.rows[k]];
        if (row >= 0) entries.emplace_back(row, col, input.values[k]);
      }
    }
    work_.check(); Matrix matrix(n_, n_);
    matrix.setFromTriplets(entries.begin(), entries.end()); work_.check(); matrix.makeCompressed(); work_.check();
    if (stats_.factor_nonzeros >= limits_.max_factor_nonzeros) throw Stop{"factor_limit"};
    pending.new_base = std::make_unique<Lu>();
    work_.check(); ++stats_.lu_analyze_calls;
    try { pending.new_base->analyzePattern(matrix); }
    catch (const std::exception&) { if (work_.budget_failed()) throw; throw Stop{"lu_failure"}; }
    work_.check(); ++stats_.lu_factor_calls;
    try { pending.new_base->factorize(matrix); }
    catch (const std::exception&) { if (work_.budget_failed()) throw; throw Stop{"lu_failure"}; }
    work_.check();
    if (!pending.new_base->factorization_ok() || pending.new_base->info() != Eigen::Success) throw Stop{"lu_failure"};
    const Index lower = static_cast<Index>(pending.new_base->nnzL()), upper = static_cast<Index>(pending.new_base->nnzU());
    if (lower < 0 || upper < 0 || lower > std::numeric_limits<Index>::max() - upper) throw Stop{"factor_limit"};
    stats_.last_factor_nonzeros = lower + upper;
    if (stats_.factor_nonzeros > std::numeric_limits<Index>::max() - stats_.last_factor_nonzeros) throw Stop{"factor_limit"};
    stats_.factor_nonzeros += stats_.last_factor_nonzeros;
    if (stats_.last_factor_nonzeros > limits_.max_single_factor_nonzeros) throw Stop{"single_factor_limit"};
    if (stats_.factor_nonzeros > limits_.max_factor_nonzeros) throw Stop{"factor_limit"};
    work_.account(lower + upper);
    if (!pending.new_base->finite_factors([&] { work_.check(); })) throw Stop{"lu_nonfinite"};
  }
  Lu* base_;
  std::unique_ptr<Lu> owned_base_;
  std::vector<Entry> chain_;
  const Limits& limits_;
  Work& work_;
  Stats& stats_;
  Index n_;
};

inline void begin_forward(ForwardStats& forward, const char* tag, Index pivot_sequence,
                          Index entering_row, Index entering_position) {
  ++forward.systems;
  forward.tag = tag;
  forward.pivot_sequence = pivot_sequence;
  forward.entering_row = entering_row;
  forward.entering_basis_position = entering_position;
  forward.before = GuardStats{}; forward.after = GuardStats{};
  forward.refinement = InitialRefinementStats{};
}

// 统一复用原因子修正一次；右端由各调用者的原始残差给出，守卫公式由调用者保留。
template<class Lu, class ResidualRhs>
inline Eigen::VectorXd correct_forward(const Eigen::VectorXd& point, Lu& lu, Work& work,
                                      InitialRefinementStats& refinement, ForwardStats* forward,
                                      ResidualRhs residual_rhs, const char* failure_status) {
  refinement.attempts = 1;
  if (forward) ++forward->attempts;
  refinement.double_digits = std::numeric_limits<double>::digits;
  refinement.long_double_digits = std::numeric_limits<long double>::digits;
  if (refinement.long_double_digits <= refinement.double_digits) {
    refinement.status = "unsupported_precision";
    throw Stop{failure_status};
  }
  const Index n = static_cast<Index>(point.size());
  refinement.status = "preparing_rhs";
  Eigen::VectorXd rhs(n), correction;
  for (Index k = 0; k < n; ++k) {
    work.take();
    const long double value = residual_rhs(k);
    const double converted = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0 && converted == 0)) {
      refinement.status = "rhs_representation_failure";
      throw Stop{failure_status};
    }
    rhs[k] = converted;
  }
  work.check(); refinement.status = "solving"; ++refinement.solves;
  if (forward) ++forward->solves;
  try { correction = lu.solve(rhs); }
  catch (const std::exception&) {
    if (work.budget_failed()) throw;
    refinement.status = "solve_failure";
    throw Stop{failure_status};
  }
  work.check();
  if (lu.info() != Eigen::Success || correction.size() != n) {
    refinement.status = "solve_failure";
    throw Stop{failure_status};
  }
  refinement.status = "updating";
  work.check(); Eigen::VectorXd trial(n);
  long double maximum_correction = 0;
  for (Index k = 0; k < n; ++k) {
    work.take();
    if (!std::isfinite(correction[k])) {
      refinement.status = "correction_nonfinite";
      throw Stop{failure_status};
    }
    const long double value = static_cast<long double>(point[k]) + correction[k];
    const double converted = static_cast<double>(value);
    if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0 && converted == 0)) {
      refinement.status = "correction_nonfinite";
      throw Stop{failure_status};
    }
    trial[k] = converted;
    maximum_correction = std::max(maximum_correction, std::abs(static_cast<long double>(correction[k])));
  }
  refinement.correction_max_abs = summary(maximum_correction);
  return trial;
}

template<class Lu>
inline void refine_point(const Input& input, const Limits& limits, State& state,
                         Lu& lu, Work& work, Stats& stats, GuardStats& after,
                         InitialRefinementStats& refinement, ForwardStats* forward,
                         bool record_point) {
  State next;
  next.x = correct_forward(state.x, lu, work, refinement, forward,
      [&](Index k) { return state.slack[state.selected[k]]; }, "basis_residual_failure");
  const Index n = static_cast<Index>(input.costs.size()), m = static_cast<Index>(input.rhs.size());
  work.check(); next.selected.resize(static_cast<std::size_t>(n));
  for (Index k = 0; k < n; ++k) { work.take(); next.selected[k] = state.selected[k]; }
  // 新分解后的原始点尚未算对偶，保留原dual长度与有效性，不能假设它已有n项。
  next.dual.resize(state.dual.size());
  for (Eigen::Index k = 0; k < state.dual.size(); ++k) { work.take(); next.dual[k] = state.dual[k]; }
  next.dual_valid = state.dual_valid;
  positions(next, m, work);
  refinement.status = "evaluating";
  evaluate(input, limits, next, work, stats, &after);
  if (!next.basis_valid) {
    if (record_point) record(state, stats);
    refinement.status = "guard_rejected";
    throw Stop{"basis_residual_failure"};
  }
  // 临时点必须通过同一个完整逐行守卫，最大绝对残差下降不能代替它。
  work.check();
  state = std::move(next);
  refinement.accepted = 1;
  if (forward) ++forward->accepted;
  refinement.status = "accepted";
  if (record_point) record(state, stats);
}

struct Event { long double theta, jump, magnitude; Index row; };

struct DirectionState {
  std::vector<long double> change, scale;
  long double maximum_change = 1;
  bool valid = false;
};

inline void evaluate_direction(const Input& input, const Limits& limits, const State& state,
                               const Eigen::VectorXd& direction, Index entering, Work& work,
                               Stats& stats, DirectionState& output, GuardStats& guard) {
  const Index n = static_cast<Index>(input.costs.size()), m = static_cast<Index>(input.rhs.size());
  guard = GuardStats{};
  output.valid = false; output.maximum_change = 1;
  work.check(); output.change.resize(static_cast<std::size_t>(m)); output.scale.resize(static_cast<std::size_t>(m));
  for (Index row = 0; row < m; ++row) { work.take(); output.change[row] = output.scale[row] = 0; }
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
      output.change[row] -= product; output.scale[row] += std::abs(product);
      finite(output.change[row]); finite(output.scale[row]);
    }
  }
  const long double factor = static_cast<long double>(limits.roundoff_multiplier) *
      std::numeric_limits<double>::epsilon();
  long double maximum_residual = 0, maximum_ratio = 0;
  bool passed = true;
  for (Index row = 0; row < m; ++row) {
    work.take();
    if (state.position[row] >= 0) {
      const long double expected = state.position[row] == entering ? 1 : 0;
      const long double residual = std::abs(output.change[row] - expected);
      // 方向保留原有公式，单位右端不额外添加到阈值中。
      const long double threshold = factor * (output.scale[row] + 1);
      const long double ratio = residual / threshold;
      finite(threshold); finite(ratio);
      maximum_residual = std::max(maximum_residual, residual);
      maximum_ratio = std::max(maximum_ratio, ratio);
      ++guard.checked_rows;
      guard.max_abs_residual = summary(maximum_residual); guard.max_ratio = summary(maximum_ratio);
      if (residual > threshold) {
        passed = false; ++guard.violations;
        if (!guard.failure_row) {
          guard.failure_row = row; guard.failure_basis_position = state.position[row];
          guard.failure_abs_residual = summary(residual); guard.failure_threshold = summary(threshold);
          guard.failure_ratio = summary(ratio);
        }
      }
    } else output.maximum_change = std::max(output.maximum_change, std::abs(output.change[row]));
  }
  stats.max_direction_residual = summary(maximum_residual);
  guard.complete = true; guard.passed = passed; output.valid = passed;
}

inline void clear_search_trial(SearchStats& search) {
  search.ordinal.reset(); search.entering_row.reset(); search.entering_basis_position.reset();
  search.priced_rc.reset(); search.linear_slope.reset(); search.zero_ineq.reset(); search.zero_eq.reset();
  search.true_derivative.reset(); search.linear_minus_price.reset();
  search.zero_ineq_rows = 0; search.zero_eq_rows = 0;
  search.derivative_complete = false; search.step_started = false;
}

inline void begin_search(SearchStats& search, Index basis_pivots) {
  ++search.bases; search.status = "preparing"; search.basis_pivots = basis_pivots;
  search.candidate_count = 0; clear_search_trial(search);
}

inline void begin_search_trial(SearchStats& search, Index ordinal, Index row, Index position, double price) {
  ++search.trials; clear_search_trial(search);
  search.status = "direction"; search.ordinal = ordinal; search.entering_row = row;
  search.entering_basis_position = position; search.priced_rc = price;
}

// 零处按真实右导数计价；分项只作诊断，不改变原有逐行累加顺序或方向判断。
inline long double right_derivative(const Input& input, const State& state,
                                   const std::vector<long double>& direction, Work& work,
                                   SearchStats* search = nullptr, long double price = 0) {
  long double derivative = 0;
  long double linear = 0, zero_ineq = 0, zero_eq = 0;
  for (Index row = 0; row < static_cast<Index>(input.rhs.size()); ++row) {
    work.take();
    if (state.position[row] >= 0) continue;
    if (search) work.take();
    const long double slack = state.slack[row], change = direction[row];
    if (!input.inequality[row]) {
      derivative += slack < 0 ? -change : (slack > 0 ? change : std::abs(change));
      if (search) {
        if (slack == 0) { zero_eq += std::abs(change); ++search->zero_eq_rows; }
        else linear += slack < 0 ? -change : change;
      }
    } else if (slack < 0) {
      derivative -= change;
      if (search) linear -= change;
    } else if (slack == 0) {
      derivative += std::max(-change, 0.0L);
      if (search) { zero_ineq += std::max(-change, 0.0L); ++search->zero_ineq_rows; }
    }
    finite(derivative);
  }
  if (search) {
    search->linear_slope = summary(linear); search->zero_ineq = summary(zero_ineq); search->zero_eq = summary(zero_eq);
    search->true_derivative = summary(derivative); search->linear_minus_price = summary(linear - price);
    search->derivative_complete = true;
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
  InitialRefinementStats* active_refinement = nullptr;
  stats.attempts = 1;
  stats.phase = "validation";
  try {
    work.check();
    const auto largest = static_cast<std::size_t>(std::min<Index>(
        std::numeric_limits<Index>::max(), std::numeric_limits<Eigen::Index>::max()));
    if (input.costs.size() >= largest || input.rhs.size() >= largest || input.values.size() >= largest ||
        limits.max_pivots < 0 || limits.max_repair_work < 0 || limits.max_factor_nonzeros < 0 ||
        limits.max_single_factor_nonzeros < 0 ||
        limits.max_eta_updates < 0 || limits.max_eta_updates > 8 ||
        !(limits.roundoff_multiplier > 0) || !std::isfinite(limits.roundoff_multiplier)) throw Stop{"invalid_input"};
    const Index n = static_cast<Index>(input.costs.size()), m = static_cast<Index>(input.rhs.size());
    if (n == 0 || m < n || input.starts.size() != input.costs.size() + 1 ||
        input.starts.front() != 0 || input.starts.back() != static_cast<Index>(input.values.size()) ||
        input.rows.size() != input.values.size() || input.inequality.size() != input.rhs.size() ||
        start.selected_rows.size() != input.costs.size() || start.x.size() != input.costs.size() ||
        start.dual.size() != input.costs.size() || start.factor_nonzeros < 0 ||
        initial_lu.rows() != n || initial_lu.cols() != n || !initial_lu.factorization_ok()) throw Stop{"invalid_input"};
    stats.factor_nonzeros = start.factor_nonzeros;
    stats.last_factor_nonzeros = start.factor_nonzeros;
    if (stats.last_factor_nonzeros > limits.max_single_factor_nonzeros) throw Stop{"single_factor_limit"};
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
    EtaFactor<Lu> factors(initial_lu, limits, work, stats);
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
    evaluate(input, limits, state, work, stats, &stats.initial_guard);
    stats.initial_phi = summary(state.phi); stats.initial_max_violation = summary(state.max_violation);
    record(state, stats);
    if (!state.basis_valid) {
      stats.phase = "initial_refinement";
      active_refinement = &stats.initial_refinement;
      refine_point(input, limits, state, factors, work, stats, stats.refined_guard,
          stats.initial_refinement, nullptr, true);
      active_refinement = nullptr;
    } else stats.initial_refinement.status = "not_needed";
    stats.final_selected_rows.resize(static_cast<std::size_t>(n));
    for (Index k = 0; k < n; ++k) { work.take(); stats.final_selected_rows[k] = state.selected[k]; }
    BasisHistory basis_history;
    basis_history.initialize(state, limits, work, stats.history);
    const long double epsilon_factor = static_cast<long double>(limits.roundoff_multiplier) *
        std::numeric_limits<double>::epsilon();
    for (;;) {
      work.check();
      const bool phase1 = !state.primal_feasible;
      stats.phase = phase1 ? "phase1" : "phase2";
      Eigen::VectorXd reduced;
      if (phase1) {
        begin_search(stats.search, stats.pivots);
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
        try { reduced = factors.transpose().solve(rhs); }
        catch (const std::exception&) { if (work.budget_failed()) throw; throw Stop{"solve_failure"}; }
        work.check();
        if (factors.info() != Eigen::Success) throw Stop{"solve_failure"};
      } else {
        if (!state.dual_valid) {
          Eigen::VectorXd rhs(n);
          for (Index col = 0; col < n; ++col) { work.take(); rhs[col] = -input.costs[col]; }
          work.check(); ++stats.dual_solves;
          try { state.dual = factors.transpose().solve(rhs); }
          catch (const std::exception&) { if (work.budget_failed()) throw; throw Stop{"solve_failure"}; }
          work.check();
          if (factors.info() != Eigen::Success) throw Stop{"solve_failure"};
          state.dual_valid = true;
        }
        work.check(); reduced = state.dual; work.check();
      }
      Index entering = -1;
      std::vector<Index> candidates;
      if (phase1) { work.check(); candidates.reserve(static_cast<std::size_t>(n)); work.check(); }
      for (Index k = 0; k < n; ++k) {
        work.take();
        if (!std::isfinite(reduced[k])) throw Stop{"solve_failure"};
        if (!input.inequality[state.selected[k]] || !(reduced[k] < 0)) continue;
        if (phase1) {
          work.take();
          candidates.push_back(k); ++stats.search.candidates; ++stats.search.candidate_count;
          continue;
        }
        if (entering < 0 || reduced[k] < reduced[entering] ||
            (reduced[k] == reduced[entering] && state.selected[k] < state.selected[entering])) entering = k;
      }
      if (phase1) {
        stats.search.status = "sorting";
        std::sort(candidates.begin(), candidates.end(), [&](Index first, Index second) {
          work.take();
          return reduced[first] != reduced[second] ? reduced[first] < reduced[second]
              : state.selected[first] < state.selected[second];
        });
        stats.search.status = "searching";
        if (!candidates.empty()) entering = candidates.front();
      }
      if (entering < 0) {
        if (phase1) { ++stats.search.exhausted; stats.search.status = "exhausted"; throw Stop{"no_entering"}; }
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
      DirectionState direction_state;
      long double derivative = 0;
      bool direction_found = false;
      const Index trial_count = phase1 ? static_cast<Index>(candidates.size()) : 1;
      for (Index ordinal = 0; ordinal < trial_count; ++ordinal) {
        if (phase1) {
          entering = candidates[ordinal];
          begin_search_trial(stats.search, ordinal, state.selected[entering], entering, reduced[entering]);
          stats.last_right_derivative.reset(); stats.last_step.reset();
        }
        for (Index k = 0; k < n; ++k) { work.take(); rhs[k] = k == entering ? -1 : 0; }
        work.check(); ++stats.direction_solves;
        try { direction = factors.solve(rhs); }
        catch (const std::exception&) { if (work.budget_failed()) throw; throw Stop{"solve_failure"}; }
        work.check();
        if (factors.info() != Eigen::Success) throw Stop{"solve_failure"};
        auto& forward = stats.direction_forward;
        begin_forward(forward, "direction", stats.pivots, state.selected[entering], entering);
        evaluate_direction(input, limits, state, direction, entering, work, stats, direction_state, forward.before);
        if (!direction_state.valid) {
          ++forward.guard_failures;
          active_refinement = &forward.refinement;
          Eigen::VectorXd trial = correct_forward(direction, factors, work, forward.refinement, &forward,
              [&](Index k) {
                const long double expected = k == entering ? 1 : 0;
                return direction_state.change[state.selected[k]] - expected;
              }, "direction_residual_failure");
          DirectionState trial_state;
          forward.refinement.status = "evaluating";
          evaluate_direction(input, limits, state, trial, entering, work, stats, trial_state, forward.after);
          if (!trial_state.valid) {
            forward.refinement.status = "guard_rejected";
            throw Stop{"direction_residual_failure"};
          }
          work.check();
          direction = std::move(trial); direction_state = std::move(trial_state);
          forward.refinement.accepted = 1; ++forward.accepted; forward.refinement.status = "accepted";
          active_refinement = nullptr;
        } else forward.refinement.status = "not_needed";
        for (Index row = 0; row < m; ++row) {
          work.take();
          if (state.position[row] >= 0) {
            const long double expected = state.position[row] == entering ? 1 : 0;
            // 每个候选都完整验算原方向，随后才能覆盖结构性单位方向。
            direction_state.change[row] = expected;
          }
        }
        if (phase1) {
          stats.search.status = "evaluating";
          derivative = right_derivative(input, state, direction_state.change, work, &stats.search, reduced[entering]);
          stats.last_right_derivative = summary(derivative);
          if (!(derivative < 0)) { ++stats.search.non_descent; stats.search.status = "non_descent"; continue; }
          ++stats.search.descent; stats.search.status = "descent";
        }
        direction_found = true;
        break;
      }
      if (!direction_found) {
        ++stats.search.exhausted; stats.search.status = "exhausted";
        throw Stop{"direction_search_exhausted"};
      }
      auto& change = direction_state.change;
      const long double maximum_change = direction_state.maximum_change;
      Event step;
      if (phase1) {
        stats.search.step_started = true; stats.search.status = "step_started";
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
      auto pending = factors.prepare(input, state, next, entering, direction);
      for (Index k = 0; k < n; ++k) { work.take(); rhs[k] = input.rhs[next.selected[k]]; }
      work.check(); ++stats.primal_solves;
      try { next.x = pending.solve(rhs); }
      catch (const std::exception&) { if (work.budget_failed()) throw; throw Stop{"solve_failure"}; }
      work.check();
      if (pending.info() != Eigen::Success) throw Stop{"solve_failure"};
      auto& rebuild = stats.rebuild_forward;
      begin_forward(rebuild, "rebuild", stats.pivots + 1, state.selected[entering], entering);
      try {
        evaluate(input, limits, next, work, stats, &rebuild.before);
        if (!next.basis_valid) {
          ++rebuild.guard_failures;
          active_refinement = &rebuild.refinement;
          refine_point(input, limits, next, pending, work, stats, rebuild.after,
              rebuild.refinement, &rebuild, false);
          active_refinement = nullptr;
        } else rebuild.refinement.status = "not_needed";
      } catch (...) {
        record(state, stats);
        throw;
      }
      // 重建点尚未通过预测及目标检查，平面摘要继续表示已保留的状态。
      record(state, stats);
      // 重分解后核对真正到达的点，避免以线性预测代替已求得的新基本解。
      long double point_scale = 1, point_error = 0;
      for (Index col = 0; col < n; ++col) {
        work.take();
        const long double predicted = static_cast<long double>(state.x[col]) + step.theta * direction[col];
        finite(predicted);
        point_scale = std::max(point_scale, std::abs(predicted) + std::abs(static_cast<long double>(state.x[col])));
        point_error = std::max(point_error, std::abs(static_cast<long double>(next.x[col]) - predicted));
      }
      const long double point_bound = epsilon_factor * point_scale;
      stats.last_point_error = summary(point_error); stats.last_point_bound = summary(point_bound);
      if (point_error > point_bound) throw Stop{"point_consistency_failure"};
      bool near_transition = false;
      if (phase1) {
        near_transition = phase1_transition(state, next, step.row, state.selected[entering],
            step.theta, work, stats.phase1_transition);
      } else {
        if (!next.primal_feasible) throw Stop{"primal_guard_failure"};
        long double difference = 0;
        for (Index col = 0; col < n; ++col) {
          work.take(); difference += static_cast<long double>(input.costs[col]) *
              (static_cast<long double>(next.x[col]) - state.x[col]); finite(difference);
        }
        if (!(difference < 0)) throw Stop{"objective_not_improved"};
      }
      const Index released_row = state.selected[entering];
      const std::uint64_t next_hash = basis_history.candidate_hash(released_row, step.row);
      if (basis_history.contains(next, next_hash, work, stats.history)) {
        if (phase1) stats.phase1_transition.status = "basis_repeated";
        throw Stop{"basis_repeated"};
      }
      work.check();
      basis_history.commit(released_row, step.row, next_hash, stats.history);
      factors.commit(pending);
      state = std::move(next);
      ++stats.pivots;
      if (phase1) {
        if (near_transition) { ++stats.phase1_transition.near_accepted; stats.phase1_transition.status = "near_accepted"; }
        else { ++stats.phase1_transition.strict_accepted; stats.phase1_transition.status = "strict_accepted"; }
        stats.search.status = "pivot_committed";
      }
      record(state, stats);
      // 映射复制也计费；中断时向量可部分更新，但已提交 pivot 计数保持真实。
      stats.final_selected_rows.resize(static_cast<std::size_t>(n));
      for (Index k = 0; k < n; ++k) { work.take(); stats.final_selected_rows[k] = state.selected[k]; }
    }
  } catch (const Stop& stop) {
    if (output.status == "budget_exception") {
      if (active_refinement && active_refinement->accepted == 0) active_refinement->status = "budget_exception";
      throw;
    }
    output.status = stop.status;
    output.candidate_available = false;
    output.x.clear(); output.dual.clear(); output.selected_rows.clear();
  } catch (...) {
    if (active_refinement && active_refinement->accepted == 0) {
      active_refinement->status = output.status == "budget_exception" ? "budget_exception" : "exception";
    }
    if (output.status != "budget_exception") output.status = "exception";
    throw;
  }
}

inline void write_guard_diagnostics(std::ostream& stream, const GuardStats& guard) {
  stream << std::setprecision(17) << "{\"complete\":" << (guard.complete ? "true" : "false")
      << ",\"passed\":" << (guard.passed ? "true" : "false")
      << ",\"checked_rows\":" << guard.checked_rows << ",\"violations\":" << guard.violations;
  const auto value = [&](const char* name, const auto& item) {
    stream << ",\"" << name << "\":";
    if (item) stream << *item; else stream << "null";
  };
  value("failure_row", guard.failure_row); value("failure_basis_position", guard.failure_basis_position);
  value("failure_abs_residual", guard.failure_abs_residual); value("failure_threshold", guard.failure_threshold);
  value("failure_ratio", guard.failure_ratio); value("max_ratio", guard.max_ratio);
  value("max_abs_residual", guard.max_abs_residual);
  stream << '}';
}

inline void write_guard_diagnostics(std::FILE* stream, const GuardStats& guard) {
  std::fprintf(stream, "{\"complete\":%s,\"passed\":%s,\"checked_rows\":%lld,\"violations\":%lld",
      guard.complete ? "true" : "false", guard.passed ? "true" : "false",
      static_cast<long long>(guard.checked_rows), static_cast<long long>(guard.violations));
  const auto index = [&](const char* name, const std::optional<Index>& item) {
    std::fprintf(stream, ",\"%s\":", name);
    if (item) std::fprintf(stream, "%lld", static_cast<long long>(*item)); else std::fputs("null", stream);
  };
  const auto number = [&](const char* name, const std::optional<double>& item) {
    std::fprintf(stream, ",\"%s\":", name);
    if (item) std::fprintf(stream, "%.17g", *item); else std::fputs("null", stream);
  };
  index("failure_row", guard.failure_row); index("failure_basis_position", guard.failure_basis_position);
  number("failure_abs_residual", guard.failure_abs_residual); number("failure_threshold", guard.failure_threshold);
  number("failure_ratio", guard.failure_ratio); number("max_ratio", guard.max_ratio);
  number("max_abs_residual", guard.max_abs_residual);
  std::fputc('}', stream);
}

inline void write_refinement_diagnostics(std::ostream& stream, const InitialRefinementStats& refinement) {
  stream << std::setprecision(17) << "{\"status\":\"" << refinement.status << "\",\"attempts\":"
      << refinement.attempts << ",\"solves\":" << refinement.solves << ",\"accepted\":" << refinement.accepted
      << ",\"double_digits\":" << refinement.double_digits << ",\"long_double_digits\":"
      << refinement.long_double_digits << ",\"correction_max_abs\":";
  if (refinement.correction_max_abs) stream << *refinement.correction_max_abs; else stream << "null";
  stream << '}';
}

inline void write_refinement_diagnostics(std::FILE* stream, const InitialRefinementStats& refinement) {
  std::fprintf(stream, "{\"status\":\"%s\",\"attempts\":%lld,\"solves\":%lld,\"accepted\":%lld,"
      "\"double_digits\":%d,\"long_double_digits\":%d,\"correction_max_abs\":",
      refinement.status.c_str(), static_cast<long long>(refinement.attempts),
      static_cast<long long>(refinement.solves), static_cast<long long>(refinement.accepted),
      refinement.double_digits, refinement.long_double_digits);
  if (refinement.correction_max_abs) std::fprintf(stream, "%.17g", *refinement.correction_max_abs);
  else std::fputs("null", stream);
  std::fputc('}', stream);
}

inline void write_forward_diagnostics(std::ostream& stream, const ForwardStats& forward) {
  stream << "{\"systems\":" << forward.systems << ",\"guard_failures\":" << forward.guard_failures
      << ",\"attempts\":" << forward.attempts << ",\"solves\":" << forward.solves
      << ",\"accepted\":" << forward.accepted << ",\"tag\":\"" << forward.tag
      << "\",\"pivot_sequence\":" << forward.pivot_sequence << ",\"entering_row\":";
  if (forward.entering_row) stream << *forward.entering_row; else stream << "null";
  stream << ",\"entering_basis_position\":";
  if (forward.entering_basis_position) stream << *forward.entering_basis_position; else stream << "null";
  stream << ",\"before\":"; write_guard_diagnostics(stream, forward.before);
  stream << ",\"after\":"; write_guard_diagnostics(stream, forward.after);
  stream << ",\"refinement\":"; write_refinement_diagnostics(stream, forward.refinement);
  stream << '}';
}

inline void write_forward_diagnostics(std::FILE* stream, const ForwardStats& forward) {
  std::fprintf(stream, "{\"systems\":%lld,\"guard_failures\":%lld,\"attempts\":%lld,\"solves\":%lld,"
      "\"accepted\":%lld,\"tag\":\"%s\",\"pivot_sequence\":%lld,\"entering_row\":",
      static_cast<long long>(forward.systems), static_cast<long long>(forward.guard_failures),
      static_cast<long long>(forward.attempts), static_cast<long long>(forward.solves),
      static_cast<long long>(forward.accepted), forward.tag.c_str(), static_cast<long long>(forward.pivot_sequence));
  if (forward.entering_row) std::fprintf(stream, "%lld", static_cast<long long>(*forward.entering_row));
  else std::fputs("null", stream);
  std::fputs(",\"entering_basis_position\":", stream);
  if (forward.entering_basis_position) std::fprintf(stream, "%lld", static_cast<long long>(*forward.entering_basis_position));
  else std::fputs("null", stream);
  std::fputs(",\"before\":", stream); write_guard_diagnostics(stream, forward.before);
  std::fputs(",\"after\":", stream); write_guard_diagnostics(stream, forward.after);
  std::fputs(",\"refinement\":", stream); write_refinement_diagnostics(stream, forward.refinement);
  std::fputc('}', stream);
}

inline void write_transition_diagnostics(std::ostream& stream, const Phase1TransitionStats& transition) {
  stream << std::setprecision(17) << "{\"status\":\"" << transition.status << "\",\"trials\":" << transition.trials
      << ",\"strict_accepted\":" << transition.strict_accepted << ",\"near_attempts\":" << transition.near_attempts
      << ",\"near_accepted\":" << transition.near_accepted << ",\"checked_rows\":" << transition.checked_rows;
  const auto value = [&](const char* name, const auto& item) {
    stream << ",\"" << name << "\":"; if (item) stream << *item; else stream << "null";
  };
  value("blocking_row", transition.blocking_row); value("released_row", transition.released_row);
  value("max_slack_change_row", transition.max_slack_change_row);
  value("old_phi", transition.old_phi); value("new_phi", transition.new_phi); value("delta_phi", transition.delta_phi);
  value("blocking_slack", transition.blocking_slack); value("blocking_tau", transition.blocking_tau);
  value("released_step", transition.released_step); value("released_tau", transition.released_tau);
  value("max_slack_change_ratio", transition.max_slack_change_ratio);
  stream << ",\"blocking_near\":" << (transition.blocking_near ? "true" : "false")
      << ",\"released_near\":" << (transition.released_near ? "true" : "false")
      << ",\"all_rows_near\":" << (transition.all_rows_near ? "true" : "false") << '}';
}

inline void write_transition_diagnostics(std::FILE* stream, const Phase1TransitionStats& transition) {
  std::fprintf(stream, "{\"status\":\"%s\",\"trials\":%lld,\"strict_accepted\":%lld,\"near_attempts\":%lld,"
      "\"near_accepted\":%lld,\"checked_rows\":%lld", transition.status.c_str(),
      static_cast<long long>(transition.trials), static_cast<long long>(transition.strict_accepted),
      static_cast<long long>(transition.near_attempts), static_cast<long long>(transition.near_accepted),
      static_cast<long long>(transition.checked_rows));
  const auto index = [&](const char* name, const std::optional<Index>& value) {
    std::fprintf(stream, ",\"%s\":", name);
    if (value) std::fprintf(stream, "%lld", static_cast<long long>(*value)); else std::fputs("null", stream);
  };
  const auto number = [&](const char* name, const std::optional<double>& value) {
    std::fprintf(stream, ",\"%s\":", name);
    if (value) std::fprintf(stream, "%.17g", *value); else std::fputs("null", stream);
  };
  index("blocking_row", transition.blocking_row); index("released_row", transition.released_row);
  index("max_slack_change_row", transition.max_slack_change_row);
  number("old_phi", transition.old_phi); number("new_phi", transition.new_phi); number("delta_phi", transition.delta_phi);
  number("blocking_slack", transition.blocking_slack); number("blocking_tau", transition.blocking_tau);
  number("released_step", transition.released_step); number("released_tau", transition.released_tau);
  number("max_slack_change_ratio", transition.max_slack_change_ratio);
  std::fprintf(stream, ",\"blocking_near\":%s,\"released_near\":%s,\"all_rows_near\":%s}",
      transition.blocking_near ? "true" : "false", transition.released_near ? "true" : "false",
      transition.all_rows_near ? "true" : "false");
}

inline void write_history_diagnostics(std::ostream& stream, const HistoryStats& history) {
  stream << "{\"status\":\"" << history.status << "\",\"initialized\":" << (history.initialized ? "true" : "false");
  const auto count = [&](const char* name, Index value) { stream << ",\"" << name << "\":" << value; };
  count("states", history.states); count("swaps", history.swaps); count("candidate_checks", history.candidate_checks);
  count("hash_comparisons", history.hash_comparisons); count("hash_matches", history.hash_matches);
  count("exact_checks", history.exact_checks); count("replay_swaps", history.replay_swaps);
  count("membership_comparisons", history.membership_comparisons); count("repeats", history.repeats);
  stream << ",\"repeated_state\":";
  if (history.repeated_state) stream << *history.repeated_state; else stream << "null";
  stream << '}';
}

inline void write_history_diagnostics(std::FILE* stream, const HistoryStats& history) {
  std::fprintf(stream, "{\"status\":\"%s\",\"initialized\":%s", history.status.c_str(), history.initialized ? "true" : "false");
  const auto count = [&](const char* name, Index value) { std::fprintf(stream, ",\"%s\":%lld", name, static_cast<long long>(value)); };
  count("states", history.states); count("swaps", history.swaps); count("candidate_checks", history.candidate_checks);
  count("hash_comparisons", history.hash_comparisons); count("hash_matches", history.hash_matches);
  count("exact_checks", history.exact_checks); count("replay_swaps", history.replay_swaps);
  count("membership_comparisons", history.membership_comparisons); count("repeats", history.repeats);
  std::fputs(",\"repeated_state\":", stream);
  if (history.repeated_state) std::fprintf(stream, "%lld", static_cast<long long>(*history.repeated_state));
  else std::fputs("null", stream);
  std::fputc('}', stream);
}

inline void write_search_diagnostics(std::ostream& stream, const SearchStats& search) {
  stream << std::setprecision(17) << "{\"status\":\"" << search.status << '"';
  const auto count = [&](const char* name, Index value) { stream << ",\"" << name << "\":" << value; };
  count("bases", search.bases); count("candidates", search.candidates); count("trials", search.trials);
  count("non_descent", search.non_descent); count("descent", search.descent); count("exhausted", search.exhausted);
  count("basis_pivots", search.basis_pivots); count("candidate_count", search.candidate_count);
  count("zero_ineq_rows", search.zero_ineq_rows); count("zero_eq_rows", search.zero_eq_rows);
  const auto value = [&](const char* name, const auto& item) {
    stream << ",\"" << name << "\":"; if (item) stream << *item; else stream << "null";
  };
  value("ordinal", search.ordinal); value("entering_row", search.entering_row);
  value("entering_basis_position", search.entering_basis_position);
  value("priced_rc", search.priced_rc); value("linear_slope", search.linear_slope);
  value("zero_ineq", search.zero_ineq); value("zero_eq", search.zero_eq);
  value("true_derivative", search.true_derivative); value("linear_minus_price", search.linear_minus_price);
  stream << ",\"derivative_complete\":" << (search.derivative_complete ? "true" : "false")
      << ",\"step_started\":" << (search.step_started ? "true" : "false") << '}';
}

inline void write_search_diagnostics(std::FILE* stream, const SearchStats& search) {
  std::fprintf(stream, "{\"status\":\"%s\"", search.status.c_str());
  const auto count = [&](const char* name, Index value) { std::fprintf(stream, ",\"%s\":%lld", name, static_cast<long long>(value)); };
  count("bases", search.bases); count("candidates", search.candidates); count("trials", search.trials);
  count("non_descent", search.non_descent); count("descent", search.descent); count("exhausted", search.exhausted);
  count("basis_pivots", search.basis_pivots); count("candidate_count", search.candidate_count);
  count("zero_ineq_rows", search.zero_ineq_rows); count("zero_eq_rows", search.zero_eq_rows);
  const auto index = [&](const char* name, const std::optional<Index>& value) {
    std::fprintf(stream, ",\"%s\":", name);
    if (value) std::fprintf(stream, "%lld", static_cast<long long>(*value)); else std::fputs("null", stream);
  };
  const auto number = [&](const char* name, const std::optional<double>& value) {
    std::fprintf(stream, ",\"%s\":", name);
    if (value) std::fprintf(stream, "%.17g", *value); else std::fputs("null", stream);
  };
  index("ordinal", search.ordinal); index("entering_row", search.entering_row);
  index("entering_basis_position", search.entering_basis_position);
  number("priced_rc", search.priced_rc); number("linear_slope", search.linear_slope);
  number("zero_ineq", search.zero_ineq); number("zero_eq", search.zero_eq);
  number("true_derivative", search.true_derivative); number("linear_minus_price", search.linear_minus_price);
  std::fprintf(stream, ",\"derivative_complete\":%s,\"step_started\":%s}",
      search.derivative_complete ? "true" : "false", search.step_started ? "true" : "false");
}

inline void write_eta_diagnostics(std::ostream& stream, const EtaStats& eta) {
  stream << std::setprecision(17) << "{\"status\":\"" << eta.status << "\",\"last_mode\":\"" << eta.last_mode
      << "\",\"last_fallback\":\"" << eta.last_fallback << '"';
  const auto count = [&](const char* name, Index value) { stream << ",\"" << name << "\":" << value; };
  count("prepares", eta.prepares); count("eta_trials", eta.eta_trials); count("full_refactor_trials", eta.full_refactor_trials);
  count("eta_commits", eta.eta_commits); count("full_refactor_commits", eta.full_refactor_commits);
  count("chain_length", eta.chain_length); count("max_chain_length", eta.max_chain_length); count("fallbacks", eta.fallbacks);
  count("forward_calls", eta.forward_calls); count("transpose_calls", eta.transpose_calls);
  count("base_forward_solves", eta.base_forward_solves); count("base_transpose_solves", eta.base_transpose_solves);
  count("forward_applications", eta.forward_applications); count("transpose_applications", eta.transpose_applications);
  count("work_units", eta.work_units); count("stored_entries", eta.stored_entries); count("peak_stored_entries", eta.peak_stored_entries);
  stream << ",\"last_alpha\":"; if (eta.last_alpha) stream << *eta.last_alpha; else stream << "null";
  stream << '}';
}

inline void write_eta_diagnostics(std::FILE* stream, const EtaStats& eta) {
  std::fprintf(stream, "{\"status\":\"%s\",\"last_mode\":\"%s\",\"last_fallback\":\"%s\"",
      eta.status, eta.last_mode, eta.last_fallback);
  const auto count = [&](const char* name, Index value) { std::fprintf(stream, ",\"%s\":%lld", name, static_cast<long long>(value)); };
  count("prepares", eta.prepares); count("eta_trials", eta.eta_trials); count("full_refactor_trials", eta.full_refactor_trials);
  count("eta_commits", eta.eta_commits); count("full_refactor_commits", eta.full_refactor_commits);
  count("chain_length", eta.chain_length); count("max_chain_length", eta.max_chain_length); count("fallbacks", eta.fallbacks);
  count("forward_calls", eta.forward_calls); count("transpose_calls", eta.transpose_calls);
  count("base_forward_solves", eta.base_forward_solves); count("base_transpose_solves", eta.base_transpose_solves);
  count("forward_applications", eta.forward_applications); count("transpose_applications", eta.transpose_applications);
  count("work_units", eta.work_units); count("stored_entries", eta.stored_entries); count("peak_stored_entries", eta.peak_stored_entries);
  std::fputs(",\"last_alpha\":", stream); if (eta.last_alpha) std::fprintf(stream, "%.17g", *eta.last_alpha);
  else std::fputs("null", stream);
  std::fputc('}', stream);
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
  integer("factor_nonzeros", stats.factor_nonzeros); integer("last_factor_nonzeros", stats.last_factor_nonzeros);
  integer("primal_solves", stats.primal_solves);
  integer("dual_solves", stats.dual_solves); integer("direction_solves", stats.direction_solves);
  integer("slack_matvecs", stats.slack_matvecs); integer("direction_matvecs", stats.direction_matvecs);
  integer("pricing_matvecs", stats.pricing_matvecs); integer("roundoff_rows", stats.roundoff_rows);
  integer("final_selected_rows_count", static_cast<Index>(stats.final_selected_rows.size()));
  number("initial_phi", stats.initial_phi); number("final_phi", stats.final_phi);
  number("initial_max_violation", stats.initial_max_violation); number("final_max_violation", stats.final_max_violation);
  number("max_roundoff_bound", stats.max_roundoff_bound); number("last_right_derivative", stats.last_right_derivative);
  number("last_step", stats.last_step); number("max_selected_residual", stats.max_selected_residual);
  number("max_direction_residual", stats.max_direction_residual);
  number("last_point_error", stats.last_point_error); number("last_point_bound", stats.last_point_bound);
  stream << ",\"initial_guard\":"; write_guard_diagnostics(stream, stats.initial_guard);
  stream << ",\"refined_guard\":"; write_guard_diagnostics(stream, stats.refined_guard);
  stream << ",\"initial_refinement\":"; write_refinement_diagnostics(stream, stats.initial_refinement);
  stream << ",\"direction_forward\":"; write_forward_diagnostics(stream, stats.direction_forward);
  stream << ",\"rebuild_forward\":"; write_forward_diagnostics(stream, stats.rebuild_forward);
  stream << ",\"phase1_transition\":"; write_transition_diagnostics(stream, stats.phase1_transition);
  stream << ",\"history\":"; write_history_diagnostics(stream, stats.history);
  stream << ",\"search\":"; write_search_diagnostics(stream, stats.search);
  stream << ",\"eta\":"; write_eta_diagnostics(stream, stats.eta);
  stream << "}}";
}

}  // solver::v81_basis_repair
