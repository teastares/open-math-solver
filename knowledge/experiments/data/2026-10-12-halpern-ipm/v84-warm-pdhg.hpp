#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace solver::v84_warm_pdhg {

using Index = std::int64_t;
inline constexpr Index budget_check_work_interval = 256;
inline constexpr long double step_safety = 0.9L;

struct View {
  Index n, m;
  std::span<const Index> starts, rows;
  std::span<const double> values, costs, rhs;
  const std::vector<bool>& inequality;
};

struct Limits {
  Index max_iterations = 20000;
  Index max_work = 500000000;
  Index check_interval = 64;
};

struct Stats {
  const char* phase = "not_started";
  bool seed_available = false;
  Index setup_started = 0, setup_completed = 0;
  Index steps_started = 0, steps_completed = 0;
  Index transpose_started = 0, transpose_completed = 0;
  Index multiply_started = 0, multiply_completed = 0;
  Index budget_checks = 0, assessments_started = 0, assessments_completed = 0, work_units = 0;
  std::optional<Index> last_assessed_step;
};

struct Result {
  const char* status = "not_started";
  bool accepted = false;
  std::vector<double> x, z;
  Stats stats;
};

using Budget = std::function<void()>;
using Assessment = std::function<bool(Index, const std::vector<double>&, const std::vector<double>&)>;

namespace implementation {

struct Stop { const char* status; };

class Work {
 public:
  Work(const Limits& limits, const Budget& budget, Result& result)
      : limits_(limits), budget_(budget), result_(result) {}

  void check() {
    ++result_.stats.budget_checks;
    try {
      if (budget_) budget_();
    } catch (...) {
      callback_failed_ = true;
      result_.status = "budget_exception";
      throw;
    }
  }

  // 工作单位为自有元素操作的代理量，不是 FLOP；外部验收只计调用及实际时间。
  void take() {
    if (result_.stats.work_units >= limits_.max_work) throw Stop{"work_limit"};
    ++result_.stats.work_units;
    if (++pending_ == budget_check_work_interval) { pending_ = 0; check(); }
  }

  void assessment_failed() noexcept {
    callback_failed_ = true;
    result_.status = "assessment_exception";
  }
  bool callback_failed() const noexcept { return callback_failed_; }

 private:
  const Limits& limits_;
  Budget budget_;
  Result& result_;
  Index pending_ = 0;
  bool callback_failed_ = false;
};

inline long double finite(long double value) {
  if (!std::isfinite(value)) throw Stop{"arithmetic_failure"};
  return value;
}

inline long double product(long double left, long double right) {
  const long double value = finite(left * right);
  if (left != 0 && right != 0 && value == 0) throw Stop{"arithmetic_failure"};
  return value;
}

inline double narrow(long double value) {
  const double converted = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(converted) || (value != 0 && converted == 0))
    throw Stop{"arithmetic_failure"};
  return converted;
}

template<class Value>
std::vector<Value> zeroes(std::size_t size, Work& work) {
  work.check();
  std::vector<Value> values;
  values.reserve(size);
  for (std::size_t i = 0; i < size; ++i) { work.take(); values.push_back(Value{}); }
  work.check();
  return values;
}

inline bool assess(const Assessment& assessment, Work& work, Result& output) {
  auto& stats = output.stats;
  const auto* previous_phase = stats.phase;
  stats.phase = "assessment";
  work.check();
  ++stats.assessments_started;
  bool accepted;
  try {
    accepted = assessment(stats.steps_completed, output.x, output.z);
  } catch (...) {
    work.assessment_failed();
    throw;
  }
  ++stats.assessments_completed;
  stats.last_assessed_step = stats.steps_completed;
  work.check();
  if (accepted) {
    output.accepted = true;
    output.status = "accepted";
    stats.phase = "finished";
    return true;
  }
  stats.phase = previous_phase;
  return false;
}

}  // 命名空间 implementation

inline void run(const View& view, const Limits& limits, std::span<const double> seed_x,
                std::span<const double> seed_z, double tau, const Budget& budget,
                const Assessment& assessment, Result& output) {
  using implementation::Stop;
  // 允许种子引用上次 output 的向量；移动旧载荷保活，避免重置结果时使 span 悬空。
  [[maybe_unused]] Result retained_input = std::move(output);
  output = Result{};
  output.status = "running";
  output.stats.phase = "validation";
  try {
    implementation::Work work(limits, budget, output);
    try {
      work.check();
      if (limits.max_iterations < 0 || limits.max_work < 0 || limits.check_interval <= 0 ||
          view.n < 0 || view.m < 0 || !std::isfinite(tau) || !(tau > 0) || !assessment ||
          static_cast<std::uint64_t>(view.n) >= std::numeric_limits<std::size_t>::max() ||
          static_cast<std::uint64_t>(view.m) > std::numeric_limits<std::size_t>::max()) throw Stop{"invalid_input"};
      const auto n = static_cast<std::size_t>(view.n), m = static_cast<std::size_t>(view.m);
      if (view.starts.size() != n + 1 || view.costs.size() != n || view.rhs.size() != m ||
          view.inequality.size() != m || seed_x.size() != n || seed_z.size() != m ||
          view.values.size() != view.rows.size() ||
          view.values.size() > static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
          view.starts.front() != 0 || view.starts.back() != static_cast<Index>(view.values.size()))
        throw Stop{"invalid_input"};
      for (std::size_t col = 0; col < n; ++col) {
        work.take();
        if (!std::isfinite(view.costs[col]) || !std::isfinite(seed_x[col]) || view.starts[col] < 0 ||
            view.starts[col] > view.starts[col + 1] || view.starts[col + 1] > static_cast<Index>(view.values.size()))
          throw Stop{"invalid_input"};
        for (Index k = view.starts[col]; k < view.starts[col + 1]; ++k) {
          work.take();
          const auto entry = static_cast<std::size_t>(k);
          if (view.rows[entry] < 0 || view.rows[entry] >= view.m || !std::isfinite(view.values[entry]))
            throw Stop{"invalid_input"};
        }
      }
      for (std::size_t row = 0; row < m; ++row) {
        work.take();
        if (!std::isfinite(view.rhs[row]) || !std::isfinite(seed_z[row]) ||
            (view.inequality[row] && seed_z[row] < 0)) throw Stop{"invalid_input"};
      }
      output.stats.phase = "seed";
      std::vector<double> x, z;
      work.check(); x.reserve(n); z.reserve(m); work.check();
      for (double value : seed_x) {
        work.take();
        const long double normalized = static_cast<long double>(value) / tau;
        if (value != 0 && normalized == 0) throw Stop{"arithmetic_failure"};
        x.push_back(implementation::narrow(normalized));
      }
      for (double value : seed_z) {
        work.take();
        const long double normalized = static_cast<long double>(value) / tau;
        if (value != 0 && normalized == 0) throw Stop{"arithmetic_failure"};
        z.push_back(implementation::narrow(normalized));
      }
      work.check();
      output.x.swap(x); output.z.swap(z);
      output.stats.seed_available = true;
      if (implementation::assess(assessment, work, output)) return;
      if (limits.max_iterations == 0) {
        output.status = "iteration_limit";
        output.stats.phase = "finished";
        return;
      }

      output.stats.phase = "setup";
      ++output.stats.setup_started;
      auto row_sum = implementation::zeroes<long double>(m, work);
      auto primal_step = implementation::zeroes<double>(n, work);
      auto dual_step = implementation::zeroes<double>(m, work);
      for (std::size_t col = 0; col < n; ++col) {
        work.take();
        long double column_sum = 0;
        for (Index k = view.starts[col]; k < view.starts[col + 1]; ++k) {
          work.take();
          const auto entry = static_cast<std::size_t>(k), row = static_cast<std::size_t>(view.rows[entry]);
          const long double value = std::abs(static_cast<long double>(view.values[entry]));
          column_sum = implementation::finite(column_sum + value);
          row_sum[row] = implementation::finite(row_sum[row] + value);
        }
        primal_step[col] = implementation::narrow(step_safety / std::max(1.0L, column_sum));
        if (!(primal_step[col] > 0)) throw Stop{"arithmetic_failure"};
      }
      for (std::size_t row = 0; row < m; ++row) {
        work.take();
        dual_step[row] = implementation::narrow(step_safety / std::max(1.0L, row_sum[row]));
        if (!(dual_step[row] > 0)) throw Stop{"arithmetic_failure"};
      }
      auto transpose = implementation::zeroes<long double>(n, work);
      auto activity = implementation::zeroes<long double>(m, work);
      auto next_x = implementation::zeroes<double>(n, work);
      auto next_z = implementation::zeroes<double>(m, work);
      auto extrapolated = implementation::zeroes<double>(n, work);
      work.check();
      ++output.stats.setup_completed;
      while (output.stats.steps_completed < limits.max_iterations) {
        work.check(); work.take();
        ++output.stats.steps_started;
        output.stats.phase = "transpose";
        ++output.stats.transpose_started;
        work.check();
        for (std::size_t col = 0; col < n; ++col) {
          work.take();
          long double sum = 0;
          for (Index k = view.starts[col]; k < view.starts[col + 1]; ++k) {
            work.take();
            const auto entry = static_cast<std::size_t>(k), row = static_cast<std::size_t>(view.rows[entry]);
            sum = implementation::finite(sum + implementation::product(view.values[entry], output.z[row]));
          }
          transpose[col] = sum;
        }
        work.check();
        ++output.stats.transpose_completed;
        output.stats.phase = "primal";
        for (std::size_t col = 0; col < n; ++col) {
          work.take();
          const long double gradient = implementation::finite(static_cast<long double>(view.costs[col]) + transpose[col]);
          next_x[col] = implementation::narrow(static_cast<long double>(output.x[col]) -
              implementation::product(primal_step[col], gradient));
          work.take();
          extrapolated[col] = implementation::narrow(2 * static_cast<long double>(next_x[col]) - output.x[col]);
        }
        output.stats.phase = "multiply";
        ++output.stats.multiply_started;
        work.check();
        for (std::size_t row = 0; row < m; ++row) { work.take(); activity[row] = 0; }
        for (std::size_t col = 0; col < n; ++col) {
          work.take();
          for (Index k = view.starts[col]; k < view.starts[col + 1]; ++k) {
            work.take();
            const auto entry = static_cast<std::size_t>(k), row = static_cast<std::size_t>(view.rows[entry]);
            activity[row] = implementation::finite(activity[row] + implementation::product(view.values[entry], extrapolated[col]));
          }
        }
        work.check();
        ++output.stats.multiply_completed;
        output.stats.phase = "dual";
        for (std::size_t row = 0; row < m; ++row) {
          work.take();
          const long double defect = implementation::finite(activity[row] - view.rhs[row]);
          long double value = implementation::finite(static_cast<long double>(output.z[row]) +
              implementation::product(dual_step[row], defect));
          if (view.inequality[row]) { work.take(); value = std::max(0.0L, value); }
          next_z[row] = implementation::narrow(value);
        }
        output.stats.phase = "commit";
        work.take(); work.check();
        // 同一无回调边界提交两向量；此前任何失败都保留上一次完整点。
        output.x.swap(next_x); output.z.swap(next_z);
        ++output.stats.steps_completed;
        if (output.stats.steps_completed % limits.check_interval == 0 &&
            implementation::assess(assessment, work, output)) return;
      }
      output.status = "iteration_limit";
      output.stats.phase = "finished";
    } catch (const Stop& failure) {
      if (work.callback_failed()) throw;
      output.status = failure.status;
    }
    work.check();
    // 有界出口只评估已提交点，不继续数学迭代；外部验收成本仍由同一预算约束。
    if (output.stats.seed_available && (!output.stats.last_assessed_step ||
        *output.stats.last_assessed_step != output.stats.steps_completed))
      implementation::assess(assessment, work, output);
  } catch (...) {
    if (std::string_view(output.status) == "running") output.status = "exception";
    throw;
  }
}

inline void write_stats_diagnostics(std::FILE* stream, const Stats& stats) {
  std::fprintf(stream, "{\"phase\":\"%s\",\"seed_available\":%s", stats.phase, stats.seed_available ? "true" : "false");
  const auto count = [&](const char* name, Index value) { std::fprintf(stream, ",\"%s\":%lld", name, static_cast<long long>(value)); };
  count("setup_started", stats.setup_started); count("setup_completed", stats.setup_completed);
  count("steps_started", stats.steps_started); count("steps_completed", stats.steps_completed);
  count("transpose_started", stats.transpose_started); count("transpose_completed", stats.transpose_completed);
  count("multiply_started", stats.multiply_started); count("multiply_completed", stats.multiply_completed);
  count("budget_checks", stats.budget_checks); count("assessments_started", stats.assessments_started);
  count("assessments_completed", stats.assessments_completed); count("work_units", stats.work_units);
  std::fputs(",\"last_assessed_step\":", stream);
  if (stats.last_assessed_step) std::fprintf(stream, "%lld", static_cast<long long>(*stats.last_assessed_step));
  else std::fputs("null", stream);
  std::fputc('}', stream);
}

inline void write_diagnostics(std::FILE* stream, const Result& result) {
  std::fprintf(stream, "{\"status\":\"%s\",\"accepted\":%s,\"stats\":", result.status, result.accepted ? "true" : "false");
  write_stats_diagnostics(stream, result.stats);
  std::fputc('}', stream);
}

}  // 命名空间 solver::v84_warm_pdhg
