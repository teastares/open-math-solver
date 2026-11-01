#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <utility>

namespace solver_v112_profile {

enum class Stage : std::size_t {
  total, canonicalize, kkt_setup, normal_setup, symbolic_analysis, factor,
  normal_factor, rectangular_assembly, explicit_gram_assembly, cholmod_numeric,
  factor_check, hsd_constant_direct, hsd_direction_direct, direct, normal_solve,
  cholmod_solve, preconditioner, hsd_column_product, hsd_true_product, coupled,
  point_update, assess_homogeneous, verify_local, initialize, run_homogeneous,
  corrector_rhs, homogeneous_step, cone_multiply, cone_transpose_multiply,
  cholmod_factor_adapter, solve_factored, presolve_control, dot, norm, axpy, count
};

inline constexpr std::size_t stage_count = static_cast<std::size_t>(Stage::count);
inline constexpr std::array<const char*, stage_count> names{
    "total", "canonicalize", "kkt_setup", "normal_setup", "symbolic_analysis", "factor",
    "normal_factor", "rectangular_assembly", "explicit_gram_assembly", "cholmod_numeric",
    "factor_check", "hsd_constant_direct", "hsd_direction_direct", "direct", "normal_solve",
    "cholmod_solve", "preconditioner", "hsd_column_product", "hsd_true_product", "coupled",
    "point_update", "assess_homogeneous", "verify_local", "initialize", "run_homogeneous",
    "corrector_rhs", "homogeneous_step", "cone_multiply", "cone_transpose_multiply",
    "cholmod_factor_adapter", "solve_factored", "presolve_control", "dot", "norm", "axpy"};

struct Entry {
  std::int64_t inclusive_ns = 0, exclusive_ns = 0;
  std::uint64_t started = 0, completed = 0, unwound = 0;
};

struct Stats {
  std::array<Entry, stage_count> entries{};
  std::uint64_t clock_reads = 0;
};

using Now = std::int64_t (*)() noexcept;
struct Scope;

namespace implementation {

struct Context {
  Stats* stats = nullptr;
  Scope* parent = nullptr;
  Now fake = nullptr;
};
inline thread_local Context* current = nullptr;

inline bool environment_enabled() noexcept {
  const auto* value = std::getenv("SOLVER_V112_PROFILE");
  return value && value[0] == '1' && value[1] == '\0';
}

inline std::int64_t now(Context& context) noexcept {
  ++context.stats->clock_reads;
  if (context.fake) return context.fake();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // 命名空间 implementation

struct Scope {
 private:
  implementation::Context* context_ = nullptr;
  Scope* previous_ = nullptr;
  std::size_t index_ = 0;
  std::int64_t start_ = 0, children_ = 0;
  int exceptions_ = 0;

 public:
  explicit Scope(Stage stage) noexcept : context_(implementation::current) {
    // 关闭时不读时钟、不查异常计数、不分配，也不修改其它Report的作用域链。
    if (!context_) return;
    index_ = static_cast<std::size_t>(stage);
    if (index_ >= stage_count) { context_ = nullptr; return; }
    previous_ = context_->parent;
    exceptions_ = std::uncaught_exceptions();
    start_ = implementation::now(*context_);
    ++context_->stats->entries[index_].started;
    context_->parent = this;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Scope(Scope&&) = delete;
  Scope& operator=(Scope&&) = delete;
  ~Scope() noexcept {
    if (!context_) return;
    const auto duration = implementation::now(*context_) - start_;
    auto& entry = context_->stats->entries[index_];
    entry.inclusive_ns += duration;
    entry.exclusive_ns += duration - children_;
    if (std::uncaught_exceptions() > exceptions_) ++entry.unwound;
    else ++entry.completed;
    context_->parent = previous_;
    if (previous_) previous_->children_ += duration;
  }
};

template<class Function> decltype(auto) call(Stage stage, Function&& function) {
  Scope scope(stage);
  return std::forward<Function>(function)();
}

template<class Value, class Function> Value materialize(Stage stage, Function&& function) {
  Scope scope(stage);
  // Eigen惰性表达式必须在此处物化后才结束计时，不能只计表达式构造。
  return Value(std::forward<Function>(function)());
}

inline std::int64_t exclusive_sum(const Stats& stats) noexcept {
  std::int64_t result = 0;
  for (const auto& entry : stats.entries) result += entry.exclusive_ns;
  return result;
}

inline bool counts_closed(const Stats& stats) noexcept {
  std::uint64_t started = 0;
  for (const auto& entry : stats.entries) {
    if (entry.started != entry.completed + entry.unwound) return false;
    started += entry.started;
  }
  return stats.clock_reads == 2 * started;
}

inline bool conserved(const Stats& stats) noexcept {
  return exclusive_sum(stats) == stats.entries[static_cast<std::size_t>(Stage::total)].inclusive_ns;
}

class Report {
 public:
  Stats stats;

 private:
  bool active_ = false, emit_ = false;
  implementation::Context context_;
  implementation::Context* previous_ = nullptr;

 public:
  Report() noexcept : Report(implementation::environment_enabled(), true) {}
  explicit Report(bool active, bool emit = false, Now fake = nullptr) noexcept
      : active_(active), emit_(emit), context_{&stats, nullptr, fake}, previous_(implementation::current) {
    // 关闭的嵌套Report也建立隔离边界；不能把其内部Scope记到外Report。
    implementation::current = active_ ? &context_ : nullptr;
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
  Report(Report&&) = delete;
  Report& operator=(Report&&) = delete;
  bool active() const noexcept { return active_; }
  bool stack_empty() const noexcept { return context_.parent == nullptr; }
  ~Report() noexcept {
    if (active_ && emit_) {
      const auto sum = exclusive_sum(stats);
      std::fputs("V112_PROFILE {\"schema\":1,\"stages\":{", stderr);
      for (std::size_t index = 0; index < stage_count; ++index) {
        const auto& entry = stats.entries[index];
        std::fprintf(stderr, "%s\"%s\":{\"inclusive_ns\":%lld,\"exclusive_ns\":%lld,"
            "\"started\":%llu,\"completed\":%llu,\"unwound\":%llu}", index ? "," : "", names[index],
            static_cast<long long>(entry.inclusive_ns), static_cast<long long>(entry.exclusive_ns),
            static_cast<unsigned long long>(entry.started), static_cast<unsigned long long>(entry.completed),
            static_cast<unsigned long long>(entry.unwound));
      }
      std::fprintf(stderr, "},\"clock_reads\":%llu,\"exclusive_sum_ns\":%lld,"
          "\"total_inclusive_ns\":%lld,\"conserved\":%s,\"stack_empty\":%s,\"counts_closed\":%s}\n",
          static_cast<unsigned long long>(stats.clock_reads), static_cast<long long>(sum),
          static_cast<long long>(stats.entries[0].inclusive_ns), conserved(stats) ? "true" : "false",
          stack_empty() ? "true" : "false", counts_closed(stats) ? "true" : "false");
    }
    implementation::current = previous_;
  }
};

}  // 命名空间 solver_v112_profile
