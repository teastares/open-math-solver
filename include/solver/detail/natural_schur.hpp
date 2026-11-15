#pragma once

#include "solver/detail/fma_axpy.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace solver::detail::natural_schur {

using Index = std::int64_t;
using Budget = std::function<void()>;
inline constexpr std::size_t budget_batch = 4096;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

struct Work {
  Index started = 0, completed = 0, terms = 0, budget_checks = 0, environment_checks = 0;
  Index divisions = 0, products = 0, accumulations = 0, abs_accumulations = 0;
  Index final_subtractions = 0, bound_operations = 0, enclosure_steps = 0;
  const char* decision = "not_started";
  std::optional<long double> natural, abs_products, lower, upper, error_bound, margin;
};

namespace implementation {

inline void increment(Index& value) {
  if (value < 0 || value == std::numeric_limits<Index>::max())
    throw std::invalid_argument("自然Schur分母的费用计数溢出");
  ++value;
}
[[noreturn]] inline void fail(Work& work, const char* decision, const char* message) {
  work.decision = decision;
  throw Failure(message);
}
inline long double finite(long double value, Work& work) {
  if (!std::isfinite(value)) fail(work, "nonfinite", "自然Schur分母的数值或包围区间非有限");
  return value;
}
inline void environment(Work& work) {
  increment(work.environment_checks);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  constexpr bool supported = std::numeric_limits<long double>::is_iec559 &&
      std::numeric_limits<long double>::radix == 2 && std::numeric_limits<long double>::digits == 64 &&
      std::numeric_limits<long double>::max_exponent == 16384 &&
      std::numeric_limits<long double>::has_denorm == std::denorm_present;
  const auto control = fma_axpy::read_control_state();
  if (supported && std::fegetround() == FE_TONEAREST &&
      fma_axpy::environment_supported(control.mxcsr, control.x87_control)) return;
#endif
  fail(work, "unsupported", "自然Schur包围要求最近舍入、渐进下溢及受支持的64位x87精度环境");
}
inline void budget(Work& work, const Budget& check) {
  increment(work.budget_checks);
  if (check) check();
  // 回调可改变控制环境；每次成功返回后重验，不改变舍入态或清除旗标。
  environment(work);
}
// volatile存储使每个基本运算先按LD精度完成，不能融合成另一个算术表达式。
inline long double add(long double left, long double right) {
  volatile long double result = left + right;
  return result;
}
inline long double subtract(long double left, long double right) {
  volatile long double result = left - right;
  return result;
}
inline long double multiply(long double left, long double right) {
  volatile long double result = left * right;
  return result;
}
inline long double divide(long double left, long double right) {
  volatile long double result = left / right;
  return result;
}
inline long double outward(long double value, bool upward, Work& work) {
  finite(value, work);
  increment(work.enclosure_steps);
  return finite(std::nextafter(value, upward ? std::numeric_limits<long double>::infinity() :
      -std::numeric_limits<long double>::infinity()), work);
}
inline long double bound_add(long double left, long double right, Work& work) {
  increment(work.bound_operations);
  return finite(add(left, right), work);
}
inline long double bound_subtract(long double left, long double right, Work& work) {
  increment(work.bound_operations);
  return finite(subtract(left, right), work);
}
inline long double bound_multiply(long double left, long double right, Work& work) {
  increment(work.bound_operations);
  return finite(multiply(left, right), work);
}

}  // 命名空间 implementation

// 只认证已表示的(c,b,d,w)所定义的实数表达式，不界定旧P或原模型的误差。
// abs_products是普通LD顺序和的观察值，不是绝对积和的严格上界。
inline long double compute(std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> constant, long double d, Work& work, const Budget& check = {}) {
  const std::array<Index, 12> counters{work.started, work.completed, work.terms, work.budget_checks,
      work.environment_checks, work.divisions, work.products, work.accumulations, work.abs_accumulations,
      work.final_subtractions, work.bound_operations, work.enclosure_steps};
  if (costs.size() > std::numeric_limits<std::size_t>::max() - bounds.size() ||
      costs.size() + bounds.size() != constant.size() ||
      std::any_of(counters.begin(), counters.end(), [](Index value) { return value != 0; }) ||
      !work.decision || std::string_view(work.decision) != "not_started" || work.natural ||
      work.abs_products || work.lower || work.upper || work.error_bound || work.margin)
    throw std::invalid_argument("自然Schur分母的维数、缩放或初始账本无效");
  implementation::budget(work, check);
  // 先确认异常已屏蔽，再检查浮点输入，避免不支持的环境先在NaN比较处陷阱。
  if (!(d > 0) || !std::isfinite(d))
    throw std::invalid_argument("自然Schur分母的缩放必须有限正");
  implementation::increment(work.started);
  work.decision = "running";
  long double dot = 0, absolute = 0, dot_lower = 0, dot_upper = 0;
  for (std::size_t index = 0; index < constant.size(); ++index) {
    if ((index & (budget_batch - 1)) == 0) implementation::budget(work, check);
    implementation::increment(work.terms);
    const long double coefficient = implementation::finite(
        index < costs.size() ? costs[index] : bounds[index - costs.size()], work);
    const long double weight = implementation::finite(constant[index], work);
    implementation::increment(work.divisions);
    const long double quotient = implementation::finite(implementation::divide(coefficient, d), work);
    implementation::increment(work.products);
    const long double product = implementation::finite(implementation::multiply(quotient, weight), work);
    implementation::increment(work.accumulations);
    dot = implementation::finite(implementation::add(dot, product), work);
    implementation::increment(work.abs_accumulations);
    absolute = implementation::finite(implementation::add(absolute, std::abs(product)), work);

    const long double quotient_lower = implementation::outward(quotient, false, work);
    const long double quotient_upper = implementation::outward(quotient, true, work);
    // 负权重交换商区间两端；零权重也执行同样的完整包围流程。
    const long double product_lower = implementation::outward(implementation::bound_multiply(
        weight < 0 ? quotient_upper : quotient_lower, weight, work), false, work);
    const long double product_upper = implementation::outward(implementation::bound_multiply(
        weight < 0 ? quotient_lower : quotient_upper, weight, work), true, work);
    dot_lower = implementation::outward(implementation::bound_add(dot_lower, product_lower, work), false, work);
    dot_upper = implementation::outward(implementation::bound_add(dot_upper, product_upper, work), true, work);
  }
  implementation::increment(work.final_subtractions);
  const long double natural = implementation::finite(implementation::subtract(1, dot), work);
  const long double lower = implementation::outward(implementation::bound_subtract(1, dot_upper, work), false, work);
  const long double upper = implementation::outward(implementation::bound_subtract(1, dot_lower, work), true, work);
  const long double left_error = implementation::outward(std::abs(
      implementation::bound_subtract(natural, lower, work)), true, work);
  const long double right_error = implementation::outward(std::abs(
      implementation::bound_subtract(upper, natural, work)), true, work);
  const bool reliable = lower <= natural && natural <= upper && natural != 0 &&
      (lower > 0 || upper < 0);
  // 只发布完整摘要；末预算或控制态失败仍保留数值事实，但不计完成或交付。
  work.natural = natural; work.abs_products = absolute; work.lower = lower; work.upper = upper;
  work.error_bound = std::max(left_error, right_error);
  work.margin = reliable ? std::min(std::abs(lower), std::abs(upper)) : 0;
  work.decision = reliable ? "reliable" : "uncertain";
  implementation::budget(work, check);
  if (!reliable) implementation::fail(work, "uncertain", "自然Schur分母的包围区间不能可靠排除零");
  implementation::increment(work.completed);
  return natural;
}

}  // 命名空间 solver::detail::natural_schur
