#pragma once

#include "solver/lp_optimality.hpp"
#include "ipm/ipx/ipx_config.h"
#include "ipm/ipx/ipx_status.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace solver::detail::logical_basis {
using Index = ipxint;
using LD = long double;

enum class Status { ready, dual_sign_infeasible, invalid_input, unsupported_empty };
inline const char* status_name(Status value) noexcept {
  switch (value) {
    case Status::ready: return "ready";
    case Status::dual_sign_infeasible: return "dual_sign_infeasible";
    case Status::invalid_input: return "invalid_input";
    case Status::unsupported_empty: return "unsupported_empty";
  }
  return "invalid_input";
}

struct Input {
  // costs 已转为最小化；该组件不转换模型、不接受整型属性，也不检查矩阵。
  std::span<const double> costs, lower, upper;
  std::size_t rows = 0;
};
struct Counts {
  std::size_t columns = 0, rows = 0;
  std::size_t fixed = 0, boxed = 0, lower_only = 0, upper_only = 0, free = 0;
  std::size_t selected_lower = 0, selected_upper = 0, selected_zero = 0;
  std::size_t eligible_columns = 0;
  std::size_t blocker_lower_only = 0, blocker_upper_only = 0, blocker_free = 0;
  std::size_t raw_sign_failures = 0;
};
struct Result {
  Status status = Status::invalid_input;
  std::string message;
  std::vector<Index> variable_status, row_status;
  std::vector<double> structural_seed;
  Counts counts;
  LD max_dual_violation = 0, max_dual_ratio = 0;
  std::optional<std::size_t> first_blocker;
};

// B=I 的逻辑列成本为零，所以 pi=0、r_N=c。这里只检查该起始基的对偶符号。
// ready 不代表原始可行、已最优或一定可由 Phase II 求解；后续仍需原暖基重建/原验。
// ready 与 dual_sign_infeasible 都返回完整状态；无效/空输入不发布状态向量。
// budget 在边界及每4096项检查，异常原样上抛；不得把预算中断冒充符号不可行。
template<class Budget>
Result build(const Input& input, const solver::LpOptimalityOptions& options, Budget&& budget) {
  budget();
  Result result;
  const std::size_t n = input.costs.size();
  result.counts.columns = n;
  result.counts.rows = input.rows;
  const auto invalid = [&](const char* message) {
    result.status = Status::invalid_input;
    result.message = message;
    result.variable_status.clear();
    result.row_status.clear();
    result.structural_seed.clear();
  };
  const auto maximum = static_cast<std::size_t>(std::numeric_limits<Index>::max());
  if (input.lower.size() != n || input.upper.size() != n ||
      n > maximum || input.rows > maximum - n ||
      !std::isfinite(options.dual_absolute) || options.dual_absolute < 0 ||
      !std::isfinite(options.dual_relative) || options.dual_relative < 0) {
    invalid("逻辑基维数、索引范围或对偶容差无效");
    budget();
    return result;
  }
  if (n == 0 || input.rows == 0) {
    result.status = Status::unsupported_empty;
    result.message = "当前暖基组件不支持零维逻辑基";
    budget();
    return result;
  }
  result.variable_status.resize(n);
  result.structural_seed.resize(n);
  budget();
  const double inf = std::numeric_limits<double>::infinity();
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095U) == 0) budget();
    const double c = input.costs[col], lower = input.lower[col], upper = input.upper[col];
    if (!std::isfinite(c) || std::isnan(lower) || std::isnan(upper) ||
        lower == inf || upper == -inf || lower > upper) {
      invalid("逻辑基成本或边界无效");
      budget();
      return result;
    }
    const bool finite_lower = std::isfinite(lower), finite_upper = std::isfinite(upper);
    Index state;
    double seed;
    LD violation = 0;
    if (lower == upper) {
      ++result.counts.fixed;
      state = IPX_nonbasic_lb; seed = lower;
    } else if (finite_lower && finite_upper) {
      ++result.counts.boxed;
      state = c < 0 ? IPX_nonbasic_ub : IPX_nonbasic_lb;
      seed = c < 0 ? upper : lower;
    } else if (finite_lower) {
      ++result.counts.lower_only;
      state = IPX_nonbasic_lb; seed = lower;
      violation = std::max<LD>(0, -static_cast<LD>(c));
    } else if (finite_upper) {
      ++result.counts.upper_only;
      state = IPX_nonbasic_ub; seed = upper;
      violation = std::max<LD>(0, static_cast<LD>(c));
    } else {
      ++result.counts.free;
      state = IPX_superbasic; seed = 0;
      violation = std::abs(static_cast<LD>(c));
    }
    if (state == IPX_nonbasic_lb) ++result.counts.selected_lower;
    else if (state == IPX_nonbasic_ub) ++result.counts.selected_upper;
    else ++result.counts.selected_zero;
    result.variable_status[col] = state;
    result.structural_seed[col] = seed;

    // 与 warm_cleanup 在 pi=0 时的真实位置资格完全一致；固定列免符号约束。
    const LD allowed = static_cast<LD>(options.dual_absolute) +
        static_cast<LD>(options.dual_relative) * std::abs(static_cast<LD>(c));
    const LD ratio = allowed > 0 ? violation / allowed : violation == 0 ? 0 :
        std::numeric_limits<LD>::infinity();
    result.max_dual_violation = std::max(result.max_dual_violation, violation);
    result.max_dual_ratio = std::max(result.max_dual_ratio, ratio);
    if (violation > 0) ++result.counts.raw_sign_failures;
    if (ratio <= 1) ++result.counts.eligible_columns;
    else {
      if (!result.first_blocker) result.first_blocker = col;
      if (finite_lower) ++result.counts.blocker_lower_only;
      else if (finite_upper) ++result.counts.blocker_upper_only;
      else ++result.counts.blocker_free;
    }
  }
  budget();
  result.row_status.resize(input.rows);
  for (std::size_t row = 0; row < input.rows; ++row) {
    if ((row & 4095U) == 0) budget();
    result.row_status[row] = IPX_basic;
  }
  budget();
  if (result.counts.eligible_columns == n) {
    result.status = Status::ready;
    result.message = "全逻辑列基满足当前容差下的对偶符号资格";
  } else {
    result.status = Status::dual_sign_infeasible;
    result.message = "全逻辑列基存在无法用有限界满足的对偶符号，不提供 Phase I";
  }
  return result;
}

inline Result build(const Input& input, const solver::LpOptimalityOptions& options = {}) {
  return build(input, options, [] {});
}
}  // 命名空间 solver::detail::logical_basis
