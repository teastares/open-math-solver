#pragma once

#include "solver/detail/current_diagonal_preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver::detail::schur_diagonal_transport {

using Index = current_diagonal::Index;
using Budget = current_diagonal::Budget;
inline constexpr std::size_t budget_batch = 4096;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

struct Work {
  std::size_t columns = 0, rows = 0, nonzeros = 0;
  Index prepare_started = 0, prepare_completed = 0, cache_epoch = 0, cache_hits = 0;
  Index old_build_started = 0, old_build_completed = 0, old_row_terms = 0;
  Index scratch_row_terms = 0, merge_terms = 0, unique_terms = 0;
  Index old_terms = 0, current_terms = 0, column_terms = 0;
  Index input_started = 0, input_completed = 0, input_terms = 0;
  Index direct_started = 0, direct_completed = 0;
  Index output_started = 0, output_completed = 0, output_terms = 0, budget_checks = 0;
  std::optional<long double> old_min, old_max, current_min, current_max, scale_min, scale_max;
};

// 键只保存存储身份；旧R仅在本次重建时读取，不能通过缓存解引用过期世代。
// 调用方在同一owner/epoch内须保持CSC与旧R存储及数值不变。
struct Cache {
  bool valid = false;
  const void* owner = nullptr;
  Index factor_epoch = 0;
  double r_old = 0;
  std::size_t columns = 0, rows = 0, nonzeros = 0;
  const Index* starts_data = nullptr;
  const Index* row_indices_data = nullptr;
  const double* values_data = nullptr;
  const double* old_rows_data = nullptr;
  std::size_t starts_size = 0, row_indices_size = 0, values_size = 0, old_rows_size = 0;
  std::vector<long double> old_diagonal;
};

namespace implementation {

inline void increment(Index& value) {
  if (value < 0 || value == std::numeric_limits<Index>::max())
    throw std::invalid_argument("Schur对角缩放的费用计数溢出");
  ++value;
}
inline void budget(Work& work, const Budget& check) {
  increment(work.budget_checks);
  if (check) check();
}
inline void coordinate_budget(std::size_t index, Work& work, const Budget& check) {
  if ((index & (budget_batch - 1)) == 0) budget(work, check);
}
inline long double finite(long double value) {
  if (!std::isfinite(value)) throw Failure("Schur对角缩放的计算非有限");
  return value;
}
inline double narrow(long double value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
    throw Failure("Schur对角缩放的结果不能表示为double");
  return result;
}
inline long double contribution(long double square, double row) {
  if (!(row > 0) || !std::isfinite(row)) throw Failure("Schur对角要求逐行有限正R");
  return finite(square / static_cast<long double>(row));
}
inline void extrema(long double value, std::optional<long double>& minimum,
    std::optional<long double>& maximum) {
  minimum = minimum ? std::min(*minimum, value) : value;
  maximum = maximum ? std::max(*maximum, value) : value;
}
inline Cache key(const current_diagonal::ValidatedView& view, std::span<const double> old_rows,
    double r_old, Index epoch, const void* owner) {
  Cache result;
  result.owner = owner; result.factor_epoch = epoch; result.r_old = r_old;
  result.columns = view.columns(); result.rows = view.rows(); result.nonzeros = view.nonzeros();
  result.starts_data = view.starts().data(); result.starts_size = view.starts().size();
  result.row_indices_data = view.row_indices().data(); result.row_indices_size = view.row_indices().size();
  result.values_data = view.values().data(); result.values_size = view.values().size();
  result.old_rows_data = old_rows.data(); result.old_rows_size = old_rows.size();
  return result;
}
inline bool same_key(const Cache& cache, const Cache& other) {
  return cache.r_old == other.r_old && cache.columns == other.columns && cache.rows == other.rows &&
      cache.nonzeros == other.nonzeros && cache.starts_data == other.starts_data &&
      cache.starts_size == other.starts_size && cache.row_indices_data == other.row_indices_data &&
      cache.row_indices_size == other.row_indices_size && cache.values_data == other.values_data &&
      cache.values_size == other.values_size && cache.old_rows_data == other.old_rows_data &&
      cache.old_rows_size == other.old_rows_size && cache.old_diagonal.size() == other.columns;
}

}  // 命名空间 implementation

// 每个数学坐标先按原存储顺序LD合并，再平方形成名义Schur对角。
// 这里只构造对角合同缩放，不重排或改写后续G/Gt的原产品顺序。
inline std::vector<double> prepare(const current_diagonal::ValidatedView& view,
    std::span<const double> old_rows, double r_old, Index factor_epoch, const void* owner,
    Cache& cache, Work& work, const Budget& check = {}) {
  work.columns = view.columns(); work.rows = view.rows(); work.nonzeros = view.nonzeros();
  // 极值描述本次准备的对角与实际double尺度，累计费用不清空。
  work.old_min.reset(); work.old_max.reset(); work.current_min.reset(); work.current_max.reset();
  work.scale_min.reset(); work.scale_max.reset();
  implementation::budget(work, check);
  implementation::increment(work.prepare_started);
  if (!owner || factor_epoch <= 0 || old_rows.size() != view.rows())
    throw std::invalid_argument("Schur对角缩放的因子归属、世代或旧R维数无效");
  work.cache_epoch = factor_epoch;
  auto pending = implementation::key(view, old_rows, r_old, factor_epoch, owner);
  const bool same_generation = cache.valid && cache.owner == owner && cache.factor_epoch == factor_epoch;
  if (same_generation && !implementation::same_key(cache, pending))
    throw std::invalid_argument("同一因子世代的Schur对角缓存键发生变化");
  if (!(r_old >= 0) || !std::isfinite(r_old))
    throw Failure("Schur对角缩放要求有限非负原始正则");
  const bool hit = same_generation;
  if (hit) {
    implementation::increment(work.cache_hits);
  } else {
    implementation::increment(work.old_build_started);
    for (std::size_t row = 0; row < old_rows.size(); ++row) {
      implementation::coordinate_budget(row, work, check);
      implementation::increment(work.old_row_terms);
      if (!(old_rows[row] > 0) || !std::isfinite(old_rows[row]))
        throw Failure("Schur对角缩放要求旧R逐行有限正");
    }
    pending.old_diagonal.reserve(view.columns());
  }
  implementation::budget(work, check);
  const auto n = view.columns(), m = view.rows();
  // 未初始化数组避免先做一遍不可分批的隐式清零，再重复记一次初始化。
  std::unique_ptr<long double[]> merged(m ? new long double[m] : nullptr);
  std::unique_ptr<std::size_t[]> stamps(m ? new std::size_t[m] : nullptr);
  for (std::size_t row = 0; row < m; ++row) {
    implementation::coordinate_budget(row, work, check);
    implementation::increment(work.scratch_row_terms); merged[row] = 0;
    implementation::increment(work.scratch_row_terms); stamps[row] = 0;
  }
  std::vector<std::size_t> touched;
  touched.reserve(std::min(m, view.nonzeros()));
  std::vector<double> scales;
  scales.reserve(n);
  const auto starts = view.starts(), indices = view.row_indices();
  const auto values = view.values(), current_rows = view.current_r();
  for (std::size_t column = 0; column < n; ++column) {
    implementation::coordinate_budget(column, work, check);
    touched.clear();
    const std::size_t stamp = column + 1;
    for (std::size_t term = static_cast<std::size_t>(starts[column]);
        term < static_cast<std::size_t>(starts[column + 1]); ++term) {
      implementation::coordinate_budget(term, work, check);
      implementation::increment(work.merge_terms);
      const auto row = static_cast<std::size_t>(indices[term]);
      if (stamps[row] != stamp) {
        stamps[row] = stamp; merged[row] = 0; touched.push_back(row);
      }
      merged[row] = implementation::finite(merged[row] + static_cast<long double>(values[term]));
    }
    long double current_value = r_old;
    long double old_value = hit ? cache.old_diagonal[column] : static_cast<long double>(r_old);
    for (std::size_t position = 0; position < touched.size(); ++position) {
      implementation::coordinate_budget(position, work, check);
      implementation::increment(work.unique_terms);
      const auto row = touched[position];
      const long double square = implementation::finite(merged[row] * merged[row]);
      // 抵消为零的坐标仍记一项，并执行相同贡献流程，不把它当成未遍历。
      implementation::increment(work.current_terms);
      current_value = implementation::finite(current_value +
          implementation::contribution(square, current_rows[row]));
      if (!hit) {
        implementation::increment(work.old_terms);
        old_value = implementation::finite(old_value + implementation::contribution(square, old_rows[row]));
      }
    }
    if (!(old_value > 0) || !std::isfinite(old_value) ||
        !(current_value > 0) || !std::isfinite(current_value))
      throw Failure("Schur对角缩放要求新旧对角逐列有限正");
    implementation::extrema(old_value, work.old_min, work.old_max);
    implementation::extrema(current_value, work.current_min, work.current_max);
    if (!hit) pending.old_diagonal.push_back(old_value);
    implementation::increment(work.column_terms);
    // 分别开方后相除，避免先形成新旧对角比值而产生不必要的中间溢出。
    const long double scale = std::sqrt(old_value) / std::sqrt(current_value);
    if (!(scale > 0)) throw Failure("Schur对角缩放系数必须有限正");
    const double stored_scale = implementation::narrow(scale);
    scales.push_back(stored_scale);
    implementation::extrema(stored_scale, work.scale_min, work.scale_max);
  }
  if (!hit) implementation::increment(work.old_build_completed);
  implementation::budget(work, check);
  implementation::increment(work.prepare_completed);
  // vector的移动赋值不分配；末预算与全部计算通过后才替换旧缓存。
  if (!hit) { pending.valid = true; cache = std::move(pending); }
  return scales;
}

// rhs属于调用方尚未发布的scratch，预算或异常可留下部分缩放，不能复用作旧输入。
// 只对头n项作两侧合同缩放；旧direct接向量常量引用，避免复制完整右端。
template <class Direct>
std::vector<double> apply(std::span<const double> scales, std::vector<double>& uncommitted_rhs,
    Direct&& old_direct, Work& work, const Budget& check = {}) {
  if (scales.size() > uncommitted_rhs.size())
    throw std::invalid_argument("Schur对角缩放的右端短于原始变量维数");
  implementation::budget(work, check);
  implementation::increment(work.input_started);
  for (std::size_t column = 0; column < scales.size(); ++column) {
    implementation::coordinate_budget(column, work, check);
    implementation::increment(work.input_terms);
    if (!(scales[column] > 0) || !std::isfinite(scales[column]))
      throw Failure("Schur对角缩放的输入系数必须有限正");
    uncommitted_rhs[column] = implementation::narrow(
        static_cast<long double>(uncommitted_rhs[column]) * scales[column]);
  }
  implementation::budget(work, check);
  implementation::increment(work.input_completed);
  implementation::budget(work, check);
  implementation::increment(work.direct_started);
  std::vector<double> result = std::invoke(std::forward<Direct>(old_direct), std::as_const(uncommitted_rhs));
  implementation::increment(work.direct_completed);
  implementation::budget(work, check);
  if (result.size() != uncommitted_rhs.size())
    throw std::invalid_argument("Schur对角缩放的旧直接解维数不一致");
  implementation::increment(work.output_started);
  for (std::size_t column = 0; column < scales.size(); ++column) {
    implementation::coordinate_budget(column, work, check);
    implementation::increment(work.output_terms);
    result[column] = implementation::narrow(static_cast<long double>(result[column]) * scales[column]);
  }
  implementation::budget(work, check);
  implementation::increment(work.output_completed);
  return result;
}

}  // 命名空间 solver::detail::schur_diagonal_transport
