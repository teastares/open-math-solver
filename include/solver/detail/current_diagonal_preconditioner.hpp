#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace solver::detail::current_diagonal {

using Index = std::int64_t;
using Budget = std::function<void()>;
inline constexpr std::size_t budget_batch = 4096;
struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

// 一个探测组共享本账本；每次应用只追加实际前缀，不清空先前费用。
struct Work {
  std::size_t columns = 0, rows = 0, nonzeros = 0;
  Index validation_started = 0, validation_completed = 0, validation_terms = 0;
  Index applications_started = 0, applications_completed = 0;
  Index transpose_started = 0, transpose_completed = 0;
  Index direct_started = 0, direct_completed = 0;
  Index product_started = 0, product_completed = 0;
  Index transpose_terms = 0, product_terms = 0, row_divisions = 0, budget_checks = 0;
};

namespace implementation {

inline void increment(Index& value) {
  if (value < 0 || value == std::numeric_limits<Index>::max())
    throw std::invalid_argument("当前对角预条件器的费用计数溢出");
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
  if (!std::isfinite(value)) throw Failure("当前对角预条件器的数值非有限");
  return value;
}
inline double narrow(long double value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
    throw Failure("当前对角预条件器的数值不能表示为double");
  return result;
}

}  // 命名空间 implementation

// 不拥有也不修改CSC或当前R；调用方须使引用在整组探测期间有效且保持不变。
// 同一探测只构造一次，重复行与显式零值均合法，数值应用仍遍历每个非零存储项。
class ValidatedView {
 public:
  ValidatedView(std::size_t columns, std::size_t rows,
      std::span<const Index> starts, std::span<const Index> row_indices,
      std::span<const double> values, std::span<const double> current_r,
      Work& work, const Budget& check = {})
      : columns_(columns), rows_(rows), starts_(starts), row_indices_(row_indices),
        values_(values), current_r_(current_r) {
    if (work.validation_started != 0 || work.validation_completed != 0)
      throw std::invalid_argument("当前对角预条件器的探测账本已执行验证");
    work.columns = columns; work.rows = rows; work.nonzeros = values.size();
    implementation::budget(work, check);
    implementation::increment(work.validation_started);
    const auto size_limit = std::numeric_limits<std::size_t>::max();
    if (columns == size_limit || columns > size_limit - rows ||
        starts.size() != columns + 1 || row_indices.size() != values.size() ||
        current_r.size() != rows ||
        values.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
        columns + rows > std::vector<double>{}.max_size() ||
        columns > std::vector<long double>{}.max_size() ||
        rows > std::vector<long double>{}.max_size())
      throw std::invalid_argument("当前对角预条件器的维数或CSC长度无效");
    Index previous = 0;
    for (std::size_t column = 0; column < starts.size(); ++column) {
      implementation::coordinate_budget(column, work, check);
      implementation::increment(work.validation_terms);
      const Index position = starts[column];
      if (position < 0 || static_cast<std::size_t>(position) > values.size() ||
          (column == 0 && position != 0) || position < previous ||
          (column == columns && static_cast<std::size_t>(position) != values.size()))
        throw std::invalid_argument("当前对角预条件器的CSC列指针无效");
      previous = position;
    }
    for (std::size_t row = 0; row < rows; ++row) {
      implementation::coordinate_budget(row, work, check);
      implementation::increment(work.validation_terms);
      if (!(current_r[row] > 0) || !std::isfinite(current_r[row]))
        throw Failure("当前对角预条件器要求当前R有限正");
    }
    for (std::size_t term = 0; term < values.size(); ++term) {
      implementation::coordinate_budget(term, work, check);
      implementation::increment(work.validation_terms);
      if (row_indices[term] < 0 || static_cast<std::size_t>(row_indices[term]) >= rows)
        throw std::invalid_argument("当前对角预条件器的CSC行号无效");
      if (!std::isfinite(values[term]))
        throw Failure("当前对角预条件器的CSC数值非有限");
    }
    implementation::budget(work, check);
    implementation::increment(work.validation_completed);
  }

  std::size_t columns() const noexcept { return columns_; }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t nonzeros() const noexcept { return values_.size(); }
  std::size_t dimension() const noexcept { return columns_ + rows_; }
  std::span<const Index> starts() const noexcept { return starts_; }
  std::span<const Index> row_indices() const noexcept { return row_indices_; }
  std::span<const double> values() const noexcept { return values_; }
  std::span<const double> current_r() const noexcept { return current_r_; }

 private:
  std::size_t columns_, rows_;
  std::span<const Index> starts_, row_indices_;
  std::span<const double> values_, current_r_;
};

// 旧直接求解只作用于[q;0]，保留其完整坐标中的x，再用当前R恢复z。
// direct可接向量引用或span；内部完整右端属于尚未提交的scratch，允许回调原位缩放。
template <class Direct>
std::vector<double> apply(const ValidatedView& view, std::span<const double> rhs,
    Direct&& direct, Work& work, const Budget& check = {}) {
  if (rhs.size() != view.dimension() || work.validation_started != 1 ||
      work.validation_completed != 1 || work.columns != view.columns() ||
      work.rows != view.rows() || work.nonzeros != view.nonzeros())
    throw std::invalid_argument("当前对角预条件器的右端或验证账本不一致");
  implementation::budget(work, check);
  implementation::increment(work.applications_started);
  const auto n = view.columns(), m = view.rows();
  const auto starts = view.starts(), indices = view.row_indices();
  const auto values = view.values(), current_r = view.current_r();
  std::vector<double> reduced_rhs(view.dimension(), 0);
  {
    std::vector<long double> q(n), divided_rhs(m);
    for (std::size_t column = 0; column < n; ++column) {
      implementation::coordinate_budget(column, work, check);
      q[column] = implementation::finite(rhs[column]);
    }
    for (std::size_t row = 0; row < m; ++row) {
      implementation::coordinate_budget(row, work, check);
      const long double value = implementation::finite(rhs[n + row]);
      implementation::increment(work.row_divisions);
      divided_rhs[row] = implementation::finite(value / static_cast<long double>(current_r[row]));
    }
    implementation::budget(work, check);
    implementation::increment(work.transpose_started);
    for (std::size_t column = 0; column < n; ++column) {
      implementation::coordinate_budget(column, work, check);
      for (std::size_t term = static_cast<std::size_t>(starts[column]);
          term < static_cast<std::size_t>(starts[column + 1]); ++term) {
        implementation::coordinate_budget(term, work, check);
        implementation::increment(work.transpose_terms);
        q[column] = implementation::finite(q[column] +
            static_cast<long double>(values[term]) * divided_rhs[static_cast<std::size_t>(indices[term])]);
      }
      reduced_rhs[column] = implementation::narrow(q[column]);
    }
    implementation::budget(work, check);
    implementation::increment(work.transpose_completed);
  }
  implementation::budget(work, check);
  implementation::increment(work.direct_started);
  // 以向量常量引用调用，避免生产旧direct接口再复制一次完整右端。
  std::vector<double> result = std::invoke(std::forward<Direct>(direct), reduced_rhs);
  implementation::increment(work.direct_completed);
  implementation::budget(work, check);
  if (result.size() != view.dimension())
    throw std::invalid_argument("当前对角预条件器的旧直接解维数不一致");
  for (std::size_t index = 0; index < result.size(); ++index) {
    implementation::coordinate_budget(index, work, check);
    implementation::finite(result[index]);
  }
  implementation::budget(work, check);
  implementation::increment(work.product_started);
  std::vector<long double> image(m, 0);
  for (std::size_t column = 0; column < n; ++column) {
    implementation::coordinate_budget(column, work, check);
    for (std::size_t term = static_cast<std::size_t>(starts[column]);
        term < static_cast<std::size_t>(starts[column + 1]); ++term) {
      implementation::coordinate_budget(term, work, check);
      implementation::increment(work.product_terms);
      const auto row = static_cast<std::size_t>(indices[term]);
      image[row] = implementation::finite(image[row] +
          static_cast<long double>(values[term]) * result[column]);
    }
  }
  implementation::budget(work, check);
  implementation::increment(work.product_completed);
  for (std::size_t row = 0; row < m; ++row) {
    implementation::coordinate_budget(row, work, check);
    const long double numerator = implementation::finite(image[row] - static_cast<long double>(rhs[n + row]));
    implementation::increment(work.row_divisions);
    result[n + row] = implementation::narrow(numerator / static_cast<long double>(current_r[row]));
  }
  implementation::budget(work, check);
  implementation::increment(work.applications_completed);
  return result;
}

}  // 命名空间 solver::detail::current_diagonal
