#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace solver::detail::cholmod_aat {

using Index = std::int64_t;

struct TransposePattern {
  std::vector<Index> starts, rows;
  std::vector<std::uint32_t> source_slots;
};

// 按原列顺序散写，使每个转置列与 CHOLMOD 的自然转置顺序相同。
// 仅记录结构和原 CSC 槽，数值分解时从该槽取值并从该槽开始扫描列后缀。
inline TransposePattern build_transpose_pattern(std::size_t nrow, std::size_t ncol,
    std::span<const Index> starts, std::span<const Index> rows, std::size_t structural_budget,
    const std::function<void()>& budget_check) {
  const auto check_coordinate = [&](std::size_t coordinate) {
    if ((coordinate & 4095) == 0 && budget_check) budget_check();
  };
  check_coordinate(0);
  if (nrow > structural_budget || ncol > structural_budget || rows.size() > structural_budget ||
      rows.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::length_error("CHOLMOD 矩形转置超过结构项预算");
  }
  const Index nonzeros = static_cast<Index>(rows.size());
  if (starts.size() != ncol + 1 || starts.front() != 0 || starts.back() != nonzeros) {
    throw std::invalid_argument("CHOLMOD 矩形转置 CSC 维数或端点无效");
  }
  TransposePattern result;
  result.starts.assign(nrow + 1, 0);
  for (std::size_t col = 0; col < ncol; ++col) {
    check_coordinate(col);
    const Index begin = starts[col], end = starts[col + 1];
    if (begin < 0 || end < begin || end > nonzeros) {
      throw std::invalid_argument("CHOLMOD 矩形转置 CSC 列指针越界");
    }
    Index previous = -1;
    for (Index source = begin; source < end; ++source) {
      check_coordinate(static_cast<std::size_t>(source));
      const Index row = rows[static_cast<std::size_t>(source)];
      if (row <= previous || static_cast<std::size_t>(row) >= nrow) {
        throw std::invalid_argument("CHOLMOD 矩形转置 CSC 行索引必须严格有序且不越界");
      }
      previous = row;
      ++result.starts[static_cast<std::size_t>(row) + 1];
    }
  }
  for (std::size_t row = 0; row < nrow; ++row) {
    check_coordinate(row);
    result.starts[row + 1] += result.starts[row];
  }
  result.rows.resize(rows.size());
  result.source_slots.resize(rows.size());
  std::vector<Index> next(result.starts.begin(), result.starts.end() - 1);
  for (std::size_t col = 0; col < ncol; ++col) {
    check_coordinate(col);
    for (Index source = starts[col]; source < starts[col + 1]; ++source) {
      check_coordinate(static_cast<std::size_t>(source));
      const auto row = static_cast<std::size_t>(rows[static_cast<std::size_t>(source)]);
      const auto destination = static_cast<std::size_t>(next[row]++);
      result.rows[destination] = static_cast<Index>(col);
      result.source_slots[destination] = static_cast<std::uint32_t>(source);
    }
  }
  check_coordinate(0);
  return result;
}

}  // 命名空间 solver::detail::cholmod_aat
