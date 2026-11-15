#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace solver::detail {

struct NormalGramPattern {
  std::vector<std::int64_t> starts;
  std::vector<std::int64_t> rows;
};

// 输入列已按保留行升序排列。调用方先核对原 Gram 贡献数和输入容量。
template <class Columns>
NormalGramPattern build_normal_gram_pattern(std::size_t row_count, const Columns& columns,
    const std::function<void()>& budget_check) {
  const auto check = [&] { if (budget_check) budget_check(); };
  check();
  if (row_count >= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::length_error("正规方程结构行数无法表示");
  }
  if (row_count == 0 && columns.empty()) return {{0}, {}};

  struct Prefix {
    std::size_t column;
    std::size_t length;
  };
  std::vector<std::size_t> prefix_starts(row_count + 1, 0);
  for (const auto& column : columns) {
    check();
    std::int64_t previous = -1;
    for (const auto& entry : column) {
      const auto row = entry.row;
      if (row <= previous || static_cast<std::size_t>(row) >= row_count) {
        throw std::invalid_argument("正规方程结构行必须严格升序且在范围内");
      }
      ++prefix_starts[static_cast<std::size_t>(row) + 1];
      previous = row;
    }
  }
  for (std::size_t row = 0; row < row_count; ++row) {
    if ((row & 4095) == 0) check();
    prefix_starts[row + 1] += prefix_starts[row];
  }
  std::vector<Prefix> prefixes(prefix_starts.back());
  {
    auto scan = prefix_starts;
    for (std::size_t col = 0; col < columns.size(); ++col) {
      check();
      const auto& column = columns[col];
      for (std::size_t entry = 0; entry < column.size(); ++entry) {
        const auto row = static_cast<std::size_t>(column[entry].row);
        prefixes[scan[row]++] = {col, entry + 1};
      }
    }
  }

  NormalGramPattern pattern;
  pattern.starts.reserve(row_count + 1);
  pattern.rows.reserve(row_count + prefixes.size());
  std::vector<std::size_t> seen(row_count, row_count);
  for (std::size_t col = 0; col < row_count; ++col) {
    check();
    const auto begin = pattern.rows.size();
    pattern.starts.push_back(static_cast<std::int64_t>(begin));
    pattern.rows.push_back(static_cast<std::int64_t>(col));
    seen[col] = col;
    for (std::size_t index = prefix_starts[col]; index < prefix_starts[col + 1]; ++index) {
      check();
      const auto [source, length] = prefixes[index];
      const auto& column = columns[source];
      for (std::size_t left = 0; left < length; ++left) {
        const auto row = static_cast<std::size_t>(column[left].row);
        if (seen[row] != col) {
          pattern.rows.push_back(static_cast<std::int64_t>(row));
          seen[row] = col;
        }
      }
    }
    // 保持 AMD 接收与原 Triplets 折叠一致的规范上三角。
    std::sort(pattern.rows.begin() + static_cast<std::ptrdiff_t>(begin), pattern.rows.end());
  }
  check();
  pattern.starts.push_back(static_cast<std::int64_t>(pattern.rows.size()));
  return pattern;
}

}  // 命名空间 solver::detail

