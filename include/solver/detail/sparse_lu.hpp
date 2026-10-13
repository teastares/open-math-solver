#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace solver::detail {

// 固定上三角模式的对称稀疏 LU；排序由调用方在构造前完成。
class SparseLu {
 public:
  SparseLu(std::size_t dimension,
           std::span<const std::int64_t> upper_column_start,
           std::span<const std::int64_t> upper_row_index);
  ~SparseLu();
  SparseLu(const SparseLu&) = delete;
  SparseLu& operator=(const SparseLu&) = delete;

  bool factor(std::span<const double> upper_values);
  // 仅成功时写回；未分解、维数错误或非有限结果均返回 false。
  bool solve(std::vector<double>& rhs) const;
  std::int64_t factor_nonzeros() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // 命名空间 solver::detail
