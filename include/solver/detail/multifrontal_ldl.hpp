#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace solver::detail {

// 独立的单线程 FactorHiGHS 适配器；只接受含完整对角的上三角 CSC。
class MultifrontalLdl {
 public:
  MultifrontalLdl(std::size_t dimension, std::span<const std::int64_t> starts,
                 std::span<const std::int64_t> rows, std::span<const std::int8_t> signs,
                 const std::function<void()>& budget_check = {});
  ~MultifrontalLdl();
  MultifrontalLdl(const MultifrontalLdl&) = delete;
  MultifrontalLdl& operator=(const MultifrontalLdl&) = delete;
  bool factor(std::span<const double> values, const std::function<void()>& budget_check = {});
  bool solve(std::vector<double>& rhs, const std::function<void()>& budget_check = {}) const;
  std::int64_t factor_nonzeros() const;
  std::int64_t regularized_pivots() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // 命名空间 solver::detail
