#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace solver::detail {

// 仅表示预先声明的结构预算；不能把分配失败、输入错误或时间中断当成该条件。
struct NormalCholeskyBudgetExceeded : std::length_error {
  using std::length_error::length_error;
};

struct NormalRefinementStatistics {
  std::int64_t residual_products = 0;
  std::int64_t refinement_steps = 0;
  std::int64_t refinement_acceptances = 0;
};

enum class NormalCholeskyKernel { eigen, cholmod_supernodal };

// 对排列后的 [B G^T; G -D] 建立正规方程；坐标标签为原始 +1、对偶 -1。
class NormalCholesky {
 public:
  NormalCholesky(std::size_t dimension,
                 std::span<const std::int64_t> upper_column_start,
                 std::span<const std::int64_t> upper_row_index,
                 std::span<const std::int8_t> coordinate_signs,
                 const std::function<void()>& budget_check = {}, bool multifrontal = false,
                 bool extended = false, int max_refinement_steps = 0,
                 NormalCholeskyKernel kernel = NormalCholeskyKernel::eigen);
  ~NormalCholesky();
  NormalCholesky(const NormalCholesky&) = delete;
  NormalCholesky& operator=(const NormalCholesky&) = delete;

  bool factor(std::span<const double> upper_values,
              const std::function<void()>& budget_check = {});
  // 仅完整回代结果有限且可表示时写回；分解失败后不能使用旧因子。
  bool solve(std::vector<double>& rhs, const std::function<void()>& budget_check = {}) const;
  // CHOLMOD 报告 xsize，即含超节点占位的实际存储项数，不等于数学非零数。
  std::int64_t factor_nonzeros() const;
  std::int64_t factor_dimension() const;
  std::int64_t regularized_pivots() const;
  // 累计本对象已启动的校正工作，含后来失败或中断的求解；重新分解不清零。
  NormalRefinementStatistics refinement_statistics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // 命名空间 solver::detail
