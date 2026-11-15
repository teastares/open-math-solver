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

enum class NormalCholeskyKernel { eigen, cholmod_supernodal, cholmod_supernodal_aat };

// 外部右端视图；扩展前缀先舍入为 double，再按 Normal 输入坐标的 scaling 缩放。
struct NormalSolveInput {
  std::span<const double> values;
  std::span<const long double> extended_prefix;
  std::span<const double> scaling;
};

// 对排列后的 [B G^T; G -D] 建立正规方程；坐标标签为原始 +1、对偶 -1。
class NormalCholesky {
 public:
  NormalCholesky(std::size_t dimension,
                 std::span<const std::int64_t> upper_column_start,
                 std::span<const std::int64_t> upper_row_index,
                 std::span<const std::int8_t> coordinate_signs,
                 const std::function<void()>& budget_check = {}, bool multifrontal = false,
                 bool extended = false, int max_refinement_steps = 0,
                 NormalCholeskyKernel kernel = NormalCholeskyKernel::eigen,
                 std::span<const std::int64_t> external_coordinates = {});
  ~NormalCholesky();
  NormalCholesky(const NormalCholesky&) = delete;
  NormalCholesky& operator=(const NormalCholesky&) = delete;

  bool factor(std::span<const double> upper_values,
              const std::function<void()>& budget_check = {});
  // 内部尚无等长数值时复制且不改输入；否则交换，调用方取得此前的完整数值 scratch。
  // 交换后数值失败、预算中断或分配异常均不回滚，失败因子不可用；重试前须完整覆盖输入。
  // 调用及预算回调期间不得修改输入、其存储别名或本对象。
  // 指向已交出存储的旧视图不会跟随调用方更新，不得跨后续分解调用继续使用。
  bool factor_exchange(std::vector<double>& upper_values,
                       const std::function<void()>& budget_check = {});
  // 仅完整回代结果有限且可表示时写回；分解失败后不能使用旧因子。
  bool solve(std::vector<double>& rhs, const std::function<void()>& budget_check = {}) const;
  // 使用构造时的外部坐标映射；空映射等同恒等。输出仅供尚未提交的临时结果。
  // values 与 output 须等长，不能与任何输入视图重叠；失败或预算中断可保留部分输出。
  // 未映射的输出位置保持原值。旧 solve 始终使用原 Normal 坐标，不受外部映射影响。
  bool solve_into(const NormalSolveInput& input, std::span<double> output,
                  const std::function<void()>& budget_check = {}) const;
  // CHOLMOD 报告 xsize，即含超节点占位的实际存储项数，不等于数学非零数。
  std::int64_t factor_nonzeros() const;
  std::int64_t factor_dimension() const;
  std::int64_t regularized_pivots() const;
  // 累计本对象已启动的校正工作，含后来失败或中断的求解；重新分解不清零。
  NormalRefinementStatistics refinement_statistics() const;

 private:
  struct Impl;
  bool factor_impl(std::span<const double> upper_values, std::vector<double>* exchange,
                   const std::function<void()>& budget_check);
  std::unique_ptr<Impl> impl_;
};

}  // 命名空间 solver::detail
