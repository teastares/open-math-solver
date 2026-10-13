#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace solver::detail {

// 完整方向的顺序为 (dx,dz,ds_I)，方程顺序为 (对偶,原始,互补)。
// 等式行没有松弛方向和互补方程，不能通过任意等式乘子作除法。
struct IpmFullNewtonLayout {
  std::size_t columns = 0;
  std::size_t rows = 0;
  std::vector<std::size_t> inequality_rows;
  std::size_t dimension() const { return columns + rows + inequality_rows.size(); }
};

IpmFullNewtonLayout make_ipm_full_newton_layout(std::size_t columns,
    const std::vector<bool>& inequality);

// 内部结构契约：CSC 已规范化且行下标有效，layout 来自上方构造器。
// 每次乘积独立累加真实 J，不用舍入后的 H=s/z 重建原始块。
void ipm_full_newton_product(const IpmFullNewtonLayout& layout,
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> slack, std::span<const double> dual,
    std::span<const double> direction, std::vector<long double>& product,
    const std::function<void()>& budget_check = {});

std::vector<double> ipm_full_newton_reduced_rhs(const IpmFullNewtonLayout& layout,
    std::span<const double> dual, std::span<const double> rhs,
    const std::function<void()>& budget_check = {});
std::vector<double> ipm_full_newton_recover(const IpmFullNewtonLayout& layout,
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> rhs, std::span<const double> reduced_direction,
    const std::function<void()>& budget_check = {});

std::array<double, 3> ipm_full_newton_tolerances(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs);

struct IpmFullNewtonWork {
  // 不含初始预条件解；只记录已完成的额外预条件解，与旧改进统计口径一致。
  std::int64_t refinement_steps = 0;
  std::int64_t krylov_steps = 0;
};

using IpmFullNewtonPreconditioner =
    std::function<std::vector<double>(std::span<const double>)>;
using IpmFullNewtonProduct =
    std::function<void(std::span<const double>, std::vector<long double>&)>;

// 固定三块尺度；最多两次改进和三轮 min(32,n+m) 列 FGMRES。
// 回调必须分别使用同一个物理 J 和当前 KKT 因子，不得在预条件器内嵌套旧 Krylov。
// 每个返回值均通过独立的长双精度物理残差检查；估计残差只触发候选构造。
std::vector<double> solve_ipm_full_newton(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs, const IpmFullNewtonPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmFullNewtonWork& work,
    const std::function<void()>& budget_check = {});

struct IpmBoundedNewtonInfo {
  // 返回方向的真实长双精度残差，按原始 RHS 的三个固定门槛归一化后取无穷范数。
  long double measure = std::numeric_limits<long double>::infinity();
  bool tolerance_satisfied = false;
};

// 独立实验：初始预条件解加最多三次改进，不调用 Krylov；仅采用真实 measure 严格下降的候选。
// 有限方向未达内层门槛时仍返回，必须由调用方保留原模型最终验算并记录非精确返回。
std::vector<double> solve_ipm_bounded_newton(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs, const IpmFullNewtonPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmFullNewtonWork& work, IpmBoundedNewtonInfo& info,
    const std::function<void()>& budget_check = {});

}  // 命名空间 solver::detail
