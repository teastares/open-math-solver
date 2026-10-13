#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <vector>

namespace solver::detail {

inline constexpr long double ipm_backward_error_tolerance =
    static_cast<long double>(std::numeric_limits<double>::epsilon()) / 2;

// 分块 forcing 的固定实验值；不由互补右端、模型身份或最终验算容差调节。
inline constexpr long double ipm_inexact_forcing = 0.1L;

// 沿用原十倍正则网格；负层级只扩展到不小于 double epsilon 的最小格点。
int ipm_min_regularization_level();
double ipm_regularization_at_level(int level);
// 返回缩放前的 (原始, 对偶) 外部均匀参数；分离模式借用 ε^(3/4)、ε^(1/2) 两个量级。
std::array<double, 2> ipm_regularization_pair(int level, bool separate);
// 已消元单行贡献须在 double 中保持有限且严格为正，才允许省掉原始均匀偏移。
bool ipm_has_positive_curvature(std::span<const long double> diagonal);
// 使用已接受预测方向的真实 K0 对偶缺陷；本次已失败的更低分解层级不重复试探。
// 返回值只是一次降正则搜索的起点，不授权放宽方向或最终解的验收。
int ipm_regularization_search_level(int preferred_level, int actual_level, double dual_error,
    std::span<const double> dual_residual, std::span<const double> primal_direction);

// 在完整 K0 的 (对偶方程, 原始方程) 两块上分别沿用 strict 的绝对/相对系数。
// 只收紧方向验收；split 是变量数，不是约化系统维数。空块仍有绝对门槛。
// 可显式传入较宽的相对系数；调用方仍必须与原全局 strict 条件取交集。
std::array<double, 2> ipm_kkt_block_tolerances(std::span<const double> rhs, std::size_t split,
                                           double relative = 1e-10);
bool ipm_kkt_blocks_satisfied(std::span<const double> residual, std::size_t split,
                            const std::array<double, 2>& tolerances);

// 静态列绝对和与两种累加顺序之间的舍入误差系数；无可靠界的列保留为无穷。
struct IpmInexactPrefilter {
  std::vector<long double> column_absolute_sums;
  std::vector<long double> rounding_factors;
};

// 内部接口：CSC 结构及重复项已由调用者验证和规范化，不在每次验收重复扫描验证。
IpmInexactPrefilter make_ipm_inexact_prefilter(
    std::span<const std::int64_t> starts, std::span<const double> values,
    const std::function<void()>& budget_check = {});

// transpose_product 是旧 K0 扫描的原始 long double 上块，未经减右端或 double 转换。
// 缓存必须来自本次同一个 dual_direction，且矩阵与静态预拒绝数据完全一致。
// true 仅表示保守界证实原 actual helper 必拒；false 必须继续原 helper，不授权接受。
bool reject_ipm_inexact_dual(const IpmInexactPrefilter& prefilter,
    std::span<const long double> transpose_product, std::span<const double> dual_residual,
    std::span<const double> dual_direction, const std::function<void()>& budget_check = {});

// 内部接口沿用下方 CSC 结构契约；direction 为完整的 (dx,dz)，slack_direction
// 必须是实际按互补方程恢复并舍入到 double 的 ds，等式行对应值为零。
// 分别检查 rp+Gdx+ds、rd+G^Tdz，不用舍入后的 H 代替实际松弛方向。
// 每块缺陷至多为该块当前残差范数的 0.1 倍；零块没有绝对或互补量地板。
bool accept_ipm_inexact_direction(
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> primal_residual,
    std::span<const double> dual_residual, std::span<const double> slack_direction,
    std::span<const double> direction, const std::function<void()>& budget_check = {});

// 内部不稳定接口：CSC G 已合并重复项并通过结构验证；H、右端和方向属于完整 K0。
// maximum_row_nonzeros 为 K0 每行非零数的上界，不含右端；不按约化矩阵计数。
// 返回分量后向误差；非有限分母或不可靠乘积使其无穷，不允许替代验收。
// 若原方程残差本身不能表示为有限 double，则沿用内核的数值失败异常。
long double assess_ipm_kkt_residual(
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> diagonal_h,
    std::span<const double> rhs, std::span<const double> point,
    std::size_t maximum_row_nonzeros, std::vector<double>& residual,
    const std::function<void()>& budget_check = {});

}  // 命名空间 solver::detail
