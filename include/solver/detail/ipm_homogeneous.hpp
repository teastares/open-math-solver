#pragma once

#include "solver/detail/ipm_full_newton.hpp"

#include <functional>
#include <span>

namespace solver::detail {

struct IpmHomogeneousScalarDirection {
  double tau = 0;
  double kappa = 0;
};

// u、v 的顺序均为 (dx,dz)；v 解常数右端 (-c,b)。
// gap_rhs 是间隙方程要求的导数 -theta*rg，complementarity_rhs 是 tau*kappa 的目标导数。
// 仅接受严格正且有限的标量分母，不用人为下限替代退化系统。
IpmHomogeneousScalarDirection ipm_homogeneous_scalar_direction(
    std::span<const double> costs, std::span<const double> rhs,
    std::span<const double> u, std::span<const double> v, double tau, double kappa,
    long double gap_rhs, long double complementarity_rhs,
    const std::function<void()>& budget_check = {});

// 将原始、对偶锥的公共边界再与 tau、kappa 的边界及 1 取交；校正步的安全比例由调用方应用。
double ipm_homogeneous_step_limit(double cone_limit, double tau, double kappa,
    const IpmHomogeneousScalarDirection& direction,
    const std::function<void()>& budget_check = {});

// 物理方向顺序为 (dx,dz,ds_I,dtau,dkappa)，五块方程顺序为
// (对偶,原始,间隙/d,互补/lambda,tau*kappa/lambda_tau)。等式 z 自由且没有互补块。
struct IpmHomogeneousScaling {
  IpmFullNewtonLayout layout;
  std::vector<long double> weights, lambda;
  long double d = 1, lambda_tau = 1;
  std::size_t dimension() const { return layout.dimension() + 2; }
};

IpmHomogeneousScaling make_ipm_homogeneous_scaling(std::size_t columns,
    const std::vector<bool>& inequality, std::span<const double> slack,
    std::span<const double> dual, double tau, double kappa,
    const std::function<void()>& budget_check = {});

std::vector<double> ipm_homogeneous_scaled_constant_rhs(const IpmHomogeneousScaling& scaling,
    std::span<const double> costs, std::span<const double> bounds,
    const std::function<void()>& budget_check = {});
long double ipm_homogeneous_scaled_denominator(const IpmHomogeneousScaling& scaling,
    std::span<const double> constant, const std::function<void()>& budget_check = {});
std::vector<double> ipm_homogeneous_scaled_reduced_rhs(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const std::function<void()>& budget_check = {});
std::vector<double> ipm_homogeneous_scaled_recover(const IpmHomogeneousScaling& scaling,
    std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> rhs, std::span<const double> reduced,
    std::span<const double> constant, long double denominator,
    const std::function<void()>& budget_check = {});

// 对实际交付的物理 double 方向，用原 S/Z/tau/kappa 独立形成五块乘积；每次恰好一遍 CSC。
// 不以舍入后的 W 或 H 代替原系数，不在验算后重算 ds。
// scaling 应由上述构造器生成，CSC 的起点与行索引沿用已校验 ConeModel 的内部契约。
void ipm_homogeneous_scaled_product(const IpmHomogeneousScaling& scaling,
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> slack, std::span<const double> dual, double tau, double kappa,
    std::span<const double> direction, std::vector<long double>& product,
    const std::function<void()>& budget_check = {});

std::array<double, 5> ipm_homogeneous_scaled_tolerances(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const std::function<void()>& budget_check = {});

using IpmHomogeneousScaledPreconditioner =
    std::function<std::vector<double>(std::span<const double>, bool refinement)>;

// 五块门槛各为 1e-12+1e-10*原始 RHS 块无穷范数；整个方向期间固定。
// 初次直接预条件加至多三次完整 IR，仅采用真实归一残差严格下降的有限候选；不调用 Krylov。
// bool 告知回调是否为 IR：调用方应在实际启动 direct 时累计方向次数，IR 另计改进次数。
// multiply 的实际启动另计乘积次数；有限未达标返回另计 inexact，不代表 LP 已达标。
// 复用的 info.measure 在本接口表示五块归一残差的最大值；改进次数属于总 refinement_steps 的子集。
// 回调异常和预算始终传播，info 只在正常返回前提交；初始非有限失败，可选候选非有限保留最佳。
std::vector<double> solve_ipm_homogeneous_scaled(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const IpmHomogeneousScaledPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmBoundedNewtonInfo& info,
    const std::function<void()>& budget_check = {});

}  // 命名空间 solver::detail
