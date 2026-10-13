#pragma once

#include <cstdint>
#include <string>

namespace solver {

struct IpmOptions {
  // homogeneous 是独立的 LP 齐次嵌入实验；默认继续使用原预测校正流程。
  std::string method = "mehrotra";
  std::int64_t max_iterations = 200;
  int scaling_passes = 5;
  bool eliminate_singletons = false;
  bool dynamic_regularization = false;
  bool equilibrate_kkt = false;
  // blockwise 收紧 Newton 两块缺陷；blockwise_weighted 还将相同尺度用于 Krylov 搜索。
  // blockwise_forcing 使用较宽的 RHS 块相对门槛，同时保留全局 strict 检查。
  // blockwise_forcing_weighted 还按全局与块门槛的交集缩放 Krylov 搜索。
  // adaptive_regularization 沿用 strict 验收，按已有预测方向缺陷试探更小正则量。
  // primal_dual_regularization 沿用 strict 验收，以不同机器精度量级构造两块外部正则。
  // curvature_regularization 利用已有正曲率，先试一次零原始偏移，失败后沿用旧网格。
  // full_newton 在未消元的三块 Newton 方程上做有界加权 FGMRES，初始化仍严格。
  // bounded_newton 仅保留最多三次改进中的最佳有限方向，内层未达标仍交由外层验算。
  // homogeneous_scaled 仅用于齐次法：正能量预条件与最多三次完整五块改进，初始化仍严格。
  std::string linear_accuracy = "strict";  // 另有 backward、inexact 实验规则。
  int central_corrections = 0;
  // 初始化/预测校正组失败后的重分解预算；另支持完整 Newton 方程模式。
  int direction_retries = 0;
  // 同一 K0 内跨右端和重启回收物理方向；0 关闭，正值仅用于 strict 或 blockwise。
  int krylov_history_size = 0;
  // normal_extended 保留 Schur 装配、分解、回代和恢复的 long double，结构超限回退增广。
  // normal_refined 复用 double 因子，在 Schur 层做有界扩展精度残差校正。
  // normal_cholmod 保持原 Gram 装配和 AMD，仅换用 CHOLMOD 超节点分解；结构超限回退增广。
  std::string linear_solver = "qdldl";
  bool inexact_prefilter = false;
  bool eliminate_fixed_variables = false;
  // 将可精确转换的单非零输入行与变量界归并，保留原行对偶的恢复来源。
  bool presolve_singleton_rows = false;
  // 仅消除本次单行归并新产生的固定列，并恢复定义该固定点的原行对偶。
  bool eliminate_presolved_fixed_variables = false;
};

}  // 命名空间 solver
