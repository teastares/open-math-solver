#pragma once

#include <qdldl.h>

#include <cstdint>

namespace solver::detail {

struct RegularizedLdlResult {
  // 负值表示数值分解失败；修正计数仍包含失败前已经处理的主元。
  QDLDL_int positive_pivots = -1;
  std::int64_t regularized_pivots = 0;
};

// 输入结构和工作区须满足 QDLDL_factor 的前置条件；符号按分解后的排列给出。
// 默认数值来自 Clarabel 的公开统一配置，独立于模型名称或基准结果：
// https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/implementations/default/settings.rs
RegularizedLdlResult factor_regularized_ldl(
    QDLDL_int n, const QDLDL_int* ap, const QDLDL_int* ai, const QDLDL_float* ax,
    QDLDL_int* lp, QDLDL_int* li, QDLDL_float* lx, QDLDL_float* diagonal,
    QDLDL_float* inverse_diagonal, const QDLDL_int* column_counts,
    const QDLDL_int* elimination_tree, QDLDL_bool* boolean_work,
    QDLDL_int* integer_work, QDLDL_float* float_work, const std::int8_t* signs,
    double threshold = 1e-13, double replacement = 2e-7);

}  // 命名空间 solver::detail
