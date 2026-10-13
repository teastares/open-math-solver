#pragma once

#include <qdldl.h>

namespace solver::detail {

// 对应同一 QDLDL v0.1.9 源码的独立 long double 构建，索引和布尔类型保持一致。
// 调用方负责沿用 QDLDL 的结构及工作区前置条件，并检查因子和回代结果。
extern "C" QDLDL_int solver_qdldl_extended_factor(
    QDLDL_int n, const QDLDL_int* ap, const QDLDL_int* ai, const long double* ax,
    QDLDL_int* lp, QDLDL_int* li, long double* lx, long double* diagonal,
    long double* inverse_diagonal, const QDLDL_int* column_counts,
    const QDLDL_int* elimination_tree, QDLDL_bool* boolean_work,
    QDLDL_int* integer_work, long double* float_work);

extern "C" void solver_qdldl_extended_solve(
    QDLDL_int n, const QDLDL_int* lp, const QDLDL_int* li, const long double* lx,
    const long double* inverse_diagonal, long double* rhs);

}  // 命名空间 solver::detail
