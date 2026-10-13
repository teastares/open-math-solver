/*
 * This file is part of QDLDL, a library for performing the LDL^T factorization
 * of a symmetric indefinite matrix.
 *
 * QDLDL is part of the OSQP project, and is available at https://github.com/osqp/qdldl.
 *
 * Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, The OSQP developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-ExternalRef: PACKAGE_MANAGER purl pkg:github/osqp/qdldl
 */

// 本文件修改自 QDLDL v0.1.9 的 QDLDL_factor，保留稀疏消元及工作区布局。
// 上游：https://github.com/osqp/qdldl/blob/v0.1.9/src/qdldl.c
// 修改：统一处理首个主元；在主元参与后续消元之前按已知符号局部正则化；
// 增加非有限数拒绝和修正计数。原许可证副本见 third_party/qdldl/LICENSE。
// 符号修正规则参考 Clarabel 的公开算法，未复制其 Rust 实现：
// https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/qdldl/qdldl.rs

#include "solver/detail/regularized_ldl.hpp"

#include <cmath>

namespace solver::detail {

RegularizedLdlResult factor_regularized_ldl(
    QDLDL_int n, const QDLDL_int* ap, const QDLDL_int* ai, const QDLDL_float* ax,
    QDLDL_int* lp, QDLDL_int* li, QDLDL_float* lx, QDLDL_float* diagonal,
    QDLDL_float* inverse_diagonal, const QDLDL_int* column_counts,
    const QDLDL_int* elimination_tree, QDLDL_bool* boolean_work,
    QDLDL_int* integer_work, QDLDL_float* float_work, const std::int8_t* signs,
    double threshold, double replacement) {
  RegularizedLdlResult result;
  if (n < 0 || !std::isfinite(threshold) || threshold < 0 ||
      !std::isfinite(replacement) || !(replacement > threshold)) return result;
  if (n == 0) { result.positive_pivots = 0; return result; }

  auto* row_indices = integer_work;
  auto* path = integer_work + n;
  auto* next_position = integer_work + 2 * n;
  lp[0] = 0;
  for (QDLDL_int i = 0; i < n; ++i) {
    lp[i + 1] = lp[i] + column_counts[i];
    boolean_work[i] = 0;
    float_work[i] = 0;
    diagonal[i] = 0;
    next_position[i] = lp[i];
  }
  QDLDL_int positives = 0;
  for (QDLDL_int k = 0; k < n; ++k) {
    QDLDL_int row_count = 0;
    for (QDLDL_int position = ap[k]; position < ap[k + 1]; ++position) {
      const QDLDL_int row = ai[position];
      if (row == k) {
        diagonal[k] = ax[position];
        continue;
      }
      float_work[row] = ax[position];
      if (!boolean_work[row]) {
        boolean_work[row] = 1;
        path[0] = row;
        QDLDL_int path_length = 1;
        QDLDL_int next = elimination_tree[row];
        while (next != -1 && next < k && !boolean_work[next]) {
          boolean_work[next] = 1;
          path[path_length++] = next;
          next = elimination_tree[next];
        }
        while (path_length > 0) row_indices[row_count++] = path[--path_length];
      }
    }
    // 按消元树得到的次序求当前行，所用前序主元已经完成局部修正。
    for (QDLDL_int i = row_count; i-- > 0;) {
      const QDLDL_int col = row_indices[i];
      const QDLDL_int destination = next_position[col];
      const double value = float_work[col];
      for (QDLDL_int j = lp[col]; j < destination; ++j) {
        float_work[li[j]] -= lx[j] * value;
      }
      li[destination] = k;
      lx[destination] = value * inverse_diagonal[col];
      diagonal[k] -= value * lx[destination];
      ++next_position[col];
      float_work[col] = 0;
      boolean_work[col] = 0;
    }
    // 非有限主元必须拒绝，不能把溢出伪装成可修正的符号错误。
    if (!std::isfinite(diagonal[k]) || (signs[k] != 1 && signs[k] != -1)) return result;
    if (signs[k] * diagonal[k] < threshold) {
      diagonal[k] = signs[k] * replacement;
      ++result.regularized_pivots;
    }
    if (diagonal[k] == 0) return result;
    inverse_diagonal[k] = 1 / diagonal[k];
    if (!std::isfinite(inverse_diagonal[k])) return result;
    if (diagonal[k] > 0) ++positives;
  }
  result.positive_pivots = positives;
  return result;
}

}  // 命名空间 solver::detail
