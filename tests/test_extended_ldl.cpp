#include "solver/detail/extended_ldl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

}  // 匿名命名空间

int main() {
  try {
    // 同一已排序稀疏结构，两种精度均不改变输入或正则项。
    const QDLDL_int ap[]{0, 1, 3}, ai[]{0, 0, 1};
    QDLDL_int work[6]{}, counts[2]{}, tree[2]{}, lp[3]{}, li[1]{};
    QDLDL_bool marks[2]{};
    require(QDLDL_etree(2, ap, ai, work, counts, tree) == 1, "两种精度必须共享同一符号结构");
    long double lx[1]{}, diagonal[2]{}, inverse[2]{}, floating_work[2]{};
    const long double values[]{2, 4, -3};
    require(solver::detail::solver_qdldl_extended_factor(2, ap, ai, values, lp, li, lx,
        diagonal, inverse, counts, tree, marks, work, floating_work) == 1,
        "扩展精度 LDL 应保留准定系统惯性");
    long double rhs[]{0, 11};
    solver::detail::solver_qdldl_extended_solve(2, lp, li, lx, inverse, rhs);
    require(std::abs(rhs[0] - 2) < 1e-12L && std::abs(rhs[1] + 1) < 1e-12L,
            "扩展精度回代应恢复解析解");
    long double second[]{8, -6};
    solver::detail::solver_qdldl_extended_solve(2, lp, li, lx, inverse, second);
    require(std::abs(second[0]) < 1e-12L && std::abs(second[1] - 2) < 1e-12L,
            "多右端必须可复用数值因子");

    if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
      const double small = std::ldexp(1.0, -54);
      const double double_values[]{small, 1, -1};
      double double_lx[1]{}, double_d[2]{}, double_inverse[2]{}, double_work[2]{}, double_rhs[]{1, 0};
      require(QDLDL_factor(2, ap, ai, double_values, lp, li, double_lx, double_d, double_inverse,
          counts, tree, marks, work, double_work) == 1, "舍入反例的 double 因子应成功产生");
      QDLDL_solve(2, lp, li, double_lx, double_inverse, double_rhs);
      const long double extended_values[]{small, 1, -1};
      require(solver::detail::solver_qdldl_extended_factor(2, ap, ai, extended_values, lp, li, lx,
          diagonal, inverse, counts, tree, marks, work, floating_work) == 1,
          "舍入反例的扩展精度因子应成功产生");
      long double extended_rhs[]{1, 0};
      solver::detail::solver_qdldl_extended_solve(2, lp, li, lx, inverse, extended_rhs);
      const long double double_residual = std::abs(static_cast<long double>(double_rhs[0]) - double_rhs[1]);
      const long double extended_residual = std::max(std::abs(small * extended_rhs[0] + extended_rhs[1] - 1),
          std::abs(extended_rhs[0] - extended_rhs[1]));
      require(double_residual > 0.5L && extended_residual < 1e-12L,
              "新增构建必须实际使用更高精度，不能只改变后端名称");
    }
    std::cout << "扩展精度 LDL 检查通过；double 尾数位 " << std::numeric_limits<double>::digits
              << "，long double 尾数位 " << std::numeric_limits<long double>::digits << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "扩展精度 LDL 检查失败：" << error.what() << '\n';
    return 1;
  }
}
