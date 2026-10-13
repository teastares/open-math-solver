#include "solver/detail/sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Index = std::int64_t;
using solver::detail::SparseLu;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> multiply(const std::vector<Index>& starts, const std::vector<Index>& rows,
                             const std::vector<double>& values, const std::vector<double>& x) {
  std::vector<long double> product(x.size(), 0);
  for (std::size_t column = 0; column < x.size(); ++column) {
    for (Index entry = starts[column]; entry < starts[column + 1]; ++entry) {
      const auto position = static_cast<std::size_t>(entry);
      const auto row = static_cast<std::size_t>(rows[position]);
      product[row] += static_cast<long double>(values[position]) * x[column];
      if (row != column) product[column] += static_cast<long double>(values[position]) * x[row];
    }
  }
  return {product.begin(), product.end()};
}

void check_solution(SparseLu& lu, const std::vector<Index>& starts, const std::vector<Index>& rows,
                    const std::vector<double>& values, const std::vector<double>& expected) {
  const auto original = multiply(starts, rows, values, expected);
  auto actual = original;
  require(lu.factor(values), "非奇异有限矩阵应成功分解");
  require(lu.solve(actual), "已分解系统应成功回代");
  const auto residual = multiply(starts, rows, values, actual);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::abs(actual[i] - expected[i]) <= 1e-11 * std::max(1.0, std::abs(expected[i])),
            "回代必须恢复独立构造的解");
    require(std::abs(residual[i] - original[i]) <= 1e-11 * std::max(1.0, std::abs(original[i])),
            "完整对称矩阵上的独立残差不合格");
  }
}

void require_bad_pattern(std::size_t dimension, const std::vector<Index>& starts,
                         const std::vector<Index>& rows) {
  bool rejected = false;
  try {
    SparseLu invalid(dimension, starts, rows);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "无效上三角 CSC 必须在访问内存前拒绝");
}

}  // 匿名命名空间

int main() {
  try {
    const std::vector<Index> starts{0, 1, 3, 5};
    const std::vector<Index> rows{0, 0, 1, 1, 2};
    SparseLu lu(3, starts, rows);
    std::vector<double> rhs{1, 2, 3};
    require(!lu.solve(rhs) && rhs == std::vector<double>({1, 2, 3}), "未分解不能修改右端项");
    check_solution(lu, starts, rows, {4, 1, -2, 2, 5}, {2, -1, 3});
    require(lu.factor_nonzeros() >= 6, "L 与 U 的统计应包含两组对角");
    // 同一模式多次更新数值，含结构位置暂为零的情况。
    check_solution(lu, starts, rows, {2, 0, 3, 0, -4}, {-1, 2, 0.5});
    check_solution(lu, starts, rows, {5, -2, -3, 1, 7}, {3, 0, -2});
    require(!lu.factor(std::vector<double>(5, 0)), "奇异矩阵必须拒绝");
    rhs = {1, 2, 3};
    require(!lu.solve(rhs) && rhs == std::vector<double>({1, 2, 3}) && lu.factor_nonzeros() == 0,
            "失败分解后不能使用上一步的旧因子");
    check_solution(lu, starts, rows, {4, 1, -2, 2, 5}, {2, -1, 3});
    auto invalid_values = std::vector<double>{4, 1, -2, 2, std::numeric_limits<double>::infinity()};
    require(!lu.factor(invalid_values) && !lu.solve(rhs), "非有限矩阵必须作废旧因子");
    require(!lu.factor(std::vector<double>{1}), "数值数组长度错误必须拒绝");
    check_solution(lu, starts, rows, {4, 1, -2, 2, 5}, {2, -1, 3});
    rhs = {std::numeric_limits<double>::quiet_NaN(), 2, 3};
    require(!lu.solve(rhs) && std::isnan(rhs[0]) && rhs[1] == 2 && rhs[2] == 3,
            "非有限右端项必须拒绝且保持输入");
    rhs = {1};
    require(!lu.solve(rhs) && rhs == std::vector<double>({1}), "回代维数不匹配必须拒绝");

    const std::vector<Index> pivot_starts{0, 1, 3};
    const std::vector<Index> pivot_rows{0, 0, 1};
    SparseLu pivoted(2, pivot_starts, pivot_rows);
    check_solution(pivoted, pivot_starts, pivot_rows, {0, 1, 0}, {2, -3});
    require(pivoted.factor_nonzeros() >= 4, "LU 结构计数包含两组对角及超节点内保留的零项");
    SparseLu no_diagonal(2, std::vector<Index>{0, 0, 1}, std::vector<Index>{0});
    check_solution(no_diagonal, {0, 0, 1}, {0}, {2}, {4, -1});

    SparseLu scalar(1, std::vector<Index>{0, 1}, std::vector<Index>{0});
    require(scalar.factor(std::vector<double>{1e-200}), "可表示标量可分解");
    require(scalar.factor_nonzeros() == 2, "标量 LU 必须计入 L 与 U 两个对角");
    rhs = {1e200};
    require(!scalar.solve(rhs) && rhs == std::vector<double>({1e200}), "回代溢出不得写回或误报成功");
    SparseLu empty(0, std::vector<Index>{0}, std::vector<Index>{});
    rhs.clear();
    require(empty.factor(std::vector<double>{}) && empty.solve(rhs) && empty.factor_nonzeros() == 0,
            "零维系统应具有明确的空分解语义");
    SparseLu structurally_empty(1, std::vector<Index>{0, 0}, std::vector<Index>{});
    require(!structurally_empty.factor(std::vector<double>{}), "非零维空矩阵是奇异系统");

    require_bad_pattern(0, {}, {});
    require_bad_pattern(1, {1, 1}, {});
    require_bad_pattern(2, {0, 2, 1}, {0});
    require_bad_pattern(2, {0, 1, 1}, {1});
    require_bad_pattern(1, {0, 1}, {-1});
    require_bad_pattern(2, {0, 1, 3}, {0, 1, 0});
    require_bad_pattern(2, {0, 1, 3}, {0, 0, 0});
    std::cout << "SparseLU 隔离检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "SparseLU 隔离检查失败：" << error.what() << '\n';
    return 1;
  }
}
