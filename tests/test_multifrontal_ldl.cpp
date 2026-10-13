#include "solver/detail/multifrontal_ldl.hpp"
#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using Index = std::int64_t;
using solver::detail::MultifrontalLdl;
int checks = 0;
void require(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
std::vector<double> multiply(const std::vector<Index>& ap, const std::vector<Index>& ai,
    const std::vector<double>& values, const std::vector<double>& x) {
  std::vector<long double> result(x.size(), 0);
  for (std::size_t col = 0; col < x.size(); ++col)
    for (auto k = ap[col]; k < ap[col + 1]; ++k) {
      const auto slot = static_cast<std::size_t>(k), row = static_cast<std::size_t>(ai[slot]);
      result[row] += static_cast<long double>(values[slot]) * x[col];
      if (row != col) result[col] += static_cast<long double>(values[slot]) * x[row];
    }
  return {result.begin(), result.end()};
}
void verify(MultifrontalLdl& factor, const std::vector<Index>& ap, const std::vector<Index>& ai,
    const std::vector<double>& values, const std::vector<double>& expected) {
  const auto original = multiply(ap, ai, values, expected);
  auto solution = original;
  require(factor.solve(solution), "有效分解必须返回有限解");
  const auto product = multiply(ap, ai, values, solution);
  for (std::size_t i = 0; i < solution.size(); ++i) {
    require(std::abs(solution[i] - expected[i]) < 1e-10 * std::max(1.0, std::abs(expected[i])), "重排、分块和回代应恢复已知解");
    require(std::abs(product[i] - original[i]) < 1e-10 * std::max(1.0, std::abs(original[i])), "原始对称系统残差超限");
  }
}
void reject(std::size_t n, const std::vector<Index>& ap, const std::vector<Index>& ai,
    const std::vector<std::int8_t>& signs) {
  bool failed = false;
  try { MultifrontalLdl factor(n, ap, ai, signs); }
  catch (const std::invalid_argument&) { failed = true; }
  require(failed, "无效结构必须在越界访问前拒绝");
}
}  // 匿名命名空间

int main() {
  try {
    require(openblas_get_num_threads() == 1 && openblas_get_parallel() == 0, "BLAS 必须编译为单线程");
    const std::vector<Index> ap{0, 1, 3}, ai{0, 0, 1};
    MultifrontalLdl simple(2, ap, ai, std::vector<std::int8_t>{-1, 1});
    std::vector<double> rhs{3, 4};
    require(!simple.solve(rhs) && rhs == std::vector<double>({3, 4}), "未分解不能覆盖右端项");
    for (auto values : {std::vector<double>{-3, 4, 2}, {-6, 0, 8}, {-5, -2, 3}}) {
      require(simple.factor(values), "准定系统应成功分解");
      verify(simple, ap, ai, values, {-1, 2});
      verify(simple, ap, ai, values, {2, -3});
      require(simple.factor_nonzeros() >= 2 && simple.regularized_pivots() == 0, "良态系统不应被额外正则");
    }
    require(!simple.factor(std::vector<double>{-1, 2}), "数值长度不匹配必须失败");
    require(!simple.solve(rhs) && simple.factor_nonzeros() == 0, "失败后不得复用旧因子");
    require(!simple.factor(std::vector<double>{1, 2, 3}), "对角符号错误必须拒绝");
    require(!simple.factor(std::vector<double>{-1, std::numeric_limits<double>::infinity(), 3}), "非有限输入必须拒绝");
    require(simple.factor(std::vector<double>{-3, 4, 2}), "失败后应能重新分解");
    const auto large = std::numeric_limits<double>::max();
    require(!simple.factor(std::vector<double>{-large, large, large}), "有限输入导致内部非有限算术时必须失败");
    require(!simple.solve(rhs) && rhs == std::vector<double>({3, 4}), "数值失败后不能使用上次因子或覆盖右端项");
    require(simple.factor(std::vector<double>{-3, 4, 2}), "内部数值失败后应能重新构建工作区");
    verify(simple, ap, ai, {-3, 4, 2}, {2, -3});
    auto nan_rhs = std::vector<double>{std::numeric_limits<double>::quiet_NaN(), 1};
    require(!simple.solve(nan_rhs) && std::isnan(nan_rhs[0]) && nan_rhs[1] == 1, "非有限右端项不能被覆盖");
    bool interrupted = false;
    int polls = 0;
    try { simple.factor(std::vector<double>{-3, 4, 2}, [&] { if (++polls == 2) throw std::runtime_error("预算耗尽"); }); }
    catch (const std::runtime_error&) { interrupted = true; }
    require(interrupted && !simple.solve(rhs), "分解后预算中断不能留下可用旧因子");
    require(simple.factor(std::vector<double>{-3, 4, 2}), "预算中断后可重新分解");
    interrupted = false; polls = 0;
    try { simple.solve(rhs, [&] { if (++polls == 2) throw std::runtime_error("预算耗尽"); }); }
    catch (const std::runtime_error&) { interrupted = true; }
    require(interrupted && rhs == std::vector<double>({3, 4}), "回代预算中断不能覆盖右端项");
    require(simple.factor(std::vector<double>{-1e-12, 1, 1e-12}), "强制二阶主元系统应成功分解");
    verify(simple, ap, ai, {-1e-12, 1, 1e-12}, {2, -3});

    MultifrontalLdl empty(0, std::vector<Index>{0}, {}, {});
    std::vector<double> empty_rhs;
    require(empty.factor({}) && empty.solve(empty_rhs) && empty.factor_nonzeros() == 0, "空系统必须安全且统计为零");
    reject(0, {}, {}, {});
    reject(2, {0, 1, 3}, {0, 0, 1}, {1, 0});
    reject(2, {0, 1, 3}, {0, 0, 0}, {1, -1});
    reject(2, {0, 1, 3}, {0, -1, 1}, {1, -1});
    reject(2, {0, 1, 3}, {0, 2, 1}, {1, -1});
    reject(2, {0, 1, 2}, {0, 0}, {1, -1});
    reject(2, {0, -1, 2}, {0, 1}, {1, -1});
    reject(2, {0, 4, 2}, {0, 1}, {1, -1});

    // 交错符号、独立分量和大于块宽的稠密交叉块覆盖多波前与 BLAS 3 路径。
    std::mt19937 generator(1729);
    std::uniform_real_distribution<double> draw(-0.25, 0.25);
    for (Index n : {1, 9, 41, 311}) {
      for (bool dense : {false, true}) {
        std::vector<Index> starts{0}, rows;
        std::vector<double> values, expected(static_cast<std::size_t>(n));
        std::vector<std::int8_t> signs(static_cast<std::size_t>(n));
        for (Index col = 0; col < n; ++col) {
          signs[static_cast<std::size_t>(col)] = col % 2 ? -1 : 1;
          for (Index row = 0; row < col; ++row) {
            if ((col - row) % 2 && (dense || (row / 8 == col / 8 && (col * 17 + row * 3) % 5 == 0))) {
              rows.push_back(row); values.push_back(draw(generator));
            }
          }
          rows.push_back(col); values.push_back(signs[static_cast<std::size_t>(col)] * (1.0 + col * 0.01));
          starts.push_back(static_cast<Index>(rows.size()));
          expected[static_cast<std::size_t>(col)] = draw(generator);
        }
        MultifrontalLdl factor(static_cast<std::size_t>(n), starts, rows, signs);
        for (int iteration = 0; iteration < 3; ++iteration) {
          require(factor.factor(values), "多波前系统数值分解失败");
          verify(factor, starts, rows, values, expected);
          for (auto& value : expected) value = draw(generator);
          verify(factor, starts, rows, values, expected);
          for (auto& value : values) value *= 1.5;
        }
      }
    }
    MultifrontalLdl small(2, std::vector<Index>{0, 1, 2}, std::vector<Index>{0, 1}, std::vector<std::int8_t>{1, -1});
    require(small.factor(std::vector<double>{1e-300, -1}) && small.regularized_pivots() > 0,
            "上游小主元扰动必须计入修正统计");
    for (Index n : {8, 40, 300}) {
      std::vector<Index> starts{0}, rows;
      std::vector<double> values, expected(static_cast<std::size_t>(n), 1);
      std::vector<std::int8_t> signs(static_cast<std::size_t>(n));
      for (Index col = 0; col < n; ++col) {
        signs[static_cast<std::size_t>(col)] = col % 2 ? -1 : 1;
        for (Index row = 0; row < col; ++row) {
          if ((col - row) % 2) {
            rows.push_back(row);
            values.push_back((col / 2 == row / 2 ? 1.0 : 0.0) + draw(generator) * 0.01);
          }
        }
        rows.push_back(col); values.push_back(signs[static_cast<std::size_t>(col)] * 1e-12);
        starts.push_back(static_cast<Index>(rows.size()));
      }
      MultifrontalLdl factor(static_cast<std::size_t>(n), starts, rows, signs);
      for (double diagonal : {1e-12, 2.0, 1e-12}) {
        for (Index col = 0; col < n; ++col)
          values[static_cast<std::size_t>(starts[static_cast<std::size_t>(col) + 1] - 1)] = signs[static_cast<std::size_t>(col)] * diagonal;
        require(factor.factor(values), "二阶主元跨分块系统应成功分解");
        verify(factor, starts, rows, values, expected);
      }
    }
    std::cout << "分块 LDL 检查通过：" << checks << "；BLAS 配置：" << openblas_get_config()
              << "；CPU 内核：" << openblas_get_corename() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
