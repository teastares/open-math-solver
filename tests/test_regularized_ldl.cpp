#include "solver/detail/regularized_ldl.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Matrix = std::vector<std::vector<double>>;
int checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

struct Factor {
  QDLDL_int n;
  std::vector<QDLDL_int> ap, ai, lp, li, counts, tree, integer_work;
  std::vector<QDLDL_float> ax, lx, diagonal, inverse_diagonal, float_work;
  std::vector<QDLDL_bool> boolean_work;

  explicit Factor(const Matrix& matrix) : n(static_cast<QDLDL_int>(matrix.size())) {
    ap.push_back(0);
    for (QDLDL_int col = 0; col < n; ++col) {
      for (QDLDL_int row = 0; row <= col; ++row) {
        if (matrix[row][col] != 0 || row == col) {
          ai.push_back(row);
          ax.push_back(matrix[row][col]);
        }
      }
      ap.push_back(static_cast<QDLDL_int>(ai.size()));
    }
    lp.resize(n + 1); counts.resize(n); tree.resize(n); integer_work.resize(3 * n);
    diagonal.resize(n); inverse_diagonal.resize(n); float_work.resize(n); boolean_work.resize(n);
    const auto nonzeros = QDLDL_etree(n, ap.data(), ai.data(), integer_work.data(), counts.data(), tree.data());
    require(nonzeros >= 0, "解析测试矩阵的符号分解应成功");
    li.resize(nonzeros); lx.resize(nonzeros);
  }

  solver::detail::RegularizedLdlResult custom(const std::vector<std::int8_t>& signs,
                                               double threshold = 1e-13, double replacement = 2e-7) {
    return solver::detail::factor_regularized_ldl(n, ap.data(), ai.data(), ax.data(), lp.data(),
        li.data(), lx.data(), diagonal.data(), inverse_diagonal.data(), counts.data(), tree.data(),
        boolean_work.data(), integer_work.data(), float_work.data(), signs.data(), threshold, replacement);
  }

  QDLDL_int upstream() {
    return QDLDL_factor(n, ap.data(), ai.data(), ax.data(), lp.data(), li.data(), lx.data(),
        diagonal.data(), inverse_diagonal.data(), counts.data(), tree.data(), boolean_work.data(),
        integer_work.data(), float_work.data());
  }

  std::vector<double> solve(std::vector<double> rhs) const {
    QDLDL_solve(n, lp.data(), li.data(), lx.data(), inverse_diagonal.data(), rhs.data());
    return rhs;
  }

  Matrix reconstruct() const {
    Matrix lower(n, std::vector<double>(n, 0)), matrix(n, std::vector<double>(n, 0));
    for (QDLDL_int col = 0; col < n; ++col) {
      lower[col][col] = 1;
      for (QDLDL_int k = lp[col]; k < lp[col + 1]; ++k) lower[li[k]][col] = lx[k];
    }
    for (QDLDL_int row = 0; row < n; ++row) {
      for (QDLDL_int col = 0; col < n; ++col) {
        long double value = 0;
        for (QDLDL_int k = 0; k < n; ++k) value += static_cast<long double>(lower[row][k]) * diagonal[k] * lower[col][k];
        matrix[row][col] = static_cast<double>(value);
      }
    }
    return matrix;
  }
};

void matrix_close(const Matrix& actual, const Matrix& expected, double tolerance) {
  for (std::size_t row = 0; row < actual.size(); ++row) {
    for (std::size_t col = 0; col < actual.size(); ++col) {
      require(std::isfinite(actual[row][col]) && std::abs(actual[row][col] - expected[row][col]) <=
          tolerance * std::max(1.0, std::abs(expected[row][col])), "重建矩阵不符合预期的局部对角修正");
    }
  }
}

}  // 匿名命名空间

int main() {
  try {
    // 首个零主元的修正必须参与后续消元，不能在整次分解完成后才改 D。
    const Matrix first_zero = {{0, 1}, {1, -2}};
    Factor baseline(first_zero), corrected(first_zero);
    require(baseline.upstream() < 0, "原始 QDLDL 应拒绝首个零主元");
    auto result = corrected.custom({1, -1});
    require(result.positive_pivots == 1 && result.regularized_pivots == 1,
            "首主元应按已知正号修正且只计一次");
    matrix_close(corrected.reconstruct(), {{2e-7, 1}, {1, -2}}, 1e-9);
    const auto exact = corrected.solve({2 + 2e-7, -3});
    require(std::abs(exact[0] - 1) < 1e-8 && std::abs(exact[1] - 2) < 1e-8,
            "修正后的因子应正确求解完整线性方程");

    // 后续主元出现错误符号时，修正应反映为可重建的局部对角扰动。
    Factor wrong_sign({{1, 1}, {1, 2}});
    result = wrong_sign.custom({1, -1});
    require(result.positive_pivots == 1 && result.regularized_pivots == 1,
            "后续错误符号主元应按负号修正");
    matrix_close(wrong_sign.reconstruct(), {{1, 1}, {1, 1 - 2e-7}}, 1e-14);
    Factor tiny({{-1e-15, 0}, {0, 1e-15}});
    result = tiny.custom({-1, 1});
    require(result.positive_pivots == 1 && result.regularized_pivots == 2,
            "双符号微小主元都应计数");
    matrix_close(tiny.reconstruct(), {{-2e-7, 0}, {0, 2e-7}}, 1e-15);

    // 一个真实准定系统先置换为负、正、正的顺序，检查不触发时与上游同解。
    const Matrix permuted = {{-4, 1, -1}, {1, 3, 0.5}, {-1, 0.5, 2}};
    Factor plain(permuted), guarded(permuted);
    require(plain.upstream() == 2, "上游应识别两个正主元");
    result = guarded.custom({-1, 1, 1});
    require(result.positive_pivots == 2 && result.regularized_pivots == 0,
            "可靠主元不应被修改");
    matrix_close(guarded.reconstruct(), permuted, 1e-14);
    const std::vector<double> known = {2, -3, 5};
    std::vector<double> rhs(3, 0);
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t col = 0; col < 3; ++col) rhs[row] += permuted[row][col] * known[col];
    }
    const auto original_solution = plain.solve(rhs), guarded_solution = guarded.solve(rhs);
    for (std::size_t i = 0; i < known.size(); ++i) {
      require(std::abs(guarded_solution[i] - known[i]) < 1e-13 &&
                  std::abs(guarded_solution[i] - original_solution[i]) < 1e-13,
              "未触发时必须同时满足解析解及原 QDLDL 对照");
    }
    for (double invalid : {std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()}) {
      Factor nonfinite({{0, 0}, {0, invalid}});
      result = nonfinite.custom({1, -1});
      require(result.positive_pivots < 0 && result.regularized_pivots == 1,
              "非有限主元必须拒绝，并保留失败前修正计数");
    }
    Factor invalid_sign({{1}});
    require(invalid_sign.custom({0}).positive_pivots < 0, "未知主元符号必须拒绝");
    require(invalid_sign.custom({1}, -1, 1).positive_pivots < 0, "非法数值配置必须拒绝");
    require(invalid_sign.custom({1}, 1, 1).positive_pivots < 0, "替换量必须大于触发阈值");
    Factor empty(Matrix{});
    require(empty.custom({}).positive_pivots == 0, "零维系统不应访问首个主元");
    std::cout << "动态主元正则检查通过，共 " << checks << " 项。\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "动态主元正则检查失败：" << error.what() << '\n';
    return 1;
  }
}
