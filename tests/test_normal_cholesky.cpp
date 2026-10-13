#include "solver/detail/normal_cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Index = std::int64_t;
using solver::detail::NormalCholesky;
using solver::detail::NormalCholeskyKernel;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> multiply(const std::vector<Index>& starts, const std::vector<Index>& rows,
                             const std::vector<double>& values, const std::vector<double>& x) {
  std::vector<long double> product(x.size(), 0);
  for (std::size_t column = 0; column < x.size(); ++column) {
    for (Index entry = starts[column]; entry < starts[column + 1]; ++entry) {
      const auto source = static_cast<std::size_t>(entry);
      const auto row = static_cast<std::size_t>(rows[source]);
      product[row] += static_cast<long double>(values[source]) * x[column];
      if (row != column) product[column] += static_cast<long double>(values[source]) * x[row];
    }
  }
  return {product.begin(), product.end()};
}

void check_solution(NormalCholesky& factor, const std::vector<Index>& starts, const std::vector<Index>& rows,
                    const std::vector<double>& values, const std::vector<double>& expected, bool refactor = true,
                    std::span<const double> coordinate_scale = {}) {
  require(coordinate_scale.empty() || coordinate_scale.size() == expected.size(), "回归坐标尺度维数必须匹配");
  const auto original = multiply(starts, rows, values, expected);
  auto actual = original;
  if (refactor) require(factor.factor(values), "正对角 B、D 的测试系统应成功分解");
  require(factor.solve(actual), "已分解系统应成功回代");
  const auto residual = multiply(starts, rows, values, actual);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    // 等价对称变换的前向误差在原坐标衡量，避免零坐标被单位变化放大。
    // 下方完整 KKT 残差仍在本次实际输入坐标中使用原门槛。
    const double scale = coordinate_scale.empty() ? 1 : coordinate_scale[i];
    const double error = std::abs((actual[i] - expected[i]) * scale);
    const double tolerance = 1e-11 * std::max(1.0, std::abs(expected[i] * scale));
    require(error <= tolerance, "正规方程回代必须恢复已知解；坐标=" + std::to_string(i) +
        "；误差/门槛=" + std::to_string(error / tolerance));
    require(std::abs(residual[i] - original[i]) <= 1e-11 * std::max(1.0, std::abs(original[i])), "完整不定系统残差不合格");
  }
}

void reject_pattern(std::size_t dimension, const std::vector<Index>& starts, const std::vector<Index>& rows,
                    const std::vector<std::int8_t>& signs) {
  bool rejected = false;
  try { NormalCholesky invalid(dimension, starts, rows, signs); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "无效结构必须在访问内存前拒绝");
}

void check_multifrontal_normal(bool extended = false, int refinement_budget = 0,
    NormalCholeskyKernel kernel = NormalCholeskyKernel::eigen) {
  // 正规方程有稠密非对角块，实际覆盖分块 BLAS 运算和完整 KKT 方向恢复。
  constexpr Index primal_count = 24, dual_count = 96, size = primal_count + dual_count;
  std::vector<Index> starts{0}, rows;
  std::vector<double> values, expected(static_cast<std::size_t>(size));
  std::vector<std::int8_t> signs(static_cast<std::size_t>(size), -1);
  for (Index col = 0; col < size; ++col) {
    if (col >= primal_count) {
      for (Index row = 0; row < primal_count; ++row) {
        rows.push_back(row);
        values.push_back(static_cast<double>((row * 13 + col * 7) % 17 - 8) / 32);
      }
    } else {
      signs[static_cast<std::size_t>(col)] = 1;
    }
    rows.push_back(col);
    values.push_back(col < primal_count ? 2.0 + col / 24.0 : -4.0 - col / 96.0);
    starts.push_back(static_cast<Index>(rows.size()));
    expected[static_cast<std::size_t>(col)] = static_cast<double>(col % 11 - 5) / 4;
  }
  const bool multifrontal = !extended && refinement_budget == 0 && kernel == NormalCholeskyKernel::eigen;
  NormalCholesky block(static_cast<std::size_t>(size), starts, rows, signs, {}, multifrontal, extended, refinement_budget, kernel);
  check_solution(block, starts, rows, values, expected);
  require(block.factor_dimension() == dual_count && block.factor_nonzeros() >= dual_count &&
              block.regularized_pivots() == 0, "良态正定块应不加主元正则并报告实际分块存储");
  for (double& value : expected) value *= -3;
  check_solution(block, starts, rows, values, expected, false);
  auto invalid = values;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  auto rhs = multiply(starts, rows, values, expected);
  const auto original = rhs;
  require(!block.factor(invalid) && !block.solve(rhs) && rhs == original && block.factor_nonzeros() == 0 &&
              block.regularized_pivots() == 0, "失败分解必须作废旧因子并清空当前统计");
  values[0] *= 2;
  check_solution(block, starts, rows, values, expected);
  // 对同一个良态问题作跨十二数量级的对称坐标变换，验收仍使用未缩放 KKT。
  auto scaled_values = values, scaled_expected = expected;
  std::vector<double> coordinate_scale(static_cast<std::size_t>(size), 1);
  for (Index i = primal_count; i < size; ++i) {
    coordinate_scale[static_cast<std::size_t>(i)] = std::pow(10.0, static_cast<double>(i % 13 - 6));
    scaled_expected[static_cast<std::size_t>(i)] /= coordinate_scale[static_cast<std::size_t>(i)];
  }
  for (Index col = 0; col < size; ++col) {
    for (Index entry = starts[static_cast<std::size_t>(col)]; entry < starts[static_cast<std::size_t>(col + 1)]; ++entry) {
      const auto k = static_cast<std::size_t>(entry);
      scaled_values[k] *= coordinate_scale[static_cast<std::size_t>(rows[k])] * coordinate_scale[static_cast<std::size_t>(col)];
    }
  }
  check_solution(block, starts, rows, scaled_values, scaled_expected, true, coordinate_scale);
  require(block.regularized_pivots() == 0, "良态问题的对称尺度变换不应引入小主元正则");
  // 再次使用原矩阵，确认每次分解重新生成尺度，不会累乘上一轮缩放。
  check_solution(block, starts, rows, values, expected);
  bool interrupted = false;
  try { block.solve(rhs, [] { throw std::runtime_error("分块预算结束"); }); }
  catch (const std::runtime_error&) { interrupted = true; }
  require(interrupted && rhs == original, "分块回代中断不能部分写回");
  NormalCholesky empty(0, std::vector<Index>{0}, {}, {}, {}, multifrontal, extended, refinement_budget, kernel);
  rhs.clear();
  require(empty.factor({}) && empty.solve(rhs) && empty.regularized_pivots() == 0,
          "分块选项的空系统应安全处理");
  NormalCholesky primal(2, std::vector<Index>{0, 1, 2}, std::vector<Index>{0, 1},
      std::vector<std::int8_t>{1, 1}, {}, multifrontal, extended, refinement_budget, kernel);
  check_solution(primal, {0, 1, 2}, {0, 1}, {2, 4}, {3, -2});
  require(primal.factor_dimension() == 0 && primal.factor_nonzeros() == 0,
          "纯原始系统不能创建无必要的分块因子");
  NormalCholesky rounded(3, std::vector<Index>{0, 1, 3, 5}, std::vector<Index>{0, 0, 1, 0, 2},
      std::vector<std::int8_t>{1, -1, -1}, {}, multifrontal, extended, refinement_budget, kernel);
  const bool rounded_success = rounded.factor(std::vector<double>{1, 1, -1e-300, 1, -1e-300});
  if (!multifrontal) {
    require(!rounded_success && rounded.regularized_pivots() == 0,
            "固定精度不能恢复已经丢失的极小对角，不得隐藏添加主元正则");
  } else {
    require(rounded_success && rounded.regularized_pivots() > 0,
            "分块内核的退化主元正则必须显式计数，外层另验原 KKT");
  }
}

void check_cholmod_normal() {
  constexpr auto kernel = NormalCholeskyKernel::cholmod_supernodal;
  check_multifrontal_normal(false, 0, kernel);
  const std::vector<Index> starts{0, 1, 3, 5}, rows{0, 0, 1, 0, 2};
  const std::vector<std::int8_t> signs{1, -1, -1};
  const std::vector<double> values{3, 2, -5, 4, -7};
  NormalCholesky factor(3, starts, rows, signs, {}, false, false, 0, kernel);
  NormalCholesky reference(3, starts, rows, signs);
  check_solution(factor, starts, rows, values, {1, 2, -3});
  check_solution(reference, starts, rows, values, {1, 2, -3});
  require(factor.factor_nonzeros() == 4 && reference.factor_nonzeros() == 3,
          "二阶稠密超节点应报告含上三角占位的四项 xsize，而非三项数学因子");
  auto rhs = multiply(starts, rows, values, {1, 2, -3});
  const auto original = rhs;
  int factor_checks = 0;
  require(factor.factor(values, [&] { ++factor_checks; }), "CHOLMOD 预算计数基准应分解成功");
  // 扫过本小系统的全部回调边界，实际覆盖库调用后的预算中断与因子失效。
  for (int stop = 1; stop <= factor_checks; ++stop) {
    int calls = 0;
    bool interrupted = false;
    try { factor.factor(values, [&] { if (++calls == stop) throw std::runtime_error("CHOLMOD 分解中断"); }); }
    catch (const std::runtime_error& error) { interrupted = std::string(error.what()) == "CHOLMOD 分解中断"; }
    require(interrupted && !factor.solve(rhs) && rhs == original,
            "分解任一边界中断必须作废因子且不写回右端");
    require(factor.factor(values), "预算中断之后应能复用符号结构重新分解");
  }
  int solve_checks = 0;
  require(factor.solve(rhs, [&] { ++solve_checks; }), "CHOLMOD 预算计数基准应回代成功");
  for (int stop = 1; stop <= solve_checks; ++stop) {
    rhs = original;
    int calls = 0;
    bool interrupted = false;
    try { factor.solve(rhs, [&] { if (++calls == stop) throw std::runtime_error("CHOLMOD 回代中断"); }); }
    catch (const std::runtime_error& error) { interrupted = std::string(error.what()) == "CHOLMOD 回代中断"; }
    require(interrupted && rhs == original, "回代任一预算边界中断不能部分提交右端");
  }
  rhs = {std::numeric_limits<double>::infinity(), 2, 3};
  require(!factor.solve(rhs) && std::isinf(rhs[0]) && rhs[1] == 2 && rhs[2] == 3,
          "非有限右端不能进入 CHOLMOD 或被写回");
  const double tiny = std::numeric_limits<double>::epsilon() / 4;
  require(!factor.factor(std::vector<double>{1, 1, -tiny, 1, -tiny}),
          "真实舍入成秩一的 Gram 必须拒绝，不能把非正定的返回真当成成功");
  rhs = original;
  require(!factor.solve(rhs) && rhs == original && factor.regularized_pivots() == 0,
          "非正定失败不能复用前一因子或偷偷正则主元");
  check_solution(factor, starts, rows, values, {-1, 3, 2});
  require(factor.refinement_statistics().refinement_steps == 0,
          "CHOLMOD 回代不能隐藏添加 Schur 校正");
  for (int mode = 0; mode < 3; ++mode) {
    bool rejected = false;
    try {
      NormalCholesky invalid(3, starts, rows, signs, {}, mode == 0, mode == 1, mode == 2 ? 1 : 0, kernel);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "CHOLMOD 不能与其他内核或 Schur 校正叠加");
  }
  int constructor_checks = 0;
  {
    NormalCholesky counted(3, starts, rows, signs, [&] { ++constructor_checks; }, false, false, 0, kernel);
  }
  for (int stop = 1; stop <= constructor_checks; ++stop) {
    int calls = 0;
    bool interrupted = false;
    try {
      NormalCholesky interrupted_factor(3, starts, rows, signs,
          [&] { if (++calls == stop) throw std::bad_alloc(); }, false, false, 0, kernel);
    } catch (const solver::detail::NormalCholeskyBudgetExceeded&) {
      require(false, "分配异常不能转换为结构预算回退");
    } catch (const std::bad_alloc&) { interrupted = true; }
    require(interrupted, "构造任一回调边界的分配异常必须原样传播");
  }
  std::cout << "CHOLMOD 超节点解析回代、重复分解与预算边界检查通过\n";
}

void check_extended_normal() {
  const std::vector<Index> starts{0, 1, 3, 5}, rows{0, 0, 1, 0, 2};
  const std::vector<std::int8_t> signs{1, -1, -1};
  if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
    bool rejected = false;
    try { NormalCholesky unsupported(3, starts, rows, signs, {}, false, true); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "没有额外有效位的平台必须显式拒绝扩展模式");
    return;
  }
  check_multifrontal_normal(true);
  bool rejected = false;
  try { NormalCholesky invalid(3, starts, rows, signs, {}, true, true); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "互斥内核选项必须在使用前拒绝");
  const double tiny = std::numeric_limits<double>::epsilon() / 4;
  const std::vector<double> values{1, 1, -tiny, 1, -tiny};
  NormalCholesky rounded(3, starts, rows, signs);
  require(!rounded.factor(values), "double Schur 应因舍入丢失此构造的对角差");
  NormalCholesky extended(3, starts, rows, signs, {}, false, true);
  check_solution(extended, starts, rows, values, {0, 1, -1});
  check_solution(extended, starts, rows, values, {0, -2, 2}, false);
  require(extended.regularized_pivots() == 0, "保留对角差不能依赖隐藏的主元扰动");
  std::vector<double> rhs{1, 2, 3};
  bool interrupted = false;
  try { extended.factor(values, [] { throw std::runtime_error("扩展分解预算结束"); }); }
  catch (const std::runtime_error&) { interrupted = true; }
  require(interrupted && !extended.solve(rhs) && rhs == std::vector<double>({1, 2, 3}),
          "扩展分解中断必须作废旧因子并保持右端原样");
}

void check_refined_normal() {
  const std::vector<Index> starts{0, 1, 3, 5}, rows{0, 0, 1, 0, 2};
  const std::vector<std::int8_t> signs{1, -1, -1};
  if constexpr (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
    bool rejected = false;
    try { NormalCholesky unsupported(3, starts, rows, signs, {}, false, false, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Schur 扩展残差要求平台实际提供更多有效位");
    return;
  }
  check_multifrontal_normal(false, 2);
  for (int budget : {-1, 3}) {
    bool rejected = false;
    try { NormalCholesky invalid(3, starts, rows, signs, {}, false, false, budget); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Schur 校正预算必须显式限制");
  }
  for (bool block : {false, true}) {
    bool rejected = false;
    try { NormalCholesky invalid(3, starts, rows, signs, {}, block, !block, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "本轮 Schur 校正只能使用 double 标量因子");
  }
  const std::vector<double> values{3, 2, -5, 4, -7};
  NormalCholesky refined(3, starts, rows, signs, {}, false, false, 2);
  int solves = 0;
  for (const auto& expected : std::vector<std::vector<double>>{{0, 1, -1}, {1, 2, 3}, {-0.25, 0.3, 1.7}}) {
    check_solution(refined, starts, rows, values, expected, solves == 0);
    ++solves;
    const auto stats = refined.refinement_statistics();
    require(stats.refinement_steps <= 2 * solves && stats.refinement_acceptances <= stats.refinement_steps &&
                stats.residual_products == solves + stats.refinement_steps,
            "每次 Schur 求解的矩阵乘积及有限候选回代必须完整累计");
  }
  const auto before = refined.refinement_statistics();
  require(before.refinement_acceptances > 0, "解析方向族必须实际覆盖残差下降校正");
  std::vector<double> rhs{1, 2, 3};
  bool interrupted = false;
  try {
    refined.solve(rhs, [&] {
      if (refined.refinement_statistics().residual_products > before.residual_products) {
        throw std::runtime_error("Schur 矩阵乘积预算结束");
      }
    });
  } catch (const std::runtime_error&) { interrupted = true; }
  const auto stopped = refined.refinement_statistics();
  require(interrupted && rhs == std::vector<double>({1, 2, 3}) &&
              stopped.residual_products == before.residual_products + 1 &&
              stopped.refinement_steps == before.refinement_steps,
          "乘积内部中断须原样传播、不部分写回，且已启动工作不丢失");
  require(!refined.factor(std::vector<double>{3, 2, 0, 4, -7}) && !refined.solve(rhs) &&
              refined.refinement_statistics().residual_products == stopped.residual_products,
          "失败分解作废旧因子，但不能清除累计校正成本");
  check_solution(refined, starts, rows, values, {1, 2, 3});
  const double tiny = std::numeric_limits<double>::epsilon() / 4;
  require(!refined.factor(std::vector<double>{1, 1, -tiny, 1, -tiny}),
          "校正不应偷偷替换 double 因子，也不能修复未成功分解的矩阵");
}

}  // 匿名命名空间

int main() {
  try {
    check_multifrontal_normal();
    check_extended_normal();
    check_refined_normal();
    check_cholmod_normal();
    const std::vector<Index> starts{0, 1, 3}, rows{0, 0, 1};
    NormalCholesky reversed(2, starts, rows, std::vector<std::int8_t>{-1, 1});
    std::vector<double> rhs{11, 0};
    require(!reversed.solve(rhs) && rhs == std::vector<double>({11, 0}), "未分解不能修改右端项");
    check_solution(reversed, starts, rows, {-3, 4, 2}, {-1, 2});
    require(reversed.factor_dimension() == 1 && reversed.factor_nonzeros() == 1, "统计只含正规方程 L 及其对角");
    check_solution(reversed, starts, rows, {-3, 4, 2}, {3, -2}, false);
    check_solution(reversed, starts, rows, {-6, 0, 8}, {2, 3});
    check_solution(reversed, starts, rows, {-5, -2, 3}, {-4, 1});
    NormalCholesky forward(2, starts, rows, std::vector<std::int8_t>{1, -1});
    check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
    // 对称均衡后的坐标仍使用同一块符号，而不是按位置猜原始/对偶身份。
    check_solution(forward, starts, rows, {200, 4, -0.03}, {0.2, -10});

    const std::vector<Index> mixed_starts{0, 1, 3, 5, 8, 11};
    const std::vector<Index> mixed_rows{0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4};
    const std::vector<double> mixed_values{-3, 1, 2, 3, -4, 2, -1, 5, -2, 4, -2};
    NormalCholesky mixed(5, mixed_starts, mixed_rows, std::vector<std::int8_t>{-1, 1, -1, 1, -1});
    check_solution(mixed, mixed_starts, mixed_rows, mixed_values, {-1, 2, 3, -2, 0.5});
    check_solution(mixed, mixed_starts, mixed_rows, mixed_values, {2, 0, -1, 3, 4}, false);
    require(mixed.factor_dimension() == 3 && mixed.factor_nonzeros() == 6, "共享列产生的 Gram 项应合并且计入 L");

    // 正规方程为中心在首位的星形图，AMD 会移动中心；验证内部排列的回代恢复。
    std::vector<Index> star_starts{0}, star_rows;
    std::vector<double> star_values;
    for (Index col = 0; col < 9; ++col) {
      if (col == 4) {
        for (Index row = 0; row < 4; ++row) { star_rows.push_back(row); star_values.push_back(1); }
      } else if (col > 4) {
        star_rows.push_back(col - 5); star_values.push_back(2);
      }
      star_rows.push_back(col);
      star_values.push_back(col < 4 ? static_cast<double>(col + 2) : -static_cast<double>(col - 2));
      star_starts.push_back(static_cast<Index>(star_rows.size()));
    }
    NormalCholesky star(9, star_starts, star_rows, std::vector<std::int8_t>{1, 1, 1, 1, -1, -1, -1, -1, -1});
    check_solution(star, star_starts, star_rows, star_values, {1, 2, -3, 4, -1, 2, 1, -3, 0.5});
    require(star.factor_dimension() == 5 && star.factor_nonzeros() == 9, "星形结构排序后应保持无额外填充");
    NormalCholesky star_cholmod(9, star_starts, star_rows,
        std::vector<std::int8_t>{1, 1, 1, 1, -1, -1, -1, -1, -1}, {}, false, false, 0,
        NormalCholeskyKernel::cholmod_supernodal);
    check_solution(star_cholmod, star_starts, star_rows, star_values, {1, 2, -3, 4, -1, 2, 1, -3, 0.5});
    check_solution(star_cholmod, star_starts, star_rows, star_values, {-2, 1, 4, -1, 2, 3, -2, 1, 0}, false);
    require(star_cholmod.factor_dimension() == 5 && star_cholmod.regularized_pivots() == 0,
            "同 AMD 的 CHOLMOD 必须恢复原坐标且不新增主元正则");

    for (const auto& invalid : std::vector<std::vector<double>>{{0, 4, -3}, {-2, 4, -3}, {2, 4, 0},
         {2, 4, 3}, {2, std::numeric_limits<double>::infinity(), -3}, {2}}) {
      require(!forward.factor(invalid), "非法对角、非有限数及长度错误必须拒绝");
      rhs = {1, 2};
      require(!forward.solve(rhs) && rhs == std::vector<double>({1, 2}) && forward.factor_nonzeros() == 0,
              "失败分解后不得使用旧因子");
      check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
    }
    rhs = {std::numeric_limits<double>::quiet_NaN(), 2};
    require(!forward.solve(rhs) && std::isnan(rhs[0]) && rhs[1] == 2, "非有限右端不得写回");
    rhs = {1};
    require(!forward.solve(rhs) && rhs == std::vector<double>({1}), "回代维数错误不得写回");
    require(!forward.factor(std::vector<double>{1e-300, 1e300, -1}), "Gram 项不可表示时必须拒绝");

    NormalCholesky rounded_singular(3, std::vector<Index>{0, 1, 3, 5}, std::vector<Index>{0, 0, 1, 0, 2},
        std::vector<std::int8_t>{1, -1, -1});
    require(!rounded_singular.factor(std::vector<double>{1, 1, -1e-300, 1, -1e-300}),
            "形成正规方程时舍入丢失正定性必须报告失败");

    NormalCholesky primal(2, std::vector<Index>{0, 1, 2}, std::vector<Index>{0, 1}, std::vector<std::int8_t>{1, 1});
    check_solution(primal, {0, 1, 2}, {0, 1}, {2, 4}, {3, -2});
    require(primal.factor_dimension() == 0 && primal.factor_nonzeros() == 0, "纯原始对角系统不构造零维 Eigen 因子");
    require(primal.factor(std::vector<double>{1e-300, 1e300}), "有限正对角可直接分解");
    rhs = {1e300, 1};
    require(!primal.solve(rhs) && rhs == std::vector<double>({1e300, 1}), "回代溢出不能写回");
    rhs = {0, 1e-300};
    require(!primal.solve(rhs) && rhs == std::vector<double>({0, 1e-300}), "非零回代结果下溢不能静默归零");

    NormalCholesky empty(0, std::vector<Index>{0}, std::vector<Index>{}, std::vector<std::int8_t>{});
    rhs.clear();
    require(empty.factor(std::vector<double>{}) && empty.solve(rhs) && empty.factor_nonzeros() == 0,
            "空系统必须有明确的成功语义");
    // 稀疏大维对角系统不应被假定稠密的维数上限误拒。
    const Index sparse_size = 10'000;
    std::vector<Index> diagonal_starts(static_cast<std::size_t>(sparse_size + 1)), diagonal_rows(static_cast<std::size_t>(sparse_size));
    for (Index i = 0; i < sparse_size; ++i) { diagonal_starts[static_cast<std::size_t>(i)] = i; diagonal_rows[static_cast<std::size_t>(i)] = i; }
    diagonal_starts.back() = sparse_size;
    NormalCholesky dual(static_cast<std::size_t>(sparse_size), diagonal_starts, diagonal_rows, std::vector<std::int8_t>(static_cast<std::size_t>(sparse_size), -1));
    require(dual.factor(std::vector<double>(static_cast<std::size_t>(sparse_size), -4)), "纯对偶大维对角系统应分解");
    rhs.assign(static_cast<std::size_t>(sparse_size), 8);
    require(dual.solve(rhs) && std::all_of(rhs.begin(), rhs.end(), [](double x) { return x == -2; }) && dual.factor_nonzeros() == sparse_size,
            "纯对偶回代符号与稀疏存储计数必须正确");

    bool interrupted = false;
    try { forward.factor(std::vector<double>{2, 4, -3}, [] { throw std::runtime_error("预算结束"); }); }
    catch (const std::runtime_error&) { interrupted = true; }
    rhs = {1, 2};
    require(interrupted && !forward.solve(rhs), "预算异常须传播并作废旧因子");
    check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
    interrupted = false;
    try { forward.solve(rhs, [] { throw std::runtime_error("预算结束"); }); }
    catch (const std::runtime_error&) { interrupted = true; }
    require(interrupted && rhs == std::vector<double>({1, 2}), "回代预算异常不能部分写回");

    reject_pattern(0, {}, {}, {});
    reject_pattern(1, {0, 1}, {0}, {});
    reject_pattern(1, {0, 1}, {0}, {0});
    reject_pattern(1, {0, 0}, {}, {1});
    reject_pattern(1, {0, 1}, {-1}, {1});
    reject_pattern(2, {0, 1, 3}, {0, 0, 1}, {1, 1});
    reject_pattern(2, {0, 1, 3}, {0, 1, 0}, {1, -1});
    reject_pattern(2, {0, 2, 1}, {0}, {1, -1});

    const Index dense_degree = 4'000;
    std::vector<Index> dense_starts{0, 1}, dense_rows{0};
    for (Index row = 1; row <= dense_degree; ++row) {
      dense_rows.push_back(0); dense_rows.push_back(row);
      dense_starts.push_back(static_cast<Index>(dense_rows.size()));
    }
    std::vector<std::int8_t> dense_signs(static_cast<std::size_t>(dense_degree + 1), -1);
    dense_signs[0] = 1;
    bool limited = false;
    try { NormalCholesky dense(static_cast<std::size_t>(dense_degree + 1), dense_starts, dense_rows, dense_signs); }
    catch (const solver::detail::NormalCholeskyBudgetExceeded&) { limited = true; }
    require(limited, "单列诱发的 Gram 贡献超预算应在实际展开前明确拒绝");
    bool budget_propagated = false;
    try {
      NormalCholesky stopped(static_cast<std::size_t>(dense_degree + 1), dense_starts, dense_rows, dense_signs,
          [] { throw std::runtime_error("独立预算中断"); });
    } catch (const solver::detail::NormalCholeskyBudgetExceeded&) {
      require(false, "预算回调异常不得转换为结构预算异常");
    } catch (const std::runtime_error& error) {
      budget_propagated = std::string(error.what()) == "独立预算中断";
    }
    require(budget_propagated, "时间预算回调必须保持原始异常");
    std::cout << "正规方程 Cholesky 隔离检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "正规方程 Cholesky 隔离检查失败：" << error.what() << '\n';
    return 1;
  }
}
