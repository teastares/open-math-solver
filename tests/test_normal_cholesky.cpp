#include "solver/detail/normal_cholesky.hpp"
#include "solver/detail/normal_gram_pattern.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Index = std::int64_t;
using solver::detail::NormalCholesky;
using solver::detail::NormalCholeskyKernel;
using solver::detail::NormalSolveInput;

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

void check_cholmod_normal(NormalCholeskyKernel kernel = NormalCholeskyKernel::cholmod_supernodal) {
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
  const std::vector<double> changed_values{5, -3, -11, 0, -13};
  int factor_checks = 0;
  require(factor.factor(changed_values, [&] { ++factor_checks; }), "CHOLMOD 预算计数基准应分解成功");
  // 扫过本小系统的全部回调边界，实际覆盖库调用后的预算中断与因子失效。
  for (int stop = 1; stop <= factor_checks; ++stop) {
    int calls = 0;
    bool interrupted = false;
    try { factor.factor(changed_values, [&] { if (++calls == stop) throw std::runtime_error("CHOLMOD 分解中断"); }); }
    catch (const std::runtime_error& error) { interrupted = std::string(error.what()) == "CHOLMOD 分解中断"; }
    require(interrupted && !factor.solve(rhs) && rhs == original,
            "分解任一边界中断必须作废因子且不写回右端");
    // 中断前后的数值不同，恢复必须覆盖缓存，不能沿用已复制的另一组数值。
    check_solution(factor, starts, rows, values, {1, 2, -3});
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

void check_extreme_gram_ranges() {
  // 两例使用 binary64 的精确二次幂；窄指数平台不承诺这些中间量可表示。
  if constexpr (std::numeric_limits<double>::radix != 2 ||
                std::numeric_limits<double>::digits != 53 ||
                std::numeric_limits<double>::max_exponent != 1024 ||
                std::numeric_limits<double>::min_exponent != -1021 ||
                std::numeric_limits<long double>::max_exponent < 4096 ||
                std::numeric_limits<long double>::min_exponent > -4096) return;
  const auto power = [](int exponent) { return std::ldexp(1.0, exponent); };
  const std::vector<Index> two_starts{0, 1, 3}, two_rows{0, 0, 1};
  const std::vector<std::int8_t> two_signs{1, -1};
  const std::vector<double> overflow_values{power(-1074), power(-37), -power(1000)};
  const std::vector<double> overflow_scale{power(-537), power(500)};
  const std::vector<Index> three_starts{0, 1, 3, 5}, three_rows{0, 0, 1, 0, 2};
  const std::vector<std::int8_t> three_signs{1, -1, -1};
  const std::vector<double> underflow_values{power(500), power(500), -power(500), power(-700), -power(-1074)};
  const std::vector<double> underflow_scale{power(500), power(500), power(-700)};
  for (const auto kernel : {NormalCholeskyKernel::eigen, NormalCholeskyKernel::cholmod_supernodal,
                           NormalCholeskyKernel::cholmod_supernodal_aat}) {
    // 这是 [[1,1],[1,-1]] 的对称尺度变换。商超过 double 范围，Gram 为有限的 2^1001。
    // 已知原坐标解为 (1,2)，完整 RHS 的两项均非零，避免对大数相消的零右端施加绝对误差要求。
    NormalCholesky overflow(2, two_starts, two_rows, two_signs, {}, false, false, 0, kernel);
    check_solution(overflow, two_starts, two_rows, overflow_values,
        {power(537), power(-499)}, true, overflow_scale);
    // 较小系数除以 B 会落到 double 以下，但它与另一系数的交叉 Gram 项仍为 2^-700。
    // 从完整 KKT 的已知解生成 RHS=(2,-1,-2^-374)，分别归一化三个解坐标以检出丢项。
    NormalCholesky underflow(3, three_starts, three_rows, three_signs, {}, false, false, 0, kernel);
    check_solution(underflow, three_starts, three_rows, underflow_values,
        {0, power(-500), power(700)}, true, underflow_scale);
  }
}

void check_rectangular_normal() {
  constexpr auto kernel = NormalCholeskyKernel::cholmod_supernodal_aat;
  // 复用完整 KKT、逐行不同对角、重复分解及全部预算边界，实际调用矩形生产分支。
  check_cholmod_normal(kernel);
  const std::vector<Index> starts{0, 1, 3}, rows{0, 0, 1};
  NormalCholesky reversed(2, starts, rows, std::vector<std::int8_t>{-1, 1}, {}, false, false, 0, kernel);
  check_solution(reversed, starts, rows, {-3, 4, 2}, {-1, 2});
  check_solution(reversed, starts, rows, {-6, 0, 8}, {2, 3});
  check_solution(reversed, starts, rows, {-5, -2, 3}, {-4, 1});
  NormalCholesky diagonal(3, std::vector<Index>{0, 1, 2, 3}, std::vector<Index>{0, 1, 2},
      std::vector<std::int8_t>{-1, -1, -1}, {}, false, false, 0, kernel);
  check_solution(diagonal, {0, 1, 2, 3}, {0, 1, 2}, {-1, -4, -9}, {2, -3, 4});
  require(diagonal.factor_dimension() == 3 && diagonal.regularized_pivots() == 0,
          "无原始列时矩形输入的逐行平方根仍须恢复独立的对偶对角");
  NormalCholesky forward(2, starts, rows, std::vector<std::int8_t>{1, -1}, {}, false, false, 0, kernel);
  check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
  for (const auto& invalid : std::vector<std::vector<double>>{{0, 4, -3}, {-2, 4, -3}, {2, 4, 0},
       {2, 4, 3}, {2, std::numeric_limits<double>::infinity(), -3},
       {2, std::numeric_limits<double>::quiet_NaN(), -3}, {2}, {1e-300, 1e300, -1}}) {
    require(!forward.factor(invalid), "矩形输入必须拒绝非法对角、非有限数、长度错误及不可表示的缩放值");
    std::vector<double> rhs{1, 2};
    require(!forward.solve(rhs) && rhs == std::vector<double>({1, 2}) && forward.factor_nonzeros() == 0,
            "矩形分解失败不能复用之前的因子或部分写回");
    check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
  }
  if constexpr (std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53 &&
                std::numeric_limits<double>::min_exponent == -1021 &&
                std::numeric_limits<double>::has_denorm == std::denorm_present) {
    // 非零 G/sqrt(D) 超出 binary64 下界时拒绝；不能让上游收到被静默删除的系数。
    const std::vector<double> underflow{std::ldexp(1.0, 1000), std::ldexp(1.0, -1074), -1};
    require(!forward.factor(underflow), "非零矩形缩放项下溢必须拒绝");
    std::vector<double> rhs{1, 2};
    require(!forward.solve(rhs) && rhs == std::vector<double>({1, 2}), "缩放下溢后旧因子必须失效");
    check_solution(forward, starts, rows, {2, 4, -3}, {2, -1});
  }
}

void check_mapped_layouts() {
  struct Mode { NormalCholeskyKernel kernel; bool multifrontal, extended; int refinement; };
  std::vector<Mode> modes{{NormalCholeskyKernel::eigen, false, false, 0},
      {NormalCholeskyKernel::eigen, true, false, 0},
      {NormalCholeskyKernel::cholmod_supernodal, false, false, 0},
      {NormalCholeskyKernel::cholmod_supernodal_aat, false, false, 0}};
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    modes.push_back({NormalCholeskyKernel::eigen, false, true, 0});
    modes.push_back({NormalCholeskyKernel::eigen, false, false, 2});
  }
  const std::vector<Index> starts{0, 1, 3, 5, 8, 11}, rows{0, 0, 1, 1, 2, 0, 2, 3, 1, 3, 4};
  const std::vector<double> original_values{-3, 1, 2, 3, -4, 2, -1, 5, -2, 4, -2};
  const std::vector<std::int8_t> signs{-1, 1, -1, 1, -1};
  const std::vector<Index> map{4, 0, 6, 2, 7};
  const std::vector<double> scales{0.5, 2, 0.25, 4, 1};
  for (const auto mode : modes) {
    NormalCholesky mapped(5, starts, rows, signs, {}, mode.multifrontal, mode.extended,
        mode.refinement, mode.kernel, map);
    NormalCholesky reference(5, starts, rows, signs, {}, mode.multifrontal, mode.extended,
        mode.refinement, mode.kernel);
    // 连续求解再改变因子，不能持有上次CHOLMOD借用的解，也不能缓存旧数值尺度。
    for (int epoch : {0, 1, 0}) {
      auto values = original_values;
      if (epoch) for (double& value : values) value *= 2;
      require(mapped.factor(values) && reference.factor(values), "映射夹具的两份真实因子应成功");
      for (const auto& point : std::vector<std::vector<double>>{{-1, 2, 3, -2, 0.5}, {2, -1, 0.5, 3, -4}}) {
        const auto normal_rhs = multiply(starts, rows, values, point);
        auto legacy = normal_rhs;
        require(reference.solve(legacy), "恒等入口应提供同核坐标对照");
        std::vector<double> input(9, 0), output(9, 73);
        for (std::size_t i = 0; i < map.size(); ++i) input[static_cast<std::size_t>(map[i])] = normal_rhs[i] / scales[i];
        std::vector<long double> prefix(input.begin(), input.begin() + 3);
        input[0] = -999; input[2] = 999;
        const auto saved_input = input;
        const auto saved_prefix = prefix;
        require(mapped.solve_into(NormalSolveInput{input, prefix, scales}, output), "带间隙的组合映射应成功求解");
        require(input == saved_input && prefix == saved_prefix, "映射入口不能改原右端及扩展前缀");
        std::vector<double> recovered(5);
        for (std::size_t i = 0; i < map.size(); ++i) {
          const auto external = static_cast<std::size_t>(map[i]);
          require(output[external] == static_cast<double>(static_cast<long double>(legacy[i]) * scales[i]),
              "融合映射的舍入结果须与旧Normal回代加独立外层缩放一致");
          recovered[i] = output[external] / scales[i];
          require(std::abs(recovered[i] - point[i]) <= 1e-11 * std::max(1.0, std::abs(point[i])),
              "融合映射须恢复独立指定的原坐标解析向量");
        }
        const auto product = multiply(starts, rows, values, recovered);
        for (std::size_t i = 0; i < product.size(); ++i)
          require(std::abs(product[i] - normal_rhs[i]) <= 1e-11 * std::max(1.0, std::abs(normal_rhs[i])),
              "组合映射后的完整不定系统残差须合格");
        for (std::size_t i = 0; i < output.size(); ++i)
          if (std::find(map.begin(), map.end(), static_cast<Index>(i)) == map.end())
            require(output[i] == 73, "Normal不能覆盖外部单例恢复预留的位置");
        auto wrapper = normal_rhs;
        require(mapped.solve(wrapper) && wrapper == legacy, "有外部映射的对象仍须保旧solve的内部坐标语义");
      }
    }
  }
  std::cout << "Normal组合映射、多内核、重复回代和重分解检查通过\n";
}

void check_mapped_rounding() {
  const std::vector<Index> starts{0, 1}, rows{0}, map{2};
  const std::vector<std::int8_t> signs{1};
  NormalCholesky primal(1, starts, rows, signs, {}, false, false, 0,
      NormalCholeskyKernel::cholmod_supernodal_aat, map);
  require(primal.factor(std::vector<double>{1}), "纯原始映射系统应无Schur分解成功");
  std::vector<double> input{0, 0, 99}, output(3, 17), scale{3};
  std::vector<long double> prefix{0, 0, 1};
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    prefix[2] = 1 + std::ldexp(1.0L, -53);
    require(primal.solve_into(NormalSolveInput{input, prefix, scale}, output) && output[2] == 9,
        "LD前缀须先舍入double再外层缩放，不能合并两次舍入");
    if constexpr (std::numeric_limits<long double>::max_exponent > std::numeric_limits<double>::max_exponent) {
      prefix[2] = static_cast<long double>(std::numeric_limits<double>::max()) * 2;
      scale[0] = 0.25;
      require(!primal.solve_into(NormalSolveInput{input, prefix, scale}, output),
          "前缀先转double溢出必须拒绝，不能用后续缩放掩盖溢出");
    }
    if constexpr (std::numeric_limits<long double>::min_exponent < -1075) {
      prefix[2] = std::ldexp(1.0L, -1075); scale[0] = 3;
      require(primal.solve_into(NormalSolveInput{input, prefix, scale}, output) && output[2] == 0,
          "原finite允许的LD前缀舍入下溢应保持，不应误加Normal输出的非零约束");
    }
  }
  if constexpr (std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits == 53 &&
      std::numeric_limits<double>::has_denorm == std::denorm_present) {
    input[2] = std::numeric_limits<double>::denorm_min(); scale[0] = 0.5;
    require(!primal.solve_into(NormalSolveInput{input, {}, scale}, output),
        "非零double的外层输入缩放下溢仍须按原scaled拒绝");
    require(primal.factor(std::vector<double>{std::ldexp(1.0, -1022)}), "小正对角的空Schur仍应成功");
    input[2] = 1; scale[0] = 2;
    require(!primal.solve_into(NormalSolveInput{input, {}, scale}, output),
        "Normal解有限但外层输出缩放溢出不能被提交");
    require(primal.factor(std::vector<double>{std::ldexp(1.0, 1022)}), "大正对角的空Schur仍应成功");
    input[2] = std::ldexp(1.0, 1022); scale[0] = std::numeric_limits<double>::denorm_min();
    require(!primal.solve_into(NormalSolveInput{input, {}, scale}, output),
        "可表示的Normal非零解在外层输出缩放下溢仍须拒绝");
  }
  NormalCholesky empty(0, std::vector<Index>{0}, {}, {}, {}, false, false, 0,
      NormalCholeskyKernel::cholmod_supernodal_aat);
  std::vector<double> nothing;
  require(empty.factor({}) && empty.solve_into(NormalSolveInput{}, nothing), "空映射系统不能解引用空Schur工作区");
  std::cout << "Normal映射分步舍入、可表示边界与空Schur检查通过\n";
}

void check_mapped_contract_and_budget() {
  constexpr auto kernel = NormalCholeskyKernel::cholmod_supernodal_aat;
  const std::vector<Index> starts{0, 1, 3, 5}, rows{0, 0, 1, 0, 2}, map{4, 0, 2};
  const std::vector<std::int8_t> signs{1, -1, -1};
  const std::vector<double> values{3, 2, -5, 4, -7}, scale{2, 0.5, 4};
  for (const auto& invalid : std::vector<std::vector<Index>>{{1, 2}, {1, -1, 2}, {1, 1, 2}}) {
    bool rejected = false;
    try { NormalCholesky bad(3, starts, rows, signs, {}, false, false, 0, kernel, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "错误长度、负坐标或重复外部映射必须在构造时拒绝");
  }
  NormalCholesky factor(3, starts, rows, signs, {}, false, false, 0, kernel, map);
  require(factor.factor(values), "映射预算夹具应分解成功");
  const auto normal_rhs = multiply(starts, rows, values, {1, 2, -3});
  std::vector<double> input(6, 0);
  for (std::size_t i = 0; i < map.size(); ++i) input[static_cast<std::size_t>(map[i])] = normal_rhs[i] / scale[i];
  const auto saved = input;
  std::vector<double> output(6, 43);
  const auto original_output = output;
  int callbacks = 0;
  require(factor.solve_into(NormalSolveInput{input, {}, scale}, output, [&] { ++callbacks; }),
      "映射回代预算基准须真实成功");
  const auto expected_output = output;
  require(callbacks > 1, "映射预算基准须覆盖入口、库调用和末次检查");
  for (int stop = 1; stop <= callbacks; ++stop) {
    int calls = 0; bool interrupted = false; output = original_output;
    try { factor.solve_into(NormalSolveInput{input, {}, scale}, output,
        [&] { if (++calls == stop) throw std::runtime_error("映射回代预算前缀"); }); }
    catch (const std::runtime_error& error) { interrupted = std::string(error.what()) == "映射回代预算前缀"; }
    require(interrupted && calls == stop && input == saved, "每个真实预算前缀须原样传播且保持输入");
    require(output[1] == 43 && output[3] == 43 && output[5] == 43, "中断也不能覆盖未映射位置");
  }
  output = original_output;
  require(factor.solve_into(NormalSolveInput{input, {}, scale}, output) && output == expected_output,
      "预算中断之后必须用当前输入重算，不得采用部分旧解");
  int wrapper_callbacks = 0; auto wrapper = normal_rhs;
  require(factor.solve(wrapper, [&] { ++wrapper_callbacks; }), "映射对象的旧wrapper预算基准须成功");
  for (int stop = 1; stop <= wrapper_callbacks; ++stop) {
    int calls = 0; bool interrupted = false; wrapper = normal_rhs;
    try { factor.solve(wrapper, [&] { if (++calls == stop) throw std::runtime_error("旧wrapper预算前缀"); }); }
    catch (const std::runtime_error& error) { interrupted = std::string(error.what()) == "旧wrapper预算前缀"; }
    require(interrupted && wrapper == normal_rhs, "旧wrapper任一预算中断含末次检查不得提交rhs");
  }
  auto alias = saved;
  require(!factor.solve_into(NormalSolveInput{alias, {}, scale}, alias) && alias == saved,
      "输入输出重叠须在任何写入前拒绝");
  std::vector<double> scale_alias(6, 1);
  require(!factor.solve_into(NormalSolveInput{input, {}, std::span<const double>(scale_alias.data(), 3)}, scale_alias),
      "外层缩放存储与输出重叠也须拒绝");
  output = original_output;
  require(!factor.solve_into(NormalSolveInput{input, {}, std::span<const double>(scale.data(), 2)}, output) && output == original_output,
      "错误缩放长度不能访问越界或写出");
  std::vector<long double> too_long(7, 1);
  require(!factor.solve_into(NormalSolveInput{input, too_long, scale}, output), "扩展前缀不能超出外部向量");
  std::vector<double> short_output(5, 43), short_input(4, 1), mapped_short_output(4, 43);
  require(!factor.solve_into(NormalSolveInput{input, {}, scale}, short_output) &&
      !factor.solve_into(NormalSolveInput{short_input, {}, scale}, mapped_short_output),
      "输出维数不符或映射外部坐标越界必须拒绝");
  input[4] = std::numeric_limits<double>::infinity();
  require(!factor.solve_into(NormalSolveInput{input, {}, scale}, output) && std::isinf(input[4]),
      "映射位置的非有限右端必须拒绝并保持输入");
  std::cout << "Normal映射合同及全部真实预算回调前缀检查通过\n";
}

struct ExchangeMode {
  NormalCholeskyKernel kernel;
  bool multifrontal = false, extended = false;
  int refinement = 0;
};

std::vector<ExchangeMode> exchange_modes() {
  std::vector<ExchangeMode> result{{NormalCholeskyKernel::eigen}, {NormalCholeskyKernel::eigen, true},
      {NormalCholeskyKernel::cholmod_supernodal}, {NormalCholeskyKernel::cholmod_supernodal_aat}};
  if constexpr (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    result.push_back({NormalCholeskyKernel::eigen, false, true});
    result.push_back({NormalCholeskyKernel::eigen, false, false, 2});
  }
  return result;
}

struct ExchangeLayout {
  const std::vector<Index> starts{0, 1, 3, 5}, rows{0, 0, 1, 0, 2};
  const std::vector<std::int8_t> signs{1, -1, -1};
};

std::unique_ptr<NormalCholesky> exchange_factor(const ExchangeMode& mode, const ExchangeLayout& layout) {
  return std::make_unique<NormalCholesky>(3, layout.starts, layout.rows, layout.signs,
      std::function<void()>{}, mode.multifrontal, mode.extended, mode.refinement, mode.kernel);
}

bool same_double_bits(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (std::bit_cast<std::uint64_t>(left[i]) != std::bit_cast<std::uint64_t>(right[i])) return false;
  return true;
}

void exchange_copy_solution(NormalCholesky& copy, NormalCholesky& exchanged,
    const ExchangeLayout& layout, const std::vector<double>& values) {
  for (const auto& point : std::vector<std::vector<double>>{{1, 2, -3}, {-1, 3, 2}}) {
    auto expected = multiply(layout.starts, layout.rows, values, point), actual = expected;
    require(copy.solve(expected) && exchanged.solve(actual) && same_double_bits(actual, expected),
        "exchange与同后端旧span-copy分解的完整回代须逐位相同");
    check_solution(copy, layout.starts, layout.rows, values, point, false);
    check_solution(exchanged, layout.starts, layout.rows, values, point, false);
  }
  const auto first = copy.refinement_statistics(), second = exchanged.refinement_statistics();
  require(copy.factor_nonzeros() == exchanged.factor_nonzeros() && copy.factor_dimension() == exchanged.factor_dimension() &&
      copy.regularized_pivots() == exchanged.regularized_pivots() && first.residual_products == second.residual_products &&
      first.refinement_steps == second.refinement_steps && first.refinement_acceptances == second.refinement_acceptances,
      "交换输入只能改变所有权，不能改变存储报告、主元正则或实际校正统计");
}

void check_factor_exchange_copy_paths() {
  const ExchangeLayout layout;
  const std::vector<std::vector<double>> epochs{{3, 2, -5, 4, -7}, {5, -3, -11, 0, -13}, {2, 5, -7, -4, -3}};
  for (const auto mode : exchange_modes()) {
    auto copy = exchange_factor(mode, layout), exchanged = exchange_factor(mode, layout);
    std::vector<double> scratch = epochs[0], owned;
    std::uintptr_t initial_address = 0;
    for (int step = 0; step < 7; ++step) {
      const auto& values = epochs[static_cast<std::size_t>(step % 3)];
      const bool transfer = step != 3 && step != 5;
      scratch = values;
      const auto address = reinterpret_cast<std::uintptr_t>(scratch.data());
      if (step == 0) initial_address = address;
      require(copy->factor(values), "旧copy参考须真实分解本次数值");
      if (transfer) {
        require(exchanged->factor_exchange(scratch), "相同数值的exchange应真实分解成功");
        require(same_double_bits(scratch, owned.empty() ? values : owned),
            "首次无等长owned时保持Caller，后续交换返回上一份完整存储输入");
        if (step == 0) require(reinterpret_cast<std::uintptr_t>(scratch.data()) == address,
            "首个非空exchange应保留旧copy路径，Caller存储不转移");
        if (step == 1) require(reinterpret_cast<std::uintptr_t>(scratch.data()) != address,
            "已有等长owned后必须交换两个独立buffer，而非再次复制当前输入");
        if (step == 2) require(reinterpret_cast<std::uintptr_t>(scratch.data()) == initial_address,
            "第三次应用须取回原先Caller所有的buffer，不借用跨调用span");
      } else {
        require(exchanged->factor(scratch) && same_double_bits(scratch, values),
            "跨API旧span-copy仍保持Caller，并替换下一次exchange将返回的owned输入");
      }
      owned = values;
      exchange_copy_solution(*copy, *exchanged, layout, values);
    }
  }
  std::cout << "Normal各类内核copy/exchange逐位、首copy后swap及跨API重复数值检查通过\n";
}

struct ExchangeBudgetStop : std::runtime_error { using std::runtime_error::runtime_error; };

void check_factor_exchange_budget() {
  const ExchangeLayout layout;
  const std::vector<double> previous{3, 2, -5, 4, -7}, current{5, -3, -11, 0, -13}, next{2, 5, -7, -4, -3};
  for (const auto mode : exchange_modes()) for (bool warm : {false, true}) {
    int total = 0, copy_total = 0;
    {
      auto factor = exchange_factor(mode, layout), copy = exchange_factor(mode, layout);
      if (warm) require(factor->factor(previous) && copy->factor(previous), "warm预算夹具须先支付同一旧copy分解");
      auto scratch = current;
      require(factor->factor_exchange(scratch, [&] { ++total; }) &&
          copy->factor(current, [&] { ++copy_total; }) && total == copy_total && total > 1,
          "exchange不能增加、减少或移动原copy数值体的预算检查次数");
    }
    for (int cutoff = 1; cutoff <= total; ++cutoff) {
      auto factor = exchange_factor(mode, layout), reference = exchange_factor(mode, layout);
      if (warm) require(factor->factor(previous) && reference->factor(previous), "每个warm预算前缀从同一完整owned输入开始");
      auto scratch = current;
      int checks = 0;
      bool original = false;
      try { factor->factor_exchange(scratch, [&] { if (++checks == cutoff) throw ExchangeBudgetStop("交换分解预算"); }); }
      catch (const ExchangeBudgetStop&) { original = true; }
      int copy_checks = 0;
      bool copy_original = false;
      try { reference->factor(current, [&] { if (++copy_checks == cutoff) throw ExchangeBudgetStop("旧copy分解预算"); }); }
      catch (const ExchangeBudgetStop&) { copy_original = true; }
      require(original && checks == cutoff && same_double_bits(scratch, warm && cutoff > 1 ? previous : current),
          "首预算在接收输入前中断保持Caller，warm接收后的任意中断均保已交换的旧owned语义");
      std::vector<double> rhs{1, 2, 3}, copy_rhs = rhs;
      require(copy_original && copy_checks == cutoff && !factor->solve(rhs) && !reference->solve(copy_rhs) &&
          rhs == std::vector<double>({1, 2, 3}) && copy_rhs == rhs &&
          factor->factor_nonzeros() == reference->factor_nonzeros() &&
          factor->regularized_pivots() == reference->regularized_pivots() &&
          factor->factor_dimension() == reference->factor_dimension(),
          "预算中断必须拒绝回代并保rhs，nnz/主元/维数保同cutoff旧copy的真实前缀，不强求完成库分解费用归零");
      auto retry = next;
      require(factor->factor_exchange(retry), "预算中断后Caller全量覆盖的新数值应能重新分解");
      const auto& returned = cutoff > 1 ? current : warm ? previous : next;
      require(same_double_bits(retry, returned), "中断后的下一次交换须返回实际持有输入，不能假装上次已回滚");
      auto copy = exchange_factor(mode, layout);
      require(copy->factor(next), "恢复对照须独立用旧copy入口分解");
      exchange_copy_solution(*copy, *factor, layout, next);
    }
  }
  std::cout << "Normal各内核cold/warm exchange全部实际预算前缀及失效恢复检查通过\n";
}

void check_factor_exchange_failure_ownership() {
  const ExchangeLayout layout;
  const std::vector<double> previous{3, 2, -5, 4, -7}, next{2, 5, -7, -4, -3};
  for (const auto mode : exchange_modes()) {
    auto factor = exchange_factor(mode, layout);
    for (int invalid = 0; invalid < 4; ++invalid) {
      require(factor->factor(previous), "校验失败夹具须从旧copy owned开始");
      auto scratch = next;
      if (invalid == 0) scratch.pop_back();
      if (invalid == 1) scratch[1] = std::numeric_limits<double>::quiet_NaN();
      if (invalid == 2) scratch[0] = 0;
      if (invalid == 3) scratch[2] = 1;
      const auto original = scratch;
      require(!factor->factor_exchange(scratch) && same_double_bits(scratch, original),
          "长度、非有限和符号校验失败须发生在交换前，Caller逐位保持");
      std::vector<double> rhs{1, 2, 3};
      require(!factor->solve(rhs) && rhs == std::vector<double>({1, 2, 3}) && factor->factor_nonzeros() == 0,
          "校验前失败也必须作废先前因子，不能保留旧成功状态");
      auto retry = next;
      require(factor->factor_exchange(retry) && same_double_bits(retry, previous),
          "校验前失败不能替换owned，下一次成功交换仍返回先前输入");
    }
    for (int failure = 0; failure < 3; ++failure) {
      require(factor->factor(previous), "原异常夹具须先建立旧owned");
      auto scratch = next;
      int checks = 0;
      bool original = false;
      try { factor->factor_exchange(scratch, [&] {
        if (++checks != 2) return;
        if (failure == 0) throw std::bad_alloc{};
        if (failure == 1) throw std::invalid_argument("交换后原输入异常");
        throw ExchangeBudgetStop("交换后原始回调异常");
      }); }
      catch (const std::bad_alloc&) { original = failure == 0; }
      catch (const std::invalid_argument&) { original = failure == 1; }
      catch (const ExchangeBudgetStop&) { original = failure == 2; }
      std::vector<double> rhs{1, 2, 3};
      require(original && same_double_bits(scratch, previous) && !factor->solve(rhs) && rhs == std::vector<double>({1, 2, 3}),
          "交换后原异常不可转换为结构回退或回滚Caller，失败因子必须保持不可用");
    }
  }
  {
    auto factor = exchange_factor({NormalCholeskyKernel::cholmod_supernodal_aat}, layout);
    require(factor->factor(previous), "数值false夹具应先建立旧owned");
    const double tiny = std::numeric_limits<double>::denorm_min();
    const std::vector<double> rejected{1, 1, -tiny, 1, -tiny};
    auto scratch = rejected;
    require(!factor->factor_exchange(scratch) && same_double_bits(scratch, previous),
        "已通过验证但Gram数值退化返回false时，输入已经归Normal且Caller保持旧scratch");
    std::vector<double> rhs{1, 2, 3};
    require(!factor->solve(rhs) && rhs == std::vector<double>({1, 2, 3}) && factor->factor_nonzeros() == 0,
        "接收输入后的数值false必须使因子失效，不能对外回代上一份成功数值");
    auto retry = next;
    require(factor->factor_exchange(retry) && same_double_bits(retry, rejected),
        "下一次交换返回最后已验证输入，即使它未形成成功因子，也不能误记为最后成功输入");
    auto copy = exchange_factor({NormalCholeskyKernel::cholmod_supernodal_aat}, layout);
    require(copy->factor(next), "数值失败恢复对照用旧copy入口");
    exchange_copy_solution(*copy, *factor, layout, next);
  }
  NormalCholesky empty(0, std::vector<Index>{0}, {}, {});
  std::vector<double> nothing;
  require(empty.factor_exchange(nothing) && empty.solve(nothing) && nothing.empty(), "空维exchange须保持原空因子语义");
  bool interrupted = false;
  try { empty.factor_exchange(nothing, [] { throw ExchangeBudgetStop("空维入口预算"); }); }
  catch (const ExchangeBudgetStop&) { interrupted = true; }
  require(interrupted && !empty.solve(nothing) && nothing.empty(), "空维预算中断也必须使因子失效且不访问不存在的存储");
  std::cout << "Normal exchange校验前/接收后失败、原异常、空维及完整owned语义检查通过\n";
}

}  // 匿名命名空间

int main() {
  try {
    {
      struct StructureEntry { Index row; };
      const std::vector<std::vector<StructureEntry>> columns{
          {{0}, {1}, {3}}, {{0}, {1}}, {{1}, {2}}, {{1}, {3}}, {}};
      const auto pattern = solver::detail::build_normal_gram_pattern(5, columns, {});
      require(pattern.starts == std::vector<Index>({0, 1, 3, 5, 8, 9}) &&
              pattern.rows == std::vector<Index>({0, 0, 1, 1, 2, 0, 1, 3, 4}),
          "Gram 结构须稳定去重、按行升序并保留孤立对角");
    }
    check_multifrontal_normal();
    check_extended_normal();
    check_refined_normal();
    check_cholmod_normal();
    check_rectangular_normal();
    check_extreme_gram_ranges();
    check_mapped_layouts();
    check_mapped_rounding();
    check_mapped_contract_and_budget();
    check_factor_exchange_copy_paths(); check_factor_exchange_budget(); check_factor_exchange_failure_ownership();
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
    for (const auto kernel : {NormalCholeskyKernel::cholmod_supernodal, NormalCholeskyKernel::cholmod_supernodal_aat}) {
      NormalCholesky star_cholmod(9, star_starts, star_rows,
          std::vector<std::int8_t>{1, 1, 1, 1, -1, -1, -1, -1, -1}, {}, false, false, 0,
          kernel);
      check_solution(star_cholmod, star_starts, star_rows, star_values, {1, 2, -3, 4, -1, 2, 1, -3, 0.5});
      check_solution(star_cholmod, star_starts, star_rows, star_values, {-2, 1, 4, -1, 2, 3, -2, 1, 0}, false);
      require(star_cholmod.factor_dimension() == 5 && star_cholmod.regularized_pivots() == 0,
              "同 AMD 的 CHOLMOD 必须恢复原坐标且不新增主元正则");
      // 稀疏星形经 AMD 后每列长度不同，逐次改变值可发现转置槽错位或旧数值残留。
      const auto star_storage = star_cholmod.factor_nonzeros();
      auto changed_star = star_values;
      for (Index col = 0; col < 9; ++col) {
        for (Index entry = star_starts[static_cast<std::size_t>(col)];
             entry < star_starts[static_cast<std::size_t>(col + 1)]; ++entry) {
          const auto source = static_cast<std::size_t>(entry);
          if (star_rows[source] == col) {
            changed_star[source] *= 2 + static_cast<double>(col) / 8;
          } else {
            // 保留结构中的零项，再恢复非零，验证缓存的是结构而不是数值支持集。
            changed_star[source] = col == 6 ? 0 : -static_cast<double>(entry + 1) / 8;
          }
        }
      }
      check_solution(star_cholmod, star_starts, star_rows, changed_star,
          {-2, 1, 4, -1, 2, 3, -2, 1, 0});
      check_solution(star_cholmod, star_starts, star_rows, changed_star,
          {0.5, -1, 2, 3, -2, 1, 0, 2, -4}, false);
      check_solution(star_cholmod, star_starts, star_rows, star_values,
          {1, 2, -3, 4, -1, 2, 1, -3, 0.5});
      require(star_cholmod.factor_nonzeros() == star_storage,
              "同一结构改变数值和显式零项不能重新改变超节点符号存储");
    }

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
    limited = false;
    try {
      NormalCholesky dense(static_cast<std::size_t>(dense_degree + 1), dense_starts, dense_rows, dense_signs,
          {}, false, false, 0, NormalCholeskyKernel::cholmod_supernodal_aat);
    } catch (const solver::detail::NormalCholeskyBudgetExceeded&) { limited = true; }
    require(limited, "矩形数值路径不能扩大原有的 Gram 贡献结构预算");
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
