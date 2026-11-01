#include "v92-coupled.hpp"
#include "v93-energy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
namespace coupled = solver::v92_coupled;
namespace energy = solver::v93_energy;
std::vector<const char*> details;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool status(const coupled::Work& work, const char* expected) {
  return std::string_view(work.status) == expected;
}

std::array<double, 5> tolerances(const coupled::Layout& layout, std::span<const double> rhs) {
  require(rhs.size() == layout.dimension(), "组件门槛维数须对应其完整方程");
  std::array<double, 5> value{};
  for (std::size_t i = 0; i < rhs.size(); ++i)
    value[layout.block(i)] = std::max(value[layout.block(i)], std::abs(rhs[i]));
  for (double& entry : value) entry = 1e-12 + 1e-10 * entry;
  return value;
}

void bounded(const coupled::Work& work) {
  require(work.preconditioner_completed <= work.preconditioner_started &&
      work.preconditioner_started - work.preconditioner_completed <= 1 &&
      work.products_completed <= work.products_started && work.products_started - work.products_completed <= 1 &&
      work.true_checks_completed <= work.true_checks_started && work.true_checks_started - work.true_checks_completed <= 1 &&
      work.krylov_completed <= work.krylov_started && work.krylov_started - work.krylov_completed <= 1,
      "中断账本保留实际开始与完成，至多一个当前调用未完成");
  require(work.window <= coupled::max_window && work.cycles_started <= coupled::max_cycles &&
      work.krylov_started <= 96 && work.preconditioner_started <= 97 &&
      work.products_started <= 193 && work.true_checks_started <= 97,
      "固定窗口、周期和全部实际回调均有预定上界");
}

void accepted(const coupled::Work& work) {
  bounded(work);
  require(status(work, "accepted") && work.final_measure && *work.final_measure <= 1 &&
      work.preconditioner_started == work.preconditioner_completed &&
      work.products_started == work.products_completed && work.true_checks_started == work.true_checks_completed &&
      work.krylov_started == work.krylov_completed &&
      work.preconditioner_started == 1 + work.krylov_started &&
      work.products_started == work.true_checks_started + work.krylov_started,
      "严格返回须由真实已完成乘积证明，不能把估计或未完成调用当成功");
  long double maximum = 0;
  for (long double value : work.block_measures) {
    require(std::isfinite(value) && value >= 0 && value <= 1, "每个完整块的缺陷都须分别通过");
    maximum = std::max(maximum, value);
  }
  require(maximum == *work.final_measure, "总缺陷须属于保留的同一个完整真验快照");
}

void exact_initial() {
  const coupled::Layout layout{0, 0, 0};
  const std::vector<double> rhs{3, 5};
  int calls = 0, products = 0;
  coupled::Work work;
  const auto answer = coupled::solve(layout, rhs, tolerances(layout, rhs),
      [&](std::span<const double> input, bool refinement) {
        ++calls;
        require(!refinement, "精确初始P不应额外求精化方向");
        return std::vector<double>{input[1] - input[0], input[0]};
      }, [&](std::span<const double> input, std::vector<long double>& output) {
        ++products;
        output = {input[1], static_cast<long double>(input[0]) + input[1]};
      }, work);
  accepted(work);
  require(answer == std::vector<double>({2, 3}) && calls == 1 && products == 1 &&
      work.dimension == 2 && work.window == 2 && work.cycles_started == 0 && work.krylov_started == 0 &&
      work.true_checks_completed == 1 && work.initial_measure == 0,
      "纯标量精确解应一次P和一次真实乘积返回");
  details.push_back("exact_initial_physical");
}

void variable_preconditioner() {
  const coupled::Layout layout{2, 0, 0};
  const std::vector<double> rhs{1, -2, 3, -4};
  const std::array<double, 4> diagonal{2, 3, 5, 7};
  int calls = 0;
  coupled::Work work;
  const auto answer = coupled::solve(layout, rhs, tolerances(layout, rhs),
      [&](std::span<const double> input, bool refinement) {
        ++calls;
        require(refinement == (calls > 1), "只有首次P标false，其余ArnoldiP如实标精化");
        std::vector<double> result(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
          result[i] = input[i] * (0.25 + 0.2 * static_cast<double>((i + calls) % 4)) / diagonal[i];
        return result;
      }, [&](std::span<const double> input, std::vector<long double>& output) {
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) output[i] = static_cast<long double>(diagonal[i]) * input[i];
      }, work);
  accepted(work);
  require(work.krylov_completed >= 2 && calls == work.preconditioner_completed,
      "可变P应实际进入多列FG，不能只重复初始固定逆");
  for (std::size_t i = 0; i < answer.size(); ++i)
    require(std::abs(static_cast<long double>(diagonal[i]) * answer[i] - rhs[i]) <= 1e-10L,
        "必须保存每次真实预条件方向，最终独立对角方程仍应通过");
  details.push_back("fgmres_variable_preconditioner");
}

void independent_blocks() {
  const coupled::Layout layout{1, 1, 1};
  const std::vector<double> rhs{1e12, 1, 1, 1, 1};
  const auto tolerance = tolerances(layout, rhs);
  coupled::Work work;
  const auto answer = coupled::solve(layout, rhs, tolerance,
      [&](std::span<const double> input, bool) {
        std::vector<double> result(input.begin(), input.end());
        result[3] *= 1 - 1e-6;
        return result;
      }, [](std::span<const double> input, std::vector<long double>& output) {
        output.assign(input.begin(), input.end());
      }, work);
  accepted(work);
  require(work.initial_measure && *work.initial_measure > 1000 && work.krylov_completed > 0 &&
      tolerance[0] > 1 && tolerance[3] < 1e-9,
      "大对偶块RHS不能掩盖小互补块的初始违反");
  for (std::size_t i = 0; i < answer.size(); ++i)
    require(std::abs(static_cast<long double>(answer[i]) - rhs[i]) <= tolerance[layout.block(i)],
        "真实方向须分别满足五个固定块门槛");
  details.push_back("block_tolerances_are_independent");
}

void rounded_candidate_must_fail() {
  const coupled::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 1};
  // 此组件用自有极小门槛隔离物理double舍入，不改变任何LP协议或其生产门槛。
  // 精确解为(1,-1e16+1)，第二坐标无法存入double；第一坐标又必须精确为1。
  const std::array<double, 5> tolerance{1e-30, 1e-30, 1e-30, 1e-30, 1e-30};
  coupled::Work work;
  bool failed = false;
  try {
    coupled::solve(layout, rhs, tolerance,
        [](std::span<const double> input, bool) {
          return std::vector<double>{input[1], static_cast<double>(static_cast<long double>(input[0]) - 1e16L * input[1])};
        }, [](std::span<const double> input, std::vector<long double>& output) {
          output = {1e16L * input[0] + input[1], input[0]};
        }, work);
  } catch (const coupled::Failure&) { failed = true; }
  bounded(work);
  require(failed && status(work, "exhausted") && work.true_checks_completed > 1 && work.krylov_completed > 0 &&
      work.final_measure && *work.final_measure > 1,
      "happy breakdown及零Hessenberg估计不能接纳未窄化或真实残差未过的物理候选");
  details.push_back("happy_breakdown_requires_true_residual");
}

void fixed_exhaustion() {
  const coupled::Layout layout{100, 0, 0};
  const auto dimension = layout.dimension();
  std::vector<double> rhs(dimension, 0);
  rhs[0] = 1;
  coupled::Work work;
  bool failed = false;
  try {
    coupled::solve(layout, rhs, tolerances(layout, rhs),
        [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
        [&](std::span<const double> input, std::vector<long double>& output) {
          output.assign(dimension, 0);
          for (std::size_t i = 0; i < dimension; ++i) output[(i + 1) % dimension] = input[i];
        }, work);
  } catch (const coupled::Failure&) { failed = true; }
  bounded(work);
  require(failed && status(work, "exhausted") && work.window == 32 && work.cycles_started == 3 &&
      work.krylov_started == 96 && work.krylov_completed == 96 && work.preconditioner_started == 97 &&
      work.preconditioner_completed == 97 && work.final_measure && *work.final_measure > 1,
      "102维循环移位应实际用尽固定三周期，不增窗口或返回有限未达标方向");
  details.push_back("fixed_window_and_cycle_exhaustion");
}

struct CallbackFailure { int value; };

void invalid_inputs_and_callbacks() {
  const coupled::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 2};
  const auto tolerance = tolerances(layout, rhs);
  const coupled::Preconditioner identity = [](std::span<const double> input, bool) {
    return std::vector<double>(input.begin(), input.end());
  };
  const coupled::Product multiply = [](std::span<const double> input, std::vector<long double>& output) {
    output.assign(input.begin(), input.end());
  };
  const auto rejected = [&](const coupled::Layout& shape, const std::vector<double>& target,
      const std::array<double, 5>& limits, const coupled::Preconditioner& p, const coupled::Product& a) {
    coupled::Work work;
    bool failed = false;
    try { coupled::solve(shape, target, limits, p, a, work); }
    catch (const std::exception&) { failed = true; }
    bounded(work);
    require(failed && !status(work, "accepted"), "错维或非有限回调不能正常返回方向");
  };
  rejected({0, 0, 1}, rhs, tolerance, identity, multiply);
  rejected({std::numeric_limits<std::size_t>::max(), 0, 0}, rhs, tolerance, identity, multiply);
  rejected(layout, {1}, tolerance, identity, multiply);
  rejected(layout, {std::numeric_limits<double>::quiet_NaN(), 2}, tolerance, identity, multiply);
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    auto limits = tolerance;
    limits[2] = invalid;
    rejected(layout, rhs, limits, identity, multiply);
  }
  rejected(layout, rhs, tolerance, {}, multiply);
  rejected(layout, rhs, tolerance, identity, {});
  rejected(layout, rhs, tolerance, [](std::span<const double>, bool) { return std::vector<double>{1}; }, multiply);
  rejected(layout, rhs, tolerance, [](std::span<const double>, bool) {
    return std::vector<double>{std::numeric_limits<double>::infinity(), 1};
  }, multiply);
  rejected(layout, rhs, tolerance, identity, [](std::span<const double>, std::vector<long double>& output) { output = {1}; });
  rejected(layout, rhs, tolerance, identity, [](std::span<const double>, std::vector<long double>& output) {
    output = {std::numeric_limits<long double>::quiet_NaN(), 1};
  });
  for (bool in_product : {false, true}) {
    coupled::Work work;
    bool original = false;
    try {
      coupled::solve(layout, rhs, tolerance,
          [&](std::span<const double> input, bool) {
            if (!in_product) throw CallbackFailure{37};
            return std::vector<double>(input.begin(), input.end());
          }, [&](std::span<const double>, std::vector<long double>&) { throw CallbackFailure{37}; }, work);
    } catch (const CallbackFailure& failure) { original = failure.value == 37; }
    bounded(work);
    require(original && status(work, "exception") &&
        (in_product ? work.products_started == 1 && work.products_completed == 0 :
            work.preconditioner_started == 1 && work.preconditioner_completed == 0),
        "预条件与乘积的非标准异常必须原类型传播并留下partial");
  }
  details.push_back("invalid_dimension_and_callback_values");
}

struct BudgetExpired { int boundary; };

void every_boundary(const coupled::Layout& layout, const std::vector<double>& rhs,
    const coupled::Preconditioner& precondition, const coupled::Product& multiply) {
  const auto tolerance = tolerances(layout, rhs);
  coupled::Work complete;
  int checks = 0;
  coupled::solve(layout, rhs, tolerance, precondition, multiply, complete, [&] { ++checks; });
  accepted(complete);
  require(checks > 2, "真实预算边界必须来自已实际执行的完整路径");
  bool saw_verified_but_not_committed = false;
  for (int boundary = 1; boundary <= checks; ++boundary) {
    coupled::Work partial;
    int observed = 0;
    bool original = false;
    std::vector<double> output{123, 456};
    try {
      output = coupled::solve(layout, rhs, tolerance, precondition, multiply, partial,
          [&] { if (++observed == boundary) throw BudgetExpired{boundary}; });
    } catch (const BudgetExpired& expired) { original = expired.boundary == boundary; }
    bounded(partial);
    require(original && observed == boundary && !status(partial, "accepted") && output == std::vector<double>({123, 456}),
        "任一真实预算边界均须原样中断，不能提前交付或覆盖调用者输出");
    if (partial.final_measure && *partial.final_measure <= 1) saw_verified_but_not_committed = true;
  }
  require(saw_verified_but_not_committed, "最后预算须覆盖真验已过而尚未提交的边界");
}

void budgets() {
  const coupled::Layout tiny{0, 0, 0};
  every_boundary(tiny, {1, 2},
      [](std::span<const double> input, bool) {
        return std::vector<double>{0.7 * input[0], 0.7 * input[1]};
      }, [](std::span<const double> input, std::vector<long double>& output) {
        output = {2 * static_cast<long double>(input[0]) + input[1],
            static_cast<long double>(input[0]) + 3 * input[1]};
      });
  // 长向量让坐标4096处的批内检查实际执行，不只验证每个算子首尾的检查。
  const coupled::Layout large{4096, 0, 0};
  every_boundary(large, std::vector<double>(large.dimension(), 1),
      [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
      [](std::span<const double> input, std::vector<long double>& output) { output.assign(input.begin(), input.end()); });
  details.push_back("budget_boundaries_keep_partial");
}

void pure_scalar_search() {
  const coupled::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 2};
  coupled::Work work;
  const auto answer = coupled::solve(layout, rhs, tolerances(layout, rhs),
      [](std::span<const double> input, bool) {
        return std::vector<double>{0.5 * (input[1] - input[0]), input[0]};
      }, [](std::span<const double> input, std::vector<long double>& output) {
        output = {input[1], static_cast<long double>(input[0]) + input[1]};
      }, work);
  accepted(work);
  require(work.dimension == 2 && work.window == 2 && work.krylov_completed > 0 &&
      answer.size() == 2 && std::abs(answer[0] - 1) < 1e-10 && std::abs(answer[1] - 1) < 1e-10,
      "没有x/z时仍有两个标量方向，不能把n+m=0当成空FG空间");
  details.push_back("pure_tau_kappa_has_search_space");
}

void energy_near(long double actual, long double expected, const char* message) {
  require(std::abs(actual - expected) <= 2e-16L * std::max(1.0L, std::abs(expected)), message);
}

void nominal_energy_identity() {
  // 显式名义矩阵，独立高斯消元不调用被测能量或生产KKT算子。
  std::array<std::array<long double, 4>, 3> matrix{{
      {{0.5L, 1, 2, -0.5L}}, {{1, -2, 0, 5}}, {{2, 0, -1, 1.5L}}}};
  for (std::size_t column = 0; column < 3; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < 3; ++row)
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
    std::swap(matrix[column], matrix[pivot]);
    require(matrix[column][column] != 0, "独立名义KKT夹具必须非奇异");
    const auto divisor = matrix[column][column];
    for (std::size_t entry = column; entry < 4; ++entry) matrix[column][entry] /= divisor;
    for (std::size_t row = 0; row < 3; ++row) if (row != column) {
      const auto multiple = matrix[row][column];
      for (std::size_t entry = column; entry < 4; ++entry)
        matrix[row][entry] -= multiple * matrix[column][entry];
    }
  }
  std::vector<double> exact{static_cast<double>(matrix[0][3]), static_cast<double>(matrix[1][3]),
      static_cast<double>(matrix[2][3])};
  require(exact == std::vector<double>({1, -2, 0.5}), "独立dense名义解须匹配手算完整向量");
  const std::vector<double> stored{2, 1};
  for (bool perturb : {false, true}) {
    auto v = exact;
    if (perturb) { v[0] += 0.25; v[1] -= 0.5; v[2] += 0.125; }
    energy::Work work;
    const auto denominator = energy::compute(1, v, stored, 0.5, work);
    const long double ed = 0.5L * v[0] + v[1] + 2 * v[2] + 0.5L;
    const long double ep0 = v[0] - 2.0L * v[1] - 5;
    const long double ep1 = 2.0L * v[0] - v[2] - 1.5L;
    const long double eta = v[0] * ed - v[1] * ep0 - v[2] * ep1;
    const long double schur = 1 - (v[0] + 10.0L * v[1] + 3.0L * v[2]) / 2;
    energy_near(denominator - schur, eta, "任意完整近似v须满足Ddelta-S=vx*ed-vz*ep");
    require(std::string_view(work.status) == "completed" && work.primal_terms == 1 && work.row_terms == 2 &&
        work.denominator == denominator, "名义能量工作须来自完整合法项");
    if (!perturb) { energy_near(denominator, 9.75L, "精确名义解的正能量须为9.75"); energy_near(eta, 0, "精确解eta应为零"); }
    else require(std::abs(eta) > 0.1L, "扰动向量须真实暴露名义求解缺陷项");
  }
  details.push_back("energy_nominal_schur_identity");
}

void stored_energy_contract() {
  const double h = 1, delta = 1e-12;
  const double rounded = static_cast<double>(static_cast<long double>(h) + delta);
  const std::vector<double> v{2, 1e16, -3, 4}, stored{rounded, 0.125, 7};
  energy::Work work;
  const auto actual = energy::compute(1, v, stored, 0.5, work);
  const long double expected = ((1 + 0.5L * v[0] * v[0]) +
      (static_cast<long double>(rounded) * v[1]) * v[1]) + 0.125L * v[2] * v[2] + 7.0L * v[3] * v[3];
  require(actual == expected && work.row_terms == 3 && work.primal_terms == 1,
      "全部行及原始r项须只使用实际stored double，不漏等式或恢复行");
  const long double recomputed = (1 + 0.5L * v[0] * v[0]) +
      ((static_cast<long double>(h) + delta) * v[1]) * v[1] + 0.125L * v[2] * v[2] + 7.0L * v[3] * v[3];
  require(actual != recomputed && std::abs(actual - recomputed) > 1e14L,
      "大向量必须区分实际double加法舍入与LD重新计算H+delta");
  energy::Work empty, zero;
  require(energy::compute(0, {}, {}, 0, empty) == 1 && empty.primal_terms == 0 && empty.row_terms == 0,
      "空名义系统合法且D=1");
  const std::vector<double> huge{std::numeric_limits<double>::max()};
  require(energy::compute(1, huge, {}, 0, zero) == 1 && zero.primal_terms == 1,
      "r=0先乘系数，不能因无意义v平方中间值而拒绝有限方向");
  details.push_back("energy_stored_diagonal_contract");
}

void energy_boundaries_for(std::size_t columns, const std::vector<double>& v, const std::vector<double>& stored) {
  energy::Work complete;
  int checks = 0;
  energy::compute(columns, v, stored, 0.5, complete, [&] { ++checks; });
  require(checks == complete.budget_checks && complete.primal_terms == static_cast<energy::Index>(columns) &&
      complete.row_terms == static_cast<energy::Index>(stored.size()), "预算边界须取自实际完成归约");
  bool formed = false;
  for (int boundary = 1; boundary <= checks; ++boundary) {
    energy::Work work;
    int observed = 0;
    bool original = false;
    long double output = 123;
    try { output = energy::compute(columns, v, stored, 0.5, work,
        [&] { if (++observed == boundary) throw BudgetExpired{boundary}; }); }
    catch (const BudgetExpired& expired) { original = expired.boundary == boundary; }
    require(original && observed == boundary && output == 123 && std::string_view(work.status) == "exception" &&
        work.budget_checks == observed && work.primal_terms <= complete.primal_terms && work.row_terms <= complete.row_terms,
        "任何真实预算中断须保原异常、实际partial项数且不交付输出");
    if (work.denominator) {
      formed = true;
      require(boundary == checks && work.denominator == complete.denominator &&
          work.primal_terms == complete.primal_terms && work.row_terms == complete.row_terms,
          "只有完整归约末预算处能保留已形成而未交付的D");
    }
  }
  require(formed, "末预算已形成D而未正常返回的事实必须实际覆盖");
}

void energy_budget_boundaries() {
  energy_boundaries_for(1, {2, 3, 4}, {1, 2});
  energy_boundaries_for(4098, std::vector<double>(8196, 1), std::vector<double>(4098, 2));
  const auto reject = [](std::size_t columns, const std::vector<double>& v,
      const std::vector<double>& stored, double r) {
    energy::Work work;
    bool failed = false;
    try { energy::compute(columns, v, stored, r, work); }
    catch (const energy::Failure&) { failed = true; }
    require(failed && std::string_view(work.status) == "numerical_error" && !work.denominator,
        "错维、负正则或非法贡献须拒绝，不能默认r0或记录完成D");
  };
  reject(1, {}, {}, 0);
  reject(std::numeric_limits<std::size_t>::max(), {}, {1}, 0);
  for (double value : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    reject(1, {1}, {}, value);
  for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    reject(1, {1, 1}, {value}, 0);
  reject(1, {std::numeric_limits<double>::infinity()}, {}, 0);
  reject(0, {std::numeric_limits<double>::quiet_NaN()}, {1}, 0);
  energy::Work partial;
  try { energy::compute(1, std::vector<double>{2, 3, 4}, std::vector<double>{1, 0}, 0.5, partial); }
  catch (const energy::Failure&) {}
  require(std::string_view(partial.status) == "numerical_error" && partial.primal_terms == 1 && partial.row_terms == 1,
      "非法第二行只保留前一完整合法行，不能提前增加term");
  details.push_back("energy_budget_partial_boundaries");
}

}  // 匿名命名空间

#include "v96-coupled.hpp"

namespace {
namespace wide = solver::v96_coupled;
using WideVector = std::vector<long double>;

std::array<double, 5> wide_tolerances(const wide::Layout& layout, std::span<const double> rhs) {
  std::array<double, 5> result{};
  require(rhs.size() == layout.dimension(), "宽组件门槛须对应真实五块维数");
  for (std::size_t i = 0; i < rhs.size(); ++i)
    result[layout.block(i)] = std::max(result[layout.block(i)], std::abs(rhs[i]));
  for (double& value : result) value = 1e-12 + 1e-10 * value;
  return result;
}

void wide_bounded(const wide::Work& work) {
  require(work.preconditioner_completed <= work.preconditioner_started &&
      work.preconditioner_started - work.preconditioner_completed <= 1 &&
      work.products_completed <= work.products_started && work.products_started - work.products_completed <= 1 &&
      work.true_checks_completed <= work.true_checks_started && work.true_checks_started - work.true_checks_completed <= 1 &&
      work.krylov_completed <= work.krylov_started && work.krylov_started - work.krylov_completed <= 1,
      "宽核中断须保真实启动/完成前缀");
  require(work.window <= wide::max_window && work.cycles_started <= wide::max_cycles &&
      work.krylov_started <= 96 && work.preconditioner_started <= 97 &&
      work.products_started <= 193 && work.true_checks_started <= 97,
      "宽核不能增加原窗口、周期或P/A次数上限");
}

void wide_accepted(const wide::Work& work) {
  wide_bounded(work);
  require(std::string_view(work.status) == "accepted" && work.final_measure && *work.final_measure <= 1 &&
      work.preconditioner_started == work.preconditioner_completed &&
      work.products_started == work.products_completed && work.true_checks_started == work.true_checks_completed &&
      work.krylov_started == work.krylov_completed &&
      work.preconditioner_started == 1 + work.krylov_started &&
      work.products_started == work.true_checks_started + work.krylov_started,
      "宽核返回只能由完整真实五块乘积与最后预算证明");
  long double maximum = 0;
  for (long double value : work.block_measures) {
    require(std::isfinite(value) && value >= 0 && value <= 1, "宽核每个真实块须分别过线");
    maximum = std::max(maximum, value);
  }
  require(maximum == *work.final_measure, "宽核五块必须属于同一保留方向");
}

template<class Operator>
WideVector wide_solve(const wide::Layout& layout, const std::vector<double>& rhs,
    const std::array<double, 5>& tolerance, const wide::Preconditioner& p,
    const Operator& operation, wide::Work& work, const wide::Budget& budget = {}) {
  return wide::solve(layout, rhs, tolerance, p,
      [&](std::span<const double> input, WideVector& output) { operation(input, output); },
      [&](std::span<const long double> input, WideVector& output) { operation(input, output); }, work, budget);
}

void wide_exact_initial() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{3, 5};
  int p_calls = 0, column_calls = 0, true_calls = 0;
  wide::Work work;
  const auto answer = wide::solve(layout, rhs, wide_tolerances(layout, rhs),
      [&](std::span<const double> input, bool refinement) {
        ++p_calls;
        require(!refinement, "宽初解精确时不许启动多余P");
        return std::vector<double>{input[1] - input[0], input[0]};
      }, [&](std::span<const double>, WideVector&) { ++column_calls; },
      [&](std::span<const long double> input, WideVector& output) {
        ++true_calls;
        output = {input[1], input[0] + input[1]};
      }, work);
  wide_accepted(work);
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(answer)>, WideVector>);
  require(answer == WideVector({2, 3}) && p_calls == 1 && column_calls == 0 && true_calls == 1 &&
      work.dimension == 2 && work.window == 2 && work.cycles_started == 0 && work.initial_measure == 0,
      "初P须只提升一次并由LD真验回调接纳");
  details.push_back("wide_exact_initial_physical");
}

void wide_variable_preconditioner() {
  const wide::Layout layout{2, 0, 0};
  const std::vector<double> rhs{1, -2, 3, -4};
  const std::array<double, 4> diagonal{2, 3, 5, 7};
  int calls = 0;
  wide::Work work;
  const auto answer = wide_solve(layout, rhs, wide_tolerances(layout, rhs),
      [&](std::span<const double> input, bool refinement) {
        ++calls;
        require(refinement == (calls > 1), "宽核P的首次与Arnoldi角色必须真实");
        std::vector<double> result(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
          result[i] = input[i] * (0.25 + 0.2 * static_cast<double>((i + calls) % 4)) / diagonal[i];
        return result;
      }, [&](const auto& input, WideVector& output) {
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) output[i] = static_cast<long double>(diagonal[i]) * input[i];
      }, work);
  wide_accepted(work);
  require(work.krylov_completed >= 2 && calls == work.preconditioner_completed,
      "宽核应保存真实可变P各列，实际进行FG搜索");
  for (std::size_t i = 0; i < answer.size(); ++i)
    require(std::abs(diagonal[i] * answer[i] - rhs[i]) <= 1e-10L, "独立对角方程须在LD方向上通过");
  details.push_back("wide_fgmres_variable_preconditioner");
}

void wide_independent_blocks() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{1e12, 1, 1, 1, 1};
  const auto tolerance = wide_tolerances(layout, rhs);
  wide::Work work;
  const auto answer = wide_solve(layout, rhs, tolerance,
      [](std::span<const double> input, bool) {
        std::vector<double> result(input.begin(), input.end());
        result[3] *= 1 - 1e-6;
        return result;
      }, [](const auto& input, WideVector& output) { output.assign(input.begin(), input.end()); }, work);
  wide_accepted(work);
  require(work.initial_measure && *work.initial_measure > 1000 && work.krylov_completed > 0 &&
      tolerance[0] > 1 && tolerance[3] < 1e-9, "大对偶块不能掩盖小互补块");
  for (std::size_t i = 0; i < answer.size(); ++i)
    require(std::abs(answer[i] - rhs[i]) <= tolerance[layout.block(i)], "宽方向须按固定五块门槛分别验收");
  details.push_back("wide_block_tolerances_are_independent");
}

void wide_happy_breakdown_must_fail() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 1};
  const std::array<double, 5> tolerance{1e-30, 1e-30, 1e-30, 1e-30, 1e-30};
  constexpr int exponent = std::numeric_limits<long double>::digits + 1;
  static_assert(exponent < std::numeric_limits<double>::max_exponent);
  const double coefficient = std::ldexp(1.0, exponent);
  // 两个系数均为double精确数；第二方向1-H连LD也不可表示，不能只因估计为0接纳。
  wide::Work work;
  bool failed = false;
  try {
    wide_solve(layout, rhs, tolerance,
        [&](std::span<const double> input, bool) {
          return std::vector<double>{input[1], static_cast<double>(static_cast<long double>(input[0]) - coefficient * input[1])};
        }, [&](const auto& input, WideVector& output) {
          output = {static_cast<long double>(coefficient) * input[0] + input[1], input[0]};
        }, work);
  } catch (const wide::Failure&) { failed = true; }
  wide_bounded(work);
  require(failed && std::string_view(work.status) == "exhausted" &&
      work.true_checks_completed > 1 && work.krylov_completed > 0 && work.final_measure && *work.final_measure > 1,
      "连LD也不可表示的方向必须在真实残差不过时失败，happy breakdown不能替代真验");
  details.push_back("wide_happy_breakdown_requires_true_residual");
}

void wide_fixed_exhaustion() {
  const wide::Layout layout{100, 0, 0};
  const auto dimension = layout.dimension();
  std::vector<double> rhs(dimension, 0);
  rhs[0] = 1;
  wide::Work work;
  bool failed = false;
  try {
    wide_solve(layout, rhs, wide_tolerances(layout, rhs),
        [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
        [&](const auto& input, WideVector& output) {
          output.assign(dimension, 0);
          for (std::size_t i = 0; i < dimension; ++i) output[(i + 1) % dimension] = input[i];
        }, work);
  } catch (const wide::Failure&) { failed = true; }
  wide_bounded(work);
  require(failed && std::string_view(work.status) == "exhausted" && work.window == 32 &&
      work.cycles_started == 3 && work.krylov_started == 96 && work.krylov_completed == 96 &&
      work.preconditioner_started == 97 && work.preconditioner_completed == 97 && work.final_measure && *work.final_measure > 1,
      "宽存储不许扩大原102维移位反例的三周期搜索");
  details.push_back("wide_fixed_window_and_cycle_exhaustion");
}

void wide_invalid_inputs_and_callbacks() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 2};
  const auto tolerance = wide_tolerances(layout, rhs);
  const wide::Preconditioner identity = [](std::span<const double> input, bool) {
    return std::vector<double>(input.begin(), input.end());
  };
  const wide::Preconditioner half = [](std::span<const double> input, bool) {
    std::vector<double> result(input.begin(), input.end());
    for (double& value : result) value *= 0.5;
    return result;
  };
  const wide::ProductDouble ad = [](std::span<const double> input, WideVector& output) {
    output.assign(input.begin(), input.end());
  };
  const wide::ProductWide aw = [](std::span<const long double> input, WideVector& output) {
    output.assign(input.begin(), input.end());
  };
  const auto rejected = [&](const wide::Layout& shape, const std::vector<double>& target,
      const std::array<double, 5>& limits, const wide::Preconditioner& p,
      const wide::ProductDouble& column, const wide::ProductWide& physical) {
    wide::Work work;
    bool failed = false;
    try { wide::solve(shape, target, limits, p, column, physical, work); }
    catch (const std::exception&) { failed = true; }
    wide_bounded(work);
    require(failed && std::string_view(work.status) != "accepted", "宽核非法维数或回调不得正常返回");
  };
  rejected({0, 0, 1}, rhs, tolerance, identity, ad, aw);
  rejected({std::numeric_limits<std::size_t>::max(), 0, 0}, rhs, tolerance, identity, ad, aw);
  rejected(layout, {1}, tolerance, identity, ad, aw);
  rejected(layout, {std::numeric_limits<double>::quiet_NaN(), 2}, tolerance, identity, ad, aw);
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    auto limits = tolerance;
    limits[2] = invalid;
    rejected(layout, rhs, limits, identity, ad, aw);
  }
  rejected(layout, rhs, tolerance, {}, ad, aw);
  rejected(layout, rhs, tolerance, identity, {}, aw);
  rejected(layout, rhs, tolerance, identity, ad, {});
  rejected(layout, rhs, tolerance, [](std::span<const double>, bool) { return std::vector<double>{1}; }, ad, aw);
  rejected(layout, rhs, tolerance, [](std::span<const double>, bool) {
    return std::vector<double>{std::numeric_limits<double>::quiet_NaN(), 2};
  }, ad, aw);
  rejected(layout, rhs, tolerance, identity, ad, [](std::span<const long double>, WideVector& output) { output = {1}; });
  rejected(layout, rhs, tolerance, identity, ad, [](std::span<const long double>, WideVector& output) {
    output = {std::numeric_limits<long double>::infinity(), 2};
  });
  rejected(layout, rhs, tolerance, half, [](std::span<const double>, WideVector& output) { output = {1}; }, aw);
  rejected(layout, rhs, tolerance, half, [](std::span<const double>, WideVector& output) {
    output = {std::numeric_limits<long double>::quiet_NaN(), 2};
  }, aw);
  for (int failure = 0; failure < 3; ++failure) {
    wide::Work work;
    bool original = false;
    WideVector output{101, 202};
    const wide::Preconditioner p = [&](std::span<const double> input, bool refinement) {
      if (failure == 0) throw CallbackFailure{73};
      return half(input, refinement);
    };
    const wide::ProductDouble column = [&](std::span<const double> input, WideVector& image) {
      if (failure == 1) throw CallbackFailure{73};
      ad(input, image);
    };
    const wide::ProductWide physical = [&](std::span<const long double> input, WideVector& image) {
      if (failure == 2) throw CallbackFailure{73};
      aw(input, image);
    };
    try { output = wide::solve(layout, rhs, tolerance, p, column, physical, work); }
    catch (const CallbackFailure& error) { original = error.value == 73; }
    wide_bounded(work);
    require(original && output == WideVector({101, 202}) && std::string_view(work.status) == "exception",
        "三种真实回调的原异常类型和未交付输出必须保持");
  }
  details.push_back("wide_invalid_dimension_and_callback_values");
}

void wide_budgets() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 2};
  const auto tolerance = wide_tolerances(layout, rhs);
  const wide::Preconditioner p = [](std::span<const double> input, bool) {
    std::vector<double> result(input.begin(), input.end());
    for (double& value : result) value *= 0.5;
    return result;
  };
  const auto operation = [](const auto& input, WideVector& output) { output.assign(input.begin(), input.end()); };
  int total = 0;
  wide::Work baseline;
  wide_solve(layout, rhs, tolerance, p, operation, baseline, [&] { ++total; });
  wide_accepted(baseline);
  require(total > 0 && baseline.krylov_completed > 0, "宽预算组件必须实际包含FG完整路径");
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    int calls = 0;
    bool original = false;
    wide::Work work;
    WideVector output{101, 202};
    try {
      output = wide_solve(layout, rhs, tolerance, p, operation, work, [&] {
        if (++calls == cutoff) throw CallbackFailure{cutoff};
      });
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    wide_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        std::string_view(work.status) == "exception", "每个实际预算入口中断均不得伪造宽方向返回");
  }
  details.push_back("wide_budget_boundaries_keep_partial");
}

void wide_pure_scalar_search() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{1, 2};
  wide::Work work;
  const auto answer = wide_solve(layout, rhs, wide_tolerances(layout, rhs),
      [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
      [](const auto& input, WideVector& output) {
        output = {2.0L * input[0] + input[1], input[0] + 3.0L * input[1]};
      }, work);
  wide_accepted(work);
  require(work.dimension == 2 && work.window == 2 && work.krylov_completed > 0 &&
      std::abs(2 * answer[0] + answer[1] - 1) <= 1e-10L &&
      std::abs(answer[0] + 3 * answer[1] - 2) <= 1e-10L,
      "空x/z/s块仍须留tau/kappa的两维搜索空间");
  details.push_back("wide_pure_tau_kappa_has_search_space");
}

// 下列5维矩阵是真实LP锥点G=1,c=b=0,s=z=tau=kappa=1的HSD物理Jacobian。
// 未使用更宽容差；大dual RHS不能掩盖小互补块所需的奇数方向。
template<class Scalar>
void wide_bridge_product(std::span<const Scalar> input, WideVector& output) {
  const long double dx = input[0], dz = input[1], ds = input[2], dt = input[3], dk = input[4];
  output = {dz, dx + ds, dk, ds + dz, dt + dk};
}
std::vector<double> wide_bridge_precondition(std::span<const double> input, bool) {
  const long double dz = input[0], ds = static_cast<long double>(input[3]) - dz;
  return {static_cast<double>(static_cast<long double>(input[1]) - ds), static_cast<double>(dz),
      static_cast<double>(ds), input[4] - input[2], input[2]};
}

void wide_late_narrow_bridge() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{1e16, 0, 0, 1, 0};
  const auto tolerance = wide_tolerances(layout, rhs);
  wide::Work work;
  const auto answer = wide_solve(layout, rhs, tolerance, wide_bridge_precondition,
      [](const auto& input, WideVector& output) { wide_bridge_product(input, output); }, work);
  wide_accepted(work);
  require(work.krylov_completed > 0 && answer == WideVector({1e16L - 1, 1e16L, -1e16L + 1, 0, 0}),
      "真实五块夹具须交付未丢小互补分量的LD方向");
  const long double defects[] = {answer[1] - 1e16L, answer[0] + answer[2], answer[4],
      answer[2] + answer[1] - 1, answer[3] + answer[4]};
  for (std::size_t i = 0; i < 5; ++i)
    require(std::abs(defects[i]) <= tolerance[layout.block(i)], "独立显式物理五块方程须分别通过");
  const double committed = static_cast<double>(1e16L + answer[2]);
  const double prematurely_narrowed = static_cast<double>(1e16L + static_cast<double>(answer[2]));
  require(committed == 1 && prematurely_narrowed == 0,
      "方向不可double表示不妨碍末状态可表示；必须在完整积加后才窄化");
  coupled::Work old_work;
  bool old_failed = false;
  try {
    coupled::solve({1, 1, 1}, rhs, tolerance, wide_bridge_precondition,
        [](std::span<const double> input, WideVector& output) { wide_bridge_product(input, output); }, old_work);
  } catch (const coupled::Failure&) { old_failed = true; }
  bounded(old_work);
  require(old_failed && status(old_work, "exhausted") && old_work.final_measure && *old_work.final_measure > 1,
      "旧double严格控制应真实失败，不能把组件成功当成真实LP性能收益");
  details.push_back("wide_direction_late_narrow_bridge");
}

void wide_dual_products() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{1e16, 0, 0, 1, 0};
  std::int64_t p_calls = 0, column_calls = 0, true_calls = 0, non_double_true_inputs = 0;
  wide::Work work;
  const auto answer = wide::solve(layout, rhs, wide_tolerances(layout, rhs),
      [&](std::span<const double> input, bool refinement) {
        ++p_calls;
        require(refinement == (p_calls > 1), "双A合同仍按真实首次/Arnoldi角色调用double P");
        return wide_bridge_precondition(input, refinement);
      }, [&](std::span<const double> input, WideVector& output) {
        ++column_calls;
        static_assert(std::is_same_v<typename decltype(input)::element_type, const double>);
        wide_bridge_product(input, output);
      }, [&](std::span<const long double> input, WideVector& output) {
        ++true_calls;
        for (long double value : input)
          if (value != static_cast<long double>(static_cast<double>(value))) { ++non_double_true_inputs; break; }
        wide_bridge_product(input, output);
      }, work);
  wide_accepted(work);
  require(!answer.empty() && non_double_true_inputs > 0 && column_calls == work.krylov_completed &&
      true_calls == work.true_checks_completed && work.products_completed == column_calls + true_calls &&
      p_calls == work.preconditioner_completed, "Arnoldi直接用double Z，真验必须实际收到LD小分量且每路仅计一次A");
  const std::vector<double> probe{2, -3, 5, -7, 11};
  const WideVector promoted(probe.begin(), probe.end());
  WideVector from_double, from_wide;
  wide_bridge_product(std::span<const double>(probe), from_double);
  wide_bridge_product(std::span<const long double>(promoted), from_wide);
  require(from_double == from_wide, "同一物理公式对double列及其精确LD提升必须一致");
  details.push_back("wide_double_column_wide_true_product");
}

void wide_promotion_budget() {
  const wide::Layout layout{wide::budget_batch + 1, 0, 0};
  const std::vector<double> rhs(layout.dimension(), 1);
  const auto tolerance = wide_tolerances(layout, rhs);
  const wide::Preconditioner p = [](std::span<const double> input, bool) {
    return std::vector<double>(input.begin(), input.end());
  };
  const auto operation = [](const auto& input, WideVector& output) { output.assign(input.begin(), input.end()); };
  wide::Work baseline;
  int post_p_checks = 0;
  wide_solve(layout, rhs, tolerance, p, operation, baseline, [&] {
    if (baseline.preconditioner_completed == 1 && baseline.true_checks_started == 0) ++post_p_checks;
  });
  wide_accepted(baseline);
  require(post_p_checks >= 8 && baseline.krylov_started == 0, "跨4096坐标的初P提升须真实覆盖分批预算");
  for (int cutoff = 1; cutoff <= post_p_checks; ++cutoff) {
    wide::Work work;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = wide_solve(layout, rhs, tolerance, p, operation, work, [&] {
        if (work.preconditioner_completed == 1 && work.true_checks_started == 0 && ++calls == cutoff)
          throw CallbackFailure{cutoff};
      });
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    wide_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        work.preconditioner_started == 1 && work.preconditioner_completed == 1 &&
        work.products_started == 0 && work.true_checks_started == 0 && work.krylov_started == 0 &&
        !work.initial_measure && !work.final_measure && std::string_view(work.status) == "exception",
        "初P提升或首真验入口中断保原类型和未交付状态，不补算P/A");
  }
  details.push_back("wide_promotion_budget_prefix");
}
}  // 匿名命名空间

namespace {
// 人工列误差只用于区分重启控制流；不是生产双入口误差或真实LP收益证据。
struct ResidualGapFixture {
  double rhs, diagonal, lower;
};

struct ResidualGapColumn {
  wide::Index cycle, krylov_started, true_checks_completed;
  std::optional<long double> retained_measure;
  std::array<long double, 5> retained_blocks;
  std::vector<double> input;
};

struct ResidualGapTrace {
  std::vector<ResidualGapColumn> columns;
  std::vector<long double> true_measures;
};

long double residual_gap_initial_estimate(const ResidualGapFixture& fixture) {
  // 初始基为e1，B*e1=(diagonal,lower)，且lower严格非零；首列不属happy breakdown。
  return std::abs(static_cast<long double>(fixture.rhs) * fixture.lower) /
      std::hypot(static_cast<long double>(fixture.diagonal), static_cast<long double>(fixture.lower));
}

WideVector residual_gap_solve(const ResidualGapFixture& fixture, wide::Work& work,
    ResidualGapTrace& trace, const wide::Budget& budget = {}) {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{fixture.rhs, 0};
  const std::array<double, 5> tolerance{1, 1, 1, 1, 1};
  return wide::solve(layout, rhs, tolerance,
      [&](std::span<const double> input, bool refinement) {
        if (!refinement) return std::vector<double>(2, 0);
        trace.columns.push_back({work.cycles_started, work.krylov_started, work.true_checks_completed,
            work.final_measure, work.block_measures, std::vector<double>(input.begin(), input.end())});
        return std::vector<double>(input.begin(), input.end());
      }, [&](std::span<const double> input, WideVector& output) {
        output = {static_cast<long double>(fixture.diagonal) * input[0],
            static_cast<long double>(fixture.lower) * input[0] +
                static_cast<long double>(fixture.diagonal) * input[1]};
      }, [&](std::span<const long double> input, WideVector& output) {
        output.assign(input.begin(), input.end());
        trace.true_measures.push_back(std::max(std::abs(static_cast<long double>(fixture.rhs) - input[0]),
            std::abs(input[1])));
      }, work, budget);
}

void residual_gap_improvement_restarts() {
  const ResidualGapFixture fixture{std::ldexp(1.0, 56), 1 + std::ldexp(1.0, -52), std::ldexp(1.0, -60)};
  require(fixture.lower > 0 && residual_gap_initial_estimate(fixture) <= 1,
      "改善夹具首列估计须过线且非happy breakdown，排除窗口末触发");
  wide::Work work;
  ResidualGapTrace trace;
  const auto answer = residual_gap_solve(fixture, work, trace);
  wide_accepted(work);
  require(work.cycles_started == 2 && work.krylov_started == 2 && work.krylov_completed == 2 &&
      work.preconditioner_started == 3 && work.products_started == 5 && work.true_checks_started == 3 &&
      trace.columns.size() == 2 && trace.true_measures.size() == 3,
      "改善后只用既有下一周期一列完成，不重复P/A或真验");
  require(trace.columns[0].cycle == 1 && trace.columns[1].cycle == 2 &&
      trace.columns[0].krylov_started == 1 && trace.columns[1].krylov_started == 2 &&
      trace.columns[1].true_checks_completed == 2 && trace.columns[1].retained_measure &&
      *trace.columns[1].retained_measure == trace.true_measures[1] &&
      trace.true_measures[1] > 1 && trace.true_measures[1] < trace.true_measures[0],
      "真验未过但严格改善后，第二个列P必须已在下一周期并持有完整最佳快照");
  require(trace.columns[0].input == std::vector<double>({1, 0}) &&
      trace.columns[1].input == std::vector<double>({1, 0}) &&
      std::abs(answer[0] - static_cast<long double>(fixture.rhs)) <= 1 && std::abs(answer[1]) <= 1,
      "下一周期使用保留真实残差归一基，最终宽方向仍须独立通过恒等物理算子");
  details.push_back("residual_gap_improvement_restarts");
}

void residual_gap_without_improvement_keeps_basis() {
  const ResidualGapFixture fixture{8, 0.25, std::ldexp(1.0, -6)};
  require(fixture.lower > 0 && residual_gap_initial_estimate(fixture) <= 1,
      "未改善控制也须首列估计过线而非happy breakdown");
  wide::Work work;
  ResidualGapTrace trace;
  bool failed = false;
  try { residual_gap_solve(fixture, work, trace); }
  catch (const wide::Failure&) { failed = true; }
  wide_bounded(work);
  require(failed && std::string_view(work.status) == "exhausted" &&
      work.cycles_started == 3 && work.krylov_started == 6 && work.krylov_completed == 6 &&
      work.preconditioner_started == 7 && work.preconditioner_completed == 7 &&
      work.products_started == 13 && work.products_completed == 13 &&
      work.true_checks_started == 7 && work.true_checks_completed == 7 &&
      trace.columns.size() == 6 && trace.true_measures.size() == 7,
      "未改善候选不得无条件提前重启，保持每周期两列及既有三周期耗尽");
  require(trace.true_measures[1] > trace.true_measures[0] &&
      trace.columns[1].cycle == 1 && trace.columns[1].true_checks_completed == 2 &&
      trace.columns[1].retained_measure == 8 && work.initial_measure == 8 && work.final_measure == 8,
      "首个更差真验后仍留在原basis，不能用未保留候选替换初始最佳点");
  for (std::size_t i = 0; i < trace.columns.size(); ++i) {
    require(trace.columns[i].cycle == static_cast<wide::Index>(i / 2 + 1) &&
        trace.columns[i].krylov_started == static_cast<wide::Index>(i + 1) &&
        trace.columns[i].retained_measure == 8,
        "未改善控制的全部列按原周期顺序使用同一最佳点");
  }
  details.push_back("residual_gap_without_improvement_keeps_basis");
}

void residual_gap_restart_budget_and_caps() {
  const ResidualGapFixture fixture{1024, 1.25, std::ldexp(1.0, -12)};
  require(fixture.lower > 0 && residual_gap_initial_estimate(fixture) <= 1,
      "三周期控制须首列估计过线、真残差改善但未严格通过");
  wide::Work baseline;
  ResidualGapTrace baseline_trace;
  int budget_calls = 0;
  bool exhausted = false;
  try { residual_gap_solve(fixture, baseline, baseline_trace, [&] { ++budget_calls; }); }
  catch (const wide::Failure&) { exhausted = true; }
  wide_bounded(baseline);
  require(exhausted && std::string_view(baseline.status) == "exhausted" &&
      baseline.cycles_started == wide::max_cycles && baseline.krylov_started == 3 && baseline.krylov_completed == 3 &&
      baseline.preconditioner_started == 4 && baseline.preconditioner_completed == 4 &&
      baseline.products_started == 7 && baseline.products_completed == 7 &&
      baseline.true_checks_started == 4 && baseline.true_checks_completed == 4 &&
      baseline_trace.columns.size() == 3 && baseline_trace.true_measures.size() == 4 && budget_calls > 0,
      "每周期一次严格改善仅消耗既有三周期，不能开第四周期或增加P/A");
  require(baseline.final_measure && *baseline.final_measure > 8 && *baseline.final_measure < 9,
      "三周期后仍有明确真实缺陷时必须失败，不把估计过线当严格接纳");
  for (std::size_t i = 0; i < baseline_trace.columns.size(); ++i) {
    require(baseline_trace.columns[i].cycle == static_cast<wide::Index>(i + 1) &&
        baseline_trace.true_measures[i + 1] > 1 &&
        baseline_trace.true_measures[i + 1] < baseline_trace.true_measures[i],
        "三个首列真验均严格改善且未达标，分别触发下一既有周期");
  }
  for (int cutoff = 1; cutoff <= budget_calls; ++cutoff) {
    wide::Work work;
    ResidualGapTrace trace;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = residual_gap_solve(fixture, work, trace, [&] {
        if (++calls == cutoff) throw CallbackFailure{cutoff};
      });
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    wide_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        std::string_view(work.status) == "exception" &&
        work.initial_measure.has_value() == work.final_measure.has_value(),
        "每个真实预算入口中断均保原异常和未交付方向，不补算或提前accepted");
    if (work.final_measure) {
      long double maximum = 0;
      for (long double block : work.block_measures) {
        require(std::isfinite(block) && block >= 0, "预算中断只保留完整有限块快照");
        maximum = std::max(maximum, block);
      }
      require(*work.final_measure > 1 && *work.final_measure <= *work.initial_measure && maximum == *work.final_measure,
          "预算前缀中的最佳块与measure属于同一个已完成真验");
    }
  }
  wide::Work work;
  ResidualGapTrace trace;
  std::optional<long double> retained_measure;
  std::array<long double, 5> retained_blocks{};
  int restart_boundary_hits = 0;
  bool original = false;
  WideVector output{101, 202};
  try {
    output = residual_gap_solve(fixture, work, trace, [&] {
      if (work.cycles_started == 2 && work.krylov_started == 1 && work.preconditioner_started == 2) {
        ++restart_boundary_hits;
        retained_measure = work.final_measure;
        retained_blocks = work.block_measures;
        throw CallbackFailure{991};
      }
    });
  } catch (const CallbackFailure& error) { original = error.value == 991; }
  wide_bounded(work);
  require(original && restart_boundary_hits == 1 && output == WideVector({101, 202}) &&
      std::string_view(work.status) == "exception" && work.cycles_started == 2 &&
      work.krylov_started == 1 && work.krylov_completed == 1 &&
      work.preconditioner_started == 2 && work.preconditioner_completed == 2 &&
      work.products_started == 3 && work.products_completed == 3 &&
      work.true_checks_started == 2 && work.true_checks_completed == 2 && trace.columns.size() == 1 &&
      retained_measure && work.initial_measure && *retained_measure > 1 && *retained_measure < *work.initial_measure &&
      work.final_measure == retained_measure && work.block_measures == retained_blocks &&
      trace.true_measures.size() == 2 && trace.true_measures[1] == *retained_measure,
      "第二周期已启动而下一P未开始时，预算异常须保最佳完整快照和准确P/A前缀");
  details.push_back("residual_gap_restart_budget_and_caps");
}
}  // 匿名命名空间

namespace {
struct ForcingInputs {
  std::array<double, 5> tolerance{};
  wide::InitialForcing initial;
};

ForcingInputs forcing_inputs(const wide::Layout& layout, std::span<const double> rhs) {
  require(rhs.size() == layout.dimension(), "forcing夹具须保完整未消元右端");
  std::array<double, 5> norms{};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    require(std::isfinite(rhs[i]), "forcing夹具右端须有限");
    norms[layout.block(i)] = std::max(norms[layout.block(i)], std::abs(rhs[i]));
  }
  ForcingInputs result;
  for (std::size_t block = 0; block < norms.size(); ++block) {
    result.tolerance[block] = 1e-12 + 1e-10 * norms[block];
    if (block < 3) result.initial.rhs_norms[block] = norms[block];
  }
  return result;
}

template<class Operator>
WideVector forcing_solve(const wide::Layout& layout, const std::vector<double>& rhs,
    const ForcingInputs& inputs, const wide::Preconditioner& p, const Operator& operation,
    wide::Work& work, const wide::Budget& budget = {}) {
  return wide::solve(layout, rhs, inputs.tolerance, p,
      [&](std::span<const double> input, WideVector& output) { operation(input, output); },
      [&](std::span<const long double> input, WideVector& output) { operation(input, output); },
      work, budget, &inputs.initial);
}

const auto forcing_identity = [](const auto& input, WideVector& output) {
  output.assign(input.begin(), input.end());
};

std::array<long double, 5> forcing_identity_defects(const wide::Layout& layout,
    std::span<const double> rhs, std::span<const double> initial) {
  std::array<long double, 5> result{};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    const auto block = layout.block(i);
    result[block] = std::max(result[block], std::abs(static_cast<long double>(rhs[i]) - initial[i]));
  }
  return result;
}

void forcing_bounded(const wide::Work& work) {
  wide_bounded(work);
  require(work.initial_forcing_returns >= 0 && work.initial_forcing_returns <= work.initial_forcing_checks &&
      work.initial_forcing_checks <= 1 && work.initial_forcing_checks <= work.true_checks_completed,
      "forcing计数仅属于完整初始真验且每个方向至多一次");
  if (!work.initial_forcing) {
    require(work.initial_forcing_checks == 0 && work.initial_forcing_returns == 0,
        "nullptr严格默认不得启动forcing决策");
    return;
  }
  const auto& proof = *work.initial_forcing;
  if (!proof.defect_norms) {
    require(work.true_checks_completed == 0 && !proof.measure && work.initial_forcing_checks == 0 &&
        std::string_view(proof.decision) == "not_checked", "未完成初始真验不得发布半份E5或决策");
  }
  if (proof.measure) {
    require(proof.defect_norms && work.initial_forcing_checks == 1 && std::isfinite(*proof.measure),
        "forcing量度只能来自完整已检查的初始五块缺陷");
  }
  if (std::string_view(proof.decision) == "strict_initial") {
    require(work.initial_measure && *work.initial_measure <= 1 && !proof.measure &&
        work.initial_forcing_checks == 0 && work.initial_forcing_returns == 0,
        "严格初P优先，不能另记forcing检查或非严格返回");
  }
  if (std::string_view(proof.decision) == "rejected")
    require(proof.measure && *proof.measure > 1 && work.initial_forcing_returns == 0,
        "拒绝快照须保真实LD未过事实，不受后来FG候选覆盖");
}

void forcing_returned(const wide::Work& work) {
  forcing_bounded(work);
  require(std::string_view(work.status) == "forced_initial" && work.initial_forcing &&
      work.initial_forcing_checks == 1 && work.initial_forcing_returns == 1 &&
      work.preconditioner_started == 1 && work.preconditioner_completed == 1 &&
      work.products_started == 1 && work.products_completed == 1 &&
      work.true_checks_started == 1 && work.true_checks_completed == 1 &&
      work.krylov_started == 0 && work.krylov_completed == 0 && work.cycles_started == 0 &&
      work.initial_measure && *work.initial_measure > 1 && work.final_measure == work.initial_measure,
      "非严格返回只能是已完整真验的初P，不能冒充accepted或包含FG列");
  const auto& proof = *work.initial_forcing;
  require(proof.defect_norms && proof.measure && *proof.measure <= 1 &&
      std::string_view(proof.decision) == "returned", "末预算成功后才记forcing完整返回");
  long double maximum = 0;
  for (std::size_t block = 0; block < 5; ++block) {
    long double limit = work.tolerances[block];
    if (block < 3) limit = std::max(limit, 0.1L * static_cast<long double>(proof.rhs_norms[block]));
    require((*proof.defect_norms)[block] <= limit && work.block_measures[block] >= 0,
        "初始绝对缺陷必须独立对应三线性界与两互补strict界");
    maximum = std::max(maximum, (*proof.defect_norms)[block] / limit);
  }
  require(maximum == *proof.measure && work.block_measures[3] <= 1 && work.block_measures[4] <= 1,
      "forcing量度与完整初始快照一致且不能放宽互补块");
}

void initial_forcing_strict_precedence() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  wide::Work work;
  const auto answer = forcing_solve(layout, rhs, inputs,
      [](std::span<const double> input, bool refinement) {
        require(!refinement, "严格初P不得启动第二个P");
        return std::vector<double>(input.begin(), input.end());
      }, forcing_identity, work);
  wide_accepted(work);
  forcing_bounded(work);
  require(answer == WideVector(rhs.begin(), rhs.end()) && work.initial_forcing &&
      work.initial_forcing->rhs_norms == inputs.initial.rhs_norms && work.initial_forcing->defect_norms &&
      *work.initial_forcing->defect_norms == std::array<long double, 5>{} &&
      std::string_view(work.initial_forcing->decision) == "strict_initial" &&
      work.initial_forcing_checks == 0 && work.initial_forcing_returns == 0,
      "严格成功应保存完整零缺陷但不进行forcing检查");
  details.push_back("initial_forcing_strict_precedence");
}

void initial_forcing_linear_boundaries() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  for (std::size_t target = 0; target < 3; ++target) {
    for (bool outside : {false, true}) {
      auto initial = rhs;
      initial[target] = outside ? std::nextafter(9.0, 0.0) : 9.0;
      wide::Work work;
      const auto answer = forcing_solve(layout, rhs, inputs,
          [&](std::span<const double> input, bool refinement) {
            return refinement ? std::vector<double>(input.begin(), input.end()) : initial;
          }, forcing_identity, work);
      require(work.initial_forcing && work.initial_forcing->defect_norms &&
          *work.initial_forcing->defect_norms == forcing_identity_defects(layout, rhs, initial),
          "三个线性块的边界分别使用真实初P绝对缺陷");
      if (!outside) {
        forcing_returned(work);
        require((*work.initial_forcing->defect_norms)[target] == 1 && work.initial_forcing->measure == 1,
            "三个线性块各自在F等号允许初始非严格返回");
      } else {
        wide_accepted(work);
        forcing_bounded(work);
        require(std::string_view(work.initial_forcing->decision) == "rejected" &&
            work.initial_forcing->measure && *work.initial_forcing->measure > 1 &&
            work.initial_forcing_checks == 1 && work.initial_forcing_returns == 0 &&
            work.krylov_completed == 1 && work.preconditioner_completed == 2 &&
            work.products_completed == 3 && work.true_checks_completed == 2,
            "边界外一个double值须拒初P并复用原缺陷进入strictFG");
      }
      for (std::size_t i = 0; i < answer.size(); ++i)
        require(std::abs(answer[i] - rhs[i]) <= (outside ? inputs.tolerance[layout.block(i)] : 1),
            "边界分支的交付值须对应独立恒等物理方程");
    }
  }
  details.push_back("initial_forcing_linear_boundaries");
}

void initial_forcing_complementarity_stays_strict() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  for (std::size_t target = 3; target < 5; ++target) {
    for (bool outside : {false, true}) {
      auto initial = rhs;
      initial[0] = 9;
      const double strict = inputs.tolerance[target];
      initial[target] = -(outside ? std::nextafter(strict, std::numeric_limits<double>::infinity()) : strict);
      wide::Work work;
      forcing_solve(layout, rhs, inputs,
          [&](std::span<const double> input, bool refinement) {
            return refinement ? std::vector<double>(input.begin(), input.end()) : initial;
          }, forcing_identity, work);
      require(work.initial_forcing && work.initial_forcing->defect_norms &&
          *work.initial_forcing->defect_norms == forcing_identity_defects(layout, rhs, initial),
          "两互补块须保初始真实缺陷，不能从后续最佳点重算");
      if (!outside) {
        forcing_returned(work);
        require((*work.initial_forcing->defect_norms)[target] == strict && work.block_measures[target] == 1,
            "互补块仅在原strict等号内准许初P返回");
      } else {
        wide_accepted(work);
        forcing_bounded(work);
        require(std::string_view(work.initial_forcing->decision) == "rejected" &&
            work.initial_forcing->measure && *work.initial_forcing->measure > 1 &&
            work.initial_forcing_checks == 1 && work.initial_forcing_returns == 0 && work.krylov_completed > 0,
            "任一互补块超原strict一个double值必须回退严格FG");
      }
    }
  }
  details.push_back("initial_forcing_complementarity_stays_strict");
}

void initial_forcing_empty_zero_extreme_rhs() {
  {
    const wide::Layout layout{0, 0, 0};
    const std::vector<double> rhs{10, 0};
    const auto inputs = forcing_inputs(layout, rhs);
    wide::Work work;
    forcing_solve(layout, rhs, inputs,
        [](std::span<const double>, bool refinement) {
          require(!refinement, "纯scalar forcing不应增加搜索列");
          return std::vector<double>{9, 0};
        }, forcing_identity, work);
    forcing_returned(work);
    require(work.initial_forcing->rhs_norms == std::array<double, 3>{0, 0, 10} &&
        (*work.initial_forcing->defect_norms)[0] == 0 && (*work.initial_forcing->defect_norms)[1] == 0 &&
        (*work.initial_forcing->defect_norms)[3] == 0, "空方程块须保B=E=0");
  }
  {
    const wide::Layout layout{1, 1, 1};
    const std::vector<double> rhs(5, 0);
    const auto inputs = forcing_inputs(layout, rhs);
    wide::Work work;
    forcing_solve(layout, rhs, inputs,
        [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
        forcing_identity, work);
    wide_accepted(work);
    forcing_bounded(work);
    require(work.initial_forcing->rhs_norms == std::array<double, 3>{} && work.initial_forcing_checks == 0,
        "零右端严格成功不检查forcing");
    forcing_solve(layout, rhs, inputs,
        [&](std::span<const double> input, bool refinement) {
          auto result = std::vector<double>(input.begin(), input.end());
          if (!refinement) result[0] = -2 * inputs.tolerance[0];
          return result;
        }, forcing_identity, work);
    wide_accepted(work);
    forcing_bounded(work);
    require(work.initial_forcing->measure == 2 && work.initial_forcing_returns == 0 &&
        std::string_view(work.initial_forcing->decision) == "rejected", "B=0必须只允许原strict绝对地板");
  }
  {
    const wide::Layout layout{1, 1, 1};
    const double large = std::numeric_limits<double>::max() / 4;
    const double tiny = std::numeric_limits<double>::denorm_min();
    const std::vector<double> rhs{large, tiny, 0, 0, 0};
    const auto inputs = forcing_inputs(layout, rhs);
    auto initial = rhs;
    initial[0] *= 0.95;
    initial[1] = 0;
    wide::Work work;
    forcing_solve(layout, rhs, inputs,
        [&](std::span<const double>, bool refinement) {
          require(!refinement, "巨大/极小右端合法初P不应启动FG");
          return initial;
        }, forcing_identity, work);
    forcing_returned(work);
    require(work.initial_forcing->rhs_norms == std::array<double, 3>{large, tiny, 0} && tiny > 0 &&
        inputs.tolerance[1] == 1e-12 && (*work.initial_forcing->defect_norms)[1] == tiny,
        "实际极小B与缺陷不能由已舍入的strict地板反推或丢为零");
  }
  details.push_back("initial_forcing_empty_zero_extreme_rhs");
}

void initial_forcing_rejection_strict_fallback() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  auto initial = rhs;
  initial[0] = 8;
  wide::Work work;
  int initial_calls = 0, refinement_calls = 0;
  std::optional<std::array<long double, 5>> before_fg;
  const auto p = [&](std::span<const double> input, bool refinement) {
    if (!refinement) { ++initial_calls; return initial; }
    ++refinement_calls;
    require(work.true_checks_completed == 1 && work.initial_forcing_checks == 1 && work.initial_forcing &&
        std::string_view(work.initial_forcing->decision) == "rejected", "首FG列必须直接复用已付初P真验");
    before_fg = work.initial_forcing->defect_norms;
    return std::vector<double>(input.begin(), input.end());
  };
  const auto answer = forcing_solve(layout, rhs, inputs, p, forcing_identity, work);
  wide_accepted(work);
  forcing_bounded(work);
  require(initial_calls == 1 && refinement_calls == 1 && work.preconditioner_completed == 2 &&
      work.products_completed == 3 && work.true_checks_completed == 2 && work.krylov_completed == 1 &&
      work.initial_forcing_checks == 1 && work.initial_forcing_returns == 0 &&
      work.initial_forcing->defect_norms == before_fg && work.initial_forcing->measure == 2 &&
      work.initial_measure && *work.initial_measure > 1 && work.final_measure && *work.final_measure <= 1,
      "拒绝后原FG须严格返回，初始E5与gate量度在后续候选中不变");
  wide::Work strict_work;
  const auto strict_answer = wide_solve(layout, rhs, inputs.tolerance,
      [&](std::span<const double> input, bool refinement) {
        return refinement ? std::vector<double>(input.begin(), input.end()) : initial;
      }, forcing_identity, strict_work);
  wide_accepted(strict_work);
  forcing_bounded(strict_work);
  require(!strict_work.initial_forcing && answer == strict_answer &&
      work.preconditioner_completed == strict_work.preconditioner_completed &&
      work.products_completed == strict_work.products_completed &&
      work.true_checks_completed == strict_work.true_checks_completed,
      "拒绝forcing与nullptr严格控制的FG方向和实际P/A账本完全一致");
  details.push_back("initial_forcing_rejection_strict_fallback");
}

void initial_forcing_only_initial_point() {
  const wide::Layout layout{0, 0, 0};
  const std::vector<double> rhs{100, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  wide::Work work;
  std::vector<long double> true_defects;
  bool exhausted = false;
  // 人工B列误差只制造后来候选已满足forcing的控制流反例，不冒称真实LP算子误差。
  try {
    wide::solve(layout, rhs, inputs.tolerance,
        [](std::span<const double> input, bool refinement) {
          return refinement ? std::vector<double>(input.begin(), input.end()) : std::vector<double>(2, 0);
        }, [](std::span<const double> input, WideVector& output) {
          output = {1.05L * input[0], 1e-16L * input[0] + 1.05L * input[1]};
        }, [&](std::span<const long double> input, WideVector& output) {
          output.assign(input.begin(), input.end());
          true_defects.push_back(std::max(std::abs(100.0L - input[0]), std::abs(input[1])));
        }, work, {}, &inputs.initial);
  } catch (const wide::Failure&) { exhausted = true; }
  forcing_bounded(work);
  require(exhausted && std::string_view(work.status) == "exhausted" &&
      work.cycles_started == 3 && work.krylov_completed == 3 && work.preconditioner_completed == 4 &&
      work.products_completed == 7 && work.true_checks_completed == 4 && true_defects.size() == 4 &&
      work.initial_forcing_checks == 1 && work.initial_forcing_returns == 0 &&
      work.initial_forcing && work.initial_forcing->defect_norms &&
      (*work.initial_forcing->defect_norms)[2] == 100 && work.initial_forcing->measure == 10 &&
      std::string_view(work.initial_forcing->decision) == "rejected",
      "初P拒绝后只能保原strictFG与三周期上限，不可对后来候选再次放行forcing");
  for (std::size_t i = 1; i < true_defects.size(); ++i)
    require(true_defects[i] > inputs.tolerance[2] && true_defects[i] <= 10 && true_defects[i] < true_defects[i - 1],
        "后来每个候选虽在三线性forcing界内，仍须因未strict而继续或失败");
  details.push_back("initial_forcing_only_initial_point");
}

void initial_forcing_budget_partial_snapshot() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  auto initial = rhs;
  initial[0] = 9;
  const wide::Preconditioner p = [&](std::span<const double>, bool refinement) {
    require(!refinement, "预算forcing夹具只可启动初P");
    return initial;
  };
  wide::Work baseline;
  int total = 0;
  forcing_solve(layout, rhs, inputs, p, forcing_identity, baseline, [&] { ++total; });
  forcing_returned(baseline);
  int eligible_prefixes = 0, completed_product_prefixes = 0;
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    wide::Work work;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = forcing_solve(layout, rhs, inputs, p, forcing_identity, work, [&] {
        if (++calls == cutoff) throw CallbackFailure{cutoff};
      });
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    forcing_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        std::string_view(work.status) == "exception" && work.initial_forcing &&
        work.initial_forcing->rhs_norms == inputs.initial.rhs_norms && work.initial_forcing_returns == 0,
        "所有真实预算入口须保原异常、实际B与未交付状态");
    if (work.true_checks_completed == 0) {
      require(!work.initial_forcing->defect_norms && !work.initial_forcing->measure &&
          !work.initial_measure && !work.final_measure && work.initial_forcing_checks == 0,
          "初始真验未完整完成时不得发布局部E5或初始measure");
      if (work.products_completed == 1) ++completed_product_prefixes;
    } else {
      ++eligible_prefixes;
      require(work.initial_forcing->defect_norms == baseline.initial_forcing->defect_norms &&
          work.initial_forcing->measure == baseline.initial_forcing->measure &&
          std::string_view(work.initial_forcing->decision) == "eligible" && work.initial_forcing_checks == 1 &&
          work.preconditioner_completed == 1 && work.products_completed == 1 && work.true_checks_completed == 1,
          "eligible末预算失败保完整决策，但不伪造kernel返回");
    }
  }
  require(eligible_prefixes == 1 && completed_product_prefixes > 0,
      "必须真实覆盖已算A但真验未完，以及唯一eligible末预算边界");
  {
    const wide::Layout large_layout{wide::budget_batch + 1, 0, 0};
    std::vector<double> large_rhs(large_layout.dimension(), 10);
    large_rhs.back() = 0;
    const auto large_inputs = forcing_inputs(large_layout, large_rhs);
    auto large_initial = large_rhs;
    large_initial[0] = 9;
    const wide::Preconditioner large_p = [&](std::span<const double>, bool refinement) {
      require(!refinement, "跨批次初始真验不应启动额外P");
      return large_initial;
    };
    wide::Work full;
    int phase_checks = 0;
    forcing_solve(large_layout, large_rhs, large_inputs, large_p, forcing_identity, full, [&] {
      if (full.products_completed == 1 && full.true_checks_completed == 0) ++phase_checks;
    });
    forcing_returned(full);
    require(phase_checks >= 7, "跨4096坐标须覆盖乘积后及真实残差两批预算");
    wide::Work work;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = forcing_solve(large_layout, large_rhs, large_inputs, large_p, forcing_identity, work, [&] {
        if (work.products_completed == 1 && work.true_checks_completed == 0 && ++calls == phase_checks - 1)
          throw CallbackFailure{4096};
      });
    } catch (const CallbackFailure& error) { original = error.value == 4096; }
    forcing_bounded(work);
    require(original && output == WideVector({101, 202}) && work.preconditioner_completed == 1 &&
        work.products_completed == 1 && work.true_checks_started == 1 && work.true_checks_completed == 0 &&
        work.initial_forcing && !work.initial_forcing->defect_norms && !work.initial_forcing->measure &&
        work.initial_forcing_checks == 0 && work.initial_forcing_returns == 0,
        "已有首批真实缺陷而第二批预算中断时仍不能发布半份E5");
  }
  {
    wide::Work work;
    bool original = false;
    try {
      forcing_solve(layout, rhs, inputs,
          [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); },
          forcing_identity, work, [&] {
            if (work.initial_forcing && std::string_view(work.initial_forcing->decision) == "strict_initial")
              throw CallbackFailure{1};
          });
    } catch (const CallbackFailure& error) { original = error.value == 1; }
    forcing_bounded(work);
    require(original && std::string_view(work.status) == "exception" && work.initial_forcing_checks == 0 &&
        work.initial_forcing_returns == 0 && work.initial_forcing->defect_norms,
        "严格初P的末预算中断也保完整E5，但不能误算forcing或accepted");
  }
  details.push_back("initial_forcing_budget_partial_snapshot");
}

void initial_forcing_invalid_norms_callbacks() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  const wide::Preconditioner identity_p = [](std::span<const double> input, bool) {
    return std::vector<double>(input.begin(), input.end());
  };
  for (std::size_t target = 0; target < 3; ++target) {
    for (double invalid : {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      auto bad = inputs;
      bad.initial.rhs_norms[target] = invalid;
      wide::Work work;
      bool rejected = false;
      try { forcing_solve(layout, rhs, bad, identity_p, forcing_identity, work); }
      catch (const std::invalid_argument&) { rejected = true; }
      forcing_bounded(work);
      require(rejected && work.preconditioner_started == 0 && work.products_started == 0 &&
          work.initial_forcing && !work.initial_forcing->defect_norms && work.initial_forcing_checks == 0 &&
          work.initial_forcing_returns == 0, "任一非法B必须在P/A之前拒绝");
    }
  }
  for (int failure = 0; failure < 7; ++failure) {
    wide::Work work;
    bool original = false, invalid = false, numerical = false;
    try {
      wide::solve(layout, rhs, inputs.tolerance,
          [&](std::span<const double> input, bool) {
            if (failure == 0) throw CallbackFailure{71};
            if (failure == 2) return std::vector<double>{0};
            auto result = std::vector<double>(input.begin(), input.end());
            if (failure == 4) result[0] = std::numeric_limits<double>::quiet_NaN();
            return result;
          }, [&](std::span<const double>, WideVector&) { throw CallbackFailure{99}; },
          [&](std::span<const long double> input, WideVector& output) {
            if (failure == 1) throw CallbackFailure{71};
            if (failure == 3) { output = {0}; return; }
            output.assign(input.begin(), input.end());
            if (failure == 5) output[0] = std::numeric_limits<long double>::quiet_NaN();
            if (failure == 6) output[0] = std::numeric_limits<long double>::infinity();
          }, work, {}, &inputs.initial);
    } catch (const CallbackFailure& error) { original = error.value == 71; }
      catch (const std::invalid_argument&) { invalid = true; }
      catch (const wide::Failure&) { numerical = true; }
    forcing_bounded(work);
    require((failure <= 1 ? original : failure <= 3 ? invalid : numerical) &&
        work.initial_forcing && !work.initial_forcing->defect_norms && !work.initial_forcing->measure &&
        work.initial_forcing_checks == 0 && work.initial_forcing_returns == 0,
        "初P或初始A的原回调异常、维数和非有限值不得留下半E或forcing成功");
    require(work.preconditioner_started == 1 && work.preconditioner_completed == (failure == 0 ? 0 : 1) &&
        work.products_started == ((failure == 0 || failure == 2 || failure == 4) ? 0 : 1) &&
        work.products_completed == ((failure == 0 || failure == 1 || failure == 2 || failure == 4) ? 0 : 1),
        "forcing回调失败的实际P/A启动与完成须精确保留");
  }
  {
    wide::Work work;
    bool original = false;
    try {
      wide::solve(layout, rhs, inputs.tolerance,
          [&](std::span<const double> input, bool refinement) {
            auto result = std::vector<double>(input.begin(), input.end());
            if (!refinement) result[0] = 8;
            return result;
          }, [](std::span<const double>, WideVector&) { throw CallbackFailure{72}; },
          [&](std::span<const long double> input, WideVector& output) { output.assign(input.begin(), input.end()); },
          work, {}, &inputs.initial);
    } catch (const CallbackFailure& error) { original = error.value == 72; }
    forcing_bounded(work);
    require(original && std::string_view(work.status) == "exception" &&
        work.initial_forcing_checks == 1 && work.initial_forcing_returns == 0 &&
        work.initial_forcing->defect_norms && work.initial_forcing->measure == 2 &&
        std::string_view(work.initial_forcing->decision) == "rejected" &&
        work.preconditioner_completed == 2 && work.products_started == 2 && work.products_completed == 1 &&
        work.true_checks_completed == 1 && work.krylov_started == 1 && work.krylov_completed == 0,
        "拒初P后的列A异常须保完整初始快照及原FG实际工作前缀");
  }
  details.push_back("initial_forcing_invalid_norms_callbacks");
}
}  // 匿名命名空间

int main() {
  try {
    exact_initial(); variable_preconditioner(); independent_blocks(); rounded_candidate_must_fail();
    fixed_exhaustion(); invalid_inputs_and_callbacks(); budgets(); pure_scalar_search();
    nominal_energy_identity(); stored_energy_contract(); energy_budget_boundaries();
    wide_exact_initial(); wide_variable_preconditioner(); wide_independent_blocks();
    wide_happy_breakdown_must_fail(); wide_fixed_exhaustion(); wide_invalid_inputs_and_callbacks();
    wide_budgets(); wide_pure_scalar_search(); wide_late_narrow_bridge(); wide_dual_products(); wide_promotion_budget();
    residual_gap_improvement_restarts(); residual_gap_without_improvement_keeps_basis(); residual_gap_restart_budget_and_caps();
    initial_forcing_strict_precedence(); initial_forcing_linear_boundaries(); initial_forcing_complementarity_stays_strict();
    initial_forcing_empty_zero_extreme_rhs(); initial_forcing_rejection_strict_fallback(); initial_forcing_only_initial_point();
    initial_forcing_budget_partial_snapshot(); initial_forcing_invalid_norms_callbacks();
    static_assert(coupled::max_window == wide::max_window && coupled::max_cycles == wide::max_cycles);
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"cases\":%zu,\"default_limits\":{"
        "\"max_window\":%zu,\"max_cycles\":%d},\"details\":[", details.size(), wide::max_window, wide::max_cycles);
    for (std::size_t i = 0; i < details.size(); ++i) std::printf("%s\"%s\"", i ? "," : "", details[i]);
    std::puts("]}");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "联立Newton独立核检查失败（已完成%zu组）：%s\n", details.size(), error.what());
    return 1;
  }
}