#include "solver/experimental/legacy_coupled.hpp"
#include "solver/experimental/energy.hpp"

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

#include "solver/experimental/coupled.hpp"

namespace {
namespace wide = solver::v100_coupled;
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

namespace {
struct CandidateFixture {
  wide::Layout layout;
  std::vector<double> rhs, initial, step;
  ForcingInputs inputs;
};

struct CandidateTrace {
  const long double* last_data = nullptr;
  WideVector last_value;
  int column_calls = 0, true_calls = 0;
};

CandidateFixture candidate_fixture(const wide::Layout& layout, const std::vector<double>& rhs,
    const std::vector<double>& target) {
  require(rhs.size() == layout.dimension() && rhs.size() == target.size() && rhs.back() == 0,
      "候选控制夹具须保完整右端与零tau互补右端");
  CandidateFixture result{layout, rhs, rhs, std::vector<double>(rhs.size()), forcing_inputs(layout, rhs)};
  result.initial.back() = -2 * result.inputs.tolerance[4];
  for (std::size_t i = 0; i < rhs.size(); ++i)
    result.step[i] = static_cast<double>(static_cast<long double>(target[i]) - result.initial[i]);
  require(result.step.back() != 0, "候选控制夹具的列枢轴必须非零");
  return result;
}

WideVector candidate_expected(const CandidateFixture& fixture) {
  WideVector result(fixture.rhs.size());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<long double>(fixture.initial[i]) + fixture.step[i];
  return result;
}

std::array<long double, 5> candidate_identity_defects(const CandidateFixture& fixture,
    std::span<const long double> point) {
  std::array<long double, 5> result{};
  for (std::size_t i = 0; i < point.size(); ++i) {
    const auto block = fixture.layout.block(i);
    result[block] = std::max(result[block], std::abs(static_cast<long double>(fixture.rhs[i]) - point[i]));
  }
  return result;
}

WideVector candidate_fixture_solve(const CandidateFixture& fixture, wide::Work& work,
    CandidateTrace& trace, const wide::Budget& budget = {}, bool allow_candidates = false) {
  // 固定线性rank1列算子人为制造误差，仅用于候选gate分支；真实物理算子始终为I。
  return wide::solve(fixture.layout, fixture.rhs, fixture.inputs.tolerance,
      [&](std::span<const double>, bool refinement) { return refinement ? fixture.step : fixture.initial; },
      [&](std::span<const double> input, WideVector& output) {
        ++trace.column_calls;
        const long double ratio = static_cast<long double>(input.back()) / fixture.step.back();
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
          output[i] = (static_cast<long double>(fixture.rhs[i]) - fixture.initial[i]) * ratio;
      }, [&](std::span<const long double> input, WideVector& output) {
        ++trace.true_calls;
        trace.last_data = input.data();
        trace.last_value.assign(input.begin(), input.end());
        output.assign(input.begin(), input.end());
      }, work, budget, &fixture.inputs.initial, allow_candidates);
}

void candidate_bounded(const wide::Work& work) {
  forcing_bounded(work);
  require(work.candidate_forcing_returns >= 0 && work.candidate_forcing_returns <= work.candidate_forcing_checks &&
      work.candidate_forcing_checks <= work.krylov_completed &&
      work.initial_forcing_checks + work.candidate_forcing_checks <= work.true_checks_completed &&
      work.initial_forcing_returns + work.candidate_forcing_returns <= 1,
      "候选gate只记已有完整FG真验，每个方向至多一种非严格返回");
  if (!work.candidate_forcing) {
    require(work.candidate_forcing_checks == 0 && work.candidate_forcing_returns == 0,
        "候选默认关闭时不得留下候选检查或返回");
    return;
  }
  const auto& proof = *work.candidate_forcing;
  if (std::string_view(proof.decision) == "not_checked") {
    require(proof.cycle == 0 && proof.krylov_step == 0 && !proof.defect_norms && !proof.strict_measure &&
        !proof.best_measure_before && !proof.measure && work.candidate_forcing_checks == 0,
        "尚未完成候选gate时必须保空证明，不得发布半E");
    return;
  }
  require(proof.cycle >= 1 && proof.cycle <= work.cycles_started && proof.krylov_step >= proof.cycle &&
      proof.krylov_step <= static_cast<wide::Index>(work.window) * proof.cycle &&
      proof.krylov_step <= work.krylov_completed && proof.defect_norms && proof.strict_measure &&
      proof.best_measure_before && proof.measure && work.candidate_forcing_checks >= 1 &&
      *proof.strict_measure > 1 && *proof.best_measure_before > 1 && std::isfinite(*proof.measure),
      "完整候选证明必须绑定真实cycle/列与同一份严格失败真验");
  if (std::string_view(proof.decision) == "rejected")
    require(*proof.measure > 1 && work.candidate_forcing_returns == 0,
        "拒绝证明保真实LD界外事实，不能被后来未完成或strict候选覆盖");
  else
    require((std::string_view(proof.decision) == "eligible" || std::string_view(proof.decision) == "returned") &&
        *proof.measure <= 1, "候选仅在相同五块forcing界内可eligible或返回");
}

void candidate_returned(const wide::Work& work) {
  candidate_bounded(work);
  require(std::string_view(work.status) == "forced_candidate" && work.candidate_forcing &&
      work.candidate_forcing_returns == 1 && work.initial_forcing_returns == 0 &&
      work.initial_forcing && std::string_view(work.initial_forcing->decision) == "rejected" &&
      work.krylov_completed >= 1 && work.cycles_started >= 1 &&
      work.preconditioner_started == work.preconditioner_completed &&
      work.preconditioner_completed == 1 + work.krylov_completed &&
      work.products_started == work.products_completed &&
      work.products_completed == work.true_checks_completed + work.krylov_completed &&
      work.true_checks_started == work.true_checks_completed,
      "候选非严格返回须有完整P/A账本且初始gate已拒绝");
  const auto& proof = *work.candidate_forcing;
  require(std::string_view(proof.decision) == "returned" && proof.krylov_step == work.krylov_completed &&
      proof.cycle == work.cycles_started && work.final_measure == proof.strict_measure,
      "候选返回后的最终摘要必须是本次交付点，允许严格残差比原best更差");
  long double strict_max = 0, forcing_max = 0;
  for (std::size_t block = 0; block < 5; ++block) {
    const long double defect = (*proof.defect_norms)[block];
    const long double strict = defect / work.tolerances[block];
    require(strict == work.block_measures[block], "返回的五块严格摘要须属于同一候选绝对缺陷");
    strict_max = std::max(strict_max, strict);
    long double limit = work.tolerances[block];
    if (block < 3) limit = std::max(limit, 0.1L * static_cast<long double>(work.initial_forcing->rhs_norms[block]));
    require(defect <= limit, "候选返回使用初始实际B3及同一三线性/两互补界");
    forcing_max = std::max(forcing_max, defect / limit);
  }
  require(strict_max == *proof.strict_measure && forcing_max == *proof.measure &&
      work.block_measures[3] <= 1 && work.block_measures[4] <= 1,
      "候选严格量度及qF必须来自交付向量，互补块始终strict");
}

void candidate_forcing_strict_precedence() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto fixture = candidate_fixture(layout, rhs, rhs);
  wide::Work work;
  CandidateTrace trace;
  const auto answer = candidate_fixture_solve(fixture, work, trace, {}, true);
  wide_accepted(work);
  candidate_bounded(work);
  require(answer == WideVector(rhs.begin(), rhs.end()) && answer.data() == trace.last_data &&
      work.krylov_completed == 1 && work.preconditioner_completed == 2 && work.products_completed == 3 &&
      work.true_checks_completed == 2 && work.initial_forcing_checks == 1 &&
      work.candidate_forcing_checks == 0 && work.candidate_forcing_returns == 0 && work.candidate_forcing &&
      std::string_view(work.candidate_forcing->decision) == "not_checked" && work.initial_forcing->measure == 2,
      "已有FG候选完整strict过线必须优先accepted，不另记候选forcing或覆盖初始证明");
  details.push_back("candidate_forcing_strict_precedence");
}

void candidate_forcing_same_bounds() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  for (std::size_t target_block = 0; target_block < 5; ++target_block) {
    for (bool outside : {false, true}) {
      auto target = rhs;
      if (target_block < 3) target[target_block] = outside ? std::nextafter(9.0, 0.0) : 9.0;
      else {
        target[0] = 9.5;
        const double strict = inputs.tolerance[target_block];
        target[target_block] = -(outside ? std::nextafter(strict, std::numeric_limits<double>::infinity()) : strict);
      }
      const auto fixture = candidate_fixture(layout, rhs, target);
      wide::Work work;
      CandidateTrace trace;
      WideVector answer;
      bool exhausted = false;
      try { answer = candidate_fixture_solve(fixture, work, trace, {}, true); }
      catch (const wide::Failure&) { exhausted = true; }
      candidate_bounded(work);
      require(work.initial_forcing && work.initial_forcing->defect_norms &&
          *work.initial_forcing->defect_norms == forcing_identity_defects(layout, rhs, fixture.initial) &&
          work.initial_forcing->measure == 2 && work.initial_forcing_returns == 0,
          "候选边界检查始终保留初始E5/qF/decision，不改B3来源");
      if (!outside) {
        candidate_returned(work);
        require(!exhausted && answer == candidate_expected(fixture) && answer == trace.last_value &&
            answer.data() == trace.last_data && work.candidate_forcing->measure == 1 &&
            *work.candidate_forcing->defect_norms == candidate_identity_defects(fixture, answer),
            "三个线性F及两个互补tau的等号都须交付真正已验候选");
      } else {
        require(exhausted && std::string_view(work.status) == "exhausted" &&
            work.cycles_started == 3 && work.krylov_completed == 3 && work.preconditioner_completed == 4 &&
            work.products_completed == 7 && work.true_checks_completed == 4 &&
            work.candidate_forcing_checks == 3 && work.candidate_forcing_returns == 0 &&
            work.candidate_forcing && std::string_view(work.candidate_forcing->decision) == "rejected" &&
            work.candidate_forcing->measure && *work.candidate_forcing->measure > 1 && work.final_measure == 2,
            "任一界外一个double值都不得返回，仍按原三周期及最佳strict点失败");
      }
    }
  }
  details.push_back("candidate_forcing_same_bounds");
}

void candidate_forcing_actual_nonbest_direction() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  for (bool improved : {false, true}) {
    auto target = rhs;
    target[0] = 10.5;
    auto fixture = candidate_fixture(layout, rhs, target);
    if (improved) {
      fixture.initial = rhs;
      fixture.initial[0] = 8;
      for (std::size_t i = 0; i < rhs.size(); ++i)
        fixture.step[i] = static_cast<double>(static_cast<long double>(target[i]) - fixture.initial[i]);
      // 此变体列枢轴为dual0，仍是把固定z映为原r0的线性rank1人工误差。
    }
    wide::Work work;
    CandidateTrace trace;
    WideVector answer;
    if (!improved) answer = candidate_fixture_solve(fixture, work, trace, {}, true);
    else {
      answer = wide::solve(layout, rhs, fixture.inputs.tolerance,
          [&](std::span<const double>, bool refinement) { return refinement ? fixture.step : fixture.initial; },
          [&](std::span<const double> input, WideVector& output) {
            ++trace.column_calls;
            output.assign(input.size(), 0);
            output[0] = static_cast<long double>(input[0]) * 4 / 5;
          }, [&](std::span<const long double> input, WideVector& output) {
            ++trace.true_calls;
            trace.last_data = input.data(); trace.last_value.assign(input.begin(), input.end());
            output.assign(input.begin(), input.end());
          }, work, {}, &fixture.inputs.initial, true);
    }
    candidate_returned(work);
    require(answer == candidate_expected(fixture) && answer == trace.last_value && answer.data() == trace.last_data &&
        work.preconditioner_completed == 2 && work.products_completed == 3 && work.true_checks_completed == 2 &&
        work.krylov_completed == 1 && work.cycles_started == 1 && work.candidate_forcing_checks == 1 &&
        *work.candidate_forcing->defect_norms == candidate_identity_defects(fixture, answer),
        "无论swap与否都须std::move最后真验所见的同一向量，不能复制或偷换成旧best");
    require(improved ? *work.candidate_forcing->strict_measure < *work.candidate_forcing->best_measure_before :
        *work.candidate_forcing->strict_measure > *work.candidate_forcing->best_measure_before,
        "分别覆盖候选strict改善与候选strict更差但同界合法的两个实际分支");
    if (!improved) require(work.final_measure && work.initial_measure && *work.final_measure > *work.initial_measure,
        "非best合法交付不能沿用final不大于initial的旧充分条件");
  }
  details.push_back("candidate_forcing_actual_nonbest_direction");
}

void candidate_forcing_default_and_invalid_policy() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  auto target = rhs;
  target[0] = 10.5;
  const auto fixture = candidate_fixture(layout, rhs, target);
  wide::Work work;
  CandidateTrace trace;
  bool exhausted = false;
  try { candidate_fixture_solve(fixture, work, trace); }
  catch (const wide::Failure&) { exhausted = true; }
  candidate_bounded(work);
  require(exhausted && std::string_view(work.status) == "exhausted" && !work.candidate_forcing &&
      work.candidate_forcing_checks == 0 && work.candidate_forcing_returns == 0 && work.final_measure == 2,
      "默认initial-only须保父33含义，不能让原控制调用暗中启用候选gate");
  int p_calls = 0, a_calls = 0;
  bool invalid = false;
  try {
    wide::solve(layout, rhs, fixture.inputs.tolerance,
        [&](std::span<const double> input, bool) { ++p_calls; return std::vector<double>(input.begin(), input.end()); },
        [&](std::span<const double>, WideVector&) { ++a_calls; },
        [&](std::span<const long double>, WideVector&) { ++a_calls; }, work, {}, nullptr, true);
  } catch (const std::invalid_argument&) { invalid = true; }
  candidate_bounded(work);
  require(invalid && p_calls == 0 && a_calls == 0 && work.preconditioner_started == 0 &&
      work.products_started == 0 && !work.initial_forcing && work.candidate_forcing &&
      std::string_view(work.candidate_forcing->decision) == "not_checked",
      "无实际B3却启用候选gate必须在P/A前拒绝");
  const auto identity_p = [](std::span<const double> input, bool) { return std::vector<double>(input.begin(), input.end()); };
  const auto strict_answer = wide::solve(layout, rhs, fixture.inputs.tolerance, identity_p,
      forcing_identity, forcing_identity, work);
  wide_accepted(work);
  candidate_bounded(work);
  require(strict_answer == WideVector(rhs.begin(), rhs.end()) && !work.initial_forcing && !work.candidate_forcing,
      "完整nullptr默认仍是纯strict API");
  forcing_solve(layout, rhs, fixture.inputs,
      [&](std::span<const double>, bool refinement) {
        require(!refinement, "初始forcing已通过时不许进入FG");
        auto initial = rhs; initial[0] = 9.5; return initial;
      }, forcing_identity, work);
  forcing_returned(work);
  require(!work.candidate_forcing, "旧forcing helper不传新bool时仍只启用初P");
  const wide::Preconditioner initially_eligible = [&](std::span<const double>, bool refinement) {
    require(!refinement, "候选开关不能抢走已通过的初始forcing返回");
    auto initial = rhs; initial[0] = 9.5; return initial;
  };
  wide::solve(layout, rhs, fixture.inputs.tolerance, initially_eligible, forcing_identity, forcing_identity,
      work, {}, &fixture.inputs.initial, true);
  forcing_returned(work);
  candidate_bounded(work);
  require(work.candidate_forcing && std::string_view(work.candidate_forcing->decision) == "not_checked" &&
      work.candidate_forcing_checks == 0 && work.candidate_forcing_returns == 0,
      "显式候选能力可空闲，初始完整证明仍保持父4键语义");
  details.push_back("candidate_forcing_default_and_invalid_policy");
}

void candidate_forcing_rejection_snapshot_and_caps() {
  {
    const wide::Layout layout{100, 0, 0};
    std::vector<double> rhs(layout.dimension(), 0);
    rhs[0] = 1;
    const auto inputs = forcing_inputs(layout, rhs);
    wide::Work work;
    bool exhausted = false;
    const auto shift = [](const auto& input, WideVector& output) {
      output.assign(input.size(), 0);
      for (std::size_t i = 0; i < input.size(); ++i) output[(i + 1) % input.size()] = input[i];
    };
    try {
      wide::solve(layout, rhs, inputs.tolerance,
          [](std::span<const double> input, bool refinement) {
            return refinement ? std::vector<double>(input.begin(), input.end()) : std::vector<double>(input.size(), 0);
          }, [&](std::span<const double> input, WideVector& output) { shift(input, output); },
          [&](std::span<const long double> input, WideVector& output) { shift(input, output); },
          work, {}, &inputs.initial, true);
    } catch (const wide::Failure&) { exhausted = true; }
    candidate_bounded(work);
    require(exhausted && std::string_view(work.status) == "exhausted" && work.window == 32 &&
        work.cycles_started == 3 && work.krylov_started == 96 && work.krylov_completed == 96 &&
        work.preconditioner_completed == 97 && work.products_completed == 100 && work.true_checks_completed == 4 &&
        work.initial_forcing_checks == 1 && work.candidate_forcing_checks == 3 && work.candidate_forcing_returns == 0 &&
        work.candidate_forcing && work.candidate_forcing->cycle == 3 && work.candidate_forcing->krylov_step == 96 &&
        std::string_view(work.candidate_forcing->decision) == "rejected" && work.final_measure == work.initial_measure &&
        work.initial_forcing->defect_norms && (*work.initial_forcing->defect_norms)[0] == 1,
        "真实循环移位算子保3乘32及原窗口末真验，不为gate增加P/A或检查时点");
  }
  for (bool fail_later_true : {false, true}) {
    const wide::Layout layout{1, 1, 1};
    const std::vector<double> rhs{10, 10, 10, 0, 0};
    auto first_target = rhs; first_target[0] = 8;
    const auto fixture = candidate_fixture(layout, rhs, first_target);
    auto strict_step = rhs;
    for (std::size_t i = 0; i < rhs.size(); ++i)
      strict_step[i] = static_cast<double>(static_cast<long double>(rhs[i]) - fixture.initial[i]);
    wide::Work work;
    int refinement_calls = 0, true_calls = 0;
    bool original = false;
    WideVector answer;
    try {
      answer = wide::solve(layout, rhs, fixture.inputs.tolerance,
          [&](std::span<const double>, bool refinement) {
            if (!refinement) return fixture.initial;
            return ++refinement_calls == 1 ? fixture.step : strict_step;
          }, [&](std::span<const double> input, WideVector& output) {
            output.assign(input.size(), 0);
            output.back() = 2 * static_cast<long double>(fixture.inputs.tolerance[4]) *
                (static_cast<long double>(input.back()) / strict_step.back());
          }, [&](std::span<const long double> input, WideVector& output) {
            ++true_calls;
            if (fail_later_true && true_calls == 3) throw CallbackFailure{73};
            output.assign(input.begin(), input.end());
          }, work, {}, &fixture.inputs.initial, true);
    } catch (const CallbackFailure& error) { original = error.value == 73; }
    candidate_bounded(work);
    require(work.candidate_forcing && std::string_view(work.candidate_forcing->decision) == "rejected" &&
        work.candidate_forcing->krylov_step == 1 && work.candidate_forcing_checks == 1 &&
        work.candidate_forcing_returns == 0 && work.candidate_forcing->defect_norms &&
        (*work.candidate_forcing->defect_norms)[0] == 2 && work.initial_forcing->defect_norms &&
        (*work.initial_forcing->defect_norms)[4] == 2 * static_cast<long double>(fixture.inputs.tolerance[4]),
        "后续strict或未完成A不能覆盖上次完整rejected候选证明，也不改初始E5");
    if (!fail_later_true) {
      wide_accepted(work);
      require(!original && answer == WideVector(rhs.begin(), rhs.end()) && work.krylov_completed == 2 &&
          work.true_checks_completed == 3 && work.candidate_forcing->krylov_step < work.krylov_completed,
          "后来strict候选优先返回，不为其新增候选gate");
    } else {
      require(original && std::string_view(work.status) == "exception" && work.final_measure == 2 &&
          work.krylov_completed == 2 && work.preconditioner_completed == 3 && work.products_started == 5 &&
          work.products_completed == 4 && work.true_checks_started == 3 && work.true_checks_completed == 2,
          "后来候选A原异常保精确P/A前缀、旧完整候选证明及best摘要");
    }
  }
  details.push_back("candidate_forcing_rejection_snapshot_and_caps");
}

void candidate_forcing_budget_partial_snapshot() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  auto target = rhs; target[0] = 10.5;
  const auto fixture = candidate_fixture(layout, rhs, target);
  wide::Work baseline;
  CandidateTrace baseline_trace;
  int total = 0;
  candidate_fixture_solve(fixture, baseline, baseline_trace, [&] { ++total; }, true);
  candidate_returned(baseline);
  int eligible_prefixes = 0, candidate_product_prefixes = 0;
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    wide::Work work;
    CandidateTrace trace;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = candidate_fixture_solve(fixture, work, trace, [&] {
        if (++calls == cutoff) throw CallbackFailure{cutoff};
      }, true);
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    candidate_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        std::string_view(work.status) == "exception" && work.candidate_forcing &&
        work.candidate_forcing_returns == 0 && work.initial_forcing_returns == 0,
        "逐真实预算入口中断必须保原异常且不伪造任一种非严格交付");
    if (work.candidate_forcing_checks == 0) {
      require(std::string_view(work.candidate_forcing->decision) == "not_checked" &&
          !work.candidate_forcing->defect_norms && !work.candidate_forcing->measure,
          "候选完整gate之前不得发布候选半E或qF");
      if (work.products_completed == 3 && work.true_checks_completed == 1) ++candidate_product_prefixes;
    } else {
      ++eligible_prefixes;
      require(work.candidate_forcing_checks == 1 && std::string_view(work.candidate_forcing->decision) == "eligible" &&
          work.candidate_forcing->defect_norms == baseline.candidate_forcing->defect_norms &&
          work.candidate_forcing->measure == baseline.candidate_forcing->measure &&
          work.final_measure == work.candidate_forcing->best_measure_before && work.final_measure == 2 &&
          work.candidate_forcing->strict_measure && *work.candidate_forcing->strict_measure > *work.final_measure,
          "eligible末预算未过时须保strict最佳摘要，不能提前切到更差但合法的候选");
    }
    if (work.initial_forcing_checks == 1)
      require(work.initial_forcing->defect_norms == baseline.initial_forcing->defect_norms &&
          work.initial_forcing->measure == baseline.initial_forcing->measure &&
          std::string_view(work.initial_forcing->decision) == "rejected", "候选预算阶段不改初始证明");
  }
  require(eligible_prefixes == 1 && candidate_product_prefixes > 0,
      "必须覆盖已有候选A但真验未完，以及唯一候选eligible末预算");
  {
    const wide::Layout large_layout{wide::budget_batch + 1, 0, 0};
    std::vector<double> large_rhs(large_layout.dimension(), 10); large_rhs.back() = 0;
    auto large_target = large_rhs; large_target[0] = 10.5;
    const auto large = candidate_fixture(large_layout, large_rhs, large_target);
    wide::Work full;
    CandidateTrace full_trace;
    int phase_checks = 0;
    candidate_fixture_solve(large, full, full_trace, [&] {
      if (full.products_completed == 3 && full.true_checks_completed == 1) ++phase_checks;
    }, true);
    candidate_returned(full);
    require(phase_checks >= 7, "跨4096候选必须走实际乘积后与两批残差预算");
    wide::Work work;
    CandidateTrace trace;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = candidate_fixture_solve(large, work, trace, [&] {
        if (work.products_completed == 3 && work.true_checks_completed == 1 && ++calls == phase_checks - 1)
          throw CallbackFailure{4097};
      }, true);
    } catch (const CallbackFailure& error) { original = error.value == 4097; }
    candidate_bounded(work);
    require(original && output == WideVector({101, 202}) && work.preconditioner_completed == 2 &&
        work.products_completed == 3 && work.true_checks_started == 2 && work.true_checks_completed == 1 &&
        work.candidate_forcing_checks == 0 && work.candidate_forcing_returns == 0 && work.candidate_forcing &&
        !work.candidate_forcing->defect_norms && !work.candidate_forcing->strict_measure &&
        work.final_measure == 2 && work.initial_forcing->defect_norms,
        "候选首批E已在局部累积也不能因第二批预算中断发布半份证明");
  }
  {
    const auto strict = candidate_fixture(layout, rhs, rhs);
    wide::Work work;
    CandidateTrace trace;
    bool original = false;
    try {
      candidate_fixture_solve(strict, work, trace, [&] {
        if (work.krylov_completed == 1 && work.final_measure && *work.final_measure <= 1)
          throw CallbackFailure{74};
      }, true);
    } catch (const CallbackFailure& error) { original = error.value == 74; }
    candidate_bounded(work);
    require(original && std::string_view(work.status) == "exception" && work.candidate_forcing_checks == 0 &&
        work.candidate_forcing_returns == 0 && work.final_measure && *work.final_measure <= 1,
        "strict候选末预算中断不提前accepted，也不误走候选forcing");
  }
  details.push_back("candidate_forcing_budget_partial_snapshot");
}
}  // 匿名命名空间
namespace {
struct ForcingWeightTrace {
  std::vector<std::vector<double>> physical_inputs;
  const long double* last_data = nullptr;
  WideVector last_value;
};

WideVector forcing_weight_identity_solve(wide::Work& work, ForcingWeightTrace& trace,
    std::optional<bool> forcing_weights = std::nullopt, const wide::Budget& budget = {}) {
  const wide::Layout layout{1, 0, 0};
  const std::vector<double> rhs{10, 10, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  const wide::Preconditioner p = [&](std::span<const double> input, bool refinement) {
    if (!refinement) return std::vector<double>(input.size(), 0);
    trace.physical_inputs.emplace_back(input.begin(), input.end());
    auto result = std::vector<double>(input.begin(), input.end());
    result[1] *= 1.1;
    return result;
  };
  const wide::ProductDouble column = [](std::span<const double> input, WideVector& output) {
    output.assign(input.begin(), input.end());
  };
  const wide::ProductWide truth = [&](std::span<const long double> input, WideVector& output) {
    trace.last_data = input.data(); trace.last_value.assign(input.begin(), input.end());
    output.assign(input.begin(), input.end());
  };
  if (forcing_weights)
    return wide::solve(layout, rhs, inputs.tolerance, p, column, truth, work, budget, &inputs.initial,
        true, *forcing_weights);
  return wide::solve(layout, rhs, inputs.tolerance, p, column, truth, work, budget, &inputs.initial, true);
}

WideVector forcing_weight_candidate_solve(const CandidateFixture& fixture, wide::Work& work,
    CandidateTrace& trace, const wide::Budget& budget = {}) {
  // 与父候选夹具同一个固定线性rank1人工列误差；只额外启用F搜索尺度。
  return wide::solve(fixture.layout, fixture.rhs, fixture.inputs.tolerance,
      [&](std::span<const double>, bool refinement) { return refinement ? fixture.step : fixture.initial; },
      [&](std::span<const double> input, WideVector& output) {
        ++trace.column_calls;
        const long double ratio = static_cast<long double>(input.back()) / fixture.step.back();
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
          output[i] = (static_cast<long double>(fixture.rhs[i]) - fixture.initial[i]) * ratio;
      }, [&](std::span<const long double> input, WideVector& output) {
        ++trace.true_calls;
        trace.last_data = input.data(); trace.last_value.assign(input.begin(), input.end());
        output.assign(input.begin(), input.end());
      }, work, budget, &fixture.inputs.initial, true, true);
}

void forcing_weights_early_candidate() {
  wide::Work old_work, work;
  ForcingWeightTrace old_trace, trace;
  forcing_weight_identity_solve(old_work, old_trace, false);
  wide_accepted(old_work);
  candidate_bounded(old_work);
  const auto answer = forcing_weight_identity_solve(work, trace, true);
  candidate_returned(work);
  require(old_work.krylov_completed >= 2 && old_work.preconditioner_completed > work.preconditioner_completed &&
      work.krylov_completed == 1 && work.preconditioner_completed == 2 && work.products_completed == 3 &&
      work.true_checks_completed == 2 && work.candidate_forcing_checks == 1 && work.candidate_forcing_returns == 1 &&
      trace.physical_inputs.size() == 1 && answer == trace.last_value && answer.data() == trace.last_data,
      "同一实际I与P对角例中，F尺度首列须由原真验确认受控返回，strict尺度仍需更多列");
  const double unit = 1 / std::sqrt(2.0);
  require(std::abs(trace.physical_inputs[0][0] - unit) < 2e-15 &&
      std::abs(trace.physical_inputs[0][1] - unit) < 2e-15 && trace.physical_inputs[0][2] == 0 &&
      work.final_measure && *work.final_measure > 1 && work.candidate_forcing->measure &&
      *work.candidate_forcing->measure < 0.7L && work.block_measures[4] == 0,
      "投影估计对应允许F单位，真实strict未过事实和互补strict不能改写");
  require(std::abs(answer[0] - 10) < 1 && std::abs(answer[1] - 10) < 1 && answer[2] == 0,
      "更早候选仍须独立通过原同一F五块物理验收");
  details.push_back("forcing_weights_early_candidate");
}

void forcing_weights_consistent_physical_system() {
  const wide::Layout layout{1, 1, 1};
  const std::array<double, 5> diagonal{2, 3, 4, 5, 6};
  const std::array<double, 5> multiplier{1, 1.05, 1.1, 1, 1};
  const std::array<std::array<double, 3>, 2> linear_rhs{{{10, 20, 40}, {11, 23, 41}}};
  const std::array<std::array<long double, 5>, 5> dense{{
      {2, 1, 0, 0, 0}, {0, 3, 1, 0, 0}, {0, 0, 4, 1, 0}, {0, 0, 0, 5, 1}, {0, 0, 0, 0, 6}}};
  for (const auto& linear : linear_rhs) {
    const std::vector<double> rhs{linear[0], linear[1], linear[2], 0, 0};
    const auto inputs = forcing_inputs(layout, rhs);
    wide::Work work;
    ForcingWeightTrace trace;
    const auto operation = [&](const auto& input, WideVector& output) {
      output.resize(5);
      for (std::size_t i = 0; i < 5; ++i)
        output[i] = static_cast<long double>(diagonal[i]) * input[i] + (i + 1 < 5 ? input[i + 1] : 0);
    };
    const auto answer = wide::solve(layout, rhs, inputs.tolerance,
        [&](std::span<const double> input, bool refinement) {
          if (!refinement) return std::vector<double>(5, 0);
          trace.physical_inputs.emplace_back(input.begin(), input.end());
          std::vector<double> result(5);
          for (std::size_t i = 5; i-- > 0;) {
            long double value = static_cast<long double>(multiplier[i]) * input[i];
            if (i + 1 < 5) value -= result[i + 1];
            result[i] = static_cast<double>(value / diagonal[i]);
          }
          return result;
        }, [&](std::span<const double> input, WideVector& output) { operation(input, output); },
        [&](std::span<const long double> input, WideVector& output) {
          trace.last_data = input.data(); trace.last_value.assign(input.begin(), input.end()); operation(input, output);
        }, work, {}, &inputs.initial, true, true);
    candidate_returned(work);
    require(work.krylov_completed == 1 && trace.physical_inputs.size() == 1 &&
        answer == trace.last_value && answer.data() == trace.last_data && work.tolerances == inputs.tolerance &&
        work.initial_forcing->rhs_norms == inputs.initial.rhs_norms,
        "非单位显式矩阵也须首列按同界返回，真实tau及完整B3保持原值");
    const double beta = static_cast<double>(10 * std::sqrt(3.0L));
    const double unit = 10 / beta;
    for (std::size_t block = 0; block < 3; ++block) {
      const long double scale = std::max(static_cast<long double>(inputs.tolerance[block]),
          0.1L * static_cast<long double>(inputs.initial.rhs_norms[block]));
      const double expected = static_cast<double>(static_cast<long double>(unit) * scale);
      require(trace.physical_inputs[0][block] == expected,
          "原残差除以LD F后再将同一LD F乘回P输入，不能错用strict或提前窄scale");
    }
    require(trace.physical_inputs[0][3] == 0 && trace.physical_inputs[0][4] == 0,
        "非单位例的两个互补搜索分量保持真实零右端");
    long double sum = 0, squares = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      sum += multiplier[i]; squares += static_cast<long double>(multiplier[i]) * multiplier[i];
    }
    long double strict_max = 0, forcing_max = 0;
    std::array<long double, 5> defects{};
    for (std::size_t i = 0; i < 5; ++i) {
      long double product = 0;
      for (std::size_t j = 0; j < 5; ++j) product += dense[i][j] * answer[j];
      const long double expected = i < 3 ? static_cast<long double>(rhs[i]) * multiplier[i] * sum / squares : 0;
      require(std::abs(product - expected) <= 1e-12L * std::max(1.0L, std::abs(expected)),
          "独立dense乘积须符合单列加权GMRES解析值，交叉核image除以同一F");
      const auto block = layout.block(i);
      defects[block] = std::abs(static_cast<long double>(rhs[i]) - product);
      strict_max = std::max(strict_max, defects[block] / inputs.tolerance[block]);
      long double limit = inputs.tolerance[block];
      if (block < 3) limit = std::max(limit, 0.1L * static_cast<long double>(inputs.initial.rhs_norms[block]));
      forcing_max = std::max(forcing_max, defects[block] / limit);
    }
    require(strict_max > 1 && forcing_max <= 1 && defects[3] == 0 && defects[4] == 0 &&
        work.candidate_forcing->defect_norms && *work.candidate_forcing->defect_norms == defects,
        "独立非单位dense系统须确认真实strict失败、同forcing界成功与互补严格零缺陷");
  }
  details.push_back("forcing_weights_consistent_physical_system");
}

void forcing_weights_default_and_invalid_policy() {
  wide::Work default_work, explicit_work;
  ForcingWeightTrace default_trace, explicit_trace;
  const auto default_answer = forcing_weight_identity_solve(default_work, default_trace);
  const auto explicit_answer = forcing_weight_identity_solve(explicit_work, explicit_trace, false);
  wide_accepted(default_work); wide_accepted(explicit_work);
  candidate_bounded(default_work); candidate_bounded(explicit_work);
  require(default_answer == explicit_answer && default_trace.physical_inputs == explicit_trace.physical_inputs &&
      default_work.krylov_completed == explicit_work.krylov_completed &&
      default_work.preconditioner_completed == explicit_work.preconditioner_completed &&
      default_work.products_completed == explicit_work.products_completed &&
      default_work.candidate_forcing_checks == explicit_work.candidate_forcing_checks,
      "省略新bool与显式false须保持父候选语义、物理P输入和实际账本");
  const wide::Layout layout{1, 0, 0};
  const std::vector<double> rhs{10, 10, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  for (bool has_initial : {false, true}) {
    for (bool allow_candidates : {false, true}) {
      if (has_initial && allow_candidates) continue;
      wide::Work work;
      int p_calls = 0, a_calls = 0;
      bool rejected = false;
      try {
        wide::solve(layout, rhs, inputs.tolerance,
            [&](std::span<const double> input, bool) { ++p_calls; return std::vector<double>(input.begin(), input.end()); },
            [&](std::span<const double>, WideVector&) { ++a_calls; },
            [&](std::span<const long double>, WideVector&) { ++a_calls; }, work, {},
            has_initial ? &inputs.initial : nullptr, allow_candidates, true);
      } catch (const std::invalid_argument&) { rejected = true; }
      candidate_bounded(work);
      require(rejected && p_calls == 0 && a_calls == 0 && work.preconditioner_started == 0 &&
          work.products_started == 0 && work.initial_forcing_checks == 0 && work.candidate_forcing_checks == 0 &&
          std::string_view(work.status) == "exception",
          "新F搜索缺allow或实际B3的任一非法组合都须在P/A之前拒绝");
    }
  }
  details.push_back("forcing_weights_default_and_invalid_policy");
}

void forcing_weights_budget_and_true_gate() {
  const wide::Layout layout{1, 1, 1};
  const std::vector<double> rhs{10, 10, 10, 0, 0};
  const auto inputs = forcing_inputs(layout, rhs);
  auto bad_target = rhs; bad_target[0] = 10.5; bad_target[3] = -2 * inputs.tolerance[3];
  const auto bad = candidate_fixture(layout, rhs, bad_target);
  wide::Work bad_work;
  CandidateTrace bad_trace;
  bool exhausted = false;
  try { forcing_weight_candidate_solve(bad, bad_work, bad_trace); }
  catch (const wide::Failure&) { exhausted = true; }
  candidate_bounded(bad_work);
  require(exhausted && std::string_view(bad_work.status) == "exhausted" &&
      bad_work.cycles_started == 3 && bad_work.krylov_completed == 3 && bad_work.preconditioner_completed == 4 &&
      bad_work.products_completed == 7 && bad_work.true_checks_completed == 4 &&
      bad_work.candidate_forcing_checks == 3 && bad_work.candidate_forcing_returns == 0 &&
      bad_work.candidate_forcing && bad_work.candidate_forcing->defect_norms &&
      (*bad_work.candidate_forcing->defect_norms)[3] == 2 * static_cast<long double>(inputs.tolerance[3]) &&
      bad_work.candidate_forcing->measure == 2,
      "即使人工列估计精确为零，真实互补越strict仍须拒绝，F搜索不是验收证书");
  auto target = rhs; target[0] = 10.5;
  const auto fixture = candidate_fixture(layout, rhs, target);
  wide::Work baseline;
  CandidateTrace baseline_trace;
  int total = 0;
  forcing_weight_candidate_solve(fixture, baseline, baseline_trace, [&] { ++total; });
  candidate_returned(baseline);
  int eligible_prefixes = 0, incomplete_candidate_prefixes = 0;
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    wide::Work work;
    CandidateTrace trace;
    int calls = 0;
    bool original = false;
    WideVector output{101, 202};
    try {
      output = forcing_weight_candidate_solve(fixture, work, trace, [&] {
        if (++calls == cutoff) throw CallbackFailure{cutoff};
      });
    } catch (const CallbackFailure& error) { original = error.value == cutoff; }
    candidate_bounded(work);
    require(original && calls == cutoff && output == WideVector({101, 202}) &&
        std::string_view(work.status) == "exception" && work.candidate_forcing &&
        work.initial_forcing_returns == 0 && work.candidate_forcing_returns == 0,
        "F搜索全部真实预算入口须保原异常、不提前交付且不添P/A");
    if (work.candidate_forcing_checks == 0) {
      require(!work.candidate_forcing->defect_norms && !work.candidate_forcing->measure,
          "F搜索真验partial不能发布半候选证明");
      if (work.products_completed == 3 && work.true_checks_completed == 1) ++incomplete_candidate_prefixes;
    } else {
      ++eligible_prefixes;
      require(std::string_view(work.candidate_forcing->decision) == "eligible" && work.final_measure == 2 &&
          work.final_measure == work.candidate_forcing->best_measure_before &&
          work.candidate_forcing->strict_measure && *work.candidate_forcing->strict_measure > *work.final_measure,
          "F搜索eligible末预算未过仍保strict best，不先替换为实际非best候选");
    }
  }
  require(eligible_prefixes == 1 && incomplete_candidate_prefixes > 0,
      "须实际覆盖候选A完成后的partial和唯一eligible末预算");
  {
    const wide::Layout large_layout{wide::budget_batch + 1, 0, 0};
    std::vector<double> large_rhs(large_layout.dimension(), 10); large_rhs.back() = 0;
    auto large_target = large_rhs; large_target[0] = 10.5;
    const auto large = candidate_fixture(large_layout, large_rhs, large_target);
    wide::Work full;
    CandidateTrace full_trace;
    int phase_checks = 0;
    forcing_weight_candidate_solve(large, full, full_trace, [&] {
      if (full.products_completed == 3 && full.true_checks_completed == 1) ++phase_checks;
    });
    candidate_returned(full);
    require(phase_checks >= 7, "启用F搜索的候选仍覆盖跨4096的真实残差两批预算");
    wide::Work work;
    CandidateTrace trace;
    int calls = 0;
    bool original = false;
    try {
      forcing_weight_candidate_solve(large, work, trace, [&] {
        if (work.products_completed == 3 && work.true_checks_completed == 1 && ++calls == phase_checks - 1)
          throw CallbackFailure{4098};
      });
    } catch (const CallbackFailure& error) { original = error.value == 4098; }
    candidate_bounded(work);
    require(original && work.preconditioner_completed == 2 && work.products_completed == 3 &&
        work.true_checks_started == 2 && work.true_checks_completed == 1 && work.final_measure == 2 &&
        work.candidate_forcing && !work.candidate_forcing->defect_norms && work.candidate_forcing_checks == 0 &&
        work.candidate_forcing_returns == 0, "跨批次候选半E仍不发布，原strict最佳摘要与P/A前缀保真");
  }
  details.push_back("forcing_weights_budget_and_true_gate");
}
namespace axpy = solver::experimental::fma_axpy;

struct FmaHardwareProbe {
  bool cpu_avx = false, fma = false, selected = false;
  axpy::ControlState control;
  double actual = 0, legacy = 0;
};
FmaHardwareProbe hardware_probe;

void write_test_control(axpy::ControlState state) noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __asm__ __volatile__("ldmxcsr %0\n\tfldcw %1" : : "m"(state.mxcsr), "m"(state.x87_control) : "memory");
#else
  static_cast<void>(state);
#endif
}

struct RestoreTestControl {
  axpy::ControlState saved = axpy::read_control_state();
  ~RestoreTestControl() noexcept { write_test_control(saved); }
};

struct ControlObservation { axpy::ControlState state; bool supported; };
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((noinline))
#endif
ControlObservation observe_integer_control(axpy::ControlState changed) noexcept {
  // 此窗口只执行整数寄存器操作和位谓词；恢复前不调用AXPY、LD算术或断言。
  const auto saved = axpy::read_control_state();
  write_test_control(changed);
  const auto observed = axpy::read_control_state();
  const bool supported = axpy::environment_supported(observed.mxcsr, observed.x87_control);
  write_test_control(saved);
  return {observed, supported};
}

void fma_hardware_probe_and_default_control() {
  hardware_probe.control = axpy::read_control_state();
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  hardware_probe.cpu_avx = __builtin_cpu_supports("avx");
  hardware_probe.fma = __builtin_cpu_supports("fma");
#endif
  // 保留进程原控制，不为测试强制改成可选择硬件的模式；实际资格由probe如实输出。
  require(axpy::hardware_available() == (hardware_probe.cpu_avx && hardware_probe.fma),
      "硬件资格必须同时来自真实AVX与FMA探测，不能用编译目标冒充运行时能力");
  std::vector<double> value{3, 5, 7, 9}, basis(4, 1);
  const auto result = axpy::subtract_scaled(value, basis, 2);
  require(result.finite && result.completed == 4 && value == std::vector<double>({1, 3, 5, 7}),
      "省略新硬件参数必须执行原LD乘减控制路径");
  bool used = true;
  value = {3, 5, 7, 9};
  axpy::subtract_scaled(value, basis, 2, {}, false, &used);
  require(!used && value == std::vector<double>({1, 3, 5, 7}), "显式关闭必须清旧used标记并保原运算");
  details.push_back("fma_hardware_probe_and_default_control");
}

void fma_environment_predicate() {
  constexpr std::uint32_t mxcsr = 0x1f80;
  constexpr std::uint16_t cw = 0x037f;
  static_assert(axpy::environment_supported(mxcsr, cw));
  for (unsigned rc = 1; rc < 4; ++rc) {
    require(!axpy::environment_supported(mxcsr | (rc << 13), cw), "纯谓词须拒绝三个非最近SSE舍入模式");
    require(!axpy::environment_supported(mxcsr, static_cast<std::uint16_t>(cw | (rc << 10))),
        "纯谓词须独立拒绝三个非最近x87舍入模式");
  }
  for (unsigned pc = 0; pc < 3; ++pc)
    require(!axpy::environment_supported(mxcsr, static_cast<std::uint16_t>((cw & ~0x0300u) | (pc << 8))),
        "x87精度控制必须实际为64位有效精度，不能只检查存储字节数");
  require(!axpy::environment_supported(mxcsr | 0x8000, cw) && !axpy::environment_supported(mxcsr | 0x0040, cw),
      "纯谓词须拒绝FTZ与DAZ；本测试不实际写可选DAZ位");
  for (unsigned bit = 0; bit < 6; ++bit) {
    require(!axpy::environment_supported(mxcsr & ~(1u << (7 + bit)), cw), "所有SSE异常均须屏蔽");
    require(!axpy::environment_supported(mxcsr, static_cast<std::uint16_t>(cw & ~(1u << bit))), "所有x87异常均须屏蔽");
  }
  for (std::uint32_t flags = 0; flags < 64; ++flags)
    require(axpy::environment_supported(mxcsr | flags, cw), "sticky状态不改变默认控制资格，不得关闭正常计算后的FMA");
  details.push_back("fma_environment_predicate");
}

void fma_exact_cancellation_and_tails() {
  constexpr double projection = 0x1.0000000000001p0, component = 0x1.ffffffffffffep-1;
  constexpr double expected = 0x1p-104;
  std::vector<double> legacy(4, 1), basis(4, component), actual(4, 1);
  bool legacy_used = true, selected = false;
  axpy::subtract_scaled(legacy, basis, projection, {}, false, &legacy_used);
  const auto result = axpy::subtract_scaled(actual, basis, projection, {}, true, &selected);
  hardware_probe.selected = selected; hardware_probe.actual = actual[0]; hardware_probe.legacy = legacy[0];
  const auto state = axpy::read_control_state();
  const bool eligible = axpy::hardware_available() && axpy::environment_supported(state.mxcsr, state.x87_control);
  require(!legacy_used && result.finite && result.completed == 4 && selected == eligible,
      "used必须反映真实target段是否启动，unsupported CPU必须真实回退");
  if (std::numeric_limits<long double>::digits == 64)
    require(legacy == std::vector<double>(4, 0), "父80位中间舍入应丢失2^-104取消误差");
  if (selected) require(actual == std::vector<double>(4, expected) && legacy[0] == 0,
      "已选择硬件时必须观察到精确2^-104，不能把父LD或未融合double运算冒充FMA");
  else require(actual == legacy, "未选硬件时须报告实际父路径结果，不能虚构取消误差");
  for (std::size_t size : {1u, 2u, 3u, 5u, 6u, 7u}) {
    std::vector<double> value(size, 1), direction(size, component);
    bool used = false;
    const auto tail = axpy::subtract_scaled(value, direction, projection, {}, true, &used);
    require(tail.finite && tail.completed == size && used == eligible, "尾部也须走同一运行时资格和有限检查");
    require(value == std::vector<double>(size, used ? expected : legacy[0]),
        "1到3个尾元素须真实融合；混合pack加尾部不能退回父取消值");
  }
  details.push_back("fma_exact_cancellation_and_tails");
}

void fma_overflow_underflow_and_finite_prefix() {
  const double largest = std::numeric_limits<double>::max();
  for (bool hardware : {false, true}) {
    std::vector<double> value(5, largest), basis(5, largest);
    const auto recovered = axpy::subtract_scaled(value, basis, 2, {}, hardware);
    require(recovered.finite && recovered.completed == 5 && value == std::vector<double>(5, -largest),
        "中间double乘积溢出但融合结果有限时不能误拒；父宽中间路径仍须保留");
    value.assign(5, 0); basis.assign(5, 0.5);
    const auto gradual = axpy::subtract_scaled(value, basis, std::numeric_limits<double>::min(), {}, hardware);
    require(gradual.finite && gradual.completed == 5 && value == std::vector<double>(5, -0x1p-1023),
        "默认控制下必须保留渐进下溢，不能把非零subnormal刷成零");
    value = {1, 2, 3, 4, 5}; basis = {0.5, 0.25, largest, 0.5, 0.5};
    const auto failed = axpy::subtract_scaled(value, basis, largest, {}, hardware);
    require(!failed.finite && failed.completed == 2 && value[0] == -largest * 0.5 && value[1] == -largest * 0.25 &&
        value[2] == 3 && value[3] == 4 && value[4] == 5,
        "首坏pack只能写已验证坐标前缀，坏坐标和后继必须原样保留");
  }
  details.push_back("fma_overflow_underflow_and_finite_prefix");
}

void fma_budget_chunk_boundaries() {
  for (bool hardware : {false, true}) {
    for (std::size_t size : {0u, 4095u, 4096u, 4097u}) {
      std::vector<double> value(size, 7), basis(size, 2);
      int checks = 0; bool used = true;
      const auto result = axpy::subtract_scaled(value, basis, 2, [&] { ++checks; }, hardware, &used);
      const int expected = static_cast<int>((size + 4095) / 4096);
      require(result.finite && result.completed == size && checks == expected && value == std::vector<double>(size, 3) &&
          (size != 0 || !used), "0/4095/4096/4097必须保原每4096一次预算，无额外末检查");
      for (int cutoff = 1; cutoff <= expected; ++cutoff) {
        value.assign(size, 7); int seen = 0; used = false; bool original = false;
        try { axpy::subtract_scaled(value, basis, 2, [&] {
          if (++seen == cutoff) throw CallbackFailure{cutoff};
        }, hardware, &used); }
        catch (const CallbackFailure& error) { original = error.value == cutoff; }
        const std::size_t prefix = static_cast<std::size_t>(cutoff - 1) * 4096;
        require(original && seen == cutoff, "预算异常类型与调用前缀必须原样传播");
        for (std::size_t i = 0; i < size; ++i)
          require(value[i] == (i < prefix ? 3 : 7), "预算失败段尚未写入，之前完整4096段不得回滚");
        if (prefix == 0) require(!used, "首预算失败不能伪报已执行硬件");
        else {
          const auto control = axpy::read_control_state();
          require(used == (hardware && axpy::hardware_available() &&
              axpy::environment_supported(control.mxcsr, control.x87_control)), "末预算失败仍保之前实际硬件使用事实");
        }
      }
    }
  }
  details.push_back("fma_budget_chunk_boundaries");
}

void fma_control_reads_and_callback_recheck() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  const auto initial = axpy::read_control_state();
  std::array<axpy::ControlState, 4> alternatives{initial, initial, initial, initial};
  alternatives[0].mxcsr |= 1u << 13;
  alternatives[1].mxcsr |= 1u << 15;
  alternatives[2].x87_control = static_cast<std::uint16_t>(initial.x87_control | (1u << 10));
  alternatives[3].x87_control = static_cast<std::uint16_t>((initial.x87_control & ~0x0300u) | 0x0200u);
  for (const auto changed : alternatives) {
    const auto observed = observe_integer_control(changed);
    const auto restored = axpy::read_control_state();
    require(observed.state.mxcsr == changed.mxcsr && observed.state.x87_control == changed.x87_control && !observed.supported &&
        restored.mxcsr == initial.mxcsr && restored.x87_control == initial.x87_control,
        "安全修改/读回/检测/恢复非默认控制只用整数，不在该状态执行任何数值路径");
  }
  RestoreTestControl restore;
  auto clean = restore.saved; clean.mxcsr &= ~0x3fu;
  write_test_control(clean);
  std::array<axpy::ControlState, 2> observed{};
  std::vector<double> value(4097, 3), basis(4097, 1);
  int checks = 0; bool used = false;
  const auto result = axpy::subtract_scaled(value, basis, 1, [&] {
    auto current = axpy::read_control_state(); current.mxcsr |= 0x20u;
    write_test_control(current);
    if (checks < 2) observed[static_cast<std::size_t>(checks)] = axpy::read_control_state();
    ++checks;
  }, true, &used);
  const bool controls_supported = axpy::environment_supported(clean.mxcsr, clean.x87_control);
  require(result.finite && checks == 2 && value == std::vector<double>(4097, 2) &&
      used == (axpy::hardware_available() && controls_supported) &&
      (observed[0].mxcsr & 0x20) != 0 && (observed[1].mxcsr & 0x20) != 0 &&
      axpy::environment_supported(observed[0].mxcsr, observed[0].x87_control) == controls_supported &&
      axpy::environment_supported(observed[1].mxcsr, observed[1].x87_control) == controls_supported,
      "每段callback仅观察默认控制和良性sticky；回调后的真实guard读取顺序另由源码审计核，不声称非默认算术支持");
#endif
  details.push_back("fma_control_reads_and_callback_recheck");
}

void fma_default_and_invalid_dimensions() {
  std::vector<double> value{1, 2}, basis{3};
  int checks = 0; bool used = true, original = false;
  try { axpy::subtract_scaled(value, basis, 1, [&] { ++checks; }, true, &used); }
  catch (const std::invalid_argument&) { original = true; }
  require(original && checks == 0 && !used && value == std::vector<double>({1, 2}),
      "尺寸错误须在预算和target前拒绝并清used，不能部分更新");
  value.clear(); basis.clear(); used = true;
  const auto empty = axpy::subtract_scaled(value, basis, 1, [&] { ++checks; }, true, &used);
  require(empty.finite && empty.completed == 0 && checks == 0 && !used, "空输入不得探称已执行target或添加预算检查");
  details.push_back("fma_default_and_invalid_dimensions");
}

WideVector fma_coupled_run(wide::Work& work, const wide::Budget& budget = {}, bool hardware = false) {
  const wide::Layout layout{2, 0, 0};
  const std::vector<double> rhs{1, -2, 3, -4};
  const std::array<double, 4> diagonal{2, 3, 5, 7};
  int calls = 0;
  const auto p = [&](std::span<const double> input, bool refinement) {
    ++calls; require(refinement == (calls > 1), "FMA组合核仍须真实区分初P及FG列");
    std::vector<double> result(input.size());
    for (std::size_t i = 0; i < input.size(); ++i)
      result[i] = input[i] * (0.25 + 0.2 * static_cast<double>((i + calls) % 4)) / diagonal[i];
    return result;
  };
  const auto a = [&](const auto& input, WideVector& output) {
    output.resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = static_cast<long double>(diagonal[i]) * input[i];
  };
  const auto ad = [&](std::span<const double> input, WideVector& output) { a(input, output); };
  const auto aw = [&](std::span<const long double> input, WideVector& output) { a(input, output); };
  if (!hardware) return wide::solve(layout, rhs, wide_tolerances(layout, rhs), p, ad, aw, work, budget, nullptr, false, false);
  return wide::solve(layout, rhs, wide_tolerances(layout, rhs), p, ad, aw, work, budget, nullptr, false, false, true);
}

void fma_coupled_true_direction_and_budget() {
  const std::array<double, 4> diagonal{2, 3, 5, 7}, rhs{1, -2, 3, -4};
  for (bool hardware : {false, true}) {
    wide::Work work;
    const auto answer = fma_coupled_run(work, {}, hardware);
    wide_accepted(work);
    require(answer.size() == rhs.size() && work.krylov_completed >= 2, "两种控制均须真实进入非平凡MGS，不能只测精确初P旁路");
    for (std::size_t i = 0; i < answer.size(); ++i)
      require(std::abs(diagonal[i] * answer[i] - rhs[i]) <= work.tolerances[wide::Layout{2, 0, 0}.block(i)],
          "硬件资格与AXPY结果都不能替代完整真实五块strict验收");
  }
  wide::Work baseline;
  int total = 0;
  fma_coupled_run(baseline, [&] { ++total; }, true);
  for (int cutoff = 1; cutoff <= total; ++cutoff) {
    wide::Work work; int checks = 0; bool original = false; WideVector output{101, 202};
    try { output = fma_coupled_run(work, [&] { if (++checks == cutoff) throw CallbackFailure{cutoff}; }, true); }
    catch (const CallbackFailure& error) { original = error.value == cutoff; }
    wide_bounded(work);
    require(original && checks == cutoff && output == WideVector({101, 202}) && std::string_view(work.status) == "exception",
        "FMA组合核每个实际预算前缀都须保原异常、P/A上限与调用者旧输出");
  }
  details.push_back("fma_coupled_true_direction_and_budget");
}
}  // 匿名命名空间
#include "solver/experimental/profile.hpp"
#include <thread>

namespace {
namespace profile = solver_v112_profile;
thread_local std::int64_t profile_tick = 0;

std::int64_t profile_fake_now() noexcept {
  const auto result = profile_tick;
  profile_tick += 10;
  return result;
}

const profile::Entry& profile_entry(const profile::Stats& stats, profile::Stage stage) {
  return stats.entries[static_cast<std::size_t>(stage)];
}

void profile_closed(const profile::Report& report) {
  require(report.stack_empty() && profile::counts_closed(report.stats) && profile::conserved(report.stats),
      "已关闭计时须有空栈、每Scope两次时钟、次数闭合及自身时间守恒");
  for (const auto& entry : report.stats.entries)
    require(entry.inclusive_ns >= 0 && entry.exclusive_ns >= 0 && entry.exclusive_ns <= entry.inclusive_ns,
        "层级计时的累计时间须非负，自身不能超过包含子调用的时间");
}

void profile_off_zero_clock() {
  profile_tick = 0;
  profile::Report report(false, false, profile_fake_now);
  {
    profile::Scope total(profile::Stage::total);
    const auto value = profile::call(profile::Stage::dot, [] { return 7; });
    const auto materialized = profile::materialize<int>(profile::Stage::norm, [] { return 9; });
    require(value == 7 && materialized == 9, "关闭计时不能改变返回值或物化结果");
  }
  require(!report.active() && profile_tick == 0 && report.stats.clock_reads == 0,
      "显式关闭不得读取真实或fake时钟，也不能由测试环境变量重新开启");
  for (const auto& entry : report.stats.entries)
    require(entry.started == 0 && entry.completed == 0 && entry.unwound == 0 &&
        entry.inclusive_ns == 0 && entry.exclusive_ns == 0, "关闭不能记任何阶段或调用");
  profile_closed(report);
  details.push_back("profile_off_zero_clock");
}

void profile_hierarchy_recursion() {
  profile_tick = 0;
  profile::Report report(true, false, profile_fake_now);
  {
    profile::Scope total(profile::Stage::total);
    profile::Scope coupled_scope(profile::Stage::coupled);
    profile::call(profile::Stage::dot, [] { profile::Scope inner(profile::Stage::dot); });
  }
  const auto& total = profile_entry(report.stats, profile::Stage::total);
  const auto& coupled_entry = profile_entry(report.stats, profile::Stage::coupled);
  const auto& dot_entry = profile_entry(report.stats, profile::Stage::dot);
  require(report.stats.clock_reads == 8 && profile_tick == 80 && total.inclusive_ns == 70 && total.exclusive_ns == 20 &&
      coupled_entry.inclusive_ns == 50 && coupled_entry.exclusive_ns == 20 &&
      dot_entry.inclusive_ns == 40 && dot_entry.exclusive_ns == 30 && dot_entry.started == 2 && dot_entry.completed == 2,
      "同名递归阶段也须只扣直接子Scope，不能按阶段总和重复扣除");
  profile_closed(report);
  details.push_back("profile_hierarchy_recursion");
}

void profile_exception_unwind() {
  profile_tick = 0;
  profile::Report report(true, false, profile_fake_now);
  bool original = false;
  {
    profile::Scope total(profile::Stage::total);
    try {
      profile::call(profile::Stage::coupled, [] {
        profile::Scope preconditioner(profile::Stage::preconditioner);
        throw CallbackFailure{37};
      });
    } catch (const CallbackFailure& error) { original = error.value == 37; }
    const auto value = profile::materialize<int>(profile::Stage::norm, [] { return 11; });
    require(value == 11, "异常后恢复父Scope仍须可正常物化和计时");
    require(!profile::call(profile::Stage::factor, [] { return false; }),
        "正常返回false须原样交付，不能由计时转换成成功或异常");
  }
  require(original && report.stats.clock_reads == 10 &&
      profile_entry(report.stats, profile::Stage::coupled).unwound == 1 &&
      profile_entry(report.stats, profile::Stage::preconditioner).unwound == 1 &&
      profile_entry(report.stats, profile::Stage::norm).completed == 1 &&
      profile_entry(report.stats, profile::Stage::factor).completed == 1 &&
      profile_entry(report.stats, profile::Stage::factor).unwound == 0 &&
      profile_entry(report.stats, profile::Stage::total).completed == 1,
      "析构计时须保原异常，正常返回false与异常展开不能混为同一计数");
  profile_closed(report);
  details.push_back("profile_exception_unwind");
}

void profile_nested_reports() {
  profile_tick = 0;
  profile::Stats inner_stats;
  profile::Report outer(true, false, profile_fake_now);
  {
    profile::Scope total(profile::Stage::total);
    profile::Scope coupled_scope(profile::Stage::coupled);
    {
      profile::Report inner(true, false, profile_fake_now);
      {
        profile::Scope inner_total(profile::Stage::total);
        profile::Scope norm_scope(profile::Stage::norm);
      }
      profile_closed(inner);
      inner_stats = inner.stats;
    }
  }
  require(outer.stats.clock_reads == 4 && inner_stats.clock_reads == 4 &&
      profile_entry(outer.stats, profile::Stage::total).inclusive_ns == 70 &&
      profile_entry(outer.stats, profile::Stage::coupled).inclusive_ns == 50 &&
      profile_entry(outer.stats, profile::Stage::norm).started == 0 &&
      profile_entry(inner_stats, profile::Stage::total).inclusive_ns == 30 &&
      profile_entry(inner_stats, profile::Stage::norm).inclusive_ns == 10,
      "嵌套开启Report须自持统计；内Report墙钟仍属于外包含时间，不能跨Report重复扣减");
  profile_closed(outer);
  details.push_back("profile_nested_reports");
}

void profile_context_restore() {
  profile_tick = 0;
  profile::Stats outer_stats, off_stats, inner_stats;
  {
    profile::Report outer(true, false, profile_fake_now);
    {
      profile::Scope total(profile::Stage::total);
      {
        profile::Report off(false, false, profile_fake_now);
        profile::Scope hidden(profile::Stage::dot);
        {
          profile::Report inner(true, false, profile_fake_now);
          { profile::Scope inner_total(profile::Stage::total); }
          profile_closed(inner);
          inner_stats = inner.stats;
        }
        { profile::Scope still_hidden(profile::Stage::norm); }
        profile_closed(off);
        off_stats = off.stats;
      }
      { profile::Scope restored(profile::Stage::norm); }
    }
    profile_closed(outer);
    outer_stats = outer.stats;
  }
  const auto after_reports = profile_tick;
  { profile::Scope orphan(profile::Stage::total); }
  require(profile_tick == after_reports && outer_stats.clock_reads == 4 && off_stats.clock_reads == 0 &&
      inner_stats.clock_reads == 2 && profile_entry(outer_stats, profile::Stage::dot).started == 0 &&
      profile_entry(outer_stats, profile::Stage::norm).started == 1 &&
      profile_entry(outer_stats, profile::Stage::total).inclusive_ns == 50 &&
      profile_entry(inner_stats, profile::Stage::total).inclusive_ns == 10,
      "开启到关闭再开启的嵌套必须逐层恢复，最后无Report的Scope不得读取旧上下文或时钟");
  details.push_back("profile_context_restore");
}

void profile_thread_isolation_and_real_clock() {
  profile_tick = 0;
  profile::Stats worker_stats;
  bool worker_closed = false;
  {
    profile::Report main_report(true, false, profile_fake_now);
    {
      profile::Scope total(profile::Stage::total);
      std::thread worker([&] {
        profile_tick = 0;
        profile::Report report(true, false, profile_fake_now);
        {
          profile::Scope worker_total(profile::Stage::total);
          profile::Scope norm_scope(profile::Stage::norm);
        }
        worker_closed = report.stack_empty() && profile::counts_closed(report.stats) && profile::conserved(report.stats);
        worker_stats = report.stats;
      });
      worker.join();
      { profile::Scope dot_scope(profile::Stage::dot); }
    }
    profile_closed(main_report);
    require(worker_closed && main_report.stats.clock_reads == 4 && worker_stats.clock_reads == 4 &&
        profile_entry(main_report.stats, profile::Stage::norm).started == 0 &&
        profile_entry(main_report.stats, profile::Stage::dot).started == 1 &&
        profile_entry(worker_stats, profile::Stage::norm).started == 1 &&
        profile_entry(worker_stats, profile::Stage::dot).started == 0,
        "线程局部当前上下文不能把工作线程阶段或fake时钟记入主线程Report");
  }
  {
    profile::Report real(true, false);
    {
      profile::Scope total(profile::Stage::total);
      profile::Scope norm_scope(profile::Stage::norm);
    }
    require(real.stats.clock_reads == 4, "真实steady_clock路径也须严格每Scope两次读取");
    profile_closed(real);
  }
  details.push_back("profile_thread_isolation_and_real_clock");
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
    candidate_forcing_strict_precedence(); candidate_forcing_same_bounds(); candidate_forcing_actual_nonbest_direction();
    candidate_forcing_default_and_invalid_policy(); candidate_forcing_rejection_snapshot_and_caps();
    candidate_forcing_budget_partial_snapshot();
    forcing_weights_early_candidate(); forcing_weights_consistent_physical_system();
    forcing_weights_default_and_invalid_policy(); forcing_weights_budget_and_true_gate();
    fma_hardware_probe_and_default_control(); fma_environment_predicate(); fma_exact_cancellation_and_tails();
    fma_overflow_underflow_and_finite_prefix(); fma_budget_chunk_boundaries();
    fma_control_reads_and_callback_recheck(); fma_default_and_invalid_dimensions(); fma_coupled_true_direction_and_budget();
    profile_off_zero_clock(); profile_hierarchy_recursion(); profile_exception_unwind();
    profile_nested_reports(); profile_context_restore(); profile_thread_isolation_and_real_clock();
    static_assert(coupled::max_window == wide::max_window && coupled::max_cycles == wide::max_cycles);
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"cases\":%zu,\"default_limits\":{"
        "\"max_window\":%zu,\"max_cycles\":%d},\"details\":[", details.size(), wide::max_window, wide::max_cycles);
    for (std::size_t i = 0; i < details.size(); ++i) std::printf("%s\"%s\"", i ? "," : "", details[i]);
    std::printf("],\"hardware_probe\":{\"cpu_avx\":%s,\"fma\":%s,\"mxcsr\":%u,\"cw\":%u,\"selected\":%s,"
        "\"cancellation_actual\":%.17g,\"cancellation_legacy\":%.17g,\"cancellation_expected\":%.17g,\"long_double_digits\":%d}}\n",
        hardware_probe.cpu_avx ? "true" : "false", hardware_probe.fma ? "true" : "false",
        static_cast<unsigned>(hardware_probe.control.mxcsr), static_cast<unsigned>(hardware_probe.control.x87_control),
        hardware_probe.selected ? "true" : "false", hardware_probe.actual, hardware_probe.legacy, 0x1p-104,
        std::numeric_limits<long double>::digits);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "联立Newton独立核检查失败（已完成%zu组）：%s\n", details.size(), error.what());
    return 1;
  }
}
