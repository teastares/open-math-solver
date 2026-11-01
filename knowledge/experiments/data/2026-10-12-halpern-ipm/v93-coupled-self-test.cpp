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

int main() {
  try {
    exact_initial(); variable_preconditioner(); independent_blocks(); rounded_candidate_must_fail();
    fixed_exhaustion(); invalid_inputs_and_callbacks(); budgets(); pure_scalar_search();
    nominal_energy_identity(); stored_energy_contract(); energy_budget_boundaries();
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"cases\":%zu,\"default_limits\":{"
        "\"max_window\":%zu,\"max_cycles\":%d},\"details\":[", details.size(), coupled::max_window, coupled::max_cycles);
    for (std::size_t i = 0; i < details.size(); ++i) std::printf("%s\"%s\"", i ? "," : "", details[i]);
    std::puts("]}");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "联立Newton独立核检查失败（已完成%zu组）：%s\n", details.size(), error.what());
    return 1;
  }
}
