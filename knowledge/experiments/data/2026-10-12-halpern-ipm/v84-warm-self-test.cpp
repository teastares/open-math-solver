#include "v84-warm-pdhg.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace warm = solver::v84_warm_pdhg;
using Index = warm::Index;
std::vector<const char*> details;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool status(const warm::Result& result, const char* expected) { return std::string_view(result.status) == expected; }

struct Fixture {
  Index n, m;
  std::vector<Index> starts, rows;
  std::vector<double> values, costs, rhs;
  std::vector<bool> inequality;
  warm::View view() const { return {n, m, starts, rows, values, costs, rhs, inequality}; }
};

// 独立按完整 LP 的原始违反、驻点残差和目标差验收，不用内核步长或迭代差代替终验。
bool optimal(const Fixture& f, const std::vector<double>& x, const std::vector<double>& z, long double tolerance = 1e-9L) {
  if (x.size() != static_cast<std::size_t>(f.n) || z.size() != static_cast<std::size_t>(f.m)) return false;
  std::vector<long double> activity(static_cast<std::size_t>(f.m), 0), transpose(static_cast<std::size_t>(f.n), 0);
  long double primal_objective = 0, dual_term = 0;
  for (Index col = 0; col < f.n; ++col) {
    if (!std::isfinite(x[static_cast<std::size_t>(col)])) return false;
    primal_objective += static_cast<long double>(f.costs[static_cast<std::size_t>(col)]) * x[static_cast<std::size_t>(col)];
    for (Index k = f.starts[static_cast<std::size_t>(col)]; k < f.starts[static_cast<std::size_t>(col + 1)]; ++k) {
      const auto entry = static_cast<std::size_t>(k), row = static_cast<std::size_t>(f.rows[entry]);
      activity[row] += static_cast<long double>(f.values[entry]) * x[static_cast<std::size_t>(col)];
      transpose[static_cast<std::size_t>(col)] += static_cast<long double>(f.values[entry]) * z[row];
    }
  }
  for (Index row = 0; row < f.m; ++row) {
    const auto i = static_cast<std::size_t>(row);
    if (!std::isfinite(z[i]) || (f.inequality[i] && z[i] < 0)) return false;
    const long double defect = activity[i] - f.rhs[i];
    const long double bound = tolerance * (1 + std::abs(static_cast<long double>(f.rhs[i])));
    if (f.inequality[i] ? defect > bound : std::abs(defect) > bound) return false;
    dual_term += static_cast<long double>(f.rhs[i]) * z[i];
  }
  for (Index col = 0; col < f.n; ++col) {
    const auto i = static_cast<std::size_t>(col);
    if (std::abs(transpose[i] + f.costs[i]) > tolerance * (1 + std::abs(static_cast<long double>(f.costs[i])))) return false;
  }
  return std::abs(primal_objective + dual_term) <= tolerance * (1 + std::abs(primal_objective) + std::abs(dual_term));
}

void ledger(const warm::Result& result, const warm::Limits& limits) {
  const auto& s = result.stats;
  require(s.setup_completed <= s.setup_started && s.setup_started <= 1 &&
      s.steps_completed <= s.multiply_completed && s.multiply_completed <= s.multiply_started &&
      s.multiply_started <= s.transpose_completed && s.transpose_completed <= s.transpose_started &&
      s.transpose_started <= s.steps_started && s.steps_started - s.steps_completed <= 1 &&
      s.work_units <= limits.max_work && s.assessments_completed <= s.assessments_started &&
      s.assessments_started - s.assessments_completed <= 1, "已提交步、两个乘积与部分工作账本必须一致");
  require(s.last_assessed_step.has_value() == (s.assessments_completed > 0) &&
      (!s.last_assessed_step || *s.last_assessed_step <= s.steps_completed), "最后验收只可关联已提交点");
}

void convergence() {
  const std::vector<Fixture> fixtures{
    {1, 1, {0, 1}, {0}, {1}, {-1}, {1}, {true}},
    {1, 1, {0, 1}, {0}, {2}, {2}, {6}, {false}},
    {2, 3, {0, 2, 5}, {0, 1, 0, 1, 2}, {1, -3, 2, 1, 1}, {4.25, -0.25}, {5, -1, 4}, {false, true, true}}
  };
  for (const auto& f : fixtures) {
    warm::Limits limits;
    std::vector<double> x(static_cast<std::size_t>(f.n), 0), z(static_cast<std::size_t>(f.m), 0);
    warm::Result result;
    warm::run(f.view(), limits, x, z, 2, {}, [&](Index, const auto& a, const auto& b) { return optimal(f, a, b); }, result);
    require(result.accepted && status(result, "accepted") && optimal(f, result.x, result.z), "不等式、等式和混合 LP 均须通过独立完整验收");
    ledger(result, limits);
  }
  details.push_back("不等式等式混合完整LP验收");
}

void seed_and_step() {
  const Fixture equality{1, 1, {0, 1}, {0}, {2}, {2}, {6}, {false}};
  const std::vector<double> exact_x{9}, exact_z{-3};
  warm::Limits limits;
  warm::Result seed;
  warm::run(equality.view(), limits, exact_x, exact_z, 3, {}, [&](Index step, const auto& x, const auto& z) {
    require(step == 0 && x == std::vector<double>{3} && z == std::vector<double>{-1}, "种子须同时除以真实 tau");
    return optimal(equality, x, z);
  }, seed);
  require(seed.accepted && seed.stats.steps_started == 0 && seed.stats.setup_started == 0 &&
      seed.stats.assessments_completed == 1, "种子已达标不得伪造迭代或范数装配");
  const Fixture f{2, 2, {0, 2, 4}, {0, 1, 0, 1}, {1, -3, 2, 4}, {0.5, -2}, {2, -1}, {true, false}};
  const std::vector<double> x{2, -4}, z{1, -2};
  warm::Limits one; one.max_iterations = 1;
  warm::Result result;
  std::vector<Index> evaluated;
  warm::run(f.view(), one, x, z, 2, {}, [&](Index step, const auto&, const auto&) { evaluated.push_back(step); return false; }, result);
  require(!result.accepted && status(result, "iteration_limit") && evaluated == std::vector<Index>({0, 1}) &&
      std::abs(result.x[0] - 0.1) < 1e-14 && std::abs(result.x[1] + 1.25) < 1e-14 &&
      result.z[0] == 0 && std::abs(result.z[1] + 0.82) < 1e-14,
      "非对称完整矩阵须使用列行和步长、primal-first、外推及正确投影");
  require(x == std::vector<double>({2, -4}) && z == std::vector<double>({1, -2}), "调用者原始种子不得被修改");
  ledger(result, one);
  const auto saved_x = result.x, saved_z = result.z;
  const std::span<const double> aliased_x(result.x), aliased_z(result.z);
  warm::Limits no_step; no_step.max_iterations = 0;
  warm::run(f.view(), no_step, aliased_x, aliased_z, 2, {}, [](Index, const auto&, const auto&) { return false; }, result);
  require(result.x[0] == saved_x[0] / 2 && result.x[1] == saved_x[1] / 2 &&
      result.z[0] == saved_z[0] / 2 && result.z[1] == saved_z[1] / 2,
      "复用输出向量作为下一次种子不得因重置结果使 span 悬空");
  details.push_back("非单位tau种子与非对称一步解析");
}

void zero_and_invalid() {
  const Fixture empty{0, 0, {0}, {}, {}, {}, {}, {}};
  const std::vector<double> nothing;
  warm::Limits limits;
  warm::Result result;
  warm::run(empty.view(), limits, nothing, nothing, 1, {}, [&](Index step, const auto& x, const auto& z) {
    return step == 0 && x.empty() && z.empty();
  }, result);
  require(result.accepted && result.stats.steps_completed == 0, "空系统须由真实种子验收明确接纳");
  const Fixture zero{1, 1, {0, 0}, {}, {}, {1}, {0}, {true}};
  const std::vector<double> x{0}, z{0};
  warm::Limits two; two.max_iterations = 2;
  warm::run(zero.view(), two, x, z, 1, {}, [](Index, const auto&, const auto&) { return false; }, result);
  require(status(result, "iteration_limit") && !result.accepted && std::abs(result.x[0] + 1.8) < 1e-14 && result.z[0] == 0,
          "零矩阵仍须使用合法非零步长，不能伪造最优或不可行证书");
  std::vector<Fixture> bad{
    {-1, 1, {0}, {}, {}, {}, {0}, {true}},
    {1, 1, {}, {}, {}, {0}, {0}, {true}},
    {1, 1, {0, 1}, {1}, {1}, {0}, {0}, {true}},
    {1, 1, {0, 1}, {0}, {std::numeric_limits<double>::infinity()}, {0}, {0}, {true}},
    {1, 1, {0, 0}, {}, {}, {std::numeric_limits<double>::quiet_NaN()}, {0}, {true}},
    {1, 1, {0, 0}, {}, {}, {0}, {0}, {}}
  };
  for (const auto& f : bad) {
    warm::run(f.view(), limits, x, z, 1, {}, [](Index, const auto&, const auto&) { return true; }, result);
    require(status(result, "invalid_input") && !result.accepted && !result.stats.seed_available &&
        result.stats.assessments_started == 0, "坏输入不能调用验收制造接纳");
  }
  for (double tau : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    warm::run(zero.view(), limits, x, z, tau, {}, [](Index, const auto&, const auto&) { return true; }, result);
    require(status(result, "invalid_input") && !result.accepted, "非法 tau 不得产生种子");
  }
  const std::vector<double> negative{-1};
  warm::run(zero.view(), limits, x, negative, 1, {}, [](Index, const auto&, const auto&) { return true; }, result);
  require(status(result, "invalid_input") && !result.accepted, "不等式种子乘子须属于非负域");
  details.push_back("零维零矩阵与坏输入拒绝");
}

void numerical_ranges() {
  const Fixture free{1, 0, {0, 0}, {}, {}, {0}, {}, {}};
  warm::Limits limits; limits.max_iterations = 1;
  const std::vector<double> no_z, tiny{std::numeric_limits<double>::denorm_min()};
  warm::Result result;
  warm::run(free.view(), limits, tiny, no_z, 2, {}, [](Index, const auto&, const auto&) { return true; }, result);
  require(status(result, "arithmetic_failure") && !result.accepted && !result.stats.seed_available,
          "非零种子除 tau 下溢不能静默归零再接纳");
  const Fixture overflow{1, 0, {0, 0}, {}, {}, {-std::numeric_limits<double>::max()}, {}, {}};
  const std::vector<double> huge{std::numeric_limits<double>::max()};
  warm::run(overflow.view(), limits, huge, no_z, 1, {}, [](Index, const auto&, const auto&) { return false; }, result);
  require(status(result, "arithmetic_failure") && !result.accepted && result.x == huge && result.stats.steps_completed == 0 &&
      result.stats.steps_started == 1, "试点溢出必须保留已归一化的旧点");
  const Fixture mixed{2, 2, {0, 1, 2}, {0, 1}, {1e150, 1e-150}, {1, -1}, {0, 0}, {true, false}};
  const std::vector<double> zero{0, 0};
  warm::run(mixed.view(), limits, zero, zero, 1, {}, [](Index, const auto&, const auto&) { return false; }, result);
  require(status(result, "iteration_limit") && result.stats.steps_completed == 1 &&
      std::all_of(result.x.begin(), result.x.end(), [](double value) { return std::isfinite(value); }) &&
      std::all_of(result.z.begin(), result.z.end(), [](double value) { return std::isfinite(value); }), "悬殊系数仍须完成有限、非零步长的一步");
  details.push_back("非零下溢试点溢出及悬殊尺度");
}

void iteration_and_work() {
  const Fixture f{1, 1, {0, 1}, {0}, {1}, {-1}, {1}, {true}};
  const std::vector<double> x{0}, z{0};
  for (Index count : {Index{0}, Index{1}, Index{63}, Index{64}, Index{65}, Index{128}}) {
    warm::Limits limits; limits.max_iterations = count;
    warm::Result result;
    std::vector<Index> seen, expected{0};
    for (Index step = 64; step <= count; step += 64) expected.push_back(step);
    if (count % 64 != 0) expected.push_back(count);
    warm::run(f.view(), limits, x, z, 1, {}, [&](Index step, const auto&, const auto&) { seen.push_back(step); return false; }, result);
    require(status(result, "iteration_limit") && !result.accepted && result.stats.steps_completed == count &&
        result.stats.transpose_completed == count && result.stats.multiply_completed == count && seen == expected,
        "固定评估周期与最后未评估点不能遗漏或重复");
    ledger(result, limits);
  }
  warm::Limits full_limits; full_limits.max_iterations = 2;
  warm::Result full, first;
  const warm::Assessment reject = [](Index, const auto&, const auto&) { return false; };
  warm::run(f.view(), full_limits, x, z, 1, {}, reject, full);
  warm::Limits one; one.max_iterations = 1;
  warm::run(f.view(), one, x, z, 1, {}, reject, first);
  for (Index cap : {Index{0}, Index{1}, full.stats.work_units - 1, full.stats.work_units}) {
    warm::Limits limits = full_limits; limits.max_work = cap;
    warm::Result result;
    warm::run(f.view(), limits, x, z, 1, {}, reject, result);
    require(result.stats.work_units <= cap && !result.accepted, "自有工作不得先超额度再拒绝");
    if (cap == full.stats.work_units) require(status(result, "iteration_limit") && result.x == full.x && result.z == full.z,
        "恰好工作额度须可完成相同两步");
    else require(status(result, "work_limit"), "不足工作额度须返回有界停止");
    if (cap == full.stats.work_units - 1) require(result.stats.steps_completed == 1 && result.x == first.x && result.z == first.z &&
        result.stats.last_assessed_step == 1, "提交前工作中断须保留并最终验收前一步");
    ledger(result, limits);
  }
  details.push_back("迭代周期和精确工作额度边界");
}

void callback_transactions() {
  struct BudgetFailure : std::runtime_error { using std::runtime_error::runtime_error; };
  const Fixture f{1, 1, {0, 1}, {0}, {1}, {0}, {1}, {true}};
  const std::vector<double> x{0}, z{0};
  warm::Limits limits; limits.max_iterations = 2;
  warm::Result result;
  bool callback_returned = false, interrupted = false;
  try {
    warm::run(f.view(), limits, x, z, 1, [&] {
      if (callback_returned) throw BudgetFailure("验收后预算结束");
    }, [&](Index, const auto& a, const auto& b) { callback_returned = true; return optimal(f, a, b); }, result);
  } catch (const BudgetFailure& error) { interrupted = std::string_view(error.what()) == "验收后预算结束"; }
  require(interrupted && !result.accepted && status(result, "budget_exception") &&
      result.stats.assessments_completed == 1 && result.stats.last_assessed_step == 0,
      "回调真值之后预算失败不能伪造接纳");
  interrupted = false;
  try { warm::run(f.view(), limits, x, z, 1, {}, [](Index, const auto&, const auto&) -> bool { throw 17; }, result); }
  catch (int value) { interrupted = value == 17; }
  require(interrupted && !result.accepted && status(result, "assessment_exception") &&
      result.stats.assessments_started == 1 && result.stats.assessments_completed == 0, "验收非标准异常须原型透传且不补回调");
  warm::Result full;
  warm::run(f.view(), limits, x, z, 1, {}, [](Index, const auto&, const auto&) { return false; }, full);
  warm::Limits short_work = limits; short_work.max_work = full.stats.work_units - 1;
  warm::run(f.view(), short_work, x, z, 1, {}, [&](Index step, const auto& a, const auto& b) {
    return step > 0 && optimal(f, a, b);
  }, result);
  require(result.accepted && result.stats.steps_completed == 1 && result.stats.steps_started == 2 && optimal(f, result.x, result.z),
      "有界出口只可接纳最后已提交的真实合格点，不提交中断试点");
  details.push_back("验收真值后预算与异常事务");
}

void every_budget_boundary() {
  struct BudgetFailure : std::runtime_error { using std::runtime_error::runtime_error; };
  Fixture f{1, 600, {0, 600}, {}, {}, {-1}, std::vector<double>(600, 1), std::vector<bool>(600, true)};
  for (Index row = 0; row < 600; ++row) { f.rows.push_back(row); f.values.push_back(1); }
  const std::vector<double> x{-2}, z(600, 0);
  warm::Limits limits; limits.max_iterations = 3; limits.check_interval = 2;
  const warm::Assessment reject = [](Index, const auto&, const auto&) { return false; };
  std::vector<warm::Result> reference;
  for (Index count = 0; count <= 3; ++count) {
    warm::Limits prefix = limits; prefix.max_iterations = count;
    warm::Result result;
    warm::run(f.view(), prefix, x, z, 1, {}, reject, result);
    reference.push_back(std::move(result));
  }
  Index checks = 0, first_transpose_checks = 0;
  warm::Result full;
  warm::run(f.view(), limits, x, z, 1, [&] {
    ++checks;
    if (full.stats.transpose_started == 1 && full.stats.transpose_completed == 0) ++first_transpose_checks;
  }, reject, full);
  require(checks == full.stats.budget_checks && first_transpose_checks >= 3,
          "长列转置乘积除前后边界外必须实际触发元素工作预算检查");
  for (Index stop = 1; stop <= checks; ++stop) {
    Index calls = 0;
    bool interrupted = false;
    warm::Result result;
    try {
      warm::run(f.view(), limits, x, z, 1, [&] { if (++calls == stop) throw BudgetFailure("长列独立预算结束"); }, reject, result);
    } catch (const BudgetFailure& error) { interrupted = std::string_view(error.what()) == "长列独立预算结束"; }
    require(interrupted && !result.accepted && status(result, "budget_exception"), "每个真实预算边界须原型透传");
    if (result.stats.seed_available) {
      const auto& saved = reference[static_cast<std::size_t>(result.stats.steps_completed)];
      require(result.x == saved.x && result.z == saved.z, "中途异常不得污染上一次完整提交的双向量");
    } else require(result.x.empty() && result.z.empty(), "不完整归一化种子不得对外提交");
    ledger(result, limits);
  }
  details.push_back("长列每真实预算边界原异常及双提交");
}

}  // 匿名命名空间

int main() {
  try {
    convergence(); seed_and_step(); zero_and_invalid(); numerical_ranges();
    iteration_and_work(); callback_transactions(); every_budget_boundary();
    const warm::Limits defaults;
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"cases\":%zu,\"default_limits\":{"
        "\"max_iterations\":%lld,\"max_work_units\":%lld,\"check_interval\":%lld,"
        "\"budget_check_work_interval\":%lld,\"step_safety\":%.17g},\"details\":[", details.size(),
        static_cast<long long>(defaults.max_iterations), static_cast<long long>(defaults.max_work),
        static_cast<long long>(defaults.check_interval), static_cast<long long>(warm::budget_check_work_interval),
        static_cast<double>(warm::step_safety));
    for (std::size_t i = 0; i < details.size(); ++i) std::printf("%s\"%s\"", i ? "," : "", details[i]);
    std::puts("]}");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "warm PDHG隔离检查失败：%s\n", error.what());
    return 1;
  }
}
