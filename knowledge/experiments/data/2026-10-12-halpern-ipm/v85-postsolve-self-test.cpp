#include "Highs.h"
#include "solver/lp_optimality.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
const double infinity = std::numeric_limits<double>::infinity();

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class SharedDeadlineExpired : public std::runtime_error {
 public:
  SharedDeadlineExpired() : std::runtime_error("组件共享预算耗尽") {}
};

// 所有同步组件调用和原模型终验共享同一个时点；不以 HiGHS 的暂停时钟续时。
// 可注入时钟仅供确定性入口/出口自测，实际组件始终使用 steady_clock。
class SharedDeadline {
 public:
  using Now = std::function<Clock::time_point()>;
  explicit SharedDeadline(Now now = [] { return Clock::now(); })
      : now_(std::move(now)), end_(now_() + std::chrono::seconds(5)) {}

  double remaining() const {
    const double value = std::chrono::duration<double>(end_ - now_()).count();
    if (!(value > 0.0)) throw SharedDeadlineExpired();
    return value;
  }

  void check() const { (void)remaining(); }

  template <class Function>
  auto call(Function&& function) const {
    check();
    auto value = function();
    // 返回值只在出口检查后交给调用者；超时的候选不能先提交。
    check();
    return value;
  }

 private:
  Now now_;
  Clock::time_point end_;
};

template <class Value>
void option(Highs& highs, const char* name, Value value) {
  require(highs.setOptionValue(name, value) == HighsStatus::kOk,
          "不能设置固定组件参数");
}

bool empty_invalid_basis(const HighsBasis& basis) {
  // valid=false 并不足够：有尺寸的 status 也会让上游视为提供了 basis。
  return !basis.valid && basis.col_status.empty() && basis.row_status.empty();
}

solver::Model scalar(double cost, double lower, double upper,
                     double offset = 0.0, bool maximum = false) {
  solver::Model model;
  model.name = "v85_component_scalar";
  model.sense = maximum ? solver::ObjectiveSense::maximize
                        : solver::ObjectiveSense::minimize;
  model.objective_offset = offset;
  model.costs = {cost};
  model.column_lower = {lower};
  model.column_upper = {upper};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  return model;
}

void row(solver::Model& model, double coefficient, double lower, double upper) {
  model.row_lower = {lower};
  model.row_upper = {upper};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {coefficient};
}

solver::Model coupled(bool maximum = false) {
  solver::Model model;
  model.name = "v85_component_nonempty";
  model.sense = maximum ? solver::ObjectiveSense::maximize
                        : solver::ObjectiveSense::minimize;
  model.objective_offset = maximum ? 7.0 : -3.0;
  model.costs = maximum ? std::vector<double>{1, 1} : std::vector<double>{-1, -1};
  model.column_lower = {0, 0};
  model.column_upper = {infinity, infinity};
  model.row_lower = {-infinity, -infinity};
  model.row_upper = {4, 4};
  model.kinds.assign(2, solver::VariableKind::continuous);
  model.column_start = {0, 2, 4};
  model.row_index = {0, 1, 0, 1};
  model.values = {2, 1, 1, 2};
  return model;
}

HighsLp minimum_copy(const solver::Model& model) {
  require(solver::validate_model(model).empty(), "组件原模型必须合法");
  require(!solver::is_mip(model), "组件仅覆盖连续 LP");
  HighsLp lp;
  lp.num_col_ = static_cast<HighsInt>(model.costs.size());
  lp.num_row_ = static_cast<HighsInt>(model.row_lower.size());
  lp.model_name_ = model.name;
  lp.sense_ = ObjSense::kMinimize;
  const double sense = model.sense == solver::ObjectiveSense::minimize ? 1.0 : -1.0;
  lp.offset_ = sense * model.objective_offset;
  lp.col_cost_ = model.costs;
  for (double& value : lp.col_cost_) value *= sense;
  lp.col_lower_ = model.column_lower;
  lp.col_upper_ = model.column_upper;
  lp.row_lower_ = model.row_lower;
  lp.row_upper_ = model.row_upper;
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.assign(model.column_start.begin(), model.column_start.end());
  lp.a_matrix_.index_.assign(model.row_index.begin(), model.row_index.end());
  lp.a_matrix_.value_ = model.values;
  lp.setMatrixDimensions();
  return lp;
}

solver::Model own_model(HighsLp lp) {
  lp.ensureColwise();
  require(lp.sense_ == ObjSense::kMinimize, "约化模型必须保持最小化副本方向");
  solver::Model model;
  model.name = lp.model_name_;
  model.objective_offset = lp.offset_;
  model.costs = lp.col_cost_;
  model.column_lower = lp.col_lower_;
  model.column_upper = lp.col_upper_;
  model.row_lower = lp.row_lower_;
  model.row_upper = lp.row_upper_;
  model.kinds.assign(model.costs.size(), solver::VariableKind::continuous);
  model.column_start.assign(lp.a_matrix_.start_.begin(), lp.a_matrix_.start_.end());
  if (model.costs.empty() && model.column_start.empty()) model.column_start = {0};
  model.row_index.assign(lp.a_matrix_.index_.begin(), lp.a_matrix_.index_.end());
  model.values = lp.a_matrix_.value_;
  require(solver::validate_model(model).empty(), "约化模型必须通过独立表示检查");
  return model;
}

bool accepted(const solver::LpVerification& quality) {
  return quality.primal.valid_input && quality.primal.feasible &&
         quality.primal.objective_consistent && quality.optimality.valid_input &&
         quality.optimality.dual_feasible && quality.optimality.gap_satisfied &&
         quality.optimality.numerically_optimal;
}

HighsSolution candidate(const solver::Model& reduced, std::vector<double> x,
                        std::vector<double> y) {
  require(x.size() == reduced.costs.size() && y.size() == reduced.row_lower.size(),
          "约化原始/行对偶维度必须完整");
  HighsSolution result;
  result.value_valid = true;
  result.dual_valid = true;
  result.col_value = std::move(x);
  result.row_dual = std::move(y);
  result.row_value.assign(reduced.row_lower.size(), 0.0);
  result.col_dual.resize(reduced.costs.size());
  for (std::size_t col = 0; col < reduced.costs.size(); ++col) {
    long double value = reduced.costs[col];
    for (std::int64_t k = reduced.column_start[col]; k < reduced.column_start[col + 1]; ++k) {
      const std::size_t entry = static_cast<std::size_t>(k);
      const std::size_t r = static_cast<std::size_t>(reduced.row_index[entry]);
      value -= static_cast<long double>(reduced.values[entry]) * result.row_dual[r];
      result.row_value[r] += reduced.values[entry] * result.col_value[col];
    }
    result.col_dual[col] = static_cast<double>(value);
    require(std::isfinite(result.col_dual[col]), "列对偶 c'−Aᵀy 必须有限");
  }
  return result;
}

struct Detail {
  const char* id;
  const char* kind;
  bool highs_observed = false;
  HighsInt original_columns = 0, original_rows = 0, reduced_columns = 0, reduced_rows = 0;
  std::optional<int> presolve_status, postsolve_status;
  bool expected_accept = false, accepted = false, original_verified = false;
  std::optional<double> original_objective, objective_gap;
  std::optional<bool> basis_valid, basis_statuses_empty;
  std::optional<HighsInt> simplex_iterations, ipm_iterations, crossover_iterations;
  bool positive_iterations_observed = false;
};

std::vector<Detail> details;
int recovery_cases = 0, rejection_cases = 0, budget_cases = 0, contract_cases = 0;
int min_cases = 0, max_cases = 0, empty_reduced_cases = 0, nonempty_reduced_cases = 0;

void observe_no_lp(Highs& highs, Detail& detail) {
  const HighsInfo& info = highs.getInfo();
  detail.highs_observed = true;
  detail.basis_valid = highs.getBasis().valid;
  detail.basis_statuses_empty = highs.getBasis().col_status.empty() &&
                                highs.getBasis().row_status.empty();
  detail.simplex_iterations = info.simplex_iteration_count;
  detail.ipm_iterations = info.ipm_iteration_count;
  detail.crossover_iterations = info.crossover_iteration_count;
  detail.positive_iterations_observed = info.simplex_iteration_count > 0 ||
      info.ipm_iteration_count > 0 || info.crossover_iteration_count > 0 ||
      info.pdlp_iteration_count > 0 || info.qp_iteration_count > 0;
  // invalidate 的原始哨兵是 −1；保留原值，不把未报告伪装成实测零。
  for (HighsInt value : {info.simplex_iteration_count, info.ipm_iteration_count,
                         info.crossover_iteration_count, info.pdlp_iteration_count,
                         info.qp_iteration_count}) {
    require(value == -1 || value == 0, "组件不得观察到任何 LP/QP 迭代");
  }
  // 对象的返回数组会由上游统一调整尺寸；一参入口另外新建的空 basis 才是输入合同。
  require(!highs.getBasis().valid, "纯恢复前后不得观察到有效 basis");
}

void configure(Highs& highs) {
  option(highs, "output_flag", false);
  option(highs, "log_to_console", false);
  option(highs, "threads", 1);
  option(highs, "random_seed", 0);
  option(highs, "presolve", "choose");
  option(highs, "allow_unbounded_or_infeasible", false);
  // 这些保护仅限制意外求解；本文件仍完全不调用求解或 crossover API。
  option(highs, "simplex_iteration_limit", 0);
  option(highs, "ipm_iteration_limit", 0);
  option(highs, "run_crossover", "off");
}

void presolve(Highs& highs, const solver::Model& original,
              const SharedDeadline& deadline, Detail& detail) {
  configure(highs);
  require(deadline.call([&] { return highs.passModel(minimum_copy(original)); }) == HighsStatus::kOk,
          "最小化副本不能载入 HiGHS");
  detail.original_columns = static_cast<HighsInt>(original.costs.size());
  detail.original_rows = static_cast<HighsInt>(original.row_lower.size());
  option(highs, "time_limit", highs.getRunTime() + deadline.remaining());
  require(deadline.call([&] { return highs.presolve(); }) == HighsStatus::kOk,
          "小模型预处理必须在共享预算内正常完成");
  detail.presolve_status = static_cast<int>(highs.getModelPresolveStatus());
  detail.reduced_columns = highs.getPresolvedLp().num_col_;
  detail.reduced_rows = highs.getPresolvedLp().num_row_;
  observe_no_lp(highs, detail);
}

enum class InputKind { empty, nonempty, bad_dual, bad_primal, wrong_dual_size };

HighsSolution reduced_candidate(Highs& highs, InputKind kind) {
  const auto reduced = own_model(highs.getPresolvedLp());
  if (kind == InputKind::empty) {
    require(reduced.costs.empty() && reduced.row_lower.empty(),
            "该恢复用例必须实际约化为空，不能猜测剩余变量映射");
    return candidate(reduced, {}, {});
  }
  // 允许上游传播变量界；若发生别的行列消去/缩放，则明确失败而不猜映射。
  require(reduced.costs == std::vector<double>({-1, -1}) &&
          reduced.row_lower.size() == 2 &&
          reduced.column_start == std::vector<std::int64_t>({0, 2, 4}) &&
          reduced.row_index == std::vector<std::int64_t>({0, 1, 0, 1}) &&
          reduced.values == std::vector<double>({2, 1, 1, 2}),
          "非空约化必须保留可直接验证的解析二列二行方程");
  std::vector<double> x{4.0 / 3.0, 4.0 / 3.0}, y{-1.0 / 3.0, -1.0 / 3.0};
  require(accepted(solver::verify_lp_solution(reduced, x, y)),
          "非空约化解析点须先通过实际约化模型完整质量检查");
  if (kind == InputKind::bad_dual) y.assign(2, 0.0);
  if (kind == InputKind::bad_primal) x.assign(2, 3.0);
  auto result = candidate(reduced, std::move(x), std::move(y));
  if (kind == InputKind::wrong_dual_size) result.row_dual.pop_back();
  return result;
}

void recovery(const char* id, const solver::Model& original, InputKind kind,
               double expected_objective) {
  SharedDeadline deadline;
  Highs highs;
  const bool expect = kind == InputKind::empty || kind == InputKind::nonempty;
  Detail detail{id, kind == InputKind::wrong_dual_size ? "api_rejection" :
                    (expect ? "recovery" : "quality_rejection")};
  detail.expected_accept = expect;
  presolve(highs, original, deadline, detail);
  require(highs.getModelPresolveStatus() == HighsPresolveStatus::kNotReduced ||
          highs.getModelPresolveStatus() == HighsPresolveStatus::kReduced ||
          highs.getModelPresolveStatus() == HighsPresolveStatus::kReducedToEmpty,
          "只有明确可恢复的预处理状态才能进入一参恢复");
  const HighsSolution input = deadline.call([&] { return reduced_candidate(highs, kind); });
  const HighsStatus status = deadline.call([&] {
    // 唯一恢复入口：不提供 basis；绝不能替换为有尺寸的无效 basis。
    return highs.postsolve(input);
  });
  detail.postsolve_status = static_cast<int>(status);
  observe_no_lp(highs, detail);
  if (kind == InputKind::wrong_dual_size) {
    require(status == HighsStatus::kError, "缺失一部分行对偶必须由真实恢复 API 拒绝");
    ++rejection_cases;
    details.push_back(std::move(detail));
    return;
  }
  require(status != HighsStatus::kError, "合法尺寸的恢复不应发生 API 错误");
  const auto& restored = highs.getSolution();
  require(restored.value_valid && restored.dual_valid &&
          restored.col_value.size() == original.costs.size() &&
          restored.row_dual.size() == original.row_lower.size(),
          "恢复必须提供原维度完整原始与行对偶向量");
  // restored.row_dual 已对应最小化副本，原模型检查器内部也使用 sense*c。
  // 最大化在这里再次翻转对偶符号会错误，故直接交给同一原模型验收器。
  const auto quality = deadline.call([&] {
    return solver::verify_lp_solution(original, restored.col_value, restored.row_dual);
  });
  detail.original_verified = true;
  detail.accepted = accepted(quality);
  detail.original_objective = quality.primal.objective;
  detail.objective_gap = quality.optimality.objective_gap;
  require(detail.accepted == expect, "原模型完整质量必须符合好/坏候选合同");
  if (expect) {
    require(quality.primal.objective &&
            std::abs(*quality.primal.objective - expected_objective) <= 1e-10,
            "原方向目标和 offset 必须恢复正确");
    ++recovery_cases;
    if (original.sense == solver::ObjectiveSense::maximize) ++max_cases;
    else ++min_cases;
    if (detail.reduced_columns == 0 && detail.reduced_rows == 0) ++empty_reduced_cases;
    else ++nonempty_reduced_cases;
  } else {
    if (kind == InputKind::bad_dual) require(!quality.optimality.dual_feasible,
                                            "错误对偶必须由原模型对偶质量拒绝");
    if (kind == InputKind::bad_primal) require(!quality.primal.feasible,
                                              "错误原始点必须由原行可行性拒绝");
    ++rejection_cases;
  }
  details.push_back(std::move(detail));
}

void infeasible_is_not_certificate() {
  auto original = scalar(0, 0, infinity);
  row(original, 1, -infinity, -1);
  SharedDeadline deadline;
  Highs highs;
  Detail detail{"presolve_infeasible_is_not_our_certificate", "status_rejection"};
  presolve(highs, original, deadline, detail);
  require(highs.getModelPresolveStatus() == HighsPresolveStatus::kInfeasible,
          "解析矛盾界应被预处理识别");
  // 不调用 postsolve，不将上游状态升级为自主算法给出的不可行证书。
  require(!detail.accepted && !detail.original_verified, "上游状态不能代替自主原模型证书");
  ++rejection_cases;
  details.push_back(std::move(detail));
}

void basis_contract() {
  HighsBasis basis;
  require(empty_invalid_basis(basis), "默认 basis 必须真正为空且无效");
  basis.col_status.assign(2, HighsBasisStatus::kLower);
  basis.row_status.assign(2, HighsBasisStatus::kBasic);
  require(!basis.valid && !empty_invalid_basis(basis),
          "有尺寸的 invalid basis 必须被合同拒绝，不能送入二参 postsolve");
  Detail detail{"sized_invalid_basis_is_not_no_basis", "basis_contract"};
  ++contract_cases;
  details.push_back(std::move(detail));
}

void deadline_contracts() {
  const auto epoch = Clock::time_point{};
  int ticks = 0, invoked = 0, committed = 11;
  SharedDeadline expired([&] { return epoch + std::chrono::seconds(ticks++ == 0 ? 0 : 6); });
  bool caught = false;
  try { committed = expired.call([&] { ++invoked; return 22; }); }
  catch (const SharedDeadlineExpired&) { caught = true; }
  require(caught && invoked == 0 && committed == 11,
          "入口过期必须保留预算异常类型且不调用组件或提交候选");
  ++budget_cases;
  details.push_back(Detail{"shared_deadline_before_call", "budget"});

  ticks = 0; invoked = 0; committed = 11; caught = false;
  SharedDeadline expires_after([&] { return epoch + std::chrono::seconds(ticks++ < 2 ? 0 : 6); });
  try { committed = expires_after.call([&] { ++invoked; return 22; }); }
  catch (const SharedDeadlineExpired&) { caught = true; }
  require(caught && invoked == 1 && committed == 11,
          "同步组件返回后过期也不得提交返回值或重新获得五秒");
  ++budget_cases;
  details.push_back(Detail{"shared_deadline_after_call", "budget"});
}

template <class Integer>
void json_integer(const std::optional<Integer>& value) {
  if (value) std::printf("%lld", static_cast<long long>(*value));
  else std::printf("null");
}

void json_bool(const std::optional<bool>& value) {
  if (value) std::printf("%s", *value ? "true" : "false");
  else std::printf("null");
}

void json_number(const std::optional<double>& value) {
  if (value && std::isfinite(*value)) std::printf("%.17g", *value);
  else std::printf("null");
}

void print_detail(const Detail& item) {
  std::printf("{\"id\":\"%s\",\"kind\":\"%s\",\"highs_observed\":%s,"
      "\"original_columns\":%lld,\"original_rows\":%lld,\"reduced_columns\":%lld,\"reduced_rows\":%lld,"
      "\"presolve_status\":", item.id, item.kind, item.highs_observed ? "true" : "false",
      static_cast<long long>(item.original_columns), static_cast<long long>(item.original_rows),
      static_cast<long long>(item.reduced_columns), static_cast<long long>(item.reduced_rows));
  json_integer(item.presolve_status);
  std::printf(",\"postsolve_status\":"); json_integer(item.postsolve_status);
  std::printf(",\"expected_accept\":%s,\"accepted\":%s,\"original_verified\":%s,\"original_objective\":",
      item.expected_accept ? "true" : "false", item.accepted ? "true" : "false",
      item.original_verified ? "true" : "false");
  json_number(item.original_objective);
  std::printf(",\"objective_gap\":"); json_number(item.objective_gap);
  std::printf(",\"basis_valid\":"); json_bool(item.basis_valid);
  std::printf(",\"basis_statuses_empty\":"); json_bool(item.basis_statuses_empty);
  std::printf(",\"simplex_iterations\":"); json_integer(item.simplex_iterations);
  std::printf(",\"ipm_iterations\":"); json_integer(item.ipm_iterations);
  std::printf(",\"crossover_iterations\":"); json_integer(item.crossover_iterations);
  std::printf(",\"positive_iterations_observed\":%s}",
      item.positive_iterations_observed ? "true" : "false");
}
}  // 命名空间

int main() {
  try {
    recovery("minimum_column_lower_offset", scalar(2, 1, 4, 7), InputKind::empty, 9);
    recovery("minimum_column_upper_negative_offset", scalar(-2, -5, -2, -3), InputKind::empty, 1);
    recovery("maximum_column_upper_offset", scalar(3, -1, 4, 7, true), InputKind::empty, 19);
    auto model = scalar(3, -infinity, infinity, -4);
    row(model, 2, 4, infinity);
    recovery("minimum_row_lower_origin", model, InputKind::empty, 2);
    model = scalar(2, -infinity, infinity, 7, true);
    row(model, 1, -infinity, 3);
    recovery("maximum_row_upper_origin", model, InputKind::empty, 13);
    model = scalar(3, -5, 8);
    row(model, 1, 2, 5);
    recovery("ranged_row_lower_origin", model, InputKind::empty, 6);
    model.sense = solver::ObjectiveSense::maximize;
    recovery("ranged_row_upper_origin", model, InputKind::empty, 15);
    model = scalar(7, 3, 3, 11);
    model.costs.push_back(3); model.column_lower.push_back(-infinity);
    model.column_upper.push_back(infinity); model.kinds.push_back(solver::VariableKind::continuous);
    model.row_lower = {19}; model.row_upper = {19};
    model.column_start = {0, 1, 2}; model.row_index = {0, 0}; model.values = {5, 2};
    recovery("fixed_column_equality_and_offset", model, InputKind::empty, 38);
    recovery("minimum_nonempty_analytic_reduced", coupled(), InputKind::nonempty, -17.0 / 3.0);
    recovery("maximum_nonempty_analytic_reduced", coupled(true), InputKind::nonempty, 29.0 / 3.0);
    recovery("wrong_dual_rejected_on_original", coupled(), InputKind::bad_dual, 0);
    recovery("wrong_primal_rejected_on_original", coupled(), InputKind::bad_primal, 0);
    recovery("incomplete_dual_rejected_by_api", coupled(), InputKind::wrong_dual_size, 0);
    infeasible_is_not_certificate();
    basis_contract();
    deadline_contracts();
    require(recovery_cases == 10 && rejection_cases == 4 && contract_cases == 1 && budget_cases == 2 &&
            min_cases > 0 && max_cases > 0 && empty_reduced_cases > 0 && nonempty_reduced_cases > 0 &&
            details.size() == static_cast<std::size_t>(recovery_cases + rejection_cases + contract_cases + budget_cases),
            "组件覆盖必须按真实完成的用例、方向与约化维度守恒");
    // 显式求解 API 调用为零；不是对预编译库内部函数的二进制拦截计数。
    // 上游内部不进入 cleanup 的证据另由一参入口源码审计与空 basis 观察提供。
    std::printf("{\"passed\":true,\"public_lp_runs\":0,\"explicit_lp_solve_calls\":0,"
        "\"cases\":%zu,\"recovery_cases\":%d,\"rejection_cases\":%d,\"budget_cases\":%d,"
        "\"contract_cases\":%d,\"min_cases\":%d,\"max_cases\":%d,"
        "\"empty_reduced_cases\":%d,\"nonempty_reduced_cases\":%d,\"details\":[",
        details.size(), recovery_cases, rejection_cases, budget_cases, contract_cases,
        min_cases, max_cases, empty_reduced_cases, nonempty_reduced_cases);
    for (std::size_t i = 0; i < details.size(); ++i) {
      if (i) std::printf(",");
      print_detail(details[i]);
    }
    std::printf("]}\n");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "组件自测失败：%s\n", error.what());
    return 1;
  }
}
