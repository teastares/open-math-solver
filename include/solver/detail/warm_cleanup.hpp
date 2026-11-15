#pragma once

// 自主有界原始/对偶单纯形清理；可选零目标可行化，数值分解和列更新复用BASICLU。
// BASICLU归属ERGO-Code，按锁定依赖许可证使用。
// 锁定来源为HiGHS 1.15.1的highs/ipm/basiclu；保留原依赖的许可证和归属。
#include "solver/detail/ipx_crossover.hpp"
#include "ipm/basiclu/basiclu.h"
#include <type_traits>

namespace solver::detail::warm_cleanup {
namespace cross = solver::detail::ipx_crossover;
using Clock = cross::Clock;
using Index = ipx::Int;
using LD = long double;
static_assert(std::is_same_v<Index, lu_int>);

enum class Status {
  accepted, rejected, invalid_input, unsupported_empty, singular_basis,
  primal_infeasible, dual_infeasible, numerical_error, step_limit, no_finite_step, time_limit, exception
};
inline const char* status_name(Status value) noexcept {
  switch (value) {
    case Status::accepted: return "accepted";
    case Status::rejected: return "rejected";
    case Status::invalid_input: return "invalid_input";
    case Status::unsupported_empty: return "unsupported_empty";
    case Status::singular_basis: return "singular_basis";
    case Status::primal_infeasible: return "primal_infeasible";
    case Status::dual_infeasible: return "dual_infeasible";
    case Status::numerical_error: return "numerical_error";
    case Status::step_limit: return "step_limit";
    case Status::no_finite_step: return "no_finite_step";
    case Status::time_limit: return "time_limit";
    case Status::exception: return "exception";
  }
  return "exception";
}
struct Failure : std::runtime_error {
  Status status;
  Failure(Status value, const char* message) : std::runtime_error(message), status(value) {}
};
struct Input {
  const solver::Model& model;
  const cross::Prepared& prepared;
  std::span<const Index> variable_status, row_status;
  std::span<const double> structural_seed;
};
struct Limits {
  std::uint64_t max_steps = 64;
  bool dual_cleanup = false;
  bool feasibility_first = false;
  bool incremental_feasibility_update = false;
  bool incremental_objective_dual_update = false;
};
struct CallStats {
  std::uint64_t started = 0, completed = 0;
  std::optional<Index> rc;
  double seconds = 0;
};
struct Checks {
  bool valid = false, primal_feasible = false, basis_dual_feasible = false;
  LD equation_residual = 0, bound_violation = 0, basic_dual_residual = 0;
  LD equation_ratio = 0, bound_ratio = 0, basic_dual_ratio = 0;
  bool nonbasic_dual_checked = false, nonbasic_dual_feasible = false;
  LD nonbasic_dual_violation = 0, nonbasic_dual_ratio = 0;
};
struct FeasibilityUpdateLive {
  bool enabled = false;
  std::uint64_t attempted = 0, predicted = 0, verification_started = 0, verification_completed = 0;
  std::uint64_t accepted = 0, fallback = 0, fallback_rebuild_started = 0, fallback_rebuild_completed = 0;
  std::uint64_t verification_columns = 0, verification_entries = 0, work_items = 0;
  std::uint64_t refresh_started = 0, refresh_completed = 0;
  LD max_equation_ratio = 0, max_bound_ratio = 0;
};
struct ObjectiveUpdateLive {
  bool enabled = false;
  std::uint64_t attempted = 0, predicted = 0, verification_started = 0, verification_completed = 0;
  std::uint64_t accepted = 0, fallback = 0, prediction_failures = 0, equation_failures = 0;
  std::uint64_t basic_dual_failures = 0, nonbasic_dual_failures = 0, reduced_failures = 0;
  std::uint64_t fallback_rebuild_started = 0, fallback_rebuild_completed = 0;
  std::uint64_t refresh_started = 0, refresh_completed = 0;
  std::uint64_t repricing_retries = 0;
  std::uint64_t verification_columns = 0, verification_entries = 0, work_items = 0;
  double update_seconds = 0, verification_seconds = 0, fallback_rebuild_seconds = 0, refresh_seconds = 0;
  LD max_equation_ratio = 0, max_bound_ratio = 0, max_basic_dual_ratio = 0;
  LD max_nonbasic_dual_ratio = 0, max_reduced_residual_ratio = 0;
};
struct Stats {
  CallStats init, factor, solve_n, solve_t, ftran, btran, update;
  std::uint64_t steps_started = 0, steps_completed = 0, pivots = 0, bound_flips = 0, zero_steps = 0;
  std::uint64_t refactors = 0, rebuild_started = 0, rebuild_completed = 0, pricing_passes = 0, work_items = 0;
  std::optional<Checks> initial_checks, last_checks;
  std::optional<Index> last_entering, last_leaving;
  std::optional<LD> last_theta, last_pivot;
  std::uint64_t verification_started = 0, verification_completed = 0;
  cross::ProjectionStats projection;
  // rebuild_seconds含两次solve及启用时的完整非基检查；不得与这些子项重复加总。
  double pricing_seconds = 0, rebuild_seconds = 0, verification_seconds = 0, total_seconds = 0;
  const char* stage = "entry";
  bool dual_enabled = false, dual_entered = false;
  std::optional<Checks> dual_entry_checks;
  CallStats dual_row_solve;
  std::uint64_t dual_checks_started = 0, dual_checks_completed = 0;
  std::uint64_t dual_steps_started = 0, dual_steps_completed = 0, dual_pivots = 0, dual_pricing_passes = 0;
  std::uint64_t dual_zero_steps = 0;
  std::optional<LD> last_dual_theta, last_dual_primal_step;
  double dual_check_seconds = 0, dual_pricing_seconds = 0;
  std::uint64_t dual_ratio_passes_started = 0, dual_ratio_passes = 0, dual_band_candidates = 0;
  std::uint64_t dual_tableau_entries_peak = 0;
  std::optional<LD> last_dual_relaxed_theta, last_dual_selected_alpha;
  bool feasibility_started = false, objective_restored = false;
  std::uint64_t feasibility_steps = 0;
  FeasibilityUpdateLive feasibility_update_live;
  ObjectiveUpdateLive objective_update_live;
};
struct Result {
  Status status = Status::exception;
  std::string message;
  std::vector<double> x, y;
  // 与交叉组件相同：仅第二验前保存raw；否则verification就是未投影报告。
  std::optional<solver::LpVerification> raw_verification, verification;
  Stats stats;
};

namespace detail {
struct Timed {
  double& seconds;
  Clock::time_point start = Clock::now();
  explicit Timed(double& destination) : seconds(destination) {}
  ~Timed() { seconds += std::chrono::duration<double>(Clock::now() - start).count(); }
};
template<class Budget> struct Work {
  Clock::time_point deadline;
  Budget& budget;
  Stats& stats;
  void check() {
    if (Clock::now() >= deadline) throw Failure(Status::time_limit, "暖基原型已达到绝对截止时点");
    budget();
    if (Clock::now() >= deadline) throw Failure(Status::time_limit, "暖基原型预算回调后已超时");
  }
  void tick() {
    if ((stats.work_items & 4095U) == 0) check();
    ++stats.work_items;
  }
};
struct FactorOwner {
  basiclu_object object{};
  ~FactorOwner() { basiclu_obj_free(&object); }
  FactorOwner() = default;
  FactorOwner(const FactorOwner&) = delete;
  FactorOwner& operator=(const FactorOwner&) = delete;
};
inline double narrow(LD value) {
  const double result = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(result) || (value != 0 && result == 0))
    throw Failure(Status::numerical_error, "暖基计算不能有限且非零保持地存为double");
  return result;
}
inline void add(LD& value, LD term) {
  value += term;
  if (!std::isfinite(term) || !std::isfinite(value))
    throw Failure(Status::numerical_error, "暖基残差或矩阵乘积非有限");
}
inline LD ratio(LD error, LD allowed) {
  return allowed > 0 ? error / allowed : error == 0 ? 0 : std::numeric_limits<LD>::infinity();
}
inline bool valid_bounds(double lower, double upper) {
  const double inf = std::numeric_limits<double>::infinity();
  return !std::isnan(lower) && !std::isnan(upper) && lower != inf && upper != -inf && lower <= upper;
}
inline bool quality_ok(const solver::LpVerification& checked) {
  return checked.primal.valid_input && checked.primal.feasible && checked.primal.objective_consistent &&
      checked.primal.objective.has_value() && checked.optimality.valid_input && checked.optimality.numerically_optimal;
}
// 复用真实验算子的账本在异常展开时也必须交回，不能丢失二验中断前缀。
struct AssessmentCollect {
  Result& target;
  cross::Result& source;
  ~AssessmentCollect() noexcept {
    static_assert(std::is_nothrow_move_assignable_v<std::optional<solver::LpVerification>>);
    target.raw_verification = std::move(source.raw_verification);
    target.verification = std::move(source.verification);
    target.stats.verification_started = source.stats.verification_started;
    target.stats.verification_completed = source.stats.verification_completed;
    target.stats.verification_seconds = source.stats.verification_seconds;
    target.stats.projection = source.stats.projection;
  }
};
}  // 命名空间detail

template<class Budget>
Result run(const Input& input, Clock::time_point absolute_deadline, const Limits& limits,
    const solver::Tolerances& tolerances, const solver::LpOptimalityOptions& optimality_options, Budget&& budget) {
  const auto start = Clock::now();
  Result result;
  result.stats.dual_enabled = limits.dual_cleanup || limits.feasibility_first;
  result.stats.feasibility_update_live.enabled = limits.incremental_feasibility_update;
  result.stats.objective_update_live.enabled = limits.incremental_objective_dual_update;
  detail::Work<Budget> work{absolute_deadline, budget, result.stats};
  try {
    {
      auto& stats = result.stats;
      stats.stage = "validation";
      work.check();
      cross::detail::validate_tolerances(tolerances, optimality_options);
      const auto errors = solver::validate_model(input.model);
      work.check();
      if (!errors.empty() || solver::is_mip(input.model))
        throw Failure(Status::invalid_input, "暖基原型只接受有效连续原模型");
      const auto& p = input.prepared;
      const std::size_t n = p.costs.size(), m = p.rhs.size();
      if (n == 0 || m == 0) throw Failure(Status::unsupported_empty, "暖基原型不构造零维基");
      const auto maximum = static_cast<std::size_t>(LU_INT_MAX);
      if (n > maximum || m > maximum - n || m > (maximum - BASICLU_SIZE_ISTORE_1) / BASICLU_SIZE_ISTORE_M ||
          p.values.size() > maximum || p.original_columns != input.model.costs.size() ||
          p.original_rows != input.model.row_lower.size() || p.original_columns > n ||
          n - p.original_columns != p.auxiliary_row.size() || p.original_row.size() != m ||
          p.lower.size() != n || p.upper.size() != n || p.row_type.size() != m ||
          p.column_start.size() != n + 1 || p.row_index.size() != p.values.size() ||
          p.column_start.front() != 0 || p.column_start.back() != static_cast<Index>(p.values.size()) ||
          input.variable_status.size() != n || input.row_status.size() != m || input.structural_seed.size() != n)
        throw Failure(Status::invalid_input, "暖基输入维数或整数范围无效");
      std::vector<bool> mapped(p.original_rows, false), auxiliary(p.original_rows, false);
      for (std::size_t row = 0; row < m; ++row) {
        work.tick();
        const auto original = p.original_row[row];
        if (original >= mapped.size() || mapped[original] || !std::isfinite(p.rhs[row]) ||
            (p.row_type[row] != '=' && p.row_type[row] != '<' && p.row_type[row] != '>'))
          throw Failure(Status::invalid_input, "暖基原行映射或行类型无效");
        mapped[original] = true;
      }
      for (const auto original : p.auxiliary_row) {
        work.tick();
        if (original >= auxiliary.size() || !mapped[original] || auxiliary[original])
          throw Failure(Status::invalid_input, "暖基范围辅助行映射无效");
        auxiliary[original] = true;
      }
      for (std::size_t col = 0; col < n; ++col) {
        work.tick();
        if (!std::isfinite(p.costs[col]) || !detail::valid_bounds(p.lower[col], p.upper[col]) ||
            p.column_start[col] < 0 || p.column_start[col] > p.column_start[col + 1] ||
            p.column_start[col + 1] > static_cast<Index>(p.values.size()))
          throw Failure(Status::invalid_input, "暖基结构列、界或CSC指针无效");
        Index previous = -1;
        for (Index k = p.column_start[col]; k < p.column_start[col + 1]; ++k) {
          work.tick();
          const auto pos = static_cast<std::size_t>(k);
          if (p.row_index[pos] <= previous || p.row_index[pos] >= static_cast<Index>(m) || !std::isfinite(p.values[pos]))
            throw Failure(Status::invalid_input, "暖基CSC索引、排序或数值无效");
          previous = p.row_index[pos];
        }
      }
      const std::size_t total = n + m;
      const double inf = std::numeric_limits<double>::infinity();
      std::vector<double> lower(total), upper(total), costs(total), nonbasic(total, 0), values(total, 0), pi(m);
      std::vector<Index> basis, position(total, -1);
      basis.reserve(m);
      for (std::size_t col = 0; col < total; ++col) {
        work.tick();
        Index state;
        if (col < n) {
          lower[col] = p.lower[col]; upper[col] = p.upper[col]; costs[col] = p.costs[col];
          state = input.variable_status[col];
        } else {
          const auto row = col - n;
          lower[col] = p.row_type[row] == '>' ? -inf : 0;
          upper[col] = p.row_type[row] == '<' ? inf : 0;
          costs[col] = 0;
          state = input.row_status[row];
          if (state != IPX_basic && state != IPX_nonbasic)
            throw Failure(Status::invalid_input, "暖基逻辑列状态无效");
          if (state == IPX_nonbasic) state = p.row_type[row] == '>' ? IPX_nonbasic_ub : IPX_nonbasic_lb;
        }
        if (state == IPX_basic) {
          if (basis.size() >= m) throw Failure(Status::invalid_input, "暖基基本列过多");
          position[col] = static_cast<Index>(basis.size()); basis.push_back(static_cast<Index>(col));
        } else if (state == IPX_nonbasic_lb) {
          if (!std::isfinite(lower[col])) throw Failure(Status::invalid_input, "下界非基列缺少有限下界");
          nonbasic[col] = lower[col];
        } else if (state == IPX_nonbasic_ub) {
          if (!std::isfinite(upper[col])) throw Failure(Status::invalid_input, "上界非基列缺少有限上界");
          nonbasic[col] = upper[col];
        } else if (state == IPX_superbasic && col < n) {
          nonbasic[col] = input.structural_seed[col];
          if (!std::isfinite(nonbasic[col]) || nonbasic[col] < lower[col] || nonbasic[col] > upper[col])
            throw Failure(Status::invalid_input, "超基列种子非有限或越界");
        } else throw Failure(Status::invalid_input, "暖基结构列状态无效");
      }
      if (basis.size() != m) throw Failure(Status::invalid_input, "暖基基本列数不等于行数");
      // 逻辑+I列按需访问，不生成整块扩展矩阵，也不转换为旧行基。
      const auto column = [&](std::size_t col, auto&& visit) {
        if (col < n) {
          for (Index k = p.column_start[col]; k < p.column_start[col + 1]; ++k) {
            work.tick(); const auto pos = static_cast<std::size_t>(k);
            visit(static_cast<std::size_t>(p.row_index[pos]), p.values[pos]);
          }
        } else { work.tick(); visit(col - n, 1.0); }
      };
      detail::FactorOwner owner;
      auto& lu = owner.object;
      const auto native = [&](CallStats& calls, const char* stage, auto&& operation) -> Index {
        stats.stage = stage;
        detail::Timed timer(calls.seconds);
        work.check(); ++calls.started;
        const Index rc = operation();
        calls.rc = rc; ++calls.completed;
        work.check();
        return rc;
      };
      const auto require_ok = [](Index rc) {
        if (rc != BASICLU_OK) throw Failure(Status::numerical_error, "BASICLU暖基运算失败");
      };
      require_ok(native(stats.init, "initialize", [&] { return basiclu_obj_initialize(&lu, static_cast<Index>(m)); }));
      std::vector<Index> bp, bi;
      std::vector<double> bx;
      std::uint64_t updates_since_factor = 0;
      const auto factorize = [&](bool refactor) {
        stats.stage = "factor"; work.check();
        bp.clear(); bi.clear(); bx.clear(); bp.push_back(0);
        for (const Index basic : basis) {
          work.tick();
          column(static_cast<std::size_t>(basic), [&](std::size_t row, double value) {
            if (bi.size() == maximum) throw Failure(Status::invalid_input, "暖基因子输入非零数超出整数范围");
            bi.push_back(static_cast<Index>(row)); bx.push_back(value);
          });
          bp.push_back(static_cast<Index>(bi.size()));
        }
        const Index dummy_index = 0;
        const double dummy_value = 0;
        const Index rc = native(stats.factor, "factor", [&] {
          if (refactor) ++stats.refactors;
          return basiclu_obj_factorize(&lu, bp.data(), bp.data() + 1,
              bi.empty() ? &dummy_index : bi.data(), bx.empty() ? &dummy_value : bx.data());
        });
        if (rc == BASICLU_WARNING_singular_matrix)
          throw Failure(Status::singular_basis, "给定基数值奇异，不接受BASICLU自动补入单位列");
        require_ok(rc);
        if (lu.xstore[BASICLU_RANK] != static_cast<double>(m))
          throw Failure(Status::singular_basis, "给定基的数值秩不足");
        updates_since_factor = 0;
      };
      factorize(false);
      std::vector<double> rhs(m), basic_values(m), dual_rhs(m);
      std::vector<LD> rhs_wide(m), activity(m), products(total), reduced(total);
      bool dual_active = false, feasibility_active = false, live_predicate = false;
      const auto check_nonbasic_dual = [&](bool may_start_feasibility = false) {
        stats.stage = "dual_check";
        detail::Timed timer(stats.dual_check_seconds);
        work.check(); ++stats.dual_checks_started;
        LD violation = 0, maximum_ratio = 0;
        for (std::size_t col = 0; col < total; ++col) {
          work.tick();
          if (position[col] >= 0 || lower[col] == upper[col]) continue;
          // 依据真实非基位置，不将IPX的-3状态误当作固定的移动方向。
          const LD error = values[col] == lower[col] ? std::max<LD>(0, -reduced[col]) :
              values[col] == upper[col] ? std::max<LD>(0, reduced[col]) : std::abs(reduced[col]);
          const LD allowed = static_cast<LD>(optimality_options.dual_absolute) +
              static_cast<LD>(optimality_options.dual_relative) * std::max(std::abs(static_cast<LD>(costs[col])), std::abs(products[col]));
          violation = std::max(violation, error);
          maximum_ratio = std::max(maximum_ratio, detail::ratio(error, allowed));
        }
        auto& checks = *stats.last_checks;
        checks.nonbasic_dual_checked = true;
        checks.nonbasic_dual_feasible = maximum_ratio <= 1;
        checks.nonbasic_dual_violation = violation; checks.nonbasic_dual_ratio = maximum_ratio;
        if (stats.rebuild_completed == 1) stats.initial_checks = checks;
        ++stats.dual_checks_completed; work.check();
        if (!checks.nonbasic_dual_feasible && !may_start_feasibility)
          throw Failure(Status::dual_infeasible, "暖基完整非基约化成本不满足当前位置的对偶资格，未执行对偶Phase I");
        return checks.nonbasic_dual_feasible;
      };
      const auto rebuild = [&](bool may_start_feasibility = false, bool restoring = false) {
        stats.stage = "rebuild";
        detail::Timed timer(stats.rebuild_seconds);
        work.check(); ++stats.rebuild_started;
        for (std::size_t row = 0; row < m; ++row) { work.tick(); rhs_wide[row] = p.rhs[row]; }
        for (std::size_t col = 0; col < total; ++col) {
          work.tick(); values[col] = nonbasic[col];
          if (position[col] < 0)
            column(col, [&](std::size_t row, double a) { detail::add(rhs_wide[row], -static_cast<LD>(a) * nonbasic[col]); });
        }
        for (std::size_t row = 0; row < m; ++row) {
          work.tick(); rhs[row] = detail::narrow(rhs_wide[row]); dual_rhs[row] = costs[static_cast<std::size_t>(basis[row])];
        }
        require_ok(native(stats.solve_n, "solve_n", [&] { return basiclu_obj_solve_dense(&lu, rhs.data(), basic_values.data(), 'N'); }));
        require_ok(native(stats.solve_t, "solve_t", [&] { return basiclu_obj_solve_dense(&lu, dual_rhs.data(), pi.data(), 'T'); }));
        stats.stage = "rebuild";
        // 零成本时随原重建检查可复用的对偶状态。
        live_predicate = limits.incremental_feasibility_update && feasibility_active;
        for (std::size_t row = 0; row < m; ++row) {
          work.tick();
          if (!std::isfinite(basic_values[row]) || !std::isfinite(pi[row]))
            throw Failure(Status::numerical_error, "暖基重建点或转置解非有限");
          if (live_predicate && pi[row] != 0) live_predicate = false;
          values[static_cast<std::size_t>(basis[row])] = basic_values[row]; activity[row] = 0;
        }
        Checks checks;
        for (std::size_t col = 0; col < total; ++col) {
          work.tick(); products[col] = 0;
          column(col, [&](std::size_t row, double a) {
            detail::add(activity[row], static_cast<LD>(a) * values[col]);
            detail::add(products[col], static_cast<LD>(a) * pi[row]);
          });
          reduced[col] = static_cast<LD>(costs[col]) - products[col];
          if (!std::isfinite(reduced[col])) throw Failure(Status::numerical_error, "暖基约化成本非有限");
          if (live_predicate && (costs[col] != 0 || products[col] != 0 || reduced[col] != 0)) live_predicate = false;
          for (int side = 0; side < 2; ++side) {
            const double bound = side == 0 ? lower[col] : upper[col];
            if (!std::isfinite(bound)) continue;
            const LD violation = std::max<LD>(0, side == 0 ? static_cast<LD>(bound) - values[col] : static_cast<LD>(values[col]) - bound);
            const LD allowed = static_cast<LD>(tolerances.feasibility_absolute) +
                static_cast<LD>(tolerances.feasibility_relative) * std::max(std::abs(static_cast<LD>(values[col])), std::abs(static_cast<LD>(bound)));
            checks.bound_violation = std::max(checks.bound_violation, violation);
            checks.bound_ratio = std::max(checks.bound_ratio, detail::ratio(violation, allowed));
          }
          if (position[col] >= 0) {
            const LD error = std::abs(reduced[col]);
            const LD allowed = static_cast<LD>(optimality_options.dual_absolute) +
                static_cast<LD>(optimality_options.dual_relative) * std::max(std::abs(static_cast<LD>(costs[col])), std::abs(products[col]));
            checks.basic_dual_residual = std::max(checks.basic_dual_residual, error);
            checks.basic_dual_ratio = std::max(checks.basic_dual_ratio, detail::ratio(error, allowed));
          }
        }
        for (std::size_t row = 0; row < m; ++row) {
          work.tick();
          const LD error = std::abs(activity[row] - p.rhs[row]);
          const LD allowed = static_cast<LD>(tolerances.feasibility_absolute) +
              static_cast<LD>(tolerances.feasibility_relative) * std::max(std::abs(activity[row]), std::abs(static_cast<LD>(p.rhs[row])));
          checks.equation_residual = std::max(checks.equation_residual, error);
          checks.equation_ratio = std::max(checks.equation_ratio, detail::ratio(error, allowed));
        }
        checks.valid = true;
        checks.primal_feasible = checks.equation_ratio <= 1 && checks.bound_ratio <= 1;
        checks.basis_dual_feasible = checks.basic_dual_ratio <= 1;
        stats.last_checks = checks;
        if (!stats.initial_checks) stats.initial_checks = checks;
        ++stats.rebuild_completed; work.check();
        if (!checks.primal_feasible) {
          if (may_start_feasibility && checks.equation_ratio <= 1 && !checks.basis_dual_feasible) return;
          if (restoring || !stats.dual_enabled || checks.equation_ratio > 1 || !checks.basis_dual_feasible)
            throw Failure(Status::primal_infeasible, "给定或更新基重建后原始不可行，未执行Phase I");
          if (!check_nonbasic_dual(may_start_feasibility)) return;
          if (!stats.dual_entry_checks) stats.dual_entry_checks = stats.last_checks;
          stats.dual_entered = true;
          dual_active = true;
        } else if (dual_active) {
          // 每个对偶换基后均重新检查全部非基，不能用基本r_B合格替代r_N资格。
          if (!checks.basis_dual_feasible)
            throw Failure(Status::numerical_error, "暖基转置解未通过基本约化成本检查");
          check_nonbasic_dual();
        }
        if (!checks.basis_dual_feasible) throw Failure(Status::numerical_error, "暖基转置解未通过基本约化成本检查");
      };
      rebuild(limits.feasibility_first);
      if (limits.feasibility_first && !stats.last_checks->primal_feasible && !dual_active) {
        // 同一基的零目标约化成本为零，可沿用已通过方程重建的基做对偶可行性阶段。
        stats.stage = "feasibility_objective"; work.check();
        stats.feasibility_started = true;
        feasibility_active = true;
        for (auto& cost : costs) { work.tick(); cost = 0; }
        rebuild();
      }
      std::vector<Index> incoming_rows;
      std::vector<double> incoming_values, direction(m);
      std::vector<bool> direction_seen(m, false);
      // 对偶路线才分配行定价空间，关闭时不增加原始路线的扫描或分配。
      std::vector<double> dual_unit, dual_row;
      std::vector<LD> dual_tableau;
      std::vector<double> predicted_basic;
      auto& live = stats.feasibility_update_live;
      bool incremental_since_rebuild = false;
      auto& objective = stats.objective_update_live;
      bool objective_since_rebuild = false;
      bool retrying_objective_price = false;
      std::vector<double> predicted_pi;
      std::vector<LD> predicted_reduced, predicted_products;
      const auto objective_tick = [&] {
        if ((objective.work_items & 4095U) == 0) work.check();
        ++objective.work_items;
      };
      const auto objective_refresh = [&] {
        detail::Timed timer(objective.refresh_seconds);
        ++objective.refresh_started;
        rebuild();
        ++objective.refresh_completed;
        objective_since_rebuild = false;
      };
      const auto objective_fallback = [&] {
        detail::Timed timer(objective.fallback_rebuild_seconds);
        ++objective.fallback;
        ++objective.fallback_rebuild_started;
        rebuild();
        ++objective.fallback_rebuild_completed;
        objective_since_rebuild = false;
      };
      const auto live_tick = [&] {
        if ((live.work_items & 4095U) == 0) work.check();
        ++live.work_items;
      };
      const auto live_rebuild = [&] {
        ++live.fallback_rebuild_started;
        rebuild();
        ++live.fallback_rebuild_completed;
        incremental_since_rebuild = false;
      };
      const auto live_refresh = [&] {
        ++live.refresh_started;
        rebuild();
        ++live.refresh_completed;
        incremental_since_rebuild = false;
      };
      const auto predict_basic = [&](std::size_t in, std::size_t leaving, int sigma, LD step) {
        stats.stage = "incremental_basic_update";
        work.check(); ++live.attempted;
        predicted_basic.resize(m);
        const LD signed_step = static_cast<LD>(sigma) * step;
        for (std::size_t row = 0; row < m; ++row) {
          live_tick();
          const LD predicted = row == leaving ? static_cast<LD>(values[in]) + signed_step :
              static_cast<LD>(values[static_cast<std::size_t>(basis[row])]) - signed_step * direction[row];
          if (!std::isfinite(predicted) || std::abs(predicted) > std::numeric_limits<double>::max()) return false;
          predicted_basic[row] = static_cast<double>(predicted);
          if (predicted != 0 && predicted_basic[row] == 0) return false;
        }
        work.check(); ++live.predicted;
        return true;
      };
      const auto verify_live = [&] {
        stats.stage = "incremental_feasibility_check";
        work.check(); ++live.verification_started;
        const auto allowed = [&](LD a, LD b) {
          return static_cast<LD>(tolerances.feasibility_absolute) +
              static_cast<LD>(tolerances.feasibility_relative) * std::max(std::abs(a), std::abs(b));
        };
        Checks checks;
        bool finite_activity = true;
        for (auto& entry : activity) { live_tick(); entry = 0; }
        for (std::size_t col = 0; col < total; ++col) {
          live_tick(); ++live.verification_columns;
          const auto pos = position[col];
          const LD predicted = pos >= 0 ? predicted_basic[static_cast<std::size_t>(pos)] : nonbasic[col];
          const auto add_entry = [&](std::size_t row, double a) {
            live_tick(); ++live.verification_entries;
            const LD term = static_cast<LD>(a) * predicted;
            activity[row] += term;
            finite_activity = finite_activity && std::isfinite(term) && std::isfinite(activity[row]);
          };
          if (col < n) {
            for (Index entry = p.column_start[col]; entry < p.column_start[col + 1]; ++entry) {
              const auto k = static_cast<std::size_t>(entry);
              add_entry(static_cast<std::size_t>(p.row_index[k]), p.values[k]);
            }
          } else add_entry(col - n, 1);
          for (int side = 0; side < 2; ++side) {
            const double bound = side == 0 ? lower[col] : upper[col];
            if (!std::isfinite(bound)) continue;
            const LD error = std::max<LD>(0, side == 0 ? static_cast<LD>(bound) - predicted : predicted - bound);
            checks.bound_violation = std::max(checks.bound_violation, error);
            checks.bound_ratio = std::max(checks.bound_ratio, detail::ratio(error, allowed(predicted, bound)));
          }
        }
        for (std::size_t row = 0; row < m; ++row) {
          live_tick();
          const LD error = std::abs(activity[row] - p.rhs[row]);
          if (!std::isfinite(error)) { finite_activity = false; continue; }
          checks.equation_residual = std::max(checks.equation_residual, error);
          checks.equation_ratio = std::max(checks.equation_ratio, detail::ratio(error, allowed(activity[row], p.rhs[row])));
        }
        if (!finite_activity) checks.equation_ratio = std::numeric_limits<LD>::infinity();
        work.check(); ++live.verification_completed;
        live.max_equation_ratio = std::max(live.max_equation_ratio, checks.equation_ratio);
        live.max_bound_ratio = std::max(live.max_bound_ratio, checks.bound_ratio);
        if (!(checks.equation_ratio <= 1)) return false;
        // 界违反可延续到后续零目标换基。所有真实非基位置仍取合法的固定值。
        checks.valid = true;
        checks.primal_feasible = checks.bound_ratio <= 1;
        checks.basis_dual_feasible = true;
        checks.nonbasic_dual_checked = checks.nonbasic_dual_feasible = true;
        // 提交期间保持原deadline，超时会丢弃整组局部基与候选。
        for (std::size_t row = 0; row < m; ++row) {
          live_tick(); values[static_cast<std::size_t>(basis[row])] = predicted_basic[row];
        }
        values[static_cast<std::size_t>(*stats.last_leaving)] = nonbasic[static_cast<std::size_t>(*stats.last_leaving)];
        work.check();
        stats.last_checks = checks;
        incremental_since_rebuild = true;
        ++live.accepted;
        return true;
      };
      const auto predict_objective = [&](std::size_t in, std::size_t leaving, int sigma, LD step, double pivot) {
        stats.stage = "incremental_objective_update";
        detail::Timed timer(objective.update_seconds);
        work.check(); ++objective.attempted;
        const LD multiplier_step = reduced[in] / static_cast<LD>(pivot);
        if (!std::isfinite(multiplier_step)) return false;
        predicted_basic.resize(m);
        predicted_pi.resize(m);
        predicted_reduced.resize(total);
        const LD signed_step = static_cast<LD>(sigma) * step;
        const auto representable = [](LD value, double& destination) {
          if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) return false;
          destination = static_cast<double>(value);
          return value == 0 || destination != 0;
        };
        for (std::size_t row = 0; row < m; ++row) {
          objective_tick();
          const LD basic = row == leaving ? static_cast<LD>(values[in]) + signed_step :
              static_cast<LD>(values[static_cast<std::size_t>(basis[row])]) - signed_step * direction[row];
          if (!representable(basic, predicted_basic[row]) ||
              !representable(static_cast<LD>(pi[row]) + multiplier_step * dual_row[row], predicted_pi[row])) return false;
        }
        for (std::size_t col = 0; col < total; ++col) {
          objective_tick();
          if (col == in) predicted_reduced[col] = 0;
          else if (position[col] >= 0) {
            // 旧基本列的表行是单位行，出基位置对应一。
            predicted_reduced[col] = reduced[col] -
                (static_cast<std::size_t>(position[col]) == leaving ? multiplier_step : 0);
          } else if (lower[col] != upper[col]) predicted_reduced[col] = reduced[col] - multiplier_step * dual_tableau[col];
          else continue;
          if (!std::isfinite(predicted_reduced[col])) return false;
        }
        work.check(); ++objective.predicted;
        return true;
      };
      const auto verify_and_commit_objective = [&](std::size_t out) {
        stats.stage = "incremental_objective_check";
        detail::Timed timer(objective.verification_seconds);
        work.check(); ++objective.verification_started;
        predicted_products.resize(total);
        const auto primal_allowed = [&](LD a, LD b) {
          return static_cast<LD>(tolerances.feasibility_absolute) +
              static_cast<LD>(tolerances.feasibility_relative) * std::max(std::abs(a), std::abs(b));
        };
        Checks checks;
        LD reduced_residual_ratio = 0;
        bool finite_primal = true, finite_dual = true;
        for (auto& entry : activity) { objective_tick(); entry = 0; }
        for (std::size_t col = 0; col < total; ++col) {
          objective_tick(); ++objective.verification_columns;
          const auto pos = position[col];
          const LD basic = pos >= 0 ? predicted_basic[static_cast<std::size_t>(pos)] : nonbasic[col];
          LD product = 0;
          const auto add_entry = [&](std::size_t row, double a) {
            objective_tick(); ++objective.verification_entries;
            activity[row] += static_cast<LD>(a) * basic;
            product += static_cast<LD>(a) * predicted_pi[row];
            finite_primal = finite_primal && std::isfinite(activity[row]);
          };
          if (col < n) {
            for (Index entry = p.column_start[col]; entry < p.column_start[col + 1]; ++entry) {
              const auto k = static_cast<std::size_t>(entry);
              add_entry(static_cast<std::size_t>(p.row_index[k]), p.values[k]);
            }
          } else add_entry(col - n, 1);
          const LD true_reduced = static_cast<LD>(costs[col]) - product;
          if (!std::isfinite(true_reduced) || !std::isfinite(product)) finite_dual = false;
          else {
            const LD allowed = static_cast<LD>(optimality_options.dual_absolute) +
                static_cast<LD>(optimality_options.dual_relative) * std::max(std::abs(static_cast<LD>(costs[col])), std::abs(product));
            const bool fixed_nonbasic = pos < 0 && lower[col] == upper[col];
            if (!fixed_nonbasic) {
              // 旧定价对固定非基列没有表行缓存。这些列仅从完整乘积恢复成本。
              const LD predicted = predicted_reduced[col];
              reduced_residual_ratio = std::max(reduced_residual_ratio, detail::ratio(std::abs(predicted - true_reduced), allowed));
              if (pos >= 0) {
                const LD error = std::max(std::abs(true_reduced), std::abs(predicted));
                checks.basic_dual_residual = std::max(checks.basic_dual_residual, error);
                checks.basic_dual_ratio = std::max(checks.basic_dual_ratio, detail::ratio(error, allowed));
              } else {
                const auto error = [&](LD value) {
                  return basic == lower[col] ? std::max<LD>(0, -value) :
                      basic == upper[col] ? std::max<LD>(0, value) : std::abs(value);
                };
                const LD violation = std::max(error(true_reduced), error(predicted));
                checks.nonbasic_dual_violation = std::max(checks.nonbasic_dual_violation, violation);
                checks.nonbasic_dual_ratio = std::max(checks.nonbasic_dual_ratio, detail::ratio(violation, allowed));
              }
            }
          }
          // 定价使用独立重算的约化成本，避免下一步继续累加公式误差。
          predicted_products[col] = product;
          predicted_reduced[col] = true_reduced;
          for (int side = 0; side < 2; ++side) {
            const double bound = side == 0 ? lower[col] : upper[col];
            if (!std::isfinite(bound)) continue;
            const LD error = std::max<LD>(0, side == 0 ? static_cast<LD>(bound) - basic : basic - bound);
            checks.bound_violation = std::max(checks.bound_violation, error);
            checks.bound_ratio = std::max(checks.bound_ratio, detail::ratio(error, primal_allowed(basic, bound)));
          }
        }
        for (std::size_t row = 0; row < m; ++row) {
          objective_tick();
          const LD error = std::abs(activity[row] - p.rhs[row]);
          if (!std::isfinite(error)) { finite_primal = false; continue; }
          checks.equation_residual = std::max(checks.equation_residual, error);
          checks.equation_ratio = std::max(checks.equation_ratio, detail::ratio(error, primal_allowed(activity[row], p.rhs[row])));
        }
        if (!finite_primal) checks.equation_ratio = std::numeric_limits<LD>::infinity();
        if (!finite_dual) reduced_residual_ratio = std::numeric_limits<LD>::infinity();
        work.check(); ++objective.verification_completed;
        objective.max_equation_ratio = std::max(objective.max_equation_ratio, checks.equation_ratio);
        objective.max_bound_ratio = std::max(objective.max_bound_ratio, checks.bound_ratio);
        objective.max_basic_dual_ratio = std::max(objective.max_basic_dual_ratio, checks.basic_dual_ratio);
        objective.max_nonbasic_dual_ratio = std::max(objective.max_nonbasic_dual_ratio, checks.nonbasic_dual_ratio);
        objective.max_reduced_residual_ratio = std::max(objective.max_reduced_residual_ratio, reduced_residual_ratio);
        const bool equation_ok = checks.equation_ratio <= 1;
        const bool basic_ok = checks.basic_dual_ratio <= 1;
        const bool nonbasic_ok = checks.nonbasic_dual_ratio <= 1;
        const bool reduced_ok = reduced_residual_ratio <= 1;
        if (!equation_ok) ++objective.equation_failures;
        if (!basic_ok) ++objective.basic_dual_failures;
        if (!nonbasic_ok) ++objective.nonbasic_dual_failures;
        if (!reduced_ok) ++objective.reduced_failures;
        if (!(equation_ok && basic_ok && nonbasic_ok && reduced_ok)) return false;
        checks.valid = true;
        checks.primal_feasible = checks.bound_ratio <= 1;
        checks.basis_dual_feasible = true;
        checks.nonbasic_dual_checked = checks.nonbasic_dual_feasible = true;
        for (std::size_t row = 0; row < m; ++row) {
          objective_tick(); values[static_cast<std::size_t>(basis[row])] = predicted_basic[row];
        }
        values[out] = nonbasic[out];
        pi.swap(predicted_pi);
        products.swap(predicted_products);
        reduced.swap(predicted_reduced);
        work.check();
        stats.last_checks = checks;
        objective_since_rebuild = true;
        ++objective.accepted;
        return true;
      };
      for (;;) {
        if (feasibility_active && stats.last_checks->primal_feasible) {
          // 用零目标原重建确认阶段终点，若界仍不合格则继续同一阶段。
          if (incremental_since_rebuild) { live_refresh(); continue; }
          feasibility_active = false;
          dual_active = false;
          stats.stage = "restore_objective";
          work.check();
          for (std::size_t col = 0; col < n; ++col) { work.tick(); costs[col] = p.costs[col]; }
          // 只恢复成本并重建原始/转置解，继续持有相同的BASICLU因子和更新次数。
          rebuild(false, true);
          stats.objective_restored = true;
          continue;
        }
        if (dual_active) {
          if (stats.last_checks->primal_feasible) {
            if (objective_since_rebuild) { objective_refresh(); continue; }
            break;
          }
          if (stats.steps_completed >= limits.max_steps)
            throw Failure(Status::step_limit, "暖基原型已用完pivot和翻界步数预算");
          if (updates_since_factor >= m) {
            factorize(true); rebuild();
            incremental_since_rebuild = objective_since_rebuild = false;
            continue;
          }
          stats.stage = "dual_choose_row"; work.check();
          std::optional<Index> leaving_position;
          double leaving_bound = 0;
          LD delta_amount = 0;
          int delta = 0;
          for (std::size_t row = 0; row < m; ++row) {
            work.tick();
            const auto col = static_cast<std::size_t>(basis[row]);
            for (int side = 0; side < 2; ++side) {
              const double bound = side == 0 ? lower[col] : upper[col];
              if (!std::isfinite(bound)) continue;
              const LD violation = side == 0 ? static_cast<LD>(bound) - values[col] : static_cast<LD>(values[col]) - bound;
              const LD allowed = static_cast<LD>(tolerances.feasibility_absolute) +
                  static_cast<LD>(tolerances.feasibility_relative) * std::max(std::abs(static_cast<LD>(values[col])), std::abs(static_cast<LD>(bound)));
              if (violation <= allowed) continue;
              if (violation > delta_amount || (violation == delta_amount &&
                  (!leaving_position || basis[row] < basis[static_cast<std::size_t>(*leaving_position)]))) {
                delta_amount = violation; leaving_position = static_cast<Index>(row);
                leaving_bound = bound; delta = side == 0 ? 1 : -1;
              }
            }
          }
          if (!leaving_position && incremental_since_rebuild) { live_refresh(); continue; }
          if (!leaving_position && objective_since_rebuild) { objective_refresh(); continue; }
          if (!leaving_position)
            throw Failure(Status::numerical_error, "暖基界检查失败但未找到超容差的基本离基量");
          if (!retrying_objective_price) { ++stats.steps_started; ++stats.dual_steps_started; }
          retrying_objective_price = false;
          stats.last_entering.reset(); stats.last_leaving = basis[static_cast<std::size_t>(*leaving_position)];
          stats.last_theta.reset(); stats.last_pivot.reset();
          stats.last_dual_theta.reset(); stats.last_dual_primal_step.reset();
          stats.last_dual_relaxed_theta.reset(); stats.last_dual_selected_alpha.reset();
          work.check(); dual_unit.assign(m, 0); dual_row.resize(m); work.check();
          dual_unit[static_cast<std::size_t>(*leaving_position)] = 1;
          require_ok(native(stats.dual_row_solve, "dual_row_solve", [&] {
            return basiclu_obj_solve_dense(&lu, dual_unit.data(), dual_row.data(), 'T');
          }));
          for (double value : dual_row) {
            work.tick();
            if (!std::isfinite(value)) throw Failure(Status::numerical_error, "暖基对偶行定价的BTRAN非有限");
          }
          stats.stage = "dual_pricing";
          std::optional<Index> entering;
          LD theta_dual = std::numeric_limits<LD>::infinity();
          int entering_sigma = 0;
          {
            detail::Timed timer(stats.dual_pricing_seconds);
            work.check(); dual_tableau.resize(total);
            stats.dual_tableau_entries_peak = std::max(stats.dual_tableau_entries_peak, static_cast<std::uint64_t>(dual_tableau.size()));
            work.check();
            // Harris首遍以当前pi对应的原dual容差给出可选区间；只计算一次表行。
            stats.stage = "dual_ratio_relaxed";
            ++stats.dual_ratio_passes_started;
            LD relaxed_theta = std::numeric_limits<LD>::infinity();
            for (std::size_t col = 0; col < total; ++col) {
              work.tick();
              if (position[col] >= 0 || lower[col] == upper[col]) continue;
              LD row_value = 0;
              column(col, [&](std::size_t row, double a) { detail::add(row_value, static_cast<LD>(a) * dual_row[row]); });
              dual_tableau[col] = row_value;
              const LD allowed = static_cast<LD>(optimality_options.dual_absolute) +
                  static_cast<LD>(optimality_options.dual_relative) * std::max(std::abs(static_cast<LD>(costs[col])), std::abs(products[col]));
              for (int sigma : {1, -1}) {
                if (sigma > 0 ? values[col] >= upper[col] : values[col] <= lower[col]) continue;
                const LD alpha = -static_cast<LD>(delta) * sigma * row_value;
                if (!(alpha > 0)) continue;
                const LD relaxed = std::max<LD>(0, static_cast<LD>(sigma) * reduced[col] + allowed) / alpha;
                relaxed_theta = std::min(relaxed_theta, relaxed);
              }
            }
            stats.last_dual_relaxed_theta = relaxed_theta;
            ++stats.dual_ratio_passes; work.check();
            // 第二遍只读缓存；区间内选最大alpha=|表行系数|，精确相等按全局列序。
            // 不截掉tiny但必要的主元；新pi下的相对容差仍须由完整重建检查。
            stats.stage = "dual_ratio_select";
            ++stats.dual_ratio_passes_started;
            LD selected_alpha = 0;
            for (std::size_t col = 0; col < total; ++col) {
              work.tick();
              if (position[col] >= 0 || lower[col] == upper[col]) continue;
              for (int sigma : {1, -1}) {
                if (sigma > 0 ? values[col] >= upper[col] : values[col] <= lower[col]) continue;
                const LD alpha = -static_cast<LD>(delta) * sigma * dual_tableau[col];
                if (!(alpha > 0)) continue;
                // tol内负约化成本仅在比值中视零，不移动成本或修改r。
                const LD possible = std::max<LD>(0, static_cast<LD>(sigma) * reduced[col]) / alpha;
                if (!std::isfinite(possible) || possible > relaxed_theta) continue;
                ++stats.dual_band_candidates;
                if (alpha > selected_alpha || (alpha == selected_alpha &&
                    (!entering || static_cast<Index>(col) < *entering))) {
                  selected_alpha = alpha; theta_dual = possible;
                  entering = static_cast<Index>(col); entering_sigma = sigma;
                }
              }
            }
            if (entering) stats.last_dual_selected_alpha = selected_alpha;
            ++stats.dual_ratio_passes; ++stats.dual_pricing_passes; work.check();
          }
          if (!entering && objective_since_rebuild) {
            objective_refresh();
            ++objective.repricing_retries;
            retrying_objective_price = true;
            continue;
          }
          if (!entering)
            throw Failure(Status::no_finite_step, "暖基对偶行没有有限合法入基比值，未发布不可行证书");
          const auto in = static_cast<std::size_t>(*entering);
          const Index out_position = *leaving_position;
          const auto out = static_cast<std::size_t>(basis[static_cast<std::size_t>(out_position)]);
          stats.last_entering = *entering; stats.last_dual_theta = theta_dual;
          incoming_rows.clear(); incoming_values.clear();
          column(in, [&](std::size_t row, double a) { incoming_rows.push_back(static_cast<Index>(row)); incoming_values.push_back(a); });
          const Index dummy_index = 0;
          const double dummy_value = 0;
          require_ok(native(stats.ftran, "ftran", [&] {
            return basiclu_obj_solve_for_update(&lu, static_cast<Index>(incoming_rows.size()),
                incoming_rows.empty() ? &dummy_index : incoming_rows.data(),
                incoming_values.empty() ? &dummy_value : incoming_values.data(), 'N', 1);
          }));
          stats.stage = "dual_pivot";
          for (std::size_t row = 0; row < m; ++row) { work.tick(); direction[row] = 0; direction_seen[row] = false; }
          if (lu.nzlhs < 0 || lu.nzlhs > static_cast<Index>(m))
            throw Failure(Status::numerical_error, "暖基FTRAN稀疏模式长度无效");
          for (Index k = 0; k < lu.nzlhs; ++k) {
            work.tick(); const Index row = lu.ilhs[k];
            if (row < 0 || row >= static_cast<Index>(m) || direction_seen[static_cast<std::size_t>(row)] || !std::isfinite(lu.lhs[row]))
              throw Failure(Status::numerical_error, "暖基FTRAN稀疏模式或数值无效");
            direction_seen[static_cast<std::size_t>(row)] = true; direction[static_cast<std::size_t>(row)] = lu.lhs[row];
          }
          const double pivot = direction[static_cast<std::size_t>(out_position)];
          const LD actual_alpha = -static_cast<LD>(delta) * entering_sigma * pivot;
          if (!(actual_alpha > 0))
            throw Failure(Status::numerical_error, "暖基对偶入基列的实际主元为零或方向与行定价不一致");
          const LD primal_step = delta_amount / actual_alpha;
          if (!std::isfinite(primal_step))
            throw Failure(Status::numerical_error, "暖基对偶换基的原始移动量非有限");
          stats.last_pivot = pivot;
          stats.last_theta = primal_step; stats.last_dual_primal_step = primal_step;
          const bool live_pivot = live.enabled && feasibility_active && live_predicate;
          const bool objective_pivot = objective.enabled && !feasibility_active;
          const bool predicted = live_pivot &&
              predict_basic(in, static_cast<std::size_t>(out_position), entering_sigma, primal_step);
          const bool objective_predicted = objective_pivot &&
              predict_objective(in, static_cast<std::size_t>(out_position), entering_sigma, primal_step, pivot);
          if (objective_pivot && !objective_predicted) ++objective.prediction_failures;
          require_ok(native(stats.btran, "btran", [&] { return basiclu_obj_solve_for_update(&lu, 1, &out_position, nullptr, 'T', 0); }));
          require_ok(native(stats.update, "update", [&] { return basiclu_obj_update(&lu, pivot); }));
          // 不做BFRT：入基量可越过自身另一界，下一轮对偶换基继续修复。
          basis[static_cast<std::size_t>(out_position)] = static_cast<Index>(in);
          position[in] = out_position; position[out] = -1; nonbasic[out] = leaving_bound;
          ++stats.pivots; ++stats.dual_pivots; ++updates_since_factor;
          if (primal_step == 0) ++stats.zero_steps;
          if (theta_dual == 0) ++stats.dual_zero_steps;
          if (live_pivot) {
            if (!predicted || !verify_live()) { ++live.fallback; live_rebuild(); }
          } else if (objective_pivot) {
            if (!objective_predicted || !verify_and_commit_objective(out)) objective_fallback();
          } else rebuild();
          ++stats.steps_completed; ++stats.dual_steps_completed;
          if (feasibility_active) ++stats.feasibility_steps;
          work.check();
          continue;
        }
        stats.stage = "pricing";
        work.check();
        std::optional<Index> entering;
        int sigma = 0;
        LD best = 0;
        {
          detail::Timed timer(stats.pricing_seconds);
          for (std::size_t col = 0; col < total; ++col) {
            work.tick();
            if (position[col] >= 0 || lower[col] == upper[col]) continue;
            const LD allowed = static_cast<LD>(optimality_options.dual_absolute) +
                static_cast<LD>(optimality_options.dual_relative) * std::max(std::abs(static_cast<LD>(costs[col])), std::abs(products[col]));
            const int move = reduced[col] < -allowed && values[col] < upper[col] ? 1 :
                reduced[col] > allowed && values[col] > lower[col] ? -1 : 0;
            if (move != 0 && std::abs(reduced[col]) > best) {
              best = std::abs(reduced[col]); entering = static_cast<Index>(col); sigma = move;
            }
          }
          ++stats.pricing_passes; work.check();
        }
        if (!entering) break;
        if (stats.steps_completed >= limits.max_steps)
          throw Failure(Status::step_limit, "暖基原型已用完pivot和翻界步数预算");
        if (updates_since_factor >= m) { factorize(true); rebuild(); continue; }
        ++stats.steps_started;
        stats.last_entering = *entering; stats.last_leaving.reset(); stats.last_theta.reset(); stats.last_pivot.reset();
        const auto in = static_cast<std::size_t>(*entering);
        incoming_rows.clear(); incoming_values.clear();
        column(in, [&](std::size_t row, double value) { incoming_rows.push_back(static_cast<Index>(row)); incoming_values.push_back(value); });
        const Index dummy_index = 0;
        const double dummy_value = 0;
        require_ok(native(stats.ftran, "ftran", [&] {
          return basiclu_obj_solve_for_update(&lu, static_cast<Index>(incoming_rows.size()),
              incoming_rows.empty() ? &dummy_index : incoming_rows.data(),
              incoming_values.empty() ? &dummy_value : incoming_values.data(), 'N', 1);
        }));
        stats.stage = "ratio";
        for (std::size_t row = 0; row < m; ++row) {
          work.tick(); direction[row] = 0; direction_seen[row] = false;
        }
        if (lu.nzlhs < 0 || lu.nzlhs > static_cast<Index>(m))
          throw Failure(Status::numerical_error, "暖基FTRAN稀疏模式长度无效");
        for (Index k = 0; k < lu.nzlhs; ++k) {
          work.tick();
          const Index row = lu.ilhs[k];
          if (row < 0 || row >= static_cast<Index>(m) || direction_seen[static_cast<std::size_t>(row)] ||
              !std::isfinite(lu.lhs[row]))
            throw Failure(Status::numerical_error, "暖基FTRAN稀疏模式或数值无效");
          direction_seen[static_cast<std::size_t>(row)] = true;
          direction[static_cast<std::size_t>(row)] = lu.lhs[row];
        }
        LD theta = std::numeric_limits<LD>::infinity();
        std::optional<Index> leaving_position;
        double leaving_bound = 0;
        for (std::size_t row = 0; row < m; ++row) {
          work.tick();
          const auto basic = static_cast<std::size_t>(basis[row]);
          const LD alpha = static_cast<LD>(sigma) * direction[row];
          const double bound = alpha > 0 ? lower[basic] : upper[basic];
          if (alpha == 0 || !std::isfinite(bound)) continue;
          const LD slack = alpha > 0 ? static_cast<LD>(values[basic]) - bound : static_cast<LD>(bound) - values[basic];
          const LD possible = std::max<LD>(0, slack) / std::abs(alpha);
          if (possible < theta || (possible == theta && (!leaving_position || basis[row] < basis[static_cast<std::size_t>(*leaving_position)]))) {
            theta = possible; leaving_position = static_cast<Index>(row); leaving_bound = bound;
          }
        }
        const double own_bound = sigma > 0 ? upper[in] : lower[in];
        const LD own_theta = std::isfinite(own_bound) ? std::max<LD>(0, sigma > 0 ? static_cast<LD>(own_bound) - values[in] : static_cast<LD>(values[in]) - own_bound) : std::numeric_limits<LD>::infinity();
        const bool flip = own_theta < theta;
        if (flip) theta = own_theta;
        if (!std::isfinite(theta) || (!flip && !leaving_position))
          throw Failure(Status::no_finite_step, "暖基方向没有有限阻挡，未发布无界证书");
        stats.last_theta = theta;
        work.check();
        if (flip) {
          nonbasic[in] = own_bound; ++stats.bound_flips;
        } else {
          const Index out_position = *leaving_position;
          const auto out = static_cast<std::size_t>(basis[static_cast<std::size_t>(out_position)]);
          const double pivot = direction[static_cast<std::size_t>(out_position)];
          stats.last_leaving = static_cast<Index>(out); stats.last_pivot = pivot;
          require_ok(native(stats.btran, "btran", [&] { return basiclu_obj_solve_for_update(&lu, 1, &out_position, nullptr, 'T', 0); }));
          require_ok(native(stats.update, "update", [&] { return basiclu_obj_update(&lu, pivot); }));
          // 更新已成功，先记录新基；后续若失败只弃整次，不能声称回滚到旧因子。
          basis[static_cast<std::size_t>(out_position)] = static_cast<Index>(in);
          position[in] = out_position; position[out] = -1; nonbasic[out] = leaving_bound;
          ++stats.pivots; ++updates_since_factor;
        }
        if (theta == 0) ++stats.zero_steps;
        rebuild();
        ++stats.steps_completed; work.check();
      }
      stats.stage = "verification"; work.check();
      std::vector<double> candidate_x(p.original_columns), candidate_y(p.original_rows, 0);
      for (std::size_t col = 0; col < candidate_x.size(); ++col) { work.tick(); candidate_x[col] = values[col]; }
      for (std::size_t row = 0; row < m; ++row) { work.tick(); candidate_y[p.original_row[row]] = pi[row]; }
      cross::Result assessment;
      detail::AssessmentCollect collect{result, assessment};
      assessment.stats.projection.enabled = true;
      const auto check_budget = [&] { work.check(); };
      cross::verify_candidate_once(input.model, candidate_x, candidate_y, assessment, tolerances, optimality_options, check_budget);
      stats.stage = "projection";
      if (auto projected = cross::project_rejected_row_dual(input.model, candidate_x, candidate_y, assessment,
          tolerances, optimality_options, check_budget)) candidate_y = std::move(*projected);
      work.check();
      if (assessment.verification && detail::quality_ok(*assessment.verification)) {
        result.x = std::move(candidate_x); result.y = std::move(candidate_y);
        result.status = Status::accepted; result.message = "自主暖基候选通过原模型完整验收";
      } else { result.status = Status::rejected; result.message = "自主暖基候选未通过原模型完整验收"; }
    }
    // BASICLU对象及全部局部候选已析构，之后仍核同一绝对期限和生产预算。
    result.stats.stage = "cleanup"; work.check(); result.stats.stage = "finished";
  } catch (const Failure& failure) {
    result.status = failure.status; result.message = failure.what();
  } catch (const cross::Failure& failure) {
    result.status = failure.status == cross::Status::time_limit ? Status::time_limit :
        failure.status == cross::Status::invalid_input ? Status::invalid_input : Status::numerical_error;
    result.message = failure.what();
  } catch (const std::exception& failure) {
    result.status = Status::exception; result.message = failure.what();
  } catch (...) {
    result.status = Status::exception; result.message = "暖基原型出现未知异常";
  }
  auto finish = Clock::now();
  if (finish >= absolute_deadline) {
    result.status = Status::time_limit; result.message = "暖基原型含清理总成本达到绝对截止时点";
  }
  if (result.status != Status::accepted) {
    std::vector<double>().swap(result.x); std::vector<double>().swap(result.y); finish = Clock::now();
  }
  result.stats.total_seconds = std::chrono::duration<double>(finish - start).count();
  return result;
}

inline Result run(const Input& input, Clock::time_point absolute_deadline, const Limits& limits = {},
    const solver::Tolerances& tolerances = {}, const solver::LpOptimalityOptions& optimality_options = {}) {
  return run(input, absolute_deadline, limits, tolerances, optimality_options, [] {});
}
}  // 命名空间solver::detail::warm_cleanup
