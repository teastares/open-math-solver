#pragma once

// 自主有界原始单纯形Phase II原型；数值分解和列更新复用BASICLU，并非自研LU。
// BASICLU归属ERGO-Code，按锁定依赖许可证使用。
// 锁定来源为HiGHS 1.15.1的highs/ipm/basiclu；保留原依赖的许可证和归属。
#include "solver/experimental/ipx_crossover.hpp"
#include "ipm/basiclu/basiclu.h"
#include <type_traits>

namespace solver_v119_warm_cleanup {
namespace cross = solver::experimental::ipx_crossover;
using Clock = cross::Clock;
using Index = ipx::Int;
using LD = long double;
static_assert(std::is_same_v<Index, lu_int>);

enum class Status {
  accepted, rejected, invalid_input, unsupported_empty, singular_basis,
  primal_infeasible, numerical_error, step_limit, no_finite_step, time_limit, exception
};
inline const char* status_name(Status value) noexcept {
  switch (value) {
    case Status::accepted: return "accepted";
    case Status::rejected: return "rejected";
    case Status::invalid_input: return "invalid_input";
    case Status::unsupported_empty: return "unsupported_empty";
    case Status::singular_basis: return "singular_basis";
    case Status::primal_infeasible: return "primal_infeasible";
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
struct Limits { std::uint64_t max_steps = 64; };
struct CallStats {
  std::uint64_t started = 0, completed = 0;
  std::optional<Index> rc;
  double seconds = 0;
};
struct Checks {
  bool valid = false, primal_feasible = false, basis_dual_feasible = false;
  LD equation_residual = 0, bound_violation = 0, basic_dual_residual = 0;
  LD equation_ratio = 0, bound_ratio = 0, basic_dual_ratio = 0;
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
  // rebuild_seconds包含其中的两次solve子项；不得与子项相加当作总成本。
  double pricing_seconds = 0, rebuild_seconds = 0, verification_seconds = 0, total_seconds = 0;
  const char* stage = "entry";
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
      const auto rebuild = [&] {
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
        for (std::size_t row = 0; row < m; ++row) {
          work.tick();
          if (!std::isfinite(basic_values[row]) || !std::isfinite(pi[row]))
            throw Failure(Status::numerical_error, "暖基重建点或转置解非有限");
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
        if (!checks.primal_feasible) throw Failure(Status::primal_infeasible, "给定或更新基重建后原始不可行，未执行Phase I");
        if (!checks.basis_dual_feasible) throw Failure(Status::numerical_error, "暖基转置解未通过基本约化成本检查");
      };
      rebuild();
      std::vector<Index> incoming_rows;
      std::vector<double> incoming_values, direction(m);
      std::vector<bool> direction_seen(m, false);
      for (;;) {
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
}  // 命名空间solver_v119_warm_cleanup
