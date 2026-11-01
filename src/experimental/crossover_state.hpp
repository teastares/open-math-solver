#pragma once

#include "diagnostic_state.hpp"
#include "solver/backend.hpp"
#include "solver/experimental/ipx_crossover.hpp"
#include "solver/experimental/warm_cleanup.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string_view>
#include <type_traits>

namespace solver_v117_crossover_state {
using Clock = std::chrono::steady_clock;
using Index = std::int64_t;
namespace cross = solver::experimental::ipx_crossover;
namespace warm = solver_v119_warm_cleanup;
inline constexpr long double gate_threshold = 1e-4L;
inline thread_local std::optional<bool> enabled_override;

inline bool requested() noexcept {
  if (enabled_override) return *enabled_override;
  const char* value = std::getenv("SOLVER_V117_CROSSOVER");
  return value && std::string_view(value) == "1";
}

struct Quality {
  bool primal_valid = false, primal_feasible = false, objective_consistent = false;
  bool dual_valid = false, dual_feasible = false, gap_satisfied = false, numerically_optimal = false;
  std::optional<double> objective, max_row_violation, max_bound_violation, max_dual_residual;
  std::optional<double> objective_gap, relative_objective_gap;
};
inline std::optional<double> summary(double value) noexcept {
  return std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
}
inline std::optional<double> summary(const std::optional<double>& value) noexcept {
  return value ? summary(*value) : std::nullopt;
}
inline Quality quality(const solver::LpVerification& q) noexcept {
  return {q.primal.valid_input, q.primal.feasible, q.primal.objective_consistent,
      q.optimality.valid_input, q.optimality.dual_feasible, q.optimality.gap_satisfied,
      q.optimality.numerically_optimal, summary(q.primal.objective), summary(q.primal.max_row_violation),
      summary(q.primal.max_bound_violation), summary(q.optimality.max_dual_residual),
      summary(q.optimality.objective_gap), summary(q.optimality.relative_objective_gap)};
}
struct WarmEvent {
  std::uint64_t started = 0, completed = 0, max_steps = 64;
  warm::Status status = warm::Status::exception;
  std::optional<Quality> raw_quality, after_quality;
  warm::Stats stats;
};
struct Event {
  Index iteration = 0;
  long double bounds_inf = 0, cost_inf = 0, primal_ratio = 0, dual_ratio = 0, relative_gap = 0;
  bool deadline_saturated = false;
  cross::Status status = cross::Status::exception;
  cross::Status crossover_status = cross::Status::exception;
  std::string message;
  Quality before_quality;
  std::optional<Quality> unprojected_quality;
  std::optional<Quality> after_quality;
  std::optional<Quality> crossover_quality;
  cross::BasisExportStats basis_export;
  std::optional<WarmEvent> warm;
  cross::Stats stats;
  // 两阶段checker成本求和；scope另含桥接及导出映射释放，子项不能重复相加。
  double verification_seconds_total = 0, scope_seconds = 0;
};
struct Log {
  bool enabled = false, attempted = false, local_accepted = false, adopted = false, norms_ready = false;
  std::uint64_t started = 0, completed = 0, norm_items = 0, gate_checks = 0;
  long double bounds_inf = 0, cost_inf = 0;
  const solver::Model* norm_model = nullptr;
  const char* oracle_decision = "not_consulted";
  double gate_seconds = 0, norm_seconds = 0;
  std::optional<Index> last_gate_iteration;
  std::optional<std::array<long double, 3>> last_gate_measures;
  std::optional<Event> last;
};
inline thread_local Log* context = nullptr;
inline bool enabled() noexcept { return context && context->enabled; }
inline Log* current() noexcept { return enabled() ? context : nullptr; }
struct Scope {
  explicit Scope(Log* next) noexcept : previous(context) { context = next; }
  ~Scope() noexcept { context = previous; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Log* previous;
};

struct Deadline { Clock::time_point value; bool saturated = false; };
// 在浮点域先判断可表示范围，不对过大的double直接做浮点到整数duration转换。
inline std::optional<Deadline> make_deadline(Clock::time_point start, double limit) noexcept {
  if (!std::isfinite(limit) || limit < 0) return std::nullopt;
  using Duration = Clock::duration;
  using Rep = Duration::rep;
  using Period = Duration::period;
  static_assert(std::is_integral_v<Rep> && std::numeric_limits<long double>::digits >= std::numeric_limits<Rep>::digits);
  const long double first = static_cast<long double>(start.time_since_epoch().count());
  const long double maximum = static_cast<long double>(Clock::time_point::max().time_since_epoch().count());
  const long double minimum = static_cast<long double>(Clock::time_point::min().time_since_epoch().count());
  const long double ticks = static_cast<long double>(limit) * Period::den / Period::num;
  if (!std::isfinite(ticks) || ticks >= maximum - first)
    return Deadline{Clock::time_point::max(), true};
  const long double last = first + std::trunc(ticks);
  if (last >= maximum) return Deadline{Clock::time_point::max(), true};
  if (!std::isfinite(last) || last < minimum) return std::nullopt;
  return Deadline{Clock::time_point(Duration(static_cast<Rep>(last))), false};
}

namespace detail {
struct Timed {
  double& seconds;
  Clock::time_point start = Clock::now();
  explicit Timed(double& destination) noexcept : seconds(destination) {}
  ~Timed() noexcept { seconds += std::chrono::duration<double>(Clock::now() - start).count(); }
};
}  // 命名空间 detail

// 只消费已经被原验拒绝的导出基。预算异常先留真实内核前缀、释放快照，再原样重抛。
template<class Budget>
void try_warm_cleanup(const solver::Model& model, cross::Result& result, Event& event,
    Clock::time_point absolute_deadline, const solver::SolveOptions& options, Budget&& budget) {
  if (result.status != cross::Status::rejected || !result.basis_export || event.warm) return;
  auto& cleanup = event.warm.emplace();
  std::exception_ptr budget_failure;
  const auto outer_budget = [&] {
    try { budget(); }
    catch (...) { budget_failure = std::current_exception(); throw; }
  };
  try {
    {
      auto basis = std::move(*result.basis_export);
      result.basis_export.reset();
      outer_budget();
      ++cleanup.started;
      auto trial = warm::run(warm::Input{model, basis.prepared, basis.variable_status,
          basis.row_status, basis.structural_seed}, absolute_deadline, warm::Limits{cleanup.max_steps},
          options.tolerances, options.pdlp.optimality, [&] {
            try { outer_budget(); }
            catch (...) { throw warm::Failure(warm::Status::time_limit, "暖基共享预算回调已中断"); }
          });
      ++cleanup.completed;
      cleanup.status = trial.status; cleanup.stats = trial.stats;
      if (trial.raw_verification) cleanup.raw_quality = quality(*trial.raw_verification);
      else if (trial.verification) cleanup.raw_quality = quality(*trial.verification);
      if (trial.verification) cleanup.after_quality = quality(*trial.verification);
      else cleanup.after_quality = cleanup.raw_quality;
      event.verification_seconds_total = event.stats.verification_seconds + trial.stats.verification_seconds;
      if (trial.status == warm::Status::accepted) {
        result.x = std::move(trial.x); result.y = std::move(trial.y);
        result.verification = std::move(trial.verification);
        result.status = cross::Status::accepted;
        result.message = "自主暖基候选通过原模型完整验收";
        event.after_quality = cleanup.after_quality;
      } else if (trial.status == warm::Status::time_limit) {
        result.status = cross::Status::time_limit;
        result.message = trial.message;
      }
    }
    // Prepared快照已经释放，清理时间仍属于原求解预算。
    if (budget_failure) std::rethrow_exception(budget_failure);
    outer_budget();
    if (Clock::now() >= absolute_deadline) {
      result.status = cross::Status::time_limit;
      result.message = "暖基桥接含清理总成本已达到共享截止时点";
      std::vector<double>().swap(result.x); std::vector<double>().swap(result.y);
    }
    event.status = result.status; event.message = result.message;
  } catch (...) {
    result.basis_export.reset();
    std::vector<double>().swap(result.x); std::vector<double>().swap(result.y);
    result.status = budget_failure ? cross::Status::time_limit : cross::Status::exception;
    if (!cleanup.completed) cleanup.status = budget_failure ? warm::Status::time_limit : warm::Status::exception;
    event.status = result.status;
    throw;
  }
}

// budget是生产原预算回调；它的异常类型原样传播，不另设时钟或测试专用注入入口。
template<class Budget>
std::optional<cross::Result> try_current(const solver::Model& model, std::span<const double> x,
    std::span<const double> y, const solver::LpVerification& checks, const solver::SolveOptions& options,
    Clock::time_point solve_start, std::optional<Index> iteration, Budget&& budget) {
  Log* const log = current();
  if (!log || log->attempted || options.ipm.method != "homogeneous" || !iteration || *iteration < 1 ||
      checks.optimality.numerically_optimal) return std::nullopt;
  ++log->gate_checks;
  log->last_gate_iteration = iteration;
  log->last_gate_measures.reset();
  const auto& p = checks.primal;
  const auto& d = checks.optimality;
  if (!p.valid_input || !p.objective_consistent || !p.objective || !std::isfinite(*p.objective) ||
      !d.valid_input || !d.dual_feasible || !d.relative_objective_gap || !std::isfinite(*d.relative_objective_gap) ||
      !std::isfinite(p.max_row_violation) || p.max_row_violation < 0 ||
      !std::isfinite(p.max_bound_violation) || p.max_bound_violation < 0 ||
      !std::isfinite(d.max_dual_residual) || d.max_dual_residual < 0) return std::nullopt;
  const auto end = make_deadline(solve_start, options.time_limit);
  if (!end) return std::nullopt;
  if (model.column_lower.size() != model.costs.size() || model.column_upper.size() != model.costs.size() ||
      model.row_lower.size() != model.row_upper.size()) return std::nullopt;
  // 一个生产solve只有一个child模型；复用其它模型的norm不会触发新尝试。
  if (log->norms_ready && log->norm_model != &model) return std::nullopt;
  std::array<long double, 3> measures{};
  {
    detail::Timed gate_timer(log->gate_seconds);
    budget();
    if (!log->norms_ready) {
      detail::Timed norm_timer(log->norm_seconds);
      long double bounds_inf = 0, cost_inf = 0;
      std::size_t item = 0;
      const auto tick = [&] {
        if ((item & 4095U) == 0) budget();
        ++item; ++log->norm_items;
      };
      const auto bound = [&](double value) {
        if (std::isfinite(value)) bounds_inf = std::max(bounds_inf, std::abs(static_cast<long double>(value)));
      };
      for (std::size_t col = 0; col < model.costs.size(); ++col) {
        tick();
        if (!std::isfinite(model.costs[col])) return std::nullopt;
        cost_inf = std::max(cost_inf, std::abs(static_cast<long double>(model.costs[col])));
        bound(model.column_lower[col]); bound(model.column_upper[col]);
      }
      for (std::size_t row = 0; row < model.row_lower.size(); ++row) {
        tick(); bound(model.row_lower[row]); bound(model.row_upper[row]);
      }
      budget();
      log->bounds_inf = bounds_inf; log->cost_inf = cost_inf;
      log->norm_model = &model; log->norms_ready = true;
    }
    measures = {std::max(static_cast<long double>(p.max_row_violation), static_cast<long double>(p.max_bound_violation)) /
        (1 + log->bounds_inf), static_cast<long double>(d.max_dual_residual) / (1 + log->cost_inf),
        std::abs(static_cast<long double>(*d.relative_objective_gap))};
    log->last_gate_measures = measures;
    budget();
  }
  for (long double value : measures)
    if (!std::isfinite(value) || value > gate_threshold) return std::nullopt;
  budget();
  auto& event = log->last.emplace();
  event.iteration = *iteration; event.bounds_inf = log->bounds_inf; event.cost_inf = log->cost_inf;
  event.primal_ratio = measures[0]; event.dual_ratio = measures[1]; event.relative_gap = measures[2];
  event.deadline_saturated = end->saturated; event.before_quality = quality(checks);
  log->attempted = true; ++log->started;
  detail::Timed scope_timer(event.scope_seconds);
  auto result = cross::run(model, x, y, end->value, options.tolerances, options.pdlp.optimality, true, true);
  // 返回后先保实际adapter账本；末预算失败可以完成本地交叉，却仍未交付/采用。
  ++log->completed;
  event.stats = result.stats; event.status = result.status; event.crossover_status = result.status;
  event.basis_export = result.basis_stats;
  event.verification_seconds_total = result.stats.verification_seconds;
  if (result.raw_verification) event.unprojected_quality = quality(*result.raw_verification);
  else if (result.verification) event.unprojected_quality = quality(*result.verification);
  if (result.verification) event.after_quality = quality(*result.verification);
  else event.after_quality = event.unprojected_quality;
  event.crossover_quality = event.after_quality;
  event.message = result.message;
  try_warm_cleanup(model, result, event, end->value, options, budget);
  log->local_accepted = result.status == cross::Status::accepted;
  budget();
  return result;
}

// decision由真实核心调用点给出；归一为固定字面量，避免保存临时字符串指针。
inline void note_oracle(const char* decision, bool adopted) noexcept {
  auto* log = current();
  if (!log || !log->attempted) return;
  const std::string_view value = decision ? decision : "error";
  log->oracle_decision = value == "accept" ? "accept" : value == "reject" ? "reject" :
      value == "budget" ? "budget" : value == "not_active" ? "not_active" : "error";
  log->adopted = adopted;
}

inline void quote(FILE* stream, const char* value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  std::fputc('"', stream);
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
    if (*p == '"' || *p == '\\') { std::fputc('\\', stream); std::fputc(*p, stream); }
    else if (*p < 32) std::fprintf(stream, "\\u%04x", static_cast<unsigned>(*p));
    else std::fputc(*p, stream);
  }
  std::fputc('"', stream);
}
inline void number(FILE* stream, double value) noexcept {
  if (std::isfinite(value)) std::fprintf(stream, "%.17g", value); else std::fputs("null", stream);
}
inline void number(FILE* stream, long double value) noexcept {
  // 公开JSON按double读取；溢出记null，下溢摘要0不代表内部LD残差精确为0。
  const double narrowed = static_cast<double>(value);
  if (std::isfinite(value) && std::isfinite(narrowed)) std::fprintf(stream, "%.17g", narrowed);
  else std::fputs("null", stream);
}
template<class T>
inline void optional(FILE* stream, const std::optional<T>& value) noexcept {
  if (!value) std::fputs("null", stream);
  else if constexpr (std::is_floating_point_v<T>) number(stream, *value);
  else std::fprintf(stream, "%lld", static_cast<long long>(*value));
}
inline void write_quality(FILE* stream, const Quality& q) noexcept {
  std::fprintf(stream, "{\"primal_valid\":%s,\"primal_feasible\":%s,\"objective_consistent\":%s,\"dual_valid\":%s,"
      "\"dual_feasible\":%s,\"gap_satisfied\":%s,\"numerically_optimal\":%s,\"objective\":",
      q.primal_valid ? "true" : "false", q.primal_feasible ? "true" : "false", q.objective_consistent ? "true" : "false",
      q.dual_valid ? "true" : "false", q.dual_feasible ? "true" : "false", q.gap_satisfied ? "true" : "false",
      q.numerically_optimal ? "true" : "false");
  optional(stream, q.objective); std::fputs(",\"max_row_violation\":", stream); optional(stream, q.max_row_violation);
  std::fputs(",\"max_bound_violation\":", stream); optional(stream, q.max_bound_violation);
  std::fputs(",\"max_dual_residual\":", stream); optional(stream, q.max_dual_residual);
  std::fputs(",\"objective_gap\":", stream); optional(stream, q.objective_gap);
  std::fputs(",\"relative_objective_gap\":", stream); optional(stream, q.relative_objective_gap); std::fputc('}', stream);
}
inline void write_residuals(FILE* stream, const cross::Residuals& r) noexcept {
  std::fputs("{\"primal\":", stream); number(stream, r.primal); std::fputs(",\"dual\":", stream); number(stream, r.dual);
  std::fputs(",\"bounds\":", stream); number(stream, r.bounds); std::fputs(",\"complementarity\":", stream); number(stream, r.complementarity);
  std::fputs(",\"dual_sign\":", stream); number(stream, r.dual_sign); std::fputc('}', stream);
}
inline void write_preparation(FILE* stream, const cross::PreparationStats& p) noexcept {
  std::fprintf(stream, "{\"duplicates_merged\":%llu,\"rounded_aggregates\":%llu,\"removed_free_rows\":%llu,"
      "\"free_row_duals_zeroed\":%llu,\"primal_changes\":%llu,\"dual_changes\":%llu,\"max_aggregation_rounding\":",
      static_cast<unsigned long long>(p.duplicates_merged), static_cast<unsigned long long>(p.rounded_aggregates),
      static_cast<unsigned long long>(p.removed_free_rows), static_cast<unsigned long long>(p.free_row_duals_zeroed),
      static_cast<unsigned long long>(p.primal_changes), static_cast<unsigned long long>(p.dual_changes));
  number(stream, p.max_aggregation_rounding); std::fputs(",\"raw\":", stream); write_residuals(stream, p.raw);
  std::fputs(",\"dropped\":", stream); write_residuals(stream, p.dropped); std::fputc('}', stream);
}
inline void write_info(FILE* stream, const ipx::Info& i) noexcept {
  std::fprintf(stream, "{\"status\":%lld", static_cast<long long>(i.status));
#define V117_INFO_INT(field) std::fprintf(stream, ",\"" #field "\":%lld", static_cast<long long>(i.field))
#define V117_INFO_NUMBER(field) do { std::fputs(",\"" #field "\":", stream); number(stream, i.field); } while (false)
  V117_INFO_INT(status_ipm); V117_INFO_INT(status_crossover); V117_INFO_INT(errflag);
  V117_INFO_INT(num_var); V117_INFO_INT(num_constr); V117_INFO_INT(num_entries);
  V117_INFO_INT(num_rows_solver); V117_INFO_INT(num_cols_solver); V117_INFO_INT(num_entries_solver);
  V117_INFO_INT(dualized); V117_INFO_INT(dense_cols); V117_INFO_INT(centring_tried); V117_INFO_INT(centring_success);
  V117_INFO_INT(dependent_rows); V117_INFO_INT(dependent_cols); V117_INFO_INT(rows_inconsistent); V117_INFO_INT(cols_inconsistent);
  V117_INFO_INT(primal_dropped); V117_INFO_INT(dual_dropped);
  V117_INFO_NUMBER(abs_presidual); V117_INFO_NUMBER(abs_dresidual); V117_INFO_NUMBER(rel_presidual); V117_INFO_NUMBER(rel_dresidual);
  V117_INFO_NUMBER(pobjval); V117_INFO_NUMBER(dobjval); V117_INFO_NUMBER(rel_objgap); V117_INFO_NUMBER(complementarity);
  V117_INFO_NUMBER(normx); V117_INFO_NUMBER(normy); V117_INFO_NUMBER(normz);
  V117_INFO_NUMBER(objval); V117_INFO_NUMBER(primal_infeas); V117_INFO_NUMBER(dual_infeas);
  V117_INFO_INT(iter); V117_INFO_INT(kktiter1); V117_INFO_INT(kktiter2); V117_INFO_INT(basis_repairs);
  V117_INFO_INT(updates_start); V117_INFO_INT(updates_ipm); V117_INFO_INT(updates_crossover);
  V117_INFO_NUMBER(time_total); V117_INFO_NUMBER(time_ipm1); V117_INFO_NUMBER(time_ipm2);
  V117_INFO_NUMBER(time_starting_basis); V117_INFO_NUMBER(time_crossover);
  V117_INFO_NUMBER(time_kkt_factorize); V117_INFO_NUMBER(time_kkt_solve); V117_INFO_NUMBER(time_maxvol);
  V117_INFO_NUMBER(time_cr1); V117_INFO_NUMBER(time_cr1_AAt); V117_INFO_NUMBER(time_cr1_pre);
  V117_INFO_NUMBER(time_cr2); V117_INFO_NUMBER(time_cr2_NNt); V117_INFO_NUMBER(time_cr2_B); V117_INFO_NUMBER(time_cr2_Bt);
  V117_INFO_NUMBER(ftran_sparse); V117_INFO_NUMBER(btran_sparse);
  V117_INFO_NUMBER(time_ftran); V117_INFO_NUMBER(time_btran); V117_INFO_NUMBER(time_lu_invert); V117_INFO_NUMBER(time_lu_update);
  V117_INFO_NUMBER(mean_fill); V117_INFO_NUMBER(max_fill); V117_INFO_NUMBER(time_symb_invert);
  V117_INFO_INT(maxvol_updates); V117_INFO_INT(maxvol_skipped); V117_INFO_INT(maxvol_passes); V117_INFO_INT(tbl_nnz);
  V117_INFO_NUMBER(tbl_max); V117_INFO_NUMBER(frobnorm_squared); V117_INFO_NUMBER(lambdamax); V117_INFO_NUMBER(volume_increase);
#undef V117_INFO_INT
#undef V117_INFO_NUMBER
  std::fputc('}', stream);
}
inline void write_projection(FILE* stream, const cross::ProjectionStats& p) noexcept {
  std::fprintf(stream, "{\"enabled\":%s,\"scan_started\":%llu,\"scan_completed\":%llu,\"scan_items\":%llu,"
      "\"eligible_rows\":%llu,\"project_started\":%llu,\"project_completed\":%llu,\"project_rc\":",
      p.enabled ? "true" : "false", static_cast<unsigned long long>(p.scan_started),
      static_cast<unsigned long long>(p.scan_completed), static_cast<unsigned long long>(p.scan_items),
      static_cast<unsigned long long>(p.eligible_rows), static_cast<unsigned long long>(p.project_started),
      static_cast<unsigned long long>(p.project_completed));
  if (p.project_rc) std::fputs(*p.project_rc ? "true" : "false", stream);
  else std::fputs("null", stream);
  std::fprintf(stream, ",\"changed_rows\":%llu,\"max_abs_change\":", static_cast<unsigned long long>(p.changed_rows));
  number(stream, p.max_abs_change); std::fputs(",\"seconds\":", stream); number(stream, p.seconds);
  std::fputc('}', stream);
}
inline void write_stats(FILE* stream, const cross::Stats& s) noexcept {
  std::fprintf(stream, "{\"load_started\":%llu,\"load_completed\":%llu,\"crossover_started\":%llu,\"crossover_completed\":%llu,"
      "\"get_started\":%llu,\"get_completed\":%llu,\"callback_checks\":%llu,\"callback_interrupts\":%llu,\"load_rc\":",
      static_cast<unsigned long long>(s.load_started), static_cast<unsigned long long>(s.load_completed),
      static_cast<unsigned long long>(s.crossover_started), static_cast<unsigned long long>(s.crossover_completed),
      static_cast<unsigned long long>(s.get_started), static_cast<unsigned long long>(s.get_completed),
      static_cast<unsigned long long>(s.callback_checks), static_cast<unsigned long long>(s.callback_interrupts));
  optional(stream, s.load_rc); std::fputs(",\"crossover_rc\":", stream); optional(stream, s.crossover_rc);
  std::fputs(",\"get_rc\":", stream); optional(stream, s.get_rc);
  std::fprintf(stream, ",\"native_threads\":%d,\"native_time_limit\":", s.native_threads); number(stream, s.native_time_limit);
  std::fputs(",\"prepare_seconds\":", stream); number(stream, s.prepare_seconds);
  std::fputs(",\"load_seconds\":", stream); number(stream, s.load_seconds);
  std::fputs(",\"crossover_seconds\":", stream); number(stream, s.crossover_seconds);
  std::fputs(",\"get_seconds\":", stream); number(stream, s.get_seconds);
  std::fputs(",\"native_scope_seconds\":", stream); number(stream, s.native_scope_seconds);
  std::fputs(",\"verification_seconds\":", stream); number(stream, s.verification_seconds);
  std::fputs(",\"total_seconds\":", stream); number(stream, s.total_seconds);
  std::fprintf(stream, ",\"verification_started\":%llu,\"verification_completed\":%llu,\"projection\":",
      static_cast<unsigned long long>(s.verification_started), static_cast<unsigned long long>(s.verification_completed));
  write_projection(stream, s.projection);
  std::fputs(",\"preparation\":", stream); write_preparation(stream, s.preparation);
  std::fputs(",\"native_info\":", stream); if (s.native_info) write_info(stream, *s.native_info); else std::fputs("null", stream);
  std::fputc('}', stream);
}
inline void write_basis_export(FILE* stream, const cross::BasisExportStats& s) noexcept {
  std::fprintf(stream, "{\"enabled\":%s,\"statuses_requested\":%s,\"statuses_received\":%s,\"published\":%s,"
      "\"columns\":%llu,\"rows\":%llu}", s.enabled ? "true" : "false", s.statuses_requested ? "true" : "false",
      s.statuses_received ? "true" : "false", s.published ? "true" : "false",
      static_cast<unsigned long long>(s.columns), static_cast<unsigned long long>(s.rows));
}
inline void write_warm_call(FILE* stream, const warm::CallStats& c) noexcept {
  std::fprintf(stream, "{\"started\":%llu,\"completed\":%llu,\"rc\":",
      static_cast<unsigned long long>(c.started), static_cast<unsigned long long>(c.completed));
  optional(stream, c.rc); std::fputs(",\"seconds\":", stream); number(stream, c.seconds); std::fputc('}', stream);
}
inline void write_warm_checks(FILE* stream, const std::optional<warm::Checks>& c) noexcept {
  if (!c) { std::fputs("null", stream); return; }
  std::fprintf(stream, "{\"valid\":%s,\"primal_feasible\":%s,\"basis_dual_feasible\":%s",
      c->valid ? "true" : "false", c->primal_feasible ? "true" : "false", c->basis_dual_feasible ? "true" : "false");
#define V120_CHECK(field) do { std::fputs(",\"" #field "\":", stream); number(stream, c->field); } while (false)
  V120_CHECK(equation_residual); V120_CHECK(bound_violation); V120_CHECK(basic_dual_residual);
  V120_CHECK(equation_ratio); V120_CHECK(bound_ratio); V120_CHECK(basic_dual_ratio);
#undef V120_CHECK
  std::fputc('}', stream);
}
inline void write_warm_stats(FILE* stream, const warm::Stats& s) noexcept {
  std::fputc('{', stream);
  std::fputs("\"init\":", stream); write_warm_call(stream, s.init);
  std::fputs(",\"factor\":", stream); write_warm_call(stream, s.factor);
  std::fputs(",\"solve_n\":", stream); write_warm_call(stream, s.solve_n);
  std::fputs(",\"solve_t\":", stream); write_warm_call(stream, s.solve_t);
  std::fputs(",\"ftran\":", stream); write_warm_call(stream, s.ftran);
  std::fputs(",\"btran\":", stream); write_warm_call(stream, s.btran);
  std::fputs(",\"update\":", stream); write_warm_call(stream, s.update);
#define V120_COUNT(field) std::fprintf(stream, ",\"" #field "\":%llu", static_cast<unsigned long long>(s.field))
  V120_COUNT(steps_started); V120_COUNT(steps_completed); V120_COUNT(pivots); V120_COUNT(bound_flips);
  V120_COUNT(zero_steps); V120_COUNT(refactors); V120_COUNT(rebuild_started); V120_COUNT(rebuild_completed);
  V120_COUNT(pricing_passes); V120_COUNT(work_items); V120_COUNT(verification_started); V120_COUNT(verification_completed);
#undef V120_COUNT
  std::fputs(",\"initial_checks\":", stream); write_warm_checks(stream, s.initial_checks);
  std::fputs(",\"last_checks\":", stream); write_warm_checks(stream, s.last_checks);
  std::fputs(",\"last_entering\":", stream); optional(stream, s.last_entering);
  std::fputs(",\"last_leaving\":", stream); optional(stream, s.last_leaving);
  std::fputs(",\"last_theta\":", stream); optional(stream, s.last_theta);
  std::fputs(",\"last_pivot\":", stream); optional(stream, s.last_pivot);
  std::fputs(",\"projection\":", stream); write_projection(stream, s.projection);
  std::fputs(",\"pricing_seconds\":", stream); number(stream, s.pricing_seconds);
  std::fputs(",\"rebuild_seconds\":", stream); number(stream, s.rebuild_seconds);
  std::fputs(",\"verification_seconds\":", stream); number(stream, s.verification_seconds);
  std::fputs(",\"total_seconds\":", stream); number(stream, s.total_seconds);
  std::fputs(",\"stage\":", stream); quote(stream, s.stage); std::fputc('}', stream);
}
inline void write_warm(FILE* stream, const WarmEvent& w) noexcept {
  std::fprintf(stream, "{\"started\":%llu,\"completed\":%llu,\"max_steps\":%llu,\"status\":",
      static_cast<unsigned long long>(w.started), static_cast<unsigned long long>(w.completed),
      static_cast<unsigned long long>(w.max_steps));
  quote(stream, warm::status_name(w.status));
  std::fputs(",\"raw_quality\":", stream); if (w.raw_quality) write_quality(stream, *w.raw_quality); else std::fputs("null", stream);
  std::fputs(",\"after_quality\":", stream); if (w.after_quality) write_quality(stream, *w.after_quality); else std::fputs("null", stream);
  std::fputs(",\"stats\":", stream); write_warm_stats(stream, w.stats); std::fputc('}', stream);
}
inline void write(FILE* stream, const Log& log) noexcept {
  std::fprintf(stream, "{\"schema\":3,\"enabled\":%s,\"attempted\":%s,\"started\":%llu,\"completed\":%llu,"
      "\"local_accepted\":%s,\"adopted\":%s,\"oracle_decision\":", log.enabled ? "true" : "false", log.attempted ? "true" : "false",
      static_cast<unsigned long long>(log.started), static_cast<unsigned long long>(log.completed),
      log.local_accepted ? "true" : "false", log.adopted ? "true" : "false");
  quote(stream, log.oracle_decision);
  std::fprintf(stream, ",\"norms_ready\":%s,\"norm_items\":%llu,\"bounds_inf\":", log.norms_ready ? "true" : "false",
      static_cast<unsigned long long>(log.norm_items));
  number(stream, log.bounds_inf); std::fputs(",\"cost_inf\":", stream); number(stream, log.cost_inf);
  std::fputs(",\"gate_seconds\":", stream); number(stream, log.gate_seconds);
  std::fputs(",\"norm_seconds\":", stream); number(stream, log.norm_seconds);
  std::fprintf(stream, ",\"gate_checks\":%llu,\"last_gate_iteration\":", static_cast<unsigned long long>(log.gate_checks));
  optional(stream, log.last_gate_iteration); std::fputs(",\"last_gate_measures\":", stream);
  if (!log.last_gate_measures) std::fputs("null", stream);
  else {
    std::fputc('[', stream);
    for (std::size_t k = 0; k < 3; ++k) { if (k) std::fputc(',', stream); number(stream, (*log.last_gate_measures)[k]); }
    std::fputc(']', stream);
  }
  std::fputs(",\"last\":", stream);
  if (!log.last) { std::fputs("null}", stream); return; }
  const auto& event = *log.last;
  std::fprintf(stream, "{\"iteration\":%lld,\"bounds_inf\":", static_cast<long long>(event.iteration)); number(stream, event.bounds_inf);
  std::fputs(",\"cost_inf\":", stream); number(stream, event.cost_inf);
  std::fputs(",\"primal_ratio\":", stream); number(stream, event.primal_ratio);
  std::fputs(",\"dual_ratio\":", stream); number(stream, event.dual_ratio);
  std::fputs(",\"relative_gap\":", stream); number(stream, event.relative_gap);
  std::fprintf(stream, ",\"deadline_saturated\":%s,\"status\":", event.deadline_saturated ? "true" : "false");
  quote(stream, cross::status_name(event.status));
  std::fputs(",\"before_quality\":", stream); write_quality(stream, event.before_quality);
  std::fputs(",\"unprojected_quality\":", stream);
  if (event.unprojected_quality) write_quality(stream, *event.unprojected_quality); else std::fputs("null", stream);
  std::fputs(",\"after_quality\":", stream); if (event.after_quality) write_quality(stream, *event.after_quality); else std::fputs("null", stream);
  std::fputs(",\"stats\":", stream); write_stats(stream, event.stats);
  std::fputs(",\"crossover_status\":", stream); quote(stream, cross::status_name(event.crossover_status));
  std::fputs(",\"crossover_quality\":", stream);
  if (event.crossover_quality) write_quality(stream, *event.crossover_quality); else std::fputs("null", stream);
  std::fputs(",\"basis_export\":", stream); write_basis_export(stream, event.basis_export);
  std::fputs(",\"warm\":", stream); if (event.warm) write_warm(stream, *event.warm); else std::fputs("null", stream);
  std::fputs(",\"scope_seconds\":", stream); number(stream, event.scope_seconds);
  std::fputs(",\"verification_seconds_total\":", stream); number(stream, event.verification_seconds_total);
  std::fputs("}}", stream);
}
struct Report {
  Log& log;
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (log.enabled && solver_v100_presolve::emit_diagnostics) {
      (void)solver_v100_presolve::emit_buffered(stderr, "V117_CROSSOVER ",
          [this](FILE* stream) noexcept { write(stream, log); });
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v117_crossover_state
