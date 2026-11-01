#pragma once
#include "diagnostic_state.hpp"
#include "solver/experimental/coupled.hpp"
#include <cstdio>
#include <type_traits>

namespace solver_v100_coupled_state {
using Index = std::int64_t;
struct Call {
  Index attempt_iteration = 0;
  const char* role = "none";
  solver::v100_coupled::Work work;
};
struct Log {
  bool enabled = false;
  Index constant_calls_started = 0, constant_calls_completed = 0;
  Index directions_started = 0, directions_completed = 0;
  Index preconditioner_started = 0, preconditioner_completed = 0;
  Index products_started = 0, products_completed = 0;
  Index krylov_started = 0, krylov_completed = 0, cycles_started = 0;
  Index true_checks_started = 0, true_checks_completed = 0;
  Index initial_forcing_checks = 0, initial_forcing_returns = 0;
  Index candidate_forcing_checks = 0, candidate_forcing_returns = 0;
  std::optional<Call> last;
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
struct Collect {
  Log& log;
  Call& call;
  ~Collect() noexcept {
    const auto& w = call.work;
    log.preconditioner_started += w.preconditioner_started;
    log.preconditioner_completed += w.preconditioner_completed;
    log.products_started += w.products_started; log.products_completed += w.products_completed;
    log.krylov_started += w.krylov_started; log.krylov_completed += w.krylov_completed;
    log.cycles_started += w.cycles_started;
    log.true_checks_started += w.true_checks_started; log.true_checks_completed += w.true_checks_completed;
    log.initial_forcing_checks += w.initial_forcing_checks; log.initial_forcing_returns += w.initial_forcing_returns;
    log.candidate_forcing_checks += w.candidate_forcing_checks; log.candidate_forcing_returns += w.candidate_forcing_returns;
  }
};
// 摘要只窄化有限值；下溢后的0不表示数学精确零，严格接纳始终读原LD。
inline void number(FILE* stream, const std::optional<long double>& value) noexcept {
  if (value) {
    const double narrowed = static_cast<double>(*value);
    if (std::isfinite(*value) && std::isfinite(narrowed)) {
      std::fprintf(stream, "%.17g", narrowed);
      return;
    }
  }
  std::fputs("null", stream);
}
inline void write_initial_forcing(FILE* stream,
    const std::optional<::solver::v100_coupled::InitialForcingWork>& value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  const auto& forcing = *value;
  std::fputs("{\"rhs_norms\":[", stream);
  for (std::size_t block = 0; block < 3; ++block) {
    if (block) std::fputc(',', stream);
    number(stream, std::optional<long double>(forcing.rhs_norms[block]));
  }
  std::fputs("],\"defect_norms\":", stream);
  if (!forcing.defect_norms) std::fputs("null", stream);
  else {
    std::fputc('[', stream);
    for (std::size_t block = 0; block < 5; ++block) {
      if (block) std::fputc(',', stream);
      number(stream, std::optional<long double>((*forcing.defect_norms)[block]));
    }
    std::fputc(']', stream);
  }
  std::fputs(",\"measure\":", stream); number(stream, forcing.measure);
  std::fprintf(stream, ",\"decision\":\"%s\"}", forcing.decision);
}
inline void write_candidate_forcing(FILE* stream,
    const std::optional<::solver::v100_coupled::CandidateForcingWork>& value) noexcept {
  if (!value) { std::fputs("null", stream); return; }
  const auto& forcing = *value;
  std::fprintf(stream, "{\"cycle\":%lld,\"krylov_step\":%lld,\"defect_norms\":",
      static_cast<long long>(forcing.cycle), static_cast<long long>(forcing.krylov_step));
  if (!forcing.defect_norms) std::fputs("null", stream);
  else {
    std::fputc('[', stream);
    for (std::size_t block = 0; block < 5; ++block) {
      if (block) std::fputc(',', stream);
      number(stream, std::optional<long double>((*forcing.defect_norms)[block]));
    }
    std::fputc(']', stream);
  }
  std::fputs(",\"strict_measure\":", stream); number(stream, forcing.strict_measure);
  std::fputs(",\"best_measure_before\":", stream); number(stream, forcing.best_measure_before);
  std::fputs(",\"measure\":", stream); number(stream, forcing.measure);
  std::fprintf(stream, ",\"decision\":\"%s\"}", forcing.decision);
}
inline void write(FILE* stream, const Log& log) noexcept {
  std::fprintf(stream, "{\"schema\":3,\"enabled\":%s", log.enabled ? "true" : "false");
  const auto count = [&](const char* key, Index value) {
    std::fprintf(stream, ",\"%s\":%lld", key, static_cast<long long>(value));
  };
  count("constant_calls_started", log.constant_calls_started); count("constant_calls_completed", log.constant_calls_completed);
  count("directions_started", log.directions_started); count("directions_completed", log.directions_completed);
  count("preconditioner_started", log.preconditioner_started); count("preconditioner_completed", log.preconditioner_completed);
  count("products_started", log.products_started); count("products_completed", log.products_completed);
  count("krylov_started", log.krylov_started); count("krylov_completed", log.krylov_completed);
  count("cycles_started", log.cycles_started);
  count("true_checks_started", log.true_checks_started); count("true_checks_completed", log.true_checks_completed);
  count("initial_forcing_checks", log.initial_forcing_checks); count("initial_forcing_returns", log.initial_forcing_returns);
  count("candidate_forcing_checks", log.candidate_forcing_checks); count("candidate_forcing_returns", log.candidate_forcing_returns);
  std::fputs(",\"last\":", stream);
  if (!log.last) { std::fputs("null}", stream); return; }
  const auto& last = *log.last;
  const auto& w = last.work;
  std::fprintf(stream, "{\"attempt_iteration\":%lld,\"role\":\"%s\",\"status\":\"%s\",\"dimension\":%llu,\"initial_measure\":",
      static_cast<long long>(last.attempt_iteration), last.role, w.status, static_cast<unsigned long long>(w.dimension));
  number(stream, w.initial_measure); std::fputs(",\"final_measure\":", stream); number(stream, w.final_measure);
  std::fputs(",\"block_measures\":[", stream);
  for (std::size_t block = 0; block < 5; ++block) {
    if (block) std::fputc(',', stream);
    number(stream, w.final_measure ? std::optional<long double>(w.block_measures[block]) : std::nullopt);
  }
  std::fputs("],\"tolerances\":[", stream);
  for (std::size_t block = 0; block < 5; ++block)
    std::fprintf(stream, "%s%.17g", block ? "," : "", w.tolerances[block]);
  std::fputs("],\"initial_forcing\":", stream);
  write_initial_forcing(stream, w.initial_forcing);
  std::fputs(",\"candidate_forcing\":", stream);
  write_candidate_forcing(stream, w.candidate_forcing);
  std::fputs("}}", stream);
}
struct Report {
  Log& log;
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (solver_v100_presolve::emit_diagnostics) {
      (void)solver_v100_presolve::emit_buffered(stderr, "V100_COUPLED ",
          [this](FILE* stream) noexcept { write(stream, log); });
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v100_coupled_state

#include "solver/experimental/energy.hpp"

namespace solver_v100_energy_state {
using Index = std::int64_t;
struct Call {
  Index attempt_iteration = 0;
  solver::v93_energy::Work work;
};
struct Log {
  bool enabled = false;
  Index started = 0, completed = 0, primal_terms = 0, row_terms = 0, budget_checks = 0;
  std::optional<Call> last;
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
struct Collect {
  Log& log;
  Call& call;
  ~Collect() noexcept {
    log.primal_terms += call.work.primal_terms;
    log.row_terms += call.work.row_terms;
    log.budget_checks += call.work.budget_checks;
  }
};
// 有限LD摘要允许下溢到0，不据摘要0断言精确零；溢出double时保null。
inline void number(FILE* stream, const std::optional<long double>& value) noexcept {
  if (value) {
    const double narrowed = static_cast<double>(*value);
    if (std::isfinite(*value) && std::isfinite(narrowed)) {
      std::fprintf(stream, "%.17g", narrowed);
      return;
    }
  }
  std::fputs("null", stream);
}
inline void write(FILE* stream, const Log& log) noexcept {
  std::fprintf(stream, "{\"schema\":1,\"enabled\":%s,\"started\":%lld,\"completed\":%lld,\"primal_terms\":%lld,\"row_terms\":%lld,\"budget_checks\":%lld,\"last\":",
      log.enabled ? "true" : "false", static_cast<long long>(log.started), static_cast<long long>(log.completed),
      static_cast<long long>(log.primal_terms), static_cast<long long>(log.row_terms), static_cast<long long>(log.budget_checks));
  if (!log.last) { std::fputs("null}", stream); return; }
  const auto& call = *log.last;
  const auto& work = call.work;
  std::fprintf(stream, "{\"attempt_iteration\":%lld,\"status\":\"%s\",\"columns\":%llu,\"rows\":%llu,\"primal_regularization\":",
      static_cast<long long>(call.attempt_iteration), work.status,
      static_cast<unsigned long long>(work.columns), static_cast<unsigned long long>(work.rows));
  number(stream, std::optional<long double>(work.primal_regularization));
  std::fprintf(stream, ",\"primal_terms\":%lld,\"row_terms\":%lld,\"primal_energy\":",
      static_cast<long long>(work.primal_terms), static_cast<long long>(work.row_terms));
  number(stream, std::optional<long double>(work.primal_energy));
  std::fputs(",\"row_energy\":", stream); number(stream, std::optional<long double>(work.row_energy));
  std::fputs(",\"denominator\":", stream); number(stream, work.denominator);
  std::fputs("}}", stream);
}
struct Report {
  Log& log;
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (solver_v100_presolve::emit_diagnostics) {
      (void)solver_v100_presolve::emit_buffered(stderr, "V100_ENERGY ",
          [this](FILE* stream) noexcept { write(stream, log); });
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
};
}  // 命名空间 solver_v100_energy_state
