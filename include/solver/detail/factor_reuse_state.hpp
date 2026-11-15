#pragma once

#include "solver/backend.hpp"
#include "solver/detail/ipm_coupled_state.hpp"
#include "solver/detail/current_diagonal_preconditioner.hpp"
#include "solver/detail/schur_diagonal_transport.hpp"
#include "solver/detail/natural_schur.hpp"

#include <cstdio>
#include <cstddef>
#include <string_view>
#include <vector>

namespace solver::detail::factor_reuse {

using Index = std::int64_t;
// 旧因子候选仅追加一次FG修正；完整新分解路径仍保留原搜索合同。
inline constexpr std::size_t reuse_krylov_budget = 1;

// 保留每次探测实际付出的公共工作；失败后不得从 SolveResult 中减回。
struct PublicCounts {
  Index constant_solves = 0, direction_solves = 0, scaled_products = 0;
  Index inexact_returns = 0, refinement_steps = 0, scaled_refinement_steps = 0, factorizations = 0;
};

inline PublicCounts counts(const SolveResult& result) noexcept {
  return {result.ipm_homogeneous_constant_solves.value_or(0),
      result.ipm_homogeneous_direction_solves.value_or(0), result.ipm_homogeneous_scaled_products.value_or(0),
      result.ipm_homogeneous_scaled_inexact_returns.value_or(0), result.ipm_refinement_steps.value_or(0),
      result.ipm_homogeneous_scaled_refinement_steps.value_or(0), result.ipm_factorizations.value_or(0)};
}

inline PublicCounts difference(PublicCounts after, const PublicCounts& before) noexcept {
  after.constant_solves -= before.constant_solves; after.direction_solves -= before.direction_solves;
  after.scaled_products -= before.scaled_products; after.inexact_returns -= before.inexact_returns;
  after.refinement_steps -= before.refinement_steps; after.scaled_refinement_steps -= before.scaled_refinement_steps;
  after.factorizations -= before.factorizations;
  return after;
}

struct ScalarConsistency {
  Index started = 0, completed = 0, terms = 0, budget_checks = 0;
  std::optional<long double> natural_denominator, positive_denominator, relative_difference;
};

struct Probe {
  Index attempt_iteration = 0, factor_epoch = 0;
  const char* status = "running";
  ipm_coupled_state::Log coupled;
  ipm_energy_state::Log energy;
  PublicCounts public_delta;
  Index current_rows_started = 0, current_rows_completed = 0, current_rows_terms = 0;
  current_diagonal::Work current_diagonal;
  schur_diagonal_transport::Work transport;
  ScalarConsistency scalar_consistency;
  natural_schur::Work natural_schur;
};

struct Log {
  bool enabled = false;
  Index fresh_started = 0, fresh_completed = 0, reused_steps = 0;
  std::vector<Probe> events;
};

inline thread_local Log* context = nullptr;
inline Log* current() noexcept { return context && context->enabled ? context : nullptr; }
inline bool enabled() noexcept { return current() != nullptr; }
struct Scope {
  explicit Scope(Log* next) noexcept : previous(context) { context = next; }
  ~Scope() noexcept { context = previous; }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  Log* previous;
};

struct Collect {
  Probe& probe;
  const SolveResult& result;
  PublicCounts before;
  ~Collect() noexcept {
    probe.public_delta = difference(counts(result), before);
    if (std::string_view(probe.status) == "running") probe.status = "exception";
  }
};

// 只合并完整接受的整组；被拒绝及中断的探测保留在独立事件中。
inline void merge(ipm_coupled_state::Log& target, const ipm_coupled_state::Log& source) noexcept {
#define SOLVER_REUSE_MERGE(name) target.name += source.name
  SOLVER_REUSE_MERGE(constant_calls_started); SOLVER_REUSE_MERGE(constant_calls_completed);
  SOLVER_REUSE_MERGE(directions_started); SOLVER_REUSE_MERGE(directions_completed);
  SOLVER_REUSE_MERGE(preconditioner_started); SOLVER_REUSE_MERGE(preconditioner_completed);
  SOLVER_REUSE_MERGE(products_started); SOLVER_REUSE_MERGE(products_completed);
  SOLVER_REUSE_MERGE(krylov_started); SOLVER_REUSE_MERGE(krylov_completed); SOLVER_REUSE_MERGE(cycles_started);
  SOLVER_REUSE_MERGE(true_checks_started); SOLVER_REUSE_MERGE(true_checks_completed);
  SOLVER_REUSE_MERGE(initial_forcing_checks); SOLVER_REUSE_MERGE(initial_forcing_returns);
  SOLVER_REUSE_MERGE(candidate_forcing_checks); SOLVER_REUSE_MERGE(candidate_forcing_returns);
#undef SOLVER_REUSE_MERGE
  if (source.last) target.last = source.last;
}

inline void merge(ipm_energy_state::Log& target, const ipm_energy_state::Log& source) noexcept {
  target.started += source.started; target.completed += source.completed;
  target.primal_terms += source.primal_terms; target.row_terms += source.row_terms;
  target.budget_checks += source.budget_checks;
  if (source.last) target.last = source.last;
}

inline void write(FILE* stream, const PublicCounts& value) noexcept {
  std::fprintf(stream, "{\"constant_solves\":%lld,\"direction_solves\":%lld,\"scaled_products\":%lld,"
      "\"inexact_returns\":%lld,\"refinement_steps\":%lld,\"scaled_refinement_steps\":%lld,\"factorizations\":%lld}",
      static_cast<long long>(value.constant_solves), static_cast<long long>(value.direction_solves),
      static_cast<long long>(value.scaled_products), static_cast<long long>(value.inexact_returns),
      static_cast<long long>(value.refinement_steps), static_cast<long long>(value.scaled_refinement_steps),
      static_cast<long long>(value.factorizations));
}

inline void write_current(FILE* stream, const Probe& event) noexcept {
  const auto& work = event.current_diagonal;
  std::fprintf(stream, "{\"columns\":%llu,\"rows\":%llu,\"nonzeros\":%llu,"
      "\"current_rows_started\":%lld,\"current_rows_completed\":%lld,\"current_rows_terms\":%lld",
      static_cast<unsigned long long>(work.columns), static_cast<unsigned long long>(work.rows),
      static_cast<unsigned long long>(work.nonzeros), static_cast<long long>(event.current_rows_started),
      static_cast<long long>(event.current_rows_completed), static_cast<long long>(event.current_rows_terms));
#define SOLVER_CURRENT_FIELD(name) \
  std::fprintf(stream, ",\"" #name "\":%lld", static_cast<long long>(work.name))
  SOLVER_CURRENT_FIELD(validation_started); SOLVER_CURRENT_FIELD(validation_completed);
  SOLVER_CURRENT_FIELD(validation_terms);
  SOLVER_CURRENT_FIELD(applications_started); SOLVER_CURRENT_FIELD(applications_completed);
  SOLVER_CURRENT_FIELD(transpose_started); SOLVER_CURRENT_FIELD(transpose_completed);
  SOLVER_CURRENT_FIELD(direct_started); SOLVER_CURRENT_FIELD(direct_completed);
  SOLVER_CURRENT_FIELD(product_started); SOLVER_CURRENT_FIELD(product_completed);
  SOLVER_CURRENT_FIELD(transpose_terms); SOLVER_CURRENT_FIELD(product_terms);
  SOLVER_CURRENT_FIELD(row_divisions); SOLVER_CURRENT_FIELD(budget_checks);
#undef SOLVER_CURRENT_FIELD
  std::fputc('}', stream);
}

inline void write_transport(FILE* stream, const schur_diagonal_transport::Work& work) noexcept {
  std::fprintf(stream, "{\"columns\":%llu,\"rows\":%llu,\"nonzeros\":%llu",
      static_cast<unsigned long long>(work.columns), static_cast<unsigned long long>(work.rows),
      static_cast<unsigned long long>(work.nonzeros));
#define SOLVER_TRANSPORT_FIELD(name) \
  std::fprintf(stream, ",\"" #name "\":%lld", static_cast<long long>(work.name))
  SOLVER_TRANSPORT_FIELD(prepare_started); SOLVER_TRANSPORT_FIELD(prepare_completed);
  SOLVER_TRANSPORT_FIELD(cache_epoch); SOLVER_TRANSPORT_FIELD(cache_hits);
  SOLVER_TRANSPORT_FIELD(old_build_started); SOLVER_TRANSPORT_FIELD(old_build_completed);
  SOLVER_TRANSPORT_FIELD(old_row_terms); SOLVER_TRANSPORT_FIELD(scratch_row_terms);
  SOLVER_TRANSPORT_FIELD(merge_terms); SOLVER_TRANSPORT_FIELD(unique_terms);
  SOLVER_TRANSPORT_FIELD(old_terms); SOLVER_TRANSPORT_FIELD(current_terms); SOLVER_TRANSPORT_FIELD(column_terms);
  SOLVER_TRANSPORT_FIELD(input_started); SOLVER_TRANSPORT_FIELD(input_completed); SOLVER_TRANSPORT_FIELD(input_terms);
  SOLVER_TRANSPORT_FIELD(direct_started); SOLVER_TRANSPORT_FIELD(direct_completed);
  SOLVER_TRANSPORT_FIELD(output_started); SOLVER_TRANSPORT_FIELD(output_completed); SOLVER_TRANSPORT_FIELD(output_terms);
  SOLVER_TRANSPORT_FIELD(budget_checks);
#undef SOLVER_TRANSPORT_FIELD
#define SOLVER_TRANSPORT_RANGE(name) \
  std::fputs(",\"" #name "\":", stream); ipm_coupled_state::number(stream, work.name)
  SOLVER_TRANSPORT_RANGE(old_min); SOLVER_TRANSPORT_RANGE(old_max);
  SOLVER_TRANSPORT_RANGE(current_min); SOLVER_TRANSPORT_RANGE(current_max);
  SOLVER_TRANSPORT_RANGE(scale_min); SOLVER_TRANSPORT_RANGE(scale_max);
#undef SOLVER_TRANSPORT_RANGE
  std::fputc('}', stream);
}

inline void write_scalar(FILE* stream, const ScalarConsistency& value) noexcept {
  std::fprintf(stream, "{\"started\":%lld,\"completed\":%lld,\"terms\":%lld,\"budget_checks\":%lld,"
      "\"natural_denominator\":", static_cast<long long>(value.started), static_cast<long long>(value.completed),
      static_cast<long long>(value.terms), static_cast<long long>(value.budget_checks));
  ipm_coupled_state::number(stream, value.natural_denominator);
  std::fputs(",\"positive_denominator\":", stream); ipm_coupled_state::number(stream, value.positive_denominator);
  std::fputs(",\"relative_difference\":", stream); ipm_coupled_state::number(stream, value.relative_difference);
  std::fputc('}', stream);
}

inline void write_natural(FILE* stream, const natural_schur::Work& work) noexcept {
  std::fprintf(stream, "{\"decision\":\"%s\"", work.decision);
#define SOLVER_NATURAL_FIELD(name) \
  std::fprintf(stream, ",\"" #name "\":%lld", static_cast<long long>(work.name))
  SOLVER_NATURAL_FIELD(started); SOLVER_NATURAL_FIELD(completed); SOLVER_NATURAL_FIELD(terms);
  SOLVER_NATURAL_FIELD(budget_checks); SOLVER_NATURAL_FIELD(environment_checks);
  SOLVER_NATURAL_FIELD(divisions); SOLVER_NATURAL_FIELD(products); SOLVER_NATURAL_FIELD(accumulations);
  SOLVER_NATURAL_FIELD(abs_accumulations); SOLVER_NATURAL_FIELD(final_subtractions);
  SOLVER_NATURAL_FIELD(bound_operations); SOLVER_NATURAL_FIELD(enclosure_steps);
#undef SOLVER_NATURAL_FIELD
#define SOLVER_NATURAL_RANGE(name) \
  std::fputs(",\"" #name "\":", stream); ipm_coupled_state::number(stream, work.name)
  SOLVER_NATURAL_RANGE(natural); SOLVER_NATURAL_RANGE(abs_products);
  SOLVER_NATURAL_RANGE(lower); SOLVER_NATURAL_RANGE(upper);
  SOLVER_NATURAL_RANGE(error_bound); SOLVER_NATURAL_RANGE(margin);
#undef SOLVER_NATURAL_RANGE
  std::fputc('}', stream);
}

inline void write(FILE* stream, const Log& log) noexcept {
  std::fprintf(stream, "{\"schema\":5,\"enabled\":%s,\"max_krylov_steps\":%llu,"
      "\"preconditioner\":\"current_diagonal_congruence\",\"energy_model\":\"current_positive_approximation\","
      "\"scalar_model\":\"natural_schur_enclosed\","
      "\"fresh_started\":%lld,\"fresh_completed\":%lld,"
      "\"reused_steps\":%lld,\"events\":[", log.enabled ? "true" : "false",
      static_cast<unsigned long long>(reuse_krylov_budget),
      static_cast<long long>(log.fresh_started), static_cast<long long>(log.fresh_completed),
      static_cast<long long>(log.reused_steps));
  bool first = true;
  for (const auto& event : log.events) {
    if (!first) std::fputc(',', stream);
    first = false;
    std::fprintf(stream, "{\"attempt_iteration\":%lld,\"factor_epoch\":%lld,\"status\":\"%s\",\"coupled\":",
        static_cast<long long>(event.attempt_iteration), static_cast<long long>(event.factor_epoch), event.status);
    ipm_coupled_state::write(stream, event.coupled);
    std::fputs(",\"energy\":", stream); ipm_energy_state::write(stream, event.energy);
    std::fputs(",\"public_delta\":", stream); write(stream, event.public_delta);
    std::fputs(",\"current_diagonal\":", stream); write_current(stream, event);
    std::fputs(",\"transport\":", stream); write_transport(stream, event.transport);
    std::fputs(",\"scalar_consistency\":", stream); write_scalar(stream, event.scalar_consistency);
    std::fputs(",\"natural_schur\":", stream); write_natural(stream, event.natural_schur);
    std::fputc('}', stream);
  }
  std::fputs("]}", stream);
}

struct Report {
  explicit Report(Log& value) noexcept : log(value) {}
  ~Report() noexcept {
    if (log.enabled && ipm_presolve::emit_diagnostics) {
      (void)ipm_presolve::emit_buffered(stderr, "V122_FACTOR_REUSE ",
          [this](FILE* stream) noexcept { write(stream, log); });
    }
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
  Log& log;
};

}  // 命名空间 solver::detail::factor_reuse
