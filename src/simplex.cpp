#include "solver/simplex.hpp"
#include "solver/detail/cold_simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace solver {
namespace {

namespace cross = detail::ipx_crossover;
namespace warm = detail::warm_cleanup;
using Clock = warm::Clock;

const char* solve_status(warm::Status status) noexcept {
  switch (status) {
    case warm::Status::accepted: return "optimal";
    case warm::Status::step_limit: return "iteration_limit";
    case warm::Status::time_limit: return "time_limit";
    case warm::Status::rejected: return "verification_failed";
    case warm::Status::singular_basis:
    case warm::Status::numerical_error: return "numerical_error";
    case warm::Status::unsupported_empty:
    case warm::Status::primal_infeasible:
    case warm::Status::dual_infeasible:
    case warm::Status::no_finite_step: return "unresolved";
    case warm::Status::invalid_input:
    case warm::Status::exception: return "error";
  }
  return "error";
}

void publish_solution(SolveResult& result, warm::Result& candidate,
                      const LpOptimalityOptions& optimality) {
  result.verification = std::move(candidate.verification->primal);
  result.lp_optimality = std::move(candidate.verification->optimality);
  result.has_solution = true;
  result.solution = std::move(candidate.x);
  result.row_dual = std::move(candidate.y);
  result.objective = result.verification.objective;
  result.verification_level = "primal_dual_residuals_checked";

  const auto bound = result.lp_optimality->corrected_dual_bound;
  if (!bound) return;
  const long double objective = *result.objective;
  const long double dual = *bound;
  const long double gap = result.model.sense == ObjectiveSense::minimize
      ? objective - dual : dual - objective;
  const long double scale = std::max({1.0L, std::abs(objective), std::abs(dual)});
  const long double tolerance = optimality.gap_absolute + optimality.gap_relative * scale;
  if (!std::isfinite(gap) || gap < -tolerance) return;
  result.best_bound = bound;
  const long double nonnegative_gap = std::max(0.0L, gap);
  const double absolute_gap = static_cast<double>(nonnegative_gap);
  if (std::isfinite(absolute_gap)) result.absolute_gap = absolute_gap;
  result.relative_gap = static_cast<double>(nonnegative_gap / scale);
}

}  // 匿名命名空间

std::string simplex_version() { return "simplex-cpu-0.1.0"; }

SolveResult solve_simplex(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  SolveResult result;
  result.backend = "simplex";
  result.backend_version = simplex_version();
  result.model = model;
  result.simplex_iterations = 0;
  try {
    const double max_seconds = std::chrono::duration<double>(Clock::time_point::max() - start).count();
    if (!std::isfinite(options.time_limit) || options.time_limit < 0 || options.time_limit >= max_seconds)
      throw std::invalid_argument("单纯形时间预算超出范围");
    if (options.simplex.max_iterations < 0)
      throw std::invalid_argument("单纯形迭代预算必须为非负整数");
    if (options.threads != 1)
      throw std::invalid_argument("自主单纯形目前仅支持单线程");
    cross::detail::validate_tolerances(options.tolerances, options.simplex.optimality);
    const auto errors = validate_model(model);
    if (!errors.empty()) throw std::invalid_argument("模型无效：" + errors.front());
    if (is_mip(model)) throw std::invalid_argument("自主单纯形仅支持连续 LP");

    if (options.time_limit == 0) {
      result.status = "time_limit";
      result.message = "时间预算为零，未启动自主单纯形";
    } else if (options.simplex.max_iterations == 0) {
      result.status = "iteration_limit";
      result.message = "迭代预算为零，未启动自主单纯形";
    } else {
      const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(options.time_limit));
      auto candidate = detail::cold_simplex::run(model, options, deadline);
      result.simplex_iterations = static_cast<std::int64_t>(candidate.stats.steps_completed);
      result.verification_seconds = candidate.stats.verification_seconds;
      result.backend_status = warm::status_name(candidate.status);
      result.status = solve_status(candidate.status);
      result.message = std::move(candidate.message);
      if (Clock::now() >= deadline) {
        result.status = result.backend_status = "time_limit";
        result.message = "自主单纯形已达到时间预算";
      } else if (candidate.status == warm::Status::accepted) {
        if (candidate.verification && warm::detail::quality_ok(*candidate.verification)) {
          publish_solution(result, candidate, options.simplex.optimality);
          result.message = "自主单纯形候选通过原模型完整验收";
        } else {
          result.status = "verification_failed";
          result.message = "自主单纯形候选未通过原模型完整验收";
        }
      }
    }
  } catch (const cross::Failure& failure) {
    result.backend_status = cross::status_name(failure.status);
    result.status = failure.status == cross::Status::time_limit ? "time_limit" :
        failure.status == cross::Status::numerical_error ? "numerical_error" : "error";
    result.message = failure.what();
  } catch (const std::exception& failure) {
    result.status = "error";
    result.message = failure.what();
  }
  if (result.backend_status.empty()) result.backend_status = result.status;
  result.total_seconds = std::chrono::duration<double>(Clock::now() - start).count();
  result.solve_seconds = std::max(0.0, result.total_seconds - result.verification_seconds);
  return result;
}

}  // 命名空间 solver
