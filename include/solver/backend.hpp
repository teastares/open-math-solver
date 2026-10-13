#pragma once

#include "solver/model.hpp"
#include "solver/highs_options.hpp"
#include "solver/ipm_options.hpp"
#include "solver/pdlp_options.hpp"
#include "solver/verification.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace solver {

struct SolveOptions {
  std::string backend = "highs";
  double time_limit = 60.0;
  int threads = 1;
  int seed = 0;
  double mip_relative_gap = 1e-4;
  Tolerances tolerances;
  HighsOptions highs;
  PdlpOptions pdlp;
  IpmOptions ipm;
};

// 状态是后端结论与独立候选检查的组合，证明来源另行记录。
struct SolveResult {
  std::string backend = "highs";
  Model model;
  std::string status = "error";
  std::string backend_status;
  std::string backend_version;
  std::string message;
  std::string verification_level = "none";
  std::vector<double> solution;
  bool has_solution = false;
  std::optional<double> objective;
  std::optional<double> best_bound;
  std::optional<double> absolute_gap;
  std::optional<double> relative_gap;
  std::optional<std::int64_t> nodes;
  std::optional<std::int64_t> simplex_iterations;
  std::optional<std::int64_t> pdlp_iterations;
  std::optional<std::int64_t> pdlp_acceleration_attempts;
  std::optional<std::int64_t> pdlp_acceleration_accepted;
  std::optional<std::int64_t> pdlp_operator_passes;
  std::optional<std::int64_t> ipm_iterations;
  std::optional<std::int64_t> ipm_factorizations;
  std::optional<std::int64_t> ipm_refinement_steps;
  std::optional<std::int64_t> ipm_weighted_krylov_steps;
  std::optional<std::int64_t> ipm_full_newton_preconditioner_solves;
  std::optional<std::int64_t> ipm_full_newton_products;
  std::optional<std::int64_t> ipm_bounded_newton_returns;
  std::optional<std::int64_t> ipm_bounded_newton_inexact_returns;
  std::optional<std::int64_t> ipm_homogeneous_constant_solves;
  std::optional<std::int64_t> ipm_homogeneous_direction_solves;
  std::optional<std::int64_t> ipm_homogeneous_scaled_products;
  std::optional<std::int64_t> ipm_homogeneous_scaled_refinement_steps;
  std::optional<std::int64_t> ipm_homogeneous_scaled_inexact_returns;
  std::optional<double> ipm_homogeneous_tau;
  std::optional<double> ipm_homogeneous_kappa;
  std::optional<std::int64_t> ipm_regularization_searches;
  std::optional<std::int64_t> ipm_regularization_reductions;
  std::optional<std::int64_t> ipm_unshifted_factorizations;
  std::optional<std::int64_t> ipm_direction_retries;
  std::optional<std::int64_t> ipm_direction_retry_successes;
  std::optional<std::int64_t> ipm_krylov_recycle_projections;
  std::optional<std::int64_t> ipm_krylov_recycle_acceptances;
  std::optional<std::int64_t> ipm_krylov_recycle_vectors;
  std::optional<std::int64_t> ipm_krylov_recycle_peak_dimension;
  std::optional<double> ipm_primal_regularization;
  std::optional<double> ipm_dual_regularization;
  std::optional<std::int64_t> ipm_schur_residual_products;
  std::optional<std::int64_t> ipm_schur_refinement_steps;
  std::optional<std::int64_t> ipm_schur_refinement_acceptances;
  std::optional<std::int64_t> ipm_kkt_dimension;
  std::optional<std::int64_t> ipm_factor_nonzeros;
  std::optional<std::string> ipm_factor_backend;
  std::optional<std::int64_t> ipm_structural_fallbacks;
  std::optional<std::int64_t> ipm_eliminated_singletons;
  std::optional<std::int64_t> ipm_eliminated_fixed_variables;
  std::optional<std::int64_t> ipm_presolved_singleton_rows;
  std::optional<std::int64_t> ipm_presolved_fixed_columns;
  std::optional<std::int64_t> ipm_regularized_pivots;
  std::optional<std::int64_t> ipm_backward_error_acceptances;
  std::optional<std::int64_t> ipm_inexact_acceptances;
  std::optional<std::int64_t> ipm_inexact_prefilter_rejections;
  std::optional<std::int64_t> ipm_central_correction_attempts;
  std::optional<std::int64_t> ipm_central_correction_acceptances;
  std::vector<double> row_dual;
  std::optional<LpOptimality> lp_optimality;
  Verification verification;
  double read_seconds = 0.0;
  double solve_seconds = 0.0;
  double verification_seconds = 0.0;
  double total_seconds = 0.0;
};

SolveResult solve_file(const std::filesystem::path& path,
                       const SolveOptions& options = {});
std::string backend_version();

}  // 命名空间 solver
