"""在冻结v98实际源上添加可逆计时边界；不改变算法、预算或数值表达式。"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = '523115acd8348c1d21c7f3bc9ce302fedcd2a5b630ec65d96ee1235c8226de15'
STAGES = (
    'total', 'canonicalize', 'kkt_setup', 'normal_setup', 'symbolic_analysis', 'factor',
    'normal_factor', 'rectangular_assembly', 'explicit_gram_assembly', 'cholmod_numeric',
    'factor_check', 'hsd_constant_direct', 'hsd_direction_direct', 'direct', 'normal_solve',
    'cholmod_solve', 'preconditioner', 'hsd_column_product', 'hsd_true_product', 'coupled',
    'point_update', 'assess_homogeneous', 'verify_local', 'initialize', 'run_homogeneous',
    'corrector_rhs', 'homogeneous_step', 'cone_multiply', 'cone_transpose_multiply',
    'cholmod_factor_adapter', 'solve_factored',
)

SCOPE_ANCHORS = [
    {'unit': 'ipm', 'label': 'canonicalize', 'stage': 'canonicalize',
     'before': 'ConeModel canonicalize(const Model& original, const SolveOptions& options,\n                         Clock::time_point start) {',
     'after': 'ConeModel canonicalize(const Model& original, const SolveOptions& options,\n                         Clock::time_point start) {\n  solver_v99_profile::Scope v99_canonicalize_scope(1);'},
    {'unit': 'ipm', 'label': 'kkt_setup', 'stage': 'kkt_setup',
     'before': '''  KktFactor(const ConeModel& model, const SolveOptions& options, Clock::time_point start,
              SolveResult& result)
      : model_(model), dimension_(model.costs.size() + model.rhs.size()),
        factor_dimension_(model.costs.size()),
        krylov_history_(static_cast<std::size_t>(options.ipm.krylov_history_size)) {''',
     'after': '''  KktFactor(const ConeModel& model, const SolveOptions& options, Clock::time_point start,
              SolveResult& result)
      : model_(model), dimension_(model.costs.size() + model.rhs.size()),
        factor_dimension_(model.costs.size()),
        krylov_history_(static_cast<std::size_t>(options.ipm.krylov_history_size)) {
    solver_v99_profile::Scope v99_kkt_setup_scope(2);'''},
    {'unit': 'ipm', 'label': 'factor', 'stage': 'factor',
     'before': '''  void factor(const std::vector<double>& h, const SolveOptions& options,
                Clock::time_point start, SolveResult& result, int starting_level = 0,
                bool allow_curvature = true) {''',
     'after': '''  void factor(const std::vector<double>& h, const SolveOptions& options,
                Clock::time_point start, SolveResult& result, int starting_level = 0,
                bool allow_curvature = true) {
    solver_v99_profile::Scope v99_factor_scope(5);'''},
    {'unit': 'ipm', 'label': 'hsd_direct', 'stage': ('hsd_constant_direct', 'hsd_direction_direct'),
     'before': '    // 两项计数在同一个预算检查后对应实际 direct 启动；异常也保留已开始的工作。',
     'after': '    // 两项计数在同一个预算检查后对应实际 direct 启动；异常也保留已开始的工作。\n    solver_v99_profile::Scope v99_hsd_direct_scope(constant ? 11 : 12);'},
    {'unit': 'ipm', 'label': 'direct', 'stage': 'direct',
     'before': '''  std::vector<double> direct(const std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {''',
     'after': '''  std::vector<double> direct(const std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    solver_v99_profile::Scope v99_direct_scope(13);'''},
    {'unit': 'ipm', 'label': 'solve_factored', 'stage': 'solve_factored',
     'before': '''  void solve_factored(std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {''',
     'after': '''  void solve_factored(std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    solver_v99_profile::Scope v99_solve_factored_scope(30);'''},
    {'unit': 'ipm', 'label': 'preconditioner', 'stage': 'preconditioner',
     'before': '  const auto precondition = [&](std::span<const double> input, bool refinement) {',
     'after': '  const auto precondition = [&](std::span<const double> input, bool refinement) {\n    solver_v99_profile::Scope v99_preconditioner_scope(16);'},
    {'unit': 'ipm', 'label': 'hsd_product', 'stage': ('hsd_column_product', 'hsd_true_product'),
     'before': '    const auto product = [&]<class Scalar>(std::span<const Scalar> value, std::vector<long double>& image) {',
     'after': '    const auto product = [&]<class Scalar>(std::span<const Scalar> value, std::vector<long double>& image) {\n      solver_v99_profile::Scope v99_hsd_product_scope(std::is_same_v<Scalar, double> ? 17 : 18);'},
    {'unit': 'ipm', 'label': 'point_update', 'stage': 'point_update',
     'before': '''V94UpdatedPoint v94_update_point(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& corrected, double step, const std::function<void()>& budget) {''',
     'after': '''V94UpdatedPoint v94_update_point(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& corrected, double step, const std::function<void()>& budget) {
  solver_v99_profile::Scope v99_point_update_scope(20);'''},
    {'unit': 'ipm', 'label': 'assess_homogeneous', 'stage': 'assess_homogeneous',
     'before': '''bool assess_homogeneous(const ConeModel& model, const Point& point, double tau,
    const SolveOptions& options, Clock::time_point start, SolveResult& result) {''',
     'after': '''bool assess_homogeneous(const ConeModel& model, const Point& point, double tau,
    const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  solver_v99_profile::Scope v99_assess_homogeneous_scope(21);'''},
    {'unit': 'ipm', 'label': 'initialize', 'stage': 'initialize',
     'before': '''void initialize(const ConeModel& model, Point& point, KktFactor& factor,
                  const SolveOptions& options, Clock::time_point start, SolveResult& result) {''',
     'after': '''void initialize(const ConeModel& model, Point& point, KktFactor& factor,
                  const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  solver_v99_profile::Scope v99_initialize_scope(23);'''},
    {'unit': 'ipm', 'label': 'run_homogeneous', 'stage': 'run_homogeneous',
     'before': '''void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,
    Clock::time_point start, SolveResult& result) {''',
     'after': '''void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,
    Clock::time_point start, SolveResult& result) {
  solver_v99_profile::Scope v99_run_homogeneous_scope(24);'''},
    {'unit': 'ipm', 'label': 'corrector_rhs', 'stage': 'corrector_rhs',
     'before': '''long double v94_corrector_rhs(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& affine, double sigma, double mu, std::vector<double>& h,
    const std::function<void()>& budget) {''',
     'after': '''long double v94_corrector_rhs(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& affine, double sigma, double mu, std::vector<double>& h,
    const std::function<void()>& budget) {
  solver_v99_profile::Scope v99_corrector_rhs_scope(25);'''},
    {'unit': 'ipm', 'label': 'homogeneous_step', 'stage': 'homogeneous_step',
     'before': '''double homogeneous_step(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& direction, const SolveOptions& options, Clock::time_point start) {''',
     'after': '''double homogeneous_step(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& direction, const SolveOptions& options, Clock::time_point start) {
  solver_v99_profile::Scope v99_homogeneous_step_scope(26);'''},
    {'unit': 'ipm', 'label': 'cone_multiply', 'stage': 'cone_multiply',
     'before': '  void multiply(const std::vector<double>& x, std::vector<double>& output) const {',
     'after': '  void multiply(const std::vector<double>& x, std::vector<double>& output) const {\n    solver_v99_profile::Scope v99_cone_multiply_scope(27);'},
    {'unit': 'ipm', 'label': 'cone_transpose_multiply', 'stage': 'cone_transpose_multiply',
     'before': '  void transpose(const std::vector<double>& z, std::vector<double>& output) const {',
     'after': '  void transpose(const std::vector<double>& z, std::vector<double>& output) const {\n    solver_v99_profile::Scope v99_cone_transpose_multiply_scope(28);'},
    {'unit': 'ipm', 'label': 'total', 'stage': 'total',
     'before': '  solver_v98_energy_state::Scope energy_isolation(nullptr);',
     'after': '  solver_v98_energy_state::Scope energy_isolation(nullptr);\n  solver_v99_profile::Scope v99_total_scope(0);'},
    {'unit': 'normal', 'label': 'normal_setup', 'stage': 'normal_setup',
     'before': '''  Impl(std::size_t size, std::span<const Index> starts, std::span<const Index> rows,
       std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool use_multifrontal,
       bool use_extended, int refinement_budget, NormalCholeskyKernel kernel)
      : dimension(size), input_nonzeros(rows.size()), extended(use_extended),
        rectangular(kernel == NormalCholeskyKernel::cholmod_supernodal_aat), max_refinement_steps(refinement_budget) {''',
     'after': '''  Impl(std::size_t size, std::span<const Index> starts, std::span<const Index> rows,
       std::span<const std::int8_t> signs, const std::function<void()>& budget_check, bool use_multifrontal,
       bool use_extended, int refinement_budget, NormalCholeskyKernel kernel)
      : dimension(size), input_nonzeros(rows.size()), extended(use_extended),
        rectangular(kernel == NormalCholeskyKernel::cholmod_supernodal_aat), max_refinement_steps(refinement_budget) {
    solver_v99_profile::Scope v99_normal_setup_scope(3);'''},
    {'unit': 'normal', 'label': 'normal_factor', 'stage': 'normal_factor',
     'before': 'bool NormalCholesky::factor(std::span<const double> values, const std::function<void()>& budget_check) {',
     'after': 'bool NormalCholesky::factor(std::span<const double> values, const std::function<void()>& budget_check) {\n  solver_v99_profile::Scope v99_normal_factor_scope(6);'},
    {'unit': 'normal', 'label': 'normal_solve', 'stage': 'normal_solve',
     'before': 'bool NormalCholesky::solve(std::vector<double>& rhs, const std::function<void()>& budget_check) const {',
     'after': 'bool NormalCholesky::solve(std::vector<double>& rhs, const std::function<void()>& budget_check) const {\n  solver_v99_profile::Scope v99_normal_solve_scope(14);'},
    {'unit': 'normal', 'label': 'cholmod_solve', 'stage': 'cholmod_solve',
     'before': '  bool solve(Eigen::VectorXd& rhs, const std::function<void()>& budget_check) const {',
     'after': '  bool solve(Eigen::VectorXd& rhs, const std::function<void()>& budget_check) const {\n    solver_v99_profile::Scope v99_cholmod_solve_scope(15);'},
    {'unit': 'normal', 'label': 'cholmod_factor_adapter', 'stage': 'cholmod_factor_adapter',
     'before': '  bool factor(std::span<const double> values, const std::function<void()>& budget_check) {',
     'after': '  bool factor(std::span<const double> values, const std::function<void()>& budget_check) {\n    solver_v99_profile::Scope v99_cholmod_factor_adapter_scope(29);'},
    {'unit': 'normal', 'label': 'cholmod_factor_check', 'stage': 'factor_check',
     'before': '  bool finite_factor(const std::function<void()>& budget_check) const {\n    const auto& factor = *resources_.factor;',
     'after': '  bool finite_factor(const std::function<void()>& budget_check) const {\n    const auto& factor = *resources_.factor;\n    solver_v99_profile::Scope v99_cholmod_factor_check_scope(10);'},
    {'unit': 'normal', 'label': 'eigen_factor_check', 'stage': 'factor_check',
     'before': '  Index stored_nonzeros() const { return this->m_matrix.nonZeros(); }\n  bool finite_factor(const std::function<void()>& budget_check) const {',
     'after': '  Index stored_nonzeros() const { return this->m_matrix.nonZeros(); }\n  bool finite_factor(const std::function<void()>& budget_check) const {\n    solver_v99_profile::Scope v99_eigen_factor_check_scope(10);'},
    {'unit': 'coupled', 'label': 'coupled', 'stage': 'coupled',
     'before': '''inline std::vector<long double> solve(const Layout& layout, std::span<const double> rhs,
    const std::array<double, 5>& tolerances, const Preconditioner& precondition,
    const ProductDouble& column_multiply, const ProductWide& true_multiply,
    Work& work, const Budget& check = {}, const InitialForcing* initial = nullptr, bool allow_candidates = false,
    bool forcing_weights = false) {''',
     'after': '''inline std::vector<long double> solve(const Layout& layout, std::span<const double> rhs,
    const std::array<double, 5>& tolerances, const Preconditioner& precondition,
    const ProductDouble& column_multiply, const ProductWide& true_multiply,
    Work& work, const Budget& check = {}, const InitialForcing* initial = nullptr, bool allow_candidates = false,
    bool forcing_weights = false) {
  solver_v99_profile::Scope v99_coupled_scope(19);'''},
]

EXPRESSION_ANCHORS = [
    {'unit': 'ipm', 'label': 'verify_local', 'stage': 'verify_local',
     'before': 'verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality)',
     'after': 'solver_v99_profile::call(22, [&] { return verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality); })'},
    {'unit': 'normal', 'label': 'cholmod_symbolic', 'stage': 'symbolic_analysis',
     'before': 'cholmod_l_analyze(&matrix_, &resources_.common)',
     'after': 'solver_v99_profile::call(4, [&] { return cholmod_l_analyze(&matrix_, &resources_.common); })'},
    {'unit': 'normal', 'label': 'ldl_symbolic', 'stage': 'symbolic_analysis',
     'before': 'QDLDL_etree(static_cast<QDLDL_int>(retained), starts.data(), rows.data(), work.data(), counts.data(), tree.data())',
     'after': 'solver_v99_profile::call(4, [&] { return QDLDL_etree(static_cast<QDLDL_int>(retained), starts.data(), rows.data(), work.data(), counts.data(), tree.data()); })'},
    {'unit': 'normal', 'label': 'extended_symbolic', 'stage': 'symbolic_analysis',
     'before': 'extended_cholesky.analyzePattern(extended_matrix)',
     'after': 'solver_v99_profile::call(4, [&] { return extended_cholesky.analyzePattern(extended_matrix); })'},
    {'unit': 'normal', 'label': 'eigen_symbolic', 'stage': 'symbolic_analysis',
     'before': 'cholesky.analyzePattern(matrix)',
     'after': 'solver_v99_profile::call(4, [&] { return cholesky.analyzePattern(matrix); })'},
    {'unit': 'normal', 'label': 'cholmod_numeric', 'stage': 'cholmod_numeric',
     'before': 'cholmod_l_factorize(input, resources_.factor, &resources_.common)',
     'after': 'solver_v99_profile::call(9, [&] { return cholmod_l_factorize(input, resources_.factor, &resources_.common); })'},
]

BLOCK_EDITS = [
    {'unit': 'ipm', 'label': 'include_ipm', 'stage': None,
     'before': '#include "v98-presolve-state.hpp"',
     'after': '#include "v99-profile.hpp"\n#include "v98-presolve-state.hpp"'},
    {'unit': 'normal', 'label': 'include_normal', 'stage': None,
     'before': '#include "solver/detail/normal_cholesky.hpp"',
     'after': '#include "v99-profile.hpp"\n#include "solver/detail/normal_cholesky.hpp"'},
    {'unit': 'coupled', 'label': 'include_coupled', 'stage': None,
     'before': '#pragma once',
     'after': '#pragma once\n#include "v99-profile.hpp"'},
    {'unit': 'ipm', 'label': 'root_report', 'stage': None,
     'before': 'SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n  const auto start = Clock::now();',
     'after': 'SolveResult solve_ipm(const Model& model, const SolveOptions& options) {\n  const auto start = Clock::now();\n  solver_v99_profile::Report v99_profile_report;'},
    {'unit': 'normal', 'label': 'rectangular_assembly', 'stage': 'rectangular_assembly',
     'before': '''      // C=[G D^(-1/2),sqrt(R)]，R 是逐行正对角，不能替换为统一的标量偏移。
      for (std::size_t col = 0; col < state.columns.size(); ++col) {
        check(budget_check);
        const long double divisor = std::sqrt(static_cast<long double>(
            values[static_cast<std::size_t>(state.primal_diagonal[col])]));
        if (!(divisor > 0) || !std::isfinite(divisor)) return false;
        for (Index entry = state.rectangular_matrix.outerIndexPtr()[col];
             entry < state.rectangular_matrix.outerIndexPtr()[col + 1]; ++entry) {
          if ((entry & 4095) == 0) check(budget_check);
          const double coefficient = values[static_cast<std::size_t>(state.rectangular_sources[static_cast<std::size_t>(entry)])];
          const long double scaled = static_cast<long double>(coefficient) / divisor;
          if ((coefficient != 0 && scaled == 0) ||
              !representable(scaled, state.rectangular_matrix.valuePtr()[entry])) return false;
        }
      }
      for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
        if ((row & 4095) == 0) check(budget_check);
        const auto col = state.columns.size() + row;
        const auto entry = state.rectangular_matrix.outerIndexPtr()[col];
        const long double diagonal = -static_cast<long double>(
            values[static_cast<std::size_t>(state.dual_diagonal[row])]);
        if (!representable(std::sqrt(diagonal), state.rectangular_matrix.valuePtr()[entry]) ||
            !(state.rectangular_matrix.valuePtr()[entry] > 0)) return false;
      }
      check(budget_check);
''',
     'after': '''      {
        solver_v99_profile::Scope v99_rectangular_assembly_scope(7);
      // C=[G D^(-1/2),sqrt(R)]，R 是逐行正对角，不能替换为统一的标量偏移。
      for (std::size_t col = 0; col < state.columns.size(); ++col) {
        check(budget_check);
        const long double divisor = std::sqrt(static_cast<long double>(
            values[static_cast<std::size_t>(state.primal_diagonal[col])]));
        if (!(divisor > 0) || !std::isfinite(divisor)) return false;
        for (Index entry = state.rectangular_matrix.outerIndexPtr()[col];
             entry < state.rectangular_matrix.outerIndexPtr()[col + 1]; ++entry) {
          if ((entry & 4095) == 0) check(budget_check);
          const double coefficient = values[static_cast<std::size_t>(state.rectangular_sources[static_cast<std::size_t>(entry)])];
          const long double scaled = static_cast<long double>(coefficient) / divisor;
          if ((coefficient != 0 && scaled == 0) ||
              !representable(scaled, state.rectangular_matrix.valuePtr()[entry])) return false;
        }
      }
      for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
        if ((row & 4095) == 0) check(budget_check);
        const auto col = state.columns.size() + row;
        const auto entry = state.rectangular_matrix.outerIndexPtr()[col];
        const long double diagonal = -static_cast<long double>(
            values[static_cast<std::size_t>(state.dual_diagonal[row])]);
        if (!representable(std::sqrt(diagonal), state.rectangular_matrix.valuePtr()[entry]) ||
            !(state.rectangular_matrix.valuePtr()[entry] > 0)) return false;
      }
      check(budget_check);
      }
'''},
    {'unit': 'normal', 'label': 'explicit_gram_assembly', 'stage': 'explicit_gram_assembly',
     'before': '''  std::fill(state.sums.begin(), state.sums.end(), 0);
  for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
    state.sums[static_cast<std::size_t>(state.diagonal_slots[row])] = -static_cast<long double>(values[static_cast<std::size_t>(state.dual_diagonal[row])]);
  }
  std::size_t product = 0;
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const long double diagonal = values[static_cast<std::size_t>(state.primal_diagonal[col])];
    const auto& column = state.columns[col];
    for (std::size_t right = 0; right < column.size(); ++right) {
      check(budget_check);
      const double b = values[static_cast<std::size_t>(column[right].source)];
      // 仅在宽指数平台复用商，避免窄精度平台的中间溢出或提前下溢。
      constexpr bool wide_range =
          std::numeric_limits<long double>::max_exponent >= 4 * std::numeric_limits<double>::max_exponent &&
          std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits <=
              4 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits);
      long double scaled_b = 0;
      if constexpr (wide_range) scaled_b = static_cast<long double>(b) / diagonal;
      for (std::size_t left = 0; left <= right; ++left) {
        const double a = values[static_cast<std::size_t>(column[left].source)];
        long double contribution;
        if constexpr (wide_range) contribution = static_cast<long double>(a) * scaled_b;
        else contribution = static_cast<long double>(a) * b / diagonal;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(contribution) || (a != 0 && b != 0 && contribution == 0)) return false;
        }
        auto& sum = state.sums[static_cast<std::size_t>(state.product_slots[product++])];
        sum += contribution;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(sum)) return false;
        }
      }
    }
  }
  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {
    if (state.extended) {
      // 保留已付出成本的扩展精度装配结果，不在分解前舍入回 double。
      state.extended_matrix.valuePtr()[entry] = state.sums[entry];
    } else if (!representable(state.sums[entry], state.matrix.valuePtr()[entry])) return false;
  }
  check(budget_check);
''',
     'after': '''  {
    solver_v99_profile::Scope v99_explicit_gram_assembly_scope(8);
  std::fill(state.sums.begin(), state.sums.end(), 0);
  for (std::size_t row = 0; row < state.dual_coordinates.size(); ++row) {
    state.sums[static_cast<std::size_t>(state.diagonal_slots[row])] = -static_cast<long double>(values[static_cast<std::size_t>(state.dual_diagonal[row])]);
  }
  std::size_t product = 0;
  for (std::size_t col = 0; col < state.columns.size(); ++col) {
    check(budget_check);
    const long double diagonal = values[static_cast<std::size_t>(state.primal_diagonal[col])];
    const auto& column = state.columns[col];
    for (std::size_t right = 0; right < column.size(); ++right) {
      check(budget_check);
      const double b = values[static_cast<std::size_t>(column[right].source)];
      // 仅在宽指数平台复用商，避免窄精度平台的中间溢出或提前下溢。
      constexpr bool wide_range =
          std::numeric_limits<long double>::max_exponent >= 4 * std::numeric_limits<double>::max_exponent &&
          std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits <=
              4 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits);
      long double scaled_b = 0;
      if constexpr (wide_range) scaled_b = static_cast<long double>(b) / diagonal;
      for (std::size_t left = 0; left <= right; ++left) {
        const double a = values[static_cast<std::size_t>(column[left].source)];
        long double contribution;
        if constexpr (wide_range) contribution = static_cast<long double>(a) * scaled_b;
        else contribution = static_cast<long double>(a) * b / diagonal;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(contribution) || (a != 0 && b != 0 && contribution == 0)) return false;
        }
        auto& sum = state.sums[static_cast<std::size_t>(state.product_slots[product++])];
        sum += contribution;
        if constexpr (!bounded_gram_arithmetic) {
          if (!std::isfinite(sum)) return false;
        }
      }
    }
  }
  for (std::size_t entry = 0; entry < state.sums.size(); ++entry) {
    if (state.extended) {
      // 保留已付出成本的扩展精度装配结果，不在分解前舍入回 double。
      state.extended_matrix.valuePtr()[entry] = state.sums[entry];
    } else if (!representable(state.sums[entry], state.matrix.valuePtr()[entry])) return false;
  }
  check(budget_check);
  }
'''},
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_files(root):
    parent = Path(root) / '.tools/v98-presolve'
    return dict(ipm=parent / 'ipm-presolve.cpp', normal=parent / 'frozen/src/normal_cholesky.cpp',
        coupled=parent / 'v98-coupled.hpp')


def literal_shape_check():
    """约束字面表只有计时声明、原表达式包装和两个装配块的必要括号。"""
    rows = [*SCOPE_ANCHORS, *EXPRESSION_ANCHORS, *BLOCK_EDITS]
    require(len({item['label'] for item in rows}) == len(rows), '所有字面标签须唯一')
    covered = set()
    dynamic = {
        ('hsd_constant_direct', 'hsd_direction_direct'): 'constant ? 11 : 12',
        ('hsd_column_product', 'hsd_true_product'): 'std::is_same_v<Scalar, double> ? 17 : 18',
    }
    for item in rows:
        require(set(item) == {'unit', 'label', 'before', 'after', 'stage'} and item['unit'] in ('ipm', 'normal', 'coupled') and
            all(type(item[key]) is str for key in ('unit', 'label', 'before', 'after')) and
            item['before'] and item['before'] != item['after'], '每个变换须有精确五字段、合法单元和非空增量')
        stage = item['stage']
        if stage is None:
            continue
        require(type(stage) is str and stage in STAGES or type(stage) is tuple and stage in dynamic,
            '阶段只能取冻结31项或两组固定条件')
        covered.update((stage,) if type(stage) is str else stage)
    for item in SCOPE_ANCHORS:
        stage = item['stage']
        expression = str(STAGES.index(stage)) if type(stage) is str else dynamic[stage]
        suffix = item['after'][len(item['before']):]
        require(item['after'].startswith(item['before']) and re.fullmatch(
            r'\n +solver_v99_profile::Scope v99_' + item['label'] + r'_scope\(' + re.escape(expression) + r'\);', suffix),
            'Scope只能在原锚后追加固定计时声明：' + item['label'])
    for item in EXPRESSION_ANCHORS:
        require(item['after'] == 'solver_v99_profile::call(' + str(STAGES.index(item['stage'])) +
            ', [&] { return ' + item['before'] + '; })', '表达式只能按原call接口包装：' + item['label'])
    fixed = {
        'include_ipm': '#include "v99-profile.hpp"\n' + '#include "v98-presolve-state.hpp"',
        'include_normal': '#include "v99-profile.hpp"\n' + '#include "solver/detail/normal_cholesky.hpp"',
        'include_coupled': '#pragma once\n#include "v99-profile.hpp"',
    }
    for item in BLOCK_EDITS:
        label = item['label']
        if label in fixed:
            require(item['stage'] is None and item['after'] == fixed[label], '只有固定三个profile头include')
        elif label == 'root_report':
            require(item['stage'] is None and item['after'] == item['before'] +
                '\n  solver_v99_profile::Report v99_profile_report;', '根Report只能在原Clock后声明')
        else:
            require(label in ('rectangular_assembly', 'explicit_gram_assembly') and item['stage'] == label,
                '只有两个原装配块可加必要括号')
            indent = '      ' if label == 'rectangular_assembly' else '  '
            require(item['after'] == indent + '{\n' + indent + '  solver_v99_profile::Scope v99_' + label +
                '_scope(' + str(STAGES.index(label)) + ');\n' + item['before'] + indent + '}\n',
                '装配只能用Scope和括号包住逐字原块，不纳入后续分解')
    require(covered == set(STAGES), '31个冻结阶段必须全部由实际锚覆盖')


def transform(source, unit):
    require(unit in ('ipm', 'normal', 'coupled'), '未知计时插桩单元')
    source = source.replace('\r\n', '\n')
    require('solver_v99_profile::' not in source and '#include "v99-profile.hpp"' not in source, '拒绝重复计时插桩')
    literal_shape_check()
    patched, changes = source, []
    for item in (*SCOPE_ANCHORS, *EXPRESSION_ANCHORS, *BLOCK_EDITS):
        if item['unit'] != unit:
            continue
        require(patched.count(item['before']) == 1, '原字面锚缺失或不唯一：' + unit + '/' + item['label'])
        patched = patched.replace(item['before'], item['after'], 1)
        changes.append({key: item[key] for key in ('label', 'before', 'after')})
    restored = patched
    for item in reversed(changes):
        require(restored.count(item['after']) == 1, '逆字面锚不唯一：' + unit + '/' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == source, '全部计时增量须逐字逆恢复原单元：' + unit)
    return patched, changes


def static_check(root=None):
    root = Path(root).resolve() if root is not None else ROOT
    data = root / 'knowledge/experiments/data/2026-10-12-halpern-ipm'
    plan_raw = (data / 'v99-plan.json').read_bytes()
    require(hashlib.sha256(plan_raw).hexdigest() == PLAN_SHA256, '冻结v99计划变化')
    plan = json.loads(plan_raw)
    require(tuple(plan['stages']) == STAGES and len(STAGES) == 31, '阶段列表须与冻结31项同序')
    literal_shape_check()
    records, generated = {}, {}
    for unit, path in source_files(root).items():
        raw = path.read_bytes()
        raw_sha256 = hashlib.sha256(raw).hexdigest()
        require(raw_sha256 == plan['protected_parent_inputs'][str(path)], '实际冻结父源变化：' + unit)
        original = raw.decode('utf-8').replace('\r\n', '\n')
        patched, changes = transform(original, unit)
        for invalid in (original + original, patched, original.replace(changes[0]['before'], '', 1)):
            rejected = False
            try:
                transform(invalid, unit)
            except (RuntimeError, ValueError):
                rejected = True
            require(rejected, '重复、已变换或缺失锚来源必须拒绝：' + unit)
        records[unit] = dict(anchors=len(changes), source=str(path), source_sha256=raw_sha256,
            source_text_sha256=digest(original), instrumented_sha256=digest(patched), reversible=True,
            scope_anchors=sum(item['unit'] == unit for item in SCOPE_ANCHORS),
            expression_anchors=sum(item['unit'] == unit for item in EXPRESSION_ANCHORS),
            block_anchors=sum(item['unit'] == unit for item in BLOCK_EDITS), labels=[item['label'] for item in changes])
        generated[unit] = patched
    ipm = generated['ipm']
    root_begin = ipm.index('SolveResult solve_ipm(const Model& model, const SolveOptions& options) {')
    outer = ipm[root_begin:]
    require(outer.index('const auto start = Clock::now();') < outer.index('solver_v99_profile::Report v99_profile_report;') <
        outer.index('solver_v98_coupled_state::Report coupled_report(coupled);') <
        outer.index('solver_v98_energy_state::Report energy_report(energy);') <
        outer.index('solver_v98_presolve::Log log(stats);') <
        outer.index('solver_v98_presolve::ScopedOracle isolation(nullptr);') <
        outer.index('solver_v98_coupled_state::Scope coupled_isolation(nullptr);') <
        outer.index('solver_v98_energy_state::Scope energy_isolation(nullptr);') <
        outer.index('solver_v99_profile::Scope v99_total_scope(0);') < outer.index('if (!stats.enabled)'),
        '原Clock不移动，Report早于旧三Report，总Scope晚于三isolation且早于旁路分支')
    require(ipm.count('solver_v99_profile::Report v99_profile_report;') == 1 and
        ipm.count('solver_v99_profile::Scope v99_total_scope(0);') == 1 and
        'namespace solver::v98_coupled {' in generated['coupled'], '只准公开根reset一次，算法命名空间98保持')
    return dict(passed=True, plan_sha256=PLAN_SHA256, stages=list(STAGES), units=records,
        total_anchors=sum(item['anchors'] for item in records.values()), root_report_after_original_start=True,
        root_total_after_isolation=True, root_total_excludes_old_markers=True, only_timer_changes=True,
        no_new_budget_checks=True, no_namespace_rename=True, parent_sources_unchanged=True,
        point_update_scope='仅原v94_update_point函数体，不含调用站的Point移动提交',
        cholmod_numeric_scope='原cholmod_l_factorize API；stype0融合隐式AA^T及数值工作',
        coupled_exclusive_scope='扣P/A回调后的其余纯核工作，含真验归约/分配/预算，不等同纯正交化')


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
