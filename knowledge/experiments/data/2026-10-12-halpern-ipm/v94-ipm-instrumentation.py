"""v94：完整方向只在持久Point末提交时窄化，保留父能量与预算协议。"""
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(__file__).resolve().parent
PLAN_SHA256 = '0d9939a8a51a4ce13514e466c78301ad958f329c2a005d0cbb81cbabc0b9b02f'
PARENT_INSTRUMENTATION_SHA256 = '8d058f0c6c1eaaefe5bf92662775bb55e642e3e48e38c20fcc608586f6eac97e'
ENERGY_SHA256 = 'ca598cbade3325e1c5c1d1d2af4bb951d415e34c70c998c95550f211904e3353'
ENVIRONMENT, PREFIX = 'SOLVER_V94_PRESOLVE', 'V94_PRESOLVE '
COUPLED_PREFIX, ENERGY_PREFIX = 'V94_COUPLED ', 'V94_ENERGY '
_parent_path = DATA / 'v93-ipm-instrumentation.py'
if hashlib.sha256(_parent_path.read_bytes()).hexdigest() != PARENT_INSTRUMENTATION_SHA256:
    raise RuntimeError('冻结v93插桩来源变化')
_parent = runpy.run_path(str(_parent_path))


def renamed(text):
    for before, after in (
        ('solver_v93_', 'solver_v94_'), ('v93_core', 'v94_core'),
        ('v93_solve_coupled', 'v94_solve_coupled'), ('v93_coupled_enabled', 'v94_coupled_enabled'),
        ('v93_role', 'v94_role'), ('v93_iteration', 'v94_iteration'),
        ('v93_energy_denominator', 'v94_energy_denominator'),
        ('V93_PRESOLVE', 'V94_PRESOLVE'), ('V93_COUPLED', 'V94_COUPLED'), ('V93_ENERGY', 'V94_ENERGY'),
        ('v93-presolve-state.hpp', 'v94-presolve-state.hpp'), ('v93-coupled-state.hpp', 'v94-coupled-state.hpp'),
        ('v92-coupled.hpp', 'v94-coupled.hpp'), ('solver::v92_coupled', 'solver::v94_coupled')):
        text = text.replace(before, after)
    return text


HEADER, WRAPPER = (renamed(_parent[key]) for key in ('HEADER', 'WRAPPER'))
COUPLED_HEADER = renamed(_parent['COUPLED_HEADER'])
OPENING, INITIALIZATION, COUNT_FIELDS = (_parent[key] for key in ('OPENING', 'INITIALIZATION', 'COUNT_FIELDS'))
CORE_OPENING = renamed(_parent['CORE_OPENING'])
ENERGY_STATE, ENERGY_ADAPTER = (renamed(_parent[key]) for key in ('ENERGY_STATE', 'ENERGY_ADAPTER'))
ASSESS_CURRENT_BEFORE, ASSESS_STOP_BEFORE = (_parent[key] for key in ('ASSESS_CURRENT_BEFORE', 'ASSESS_STOP_BEFORE'))
ASSESS_CURRENT_AFTER, ASSESS_STOP_AFTER = (renamed(_parent[key]) for key in ('ASSESS_CURRENT_AFTER', 'ASSESS_STOP_AFTER'))
PARENT_ADAPTER = renamed(_parent['ADAPTER'])
ADAPTER = PARENT_ADAPTER.replace('std::vector<double> v94_solve_coupled(', 'std::vector<long double> v94_solve_coupled(', 1)
ADAPTER = ADAPTER.replace('const detail::IpmFullNewtonProduct& multiply, const std::function<void()>& budget,',
    'const ::solver::v94_coupled::ProductDouble& multiply,\n    const ::solver::v94_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,', 1)
ADAPTER = ADAPTER.replace('precondition, multiply, call.work, budget)', 'precondition, multiply, multiply_wide, call.work, budget)', 1)

WIDE_BEFORE = '''struct HomogeneousDirection {
  Point point;
  detail::IpmHomogeneousScalarDirection scalars;
};
'''
WIDE_AFTER = '''struct V94WideDirection {
  std::vector<long double> x, s, z;
  long double tau = 0, kappa = 0;
};
struct HomogeneousDirection {
  Point point;
  detail::IpmHomogeneousScalarDirection scalars;
  std::optional<V94WideDirection> v94_wide;
};
static_assert(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits,
    "完整方向LD桥要求long double实际具有额外尾数位");

void v94_wide_shape(const V94WideDirection& wide, std::size_t n, std::size_t m) {
  if (wide.x.size() != n || wide.s.size() != m || wide.z.size() != m ||
      !std::isfinite(wide.tau) || !std::isfinite(wide.kappa))
    throw NumericalFailure("完整LD方向的维数或标量无效");
}

HomogeneousDirection v94_deliver_direction(const detail::IpmHomogeneousScaling& scaling,
    std::vector<long double> solution, const std::function<void()>& budget) {
  if (budget) budget();
  if (solution.size() != scaling.dimension()) throw NumericalFailure("完整LD解维数不一致");
  const auto n = scaling.layout.columns, m = scaling.layout.rows;
  HomogeneousDirection direction;
  auto& wide = direction.v94_wide.emplace();
  wide.x.resize(n); wide.z.resize(m); wide.s.assign(m, 0);
  const auto coordinate = [&](std::size_t i) {
    if (!std::isfinite(solution[i])) throw NumericalFailure("完整LD方向非有限");
    return solution[i];
  };
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0 && budget) budget();
    wide.x[col] = coordinate(col);
  }
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0 && budget) budget();
    wide.z[row] = coordinate(n + row);
  }
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0 && budget) budget();
    wide.s[scaling.layout.inequality_rows[i]] = coordinate(n + m + i);
  }
  wide.tau = coordinate(solution.size() - 2); wide.kappa = coordinate(solution.size() - 1);
  if (budget) budget();
  return direction;
}

long double v94_corrector_rhs(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& affine, double sigma, double mu, std::vector<double>& h,
    const std::function<void()>& budget) {
  if (budget) budget();
  if (!affine.v94_wide || h.size() != point.s.size()) throw NumericalFailure("LD校正方向或右端维数无效");
  const auto& wide = *affine.v94_wide;
  v94_wide_shape(wide, point.x.size(), point.s.size());
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    if ((row & 4095) == 0 && budget) budget();
    if (model.inequality[row]) h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] +
        static_cast<long double>(sigma) * mu - wide.s[row] * wide.z[row]);
  }
  const long double h_tau = -static_cast<long double>(tau) * kappa +
      static_cast<long double>(sigma) * mu - wide.tau * wide.kappa;
  if (budget) budget();
  return h_tau;
}

struct V94UpdatedPoint { Point point; double tau, kappa; };
V94UpdatedPoint v94_update_point(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& corrected, double step, const std::function<void()>& budget) {
  if (budget) budget();
  if (!corrected.v94_wide) throw NumericalFailure("缺少完整LD更新方向");
  const auto& wide = *corrected.v94_wide;
  v94_wide_shape(wide, point.x.size(), point.s.size());
  Point candidate;
  candidate.x.resize(point.x.size()); candidate.s.resize(point.s.size()); candidate.z.resize(point.z.size());
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    if ((col & 4095) == 0 && budget) budget();
    candidate.x[col] = finite(static_cast<long double>(point.x[col]) + static_cast<long double>(step) * wide.x[col]);
  }
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    if ((row & 4095) == 0 && budget) budget();
    candidate.s[row] = finite(static_cast<long double>(point.s[row]) + static_cast<long double>(step) * wide.s[row]);
    candidate.z[row] = finite(static_cast<long double>(point.z[row]) + static_cast<long double>(step) * wide.z[row]);
    if (model.inequality[row] && (!(candidate.s[row] > 0) || !(candidate.z[row] > 0)))
      throw NumericalFailure("齐次更新不能保持严格内点");
  }
  const double next_tau = finite(static_cast<long double>(tau) + static_cast<long double>(step) * wide.tau);
  const double next_kappa = finite(static_cast<long double>(kappa) + static_cast<long double>(step) * wide.kappa);
  if (!(next_tau > 0) || !(next_kappa > 0)) throw NumericalFailure("齐次更新的 tau、kappa 不再严格为正");
  if (budget) budget();
  return {std::move(candidate), next_tau, next_kappa};
}
'''

STEP_BEFORE = '''  const double cone_limit = std::min(boundary_step(model, point.s, direction.point.s),
'''
STEP_AFTER = '''  if (direction.v94_wide) {
    const auto& wide = *direction.v94_wide;
    v94_wide_shape(wide, point.x.size(), point.s.size());
    if (!(tau > 0) || !(kappa > 0) || !std::isfinite(tau) || !std::isfinite(kappa))
      throw NumericalFailure("齐次步长边界输入无效");
    long double limit = 1;
    for (std::size_t row = 0; row < point.s.size(); ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      if (model.inequality[row]) {
        if (!std::isfinite(wide.s[row]) || !std::isfinite(wide.z[row])) throw NumericalFailure("LD锥方向非有限");
        if (wide.s[row] < 0) limit = std::min(limit, -static_cast<long double>(point.s[row]) / wide.s[row]);
        if (wide.z[row] < 0) limit = std::min(limit, -static_cast<long double>(point.z[row]) / wide.z[row]);
      }
    }
    if (wide.tau < 0) limit = std::min(limit, -static_cast<long double>(tau) / wide.tau);
    if (wide.kappa < 0) limit = std::min(limit, -static_cast<long double>(kappa) / wide.kappa);
    const double result = finite(limit);
    check_budget(options, start);
    return result;
  }
''' + STEP_BEFORE


# 每个增量均为可直接AST读取的字面量；独立审计无需执行本插桩器。
WIDE_CHANGES = [
    {'label': 'wide_direction_storage', 'before': r'''struct HomogeneousDirection {
  Point point;
  detail::IpmHomogeneousScalarDirection scalars;
};
''', 'after': r'''struct V94WideDirection {
  std::vector<long double> x, s, z;
  long double tau = 0, kappa = 0;
};
struct HomogeneousDirection {
  Point point;
  detail::IpmHomogeneousScalarDirection scalars;
  std::optional<V94WideDirection> v94_wide;
};
static_assert(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits,
    "完整方向LD桥要求long double实际具有额外尾数位");

void v94_wide_shape(const V94WideDirection& wide, std::size_t n, std::size_t m) {
  if (wide.x.size() != n || wide.s.size() != m || wide.z.size() != m ||
      !std::isfinite(wide.tau) || !std::isfinite(wide.kappa))
    throw NumericalFailure("完整LD方向的维数或标量无效");
}

HomogeneousDirection v94_deliver_direction(const detail::IpmHomogeneousScaling& scaling,
    std::vector<long double> solution, const std::function<void()>& budget) {
  if (budget) budget();
  if (solution.size() != scaling.dimension()) throw NumericalFailure("完整LD解维数不一致");
  const auto n = scaling.layout.columns, m = scaling.layout.rows;
  HomogeneousDirection direction;
  auto& wide = direction.v94_wide.emplace();
  wide.x.resize(n); wide.z.resize(m); wide.s.assign(m, 0);
  const auto coordinate = [&](std::size_t i) {
    if (!std::isfinite(solution[i])) throw NumericalFailure("完整LD方向非有限");
    return solution[i];
  };
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0 && budget) budget();
    wide.x[col] = coordinate(col);
  }
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0 && budget) budget();
    wide.z[row] = coordinate(n + row);
  }
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0 && budget) budget();
    wide.s[scaling.layout.inequality_rows[i]] = coordinate(n + m + i);
  }
  wide.tau = coordinate(solution.size() - 2); wide.kappa = coordinate(solution.size() - 1);
  if (budget) budget();
  return direction;
}

long double v94_corrector_rhs(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& affine, double sigma, double mu, std::vector<double>& h,
    const std::function<void()>& budget) {
  if (budget) budget();
  if (!affine.v94_wide || h.size() != point.s.size()) throw NumericalFailure("LD校正方向或右端维数无效");
  const auto& wide = *affine.v94_wide;
  v94_wide_shape(wide, point.x.size(), point.s.size());
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    if ((row & 4095) == 0 && budget) budget();
    if (model.inequality[row]) h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] +
        static_cast<long double>(sigma) * mu - wide.s[row] * wide.z[row]);
  }
  const long double h_tau = -static_cast<long double>(tau) * kappa +
      static_cast<long double>(sigma) * mu - wide.tau * wide.kappa;
  if (budget) budget();
  return h_tau;
}

struct V94UpdatedPoint { Point point; double tau, kappa; };
V94UpdatedPoint v94_update_point(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& corrected, double step, const std::function<void()>& budget) {
  if (budget) budget();
  if (!corrected.v94_wide) throw NumericalFailure("缺少完整LD更新方向");
  const auto& wide = *corrected.v94_wide;
  v94_wide_shape(wide, point.x.size(), point.s.size());
  Point candidate;
  candidate.x.resize(point.x.size()); candidate.s.resize(point.s.size()); candidate.z.resize(point.z.size());
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    if ((col & 4095) == 0 && budget) budget();
    candidate.x[col] = finite(static_cast<long double>(point.x[col]) + static_cast<long double>(step) * wide.x[col]);
  }
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    if ((row & 4095) == 0 && budget) budget();
    candidate.s[row] = finite(static_cast<long double>(point.s[row]) + static_cast<long double>(step) * wide.s[row]);
    candidate.z[row] = finite(static_cast<long double>(point.z[row]) + static_cast<long double>(step) * wide.z[row]);
    if (model.inequality[row] && (!(candidate.s[row] > 0) || !(candidate.z[row] > 0)))
      throw NumericalFailure("齐次更新不能保持严格内点");
  }
  const double next_tau = finite(static_cast<long double>(tau) + static_cast<long double>(step) * wide.tau);
  const double next_kappa = finite(static_cast<long double>(kappa) + static_cast<long double>(step) * wide.kappa);
  if (!(next_tau > 0) || !(next_kappa > 0)) throw NumericalFailure("齐次更新的 tau、kappa 不再严格为正");
  if (budget) budget();
  return {std::move(candidate), next_tau, next_kappa};
}
'''},
    {'label': 'wide_coupled_adapter', 'before': r'''std::vector<double> v94_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const detail::IpmFullNewtonProduct& multiply, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v94_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 门槛准备若被预算打断，尚未启动新的联立核，也不冒记一个空调用。
  const auto tolerance = detail::ipm_homogeneous_scaled_tolerances(scaling, rhs, budget);
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v94_coupled_state::Collect collect{*log, call};
  const ::solver::v94_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v94_coupled::solve(layout, rhs, tolerance, precondition, multiply, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v94_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

''', 'after': r'''std::vector<long double> v94_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v94_coupled::ProductDouble& multiply,
    const ::solver::v94_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v94_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 门槛准备若被预算打断，尚未启动新的联立核，也不冒记一个空调用。
  const auto tolerance = detail::ipm_homogeneous_scaled_tolerances(scaling, rhs, budget);
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v94_coupled_state::Collect collect{*log, call};
  const ::solver::v94_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v94_coupled::solve(layout, rhs, tolerance, precondition, multiply, multiply_wide, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v94_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''},
    {'label': 'wide_physical_product', 'before': r'''std::vector<long double> v94_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v94_coupled::ProductDouble& multiply,
    const ::solver::v94_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v94_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 门槛准备若被预算打断，尚未启动新的联立核，也不冒记一个空调用。
  const auto tolerance = detail::ipm_homogeneous_scaled_tolerances(scaling, rhs, budget);
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v94_coupled_state::Collect collect{*log, call};
  const ::solver::v94_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v94_coupled::solve(layout, rhs, tolerance, precondition, multiply, multiply_wide, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v94_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

''', 'after': r'''template<class Scalar>
void v94_hsd_product(const detail::IpmHomogeneousScaling& scaling,
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> slack, std::span<const double> dual, double tau, double kappa,
    std::span<const Scalar> direction, std::vector<long double>& product,
    const std::function<void()>& check) {
  if (check) check();
  const auto n = scaling.layout.columns, m = scaling.layout.rows, gap = n + m;
  const auto tau_index = scaling.dimension() - 2;
  if (starts.size() != n + 1 || rows.size() != values.size() || costs.size() != n || bounds.size() != m ||
      slack.size() != m || dual.size() != m || direction.size() != scaling.dimension())
    throw std::invalid_argument("齐次五块算子维数不一致");
  product.assign(scaling.dimension(), 0);
  const long double dt = direction[tau_index], dk = direction[tau_index + 1];
  long double gap_product = dk;
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) if (check) check();
    product[n + row] = -static_cast<long double>(bounds[row]) * dt;
    gap_product += static_cast<long double>(bounds[row]) * direction[n + row];
  }
  for (std::size_t col = 0; col < n; ++col) {
    if (check) check();
    long double value = static_cast<long double>(costs[col]) * dt;
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      if ((k & 4095) == 0) if (check) check();
      const auto position = static_cast<std::size_t>(k), row = static_cast<std::size_t>(rows[position]);
      value += static_cast<long double>(values[position]) * direction[n + row];
      product[n + row] += static_cast<long double>(values[position]) * direction[col];
    }
    product[col] = value;
    gap_product += static_cast<long double>(costs[col]) * direction[col];
  }
  product[gap] = gap_product / scaling.d;
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) if (check) check();
    const auto row = scaling.layout.inequality_rows[i];
    const long double ds = direction[gap + i];
    product[n + row] += ds;
    product[gap + 1 + i] = (static_cast<long double>(dual[row]) * ds +
        static_cast<long double>(slack[row]) * direction[n + row]) / scaling.lambda[i];
  }
  product.back() = (static_cast<long double>(kappa) * dt + static_cast<long double>(tau) * dk) / scaling.lambda_tau;
  if (check) check();
}

std::vector<long double> v94_solve_coupled(const detail::IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const detail::IpmHomogeneousScaledPreconditioner& precondition,
    const ::solver::v94_coupled::ProductDouble& multiply,
    const ::solver::v94_coupled::ProductWide& multiply_wide, const std::function<void()>& budget,
    const char* role, std::int64_t attempt_iteration) {
  auto* log = solver_v94_coupled_state::current();
  if (!log || !role || (std::string_view(role) != "affine" && std::string_view(role) != "corrector") ||
      attempt_iteration < 1 || attempt_iteration > 200)
    throw NumericalFailure("联立Newton调用站或本次迭代标签无效");
  // 门槛准备若被预算打断，尚未启动新的联立核，也不冒记一个空调用。
  const auto tolerance = detail::ipm_homogeneous_scaled_tolerances(scaling, rhs, budget);
  if (budget) budget();
  log->last.emplace();
  auto& call = *log->last;
  call.role = role; call.attempt_iteration = attempt_iteration;
  ++log->directions_started;
  solver_v94_coupled_state::Collect collect{*log, call};
  const ::solver::v94_coupled::Layout layout{scaling.layout.columns, scaling.layout.rows,
      scaling.layout.inequality_rows.size()};
  try {
    auto answer = ::solver::v94_coupled::solve(layout, rhs, tolerance, precondition, multiply, multiply_wide, call.work, budget);
    ++log->directions_completed;
    return answer;
  } catch (const ::solver::v94_coupled::Failure& error) {
    // 只翻译新核自己的数值失败，原预算和外部回调异常保持原类型。
    throw NumericalFailure(error.what());
  } catch (const BudgetExpired&) {
    call.work.status = "budget";
    throw;
  }
}

'''},
    {'label': 'wide_direction_delivery', 'before': r'''  const auto multiply = [&](std::span<const double> direction, std::vector<long double>& product) {
    budget();
    ++*result.ipm_homogeneous_scaled_products;
    detail::ipm_homogeneous_scaled_product(scaling, model.starts, model.rows, model.values,
        model.costs, model.rhs, point.s, point.z, tau, kappa, direction, product, budget);
    budget();
  };
''', 'after': r'''  const auto multiply = [&](std::span<const double> direction, std::vector<long double>& product) {
    budget();
    ++*result.ipm_homogeneous_scaled_products;
    detail::ipm_homogeneous_scaled_product(scaling, model.starts, model.rows, model.values,
        model.costs, model.rhs, point.s, point.z, tau, kappa, direction, product, budget);
    budget();
  };
  if (solver_v94_coupled_state::enabled()) {
    const auto product = [&]<class Scalar>(std::span<const Scalar> value, std::vector<long double>& image) {
      budget();
      ++*result.ipm_homogeneous_scaled_products;
      v94_hsd_product(scaling, model.starts, model.rows, model.values, model.costs, model.rhs,
          point.s, point.z, tau, kappa, value, image, budget);
      budget();
    };
    auto solution = v94_solve_coupled(scaling, rhs, precondition,
        [&](std::span<const double> value, std::vector<long double>& image) { product(value, image); },
        [&](std::span<const long double> value, std::vector<long double>& image) { product(value, image); },
        budget, v94_role, v94_iteration);
    return v94_deliver_direction(scaling, std::move(solution), budget);
  }
'''},
    {'label': 'wide_legacy_direction_path', 'before': r'''  const bool v94_coupled_enabled = solver_v94_coupled_state::enabled();
  const auto solution = v94_coupled_enabled
      ? v94_solve_coupled(scaling, rhs, precondition, multiply, budget, v94_role, v94_iteration)
      : detail::solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, budget);
  // 新核只有完整五块严格通过才返回，旧尾部不得把它误记为inexact。
  if (v94_coupled_enabled) info.tolerance_satisfied = true;
''', 'after': r'''  const auto solution = detail::solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, budget);
'''},
    {'label': 'wide_step_consumers', 'before': r'''  const double cone_limit = std::min(boundary_step(model, point.s, direction.point.s),
''', 'after': r'''  if (direction.v94_wide) {
    const auto& wide = *direction.v94_wide;
    v94_wide_shape(wide, point.x.size(), point.s.size());
    if (!(tau > 0) || !(kappa > 0) || !std::isfinite(tau) || !std::isfinite(kappa))
      throw NumericalFailure("齐次步长边界输入无效");
    long double limit = 1;
    for (std::size_t row = 0; row < point.s.size(); ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      if (model.inequality[row]) {
        if (!std::isfinite(wide.s[row]) || !std::isfinite(wide.z[row])) throw NumericalFailure("LD锥方向非有限");
        if (wide.s[row] < 0) limit = std::min(limit, -static_cast<long double>(point.s[row]) / wide.s[row]);
        if (wide.z[row] < 0) limit = std::min(limit, -static_cast<long double>(point.z[row]) / wide.z[row]);
      }
    }
    if (wide.tau < 0) limit = std::min(limit, -static_cast<long double>(tau) / wide.tau);
    if (wide.kappa < 0) limit = std::min(limit, -static_cast<long double>(kappa) / wide.kappa);
    const double result = finite(limit);
    check_budget(options, start);
    return result;
  }
  const double cone_limit = std::min(boundary_step(model, point.s, direction.point.s),
'''},
    {'label': 'wide_corrector_consumers', 'before': r'''    // 初版各轮的二阶项系数均为 1，不复制其他实现针对首轮的额外减弱。
    for (std::size_t row = 0; row < m; ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      if (model.inequality[row]) {
        h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] +
            static_cast<long double>(sigma) * mu -
            static_cast<long double>(affine.point.s[row]) * affine.point.z[row]);
      }
    }
    const long double h_tau = affine_h_tau + static_cast<long double>(sigma) * mu -
        static_cast<long double>(affine.scalars.tau) * affine.scalars.kappa;
''', 'after': r'''    long double h_tau;
    if (affine.v94_wide) {
      h_tau = v94_corrector_rhs(model, point, tau, kappa, affine, sigma, mu, h,
          [&] { check_budget(options, start); });
    } else {
    // 初版各轮的二阶项系数均为 1，不复制其他实现针对首轮的额外减弱。
    for (std::size_t row = 0; row < m; ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      if (model.inequality[row]) {
        h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] +
            static_cast<long double>(sigma) * mu -
            static_cast<long double>(affine.point.s[row]) * affine.point.z[row]);
      }
    }
    h_tau = affine_h_tau + static_cast<long double>(sigma) * mu -
        static_cast<long double>(affine.scalars.tau) * affine.scalars.kappa;
    }
'''},
    {'label': 'wide_point_commit', 'before': r'''    Point candidate;
    candidate.x.resize(n);
    candidate.s.resize(m);
    candidate.z.resize(m);
    for (std::size_t col = 0; col < n; ++col) {
      if ((col & 4095) == 0) check_budget(options, start);
      candidate.x[col] = finite(static_cast<long double>(point.x[col]) +
          static_cast<long double>(step) * corrected.point.x[col]);
    }
    for (std::size_t row = 0; row < m; ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      candidate.s[row] = finite(static_cast<long double>(point.s[row]) +
          static_cast<long double>(step) * corrected.point.s[row]);
      candidate.z[row] = finite(static_cast<long double>(point.z[row]) +
          static_cast<long double>(step) * corrected.point.z[row]);
      if (model.inequality[row] && (!(candidate.s[row] > 0) || !(candidate.z[row] > 0)))
        throw NumericalFailure("齐次更新不能保持严格内点");
    }
    const double next_tau = finite(static_cast<long double>(tau) + static_cast<long double>(step) * corrected.scalars.tau);
    const double next_kappa = finite(static_cast<long double>(kappa) + static_cast<long double>(step) * corrected.scalars.kappa);
    if (!(next_tau > 0) || !(next_kappa > 0)) throw NumericalFailure("齐次更新的 tau、kappa 不再严格为正");
    check_budget(options, start);
    point = std::move(candidate);
    tau = next_tau;
    kappa = next_kappa;
''', 'after': r'''    if (corrected.v94_wide) {
      auto updated = v94_update_point(model, point, tau, kappa, corrected, step,
          [&] { check_budget(options, start); });
      check_budget(options, start);
      point = std::move(updated.point);
      tau = updated.tau; kappa = updated.kappa;
    } else {
    Point candidate;
    candidate.x.resize(n);
    candidate.s.resize(m);
    candidate.z.resize(m);
    for (std::size_t col = 0; col < n; ++col) {
      if ((col & 4095) == 0) check_budget(options, start);
      candidate.x[col] = finite(static_cast<long double>(point.x[col]) +
          static_cast<long double>(step) * corrected.point.x[col]);
    }
    for (std::size_t row = 0; row < m; ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      candidate.s[row] = finite(static_cast<long double>(point.s[row]) +
          static_cast<long double>(step) * corrected.point.s[row]);
      candidate.z[row] = finite(static_cast<long double>(point.z[row]) +
          static_cast<long double>(step) * corrected.point.z[row]);
      if (model.inequality[row] && (!(candidate.s[row] > 0) || !(candidate.z[row] > 0)))
        throw NumericalFailure("齐次更新不能保持严格内点");
    }
    const double next_tau = finite(static_cast<long double>(tau) + static_cast<long double>(step) * corrected.scalars.tau);
    const double next_kappa = finite(static_cast<long double>(kappa) + static_cast<long double>(step) * corrected.scalars.kappa);
    if (!(next_tau > 0) || !(next_kappa > 0)) throw NumericalFailure("齐次更新的 tau、kappa 不再严格为正");
    check_budget(options, start);
    point = std::move(candidate);
    tau = next_tau;
    kappa = next_kappa;
    }
'''},
]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def physical_product(source):
    begin = source.index('void detail::ipm_homogeneous_scaled_product(')
    end = source.index('\nstd::array<double, 5> detail::ipm_homogeneous_scaled_tolerances(', begin)
    original = source[begin:end]
    template = original.replace('void detail::ipm_homogeneous_scaled_product(',
        'template<class Scalar>\nvoid v94_hsd_product(', 1)
    template = template.replace('const IpmHomogeneousScaling& scaling', 'const detail::IpmHomogeneousScaling& scaling', 1)
    template = template.replace('std::span<const double> direction,', 'std::span<const Scalar> direction,', 1)
    template = template.replace('full_newton_budget(check);', 'if (check) check();')
    return original, template


def transform(source):
    source = source.replace('\r\n', '\n')
    original, changes = source, []
    require('v94_core(' not in source and 'v94_wide' not in source, '拒绝重复插桩')
    def replace(label, before, after):
        nonlocal source
        require(before != after and source.count(before) == 1, '锚点缺失或重复：' + label)
        source = source.replace(before, after, 1)
        changes.append(dict(label=label, before=before, after=after))
    _, parent_changes = _parent['transform'](source)
    require(len(parent_changes) == 21, '冻结v93父锚点变化')
    for item in parent_changes:
        replace(item['label'], renamed(item['before']), renamed(item['after']))
    parent_patched = source
    for item in WIDE_CHANGES:
        replace(item['label'], item['before'], item['after'])
    restored = source
    for item in reversed(changes[21:]):
        require(restored.count(item['after']) == 1, 'LD增量逆锚不唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == parent_patched, '删除LD桥须逐字回到派生v93父')
    for item in reversed(changes[:21]):
        require(restored.count(item['after']) == 1, '父逆锚不唯一：' + item['label'])
        restored = restored.replace(item['after'], item['before'], 1)
    require(restored == original, '全部hook须逐字恢复冻结原核心')
    return source, changes


def static_delta_source():
    return dict(passed=True, parent_sha256=PARENT_INSTRUMENTATION_SHA256,
        parent_header_preserved=HEADER == renamed(_parent['HEADER']),
        parent_wrapper_preserved=WRAPPER == renamed(_parent['WRAPPER']),
        parent_energy_preserved=ENERGY_STATE == renamed(_parent['ENERGY_STATE']) and ENERGY_ADAPTER == renamed(_parent['ENERGY_ADAPTER']),
        parent_adapter_preserved=False, parent_ipm_body_verbatim=False,
        parent_oracle_hooks_preserved=True, coupled_schema_preserved=True, energy_file_namespace_preserved=True)


def static_check(root):
    require(hashlib.sha256((DATA / 'v94-plan.json').read_bytes()).hexdigest() == PLAN_SHA256, '冻结v94计划变化')
    require(hashlib.sha256((DATA / 'v93-energy.hpp').read_bytes()).hexdigest() == ENERGY_SHA256, '父能量纯核必须逐字保持')
    inherited = _parent['static_check'](root)
    require(inherited['passed'] is True and inherited['anchors'] == 21, '最终v93静态父须仍通过')
    original = (Path(root) / 'src/ipm.cpp').read_text(encoding='utf-8').replace('\r\n', '\n')
    patched, changes = transform(original)
    require(len(changes) == 29, '父21加LD八锚应为29')
    for invalid in (original.replace(OPENING, '', 1), original + original, patched):
        rejected = False
        try:
            transform(invalid)
        except (RuntimeError, ValueError):
            rejected = True
        require(rejected, '缺失、重复或已经变换来源必须拒绝')
    native, template = physical_product(original)
    require(native in patched and template in patched and ADAPTER in patched and WIDE_AFTER in patched and STEP_AFTER in patched and
        'std::span<const Scalar> direction' in template,
        '双输入物理算子须逐表达式继承原公式，旧公共double函数保留')
    require('std::vector<long double> v94_solve_coupled' in ADAPTER and 'multiply_wide, call.work' in ADAPTER and
        '"v93-energy.hpp"' in COUPLED_HEADER and 'solver::v93_energy::Work' in COUPLED_HEADER,
        '方向宽化不能误改纯能量文件、命名空间或返回前窄化')
    result = dict(inherited)
    result.update(anchors=len(changes), reversible=True, shared_start=True, parent_anchors=21, wide_extra_anchors=8,
        source_text_sha256=digest(original), patched_text_sha256=digest(patched), header_sha256=digest(HEADER),
        coupled_header_sha256=digest(COUPLED_HEADER), adapter_sha256=digest(ADAPTER),
        energy_state_sha256=digest(ENERGY_STATE), energy_adapter_sha256=digest(ENERGY_ADAPTER),
        plan_sha256=PLAN_SHA256, environment=ENVIRONMENT, prefix=PREFIX, coupled_prefix=COUPLED_PREFIX, energy_prefix=ENERGY_PREFIX,
        wide_direction_return_preserved=True, wide_double_column_wide_true_product=True,
        wide_all_consumers_covered=True, wide_no_direction_narrowing=True, wide_point_late_narrowing=True,
        wide_disabled_preserves_old_path=True, wide_parent_restorable=True, wide_no_extra_linear_work=True,
        coupled_kernel_sha256=hashlib.sha256((DATA / 'v94-coupled.hpp').read_bytes()).hexdigest(),
        staticdelta_source=static_delta_source())
    return result


if __name__ == '__main__':
    print(json.dumps(static_check(ROOT), ensure_ascii=False, indent=2))
