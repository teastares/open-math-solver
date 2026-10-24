#include "solver/ipm.hpp"
#include "solver/detail/extended_ldl.hpp"
#include "solver/detail/ipm_centrality.hpp"
#include "solver/detail/ipm_full_newton.hpp"
#include "solver/detail/ipm_homogeneous.hpp"
#include "solver/detail/ipm_presolve.hpp"
#include "solver/detail/ipm_recycling.hpp"
#include "solver/detail/ipm_residual.hpp"
#include "solver/detail/normal_cholesky.hpp"
#include "solver/detail/multifrontal_ldl.hpp"
#include "solver/detail/regularized_ldl.hpp"
#include "solver/detail/sparse_lu.hpp"

#include <amd.h>
#include <qdldl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef SOLVER_IPM_TRACE
#include <iostream>
#endif

namespace solver {
namespace {

using Clock = std::chrono::steady_clock;
using Index = std::int64_t;

struct NumericalFailure : std::runtime_error { using std::runtime_error::runtime_error; };
struct HomogeneousNonpositiveDenominator : NumericalFailure { using NumericalFailure::NumericalFailure; };
struct BudgetExpired {};

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_budget(const SolveOptions& options, Clock::time_point start) {
  if (elapsed(start) >= options.time_limit) throw BudgetExpired{};
}

double finite(long double value) {
  const double converted = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(converted)) {
    throw NumericalFailure("内点计算出现非有限数");
  }
  return converted;
}

double scaled(double value, double factor) {
  const double result = finite(static_cast<long double>(value) * factor);
  if (value != 0 && result == 0) throw NumericalFailure("内点缩放导致非零数据下溢");
  return result;
}

double symmetric_scaled(double value, double row_factor, double column_factor) {
  double result;
  if constexpr (std::numeric_limits<long double>::max_exponent >=
                    3 * std::numeric_limits<double>::max_exponent &&
                std::numeric_limits<long double>::min_exponent - std::numeric_limits<long double>::digits <=
                    3 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits)) {
    result = finite(static_cast<long double>(value) * row_factor * column_factor);
  } else {
    // long double 未扩展指数范围的平台用尾数和指数合成，避免中间乘积溢出或下溢。
    int value_exponent = 0, row_exponent = 0, column_exponent = 0;
    const double mantissa = std::frexp(value, &value_exponent) *
        std::frexp(row_factor, &row_exponent) * std::frexp(column_factor, &column_exponent);
    result = finite(std::scalbn(mantissa, value_exponent + row_exponent + column_exponent));
  }
  if (value != 0 && result == 0) throw NumericalFailure("KKT 对称均衡导致非零数据下溢");
  return result;
}

double norm_inf(const std::vector<double>& values) {
  double norm = 0;
  for (double value : values) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    norm = std::max(norm, std::abs(value));
  }
  return norm;
}

struct BoundSource {
  Index row = -1;
  double coefficient = 1;
};

struct PresolvedBoundRecovery {
  Index cone_row;
  double sign;
  BoundSource lower, upper;
};

struct PresolvedFixedRecovery {
  std::size_t original_column;
  BoundSource lower, upper;
};

struct SingletonBounds {
  std::vector<double> lower, upper;
  std::vector<BoundSource> lower_source, upper_source;
  std::vector<bool> removed;
  Index count = 0;
};

SingletonBounds presolve_singleton_bounds(const Model& original, const SolveOptions& options,
                                         Clock::time_point start) {
  SingletonBounds plan;
  plan.lower = original.column_lower;
  plan.upper = original.column_upper;
  plan.lower_source.resize(original.costs.size());
  plan.upper_source.resize(original.costs.size());
  plan.removed.assign(original.row_lower.size(), false);
  std::vector<Index> column(original.row_lower.size(), -1);
  std::vector<double> coefficient(original.row_lower.size(), 0);
  for (std::size_t col = 0; col < original.costs.size(); ++col) {
    check_budget(options, start);
    for (Index k = original.column_start[col]; k < original.column_start[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      if (original.values[position] == 0) continue;
      const auto row = static_cast<std::size_t>(original.row_index[position]);
      // 未经可靠合并的重复非零项也保留原行，不将原始存储项误认为独立约束。
      if (column[row] != -1) column[row] = -2;
      else { column[row] = static_cast<Index>(col); coefficient[row] = original.values[position]; }
    }
  }
  for (std::size_t row = 0; row < column.size(); ++row) {
    check_budget(options, start);
    if (column[row] < 0 || (!std::isfinite(original.row_lower[row]) &&
        !std::isfinite(original.row_upper[row]))) continue;
    double lower = original.row_lower[row], upper = original.row_upper[row];
    if (coefficient[row] < 0) std::swap(lower, upper);
    bool reliable = true;
    for (double* bound : {&lower, &upper}) {
      if (std::isfinite(*bound)) {
        const auto quotient = detail::ipm_exact_bound_quotient(*bound, coefficient[row]);
        if (!quotient) { reliable = false; break; }
        *bound = *quotient;
      } else if (coefficient[row] < 0) *bound = -*bound;
    }
    if (!reliable) continue;
    const auto col = static_cast<std::size_t>(column[row]);
    // 严格比较自然保留原列界和先出现行的等值来源，不使用近似活跃判断。
    if (lower > plan.lower[col]) {
      plan.lower[col] = lower;
      plan.lower_source[col] = {static_cast<Index>(row), coefficient[row]};
    }
    if (upper < plan.upper[col]) {
      plan.upper[col] = upper;
      plan.upper_source[col] = {static_cast<Index>(row), coefficient[row]};
    }
    plan.removed[row] = true;
  }
  std::vector<bool> conflict(original.costs.size(), false);
  for (std::size_t col = 0; col < original.costs.size(); ++col) {
    check_budget(options, start);
    if (plan.lower[col] <= plan.upper[col]) continue;
    // 尚无不可行证书；整列撤销本次归并，继续由原模型和旧流程判断。
    conflict[col] = true;
    plan.lower[col] = original.column_lower[col];
    plan.upper[col] = original.column_upper[col];
    plan.lower_source[col] = {};
    plan.upper_source[col] = {};
  }
  for (std::size_t row = 0; row < column.size(); ++row) {
    check_budget(options, start);
    if (!plan.removed[row]) continue;
    if (conflict[static_cast<std::size_t>(column[row])]) plan.removed[row] = false;
    else ++plan.count;
  }
  return plan;
}

struct ConeModel {
  std::vector<Index> starts;
  std::vector<Index> rows;
  std::vector<double> values;
  std::vector<double> costs;
  std::vector<double> rhs;
  std::vector<bool> inequality;
  std::vector<Index> original_row;
  std::vector<double> dual_sign;
  std::vector<double> row_scale;
  std::vector<double> column_scale;
  // 仅固定变量确实消除后保存；空映射表示恒等列映射。
  std::vector<std::size_t> original_columns;
  std::vector<double> fixed_values;
  std::vector<PresolvedBoundRecovery> presolved_bounds;
  std::vector<PresolvedFixedRecovery> presolved_fixed_columns;
  Index presolved_singleton_rows = 0;
  double objective_scale = 1;
  std::size_t barrier_rows = 0;

  Index append_row(double bound, bool nonnegative, Index source, double sign) {
    const auto row = static_cast<Index>(rhs.size());
    rhs.push_back(bound);
    inequality.push_back(nonnegative);
    original_row.push_back(source);
    dual_sign.push_back(sign);
    if (nonnegative) ++barrier_rows;
    return row;
  }

  void multiply(const std::vector<double>& x, std::vector<double>& output) const {
    output.assign(rhs.size(), 0);
    for (std::size_t col = 0; col < costs.size(); ++col) {
      for (Index k = starts[col]; k < starts[col + 1]; ++k) {
        output[static_cast<std::size_t>(rows[static_cast<std::size_t>(k)])] +=
            values[static_cast<std::size_t>(k)] * x[col];
      }
    }
  }

  void transpose(const std::vector<double>& z, std::vector<double>& output) const {
    output.assign(costs.size(), 0);
    for (std::size_t col = 0; col < costs.size(); ++col) {
      for (Index k = starts[col]; k < starts[col + 1]; ++k) {
        output[col] += values[static_cast<std::size_t>(k)] *
            z[static_cast<std::size_t>(rows[static_cast<std::size_t>(k)])];
      }
    }
  }
};

ConeModel canonicalize(const Model& original, const SolveOptions& options,
                         Clock::time_point start) {
  if (options.ipm.eliminate_fixed_variables) {
    bool has_fixed = false;
    for (std::size_t col = 0; col < original.costs.size(); ++col) {
      check_budget(options, start);
      has_fixed = has_fixed || (std::isfinite(original.column_lower[col]) &&
          original.column_lower[col] == original.column_upper[col]);
    }
    if (has_fixed) {
      Model reduced;
      reduced.sense = original.sense;
      reduced.objective_offset = original.objective_offset;
      reduced.row_lower = original.row_lower;
      reduced.row_upper = original.row_upper;
      std::vector<std::size_t> original_columns;
      std::vector<double> fixed_values(original.costs.size(), 0);
      std::vector<long double> shift(original.row_lower.size(), 0);
      bool reliable = true;
      for (std::size_t col = 0; col < original.costs.size(); ++col) {
        check_budget(options, start);
        const bool fixed = std::isfinite(original.column_lower[col]) &&
            original.column_lower[col] == original.column_upper[col];
        if (fixed) {
          fixed_values[col] = original.column_lower[col];
          for (Index k = original.column_start[col]; k < original.column_start[col + 1]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            const auto row = static_cast<std::size_t>(original.row_index[position]);
            const long double product = static_cast<long double>(original.values[position]) * fixed_values[col];
            reliable = reliable && std::isfinite(product) &&
                (original.values[position] == 0 || fixed_values[col] == 0 || product != 0);
            shift[row] += product;
            reliable = reliable && std::isfinite(shift[row]);
          }
        } else {
          original_columns.push_back(col);
          reduced.costs.push_back(original.costs[col]);
          reduced.column_lower.push_back(original.column_lower[col]);
          reduced.column_upper.push_back(original.column_upper[col]);
          reduced.kinds.push_back(original.kinds[col]);
          for (Index k = original.column_start[col]; k < original.column_start[col + 1]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            reduced.row_index.push_back(original.row_index[position]);
            reduced.values.push_back(original.values[position]);
          }
          reduced.column_start.push_back(static_cast<Index>(reduced.values.size()));
        }
      }
      for (std::size_t row = 0; row < shift.size(); ++row) {
        check_budget(options, start);
        for (double* bound : {&reduced.row_lower[row], &reduced.row_upper[row]}) {
          if (!std::isfinite(*bound)) continue;
          const long double shifted = static_cast<long double>(*bound) - shift[row];
          *bound = static_cast<double>(shifted);
          reliable = reliable && std::isfinite(shifted) && std::isfinite(*bound) &&
              (shifted == 0 || *bound != 0);
        }
        if (original.row_lower[row] < original.row_upper[row] &&
            reduced.row_lower[row] >= reduced.row_upper[row]) reliable = false;
      }
      auto unmodified = options;
      unmodified.ipm.eliminate_fixed_variables = false;
      // 仅在变换表示可靠时缩小系统；终验始终基于未修改的原模型。
      if (!reliable) return canonicalize(original, unmodified, start);
      auto result = canonicalize(reduced, unmodified, start);
      if (result.fixed_values.empty()) {
        result.original_columns = std::move(original_columns);
      } else {
        // 内层单行归并也可能消列；先合成映射，再将两层固定值放回原列坐标。
        for (auto& col : result.original_columns) col = original_columns[col];
        for (std::size_t col = 0; col < original_columns.size(); ++col) {
          fixed_values[original_columns[col]] = result.fixed_values[col];
        }
        for (auto& recovery : result.presolved_fixed_columns) {
          recovery.original_column = original_columns[recovery.original_column];
        }
      }
      result.fixed_values = std::move(fixed_values);
      return result;
    }
  }
  ConeModel model;
  std::optional<SingletonBounds> presolved;
  if (options.ipm.presolve_singleton_rows && detail::ipm_exact_bound_checks_available) {
    presolved = presolve_singleton_bounds(original, options, start);
    model.presolved_singleton_rows = presolved->count;
  }
  std::vector<bool> presolved_fixed;
  std::vector<double> shifted_lower, shifted_upper;
  if (options.ipm.eliminate_presolved_fixed_variables && presolved) {
    for (std::size_t col = 0; col < original.costs.size(); ++col) {
      check_budget(options, start);
      const bool originally_fixed = std::isfinite(original.column_lower[col]) &&
          original.column_lower[col] == original.column_upper[col];
      if (!originally_fixed && std::isfinite(presolved->lower[col]) &&
          presolved->lower[col] == presolved->upper[col]) {
        model.presolved_fixed_columns.push_back({col,
            presolved->lower_source[col], presolved->upper_source[col]});
      }
    }
    if (!model.presolved_fixed_columns.empty()) {
      presolved_fixed.assign(original.costs.size(), false);
      model.fixed_values.assign(original.costs.size(), 0);
      for (const auto& recovery : model.presolved_fixed_columns) {
        presolved_fixed[recovery.original_column] = true;
        model.fixed_values[recovery.original_column] = presolved->lower[recovery.original_column];
      }
      shifted_lower = original.row_lower;
      shifted_upper = original.row_upper;
      std::vector<long double> shift(original.row_lower.size(), 0);
      bool reliable = true;
      for (std::size_t col = 0; col < original.costs.size(); ++col) {
        check_budget(options, start);
        if (!presolved_fixed[col]) {
          model.original_columns.push_back(col);
          model.costs.push_back(original.costs[col]);
          continue;
        }
        for (Index k = original.column_start[col]; k < original.column_start[col + 1]; ++k) {
          const auto position = static_cast<std::size_t>(k);
          const auto row = static_cast<std::size_t>(original.row_index[position]);
          if (presolved->removed[row]) continue;
          const long double product = static_cast<long double>(original.values[position]) *
              model.fixed_values[col];
          reliable = reliable && std::isfinite(product) &&
              (original.values[position] == 0 || model.fixed_values[col] == 0 || product != 0);
          shift[row] += product;
          reliable = reliable && std::isfinite(shift[row]);
        }
      }
      for (std::size_t row = 0; row < shift.size(); ++row) {
        check_budget(options, start);
        if (presolved->removed[row]) continue;
        for (double* bound : {&shifted_lower[row], &shifted_upper[row]}) {
          if (!std::isfinite(*bound)) continue;
          const long double shifted = static_cast<long double>(*bound) - shift[row];
          *bound = static_cast<double>(shifted);
          reliable = reliable && std::isfinite(shifted) && std::isfinite(*bound) &&
              (shifted == 0 || *bound != 0);
        }
        if (original.row_lower[row] < original.row_upper[row] &&
            shifted_lower[row] >= shifted_upper[row]) reliable = false;
      }
      // 不做级联单行识别。位移不可靠时整批撤销本轮消列，仍保留原有单行归并。
      if (!reliable) {
        auto unmodified = options;
        unmodified.ipm.eliminate_presolved_fixed_variables = false;
        return canonicalize(original, unmodified, start);
      }
    }
  }
  if (presolved_fixed.empty()) model.costs = original.costs;
  const double sense = original.sense == ObjectiveSense::minimize ? 1 : -1;
  for (double& cost : model.costs) cost *= sense;
  using SignedRow = std::pair<Index, double>;
  std::vector<std::vector<SignedRow>> row_map(original.row_lower.size());
  std::vector<std::vector<SignedRow>> column_bounds(original.costs.size());
  const auto append_bounds = [&](double lower, double upper, Index source,
                                 std::vector<SignedRow>& entries) {
    if (lower == upper) {
      entries.emplace_back(model.append_row(lower, false, source, -1), 1);
    } else {
      if (std::isfinite(lower)) entries.emplace_back(model.append_row(-lower, true, source, 1), -1);
      if (std::isfinite(upper)) entries.emplace_back(model.append_row(upper, true, source, -1), 1);
    }
  };
  for (std::size_t row = 0; row < row_map.size(); ++row) {
    if (presolved && presolved->removed[row]) continue;
    append_bounds(shifted_lower.empty() ? original.row_lower[row] : shifted_lower[row],
        shifted_upper.empty() ? original.row_upper[row] : shifted_upper[row],
        static_cast<Index>(row), row_map[row]);
  }
  for (std::size_t col = 0; col < original.costs.size(); ++col) {
    if (!presolved_fixed.empty() && presolved_fixed[col]) continue;
    if (!presolved) {
      append_bounds(original.column_lower[col], original.column_upper[col], -1, column_bounds[col]);
    } else {
      append_bounds(presolved->lower[col], presolved->upper[col], -1, column_bounds[col]);
      if (presolved->lower_source[col].row >= 0 || presolved->upper_source[col].row >= 0) {
        for (const auto& [row, sign] : column_bounds[col]) {
          model.presolved_bounds.push_back({row, -sign,
              presolved->lower_source[col], presolved->upper_source[col]});
        }
      }
    }
  }
  model.starts.push_back(0);
  for (std::size_t col = 0; col < original.costs.size(); ++col) {
    check_budget(options, start);
    if (!presolved_fixed.empty() && presolved_fixed[col]) continue;
    std::vector<SignedRow> entries = column_bounds[col];
    for (Index k = original.column_start[col]; k < original.column_start[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      for (const auto& [row, sign] : row_map[static_cast<std::size_t>(original.row_index[position])]) {
        entries.emplace_back(row, sign * original.values[position]);
      }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
      return left.first < right.first;
    });
    for (std::size_t i = 0; i < entries.size();) {
      const Index row = entries[i].first;
      long double sum = 0;
      do { sum += entries[i++].second; } while (i < entries.size() && entries[i].first == row);
      const double value = finite(sum);
      if (sum != 0 && value == 0) throw NumericalFailure("重复项合并导致非零数据下溢");
      if (value != 0) { model.rows.push_back(row); model.values.push_back(value); }
    }
    model.starts.push_back(static_cast<Index>(model.values.size()));
  }
  model.row_scale.assign(model.rhs.size(), 1);
  model.column_scale.assign(model.costs.size(), 1);
  std::vector<double> row_norm(model.rhs.size());
  for (int pass = 0; pass < options.ipm.scaling_passes; ++pass) {
    check_budget(options, start);
    std::fill(row_norm.begin(), row_norm.end(), 0);
    for (std::size_t k = 0; k < model.values.size(); ++k) {
      const auto row = static_cast<std::size_t>(model.rows[k]);
      row_norm[row] = std::max(row_norm[row], std::abs(model.values[k]));
    }
    for (std::size_t row = 0; row < model.rhs.size(); ++row) {
      row_norm[row] = row_norm[row] > 0 ? std::clamp(1 / std::sqrt(row_norm[row]), 1e-4, 1e4) : 1;
      model.row_scale[row] = scaled(model.row_scale[row], row_norm[row]);
      model.rhs[row] = scaled(model.rhs[row], row_norm[row]);
    }
    for (std::size_t k = 0; k < model.values.size(); ++k) {
      model.values[k] = scaled(model.values[k], row_norm[static_cast<std::size_t>(model.rows[k])]);
    }
    for (std::size_t col = 0; col < model.costs.size(); ++col) {
      double norm = 0;
      for (Index k = model.starts[col]; k < model.starts[col + 1]; ++k) {
        norm = std::max(norm, std::abs(model.values[static_cast<std::size_t>(k)]));
      }
      const double factor = norm > 0 ? std::clamp(1 / std::sqrt(norm), 1e-4, 1e4) : 1;
      model.column_scale[col] = scaled(model.column_scale[col], factor);
      model.costs[col] = scaled(model.costs[col], factor);
      for (Index k = model.starts[col]; k < model.starts[col + 1]; ++k) {
        model.values[static_cast<std::size_t>(k)] = scaled(model.values[static_cast<std::size_t>(k)], factor);
      }
    }
  }
  if (options.ipm.objective_scaling) {
    for (double cost : model.costs) model.objective_scale = std::max(model.objective_scale, std::abs(cost));
    for (double& cost : model.costs) cost = scaled(cost, 1 / model.objective_scale);
  }
  return model;
}

struct Point {
  std::vector<double> x;
  std::vector<double> s;
  std::vector<double> z;
};

std::vector<double> recover_slack_direction(const ConeModel& model, const Point& point,
    const std::vector<double>& h, const std::vector<double>& solution) {
  std::vector<double> slack(point.s.size(), 0);
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    if (model.inequality[row]) {
      // 按互补方程恢复，避免原始残差的绝对误差淹没接近边界的松弛量。
      slack[row] = finite((static_cast<long double>(h[row]) -
          static_cast<long double>(point.s[row]) * solution[point.x.size() + row]) / point.z[row]);
    }
  }
  return slack;
}

struct InexactDirectionContext {
  const Point& point;
  const std::vector<double>& primal_residual;
  const std::vector<double>& dual_residual;
  const std::vector<double>& h;
  double candidate_tolerance;
};

struct KktEntry { Index row; Index col; double value; Index diagonal; };

class KktFactor {
 public:
  KktFactor(const ConeModel& model, const SolveOptions& options, Clock::time_point start,
              SolveResult& result)
      : model_(model), dimension_(model.costs.size() + model.rhs.size()),
        factor_dimension_(model.costs.size()),
        krylov_history_(static_cast<std::size_t>(options.ipm.krylov_history_size)) {
    check_budget(options, start);
    const auto n = model.costs.size();
    std::vector<Index> starts(n + 1, 0);
    std::vector<Index> rows;
    std::vector<std::vector<Index>> cross(model.rhs.size());
    for (std::size_t col = 0; col < n; ++col) {
      for (Index k = model.starts[col]; k < model.starts[col + 1]; ++k) {
        cross[static_cast<std::size_t>(model.rows[static_cast<std::size_t>(k)])].push_back(static_cast<Index>(col));
      }
    }
    if (options.ipm.linear_accuracy == "backward") {
      for (std::size_t col = 0; col < n; ++col) {
        backward_error_nonzeros_ = std::max(backward_error_nonzeros_,
            static_cast<std::size_t>(model.starts[col + 1] - model.starts[col]));
      }
      for (const auto& row : cross) backward_error_nonzeros_ = std::max(backward_error_nonzeros_, row.size() + 1);
    }
    if (options.ipm.inexact_prefilter) {
      inexact_prefilter_ = detail::make_ipm_inexact_prefilter(model.starts, model.values,
          [&options, &start] { check_budget(options, start); });
    }
    retained_index_.resize(model.rhs.size(), -1);
    singleton_column_.resize(model.rhs.size(), -1);
    singleton_value_.resize(model.rhs.size(), 0);
    for (std::size_t col = 0; col < n; ++col) {
      rows.push_back(static_cast<Index>(col));
      starts[col + 1] = static_cast<Index>(rows.size());
    }
    for (std::size_t row = 0; row < model.rhs.size(); ++row) {
      check_budget(options, start);
      if (options.ipm.eliminate_singletons && model.inequality[row] && cross[row].size() == 1) {
        singleton_column_[row] = cross[row][0];
      } else {
        retained_index_[row] = static_cast<Index>(factor_dimension_);
        rows.insert(rows.end(), cross[row].begin(), cross[row].end());
        rows.push_back(static_cast<Index>(factor_dimension_++));
        starts.push_back(static_cast<Index>(rows.size()));
      }
    }
    std::vector<Index> permutation(factor_dimension_);
    if (factor_dimension_ > 0) {
      const auto status = amd_l_order(static_cast<Index>(factor_dimension_), starts.data(), rows.data(),
                                      permutation.data(), nullptr, nullptr);
      if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) throw NumericalFailure("AMD 稀疏排序失败");
    }
    inverse_.resize(factor_dimension_);
    for (std::size_t i = 0; i < factor_dimension_; ++i) inverse_[static_cast<std::size_t>(permutation[i])] = i;
    if (options.ipm.dynamic_regularization) {
      pivot_signs_.resize(factor_dimension_);
      for (std::size_t i = 0; i < factor_dimension_; ++i) pivot_signs_[inverse_[i]] = i < n ? 1 : -1;
    }
    std::vector<KktEntry> entries;
    entries.reserve(factor_dimension_ + model.values.size());
    for (std::size_t i = 0; i < factor_dimension_; ++i) {
      entries.push_back({static_cast<Index>(inverse_[i]), static_cast<Index>(inverse_[i]), 0, static_cast<Index>(i)});
    }
    for (std::size_t col = 0; col < n; ++col) {
      for (Index k = model.starts[col]; k < model.starts[col + 1]; ++k) {
        const auto position = static_cast<std::size_t>(k);
        const auto row = static_cast<std::size_t>(model.rows[position]);
        if (singleton_column_[row] >= 0) {
          singleton_value_[row] = model.values[position];
          continue;
        }
        const auto a = inverse_[col];
        const auto b = inverse_[static_cast<std::size_t>(retained_index_[row])];
        entries.push_back({static_cast<Index>(std::min(a, b)), static_cast<Index>(std::max(a, b)), model.values[position], -1});
      }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
      return a.col != b.col ? a.col < b.col : a.row < b.row;
    });
    ap_.assign(factor_dimension_ + 1, 0);
    diagonal_.resize(factor_dimension_);
    for (const auto& entry : entries) {
      ++ap_[static_cast<std::size_t>(entry.col) + 1];
      ai_.push_back(static_cast<QDLDL_int>(entry.row));
      if (entry.diagonal >= 0) diagonal_[static_cast<std::size_t>(entry.diagonal)] = ax_.size();
      ax_.push_back(entry.value);
    }
    for (std::size_t col = 0; col < factor_dimension_; ++col) ap_[col + 1] += ap_[col];
    if (options.ipm.equilibrate_kkt) {
      off_diagonal_max_.assign(factor_dimension_, 0);
      equilibration_.resize(factor_dimension_);
      factor_values_.resize(ax_.size());
      for (std::size_t col = 0; col < factor_dimension_; ++col) {
        check_budget(options, start);
        for (QDLDL_int k = ap_[col]; k < ap_[col + 1]; ++k) {
          const auto position = static_cast<std::size_t>(k);
          const auto row = static_cast<std::size_t>(ai_[position]);
          if (row == col) continue;
          const double magnitude = std::abs(ax_[position]);
          off_diagonal_max_[row] = std::max(off_diagonal_max_[row], magnitude);
          off_diagonal_max_[col] = std::max(off_diagonal_max_[col], magnitude);
        }
      }
    }
    const bool normal_multifrontal = options.ipm.linear_solver == "normal_multifrontal";
    const bool normal_extended = options.ipm.linear_solver == "normal_extended";
    const bool normal_refined = options.ipm.linear_solver == "normal_refined";
    const bool normal_cholmod = options.ipm.linear_solver == "normal_cholmod";
    const bool normal_cholmod_aat = options.ipm.linear_solver == "normal_cholmod_aat";
    const bool normal_augmented = options.ipm.linear_solver == "normal_augmented" || normal_multifrontal || normal_extended || normal_refined || normal_cholmod || normal_cholmod_aat;
    if (options.ipm.linear_solver == "sparselu" || options.ipm.linear_solver == "normal" || normal_augmented ||
        options.ipm.linear_solver == "factorhighs") {
      // 适配器均接收实际 AMD 后的 KKT；外部缩放、消元和完整方向恢复保持一致。
      // 显式转换索引，兼容 int64_t 与 QDLDL_int 为不同 64 位整数类型的平台。
      starts.assign(ap_.begin(), ap_.end());
      rows.assign(ai_.begin(), ai_.end());
      check_budget(options, start);
      if (options.ipm.linear_solver == "normal" || normal_augmented || options.ipm.linear_solver == "factorhighs") {
        std::vector<std::int8_t> signs(factor_dimension_);
        for (std::size_t i = 0; i < factor_dimension_; ++i) signs[inverse_[i]] = i < n ? 1 : -1;
        if (options.ipm.linear_solver != "factorhighs") {
          try {
            normal_cholesky_ = std::make_unique<detail::NormalCholesky>(factor_dimension_, starts, rows, signs,
                [&options, &start] { check_budget(options, start); }, normal_multifrontal, normal_extended, normal_refined ? 2 : 0,
                normal_cholmod_aat ? detail::NormalCholeskyKernel::cholmod_supernodal_aat :
                normal_cholmod ? detail::NormalCholeskyKernel::cholmod_supernodal : detail::NormalCholeskyKernel::eigen);
          } catch (const detail::NormalCholeskyBudgetExceeded&) {
            if (!normal_augmented) throw;
            // 仅结构预算触发一次切换；已消耗的分析时间不重置，其他异常继续传播。
            check_budget(options, start);
            ++*result.ipm_structural_fallbacks;
          }
        }
        if (!normal_cholesky_) {
          multifrontal_ldl_ = std::make_unique<detail::MultifrontalLdl>(factor_dimension_, starts, rows, signs,
              [&options, &start] { check_budget(options, start); });
          result.ipm_kkt_dimension = static_cast<Index>(factor_dimension_);
          result.ipm_factor_backend = "factorhighs";
        } else {
          // factor_dimension_ 仍是输入向量的维数；报告字段使用实际 Cholesky 分解维数。
          result.ipm_kkt_dimension = normal_cholesky_->factor_dimension();
          result.ipm_factor_backend = normal_cholmod_aat ? "normal_cholmod_aat" : normal_cholmod ? "normal_cholmod" : normal_multifrontal ? "normal_multifrontal" : normal_extended ? "normal_extended" : normal_refined ? "normal_refined" : "normal";
        }
      } else {
        sparse_lu_ = std::make_unique<detail::SparseLu>(factor_dimension_, starts, rows);
        result.ipm_kkt_dimension = static_cast<Index>(factor_dimension_);
        result.ipm_factor_backend = "sparselu";
      }
      check_budget(options, start);
      result.ipm_eliminated_singletons = static_cast<Index>(dimension_ - factor_dimension_);
      // 非零数在成功数值分解后才报告，不借用 LDL 的符号计数或分配其工作区。
      return;
    }
    extended_ = options.ipm.linear_solver == "qdldl_extended";
    etree_.resize(factor_dimension_); lnz_.resize(factor_dimension_); iwork_.resize(factor_dimension_ * 3);
    bwork_.resize(factor_dimension_);
    if (extended_) {
      extended_ax_.resize(ax_.size());
      extended_work_.resize(factor_dimension_);
      extended_d_.resize(factor_dimension_);
      extended_inverse_.resize(factor_dimension_);
    } else {
      fwork_.resize(factor_dimension_); d_.resize(factor_dimension_); dinv_.resize(factor_dimension_);
    }
    lp_.resize(factor_dimension_ + 1);
    const auto nonzeros = QDLDL_etree(static_cast<QDLDL_int>(factor_dimension_), ap_.data(), ai_.data(),
                                     iwork_.data(), lnz_.data(), etree_.data());
    if (nonzeros < 0) throw NumericalFailure("稀疏 KKT 符号分析失败");
    li_.resize(static_cast<std::size_t>(nonzeros));
    if (extended_) extended_lx_.resize(static_cast<std::size_t>(nonzeros));
    else lx_.resize(static_cast<std::size_t>(nonzeros));
    result.ipm_kkt_dimension = static_cast<Index>(factor_dimension_);
    result.ipm_factor_nonzeros = static_cast<Index>(factor_dimension_) + nonzeros;
    result.ipm_factor_backend = extended_ ? "qdldl_extended" : "qdldl";
    result.ipm_eliminated_singletons = static_cast<Index>(dimension_ - factor_dimension_);
    check_budget(options, start);
  }

  void factor(const std::vector<double>& h, const SolveOptions& options,
                Clock::time_point start, SolveResult& result, int starting_level = 0,
                bool allow_curvature = true) {
    if (krylov_history_.capacity() > 0 && diagonal_h_ != h) {
      // G 固定而 H 改变时 K0 才改变；同一 H 的正则重试仅更换预条件器。
      check_budget(options, start);
      krylov_history_.clear();
    }
    diagonal_h_ = h;
    regularized_h_.resize(h.size());
    const bool use_curvature = allow_curvature && options.ipm.linear_accuracy == "curvature_regularization";
    for (int trial = use_curvature ? -1 : starting_level; trial < 8; ++trial) {
      check_budget(options, start);
      const bool unshifted = use_curvature && trial == -1;
      const int level = unshifted ? 0 : trial;
      const auto regularizations = detail::ipm_regularization_pair(level,
          options.ipm.linear_accuracy == "primal_dual_regularization");
      const double regularization = unshifted ? 0 : regularizations[0];
      const double dual_regularization = regularizations[1];
      regularization_level_ = level;
      std::vector<long double> primal_diagonal(model_.costs.size(), regularization);
      for (std::size_t row = 0; row < h.size(); ++row) {
        const double denominator = finite(static_cast<long double>(h[row]) + dual_regularization);
        if (!(denominator > 0)) throw NumericalFailure("KKT 消元对角必须严格为正");
        regularized_h_[row] = denominator;
        if (singleton_column_[row] >= 0) {
          const auto col = static_cast<std::size_t>(singleton_column_[row]);
          primal_diagonal[col] += static_cast<long double>(singleton_value_[row]) * singleton_value_[row] / denominator;
        } else {
          ax_[diagonal_[static_cast<std::size_t>(retained_index_[row])]] = -denominator;
        }
      }
      // 仅使用本轮已形成的消元曲率；任何坐标不满足正定条件时跳过额外分解。
      // 对偶偏移保持原起点，保证 D 严格为正；原始零偏移失败后回到旧八级网格。
      if (unshifted && !detail::ipm_has_positive_curvature(primal_diagonal)) continue;
      for (std::size_t col = 0; col < model_.costs.size(); ++col) ax_[diagonal_[col]] = finite(primal_diagonal[col]);
      const double* factor_values = ax_.data();
      if (options.ipm.equilibrate_kkt) {
        // 对本轮实际 Kδ 做一次 Ruiz 对称均衡；原 ax_ 保留，重试不累乘旧尺度。
        // s_i = 1 / sqrt(max_j |Kδ_ij|)，依据 Ruiz 2001，算法 2.1。
        // https://epubs.stfc.ac.uk/manifestation/271/raltr-2001034.pdf
        for (std::size_t i = 0; i < factor_dimension_; ++i) {
          const auto permuted = inverse_[i];
          const double norm = std::max(off_diagonal_max_[permuted], std::abs(ax_[diagonal_[i]]));
          equilibration_[permuted] = norm > 0 ? finite(1 / std::sqrt(static_cast<long double>(norm))) : 1;
          if (!(equilibration_[permuted] > 0)) throw NumericalFailure("KKT 均衡因子必须为有限正数");
        }
        for (std::size_t col = 0; col < factor_dimension_; ++col) {
          check_budget(options, start);
          for (QDLDL_int k = ap_[col]; k < ap_[col + 1]; ++k) {
            const auto position = static_cast<std::size_t>(k);
            factor_values_[position] = symmetric_scaled(ax_[position],
                equilibration_[static_cast<std::size_t>(ai_[position])], equilibration_[col]);
          }
        }
        factor_values = factor_values_.data();
      }
      if (sparse_lu_ || normal_cholesky_ || multifrontal_ldl_) check_budget(options, start);
      ++*result.ipm_factorizations;
      if (unshifted) ++*result.ipm_unshifted_factorizations;
      last_unshifted_ = unshifted;
      result.ipm_primal_regularization = regularization;
      result.ipm_dual_regularization = dual_regularization;
      if (multifrontal_ldl_) {
        const bool success = multifrontal_ldl_->factor(std::span<const double>(factor_values, ax_.size()),
            [&options, &start] { check_budget(options, start); });
        check_budget(options, start);
        *result.ipm_regularized_pivots += multifrontal_ldl_->regularized_pivots();
        if (success) {
          result.ipm_factor_nonzeros = multifrontal_ldl_->factor_nonzeros();
          return;
        }
        continue;
      }
      if (normal_cholesky_) {
        const bool success = normal_cholesky_->factor(std::span<const double>(factor_values, ax_.size()),
            [&options, &start] { check_budget(options, start); });
        check_budget(options, start);
        *result.ipm_regularized_pivots += normal_cholesky_->regularized_pivots();
#ifdef SOLVER_IPM_TRACE
        std::cerr << "正规方程分解 " << *result.ipm_iterations << " 正则 " << regularization
                  << " 成功 " << success << '\n';
#endif
        if (success) {
          result.ipm_factor_nonzeros = normal_cholesky_->factor_nonzeros();
          return;
        }
        continue;
      }
      if (sparse_lu_) {
        const bool success = sparse_lu_->factor(std::span<const double>(factor_values, ax_.size()));
        check_budget(options, start);
#ifdef SOLVER_IPM_TRACE
        std::cerr << "LU 分解 " << *result.ipm_iterations << " 正则 " << regularization
                  << " 成功 " << success << '\n';
#endif
        if (success) {
          result.ipm_factor_nonzeros = sparse_lu_->factor_nonzeros();
          return;
        }
        continue;
      }
      QDLDL_int positives;
      if (extended_) {
        check_budget(options, start);
        std::copy(factor_values, factor_values + ax_.size(), extended_ax_.begin());
        positives = factor_dimension_ == 0 ? 0 : detail::solver_qdldl_extended_factor(
            static_cast<QDLDL_int>(factor_dimension_), ap_.data(), ai_.data(), extended_ax_.data(),
            lp_.data(), li_.data(), extended_lx_.data(), extended_d_.data(), extended_inverse_.data(),
            lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), extended_work_.data());
        check_budget(options, start);
        const auto finite_vector = [](const auto& values) {
          return std::all_of(values.begin(), values.end(), [](long double value) { return std::isfinite(value); });
        };
        if (positives == static_cast<QDLDL_int>(model_.costs.size()) &&
            finite_vector(extended_d_) && finite_vector(extended_inverse_) && finite_vector(extended_lx_)) return;
        continue;
      } else if (options.ipm.dynamic_regularization) {
        const auto factored = detail::factor_regularized_ldl(static_cast<QDLDL_int>(factor_dimension_),
            ap_.data(), ai_.data(), factor_values, lp_.data(), li_.data(), lx_.data(), d_.data(), dinv_.data(),
            lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), fwork_.data(), pivot_signs_.data());
        positives = factored.positive_pivots;
        *result.ipm_regularized_pivots += factored.regularized_pivots;
      } else {
        positives = QDLDL_factor(static_cast<QDLDL_int>(factor_dimension_), ap_.data(), ai_.data(), factor_values,
            lp_.data(), li_.data(), lx_.data(), d_.data(), dinv_.data(), lnz_.data(), etree_.data(),
            bwork_.data(), iwork_.data(), fwork_.data());
      }
#ifdef SOLVER_IPM_TRACE
      std::cerr << "分解 " << *result.ipm_iterations << " 正则 " << regularization
                << " 正惯性 " << positives << " 最大逆主元 " << norm_inf(dinv_)
                << " 最大因子 " << norm_inf(lx_) << '\n';
#endif
      check_budget(options, start);
      if (positives == static_cast<QDLDL_int>(model_.costs.size()) &&
          std::isfinite(norm_inf(dinv_)) && std::isfinite(norm_inf(lx_))) return;
    }
    throw NumericalFailure(multifrontal_ldl_ ? "分块 LDL 分解重试未成功" : normal_cholesky_ ? "正则化正规方程 Cholesky 分解重试未成功" :
        sparse_lu_ ? "正则化 KKT 稀疏 LU 分解重试未成功" :
        "正则化 KKT 分解未得到可靠惯性");
  }

  int regularization_level() const { return regularization_level_; }
  int next_regularization_level() const { return last_unshifted_ ? 0 : regularization_level_ + 1; }
  double last_dual_error() const { return last_dual_error_; }

  std::vector<double> solve(const std::vector<double>& rhs, const SolveOptions& options,
                              Clock::time_point start, SolveResult& result,
                              const InexactDirectionContext* inexact = nullptr,
                              bool initialization = false) const {
    check_budget(options, start);
    auto answer = direct(rhs, options, start, result);
    const double tolerance = 1e-12 + 1e-10 * norm_inf(rhs);
    std::optional<std::array<double, 2>> block_tolerances;
    if ((options.ipm.linear_accuracy == "blockwise" || options.ipm.linear_accuracy == "blockwise_weighted" ||
         options.ipm.linear_accuracy == "blockwise_forcing" ||
         options.ipm.linear_accuracy == "blockwise_forcing_weighted") &&
        !initialization) {
      const double relative = (options.ipm.linear_accuracy == "blockwise_forcing" ||
          options.ipm.linear_accuracy == "blockwise_forcing_weighted") ?
          static_cast<double>(detail::ipm_inexact_forcing) : 1e-10;
      block_tolerances = detail::ipm_kkt_block_tolerances(rhs, model_.costs.size(), relative);
    }
    std::vector<double> residual(rhs.size());
    std::vector<long double> product_cache;
    auto* cached_product = krylov_history_.capacity() > 0 || (inexact && options.ipm.inexact_prefilter) ?
        &product_cache : nullptr;
    double previous = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration <= 10; ++iteration) {
      check_budget(options, start);
      const long double backward_error = assess_residual(rhs, answer, residual, options, start, cached_product);
      const double norm = norm_inf(residual);
#ifdef SOLVER_IPM_TRACE
      std::cerr << "改进 " << *result.ipm_iterations << " 轮 " << iteration
                << " 残差 " << norm << " 阈值 " << tolerance
                << " 相对阈值 " << norm / tolerance << " 后向误差 " << backward_error
                << " 方向 " << norm_inf(answer) << '\n';
#endif
      if (norm <= tolerance && (!block_tolerances ||
          detail::ipm_kkt_blocks_satisfied(residual, model_.costs.size(), *block_tolerances))) {
        remember_dual_error(residual, options);
        remember_recycle_direction(answer, product_cache, options, start, result);
        return answer;
      }
      if (!std::isfinite(norm)) throw NumericalFailure("KKT 迭代改进残差非有限");
      if (accept_backward_error(backward_error, options, start, result)) return answer;
      if (accept_inexact_direction(answer, inexact, options, start, result, cached_product)) return answer;
      if (iteration == 2 || norm >= previous) {
        return krylov_refine(rhs, answer, residual, tolerance, options, start, result, inexact,
            block_tolerances ? &*block_tolerances : nullptr,
            krylov_history_.capacity() > 0 ? &product_cache : nullptr);
      }
      previous = norm;
      const auto correction = direct(residual, options, start, result);
      ++*result.ipm_refinement_steps;
      for (std::size_t i = 0; i < answer.size(); ++i) answer[i] = finite(static_cast<long double>(answer[i]) + correction[i]);
    }
    return answer;
  }

  Point solve_full_newton(const Point& point, const std::vector<double>& primal_residual,
      const std::vector<double>& dual_residual, const std::vector<double>& h,
      const SolveOptions& options, Clock::time_point start, SolveResult& result) const {
    check_budget(options, start);
    const auto layout = detail::make_ipm_full_newton_layout(model_.costs.size(), model_.inequality);
    const auto n = layout.columns, m = layout.rows;
    std::vector<double> rhs(layout.dimension());
    for (std::size_t col = 0; col < n; ++col) rhs[col] = -dual_residual[col];
    for (std::size_t row = 0; row < m; ++row) rhs[n + row] = -primal_residual[row];
    for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) rhs[n + m + i] = h[layout.inequality_rows[i]];
    const auto budget = [&] { check_budget(options, start); };
    const auto precondition = [&](std::span<const double> physical_rhs) {
      budget();
      const auto reduced_rhs = detail::ipm_full_newton_reduced_rhs(layout, point.z, physical_rhs, budget);
      budget();
      ++*result.ipm_full_newton_preconditioner_solves;
      const auto reduced = direct(reduced_rhs, options, start, result);
      budget();
      return detail::ipm_full_newton_recover(layout, point.s, point.z, physical_rhs, reduced, budget);
    };
    const auto multiply = [&](std::span<const double> direction, std::vector<long double>& product) {
      budget();
      ++*result.ipm_full_newton_products;
      detail::ipm_full_newton_product(layout, model_.starts, model_.rows, model_.values,
          point.s, point.z, direction, product, budget);
      budget();
    };
    detail::IpmFullNewtonWork work;
    const auto collect = [&] {
      *result.ipm_refinement_steps += work.refinement_steps;
      *result.ipm_weighted_krylov_steps += work.krylov_steps;
    };
    std::vector<double> solution;
    detail::IpmBoundedNewtonInfo bounded_info;
    const bool bounded = options.ipm.linear_accuracy == "bounded_newton";
    try {
      if (bounded) {
        solution = detail::solve_ipm_bounded_newton(layout, rhs, precondition, multiply, work, bounded_info, budget);
      } else {
        solution = detail::solve_ipm_full_newton(layout, rhs, precondition, multiply, work, budget);
      }
    } catch (...) {
      collect();
      throw;
    }
    collect();
    Point delta;
    delta.x.assign(solution.begin(), solution.begin() + static_cast<std::ptrdiff_t>(n));
    delta.z.assign(solution.begin() + static_cast<std::ptrdiff_t>(n),
        solution.begin() + static_cast<std::ptrdiff_t>(n + m));
    delta.s.assign(m, 0);
    for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i)
      delta.s[layout.inequality_rows[i]] = solution[n + m + i];
    budget();
    if (bounded) {
      ++*result.ipm_bounded_newton_returns;
      if (!bounded_info.tolerance_satisfied) ++*result.ipm_bounded_newton_inexact_returns;
    }
    return delta;
  }

  std::vector<double> solve_homogeneous_direct(const std::vector<double>& rhs,
      const SolveOptions& options, Clock::time_point start, SolveResult& result,
      bool constant, bool refinement = false) const {
    check_budget(options, start);
    if (constant) {
      ++*result.ipm_homogeneous_constant_solves;
    } else {
      ++*result.ipm_homogeneous_direction_solves;
      if (refinement) {
        ++*result.ipm_homogeneous_scaled_refinement_steps;
        ++*result.ipm_refinement_steps;
      }
    }
    // 两项计数在同一个预算检查后对应实际 direct 启动；异常也保留已开始的工作。
    auto answer = direct(rhs, options, start, result);
    check_budget(options, start);
    return answer;
  }

 private:
  void remember_recycle_pair(const std::vector<double>& direction, const std::vector<double>& image,
      const SolveOptions& options, Clock::time_point start, SolveResult& result) const {
    if (krylov_history_.add(direction, image, [&options, start] { check_budget(options, start); })) {
      ++*result.ipm_krylov_recycle_vectors;
      *result.ipm_krylov_recycle_peak_dimension = std::max(*result.ipm_krylov_recycle_peak_dimension,
          static_cast<Index>(krylov_history_.size()));
    }
  }

  void remember_recycle_direction(const std::vector<double>& answer,
      const std::vector<long double>& product, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    if (krylov_history_.capacity() == 0) return;
    check_budget(options, start);
    std::vector<double> image(answer.size());
    for (std::size_t i = 0; i < answer.size(); ++i) {
      const long double value = product[i];
      image[i] = static_cast<double>(value);
      if (!std::isfinite(value) || !std::isfinite(image[i]) || (value != 0 && image[i] == 0)) return;
    }
    remember_recycle_pair(answer, image, options, start, result);
  }

  void remember_recycle_cycle(const std::vector<double>& base, const std::vector<long double>& base_product,
      const std::vector<double>& answer, const std::vector<long double>& product,
      const SolveOptions& options, Clock::time_point start, SolveResult& result) const {
    if (krylov_history_.capacity() == 0) return;
    check_budget(options, start);
    std::vector<double> update(answer.size()), image(answer.size());
    for (std::size_t i = 0; i < answer.size(); ++i) {
      const long double direction = static_cast<long double>(answer[i]) - base[i];
      const long double image_value = product[i] - base_product[i];
      update[i] = static_cast<double>(direction);
      image[i] = static_cast<double>(image_value);
      if (!std::isfinite(direction) || !std::isfinite(image_value) ||
          !std::isfinite(update[i]) || !std::isfinite(image[i]) ||
          (direction != 0 && update[i] == 0) || (image_value != 0 && image[i] == 0)) return;
    }
    remember_recycle_pair(update, image, options, start, result);
  }

  long double recycle_residual_measure(const std::vector<double>& residual, double tolerance,
      const std::array<double, 2>* block_tolerances) const {
    long double measure = 0;
    for (std::size_t i = 0; i < residual.size(); ++i) {
      if (!std::isfinite(residual[i])) return std::numeric_limits<long double>::infinity();
      const double allowed = block_tolerances ?
          std::min(tolerance, (*block_tolerances)[i < model_.costs.size() ? 0 : 1]) : tolerance;
      measure = std::max(measure, std::abs(static_cast<long double>(residual[i])) / allowed);
    }
    return measure;
  }

  void remember_dual_error(const std::vector<double>& residual, const SolveOptions& options) const {
    if (options.ipm.linear_accuracy != "adaptive_regularization") return;
    last_dual_error_ = 0;
    for (std::size_t col = 0; col < model_.costs.size(); ++col) {
      last_dual_error_ = std::max(last_dual_error_, std::abs(residual[col]));
    }
  }

  static double dot(const std::vector<double>& left, const std::vector<double>& right) {
    long double sum = 0;
    for (std::size_t i = 0; i < left.size(); ++i) sum += static_cast<long double>(left[i]) * right[i];
    return finite(sum);
  }

  std::vector<double> krylov_refine(const std::vector<double>& rhs,
      std::vector<double> answer, std::vector<double> residual, double tolerance,
      const SolveOptions& options, Clock::time_point start, SolveResult& result,
      const InexactDirectionContext* inexact, const std::array<double, 2>* block_tolerances,
      const std::vector<long double>* initial_product) const {
    // 以已有正则化 LDL 因子作右预条件器，最小化未正则化方程的残差。
    // 内存由固定 Krylov 窗口约束；最终仍检查真实残差而非 Hessenberg 估计。
    constexpr std::size_t window = 32;
    const auto limit = std::min(window, dimension_);
    // 估计值只决定何时构造真实候选；最终仍分别检查两块实际 Newton 缺陷。
    // 任一块预算为零时回到旧触发阈值；窗口末尾仍会检查候选。
    const bool weighted = block_tolerances && (options.ipm.linear_accuracy == "blockwise_weighted" ||
        options.ipm.linear_accuracy == "blockwise_forcing_weighted");
    std::array<double, 2> search_tolerances{};
    if (weighted) {
      // 搜索尺度对应全局与分块条件的交集；只使用块门槛会忽略更严的全局要求。
      for (std::size_t block = 0; block < 2; ++block) {
        search_tolerances[block] = std::min(tolerance, (*block_tolerances)[block]);
      }
    }
    const double candidate_tolerance = weighted ? 1.0 : inexact ?
        std::max(tolerance, inexact->candidate_tolerance) : tolerance;
    // blockwise 仍用全局估计触发真实候选构造；返回时额外检查两块，估计值不能授权接受。
    std::vector<long double> product_cache = initial_product ? *initial_product : std::vector<long double>{};
    auto* cached_product = krylov_history_.capacity() > 0 || (inexact && options.ipm.inexact_prefilter) ?
        &product_cache : nullptr;
    std::vector<double> zero(dimension_, 0);
    for (int restart = 0; restart < 3; ++restart) {
      check_budget(options, start);
      bool augment = false;
      if (krylov_history_.size() > 0) {
        ++*result.ipm_krylov_recycle_projections;
        std::vector<double> correction, projected_residual;
        if (krylov_history_.project(residual, correction, projected_residual,
            [&options, start] { check_budget(options, start); })) {
          try {
            auto trial = answer;
            for (std::size_t i = 0; i < trial.size(); ++i) {
              trial[i] = finite(static_cast<long double>(trial[i]) + correction[i]);
            }
            std::vector<double> trial_residual(dimension_);
            std::vector<long double> trial_product;
            assess_residual(rhs, trial, trial_residual, options, start, &trial_product);
            check_budget(options, start);
            // 以原验收各门槛的交集比较真实残差；预测投影下降本身不授权采用。
            if (recycle_residual_measure(trial_residual, tolerance, block_tolerances) <
                recycle_residual_measure(residual, tolerance, block_tolerances)) {
              answer = std::move(trial);
              residual = std::move(trial_residual);
              product_cache = std::move(trial_product);
              augment = true;
              if (norm_inf(residual) <= tolerance && (!block_tolerances ||
                  detail::ipm_kkt_blocks_satisfied(residual, model_.costs.size(), *block_tolerances))) {
                ++*result.ipm_krylov_recycle_acceptances;
                remember_dual_error(residual, options);
                return answer;
              }
            }
          } catch (const NumericalFailure&) {
            // 只撤销额外投影的数值失败；预算异常仍由外层处理，旧搜索继续可用。
          }
        }
      }
      const auto cycle_base = krylov_history_.capacity() > 0 ? answer : std::vector<double>{};
      const auto cycle_product = krylov_history_.capacity() > 0 ? product_cache : std::vector<long double>{};
      std::vector<std::vector<double>> basis(1, residual), preconditioned;
      if (weighted) {
        for (std::size_t i = 0; i < dimension_; ++i) {
          basis[0][i] = finite(static_cast<long double>(basis[0][i]) /
              search_tolerances[i < model_.costs.size() ? 0 : 1]);
        }
      }
      const double beta = std::sqrt(dot(basis[0], basis[0]));
      if (!(beta > 0)) throw NumericalFailure("KKT Krylov 初始残差无效");
      for (double& value : basis[0]) value /= beta;
      std::vector<std::vector<double>> upper(limit, std::vector<double>(limit + 1, 0));
      std::vector<double> cosine(limit), sine(limit), transformed(limit + 1, 0);
      transformed[0] = beta;
      std::vector<double> candidate = answer;
      for (std::size_t col = 0; col < limit; ++col) {
        check_budget(options, start);
        if (weighted) {
          // W=diag(1/块门槛)：右预条件器为 Kδ^-1 W^-1，算子为 W K0 Kδ^-1 W^-1。
          // 因子仍解原尺度右端，最终候选仍按完整未缩放 K0 的两块缺陷验收。
          auto unweighted_rhs = basis[col];
          for (std::size_t i = 0; i < dimension_; ++i) {
            unweighted_rhs[i] = finite(static_cast<long double>(unweighted_rhs[i]) *
                search_tolerances[i < model_.costs.size() ? 0 : 1]);
          }
          preconditioned.push_back(direct(unweighted_rhs, options, start, result));
          ++*result.ipm_weighted_krylov_steps;
        } else {
          preconditioned.push_back(direct(basis[col], options, start, result));
        }
        ++*result.ipm_refinement_steps;
        std::vector<double> work(dimension_);
        unregularized_residual(zero, preconditioned.back(), work);
        for (std::size_t i = 0; i < dimension_; ++i) {
          work[i] = weighted ? finite(-static_cast<long double>(work[i]) /
              search_tolerances[i < model_.costs.size() ? 0 : 1]) : -work[i];
        }
        if (augment && !krylov_history_.orthogonalize(preconditioned.back(), work,
            [&options, start] { check_budget(options, start); })) {
          // 事务失败保留本次原 p/K0p，此周期后续沿用普通 Arnoldi。
          augment = false;
        }
        for (int pass = 0; pass < 2; ++pass) {
          for (std::size_t row = 0; row <= col; ++row) {
            const double projection = dot(basis[row], work);
            upper[col][row] += projection;
            for (std::size_t i = 0; i < dimension_; ++i) work[i] -= projection * basis[row][i];
          }
        }
        const double remaining = std::sqrt(dot(work, work));
        upper[col][col + 1] = remaining;
        if (remaining > 0) {
          for (double& value : work) value /= remaining;
          basis.push_back(std::move(work));
        }
        for (std::size_t row = 0; row < col; ++row) {
          const double first = upper[col][row], second = upper[col][row + 1];
          upper[col][row] = cosine[row] * first + sine[row] * second;
          upper[col][row + 1] = -sine[row] * first + cosine[row] * second;
        }
        const double diagonal = std::hypot(upper[col][col], upper[col][col + 1]);
        if (!(diagonal > 0) || !std::isfinite(diagonal)) throw NumericalFailure("KKT Krylov 子空间退化");
        cosine[col] = upper[col][col] / diagonal;
        sine[col] = upper[col][col + 1] / diagonal;
        upper[col][col] = diagonal;
        upper[col][col + 1] = 0;
        transformed[col + 1] = -sine[col] * transformed[col];
        transformed[col] *= cosine[col];
        if (std::abs(transformed[col + 1]) <= candidate_tolerance || col + 1 == limit || remaining == 0) {
          std::vector<double> coefficients(transformed.begin(), transformed.begin() + static_cast<std::ptrdiff_t>(col + 1));
          for (std::size_t row = col + 1; row-- > 0;) {
            for (std::size_t j = row + 1; j <= col; ++j) coefficients[row] -= upper[j][row] * coefficients[j];
            coefficients[row] /= upper[row][row];
          }
          candidate = answer;
          for (std::size_t j = 0; j <= col; ++j) {
            check_budget(options, start);
            for (std::size_t i = 0; i < dimension_; ++i) candidate[i] = finite(static_cast<long double>(candidate[i]) + coefficients[j] * preconditioned[j][i]);
          }
          const long double backward_error = assess_residual(rhs, candidate, residual, options, start, cached_product);
#ifdef SOLVER_IPM_TRACE
          std::cerr << "Krylov " << *result.ipm_iterations << " 重启 " << restart << " 列 " << col + 1
                    << " 残差 " << norm_inf(residual) << " 阈值 " << tolerance
                    << " 相对阈值 " << norm_inf(residual) / tolerance
                    << " 后向误差 " << backward_error << '\n';
#endif
          if (norm_inf(residual) <= tolerance && (!block_tolerances ||
              detail::ipm_kkt_blocks_satisfied(residual, model_.costs.size(), *block_tolerances))) {
            remember_dual_error(residual, options);
            remember_recycle_cycle(cycle_base, cycle_product, candidate, product_cache, options, start, result);
            return candidate;
          }
          if (accept_backward_error(backward_error, options, start, result)) return candidate;
          if (accept_inexact_direction(candidate, inexact, options, start, result, cached_product)) return candidate;
          if (remaining == 0) break;
        }
      }
      // 一个周期内 U/C 固定；只保存完成周期的整体修正，不收集每一列 Arnoldi 向量。
      remember_recycle_cycle(cycle_base, cycle_product, candidate, product_cache, options, start, result);
      answer = std::move(candidate);
    }
    throw NumericalFailure("KKT Krylov 改进未达到原方程精度");
  }

  void solve_factored(std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    if (multifrontal_ldl_) {
      const bool success = multifrontal_ldl_->solve(rhs,
          [&options, &start] { check_budget(options, start); });
      if (!success) throw NumericalFailure("分块 LDL 三角求解失败");
    } else if (normal_cholesky_) {
      check_budget(options, start);
      const auto collect = [&] {
        const auto statistics = normal_cholesky_->refinement_statistics();
        result.ipm_schur_residual_products = statistics.residual_products;
        result.ipm_schur_refinement_steps = statistics.refinement_steps;
        result.ipm_schur_refinement_acceptances = statistics.refinement_acceptances;
      };
      bool success;
      try {
        success = normal_cholesky_->solve(rhs, [&options, &start] { check_budget(options, start); });
      } catch (...) {
        // 中断及数值失败也上报已经启动的内部工作，不吞掉原异常。
        collect();
        throw;
      }
      collect();
      check_budget(options, start);
      if (!success) throw NumericalFailure("正规方程 Cholesky 方向恢复失败");
    } else if (sparse_lu_) {
      check_budget(options, start);
      const bool success = sparse_lu_->solve(rhs);
      check_budget(options, start);
      if (!success) throw NumericalFailure("KKT 稀疏 LU 三角求解失败");
    } else if (extended_) {
      check_budget(options, start);
      std::vector<long double> solution(rhs.begin(), rhs.end());
      detail::solver_qdldl_extended_solve(static_cast<QDLDL_int>(factor_dimension_),
          lp_.data(), li_.data(), extended_lx_.data(), extended_inverse_.data(), solution.data());
      check_budget(options, start);
      std::vector<double> converted(rhs.size());
      for (std::size_t i = 0; i < solution.size(); ++i) {
        converted[i] = finite(solution[i]);
        if (solution[i] != 0 && converted[i] == 0) throw NumericalFailure("扩展精度方向转为 double 时下溢");
      }
      rhs = std::move(converted);
    } else {
      QDLDL_solve(static_cast<QDLDL_int>(factor_dimension_), lp_.data(), li_.data(), lx_.data(), dinv_.data(), rhs.data());
    }
  }

  std::vector<double> direct(const std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) const {
    if (factor_dimension_ == dimension_) {
      std::vector<double> permuted(dimension_);
      // 没有消元时不需要构造扩展精度的约化右端；均衡与置换可在同一次扫描完成。
      for (std::size_t i = 0; i < dimension_; ++i) {
        const auto position = inverse_[i];
        permuted[position] = equilibration_.empty() ? rhs[i] : scaled(rhs[i], equilibration_[position]);
      }
      solve_factored(permuted, options, start, result);
      std::vector<double> answer(dimension_);
      for (std::size_t i = 0; i < dimension_; ++i) {
        const auto position = inverse_[i];
        answer[i] = equilibration_.empty() ? permuted[position] : scaled(permuted[position], equilibration_[position]);
      }
      return answer;
    }
    const auto n = model_.costs.size();
    std::vector<long double> reduced_rhs(factor_dimension_, 0);
    for (std::size_t col = 0; col < n; ++col) reduced_rhs[col] = rhs[col];
    for (std::size_t row = 0; row < model_.rhs.size(); ++row) {
      if (singleton_column_[row] >= 0) {
        const auto col = static_cast<std::size_t>(singleton_column_[row]);
        reduced_rhs[col] += static_cast<long double>(singleton_value_[row]) * rhs[n + row] / regularized_h_[row];
      } else {
        reduced_rhs[static_cast<std::size_t>(retained_index_[row])] = rhs[n + row];
      }
    }
    std::vector<double> permuted(factor_dimension_);
    for (std::size_t i = 0; i < factor_dimension_; ++i) {
      const auto position = inverse_[i];
      const double value = finite(reduced_rhs[i]);
      permuted[position] = equilibration_.empty() ? value : scaled(value, equilibration_[position]);
    }
    solve_factored(permuted, options, start, result);
    if (!equilibration_.empty()) {
      for (std::size_t i = 0; i < factor_dimension_; ++i) permuted[i] = scaled(permuted[i], equilibration_[i]);
    }
    std::vector<double> answer(dimension_);
    for (std::size_t col = 0; col < n; ++col) answer[col] = finite(permuted[inverse_[col]]);
    for (std::size_t row = 0; row < model_.rhs.size(); ++row) {
      if (singleton_column_[row] >= 0) {
        const auto col = static_cast<std::size_t>(singleton_column_[row]);
        // 恢复完整对偶方向，使迭代改进仍直接验算原始 KKT 方程。
        answer[n + row] = finite((static_cast<long double>(singleton_value_[row]) * answer[col] -
                                    rhs[n + row]) / regularized_h_[row]);
      } else {
        answer[n + row] = finite(permuted[inverse_[static_cast<std::size_t>(retained_index_[row])]]);
      }
    }
    return answer;
  }

  void unregularized_residual(const std::vector<double>& rhs, const std::vector<double>& point,
                                std::vector<double>& residual,
                                std::vector<long double>* product_cache = nullptr) const {
    const auto n = model_.costs.size();
    std::vector<long double> workspace;
    auto& product = product_cache ? *product_cache : workspace;
    product.assign(dimension_, 0);
    for (std::size_t row = 0; row < model_.rhs.size(); ++row) product[n + row] = -static_cast<long double>(diagonal_h_[row]) * point[n + row];
    for (std::size_t col = 0; col < n; ++col) {
      // 上块各列互不重叠；在寄存器中按原顺序累加，避免每个非零项回写整段工作区。
      long double column_product = 0;
      const long double column_point = point[col];
      for (Index k = model_.starts[col]; k < model_.starts[col + 1]; ++k) {
        const auto position = static_cast<std::size_t>(k);
        const auto row = n + static_cast<std::size_t>(model_.rows[position]);
        column_product += static_cast<long double>(model_.values[position]) * point[row];
        product[row] += static_cast<long double>(model_.values[position]) * column_point;
      }
      product[col] = column_product;
    }
    for (std::size_t i = 0; i < dimension_; ++i) residual[i] = finite(static_cast<long double>(rhs[i]) - product[i]);
  }

  long double assess_residual(const std::vector<double>& rhs, const std::vector<double>& point,
      std::vector<double>& residual, const SolveOptions& options, Clock::time_point start,
      std::vector<long double>* product_cache = nullptr) const {
    if (options.ipm.linear_accuracy != "backward") {
      // 默认路径保持原累加顺序，不分配或计算绝对乘积和；Arnoldi 也直接走原算子。
      unregularized_residual(rhs, point, residual, product_cache);
      return std::numeric_limits<long double>::infinity();
    }
    return detail::assess_ipm_kkt_residual(model_.starts, model_.rows, model_.values,
        diagonal_h_, rhs, point, backward_error_nonzeros_, residual,
        [&options, &start] { check_budget(options, start); });
  }

  static bool accept_backward_error(long double error, const SolveOptions& options,
      Clock::time_point start, SolveResult& result) {
    if (options.ipm.linear_accuracy != "backward" || !std::isfinite(error) ||
        error > detail::ipm_backward_error_tolerance) return false;
    check_budget(options, start);
    ++*result.ipm_backward_error_acceptances;
    return true;
  }

  bool accept_inexact_direction(const std::vector<double>& answer,
      const InexactDirectionContext* context, const SolveOptions& options,
      Clock::time_point start, SolveResult& result,
      const std::vector<long double>* product_cache) const {
    if (!context) return false;
    check_budget(options, start);
    if (product_cache) {
      const auto n = model_.costs.size();
      const bool rejected = detail::reject_ipm_inexact_dual(inexact_prefilter_,
          std::span<const long double>(*product_cache).first(n), context->dual_residual,
          std::span<const double>(answer).subspan(n),
          [&options, &start] { check_budget(options, start); });
      check_budget(options, start);
      if (rejected) {
        ++*result.ipm_inexact_prefilter_rejections;
        return false;
      }
    }
    try {
      const auto slack = recover_slack_direction(model_, context->point, context->h, answer);
      const bool accepted = detail::accept_ipm_inexact_direction(model_.starts, model_.rows,
          model_.values, context->primal_residual, context->dual_residual, slack, answer,
          [&options, &start] { check_budget(options, start); });
      check_budget(options, start);
      if (!accepted) return false;
      ++*result.ipm_inexact_acceptances;
      return true;
    } catch (const NumericalFailure&) {
      // 可选方向不能表示时继续原有改进；预算异常必须传到外层。
      check_budget(options, start);
      return false;
    }
  }

  const ConeModel& model_;
  std::size_t dimension_, factor_dimension_;
  mutable detail::IpmKrylovHistory krylov_history_;
  std::size_t backward_error_nonzeros_ = 0;
  std::vector<std::size_t> inverse_, diagonal_;
  std::vector<Index> retained_index_, singleton_column_;
  std::vector<std::int8_t> pivot_signs_;
  std::vector<QDLDL_int> ap_, ai_, lp_, li_, etree_, lnz_, iwork_;
  std::vector<QDLDL_bool> bwork_;
  std::vector<double> ax_, lx_, d_, dinv_, fwork_, diagonal_h_, regularized_h_, singleton_value_;
  std::vector<double> off_diagonal_max_, equilibration_, factor_values_;
  bool extended_ = false;
  int regularization_level_ = 0;
  bool last_unshifted_ = false;
  mutable double last_dual_error_ = 0;
  std::vector<long double> extended_ax_, extended_lx_, extended_d_, extended_inverse_, extended_work_;
  std::unique_ptr<detail::SparseLu> sparse_lu_;
  std::unique_ptr<detail::NormalCholesky> normal_cholesky_;
  std::unique_ptr<detail::MultifrontalLdl> multifrontal_ldl_;
  detail::IpmInexactPrefilter inexact_prefilter_;
};

bool assess(const ConeModel& model, const Point& point, const SolveOptions& options,
               Clock::time_point solve_start, SolveResult& result) {
  std::vector<double> x = model.fixed_values.empty() ? std::vector<double>(point.x.size()) : model.fixed_values;
  std::vector<double> y(result.model.row_lower.size(), 0);
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const auto original_col = model.original_columns.empty() ? col : model.original_columns[col];
    x[original_col] = scaled(point.x[col], model.column_scale[col]);
  }
  std::vector<long double> dual(y.size(), 0);
  for (std::size_t row = 0; row < point.z.size(); ++row) {
    if (model.original_row[row] < 0) continue;
    dual[static_cast<std::size_t>(model.original_row[row])] +=
        static_cast<long double>(point.z[row]) * model.dual_sign[row] *
        model.row_scale[row] * model.objective_scale;
  }
  for (const auto& recovery : model.presolved_bounds) {
    check_budget(options, solve_start);
    const auto row = static_cast<std::size_t>(recovery.cone_row);
    const long double multiplier = static_cast<long double>(point.z[row]) * recovery.sign *
        model.row_scale[row] * model.objective_scale;
    if (!std::isfinite(multiplier)) throw NumericalFailure("归并界乘子恢复出现非有限数");
    if (multiplier == 0) continue;
    // 合并为等式时，按自由乘子的符号选择来源，不假设最优基或近似活跃界。
    const auto source = multiplier > 0 ? recovery.lower : recovery.upper;
    if (source.row < 0) continue;
    const long double contribution = multiplier / source.coefficient;
    if (!std::isfinite(contribution) || contribution == 0) {
      throw NumericalFailure("归并行对偶恢复超出可表示范围");
    }
    dual[static_cast<std::size_t>(source.row)] += contribution;
  }
  for (const auto& recovery : model.presolved_fixed_columns) {
    check_budget(options, solve_start);
    const auto col = recovery.original_column;
    const double sense = result.model.sense == ObjectiveSense::minimize ? 1 : -1;
    long double multiplier = static_cast<long double>(sense) * result.model.costs[col];
    for (Index k = result.model.column_start[col]; k < result.model.column_start[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      const auto row = static_cast<std::size_t>(result.model.row_index[position]);
      const long double product = static_cast<long double>(result.model.values[position]) * dual[row];
      if (!std::isfinite(product) || (result.model.values[position] != 0 && dual[row] != 0 && product == 0)) {
        throw NumericalFailure("新固定列乘子恢复超出可表示范围");
      }
      multiplier -= product;
      if (!std::isfinite(multiplier)) throw NumericalFailure("新固定列乘子恢复出现非有限数");
    }
    if (multiplier == 0) continue;
    // 本层来源行仅含本列及已消除的输入固定列，故各新固定列的恢复互不依赖。
    const auto source = multiplier > 0 ? recovery.lower : recovery.upper;
    if (source.row < 0) continue;
    const long double contribution = multiplier / source.coefficient;
    if (!std::isfinite(contribution) || contribution == 0) {
      throw NumericalFailure("新固定列原行对偶恢复超出可表示范围");
    }
    dual[static_cast<std::size_t>(source.row)] += contribution;
  }
  for (std::size_t row = 0; row < y.size(); ++row) {
    y[row] = finite(dual[row]);
    if ((!model.presolved_bounds.empty() || !model.presolved_fixed_columns.empty()) &&
        dual[row] != 0 && y[row] == 0) {
      throw NumericalFailure("归并行对偶转换导致非零数据下溢");
    }
  }
  const auto start = Clock::now();
  auto checks = verify_lp_solution(result.model, x, y, options.tolerances, options.pdlp.optimality);
  result.verification_seconds += elapsed(start);
  const bool feasible = checks.primal.valid_input && checks.primal.feasible &&
      checks.primal.objective_consistent && checks.primal.objective.has_value();
  bool replace = !result.has_solution || checks.optimality.numerically_optimal;
  if (feasible && result.has_solution) {
    const double sense = result.model.sense == ObjectiveSense::minimize ? 1 : -1;
    replace = replace || sense * *checks.primal.objective <= sense * *result.objective;
  }
  if (replace) {
    result.verification = std::move(checks.primal);
    result.lp_optimality = std::move(checks.optimality);
    result.row_dual = std::move(y);
    result.has_solution = feasible;
    result.solution = feasible ? std::move(x) : std::vector<double>{};
    result.objective = feasible ? result.verification.objective : std::nullopt;
    result.best_bound = result.lp_optimality->corrected_dual_bound;
    result.absolute_gap.reset();
    result.relative_gap.reset();
    result.verification_level = feasible ? "primal_checked" : "residuals_checked";
    if (result.objective && result.best_bound) {
      const double sense = result.model.sense == ObjectiveSense::minimize ? 1 : -1;
      const long double gap = sense * (static_cast<long double>(*result.objective) - *result.best_bound);
      const long double scale = std::max({1.0L, std::abs(static_cast<long double>(*result.objective)),
                                         std::abs(static_cast<long double>(*result.best_bound))});
      const long double tolerance = options.pdlp.optimality.gap_absolute + options.pdlp.optimality.gap_relative * scale;
      if (!std::isfinite(gap) || gap < -tolerance) result.best_bound.reset();
      else {
        const long double nonnegative = std::max(0.0L, gap);
        if (std::isfinite(static_cast<double>(nonnegative))) result.absolute_gap = static_cast<double>(nonnegative);
        result.relative_gap = finite(nonnegative / scale);
      }
    }
  }
  if (result.has_solution && result.lp_optimality && result.lp_optimality->numerically_optimal) {
    result.status = "optimal";
    result.verification_level = "primal_dual_residuals_checked";
    return true;
  }
  return false;
}

Point direction(const ConeModel& model, const Point& point,
                   const std::vector<double>& primal_residual,
                   const std::vector<double>& dual_residual,
                   const std::vector<double>& h,
                   const KktFactor& factor, const SolveOptions& options,
                   Clock::time_point start, SolveResult& result) {
  if (options.ipm.linear_accuracy == "full_newton" || options.ipm.linear_accuracy == "bounded_newton") {
    return factor.solve_full_newton(point, primal_residual, dual_residual, h, options, start, result);
  }
  const auto n = point.x.size();
  std::vector<double> rhs(n + point.s.size());
  for (std::size_t col = 0; col < n; ++col) rhs[col] = -dual_residual[col];
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    rhs[n + row] = finite(-static_cast<long double>(primal_residual[row]) -
        (model.inequality[row] ? static_cast<long double>(h[row]) / point.z[row] : 0));
  }
  std::optional<InexactDirectionContext> inexact;
  if (options.ipm.linear_accuracy == "inexact") {
    const double threshold = static_cast<double>(detail::ipm_inexact_forcing *
        std::min(norm_inf(primal_residual), norm_inf(dual_residual)));
    inexact.emplace(InexactDirectionContext{point, primal_residual, dual_residual, h, threshold});
  }
  const auto solution = factor.solve(rhs, options, start, result, inexact ? &*inexact : nullptr);
  Point delta;
  delta.x.assign(solution.begin(), solution.begin() + static_cast<std::ptrdiff_t>(n));
  delta.z.assign(solution.begin() + static_cast<std::ptrdiff_t>(n), solution.end());
  delta.s = recover_slack_direction(model, point, h, solution);
  return delta;
}

double boundary_step(const ConeModel& model, const std::vector<double>& point,
                       const std::vector<double>& direction) {
  double step = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < point.size(); ++row) {
    if (model.inequality[row] && direction[row] < 0) {
      step = std::min(step, -point[row] / direction[row]);
    }
  }
  return step;
}

void central_correction(const ConeModel& model, const Point& point, Point& delta,
    double sigma, double mu, const KktFactor& factor, const SolveOptions& options,
    Clock::time_point start, SolveResult& result) {
  check_budget(options, start);
  if (model.barrier_rows == 0) return;
  try {
    const double target_mu = finite(static_cast<long double>(sigma) * mu);
    if (!(target_mu > 0)) return;
    const auto budget = [&options, &start] { check_budget(options, start); };
    const auto trial = detail::ipm_centrality_target(point.s, point.z, delta.s, delta.z,
        model.inequality, target_mu, budget);
    const bool nonzero_target = std::any_of(trial.target.begin(), trial.target.end(), [](double value) { return value != 0; });
    check_budget(options, start);
    if (!nonzero_target) return;
    const std::vector<double> zero_primal(point.s.size(), 0), zero_dual(point.x.size(), 0);
    check_budget(options, start);
    ++*result.ipm_central_correction_attempts;
    // 纯中心校正只改变互补方程，不重复添加原始或对偶残差；沿用同一 H 与分解。
    const auto correction = direction(model, point, zero_primal, zero_dual, trial.target,
        factor, options, start, result);
    const bool zero_correction = norm_inf(correction.x) == 0 && norm_inf(correction.s) == 0 && norm_inf(correction.z) == 0;
    check_budget(options, start);
    if (zero_correction) return;
    const auto weights = detail::ipm_centrality_weights(point.s, point.z, delta.s, delta.z,
        correction.s, correction.z, model.inequality, trial.primal_step, trial.dual_step, budget);
    if (weights.primal_weight == 0 && weights.dual_weight == 0) return;
    Point candidate = delta;
    if (weights.primal_weight != 0) {
      check_budget(options, start);
      for (std::size_t col = 0; col < delta.x.size(); ++col) {
        candidate.x[col] = finite(static_cast<long double>(delta.x[col]) + weights.primal_weight * correction.x[col]);
      }
      for (std::size_t row = 0; row < delta.s.size(); ++row) {
        candidate.s[row] = finite(static_cast<long double>(delta.s[row]) + weights.primal_weight * correction.s[row]);
      }
    }
    if (weights.dual_weight != 0) {
      check_budget(options, start);
      for (std::size_t row = 0; row < delta.z.size(); ++row) {
        candidate.z[row] = finite(static_cast<long double>(delta.z[row]) + weights.dual_weight * correction.z[row]);
      }
    }
    check_budget(options, start);
    delta = std::move(candidate);
    ++*result.ipm_central_correction_acceptances;
  } catch (const NumericalFailure&) {
    // 可选方向失败时保留原 Mehrotra 方向；预算异常继续传递，不在此吞掉。
    check_budget(options, start);
  }
}

bool retry_direction_factor(const std::vector<double>& diagonal, KktFactor& factor,
                            const SolveOptions& options, Clock::time_point start,
                            SolveResult& result, int& retries) {
  const int next_level = factor.next_regularization_level();
  if (retries >= options.ipm.direction_retries || next_level >= 8) return false;
  check_budget(options, start);
  ++retries;
  ++*result.ipm_direction_retries;
  // 跳过本次矩阵已经分解或求方向失败的层级；不再尝试零原始偏移。
  factor.factor(diagonal, options, start, result, next_level, false);
  return true;
}

void initialize(const ConeModel& model, Point& point, KktFactor& factor,
                  const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  const auto n = point.x.size();
  const auto m = point.s.size();
  // 同一个增广矩阵分别求原始最小残差点和最小范数对偶点。
  const std::vector<double> diagonal(m, 1);
  factor.factor(diagonal, options, start, result);
  std::vector<double> rhs(n + m, 0);
  std::vector<double> primal, dual;
  int retries = 0;
  for (;;) {
    try {
      std::copy(model.rhs.begin(), model.rhs.end(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
      // 两个辅助投影都沿用原 strict 门槛；重试时必须重求两者。
      primal = factor.solve(rhs, options, start, result, nullptr, true);
      if (options.ipm.direction_retries == 0) {
        // 保留关闭重试时的旧失败轨迹；启用后不在双求解完成前修改初始点。
        std::copy(primal.begin(), primal.begin() + static_cast<std::ptrdiff_t>(n), point.x.begin());
      }
      std::fill(rhs.begin(), rhs.end(), 0);
      for (std::size_t col = 0; col < n; ++col) rhs[col] = -model.costs[col];
      dual = factor.solve(rhs, options, start, result, nullptr, true);
      break;
    } catch (const NumericalFailure&) {
      if (!retry_direction_factor(diagonal, factor, options, start, result, retries)) throw;
      std::fill(rhs.begin(), rhs.end(), 0);
    }
  }
  if (retries > 0) ++*result.ipm_direction_retry_successes;
  if (options.ipm.direction_retries > 0) {
    std::copy(primal.begin(), primal.begin() + static_cast<std::ptrdiff_t>(n), point.x.begin());
  }
  double minimum_s = 0, minimum_z = 0;
  for (std::size_t row = 0; row < m; ++row) {
    point.s[row] = model.inequality[row] ? -primal[n + row] : 0;
    point.z[row] = dual[n + row];
    if (model.inequality[row]) {
      minimum_s = std::min(minimum_s, point.s[row]);
      minimum_z = std::min(minimum_z, point.z[row]);
    }
  }
  for (std::size_t row = 0; row < m; ++row) {
    if (model.inequality[row]) {
      // 分两次相加，使极大负初值的平移仍留下严格正的余量。
      point.s[row] = finite(static_cast<long double>(point.s[row]) - minimum_s);
      point.s[row] = finite(static_cast<long double>(point.s[row]) + 1);
      point.z[row] = finite(static_cast<long double>(point.z[row]) - minimum_z);
      point.z[row] = finite(static_cast<long double>(point.z[row]) + 1);
    }
  }
  check_budget(options, start);
}

void run_ipm(const ConeModel& model, const SolveOptions& options,
                Clock::time_point start, SolveResult& result) {
  Point point;
  point.x.resize(model.costs.size());
  point.s.resize(model.rhs.size());
  point.z.resize(model.rhs.size());
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const auto original_col = model.original_columns.empty() ? col : model.original_columns[col];
    point.x[col] = finite(static_cast<long double>(std::clamp(0.0, result.model.column_lower[original_col],
        result.model.column_upper[original_col])) / model.column_scale[col]);
  }
  std::vector<double> activity;
  model.multiply(point.x, activity);
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    point.s[row] = model.inequality[row] ? std::max(1.0, finite(static_cast<long double>(model.rhs[row]) - activity[row])) : 0;
    point.z[row] = model.inequality[row] ? 1 : 0;
  }
  check_budget(options, start);
  if (model.costs.empty() && !model.presolved_fixed_columns.empty()) {
    // 全部列已可靠代入时先用零行对偶验算，仍由原模型检查可行性和固定列的恢复。
    std::fill(point.z.begin(), point.z.end(), 0);
    if (assess(model, point, options, start, result)) return;
    for (std::size_t row = 0; row < point.z.size(); ++row) {
      point.z[row] = model.inequality[row] ? 1 : 0;
    }
  }
  if (assess(model, point, options, start, result)) return;
  KktFactor factor(model, options, start, result);
  std::vector<double> primal_residual, dual_residual;
  std::vector<double> diagonal(point.s.size()), h(point.s.size());
  result.status = "iteration_limit";
  try {
    initialize(model, point, factor, options, start, result);
    if (assess(model, point, options, start, result)) return;
    int preferred_regularization_level = 0;
    for (std::int64_t iteration = 1; iteration <= options.ipm.max_iterations; ++iteration) {
      check_budget(options, start);
      model.multiply(point.x, primal_residual);
      model.transpose(point.z, dual_residual);
      for (std::size_t col = 0; col < dual_residual.size(); ++col) dual_residual[col] = finite(static_cast<long double>(dual_residual[col]) + model.costs[col]);
      long double complementarity = 0;
      for (std::size_t row = 0; row < point.s.size(); ++row) {
        primal_residual[row] = finite(static_cast<long double>(primal_residual[row]) + point.s[row] - model.rhs[row]);
        if (model.inequality[row]) {
          if (!(point.s[row] > 0) || !(point.z[row] > 0)) throw NumericalFailure("内点候选离开正锥内部");
          diagonal[row] = finite(static_cast<long double>(point.s[row]) / point.z[row]);
          h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row]);
          complementarity -= h[row];
        } else { diagonal[row] = 0; h[row] = 0; }
      }
      const double mu = model.barrier_rows > 0 ? finite(complementarity / model.barrier_rows) : 0;
#ifdef SOLVER_IPM_TRACE
      std::cerr << "迭代 " << iteration << " 均互补 " << mu
                << " 原始残差 " << norm_inf(primal_residual) << " 对偶残差 " << norm_inf(dual_residual)
                << " 最大松弛 " << norm_inf(point.s) << " 最大对偶 " << norm_inf(point.z)
                << " 最大对角 " << norm_inf(diagonal) << '\n';
#endif
      factor.factor(diagonal, options, start, result, preferred_regularization_level);
      // 仅在启用重试时保存预测右端，避免把已覆盖的校正右端用于下一次预测。
      const auto affine_h = options.ipm.direction_retries > 0 ? h : std::vector<double>{};
      Point corrected;
      double sigma = 0;
      int retries = 0;
      for (;;) {
        try {
          auto affine = direction(model, point, primal_residual, dual_residual, h,
                                  factor, options, start, result);
          if (options.ipm.linear_accuracy == "adaptive_regularization") {
            check_budget(options, start);
            const int old_level = factor.regularization_level();
            const int search_level = detail::ipm_regularization_search_level(
                preferred_regularization_level, old_level, factor.last_dual_error(), dual_residual, affine.x);
            if (search_level < old_level) {
              // 每个外层最多一次降正则搜索；全部重分解与重求方向均使用原预算并累计计数。
              ++*result.ipm_regularization_searches;
              factor.factor(diagonal, options, start, result, search_level);
              if (factor.regularization_level() < old_level) {
                ++*result.ipm_regularization_reductions;
                preferred_regularization_level = factor.regularization_level();
              }
              affine = direction(model, point, primal_residual, dual_residual, h, factor, options, start, result);
            }
          }
          const double affine_primal = std::min(1.0, boundary_step(model, point.s, affine.s));
          const double affine_dual = std::min(1.0, boundary_step(model, point.z, affine.z));
          long double affine_product = 0;
          for (std::size_t row = 0; row < point.s.size(); ++row) {
            if (model.inequality[row]) {
              affine_product += (static_cast<long double>(point.s[row]) + affine_primal * affine.s[row]) *
                                (static_cast<long double>(point.z[row]) + affine_dual * affine.z[row]);
            }
          }
          const double affine_mu = model.barrier_rows > 0 ? finite(affine_product / model.barrier_rows) : 0;
          const double ratio = mu > 0 ? std::clamp(affine_mu / mu, 0.0, 1.0) : 0;
          sigma = ratio * ratio * ratio;
          for (std::size_t row = 0; row < point.s.size(); ++row) {
            if (model.inequality[row]) {
              h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] + sigma * mu -
                                 static_cast<long double>(affine.s[row]) * affine.z[row]);
            }
          }
          // 预测和校正只更新右端项，共享同一次数值分解。
          corrected = direction(model, point, primal_residual, dual_residual, h,
                                            factor, options, start, result);
          break;
        } catch (const NumericalFailure&) {
          if (!retry_direction_factor(diagonal, factor, options, start, result, retries)) throw;
          h = affine_h;
        }
      }
      if (retries > 0) ++*result.ipm_direction_retry_successes;
      if (options.ipm.central_corrections == 1) {
        central_correction(model, point, corrected, sigma, mu, factor, options, start, result);
      }
      const double primal_step = std::min(1.0, 0.995 * boundary_step(model, point.s, corrected.s));
      const double dual_step = std::min(1.0, 0.995 * boundary_step(model, point.z, corrected.z));
#ifdef SOLVER_IPM_TRACE
      std::cerr << "步长 " << primal_step << ' ' << dual_step << " 中心权重 " << sigma << '\n';
#endif
      for (std::size_t col = 0; col < point.x.size(); ++col) point.x[col] = finite(static_cast<long double>(point.x[col]) + primal_step * corrected.x[col]);
      for (std::size_t row = 0; row < point.s.size(); ++row) {
        point.s[row] = finite(static_cast<long double>(point.s[row]) + primal_step * corrected.s[row]);
        point.z[row] = finite(static_cast<long double>(point.z[row]) + dual_step * corrected.z[row]);
      }
      result.ipm_iterations = iteration;
      if (assess(model, point, options, start, result)) return;
      if (iteration == options.ipm.max_iterations) break;
    }
  } catch (const BudgetExpired&) {
    result.status = "time_limit";
    result.message = "内点求解已达到时间预算";
  } catch (const NumericalFailure& error) {
    result.status = "numerical_error";
    result.message = error.what();
  }
  assess(model, point, options, start, result);
}

struct HomogeneousDirection {
  Point point;
  detail::IpmHomogeneousScalarDirection scalars;
};

bool assess_homogeneous(const ConeModel& model, const Point& point, double tau,
    const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  check_budget(options, start);
  if (!(tau > 0) || !std::isfinite(tau)) throw NumericalFailure("齐次解提取要求有限正 tau");
  Point extracted;
  const auto divide = [&](const std::vector<double>& source, std::vector<double>& target) {
    target.resize(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      if ((i & 4095) == 0) check_budget(options, start);
      target[i] = finite(static_cast<long double>(source[i]) / tau);
    }
  };
  // 先提取工作坐标，再由原验算恢复固定列、缩放和原行对偶；固定常量不参与除 tau。
  divide(point.x, extracted.x);
  divide(point.s, extracted.s);
  divide(point.z, extracted.z);
  check_budget(options, start);
  const bool optimal = assess(model, extracted, options, start, result);
  check_budget(options, start);
  return optimal;
}

HomogeneousDirection homogeneous_direction(const ConeModel& model, const Point& point,
    double tau, double kappa, const std::vector<double>& primal_residual,
    const std::vector<double>& dual_residual, long double gap_residual, double theta,
    const std::vector<double>& h, long double h_tau, std::vector<double>& constant,
    const KktFactor& factor, const SolveOptions& options, Clock::time_point start, SolveResult& result,
    detail::IpmHomogeneousRepairInfo* repair_info = nullptr) {
  check_budget(options, start);
  const auto n = point.x.size(), m = point.s.size();
  std::vector<double> rhs(n + m);
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0) check_budget(options, start);
    rhs[col] = finite(-static_cast<long double>(theta) * dual_residual[col]);
  }
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) check_budget(options, start);
    rhs[n + row] = finite(-static_cast<long double>(theta) * primal_residual[row] -
        (model.inequality[row] ? static_cast<long double>(h[row]) / point.z[row] : 0));
  }
  check_budget(options, start);
  ++*result.ipm_homogeneous_direction_solves;
  const auto u = factor.solve(rhs, options, start, result);
  const auto budget = [&] { check_budget(options, start); };
  HomogeneousDirection direction;
  if (repair_info) {
    const auto repair = [&] {
      budget();
      auto recovery_options = options;
      recovery_options.ipm.linear_accuracy = "blockwise";
      std::vector<double> constant_rhs(n + m);
      for (std::size_t col = 0; col < n; ++col) {
        if ((col & 4095) == 0) budget();
        constant_rhs[col] = -model.costs[col];
      }
      std::copy(model.rhs.begin(), model.rhs.end(), constant_rhs.begin() + static_cast<std::ptrdiff_t>(n));
      budget();
      ++*result.ipm_homogeneous_constant_repairs;
      ++*result.ipm_homogeneous_constant_solves;
      // 复用同一因子，原全局门槛与两块门槛同时满足才返回；不重建分解。
      return factor.solve(constant_rhs, recovery_options, start, result);
    };
    const bool previously_succeeded = repair_info->succeeded;
    direction.scalars = detail::ipm_homogeneous_scalar_direction_repair(model.costs, model.rhs, u, constant,
        tau, kappa, -static_cast<long double>(theta) * gap_residual, h_tau, repair, *repair_info, budget);
    if (!previously_succeeded && repair_info->succeeded) ++*result.ipm_homogeneous_constant_repair_successes;
  } else {
    direction.scalars = detail::ipm_homogeneous_scalar_direction(model.costs, model.rhs, u, constant,
        tau, kappa, -static_cast<long double>(theta) * gap_residual, h_tau, budget);
  }
  direction.point.x.resize(n);
  direction.point.z.resize(m);
  direction.point.s.assign(m, 0);
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0) budget();
    direction.point.x[col] = finite(static_cast<long double>(u[col]) +
        static_cast<long double>(constant[col]) * direction.scalars.tau);
  }
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) budget();
    direction.point.z[row] = finite(static_cast<long double>(u[n + row]) +
        static_cast<long double>(constant[n + row]) * direction.scalars.tau);
    if (model.inequality[row]) {
      direction.point.s[row] = finite((static_cast<long double>(h[row]) -
          static_cast<long double>(point.s[row]) * direction.point.z[row]) / point.z[row]);
    }
  }
  budget();
  return direction;
}

HomogeneousDirection homogeneous_scaled_direction(const ConeModel& model, const Point& point,
    double tau, double kappa, const std::vector<double>& primal_residual,
    const std::vector<double>& dual_residual, long double gap_residual, double theta,
    const std::vector<double>& h, long double h_tau, const std::vector<double>& constant,
    const detail::IpmHomogeneousScaling& scaling, long double denominator,
    const KktFactor& factor, const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  const auto budget = [&] { check_budget(options, start); };
  budget();
  const auto n = point.x.size(), m = point.s.size(), gap = n + m;
  std::vector<double> rhs(scaling.dimension());
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0) budget();
    rhs[col] = finite(-static_cast<long double>(theta) * dual_residual[col]);
  }
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) budget();
    rhs[n + row] = finite(-static_cast<long double>(theta) * primal_residual[row]);
  }
  rhs[gap] = finite(-static_cast<long double>(theta) * gap_residual / scaling.d);
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) budget();
    rhs[gap + 1 + i] = finite(static_cast<long double>(h[scaling.layout.inequality_rows[i]]) / scaling.lambda[i]);
  }
  rhs.back() = finite(h_tau / scaling.lambda_tau);
  const auto precondition = [&](std::span<const double> input, bool refinement) {
    const auto reduced_rhs = detail::ipm_homogeneous_scaled_reduced_rhs(scaling, input, budget);
    const auto reduced = factor.solve_homogeneous_direct(reduced_rhs, options, start, result, false, refinement);
    return detail::ipm_homogeneous_scaled_recover(scaling, model.costs, model.rhs,
        input, reduced, constant, denominator, budget);
  };
  const auto multiply = [&](std::span<const double> direction, std::vector<long double>& product) {
    budget();
    ++*result.ipm_homogeneous_scaled_products;
    detail::ipm_homogeneous_scaled_product(scaling, model.starts, model.rows, model.values,
        model.costs, model.rhs, point.s, point.z, tau, kappa, direction, product, budget);
    budget();
  };
  detail::IpmBoundedNewtonInfo info;
  const auto solution = detail::solve_ipm_homogeneous_scaled(scaling, rhs, precondition, multiply, info, budget);
  HomogeneousDirection direction;
  direction.point.x.assign(solution.begin(), solution.begin() + static_cast<std::ptrdiff_t>(n));
  direction.point.z.assign(solution.begin() + static_cast<std::ptrdiff_t>(n),
      solution.begin() + static_cast<std::ptrdiff_t>(n + m));
  direction.point.s.assign(m, 0);
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) budget();
    direction.point.s[scaling.layout.inequality_rows[i]] = solution[n + m + i];
  }
  direction.scalars.tau = solution[solution.size() - 2];
  direction.scalars.kappa = solution.back();
  budget();
  if (!info.tolerance_satisfied) ++*result.ipm_homogeneous_scaled_inexact_returns;
  return direction;
}

double homogeneous_step(const ConeModel& model, const Point& point, double tau, double kappa,
    const HomogeneousDirection& direction, const SolveOptions& options, Clock::time_point start) {
  check_budget(options, start);
  const double cone_limit = std::min(boundary_step(model, point.s, direction.point.s),
      boundary_step(model, point.z, direction.point.z));
  return detail::ipm_homogeneous_step_limit(cone_limit, tau, kappa, direction.scalars,
      [&] { check_budget(options, start); });
}

void run_homogeneous_ipm(const ConeModel& model, const SolveOptions& options,
    Clock::time_point start, SolveResult& result) {
  const auto n = model.costs.size(), m = model.rhs.size();
  Point point;
  point.x.resize(n);
  point.s.resize(m);
  point.z.resize(m);
  for (std::size_t col = 0; col < n; ++col) {
    if ((col & 4095) == 0) check_budget(options, start);
    const auto original_col = model.original_columns.empty() ? col : model.original_columns[col];
    point.x[col] = finite(static_cast<long double>(std::clamp(0.0, result.model.column_lower[original_col],
        result.model.column_upper[original_col])) / model.column_scale[col]);
  }
  std::vector<double> activity;
  model.multiply(point.x, activity);
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) check_budget(options, start);
    point.s[row] = model.inequality[row] ? std::max(1.0, finite(static_cast<long double>(model.rhs[row]) - activity[row])) : 0;
    point.z[row] = model.inequality[row] ? 1 : 0;
  }
  // 尚未进入齐次初始化的早期最优点不伪造 tau、kappa 统计。
  if (model.costs.empty() && !model.presolved_fixed_columns.empty()) {
    std::fill(point.z.begin(), point.z.end(), 0);
    if (assess_homogeneous(model, point, 1, options, start, result)) return;
    for (std::size_t row = 0; row < m; ++row) point.z[row] = model.inequality[row] ? 1 : 0;
  }
  if (assess_homogeneous(model, point, 1, options, start, result)) return;
  result.status = "iteration_limit";
  KktFactor factor(model, options, start, result);
  initialize(model, point, factor, options, start, result);
  double tau = 1, kappa = 1;
  check_budget(options, start);
  result.ipm_homogeneous_tau = tau;
  result.ipm_homogeneous_kappa = kappa;
  if (assess_homogeneous(model, point, tau, options, start, result)) return;
  std::vector<double> primal_residual, dual_residual, diagonal(m), h(m), constant_rhs(n + m);
  for (std::size_t col = 0; col < n; ++col) constant_rhs[col] = -model.costs[col];
  std::copy(model.rhs.begin(), model.rhs.end(), constant_rhs.begin() + static_cast<std::ptrdiff_t>(n));
  const bool scaled_direction = options.ipm.linear_accuracy == "homogeneous_scaled";
  for (std::int64_t iteration = 1; iteration <= options.ipm.max_iterations; ++iteration) {
    check_budget(options, start);
    model.multiply(point.x, primal_residual);
    check_budget(options, start);
    model.transpose(point.z, dual_residual);
    check_budget(options, start);
    long double gap_residual = kappa;
    long double complementarity = static_cast<long double>(tau) * kappa;
    for (std::size_t col = 0; col < n; ++col) {
      if ((col & 4095) == 0) check_budget(options, start);
      dual_residual[col] = finite(static_cast<long double>(dual_residual[col]) +
          static_cast<long double>(model.costs[col]) * tau);
      gap_residual += static_cast<long double>(model.costs[col]) * point.x[col];
    }
    for (std::size_t row = 0; row < m; ++row) {
      if ((row & 4095) == 0) check_budget(options, start);
      primal_residual[row] = finite(static_cast<long double>(primal_residual[row]) + point.s[row] -
          static_cast<long double>(model.rhs[row]) * tau);
      gap_residual += static_cast<long double>(model.rhs[row]) * point.z[row];
      if (model.inequality[row]) {
        if (!(point.s[row] > 0) || !(point.z[row] > 0)) throw NumericalFailure("齐次候选离开正锥内部");
        diagonal[row] = finite(static_cast<long double>(point.s[row]) / point.z[row]);
        h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row]);
        complementarity += static_cast<long double>(point.s[row]) * point.z[row];
      } else {
        diagonal[row] = 0;
        h[row] = 0;
      }
    }
    if (!std::isfinite(gap_residual)) throw NumericalFailure("齐次间隙残差非有限");
    const double mu = finite(complementarity / (static_cast<long double>(model.barrier_rows) + 1));
    factor.factor(diagonal, options, start, result);
    check_budget(options, start);
    std::vector<double> constant;
    detail::IpmHomogeneousScaling scaling;
    long double denominator = 0;
    if (scaled_direction) {
      const auto budget = [&] { check_budget(options, start); };
      scaling = detail::make_ipm_homogeneous_scaling(n, model.inequality, point.s, point.z, tau, kappa, budget);
      const auto scaled_rhs = detail::ipm_homogeneous_scaled_constant_rhs(scaling, model.costs, model.rhs, budget);
      constant = factor.solve_homogeneous_direct(scaled_rhs, options, start, result, true);
      denominator = detail::ipm_homogeneous_scaled_denominator(scaling, constant, budget);
    } else {
      ++*result.ipm_homogeneous_constant_solves;
      constant = factor.solve(constant_rhs, options, start, result);
    }
    const long double affine_h_tau = -static_cast<long double>(tau) * kappa;
    detail::IpmHomogeneousRepairInfo repair_info;
    auto* repair = options.ipm.linear_accuracy == "homogeneous_repair" ? &repair_info : nullptr;
    const auto affine = scaled_direction ? homogeneous_scaled_direction(model, point, tau, kappa, primal_residual,
        dual_residual, gap_residual, 1, h, affine_h_tau, constant, scaling, denominator, factor, options, start, result) :
        homogeneous_direction(model, point, tau, kappa, primal_residual,
            dual_residual, gap_residual, 1, h, affine_h_tau, constant, factor, options, start, result, repair);
    const double affine_step = homogeneous_step(model, point, tau, kappa, affine, options, start);
    const double remaining = 1 - affine_step;
    const double sigma = remaining * remaining * remaining;
    // 初版各轮的二阶项系数均为 1，不复制其他实现针对首轮的额外减弱。
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
    const auto corrected = scaled_direction ? homogeneous_scaled_direction(model, point, tau, kappa, primal_residual,
        dual_residual, gap_residual, 1 - sigma, h, h_tau, constant, scaling, denominator, factor, options, start, result) :
        homogeneous_direction(model, point, tau, kappa, primal_residual,
            dual_residual, gap_residual, 1 - sigma, h, h_tau, constant, factor, options, start, result, repair);
    const double step = 0.995 * homogeneous_step(model, point, tau, kappa, corrected, options, start);
    if (!(step > 0) || !std::isfinite(step)) throw NumericalFailure("齐次共同步长不是有限正数");
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
    result.ipm_homogeneous_tau = tau;
    result.ipm_homogeneous_kappa = kappa;
    result.ipm_iterations = iteration;
    if (assess_homogeneous(model, point, tau, options, start, result)) return;
    if (iteration == options.ipm.max_iterations) break;
  }
}

void validate_options(const SolveOptions& options) {
  if (options.ipm.method != "mehrotra" && options.ipm.method != "homogeneous") {
    throw std::invalid_argument("内点方法必须为 mehrotra 或 homogeneous");
  }
  if (options.ipm.method == "homogeneous" && ((options.ipm.linear_accuracy != "strict" &&
      options.ipm.linear_accuracy != "homogeneous_scaled" && options.ipm.linear_accuracy != "homogeneous_repair") ||
      options.ipm.krylov_history_size != 0 || options.ipm.direction_retries != 0 ||
      options.ipm.central_corrections != 0)) {
    throw std::invalid_argument("齐次嵌入仅支持 strict、homogeneous_scaled 或 homogeneous_repair，且要求关闭历史、方向重试和额外中心校正");
  }
  if (options.ipm.linear_accuracy == "homogeneous_scaled" && options.ipm.method != "homogeneous") {
    throw std::invalid_argument("homogeneous_scaled 精度仅支持 homogeneous 方法");
  }
  if (options.ipm.linear_accuracy == "homogeneous_repair" && options.ipm.method != "homogeneous") {
    throw std::invalid_argument("homogeneous_repair 精度仅支持 homogeneous 方法");
  }
  if (options.ipm.krylov_history_size < 0 || options.ipm.krylov_history_size > 8) {
    throw std::invalid_argument("Krylov 历史维数必须为 0 到 8");
  }
  if (options.ipm.krylov_history_size > 0 && options.ipm.linear_accuracy != "strict" &&
      options.ipm.linear_accuracy != "blockwise") {
    throw std::invalid_argument("Krylov 历史当前仅支持 strict 或 blockwise 精度模式");
  }
  if (options.ipm.eliminate_presolved_fixed_variables && !options.ipm.presolve_singleton_rows) {
    throw std::invalid_argument("消除归并后新固定列要求启用单非零输入行归并");
  }
  if (options.ipm.direction_retries < 0 || options.ipm.direction_retries > 8) {
    throw std::invalid_argument("内点方向重试预算必须为 0 到 8");
  }
  if (options.ipm.direction_retries > 0 && options.ipm.linear_accuracy != "strict" &&
      options.ipm.linear_accuracy != "blockwise" && options.ipm.linear_accuracy != "blockwise_weighted" &&
      options.ipm.linear_accuracy != "primal_dual_regularization" &&
      options.ipm.linear_accuracy != "curvature_regularization" && options.ipm.linear_accuracy != "full_newton") {
    throw std::invalid_argument("方向重试暂仅支持 strict、blockwise、blockwise_weighted、primal_dual_regularization、curvature_regularization 或 full_newton");
  }
  if (options.ipm.linear_solver != "qdldl" && options.ipm.linear_solver != "sparselu" &&
      options.ipm.linear_solver != "normal" && options.ipm.linear_solver != "qdldl_extended" &&
      options.ipm.linear_solver != "factorhighs" && options.ipm.linear_solver != "normal_augmented" &&
      options.ipm.linear_solver != "normal_multifrontal" && options.ipm.linear_solver != "normal_extended" && options.ipm.linear_solver != "normal_refined" &&
      options.ipm.linear_solver != "normal_cholmod" && options.ipm.linear_solver != "normal_cholmod_aat") {
    throw std::invalid_argument("内点法线性求解器必须为 qdldl、sparselu、normal、qdldl_extended、factorhighs、normal_augmented、normal_multifrontal、normal_extended、normal_refined、normal_cholmod 或 normal_cholmod_aat");
  }
  if ((options.ipm.linear_solver == "qdldl_extended" || options.ipm.linear_solver == "normal_extended" || options.ipm.linear_solver == "normal_refined") &&
      std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
    throw std::invalid_argument("本平台 long double 未提供比 double 更高的有效精度");
  }
  if (options.ipm.linear_solver != "qdldl" && options.ipm.dynamic_regularization) {
    throw std::invalid_argument("局部动态主元正则仅支持 QDLDL");
  }
  if ((options.ipm.linear_solver == "normal" || options.ipm.linear_solver == "normal_augmented" ||
       options.ipm.linear_solver == "normal_multifrontal" || options.ipm.linear_solver == "normal_extended" || options.ipm.linear_solver == "normal_refined" ||
       options.ipm.linear_solver == "normal_cholmod" || options.ipm.linear_solver == "normal_cholmod_aat") &&
      !options.ipm.eliminate_singletons) {
    throw std::invalid_argument("正规方程实验要求启用单非零不等式行消元");
  }
  if (options.ipm.central_corrections < 0 || options.ipm.central_corrections > 1) {
    throw std::invalid_argument("额外中心校正次数当前仅支持 0 或 1");
  }
  if (options.ipm.linear_accuracy != "strict" && options.ipm.linear_accuracy != "backward" &&
      options.ipm.linear_accuracy != "inexact" && options.ipm.linear_accuracy != "blockwise" &&
      options.ipm.linear_accuracy != "blockwise_weighted" && options.ipm.linear_accuracy != "blockwise_forcing" &&
      options.ipm.linear_accuracy != "blockwise_forcing_weighted" &&
      options.ipm.linear_accuracy != "adaptive_regularization" &&
      options.ipm.linear_accuracy != "primal_dual_regularization" &&
      options.ipm.linear_accuracy != "curvature_regularization" && options.ipm.linear_accuracy != "full_newton" &&
      options.ipm.linear_accuracy != "bounded_newton" && options.ipm.linear_accuracy != "homogeneous_scaled" &&
      options.ipm.linear_accuracy != "homogeneous_repair") {
    throw std::invalid_argument("内点法线性精度模式必须为 strict、backward、inexact、blockwise、blockwise_weighted、blockwise_forcing、blockwise_forcing_weighted、adaptive_regularization、primal_dual_regularization、curvature_regularization、full_newton、bounded_newton、homogeneous_scaled 或 homogeneous_repair");
  }
  if ((options.ipm.linear_accuracy == "full_newton" || options.ipm.linear_accuracy == "bounded_newton") &&
      options.ipm.central_corrections != 0) {
    throw std::invalid_argument("完整 Newton 方程实验暂不支持额外中心校正");
  }
  if (options.ipm.linear_accuracy == "inexact" && options.ipm.central_corrections != 0) {
    throw std::invalid_argument("分块非精确方向实验暂不支持额外中心校正");
  }
  if (options.ipm.inexact_prefilter && options.ipm.linear_accuracy != "inexact") {
    throw std::invalid_argument("非精确方向预拒绝仅支持 inexact 精度模式");
  }
  if (options.highs.solver != "choose" || options.highs.run_crossover != "on" ||
      options.highs.presolve != "choose") {
    throw std::invalid_argument("自主内点法不接受 HiGHS 专属选项");
  }
  if (options.threads != 1) throw std::invalid_argument("自主内点法当前仅支持单线程");
  if (!std::isfinite(options.time_limit) || options.time_limit < 0 || options.ipm.max_iterations < 0 ||
      options.ipm.scaling_passes < 0 || options.ipm.scaling_passes > 20) {
    throw std::invalid_argument("自主内点法的预算或缩放轮数无效");
  }
  const auto& t = options.tolerances;
  const auto& o = options.pdlp.optimality;
  for (double value : {t.feasibility_absolute, t.feasibility_relative, t.integrality,
                       t.objective_absolute, t.objective_relative, o.dual_absolute,
                       o.dual_relative, o.gap_absolute, o.gap_relative}) {
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("内点法验算容差必须为有限非负数");
  }
}

}  // 匿名命名空间

/*
以下中心校正辅助函数依据 HiGHS v1.15.1 的 HiPO Solver.cpp 与 Parameters.h 适配。
来源：https://github.com/ERGO-Code/HiGHS/tree/v1.15.1/highs/ipm/hipo/ipm
保留其试探 max、中心区间与截断、九点权重及步长接受规则；主算法的 sigma 与 .995 仍由本仓库计算。

MIT License

Copyright (c) 2026 HiGHS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

namespace {

double central_boundary(std::span<const double> point, std::span<const double> direction,
    std::span<const double> correction, const std::vector<bool>& inequality, double weight) {
  constexpr double damp = 1 - 1000 * std::numeric_limits<double>::epsilon();
  double alpha = 1;
  for (std::size_t row = 0; row < point.size(); ++row) {
    if (!inequality[row]) continue;
    const double value = correction.empty() ? direction[row] :
        finite(static_cast<long double>(direction[row]) + weight * correction[row]);
    if (!std::isfinite(point[row]) || !(point[row] > 0) || !std::isfinite(value)) {
      throw NumericalFailure("中心校正边界数据无效");
    }
    if (static_cast<long double>(point[row]) + static_cast<long double>(alpha) * value < 0) {
      alpha = finite(-static_cast<long double>(point[row]) * damp / value);
    }
  }
  return alpha;
}

void central_sizes(std::size_t size, std::span<const double> dual,
    std::span<const double> slack_direction, std::span<const double> dual_direction,
    const std::vector<bool>& inequality) {
  if (dual.size() != size || slack_direction.size() != size ||
      dual_direction.size() != size || inequality.size() != size) {
    throw std::invalid_argument("内部中心校正向量维数不一致");
  }
}

}  // 匿名命名空间

detail::IpmCentralityTarget detail::ipm_centrality_target(
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> slack_direction, std::span<const double> dual_direction,
    const std::vector<bool>& inequality, double target_mu,
    const std::function<void()>& budget_check) {
  central_sizes(slack.size(), dual, slack_direction, dual_direction, inequality);
  if (budget_check) budget_check();
  IpmCentralityTarget result;
  if (!std::isfinite(target_mu)) throw NumericalFailure("中心校正目标非有限");
  if (!(target_mu > 0)) return result;
  result.primal_step = central_boundary(slack, slack_direction, {}, inequality, 0);
  result.dual_step = central_boundary(dual, dual_direction, {}, inequality, 0);
  if (result.primal_step == 1 && result.dual_step == 1) return result;
  const double trial_primal = std::max(1.0, result.primal_step + 0.1);
  const double trial_dual = std::max(1.0, result.dual_step + 0.1);
  const long double lower = static_cast<long double>(target_mu) * 0.1;
  const long double upper = static_cast<long double>(target_mu) / 0.1;
  result.target.assign(slack.size(), 0);
  if (budget_check) budget_check();
  for (std::size_t row = 0; row < slack.size(); ++row) {
    if (!inequality[row]) continue;
    const long double product = (static_cast<long double>(slack[row]) + static_cast<long double>(trial_primal) * slack_direction[row]) *
        (static_cast<long double>(dual[row]) + static_cast<long double>(trial_dual) * dual_direction[row]);
    if (!std::isfinite(product)) throw NumericalFailure("中心校正试探乘积非有限");
    if (product < lower) result.target[row] = finite(lower - product);
    else if (product > upper) result.target[row] = finite(std::max(upper - product, -upper));
  }
  if (budget_check) budget_check();
  return result;
}

detail::IpmCentralityWeights detail::ipm_centrality_weights(
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> slack_direction, std::span<const double> dual_direction,
    std::span<const double> slack_correction, std::span<const double> dual_correction,
    const std::vector<bool>& inequality, double old_primal_step, double old_dual_step,
    const std::function<void()>& budget_check) {
  central_sizes(slack.size(), dual, slack_direction, dual_direction, inequality);
  if (slack_correction.size() != slack.size() || dual_correction.size() != slack.size() ||
      !std::isfinite(old_primal_step) || !std::isfinite(old_dual_step) ||
      old_primal_step < 0 || old_primal_step > 1 || old_dual_step < 0 || old_dual_step > 1) {
    throw std::invalid_argument("内部中心校正权重参数无效");
  }
  const double initial_weight = old_primal_step * old_dual_step;
  double best_primal = 0, best_dual = 0, primal_weight = 0, dual_weight = 0;
  const int candidates = initial_weight == 1 ? 1 : 9;
  for (int j = 0; j < candidates; ++j) {
    if (budget_check) budget_check();
    const double weight = j == 8 ? 1 : initial_weight + j * (1 - initial_weight) / 8;
    const double primal = central_boundary(slack, slack_direction, slack_correction, inequality, weight);
    const double dual_step = central_boundary(dual, dual_direction, dual_correction, inequality, weight);
    if (primal > best_primal) { best_primal = primal; primal_weight = weight; }
    if (dual_step > best_dual) { best_dual = dual_step; dual_weight = weight; }
  }
  if (budget_check) budget_check();
  IpmCentralityWeights result;
  if (best_primal >= old_primal_step + 0.1 * 0.1) result.primal_weight = primal_weight;
  if (best_dual >= old_dual_step + 0.1 * 0.1) result.dual_weight = dual_weight;
  return result;
}

namespace {

// 本快路要求扩展指数范围覆盖 double 乘积、全部可寻址条目的和及误差界。
// 不支持的格式仍调用原 actual helper，不能借类型名称假定更宽精度。
constexpr bool inexact_prefilter_supported =
    std::numeric_limits<double>::radix == 2 && std::numeric_limits<double>::digits == 53 &&
    std::numeric_limits<long double>::radix == 2 && std::numeric_limits<long double>::digits >= 64 &&
    std::numeric_limits<long double>::max_exponent >=
        2 * std::numeric_limits<double>::max_exponent + std::numeric_limits<std::size_t>::digits + 2 &&
    std::numeric_limits<long double>::min_exponent - 1 <=
        2 * (std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits) -
            std::numeric_limits<long double>::digits - 2;

long double upward(long double value) {
  if (value == 0) return 0;
  if (!(value > 0) || !std::isfinite(value)) return std::numeric_limits<long double>::infinity();
  return std::nextafter(value, std::numeric_limits<long double>::infinity());
}

long double rounding_bound(long double operations) {
  if (operations == 0) return 0;
  // 用 epsilon 而非 epsilon/2，覆盖最近舍入以及常见有向舍入环境。
  const long double numerator = upward(operations * std::numeric_limits<long double>::epsilon());
  const long double denominator = std::nextafter(1 - numerator,
      -std::numeric_limits<long double>::infinity());
  if (!(denominator > 0) || !std::isfinite(numerator)) {
    return std::numeric_limits<long double>::infinity();
  }
  return upward(numerator / denominator);
}

}  // 匿名命名空间

detail::IpmInexactPrefilter detail::make_ipm_inexact_prefilter(
    std::span<const std::int64_t> starts, std::span<const double> values,
    const std::function<void()>& budget_check) {
  if (starts.empty() || starts.back() < 0 ||
      static_cast<std::size_t>(starts.back()) != values.size()) {
    throw std::invalid_argument("内部预拒绝 CSC 接口维数不一致");
  }
  if (budget_check) budget_check();
  IpmInexactPrefilter result;
  if constexpr (!inexact_prefilter_supported) return result;
  result.column_absolute_sums.resize(starts.size() - 1);
  result.rounding_factors.resize(starts.size() - 1);
  for (std::size_t col = 0; col + 1 < starts.size(); ++col) {
    if (budget_check) budget_check();
    long double sum = 0;
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      sum += std::abs(static_cast<long double>(values[static_cast<std::size_t>(k)]));
    }
    const long double degree = static_cast<long double>(starts[col + 1] - starts[col]);
    const long double gamma = rounding_bound(degree);
    const long double denominator = std::nextafter(1 - gamma,
        -std::numeric_limits<long double>::infinity());
    result.column_absolute_sums[col] = denominator > 0 ? upward(sum / denominator) :
        std::numeric_limits<long double>::infinity();
    // 旧缓存含至多 2k 次乘加及最后加 rd；原 helper 的正确舍入 FMA 至多 k 次。
    // 锁定 MinGW x87 fmal 另有 31/22 位拆分：double 输入的四个部分积可精确表示，
    // 每次调用的 cross 加法与三次 ret 累加至多四次舍入，故用 gamma_(4k) 覆盖。
    // https://raw.githubusercontent.com/mingw-w64/mingw-w64/master/mingw-w64-crt/math/fmal.c
    // gamma_l=l*u/(1-l*u) 的求和界见 Higham 1993 §2，所有界均向上保护。
    // https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf
    result.rounding_factors[col] = upward(rounding_bound(2 * degree + 1) + rounding_bound(4 * degree));
  }
  if (budget_check) budget_check();
  return result;
}

bool detail::reject_ipm_inexact_dual(const IpmInexactPrefilter& prefilter,
    std::span<const long double> transpose_product, std::span<const double> dual_residual,
    std::span<const double> dual_direction, const std::function<void()>& budget_check) {
  if (transpose_product.size() != dual_residual.size()) {
    throw std::invalid_argument("内部预拒绝向量维数不一致");
  }
  if (budget_check) budget_check();
  if constexpr (!inexact_prefilter_supported) return false;
  if (prefilter.column_absolute_sums.size() != dual_residual.size() ||
      prefilter.rounding_factors.size() != dual_residual.size()) return false;
  long double direction_norm = 0, residual_norm = 0;
  for (double value : dual_direction) {
    if (!std::isfinite(value)) return false;
    direction_norm = std::max(direction_norm, std::abs(static_cast<long double>(value)));
  }
  for (double value : dual_residual) {
    if (!std::isfinite(value)) return false;
    residual_norm = std::max(residual_norm, std::abs(static_cast<long double>(value)));
  }
  const long double threshold = ipm_inexact_forcing * residual_norm;
  for (std::size_t col = 0; col < dual_residual.size(); ++col) {
    if (budget_check) budget_check();
    const long double sum_bound = prefilter.column_absolute_sums[col];
    const long double coefficient = prefilter.rounding_factors[col];
    if (!(sum_bound >= 0) || !std::isfinite(sum_bound) || !(coefficient >= 0) ||
        !std::isfinite(coefficient) || !std::isfinite(transpose_product[col])) return false;
    const long double product_bound = upward(sum_bound * direction_norm);
    if (sum_bound != 0 && direction_norm != 0 && product_bound == 0) return false;
    const long double magnitude = upward(product_bound + std::abs(static_cast<long double>(dual_residual[col])));
    const long double error = upward(coefficient * magnitude);
    if (coefficient != 0 && magnitude != 0 && error == 0) return false;
    const long double limit = upward(threshold + error);
    const long double approximate = transpose_product[col] + dual_residual[col];
    if (!std::isfinite(limit) || !std::isfinite(approximate)) return false;
    // |q_cache-q_actual|<=B；只有 |q_cache|>up(threshold+B) 才能证实原 helper 必拒。
    // 接近门槛、零块相消、特殊值及不可靠界均继续原 helper，绝不授权接受。
    if (std::abs(approximate) > limit) {
      if (budget_check) budget_check();
      return true;
    }
  }
  if (budget_check) budget_check();
  return false;
}

bool detail::accept_ipm_inexact_direction(
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> primal_residual,
    std::span<const double> dual_residual, std::span<const double> slack_direction,
    std::span<const double> direction, const std::function<void()>& budget_check) {
  if (starts.empty() || rows.size() != values.size() ||
      dual_residual.size() != starts.size() - 1 ||
      slack_direction.size() != primal_residual.size() ||
      direction.size() != dual_residual.size() + primal_residual.size()) {
    throw std::invalid_argument("内部非精确方向接口维数不一致");
  }
  if (budget_check) budget_check();
  const auto n = dual_residual.size(), m = primal_residual.size();
  std::vector<long double> primal(slack_direction.begin(), slack_direction.end());
  std::vector<long double> dual(dual_residual.begin(), dual_residual.end());
  long double primal_norm = 0, dual_norm = 0;
  for (double value : direction) if (!std::isfinite(value)) return false;
  for (std::size_t row = 0; row < m; ++row) {
    if (!std::isfinite(primal_residual[row]) || !std::isfinite(primal[row])) return false;
    primal_norm = std::max(primal_norm, std::abs(static_cast<long double>(primal_residual[row])));
  }
  for (std::size_t col = 0; col < n; ++col) {
    if (!std::isfinite(dual[col])) return false;
    dual_norm = std::max(dual_norm, std::abs(dual[col]));
  }
  const auto accumulate = [](double left, double right, long double& sum) {
    const long double product = static_cast<long double>(left) * right;
    if (!std::isfinite(product) || (left != 0 && right != 0 && product == 0)) return false;
    // 融合乘加保留与已恢复 ds 相消的乘积低位；不把舍入后的 H 误当实际方程。
    sum = std::fma(static_cast<long double>(left), static_cast<long double>(right), sum);
    return std::isfinite(sum);
  };
  for (std::size_t col = 0; col < n; ++col) {
    if (budget_check) budget_check();
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      const auto row = static_cast<std::size_t>(rows[position]);
      if (!accumulate(values[position], direction[col], primal[row]) ||
          !accumulate(values[position], direction[n + row], dual[col])) return false;
    }
  }
  long double primal_defect = 0, dual_defect = 0;
  for (std::size_t row = 0; row < m; ++row) {
    primal[row] += primal_residual[row];
    if (!std::isfinite(primal[row])) return false;
    primal_defect = std::max(primal_defect, std::abs(primal[row]));
  }
  for (long double value : dual) dual_defect = std::max(dual_defect, std::abs(value));
  if (budget_check) budget_check();
  // 精确算术下该分块界推出可行性收缩；浮点缺陷只是数值诊断，不是证书。
  // 不把完整右端 h/z 或 mu 用作地板，也不宣称完整 Mehrotra 的全局收敛定理。
  // https://users.wpi.edu/~walker/Papers/forcing_terms,SISC_17,1996,16-32.pdf
  return primal_defect <= ipm_inexact_forcing * primal_norm &&
      dual_defect <= ipm_inexact_forcing * dual_norm;
}

long double detail::assess_ipm_kkt_residual(
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> diagonal_h,
    std::span<const double> rhs, std::span<const double> point,
    std::size_t maximum_row_nonzeros, std::vector<double>& residual,
    const std::function<void()>& budget_check) {
  if (starts.empty() || rows.size() != values.size() ||
      rhs.size() != starts.size() - 1 + diagonal_h.size() || point.size() != rhs.size()) {
    throw std::invalid_argument("内部 KKT 残差接口维数不一致");
  }
  if (budget_check) budget_check();
  const auto n = starts.size() - 1;
  std::vector<long double> product(rhs.size(), 0), denominator(rhs.size());
  bool reliable = true;
  for (std::size_t i = 0; i < rhs.size(); ++i) denominator[i] = std::abs(static_cast<long double>(rhs[i]));
  const auto multiply = [&reliable](double left, double right) {
    const long double value = static_cast<long double>(left) * right;
    if (!std::isfinite(value) || (left != 0 && right != 0 && value == 0)) reliable = false;
    return value;
  };
  for (std::size_t row = 0; row < diagonal_h.size(); ++row) {
    const auto position = n + row;
    product[position] = -multiply(diagonal_h[row], point[position]);
    denominator[position] += std::abs(product[position]);
  }
  for (std::size_t col = 0; col < n; ++col) {
    if (budget_check) budget_check();
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      const auto position = static_cast<std::size_t>(k);
      const auto row = n + static_cast<std::size_t>(rows[position]);
      const long double primal_product = multiply(values[position], point[row]);
      const long double dual_product = multiply(values[position], point[col]);
      product[col] += primal_product;
      product[row] += dual_product;
      denominator[col] += std::abs(primal_product);
      denominator[row] += std::abs(dual_product);
    }
  }
  // BERR 使用完整、未正则化 K0 的扩展精度残差和绝对乘积和。
  // 阈值 u = epsilon / 2；小分母保护按 LAPACK DSYRFS 的 SAFE1、SAFE2。
  // https://netlib.org/lapack/explore-html/d9/d6a/dsyrfs_8f_source.html
  // HiPO §4.5 使用同类分量后向误差；这里的小分母实现采用 LAPACK，而非其替代公式。
  // https://arxiv.org/html/2508.04370v1#S4.SS5
  const long double safe1 = (static_cast<long double>(maximum_row_nonzeros) + 1) *
      std::numeric_limits<double>::min();
  const long double safe2 = safe1 / detail::ipm_backward_error_tolerance;
  long double error = 0;
  residual.resize(rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    const long double difference = static_cast<long double>(rhs[i]) - product[i];
    residual[i] = finite(difference);
    const long double numerator = std::abs(difference);
    const long double divisor = denominator[i];
    if (!std::isfinite(numerator) || !std::isfinite(divisor) || divisor < 0) {
      reliable = false;
    } else if (divisor == 0) {
      if (numerator != 0) reliable = false;
    } else {
      const long double ratio = divisor > safe2 ? numerator / divisor :
          (numerator + safe1) / (divisor + safe1);
      if (!std::isfinite(ratio)) reliable = false;
      else error = std::max(error, ratio);
    }
  }
  if (budget_check) budget_check();
  return reliable ? error : std::numeric_limits<long double>::infinity();
}

std::array<double, 2> detail::ipm_kkt_block_tolerances(std::span<const double> rhs, std::size_t split,
                                                   double relative) {
  if (split > rhs.size()) throw std::invalid_argument("KKT 分块右端维数无效");
  if (!(relative > 0 && relative < 1)) throw std::invalid_argument("KKT 分块相对系数必须介于零与一之间");
  std::array<double, 2> norms{};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    if (!std::isfinite(rhs[i])) throw std::invalid_argument("KKT 分块右端必须有限");
    auto& norm = norms[i < split ? 0 : 1];
    norm = std::max(norm, std::abs(rhs[i]));
  }
  return {1e-12 + relative * norms[0], 1e-12 + relative * norms[1]};
}

bool detail::ipm_kkt_blocks_satisfied(std::span<const double> residual, std::size_t split,
                                    const std::array<double, 2>& tolerances) {
  if (split > residual.size()) throw std::invalid_argument("KKT 分块残差维数无效");
  for (double tolerance : tolerances) {
    if (!(tolerance > 0) || !std::isfinite(tolerance)) throw std::invalid_argument("KKT 分块门槛必须为有限正数");
  }
  for (std::size_t i = 0; i < residual.size(); ++i) {
    if (!std::isfinite(residual[i]) || std::abs(residual[i]) > tolerances[i < split ? 0 : 1]) return false;
  }
  return true;
}

int detail::ipm_min_regularization_level() {
  double value = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
  int level = 0;
  while (value / 10 >= std::numeric_limits<double>::epsilon()) {
    value /= 10;
    --level;
  }
  return level;
}

double detail::ipm_regularization_at_level(int level) {
  if (level < ipm_min_regularization_level() || level >= 8) {
    throw std::invalid_argument("正则化层级超出允许的有限网格");
  }
  double value = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
  for (int i = 0; i < level; ++i) value *= 10;
  for (int i = 0; i > level; --i) value /= 10;
  return value;
}

std::array<double, 2> detail::ipm_regularization_pair(int level, bool separate) {
  const double primal = ipm_regularization_at_level(level);
  // Altman–Gondzio (5.9) 给出两个默认量级；这里只做静态成对消融，不复现局部主元算法。
  // 正的二进制尺度不改变原十倍网格及重试预算，默认仍返回完全相同的两项。
  const double dual = separate ? primal / std::pow(std::numeric_limits<double>::epsilon(), 0.25) : primal;
  return {primal, dual};
}

bool detail::ipm_has_positive_curvature(std::span<const long double> diagonal) {
  return !diagonal.empty() && std::all_of(diagonal.begin(), diagonal.end(), [](long double value) {
    const double stored = static_cast<double>(value);
    return std::isfinite(value) && std::isfinite(stored) && stored > 0;
  });
}

int detail::ipm_regularization_search_level(int preferred_level, int actual_level, double dual_error,
    std::span<const double> dual_residual, std::span<const double> primal_direction) {
  ipm_regularization_at_level(preferred_level);
  const double regularization = ipm_regularization_at_level(actual_level);
  if (actual_level < preferred_level || dual_residual.size() != primal_direction.size() ||
      !std::isfinite(dual_error) || dual_error < 0) {
    throw std::invalid_argument("正则化搜索的层级、误差或向量维数无效");
  }
  long double residual_norm = 0, direction_norm = 0;
  for (std::size_t i = 0; i < dual_residual.size(); ++i) {
    if (!std::isfinite(dual_residual[i]) || !std::isfinite(primal_direction[i])) {
      throw std::invalid_argument("正则化搜索要求有限的残差与预测方向");
    }
    residual_norm = std::max(residual_norm, std::abs(static_cast<long double>(dual_residual[i])));
    direction_norm = std::max(direction_norm, std::abs(static_cast<long double>(primal_direction[i])));
  }
  const long double target = 1e-12L + ipm_inexact_forcing * residual_norm;
  // 同一矩阵的较小层级刚刚失败时，不重复支付这次已知无效的搜索成本。
  // 真实缺陷已经足够小（包括 IR/GMRES 消除偏移）时，也不靠 δ||dx|| 单独触发。
  if (actual_level != preferred_level || dual_error <= target ||
      static_cast<long double>(regularization) * direction_norm <= target) return actual_level;
  int level = actual_level;
  while (level > ipm_min_regularization_level() &&
         static_cast<long double>(ipm_regularization_at_level(level)) * direction_norm > target) --level;
  return level;
}

namespace {

void full_newton_budget(const std::function<void()>& check) {
  if (check) check();
}

std::size_t full_newton_block(const detail::IpmFullNewtonLayout& layout, std::size_t index) {
  return index < layout.columns ? 0 : index < layout.columns + layout.rows ? 1 : 2;
}

std::size_t homogeneous_scaled_block(const detail::IpmHomogeneousScaling& scaling, std::size_t index) {
  const auto n = scaling.layout.columns, gap = n + scaling.layout.rows;
  return index < n ? 0 : index < gap ? 1 : index == gap ? 2 : index + 1 < scaling.dimension() ? 3 : 4;
}

double full_newton_dot(const std::vector<double>& left, const std::vector<double>& right,
    const std::function<void()>& check) {
  full_newton_budget(check);
  long double sum = 0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    sum += static_cast<long double>(left[i]) * right[i];
  }
  full_newton_budget(check);
  return finite(sum);
}

double full_newton_norm(const std::vector<double>& values, const std::function<void()>& check) {
  full_newton_budget(check);
  long double scale = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const double value = values[i];
    if (!std::isfinite(value)) throw NumericalFailure("完整 Newton 搜索向量非有限");
    scale = std::max(scale, std::abs(static_cast<long double>(value)));
  }
  full_newton_budget(check);
  if (scale == 0) return 0;
  long double squares = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const long double ratio = values[i] / scale;
    squares += ratio * ratio;
  }
  full_newton_budget(check);
  return finite(scale * std::sqrt(squares));
}

}  // 匿名命名空间

detail::IpmFullNewtonLayout detail::make_ipm_full_newton_layout(std::size_t columns,
    const std::vector<bool>& inequality) {
  IpmFullNewtonLayout layout;
  layout.columns = columns;
  layout.rows = inequality.size();
  for (std::size_t row = 0; row < inequality.size(); ++row)
    if (inequality[row]) layout.inequality_rows.push_back(row);
  if (columns > std::numeric_limits<std::size_t>::max() - layout.rows ||
      columns + layout.rows > std::numeric_limits<std::size_t>::max() - layout.inequality_rows.size())
    throw std::invalid_argument("完整 Newton 方程维数超出范围");
  return layout;
}

void detail::ipm_full_newton_product(const IpmFullNewtonLayout& layout,
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> slack, std::span<const double> dual,
    std::span<const double> direction, std::vector<long double>& product,
    const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = layout.columns, m = layout.rows;
  if (starts.size() != n + 1 || rows.size() != values.size() || slack.size() != m || dual.size() != m ||
      direction.size() != layout.dimension()) throw std::invalid_argument("完整 Newton 乘积维数不一致");
  product.assign(layout.dimension(), 0);
  for (std::size_t col = 0; col < n; ++col) {
    full_newton_budget(check);
    long double upper = 0;
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      if ((k & 4095) == 0) full_newton_budget(check);
      const auto pos = static_cast<std::size_t>(k), row = static_cast<std::size_t>(rows[pos]);
      upper += static_cast<long double>(values[pos]) * direction[n + row];
      product[n + row] += static_cast<long double>(values[pos]) * direction[col];
    }
    product[col] = upper;
  }
  for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = layout.inequality_rows[i];
    const long double ds = direction[n + m + i];
    product[n + row] += ds;
    product[n + m + i] = static_cast<long double>(dual[row]) * ds +
        static_cast<long double>(slack[row]) * direction[n + row];
  }
  for (long double value : product)
    if (!std::isfinite(value)) throw NumericalFailure("完整 Newton 乘积非有限");
  full_newton_budget(check);
}

std::vector<double> detail::ipm_full_newton_reduced_rhs(const IpmFullNewtonLayout& layout,
    std::span<const double> dual, std::span<const double> rhs, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = layout.columns, m = layout.rows;
  if (dual.size() != m || rhs.size() != layout.dimension())
    throw std::invalid_argument("完整 Newton 预条件右端维数不一致");
  std::vector<double> reduced(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(n + m));
  for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = layout.inequality_rows[i];
    if (!(dual[row] > 0) || !std::isfinite(dual[row]))
      throw NumericalFailure("完整 Newton 预条件要求不等式乘子严格为正");
    reduced[n + row] = finite(static_cast<long double>(rhs[n + row]) -
        static_cast<long double>(rhs[n + m + i]) / dual[row]);
  }
  for (double value : reduced) if (!std::isfinite(value))
    throw NumericalFailure("完整 Newton 预条件右端非有限");
  full_newton_budget(check);
  return reduced;
}

std::vector<double> detail::ipm_full_newton_recover(const IpmFullNewtonLayout& layout,
    std::span<const double> slack, std::span<const double> dual,
    std::span<const double> rhs, std::span<const double> reduced_direction,
    const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = layout.columns, m = layout.rows;
  if (slack.size() != m || dual.size() != m || rhs.size() != layout.dimension() || reduced_direction.size() != n + m)
    throw std::invalid_argument("完整 Newton 预条件恢复维数不一致");
  std::vector<double> answer(reduced_direction.begin(), reduced_direction.end());
  answer.resize(layout.dimension());
  for (std::size_t i = 0; i < layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = layout.inequality_rows[i];
    if (!(dual[row] > 0) || !std::isfinite(dual[row]))
      throw NumericalFailure("完整 Newton 恢复要求不等式乘子严格为正");
    answer[n + m + i] = finite((static_cast<long double>(rhs[n + m + i]) -
        static_cast<long double>(slack[row]) * answer[n + row]) / dual[row]);
  }
  for (double value : answer) if (!std::isfinite(value))
    throw NumericalFailure("完整 Newton 预条件方向非有限");
  full_newton_budget(check);
  return answer;
}

std::array<double, 3> detail::ipm_full_newton_tolerances(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs) {
  if (rhs.size() != layout.dimension()) throw std::invalid_argument("完整 Newton 右端维数不一致");
  std::array<double, 3> tolerance{};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    if (!std::isfinite(rhs[i])) throw NumericalFailure("完整 Newton 右端非有限");
    const auto block = full_newton_block(layout, i);
    tolerance[block] = std::max(tolerance[block], std::abs(rhs[i]));
  }
  for (double& value : tolerance) value = 1e-12 + 1e-10 * value;
  return tolerance;
}

std::vector<double> detail::solve_ipm_full_newton(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs, const IpmFullNewtonPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmFullNewtonWork& work,
    const std::function<void()>& check) {
  full_newton_budget(check);
  const auto dimension = layout.dimension();
  const auto tolerance = ipm_full_newton_tolerances(layout, rhs);
  full_newton_budget(check);
  // 列数仍使用旧 KKT 维数，完整方向增加长度但不增加小问题的搜索预算。
  const auto limit = std::min<std::size_t>(32, layout.columns + layout.rows);
  if (dimension == 0) return {};
  const auto apply_preconditioner = [&](std::span<const double> value) {
    full_newton_budget(check);
    auto answer = precondition(value);
    full_newton_budget(check);
    if (answer.size() != dimension) throw std::invalid_argument("完整 Newton 预条件器返回维数不一致");
    for (double entry : answer) if (!std::isfinite(entry))
      throw NumericalFailure("完整 Newton 预条件器返回非有限方向");
    return answer;
  };
  const auto product = [&](const std::vector<double>& value, std::vector<long double>& image) {
    full_newton_budget(check);
    multiply(value, image);
    full_newton_budget(check);
    if (image.size() != dimension) throw std::invalid_argument("完整 Newton 算子返回维数不一致");
    for (long double entry : image) if (!std::isfinite(entry))
      throw NumericalFailure("完整 Newton 算子返回非有限值");
  };
  const auto assess = [&](const std::vector<double>& answer, std::vector<long double>& residual) {
    product(answer, residual);
    long double measure = 0;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      residual[i] = static_cast<long double>(rhs[i]) - residual[i];
      const long double scaled_residual = std::abs(residual[i]) / tolerance[full_newton_block(layout, i)];
      if (!std::isfinite(scaled_residual)) throw NumericalFailure("完整 Newton 真实残差非有限");
      measure = std::max(measure, scaled_residual);
    }
    full_newton_budget(check);
    return measure;
  };
  auto answer = apply_preconditioner(rhs);
  std::vector<long double> residual;
  long double previous = std::numeric_limits<long double>::infinity();
  for (int iteration = 0; ; ++iteration) {
    const long double measure = assess(answer, residual);
    if (measure <= 1) return answer;
    if (iteration == 2 || measure >= previous) break;
    previous = measure;
    std::vector<double> physical_residual(dimension);
    for (std::size_t i = 0; i < dimension; ++i) physical_residual[i] = finite(residual[i]);
    const auto correction = apply_preconditioner(physical_residual);
    ++work.refinement_steps;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      answer[i] = finite(static_cast<long double>(answer[i]) + correction[i]);
    }
  }
  for (int restart = 0; restart < 3; ++restart) {
    full_newton_budget(check);
    std::vector<std::vector<double>> basis(1, std::vector<double>(dimension)), directions;
    for (std::size_t i = 0; i < dimension; ++i)
      basis[0][i] = finite(residual[i] / tolerance[full_newton_block(layout, i)]);
    const double beta = full_newton_norm(basis[0], check);
    if (!(beta > 0)) throw NumericalFailure("完整 Newton Krylov 初始残差无效");
    for (double& value : basis[0]) value /= beta;
    std::vector<std::vector<double>> upper(limit, std::vector<double>(limit + 1, 0));
    std::vector<double> cosine(limit), sine(limit), transformed(limit + 1, 0);
    transformed[0] = beta;
    auto candidate = answer;
    for (std::size_t col = 0; col < limit; ++col) {
      full_newton_budget(check);
      auto physical_rhs = basis[col];
      for (std::size_t i = 0; i < dimension; ++i)
        physical_rhs[i] = finite(static_cast<long double>(physical_rhs[i]) * tolerance[full_newton_block(layout, i)]);
      directions.push_back(apply_preconditioner(physical_rhs));
      ++work.refinement_steps;
      ++work.krylov_steps;
      std::vector<long double> image;
      product(directions.back(), image);
      std::vector<double> column(dimension);
      for (std::size_t i = 0; i < dimension; ++i)
        column[i] = finite(image[i] / tolerance[full_newton_block(layout, i)]);
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t row = 0; row <= col; ++row) {
          const double projection = full_newton_dot(basis[row], column, check);
          upper[col][row] = finite(static_cast<long double>(upper[col][row]) + projection);
          for (std::size_t i = 0; i < dimension; ++i) {
            if ((i & 4095) == 0) full_newton_budget(check);
            column[i] = finite(static_cast<long double>(column[i]) - static_cast<long double>(projection) * basis[row][i]);
          }
        }
      }
      const double remaining = full_newton_norm(column, check);
      upper[col][col + 1] = remaining;
      if (remaining > 0) {
        for (double& value : column) value /= remaining;
        basis.push_back(std::move(column));
      }
      for (std::size_t row = 0; row < col; ++row) {
        const double first = upper[col][row], second = upper[col][row + 1];
        upper[col][row] = finite(static_cast<long double>(cosine[row]) * first + static_cast<long double>(sine[row]) * second);
        upper[col][row + 1] = finite(-static_cast<long double>(sine[row]) * first + static_cast<long double>(cosine[row]) * second);
      }
      const double diagonal = std::hypot(upper[col][col], upper[col][col + 1]);
      if (!(diagonal > 0) || !std::isfinite(diagonal)) throw NumericalFailure("完整 Newton Krylov 子空间退化");
      cosine[col] = upper[col][col] / diagonal;
      sine[col] = upper[col][col + 1] / diagonal;
      upper[col][col] = diagonal;
      upper[col][col + 1] = 0;
      transformed[col + 1] = finite(-static_cast<long double>(sine[col]) * transformed[col]);
      transformed[col] = finite(static_cast<long double>(transformed[col]) * cosine[col]);
      if (std::abs(transformed[col + 1]) <= 1 || col + 1 == limit || remaining == 0) {
        std::vector<double> coefficients(transformed.begin(), transformed.begin() + static_cast<std::ptrdiff_t>(col + 1));
        for (std::size_t row = col + 1; row-- > 0;) {
          long double value = coefficients[row];
          for (std::size_t j = row + 1; j <= col; ++j) value -= static_cast<long double>(upper[j][row]) * coefficients[j];
          coefficients[row] = finite(value / upper[row][row]);
        }
        candidate = answer;
        for (std::size_t j = 0; j <= col; ++j) {
          full_newton_budget(check);
          for (std::size_t i = 0; i < dimension; ++i) {
            if ((i & 4095) == 0) full_newton_budget(check);
            candidate[i] = finite(static_cast<long double>(candidate[i]) +
                static_cast<long double>(coefficients[j]) * directions[j][i]);
          }
        }
        if (assess(candidate, residual) <= 1) return candidate;
        if (remaining == 0) break;
      }
    }
    answer = std::move(candidate);
  }
  throw NumericalFailure("完整 Newton Krylov 改进未达到三块原方程精度");
}

std::vector<double> detail::solve_ipm_bounded_newton(const IpmFullNewtonLayout& layout,
    std::span<const double> rhs, const IpmFullNewtonPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmFullNewtonWork& work, IpmBoundedNewtonInfo& info,
    const std::function<void()>& check) {
  full_newton_budget(check);
  info = {};
  const auto dimension = layout.dimension();
  const auto tolerance = ipm_full_newton_tolerances(layout, rhs);
  full_newton_budget(check);
  if (dimension == 0) {
    info.measure = 0;
    info.tolerance_satisfied = true;
    return {};
  }
  const auto apply_preconditioner = [&](std::span<const double> value) {
    full_newton_budget(check);
    auto answer = precondition(value);
    full_newton_budget(check);
    if (answer.size() != dimension) throw std::invalid_argument("有界 Newton 预条件器返回维数不一致");
    return answer;
  };
  const auto finite_direction = [&](const std::vector<double>& value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      if (!std::isfinite(value[i])) return false;
    }
    full_newton_budget(check);
    return true;
  };
  const auto assess = [&](const std::vector<double>& value, std::vector<long double>& residual) {
    full_newton_budget(check);
    multiply(value, residual);
    full_newton_budget(check);
    if (residual.size() != dimension) throw std::invalid_argument("有界 Newton 算子返回维数不一致");
    long double measure = 0;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      residual[i] = static_cast<long double>(rhs[i]) - residual[i];
      const long double scaled_residual = std::abs(residual[i]) / tolerance[full_newton_block(layout, i)];
      if (!std::isfinite(scaled_residual)) return std::numeric_limits<long double>::infinity();
      measure = std::max(measure, scaled_residual);
    }
    full_newton_budget(check);
    return measure;
  };
  auto answer = apply_preconditioner(rhs);
  if (!finite_direction(answer)) throw NumericalFailure("有界 Newton 初始方向非有限");
  std::vector<long double> residual;
  long double measure = assess(answer, residual);
  if (!std::isfinite(measure)) throw NumericalFailure("有界 Newton 初始真实残差非有限");
  // 只改变内层停机政策，真实 J、三块门槛及外层验算均保持原定义。
  for (int iteration = 0; iteration < 3 && measure > 1; ++iteration) {
    full_newton_budget(check);
    std::vector<double> physical_residual(dimension);
    bool representable = true;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      physical_residual[i] = static_cast<double>(residual[i]);
      representable = representable && std::isfinite(physical_residual[i]);
    }
    if (!representable) break;
    const auto correction = apply_preconditioner(physical_residual);
    ++work.refinement_steps;
    if (!finite_direction(correction)) break;
    auto candidate = answer;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      candidate[i] = static_cast<double>(static_cast<long double>(answer[i]) + correction[i]);
    }
    if (!finite_direction(candidate)) break;
    std::vector<long double> candidate_residual;
    const long double candidate_measure = assess(candidate, candidate_residual);
    if (!(candidate_measure < measure)) break;
    answer.swap(candidate);
    residual.swap(candidate_residual);
    measure = candidate_measure;
  }
  // 未采用的候选和回调异常不能污染已验算方向；预算异常始终向上传播。
  full_newton_budget(check);
  info.measure = measure;
  info.tolerance_satisfied = measure <= 1;
  return answer;
}

detail::IpmHomogeneousScaling detail::make_ipm_homogeneous_scaling(std::size_t columns,
    const std::vector<bool>& inequality, std::span<const double> slack,
    std::span<const double> dual, double tau, double kappa,
    const std::function<void()>& check) {
  full_newton_budget(check);
  if (slack.size() != inequality.size() || dual.size() != inequality.size())
    throw std::invalid_argument("齐次缩放的行维数不一致");
  if (!(tau > 0) || !(kappa > 0) || !std::isfinite(tau) || !std::isfinite(kappa))
    throw NumericalFailure("齐次缩放要求有限正 tau、kappa");
  IpmHomogeneousScaling scaling;
  scaling.layout = make_ipm_full_newton_layout(columns, inequality);
  if (scaling.layout.dimension() > std::numeric_limits<std::size_t>::max() - 2)
    throw std::invalid_argument("齐次五块系统维数溢出");
  scaling.d = std::sqrt(static_cast<long double>(kappa) / tau);
  scaling.lambda_tau = std::sqrt(static_cast<long double>(tau) * kappa);
  const auto valid_scale = [](long double value) { return value > 0 && std::isfinite(value); };
  if (!valid_scale(scaling.d) || !valid_scale(scaling.lambda_tau))
    throw NumericalFailure("齐次标量缩放超出有限正数范围");
  for (const auto row : scaling.layout.inequality_rows) {
    if ((scaling.weights.size() & 4095) == 0) full_newton_budget(check);
    if (!(slack[row] > 0) || !(dual[row] > 0) || !std::isfinite(slack[row]) || !std::isfinite(dual[row]))
      throw NumericalFailure("齐次锥缩放要求有限正松弛和乘子");
    const long double weight = std::sqrt(static_cast<long double>(slack[row]) / dual[row]);
    const long double lambda = std::sqrt(static_cast<long double>(slack[row]) * dual[row]);
    if (!valid_scale(weight) || !valid_scale(lambda)) throw NumericalFailure("齐次锥缩放超出有限正数范围");
    scaling.weights.push_back(weight);
    scaling.lambda.push_back(lambda);
  }
  full_newton_budget(check);
  return scaling;
}

std::vector<double> detail::ipm_homogeneous_scaled_constant_rhs(const IpmHomogeneousScaling& scaling,
    std::span<const double> costs, std::span<const double> bounds, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = scaling.layout.columns, m = scaling.layout.rows;
  if (costs.size() != n || bounds.size() != m) throw std::invalid_argument("齐次常数右端维数不一致");
  std::vector<double> rhs(n + m);
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    rhs[i] = finite((i < n ? -static_cast<long double>(costs[i]) : bounds[i - n]) / scaling.d);
  }
  full_newton_budget(check);
  return rhs;
}

long double detail::ipm_homogeneous_scaled_denominator(const IpmHomogeneousScaling& scaling,
    std::span<const double> constant, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = scaling.layout.columns, m = scaling.layout.rows;
  if (constant.size() != n + m) throw std::invalid_argument("齐次常数方向维数不一致");
  for (std::size_t i = 0; i < constant.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    if (!std::isfinite(constant[i])) throw NumericalFailure("齐次常数方向非有限");
  }
  long double denominator = 1;
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const long double z = scaling.weights[i] * constant[n + scaling.layout.inequality_rows[i]];
    denominator += z * z;
  }
  if (!(denominator > 0) || !std::isfinite(denominator))
    throw NumericalFailure("齐次正能量分母非有限");
  full_newton_budget(check);
  return denominator;
}

std::vector<double> detail::ipm_homogeneous_scaled_reduced_rhs(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = scaling.layout.columns, gap = n + scaling.layout.rows;
  if (rhs.size() != scaling.dimension()) throw std::invalid_argument("齐次五块右端维数不一致");
  std::vector<double> reduced(rhs.begin(), rhs.begin() + static_cast<std::ptrdiff_t>(gap));
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = scaling.layout.inequality_rows[i];
    reduced[n + row] = finite(static_cast<long double>(rhs[n + row]) - scaling.weights[i] * rhs[gap + 1 + i]);
  }
  full_newton_budget(check);
  return reduced;
}

std::vector<double> detail::ipm_homogeneous_scaled_recover(const IpmHomogeneousScaling& scaling,
    std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> rhs, std::span<const double> reduced,
    std::span<const double> constant, long double denominator, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = scaling.layout.columns, m = scaling.layout.rows, gap = n + m;
  const auto tau_index = scaling.dimension() - 2, kappa_index = tau_index + 1;
  if (costs.size() != n || bounds.size() != m || rhs.size() != scaling.dimension() ||
      reduced.size() != gap || constant.size() != gap)
    throw std::invalid_argument("齐次缩放方向恢复维数不一致");
  if (!(denominator > 0) || !std::isfinite(denominator)) throw NumericalFailure("齐次正能量分母无效");
  long double numerator = static_cast<long double>(rhs.back()) - rhs[gap];
  for (std::size_t i = 0; i < gap; ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    numerator += static_cast<long double>(i < n ? costs[i] : bounds[i - n]) * reduced[i] / scaling.d;
  }
  const long double t = numerator / denominator;
  if (!std::isfinite(t)) throw NumericalFailure("齐次缩放标量方向非有限");
  std::vector<double> direction(scaling.dimension());
  direction[tau_index] = finite(t / scaling.d);
  direction[kappa_index] = finite(scaling.d * (static_cast<long double>(rhs.back()) - t));
  for (std::size_t i = 0; i < gap; ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    direction[i] = finite(static_cast<long double>(reduced[i]) + constant[i] * t);
  }
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = scaling.layout.inequality_rows[i];
    direction[gap + i] = finite(scaling.weights[i] *
        (static_cast<long double>(rhs[gap + 1 + i]) - scaling.weights[i] * direction[n + row]));
  }
  full_newton_budget(check);
  return direction;
}

void detail::ipm_homogeneous_scaled_product(const IpmHomogeneousScaling& scaling,
    std::span<const std::int64_t> starts, std::span<const std::int64_t> rows,
    std::span<const double> values, std::span<const double> costs, std::span<const double> bounds,
    std::span<const double> slack, std::span<const double> dual, double tau, double kappa,
    std::span<const double> direction, std::vector<long double>& product,
    const std::function<void()>& check) {
  full_newton_budget(check);
  const auto n = scaling.layout.columns, m = scaling.layout.rows, gap = n + m;
  const auto tau_index = scaling.dimension() - 2;
  if (starts.size() != n + 1 || rows.size() != values.size() || costs.size() != n || bounds.size() != m ||
      slack.size() != m || dual.size() != m || direction.size() != scaling.dimension())
    throw std::invalid_argument("齐次五块算子维数不一致");
  product.assign(scaling.dimension(), 0);
  const long double dt = direction[tau_index], dk = direction[tau_index + 1];
  long double gap_product = dk;
  for (std::size_t row = 0; row < m; ++row) {
    if ((row & 4095) == 0) full_newton_budget(check);
    product[n + row] = -static_cast<long double>(bounds[row]) * dt;
    gap_product += static_cast<long double>(bounds[row]) * direction[n + row];
  }
  for (std::size_t col = 0; col < n; ++col) {
    full_newton_budget(check);
    long double value = static_cast<long double>(costs[col]) * dt;
    for (Index k = starts[col]; k < starts[col + 1]; ++k) {
      if ((k & 4095) == 0) full_newton_budget(check);
      const auto position = static_cast<std::size_t>(k), row = static_cast<std::size_t>(rows[position]);
      value += static_cast<long double>(values[position]) * direction[n + row];
      product[n + row] += static_cast<long double>(values[position]) * direction[col];
    }
    product[col] = value;
    gap_product += static_cast<long double>(costs[col]) * direction[col];
  }
  product[gap] = gap_product / scaling.d;
  for (std::size_t i = 0; i < scaling.layout.inequality_rows.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const auto row = scaling.layout.inequality_rows[i];
    const long double ds = direction[gap + i];
    product[n + row] += ds;
    product[gap + 1 + i] = (static_cast<long double>(dual[row]) * ds +
        static_cast<long double>(slack[row]) * direction[n + row]) / scaling.lambda[i];
  }
  product.back() = (static_cast<long double>(kappa) * dt + static_cast<long double>(tau) * dk) / scaling.lambda_tau;
  full_newton_budget(check);
}

std::array<double, 5> detail::ipm_homogeneous_scaled_tolerances(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const std::function<void()>& check) {
  full_newton_budget(check);
  if (rhs.size() != scaling.dimension()) throw std::invalid_argument("齐次五块门槛维数不一致");
  std::array<double, 5> tolerance{};
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    if (!std::isfinite(rhs[i])) throw NumericalFailure("齐次五块右端非有限");
    const auto block = homogeneous_scaled_block(scaling, i);
    tolerance[block] = std::max(tolerance[block], std::abs(rhs[i]));
  }
  for (auto& value : tolerance) value = 1e-12 + 1e-10 * value;
  full_newton_budget(check);
  return tolerance;
}

std::vector<double> detail::solve_ipm_homogeneous_scaled(const IpmHomogeneousScaling& scaling,
    std::span<const double> rhs, const IpmHomogeneousScaledPreconditioner& precondition,
    const IpmFullNewtonProduct& multiply, IpmBoundedNewtonInfo& info, const std::function<void()>& check) {
  full_newton_budget(check);
  const auto dimension = scaling.dimension();
  const auto tolerance = ipm_homogeneous_scaled_tolerances(scaling, rhs, check);
  const auto apply = [&](std::span<const double> input, bool refinement) {
    full_newton_budget(check);
    auto value = precondition(input, refinement);
    full_newton_budget(check);
    if (value.size() != dimension) throw std::invalid_argument("齐次预条件器返回维数不一致");
    return value;
  };
  const auto finite_direction = [&](std::span<const double> value) {
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      if (!std::isfinite(value[i])) return false;
    }
    full_newton_budget(check);
    return true;
  };
  const auto assess = [&](std::span<const double> direction, std::vector<long double>& residual) {
    full_newton_budget(check);
    multiply(direction, residual);
    full_newton_budget(check);
    if (residual.size() != dimension) throw std::invalid_argument("齐次五块算子返回维数不一致");
    long double measure = 0;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      residual[i] = static_cast<long double>(rhs[i]) - residual[i];
      const long double normalized = std::abs(residual[i]) / tolerance[homogeneous_scaled_block(scaling, i)];
      if (!std::isfinite(normalized)) return std::numeric_limits<long double>::infinity();
      measure = std::max(measure, normalized);
    }
    full_newton_budget(check);
    return measure;
  };
  auto answer = apply(rhs, false);
  if (!finite_direction(answer)) throw NumericalFailure("齐次缩放初始方向非有限");
  std::vector<long double> residual;
  long double measure = assess(answer, residual);
  if (!std::isfinite(measure)) throw NumericalFailure("齐次缩放初始真实残差非有限");
  for (int iteration = 0; iteration < 3 && measure > 1; ++iteration) {
    full_newton_budget(check);
    std::vector<double> scaled_residual(dimension);
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      scaled_residual[i] = static_cast<double>(residual[i]);
    }
    if (!finite_direction(scaled_residual)) break;
    const auto correction = apply(scaled_residual, true);
    if (!finite_direction(correction)) break;
    auto candidate = answer;
    for (std::size_t i = 0; i < dimension; ++i) {
      if ((i & 4095) == 0) full_newton_budget(check);
      candidate[i] = static_cast<double>(static_cast<long double>(answer[i]) + correction[i]);
    }
    if (!finite_direction(candidate)) break;
    std::vector<long double> candidate_residual;
    const long double candidate_measure = assess(candidate, candidate_residual);
    if (!(candidate_measure < measure)) break;
    answer.swap(candidate);
    residual.swap(candidate_residual);
    measure = candidate_measure;
  }
  full_newton_budget(check);
  info.measure = measure;
  info.tolerance_satisfied = measure <= 1;
  return answer;
}

detail::IpmHomogeneousScalarDirection detail::ipm_homogeneous_scalar_direction(
    std::span<const double> costs, std::span<const double> rhs,
    std::span<const double> u, std::span<const double> v, double tau, double kappa,
    long double gap_rhs, long double complementarity_rhs, const std::function<void()>& check) {
  full_newton_budget(check);
  if (costs.size() > std::numeric_limits<std::size_t>::max() - rhs.size() ||
      u.size() != costs.size() + rhs.size() || v.size() != u.size())
    throw std::invalid_argument("齐次标量消元的向量维数不一致");
  if (!(tau > 0) || !(kappa > 0) || !std::isfinite(tau) || !std::isfinite(kappa) ||
      !std::isfinite(gap_rhs) || !std::isfinite(complementarity_rhs))
    throw NumericalFailure("齐次标量消元要求有限右端和严格正的 tau、kappa");
  long double dot_u = 0, dot_v = 0;
  for (std::size_t i = 0; i < u.size(); ++i) {
    if ((i & 4095) == 0) full_newton_budget(check);
    const double coefficient = i < costs.size() ? costs[i] : rhs[i - costs.size()];
    if (!std::isfinite(coefficient) || !std::isfinite(u[i]) || !std::isfinite(v[i]))
      throw NumericalFailure("齐次标量消元输入非有限");
    dot_u += static_cast<long double>(coefficient) * u[i];
    dot_v += static_cast<long double>(coefficient) * v[i];
  }
  const long double denominator = static_cast<long double>(kappa) / tau - dot_v;
  if (!(denominator > 0) || !std::isfinite(denominator) || !std::isfinite(dot_u)) {
    if (std::isfinite(denominator) && denominator <= 0 && std::isfinite(dot_u))
      throw HomogeneousNonpositiveDenominator("齐次标量消元的分母不是有限正数");
    throw NumericalFailure("齐次标量消元的分母不是有限正数");
  }
  const long double dtau = (complementarity_rhs / tau + dot_u - gap_rhs) / denominator;
  IpmHomogeneousScalarDirection result;
  result.tau = finite(dtau);
  // 与最终存储的 tau 方向配对恢复，避免使用已丢弃的扩展精度方向。
  result.kappa = finite((complementarity_rhs - static_cast<long double>(kappa) * result.tau) / tau);
  full_newton_budget(check);
  return result;
}

detail::IpmHomogeneousScalarDirection detail::ipm_homogeneous_scalar_direction_repair(
    std::span<const double> costs, std::span<const double> rhs,
    std::span<const double> u, std::vector<double>& v, double tau, double kappa,
    long double gap_rhs, long double complementarity_rhs,
    const IpmHomogeneousConstantRepair& repair, IpmHomogeneousRepairInfo& info,
    const std::function<void()>& check) {
  try {
    return ipm_homogeneous_scalar_direction(costs, rhs, u, v, tau, kappa, gap_rhs, complementarity_rhs, check);
  } catch (const HomogeneousNonpositiveDenominator&) {
    if (info.attempted) throw;
    full_newton_budget(check);
    info.attempted = true;
    auto candidate = repair();
    full_newton_budget(check);
    const auto direction = ipm_homogeneous_scalar_direction(
        costs, rhs, u, candidate, tau, kappa, gap_rhs, complementarity_rhs, check);
    full_newton_budget(check);
    v = std::move(candidate);
    info.succeeded = true;
    return direction;
  }
}

double detail::ipm_homogeneous_step_limit(double cone_limit, double tau, double kappa,
    const IpmHomogeneousScalarDirection& direction, const std::function<void()>& check) {
  full_newton_budget(check);
  if (std::isnan(cone_limit) || cone_limit < 0 || !(tau > 0) || !(kappa > 0) ||
      !std::isfinite(tau) || !std::isfinite(kappa) ||
      !std::isfinite(direction.tau) || !std::isfinite(direction.kappa))
    throw NumericalFailure("齐次步长边界输入无效");
  long double step = std::min(1.0L, static_cast<long double>(cone_limit));
  if (direction.tau < 0) step = std::min(step, -static_cast<long double>(tau) / direction.tau);
  if (direction.kappa < 0) step = std::min(step, -static_cast<long double>(kappa) / direction.kappa);
  const double result = finite(step);
  full_newton_budget(check);
  return result;
}

std::string ipm_version() { return "ipm-cpu-0.53.0"; }

SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  SolveResult result;
  result.backend = "ipm";
  result.backend_version = ipm_version();
  result.model = model;
  result.ipm_iterations = 0;
  result.ipm_factorizations = 0;
  result.ipm_refinement_steps = 0;
  result.ipm_weighted_krylov_steps = 0;
  result.ipm_full_newton_preconditioner_solves = 0;
  result.ipm_full_newton_products = 0;
  result.ipm_bounded_newton_returns = 0;
  result.ipm_bounded_newton_inexact_returns = 0;
  result.ipm_homogeneous_constant_solves = 0;
  result.ipm_homogeneous_constant_repairs = 0;
  result.ipm_homogeneous_constant_repair_successes = 0;
  result.ipm_homogeneous_direction_solves = 0;
  result.ipm_homogeneous_scaled_products = 0;
  result.ipm_homogeneous_scaled_refinement_steps = 0;
  result.ipm_homogeneous_scaled_inexact_returns = 0;
  result.ipm_regularization_searches = 0;
  result.ipm_regularization_reductions = 0;
  result.ipm_unshifted_factorizations = 0;
  result.ipm_direction_retries = 0;
  result.ipm_direction_retry_successes = 0;
  result.ipm_krylov_recycle_projections = 0;
  result.ipm_krylov_recycle_acceptances = 0;
  result.ipm_krylov_recycle_vectors = 0;
  result.ipm_krylov_recycle_peak_dimension = 0;
  result.ipm_schur_residual_products = 0;
  result.ipm_schur_refinement_steps = 0;
  result.ipm_schur_refinement_acceptances = 0;
  result.ipm_structural_fallbacks = 0;
  result.ipm_eliminated_fixed_variables = 0;
  result.ipm_presolved_singleton_rows = 0;
  result.ipm_presolved_fixed_columns = 0;
  result.ipm_regularized_pivots = 0;
  result.ipm_backward_error_acceptances = 0;
  result.ipm_inexact_acceptances = 0;
  result.ipm_inexact_prefilter_rejections = 0;
  result.ipm_central_correction_attempts = 0;
  result.ipm_central_correction_acceptances = 0;
  try {
    validate_options(options);
    const auto errors = validate_model(model);
    if (!errors.empty()) throw std::invalid_argument("模型无效：" + errors.front());
    if (is_mip(model)) throw std::invalid_argument("自主内点法仅支持连续 LP");
    if (options.time_limit == 0) {
      result.status = "time_limit";
      result.message = "时间预算为零，未启动自主内点法";
    } else if (options.ipm.max_iterations == 0) {
      result.status = "iteration_limit";
      result.message = "迭代预算为零，未启动自主内点法";
    } else {
      const auto working = canonicalize(model, options, start);
      result.ipm_eliminated_fixed_variables = static_cast<Index>(model.costs.size() - working.costs.size());
      result.ipm_presolved_singleton_rows = working.presolved_singleton_rows;
      result.ipm_presolved_fixed_columns = static_cast<Index>(working.presolved_fixed_columns.size());
      if (options.ipm.method == "homogeneous") {
        run_homogeneous_ipm(working, options, start, result);
      } else {
        run_ipm(working, options, start, result);
      }
    }
  } catch (const BudgetExpired&) {
    result.status = "time_limit";
    result.message = "内点求解已达到时间预算";
  } catch (const NumericalFailure& error) {
    result.status = "numerical_error";
    result.message = error.what();
  } catch (const std::exception& error) {
    result.status = "error";
    result.message = "自主内点法失败：" + std::string(error.what());
  }
  result.backend_status = result.status;
  result.total_seconds = elapsed(start);
  result.solve_seconds = std::max(0.0, result.total_seconds - result.verification_seconds);
  return result;
}

}  // 命名空间 solver
