#include "solver/ipm.hpp"
#include "solver/detail/extended_ldl.hpp"
#include "solver/detail/ipm_centrality.hpp"
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


#include <array>
#include <cstdio>
namespace solver_ipm_profile {
using Clock = std::chrono::steady_clock;
struct Entry { double inclusive = 0, exclusive = 0; unsigned long long calls = 0; };
constexpr const char* names[] = {"total", "canonicalize", "symbolic", "factor",
  "solve", "krylov", "direct", "solve_factored", "kkt_product", "assess",
  "direction", "initialize", "run_ipm", "inexact_check", "factor_kernel"};
struct Scope;
inline thread_local std::array<Entry, 15> entries{};
inline thread_local Scope* parent = nullptr;
struct Scope {
  int index;
  Scope* previous;
  Clock::time_point start;
  double children = 0;
  explicit Scope(int value) : index(value), previous(parent), start(Clock::now()) { parent = this; }
  ~Scope() {
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    entries[index].inclusive += seconds;
    entries[index].exclusive += seconds - children;
    ++entries[index].calls;
    parent = previous;
    if (previous) previous->children += seconds;
  }
};
template<class Function> auto factor_call(Function&& function) {
  Scope scope(14);
  return function();
}
struct Report {
  Report() { entries = {}; }
  ~Report() {
    std::fputs("IPM_PROFILE {", stderr);
    for (int i = 0; i < 15; ++i) {
      std::fprintf(stderr, "%s\"%s\":{\"inclusive\":%.17g,\"exclusive\":%.17g,\"calls\":%llu}",
          i ? "," : "", names[i], entries[i].inclusive, entries[i].exclusive, entries[i].calls);
    }
    std::fputs("}\n", stderr);
  }
};
}

namespace solver {
namespace {

using Clock = std::chrono::steady_clock;
using Index = std::int64_t;

struct NumericalFailure : std::runtime_error { using std::runtime_error::runtime_error; };
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
  solver_ipm_profile::Scope profile_scope(1);
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
      result.original_columns = std::move(original_columns);
      result.fixed_values = std::move(fixed_values);
      return result;
    }
  }
  ConeModel model;
  model.costs = original.costs;
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
    append_bounds(original.row_lower[row], original.row_upper[row], static_cast<Index>(row), row_map[row]);
  }
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    append_bounds(original.column_lower[col], original.column_upper[col], -1, column_bounds[col]);
  }
  model.starts.push_back(0);
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    check_budget(options, start);
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
  for (double cost : model.costs) model.objective_scale = std::max(model.objective_scale, std::abs(cost));
  for (double& cost : model.costs) cost = scaled(cost, 1 / model.objective_scale);
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
        factor_dimension_(model.costs.size()) {
  solver_ipm_profile::Scope profile_scope(2);
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
    const bool normal_augmented = options.ipm.linear_solver == "normal_augmented";
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
                [&options, &start] { check_budget(options, start); });
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
          result.ipm_factor_backend = "normal";
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
                Clock::time_point start, SolveResult& result) {
  solver_ipm_profile::Scope profile_scope(3);
    diagonal_h_ = h;
    regularized_h_.resize(h.size());
    double regularization = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    for (int attempt = 0; attempt < 8; ++attempt) {
      check_budget(options, start);
      std::vector<long double> primal_diagonal(model_.costs.size(), regularization);
      for (std::size_t row = 0; row < h.size(); ++row) {
        const double denominator = finite(static_cast<long double>(h[row]) + regularization);
        if (!(denominator > 0)) throw NumericalFailure("KKT 消元对角必须严格为正");
        regularized_h_[row] = denominator;
        if (singleton_column_[row] >= 0) {
          const auto col = static_cast<std::size_t>(singleton_column_[row]);
          primal_diagonal[col] += static_cast<long double>(singleton_value_[row]) * singleton_value_[row] / denominator;
        } else {
          ax_[diagonal_[static_cast<std::size_t>(retained_index_[row])]] = -denominator;
        }
      }
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
      if (multifrontal_ldl_) {
        const bool success = solver_ipm_profile::factor_call([&] { return multifrontal_ldl_->factor(std::span<const double>(factor_values, ax_.size()),
            [&options, &start] { check_budget(options, start); }); });
        check_budget(options, start);
        *result.ipm_regularized_pivots += multifrontal_ldl_->regularized_pivots();
        if (success) {
          result.ipm_factor_nonzeros = multifrontal_ldl_->factor_nonzeros();
          return;
        }
        regularization *= 10;
        continue;
      }
      if (normal_cholesky_) {
        const bool success = solver_ipm_profile::factor_call([&] { return normal_cholesky_->factor(std::span<const double>(factor_values, ax_.size()),
            [&options, &start] { check_budget(options, start); }); });
        check_budget(options, start);
#ifdef SOLVER_IPM_TRACE
        std::cerr << "正规方程分解 " << *result.ipm_iterations << " 正则 " << regularization
                  << " 成功 " << success << '\n';
#endif
        if (success) {
          result.ipm_factor_nonzeros = normal_cholesky_->factor_nonzeros();
          return;
        }
        regularization *= 10;
        continue;
      }
      if (sparse_lu_) {
        const bool success = solver_ipm_profile::factor_call([&] { return sparse_lu_->factor(std::span<const double>(factor_values, ax_.size())); });
        check_budget(options, start);
#ifdef SOLVER_IPM_TRACE
        std::cerr << "LU 分解 " << *result.ipm_iterations << " 正则 " << regularization
                  << " 成功 " << success << '\n';
#endif
        if (success) {
          result.ipm_factor_nonzeros = sparse_lu_->factor_nonzeros();
          return;
        }
        regularization *= 10;
        continue;
      }
      QDLDL_int positives;
      if (extended_) {
        check_budget(options, start);
        std::copy(factor_values, factor_values + ax_.size(), extended_ax_.begin());
        positives = factor_dimension_ == 0 ? 0 : solver_ipm_profile::factor_call([&] { return detail::solver_qdldl_extended_factor(
            static_cast<QDLDL_int>(factor_dimension_), ap_.data(), ai_.data(), extended_ax_.data(),
            lp_.data(), li_.data(), extended_lx_.data(), extended_d_.data(), extended_inverse_.data(),
            lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), extended_work_.data()); });
        check_budget(options, start);
        const auto finite_vector = [](const auto& values) {
          return std::all_of(values.begin(), values.end(), [](long double value) { return std::isfinite(value); });
        };
        if (positives == static_cast<QDLDL_int>(model_.costs.size()) &&
            finite_vector(extended_d_) && finite_vector(extended_inverse_) && finite_vector(extended_lx_)) return;
        regularization *= 10;
        continue;
      } else if (options.ipm.dynamic_regularization) {
        const auto factored = solver_ipm_profile::factor_call([&] { return detail::factor_regularized_ldl(static_cast<QDLDL_int>(factor_dimension_),
            ap_.data(), ai_.data(), factor_values, lp_.data(), li_.data(), lx_.data(), d_.data(), dinv_.data(),
            lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), fwork_.data(), pivot_signs_.data()); });
        positives = factored.positive_pivots;
        *result.ipm_regularized_pivots += factored.regularized_pivots;
      } else {
        positives = solver_ipm_profile::factor_call([&] { return QDLDL_factor(static_cast<QDLDL_int>(factor_dimension_), ap_.data(), ai_.data(), factor_values,
            lp_.data(), li_.data(), lx_.data(), d_.data(), dinv_.data(), lnz_.data(), etree_.data(),
            bwork_.data(), iwork_.data(), fwork_.data()); });
      }
#ifdef SOLVER_IPM_TRACE
      std::cerr << "分解 " << *result.ipm_iterations << " 正则 " << regularization
                << " 正惯性 " << positives << " 最大逆主元 " << norm_inf(dinv_)
                << " 最大因子 " << norm_inf(lx_) << '\n';
#endif
      check_budget(options, start);
      if (positives == static_cast<QDLDL_int>(model_.costs.size()) &&
          std::isfinite(norm_inf(dinv_)) && std::isfinite(norm_inf(lx_))) return;
      regularization *= 10;
    }
    throw NumericalFailure(multifrontal_ldl_ ? "分块 LDL 分解重试未成功" : normal_cholesky_ ? "正则化正规方程 Cholesky 分解重试未成功" :
        sparse_lu_ ? "正则化 KKT 稀疏 LU 分解重试未成功" :
        "正则化 KKT 分解未得到可靠惯性");
  }

  std::vector<double> solve(const std::vector<double>& rhs, const SolveOptions& options,
                              Clock::time_point start, SolveResult& result,
                              const InexactDirectionContext* inexact = nullptr) const {
  solver_ipm_profile::Scope profile_scope(4);
    check_budget(options, start);
    auto answer = direct(rhs, options, start);
    const double tolerance = 1e-12 + 1e-10 * norm_inf(rhs);
    std::vector<double> residual(rhs.size());
    std::vector<long double> product_cache;
    auto* cached_product = inexact && options.ipm.inexact_prefilter ? &product_cache : nullptr;
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
      if (norm <= tolerance) return answer;
      if (!std::isfinite(norm)) throw NumericalFailure("KKT 迭代改进残差非有限");
      if (accept_backward_error(backward_error, options, start, result)) return answer;
      if (accept_inexact_direction(answer, inexact, options, start, result, cached_product)) return answer;
      if (iteration == 2 || norm >= previous) {
        return krylov_refine(rhs, answer, residual, tolerance, options, start, result, inexact);
      }
      previous = norm;
      const auto correction = direct(residual, options, start);
      ++*result.ipm_refinement_steps;
      for (std::size_t i = 0; i < answer.size(); ++i) answer[i] = finite(static_cast<long double>(answer[i]) + correction[i]);
    }
    return answer;
  }

 private:
  static double dot(const std::vector<double>& left, const std::vector<double>& right) {
    long double sum = 0;
    for (std::size_t i = 0; i < left.size(); ++i) sum += static_cast<long double>(left[i]) * right[i];
    return finite(sum);
  }

  std::vector<double> krylov_refine(const std::vector<double>& rhs,
      std::vector<double> answer, std::vector<double> residual, double tolerance,
      const SolveOptions& options, Clock::time_point start, SolveResult& result,
      const InexactDirectionContext* inexact) const {
  solver_ipm_profile::Scope profile_scope(5);
    // 以已有正则化 LDL 因子作右预条件器，最小化未正则化方程的残差。
    // 内存由固定 Krylov 窗口约束；最终仍检查真实残差而非 Hessenberg 估计。
    constexpr std::size_t window = 32;
    const auto limit = std::min(window, dimension_);
    // 估计值只决定何时构造真实候选；最终仍分别检查两块实际 Newton 缺陷。
    // 任一块预算为零时回到旧触发阈值；窗口末尾仍会检查候选。
    const double candidate_tolerance = inexact ?
        std::max(tolerance, inexact->candidate_tolerance) : tolerance;
    std::vector<long double> product_cache;
    auto* cached_product = inexact && options.ipm.inexact_prefilter ? &product_cache : nullptr;
    std::vector<double> zero(dimension_, 0);
    for (int restart = 0; restart < 3; ++restart) {
      check_budget(options, start);
      const double beta = std::sqrt(dot(residual, residual));
      if (!(beta > 0)) throw NumericalFailure("KKT Krylov 初始残差无效");
      std::vector<std::vector<double>> basis(1, residual), preconditioned;
      for (double& value : basis[0]) value /= beta;
      std::vector<std::vector<double>> upper(limit, std::vector<double>(limit + 1, 0));
      std::vector<double> cosine(limit), sine(limit), transformed(limit + 1, 0);
      transformed[0] = beta;
      std::vector<double> candidate = answer;
      for (std::size_t col = 0; col < limit; ++col) {
        check_budget(options, start);
        preconditioned.push_back(direct(basis[col], options, start));
        ++*result.ipm_refinement_steps;
        std::vector<double> work(dimension_);
        unregularized_residual(zero, preconditioned.back(), work);
        for (double& value : work) value = -value;
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
          if (norm_inf(residual) <= tolerance) return candidate;
          if (accept_backward_error(backward_error, options, start, result)) return candidate;
          if (accept_inexact_direction(candidate, inexact, options, start, result, cached_product)) return candidate;
          if (remaining == 0) break;
        }
      }
      answer = std::move(candidate);
    }
    throw NumericalFailure("KKT Krylov 改进未达到原方程精度");
  }

  void solve_factored(std::vector<double>& rhs, const SolveOptions& options,
      Clock::time_point start) const {
  solver_ipm_profile::Scope profile_scope(7);
    if (multifrontal_ldl_) {
      const bool success = multifrontal_ldl_->solve(rhs,
          [&options, &start] { check_budget(options, start); });
      if (!success) throw NumericalFailure("分块 LDL 三角求解失败");
    } else if (normal_cholesky_) {
      check_budget(options, start);
      const bool success = normal_cholesky_->solve(rhs,
          [&options, &start] { check_budget(options, start); });
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
      Clock::time_point start) const {
  solver_ipm_profile::Scope profile_scope(6);
    if (factor_dimension_ == dimension_) {
      std::vector<double> permuted(dimension_);
      // 没有消元时不需要构造扩展精度的约化右端；均衡与置换可在同一次扫描完成。
      for (std::size_t i = 0; i < dimension_; ++i) {
        const auto position = inverse_[i];
        permuted[position] = equilibration_.empty() ? rhs[i] : scaled(rhs[i], equilibration_[position]);
      }
      solve_factored(permuted, options, start);
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
    solve_factored(permuted, options, start);
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
  solver_ipm_profile::Scope profile_scope(8);
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
  solver_ipm_profile::Scope profile_scope(13);
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
  std::size_t backward_error_nonzeros_ = 0;
  std::vector<std::size_t> inverse_, diagonal_;
  std::vector<Index> retained_index_, singleton_column_;
  std::vector<std::int8_t> pivot_signs_;
  std::vector<QDLDL_int> ap_, ai_, lp_, li_, etree_, lnz_, iwork_;
  std::vector<QDLDL_bool> bwork_;
  std::vector<double> ax_, lx_, d_, dinv_, fwork_, diagonal_h_, regularized_h_, singleton_value_;
  std::vector<double> off_diagonal_max_, equilibration_, factor_values_;
  bool extended_ = false;
  std::vector<long double> extended_ax_, extended_lx_, extended_d_, extended_inverse_, extended_work_;
  std::unique_ptr<detail::SparseLu> sparse_lu_;
  std::unique_ptr<detail::NormalCholesky> normal_cholesky_;
  std::unique_ptr<detail::MultifrontalLdl> multifrontal_ldl_;
  detail::IpmInexactPrefilter inexact_prefilter_;
};

bool assess(const ConeModel& model, const Point& point, const SolveOptions& options,
               SolveResult& result) {
  solver_ipm_profile::Scope profile_scope(9);
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
  for (std::size_t row = 0; row < y.size(); ++row) y[row] = finite(dual[row]);
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
  solver_ipm_profile::Scope profile_scope(10);
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

void initialize(const ConeModel& model, Point& point, KktFactor& factor,
                  const SolveOptions& options, Clock::time_point start, SolveResult& result) {
  solver_ipm_profile::Scope profile_scope(11);
  const auto n = point.x.size();
  const auto m = point.s.size();
  // 同一个增广矩阵分别求原始最小残差点和最小范数对偶点。
  factor.factor(std::vector<double>(m, 1), options, start, result);
  std::vector<double> rhs(n + m, 0);
  std::copy(model.rhs.begin(), model.rhs.end(), rhs.begin() + static_cast<std::ptrdiff_t>(n));
  const auto primal = factor.solve(rhs, options, start, result);
  std::copy(primal.begin(), primal.begin() + static_cast<std::ptrdiff_t>(n), point.x.begin());
  std::fill(rhs.begin(), rhs.end(), 0);
  for (std::size_t col = 0; col < n; ++col) rhs[col] = -model.costs[col];
  const auto dual = factor.solve(rhs, options, start, result);
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
  solver_ipm_profile::Scope profile_scope(12);
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
  if (assess(model, point, options, result)) return;
  KktFactor factor(model, options, start, result);
  std::vector<double> primal_residual, dual_residual;
  std::vector<double> diagonal(point.s.size()), h(point.s.size());
  result.status = "iteration_limit";
  try {
    initialize(model, point, factor, options, start, result);
    if (assess(model, point, options, result)) return;
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
      factor.factor(diagonal, options, start, result);
      const auto affine = direction(model, point, primal_residual, dual_residual, h,
                                     factor, options, start, result);
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
      const double sigma = ratio * ratio * ratio;
      for (std::size_t row = 0; row < point.s.size(); ++row) {
        if (model.inequality[row]) {
          h[row] = finite(-static_cast<long double>(point.s[row]) * point.z[row] + sigma * mu -
                             static_cast<long double>(affine.s[row]) * affine.z[row]);
        }
      }
      // 预测和校正只更新右端项，共享同一次数值分解。
      auto corrected = direction(model, point, primal_residual, dual_residual, h,
                                        factor, options, start, result);
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
      if (assess(model, point, options, result)) return;
      if (iteration == options.ipm.max_iterations) break;
    }
  } catch (const BudgetExpired&) {
    result.status = "time_limit";
    result.message = "内点求解已达到时间预算";
  } catch (const NumericalFailure& error) {
    result.status = "numerical_error";
    result.message = error.what();
  }
  assess(model, point, options, result);
}

void validate_options(const SolveOptions& options) {
  if (options.ipm.linear_solver != "qdldl" && options.ipm.linear_solver != "sparselu" &&
      options.ipm.linear_solver != "normal" && options.ipm.linear_solver != "qdldl_extended" &&
      options.ipm.linear_solver != "factorhighs" && options.ipm.linear_solver != "normal_augmented") {
    throw std::invalid_argument("内点法线性求解器必须为 qdldl、sparselu、normal、qdldl_extended、factorhighs 或 normal_augmented");
  }
  if (options.ipm.linear_solver == "qdldl_extended" &&
      std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits) {
    throw std::invalid_argument("本平台 long double 未提供比 double 更高的有效精度");
  }
  if (options.ipm.linear_solver != "qdldl" && options.ipm.dynamic_regularization) {
    throw std::invalid_argument("局部动态主元正则仅支持 QDLDL");
  }
  if ((options.ipm.linear_solver == "normal" || options.ipm.linear_solver == "normal_augmented") &&
      !options.ipm.eliminate_singletons) {
    throw std::invalid_argument("正规方程实验要求启用单非零不等式行消元");
  }
  if (options.ipm.central_corrections < 0 || options.ipm.central_corrections > 1) {
    throw std::invalid_argument("额外中心校正次数当前仅支持 0 或 1");
  }
  if (options.ipm.linear_accuracy != "strict" && options.ipm.linear_accuracy != "backward" &&
      options.ipm.linear_accuracy != "inexact") {
    throw std::invalid_argument("内点法线性精度模式必须为 strict、backward 或 inexact");
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

std::string ipm_version() { return "ipm-cpu-0.19.0"; }

SolveResult solve_ipm(const Model& model, const SolveOptions& options) {
  solver_ipm_profile::Report profile_report;
  solver_ipm_profile::Scope profile_scope(0);
  const auto start = Clock::now();
  SolveResult result;
  result.backend = "ipm";
  result.backend_version = ipm_version();
  result.model = model;
  result.ipm_iterations = 0;
  result.ipm_factorizations = 0;
  result.ipm_refinement_steps = 0;
  result.ipm_structural_fallbacks = 0;
  result.ipm_eliminated_fixed_variables = 0;
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
      run_ipm(working, options, start, result);
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
