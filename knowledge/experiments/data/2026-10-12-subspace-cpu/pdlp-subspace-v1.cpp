#include "solver/pdlp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace solver {
namespace {

using Clock = std::chrono::steady_clock;

struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct BudgetExpired {};

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double finite_double(long double value) {
  const double converted = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(converted)) {
    throw NumericalFailure("自主迭代或缩放出现非有限数，无法继续可靠计算");
  }
  return converted;
}

double scale_data(double value, long double factor) {
  if (!std::isfinite(value)) return value;
  const double scaled = finite_double(static_cast<long double>(value) * factor);
  if (value != 0 && scaled == 0) {
    throw NumericalFailure("模型缩放导致有限非零数据下溢为零，已停止求解");
  }
  return scaled;
}

void validate_options(const SolveOptions& options) {
  if (options.threads != 1) {
    throw std::invalid_argument("自主 PDHG 原型仅支持单线程，请设置 threads=1");
  }
  if (!std::isfinite(options.time_limit) || options.time_limit < 0 ||
      options.pdlp.max_iterations < 0 || options.pdlp.check_interval < 1 ||
      options.pdlp.scaling_passes < 0 || options.pdlp.scaling_passes > 20 ||
      options.pdlp.history_size < 0 || options.pdlp.history_size > 32) {
    throw std::invalid_argument("自主求解的预算、检查间隔或缩放轮数无效");
  }
  if (options.pdlp.mode != "baseline" && options.pdlp.mode != "adaptive" &&
      options.pdlp.mode != "subspace") {
    throw std::invalid_argument("自主算法模式必须为 baseline、adaptive 或 subspace");
  }
  const auto& t = options.tolerances;
  const auto& o = options.pdlp.optimality;
  for (double value : {t.feasibility_absolute, t.feasibility_relative,
                       t.integrality, t.objective_absolute, t.objective_relative,
                       o.dual_absolute, o.dual_relative,
                       o.gap_absolute, o.gap_relative}) {
    if (!std::isfinite(value) || value < 0) {
      throw std::invalid_argument("自主求解的验证容差必须是有限非负数");
    }
  }
}

struct WorkingModel {
  Model model;
  std::vector<double> row_scale;
  std::vector<double> column_scale;
  double objective_scale = 1;
};

double ruiz_factor(double norm) {
  if (norm == 0) return 1;
  return std::clamp(1.0 / std::sqrt(norm), 1e-4, 1e4);
}

WorkingModel prepare_model(const Model& original, const SolveOptions& options,
                            Clock::time_point start) {
  WorkingModel working{original, std::vector<double>(original.row_lower.size(), 1),
                       std::vector<double>(original.costs.size(), 1), 1};
  auto& model = working.model;
  const double sense = original.sense == ObjectiveSense::minimize ? 1 : -1;
  for (double& cost : model.costs) cost *= sense;
  // 常数不参与迭代；保留在 result.model 中并只由原坐标验证器计算。
  model.objective_offset = 0;
  model.sense = ObjectiveSense::minimize;
  std::vector<double> row_norm(model.row_lower.size());
  for (int pass = 0; pass < options.pdlp.scaling_passes; ++pass) {
    if (seconds_since(start) >= options.time_limit) throw BudgetExpired{};
    std::fill(row_norm.begin(), row_norm.end(), 0);
    for (std::size_t k = 0; k < model.values.size(); ++k) {
      const auto row = static_cast<std::size_t>(model.row_index[k]);
      row_norm[row] = std::max(row_norm[row], std::abs(model.values[k]));
    }
    for (std::size_t row = 0; row < row_norm.size(); ++row) {
      row_norm[row] = ruiz_factor(row_norm[row]);
      working.row_scale[row] = scale_data(working.row_scale[row], row_norm[row]);
      model.row_lower[row] = scale_data(model.row_lower[row], row_norm[row]);
      model.row_upper[row] = scale_data(model.row_upper[row], row_norm[row]);
    }
    for (std::size_t k = 0; k < model.values.size(); ++k) {
      model.values[k] = scale_data(model.values[k],
          row_norm[static_cast<std::size_t>(model.row_index[k])]);
    }
    for (std::size_t col = 0; col < model.costs.size(); ++col) {
      double norm = 0;
      for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
        norm = std::max(norm, std::abs(model.values[static_cast<std::size_t>(k)]));
      }
      const double factor = ruiz_factor(norm);
      working.column_scale[col] = scale_data(working.column_scale[col], factor);
      model.costs[col] = scale_data(model.costs[col], factor);
      model.column_lower[col] = scale_data(model.column_lower[col], 1.0L / factor);
      model.column_upper[col] = scale_data(model.column_upper[col], 1.0L / factor);
      for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
        auto& value = model.values[static_cast<std::size_t>(k)];
        value = scale_data(value, factor);
      }
    }
  }
  if (options.pdlp.mode != "baseline" && options.pdlp.scaling_passes > 0) {
    // Ruiz 后补一次行列绝对和缩放，保持稀疏结构；固定步长基线不变。
    std::vector<long double> row_sum(model.row_lower.size(), 0);
    std::vector<long double> column_sum(model.costs.size(), 0);
    for (std::size_t col = 0; col < model.costs.size(); ++col) {
      for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
        const auto pos = static_cast<std::size_t>(k);
        const long double value = std::abs(model.values[pos]);
        column_sum[col] += value;
        row_sum[static_cast<std::size_t>(model.row_index[pos])] += value;
      }
    }
    for (std::size_t row = 0; row < row_sum.size(); ++row) {
      if (!std::isfinite(row_sum[row])) throw NumericalFailure("行均衡求和溢出");
      row_sum[row] = row_sum[row] > 0 ? 1.0L / std::sqrt(row_sum[row]) : 1.0L;
      working.row_scale[row] = scale_data(working.row_scale[row], row_sum[row]);
      model.row_lower[row] = scale_data(model.row_lower[row], row_sum[row]);
      model.row_upper[row] = scale_data(model.row_upper[row], row_sum[row]);
    }
    for (std::size_t col = 0; col < column_sum.size(); ++col) {
      if (!std::isfinite(column_sum[col])) throw NumericalFailure("列均衡求和溢出");
      const long double factor = column_sum[col] > 0 ? 1.0L / std::sqrt(column_sum[col]) : 1.0L;
      working.column_scale[col] = scale_data(working.column_scale[col], factor);
      model.costs[col] = scale_data(model.costs[col], factor);
      model.column_lower[col] = scale_data(model.column_lower[col], 1.0L / factor);
      model.column_upper[col] = scale_data(model.column_upper[col], 1.0L / factor);
      for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
        const auto pos = static_cast<std::size_t>(k);
        model.values[pos] = scale_data(model.values[pos],
            factor * row_sum[static_cast<std::size_t>(model.row_index[pos])]);
      }
    }
  }
  for (double cost : model.costs) {
    working.objective_scale = std::max(working.objective_scale, std::abs(cost));
  }
  const long double inverse = 1.0L / working.objective_scale;
  for (double& cost : model.costs) cost = scale_data(cost, inverse);
  return working;
}

void make_steps(const Model& model, std::vector<double>& primal_step,
                 std::vector<double>& dual_step) {
  std::vector<long double> row_sum(model.row_lower.size(), 0);
  primal_step.resize(model.costs.size());
  dual_step.resize(model.row_lower.size());
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    long double column_sum = 0;
    for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      const long double value = std::abs(model.values[pos]);
      column_sum += value;
      row_sum[static_cast<std::size_t>(model.row_index[pos])] += value;
    }
    if (!std::isfinite(column_sum)) throw NumericalFailure("列预条件求和溢出");
    primal_step[col] = finite_double(0.9L / std::max(1.0L, column_sum));
    if (primal_step[col] == 0) throw NumericalFailure("原始步长下溢为零");
  }
  for (std::size_t row = 0; row < row_sum.size(); ++row) {
    if (!std::isfinite(row_sum[row])) throw NumericalFailure("行预条件求和溢出");
    dual_step[row] = finite_double(0.9L / std::max(1.0L, row_sum[row]));
    if (dual_step[row] == 0) throw NumericalFailure("对偶步长下溢为零");
  }
  // 行列绝对和的 Schur 界保证 ||sqrt(Sigma) A sqrt(Tau)|| <= 0.9。
}

double project(long double value, double lower, double upper) {
  if (!std::isfinite(value)) throw NumericalFailure("原始更新出现非有限数");
  return finite_double(std::max(static_cast<long double>(lower),
                                std::min(static_cast<long double>(upper), value)));
}

void iterate(const Model& model, const std::vector<double>& primal_step,
              const std::vector<double>& dual_step, std::vector<double>& x,
              std::vector<double>& y, std::vector<double>& next_x,
              std::vector<long double>& row_activity) {
  std::fill(row_activity.begin(), row_activity.end(), 0);
  for (std::size_t col = 0; col < x.size(); ++col) {
    long double aty = 0;
    for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      aty += static_cast<long double>(model.values[pos]) *
             y[static_cast<std::size_t>(model.row_index[pos])];
    }
    const long double proposed = x[col] - static_cast<long double>(primal_step[col]) *
        (static_cast<long double>(model.costs[col]) - aty);
    next_x[col] = project(proposed, model.column_lower[col], model.column_upper[col]);
    const long double extrapolated = 2.0L * next_x[col] - x[col];
    for (auto k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      row_activity[static_cast<std::size_t>(model.row_index[pos])] +=
          static_cast<long double>(model.values[pos]) * extrapolated;
    }
  }
  for (std::size_t row = 0; row < y.size(); ++row) {
    const long double step = dual_step[row];
    const long double q = y[row] - step * row_activity[row];
    if (!std::isfinite(q)) throw NumericalFailure("对偶更新出现非有限数");
    // 等价于 q + sigma * project(-q/sigma)，避免无穷边界乘法和大数相消。
    const long double at_lower = std::isfinite(model.row_lower[row])
        ? q + step * model.row_lower[row] : -std::numeric_limits<long double>::infinity();
    const long double at_upper = std::isfinite(model.row_upper[row])
        ? q + step * model.row_upper[row] : std::numeric_limits<long double>::infinity();
    if (at_lower > 0) y[row] = finite_double(at_lower);
    else if (at_upper < 0) y[row] = finite_double(at_upper);
    else y[row] = 0;
  }
  x.swap(next_x);
}

void update_average(std::vector<double>& average, const std::vector<double>& point,
                     std::int64_t count) {
  for (std::size_t i = 0; i < point.size(); ++i) {
    average[i] = finite_double(static_cast<long double>(average[i]) +
        (static_cast<long double>(point[i]) - average[i]) / count);
  }
}

double quality(const LpOptimality& check) {
  return std::max(check.max_dual_residual,
      check.relative_objective_gap.value_or(std::numeric_limits<double>::infinity()));
}

void copy_bound_and_gap(SolveResult& result, const SolveOptions& options) {
  result.best_bound = result.lp_optimality->corrected_dual_bound;
  result.absolute_gap.reset();
  result.relative_gap.reset();
  if (!result.has_solution || !result.objective || !result.best_bound) return;
  const long double objective = *result.objective;
  const long double bound = *result.best_bound;
  const long double scale = std::max({1.0L, std::abs(objective), std::abs(bound)});
  const long double gap = result.model.sense == ObjectiveSense::minimize
      ? objective - bound : bound - objective;
  const long double tolerance = options.pdlp.optimality.gap_absolute +
      static_cast<long double>(options.pdlp.optimality.gap_relative) * scale;
  if (!std::isfinite(gap) || gap < -tolerance) {
    result.best_bound.reset();
    result.absolute_gap.reset();
    result.relative_gap.reset();
    if (!std::isfinite(gap)) {
      throw NumericalFailure("自主候选与对偶界的间隙溢出");
    }
    // 容差内的原始违反仍可能被大乘子放大；中间点继续迭代改善。
    // 不发布与该候选矛盾的界，最优仍须完整间隙检查通过。
    return;
  }
  const long double nonnegative_gap = std::max(0.0L, gap);
  const double absolute = static_cast<double>(nonnegative_gap);
  if (std::isfinite(absolute)) result.absolute_gap = absolute;
  result.relative_gap = finite_double(nonnegative_gap / scale);
}

struct AssessmentBuffers {
  std::vector<double> original_x;
  std::vector<double> original_y;
};

bool assess(const WorkingModel& working, const std::vector<double>& x,
              const std::vector<double>& y, const SolveOptions& options,
              SolveResult& result, AssessmentBuffers* buffers = nullptr) {
  AssessmentBuffers local;
  auto& storage = buffers ? *buffers : local;
  auto& original_x = storage.original_x;
  auto& original_y = storage.original_y;
  original_x.resize(x.size());
  original_y.resize(y.size());
  for (std::size_t col = 0; col < x.size(); ++col) {
    original_x[col] = scale_data(x[col], working.column_scale[col]);
  }
  for (std::size_t row = 0; row < y.size(); ++row) {
    original_y[row] = scale_data(y[row],
        static_cast<long double>(working.row_scale[row]) * working.objective_scale);
  }
  const auto start = Clock::now();
  if (!project_lp_row_dual_domain(result.model, original_y)) {
    throw NumericalFailure("原坐标行对偶维度或行界无效");
  }
  auto checks = verify_lp_solution(result.model, original_x, original_y,
                                    options.tolerances, options.pdlp.optimality);
  auto primal = std::move(checks.primal);
  auto optimality = std::move(checks.optimality);
  result.verification_seconds += seconds_since(start);
  const bool feasible = primal.valid_input && primal.feasible &&
                        primal.objective_consistent && primal.objective.has_value();
  bool replace = !result.has_solution;
  if (feasible && result.has_solution) {
    const double sense = result.model.sense == ObjectiveSense::minimize ? 1 : -1;
    replace = sense * *primal.objective < sense * *result.objective ||
        (*primal.objective == *result.objective &&
         quality(optimality) < quality(*result.lp_optimality));
  }
  if (optimality.numerically_optimal) replace = true;
  if (replace) {
    result.verification = std::move(primal);
    result.lp_optimality = std::move(optimality);
    result.row_dual = original_y;
    result.has_solution = feasible;
    result.solution = feasible ? original_x : std::vector<double>{};
    result.objective = feasible ? result.verification.objective : std::nullopt;
    result.verification_level = feasible ? "primal_checked" : "residuals_checked";
    copy_bound_and_gap(result, options);
  }
  if (result.has_solution && result.lp_optimality &&
      result.lp_optimality->numerically_optimal) {
    result.status = "optimal";
    result.verification_level = "primal_dual_residuals_checked";
    return true;
  }
  return false;
}

// 热循环只依赖以下两个 FP64 稀疏算子；独立验算仍使用原模型长累加。
// 未来设备后端可替换算子而保持步长、重启和原坐标检查协议。
class CscOperator {
 public:
  CscOperator(const Model& model, std::int64_t& passes) : model_(model), passes_(passes) {}

  void multiply(const std::vector<double>& x, std::vector<double>& ax) const {
    if (passes_ < std::numeric_limits<std::int64_t>::max()) ++passes_;
    std::fill(ax.begin(), ax.end(), 0.0);
    for (std::size_t col = 0; col < x.size(); ++col) {
      const double value = x[col];
      for (auto k = model_.column_start[col]; k < model_.column_start[col + 1]; ++k) {
        const auto pos = static_cast<std::size_t>(k);
        ax[static_cast<std::size_t>(model_.row_index[pos])] += model_.values[pos] * value;
      }
    }
  }

  void transpose_multiply(const std::vector<double>& y, std::vector<double>& aty) const {
    if (passes_ < std::numeric_limits<std::int64_t>::max()) ++passes_;
    for (std::size_t col = 0; col < aty.size(); ++col) {
      double sum = 0.0;
      for (auto k = model_.column_start[col]; k < model_.column_start[col + 1]; ++k) {
        const auto pos = static_cast<std::size_t>(k);
        sum += model_.values[pos] * y[static_cast<std::size_t>(model_.row_index[pos])];
      }
      aty[col] = sum;
    }
  }

 private:
  const Model& model_;
  std::int64_t& passes_;
};

struct AdaptivePoint {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> ax;
  std::vector<double> aty;

  AdaptivePoint(std::size_t columns, std::size_t rows)
      : x(columns, 0), y(rows, 0), ax(rows, 0), aty(columns, 0) {}
};

bool all_finite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}

double dual_prox(double q, double sigma, double lower, double upper) {
  const double at_lower = std::isfinite(lower) ? q + sigma * lower
      : -std::numeric_limits<double>::infinity();
  const double at_upper = std::isfinite(upper) ? q + sigma * upper
      : std::numeric_limits<double>::infinity();
  if (at_lower > 0) return at_lower;
  if (at_upper < 0) return at_upper;
  return 0;
}

double initial_weight(const Model& model) {
  double cost_norm = 0;
  double rhs_norm = 0;
  double fallback_rhs_norm = 0;
  for (double cost : model.costs) cost_norm = std::hypot(cost_norm, cost);
  for (std::size_t row = 0; row < model.row_lower.size(); ++row) {
    const double nearest = std::clamp(0.0, model.row_lower[row], model.row_upper[row]);
    rhs_norm = std::hypot(rhs_norm, nearest);
    double largest = 0;
    if (std::isfinite(model.row_lower[row])) largest = std::abs(model.row_lower[row]);
    if (std::isfinite(model.row_upper[row])) largest = std::max(largest, std::abs(model.row_upper[row]));
    fallback_rhs_norm = std::hypot(fallback_rhs_norm, largest);
  }
  if (rhs_norm == 0) rhs_norm = fallback_rhs_norm;
  if (!std::isfinite(cost_norm) || !std::isfinite(rhs_norm)) {
    throw NumericalFailure("初始化原始对偶权重时范数溢出");
  }
  if (cost_norm == 0 || rhs_norm == 0) return 1;
  return std::clamp(finite_double(static_cast<long double>(cost_norm) / rhs_norm), 1e-100, 1e100);
}

double initial_step(const Model& model) {
  std::vector<double> row_sum(model.row_lower.size(), 0);
  for (std::size_t pos = 0; pos < model.values.size(); ++pos) {
    row_sum[static_cast<std::size_t>(model.row_index[pos])] += std::abs(model.values[pos]);
  }
  double largest = 0;
  for (double value : row_sum) largest = std::max(largest, value);
  if (!std::isfinite(largest)) throw NumericalFailure("初始步长范数溢出");
  const double eta = largest > 0 ? 1.0 / largest : 1.0;
  if (!(eta > 0) || !std::isfinite(eta)) throw NumericalFailure("初始步长不可表示");
  return eta;
}

// 缩放坐标的投影 KKT 残差，用于重启选择；这是工程变体，非论文的归一化间隙。
// 它包含原始可行性与互补条件，但绝不替代原坐标的终止检查。
double restart_merit(const Model& model, const AdaptivePoint& point, double omega) {
  double primal_squared = 0;
  double dual_squared = 0;
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const double proposed = point.x[col] - (model.costs[col] - point.aty[col]) / omega;
    if (!std::isfinite(proposed)) return std::numeric_limits<double>::infinity();
    const double residual = point.x[col] -
        std::clamp(proposed, model.column_lower[col], model.column_upper[col]);
    primal_squared += residual * residual;
  }
  for (std::size_t row = 0; row < point.y.size(); ++row) {
    const double q = point.y[row] - omega * point.ax[row];
    if (!std::isfinite(q)) return std::numeric_limits<double>::infinity();
    const double residual = point.y[row] -
        dual_prox(q, omega, model.row_lower[row], model.row_upper[row]);
    dual_squared += residual * residual;
  }
  const double merit = std::sqrt(omega * primal_squared + dual_squared / omega);
  return std::isfinite(merit) ? merit : std::numeric_limits<double>::infinity();
}

bool adaptive_step(const Model& model, const CscOperator& matrix,
                     AdaptivePoint& point, AdaptivePoint& trial, double omega,
                     double& eta, double& accepted_eta, std::uint64_t& attempts,
                     const SolveOptions& options, Clock::time_point start) {
  // 绝对交叉项与作者参考实现一致。每次拒绝都保留原点和加权均值。
  for (int backtrack = 0; backtrack < 100; ++backtrack) {
    if (seconds_since(start) >= options.time_limit) throw BudgetExpired{};
    ++attempts;
    const double tau = eta / omega;
    const double sigma = eta * omega;
    bool valid = std::isfinite(tau) && std::isfinite(sigma) && tau > 0 && sigma > 0;
    if (valid) {
      for (std::size_t col = 0; col < point.x.size(); ++col) {
        const double proposed = point.x[col] - tau * (model.costs[col] - point.aty[col]);
        if (!std::isfinite(proposed)) { valid = false; break; }
        trial.x[col] = std::clamp(proposed, model.column_lower[col], model.column_upper[col]);
      }
    }
    if (valid) {
      matrix.multiply(trial.x, trial.ax);
      valid = all_finite(trial.ax);
    }
    if (valid) {
      for (std::size_t row = 0; row < point.y.size(); ++row) {
        const double q = point.y[row] - sigma * (2.0 * trial.ax[row] - point.ax[row]);
        if (!std::isfinite(q)) { valid = false; break; }
        trial.y[row] = dual_prox(q, sigma, model.row_lower[row], model.row_upper[row]);
        if (!std::isfinite(trial.y[row])) { valid = false; break; }
      }
    }
    if (valid) {
      matrix.transpose_multiply(trial.y, trial.aty);
      valid = all_finite(trial.aty);
    }
    double delta_x_squared = 0;
    double delta_y_squared = 0;
    double interaction = 0;
    if (valid) {
      for (std::size_t col = 0; col < point.x.size(); ++col) {
        const double delta = trial.x[col] - point.x[col];
        delta_x_squared += delta * delta;
        interaction += delta * (trial.aty[col] - point.aty[col]);
      }
      for (std::size_t row = 0; row < point.y.size(); ++row) {
        const double delta = trial.y[row] - point.y[row];
        delta_y_squared += delta * delta;
      }
    }
    const double movement = 0.5 * (omega * delta_x_squared + delta_y_squared / omega);
    interaction = std::abs(interaction);
    valid = valid && std::isfinite(movement) && std::isfinite(interaction);
    if (!valid) {
      eta *= 0.5;
    } else {
      if (movement == 0) return false;
      const double limit = interaction > 0 ? movement / interaction
                                          : std::numeric_limits<double>::infinity();
      const double counter = static_cast<double>(attempts) + 1;
      double next_eta = std::min((1.0 - std::pow(counter, -0.3)) * limit,
                                (1.0 + std::pow(counter, -0.6)) * eta);
      if (!std::isfinite(next_eta)) next_eta = eta;
      if (eta <= limit) {
        accepted_eta = eta;
        eta = next_eta;
        std::swap(point, trial);
        return true;
      }
      eta = next_eta;
    }
    if (!(eta > 0) || !std::isfinite(eta)) throw NumericalFailure("自适应线搜索步长不可表示");
  }
  throw NumericalFailure("自适应线搜索连续回退后仍未满足能量判据");
}

void weighted_average(AdaptivePoint& average, const AdaptivePoint& point,
                        double eta, double& total_weight) {
  const double next_weight = total_weight + eta;
  if (!std::isfinite(next_weight) || !(next_weight > 0)) {
    throw NumericalFailure("步长加权均值的累计权重溢出");
  }
  const double ratio = eta / next_weight;
  for (std::size_t i = 0; i < point.x.size(); ++i) {
    average.x[i] = (1.0 - ratio) * average.x[i] + ratio * point.x[i];
  }
  for (std::size_t i = 0; i < point.y.size(); ++i) {
    average.y[i] = (1.0 - ratio) * average.y[i] + ratio * point.y[i];
  }
  total_weight = next_weight;
}

double updated_weight(const AdaptivePoint& current, const AdaptivePoint& previous,
                         double omega) {
  double dx = 0;
  double dy = 0;
  for (std::size_t i = 0; i < current.x.size(); ++i) dx = std::hypot(dx, current.x[i] - previous.x[i]);
  for (std::size_t i = 0; i < current.y.size(); ++i) dy = std::hypot(dy, current.y[i] - previous.y[i]);
  if (!std::isfinite(dx) || !std::isfinite(dy)) throw NumericalFailure("重启位移范数溢出");
  if (dx <= 1e-16 || dy <= 1e-16) return omega;
  return std::clamp(std::exp(0.5 * (std::log(dy) - std::log(dx) + std::log(omega))),
                    1e-100, 1e100);
}

bool residual_vector(const Model& model, const AdaptivePoint& point, double omega,
                       std::vector<double>& residual) {
  residual.resize(point.x.size() + point.y.size());
  const double root_weight = std::sqrt(omega);
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const double proposed = point.x[col] - (model.costs[col] - point.aty[col]) / omega;
    if (!std::isfinite(proposed)) return false;
    residual[col] = root_weight * (point.x[col] -
        std::clamp(proposed, model.column_lower[col], model.column_upper[col]));
  }
  for (std::size_t row = 0; row < point.y.size(); ++row) {
    const double q = point.y[row] - omega * point.ax[row];
    if (!std::isfinite(q)) return false;
    residual[point.x.size() + row] = (point.y[row] -
        dual_prox(q, omega, model.row_lower[row], model.row_upper[row])) / root_weight;
  }
  return all_finite(residual);
}

double vector_dot(const std::vector<double>& a, const std::vector<double>& b) {
  double value = 0;
  for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
  return value;
}

class ResidualSubspace {
 public:
  explicit ResidualSubspace(std::size_t capacity) : capacity_(capacity) {
    history_.reserve(capacity);
  }

  bool candidate(const Model& model, const CscOperator& matrix,
                    const AdaptivePoint& current, double omega, double merit_limit,
                    const SolveOptions& options, Clock::time_point start,
                    AdaptivePoint& output, std::int64_t& attempts) const {
    if (history_.empty() || capacity_ == 0) return false;
    if (attempts < std::numeric_limits<std::int64_t>::max()) ++attempts;
    const auto check_budget = [&]() {
      if (seconds_since(start) >= options.time_limit) throw BudgetExpired{};
    };
    check_budget();
    std::vector<double> residual;
    if (!residual_vector(model, current, omega, residual)) return false;
    const std::size_t count = history_.size();
    const std::size_t dimension = residual.size();
    std::vector<std::vector<double>> orthogonal;
    std::vector<std::size_t> selected;
    std::vector<double> scales;
    std::vector<double> triangular(count * count, 0);
    std::vector<double> rhs;
    std::vector<double> work;
    orthogonal.reserve(count);
    selected.reserve(count);
    scales.reserve(count);
    rhs.reserve(count);
    // 所有历史残差在当前权重下重新求值，不把变化的 PDHG 步长当成固定映射。
    // 列归一化和两遍正交化避免正规方程平方放大条件数。
    const double rank_tolerance = std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(dimension, count));
    for (std::size_t old = count; old-- > 0;) {
      check_budget();
      if (!residual_vector(model, history_[old], omega, work)) continue;
      double scale = 0;
      for (std::size_t i = 0; i < dimension; ++i) {
        work[i] -= residual[i];
        scale = std::max(scale, std::abs(work[i]));
      }
      if (!(scale > 0) || !std::isfinite(scale)) continue;
      for (double& value : work) value /= scale;
      const double original_norm = std::sqrt(vector_dot(work, work));
      if (!(original_norm > 0) || !std::isfinite(original_norm)) continue;
      scale *= original_norm;
      if (!std::isfinite(scale)) continue;
      for (double& value : work) value /= original_norm;
      const std::size_t rank = orthogonal.size();
      for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t j = 0; j < rank; ++j) {
          const double dot = vector_dot(orthogonal[j], work);
          triangular[j * count + rank] += dot;
          for (std::size_t i = 0; i < dimension; ++i) work[i] -= dot * orthogonal[j][i];
        }
      }
      const double norm = std::sqrt(vector_dot(work, work));
      if (!(norm > rank_tolerance) || !std::isfinite(norm)) {
        for (std::size_t j = 0; j < rank; ++j) triangular[j * count + rank] = 0;
        continue;
      }
      for (double& value : work) value /= norm;
      triangular[rank * count + rank] = norm;
      rhs.push_back(vector_dot(work, residual));
      selected.push_back(old);
      scales.push_back(scale);
      orthogonal.push_back(work);
    }
    const std::size_t rank = selected.size();
    if (rank == 0) return false;
    check_budget();
    std::vector<double> coefficient(rank);
    for (std::size_t i = rank; i-- > 0;) {
      double value = rhs[i];
      for (std::size_t j = i + 1; j < rank; ++j) value -= triangular[i * count + j] * coefficient[j];
      coefficient[i] = value / triangular[i * count + i];
    }
    double coefficient_norm = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      coefficient[i] /= scales[i];
      coefficient_norm = std::hypot(coefficient_norm, coefficient[i]);
    }
    // 机器精度护栏只会拒绝不可靠候选；被拒绝时保留普通 PDHG 轨迹。
    if (!std::isfinite(coefficient_norm) ||
        coefficient_norm > 1.0 / std::sqrt(std::numeric_limits<double>::epsilon())) return false;
    output = current;
    for (std::size_t j = 0; j < rank; ++j) {
      check_budget();
      const auto& old = history_[selected[j]];
      for (std::size_t col = 0; col < output.x.size(); ++col) {
        output.x[col] -= coefficient[j] * (old.x[col] - current.x[col]);
      }
      for (std::size_t row = 0; row < output.y.size(); ++row) {
        output.y[row] -= coefficient[j] * (old.y[row] - current.y[row]);
      }
    }
    if (!all_finite(output.x) || !all_finite(output.y)) return false;
    for (std::size_t col = 0; col < output.x.size(); ++col) {
      output.x[col] = std::clamp(output.x[col], model.column_lower[col], model.column_upper[col]);
    }
    if (!project_lp_row_dual_domain(model, output.y)) return false;
    check_budget();
    // 外推后投影会改变坐标，必须重新计算乘积，不能使用投影前的仿射缓存。
    matrix.multiply(output.x, output.ax);
    check_budget();
    matrix.transpose_multiply(output.y, output.aty);
    check_budget();
    if (!all_finite(output.ax) || !all_finite(output.aty)) return false;
    // 这里只保护真实投影残差，不能替代原模型最优性验算或全局收敛证明。
    return restart_merit(model, output, omega) < merit_limit;
  }

  void remember(const AdaptivePoint& point) {
    if (capacity_ == 0) return;
    if (history_.size() == capacity_) history_.erase(history_.begin());
    history_.push_back(point);
  }

 private:
  std::size_t capacity_;
  std::vector<AdaptivePoint> history_;
};

bool screening_candidate(const WorkingModel& working, const Model& original,
                            const AdaptivePoint& point, const SolveOptions& options) {
  // 只决定是否提前进行昂贵验算；缓存经历 FP64 缩放与累加，不能作为最优证据。
  // 100 倍窗口减少漏筛；每 1000 步及所有迭代出口仍强制原模型完整验算。
  constexpr double window = 100;
  const auto acceptable = [](double activity, double bound, bool lower,
                             double absolute, double relative) {
    if (!std::isfinite(bound)) return true;
    const double violation = lower ? bound - activity : activity - bound;
    const double tolerance = window * (absolute + relative * std::max(std::abs(activity), std::abs(bound)));
    return !std::isfinite(violation) || !std::isfinite(tolerance) || violation <= tolerance;
  };
  double primal_objective = 0;
  double dual_objective = 0;
  const double sense = original.sense == ObjectiveSense::minimize ? 1 : -1;
  for (std::size_t row = 0; row < point.y.size(); ++row) {
    const double activity = point.ax[row] / working.row_scale[row];
    if (!std::isfinite(activity)) return true;
    if (!acceptable(activity, original.row_lower[row], true,
                     options.tolerances.feasibility_absolute, options.tolerances.feasibility_relative) ||
        !acceptable(activity, original.row_upper[row], false,
                     options.tolerances.feasibility_absolute, options.tolerances.feasibility_relative)) return false;
    const double dual = point.y[row] * working.row_scale[row] * working.objective_scale;
    if (!std::isfinite(dual)) return true;
    if (dual != 0) {
      const double bound = dual > 0 ? original.row_lower[row] : original.row_upper[row];
      if (!std::isfinite(bound)) return false;
      dual_objective += dual * bound;
    }
  }
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    const double primal = point.x[col] * working.column_scale[col];
    const double transpose = point.aty[col] * working.objective_scale / working.column_scale[col];
    if (!std::isfinite(primal) || !std::isfinite(transpose)) return true;
    if (!acceptable(primal, original.column_lower[col], true,
                     options.tolerances.feasibility_absolute, options.tolerances.feasibility_relative) ||
        !acceptable(primal, original.column_upper[col], false,
                     options.tolerances.feasibility_absolute, options.tolerances.feasibility_relative)) return false;
    const double cost = sense * original.costs[col];
    const double reduced = cost - transpose;
    primal_objective += cost * primal;
    if (!std::isfinite(reduced)) return true;
    if (reduced != 0) {
      double bound = reduced > 0 ? original.column_lower[col] : original.column_upper[col];
      if (!std::isfinite(bound)) {
        const double tolerance = window * (options.pdlp.optimality.dual_absolute +
            options.pdlp.optimality.dual_relative * std::max(std::abs(cost), std::abs(transpose)));
        if (std::isfinite(tolerance) && std::abs(reduced) > tolerance) return false;
        bound = reduced > 0 ? original.column_upper[col] : original.column_lower[col];
      }
      if (std::isfinite(bound)) dual_objective += reduced * bound;
    }
  }
  const double gap = std::abs(primal_objective - dual_objective);
  const double tolerance = window * (options.pdlp.optimality.gap_absolute +
      options.pdlp.optimality.gap_relative * std::max(std::abs(primal_objective), std::abs(dual_objective)));
  return !std::isfinite(gap) || !std::isfinite(tolerance) || gap <= tolerance;
}

void run_adaptive(const WorkingModel& working, const SolveOptions& options,
                    Clock::time_point start, SolveResult& result) {
  const auto& model = working.model;
  result.pdlp_operator_passes = 0;
  result.pdlp_acceleration_attempts = 0;
  result.pdlp_acceleration_accepted = 0;
  const CscOperator matrix(model, *result.pdlp_operator_passes);
  AdaptivePoint point(model.costs.size(), model.row_lower.size());
  AdaptivePoint trial(model.costs.size(), model.row_lower.size());
  AdaptivePoint average(model.costs.size(), model.row_lower.size());
  AdaptivePoint restart_point(model.costs.size(), model.row_lower.size());
  AdaptivePoint extrapolated(model.costs.size(), model.row_lower.size());
  ResidualSubspace subspace(options.pdlp.mode == "subspace" ? options.pdlp.history_size : 0);
  AssessmentBuffers verification_buffers;
  for (std::size_t col = 0; col < point.x.size(); ++col) {
    point.x[col] = std::clamp(0.0, model.column_lower[col], model.column_upper[col]);
  }
  matrix.multiply(point.x, point.ax);
  if (!all_finite(point.ax)) throw NumericalFailure("自主初始行活动值溢出");
  if (seconds_since(start) >= options.time_limit) throw BudgetExpired{};
  if (assess(working, point.x, point.y, options, result, &verification_buffers)) return;
  double omega = initial_weight(model);
  double eta = initial_step(model);
  double average_weight = 0;
  std::uint64_t attempts = 0;
  std::int64_t restart_iteration = 0;
  std::int64_t last_full_check = 0;
  restart_point = point;
  double start_merit = restart_merit(model, point, omega);
  double previous_merit = start_merit;
  result.status = "iteration_limit";
  try {
  for (std::int64_t iteration = 1; iteration <= options.pdlp.max_iterations; ++iteration) {
    if (seconds_since(start) >= options.time_limit) { result.status = "time_limit"; break; }
    double accepted_eta = 0;
    if (!adaptive_step(model, matrix, point, trial, omega, eta, accepted_eta,
                        attempts, options, start)) {
      if (assess(working, point.x, point.y, options, result, &verification_buffers)) return;
      throw NumericalFailure("自主迭代在浮点精度下停滞且未通过最优性检查");
    }
    result.pdlp_iterations = iteration;
    weighted_average(average, point, accepted_eta, average_weight);
    const bool periodic_check = iteration - last_full_check >= 1000;
    if (iteration % options.pdlp.check_interval == 0 || periodic_check) {
      matrix.multiply(average.x, average.ax);
      matrix.transpose_multiply(average.y, average.aty);
      if ((periodic_check || screening_candidate(working, result.model, point, options)) &&
          assess(working, point.x, point.y, options, result, &verification_buffers)) return;
      if ((periodic_check || screening_candidate(working, result.model, average, options)) &&
          assess(working, average.x, average.y, options, result, &verification_buffers)) return;
      if (periodic_check) last_full_check = iteration;
      const double current_merit = restart_merit(model, point, omega);
      const double average_merit = restart_merit(model, average, omega);
      const bool use_average = average_merit < current_merit;
      const double candidate_merit = std::min(current_merit, average_merit);
      const auto& standard_candidate = use_average ? average : point;
      const bool use_subspace = options.pdlp.mode == "subspace" &&
          subspace.candidate(model, matrix, standard_candidate, omega, candidate_merit,
                              options, start, extrapolated, *result.pdlp_acceleration_attempts);
      subspace.remember(standard_candidate);
      if (use_subspace && *result.pdlp_acceleration_accepted < std::numeric_limits<std::int64_t>::max()) {
        ++*result.pdlp_acceleration_accepted;
      }
      if (use_subspace && screening_candidate(working, result.model, extrapolated, options) &&
          assess(working, extrapolated.x, extrapolated.y, options, result, &verification_buffers)) return;
      const auto inner_iterations = iteration - restart_iteration;
      const bool sufficient = candidate_merit <= 0.2 * start_merit;
      const bool stalled = candidate_merit <= 0.9 * start_merit && candidate_merit > previous_merit;
      const bool long_epoch = static_cast<double>(inner_iterations) >= 0.5 * iteration;
      if (use_subspace || sufficient || stalled || long_epoch) {
        if (use_subspace) point = extrapolated;
        else if (use_average) point = average;
        omega = updated_weight(point, restart_point, omega);
        // 当前点、均值点和外推点的缓存都在本检查点直接计算，可直接复用。
        restart_point = point;
        restart_iteration = iteration;
        average_weight = 0;
        start_merit = restart_merit(model, point, omega);
        previous_merit = start_merit;
      } else {
        previous_merit = candidate_merit;
      }
    }
    if (iteration == options.pdlp.max_iterations) break;
  }
  } catch (const BudgetExpired&) {
    result.status = "time_limit";
    result.message = "自主求解已达到时间预算";
  } catch (const NumericalFailure& error) {
    result.status = "numerical_error";
    result.message = error.what();
  }
  // 覆盖正常预算出口和线搜索内部异常；不能让预筛漏掉最后的已接受迭代点。
  if (!assess(working, point.x, point.y, options, result, &verification_buffers) && average_weight > 0) {
    assess(working, average.x, average.y, options, result, &verification_buffers);
  }
}

}  // 匿名命名空间

std::string pdlp_version() { return "pdhg-cpu-0.3.0"; }

SolveResult solve_pdlp(const Model& model, const SolveOptions& options) {
  const auto start = Clock::now();
  SolveResult result;
  result.backend = "pdlp";
  result.backend_version = pdlp_version();
  result.model = model;
  result.pdlp_iterations = 0;
  try {
    validate_options(options);
    const auto errors = validate_model(model);
    if (!errors.empty()) throw std::invalid_argument("模型无效：" + errors.front());
    if (is_mip(model)) throw std::invalid_argument("自主 PDHG 原型只支持连续 LP，不支持 MILP");
    if (options.time_limit == 0) {
      result.status = "time_limit";
      result.message = "求解时间预算为零，未启动自主迭代";
    } else if (options.pdlp.max_iterations == 0) {
      result.status = "iteration_limit";
      result.message = "迭代预算为零，未启动自主迭代";
    } else {
      const WorkingModel working = prepare_model(model, options, start);
      if (options.pdlp.mode != "baseline") {
        run_adaptive(working, options, start, result);
      } else {
      std::vector<double> primal_step;
      std::vector<double> dual_step;
      make_steps(working.model, primal_step, dual_step);
      std::vector<double> x(model.costs.size());
      std::vector<double> y(model.row_lower.size(), 0);
      for (std::size_t col = 0; col < x.size(); ++col) {
        x[col] = project(0, working.model.column_lower[col], working.model.column_upper[col]);
      }
      std::vector<double> next_x(x.size());
      std::vector<double> average_x(x.size(), 0);
      std::vector<double> average_y(y.size(), 0);
      std::vector<long double> row_activity(y.size());
      if (seconds_since(start) >= options.time_limit) throw BudgetExpired{};
      if (!assess(working, x, y, options, result)) {
        result.status = "iteration_limit";
        std::int64_t last_check = 0;
        for (std::int64_t iteration = 1; iteration <= options.pdlp.max_iterations; ++iteration) {
          if (seconds_since(start) >= options.time_limit) {
            result.status = "time_limit";
            break;
          }
          iterate(working.model, primal_step, dual_step, x, y, next_x, row_activity);
          result.pdlp_iterations = iteration;
          update_average(average_x, x, iteration);
          update_average(average_y, y, iteration);
          if (iteration % options.pdlp.check_interval == 0) {
            last_check = iteration;
            if (assess(working, x, y, options, result) ||
                assess(working, average_x, average_y, options, result)) break;
          }
          // 避免用户给出 INT64_MAX 预算时的循环计数溢出。
          if (iteration == options.pdlp.max_iterations) break;
        }
        if (result.status != "optimal" && *result.pdlp_iterations != last_check) {
          if (!assess(working, x, y, options, result)) {
            assess(working, average_x, average_y, options, result);
          }
        }
      }
      }
    }
  } catch (const BudgetExpired&) {
    result.status = "time_limit";
    result.message = "自主求解已达到时间预算";
  } catch (const NumericalFailure& error) {
    result.status = "numerical_error";
    result.message = error.what();
  } catch (const std::exception& error) {
    result.status = "error";
    result.message = "自主求解失败：" + std::string(error.what());
  }
  result.backend_status = result.status;
  result.total_seconds = seconds_since(start);
  result.solve_seconds = std::max(0.0, result.total_seconds - result.verification_seconds);
  return result;
}

}  // 命名空间 solver
