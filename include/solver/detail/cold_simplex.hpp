#pragma once

#include "solver/backend.hpp"
#include "solver/detail/logical_basis.hpp"
#include "solver/detail/warm_cleanup.hpp"

namespace solver::detail::cold_simplex {
namespace cross = solver::detail::ipx_crossover;
namespace warm = solver::detail::warm_cleanup;
using Clock = warm::Clock;

// 只构造原模型的标准化结构。与交叉组件共用Prepared数据合同和原坐标恢复，
// 不调用IPX，也不生成、贴界或投影一个虚构的原始/对偶中间点。
// 由公开入口先验证Model；自由行去除、范围活动列和重复项求和保持已验证顺序。
inline cross::Prepared prepare_structure(const Model& model, Clock::time_point deadline) {
  cross::Prepared p;
  const auto cols = model.costs.size(), rows = model.row_lower.size();
  p.original_columns = cols;
  p.original_rows = rows;
  std::vector<logical_basis::Index> row_map(rows, -1);
  for (std::size_t row = 0; row < rows; ++row) {
    cross::detail::tick(row, deadline);
    const double lower = model.row_lower[row], upper = model.row_upper[row];
    const bool has_lower = std::isfinite(lower), has_upper = std::isfinite(upper);
    if (!has_lower && !has_upper) continue;
    row_map[row] = cross::detail::index(p.original_row.size());
    p.original_row.push_back(row);
    if (has_lower && has_upper && lower != upper) {
      p.row_type.push_back('=');
      p.rhs.push_back(0);
      p.auxiliary_row.push_back(row);
    } else {
      p.row_type.push_back(has_lower && has_upper ? '=' : has_lower ? '>' : '<');
      p.rhs.push_back(has_lower ? lower : upper);
    }
  }
  cross::detail::index(cols + p.auxiliary_row.size() + p.original_row.size());
  p.lower = model.column_lower;
  p.upper = model.column_upper;
  const double sense = model.sense == ObjectiveSense::minimize ? 1 : -1;
  p.objective_offset = sense * model.objective_offset;
  p.costs.reserve(cols + p.auxiliary_row.size());
  p.column_start.reserve(cols + p.auxiliary_row.size() + 1);
  p.values.reserve(model.values.size() + p.auxiliary_row.size());
  p.row_index.reserve(model.values.size() + p.auxiliary_row.size());
  p.column_start.push_back(0);
  std::vector<std::pair<logical_basis::Index, double>> column;
  for (std::size_t col = 0; col < cols; ++col) {
    cross::detail::tick(col, deadline);
    p.costs.push_back(sense * model.costs[col]);
    column.clear();
    for (std::int64_t k = model.column_start[col]; k < model.column_start[col + 1]; ++k) {
      const auto pos = static_cast<std::size_t>(k);
      cross::detail::tick(pos, deadline);
      const auto row = row_map[static_cast<std::size_t>(model.row_index[pos])];
      if (row >= 0) column.emplace_back(row, model.values[pos]);
    }
    const auto order = [](const auto& a, const auto& b) { return a.first < b.first; };
    if (!std::is_sorted(column.begin(), column.end(), order))
      std::stable_sort(column.begin(), column.end(), order);
    cross::detail::check(deadline);
    for (std::size_t first = 0; first < column.size();) {
      std::size_t end = first;
      long double sum = 0;
      do {
        cross::detail::tick(end, deadline);
        cross::detail::finite_add(sum, column[end].second);
        ++end;
      } while (end < column.size() && column[end].first == column[first].first);
      const double value = cross::detail::narrow(sum);
      if (value != 0) {
        p.row_index.push_back(column[first].first);
        p.values.push_back(value);
      }
      first = end;
    }
    p.column_start.push_back(cross::detail::index(p.values.size()));
  }
  for (std::size_t k = 0; k < p.auxiliary_row.size(); ++k) {
    cross::detail::tick(k, deadline);
    const auto original = p.auxiliary_row[k];
    p.costs.push_back(0);
    p.lower.push_back(model.row_lower[original]);
    p.upper.push_back(model.row_upper[original]);
    p.row_index.push_back(row_map[original]);
    p.values.push_back(-1);
    p.column_start.push_back(cross::detail::index(p.values.size()));
  }
  cross::detail::index(p.values.size() + p.original_row.size());
  cross::detail::check(deadline);
  return p;
}

// 公开包装器负责模型/整数域、线程、容差和迭代上限的合法性检查与异常映射。
// 此处共用原绝对期限，返回现有引擎的真实状态、候选及原模型验算结果。
// 初始符号阻塞只请求零目标可行化，不能据此宣告整个LP不可行或无界。
inline warm::Result run(const Model& model, const SolveOptions& options,
    Clock::time_point absolute_deadline, bool incremental_objective_dual_update = true) {
  const auto budget = [&] { cross::detail::check(absolute_deadline); };
  budget();
  std::size_t rows = 0;
  for (std::size_t row = 0; row < model.row_lower.size(); ++row) {
    cross::detail::tick(row, absolute_deadline);
    rows += std::isfinite(model.row_lower[row]) || std::isfinite(model.row_upper[row]);
  }
  const double sense = model.sense == ObjectiveSense::minimize ? 1 : -1;
  std::vector<double> costs;
  costs.reserve(model.costs.size());
  for (std::size_t col = 0; col < model.costs.size(); ++col) {
    cross::detail::tick(col, absolute_deadline);
    costs.push_back(sense * model.costs[col]);
  }
  auto basis = logical_basis::build({costs, model.column_lower, model.column_upper, rows},
      options.simplex.optimality, budget);
  const bool sign_blocked = basis.status == logical_basis::Status::dual_sign_infeasible;
  if (basis.status != logical_basis::Status::ready && !sign_blocked) {
    warm::Result rejected;
    rejected.status = basis.status == logical_basis::Status::unsupported_empty ?
        warm::Status::unsupported_empty : warm::Status::invalid_input;
    rejected.message = basis.message;
    budget();
    return rejected;
  }
  auto prepared = prepare_structure(model, absolute_deadline);
  for (std::size_t pos = 0; pos < prepared.auxiliary_row.size(); ++pos) {
    cross::detail::tick(pos, absolute_deadline);
    basis.variable_status.push_back(IPX_nonbasic_lb);
    basis.structural_seed.push_back(prepared.lower[prepared.original_columns + pos]);
  }
  budget();
  warm::Limits limits;
  limits.max_steps = static_cast<std::uint64_t>(options.simplex.max_iterations);
  limits.dual_cleanup = true;
  limits.feasibility_first = sign_blocked;
  limits.incremental_feasibility_update = true;
  limits.incremental_objective_dual_update = incremental_objective_dual_update;
  return warm::run({model, prepared, basis.variable_status, basis.row_status, basis.structural_seed},
      absolute_deadline, limits, options.tolerances, options.simplex.optimality);
}
}  // 命名空间 solver::detail::cold_simplex
