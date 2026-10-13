#include "solver/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace solver {
namespace {

void validate_bounds(const std::vector<double>& lower,
                     const std::vector<double>& upper, const char* label,
                     std::vector<std::string>& errors) {
  for (std::size_t i = 0; i < lower.size(); ++i) {
    if (std::isnan(lower[i]) || lower[i] == std::numeric_limits<double>::infinity()) {
      errors.push_back(std::string(label) + "下界必须为有限数或负无穷：" + std::to_string(i));
    }
  }
  for (std::size_t i = 0; i < upper.size(); ++i) {
    if (std::isnan(upper[i]) || upper[i] == -std::numeric_limits<double>::infinity()) {
      errors.push_back(std::string(label) + "上界必须为有限数或正无穷：" + std::to_string(i));
    }
  }
  for (std::size_t i = 0; i < std::min(lower.size(), upper.size()); ++i) {
    if (lower[i] > upper[i]) {
      errors.push_back(std::string(label) + "下界大于上界：" + std::to_string(i));
    }
  }
}

}  // 匿名命名空间

std::vector<std::string> validate_model(const Model& model) {
  std::vector<std::string> errors;
  const auto columns = model.costs.size();
  const auto rows = model.row_lower.size();
  const auto nonzeros = model.values.size();
  if (model.sense != ObjectiveSense::minimize && model.sense != ObjectiveSense::maximize) {
    errors.emplace_back("目标方向无效");
  }
  if (!std::isfinite(model.objective_offset)) {
    errors.emplace_back("目标常数必须为有限数");
  }
  if (model.column_lower.size() != columns || model.column_upper.size() != columns ||
      model.kinds.size() != columns) {
    errors.emplace_back("变量界或变量类型的长度与目标系数长度不一致");
  }
  if (model.row_upper.size() != rows) {
    errors.emplace_back("行上下界的长度不一致");
  }
  validate_bounds(model.column_lower, model.column_upper, "变量", errors);
  validate_bounds(model.row_lower, model.row_upper, "行", errors);
  for (std::size_t column = 0; column < columns; ++column) {
    if (!std::isfinite(model.costs[column])) {
      errors.emplace_back("目标系数必须为有限数：" + std::to_string(column));
    }
  }
  for (std::size_t column = 0; column < model.kinds.size(); ++column) {
    if (model.kinds[column] != VariableKind::continuous &&
        model.kinds[column] != VariableKind::integer) {
      errors.emplace_back("变量类型无效：" + std::to_string(column));
    }
  }
  if (model.column_start.size() != columns + 1) {
    errors.emplace_back("CSC 列指针长度必须等于变量数加一");
  }
  if (model.row_index.size() != nonzeros) {
    errors.emplace_back("CSC 行索引与系数的长度不一致");
  }
  if (!model.column_start.empty()) {
    if (model.column_start.front() != 0) {
      errors.emplace_back("CSC 首个列指针必须为零");
    }
    if (model.column_start.back() < 0 ||
        static_cast<std::uint64_t>(model.column_start.back()) != nonzeros) {
      errors.emplace_back("CSC 末尾列指针必须等于非零元数量");
    }
  }
  for (std::size_t i = 0; i < model.column_start.size(); ++i) {
    const auto position = model.column_start[i];
    if (position < 0 || static_cast<std::uint64_t>(position) > nonzeros) {
      errors.emplace_back("CSC 列指针超出存储范围：" + std::to_string(i));
    }
    if (i > 0 && position < model.column_start[i - 1]) {
      errors.emplace_back("CSC 列指针必须单调不减：" + std::to_string(i));
    }
  }
  for (std::size_t i = 0; i < model.row_index.size(); ++i) {
    if (model.row_index[i] < 0 || static_cast<std::uint64_t>(model.row_index[i]) >= rows) {
      errors.emplace_back("CSC 行索引超出行范围：" + std::to_string(i));
    }
  }
  for (std::size_t i = 0; i < nonzeros; ++i) {
    if (!std::isfinite(model.values[i])) {
      errors.emplace_back("矩阵系数必须为有限数：" + std::to_string(i));
    }
  }
  return errors;
}

bool is_mip(const Model& model) {
  return std::find(model.kinds.begin(), model.kinds.end(), VariableKind::integer) !=
         model.kinds.end();
}

}  // 命名空间 solver
