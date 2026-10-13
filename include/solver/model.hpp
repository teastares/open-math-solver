#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace solver {

enum class ObjectiveSense { minimize, maximize };
enum class VariableKind { continuous, integer };

// 模型始终使用原始目标方向与列压缩稀疏矩阵。
struct Model {
  std::string name;
  ObjectiveSense sense = ObjectiveSense::minimize;
  double objective_offset = 0.0;
  std::vector<double> costs;
  std::vector<double> column_lower;
  std::vector<double> column_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;
  std::vector<VariableKind> kinds;
  // 空模型仍保留末尾指针；列内允许无序与重复行索引，重复项按求和解释。
  std::vector<std::int64_t> column_start{0};
  std::vector<std::int64_t> row_index;
  std::vector<double> values;
};

std::vector<std::string> validate_model(const Model& model);
bool is_mip(const Model& model);

}  // 命名空间 solver
