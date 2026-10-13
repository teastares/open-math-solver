#include "solver/model.hpp"
#include "solver/verification.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int checks = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    std::cerr << "验证失败：" << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void near(double actual, double expected, const std::string& message) {
  require(std::abs(actual - expected) <= 1e-12, message);
}

constexpr auto infinity = std::numeric_limits<double>::infinity();
constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

solver::Model sample() {
  solver::Model model;
  model.name = "验证样例";
  model.objective_offset = 7.0;
  model.costs = {2.0, -3.0};
  model.column_lower = {0.0, -infinity};
  model.column_upper = {4.0, infinity};
  model.row_lower = {1.0, -infinity};
  model.row_upper = {3.0, 5.0};
  model.kinds = {solver::VariableKind::continuous, solver::VariableKind::continuous};
  model.column_start = {0, 2, 4};
  model.row_index = {0, 1, 0, 1};
  model.values = {1.0, 2.0, 1.0, -1.0};
  return model;
}

void objectives_and_feasibility() {
  auto model = sample();
  require(solver::validate_model(model).empty(), "合法模型通过结构检查");
  const std::vector<double> point{2.0, 1.0};
  auto result = solver::verify_solution(model, point, 8.0);
  require(result.valid_input && result.feasible && result.objective_consistent,
          "可行最小化解及目标常数");
  require(result.objective.has_value(), "重算目标存在");
  near(*result.objective, 8.0, "最小化目标保留常数");
  require(result.errors.empty(), "可行解无错误");
  model.sense = solver::ObjectiveSense::maximize;
  result = solver::verify_solution(model, point, 8.0);
  require(result.feasible && result.objective_consistent, "最大化目标保持原方向");
  near(*result.objective, 8.0, "最大化目标不反转符号");
  result = solver::verify_solution(model, point, -8.0);
  require(result.feasible && !result.objective_consistent, "目标不一致不改变原始可行性");
  require(!result.errors.empty(), "目标不一致提供诊断");
  result = solver::verify_solution(model, point);
  require(result.feasible && result.objective_consistent, "允许不提供报告目标");

  result = solver::verify_solution(model, std::vector<double>{0.0, 0.0});
  require(!result.feasible, "拒绝违反行下界的解");
  near(result.max_row_violation, 1.0, "行下界违反量");
  result = solver::verify_solution(model, std::vector<double>{2.0, 2.0});
  require(!result.feasible, "拒绝违反行上界的解");
  near(result.max_row_violation, 1.0, "行上界违反量");
  result = solver::verify_solution(model, std::vector<double>{-0.5, 2.0});
  require(!result.feasible, "拒绝违反变量下界的解");
  near(result.max_bound_violation, 0.5, "变量下界违反量");
  result = solver::verify_solution(model, std::vector<double>{4.5, -2.0});
  require(!result.feasible, "拒绝违反变量上界的解");
  near(result.max_bound_violation, 0.5, "变量上界违反量");

  model.kinds[0] = solver::VariableKind::integer;
  require(solver::is_mip(model), "识别整数变量");
  result = solver::verify_solution(model, std::vector<double>{1.5, 1.0});
  require(!result.feasible, "拒绝违反整数性的解");
  near(result.max_integrality_violation, 0.5, "整数违反量");
  require(solver::verify_solution(model, point).feasible, "整数可行解通过");
  model.kinds[0] = solver::VariableKind::continuous;
  require(!solver::is_mip(model), "纯连续模型不标记为整数模型");
}

void empty_and_sparse_models() {
  solver::Model model;
  model.objective_offset = -4.0;
  auto result = solver::verify_solution(model, std::vector<double>{}, -4.0);
  require(result.valid_input && result.feasible && result.objective_consistent,
          "空模型及非零目标常数");
  model.row_lower = {1.0};
  model.row_upper = {infinity};
  result = solver::verify_solution(model, std::vector<double>{});
  require(result.valid_input && !result.feasible, "零变量约束仍检查可行性");
  near(result.max_row_violation, 1.0, "空行违反量");
  model.row_lower[0] = 0.0;
  require(solver::verify_solution(model, std::vector<double>{}).feasible,
          "零变量的可行空行");

  model = {};
  model.costs = {1.0};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  result = solver::verify_solution(model, std::vector<double>{-10.0}, -10.0);
  require(result.feasible && result.objective_consistent, "无行约束的自由变量");
  model.row_lower = {6.0, 4.0};
  model.row_upper = {6.0, 4.0};
  model.column_start = {0, 3};
  model.row_index = {1, 0, 0};
  model.values = {2.0, 1.0, 2.0};
  result = solver::verify_solution(model, std::vector<double>{2.0});
  require(result.feasible, "列内无序与重复项按求和解释");
  model.column_lower = {2.0};
  model.column_upper = {2.0};
  require(solver::verify_solution(model, std::vector<double>{2.0}).feasible,
          "固定变量及等式约束");
}

void invalid_models_and_inputs() {
  const auto bad_model = [](const solver::Model& model, const std::string& label) {
    require(!solver::validate_model(model).empty(), label);
    const auto result = solver::verify_solution(model, std::vector<double>{2.0, 1.0});
    require(!result.valid_input && !result.feasible && !result.objective.has_value(),
            label + "不能绕过解验证");
  };
  auto model = sample();
  model.column_start = {0, 4};
  bad_model(model, "错误列指针长度");
  model = sample(); model.column_start[0] = 1;
  bad_model(model, "列起点非零");
  model = sample(); model.column_start[1] = -1;
  bad_model(model, "负列指针");
  model = sample(); model.column_start[1] = 5;
  bad_model(model, "越界且下降列指针");
  model = sample(); model.column_start.back() = 3;
  bad_model(model, "末尾列指针与非零元数量不符");
  model = sample(); model.column_start.clear();
  bad_model(model, "缺失列指针");
  model = sample(); model.row_index.pop_back();
  bad_model(model, "行索引和系数长度不符");
  model = sample(); model.row_index[0] = -1;
  bad_model(model, "负行索引");
  model = sample(); model.row_index[0] = 2;
  bad_model(model, "越界行索引");
  model = sample(); model.values[0] = nan;
  bad_model(model, "非数矩阵系数");
  model = sample(); model.values[0] = infinity;
  bad_model(model, "无穷矩阵系数");
  model = sample(); model.costs[0] = infinity;
  bad_model(model, "无穷目标系数");
  model = sample(); model.objective_offset = nan;
  bad_model(model, "非数目标常数");
  model = sample(); model.column_lower[0] = nan;
  bad_model(model, "非数变量界");
  model = sample(); model.column_lower[0] = infinity;
  bad_model(model, "正无穷下界");
  model = sample(); model.column_upper[0] = -infinity;
  bad_model(model, "负无穷上界");
  model = sample(); model.row_upper[0] = nan;
  bad_model(model, "非数行界");
  model = sample(); model.column_lower[0] = 5.0;
  bad_model(model, "矛盾变量上下界");
  model = sample(); model.row_upper[0] = 0.0;
  bad_model(model, "矛盾行上下界");
  model = sample(); model.column_lower.pop_back();
  bad_model(model, "变量界长度错误");
  model = sample(); model.row_upper.pop_back();
  bad_model(model, "行界长度错误");
  model = sample(); model.kinds.pop_back();
  bad_model(model, "变量类型长度错误");
  model = sample(); model.kinds[0] = static_cast<solver::VariableKind>(5);
  bad_model(model, "未知变量类型");
  model = sample(); model.sense = static_cast<solver::ObjectiveSense>(5);
  bad_model(model, "未知目标方向");
  model = sample();
  auto result = solver::verify_solution(model, std::vector<double>{2.0});
  require(!result.valid_input, "候选解长度错误");
  result = solver::verify_solution(model, std::vector<double>{nan, 1.0});
  require(!result.valid_input, "非数候选解");
  result = solver::verify_solution(model, std::vector<double>{2.0, infinity});
  require(!result.valid_input, "无穷候选解");
  result = solver::verify_solution(model, std::vector<double>{2.0, 1.0}, infinity);
  require(!result.valid_input, "无穷报告目标");
  for (int field = 0; field < 5; ++field) {
    for (double invalid : {-1.0, nan, infinity}) {
      solver::Tolerances tolerance;
      double* fields[] = {&tolerance.feasibility_absolute, &tolerance.feasibility_relative,
                          &tolerance.integrality, &tolerance.objective_absolute,
                          &tolerance.objective_relative};
      *fields[field] = invalid;
      result = solver::verify_solution(model, std::vector<double>{2.0, 1.0}, {}, tolerance);
      require(!result.valid_input, "所有容差字段拒绝负数与非有限值");
    }
  }
}

void tolerance_boundaries() {
  solver::Model model;
  model.costs = {1.0};
  model.column_lower = {0.0};
  model.column_upper = {1.0};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {1.0};
  model.row_lower = {0.0};
  model.row_upper = {1.0};
  solver::Tolerances tolerance{0.125, 0.0, 0.125, 0.125, 0.0};
  auto result = solver::verify_solution(model, std::vector<double>{1.125}, 1.25, tolerance);
  require(result.feasible && result.objective_consistent, "绝对容差边界包含等号");
  const auto outside = std::nextafter(1.125, infinity);
  result = solver::verify_solution(model, std::vector<double>{outside}, {}, tolerance);
  require(!result.feasible, "刚超出绝对可行性容差被拒绝");
  result = solver::verify_solution(model, std::vector<double>{1.125},
                                   std::nextafter(1.25, infinity), tolerance);
  require(!result.objective_consistent, "刚超出目标绝对容差被拒绝");
  require(solver::verify_solution(model, std::vector<double>{-0.125}, {}, tolerance).feasible,
          "下界绝对容差边界包含等号");
  model.kinds[0] = solver::VariableKind::integer;
  result = solver::verify_solution(model, std::vector<double>{1.125}, {}, tolerance);
  require(result.feasible, "整数容差边界包含等号");
  tolerance.feasibility_absolute = 1.0;
  result = solver::verify_solution(model, std::vector<double>{outside}, {}, tolerance);
  require(!result.feasible, "整数容差不受可行性相对尺度放宽");

  model.kinds[0] = solver::VariableKind::continuous;
  tolerance = {0.0, 0.125, 0.0, 0.0, 0.125};
  model.column_upper[0] = 7.0;
  model.row_upper[0] = 7.0;
  result = solver::verify_solution(model, std::vector<double>{8.0}, 7.0, tolerance);
  require(result.feasible && result.objective_consistent, "相对容差以最大绝对值缩放");
  result = solver::verify_solution(model, std::vector<double>{std::nextafter(8.0, infinity)},
                                   7.0, tolerance);
  require(!result.feasible && !result.objective_consistent, "超出相对容差边界");
  tolerance = {0.0, 0.0, 0.0, 0.0, 0.0};
  require(solver::verify_solution(model, std::vector<double>{7.0}, 7.0, tolerance).feasible,
          "零容差允许精确边界值");
}

void overflow_is_not_success() {
  solver::Model model;
  model.costs = {std::numeric_limits<double>::max()};
  model.column_lower = {-infinity};
  model.column_upper = {infinity};
  model.kinds = {solver::VariableKind::continuous};
  model.column_start = {0, 0};
  auto result = solver::verify_solution(model, std::vector<double>{2.0});
  require(!result.feasible && !result.objective_consistent && !result.objective.has_value(),
          "目标溢出不报告成功");
  require(!result.errors.empty(), "目标溢出提供诊断");
  model.costs[0] = 0.0;
  model.row_lower = {-infinity};
  model.row_upper = {0.0};
  model.column_start = {0, 1};
  model.row_index = {0};
  model.values = {std::numeric_limits<double>::max()};
  result = solver::verify_solution(model, std::vector<double>{std::numeric_limits<double>::max()});
  require(!result.feasible && !result.errors.empty(), "行乘积或违反量溢出不报告成功");
  model = sample();
  model.costs = {1e16, -1e16};
  model.objective_offset = 1.0;
  result = solver::verify_solution(model, std::vector<double>{1.0, 1.0});
  if (std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits) {
    near(*result.objective, 1.0, "可用时通过扩展精度累计保留目标常数");
  }
}

}  // 匿名命名空间

int main() {
  objectives_and_feasibility();
  empty_and_sparse_models();
  invalid_models_and_inputs();
  tolerance_boundaries();
  overflow_is_not_success();
  std::cout << "模型与解验证通过，共 " << checks << " 项检查。\n";
  return EXIT_SUCCESS;
}
