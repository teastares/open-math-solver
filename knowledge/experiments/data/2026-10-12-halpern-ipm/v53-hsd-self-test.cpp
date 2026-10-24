// 独立采集接线自测；使用手算矩阵，不链接求解器，不运行 LP。
#include "v53-hsd-diagnostic.hpp"
#include <iostream>
#include <stdexcept>

namespace {
struct Model {
  std::vector<std::int64_t> starts{0, 1}, rows{0}, original_row{0};
  std::vector<double> values{1}, costs{-1}, rhs{2}, dual_sign{1}, row_scale{1}, column_scale{1};
  std::vector<bool> inequality{true};
  std::vector<std::size_t> original_columns;
  std::vector<double> fixed_values;
  std::vector<int> presolved_bounds, presolved_fixed_columns;
  double objective_scale = 1;
};
struct Original {
  int sense = 0;
  double objective_offset = 0;
  std::vector<std::int64_t> column_start{0, 1}, row_index{0};
  std::vector<double> values{1}, costs{-1}, column_lower{0}, column_upper{4}, row_lower{2}, row_upper{2};
};
struct Options {
  struct { double feasibility_absolute = 1e-7, feasibility_relative = 1e-9;
    double objective_absolute = 1e-7, objective_relative = 1e-9; } tolerances;
  struct { struct { double dual_absolute = 1e-7, dual_relative = 1e-9;
    double gap_absolute = 1e-7, gap_relative = 1e-9; } optimality; } pdlp;
};
struct Point { std::vector<double> x{1}, z{2}; };
struct Direction { Point point; struct { double tau = 0.25; } scalars; };
struct Result {
  std::optional<std::int64_t> ipm_iterations{0}, ipm_factorizations{1}, ipm_refinement_steps{0};
  std::optional<std::int64_t> ipm_homogeneous_constant_solves{1}, ipm_homogeneous_direction_solves{1};
  std::optional<std::int64_t> ipm_homogeneous_constant_repairs{0}, ipm_homogeneous_constant_repair_successes{0};
};
struct Checks {
  struct { bool valid_input = true, feasible = true, objective_consistent = true;
    std::optional<double> objective{-1};
    double max_row_violation = 0, max_bound_violation = 0, max_integrality_violation = 0;
    std::vector<std::string> errors; } primal;
  struct { bool valid_input = true, dual_feasible = false, gap_satisfied = false, numerically_optimal = false;
    double max_dual_residual = 0.5;
    std::optional<double> dual_objective_estimate{-2}, corrected_dual_bound, objective_gap{1}, relative_objective_gap{0.5};
    std::vector<std::string> errors{"合成对偶误差"}; } optimality;
};
}

int main(int argc, char**) {
  if (argc != 1) return 2;
  try {
    std::size_t checks = 0;
    const auto check = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("v53 诊断接线自测失败"); };
    auto& state = solver::v53hsd::state();
    const bool enabled = state.enabled;
    Model model; Original original; Options options; Result result; Checks source_checks;
    Point point; Direction direction; direction.point.z[0] = -0.75;
    const std::vector<double> rd{1}, rhs{-1, 1}, diagonal{0.5}, u{0.5, -1}, v{2.5, 1};
    {
      solver::v53hsd::RunScope run(model, original, options);
      check(state.ready == enabled);
      if (enabled) {
        check(state.cache_bytes == state.static_bytes + 4 * state.step_bytes + solver::v53hsd::metadata_reserve);
        check(state.cache_bytes <= solver::v53hsd::cache_limit);
        check(state.static_fields[0].offset == 0 && state.step_fields[0].offset == 0);
      }
      for (std::int64_t iteration = 1; iteration <= 4; ++iteration) {
        // 同一轮 affine/corrector 可拥有相同 theta；阶段识别不能依赖 theta。
        for (int phase = 0; phase < 2; ++phase) {
          solver::v53hsd::DirectionScope observation(point, rd, rhs, diagonal, 1, 1, 1, result);
          observation.got_u(u); observation.returned(direction, v, result);
        }
        Point candidate = point;
        candidate.z[0] = point.z[0] + 0.5 * direction.point.z[0];
        solver::v53hsd::commit(candidate, 1.125, 0.5, iteration);
        result.ipm_iterations = iteration;
        Point extracted = candidate; extracted.z[0] /= 1.125;
        // 第四轮不替换：最后工作点和最终保留候选应来自不同迭代。
        const bool replace = iteration != 4;
        solver::v53hsd::assess(extracted, candidate.x, extracted.z, source_checks, replace, result);
        if (enabled) {
          check(state.completed_count == static_cast<std::size_t>(iteration));
          check(state.saved_count == std::min<std::size_t>(iteration, 3));
          const auto& slot = state.slots[state.saved[state.saved_count - 1]];
          check(slot.iteration == iteration && slot.ordinal == 2 && slot.assessed && slot.step_complete);
          double captured = 0;
          std::memcpy(&captured, slot.payload.get() + state.step_fields[9].offset, sizeof captured);
          check(captured == direction.point.z[0]);
          check(slot.assessment.replace == replace);
        }
      }
      if (enabled) {
        check(state.slots[state.saved[0]].iteration == 1);
        check(state.slots[state.saved[1]].iteration == 3);
        check(state.slots[state.saved[2]].iteration == 4);
        check(state.retained_assessment.iteration == 3 && state.latest_assessment.iteration == 4);
      }
      // 模拟第五轮在 u 求解中失败；只保留输入及 unwind 标记，不伪造 u 或完成步。
      try {
        solver::v53hsd::DirectionScope failed(point, rd, rhs, diagonal, 1, 1, 1, result);
        throw std::runtime_error("合成失败");
      } catch (const std::runtime_error&) {}
      if (enabled) {
        const auto& failed = state.slots[state.pending];
        check(failed.iteration == 5 && failed.ordinal == 1 && failed.have_input && failed.unwound);
        check(!failed.have_u && !failed.direction_returned && !failed.step_complete && !failed.assessed);
        std::size_t bytes = solver::v53hsd::cache_limit;
        solver::v53hsd::Field field;
        check(!state.add(field, "too_large", "f64_le", 1, 8, bytes));
        check(bytes == solver::v53hsd::cache_limit);
      } else {
        check(state.bytes_copied == 0 && state.completed_count == 0 && state.assessments == 0);
        check(!state.static_payload && !state.slots[0].payload && !state.active && !state.run_started);
      }
    }
    check(!state.active);
    std::cout << "{\"passed\":true,\"mode\":\"" << (enabled ? "on" : "off") << "\",\"checks\":" << checks
        << ",\"lp_runs\":0,\"expected_iterations\":[1,3,4],\"expected_retained_iteration\":3,\"expected_failed_iteration\":5}\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
