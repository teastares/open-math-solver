// 解析构造验证诊断算子、舍入阶段和关闭路径，不求解公开 LP。
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <ostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "instrumented.cpp"

int main() {
  int checks = 0;
  const auto require = [&](bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
  };
  try {
    const solver::detail::IpmFullNewtonLayout mixed{2, 2, {0}};
    const std::vector<solver::Index> starts{0, 2, 4}, rows{0, 1, 0, 1};
    const std::vector<double> values{1, 3, 2, -1}, slack{2, 0}, dual{3, 0};
    const std::vector<long double> direction{-1, 2, .5, -1, 3};
    solver::V42Work work;
    const auto product = solver::v42_product(mixed, starts, rows, values, slack, dual, direction, &work);
    require(product == std::vector<long double>{-2.5L, 2, 6, -5, 10}, "混合等式的手算 J 不一致");
    require(work.j_started == 1 && work.j_completed == 1 && work.csc_started == 1 &&
        work.csc_completed == 1 && work.nnz_visits == 4, "J 扫描账本错误");
    const std::array<double, 3> tolerance{.5, 2, 5};
    const auto metrics = solver::v42_metrics(mixed, product, tolerance, &work);
    require(metrics.block_scaled_inf == std::array<long double, 3>{5, 3, 2}, "三块尺度错误");
    require(metrics.block_inf == std::array<long double, 3>{2.5L, 6, 10}, "物理块范数错误");
    require(metrics.scaled_inf == 5 && std::abs(metrics.weighted_l2 - std::sqrt(60.25L)) < 1e-17L,
        "加权 L2 与无穷范数混淆");
    require(work.metric_elements == 5, "范数元素账本错误");
    const auto invalid = solver::v42_metrics(mixed, product, std::array<double, 3>{0, 2, 5});
    require(!std::isfinite(invalid.weighted_l2), "零尺度没有报告无效");
    bool dimension_rejected = false;
    try { solver::v42_metrics(mixed, std::vector<long double>{1}, tolerance); }
    catch (const std::invalid_argument&) { dimension_rejected = true; }
    require(dimension_rejected, "残差维数未校验");

    const solver::detail::IpmFullNewtonLayout scalar{1, 1, {0}};
    const double large = std::ldexp(1.0, 53);
    const std::vector<double> base{0, large, 0}, coefficients{1, 1}, rhs{1, 0, 1};
    const std::vector<std::vector<double>> directions{{0, 1, 0}, {0, -large, 0}};
    auto current = base;
    for (std::size_t j = 0; j < coefficients.size(); ++j)
      for (std::size_t i = 0; i < current.size(); ++i)
        current[i] = static_cast<double>(static_cast<long double>(current[i]) +
            static_cast<long double>(coefficients[j]) * directions[j][i]);
    const auto reconstructed = solver::v42_reconstruct(base, directions, coefficients, &work);
    require(std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits,
        "此手算舍入例需要扩展 long double");
    require(current == std::vector<double>{0, 0, 0}, "逐项舍入路径错误");
    require(reconstructed.ld_sum == std::vector<long double>{0, 1, 0} &&
        reconstructed.once_cast == std::vector<double>{0, 1, 0}, "完整累加路径错误");
    require(reconstructed.ld_sum_finite && reconstructed.once_cast_finite &&
        reconstructed.once_cast_underflows == 0, "正常重构误报范围错误");
    require(work.reconstruction_terms == 6, "重构项账本错误");
    const std::vector<solver::Index> scalar_starts{0, 1}, scalar_rows{0};
    const std::vector<double> ones{1};
    const auto residual_of = [&](const auto& input) {
      const std::vector<long double> converted(input.begin(), input.end());
      auto image = solver::v42_product(scalar, scalar_starts, scalar_rows, ones, ones, ones, converted);
      for (std::size_t i = 0; i < image.size(); ++i) image[i] = rhs[i] - image[i];
      return image;
    };
    const auto current_residual = residual_of(current), ld_residual = residual_of(reconstructed.ld_sum);
    require(current_residual == std::vector<long double>{1, 0, 1}, "逐项舍入真残差错误");
    require(ld_residual == std::vector<long double>{0, 0, 0} &&
        residual_of(reconstructed.once_cast) == ld_residual, "一次舍入真残差错误");
    auto image_residual = residual_of(base);
    for (std::size_t j = 0; j < directions.size(); ++j) {
      const std::vector<long double> column(directions[j].begin(), directions[j].end());
      const auto image = solver::v42_product(scalar, scalar_starts, scalar_rows, ones, ones, ones, column);
      for (std::size_t i = 0; i < image.size(); ++i) image_residual[i] -= coefficients[j] * image[i];
    }
    require(image_residual == ld_residual, "缓存图像线性组合错误");
    const std::array<double, 3> unit_tolerance{1, 1, 1};
    const auto difference = solver::v42_residual_difference(scalar, image_residual, current_residual, unit_tolerance);
    require(std::abs(difference.weighted_l2 - std::sqrt(2.L)) < 1e-18L && difference.scaled_inf == 1,
        "残差向量差应直接计算");
    const double maximum = std::numeric_limits<double>::max();
    const auto overflow = solver::v42_reconstruct(std::vector<double>{maximum},
        std::vector<std::vector<double>>{{maximum}}, std::vector<double>{1});
    require(overflow.ld_sum_finite && !overflow.once_cast_finite, "转回 double 溢出未标记");
    const auto underflow = solver::v42_reconstruct(std::vector<double>{0},
        std::vector<std::vector<double>>{{std::numeric_limits<double>::denorm_min()}}, std::vector<double>{.25});
    require(underflow.once_cast_finite && underflow.once_cast_underflows == 1, "转回 double 下溢未标记");
    const solver::detail::IpmFullNewtonLayout equality{1, 1, {}};
    const auto empty_block = solver::v42_metrics(equality, std::vector<long double>{3, 4}, unit_tolerance);
    require(empty_block.weighted_l2 == 5 && empty_block.block_scaled_inf[2] == 0, "空互补块错误");

    solver::v42_state.enabled = false;
    const auto events_before = solver::v42_state.event_id;
    const auto scans_before = solver::v42_state.work.csc_started;
    bool invoked = false;
    solver::v42_observe([&] { invoked = true; });
    solver::v42_phase("关闭测试", 99);
    solver::v42_preconditioner_begin();
    solver::v42_product_begin();
    solver::v42_assessed({}, {}, 0);
    require(!invoked && solver::v42_state.event_id == events_before &&
        solver::v42_state.work.csc_started == scans_before && solver::v42_state.iteration == 0,
        "关闭路径仍执行诊断");
    solver::v42_state.enabled = true;
    solver::v42_state.solve_start = solver::Clock::now();
    // 接线测试使用同一解析例，明确标作自测，不冒充实际 FGMRES 轨迹。
    solver::ConeModel model;
    model.starts = scalar_starts; model.rows = scalar_rows; model.values = ones;
    model.costs = {0}; model.rhs = {0}; model.inequality = {true}; model.barrier_rows = 1;
    const solver::Point point{{0}, {1}, {1}};
    solver::SolveResult result;
    result.ipm_full_newton_preconditioner_solves = 11;
    result.ipm_full_newton_products = 13;
    result.ipm_refinement_steps = 7;
    result.ipm_weighted_krylov_steps = 5;
    result.ipm_factorizations = 2;
    result.ipm_primal_regularization = 1e-8;
    result.ipm_dual_regularization = 1e-8;
    solver::detail::IpmFullNewtonWork observed;
    solver::v42_phase("self_test", 1);
    {
      solver::V42DirectionScope scope(model, point, scalar, rhs, result);
      solver::v42_helper_begin(std::array<double, 3>{.5, .5, .5});
      const auto base_residual = residual_of(base);
      solver::v42_restart(0, base, base_residual, observed);
      for (std::size_t j = 0; j < directions.size(); ++j) {
        solver::v42_arnoldi_begin(j);
        const std::vector<long double> column(directions[j].begin(), directions[j].end());
        const auto image = solver::v42_product(scalar, scalar_starts, scalar_rows, ones, ones, ones, column);
        solver::v42_arnoldi_image(j, image, observed);
        solver::v42_arnoldi_end(1, 0);
      }
      const auto before_candidate = solver::v42_state.work;
      solver::v42_candidate_prepare(directions, coefficients, 0, 1, true);
      solver::v42_assessed(current, current_residual, 2);
      require(solver::v42_state.work.j_started == before_candidate.j_started + 2 &&
          solver::v42_state.work.j_completed == before_candidate.j_completed + 2 &&
          solver::v42_state.work.csc_completed == before_candidate.csc_completed + 2,
          "候选接线没有执行且仅执行两次额外 J");
      require(solver::v42_state.candidates_started == 1 && solver::v42_state.candidates_completed == 1 &&
          !solver::v42_state.candidate_pending && solver::v42_state.last_metrics.scaled_inf == 2,
          "候选阶段或真实验收指标接线错误");
      require(*result.ipm_full_newton_preconditioner_solves == 11 && *result.ipm_full_newton_products == 13 &&
          *result.ipm_refinement_steps == 7 && *result.ipm_weighted_krylov_steps == 5,
          "诊断更改了生产工作量");
      try { throw solver::NumericalFailure("解析接线例的候选未通过"); }
      catch (...) { solver::v42_direction_failure(observed); }
    }
    require(solver::v42_state.terminal && !solver::v42_state.active, "接线例失败终点缺失");
    solver::v42_state.result = nullptr;
    solver::V42Event range("range_test");
    range.number("overflow", std::numeric_limits<long double>::max());
    range.number("underflow", std::numeric_limits<long double>::min());
    range.number("infinite", std::numeric_limits<long double>::infinity());
    range.number("zero", 0);
    range.emit();
    solver::v42_observe([] { throw std::runtime_error("诊断故障自测"); });
    require(!solver::v42_state.enabled, "诊断异常没有关闭观测");
    std::cout << "{\"passed\":true,\"checks\":" << checks
              << ",\"long_double_digits\":" << std::numeric_limits<long double>::digits << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    std::cout << "{\"passed\":false,\"checks\":" << checks << "}\n";
    return 1;
  }
}
