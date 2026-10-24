// 独立采集接线自测；只用合成数组，不链接求解器，不运行 LP。
#include "v59-kkt-diagnostic.hpp"
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {
struct Model {
  std::vector<std::int64_t> starts{0, 2}, rows{0, 1};
  std::vector<double> values{2, 3}, costs{-1}, rhs{2, 3};
};
struct Result {
  std::optional<double> ipm_primal_regularization{0.25}, ipm_dual_regularization{0.5};
  std::optional<std::string> ipm_factor_backend{"normal_cholmod"};
};
double saved(const solver::v59kkt::State& state, std::size_t field, std::size_t position) {
  double value = 0;
  std::memcpy(&value, state.pending_payload.get() + state.fields[field].offset + position * 8, 8);
  return value;
}
}

int main(int argc, char**) {
  if (argc != 1) return 2;
  try {
    std::size_t checks = 0;
    const auto check = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("v59 诊断接线自测失败"); };
    using namespace solver::v59kkt;
    auto& observation = state(); const bool enabled = observation.enabled;
    const Model model; const Result result;
    const std::vector<double> h{4, 5}, hreg{4.5, 5.5}, rhs{1, 2, 3};
    const std::vector<std::int64_t> ap{0, 1, 3}, ai{0, 0, 1}, retained{-1, 1}, singleton{0, -1};
    const std::vector<std::size_t> inverse{1, 0};
    const std::vector<double> ax{-5.5, 3, 0.25 + 4.0 / 4.5};
    const std::vector<double> values{-22, 3, ax[2] * 0.25}, eq{2, 0.5}, singleton_value{2, 0};
    const std::vector<double> first_answer{0.125, (0.25 - 2) / 4.5, 1};
    const std::vector<double> terminal{0.5, -0.25, 1}, residual{-1.5, 0, 6.5};
    {
      RunScope run;
      check(observation.ready == enabled);
      // 未处在正式 constant scope 的 direct 不采集。
      { DirectScope ignored(rhs, 2); ignored.complete(first_answer); }
      check(observation.copied_bytes == 0);
      // 首次因子求解异常时，不把尚未成功返回的向量标为输出。
      {
        ConstantScope constant(0);
        try {
          DirectScope direct(rhs, 2);
          reduced_rhs(0, 1); reduced_rhs(1, 2);
          std::vector<double> interrupted{3, 4};
          FactorScope factor(interrupted);
          interrupted[0] = 99;
          throw std::runtime_error("合成中断");
        } catch (const std::runtime_error&) {}
        if (enabled) check(observation.have_first_rhs && !observation.have_first_solution && !observation.have_first_answer);
      }
      for (std::int64_t iteration = 1; iteration <= 3; ++iteration) {
        ConstantScope constant(iteration);
        DirectScope direct(rhs, 2);
        reduced_rhs(0, 1 + 4.0 / 4.5); reduced_rhs(1, 3);
        std::vector<double> adapter_rhs{6, (1 + 4.0 / 4.5) * 0.5};
        {
          FactorScope factor(adapter_rhs);
          adapter_rhs = {0.5, 0.25};
        }
        direct.complete(first_answer);
        // 完成首次 direct 后的内部求解不覆盖首次输入输出。
        {
          DirectScope ignored_direct(rhs, 2);
          reduced_rhs(0, 99); reduced_rhs(1, 98);
          { std::vector<double> later{99, 98}; FactorScope ignored(later); later[0] = 97; }
          ignored_direct.complete(terminal);
        }
        if (enabled) {
          check(observation.ready);
          check(observation.have_first_rhs && observation.have_first_solution && observation.have_first_answer);
          check(saved(observation, first_rhs, 0) == 6 && saved(observation, first_solution, 0) == 0.5);
          check(saved(observation, first_reduced, 0) == 1 + 4.0 / 4.5);
          check(saved(observation, first_full, 1) == first_answer[1]);
        }
        if (iteration == 3) {
          observation.failure(model, h, hreg, ap, ai, ax, values, eq, inverse, retained, singleton,
              singleton_value, rhs, terminal, residual, 3.01e-10, 0, result);
          if (enabled) {
            check(observation.captured && observation.capture_count == 1 && observation.target_failures == 1);
            const auto copied = observation.copied_bytes;
            observation.failure(model, h, hreg, ap, ai, ax, values, eq, inverse, retained, singleton,
                singleton_value, rhs, terminal, residual, 3.01e-10, 0, result);
            check(observation.copied_bytes == copied && observation.capture_count == 1);
          }
        }
      }
      if (enabled) {
        check(observation.iteration == 3 && observation.constant_calls == 4);
        check(observation.cache_bytes == metadata_reserve + observation.pending_bytes + observation.terminal_bytes);
        check(observation.cache_bytes <= cache_limit);
        check(observation.fields[0].count == 2 && observation.fields[22].count == 3);
      } else {
        check(!observation.run_started && !observation.pending_payload && !observation.terminal_payload);
        check(observation.copied_bytes == 0 && observation.capture_count == 0 && observation.constant_calls == 0);
      }
    }
    check(!observation.constant_active && !observation.direct_active && !observation.run_active);
    // 独立状态只测接线边界，不写出额外文件，不改变正式状态。
    {
      State isolated; isolated.enabled = true; isolated.ready = true; isolated.run_active = true;
      const std::vector<double> no_singletons{1.25, -2};
      check(isolated.begin_constant(1));
      check(isolated.begin_direct(no_singletons, 2));
      check(isolated.reduced_written == 2 && saved(isolated, first_reduced, 1) == -2);
      check(isolated.factor_input(no_singletons));
      isolated.factor_output(no_singletons); isolated.complete_direct(no_singletons);
      check(isolated.have_first_answer && !isolated.direct_active);
      isolated.constant_active = false;
      check(isolated.begin_constant(2));
      check(!isolated.begin_direct(rhs, 2) && !isolated.ready);
      isolated.enabled = false;
    }
    {
      State isolated; isolated.enabled = false;
      std::size_t bytes = cache_limit; Field field;
      check(!isolated.add(field, "too_large", "f64_le", 1, 8, bytes));
      check(bytes == cache_limit);
      bytes = 0;
      check(!isolated.add(field, "overflow", "f64_le", std::numeric_limits<std::size_t>::max(), 8, bytes));
      check(bytes == 0);
    }
    std::cout << "{\"passed\":true,\"mode\":\"" << (enabled ? "on" : "off") << "\",\"checks\":" << checks
        << ",\"lp_runs\":0,\"expected_capture_count\":" << (enabled ? 1 : 0)
        << ",\"expected_iteration\":3,\"expected_array_count\":23}\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
