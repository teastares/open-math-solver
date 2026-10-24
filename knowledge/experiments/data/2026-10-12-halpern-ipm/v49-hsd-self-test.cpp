// 独立诊断接线自测；只使用手算矩阵，不运行 LP 或生产求解器。
#include "v49-hsd-diagnostic.hpp"
#include <iostream>
#include <stdexcept>

namespace {
struct Model {
  std::vector<std::int64_t> starts{0, 1}, rows{0};
  std::vector<double> values{1}, costs{-1}, rhs{2};
  std::vector<bool> inequality{true};
};
struct Point { std::vector<double> s{1}, z{2}; };
struct Result {
  std::optional<double> ipm_primal_regularization{0.25}, ipm_dual_regularization{0.25};
  std::optional<std::int64_t> ipm_iterations{0}, ipm_factorizations{1}, ipm_refinement_steps{0};
  std::optional<std::int64_t> ipm_homogeneous_constant_solves{1}, ipm_homogeneous_direction_solves{1};
};
}

int main(int argc, char**) {
  if (argc != 1) return 2;
  try {
    std::size_t checks = 0;
    const auto check = [&](bool condition) {
      ++checks;
      if (!condition) throw std::runtime_error("HSD 诊断接线自测失败");
    };
    solver::v49hsd::begin_run();
    auto& state = solver::v49hsd::state();
    const bool enabled = state.enabled;
    const long double target = 1.0L + std::ldexp(1.0L, 1 - std::numeric_limits<long double>::digits);
    const bool non_double_expected = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;
    check(target > 1);
    check(!non_double_expected || static_cast<long double>(static_cast<double>(target)) != target);
    Model model;
    Point point;
    Result result;
    const std::vector<double> diagonal{0.5}, regularized{0.75}, u{1.5, -1}, v{2.5, 1};
    solver::v49hsd::phase("self_test", 1);
    {
      solver::v49hsd::Scope scope(model, point, diagonal, regularized, 0, u, v, 1, 1, result);
      check((state.context != nullptr) == enabled);
      solver::v49hsd::ScalarObservation observation(target, -1);
      observation.values(-3.5L, -0.5L, 1.5L);
      // 使用实际异常捕获接线，但不改变生产的异常类型或控制流。
      try { throw std::runtime_error("synthetic_first"); }
      catch (const std::runtime_error& error) { observation.failure(error.what()); }
      if (enabled) {
        check(state.attempted && state.snapshot && std::string(state.status) == "captured");
        check(state.snapshot->gap_rhs == target && state.snapshot->scalar_values_available);
        check(state.snapshot->dot_v == -0.5L && state.snapshot->denominator == 1.5L);
        check(std::string(state.snapshot->failure.data()) == "synthetic_first");
        check(state.snapshot->bytes + solver::v49hsd::metadata_reserve <= solver::v49hsd::cache_limit);
        check(state.snapshot->context.fields.front().offset == 0);
        const auto before = state.snapshot.get();
        solver::v49hsd::ScalarObservation second(8, 9);
        second.values(10, 11, 12);
        second.failure("synthetic_second");
        check(state.snapshot.get() == before && state.snapshot->gap_rhs == target);
        check(std::string(state.snapshot->failure.data()) == "synthetic_first");
      } else {
        check(!state.attempted && !state.snapshot && state.context == nullptr);
        check(std::string(state.status) == "no_scalar_failure");
      }
    }
    check(state.context == nullptr);
    std::cout.imbue(std::locale::classic());
    std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"mode\":\"" << (enabled ? "on" : "off")
        << "\",\"lp_runs\":0,\"long_double_digits\":" << std::numeric_limits<long double>::digits
        << ",\"non_double_expected\":" << (non_double_expected ? "true" : "false") << ",\"target_ld\":";
    solver::v49hsd::real_hex(std::cout, target);
    std::cout << ",\"expected_exact_denominator\":\"3/2\",\"expected_gap_rhs_exponent\":"
        << 1 - std::numeric_limits<long double>::digits << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
