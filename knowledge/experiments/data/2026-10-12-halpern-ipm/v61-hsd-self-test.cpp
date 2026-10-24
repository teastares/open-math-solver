// 独立诊断接线自测；只使用手算数组，不运行 LP 或生产求解器。
#include "v61-hsd-diagnostic.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
struct Model {
  std::vector<std::int64_t> starts{0, 1}, rows{0};
  std::vector<double> values{1}, costs{-1}, rhs{2};
  std::vector<bool> inequality{true};
};
struct Point { std::vector<double> x{4}, s{1}, z{2}; };
struct Result {
  std::optional<double> ipm_primal_regularization{0.25}, ipm_dual_regularization{0.25};
  std::optional<std::int64_t> ipm_iterations{2}, ipm_factorizations{3}, ipm_refinement_steps{4};
  std::optional<std::int64_t> ipm_homogeneous_constant_solves{3}, ipm_homogeneous_direction_solves{5};
  std::optional<std::int64_t> ipm_homogeneous_constant_repairs{0}, ipm_homogeneous_constant_repair_successes{0};
  std::optional<std::int64_t> ipm_extended_krylov_attempts{2}, ipm_extended_krylov_successes{2};
};
}

int main(int argc, char** argv) {
  const std::string test_case = argc == 1 ? "success" : argc == 3 && std::string(argv[1]) == "--case" ? argv[2] : "invalid";
  if (test_case != "success" && test_case != "failed_candidate" && test_case != "failed_solve" &&
      test_case != "no_trigger" && test_case != "event_limit" && test_case != "cache_limit") return 2;
  try {
    std::size_t checks = 0;
    const auto check = [&](bool condition) {
      ++checks;
      if (!condition) throw std::runtime_error("v61 齐次诊断合成自测失败");
    };
    solver::v61hsd::begin_run();
    auto& state = solver::v61hsd::state();
    const bool enabled = state.enabled;
    check(state.count == 0 && state.trigger_count == 0 && state.cache_bytes == 0);
    const long double target = 1.0L + std::ldexp(1.0L, 1 - std::numeric_limits<long double>::digits);
    check(target > 1);
    const bool wider = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;
    check(!wider || static_cast<long double>(static_cast<double>(target)) != target);
    std::ostringstream text;
    solver::v61hsd::real_hex(text, target);
    const auto representation = text.str();
    const auto body = representation.substr(1, representation.size() - 2);
    char* tail = nullptr;
    check(std::strtold(body.c_str(), &tail) == target && tail && *tail == 0);
    Model model;
    Point point;
    Result result;
    const std::vector<double> diagonal{0.5}, regularized{0.75}, rhs{1.5, -1}, h{-2}, primal{-1}, dual{1.5};
    std::vector<long double> u{target, -1}, before{2, 2};
    const auto repetitions = test_case == "event_limit" ? solver::v61hsd::event_limit + 1 : 1;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
      solver::v61hsd::phase("self_test", static_cast<std::int64_t>(3 + repetition));
      {
        solver::v61hsd::Scope scope(model, point, diagonal, regularized, 1, u, rhs, h, primal, dual,
            1, 1, 0.5, result);
        check((state.context != nullptr) == enabled);
        solver::v61hsd::observe_values<long double>(-target - 2, test_case == "no_trigger" ? -2 : 2,
            test_case == "no_trigger" ? 3 : -1);
        if (test_case != "no_trigger") {
          const auto original_cache = state.cache_bytes;
          if (test_case == "cache_limit") state.cache_bytes = solver::v61hsd::cache_limit - solver::v61hsd::metadata_reserve;
          solver::v61hsd::begin_repair<long double>(u, before, target, -1);
          if (test_case == "cache_limit") state.cache_bytes = original_cache;
          ++*result.ipm_homogeneous_constant_repairs;
          ++*result.ipm_homogeneous_constant_solves;
          if (test_case == "failed_solve") {
            try { throw std::runtime_error("synthetic_solve_failure"); }
            catch (const std::exception& error) { solver::v61hsd::failure(error.what()); }
          } else {
            const std::vector<long double> after = test_case == "failed_candidate" ?
                std::vector<long double>{3, 3} : std::vector<long double>{target, 0};
            solver::v61hsd::candidate(after);
            solver::v61hsd::observe_values<long double>(-target - 2, test_case == "failed_candidate" ? 3 : -target,
                test_case == "failed_candidate" ? -2 : 1 + target);
            if (test_case == "failed_candidate") solver::v61hsd::failure("synthetic_scalar_failure");
            else {
              solver::v61hsd::committed();
              ++*result.ipm_homogeneous_constant_repair_successes;
            }
          }
          if (enabled && state.current_event) {
            const auto& event = *state.current_event;
            check(event.before.denominator == -1 && event.before.dot_v == 2);
            check(event.u[0] == target && event.gap_rhs == target);
            check(event.before_v[0] == 2 && event.before_v[1] == 2);
            check(event.context.fields.front().offset >= 8);
            check(event.context.fields.front().data == nullptr && event.context.result == nullptr);
            check(event.candidate_available == (test_case != "failed_solve"));
            check(event.after_scalar_available == (test_case != "failed_solve"));
            check(event.committed == (test_case != "failed_solve" && test_case != "failed_candidate"));
            if (test_case == "failed_candidate") {
              check(event.after.denominator == -2 && event.after_v[0] == 3);
              check(std::string(event.failure.data()) == "synthetic_scalar_failure");
            } else if (test_case == "failed_solve") {
              check(std::string(event.failure.data()) == "synthetic_solve_failure");
            } else {
              check(event.after_v[0] == target && event.after.denominator == 1 + target);
              check(std::string(event.failure.data()).empty());
            }
          }
        }
      }
      check(state.context == nullptr && state.current_event == nullptr);
      if (enabled && state.count) {
        check(state.events[state.count - 1]->after_counts.constant_repairs ==
            static_cast<std::int64_t>(std::min(repetition + 1, solver::v61hsd::event_limit)));
      }
    }
    check(state.cache_bytes + solver::v61hsd::metadata_reserve <= solver::v61hsd::cache_limit);
    check(state.trigger_count == state.count + state.dropped_events);
    if (!enabled) check(state.count == 0 && state.trigger_count == 0 && state.payload_bytes == 0 && state.cache_bytes == 0);
    else if (test_case == "no_trigger") check(state.count == 0 && std::string(state.status) == "no_repair_events");
    else if (test_case == "cache_limit") check(state.count == 0 && state.dropped_events == 1 && std::string(state.status) == "cache_limit");
    else if (test_case == "event_limit") check(state.count == 8 && state.dropped_events == 1 && std::string(state.status) == "event_limit");
    else check(state.count == 1 && state.dropped_events == 0 && std::string(state.status) == "captured");
    // 源数组修改后，归档仍须保留实际捕获时的宽低位及工作点。
    u[0] = 8; before[0] = 9; point.x[0] = 10;
    if (enabled && state.count) {
      check(state.events[0]->u[0] == target && state.events[0]->before_v[0] == 2);
      const auto& xfield = state.events[0]->context.fields[5];
      double copied_x = 0;
      std::memcpy(&copied_x, state.events[0]->payload.get() + xfield.offset - 8, sizeof(double));
      check(copied_x == 4);
    }
    std::cout.imbue(std::locale::classic());
    std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"case\":\"" << test_case
        << "\",\"mode\":\"" << (enabled ? "on" : "off") << "\",\"lp_runs\":0,\"long_double_digits\":"
        << std::numeric_limits<long double>::digits << ",\"wide_target\":";
    solver::v61hsd::real_hex(std::cout, target);
    std::cout << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
