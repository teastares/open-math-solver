// 独立解析构造检验观测恒等式和不同原始/对偶步长，不调用数值分解。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "instrumented.cpp"

int main() {
  solver::ConeModel model;
  model.starts = {0, 2, 4};
  model.rows = {0, 1, 0, 1};
  model.values = {1, 3, 2, -1};
  model.costs = {1, -1};
  model.rhs = {5, 0};
  model.inequality = {true, false};
  model.barrier_rows = 1;
  solver::Point point{{4, 1}, {2, 0}, {3, -2}};
  const solver::Point delta{{-1, 2}, {3, 0}, {.5, -1}};
  const std::vector<double> rp{3, 11}, rd{-2, 7}, diagonal{2.0 / 3.0, 0}, h{1, 0};
  solver::SolveOptions options;
  solver::SolveResult result;
  result.ipm_iterations = 0;
  result.ipm_factorizations = 0;
  result.ipm_refinement_steps = 0;
  result.ipm_direction_retries = 0;
  result.ipm_direction_retry_successes = 0;
  solver::v40_state.enabled = true;
  solver::v40_run_start(options, solver::Clock::now());
  solver::v40_begin(model, point, rp, rd, 6, 1, options, result);
  solver::v40_before_step(model, point, delta, rp, rd, diagonal, h,
                         .25, .5, 0, 6, 1, 0, options, result);
  for (std::size_t col = 0; col < point.x.size(); ++col)
    point.x[col] = static_cast<double>(static_cast<long double>(point.x[col]) + .25 * delta.x[col]);
  for (std::size_t row = 0; row < point.s.size(); ++row) {
    point.s[row] = static_cast<double>(static_cast<long double>(point.s[row]) + .25 * delta.s[row]);
    point.z[row] = static_cast<double>(static_cast<long double>(point.z[row]) + .5 * delta.z[row]);
  }
  result.ipm_iterations = 1;
  solver::v40_after_step(model, point, 1, options, result);
  solver::V40Event range("range_test");
  range.number("overflow", std::numeric_limits<long double>::max());
  range.number("underflow", std::numeric_limits<long double>::min());
  range.number("infinite", std::numeric_limits<long double>::infinity());
  range.emit();
  const auto scans = solver::v40_state.csc_scans;
  const auto events = solver::v40_state.event_id;
  solver::v40_state.enabled = false;
  solver::v40_begin(model, point, rp, rd, 6, 2, options, result);
  const bool unchanged = scans == solver::v40_state.csc_scans && events == solver::v40_state.event_id;
  std::cout << "{\"disabled_no_scans\":" << (unchanged ? "true" : "false") << "}\n";
  return unchanged ? 0 : 1;
}
