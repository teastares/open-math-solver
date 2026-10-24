#include "v71-basis-capture.hpp"
#include "v70-basis-recovery.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--case") throw std::runtime_error("缺少自测用例");
    const std::string which = argv[2];
    const bool candidate = which == "candidate", empty = which == "no_candidate";
    const bool cap = which == "cache_limit", expired = which == "budget_interrupted";
    if (!candidate && !empty && !cap && !expired) throw std::runtime_error("未知自测用例");
    const std::vector<std::int64_t> starts{0, 2, 4}, rows{0, 2, 1, 2};
    const std::vector<double> values{1, 2, 3, 4}, costs{5, 6}, rhs{7, 8, 9};
    const std::vector<double> slack{1, 2, 3}, dual{4, 5, 6};
    const std::vector<bool> inequality{false, true, true};
    const solver::v70_basis::Input input{starts, rows, values, costs, rhs, slack, dual, inequality};
    solver::v70_basis::Result result;
    result.candidate_available = true;
    result.x = {0.5, -0.25}; result.s = {0, 1.5, 2}; result.z = {-1, 2, 0};
    result.stats.selected_rows = {0, 1}; result.stats.pivot_columns = {1, 0};
    {
      solver::v71_basis_capture::Report report;
      const bool enabled = report.state.enabled;
      if (cap) report.state.limit = report.state.fixed_cache_bytes + 1;
      struct Expired {};
      int checks = 0;
      bool threw = false;
      if (!empty) {
        try {
          solver::v71_basis_capture::capture(input, result, [&] {
            if (++checks == 6 && expired) throw Expired{};
          });
        } catch (const Expired&) { threw = true; }
      }
      if (threw != (enabled && expired)) throw std::runtime_error("预算异常传播不符");
      if (!enabled && (checks || report.state.attempts || report.state.payload))
        throw std::runtime_error("关闭捕获仍执行复制");
      if (enabled && candidate && !report.state.complete) throw std::runtime_error("正常捕获未完成");
      if (enabled && cap && (!report.state.truncated || report.state.capture_errors != 1))
        throw std::runtime_error("容量拒绝未记账");
      if (enabled && expired && (report.state.complete || report.state.capture_errors != 1))
        throw std::runtime_error("中断捕获被标记完整");
      if (result.x != std::vector<double>{0.5, -0.25} || result.s != std::vector<double>{0, 1.5, 2} ||
          result.z != std::vector<double>{-1, 2, 0}) throw std::runtime_error("捕获修改了候选");
    }
    std::puts("{\"passed\":true,\"public_lp_runs\":0}");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "捕获自测失败：%s\n", error.what());
    return 1;
  }
}
