// 合成观察流仅检验采集边界和原始字节；不执行 LP 或额外算子。
#define SOLVER_V67_SELFTEST
#include "v67-krylov-diagnostic.hpp"
#include <iostream>
#include <optional>
#include <string>

namespace d = solver::v67krylov;
struct Model {
  std::vector<std::int64_t> starts{0, 2, 3}, rows{0, 1, 1};
  std::vector<double> values{2, -3, 4}, costs{1, 2}, rhs{3, 4};
};
struct Result {
  std::optional<std::int64_t> ipm_iterations = 7, ipm_factorizations = 9, ipm_refinement_steps = 0;
  std::optional<std::int64_t> ipm_homogeneous_constant_solves = 8, ipm_homogeneous_direction_solves = 14;
  std::optional<std::int64_t> ipm_homogeneous_constant_repairs = 0, ipm_homogeneous_constant_repair_successes = 0;
};
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    require(argc == 3 && std::string(argv[1]) == "--case", "自测参数错误");
    const std::string test = argv[2];
    require(test == "captured" || test == "no_trigger" || test == "nonterminal" ||
        test == "prepared_nonterminal" || test == "cache_limit" || test == "short_cycle" ||
        test == "missing_candidate" || test == "event_limit", "未知自测场景");
    Model model;
    Result result;
    const std::vector<double> h{5, 6}, rhs{-1, -2, 3, 4}, base{-0.0, 2, 3, 4};
    const std::vector<std::vector<double>> z{{1, 2, 3, 4}, {5, 6, 7, 8}};
    const std::vector<double> coefficients{0.5, -0.25}, candidate{-0.75, 1.5, 2.75, 4};
    const std::vector<double> residual{0.0, -0.0, 0x1p-1074, -0x1p-1074};
    const auto original_values = model.values;
    const auto original_rhs = rhs;
    if (test == "cache_limit") d::state.test_cache_limit = sizeof(d::State);
    d::begin_run();
    bool caught = false;
    try {
      d::Scope scope(model, h, rhs, 8, result);
      d::ir_candidate();
      if (test != "no_trigger") {
        d::ir_correction_started();
        ++*result.ipm_refinement_steps;
        d::ir_correction_completed();
        d::ir_candidate();
        d::krylov_begin();
        for (int restart = 0; restart < 3; ++restart) {
          d::cycle_begin(restart);
          const std::size_t count = test == "short_cycle" ? 1 : 2;
          for (std::size_t col = 0; col < count; ++col) {
            d::column_begin(restart);
            if (test == "nonterminal" && restart == 2 && col == 0) throw std::runtime_error("合成中断");
            d::preconditioner_complete(restart);
            ++*result.ipm_refinement_steps;
            d::column_complete(restart, test == "short_cycle" ? 0.0 : 1.0);
          }
          const std::vector<std::vector<double>> used_z(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(count));
          const std::span<const double> used_coeff(coefficients.data(), count);
          if (test != "missing_candidate") d::candidate(restart, count - 1, used_coeff);
          d::cycle_end(restart, base, used_z, candidate, residual, 0x1p-20, nullptr);
        }
        if (test == "prepared_nonterminal") throw std::runtime_error("合成末周期后中断");
        d::terminal_exhausted();
        throw std::runtime_error("合成原终端异常");
      }
    } catch (const std::runtime_error&) { caught = true; }
    const auto expected_refinements = test == "no_trigger" ? 0 : test == "short_cycle" ? 4 : test == "nonterminal" ? 5 : 7;
    require(*result.ipm_refinement_steps == expected_refinements, "观察改变了工作计数");
    require(caught == (test != "no_trigger"), "观察改变了原异常流");
    require(model.values == original_values && rhs == original_rhs, "观察修改了输入");
    if (d::state.enabled) {
      require(!d::state.active && !d::state.prepared, "观察作用域没有闭合");
      require(d::state.ordinary_scopes_started == 1 &&
          d::state.ordinary_scopes_completed + d::state.ordinary_scopes_unwound == 1, "作用域账本错误");
      if (test == "no_trigger" || test == "nonterminal" || test == "prepared_nonterminal") {
        require(!d::state.terminal && !d::state.event_available && !d::state.capture_errors, "非终端被误作捕获");
        require(d::state.capture_started == (test == "prepared_nonterminal" ? 1U : 0U), "准备次数错误");
      } else if (test == "cache_limit" || test == "missing_candidate") {
        require(d::state.terminal && !d::state.event_available && d::state.capture_errors == 1, "缺失捕获未显式报错");
        require(d::state.truncated == (test == "cache_limit"), "截断标识错误");
      } else {
        require(d::state.terminal && d::state.event_available && !d::state.capture_errors, "完整事件未提交");
        const auto& event = d::state.event;
        require(event.generated_columns == (test == "short_cycle" ? 1U : 2U) &&
            event.coefficient_count == event.generated_columns, "实际列数错误");
        require(event.before_counts.refinements == 0 && event.after_counts.refinements == expected_refinements, "阶段计数错误");
        const auto& saved_base = event.arrays[5];
        const auto& saved_residual = event.arrays[9];
        require(std::memcmp(event.payload.get() + saved_base.offset - 8, base.data(), base.size() * 8) == 0,
            "基点原始字节损失");
        require(std::memcmp(event.payload.get() + saved_residual.offset - 8, residual.data(), residual.size() * 8) == 0,
            "残差符号零或低位损失");
        require(event.payload_bytes + sizeof(d::State) <= d::cache_limit, "捕获越过缓存上限");
        std::size_t offset = 8;
        for (const auto& field : event.arrays) {
          require(field.offset == offset && field.source == nullptr, "描述偏移或借用指针错误");
          offset += field.count * field.width;
        }
        require(offset == 8 + event.payload_bytes, "载荷大小错误");
      }
      if (test == "event_limit") {
        const auto* first = d::state.event.payload.get();
        { d::Scope second(model, h, rhs, 9, result); d::krylov_begin(); d::terminal_exhausted(); }
        require(d::state.event.payload.get() == first && d::state.terminal_failures == 1 &&
            d::state.ordinary_scopes_started == 1, "首事件上限没有保持");
      }
    } else {
      require(!d::state.run_started && !d::state.event_available && !d::state.capture_started &&
          !d::state.ordinary_scopes_started && !d::state.event.payload, "关闭采集仍修改观察状态");
    }
    // 关闭与开启都打印相同的模拟生产结果，外部驱动另核文件和错误标识。
    std::cout << "{\"passed\":true,\"case\":\"" << test << "\",\"refinements\":" << expected_refinements
              << ",\"caught\":" << (caught ? "true" : "false") << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "采集自测失败：" << error.what() << '\n';
    return 1;
  }
}
