#include "solver/backend.hpp"
#include "solver/pdlp.hpp"

#include <cmath>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_optimum(const solver::SolveResult& result, double expected) {
  require(result.status == "optimal", "应返回最优：" + result.status + "；" + result.message);
  require(result.has_solution && result.objective.has_value(), "最优结果必须含有效候选");
  require(std::abs(*result.objective - expected) < 1e-7, "最优目标与手算值不同");
  require(result.verification.valid_input && result.verification.feasible &&
              result.verification.objective_consistent, "候选必须通过原模型检查");
  if (solver::is_mip(result.model)) {
    require(result.verification_level == "primal_checked_backend_claim", "不能把 MIP 后端结论标成自主证明");
  } else {
    require(result.verification_level == "primal_dual_residuals_checked_backend_claim" &&
                result.lp_optimality && result.lp_optimality->numerically_optimal,
            "HiGHS LP 必须通过共同原始/对偶数值检查");
    require(solver::verify_lp_optimality(result.model, result.solution, result.row_dual).numerically_optimal,
            "最大化归一化后的行对偶必须可重新独立检查");
  }
  require(result.best_bound && std::abs(*result.best_bound - expected) < 1e-7,
          "后端界必须保留原目标方向和常数");
  require(result.relative_gap && *result.relative_gap < 1e-7, "最优模型间隙应为零");
  require(result.total_seconds >= result.solve_seconds &&
              result.total_seconds >= result.read_seconds,
          "计时必须覆盖实际读取与求解");
}

void require_rejected_options(const solver::SolveResult& result,
                              const std::string& message_fragment) {
  require(result.status == "error" && !result.has_solution && !result.objective &&
              !result.best_bound && result.solve_seconds == 0.0 &&
              result.message.find(message_fragment) != std::string::npos,
          "无效后端选项必须在求解前明确拒绝：" + result.status + "；" + result.message);
}

}  // 匿名命名空间

int main(int argc, char** argv) {
  try {
    require(argc <= 2, "测试至多接受一个样例目录参数");
    const std::filesystem::path fixtures(argc == 2 ? argv[1] : SOLVER_FIXTURE_DIR);
    const auto solve = [&](const char* name, const solver::SolveOptions& options = {}) {
      return solver::solve_file(fixtures / name, options);
    };
    require(!solver::backend_version().empty(), "必须报告后端版本");
    const auto lp = solve("lp_min.lp");
    require_optimum(lp, 4);
    require(lp.model.costs.size() == 2 && lp.model.row_lower.size() == 1,
            "必须保存原模型而非预处理后的模型");
    require_optimum(solve("lp_min.mps"), 4);
    // 固定字段中的空格不改变模型语义；目标行故意不放在首行。
    require_optimum(solve("fixed_spaces.mps"), 13.5);
    for (const char* backend : {"highs", "pdlp", "ipm"}) {
      solver::SolveOptions read_only;
      read_only.backend = backend;
      read_only.time_limit = 0;
      const auto fixed = solve("fixed_spaces.mps", read_only);
      require(fixed.status == "time_limit" && !fixed.has_solution &&
                  !fixed.objective && !fixed.best_bound,
              "固定格式重读必须保留零预算语义：" + fixed.message);
      const auto& model = fixed.model;
      const double infinity = std::numeric_limits<double>::infinity();
      require(model.sense == solver::ObjectiveSense::minimize &&
                  model.objective_offset == 7 &&
                  model.costs == std::vector<double>({1, 2, 3}) &&
                  model.column_lower == std::vector<double>({0, -infinity, 1}) &&
                  model.column_upper == std::vector<double>({4, infinity, 1}) &&
                  model.row_lower == std::vector<double>({4, -2, 1}) &&
                  model.row_upper == std::vector<double>({infinity, 2, 1}) &&
                  model.kinds == std::vector<solver::VariableKind>(3, solver::VariableKind::continuous),
              "重读必须完整保留目标常数、行范围、自由变量及固定界");
      require(model.column_start == std::vector<std::int64_t>({0, 2, 4, 6}) &&
                  model.row_index == std::vector<std::int64_t>({0, 1, 0, 1, 0, 2}) &&
                  model.values == std::vector<double>({1, 1, 1, -1, 1, 1}),
              "重读不能丢失或改变任何矩阵条目");
      require(fixed.read_seconds > 0 && fixed.total_seconds >= fixed.read_seconds,
              "格式切换与重读必须计入读取和总时间");
      for (const char* name : {"fixed_spaces_unknown_row.mps", "fixed_spaces_huge_bound.mps"}) {
        const auto rejected = solve(name, read_only);
        const std::string reason = std::string(name) == "fixed_spaces_unknown_row.mps" ?
            "读取产生警告" : "转换为无穷";
        require(rejected.status == "error" && !rejected.has_solution &&
                    !rejected.objective && !rejected.best_bound &&
                    rejected.model.costs.empty() && rejected.solve_seconds == 0 &&
                    rejected.read_seconds > 0 && rejected.total_seconds >= rejected.read_seconds &&
                    rejected.message.find(reason) != std::string::npos,
                "格式切换不能掩盖数据丢弃或有限值替换：" + std::string(name) + "；" + rejected.message);
      }
    }
    const auto mip = solve("milp_min.lp");
    require_optimum(mip, 2);
    require(solver::is_mip(mip.model), "必须保留整数变量类型");
    require(mip.nodes.has_value(), "MIP 必须报告有效节点计数");
    const auto max_lp = solve("max_offset.lp");
    require_optimum(max_lp, 17);
    require(max_lp.model.sense == solver::ObjectiveSense::maximize &&
                max_lp.model.objective_offset == 7,
            "必须保留最大化方向和目标常数");
    require_optimum(solve("milp_max_offset.lp"), 12);

    for (const char* algorithm : {"choose", "simplex", "ipm"}) {
      for (const char* crossover : {"on", "off"}) {
        solver::SolveOptions configured;
        configured.highs.solver = algorithm;
        configured.highs.run_crossover = crossover;
        if (configured.highs.solver == "ipm" && configured.highs.run_crossover == "off") {
          configured.highs.presolve = "off";
        }
        require_optimum(solve("lp_min.lp", configured), 4);
        require_optimum(solve("max_offset.lp", configured), 17);
        if (configured.highs.solver != "choose" ||
            configured.highs.run_crossover != "on") {
          require_rejected_options(solve("milp_min.lp", configured), "仅适用于连续 LP");
          require_rejected_options(solve("milp_max_offset.lp", configured), "仅适用于连续 LP");
        }
      }
    }
    solver::SolveOptions no_crossover_presolve;
    no_crossover_presolve.highs.solver = "ipm";
    no_crossover_presolve.highs.run_crossover = "off";
    const auto unrecovered_dual = solve("max_offset.lp", no_crossover_presolve);
    require(unrecovered_dual.status == "unknown" && unrecovered_dual.has_solution &&
                unrecovered_dual.lp_optimality && !unrecovered_dual.lp_optimality->numerically_optimal &&
                !unrecovered_dual.best_bound,
            "固定 HiGHS 版本的无基最大化 postsolve 缺陷不能被误报为最优");
    for (const char* presolve : {"on", "off"}) {
      solver::SolveOptions configured;
      configured.highs.presolve = presolve;
      require_optimum(solve("lp_min.lp", configured), 4);
      require_optimum(solve("max_offset.lp", configured), 17);
      require_rejected_options(solve("milp_min.lp", configured), "仅适用于连续 LP");
    }
    for (const char* algorithm : {"", "pdlp", "ipx", "hipo", "SIMPLEX"}) {
      solver::SolveOptions configured;
      configured.highs.solver = algorithm;
      require_rejected_options(solve("lp_min.lp", configured), "HiGHS 算法必须是");
    }
    for (const char* crossover : {"", "choose", "ON"}) {
      solver::SolveOptions configured;
      configured.highs.run_crossover = crossover;
      require_rejected_options(solve("lp_min.lp", configured), "HiGHS crossover 必须是");
    }
    for (const char* presolve : {"", "auto", "ON"}) {
      solver::SolveOptions configured;
      configured.highs.presolve = presolve;
      require_rejected_options(solve("lp_min.lp", configured), "HiGHS presolve 必须是");
    }
    for (const char* backend : {"pdlp", "ipm"}) {
      for (int selected_option : {0, 1, 2}) {
        solver::SolveOptions configured;
        configured.backend = backend;
        if (selected_option == 0) configured.highs.solver = "ipm";
        else if (selected_option == 1) configured.highs.run_crossover = "off";
        else configured.highs.presolve = "off";
        require_rejected_options(solve("lp_min.lp", configured), "仅适用于 highs 后端");
        if (configured.backend == "pdlp") {
          const auto direct = solver::solve_pdlp(lp.model, configured);
          require(direct.status == "error" && !direct.has_solution &&
                      direct.pdlp_iterations == 0 &&
                      direct.message.find("仅适用于 highs 后端") != std::string::npos,
                  "自主直接接口也必须拒绝未生效的 HiGHS 选项");
        }
      }
    }

    const auto infeasible = solve("infeasible.lp");
    require(infeasible.status == "infeasible", "矛盾约束应识别不可行");
    require(!infeasible.has_solution && !infeasible.objective &&
                !infeasible.absolute_gap && !infeasible.relative_gap,
            "不可行模型不能输出默认零目标或伪造间隙");
    require(infeasible.verification_level == "backend_claim_only",
            "不可行结论当前只来自后端");
    const auto unbounded = solve("unbounded.lp");
    require(unbounded.status == "unbounded", "无界模型应识别无界");
    require(!unbounded.absolute_gap && !unbounded.relative_gap, "无界模型不应给出有限间隙");

    const auto empty = solve("empty.mps");
    require(empty.status == "empty" && empty.has_solution && empty.objective == 0.0,
            "空模型应验证零维候选：" + empty.status + "；" + empty.message);
    require(empty.solution.empty() && empty.verification_level == "primal_dual_residuals_checked_backend_claim",
            "零维模型必须单独注明验证来源");

    for (const char* name : {"quadratic.lp", "semi_continuous.mps",
                             "semi_integer.mps", "read_warning.lp",
                             "small_matrix_value.lp"}) {
      const auto unsupported = solve(name);
      require(unsupported.status == "error" && !unsupported.has_solution &&
                  !unsupported.objective, std::string("必须拒绝不支持或有读入警告的模型：") + name);
    }
    for (const char* name : {"huge_column_upper.lp", "huge_column_lower.lp",
                             "huge_row_upper.lp", "huge_row_lower.lp",
                             "huge_bounds.mps", "huge_positive_cost.lp",
                             "huge_negative_cost.lp"}) {
      const auto changed_model = solve(name);
      require(changed_model.status == "error" && !changed_model.has_solution &&
                  !changed_model.objective && !changed_model.best_bound &&
                  changed_model.message.find("转换为无穷") != std::string::npos,
              std::string("不能求解被读取器改写的有限数据：") + name +
                  "；" + changed_model.status + "；" + changed_model.message);
      require(changed_model.model.costs.empty(), "被改写的数据不能进入原模型表示");
    }
    solver::SolveOptions zero_budget;
    zero_budget.time_limit = 0;
    const auto limit = solve("milp_min.lp", zero_budget);
    require(limit.status == "time_limit" && !limit.has_solution && !limit.objective &&
                !limit.relative_gap && !limit.absolute_gap && !limit.best_bound,
            "零预算必须保留超时状态且不能输出虚构候选或界");
    zero_budget.highs.solver = "ipm";
    zero_budget.highs.run_crossover = "off";
    const auto ipm_limit = solve("lp_min.lp", zero_budget);
    require(ipm_limit.status == "time_limit" && !ipm_limit.has_solution &&
                !ipm_limit.objective && !ipm_limit.best_bound && ipm_limit.solve_seconds == 0.0,
            "固定内点算法且关闭 crossover 时仍必须遵守零预算");

    solver::SolveOptions two_threads;
    two_threads.threads = 2;
    require_optimum(solve("lp_min.lp", two_threads), 4);
    require_optimum(solve("lp_min.lp"), 4);
    solver::SolveOptions invalid;
    invalid.time_limit = std::numeric_limits<double>::quiet_NaN();
    require(solve("lp_min.lp", invalid).status == "error", "必须拒绝非有限时间预算");
    require(solve("missing.lp").status == "error", "不存在的模型应返回明确错误");

    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto unicode_path = std::filesystem::temp_directory_path() /
        (std::filesystem::path("solver-" + unique + "-") +=
         std::filesystem::path(u8"中文模型🧮.lp"));
    std::filesystem::copy_file(fixtures / "lp_min.lp", unicode_path);
    const auto unicode_result = solver::solve_file(unicode_path);
    std::filesystem::remove(unicode_path);
    if (unicode_result.status == "optimal") {
      require_optimum(unicode_result, 4);
    } else {
#ifdef _WIN32
      require(unicode_result.status == "error" &&
                  unicode_result.message.find("ASCII") != std::string::npos,
              "不能表示的 Unicode 路径应明确拒绝，不能读错文件");
#else
      require(false, "当前平台应支持 UTF-8 模型路径");
#endif
    }
    std::cout << "后端集成检查通过\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "后端集成检查失败：" << error.what() << '\n';
    return 1;
  }
}
