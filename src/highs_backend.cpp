#include "solver/backend.hpp"
#include "solver/pdlp.hpp"
#include "solver/ipm.hpp"

#include "Highs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace solver {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ReadDiagnostics {
  bool warning = false;
  bool fixed_format_switch = false;
  bool other_warning = false;
  bool error = false;
  bool finite_data_replaced_by_infinity = false;
};

void collect_diagnostics(int type, const std::string& message,
                         const HighsCallbackOutput* output,
                         HighsCallbackInput*, void* context) {
  if (type != kCallbackLogging || output == nullptr) return;
  auto& diagnostics = *static_cast<ReadDiagnostics*>(context);
  diagnostics.warning |= output->log_type == HighsLogType::kWarning;
  if (output->log_type == HighsLogType::kWarning) {
    // 仅识别固定版本 1.15.1 的完整格式切换通知，其余警告保持拒绝。
    const bool format_switch = message ==
        "WARNING: Free format reader has detected row/col names with "
        "spaces: switching to fixed format parser\n";
    diagnostics.fixed_format_switch |= format_switch;
    diagnostics.other_warning |= !format_switch;
  }
  diagnostics.error |= output->log_type == HighsLogType::kError;
  // 固定版本 1.15.1 的 assessBounds/assessCosts 对实际数据转换仅记录 kInfo。
  // 保留后端数值范围，拒绝这些转换；普通信息日志不会使合法模型被拒绝。
  if (output->log_type == HighsLogType::kInfo &&
      message.find(" are treated as ") != std::string::npos &&
      message.find("Infinity") != std::string::npos &&
      (message.find("lower bounds") != std::string::npos ||
       message.find("upper bounds") != std::string::npos ||
       message.find("|cost|") != std::string::npos)) {
    diagnostics.finite_data_replaced_by_infinity = true;
  }
}

template <typename T>
void set_option(Highs& highs, const char* name, T value) {
  if (highs.setOptionValue(name, value) != HighsStatus::kOk) {
    throw std::invalid_argument(std::string("无法设置后端参数：") + name);
  }
}

std::unique_ptr<Highs> make_reader(const SolveOptions& options,
                                  ReadDiagnostics& diagnostics, bool fixed_format) {
  auto highs = std::make_unique<Highs>();
  // 首读和重读共用初始化，数值阈值等未覆盖选项均保留相同上游默认值。
  set_option(*highs, "log_to_console", false);
  set_option(*highs, "output_flag", true);
  if (highs->setCallback(HighsCallbackFunctionType(collect_diagnostics),
                         &diagnostics) != HighsStatus::kOk ||
      highs->startCallback(kCallbackLogging) != HighsStatus::kOk) {
    throw std::runtime_error("无法建立模型读取诊断回调");
  }
  set_option(*highs, "threads", options.threads);
  set_option(*highs, "random_seed", options.seed);
  set_option(*highs, "time_limit", options.time_limit);
  set_option(*highs, "mip_rel_gap", options.mip_relative_gap);
  set_option(*highs, "allow_unbounded_or_infeasible", false);
  set_option(*highs, "mps_parser_type_free", !fixed_format);
  return highs;
}

void check_options(const SolveOptions& options) {
  if (options.backend != "highs" && options.backend != "pdlp" && options.backend != "ipm") {
    throw std::invalid_argument("求解后端必须是 highs、pdlp 或 ipm");
  }
  if (options.highs.solver != "choose" && options.highs.solver != "simplex" &&
      options.highs.solver != "ipm") {
    throw std::invalid_argument("HiGHS 算法必须是 choose、simplex 或 ipm");
  }
  if (options.highs.run_crossover != "on" && options.highs.run_crossover != "off") {
    throw std::invalid_argument("HiGHS crossover 必须是 on 或 off");
  }
  if (options.highs.presolve != "choose" && options.highs.presolve != "on" &&
      options.highs.presolve != "off") {
    throw std::invalid_argument("HiGHS presolve 必须是 choose、on 或 off");
  }
  if (options.backend != "highs" &&
      (options.highs.solver != "choose" || options.highs.run_crossover != "on" ||
       options.highs.presolve != "choose")) {
    throw std::invalid_argument("非默认 HiGHS 算法、crossover 与 presolve 选项仅适用于 highs 后端");
  }
  if (!std::isfinite(options.time_limit) || options.time_limit < 0 ||
      options.threads < 1 || options.seed < 0 ||
      !std::isfinite(options.mip_relative_gap) || options.mip_relative_gap < 0) {
    throw std::invalid_argument("时间限制、线程数、随机种子或相对间隙参数无效");
  }
  const auto& t = options.tolerances;
  const double values[] = {t.feasibility_absolute, t.feasibility_relative,
                           t.integrality, t.objective_absolute,
                           t.objective_relative};
  for (double value : values) {
    if (!std::isfinite(value) || value < 0) {
      throw std::invalid_argument("独立验证容差必须是有限的非负数");
    }
  }
}

std::string backend_path(const std::filesystem::path& path) {
#ifdef _WIN32
  // 当前 HiGHS 读取器使用窄字符文件 API，拒绝不能无损表示的路径。
  const auto& native = path.native();
  if (native.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("输入路径过长");
  }
  const UINT code_page = GetACP();
  const DWORD flags = code_page == CP_UTF8 ? WC_ERR_INVALID_CHARS : WC_NO_BEST_FIT_CHARS;
  BOOL default_used = FALSE;
  BOOL* default_pointer = code_page == CP_UTF8 ? nullptr : &default_used;
  const int size = WideCharToMultiByte(code_page, flags, native.data(),
                                      static_cast<int>(native.size()), nullptr,
                                      0, nullptr, default_pointer);
  if (size == 0 || default_used) {
    throw std::invalid_argument("输入路径无法由 HiGHS 的窄字符读取器无损表示，请使用 ASCII 路径");
  }
  std::string converted(static_cast<std::size_t>(size), '\0');
  if (WideCharToMultiByte(code_page, flags, native.data(),
                          static_cast<int>(native.size()), converted.data(), size,
                          nullptr, default_pointer) != size || default_used) {
    throw std::invalid_argument("输入路径字符转换失败，请使用 ASCII 路径");
  }
  return converted;
#else
  return path.string();
#endif
}

Model convert_model(const Highs& highs) {
  if (highs.getModel().hessian_.dim_ != 0 ||
      highs.getNumLinearObjectives() != 0) {
    throw std::invalid_argument("首版仅支持单目标 LP/MILP，不支持二次或多目标模型");
  }
  HighsLp lp = highs.getLp();
  lp.ensureColwise();
  Model model;
  model.name = lp.model_name_;
  model.sense = lp.sense_ == ObjSense::kMinimize ? ObjectiveSense::minimize
                                              : ObjectiveSense::maximize;
  model.objective_offset = lp.offset_;
  model.costs = lp.col_cost_;
  model.column_lower = lp.col_lower_;
  model.column_upper = lp.col_upper_;
  model.row_lower = lp.row_lower_;
  model.row_upper = lp.row_upper_;
  model.kinds.assign(model.costs.size(), VariableKind::continuous);
  if (!lp.integrality_.empty()) {
    if (lp.integrality_.size() != model.costs.size()) {
      throw std::runtime_error("后端变量类型数组尺寸无效");
    }
    for (std::size_t i = 0; i < lp.integrality_.size(); ++i) {
      switch (lp.integrality_[i]) {
        case HighsVarType::kContinuous:
          break;
        case HighsVarType::kInteger:
          model.kinds[i] = VariableKind::integer;
          break;
        default:
          throw std::invalid_argument("首版不支持半连续、半整数或隐式整数变量");
      }
    }
  }
  model.column_start.assign(lp.a_matrix_.start_.begin(),
                            lp.a_matrix_.start_.end());
  if (model.costs.empty() && model.column_start.empty()) {
    model.column_start.push_back(0);
  }
  model.row_index.assign(lp.a_matrix_.index_.begin(), lp.a_matrix_.index_.end());
  model.values = lp.a_matrix_.value_;
  const auto errors = validate_model(model);
  if (!errors.empty()) {
    throw std::invalid_argument("模型未通过自主表示检查：" + errors.front());
  }
  return model;
}

std::string map_status(HighsModelStatus status) {
  switch (status) {
    case HighsModelStatus::kOptimal: return "optimal";
    case HighsModelStatus::kInfeasible: return "infeasible";
    case HighsModelStatus::kUnbounded: return "unbounded";
    case HighsModelStatus::kUnboundedOrInfeasible: return "infeasible_or_unbounded";
    case HighsModelStatus::kModelEmpty: return "empty";
    case HighsModelStatus::kTimeLimit: return "time_limit";
    case HighsModelStatus::kIterationLimit: return "iteration_limit";
    case HighsModelStatus::kSolutionLimit: return "solution_limit";
    case HighsModelStatus::kObjectiveBound: return "objective_bound";
    case HighsModelStatus::kObjectiveTarget: return "objective_target";
    case HighsModelStatus::kInterrupt:
    case HighsModelStatus::kHighsInterrupt: return "interrupted";
    case HighsModelStatus::kMemoryLimit: return "memory_limit";
    case HighsModelStatus::kUnknown:
    case HighsModelStatus::kNotset: return "unknown";
    default: return "error";
  }
}

void verify_candidate(SolveResult& result, const std::vector<double>& values,
                      std::optional<double> objective,
                      const Tolerances& tolerances) {
  const auto start = Clock::now();
  result.verification = verify_solution(result.model, values, objective, tolerances);
  result.verification_seconds = elapsed(start);
  if (result.verification.valid_input && result.verification.feasible &&
      result.verification.objective_consistent && result.verification.objective) {
    result.has_solution = true;
    result.solution = values;
    result.objective = result.verification.objective;
    result.verification_level = "primal_checked_backend_claim";
  } else {
    result.status = "verification_failed";
    result.verification_level = "primal_check_failed";
    result.message = "后端候选未通过原模型的独立可行性或目标检查";
  }
}

void compute_gap(SolveResult& result, const Tolerances& tolerances) {
  if (!result.has_solution || !result.objective || !result.best_bound) return;
  const long double objective = *result.objective;
  const long double bound = *result.best_bound;
  const long double scale = std::max({1.0L, std::abs(objective), std::abs(bound)});
  const long double signed_gap = result.model.sense == ObjectiveSense::minimize
                                     ? objective - bound : bound - objective;
  const long double tolerance = tolerances.objective_absolute +
      static_cast<long double>(tolerances.objective_relative) * scale;
  if (signed_gap < -tolerance) {
    result.status = "verification_failed";
    result.message = "后端最优界与已验证候选的目标方向矛盾";
    result.verification_level = "primal_checked_inconsistent_backend_bound";
    result.has_solution = false;
    result.solution.clear();
    result.objective.reset();
    result.best_bound.reset();
    return;
  }
  // 仅容忍目标检查容差内的微小负间隙，不能取绝对值掩盖方向错误。
  const long double gap = std::max(0.0L, signed_gap);
  const double absolute = static_cast<double>(gap);
  if (std::isfinite(absolute)) result.absolute_gap = absolute;
  const double relative = static_cast<double>(gap / scale);
  if (std::isfinite(relative)) result.relative_gap = relative;
}

}  // 匿名命名空间

std::string backend_version() { return highsVersion(); }

SolveResult solve_file(const std::filesystem::path& path,
                       const SolveOptions& options) {
  const auto total_start = Clock::now();
  SolveResult result;
  result.backend = options.backend;
  result.backend_version = options.backend == "pdlp" ? pdlp_version() :
      (options.backend == "ipm" ? ipm_version() : backend_version());
  try {
    check_options(options);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension != ".mps" && extension != ".lp") {
      throw std::invalid_argument("首版仅接受未压缩的 .mps 或 .lp 文件");
    }
    if (!std::filesystem::is_regular_file(path)) {
      throw std::invalid_argument("输入模型不存在或不是普通文件");
    }

    // HiGHS 调度器为进程全局状态；首版在本接口内串行使用，允许逐次改变线程数。
    static std::mutex backend_mutex;
    std::unique_lock<std::mutex> lock(backend_mutex);
    Highs::resetGlobalScheduler(true);
    ReadDiagnostics diagnostics;
    auto reader = make_reader(options, diagnostics, false);

    const auto read_start = Clock::now();
    auto read_status = HighsStatus::kError;
    try {
      const auto filename = backend_path(path);
      read_status = reader->readModel(filename);
      if (extension == ".mps" && read_status != HighsStatus::kError &&
          diagnostics.fixed_format_switch && !diagnostics.other_warning &&
          !diagnostics.error && !diagnostics.finite_data_replaced_by_infinity) {
        // 仅格式切换可重试；新实例不继承首读的模型或求解状态。
        // 先销毁旧回调持有者，再清空诊断，保证回调上下文一直有效。
        reader.reset();
        diagnostics = {};
        reader = make_reader(options, diagnostics, true);
        read_status = reader->readModel(filename);
      }
    } catch (...) {
      result.read_seconds = elapsed(read_start);
      throw;
    }
    result.read_seconds = elapsed(read_start);
    if (read_status == HighsStatus::kError || diagnostics.error) {
      throw std::invalid_argument("HiGHS 无法读取模型；请检查格式和模型语义");
    }
    if (diagnostics.finite_data_replaced_by_infinity) {
      throw std::invalid_argument("读取器将有限界或目标系数转换为无穷，首版拒绝该模型；请调整模型尺度");
    }
    // 某些读取器只写 warning 而仍返回 kOk；不能仅检查返回状态。
    if (read_status != HighsStatus::kOk || diagnostics.warning) {
      throw std::invalid_argument("模型读取产生警告，已拒绝求解以避免静默修改或丢失模型语义");
    }
    Highs& highs = *reader;
    result.model = convert_model(highs);
    set_option(highs, "output_flag", false);
    if (options.backend == "pdlp" || options.backend == "ipm") {
      const double read_seconds = result.read_seconds;
      // 自主路径只复用文件解析；迭代内核不执行 HiGHS 的求解或预处理。
      lock.unlock();
      result = options.backend == "pdlp" ? solve_pdlp(result.model, options) :
          solve_ipm(result.model, options);
      result.read_seconds = read_seconds;
      result.total_seconds = elapsed(total_start);
      return result;
    }
    if (is_mip(result.model) &&
        (options.highs.solver != "choose" || options.highs.run_crossover != "on" ||
         options.highs.presolve != "choose")) {
      throw std::invalid_argument("非默认 HiGHS 算法、crossover 与 presolve 选项仅适用于连续 LP");
    }
    set_option(highs, "solver", options.highs.solver);
    set_option(highs, "run_crossover", options.highs.run_crossover);
    set_option(highs, "presolve", options.highs.presolve);
    if (options.time_limit == 0) {
      result.status = "time_limit";
      result.backend_status = "未启动求解";
      result.message = "求解时间预算为零，模型已读取但未启动求解";
      result.total_seconds = elapsed(total_start);
      return result;
    }
    const auto solve_start = Clock::now();
    const auto run_status = highs.run();
    result.solve_seconds = elapsed(solve_start);
    const auto status = highs.getModelStatus();
    result.backend_status = highs.modelStatusToString(status);
    result.status = map_status(status);
    result.verification_level = "backend_claim_only";
    const auto& info = highs.getInfo();
    const auto& solution = highs.getSolution();
    if (run_status == HighsStatus::kError) {
      result.status = "error";
      result.message = "HiGHS 求解失败";
    } else if (status == HighsModelStatus::kModelEmpty) {
      // 空向量是合法候选，仍需检查所有零变量约束，不能直接宣称可行。
      verify_candidate(result, {}, result.model.objective_offset,
                        options.tolerances);
      if (result.has_solution) {
        result.verification_level = "empty_model_checked";
        result.best_bound = result.objective;
      }
    } else {
      const bool claims_candidate = solution.value_valid && info.valid &&
          info.primal_solution_status == kSolutionStatusFeasible;
      if (claims_candidate) {
        verify_candidate(result, solution.col_value, info.objective_function_value,
                          options.tolerances);
      } else if (status == HighsModelStatus::kOptimal) {
        result.status = "verification_failed";
        result.verification_level = "primal_check_failed";
        result.message = "后端声称最优，但未提供可独立复核的可行候选";
      }
      if (info.valid && result.status != "verification_failed") {
        if (info.simplex_iteration_count >= 0) {
          result.simplex_iterations = info.simplex_iteration_count;
        }
        if (is_mip(result.model)) {
          if (info.mip_node_count >= 0) result.nodes = info.mip_node_count;
          // HiGHS 已把 MIP 界恢复到原目标方向，并包含目标常数；不可再次翻转。
          if (std::isfinite(info.mip_dual_bound) &&
              status != HighsModelStatus::kInfeasible &&
              status != HighsModelStatus::kUnbounded &&
              status != HighsModelStatus::kUnboundedOrInfeasible) {
            result.best_bound = info.mip_dual_bound;
          }
        } else if (result.has_solution && status == HighsModelStatus::kOptimal) {
          // LP 界仍复述后端结论；下方残差检查不提供精确算术证书。
          result.best_bound = info.objective_function_value;
        }
      }
    }
    if (!is_mip(result.model) && result.has_solution) {
      const auto check_start = Clock::now();
      const double sense = result.model.sense == ObjectiveSense::minimize ? 1.0 : -1.0;
      if (solution.dual_valid && solution.row_dual.size() == result.model.row_lower.size()) {
        result.row_dual = solution.row_dual;
        for (double& value : result.row_dual) value *= sense;
        if (!project_lp_row_dual_domain(result.model, result.row_dual)) {
          throw std::runtime_error("原坐标行对偶维度或行界无效");
        }
        result.lp_optimality = verify_lp_optimality(result.model, result.solution,
            result.row_dual, options.tolerances, options.pdlp.optimality);
      } else if (status == HighsModelStatus::kModelEmpty) {
        result.row_dual.assign(result.model.row_lower.size(), 0.0);
        result.lp_optimality = verify_lp_optimality(result.model, result.solution,
            result.row_dual, options.tolerances, options.pdlp.optimality);
      }
      result.verification_seconds += elapsed(check_start);
      if (result.lp_optimality && result.lp_optimality->numerically_optimal) {
        result.verification_level = "primal_dual_residuals_checked_backend_claim";
      } else if (status == HighsModelStatus::kOptimal || status == HighsModelStatus::kModelEmpty) {
        result.status = "verification_failed";
        result.verification_level = "primal_checked_dual_check_failed";
        result.message = "HiGHS 声称最优，但未通过与自主路径相同的原始/对偶数值检查";
        result.best_bound.reset();
      }
    }
    compute_gap(result, options.tolerances);
  } catch (const std::exception& exception) {
    result.status = "error";
    result.message = exception.what();
    result.verification_level = "none";
    result.has_solution = false;
    result.solution.clear();
    result.objective.reset();
    result.best_bound.reset();
    result.absolute_gap.reset();
    result.relative_gap.reset();
  }
  result.total_seconds = elapsed(total_start);
  return result;
}

}  // 命名空间 solver
