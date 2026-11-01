#include "Highs.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double external_seconds = 5;
constexpr std::uint64_t budget_check_work_interval = 256;

struct DeadlineExpired {};
struct InputError : std::runtime_error { using std::runtime_error::runtime_error; };
struct ApiError : std::runtime_error { using std::runtime_error::runtime_error; };
struct ReadError : std::runtime_error { using std::runtime_error::runtime_error; };

// 原生库调用和容器整批复制只能在边界检查；自有逐项循环每256单位检查。
class Deadline {
 public:
  explicit Deadline(std::function<double()> elapsed, double limit = external_seconds)
      : elapsed_(std::move(elapsed)), limit_(limit) {}
  double elapsed() const { return elapsed_(); }
  void check() {
    ++checks;
    if (elapsed() >= limit_) throw DeadlineExpired{};
  }
  void tick() {
    ++loop_units;
    if (loop_units % budget_check_work_interval == 0) check();
  }
  double remaining() {
    check();
    const double left = limit_ - elapsed();
    if (!(left > 0)) throw DeadlineExpired{};
    return left;
  }
  std::uint64_t checks = 0, loop_units = 0;
 private:
  std::function<double()> elapsed_;
  double limit_;
};

struct Range {
  std::uint64_t finite_count = 0, zero_count = 0, nonfinite_count = 0;
  std::optional<double> min_abs, max_abs;
  void add(double value, bool include_zero) {
    if (!std::isfinite(value)) { ++nonfinite_count; return; }
    ++finite_count;
    if (value == 0) ++zero_count;
    if (value == 0 && !include_zero) return;
    const double magnitude = std::abs(value);
    min_abs = min_abs ? std::min(*min_abs, magnitude) : magnitude;
    max_abs = max_abs ? std::max(*max_abs, magnitude) : magnitude;
  }
};

struct Summary {
  HighsInt n = 0, m = 0;
  std::uint64_t nnz = 0;
  Range matrix, cost, bounds;
};

// 矩阵/目标范围排除零，界范围包含有限零；非有限界仅允许正确方向的无穷。
Summary summarize(const HighsLp& lp, Deadline& budget) {
  budget.check();
  if (lp.num_col_ < 0 || lp.num_row_ < 0 || !lp.a_matrix_.isColwise())
    throw InputError("LP维数或矩阵方向无效");
  const auto n = static_cast<std::size_t>(lp.num_col_), m = static_cast<std::size_t>(lp.num_row_);
  if (lp.col_cost_.size() != n || lp.col_lower_.size() != n || lp.col_upper_.size() != n ||
      lp.row_lower_.size() != m || lp.row_upper_.size() != m || lp.a_matrix_.start_.size() != n + 1 ||
      lp.a_matrix_.index_.size() != lp.a_matrix_.value_.size() || lp.a_matrix_.start_.front() != 0 ||
      lp.a_matrix_.start_.back() < 0 || static_cast<std::uint64_t>(lp.a_matrix_.start_.back()) != lp.a_matrix_.value_.size())
    throw InputError("LP数组形状无效");
  if (!std::isfinite(lp.offset_)) throw InputError("目标常数不是有限数");
  if (!lp.integrality_.empty() && lp.integrality_.size() != n) throw InputError("变量类型数组形状无效");
  for (const auto kind : lp.integrality_) {
    budget.tick();
    if (kind != HighsVarType::kContinuous) throw InputError("结构探针仅接受连续LP");
  }
  Summary answer;
  answer.n = lp.num_col_;
  answer.m = lp.num_row_;
  answer.nnz = lp.a_matrix_.value_.size();
  for (std::size_t col = 0; col < n; ++col) {
    budget.tick();
    const auto first = lp.a_matrix_.start_[col], last = lp.a_matrix_.start_[col + 1];
    if (first < 0 || first > last || static_cast<std::uint64_t>(last) > answer.nnz)
      throw InputError("CSC列边界无效");
    for (HighsInt k = first; k < last; ++k) {
      budget.tick();
      const auto entry = static_cast<std::size_t>(k);
      if (lp.a_matrix_.index_[entry] < 0 || lp.a_matrix_.index_[entry] >= lp.num_row_ ||
          !std::isfinite(lp.a_matrix_.value_[entry])) throw InputError("CSC非零项无效");
      answer.matrix.add(lp.a_matrix_.value_[entry], false);
    }
    if (!std::isfinite(lp.col_cost_[col])) throw InputError("目标系数不是有限数");
    answer.cost.add(lp.col_cost_[col], false);
  }
  const auto append_bounds = [&](const std::vector<double>& lower, const std::vector<double>& upper) {
    for (std::size_t k = 0; k < lower.size(); ++k) {
      budget.tick();
      const double lo = lower[k], hi = upper[k];
      if (std::isnan(lo) || std::isnan(hi) || lo == kHighsInf || hi == -kHighsInf || lo > hi)
        throw InputError("LP行界或列界无效");
      answer.bounds.add(lo, true);
      answer.bounds.add(hi, true);
    }
  };
  append_bounds(lp.col_lower_, lp.col_upper_);
  append_bounds(lp.row_lower_, lp.row_upper_);
  budget.check();
  return answer;
}

void normalize_minimum(HighsLp& lp, Deadline& budget) {
  budget.check();
  if (lp.sense_ != ObjSense::kMinimize && lp.sense_ != ObjSense::kMaximize)
    throw InputError("目标方向无效");
  const double sign = lp.sense_ == ObjSense::kMinimize ? 1 : -1;
  for (double& value : lp.col_cost_) {
    budget.tick();
    if (!std::isfinite(value)) throw InputError("目标系数不是有限数");
    value *= sign;
  }
  if (!std::isfinite(lp.offset_)) throw InputError("目标常数不是有限数");
  lp.offset_ *= sign;
  lp.sense_ = ObjSense::kMinimize;
  budget.check();
}

struct ReadDiagnostics {
  bool warning = false, fixed_format_switch = false, other_warning = false, error = false;
  bool finite_data_replaced_by_infinity = false;
};

void collect_diagnostics(int type, const std::string& message, const HighsCallbackOutput* output,
                         HighsCallbackInput*, void* context) {
  if (type != kCallbackLogging || !output) return;
  auto& diagnostic = *static_cast<ReadDiagnostics*>(context);
  diagnostic.warning |= output->log_type == HighsLogType::kWarning;
  if (output->log_type == HighsLogType::kWarning) {
    // 与冻结读取器相同，只允许明确的free→fixed格式通知触发重读。
    const bool format_switch = message == "WARNING: Free format reader has detected row/col names with "
        "spaces: switching to fixed format parser\n";
    diagnostic.fixed_format_switch |= format_switch;
    diagnostic.other_warning |= !format_switch;
  }
  diagnostic.error |= output->log_type == HighsLogType::kError;
  if (output->log_type == HighsLogType::kInfo && message.find(" are treated as ") != std::string::npos &&
      message.find("Infinity") != std::string::npos &&
      (message.find("lower bounds") != std::string::npos || message.find("upper bounds") != std::string::npos ||
       message.find("|cost|") != std::string::npos)) diagnostic.finite_data_replaced_by_infinity = true;
}

struct Calls { std::uint64_t started = 0, completed = 0; };
struct NativeOptions {
  double primal_feasibility_tolerance = 0, dual_feasibility_tolerance = 0, ipm_optimality_tolerance = 0;
  bool lp_presolve_requires_basis_postsolve = true;
};
struct Native {
  std::optional<int> read_status, pass_status, presolve_call_status, presolve_status_code, model_status_code;
  std::optional<std::string> presolve_status, model_status;
  bool info_valid = false;
  HighsInt simplex_iterations = -1, ipm_iterations = -1, pdlp_iterations = -1, crossover_iterations = -1;
  std::optional<double> native_time_limit;
};
struct Timings {
  double configuration = 0, read = 0, copy = 0, original_statistics = 0, normalization = 0;
  double pass_model = 0, presolve = 0, reduced_statistics = 0, total = 0;
};
struct Report {
  const char* status = "exception";
  const char* phase = "arguments";
  bool scan_complete = false, deadline_exceeded = false, read_fixed_retry = false;
  std::optional<std::string> input_sense;
  std::optional<int> normalization_sign;
  std::optional<Summary> original, reduced;
  std::optional<NativeOptions> native_options;
  Calls read_calls, pass_calls, presolve_calls;
  ReadDiagnostics read_diagnostics;
  Native native;
  Timings timings;
  std::string error;
};

struct Phase {
  Phase(Report& report, const char* name, double& total, Deadline& budget)
      : total_(total), budget_(budget), begin_(budget.elapsed()) { report.phase = name; }
  ~Phase() { total_ += std::max(0.0, budget_.elapsed() - begin_); }
 private:
  double& total_;
  Deadline& budget_;
  double begin_;
};

template<class T> void set_option(Highs& highs, const char* name, T value, Deadline& budget) {
  budget.check();
  if (highs.setOptionValue(name, value) != HighsStatus::kOk) throw ApiError(std::string("无法设置参数：") + name);
  budget.check();
}
template<class T> void get_option(const Highs& highs, const char* name, T& value, Deadline& budget) {
  budget.check();
  if (highs.getOptionValue(name, value) != HighsStatus::kOk) throw ApiError(std::string("无法读取参数：") + name);
  budget.check();
}

std::unique_ptr<Highs> make_reader(ReadDiagnostics& diagnostics, bool fixed, Deadline& budget) {
  budget.check();
  auto reader = std::make_unique<Highs>();
  budget.check();
  set_option(*reader, "log_to_console", false, budget);
  set_option(*reader, "output_flag", true, budget);
  budget.check();
  if (reader->setCallback(HighsCallbackFunctionType(collect_diagnostics), &diagnostics) != HighsStatus::kOk ||
      reader->startCallback(kCallbackLogging) != HighsStatus::kOk) throw ApiError("无法建立读取诊断回调");
  budget.check();
  set_option(*reader, "threads", HighsInt{1}, budget);
  set_option(*reader, "random_seed", HighsInt{0}, budget);
  set_option(*reader, "allow_unbounded_or_infeasible", false, budget);
  set_option(*reader, "presolve", std::string("choose"), budget);
  set_option(*reader, "lp_presolve_requires_basis_postsolve", true, budget);
  set_option(*reader, "mps_parser_type_free", !fixed, budget);
  return reader;
}

void capture_native(const Highs& highs, Report& report) {
  const auto& info = highs.getInfo();
  report.native.info_valid = info.valid;
  report.native.simplex_iterations = info.simplex_iteration_count;
  report.native.ipm_iterations = info.ipm_iteration_count;
  report.native.pdlp_iterations = info.pdlp_iteration_count;
  report.native.crossover_iterations = info.crossover_iteration_count;
  const auto model = highs.getModelStatus();
  report.native.model_status_code = static_cast<int>(model);
  report.native.model_status = highs.modelStatusToString(model);
  const auto presolve = highs.getModelPresolveStatus();
  report.native.presolve_status_code = static_cast<int>(presolve);
  report.native.presolve_status = highs.presolveStatusToString(presolve);
}

void scan(const std::string& filename, Report& report, Deadline& budget) {
  ReadDiagnostics diagnostics;
  std::unique_ptr<Highs> reader;
  try {
    {
      Phase phase(report, "configuration", report.timings.configuration, budget);
      reader = make_reader(diagnostics, false, budget);
    }
    {
      Phase phase(report, "read", report.timings.read, budget);
      budget.check();
      ++report.read_calls.started;
      const auto status = reader->readModel(filename);
      ++report.read_calls.completed;
      report.native.read_status = static_cast<int>(status);
      report.read_diagnostics = diagnostics;
      budget.check();
      auto extension = std::filesystem::path(filename).extension().string();
      for (char& value : extension) { budget.tick(); if (value >= 'A' && value <= 'Z') value += 'a' - 'A'; }
      if (extension == ".mps" && status != HighsStatus::kError && diagnostics.fixed_format_switch &&
          !diagnostics.other_warning && !diagnostics.error && !diagnostics.finite_data_replaced_by_infinity) {
        report.read_fixed_retry = true;
        budget.check();
        reader.reset();
        diagnostics = {};
        reader = make_reader(diagnostics, true, budget);
        budget.check();
        ++report.read_calls.started;
        report.native.read_status = static_cast<int>(reader->readModel(filename));
        ++report.read_calls.completed;
        report.read_diagnostics = diagnostics;
        budget.check();
      }
      if (*report.native.read_status != static_cast<int>(HighsStatus::kOk) || diagnostics.warning ||
          diagnostics.error || diagnostics.finite_data_replaced_by_infinity) throw ReadError("模型读取产生错误、警告或有限数据转无穷");
      if (reader->getModel().hessian_.dim_ != 0 || reader->getNumLinearObjectives() != 0)
        throw InputError("结构探针不接受二次或多目标模型");
    }
    {
      Phase phase(report, "original_statistics", report.timings.original_statistics, budget);
      report.original = summarize(reader->getLp(), budget);
      const auto sense = reader->getLp().sense_;
      if (sense != ObjSense::kMinimize && sense != ObjSense::kMaximize) throw InputError("目标方向无效");
      report.input_sense = sense == ObjSense::kMinimize ? "minimize" : "maximize";
      report.normalization_sign = sense == ObjSense::kMinimize ? 1 : -1;
    }
    HighsLp normalized;
    {
      Phase phase(report, "copy", report.timings.copy, budget);
      budget.check();
      normalized = reader->getLp();
      budget.check();
    }
    {
      Phase phase(report, "normalization", report.timings.normalization, budget);
      normalize_minimum(normalized, budget);
    }
    {
      Phase phase(report, "configuration", report.timings.configuration, budget);
      // 后续库日志完全关闭；原生数值选项保持固定版本默认，只查询并报告。
      set_option(*reader, "output_flag", false, budget);
      NativeOptions options;
      get_option(*reader, "primal_feasibility_tolerance", options.primal_feasibility_tolerance, budget);
      get_option(*reader, "dual_feasibility_tolerance", options.dual_feasibility_tolerance, budget);
      get_option(*reader, "ipm_optimality_tolerance", options.ipm_optimality_tolerance, budget);
      get_option(*reader, "lp_presolve_requires_basis_postsolve", options.lp_presolve_requires_basis_postsolve, budget);
      report.native_options = options;
    }
    {
      Phase phase(report, "pass_model", report.timings.pass_model, budget);
      budget.check();
      ++report.pass_calls.started;
      const auto status = reader->passModel(std::move(normalized));
      ++report.pass_calls.completed;
      report.native.pass_status = static_cast<int>(status);
      budget.check();
      if (status != HighsStatus::kOk) throw ApiError("归一化LP传入原生预处理器失败");
    }
    {
      Phase phase(report, "presolve", report.timings.presolve, budget);
      budget.check();
      const double native_limit = reader->getRunTime() + budget.remaining();
      if (!std::isfinite(native_limit) || native_limit <= 0) throw ApiError("原生剩余预算无效");
      set_option(*reader, "time_limit", native_limit, budget);
      report.native.native_time_limit = native_limit;
      budget.check();
      ++report.presolve_calls.started;
      const auto status = reader->presolve();
      ++report.presolve_calls.completed;
      report.native.presolve_call_status = static_cast<int>(status);
      capture_native(*reader, report);
      budget.check();
      if (status == HighsStatus::kError) throw ApiError("原生预处理调用失败");
    }
    {
      Phase phase(report, "reduced_statistics", report.timings.reduced_statistics, budget);
      // 不可行、超时等状态可能仍带部分模型；原样记录，绝不把空输出当作已求解。
      const auto status = reader->getModelPresolveStatus();
      if (status == HighsPresolveStatus::kNotReduced || status == HighsPresolveStatus::kReduced ||
          status == HighsPresolveStatus::kReducedToEmpty || status == HighsPresolveStatus::kTimeout ||
          status == HighsPresolveStatus::kOutOfMemory) report.reduced = summarize(reader->getPresolvedLp(), budget);
      budget.check();
    }
    report.phase = "finished";
    budget.check();
    const auto final_status = reader->getModelPresolveStatus();
    const bool stopped = final_status == HighsPresolveStatus::kTimeout || final_status == HighsPresolveStatus::kOutOfMemory;
    report.status = stopped ? "presolve_stopped" : "complete";
    report.scan_complete = !stopped;
  } catch (...) {
    report.read_diagnostics = diagnostics;
    // 补充诊断不能用新的字符串分配失败覆盖原异常。
    if (reader) { try { capture_native(*reader, report); } catch (...) {} }
    throw;
  }
}

void quoted(std::FILE* stream, std::string_view value) {
  std::fputc('"', stream);
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') { std::fputc('\\', stream); std::fputc(ch, stream); }
    else if (ch < 0x20) std::fprintf(stream, "\\u%04x", static_cast<unsigned>(ch));
    else std::fputc(ch, stream);
  }
  std::fputc('"', stream);
}
template<class T> void number(std::FILE* stream, std::optional<T> value) {
  if (!value) std::fputs("null", stream);
  else if constexpr (std::is_floating_point_v<T>) {
    if (std::isfinite(*value)) std::fprintf(stream, "%.17g", static_cast<double>(*value));
    else std::fputs("null", stream);
  } else std::fprintf(stream, "%lld", static_cast<long long>(*value));
}
void string_value(std::FILE* stream, const std::optional<std::string>& value) {
  if (value) quoted(stream, *value); else std::fputs("null", stream);
}
void write_range(std::FILE* stream, const Range& range) {
  std::fprintf(stream, "{\"finite_count\":%llu,\"zero_count\":%llu,\"nonfinite_count\":%llu,\"min_abs\":",
      static_cast<unsigned long long>(range.finite_count), static_cast<unsigned long long>(range.zero_count),
      static_cast<unsigned long long>(range.nonfinite_count));
  number(stream, range.min_abs);
  std::fputs(",\"max_abs\":", stream); number(stream, range.max_abs); std::fputc('}', stream);
}
void write_summary(std::FILE* stream, const std::optional<Summary>& summary) {
  if (!summary) { std::fputs("null", stream); return; }
  const auto& s = *summary;
  std::fprintf(stream, "{\"n\":%lld,\"m\":%lld,\"nnz\":%llu,\"matrix\":", static_cast<long long>(s.n),
      static_cast<long long>(s.m), static_cast<unsigned long long>(s.nnz));
  write_range(stream, s.matrix); std::fputs(",\"cost\":", stream); write_range(stream, s.cost);
  std::fputs(",\"bounds\":", stream); write_range(stream, s.bounds); std::fputc('}', stream);
}
void write_report(std::FILE* stream, const Report& report, const Deadline& budget) {
  std::fputs("{\"schema\":1,\"mode\":\"structure_only\",\"status\":", stream); quoted(stream, report.status);
  std::fputs(",\"phase\":", stream); quoted(stream, report.phase);
  std::fprintf(stream, ",\"scan_complete\":%s,\"deadline_exceeded\":%s,\"explicit_lp_solve_calls\":0,\"lp_solution_claimed\":false,\"read_fixed_retry\":%s,\"input_sense\":",
      report.scan_complete ? "true" : "false", report.deadline_exceeded ? "true" : "false", report.read_fixed_retry ? "true" : "false");
  string_value(stream, report.input_sense); std::fputs(",\"normalization_sign\":", stream); number(stream, report.normalization_sign);
  std::fputs(",\"original\":", stream); write_summary(stream, report.original);
  std::fputs(",\"reduced\":", stream); write_summary(stream, report.reduced);
  std::fputs(",\"native_options\":", stream);
  if (report.native_options) {
    const auto& options = *report.native_options;
    std::fprintf(stream, "{\"primal_feasibility_tolerance\":%.17g,\"dual_feasibility_tolerance\":%.17g,\"ipm_optimality_tolerance\":%.17g,\"lp_presolve_requires_basis_postsolve\":%s}",
        options.primal_feasibility_tolerance, options.dual_feasibility_tolerance, options.ipm_optimality_tolerance,
        options.lp_presolve_requires_basis_postsolve ? "true" : "false");
  } else std::fputs("null", stream);
  const auto& d = report.read_diagnostics;
  std::fprintf(stream, ",\"read_diagnostics\":{\"warning\":%s,\"fixed_format_switch\":%s,\"other_warning\":%s,\"error\":%s,\"finite_data_replaced_by_infinity\":%s},\"calls\":{\"read\":{\"started\":%llu,\"completed\":%llu},\"pass_model\":{\"started\":%llu,\"completed\":%llu},\"presolve\":{\"started\":%llu,\"completed\":%llu}},\"native\":{\"read_status\":",
      d.warning ? "true" : "false", d.fixed_format_switch ? "true" : "false", d.other_warning ? "true" : "false",
      d.error ? "true" : "false", d.finite_data_replaced_by_infinity ? "true" : "false",
      static_cast<unsigned long long>(report.read_calls.started), static_cast<unsigned long long>(report.read_calls.completed),
      static_cast<unsigned long long>(report.pass_calls.started), static_cast<unsigned long long>(report.pass_calls.completed),
      static_cast<unsigned long long>(report.presolve_calls.started), static_cast<unsigned long long>(report.presolve_calls.completed));
  const auto& native = report.native;
  number(stream, native.read_status); std::fputs(",\"pass_status\":", stream); number(stream, native.pass_status);
  std::fputs(",\"presolve_call_status\":", stream); number(stream, native.presolve_call_status);
  std::fputs(",\"presolve_status_code\":", stream); number(stream, native.presolve_status_code);
  std::fputs(",\"presolve_status\":", stream); string_value(stream, native.presolve_status);
  std::fputs(",\"model_status_code\":", stream); number(stream, native.model_status_code);
  std::fputs(",\"model_status\":", stream); string_value(stream, native.model_status);
  std::fprintf(stream, ",\"info_valid\":%s,\"simplex_iterations\":%lld,\"ipm_iterations\":%lld,\"pdlp_iterations\":%lld,\"crossover_iterations\":%lld,\"native_time_limit\":",
      native.info_valid ? "true" : "false", static_cast<long long>(native.simplex_iterations), static_cast<long long>(native.ipm_iterations),
      static_cast<long long>(native.pdlp_iterations), static_cast<long long>(native.crossover_iterations));
  number(stream, native.native_time_limit);
  const auto& t = report.timings;
  std::fprintf(stream, "},\"timings_seconds\":{\"configuration\":%.17g,\"read\":%.17g,\"copy\":%.17g,\"original_statistics\":%.17g,\"normalization\":%.17g,\"pass_model\":%.17g,\"presolve\":%.17g,\"reduced_statistics\":%.17g,\"total\":%.17g},\"budget\":{\"external_seconds\":%.17g,\"budget_check_work_interval\":%llu,\"checks\":%llu,\"loop_units\":%llu},\"error\":",
      t.configuration, t.read, t.copy, t.original_statistics, t.normalization, t.pass_model, t.presolve, t.reduced_statistics, t.total,
      external_seconds, static_cast<unsigned long long>(budget_check_work_interval), static_cast<unsigned long long>(budget.checks),
      static_cast<unsigned long long>(budget.loop_units));
  quoted(stream, report.error); std::fputs("}\n", stream);
}

void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
HighsLp fixture() {
  HighsLp lp;
  lp.num_col_ = 2; lp.num_row_ = 2; lp.sense_ = ObjSense::kMinimize; lp.offset_ = 7;
  lp.col_cost_ = {0, -3}; lp.col_lower_ = {0, -kHighsInf}; lp.col_upper_ = {4, kHighsInf};
  lp.row_lower_ = {-kHighsInf, 2}; lp.row_upper_ = {5, 2};
  lp.a_matrix_.format_ = MatrixFormat::kColwise; lp.a_matrix_.num_col_ = 2; lp.a_matrix_.num_row_ = 2;
  lp.a_matrix_.start_ = {0, 2, 4}; lp.a_matrix_.index_ = {0, 1, 0, 1}; lp.a_matrix_.value_ = {0, -2, 4, 0.5};
  return lp;
}

int self_test() {
  std::vector<const char*> details;
  double time = 0;
  Deadline budget([&] { return time; });
  {
    const auto original = fixture();
    auto minimum = original;
    normalize_minimum(minimum, budget);
    require(minimum.col_cost_ == original.col_cost_ && minimum.offset_ == 7 && minimum.sense_ == ObjSense::kMinimize, "最小化归一化改变数据");
    auto maximum = original; maximum.sense_ = ObjSense::kMaximize;
    normalize_minimum(maximum, budget);
    require(maximum.col_cost_ == std::vector<double>({0, 3}) && maximum.offset_ == -7 && maximum.sense_ == ObjSense::kMinimize &&
        maximum.a_matrix_.value_ == original.a_matrix_.value_ && original.col_cost_[1] == -3, "最大化副本归一化错误");
    details.push_back("最小化与最大化副本归一化保持约束和原输入");
  }
  {
    const auto s = summarize(fixture(), budget);
    require(s.n == 2 && s.m == 2 && s.nnz == 4 && s.matrix.finite_count == 4 && s.matrix.zero_count == 1 &&
        s.matrix.min_abs == 0.5 && s.matrix.max_abs == 4 && s.cost.zero_count == 1 && s.cost.min_abs == 3 &&
        s.bounds.finite_count == 5 && s.bounds.nonfinite_count == 3 && s.bounds.min_abs == 0 && s.bounds.max_abs == 5,
        "有限非零范围或无穷界统计错误");
    Range empty; empty.add(0, false); empty.add(-0.0, false); empty.add(kHighsInf, false);
    require(!empty.min_abs && !empty.max_abs && empty.zero_count == 2 && empty.nonfinite_count == 1, "全零范围必须为null");
    details.push_back("系数排零范围与有限界含零范围及无样本null");
  }
  {
    for (int mode = 0; mode < 6; ++mode) {
      auto lp = fixture();
      if (mode == 0) lp.col_cost_[0] = std::numeric_limits<double>::quiet_NaN();
      if (mode == 1) lp.a_matrix_.index_[0] = 2;
      if (mode == 2) lp.a_matrix_.start_[1] = -1;
      if (mode == 3) lp.col_lower_[0] = kHighsInf;
      if (mode == 4) lp.a_matrix_.value_[0] = kHighsInf;
      if (mode == 5) lp.integrality_ = {HighsVarType::kInteger, HighsVarType::kContinuous};
      bool rejected = false;
      try { static_cast<void>(summarize(lp, budget)); } catch (const InputError&) { rejected = true; }
      require(rejected, "非法输入未被拒绝");
    }
    details.push_back("非法系数索引列边界与界方向在库调用前拒绝");
  }
  {
    double now = 0;
    Deadline exact([&] { return now; });
    now = std::nextafter(external_seconds, 0.0); exact.check();
    require(exact.remaining() > 0, "截止前剩余预算必须为正");
    now = external_seconds;
    bool expired = false;
    try { exact.check(); } catch (const DeadlineExpired&) { expired = true; }
    require(expired, "恰好截止未被拒绝");
    details.push_back("同一外部截止在等于上限时拒绝接纳");
  }
  {
    auto lp = fixture();
    lp.num_col_ = 1; lp.num_row_ = 600; lp.col_cost_ = {1}; lp.col_lower_ = {0}; lp.col_upper_ = {kHighsInf};
    lp.row_lower_.assign(600, -kHighsInf); lp.row_upper_.assign(600, 1);
    lp.a_matrix_.num_col_ = 1; lp.a_matrix_.num_row_ = 600; lp.a_matrix_.start_ = {0, 600};
    lp.a_matrix_.index_.resize(600); lp.a_matrix_.value_.assign(600, 1);
    for (HighsInt k = 0; k < 600; ++k) lp.a_matrix_.index_[static_cast<std::size_t>(k)] = k;
    int samples = 0;
    Deadline periodic([&] { return ++samples == 1 ? 0.0 : external_seconds; });
    std::optional<Summary> committed;
    bool expired = false;
    try { committed = summarize(lp, periodic); } catch (const DeadlineExpired&) { expired = true; }
    require(expired && !committed && periodic.loop_units == budget_check_work_interval && periodic.checks == 2 &&
        lp.a_matrix_.value_.size() == 600 && lp.a_matrix_.value_.front() == 1, "真实长列统计未中断或部分结果被提交");
    auto original = fixture(); original.sense_ = ObjSense::kMaximize;
    original.num_col_ = 600; original.num_row_ = 0; original.col_cost_.assign(600, 1);
    original.col_lower_.assign(600, 0); original.col_upper_.assign(600, kHighsInf);
    original.row_lower_.clear(); original.row_upper_.clear(); original.a_matrix_.clear();
    original.a_matrix_.num_col_ = 600; original.a_matrix_.start_.assign(601, 0);
    auto trial = original;
    samples = 0;
    Deadline normalization_budget([&] { return ++samples == 1 ? 0.0 : external_seconds; });
    expired = false;
    try { normalize_minimum(trial, normalization_budget); } catch (const DeadlineExpired&) { expired = true; }
    require(expired && normalization_budget.loop_units == budget_check_work_interval && original.col_cost_.back() == 1 &&
        original.sense_ == ObjSense::kMaximize, "真实归一化循环未中断或污染原副本");
    details.push_back("统计与归一化自有循环共享固定256检查周期");
  }
  {
    double now = 0;
    Deadline post([&] { return now; });
    Calls call;
    bool expired = false;
    try {
      post.check(); ++call.started;
      now = external_seconds; ++call.completed;
      post.check();
    } catch (const DeadlineExpired&) { expired = true; }
    require(expired && call.started == 1 && call.completed == 1, "库正常返回后超时账本错误");
    details.push_back("不透明调用返回后超时保留完成账本但不接纳");
  }
  std::fprintf(stdout, "{\"passed\":true,\"public_lp_runs\":0,\"explicit_lp_solve_calls\":0,\"cases\":%llu,\"default_limits\":{\"external_seconds\":%.17g,\"budget_check_work_interval\":%llu},\"details\":[",
      static_cast<unsigned long long>(details.size()), external_seconds, static_cast<unsigned long long>(budget_check_work_interval));
  for (std::size_t k = 0; k < details.size(); ++k) { if (k) std::fputc(',', stdout); quoted(stdout, details[k]); }
  std::fputs("]}\n", stdout);
  return 0;
}

}  // 匿名命名空间

int main(int argc, char** argv) {
  const auto start = Clock::now();
  if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
    try { return self_test(); }
    catch (const std::exception& error) { std::fputs("{\"passed\":false,\"error\":", stdout); quoted(stdout, error.what()); std::fputs("}\n", stdout); return 2; }
  }
  Deadline budget([&] { return std::chrono::duration<double>(Clock::now() - start).count(); });
  Report report;
  int exit_code = 2;
  try {
    budget.check();
    if (argc != 2 || !std::filesystem::is_regular_file(std::filesystem::path(argv[1]))) throw InputError("需要一个有效的MPS输入文件");
    scan(argv[1], report, budget);
    // 统计对象完整后检查；stdout序列化和写出由外部35秒进程上限约束。
    budget.check();
    exit_code = report.scan_complete ? 0 : 2;
  } catch (const DeadlineExpired&) {
    report.status = "deadline"; report.scan_complete = false; report.deadline_exceeded = true; exit_code = 4;
  } catch (const ReadError& error) { report.status = "read_error"; report.error = error.what(); }
  catch (const InputError& error) { report.status = "input_error"; report.error = error.what(); }
  catch (const ApiError& error) { report.status = "api_error"; report.error = error.what(); }
  catch (const std::exception& error) { report.status = "exception"; report.error = error.what(); }
  catch (...) { report.status = "exception"; report.error = "未知异常"; }
  report.timings.total = budget.elapsed();
  if (report.timings.total >= external_seconds) {
    report.status = "deadline"; report.scan_complete = false; report.deadline_exceeded = true; exit_code = 4;
  }
  write_report(stdout, report, budget);
  return exit_code;
}
