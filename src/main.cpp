#include "solver/backend.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr std::string_view version = SOLVER_VERSION;
constexpr std::string_view usage = "solver solve <模型.mps|模型.lp> [--backend highs|pdlp|ipm] [--time-limit 秒] [--threads 数量] [--seed 种子] [--mip-gap 比例] [--iteration-limit 迭代次数] [--scaling-passes 缩放轮数] [--pdlp-mode baseline|adaptive|subspace|halpern|halpern_reflected] [--pdlp-history-size 历史数量] [--highs-solver choose|simplex|ipm] [--highs-crossover on|off] [--highs-presolve choose|on|off] [--ipm-method mehrotra|homogeneous] [--ipm-singleton-elimination on|off] [--ipm-dynamic-regularization on|off] [--ipm-kkt-equilibration on|off] [--ipm-linear-accuracy strict|backward|inexact|blockwise|blockwise_weighted|blockwise_forcing|blockwise_forcing_weighted|adaptive_regularization|primal_dual_regularization|curvature_regularization|full_newton|bounded_newton|homogeneous_scaled] [--ipm-central-corrections 0|1] [--ipm-direction-retries 0..8] [--ipm-krylov-history-size 0..8] [--ipm-linear-solver qdldl|sparselu|normal|qdldl_extended|factorhighs|normal_augmented|normal_multifrontal|normal_extended|normal_refined|normal_cholmod] [--ipm-inexact-prefilter on|off] [--ipm-fixed-variable-elimination on|off] [--ipm-singleton-presolve on|off] [--ipm-presolve-fixed-elimination on|off] [--output 报告.json]；homogeneous_scaled 仅用于 homogeneous，以有限精化保留最佳有限方向，未达内层精度仍须通过原模型终验";

// 部分第三方解析诊断直接写入 stdout，在命令行边界将它们转发至 stderr。
class DiagnosticRedirect {
 public:
  DiagnosticRedirect() {
    std::cout.flush();
    std::fflush(stdout);
#ifdef _WIN32
    saved_ = _dup(_fileno(stdout));
    if (saved_ < 0) throw std::runtime_error("无法保存标准输出描述符");
    if (_dup2(_fileno(stderr), _fileno(stdout)) != 0) {
      _close(saved_);
      throw std::runtime_error("无法重定向第三方诊断输出");
    }
#else
    saved_ = dup(fileno(stdout));
    if (saved_ < 0) throw std::runtime_error("无法保存标准输出描述符");
    if (dup2(fileno(stderr), fileno(stdout)) < 0) {
      close(saved_);
      throw std::runtime_error("无法重定向第三方诊断输出");
    }
#endif
  }
  DiagnosticRedirect(const DiagnosticRedirect&) = delete;
  DiagnosticRedirect& operator=(const DiagnosticRedirect&) = delete;
  ~DiagnosticRedirect() {
    std::cout.flush();
    std::fflush(stdout);
#ifdef _WIN32
    _dup2(saved_, _fileno(stdout));
    _close(saved_);
#else
    dup2(saved_, fileno(stdout));
    close(saved_);
#endif
  }

 private:
  int saved_ = -1;
};

std::filesystem::path utf8_path(const std::string& value) {
  return std::filesystem::path(std::u8string(value.begin(), value.end()));
}

std::string quote(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (std::size_t index = 0; index < value.size();) {
    const auto ch = static_cast<unsigned char>(value[index]);
    if (ch >= 0x80) {
      // 原生诊断可能混有系统编码；仅保留合法 UTF-8，其余字节显式替换。
      const std::size_t length = ch >= 0xc2 && ch <= 0xdf ? 2 :
                                 ch >= 0xe0 && ch <= 0xef ? 3 :
                                 ch >= 0xf0 && ch <= 0xf4 ? 4 : 0;
      bool valid = length != 0 && index + length <= value.size();
      for (std::size_t offset = 1; valid && offset < length; ++offset) {
        const auto next = static_cast<unsigned char>(value[index + offset]);
        valid = next >= 0x80 && next <= 0xbf;
      }
      if (valid) {
        const auto second = static_cast<unsigned char>(value[index + 1]);
        valid = !(ch == 0xe0 && second < 0xa0) && !(ch == 0xed && second >= 0xa0) &&
                !(ch == 0xf0 && second < 0x90) && !(ch == 0xf4 && second >= 0x90);
      }
      if (valid) {
        out.write(value.data() + index, static_cast<std::streamsize>(length));
        index += length;
      } else {
        out << "\\ufffd";
        ++index;
      }
      continue;
    }
    ++index;
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<unsigned int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
    }
  }
  out << '"';
  return out.str();
}

std::string number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return out.str();
}

std::string number(const std::optional<double>& value) {
  return value ? number(*value) : "null";
}

std::string integer(const std::optional<std::int64_t>& value) {
  return value ? std::to_string(*value) : "null";
}

std::string boolean(bool value) { return value ? "true" : "false"; }

std::string strings(const std::vector<std::string>& values) {
  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) result += ',';
    result += quote(values[index]);
  }
  return result + ']';
}

std::string numbers(const std::vector<double>& values) {
  std::string result = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) result += ',';
    result += number(values[index]);
  }
  return result + ']';
}

std::string pdlp_options(const solver::PdlpOptions& options) {
  return "{\"mode\":" + quote(options.mode) +
         ",\"history_size\":" + std::to_string(options.history_size) +
         ",\"max_iterations\":" + std::to_string(options.max_iterations) +
         ",\"check_interval\":" + std::to_string(options.check_interval) +
         ",\"scaling_passes\":" + std::to_string(options.scaling_passes) +
         ",\"optimality\":{\"dual_absolute\":" + number(options.optimality.dual_absolute) +
         ",\"dual_relative\":" + number(options.optimality.dual_relative) +
         ",\"gap_absolute\":" + number(options.optimality.gap_absolute) +
         ",\"gap_relative\":" + number(options.optimality.gap_relative) + "}}";
}

std::string ipm_options(const solver::IpmOptions& options) {
  return "{\"method\":" + quote(options.method) +
         ",\"max_iterations\":" + std::to_string(options.max_iterations) +
         ",\"scaling_passes\":" + std::to_string(options.scaling_passes) +
         ",\"eliminate_singletons\":" + boolean(options.eliminate_singletons) +
         ",\"dynamic_regularization\":" + boolean(options.dynamic_regularization) +
         ",\"equilibrate_kkt\":" + boolean(options.equilibrate_kkt) +
         ",\"linear_accuracy\":" + quote(options.linear_accuracy) +
         ",\"central_corrections\":" + std::to_string(options.central_corrections) +
         ",\"direction_retries\":" + std::to_string(options.direction_retries) +
         ",\"krylov_history_size\":" + std::to_string(options.krylov_history_size) +
         ",\"linear_solver\":" + quote(options.linear_solver) +
         ",\"eliminate_fixed_variables\":" + boolean(options.eliminate_fixed_variables) +
         ",\"presolve_singleton_rows\":" + boolean(options.presolve_singleton_rows) +
         ",\"eliminate_presolved_fixed_variables\":" + boolean(options.eliminate_presolved_fixed_variables) +
         ",\"inexact_prefilter\":" + boolean(options.inexact_prefilter) + "}";
}

std::string highs_options(const solver::HighsOptions& options) {
  return "{\"solver\":" + quote(options.solver) +
         ",\"run_crossover\":" + quote(options.run_crossover) +
         ",\"presolve\":" + quote(options.presolve) + "}";
}

std::string path_string(const std::filesystem::path& value) {
  const auto utf8 = value.u8string();
  return {reinterpret_cast<const char*>(utf8.data()), utf8.size()};
}

struct Arguments {
  std::filesystem::path input;
  std::optional<std::filesystem::path> output;
  solver::SolveOptions options;
};

int parse_integer(std::string_view value, std::string_view name, int minimum) {
  int parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < minimum)
    throw std::invalid_argument(std::string(name) + " 必须是大于等于 " + std::to_string(minimum) + " 的整数");
  return parsed;
}

std::int64_t parse_iteration_limit(std::string_view value) {
  std::int64_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed < 0)
    throw std::invalid_argument("--iteration-limit 必须是非负 64 位整数");
  return parsed;
}

double parse_number(std::string_view value, std::string_view name, bool allow_zero) {
  double parsed = 0.0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      !std::isfinite(parsed) || parsed < 0.0 || (!allow_zero && parsed == 0.0))
    throw std::invalid_argument(std::string(name) + (allow_zero ? " 必须是有限非负数" : " 必须是有限正数"));
  return parsed;
}

Arguments parse(const std::vector<std::string>& args) {
  if (args.size() < 3 || args[1] != "solve" || args[2].empty() || args[2].starts_with("--"))
    throw std::invalid_argument("用法：" + std::string(usage));
  Arguments parsed;
  parsed.input = utf8_path(args[2]);
  std::optional<std::int64_t> iteration_limit;
  std::optional<int> scaling_passes;
  std::vector<std::string> seen;
  for (std::size_t index = 3; index < args.size(); index += 2) {
    const auto& option = args[index];
    if (option != "--time-limit" && option != "--threads" && option != "--seed" &&
        option != "--mip-gap" && option != "--output" && option != "--backend" &&
        option != "--iteration-limit" && option != "--scaling-passes" && option != "--pdlp-mode" &&
        option != "--pdlp-history-size" && option != "--highs-solver" && option != "--highs-crossover" &&
        option != "--highs-presolve" && option != "--ipm-singleton-elimination" &&
        option != "--ipm-dynamic-regularization" && option != "--ipm-kkt-equilibration" &&
        option != "--ipm-method" && option != "--ipm-linear-accuracy" && option != "--ipm-central-corrections" &&
        option != "--ipm-direction-retries" && option != "--ipm-krylov-history-size" &&
        option != "--ipm-linear-solver" && option != "--ipm-inexact-prefilter" &&
        option != "--ipm-fixed-variable-elimination" && option != "--ipm-singleton-presolve" &&
        option != "--ipm-presolve-fixed-elimination")
      throw std::invalid_argument("未知参数：" + option);
    if (std::find(seen.begin(), seen.end(), option) != seen.end())
      throw std::invalid_argument("重复参数：" + option);
    seen.push_back(option);
    if (index + 1 >= args.size() || args[index + 1].empty())
      throw std::invalid_argument("参数缺少值：" + option);
    const auto& value = args[index + 1];
    if (option == "--time-limit") parsed.options.time_limit = parse_number(value, option, true);
    else if (option == "--threads") parsed.options.threads = parse_integer(value, option, 1);
    else if (option == "--seed") parsed.options.seed = parse_integer(value, option, 0);
    else if (option == "--mip-gap") parsed.options.mip_relative_gap = parse_number(value, option, true);
    else if (option == "--backend") {
      if (value != "highs" && value != "pdlp" && value != "ipm")
        throw std::invalid_argument("--backend 仅支持 highs、pdlp 或 ipm");
      parsed.options.backend = value;
    } else if (option == "--iteration-limit") iteration_limit = parse_iteration_limit(value);
    else if (option == "--scaling-passes") {
      scaling_passes = parse_integer(value, option, 0);
      if (*scaling_passes > 20)
        throw std::invalid_argument("--scaling-passes 必须在 0 到 20 之间");
    }
    else if (option == "--pdlp-mode") {
      if (value != "baseline" && value != "adaptive" && value != "subspace" &&
          value != "halpern" && value != "halpern_reflected")
        throw std::invalid_argument("--pdlp-mode 仅支持 baseline、adaptive、subspace、halpern 或 halpern_reflected");
      parsed.options.pdlp.mode = value;
    } else if (option == "--pdlp-history-size") {
      parsed.options.pdlp.history_size = parse_integer(value, option, 0);
      if (parsed.options.pdlp.history_size > 32)
        throw std::invalid_argument("--pdlp-history-size 必须在 0 到 32 之间");
    } else if (option == "--highs-solver") {
      if (value != "choose" && value != "simplex" && value != "ipm")
        throw std::invalid_argument("--highs-solver 仅支持 choose、simplex 或 ipm");
      parsed.options.highs.solver = value;
    } else if (option == "--highs-crossover") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--highs-crossover 仅支持 on 或 off");
      parsed.options.highs.run_crossover = value;
    } else if (option == "--highs-presolve") {
      if (value != "choose" && value != "on" && value != "off")
        throw std::invalid_argument("--highs-presolve 仅支持 choose、on 或 off");
      parsed.options.highs.presolve = value;
    } else if (option == "--ipm-method") {
      if (value != "mehrotra" && value != "homogeneous")
        throw std::invalid_argument("--ipm-method 仅支持 mehrotra 或 homogeneous");
      parsed.options.ipm.method = value;
    } else if (option == "--ipm-singleton-elimination") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-singleton-elimination 仅支持 on 或 off");
      parsed.options.ipm.eliminate_singletons = value == "on";
    } else if (option == "--ipm-dynamic-regularization") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-dynamic-regularization 仅支持 on 或 off");
      parsed.options.ipm.dynamic_regularization = value == "on";
    } else if (option == "--ipm-kkt-equilibration") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-kkt-equilibration 仅支持 on 或 off");
      parsed.options.ipm.equilibrate_kkt = value == "on";
    } else if (option == "--ipm-linear-accuracy") {
      if (value != "strict" && value != "backward" && value != "inexact" && value != "blockwise" && value != "blockwise_weighted" && value != "blockwise_forcing" && value != "blockwise_forcing_weighted" && value != "adaptive_regularization" && value != "primal_dual_regularization" && value != "curvature_regularization" && value != "full_newton" && value != "bounded_newton" && value != "homogeneous_scaled")
        throw std::invalid_argument("--ipm-linear-accuracy 仅支持 strict、backward、inexact、blockwise、blockwise_weighted、blockwise_forcing、blockwise_forcing_weighted、adaptive_regularization、primal_dual_regularization、curvature_regularization、full_newton、bounded_newton 或 homogeneous_scaled");
      parsed.options.ipm.linear_accuracy = value;
    } else if (option == "--ipm-direction-retries") {
      parsed.options.ipm.direction_retries = parse_integer(value, option, 0);
      if (parsed.options.ipm.direction_retries > 8)
        throw std::invalid_argument("--ipm-direction-retries 仅支持 0 到 8");
    } else if (option == "--ipm-krylov-history-size") {
      parsed.options.ipm.krylov_history_size = parse_integer(value, option, 0);
      if (parsed.options.ipm.krylov_history_size > 8)
        throw std::invalid_argument("--ipm-krylov-history-size 仅支持 0 到 8");
    } else if (option == "--ipm-central-corrections") {
      parsed.options.ipm.central_corrections = parse_integer(value, option, 0);
      if (parsed.options.ipm.central_corrections > 1)
        throw std::invalid_argument("--ipm-central-corrections 当前仅支持 0 或 1");
    } else if (option == "--ipm-linear-solver") {
      if (value != "qdldl" && value != "sparselu" && value != "normal" && value != "qdldl_extended" && value != "factorhighs" && value != "normal_augmented" && value != "normal_multifrontal" && value != "normal_extended" && value != "normal_refined" && value != "normal_cholmod")
        throw std::invalid_argument("--ipm-linear-solver 仅支持 qdldl、sparselu、normal、qdldl_extended、factorhighs、normal_augmented、normal_multifrontal、normal_extended、normal_refined 或 normal_cholmod");
      parsed.options.ipm.linear_solver = value;
    } else if (option == "--ipm-fixed-variable-elimination") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-fixed-variable-elimination 仅支持 on 或 off");
      parsed.options.ipm.eliminate_fixed_variables = value == "on";
    } else if (option == "--ipm-singleton-presolve") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-singleton-presolve 仅支持 on 或 off");
      parsed.options.ipm.presolve_singleton_rows = value == "on";
    } else if (option == "--ipm-presolve-fixed-elimination") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-presolve-fixed-elimination 仅支持 on 或 off");
      parsed.options.ipm.eliminate_presolved_fixed_variables = value == "on";
    } else if (option == "--ipm-inexact-prefilter") {
      if (value != "on" && value != "off")
        throw std::invalid_argument("--ipm-inexact-prefilter 仅支持 on 或 off");
      parsed.options.ipm.inexact_prefilter = value == "on";
    } else parsed.output = utf8_path(value);
  }
  if (parsed.options.backend == "highs" && (iteration_limit || scaling_passes))
    throw std::invalid_argument("--iteration-limit 和 --scaling-passes 仅适用于 pdlp 或 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-method") != seen.end())
    throw std::invalid_argument("--ipm-method 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-singleton-elimination") != seen.end())
    throw std::invalid_argument("--ipm-singleton-elimination 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-dynamic-regularization") != seen.end())
    throw std::invalid_argument("--ipm-dynamic-regularization 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-kkt-equilibration") != seen.end())
    throw std::invalid_argument("--ipm-kkt-equilibration 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-linear-accuracy") != seen.end())
    throw std::invalid_argument("--ipm-linear-accuracy 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-direction-retries") != seen.end())
    throw std::invalid_argument("--ipm-direction-retries 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-krylov-history-size") != seen.end())
    throw std::invalid_argument("--ipm-krylov-history-size 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-central-corrections") != seen.end())
    throw std::invalid_argument("--ipm-central-corrections 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-linear-solver") != seen.end())
    throw std::invalid_argument("--ipm-linear-solver 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-inexact-prefilter") != seen.end())
    throw std::invalid_argument("--ipm-inexact-prefilter 仅适用于 ipm 后端");
  if (parsed.options.backend != "pdlp" &&
      (std::find(seen.begin(), seen.end(), "--pdlp-mode") != seen.end() ||
       std::find(seen.begin(), seen.end(), "--pdlp-history-size") != seen.end()))
    throw std::invalid_argument("--pdlp-mode 和 --pdlp-history-size 仅适用于 pdlp 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-fixed-variable-elimination") != seen.end())
    throw std::invalid_argument("--ipm-fixed-variable-elimination 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-singleton-presolve") != seen.end())
    throw std::invalid_argument("--ipm-singleton-presolve 仅适用于 ipm 后端");
  if (parsed.options.backend != "ipm" &&
      std::find(seen.begin(), seen.end(), "--ipm-presolve-fixed-elimination") != seen.end())
    throw std::invalid_argument("--ipm-presolve-fixed-elimination 仅适用于 ipm 后端");
  if (parsed.options.backend != "highs" && parsed.options.threads != 1)
    throw std::invalid_argument("当前 pdlp 和 ipm 后端仅支持 --threads 1");
  if (parsed.options.backend != "highs" &&
      (std::find(seen.begin(), seen.end(), "--highs-solver") != seen.end() ||
       std::find(seen.begin(), seen.end(), "--highs-crossover") != seen.end() ||
       std::find(seen.begin(), seen.end(), "--highs-presolve") != seen.end()))
    throw std::invalid_argument("HiGHS 算法、crossover 与 presolve 参数仅适用于 highs 后端");
  if (parsed.options.backend == "ipm") {
    if (iteration_limit) parsed.options.ipm.max_iterations = *iteration_limit;
    if (scaling_passes) parsed.options.ipm.scaling_passes = *scaling_passes;
  } else if (parsed.options.backend == "pdlp") {
    if (iteration_limit) parsed.options.pdlp.max_iterations = *iteration_limit;
    if (scaling_passes) parsed.options.pdlp.scaling_passes = *scaling_passes;
  }
  return parsed;
}

std::string report(const Arguments& args, const solver::SolveResult& result) {
  const auto& model = result.model;
  const auto& options = args.options;
  const auto& checks = result.verification;
  const auto& tolerances = options.tolerances;
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"schema_version\":1,\"solver_version\":" << quote(version)
      << ",\"backend\":" << quote(result.backend) << ",\"backend_version\":" << quote(result.backend_version)
      << ",\"input\":" << quote(path_string(std::filesystem::absolute(args.input)))
      << ",\"options\":{\"backend\":" << quote(options.backend)
      << ",\"time_limit\":" << number(options.time_limit)
      << ",\"threads\":" << options.threads << ",\"seed\":" << options.seed
      << ",\"mip_relative_gap\":" << number(options.mip_relative_gap)
      << ",\"pdlp\":" << (options.backend == "pdlp" ? pdlp_options(options.pdlp) : "null")
      << ",\"ipm\":" << (options.backend == "ipm" ? ipm_options(options.ipm) : "null")
      << ",\"highs\":" << (options.backend == "highs" ? highs_options(options.highs) : "null")
      << ",\"tolerances\":{\"feasibility_absolute\":" << number(tolerances.feasibility_absolute)
      << ",\"feasibility_relative\":" << number(tolerances.feasibility_relative)
      << ",\"integrality\":" << number(tolerances.integrality)
      << ",\"objective_absolute\":" << number(tolerances.objective_absolute)
      << ",\"objective_relative\":" << number(tolerances.objective_relative) << "}}"
      << ",\"status\":" << quote(result.status)
      << ",\"backend_status\":" << quote(result.backend_status)
      << ",\"verification_level\":" << quote(result.verification_level)
      << ",\"message\":" << quote(result.message)
      << ",\"model\":{\"name\":" << quote(model.name)
      << ",\"columns\":" << model.costs.size() << ",\"rows\":" << model.row_lower.size()
      << ",\"nonzeros\":" << model.values.size()
      << ",\"sense\":" << quote(model.sense == solver::ObjectiveSense::minimize ? "minimize" : "maximize")
      << ",\"objective_offset\":" << number(model.objective_offset)
      << ",\"is_mip\":" << boolean(solver::is_mip(model)) << "}"
      << ",\"has_solution\":" << boolean(result.has_solution) << ",\"solution\":";
  out << (result.has_solution ? numbers(result.solution) : "null");
  out << ",\"objective\":" << number(result.objective)
      << ",\"best_bound\":" << number(result.best_bound)
      << ",\"absolute_gap\":" << number(result.absolute_gap)
      << ",\"relative_gap\":" << number(result.relative_gap)
      << ",\"nodes\":" << integer(result.nodes)
      << ",\"simplex_iterations\":" << integer(result.simplex_iterations)
      << ",\"pdlp_iterations\":" << integer(result.pdlp_iterations)
      << ",\"pdlp_acceleration_attempts\":" << integer(result.pdlp_acceleration_attempts)
      << ",\"pdlp_acceleration_accepted\":" << integer(result.pdlp_acceleration_accepted)
      << ",\"pdlp_operator_passes\":" << integer(result.pdlp_operator_passes)
      << ",\"ipm_iterations\":" << integer(result.ipm_iterations)
      << ",\"ipm_factorizations\":" << integer(result.ipm_factorizations)
      << ",\"ipm_refinement_steps\":" << integer(result.ipm_refinement_steps)
      << ",\"ipm_weighted_krylov_steps\":" << integer(result.ipm_weighted_krylov_steps)
      << ",\"ipm_krylov_recycle_projections\":" << integer(result.ipm_krylov_recycle_projections)
      << ",\"ipm_krylov_recycle_acceptances\":" << integer(result.ipm_krylov_recycle_acceptances)
      << ",\"ipm_krylov_recycle_vectors\":" << integer(result.ipm_krylov_recycle_vectors)
      << ",\"ipm_krylov_recycle_peak_dimension\":" << integer(result.ipm_krylov_recycle_peak_dimension)
      << ",\"ipm_regularization_searches\":" << integer(result.ipm_regularization_searches)
      << ",\"ipm_regularization_reductions\":" << integer(result.ipm_regularization_reductions)
      << ",\"ipm_unshifted_factorizations\":" << integer(result.ipm_unshifted_factorizations)
      << ",\"ipm_full_newton_preconditioner_solves\":" << integer(result.ipm_full_newton_preconditioner_solves)
      << ",\"ipm_full_newton_products\":" << integer(result.ipm_full_newton_products)
      << ",\"ipm_bounded_newton_returns\":" << integer(result.ipm_bounded_newton_returns)
      << ",\"ipm_bounded_newton_inexact_returns\":" << integer(result.ipm_bounded_newton_inexact_returns)
      << ",\"ipm_homogeneous_constant_solves\":" << integer(result.ipm_homogeneous_constant_solves)
      << ",\"ipm_homogeneous_direction_solves\":" << integer(result.ipm_homogeneous_direction_solves)
      << ",\"ipm_homogeneous_tau\":" << number(result.ipm_homogeneous_tau)
      << ",\"ipm_homogeneous_kappa\":" << number(result.ipm_homogeneous_kappa)
      << ",\"ipm_homogeneous_scaled_products\":" << integer(result.ipm_homogeneous_scaled_products)
      << ",\"ipm_homogeneous_scaled_refinement_steps\":" << integer(result.ipm_homogeneous_scaled_refinement_steps)
      << ",\"ipm_homogeneous_scaled_inexact_returns\":" << integer(result.ipm_homogeneous_scaled_inexact_returns)
      << ",\"ipm_direction_retries\":" << integer(result.ipm_direction_retries)
      << ",\"ipm_direction_retry_successes\":" << integer(result.ipm_direction_retry_successes)
      << ",\"ipm_primal_regularization\":" << number(result.ipm_primal_regularization)
      << ",\"ipm_dual_regularization\":" << number(result.ipm_dual_regularization)
      << ",\"ipm_schur_residual_products\":" << integer(result.ipm_schur_residual_products)
      << ",\"ipm_schur_refinement_steps\":" << integer(result.ipm_schur_refinement_steps)
      << ",\"ipm_schur_refinement_acceptances\":" << integer(result.ipm_schur_refinement_acceptances)
      << ",\"ipm_kkt_dimension\":" << integer(result.ipm_kkt_dimension)
      << ",\"ipm_factor_nonzeros\":" << integer(result.ipm_factor_nonzeros)
      << ",\"ipm_factor_backend\":" << (result.ipm_factor_backend ? quote(*result.ipm_factor_backend) : "null")
      << ",\"ipm_structural_fallbacks\":" << integer(result.ipm_structural_fallbacks)
      << ",\"ipm_eliminated_singletons\":" << integer(result.ipm_eliminated_singletons)
      << ",\"ipm_regularized_pivots\":" << integer(result.ipm_regularized_pivots)
      << ",\"ipm_backward_error_acceptances\":" << integer(result.ipm_backward_error_acceptances)
      << ",\"ipm_inexact_acceptances\":" << integer(result.ipm_inexact_acceptances)
      << ",\"ipm_inexact_prefilter_rejections\":" << integer(result.ipm_inexact_prefilter_rejections)
      << ",\"ipm_eliminated_fixed_variables\":" << integer(result.ipm_eliminated_fixed_variables)
      << ",\"ipm_presolved_singleton_rows\":" << integer(result.ipm_presolved_singleton_rows)
      << ",\"ipm_presolved_fixed_columns\":" << integer(result.ipm_presolved_fixed_columns)
      << ",\"ipm_central_correction_attempts\":" << integer(result.ipm_central_correction_attempts)
      << ",\"ipm_central_correction_acceptances\":" << integer(result.ipm_central_correction_acceptances)
      << ",\"lp_optimality\":";
  if (result.lp_optimality) {
    const auto& optimality = *result.lp_optimality;
    out << "{\"valid_input\":" << boolean(optimality.valid_input)
        << ",\"dual_feasible\":" << boolean(optimality.dual_feasible)
        << ",\"gap_satisfied\":" << boolean(optimality.gap_satisfied)
        << ",\"numerically_optimal\":" << boolean(optimality.numerically_optimal)
        << ",\"max_dual_residual\":" << number(optimality.max_dual_residual)
        << ",\"dual_objective_estimate\":" << number(optimality.dual_objective_estimate)
        << ",\"corrected_dual_bound\":" << number(optimality.corrected_dual_bound)
        << ",\"objective_gap\":" << number(optimality.objective_gap)
        << ",\"relative_objective_gap\":" << number(optimality.relative_objective_gap)
        << ",\"errors\":" << strings(optimality.errors) << '}';
    out << ",\"row_dual\":" << numbers(result.row_dual);
  } else {
    out << "null";
  }
  out
      << ",\"timings\":{\"read_seconds\":" << number(result.read_seconds)
      << ",\"solve_seconds\":" << number(result.solve_seconds)
      << ",\"verification_seconds\":" << number(result.verification_seconds)
      << ",\"total_seconds\":" << number(result.total_seconds) << "}"
      << ",\"verification\":{\"valid_input\":" << boolean(checks.valid_input)
      << ",\"feasible\":" << boolean(checks.feasible)
      << ",\"objective_consistent\":" << boolean(checks.objective_consistent)
      << ",\"recomputed_objective\":" << number(checks.objective)
      << ",\"max_row_violation\":" << number(checks.max_row_violation)
      << ",\"max_bound_violation\":" << number(checks.max_bound_violation)
      << ",\"max_integrality_violation\":" << number(checks.max_integrality_violation)
      << ",\"errors\":" << strings(checks.errors) << "}}";
  return out.str();
}

int exit_code(std::string_view status) {
  if (status == "optimal" || status == "infeasible" || status == "unbounded" || status == "empty") return 0;
  if (status == "error") return 2;
  if (status == "verification_failed") return 3;
  return 4;
}

int run(const std::vector<std::string>& args) {
  std::string requested_backend = solver::SolveOptions{}.backend;
  for (std::size_t index = 3; index + 1 < args.size(); index += 2)
    if (args[index] == "--backend") requested_backend = args[index + 1];
  try {
    if (args.size() == 2 && args[1] == "--version") {
      std::cout << "{\"schema_version\":1,\"solver_version\":" << quote(version)
                << ",\"backend\":\"highs\",\"backend_version\":" << quote(solver::backend_version())
                << ",\"supported_backends\":[\"highs\",\"pdlp\",\"ipm\"]}\n";
      return 0;
    }
    if (args.size() == 2 && args[1] == "--help") {
      const solver::SolveOptions defaults;
      std::cout << "{\"usage\":" << quote(usage)
                << ",\"defaults\":{\"backend\":" << quote(defaults.backend)
                << ",\"time_limit\":" << number(defaults.time_limit)
                << ",\"threads\":" << defaults.threads << ",\"seed\":" << defaults.seed
                << ",\"mip_gap\":" << number(defaults.mip_relative_gap)
                << ",\"pdlp\":" << pdlp_options(defaults.pdlp)
                << ",\"ipm\":" << ipm_options(defaults.ipm)
                << ",\"highs\":" << highs_options(defaults.highs) << "}"
                << ",\"exit_codes\":{\"0\":\"后端已判定结束\",\"2\":\"参数、输入或运行错误\",\"3\":\"独立候选解检查失败\",\"4\":\"预算耗尽或求解未完成\"}}\n";
      return 0;
    }
    const auto args_parsed = parse(args);
    // 先检查输出目标，避免计算完成后才发现不能写入或覆盖输入模型。
    if (args_parsed.output) {
      std::error_code error;
      if (std::filesystem::exists(*args_parsed.output, error))
        throw std::invalid_argument("输出文件已存在，拒绝覆盖：" + path_string(*args_parsed.output));
      if (error) throw std::invalid_argument("无法检查输出路径：" + error.message());
      const auto parent = args_parsed.output->parent_path();
      if (!parent.empty() && !std::filesystem::is_directory(parent))
        throw std::invalid_argument("输出目录不存在：" + path_string(parent));
    }
    solver::SolveResult result;
    {
      const DiagnosticRedirect redirect;
      result = solver::solve_file(args_parsed.input, args_parsed.options);
    }
    const auto serialized = report(args_parsed, result);
    if (args_parsed.output) {
      std::ofstream file(*args_parsed.output, std::ios::binary);
      file << serialized << '\n';
      file.close();
      if (!file) throw std::runtime_error("写入报告失败：" + path_string(*args_parsed.output));
    }
    std::cout << serialized << '\n';
    return exit_code(result.status);
  } catch (const std::exception& error) {
    std::cout << "{\"schema_version\":1,\"solver_version\":" << quote(version)
              << ",\"backend\":null,\"requested_backend\":" << quote(requested_backend)
              << ",\"status\":\"error\",\"verification_level\":\"none\",\"message\":"
              << quote(error.what()) << "}\n";
    return 2;
  }
}

}  // 匿名命名空间

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
  std::vector<std::string> args;
  args.reserve(static_cast<std::size_t>(argc));
  for (int index = 0; index < argc; ++index) {
    const int length = WideCharToMultiByte(CP_UTF8, 0, argv[index], -1, nullptr, 0, nullptr, nullptr);
    std::string value(static_cast<std::size_t>(length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, argv[index], -1, value.data(), length, nullptr, nullptr);
    value.resize(static_cast<std::size_t>(length - 1));
    args.push_back(std::move(value));
  }
  return run(args);
}
#else
int main(int argc, char* argv[]) { return run({argv, argv + argc}); }
#endif
