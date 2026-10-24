#pragma once
// 冻结副本专用：只复制已有量，矩阵乘积及误差分解均留给离线工具。
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace solver::v59kkt {
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_reserve = 64ULL * 1024;
inline constexpr std::size_t field_count = 23;
inline constexpr std::size_t first_reduced = 17, first_rhs = 18, first_solution = 19, first_full = 20;

struct Field {
  const char* name = "";
  const char* dtype = "";
  std::size_t count = 0, width = 0, offset = 0;
};
inline void quoted(std::ostream& out, const char* value) {
  out << '"';
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(value); *p; ++p) {
    if (*p == '"' || *p == '\\') out << '\\' << static_cast<char>(*p);
    else if (*p < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(*p) << std::dec;
    else out << static_cast<char>(*p);
  }
  out << '"';
}
inline void real_hex(std::ostream& out, double value) {
  out << '"' << std::hexfloat << std::setprecision(std::numeric_limits<double>::max_digits10)
      << value << std::defaultfloat << '"';
}

struct State {
  const char* prefix = std::getenv("SOLVER_V59_DIAGNOSTICS");
  bool enabled = prefix && *prefix;
  bool run_started = false, run_active = false, run_unwound = false, ready = false, captured = false;
  bool constant_active = false, direct_active = false, have_first_rhs = false, have_first_solution = false;
  bool have_first_answer = false;
  const char* status = "not_entered";
  std::size_t pending_bytes = 0, terminal_bytes = 0, cache_bytes = metadata_reserve, copied_bytes = 0;
  std::size_t constant_calls = 0, target_failures = 0, capture_count = 0, reduced_written = 0;
  std::size_t n = 0, m = 0, factor_dimension = 0, full_dimension = 0;
  std::int64_t iteration = 0;
  int regularization_level = 0;
  double primal_regularization = 0, dual_regularization = 0, tolerance = 0;
  std::array<char, 64> backend{};
  std::array<Field, field_count> fields{};
  std::unique_ptr<unsigned char[]> pending_payload, terminal_payload;

  ~State() noexcept {
    if (!enabled) return;
    try { write(); }
    catch (...) { std::fputs("V59_KKT {\"write_error\":true}\n", stderr); }
  }
  void reject(const char* reason) noexcept { ready = false; status = reason; direct_active = false; }
  bool add(Field& field, const char* name, const char* dtype, std::size_t count,
      std::size_t width, std::size_t& bytes) noexcept {
    if (width == 0 || bytes > cache_limit || count > (cache_limit - bytes) / width) return false;
    field = {name, dtype, count, width, bytes}; bytes += count * width; return true;
  }
  void copy(unsigned char* destination, const Field& field, const void* source) noexcept {
    const auto bytes = field.count * field.width;
    if (bytes) std::memcpy(destination + field.offset, source, bytes);
    copied_bytes += bytes;
  }
  void begin_run() noexcept {
    if (!enabled) return;
    if (run_started) { reject("multiple_runs_unsupported"); return; }
    run_started = true; run_active = true;
    if constexpr (std::endian::native != std::endian::little || sizeof(double) != 8 ||
        sizeof(std::size_t) != 8 || !std::numeric_limits<double>::is_iec559) {
      reject("unsupported_binary_format"); return;
    }
    if (std::strlen(prefix) > 4000) { reject("prefix_too_long"); return; }
    ready = true; status = "no_target_failure";
  }
  bool begin_constant(std::int64_t current_iteration) noexcept {
    if (!enabled || !run_active || !ready || captured) return false;
    if (constant_active) { reject("nested_constant_scope"); return false; }
    constant_active = true; iteration = current_iteration; ++constant_calls;
    have_first_rhs = have_first_solution = have_first_answer = false; reduced_written = 0;
    return true;
  }
  bool begin_direct(std::span<const double> rhs, std::size_t dimension) noexcept {
    if (!constant_active || !ready || captured) return false;
    if (have_first_answer) return false;
    if (direct_active) { reject("direct_sequence_mismatch"); return false; }
    if (!pending_payload) {
      full_dimension = rhs.size(); factor_dimension = dimension;
      bool valid = add(fields[first_reduced], "first_reduced_rhs", "f64_le", dimension, 8, pending_bytes);
      valid = add(fields[first_rhs], "first_factor_rhs", "f64_le", dimension, 8, pending_bytes) && valid;
      valid = add(fields[first_solution], "first_factor_solution", "f64_le", dimension, 8, pending_bytes) && valid;
      valid = add(fields[first_full], "first_answer", "f64_le", rhs.size(), 8, pending_bytes) && valid;
      if (!valid || pending_bytes > cache_limit - metadata_reserve) { reject("cache_limit"); return false; }
      cache_bytes = metadata_reserve + pending_bytes;
      try { pending_payload.reset(new unsigned char[pending_bytes]); }
      catch (const std::bad_alloc&) { reject("memory_allocation_failed"); return false; }
      catch (...) { reject("capture_exception"); return false; }
    }
    if (rhs.size() != full_dimension || dimension != factor_dimension) {
      reject("dimension_mismatch"); return false;
    }
    direct_active = true;
    if (dimension == rhs.size()) {
      // 无消元时已有 rhs 就是未置换、未均衡的约化右端。
      copy(pending_payload.get(), fields[first_reduced], rhs.data()); reduced_written = dimension;
    }
    return true;
  }
  void reduced(std::size_t position, double value) noexcept {
    if (!direct_active || !ready) return;
    if (position != reduced_written || position >= factor_dimension) { reject("reduced_rhs_sequence_mismatch"); return; }
    std::memcpy(pending_payload.get() + fields[first_reduced].offset + position * 8, &value, 8);
    copied_bytes += 8; ++reduced_written;
  }
  bool factor_input(std::span<const double> rhs) noexcept {
    if (!direct_active || !ready) return false;
    if (have_first_rhs || rhs.size() != factor_dimension || reduced_written != factor_dimension) {
      reject("factor_input_sequence_mismatch"); return false;
    }
    copy(pending_payload.get(), fields[first_rhs], rhs.data()); have_first_rhs = true; return true;
  }
  void factor_output(std::span<const double> solution) noexcept {
    if (!direct_active || !ready) return;
    if (!have_first_rhs || have_first_solution || solution.size() != factor_dimension) {
      reject("factor_output_sequence_mismatch"); return;
    }
    copy(pending_payload.get(), fields[first_solution], solution.data()); have_first_solution = true;
  }
  void complete_direct(std::span<const double> answer) noexcept {
    if (!direct_active || !ready) return;
    if (!have_first_solution || answer.size() != full_dimension) { reject("first_answer_sequence_mismatch"); return; }
    copy(pending_payload.get(), fields[first_full], answer.data()); have_first_answer = true; direct_active = false;
  }

  template<class Model, class Ap, class Ai, class Inverse, class Retained, class Singleton, class Result>
  void failure(const Model& model, const std::vector<double>& diagonal_h, const std::vector<double>& regularized_h,
      const Ap& ap, const Ai& ai, const std::vector<double>& ax, const std::vector<double>& factor_values,
      const std::vector<double>& equilibration, const Inverse& inverse, const Retained& retained,
      const Singleton& singleton, const std::vector<double>& singleton_value,
      const std::vector<double>& rhs, const std::vector<double>& answer, const std::vector<double>& residual,
      double original_tolerance, int level, const Result& result) noexcept {
    if (!constant_active || !ready || captured) return;
    ++target_failures;
    if (!result.ipm_primal_regularization || !result.ipm_dual_regularization) {
      reject("regularization_unavailable"); return;
    }
    if (direct_active || !have_first_answer || rhs.size() != full_dimension || answer.size() != full_dimension ||
        residual.size() != full_dimension || model.costs.size() + model.rhs.size() != full_dimension ||
        diagonal_h.size() != model.rhs.size() || regularized_h.size() != model.rhs.size() ||
        inverse.size() != factor_dimension || ap.size() != factor_dimension + 1 || ai.size() != ax.size() ||
        factor_values.size() != ax.size() || (!equilibration.empty() && equilibration.size() != factor_dimension) ||
        retained.size() != model.rhs.size() || singleton.size() != model.rhs.size() ||
        singleton_value.size() != model.rhs.size()) { reject("failure_dimension_mismatch"); return; }
    std::array<const void*, field_count> sources{};
    bool valid = true;
    const auto field = [&](std::size_t index, const char* name, const char* dtype, const auto& source) {
      if (sizeof(source[0]) != 8) { valid = false; return; }
      valid = add(fields[index], name, dtype, source.size(), 8, terminal_bytes) && valid;
      sources[index] = source.data();
    };
    field(0, "g_starts", "i64_le", model.starts); field(1, "g_rows", "i64_le", model.rows);
    field(2, "g_values", "f64_le", model.values); field(3, "costs", "f64_le", model.costs);
    field(4, "b", "f64_le", model.rhs); field(5, "diagonal_h", "f64_le", diagonal_h);
    field(6, "regularized_h", "f64_le", regularized_h); field(7, "ap", "i64_le", ap);
    field(8, "ai", "i64_le", ai); field(9, "ax", "f64_le", ax);
    field(10, "factor_values", "f64_le", factor_values); field(11, "equilibration", "f64_le", equilibration);
    field(12, "inverse", "u64_le", inverse); field(13, "retained_index", "i64_le", retained);
    field(14, "singleton_column", "i64_le", singleton); field(15, "singleton_value", "f64_le", singleton_value);
    field(16, "rhs", "f64_le", rhs); field(21, "terminal_answer", "f64_le", answer);
    field(22, "stored_residual", "f64_le", residual);
    if (!valid || terminal_bytes > cache_limit - metadata_reserve - pending_bytes) { reject("cache_limit"); return; }
    cache_bytes = metadata_reserve + pending_bytes + terminal_bytes;
    try {
      terminal_payload.reset(new unsigned char[terminal_bytes]);
      for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i >= first_reduced && i <= first_full) continue;
        copy(terminal_payload.get(), fields[i], sources[i]);
      }
      n = model.costs.size(); m = model.rhs.size(); regularization_level = level; tolerance = original_tolerance;
      primal_regularization = *result.ipm_primal_regularization;
      dual_regularization = *result.ipm_dual_regularization;
      if (!result.ipm_factor_backend || result.ipm_factor_backend->size() >= backend.size()) {
        reject("backend_unavailable"); return;
      }
      std::memcpy(backend.data(), result.ipm_factor_backend->c_str(), result.ipm_factor_backend->size() + 1);
      captured = true; capture_count = 1; status = "captured";
    } catch (const std::bad_alloc&) { reject("memory_allocation_failed"); }
    catch (...) { reject("capture_exception"); }
  }

  void write() const {
    std::array<char, 4096> json_path{}, binary_path{};
    if (!prefix || std::strlen(prefix) > 4000) {
      std::fputs("V59_KKT {\"write_error\":true,\"reason\":\"prefix_too_long\"}\n", stderr); return;
    }
    std::snprintf(json_path.data(), json_path.size(), "%s-kkt.json", prefix);
    std::snprintf(binary_path.data(), binary_path.size(), "%s-kkt.bin", prefix);
    std::ofstream binary(binary_path.data(), std::ios::binary | std::ios::trunc);
    std::ofstream out(json_path.data(), std::ios::trunc); out.imbue(std::locale::classic());
    const char magic[8] = {'V', '5', '9', 'K', 'K', 'T', '1', '\0'}; binary.write(magic, 8);
    std::size_t offset = 8;
    out << "{\"schema\":\"v59_kkt_1\",\"enabled\":true,\"run_started\":" << (run_started ? "true" : "false")
        << ",\"run_unwound\":" << (run_unwound ? "true" : "false") << ",\"ready\":" << (ready ? "true" : "false")
        << ",\"captured\":" << (captured ? "true" : "false") << ",\"status\":"; quoted(out, status);
    out << ",\"cache_limit\":" << cache_limit << ",\"metadata_reserve\":" << metadata_reserve
        << ",\"cache_bytes\":" << cache_bytes << ",\"pending_payload_bytes\":" << pending_bytes
        << ",\"terminal_payload_bytes\":" << terminal_bytes << ",\"copied_bytes\":" << copied_bytes
        << ",\"constant_calls\":" << constant_calls << ",\"target_failures\":" << target_failures
        << ",\"capture_count\":" << capture_count
        << ",\"extra_lp_products\":0,\"extra_lp_solves\":0,\"diagnostic_clock_reads\":0,\"scalars\":{\"n\":" << n
        << ",\"m\":" << m << ",\"factor_dimension\":" << factor_dimension << ",\"iteration\":" << iteration
        << ",\"regularization_level\":" << regularization_level << ",\"primal_regularization\":";
    real_hex(out, primal_regularization); out << ",\"dual_regularization\":"; real_hex(out, dual_regularization);
    out << ",\"tolerance\":"; real_hex(out, tolerance); out << ",\"backend\":"; quoted(out, backend.data());
    out << "},\"arrays\":[";
    if (captured) for (std::size_t i = 0; i < fields.size(); ++i) {
      if (i) out << ',';
      const auto& field = fields[i];
      out << "{\"name\":"; quoted(out, field.name); out << ",\"dtype\":"; quoted(out, field.dtype);
      out << ",\"count\":" << field.count << ",\"offset\":" << offset << '}';
      const auto* payload = i >= first_reduced && i <= first_full ? pending_payload.get() : terminal_payload.get();
      const auto bytes = field.count * field.width;
      if (bytes) binary.write(reinterpret_cast<const char*>(payload + field.offset), static_cast<std::streamsize>(bytes));
      offset += bytes;
    }
    binary.flush(); const bool binary_ok = static_cast<bool>(binary);
    out << "],\"binary_bytes\":" << offset << ",\"binary_written\":" << (binary_ok ? "true" : "false") << "}\n";
    out.flush();
    std::fprintf(stderr, "V59_KKT {\"write_error\":%s,\"captured\":%s}\n",
        out && binary_ok ? "false" : "true", captured ? "true" : "false");
  }
};
static_assert(sizeof(State) + 16384 < metadata_reserve, "诊断元数据必须落在总缓存预留内");
inline State& state() noexcept { static State value; return value; }

class RunScope {
  bool enabled_ = false;
  int exceptions_ = 0;
 public:
  RunScope() noexcept {
    auto& s = state(); if (!s.enabled) return;
    enabled_ = true; exceptions_ = std::uncaught_exceptions(); s.begin_run();
  }
  ~RunScope() noexcept {
    if (!enabled_) return;
    auto& s = state(); s.run_unwound = std::uncaught_exceptions() > exceptions_; s.run_active = false;
  }
};
class ConstantScope {
  bool active_ = false;
 public:
  explicit ConstantScope(std::int64_t iteration) noexcept {
    auto& s = state(); if (s.enabled) active_ = s.begin_constant(iteration);
  }
  ~ConstantScope() noexcept { if (active_) { state().constant_active = false; state().direct_active = false; } }
};
class DirectScope {
  bool active_ = false;
 public:
  DirectScope(std::span<const double> rhs, std::size_t dimension) noexcept {
    auto& s = state(); if (s.enabled) active_ = s.begin_direct(rhs, dimension);
  }
  void complete(std::span<const double> answer) noexcept { if (active_) state().complete_direct(answer); }
  ~DirectScope() noexcept { if (active_) state().direct_active = false; }
};
class FactorScope {
  bool active_ = false;
  int exceptions_ = 0;
  const std::vector<double>& rhs_;
 public:
  explicit FactorScope(const std::vector<double>& rhs) noexcept : rhs_(rhs) {
    auto& s = state(); if (!s.enabled) return;
    active_ = s.factor_input(rhs); if (active_) exceptions_ = std::uncaught_exceptions();
  }
  ~FactorScope() noexcept {
    if (active_ && std::uncaught_exceptions() == exceptions_) state().factor_output(rhs_);
  }
};
inline void reduced_rhs(std::size_t position, double value) noexcept {
  auto& s = state(); if (s.enabled) s.reduced(position, value);
}
}  // 命名空间 solver::v59kkt
