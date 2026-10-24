#pragma once
// 仅在冻结副本中观察已有量；不新增求解、矩阵乘积、时钟或预算检查。
#include <array>
#include <bit>
#include <cfenv>
#include <cfloat>
#include <cmath>
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
#include <ostream>
#include <span>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace solver::v67krylov {
inline constexpr std::size_t maximum_events = 1, maximum_columns = 32;
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_limit = 65536;
inline constexpr char magic[] = "V67KRY01";

struct Counts {
  std::int64_t completed_iterations = 0, factorizations = 0, refinements = 0;
  std::int64_t constant_solves = 0, direction_solves = 0, constant_repairs = 0;
  std::int64_t constant_repair_successes = 0;
};
struct Cycle {
  bool started = false, ended = false;
  std::size_t columns_started = 0, preconditioner_solves_completed = 0, columns_completed = 0, candidates = 0, remaining_zero_count = 0;
};
struct Work {
  std::size_t ir_candidates = 0, ir_corrections_started = 0, ir_corrections_completed = 0;
  std::size_t krylov_calls = 0;
  std::array<Cycle, 3> cycles{};
};
struct Field {
  const char* name = "", *dtype = "float64";
  const void* source = nullptr;
  std::size_t count = 0, width = 8, offset = 0, rank = 1;
  std::array<std::size_t, 2> shape{};
};
struct Context {
  std::span<const std::int64_t> starts, rows;
  std::span<const double> values, h, rhs;
  std::size_t n = 0, m = 0;
  const void* result = nullptr;
  Counts (*read_counts)(const void*) = nullptr;
  std::int64_t iteration = 0;
};
struct Event {
  std::unique_ptr<unsigned char[]> payload;
  std::array<Field, 10> arrays{};
  std::size_t payload_bytes = 0, n = 0, m = 0, nnz = 0, generated_columns = 0, coefficient_count = 0;
  std::int64_t iteration = 0;
  int restart_index = -1, column_index = -1;
  double global_tolerance = 0;
  bool block_tolerances_available = false;
  std::array<double, 2> block_tolerances{};
  Counts before_counts, after_counts;
  Work work;
};
struct State {
  bool enabled = false, run_started = false, active = false, prepared = false, terminal = false;
  bool truncated = false, event_available = false, current_terminal = false, flushed = false;
  std::size_t capture_started = 0, capture_errors = 0, terminal_failures = 0;
  std::size_t ordinary_scopes_started = 0, ordinary_scopes_completed = 0, ordinary_scopes_unwound = 0;
  std::array<char, 4096> prefix{};
  std::array<char, 256> error{};
  std::array<char, metadata_limit> metadata{};
  Context context;
  Counts before_counts, last_after_counts;
  Work work;
  std::array<double, maximum_columns> coefficients{};
  std::size_t coefficient_count = 0;
  int coefficient_restart = -1, coefficient_column = -1, rounding_mode = -1;
  const char* last_scope_exit = "not_started";
  Event event;
#ifdef SOLVER_V67_SELFTEST
  std::size_t test_cache_limit = cache_limit;
#endif
};
inline State state;
static_assert(sizeof(double) == 8 && sizeof(std::int64_t) == 8);
static_assert(sizeof(State) < cache_limit);

inline void fail(const char* reason, bool truncated = false) noexcept {
  ++state.capture_errors;
  state.truncated = state.truncated || truncated;
  if (!state.error[0]) std::snprintf(state.error.data(), state.error.size(), "%s", reason);
}
inline void quoted(std::ostream& out, const char* text) {
  out << '"';
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
    if (*p == '"' || *p == '\\') out << '\\' << static_cast<char>(*p);
    else if (*p < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(*p) << std::dec;
    else out << static_cast<char>(*p);
  }
  out << '"';
}
inline void real_hex(std::ostream& out, double value) {
  if (std::isnan(value)) { out << "\"nan\""; return; }
  if (std::isinf(value)) { out << (value < 0 ? "\"-inf\"" : "\"inf\""); return; }
  out << '"' << std::hexfloat << std::setprecision(std::numeric_limits<double>::max_digits10)
      << value << std::defaultfloat << '"';
}
inline void write_counts(std::ostream& out, const Counts& c) {
  out << "{\"completed_iterations\":" << c.completed_iterations << ",\"factorizations\":" << c.factorizations
      << ",\"refinements\":" << c.refinements << ",\"constant_solves\":" << c.constant_solves
      << ",\"direction_solves\":" << c.direction_solves << ",\"constant_repairs\":" << c.constant_repairs
      << ",\"constant_repair_successes\":" << c.constant_repair_successes << '}';
}
inline void write_work(std::ostream& out, const Work& w) {
  out << "{\"ir_candidates\":" << w.ir_candidates << ",\"ir_corrections_started\":" << w.ir_corrections_started
      << ",\"ir_corrections_completed\":" << w.ir_corrections_completed << ",\"krylov_calls\":" << w.krylov_calls
      << ",\"cycles\":[";
  for (std::size_t i = 0; i < w.cycles.size(); ++i) {
    const auto& c = w.cycles[i];
    out << (i ? "," : "") << "{\"restart_index\":" << i << ",\"started\":" << c.started
        << ",\"ended\":" << c.ended << ",\"columns_started\":" << c.columns_started
        << ",\"preconditioner_solves_completed\":" << c.preconditioner_solves_completed
        << ",\"columns_completed\":" << c.columns_completed << ",\"candidates\":" << c.candidates
        << ",\"remaining_zero_count\":" << c.remaining_zero_count << '}';
  }
  out << "]}";
}
template<class Scalar> inline void write_floating(std::ostream& out) {
  out << "{\"bytes\":" << sizeof(Scalar) << ",\"digits\":" << std::numeric_limits<Scalar>::digits
      << ",\"min_exponent\":" << std::numeric_limits<Scalar>::min_exponent
      << ",\"max_exponent\":" << std::numeric_limits<Scalar>::max_exponent
      << ",\"is_iec559\":" << std::numeric_limits<Scalar>::is_iec559 << '}';
}
inline const char* status() noexcept {
  if (state.capture_errors) return "capture_error";
  if (state.event_available) return "captured";
  return state.run_started ? "no_target_failure" : "not_started";
}
inline void write_metadata(std::ostream& out) {
  out << std::boolalpha << "{\"schema\":1,\"status\":"; quoted(out, status());
  out << ",\"run_started\":" << state.run_started << ",\"capture_started\":" << state.capture_started
      << ",\"capture_errors\":" << state.capture_errors << ",\"truncated\":" << state.truncated
      << ",\"terminal\":" << state.terminal << ",\"terminal_failures\":" << state.terminal_failures
      << ",\"event_count\":" << (state.event_available ? 1 : 0) << ",\"error\":"; quoted(out, state.error.data());
  out << ",\"ordinary_scopes_started\":" << state.ordinary_scopes_started
      << ",\"ordinary_scopes_completed\":" << state.ordinary_scopes_completed
      << ",\"ordinary_scopes_unwound\":" << state.ordinary_scopes_unwound << ",\"last_scope_exit\":";
  quoted(out, state.last_scope_exit);
  out << ",\"last_scope_work\":"; write_work(out, state.work);
  out << ",\"last_scope_after_counts\":"; write_counts(out, state.last_after_counts);
  out << ",\"fixed_cache_bytes\":" << sizeof(State)
      << ",\"binary_bytes\":" << (8 + (state.event_available ? state.event.payload_bytes : 0))
      << ",\"byte_order\":"; quoted(out, std::endian::native == std::endian::little ? "little" : "big");
  out << ",\"limits\":{\"maximum_events\":1,\"maximum_columns\":32,\"cache_limit_bytes\":" << cache_limit
      << ",\"metadata_limit_bytes\":" << metadata_limit << "},\"floating_point\":{\"double\":";
  write_floating<double>(out); out << ",\"long_double\":"; write_floating<long double>(out);
  out << ",\"flt_eval_method\":" << FLT_EVAL_METHOD << "},\"rounding_mode\":{\"value\":" << state.rounding_mode
      << ",\"is_to_nearest\":" << (state.rounding_mode == FE_TONEAREST) << "},\"events\":[";
  if (state.event_available) {
    const auto& e = state.event;
    out << "{\"index\":0,\"phase\":\"ordinary_constant\",\"linear_exit\":\"krylov_exhausted\",\"terminal\":true"
        << ",\"iteration\":" << e.iteration << ",\"n\":" << e.n << ",\"m\":" << e.m
        << ",\"dimension\":" << (e.n + e.m) << ",\"nnz\":" << e.nnz
        << ",\"restart_index\":" << e.restart_index << ",\"column_index\":" << e.column_index
        << ",\"generated_columns\":" << e.generated_columns << ",\"coefficient_count\":" << e.coefficient_count
        << ",\"payload_bytes\":" << e.payload_bytes << ",\"cache_bytes\":" << (sizeof(State) + e.payload_bytes)
        << ",\"base_available\":true,\"z_available\":true,\"coefficients_available\":true"
        << ",\"last_candidate_available\":true,\"last_residual_available\":true,\"linear_rhs_available\":true"
        << ",\"candidate_submitted\":false,\"residual_sign\":\"rhs_minus_K0_candidate\",\"global_tolerance\":";
    real_hex(out, e.global_tolerance); out << ",\"block_tolerances\":";
    if (e.block_tolerances_available) {
      out << '['; real_hex(out, e.block_tolerances[0]); out << ','; real_hex(out, e.block_tolerances[1]); out << ']';
    } else out << "null";
    out << ",\"before_counts\":"; write_counts(out, e.before_counts);
    out << ",\"after_counts\":"; write_counts(out, e.after_counts);
    out << ",\"work\":"; write_work(out, e.work);
    out << ",\"arrays\":[";
    for (std::size_t i = 0; i < e.arrays.size(); ++i) {
      const auto& f = e.arrays[i];
      out << (i ? "," : "") << "{\"name\":"; quoted(out, f.name); out << ",\"dtype\":"; quoted(out, f.dtype);
      out << ",\"count\":" << f.count << ",\"width\":" << f.width << ",\"offset\":" << f.offset
          << ",\"shape\":[" << f.shape[0];
      if (f.rank == 2) out << ',' << f.shape[1];
      out << "]}";
    }
    out << "]}";
  }
  out << "]}\n";
}
struct FixedBuffer : std::streambuf {
  FixedBuffer(char* data, std::size_t size) { setp(data, data + size); }
  std::size_t size() const noexcept { return static_cast<std::size_t>(pptr() - pbase()); }
};
inline void flush() noexcept {
  if (!state.enabled || state.flushed) return;
  state.flushed = true;
  bool write_ok = false;
  try {
    if (!state.prefix[0]) throw std::runtime_error("采集输出前缀无效");
    FixedBuffer buffer(state.metadata.data(), state.metadata.size());
    std::ostream metadata(&buffer); metadata.imbue(std::locale::classic());
    write_metadata(metadata);
    if (!metadata) { fail("metadata_limit", true); }
    else {
      std::array<char, 4102> path{};
      std::snprintf(path.data(), path.size(), "%s.bin", state.prefix.data());
      std::ofstream binary(path.data(), std::ios::binary | std::ios::trunc);
      binary.write(magic, 8);
      if (state.event_available) binary.write(reinterpret_cast<const char*>(state.event.payload.get()),
          static_cast<std::streamsize>(state.event.payload_bytes));
      binary.close();
      if (!binary) fail("binary_write");
      else {
        std::snprintf(path.data(), path.size(), "%s.json", state.prefix.data());
        std::ofstream json(path.data(), std::ios::binary | std::ios::trunc);
        json.write(state.metadata.data(), static_cast<std::streamsize>(buffer.size()));
        json.close();
        if (!json) fail("metadata_write"); else write_ok = true;
      }
    }
  } catch (...) { fail("output_exception"); }
  std::fprintf(stderr, "V67_KRYLOV {\"schema\":1,\"status\":\"%s\",\"event_count\":%u,\"capture_errors\":%llu,"
      "\"truncated\":%s,\"terminal_failures\":%llu,\"write_ok\":%s}\n", status(), state.event_available ? 1U : 0U,
      static_cast<unsigned long long>(state.capture_errors), state.truncated ? "true" : "false",
      static_cast<unsigned long long>(state.terminal_failures), write_ok ? "true" : "false");
}
struct Report {
  Report() noexcept {
    const auto* value = std::getenv("SOLVER_V67_CAPTURE");
    if (!value || !*value) return;
    state.enabled = true;
    const auto length = std::strlen(value);
    if (length >= state.prefix.size()) { fail("prefix_limit", true); return; }
    std::memcpy(state.prefix.data(), value, length + 1);
    state.rounding_mode = std::fegetround();
  }
  ~Report() noexcept { flush(); }
};
inline Report report;
inline void begin_run() noexcept { if (state.enabled) state.run_started = true; }
template<class Result> Counts read_counts(const void* ptr) noexcept {
  const auto& r = *static_cast<const Result*>(ptr);
  return {r.ipm_iterations.value_or(0), r.ipm_factorizations.value_or(0), r.ipm_refinement_steps.value_or(0),
      r.ipm_homogeneous_constant_solves.value_or(0), r.ipm_homogeneous_direction_solves.value_or(0),
      r.ipm_homogeneous_constant_repairs.value_or(0), r.ipm_homogeneous_constant_repair_successes.value_or(0)};
}
struct Scope {
  bool active = false;
  int exceptions = 0;
  template<class Model, class Result> Scope(const Model& model, std::span<const double> diagonal,
      std::span<const double> rhs, std::int64_t iteration, const Result& result) noexcept {
    if (!state.enabled || state.event_available) return;
    if (state.active) { fail("nested_scope"); return; }
    active = state.active = true;
    exceptions = std::uncaught_exceptions();
    ++state.ordinary_scopes_started;
    state.context = {model.starts, model.rows, model.values, diagonal, rhs, model.costs.size(), model.rhs.size(),
        &result, &read_counts<Result>, iteration};
    state.before_counts = read_counts<Result>(&result);
    state.work = {};
    state.prepared = state.current_terminal = false;
    state.coefficient_count = 0;
    state.coefficient_restart = state.coefficient_column = -1;
    state.last_scope_exit = "active";
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept {
    if (!active) return;
    state.last_after_counts = state.context.read_counts(state.context.result);
    if (std::uncaught_exceptions() > exceptions) {
      ++state.ordinary_scopes_unwound;
      state.last_scope_exit = state.current_terminal ? "krylov_exhausted" : "nonterminal_exception";
    } else {
      ++state.ordinary_scopes_completed;
      state.last_scope_exit = state.current_terminal ? "terminal_without_throw" : "returned";
      if (state.current_terminal) fail("terminal_without_throw");
    }
    if (!state.event_available) state.event = {};
    state.prepared = state.active = false;
    state.context = {};
  }
};
inline void ir_candidate() noexcept { if (state.active) ++state.work.ir_candidates; }
inline void ir_correction_started() noexcept { if (state.active) ++state.work.ir_corrections_started; }
inline void ir_correction_completed() noexcept { if (state.active) ++state.work.ir_corrections_completed; }
inline void krylov_begin() noexcept { if (state.active) ++state.work.krylov_calls; }
inline void cycle_begin(int restart) noexcept {
  if (!state.active) return;
  if (restart < 0 || restart >= 3) { fail("restart_range"); return; }
  state.work.cycles[static_cast<std::size_t>(restart)].started = true;
}
inline void column_begin(int restart) noexcept {
  if (state.active) ++state.work.cycles[static_cast<std::size_t>(restart)].columns_started;
}
inline void preconditioner_complete(int restart) noexcept {
  if (state.active) ++state.work.cycles[static_cast<std::size_t>(restart)].preconditioner_solves_completed;
}
inline void column_complete(int restart, double remaining) noexcept {
  if (!state.active) return;
  auto& cycle = state.work.cycles[static_cast<std::size_t>(restart)];
  ++cycle.columns_completed;
  if (remaining == 0) ++cycle.remaining_zero_count;
}
inline void candidate(int restart, std::size_t column, std::span<const double> coefficients) noexcept {
  if (!state.active) return;
  ++state.work.cycles[static_cast<std::size_t>(restart)].candidates;
  if (restart != 2) return;
  if (coefficients.empty() || coefficients.size() > maximum_columns || column + 1 != coefficients.size()) {
    fail("coefficient_shape"); return;
  }
  std::memcpy(state.coefficients.data(), coefficients.data(), coefficients.size_bytes());
  state.coefficient_count = coefficients.size();
  state.coefficient_restart = restart;
  state.coefficient_column = static_cast<int>(column);
}
inline void cycle_end(int restart, std::span<const double> base,
    const std::vector<std::vector<double>>& z, std::span<const double> last_candidate,
    std::span<const double> last_residual, double tolerance, const std::array<double, 2>* block_tolerances) noexcept {
  if (!state.active) return;
  state.work.cycles[static_cast<std::size_t>(restart)].ended = true;
  if (restart != 2 || state.event_available) return;
  ++state.capture_started;
  try {
    state.rounding_mode = std::fegetround();
    const auto& c = state.context;
    const auto dimension = c.n + c.m;
    if (c.n > std::numeric_limits<std::size_t>::max() - c.m || c.starts.size() != c.n + 1 ||
        c.rows.size() != c.values.size() || c.h.size() != c.m || c.rhs.size() != dimension ||
        base.size() != dimension || last_candidate.size() != dimension || last_residual.size() != dimension ||
        z.empty() || z.size() > maximum_columns || state.coefficient_restart != restart ||
        !state.coefficient_count || state.coefficient_count > z.size()) { fail("capture_shape"); return; }
    for (const auto& column : z) if (column.size() != dimension) { fail("z_shape"); return; }
    auto& e = state.event;
    e = {};
    e.n = c.n; e.m = c.m; e.nnz = c.values.size(); e.iteration = c.iteration;
    e.generated_columns = z.size(); e.coefficient_count = state.coefficient_count;
    e.restart_index = restart; e.column_index = state.coefficient_column;
    e.global_tolerance = tolerance;
    e.block_tolerances_available = block_tolerances != nullptr;
    if (block_tolerances) e.block_tolerances = *block_tolerances;
    e.before_counts = state.before_counts;
    const auto add = [&](std::size_t index, const char* name, const char* dtype, const void* data,
                         std::size_t count, std::size_t rank = 1, std::size_t rows = 0, std::size_t cols = 0) {
      if (count > (std::numeric_limits<std::size_t>::max() - e.payload_bytes) / 8) return false;
      e.arrays[index] = {name, dtype, data, count, 8, 8 + e.payload_bytes, rank, {rank == 1 ? count : rows, cols}};
      e.payload_bytes += count * 8;
      return true;
    };
    if (dimension > std::numeric_limits<std::size_t>::max() / z.size() ||
        !add(0, "g_starts", "int64", c.starts.data(), c.starts.size()) ||
        !add(1, "g_rows", "int64", c.rows.data(), c.rows.size()) ||
        !add(2, "g_values", "float64", c.values.data(), c.values.size()) ||
        !add(3, "h", "float64", c.h.data(), c.h.size()) ||
        !add(4, "linear_rhs", "float64", c.rhs.data(), c.rhs.size()) ||
        !add(5, "base", "float64", base.data(), base.size()) ||
        !add(6, "z", "float64", nullptr, dimension * z.size(), 2, z.size(), dimension) ||
        !add(7, "coefficients", "float64", state.coefficients.data(), state.coefficient_count) ||
        !add(8, "last_candidate", "float64", last_candidate.data(), last_candidate.size()) ||
        !add(9, "last_residual", "float64", last_residual.data(), last_residual.size())) {
      fail("capture_size_overflow", true); return;
    }
    std::size_t allowed = cache_limit;
#ifdef SOLVER_V67_SELFTEST
    allowed = state.test_cache_limit;
#endif
    if (allowed < sizeof(State) || e.payload_bytes > allowed - sizeof(State)) { fail("cache_limit", true); return; }
    // 所有大数组的字节上限已先核；一次分配后直接复制，不重算原算子。
    e.payload = std::make_unique<unsigned char[]>(e.payload_bytes);
    for (auto& f : e.arrays) {
      auto* destination = e.payload.get() + f.offset - 8;
      if (f.source && f.count) std::memcpy(destination, f.source, f.count * f.width);
      f.source = nullptr;
    }
    auto* z_destination = e.payload.get() + e.arrays[6].offset - 8;
    for (std::size_t j = 0; j < z.size(); ++j)
      if (dimension) std::memcpy(z_destination + j * dimension * 8, z[j].data(), dimension * 8);
    state.prepared = true;
  } catch (...) { fail("capture_exception"); }
}
inline void terminal_exhausted() noexcept {
  if (!state.active) return;
  state.terminal = state.current_terminal = true;
  ++state.terminal_failures;
  if (!state.prepared) { if (!state.capture_errors) fail("terminal_without_complete_capture"); return; }
  state.event.after_counts = state.context.read_counts(state.context.result);
  state.event.work = state.work;
  state.event_available = true;
}
}  // 命名空间 solver::v67krylov
