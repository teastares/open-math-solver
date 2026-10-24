#pragma once
// 仅用于冻结副本：复制已有量，不新增求解、矩阵乘积或时钟读取。
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace solver::v61hsd {
inline constexpr std::size_t event_limit = 8;
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_reserve = 64ULL * 1024;

struct Field {
  const char* name;
  const char* dtype;
  const void* data;
  std::size_t count, width, offset;
};

struct Counts {
  std::int64_t completed_iterations = 0, factorizations = 0, refinements = 0;
  std::int64_t constant_solves = 0, direction_solves = 0, constant_repairs = 0;
  std::int64_t constant_repair_successes = 0, extended_krylov_attempts = 0, extended_krylov_successes = 0;
};

struct ScalarValues {
  long double dot_u = 0, dot_v = 0, denominator = 0;
};

struct Context {
  std::array<Field, 15> fields{};
  const std::vector<bool>* inequality = nullptr;
  const void* result = nullptr;
  Counts (*read_counts)(const void*) = nullptr;
  double tau = 0, kappa = 0, theta = 0, primal_regularization = 0, dual_regularization = 0;
  int level = 0;
};

struct Event {
  std::unique_ptr<unsigned char[]> payload;
  std::unique_ptr<long double[]> u, before_v, after_v;
  std::size_t bytes = 0, dimension = 0, index = 0;
  Context context;
  Counts before_counts, after_counts;
  ScalarValues before, after;
  std::int64_t iteration = 0;
  const char* phase = "unknown";
  long double gap_rhs = 0, complementarity_rhs = 0;
  bool candidate_available = false, after_scalar_available = false, committed = false;
  std::array<char, 1024> failure{};
};
static_assert(event_limit * sizeof(Event) < metadata_reserve);

inline void quoted(std::ostream& out, const char* text) {
  out << '"';
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
    if (*p == '"' || *p == '\\') out << '\\' << static_cast<char>(*p);
    else if (*p < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << unsigned(*p) << std::dec;
    else out << static_cast<char>(*p);
  }
  out << '"';
}

inline void real_hex(std::ostream& out, long double value) {
  if (std::isnan(value)) { out << "\"nan\""; return; }
  if (std::isinf(value)) { out << (value < 0 ? "\"-inf\"" : "\"inf\""); return; }
  // 保留原 long double 全部位；写出时不得经由 double。
  out << '"' << std::hexfloat << std::setprecision(std::numeric_limits<long double>::max_digits10)
      << value << std::defaultfloat << '"';
}

inline void write_counts(std::ostream& out, const Counts& c) {
  out << "{\"completed_iterations\":" << c.completed_iterations << ",\"factorizations\":" << c.factorizations
      << ",\"refinements\":" << c.refinements << ",\"constant_solves\":" << c.constant_solves
      << ",\"direction_solves\":" << c.direction_solves << ",\"constant_repairs\":" << c.constant_repairs
      << ",\"constant_repair_successes\":" << c.constant_repair_successes
      << ",\"extended_krylov_attempts\":" << c.extended_krylov_attempts
      << ",\"extended_krylov_successes\":" << c.extended_krylov_successes << '}';
}

inline void write_scalar_values(std::ostream& out, const ScalarValues& v) {
  out << "{\"dot_u\":"; real_hex(out, v.dot_u);
  out << ",\"dot_v\":"; real_hex(out, v.dot_v);
  out << ",\"denominator\":"; real_hex(out, v.denominator); out << '}';
}

inline void write_wide(std::ostream& out, const long double* values, std::size_t count) {
  out << '[';
  for (std::size_t i = 0; i < count; ++i) {
    if (i) out << ',';
    real_hex(out, values[i]);
  }
  out << ']';
}

struct State {
  const char* prefix = std::getenv("SOLVER_V61_DIAGNOSTICS");
  bool enabled = prefix && *prefix;
  const char* status = "no_repair_events";
  const char* current_phase = "unknown";
  std::int64_t iteration = 0;
  std::size_t trigger_count = 0, count = 0, dropped_events = 0, payload_bytes = 0, cache_bytes = 0;
  Context* context = nullptr;
  Event* current_event = nullptr;
  ScalarValues last_values;
  bool have_last_values = false;
  std::array<std::unique_ptr<Event>, event_limit> events;

  ~State() noexcept {
    if (!enabled) return;
    try { write(); }
    catch (...) { std::fputs("V61_HSD {\"write_error\":true,\"status\":\"write_exception\"}\n", stderr); }
  }

  void write() const {
    const std::string base(prefix);
    std::ofstream binary(base + ".hsd.bin", std::ios::binary | std::ios::trunc);
    const char magic[8] = {'V', '6', '1', 'H', 'S', 'D', '1', '\0'};
    binary.write(magic, 8);
    for (std::size_t i = 0; i < count; ++i)
      binary.write(reinterpret_cast<const char*>(events[i]->payload.get()), static_cast<std::streamsize>(events[i]->bytes));
    binary.flush();
    const bool binary_written = static_cast<bool>(binary);
    std::ofstream out(base + ".hsd.json", std::ios::trunc);
    out.imbue(std::locale::classic());
    out << "{\"schema\":\"v61_hsd_1\",\"enabled\":true,\"status\":";
    quoted(out, binary_written ? status : "binary_write_failed");
    out << ",\"event_count\":" << count << ",\"trigger_count\":" << trigger_count
        << ",\"dropped_events\":" << dropped_events << ",\"payload_bytes\":" << payload_bytes
        << ",\"cache_bytes\":" << cache_bytes << ",\"limits\":{\"max_events\":" << event_limit
        << ",\"payload_limit_bytes\":" << cache_limit << ",\"metadata_reserve_bytes\":" << metadata_reserve
        << "},\"extra_lp_products\":0,\"extra_lp_solves\":0,\"long_double_digits\":"
        << std::numeric_limits<long double>::digits << ",\"long_double_bytes\":" << sizeof(long double)
        << ",\"long_double_max_exponent\":" << std::numeric_limits<long double>::max_exponent
        << ",\"long_double_min_exponent\":" << std::numeric_limits<long double>::min_exponent << ",\"events\":[";
    for (std::size_t i = 0; i < count; ++i) {
      if (i) out << ',';
      const auto& e = *events[i];
      const auto& c = e.context;
      out << "{\"index\":" << e.index << ",\"iteration\":" << e.iteration << ",\"phase\":";
      quoted(out, e.phase);
      out << ",\"regularization_level\":" << c.level << ",\"before_counts\":"; write_counts(out, e.before_counts);
      out << ",\"after_counts\":"; write_counts(out, e.after_counts);
      out << ",\"committed\":" << (e.committed ? "true" : "false")
          << ",\"candidate_available\":" << (e.candidate_available ? "true" : "false")
          << ",\"after_scalar_available\":" << (e.after_scalar_available ? "true" : "false") << ",\"failure\":";
      quoted(out, e.failure.data());
      out << ",\"scalars\":{\"tau\":"; real_hex(out, c.tau);
      out << ",\"kappa\":"; real_hex(out, c.kappa);
      out << ",\"theta\":"; real_hex(out, c.theta);
      out << ",\"primal_regularization\":"; real_hex(out, c.primal_regularization);
      out << ",\"dual_regularization\":"; real_hex(out, c.dual_regularization);
      out << ",\"gap_rhs\":"; real_hex(out, e.gap_rhs);
      out << ",\"complementarity_rhs\":"; real_hex(out, e.complementarity_rhs);
      out << "},\"before\":"; write_scalar_values(out, e.before);
      out << ",\"after\":";
      if (e.after_scalar_available) write_scalar_values(out, e.after); else out << "null";
      out << ",\"wide\":{\"u\":"; write_wide(out, e.u.get(), e.dimension);
      out << ",\"before_v\":"; write_wide(out, e.before_v.get(), e.dimension);
      out << ",\"after_v\":";
      if (e.candidate_available) write_wide(out, e.after_v.get(), e.dimension); else out << "null";
      out << "},\"arrays\":[";
      for (std::size_t j = 0; j < c.fields.size(); ++j) {
        if (j) out << ',';
        const auto& f = c.fields[j];
        out << "{\"name\":"; quoted(out, f.name);
        out << ",\"dtype\":"; quoted(out, f.dtype);
        out << ",\"count\":" << f.count << ",\"offset\":" << f.offset << '}';
      }
      out << "]}";
    }
    out << "]}\n";
    out.flush();
    std::fprintf(stderr, "V61_HSD {\"write_error\":%s,\"event_count\":%llu,\"status\":\"%s\"}\n",
        out && binary_written ? "false" : "true", static_cast<unsigned long long>(count),
        binary_written ? status : "binary_write_failed");
  }
};

inline State& state() noexcept { static State result; return result; }
inline void begin_run() noexcept { (void)state(); }
inline void phase(const char* value, std::int64_t iteration) noexcept {
  auto& s = state();
  if (!s.enabled) return;
  s.current_phase = value;
  s.iteration = iteration;
}

class Scope {
  std::optional<Context> value_;
  Context* previous_ = nullptr;
  Event* previous_event_ = nullptr;
  bool active_ = false;
 public:
  template<class Scalar, class Model, class Point, class Result>
  Scope(const Model& model, const Point& point, std::span<const double> diagonal,
      std::span<const double> regularized, int level, const std::vector<Scalar>&,
      const std::vector<double>& direction_rhs, const std::vector<double>& h,
      const std::vector<double>& primal_residual, const std::vector<double>& dual_residual,
      double tau, double kappa, double theta, const Result& result) noexcept {
    if constexpr (!std::is_same_v<Scalar, long double>) return;
    auto& s = state();
    if (!s.enabled) return;
    auto& value = value_.emplace();
    auto field = [](const char* name, const char* dtype, const auto& values) {
      return Field{name, dtype, values.data(), values.size(), sizeof(values[0]), 0};
    };
    value.fields = {field("starts", "i64_le", model.starts), field("rows", "i64_le", model.rows),
        field("values", "f64_le", model.values), field("costs", "f64_le", model.costs),
        field("rhs", "f64_le", model.rhs), field("x", "f64_le", point.x), field("s", "f64_le", point.s),
        field("z", "f64_le", point.z), field("diagonal_h", "f64_le", diagonal),
        field("regularized_h", "f64_le", regularized),
        Field{"inequality", "u8", nullptr, model.inequality.size(), 1, 0},
        field("direction_rhs", "f64_le", direction_rhs), field("h", "f64_le", h),
        field("primal_residual", "f64_le", primal_residual), field("dual_residual", "f64_le", dual_residual)};
    value.inequality = &model.inequality;
    value.tau = tau; value.kappa = kappa; value.theta = theta; value.level = level;
    value.primal_regularization = result.ipm_primal_regularization.value_or(0);
    value.dual_regularization = result.ipm_dual_regularization.value_or(0);
    value.result = &result;
    value.read_counts = [](const void* pointer) {
      const auto& r = *static_cast<const Result*>(pointer);
      return Counts{r.ipm_iterations.value_or(0), r.ipm_factorizations.value_or(0), r.ipm_refinement_steps.value_or(0),
          r.ipm_homogeneous_constant_solves.value_or(0), r.ipm_homogeneous_direction_solves.value_or(0),
          r.ipm_homogeneous_constant_repairs.value_or(0), r.ipm_homogeneous_constant_repair_successes.value_or(0),
          r.ipm_extended_krylov_attempts.value_or(0), r.ipm_extended_krylov_successes.value_or(0)};
    };
    previous_ = s.context; previous_event_ = s.current_event;
    s.context = &value; s.current_event = nullptr; s.have_last_values = false; active_ = true;
  }
  ~Scope() noexcept {
    if (!active_) return;
    auto& s = state();
    if (s.current_event) s.current_event->after_counts = value_->read_counts(value_->result);
    s.context = previous_; s.current_event = previous_event_; s.have_last_values = false;
  }
};

template<class Scalar>
inline void observe_values(long double dot_u, long double dot_v, long double denominator) noexcept {
  if constexpr (!std::is_same_v<Scalar, long double>) return;
  auto& s = state();
  if (!s.enabled || !s.context) return;
  s.last_values = {dot_u, dot_v, denominator}; s.have_last_values = true;
  if (s.current_event && s.current_event->candidate_available) {
    s.current_event->after = s.last_values;
    s.current_event->after_scalar_available = true;
  }
}

template<class Scalar>
inline void begin_repair(std::span<const Scalar> u, const std::vector<Scalar>& v,
    long double gap, long double complementarity) noexcept {
  if constexpr (std::is_same_v<Scalar, long double>) {
    auto& s = state();
    if (!s.enabled || !s.context) return;
    ++s.trigger_count;
    ++s.dropped_events;
    s.current_event = nullptr;
    if (!s.have_last_values || !std::isfinite(s.last_values.denominator) ||
        s.last_values.denominator > 0 || !std::isfinite(s.last_values.dot_u)) {
      s.status = "invalid_trigger"; return;
    }
    if (s.count >= event_limit) { s.status = "event_limit"; return; }
    if constexpr (std::endian::native != std::endian::little || sizeof(double) != 8 ||
        !std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::digits != 53) {
      s.status = "unsupported_binary_format"; return;
    }
    if (u.size() != v.size()) { s.status = "invalid_dimension"; return; }
    try {
      const auto remaining = cache_limit - metadata_reserve - s.cache_bytes;
      if (u.size() > remaining / (3 * sizeof(long double))) { s.status = "cache_limit"; return; }
      const auto wide_bytes = 3 * u.size() * sizeof(long double);
      std::size_t bytes = 0;
      for (const auto& f : s.context->fields) {
        if (f.count > (remaining - wide_bytes - bytes) / f.width) { s.status = "cache_limit"; return; }
        bytes += f.count * f.width;
      }
      auto event = std::make_unique<Event>();
      event->payload = std::make_unique<unsigned char[]>(bytes);
      event->u = std::make_unique<long double[]>(u.size());
      event->before_v = std::make_unique<long double[]>(u.size());
      event->after_v = std::make_unique<long double[]>(u.size());
      event->bytes = bytes; event->dimension = u.size(); event->index = s.count;
      event->context = *s.context; event->before_counts = s.context->read_counts(s.context->result);
      event->after_counts = event->before_counts;
      event->before = s.last_values; event->gap_rhs = gap; event->complementarity_rhs = complementarity;
      event->phase = s.current_phase; event->iteration = s.iteration;
      std::copy(u.begin(), u.end(), event->u.get());
      std::copy(v.begin(), v.end(), event->before_v.get());
      std::size_t offset = 0;
      for (auto& f : event->context.fields) {
        f.offset = 8 + s.payload_bytes + offset;
        if (f.width == 1) {
          for (std::size_t i = 0; i < f.count; ++i) event->payload[offset + i] = (*s.context->inequality)[i] ? 1 : 0;
        } else if (f.count) std::memcpy(event->payload.get() + offset, f.data, f.count * f.width);
        offset += f.count * f.width; f.data = nullptr;
      }
      event->context.inequality = nullptr; event->context.result = nullptr; event->context.read_counts = nullptr;
      s.current_event = event.get(); s.events[s.count++] = std::move(event);
      s.payload_bytes += bytes; s.cache_bytes += bytes + wide_bytes; --s.dropped_events;
      if (!s.dropped_events) s.status = "captured";
    } catch (...) { s.status = "capture_exception"; }
  }
}

template<class Scalar>
inline void candidate(const std::vector<Scalar>& v) noexcept {
  if constexpr (std::is_same_v<Scalar, long double>) {
    auto& s = state();
    if (!s.enabled || !s.current_event) return;
    auto& e = *s.current_event;
    if (v.size() != e.dimension) { s.status = "invalid_candidate_dimension"; return; }
    std::copy(v.begin(), v.end(), e.after_v.get());
    e.candidate_available = true;
  }
}

inline void committed() noexcept {
  auto& s = state();
  if (s.enabled && s.current_event) s.current_event->committed = true;
}

inline void failure(const char* message) noexcept {
  auto& s = state();
  if (s.enabled && s.current_event)
    std::strncpy(s.current_event->failure.data(), message, s.current_event->failure.size() - 1);
}
}  // 命名空间 solver::v61hsd
