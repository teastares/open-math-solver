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

namespace solver::v63hsd {
inline constexpr std::size_t event_limit = 8;
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_reserve = 64ULL * 1024;
inline constexpr std::size_t step_limit = 256;

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
  std::int64_t scalar_checks = 0, scalar_rejections = 0;
};

struct Step {
  const char* kind = "unknown";
  int ir_iteration = -1, restart = -1, column = -1;
  double global_norm = 0, global_tolerance = 0;
  std::array<double, 2> block_norms{}, block_tolerances{};
  bool block_tolerances_available = false;
  bool scalar_values_available = false, scalar_return_available = false, scalar_accepted = false;
  bool scalar_accounted = false;
  long double dot_v = 0, denominator = 0;
};

struct Cycle {
  bool started = false, ended = false;
  std::size_t columns_started = 0, columns_completed = 0, remaining_zero_count = 0;
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
  std::unique_ptr<long double[]> last_linear_candidate;
  std::unique_ptr<double[]> last_linear_residual, linear_rhs;
  std::unique_ptr<Step[]> small_events;
  std::array<Cycle, 3> cycles{};
  std::size_t step_count = 0, step_total = 0, step_dropped = 0;
  std::int64_t last_candidate_step_index = -1;
  bool last_linear_available = false, linear_rhs_available = false;
  const char* linear_exit = "not_started";
  int current_ir = -1, current_restart = -1, current_column = -1;
  std::size_t bytes = 0, dimension = 0, index = 0;
  std::size_t repair_ordinal = 0;
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
      << ",\"extended_krylov_successes\":" << c.extended_krylov_successes
      << ",\"scalar_checks\":" << c.scalar_checks << ",\"scalar_rejections\":" << c.scalar_rejections << '}';
}

inline void write_scalar_values(std::ostream& out, const ScalarValues& v) {
  out << "{\"dot_u\":"; real_hex(out, v.dot_u);
  out << ",\"dot_v\":"; real_hex(out, v.dot_v);
  out << ",\"denominator\":"; real_hex(out, v.denominator); out << '}';
}

template<class Scalar>
inline void write_wide(std::ostream& out, const Scalar* values, std::size_t count) {
  out << '[';
  for (std::size_t i = 0; i < count; ++i) {
    if (i) out << ',';
    real_hex(out, values[i]);
  }
  out << ']';
}

inline void write_step(std::ostream& out, const Step& step) {
  out << "{\"kind\":"; quoted(out, step.kind);
  out << ",\"ir_iteration\":" << step.ir_iteration << ",\"restart\":" << step.restart
      << ",\"column\":" << step.column << ",\"global_norm\":"; real_hex(out, step.global_norm);
  out << ",\"global_tolerance\":"; real_hex(out, step.global_tolerance);
  out << ",\"block_norms\":"; write_wide(out, step.block_norms.data(), 2);
  out << ",\"block_tolerances\":";
  if (step.block_tolerances_available) write_wide(out, step.block_tolerances.data(), 2); else out << "null";
  out << ",\"scalar_values_available\":" << (step.scalar_values_available ? "true" : "false")
      << ",\"scalar_return_available\":" << (step.scalar_return_available ? "true" : "false")
      << ",\"scalar_accepted\":";
  if (step.scalar_return_available) out << (step.scalar_accepted ? "true" : "false"); else out << "null";
  out << ",\"scalar_accounted\":" << (step.scalar_accounted ? "true" : "false") << ",\"dot_v\":";
  if (step.scalar_values_available) real_hex(out, step.dot_v); else out << "null";
  out << ",\"denominator\":";
  if (step.scalar_values_available) real_hex(out, step.denominator); else out << "null";
  out << '}';
}

struct State {
  const char* prefix = std::getenv("SOLVER_V63_DIAGNOSTICS");
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
    catch (...) { std::fputs("V63_HSD {\"write_error\":true,\"status\":\"write_exception\"}\n", stderr); }
  }

  void write() const {
    const std::string base(prefix);
    std::ofstream binary(base + ".hsd.bin", std::ios::binary | std::ios::trunc);
    const char magic[8] = {'V', '6', '3', 'H', 'S', 'D', '1', '\0'};
    binary.write(magic, 8);
    for (std::size_t i = 0; i < count; ++i)
      binary.write(reinterpret_cast<const char*>(events[i]->payload.get()), static_cast<std::streamsize>(events[i]->bytes));
    binary.flush();
    const bool binary_written = static_cast<bool>(binary);
    std::ofstream out(base + ".hsd.json", std::ios::trunc);
    out.imbue(std::locale::classic());
    out << "{\"schema\":\"v63_hsd_1\",\"enabled\":true,\"status\":";
    quoted(out, binary_written ? status : "binary_write_failed");
    out << ",\"event_count\":" << count << ",\"trigger_count\":" << trigger_count
        << ",\"dropped_events\":" << dropped_events << ",\"payload_bytes\":" << payload_bytes
        << ",\"cache_bytes\":" << cache_bytes << ",\"limits\":{\"max_events\":" << event_limit
        << ",\"payload_limit_bytes\":" << cache_limit << ",\"metadata_reserve_bytes\":" << metadata_reserve
        << ",\"max_steps_per_event\":" << step_limit << "},\"step_bytes\":" << sizeof(Step)
        << ",\"extra_lp_products\":0,\"extra_lp_solves\":0,\"long_double_digits\":"
        << std::numeric_limits<long double>::digits << ",\"long_double_bytes\":" << sizeof(long double)
        << ",\"long_double_max_exponent\":" << std::numeric_limits<long double>::max_exponent
        << ",\"long_double_min_exponent\":" << std::numeric_limits<long double>::min_exponent << ",\"events\":[";
    for (std::size_t i = 0; i < count; ++i) {
      if (i) out << ',';
      const auto& e = *events[i];
      const auto& c = e.context;
      out << "{\"index\":" << e.index << ",\"repair_ordinal\":" << e.repair_ordinal
          << ",\"iteration\":" << e.iteration << ",\"phase\":";
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
      out << ",\"last_linear_available\":" << (e.last_linear_available ? "true" : "false")
          << ",\"linear_rhs_available\":" << (e.linear_rhs_available ? "true" : "false")
          << ",\"step_capacity\":" << step_limit << ",\"step_count\":" << e.step_count
          << ",\"step_total\":" << e.step_total << ",\"step_dropped\":" << e.step_dropped
          << ",\"last_candidate_step_index\":" << e.last_candidate_step_index << ",\"linear_exit\":";
      quoted(out, e.linear_exit);
      out << ",\"linear_position\":{\"ir_iteration\":" << e.current_ir
          << ",\"restart\":" << e.current_restart << ",\"column\":" << e.current_column << '}'
          << ",\"last_linear_residual\":";
      if (e.last_linear_available) write_wide(out, e.last_linear_residual.get(), e.dimension); else out << "null";
      out << ",\"linear_rhs\":";
      if (e.linear_rhs_available) write_wide(out, e.linear_rhs.get(), e.dimension); else out << "null";
      out << ",\"cycles\":[";
      for (std::size_t j = 0; j < e.cycles.size(); ++j) {
        if (j) out << ',';
        const auto& cycle = e.cycles[j];
        out << "{\"started\":" << (cycle.started ? "true" : "false")
            << ",\"ended\":" << (cycle.ended ? "true" : "false")
            << ",\"columns_started\":" << cycle.columns_started << ",\"columns_completed\":" << cycle.columns_completed
            << ",\"remaining_zero_count\":" << cycle.remaining_zero_count << '}';
      }
      out << "],\"small_events\":[";
      for (std::size_t j = 0; j < e.step_count; ++j) { if (j) out << ','; write_step(out, e.small_events[j]); }
      out << ']';
      out << ",\"wide\":{\"u\":"; write_wide(out, e.u.get(), e.dimension);
      out << ",\"before_v\":"; write_wide(out, e.before_v.get(), e.dimension);
      out << ",\"after_v\":";
      if (e.candidate_available) write_wide(out, e.after_v.get(), e.dimension); else out << "null";
      out << ",\"last_linear_candidate\":";
      if (e.last_linear_available) write_wide(out, e.last_linear_candidate.get(), e.dimension); else out << "null";
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
    std::fprintf(stderr, "V63_HSD {\"write_error\":%s,\"event_count\":%llu,\"status\":\"%s\"}\n",
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
          r.ipm_extended_krylov_attempts.value_or(0), r.ipm_extended_krylov_successes.value_or(0),
          r.ipm_homogeneous_scalar_checks.value_or(0), r.ipm_homogeneous_scalar_rejections.value_or(0)};
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
      if (s.cache_bytes > cache_limit - metadata_reserve) { s.status = "cache_limit"; return; }
      const auto remaining = cache_limit - metadata_reserve - s.cache_bytes;
      const auto stage_bytes = step_limit * sizeof(Step);
      if (remaining < stage_bytes || u.size() > (remaining - stage_bytes) /
          (4 * sizeof(long double) + 2 * sizeof(double))) { s.status = "cache_limit"; return; }
      const auto wide_bytes = u.size() * (4 * sizeof(long double) + 2 * sizeof(double)) + stage_bytes;
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
      event->last_linear_candidate = std::make_unique<long double[]>(u.size());
      event->last_linear_residual = std::make_unique<double[]>(u.size());
      event->linear_rhs = std::make_unique<double[]>(u.size());
      event->small_events = std::make_unique<Step[]>(step_limit);
      event->bytes = bytes; event->dimension = u.size(); event->index = s.count;
      event->repair_ordinal = s.trigger_count;
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

inline Event* active_event() noexcept {
  auto& s = state();
  return s.enabled ? s.current_event : nullptr;
}

inline Step* active_step() noexcept {
  auto* e = active_event();
  if (!e || e->last_candidate_step_index < 0 ||
      static_cast<std::size_t>(e->last_candidate_step_index) >= e->step_count) return nullptr;
  return &e->small_events[static_cast<std::size_t>(e->last_candidate_step_index)];
}

inline void linear_begin(const std::vector<double>& rhs, bool enabled) noexcept {
  if (!enabled) return;
  auto* e = active_event();
  if (!e) return;
  if (rhs.size() != e->dimension) { state().status = "invalid_linear_dimension"; return; }
  std::copy(rhs.begin(), rhs.end(), e->linear_rhs.get());
  e->linear_rhs_available = true;
  e->linear_exit = "searching";
}

template<class Scalar>
inline void linear_candidate(const char* kind, int ir, int restart, int column,
    const std::vector<Scalar>& values, const std::vector<double>& residual,
    double tolerance, const std::array<double, 2>* block_tolerances, std::size_t columns) noexcept {
  auto* e = active_event();
  if (!e || std::strcmp(e->linear_exit, "searching") != 0) return;
  if (values.size() != e->dimension || residual.size() != e->dimension || columns > e->dimension) {
    state().status = "invalid_linear_dimension"; return;
  }
  e->current_ir = ir; e->current_restart = restart; e->current_column = column;
  std::copy(values.begin(), values.end(), e->last_linear_candidate.get());
  std::copy(residual.begin(), residual.end(), e->last_linear_residual.get());
  e->last_linear_available = true;
  e->last_candidate_step_index = static_cast<std::int64_t>(e->step_total++);
  if (e->step_count == step_limit) { ++e->step_dropped; state().status = "step_limit"; return; }
  auto& step = e->small_events[e->step_count++];
  step.kind = kind; step.ir_iteration = ir; step.restart = restart; step.column = column;
  step.global_tolerance = tolerance;
  step.block_tolerances_available = block_tolerances != nullptr;
  if (block_tolerances) step.block_tolerances = *block_tolerances;
  // 只扫描已有残差作摘要；不形成新的 K0 乘积，成本计入诊断求解时间。
  for (std::size_t i = 0; i < residual.size(); ++i) {
    const auto value = std::abs(residual[i]);
    step.global_norm = std::max(step.global_norm, value);
    step.block_norms[i < columns ? 0 : 1] = std::max(step.block_norms[i < columns ? 0 : 1], value);
  }
}

inline void application_values(long double dot_v, long double denominator) noexcept {
  auto* step = active_step();
  if (!step) return;
  if (step->scalar_values_available) { state().status = "duplicate_scalar_values"; return; }
  step->dot_v = dot_v; step->denominator = denominator; step->scalar_values_available = true;
}
inline void application_return(bool accepted) noexcept {
  auto* step = active_step();
  if (!step) return;
  step->scalar_return_available = true; step->scalar_accepted = accepted;
}
inline void application_accounted() noexcept {
  if (auto* step = active_step()) step->scalar_accounted = true;
}
inline void cycle_begin(int restart) noexcept {
  auto* e = active_event();
  if (!e || std::strcmp(e->linear_exit, "searching") != 0) return;
  if (restart < 0 || restart >= 3) { state().status = "invalid_cycle"; return; }
  e->cycles[restart].started = true;
  e->current_ir = -1; e->current_restart = restart; e->current_column = -1;
}
inline void column_begin(int restart, int column) noexcept {
  auto* e = active_event();
  if (!e || std::strcmp(e->linear_exit, "searching") != 0) return;
  if (restart < 0 || restart >= 3) { state().status = "invalid_cycle"; return; }
  ++e->cycles[restart].columns_started; e->current_column = column;
}
inline void column_complete(int restart, double remaining) noexcept {
  auto* e = active_event();
  if (!e || std::strcmp(e->linear_exit, "searching") != 0) return;
  if (restart < 0 || restart >= 3) { state().status = "invalid_cycle"; return; }
  ++e->cycles[restart].columns_completed;
  if (remaining == 0) ++e->cycles[restart].remaining_zero_count;
}
inline void cycle_end(int restart) noexcept {
  auto* e = active_event();
  if (!e || std::strcmp(e->linear_exit, "searching") != 0) return;
  if (restart < 0 || restart >= 3) { state().status = "invalid_cycle"; return; }
  e->cycles[restart].ended = true;
}
inline void linear_exit(const char* reason) noexcept {
  auto* e = active_event();
  if (e && std::strcmp(e->linear_exit, "searching") == 0) e->linear_exit = reason;
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
  if (s.enabled && s.current_event) {
    std::strncpy(s.current_event->failure.data(), message, s.current_event->failure.size() - 1);
    if (std::strcmp(s.current_event->linear_exit, "searching") == 0)
      s.current_event->linear_exit = "exception";
    else if (!s.current_event->candidate_available &&
        (std::strcmp(s.current_event->linear_exit, "ir_return") == 0 ||
         std::strcmp(s.current_event->linear_exit, "krylov_return") == 0))
      s.current_event->linear_exit = "exception_after_acceptance";
  }
}
}  // 命名空间 solver::v63hsd
