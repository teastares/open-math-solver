#pragma once
// 仅用于冻结副本；不增加求解、KKT 乘积或生产时钟读取。
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
#include <vector>

namespace solver::v49hsd {
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_reserve = 64ULL * 1024;

struct Field {
  const char* name;
  const char* dtype;
  const void* data;
  std::size_t count, width, offset;
};

struct Context {
  std::array<Field, 12> fields{};
  const std::vector<bool>* inequality = nullptr;
  double tau = 0, kappa = 0, primal_regularization = 0, dual_regularization = 0;
  std::int64_t completed_iterations = 0, factorizations = 0, refinements = 0;
  std::int64_t constant_solves = 0, direction_solves = 0;
  int level = 0;
};

struct Snapshot {
  std::unique_ptr<unsigned char[]> payload;
  std::size_t bytes = 0;
  Context context;
  long double gap_rhs = 0, complementarity_rhs = 0, dot_u = 0, dot_v = 0, denominator = 0;
  bool scalar_values_available = false;
  std::array<char, 1024> failure{};
};

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
  // 十六进制有效位覆盖 long double 的所有二进制位；由离线有理数解析，不经 double。
  out << '"' << std::hexfloat << std::setprecision(std::numeric_limits<long double>::max_digits10)
      << value << std::defaultfloat << '"';
}

struct State {
  const char* prefix = std::getenv("SOLVER_V49_DIAGNOSTICS");
  bool enabled = prefix && *prefix;
  bool attempted = false;
  const char* status = "no_scalar_failure";
  const char* current_phase = "unknown";
  std::int64_t iteration = 0;
  Context* context = nullptr;
  std::unique_ptr<Snapshot> snapshot;

  ~State() noexcept {
    if (!enabled) return;
    try { write(); }
    catch (...) { std::fputs("V49_HSD {\"write_error\":true}\n", stderr); }
  }

  void write() const {
    const std::string base(prefix);
    bool binary_written = false;
    if (snapshot) {
      std::ofstream binary(base + ".hsd.bin", std::ios::binary | std::ios::trunc);
      const char magic[8] = {'V', '4', '9', 'H', 'S', 'D', '1', '\0'};
      binary.write(magic, 8);
      binary.write(reinterpret_cast<const char*>(snapshot->payload.get()), static_cast<std::streamsize>(snapshot->bytes));
      binary.flush();
      binary_written = static_cast<bool>(binary);
    }
    std::ofstream out(base + ".hsd.json", std::ios::trunc);
    out.imbue(std::locale::classic());
    out << "{\"schema\":\"v49_hsd_1\",\"enabled\":true,\"attempted\":" << (attempted ? "true" : "false")
        << ",\"captured\":" << (snapshot && binary_written ? "true" : "false") << ",\"status\":";
    quoted(out, snapshot && !binary_written ? "binary_write_failed" : status);
    out << ",\"component_cache_limit\":" << cache_limit << ",\"metadata_reserve\":" << metadata_reserve
        << ",\"payload_bytes\":" << (snapshot ? snapshot->bytes : 0)
        << ",\"extra_lp_products\":0,\"extra_lp_solves\":0,\"event_count\":0"
        << ",\"long_double_digits\":" << std::numeric_limits<long double>::digits
        << ",\"long_double_max_exponent\":" << std::numeric_limits<long double>::max_exponent
        << ",\"long_double_min_exponent\":" << std::numeric_limits<long double>::min_exponent;
    if (snapshot) {
      const auto& s = *snapshot;
      const auto& c = s.context;
      out << ",\"phase\":"; quoted(out, current_phase);
      out << ",\"iteration\":" << iteration << ",\"regularization_level\":" << c.level
          << ",\"completed_iterations\":" << c.completed_iterations
          << ",\"factorizations\":" << c.factorizations << ",\"refinements\":" << c.refinements
          << ",\"constant_solves\":" << c.constant_solves << ",\"direction_solves\":" << c.direction_solves;
      out << ",\"failure\":"; quoted(out, s.failure.data());
      out << ",\"scalar_values_available\":" << (s.scalar_values_available ? "true" : "false");
      out << ",\"scalars\":{\"tau\":"; real_hex(out, c.tau);
      out << ",\"kappa\":"; real_hex(out, c.kappa);
      out << ",\"primal_regularization\":"; real_hex(out, c.primal_regularization);
      out << ",\"dual_regularization\":"; real_hex(out, c.dual_regularization);
      out << ",\"gap_rhs\":"; real_hex(out, s.gap_rhs);
      out << ",\"complementarity_rhs\":"; real_hex(out, s.complementarity_rhs);
      out << ",\"dot_u\":"; real_hex(out, s.dot_u);
      out << ",\"dot_v\":"; real_hex(out, s.dot_v);
      out << ",\"denominator\":"; real_hex(out, s.denominator);
      out << "},\"arrays\":[";
      for (std::size_t i = 0; i < c.fields.size(); ++i) {
        if (i) out << ',';
        const auto& f = c.fields[i];
        out << "{\"name\":"; quoted(out, f.name);
        out << ",\"dtype\":"; quoted(out, f.dtype);
        out << ",\"count\":" << f.count << ",\"offset\":" << f.offset + 8 << '}';
      }
      out << ']';
    }
    out << "}\n";
    out.flush();
    std::fprintf(stderr, "V49_HSD {\"captured\":%s,\"write_error\":%s}\n",
        snapshot && binary_written ? "true" : "false", out && (!snapshot || binary_written) ? "false" : "true");
  }
};

inline State& state() noexcept { static State result; return result; }
inline void begin_run() noexcept { (void)state(); }
inline void phase(const char* value, std::int64_t iteration) noexcept {
  auto& s = state();
  if (!s.enabled || s.attempted) return;
  s.current_phase = value;
  s.iteration = iteration;
}

class Scope {
  std::optional<Context> value_;
  Context* previous_ = nullptr;
  bool active_ = false;
 public:
  template<class Model, class Point, class Result>
  Scope(const Model& model, const Point& point, std::span<const double> diagonal,
      std::span<const double> regularized, int level, std::span<const double> u,
      std::span<const double> v, double tau, double kappa, const Result& result) noexcept {
    auto& s = state();
    if (!s.enabled || s.attempted) return;
    auto& value = value_.emplace();
    auto field = [](const char* name, const char* dtype, const auto& values) {
      return Field{name, dtype, values.data(), values.size(), sizeof(values[0]), 0};
    };
    value.fields = {field("starts", "i64_le", model.starts), field("rows", "i64_le", model.rows),
        field("values", "f64_le", model.values), field("costs", "f64_le", model.costs),
        field("rhs", "f64_le", model.rhs), field("s", "f64_le", point.s), field("z", "f64_le", point.z),
        field("diagonal_h", "f64_le", diagonal), field("regularized_h", "f64_le", regularized),
        field("u", "f64_le", u), field("v", "f64_le", v),
        Field{"inequality", "u8", nullptr, model.inequality.size(), 1, 0}};
    value.inequality = &model.inequality;
    value.tau = tau; value.kappa = kappa; value.level = level;
    value.primal_regularization = result.ipm_primal_regularization.value_or(0);
    value.dual_regularization = result.ipm_dual_regularization.value_or(0);
    value.completed_iterations = result.ipm_iterations.value_or(0);
    value.factorizations = result.ipm_factorizations.value_or(0);
    value.refinements = result.ipm_refinement_steps.value_or(0);
    value.constant_solves = result.ipm_homogeneous_constant_solves.value_or(0);
    value.direction_solves = result.ipm_homogeneous_direction_solves.value_or(0);
    previous_ = s.context; s.context = &value; active_ = true;
  }
  ~Scope() noexcept { if (active_) state().context = previous_; }
};

class ScalarObservation {
  bool active_ = false, have_values_ = false;
  long double gap_ = 0, complementarity_ = 0, dot_u_ = 0, dot_v_ = 0, denominator_ = 0;
 public:
  ScalarObservation(long double gap, long double complementarity) noexcept {
    auto& s = state();
    if (!s.enabled || s.attempted) return;
    active_ = true; gap_ = gap; complementarity_ = complementarity;
  }
  void values(long double u, long double v, long double denominator) noexcept {
    if (!active_) return;
    dot_u_ = u; dot_v_ = v; denominator_ = denominator; have_values_ = true;
  }
  void failure(const char* message) noexcept {
    if (!active_) return;
    auto& s = state();
    if (s.attempted) return;
    s.attempted = true;
    if (!s.context) { s.status = "missing_context"; return; }
    if constexpr (std::endian::native != std::endian::little || sizeof(double) != 8 ||
        !std::numeric_limits<double>::is_iec559 || std::numeric_limits<double>::digits != 53) {
      s.status = "unsupported_binary_format"; return;
    }
    try {
      std::size_t bytes = 0;
      for (const auto& f : s.context->fields) {
        if (f.count > (cache_limit - metadata_reserve - bytes) / f.width) {
          s.status = "cache_limit"; return;
        }
        bytes += f.count * f.width;
      }
      auto snapshot = std::make_unique<Snapshot>();
      snapshot->payload = std::make_unique<unsigned char[]>(bytes);
      snapshot->bytes = bytes; snapshot->context = *s.context;
      snapshot->gap_rhs = gap_; snapshot->complementarity_rhs = complementarity_;
      snapshot->dot_u = dot_u_; snapshot->dot_v = dot_v_; snapshot->denominator = denominator_;
      snapshot->scalar_values_available = have_values_;
      std::strncpy(snapshot->failure.data(), message, snapshot->failure.size() - 1);
      std::size_t offset = 0;
      for (auto& f : snapshot->context.fields) {
        f.offset = offset;
        if (f.width == 1) {
          for (std::size_t i = 0; i < f.count; ++i) snapshot->payload[offset + i] = (*s.context->inequality)[i] ? 1 : 0;
        } else if (f.count) std::memcpy(snapshot->payload.get() + offset, f.data, f.count * f.width);
        offset += f.count * f.width;
        f.data = nullptr;
      }
      snapshot->context.inequality = nullptr;
      s.snapshot = std::move(snapshot);
      s.status = "captured";
    } catch (...) { s.status = "capture_exception"; }
  }
};
}  // namespace solver::v49hsd
