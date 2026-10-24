#pragma once
// 冻结副本专用：仅复制已有量，所有新增矩阵乘积和误差分解均留给离线工具。
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
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace solver::v53hsd {
inline constexpr std::size_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t metadata_reserve = 64ULL * 1024;
inline constexpr std::size_t static_field_count = 20, step_field_count = 13;

struct Field {
  const char* name = "";
  const char* dtype = "";
  std::size_t count = 0, width = 0, offset = 0;
};
struct Errors {
  std::array<std::array<char, 192>, 8> sample{};
  std::size_t count = 0, saved = 0;
  bool truncated = false;
  template<class Source> void take(const Source& source) noexcept {
    count = source.size(); saved = std::min(count, sample.size()); truncated = count > saved;
    for (std::size_t i = 0; i < saved; ++i) {
      auto length = std::min(source[i].size(), sample[i].size() - 1);
      // 截断样本时保留完整 UTF-8 字符，避免写出不可解码的 JSON。
      while (length < source[i].size() && length > 0 &&
          (static_cast<unsigned char>(source[i][length]) & 0xc0) == 0x80) --length;
      std::memcpy(sample[i].data(), source[i].data(), length);
      sample[i][length] = 0;
      truncated = truncated || length < source[i].size();
    }
  }
};
struct Checks {
  bool primal_valid = false, feasible = false, objective_consistent = false;
  bool dual_valid = false, dual_feasible = false, gap_satisfied = false, optimal = false;
  double max_row = 0, max_bound = 0, max_integrality = 0, max_dual = 0;
  std::optional<double> objective, dual_objective, corrected_bound, gap, relative_gap;
  Errors primal_errors, dual_errors;
  template<class Source> void take(const Source& source) noexcept {
    const auto& p = source.primal; const auto& d = source.optimality;
    primal_valid = p.valid_input; feasible = p.feasible; objective_consistent = p.objective_consistent;
    dual_valid = d.valid_input; dual_feasible = d.dual_feasible; gap_satisfied = d.gap_satisfied;
    optimal = d.numerically_optimal; max_row = p.max_row_violation; max_bound = p.max_bound_violation;
    max_integrality = p.max_integrality_violation; max_dual = d.max_dual_residual;
    objective = p.objective; dual_objective = d.dual_objective_estimate;
    corrected_bound = d.corrected_dual_bound; gap = d.objective_gap; relative_gap = d.relative_objective_gap;
    primal_errors.take(p.errors); dual_errors.take(d.errors);
  }
};
struct Assessment {
  std::int64_t id = 0, iteration = 0;
  bool replace = false, available = false;
  Checks checks;
};
struct Slot {
  std::unique_ptr<unsigned char[]> payload;
  std::int64_t iteration = 0, ordinal = 0;
  bool have_input = false, have_u = false, direction_returned = false;
  bool step_complete = false, assessed = false, unwound = false;
  double tau = 0, kappa = 0, theta = 0, dtau = 0, step = 0, next_tau = 0;
  std::int64_t factorizations = 0, refinements = 0, constant_solves = 0, direction_solves = 0;
  std::int64_t constant_repairs = 0, successful_repairs = 0;
  Assessment assessment;
  void reset() noexcept {
    // 仅清元数据；未标有效的数组从不写出，避免额外扫描整个缓存。
    iteration = ordinal = 0;
    have_input = have_u = direction_returned = step_complete = assessed = unwound = false;
    tau = kappa = theta = dtau = step = next_tau = 0;
    factorizations = refinements = constant_solves = direction_solves = constant_repairs = successful_repairs = 0;
    assessment = Assessment{};
  }
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
  // 全部来自实际 binary64 标量；十六进制写出不改变其二进制值。
  out << '"' << std::hexfloat << std::setprecision(std::numeric_limits<double>::max_digits10)
      << value << std::defaultfloat << '"';
}
inline void optional_hex(std::ostream& out, const std::optional<double>& value) {
  if (value) real_hex(out, *value); else out << "null";
}
inline void errors_json(std::ostream& out, const Errors& errors) {
  out << "{\"count\":" << errors.count << ",\"truncated\":" << (errors.truncated ? "true" : "false")
      << ",\"sample\":[";
  for (std::size_t i = 0; i < errors.saved; ++i) { if (i) out << ','; quoted(out, errors.sample[i].data()); }
  out << "]}";
}
inline void assessment_json(std::ostream& out, const Assessment& a) {
  const auto& c = a.checks;
  out << "{\"available\":" << (a.available ? "true" : "false") << ",\"assessment_id\":" << a.id
      << ",\"iteration\":" << a.iteration << ",\"replace\":" << (a.replace ? "true" : "false")
      << ",\"checks\":{\"primal\":{\"valid_input\":" << (c.primal_valid ? "true" : "false")
      << ",\"feasible\":" << (c.feasible ? "true" : "false")
      << ",\"objective_consistent\":" << (c.objective_consistent ? "true" : "false") << ",\"objective\":";
  optional_hex(out, c.objective);
  out << ",\"max_row_violation\":"; real_hex(out, c.max_row);
  out << ",\"max_bound_violation\":"; real_hex(out, c.max_bound);
  out << ",\"max_integrality_violation\":"; real_hex(out, c.max_integrality);
  out << ",\"errors\":"; errors_json(out, c.primal_errors);
  out << "},\"optimality\":{\"valid_input\":" << (c.dual_valid ? "true" : "false")
      << ",\"dual_feasible\":" << (c.dual_feasible ? "true" : "false")
      << ",\"gap_satisfied\":" << (c.gap_satisfied ? "true" : "false")
      << ",\"numerically_optimal\":" << (c.optimal ? "true" : "false") << ",\"max_dual_residual\":";
  real_hex(out, c.max_dual);
  out << ",\"dual_objective_estimate\":"; optional_hex(out, c.dual_objective);
  out << ",\"corrected_dual_bound\":"; optional_hex(out, c.corrected_bound);
  out << ",\"objective_gap\":"; optional_hex(out, c.gap);
  out << ",\"relative_objective_gap\":"; optional_hex(out, c.relative_gap);
  out << ",\"errors\":"; errors_json(out, c.dual_errors);
  out << "}}}";
}

struct State {
  const char* prefix = std::getenv("SOLVER_V53_DIAGNOSTICS");
  bool enabled = prefix && *prefix, active = false, run_started = false, ready = false, run_unwound = false;
  const char* status = "not_entered";
  std::size_t static_bytes = 0, step_bytes = 0, cache_bytes = metadata_reserve;
  std::size_t bytes_copied = 0, completed_count = 0, saved_count = 0;
  std::int64_t direction_iteration = 0, direction_ordinal = 0, assessments = 0;
  std::array<Field, static_field_count> static_fields{};
  std::array<Field, step_field_count> step_fields{};
  std::unique_ptr<unsigned char[]> static_payload;
  std::array<Slot, 4> slots{};
  std::array<std::size_t, 3> saved{0, 1, 2};
  std::size_t pending = 3;
  Assessment retained_assessment, latest_assessment;
  double objective_scale = 1, sense = 1, objective_offset = 0;
  std::array<double, 8> tolerances{};
  std::size_t original_rows = 0, original_columns = 0, n = 0, m = 0;
  ~State() noexcept {
    if (!enabled) return;
    try { write(); }
    catch (...) { std::fputs("V53_HSD {\"write_error\":true}\n", stderr); }
  }

  bool add(Field& field, const char* name, const char* dtype, std::size_t count,
      std::size_t width, std::size_t& bytes) noexcept {
    if (width == 0 || bytes > cache_limit || count > (cache_limit - bytes) / width) return false;
    field = {name, dtype, count, width, bytes}; bytes += count * width; return true;
  }
  void copy(unsigned char* destination, const Field& field, const void* source) noexcept {
    const auto bytes = field.count * field.width;
    if (bytes) std::memcpy(destination + field.offset, source, bytes);
    bytes_copied += bytes;
  }
  template<class Model, class Original, class Options>
  void begin(const Model& model, const Original& original, const Options& options) noexcept {
    if (!enabled) return;
    run_started = true; active = true;
    if (ready) { status = "multiple_runs_unsupported"; active = false; return; }
    if constexpr (std::endian::native != std::endian::little || sizeof(double) != 8 ||
        sizeof(std::size_t) != 8 || !std::numeric_limits<double>::is_iec559) {
      status = "unsupported_binary_format"; return;
    }
    if (std::strlen(prefix) > 4000) { status = "prefix_too_long"; return; }
    if (!model.presolved_bounds.empty() || !model.presolved_fixed_columns.empty()) {
      status = "unsupported_presolve_recovery"; return;
    }
    n = model.costs.size(); m = model.rhs.size();
    original_rows = original.row_lower.size(); original_columns = original.costs.size();
    std::array<const void*, static_field_count> sources{};
    std::size_t index = 0; bool valid = true;
    const auto field = [&](const char* name, const char* dtype, const auto& source) {
      valid = add(static_fields[index], name, dtype, source.size(), sizeof(source[0]), static_bytes) && valid;
      sources[index++] = source.data();
    };
    field("g_starts", "i64_le", model.starts); field("g_rows", "i64_le", model.rows);
    field("g_values", "f64_le", model.values); field("costs", "f64_le", model.costs);
    field("b", "f64_le", model.rhs);
    valid = add(static_fields[index], "inequality", "u8", model.inequality.size(), 1, static_bytes) && valid; ++index;
    field("original_row", "i64_le", model.original_row); field("dual_sign", "f64_le", model.dual_sign);
    field("row_scale", "f64_le", model.row_scale); field("column_scale", "f64_le", model.column_scale);
    field("original_columns", "u64_le", model.original_columns); field("fixed_values", "f64_le", model.fixed_values);
    field("a_starts", "i64_le", original.column_start); field("a_rows", "i64_le", original.row_index);
    field("a_values", "f64_le", original.values); field("original_costs", "f64_le", original.costs);
    field("column_lower", "f64_le", original.column_lower); field("column_upper", "f64_le", original.column_upper);
    field("row_lower", "f64_le", original.row_lower); field("row_upper", "f64_le", original.row_upper);
    constexpr std::array<const char*, step_field_count> names{"old_x", "old_z", "new_x", "new_z", "stored_rd",
        "rhs_u", "diagonal_h", "u", "v", "dz", "extracted_z", "assess_x", "assess_y"};
    const std::array<std::size_t, step_field_count> counts{n, m, n, m, n, n + m, m, n + m, n + m, m, m,
        original_columns, original_rows};
    for (std::size_t i = 0; i < names.size(); ++i)
      valid = add(step_fields[i], names[i], "f64_le", counts[i], 8, step_bytes) && valid;
    if (!valid || static_bytes > cache_limit - metadata_reserve ||
        step_bytes > (cache_limit - metadata_reserve - static_bytes) / slots.size()) {
      status = "cache_limit"; return;
    }
    cache_bytes = metadata_reserve + static_bytes + slots.size() * step_bytes;
    try {
      // 不值初始化整块，防止将分配容量误算为实际复制量。
      static_payload.reset(new unsigned char[static_bytes]);
      for (auto& slot : slots) slot.payload.reset(new unsigned char[step_bytes]);
      for (std::size_t i = 0; i < static_fields.size(); ++i) {
        if (i == 5) {
          for (std::size_t j = 0; j < model.inequality.size(); ++j)
            static_payload[static_fields[i].offset + j] = model.inequality[j] ? 1 : 0;
          bytes_copied += model.inequality.size();
        } else copy(static_payload.get(), static_fields[i], sources[i]);
      }
      objective_scale = model.objective_scale; sense = static_cast<int>(original.sense) == 0 ? 1 : -1;
      objective_offset = original.objective_offset;
      tolerances = {options.tolerances.feasibility_absolute, options.tolerances.feasibility_relative,
          options.tolerances.objective_absolute, options.tolerances.objective_relative,
          options.pdlp.optimality.dual_absolute, options.pdlp.optimality.dual_relative,
          options.pdlp.optimality.gap_absolute, options.pdlp.optimality.gap_relative};
      ready = true; status = "captured";
    } catch (const std::bad_alloc&) { status = "memory_allocation_failed"; }
    catch (...) { status = "capture_exception"; }
  }
  void copy_step(Slot& slot, std::size_t field, std::span<const double> source) noexcept {
    if (source.size() != step_fields[field].count) {
      ready = false; status = "dimension_mismatch"; return;
    }
    copy(slot.payload.get(), step_fields[field], source.data());
  }
  template<class Result> void counters(Slot& slot, const Result& result) noexcept {
    slot.factorizations = result.ipm_factorizations.value_or(0);
    slot.refinements = result.ipm_refinement_steps.value_or(0);
    slot.constant_solves = result.ipm_homogeneous_constant_solves.value_or(0);
    slot.direction_solves = result.ipm_homogeneous_direction_solves.value_or(0);
    slot.constant_repairs = result.ipm_homogeneous_constant_repairs.value_or(0);
    slot.successful_repairs = result.ipm_homogeneous_constant_repair_successes.value_or(0);
  }
  template<class Point, class Result>
  void input(const Point& point, std::span<const double> rd, std::span<const double> rhs,
      std::span<const double> diagonal, double tau, double kappa, double theta, const Result& result) noexcept {
    if (!active || !ready) return;
    auto& slot = slots[pending]; slot.reset();
    const auto iteration = result.ipm_iterations.value_or(0) + 1;
    if (direction_iteration != iteration) { direction_iteration = iteration; direction_ordinal = 0; }
    slot.iteration = iteration; slot.ordinal = ++direction_ordinal;
    slot.tau = tau; slot.kappa = kappa; slot.theta = theta; counters(slot, result);
    copy_step(slot, 0, point.x); copy_step(slot, 1, point.z); copy_step(slot, 4, rd);
    copy_step(slot, 5, rhs); copy_step(slot, 6, diagonal);
    slot.have_input = ready;
  }
  template<class Direction, class Result>
  void returned(const Direction& direction, std::span<const double> constant, const Result& result) noexcept {
    if (!active || !ready) return;
    auto& slot = slots[pending]; copy_step(slot, 8, constant); copy_step(slot, 9, direction.point.z);
    slot.dtau = direction.scalars.tau; counters(slot, result); slot.direction_returned = ready;
  }
  template<class Point>
  void commit(const Point& candidate, double next_tau, double step, std::int64_t iteration) noexcept {
    if (!active || !ready) return;
    auto& slot = slots[pending];
    if (!slot.direction_returned || slot.ordinal != 2 || slot.iteration != iteration) {
      ready = false; status = "direction_sequence_mismatch"; return;
    }
    copy_step(slot, 2, candidate.x); copy_step(slot, 3, candidate.z);
    slot.next_tau = next_tau; slot.step = step; slot.step_complete = ready;
    ++completed_count;
    if (saved_count < saved.size()) { std::swap(pending, saved[saved_count++]); }
    else {
      const auto recycled = saved[1]; saved[1] = saved[2]; saved[2] = pending; pending = recycled;
    }
    slots[pending].reset();
  }
  template<class Point, class ChecksSource, class Result>
  void assess(const Point& extracted, std::span<const double> x, std::span<const double> y,
      const ChecksSource& checks, bool replace, const Result& result) noexcept {
    if (!active || !ready) return;
    latest_assessment.id = ++assessments; latest_assessment.iteration = result.ipm_iterations.value_or(0);
    latest_assessment.available = true; latest_assessment.replace = replace; latest_assessment.checks.take(checks);
    if (replace) retained_assessment = latest_assessment;
    if (!saved_count) return;
    auto& slot = slots[saved[saved_count - 1]];
    if (slot.iteration != latest_assessment.iteration) return;
    copy_step(slot, 10, extracted.z); copy_step(slot, 11, x); copy_step(slot, 12, y);
    slot.assessment = latest_assessment; slot.assessed = ready;
  }

  void field_json(std::ostream& out, const Field& f, std::size_t offset) const {
    out << "{\"name\":"; quoted(out, f.name); out << ",\"dtype\":"; quoted(out, f.dtype);
    out << ",\"count\":" << f.count << ",\"offset\":" << offset << '}';
  }
  bool field_valid(const Slot& slot, std::size_t i) const noexcept {
    if (i == 2 || i == 3) return slot.step_complete;
    if (i == 7) return slot.have_u;
    if (i == 8 || i == 9) return slot.direction_returned;
    if (i >= 10) return slot.assessed;
    return slot.have_input;
  }
  void slot_json(std::ostream& out, std::ostream& binary, const Slot& slot, std::size_t& offset,
      bool include_arrays = true) const {
    out << "{\"iteration\":" << slot.iteration << ",\"ordinal\":" << slot.ordinal << ",\"phase\":";
    quoted(out, slot.ordinal == 1 ? "affine" : slot.ordinal == 2 ? "corrector" : "unknown");
    out << ",\"flags\":{\"have_input\":" << (slot.have_input ? "true" : "false")
        << ",\"have_u\":" << (slot.have_u ? "true" : "false")
        << ",\"direction_returned\":" << (slot.direction_returned ? "true" : "false")
        << ",\"step_complete\":" << (slot.step_complete ? "true" : "false")
        << ",\"assessed\":" << (slot.assessed ? "true" : "false")
        << ",\"unwound\":" << (slot.unwound ? "true" : "false") << "},\"scalars\":{\"tau\":";
    real_hex(out, slot.tau); out << ",\"kappa\":"; real_hex(out, slot.kappa);
    out << ",\"theta\":"; real_hex(out, slot.theta); out << ",\"dtau\":"; real_hex(out, slot.dtau);
    out << ",\"step\":"; real_hex(out, slot.step); out << ",\"next_tau\":"; real_hex(out, slot.next_tau);
    out << "},\"counters\":{\"factorizations\":" << slot.factorizations << ",\"refinements\":" << slot.refinements
        << ",\"constant_solves\":" << slot.constant_solves << ",\"direction_solves\":" << slot.direction_solves
        << ",\"constant_repairs\":" << slot.constant_repairs << ",\"successful_repairs\":" << slot.successful_repairs
        << "},\"assessment\":"; assessment_json(out, slot.assessment);
    out << ",\"arrays\":["; bool first = true;
    for (std::size_t i = 0; include_arrays && i < step_fields.size(); ++i) {
      if (!field_valid(slot, i)) continue;
      if (!first) out << ',';
      first = false;
      const auto& field = step_fields[i]; field_json(out, field, offset);
      const auto bytes = field.count * field.width;
      binary.write(reinterpret_cast<const char*>(slot.payload.get() + field.offset), static_cast<std::streamsize>(bytes));
      offset += bytes;
    }
    out << "]}";
  }
  void write() const {
    // 路径和描述符均固定上限；64 KiB 元数据预留覆盖 State、局部描述符与路径暂存。
    std::array<char, 4096> json_path{}, binary_path{};
    if (std::strlen(prefix) > 4000) { std::fputs("V53_HSD {\"write_error\":true,\"reason\":\"prefix_too_long\"}\n", stderr); return; }
    std::snprintf(json_path.data(), json_path.size(), "%s-hsd.json", prefix);
    std::snprintf(binary_path.data(), binary_path.size(), "%s-hsd.bin", prefix);
    std::ofstream binary(binary_path.data(), std::ios::binary | std::ios::trunc);
    std::ofstream out(json_path.data(), std::ios::trunc); out.imbue(std::locale::classic());
    const char magic[8] = {'V', '5', '3', 'H', 'S', 'D', '1', '\0'}; binary.write(magic, 8);
    std::size_t offset = 8;
    out << "{\"schema\":\"v53_hsd_1\",\"enabled\":true,\"run_started\":" << (run_started ? "true" : "false")
        << ",\"ready\":" << (ready ? "true" : "false") << ",\"status\":"; quoted(out, status);
    out << ",\"run_unwound\":" << (run_unwound ? "true" : "false") << ",\"cache_limit\":" << cache_limit
        << ",\"metadata_reserve\":" << metadata_reserve << ",\"cache_bytes\":" << cache_bytes
        << ",\"static_payload_bytes\":" << static_bytes << ",\"step_slot_bytes\":" << step_bytes
        << ",\"allocated_step_slots\":4,\"copied_bytes\":" << bytes_copied
        << ",\"completed_count\":" << completed_count << ",\"assessment_count\":" << assessments
        << ",\"extra_lp_products\":0,\"extra_lp_solves\":0,\"diagnostic_clock_reads\":0"
        << ",\"static\":{\"scalars\":{\"objective_scale\":"; real_hex(out, objective_scale);
    out << ",\"sense\":"; real_hex(out, sense); out << ",\"objective_offset\":"; real_hex(out, objective_offset);
    constexpr std::array<const char*, 8> tolerance_names{"feasibility_absolute", "feasibility_relative",
        "objective_absolute", "objective_relative", "dual_absolute", "dual_relative", "gap_absolute", "gap_relative"};
    for (std::size_t i = 0; i < tolerance_names.size(); ++i) { out << ','; quoted(out, tolerance_names[i]); out << ':'; real_hex(out, tolerances[i]); }
    out << "},\"arrays\":[";
    if (ready) for (std::size_t i = 0; i < static_fields.size(); ++i) {
      if (i) out << ',';
      const auto& field = static_fields[i]; field_json(out, field, offset);
      const auto bytes = field.count * field.width;
      binary.write(reinterpret_cast<const char*>(static_payload.get() + field.offset), static_cast<std::streamsize>(bytes)); offset += bytes;
    }
    out << "]},\"steps\":[";
    if (ready) for (std::size_t i = 0; i < saved_count; ++i) { if (i) out << ','; slot_json(out, binary, slots[saved[i]], offset); }
    out << "],\"last_attempt\":";
    if (ready && slots[pending].have_input) slot_json(out, binary, slots[pending], offset, false); else out << "null";
    out << ",\"latest_assessment\":"; assessment_json(out, latest_assessment);
    out << ",\"retained_assessment\":"; assessment_json(out, retained_assessment);
    binary.flush(); const bool binary_ok = static_cast<bool>(binary);
    out << ",\"binary_bytes\":" << offset << ",\"binary_written\":" << (binary_ok ? "true" : "false") << "}\n";
    out.flush();
    std::fprintf(stderr, "V53_HSD {\"write_error\":%s,\"completed_snapshots\":%zu}\n", out && binary_ok ? "false" : "true", saved_count);
  }
};
static_assert(sizeof(State) + 16384 < metadata_reserve, "诊断元数据必须落在总缓存预留内");
inline State& state() noexcept { static State value; return value; }

class RunScope {
  bool enabled_ = false; int exceptions_ = 0;
 public:
  template<class Model, class Original, class Options>
  RunScope(const Model& model, const Original& original, const Options& options) noexcept {
    auto& s = state(); if (!s.enabled) return;
    enabled_ = true; exceptions_ = std::uncaught_exceptions(); s.begin(model, original, options);
  }
  ~RunScope() noexcept {
    if (!enabled_) return;
    auto& s = state(); s.run_unwound = std::uncaught_exceptions() > exceptions_; s.active = false;
  }
};
class DirectionScope {
  bool enabled_ = false; int exceptions_ = 0;
 public:
  template<class Point, class Result>
  DirectionScope(const Point& point, std::span<const double> rd, std::span<const double> rhs,
      std::span<const double> diagonal, double tau, double kappa, double theta, const Result& result) noexcept {
    auto& s = state(); if (!s.active || !s.ready) return;
    enabled_ = true; exceptions_ = std::uncaught_exceptions(); s.input(point, rd, rhs, diagonal, tau, kappa, theta, result);
  }
  void got_u(std::span<const double> u) noexcept {
    if (!enabled_) return;
    auto& s = state(); if (!s.ready) return;
    s.copy_step(s.slots[s.pending], 7, u); s.slots[s.pending].have_u = s.ready;
  }
  template<class Direction, class Result>
  void returned(const Direction& direction, std::span<const double> constant, const Result& result) noexcept {
    if (enabled_) state().returned(direction, constant, result);
  }
  ~DirectionScope() noexcept {
    if (enabled_ && state().ready && std::uncaught_exceptions() > exceptions_) state().slots[state().pending].unwound = true;
  }
};
template<class Point> inline void commit(const Point& candidate, double next_tau, double step, std::int64_t iteration) noexcept {
  auto& s = state(); if (s.enabled) s.commit(candidate, next_tau, step, iteration);
}
template<class Point, class ChecksSource, class Result>
inline void assess(const Point& point, std::span<const double> x, std::span<const double> y,
    const ChecksSource& checks, bool replace, const Result& result) noexcept {
  auto& s = state(); if (s.enabled) s.assess(point, x, y, checks, replace, result);
}
}  // 命名空间 solver::v53hsd
