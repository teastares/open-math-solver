#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <limits>
#include <type_traits>

namespace solver::v71_basis_capture {

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
static_assert(sizeof(std::int64_t) == 8);
static_assert(std::endian::native == std::endian::little || std::endian::native == std::endian::big);

inline constexpr std::size_t cache_limit = 134217728;
inline constexpr std::size_t metadata_limit = 65536;
inline constexpr char magic[] = "V71BAS01";
inline constexpr const char* names[] = {"g_starts", "g_rows", "g_values", "costs", "rhs", "slack", "dual",
    "inequality", "candidate_x", "candidate_s", "candidate_z", "selected_rows", "pivot_columns"};
inline constexpr bool integer_fields[] = {true, true, false, false, false, false, false,
    false, false, false, false, true, true};

struct Field {
  std::size_t count = 0, offset = 0, width = 8;
};

struct State {
  bool enabled = false, complete = false, truncated = false;
  std::size_t attempts = 0, capture_errors = 0, fixed_cache_bytes = 0, limit = cache_limit;
  std::size_t n = 0, m = 0, nnz = 0, payload_bytes = 0, binary_bytes = 0;
  const char* error = "none";
  std::array<char, 4096> prefix{};
  std::array<Field, 13> fields{};
  std::unique_ptr<std::byte[]> payload;
  void fail(const char* reason, bool is_truncated = false) noexcept {
    capture_errors = 1;
    complete = false;
    truncated = truncated || is_truncated;
    error = reason;
  }
  const char* status() const noexcept {
    return capture_errors ? "capture_error" : complete ? "captured" : "no_candidate";
  }
};

inline thread_local State* current = nullptr;

inline bool write_binary(State& state) noexcept {
  if (!state.prefix[0]) return false;
  std::array<char, 4110> path{};
  std::snprintf(path.data(), path.size(), "%s.bin", state.prefix.data());
  auto* output = std::fopen(path.data(), "wb");
  if (!output) return false;
  state.binary_bytes = std::fwrite(magic, 1, 8, output);
  bool ok = state.binary_bytes == 8;
  if (ok && state.complete) {
    const auto written = std::fwrite(state.payload.get(), 1, state.payload_bytes, output);
    state.binary_bytes += written;
    ok = written == state.payload_bytes;
  }
  if (std::fclose(output) != 0) ok = false;
  return ok;
}

inline bool write_metadata(const State& state) noexcept {
  if (!state.prefix[0]) return false;
  std::array<char, 4110> path{};
  std::snprintf(path.data(), path.size(), "%s.json", state.prefix.data());
  auto* output = std::fopen(path.data(), "wb");
  if (!output) return false;
  std::fprintf(output, "{\"schema\":1,\"version\":\"v71\",\"status\":\"%s\","
      "\"attempts\":%llu,\"event_count\":%d,\"capture_errors\":%llu,\"truncated\":%s,"
      "\"error\":\"%s\",\"n\":%llu,\"m\":%llu,\"nnz\":%llu,"
      "\"fixed_cache_bytes\":%llu,\"payload_bytes\":%llu,\"cache_bytes\":%llu,"
      "\"cache_limit_bytes\":%llu,\"metadata_limit_bytes\":%llu,\"binary_bytes\":%llu,"
      "\"byte_order\":\"%s\",\"events\":[",
      state.status(), static_cast<unsigned long long>(state.attempts), state.complete ? 1 : 0,
      static_cast<unsigned long long>(state.capture_errors), state.truncated ? "true" : "false", state.error,
      static_cast<unsigned long long>(state.n), static_cast<unsigned long long>(state.m),
      static_cast<unsigned long long>(state.nnz), static_cast<unsigned long long>(state.fixed_cache_bytes),
      static_cast<unsigned long long>(state.payload_bytes),
      static_cast<unsigned long long>(state.fixed_cache_bytes + (state.payload ? state.payload_bytes : 0)),
      static_cast<unsigned long long>(state.limit), static_cast<unsigned long long>(metadata_limit),
      static_cast<unsigned long long>(state.binary_bytes), std::endian::native == std::endian::little ? "little" : "big");
  if (state.complete) {
    std::fputs("{\"index\":0,\"phase\":\"recovered_basis_before_assess\",\"arrays\":[", output);
    for (std::size_t i = 0; i < state.fields.size(); ++i) {
      const auto& field = state.fields[i];
      std::fprintf(output, "%s{\"name\":\"%s\",\"dtype\":\"%s\",\"width\":%llu,"
          "\"count\":%llu,\"shape\":[%llu],\"offset\":%llu}", i ? "," : "", names[i],
          i == 7 ? "uint8" : integer_fields[i] ? "int64" : "float64",
          static_cast<unsigned long long>(field.width), static_cast<unsigned long long>(field.count),
          static_cast<unsigned long long>(field.count), static_cast<unsigned long long>(field.offset));
    }
    std::fputs("]}", output);
  }
  std::fputs("]}\n", output);
  const auto bytes = std::ftell(output);
  bool ok = !std::ferror(output) && bytes >= 0 && static_cast<unsigned long>(bytes) <= metadata_limit;
  if (std::fclose(output) != 0) ok = false;
  return ok;
}

struct Report {
  State state;
  State* previous = nullptr;
  Report() noexcept {
    previous = current;
    current = &state;
    state.fixed_cache_bytes = sizeof(Report);
    const auto* value = std::getenv("SOLVER_V71_CAPTURE");
    state.enabled = value && value[0];
    if (!state.enabled) return;
    const auto length = std::strlen(value);
    if (length >= state.prefix.size()) { state.fail("prefix_too_long"); return; }
    std::memcpy(state.prefix.data(), value, length + 1);
  }
  Report(const Report&) = delete;
  Report& operator=(const Report&) = delete;
  ~Report() noexcept {
    current = previous;
    if (!state.enabled) return;
    bool ok = write_binary(state);
    if (!ok) state.fail("binary_write_failed");
    if (!write_metadata(state)) { state.fail("metadata_write_failed"); ok = false; }
    std::fprintf(stderr, "V71_CAPTURE {\"schema\":1,\"version\":\"v71\",\"status\":\"%s\","
        "\"attempts\":%llu,\"event_count\":%d,\"capture_errors\":%llu,\"truncated\":%s,"
        "\"binary_bytes\":%llu,\"write_ok\":%s}\n", state.status(),
        static_cast<unsigned long long>(state.attempts), state.complete ? 1 : 0,
        static_cast<unsigned long long>(state.capture_errors), state.truncated ? "true" : "false",
        static_cast<unsigned long long>(state.binary_bytes), ok ? "true" : "false");
  }
};

template<class Input, class Result, class Budget>
void capture(const Input& input, const Result& result, Budget&& budget) {
  auto* state = current;
  if (!state || !state->enabled || state->attempts || state->capture_errors) return;
  ++state->attempts;
  const auto check = [&] {
    try { budget(); }
    catch (...) { state->fail("budget_interrupted"); throw; }
  };
  check();
  state->n = input.costs.size(); state->m = input.rhs.size(); state->nnz = input.values.size();
  if (!result.candidate_available || input.starts.size() != state->n + 1 || input.rows.size() != state->nnz ||
      input.slack.size() != state->m || input.dual.size() != state->m || input.inequality.size() != state->m ||
      result.x.size() != state->n || result.s.size() != state->m || result.z.size() != state->m ||
      result.stats.selected_rows.size() != state->n || result.stats.pivot_columns.size() != state->n) {
    state->fail("inconsistent_dimensions"); return;
  }
  const std::array<std::size_t, 13> counts{input.starts.size(), input.rows.size(), input.values.size(),
      input.costs.size(), input.rhs.size(), input.slack.size(), input.dual.size(), input.inequality.size(),
      result.x.size(), result.s.size(), result.z.size(), result.stats.selected_rows.size(), result.stats.pivot_columns.size()};
  if (state->limit < state->fixed_cache_bytes) { state->fail("cache_limit", true); return; }
  const auto available = state->limit - state->fixed_cache_bytes;
  std::size_t total = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    const auto width = i == 7 ? 1u : 8u;
    if (counts[i] > (available - total) / width) { state->fail("cache_limit", true); return; }
    state->fields[i] = {counts[i], 8 + total, width};
    total += counts[i] * width;
  }
  state->payload_bytes = total;
  try { state->payload.reset(new std::byte[total]); }
  catch (const std::bad_alloc&) { state->fail("allocation_failed"); return; }
  const auto copy = [&](std::size_t field_index, const auto& source) {
    static_assert(sizeof(typename std::remove_reference_t<decltype(source)>::value_type) == 8);
    const auto& field = state->fields[field_index];
    for (std::size_t first = 0; first < field.count; first += 4096) {
      check();
      const auto count = std::min<std::size_t>(4096, field.count - first);
      std::memcpy(state->payload.get() + field.offset - 8 + first * 8, source.data() + first, count * 8);
    }
  };
  copy(0, input.starts); copy(1, input.rows); copy(2, input.values); copy(3, input.costs);
  copy(4, input.rhs); copy(5, input.slack); copy(6, input.dual);
  for (std::size_t i = 0; i < state->m; ++i) {
    if ((i & 4095) == 0) check();
    state->payload[state->fields[7].offset - 8 + i] = static_cast<std::byte>(input.inequality[i] ? 1 : 0);
  }
  copy(8, result.x); copy(9, result.s); copy(10, result.z);
  copy(11, result.stats.selected_rows); copy(12, result.stats.pivot_columns);
  check();
  state->complete = true;
}

}  // 命名空间 solver::v71_basis_capture
