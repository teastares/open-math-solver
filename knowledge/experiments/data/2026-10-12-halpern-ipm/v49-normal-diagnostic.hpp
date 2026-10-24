#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <cholmod.h>

namespace solver::detail::v49normal {

// 两个诊断组件各用 128 MiB 数组额度；这不是整个进程 RSS 的上界。
inline constexpr std::uint64_t cache_limit = 128ULL * 1024 * 1024;
inline constexpr std::size_t event_limit = 2048;
enum class Phase { input, gram, conversion, cholmod, complete };
inline const char* phase_name(Phase phase) noexcept {
  switch (phase) {
    case Phase::input: return "input";
    case Phase::gram: return "gram_accumulation";
    case Phase::conversion: return "double_conversion";
    case Phase::cholmod: return "cholmod_numeric";
    case Phase::complete: return "complete";
  }
  return "unknown";
}
inline void quoted(std::ostream& out, const std::string& value) {
  const char* digits = "0123456789abcdef";
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 32) out << "\\u00" << digits[c >> 4] << digits[c & 15];
    else out << static_cast<char>(c);
  }
  out << '"';
}

struct Event {
  std::uint64_t attempt = 0, numeric_calls = 0;
  std::uint64_t minor = 0, xsize = 0, nsuper = 0;
  Phase phase = Phase::input;
  int success = 0, status = 0;
  // kind=0 是外层退出，kind=1 是既有 CHOLMOD 判断返回 false。
  int kind = 0, outcome = 0;
  bool numeric_started = false;
};

struct State {
  std::string prefix;
  bool enabled = false, diagnostic_failed = false, event_limit_reached = false;
  bool backend_selected = false;
  bool snapshot_attempted = false, captured = false, snapshot_oversize = false;
  bool snapshot_invalid = false, binary_written = false, write_failed = false;
  std::uint64_t attempts = 0, numeric_calls = 0, current_attempt = 0;
  bool current_numeric_started = false;
  std::uint64_t n = 0, nnz = 0, snapshot_attempt = 0, snapshot_bytes = 0;
  Event failure;
  std::vector<Event> events;
  std::vector<std::int64_t> starts, rows;
  std::vector<double> values;

  State() noexcept {
    const char* value = std::getenv("SOLVER_V49_DIAGNOSTICS");
    if (!value || !*value) return;
    try {
      prefix = value;
      events.reserve(event_limit);
      enabled = true;
    } catch (...) { diagnostic_failed = true; }
  }
  void add(Event event) noexcept {
    if (!enabled) return;
    if (events.size() == event_limit) { event_limit_reached = true; return; }
    // 首次启用已保留容量，后续事件不会扩大缓存。
    events.push_back(event);
  }
  void capture(const cholmod_sparse& matrix, const cholmod_factor& factor, int success, int status) noexcept {
    if (!enabled) return;
    Event event;
    event.attempt = current_attempt;
    event.numeric_calls = numeric_calls;
    event.numeric_started = current_numeric_started;
    event.kind = 1;
    event.phase = Phase::cholmod;
    event.success = success;
    event.status = status;
    event.minor = factor.minor;
    event.xsize = factor.xsize;
    event.nsuper = factor.nsuper;
    add(event);
    if (snapshot_attempted) return;
    snapshot_attempted = true;
    failure = event;
    try {
      if (std::endian::native != std::endian::little || matrix.nrow != matrix.ncol ||
          matrix.itype != CHOLMOD_LONG || matrix.xtype != CHOLMOD_REAL || matrix.dtype != CHOLMOD_DOUBLE ||
          matrix.stype != 1 || !matrix.p || !matrix.i || !matrix.x || !matrix.packed || !matrix.sorted) {
        snapshot_invalid = true;
        return;
      }
      n = matrix.nrow;
      const auto* p = static_cast<const std::int64_t*>(matrix.p);
      if (p[n] < 0 || static_cast<std::uint64_t>(p[n]) > matrix.nzmax) { snapshot_invalid = true; return; }
      nnz = static_cast<std::uint64_t>(p[n]);
      const std::uint64_t available = cache_limit - events.capacity() * sizeof(Event);
      // 含文件头的字节数保守覆盖本组件三个快照数组，不产生临时整矩阵副本。
      if (n > (available - 40) / 8 || nnz > (available - 40 - 8 * n) / 16) {
        snapshot_oversize = true;
        return;
      }
      snapshot_bytes = 40 + 8 * n + 16 * nnz;
      snapshot_attempt = current_attempt;
      starts.assign(p, p + n + 1);
      const auto* i = static_cast<const std::int64_t*>(matrix.i);
      const auto* x = static_cast<const double*>(matrix.x);
      rows.assign(i, i + nnz);
      values.assign(x, x + nnz);
      captured = true;
    } catch (...) { diagnostic_failed = true; enabled = false; }
  }
  static void write_bytes(std::ostream& out, const void* data, std::size_t bytes) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  }
  void metadata(std::ostream& out) const {
    out << "{\"schema\":1,\"component\":\"normal\",\"snapshot_attempted\":" << snapshot_attempted
        << ",\"backend_selected\":" << backend_selected
        << ",\"captured\":" << captured << ",\"snapshot_oversize\":" << snapshot_oversize
        << ",\"snapshot_invalid\":" << snapshot_invalid << ",\"diagnostic_failed\":" << diagnostic_failed
        << ",\"event_limit_reached\":" << event_limit_reached << ",\"write_failed\":" << write_failed
        << ",\"binary_written\":" << binary_written << ",\"cache_limit_bytes\":" << cache_limit
        << ",\"event_storage_bytes\":" << events.capacity() * sizeof(Event)
        << ",\"snapshot_bytes\":" << snapshot_bytes << ",\"byte_order\":\"little\",\"n\":" << n
        << ",\"nnz\":" << nnz << ",\"snapshot_attempt\":" << snapshot_attempt
        << ",\"attempts\":" << attempts << ",\"numeric_calls\":" << numeric_calls << ",\"events\":[";
    bool first = true;
    for (const auto& event : events) {
      if (!first) out << ',';
      first = false;
      out << "{\"kind\":\"" << (event.kind ? "cholmod_failure" : "attempt_exit")
          << "\",\"attempt\":" << event.attempt << ",\"phase\":\"" << phase_name(event.phase)
          << "\",\"outcome\":\"" << (event.kind ? "false" : event.outcome == 2 ? "unwound" : event.outcome == 1 ? "success" : "false")
          << "\",\"numeric_started\":" << event.numeric_started << ",\"numeric_calls\":" << event.numeric_calls;
      if (event.kind) out << ",\"success\":" << event.success << ",\"status\":" << event.status
          << ",\"minor\":" << event.minor << ",\"xsize\":" << event.xsize << ",\"nsuper\":" << event.nsuper;
      out << '}';
    }
    out << "]}";
  }
  ~State() noexcept {
    if (prefix.empty()) return;
    try {
      if (captured) {
        std::ofstream binary(prefix + ".normal.bin", std::ios::binary | std::ios::trunc);
        binary.exceptions(std::ios::badbit | std::ios::failbit);
        write_bytes(binary, "V49GRAM1", 8);
        for (const std::uint64_t field : {n, nnz, snapshot_attempt}) write_bytes(binary, &field, 8);
        write_bytes(binary, starts.data(), starts.size() * sizeof(std::int64_t));
        write_bytes(binary, rows.data(), rows.size() * sizeof(std::int64_t));
        write_bytes(binary, values.data(), values.size() * sizeof(double));
        binary.close();
        binary_written = true;
      }
    } catch (...) { write_failed = true; }
    try {
      std::ofstream json(prefix + ".normal.json", std::ios::trunc);
      json.exceptions(std::ios::badbit | std::ios::failbit);
      json << std::boolalpha;
      metadata(json);
      json << '\n';
      json.close();
    } catch (...) { write_failed = true; }
    try {
      // 文件写失败也必须留独立可解析标记；这里不改变原程序的退出状态。
      std::cerr << "V49_NORMAL {\"schema\":1,\"captured\":" << (captured ? "true" : "false")
          << ",\"binary_written\":" << (binary_written ? "true" : "false")
          << ",\"write_failed\":" << (write_failed ? "true" : "false")
          << ",\"diagnostic_failed\":" << (diagnostic_failed ? "true" : "false") << ",\"prefix\":";
      quoted(std::cerr, prefix);
      std::cerr << "}\n";
    } catch (...) {}
  }
};

inline State& state() noexcept { static thread_local State value; return value; }
class Scope {
  State* state_ = nullptr;
  const bool* valid_ = nullptr;
  int exceptions_ = 0;
  Phase phase_ = Phase::input;
 public:
  Scope(bool selected, const bool& valid) noexcept {
    auto& current = state();
    if (!current.enabled) return;
    if (!selected) return;
    current.backend_selected = true;
    state_ = &current;
    valid_ = &valid;
    exceptions_ = std::uncaught_exceptions();
    current.current_attempt = ++current.attempts;
    current.current_numeric_started = false;
  }
  void phase(Phase phase) noexcept { if (state_) phase_ = phase; }
  ~Scope() noexcept {
    if (!state_) return;
    Event event;
    event.attempt = state_->current_attempt;
    event.numeric_calls = state_->numeric_calls;
    event.numeric_started = state_->current_numeric_started;
    event.phase = phase_;
    event.outcome = std::uncaught_exceptions() > exceptions_ ? 2 : *valid_ ? 1 : 0;
    state_->add(event);
    state_->current_attempt = 0;
  }
};
inline void numeric_start() noexcept {
  auto& current = state();
  if (!current.enabled) return;
  ++current.numeric_calls;
  current.current_numeric_started = true;
}
inline void failure(const cholmod_sparse& matrix, const cholmod_factor& factor, int success, int status) noexcept {
  auto& current = state();
  if (!current.enabled) return;
  current.capture(matrix, factor, success, status);
}
}  // 命名空间 solver::detail::v49normal
