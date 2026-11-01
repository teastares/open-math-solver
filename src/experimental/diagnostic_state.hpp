#pragma once

#include <array>
#include <cstdio>
#include <type_traits>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// 各实验状态头共享同一线程的诊断输出开关，不依赖包含顺序。
namespace solver_v100_presolve {
inline thread_local bool emit_diagnostics = true;

enum class EmitStatus { buffered, direct, failed };

namespace diagnostic_detail {
// 锁原FILE以保持同一流的stdio顺序；不宣称锁住其它描述符或其它进程。
struct FileLock {
  FILE* stream;
  explicit FileLock(FILE* value) noexcept : stream(value) {
#if defined(_WIN32)
    _lock_file(stream);
#elif defined(__unix__) || defined(__APPLE__)
    ::flockfile(stream);
#endif
  }
  ~FileLock() noexcept {
#if defined(_WIN32)
    _unlock_file(stream);
#elif defined(__unix__) || defined(__APPLE__)
    ::funlockfile(stream);
#endif
  }
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
};

inline bool prepare_target(FILE* stream) noexcept {
  if (std::ferror(stream)) return false;
  return std::fflush(stream) == 0 && std::ferror(stream) == 0;
}

template<class Writer>
bool write_once(FILE* stream, const char* prefix, Writer& writer) noexcept {
  bool ok = std::fputs(prefix, stream) != EOF;
  if (ok) {
    writer(stream);
    ok = std::ferror(stream) == 0;
    if (ok) ok = std::fputc('\n', stream) != EOF;
  }
  const bool flushed = std::fflush(stream) == 0;
  return ok && flushed && std::ferror(stream) == 0;
}

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
// fdopen成功后所有权转给FILE；无论close是否成功，都只关闭一次。
struct DuplicateOwner {
  int descriptor = -1;
  FILE* stream = nullptr;
  bool close() noexcept {
    if (stream) {
      FILE* const owned = stream;
      stream = nullptr;
      return std::fclose(owned) == 0;
    }
    if (descriptor >= 0) {
      const int owned = descriptor;
      descriptor = -1;
#if defined(_WIN32)
      return _close(owned) == 0;
#else
      return ::close(owned) == 0;
#endif
    }
    return true;
  }
  ~DuplicateOwner() noexcept { (void)close(); }
  DuplicateOwner() = default;
  DuplicateOwner(const DuplicateOwner&) = delete;
  DuplicateOwner& operator=(const DuplicateOwner&) = delete;
};
#endif
}  // 命名空间 diagnostic_detail

template<class Writer>
EmitStatus emit_direct(FILE* target, const char* prefix, Writer&& writer) noexcept {
  static_assert(std::is_nothrow_invocable_v<Writer&, FILE*>);
  if (!target || !prefix) return EmitStatus::failed;
  diagnostic_detail::FileLock lock(target);
  if (!diagnostic_detail::prepare_target(target)) return EmitStatus::failed;
  return diagnostic_detail::write_once(target, prefix, writer) ? EmitStatus::direct : EmitStatus::failed;
}

template<class Writer>
EmitStatus emit_buffered(FILE* target, const char* prefix, Writer&& writer) noexcept {
  static_assert(std::is_nothrow_invocable_v<Writer&, FILE*>);
  if (!target || !prefix) return EmitStatus::failed;
  diagnostic_detail::FileLock lock(target);
  if (!diagnostic_detail::prepare_target(target)) return EmitStatus::failed;
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
  // 每次调用独有，声明在owner之前，使缓冲区在副本关闭前始终存活。
  std::array<char, 64 * 1024> buffer;
  diagnostic_detail::DuplicateOwner owner;
  const auto fallback = [&]() noexcept {
    if (!owner.close()) return EmitStatus::failed;
    return diagnostic_detail::write_once(target, prefix, writer) ? EmitStatus::direct : EmitStatus::failed;
  };
#if defined(_WIN32)
  const int original = _fileno(target);
  if (original < 0) return fallback();
  owner.descriptor = _dup(original);
  if (owner.descriptor < 0) return fallback();
  // 只查询和恢复副本的翻译模式，不改变原描述符或全局默认模式。
  const int previous = _setmode(owner.descriptor, _O_BINARY);
  if (previous == -1 || _setmode(owner.descriptor, previous) == -1) return EmitStatus::failed;
  if (previous != _O_TEXT && previous != _O_BINARY) return EmitStatus::failed;
  owner.stream = _fdopen(owner.descriptor, previous == _O_TEXT ? "wt" : "wb");
#else
  const int original = ::fileno(target);
  if (original < 0) return fallback();
  owner.descriptor = ::dup(original);
  if (owner.descriptor < 0) return fallback();
  owner.stream = ::fdopen(owner.descriptor, "w");
#endif
  if (!owner.stream) return fallback();
  owner.descriptor = -1;
  if (std::setvbuf(owner.stream, buffer.data(), _IOFBF, buffer.size()) != 0) return fallback();
  // 从prefix开始后，不论写入、刷新还是关闭失败，都不再回放到原流。
  const bool written = diagnostic_detail::write_once(owner.stream, prefix, writer);
  const bool closed = owner.close();
  return written && closed ? EmitStatus::buffered : EmitStatus::failed;
#else
  return diagnostic_detail::write_once(target, prefix, writer) ? EmitStatus::direct : EmitStatus::failed;
#endif
}
}
