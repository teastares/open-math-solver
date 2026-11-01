#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace solver::experimental::fma_axpy {

using Budget = std::function<void()>;
inline constexpr std::size_t budget_batch = 4096;

struct Result {
  // 成功为完整长度；失败为首个未写入的非有限坐标。
  std::size_t completed = 0;
  bool finite = true;
};

struct ControlState {
  std::uint32_t mxcsr = 0;
  std::uint16_t x87_control = 0;
};

inline constexpr bool environment_supported(std::uint32_t mxcsr, std::uint16_t x87_control) noexcept {
  // 忽略sticky状态和保留位；要求渐进下溢、最近舍入、64位x87有效精度和屏蔽异常。
  return (mxcsr & 0xffc0u) == 0x1f80u && (x87_control & 0x0f3fu) == 0x033fu;
}

inline ControlState read_control_state() noexcept {
  ControlState result;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  // 只读取控制字；不执行fwait、不清旗标、不改变调用方环境。
  __asm__ __volatile__("stmxcsr %0\n\tfnstcw %1"
      : "=m"(result.mxcsr), "=m"(result.x87_control) : : "memory");
#endif
  return result;
}

inline bool hardware_available() noexcept {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  // Clang的FMA位不单独保证OS保存YMM；AVX和FMA必须同时满足。
  // https://raw.githubusercontent.com/llvm/llvm-project/main/compiler-rt/lib/builtins/cpu_model/x86.c
  static const bool result = __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
  return result;
#else
  return false;
#endif
}

namespace implementation {

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
// 连续向量的非对齐FMA流程参考OpenBLAS及BLIS；每个结果仍作有限检查。
// https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.33/kernel/x86_64/daxpy_microk_haswell-2.c
// https://github.com/flame/blis/blob/master/kernels/zen/1/bli_axpyv_zen_int.c
// target只收原始指针/标量并返回整数状态，异常由外层在vzeroupper后处理。
__attribute__((target("avx,fma"), noinline))
inline Result hardware_segment(double* destination, const double* basis, std::size_t count,
    double projection) noexcept {
  const __m256d scalar = _mm256_set1_pd(projection);
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256d largest = _mm256_set1_pd(std::numeric_limits<double>::max());
  const __m128d scalar_tail = _mm_set_sd(projection);
  std::size_t index = 0;
  for (; count - index >= 4; index += 4) {
    const __m256d old_value = _mm256_loadu_pd(destination + index);
    const __m256d direction = _mm256_loadu_pd(basis + index);
    const __m256d updated = _mm256_fnmadd_pd(scalar, direction, old_value);
    const __m256d absolute = _mm256_andnot_pd(sign, updated);
    const int valid = _mm256_movemask_pd(_mm256_cmp_pd(absolute, largest, _CMP_LE_OQ));
    if (valid == 15) {
      _mm256_storeu_pd(destination + index, updated);
      continue;
    }
    // 坏pack尚未写入；按原坐标顺序重放到首坏值，保留完全相同的写入前缀。
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const double value = _mm_cvtsd_f64(_mm_fnmadd_sd(scalar_tail,
          _mm_load_sd(basis + index + lane), _mm_load_sd(destination + index + lane)));
      if (!std::isfinite(value)) {
        _mm256_zeroupper();
        return {index + lane, false};
      }
      destination[index + lane] = value;
    }
  }
  for (; index < count; ++index) {
    const double value = _mm_cvtsd_f64(_mm_fnmadd_sd(scalar_tail,
        _mm_load_sd(basis + index), _mm_load_sd(destination + index)));
    if (!std::isfinite(value)) {
      _mm256_zeroupper();
      return {index, false};
    }
    destination[index] = value;
  }
  _mm256_zeroupper();
  return {count, true};
}
#endif

}  // 命名空间 implementation

// 单次double舍入可能不同于父LD中间舍入；本函数不代替调用方的真实残差验收。
// 非默认控制态只选择父表达式，不声明整个求解器支持任意动态浮点环境。
inline Result subtract_scaled(std::span<double> destination, std::span<const double> basis,
    double projection, const Budget& check = {}, bool hardware = false, bool* used = nullptr) {
  if (used) *used = false;
  if (destination.size() != basis.size()) throw std::invalid_argument("AXPY完整向量维数不一致");
  for (std::size_t begin = 0; begin < destination.size();) {
    // 原坐标0、4096……处各一次；回调返回后才决定当前段，不能缓存控制环境。
    if (check) check();
    const auto count = std::min(budget_batch, destination.size() - begin);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    if (hardware) {
      const bool available = hardware_available();
      const auto control = read_control_state();
      if (available && environment_supported(control.mxcsr, control.x87_control)) {
        if (used) *used = true;
        const auto result = implementation::hardware_segment(destination.data() + begin,
            basis.data() + begin, count, projection);
        if (!result.finite) return {begin + result.completed, false};
        begin += count;
        continue;
      }
    }
#else
    (void)hardware;
#endif
    // 当前段预算已经支付；保留父LD乘减和先窄化再检查的顺序，不重复调用budget。
    for (std::size_t offset = 0; offset < count; ++offset) {
      const auto index = begin + offset;
      const long double value = static_cast<long double>(destination[index]) -
          static_cast<long double>(projection) * basis[index];
      const double narrowed = static_cast<double>(value);
      if (!std::isfinite(value) || !std::isfinite(narrowed)) return {index, false};
      destination[index] = narrowed;
    }
    begin += count;
  }
  return {destination.size(), true};
}

}  // 命名空间 solver::experimental::fma_axpy
