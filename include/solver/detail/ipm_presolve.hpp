#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

namespace solver::detail {

// 用 binary64 的精确整数表示核对等价关系，不依赖扩展精度或融合运算库。
inline constexpr bool ipm_exact_bound_checks_available =
    std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::radix == 2 &&
    std::numeric_limits<double>::digits == 53 && std::numeric_limits<double>::max_exponent == 1024 &&
    std::numeric_limits<double>::min_exponent == -1021 && sizeof(double) == sizeof(std::uint64_t) &&
    (std::endian::native == std::endian::little || std::endian::native == std::endian::big);

struct IpmBinaryMagnitude {
  std::uint64_t odd;
  int exponent;
};

template <typename Float>
inline std::uint64_t ipm_binary_bits(Float value) {
  // 依赖类型的分支避免在非 binary64 平台实例化大小不匹配的 bit_cast。
  if constexpr (sizeof(Float) == sizeof(std::uint64_t)) return std::bit_cast<std::uint64_t>(value);
  else return 0;
}

inline IpmBinaryMagnitude ipm_binary_magnitude(double value) {
  const auto bits = ipm_binary_bits(value);
  const int encoded_exponent = static_cast<int>((bits >> 52) & 0x7ff);
  auto significand = bits & ((std::uint64_t{1} << 52) - 1);
  if (encoded_exponent != 0) significand |= std::uint64_t{1} << 52;
  if (significand == 0) return {0, 0};
  const int zeros = std::countr_zero(significand);
  return {significand >> zeros, (encoded_exponent == 0 ? -1074 : encoded_exponent - 1075) + zeros};
}

inline std::optional<double> ipm_exact_bound_quotient(double bound, double coefficient) {
  if constexpr (!ipm_exact_bound_checks_available) return std::nullopt;
  constexpr auto magnitude_mask = (std::uint64_t{1} << 63) - 1;
  const auto bound_bits = ipm_binary_bits(bound);
  const auto coefficient_bits = ipm_binary_bits(coefficient);
  if (!std::isfinite(bound) || !std::isfinite(coefficient) ||
      (coefficient_bits & magnitude_mask) == 0) return std::nullopt;
  const double quotient = bound / coefficient;
  if (!std::isfinite(quotient)) return std::nullopt;
  const auto quotient_bits = ipm_binary_bits(quotient);
  const bool quotient_zero = (quotient_bits & magnitude_mask) == 0;
  if ((bound_bits & magnitude_mask) == 0) return quotient_zero ? std::optional<double>{quotient} : std::nullopt;
  if (quotient_zero || ((coefficient_bits ^ quotient_bits ^ bound_bits) >> 63) != 0) {
    return std::nullopt;
  }
  const auto a = ipm_binary_magnitude(coefficient), q = ipm_binary_magnitude(quotient);
  const auto b = ipm_binary_magnitude(bound);
  // 三个尾数已去掉全部二因子。用整除核对乘积，避免 106 位乘法或溢出。
  if (a.exponent + q.exponent != b.exponent || b.odd % a.odd != 0 || b.odd / a.odd != q.odd) {
    return std::nullopt;
  }
  return quotient;
}

}  // 命名空间 solver::detail
