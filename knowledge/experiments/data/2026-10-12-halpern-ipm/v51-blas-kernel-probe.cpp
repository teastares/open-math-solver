// 直接调用旧冻结库中的内核，诊断六列路径的临时缓冲区容量；不计入求解性能。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <malloc.h>

extern "C" int dgemm_kernel_COOPERLAKE(std::int64_t, std::int64_t, std::int64_t,
                                     double, const double*, const double*, double*, std::int64_t);

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::int64_t m = 16, n = 6, k = std::strtoll(argv[1], nullptr, 10);
    if (k < 1 || k > 512) return 2;
    auto* a = static_cast<double*>(_aligned_malloc((m * k + 128) * sizeof(double), 64));
    auto* b = static_cast<double*>(_aligned_malloc((n * k + 128) * sizeof(double), 64));
    auto* c = static_cast<double*>(_aligned_malloc((m * n + 128) * sizeof(double), 64));
    if (!a || !b || !c) return 3;
    for (std::int64_t p = 0; p < k; ++p)
        for (std::int64_t i = 0; i < m; ++i) a[p * m + i] = double(i + 1);
    for (std::int64_t j = 0; j < n; j += 2)
        for (std::int64_t p = 0; p < k; ++p)
            for (std::int64_t t = 0; t < 2; ++t) b[j * k + p * 2 + t] = double(j + t + 1);
    for (std::int64_t i = 0; i < m * n; ++i) c[i] = 0.0;
    std::printf("{\"phase\":\"before\",\"m\":%lld,\"n\":%lld,\"k\":%lld}\n", m, n, k);
    std::fflush(stdout);
    const int code = dgemm_kernel_COOPERLAKE(m, n, k, 1.0, a, b, c, m);
    std::int64_t wrong = 0;
    for (std::int64_t j = 0; j < n; ++j)
        for (std::int64_t i = 0; i < m; ++i) {
            const double expected = double(k * (i + 1) * (j + 1));
            if (c[j * m + i] != expected) {
                if (wrong < 3) std::printf("{\"row\":%lld,\"column\":%lld,\"actual\":%.17g,\"expected\":%.17g}\n", i, j, c[j * m + i], expected);
                ++wrong;
            }
        }
    std::printf("{\"phase\":\"after\",\"kernel_code\":%d,\"wrong\":%lld}\n", code, wrong);
    _aligned_free(a); _aligned_free(b); _aligned_free(c);
    return wrong ? 1 : 0;
}
