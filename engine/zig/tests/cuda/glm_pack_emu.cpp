// glm_pack.cu and python-0.6's experts_pack.cu (the tree's checked copy of its device code) on a CPU, for tests without
// a GPU (glm/cuda_pack_test.zig): each launch's CUDA threads run one after another, a call each, which is exact for
// kernels that share nothing between threads; the arguments come as cuLaunchKernel takes them, count and sizes checked.
#include <stddef.h>
#include <stdint.h>

#include <tuple>
#include <utility>

namespace {
struct Dim {
    unsigned x, y, z;
};
Dim blockIdx, threadIdx, blockDim, gridDim;
}  // namespace

#define __global__
#define __device__
#define __forceinline__ inline
#define __restrict__ __restrict
#define __launch_bounds__(...)
#include <math.h>
#include <string.h>
// the CUDA intrinsics glm_pack.cu's quantize4 calls: IEEE single ops rounded to nearest, and the bit casts
static inline float __uint_as_float(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static inline uint32_t __float_as_uint(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
#include "../../kernels/cuda/glm_pack.cu"
#include "../../kernels/cuda/experts_pack.cu"

namespace {

template <class... A, size_t... I>
std::tuple<A...> unpackArgs(void** p, std::index_sequence<I...>) {
    return std::tuple<A...>(*static_cast<A*>(p[I])...);
}

// `kern` over the grid, every thread of a block in order (x fastest), block after block: 0, or 2 (count) or 3 (a size)
template <class... A>
int call(void (*kern)(A...), const unsigned* grid, const unsigned* block, void** p, const uint16_t* sizes, unsigned count) {
    constexpr unsigned N = sizeof...(A);
    if (count != N) return 2;
    const size_t want[N] = {sizeof(A)...};
    for (unsigned i = 0; i < N; i++)
        if (want[i] != sizes[i]) return 3;
    const auto args = unpackArgs<A...>(p, std::index_sequence_for<A...>{});
    gridDim = {grid[0], grid[1], grid[2]};
    blockDim = {block[0], block[1], block[2]};
    for (unsigned z = 0; z < grid[2]; z++)
        for (unsigned y = 0; y < grid[1]; y++)
            for (unsigned x = 0; x < grid[0]; x++)
                for (unsigned t = 0; t < block[0] * block[1] * block[2]; t++) {
                    blockIdx = {x, y, z};
                    threadIdx = {t % block[0], (t / block[0]) % block[1], t / (block[0] * block[1])};
                    std::apply(kern, args);
                }
    return 0;
}

}  // namespace

// kernel 0 tf_glm_pack_q4, 1 tf_glm_pack_major, 2 tf_glm_pack_rows, 3 tf_glm_pack_experts, 4 python's pack_kernel<2>,
// 5 tf_glm_pack_quant4, 6 tf_glm_pack_quant4s (EXL3's draft head), 7 tf_glm_pack_f8, 8 tf_glm_pack_f8s (the FP8 head)
extern "C" int tf_glm_pack_emu(int kernel, const unsigned* grid, const unsigned* block, void** params,
                               const uint16_t* sizes, unsigned count) {
    switch (kernel) {
        case 0: return call(tf_glm_pack_q4, grid, block, params, sizes, count);
        case 1: return call(tf_glm_pack_major, grid, block, params, sizes, count);
        case 2: return call(tf_glm_pack_rows, grid, block, params, sizes, count);
        case 3: return call(tf_glm_pack_experts, grid, block, params, sizes, count);
        case 4: return call(tf_experts_pack::pack_kernel<2>, grid, block, params, sizes, count);
        case 5: return call(tf_glm_pack_quant4, grid, block, params, sizes, count);
        case 6: return call(tf_glm_pack_quant4s, grid, block, params, sizes, count);
        case 7: return call(tf_glm_pack_f8, grid, block, params, sizes, count);
        case 8: return call(tf_glm_pack_f8s, grid, block, params, sizes, count);
        default: return 1;
    }
}
