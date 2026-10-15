// GLM-5.3-Flash's pack pass on the GPU (glm/cuda_pack.zig): python-0.6's load-time layouts from the stored MLX bytes.
// One thread an output element, nothing shared between threads, so zig/tests/cuda/glm_pack_emu.cpp runs it on a CPU.
// Output column c of a pass reads stored row (c / rb) * rs + c % rb: rb = n is plain rows, (256, 512) kv_b's by head.

#include <stdint.h>

// kernels/qmm.py `pack` of groups of 64: MLX words [rows][k8] -> [npad/64][kg][8][32][2], zeros past n;
// nibble p of lane l's word v holds input 32 v + 2 (l % 4) + (p % 4) 8 + p / 4 of its group (OFFSETS)
extern "C" __global__ void tf_glm_pack_q4(const uint32_t* __restrict__ w, int n, int k8, int kg, int rb, int rs,
                                          uint32_t* __restrict__ out, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = static_cast<int>(idx & 1), lane = static_cast<int>((idx >> 1) & 31);
    const int j = static_cast<int>((idx >> 6) & 7);
    const long long tg = idx >> 9;
    const int g = static_cast<int>(tg % kg);
    const long long c = (tg / kg) * 64 + j * 8 + (lane >> 2);
    uint32_t word = 0;
    if (c < n) {
        const uint32_t* row = w + ((c / rb) * rs + c % rb) * static_cast<long long>(k8);
        for (int p = 0; p < 8; ++p) {
            const int in = g * 64 + 32 * v + 2 * (lane & 3) + (p & 3) * 8 + (p >> 2);
            word |= ((row[in >> 3] >> (4 * (in & 7))) & 0xFu) << (4 * p);
        }
    }
    out[idx] = word;
}

// The packed scales' and biases' layout: [rows][kg] 16-bit -> [kg][npad], zeros past n
extern "C" __global__ void tf_glm_pack_major(const uint16_t* __restrict__ x, int n, int kg, int npad, int rb, int rs,
                                             uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), c = static_cast<int>(idx % npad);
    out[idx] = c < n ? x[(static_cast<long long>(c / rb) * rs + c % rb) * kg + g] : 0;
}

// Rows of `bytes` bytes gathered in order (latent.AbsorbQ4's key or value rows, as stored)
extern "C" __global__ void tf_glm_pack_rows(const uint8_t* __restrict__ src, int rb, int rs, int bytes, long long rows,
                                            uint8_t* __restrict__ dst) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * bytes) return;
    const long long c = idx / bytes, x = idx % bytes;
    dst[idx] = src[((c / rb) * rs + c % rb) * bytes + x];
}

// nibbles (i0 .. i7) -> (i0, i2, i4, i6, i1, i3, i5, i7), as experts_pack.cu's shuffle
__device__ __forceinline__ uint32_t tf_glm_shuffle(uint32_t w) {
    uint32_t even = w & 0x0F0F0F0Fu, odd = (w >> 4) & 0x0F0F0F0Fu;
    even = (even | (even >> 4)) & 0x00FF00FFu;
    odd = (odd | (odd >> 4)) & 0x00FF00FFu;
    even = (even | (even >> 8)) & 0x0000FFFFu;
    odd = (odd | (odd >> 8)) & 0x0000FFFFu;
    return even | (odd << 16);
}

// cuda/experts.py `pack` (experts_pack.cu's pack_kernel<2>, groups of 64) of E experts' words [E][n][k8], scales and
// biases [E][n][kg], written as projection m of mm in `make`'s stack: out [E][n/32][kg][mm][288], a block per (g, b, e)
extern "C" __global__ void __launch_bounds__(288)
tf_glm_pack_experts(const uint32_t* __restrict__ words, const uint16_t* __restrict__ scales,
                    const uint16_t* __restrict__ biases, uint32_t* __restrict__ out, int n, int k8, int kg, int nb,
                    int m, int mm) {
    const int g = blockIdx.x, b = blockIdx.y, e = blockIdx.z, i = threadIdx.x;
    uint32_t* dst = out + (((static_cast<int64_t>(e) * nb + b) * kg + g) * mm + m) * 288;
    if (i < 256) {
        const int row = i / 8, kk = i % 8;                            // row = t * 8 + r; kk = q * 2 + j
        const int t = row / 8, r = row % 8, q = kk / 2, j = kk % 2;
        const uint32_t w = words[(static_cast<int64_t>(e) * n + b * 32 + row) * k8 + g * 8 + kk];
        const int f = ((t * 2 + j) * 8 + r) * 4 + q;
        dst[(f / 128) * 128 + (f % 32) * 4 + (f / 32) % 4] = tf_glm_shuffle(w);
    } else {
        const int s = i - 256;                                        // p * 8 + (0 scales, 1 biases) * 4 + t
        const int p = s / 8, t = s % 4;
        const uint16_t* src = (s / 4) % 2 ? biases : scales;
        const int64_t at = (static_cast<int64_t>(e) * n + b * 32 + t * 8 + p * 2) * kg + g;
        dst[256 + s] = static_cast<uint32_t>(src[at]) | (static_cast<uint32_t>(src[at + kg]) << 16);
    }
}

// qmm.quantize4 (EXL3's draft head, drafts only): bf16 rows [n][k] in affine 4-bit groups of 64, torch's CUDA ops in
// order: lo and hi of the group, scale = bf16(max((hi - lo) * fp32(1 / 15), 1e-8)) (a division by a CPU scalar is a
// product with its fp32 reciprocal), bias = bf16(lo), q = clamp(rint((x - bias) / scale), 0, 15); NaN propagates
__device__ __forceinline__ float tf_glm_f32(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

__device__ __forceinline__ uint16_t tf_glm_bf16(float x) {  // round to nearest even, as torch's .to(bfloat16)
    const uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

struct tf_glm_q4g {
    uint16_t scale, bias;
};

__device__ __forceinline__ tf_glm_q4g tf_glm_group(const uint16_t* __restrict__ g) {
    float lo = tf_glm_f32(g[0]), hi = lo;
    for (int i = 1; i < 64; ++i) {
        const float v = tf_glm_f32(g[i]);
        lo = (v < lo || v != v || lo != lo) ? (lo != lo ? lo : v) : lo;
        hi = (v > hi || v != v || hi != hi) ? (hi != hi ? hi : v) : hi;
    }
    float s = __fmul_rn(__fsub_rn(hi, lo), 1.0f / 15.0f);
    s = (s != s || s >= 1e-8f) ? s : 1e-8f;
    return {tf_glm_bf16(s), tf_glm_bf16(lo)};
}

__device__ __forceinline__ uint32_t tf_glm_q4(uint16_t x, tf_glm_q4g p) {
    const float q = rintf(__fdiv_rn(__fsub_rn(tf_glm_f32(x), tf_glm_f32(p.bias)), tf_glm_f32(p.scale)));
    return q >= 15.0f ? 15u : (q > 0.0f ? static_cast<uint32_t>(q) : 0u);
}

// quantize4's words, written as make_q4 packs them (tf_glm_pack_q4's layout over the quantized rows)
extern "C" __global__ void tf_glm_pack_quant4(const uint16_t* __restrict__ w, int n, int k, uint32_t* __restrict__ out,
                                              long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = static_cast<int>(idx & 1), lane = static_cast<int>((idx >> 1) & 31);
    const int j = static_cast<int>((idx >> 6) & 7), kg = k / 64;
    const long long tg = idx >> 9;
    const int g = static_cast<int>(tg % kg);
    const long long c = (tg / kg) * 64 + j * 8 + (lane >> 2);
    uint32_t word = 0;
    if (c < n) {
        const uint16_t* row = w + c * static_cast<long long>(k);
        const tf_glm_q4g p = tf_glm_group(row + g * 64);
        for (int q = 0; q < 8; ++q) {
            const int in = g * 64 + 32 * v + 2 * (lane & 3) + (q & 3) * 8 + (q >> 2);
            word |= tf_glm_q4(row[in], p) << (4 * q);
        }
    }
    out[idx] = word;
}

// quantize4's scales (which 0) or biases (1), written as make_q4 packs them: [kg][npad], zeros past n
extern "C" __global__ void tf_glm_pack_quant4s(const uint16_t* __restrict__ w, int n, int k, int npad, int which,
                                               uint16_t* __restrict__ out) {
    const int kg = k / 64;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), c = static_cast<int>(idx % npad);
    if (c >= n) {
        out[idx] = 0;
        return;
    }
    const tf_glm_q4g p = tf_glm_group(w + static_cast<long long>(c) * k + g * 64);
    out[idx] = which ? p.bias : p.scale;
}

// qmm.make_f8 (patch 0002, the dense-q4 form's head): bf16 rows [n][k] as e4m3 with an fp32 scale a 128-column block,
// torch's CUDA ops in order: s = max(absmax * fp32(1 / 448), 1e-30) (a CPU scalar's reciprocal, as above), then
// e4m3(clamp(x / s, -448, 448)) rounded to nearest even; NaN propagates (written as c10 writes it, sign kept)
__device__ __forceinline__ float tf_glm_f8_scale(const uint16_t* __restrict__ g) {
    float a = 0.0f;
    for (int i = 0; i < 128; ++i) {
        const float v = fabsf(tf_glm_f32(g[i]));
        a = ((v > a || v != v) && a == a) ? v : a;
    }
    const float s = __fmul_rn(a, 1.0f / 448.0f);
    return (s != s || s > 1e-30f) ? s : 1e-30f;
}

__device__ __forceinline__ uint32_t tf_glm_e4m3(float x) {  // |x| <= 448 or NaN
    const uint32_t u = __float_as_uint(x), sign = (u >> 24) & 0x80u, a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return sign | 0x7fu;
    if (a >= 0x3c800000u) return sign | ((a - (120u << 23) + 0x7ffffu + ((a >> 20) & 1u)) >> 20);  // 2^-6 up: 3 bits
    const uint32_t e = a >> 23;  // under 2^-6: multiples of 2^-9, the significand shifted down and rounded
    if (e < 117u) return sign;
    const uint32_t m = (a & 0x7fffffu) | 0x800000u, r = 141u - e, half = 1u << (r - 1u), rest = m & ((1u << r) - 1u);
    const uint32_t q = m >> r;
    return sign | (q + ((rest > half || (rest == half && (q & 1u))) ? 1u : 0u));
}

// make_f8's bytes, four a thread as a little-endian word: out [n][k / 4]
extern "C" __global__ void tf_glm_pack_f8(const uint16_t* __restrict__ w, int n, int k, uint32_t* __restrict__ out,
                                          long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const long long c = idx / (k / 4);
    const int x = static_cast<int>(idx % (k / 4)) * 4;
    const uint16_t* row = w + c * static_cast<long long>(k);
    const float s = tf_glm_f8_scale(row + (x / 128) * 128);
    uint32_t word = 0;
    for (int j = 0; j < 4; ++j) {
        const float q = __fdiv_rn(tf_glm_f32(row[x + j]), s);
        word |= tf_glm_e4m3(q != q ? q : fminf(fmaxf(q, -448.0f), 448.0f)) << (8 * j);
    }
    out[idx] = word;
}

// make_f8's scales: out [n][k / 128] fp32
extern "C" __global__ void tf_glm_pack_f8s(const uint16_t* __restrict__ w, int n, int k, float* __restrict__ out,
                                           long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = tf_glm_f8_scale(w + (idx / (k / 128)) * static_cast<long long>(k) + (idx % (k / 128)) * 128);
}
