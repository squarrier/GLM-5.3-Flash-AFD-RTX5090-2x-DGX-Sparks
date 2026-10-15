// glm53f-afd's large-M expert kernels in TensorFold's AFD expert node (TF_GLM_EXPERT_KERNEL=g53): one TP2 half's
// routed experts of a prompt chunk, GLM-5.3-Flash, EXL3 4-bit "mcg" trellises.
//
// Expert prompt kernels from glm53f-afd, written by Hugh Madden (@dangerm00se, github.com/hughmadden),
// glm53f-afd @ 91db3cc (https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6),
// crates/glm53f-rank/kernels/exl3_rank.cu: its plan, its large-M gate/up and down kernels, its fused epilogue and
// reduce, its persistent launch and its group order. MIT License, Copyright (c) 2026 Turquoise Bay AI Pty Ltd
// (LICENSES/glm53f-afd-MIT.txt). The file's EXL3 decoder (mcg2, decode_tile), fragment MMA (mma16816) and Hadamard
// butterfly (fwht128) come from TensorFold by Ash Hart (@ashxhart, github.com/ashhart), TensorFold @ bb4b4a3, MIT
// License, Copyright (c) 2026 TensorFold contributors (glm53f-rank's LICENSE.tensorfold, kept as
// LICENSES/glm53f-afd-LICENSE.tensorfold.txt). Its TP split of every expert by intermediate channel follows glmrt by
// T.J. Purtell (@wrldsuksgo2mars, github.com/tpurtell), a design reused there with no code copied. Its once-a-row
// rotation (rotrows_kernel, gateup_big's XH) and its group order (order_kernel) were reimplemented for glm53f-afd from
// MiaAI-Lab's TensorFold GLM-5.3-Flash recipe patches 0009 (ROWROT/SHX) and 0020 (ORDER), Apache-2.0, with no code
// copied. The rank width of 1,024 channels (two ranks) and those two schedules are glm53f-afd at 91db3cc with two
// evaluation patches applied, one for a two-rank build and one for the row rotation and group order (each kept in the
// code's own form below); see THIRD_PARTY_NOTICES.md, "GLM expert prompt kernels from glm53f-afd".
//
// Ported here (TensorFold side), data movement only; every product, sum, rounding and its order is glm53f-rank's:
// - weights: TensorFold's resident layer (Exl3RoutedExperts) in place of glm53f-rank's layer image: each expert's
//   gate, up and down trellis through the layer's pointer tables (the same tile bytes: [K/16][N/16][64] int16, the
//   half's columns of gate and up, rows of down), and the stacked suh / svh vectors; no second copy of any weight;
// - inputs: BF16 rows (the expert node's wire rows, any row stride) in place of FP8 E4M3 rows with UE8M0 scales:
//   row_rotate widens a BF16 value to FP32 where glm53f-rank widens an E4M3 value and multiplies its scale, then x
//   suh, the butterfly in FP32, x 1/sqrt(128) and FP16 as there (a row whose values are E4M3 values times their scale
//   gives the same FP32 products, so the same bits); a NaN input raises the row fault as a NaN code does there;
// - routes: the node's pick / wts [rows][slots] (the 8 routed slots; slot 8, the shared expert, is not read) in place
//   of ids / weights [rows * 8];
// - output: FP32 rows (glm53f-rank's g53r_ffn_f32: the reduce's sums before their BF16 rounding) into the caller's
//   [rows][4096] buffer, which the node returns as its half, as before;
// - host side: no uploads, downloads or host sync; the caller's scratch tensors and torch's current stream. An
//   invalid route (glm53f-rank's F_ROUTE: an id outside the layer, a negative or non-finite weight, an id twice in a
//   row) leaves the plan empty as there, and the plan then writes NaN into the call's output rows (glm53f-rank's call
//   returns an error instead);
// - scope: the large-M kernels at the TP2 width only (gate/up rotation warps, gp 2, the split kernels, the plan in the
//   split blocks and the dependent launch are not ported: those serve glm53f-rank's decode sizes).
//
// The arithmetic (glm53f-rank's own description): every output depends only on its own row: the tensor cores keep
// rows independent, and every sum (K splits, the 8 slots) runs in a fixed order. With the same K splits (sk, skd) and
// SwiGLU a row gets the same bits whatever else is in the batch; the rows per group, the tilings, the plan, the
// fusion, the L2 policies, the block order do not change a bit. Build with --fmad=false (the epilogues' separate
// multiplies and adds). The large-M defaults are one K split each (sk 1, skd 1): one FP32 MMA chain a sum over
// ascending k tiles.
//
// Zig engine (TensorFold zig/kernels/cuda/g53/, qualification_1b9bf461): the device code of python TF-AFD 2.15's
// families/glm5_next/cuda/g53_rank.cu (squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks 2.15, patch 0034), unchanged and
// split at function boundaries by zig/tests/cuda/g53_copies.py: this file (its lines 1-427 less the host header's
// include), rank_kernels.cuh (lines 428-885), rank.cu (the kernels its host launcher instantiated). The launcher
// (launch_persistent, g53_ffn_launch) is zig/src/families/glm/cuda_g53.zig.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace g53 {

constexpr int H = 4096;      // model width: gate/up K, down N
constexpr int WID = 1024;    // the half's width: gate/up N, down K (two ranks: G53R_WID 1,024)
constexpr int CB = WID / 512;  // 512-column blocks of a gate/up slice: a large-M item covers one
constexpr int NB = WID / 128;  // 128-wide Hadamard blocks of the rank width (epilogue items per pair)
constexpr int NTG = WID / 16;  // n tiles of a gate/up slice, k tiles of a down slice
constexpr int LOG_NB = 3;      // log2 NB
constexpr int LOG_2CB = 2;     // log2 (2 CB): (matrix, column block) pairs
constexpr int LOG_LPP = 5;     // log2 (WID / 32)
constexpr int E = 288;       // routed experts per layer
constexpr int TOPK = 8;
constexpr int MAX_ROWS = 4096;
constexpr float HAD_SCALE = 0.08838834764831845f;  // 1 / sqrt(128)
static_assert(E % 32 == 0, "the plan scans the experts a warp at a time");

// Fault bits (OR-ed into one word).
constexpr unsigned F_ROUTE = 1u;   // bad expert id, weight or a duplicate id in a row
constexpr unsigned F_ROW = 2u;     // a NaN value in an input row (glm53f-rank: an E4M3 NaN code or a UE8M0 NaN scale)
constexpr unsigned F_GATEUP = 4u;  // a non-finite gate/up output
constexpr unsigned F_OUT = 8u;     // a non-finite output sum

struct Group {
    int expert;  // expert id
    int start;   // first grouped pair
    int count;   // pairs (<= 16 * MT)
    int pad;
};

// ---- TensorFold exl3.cu (bb4b4a3), verbatim (as glm53f-rank carries them) -----------------------------------

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// ---- end of the TensorFold functions -------------------------------------------------------------------------

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// The A fragment of an m16n8k16 MMA (rows `r0..r0+15`, columns `c0..c0+15` of a row-major FP16 tile in shared
// memory) with one ldmatrix: lane L gives the address of row (L & 7) + 8 ((L >> 3) & 1), column 8 (L >> 4).
// The same registers as four 32-bit loads (a[0] rows g, cols 2t; a[1] rows g + 8; a[2], a[3] cols + 8).
__device__ __forceinline__ void ldsm_a(uint32_t (&a)[4], const half* p) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// A trellis word through L2 with an evict-first policy (`pol` from `evict_first_policy`): streamed weights then
// give way to the partial sums and down inputs that are read again soon. Plain `__ldg` otherwise.
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t pol = 0;
#if __CUDA_ARCH__ >= 800
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
#endif
    return pol;
}

__device__ __forceinline__ uint32_t ldg_w(const uint32_t* p, bool hint, uint64_t pol) {
#if __CUDA_ARCH__ >= 800
    if (hint) {
        uint32_t v;
        asm("ld.global.nc.L2::cache_hint.b32 %0, [%1], %2;\n" : "=r"(v) : "l"(p), "l"(pol));
        return v;
    }
#endif
    return __ldg(p);
}

// Drop one 128-byte line of scratch from L2 without writing it back (its contents become undefined).
__device__ __forceinline__ void discard_line(const void* p) {
#if __CUDA_ARCH__ >= 800
    asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
#endif
}

// Everything a call's kernels read and write (passed by value to every kernel).
// TF port: the layer is TensorFold's resident layer (pointer tables and stacked scale vectors), the rows are BF16
// with a row stride, the routes are the node's pick / wts with a row stride, and the output is FP32.
struct Args {
    const int64_t* gate_t;   // per expert: its gate trellis [256][64][64] int16 (the half's 64 n tiles)
    const int64_t* up_t;     // per expert: its up trellis, the same shape
    const int64_t* down_t;   // per expert: its down trellis [64][256][64] int16 (the half's 64 k tiles)
    const half* suh_g;       // [E][H]
    const half* suh_u;       // [E][H]
    const half* svh_g;       // [E][WID]
    const half* svh_u;       // [E][WID]
    const half* suh_d;       // [E][WID]
    const half* svh_d;       // [E][H]
    const __nv_bfloat16* x;  // rows [rows][xs] BF16
    int xs;                  // row stride of x (elements)
    const int32_t* pick;     // [rows][ps]: slots 0..7 the routed experts
    const float* wts;        // [rows][ws]
    int ps, ws;              // row strides of pick and wts (elements)
    const Group* groups;
    const int* counts;       // counts[0]: groups the plan formed (0: invalid routes)
    const int* pair_row;     // per grouped pair: its row
    const int* pair_route;   // per grouped pair: its route (row * 8 + slot)
    const int* inverse;      // per route: its grouped pair
    float* z;                // gate/up split partials [2][sk][P][WID]
    half* xd;                // down input [P][WID]
    float* zd;               // down split partials [skd][P][4096]
    float* out;              // [rows][4096] FP32
    int* gcnt;               // fused epilogue: arrivals per group
    int* rcnt;               // fused reduce: arrivals per (row, output chunk)
    int* work;               // large-M kernels: the next item of gate/up [0] and down [1] (zero at the start)
    unsigned* fault;
    int P, rows, sk, skd;
    int bf16;                // BF16 SwiGLU (the default) or FP32
    int fused;               // epilogue and reduce fused into gate/up and down
    int discard;             // drop consumed partial sums from L2
    int nch;                 // down output chunks (4,096 / chunk width)
    int l2hint;              // trellis words loaded evict-first
    float limit;             // swiglu_limit (10 for GLM-5.3-Flash, glm53f-rank's LIMIT)
    const half* xh;          // rowrot: every row rotated once [rows][4096] FP16 (large-M gate/up), or null
    const int* gorder;       // order: the large-M kernels' group launch order [groups], or null (plan order)
};

// ---- the arithmetic outside the matrix products, shared by every kernel ---------------------------------------

// A lane's inputs for rotating one row's 128-wide K block `kb`: 4 values and 4 suh values, loaded ahead of
// `row_rotate` so that several rows' loads are in flight at once.
// TF port: 4 BF16 values (8 bytes) in place of 4 E4M3 codes and their UE8M0 scale byte.
struct RowIn {
    uint2 q;
    uint2 su;
};

__device__ __forceinline__ RowIn row_load(const __nv_bfloat16* __restrict__ xr, const half* __restrict__ suh, int kb,
                                          int lane) {
    RowIn r;
    r.q = *reinterpret_cast<const uint2*>(xr + kb * 128 + lane * 4);
    r.su = *reinterpret_cast<const uint2*>(suh + kb * 128 + lane * 4);
    return r;
}

// The row block rotated for the tensor cores: widened to FP32, x suh, the butterfly in FP32, x 1/sqrt(128), FP16.
// Lane L gets values 4L..4L+3. `bad` collects NaN values.
// TF port: a BF16 value widened to FP32 (exact) where glm53f-rank computes e4m3(code) * scale (exact).
__device__ __forceinline__ uint2 row_rotate(const RowIn& in, int lane, bool& bad) {
    const half2 s01 = *reinterpret_cast<const half2*>(&in.su.x), s23 = *reinterpret_cast<const half2*>(&in.su.y);
    const float su[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
    const __nv_bfloat162 q01 = *reinterpret_cast<const __nv_bfloat162*>(&in.q.x);
    const __nv_bfloat162 q23 = *reinterpret_cast<const __nv_bfloat162*>(&in.q.y);
    const float q[4] = {__low2float(q01), __high2float(q01), __low2float(q23), __high2float(q23)};
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        bad = bad || isnan(q[j]);
        v[j] = q[j] * su[j];
    }
    fwht128(v, lane);
    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    return make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
}

__device__ __forceinline__ uint2 rotate_row(const __nv_bfloat16* __restrict__ xr, const half* __restrict__ suh, int kb,
                                            int lane, bool& bad) {
    return row_rotate(row_load(xr, suh, kb, lane), lane, bad);
}

// Four FP16 values (8 bytes) as floats.
__device__ __forceinline__ void half4(uint2 h, float (&f)[4]) {
    const half2 h01 = *reinterpret_cast<const half2*>(&h.x), h23 = *reinterpret_cast<const half2*>(&h.y);
    f[0] = __low2float(h01);
    f[1] = __high2float(h01);
    f[2] = __low2float(h23);
    f[3] = __high2float(h23);
}

// The gate/up epilogue of one (pair gp, 128-block blk), one warp (after TensorFold's gateup_epilogue_kernel): the
// splits summed in order, rotated back, x svh, GLM's clamped SwiGLU, x down suh, rotated, FP16 into Xd. `e` is
// the pair's expert. Returns whether a gate/up sum was not finite. The scale vectors are loaded with the first
// splits' partial sums, not one value at a time after them.
__device__ __forceinline__ bool epilogue_item(const Args& a, int e, int gp, int blk, int lane) {
    const int n = blk * 128 + lane * 4;
    const uint2 sg = *reinterpret_cast<const uint2*>(a.svh_g + size_t(e) * WID + n);
    const uint2 su = *reinterpret_cast<const uint2*>(a.svh_u + size_t(e) * WID + n);
    const uint2 sd = *reinterpret_cast<const uint2*>(a.suh_d + size_t(e) * WID + n);
    float gv[4] = {0.f, 0.f, 0.f, 0.f}, uv[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < a.sk; ++s) {
        const float4 zg = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t(0 * a.sk + s) * a.P + gp) * WID + n));
        const float4 zu = __ldcg(reinterpret_cast<const float4*>(a.z + (size_t(1 * a.sk + s) * a.P + gp) * WID + n));
        gv[0] += zg.x; gv[1] += zg.y; gv[2] += zg.z; gv[3] += zg.w;
        uv[0] += zu.x; uv[1] += zu.y; uv[2] += zu.z; uv[3] += zu.w;
    }
    float svh_g[4], svh_u[4], suh_d[4];
    half4(sg, svh_g);
    half4(su, svh_u);
    half4(sd, suh_d);
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(gv[j]) || !isfinite(uv[j]);
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = gv[j] * HAD_SCALE * svh_g[j];
        float uu = uv[j] * HAD_SCALE * svh_u[j];
        float act;
        if (a.bf16) {  // the reference model's BF16 SwiGLU (TensorFold's default)
            gg = fminf(bf16r(gg), a.limit);
            uu = fminf(fmaxf(bf16r(uu), -a.limit), a.limit);
            act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        } else {       // FP32 throughout (an option for the KL gate and for tight kernel tests)
            gg = fminf(gg, a.limit);
            uu = fminf(fmaxf(uu, -a.limit), a.limit);
            act = gg / (1.f + expf(-gg)) * uu;
        }
        v[j] = act * suh_d[j];
    }
    fwht128(v, lane);
    const half2 h0 = __floats2half2_rn(v[0] * HAD_SCALE, v[1] * HAD_SCALE);
    const half2 h1 = __floats2half2_rn(v[2] * HAD_SCALE, v[3] * HAD_SCALE);
    *reinterpret_cast<uint2*>(a.xd + size_t(gp) * WID + n) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
    return bad;
}

// The route reduce of one (row, 128-block blk), one warp: out[row] = sum over the 8 slots, in slot order, of
// w * (rot(sum of the down splits) * svh), in FP32 (glm53f-rank's FP32 output: the sums before its BF16 rounding).
// Every slot's inputs are loaded before the first is used. Returns whether a sum was not finite.
__device__ __forceinline__ bool reduce_item(const Args& a, int row, int blk, int lane) {
    const int n = blk * 128 + lane * 4;
    int gp[TOPK];
    float wt[TOPK];
    uint2 sv[TOPK];
    float4 qualification_qualification_f8e8e742[TOPK];
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
        const int r = row * TOPK + s;
        gp[s] = a.inverse[r];
        wt[s] = a.wts[size_t(row) * a.ws + s];
        sv[s] = *reinterpret_cast<const uint2*>(a.svh_d + size_t(a.pick[size_t(row) * a.ps + s]) * H + n);
    }
#pragma unroll
    for (int s = 0; s < TOPK; ++s) qualification_qualification_f8e8e742[s] = __ldcg(reinterpret_cast<const float4*>(a.zd + size_t(gp[s]) * H + n));
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < TOPK; ++s) {
        float v[4] = {0.f + qualification_qualification_f8e8e742[s].x, 0.f + qualification_qualification_f8e8e742[s].y, 0.f + qualification_qualification_f8e8e742[s].z, 0.f + qualification_qualification_f8e8e742[s].w};
        for (int k = 1; k < a.skd; ++k) {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(a.zd + (size_t(k) * a.P + gp[s]) * H + n));
            v[0] += t.x; v[1] += t.y; v[2] += t.z; v[3] += t.w;
        }
        fwht128(v, lane);
        const half2 s01 = *reinterpret_cast<const half2*>(&sv[s].x), s23 = *reinterpret_cast<const half2*>(&sv[s].y);
        const float svh[4] = {__low2float(s01), __high2float(s01), __low2float(s23), __high2float(s23)};
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[j] = acc[j] + wt[s] * (v[j] * HAD_SCALE * svh[j]);
    }
    bool bad = false;
#pragma unroll
    for (int j = 0; j < 4; ++j) bad = bad || !isfinite(acc[j]);
    *reinterpret_cast<float4*>(a.out + size_t(row) * H + n) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    return bad;
}

// Fused epilogue: after a gate/up block has stored its partial sums, the last of the group's 2 * CB * SK blocks
// (both matrices, both column blocks, every split) runs the epilogue for all of the group's pairs, reading the
// others' partials from L2 (`last` is a shared flag). It also clears the group's counter for the next call
// (counters are zeroed when allocated and every arrival of a call comes before its last). Called by every thread.
template <int NW>
__device__ __forceinline__ void gateup_tail(const Args& a, const Group& G, int g, int* last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        *last = atomicAdd(&a.gcnt[g], 1) == 2 * CB * a.sk - 1;
        if (*last) a.gcnt[g] = 0;
    }
    __syncthreads();
    if (!*last) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bool bad = false;
    for (int item = warp; item < G.count * NB; item += NW)
        bad = epilogue_item(a, G.expert, G.start + (item >> LOG_NB), item & (NB - 1), lane) || bad;
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_GATEUP);
    if (a.discard) {
        __syncthreads();  // every item has read its partials
        constexpr int LPP = WID / 32;  // lines of 128 bytes per pair and (matrix, split): 32 at TP2
        const int lines = 2 * a.sk * G.count * LPP;
        for (int i = threadIdx.x; i < lines; i += NW * 32) {
            const int line = i & (LPP - 1), rest = i >> LOG_LPP, pr = rest % G.count, ms = rest / G.count;
            discard_line(a.z + (size_t(ms) * a.P + G.start + pr) * WID + line * 32);
        }
    }
}

// Fused reduce: after a down block has stored its partial sums (chunk `chunk` of the output, `cwb` 128-blocks
// wide), each (row, chunk) of its group counts one arrival; the block that brings a (row, chunk) to all 8 * SKD
// arrivals (8 slots, every split) reduces it and clears its counter. Thread t holds the row of the group's pair t
// in `my_row` (loaded when the block started; a group has at most 64 pairs), -1 past the group's pairs.
// `list` holds up to the group's pair count, `nl` is shared.
template <int NW>
__device__ __forceinline__ void down_tail(const Args& a, int chunk, int cwb, int my_row, int* list, int* nl) {
    static_assert(NW * 32 >= 64, "a thread per pair of a group of up to 64");
    __threadfence();
    if (threadIdx.x == 0) *nl = 0;
    __syncthreads();
    const int target = TOPK * a.skd;
    if (my_row >= 0 && atomicAdd(&a.rcnt[my_row * a.nch + chunk], 1) == target - 1) {
        a.rcnt[my_row * a.nch + chunk] = 0;
        list[atomicAdd(nl, 1)] = my_row;
    }
    __syncthreads();
    const int nrows = *nl;
    if (nrows == 0) return;
    __threadfence();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bool bad = false;
    for (int item = warp; item < nrows * cwb; item += NW)
        bad = reduce_item(a, list[item / cwb], chunk * cwb + item % cwb, lane) || bad;
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_OUT);
    if (a.discard) {
        __syncthreads();  // every item has read its partials
        const int lines = cwb * 4, per = TOPK * a.skd * lines;  // 4 lines of 128 bytes per 128-block
        for (int i = threadIdx.x; i < nrows * per; i += NW * 32) {
            const int row = list[i / per], rem = i % per, line = rem % lines, ks = rem / lines;
            const int k = ks % a.skd, s = ks / a.skd;
            discard_line(a.zd + (size_t(k) * a.P + a.inverse[row * TOPK + s]) * H + chunk * cwb * 128 + line * 32);
        }
    }
}

}  // namespace g53
