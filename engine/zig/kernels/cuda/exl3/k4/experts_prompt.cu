// Device code of python 2.15's src/tensorfold/cuda/exl3/experts_prompt.cu (tensorfold-afd 0849f3f): lines 1-22, the ATen includes as
// <cstdint>, the anonymous namespace named tf_exl3_k4, lines 28-486; the host launches are cuda_exl3_k4.zig's (qualification_e4ad04cd).
// Ported from MiaAI-Lab's GLM-5.3-Flash EXL3 2x DGX Sparks recipe @ 1576746, patch 0004-glm-prompt-kernels
// (Apache-2.0, Copyright 2026 MiaAI-Lab; THIRD_PARTY_NOTICES.md "GLM EXL3 prompt kernel").
//
// The prompt chunks' routed-expert kernels of her dedicated glm5_next/cuda/exl3.cu, at the end state of her chain
// (0004's kernels with 0009's shared-row inputs and 0020's item order: the recipe's patches 0001-0068 applied to
// TensorFold v0.6.0), on this directory's grouping: the routed experts' outputs Y^T = W^T X^T of a prompt window,
// one fp32 chain of mma over ascending k tiles an output (no split K, no partials in memory), with her gate/up and down
// epilogues fused. A row's bits depend on its own values only (any chunking gives the same prompt state), and they
// are her lane's prompt bits, not the grouped kernel's (which sums split and warp partials): outputs class, so the
// GLM call site takes it only for prompt chunks under TF_GLM_EXL3_PROMPT=1 (off unless set; her default is on).
//
// What is hers, unchanged: the helpers (mcg2, decode_tile_p, mma16816, fwht128, bf16r, cp.async, ldmatrix), the
// pass configuration, prompt_mainloop, prompt_spill, prompt_smem, prompt_order_kernel, the gate/up and down kernels'
// arithmetic and launch (prompt_launch). What the port changes, all data movement (marked "port:"):
//   - a layer's trellises are this directory's per-expert pointers (Exl3RoutedExperts.gate_ptr ..), so a kernel reads
//     its expert's base from the table and runs her main loop on it as expert 0 of a one-expert stack;
//   - members are group()'s codes row * 32 + slot, read as her plan's pair p = row * slots + slot;
//   - prompt_plan_kernel (new) lays group()'s grouping out as her plan's items (expert, first member, count) of up to
//     a pass's members, in expert order, then pass order: the items her plan_kernel makes for the routed experts.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tf_exl3_k4 {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

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

// decode_tile's values with fewer integer instructions (the prompt kernels): the byte-aligned 16-bit states taken
// with one PRMT each and w >> 4 shared by two states. The same states, so the same bits.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
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

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// ---- prompt chunks: Y^T = W^T X^T --------------------------------------------------------------------------------
// A decoded trellis tile is, reordered, an m16n8k16 A fragment of W^T (lane (g, t): a0 = b0[0], a1 = b1[0],
// a2 = b0[1], a3 = b1[1]), so each warp decodes its own tiles straight into registers and every decoded fragment
// feeds the whole pass's member rows (up to 64 tokens, the prefill plan's item) as B fragments from shared memory.
// Each output is one fp32 chain of mma over ascending k tiles (no split K, no partials in memory), so a row's bits
// depend on its own values only (any chunking gives the same prompt state), though they are not the decode
// kernel's (which sums split and warp partials). The epilogues are gateup/down_epilogue_kernel's formulas.
constexpr int PBK = 32;              // k values a shared-memory stage (2 k tiles)
constexpr int PKT = PBK / 16;
constexpr int PSTAGES = 3;
constexpr int PXS = PBK + 8;         // stage row stride in halves (+16 B: ldmatrix rows on distinct banks)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// A block covers 16 m tiles (gate/up: 8 of each matrix = one 128-column Hadamard block; down: 256 columns) for a
// pass of PM members with NW warps: each warp holds MT = 16 / NW m tiles for all PM members (PM / 8 n blocks), so a
// decoded tile feeds PM / 8 mma. PM 64 / 8 warps (two blocks an SM) or PM 128 / 16 warps (one): the same per-element
// chains, so the same bits. Every stage brings the members' next PBK k values and the 16 tiles' trellis words into
// shared memory (cp.async, PSTAGES deep). SHX (gate/up of a layer whose experts share one suh for both): gate and up
// read the same member rows (X0 = X1 = the per-row Xh, the members' rows from xrows), so a stage holds them once.
constexpr int PWORDS = 16 * PKT * 32;          // a stage's trellis words (16 m tiles x PKT k tiles x 32)

template <int MATS, int PM, int NW, bool SHX = false>
struct PromptCfg {
    static constexpr int MT = 16 / NW, NB = PM / 8, T = NW * 32;
    static constexpr int XM = SHX ? 1 : MATS;               // member-row copies a stage holds
    static constexpr size_t STAGE_HALVES = (size_t)XM * PM * PXS + 2 * PWORDS;
};

// Which (matrix, m tile) warp-tile q (0..15) of block cb is.
template <int MATS>
__device__ __forceinline__ void prompt_tile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

template <int MATS, int PM, int NW, bool SHX = false>
__device__ __forceinline__ void prompt_mainloop(const half* __restrict__ X0, const half* __restrict__ X1,
                                                const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1,
                                                const int* rows_sh, int cnt, int e, int K, int NTILES, int cb,
                                                half* stage, float (&acc)[16 / NW][PM / 8][4]) {
    using C = PromptCfg<MATS, PM, NW, SHX>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    prompt_tile<MATS>(warp * C::MT, cb, mat, nt_unused);
    const int KT = K >> 4, chunks = K / PBK;
    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use

#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % PSTAGES) * C::STAGE_HALVES;
        const int k0 = chunk * PBK;
        for (int idx = threadIdx.x; idx < C::XM * PM * (PBK / 8); idx += C::T) {
            const int m = idx / (PM * (PBK / 8)), rem = idx % (PM * (PBK / 8)), i = rem / (PBK / 8),
                      j = rem % (PBK / 8);
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : r) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * PM + i) * PXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)C::XM * PM * PXS);
        for (int idx = threadIdx.x; idx < 16 * PKT * 8; idx += C::T) {       // 16 B pieces of 128-B tile words
            const int q = idx / (PKT * 8), kk = (idx / 8) % PKT, part = idx % 8;
            int wm, nt;
            prompt_tile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * 32 + part * 4;
            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < PSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<PSTAGES - 2>();
        __syncthreads();
        if (c + PSTAGES - 1 < chunks) load(c + PSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % PSTAGES) * C::STAGE_HALVES;
        const half* xs = base + (size_t)(SHX ? 0 : mat) * PM * PXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)C::XM * PM * PXS);
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t a[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile_p(ws[((size_t)(warp * C::MT + i) * PKT + kk) * 32 + lane], lane, b0, b1);
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < PM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldsm_x4(r, xs + (size_t)row * PXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                    const bool upper = jb * 16 + 8 < cnt;         // members 8..15 of the block (else zero rows)
#pragma unroll
                    for (int i = 0; i < C::MT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        if (upper) mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();                                   // the stages become the epilogue's rows
}

// acc (this warp's m tiles, members PM / 2 * hf .. + PM / 2 - 1) -> E[member - PM / 2 * hf][col] fp32, row stride
// ES; col0: the first m tile's first column within the block's E columns.
template <int MT, int PM>
__device__ __forceinline__ void prompt_spill(float* E, int ES, int col0, const float (&acc)[MT][PM / 8][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < PM / 16; ++q) {
            const int nb = hf * (PM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;
template <int PM, int XM = 2>
constexpr size_t prompt_smem() {
    // max of: gate/up stages (XM copies of the member rows), down stages, gate/up epilogue half (2 x PM/2 x GU_ES),
    // down half (PM/2 x DN_ES)
    const size_t gu = (size_t)PSTAGES * (XM * PM * PXS + 2 * PWORDS) * 2;
    const size_t ep = (size_t)2 * (PM / 2) * GU_ES * 4;
    const size_t dn = (size_t)(PM / 2) * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// port (new, not in the recipe): group()'s grouping as MiaAI-Lab's prompt plan. group() leaves the distinct routed
// experts in id order (uids[u], u < ucount) and each one's members in row order (members[u * maxm + m] = row * 32 +
// slot, -1 after the last); her plan_kernel (tensorfold/cuda/experts.cu, place_items) lays a plan out as items
// (expert, first member, count) of up to `pass` members, in expert order, then pass order. These are the same items
// for the routed experts, with `first` indexing this grouping's members (u * maxm + j * pass); her plan also holds the
// shared expert's (id E, last), which her kernels skip. counts = [items, experts]. One block; `cap` (the launch's grid
// y, her max_items) bounds the items.
__global__ void __launch_bounds__(1024) prompt_plan_kernel(const int* __restrict__ uids,
                                                           const int* __restrict__ ucount,
                                                           const int* __restrict__ members, int maxm, int pass,
                                                           int cap, int E, int* __restrict__ items,
                                                           int* __restrict__ counts) {
    __shared__ int wsum[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int U = ucount[0];
    const int per = (U + 1023) / 1024;
    const int lo = min(U, tid * per), hi = min(U, lo + per);
    auto members_of = [&](int u) {                     // members come first, -1 after: find the first -1
        if (uids[u] >= E) return 0;
        const int* m = members + (size_t)u * maxm;
        int a = 0, b = maxm;
        while (a < b) {
            const int c = (a + b) >> 1;
            if (m[c] >= 0) a = c + 1; else b = c;
        }
        return a;
    };
    int t = 0;                                         // this thread's items
    for (int u = lo; u < hi; ++u) t += (members_of(u) + pass - 1) / pass;
    int v = t;                                         // inclusive scan over the block
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int x = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += x;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int total = wsum[31];
    if (total > cap) __trap();                         // cap bounds every grouping's items: never
    int at = (warp ? wsum[warp - 1] : 0) + v - t;      // items before this thread's first expert
    for (int u = lo; u < hi; ++u) {
        const int n = members_of(u);
        for (int j = 0; j * pass < n; ++j, ++at) {
            items[3 * at] = uids[u];
            items[3 * at + 1] = u * maxm + j * pass;
            items[3 * at + 2] = min(pass, n - j * pass);
        }
    }
    if (tid == 0) {
        counts[0] = total;
        counts[1] = U;
    }
}

// The launch order of a prompt plan's items (ORDER): the items of experts that take one item (their weights come
// from DRAM, few mma) and of experts that take several (their passes run together and share the weights through L2:
// mostly mma) merged evenly, each list in plan order, so DRAM and the tensor cores stay busy together instead of in
// turns. Only which block runs which item changes: every output keeps its bits. The shared expert's items (id E, the
// plan's last) are left out: order[slot] = item for slot < the routed items, -1 after. One block.
__global__ void __launch_bounds__(1024) prompt_order_kernel(const int* __restrict__ items,
                                                            const int* __restrict__ counts, int* __restrict__ order,
                                                            int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)               // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;       // heavy items before lo
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {                                // heavy rank r sits at (r + 1/2) / nh; lights before it: strictly
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {                                       // light rank l at (l + 1/2) / nl; heavies at or before it
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}

// SHX: Xg = Xu = Xh [rows, K], one row a token (pair p reads row p / slots); otherwise Xg, Xu [pairs, K].
// port: Tg / Tu are the layer's per-expert trellis pointers; members hold group()'s codes (row * 32 + slot).
template <int PM, int NW, bool SHX>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_gateup_kernel(
    const half* __restrict__ Xg, const half* __restrict__ Xu, const int64_t* __restrict__ Tg,
    const int64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int E, float limit, int slots,
    const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    __shared__ int xrows_sh[SHX ? PM : 1];              // SHX: the members' token rows
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;                                // ORDER leaves the shared expert's items out
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert
    if (threadIdx.x < PM) {
        // port: a member is group()'s code row * 32 + slot; her plan's member is the pair p = row * slots + slot
        const int code = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        const int p = code < 0 ? -1 : (code >> 5) * slots + (code & 31);
        rows_sh[threadIdx.x] = p;
        if (SHX) xrows_sh[threadIdx.x] = p < 0 ? -1 : p / slots;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    // port: the expert's own trellises, read by her main loop as expert 0 of a one-expert stack
    const uint32_t* tg = reinterpret_cast<const uint32_t*>(Tg[e]);
    const uint32_t* tu = reinterpret_cast<const uint32_t*>(Tu[e]);
    prompt_mainloop<2, PM, NW, SHX>(Xg, Xu, tg, tu, SHX ? xrows_sh : rows_sh, cnt, 0, K, N >> 4, cb,
                                    reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    prompt_tile<2>(warp * MT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();                             // the first half's rows are read
    prompt_spill<MT, PM>(Eg + (size_t)mat * PH * GU_ES, GU_ES, col0, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
            uv[j] = Eg[(size_t)(PH + i) * GU_ES + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        fwht128(v, lane);
        const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
        const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&o01);
        packed.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
    }
    }  // halves
}

// port: Td is the layer's per-expert trellis pointers; members hold group()'s codes (row * 32 + slot).
template <int PM, int NW>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_down_kernel(
    const half* __restrict__ Xd, const int64_t* __restrict__ Td, const int* __restrict__ items,
    const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
    float* __restrict__ y, int K, int D, int E, int slots, const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;
    if (threadIdx.x < PM) {
        // port: group()'s code row * 32 + slot as her plan's pair p = row * slots + slot
        const int code = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        rows_sh[threadIdx.x] = code < 0 ? -1 : (code >> 5) * slots + (code & 31);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    // port: the expert's own trellis, read by her main loop as expert 0 of a one-expert stack
    const uint32_t* td = reinterpret_cast<const uint32_t*>(Td[e]);
    prompt_mainloop<1, PM, NW>(Xd, Xd, td, td, rows_sh, cnt, 0, K, D >> 4, cb, reinterpret_cast<half*>(smem_raw),
                               acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();
    prompt_spill<MT, PM>(Ed, DN_ES, warp * MT * 16, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int n = cb * 256 + blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
            fwht128(v, lane);
            const uint2 sv = *reinterpret_cast<const uint2*>(svh_d + (size_t)e * D + n);
            const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
            *reinterpret_cast<float4*>(y + (size_t)p * D + n) =
                make_float4(v[0] * HAD_SCALE * __low2float(s01), v[1] * HAD_SCALE * __high2float(s01),
                            v[2] * HAD_SCALE * __low2float(s23), v[3] * HAD_SCALE * __high2float(s23));
        }
    }
    }  // halves
}

}  // namespace tf_exl3_k4

// The instances python 2.15's exl3x_prompt_cuda launches: prompt_launch<64, 8> (TF_GLM_EXL3_PASS 64, her default) and
// prompt_launch<128, 16> (128, the same bits), each gate/up without and with SHX and down (qualification_e4ad04cd).
#define TF_K4_GATEUP(PM, NW, SHX) template __global__ void tf_exl3_k4::prompt_gateup_kernel<PM, NW, SHX>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const half*, \
    const half*, const half*, half*, int, int, int, float, int, const int*);
#define TF_K4_DOWN(PM, NW) template __global__ void tf_exl3_k4::prompt_down_kernel<PM, NW>( \
    const half*, const int64_t*, const int*, const int*, const int*, const half*, float*, int, int, int, int, const int*);
TF_K4_GATEUP(64, 8, false)
TF_K4_GATEUP(64, 8, true)
TF_K4_GATEUP(128, 16, false)
TF_K4_GATEUP(128, 16, true)
TF_K4_DOWN(64, 8)
TF_K4_DOWN(128, 16)
