// qualification_8724e161 (qualification_7af419c1's qualification_babb0a2d): bertholomus's prompt-chunk kernels for the halves, on X1's unchanged device code: grouped_mma_kernel
// (64 member rows a program, each weight tile decoded once into shared memory as mma B fragments, the K splits folded)
// and work_list_kernel (the (expert place, 64-row block) pairs a launch walks). Copied from bertholomus/TensorFold
// deepseek-v41-tp2 90b68e0 (Albert Lee, BertholomusAI, Apache-2.0; his commits name Claude Opus 5.5 as co-author):
// src/tensorfold/cuda/exl3/experts_grouped.cuh:521-760 and experts.cu:146-197, his comments kept. One change, marked
// "qualification_8724e161": XROW (X1's rowrot, Mia's 0009) loads a member's row r of rows rotated once (X + r K) instead of its slot's
// (X + p K), as X1's grouped_kernel does (exl3/x1/experts_grouped.cuh:234). X1's helpers (exl3/x1/experts_grouped.cuh:
// IMP2's adaptation of Mia's 0001/0020/0009 onto Ash Hart's universal kernel) equal his (evidence/qualification_8724e161/scans).
#pragma once

#include "../x1/experts_grouped.cuh"

namespace tf_exl3x {

// Prompt chunks with the weights shared through shared memory: a program per (64 member rows, 64 columns, expert u,
// matrix), eight warps: warp w runs member rows 16 (w % 4) .. + 15 against columns 32 (w / 4) .. + 31. The k tiles go
// in steps of MMA_KB (4): during a step the warps decode the next step's sixteen 16 x 16 weight tiles (two each) into
// shared memory as mma B fragments, while the step's mma read the fragments decoded during the step before (two
// buffers, one barrier a step), so a weight is decoded once for up to 64 rows (grouped_rows: 32). The rows come through
// a ring of shared-memory stages (cp.async, MMA_STAGES - 1 steps ahead, ldmatrix), the trellis words two steps ahead.
// (Measured on GB10, 2,048 rows: 2 k tiles a step or 128 rows a program were slower; ncu showed the kernel latency- and
// issue-bound, now close to its weight and row traffic.)
// Every row keeps grouped_kernel's bits: the K range splits into SK x WK chains, each an mma chain from zero over the
// same k tiles in the same order; the WK chains of a split are added in order; with FOLD the splits are added in order
// from 0 (the gate/up epilogue's order: it then reads one split), without (SK 1) the split's sum is written as is.
constexpr int MMA_ROWS = 64;        // member rows a program
constexpr int MMA_COLS = 64;        // columns a program
constexpr int MMA_KB = 4;           // k tiles a step (it divides every chain: 24 for GLM's gate/up, 8 for its down);
                                    // chains it does not divide (a TP2 DeepSeek-V4.1 rank's down: 18) take 2 a step
constexpr int MMA_STAGES = 3;       // row stages in flight
constexpr int MMA_LD = MMA_KB * 16 + 8;     // halfs a staged row (padded: ldmatrix rows fall in distinct banks)
constexpr int MMA_DEC = MMA_KB / 2;         // weight tiles a warp decodes a step (8 warps, 4 n16 tiles a k tile)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&a)[4], const void* smem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"(s));
}

template <int CB, int K2, bool FOLD, int KB, bool XROW>
__device__ __forceinline__ void mma_slice(const uint32_t* __restrict__ T, int NTILES, int KT, int per_range, int WK,
                                          int nt0, const half* __restrict__ X, int K, const int* rows_sh, bool live,
                                          int warp, int lane, half (&As)[MMA_STAGES][MMA_ROWS][KB * 16 + 8],
                                          uint4 (&bsh)[2][KB][4][32], float (&tot)[4][4], int slots) {
    constexpr int MMA_DEC_ = KB / 2;                       // weight tiles a warp decodes a step
    constexpr int LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * Fmt<K2>::TW;
    const int dk = (warp >> 2) * MMA_DEC_, dn = warp & 3;      // the tiles this warp decodes: k tiles dk .. of a step, n16 dn
    const uint32_t* tp = T + ((size_t)dk * NTILES + nt0 + dn) * Fmt<K2>::TW + lane;
    const int rg = warp & 3, cg = warp >> 2;                  // its rows 16 rg .., its n16 tiles 2 cg, 2 cg + 1
    const int NS = KT / KB;
    // the staged chunks this thread copies: row tid / 4, 16 bytes (tid % 4) + 4 c of the step's KB * 16 columns
    constexpr int CPR = KB / 2;                           // chunks a thread a step
    const int crow = threadIdx.x >> 2, cpart = threadIdx.x & 3;
    const int cr = rows_sh[crow];
    // qualification_8724e161: with XROW the member's row of rows rotated once (r = p / slots), as X1's grouped_kernel loads it
    const half* src = X + (size_t)(cr < 0 ? 0 : (XROW ? cr / slots : cr)) * K + cpart * 8;
    // ldmatrix: lane l reads row (l & 7) + 8 ((l >> 3) & 1) of the warp's 16, columns 8 (l >> 4) of a k tile
    const int lrow = rg * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

#pragma unroll
    for (int st = 0; st < MMA_STAGES - 1; ++st) {
        if (st < NS)
#pragma unroll
            for (int c = 0; c < CPR; ++c)
                cp_async16(&As[st][crow][(cpart + 4 * c) * 8], src + st * KB * 16 + 32 * c, cr >= 0);
        cp_async_commit();
    }
    uint32_t words[MMA_DEC_][LW];
#pragma unroll
    for (int d = 0; d < MMA_DEC_; ++d) load_words<K2>(words[d], tp + (size_t)d * kstride, lane);
#pragma unroll
    for (int d = 0; d < MMA_DEC_; ++d) {
        uint32_t b0[2], b1[2];
        decode_tile<CB, K2>(words[d], map, lane, b0, b1);
        bsh[0][dk + d][dn][lane] = make_uint4(b0[0], b0[1], b1[0], b1[1]);
    }
    if (NS > 1)
#pragma unroll
        for (int d = 0; d < MMA_DEC_; ++d) load_words<K2>(words[d], tp + (size_t)(KB + d) * kstride, lane);
    cp_async_wait<MMA_STAGES - 2>();
    __syncthreads();

    float acc[4][4], sacc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[i][c] = sacc[i][c] = 0.f;
    int left = per_range, wk = 0;                             // k tiles left in this chain; its place in the split

    for (int st = 0; st < NS; ++st) {
        const int buf = st & 1, stage = st % MMA_STAGES;
        {                                                     // the rows of the step MMA_STAGES - 1 ahead
            const int ahead = st + MMA_STAGES - 1;
            if (ahead < NS)
#pragma unroll
                for (int c = 0; c < CPR; ++c)
                    cp_async16(&As[ahead % MMA_STAGES][crow][(cpart + 4 * c) * 8],
                               src + (size_t)ahead * KB * 16 + 32 * c, cr >= 0);
            cp_async_commit();
        }
        uint32_t fr[MMA_DEC_][4];                              // the next step's tiles (their words came a step ago)
        if (st + 1 < NS) {
#pragma unroll
            for (int d = 0; d < MMA_DEC_; ++d) {
                uint32_t b0[2], b1[2];
                decode_tile<CB, K2>(words[d], map, lane, b0, b1);
                fr[d][0] = b0[0]; fr[d][1] = b0[1]; fr[d][2] = b1[0]; fr[d][3] = b1[1];
            }
            if (st + 2 < NS)
#pragma unroll
                for (int d = 0; d < MMA_DEC_; ++d)
                    load_words<K2>(words[d], tp + (size_t)((st + 2) * KB + d) * kstride, lane);
        }
        if (live) {
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                uint32_t a[4];
                ldmatrix_x4(a, &As[stage][lrow][kk * 16 + lcol]);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint4 b = bsh[buf][kk][2 * cg + j][lane];
                    const uint32_t bl[2] = {b.x, b.y}, bh[2] = {b.z, b.w};
                    mma16816(acc[2 * j], a, bl);
                    mma16816(acc[2 * j + 1], a, bh);
                }
            }
        }
        left -= KB;
        if (left == 0) {                                      // a chain ends: into its split's sum, in order
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sacc[i][c] = wk == 0 ? acc[i][c] : sacc[i][c] + acc[i][c];
                    acc[i][c] = 0.f;
                }
            left = per_range;
            if (++wk == WK) {                                 // a split ends
                wk = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int c = 0; c < 4; ++c) tot[i][c] = FOLD ? tot[i][c] + sacc[i][c] : sacc[i][c];
            }
        }
        if (st + 1 < NS)
#pragma unroll
            for (int d = 0; d < MMA_DEC_; ++d)
                bsh[buf ^ 1][dk + d][dn][lane] = make_uint4(fr[d][0], fr[d][1], fr[d][2], fr[d][3]);
        cp_async_wait<MMA_STAGES - 2>();                     // this thread's copies for the next step have landed
        __syncthreads();
    }
}

template <int CB, int LO, int HI, bool FOLD, int KB, bool XROW = false>
__global__ void __launch_bounds__(256) grouped_mma_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int WK, int maxm, int slots, int nexp,
    const int* __restrict__ work) {
    // the work list (as grouped_rows_kernel's): grid.z walks (expert place, 64-row member block) pairs
    const int zz = (int)(blockIdx.z % nexp);
    const int u = work ? work[2 * zz] : zz;
    if (u >= ucount[0]) return;
    const int mb = work ? work[2 * zz + 1] : (int)blockIdx.x;
    const int mat = (int)(blockIdx.z / nexp);
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g8 = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;
    const int per_range = KT / (SK * WK);

    __shared__ int rows_sh[MMA_ROWS];
    // the staged rows and two steps' B fragments ([step][k tile][n16][lane]): static up to 4 k tiles a step, dynamic
    // at 6 (64.5 KB, past the 48 KB of static shared memory; the launcher sets the attribute)
    using AsT = half[MMA_STAGES][MMA_ROWS][KB * 16 + 8];
    using BsT = uint4[2][KB][4][32];
    constexpr bool DYN = KB > 4;
    __shared__ __align__(16) half As_s[DYN ? 1 : MMA_STAGES][DYN ? 1 : MMA_ROWS][DYN ? 8 : KB * 16 + 8];
    __shared__ uint4 bsh_s[DYN ? 1 : 2][DYN ? 1 : KB][4][32];
    extern __shared__ __align__(16) unsigned char mma_dyn[];
    AsT& As = DYN ? *reinterpret_cast<AsT*>(mma_dyn) : *reinterpret_cast<AsT*>(&As_s[0][0][0]);
    BsT& bsh = DYN ? *reinterpret_cast<BsT*>(mma_dyn + sizeof(AsT)) : *reinterpret_cast<BsT*>(&bsh_s[0][0][0][0]);
    for (int i = threadIdx.x; i < MMA_ROWS; i += 256) {
        const int m = mb * MMA_ROWS + i;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[i] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                               // members come first, so this block is empty
    const int rg = warp & 3, cg = warp >> 2;
    const int r0 = rows_sh[rg * 16 + g8], r1 = rows_sh[rg * 16 + g8 + 8];
    const bool live = rows_sh[rg * 16] >= 0;                  // the warp has rows (they are a prefix)
    const int nt0 = blockIdx.y * (MMA_COLS / 16);

    float tot[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int c = 0; c < 4; ++c) tot[i][c] = 0.f;

    switch (k2) {
#define TF_EXL3X_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            mma_slice<CB, K2_, FOLD, KB, XROW>(T, NTILES, KT, per_range, WK, nt0, X, K, rows_sh, live, warp, lane, \
                                           As, bsh, tot, slots);                                                \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        TF_EXL3X_CASE(2)
        TF_EXL3X_CASE(3)
        TF_EXL3X_CASE(4)
        TF_EXL3X_CASE(5)
        TF_EXL3X_CASE(6)
        TF_EXL3X_CASE(7)
        TF_EXL3X_CASE(8)
        TF_EXL3X_CASE(9)
        TF_EXL3X_CASE(10)
        TF_EXL3X_CASE(11)
        TF_EXL3X_CASE(12)
        TF_EXL3X_CASE(13)
        TF_EXL3X_CASE(14)
        TF_EXL3X_CASE(15)
        TF_EXL3X_CASE(16)
#undef TF_EXL3X_CASE
        default:
            __trap();
    }
    if (!live) return;
    // Z [mats, 1, P, N]: rows g8 and g8 + 8 of the warp's 16, columns 8 i + 2 t of its 32
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = nt0 * 16 + cg * 32 + i * 8 + 2 * t;
        if (r0 >= 0)
            *reinterpret_cast<float2*>(Z + ((size_t)mat * P + r0) * N + col) = make_float2(tot[i][0], tot[i][1]);
        if (r1 >= 0)
            *reinterpret_cast<float2*>(Z + ((size_t)mat * P + r1) * N + col) = make_float2(tot[i][2], tot[i][3]);
    }
}

}  // namespace tf_exl3x

namespace tf_exl3_experts {

// Prompt chunks' work lists (experts.GROUP_LIST): the (expert place, member group) pairs of the used places, groups of
// rows0 (rows1) member rows, in place order and in group order within a place (the order the grids ran them in), then
// sentinel pairs (a place past every used one: its programs return at once) up to each list's length n0 (n1), a bound
// the host takes from the shape alone. One block; no host read of the lists' sizes.
constexpr int LIST_THREADS = 1024;
constexpr int LIST_NONE = 0x7fffffff;

__global__ void __launch_bounds__(LIST_THREADS) work_list_kernel(const int* __restrict__ counts,
                                                                 const int* __restrict__ uids,
                                                                 const int* __restrict__ ucount, int maxu,
                                                                 int* __restrict__ work0, int rows0, int n0,
                                                                 int* __restrict__ work1, int rows1, int n1) {
    __shared__ int warp_tot[LIST_THREADS / 32];
    const int nu = ucount[0];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int l = 0; l < (work1 ? 2 : 1); ++l) {
        int* work = l ? work1 : work0;
        const int rows = l ? rows1 : rows0, n = l ? n1 : n0;
        int base = 0;
        for (int start = 0; start < maxu; start += LIST_THREADS) {
            const int p = start + threadIdx.x;
            const int g = (p < maxu && p < nu) ? (counts[uids[p]] + rows - 1) / rows : 0;
            int inc = g;                              // inclusive scan of the groups in thread (= place) order
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, inc, o);
                if (lane >= o) inc += v;
            }
            if (lane == 31) warp_tot[warp] = inc;
            __syncthreads();
            int prior = 0, total = 0;
            for (int w = 0; w < LIST_THREADS / 32; ++w) {
                prior += w < warp ? warp_tot[w] : 0;
                total += warp_tot[w];
            }
            const int first = base + prior + inc - g;
            for (int j = 0; j < g; ++j) {
                const int q = first + j;
                if (q < n) {                          // (always: n bounds every list the counts allow)
                    work[2 * q] = p;
                    work[2 * q + 1] = j;
                }
            }
            base += total;
            __syncthreads();                          // warp_tot is rewritten by the next round
        }
        for (int q = base + threadIdx.x; q < n; q += LIST_THREADS) {
            work[2 * q] = LIST_NONE;
            work[2 * q + 1] = 0;
        }
    }
}

}  // namespace tf_exl3_experts
