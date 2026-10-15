// SEQ on X1's grouped kernel: the K splits of a gate/up item summed in one program, in today's order (Mia's TF_GLM_EXL3_SEQ design, MiaAI-Lab, recipe patch 0004-glm-prompt-kernels, Apache-2.0).
// X1's device code (exl3/x1/experts_grouped.cuh: IMP2's adaptation of Mia's 0001/0020/0009 onto Ash Hart's universal kernel) is included unchanged; only the head, the split loop and the one plane are new.
#pragma once

#include "../x1/experts_grouped.cuh"

namespace tf_exl3x {

template <int CB, int NT, int W, int PF, int LO, int HI, bool NFIRST, bool ORDER, bool PA, bool XROW>
__global__ void __launch_bounds__(W * 32) grouped_seq_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots, const int* __restrict__ order) {
    int u, mtile, mat;
    if constexpr (ORDER) {
        const int MT = (maxm + 15) / 16;
        const int item = order[NFIRST ? blockIdx.y : blockIdx.x];   // the item's launch slot
        if (item < 0) return;                                     // a slot past the routed items
        u = item / MT;
        mtile = item - u * MT;
        mat = blockIdx.z;                                         // z holds the mat alone: the splits run in here
    } else {
        u = NFIRST ? blockIdx.y : blockIdx.x;
        if (u >= ucount[0]) return;
        const int MT = (maxm + 15) / 16;
        mtile = blockIdx.z % MT;
        mat = blockIdx.z / MT;
    }
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) {
        const int m = mtile * 16 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : (XROW ? r0 / slots : r0)) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : (XROW ? r1 / slots : r1)) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = (NFIRST ? blockIdx.x : blockIdx.y) * NT;     // the n block

    constexpr int OUTS = 16 * NT * 16, PER = (OUTS + W * 32 - 1) / (W * 32);
    float tot[PER];                                       // this thread's outputs: the split sums, from +0 as the epilogue's
#pragma unroll
    for (int q = 0; q < PER; ++q) tot[q] = 0.f;
    __shared__ float red[W][16][NT * 16];
    for (int split = 0; split < SK; ++split) {
        const int kt0 = split * per_split + warp * per_warp;     // today's K range of this split's warp
        float acc[NT][2][4];
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        switch (k2) {
#define TF_EXL3S_CASE(K2_)                                                                                      \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles<CB, K2_, NT, PF, PA>(T, NTILES, kt0, per_warp, nt0, x0, x1, r0 >= 0, r1 >= 0, lane, acc); \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
            TF_EXL3S_CASE(2)
            TF_EXL3S_CASE(3)
            TF_EXL3S_CASE(4)
            TF_EXL3S_CASE(5)
            TF_EXL3S_CASE(6)
            TF_EXL3S_CASE(7)
            TF_EXL3S_CASE(8)
            TF_EXL3S_CASE(9)
            TF_EXL3S_CASE(10)
            TF_EXL3S_CASE(11)
            TF_EXL3S_CASE(12)
            TF_EXL3S_CASE(13)
            TF_EXL3S_CASE(14)
            TF_EXL3S_CASE(15)
            TF_EXL3S_CASE(16)
#undef TF_EXL3S_CASE
            default:
                __trap();
        }

        if (split > 0) __syncthreads();                   // the last split's sums are done reading red
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[i][h][0];
                red[warp][g][col + 1] = acc[i][h][1];
                red[warp][g + 8][col] = acc[i][h][2];
                red[warp][g + 8][col + 1] = acc[i][h][3];
            }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < PER; ++q) {
            const int idx = threadIdx.x + q * W * 32;     // today's output of this thread
            if (idx >= OUTS) break;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            if (rows_sh[row] < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];      // today's warp order
            tot[q] += s;                                  // today's split order, from +0
        }
    }
#pragma unroll
    for (int q = 0; q < PER; ++q) {
        const int idx = threadIdx.x + q * W * 32;
        if (idx >= OUTS) break;
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        if (r < 0) continue;
        Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[q];     // one plane: the epilogue runs at SK = 1
    }
}

}  // namespace tf_exl3x
