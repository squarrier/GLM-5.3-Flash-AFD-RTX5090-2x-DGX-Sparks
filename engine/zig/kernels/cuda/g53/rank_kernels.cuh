// glm53f-afd's large-M expert kernels (Hugh Madden, @dangerm00se, github.com/hughmadden; glm53f-afd @ 91db3cc, MIT
// License, Copyright (c) 2026 Turquoise Bay AI Pty Ltd): python TF-AFD 2.15's g53_rank.cu lines 428-885, unchanged (the
// plan, gate/up, the unfused epilogue, down, the unfused reduce, rotrows and order; zig/tests/cuda/g53_copies.py).
// rank_math.cuh's header has the provenance and the port's rules; rank.cu instantiates the templates.
#pragma once

#include "rank_math.cuh"

namespace g53 {

// ---- plan: rows x 8 routes -> pairs grouped by expert ---------------------------------------------------------
// One 1,024-thread CTA (after MiMo's plan_parallel). Pairs of expert e occupy [base[e], base[e] + cnt[e]) in
// ascending expert order, cut into groups of at most `gr` pairs. The order of an expert's pairs follows the
// atomics and is not stable; that cannot change a value, since every pair is computed on its own rows.
// TF port: the routes are pick / wts [rows][ps / ws] (the 8 routed slots); on invalid routes the plan also writes
// NaN into the call's output rows (no host sync reads the fault word here).
__global__ void __launch_bounds__(1024) plan_kernel(const int32_t* __restrict__ ids, int ps,
                                                    const float* __restrict__ w,
                                                    int ws, int rows, int gr, Group* __restrict__ groups,
                                                    int* __restrict__ counts, int* __restrict__ pair_row,
                                                    int* __restrict__ pair_route, int* __restrict__ inverse,
                                                    unsigned* __restrict__ fault, float* __restrict__ out) {
    __shared__ int cnt[E], base[E], gbase[E], cursor[E];
    __shared__ int bad;
    const int tid = threadIdx.x, nt = blockDim.x, routes = rows * TOPK;
    for (int e = tid; e < E; e += nt) { cnt[e] = 0; cursor[e] = 0; }
    if (tid == 0) bad = 0;
    __syncthreads();
    for (int r = tid; r < routes; r += nt) {
        const int row = r / TOPK, s = r % TOPK;
        const int e = ids[size_t(row) * ps + s];
        const float wt = w[size_t(row) * ws + s];
        bool ok = e >= 0 && e < E && wt >= 0.0f && wt <= 0x1.fffffep127f;  // finite, not negative, not NaN
        if (ok)
            for (int k = 0; k < s; ++k) ok = ok && ids[size_t(row) * ps + k] != e;  // no expert twice in a row
        if (!ok) atomicOr(&bad, 1); else atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (bad) {
        if (tid == 0) { atomicOr(fault, F_ROUTE); counts[0] = 0; }
        const float qnan = __int_as_float(0x7fc00000);
        for (size_t i = tid; i < size_t(rows) * H; i += nt) out[i] = qnan;
        return;
    }
    // Exclusive prefix sums of the pair and group counts over the 288 experts: nine warps scan 32 experts each
    // with shuffles, then every warp adds the totals of the warps before it (a serial scan here cost ~10 us).
    __shared__ int wsum[E / 32], wgsum[E / 32];
    const int warp = tid >> 5, lane = tid & 31;
    int c = 0, gc = 0, ic = 0, igc = 0;
    if (warp < E / 32) {
        c = cnt[warp * 32 + lane];
        gc = (c + gr - 1) / gr;
        ic = c;
        igc = gc;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, ic, d), ug = __shfl_up_sync(0xffffffffu, igc, d);
            if (lane >= d) { ic += u; igc += ug; }
        }
        if (lane == 31) { wsum[warp] = ic; wgsum[warp] = igc; }
    }
    __syncthreads();
    if (warp < E / 32) {
        int off = 0, goff = 0;
        for (int w2 = 0; w2 < warp; ++w2) { off += wsum[w2]; goff += wgsum[w2]; }
        base[warp * 32 + lane] = off + ic - c;
        gbase[warp * 32 + lane] = goff + igc - gc;
        if (warp == E / 32 - 1 && lane == 31) counts[0] = goff + igc;
    }
    __syncthreads();
    for (int r = tid; r < routes; r += nt) {
        const int e = ids[size_t(r / TOPK) * ps + r % TOPK];
        const int gp = base[e] + atomicAdd(&cursor[e], 1);
        pair_row[gp] = r / TOPK;
        pair_route[gp] = r;
        inverse[r] = gp;
    }
    for (int e = tid; e < E; e += nt) {
        const int c2 = cnt[e];
        for (int k = 0; k * gr < c2; ++k)
            groups[gbase[e] + k] = Group{e, base[e] + k * gr, c2 - k * gr < gr ? c2 - k * gr : gr, 0};
    }
}

// ---- gate and up, large-M kernel -----------------------------------------------------------------------------
// Persistent blocks of NW MMA warps (8 or 16) take (mat, column block, group, split) items in grid order
// (launch_persistent). An item covers the group's rows (up to 16 MT) and one matrix's 512 columns of a column
// block; MMA warp w owns columns [16 NT w, 16 NT (w + 1)) of it with NT = 32 / NW, so each decoded tile feeds 2 MT
// MMAs. The rotated rows are double-buffered in shared memory with one barrier per 128-wide K block: while the MMA
// warps multiply one K block, the next one is rotated by the MMA warps themselves, one row per k tile of the first
// R / NW k tiles with its inputs loaded a k tile ahead. The trellis words are loaded PF k tiles ahead; the A
// fragments come from ldmatrix and the MMAs of m tiles past the group's rows are predicated off. The products are
// the split kernel's, accumulated in the same order.
//
// XH (rowrot): when every expert of the layer has one `suh` for gate and up, `rotrows_kernel` has rotated every
// row once into a.xh, and the blocks copy the rotated K blocks into the same shared buffers with cp.async (one K
// block ahead) instead of rotating them per (row, matrix). row_rotate's FP16 values are the same whichever kernel
// computes them, so the products are the same bits.
// TF port: gate/up rotation warps (NP > 0, glm53f-rank's gp 2) are not ported; the trellis of an item comes from
// the layer's pointer table, and suh from the stacked vectors.
template <int MT, int NW, bool XH = false>
__global__ void __launch_bounds__(NW * 32, (MT == 2 && NW == 8) ? 2 : 1) gateup_big(const Args a) {
    constexpr int NT = 32 / NW, R = MT * 16, RPW = XH ? 0 : R / NW, PF = NT >= 4 ? 2 : 4;
    static_assert(XH || (RPW >= 1 && RPW <= 8), "a row a warp per k tile rotates a whole K block");
    __shared__ __align__(16) half As[2][R][136];
    __shared__ int rows_sh[R];
    __shared__ int last_sh, item_sh;
    // Persistent blocks take (mat, column block, group, split) items in grid order, the matrix varying fastest.
    const int ngroups = a.counts[0], items = 2 * CB * ngroups * a.sk;
    for (;;) {
        if (threadIdx.x == 0) item_sh = atomicAdd(a.work, 1);
        __syncthreads();
        const int item = item_sh;
        if (item >= items) break;
        const int mc = item & (2 * CB - 1), rest = item >> LOG_2CB;
        const int mat = mc & 1, cb = mc >> 1, gs = rest % ngroups, split = rest / ngroups;
        const int g = a.gorder ? a.gorder[gs] : gs;
        const Group G = a.groups[g];
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
        if (tid < R) rows_sh[tid] = tid < G.count ? a.pair_row[G.start + tid] : -1;
        __syncthreads();

        const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? a.up_t[G.expert] : a.gate_t[G.expert]);
        const half* suh = (mat ? a.suh_u : a.suh_g) + size_t(G.expert) * H;
        const int mt_used = (G.count + 15) >> 4, rows_used = mt_used * 16;
        const int kbs = 32 / a.sk, kb0 = split * kbs, kt0 = kb0 * 8, kt_last = kt0 + kbs * 8 - 1;
        bool bad = false;

        // XH: copy K block kb0 + kbi of the group's rotated rows into buffer `buf` (16-byte cp.async, 16 per row);
        // rows past the group are zeroed (their products are never stored).
        auto stage_xh = [&](int kbi, int buf) {
            const int kb = kb0 + kbi;
            for (int i = tid; i < rows_used * 16; i += NW * 32) {
                const int rr = i >> 4, p = i & 15, r = rows_sh[rr];
                half* dst = &As[buf][rr][p * 8];
                if (r >= 0) cp_async16(dst, a.xh + size_t(r) * H + kb * 128 + p * 8);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
            cp_async_commit();
        };

        if constexpr (XH) {
            stage_xh(0, 0);
        } else {
            // The first K block, every warp a row at a time.
            for (int rr = warp; rr < rows_used; rr += NW) {
                const int r = rows_sh[rr];
                const uint2 packed = r >= 0 ? rotate_row(a.x + size_t(r) * a.xs, suh, kb0, lane, bad)
                                            : make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&As[0][rr][lane * 4]) = packed;
            }
        }

        {
            // Trellis words of tile (kt, nt) are at (kt * NTG + nt) * 32 + lane; this warp's NT n tiles are
            // contiguous. A ring of PF k tiles of words in flight.
            const uint32_t* Tw = T + size_t(cb * 32 + warp * NT) * 32 + lane;
            const bool hint = a.l2hint != 0;
            const uint64_t pol = hint ? evict_first_policy() : 0;
            uint32_t wr[PF][NT];
#pragma unroll
            for (int p = 0; p < PF; ++p)
#pragma unroll
                for (int i = 0; i < NT; ++i)
                    wr[p][i] = ldg_w(Tw + (size_t(kt0 + p < kt_last ? kt0 + p : kt_last) * NTG + i) * 32, hint, pol);
            // The inputs of row warp + NW j of K block kb0 + kbi + 1 (rows past the group read row 0's inputs; their
            // results are not stored).
            auto load_next = [&](int kbi, int j) -> RowIn {
                const int r = rows_sh[warp + NW * j];
                const int kb = kb0 + kbi + 1 < 32 ? kb0 + kbi + 1 : 31;
                return row_load(a.x + size_t(r >= 0 ? r : rows_sh[0]) * a.xs, suh, kb, lane);
            };
            RowIn nx;
            if constexpr (RPW > 0) nx = load_next(0, 0);

            float acc[MT][NT][2][4];
#pragma unroll
            for (int m = 0; m < MT; ++m)
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h)
#pragma unroll
                        for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
            const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

            for (int kbi = 0; kbi < kbs; ++kbi) {
                if constexpr (XH) cp_async_wait<0>();  // this thread's copies into buffer kbi & 1 have landed
                __syncthreads();  // buffer kbi & 1 is complete; the other one's last readers are done
                const int buf = kbi & 1;
                const bool more = kbi + 1 < kbs;
                if constexpr (XH) {
                    if (more) stage_xh(kbi + 1, buf ^ 1);
                }
#pragma unroll
                for (int ktl = 0; ktl < 8; ++ktl) {
                    if (ktl < RPW) {  // rotate this warp's row ktl of the next K block, then load the next one's inputs
                        const int rr = warp + NW * ktl;
                        bool b = false;
                        const uint2 packed = row_rotate(nx, lane, b);
                        if (more && rr < rows_used) {
                            const bool real = rows_sh[rr] >= 0;
                            *reinterpret_cast<uint2*>(&As[buf ^ 1][rr][lane * 4]) = real ? packed : make_uint2(0u, 0u);
                            bad = bad || (real && b);
                        }
                        nx = ktl + 1 < RPW ? load_next(kbi, ktl + 1) : load_next(kbi + 1, 0);
                    }
                    const int kt = kt0 + kbi * 8 + ktl, slot = ktl % PF;
                    uint32_t w[NT];
#pragma unroll
                    for (int i = 0; i < NT; ++i) w[i] = wr[slot][i];
                    const int ktn = kt + PF < kt_last ? kt + PF : kt_last;
#pragma unroll
                    for (int i = 0; i < NT; ++i) wr[slot][i] = ldg_w(Tw + (size_t(ktn) * NTG + i) * 32, hint, pol);
                    uint32_t af[MT][4];
#pragma unroll
                    for (int m = 0; m < MT; ++m) ldsm_a(af[m], &As[buf][m * 16 + lrow][ktl * 16 + lcol]);
#pragma unroll
                    for (int i = 0; i < NT; ++i) {
                        uint32_t b0[2], b1[2];
                        decode_tile(w[i], lane, b0, b1);
#pragma unroll
                        for (int m = 0; m < MT; ++m)
                            if (m < mt_used) {
                                mma16816(acc[m][i][0], af[m], b0);
                                mma16816(acc[m][i][1], af[m], b1);
                            }
                    }
                }
            }

            float* zbase = a.z + (size_t(mat) * a.sk + split) * size_t(a.P) * WID;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                if (m >= mt_used) break;
                const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = cb * 512 + warp * NT * 16 + i * 16 + h * 8 + 2 * t4;
                        if (r0 < G.count)
                            *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * WID + col) =
                                make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                        if (r1 < G.count)
                            *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * WID + col) =
                                make_float2(acc[m][i][h][2], acc[m][i][h][3]);
                    }
            }
        }
        if (bad) atomicOr(a.fault, F_ROW);
        if (a.fused) gateup_tail<NW>(a, G, g, &last_sh);
        __syncthreads();  // the next item reuses the shared buffers and item_sh
    }
}

// ---- epilogue kernel (unfused): one warp per (pair, 128-block of the rank width) --------------------------------
__global__ void __launch_bounds__(128) gateup_epilogue(const Args a) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int gp = item >> LOG_NB, blk = item & (NB - 1);
    if (gp >= a.P || a.counts[0] <= 0) return;  // no plan (invalid routes): ids may be out of range
    const int r = a.pair_route[gp];
    const int e = a.pick[size_t(r / TOPK) * a.ps + r % TOPK];
    const bool bad = epilogue_item(a, e, gp, blk, lane);
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_GATEUP);
}

// ---- down, large-M kernel -------------------------------------------------------------------------------------
// Persistent blocks of 8 warps take (group, chunk of 128 NT output columns, split) items in grid order: the
// group's rows (up to 16 MT), warp w owning NT n tiles of the chunk, so each decoded tile feeds 2 MT MMAs. The
// group's down input is staged in shared memory with cp.async, 128 K columns at a time, double-buffered; the A
// fragments come from ldmatrix and the MMAs of m tiles past the group's rows are predicated off (the loop body
// has no branch); the trellis words are loaded PF k tiles ahead. Items run chunk by chunk (the group index
// varies fastest), so a chunk's partial sums are written and reduced (fused) while still in L2.
// TF port: the trellis of an item comes from the layer's pointer table.
template <int MT, int NT>
__global__ void __launch_bounds__(256) down_big(const Args a) {
    constexpr int NW = 8, R = MT * 16, CW = NW * NT * 16, KC = 8, PF = 4;
    __shared__ __align__(16) half As[2][R][136];
    __shared__ int list_sh[R], nl_sh, item_sh;
    // Persistent blocks take (group, chunk, split) items in grid order, the group varying fastest.
    const int ngroups = a.counts[0], items = ngroups * a.nch * a.skd;
    for (;;) {
        if (threadIdx.x == 0) item_sh = atomicAdd(a.work + 1, 1);
        __syncthreads();
        const int item = item_sh;
        if (item >= items) break;
        const int g = a.gorder ? a.gorder[item % ngroups] : item % ngroups;
        const int chunk = (item / ngroups) % a.nch, split = item / (ngroups * a.nch);
        const Group G = a.groups[g];
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g8 = lane >> 2, t4 = lane & 3;
        const int my_row = tid < G.count ? a.pair_row[G.start + tid] : -1;  // for the fused reduce
        const int mt_used = (G.count + 15) >> 4;
        const int kts = NTG / a.skd, kt0 = split * kts, nkc = kts / KC, kt_last = kt0 + kts - 1;
        const half* xg = a.xd + size_t(G.start) * WID;

        // Stage K columns [16 (kt0 + KC c), + 16 KC) of the group's rows into buffer b: 16-byte copies; rows past the
        // group are zeroed (their products are never stored).
        auto stage = [&](int c, int b) {
            const int k0 = (kt0 + c * KC) * 16;
            for (int i = tid; i < mt_used * 16 * 2 * KC; i += NW * 32) {
                const int r = i / (2 * KC), p = i % (2 * KC);
                half* dst = &As[b][r][p * 8];
                if (r < G.count) cp_async16(dst, xg + size_t(r) * WID + k0 + p * 8);
                else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
            }
            cp_async_commit();
        };
        stage(0, 0);

        const uint32_t* T = reinterpret_cast<const uint32_t*>(a.down_t[G.expert]);
        // Down tiles are [64 k tiles][256 n tiles]; this warp's NT n tiles start at chunk * CW / 16 + warp * NT.
        // A ring of PF k tiles of words in flight.
        const uint32_t* Tw = T + size_t(chunk * (CW / 16) + warp * NT) * 32 + lane;
        const bool hint = a.l2hint != 0;
        const uint64_t pol = hint ? evict_first_policy() : 0;
        uint32_t wr[PF][NT];
#pragma unroll
        for (int p = 0; p < PF; ++p)
#pragma unroll
            for (int i = 0; i < NT; ++i)
                wr[p][i] = ldg_w(Tw + (size_t(kt0 + p < kt_last ? kt0 + p : kt_last) * 256 + i) * 32, hint, pol);

        float acc[MT][NT][2][4];
#pragma unroll
        for (int m = 0; m < MT; ++m)
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int c = 0; c < 4; ++c) acc[m][i][h][c] = 0.f;
        const int lrow = (lane & 7) + ((lane >> 3) & 1) * 8, lcol = (lane >> 4) * 8;

        for (int c = 0; c < nkc; ++c) {
            if (c + 1 < nkc) {
                stage(c + 1, (c + 1) & 1);
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncthreads();
            const int b = c & 1;
#pragma unroll
            for (int ktl = 0; ktl < KC; ++ktl) {
                const int kt = kt0 + c * KC + ktl, slot = ktl % PF;
                uint32_t w[NT];
#pragma unroll
                for (int i = 0; i < NT; ++i) w[i] = wr[slot][i];
                const int ktn = kt + PF < kt_last ? kt + PF : kt_last;
#pragma unroll
                for (int i = 0; i < NT; ++i) wr[slot][i] = ldg_w(Tw + (size_t(ktn) * 256 + i) * 32, hint, pol);
                uint32_t af[MT][4];
#pragma unroll
                for (int m = 0; m < MT; ++m) ldsm_a(af[m], &As[b][m * 16 + lrow][ktl * 16 + lcol]);
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(w[i], lane, b0, b1);
#pragma unroll
                    for (int m = 0; m < MT; ++m)
                        if (m < mt_used) {
                            mma16816(acc[m][i][0], af[m], b0);
                            mma16816(acc[m][i][1], af[m], b1);
                        }
                }
            }
            __syncthreads();  // buffer b is read; the next stage may overwrite it
        }
        float* zbase = a.zd + size_t(split) * size_t(a.P) * H;
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            if (m >= mt_used) break;
            const int r0 = m * 16 + g8, r1 = r0 + 8;
#pragma unroll
            for (int i = 0; i < NT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int col = chunk * CW + warp * NT * 16 + i * 16 + h * 8 + 2 * t4;
                    if (r0 < G.count)
                        *reinterpret_cast<float2*>(zbase + size_t(G.start + r0) * H + col) =
                            make_float2(acc[m][i][h][0], acc[m][i][h][1]);
                    if (r1 < G.count)
                        *reinterpret_cast<float2*>(zbase + size_t(G.start + r1) * H + col) =
                            make_float2(acc[m][i][h][2], acc[m][i][h][3]);
                }
        }
        if (a.fused) down_tail<NW>(a, chunk, CW / 128, my_row, list_sh, &nl_sh);
        __syncthreads();  // the next item reuses the shared buffers and item_sh
    }
}

// ---- reduce kernel (unfused): one warp per (row, 128-block of the model width) ---------------------------------
__global__ void __launch_bounds__(128) reduce_kernel(const Args a) {
    const int item = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = item >> 5, blk = item & 31;
    if (row >= a.rows || a.counts[0] <= 0) return;  // no plan (invalid routes)
    const bool bad = reduce_item(a, row, blk, lane);
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_OUT);
}

// ---- rowrot and order (glm53f-afd's reimplementations of MiaAI-Lab's recipe patches 0009 and 0020) -----------
// rowrot: every row's 32 K blocks rotated once with the layer's shared gate/up suh, into xh [rows][4096] FP16, one
// warp per (row, K block). The values are row_rotate's, so the large-M gate/up kernel that reads them (XH) does the
// products of the per-(row, matrix) rotation bit for bit. NaN values fault as in the gate/up kernels.
__global__ void __launch_bounds__(256) rotrows_kernel(const Args a, const half* __restrict__ suh,
                                                      half* __restrict__ xh) {
    const int w = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int row = w >> 5, kb = w & 31;
    if (row >= a.rows) return;  // warp-uniform
    bool bad = false;
    const uint2 packed = rotate_row(a.x + size_t(row) * a.xs, suh, kb, lane, bad);
    *reinterpret_cast<uint2*>(xh + size_t(row) * H + kb * 128 + lane * 4) = packed;
    if (__any_sync(0xffffffffu, bad) && lane == 0) atomicOr(a.fault, F_ROW);
}

// order: the large-M kernels' group launch order. Groups of experts that have one group (their weights come from
// DRAM for few rows) and of experts that have several (read once from DRAM, then from L2: mostly MMA) are merged
// evenly, each list in plan order (an expert's groups stay adjacent), so DRAM and the tensor cores stay busy
// together instead of in turns. Only which block runs which item changes: every output keeps its bits.
// gorder[slot] = group. One block of 1,024 threads, each over a contiguous range of groups.
__global__ void __launch_bounds__(1024) order_kernel(const Group* __restrict__ groups, const int* __restrict__ counts,
                                                     int* __restrict__ gorder) {
    __shared__ int wsum[32];
    const int n = counts[0];
    if (n <= 0) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int per = (n + 1023) / 1024, lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = groups[i].expert;
        return (i > 0 && groups[i - 1].expert == e) || (i + 1 < n && groups[i + 1].expert == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;  // inclusive scan of the heavy counts over the threads
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, d);
        if (lane >= d) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int s = wsum[lane];
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, s, d);
            if (lane >= d) s += u;
        }
        wsum[lane] = s;
    }
    __syncthreads();
    int hb = v - h + (warp ? wsum[warp - 1] : 0);  // heavy groups before `lo`
    const long long nh = wsum[31], nl = n - nh;
    // Heavy h sits at key (2h + 1) / 2nh, light l at (2l + 1) / 2nl; equal keys put the heavy group first.
    for (int i = lo; i < hi; ++i) {
        int pos;
        if (heavy(i)) {
            const long long hh = hb++, q = nh ? ((2 * hh + 1) * nl + nh - 1) / nh : 0;
            pos = int(hh + (q / 2 < nl ? q / 2 : nl));
        } else {
            const long long l = i - hb, f = nl ? ((2 * l + 1) * nh) / nl : 0;
            pos = int(l + ((f + 1) / 2 < nh ? (f + 1) / 2 : nh));
        }
        gorder[pos] = i;
    }
}

}  // namespace g53
