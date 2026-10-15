// glm53f-afd's large-M expert kernels for the Zig engine's expert halves (TF_GLM_HALVES_KERNEL=g53kt3r): the image
// g53_rank. Written by Hugh Madden (@dangerm00se, github.com/hughmadden), glm53f-afd @ 91db3cc, MIT License,
// Copyright (c) 2026 Turquoise Bay AI Pty Ltd (LICENSES/glm53f-afd-MIT.txt), as python TF-AFD 2.15 ports them
// (families/glm5_next/cuda/g53_rank.cu, patch 0034); rank_math.cuh's header has the provenance and the port's rules.
// This file instantiates every kernel 2.15's host launcher (g53_ffn_launch) launches; zig/src/families/glm/cuda_g53.zig
// makes those launches. Build with python's flags: -O3 --fmad=false --ftz=false --prec-div=true.
#include "rank_kernels.cuh"

namespace g53 {

template __global__ void gateup_big<2, 8, false>(const Args);
template __global__ void gateup_big<4, 8, false>(const Args);
template __global__ void gateup_big<2, 16, false>(const Args);
template __global__ void gateup_big<4, 16, false>(const Args);
template __global__ void gateup_big<2, 8, true>(const Args);
template __global__ void gateup_big<4, 8, true>(const Args);
template __global__ void gateup_big<2, 16, true>(const Args);
template __global__ void gateup_big<4, 16, true>(const Args);
template __global__ void down_big<2, 1>(const Args);
template __global__ void down_big<2, 2>(const Args);
template __global__ void down_big<2, 4>(const Args);
template __global__ void down_big<4, 1>(const Args);
template __global__ void down_big<4, 2>(const Args);
template __global__ void down_big<4, 4>(const Args);

}  // namespace g53
