// GLM-5.3-Flash's sparse.select_tokens tail (python-0.6 ed78d6f, sparse.py 205-216) in one launch: python's 21 torch
// ops on integer tensors after its kernels, here one thread a token. Row r (q = pos + r, npool = (q + 1) / 4) gets the
// 512 pools' tokens (pool * 4 + i, int64 cast to int32 as torch's copy casts), then the incomplete last pool's visible
// tokens (npool * 4 + j <= q, else -1), and its count: 2048 + those tokens past the dense limit (npool > 512), else 0.
// qualification_qualification_44ba554e.5 lane A3; the host reference is cuda_dsa_kernels.zig's `tail`.
#include <cstdint>

extern "C" __global__ void tf_glm_select_tail(const int64_t* pools, const int32_t* pos, int32_t* tokens,
                                              int32_t* counts, uint32_t rows) {
  constexpr int64_t kPools = 512, kPool = 4, kWidth = kPools * kPool + kPool - 1;
  const int64_t r = blockIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (r >= rows || i >= kWidth) return;
  const int64_t q = static_cast<int64_t>(pos[0]) + r;
  const int64_t npool = (q + 1) / kPool;
  int64_t t;
  if (i < kPools * kPool) {
    t = pools[r * kPools + i / kPool] * kPool + i % kPool;
  } else {
    t = npool * kPool + (i - kPools * kPool);
    t = t <= q ? t : -1;
  }
  tokens[r * kWidth + i] = static_cast<int32_t>(t);
  if (i == 0) {
    int64_t kept = 0;
    for (int64_t j = 0; j < kPool - 1; ++j) kept += npool * kPool + j <= q;
    counts[r] = static_cast<int32_t>(npool > kPools ? kPools * kPool + kept : 0);
  }
}

// forward.dense_attention's and sparse.select_tokens' `pos_dev + n` (an int32 [1] tensor plus a python int: torch's
// add on the device, int32 out): a block's first row's position. One thread; the add wraps as int32's. qualification_qualification_44ba554e.5 lane A4.
extern "C" __global__ void tf_glm_pos_add(const int32_t* pos, int32_t n, int32_t* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  out[0] = static_cast<int32_t>(static_cast<uint32_t>(pos[0]) + static_cast<uint32_t>(n));
}
