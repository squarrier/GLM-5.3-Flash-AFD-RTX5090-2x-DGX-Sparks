"""patch 0002-glm-dense-fp8.patch's FP8 weights (MiaAI-Lab's v1.10 recipe, sha256 179049d9fe96084e),
vendored for python-0.6.6, which has no 0002 (qualification_qualification_ebed70b7): F8, make_f8, stack_f8, the Triton _fmm and _matmul_f8, below
the marker byte for byte as the patch adds them to glm5_next/cuda/qmm.py; what they use comes from python-0.6.6's qmm.
Written by Mia (MiaAI-Lab, @MiaAI_lab); TensorFold by Ash Hart (@ashxhart). Only this header is new.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda.qmm import B16, B16_BK, _reduce, b16_split_k, bucket, make_b16  # noqa: F401

# ---- patch 0002, glm5_next/cuda/qmm.py, the added lines from "# -- FP8 weights" through F8_CONFIG, unchanged ----
# -- FP8 weights (TF_GLM_DENSE=fp8): the checkpoint's BF16 matrices stored as e4m3 with an fp32 scale per row and
# 128-column block, quantized once at load. Lossy (replies differ from BF16 weights); the activations stay bf16 and
# the kernel keeps _bmm's structure (a row's bits never depend on the other rows), so drafted replies still equal
# serial ones under it.
F8_BLOCK = 128
F8_MAX = 448.0


@dataclass
class F8:
    """A BF16 matrix [n, k] held as FP8 e4m3 [n, k] and fp32 scales [n, k / 128]: W ~ q * scale."""

    weight: torch.Tensor      # [n, k] torch.float8_e4m3fn
    scale: torch.Tensor       # [n, k // 128] fp32
    n: int
    k: int

    def nbytes(self) -> int:
        return self.weight.numel() + self.scale.numel() * 4


def make_f8(weight: torch.Tensor, rows: int = 8192) -> F8 | B16:
    """Quantize a BF16 matrix a block of 128 columns at a time (absmax / 448 scale, round to nearest); a matrix whose
    width is not a multiple of 128 stays BF16."""

    n, k = weight.shape
    if k % F8_BLOCK:
        return make_b16(weight)
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
    scale = torch.empty((n, k // F8_BLOCK), dtype=torch.float32, device=weight.device)
    for r in range(0, n, rows):
        g = weight[r:r + rows].float().view(-1, k // F8_BLOCK, F8_BLOCK)
        s = (g.abs().amax(-1) / F8_MAX).clamp_min(1e-30)
        q[r:r + rows] = (g / s[..., None]).clamp(-F8_MAX, F8_MAX).view(-1, k).to(torch.float8_e4m3fn)
        scale[r:r + rows] = s
    return F8(q, scale, int(n), int(k))


def stack_f8(parts: list[torch.Tensor]) -> F8 | B16:
    return make_f8(torch.cat([p.to(torch.bfloat16) for p in parts]))


@triton.jit
def _fmm(X, W, S, OUT, PART, M, x_stride, N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, BK: tl.constexpr, F32: tl.constexpr):
    """_bmm over FP8 weights: each BK step's product (bf16 inputs, fp32 sums) times its 128-column block's scales."""

    PER: tl.constexpr = K // SK
    KB: tl.constexpr = K // 128
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_s * PER, pid_s * PER + PER, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
        s = tl.load(S + rn.to(tl.int64) * KB + k0 // 128, mask=n_ok, other=0.0)
        acc = acc + tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def _matmul_f8(x: torch.Tensor, q: F8, *, out: torch.Tensor | None, f32: bool,
               part: torch.Tensor | None) -> torch.Tensor:
    m, k = x.shape
    if k != q.k or x.stride(1) != 1 or x.dtype != torch.bfloat16 or k % F8_BLOCK:
        raise ValueError(f"matmul: x {tuple(x.shape)} {x.dtype} does not match K={q.k}")
    bm = bucket(min(m, 128))
    warps, stages = F8_CONFIG[bm]
    sk = b16_split_k(q.n, q.k)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1:
        need = sk * m * q.n
        if part is None or part.numel() < need:
            part = torch.empty((need,), dtype=torch.float32, device=x.device)
    bn = F8_BN_DECODE if bm == 16 else F8_BN
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), sk)
    _fmm[grid](x, q.weight, q.scale, out, part if sk > 1 else out, m, x.stride(0), N=q.n, K=k, SK=sk, BM=bm,
               BLOCK_N=bn, BK=B16_BK, F32=f32, num_warps=warps, num_stages=stages)
    if sk > 1:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


F8_BN = 64
F8_BN_DECODE = 32          # decode windows (up to 16 rows): narrower column tiles, deeper pipeline (measured +8%)
# (warps, stages) by row bucket; neither they nor the column tile change a row's bits
F8_CONFIG = {16: (4, 6), 32: (4, 3), 64: (4, 2), 128: (8, 2)}
