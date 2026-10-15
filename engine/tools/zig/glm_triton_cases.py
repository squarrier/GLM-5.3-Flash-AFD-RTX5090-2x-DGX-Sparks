"""GLM-5.3-Flash's Triton launches by case: python-0.6's glm5_next/cuda wrappers on CPU tensors of TP rank shapes."""
# Adapted from the case list of tools/zig/flashnext_triton_fixtures.py, written by Mia (MiaAI-Lab, @MiaAI_lab) in
# MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, Apache License 2.0; GLM-5.3-Flash's cases are new.

from __future__ import annotations

import torch

from tensorfold.families.glm5_next.cuda import attention, forward, glue, latent, qmm, sparse

bf, f32, i32, i64 = torch.bfloat16, torch.float32, torch.int32, torch.int64
# verify windows (a pending token and up to 7 drafts; the MTP head's too) and prompt-chunk rows with their tails
ROWS_DEC = tuple(range(1, 9))
ROWS_PRE = (1, 2, 15, 16, 17, 32, 33, 63, 64, 65, 100, 128, 129, 256, 511, 512, 513, 1000, 1024, 2047, 2048)
ROWS_ALL = tuple(sorted(set(ROWS_DEC) | set(ROWS_PRE)))
NCH_DEC = tuple(range(1, 7))        # latent chunks of a verify window: Buffers(8 rows) holds chunks_for(2560 + 8) = 6
NCH_PRE = tuple(range(1, 10))       # a prompt chunk's: chunks_for(2560 + 2048) = 9
NCH_AB = (1, 2, 3, 4, 5, 6, 513)    # TF_GLM_LATENT=0: eager windows to the dense limit, a 262,144-slot graph
POOLS = (1024, 65536, 65537)        # pool buckets: powers of two, a 262,144-slot cache's, an odd capacity's
KDA_LAYERS = 34                     # 45 layers less the 11 DSA ones (3, 7, ..., 43)
SITES_OUT = {"dflash2.py": "DFlash2 drafter: its own checkpoint and shapes, out of scope (qualification_qualification_cfb9554b NOTES 3)"}


class Dims:
    """GLM-5.3-Flash's config.json dimensions and one rank's share of them at ``world`` ranks."""

    def __init__(self, world: int = 2) -> None:
        self.world, self.d, self.s = world, 4096, 4
        self.hl = self.ll = 64 // world                       # DSA and KDA heads a rank
        self.qk = self.vd = 256
        self.ql, self.kl, self.ih, self.id = 1536, 512, 32, 128
        self.e, self.topk = 288, 8
        self.dl, self.ml, self.sl = 12288 // world, 2048 // world, 2048 // world
        self.vocab = 154880 // world
        self.eps, self.hc_eps, self.iters, self.limit, self.scale = 1e-5, 1e-6, 20, 10.0, 2.5
        self.c = 3 * self.ll * 128                             # the KDA conv channels: q | k | v
        self.kw = self.c + 2 * 128 + self.ll                   # the stacked KDA in-projection row
        self.fa = self.c

    def mats(self) -> dict[str, tuple[int, int, bool, int]]:
        """EXL3's BF16 projections a rank runs: name -> (N, K, fp32 out, x row stride)."""
        h = self
        return {"kda_proj": (h.kw, h.d, False, h.d), "kda_fb": (h.ll * 128, 128, False, h.kw),
                "kda_o": (h.d, h.ll * 128, True, h.ll * 128), "dsa_proj": (h.ql + h.kl, h.d, False, h.d),
                "dsa_qb": (h.hl * h.qk, h.ql, False, h.ql), "dsa_o": (h.d, h.hl * h.vd, True, h.hl * h.vd),
                "idx_kw": (h.id + h.ih, h.d, False, h.d), "idx_qb": (h.ih * h.id, h.ql, False, h.ql),
                "mlp_gu": (2 * h.dl, h.d, False, h.d), "mlp_down": (h.d, h.dl, True, h.dl),
                "sh_gu": (2 * h.sl, h.d, False, h.d), "sh_down": (h.d, h.sl, True, h.sl),
                "mtp_eh": (h.d, 2 * h.d, False, 2 * h.d), "head": (h.vocab, h.d, False, h.d),
                "ab_kv_k": (h.hl * h.qk, h.kl, False, h.kl), "ab_kv_v": (h.hl * h.vd, h.kl, False, h.kl)}


def z(*shape, dtype=bf) -> torch.Tensor:
    """A real CPU tensor (its strides and contiguity are what the wrapper reads); pages stay untouched."""
    return torch.empty(shape, dtype=dtype)


def e(*shape, dtype=bf) -> torch.Tensor:
    """A tensor of ``shape`` without storage, for weights whose strides no wrapper reads."""
    return torch.empty((1,) * len(shape), dtype=dtype).expand(*shape)


def glue_cases(h: Dims):
    """Embedding, norms, hyper-connections, residuals, MLP and router glue (glue.py), every row count."""
    D, S = h.d, h.s
    q4 = (e(h.vocab * h.world, D // 8, dtype=i32), e(h.vocab * h.world, D // 64), e(h.vocab * h.world, D // 64))
    b16 = e(h.vocab * h.world, D)
    norm = e(D)
    for R in ROWS_ALL:
        ids = z(R, dtype=i32)
        for s in (S, 1):
            yield f"embed/q4/r{R}/s{s}", lambda s=s, ids=ids: glue.embed(ids, q4, D, s, z(R, s * D))
            yield f"embed/b16/r{R}/s{s}", lambda s=s, ids=ids: glue.embed(ids, b16, D, s, z(R, s * D))
        dp = z(R, h.ql + h.kl)
        mcat = z(R, 2 * D)
        yield f"rmsnorm/q/r{R}", lambda dp=dp: glue.rmsnorm(dp[:, :h.ql], norm[:h.ql], h.eps, z(R, h.ql),
                                                           z(R, h.ql // 64, dtype=f32))
        yield f"rmsnorm/kv/r{R}", lambda dp=dp: glue.rmsnorm(dp[:, h.ql:], norm[:h.kl], h.eps, z(R, h.kl),
                                                            z(R, h.kl // 64, dtype=f32))
        yield f"rmsnorm/d/r{R}", lambda: glue.rmsnorm(z(R, D), norm, h.eps, z(R, D), z(R, D // 64, dtype=f32))
        yield f"rmsnorm/half/r{R}", lambda mcat=mcat: glue.rmsnorm(z(R, D), norm, h.eps, mcat[:, D:])
        x = z(R, S * D)
        post, comb = z(R, S, dtype=f32), z(R, S * S, dtype=f32)
        hc = (e(24, S * D), e(24, dtype=f32), e(3, dtype=f32))
        yield f"hc_pre/r{R}", lambda x=x, post=post, comb=comb: glue.hc_pre(
            x, *hc, norm, z(R, D), z(R, D // 64, dtype=f32), post, comb, z(R, glue.HC_BLOCKS, 32, dtype=f32),
            h.eps, h.hc_eps, h.iters)
        gath = z(h.world, R, D, dtype=f32)
        yield f"hc_post/r{R}", lambda x=x, gath=gath, post=post, comb=comb: glue.hc_post(x, x, gath, post, comb)
        yield f"residual_add/r{R}", lambda gath=gath: glue.residual_add(z(R, D), z(R, D), gath)
        yield f"stream_mean/r{R}", lambda x=x: glue.stream_mean(x, z(R, D))
        for name, w in (("dense", h.dl), ("shared", h.sl)):
            yield f"swiglu/{name}/r{R}", lambda w=w: glue.swiglu(z(R, 2 * w), z(R, w), z(R, w // 64, dtype=f32),
                                                                h.limit)
        xn = z(R, D)
        yield f"router/moe/r{R}", lambda xn=xn: glue.router(xn, e(h.e, D), z(R, h.e, dtype=f32))
        yield f"router/gate/r{R}", lambda xn=xn: glue.router(xn, e(h.id, D), z(R, h.id, dtype=f32))
        yield f"router/strided/r{R}", lambda xn=xn: glue.router(xn, e(h.e, D), z(R, 2 * h.e, dtype=f32)[:, :h.e])
        yield f"select/r{R}", lambda: glue.select(z(R, h.e, dtype=f32), z(h.e, dtype=f32), z(R, h.topk + 1, dtype=i32),
                                                  z(R, h.topk + 1, dtype=f32), h.topk, h.e, h.scale, True)
        for name, dt in (("f32", f32), ("bf16", bf)):
            yield f"combine/{name}/r{R}", lambda dt=dt: glue.combine(z(R, h.topk + 1, D, dtype=dt),
                                                                     z(R, h.topk + 1, dtype=f32), z(R, D, dtype=f32))


def qmm_cases(h: Dims):
    """qmm.group_sums on each strided input and qmm.matmul's BF16 path (EXL3) at each projection's shape."""
    for R in ROWS_ALL:
        ins = {"fa": z(R, h.kw)[:, h.fa:h.fa + 128], "kda": z(R, h.ll * 128), "dsa": z(R, h.hl * h.vd),
               "mtp": z(R, 2 * h.d)}
        for name, x in ins.items():
            yield f"group_sums/{name}/r{R}", lambda x=x: qmm.group_sums(x, z(R, x.shape[1] // 64, dtype=f32))
        for name, (n, k, fp32, stride) in h.mats().items():
            x = z(R, stride)[:, h.fa:h.fa + k] if name == "kda_fb" else z(R, stride)[:, :k]
            yield f"bmm/{name}/r{R}", lambda n=n, k=k, fp32=fp32, x=x: qmm.matmul(
                x, qmm.B16(e(n, k), n, k), out=z(R, n, dtype=f32 if fp32 else bf), f32=fp32)


def conv_cases(h: Dims):
    """forward._shift_conv: a verify window's commit over every KDA layer, and a prompt chunk's layer."""
    conv, proj = z(KDA_LAYERS, 3, h.c), z(KDA_LAYERS, 8, h.kw)
    for R in ROWS_DEC:
        for keep in range(1, R + 1):
            yield f"conv/commit/r{R}/k{keep}", lambda keep=keep: forward._shift_conv(conv, proj, keep)
    kproj = z(1, 2048, h.kw)
    for R in ROWS_PRE:
        yield f"conv/prefill/r{R}", lambda R=R: forward._shift_conv(conv[5:6], kproj[:, :R], R)


def absorb(h: Dims):
    """latent.AbsorbQ4 over the checkpoint's 4-bit kv_b rows and latent.AbsorbW over BF16 ones (EXL3)."""
    k = (z(h.hl * h.qk, h.kl // 8, dtype=i32), z(h.hl * h.qk, h.kl // 64), z(h.hl * h.qk, h.kl // 64))
    v = (z(h.hl * h.vd, h.kl // 8, dtype=i32), z(h.hl * h.vd, h.kl // 64), z(h.hl * h.vd, h.kl // 64))
    return {"q4": latent.AbsorbQ4(k, v, h.hl), "b16": latent.AbsorbW(z(h.hl, h.qk, h.kl), z(h.hl, h.vd, h.kl))}


def latent_cases(h: Dims, capacity: int = 262144):
    """DSA on the latent cache (latent.py, forward.dense_attention): absorb, write, dense and sparse attention."""
    lc, pos = z(capacity, h.kl), torch.zeros(1, dtype=i32)
    scale = h.qk ** -0.5
    ab = absorb(h)
    dec = latent.LatentScratch(8, h.hl, latent.chunks_for(2560 + 8), "cpu", lw=h.kl, part_rows=8)
    pre = latent.LatentScratch(2048, h.hl, latent.chunks_for(2560 + 2048), "cpu", lw=h.kl,
                               part_rows=forward.PROMPT_ATT_ROWS)
    tokens = z(2048, sparse.TOPK_POOLS * sparse.POOL + sparse.POOL - 1, dtype=i32)
    counts = z(2048, dtype=i32)
    for R in ROWS_ALL:
        qa, ol = z(R, h.hl, h.kl), z(R, h.hl, h.kl)
        for name, a in ab.items():
            yield f"absorb_q/{name}/r{R}", lambda a=a, qa=qa: latent.absorb_q(z(R, h.hl, h.qk), a, qa)
            yield f"expand_v/{name}/r{R}", lambda a=a, ol=ol: latent.expand_v(ol, a, z(R, h.hl, h.vd))
        yield f"latent_write/r{R}", lambda: latent.latent_write(z(R, h.kl), lc, pos)
        yield f"latent_sparse/r{R}", lambda qa=qa, ol=ol: latent.sparse_attention(qa, lc, tokens[:R], counts[:R],
                                                                                  ol, scale)
        for mode, s, rows, nchs in (("dec", dec, ROWS_DEC, NCH_DEC), ("pre", pre, ROWS_PRE, NCH_PRE)):
            if R not in rows:
                continue
            for nch in nchs:
                yield f"latent_attention/{mode}/r{R}/n{nch}", lambda s=s, nch=nch, qa=qa, ol=ol: \
                    forward.dense_attention(qa, lc, pos, s, scale=scale, nch=nch, out=ol)


def sparse_cases(h: Dims, capacity: int = 262144):
    """The DSA indexer (sparse.py): index keys and pools, then each row's pool scores and top-512 selection."""
    ik, ig, pk = z(capacity, h.id), z(capacity, h.id), z(capacity // 4 + 2, h.id)
    ln = (e(h.id), e(h.id), e(4, h.id))
    pos = torch.zeros(1, dtype=i32)
    for R in ROWS_ALL:
        ikr = z(R, h.id + h.ih)
        yield f"index_update/r{R}", lambda ikr=ikr: sparse.index_update(
            ikr[:, :h.id], z(R, h.id, dtype=f32), ln[0], ln[1], ln[2], ik, ig, pk, pos)
        for np_ in POOLS:
            yield f"select_tokens/r{R}/np{np_}", lambda ikr=ikr, np_=np_: sparse.select_tokens(
                z(R, h.ih * h.id), ikr[:, h.id:], pk, 100000, R, pk.shape[0] - 2, pos, bucket=np_)


def ab_cases(h: Dims, capacity: int = 4096):
    """TF_GLM_LATENT=0, the A/B path: per-head key and value caches, chunked and sparse attention."""
    kc, vc = z(capacity, h.hl, h.qk), z(capacity, h.hl, h.vd)
    pos, scale = torch.zeros(1, dtype=i32), h.qk ** -0.5
    tokens = z(2048, sparse.TOPK_POOLS * sparse.POOL + sparse.POOL - 1, dtype=i32)
    counts = z(2048, dtype=i32)
    for R in ROWS_ALL:
        q = z(R, h.hl, h.qk)
        yield f"ab/kv_write/r{R}", lambda q=q: attention.kv_write(q, z(R, h.hl, h.vd), kc, vc, pos)
        yield f"ab/sparse/r{R}", lambda q=q: sparse.sparse_attention(q, kc, vc, tokens[:R], counts[:R],
                                                                     z(R, h.hl, h.vd), scale)
        if R in ROWS_DEC:
            for nch in NCH_AB:
                sc = attention.AttnScratch(8, h.hl, h.qk, nch * 512 - 8, "cpu")
                yield f"ab/attention/r{R}/n{nch}", lambda q=q, sc=sc, nch=nch: attention.attention(
                    q, kc, vc, pos, sc, scale=scale, nch=nch)


def cases(world: int = 2):
    """Every case: (name, a call of python-0.6's wrappers) at one rank of ``world``."""
    h = Dims(world)
    for group in (glue_cases(h), qmm_cases(h), conv_cases(h), latent_cases(h), sparse_cases(h), ab_cases(h)):
        yield from group
