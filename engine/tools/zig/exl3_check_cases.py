"""exl3_check.py's case table on layer L's routed experts of TP2 rank R, as python-0.6 builds and
calls them: experts.prepare and experts.routed (GLM's exl3_generic), the extension's combine, Exl3Linear and its unpack,
experts.dequant's kernel, and GLM's exl3_mm.routed over experts.route's plan. Every output is zeroed and every scratch
fresh (members -1) before its call; inputs are seeded (numpy PCG64). Python-0.6's code is Ash Hart's and the
TensorFold contributors'; the EXL3 format turboderp's (ExLlamaV3, MIT).
"""

from __future__ import annotations

import hashlib
import os
import time

from exl3_check import COMBINE, EXPERTS, FP16_LIMIT, GLM_MM, LINEAR_M, PREFIX, ROUTED, SLOTS, normal, picks, \
    scratch_bytes, weights


def sha(run, t) -> str:
    return hashlib.sha256(memoryview(run.raw(t))).hexdigest()


def f32bits(x: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", x))[0]


def timed(run, fn) -> float:
    """fn() between two device syncs (nothing runs in --dry); its milliseconds."""

    if run.a.dry:
        return 0.0
    run.sync()
    t = time.time()
    fn()
    run.sync()
    return round((time.time() - t) * 1e3, 3)


def routed_cases(run, m, ex, generic, exl3_generic, ys: dict) -> None:
    torch, dev, E, D = run.torch, run.dev, ex.count, ex.dims
    for kind, rows in ROUTED.items():
        for R in rows:
            cid = f"{kind}-{R}"
            xd = torch.float16 if kind == "routed-fp16" else torch.bfloat16
            act = generic.ACT_F32 if kind == "routed-f32" else generic.ACT_BF16
            limit = FP16_LIMIT if kind == "routed-fp16" else m["limit"]
            x, pick = normal(run, cid, R, D, xd), picks(run, cid, R, E, hot=kind == "routed-hot")
            wts = weights(run, cid, R) if kind == "routed-wts" else None
            s = generic.Scratch(ex, R, SLOTS, device=dev)             # zeros, members -1: as a fresh window finds it
            out = torch.zeros((R, D), dtype=torch.float32, device=dev) if wts is not None else None
            xi, pi = x.to(dev), pick.to(dev)
            wi = wts.to(dev) if wts is not None else None
            if kind == "routed-bf16":                                  # GLM's served call (forward.moe_block)
                ms = timed(run, lambda: exl3_generic.routed(xi, pi, ex, s, R, limit))
            else:
                ms = timed(run, lambda: generic.routed(xi, pi, wi, ex, s, out, R, limit, act_mode=act))
            P = R * SLOTS
            ins = [run.inp(cid, "x", x), run.inp(cid, "pick", pick)] + ([run.inp(cid, "wts", wts)] if wts is not None
                                                                         else [])
            outs = [run.outp(cid, "y", s.y[:P])] + ([run.outp(cid, "out", out)] if out is not None else [])
            call = {"R": R, "rows": R, "slots": SLOTS, "act": "f32" if act == generic.ACT_F32 else "bf16",
                    "limit": limit, "limit_f32_bits": f32bits(limit), "x_dtype": "f16" if xd == torch.float16
                    else "bf16", "weighted": wts is not None, "hot": kind == "routed-hot", "group": True,
                    "cfg_gu": list(s.cfg_gu), "cfg_d": list(s.cfg_d), "ms": ms,
                    "python": "exl3_generic.routed(x, pick, ex, s, R, limit)" if kind == "routed-bf16" else
                    "experts.routed(x, pick, wts, ex, s, out, R, limit, act_mode)"}
            run.case(cid, kind, call, ins, outs, scratch_bytes(xg=s.xg, xu=s.xu, xd=s.xd, z=s.z, y=s.y, ids=s.ids,
                                                               count=s.count, members=s.members_buf))
            if kind == "routed-bf16" and R in COMBINE:
                ys[R] = s.y[:P].clone()
            del s, out, xi, pi, wi
        run.mark(f"{kind}: {len(rows)} cases")


def combine_cases(run, ex, generic, ys: dict) -> None:
    torch, dev, D = run.torch, run.dev, ex.dims
    for R in COMBINE:
        cid = f"combine-{R}"
        wts = weights(run, cid, R)
        out = torch.zeros((R, D), dtype=torch.float32, device=dev)
        y, wi = ys[R], wts.to(dev)
        ms = timed(run, lambda: generic._ext().combine(y, wi, out, R, D, SLOTS))
        run.case(cid, "combine", {"R": R, "slots": SLOTS, "ms": ms, "y_from": f"routed-bf16-{R}",
                                  "python": "experts._ext().combine(y, wts, out, R, D, slots)"},
                 [run.inp(cid, "y", y, derived=True), run.inp(cid, "wts", wts)], [run.outp(cid, "out", out)], {})
    run.mark(f"combine: {len(COMBINE)} cases")


def linear_cases(run, host, L: int, lin) -> None:
    torch, dev = run.torch, run.dev
    for proj in ("gate", "down"):
        for e in EXPERTS:
            g = lambda x: host[PREFIX + f"layers.{L}.mlp.experts.{e}.{proj}_proj.{x}"]  # noqa: E731
            for layout in ("strips", "stored"):
                layer = lin.Exl3Linear.from_tensors(g("trellis"), g("suh"), g("svh"), "mcg", None, dev, layout)
                K, N, (sk, wk) = layer.k, layer.n, layer.split
                base = {"proj": proj, "expert": e, "layout": layout, "k": K, "n": N, "k2": layer.k2, "cb": 1,
                        "split": [sk, wk]}
                for M in LINEAR_M:
                    for od in (torch.bfloat16, torch.float32):
                        cid = f"linear-{proj}-e{e}-{M}-{layout}-{'bf16' if od == torch.bfloat16 else 'f32'}"
                        x = normal(run, cid, M, K, torch.bfloat16)
                        y = torch.zeros((M, N), dtype=od, device=dev)
                        xh = torch.zeros((M, K), dtype=torch.float16, device=dev)
                        z = torch.zeros((sk * M * N,), dtype=torch.float32, device=dev) if sk > 1 else None
                        layer.counters.zero_()                         # split-K's counters start at zero
                        xi = x.to(dev)
                        ms = timed(run, lambda: layer(xi, out=y, xh=xh, z=z))
                        call = dict(base, M=M, out_dtype="bf16" if od == torch.bfloat16 else "f32", ms=ms,
                                    python="Exl3Linear.from_tensors(trellis, suh, svh, 'mcg', None, dev, layout)(x, "
                                    "out=y, xh=xh, z=z)")
                        run.case(cid, "linear", call, [run.inp(cid, "x", x)], [run.outp(cid, "y", y)],
                                 scratch_bytes(xh=xh, z=z, counters=layer.counters))
                cid = f"unpack-{proj}-e{e}-{layout}"
                w = torch.zeros((K, N), dtype=torch.float16, device=dev)
                ms = timed(run, lambda: layer.unpack(out=w))
                run.case(cid, "unpack", dict(base, ms=ms, python="Exl3Linear.unpack(out=w)"), [],
                         [run.outp(cid, "w", w)], {})
                del layer
    run.mark("linear and unpack")


def dequant_cases(run, host, L: int, generic) -> None:
    torch, dev = run.torch, run.dev
    for proj in ("gate", "up", "down"):
        for e in EXPERTS:
            cid = f"dequant-{proj}-e{e}"
            t = host[PREFIX + f"layers.{L}.mlp.experts.{e}.{proj}_proj.trellis"].to(dev).contiguous()
            K, N = 16 * t.shape[0], 16 * t.shape[1]
            out = torch.zeros((K, N), dtype=torch.float16, device=dev)
            k2, cb = generic.k2_of(t), generic.codebook_id("mcg")
            ms = timed(run, lambda: generic._ext().dequant(t, out, k2, cb))
            run.case(cid, "dequant", {"proj": proj, "expert": e, "k": K, "n": N, "k2": k2, "cb": cb, "ms": ms,
                                      "python": "experts.dequant(trellis, 'mcg')'s call, its output zeroed first"},
                     [], [run.outp(cid, "out", out)], {})
    run.mark("dequant")


def glm_cases(run, m, host, L: int, grouped, exl3_mm) -> None:
    """GLM's own EXL3 kernels: exl3_mm.Exl3Experts from words(stack), the pairs planned by experts.route."""

    torch, dev, E, D, NI = run.torch, run.dev, m["count"], m["dims"], m["width"]

    def stack(proj, x):
        return torch.stack([host[PREFIX + f"layers.{L}.mlp.experts.{e}.{proj}_proj.{x}"] for e in range(E)]).to(dev)
    ex = exl3_mm.Exl3Experts(exl3_mm.words(stack("gate", "trellis")), exl3_mm.words(stack("up", "trellis")),
                             exl3_mm.words(stack("down", "trellis")), stack("gate", "suh"), stack("up", "suh"),
                             stack("gate", "svh"), stack("up", "svh"), stack("down", "suh"), stack("down", "svh"),
                             E, NI, D)
    run.mark("glm-mm: the stacked experts uploaded")
    for R in GLM_MM:
        cid = f"glm-mm-{R}"
        x, pick = normal(run, cid, R, D, torch.bfloat16), picks(run, cid, R, E)
        plan = grouped.Plan(R, SLOTS, E + 1, dev)
        s = exl3_mm.Scratch(R, SLOTS, D, NI, dev)
        y = torch.zeros((R * SLOTS, D), dtype=torch.float32, device=dev)
        xi, pi = x.to(dev), pick.to(dev)
        timed(run, lambda: grouped.route(pi, plan))                    # the plan: an input of the EXL3 kernels
        ins = [run.inp(cid, "x", x), run.inp(cid, "pick", pick), run.inp(cid, "items", plan.items, derived=True),
               run.inp(cid, "counts", plan.counts, derived=True), run.inp(cid, "members", plan.members, derived=True)]
        ms = timed(run, lambda: exl3_mm.routed(xi, pi, plan, ex, s, y, R, m["limit"]))
        call = {"R": R, "rows": R, "slots": SLOTS, "experts": E + 1, "limit": m["limit"],
                "limit_f32_bits": f32bits(m["limit"]), "ms": ms,
                "python": "experts.route(pick, Plan(R, 9, E + 1)); exl3_mm.routed(x, pick, plan, ex, s, y, R, limit)"}
        run.case(cid, "glm-mm", call, ins, [run.outp(cid, "y", y)], scratch_bytes(xg=s.xg, xu=s.xu, xd=s.xd, z=s.z))
        del plan, s, y, xi, pi
    run.mark(f"glm-mm: {len(GLM_MM)} cases")


def run_all(run, names_of, OnCuda) -> dict:
    """Read, build and run every case; the manifest's head (the layer's facts and the dry listing's first lines)."""

    torch, a, dev = run.torch, run.a, run.dev
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.exl3 import experts as generic
    from tensorfold.cuda.exl3 import linear as lin
    from tensorfold.families.glm5_next.cuda import exl3_generic, exl3_mm, split
    from tensorfold.families.glm5_next.cuda.weights import Config

    cfg = Config.read(a.ckpt)
    L = a.layer
    if not (cfg.mlp_kinds[L] == "moe" if L < cfg.layers else L == cfg.layers and cfg.mtp_layers):
        raise SystemExit(f"layer {L} has no routed experts")
    E = cfg.experts
    rd = split.RankReader(a.ckpt, a.rank, 2)
    names = names_of(L, E)
    t = time.time()
    rd.prefetch(names)                                                   # weights.load's reads of a layer's experts
    host = {n: rd.get(n) for n in names}
    rd.close()
    nb = sum(v.numel() * v.element_size() for v in host.values())
    run.mark(f"layer {L} rank {a.rank}: {len(names)} tensors, {nb} B read in {time.time() - t:.1f} s")

    def trip(proj, e):                                                   # moe_exl3's (trellis, suh, svh) .to(dev)
        g = lambda x: host[PREFIX + f"layers.{L}.mlp.experts.{e}.{proj}_proj.{x}"].to(dev)  # noqa: E731
        tr = g("trellis")
        return OnCuda(tr) if a.dry else tr, g("suh"), g("svh")
    ex = generic.prepare([trip("gate", e) for e in range(E)], [trip("up", e) for e in range(E)],
                         [trip("down", e) for e in range(E)], "mcg", device=dev)
    run.sync()
    run.mark("experts.prepare: the layer on the device")
    m = {"count": ex.count, "dims": ex.dims, "width": ex.width, "cb": ex.cb, "k2_gu": list(ex.k2_gu),
         "k2_d": list(ex.k2_d), "limit": cfg.limit, "codebook": "mcg"}
    tsha = {p: [hashlib.sha256(memoryview(run.raw(host[PREFIX + f"layers.{L}.mlp.experts.{e}.{p}_proj.trellis"])))
                .hexdigest() for e in range(E)] for p in ("gate", "up", "down")}
    m["trellis_bytes"] = {p: host[PREFIX + f"layers.{L}.mlp.experts.0.{p}_proj.trellis"].numel() * 2
                          for p in ("gate", "up", "down")}
    m["trellis_sha256"] = tsha
    m["scales_sha256"] = {k: sha(run, getattr(ex, k)) for k in ("suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d")}
    m["scales_shape"] = {k: list(getattr(ex, k).shape) for k in ("suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d")}
    m["k2_sha256"] = {k: sha(run, getattr(ex, k + "_k2")) for k in ("gate", "up", "down")}
    tb = " ".join(f"{p}={m['trellis_bytes'][p]}" for p in ("gate", "up", "down"))
    head = [f"layer {L} rank {a.rank}/2: {E} experts, dims {ex.dims}, width {ex.width}, codebook {ex.cb}, "
            f"k2 gate/up {ex.k2_gu[0]}-{ex.k2_gu[1]}, down {ex.k2_d[0]}-{ex.k2_d[1]}, limit {cfg.limit}",
            "layout trellis bytes an expert: " + tb,
            "layout scales: " + " ".join(f"{k}:f16[{','.join(map(str, v))}]" for k, v in m["scales_shape"].items())]
    ys: dict = {}
    if os.environ.get("TF_EXL3_CHECK_K4", "") == "1":                   # qualification_e4ad04cd: K4's table alone (exl3_check_k4.py)
        from exl3_check_k4 import k4_cases
        k4_cases(run, m, ex, generic, exl3_generic)
        head[0] += f", K4 (shared suh {int(m['shared_suh'])})"
        return {"layer_facts": m, "listing_head": head, "read_bytes": nb}
    routed_cases(run, m, ex, generic, exl3_generic, ys)
    combine_cases(run, ex, generic, ys)
    del ys
    linear_cases(run, host, L, lin)
    dequant_cases(run, host, L, generic)
    del ex
    if not a.dry:
        torch.cuda.empty_cache()                                         # the per-expert copies go before the stacks
    glm_cases(run, m, host, L, grouped, exl3_mm)
    return {"layer_facts": m, "listing_head": head, "read_bytes": nb}
