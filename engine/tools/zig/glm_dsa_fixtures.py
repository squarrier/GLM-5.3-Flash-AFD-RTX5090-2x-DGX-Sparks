#!/usr/bin/env python3
"""GLM-5.3-Flash's DSA block (latent DSA, indexer, sparse rows) as python-0.6's glm5_next/cuda runs it, recorded on a CPU."""
# qualification_qualification_44ba554e.5 A3 (evidence/qualification_qualification_7c75d777/PLAN.md §4), beside glm_forward_fixtures.py's cases, which calls `cases` and `mtp_cases` from
# its --dsa run: layer 3's attention half (forward.py 423-436) and whole layer, decode windows, with long contexts off
# (no indexer) and on (indexer; rows past the dense limit select 512 pools and attend sparsely), and the MTP layer's
# dsa_block as mtp.py calls it (its own cache, position and the indexer's last cache). Each case keeps every tensor it
# makes alive, so a temporary's storage is never reused under another name. select_tokens' torch ops after its
# kernels (integer index arithmetic) are checked against the list below and folded into one event, torch.select_tail;
# `tail_fixture` records what they compute on drawn pools (fixtures_cuda_select.json) for the Zig kernel's reference.
# qualification_qualification_44ba554e.5 A4 (--dsa-prompt, fixtures_cuda_dsa*.json): prompt chunks past 512 rows, whose dense blocks (dense_attention)
# and select blocks (select_tokens) start at `pos_dev + r0` (one torch.add event each) and whose blocks' pools are
# joined by torch.cat (one event), for layer 3 whole and the MTP layer's dsa_block as decode.prefill runs them.

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

import glm_forward_fixtures as gf
from tensorfold.families.glm5_next.cuda import decode, forward, glue, latent, sparse

LONG_CAPACITY = 6144          # qualification_6607f886's --context: room past the dense limit and 2,560 slots
POSITIONS = {False: (0, 37), True: (0, 37, 2048, 2560)}    # 2,048 with 4 or 8 rows straddles the dense limit (2,051)
QUIET = {"__get__", "size", "dim", "stride", "numel", "data_ptr", "untyped_storage", "element_size", "is_contiguous",
         "__len__", "__index__", "__int__", "__bool__", "__getitem__", "reshape", "view", "contiguous", "nbytes"}
# select_tokens after its last kernel (one block of rows): python-0.6 ed78d6f sparse.py 208-216, in order
TAIL_OPS = ("to", "arange", "add", "add", "floordiv", "empty", "mul", "arange", "add", "__setitem__", "mul",
            "arange", "add", "le", "where", "__setitem__", "gt", "sum", "add", "where", "to")
TAIL: dict = {"blocks": [], "mark": None, "cat": None}
# A4's temporaries: a block's position (pos_dev + r0) where the kernels read it, and the select blocks' torch.cat
BLOCK_TMP = {("torch.add", "out"), ("torch.cat", "blocks"), ("torch.cat", "out"), ("_dense_chunks", "POS"),
             ("_scores", "POS"), ("_select_rows", "POS")}


def _is_pos(t) -> bool:
    """A State's device position (st.pos_dev, st.mtp_pos_dev) itself, not a tensor made from it."""
    if not isinstance(t, torch.Tensor):
        return False
    p = gf.NAMES.by_ptr.get(t.untyped_storage().data_ptr())
    return p is not None and p[0].endswith("pos_dev")


class Keep(TorchFunctionMode):
    """Every tensor a case makes stays alive until the case ends (no storage reused under another temporary's name);
    the op names after `mark` are select_tokens' tail. `pos_dev + n` is logged as torch.add; the blocks' cat is kept."""

    def __init__(self) -> None:
        super().__init__()
        self.kept: list = []
        self.ops: list[str] = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", str(func))
        if name not in QUIET:
            self.ops.append(name.strip("_") if name not in ("__setitem__",) else name)
        self.kept.append(out)
        if name in ("add", "__add__") and len(args) == 2 and type(args[1]) is int and _is_pos(args[0]):
            gf.LOG.append({"op": "ext", "fn": "torch.add", "args": [{"t": gf.NAMES.tensor(args[0])}, {"int": args[1]},
                                                                    _tmp(out, ("torch.add", "out"))]})
        elif name == "cat" and TAIL["blocks"] and len(args[0]) == len(TAIL["blocks"]) and all(
                x is y for x, y in zip(args[0], TAIL["blocks"])):
            TAIL["cat"] = out
        return out


MODE: list[Keep] = []
REAL_SELECT, REAL_TOP = sparse.select_tokens, sparse.top_pools


def _tmp(t: torch.Tensor, site) -> dict:
    buf, off = gf.NAMES.where(t, site if isinstance(site, tuple) else ("select_tail", site))
    return {"t": {"buf": buf, "off": off, "dtype": gf.DTYPES[t.dtype], "shape": list(t.shape),
                  "stride": list(t.stride())}}


def _top(scores, k, pos_dev=None):
    out = REAL_TOP(scores, k, pos_dev)
    TAIL["blocks"].append(out)
    TAIL["mark"] = len(MODE[-1].ops)
    return out


def select_tokens(qi, wts, pk, pos, R, np_max, pos_dev, *, bucket=None):
    """python's own select_tokens; its kernels are recorded as usual, the blocks' cat (past SELECT_ROWS rows) and its
    tail become one event each."""
    TAIL["blocks"], TAIL["mark"], TAIL["cat"] = [], None, None
    sparse.top_pools = _top
    try:
        tokens, counts = REAL_SELECT(qi, wts, pk, pos, R, np_max, pos_dev, bucket=bucket)
    finally:
        sparse.top_pools = REAL_TOP
    blocks = TAIL["blocks"]
    ops = tuple(MODE[-1].ops[TAIL["mark"]:])
    if ops != (("cat",) if len(blocks) > 1 else ()) + TAIL_OPS:
        raise RuntimeError(f"select_tokens' tail is not python-0.6 ed78d6f's: {ops}")
    pools = blocks[0]
    if len(blocks) > 1:                           # sparse.py 204: torch.cat(blocks)
        pools = TAIL["cat"]
        gf.LOG.append({"op": "ext", "fn": "torch.cat", "args": [
            {"list": [_tmp(x, ("torch.cat", "blocks")) for x in blocks]}, _tmp(pools, ("torch.cat", "out"))]})
    gf.LOG.append({"op": "ext", "fn": "torch.select_tail", "ops": list(TAIL_OPS),
                   "args": [_tmp(pools, "pools"), {"t": gf.NAMES.tensor(pos_dev)}, {"int": R},
                            _tmp(tokens, "tokens"), _tmp(counts, "counts")]})
    TAIL["blocks"], TAIL["cat"] = [], None
    return tokens, counts


def _run(fn):
    """`fn()` with every tensor kept and select_tokens traced as one tail event."""
    gf.Names.TMP_OK |= BLOCK_TMP
    MODE.append(Keep())
    sparse.select_tokens = select_tokens
    try:
        with MODE[-1]:
            return fn()
    finally:
        sparse.select_tokens = REAL_SELECT
        MODE.pop()


def _window(w, pos: int, long: bool, max_rows: int):
    w.meta["long_context"] = long                 # State builds the indexer caches from it, as decode.Engine sets it
    try:
        return gf.window(w, pos, 0, max_rows, LONG_CAPACITY if long else 2560)
    finally:
        w.meta.pop("long_context", None)


def attn_half(w, layer: int, rows: int, pos: int, long: bool, max_rows: int) -> dict:
    """layer_forward's attention half (forward.py 423-436): hc_pre with attn_hc and in_norm, dsa_block, hc_post."""
    b, st = _window(w, pos, long, max_rows)
    nch = forward.chunks_for(st, rows)
    c, lw, x = w.cfg, w.layers[layer], b.x[:rows]

    def go():
        h = lw.attn_hc
        glue.hc_pre(x, h.fn, h.base, h.scale, lw.in_norm, b.normed[:rows], b.xs[:rows], b.post[:rows], b.comb[:rows],
                    b.hcpart[:rows], c.eps, c.hc_eps, c.hc_iters)
        di = st.dsa_index[lw.index]
        g = forward.dsa_block(lw, w, st.kc[di], st.vc[di], st.pos_dev, b, rows, nch,
                              st.index[di] if st.index is not None else None, pos, None)
        glue.hc_post(x, x, g, b.post[:rows], b.comb[:rows])

    _run(go)
    return gf.done(layer, rows, pos, 0, nch, max_rows, st.capacity, block="attn", long=long)


def whole_layer(w, layer: int, rows: int, pos: int, long: bool, max_rows: int) -> dict:
    """forward.layer_forward of a DSA + MoE layer, as compute calls it."""
    b, st = _window(w, pos, long, max_rows)
    nch = forward.chunks_for(st, rows)
    _run(lambda: forward.layer_forward(w.layers[layer], w, st, b, rows, nch, host_pos=pos))
    return gf.done(layer, rows, pos, 0, nch, max_rows, st.capacity, long=long)


def _prompt_buffers(w, st):
    """decode.Engine's pbuf: Buffers for prompt chunks (PREFILL_ROWS rows, prefill=True), every tensor named pb...."""
    pb = forward.Buffers(w, decode.PREFILL_ROWS, st.capacity, prefill=True)
    gf.NAMES.walk("pb", pb)
    return pb


def prompt_layer(w, layer: int, rows: int, pos: int, long: bool, max_rows: int) -> dict:
    """forward.layer_forward of a DSA + MoE layer on a prompt chunk, as decode.prefill runs it on the Engine's pbuf."""
    _, st = _window(w, pos, long, max_rows)
    pb = _prompt_buffers(w, st)
    nch = forward.chunks_for(st, rows)
    _run(lambda: forward.layer_forward(w.layers[layer], w, st, pb, rows, nch, host_pos=pos))
    return gf.done(layer, rows, pos, 0, nch, max_rows, st.capacity, long=long, prefill=decode.PREFILL_ROWS)


def mtp_attn(w, rows: int, mtp_len: int, long: bool, max_rows: int) -> dict:
    """mtp.py 44-45: dsa_block on w.mtp.layer with st.mtp_kc, st.mtp_pos_dev and st.index[-1], host_pos mtp_len."""
    b, st = _window(w, 0, long, max_rows)
    st.set_mtp_len(mtp_len)
    nch = -(-(mtp_len + rows) // 512)
    _run(lambda: forward.dsa_block(w.mtp.layer, w, st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, b, rows, nch,
                                   st.index[-1] if st.index is not None else None, mtp_len, None))
    return gf.done("mtp", rows, mtp_len, 0, nch, max_rows, st.capacity, block="dsa", long=long)


def state_buffers(w, long: bool, max_rows: int, prompt: bool = False) -> dict:
    """python's State for these cases (``st.*``: dtype, shape, bytes), beside the main load's top-level list; with
    ``prompt`` also the Engine's prompt Buffers (``pb.*``) the prompt-chunk cases ran on."""
    _, st = _window(w, 0, long, max_rows)
    if prompt:
        _prompt_buffers(w, st)
    return {k: v for k, v in gf.NAMES.buffers().items() if k.startswith(("st.", "pb.") if prompt else "st.")}


def cases(w, max_rows: int = 8) -> tuple[dict, dict]:
    """Layer 3's attention half at R = 1, 4, 8 (long contexts off: positions 0 and 37; on: 0, 37, 2,048, 2,560), the
    whole layer at (1, 0) and (4, 37), and on a prompt chunk (long contexts on: 64 and 512 rows from 0, qualification_6607f886's short
    prompt and one dense latent call's most); with python's long-context State's buffers and its prompt Buffers'."""
    out = {}
    for layer in range(w.cfg.layers):
        if w.layers[layer].kind != "dsa":
            continue
        for long in (False, True):
            for pos in POSITIONS[long]:
                for rows in (1, 4, 8):
                    if pos == 2048 and rows == 1:
                        continue
                    tag = "-long" if long else ""
                    out[f"l{layer}-attn{tag}-r{rows}-p{pos}"] = attn_half(w, layer, rows, pos, long, max_rows)
        for rows, pos in ((1, 0), (4, 37)):
            out[f"l{layer}-r{rows}-p{pos}"] = whole_layer(w, layer, rows, pos, False, max_rows)
        for rows in (64, 512):
            out[f"l{layer}-prompt-long-r{rows}-p0"] = prompt_layer(w, layer, rows, 0, True, max_rows)
    return out, state_buffers(w, True, max_rows, prompt=True)


def mtp_cases(w, max_rows: int = 8) -> tuple[dict, dict]:
    """The MTP layer's DSA arm: long contexts off at (1, 0) and (4, 37), on at (1, 37), (4, 2,048), (8, 2,560)."""
    out = {}
    for rows, mtp_len, long in ((1, 0, False), (4, 37, False), (1, 37, True), (4, 2048, True), (8, 2560, True)):
        out[f"mtp-dsa{'-long' if long else ''}-r{rows}-m{mtp_len}"] = mtp_attn(w, rows, mtp_len, long, max_rows)
    return out, {**state_buffers(w, False, max_rows), **{"long." + k: v for k, v in state_buffers(w, True, max_rows).items()}}


# A4: qualification_6607f886's long prompt (5,000 tokens: 2,048 rows at 0 and at 2,048, 904 at 4,096), a 2,085-token prompt's 37-row
# tail at 2,048 (it straddles the dense limit), and 1,537 rows at 0 (a last dense block of one row)
PROMPT_SHAPES = ((2048, 0), (37, 2048), (2048, 2048), (904, 4096), (1537, 0))
# the MTP head absorbs each chunk's next tokens (decode._absorb_rows): one row fewer on a prompt's last chunk
MTP_SHAPES = ((2048, 0), (36, 2048), (2048, 2048), (903, 4096))


def prompt_cases(w, max_rows: int = 8) -> tuple[dict, dict]:
    """Layer 3 whole on prompt chunks past 512 rows, with long contexts, on the Engine's pbuf (decode.prefill)."""
    out = {}
    for layer in range(w.cfg.layers):
        if w.layers[layer].kind == "dsa":
            for rows, pos in PROMPT_SHAPES:
                out[f"l{layer}-prompt-long-r{rows}-p{pos}"] = prompt_layer(w, layer, rows, pos, True, max_rows)
    return out, state_buffers(w, True, max_rows, prompt=True)


def mtp_prompt(w, rows: int, mtp_len: int, max_rows: int) -> dict:
    """mtp.py 44-45 on the Engine's pbuf: decode._absorb_rows runs mtp_forward over a prompt chunk's next tokens."""
    _, st = _window(w, 0, True, max_rows)
    st.set_mtp_len(mtp_len)
    pb = _prompt_buffers(w, st)
    nch = -(-(mtp_len + rows) // 512)
    _run(lambda: forward.dsa_block(w.mtp.layer, w, st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, pb, rows, nch,
                                   st.index[-1] if st.index is not None else None, mtp_len, None))
    return gf.done("mtp", rows, mtp_len, 0, nch, max_rows, st.capacity, block="dsa", long=True,
                   prefill=decode.PREFILL_ROWS)


def mtp_prompt_cases(w, max_rows: int = 8) -> tuple[dict, dict]:
    """The MTP layer's DSA arm on the long prompt's chunks and the 2,085-token prompt's tail."""
    out = {f"mtp-dsa-prompt-long-r{rows}-m{m}": mtp_prompt(w, rows, m, max_rows) for rows, m in MTP_SHAPES}
    return out, state_buffers(w, True, max_rows, prompt=True)


def prompt_fixture(a) -> int:
    """fixtures_cuda_dsa*.json (glm_forward_fixtures.py --dsa-prompt, after patch()): the main load's buffers and
    layer 3's prompt cases, then a second load with MTP on for the MTP layer's (its buffers and State beside)."""
    full, w = gf.load(a.mirror, a.layers, a.rank, a.world)
    cases, long_state = prompt_cases(w)
    b, st = forward.Buffers(w, 8, 2560), forward.State(w, 2560, 8)
    gf.NAMES.by_ptr.clear()
    for name, x in (("w", w), ("b", b), ("st", st)):
        gf.NAMES.walk(name, x)
    out = {"generator": "tools/zig/glm_forward_fixtures.py --dsa-prompt", "schema": "glm-forward-trace/1",
           "python": "ed78d6f", "rank": a.rank, "world": a.world, "layers": a.layers, "max_rows": 8, "capacity": 2560,
           "latent": latent.ENABLED, "config_json": (a.mirror / "config.json").read_text(),
           "config": gf.config_dict(full), "buffers": gf.NAMES.buffers(), "long_state": long_state}
    del w, b, st
    gf.NAMES.by_ptr.clear()
    gc.collect()
    _, w = gf.load(a.mirror, a.layers, a.rank, a.world, mtp=True)
    got, out["mtp_state"] = mtp_prompt_cases(w)
    cases.update(got)
    out["mtp_buffers"] = {k: v for k, v in gf.NAMES.buffers().items()
                          if k.startswith(("w.mtp.layer.moe.", "w.mtp.layer.dsa."))}
    out["cases"] = cases
    a.out.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"{len(cases)} cases, {sum(len(c['events']) for c in cases.values())} events -> {a.out}")
    return 0


def tail_fixture(out: Path) -> int:
    """select_tokens' tail on drawn pools: python's own code (its kernels recorded, top_pools giving the draw: splitmix64
    from the case's seed, each value modulo the pool count, row by row), each case's position, rows, the sha256 of its
    output (int32 [rows, 2051], little-endian), each row's last three and the counts (cuda_dsa_kernels.zig's reference;
    the digest's key names no "token": gitleaks reads a hex value under such a key as a credential)."""
    import hashlib

    gf.patch()
    cases = []
    for seed, (pos, R) in enumerate(((0, 1), (37, 4), (2044, 8), (2047, 4), (2048, 8), (2050, 2), (2051, 1), (2052, 3),
                                     (2559, 2), (2560, 8), (4093, 5), (6135, 8)), 1):
        np_max = sparse.pool_bucket(pos, R, LONG_CAPACITY // 4)
        g = gf.draws(seed)
        pools = torch.tensor([[next(g) % np_max for _ in range(sparse.TOPK_POOLS)] for _ in range(R)], dtype=torch.int64)
        qi = torch.zeros((R, 32 * 128), dtype=torch.bfloat16)
        wts = torch.zeros((R, 160), dtype=torch.bfloat16)[:, 128:]
        pk = torch.zeros((LONG_CAPACITY // 4 + 2, 128), dtype=torch.bfloat16)
        pos_dev = torch.tensor([pos], dtype=torch.int32)
        gf.NAMES.by_ptr.clear()
        for name, t in (("qi", qi), ("wts", wts), ("pk", pk), ("pos", pos_dev)):
            gf.NAMES.walk(name, t)
        sparse.top_pools = lambda scores, k, pos_dev=None, p=pools: p
        try:
            tokens, counts = REAL_SELECT(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev)
        finally:
            sparse.top_pools = REAL_TOP
        cases.append({"pos": pos, "rows": R, "np": np_max, "seed": seed, "counts": counts.tolist(),
                      "tails": tokens[:, -3:].tolist(),
                      "sha256": hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest()})
    gf.LOG.clear()
    doc = {"generator": "tools/zig/glm_dsa_fixtures.py (glm_forward_fixtures.py --select)", "schema": "glm-select-tail/1",
           "python": "ed78d6f", "cases": cases}
    out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"{len(cases)} select_tokens tail cases -> {out}")
    return 0
