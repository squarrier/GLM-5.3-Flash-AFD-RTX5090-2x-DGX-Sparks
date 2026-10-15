#!/usr/bin/env python3
"""GLM-5.3-Flash's two AFD roles as python 2.15's glm5_next/cuda runs them, recorded on a CPU: each launch in order."""
# qualification_064ac7c1's traces (evidence/qualification_25364d42/PLAN.md §5.1): python 2.15 (src@0849f3f) with every TF_GLM_* switch unset (its code
# defaults: BF16 dense, the path Zig ports), over qualification_765330ce's header-only mirror of Mia's EXL3 078455ff, through
# glm_forward_fixtures.py's recorders (ZK4's, after Mia's flashnext_triton_fixtures.py); schema glm-forward-trace/1.
# attn (G-trace-A): weights.load(world=1, experts=False) as engine.py's attention node, w.afd a recorder, Solo's
# all_gather. expert (G-trace-E): experts_node.ExpertNode.compute over load_half(h) and a landing buffer of python's;
# a 2,048-row window twice: as python defaults its prompt switches (on), and with them off (`-off`, "switches": "off").

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import types
from pathlib import Path

import torch

import glm_forward_fixtures as gf
from tensorfold.cuda.exl3 import experts as generic
from tensorfold.families.glm5_next.cuda import decode, exl3_generic, experts_node, forward, glue, latent, mcdma
from tensorfold.families.glm5_next.cuda import kda as kda_mod
from tensorfold.families.glm5_next.cuda import weights as wmod

PYTHON = "0849f3f"
ATTN_ROWS, ATTN_POS = (1, 2, 4, 8), ((0, 0), (37, 1))         # (position, KDA parity) as glm_forward_fixtures' cases
EXPERT_ROWS, EXPERT_LAYERS = (1, 2, 4, 8, 2048), (3, 44)
SKIP = ("caches", "slots")                     # State's pool and slot objects: State's own views carry 0.6's names
# exl3_generic.switches reads these at each call: a prompt window's cases again with Mia's launch switches off
SWITCHES_OFF = {"TF_GLM_EXL3_NFIRST": "0", "TF_GLM_EXL3_ORDER": "0", "TF_GLM_EXL3_PF": "0", "TF_GLM_EXL3_ROWROT": "0"}


class Afd:
    """``w.afd`` on the attention node (afd.AfdWire's forward hooks): nothing is sent; exchange and collect are logged
    as ``ext`` events. exchange: the rows on the wire (b.normed, b.pick, b.wts [:R]) and R. collect: the two landing
    planes of b.afd_recv, plane 2 and b.sy [:R] (python's own copy, done here), and R; it returns [3, R, D]."""

    def exchange(self, b, rows: int) -> None:
        gf.LOG.append({"op": "ext", "fn": "afd.exchange",
                       "args": [gf.val(b.normed[:rows]), gf.val(b.pick[:rows]), gf.val(b.wts[:rows]), gf.val(rows)]})

    def collect(self, b, rows: int, d: int):
        n = rows * d
        flat = b.afd_recv.view(-1)
        planes = [flat[k * n:(k + 1) * n].view(rows, d) for k in range(3)]
        gf.LOG.append({"op": "ext", "fn": "afd.collect", "args": [gf.val(p) for p in planes] + [gf.val(b.sy[:rows]),
                                                                                                gf.val(rows)]})
        planes[2].copy_(b.sy[:rows])                       # afd.py collect's plane-2 copy (a torch copy: no event)
        return b.afd_recv[:3 * n].view(3, rows, d)


class Solo(gf.Comm):
    """afd.Solo, the attention node's world-1 comm: its all_gather (a local copy of b.part into b.gath) is logged."""

    def __init__(self) -> None:
        super().__init__(0, 1)


def env_check() -> dict:
    """Every TF_GLM_* / TF_AFD_* / TF_GLM-prefixed switch must be unset: the fixtures are python 2.15's defaults."""
    bad = sorted(k for k in os.environ if k.startswith(("TF_GLM_", "TF_AFD")))
    if bad:
        raise SystemExit(f"switches set: {bad}; the fixtures record python 2.15's defaults")
    return {}


def cut_config(layers: int):
    read = wmod.Config.read

    def cut(cls, model_dir):
        c = read(model_dir)
        c.layers, c.kinds, c.mlp_kinds = layers, c.kinds[:layers], c.mlp_kinds[:layers]
        return c

    return read, classmethod(cut)


def load_attn(mirror: Path, layers: int):
    """engine.py's attention node (py215 engine.py:677-681): load(world=1, experts=False, mtp off), Solo, w.afd."""
    full = wmod.Config.read(mirror)
    read, cut = cut_config(layers)
    wmod.Config.read = cut
    try:
        w = wmod.load(mirror, rank=0, device="cpu", mtp=False, world=1, experts=False)
    finally:
        wmod.Config.read = read
    hw = w.head.weight
    if hw.untyped_storage().nbytes() != hw.numel() * hw.element_size():
        object.__setattr__(w.head, "weight", hw.clone())
    w.comm = Solo()
    w.afd = Afd()
    return full, w


def names(w, b, st) -> None:
    gf.NAMES.by_ptr.clear()
    gf.NAMES.tmp.clear()
    gf.NAMES.walk("w", w)
    gf.NAMES.walk("b", b)
    if st is not None:
        gf.NAMES.walk("st", types.SimpleNamespace(**{k: v for k, v in vars(st).items() if k not in SKIP}))


def window(w, pos: int, parity: int, max_rows: int, capacity: int):
    b = forward.Buffers(w, max_rows, capacity)
    st = forward.State(w, capacity, max_rows)
    st.set_pos(pos)
    st.cur = [parity] * len(st.cur)
    names(w, b, st)
    gf.LOG.clear()
    return b, st


def attn_case(w, layer: int, rows: int, pos: int, parity: int) -> dict:
    """layer_forward of a KDA layer whole (layer_front, the dense MLP, hc_post)."""
    b, st = window(w, pos, parity, 8, 2560)
    nch = forward.chunks_for(st, rows)
    forward.layer_forward(w.layers[layer], w, st, b, rows, nch, host_pos=pos)
    return gf.done(layer, rows, pos, parity, nch, 8, 2560)


def ffn_case(w, layer: int, rows: int) -> dict:
    """A MoE layer's FFN half (layer_front's hc_pre with ffn_hc, moe_block's AFD branch, hc_post over 3 planes)."""
    b, st = window(w, 0, 0, 8, 2560)
    c, lw, x = w.cfg, w.layers[layer], b.x[:rows]
    h = lw.ffn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, lw.post_norm, b.normed[:rows], b.xs[:rows], b.post[:rows], b.comb[:rows],
                b.hcpart[:rows], c.eps, c.hc_eps, c.hc_iters)
    g = forward.moe_block(lw, w, b, rows)
    glue.hc_post(x, x, g, b.post[:rows], b.comb[:rows])
    return gf.done(layer, rows, 0, 0, forward.chunks_for(st, rows), 8, 2560, block="ffn")


def head_case(w, rows: int) -> dict:
    """compute's tail after the last layer (py215 forward.py:868-876): the stream mean, the final norm, the head."""
    b, st = window(w, 0, 0, 8, 2560)
    c = w.cfg
    glue.stream_mean(b.x[:rows], b.hidden[:rows])
    glue.rmsnorm(b.hidden[:rows], w.norm, c.eps, b.fnormed[:rows], b.fxs[:rows])
    forward.mm(b, b.fnormed[:rows], w.head, b.fxs[:rows], b.logits[:rows])
    return gf.done("head", rows, 0, 0, forward.chunks_for(st, rows), 8, 2560, block="head")


def attn(mirror: Path, out: Path, layers: int) -> int:
    env = env_check()
    gf.patch()
    full, w = load_attn(mirror, layers)
    cases = {}
    for layer in range(layers):
        if w.layers[layer].kind == "kda":
            for rows in ATTN_ROWS:
                for pos, parity in ATTN_POS:
                    cases[f"l{layer}-r{rows}-p{pos}"] = attn_case(w, layer, rows, pos, parity)
    for layer in range(layers):
        if w.layers[layer].moe is not None:
            for rows in ATTN_ROWS:
                cases[f"l{layer}-ffn-r{rows}"] = ffn_case(w, layer, rows)
    for rows in ATTN_ROWS:
        cases[f"head-r{rows}"] = head_case(w, rows)
    b = forward.Buffers(w, 8, 2560)
    st = forward.State(w, 2560, 8)
    names(w, b, st)
    doc = {"generator": "tools/zig/glm_afd_fixtures.py attn", "schema": "glm-forward-trace/1", "python": PYTHON,
           "role": "attn", "env": env, "rank": 0, "world": 1, "layers": layers, "max_rows": 8, "capacity": 2560,
           "latent": latent.ENABLED, "config_json": (mirror / "config.json").read_text(),
           "config": gf.config_dict(full), "buffers": gf.NAMES.buffers(), "cases": cases}
    return write(out, doc)


# -- G-trace-A full (qualification_0fa2bee9): every layer at world 1, long contexts on; decode R = 1, 4 at 0, 37, 2,560; one chunk ----
LONG_CAPACITY = 6144                    # qualification_6607f886's --context: glm_dsa_fixtures' long-context State (A3's cases)
FULL_ROWS, FULL_POS = (1, 4), (0, 37, 2560)


def full_window(w, pos: int, prompt: bool):
    """The attention node's decode.Engine with long_context=True (State builds the indexer caches from w.meta): b (8
    rows), st from ``pos`` (KDA parity pos % 2), and with ``prompt`` the Engine's pbuf (PREFILL_ROWS rows, prefill)."""
    w.meta["long_context"] = True
    try:
        b = forward.Buffers(w, 8, LONG_CAPACITY)
        st = forward.State(w, LONG_CAPACITY, 8)
    finally:
        w.meta.pop("long_context", None)
    st.set_pos(pos)
    st.cur = [pos % 2] * len(st.cur)
    pb = forward.Buffers(w, decode.PREFILL_ROWS, LONG_CAPACITY, prefill=True) if prompt else None
    names(w, b, st)
    if pb is not None:
        gf.NAMES.walk("pb", pb)
    gf.LOG.clear()
    return b, st, pb


def full_case(dsa, w, layer: int, rows: int, pos: int) -> dict:
    """layer_forward of any layer (a DSA layer's indexer and, from 2,560, its sparse rows; a MoE layer's FFN half
    through w.afd), every tensor kept and select_tokens' tail one event (glm_dsa_fixtures._run)."""
    b, st, _ = full_window(w, pos, False)
    nch = forward.chunks_for(st, rows)
    dsa._run(lambda: forward.layer_forward(w.layers[layer], w, st, b, rows, nch, host_pos=pos))
    return gf.done(layer, rows, pos, pos % 2, nch, 8, LONG_CAPACITY, long=True)


def chunk_case(dsa, w, layer: int, rows: int, pos: int = 0) -> dict:
    """A prompt chunk's layer as decode.prefill runs it under AFD (decode.py:555-556: afd.begin MODE_CHUNK, compute on
    the Engine's pbuf), from ``pos``. A KDA layer's wide chain takes kda._wide_scratch's q and y, made anew for the
    case (as glm_step_fixtures' name_engine clears kda._tmp): the case's temporaries, which ext calls name too."""
    _, st, pb = full_window(w, pos, True)
    kda_mod._tmp.clear()
    if w.layers[layer].kind == "kda" and rows >= kda_mod.WIDE_ROWS:
        for t in kda_mod._wide_scratch(rows, w.cfg.lin_heads // w.world, "cpu"):
            key = t.untyped_storage().data_ptr()
            info = {"dtype": gf.DTYPES[t.dtype], "shape": list(t.shape), "nbytes": t.untyped_storage().nbytes()}
            gf.NAMES.tmp[key] = (f"tmp.{len(gf.NAMES.tmp)}", info)
    nch = forward.chunks_for(st, rows)
    dsa._run(lambda: forward.layer_forward(w.layers[layer], w, st, pb, rows, nch, host_pos=pos))
    return gf.done(layer, rows, pos, pos % 2, nch, 8, LONG_CAPACITY, long=True, prefill=decode.PREFILL_ROWS)


def chunk_head(w, rows: int) -> dict:
    """compute's tail on a prompt chunk (forward.py:868-876, b.prefill): the stream mean, the final norm over every
    row (the MTP reads b.fnormed), the head on the last row."""
    _, st, pb = full_window(w, 0, True)
    assert pb is not None
    c = w.cfg
    glue.stream_mean(pb.x[:rows], pb.hidden[:rows])
    glue.rmsnorm(pb.hidden[:rows], w.norm, c.eps, pb.fnormed[:rows], pb.fxs[:rows])
    forward.mm(pb, pb.fnormed[rows - 1:rows], w.head, pb.fxs[rows - 1:rows], pb.logits[:1])
    return gf.done("head", rows, 0, 0, forward.chunks_for(st, rows), 8, LONG_CAPACITY, block="head", long=True,
                   prefill=decode.PREFILL_ROWS)


def attn_full(mirror: Path, out: Path, layers: int) -> int:
    """G-trace-A full's cases (PLAN §5.1): each layer whole at R = 1, 4 from 0, 37 and 2,560, each layer on one
    2,048-row prompt chunk from 0, the chunk's head; beside them, where A3's DSA block takes prompt chunks (512 rows at
    most), a DSA layer's 512-row chunk from 0 and each layer's 37-row chunk from 2,048 (a 2,085-token prompt's tail:
    rows past the dense limit select sparsely); python's long-context buffers (b, st, pb) in `buffers`."""
    import glm_dsa_fixtures as dsa

    dsa.gf = gf
    where = gf.Names.where

    def named(self, t, site=None):              # a case's temporaries as any argument names them (ext calls too)
        key = t.untyped_storage().data_ptr()
        if key not in self.by_ptr and key in self.tmp:
            return self.tmp[key][0], t.data_ptr() - key
        return where(self, t, site)

    gf.Names.where = named
    # a prompt chunk past part_rows rows: each dense block's first position, pos_dev + r0, a device add's temporary
    # (forward.dense_attention; A3's NotPorted, lane A's A4)
    gf.Names.TMP_OK = gf.Names.TMP_OK | {("_dense_chunks", "POS")}
    env = env_check()
    gf.patch()
    full, w = load_attn(mirror, layers)
    cases = {}
    for layer in range(layers):
        for rows in FULL_ROWS:
            for pos in FULL_POS:
                cases[f"l{layer}-r{rows}-p{pos}"] = full_case(dsa, w, layer, rows, pos)
    for layer in range(layers):
        cases[f"l{layer}-chunk-r{decode.PREFILL_ROWS}"] = chunk_case(dsa, w, layer, decode.PREFILL_ROWS)
    cases[f"head-chunk-r{decode.PREFILL_ROWS}"] = chunk_head(w, decode.PREFILL_ROWS)
    for layer in range(layers):
        if w.layers[layer].kind == "dsa":
            cases[f"l{layer}-chunk-r512"] = chunk_case(dsa, w, layer, 512)
        cases[f"l{layer}-chunk-r37-p2048"] = chunk_case(dsa, w, layer, 37, 2048)
    full_window(w, 0, True)
    doc = {"generator": "tools/zig/glm_afd_fixtures.py attn-full", "schema": "glm-forward-trace/1", "python": PYTHON,
           "role": "attn", "env": env, "rank": 0, "world": 1, "layers": layers, "max_rows": 8,
           "capacity": LONG_CAPACITY, "long": True, "prefill_rows": decode.PREFILL_ROWS, "latent": latent.ENABLED,
           "config_json": (mirror / "config.json").read_text(), "config": gf.config_dict(full),
           "buffers": gf.NAMES.buffers(), "cases": cases}
    return write(out, doc)


# -- the expert node ------------------------------------------------------------------------------------------------
class Node(experts_node.ExpertNode):
    """ExpertNode as experts_node.ExpertNode.__init__ builds it, on the CPU: load_half's trellises pass prepare's
    device check (glm_forward_fixtures.prepare_cpu), the buffers are python's own."""

    def __init__(self, mirror: Path, half: int, layers, rows: int, prefill_rows: int) -> None:
        generic.prepare = gf.prepare_cpu
        try:
            super().__init__(mirror, half, layers, device="cpu", rows=rows, prefill_rows=prefill_rows)
        finally:
            generic.prepare = gf.PREPARE


def expert_names(node, link) -> None:
    w = types.SimpleNamespace(layers={i: types.SimpleNamespace(moe=types.SimpleNamespace(experts=ex))
                                      for i, ex in node.by_layer.items()})
    gf.NAMES.by_ptr.clear()
    gf.NAMES.tmp.clear()
    gf.NAMES.walk("w", w)
    gf.NAMES.walk("b", node.buf)
    gf.NAMES.walk("link", link)


def expert_case(node, link, layer: int, rows: int) -> dict:
    """One MoE layer of a forward on the expert (experts_node._window's body): land's views of the request (python's
    landing buffer, the header slot skipped), then compute; the reply is the link's (no event)."""
    lay = mcdma.Layout(node.cfg.hidden, node.cfg.top_k + 1)
    x = lay.views(link.landing, rows)
    node.prompt = rows > node.buf.rows             # a 2,048-row window is a prompt chunk's (MODE_CHUNK), as forwards sets
    expert_names(node, link)
    gf.LOG.clear()
    node.compute(layer, rows, x)
    out = {"layer": layer, "rows": rows, "half": node.half, "prompt": node.prompt, "events": list(gf.LOG)}
    if gf.NAMES.tmp:
        out["tmp"] = dict(gf.NAMES.tmp.values())
    return out


def expert(mirror: Path, out: Path) -> int:
    env = env_check()
    gf.patch()
    cfg = wmod.Config.read(mirror)
    cases, buffers = {}, None
    for half in (0, 1):
        node = Node(mirror, half, EXPERT_LAYERS, 8, 2048)
        lay = mcdma.Layout(cfg.hidden, cfg.top_k + 1)
        link = types.SimpleNamespace(landing=torch.empty((lay.req(2048) - mcdma.HDR_BYTES,), dtype=torch.uint8))
        for layer in EXPERT_LAYERS:
            for rows in EXPERT_ROWS:
                cases[f"h{half}-l{layer}-r{rows}"] = expert_case(node, link, layer, rows)
                if rows >= exl3_generic.PROMPT_ROWS:          # python's prompt switches off as well (qualification_cec3c04c's ask)
                    os.environ.update(SWITCHES_OFF)
                    try:
                        case = expert_case(node, link, layer, rows)
                    finally:
                        for k in SWITCHES_OFF:
                            del os.environ[k]
                    cases[f"h{half}-l{layer}-r{rows}-off"] = dict(case, switches="off", env=SWITCHES_OFF)
        expert_names(node, link)
        mine = gf.NAMES.buffers()
        if buffers is not None and mine != buffers:
            raise SystemExit("half 1's buffers differ from half 0's in path, dtype or shape")
        buffers = mine
        del node, link
        gf.NAMES.by_ptr.clear()
        gc.collect()
    doc = {"generator": "tools/zig/glm_afd_fixtures.py expert", "schema": "glm-forward-trace/1", "python": PYTHON,
           "role": "expert", "env": env, "world": 2, "halves": [0, 1], "layers": list(EXPERT_LAYERS), "max_rows": 8,
           "prefill_rows": 2048, "config_json": (mirror / "config.json").read_text(), "config": gf.config_dict(cfg),
           "buffers": buffers, "cases": cases}
    return write(out, doc)


def write(out: Path, doc: dict) -> int:
    out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    n = sum(len(c["events"]) for c in doc["cases"].values())
    print(f"{doc['role']}: {len(doc['cases'])} cases, {n} events, {len(doc['buffers'])} buffers -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("role", choices=("attn", "attn-full", "expert"))
    ap.add_argument("--mirror", required=True, type=Path, help="the EXL3 checkpoint (a header-only sparse mirror serves)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=4, help="attn: layers loaded (Config cut to the first n)")
    a = ap.parse_args()
    if a.role == "attn":
        return attn(a.mirror, a.out, a.layers)
    if a.role == "attn-full":
        return attn_full(a.mirror, a.out, a.layers)
    return expert(a.mirror, a.out)


if __name__ == "__main__":
    sys.exit(main())
