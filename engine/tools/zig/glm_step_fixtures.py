#!/usr/bin/env python3
"""GLM-5.3-Flash's whole decode step as python-0.6's glm5_next/cuda runs it, recorded on a CPU: nothing is launched."""
# qualification_qualification_44ba554e.5 B2's launch trace (evidence/qualification_qualification_7c75d777/PLAN.md §4), beside glm_forward_fixtures.py's layer cases: python-0.6's own
# weights.load of a rank (CPU) over a checkpoint's header-only mirror, cut to n layers with the MTP head read from the
# checkpoint's layers.<num_hidden_layers>. (qualification_6607f886's 8-layer-copy rule), then decode steps (forward.stage, compute,
# decode.sample_rows greedy, forward.commit) and mtp_stage + mtp_compute. glm_forward_fixtures.py's recorders log the
# Triton launches, extension calls and all_gather; a TorchFunctionMode logs the torch ops that do device work (copies,
# fills, the sampler's float/topk/add/to/cat, .cpu()), each temporary named by the op that makes it (tmp.<op>).

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

import glm_forward_fixtures as gf
from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3 import experts as exl3_experts
from tensorfold.families.glm5_next.cuda import decode, forward, split
from tensorfold.families.glm5_next.cuda import kda as kda_mod
from tensorfold.families.glm5_next.cuda import mtp as mtp_mod
from tensorfold.families.glm5_next.cuda import weights as wmod

DEVICE_OPS = {"copy_", "fill_", "zero_", "float", "topk", "add", "__add__", "to", "cat", "cpu", "clone"}
ALLOCS = {"empty", "empty_like"}                       # a new tensor and no device work: named, not logged
IN_PLACE = {"copy_", "fill_", "zero_", "cpu"}         # no new device tensor (.cpu() is the host's; self on a CPU)
QUIET = {"__getitem__", "view", "numpy", "__get__", "stride", "size", "dim", "numel", "data_ptr", "untyped_storage",
         "element_size", "__len__", "is_contiguous", "tolist", "__index__", "__int__", "__bool__", "contiguous",
         "reshape", "_set_grad_enabled"}
STATE = {"host": set(), "ids_host": set(), "tokens": None, "count": {}}


def _name(op: str) -> str:
    return op[2:-2] if op.startswith("__") and op.endswith("__") else op


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)


def _arg(v):
    if isinstance(v, torch.Tensor):
        return {"t": gf.NAMES.tensor(v)}
    if isinstance(v, (list, tuple)):
        return {"list": [_arg(x) for x in v]}
    if isinstance(v, (torch.dtype, torch.device, torch.memory_format)):
        return None                                   # the output tensor's dtype says it
    return gf.val(v)


def _register(op: str, out) -> None:
    """A new tensor an op made: named tmp.<op> (tmp.<op>.<field> for topk's values and indices)."""
    n = STATE["count"].get(op, 0)
    STATE["count"][op] = n + 1
    base = "tmp." + _name(op) + (f".{n}" if n else "")
    if isinstance(out, torch.Tensor):
        gf.NAMES.by_ptr[_ptr(out)] = (base, out)
        return
    fields = getattr(out, "_fields", None) or {"topk": ("values", "indices")}.get(op) or [str(i) for i in range(len(out))]
    for field, t in zip(fields, out):
        gf.NAMES.by_ptr[_ptr(t)] = (f"{base}.{field}", t)


class Torch(TorchFunctionMode):
    """Logs every torch op with device work as an ``ext`` event ``torch.<op>``: its arguments, then its new outputs.
    Work on host tensors (the pinned ids, what .cpu() returned) is the host's and is not traced, except the token write."""

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        op = getattr(func, "__name__", str(func))
        out = func(*args, **kwargs)
        ins = list(_tensors(list(args) + list(kwargs.values())))
        if ins and all(_ptr(t) in STATE["host"] for t in ins):
            if op == "numpy" and _ptr(ins[0]) in STATE["ids_host"]:
                gf.LOG.append({"op": "ext", "fn": "host.write",
                               "args": [{"t": gf.NAMES.tensor(ins[0])}, gf.val(list(STATE["tokens"]))]})
            for t in _tensors(out):
                STATE["host"].add(_ptr(t))
            return out
        if op in ALLOCS:
            _register(op, out)
        elif op in ("contiguous", "reshape") and _ptr(out) != _ptr(ins[0]):
            raise RuntimeError(f"{op} copied a device tensor: work the trace does not name")
        elif op in DEVICE_OPS:
            vals = [_arg(a) for a in args] + [_arg(kwargs[k]) for k in sorted(kwargs)]
            if op not in IN_PLACE:
                _register(op, out)
                vals += [_arg(t) for t in _tensors(out)]
            gf.LOG.append({"op": "ext", "fn": "torch." + _name(op), "args": [v for v in vals if v is not None]})
            if op == "cpu":
                STATE["host"].add(_ptr(out))
        elif op not in QUIET:
            raise RuntimeError(f"torch.{op}: device work the step trace does not know")
        return out


class Staged:
    """Buffers.staged (a CUDA event): synchronize and record are logged in order."""

    def synchronize(self) -> None:
        gf.LOG.append({"op": "ext", "fn": "staged.synchronize", "args": []})

    def record(self) -> None:
        gf.LOG.append({"op": "ext", "fn": "staged.record", "args": []})


def load(mirror: Path, layers: int, rank: int, world: int, mtp: bool):
    """weights.load of the first ``layers`` layers and, with ``mtp``, the MTP head (layers.<full>. read as layers.<n>.)."""
    full = wmod.Config.read(mirror)
    read, get, prefetch = wmod.Config.read, split.RankReader.get, split.RankReader.prefetch
    a, b = f".layers.{layers}.", f".layers.{full.layers}."
    fix = (lambda n: n.replace(a, b, 1) if a in n else n)  # noqa: E731

    def cut(cls, model_dir):
        c = read(model_dir)
        c.layers, c.kinds, c.mlp_kinds = layers, c.kinds[:layers], c.mlp_kinds[:layers]
        return c

    prepare, pack_ext = exl3_experts.prepare, grouped._ext
    wmod.Config.read = classmethod(cut)
    split.RankReader.get = lambda self, name: get(self, fix(name))
    split.RankReader.prefetch = lambda self, names, device=None: prefetch(self, [fix(n) for n in names], device)
    exl3_experts.prepare = lambda *a, **k: types.SimpleNamespace(cpu_stub=True)   # CUDA-only; lane A's MoE
    grouped._ext = lambda: types.SimpleNamespace(pack=lambda *a: None)            # experts.pack at load: not traced
    try:
        w = wmod.load(mirror, rank=rank, device="cpu", mtp=mtp, world=world)
    finally:
        wmod.Config.read, split.RankReader.get, split.RankReader.prefetch = read, get, prefetch
        exl3_experts.prepare, grouped._ext = prepare, pack_ext
    hw = w.head.weight                            # a GPU's .to(dev) copies the rank's head rows; on the CPU it is a no-op
    if hw.untyped_storage().nbytes() != hw.numel() * hw.element_size():
        object.__setattr__(w.head, "weight", hw.clone())
    w.comm = gf.Comm(rank, world) if world > 1 else None
    return full, w


def objects(w, rows: int, capacity: int):
    b, mb = forward.Buffers(w, rows, capacity), forward.Buffers(w, rows, capacity)
    b.staged = mb.staged = Staged()
    st = forward.State(w, capacity, rows)
    return b, mb, st


def name_all(w, b, mb, st) -> None:
    gf.NAMES.by_ptr.clear()
    gf.NAMES.walk("w", w)
    gf.NAMES.walk("b", b)
    gf.NAMES.walk("mbuf", mb)
    gf.NAMES.walk("st", st)
    STATE["ids_host"] = {_ptr(b.ids_host), _ptr(mb.ids_host)}
    STATE["host"] = set(STATE["ids_host"])
    STATE["count"] = {}


def step_case(w, rows: int, pos: int, parity: int, keep: int, max_rows: int, capacity: int) -> dict:
    """One decode step: stage, compute, greedy sample_rows, commit of the first ``keep`` rows."""
    b, mb, st = objects(w, max_rows, capacity)
    st.set_pos(pos)
    st.cur = [parity] * len(st.cur)
    name_all(w, b, mb, st)
    gf.LOG.clear()
    tokens = [(1000 + 37 * i + pos) % w.cfg.vocab for i in range(rows)]
    STATE["tokens"] = tokens
    with Torch():
        R = forward.stage(w, st, b, tokens)
        nch = forward.chunks_for(st, R)
        logits = forward.compute(w, st, b, R, nch=nch, host_pos=st.pos)
        decode.sample_rows(w, logits[:R], [st.pos + 1 + r for r in range(R)], None)
        forward.commit(w, st, b, R, keep)
    return {"kind": "step", "rows": rows, "pos": pos, "parity": parity, "keep": keep, "nch": nch, "tokens": tokens,
            "events": list(gf.LOG)}


def lane_a_marks():
    """The MTP layer's DSA and MoE blocks are lane A's: each is a ``mark.<block>`` event whose gathered partials are
    the caller's Buffers' gath, so every launch of mtp_compute around them is python's own. Gives the undo."""
    real = mtp_mod.dsa_block, mtp_mod.moe_block

    def block(name):
        def stub(layer, w_, *a):
            b = next(x for x in a if isinstance(x, forward.Buffers))
            R = next(x for x in a if isinstance(x, int) and not isinstance(x, bool))   # the rows after b
            gf.LOG.append({"op": "ext", "fn": f"mark.{name}", "args": []})
            d = w_.cfg.hidden
            return b.gath[:w_.world * R * d].view(w_.world, R, d)
        return stub

    mtp_mod.dsa_block, mtp_mod.moe_block = block("dsa_block"), block("moe_block")
    return lambda: (setattr(mtp_mod, "dsa_block", real[0]), setattr(mtp_mod, "moe_block", real[1]))


def mtp_case(w, n: int, mtp_len: int, max_rows: int, capacity: int) -> dict:
    """mtp_stage of n next tokens with the main rows' hidden (b.fnormed[:n]), then mtp_compute, around lane A's marks."""
    b, mb, st = objects(w, max_rows, capacity)
    st.set_mtp_len(mtp_len)
    name_all(w, b, mb, st)
    gf.LOG.clear()
    tokens = [(2000 + 41 * i + mtp_len) % w.cfg.vocab for i in range(n)]
    STATE["tokens"] = tokens
    undo = lane_a_marks()
    try:
        with Torch():
            m = mtp_mod.mtp_stage(w, st, mb, tokens, b.fnormed[:n])
            nch = -(-(st.mtp_len + m) // 512)
            mtp_mod.mtp_compute(w, st, mb, m, nch=nch, host_pos=st.mtp_len)
    finally:
        undo()
    return {"kind": "mtp", "rows": n, "mtp_len": mtp_len, "nch": nch, "tokens": tokens, "events": list(gf.LOG)}


# -- prompt chunks (qualification_qualification_44ba554e.5 B3): decode.Engine's prefill, and compute's cut, on the prefill Buffers -----------------------
def engine(w, rows: int, capacity: int, prefill_rows: int):
    """decode.Engine as GlmEngine builds it for a long context, eager (no graphs), each Buffers' staged event logged."""
    e = decode.Engine(w, capacity=capacity, max_rows=rows, prefill_rows=prefill_rows, long_context=True)
    for x in (e.buf, e.pbuf, e.mbuf):
        if x is not None:
            x.staged = Staged()
    return e


def name_engine(w, e) -> None:
    gf.NAMES.by_ptr.clear()
    gf.NAMES.walk("w", w)
    for name, x in (("b", e.buf), ("pbuf", e.pbuf), ("mbuf", e.mbuf), ("st", e.st)):
        if x is not None:
            gf.NAMES.walk(name, x)
    STATE["ids_host"] = {_ptr(x.ids_host) for x in (e.buf, e.pbuf, e.mbuf) if x is not None}
    STATE["host"] = set(STATE["ids_host"])
    STATE["count"] = {}
    kda_mod._tmp.clear()                              # kda._wide_scratch's q and y, made anew in each case


def staging():
    """forward.stage and mtp_stage note the ids they write (the host.write event's values). Gives the undo."""
    s0, m0 = decode.stage, mtp_mod.mtp_stage

    def stage(w, st, b, tokens):
        STATE["tokens"] = list(tokens)
        return s0(w, st, b, tokens)

    def mstage(w, st, b, nxt, hidden):
        STATE["tokens"] = list(nxt)
        return m0(w, st, b, nxt, hidden)

    decode.stage, mtp_mod.mtp_stage = stage, mstage
    return lambda: (setattr(decode, "stage", s0), setattr(mtp_mod, "mtp_stage", m0))


def prefill_case(w, n: int, rows: int, capacity: int, prefill_rows: int) -> dict:
    """decode.prefill of an n-token prompt (chunks of prefill_rows, MTP absorb, commit), then its greedy first token."""
    e = engine(w, rows, capacity, prefill_rows)
    name_engine(w, e)
    gf.LOG.clear()
    prompt = [(3000 + 53 * i) % w.cfg.vocab for i in range(n)]
    undo, undo2 = lane_a_marks(), staging()
    try:
        with Torch():
            decode.prefill(e, prompt, None, mtp=True)
    finally:
        undo()
        undo2()
    chunks = [min(prefill_rows, n - s) for s in range(0, n, prefill_rows)]
    return {"kind": "prefill", "rows": n, "chunks": chunks, "tokens": prompt, "events": list(gf.LOG)}


def chunk_case(w, R: int, pos: int, point: int, rows: int, capacity: int, prefill_rows: int) -> dict:
    """One prompt chunk cut at row ``point`` (forward.Cut: the KDA state and conv windows there): stage, compute, commit."""
    e = engine(w, rows, capacity, prefill_rows)
    st, b = e.st, e.pbuf
    st.set_pos(pos)
    name_engine(w, e)
    gf.LOG.clear()
    tokens = [(5000 + 61 * i + pos) % w.cfg.vocab for i in range(R)]
    STATE["tokens"] = tokens
    with Torch():
        cut = forward.Cut(point, torch.empty_like(st.rec[0]), st.conv.clone())
        n = forward.stage(w, st, b, tokens)
        nch = forward.chunks_for(st, n)
        forward.compute(w, st, b, n, nch=nch, host_pos=st.pos, cut=cut)
        forward.commit(w, st, b, n, n)
    return {"kind": "chunk", "rows": R, "pos": pos, "point": point, "nch": nch, "tokens": tokens,
            "events": list(gf.LOG)}


def lexsort_fixture(out: Path) -> int:
    """decode.sample_rows' greedy host choice (np.lexsort((tokens, -values))) on drawn rows with ties, signed zeros,
    infinities and NaNs: cuda_sample.zig's `choose` must pick the same token."""
    rng = np.random.default_rng(15)
    pool = np.array([0.0, -0.0, 1.5, 1.5, -2.25, 7.0, np.inf, -np.inf, np.nan, 3.0], dtype=np.float32)
    cases = []
    for i in range(64):
        world, rows, k = int(rng.integers(1, 5)), int(rng.integers(1, 9)), (1, 1, 3)[i % 3]
        values = pool[rng.integers(0, len(pool), size=(rows, world * k))].astype(np.float32)
        if i % 5 == 0:
            values[:] = np.float32(1.5)                   # every candidate tied: the smallest id wins
        tokens = rng.integers(0, 154880, size=(rows, world * k)).astype(np.int64)
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[r, order[r, 0]]) for r in range(rows)]
        cases.append({"world": world, "rows": rows, "k": k, "chosen": chosen,
                      "values": [[int(x) for x in row.view(np.uint32)] for row in values],
                      "tokens": tokens.tolist()})
    # qualification_2b1188e0's exact top-1 tie (long teacher step 25): tokens 4663 and 83 both 7.65625 on rank 0's shard, the teacher's
    # topk(5) listing 4663 first; python's serial reply took 83, the lower id. Rank 1's candidates are drawn below it.
    tie = [(4663, 7.65625), (83, 7.65625), (151, 7.5), (9, 7.25), (12, 7.0)]
    other = [(77440 + 5, 7.4375), (77440 + 900, 7.0), (77440 + 77, 6.5), (77440 + 2, 6.25), (77440 + 40000, 6.0)]
    for name, k, cands in (("sample_tie_pair_top5", 5, tie + other), ("sample_tie_pair_top1", 1, [tie[1], (77440 + 3, 7.65625)])):
        values = np.array([[v for _, v in cands]], dtype=np.float32)
        tokens = np.array([[t for t, _ in cands]], dtype=np.int64)
        order = np.lexsort((tokens, -values), axis=-1)
        cases.append({"name": name, "world": 2, "rows": 1, "k": k, "chosen": [int(tokens[0, order[0, 0]])],
                      "values": [[int(x) for x in values[0].view(np.uint32)]], "tokens": tokens.tolist()})
        assert cases[-1]["chosen"] == [83], cases[-1]
    doc = {"generator": "tools/zig/glm_step_fixtures.py --lexsort", "schema": "glm-sample-choose/1", "python": "ed78d6f",
           "cases": cases}
    out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"{len(cases)} cases -> {out}")
    return 0


def _keys(row: np.ndarray) -> np.ndarray:
    """torch's radix keys of fp32 values (TopKTypeConfig<float>::convert), for rows with no NaN."""
    x = row.view(np.uint32).astype(np.uint64)
    return x ^ np.where(x >> np.uint64(31) != 0, np.uint64(0xFFFFFFFF), np.uint64(0x80000000))


def topk_fixture(out: Path, runs: list[str]) -> int:
    """The teacher's shard top 5 as python-0.6.6 sessions recorded them on two GB10s (torch.topk(lg[0], 5) on CUDA), for
    every step of the given runs (LABEL=DIR, qualification_6607f886's layout: rank{0,1}/teacher/{short,long}/steps.json + logits-KK.bin):
    each rank's logits row cut to its columns at or above the fifth value (the rest cannot change topk's answer), the
    order python listed, and the token its serial reply took. cuda_sample.zig's topkOrder must list python's order,
    ties included, and topk 1 a rank with choose must take the serial token."""
    cases, sources, columns = [], [], None
    for spec in runs:
        label, run = spec.split("=", 1)
        for p in ("short", "long"):
            files = [Path(run) / f"rank{r}" / "teacher" / p / "steps.json" for r in (0, 1)]
            docs = [json.loads(f.read_text()) for f in files]
            sources.append({"run": f"{label}:{Path(run).name}", "prompt": p,
                            "steps_sha256": [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]})
            for s0, s1 in zip(docs[0]["steps"], docs[1]["steps"], strict=True):
                ranks = []
                for r, s in ((0, s0), (1, s1)):
                    raw = (Path(run) / f"rank{r}" / s["logits"]["file"]).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == s["logits"]["sha256"], (label, p, r, s["k"])
                    row = np.frombuffer(raw, dtype=np.float32)
                    assert np.isfinite(row).all() and columns in (None, row.size), (label, p, r, s["k"])
                    columns = row.size
                    keys = _keys(row)
                    keep = np.nonzero(keys >= np.sort(keys)[-5])[0]
                    top = [[int(i), int(np.float32(v).view(np.uint32))] for i, v in s["shard_top5"]]
                    ranks.append({"vocab_offset": s["logits"]["vocab_offset"], "logits_sha256": s["logits"]["sha256"],
                                  "entries": [[int(c), int(row[c:c + 1].view(np.uint32)[0])] for c in keep],
                                  "top5": top})
                assert s0["next_serial"] == s1["next_serial"] and s0["k"] == s1["k"]
                cases.append({"name": f"{label}-{p}-s{s0['k']:02d}", "next_serial": s0["next_serial"], "ranks": ranks})
    doc = {"generator": "tools/zig/glm_step_fixtures.py --topk", "schema": "glm-topk-order/1", "python": "ed78d6f",
           "recorded": "torch.topk(logits[:1].float()[0], 5) on CUDA (sm_121), python-0.6.6 sessions",
           "columns": columns, "sources": sources, "cases": cases}
    out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    ties = sum(1 for c in cases for r in c["ranks"] if any(a[1] == b[1] for a, b in zip(r["top5"], r["top5"][1:])))
    print(f"{len(cases)} cases, {2 * len(cases)} rows, {ties} with a tie in the top 5 -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", type=Path, help="the checkpoint (a header-only sparse mirror serves)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=3, help="backbone layers loaded (layers 3+ are lane A's blocks)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--no-mtp", action="store_true", help="load without the MTP head (no MTP cases)")
    ap.add_argument("--lexsort", action="store_true", help="the greedy host choice's cases, not the trace")
    ap.add_argument("--topk", nargs="+", metavar="LABEL=RUN", help="the teacher's top-5 order as these recorded runs "
                    "hold it (python-0.6.6 on CUDA), not the trace")
    ap.add_argument("--prompt", action="store_true", help="prompt chunks (decode.prefill, a cut chunk), not steps")
    a = ap.parse_args()
    if a.lexsort:
        return lexsort_fixture(a.out)
    if a.topk:
        return topk_fixture(a.out, a.topk)
    if a.mirror is None:
        ap.error("--mirror is required for the trace")
    gf.patch()
    mtp = not a.no_mtp
    full, w = load(a.mirror, a.layers, a.rank, a.world, mtp)
    rows, capacity, prefill_rows = 8, 2560, decode.PREFILL_ROWS
    cases = {}
    if a.prompt:                                      # a 64-row chunk; a 2,048-row chunk and a 37-row tail; two cuts
        for n in (64, 2085):
            cases[f"prefill-{n}"] = prefill_case(w, n, rows, capacity, prefill_rows)
        for R, pos, point in ((100, 0, 40), (2048, 0, 1000)):
            cases[f"chunk-r{R}-p{pos}-cut{point}"] = chunk_case(w, R, pos, point, rows, capacity, prefill_rows)
        e = engine(w, rows, capacity, prefill_rows)
        name_engine(w, e)
    else:
        for R, pos, parity, keep in ((1, 0, 0, 1), (1, 37, 1, 1), (4, 0, 0, 4), (4, 0, 0, 1), (4, 37, 1, 2), (4, 37, 1, 4)):
            cases[f"step-r{R}-p{pos}-k{keep}"] = step_case(w, R, pos, parity, keep, rows, capacity)
        if mtp:
            for n, mtp_len in ((1, 0), (4, 37)):
                cases[f"mtp-n{n}-m{mtp_len}"] = mtp_case(w, n, mtp_len, rows, capacity)
        b, mb, st = objects(w, rows, capacity)
        name_all(w, b, mb, st)
    out = {"generator": "tools/zig/glm_step_fixtures.py", "schema": "glm-forward-trace/1", "python": "ed78d6f",
           "rank": a.rank, "world": a.world, "layers": a.layers, "mtp": mtp, "max_rows": rows, "capacity": capacity,
           "latent": forward.latent.ENABLED, "config_json": (a.mirror / "config.json").read_text(),
           "config": gf.config_dict(full), "vocab_offset": w.vocab_offset, "buffers": gf.NAMES.buffers(),
           "cases": cases}
    if a.prompt:
        out.update(prefill_rows=prefill_rows, long_context=True)
    a.out.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    n = sum(len(c["events"]) for c in cases.values())
    print(f"{len(cases)} cases, {n} events, {len(out['buffers'])} buffers -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
