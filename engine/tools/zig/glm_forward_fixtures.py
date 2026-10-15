#!/usr/bin/env python3
"""GLM-5.3-Flash's forward as python-0.6's glm5_next/cuda runs it, recorded on a CPU: each launch in order, nothing run."""
# qualification_qualification_44ba554e.5's launch trace (evidence/qualification_qualification_7c75d777/PLAN.md §4): python-0.6's own weights.load (rank of 2, CPU) over a checkpoint's
# header-only sparse mirror (real shapes, zero bytes; MLX 4-bit -> fixtures_cuda_forward.json, EXL3 -> *_exl3.json),
# Buffers and State, then layer_forward with every Triton kernel, extension call (kernels/qmm.py, kda.py,
# cuda/experts.py) and all_gather replaced by recorders. Each tensor argument is its buffer's python path (w..., b...,
# st...) and byte offset. The Triton recorder follows ZK4's tools/zig/glm_triton_fixtures.py, itself adapted from Mia's
# (MiaAI-Lab) flashnext_triton_fixtures.py. MoE cases (qualification_qualification_736508ad): layer 3's MLP half and, from a second load with MTP on
# (qualification_6607f886's rule: the cut copy's layers.<n>. read from the checkpoint's own MTP layer), the MTP layer's moe_block.

from __future__ import annotations

import argparse
import gc
import json
import struct
import sys
import types
from pathlib import Path

import torch
from triton.runtime.jit import JITFunction

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3 import experts as generic
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.glm5_next.cuda import attention, forward, glue, latent, qmm, sparse, split
from tensorfold.families.glm5_next.cuda import kda as kda_mod
from tensorfold.families.glm5_next.cuda import weights as wmod

MODULES = (attention, forward, glue, latent, qmm, sparse)
TYPES = {torch.bfloat16: "*bf16", torch.float32: "*fp32", torch.float16: "*fp16", torch.int32: "*i32",
         torch.int64: "*i64", torch.uint8: "*u8", torch.int8: "*i8", torch.int16: "*i16"}
DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32", torch.float16: "f16", torch.int32: "i32",
          torch.int64: "i64", torch.uint8: "u8", torch.int8: "i8", torch.int16: "i16"}
LOG: list[dict] = []


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


class Names:
    """Every tensor reachable from the engine's objects by its attribute path; a view resolves to its storage's owner."""

    TMP_OK = {("_router_part", "PART"), ("_router_sum", "PART")}  # glue.router's K-slice partials: torch.empty a call
    TMP_OK |= {("_scores", "OUT"), ("_select_rows", "S"), ("_select_rows", "OUT"), ("_merge", "CNT"),  # DSA's
               ("_sparse_chunks", "TOK"), ("_sparse_chunks", "CNT"), ("_sparse_chunks", "PO"), ("_sparse_chunks", "PM"),
               ("_sparse_chunks", "PL"), ("_merge", "PO"), ("_merge", "PM"), ("_merge", "PL"), ("select_tail", "pools"),
               ("select_tail", "tokens"), ("select_tail", "counts")}   # select_tokens' and sparse_attention's

    def __init__(self) -> None:
        self.by_ptr: dict[int, tuple[str, torch.Tensor]] = {}
        self.tmp: dict[int, tuple[str, dict]] = {}       # a case's temporaries by storage: tmp.<k> in order of first use

    def walk(self, path: str, obj, seen: set | None = None) -> None:
        seen = set() if seen is None else seen
        if isinstance(obj, torch.Tensor):
            self._add(path, obj)
            return
        if obj is None or isinstance(obj, (int, float, bool, str, bytes, torch.device, torch.dtype)) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj):
                self.walk(f"{path}.{i}", x, seen)
        elif isinstance(obj, dict):
            for k, x in obj.items():
                if not str(k).startswith("_"):
                    self.walk(f"{path}.{k}", x, seen)
        elif hasattr(obj, "__dict__") and not isinstance(obj, (types.ModuleType, types.FunctionType)):
            for k, x in vars(obj).items():
                if not k.startswith("_") and k not in ("cfg", "meta", "comm", "staged"):
                    self.walk(f"{path}.{k}", x, seen)

    def _add(self, path: str, t: torch.Tensor) -> None:
        s = t.untyped_storage()
        key = s.data_ptr()
        owner = t.data_ptr() == key and t.numel() * t.element_size() == s.nbytes()
        old = self.by_ptr.get(key)
        if old is None or (owner and not self._owns(old[1])):
            self.by_ptr[key] = (path, t)

    @staticmethod
    def _owns(t: torch.Tensor) -> bool:
        s = t.untyped_storage()
        return t.data_ptr() == s.data_ptr() and t.numel() * t.element_size() == s.nbytes()

    def where(self, t: torch.Tensor, site: tuple | None = None) -> tuple[str, int]:
        key = t.untyped_storage().data_ptr()
        if key not in self.by_ptr and site in self.TMP_OK:
            if key not in self.tmp:
                info = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "nbytes": t.untyped_storage().nbytes()}
                self.tmp[key] = (f"tmp.{len(self.tmp)}", info)
            return self.tmp[key][0], t.data_ptr() - key
        if key not in self.by_ptr:
            raise KeyError(f"a tensor of no known buffer: {tuple(t.shape)} {t.dtype} (a temporary python made?)")
        return self.by_ptr[key][0], t.data_ptr() - key

    def tensor(self, t: torch.Tensor) -> dict:
        buf, off = self.where(t)
        return {"buf": buf, "off": off, "dtype": DTYPES[t.dtype], "shape": list(t.shape), "stride": list(t.stride())}

    def buffers(self) -> dict:
        out = {}
        for path, t in sorted(self.by_ptr.values()):
            s = t.untyped_storage()
            out[path] = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "nbytes": s.nbytes(), "owner": self._owns(t)}
        return out


NAMES = Names()


def val(v):
    """An extension call's argument: a tensor by buffer and offset, lists element by element, scalars by value."""
    if isinstance(v, torch.Tensor):
        return {"t": NAMES.tensor(v)}
    if isinstance(v, (list, tuple)):
        return {"list": [val(x) for x in v]}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": f32_bits(v)}
    if v is None:
        return {"none": True}
    raise TypeError(f"argument {type(v)}")


def const(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": f32_bits(v)}
    raise TypeError(f"constexpr {v!r}")


class Triton:
    """Stands in for a JITFunction: ``kernel[grid](...)`` appends the launch with each pointer's buffer and offset."""

    def __init__(self, jit: JITFunction) -> None:
        self.jit = jit

    def __getitem__(self, grid):
        def call(*args, **kw):
            names = self.jit.arg_names
            bound = dict(zip(names, args))
            bound.update({k: v for k, v in kw.items() if k in names})
            runtime, consts = [], {}
            for p in self.jit.params:
                if p.name not in bound:
                    if p.is_constexpr and p.has_default:
                        consts[p.name] = const(p.default)
                    continue
                v = bound[p.name]
                if p.is_constexpr or v is None:
                    consts[p.name] = const(v)
                elif isinstance(v, torch.Tensor):
                    buf, off = NAMES.where(v, (self.jit.fn.__name__, p.name))
                    runtime.append({"name": p.name, "type": TYPES[v.dtype], "buf": buf, "off": off})
                elif isinstance(v, bool):
                    raise TypeError(f"{p.name}: a runtime bool")
                elif isinstance(v, int):
                    runtime.append({"name": p.name, "type": "i32", "int": v})
                elif isinstance(v, float):
                    runtime.append({"name": p.name, "type": "fp32", "f32": f32_bits(v)})
                else:
                    raise TypeError(f"{p.name}: {type(v)}")
            g = [int(x) for x in grid] + [1] * (3 - len(grid))
            LOG.append({"op": "triton", "fn": self.jit.fn.__name__, "grid": g, "args": runtime, "consts": consts,
                        "num_warps": kw.get("num_warps"), "num_stages": kw.get("num_stages")})

        return call


class Ext:
    """Stands in for a compiled extension module: each call is logged as ``<module>.<op>`` with its arguments."""

    def __init__(self, module: str) -> None:
        self.module = module

    def __getattr__(self, op: str):
        def call(*args):
            LOG.append({"op": "ext", "fn": f"{self.module}.{op}", "args": [val(a) for a in args]})
        return call


class Comm:
    """Rank `rank` of `world`: all_gather is logged, never run."""

    def __init__(self, rank: int, world: int) -> None:
        self.rank, self.world = rank, world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        LOG.append({"op": "comm", "fn": "all_gather", "args": [val(send), val(recv)]})


def patch() -> None:
    for mod in MODULES:
        for name, v in list(vars(mod).items()):
            if isinstance(v, JITFunction):
                setattr(mod, name, Triton(v))
    shared._ext = lambda: Ext("qmm")
    kda_mod._ext = lambda: Ext("kda")
    grouped._ext = lambda: Ext("experts")
    generic._ext = lambda: Ext("exl3_experts")           # EXL3's routed experts (exl3_generic.routed)
    shared.grouped = lambda device: True                 # sm_12x: groups of 64 through qmm_group
    shared._chip = lambda device: (12, 1, 48)            # a GB10: group_tile 0 at every row count
    event = types.SimpleNamespace(synchronize=lambda: None)
    torch.cuda.current_stream = lambda *a, **k: types.SimpleNamespace(record_event=lambda: event)
    torch.cuda.empty_cache = lambda: None


class OnCuda:
    """A trellis as experts.prepare checks it (contiguous CUDA int16; qualification_765330ce's exl3_check.py stand-in); ``_t`` is unwalked."""

    def __init__(self, t: torch.Tensor) -> None:
        self._t = t

    dtype = property(lambda s: s._t.dtype)
    shape = property(lambda s: s._t.shape)
    device = property(lambda s: types.SimpleNamespace(type="cuda"))

    def is_contiguous(self) -> bool:
        return self._t.is_contiguous()

    def data_ptr(self) -> int:
        return self._t.data_ptr()


PREPARE = generic.prepare


def prepare_cpu(gate, up, down, codebook, device="cuda"):
    wrap = (lambda mats: [(OnCuda(t), s, v) for t, s, v in mats])  # noqa: E731
    return PREPARE(wrap(gate), wrap(up), wrap(down), codebook, device=device)


def config_dict(c) -> dict:
    out = {}
    for k, v in vars(c).items():
        out[k] = f32_bits(v) if isinstance(v, float) else (list(v) if isinstance(v, tuple) else v)
    out["dense_limit"] = c.dense_limit
    return out


def load(mirror: Path, layers: int, rank: int, world: int, mtp: bool = False):
    """python-0.6's weights.load of the first ``layers`` layers (Config.read cut there) on the CPU; with ``mtp`` the
    cut copy's MTP layer (layers.<layers>.) is read from the checkpoint's own (layers.<num_hidden_layers>.), as qualification_6607f886's
    oracle reads it. experts.pack (a CUDA kernel) leaves shapes only; EXL3 trellises pass prepare's device check."""
    full = wmod.Config.read(mirror)
    read = wmod.Config.read
    get, prefetch = split.RankReader.get, split.RankReader.prefetch

    def cut(cls, model_dir):
        c = read(model_dir)
        c.layers, c.kinds, c.mlp_kinds = layers, c.kinds[:layers], c.mlp_kinds[:layers]
        return c

    wmod.Config.read = classmethod(cut)
    grouped._ext = lambda: types.SimpleNamespace(pack=lambda *a: None)
    generic.prepare = prepare_cpu
    if mtp:
        a, b = f".layers.{layers}.", f".layers.{full.layers}."
        fix = (lambda n: n.replace(a, b, 1))  # noqa: E731
        split.RankReader.get = lambda self, name: get(self, fix(name))
        split.RankReader.prefetch = lambda self, names, device=None: prefetch(self, [fix(n) for n in names], device)
    try:
        w = wmod.load(mirror, rank=rank, device="cpu", mtp=mtp, world=world)
    finally:
        wmod.Config.read = read
        grouped._ext = lambda: Ext("experts")
        generic.prepare = PREPARE
        split.RankReader.get, split.RankReader.prefetch = get, prefetch
    hw = w.head.weight                            # a GPU's .to(dev) copies the rank's head rows; on the CPU it is a no-op
    if hw.untyped_storage().nbytes() != hw.numel() * hw.element_size():
        object.__setattr__(w.head, "weight", hw.clone())
    w.comm = Comm(rank, world) if world > 1 else None
    return full, w


def window(w, pos: int, parity: int, max_rows: int, capacity: int):
    """A decode window's Buffers and State, every tensor named (w..., b..., st...), and an empty log."""
    b = forward.Buffers(w, max_rows, capacity)
    st = forward.State(w, capacity, max_rows)
    st.set_pos(pos)
    st.cur = [parity] * len(st.cur)
    NAMES.by_ptr.clear()
    NAMES.tmp.clear()
    NAMES.walk("w", w)
    NAMES.walk("b", b)
    NAMES.walk("st", st)
    LOG.clear()
    return b, st


def done(layer, rows: int, pos: int, parity: int, nch: int, max_rows: int, capacity: int, **extra) -> dict:
    out = {"layer": layer, "rows": rows, "pos": pos, "parity": parity, "nch": nch, "max_rows": max_rows,
           "capacity": capacity, "events": list(LOG), **extra}
    if NAMES.tmp:
        out["tmp"] = dict(NAMES.tmp.values())
    return out


def run_case(w, layer: int, rows: int, pos: int, parity: int, max_rows: int, capacity: int) -> dict:
    b, st = window(w, pos, parity, max_rows, capacity)
    nch = forward.chunks_for(st, rows)
    forward.layer_forward(w.layers[layer], w, st, b, rows, nch, host_pos=pos)
    return done(layer, rows, pos, parity, nch, max_rows, capacity)


def run_mlp_half(w, layer: int, rows: int, max_rows: int, capacity: int) -> dict:
    """layer_forward's MLP half (forward.py 437-442): hc_pre with ffn_hc and post_norm, the MoE (or MLP), hc_post."""
    b, st = window(w, 0, 0, max_rows, capacity)
    c, lw, x = w.cfg, w.layers[layer], b.x[:rows]
    h = lw.ffn_hc
    glue.hc_pre(x, h.fn, h.base, h.scale, lw.post_norm, b.normed[:rows], b.xs[:rows], b.post[:rows], b.comb[:rows],
                b.hcpart[:rows], c.eps, c.hc_eps, c.hc_iters)
    g = forward.mlp_block(lw, w, b, rows) if lw.mlp is not None else forward.moe_block(lw, w, b, rows)
    glue.hc_post(x, x, g, b.post[:rows], b.comb[:rows])
    return done(layer, rows, 0, 0, forward.chunks_for(st, rows), max_rows, capacity, block="mlp")


def run_mtp_moe(w, rows: int, max_rows: int, capacity: int) -> dict:
    """mtp.py's MoE step (mtp.py 47-49 between its norm and residual_add): forward.moe_block on w.mtp.layer."""
    b, st = window(w, 0, 0, max_rows, capacity)
    forward.moe_block(w.mtp.layer, w, b, rows)
    return done("mtp", rows, 0, 0, forward.chunks_for(st, rows), max_rows, capacity, block="moe")


# -- G-layout (qualification_qualification_7c75d777 PLAN §4): python-0.6's load-time packing of drawn or real tensors, by sha256 ------------------------
MASK = (1 << 64) - 1
PREFIX = "model.language_model."


def draws(seed: int):
    """splitmix64 from ``seed`` (glm/cuda_pack_ref.zig's Draw): a word is a draw's high half, a 16-bit value its top quarter."""
    s = seed & MASK
    while True:
        s = (s + 0x9E3779B97F4A7C15) & MASK
        z = ((s ^ (s >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        yield z ^ (z >> 31)


def drawn(g, shape: tuple, half: bool) -> torch.Tensor:
    import numpy as np

    n = 1
    for x in shape:
        n *= x
    if half:
        return torch.from_numpy(np.array([next(g) >> 48 for _ in range(n)], dtype=np.uint16).view(np.int16)).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(np.array([next(g) >> 32 for _ in range(n)], dtype=np.uint32).view(np.int32)).reshape(shape)


def finite(t: torch.Tensor) -> torch.Tensor:
    """Drawn bf16 made finite: sign and mantissa kept, the exponent folded into 2^-15 .. 2^0 (a weight's range)."""
    import numpy as np

    h = t.contiguous().view(torch.int16).numpy().view(np.uint16)
    h = (h & 0x807F) | ((0x70 + ((h >> 7) & 0xF)) << 7)
    return torch.from_numpy(h.astype(np.uint16).view(np.int16)).view(torch.bfloat16).reshape(t.shape)


def digest(t: torch.Tensor) -> dict:
    import hashlib

    t = t.contiguous()
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "sha256": hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def q4_out(prefix: str, q) -> dict:
    return {prefix + "weight": digest(q.weight), prefix + "scales": digest(q.scales), prefix + "biases": digest(q.biases)}


def pack_case(c: dict, parts: list) -> dict:
    """weights.py load's packing of one case's inputs: ``parts`` are (words, scales, biases) triplets or plain tensors."""
    op = c["op"]
    if op == "q4":
        return q4_out("", qmm.make_q4(*parts[0]))
    if op == "stack":
        out = {}
        for i, (lo, hi) in enumerate(c["ranges"]):
            out.update(q4_out(f"r{i}.", qmm.stack_q4(parts[lo:hi])))
        return out
    if op == "kv_b":                              # weights.py dsa(): the rank's rows by head, key rows then value rows
        w, s, b = parts[0]
        rows = torch.arange(c["heads"] * (c["qk"] + c["v"])).view(c["heads"], c["qk"] + c["v"])
        krows, vrows = rows[:, :c["qk"]].reshape(-1), rows[:, c["qk"]:].reshape(-1)
        out = q4_out("kv_k.", qmm.make_q4(w[krows], s[krows], b[krows]))
        out.update(q4_out("kv_v.", qmm.make_q4(w[vrows], s[vrows], b[vrows])))
        a = latent.AbsorbQ4((w[krows], s[krows], b[krows]), (w[vrows], s[vrows], b[vrows]), c["heads"])
        out.update({"absorb." + k: digest(getattr(a, k)) for k in ("wkw", "wks", "wkb", "wvw", "wvs", "wvb")})
        return out
    if op == "conv":                              # weights.py kda(): q, k, v's [W, 1, taps] -> [3 W, taps]
        return {"conv": digest(torch.cat(parts).reshape(3 * c["w"], c["taps"]).contiguous())}
    if op == "f32":                               # t(name, torch.float32): A_log, dt_bias, hc base and scale, router bias
        return {"f32": digest(parts[0].to(torch.float32).contiguous())}
    if op == "quant4":                            # weights.py exl3: draft_head = quantize4(head.weight)
        d = parts[0].float().view(-1, c["k"] // 64, 64)
        d = d.amax(-1) - d.amin(-1)               # CUDA divides by a CPU scalar as a product with its fp32 reciprocal
        if not torch.equal((d / 15).clamp_min(1e-8).to(torch.bfloat16),
                           (d * torch.tensor(1 / 15, dtype=torch.float32)).clamp_min(1e-8).to(torch.bfloat16)):
            raise ValueError(f"{c['name']}: a scale differs between the CPU's division and CUDA's product: pick a seed")
        return q4_out("", qmm.quantize4(parts[0]))
    if op == "cat":                               # weights.py exl3: stack_b16 (torch.cat) of a layer's BF16 parts
        return {"weight": digest(qmm.stack_b16(parts).weight)}
    raise ValueError(op)


DRAWN = [
    {"name": "q4-256x128", "op": "q4", "seed": 1, "n": 256, "k": 128},
    {"name": "q4-200x192-pad", "op": "q4", "seed": 2, "n": 200, "k": 192},
    {"name": "q4-64x64-pad", "op": "q4", "seed": 3, "n": 64, "k": 64},
    {"name": "q4-130x256-pad", "op": "q4", "seed": 4, "n": 130, "k": 256},
    {"name": "stack-4-parts", "op": "stack", "seed": 5, "ns": [64, 32, 8, 24], "k": 256, "ranges": [[0, 4], [0, 2], [2, 4]]},
    {"name": "kv_b-2-heads", "op": "kv_b", "seed": 6, "heads": 2, "qk": 256, "v": 256, "k": 512},
    {"name": "conv-256x4", "op": "conv", "seed": 7, "w": 256, "taps": 4},
    {"name": "f32-24", "op": "f32", "seed": 8, "n": 24},
    {"name": "quant4-200x192-pad", "op": "quant4", "seed": 9, "n": 200, "k": 192, "flat": 3},
    {"name": "quant4-128x256", "op": "quant4", "seed": 10, "n": 128, "k": 256},
    {"name": "cat-6-parts", "op": "cat", "seed": 11, "ns": [64, 64, 64, 16, 16, 8], "k": 128},
]


def drawn_parts(c: dict) -> list:
    """The case's inputs from its seed, in glm_pack_host.zig's order: part by part, its words, scales, then biases."""
    g = draws(c["seed"])
    if c["op"] in ("q4", "stack", "kv_b"):
        ns = c["ns"] if c["op"] == "stack" else [c["heads"] * (c["qk"] + c["v"]) if c["op"] == "kv_b" else c["n"]]
        k = c["k"]
        return [(drawn(g, (n, k // 8), False), drawn(g, (n, k // 64), True), drawn(g, (n, k // 64), True)) for n in ns]
    if c["op"] == "conv":
        return [drawn(g, (c["w"], 1, c["taps"]), True) for _ in range(3)]
    if c["op"] == "quant4":                       # finite rows (n not a multiple of 128: make_q4's padding)
        w = finite(drawn(g, (c["n"], c["k"]), True))
        if c.get("flat"):                         # every flat-th row's first group one value: quantize4's 1e-8 floor
            w[::c["flat"], :64] = w[::c["flat"], :1]
        return [w]
    if c["op"] == "cat":
        return [drawn(g, (n, c["k"]), True) for n in c["ns"]]
    return [drawn(g, (c["n"],), False).view(torch.float32)]


def real_parts(c: dict, files: dict) -> list:
    """The case's ``src`` rows ([name, first row, rows, component]) read from the checkpoint as safetensors stores them."""
    from safetensors import safe_open

    got: list[list] = []
    for name, first, rows, comp in c["src"]:
        with safe_open(files[name], framework="pt") as f:
            t = f.get_slice(name)[first:first + rows]
        t = t.view(torch.int32) if t.dtype == torch.uint32 else t
        if comp == 0:
            got.append([t])
        else:
            got[-1].append(t)
    if c["op"] in ("q4", "stack", "kv_b"):
        return [tuple(p) for p in got] if c["op"] != "kv_b" else [tuple(torch.cat(x) for x in zip(*got))]
    if c["op"] == "conv":
        return [p[0] for p in got]
    return [got[0][0]]


def real_cases(cfg_dir: Path) -> list:
    """Rank 0 of 2's slices of layers 0 (KDA) and 3 (DSA) of d2b17f52, each with the weights.json paths it equals."""
    L0, L3 = PREFIX + "layers.0.self_attn.", PREFIX + "layers.3.self_attn."

    def trip(name: str, first: int, rows: int) -> list:
        return [[name + ".weight", first, rows, 0], [name + ".scales", first, rows, 1], [name + ".biases", first, rows, 2]]

    return [
        {"name": "real-l0-fb", "op": "q4", "n": 4096, "k": 128, "src": trip(L0 + "f_b_proj", 0, 4096), "paths": "w.layers[0].kda.fb."},
        {"name": "real-l0-gb", "op": "q4", "n": 4096, "k": 128, "src": trip(L0 + "g_b_proj", 0, 4096), "paths": "w.layers[0].kda.gb."},
        {"name": "real-l0-conv", "op": "conv", "w": 4096, "taps": 4,
         "src": [[L0 + f"{x}_conv1d.weight", 0, 4096, 0] for x in "qkv"], "paths": "w.layers[0].kda."},
        {"name": "real-l0-a_log", "op": "f32", "n": 32, "src": [[L0 + "A_log", 0, 32, 0]], "paths": "w.layers[0].kda.a_log"},
        {"name": "real-l0-dt_bias", "op": "f32", "n": 4096, "src": [[L0 + "dt_bias", 0, 4096, 0]], "paths": "w.layers[0].kda.dt_bias"},
        {"name": "real-l3-kv_b", "op": "kv_b", "heads": 32, "qk": 256, "v": 256, "k": 512,
         "src": trip(L3 + "kv_b_proj", 0, 16384), "paths": "w.layers[3].dsa."},
        {"name": "real-l3-proj-index", "op": "stack", "ns": [1536, 512, 128, 32], "k": 4096, "ranges": [[0, 2], [2, 4]],
         "src": trip(L3 + "q_a_proj", 0, 1536) + trip(L3 + "kv_a_proj_with_mqa", 0, 512) + trip(L3 + "indexer.wk", 0, 128)
         + trip(L3 + "indexer.weights_proj", 0, 32), "paths": "w.layers[3].dsa.proj. w.layers[3].dsa.index.kw."},
        {"name": "real-l3-q_b", "op": "q4", "n": 8192, "k": 1536, "src": trip(L3 + "q_b_proj", 0, 8192), "paths": "w.layers[3].dsa.q_b."},
        {"name": "real-l3-index-qb", "op": "q4", "n": 4096, "k": 1536, "src": trip(L3 + "indexer.wq_b", 0, 4096), "paths": "w.layers[3].dsa.index.qb."},
    ]


def pack_fixture(out: Path, real: Path | None) -> int:
    """fixtures_cuda_pack.json (drawn inputs), or with ``real`` the real slices' cases of that checkpoint folder."""
    if real is None:
        cases = [dict(c, out=pack_case(c, drawn_parts(c))) for c in DRAWN]
        doc = {"generator": "tools/zig/glm_forward_fixtures.py --pack", "schema": "glm-pack-layout/1", "python": "ed78d6f",
               "draw": "splitmix64", "cases": cases}
    else:
        files = {k: str(real / v) for k, v in json.loads((real / "model.safetensors.index.json").read_text())["weight_map"].items()}
        cases = [dict(c, out=pack_case(c, real_parts(c, files))) for c in real_cases(real)]
        doc = {"generator": "tools/zig/glm_forward_fixtures.py --pack-real", "schema": "glm-pack-layout/1",
               "python": "ed78d6f", "dir": str(real), "cases": cases}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"{len(cases)} cases, {sum(len(c['out']) for c in cases)} outputs -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", type=Path, help="the checkpoint (a header-only sparse mirror serves)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=3, help="layers loaded (python-0.6's Config cut to the first n)")
    ap.add_argument("--mtp", action="store_true", help="also the MTP layer's moe_block, from a second load with MTP on")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--pack", action="store_true", help="G-layout: python's packing of drawn tensors, not the trace")
    ap.add_argument("--pack-real", type=Path, help="G-layout: python's packing of this checkpoint's real slices")
    ap.add_argument("--dsa", action="store_true", help="also layer 3's attention and the MTP layer's dsa_block (qualification_qualification_973313f6)")
    ap.add_argument("--select", action="store_true", help="select_tokens' tail on drawn pools, not the trace")
    ap.add_argument("--dsa-prompt", action="store_true", help="A4: the DSA arms on prompt chunks past 512 rows alone")
    a = ap.parse_args()
    if a.pack or a.pack_real:
        return pack_fixture(a.out, a.pack_real)
    if a.dsa or a.select or a.dsa_prompt:         # glm_dsa_fixtures.py over this module's recorders (run as __main__)
        import glm_dsa_fixtures as dsa
        dsa.gf = sys.modules[__name__]
    if a.select:
        return dsa.tail_fixture(a.out)
    if a.mirror is None:
        ap.error("--mirror is required for the trace")
    patch()
    if a.dsa_prompt:                              # fixtures_cuda_dsa*.json (qualification_qualification_ea348a94)
        return dsa.prompt_fixture(a)
    full, w = load(a.mirror, a.layers, a.rank, a.world)
    cases = {}
    for layer in range(a.layers):
        if w.layers[layer].kind != "kda":         # DSA attention is lane A3's: its layers' MLP halves only
            continue
        for rows in (1, 2, 4, 8):
            for pos, parity in ((0, 0), (37, 1)):
                if layer and (rows, pos) not in ((1, 0), (4, 37)):
                    continue
                cases[f"l{layer}-r{rows}-p{pos}"] = run_case(w, layer, rows, pos, parity, 8, 2560)
    for layer in range(a.layers):
        if w.layers[layer].moe is not None:
            for rows in (1, 2, 4, 8):
                cases[f"l{layer}-mlp-r{rows}"] = run_mlp_half(w, layer, rows, 8, 2560)
    long_state = None
    if a.dsa:                                     # lane A3: the DSA layers' attention (glm_dsa_fixtures.py)
        got, long_state = dsa.cases(w)
        cases.update(got)
    b = forward.Buffers(w, 8, 2560)
    st = forward.State(w, 2560, 8)
    NAMES.by_ptr.clear()
    NAMES.walk("w", w)
    NAMES.walk("b", b)
    NAMES.walk("st", st)
    out = {"generator": "tools/zig/glm_forward_fixtures.py", "schema": "glm-forward-trace/1", "python": "ed78d6f",
           "rank": a.rank, "world": a.world, "layers": a.layers, "max_rows": 8, "capacity": 2560,
           "latent": latent.ENABLED, "config_json": (a.mirror / "config.json").read_text(),
           "config": config_dict(full), "buffers": NAMES.buffers(), "cases": cases}
    if long_state is not None:
        out["long_state"] = long_state            # the long-context State (capacity 6,144) the DSA cases ran on
    if a.mtp:                                     # the MTP layer's buffers: their own list beside the main load's
        del w, b, st
        NAMES.by_ptr.clear()                      # the names hold the first load's tensors
        gc.collect()
        _, w = load(a.mirror, a.layers, a.rank, a.world, mtp=True)
        for rows in (1, 2, 4, 8):
            cases[f"mtp-moe-r{rows}"] = run_mtp_moe(w, rows, 8, 2560)
        keep = ("w.mtp.layer.moe.",)
        if a.dsa:                                 # the MTP layer's DSA arm, and the States of the MTP load
            got, out["mtp_state"] = dsa.mtp_cases(w)
            cases.update(got)
            keep += ("w.mtp.layer.dsa.",)
        out["mtp_buffers"] = {k: v for k, v in NAMES.buffers().items() if k.startswith(keep)}
    a.out.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    n = sum(len(c["events"]) for c in cases.values())
    print(f"{len(cases)} cases, {n} events, {len(out['buffers'])} buffers -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
