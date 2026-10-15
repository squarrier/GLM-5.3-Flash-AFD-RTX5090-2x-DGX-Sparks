#!/usr/bin/env python3
"""qualification_qualification_5322571c: Mia's q4 dense triplets for one TP rank of the EXL3 GLM-5.3-Flash checkpoint (078455ff), as the Zig engine's
dense-q4 form reads them (TF_GLM_DENSE=q4:DIR).

Mia's lane (recipe patch 0005-glm-dense-q4 on 0002, TF_GLM_DENSE=q4) quantizes at load: weights.py's q4(name) is
make_dense_q4(t(name + ".weight")) and stack(names) is stack_dense_q4([...]), both quantize4_mse in 4,096-row chunks.
This tool runs that code: the three functions are read from the patch file (its sha256 checked) and executed as they
are. Only make_q4 (the tiling, a CUDA extension) is replaced by a capture of its inputs, the MLX affine 4-bit triplet:
words int32 [n, k/8], scales and biases bf16 [n, k/64]. make_f8 (the fallback for k % 64 != 0) is captured too and the
tensor is reported, not written. The parts are the rank's, cut by python split.py's rule (copied below), stacked in
weights.py's order. kv_b_proj and lm_head are not made: the lane holds them FP8.

usage: glm_q4_dense.py make CKPT OUT --rank R [--world 2] [--layers N] [--only-layers I,J] [--no-mtp]
                            [--device cpu|cuda] [--headers-only] [--ties] [--source-id ID] [--partial]
  OUT gets q4-L000.safetensors ... (one file a layer, q4-mtp.safetensors for the MTP layer) and manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "0005-glm-dense-q4.patch")
PATCH_SHA256 = "da13106efa1e2d4ef77e289f60ceba1bcbe1139cabec1595cac1eb6fa73bf91f"  # evidence/G5/capture/site/patches (v1.10)
PREFIX = "model.language_model."
FORMAT = "glm-q4-dense/1"

# python-0.6 split.py (ed78d6f / v0.6.x): the rules of the tensors this tool reads, copied as they are
ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)


def rule(name: str) -> str:
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def lane_functions(patch: str = PATCH) -> tuple[dict, str]:
    """quantize4_mse, make_dense_q4 and stack_dense_q4 as the patch adds them to qmm.py, executed unmodified; make_q4
    and make_f8 capture their arguments. Returns the namespace and the sha256 of the functions' source."""

    raw = open(patch, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != PATCH_SHA256:
        raise SystemExit(f"{patch}: sha256 {got}, not v1.10's {PATCH_SHA256}: refused")
    lines, keep = raw.decode().splitlines(), []
    on = False
    for ln in lines:
        if ln.startswith("+def quantize4_mse("):
            on = True
        if on:
            if not ln.startswith("+"):
                break
            keep.append(ln[1:])
    src = "\n".join(keep).rstrip() + "\n"
    if "def stack_dense_q4(" not in src or "def make_dense_q4(" not in src:
        raise SystemExit(f"{patch}: the q4 functions were not found")
    import torch

    class Q4:  # the annotation's name; make_q4 below returns the triplet itself
        pass

    ns = {"torch": torch, "Q4": Q4, "F8": Q4,
          "make_q4": lambda words, scales, biases: ("q4", words, scales, biases),
          "make_f8": lambda weight, *a, **k: ("f8", tuple(weight.shape))}
    exec(compile(src, f"{os.path.basename(patch)}:qmm.py", "exec"), ns)
    return ns, hashlib.sha256(src.encode()).hexdigest()


class Checkpoint:
    """A safetensors checkpoint folder: the index, each shard's header, and a rank's part of a tensor."""

    def __init__(self, d: str):
        self.dir = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.headers: dict[str, tuple[dict, int]] = {}
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.layers = int((cfg.get("text_config") or cfg).get("num_hidden_layers", cfg.get("num_hidden_layers")))

    def has(self, name: str) -> bool:
        return name in self.map

    def info(self, name: str) -> tuple[str, dict, int]:
        f = self.map[name]
        if f not in self.headers:
            with open(os.path.join(self.dir, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (json.loads(fh.read(n)), 8 + n)
        h, data = self.headers[f]
        return f, h[name], data

    def part_shape(self, name: str, rank: int, world: int) -> list[int]:
        _, h, _ = self.info(name)
        shape, kind = list(h["shape"]), rule(name)
        if kind == "row":
            shape[0] //= world
        elif kind == "col":
            shape[1] //= world
        return shape

    def part(self, name: str, rank: int, world: int):
        """split.py's split_bytes of a BF16 tensor for `rank`, as a torch bf16 tensor on the host."""

        import torch

        f, h, data = self.info(name)
        if h["dtype"] != "BF16" or len(h["shape"]) != 2:
            raise ValueError(f"{name}: {h['dtype']} {h['shape']}, not a 2-D BF16 matrix")
        (r, c), (b0, b1), kind = h["shape"], h["data_offsets"], rule(name)
        path = os.path.join(self.dir, f)
        if kind == "row":                                   # the rank's rows are one run: read only those
            if r % world:
                raise ValueError(f"row split of odd leading dim {h['shape']}")
            half = r // world
            a = np.fromfile(path, dtype=np.int16, count=half * c, offset=data + b0 + rank * half * c * 2)
            a = a.reshape(half, c)
        else:
            a = np.fromfile(path, dtype=np.int16, count=r * c, offset=data + b0).reshape(r, c)
            if kind == "col":
                if c % world:
                    raise ValueError(f"column split needs an even 2-D shape, got {h['shape']}")
                half = c // world
                a = np.ascontiguousarray(a[:, rank * half:(rank + 1) * half])
        if a.size * 2 > b1 - b0:
            raise ValueError(f"{name}: short read")
        return torch.from_numpy(a).view(torch.bfloat16)


def layer_plan(ck: Checkpoint, i: int, mtp: bool) -> list[tuple[str, str, list[str]]]:
    """weights.py load's q4() and stack() calls for layer i in its order: (python path, q4 | stack, checkpoint names);
    the MTP layer (plain) is MTPW's eh, then its DSA and MoE layer."""

    p = f"layers.{i}."
    base = "w.mtp.layer." if mtp else f"w.layers[{i}]."
    out: list[tuple[str, str, list[str]]] = []
    if mtp:
        out.append(("w.mtp.eh", "q4", [p + "eh_proj"]))
    a = p + "self_attn."
    if ck.has(PREFIX + a + "q_proj.weight") and not mtp:
        out += [(base + "kda.proj", "stack", [a + x for x in ("q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj")]),
                (base + "kda.fb", "q4", [a + "f_b_proj"]), (base + "kda.gb", "q4", [a + "g_b_proj"]),
                (base + "kda.o", "q4", [a + "o_proj"])]
    elif ck.has(PREFIX + a + "q_a_proj.weight"):
        out += [(base + "dsa.proj", "stack", [a + "q_a_proj", a + "kv_a_proj_with_mqa"]),
                (base + "dsa.index.kw", "stack", [a + "indexer.wk", a + "indexer.weights_proj"]),
                (base + "dsa.index.qb", "q4", [a + "indexer.wq_b"]),
                (base + "dsa.q_b", "q4", [a + "q_b_proj"]), (base + "dsa.o", "q4", [a + "o_proj"])]
    m = p + "mlp."
    if ck.has(PREFIX + m + "gate_proj.weight") and not mtp:
        out += [(base + "mlp.gu", "stack", [m + "gate_proj", m + "up_proj"]), (base + "mlp.down", "q4", [m + "down_proj"])]
    elif ck.has(PREFIX + m + "shared_experts.gate_proj.weight"):
        s = m + "shared_experts."
        out += [(base + "moe.shared.gu", "stack", [s + "gate_proj", s + "up_proj"]),
                (base + "moe.shared.down", "q4", [s + "down_proj"])]
    return out


def write_safetensors(path: str, tensors: list[tuple[str, str, list[int], bytes | int]], meta: dict) -> str:
    """Tensors in order, each data offset 256-aligned past the header; an int payload is that many zero bytes, left
    sparse (the headers-only mirror). Returns the file's sha256 (of the bytes as written)."""

    header, at = {"__metadata__": {k: str(v) for k, v in meta.items()}}, 0
    for name, dtype, shape, data in tensors:
        n = data if isinstance(data, int) else len(data)
        at = (at + 255) // 256 * 256
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
        at += n
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 + len(hj) + 255) // 256 * 256 - 8 - len(hj))   # the data starts 256-aligned in the file
    h = hashlib.sha256()
    with open(path + ".tmp", "wb") as f:
        head = struct.pack("<Q", len(hj)) + hj
        f.write(head)
        h.update(head)
        pos = 0
        for name, dtype, shape, data in tensors:
            b0, b1 = header[name]["data_offsets"]
            if b0 > pos:
                pad = b"\0" * (b0 - pos)
                f.write(pad) if not isinstance(data, int) else f.seek(b0 - pos, 1)
                h.update(pad)
            if isinstance(data, int):
                f.seek(data, 1)
                h.update(b"\0" * data)
            else:
                f.write(data)
                h.update(data)
            pos = b1
        f.truncate(len(head) + pos)
    os.replace(path + ".tmp", path)
    return h.hexdigest()


def audit(w, chunk: int = 4096, grid: int = 16):
    """quantize4_mse's search again, recording every clipping's error: the pick (it must equal the lane's), and the
    groups whose best and second-best errors are within 1e-6 relative (a GPU's sums may pick the other)."""

    import torch

    n, k = w.shape
    ties, groups, scales, biases = 0, 0, [], []
    for r in range(0, n, chunk):
        g = w[r:r + chunk].float().view(-1, k // 64, 64)
        lo0, hi0 = g.amin(-1), g.amax(-1)
        errs, ss, bs = [], [], []
        for i in range(grid):
            a = 1.0 - 0.45 * i / (grid - 1)
            lo, hi = lo0 * a, hi0 * a
            s = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
            b = lo.to(torch.bfloat16)
            q = torch.round((g - b.float()[..., None]) / s.float()[..., None]).clamp(0, 15)
            errs.append(((q * s.float()[..., None] + b.float()[..., None] - g) ** 2).sum(-1))
            ss.append(s)
            bs.append(b)
        e = torch.stack(errs)                               # [grid, rows, groups]
        best = torch.zeros_like(errs[0], dtype=torch.long)
        cur = errs[0].clone()
        for i in range(1, grid):                            # the lane's strict `<`: the first minimum wins
            take = errs[i] < cur
            cur = torch.where(take, errs[i], cur)
            best = torch.where(take, torch.full_like(best, i), best)
        srt = e.sort(0).values
        ties += int(((srt[1] - srt[0]) <= 1e-6 * srt[0].abs()).sum())
        groups += srt[0].numel()
        idx = best[None]
        scales.append(torch.stack(ss).gather(0, idx)[0])
        biases.append(torch.stack(bs).gather(0, idx)[0])
    return torch.cat(scales), torch.cat(biases), ties, groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=["make"])
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--layers", type=int, default=None, help="the first N backbone layers (default: all)")
    ap.add_argument("--only-layers", default=None, help="these backbone layers only (the self-test's slices)")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--headers-only", action="store_true", help="shapes from the headers, data left sparse (--dry)")
    ap.add_argument("--ties", action="store_true", help="count near-tie groups (doubles the time)")
    ap.add_argument("--source-id", default="078455ffe6472f9a52fbc1139f58b9db2881b25c")
    ap.add_argument("--partial", action="store_true", help="a layer may lack its MLP (the self-test's slices)")
    a = ap.parse_args(argv)
    if not 0 <= a.rank < a.world:
        ap.error("--rank must be below --world")
    ck = Checkpoint(a.ckpt)
    os.makedirs(a.out, exist_ok=True)
    ns, fsha = lane_functions()
    import torch

    dev = torch.device(a.device)
    which = list(range(a.layers if a.layers is not None else ck.layers))
    if a.only_layers:
        which = [int(x) for x in a.only_layers.split(",")]
    jobs = [(f"q4-L{i:03d}.safetensors", i, False) for i in which]
    if not a.no_mtp and ck.has(PREFIX + f"layers.{ck.layers}.eh_proj.weight"):
        jobs.append(("q4-mtp.safetensors", ck.layers, True))
    man = {"format": FORMAT, "rank": a.rank, "world": a.world, "source": {"dir": a.ckpt, "id": a.source_id},
           "patch": {"file": os.path.basename(PATCH), "sha256": PATCH_SHA256}, "functions_sha256": fsha,
           "chunk": 4096, "torch": torch.__version__, "device": str(dev), "headers_only": a.headers_only,
           "files": [], "f8": [], "ties": {"groups": 0, "near": 0}}
    t0 = time.time()
    for fname, i, mtp in jobs:
        plan = layer_plan(ck, i, mtp)
        kinds = {x[0].rsplit(".", 2)[-2] for x in plan}
        if not a.partial and not ({"mlp", "shared"} & kinds):
            raise SystemExit(f"layer {i}: no MLP found")
        tensors, rows = [], []
        for path, how, names in plan:
            full = [PREFIX + x + ".weight" for x in names]
            if a.headers_only:
                shapes = [ck.part_shape(x, a.rank, a.world) for x in full]
                n, k = sum(s[0] for s in shapes), shapes[0][1]
                if k % 64:
                    man["f8"].append(path)
                    continue
                trip = [(path + ".weight", "U32", [n, k // 8], n * k // 2),
                        (path + ".scales", "BF16", [n, k // 64], n * (k // 64) * 2),
                        (path + ".biases", "BF16", [n, k // 64], n * (k // 64) * 2)]
            else:
                parts = [ck.part(x, a.rank, a.world).to(dev) for x in full]
                res = ns["stack_dense_q4"](parts) if how == "stack" else ns["make_dense_q4"](parts[0])
                if res[0] == "f8":
                    man["f8"].append(path)
                    continue
                _, words, scales, biases = res
                if a.ties:
                    s2, b2, near, groups = audit(torch.cat(parts) if how == "stack" else parts[0])
                    if not (torch.equal(s2.view(torch.int16), scales.view(torch.int16))
                            and torch.equal(b2.view(torch.int16), biases.view(torch.int16))):
                        raise SystemExit(f"{path}: the audit's pick differs from quantize4_mse's")
                    man["ties"]["near"] += near
                    man["ties"]["groups"] += groups
                    rows.append({"path": path, "near": near, "groups": groups})
                n, k8 = words.shape
                trip = [(path + ".weight", "U32", [n, k8], words.cpu().contiguous().numpy().view(np.uint32).tobytes()),
                        (path + ".scales", "BF16", [n, k8 // 8], scales.cpu().contiguous().view(torch.int16).numpy().tobytes()),
                        (path + ".biases", "BF16", [n, k8 // 8], biases.cpu().contiguous().view(torch.int16).numpy().tobytes())]
                del parts, words, scales, biases, res
            tensors += trip
        meta = {"format": FORMAT, "rank": a.rank, "world": a.world, "layer": "mtp" if mtp else i, "source": a.source_id,
                "patch_sha256": PATCH_SHA256, "functions_sha256": fsha}
        sha = write_safetensors(os.path.join(a.out, fname), tensors, meta)
        size = os.path.getsize(os.path.join(a.out, fname))
        man["files"].append({"name": fname, "layer": "mtp" if mtp else i, "sha256": sha, "bytes": size,
                             "tensors": [{"name": t[0], "dtype": t[1], "shape": t[2]} for t in tensors],
                             "ties": rows})
        print(f"{fname}: {len(tensors) // 3} triplets, {size} bytes, sha256 {sha[:16]} ({time.time() - t0:.1f} s)", flush=True)
    man["seconds"] = round(time.time() - t0, 1)
    man["bytes"] = sum(f["bytes"] for f in man["files"])
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"rank {a.rank}: {len(man['files'])} files, {man['bytes']} bytes, f8 fallbacks {len(man['f8'])}, "
          f"near ties {man['ties']['near']} of {man['ties']['groups']} groups, {man['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
