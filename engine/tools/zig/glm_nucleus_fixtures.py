#!/usr/bin/env python3
"""python-0.6's TP nucleus draw on a CPU, stock and with Mia's union cover (patch 0034): fixtures for Zig's host draw."""
# qualification_45c10bcf (evidence/qualification_c949ced8/NOTES.md §3), in the pattern of Mia's tools/test_nucleus_union.py (MiaAI-Lab recipe 549fdc5):
# rows over 2-4 ranks' vocabulary shards, packed as nucleus_rows' `_shares` packs them, drawn by `_draw`
# (cuda/sampling.py, ed78d6f). Concentrated, normal, flat, tied and broad rows of 20,000 tokens; GLM's 154,880 over two
# ranks (Zipf, peaked, broad, flat; verify windows of 1-4 rows). Logits and masses come from integer arithmetic and
# IEEE double ops only (splitmix64, bf16 rounding, `texp`), so zig/src/families/glm/cuda_nucleus_test.zig remakes them
# bit for bit. Expected: the unpatched `_draw`'s tokens and shares and its gathers, and the gathers of 0034's path
# (TENSORFOLD_NUCLEUS_UNION=1), whose draws must be the same.
# usage: glm_nucleus_fixtures.py --stock PY06/src --union PY06_0034/src --out fixtures_cuda_nucleus.json

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

M64 = (1 << 64) - 1
GOLDEN, STEP = 0x9E3779B97F4A7C15, 0xD1B54A32D192ED03
# texp: exp by ln2 range reduction (fdlibm's split) and a degree-11 Taylor polynomial, Horner, no fused ops
LN2_HI, LN2_LO = float.fromhex("0x1.62e42fee00000p-1"), float.fromhex("0x1.a39ef35793c76p-33")
INV_LN2 = float.fromhex("0x1.71547652b82fep+0")
TAYLOR = [float.fromhex(h) for h in (
    "0x1.0000000000000p+0", "0x1.0000000000000p+0", "0x1.0000000000000p-1", "0x1.5555555555555p-3",
    "0x1.5555555555555p-5", "0x1.1111111111111p-7", "0x1.6c16c16c16c17p-10", "0x1.a01a01a01a01ap-13",
    "0x1.a01a01a01a01ap-16", "0x1.71de3a556c734p-19", "0x1.27e4fb7789f5cp-22", "0x1.ae64567f544e4p-26")]
CUTOFF = -45.0                  # d below it: mass 0 (floor(e^d * 2^40) is 0 there anyway)
SMALL, GLM = 20000, 154880


def load(name: str, root: str):
    """``tensorfold/cuda/sampling.py`` under ``root`` as module ``name`` (its exact_sampling from sys.path)."""

    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "tensorfold/cuda/sampling.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(30))
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x = x ^ (x >> np.uint64(27))
        x = x * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def words(seed: int, stream: int, n: int) -> np.ndarray:
    """n splitmix64 words of (seed, stream): mix(key + j * STEP), key = mix(seed * GOLDEN + stream)."""

    key = mix(np.array([(seed * GOLDEN + stream) & M64], dtype=np.uint64))[0]
    with np.errstate(over="ignore"):
        return mix(key + np.arange(n, dtype=np.uint64) * np.uint64(STEP))


def normal(seed: int, stream: int, n: int) -> np.ndarray:
    """About N(0, 1.15): four 16-bit fields of each word summed, centred, times 2^-15 (exact doubles)."""

    w = words(seed, stream, n)
    f = np.full(n, -131070, dtype=np.int64)
    for s in (0, 16, 32, 48):
        f += ((w >> np.uint64(s)) & np.uint64(0xFFFF)).astype(np.int64)
    return f.astype(np.float64) * 2.0 ** -15


def pick(seed: int, stream: int, k: int, vocab: int) -> np.ndarray:
    """k distinct ids: the vocabulary ordered by (word, id), its first k."""

    return np.lexsort((np.arange(vocab), words(seed, stream, vocab)))[:k]


def lg(n: np.ndarray) -> np.ndarray:
    """log2(n) to 1/32 from the bits: the leading bit's place plus the next five bits over 32."""

    e = np.zeros_like(n)
    for k in range(1, 63):
        e += (n >> k) > 0
    return e.astype(np.float64) + (((n << 5) >> e) & 31).astype(np.float64) / 32.0


def logits(seed: int, kind: str, rows: int, vocab: int) -> np.ndarray:
    """A case's rows as fp32 values of bf16 logits (rounded to nearest even), as the model's head leaves them."""

    out = np.zeros((rows, vocab))
    for r in range(rows):
        z = normal(seed, 10 + r, vocab)
        if kind == "peaked":                                   # concentrated: 40 ids well above the rest
            out[r] = z
            out[r, :40] += 12.0
        elif kind == "normal":
            out[r] = 4.0 * z
        elif kind == "flat":
            out[r] = 0.25 * z
        elif kind == "tied":                                   # zeros, and 8 tied peaks shared by the rows
            out[r, pick(seed, 99, 8, vocab)] = 2.0
        elif kind == "broad":                                  # ~4,000 near-equal candidates over the ranks
            out[r] = z
            out[r, pick(seed, 200 + r, 4000, vocab)] += 7.0
        elif kind == "zipf":                                   # an LM-like long tail: -0.75 log2(rank)
            out[r, pick(seed, 300 + r, vocab, vocab)] = -0.75 * lg(np.arange(1, vocab + 1, dtype=np.int64))
        elif kind == "gpeaked":                                # one clear choice and four alternatives
            out[r] = z
            out[r, pick(seed, 98, 5, vocab)] += 14.0
        elif kind == "gbroad":
            out[r] = z
            out[r, pick(seed, 400 + r, 4000, vocab)] += 9.0
        elif kind == "gflat":
            out[r] = z * 2.0 ** -10
        else:
            raise ValueError(kind)
    b = out.astype(np.float32).view(np.uint32)
    b = (b + np.uint32(0x7FFF) + ((b >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return b.view(np.float32)


def texp(d: np.ndarray) -> np.ndarray:
    k = np.floor(d * INV_LN2 + 0.5)
    r = (d - k * LN2_HI) - k * LN2_LO
    p = np.full_like(d, TAYLOR[-1])
    for c in TAYLOR[-2::-1]:
        p = p * r + c
    return np.ldexp(p, k.astype(np.int32))


def scaled_mass(x: np.ndarray, temperature: float):
    """nucleus_rows' f64 scaling (the GPU's x * (1 / t)), and its fixed-point masses with texp for torch.exp."""

    scaled = x.astype(np.float64) * (1.0 / max(temperature, 1e-6))
    d = scaled - scaled.max(axis=1)[:, None]
    e = texp(np.maximum(d, CUTOFF))
    mass = np.where(d < CUTOFF, 0, np.floor(e * 2.0 ** 40)).astype(np.int64)
    return scaled, mass


def shares(scaled: np.ndarray, mass: np.ndarray, count: int, cuts: list[int]):
    """``_shares``' gathered arrays: each rank's top ``count`` per row (ties to the lower id), padded; sums, widths."""

    rows = scaled.shape[0]
    vs, ids, ms, sums, wid = [], [], [], [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        v = np.full((rows, count), -np.inf)
        i = np.full((rows, count), -1, np.int64)
        m = np.zeros((rows, count), np.int64)
        for r in range(rows):
            order = np.lexsort((np.arange(lo, hi), -scaled[r, lo:hi]))[:count]
            n = len(order)
            v[r, :n], i[r, :n], m[r, :n] = scaled[r, lo:hi][order], order + lo, mass[r, lo:hi][order]
        vs.append(v), ids.append(i), ms.append(m)
        sums.append(mass[:, lo:hi].sum(axis=1)), wid.append(np.full(rows, hi - lo))
    return np.stack(vs), np.stack(ids), np.stack(ms), np.stack(sums), np.stack(wid)


def nucleus(mod, scaled, mass, positions, smp, cuts, union: bool):
    """nucleus_rows' gathers: NUCLEUS candidates; with 0034's flag WIDER next when a shard is wider; whole shards."""

    os.environ["TENSORFOLD_NUCLEUS_UNION"] = "1" if union else "0"
    got = shares(scaled, mass, mod.NUCLEUS, cuts)
    drawn, trips = mod._draw(got, positions, smp), 1
    if drawn is None and union and mod.union_cover() and int(got[4].max()) > mod.WIDER:
        drawn, trips = mod._draw(shares(scaled, mass, mod.WIDER, cuts), positions, smp), 2
    if drawn is None:
        drawn, trips = mod._draw(shares(scaled, mass, int(got[4].max()), cuts), positions, smp), trips + 1
    return drawn, trips


def cases() -> list[dict]:
    """The card's ranges: 2-4 ranks, temperature 0.7-1.3, top_p 0.6-0.99, min_p 0 and 0.05; top_p 1 (no cut) too."""

    out = []
    kinds = ("peaked", "normal", "flat", "tied", "broad")
    for c in range(120):
        w = [int(x) for x in words(7, 1000 + c, 8)]
        ranks = 2 + w[0] % 3
        out.append(dict(seed=w[1] >> 1, kind=kinds[c % 5], vocab=SMALL, ranks=ranks, rows=3,
                        temperature=(1.0, 0.7, 1.3, 0.85, 1.15)[w[2] % 5], top_p=(0.95, 0.9, 0.6, 0.99, 0.75)[w[3] % 5],
                        min_p=(0.0, 0.05)[w[4] % 2], sampling_seed=(w[5] >> 1) if c % 2 else 1 + w[5] % 10 ** 6,
                        positions=[w[6] % 100000 + r for r in range(3)]))
    for c, (kind, min_p) in enumerate([(k, p) for k in ("peaked", "normal", "tied", "broad") for p in (0.0, 0.05)]):
        w = [int(x) for x in words(7, 2000 + c, 8)]
        out.append(dict(seed=w[1] >> 1, kind=kind, vocab=SMALL, ranks=2 + w[0] % 3, rows=3, temperature=1.0,
                        top_p=1.0, min_p=min_p, sampling_seed=1 + w[5] % 10 ** 6, positions=[w[6] % 100000 + r for r in range(3)]))
    gkinds = ("zipf", "gpeaked", "gbroad", "gflat")
    for c in range(24):
        w = [int(x) for x in words(7, 3000 + c, 8)]
        rows = 1 + c % 4                                       # a decode row, and MTP verify windows of 2-4 rows
        out.append(dict(seed=w[1] >> 1, kind=gkinds[c % 4], vocab=GLM, ranks=2, rows=rows,
                        temperature=(1.0, 0.7, 1.3)[(c // 4) % 3], top_p=(0.95, 0.95, 0.9, 0.99, 0.6, 0.95)[c // 4],
                        min_p=(0.0, 0.05)[(c // 8) % 2], sampling_seed=w[5] >> 1,
                        positions=[w[6] % 200000 + r for r in range(rows)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", required=True, help="python-0.6's src (ed78d6f)")
    ap.add_argument("--union", required=True, help="the same src with Mia's 0034 applied")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.stock)
    from tensorfold.engine.exact_sampling import Sampling
    stock, union = load("stock_sampling", a.stock), load("union_sampling", a.union)
    assert not hasattr(stock, "union_cover") and hasattr(union, "union_cover") and union.WIDER == 16 * stock.NUCLEUS
    sha = {k: hashlib.sha256(open(os.path.join(p, "tensorfold/cuda/sampling.py"), "rb").read()).hexdigest()
           for k, p in (("stock", a.stock), ("union", a.union))}
    rows_out, trips = [], {"stock": [0, 0, 0], "union": [0, 0, 0]}
    for i, c in enumerate(cases()):
        x = logits(c["seed"], c["kind"], c["rows"], c["vocab"])
        scaled, mass = scaled_mass(x, c["temperature"])
        cuts = [int(v) for v in np.linspace(0, c["vocab"], c["ranks"] + 1).astype(int)]
        smp = Sampling(c["sampling_seed"], c["temperature"], 0, c["top_p"], c["min_p"])
        a_drawn, a_trips = nucleus(stock, scaled, mass, c["positions"], smp, cuts, False)
        b_drawn, b_trips = nucleus(union, scaled, mass, c["positions"], smp, cuts, True)
        o_drawn, o_trips = nucleus(union, scaled, mass, c["positions"], smp, cuts, False)
        assert a_drawn == b_drawn == o_drawn and a_trips == o_trips, f"case {i}: {a_drawn} {b_drawn} {o_drawn}"
        trips["stock"][a_trips - 1] += 1
        trips["union"][b_trips - 1] += 1
        digest = hashlib.sha256(x.astype("<f4").tobytes() + mass.astype("<i8").tobytes()).hexdigest()[:16]
        rows_out.append(dict(c, cuts=cuts, digest=digest, tokens=[t for t, _ in a_drawn],
                             shares=[int(np.float64(s).view(np.uint64)) for _, s in a_drawn],
                             gathers_stock=a_trips, gathers_union=b_trips))
        print(f"case {i:3d} {c['kind']:7s} V={c['vocab']} ranks={c['ranks']} rows={c['rows']} t={c['temperature']} "
              f"p={c['top_p']} min_p={c['min_p']}: tokens {rows_out[-1]['tokens']} gathers {a_trips} -> {b_trips}",
              flush=True)
    fx = {"schema": "glm-nucleus-draw/1", "python": "ed78d6f", "patch": "0034 (MiaAI-Lab recipe 549fdc5)",
          "sampling_sha256": sha, "numpy": np.__version__, "nucleus": stock.NUCLEUS, "wider": union.WIDER,
          "mass": stock.MASS, "texp_cutoff": CUTOFF, "gathers": trips}
    with open(a.out, "w") as f:                       # a case a line
        f.write(json.dumps(fx, separators=(",", ":"))[:-1] + ',"cases":[\n')
        f.write(",\n".join(json.dumps(r, separators=(",", ":")) for r in rows_out))
        f.write("\n]}\n")
    n = sum(len(r["tokens"]) for r in rows_out)
    print(f"{len(rows_out)} cases, {n} draws: stock == union; gathers (1, 2, 3) stock {trips['stock']}, "
          f"union {trips['union']}")


if __name__ == "__main__":
    main()
