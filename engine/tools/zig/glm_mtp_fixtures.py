#!/usr/bin/env python3
"""python-0.6's decode loops with MTP drafts (glm5_next/cuda/decode.py), recorded on a CPU over a scripted model."""
# qualification_qualification_44ba554e.7 day 1 (evidence/qualification_qualification_86ea5170/PLAN.md): decode.serial_decode, decode.mtp_decode and drafter_choice.auto_decode's MTP arm,
# as python-0.6 ed78d6f runs them on decode.Engine after decode.prefill, with glm_step_fixtures.py's recorders (Triton,
# extension calls, all_gather, the torch ops with device work) over a TP2 rank cut to layers 0-2 with the MTP head
# (the MTP layer's blocks are lane A's marks). The model is scripted: sample_rows runs, so its launches are recorded,
# but it answers from a toy (the target's token at absolute position p, and the head's draft there, right or wrong by
# the case's schedule). Each sampler call is logged with its kind, positions and answer: the Zig engine replays the
# answers and checks its own positions against python's. Drafts, accept/reject, commit (KDA replay, conv shift,
# set_pos) and the MTP cache's trim of rejected drafts are all in the trace. --probs writes decode._probability's
# cases instead (the draft's temperature-1 probability behind the confidence floor), from python's own function.

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import glm_forward_fixtures as gf
import glm_step_fixtures as gs
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.glm5_next.cuda import decode, drafter_choice
from tensorfold.families.glm5_next.cuda import mtp as mtp_mod

HITS = {  # which positions the head drafts right
    "all": lambda p: True,
    "none": lambda p: False,
    "mod3": lambda p: p % 3 != 0,
    "mod5": lambda p: p % 5 not in (0, 3),
    "mod7": lambda p: p % 7 != 4,
    "zero": lambda p: True,          # --dry's answers: every draw token 0, every probability 0
}


class Toy:
    """The scripted model: the target's greedy token at absolute position p, and the head's draft there."""

    def __init__(self, vocab: int, eos: tuple[int, ...], seed: int, hits: str, eos_at: int | None,
                 prob: float = 0.9) -> None:
        self.vocab, self.eos, self.seed, self.hits, self.eos_at, self.prob = vocab, eos, seed, HITS[hits], eos_at, prob

    def truth(self, p: int) -> int:
        if self.hits is HITS["zero"]:
            return 0
        if p == self.eos_at:
            return self.eos[0]
        t = (7919 * p + 104729 * self.seed + 13) % self.vocab
        return t if t not in self.eos else (t + 1) % self.vocab

    def guess(self, p: int) -> int:
        t = self.truth(p)
        return t if self.hits(p) else (t + 1 + p % 7) % self.vocab   # zero: truth 0, always hit

    def p_of(self, p: int) -> float:
        """The head's scripted temperature-1 probability for its draft at p (the confidence floor's input)."""
        if self.hits is HITS["zero"]:
            return 0.0
        return (0.95, 0.6, 0.8, 0.3, 0.99)[(p + self.seed) % 5] if self.prob < 0 else self.prob


CALLS: list[dict] = []


def scripted(e, toy: Toy):
    """decode.sample_rows runs (its launches are python's) and answers from the toy; every call is logged."""
    real = decode.sample_rows
    main, draft = gs._ptr(e.buf.logits), gs._ptr(e.mbuf.logits)

    def sample_rows(w, logits, positions, sampling, offset=None, probs=None):
        mine: list[float] | None = [] if probs is not None else None
        host = set(gs.STATE["host"])                     # the call's .cpu() rows die with it: their addresses come back
        real(w, logits, positions, sampling, offset, mine)
        gs.STATE["host"] = host
        ptr = gs._ptr(logits)
        kind = "main" if ptr == main else "draft" if ptr == draft else "first"
        picks = [toy.guess(p) if kind == "draft" else toy.truth(p) for p in positions]
        call = {"kind": kind, "positions": list(positions), "picks": picks}
        if probs is not None:
            ps = [toy.p_of(p) for p in positions]
            probs.extend(ps)
            call["probs"] = ps
        CALLS.append(call)
        return picks

    decode.sample_rows = sample_rows
    return lambda: setattr(decode, "sample_rows", real)


def staging():
    """forward.stage and both mtp_stage bindings note the ids they write (the host.write event's values)."""
    s0, m0, d0 = decode.stage, mtp_mod.mtp_stage, decode.mtp_stage

    def stage(w, st, b, tokens):
        gs.STATE["tokens"] = list(tokens)
        return s0(w, st, b, tokens)

    def mstage(w, st, b, nxt, hidden):
        gs.STATE["tokens"] = list(nxt)
        return m0(w, st, b, nxt, hidden)

    decode.stage, mtp_mod.mtp_stage, decode.mtp_stage = stage, mstage, mstage
    return lambda: (setattr(decode, "stage", s0), setattr(mtp_mod, "mtp_stage", m0), setattr(decode, "mtp_stage", d0))


def policy_of(spec: str) -> decode.DepthPolicy:
    """A case's DepthPolicy: fN (fixed N), a (adaptive, python's defaults), a:LOW:HIGH (adaptive; engine.py's sampled
    auto policy is a:0.6:0.85), cN:P (fixed N, confidence P)."""
    if spec.startswith("f"):
        return decode.DepthPolicy(int(spec[1:]), fixed=True)
    if spec == "a":
        return decode.DepthPolicy()
    if spec.startswith("a:"):
        _, low, high = spec.split(":")
        return decode.DepthPolicy(3, low=float(low), high=float(high))
    most, conf = spec[1:].split(":")
    return decode.DepthPolicy(int(most), fixed=True, confidence=float(conf))


def nucleus_marks():
    """qualification_qualification_44ba554e.7 day 2: decode.sample_rows' nucleus call (top_k 0) logs one `sample.nucleus` event on its logits rows in
    place of python's stock nucleus_rows launches. The Zig engine draws through qualification_45c10bcf's port (Mia's union cover and 0092
    packets on Ash's kernels, whose own launches cuda_nucleus_test.zig fixes), so the trace marks where each draw sits
    in the loop and which rows it reads; the scripted model answers it, as every sample call here."""
    real = decode.nucleus_rows

    def nucleus_rows(logits, positions, sampling, *, offset=0, id_map=None, gather=None, probs=None):
        assert sampling.top_k == 0 and sampling.temperature > 0, sampling
        gf.LOG.append({"op": "ext", "fn": "sample.nucleus", "args": [gs._arg(logits)]})
        if probs is not None:
            probs.extend([0.0] * len(positions))
        return [0] * len(positions)

    decode.nucleus_rows = nucleus_rows
    return lambda: setattr(decode, "nucleus_rows", real)


def decode_case(w, loop: str, policy: str, hits: str, count: int, stop_eos: bool, eos_at: int | None, seed: int,
                prob: float, n: int, rows: int, capacity: int, sampling: dict | None = None) -> dict:
    """decode.prefill of an n-token prompt (not logged), then one decode loop from its first token (logged); with
    `sampling` (exact_sampling.Sampling's fields, top_k 0) every draw is the request's nucleus draw."""
    e = gs.engine(w, rows, capacity, decode.PREFILL_ROWS)
    gs.name_engine(w, e)
    toy = Toy(w.cfg.vocab, w.cfg.eos, seed, hits, eos_at, prob)
    prompt = [(3000 + 53 * i + seed) % w.cfg.vocab for i in range(n)]
    smp = Sampling(**sampling) if sampling is not None else None
    CALLS.clear()
    undo = [gs.lane_a_marks(), staging(), scripted(e, toy), nucleus_marks()]
    try:
        with gs.Torch():
            first = decode.prefill(e, prompt, smp, mtp=True)
            gf.LOG.clear()
            prefill_calls = list(CALLS)
            CALLS.clear()
            if loop == "serial":
                res = decode.serial_decode(e, first, count, smp, stop_eos=stop_eos)
            elif loop == "mtp":
                res = decode.mtp_decode(e, first, count, smp, policy=policy_of(policy), stop_eos=stop_eos)
            else:  # auto with no DFlash2 drafter: every round MTP, under m_policy
                res = drafter_choice.auto_decode(e, None, first, count, smp, choice=None, m_policy=policy_of(policy),
                                                 f_policy=decode.DepthPolicy(5, fixed=True, confidence=0.3),
                                                 stop_eos=stop_eos)
    finally:
        for u in reversed(undo):
            u()
    assert prefill_calls == [{"kind": "first", "positions": [n], "picks": [toy.truth(n)]}], prefill_calls
    want = [toy.truth(n + i) for i in range(count)]
    if stop_eos and eos_at is not None and eos_at < n + count:
        want = want[:eos_at - n + 1]
    assert res.tokens == want, (res.tokens, want)                 # drafted == serial == the toy's truth
    st = e.st
    return {"kind": "decode", "loop": loop, "policy": policy, "hits": hits, "count": count, "stop_eos": stop_eos,
            "eos_at": eos_at, "seed": seed, "prob": prob, "prompt": prompt, "first": first, "calls": list(CALLS),
            "tokens": res.tokens, "rounds": res.rounds, "drafted": res.drafted, "accepted": res.accepted,
            "depths": res.depths, "keeps": res.keeps, "pos": st.pos, "mtp_len": st.mtp_len,
            "mtp_drafted": st.mtp_drafted, "parity": st.cur[0] if st.cur else 0, "events": list(gf.LOG),
            **({"sampling": sampling} if sampling is not None else {})}


CASES = (  # name: loop, policy, hits, count, stop_eos, eos_at (prompt offset), seed, scripted probability
    ("serial-c5", "serial", "-", "all", 5, False, None, 1, 0.9),
    ("mtp-f3-all-c9", "mtp", "f3", "all", 9, False, None, 1, 0.9),
    ("mtp-f3-none-c5", "mtp", "f3", "none", 5, False, None, 2, 0.9),
    ("mtp-f3-mod3-c10", "mtp", "f3", "mod3", 10, False, None, 3, 0.9),
    ("mtp-a-mod5-c10", "mtp", "a", "mod5", 10, False, None, 4, 0.9),
    ("mtp-f7-mod7-c13", "mtp", "f7", "mod7", 13, False, None, 5, 0.9),
    ("mtp-f3-eos-c9", "mtp", "f3", "all", 9, True, 4, 6, 0.9),
    ("auto-f3-mod3-c10", "auto", "f3", "mod3", 10, False, None, 7, 0.9),
    ("auto-c3-mod5-c10", "auto", "c3:0.35", "mod5", 10, False, None, 8, -1.0),
    ("mtp-c3-mod3-c10", "mtp", "c3:0.35", "mod3", 10, False, None, 9, -1.0),
    # every draft under the floor: python keeps each round's first draft anyway (j == 0) and chains none after it
    ("mtp-c3-low-c8", "mtp", "c3:0.35", "mod3", 8, False, None, 10, 0.2),
    ("auto-c3-low-c8", "auto", "c3:0.35", "all", 8, False, None, 11, 0.2),
)

# qualification_qualification_44ba554e.7 day 2: seeded nucleus requests (exact_sampling.Sampling, top_k 0): python's keyed draws make drafted == serial
S1 = {"seed": 1234, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0}
S2 = {"seed": 99, "temperature": 0.7, "top_p": 0.9, "top_k": 0, "min_p": 0.05}
S3 = {"seed": 7, "temperature": 1.3, "top_p": 1.0, "top_k": 0, "min_p": 0.1}   # no top_p cut: min_p alone
SAMPLED = (  # as CASES, then the request's sampling
    ("s-serial-c5", "serial", "-", "all", 5, False, None, 21, 0.9, S1),
    ("s-mtp-f3-mod3-c10", "mtp", "f3", "mod3", 10, False, None, 22, 0.9, S1),
    ("s-mtp-a-mod5-c10", "mtp", "a", "mod5", 10, False, None, 23, 0.9, S2),
    ("s-mtp-c3-mod3-c10", "mtp", "c3:0.35", "mod3", 10, False, None, 24, -1.0, S2),
    ("s-mtp-c3-low-c8", "mtp", "c3:0.35", "mod3", 8, False, None, 25, 0.2, S3),
    ("s-mtp-f7-eos-c13", "mtp", "f7", "mod7", 13, True, 6, 26, 0.9, S3),
    # engine.py's sampled auto policy (a 0.6/0.85): some drafts wrong (mod5), every draft wrong (none)
    ("s-auto-a-mod5-c10", "auto", "a:0.6:0.85", "mod5", 10, False, None, 27, 0.9, S1),
    ("s-auto-a-none-c8", "auto", "a:0.6:0.85", "none", 8, False, None, 28, 0.9, S2),
)


def _pairwise(a: list[float]) -> float:
    """numpy's add.reduce of a contiguous float64 row: 0 + DOUBLE_pairwise_sum (8 accumulators to 128, halves above)."""
    n = len(a)
    if n < 8:
        res = 0.0
        for x in a:
            res += x
        return res
    if n <= 128:
        r, i = list(a[:8]), 8
        while i < n - n % 8:
            for j in range(8):
                r[j] += a[i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for x in a[i:]:
            res += x
        return res
    half = n // 2 - (n // 2) % 8
    return _pairwise(a[:half]) + _pairwise(a[half:])


def _libm_probability(values: np.ndarray, tokens: np.ndarray, chosen: int, top_p: float = 1.0, top_k: int = 0) -> float:
    """decode._probability's arithmetic for a greedy draw with libm's exp (math.exp), as numpy runs it where it has no
    SIMD float64 exp of its own (aarch64): lexsort order (its first top_k), exp(v - max), the pairwise sum, the chosen
    share; with 0 < top_p < 1 (a temperature-0 request's cut) the shares' sequential cumsum to top_p (searchsorted + 1)
    and the kept shares over their pairwise sum."""
    order = np.lexsort((tokens, -values))[:top_k or len(values)]
    v = [float(x) for x in values[order].astype(np.float64)]
    top = max(v)
    p = [math.exp(x - top) for x in v]
    s = _pairwise(p)
    p = [x / s for x in p]
    if 0.0 < top_p < 1.0:
        run, keep = 0.0, len(p) + 1
        for i, x in enumerate(p):
            run += x
            if run >= top_p:
                keep = i + 1
                break
        p = p[:keep]
        s = _pairwise(p)
        p = [x / s for x in p]
        order = order[:keep]
    ids = tokens[order]
    hit = np.nonzero(ids == chosen)[0]
    return p[int(hit[0])] if len(hit) else 0.0


def probs_fixture(out: Path) -> int:
    """decode._probability (greedy: temperature 1, top_p 1, every gathered candidate) on drawn candidate rows, with
    sample_rows' greedy choice: the chosen token's probability as float64 bits, libm's (math.exp: numpy's on aarch64)
    and numpy's on this host (an x86_64 numpy may use its own AVX512F exp: counted, kept beside). Day 2 adds 48 cases
    of a temperature-0 request with top_p and top_k (the Sampling _probability reads)."""
    rng = np.random.default_rng(17)
    cases, differ = [], 0
    for i in range(96 + 48):
        cut = i >= 96
        world, k = (1, 2, 2, 3)[i % 4], (28, 28, 5, 1)[(i // 4) % 4] if not cut else 28
        rows = int(rng.integers(1, 4))
        scale = (0.5, 2.0, 6.0, 20.0)[(i // 16) % 4]
        values = (rng.standard_normal((rows, world * k)) * scale).astype(np.float32)
        values = values.astype(np.float16).astype(np.float32) if i % 3 == 0 else values   # coarse: ties
        if i % 11 == 0:
            values[:, : world * k // 2] = values[:, :1]            # a block of equal values
        tokens = rng.choice(154880, size=(rows, world * k), replace=False).astype(np.int64)
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[r, order[r, 0]]) for r in range(rows)]
        top_p, top_k = ((0.5, 0.8, 0.9, 0.95, 0.99, 1.0)[i % 6], (0, 0, 5, 40)[(i // 6) % 4]) if cut else (1.0, 0)
        smp = Sampling(seed=1, temperature=0.0, top_p=top_p, top_k=top_k) if cut else None
        p = decode._probability(values, tokens, chosen, smp)
        libm = [_libm_probability(values[r], tokens[r], chosen[r], top_p, top_k) for r in range(rows)]
        differ += sum(a != b for a, b in zip(p, libm))
        cases.append({"world": world, "k": k, "rows": rows, "chosen": chosen,
                      "values": [[int(x) for x in row.view(np.uint32)] for row in values], "tokens": tokens.tolist(),
                      "probs": [int(np.float64(x).view(np.uint64)) for x in libm],
                      "numpy": [int(np.float64(x).view(np.uint64)) for x in p],
                      **({"top_p": top_p, "top_k": top_k} if cut else {})})
    doc = {"generator": "tools/zig/glm_mtp_fixtures.py --probs", "schema": "glm-draft-probability/1",
           "python": "ed78d6f", "numpy": np.__version__, "numpy_differs": differ, "cases": cases}
    out.write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n")
    print(f"{len(cases)} cases, {sum(c['rows'] for c in cases)} rows; numpy {np.__version__} differs from libm in "
          f"{differ} -> {out}")
    return 0


DRY_FORMS = (  # tf-glm-forward --drafts: python's loop and policy (auto: engine.py's greedy m_policy c3:0.35)
    ("0", "serial", "-"), ("f3", "mtp", "f3"), ("f7", "mtp", "f7"), ("a", "mtp", "a"), ("c3:0.35", "mtp", "c3:0.35"),
    ("auto", "auto", "c3:0.35"),
)


def dry_log(w, out: Path, n: int, count: int, rows: int, capacity: int) -> int:
    """Each --drafts form's loop on --dry's answers: the draws (kind, positions) in order, the counters, the state."""
    forms = {}
    for spec, loop, policy in DRY_FORMS:
        c = decode_case(w, loop, policy, "zero", count, False, None, 0, 0.0, n, rows, capacity)
        forms[spec] = {k: c[k] for k in ("loop", "policy", "first", "tokens", "rounds", "drafted", "accepted", "depths",
                                         "keeps", "pos", "mtp_len", "mtp_drafted")}
        forms[spec]["calls"] = [{"kind": x["kind"], "positions": x["positions"]} for x in c["calls"]]
        print(f"{spec}: {len(c['calls'])} draws, rounds {c['rounds']}, drafted {c['drafted']}, accepted "
              f"{c['accepted']}, pos {c['pos']}, mtp_len {c['mtp_len']}")
    doc = {"generator": "tools/zig/glm_mtp_fixtures.py --dry-log", "schema": "glm-decode-dry-log/1", "python": "ed78d6f",
           "prompt_tokens": n, "count": count, "forms": forms}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", type=Path, help="the checkpoint (a header-only sparse mirror serves)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--layers", type=int, default=3, help="backbone layers loaded (layers 3+ are lane A's blocks)")
    ap.add_argument("--probs", action="store_true", help="decode._probability's cases, not the trace")
    ap.add_argument("--sampled", action="store_true", help="the seeded nucleus cases (SAMPLED), not CASES")
    ap.add_argument("--only", nargs="*", help="case names (default: all)")
    ap.add_argument("--dry-log", type=int, nargs=2, metavar=("PROMPT", "COUNT"),
                    help="the loops on --dry's answers for each --drafts form: a PROMPT-token prompt, COUNT tokens")
    a = ap.parse_args()
    if a.probs:
        return probs_fixture(a.out)
    if a.mirror is None:
        ap.error("--mirror is required for the trace")
    gf.patch()
    torch.cuda.synchronize = lambda *x, **k: None             # decode's _sync and its loops' synchronize: no GPU
    full, w = gs.load(a.mirror, a.layers, 0, 2, True)
    rows, capacity, n = 8, 2560, 5
    if a.dry_log:
        return dry_log(w, a.out, a.dry_log[0], a.dry_log[1], rows, capacity)
    cases = {}
    for name, loop, policy, hits, count, stop_eos, eos_at, seed, prob, *smp in (SAMPLED if a.sampled else CASES):
        if a.only and name not in a.only:
            continue
        cases[name] = decode_case(w, loop, policy, hits, count, stop_eos, None if eos_at is None else n + eos_at,
                                  seed, prob, n, rows, capacity, smp[0] if smp else None)
        c = cases[name]
        print(f"{name}: {len(c['events'])} events, {len(c['calls'])} sampler calls, rounds {c['rounds']}, "
              f"drafted {c['drafted']}, accepted {c['accepted']}, keeps {c['keeps']}")
    e = gs.engine(w, rows, capacity, decode.PREFILL_ROWS)
    gs.name_engine(w, e)
    out = {"generator": "tools/zig/glm_mtp_fixtures.py", "schema": "glm-decode-trace/1", "python": "ed78d6f",
           "rank": 0, "world": 2, "layers": a.layers, "mtp": True, "max_rows": rows, "capacity": capacity,
           "prefill_rows": decode.PREFILL_ROWS, "long_context": True, "latent": gs.forward.latent.ENABLED,
           "config_json": (a.mirror / "config.json").read_text(), "config": gf.config_dict(full),
           "vocab_offset": w.vocab_offset, "buffers": gf.NAMES.buffers(), "cases": cases}
    a.out.write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    nev = sum(len(c["events"]) for c in cases.values())
    print(f"{len(cases)} cases, {nev} events, {len(out['buffers'])} buffers -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
