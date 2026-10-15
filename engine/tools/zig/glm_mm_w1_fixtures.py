"""GLM-5.3-Flash's qmm_group launches at the 3-node attention role's world-1 shapes, as python-0.6 makes them on attention-host's
RTX 5090 (12.0, 170 SMs) and on a GB10 (12.1, 48 SMs), for zig/src/families/glm/cuda_kernels_mm_test.zig (qualification_18588626).

The shapes are tools/zig/glm_mm_w1_shapes.json: every qmm_group call shape (rows, n, k, x row stride, fp32 out, K slices)
the 3-node kits' attention cases record under `attn --dry` (zig/tests/cuda/afd_cover_qmm.zig's kit replica: decode, the
drafts' verify windows, the q4 draft head); each projection (n, k, x stride, out) at every window of 1-8 rows (decode,
the verify windows of c7's drafts up to 7) and at 9 and 16 rows (tile 8), whether a kit records that window or not.
Each goes through glm5_next qmm.matmul (python's own K slices and tile pick: split_k, group_tile) with the extension
recorded, then through qmm_group.cu's own C++ (tools/zig/glm_mm_fixtures.py's stand-in build), which prints the launch.
python's K slices must be the Zig calls'; the draft head (vocab x hidden, from --config's config.json) must be a shape.

Runs in NVIDIA's PyTorch 26.07 image, no GPU: python-0.6 on PYTHONPATH, its tree at --py06, this tree at --fork.
usage: glm_mm_w1_fixtures.py --py06 /tensorfold --fork /fork --work DIR --shapes FILE --config FILE --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

import glm_mm_fixtures as base
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.glm5_next.cuda import qmm as glm_qmm

CHIPS = {"rtx5090": base.RTX5090, "gb10": base.GB10}
WINDOWS = tuple(range(1, 9)) + (9, 16)  # decode, the verify windows of 2-8 rows (c7 drafts up to 7), tile 8's ends


def cases(shapes: list[dict]):
    """(id, chip, shape, the call) for every projection the kits record (n, k, x row stride, fp32 out) at every window of
    WINDOWS, on both chips: a recorded (rows, projection) as recorded (its K slices), the others with the projection's
    first recorded K slices; a recorded shape at another row count is kept too."""
    proj = lambda s: (s["n"], s["k"], s["x_stride"], s["f32"])
    first, recorded = {}, {}
    for s in shapes:
        first.setdefault(proj(s), s)
        recorded[(s["m"], *proj(s))] = s
    rows = [recorded.get((m, *p), {**s, "m": m, "calls": 0}) for p, s in first.items() for m in WINDOWS]
    rows += [s for s in shapes if s["m"] not in WINDOWS]
    for name, chip in CHIPS.items():
        for s in rows:
            m, n, k, stride, f32 = s["m"], s["n"], s["k"], s["x_stride"], s["f32"]
            cid = f"w1-{name}/m{m}/n{n}/k{k}/x{stride}/{'f32' if f32 else 'bf16'}"

            def call(m=m, n=n, k=k, stride=stride, f32=f32):
                glm_qmm.matmul(base.rows(m, k, stride), base.q4(n, k), torch.empty(m, k // 64, dtype=torch.float32),
                               out=torch.empty(m, n, dtype=torch.float32 if f32 else torch.bfloat16), f32=f32)
            yield cid, chip, s, call


def main() -> int:
    ap = argparse.ArgumentParser()
    for opt in ("--py06", "--fork", "--work", "--shapes", "--config", "--out"):
        ap.add_argument(opt, type=Path, required=True)
    a = ap.parse_args()
    shapes = json.loads(a.shapes.read_text())["shapes"]
    cfg = json.loads(a.config.read_text())
    tc = cfg.get("text_config", cfg)
    head = (tc["vocab_size"], tc["hidden_size"])
    if not any((s["n"], s["k"]) == head and not s["f32"] for s in shapes):
        raise SystemExit(f"the draft head {head} (config.json) is not among the shapes")
    calls: list = []
    shared._ext = lambda: base.Recorder("qmm", calls)
    shared.grouped = lambda device: True
    chip = list(base.GB10)
    shared._chip = lambda device: tuple(chip[:3])
    out_cases = []
    for cid, c, s, run in cases(shapes):
        chip[:] = c
        del calls[:]
        run()
        if len(calls) != 1 or calls[0][1] != "qmm_group":
            raise SystemExit(f"{cid}: python made {[(m, fn) for m, fn, _ in calls]}, not one qmm_group")
        sks = calls[0][2][7]
        if list(sks) != [s["sk"]]:
            raise SystemExit(f"{cid}: python's K slices {sks}, the Zig call's {s['sk']}")
        space = base.Space()
        out_cases.append({"id": cid, "chip": list(c), "calls": [{"module": m, "fn": fn, "args": [base.encode(x, space)
                                                                                                 for x in args]}
                                                                for m, fn, args in calls]})
    exe = base.build(a.py06, a.fork, a.work)
    table = base.kparams(exe)
    (a.work / "params.txt").write_text("".join(f"{k} {len(v)} {' '.join(str(s) for _, s in v)}\n"
                                               for k, v in sorted(table.items())))
    lines = [f"OCC {base.OCCUPANCY}"]
    for c in out_cases:
        lines.append(f"CASE {c['id']}")
        lines.append("CHIP " + " ".join(map(str, c["chip"])))
        for call in c["calls"]:
            lines.append(" ".join(["CALL", call["module"], call["fn"], str(len(call["args"])), *map(base.wire,
                                                                                                  call["args"])]))
    got = subprocess.run([str(exe), str(a.work / "params.txt")], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout
    (a.work / "launches-w1.txt").write_text(got)
    by_id = {c["id"]: c for c in out_cases}
    attrs, current = set(), None
    for line in got.splitlines():
        w = line.split()
        if w[0] == "CASE":
            current = by_id[w[1]]
        elif w[0] == "LAUNCH":
            n = int(w[14])
            current["calls"][int(w[1])].setdefault("launches", []).append(
                {"kernel": base.key(w[2]), "grid": list(map(int, w[3:6])), "block": list(map(int, w[6:9])),
                 "smem": int(w[9]), "cluster": list(map(int, w[10:13])), "pdl": int(w[13]), "args": w[15:15 + n]})
        elif w[0] == "ATTR":
            attrs.add((base.key(w[2]), int(w[3]), int(w[4])))
        elif w[0] == "ERROR":
            current["calls"][int(w[1])]["error"] = line.split(" ", 2)[2]
    srcs = [a.py06 / "src/tensorfold/cuda/kernels" / s for s in base.QMM_SOURCES]
    out = {"schema": "glm-mm-launches/1", "generator": "tools/zig/glm_mm_w1_fixtures.py",
           "python": {"tree": "python-0.6 ed78d6f", "sources": {str(s.relative_to(a.py06)): hashlib.sha256(
               s.read_bytes()).hexdigest() for s in srcs}},
           "shapes": {"file": "tools/zig/glm_mm_w1_shapes.json", "sha256": hashlib.sha256(a.shapes.read_bytes()).hexdigest(),
                      "draft_head": list(head)},
           "occupancy": base.OCCUPANCY, "attrs": sorted([list(x) for x in attrs]),
           "kparams": {k: v for k, v in table.items() if any(
               ln.get("kernel") == k for c in out_cases for call in c["calls"] for ln in call.get("launches", []))},
           "cases": out_cases}
    a.out.write_text(json.dumps(out, indent=None, separators=(",", ":"), sort_keys=True) + "\n")
    n_launch = sum(len(call.get("launches", [])) for c in out_cases for call in c["calls"])
    n_err = sum("error" in call for c in out_cases for call in c["calls"])
    kinds = sorted({ln["kernel"][:44] for c in out_cases for call in c["calls"] for ln in call.get("launches", [])})
    projs = len({(s["n"], s["k"], s["x_stride"], s["f32"]) for s in shapes})
    print(f"{len(out_cases)} cases ({len(shapes)} shapes, {projs} projections at windows {WINDOWS}, both chips), "
          f"{n_launch} launches, {n_err} "
          f"errors, {len(attrs)} raised limits, kernels {kinds}; {a.out} sha256 "
          f"{hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
