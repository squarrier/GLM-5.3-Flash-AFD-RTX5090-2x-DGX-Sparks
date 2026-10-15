#!/usr/bin/env python3
"""python-0.6's TP reads of Mia's EXL3 GLM-5.3-Flash checkpoint from its headers alone (a header-only copy serves).

For each rank, every language-model tensor's read is python-0.6 split.py's own RankReader._span (by Ash Hart; loaded by
path and run as it is on a stand-in reader holding the index and headers), and the part it keeps is split_bytes'
(checked against split_bytes itself on each kind of tensor up to 8 MB). The head keeps its vocabulary rows, as
weights.py load() does; the EXL3 codebook markers (`.mcg`) are never read by weights.py (moe_exl3 names the codebook),
so they are listed apart. Outputs, each optional:
  --spans F    one line a copy, sorted: rank shard offset length row at width name (tf-glm-load exl3's format)
  --map F      every tensor: rule, dtype, shape, bytes, and each rank's read and kept bytes (CSV)
  --summary F  the same by name pattern (layer and expert numbers as *), with totals (JSON)
  --fixture F  layers 0-3 and 45 with experts 0-3, and their headers, for cuda_exl3_split_test.zig
usage: python -I exl3_split_fixtures.py <src/tensorfold> <CKPT> [--ranks N] [--spans F] [--map F] [--summary F] [--fixture F]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("ckpt", type=Path)
ap.add_argument("--ranks", type=int, default=2)
ap.add_argument("--spans", type=Path)
ap.add_argument("--map", type=Path)
ap.add_argument("--summary", type=Path)
ap.add_argument("--fixture", type=Path)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("glm_split", args.src / "families/glm5_next/cuda/split.py")
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)

index = json.loads((args.ckpt / "model.safetensors.index.json").read_text())["weight_map"]
shards = sorted(set(index.values()))
shard_no = {s: i for i, s in enumerate(shards)}
headers = {s: split.read_header(args.ckpt / s) for s in shards}
MARKER = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(mcg|mul1)$")


def reader(rank: int):
    """split.RankReader's state for _span on the full checkpoint: its index and headers, no files opened."""
    return SimpleNamespace(split=False, dir=args.ckpt, index=index, ranks=args.ranks, rank=rank,
                           files={str(args.ckpt / s): headers[s] for s in shards})


def cut(kind: str, shape: list, isz: int, rank: int) -> tuple[int, int, int]:
    """(row, at, width) of what split_bytes keeps of a read (0, 0, 0: all of it)."""
    if kind == "rep":
        return 0, 0, 0
    if kind == "row":       # _span reads a divisible row split's rows alone (as rep); split_bytes refuses the rest
        raise SystemExit(f"row split of {shape} does not divide by {args.ranks}")
    inner = int(np.prod(shape[2:])) * isz if kind == "dim1" else isz
    chunk = shape[1] // args.ranks * inner
    return shape[1] * inner, rank * chunk, chunk


checked: set = set()


def check(kind: str, shape: list, isz: int, rank: int) -> None:
    """The cut above against split_bytes itself: each byte's index through split.py's slicing."""
    n = int(np.prod(shape)) * isz
    key = (kind, tuple(shape), isz, rank)
    if kind == "rep" or key in checked or n > 8 << 20:
        return
    checked.add(key)
    got, _ = split.split_bytes(np.arange(n, dtype=np.int64), list(shape), isz, kind, rank, args.ranks)
    row, at, width = cut(kind, shape, isz, rank)
    want = (np.arange(n // row)[:, None] * row + at + np.arange(width)[None, :]).reshape(-1)
    if not np.array_equal(got, want):
        raise SystemExit(f"cut of {kind} {shape} differs from split_bytes")


spans, rows, markers = [], [], []
for name in sorted(index):
    kind0 = split.rule(name)
    if kind0 == "drop":
        continue
    info = headers[index[name]][0][name]
    a, b = info["data_offsets"]
    nbytes, isz = b - a, split.DTYPE_BYTES[info["dtype"]]
    if MARKER.search(name):
        markers.append(name)
        rows.append([name, kind0, info["dtype"], "x".join(map(str, info["shape"])), nbytes, "marker", 0, 0, 0, 0])
        continue
    per_rank = []
    for rank in range(args.ranks):
        file, begin, end, kind, shape, dtype = split.RankReader._span(reader(rank), name)
        row, at, width = cut(kind, shape, isz, rank)
        check(kind, shape, isz, rank)
        if name.startswith("lm_head."):          # weights.py load(): the rank's vocabulary rows of the replicated head
            vl, per = shape[0] // args.ranks, (end - begin) // shape[0]
            begin, end, row, at, width = begin + rank * vl * per, begin + (rank + 1) * vl * per, 0, 0, 0
        kept = (end - begin) // row * width if row else end - begin
        spans.append((rank, shard_no[Path(file).name], begin, end - begin, row, at, width, name))
        per_rank += [end - begin, kept]
    rows.append([name, kind0, info["dtype"], "x".join(map(str, info["shape"])), nbytes, "read", *per_rank])
spans.sort()

if args.spans:
    args.spans.write_text("".join(" ".join(map(str, s)) + "\n" for s in spans))
if args.map:
    with open(args.map, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "rule", "dtype", "shape", "bytes", "use"] +
                   [f"rank{r}_{x}" for r in range(args.ranks) for x in ("read", "kept")])
        w.writerows(rows)
totals = {"tensors": len(rows), "markers": len(markers), "bytes": sum(r[4] for r in rows),
          "visual": sum(1 for n in index if split.rule(n) == "drop"), "shards": len(shards),
          "split_py_sha256": hashlib.sha256((args.src / "families/glm5_next/cuda/split.py").read_bytes()).hexdigest()}
for r in range(args.ranks):
    totals[f"rank{r}_read"] = sum(x[6 + 2 * r] for x in rows)
    totals[f"rank{r}_kept"] = sum(x[7 + 2 * r] for x in rows)
    totals[f"rank{r}_copies"] = sum(1 for s in spans if s[0] == r)
if args.summary:
    pat: dict[str, dict] = {}
    for x in rows:
        k = re.sub(r"\.\d+\.", ".*.", re.sub(r"\.\d+\.", ".*.", x[0]))
        e = pat.setdefault(k, {"n": 0, "rule": x[1], "dtype": x[2], "shapes": set(), "bytes": 0,
                               **{f"rank{r}_{y}": 0 for r in range(args.ranks) for y in ("read", "kept")}})
        e["n"] += 1
        e["shapes"].add(x[3])
        e["bytes"] += x[4]
        for r in range(args.ranks):
            e[f"rank{r}_read"] += x[6 + 2 * r]
            e[f"rank{r}_kept"] += x[7 + 2 * r]
    for e in pat.values():
        e["shapes"] = sorted(e["shapes"])
    args.summary.write_text(json.dumps({"totals": totals, "patterns": pat}, indent=1) + "\n")
if args.fixture:
    keep = re.compile(r"^(lm_head\.|model\.language_model\.(embed_tokens|norm)\.|model\.language_model\.layers\.(0|1|2|3|45)\.)")
    expert = re.compile(r"\.mlp\.experts\.(\d+)\.")

    def wanted(n: str) -> bool:
        m = expert.search(n)
        return bool(keep.search(n)) and (m is None or int(m.group(1)) < 4)

    tensors = {}
    for n in sorted(index):
        if wanted(n):
            info = headers[index[n]][0][n]
            a, b = info["data_offsets"]
            tensors[n] = [shard_no[index[n]], headers[index[n]][1] + a, b - a, info["dtype"], info["shape"]]
    fx = {"source": "python-0.6 split.py RankReader._span and split_bytes, weights.py's head rows",
          "checkpoint": "078455ff headers", "ranks": args.ranks, "run": 4, "experts": 4,
          "tensors": tensors, "spans": [list(s) for s in spans if wanted(s[7])]}
    head = json.dumps({k: v for k, v in fx.items() if k not in ("tensors", "spans")}, indent=1)[:-2]
    body_t = ",\n".join(f"  {json.dumps(n)}: {json.dumps(t)}" for n, t in fx["tensors"].items())
    body_s = ",\n".join(f"  {json.dumps(s)}" for s in fx["spans"])
    args.fixture.write_text(f'{head},\n "tensors": {{\n{body_t}\n }},\n "spans": [\n{body_s}\n ]\n}}\n')
print(json.dumps(totals))
