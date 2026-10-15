"""Checks (or rewrites) zig/kernels/cuda/g53/: python TF-AFD 2.15's families/glm5_next/cuda/g53_rank.cu (blob 798130a, its
patch 0034: glm53f-afd's large-M expert kernels by Hugh Madden, @dangerm00se, glm53f-afd @ 91db3cc, MIT) split at
function boundaries with every device line unchanged. rank_math.cuh: its lines 1-427 (the provenance header, the
constants, the TensorFold decoder, Args, the row ops, the epilogue and reduce items, the fused tails) less line 50's
#include "g53_rank.h" (the host launcher's header) and the blank line before it, then the namespace closed; rank_kernels.cuh:
its lines 428-885 (the plan, gate/up, the unfused epilogue, down, the unfused reduce, rotrows, order) in the namespace;
rank.cu (written by hand) instantiates what g53_ffn_launch launched. The launcher itself (lines 886-994) is Zig:
zig/src/families/glm/cuda_g53.zig.

usage: python3 zig/tests/cuda/g53_copies.py PY_TREE [--write]   (PY_TREE: a python TF-AFD 2.15 checkout)"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

USAGE = "usage: python3 zig/tests/cuda/g53_copies.py PY_TREE [--write]   (PY_TREE: a python TF-AFD 2.15 checkout)"
SRC = "src/tensorfold/families/glm5_next/cuda/g53_rank.cu"
BLOB = "798130a"
OUT = Path(__file__).resolve().parents[2] / "kernels/cuda/g53"
PORT = [
    "//",
    "// Zig engine (TensorFold zig/kernels/cuda/g53/, qualification_1b9bf461): the device code of python TF-AFD 2.15's",
    "// families/glm5_next/cuda/g53_rank.cu (squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks 2.15, patch 0034), unchanged and",
    "// split at function boundaries by zig/tests/cuda/g53_copies.py: this file (its lines 1-427 less the host header's",
    "// include), rank_kernels.cuh (lines 428-885), rank.cu (the kernels its host launcher instantiated). The launcher",
    "// (launch_persistent, g53_ffn_launch) is zig/src/families/glm/cuda_g53.zig.",
]
KERN = [
    "// glm53f-afd's large-M expert kernels (Hugh Madden, @dangerm00se, github.com/hughmadden; glm53f-afd @ 91db3cc, MIT",
    "// License, Copyright (c) 2026 Turquoise Bay AI Pty Ltd): python TF-AFD 2.15's g53_rank.cu lines 428-885, unchanged (the",
    "// plan, gate/up, the unfused epilogue, down, the unfused reduce, rotrows and order; zig/tests/cuda/g53_copies.py).",
    "// rank_math.cuh's header has the provenance and the port's rules; rank.cu instantiates the templates.",
    "#pragma once",
    "",
    '#include "rank_math.cuh"',
    "",
    "namespace g53 {",
    "",
]


def find(lines: list[str], text: str) -> int:
    hits = [i for i, x in enumerate(lines) if x.startswith(text)]
    if len(hits) != 1:
        raise SystemExit(f"{SRC}: {len(hits)} lines start with {text!r}, not 1")
    return hits[0]


def split(raw: bytes) -> dict[str, str]:
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    if not blob.startswith(BLOB):
        raise SystemExit(f"{SRC} is blob {blob[:7]}, not 2.15's {BLOB}")
    lines = raw.decode().split("\n")
    inc, ns = find(lines, '#include "g53_rank.h"'), find(lines, "namespace g53 {")
    plan, launch = find(lines, "// ---- plan: rows x 8 routes"), find(lines, "// Launch a persistent large-M kernel")
    end = find(lines, "}  // namespace g53")
    head = lines[:43]
    assert head[-1].startswith("// ascending k tiles.") and lines[inc - 2] == "#include <cstdint>", "2.15's layout"
    assert lines[inc - 1] == "" and lines[inc + 1] == "" and ns == inc + 2 and (inc, plan, launch) == (49, 427, 885)
    math = head + PORT + ["#pragma once"] + lines[43:inc - 1] + lines[inc + 1:plan] + ["}  // namespace g53", ""]
    kern = KERN + lines[plan:launch] + [lines[end], ""]
    dev = math[len(head) + len(PORT) + 1:-2] + kern[len(KERN):-2]
    assert dev == lines[43:inc - 1] + lines[inc + 1:launch], "a device line changed"
    return {"rank_math.cuh": "\n".join(math), "rank_kernels.cuh": "\n".join(kern)}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1].startswith("-"):
        print(USAGE)
        return 2
    want = split((Path(argv[1]) / SRC).read_bytes())
    write = "--write" in argv[2:]
    bad = 0
    for name, text in want.items():
        path = OUT / name
        if write:
            path.write_text(text)
            print(f"wrote {path.name}: {text.count(chr(10))} lines")
        elif not path.exists() or path.read_text() != text:
            print(f"DIFFER {path.name}: not python 2.15's lines (rewrite with --write)")
            bad += 1
        else:
            print(f"SAME {path.name}: python 2.15's g53_rank.cu device lines, unchanged")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
