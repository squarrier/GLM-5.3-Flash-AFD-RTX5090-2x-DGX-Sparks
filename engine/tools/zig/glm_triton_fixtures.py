#!/usr/bin/env python3
"""GLM-5.3-Flash's Triton launches as python-0.6's glm5_next/cuda wrappers make them, recorded on CPU, none run."""
# Adapted from tools/zig/flashnext_triton_fixtures.py, written by Mia (MiaAI-Lab, @MiaAI_lab) in
# MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429 (patches/0004-triton-replay-torch-ops.patch), Apache 2.0.
# Changed for GLM-5.3-Flash: the recorders patch python-0.6's glm5_next/cuda modules, the case list is
# glm_triton_cases.py, and every call site of a launched kernel must be reached (a miss fails the run).

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import torch
from triton.runtime.jit import JITFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_triton_cases as cases_mod  # noqa: E402
from tensorfold.families.glm5_next.cuda import attention, forward, glue, latent, qmm, sparse  # noqa: E402

MODULES = (attention, forward, glue, latent, qmm, sparse)
TYPES = {torch.bfloat16: "*bf16", torch.float32: "*fp32", torch.float16: "*fp16", torch.int32: "*i32",
         torch.int64: "*i64", torch.uint8: "*u8", torch.int8: "*i8"}
LOG: list[dict] = []


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def const(v):
    """A constexpr as the Zig wrappers pass it: ints and bools as ints, floats as fp32 bits, None as null."""
    if v is None:
        return None
    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": f32_bits(v)}
    raise TypeError(f"constexpr {v!r}")


class Recorder:
    """Stands in for a JITFunction: ``kernel[grid](*args, **kw)`` appends the launch and its call site to LOG."""

    def __init__(self, jit: JITFunction) -> None:
        self.jit = jit

    def __getitem__(self, grid):
        def call(*args, **kw):
            frame = sys._getframe(1)
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
                    runtime.append({"name": p.name, "type": TYPES[v.dtype], "align16": v.data_ptr() % 16 == 0})
                elif isinstance(v, bool):
                    raise TypeError(f"{p.name}: a runtime bool")
                elif isinstance(v, int):
                    runtime.append({"name": p.name, "type": "i32", "int": v})
                elif isinstance(v, float):
                    runtime.append({"name": p.name, "type": "fp32", "f32": f32_bits(v)})
                else:
                    raise TypeError(f"{p.name}: {type(v)}")
            g = list(grid) + [1] * (3 - len(grid))
            LOG.append({"fn": self.jit.fn.__name__, "grid": [int(x) for x in g], "args": runtime, "consts": consts,
                        "num_warps": kw.get("num_warps"), "num_stages": kw.get("num_stages"),
                        "site": f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}"})

        return call


def patch() -> list[str]:
    """Every module-level JITFunction of the six modules becomes a Recorder (helpers are never called by a wrapper)."""
    done = []
    for mod in MODULES:
        for name, v in list(vars(mod).items()):
            if isinstance(v, JITFunction):
                setattr(mod, name, Recorder(v))
                done.append(f"{mod.__name__.rsplit('.', 1)[-1]}.{name}")
    return done


def sites(pkg: Path) -> list[dict]:
    """Every ``kernel[grid](...)`` of a module-level @triton.jit kernel in glm5_next/cuda: file, kernel, wrapper."""
    out = []
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text())
        jits = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                and any("jit" in ast.unparse(d) for d in n.decorator_list)}

        def walk(node, wrapper):
            for child in ast.iter_child_nodes(node):
                inner = child.name if isinstance(child, ast.FunctionDef) else wrapper
                f = getattr(child, "func", None)
                if isinstance(child, ast.Call) and isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) \
                        and f.value.id in jits:
                    out.append({"file": path.name, "kernel": f.value.id, "wrapper": wrapper, "line": child.lineno})
                walk(child, inner)

        walk(tree, "<module>")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--world", type=int, default=2, help="ranks the rank shapes are cut for (python-0.6: 1, 2 or 4)")
    a = ap.parse_args()
    patch()
    cases: dict[str, list] = {}
    for name, fn in cases_mod.cases(a.world):
        LOG.clear()
        fn()
        if name in cases:
            raise SystemExit(f"case {name} twice")
        cases[name] = list(LOG)
    want = sites(Path(glue.__file__).parent)
    hit = Counter((x["site"], x["fn"]) for launches in cases.values() for x in launches)
    rows, missing = [], []
    for s in want:
        n = hit.get((f"{s['file']}:{s['wrapper']}", s["kernel"]), 0)
        why = cases_mod.SITES_OUT.get(s["file"])
        rows.append({**s, "launches": n, **({"excluded": why} if why else {})})
        if n == 0 and not why:
            missing.append(s)
    kernels = {(s["file"], s["kernel"]) for s in want if s["file"] not in cases_mod.SITES_OUT}
    seen = {(x["site"].split(":")[0], x["fn"]) for launches in cases.values() for x in launches}
    out = {"generator": "tools/zig/glm_triton_fixtures.py", "world": a.world, "sites": rows, "cases": cases}
    a.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    total = sum(len(v) for v in cases.values())
    print(f"{len(cases)} cases, {total} launches -> {a.out}")
    print(f"{len(want)} call sites ({len(want) - len(missing) - sum(1 for r in rows if 'excluded' in r)} hit, "
          f"{sum(1 for r in rows if 'excluded' in r)} excluded, {len(missing)} missed); "
          f"{len(kernels & seen)} of {len(kernels)} launched kernels hit")
    for s in missing:
        print(f"MISSED {s['file']}:{s['line']} {s['wrapper']} -> {s['kernel']}")
    for k in sorted(kernels - seen):
        print(f"MISSED kernel {k[0]} {k[1]}")
    return 1 if missing or kernels - seen else 0


if __name__ == "__main__":
    sys.exit(main())
