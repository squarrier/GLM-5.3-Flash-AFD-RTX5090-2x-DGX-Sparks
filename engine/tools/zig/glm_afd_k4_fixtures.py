#!/usr/bin/env python3
"""K4 on GLM-5.3-Flash's expert halves as python 2.15 runs it (TF_GLM_EXL3_PROMPT=1), recorded on a CPU (qualification_e4ad04cd)."""
# glm_afd_fixtures.py's expert role (qualification_955b7191's G-trace-E: python 2.15 src@0849f3f over qualification_765330ce's header-only mirror of Mia's
# EXL3 078455ff, its node and recorders as they are), each window a prompt forward's (node.prompt, which
# experts_node.forwards sets for every mode but eager) with TF_GLM_EXL3_PROMPT=1: her prompt_switches at their
# defaults (pass 64, order and rowrot on) at 1, 8, 64, 594 and 2,048 rows, then each varied (TF_GLM_EXL3_PASS=128,
# TF_GLM_EXL3_ORDER=0, TF_GLM_EXL3_ROWROT=0) at 8 and 2,048. The scratches' prompt buffers (_prompt_buffers, made on
# a scratch's first prompt window) are made first, so they carry names (b.decode.prompt_bufs.0-2, b.pref.*).

from __future__ import annotations

import argparse
import gc
import json
import os
import types
from pathlib import Path

import torch

import glm_afd_fixtures as af
import glm_forward_fixtures as gf
from tensorfold.cuda.exl3 import experts as generic
from tensorfold.families.glm5_next.cuda import mcdma
from tensorfold.families.glm5_next.cuda import weights as wmod

ROWS = (1, 8, 64, 594, 2048)
PROMPT = {"TF_GLM_EXL3_PROMPT": "1"}
VARIANTS = {"p128": {"TF_GLM_EXL3_PASS": "128"}, "order0": {"TF_GLM_EXL3_ORDER": "0"},
            "rowrot0": {"TF_GLM_EXL3_ROWROT": "0"}}
VARIANT_ROWS = (8, 2048)


def case(node, link, layer: int, rows: int, env: dict) -> dict:
    """af.expert_case with the window a prompt forward's at any row count, under `env`."""
    os.environ.update(env)
    try:
        lay = mcdma.Layout(node.cfg.hidden, node.cfg.top_k + 1)
        x = lay.views(link.landing, rows)
        node.prompt = True
        af.expert_names(node, link)
        gf.LOG.clear()
        node.compute(layer, rows, x)
    finally:
        for k in env:
            del os.environ[k]
    out = {"layer": layer, "rows": rows, "half": node.half, "prompt": True, "env": env, "events": list(gf.LOG)}
    if gf.NAMES.tmp:
        out["tmp"] = dict(gf.NAMES.tmp.values())
    return out


def expert(mirror: Path, out: Path) -> int:
    env = af.env_check()
    gf.patch()
    cfg = wmod.Config.read(mirror)
    cases, buffers, ready = {}, None, {}
    for half in (0, 1):
        node = af.Node(mirror, half, af.EXPERT_LAYERS, 8, 2048)
        lay = mcdma.Layout(cfg.hidden, cfg.top_k + 1)
        link = types.SimpleNamespace(landing=torch.empty((lay.req(2048) - mcdma.HDR_BYTES,), dtype=torch.uint8))
        counts = {ex.count for ex in node.by_layer.values()}
        if len(counts) != 1:
            raise SystemExit(f"the layers' expert counts differ: {counts}")
        (E,) = counts
        for s in (node.buf.decode, node.buf.pref):              # python makes them on the first prompt window
            generic._prompt_buffers(s, E)
        for layer in af.EXPERT_LAYERS:
            ready[f"h{half}-l{layer}"] = generic.prompt_kernel_ready(node.by_layer[layer], node.buf.pref.slots)
            for rows in ROWS:
                cases[f"h{half}-l{layer}-r{rows}"] = case(node, link, layer, rows, dict(PROMPT))
            for name, extra in VARIANTS.items():
                for rows in VARIANT_ROWS:
                    cases[f"h{half}-l{layer}-r{rows}-{name}"] = case(node, link, layer, rows, {**PROMPT, **extra})
        af.expert_names(node, link)
        mine = gf.NAMES.buffers()
        if buffers is not None and mine != buffers:
            raise SystemExit("half 1's buffers differ from half 0's in path, dtype or shape")
        buffers = mine
        del node, link
        gf.NAMES.by_ptr.clear()
        gc.collect()
    doc = {"generator": "tools/zig/glm_afd_k4_fixtures.py", "schema": "glm-forward-trace/1", "python": af.PYTHON,
           "role": "expert", "env": env, "world": 2, "halves": [0, 1], "layers": list(af.EXPERT_LAYERS), "max_rows": 8,
           "prefill_rows": 2048, "config": gf.config_dict(cfg), "buffers": buffers, "ready": ready, "cases": cases}
    return af.write(out, doc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", required=True, type=Path, help="the EXL3 checkpoint (a header-only sparse mirror serves)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    return expert(a.mirror, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
