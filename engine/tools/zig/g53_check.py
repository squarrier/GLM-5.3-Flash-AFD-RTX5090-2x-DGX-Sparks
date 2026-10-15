"""qualification_1b9bf461's G-bits reference: python TF-AFD 2.15's g53 + KT3R on one MoE layer's TP2 half of an EXL3 checkpoint.

As 2.15's expert node runs a prompt window under its shipped .env.example (TF_GLM_EXPERT_KERNEL=g53 with
TF_GLM_EXPERT_KERNEL_MT=4, _GW=16, _NT=4, _L2=1 and _TIER_ROWS=1536; rowrot on, order off): g53_experts.routed at
tuned_cfg(R, fields=KT3, tier=1536) on layer L's half (experts_node.load_half, so share_suh is applied), at the R list of
2.15's BENCHMARKS §Quality, on seeded rows and routes (the first R of 4,096). Writes OUT/manifest.json, in/x.bin (bf16
[4096, 4096]), in/pick.bin (int32 [4096, 9]), in/wts.bin (fp32 [4096, 9]) and out/R.bin (fp32 [R, 4096]), each listed
with its sha256; tf-g53-check (zig/tests/cuda/glm_g53_check.zig) runs the Zig port on the same inputs and compares
every byte. The kernels are Hugh Madden's (@dangerm00se), glm53f-afd @ 91db3cc (MIT); KT3's schedule is glm53f-rank's.

  --dry        the inputs and each window's configuration only (no GPU, no weights): tf-g53-check --dry's manifest
  --build-ext  build 2.15's g53 extension into TORCH_EXTENSIONS_DIR and stop (no checkpoint read)
  --bench N    each window's call N times after its reference (median, p10, p90 us in the manifest)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROWS = [1, 2, 7, 16, 33, 64, 65, 128, 594, 2048, 2049, 4096]   # 2.15's BENCHMARKS §Quality
KT3 = {"mt": 4, "gw": 16, "nt": 4, "l2": 1}                      # 2.15's .env.example knobs (patch 0049)
TIER = 1536                                                       # TF_GLM_EXPERT_KERNEL_TIER_ROWS (patch 0051)
H, SLOTS, E, MAXR = 4096, 9, 288, 4096
G53 = "src/tensorfold/families/glm5_next/cuda/"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(p: Path) -> str:
    raw = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()


def inputs(torch, seed: int):
    """x bf16 [4096, 4096] ~ N(0, 1); pick: 8 distinct experts a row, the shared slot 288 (not read by the kernels);
    wts: positive, summing to 2.5 a row over the 8 routed slots, the shared slot 1.0. CPU generator, so any host
    gives the same bytes."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn((MAXR, H), generator=g, dtype=torch.float32).to(torch.bfloat16)
    pick = torch.full((MAXR, SLOTS), E, dtype=torch.int32)
    for r in range(MAXR):
        pick[r, :8] = torch.randperm(E, generator=g)[:8].to(torch.int32)
    w = torch.rand((MAXR, 8), generator=g, dtype=torch.float32) + 0.05
    wts = torch.ones((MAXR, SLOTS), dtype=torch.float32)
    wts[:, :8] = w / w.sum(dim=1, keepdim=True) * 2.5
    return x, pick, wts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("--tree", required=True, help="a python TF-AFD 2.15 checkout")
    ap.add_argument("--model", help="Mia's EXL3 checkpoint folder")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--out", help="the reference folder (created)")
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--seed", type=int, default=5302)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--build-ext", action="store_true")
    a = ap.parse_args(argv)
    tree = Path(a.tree)
    sys.path.insert(0, str(tree / "src"))
    import torch
    from tensorfold.families.glm5_next.cuda import g53_experts as g53

    if a.build_ext:
        t0 = time.time()
        g53._ext()
        print(f"g53 extension built in {time.time() - t0:.1f} s", flush=True)
        return 0
    if not a.out or (not a.dry and not a.model):
        ap.error("--out (and --model unless --dry) are needed")
    rows = [int(r) for r in a.rows.split(",")]
    if not all(1 <= r <= MAXR for r in rows):
        ap.error(f"rows are 1..{MAXR}")
    out = Path(a.out)
    (out / "in").mkdir(parents=True, exist_ok=False)
    (out / "out").mkdir()
    x, pick, wts = inputs(torch, a.seed)
    man: dict = {"tool": "tools/zig/g53_check.py", "card": "qualification_1b9bf461", "layer": a.layer, "rank": a.rank, "seed": a.seed,
                 "dry": int(a.dry), "kt3": KT3, "tier": TIER, "rowrot": True, "order": False, "inputs": {}, "cases": []}
    try:
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    man["python"] = {"tree_head": head, "g53_rank.cu": blob(tree / G53 / "g53_rank.cu"),
                     "g53_rank.cpp": blob(tree / G53 / "g53_rank.cpp"), "g53_experts.py": blob(tree / G53 / "g53_experts.py")}
    arrays = {"x": ("bf16", [MAXR, H], x.view(torch.int16).numpy().tobytes()),
              "pick": ("i32", [MAXR, SLOTS], pick.numpy().tobytes()),
              "wts": ("f32", [MAXR, SLOTS], wts.numpy().tobytes())}
    for k, (dt, shape, b) in arrays.items():
        (out / "in" / f"{k}.bin").write_bytes(b)
        man["inputs"][k] = {"file": f"in/{k}.bin", "dtype": dt, "shape": shape, "sha256": sha(b)}
    for r in rows:
        man["cases"].append({"rows": r, "cfg": dataclasses.asdict(g53.tuned_cfg(r, rowrot=True, order=False,
                                                                               fields=KT3, tier=TIER))})
    if a.dry:
        (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"DRY g53_check: {len(rows)} windows' configurations and the inputs (seed {a.seed}) in {out}")
        return 0

    from tensorfold.families.glm5_next.cuda.experts_node import load_half
    from tensorfold.families.glm5_next.cuda.weights import Config

    limit = float(Config.read(a.model).limit)
    t0 = time.time()
    ex = load_half(a.model, a.rank, [a.layer])[0]
    why = g53.layer_ready(ex)
    if why is not None:
        print(f"FAIL g53_check: layer {a.layer} rank {a.rank} is out of the kernels' scope: {why}")
        return 1
    props = torch.cuda.get_device_properties(0)
    man.update({"limit": limit, "shared_suh": ex.shared_suh is not None, "device": props.name,
                "sms": props.multi_processor_count, "capability": f"{props.major}{props.minor}",
                "torch": torch.__version__, "cuda": torch.version.cuda})
    print(f"REF g53_check: layer {a.layer} rank {a.rank}/2 loaded in {time.time() - t0:.1f} s on {props.name}; shared "
          f"gate/up suh {ex.shared_suh is not None}; limit {limit}", flush=True)
    s = g53.Scratch(max(rows), torch.device("cuda"))
    xd, pd, wd = x.cuda(), pick.cuda(), wts.cuda()
    o = torch.empty((max(rows), H), dtype=torch.float32, device="cuda")
    for case in man["cases"]:
        r = case["rows"]
        c = g53.tuned_cfg(r, rowrot=True, order=False, fields=KT3, tier=TIER)
        o.view(torch.int32).fill_(-1)          # NaN in every word: an unwritten one shows in the comparison
        g53.routed(xd, pd, wd, ex, s, o, r, limit, c)
        torch.cuda.synchronize()
        case["fault"] = s.fault()
        b = o[:r].contiguous().cpu().numpy().tobytes()
        (out / "out" / f"{r}.bin").write_bytes(b)
        case["out"] = {"file": f"out/{r}.bin", "sha256": sha(b)}
        if a.bench > 0:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(a.bench + 1)]
            for i in range(a.bench):
                ev[i].record()
                g53.routed(xd, pd, wd, ex, s, o, r, limit, c)
            ev[a.bench].record()
            torch.cuda.synchronize()
            us = sorted(ev[i].elapsed_time(ev[i + 1]) * 1000 for i in range(a.bench))
            n = a.bench
            case["bench_us"] = [us[n // 2], us[n // 10], us[n - 1 - n // 10]]
        print(f"REF g53 r{r}: mt {c.mt} gw {c.gw} nt {c.nt} l2 {c.l2} rot {c.rot if ex.shared_suh is not None else 1}; "
              f"fault {case['fault']}; out sha256 {case['out']['sha256'][:16]}"
              + (f"; median {case['bench_us'][0]:.1f} us" if "bench_us" in case else ""), flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    bad = sum(1 for c in man["cases"] if c["fault"])
    print(f"{'FAIL' if bad else 'DONE'} g53_check: {len(rows)} windows, {bad} with fault bits; {out}/manifest.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
