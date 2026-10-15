"""Dev-time oracle for GLM-5.3-Flash's KDA copy (zig/kernels/cuda/glm_kda.cu): fixtures from python-0.6's kda.py.

Runs in the PyTorch image on the GPU under test, with python-0.6's src/ on PYTHONPATH (zig/tests/cuda/glm/box/kda.sh);
zig/tests/cuda/glm_kernels.zig replays every case through the copy and compares the bytes.
Shapes are a TP=2 rank's (32 of GLM's 64 KDA heads; the stacked in-projection 3 x 32 x 128 + 32 + 2 x 128 = 12,576 wide).
usage: kda_oracle.py OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from tensorfold.families.glm5_next.cuda import kda

DEV = torch.device("cuda:0")
DK, TAPS = 128, 4
EPS, LOWER = 1e-5, -5.0          # GLM-5.3-Flash's rms_norm_eps and gate_lower_bound (python-0.6 weights.Config)


def save(out: Path, params: dict, arrays: dict[str, torch.Tensor]) -> None:
    """manifest.json plus one raw little-endian file per tensor (zig/tests/cuda/fixture.zig reads it)."""

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"params": params, "arrays": {}}
    for name, t in arrays.items():
        t = t.detach().contiguous().cpu()
        (out / f"{name}.bin").write_bytes(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        manifest["arrays"][name] = {"file": f"{name}.bin", "dtype": str(t.dtype).replace("torch.", ""),
                                    "shape": list(t.shape)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


def inputs(g: torch.Generator, heads: int, rows: int) -> dict[str, torch.Tensor]:
    """One layer's window: projection rows [q | k | v | beta | decay and gate low-rank inputs]; a and g are column views
    of one bf16 buffer ``ag`` [rows, 2 x heads x 128] (a first), so their row stride is not their width."""

    c = 3 * heads * DK
    width = c + heads + 2 * 128
    r = lambda *s: torch.randn(*s, generator=g, device=DEV)            # noqa: E731
    ag = (r(rows, 2 * heads * DK) * 0.7).to(torch.bfloat16)
    return {"p": (r(rows, width) * 0.5).to(torch.bfloat16), "ag": ag, "a": ag[:, :heads * DK], "g": ag[:, heads * DK:],
            "conv_state": (r(TAPS - 1, c) * 0.5).to(torch.bfloat16), "conv_w": (r(c, TAPS) * 0.4).to(torch.bfloat16),
            "state_in": r(heads, DK, DK) * 0.05, "a_log": r(heads) * 0.5, "dt_bias": r(heads * DK) * 0.1,
            "norm_w": (1.0 + 0.1 * r(DK)).to(torch.bfloat16)}


def run_chain(x: dict, heads: int, rows: int, wide: bool) -> tuple[kda.KDAScratch, torch.Tensor]:
    scratch = kda.KDAScratch(rows, heads, DEV)
    state_out = torch.empty_like(x["state_in"])
    kda.chain(x["p"], 3 * heads * DK, x["a"], x["g"], x["conv_state"], x["conv_w"], x["state_in"], x["a_log"],
              x["dt_bias"], x["norm_w"], EPS, LOWER, rows, scratch, state_out, wide=wide)
    torch.cuda.synchronize()
    return scratch, state_out


def chain_case(out: Path, g: torch.Generator, heads: int, rows: int, wide: bool) -> dict:
    x = inputs(g, heads, rows)
    s, state_out = run_chain(x, heads, rows, wide)
    s2, state2 = run_chain(x, heads, rows, not wide)                  # kda.py: both paths give the same bits
    same = bool(torch.equal(s.out, s2.out) and torch.equal(state_out, state2) and torch.equal(s.k, s2.k)
                and torch.equal(s.v, s2.v) and torch.equal(s.g, s2.g) and torch.equal(s.b, s2.b))
    arrays = {k: v for k, v in x.items() if k not in ("a", "g")}
    arrays.update(out=s.out, state_out=state_out, k_save=s.k, v_save=s.v, g_save=s.g, b_save=s.b)
    save(out, {"op": "chain", "wide": int(wide), "heads": heads, "rows": rows, "p_stride": x["p"].stride(0),
               "b_off": 3 * heads * DK, "ag_stride": x["ag"].stride(0), "g_column": heads * DK, "eps": EPS,
               "lower": LOWER}, arrays)
    return {"narrow_equals_wide": same}


def replay_case(out: Path, g: torch.Generator, heads: int, rows: int, keep: int) -> dict:
    x = inputs(g, heads, rows)
    s, _ = run_chain(x, heads, rows, False)
    state_out = torch.empty_like(x["state_in"])
    kda.replay(x["state_in"], s, keep, state_out)
    torch.cuda.synchronize()
    save(out, {"op": "replay", "heads": heads, "rows": keep},
         {"state_in": x["state_in"], "k_save": s.k, "v_save": s.v, "g_save": s.g, "b_save": s.b, "state_out": state_out})
    return {}


def layers_case(out: Path, g: torch.Generator, heads: int, rows: int, layers: int, keep: int) -> dict:
    ss = kda.KDAScratchSet(layers, rows, heads, DEV)
    states = torch.empty(layers, heads, DK, DK, device=DEV)
    for i in range(layers):
        x = inputs(g, heads, rows)
        states[i] = x["state_in"]
        kda.chain(x["p"], 3 * heads * DK, x["a"], x["g"], x["conv_state"], x["conv_w"], x["state_in"], x["a_log"],
                  x["dt_bias"], x["norm_w"], EPS, LOWER, rows, ss.views[i], torch.empty_like(x["state_in"]), wide=False)
    state_out = torch.empty_like(states)
    kda.replay_layers(states, ss, keep, state_out)
    torch.cuda.synchronize()
    save(out, {"op": "layers", "heads": heads, "rows": keep, "layers": layers, "window": rows},
         {"state_in": states, "k_save": ss.k, "v_save": ss.v, "g_save": ss.g, "b_save": ss.b, "state_out": state_out})
    return {}


def main() -> int:
    out = Path(sys.argv[1])
    g = torch.Generator(device=DEV)
    g.manual_seed(5300)
    cases = {
        "kda_r1": lambda o: chain_case(o, g, 32, 1, False),          # one decode row
        "kda_r16": lambda o: chain_case(o, g, 32, 16, False),        # a drafted window
        "kda_r63": lambda o: chain_case(o, g, 32, 63, False),        # the longest one-kernel window
        "kda_r64": lambda o: chain_case(o, g, 32, 64, True),         # the first three-kernel window
        "kda_r80": lambda o: chain_case(o, g, 32, 80, True),         # a part pass of step_kernel's 16-row staging
        "kda_h64_r5": lambda o: chain_case(o, g, 64, 5, False),      # one GPU's 64 heads
        "kda_replay": lambda o: replay_case(o, g, 32, 16, 5),        # a kept prefix of a window
        "kda_layers": lambda o: layers_case(o, g, 32, 16, 3, 7),     # every layer's prefix at once
    }
    report = {}
    for name, fn in cases.items():
        report[name] = fn(out / name)
        print(name, report[name], flush=True)
    (out / "kda_oracle.json").write_text(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(),
                                                     "capability": list(torch.cuda.get_device_capability()),
                                                     "cases": report}, indent=1))
    bad = [n for n, r in report.items() if r.get("narrow_equals_wide") is False]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
