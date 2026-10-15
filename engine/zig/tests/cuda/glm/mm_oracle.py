"""Dev-time oracle for GLM-5.3-Flash's affine kernel copies (zig/kernels/cuda/glm_qmm_group.cu, glm_qmm_prefill.cu,
glm_experts.cu, glm_experts_prefill.cu): python-0.6's own qmm and experts wrappers on the GPU under test.

Runs in the PyTorch image on that GPU with python-0.6's src/ on PYTHONPATH (zig/tests/cuda/glm/box/mm.sh). Each case
directory holds every extension call the wrappers made (fn and arguments; a tensor as storage, byte offset, shape and
strides) and every storage those calls touched: its bytes when first seen, and its bytes after the case if they changed.
zig/tests/cuda/glm_mm.zig (`tf-glm-mm replay`) uploads the first, replays the calls through the bindings and compares the
second. Shapes are one TP=2 rank's (hidden 4096, 32 KDA heads, MLP 6144, experts 1024 wide, 9 slots a row).
usage: mm_oracle.py OUT_DIR
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.glm5_next.cuda import qmm as glm_qmm

DEV = torch.device("cuda:0")
DTYPE = {torch.int32: 3, torch.float32: 6, torch.bfloat16: 8}
D, ML, SLOTS, LIMIT = 4096, 1024, 9, 10.0
E = 17                                   # 16 routed experts and the shared one (the kernels index any count alike)
# MLX 4-bit projections: name -> (n, k, fp32 out, x row stride, x column offset); kda_fb reads the projection's rows
MATS = {"kda_proj": (12576, 4096, False, 4096, 0), "kda_fb": (4096, 128, False, 12576, 12288),
        "kda_o": (4096, 4096, True, 4096, 0), "mlp_down": (4096, 6144, True, 6144, 0)}
DECODE, PROMPT = (1, 2, 5, 8, 16), (17, 128, 300)


class Case:
    """The calls of one case and the storages they touched (by base address: name, whole-storage bytes, first bytes)."""

    def __init__(self) -> None:
        self.calls: list = []
        self.storages: dict = {}

    def encode(self, v):
        if isinstance(v, torch.Tensor):
            st = v.untyped_storage()
            base = st.data_ptr()
            if base not in self.storages:
                raw = torch.empty(0, dtype=torch.uint8, device=v.device).set_(st)
                self.storages[base] = (f"s{len(self.storages)}", raw, raw.cpu().clone())
            return {"t": {"array": self.storages[base][0], "off": v.data_ptr() - base, "dtype": DTYPE[v.dtype],
                          "shape": list(v.shape), "stride": list(v.stride())}}
        if isinstance(v, (list, tuple)):
            return {"list": [self.encode(x) for x in v]}
        if isinstance(v, bool):
            return {"bool": v}
        if isinstance(v, int):
            return {"int": v}
        if isinstance(v, float):
            return {"float": struct.unpack("<Q", struct.pack("<d", v))[0]}
        raise TypeError(f"no encoding for {type(v)}")

    def save(self, out: Path, what: str, params: dict) -> dict:
        torch.cuda.synchronize()
        out.mkdir(parents=True, exist_ok=True)
        arrays, written = {}, 0
        for name, raw, before in self.storages.values():
            (out / f"{name}.bin").write_bytes(before.numpy().tobytes())
            entry = {"file": f"{name}.bin", "dtype": "uint8", "shape": [before.numel()]}
            after = raw.cpu()
            if not torch.equal(after, before):
                (out / f"{name}.after.bin").write_bytes(after.numpy().tobytes())
                entry["after"] = f"{name}.after.bin"
                written += 1
            arrays[name] = entry
        (out / "manifest.json").write_text(json.dumps({"params": {"what": what, **params}, "arrays": arrays,
                                                       "calls": self.calls}))
        return {"calls": len(self.calls), "storages": len(arrays), "written": written}


class Recorder:
    """One extension: each call is kept (its storages' bytes first, the GPU idle) and then made."""

    def __init__(self, ext, module: str, case: Case) -> None:
        self.ext, self.module, self.case = ext, module, case

    def __getattr__(self, fn: str):
        real = getattr(self.ext, fn)

        def call(*args):
            torch.cuda.synchronize()
            self.case.calls.append({"module": self.module, "fn": fn, "args": [self.case.encode(a) for a in args]})
            return real(*args)
        return call


def recording(case: Case):
    """Both extensions' _ext() through a Recorder until the returned function restores them."""

    qmm_ext, experts_ext = shared._ext, grouped._ext
    q, e = qmm_ext(), experts_ext()
    shared._ext = lambda: Recorder(q, "qmm", case)
    grouped._ext = lambda: Recorder(e, "experts", case)

    def restore() -> None:
        shared._ext, grouped._ext = qmm_ext, experts_ext
    return restore


def mlx(g: torch.Generator, *shape: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random MLX 4-bit arrays: words (..., n, k / 8) int32, scales and biases (..., n, k / 64) bf16."""

    *lead, n, k = shape
    words = torch.randint(-2 ** 31, 2 ** 31 - 1, (*lead, n, k // 8), dtype=torch.int32, device=DEV, generator=g)
    scales = (torch.rand(*lead, n, k // 64, device=DEV, generator=g) * 0.02 + 0.002).to(torch.bfloat16)
    biases = (torch.randn(*lead, n, k // 64, device=DEV, generator=g) * 0.01).to(torch.bfloat16)
    return words, scales, biases


def qmm_case(out: Path, g: torch.Generator, name: str) -> dict:
    """Decode windows through glm5_next qmm.matmul (qmm_group), prompt chunks through prefill_matmul, on one weight."""

    n, k, f32, stride, off = MATS[name]
    q = glm_qmm.make_q4(*mlx(g, n, k))
    case = Case()
    restore = recording(case)
    try:
        for r in DECODE + PROMPT:
            x = torch.randn(r, max(stride, off + k), device=DEV, generator=g).to(torch.bfloat16)[:, off:off + k]
            y = torch.zeros(r, n, dtype=torch.float32 if f32 else torch.bfloat16, device=DEV)
            if r in DECODE:
                glm_qmm.matmul(x, q, glm_qmm.group_sums(x), out=y, f32=f32)
            else:
                shared.prefill_matmul(x, q, f32=f32, out=y)
    finally:
        restore()
    return case.save(out / f"mm_{name}", f"qmm {name}: n {n}, k {k}, {'fp32' if f32 else 'bf16'} out, rows {DECODE} "
                     f"decode and {PROMPT} prompt", {"n": n, "k": k})


def experts_case(out: Path, g: torch.Generator, prefill: bool, rows: tuple[int, ...]) -> dict:
    """moe_block's MLX path: route (plan), gate_up (SwiGLU, limit 10), down; each row's 8 routed experts and the shared."""

    ex = grouped.make([mlx(g, E, ML, D), mlx(g, E, ML, D)], mlx(g, E, D, ML), 64, limit=LIMIT)
    case = Case()
    restore = recording(case)
    try:
        for r in rows:
            routed = [torch.randperm(E - 1, device=DEV, generator=g)[:SLOTS - 1] for _ in range(r)]
            picks = torch.stack([torch.cat([p, torch.tensor([E - 1], device=DEV)]) for p in routed]).to(torch.int32)
            plan = grouped.Plan(r, SLOTS, E, DEV, prefill=prefill)
            grouped.route(picks.contiguous(), plan)
            x = torch.randn(r, D, device=DEV, generator=g).to(torch.bfloat16)
            act = torch.zeros(r * SLOTS, ML, dtype=torch.bfloat16, device=DEV)
            grouped.gate_up(x, ex, plan, act, r)
            y = torch.zeros(r * SLOTS, D, dtype=torch.bfloat16 if prefill else torch.float32, device=DEV)
            grouped.down(act, ex, plan, y, r)
    finally:
        restore()
    form = "prompt" if prefill else "decode"
    return case.save(out / f"experts_{form}", f"experts {form}: {E} experts, {SLOTS} slots, rows {rows}",
                     {"experts": E})


def main() -> int:
    out = Path(sys.argv[1])
    g = torch.Generator(device=DEV)
    g.manual_seed(5315)
    report = {name: qmm_case(out, g, name) for name in MATS}
    report["experts_decode"] = experts_case(out, g, False, (1, 4, 8, 16))
    report["experts_prompt"] = experts_case(out, g, True, (17, 64, 113, 114, 300))
    for name, r in report.items():
        print(name, r, flush=True)
    (out / "mm_oracle.json").write_text(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(),
                                                    "capability": list(torch.cuda.get_device_capability()),
                                                    "cases": report}, indent=1))
    return 0 if all(r["calls"] and r["written"] for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
