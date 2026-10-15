"""G-ident's oracle at the 3-node attention's world-1 shapes (qualification_18588626), for `tf-glm-mm replay` (zig/tests/cuda/glm_mm.zig):
python-0.6's glm5_next qmm.matmul on the GPU under test (attention-host's RTX 5090: its qmm_group dispatch takes tile 7 to 8
rows and tile 8 to 16), through a prebuilt tensorfold_qmm_v5 extension (--ext, built on a CPU box for the GPU's SASS
target with python-0.6's own flags), so nothing is compiled on the lab host. Each projection of --shapes
(tools/zig/glm_mm_w1_shapes.json: n, k, x row stride, fp32 out; the kits' q4 projections and the q4 draft head) is one
case directory in mm_oracle.py's format, at windows of 1-8 rows (decode, c7's verify windows) and 9 and 16 (tile 8).
The group sums are torch's (python's are a Triton kernel, JIT-compiled): the replay reads the recorded bytes, so the
qmm_group launches see the same inputs on both sides.
usage: mm_oracle_w1.py --ext tensorfold_qmm_v5.so --shapes glm_mm_w1_shapes.json [--plan] OUT_DIR
  --plan  print the cases and exit (no GPU needed)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mm_oracle as base  # noqa: E402  Case, Recorder, mlx, DEV; importing builds nothing

from tensorfold.cuda.kernels import qmm as shared  # noqa: E402
from tensorfold.families.glm5_next.cuda import qmm as glm_qmm  # noqa: E402

WINDOWS = tuple(range(1, 9)) + (9, 16)


def projections(shapes: list[dict]) -> list[tuple[int, int, int, bool]]:
    """Each (n, k, x row stride, fp32 out) once, in the order the shapes file first gives it."""

    seen: dict = {}
    for s in shapes:
        seen.setdefault((s["n"], s["k"], s["x_stride"], s["f32"]), None)
    return list(seen)


def name_of(p: tuple[int, int, int, bool]) -> str:
    n, k, stride, f32 = p
    return f"w1_n{n}_k{k}_x{stride}_{'f32' if f32 else 'bf16'}"


def load_ext(path: Path):
    """The prebuilt extension, imported as torch's cpp_extension.load imports the module it builds."""

    spec = importlib.util.spec_from_file_location("tensorfold_qmm_v5", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: not an importable extension")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def case(out: Path, g: torch.Generator, ext, p: tuple[int, int, int, bool]) -> dict:
    """One projection's windows through glm5_next qmm.matmul (python's split_k and group_tile), every call recorded."""

    n, k, stride, f32 = p
    q = glm_qmm.make_q4(*base.mlx(g, n, k))
    c = base.Case()
    rec = base.Recorder(ext, "qmm", c)
    shared._ext = lambda: rec
    try:
        for m in WINDOWS:
            x = torch.randn(m, stride, device=base.DEV, generator=g).to(torch.bfloat16)[:, :k]
            xs = x.float().reshape(m, k // 64, 64).sum(-1)
            y = torch.zeros(m, n, dtype=torch.float32 if f32 else torch.bfloat16, device=base.DEV)
            glm_qmm.matmul(x, q, xs, out=y, f32=f32)
    finally:
        shared._ext = lambda: ext
    fns = [call["fn"] for call in c.calls]
    if fns != ["qmm_group"] * len(WINDOWS):
        raise SystemExit(f"{name_of(p)}: python made {fns}, not {len(WINDOWS)} qmm_group calls")
    tiles = [call["args"][9]["int"] for call in c.calls]
    r = c.save(out / name_of(p), f"qmm_group world 1: n {n}, k {k}, x stride {stride}, {'fp32' if f32 else 'bf16'} "
               f"out, rows {WINDOWS}", {"n": n, "k": k})
    return {**r, "sk": glm_qmm.split_k(n, k), "tiles": tiles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", type=Path, required=True)
    ap.add_argument("--shapes", type=Path, required=True)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    projs = projections(json.loads(a.shapes.read_text())["shapes"])
    ext_sha = hashlib.sha256(a.ext.read_bytes()).hexdigest()
    print(f"{len(projs)} projections x windows {WINDOWS}; ext {a.ext} sha256 {ext_sha}; torch {torch.__version__}",
          flush=True)
    for p in projs:
        print(f"  {name_of(p)}: K slices {glm_qmm.split_k(p[0], p[1])}", flush=True)
    ext = load_ext(a.ext)  # the import must work on any box with this image (no GPU needed)
    print(f"ext loaded: {ext.__name__} qmm_group={hasattr(ext, 'qmm_group')}", flush=True)
    if a.plan:
        return 0
    shared._ext = lambda: ext  # python-0.6's wrappers take the prebuilt extension: no JIT build
    prop = torch.cuda.get_device_properties(0)
    chip = [prop.major, prop.minor, prop.multi_processor_count]
    print(f"GPU {prop.name}: {chip[0]}.{chip[1]}, {chip[2]} SMs; python's qmm_group tile pick at 1-16 rows: "
          f"{[shared.group_tile(m, *chip) for m in WINDOWS]} (0: qmm_group.cu's own, 7 to 8 rows and 8 to 16 off a GB10)",
          flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=base.DEV)
    g.manual_seed(0x2F4)
    report = {}
    for p in projs:
        report[name_of(p)] = case(a.out, g, ext, p)
        torch.cuda.empty_cache()
        print(name_of(p), report[name_of(p)], flush=True)
    (a.out / "mm_oracle_w1.json").write_text(json.dumps({
        "torch": torch.__version__, "device": prop.name, "chip": chip, "ext": str(a.ext), "ext_sha256": ext_sha,
        "shapes": str(a.shapes), "shapes_sha256": hashlib.sha256(a.shapes.read_bytes()).hexdigest(),
        "windows": WINDOWS, "cases": report}, indent=1))
    return 0 if all(r["calls"] == len(WINDOWS) and r["written"] for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
