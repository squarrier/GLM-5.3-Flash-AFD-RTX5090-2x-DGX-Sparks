#!/usr/bin/env python3
"""EXL3 launches as python-0.6's wrappers make them, recorded on a CPU: the fixtures of the Zig bindings' launches.

Both halves are python-0.6's own code (ed78d6f):
1. the Python wrappers (cuda/exl3/experts.py routed and dequant, cuda/exl3/linear.py Exl3Linear, GLM's
   glm5_next/cuda/exl3_mm.py routed) run on CPU tensors, each extension replaced by a recorder of the calls made into it;
2. the extensions' C++ wrappers (experts.cpp/.cu + experts_cb0-2.cu, linear.cpp/.cu, GLM's exl3.cpp/.cu), compiled from
   the same tree against tools/zig/exl3_launches (a torch and CUDA runtime stand-in), run those calls; each launch they
   make comes out with its kernel, grid, block, shared bytes, attribute and argument bytes (sizes from the cubins).
Addresses are stand-ins (one 4 GiB-aligned base a buffer). The output, zig/src/families/glm/fixtures_cuda_exl3.json,
holds each case's inputs and the launches of each step; cuda_exl3_test.zig replays them through the Zig bindings.
usage (PyTorch 26.07 image, python-0.6 at /tensorfold, this tree at /fork): python exl3_fixtures.py OUT.json WORKDIR
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.cuda.exl3 import linear as lin
from tensorfold.families.glm5_next.cuda import exl3_mm

PY = Path("/tensorfold/src/tensorfold")
FORK = Path("/fork")
MODULES = {"experts": "tensorfold_exl3_experts_v1", "linear": "tensorfold_exl3_linear_v3", "glm": "tensorfold_glm_exl3_v1"}
IMAGES = {"tensorfold_exl3_experts_v1": ["exl3_experts", "exl3_grouped_cb0", "exl3_grouped_cb1", "exl3_grouped_cb2"],
          "tensorfold_exl3_linear_v3": ["exl3_linear"], "tensorfold_glm_exl3_v1": ["glm_exl3"]}
DTYPES = {torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3, torch.int64: 4, torch.float16: 5,
          torch.float32: 6, torch.float64: 7, torch.bfloat16: 8}
OPTIN = 101376      # sharedMemPerBlockOptin of GB10 (sm_121) and the RTX 5090 (sm_120): 99 KiB


class Big:
    """A large CUDA tensor of which the wrappers read only the shape, dtype and address (a trellis, a stacked layer)."""

    def __init__(self, shape, dtype, addr):
        self.shape, self.dtype, self.addr = torch.Size(shape), dtype, addr
        self.device = SimpleNamespace(type="cuda")

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self.addr

    def stride(self, d=None):
        st, acc = [], 1
        for s in reversed(self.shape):
            st.append(acc)
            acc *= s
        st = st[::-1]
        return st if d is None else st[d]

    def size(self, d):
        return self.shape[d]


class Addresses:
    """Stand-in device addresses: each storage gets the next 4 GiB-aligned base; views keep their byte offsets. Every
    tensor given one stays alive, so no later storage can reuse its host address (the stand-ins are deterministic)."""

    def __init__(self):
        self.bases: dict[int, int] = {}
        self.keep: list = []
        self.next = 1 << 40

    def fresh(self) -> int:
        a, self.next = self.next, self.next + (1 << 32)
        return a

    def of(self, t) -> int:
        if isinstance(t, Big):
            return t.addr
        self.keep.append(t)
        s = t.untyped_storage().data_ptr()
        if s not in self.bases:
            self.bases[s] = self.fresh()
        return self.bases[s] + (t.data_ptr() - s)


ADDR = Addresses()
CALLS: list[tuple] = []       # (module, fn, described args) of the current step


def describe(a):
    if isinstance(a, (torch.Tensor, Big)):
        st = a.stride() if isinstance(a, Big) else list(a.stride())
        return ("T", ADDR.of(a), DTYPES[a.dtype], list(a.shape), list(st))
    if a is None:
        return ("N",)
    if isinstance(a, bool):
        return ("I", int(a))
    if isinstance(a, int):
        return ("I", a)
    if isinstance(a, float):
        return ("F", struct.unpack("<Q", struct.pack("<d", a))[0])
    raise TypeError(f"argument {a!r}")


class Ext:
    """Stands in for a compiled extension: each call is recorded, none is run."""

    def __init__(self, module: str):
        self.module = module

    def __getattr__(self, fn: str):
        def call(*args):
            CALLS.append((self.module, fn, [describe(a) for a in args]))
        return call


ex3._ext = lambda: Ext(MODULES["experts"])
lin._ext = lambda: Ext(MODULES["linear"])
exl3_mm._ext = lambda: Ext(MODULES["glm"])

CASES: list[dict] = []        # id, kind, inputs, steps [{args, calls}]


def case(cid: str, kind: str, inputs: dict) -> dict:
    c = {"id": cid, "kind": kind, "inputs": inputs, "steps": []}
    CASES.append(c)
    return c


def step(c: dict, args: dict, fn) -> None:
    CALLS.clear()
    fn()
    c["steps"].append({"args": args, "calls": list(CALLS)})


def a(t) -> int:
    return ADDR.of(t)


def f32(v: float) -> int:
    """A float as the kernels take it, (float)v, by its bits (JSON has no infinity)."""
    return struct.unpack("<I", struct.pack("<f", v))[0]


def scales(e: int, n: int) -> list:
    return [torch.zeros((n,), dtype=torch.float16) for _ in range(e)]


def generic_layer(E: int, D: int, I: int, cb: int, k2s: tuple[list, list, list]):
    """experts.prepare on stand-in trellises (int16 [K/16, N/16, 8 * k2] each) and real fp16 scales."""

    def mats(k, n, widths, rows_s, cols_s):
        return [(Big((k // 16, n // 16, 8 * w), torch.int16, ADDR.fresh()), s, v)
                for w, s, v in zip(widths, rows_s, cols_s)]

    g = mats(D, I, k2s[0], scales(E, D), scales(E, I))
    u = mats(D, I, k2s[1], scales(E, D), scales(E, I))
    d = mats(I, D, k2s[2], scales(E, I), scales(E, D))
    return ex3.prepare(g, u, d, cb, device="cpu")


def layer_inputs(ex) -> dict:
    return {"gate_ptr": a(ex.gate_ptr), "up_ptr": a(ex.up_ptr), "down_ptr": a(ex.down_ptr), "gate_k2": a(ex.gate_k2),
            "up_k2": a(ex.up_k2), "down_k2": a(ex.down_k2), "suh_g": a(ex.suh_g), "suh_u": a(ex.suh_u),
            "svh_g": a(ex.svh_g), "svh_u": a(ex.svh_u), "suh_d": a(ex.suh_d), "svh_d": a(ex.svh_d),
            "count": ex.count, "dims": ex.dims, "width": ex.width, "cb": ex.cb, "k2_gu": list(ex.k2_gu),
            "k2_d": list(ex.k2_d)}


def scratch_inputs(s) -> dict:
    return {"xg": a(s.xg), "xu": a(s.xu), "xd": a(s.xd), "z": a(s.z), "y": a(s.y), "ids": a(s.ids),
            "count": a(s.count), "members": a(s.members_buf), "rows": s.rows, "slots": s.slots,
            "experts": s.count_experts, "cfg_gu": list(s.cfg_gu), "cfg_d": list(s.cfg_d)}


def routed_case(cid, E, D, I, cb, k2s, rows, slots, steps, cfg_gu=None, cfg_d=None, x_dtype=torch.bfloat16,
                weighted=False, act=ex3.ACT_BF16, limit=math.inf):
    """GLM's EXL3 route (exl3_generic.routed: weights None, act bf16) or the weighted one; steps (R, group) share a
    process, so a raised shared-memory limit stays raised."""

    ex = generic_layer(E, D, I, cb, k2s)
    s = ex3.Scratch(ex, rows, slots, cfg_gu, cfg_d, device="cpu")
    x = torch.zeros((rows, D), dtype=x_dtype)
    pick = torch.zeros((rows, slots), dtype=torch.int32)
    wts = torch.zeros((rows, slots), dtype=torch.float32) if weighted else None
    out = torch.zeros((rows, D), dtype=torch.float32) if weighted else None
    c = case(cid, "routed", {"layer": layer_inputs(ex), "scratch": scratch_inputs(s), "x": a(x), "x_stride": D,
                             "x_bf16": x_dtype == torch.bfloat16, "pick": a(pick), "wts": a(wts) if weighted else None,
                             "out": a(out) if weighted else None, "act": act, "limit": f32(limit), "optin": OPTIN})
    for R, group in steps:
        step(c, {"R": R, "group": group},
             lambda: ex3.routed(x[:R], pick, wts[:R] if weighted else None, ex, s, out[:R] if weighted else None, R,
                                limit, act, group))


def routed_cases():
    g8 = ([8] * 288, [8] * 288, [8] * 288)
    # GLM-5.3-Flash on a TP2 rank (I = 1024): decode windows and prompt chunks, GLM's own path
    routed_case("glm-tp2-decode", 288, 4096, 1024, 1, g8, 8, 9, [(r, True) for r in (1, 2, 3, 4, 5, 8)])
    routed_case("glm-tp2-prompt", 288, 4096, 1024, 1, g8, 2048, 9,
                [(r, True) for r in (9, 16, 17, 64, 127, 128, 256, 512, 1024, 1361, 1362, 1536, 2048, 1362)])
    routed_case("glm-tp1-decode", 288, 4096, 2048, 1, g8, 8, 9, [(r, True) for r in (1, 2, 8)])
    routed_case("glm-tp2-regroup", 288, 4096, 1024, 1, g8, 8, 9, [(4, True), (4, False)])
    routed_case("glm-tp2-weighted", 288, 4096, 1024, 1, g8, 8, 9, [(1, True), (8, True)], weighted=True,
                act=ex3.ACT_F32)
    routed_case("glm-tp2-fp16-limit", 288, 4096, 1024, 1, g8, 8, 9, [(3, True)], x_dtype=torch.float16, limit=7.0)
    # every grouped_kernel instance: 3 codebooks x 3 tile settings x 3 width ranges, on small mixed layers
    tiles = [(8, 4, 1, 1), (8, 4, 2, 2), (4, 4, 2, 2)]
    ranges = [[8, 8, 8, 8], [2, 4, 6, 10], [3, 8, 16, 12]]
    for cb in range(3):
        for ti in range(3):
            for ri in range(3):
                k2s = (ranges[ri], ranges[ri][::-1], ranges[(ri + 1) % 3])
                routed_case(f"mixed-cb{cb}-t{ti}-r{ri}", 4, 1024, 512, cb, k2s, 4, 5, [(1, True), (4, True)],
                            cfg_gu=tiles[ti], cfg_d=tiles[(ti + 1) % 3], weighted=ri == 1)


def dequant_cases():
    for cb in range(3):
        for k2 in range(2, 17):
            t = torch.zeros((4, 8, 8 * k2), dtype=torch.int16)
            c = case(f"dequant-cb{cb}-k{k2}", "dequant", {"trellis": a(t), "K": 64, "N": 128, "k2": k2, "cb": cb})
            step(c, {}, lambda: ex3.dequant(t, cb))
            c["inputs"]["out"] = c["steps"][0]["calls"][0][2][1][1]   # dequant allocates its output


def combine_cases():
    for rows, D, slots in ((1, 4096, 9), (8, 4096, 9), (3, 1000, 5)):
        y = torch.zeros((rows * slots, D), dtype=torch.float32)
        w = torch.zeros((rows, slots), dtype=torch.float32)
        o = torch.zeros((rows, D), dtype=torch.float32)
        c = case(f"combine-{rows}-{D}-{slots}", "combine", {"y": a(y), "wts": a(w), "out": a(o), "rows": rows,
                                                            "D": D, "slots": slots})
        step(c, {}, lambda: ex3._ext().combine(y, w, o, rows, D, slots))


def linear_cases():
    shapes = [(4096, 4096), (4096, 12288), (2048, 1024), (128, 128), (1024, 2048)]
    kinds = [("mcg", 4), ("3inst", 2), ("mul1", 1.5), ("mul1", 8), ("mcg", 1), ("3inst", 5), ("mul1", 3.5)]
    xs = [torch.bfloat16, torch.float16, torch.float32]
    n = 0
    for si, (K, N) in enumerate(shapes):
        for ki, (cbn, bits) in enumerate(kinds):
            if (si + ki) % 2 and si != 0:
                continue
            layout = "stored" if (si + ki) % 3 == 2 else "strips"
            trellis = torch.zeros((K // 16, N // 16, int(16 * bits)), dtype=torch.int16)
            bias = torch.zeros((N,), dtype=torch.float16) if ki % 2 else None
            L = lin.Exl3Linear.from_tensors(trellis, torch.zeros((K,), dtype=torch.float16),
                                            torch.zeros((N,), dtype=torch.float16), cbn, bias, "cpu", layout)
            if ki == 6:
                L.split = (2, 2)       # a split the plan never picks: the two-warp instances
            counters = L.counters
            inputs = {"words": a(L.words), "suh": a(L.suh), "svh": a(L.svh), "bias": a(L.bias) if bias is not None
                      else None, "counters": a(counters), "k": K, "n": N, "k2": L.k2, "cb": lin.CODEBOOK_IDS[cbn],
                      "layout": layout, "split": list(L.split)}
            c = case(f"linear-{K}x{N}-{cbn}-{bits}-{layout}", "linear", inputs)
            for M in (1, 3, 8, 9, 64, 128):
                xd, yd = xs[(n + M) % 3], xs[(n + M + 1) % 3]
                n += 1
                x = torch.zeros((M, K), dtype=xd)
                y = torch.zeros((M, N), dtype=yd)
                xh = torch.zeros((M, K), dtype=torch.float16)
                z = torch.zeros((L.split[0] * M * N,), dtype=torch.float32)
                step(c, {"M": M, "x": a(x), "x_dtype": DTYPES[xd], "y": a(y), "y_dtype": DTYPES[yd], "xh": a(xh),
                         "z": a(z)}, lambda: L(x, out=y, xh=xh, z=z))
            w = torch.zeros((K, N), dtype=torch.float16)
            step(c, {"unpack": a(w)}, lambda: L.unpack(out=w))
    # every linear_kernel and unpack_kernel instance: each width and codebook TF_EXL3_ALL compiles, at 2, 4 and 8 warps
    widths = [(k2, cb) for cb in range(3) for k2 in range(2, 17, 2)] + [(3, 2), (5, 2), (7, 2)]
    names = {v: k for k, v in lin.CODEBOOK_IDS.items()}
    for k2, cb in widths:
        K, N = 1024, 128
        trellis = torch.zeros((K // 16, N // 16, 8 * k2), dtype=torch.int16)
        L = lin.Exl3Linear.from_tensors(trellis, torch.zeros((K,), dtype=torch.float16),
                                        torch.zeros((N,), dtype=torch.float16), names[cb], None, "cpu", "strips")
        inputs = {"words": a(L.words), "suh": a(L.suh), "svh": a(L.svh), "bias": None, "counters": a(L.counters),
                  "k": K, "n": N, "k2": L.k2, "cb": cb, "layout": "strips", "split": list(L.split)}
        c = case(f"linear-all-k{k2}-cb{cb}", "linear", inputs)
        x = torch.zeros((2, K), dtype=torch.bfloat16)
        y = torch.zeros((2, N), dtype=torch.float32)
        xh = torch.zeros((2, K), dtype=torch.float16)
        z = torch.zeros((4 * 2 * N,), dtype=torch.float32)
        for split in ((1, 2), (2, 4), (1, 8)):
            L.split = split
            step(c, {"M": 2, "x": a(x), "x_dtype": DTYPES[x.dtype], "y": a(y), "y_dtype": DTYPES[y.dtype],
                     "xh": a(xh), "z": a(z), "split": list(split)}, lambda: L(x, out=y, xh=xh, z=z))
        w = torch.zeros((K, N), dtype=torch.float16)
        step(c, {"unpack": a(w)}, lambda: L.unpack(out=w))


def glm_cases():
    E, D, NI, slots = 288, 4096, 1024, 9
    gt = Big((E, D // 16, NI // 16, 32), torch.int32, ADDR.fresh())
    ut = Big((E, D // 16, NI // 16, 32), torch.int32, ADDR.fresh())
    dt = Big((E, NI // 16, D // 16, 32), torch.int32, ADDR.fresh())
    f = lambda n: torch.zeros((E, n), dtype=torch.float16)  # noqa: E731
    ex = exl3_mm.Exl3Experts(gt, ut, dt, f(D), f(D), f(NI), f(NI), f(NI), f(D), E, NI, D)
    exi = {"gt": a(gt), "ut": a(ut), "dt": a(dt), "suh_g": a(ex.suh_g), "suh_u": a(ex.suh_u), "svh_g": a(ex.svh_g),
           "svh_u": a(ex.svh_u), "suh_d": a(ex.suh_d), "svh_d": a(ex.svh_d), "count": E, "width": NI, "dims": D}
    for rows, Rs in ((8, (1, 2, 8)), (64, (16, 64))):
        plan = grouped.Plan(rows, slots, E + 1, "cpu")
        s = exl3_mm.Scratch(rows, slots, D, NI, "cpu")
        x = torch.zeros((rows, D), dtype=torch.bfloat16)
        pick = torch.zeros((rows, slots), dtype=torch.int32)
        y = torch.zeros((rows * slots, D), dtype=torch.float32)
        c = case(f"glm-mm-{rows}", "glm_routed", {
            "experts": exi, "plan": {"items": a(plan.items), "counts": a(plan.counts), "members": a(plan.members),
                                     "experts": plan.experts},
            "scratch": {"xg": a(s.xg), "xu": a(s.xu), "xd": a(s.xd), "z": a(s.z), "rows": rows, "slots": slots},
            "x": a(x), "x_stride": D, "pick": a(pick), "y": a(y), "limit": f32(7.0)})
        for R in Rs:
            step(c, {"R": R}, lambda: exl3_mm.routed(x[:R], pick, plan, ex, s, y, R, 7.0))
    # the tile settings exl3_mm never picks, straight through the extension
    plan = grouped.Plan(8, slots, E + 1, "cpu")
    s = exl3_mm.Scratch(8, slots, D, NI, "cpu")
    for nt, w in ((8, 4), (4, 4), (4, 8), (2, 4), (2, 8)):
        c = case(f"glm-grouped-{nt}x{w}", "glm_grouped", {
            "x0": a(s.xg), "x1": a(s.xu), "t0": a(gt), "t1": a(ut), "items": a(plan.items), "counts": a(plan.counts),
            "members": a(plan.members), "z": a(s.z), "mats": 2, "K": D, "N": NI, "P": 72, "SK": 1, "max_items": 20,
            "nt": nt, "warps": w, "E": E})
        step(c, {}, lambda: exl3_mm._ext().grouped(s.xg, s.xu, gt, ut, plan.items, plan.counts, plan.members, s.z,
                                                   2, D, NI, 72, 1, 20, nt, w))


def build(work: Path) -> Path:
    """The extensions' C++ wrappers from python-0.6, against the stand-in, into one executable."""

    fake = FORK / "tools/zig/exl3_launches/fake"
    torch_flags = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                   "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
                   "-std=c++20"]
    e, g = PY / "cuda/exl3", PY / "families/glm5_next/cuda"
    units = [(e / "experts.cu", MODULES["experts"]), (e / "experts_cb0.cu", MODULES["experts"]),
             (e / "experts_cb1.cu", MODULES["experts"]), (e / "experts_cb2.cu", MODULES["experts"]),
             (e / "linear.cu", MODULES["linear"]), (g / "exl3.cu", MODULES["glm"]),
             (e / "experts.cpp", MODULES["experts"]), (e / "linear.cpp", MODULES["linear"]),
             (g / "exl3.cpp", MODULES["glm"]), (FORK / "tools/zig/exl3_launches/exl3_launches.cpp", "none")]
    objs = []
    newest = max(p.stat().st_mtime for p in fake.rglob("*.h"))
    for i, (src, mod) in enumerate(units):
        obj = work / f"{i}-{src.stem}.o"
        objs.append(str(obj))
        if obj.exists() and obj.stat().st_mtime > max(src.stat().st_mtime, newest):
            continue            # a rerun in the same work directory
        inc = [f"-I{fake}", f"-I{e}"]
        if src.suffix == ".cu":
            cmd = ["nvcc", "-c", *torch_flags, "-O1", "-gencode=arch=compute_121,code=compute_121", *inc,
                   f"-DTORCH_EXTENSION_NAME={mod}", str(src), "-o", str(obj)]
        else:
            cmd = ["g++", "-c", "-std=c++20", "-O1", *inc, "-I/usr/local/cuda/include", f"-DTORCH_EXTENSION_NAME={mod}",
                   str(src), "-o", str(obj)]
        subprocess.run(cmd, check=True)
    exe = work / "exl3_launches"
    subprocess.run(["g++", "-o", str(exe), *objs], check=True)
    return exe


def key(name: str) -> str:
    if not name.startswith("_ZN"):
        return name
    i, n = 3, 0
    while name[i].isdigit():
        n, i = n * 10 + int(name[i]), i + 1
    return name[i + n:]


def main() -> int:
    out, work = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    kernels = json.loads((FORK / "zig/src/families/glm/exl3_kernels.json").read_text())
    ours: dict[tuple[str, str], tuple[str, str, dict]] = {}
    with open(work / "params.txt", "w") as f:
        for mod, images in IMAGES.items():
            for image in images:
                for k in kernels[image]:
                    ours[(mod, key(k["symbol"]))] = (image, k["symbol"], k)
                    sizes = " ".join(str(p[1]) for p in k["params"])
                    f.write(f"{mod} {key(k['symbol'])} {len(k['params'])} {sizes} {k['shared']}\n")
    exe = build(work)
    routed_cases()
    dequant_cases()
    combine_cases()
    linear_cases()
    glm_cases()
    lines = []
    for c in CASES:
        lines.append(f"CASE {c['id']}\nOPTIN {OPTIN}")
        for s in c["steps"]:
            for mod, fn, args in s["calls"]:
                lines.append(f"CALL {mod} {fn} {len(args)}")
                for d in args:
                    if d[0] == "T":
                        lines.append(f"T {d[1]:x} {d[2]} {len(d[3])} {' '.join(map(str, d[3]))} {' '.join(map(str, d[4]))}")
                    elif d[0] == "F":
                        lines.append(f"F {d[1]:x}")
                    else:
                        lines.append(" ".join(str(v) for v in d))
    (work / "calls.txt").write_text("\n".join(lines) + "\n")
    got = subprocess.run([str(exe), str(work / "params.txt")], input=(work / "calls.txt").read_text(),
                         capture_output=True, text=True, check=True).stdout
    (work / "launches.txt").write_text(got)
    by_case: dict[str, list[list[str]]] = {}
    cur = None
    for line in got.splitlines():
        w = line.split()
        if w[0] == "CASE":
            cur = by_case.setdefault(w[1], [])
        else:
            cur.append(w)
    errors, total = [], 0
    for c in CASES:
        events = by_case[c["id"]]
        call_no = 0
        for s in c["steps"]:
            mods = [m for m, _, _ in s["calls"]]
            first, s["launches"] = call_no, []
            call_no += len(s["calls"])
            pending = {}
            for w in events:
                n = int(w[1])
                if not first <= n < call_no:
                    continue
                if w[0] == "ERROR":
                    errors.append(f"{c['id']}: {' '.join(w[2:])}")
                    continue
                image, sym, _ = ours[(mods[n - first], key(w[2]))]
                if w[0] == "ATTR":
                    pending[sym] = int(w[4])
                elif w[0] == "LAUNCH":
                    s["launches"].append({"image": image, "symbol": sym, "grid": [int(v) for v in w[3:6]],
                                          "block": [int(v) for v in w[6:9]], "shared": int(w[9]),
                                          "max_dynamic": pending.pop(sym, None), "params": w[11:11 + int(w[10])]})
                    total += 1
            del s["calls"]
    if errors:
        raise SystemExit("the C++ wrappers refused calls:\n" + "\n".join(errors[:20]))
    launched = {(x["image"], x["symbol"]) for c in CASES for s in c["steps"] for x in s["launches"]}
    head = {"source": "python-0.6 (ed78d6f) cuda/exl3 experts.py, linear.py and glm5_next/cuda/exl3_mm.py, with "
                      "their extensions' C++ wrappers, run on a CPU by tools/zig/exl3_fixtures.py",
            "harness_sha256": hashlib.sha256((FORK / "tools/zig/exl3_launches/exl3_launches.cpp").read_bytes()
                                             + (FORK / "tools/zig/exl3_launches/fake/tf_fake.h").read_bytes()).hexdigest(),
            "kernels_launched": len(launched), "kernels": sum(len(v) for v in kernels.values()),
            "case_count": len(CASES), "launches": total}
    body = ",\n".join(json.dumps(c, separators=(",", ":")) for c in CASES)
    out.write_text(json.dumps(head, indent=1)[:-2] + ',\n "cases": [\n' + body + "\n ]\n}\n")
    print(f"{len(CASES)} cases, {total} launches, {len(launched)} of {head['kernels']} kernels launched -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
