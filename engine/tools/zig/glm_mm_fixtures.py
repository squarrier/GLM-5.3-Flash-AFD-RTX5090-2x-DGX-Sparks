"""GLM-5.3-Flash's affine-kernel launches as python-0.6 makes them, for zig/src/families/glm/cuda_kernels_mm_test.zig.

python-0.6's Python wrappers (kernels/qmm.py matmul and prefill_matmul through glm5_next/cuda/qmm.py and forward.mm;
cuda/experts.py route, gate_up and down) run on CPU tensors of one TP=2 rank's shapes, with each extension's ``_ext()``
replaced by a recorder; the slice-1 forward trace's own qmm calls (fixtures_cuda_forward.json) join them. Every recorded
call then goes through the extensions' own C++ wrappers (qmm.cpp, qmm*.cu, experts.cpp, experts*.cu, compiled from the
python-0.6 tree as they are against tools/zig/glm_mm_launches' torch and CUDA-runtime stand-in), which print each launch:
kernel, grid, block, dynamic shared bytes, cluster, PDL, and every argument's bytes. Design: qualification_765330ce's exl3_fixtures.py.

Runs in NVIDIA's PyTorch 26.07 image, no GPU: python-0.6 on PYTHONPATH, its tree at --py06, this tree at --fork.
usage: glm_mm_fixtures.py --py06 /tensorfold --fork /fork --work DIR --out FILE
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.kernels import qmm as shared
from tensorfold.families.glm5_next.cuda import qmm as glm_qmm

bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
DTYPE = {torch.uint8: 0, torch.int8: 1, torch.int16: 2, torch.int32: 3, torch.int64: 4, torch.float16: 5,
         torch.float32: 6, torch.float64: 7, torch.bfloat16: 8}
TRACE_DTYPE = {"bf16": 8, "f32": 6, "i32": 3}
GB10 = (12, 1, 48, 101376)
RTX5090 = (12, 0, 170, 101376)
OCCUPANCY = 4                       # resident decode-expert blocks an SM the stand-in answers (the formula's input)
D, LL, HL, QK, VD, QL, KL, IH, ID, DL, ML, VOCAB, E, SLOTS, LIMIT = (4096, 32, 32, 256, 256, 1536, 512, 32, 128, 6144,
                                                                     1024, 77440, 289, 9, 10.0)
KW = 3 * LL * 128 + 2 * 128 + LL    # the stacked KDA in-projection row (q | k | v | f_a | g_a | beta)
# MLX 4-bit projections a rank runs: name -> (n, k, fp32 out, x row stride) (ZK4's glm_triton_cases.Dims.mats)
MATS = {"kda_proj": (KW, D, False, D), "kda_fb": (LL * 128, 128, False, KW), "kda_o": (D, LL * 128, True, LL * 128),
        "dsa_proj": (QL + KL, D, False, D), "dsa_qb": (HL * QK, QL, False, QL), "dsa_o": (D, HL * VD, True, HL * VD),
        "idx_kw": (ID + IH, D, False, D), "idx_qb": (IH * ID, QL, False, QL), "mlp_gu": (2 * DL, D, False, D),
        "mlp_down": (D, DL, True, DL), "mtp_eh": (D, 2 * D, False, 2 * D), "head": (VOCAB, D, False, D)}
DECODE_ROWS = (1, 2, 3, 4, 5, 7, 8, 16)
PROMPT_ROWS = (1, 17, 128, 129, 2048)
EXPERT_DECODE_ROWS = (1, 2, 3, 4, 8, 16)
EXPERT_PROMPT_ROWS = (1, 17, 64, 113, 114, 2048)
QMM_SOURCES = ("qmm.cpp", "qmm.cu", "qmm_group.cu", "qmm_prefill.cu", "qmm_prefill8.cu")
EXPERTS_SOURCES = ("experts.cpp", "experts.cu", "experts_prefill.cu", "experts_pack.cu")
TORCH_FLAGS = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
               "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-std=c++20", "-O3"]


class Recorder:
    """Stands in for one extension: each call python-0.6 makes is kept, in order, nothing runs."""

    def __init__(self, module: str, calls: list) -> None:
        self.module, self.calls = module, calls

    def __getattr__(self, fn: str):
        return lambda *args: self.calls.append((self.module, fn, args))


class Space:
    """Stand-in addresses: each storage a base 2^36 bytes from the next, a view its byte offset into it."""

    def __init__(self) -> None:
        self.bases: dict = {}

    def addr(self, key, offset: int) -> int:
        return (self.bases.setdefault(key, len(self.bases) + 1) << 36) + offset


def encode(v, space: Space):
    if isinstance(v, torch.Tensor):
        st = v.untyped_storage()
        addr = space.addr(st.data_ptr(), v.data_ptr() - st.data_ptr())
        return {"t": {"addr": addr, "dtype": DTYPE[v.dtype], "shape": list(v.shape), "stride": list(v.stride())}}
    if isinstance(v, (list, tuple)):
        return {"list": [encode(x, space) for x in v]}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"float": struct.unpack("<Q", struct.pack("<d", v))[0]}
    if v is None:
        return {"none": True}
    raise TypeError(f"no encoding for {type(v)}")


def from_trace(v, space: Space):
    """An argument of a slice-1 trace event (buffer path and byte offset) at a stand-in address."""
    if "t" in v:
        t = v["t"]
        return {"t": {"addr": space.addr(t["buf"], t["off"]), "dtype": TRACE_DTYPE[t["dtype"]], "shape": t["shape"],
                      "stride": t["stride"], "buf": t["buf"], "off": t["off"]}}
    if "list" in v:
        return {"list": [from_trace(x, space) for x in v["list"]]}
    return dict(v)


def q4(n: int, k: int) -> shared.Q4:
    npad = -(-n // 128) * 128
    return shared.Q4(torch.empty(npad // 64, k // 64, 8, 32, 2, dtype=i32), torch.empty(k // 64, npad, dtype=bf),
                     torch.empty(k // 64, npad, dtype=bf), n, k, 64)


def rows(r: int, k: int, stride: int, off: int = 0) -> torch.Tensor:
    """r rows of k bf16 inputs, `stride` elements apart, `off` elements into each row (forward.py's views)."""
    return torch.empty(r, max(stride, off + k), dtype=bf)[:, off:off + k]


def qmm_cases():
    """Decode windows through glm5_next qmm.matmul (on a GB10, then a few on an RTX 5090: python's other tiles),
    prompt chunks through forward.mm's prefill_matmul, at every MLX projection's shape."""
    for name, (n, k, out32, stride) in MATS.items():
        w = q4(n, k)
        off = 3 * LL * 128 if name == "kda_fb" else 0
        for r in DECODE_ROWS:
            def call(r=r, w=w, n=n, k=k, out32=out32, stride=stride, off=off):
                x = rows(r, k, stride, off)
                glm_qmm.matmul(x, w, torch.empty(r, k // 64, dtype=f32), out=torch.empty(r, n, dtype=f32 if out32 else bf),
                               f32=out32)
            yield f"decode/{name}/r{r}", GB10, call
        for r in (1, 8, 16, 17):
            if name in ("kda_proj", "kda_o"):
                yield f"decode-rtx5090/{name}/r{r}", RTX5090, lambda r=r, w=w, n=n, k=k, out32=out32, stride=stride: \
                    glm_qmm.matmul(rows(r, k, stride), w, torch.empty(r, k // 64, dtype=f32),
                                   out=torch.empty(r, n, dtype=f32 if out32 else bf), f32=out32)
        for r in PROMPT_ROWS:
            yield f"prompt/{name}/r{r}", GB10, lambda r=r, w=w, n=n, k=k, out32=out32, stride=stride, off=off: \
                shared.prefill_matmul(rows(r, k, stride, off), w, f32=out32,
                                      out=torch.empty(r, n, dtype=f32 if out32 else bf))


def expert_cases():
    """moe_block's MLX path: route, gate_up (SwiGLU, limit 10), down; decode windows and prompt chunks."""
    ex = grouped.Experts(torch.empty(E, ML // 32, D // 64, 2, 288, dtype=i32), torch.empty(E, D // 32, ML // 64, 1, 288,
                         dtype=i32), 64, ML, D, LIMIT)
    for prefill, row_set in ((False, EXPERT_DECODE_ROWS), (True, EXPERT_PROMPT_ROWS)):
        for r in row_set:
            def call(r=r, prefill=prefill):
                plan = grouped.Plan(r, SLOTS, E, "cpu", prefill=prefill)
                grouped.route(torch.empty(r, SLOTS, dtype=i32), plan)
                eact = torch.empty(r * SLOTS, ML, dtype=bf)
                grouped.gate_up(torch.empty(r, D, dtype=bf), ex, plan, eact, r)
                grouped.down(eact, ex, plan, torch.empty(r, SLOTS, D, dtype=bf if prefill else f32).view(-1, D), r)
            yield f"{'prompt' if prefill else 'decode'}/experts/r{r}", GB10, call


def trace_cases(fixture: Path):
    """Every qmm call of the slice-1 forward trace (python-0.6's layer_forward, MLX checkpoint), as the Zig launch
    halves receive it: buffer paths and offsets at stand-in addresses."""
    f = json.loads(fixture.read_text())
    seen = set()
    for cid, case in f["cases"].items():
        for i, ev in enumerate(case["events"]):
            if ev["op"] != "ext" or ev["fn"] not in ("qmm.qmm_group", "qmm.qmm_prefill"):
                continue
            sig = json.dumps(ev["args"], sort_keys=True)
            if sig in seen:
                continue
            seen.add(sig)
            space = Space()
            yield f"trace/{cid}/e{i}", GB10, ("qmm", ev["fn"].split(".")[1], [from_trace(a, space) for a in ev["args"]])


def wire(a) -> str:
    """One argument in glm_mm_launches.cpp's input format."""
    if "t" in a:
        t = a["t"]
        return " ".join(["T", f"{t['addr']:x}", str(t["dtype"]), str(len(t["shape"])), *map(str, t["shape"]),
                         *map(str, t["stride"])])
    if "list" in a:
        return " ".join(["L", str(len(a["list"])), *(wire(x) for x in a["list"])])
    if "bool" in a:
        return f"B {int(a['bool'])}"
    if "int" in a:
        return f"I {a['int']}"
    if "float" in a:
        return f"F {a['float']:x}"
    return "N"


def build(py06: Path, fork: Path, work: Path) -> Path:
    """The stand-in: both extensions' C++ and CUDA sources as they are, against tools/zig/glm_mm_launches/fake."""
    work.mkdir(parents=True, exist_ok=True)
    fake = fork / "tools/zig/glm_mm_launches/fake"
    jobs = [(fork / "tools/zig/glm_mm_launches/glm_mm_launches.cpp", "stand-in")]
    jobs += [(py06 / "src/tensorfold/cuda/kernels" / s, "qmm") for s in QMM_SOURCES]
    jobs += [(py06 / "src/tensorfold/cuda" / s, "experts") for s in EXPERTS_SOURCES]

    def compile_one(job):
        src, mod = job
        obj = work / f"{mod}-{src.name}.o"
        cmd = ["nvcc", "-c", *TORCH_FLAGS, "-gencode=arch=compute_121,code=sm_121", f"-I{fake}", "-x",
               "cu" if src.suffix == ".cu" else "c++", f"-DTORCH_EXTENSION_NAME={mod}", str(src), "-o", str(obj)]
        got = subprocess.run(cmd, capture_output=True, text=True)
        if got.returncode:
            raise SystemExit(f"compile {src}: {got.stderr[-3000:]}")
        return obj

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        objs = list(pool.map(compile_one, jobs))
    exe = work / "glm_mm_launches"
    got = subprocess.run(["nvcc", "-cudart", "none", *map(str, objs), "-o", str(exe)], capture_output=True, text=True)
    if got.returncode:
        raise SystemExit(f"link: {got.stderr[-3000:]}")
    return exe


def kparams(exe: Path) -> dict[str, list[list[int]]]:
    """Every kernel's parameters (EIATTR_KPARAM_INFO), keyed by its name past the namespace, as the cubins hold them."""
    text = subprocess.run(["cuobjdump", "-elf", str(exe)], capture_output=True, text=True, check=True).stdout
    out: dict[str, list] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"\s*\.nv\.info\.(\S+)", line)
        if m:
            name = key(m.group(1))
            out.setdefault(name, [])
            continue
        m = re.search(r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*(0x[0-9a-f]+)\s+Size\s*:\s*(0x[0-9a-f]+)", line)
        if m and name is not None:
            out[name].append([int(x, 16) for x in m.groups()])
    return {k: [[o, s] for _, o, s in sorted(v)] for k, v in out.items() if v}


def key(symbol: str) -> str:
    m = re.match(r"_ZN(\d+)", symbol)
    return symbol if m is None else symbol[m.end() + int(m.group(1)):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--py06", type=Path, required=True)
    ap.add_argument("--fork", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    calls: list = []
    shared._ext = lambda: Recorder("qmm", calls)
    grouped._ext = lambda: Recorder("experts", calls)
    shared.grouped = lambda device: True
    chip = list(GB10)
    shared._chip = lambda device: tuple(chip[:3])
    cases = []
    for cid, c, run in [*qmm_cases(), *expert_cases()]:
        chip[:] = c
        del calls[:]
        run()
        space = Space()
        cases.append({"id": cid, "chip": list(c), "calls": [{"module": m, "fn": fn, "args": [encode(x, space) for x in args]}
                                                         for m, fn, args in calls]})
    for cid, c, (m, fn, args) in trace_cases(a.fork / "zig/src/families/glm/fixtures_cuda_forward.json"):
        cases.append({"id": cid, "chip": list(c), "calls": [{"module": m, "fn": fn, "args": args}]})
    exe = build(a.py06, a.fork, a.work)
    table = kparams(exe)
    (a.work / "params.txt").write_text("".join(f"{k} {len(v)} {' '.join(str(s) for _, s in v)}\n"
                                               for k, v in sorted(table.items())))
    lines = [f"OCC {OCCUPANCY}"]
    for c in cases:
        lines.append(f"CASE {c['id']}")
        lines.append("CHIP " + " ".join(map(str, c["chip"])))
        for call in c["calls"]:
            lines.append(" ".join(["CALL", call["module"], call["fn"], str(len(call["args"])), *map(wire, call["args"])]))
    got = subprocess.run([str(exe), str(a.work / "params.txt")], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout
    (a.work / "launches.txt").write_text(got)
    by_id = {c["id"]: c for c in cases}
    attrs, current = set(), None
    for line in got.splitlines():
        w = line.split()
        if w[0] == "CASE":
            current = by_id[w[1]]
        elif w[0] == "LAUNCH":
            n = int(w[14])
            current["calls"][int(w[1])].setdefault("launches", []).append(
                {"kernel": key(w[2]), "grid": list(map(int, w[3:6])), "block": list(map(int, w[6:9])), "smem": int(w[9]),
                 "cluster": list(map(int, w[10:13])), "pdl": int(w[13]), "args": w[15:15 + n]})
        elif w[0] == "ATTR":
            attrs.add((key(w[2]), int(w[3]), int(w[4])))
        elif w[0] == "OCCQ":
            current["calls"][int(w[1])].setdefault("occupancy_queries", []).append([key(w[2]), int(w[3]), int(w[4])])
        elif w[0] == "ERROR":
            current["calls"][int(w[1])]["error"] = line.split(" ", 2)[2]
    srcs = [a.py06 / "src/tensorfold/cuda/kernels" / s for s in QMM_SOURCES] + \
           [a.py06 / "src/tensorfold/cuda" / s for s in EXPERTS_SOURCES]
    out = {"schema": "glm-mm-launches/1", "generator": "tools/zig/glm_mm_fixtures.py",
           "python": {"tree": "python-0.6 ed78d6f", "sources": {str(s.relative_to(a.py06)): hashlib.sha256(
               s.read_bytes()).hexdigest() for s in srcs}},
           "occupancy": OCCUPANCY, "attrs": sorted([list(x) for x in attrs]),
           "kparams": {k: v for k, v in table.items() if any(
               ln.get("kernel") == k for c in cases for call in c["calls"] for ln in call.get("launches", []))},
           "cases": cases}
    a.out.write_text(json.dumps(out, indent=None, separators=(",", ":"), sort_keys=True) + "\n")
    n_calls = sum(len(c["calls"]) for c in cases)
    n_launch = sum(len(call.get("launches", [])) for c in cases for call in c["calls"])
    n_err = sum("error" in call for c in cases for call in c["calls"])
    print(f"{len(cases)} cases, {n_calls} calls, {n_launch} launches, {n_err} errors, {len(attrs)} raised limits, "
          f"{len(out['kparams'])} kernels; {a.out} sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
