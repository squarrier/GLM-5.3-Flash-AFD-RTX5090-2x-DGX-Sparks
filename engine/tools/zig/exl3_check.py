#!/usr/bin/env python3
"""python-0.6's half of the EXL3 kernel check: layer L's routed experts of TP2 rank R of Mia's EXL3
GLM-5.3-Flash, read with python-0.6's own split.RankReader and built as weights.load's moe_exl3 builds them
(experts.prepare(..., "mcg")), then every EXL3 kernel GLM can run on them, on seeded inputs. OUT/manifest.json holds
the cases, shapes, dtypes, scratch bytes and each array's sha256; OUT/in and OUT/out hold the arrays (raw, little
endian). tf-exl3-check (zig/tests/cuda/glm_exl3_check.zig) replays each case through the Zig bindings, compares bytes.
--dry stops before the kernels (CPU; a header-only copy of the checkpoint serves). --build-ext builds the four CUDA
extensions alone (a container without the checkpoint); --no-build refuses to read weights if one would build.
--compare A B: two manifests' output sha256s (a rerun's determinism); --hash-only keeps the outputs' sha256s alone.
The kernels and wrappers are Ash Hart's and the TensorFold contributors' (python-0.6 ed78d6f); the EXL3 format is
turboderp's (ExLlamaV3, MIT); the checkpoint is Mia's (MiaAI-Lab, @MiaAI_lab, 078455ff).
usage: exl3_check.py --src /tensorfold/src --ckpt DIR --layer 3 --rank R --out DIR [--dry] [--no-build] [--hash-only]
       exl3_check.py --src /tensorfold/src --build-ext | exl3_check.py --compare A/manifest.json B/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

T0 = time.time()
PREFIX = "model.language_model."
SEED = 20261007
SLOTS = 9                                   # GLM's top 8 and the shared expert's slot (pick = E, skipped)
ROUTED = {"routed-bf16": (1, 2, 3, 4, 5, 8, 16, 64, 128, 256, 594, 1024, 2048), "routed-f32": (1, 8, 2048),
          "routed-wts": (1, 8, 64), "routed-fp16": (3, 8), "routed-hot": (64, 2048)}
COMBINE = (1, 8)
GLM_MM = (1, 8, 2048)
LINEAR_M = (1, 8, 64, 128)
EXPERTS = (0, 7)
FP16_LIMIT = 7.0
# the extensions the cases load: name, module (experts.route, the affine module's, plans glm-mm's pairs);
# TF_EXL3_CHECK_EXPERTS names the experts one (qualification_6052b66d's oracle, python-0.6.6 + 0128: tensorfold_exl3_experts_v1_0128)
EXTS = ((os.environ.get("TF_EXL3_CHECK_EXPERTS", "tensorfold_exl3_experts_v1"), "tensorfold.cuda.exl3.experts"),
        ("tensorfold_exl3_linear_v3", "tensorfold.cuda.exl3.linear"),
        ("tensorfold_glm_exl3_v1", "tensorfold.families.glm5_next.cuda.exl3_mm"),
        ("tensorfold_experts_v7", "tensorfold.cuda.experts"))
SOURCES = ("cuda/exl3/experts.py", "cuda/exl3/experts.cpp", "cuda/exl3/experts.cu", "cuda/exl3/linear.py",
           "cuda/exl3/linear.cu", "families/glm5_next/cuda/exl3_mm.py", "families/glm5_next/cuda/exl3.cu",
           "families/glm5_next/cuda/exl3_generic.py", "families/glm5_next/cuda/split.py",
           "families/glm5_next/cuda/weights.py", "cuda/experts.py")
DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32", "int32": "i32", "int16": "i16", "int64": "i64"}
LOW = {"gib": 1e9, "at": ""}
NOW = {"at": "start"}


def say(*a) -> None:
    print(f"[{time.time() - T0:8.2f}]", *a, flush=True)


def avail() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    return -1.0


def sampler() -> None:
    while True:
        a = avail()
        if a < LOW["gib"]:
            LOW.update(gib=a, at=NOW["at"])
        time.sleep(0.25)


class Run:
    """The output folder, the manifest's cases and the memory marks."""

    def __init__(self, a, torch, np) -> None:
        self.a, self.torch, self.np = a, torch, np
        self.out = Path(a.out)
        for d in ("in", "out"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.dev = "cpu" if a.dry else "cuda"
        self.cases: list[dict] = []
        self.marks: list[dict] = []
        self.listing: list[str] = []

    def mark(self, what: str) -> None:
        NOW["at"] = what
        d = {"t": round(time.time() - T0, 2), "at": what, "avail_gib": round(avail(), 3)}
        if not self.a.dry:
            d["device_peak_gib"] = round(self.torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        self.marks.append(d)
        say("mark", json.dumps(d))

    def sync(self) -> None:
        if not self.a.dry:
            self.torch.cuda.synchronize()

    def meta(self, name: str, t) -> dict:
        return {"name": name, "dtype": DT[str(t.dtype).replace("torch.", "")], "shape": list(t.shape),
                "bytes": t.numel() * t.element_size()}

    def raw(self, t):
        return t.detach().contiguous().reshape(-1).view(self.torch.uint8).cpu().numpy()

    def array(self, sub: str, cid: str, name: str, t, write: bool, derived: bool = False) -> dict:
        rec = self.meta(name, t)
        if sub == "out" and self.a.dry:
            return rec                                   # no kernel ran: the shape and dtype alone
        b = self.raw(t)
        rec["sha256"] = hashlib.sha256(memoryview(b)).hexdigest()
        if derived:
            rec["derived"] = True                        # made on the GPU (a routed y, a plan): zeros in a dry run
        if write:
            rec["file"] = f"{sub}/{cid}.{name}.bin"
            b.tofile(self.out / rec["file"])
        return rec

    def inp(self, cid: str, name: str, t, derived: bool = False) -> dict:
        return self.array("in", cid, name, t, True, derived)

    def outp(self, cid: str, name: str, t) -> dict:
        return self.array("out", cid, name, t, not self.a.hash_only)

    def case(self, cid: str, kind: str, call: dict, inputs: list, outputs: list, scratch: dict) -> None:
        self.cases.append({"id": cid, "kind": kind, "call": call, "inputs": inputs, "outputs": outputs,
                           "scratch": scratch})
        self.listing.append(listing(cid, kind, inputs, outputs, scratch))


def listing(cid: str, kind: str, inputs: list, outputs: list, scratch: dict) -> str:
    """One case as both tools' --dry print it: names sorted, shapes and dtypes, scratch bytes."""

    def arr(rs):
        return " ".join(f"{r['name']}:{r['dtype']}[{','.join(str(v) for v in r['shape'])}]"
                        for r in sorted(rs, key=lambda r: r["name"]))
    sc = " ".join(f"{k}={v}" for k, v in sorted(scratch.items()))
    return f"{cid} {kind} | in {arr(inputs)} | out {arr(outputs)} | scratch {sc or '-'}"


def rng(np, cid: str, what: str):
    """numpy's PCG64 (the same draws on x86_64 and arm64), one stream an input."""

    h = hashlib.sha256(f"{SEED} {cid} {what}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def normal(run: Run, cid: str, rows: int, cols: int, dtype):
    a = rng(run.np, cid, "x").standard_normal((rows, cols), dtype=run.np.float32)
    return run.torch.from_numpy(a).to(dtype)


def picks(run: Run, cid: str, R: int, E: int, hot: bool = False):
    """8 distinct experts a row, then slot 8 = E (the shared expert's, which the kernels skip)."""

    g, np = rng(run.np, cid, "pick"), run.np
    p = np.full((R, SLOTS), E, dtype=np.int32)
    first = g.permutation(E)[:SLOTS - 1]
    for r in range(R):
        p[r, :SLOTS - 1] = first if hot else g.permutation(E)[:SLOTS - 1]
    return run.torch.from_numpy(p)


def weights(run: Run, cid: str, R: int):
    return run.torch.from_numpy(rng(run.np, cid, "wts").random((R, SLOTS), dtype=run.np.float32))


def scratch_bytes(**bufs) -> dict:
    return {k: v.numel() * v.element_size() for k, v in bufs.items() if v is not None}


class OnCuda:
    """--dry: a host trellis that passes prepare()'s device check (no kernel ever reads it)."""

    def __init__(self, t) -> None:
        self.t, self.dtype, self.shape = t, t.dtype, t.shape
        self.device = SimpleNamespace(type="cuda")

    def is_contiguous(self) -> bool:
        return self.t.is_contiguous()

    def data_ptr(self) -> int:
        return self.t.data_ptr()


def names_of(L: int, E: int) -> list[str]:
    """weights.load's expert_names for an EXL3 layer: gate, up, down; each expert's trellis, suh, svh."""

    p = PREFIX + f"layers.{L}.mlp."
    return [p + f"experts.{e}.{proj}.{x}" for proj in ("gate_proj", "up_proj", "down_proj") for e in range(E)
            for x in ("trellis", "suh", "svh")]


def ext_so(name: str) -> str:
    from torch.utils import cpp_extension

    return os.path.join(cpp_extension._get_build_directory(name, verbose=False), name + ".so")


def load_exts(no_build: bool) -> list[dict]:
    """Every extension the cases use, before any weight is read: a build here (--no-build) stops the run."""

    import importlib

    from tensorfold.cuda import build

    got = []
    for name, mod in EXTS:
        m = importlib.import_module(mod)
        so = ext_so(name)
        srcs = [str(p) for p in Path(m.__file__).parent.glob("*.c*")]
        before = os.path.getmtime(so) if os.path.exists(so) else None
        if no_build and (before is None or build._needs_build(so, srcs)):
            raise SystemExit(f"exl3_check: extension {name} is not built at {so}: run --build-ext in a container "
                             "without the checkpoint first (no weight was read)")
        t = time.time()
        m._ext()
        after = os.path.getmtime(so)
        built = before is None or after != before
        if no_build and built:
            raise SystemExit(f"exl3_check: extension {name} was built in this run (no weight was read)")
        d = {"name": name, "so": so, "seconds": round(time.time() - t, 2), "built_now": built,
             "sha256": hashlib.sha256(Path(so).read_bytes()).hexdigest()}
        say("extension", json.dumps(d))
        got.append(d)
    return got


def build_ext(a) -> int:
    import torch

    say("build-ext: torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(),
        "TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"), "MemAvailable %.2f GiB" % avail())
    got = load_exts(False)
    say(f"build-ext done: {len(got)} extensions, {sum(d['built_now'] for d in got)} built now; "
        f"lowest MemAvailable {LOW['gib']:.2f} GiB ({LOW['at']})")
    return 0


def compare(a, b) -> int:
    """A rerun's determinism: the outputs of two runs of one rank, by sha256 (inputs too, which are seeded)."""

    ma, mb = (json.loads(Path(p).read_text()) for p in (a, b))

    def hashes(m):
        return {(c["id"], io, r["name"]): r.get("sha256") for c in m["cases"] for io in ("inputs", "outputs")
                for r in c[io]}
    ha, hb = hashes(ma), hashes(mb)
    bad = [k for k in sorted(set(ha) | set(hb)) if ha.get(k) is None or ha.get(k) != hb.get(k)]
    outs = [k for k in ha if k[1] == "outputs"]
    for k in bad[:20]:
        print(f"DIFFER {k[0]} {k[1][:-1]} {k[2]}: {ha.get(k)} vs {hb.get(k)}")
    same = len([k for k in outs if k not in bad])
    verdict = "PASS" if not bad else "FAIL"
    print(f"{verdict} determinism: {same} of {len(outs)} outputs (and {len(ha) - len(outs)} inputs) "
          f"of {len(ma['cases'])} cases have the same sha256 in both runs")
    return 0 if not bad else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="/tensorfold/src")
    p.add_argument("--ckpt")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--hash-only", action="store_true", help="outputs hashed, not written (a determinism rerun)")
    p.add_argument("--build-ext", action="store_true")
    p.add_argument("--compare", nargs=2)
    a = p.parse_args()
    if a.compare:
        return compare(*a.compare)
    sys.path.insert(0, a.src)
    threading.Thread(target=sampler, daemon=True).start()
    if a.build_ext:
        return build_ext(a)
    if not (a.ckpt and a.out):
        p.error("--ckpt and --out are required")
    import numpy as np
    import torch

    from exl3_check_cases import run_all  # beside this file
    run = Run(a, torch, np)
    run.mark("start")
    exts = [] if a.dry else load_exts(a.no_build)
    run.mark("extensions loaded")
    head = run_all(run, names_of, OnCuda)
    head.update(tool="exl3_check.py", version=1, dry=a.dry, ckpt=str(a.ckpt), layer=a.layer, rank=a.rank, ranks=2,
                seed=SEED, slots=SLOTS, extensions=exts, torch=torch.__version__, cuda=torch.version.cuda,
                device=None if a.dry else torch.cuda.get_device_name(), argv=sys.argv,
                sources={s: hashlib.sha256((Path(a.src) / "tensorfold" / s).read_bytes()).hexdigest()
                         for s in SOURCES},
                marks=run.marks, lowest_avail={"gib": round(LOW["gib"], 3), "at": LOW["at"]})
    kinds: dict[str, int] = {}
    for c in run.cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    head["kinds"] = kinds
    head["cases"] = run.cases
    (run.out / "manifest.json").write_text(json.dumps(head, indent=1) + "\n")
    (run.out / "dry.txt").write_text("\n".join(head["listing_head"] + run.listing) + "\n")
    say(f"{len(run.cases)} cases ({', '.join(f'{k} {v}' for k, v in kinds.items())}) -> {run.out}/manifest.json; "
        f"lowest MemAvailable {LOW['gib']:.2f} GiB ({LOW['at']})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())
