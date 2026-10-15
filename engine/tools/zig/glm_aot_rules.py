"""glm_aot.py's GLM-5.3-Flash rules: the CPU capture of the case list; derive's row ints, buckets and chunk counts; the
AFD attention node's world-1 set (a world-1 capture with hc_post over three planes, merged into the TP=2 spec)."""
# The CPU capture is the method of tools/zig/flashnext_aot.py, written by Mia (MiaAI-Lab, @MiaAI_lab) in
# MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, Apache License 2.0; GLM's derive rules are new.

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# B16_CONFIG of glm5_next/cuda/qmm.py: (warps, stages) by row bucket; latent.py's head and row blocks
B16_CONFIG = {16: (4, 3), 32: (4, 3), 64: (4, 2), 128: (8, 2)}
BUCKETS = (16, 32, 64, 128)
NCH_DENSE = tuple(range(1, 10))     # latent._merge over a window's chunks: verify windows 1-6, prompt chunks 1-9
NCH_SPARSE = (5,)                   # latent._merge of the sparse rows: triton.cdiv(2051 tokens, 512)
NCH_AB = tuple(range(1, 10)) + (513,)
# the runtime int that follows a window's rows, by kernel; True: the value 1 is reachable
ROW_INTS = {"latent._absorb_q": ("R", True), "latent._expand_v": ("R", True), "latent._absorb_q4": ("R", True),
            "latent._expand_v4": ("R", True), "latent._dense_chunks": ("R", True), "latent._merge": ("R", True),
            "latent._sparse_chunks": ("R", True), "sparse._pool_keys": ("R", True), "sparse._scores": ("R", True),
            "sparse._select_rows": ("NP", False), "glue._router_part": ("M", True), "glue._router": ("M", True),
            "qmm._bmm": ("M", True), "forward._conv_shift": ("keep", True), "attention._chunks": ("R", True),
            "attention._merge": ("R", True)}


def short(k: dict) -> str:
    """``latent._merge`` for tensorfold.families.glm5_next.cuda.latent._merge."""
    return k["function"].split("glm5_next.cuda.")[-1]


def ints(k: dict) -> dict:
    return {n: (v.get("int") if "int" in v else v.get("bool")) for n, v in k["constexprs"].items()}


def reorder(k: dict) -> dict:
    """Signature, attrs and constexprs in parameter order, as the JIT builds them (fp32 arguments get no attrs)."""
    p = k["params"]
    k["signature"] = {n: k["signature"][n] for n in p}
    k["attrs"] = {n: k["attrs"].get(n, []) for n in p if n in k["attrs"] or k["signature"][n] != "fp32"}
    k["constexprs"] = {n: k["constexprs"][n] for n in sorted(k["constexprs"], key=lambda x: p.index(x.split(".")[0]))}
    return k


def with_consts(k: dict, consts: dict, options: dict | None = None) -> dict:
    out = json.loads(json.dumps(k))
    for n, v in consts.items():
        out["constexprs"][n] = {"bool": v} if isinstance(v, bool) else {"int": v}
    if options:
        out["options"].update(options)
    return out


def int_form(k: dict, name: str, form: str) -> dict:
    """``name`` as Triton specializes a runtime int: the value 1 (a constexpr), a multiple of 16, or neither."""
    out = json.loads(json.dumps(k))
    if form == "one":
        out["signature"][name], out["constexprs"][name], out["attrs"][name] = "constexpr", {"int": 1}, []
    else:
        out["signature"][name] = "i32"
        out["constexprs"].pop(name, None)
        out["attrs"][name] = [["tt.divisibility", 16]] if form == "div16" else []
    return reorder(out)


def derived(k: dict, why: str) -> dict:
    out = json.loads(json.dumps(k))
    src = out.pop("hash", None) or out.get("expanded_from")
    out.pop("cubin_sha256", None)
    out["expanded_from"] = src
    out["rule"] = (out["rule"] + "; " if out.get("rule") else "") + why
    return out


def key(k: dict) -> str:
    return json.dumps([k["function"], k["signature"], k["constexprs"], k["attrs"], k["options"]], sort_keys=True)


def spec_key(k: dict) -> str:
    """A specialization across hosts: Triton's kernel hash also covers libtriton.so (qualification_4cb8c4ff NOTES 3), so it differs
    between an arm64 capture and an amd64 build of the same cubin; extern_libs' paths are the install's too."""
    options = {n: v for n, v in k["options"].items() if n != "extern_libs"}
    text = json.dumps([k["function"], k["signature"], k["constexprs"], k["attrs"], options], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def buckets(k: dict) -> list[tuple[dict, str]]:
    """Every row bucket, head block and chunk count the wrappers can pick for this entry's kernel."""
    f, c = short(k), ints(k)
    if f == "qmm._bmm":
        return [(with_consts(k, {"BM": bm}, {"num_warps": B16_CONFIG[bm][0], "num_stages": B16_CONFIG[bm][1]}),
                 f"rows bucket BM {bm}") for bm in BUCKETS]
    if f in ("glue._router_part", "glue._router"):
        return [(with_consts(k, {"BM": bm}), f"rows bucket BM {bm}") for bm in BUCKETS]
    if f == "latent._absorb_q4":
        return [(with_consts(k, {"RB": rb}), f"row block RB {rb}") for rb in (1, 16)]
    if f in ("latent._dense_chunks", "latent._sparse_chunks"):
        return [(with_consts(k, {"HBT": hb}), f"head block HBT {hb}") for hb in (16, 32)]
    if f == "latent._merge":
        nchs = NCH_SPARSE if c["SPARSE"] else NCH_DENSE
        return [(with_consts(k, {"NCH": n}), f"chunks NCH {n}") for n in nchs]
    if f == "attention._merge":
        return [(with_consts(k, {"NCH": n}), f"chunks NCH {n}") for n in NCH_AB]
    return [(k, "")]


def forms(k: dict) -> list[tuple[dict, str]]:
    """The row int's three specializations (two when it never is 1, or when a bucket past 16 rows holds it)."""
    f, c = short(k), ints(k)
    name, one = ROW_INTS.get(f, (None, False))
    if name is None or name not in k["params"]:
        return [(k, "")]
    if c.get("BM", 16) > 16 or c.get("RB", 1) > 1:
        one = False
    return [(int_form(k, name, form), f"{name} {form}") for form in ("div16", "plain") + (("one",) if one else ())]


def derive(spec: Path, out: Path) -> int:
    """The extracted entries plus every row bucket, chunk count and row-int form GLM's wrappers can launch."""
    data = json.loads(spec.read_text())
    captured = [k for k in data["kernels"] if "expanded_from" not in k]
    have = {key(k) for k in captured}
    uniq: dict[str, dict] = {}
    for k in captured:
        for b, why_b in buckets(k):
            for v, why_f in forms(b):
                why = "; ".join(x for x in (why_b, why_f) if x)
                kk = key(v)
                if kk not in have and kk not in uniq:
                    uniq[kk] = derived(v, why)
    data["kernels"] = captured + list(uniq.values())
    data["generator"] = "tools/zig/glm_aot.py extract + derive"
    out.write_text(dump(data))
    print(f"{len(captured)} extracted, {len(uniq)} derived (row buckets, head blocks, chunk counts, row-int forms) "
          f"-> {out}")
    return 0


def world_merge(spec: Path, add: Path, out: Path) -> int:
    """``spec`` unchanged, then a world-1 capture's entries (kept as captured: build checks their cubins) and every bucket
    and row-int form derive gives them, where no entry takes the same launches; tp [1], ruled "world 1"."""
    data = json.loads(spec.read_text())
    have = {key(k) for k in data["kernels"]}
    captured = json.loads(add.read_text())["kernels"]
    new: dict[str, dict] = {}
    for k in captured:
        if key(k) not in have and key(k) not in new:
            new[key(k)] = dict(k, rule="world 1", tp=[1])
    for k in captured:
        for b, why_b in buckets(k):
            for v, why_f in forms(b):
                kk = key(v)
                if kk not in have and kk not in new:
                    new[kk] = dict(derived(v, "; ".join(x for x in ("world 1", why_b, why_f) if x)), tp=[1])
    data["kernels"] = data["kernels"] + list(new.values())
    data["generator"] = "tools/zig/glm_aot.py extract + derive + world1"
    out.write_text(dump(data))
    n_cap = sum(1 for k in new.values() if "cubin_sha256" in k)
    print(f"{len(data['kernels']) - len(new)} entries kept; world 1: {len(captured)} captured, {n_cap} new, "
          f"{len(new) - n_cap} derived -> {out}")
    return 0


def afd_cases(world: int):
    """The AFD attention node's own launch (world 1 only): hc_post over python 2.15's [3, R, D] planes, the two expert
    halves' partials then the shared expert (py215 forward.py:765), at every row count."""
    if world != 1:
        return
    import torch

    import glm_triton_cases as c
    from tensorfold.families.glm5_next.cuda import glue

    h, f32 = c.Dims(1), torch.float32
    for R in c.ROWS_ALL:
        x = c.z(R, h.s * h.d)
        yield f"hc_post/afd/r{R}", lambda x=x, R=R: glue.hc_post(x, x, c.z(3, R, h.d, dtype=f32), c.z(R, h.s, dtype=f32),
                                                                c.z(R, h.s * h.s, dtype=f32))


def dump(data: dict) -> str:
    """The spec as JSON with one kernel a line (diffs stay readable)."""
    head = {k: v for k, v in data.items() if k != "kernels"}
    rows = ",\n".join(" " + json.dumps(k, sort_keys=False) for k in data["kernels"])
    return json.dumps(head)[:-1] + ', "kernels": [\n' + rows + "\n]}\n"


def capture(out: Path, target: int, world: int, only: str) -> int:
    """The case list's launches compiled, never run, on a CPU for ``target``: launches.json in the capture's format."""
    import re

    from triton.backends.compiler import GPUTarget
    from triton.runtime.driver import driver
    from triton.runtime.jit import JITFunction

    class Utils:
        @staticmethod
        def get_device_properties(device):
            return {"max_shared_mem": 101376, "multiprocessor_count": 48}

    class CpuDriver:
        """A fixed target and no device: the JIT binds, specializes and compiles; nothing launches."""

        utils = Utils()

        def __init__(self) -> None:
            self.target = GPUTarget("cuda", target, 32)

        def is_active(self):
            return True

        def get_current_target(self):
            return self.target

        def get_current_device(self):
            return 0

        def get_current_stream(self, device=None):
            return 0

        def get_active_torch_device(self):
            return "cpu"

        def get_device_interface(self):
            raise RuntimeError("no device on a CPU compile box")

        def get_benchmarker(self):
            raise RuntimeError("no autotuning on a CPU compile box")

    driver.set_active(CpuDriver())
    sys.path.insert(0, str(TOOLS))
    import glm_triton_cases as cases_mod
    import triton_aot_manifest as manifest

    rec = manifest.Recorder()
    original = JITFunction.run

    def run(fn, *args, grid, warmup, **kwargs):
        kernel = original(fn, *args, grid=grid, warmup=True, **kwargs)
        rec._triton(fn, kernel, args, kwargs, grid)
        return kernel

    JITFunction.run = run
    pick = re.compile(only) if only else None
    n = 0
    for name, fn in itertools.chain(cases_mod.cases(world), afd_cases(world)):
        if pick is not None and not pick.search(name):
            continue
        with rec.scope(name.split("/")[0]):
            fn()
        n += 1
    out.mkdir(parents=True, exist_ok=True)
    rec.dump(out / "launches.json")
    print(f"capture sm_{target}: {n} cases, {len(rec.kernels)} specializations -> {out / 'launches.json'}")
    return 0
