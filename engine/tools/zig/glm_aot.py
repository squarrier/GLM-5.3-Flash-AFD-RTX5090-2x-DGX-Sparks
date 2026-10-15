#!/usr/bin/env python3
"""GLM-5.3-Flash's Triton kernel set built on a CPU for sm_121a (GB10) or sm_120a (RTX 5090): spec, build, compare."""
# Adapted from tools/zig/flashnext_aot.py, written by Mia (MiaAI-Lab, @MiaAI_lab) in
# MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429 (patches/0008-kernel-spec-tools.patch), Apache 2.0.
# Changed for GLM-5.3-Flash: --target 121|120, the CPU capture of a case list (glm_aot_rules.capture), GLM's derive
# rules (TP=2 rows, row buckets, prompt chunks, latent and sparse chunk counts), the fixture cover and compare steps.

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_aot_rules as rules  # noqa: E402

TARGET = ("cuda", 121, 32)
OPTION_KEYS = ("num_warps", "num_ctas", "num_stages", "warp_size", "maxnreg", "ptx_version", "ptx_options",
               "ir_override", "enable_fp_fusion", "enable_reflect_ftz", "launch_cooperative_grid", "launch_pdl",
               "supported_fp8_dtypes", "deprecated_fp8_dot_operand_dtypes", "default_dot_input_precision",
               "allowed_dot_input_precisions", "max_num_imprecise_acc_default", "extern_libs", "debug",
               "backend_name", "sanitize_overflow", "arch", "instrumentation_mode")
DTYPES = ("int1", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "fp8e4nv", "fp8e5",
          "fp8e4b15", "fp8e4b8", "fp8e5b16", "fp16", "bf16", "fp32", "fp64", "void")


def typed(v):
    """A Recorder constexpr (``_jsonable``) -> the spec's typed form."""
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if v is None:
        return {"none": True}
    if isinstance(v, dict) and "fp64_bits" in v:
        return {"float": v["float"], "fp64_bits": v["fp64_bits"]}
    if isinstance(v, str) and v in DTYPES:
        return {"dtype": v}
    if isinstance(v, str) and v.startswith("JITFunction("):
        return {"jit": v[len("JITFunction("):-1]}
    if isinstance(v, list):
        return {"tuple": [typed(x) for x in v]}
    return {"str": str(v)}


def extract(launches: Path, cache: Path, mount: str, out: Path, world: int) -> int:
    """A capture's Recorder dump and Triton cache (GPU, or ``capture`` on a CPU) -> spec entries with their cubins."""
    rec = json.loads(launches.read_text())
    kernels, targets = [], set()
    for h, info in sorted(rec["kernels"].items(), key=lambda kv: (kv[1]["function"], kv[0])):
        meta_path = next(p for leaf, p in info["files"].items()
                         if leaf.endswith(".json") and not leaf.startswith("__grp__"))
        meta_path = cache / meta_path[len(mount):].lstrip("/") if meta_path.startswith(mount) else Path(meta_path)
        meta = json.loads(meta_path.read_text())
        targets.add(json.dumps(meta.get("target")))
        src = info["source"]
        kernels.append({"function": info["function"], "name": info["name"], "hash": h,
                        "source": {"file": src["file"].split("/src/", 1)[-1], "line": src["line"]},
                        "signature": info["signature"], "params": info["params"],
                        "constexprs": {k: typed(v) for k, v in info["constexprs"].items()},
                        "attrs": dict(info["attrs"].items()),
                        "options": {k: meta[k] for k in OPTION_KEYS if k in meta}, "tp": [world],
                        "cubin_sha256": hashlib.sha256(meta_path.with_suffix(".cubin").read_bytes()).hexdigest()})
    roots = {info["source"]["file"].rsplit("/tensorfold/", 1)[0] for info in rec["kernels"].values()}
    target = json.loads(targets.pop()) if len(targets) == 1 else None
    spec = {"generator": "tools/zig/glm_aot.py extract", "capture_target": target,
            "source_root": sorted(roots)[0] if len(roots) == 1 else sorted(roots), "kernels": kernels}
    out.write_text(rules.dump(spec))
    print(f"{len(kernels)} captured specializations (target {target}) -> {out}")
    return 0


def zig_const(v):
    """A spec constexpr as zig/src/cuda/aot.zig compares it: ints (bools too) and fp32 bits; others are not compared."""
    if "bool" in v:
        return {"int": int(v["bool"])}
    if "int" in v:
        return {"int": v["int"]}
    if "fp64_bits" in v:
        x = struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]
        return {"f32": struct.unpack("<I", struct.pack("<f", x))[0]}
    if "none" in v:
        return {"none": True}
    return {"str": json.dumps(v)}


def fixture_matches(k: dict, launch: dict) -> bool:
    """aot.zig's ``matches`` on a fixture launch: constexprs, then each runtime argument's type and specialization."""
    if k["name"] != launch["fn"]:
        return False
    zc = {n: zig_const(v) for n, v in k["constexprs"].items()}
    for n, v in launch["consts"].items():
        if not isinstance(v, dict):
            continue
        got = zc.get(n)
        if got is None or any(got.get(t) != v.get(t) for t in ("int", "f32") if t in v):
            return False
    runtime = 0
    params = {n for n in k["params"] if k["signature"].get(n) != "constexpr"}
    for a in launch["args"]:
        n, ty = a["name"], a["type"]
        div = bool(k["attrs"].get(n))
        if ty.startswith("*"):
            if n not in params or k["signature"][n] != ty or div != a.get("align16", True):
                return False
        elif "int" in a:
            x = a["int"]
            if n not in params:
                if x != 1 or zc.get(n, {}).get("int") != 1:
                    return False
                continue
            if k["signature"][n] != ty or x == 1 or div != (x % 16 == 0):
                return False
        elif n not in params or k["signature"][n] != ty:
            return False
        runtime += 1
    return runtime == len(params)


def from_fixture(t: dict, launch: dict, case: str) -> dict:
    """A spec entry for a fixture launch, on a template of the same kernel: its types, int forms, constexprs, warps."""
    out = json.loads(json.dumps(t))
    for drop in ("hash", "cubin_sha256"):
        out.pop(drop, None)
    for a in launch["args"]:
        n = a["name"]
        if "int" in a and a["int"] == 1:
            out["signature"][n], out["constexprs"][n], out["attrs"][n] = "constexpr", {"int": 1}, []
            continue
        out["signature"][n] = a["type"]
        out["constexprs"].pop(n, None)
        if a["type"].startswith("*"):
            out["attrs"][n] = [["tt.divisibility", 16]] if a.get("align16", True) else []
        elif "int" in a:
            out["attrs"][n] = [["tt.divisibility", 16]] if a["int"] % 16 == 0 else []
    for n, v in launch["consts"].items():
        if not isinstance(v, dict):
            continue
        old = t["constexprs"].get(n, {})
        if "f32" in v:
            if old and zig_const(old).get("f32") == v["f32"]:
                continue
            x = struct.unpack("<f", struct.pack("<I", v["f32"]))[0]
            out["constexprs"][n] = {"float": x, "fp64_bits": "0x%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]}
            continue
        out["constexprs"][n] = {"bool": bool(v["int"])} if "bool" in old else {"int": v["int"]}
    if launch.get("num_warps") is not None:
        out["options"]["num_warps"] = launch["num_warps"]
    if launch.get("num_stages") is not None:
        out["options"]["num_stages"] = launch["num_stages"]
    out["expanded_from"] = t.get("hash") or t.get("expanded_from")
    out["rule"] = f"fixture {case}"
    return rules.reorder(out)


def cover(spec: Path, fixtures: Path, out: Path, strict: bool) -> int:
    """Every launch of the fixtures (glm_triton_fixtures.py) must find a variant: misses are added, or fail strict."""
    data = json.loads(spec.read_text())
    kernels = data["kernels"]
    cases = json.loads(fixtures.read_text())["cases"]
    added, missing, total = [], 0, 0
    by_name: dict[str, list] = {}
    for k in kernels:
        by_name.setdefault(k["name"], []).append(k)
    for case, launches in cases.items():
        for launch in launches:
            total += 1
            pool = by_name.get(launch["fn"], []) + [k for k in added if k["name"] == launch["fn"]]
            if any(fixture_matches(k, launch) for k in pool):
                continue
            missing += 1
            same = [k for k in by_name.get(launch["fn"], []) if {a["name"] for a in launch["args"]} <= set(k["params"])]
            if not same:
                print(f"NO TEMPLATE {case} {launch['fn']}")
                continue
            best = max(same, key=lambda k: sum(zig_const(k["constexprs"].get(n, {"none": True})) == v
                                               for n, v in launch["consts"].items() if isinstance(v, dict)))
            new = from_fixture(best, launch, case)
            if not fixture_matches(new, launch):
                raise SystemExit(f"{case}: the synthesized entry does not match its launch")
            added.append(new)
            print(f"MISS {case} {launch['fn']}: added from {best.get('hash', best.get('expanded_from', '?'))[:12]}")
    data["kernels"] = kernels + added
    out.write_text(rules.dump(data))
    print(f"{total} fixture launches in {len(cases)} cases, {missing} without a variant, {len(added)} entries added "
          f"-> {out}")
    return 1 if strict and missing else 0


def value(v, tl, registry):
    if "bool" in v:
        return v["bool"]
    if "int" in v:
        return v["int"]
    if "none" in v:
        return None
    if "fp64_bits" in v:
        return struct.unpack("<d", struct.pack("<Q", int(v["fp64_bits"], 16)))[0]
    if "dtype" in v:
        return tl.dtype(v["dtype"])
    if "jit" in v:
        return registry(v["jit"])
    if "tuple" in v:
        return tuple(value(x, tl, registry) for x in v["tuple"])
    return v["str"]


def resolve(function: str):
    """``package.module.qualname`` -> the JITFunction object (the longest importable module prefix)."""
    parts = function.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise SystemExit(f"cannot import {function}")


def compile_one(k: dict):
    """``triton.compile`` of one entry for TARGET; the options' arch is the target's (a spec may come from another)."""
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    fn = resolve(k["function"])
    params = k["params"]
    registry = lambda key: resolve(key.replace(":", "."))       # noqa: E731
    path = lambda n: (params.index(n.split(".")[0]), *map(int, n.split(".")[1:]))  # noqa: E731
    constexprs = {path(n): value(v, tl, registry) for n, v in k["constexprs"].items()}
    attrs = {path(n): v for n, v in k["attrs"].items()}
    options = {n: tuple(v) if isinstance(v, list) else v for n, v in k["options"].items()}
    options["arch"] = f"sm{TARGET[1]}"
    options["extern_libs"] = tuple(tuple(x) for x in options.get("extern_libs") or ()) or None
    return compile(ASTSource(fn, dict(k["signature"]), constexprs, attrs), target=GPUTarget(*TARGET), options=options)


class Compiled:
    """What a worker sends back: the kernel hash, its cubin and metadata (picklable)."""

    def __init__(self, ck) -> None:
        self.hash, self.cubin = ck.hash, ck.asm["cubin"]
        self.metadata = {k: v for k, v in ck.metadata._asdict().items() if isinstance(v, (int, str, bool, float))}
        self.n_regs, self.n_spills = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)


def compile_job(k: dict):
    try:
        return Compiled(compile_one(k))
    except Exception as exc:                                      # noqa: BLE001  (listed, then the run fails)
        return f"{type(exc).__name__}: {str(exc)[:300]}"


def aot_row(k: dict, ck: Compiled) -> dict:
    """One aot.json row: launch facts zig/src/cuda/aot.zig loads, and what the cubin is."""
    md = ck.metadata
    div = {n for n, v in k["attrs"].items() if v}
    return {"fn": md["name"], "hash": ck.hash, "name": md["name"], "num_warps": md["num_warps"],
            "num_ctas": md.get("num_ctas", 1), "shared": md.get("shared", 0),
            "global_scratch": md.get("global_scratch_size", 0), "global_align": md.get("global_scratch_align", 1),
            "profile_scratch": md.get("profile_scratch_size", 0), "pdl": bool(md.get("launch_pdl", False)),
            "params": [{"name": n, "type": k["signature"][n], "div16": n in div, "nospec": False}
                       for n in k["params"] if k["signature"].get(n) != "constexpr"],
            "consts": {n: zig_const(v) for n, v in k["constexprs"].items()},
            "function": k["function"], "tp": k.get("tp", [2]), "cubin_sha256": hashlib.sha256(ck.cubin).hexdigest(),
            "n_regs": ck.n_regs, "n_spills": ck.n_spills}


def build(spec: Path, out: Path, check: list[Path], jobs: int) -> int:
    """Every entry compiled for TARGET -> out/aot.json + out/cubins/<hash>.cubin; problems fail the run."""
    data = json.loads(spec.read_text())
    root = data.get("source_root")
    import tensorfold

    here = str(Path(tensorfold.__file__).parent.parent)
    if root and here != root:
        raise SystemExit(f"the spec was captured with the Python source at {root}, this run imports it from {here}: "
                         f"mount the tree so that {root}/tensorfold is the package (the cubins embed the path)")
    same_target = data.get("capture_target", {}).get("arch") == TARGET[1] if data.get("capture_target") else False
    captured = {}
    for m in check:
        for k in json.loads(m.read_text())["kernels"]:
            captured[k["hash"]] = k["cubin_sha256"]
    (out / "cubins").mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        done = list(pool.map(compile_job, data["kernels"], chunksize=1))
    rows, problems, by_hash, launch_keys, same = [], [], {}, {}, 0
    for k, got in zip(data["kernels"], done):
        if isinstance(got, str):
            problems.append(f"{k['function']} {k.get('hash', '?')[:12]}: {got}")
            continue
        sha = hashlib.sha256(got.cubin).hexdigest()
        want = (k.get("cubin_sha256") if same_target else None) or captured.get(k.get("hash", ""))
        if want and sha != want:
            problems.append(f"{k['function']} {k['hash'][:12]}: cubin sha256 {sha[:12]} != captured {want[:12]}")
        elif want:
            same += 1
        if got.hash in by_hash:
            by_hash[got.hash]["specs"].append(rules.spec_key(k))
            continue
        row = aot_row(k, got)
        row["specs"] = [rules.spec_key(k)]  # the spec entries it serves (compare's match across hosts)
        by_hash[got.hash] = row
        if row["global_scratch"] or row["profile_scratch"]:
            problems.append(f"{row['fn']} {got.hash[:12]}: global or profile scratch (aot.zig loads neither)")
        lk = json.dumps([row["fn"], row["params"], row["consts"]], sort_keys=True)
        if lk in launch_keys:
            problems.append(f"{row['fn']}: {got.hash[:12]} and {launch_keys[lk][:12]} take the same launches")
        launch_keys[lk] = got.hash
        (out / "cubins" / f"{got.hash}.cubin").write_bytes(got.cubin)
        rows.append(row)
    rows.sort(key=lambda x: (x["fn"], x["hash"]))
    (out / "aot.json").write_text(json.dumps({"generator": "tools/zig/glm_aot.py", "target": list(TARGET),
                                              "kernels": rows}, indent=1) + "\n")
    for p in problems:
        print("PROBLEM", p)
    print(f"sm_{TARGET[1]}: {len(rows)} kernels from {len(data['kernels'])} entries -> {out}; {same} cubins equal to "
          f"the capture's" + (f"; {len(problems)} problems" if problems else ""))
    return 1 if problems else 0


def compare(built: Path, capture: Path) -> int:
    """A GPU capture's extracted spec against our build: each captured specialization built, with the same cubin.
    Matched by rules.spec_key, not by Triton's kernel hash, which differs between hosts (qualification_4cb8c4ff NOTES 3)."""
    ours = {}
    for r in json.loads(built.read_text())["kernels"]:
        for s in r["specs"]:
            ours[s] = r
    equal, differ, absent = 0, [], []
    for k in json.loads(capture.read_text())["kernels"]:
        r = ours.get(rules.spec_key(k))
        if r is None:
            absent.append(k)
        elif r["cubin_sha256"] == k["cubin_sha256"]:
            equal += 1
        else:
            differ.append((k, r))
    for k in absent:
        print(f"ABSENT {k['function']} {k['hash'][:12]} {json.dumps(rules.ints(k))}")
    for k, r in differ:
        print(f"DIFFERS {k['function']} {k['hash'][:12]}: {k['cubin_sha256'][:12]} != ours {r['cubin_sha256'][:12]}")
    print(f"{equal} equal, {len(differ)} differ, {len(absent)} absent of {equal + len(differ) + len(absent)} captured")
    return 0 if not differ and not absent else 1


def main() -> int:
    global TARGET
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, choices=(120, 121), default=121, help="SASS target: 121 GB10, 120 RTX 5090")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="compile the case list's launches on this CPU (no GPU): launches.json")
    c.add_argument("--out", required=True)
    c.add_argument("--world", type=int, default=2)
    c.add_argument("--only", default="", help="a regular expression over case names")
    x = sub.add_parser("extract", help="a capture's launches.json + Triton cache -> spec")
    x.add_argument("--launches", required=True)
    x.add_argument("--cache", required=True, help="the capture's TRITON_CACHE_DIR as seen here")
    x.add_argument("--mount", default="/aot/triton", help="TRITON_CACHE_DIR as the capture saw it")
    x.add_argument("--world", type=int, default=2)
    x.add_argument("--out", required=True)
    d = sub.add_parser("derive", help="spec -> spec + row buckets, head blocks, chunk counts, row-int forms")
    d.add_argument("--spec", required=True)
    d.add_argument("--out", required=True)
    w = sub.add_parser("world1", help="spec + a world-1 capture's extract -> spec + the AFD attention node's variants")
    w.add_argument("--spec", required=True)
    w.add_argument("--add", required=True, help="`extract --world 1` of `capture --world 1`")
    w.add_argument("--out", required=True)
    v = sub.add_parser("cover", help="every fixture launch must find a variant")
    v.add_argument("--spec", required=True)
    v.add_argument("--fixtures", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--strict", action="store_true", help="a launch without a variant fails the run")
    b = sub.add_parser("build", help="spec -> aot.json + cubins/ for --target")
    b.add_argument("--spec", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--check", action="append", default=[], help="a capture's manifest.json (repeatable)")
    b.add_argument("--jobs", type=int, default=8)
    m = sub.add_parser("compare", help="a GPU capture's extracted spec against our build (specializations, cubins)")
    m.add_argument("--built", required=True, help="aot.json of `build` for the capture's target")
    m.add_argument("--capture", required=True, help="`extract` of the capture")
    a = ap.parse_args()
    TARGET = ("cuda", a.target, 32)
    if a.cmd == "capture":
        return rules.capture(Path(a.out), a.target, a.world, a.only)
    if a.cmd == "extract":
        return extract(Path(a.launches), Path(a.cache), a.mount, Path(a.out), a.world)
    if a.cmd == "derive":
        return rules.derive(Path(a.spec), Path(a.out))
    if a.cmd == "world1":
        return rules.world_merge(Path(a.spec), Path(a.add), Path(a.out))
    if a.cmd == "cover":
        return cover(Path(a.spec), Path(a.fixtures), Path(a.out), a.strict)
    if a.cmd == "compare":
        return compare(Path(a.built), Path(a.capture))
    return build(Path(a.spec), Path(a.out), [Path(p) for p in a.check], a.jobs)


if __name__ == "__main__":
    sys.exit(main())
