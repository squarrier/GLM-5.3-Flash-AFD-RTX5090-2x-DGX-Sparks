#!/usr/bin/env python3
"""EXL3 expert launches as python-0.6.6 with Mia's 0128 makes them, recorded on a CPU: the fixtures of the Zig bindings'
launches with TF_GLM_EXL3_L2D / TF_GLM_EXL3_PDL on (zig/src/families/glm/fixtures_cuda_exl3_l2d.json).

exl3_fixtures.py's method on the oracle: python-0.6.6 (ed78d6f) with zig/tests/cuda/exl3_l2d.patch, i.e. 0128 by Mia
(MiaAI-Lab, GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold a687724, Apache-2.0) with its three rejected hunks finished by hand.
Its experts.py routed runs on CPU tensors with a recorder in place of the extension; the extension's own C++ wrappers
(experts.cpp/.cu, experts_cb0-2.cu as tensorfold_exl3_experts_v1_0128), compiled against tools/zig/exl3_launches (with
launch_ex.cpp for cudaLaunchKernelEx), run the calls once per (TF_EXL3X_L2D, TF_EXL3X_PDL) setting. Each launch is
named by the image the Zig bindings take it from: 0128's grouped kernels and gate/up epilogue from exl3_l2d_*, the
rest (SASS-equal to python-0.6.6's own) from today's images.
usage (PyTorch 26.07 image, the oracle tree at /tensorfold, this tree at /fork): python exl3_fixtures_l2d.py OUT.json WORK
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3_fixtures as fx  # noqa: E402  (imports the oracle's wrappers from /tensorfold/src)

MOD = "tensorfold_exl3_experts_v1_0128"
L2D = {"exl3_experts": "exl3_l2d_experts", "exl3_grouped_cb0": "exl3_l2d_grouped_cb0",
       "exl3_grouped_cb1": "exl3_l2d_grouped_cb1", "exl3_grouped_cb2": "exl3_l2d_grouped_cb2"}
SETTINGS = [(4, 1), (0, 0), (0, 1), (2, 0), (2, 1), (4, 0), (8, 0), (8, 1)]   # 0128's own default first
HARNESS = ["tools/zig/exl3_launches/exl3_launches.cpp", "tools/zig/exl3_launches/fake/tf_fake.h",
           "tools/zig/exl3_launches/launch_ex.cpp"]


def zero128(sym: str) -> bool:
    """A kernel 0128 changed: launched from exl3/l2d/'s image."""
    k = fx.key(sym)
    return k.startswith("14grouped_kernel") or k.startswith("22gateup_epilogue_kernel")


def cases(setting) -> list[dict]:
    fx.CASES.clear()
    g8 = ([8] * 288, [8] * 288, [8] * 288)
    # GLM-5.3-Flash's TP2 and TP1 decode (the microbench's rows), every setting
    fx.routed_case("glm-tp2-decode", 288, 4096, 1024, 1, g8, 8, 9, [(r, True) for r in (1, 2, 3, 4, 5, 8)])
    fx.routed_case("glm-tp1-decode", 288, 4096, 2048, 1, g8, 8, 9, [(r, True) for r in (1, 2, 8)])
    if setting == SETTINGS[0]:
        fx.routed_case("glm-tp2-prompt", 288, 4096, 1024, 1, g8, 2048, 9,
                       [(r, True) for r in (9, 16, 17, 64, 127, 128, 256, 512, 1024, 1361, 1362, 1536, 2048, 1362)])
        fx.routed_case("glm-tp2-regroup", 288, 4096, 1024, 1, g8, 8, 9, [(4, True), (4, False)])
        fx.routed_case("glm-tp2-weighted", 288, 4096, 1024, 1, g8, 8, 9, [(1, True), (8, True)], weighted=True,
                       act=fx.ex3.ACT_F32)
        # every grouped_kernel instance of 0128's: 3 codebooks x its 4 tile settings (with 4 4 1) x 3 width ranges
        tiles = [(8, 4, 1, 1), (8, 4, 2, 2), (4, 4, 2, 2), (4, 4, 1, 1)]
        ranges = [[8, 8, 8, 8], [2, 4, 6, 10], [3, 8, 16, 12]]
        for cb in range(3):
            for ti in range(4):
                for ri in range(3):
                    k2s = (ranges[ri], ranges[ri][::-1], ranges[(ri + 1) % 3])
                    fx.routed_case(f"mixed-cb{cb}-t{ti}-r{ri}", 4, 1024, 512, cb, k2s, 4, 5, [(1, True), (4, True)],
                                   cfg_gu=tiles[ti], cfg_d=tiles[(ti + 1) % 4], weighted=ri == 1)
    out = list(fx.CASES)
    for c in out:
        c["id"] = f"{c['id']}@l2d{setting[0]}-pdl{setting[1]}"
        c["inputs"]["sched"] = {"l2d": setting[0], "pdl": setting[1]}
    return out


def build(work: Path) -> Path:
    fake = fx.FORK / "tools/zig/exl3_launches/fake"
    flags = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
             "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-std=c++20"]
    e = fx.PY / "cuda/exl3"
    units = [e / "experts.cu", e / "experts_cb0.cu", e / "experts_cb1.cu", e / "experts_cb2.cu", e / "experts.cpp",
             fx.FORK / "tools/zig/exl3_launches/exl3_launches.cpp", fx.FORK / "tools/zig/exl3_launches/launch_ex.cpp"]
    objs = []
    for i, src in enumerate(units):
        obj = work / f"{i}-{src.stem}.o"
        objs.append(str(obj))
        inc = [f"-I{fake}", f"-I{e}"]
        if src.suffix == ".cu":
            cmd = ["nvcc", "-c", *flags, "-O1", "-gencode=arch=compute_121,code=compute_121", *inc,
                   f"-DTORCH_EXTENSION_NAME={MOD}", str(src), "-o", str(obj)]
        else:
            cmd = ["g++", "-c", "-std=c++20", "-O1", *inc, "-I/usr/local/cuda/include", f"-DTORCH_EXTENSION_NAME={MOD}",
                   str(src), "-o", str(obj)]
        subprocess.run(cmd, check=True)
    exe = work / "exl3_launches_0128"
    subprocess.run(["g++", "-o", str(exe), *objs], check=True)
    return exe


def calls_text(cs: list[dict]) -> str:
    lines = []
    for c in cs:
        lines.append(f"CASE {c['id']}\nOPTIN {fx.OPTIN}")
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
    return "\n".join(lines) + "\n"


def parse(got: str, cs: list[dict], ours: dict) -> int:
    """exl3_fixtures.main's reading, with each LAUNCHATTR line (launch_ex's attribute) put on the launch after it."""
    by_case: dict[str, list[list[str]]] = {}
    cur = None
    for line in got.splitlines():
        w = line.split()
        if w[0] == "CASE":
            cur = by_case.setdefault(w[1], [])
        else:
            cur.append(w)
    errors, total = [], 0
    for c in cs:
        events = by_case[c["id"]]
        call_no = 0
        for s in c["steps"]:
            first, s["launches"] = call_no, []
            call_no += len(s["calls"])
            pending, attr = {}, None
            for w in events:
                if w[0] == "LAUNCHATTR":
                    if int(w[2]) != 1:
                        errors.append(f"{c['id']}: launch attribute {w[1]} = {w[2]}")
                    attr = int(w[2])
                    continue
                n, a, attr = int(w[1]), attr, None
                if not first <= n < call_no:
                    continue
                if w[0] == "ERROR":
                    errors.append(f"{c['id']}: {' '.join(w[2:])}")
                    continue
                image, sym, _ = ours[fx.key(w[2])]
                if w[0] == "ATTR":
                    pending[sym] = int(w[4])
                elif w[0] == "LAUNCH":
                    rec = {"image": image, "symbol": sym, "grid": [int(v) for v in w[3:6]],
                           "block": [int(v) for v in w[6:9]], "shared": int(w[9]),
                           "max_dynamic": pending.pop(sym, None), "params": w[11:11 + int(w[10])]}
                    if a is not None:
                        rec["pdl"] = a
                    s["launches"].append(rec)
                    total += 1
            del s["calls"]
    if errors:
        raise SystemExit("the C++ wrappers refused calls:\n" + "\n".join(errors[:20]))
    return total


def main() -> int:
    out, work = Path(sys.argv[1]), Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    fx.MODULES["experts"] = MOD
    kernels = json.loads((fx.FORK / "zig/src/families/glm/exl3_kernels.json").read_text())
    ours: dict[str, tuple[str, str, dict]] = {}
    with open(work / "params.txt", "w") as f:
        for today, l2d in L2D.items():
            for k in kernels[l2d]:
                image = l2d if zero128(k["symbol"]) else today
                if image == today:   # the binding's symbol in today's image, its parameters the same
                    assert any(t == k for t in kernels[today]), k["symbol"]
                ours[fx.key(k["symbol"])] = (image, k["symbol"], k)
                sizes = " ".join(str(p[1]) for p in k["params"])
                f.write(f"{MOD} {fx.key(k['symbol'])} {len(k['params'])} {sizes} {k['shared']}\n")
    exe = build(work)
    every, total = [], 0
    for setting in SETTINGS:
        cs = cases(setting)
        name = f"l2d{setting[0]}-pdl{setting[1]}"
        (work / f"calls-{name}.txt").write_text(calls_text(cs))
        env = dict(os.environ, TF_EXL3X_L2D=str(setting[0]), TF_EXL3X_PDL=str(setting[1]))
        got = subprocess.run([str(exe), str(work / "params.txt")], input=(work / f"calls-{name}.txt").read_text(),
                             capture_output=True, text=True, check=True, env=env).stdout
        (work / f"launches-{name}.txt").write_text(got)
        total += parse(got, cs, ours)
        every += cs
    launched = {(x["image"], x["symbol"]) for c in every for s in c["steps"] for x in s["launches"]}
    head = {"source": "python-0.6.6 (ed78d6f) cuda/exl3 experts.py with Mia's 0128 (MiaAI-Lab, "
                      "GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold a687724; zig/tests/cuda/exl3_l2d.patch) and its "
                      "extension's C++ wrappers, run on a CPU by tools/zig/exl3_fixtures_l2d.py, once per "
                      "TF_EXL3X_L2D/TF_EXL3X_PDL setting",
            "patch_sha256": hashlib.sha256((fx.FORK / "zig/tests/cuda/exl3_l2d.patch").read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256(b"".join((fx.FORK / p).read_bytes() for p in HARNESS)).hexdigest(),
            "settings": [list(s) for s in SETTINGS], "kernels_launched": len(launched), "case_count": len(every),
            "launches": total}
    body = ",\n".join(json.dumps(c, separators=(",", ":")) for c in every)
    out.write_text(json.dumps(head, indent=1)[:-2] + ',\n "cases": [\n' + body + "\n ]\n}\n")
    print(f"{len(every)} cases, {total} launches, {len(launched)} kernels launched -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
