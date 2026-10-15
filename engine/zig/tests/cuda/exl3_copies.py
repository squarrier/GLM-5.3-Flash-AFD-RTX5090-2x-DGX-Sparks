"""Checks (or rewrites) zig/kernels/cuda's copies of python-0.6's EXL3 device code: same lines, same bits.

copies.py's rules (comments and ATen includes dropped, the anonymous namespace named), with line ranges, for TensorFold's
shared EXL3 module (src/tensorfold/cuda/exl3: decode.cuh, linear.cu, experts_grouped.cuh, experts.cu) and GLM-5.3-Flash's
exl3.cu. The template instances each copy compiles are read from the host code that launches them (TF_EXL3_ALL,
TF_RANGES, TF_DQ, LAUNCH), so a new instance in python-0.6 shows up as drift. Sources come from a git ref (default
`python-0.6`, or `--ref REF`) or a checkout (`--tree DIR`). The EXL3 format and kernels follow ExLlamaV3 (turboderp,
MIT); TensorFold's EXL3 module is by Ash Hart and the TensorFold contributors.

exl3/l2d/ holds the same two files with Mia's 0128 applied (exl3_l2d.patch beside this file: L2 prefetch and
programmatic dependent launch in the grouped expert kernel, by Mia, MiaAI-Lab, @MiaAI_lab; GLM-5.3-EXL3-3x-DGX-Sparks-
TensorFold a687724, Apache-2.0), copied by the same rules from python-0.6's text with that patch applied in memory.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import copies  # noqa: E402  copies.py beside this file: its source() reads a ref or a tree

EXL3 = "src/tensorfold/cuda/exl3/"
GLM = "src/tensorfold/families/glm5_next/cuda/"
CHECKER = "zig/tests/cuda/exl3_copies.py"
L2D_PATCH = Path(__file__).resolve().parent / "exl3_l2d.patch"
# exl3/x1/: IMP2's prompt switches (Mia's 0001, 0020, 0009 adapted onto the universal kernel; exl3_x1.patch's header)
X1_PATCH = Path(__file__).resolve().parent / "exl3_x1.patch"
# qualification_3e1f1105: bertholomus's parallel grouping (glm-dsa-tp4 dbe0773, Apache-2.0) in exl3/experts.cu and exl3/x1/experts.cu
PGROUP_PATCH = Path(__file__).resolve().parent / "exl3_pgroup.patch"


def patched(src: str, text: str, patch: Path = L2D_PATCH) -> str:
    """`text` (python-0.6's file `src`) with `patch`'s hunks for it applied: exact context, no fuzz."""

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    at = 0
    for part in re.split(r"(?m)^--- a/", patch.read_text())[1:]:
        if not part.startswith(src + "\n"):
            continue
        for hunk in re.split(r"(?m)^(?=@@ )", part)[1:]:
            head, *rows = hunk.splitlines(keepends=True)
            start = int(re.match(r"@@ -(\d+)", head).group(1)) - 1
            old = [r[1:] for r in rows if r[:1] in (" ", "-")]
            if start < at or lines[start:start + len(old)] != old:
                raise SystemExit(f"{patch.name} does not apply to {src} at line {start + 1}; update it")
            out += lines[at:start] + [r[1:] for r in rows if r[:1] in (" ", "+")]
            at = start + len(old)
        return "".join(out + lines[at:])
    raise SystemExit(f"{patch.name} has no hunks for {src}")


def source(src: str, ref: str, tree: Path | None, l2d: bool | Path | tuple) -> str:
    """python-0.6's `src`, with exl3_l2d.patch (l2d True) or the patch `l2d` names (a tuple: each, in order) applied."""

    text = copies.source(src, ref, tree)
    if not l2d:
        return text
    for p in l2d if isinstance(l2d, tuple) else (L2D_PATCH if l2d is True else l2d,):
        text = patched(src, text, p)
    return text


def prompt_instances(ref: str, tree: Path | None) -> dict:
    """The prompt switches' instances, read from prompt_grouped_launch in experts_grouped.cuh with exl3_x1.patch: its
    fixed tile (CB, NT, W, PF, LO, HI) and each case's selector (NFIRST 8, ORDER 4, PA 2, XROW 1)."""

    grp = source(EXL3 + "experts_grouped.cuh", ref, tree, X1_PATCH)
    tile = re.search(r"grouped_kernel<CB, (\d+), (\d+), (\d+), (\d+), (\d+), \(S_ & 8\) != 0, \(S_ & 4\) != 0, "
                     r"\(S_ & 2\) != 0, \(S_ & 1\) != 0>", grp)
    cb = re.search(r"if constexpr \(CB == (\d+)\) prompt_grouped_launch<CB>", grp)
    sels = [int(s) for s in re.findall(r"TF_PROMPT\((\d+)\)", grp)]
    if tile is None or cb is None or not sels:
        raise SystemExit(f"no prompt_grouped_launch instances found with {X1_PATCH.name}: update {CHECKER}")
    return {"tile": (int(cb.group(1)), *(int(v) for v in tile.groups())), "sels": sels}


def instances(ref: str, tree: Path | None, l2d: bool = False) -> dict:
    """Each launch site's template arguments, read from the python-0.6 host code that launches them."""

    lin = copies.source(EXL3 + "linear.cu", ref, tree)
    widths = re.search(r"#define TF_EXL3_WIDTHS\(X, CB\)(.*)", lin).group(1)
    every = re.search(r"#define TF_EXL3_ALL\(X\)(.*)", lin).group(1)
    pairs = [(int(k2), int(cb)) for cb in re.findall(r"TF_EXL3_WIDTHS\(X, (\d+)\)", every)
             for k2 in re.findall(r"X\((\d+), CB\)", widths)]
    pairs += [(int(k2), int(cb)) for k2, cb in re.findall(r"X\((\d+), (\d+)\)", every)]
    wks = sorted({int(w) for w in re.findall(r"linear_kernel<K2_, CB_, (\d+)>", lin)})
    grp = source(EXL3 + "experts_grouped.cuh", ref, tree, l2d)
    tiles = [tuple(int(v) for v in t) for t in re.findall(r"\{ TF_RANGES\((\d+), (\d+), (\d+)\) \}", grp)]
    ranges = [tuple(int(v) for v in r) for r in re.findall(r"TF_LAUNCH\(NT_, W_, PF_, (\d+), (\d+)\)", grp)]
    k2s = [int(k) for k in re.findall(r"TF_DQ\((\d+)\)", grp)]
    glm = copies.source(GLM + "exl3.cu", ref, tree)
    glm_tiles = [tuple(int(v) for v in t) for t in re.findall(r"LAUNCH\((\d+), (\d+)\);", glm)]
    got = {"linear": pairs, "wk": wks, "tiles": tiles, "ranges": ranges, "dequant": k2s, "glm": glm_tiles}
    for k, v in got.items():
        if not v:
            raise SystemExit(f"no {k} instances found in python-0.6's host code: update {CHECKER}")
    return got


def linear_footer(i: dict) -> str:
    out = ["// The instances exl3_linear_cuda and exl3_unpack_cuda launch (TF_EXL3_ALL: width K2 and codebook, WK warps).\n",
           "#define TF_LINEAR(K2, CB, WK) template __global__ void tf_exl3_linear::linear_kernel<K2, CB, WK>( \\\n",
           "    const half*, const uint32_t*, long long, long long, const half*, const half*, void*, int, float*, int*, int, \\\n",
           "    int, int, int);\n",
           "#define TF_UNPACK(K2, CB) template __global__ void tf_exl3_linear::unpack_kernel<K2, CB>(const uint32_t*, half*, \\\n",
           "    int, int64_t, int64_t);\n"]
    for k2, cb in i["linear"]:
        out.append(" ".join(f"TF_LINEAR({k2}, {cb}, {w})" for w in i["wk"]) + f" TF_UNPACK({k2}, {cb})\n")
    return "".join(out)


EXPERTS_FOOTER = (
    "// rot_in_kernel's two input types (exl3x_rot_in_cuda: bf16 or fp16 x); the other six kernels are plain.\n"
    "#define TF_ROT_IN(T) template __global__ void tf_exl3_experts::rot_in_kernel<T>(const T*, int, const int*, \\\n"
    "    const half*, const half*, half*, half*, int, int, int);\n"
    "TF_ROT_IN(__nv_bfloat16)\n"
    "TF_ROT_IN(half)\n"
)

X1_EXPERTS_FOOTER = (
    "// rot_in_kernel's instances: exl3x_rot_in_cuda's (ROWS false) and exl3x_rot_rows_cuda's (ROWS true), bf16 or fp16\n"
    "// x; prompt_order_kernel and the other six kernels are plain.\n"
    "#define TF_ROT_IN(T, ROWS) template __global__ void tf_exl3_experts::rot_in_kernel<T, ROWS>(const T*, int, \\\n"
    "    const int*, const half*, const half*, half*, half*, int, int, int);\n"
    "TF_ROT_IN(__nv_bfloat16, false)\n"
    "TF_ROT_IN(half, false)\n"
    "TF_ROT_IN(__nv_bfloat16, true)\n"
    "TF_ROT_IN(half, true)\n"
)

# qualification_3e1f1105: the copies with exl3_pgroup.patch hold two more plain kernels (group_count_kernel, group_place_kernel)
PG_FOOTER = EXPERTS_FOOTER.replace("the other six kernels", "the other eight kernels")
X1_PG_FOOTER = X1_EXPERTS_FOOTER.replace("the other six kernels", "the other eight kernels")


def grouped_x1(p: dict) -> str:
    """exl3/x1/grouped_cb1.cu: the instances prompt_grouped_launch<1> launches (its one tile, each switch selector)."""

    cb, nt, w, pf, lo, hi = p["tile"]
    b = lambda s, bit: "true" if s & bit else "false"  # noqa: E731
    out = [f"// The kernels {EXL3}experts_cb{cb}.cu with {X1_PATCH.name} compiles for the prompt switches (codebook {cb}, "
           f"mcg): prompt_grouped_launch<{cb}>'s instances, NFIRST, ORDER, PA and XROW by its selector, written by "
           f"{CHECKER}.\n",
           "#include \"experts_grouped.cuh\"\n\n",
           f"#define TF_PROMPT(N, O, P, X) template __global__ void tf_exl3x::grouped_kernel<{cb}, {nt}, {w}, {pf}, {lo}, "
           f"{hi}, N, O, P, X>( \\\n",
           "    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \\\n",
           "    const int*, float*, int, int, int, int, int, int, const int*);\n"]
    for s in p["sels"]:
        out.append(f"TF_PROMPT({b(s, 8)}, {b(s, 4)}, {b(s, 2)}, {b(s, 1)})\n")
    return "".join(out)


def glm_footer(i: dict) -> str:
    out = ["// The grouped_kernel instances exl3_grouped_cuda launches (NT n tiles a block, W warps); the other three are plain.\n",
           "#define TF_GROUPED(NT, W) template __global__ void tf_glm_exl3::grouped_kernel<NT, W>(const half*, const half*, \\\n",
           "    const uint32_t*, const uint32_t*, const int*, const int*, const int*, float*, int, int, int, int, int);\n"]
    out.append(" ".join(f"TF_GROUPED({nt}, {w})" for nt, w in i["glm"]) + "\n")
    return "".join(out)


def grouped_cb(cb: int, i: dict, l2d: bool = False) -> str:
    """experts_cb<cb>.cu's kernels: what grouped_launch<cb> and dequant_launch<cb> instantiate, as instances."""

    names = {0: "3inst", 1: "mcg", 2: "mul1"}
    what = f"{EXL3}experts_cb{cb}.cu with exl3_l2d.patch" if l2d else f"{EXL3}experts_cb{cb}.cu"
    out = [f"// The kernels {what} compiles (codebook {cb}, {names[cb]}): grouped_launch<{cb}>'s tile "
           "settings x K2 ranges and dequant_launch<{cb}>'s widths, written by {CHECKER}.\n".format(cb=cb, CHECKER=CHECKER),
           "#include \"experts_grouped.cuh\"\n\n",
           f"#define TF_GROUPED(NT, W, PF, LO, HI) template __global__ void tf_exl3x::grouped_kernel<{cb}, NT, W, PF, LO, HI>( \\\n",
           "    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \\\n",
           "    const int*, float*, int, int, int, int, int, int, int);\n" if l2d else
           "    const int*, float*, int, int, int, int, int, int);\n"]
    for nt, w, pf in i["tiles"]:
        out.append(" ".join(f"TF_GROUPED({nt}, {w}, {pf}, {lo}, {hi})" for lo, hi in i["ranges"]) + "\n")
    out.append(f"#define TF_DEQUANT(K2) template __global__ void tf_exl3x::dequant_kernel<{cb}, K2>(const uint32_t*, half*, "
               "int, int);\n")
    out.append(" ".join(f"TF_DEQUANT({k})" for k in i["dequant"]) + "\n")
    return "".join(out)


# name: source, line ranges taken, lines dropped (ATen includes), (anonymous namespace line, new name), footer
def table(i: dict) -> dict:
    return {
        "exl3/decode.cuh": (EXL3 + "decode.cuh", [(1, 162)], (), None, ""),
        "exl3/linear.cu": (EXL3 + "linear.cu", [(1, 283)], (5, 6, 7), (14, "tf_exl3_linear"), linear_footer(i)),
        "exl3/experts_grouped.cuh": (EXL3 + "experts_grouped.cuh", [(1, 306), (356, 356)], (6, 7, 8), None, ""),
        "exl3/experts.cu": (EXL3 + "experts.cu", [(1, 310)], (5, 6, 7), (11, "tf_exl3_experts"), PG_FOOTER,
                            {311: "}  // namespace\n"}, (PGROUP_PATCH,)),
        "glm_exl3.cu": (GLM + "exl3.cu", [(1, 236)], (14, 15, 16), (18, "tf_glm_exl3"), glm_footer(i)),
    }


# exl3/l2d/: the same two files with exl3_l2d.patch applied; each span's end and the namespace's close are checked
L2D_TABLE = {
    "exl3/l2d/experts_grouped.cuh": (EXL3 + "experts_grouped.cuh", [(1, 332), (401, 401)], (6, 7, 8), None, "",
                                     {334: "struct GroupedArgs {\n", 401: "}  // namespace tf_exl3x\n"}),
    "exl3/l2d/experts.cu": (EXL3 + "experts.cu", [(1, 240)], (5, 6, 7), (11, "tf_exl3_experts"), EXPERTS_FOOTER,
                            {241: "}  // namespace\n"}),
}

# exl3/x1/: the same two files with exl3_x1.patch applied (rot_rows, prompt_order_kernel, the switch template); checked
X1_TABLE = {
    "exl3/x1/experts_grouped.cuh": (EXL3 + "experts_grouped.cuh", [(1, 346), (433, 433)], (6, 7, 8), None, "",
                                    {348: "struct GroupedArgs {\n", 433: "}  // namespace tf_exl3x\n"}, X1_PATCH),
    "exl3/x1/experts.cu": (EXL3 + "experts.cu", [(1, 387)], (5, 6, 7), (11, "tf_exl3_experts"), X1_PG_FOOTER,
                           {388: "}  // namespace\n"}, (X1_PATCH, PGROUP_PATCH)),
}


def render(name: str, entry: tuple, ref: str, tree: Path | None) -> tuple[str, str]:
    """The copy's expected text and the sha256 of the exact source lines it came from."""

    src, spans, drop, ns, footer, *anchors = entry
    patch: Path | tuple | None = (anchors[1] if len(anchors) > 1 else L2D_PATCH) if anchors else None
    lines = source(src, ref, tree, patch or False).splitlines(keepends=True)
    for number, want in (anchors[0] if anchors else {}).items():
        if number > len(lines) or lines[number - 1] != want:
            raise SystemExit(f"{src} with {patch}: line {number} is not {want.strip()!r}; update {CHECKER}")
    taken: list[tuple[int, str]] = []
    for first, last in spans:
        if last > len(lines):
            raise SystemExit(f"{src} has {len(lines)} lines, fewer than the copy's {last}; update {CHECKER}")
        taken += [(n, lines[n - 1]) for n in range(first, last + 1)]
    digest = hashlib.sha256("".join(t for _, t in taken).encode()).hexdigest()
    where = ", ".join(f"{a}-{b}" for a, b in spans)
    note = ", comments dropped, ATen includes as <cstdint>" if drop else ", comments dropped"
    names = " and ".join(p.name for p in patch) if isinstance(patch, tuple) else patch.name if patch else ""
    what = f"{src} with {names}" if patch is not None else src
    out = [f"// Device code of {what} (lines {where}{note}), checked by {CHECKER}.\n"]
    for number, line in taken:
        if line.lstrip().startswith("//"):
            continue                # comments stay in the source; the copy keeps code only (same SASS)
        if number in drop:
            if "ATen" not in line and "c10" not in line and "torch/" not in line:
                raise SystemExit(f"{src}:{number} is no longer an ATen include; update {CHECKER}")
            if number == min(drop):
                out.append("#include <cstdint>\n")   # the fixed-width integers ATen's headers brought in
            continue
        if ns is not None and number == ns[0]:
            if line != "namespace {\n":
                raise SystemExit(f"{src}:{number} is no longer the anonymous namespace; update {CHECKER}")
            line = f"namespace {ns[1]} {{\n"
        out.append(line)
    if ns is not None:
        out.append(f"}} // namespace {ns[1]}\n\n")
    out.append(footer)
    return "".join(out), digest


def expected(ref: str, tree: Path | None) -> dict[str, tuple[str, str]]:
    i = instances(ref, tree)
    files = {name: render(name, e, ref, tree) for name, e in table(i).items()}
    for cb in (0, 1, 2):
        text = grouped_cb(cb, i)
        files[f"exl3/grouped_cb{cb}.cu"] = (text, hashlib.sha256(text.encode()).hexdigest())
    i2 = instances(ref, tree, l2d=True)
    files.update({name: render(name, e, ref, tree) for name, e in L2D_TABLE.items()})
    for cb in (0, 1, 2):
        text = grouped_cb(cb, i2, l2d=True)
        files[f"exl3/l2d/grouped_cb{cb}.cu"] = (text, hashlib.sha256(text.encode()).hexdigest())
    files.update({name: render(name, e, ref, tree) for name, e in X1_TABLE.items()})
    text = grouped_x1(prompt_instances(ref, tree))
    files["exl3/x1/grouped_cb1.cu"] = (text, hashlib.sha256(text.encode()).hexdigest())
    return files


def main() -> int:
    args = sys.argv[1:]
    write = "--write" in args
    ref = args[args.index("--ref") + 1] if "--ref" in args else "python-0.6"
    tree = Path(args[args.index("--tree") + 1]) if "--tree" in args else None
    files = expected(ref, tree)
    names = [a for a in args if a in files] or list(files)
    bad = 0
    for name in names:
        text, digest = files[name]
        path = copies.ROOT / "zig/kernels/cuda" / name
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(copies.ROOT)} from source lines sha256 {digest}")
        elif not path.exists() or path.read_text() != text:
            print(f"DRIFT {path.relative_to(copies.ROOT)}: differs from its source; rerun with --write and re-prove its bits")
            bad += 1
        else:
            print(f"ok {name} (source lines sha256 {digest})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
