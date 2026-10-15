"""The EXL3 fatbins' kernels as the cubins record them: each symbol, its parameters (EIATTR_KPARAM_INFO offset and size,
by ordinal, from `cuobjdump -elf`) and its static shared bytes (`cuobjdump -res-usage`), for
zig/src/families/glm/exl3_kernels.json. The bindings' tests check every launch against it and that every kernel is
launched. usage: python3 tools/zig/exl3_kernels.py OUT.json NAME=ELF.txt:RES.txt [...]   (sm_121 and sm_120 agree)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def params(text: str) -> dict[str, list[list[int]]]:
    out: dict[str, list[tuple[int, ...]]] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"\s*\.nv\.info\.(\S+)", line)
        if m:
            name = m.group(1)
            out.setdefault(name, [])
            continue
        if name is None:
            continue
        m = re.search(r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*(0x[0-9a-f]+)\s+Size\s*:\s*(0x[0-9a-f]+)", line)
        if m:
            out[name].append(tuple(int(x, 16) for x in m.groups()))
    return {k: [[t[1], t[2]] for t in sorted(v)] for k, v in sorted(out.items()) if v}


def shared(text: str) -> dict[str, int]:
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        m = re.search(r"\bSHARED:(\d+)", line)
        if m and name is not None:
            out[name] = int(m.group(1))
            name = None
    return out


def main() -> int:
    out = {}
    for arg in sys.argv[2:]:
        name, paths = arg.split("=", 1)
        elf, res = paths.split(":", 1)
        got, sh = params(Path(elf).read_text()), shared(Path(res).read_text())
        if set(got) != set(sh):
            raise SystemExit(f"{name}: -elf and -res-usage list different kernels")
        out[name] = [{"symbol": k, "params": v, "shared": sh[k]} for k, v in got.items()]
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{sys.argv[1]}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
