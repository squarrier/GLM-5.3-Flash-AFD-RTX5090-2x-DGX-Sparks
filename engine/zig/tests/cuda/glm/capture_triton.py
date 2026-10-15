#!/usr/bin/env python3
"""Every Triton launch of a python-0.6 GLM-5.3-Flash run, for tools/zig/glm_aot.py extract (no change to python-0.6).

Installs tools/zig/triton_aot_manifest.py's Recorder, then runs MODULE as __main__ with its ARGS; each process writes
OUT/launches-<host>-<pid>.json at exit, on SIGTERM or on SIGINT. Kernels land in TRITON_CACHE_DIR, which extract reads.
Usage: python3 -B capture_triton.py --tools TOOLS --out OUT MODULE [ARGS...]   (LAB-PLAN: evidence/ZK4/LAB-PLAN.md)
"""

from __future__ import annotations

import argparse
import atexit
import os
import runpy
import signal
import socket
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True, help="the folder of triton_aot_manifest.py (the fork's tools/zig)")
    ap.add_argument("--out", required=True, help="folder for each process's launches-<host>-<pid>.json")
    ap.add_argument("module", help="the module to run as __main__, e.g. tensorfold.cli")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="its arguments")
    a = ap.parse_args()
    if not os.environ.get("TRITON_CACHE_DIR"):
        sys.exit("capture_triton: set TRITON_CACHE_DIR (extract reads the kernels from it)")
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as manifest

    rec = manifest.Recorder().install()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"launches-{socket.gethostname()}-{os.getpid()}.json"
    done = []

    def dump() -> None:
        if not done:
            done.append(True)
            rec.dump(path)
            print(f"capture_triton: {len(rec.kernels)} specializations -> {path}", file=sys.stderr)

    def stop(signum, frame) -> None:
        dump()
        sys.exit(128 + signum)

    atexit.register(dump)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sys.argv = [a.module, *a.args]
    runpy.run_module(a.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
