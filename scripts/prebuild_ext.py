#!/usr/bin/env python3
"""Prebuild TensorFold's CUDA extensions into $TORCH_EXTENSIONS_DIR (bind-mounted, persistent per host and tree).

Why: on a GB10, building them inside `tensorfold serve` happens after ~90 GiB of weights are resident, and
nvcc/ninja can then push the host's available memory under a few GiB. Run it with no model loaded (./build.sh ext
and ./start.sh up do); the serving containers then mount the same directory and reuse the builds. A no-op of
about 30 s when the builds are current, 5-10 min a host the first time.
"""
import importlib
import os
import sys
import time

os.environ.setdefault("MAX_JOBS", "4")
MODS = [
    ("tensorfold.families.glm5_next.cuda.kda", "_ext"),
    ("tensorfold.cuda.exl3.experts", "_ext"),
    ("tensorfold.cuda.exl3.linear", "_ext"),
    ("tensorfold.families.glm5_next.cuda.exl3_mm", "_ext"),
    ("tensorfold.cuda.experts", "_ext"),
    ("tensorfold.cuda.kernels.prefill_attention", "_ext"),
    ("tensorfold.cuda.kernels.qmm", "_ext"),
    ("tensorfold.families.glm5_next.cuda.glue", "_hc_ext"),      # TF_GLM_HC_DEC's hc.cu (patch 0017)
    ("tensorfold.cuda.nvfp4.linear", "_ext"),
    ("tensorfold.cuda.kernels.gdn", "_ext"),
    ("tensorfold.families.glm5_next.cuda.kda_chunked", "_ext"),  # TF_GLM_KDA_CHUNKED's kda_chunk.cu (patch 0023)
    ("tensorfold.families.glm5_next.cuda.g53_experts", "_ext"),  # TF_GLM_EXPERT_KERNEL=g53's g53_rank.cu (patch 0034)
]
import torch  # noqa: E402

torch.cuda.set_device(0)
print("TORCH_EXTENSIONS_DIR", os.environ.get("TORCH_EXTENSIONS_DIR"), "cap", torch.cuda.get_device_capability(0), flush=True)
bad = 0
for mod, fn in MODS:
    t = time.time()
    try:
        m = importlib.import_module(mod)
        getattr(m, fn)()
        print(f"OK   {mod}.{fn}  {time.time()-t:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - report every module, then fail
        bad += 1
        print(f"FAIL {mod}.{fn}: {type(e).__name__}: {str(e)[:300]}", flush=True)
print("prebuild done, failures:", bad, flush=True)
sys.exit(1 if bad else 0)
