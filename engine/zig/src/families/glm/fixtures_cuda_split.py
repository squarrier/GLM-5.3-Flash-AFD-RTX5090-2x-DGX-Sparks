"""Digests of python-0.6's GLM-5.3-Flash TP2 rank slices on the synthetic checkpoint, for cuda_split_test.zig.

Run in the pytorch image (numpy) with TensorFold python-0.6's source and the checkpoint `tf-glm-load synth DIR` writes:
    python -I fixtures_cuda_split.py <src/tensorfold> <DIR> <out.json>

split.py (src/tensorfold/families/glm5_next/cuda/split.py, by Ash Hart) is loaded by path and run as it is: its
split_file() writes each rank's share of every shard and rule() names each tensor's split. The one step outside it
restates weights.py load() line for line (by Ash Hart): the head is replicated by split.py, then each rank keeps its
vocabulary rows, `hw[rank * vl:(rank + 1) * vl]`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC, CK, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
RANKS = 2
spec = importlib.util.spec_from_file_location("glm_split", SRC / "families/glm5_next/cuda/split.py")
assert spec is not None and spec.loader is not None
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


def sha(data) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(data))).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


shards = sorted(CK.glob("model-*.safetensors"))
out = {"source": "python-0.6 src/tensorfold/families/glm5_next/cuda/split.py (split_file, rule, read_header)",
       "split_py_sha256": file_sha(SRC / "families/glm5_next/cuda/split.py"),
       "shards": len(shards), "files": {p.name: file_sha(p) for p in shards},
       "index_sha256": file_sha(CK / "model.safetensors.index.json")}
tensors: dict[str, list[str]] = {}  # name: [split.py's rule, rank 0's sha256, rank 1's sha256]
with tempfile.TemporaryDirectory() as tmp:
    for rank in range(RANKS):
        for src in shards:
            split.split_file(src, tmp, rank, RANKS)
        for part in sorted(Path(tmp).glob(f"*.rank{rank}.safetensors")):
            header, base = split.read_header(part)
            header.pop("__metadata__", None)
            mm = np.memmap(part, dtype=np.uint8, mode="r")
            for name, info in header.items():
                a, b = info["data_offsets"]
                data, kind = mm[base + a:base + b], split.rule(name)
                if name.startswith("lm_head."):  # weights.py load(): the rank's vocabulary rows of the replicated head
                    rows = info["shape"][0]
                    vl, per = rows // RANKS, (b - a) // rows
                    data, kind = data[rank * vl * per:(rank + 1) * vl * per], "vocab"
                tensors.setdefault(name, [kind])
                assert len(tensors[name]) == 1 + rank, name
                tensors[name].append(sha(data))
            del mm
            part.unlink()
# one tensor a line
head = json.dumps(out, indent=1, sort_keys=True)[:-2]
body = ",\n".join(f"  {json.dumps(n)}: {json.dumps(t)}" for n, t in sorted(tensors.items()))
OUT.write_text(f'{head},\n "tensors": {{\n{body}\n }}\n}}\n')
print(f"{len(tensors)} tensors from {len(shards)} shards -> {OUT}")
