# NOTICE: upstream projects, weights and licences

This repository (scripts, systemd units, docs, patch) is licensed **MIT**; see [LICENSE](LICENSE). It is a deployment recipe. It contains **no model weights, no engine source, and no code copied from the projects below**, except `patches/0001-*.patch`, a modification of glm53f-afd offered under glm53f-afd's own MIT terms.

`build.sh` downloads and builds the engine from its upstream repository. `download.sh` downloads the weights from their publishers. When you run them, you obtain those works directly from their authors, under **their** licences listed below. Read those licences before you use, deploy, or redistribute anything.

## Required attribution: EXL3 expert weights

The routed-expert weights used by this recipe are **GLM-5.3-Flash TR3 4bpw (EXL3 K4)** by **Local Inference Lab, Inc.** (published by Brandon M. Music):

- Upstream source: <https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw>
- Project home: <https://local-inference-lab.ai/>
- Licence: `LicenseRef-LIL-Attribution-1.0` (the model card calls it *ShapleyMCG License 1.0*). It is MIT-derived with an attribution **condition**. The rank images `download.sh` creates are derivative works (TP4 slices) and fall under the same licence. This repository does not distribute them.

## Engine (built from source by `build.sh`, not vendored)

| Project | Licence | Role |
|---|---|---|
| [hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd) v1.1.0 `91db3cc`, © 2026 Turquoise Bay AI Pty Ltd | MIT | The serving engine: coordinator, expert ranks, RDMA wire, API, KL gate. `patches/0001` modifies it. |

glm53f-afd itself incorporates or derives from the following. Its [`NOTICE.md`](https://github.com/hughmadden/glm53f-afd/blob/91db3cc6fe672e2724efa3464f63bd31493f63f6/NOTICE.md) and `docs/REUSE.md` are the authoritative ledger.

| Project | Licence |
|---|---|
| [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) | MIT |
| [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) (T.J. Purtell) | MIT |
| [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) (T.J. Purtell) | MIT |
| [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) / [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 |
| [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | MIT |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT |
| [z-lab/dflash](https://github.com/z-lab/dflash) | MIT |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| linux-rdma/rdma-core headers | GPL-2.0 or OpenIB.org BSD (used under BSD) |

## Weights and models (downloaded by `download.sh`, not included)

| Work | Publisher | Revision used | Licence |
|---|---|---|---|
| GLM-5.3-Flash, official FP8 (non-expert tensors) | [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) | `eb9eb208` | MIT |
| GLM-5.3-Flash TR3 4bpw EXL3 K4 (routed experts) | [Local Inference Lab / brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | `5ab363a8` | LIL Attribution 1.0 / ShapleyMCG 1.0 (see above) |
| GLM-5.3-Flash DFlash2 drafter | [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | `bf582e4e` | **CC BY-NC-ND 4.0: non-commercial, no derivatives** |

The DFlash2 drafter's licence forbids commercial use. For a commercial deployment, run without it: remove `--drafter` from `site/glm53f-afd-coord-run`. Single-stream decode drops to about 45 tok/s.

## Build and runtime dependencies (not vendored)

NVIDIA CUDA Toolkit container images (`nvidia/cuda:*-devel`; NVIDIA Deep Learning Container License), the NVIDIA driver and CUDA MPS, the Rust toolchain (MIT/Apache-2.0), Docker, rdma-core, and `huggingface_hub` (Apache-2.0).

**Not used:** vLLM. This recipe does not use or include vLLM. vLLM appears only in the README comparison, which cites Mia's two-Spark vLLM lane.

## Reference projects (layout and comparison only, no code taken)

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) (AGPL-3.0; MIT before 2026-09-07). This README follows its layout, and its lane is the comparison baseline. **No code from it is included or adapted here.**
- The public four-Spark recipes cited by glm53f-afd: [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) (no licence file) and [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) (MIT). Upstream reimplements ideas from them; nothing of theirs is in this repo.
