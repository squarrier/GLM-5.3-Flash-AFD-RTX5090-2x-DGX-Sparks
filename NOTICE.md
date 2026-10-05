# NOTICE

GLM-5.3-Flash on 1x RTX 5090 + 2x DGX Spark, version 2.0: TensorFold with MiaAI-Lab's GLM work
Copyright 2026 Scott Quarrier

This version's own scripts, patches and documentation are licensed under the Apache License, Version 2.0
([LICENSE](LICENSE)). If you redistribute it or a modified version of it, keep this NOTICE file and state what you
changed (Apache-2.0, section 4). Version 1.0, the glm53f-afd version of 2026-09-30, stays MIT-licensed at the tag
[`v1.0`](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0); its credits are kept
below, verbatim, in [Previous version (v1.0)](#previous-version-v10).

This version ships no weights, no binaries and no container images. It builds TensorFold and MCDMA from source and
downloads the weights from their own repositories. Each component keeps its own licence, stated below for the revision
this version pins.

## TensorFold (the engine)

- <https://github.com/ashhart/TensorFold>, by Ash Hart ([ashhart](https://github.com/ashhart),
  [@ashxhart](https://x.com/ashxhart)) and the TensorFold contributors.
- **Apache-2.0 at v0.6.5** (`609ca419abecebdc5a059498a613680bd3aa847f`), the pinned revision. TensorFold is Apache-2.0
  from v0.6.0; releases up to v0.5.0 were MIT, and code written before v0.6.0 keeps its MIT notice in TensorFold's
  `LICENSES/MIT.txt`.
- TensorFold's NOTICE reads: "TensorFold / Copyright 2026 TensorFold contributors / https://github.com/ashhart/TensorFold".
  Keep it, and TensorFold's `LICENSES/MIT.txt` notice, with any copy of a patched tree.
- TensorFold's `THIRD_PARTY_NOTICES.md` lists its own upstreams, among them
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp, MIT), whose EXL3 format the routed experts are
  stored in; Hugging Face [transformers](https://github.com/huggingface/transformers) (Apache-2.0), whose `glm5_next`
  model math TensorFold's GLM CUDA engine implements; and [z-lab/dflash](https://github.com/z-lab/dflash) (Z Lab, MIT),
  whose DFlash2 architecture its drafter ports.
- `build.sh` clones it at that revision and applies `patches/`. The patches' changes are offered under Apache-2.0; the
  TensorFold code they modify, and quote as diff context, stays under TensorFold's licences.

## MiaAI-Lab's TensorFold pull requests

By MiaAI-Lab (Mia's AI Lab; [MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)), open
pull requests to TensorFold, under TensorFold's licence (Apache-2.0). Her commits are carried unchanged in substance and
with her authorship; each patch names its pull request and her upstream commit:

- [ashhart/TensorFold#243](https://github.com/ashhart/TensorFold/pull/243), four commits: patches 0002, 0003, 0004
  (with a port note for the split) and 0006;
- [#285](https://github.com/ashhart/TensorFold/pull/285): patch 0005;
- [#301](https://github.com/ashhart/TensorFold/pull/301): patch 0007.

## MiaAI-Lab's GLM-5.3-Flash EXL3 2x DGX Sparks recipe (ported patches)

- <https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold>, Copyright 2026 MiaAI-Lab, Apache-2.0.
  Its NOTICE reads: "GLM-5.3-Flash EXL3 on 2x DGX Spark with TensorFold / Copyright 2026 MiaAI-Lab".
- Ported onto this project's tree, each with `Co-authored-by: MiaAI-Lab` and a credit header naming the recipe
  patches and the recipe commit:

| This project's patch | Her recipe patches | Her recipe commit |
| --- | --- | --- |
| 0009 (`TF_GLM_DENSE=q4`) | 0002, 0005, 0013, 0028 | `cf28cc4f8038be322cdeda220c6f1c8ace8f27d1` (v1.4) |
| 0010 (`TF_GLM_KV=fp8`) | 0038, 0041 (its ring-base span) | `cf28cc4` (v1.4) |
| 0012 (`TF_GLM_SHARED_PREFIX`) | 0015 (with 0030's multi-stream parts), 0063 | `1576746a04983b6eded0551dbf22512ee9e95654` (v1.5) |
| 0013 (expert prompt kernels' launch switches) | 0001, 0020, 0009 | `1576746` (v1.5) |
| 0014 (`TF_GLM_MULTI_PREFILL`) | 0049 | `1576746` (v1.5) |
| 0015 (`TF_GLM_STREAM_SMOOTH`) | 0061 | `1576746` (v1.5) |
| 0016 (`TF_GLM_FILL_BUDGET_MS`) | 0062 (the slice-aware wire is this project's) | `1576746` (v1.5) |
| 0017 (decode kernels: `TF_GLM_Q4_TILE`, `TF_GLM_HC_DEC`, `TF_GLM_LATENT_STAGES`, `TF_GLM_KDA_WIDE_ROWS`, `TF_GLM_DECODE_SELECT`, `TF_GLM_EXL3_*`) | 0016, 0019, 0031, 0043, 0047 (with 0029's decode-row routing and 0009's `_tokens` kernel) | `1576746` (v1.5) |
| 0018 (a test follow-up to 0017) | 0016, 0047 | `1576746` (v1.5) |
| 0019 (`TF_GLM_DFLASH_POLICY`; `fnc5:0.2` in `.env.example`) | 0018, 0021 (with the multi-stream parts of 0027, 0030, 0035) | `1576746` (v1.5) |
| 0020 (`TF_GLM_TOOL_CALLS`, `TF_GLM_KEEP_REASONING`) | 0036 (with 0053's and 0055's changes), 0051, the text part of 0056 | `1576746` (v1.5) |
| 0023 (`TF_GLM_KDA_CHUNKED`: the chunked KDA prompt kernel, prompt grid and prompt replay) | 0012, 0014, 0039, 0008, 0042 | `1576746` (v1.5) |
| 0024 (`TF_GLM_EXL3_PROMPT`: the EXL3 prompt kernel for the routed experts' prompt chunks) | 0004 (its EXL3 prompt kernels), with 0009's shared-row inputs and 0020's item order | `1576746` (v1.5) |
| 0028 (`TF_GLM_LATENT_MMA`, `TF_GLM_HC_MMA`, `TF_GLM_SEQ_ROWS`, `TF_GLM_SPARSE_ONEPASS`, `TF_GLM_PROMPT_DSA`: the attention node's prompt-chunk kernels) | 0004, 0009, 0028 (with 0038's FP8 cache rows in the one-pass kernel) | `1576746` (v1.5) |
| 0030 (`TF_GLM_MULTI_WINDOW`: up to eight concurrent streams) | 0069 | `1576746` (v1.5) |

  Two of the notices in her NOTICE apply here, because patches 0017 and 0020 carry code she adapted from Jay
  Leaton's project (next section). Her other third-party notices cover recipe patches that are not ported here
  (0006, 0044, 0045, 0046, 0058, 0059, and the contributed 0054, 0057, 0060), so they do not apply to this project.
- Two of this project's patches change code ported from her recipe, and keep `Co-authored-by: MiaAI-Lab` and a credit
  header naming her patch and commit (`1576746`, v1.5): 0025 (`TF_GLM_CACHE_ROOM`, the pool's room rule) changes her
  patch 0030's `MultiDecoder._room` and `_grow`, and 0029 (`TF_GLM_PREFILL_ORDER`, shortest-first prompt order)
  changes her patch 0049's grouped chunk (`_group`). TensorFold's `THIRD_PARTY_NOTICES.md` in the patched tree records
  both changes in the sections of the code they change.
- This project's own patches are 0001, 0008, 0011, 0021, 0022 (prompt chunk pairs), 0026 (prefill lanes and FP8 wire
  rows, off) and 0027 (exchanges in flight); 0026 and 0027 write designs of Hugh Madden's (below).

## glm53-tensorfold-spark (code adapted in two of her ported patches)

- <https://github.com/jayleaton/glm53-tensorfold-spark>, pinned at `59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4`, the
  commit that adds its patch 0620 (its patch 0580 came with `2b6ec57` and is unchanged since), **Apache-2.0**.
  By Jay Leaton ([jayleaton](https://github.com/jayleaton), [@jayleaton](https://x.com/jayleaton)). Its NOTICE reads:
  "Copyright 2026 Jay Leaton (https://x.com/jayleaton)".
- Patch 0017: the routed experts' 16-byte non-coherent trellis loads (`TF_GLM_EXL3_LOADS`, her recipe patch 0047)
  are, as her recipe credits them, adapted from that project's patch 0580 (`exl3_ld.cu`'s `ld_kernel`, `LD_NC`).
- Patch 0020: parts of her recipe patch 0036 are adapted from that project's patch 0620: the rule that tool calls
  written inside the think block are the reply's calls when they end it, the reasoning store (keyed by the server's
  call ids and a signature of the conversation), and `const` schemas and `null` read as None for a nullable string.
- TensorFold's `THIRD_PARTY_NOTICES.md` in the patched tree names both, in its "GLM decode kernels" and "GLM tool
  calls" sections.

## MCDMA (the wire)

- <https://github.com/ashhart/MCDMA>, by Ash Hart ([ashhart](https://github.com/ashhart),
  [@ashxhart](https://x.com/ashxhart)), **Apache-2.0**, pinned at `e672c14ff9fc7b38994caf73025cf1588b4de74e`.
- `build.sh` builds its link daemon (`mcdma-rpcd`) and `libmcdma-rpc` from source on each host; the split uses them at
  runtime. None of MCDMA's code is copied into this project or into the patches.
- Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp), [@petruspennanen](https://x.com/petruspennanen))
  wrote [ashhart/MCDMA#5](https://github.com/ashhart/MCDMA/pull/5): its setup notes and build fix. `build.sh` does
  not apply #5; it keeps the one compiler warning that #5 fixes as a warning.

## The attention/expert split: its authors (no code from them is used)

- Hugh Madden / Turquoise Bay AI ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden))
  wrote [glm53f-afd](https://github.com/hughmadden/glm53f-afd) (MIT; v1.1.0, `91db3cc6fe672e2724efa3464f63bd31493f63f6`,
  the engine of this repo's v1.0). Its README describes it as an inference engine that runs GLM-5.3-Flash with the RTX
  5090 running attention, the KV cache, drafting, sampling and the API, and the DGX Sparks running the routed experts:
  attention-FFN disaggregation (AFD). He also wrote [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) (MIT;
  v1.2.0, `bab9fa2`), the source of glm53f-afd's serving shell, wire codec and RDMA transport. This version runs the
  same split on TensorFold over MCDMA.
- T.J. Purtell ([tpurtell](https://github.com/tpurtell), [@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars)) wrote
  [ds41rt](https://github.com/tpurtell/ds41rt) (`3067d06`), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark)
  (`dc6d9b8`) and [cuteafd](https://github.com/tpurtell/cuteafd) (`c650feb`), engines that run attention on RTX GPUs and
  the routed experts on DGX Sparks over RoCE; glmrt's README calls itself an Attention-FFN Disaggregation (AFD) engine
  for GLM-5.3 on one RTX PRO 6000 and four Sparks. Per glm53f-afd's NOTICE, glm53f-afd adapted kernels from ds41rt
  (`3067d06`) and reimplemented ds41rt's wire format and RDMA transport (through mimo26f-afd) and glmrt's TP4 EXL3 split
  and prefill reduce-scatter, which T.J. Purtell designed.
- Patches 0026 and 0027 write three of Hugh Madden's designs for this tree, with no code copied: the exchanges kept in
  flight ahead of the expert ranks (glm53f-afd `91db3cc`, `crates/glm53f-serve/src/lib.rs`; `TF_GLM_MCDMA_INFLIGHT`,
  on in `.env.example`), the prefill in two and four lanes (mimo26f-afd `bab9fa2`,
  `crates/mimo26-coordinator/src/dforward.rs`; glm53f-afd `crates/glm53f-forward/src/forward.rs`;
  `TF_GLM_PREFILL_LANES`, off) and FP8 wire rows (glm53f-afd's `Fp8E4m3Ue8m0K32`; `TF_GLM_WIRE_FP8`, off). The FP8
  rows use the row format of T.J. Purtell's ds41rt, E4M3 with a UE8M0 scale per 32 values, which reached glm53f-afd
  through mimo26f-afd. TensorFold's `THIRD_PARTY_NOTICES.md` in the patched tree names them, in its "GLM prefill lanes
  and FP8 wire rows" and "GLM exchanges in flight" sections, with glm53f-afd's MIT notice (Copyright (c) 2026
  Turquoise Bay AI Pty Ltd).

## Other authors of work the ported patches build on (no code from them is used)

- fla-org ([fla-org](https://github.com/fla-org)): [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  (MIT) implements the published chunkwise algorithm of gated delta networks and Kimi Delta Attention. Mia's chunked
  KDA kernel (patch 0023) computes that algorithm's chunked WY / UT form of the delta-rule recurrence; as her recipe's
  CREDITS say, her kernels are written anew.
- Patryk Mikołajczyk ([mikolaj92](https://github.com/mikolaj92)) wrote
  [ashhart/TensorFold#140](https://github.com/ashhart/TensorFold/pull/140), which skips the DSA pool tiles a row cannot
  see and bounds GLM's radix selection to the pools it can; TensorFold v0.6.0 has that bound for its selection kernel.
  Her recipe patch 0043 (in patch 0017, `TF_GLM_DECODE_SELECT`) adds the same bound to decode scoring and the split
  selection.
- Aditya Thyagarajan ([aditya1503](https://github.com/aditya1503)) wrote
  [ashhart/TensorFold#200](https://github.com/ashhart/TensorFold/pull/200), MTP concurrency (closed; #243 superseded
  it). Patch 0008's message credits it.

## Weights (downloaded by `download.sh`; never included in this repository)

- **GLM-5.3-Flash** by Z.AI ([zai-org](https://huggingface.co/zai-org), [@Zai_org](https://x.com/Zai_org)),
  <https://huggingface.co/zai-org/GLM-5.3-Flash> (revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`): **MIT**. Z.AI's
  licence for the base model ships with the checkpoint as `LICENSE-GLM-5.3-Flash`.
- **The quantization**, <https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold> by Mia's AI Lab:
  **Apache-2.0** (its `LICENSE`), pinned at `76c0b5173166d2795dd48860f45d8224817f894c`. Its safetensors are
  byte-identical to revision `078455ff`, the revision measured; only the licence files and the model card differ. Made
  with [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp ([turboderp](https://github.com/turboderp),
  [@turboderp_](https://x.com/turboderp_); MIT); the checkpoint records no exllamav3 revision.
- **The DFlash2 drafter**, <https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2> by IncoAI
  ([incoai](https://huggingface.co/incoai)), pinned at `bf582e4eacc1810f76656d1811693ff6c6737d2a`: **CC BY-NC-ND 4.0**,
  non-commercial use only, no derivatives. Referenced by name and downloaded from its own repository; never
  redistributed here. Leave `DRAFTER_DIR` empty to serve without it.

## Build and runtime dependencies (not included)

- NVIDIA's PyTorch container, `nvcr.io/nvidia/pytorch:26.07-py3`, the base of the serving image that `build.sh` builds
  on each host, under the NVIDIA Software License Agreement and the Product-Specific Terms for NVIDIA AI Products,
  which the container prints at start. TensorFold's Python dependencies are installed from PyPI into that image.
- [rdma-core](https://github.com/linux-rdma/rdma-core) (`libibverbs`), a C compiler and make on the hosts, for MCDMA;
  Docker and the NVIDIA Container Toolkit; git; `huggingface_hub` (`hf`) for the downloads.
- `extras/gb10-hostguard/` is v1.0's host guard, carried over unchanged in its code: Python's standard library and
  systemd only.

## AGENTS.md (rules for agents that edit this repo)

`AGENTS.md` is "AGENTS.md - credit and attribution" by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), included unchanged (sha256 `81a6dd958181aa1d`) from <https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution> (added 4 October 2026). Her page publishes it for use in other projects: "Drop this file in a project as AGENTS.md."

## Not used

- No vLLM, and no code from glm53f-afd, mimo26f-afd, ds41rt, glmrt or cuteafd.
- No vision: GLM's vision encoder and image paths (her recipe patches 0003, 0050, 0054 and 0056's image part) are not
  ported; this project serves text only.
- No Local Inference Lab weights: the EXL3 TR3 4-bpw quantization by Local Inference Lab, Inc. that this repo's v1.0
  serves is not used by this version (v1.0's licence-required attribution is kept below).

## Previous version (v1.0)

[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (commit `b8f1743`: the
glm53f-afd version of 2026-09-30, with the credit update of 2026-10-04) runs glm53f-afd v1.1.0 with Local Inference
Lab's TR3 4-bpw experts. It stays MIT-licensed at that tag, and its weights' licence-required attribution applies to
anyone who runs it. Its credits follow verbatim: its README's first-screen credits, its README's "License and
credits" section, then its whole NOTICE.md. In them, "this repo", "this repository", "this recipe" and "this
README" mean v1.0, and their links to `LICENSE`, `NOTICE.md` and the README's Performance section point at v1.0's
own files at the tag.

### v1.0's README: first-screen credits

This repo is a deployment recipe, not a new engine. It runs Z.ai's [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`eb9eb208`) with **[glm53f-afd](https://github.com/hughmadden/glm53f-afd)** v1.1.0 (`91db3cc`), by Hugh Madden / Turquoise Bay AI ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)), on a mixed box:

> **Weights attribution (required by licence):** routed-expert weights are *GLM-5.3-Flash TR3 4bpw* by **Local Inference Lab, Inc.** (published by Brandon M. Music, revision `5ab363a8`). Upstream: <https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw>, project home: <https://local-inference-lab.ai/>, licensed LicenseRef-LIL-Attribution-1.0 / ShapleyMCG 1.0. The DFlash2 drafter is [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) by [IncoAI](https://huggingface.co/incoai) (revision `bf582e4e`), CC BY-NC-ND 4.0 (non-commercial). This repo contains no weights; see [NOTICE.md](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v1.0/NOTICE.md).

> **README layout:** this README uses the layout of the README of [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf). Her two-Spark lane is the comparison baseline in [Performance](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v1.0/README.md#performance). No code from her repo is used.

### v1.0's README: License and credits

This repo's scripts, units and docs are **MIT** ([LICENSE](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v1.0/LICENSE)). MIT matches the engine it wraps (glm53f-afd, MIT) and most of that engine's upstreams, and it keeps the recipe easy to reuse. The repo vendors no third-party code and ships no weights. Everything it downloads or builds keeps its own licence, listed in **[NOTICE.md](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v1.0/NOTICE.md)**:

- **engine:** glm53f-afd, MIT;
- **base model:** zai-org GLM-5.3-Flash, MIT;
- **EXL3 experts:** Local Inference Lab, LIL Attribution 1.0 / ShapleyMCG 1.0, with **attribution required**;
- **DFlash2 drafter:** CC BY-NC-ND 4.0, **non-commercial**.

This recipe does not use vLLM, and it contains no code from Mia's AGPL-3.0 repo. It uses that repo's README layout, credited at the top.

All the hard parts are other people's work:

- **[glm53f-afd](https://github.com/hughmadden/glm53f-afd)** by Hugh Madden / Turquoise Bay AI ([@dangerm00se](https://x.com/dangerm00se)): the engine, its KL gate, and [the report](https://services.turquoisebay.ai/share/glm53f-afd/). It incorporates or derives from, per its NOTICE: Hugh Madden's [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd); T.J. Purtell's ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars)) [ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and [sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt); Ash Hart's ([@ashhart](https://github.com/ashhart)) [TensorFold](https://github.com/ashhart/TensorFold); fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention); Z Lab's [z-lab/dflash](https://github.com/z-lab/dflash); the sgl-project's [SGLang](https://github.com/sgl-project/sglang); Local Inference Lab's [b12x](https://github.com/local-inference-lab/b12x); turboderp's [ExLlamaV3](https://github.com/turboderp-org/exllamav3); and Hugging Face's [transformers](https://github.com/huggingface/transformers).
- **Weights:**
  - [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash): the base model.
  - **Local Inference Lab, Inc.** (published by Brandon M. Music): EXL3 TR3 4bpw experts, BF16 teacher logits and KL method. Upstream <https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw>, project home <https://local-inference-lab.ai/>.
  - [IncoAI](https://huggingface.co/incoai): the DFlash2 drafter.
  - [malaiwah](https://huggingface.co/malaiwah) (Michel Belleau): the quant-fidelity registry.
- **The two-Spark reference:** [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab)). This README uses its layout, at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf), and her lane is our comparison baseline.
- **[AGENTS.md](AGENTS.md):** the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), unchanged. Agents that edit this repo follow them.
- **This recipe:** [@squarrier](https://github.com/squarrier), with the AI agents above.

### v1.0's NOTICE.md: upstream projects, weights and licences

This repository (scripts, systemd units, docs, patch) is licensed **MIT**; see [LICENSE](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v1.0/LICENSE). It is a deployment recipe. It contains **no model weights, no engine source, and no code copied from the projects below**, except `patches/0001-*.patch`, a modification of glm53f-afd offered under glm53f-afd's own MIT terms.

`build.sh` downloads and builds the engine from its upstream repository. `download.sh` downloads the weights from their publishers. When you run them, you obtain those works directly from their authors, under **their** licences listed below. Read those licences before you use, deploy, or redistribute anything.

#### Required attribution: EXL3 expert weights

The routed-expert weights used by this recipe are **GLM-5.3-Flash TR3 4bpw (EXL3 K4)** by **Local Inference Lab, Inc.** (published by Brandon M. Music):

- Upstream source: <https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw>
- Project home: <https://local-inference-lab.ai/>
- Licence: `LicenseRef-LIL-Attribution-1.0` (the model card calls it *ShapleyMCG License 1.0*). It is MIT-derived with an attribution **condition**. The rank images `download.sh` creates are derivative works (TP4 slices) and fall under the same licence. This repository does not distribute them.

#### Engine (built from source by `build.sh`, not vendored)

| Project | Licence | Role |
|---|---|---|
| [hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd) v1.1.0 `91db3cc`, © 2026 Turquoise Bay AI Pty Ltd | MIT | The serving engine: coordinator, expert ranks, RDMA wire, API, KL gate. `patches/0001` modifies it. |

glm53f-afd itself incorporates or derives from the following. Its [`NOTICE.md`](https://github.com/hughmadden/glm53f-afd/blob/91db3cc6fe672e2724efa3464f63bd31493f63f6/NOTICE.md) and `docs/REUSE.md` are the authoritative ledger. Each licence below is the one that applies to the revision glm53f-afd v1.1.0 took code or constants from. A project may have relicensed later releases.

| Project | Licence |
|---|---|
| [hughmadden/mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) | MIT |
| [tpurtell/ds41rt](https://github.com/tpurtell/ds41rt) (T.J. Purtell) | MIT |
| [tpurtell/glmrt-5.3-1rtx-4spark](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) (T.J. Purtell) | MIT |
| [tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt) / [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | Apache-2.0 |
| [ashhart/TensorFold](https://github.com/ashhart/TensorFold) @ `bb4b4a3` (v0.3.4.1), plus copy-window constants from `a83ed1e` (v0.3.2) | MIT for those revisions. TensorFold is Apache-2.0 from v0.6.0 (upstream `e3ac0ea`). |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT |
| [z-lab/dflash](https://github.com/z-lab/dflash) | MIT |
| [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache-2.0 |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 |
| linux-rdma/rdma-core headers | GPL-2.0 or OpenIB.org BSD (used under BSD) |

#### Weights and models (downloaded by `download.sh`, not included)

| Work | Publisher | Revision used | Licence |
|---|---|---|---|
| GLM-5.3-Flash, official FP8 (non-expert tensors) | [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) | `eb9eb208` | MIT |
| GLM-5.3-Flash TR3 4bpw EXL3 K4 (routed experts) | [Local Inference Lab / brandonmusic](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | `5ab363a8` | LIL Attribution 1.0 / ShapleyMCG 1.0 (see above) |
| GLM-5.3-Flash DFlash2 drafter | [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) | `bf582e4e` | **CC BY-NC-ND 4.0: non-commercial, no derivatives** |

The DFlash2 drafter's licence forbids commercial use. For a commercial deployment, run without it: remove `--drafter` from `site/glm53f-afd-coord-run`. Single-stream decode drops to about 45 tok/s.

#### Build and runtime dependencies (not vendored)

NVIDIA CUDA Toolkit container images (`nvidia/cuda:*-devel`; NVIDIA Deep Learning Container License), the NVIDIA driver and CUDA MPS, the Rust toolchain (MIT/Apache-2.0), Docker, rdma-core, and `huggingface_hub` (Apache-2.0).

**Not used:** vLLM. This recipe does not use or include vLLM. vLLM appears only in the README comparison, which cites Mia's two-Spark vLLM lane.

#### Reference projects (layout and comparison only, no code taken)

- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) (AGPL-3.0; MIT before 2026-09-07), by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)). This README uses its layout, at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf), and her lane is the comparison baseline. **No code from it is included or adapted here.**
- The public four-Spark recipes that glm53f-afd's NOTICE lists as sources: Matt Mastracci's ([@mmastrac](https://github.com/mmastrac)) [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) at `5ea4121` (no licence file), and Zbigniew Majewski's ([knapcio](https://github.com/knapcio)) [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4) at `beca637` (MIT). glm53f-afd adapts parts of them, recorded in its `docs/REUSE.md`; nothing of theirs is in this repo.

#### AGENTS.md (rules for agents that edit this repo)

`AGENTS.md` is "AGENTS.md - credit and attribution" by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), included unchanged (sha256 `81a6dd958181aa1d`) from <https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution> (added 4 October 2026). Her page publishes it for use in other projects: "Drop this file in a project as AGENTS.md."
