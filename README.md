<p align="center">
  <b>GLM-5.3-Flash on 1× RTX 5090 + 2× DGX Spark</b><br>
  <sub>An attention/expert split on <a href="https://github.com/ashhart/TensorFold">TensorFold</a>: attention, KV cache and drafter on the 5090, the routed experts on two Sparks, one RDMA exchange per MoE layer over <a href="https://github.com/ashhart/MCDMA">MCDMA</a></sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RTX_5090-attention_node-76b900?style=flat-square" alt="RTX 5090 attention node">
  <img src="https://img.shields.io/badge/2%C3%97_DGX_Spark-expert_nodes-2ea44f?style=flat-square" alt="2× DGX Spark expert nodes">
  <img src="https://img.shields.io/badge/MCDMA-RoCE_v2_RDMA-0969da?style=flat-square" alt="MCDMA over RoCE v2">
  <img src="https://img.shields.io/badge/TensorFold-v0.6.5_%2B_51_patches-6f42c1?style=flat-square" alt="TensorFold v0.6.5 + 51 patches">
  <img src="https://img.shields.io/badge/license-Apache--2.0-555?style=flat-square" alt="Apache-2.0">
</p>

This repo is a deployment recipe, not a new engine. Version 2.15 serves Z.ai's [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as MiaAI-Lab's EXL3 checkpoint with **[TensorFold](https://github.com/ashhart/TensorFold)** v0.6.5 by Ash Hart ([@ashhart](https://github.com/ashhart)), plus 51 patches (0046-0048 are TensorFold v0.6.6's own commits), on three machines:

- An **x86 host with one RTX 5090** is the attention node. It runs attention, holds the KV cache, runs the DFlash2 drafter and sampling, and serves the OpenAI-compatible API. Its RAM holds the kept prompts' states and the prompts the 5090's pool evicts.
- **Two DGX Sparks (GB10)** each hold one half of every MoE layer's routed experts.
- In every MoE layer the 5090 sends its rows to both Sparks and gets two partial sums back over **RoCE v2 RDMA**, through Ash Hart's **[MCDMA](https://github.com/ashhart/MCDMA)** link daemons, with four exchanges in flight to each Spark.

> **Where v1.0 is better: requests at once, four-stream decode, one-stream prose and testing; on long prompts this version is now ahead.** [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0), this repo's glm53f-afd version of 2026-09-30, serves 16 requests at once (this version decodes eight and queues the rest), decodes 117.6 tok/s at four streams against this version's 114.1 in the lab and prose at one stream 73.1 against 71.3, and passed a 4-hour soak with failure drills; this version's soak passed its four failure drills but not its probe check ([Soak and failure drills](#soak-and-failure-drills)). v1.0 prefills 3.1-3.3K tok/s on 5K-95K prompts; this version prefills 3,374 / 3,642 / 3,707 tok/s on fresh 8K / 31K / 62K prompts in the lab. Both keep a ~1.58M-token KV pool (1,585,152 here): [What changed since v1.0](#what-changed-since-v10).

> **Where Mia's two-Spark lane is better:** it needs two machines, not three, and no RTX 5090; it reads pictures (this split serves text only); and in 2.1's same-window run against her v1.8, each request generated faster at ten clients (16.0 against 12.7 tok/s: her lane decodes four at a time and queues the rest, this split eight) and her KV pool was 8% larger (1,710,080 against 1,579,008 tokens). This version has not been measured against her lane. See [Performance](#performance).

It serves MiaAI-Lab's checkpoint the way her own [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) does (4-bit dense weights, FP8 KV cache, DFlash2, her decode kernels, chunked KDA prompts, her attention-side prompt kernels, her v1.8 serving fixes) and carries her GLM work: her three open TensorFold pull requests as her commits, and ports of her recipe's patches through her v1.8, each credited. Where it differs from her main lane, the difference was measured: eight streams instead of four, with a 32-row verify window; her draft policy at `fnc5:0.2` instead of `fnc7:0.3`; 2,048-row prompt chunks instead of her 4,096 (GB10's limit on this split); her v1.7.1 kept-prompt patches off; this recipe's own levers for the split (prompt chunk pairs, four MoE exchanges in flight with four prompt lanes, the pool's room rule, shortest-first prompt order, prompt chunks filled as pairs while streams decode); and four pieces of Hugh Madden's glm53f-afd: kept prompts in host RAM, his expert prompt kernels on the Sparks in place of her EXL3 prompt kernel (with glm53f-rank's own schedule on the biggest prompt windows), the four-lane prefill, and BF16 partial sums on the return wire for prompt windows. Her attention-side prompt kernels, his expert prompt kernels and the BF16 partial sums all change the replies; with all three on, AEON-30 scored 22 of 30 (the bar is 21; 2.1 scored 23 without the BF16 sums). See [What the patches are](#what-the-patches-are).

## Credits

| Author | Profiles | What they authored, as used here | Repo, commit |
|---|---|---|---|
| **Mia** (Mia's AI Lab) | [MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab) | the EXL3 4-bpw checkpoint; her TensorFold pull requests #243, #285 and #301 (patches 0002-0007, her commits); the 26 patches here that port her recipe's patches (among them her EXL3 prompt kernel, 0004; her attention-side prompt kernels, 0004, 0009 and 0028; eight streams, 0069; her v1.7.1 and v1.8 fixes, 0071-0083), the code of hers that patches 0025, 0029 and 0033 change (her 0030, 0049 and 0062), and the parts of her 0009 and 0020 that patch 0034 reimplements; [AGENTS.md](AGENTS.md); this README's layout | [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at [`cf28cc4`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1) (v1.4), [`1576746`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) (v1.5), [`68ebd67`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/68ebd67b5326974b8004009e202268b1fa7c551d) (v1.7.1) and [`33b50fd`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/33b50fde06fd7ea604cbc6a663880068ab1e2ee4) (v1.8); [the checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) at [`76c0b517`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/76c0b5173166d2795dd48860f45d8224817f894c) |
| **Ash Hart** | [ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart) | TensorFold, the engine, and two of TensorFold v0.6.6's commits (patches 0047 and 0048, his commits); TensorFold's MIT code that Hugh Madden's expert prompt kernels carry (the EXL3 decoder, fragment MMA and Hadamard butterfly, patch 0034); MCDMA, the RDMA wire | [TensorFold](https://github.com/ashhart/TensorFold) at [`609ca41`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f) (v0.6.5), [`cb2ebf0`](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248) (v0.6.6) and `bb4b4a3` (v0.3.4.1, MIT); [MCDMA](https://github.com/ashhart/MCDMA) at [`e672c14`](https://github.com/ashhart/MCDMA/tree/e672c14ff9fc7b38994caf73025cf1588b4de74e) |
| **Hugh Madden** / Turquoise Bay AI | [hughmadden](https://github.com/hughmadden), [@dangerm00se](https://x.com/dangerm00se) | glm53f-afd, the engine of v1.0: GLM-5.3-Flash with attention on an RTX 5090 and the routed experts on DGX Sparks, the split this version runs on TensorFold; his expert prompt kernels (glm53f-rank's large-M EXL3 kernels), whose code patches 0034 and 0035 carry for the Sparks' prompt chunks (on), and glm53f-rank's own schedule for its biggest windows, which patches 0049 and 0051 run on the Sparks' prompt windows of 1,536 rows and more (on); the designs patches 0026, 0027, 0032 and 0050 write for it: the exchanges kept in flight ahead of the expert ranks (on), the host RAM tier for kept prompts (on), the two- and four-lane prefill (four lanes, on), BF16 partial sums on the return wire (his BF16 return planes, on) and the FP8 wire rows (off) | [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at [`91db3cc`](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) (v1.1.0); [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) at [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79) |
| **T.J. Purtell** | [tpurtell](https://github.com/tpurtell), [@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars) | ds41rt, glmrt and cuteafd: engines that run attention on RTX GPUs and the routed experts on DGX Sparks; DS41RT's wire-row format, which patch 0026's FP8 wire rows use (off); glmrt's split of every expert by intermediate channel, which Hugh Madden's expert prompt kernels follow (patch 0034); ds41rt's host cache eviction design, in glm53f-afd's host RAM tier (patch 0032) | [ds41rt](https://github.com/tpurtell/ds41rt) at `3067d06`, [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) at `dc6d9b8`, [cuteafd](https://github.com/tpurtell/cuteafd) at `c650feb` |
| **E-Zou Shen** | [ezoushen](https://github.com/ezoushen), [@ezoushen](https://x.com/ezoushen) | her recipe patches 0071 (a resumed shared prefix copies its rows) and 0074 (the pool compacts before it evicts), contributed to her recipe; here 0036, 0038 and their tests in 0041 (off) | her recipe's pull requests [#44](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/44) and [#62](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/62), at `68ebd67` (v1.7.1) |
| **desy0305** | [desy0305](https://github.com/desy0305) | her recipe patch 0073 (waiting requests whose client left are dropped at once) and its checks; here 0037 (off) | her recipe's pull request [#51](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/51), at `68ebd67` (v1.7.1) |
| **johnwhited** | [johnwhited](https://github.com/johnwhited) | her recipe patches 0081 (capacity refusals answer 429 with Retry-After), 0082 (admission at saturation) and 0083 (a stream whose delivery fails ends at once), and the delivery-failure handling in 0073; here 0043 and 0045 (on), 0044 (off, her default) and 0037 | her recipe's pull request [#48](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/48), at `33b50fd` (v1.8) |
| **m-naoki-m** | [m-naoki-m](https://github.com/m-naoki-m), [@\_m\_naoki\_m\_](https://x.com/_m_naoki_m_) | her recipe patch 0078 (a take-over decides which kept prompts stay before it copies any); here 0042 (on) | her recipe's pull request [#71](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/71), at `33b50fd` (v1.8); also [TensorFold#421](https://github.com/ashhart/TensorFold/pull/421) |
| **Philip Mossop** | [philip-pentatonic](https://github.com/philip-pentatonic) | TensorFold v0.6.6's `--name-priority ID=background` (patch 0046, his commit) | [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445), [`dce62cf`](https://github.com/ashhart/TensorFold/commit/dce62cfa3c6752c828c4a17338d3ae585e43e65a) |
| **Jay Leaton** | [jayleaton](https://github.com/jayleaton), [@jayleaton](https://x.com/jayleaton) | code Mia adapted from his patches 0580 (the experts' decode loads, in patch 0017) and 0620 (tool calls, in patch 0020) | [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4) |
| **Z.ai** | [zai-org](https://huggingface.co/zai-org), [@Zai_org](https://x.com/Zai_org) | GLM-5.3-Flash, the model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at [`eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) |
| **turboderp** | [turboderp](https://github.com/turboderp), [@turboderp_](https://x.com/turboderp_) | ExLlamaV3, which made Mia's checkpoint in its EXL3 format | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| **IncoAI** | [incoai](https://huggingface.co/incoai) | the DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at [`bf582e4e`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) |

Also credited in [License and credits](#license-and-credits) and [NOTICE.md](NOTICE.md): meleesciony ([@meleesciony](https://github.com/meleesciony)), who diagnosed the kept-cap failure of her recipe's [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75) and proposed the fix her patch 0077 makes (at `68ebd67`, v1.7.1; here 0040, 0041); Lukas-tek-no-logic ([@Lukas-tek-no-logic](https://github.com/Lukas-tek-no-logic)), who reported the `<|assistant|>` leak her patch 0075 fixes ([issue #60](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/60), at `68ebd67`; here 0039); Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp)) for [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5), Patryk Mikołajczyk ([@mikolaj92](https://github.com/mikolaj92)) and Aditya Thyagarajan ([@aditya1503](https://github.com/aditya1503)) for TensorFold #140 and #200, fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and every credit of v1.0, unchanged.

> **Weights and licences (read first):** the checkpoint is [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) by **Mia's AI Lab**, a quantization licensed **Apache-2.0**, derived from Z.AI's GLM-5.3-Flash (**MIT**). The drafter, [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), is **CC BY-NC-ND 4.0: non-commercial use only, no derivatives**. This repo ships no weights: `download.sh` fetches both from their own repos. See [NOTICE.md](NOTICE.md).

> **Provided as-is, with no support.** This is a personal homelab recipe, shared in case it helps someone. Issues and PRs may go unanswered. See [Support](#support).

| I want to... | Go to |
|---|---|
| See how fast it is | [Performance](#performance) |
| Know what hardware I need | [Hardware](#hardware) |
| Lock it down | [Security](#security) |
| Run it | [Quick start](#quick-start) |
| Understand how it works | [docs/DESIGN.md](docs/DESIGN.md) |
| Keep a DGX Spark from hard-hanging | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#dgx-spark-gb10-memory-safety-read-this-first) and [extras/gb10-hostguard](extras/gb10-hostguard/README.md) |
| See what changed since 2.1, or since v1.0 | [What changed since 2.1](#what-changed-since-21), [since v1.0](#what-changed-since-v10) |
| Know where TensorFold's own split work stands | [Upstream](#upstream) |
| Run the earlier versions | [2.1](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.1), [2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0) (TensorFold), [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (glm53f-afd) |

## What changed since 2.1

[2.1](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.1) (2026-10-06) keeps its engine, machines, checkpoint and drafter in 2.15, which adds 3 patches (0049-0051), four settings and the watcher. Each lever below was measured in the lab against the same configuration without it, in this order ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#215s-levers-one-at-a-time)):

- **Four MoE exchanges in flight, with four prompt lanes** (`MCDMA_INFLIGHT=4` and `TF_GLM_PREFILL_LANES=4`; patches 0027 and 0026, written to Hugh Madden's glm53f-afd designs; 2.1 ran two in flight and two lanes): four links to each Spark, and up to four prompt chunks fill together, each with its own exchange. Fresh 8K prompts 11.0% faster and a cold ~2.6K prompt's first token 3.9% sooner; the other measures within their noise; the same replies as 2.1, bit for bit. The 5090 peaked at 25.9 GiB, under its memory gate. A fourth link needs a second MCDMA connect daemon on the attention host, which `./start.sh` starts (one daemon holds six peers).
- **A 32-row verify window at eight streams** (`TF_GLM_MULTI_WINDOW=32`; 2.1 ran 64): with eight streams decoding, a verify round carries at most 32 rows, so each round is shorter and the drafter's longest drafts are cut. The coding-agent load at eight requests 9.6% faster (50.0 tok/s a stream against 45.7), with the same replies; at four streams the window does not bind. The KV pool grows by 6,144 tokens to 1,585,152.
- **BF16 partial sums on the return wire** (patch 0050, `TF_GLM_PARTIALS_BF16=1` on all three nodes; written to Hugh Madden's design, glm53f-afd's BF16 return planes): each Spark rounds its half's sum for a prompt window to BF16 before it replies (8,192 bytes a row instead of 16,384), and the 5090 adds the halves in FP32. Decode windows keep FP32 replies. A cold ~2.6K prompt's first token 8.3% sooner, fresh 31K / 62K prompts 2.9% faster, a cold 100K prompt's first token 2.7% sooner, and the ten-client arena cell's 5.8% sooner. **It changes the replies:** against the same configuration without it the mean KL is 0.00218 (the gate is 0.003; p99 0.0351, top-1 99.04%), and AEON-30 scored 22 of 30 (23 without it; the bar is 21). To keep 2.1's replies bit for bit, remove `TF_GLM_PARTIALS_BF16=1` from both `ATTN_ENV` and `EXPERT_ENV`.
- **glm53f-rank's own schedule on the Sparks' big prompt windows** (patches 0049 and 0051: `TF_GLM_EXPERT_KERNEL_MT=4`, `_GW=16`, `_NT=4`, `_L2=1` with `TF_GLM_EXPERT_KERNEL_TIER_ROWS=1536`): Hugh Madden's expert prompt kernels run his kernels' schedule for windows above 2,048 rows on every prompt window of 1,536 rows and more, and their default below. Fresh 31K / 62K prompts 6.7% faster and a cold 100K prompt's first token 6.7% sooner, with the same replies. On every window the same schedule ran 4-10% slower on windows of 256 to 1,024 rows and made the ten-client cell's first token 3.5% later, so it is tiered.
- **The watcher is back** (`extras/watch`, ported from v1.0), with `./start.sh recover`, `SIGTERM`-only stops, a control lock and a stop marker: [Operations](#operations).
- **Measured and left off:** three exchanges in flight (four replaced them), her draft policy at `fnc7:0.3`, her queued-cancel fix alone (0037), her copy drafts (not ported) and the big-window schedule on every window: [docs/DESIGN.md](docs/DESIGN.md#measured-and-off).

2.1 as published against 2.15 in the lab, with the same harnesses as 2.1's numbers:

| | 2.1 as published¹ | 2.15, lab² | 2.15 / 2.1 |
|---|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 75.7 / 81.5 / 83.2 | 71.3 / 86.1 / 91.4 | 0.94x / 1.06x / 1.10x³ |
| Decode, four streams (aggregate tok/s) | 110.6 | 114.1 | 1.03x³ |
| spark-bench's ~2.6K-token prompt (tok/s) | 1,953 | 2,420 | 1.24x |
| Cold ~2.6K prompt, first token | 1.313 s | 1.058 s | 0.81x |
| Fresh 8K / 31K / 62K prompts (tok/s) | 2,946 / 3,030 / 3,138 | 3,374 / 3,642 / 3,707 | 1.15x / 1.20x / 1.18x |
| Cold 100K prompt, first token | 32.4 s | 27.1 s | 0.84x |
| Coding agents, 4 x 1,024 / 8 x 1,024 (tok/s) | 236.2 / — | 255.8 / 421.7 | 1.08x³ / — |
| Arena 65,535 x 10: prompt / gen t/s, first token, gen a request | 1,198 / 64.2, 7.70 s, 12.7 | 1,151 / 64.5, 7.67 s, 13.9 | 0.96x / 1.01x, 1.00x, 1.09x |
| Arena 100,000 x 5: the same | 1,872 / 65.8, 4.00 s, 18.6 | 1,935 / 68.0, 3.85 s, 18.2 | 1.03x / 1.03x, 0.96x, 0.98x |
| KV pool (tokens, all requests) | 1,579,008 | 1,585,152 | 1.00x |
| AEON-30 (idle) | 23 of 30 | 22 of 30 | one fewer |
| Soak and failure drills | none | a 4.11-hour window, 4,071 requests, 0 errors, 0 restarts; the probe check failed (58 of 60: a long tool-call stream paused 6.3 and 7.0 s, the limit is 6); four drills passed, serving again in 309-384 s | |
| AEON-30 under load | — | 22 of 30 (the same per-task scores as idle) | |

¹ 2.1's same-window run of 2026-10-06 ([2.1's numbers](docs/BENCHMARKS.md#21-in-the-same-window-as-mias-recipe-v18)).
² One boot of this `.env.example` on 2026-10-07, the lab's final check of the combined configuration on these
patches, with the same harnesses; the arena cells are the mean of two runs. Its replies equal, byte for byte, those
of the lab's boot of the same configuration without the tiered schedule, which scored the AEON-30 above. ³ With the
BF16 partial sums the replies differ from 2.1's, so the decode benchmarks generate different text and the drafter
accepts a different share of it: these ratios follow the texts, not a decode change (decode runs as in 2.1). The
soak rows are the soak of this configuration on 2026-10-07: [Soak and failure drills](#soak-and-failure-drills).

[2.1's changes since 2.0](CHANGELOG.md#21-kept-prompts-in-host-ram-hugh-maddens-expert-prompt-kernels-her-v18-fixes-tensorfold-066): kept prompts in host RAM, prompt chunks filled as pairs while streams decode, Hugh Madden's expert prompt kernels, her v1.8 serving fixes and TensorFold v0.6.6.

## What changed since v1.0

[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (2026-09-30) ran Hugh Madden's glm53f-afd with Local Inference Lab's TR3 4-bpw experts. Versions 2.0 to 2.15 keep the three machines and the split, and run the split on TensorFold v0.6.5 with Mia's GLM work and her checkpoint. Against v1.0's published numbers:

- **v1.0 is still better at:**
  - requests at once: 16 slots; 2.15 decodes eight together and queues the rest;
  - decode at four streams, 117.6 tok/s against 2.15's 114.1 in the lab (0.97x), and prose at one stream, 73.1
    against 71.3 (0.98x);
  - AEON-30: 25 of 30, run under its soak load; 2.15 scored 22 of 30, idle and under its soak's load alike;
  - testing: a 4-hour soak with failure drills (on glm53f-afd v1.0.0). 2.15's soak (a 4.11-hour window, 4,071
    requests, no errors) passed its four failure drills but not its probe check: a long streamed tool call
    under load paused 6.3 and 7.0 s between events, over the 6 s allowed, and one gap between probe rounds was
    60 minutes, over the 45 allowed ([Soak and failure drills](#soak-and-failure-drills)).
- **2.15 is better at** prompt processing on long prompts: 3,374 / 3,642 / 3,707 tok/s on fresh 8K / 31K / 62K
  prompts in the lab against v1.0's 3.1-3.3K tok/s on 5K-95K prompts (1.02-1.20x). One stream decodes code and JSON
  faster in the lab (86.1 and 91.4 tok/s against v1.0's 83.7 and 76.2).
  It holds the same ~1.58M-token pool (1,585,152 tokens against ~1.58M) and ships
  v1.0's watcher again. It runs Mia's checkpoint (Apache-2.0) and her serving work: her GLM tool-call fixes, the
  shared-prefix prompt cache and smooth streaming.
- v1.0's numbers come from its own README and BENCHMARKS (decode on its spark-bench-style harness, 1,024-token
  prompts) and were not re-measured here. If many requests at once matter most to you, run
  [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0).

## At a glance

| | |
|---|---|
| **Engine** | [TensorFold](https://github.com/ashhart/TensorFold) v0.6.5 (`609ca41`, Apache-2.0), built from source, plus the 51 patches in [`patches/`](patches/); 0046-0048 are TensorFold v0.6.6's own commits |
| **Wire** | [MCDMA](https://github.com/ashhart/MCDMA) `e672c14` (Apache-2.0): link daemons and `libmcdma-rpc`, built from source on each host; GPU-driven where the device supports CUDA stream memory operations; four links to each Spark, so four MoE exchanges are in flight to it; prompt windows come back as BF16 partial sums |
| **Model id** | `GLM-5.3-Flash-EXL3` (alias `glm-5.3-flash`) on `http://<attention-host>:8000/v1` (configurable) |
| **Weights** | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) `76c0b517` (Apache-2.0; its safetensors are byte-identical to `078455ff`, the revision measured); [incoai DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter `bf582e4e` (CC BY-NC-ND 4.0) |
| **Context** | 262,144 tokens a request; one KV pool of 1,585,152 tokens for all requests; 20 kept prompts' states in pinned host RAM, and up to 24 GiB of evicted prompts parked there |
| **Concurrency** | 8 requests decoding together, one shared cache pool; requests beyond them wait in the queue |
| **Features** | Tool calling (with Mia's GLM tool-call fixes), reasoning (thinking on by default), smooth streaming, the prompt cache with shared-prefix states on a 64-token grid, prompt chunk pairs and four-lane prompt fills, shortest-first prompt order, capacity refusals as HTTP 429 with `Retry-After`, the watcher. Text only: no vision. |

## Performance

2.15 has not been measured against Mia's recipe in a same-window run. The numbers below are the lab's (lab, the same
harnesses as 2.1's numbers): one boot of this `.env.example` on 2026-10-07, against 2.1 as published, which is 2.1's
same-window run of 2026-10-06. That run's comparison with Mia's recipe v1.8 follows
([2.1 against Mia's recipe v1.8](#21-against-mias-recipe-v18)), as 2.1's. v1.0's column is its published numbers (its
own checkpoint).

| | 2.15, lab² | 2.1 as published¹ | 2.15 / 2.1 | v1.0 (glm53f-afd)³ |
|---|---:|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 71.3 / 86.1 / 91.4 | 75.7 / 81.5 / 83.2 | 0.94x / 1.06x / 1.10x⁴ | 73.1 / 83.7 / 76.2 |
| Decode, four streams (aggregate tok/s) | 114.1 | 110.6 | 1.03x⁴ | 117.6 |
| Coding agents, 4 x 1,024 / 8 x 1,024 (tok/s) | 255.8 / 421.7 | 236.2 / — | 1.08x⁴ / — | — |
| Prompt processing, fresh 8K / 31K / 62K prompts (tok/s) | 3,374 / 3,642 / 3,707 | 2,946 / 3,030 / 3,138 | 1.15x / 1.20x / 1.18x | 3.1-3.3K (5K-95K) |
| Cold ~2.6K prompt, first token | 1.058 s | 1.313 s | 0.81x | — |
| Cold 100K prompt, first token | 27.1 s | 32.4 s | 0.84x | — |
| Arena 65,535 x 10: prompt / gen t/s, first token, gen a request | 1,151 / 64.5, 7.67 s, 13.9 | 1,198 / 64.2, 7.70 s, 12.7 | 0.96x / 1.01x, 1.00x, 1.09x | — |
| Arena 100,000 x 5: the same | 1,935 / 68.0, 3.85 s, 18.2 | 1,872 / 65.8, 4.00 s, 18.6 | 1.03x / 1.03x, 0.96x, 0.98x | — |
| KV pool (tokens, all requests) | 1,585,152 | 1,579,008 | 1.00x | ~1.58M |
| AEON-30 | 22 of 30, idle and under the soak's load | 23 of 30 | one fewer | 25 of 30 (under soak load) |

¹ 2.1's same-window run of 2026-10-06 ([below](#21-against-mias-recipe-v18)); its AEON-30 is the lab's boot of 2.1's
configuration, idle. ² One boot of this `.env.example` on 2026-10-07, the lab's final check of the combined
configuration on these patches, with the same harnesses as 2.1's numbers; each arena cell is the mean of two runs.
AEON-30 idle: the lab's boot of this configuration without the tiered schedule, whose replies this boot's equal byte
for byte; under load: the [soak](#soak-and-failure-drills), with the same per-task scores. ³ v1.0's own README
(2026-09-30): glm53f-afd with Local Inference Lab's TR3 4-bpw experts, measured with its spark-bench-style harness on
1,024-token prompts; not re-measured here. ⁴ With the BF16 partial sums the replies differ from 2.1's, so the decode
benchmarks generate different text and the drafter accepts a different share of it: these ratios follow the texts,
not a decode change (decode runs as in 2.1).

- **Against 2.1:** prompts are faster. Fresh 8K-62K prompts 1.15-1.20x, a cold 100K prompt's first token in 0.84x the
  time and a cold ~2.6K one's in 0.81x, spark-bench's ~2.6K-token prompt 1.24x (2,420 against 1,953 tok/s); the
  coding-agent load at four requests 1.08x; the two arena cells within 4% of 2.1's on prompt rate, generation and
  first token, with the ten-client cell's generation per request 1.09x. One-stream decode moves with the BF16 sums'
  texts (0.94x to 1.10x).
- **2.15's levers**, each measured against the configuration without it: [What changed since 2.1](#what-changed-since-21);
  2.1's and 2.0's: [docs/BENCHMARKS.md](docs/BENCHMARKS.md#21s-levers-one-at-a-time).
- Replies are deterministic, drafted replies equal undrafted ones, and each concurrent reply equals its solo reply.
  The 4-bit weights, the FP8 cache and the chunked KDA kernel are byte-identical to her recipe's; the decode kernels
  are byte-identical to the kernels they replace; Hugh Madden's expert prompt kernels are byte-identical to
  glm53f-rank's own at 1 to 4,096 rows, under the default schedule and the big-window one alike; a BF16 partial sum
  is the round-to-nearest-even of the FP32 sum, the same in any window.

### 2.1 against Mia's recipe v1.8

2.1's same-window run, as 2.1 published it: measured in one window on 2026-10-06 (15:57-19:04 EDT), with the same
harnesses on both arms and the arms alternated: 2.1 on one RTX 5090 plus two DGX Sparks, and Mia's recipe v1.8
(`33b50fd`, her latest release that day) on its own two DGX Sparks, with the same checkpoint and drafter. 2.15 was not
in this run; the table says nothing about it.

| | 2.1 | Mia's recipe v1.8, 2x DGX Spark | 2.1 / v1.8 |
|---|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 75.7 / 81.5 / 83.2 | 58.4 / 65.4 / 66.7 | 1.30x / 1.25x / 1.25x |
| Decode, four streams (aggregate tok/s) | 110.6 | 100.7 | 1.10x |
| Coding agents, 4 x 1,024 / 1 x 4,096 (tok/s) | 236.2 / 71.3 | 185.8 / 52.6 | 1.27x / 1.36x |
| Prompt processing, fresh 8K / 31K / 62K prompts (tok/s) | 2,946 / 3,030 / 3,138 | 1,875 / 1,897 / 1,866 | 1.57x / 1.60x / 1.68x |
| Cold ~2.6K prompt, first token | 1.313 s | 1.629 s | 0.81x |
| Cold 100K prompt, first token | 32.4 s | 55.4 s | 0.58x |
| Arena 65,535 x 10: prompt / gen t/s, first token, gen a request | 1,198 / 64.2, 7.7 s, 12.7 | 618.4 / 37.9, 17.3 s, 16.0 | 1.94x / 1.69x, 0.45x, 0.80x |
| Arena 100,000 x 5: the same | 1,872 / 65.8, 4.0 s, 18.6 | 725.3 / 42.0, 8.2 s, 17.2 | 2.58x / 1.57x, 0.49x, 1.08x |
| A needle in ~195K tokens | found (194,832 tokens, prefill 64.1 s) | found (194,832 tokens, prefill 117.1 s) | — |
| KV pool (tokens, all requests) | 1,579,008 | 1,710,080 | 0.92x |
| AEON-30 | 23 of 30⁵ | 23 of 30⁶ | same |

⁵ The lab's boot of 2.1's configuration on 2026-10-06, idle. ⁶ Her recipe v1.8 on its two Sparks on 2026-10-06; another
client's request overlapped that run, so only the score is quoted. Decode: spark-bench's decode script, median of
three runs; the coding agents and the fresh prompts: the mean of two runs; cold first tokens: the median of three
(~2.6K) and the mean of two (100K); each fresh or cold prompt after a warm-up of the same size, with the same text on
both arms; the arena cells with llama-benchy's Spark Arena v2 settings.

In that run 2.1 led her lane on one-stream decode (1.25-1.30x) and four streams (1.10x), the coding-agent load (1.27x
and 1.36x), fresh prompts (1.57-1.68x), cold first tokens (0.81x the time at ~2.6K and 0.58x at 100K), the ~195K
needle (found by both, prefilled in 0.55x the time), and in both arena cells the prompt rate (1.94x and 2.58x),
aggregate generation (1.69x and 1.57x) and the first token (0.45x and 0.49x the time), with generation per request
ahead at five clients too (1.08x). Her lane led on generation per request at ten clients (0.80x: 12.7 against 16.0
tok/s; it decodes four at a time and queues the rest, this split eight) and on the KV pool (0.92x: 1,579,008 against
1,710,080 tokens). Her lane also needs no RTX 5090 and no third machine, and reads pictures; this split serves text
only.

### Soak and failure drills

The soak ran this `.env.example` on 2026-10-07 with the watcher on: a 4.11-hour window of mixed load (agentic, chat,
coding and tool-calling requests, and long prompts of 16K to 100K tokens) at up to eight requests at once, 3.59 hours
of it under load. All 4,071 requests were answered 200, with no restarts. Decode drifted at most -1.9% (spark-bench at
the start, middle and end), memory crept at most +0.23 GiB/h (the attention host; the Sparks' went down), a ~240K
needle was found with eight requests running (239,864 tokens, prefill 80.6 s), AEON-30 scored 22 of 30 under the load
(the same per-task scores as idle), and after the soak and the drills the prompt set ran at the lab's speed (fresh 8K /
31K / 62K 3,290 / 3,647 / 3,709 tok/s, a cold 100K prompt's first token in 27.0 s).

**The soak did not pass its probe check.** 58 of its 60 checked probe replies were right. The two misses were the same
check in two of the nine rounds run under load: a long streamed tool call paused 6.3 and 7.0 s between stream events,
over the probe's 6 s limit. Both calls finished with HTTP 200 and the right tool call; the other loaded rounds peaked
at 2.7-5.0 s, the quiet ones at 2.0 s. Known limitation: under eight requests of mixed load with long prompts, a long
streamed tool call can pause about 7 s between events. The soak's coverage rule missed once too: the largest gap
between probe rounds was 60.0 minutes (the rule: at most 45.0), while AEON-30 ran under the load.

The drills, each fault a `SIGTERM` with the watcher running
([details](docs/BENCHMARKS.md#215s-soak-and-failure-drills)):

| Drill | What the clients saw | Serving again |
|---|---|---|
| An expert node's process stopped ([TensorFold#214](https://github.com/ashhart/TensorFold/issues/214)) | the request in flight got an error event after 23.3 s; requests sent meanwhile got HTTP 429 with `Retry-After: 5` at that moment, later ones HTTP 500 in about 5 ms; `/health` answered 200 throughout | after 309 s, through the watcher's `recover` |
| The attention node's process stopped | connections refused until the recover | after 310 s, through the watcher's `recover` |
| All three containers, then the MCDMA link daemons, stopped | the request in flight was cut after 11.5 s, with no error event | after 384 s, through the watcher's `recover` |
| A cold start of the whole stack by hand | the watcher held off | after 317 s |

Not run for this release: a same-window run against Mia's lane, and the full 28-cell arena grid on the shipped
configuration. Method, every table and the caveats: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Hardware

| Box | What it needs |
|---|---|
| **Attention node** | x86_64 Linux, **RTX 5090 32 GB** (sm_120), Docker + NVIDIA Container Toolkit, a ConnectX-7 (or another RoCE v2 NIC) on the fabric, and about 34 GiB of RAM it can pin for the kept prompts' host tier (`TF_GLM_HOST_CACHE_GIB=24` plus the states; lower it, or drop `TF_GLM_KEPT_HOST`, on a smaller host) |
| **2 × DGX Spark** | GB10 (sm_121), DGX OS, Docker, a ConnectX-7 port on the fabric, the checkpoint on disk (~176 GB), and about 100 GB of free memory before start |
| **Fabric** | RoCE v2, one IPv4 subnet for the three hosts. Every MoE layer of every forward crosses it twice, so latency matters more than bandwidth. |
| **Controller** | Any Linux box with git and passwordless SSH to the three hosts (it can be the attention host) |

## Quick start

```bash
git clone https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks.git
cd GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks
cp .env.example .env     # hosts, fabric addresses, RDMA devices, paths
./build.sh               # TensorFold v0.6.5 + patches/, the images, MCDMA, the CUDA extensions (no model loaded)
./download.sh            # the checkpoint on all three hosts, the drafter on the attention host
./start.sh up            # MCDMA links -> attention node -> expert nodes -> /health -> one real reply
```

```bash
curl http://<attention-host>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-EXL3","messages":[{"role":"user","content":"Hello"}]}'
```

Then `./start.sh status`, `./start.sh logs`, and `./stop.sh`. Without any hardware, `tests/test_static.sh` checks the
scripts, the topology `.env.example` gives and the patch series, `tests/test_dryrun.sh` runs `start.sh up`,
`stop.sh`, `start.sh recover`, `build.sh mcdma` and `download.sh` against stand-ins for ssh, scp and curl and checks
every command line they would run, and `tests/test_watch.sh` runs the watcher against a stand-in server.

Coming from 2.1: `git pull`, keep your `.env` but take `.env.example`'s new `MCDMA_INFLIGHT`, `ATTN_ENV` and
`EXPERT_ENV` lines, then `./stop.sh && ./build.sh && ./start.sh up` (the tree and its CUDA extensions change, and
`up` starts four link daemons on each Spark and two connect daemons). To run the watcher, see
[extras/watch](extras/watch/README.md).

> [!WARNING]
> **One GPU tenant per Spark.** On GB10, CUDA allocations are unified memory and are *not* charged to a container's
> cgroup. A second GPU process on a Spark that serves experts can starve it until only a power cycle recovers it.
> `./start.sh` refuses to start beside another GPU process and waits for memory to come back between runs. Read
> [the troubleshooting page](docs/TROUBLESHOOTING.md#dgx-spark-gb10-memory-safety-read-this-first) first, and consider
> [gb10-hostguard](extras/gb10-hostguard/README.md) on each Spark.

## Using the API

- **Thinking** is on by default. Send `"chat_template_kwargs": {"enable_thinking": false}` (or `"reasoning_effort": "none"`) to turn it off.
- **Tools**: GLM tool calls are parsed by TensorFold's server, with MiaAI-Lab's fix for a call the end token left open. With `TF_GLM_TOOL_CALLS=1` (on in `.env.example`) her fixes for agent clients are on too: each call is streamed whole in its place, a call that ends the think block counts as the reply's call, agent histories with empty or odd arguments render, and a tool-calling reply's reasoning is put back for clients that drop it (`TF_GLM_KEEP_REASONING`). With `tool_choice: "none"` the model can still write call markup as text ([Limits](docs/DESIGN.md#limits)).
- **Capacity**: requests beyond the eight decoding wait in the queue. A request refused for capacity answers HTTP 429 with `Retry-After: 5` (`TF_GLM_CAPACITY_STATUS=1`, her v1.8's patch 0081 by johnwhited), so a client backs off and retries.
- **Health**: `/health` on the API port; its `kept_host` field shows the host RAM tier. If an expert node or a link fails, the requests in flight fail (a streamed one with an error event), new ones get HTTP 429 and then 500 until a restart, and the attention log names the cause, while `/health` still answers 200 ([Operations](#operations)); restart the whole stack (`./start.sh recover`). [`extras/watch`](extras/watch/README.md) checks `/health`, `/v1/models` and a tiny reply every minute and does that for you after two failed checks.

## Operations

| Command | What |
|---|---|
| `./start.sh up` | check, no-model prebuild, MCDMA daemons, attention node, expert nodes, health wait, startup lines, one smoke reply; any failure takes the stack down again |
| `./start.sh check` / `status` / `logs` / `smoke` / `probe` | read-only preflight / containers, health and links / log tails / one real reply / the GPU-driven exchange probe (no model) |
| `./start.sh recover` | after a failure: healthy, it does nothing; else the three containers get `SIGTERM` and start again (an AFD stack cannot take one node back), and the MCDMA daemons too when a link or a daemon is down; rc 3 when it needs you |
| `./stop.sh` / `./stop.sh tf` | `SIGTERM` to the containers, then the connect daemons, then the listen daemons, never `SIGKILL` / containers only |
| `./build.sh [tree\|sync\|images\|mcdma\|ext]` | one build step at a time |
| `tools/export_patches.sh HEAD` | regenerate `patches/` from a TensorFold branch, with every check (maintainers) |
| [`extras/watch/`](extras/watch/README.md) | keep-alive and telemetry from cron or a timer: `/health`, `/v1/models` and a tiny reply every minute, `./start.sh recover` after two failed checks, a circuit breaker, a daily report (ported from v1.0) |
| [`extras/gb10-hostguard/`](extras/gb10-hostguard/README.md) | an optional on-host memory floor and single-GPU-tenant guard for each Spark, with host hardening (from v1.0) |

The containers start with `--restart=no` and `--init`: the scripts decide, and a `SIGTERM` reaches TensorFold. Nothing
starts at boot. `up`, `recover`, `probe` and `stop.sh` take a lock on the controller, one at a time.

When an expert node fails, the request in flight fails rather than hangs (as Ash Hart's answer on
[TensorFold#214](https://github.com/ashhart/TensorFold/issues/214) asks of the split), and the stack needs
`./start.sh recover`, which the watcher runs after two failed checks. As the soak's drill measured it (an expert's
process stopped with `SIGTERM`): the streamed request in flight got an error event and `[DONE]` 23.3 s later, when
the attention node's 20 s heartbeat watchdog fired; requests sent in between waited for that moment and then got
HTTP 429 with `Retry-After: 5` (`TF_GLM_CAPACITY_STATUS=1`; 503 without it); every later request got HTTP 500 in
about 5 ms until the restart. `/health` kept answering 200, so a load balancer that checks only `/health` keeps
sending traffic to a stack that needs a restart. The watcher's `recover` had it serving again 309 s after the
fault. [Troubleshooting](docs/TROUBLESHOOTING.md#serving) has the details.

## What the patches are

| Patches | What | Source |
|---|---|---|
| 0001, 0008, 0011, 0021 | the attention/expert split over MCDMA; `--parallel` on its attention node; its docs page; a GPU test fix | this recipe |
| 0002-0004, 0006 | `--parallel N`: several requests decode together over one cache pool, each reply its solo bits | MiaAI-Lab, [TensorFold#243](https://github.com/ashhart/TensorFold/pull/243), her commits |
| 0005 | a tool call the end token left open is still sent when it parses whole | MiaAI-Lab, [#285](https://github.com/ashhart/TensorFold/pull/285), her commit |
| 0007 | a stopped request ends on every rank within a round | MiaAI-Lab, [#301](https://github.com/ashhart/TensorFold/pull/301), her commit |
| 0009, 0010 | 4-bit dense weights and the FP8 KV cache on the attention node | ports of her recipe patches 0002, 0005, 0013, 0028; 0038, 0041 |
| 0012-0016 | shared-prefix prompt states, the expert prompt kernels' launch order, grouped and layer-sliced prompt fills, smooth streaming | ports of her recipe patches 0015, 0063; 0001, 0020, 0009; 0049; 0062; 0061 |
| 0017, 0018 | her decode kernels: the attention node's decode windows and the routed experts' decode kernel | port of her recipe patches 0016, 0019, 0031, 0043, 0047 |
| 0019 | her noise-aware DFlash2 draft policy (at `fnc5:0.2` here: [the draft policy](docs/BENCHMARKS.md#the-draft-policy)) | port of her recipe patches 0018, 0021 |
| 0020 | her GLM tool-call fixes for agent clients, and kept reasoning | port of her recipe patches 0036, 0051, 0056 |
| 0022 | prompt chunk pairs: one chunk's attention on the 5090 while the Sparks compute the other's experts | this recipe |
| 0023 | her chunked KDA prompt kernel, with the 64-token prompt grid and prompt replay | port of her recipe patches 0012, 0014, 0039, 0008, 0042 |
| 0024 | her EXL3 prompt kernel for the routed experts' prompt chunks | port of her recipe patches 0004, 0009, 0020 |
| 0025 | the pool's room rule: room from the free rows first, evictions only for the rows the pool lacks | this recipe's change to her recipe patch 0030's pool code |
| 0026 | prefill lanes (four, on) and FP8 wire rows (shipped, off: [measured](docs/DESIGN.md#measured-and-off)) | this recipe's code; Hugh Madden authored the designs (glm53f-afd, mimo26f-afd), the row format is T.J. Purtell's (ds41rt) |
| 0027 | several MoE exchanges in flight to each expert node | this recipe's code; Hugh Madden authored the design (glm53f-afd) |
| 0028 | her attention-side prompt kernels behind five switches (on in `.env.example`) | port of her recipe patches 0004, 0009, 0028 |
| 0029 | shortest-first prompt order, with aging | this recipe's change to her recipe patch 0049's grouped chunk |
| 0030 | up to eight concurrent streams | port of her recipe patch 0069 |
| 0031 | 4,096-row prompt chunks and expert calls (`TF_GLM_PREFILL_ROWS`; shipped, off: [measured](docs/DESIGN.md#measured-and-off)) | port of her recipe patches 0004, 0008 |
| 0032 | kept prompts' states, and the rows of prompts the pool evicts, in pinned host RAM (on) | this recipe's code; Hugh Madden authored the design (glm53f-afd's host RAM tier) |
| 0033 | prompt chunks fill as pairs while streams decode (on) | this recipe's change to her recipe patch 0062's sliced fills |
| 0034, 0035 | Hugh Madden's expert prompt kernels for the Sparks' prompt chunks, on every prompt window from 1 row (on) | his code (glm53f-afd, MIT), with parts reimplemented from her recipe patches 0009, 0020 |
| 0036-0041 | her v1.7.1 kept-prompt fixes: a resumed shared prefix copies its rows, waiting requests whose client left are dropped, the pool compacts before it evicts, `<\|assistant\|>` ends a reply, shared-prefix states go by recency past the kept cap; their tests (shipped, off: [measured](docs/DESIGN.md#measured-and-off)) | ports of her recipe patches 0071 (by E-Zou Shen), 0073 (by desy0305), 0074 (by E-Zou Shen), 0075, 0077 |
| 0042-0045 | her v1.8 fixes: a take-over decides which kept prompts stay before it copies any, capacity refusals answer 429, admission at saturation (unset, her default), a stream whose delivery fails ends at once | ports of her recipe patches 0078 (by m-naoki-m), 0081, 0082, 0083 (by johnwhited) |
| 0046-0048 | TensorFold v0.6.6: `--name-priority ID=background`, its local-path fix, the release | TensorFold's own commits, by Philip Mossop and Ash Hart |
| 0049, 0051 | launch-configuration knobs for Hugh Madden's expert prompt kernels, and a row tier: glm53f-rank's own big-window schedule on the Sparks' prompt windows of 1,536 rows and more (on) | this recipe's code; the schedule is glm53f-rank's (glm53f-afd) |
| 0050 | BF16 partial sums on the return wire for prompt windows (on; it changes the replies) | this recipe's code; Hugh Madden authored the design (glm53f-afd's BF16 return planes) |

Details, switches and defaults: [docs/DESIGN.md](docs/DESIGN.md#the-patches). Each port keeps `Co-authored-by: MiaAI-Lab` and names her recipe commit in its credit header, and so do the three changes of this recipe's to her ported code (0025, 0029, 0033) and the patch that reimplements parts of her patches (0034); 0021, 0032 and 0035 keep `Co-authored-by: MiaAI-Lab` too. Two of the ports carry code she adapted from Jay Leaton's glm53-tensorfold-spark (see [NOTICE.md](NOTICE.md)). Patches 0046-0048 keep their TensorFold authors.

## Upstream

This split runs on TensorFold's Python engine. In [TensorFold#214](https://github.com/ashhart/TensorFold/issues/214)
this repo's author proposed the split for TensorFold, and Ash Hart answered on 2026-10-06
([his reply](https://github.com/ashhart/TensorFold/issues/214#issuecomment-6010581196)):

- TensorFold's Python engine is frozen ([#286](https://github.com/ashhart/TensorFold/issues/286), at 0.6.5; a 0.6.6
  followed on 2026-10-06 with one CUDA-server option, `--name-priority`). New work goes into the native Zig engine.
- Cross-machine splits now live in the Zig engine's cluster layer (`zig/src/cluster`), which places one model over
  several machines through mcdma links.
- An attention-on-one-GPU, experts-on-the-Sparks layout would be a new placement there. It waits for GLM-5.3-Flash on
  the Zig engine and for the Zig engine's CUDA serving.
- He cites this split's finding that expert compute, not the wire, sets the round time as the reference for that
  design.

He answered again on 2026-10-07 ([his second reply](https://github.com/ashhart/TensorFold/issues/214#issuecomment-6033151997)):

- The split is a real one for GLM-5.3-Flash, and it wants two commands: a serve process for attention, an expert
  process for the routed experts, and a transport between them.
- The Zig engine has no GLM engine and no remote-expert transport yet. Its single-box GLM port comes first, and the
  split follows it, on the same expert weights rather than a second quantization; the transport will not start ahead
  of the model.
- A failed expert process has to fail the request, not hang the attention side. This recipe does that: in the
  soak's drill the request in flight failed 23.3 s after an expert's process stopped, and the attention side kept
  answering ([Operations](#operations)).

This repo stays on the Python engine: TensorFold v0.6.5 plus v0.6.6's two commits and its release commit
(patches 0046-0048), with the patches above.

## Layout

| Path | What |
|---|---|
| `build.sh`, `download.sh`, `start.sh`, `stop.sh`, `scripts/` | build, weights and control; `scripts/lib.sh` reads `.env` |
| `patches/` | the 51 patches on TensorFold v0.6.5, applied in order with `git am` |
| `docker/Dockerfile` | the serving image: NVIDIA's PyTorch container plus TensorFold's dependencies |
| `tools/` | the patch export and its credits table |
| `docs/` | design, benchmarks, troubleshooting |
| `extras/watch/` | the watcher (keep-alive, telemetry, report), ported from v1.0 |
| `extras/gb10-hostguard/` | the GB10 host guard, from v1.0 |
| `tests/test_static.sh`, `tests/test_dryrun.sh`, `tests/test_watch.sh` | static checks, a dry run of the scripts against stand-ins, and the watcher against a stand-in server; no hardware |
| `AGENTS.md` | Credit and attribution rules for agents that edit this repo, by Mia, unchanged |

## Security

Read this before exposing anything.

- **The API** listens on `API_BIND:API_PORT` (default `0.0.0.0:8000`) with **no authentication** in this recipe. Keep the port on a trusted LAN or VPN, behind a firewall. Never port-forward it to the internet.
- **The rendezvous** (`MASTER_PORT` on the attention host), **the MCDMA control ports** (`MCDMA_CTRL_PORT` on the Sparks) and **the RDMA queue pairs** are **unauthenticated by design**. Put the fabric on its own isolated subnet or VLAN, with no route to your LAN or the internet, and firewall those ports to the three hosts.
- **Privileges:** the containers run with `--network host` (rendezvous and RDMA), `--ipc=host` (MCDMA's mailboxes live in the host's `/dev/shm`), `--device /dev/infiniband`, `--cap-add IPC_LOCK` and `--ulimit memlock=-1` (RDMA memory registration), and `--gpus all`. They run as the image's default user, root. The SSH user needs Docker, which is root-equivalent on those hosts; the scripts use no `sudo`. The MCDMA daemons run as the SSH user.
- **Kept prompts in host RAM:** with `TF_GLM_KEPT_HOST=1` the attention host's RAM holds the kept prompts' states and evicted prompts' KV rows, as the 5090 holds the rest of the cache. They live in the container's pinned memory only and go when it stops; nothing is written to disk.
- **Remote commands are built from `.env`** and run over SSH: treat `.env` as code. The extra `TF_GLM_*` switches (`ATTN_ENV`, `EXPERT_ENV`) are checked against a strict pattern before they reach `docker run`.
- **Secrets:** none are needed; `.env` is git-ignored anyway. Don't put keys in `.env` values that end up in container arguments, which `docker inspect` shows.
- **`extras/gb10-hostguard`**, if you install it, enables kernel panic-on-lockup and the hardware watchdog, so a hung box reboots itself. Read its README first.

## AI-assisted development

Most of this recipe was planned, written, run and measured by AI agents, with a human (me) setting direction and approving anything that touched the machines or went public. Every number in it was measured, not generated. Still, read the scripts before running them.

| Agent | Model | How it was used |
|---|---|---|
| Planner and executor | **Claude Opus 5.5** (Anthropic), in the [Hermes](https://hermes-agent.nousresearch.com) agent harness | Planned the work as a Kanban board of cards, and did much of the execution: the TensorFold 0.6.5 rebase, the ports of Mia's patches, prompt chunk pairs, the exchanges in flight, the pool's room rule, shortest-first prompt order, the host RAM tier, the fills during decode, the port of Hugh Madden's expert prompt kernels, the BF16 partial sums, the kernels' schedule knobs and their tier, the second connect daemon for four links, the watcher's port from v1.0, the hardware checks, these scripts and docs. |
| Earlier executor | **GLM-5.3-Flash** (zai-org), self-hosted, via Hermes | Earlier execution work on the split. |

The engine, the transport, the checkpoint, the expert prompt kernels and most of the hard ideas belong to the people credited below. The agents' part is the integration: the split over MCDMA, the ports onto it, prompt chunk pairs, the exchanges in flight, the room rule and shortest-first order, the host RAM tier, the fills during decode and the BF16 return wire written for this tree, and the operations and measurement around them. The watcher is v1.0's: v1.0's orchestrating agent (Claude Opus 5.5 in Hermes) wrote it, as v1.0's credits say, and this version ports it.

## Support

**None.** This is provided **as-is**, without warranty of any kind (see [LICENSE](LICENSE)). It runs on one homelab; your hardware, firmware, fabric and driver versions will differ. It drives GPUs and hosts hard, and a DGX Spark that runs out of memory can need a power cycle. You use it at your own risk. Engine questions belong upstream at [ashhart/TensorFold](https://github.com/ashhart/TensorFold), transport questions at [ashhart/MCDMA](https://github.com/ashhart/MCDMA), and questions about Mia's recipe at [hers](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). Issues and PRs here may not get a response.

## License and credits

This version's scripts, patches and docs are **Apache-2.0** ([LICENSE](LICENSE)), the licence of TensorFold from v0.6.0 and of Mia's recipe, whose work it carries. [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) stays **MIT**, as published. This version vendors no third-party code beyond the patches and ships no weights or binaries. Everything it builds or downloads keeps its own licence, listed in **[NOTICE.md](NOTICE.md)**:

- **engine:** TensorFold, Apache-2.0 at v0.6.5 and v0.6.6;
- **wire:** MCDMA, Apache-2.0;
- **MiaAI-Lab's pull requests and recipe patches:** Apache-2.0, with code she adapted from Jay Leaton's glm53-tensorfold-spark (Apache-2.0) in two of them, and her recipe's contributed patches (E-Zou Shen's, desy0305's, m-naoki-m's and johnwhited's) under her recipe's licence;
- **Hugh Madden's expert prompt kernels in patches 0034 and 0035:** glm53f-afd's code, **MIT** (Copyright (c) 2026 Turquoise Bay AI Pty Ltd), with the MIT code of TensorFold's contributors it carries; both licence texts ship in the patch;
- **designs written anew in patches 0026, 0027, 0032 and 0050:** Hugh Madden's glm53f-afd and mimo26f-afd (MIT) and T.J. Purtell's ds41rt (MIT); no code from them is copied;
- **base model:** Z.AI's GLM-5.3-Flash, MIT;
- **checkpoint:** Mia's AI Lab's EXL3 quantization, Apache-2.0;
- **DFlash2 drafter:** CC BY-NC-ND 4.0, **non-commercial**, referenced only.

All the hard parts are other people's work:

- **[TensorFold](https://github.com/ashhart/TensorFold)** and **[MCDMA](https://github.com/ashhart/MCDMA)** by Ash Hart ([@ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart)). Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp), [@petruspennanen](https://x.com/petruspennanen)) wrote MCDMA's setup notes and build fix, [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5). Philip Mossop ([@philip-pentatonic](https://github.com/philip-pentatonic)) wrote TensorFold v0.6.6's `--name-priority`, [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445) (patch 0046).
- **Mia's AI Lab** ([@MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)): the checkpoint; the [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) whose patches are ported here and against which this version is measured; her TensorFold pull requests #243, #285 and #301. This README uses the layout of the README of her [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf).
- **The contributors to her recipe whose patches are ported here:** E-Zou Shen ([@ezoushen](https://github.com/ezoushen), [@ezoushen](https://x.com/ezoushen)) wrote her patches 0071 and 0074 (here 0036 and 0038, tests in 0041; her recipe's pull requests #44 and #62); desy0305 ([@desy0305](https://github.com/desy0305)) wrote her patch 0073 and its checks (here 0037; #51); johnwhited ([@johnwhited](https://github.com/johnwhited)) wrote her patches 0081, 0082 and 0083 and 0073's delivery-failure handling (here 0043-0045 and 0037; #48), with vLLM v1's admission and abort behaviour as his reference; m-naoki-m ([@m-naoki-m](https://github.com/m-naoki-m), [@\_m\_naoki\_m\_](https://x.com/_m_naoki_m_)) wrote her patch 0078 (here 0042; #71, also TensorFold#421). meleesciony ([@meleesciony](https://github.com/meleesciony)) diagnosed the failure of her [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75) and proposed the fix her patch 0077 makes (here 0040 and 0041); Lukas-tek-no-logic ([@Lukas-tek-no-logic](https://github.com/Lukas-tek-no-logic)) reported the `<|assistant|>` leak her patch 0075 fixes ([issue #60](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/60); here 0039).
- **Jay Leaton** ([@jayleaton](https://x.com/jayleaton), [jayleaton](https://github.com/jayleaton)) wrote [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark); two of Mia's ported patches (here 0017 and 0020) carry code she adapted from its patches 0580 and 0620, at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4).
- **Hugh Madden / Turquoise Bay AI** ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)) wrote [glm53f-afd](https://github.com/hughmadden/glm53f-afd) (`91db3cc`), the engine of v1.0, which serves GLM-5.3-Flash with attention, the KV cache, drafting and sampling on an RTX 5090 and the routed experts on DGX Sparks, and [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd), its serving shell, wire and RDMA transport. This version runs that split on TensorFold. Patches 0034 and 0035 carry his code: glm53f-rank's expert prompt kernels (the plan, the large-M gate/up and down kernels, the fused epilogue and reduce of `crates/glm53f-rank/kernels/exl3_rank.cu`), which the Sparks run for every prompt chunk; their EXL3 decoder, fragment MMA and Hadamard butterfly are TensorFold's MIT code (`bb4b4a3`) as glm53f-afd carries it. Patches 0026, 0027 and 0032 write four of his designs for this tree, with no code copied: the exchanges kept in flight ahead of the expert ranks (glm53f-afd `91db3cc`, `crates/glm53f-serve/src/lib.rs`; on here), the host RAM tier for kept prompts (`crates/glm53f-coordinator/src/hostcache.rs` and `scheduler.rs`; on), the prefill in two and four lanes (mimo26f-afd [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79), glm53f-afd `91db3cc`; four lanes, on) and FP8 wire rows (off). Patch 0050 writes a fifth: glm53f-afd's BF16 return planes, each expert rank's sum rounded to BF16 and the planes added in FP32 (`crates/glm53f-rank/README.md`; on, for prompt windows). Patches 0049 and 0051 run glm53f-rank's own schedule for its biggest windows on the Sparks' prompt windows of 1,536 rows and more.
- **T.J. Purtell** ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell)) wrote [ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and [cuteafd](https://github.com/tpurtell/cuteafd), engines that run attention on RTX GPUs and the routed experts on DGX Sparks. This version uses no code from them. Hugh Madden's expert prompt kernels (patch 0034) split every expert by intermediate channel as glmrt does, which T.J. Purtell designed; glm53f-afd's host RAM tier, whose design patch 0032 follows, takes its host cache's eviction design from ds41rt, through mimo26f-afd. Patch 0026's FP8 wire rows (off) use ds41rt's row format, E4M3 with a UE8M0 scale per 32 values, which reached glm53f-afd through mimo26f-afd.
- **fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** (MIT) implements the chunkwise algorithm of gated delta networks and Kimi Delta Attention that Mia's chunked KDA kernel computes in its chunked WY / UT form; her kernel is her own code.
- **Patryk Mikołajczyk** ([@mikolaj92](https://github.com/mikolaj92)) wrote [TensorFold#140](https://github.com/ashhart/TensorFold/pull/140), which bounds GLM's DSA selection to the pools a row can see; Mia's decode patch 0043 (here in 0017) extends that bound to decode. **Aditya Thyagarajan** ([@aditya1503](https://github.com/aditya1503)) wrote [TensorFold#200](https://github.com/ashhart/TensorFold/pull/200), MTP concurrency, which #243 superseded.
- **Weights:** [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) (the base model), Mia's AI Lab (the EXL3 quantization, made with turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3)), [IncoAI](https://huggingface.co/incoai) (the DFlash2 drafter).
- **[AGENTS.md](AGENTS.md):** the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), unchanged. Agents that edit this repo follow them.
- **v1.0's credits** (glm53f-afd and its upstreams, Local Inference Lab's weights, and everyone else v1.0 names) stay in [NOTICE.md](NOTICE.md#previous-version-v10), verbatim.
- **This recipe:** [@squarrier](https://github.com/squarrier), with the AI agents above.
