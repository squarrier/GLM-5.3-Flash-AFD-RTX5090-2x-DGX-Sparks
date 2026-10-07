<p align="center">
  <b>GLM-5.3-Flash on 1× RTX 5090 + 2× DGX Spark</b><br>
  <sub>An attention/expert split on <a href="https://github.com/ashhart/TensorFold">TensorFold</a>: attention, KV cache and drafter on the 5090, the routed experts on two Sparks, one RDMA exchange per MoE layer over <a href="https://github.com/ashhart/MCDMA">MCDMA</a></sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RTX_5090-attention_node-76b900?style=flat-square" alt="RTX 5090 attention node">
  <img src="https://img.shields.io/badge/2%C3%97_DGX_Spark-expert_nodes-2ea44f?style=flat-square" alt="2× DGX Spark expert nodes">
  <img src="https://img.shields.io/badge/MCDMA-RoCE_v2_RDMA-0969da?style=flat-square" alt="MCDMA over RoCE v2">
  <img src="https://img.shields.io/badge/TensorFold-v0.6.5_%2B_48_patches-6f42c1?style=flat-square" alt="TensorFold v0.6.5 + 48 patches">
  <img src="https://img.shields.io/badge/license-Apache--2.0-555?style=flat-square" alt="Apache-2.0">
</p>

This repo is a deployment recipe, not a new engine. Version 2.1 serves Z.ai's [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as MiaAI-Lab's EXL3 checkpoint with **[TensorFold](https://github.com/ashhart/TensorFold)** v0.6.5 by Ash Hart ([@ashhart](https://github.com/ashhart)), plus 48 patches (the last three are TensorFold v0.6.6's own commits), on three machines:

- An **x86 host with one RTX 5090** is the attention node. It runs attention, holds the KV cache, runs the DFlash2 drafter and sampling, and serves the OpenAI-compatible API. Its RAM holds the kept prompts' states and the prompts the 5090's pool evicts.
- **Two DGX Sparks (GB10)** each hold one half of every MoE layer's routed experts.
- In every MoE layer the 5090 sends its rows to both Sparks and gets two partial sums back over **RoCE v2 RDMA**, through Ash Hart's **[MCDMA](https://github.com/ashhart/MCDMA)** link daemons, with two exchanges in flight to each Spark.

> **Where v1.0 is better: requests at once, four-stream decode, one-stream code and testing; on long prompts it is as fast or a little faster.** [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0), this repo's glm53f-afd version of 2026-09-30, prefills 3.1-3.3K tok/s on 5K-95K prompts; this version prefills 2,946 / 3,030 / 3,138 tok/s on fresh 8K / 31K / 62K prompts (0.89-1.01x). At four streams it decodes 110.6 tok/s against v1.0's 117.6; at one stream 75.7 / 81.5 / 83.2 against 73.1 / 83.7 / 76.2. Both now keep a ~1.58M-token KV pool (1,579,008 here). v1.0 also serves more requests at once and was tested longer: [What changed since v1.0](#what-changed-since-v10).

> **Where Mia's two-Spark lane is better:** it needs two machines, not three, and no RTX 5090; it reads pictures (this split serves text only); and, in the same window as this version, each request generates faster at ten clients (16.0 against 12.7 tok/s: her lane decodes four at a time and queues the rest, this split eight) and her KV pool is 8% larger (1,710,080 against 1,579,008 tokens). See [Performance](#performance).

It serves MiaAI-Lab's checkpoint the way her own [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) does (4-bit dense weights, FP8 KV cache, DFlash2, her decode kernels, chunked KDA prompts, her attention-side prompt kernels, her v1.8 serving fixes) and carries her GLM work: her three open TensorFold pull requests as her commits, and ports of her recipe's patches through her v1.8, each credited. Where it differs from her main lane, the difference was measured: eight streams instead of four, her draft policy at `fnc5:0.2` instead of `fnc7:0.3`, 2,048-row prompt chunks instead of her 4,096 (GB10's limit on this split), her v1.7.1 kept-prompt patches off, this recipe's own levers for the split (prompt chunk pairs, two MoE exchanges in flight, the pool's room rule, shortest-first prompt order, prompt chunks filled as pairs while streams decode), and two of Hugh Madden's glm53f-afd designs: kept prompts in host RAM, and his expert prompt kernels on the Sparks in place of her EXL3 prompt kernel. Her attention-side prompt kernels are on, as in her lane, and so are his expert prompt kernels: both change the replies, and AEON-30 scored 23 of 30 with them (the bar was 21). See [What the patches are](#what-the-patches-are).

## Credits

| Author | Profiles | What they authored, as used here | Repo, commit |
|---|---|---|---|
| **Mia** (Mia's AI Lab) | [MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab) | the EXL3 4-bpw checkpoint; her TensorFold pull requests #243, #285 and #301 (patches 0002-0007, her commits); the 26 patches here that port her recipe's patches (among them her EXL3 prompt kernel, 0004; her attention-side prompt kernels, 0004, 0009 and 0028; eight streams, 0069; her v1.7.1 and v1.8 fixes, 0071-0083), the code of hers that patches 0025, 0029 and 0033 change (her 0030, 0049 and 0062), and the parts of her 0009 and 0020 that patch 0034 reimplements; [AGENTS.md](AGENTS.md); this README's layout | [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at [`cf28cc4`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1) (v1.4), [`1576746`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) (v1.5), [`68ebd67`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/68ebd67b5326974b8004009e202268b1fa7c551d) (v1.7.1) and [`33b50fd`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/33b50fde06fd7ea604cbc6a663880068ab1e2ee4) (v1.8); [the checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) at [`76c0b517`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/76c0b5173166d2795dd48860f45d8224817f894c) |
| **Ash Hart** | [ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart) | TensorFold, the engine, and two of TensorFold v0.6.6's commits (patches 0047 and 0048, his commits); TensorFold's MIT code that Hugh Madden's expert prompt kernels carry (the EXL3 decoder, fragment MMA and Hadamard butterfly, patch 0034); MCDMA, the RDMA wire | [TensorFold](https://github.com/ashhart/TensorFold) at [`609ca41`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f) (v0.6.5), [`cb2ebf0`](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248) (v0.6.6) and `bb4b4a3` (v0.3.4.1, MIT); [MCDMA](https://github.com/ashhart/MCDMA) at [`e672c14`](https://github.com/ashhart/MCDMA/tree/e672c14ff9fc7b38994caf73025cf1588b4de74e) |
| **Hugh Madden** / Turquoise Bay AI | [hughmadden](https://github.com/hughmadden), [@dangerm00se](https://x.com/dangerm00se) | glm53f-afd, the engine of v1.0: GLM-5.3-Flash with attention on an RTX 5090 and the routed experts on DGX Sparks, the split this version runs on TensorFold; his expert prompt kernels (glm53f-rank's large-M EXL3 kernels), whose code patches 0034 and 0035 carry for the Sparks' prompt chunks (on); the designs patches 0026, 0027 and 0032 write for it: the exchanges kept in flight ahead of the expert ranks (on), the host RAM tier for kept prompts (on), the two- and four-lane prefill and the FP8 wire rows (off) | [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at [`91db3cc`](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) (v1.1.0); [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) at [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79) |
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
| See what changed since 2.0, or since v1.0 | [What changed since 2.0](#what-changed-since-20), [since v1.0](#what-changed-since-v10) |
| Know where TensorFold's own split work stands | [Upstream](#upstream) |
| Run the earlier versions | [2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0) (TensorFold), [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (glm53f-afd) |

## What changed since 2.0

[2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0) (2026-10-05) is this repo's first TensorFold version. 2.1 keeps its engine, machines, checkpoint and drafter, and adds 18 patches (0031-0048) and six switches. Each lever below was measured in the lab against the same configuration without it ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#21s-levers-one-at-a-time)):

- **Kept prompts in host RAM** (patch 0032; Hugh Madden authored the design, glm53f-afd's host RAM tier): the 20 kept prompts' states live in pinned host RAM instead of the 5090, so the KV pool grows from 727,040 to 1,579,008 tokens (2.17x) under the same memory gate, with the cache budget at 8.5 GiB instead of 6.5. A prompt the pool evicts for room parks its rows in 24 GiB of pinned pages and resumes without a prefill: first token 0.15-0.44 s instead of 12-67 s in the lab's eviction test (a prompt dropped by the 20-prompt cap is not parked). The lab's arena cells never needed the bigger pool, so it made them no faster (up to 5% slower, single runs). It pins 33.35 GiB of the attention host's RAM.
- **Prompt chunks fill as pairs while streams decode** (patch 0033, this recipe's change to her recipe patch 0062's sliced fills): on the arena cells at five and ten clients the first token comes 30-33% sooner and aggregate decode is 24-27% higher, with the same replies; decode per request barely moves (+4.1% at 65,535 x 10). The cost: cold long prompts that arrive together fill 7-10% slower, and two cold 200K prompts at once wait 16% longer on average.
- **Hugh Madden's expert prompt kernels on the Sparks** (patches 0034 and 0035, his code from glm53f-afd): fresh prompts 13.6% faster (geometric mean of 8K / 31K / 62K), a cold 100K prompt's first token 11.8% sooner, the coding-agent load 3.9% and 10.3% faster. They change the replies, and decode moves with the texts: spark-bench's geometric mean 2.1% lower (JSON 8.3%), because the drafter accepts the new texts at a different rate. AEON-30 23 of 30 with them.
- **Her v1.8 serving fixes** (patches 0042, 0043 and 0045, by m-naoki-m and johnwhited, on as in her lane): a request refused for capacity answers HTTP 429 with `Retry-After: 5`. The take-over decision (0042) and the delivery abort (0045) guard paths this split does not take at eight streams. No measured cost (spark-bench +0.06%).
- **TensorFold v0.6.6** (patches 0046-0048: Philip Mossop's `--name-priority ID=background`, Ash Hart's fix to it and his release commit), carried with their authorship. This recipe sets no `--name-priority`.
- **In the patches, off:** 4,096-row prompt chunks (0031: on GB10 the expert nodes' grouping kernel cannot launch a prompt window over 2,812 rows, so do not set it), her v1.7.1 kept-prompt five (0036-0041: the ten-client cell's prompt rate 8.1% lower) and admission at saturation (0044, unset as in her lane). [docs/DESIGN.md](docs/DESIGN.md#measured-and-off) has each.

2.0 as shipped against 2.1: the ratios use the lab's boot of 2.1; the last column is 2.1 in the same-window run.

| | 2.0 as shipped¹ | 2.1, lab² | 2.1 / 2.0 | 2.1, same window³ |
|---|---:|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 77.4 / 79.0 / 90.4 | 75.0 / 80.8 / 83.1 | 0.97x / 1.02x / 0.92x | 75.7 / 81.5 / 83.2 |
| Decode, four streams (aggregate tok/s) | 111.6 | 109.0 | 0.98x | 110.6 |
| spark-bench's ~2.6K-token prompt (tok/s) | 1,779 | 1,953 | 1.10x | 1,953 |
| Cold ~2.6K prompt, first token | 1.449 s | 1.325 s | 0.91x | 1.313 s |
| Fresh 8K / 31K / 62K prompts (tok/s) | 2,599 / 2,618 / 2,762 | 2,890 / 2,992 / 3,145 | 1.11x / 1.14x / 1.14x | 2,946 / 3,030 / 3,138 |
| Cold 100K prompt, first token | 36.8 s | 32.3 s | 0.88x | 32.4 s |
| Coding agents, 4 x 1,024 / 1 x 4,096 (tok/s) | 229.3 / 64.4 | 237.4 / 70.9 | 1.04x / 1.10x | 236.2 / 71.3 |
| Arena 65,535 x 10: prompt / gen t/s, first token, gen a request | 833.9 / 46.1, 13.4 s, 12.0 | 1,147 / 63.9, 7.8 s, 13.2 | 1.38x / 1.39x, 0.58x, 1.10x | 1,198 / 64.2, 7.7 s, 12.7 |
| Arena 100,000 x 5: the same | 1,047.6 / 50.3, 6.4 s, 18.6 | 1,904 / 68.4, 3.9 s, 18.9 | 1.82x / 1.36x, 0.61x, 1.02x | 1,872 / 65.8, 4.0 s, 18.6 |
| KV pool (tokens, all requests) | 727,040 | 1,579,008 | 2.17x | 1,579,008 |
| AEON-30 (idle) | 23 of 30 | 23 of 30 | same | (the lab's) |

¹ One boot of 2.0's `.env.example` on 2026-10-05 ([2.0's numbers](docs/BENCHMARKS.md#20-as-shipped)). ² One boot of this
`.env.example` on 2026-10-06, the lab's final check of the combined configuration on these patches, with the same
harnesses; a different day and boot from 2.0's. ³ The same-window run that [Performance](#performance) reports.

## What changed since v1.0

[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (2026-09-30) ran Hugh Madden's glm53f-afd with Local Inference Lab's TR3 4-bpw experts. Versions 2.0 and 2.1 keep the three machines and the split, and run the split on TensorFold v0.6.5 with Mia's GLM work and her checkpoint. Against v1.0's published numbers:

- **v1.0 is still better at:**
  - prompt processing on long prompts, where it is as fast or a little faster: 3.1-3.3K tok/s on 5K-95K prompts;
    2.1 does 2,946 / 3,030 / 3,138 tok/s on fresh 8K / 31K / 62K prompts (0.89-1.01x);
  - requests at once: 16 slots; 2.1 decodes eight together and queues the rest;
  - decode at four streams, 117.6 tok/s against 2.1's 110.6 (0.94x), and code at one stream, 83.7 against 81.5
    (0.97x);
  - AEON-30: 25 of 30, run under its soak load; 2.1 scored 23 of 30, idle;
  - testing: a 4-hour soak with failure drills (on glm53f-afd v1.0.0), and `extras/watch`, which runs `recover`
    after two failed checks; 2.1 has had neither a soak nor failure drills (2.0 ran 81.6 minutes of mixed load), and
    does not ship the watcher.
- **2.1 is better at** one-stream decode on prose and JSON (75.7 and 83.2 against 73.1 and 76.2: 1.04x and 1.09x).
  It holds the same ~1.58M-token pool (1,579,008 tokens against ~1.58M). It runs Mia's
  checkpoint (Apache-2.0) and her serving work: her GLM tool-call fixes, the shared-prefix prompt cache and smooth
  streaming.
- v1.0's numbers come from its own README and BENCHMARKS (decode on its spark-bench-style harness, 1,024-token
  prompts) and were not re-measured here. If long prompts, many requests at once, or a longer-tested build matter
  most to you, run [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0).

## At a glance

| | |
|---|---|
| **Engine** | [TensorFold](https://github.com/ashhart/TensorFold) v0.6.5 (`609ca41`, Apache-2.0), built from source, plus the 48 patches in [`patches/`](patches/); the last three are TensorFold v0.6.6's own commits |
| **Wire** | [MCDMA](https://github.com/ashhart/MCDMA) `e672c14` (Apache-2.0): link daemons and `libmcdma-rpc`, built from source on each host; GPU-driven where the device supports CUDA stream memory operations; two links to each Spark, so two MoE exchanges are in flight to it |
| **Model id** | `GLM-5.3-Flash-EXL3` (alias `glm-5.3-flash`) on `http://<attention-host>:8000/v1` (configurable) |
| **Weights** | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) `76c0b517` (Apache-2.0; its safetensors are byte-identical to `078455ff`, the revision measured); [incoai DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter `bf582e4e` (CC BY-NC-ND 4.0) |
| **Context** | 262,144 tokens a request; one KV pool of 1,579,008 tokens for all requests; 20 kept prompts' states in pinned host RAM, and up to 24 GiB of evicted prompts parked there |
| **Concurrency** | 8 requests decoding together, one shared cache pool; requests beyond them wait in the queue |
| **Features** | Tool calling (with Mia's GLM tool-call fixes), reasoning (thinking on by default), smooth streaming, the prompt cache with shared-prefix states on a 64-token grid, prompt chunk pairs, shortest-first prompt order, capacity refusals as HTTP 429 with `Retry-After`. Text only: no vision. |

## Performance

Measured in one window on 2026-10-06 (15:57-19:04 EDT), with the same harnesses on both arms and the arms alternated:
this version on one RTX 5090 plus two DGX Sparks, and Mia's recipe v1.8 (`33b50fd`, her latest release that day) on
its own two DGX Sparks, with the same checkpoint and drafter. v1.0's column is its published numbers (its own checkpoint).

| | This version (2.1) | Mia's recipe v1.8, 2x DGX Spark | 2.1 / v1.8 | v1.0 (glm53f-afd)¹ |
|---|---:|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 75.7 / 81.5 / 83.2 | 58.4 / 65.4 / 66.7 | 1.30x / 1.25x / 1.25x | 73.1 / 83.7 / 76.2 |
| Decode, four streams (aggregate tok/s) | 110.6 | 100.7 | 1.10x | 117.6 |
| Coding agents, 4 x 1,024 / 1 x 4,096 (tok/s) | 236.2 / 71.3 | 185.8 / 52.6 | 1.27x / 1.36x | — |
| Prompt processing, fresh 8K / 31K / 62K prompts (tok/s) | 2,946 / 3,030 / 3,138 | 1,875 / 1,897 / 1,866 | 1.57x / 1.60x / 1.68x | 3.1-3.3K (5K-95K) |
| Cold ~2.6K prompt, first token | 1.313 s | 1.629 s | 0.81x | — |
| Cold 100K prompt, first token | 32.4 s | 55.4 s | 0.58x | — |
| Arena 65,535 x 10: prompt / gen t/s, first token, gen a request | 1,198 / 64.2, 7.7 s, 12.7 | 618.4 / 37.9, 17.3 s, 16.0 | 1.94x / 1.69x, 0.45x, 0.80x | — |
| Arena 100,000 x 5: the same | 1,872 / 65.8, 4.0 s, 18.6 | 725.3 / 42.0, 8.2 s, 17.2 | 2.58x / 1.57x, 0.49x, 1.08x | — |
| A needle in ~195K tokens | found (194,832 tokens, prefill 64.1 s) | found (194,832 tokens, prefill 117.1 s) | — | — |
| KV pool (tokens, all requests) | 1,579,008 | 1,710,080 | 0.92x | ~1.58M |
| AEON-30 | 23 of 30² | 23 of 30³ | same | 25 of 30 (under soak load) |

¹ v1.0's own README (2026-09-30): glm53f-afd with Local Inference Lab's TR3 4-bpw experts, measured with its
spark-bench-style harness on 1,024-token prompts; not re-measured here. ² The lab's boot of this configuration on
2026-10-06, idle. ³ Her recipe v1.8 on its two Sparks on 2026-10-06; another client's request overlapped that run, so
only the score is quoted. Decode: spark-bench's decode script, median of three runs; the coding agents and the fresh
prompts: the mean of two runs; cold first tokens: the median of three (~2.6K) and the mean of two (100K); each
fresh or cold prompt after a warm-up of the same size, with the same text on both arms; the arena cells with
llama-benchy's Spark Arena v2 settings.

- **Where this version leads her lane:** one-stream decode (1.25-1.30x) and four streams (1.10x); the coding-agent load
  (1.27x and 1.36x); fresh prompts (1.57-1.68x); cold first tokens in 0.81x the time at ~2.6K and 0.58x at 100K; the
  ~195K needle, found by both, prefilled in 0.55x the time; and in both arena cells the prompt rate (1.94x and 2.58x),
  aggregate generation (1.69x and 1.57x) and the first token (0.45x and 0.49x the time), with generation per request
  ahead at five clients too (1.08x).
- **Where her lane leads this version:** generation per request at ten clients (0.80x: 12.7 against 16.0 tok/s; her
  lane decodes four at a time and queues the rest, this one eight), and the KV pool (0.92x: 1,579,008 against
  1,710,080 tokens). Her lane also needs no RTX 5090 and no third machine, and reads pictures; this split serves text
  only.
- **2.1's levers**, each measured against the configuration without it: [What changed since 2.0](#what-changed-since-20);
  2.0's levers (her EXL3 prompt kernel, two exchanges in flight, the room rule and twenty kept prompts, eight
  streams, shortest-first order, her draft policy and her attention-side prompt kernels):
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#20s-levers-one-at-a-time).
- Replies are deterministic, drafted replies equal undrafted ones, and each concurrent reply equals its solo reply.
  The 4-bit weights, the FP8 cache and the chunked KDA kernel are byte-identical to her recipe's; the decode kernels
  are byte-identical to the kernels they replace; Hugh Madden's expert prompt kernels are byte-identical to
  glm53f-rank's own at 1 to 4,096 rows.

Not run for this release: the full 28-cell arena grid on the shipped configuration, a soak, and failure drills. Method,
every table and the caveats: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

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
scripts, the topology `.env.example` gives and the patch series, and `tests/test_dryrun.sh` runs `start.sh up`,
`stop.sh`, `build.sh mcdma` and `download.sh` against stand-ins for ssh, scp and curl and checks every command line they
would run.

Coming from 2.0: `git pull`, keep your `.env` but take `.env.example`'s new `CACHE_GIB`, `ATTN_ENV` and `EXPERT_ENV`
lines, then `./stop.sh && ./build.sh && ./start.sh up` (the tree, the images and the CUDA extensions all change).

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
- **Health**: `/health` on the API port; its `kept_host` field shows the host RAM tier. If an expert node or a link fails, in-flight requests fail with HTTP 503 and the attention log names the cause; restart the whole stack (`./stop.sh && ./start.sh up`).

## Operations

| Command | What |
|---|---|
| `./start.sh up` | check, no-model prebuild, MCDMA daemons, attention node, expert nodes, health wait, startup lines, one smoke reply; any failure takes the stack down again |
| `./start.sh check` / `status` / `logs` / `smoke` / `probe` | read-only preflight / containers, health and links / log tails / one real reply / the GPU-driven exchange probe (no model) |
| `./stop.sh` / `./stop.sh tf` | containers, then the connect daemon, then the listen daemons (never SIGKILL) / containers only |
| `./build.sh [tree\|sync\|images\|mcdma\|ext]` | one build step at a time |
| `tools/export_patches.sh HEAD` | regenerate `patches/` from a TensorFold branch, with every check (maintainers) |
| [`extras/gb10-hostguard/`](extras/gb10-hostguard/README.md) | an optional on-host memory floor and single-GPU-tenant guard for each Spark, with host hardening (from v1.0) |

The containers start with `--restart=no`: the scripts decide. Nothing starts at boot. The watcher and recovery
helper of v1.0 (`extras/watch`) are not in 2.1; they come back once a soak has exercised them on this version.

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
| 0026 | prefill lanes and FP8 wire rows (shipped, off: [measured](docs/DESIGN.md#measured-and-off)) | this recipe's code; Hugh Madden authored the designs (glm53f-afd, mimo26f-afd), the row format is T.J. Purtell's (ds41rt) |
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

This repo stays on the Python engine: TensorFold v0.6.5 plus v0.6.6's two commits and its release commit
(patches 0046-0048), with the patches above.

## Layout

| Path | What |
|---|---|
| `build.sh`, `download.sh`, `start.sh`, `stop.sh`, `scripts/` | build, weights and control; `scripts/lib.sh` reads `.env` |
| `patches/` | the 48 patches on TensorFold v0.6.5, applied in order with `git am` |
| `docker/Dockerfile` | the serving image: NVIDIA's PyTorch container plus TensorFold's dependencies |
| `tools/` | the patch export and its credits table |
| `docs/` | design, benchmarks, troubleshooting |
| `extras/gb10-hostguard/` | the GB10 host guard, from v1.0 |
| `tests/test_static.sh`, `tests/test_dryrun.sh` | static checks, and a dry run of the scripts against stand-ins; no hardware |
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
| Planner and executor | **Claude Opus 5.5** (Anthropic), in the [Hermes](https://hermes-agent.nousresearch.com) agent harness | Planned the work as a Kanban board of cards, and did much of the execution: the TensorFold 0.6.5 rebase, the ports of Mia's patches, prompt chunk pairs, the exchanges in flight, the pool's room rule, shortest-first prompt order, the host RAM tier, the fills during decode, the port of Hugh Madden's expert prompt kernels, the hardware checks, these scripts and docs. |
| Earlier executor | **GLM-5.3-Flash** (zai-org), self-hosted, via Hermes | Earlier execution work on the split. |

The engine, the transport, the checkpoint, the expert prompt kernels and most of the hard ideas belong to the people credited below. The agents' part is the integration: the split over MCDMA, the ports onto it, prompt chunk pairs, the exchanges in flight, the room rule and shortest-first order, the host RAM tier and the fills during decode written for this tree, and the operations and measurement around them.

## Support

**None.** This is provided **as-is**, without warranty of any kind (see [LICENSE](LICENSE)). It runs on one homelab; your hardware, firmware, fabric and driver versions will differ. It drives GPUs and hosts hard, and a DGX Spark that runs out of memory can need a power cycle. You use it at your own risk. Engine questions belong upstream at [ashhart/TensorFold](https://github.com/ashhart/TensorFold), transport questions at [ashhart/MCDMA](https://github.com/ashhart/MCDMA), and questions about Mia's recipe at [hers](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). Issues and PRs here may not get a response.

## License and credits

This version's scripts, patches and docs are **Apache-2.0** ([LICENSE](LICENSE)), the licence of TensorFold from v0.6.0 and of Mia's recipe, whose work it carries. [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) stays **MIT**, as published. This version vendors no third-party code beyond the patches and ships no weights or binaries. Everything it builds or downloads keeps its own licence, listed in **[NOTICE.md](NOTICE.md)**:

- **engine:** TensorFold, Apache-2.0 at v0.6.5 and v0.6.6;
- **wire:** MCDMA, Apache-2.0;
- **MiaAI-Lab's pull requests and recipe patches:** Apache-2.0, with code she adapted from Jay Leaton's glm53-tensorfold-spark (Apache-2.0) in two of them, and her recipe's contributed patches (E-Zou Shen's, desy0305's, m-naoki-m's and johnwhited's) under her recipe's licence;
- **Hugh Madden's expert prompt kernels in patches 0034 and 0035:** glm53f-afd's code, **MIT** (Copyright (c) 2026 Turquoise Bay AI Pty Ltd), with the MIT code of TensorFold's contributors it carries; both licence texts ship in the patch;
- **designs written anew in patches 0026, 0027 and 0032:** Hugh Madden's glm53f-afd and mimo26f-afd (MIT) and T.J. Purtell's ds41rt (MIT); no code from them is copied;
- **base model:** Z.AI's GLM-5.3-Flash, MIT;
- **checkpoint:** Mia's AI Lab's EXL3 quantization, Apache-2.0;
- **DFlash2 drafter:** CC BY-NC-ND 4.0, **non-commercial**, referenced only.

All the hard parts are other people's work:

- **[TensorFold](https://github.com/ashhart/TensorFold)** and **[MCDMA](https://github.com/ashhart/MCDMA)** by Ash Hart ([@ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart)). Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp), [@petruspennanen](https://x.com/petruspennanen)) wrote MCDMA's setup notes and build fix, [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5). Philip Mossop ([@philip-pentatonic](https://github.com/philip-pentatonic)) wrote TensorFold v0.6.6's `--name-priority`, [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445) (patch 0046).
- **Mia's AI Lab** ([@MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)): the checkpoint; the [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) whose patches are ported here and against which this version is measured; her TensorFold pull requests #243, #285 and #301. This README uses the layout of the README of her [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf).
- **The contributors to her recipe whose patches are ported here:** E-Zou Shen ([@ezoushen](https://github.com/ezoushen), [@ezoushen](https://x.com/ezoushen)) wrote her patches 0071 and 0074 (here 0036 and 0038, tests in 0041; her recipe's pull requests #44 and #62); desy0305 ([@desy0305](https://github.com/desy0305)) wrote her patch 0073 and its checks (here 0037; #51); johnwhited ([@johnwhited](https://github.com/johnwhited)) wrote her patches 0081, 0082 and 0083 and 0073's delivery-failure handling (here 0043-0045 and 0037; #48), with vLLM v1's admission and abort behaviour as his reference; m-naoki-m ([@m-naoki-m](https://github.com/m-naoki-m), [@\_m\_naoki\_m\_](https://x.com/_m_naoki_m_)) wrote her patch 0078 (here 0042; #71, also TensorFold#421). meleesciony ([@meleesciony](https://github.com/meleesciony)) diagnosed the failure of her [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75) and proposed the fix her patch 0077 makes (here 0040 and 0041); Lukas-tek-no-logic ([@Lukas-tek-no-logic](https://github.com/Lukas-tek-no-logic)) reported the `<|assistant|>` leak her patch 0075 fixes ([issue #60](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/60); here 0039).
- **Jay Leaton** ([@jayleaton](https://x.com/jayleaton), [jayleaton](https://github.com/jayleaton)) wrote [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark); two of Mia's ported patches (here 0017 and 0020) carry code she adapted from its patches 0580 and 0620, at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4).
- **Hugh Madden / Turquoise Bay AI** ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)) wrote [glm53f-afd](https://github.com/hughmadden/glm53f-afd) (`91db3cc`), the engine of v1.0, which serves GLM-5.3-Flash with attention, the KV cache, drafting and sampling on an RTX 5090 and the routed experts on DGX Sparks, and [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd), its serving shell, wire and RDMA transport. This version runs that split on TensorFold. Patches 0034 and 0035 carry his code: glm53f-rank's expert prompt kernels (the plan, the large-M gate/up and down kernels, the fused epilogue and reduce of `crates/glm53f-rank/kernels/exl3_rank.cu`), which the Sparks run for every prompt chunk; their EXL3 decoder, fragment MMA and Hadamard butterfly are TensorFold's MIT code (`bb4b4a3`) as glm53f-afd carries it. Patches 0026, 0027 and 0032 write four of his designs for this tree, with no code copied: the exchanges kept in flight ahead of the expert ranks (glm53f-afd `91db3cc`, `crates/glm53f-serve/src/lib.rs`; on here), the host RAM tier for kept prompts (`crates/glm53f-coordinator/src/hostcache.rs` and `scheduler.rs`; on), the prefill in two and four lanes (mimo26f-afd [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79), glm53f-afd `91db3cc`) and FP8 wire rows (both off).
- **T.J. Purtell** ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell)) wrote [ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and [cuteafd](https://github.com/tpurtell/cuteafd), engines that run attention on RTX GPUs and the routed experts on DGX Sparks. This version uses no code from them. Hugh Madden's expert prompt kernels (patch 0034) split every expert by intermediate channel as glmrt does, which T.J. Purtell designed; glm53f-afd's host RAM tier, whose design patch 0032 follows, takes its host cache's eviction design from ds41rt, through mimo26f-afd. Patch 0026's FP8 wire rows (off) use ds41rt's row format, E4M3 with a UE8M0 scale per 32 values, which reached glm53f-afd through mimo26f-afd.
- **fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** (MIT) implements the chunkwise algorithm of gated delta networks and Kimi Delta Attention that Mia's chunked KDA kernel computes in its chunked WY / UT form; her kernel is her own code.
- **Patryk Mikołajczyk** ([@mikolaj92](https://github.com/mikolaj92)) wrote [TensorFold#140](https://github.com/ashhart/TensorFold/pull/140), which bounds GLM's DSA selection to the pools a row can see; Mia's decode patch 0043 (here in 0017) extends that bound to decode. **Aditya Thyagarajan** ([@aditya1503](https://github.com/aditya1503)) wrote [TensorFold#200](https://github.com/ashhart/TensorFold/pull/200), MTP concurrency, which #243 superseded.
- **Weights:** [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) (the base model), Mia's AI Lab (the EXL3 quantization, made with turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3)), [IncoAI](https://huggingface.co/incoai) (the DFlash2 drafter).
- **[AGENTS.md](AGENTS.md):** the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), unchanged. Agents that edit this repo follow them.
- **v1.0's credits** (glm53f-afd and its upstreams, Local Inference Lab's weights, and everyone else v1.0 names) stay in [NOTICE.md](NOTICE.md#previous-version-v10), verbatim.
- **This recipe:** [@squarrier](https://github.com/squarrier), with the AI agents above.
