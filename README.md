<p align="center">
  <b>GLM-5.3-Flash on 1× RTX 5090 + 2× DGX Spark</b><br>
  <sub>An attention/expert split on <a href="https://github.com/ashhart/TensorFold">TensorFold</a>: attention, KV cache and drafter on the 5090, the routed experts on two Sparks, one RDMA exchange per MoE layer over <a href="https://github.com/ashhart/MCDMA">MCDMA</a></sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RTX_5090-attention_node-76b900?style=flat-square" alt="RTX 5090 attention node">
  <img src="https://img.shields.io/badge/2%C3%97_DGX_Spark-expert_nodes-2ea44f?style=flat-square" alt="2× DGX Spark expert nodes">
  <img src="https://img.shields.io/badge/MCDMA-RoCE_v2_RDMA-0969da?style=flat-square" alt="MCDMA over RoCE v2">
  <img src="https://img.shields.io/badge/TensorFold-v0.6.5_%2B_30_patches-6f42c1?style=flat-square" alt="TensorFold v0.6.5 + 30 patches">
  <img src="https://img.shields.io/badge/license-Apache--2.0-555?style=flat-square" alt="Apache-2.0">
</p>

This repo is a deployment recipe, not a new engine. Version 2.0 serves Z.ai's [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) as MiaAI-Lab's EXL3 checkpoint with **[TensorFold](https://github.com/ashhart/TensorFold)** v0.6.5 by Ash Hart ([@ashhart](https://github.com/ashhart)), plus 30 patches, on three machines:

- An **x86 host with one RTX 5090** is the attention node. It runs attention, holds the KV cache, runs the DFlash2 drafter and sampling, and serves the OpenAI-compatible API.
- **Two DGX Sparks (GB10)** each hold one half of every MoE layer's routed experts.
- In every MoE layer the 5090 sends its rows to both Sparks and gets two partial sums back over **RoCE v2 RDMA**, through Ash Hart's **[MCDMA](https://github.com/ashhart/MCDMA)** link daemons, with two exchanges in flight to each Spark.

> **Where v1.0 is better: prompt processing, context, four-stream decode and one-stream code.** [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0), this repo's glm53f-afd version of 2026-09-30, prefills 3.1-3.3K tok/s on 5K-95K prompts and keeps a ~1.58M-token KV pool; this version prefills 2,599-2,762 tok/s on fresh 8K-62K prompts and keeps 727,040 tokens. At four streams it decodes 111.6 tok/s against v1.0's 117.6; at one stream 77.4 / 79.0 / 90.4 (prose / code / JSON) against 73.1 / 83.7 / 76.2. v1.0 also serves more requests at once and was tested longer: [What changed since v1.0](#what-changed-since-v10).

It serves MiaAI-Lab's checkpoint the way her own [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) does (4-bit dense weights, FP8 KV cache, DFlash2, her decode kernels, her EXL3 prompt kernel, chunked KDA prompts) and carries her GLM work: her three open TensorFold pull requests as her commits, and ports of her recipe's patches, each credited. Where it differs from her main lane, the difference was measured: eight streams instead of four, her draft policy at `fnc5:0.2` instead of `fnc7:0.3`, and this recipe's own levers for the split (prompt chunk pairs, two MoE exchanges in flight, the pool's room rule, shortest-first prompt order). Her attention-side prompt kernels are on, as in her lane: they change the replies, and AEON-30 scored 23 of 30 with them (the bar was 21). See [What the patches are](#what-the-patches-are).

## Credits

| Author | Profiles | What they authored, as used here | Repo, commit |
|---|---|---|---|
| **Mia** (Mia's AI Lab) | [MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab) | the EXL3 4-bpw checkpoint; her TensorFold pull requests #243, #285 and #301 (patches 0002-0007, her commits); the 15 patches here that port her recipe's patches (among them her EXL3 prompt kernel, 0004; her attention-side prompt kernels, 0004, 0009 and 0028; eight streams, 0069), and the code of hers that patches 0025 and 0029 change (her 0030 and 0049); [AGENTS.md](AGENTS.md); this README's layout | [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at [`cf28cc4`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1) (v1.4) and [`1576746`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) (v1.5); [the checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) at [`76c0b517`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/76c0b5173166d2795dd48860f45d8224817f894c) |
| **Ash Hart** | [ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart) | TensorFold, the engine; MCDMA, the RDMA wire | [TensorFold](https://github.com/ashhart/TensorFold) at [`609ca41`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f) (v0.6.5); [MCDMA](https://github.com/ashhart/MCDMA) at [`e672c14`](https://github.com/ashhart/MCDMA/tree/e672c14ff9fc7b38994caf73025cf1588b4de74e) |
| **Hugh Madden** / Turquoise Bay AI | [hughmadden](https://github.com/hughmadden), [@dangerm00se](https://x.com/dangerm00se) | glm53f-afd, the engine of v1.0: GLM-5.3-Flash with attention on an RTX 5090 and the routed experts on DGX Sparks, the split this version runs on TensorFold; the designs patches 0026 and 0027 write for it: the exchanges kept in flight ahead of the expert ranks (on here), the two- and four-lane prefill and the FP8 wire rows (off) | [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at [`91db3cc`](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) (v1.1.0); [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) at [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79) |
| **T.J. Purtell** | [tpurtell](https://github.com/tpurtell), [@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars) | ds41rt, glmrt and cuteafd: engines that run attention on RTX GPUs and the routed experts on DGX Sparks; DS41RT's wire-row format, which patch 0026's FP8 wire rows use (off) | [ds41rt](https://github.com/tpurtell/ds41rt) at `3067d06`, [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) at `dc6d9b8`, [cuteafd](https://github.com/tpurtell/cuteafd) at `c650feb` |
| **Jay Leaton** | [jayleaton](https://github.com/jayleaton), [@jayleaton](https://x.com/jayleaton) | code Mia adapted from his patches 0580 (the experts' decode loads, in patch 0017) and 0620 (tool calls, in patch 0020) | [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4) |
| **Z.ai** | [zai-org](https://huggingface.co/zai-org), [@Zai_org](https://x.com/Zai_org) | GLM-5.3-Flash, the model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at [`eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) |
| **turboderp** | [turboderp](https://github.com/turboderp), [@turboderp_](https://x.com/turboderp_) | ExLlamaV3, which made Mia's checkpoint in its EXL3 format | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| **IncoAI** | [incoai](https://huggingface.co/incoai) | the DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at [`bf582e4e`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) |

Also credited in [License and credits](#license-and-credits) and [NOTICE.md](NOTICE.md): Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp)) for [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5), Patryk Mikołajczyk ([@mikolaj92](https://github.com/mikolaj92)) and Aditya Thyagarajan ([@aditya1503](https://github.com/aditya1503)) for TensorFold #140 and #200, fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and every credit of v1.0, unchanged.

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
| See what changed since v1.0 | [What changed since v1.0](#what-changed-since-v10) |
| Run the glm53f-afd version | [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) |

## What changed since v1.0

[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (2026-09-30) ran Hugh Madden's glm53f-afd with Local Inference Lab's TR3 4-bpw experts. Version 2.0 keeps the three machines and the split, and runs the split on TensorFold v0.6.5 with Mia's GLM work and her checkpoint. Against v1.0's published numbers:

- **v1.0 is still better at:**
  - prompt processing: 3.1-3.3K tok/s on 5K-95K prompts; 2.0 does 2,599-2,762 tok/s on fresh 8K-62K prompts
    (0.79-0.89x);
  - context held at once: a ~1.58M-token KV pool; 2.0 keeps 727,040 tokens (0.46x);
  - requests at once: 16 slots; 2.0 decodes eight together and queues the rest;
  - decode at four streams, 117.6 tok/s against 2.0's 111.6 (0.95x), and code at one stream, 83.7 against 79.0
    (0.94x);
  - AEON-30: 25 of 30, run under its soak load; 2.0 scored 23 of 30, idle;
  - testing: a 4-hour soak with failure drills (on glm53f-afd v1.0.0), and `extras/watch`, which runs `recover`
    after two failed checks; 2.0 ran 81.6 minutes of mixed load without drills and does not ship the watcher.
- **2.0 is better at** one-stream decode on prose and JSON (77.4 and 90.4 tok/s against 73.1 and 76.2: 1.06x and
  1.19x). It runs Mia's checkpoint (Apache-2.0) and her serving work: her GLM tool-call fixes, the shared-prefix
  prompt cache and smooth streaming.
- v1.0's numbers come from its own README and BENCHMARKS (decode on its spark-bench-style harness, 1,024-token
  prompts) and were not re-measured here. If long prompts, many long contexts or requests at once, or a
  longer-tested build matter most to you, run [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0).

## At a glance

| | |
|---|---|
| **Engine** | [TensorFold](https://github.com/ashhart/TensorFold) v0.6.5 (`609ca41`, Apache-2.0), built from source, plus the 30 patches in [`patches/`](patches/) |
| **Wire** | [MCDMA](https://github.com/ashhart/MCDMA) `e672c14` (Apache-2.0): link daemons and `libmcdma-rpc`, built from source on each host; GPU-driven where the device supports CUDA stream memory operations; two links to each Spark, so two MoE exchanges are in flight to it |
| **Model id** | `GLM-5.3-Flash-EXL3` (alias `glm-5.3-flash`) on `http://<attention-host>:8000/v1` (configurable) |
| **Weights** | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) `76c0b517` (Apache-2.0; its safetensors are byte-identical to `078455ff`, the revision measured); [incoai DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter `bf582e4e` (CC BY-NC-ND 4.0) |
| **Context** | 262,144 tokens a request; one KV pool of 727,040 tokens for all requests |
| **Concurrency** | 8 requests decoding together, one shared cache pool |
| **Features** | Tool calling (with Mia's GLM tool-call fixes), reasoning (thinking on by default), smooth streaming, the prompt cache with shared-prefix states on a 64-token grid, prompt chunk pairs, shortest-first prompt order. Text only: no vision. |

## Performance

Measured on one RTX 5090 plus two DGX Sparks, against v1.0's published numbers, Mia's latest recipe (v1.5) on its own
two Sparks and stock TensorFold 0.6.5 on the same two Sparks, with the same checkpoint and drafter (v1.0: its own
checkpoint):

| | This version | v1.0 (glm53f-afd)¹ | Mia's recipe v1.5 | Stock 0.6.5 TP2 |
|---|---:|---:|---:|---:|
| Decode, one stream: prose / code / JSON (tok/s) | 77.4 / 79.0 / 90.4² | 73.1 / 83.7 / 76.2 | 58.5 / 65.7 / 66.6 | 32.5 / 38.0 / 35.7 |
| Decode, four streams (aggregate tok/s) | 111.6² | **117.6** | 101.5 | 35.2 |
| Prompt processing, ~2.6K-token prompt (tok/s) | **1,779**² | — | 1,569 | 507 |
| Prompt processing, fresh 8K / 31K / 62K prompts (tok/s) | 2,599 / 2,618 / 2,762² | **3.1-3.3K** (5K-95K) | — | — |
| Cold 100K-token prompt, time to first token | 36.8 s² | — | — | — |
| KV pool (tokens, all requests) | 727,040 | **~1.58M** | — | — |
| Context a request | 262,144 | 262,144 | 262,144 | 131,072 |

¹ v1.0's own README (2026-09-30): glm53f-afd with Local Inference Lab's TR3 4-bpw experts, measured with its
spark-bench-style harness on 1,024-token prompts; not re-measured here. ² As shipped (`.env.example`), on one boot:
spark-bench's decode script, median of three runs; each fresh or cold prompt after a warm-up of the same size.

- **This update's levers**, each measured against the configuration without it ([docs/DESIGN.md](docs/DESIGN.md#the-switches)):
  - Mia's EXL3 prompt kernel on the Sparks (patch 0024): a cold 100K-token prompt's first token 76.0 -> 52.7 s, fresh
    prompts 45-49% faster;
  - two MoE exchanges in flight to each Spark (patch 0027): fresh prompts 23-26% faster, a cold 100K prompt's first
    token 52.1 -> 41.6 s, replies bit-identical;
  - twenty kept prompts and the pool's room rule (patch 0025): the arena's 100,000 x 5 and 65,535 x 10 cells resumed
    every request, first token 83.1 -> 7.4 s and 151.7 -> 15.8 s, ahead of Mia's two Sparks (8.1 s and 16.9 s);
  - eight streams (patch 0030): at ten clients on 32K and 64K contexts the first token 0.89x the time, prompt rates
    1.12-1.34x and generation 1.05-1.08x in the cells measured, each ahead of Mia's recipe v1.4;
  - shortest-first prompt order (patch 0029): from two clients the median first token 28.3 -> 14.6 s; from five,
    short prompts wait longer while long ones come sooner;
  - her draft policy at `fnc5:0.2`: decode +7.0% (spark-bench geometric mean);
  - her attention-side prompt kernels (patch 0028): a cold 100K prompt 11.5% sooner and fresh prompts 13% faster, with
    a reference KL just over the gate; on, because AEON-30 held 23 of 30 with them (the bar was 21).
- **Prompt processing is still behind v1.0** on long prompts (2,599-2,762 against 3.1-3.3K tok/s); on the ~2.6K prompt
  it is 1.13x Mia's two Sparks (1,779 against 1,569 tok/s).
- Replies are deterministic, drafted replies equal undrafted ones, and each concurrent reply equals its solo reply.
  The 4-bit weights, the FP8 cache, the chunked KDA kernel and the EXL3 prompt kernel are byte-identical to her
  recipe's; the decode kernels are byte-identical to the kernels they replace.
- **As shipped, on the same boot:** AEON-30 23 of 30 at 83.1 tok/s (Mia's recipe v1.5: 23 of 30 at 66.3); three
  Spark Arena cells at five and ten clients on 32K-100K contexts, each ahead of Mia's recipe v1.4 in prompt rate
  (1.29-1.54x), generation (1.16-1.19x) and first token (0.79x the time), though at ten clients each request decodes
  slower (0.66-0.69x); and 81.6 minutes of mixed load at up to eight requests: 1,169 requests, no error, no restart.

Not run for this release: the full 28-cell arena grid on the shipped configuration, and failure drills. Method, every
table and the caveats: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Hardware

| Box | What it needs |
|---|---|
| **Attention node** | x86_64 Linux, **RTX 5090 32 GB** (sm_120), Docker + NVIDIA Container Toolkit, a ConnectX-7 (or another RoCE v2 NIC) on the fabric |
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

> [!WARNING]
> **One GPU tenant per Spark.** On GB10, CUDA allocations are unified memory and are *not* charged to a container's
> cgroup. A second GPU process on a Spark that serves experts can starve it until only a power cycle recovers it.
> `./start.sh` refuses to start beside another GPU process and waits for memory to come back between runs. Read
> [the troubleshooting page](docs/TROUBLESHOOTING.md#dgx-spark-gb10-memory-safety-read-this-first) first, and consider
> [gb10-hostguard](extras/gb10-hostguard/README.md) on each Spark.

## Using the API

- **Thinking** is on by default. Send `"chat_template_kwargs": {"enable_thinking": false}` (or `"reasoning_effort": "none"`) to turn it off.
- **Tools**: GLM tool calls are parsed by TensorFold's server, with MiaAI-Lab's fix for a call the end token left open. With `TF_GLM_TOOL_CALLS=1` (on in `.env.example`) her fixes for agent clients are on too: each call is streamed whole in its place, a call that ends the think block counts as the reply's call, agent histories with empty or odd arguments render, and a tool-calling reply's reasoning is put back for clients that drop it (`TF_GLM_KEEP_REASONING`). With `tool_choice: "none"` the model can still write call markup as text ([Limits](docs/DESIGN.md#limits)).
- **Health**: `/health` on the API port. If an expert node or a link fails, in-flight requests fail with HTTP 503 and the attention log names the cause; restart the whole stack (`./stop.sh && ./start.sh up`).

## Operations

| Command | What |
|---|---|
| `./start.sh up` | check, no-model prebuild, MCDMA daemons, attention node, expert nodes, health wait, startup lines, one smoke reply; any failure takes the stack down again |
| `./start.sh check` / `status` / `logs` / `smoke` / `probe` | read-only preflight / containers, health and links / log tails / one real reply / the GPU-driven exchange probe (no model) |
| `./stop.sh` / `./stop.sh tf` | containers, then the connect daemon, then the listen daemons (never SIGKILL) / containers only |
| `./build.sh [tree\|sync\|images\|mcdma\|ext]` | one build step at a time |
| `tools/export_patches.sh HEAD` | regenerate `patches/` from a TensorFold branch, with every check (maintainers) |
| [`extras/gb10-hostguard/`](extras/gb10-hostguard/README.md) | an optional on-host memory floor and single-GPU-tenant guard for each Spark, with host hardening (from v1.0) |

The containers start with `--restart=no`: the scripts decide. Nothing starts at boot.

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

Details, switches and defaults: [docs/DESIGN.md](docs/DESIGN.md#the-patches). Each port keeps `Co-authored-by: MiaAI-Lab` and names her recipe commit in its credit header, and so do the two changes of this recipe's to her ported code (0025, 0029). Two of the ports carry code she adapted from Jay Leaton's glm53-tensorfold-spark (see [NOTICE.md](NOTICE.md)).

## Layout

| Path | What |
|---|---|
| `build.sh`, `download.sh`, `start.sh`, `stop.sh`, `scripts/` | build, weights and control; `scripts/lib.sh` reads `.env` |
| `patches/` | the 30 patches on TensorFold v0.6.5, applied in order with `git am` |
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
- **Remote commands are built from `.env`** and run over SSH: treat `.env` as code. The extra `TF_GLM_*` switches (`ATTN_ENV`, `EXPERT_ENV`) are checked against a strict pattern before they reach `docker run`.
- **Secrets:** none are needed; `.env` is git-ignored anyway. Don't put keys in `.env` values that end up in container arguments, which `docker inspect` shows.
- **`extras/gb10-hostguard`**, if you install it, enables kernel panic-on-lockup and the hardware watchdog, so a hung box reboots itself. Read its README first.

## AI-assisted development

Most of this recipe was planned, written, run and measured by AI agents, with a human (me) setting direction and approving anything that touched the machines or went public. Every number in it was measured, not generated. Still, read the scripts before running them.

| Agent | Model | How it was used |
|---|---|---|
| Planner and executor | **Claude Opus 5.5** (Anthropic), in the [Hermes](https://hermes-agent.nousresearch.com) agent harness | Planned the work as a Kanban board of cards, and did much of the execution: the TensorFold 0.6.5 rebase, the ports of Mia's patches, prompt chunk pairs, the exchanges in flight, the pool's room rule, shortest-first prompt order, the hardware checks, these scripts and docs. |
| Earlier executor | **GLM-5.3-Flash** (zai-org), self-hosted, via Hermes | Earlier execution work on the split. |

The engine, the transport, the checkpoint and most of the hard ideas belong to the people credited below. The agents' part is the integration: the split over MCDMA, the ports onto it, prompt chunk pairs, the exchanges in flight, the room rule and shortest-first order, and the operations and measurement around them.

## Support

**None.** This is provided **as-is**, without warranty of any kind (see [LICENSE](LICENSE)). It runs on one homelab; your hardware, firmware, fabric and driver versions will differ. It drives GPUs and hosts hard, and a DGX Spark that runs out of memory can need a power cycle. You use it at your own risk. Engine questions belong upstream at [ashhart/TensorFold](https://github.com/ashhart/TensorFold), transport questions at [ashhart/MCDMA](https://github.com/ashhart/MCDMA), and questions about Mia's recipe at [hers](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). Issues and PRs here may not get a response.

## License and credits

This version's scripts, patches and docs are **Apache-2.0** ([LICENSE](LICENSE)), the licence of TensorFold from v0.6.0 and of Mia's recipe, whose work it carries. [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) stays **MIT**, as published. This version vendors no third-party code beyond the patches and ships no weights or binaries. Everything it builds or downloads keeps its own licence, listed in **[NOTICE.md](NOTICE.md)**:

- **engine:** TensorFold, Apache-2.0 at v0.6.5;
- **wire:** MCDMA, Apache-2.0;
- **MiaAI-Lab's pull requests and recipe patches:** Apache-2.0, with code she adapted from Jay Leaton's glm53-tensorfold-spark (Apache-2.0) in two of them;
- **designs written anew in patches 0026 and 0027:** Hugh Madden's glm53f-afd and mimo26f-afd (MIT) and the row format of T.J. Purtell's ds41rt (MIT); no code from them is copied;
- **base model:** Z.AI's GLM-5.3-Flash, MIT;
- **checkpoint:** Mia's AI Lab's EXL3 quantization, Apache-2.0;
- **DFlash2 drafter:** CC BY-NC-ND 4.0, **non-commercial**, referenced only.

All the hard parts are other people's work:

- **[TensorFold](https://github.com/ashhart/TensorFold)** and **[MCDMA](https://github.com/ashhart/MCDMA)** by Ash Hart ([@ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart)). Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp), [@petruspennanen](https://x.com/petruspennanen)) wrote MCDMA's setup notes and build fix, [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5).
- **Mia's AI Lab** ([@MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)): the checkpoint; the [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) whose patches are ported here and against which this version is measured; her TensorFold pull requests #243, #285 and #301. This README uses the layout of the README of her [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf).
- **Jay Leaton** ([@jayleaton](https://x.com/jayleaton), [jayleaton](https://github.com/jayleaton)) wrote [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark); two of Mia's ported patches (here 0017 and 0020) carry code she adapted from its patches 0580 and 0620, at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4).
- **Hugh Madden / Turquoise Bay AI** ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)) wrote [glm53f-afd](https://github.com/hughmadden/glm53f-afd) (`91db3cc`), the engine of v1.0, which serves GLM-5.3-Flash with attention, the KV cache, drafting and sampling on an RTX 5090 and the routed experts on DGX Sparks, and [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd), its serving shell, wire and RDMA transport. This version runs that split on TensorFold; it uses no code from them. Patches 0026 and 0027 write three of his designs for this tree: the exchanges kept in flight ahead of the expert ranks (glm53f-afd `91db3cc`, `crates/glm53f-serve/src/lib.rs`; on here), the prefill in two and four lanes (mimo26f-afd [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79), glm53f-afd `91db3cc`) and FP8 wire rows (both off).
- **T.J. Purtell** ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell)) wrote [ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and [cuteafd](https://github.com/tpurtell/cuteafd), engines that run attention on RTX GPUs and the routed experts on DGX Sparks. This version uses no code from them. Patch 0026's FP8 wire rows (off) use ds41rt's row format, E4M3 with a UE8M0 scale per 32 values, which reached glm53f-afd through mimo26f-afd.
- **fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** (MIT) implements the chunkwise algorithm of gated delta networks and Kimi Delta Attention that Mia's chunked KDA kernel computes in its chunked WY / UT form; her kernel is her own code.
- **Patryk Mikołajczyk** ([@mikolaj92](https://github.com/mikolaj92)) wrote [TensorFold#140](https://github.com/ashhart/TensorFold/pull/140), which bounds GLM's DSA selection to the pools a row can see; Mia's decode patch 0043 (here in 0017) extends that bound to decode. **Aditya Thyagarajan** ([@aditya1503](https://github.com/aditya1503)) wrote [TensorFold#200](https://github.com/ashhart/TensorFold/pull/200), MTP concurrency, which #243 superseded.
- **Weights:** [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) (the base model), Mia's AI Lab (the EXL3 quantization, made with turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3)), [IncoAI](https://huggingface.co/incoai) (the DFlash2 drafter).
- **[AGENTS.md](AGENTS.md):** the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), unchanged. Agents that edit this repo follow them.
- **v1.0's credits** (glm53f-afd and its upstreams, Local Inference Lab's weights, and everyone else v1.0 names) stay in [NOTICE.md](NOTICE.md#previous-version-v10), verbatim.
- **This recipe:** [@squarrier](https://github.com/squarrier), with the AI agents above.
