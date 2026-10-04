<p align="center">
  <b>GLM-5.3-Flash on 1× RTX 5090 + 2× DGX Spark</b><br>
  <sub>Attention/FFN-disaggregated serving over ConnectX-7 RoCE, using <a href="https://github.com/hughmadden/glm53f-afd">glm53f-afd</a></sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/RTX_5090-coordinator-76b900?style=flat-square" alt="RTX 5090 coordinator">
  <img src="https://img.shields.io/badge/2%C3%97_DGX_Spark-expert_ranks-2ea44f?style=flat-square" alt="2× DGX Spark expert ranks">
  <img src="https://img.shields.io/badge/ConnectX--7-RoCE_v2_RDMA-0969da?style=flat-square" alt="ConnectX-7 RoCE v2">
  <img src="https://img.shields.io/badge/API-OpenAI_compatible-6f42c1?style=flat-square" alt="OpenAI-compatible API">
  <img src="https://img.shields.io/badge/license-MIT-555?style=flat-square" alt="MIT">
</p>

This repo is a deployment recipe, not a new engine. It runs Z.ai's [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`eb9eb208`) with **[glm53f-afd](https://github.com/hughmadden/glm53f-afd)** v1.1.0 (`91db3cc`), by Hugh Madden / Turquoise Bay AI ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)), on a mixed box:

- An **x86 host with one RTX 5090** is the coordinator. It runs attention, holds all KV and KDA state, and runs the DFlash2 drafter, sampling and the OpenAI API.
- **Two DGX Sparks (GB10)** hold the routed experts as four TP4 ranks (EXL3 4-bpw), two per Spark under CUDA MPS.
- A ConnectX-7 in the 5090 box connects to the Sparks' ConnectX-7s over **RoCE v2 RDMA**.

> **Weights attribution (required by licence):** routed-expert weights are *GLM-5.3-Flash TR3 4bpw* by **Local Inference Lab, Inc.** (published by Brandon M. Music, revision `5ab363a8`). Upstream: <https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw>, project home: <https://local-inference-lab.ai/>, licensed LicenseRef-LIL-Attribution-1.0 / ShapleyMCG 1.0. The DFlash2 drafter is [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) by [IncoAI](https://huggingface.co/incoai) (revision `bf582e4e`), CC BY-NC-ND 4.0 (non-commercial). This repo contains no weights; see [NOTICE.md](NOTICE.md).

> **README layout:** this README uses the layout of the README of [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf). Her two-Spark lane is the comparison baseline in [Performance](#performance). No code from her repo is used.

> **Provided as-is, with no support.** This is a personal homelab recipe, shared in case it helps someone. Issues and PRs may go unanswered. See [Support](#support).

Upstream's reference is 1 RTX + **4** Sparks. This recipe runs the same four ranks on **2** Sparks with no change to the engine's compute path. The quality gate (KL against BF16) comes out identical to upstream's 4-Spark numbers.

| I want to... | Go to |
|---|---|
| See how fast it is | [Performance](#performance) |
| Know what hardware I need | [Hardware](#hardware) |
| Lock it down | [Security](#security) |
| Run it | [Quick start](#quick-start) |
| Keep it alive and collect telemetry | [Operations](#operations) |
| Understand why it is built this way | [docs/DESIGN.md](docs/DESIGN.md) |
| Stop a DGX Spark from hard-hanging | [extras/gb10-hostguard](extras/gb10-hostguard/README.md) |

## At a glance

| | |
|---|---|
| **Engine** | [hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd) v1.1.0 `91db3cc` (MIT), plus one small patch: `--served-model-name`, API layer only |
| **Model id** | `glm-5.3-flash` on `http://<rtx-host>:8000/v1` (configurable) |
| **Weights** | Official FP8 non-expert tensors ([zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) `eb9eb208`, MIT, ~15.5 GB, on the 5090); EXL3 K4 experts by Local Inference Lab ([brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) `5ab363a8`, LIL Attribution 1.0, 4 × 38.4 GB rank images); [incoai DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) drafter `bf582e4e` (CC BY-NC-ND 4.0) |
| **Context** | 262,144 tokens per request by default. The engine does 1M; the KV pool on a 32 GB 5090 holds ~1.58M tokens. |
| **Concurrency** | 16 slots |
| **Features** | Tool calling (`tool_choice: auto`), reasoning (thinking on by default), streaming, prefix/host cache, optional API key. Text only. |

## Performance

All numbers are from our kit, on 2026-09-30, measured on the served endpoint. Hardware: RTX 5090 at a 400 W cap in a PCIe Gen5 x8 slot with a ConnectX-7 MCX755106AS-HEAT at 200 GbE, plus two DGX Sparks, all through a 200G switch.

| Metric | **This recipe** (5090 + 2 Sparks) | Same 2 Sparks, [Mia EXL3 TP2](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | Upstream 5090 + 4 Sparks (published) |
|---|---:|---:|---:|
| Single stream, prose / code / JSON (tok/s)¹ | **73.1 / 83.7 / 76.2** | 30.5 / 35.1 / 34.6 | — |
| 4 concurrent, aggregate (tok/s)¹ | **117.6** | 57.8 | — |
| Engine bench C1 code / prose / counting (tok/s)² | 80.7 / 61.2 / 124.1 | — | 133.2 / 78.8 / 195.1 |
| Engine bench C1 without drafter² | 45.1 | — | 52.7 |
| Engine bench C4 / C16 aggregate² | 103.2 / 184.1 | — | 159.5 / ~290 |
| Prefill (tok/s, 5K–95K prompt) | 3.1–3.3K | — | 5.0–5.5K |
| KL vs BF16 (4,096 / 8 rows) | 0.0268 / 0.0240, **pass** | — | identical |
| Free memory per Spark while serving | 27–35 GB | 4–6 GB | — |

¹ The same `spark-bench`-style harness for both our columns (1,024-token prose/code/JSON prompts; C4 = 4 streams). The Mia column is a different engine and quant (ablit EXL3) on the same two Sparks, so treat it as a rough comparison, not a controlled A/B.
² Upstream's own bench (`harness/`): greedy, 1,024 tokens, median of 3. The upstream column is copied from its README.

That works out to **~2.2–2.4× the two-Spark vLLM lane single-stream and ~2× at 4 streams**, and about 60–65% of upstream's four-Spark numbers. Details, soak and drills: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Hardware

| Box | What it needs |
|---|---|
| **Coordinator** | x86_64 Linux, **RTX 5090 32 GB** (sm_120), ≥ 64 GB RAM, Docker + NVIDIA Container Toolkit, a ConnectX-7 (or other RoCE v2 NIC) with an address on **both** fabric subnets |
| **2 × DGX Spark** | GB10 (sm_121), stock DGX OS, Docker, both CX-7 ports cabled. About 77 GB of rank images each; ≥ 100 GB free memory before start. |
| **Fabric** | RoCE v2, MTU 9000, two IPv4 subnets (one per rail). We run the 5090's single 200G port with both subnets into a switch that also carries both Sparks' ports. A direct hub-and-spoke layout without a switch should also work but is untested here. |

The RDMA path matters: at M1 each decode step makes 42 round trips to the ranks. See [docs/DESIGN.md](docs/DESIGN.md#why-the-fabric-matters).

## Quick start

On a Linux box with passwordless SSH (and `sudo -n systemctl`) to all three hosts:

```bash
git clone https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks.git
cd GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks
cp .env.example .env        # hosts, fabric IPs, GLM_HOME, SSH user
./build.sh                  # builds glm53f-afd (pinned) + patches natively on each host, in CPU-only containers
./download.sh               # FP8 non-expert + drafter on the 5090 host; EXL3 + slice + verify the 4 rank dirs
./scripts/install.sh        # systemd units, wrappers, /etc/glm53f-afd/*.env, runs preflight
./start.sh                  # ranks -> 4 listening -> coordinator -> /health -> one real completion
```

```bash
curl http://<rtx-host>:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Hello"}]}'
```

Then `./start.sh status`, `./start.sh recover`, and `./stop.sh`. Starting takes about 60–75 s from cold (ranks ~45 s, coordinator ~15 s). Stopping takes about 7 s.

> [!WARNING]
> **One GPU tenant per Spark.** On GB10, CUDA allocations are unified memory and are *not* charged to a container's cgroup. A second model on a Spark that is serving ranks can starve the box until only a power cycle recovers it. Preflight refuses to start if another GPU process is running, and [gb10-hostguard](extras/gb10-hostguard/README.md) enforces the rule continuously.

## Using the API

- **Thinking** is on by default. `"reasoning_effort": "none"` (or `"minimal"`) turns it off. `"low"`/`"high"` map to the GLM template's Low and High; anything else maps to Max.
- **Tools**: `tool_choice: "auto"` works. `"required"` and named-function forcing return HTTP 400 (upstream limitation).
- **Health**: after any rank or wire failure, `/health` latches `503` with a reason until the coordinator restarts. `./start.sh recover` handles that. `/health` can stay `200` for about 2 min after a rank dies, so the watcher also probes a real completion.

## Operations

| Piece | What |
|---|---|
| `./start.sh {up,restart,status,recover,probe}` / `./stop.sh` | A single controller that orders the lane. The units are not enabled at boot. For boot, add `@reboot sleep 240 && /path/start.sh` to the controller's crontab. |
| `site/systemd/*.service` | One container per Spark (both ranks plus the MPS daemon), and one coordinator container. `Restart=no`, because the controller decides. |
| `site/glm53f-afd-preflight` | Runs before every unit starts. Checks the memory floor, single GPU tenant, fabric IPs, stale MPS, rank dirs, optional sha256 and power-cap pins. |
| [`extras/watch/`](extras/watch/README.md) | Keep-alive plus telemetry. Every 2 min it checks health and a real completion; 2 failures run `recover`. A circuit breaker latches after 3 recoveries in 6 h. It logs per-host memory, PSI, GPU power and temperature, MPS, drafter acceptance, slot pressure, TTFT, and an hourly decode benchmark to JSONL, with a roll-up report. |
| [`extras/gb10-hostguard/`](extras/gb10-hostguard/README.md) | An on-host memory floor and single-GPU-tenant guard, plus hardening: sshd/journald OOM protection, a hardware watchdog, panic on lockup, and a persistent journal. |

Measured recovery on our kit, through the watcher: coordinator fault 21 s, a Spark rank fault 99 s.

## Layout

| Path | What |
|---|---|
| `build.sh`, `download.sh`, `start.sh`, `stop.sh`, `scripts/` | Build, weights, install, and control |
| `site/` | Units, in-container wrappers, and preflight, as installed on the hosts |
| `patches/` | Local patches on top of upstream (0001: `--served-model-name`) |
| `docker/Dockerfile.build` | The CPU-only build image (CUDA devel, Rust 1.98.1, rdma-core) |
| `docs/` | Design, benchmarks, troubleshooting |
| `extras/` | Watcher/telemetry and the GB10 host guard |
| `AGENTS.md` | Credit and attribution rules for agents that edit this repo, by Mia, unchanged |

## Security

Read this before exposing anything.

- **The API** listens on `API_LISTEN` (default `0.0.0.0:8000`). With `API_KEY_FILE` empty there is **no authentication**. Set `API_KEY_FILE` (a root-owned `0600` file under `/etc/glm53f-afd/secrets/`, chowned to `RUN_UID`) to require `Authorization: Bearer <key>` on `/v1/*`; `/health` stays open. Export `API_KEY` for `./start.sh probe` and the watcher. Either way, keep the port on a trusted LAN or VPN, behind a firewall. Never port-forward it to the internet.
- **The expert ranks** (`RANK_PORT` 8600, `PEER_PORT` 8601) and the RDMA wire are **unauthenticated by design**. They bind only to the fabric IPs, so put the fabric on its own isolated subnet or VLAN, with no route to your LAN or the internet.
- **Privileges:** containers run as `RUN_UID` (not root), with `--network host`, `--ipc host`, `IPC_LOCK` and `/dev/infiniband`, which RDMA and MPS require. Install uses `sudo -n` over SSH. Grant the SSH user sudo **only** for what `scripts/install.sh` and `start.sh` run (`systemctl start/stop/daemon-reload` of the two units, `install` into `/opt/glm53f-afd` and `/etc/glm53f-afd`), not blanket root.
- **Staging:** files are staged in a private `0700` dir under `GLM_HOME` before root installs them, never in `/tmp`.
- **Secrets:** `.env` is git-ignored. Don't commit it, and don't put keys in `EXTRA_ARGS`, because the unit logs its arguments.
- **`extras/gb10-hostguard`** enables kernel panic-on-lockup and the hardware watchdog, so a hung box reboots itself. That is intended, but read its README first.

## AI-assisted development

Most of this recipe was planned, written, run and measured by AI agents, with a human (me) setting direction and approving anything that touched the live machines. The scripts, benchmarks and docs here came out of that process, and every number was measured, not generated. Still, read the scripts before running them as root.

| Agent | Model | How it was used |
|---|---|---|
| Orchestrator | **Claude Opus 5.5** (Anthropic), in the [Hermes](https://hermes-agent.nousresearch.com) agent harness | Researched the approach, and planned the work as a Kanban board of cards (feasibility, bring-up, fallback route, soak, v1.1.0 re-baseline, go-live). Ran the builds, benchmarks, KL gate and failure drills over SSH. Wrote the systemd units, controller, preflight, watcher, host guard, patch 0001 and these docs. |
| Card workers | Claude Opus 5.5 sub-agents in the same harness | Executed individual cards in parallel, for example the 4-hour soak with drills and the v1.1.0 validation, then reported evidence back to the orchestrator. |
| Local daily-driver model | **GLM-5.3-Flash** (zai-org), self-hosted on the two Sparks via the vLLM lane and later via this recipe | Ran the local agent turns and evals (AEON, tool-call and agent-loop checks), and was the model under test. It now serves the lane described here. |

The engine and nearly all the hard ideas belong to the people credited below. The agents' contribution is the integration: fitting four TP4 ranks onto two Sparks with MPS, plus the safety, operations and measurement around it.

## Support

**None.** This is provided **as-is**, without warranty of any kind (see [LICENSE](LICENSE)). It runs on one homelab; your hardware, firmware, fabric and driver versions will differ. It can drive GPUs and hosts hard, including enabling a hardware watchdog and kernel panic-on-lockup through `extras/gb10-hostguard`. You use it at your own risk. Questions about the engine belong upstream at [hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd). Issues and PRs here may not get a response.

## License and credits

This repo's scripts, units and docs are **MIT** ([LICENSE](LICENSE)). MIT matches the engine it wraps (glm53f-afd, MIT) and most of that engine's upstreams, and it keeps the recipe easy to reuse. The repo vendors no third-party code and ships no weights. Everything it downloads or builds keeps its own licence, listed in **[NOTICE.md](NOTICE.md)**:

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
