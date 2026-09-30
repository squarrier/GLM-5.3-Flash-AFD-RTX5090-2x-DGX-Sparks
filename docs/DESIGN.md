# Design

## The split (attention/FFN disaggregation)

GLM-5.3-Flash is a large MoE. Per token, the dense parts (attention, KDA linear attention, the DSA indexer, shared experts, router, LM head) are small, but they are latency-bound and hold all the KV state. The routed experts are most of the weights, and each token needs only a few of them.

glm53f-afd splits the model along that line:

| Where | What runs | Memory |
|---|---|---|
| **RTX 5090** (coordinator) | Everything except routed experts: attention, KDA and all KV/KDA state, the DFlash2 drafter, sampling, scheduler, OpenAI API | ~15.5 GB FP8 weights + drafter + a 9 GiB KV page pool (~1.58M tokens) + a host-RAM prefix cache |
| **Spark ranks** | Routed experts only. Each rank holds a TP4 quarter (512 of 2,048 intermediate channels) of **every** routed expert, as EXL3 K4. | 38.4 GB per rank |

For each of the 42 MoE layers, the coordinator sends the hidden rows and routing to all four ranks over RDMA. Each rank computes its partial expert output, and the coordinator sums the four returns. The ranks are stateless between steps, so all context lives on the 5090.

## Four ranks on two Sparks (MPS)

Upstream's layout is one rank per Spark on four Sparks. Rather than rewrite the ~120 places that assume a world size of 4, this recipe runs **two unmodified rank processes per Spark**, one per fabric rail. Measured alternatives:

| Option | Decode C1 without drafter | Notes |
|---|---:|---|
| 2 ranks per Spark, time-sliced (no MPS) | ~26 tok/s | The GPU context-switches between ranks. Per-layer M1 rose from 0.145 ms to about 0.43 ms. |
| **2 ranks per Spark, CUDA MPS 100%** | **42 → 45 tok/s** (v1.0 → v1.1) | Both ranks run concurrently on the SMs. This is what the recipe ships. |
| 1 rank per Spark, native world size 2 (engine patches) | ~same speed; decode KL slightly worse | Needed 3–4 engine patches (including an RDMA buffer-reuse fix). Kept as a fallback, not shipped. |

MPS is part of the rank container (`site/glm53f-afd-ranks-run`). **Shut it down gracefully** (`echo quit | nvidia-cuda-mps-control`). In our failure drills, SIGKILLing the MPS server on GB10 stranded ~6 GiB of unified memory until the Spark was rebooted. The wrapper and `./stop.sh` never hard-kill MPS.

## Why the fabric matters

At a single stream, every decode step makes 42 layer round trips. On our kit a round trip is about 40 µs one-way RDMA plus rank compute, and the rank side, not the coordinator, dominates the step. That is why:

- the 5090 talks to the ranks over **RoCE v2 RDMA** (`GLM53F_RDMA=1` on the coordinator), not TCP;
- both rails are used: each Spark's two ranks listen on different subnets or ports;
- the rank↔rank mesh stays on TCP (upstream default; its RDMA mesh is untested upstream).

Our coordinator NIC is a ConnectX-7 MCX755106AS-HEAT (200GbE, QSFP112) in a PCIe Gen5 **x8** slot. That is still ~250 Gb/s per direction, so the slot is not the bottleneck. Its single port carries both subnets into a 200G switch, and each Spark's two CX-7 ports connect to the same switch.

## Why these defaults

- **262,144 context per request.** The engine does 1M. We cap it lower, which leaves more of the KV pool for concurrency. `MAX_CONTEXT` changes it.
- **16 slots.** On this pool, 32 slots did not improve C8/C16 throughput in our tests.
- **v1.1.0 `--decode-share` default (0.2).** During a ~240K-token prefill, other streams keep about 20–24% of their decode speed instead of 2–4%. The cost is about 20% longer TTFT on the long prompt.
- **`--kda-fp8-pow2` is off.** It gives about 5% more single-stream speed and still passes KL, but changes some greedy outputs. It is not worth it at a 262K cap.
- **`Restart=no` plus a controller.** The coordinator connects to the ranks once at start and latches `/health` 503 after any wire error, so the correct recovery is ordered: restart the failed Spark tenant, then the coordinator. Blind per-unit restarts would come back half-connected. `./start.sh recover` encodes the order, and `extras/watch` calls it.

## Safety on GB10

On a DGX Spark, CUDA memory is unified with system RAM and is not charged to a container's cgroup, so `docker --memory` cannot stop a model from starving the host. A starved GB10 livelocks: ping answers, SSH never completes, and only a power cycle helps. We hit this once while benchmarking two lab models on the same Spark. The countermeasures here:

- preflight refuses to start with another GPU process present, or with less than 100 GiB free on a Spark;
- [gb10-hostguard](../extras/gb10-hostguard/README.md) keeps a memory floor, enforces one GPU tenant, protects sshd and journald, and arms the hardware watchdog;
- the watcher never reboots anything. If a Spark stays below the floor after a graceful stop, it latches and asks a human.

## How this was built

The work ran over three days (2026-09-28 to 30) as a Kanban-driven program in the [Hermes](https://hermes-agent.nousresearch.com) agent harness. See the README section [AI-assisted development](../README.md#ai-assisted-development) for who did what. In order:

1. Offline feasibility: build and CPU tests on both architectures, fetch and slice the weights, and a perf model.
2. Route A live bring-up with KL, then the Route B fallback.
3. A 4-hour mixed-load soak with failure drills and AEON.
4. The v1.1.0 re-baseline.
5. Go-live with a watcher and a rehearsed rollback.

Every number in this repo comes from those runs.
