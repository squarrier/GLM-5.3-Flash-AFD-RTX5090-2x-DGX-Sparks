# Benchmarks

**Kit:**
- **Coordinator:** RTX 5090 32 GB at a 400 W power cap, PCIe Gen5 x8, ConnectX-7 MCX755106AS-HEAT at 200 GbE.
- **Ranks:** two DGX Spark (GB10), DGX OS, both CX-7 ports into a 200G switch, MTU 9000, RoCE v2.
- **Software:** glm53f-afd v1.1.0 `91db3cc` plus patch 0001, MPS 100, `MAX_CONTEXT=262144`, 16 slots, drafter on.
- **Date:** 2026-09-29/30.

## Canonical harness (1,024-token prose/code/JSON, then C4)

| Run | prose | code | JSON | C4 aggregate |
|---|---:|---:|---:|---:|
| v1.0.0 (first live run) | 67.4 | 78.1 | 72.0 | 111.6 |
| v1.0.0 after 4 h soak + drills | 67.8 | 78.4 | 72.5 | 112.3 |
| **v1.1.0** | **73.0** | **83.0** | **76.1** | **117.2** |
| v1.1.0 on the served port, run 1 | 73.1 | 83.7 | 76.2 | 117.6 |
| v1.1.0 on the served port, run 2 | 72.8 | 83.1 | 75.9 | 95.8 |
| Mia EXL3 TP2, same 2 Sparks (reference) | 30.5 | 35.1 | 34.6 | 57.8 |

The C4 figure varies about ±6% between repeats. Run 2's 95.8 is an outlier we have not explained yet. A coding-agent-style C4 (1,024 tokens, T 0.2) measured a median of 118.7 tok/s, with TTFT about 1.0 s.

## Upstream engine bench (greedy, 1,024 tokens, median of 3)

| Metric | v1.0.0 | **v1.1.0** | Upstream 5090 + 4 Sparks |
|---|---:|---:|---:|
| C1 code / prose / counting (drafter) | 77.4 / 57.2 / 120.5 | **80.7 / 61.2 / 124.1** | 133.2 / 78.8 / 195.1 |
| C1 without drafter | 42.2 | **45.1** | 52.7 |
| C4 aggregate | 99.7 | **103.2** | 159.5 |
| C16 aggregate | 187.8 | **184.1** | 288.5–294.5 |
| Prefill 5K / 23K / 95K | 3,048 / 3,152 / 3,260 | 3,097 / 3,192 / 3,261 | ~5,000–5,500 |

## Quality

- **KL gate** (upstream `glm53f-score`, BF16 teacher, 25 windows):
  - 4,096 rows: mean 0.02678 (+1.96·SE 0.02933), top-1 0.948, **pass**;
  - 8 rows: 0.02398, **pass**.
  - Both are identical to upstream's 4-Spark numbers. v1.1.0 logits are byte-identical to v1.0.0 across all 25 files.
- **AEON 30-case eval:** 25/30, run *under* the soak load. Two earlier two-Spark vLLM GLM runs scored 22 and 25.
- **Fresh needles:** a 165K-token needle was found in 50.9 s. A ~240K needle was found while four coding streams ran.

## 4-hour soak (v1.0.0)

Mixed load: coding-agent streams at C1/C4/C8, forced-arg tool calls, exact-string checks, a fresh 64K–200K prompt every 30 minutes, and a 16-request burst every hour.

- 2,279 requests with **0 errors**; tool calls 367/367 and exact strings 364/364 correct.
- Decode drift over 4 hours was −0.05%. No memory creep on any host, no swap, no guard events.

## Long prompt vs other streams (v1.1.0 `--decode-share`)

| Setting | Needle TTFT (~240K) | Other streams keep |
|---|---:|---:|
| `--decode-share 0` (v1.0 behaviour) | 75–78 s | 2–5% of their speed |
| **0.2 (default)** | 92–93 s | **19–24%** |
| 0.4 | 125 s | 41% |

## Memory while serving (16 streams + a 240K prompt)

| Host | Free (MemAvailable) | Swap |
|---|---:|---:|
| Spark 1 (r0, r1) | 34.6 GB | 0 |
| Spark 2 (r2, r3) | 27.1 GB | 0 |
| 5090 host | 184 GB (KV lives in 5090 VRAM; host RAM holds the prefix cache) | 0 |

## Failure drills (through `extras/watch` and `./start.sh recover`)

| Fault | Seen as | Recovery | Time to healthy |
|---|---|---|---:|
| Coordinator container stopped | connection refused | coordinator restart | 21 s |
| One Spark's rank tenant stopped | `/health` 503 "expert wire: rdma recv completion status 5" | restart that Spark, then the coordinator | 99 s |
| MPS server SIGKILLed on a Spark | ranks die | Spark restart; ~6 GiB stranded until reboot | needs reboot |
| Cold start | — | `./start.sh` | ~60–75 s |
