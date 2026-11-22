# V4.2 / TensorFold v1.3.3 — speed, concurrency and context

The tested system uses RTX 5090 attention, two DGX Spark expert roles, EXL3 4-bpw experts, 16-bit dense/head, Q4 external DFlash2, and pinned BF16 host history. The public deliverable is a source overlay, not a fresh native build. See the README for external drafter restrictions. No fixture, response body, raw receipt or quality score is redistributed.

## Same-harness short-run refresh

Both arms use the same short harness and 300 generated tokens per stream. The v1.3.3 sequence was C4 → C4 repeat → C8, each preceded by C1 prose/code/JSON probes. The v1.3.0 baseline is from the earlier retained V4.2 measurements, not a public V4.1 measurement. This is a later-run comparison, not randomized or alternating A/B. All 16 new concurrent streams completed. The retained short-run records have rounded emitter rates and clocks, not individual raw timing traces: the C1 medians are recomputed from emitter observations. Aggregate token/wall checks use rounded wall times; they do not recover hidden precision.

| Engine | Cell | Aggregate tok/s | Post-first-token stream range, tok/s | Mean TTFT |
|---|---|---:|---:|---:|
| v1.3.3 | C4 | 95.0 | 24.3–28.9 | 446 ms |
| v1.3.3 | C4 repeat | 94.8 | 24.3–28.8 | 459 ms |
| v1.3.3 | C8 | 122.2 | 15.7–18.6 | 839 ms |
| v1.3.0 (earlier) | C4 | 92.9 | 23.8–28.1 | 450 ms |
| v1.3.0 (earlier) | C4 repeat | 93.1 | 23.5–28.2 | 450 ms |
| v1.3.0 (earlier) | C8 | 118.4 | 15.2–18.0 | 836 ms |

C4 headline uses two-wave medians (94.9 vs 93.0); C8 uses one wave (122.2 vs 118.4). C1 decode medians across three probes are prose 65.5 vs 63.9, code 70.9 vs 69.4, JSON 81.9 vs 80.6 tok/s. Do not divide aggregate rate by offered concurrency and call it stream decode speed. None of these short requests demonstrates eight full-context histories. No new Mia comparator run was collected for v1.3.3; earlier comparator rows below are historical only.

## Hardware and power disclosure

The short and exact-fit runs use the three-role hardware above. RTX 5090 power limit/clock cap and both Spark power/clock settings are not established by these measurement receipts. They are unknown, not inferred from current settings or stock specifications. No power-efficiency or matched-power assertion is made. Historical V4/v1.2.8 and V4.1/v1.2.9 rows retain their separately recorded 400 W 5090 limit in the README; that value is not transferred to v1.3.3.

## Exact cold capacity

One stream: 419,488 prompt tokens plus 512 output tokens, zero reported cache tokens, exact retrieval and full completion. TTFT **314.266 s**, E2E **319.209 s**. Cap-plus-one was refused before inference. Startup attached 6,568 BF16 pages / 41,871 MiB host RAM and 2,463 MiB pooled GPU keys, with eight slots configured. Eight occupied 420K windows, full-occupancy cancellation/refill, net VRAM savings and causal host-KV on/off timing remain unqualified. The 1,812 MiB lowest sampled startup free VRAM is a startup sample only, not the later matrix minimum or an all-phase bound.

## Extended context matrix — partial

Separate harness: temperature 0, 300 completion tokens per stream, prose/code/JSON at C1/C4/C8 and three prompt tiers. Usage counts can include reasoning tokens; these are not guaranteed visible final-answer throughput or correctness results. Random per-request prefixes and API-reported zero cached tokens do not independently establish backend cache behavior; cache state remains unknown. Short prompts are ~70–99 tokens; mid prompts ~13.6K; long prose ~82.5K, not the nominal 420K cap. This was one wave per cell, not a replicated median-speed study.

**20/27 planned cells qualified.** Short/mid are complete. Long prose C1/C4 qualified, long prose C8 had 1/8 completions and is unqualified after a controller-local disk guard. Long code/JSON cells were not run. No GPU OOM is established. No new Mia matrix comparison is available. Aggregate rates include long prefill and admission latency; post-first-token decode does not.

| Tier / workload | Offered | Prompt tokens / stream | Aggregate E2E tok/s | Stream decode range, tok/s | Median / p95 TTFT |
|---|---:|---:|---:|---:|---:|
| short / Prose | 1 | 72–72 | 65.12 | 68.52–68.52 | 0.243 / 0.243 s |
| short / Prose | 4 | 69–75 | 102.75 | 27.27–28.99 | 0.586 / 0.903 s |
| short / Prose | 8 | 70–77 | 127.43 | 16.72–18.85 | 1.072 / 1.818 s |
| short / Code | 1 | 82–82 | 67.47 | 71.49–71.49 | 0.264 / 0.264 s |
| short / Code | 4 | 80–86 | 104.66 | 27.32–31.46 | 0.652 / 0.998 s |
| short / Code | 8 | 77–86 | 129.93 | 16.66–19.07 | 1.158 / 1.984 s |
| short / JSON | 1 | 91–91 | 82.29 | 88.63–88.63 | 0.272 / 0.272 s |
| short / JSON | 4 | 92–99 | 109.47 | 28.72–36.90 | 0.688 / 1.064 s |
| short / JSON | 8 | 91–97 | 143.07 | 19.09–25.11 | 1.248 / 2.107 s |
| mid / Prose | 1 | 13653–13653 | 18.83 | 63.78–63.78 | 11.244 / 11.244 s |
| mid / Prose | 4 | 13634–13659 | 20.79 | 6.79–27.92 | 32.133 / 45.717 s |
| mid / Prose | 8 | 13588–13660 | 21.38 | 3.19–18.20 | 80.733 / 94.810 s |
| mid / Code | 1 | 13635–13635 | 19.94 | 77.17–77.17 | 11.173 / 11.173 s |
| mid / Code | 4 | 13643–13659 | 20.70 | 6.65–29.71 | 32.976 / 45.697 s |
| mid / Code | 8 | 13633–13664 | 21.36 | 3.30–21.28 | 80.408 / 95.175 s |
| mid / JSON | 1 | 13633–13633 | 19.09 | 66.47–66.47 | 11.215 / 11.215 s |
| mid / JSON | 4 | 13669–13686 | 20.42 | 7.15–30.31 | 32.926 / 45.624 s |
| mid / JSON | 8 | 13640–13685 | 21.47 | 3.35–18.47 | 79.600 / 93.630 s |
| long / Prose | 1 | 82533–82533 | 3.67 | 62.96–62.96 | 77.035 / 77.035 s |
| long / Prose | 4 | 82475–82531 | 3.61 | 14.94–61.83 | 236.227 / 318.987 s |

The matrix receipts report a sampled 1,532 MiB minimum free 5090 VRAM during the qualifying waves. Sampled native running counts are not a complete residency trace; one short C1 response finished between polls. No eight-full-history or independent state-equivalence claim follows from eight offered requests.

## Historical V4.2 / v1.3.0 methodology and comparison

**The remaining sections were previously published for the v1.3.0 deployment. They are not new v1.3.3 results, a new comparison to Mia's latest, or proof of hardware/precision parity.**


These measurements are from the live, repaired V4.2 deployment (TensorFold v1.3.0 base plus the V4.2 AFD overlay with the allocation repair, RTX 5090 attention and two DGX Spark experts, EXL3 4-bpw experts, 16-bit dense/head, Q4 DFlash2 and pinned BF16 history). The comparator is the actual same-window Mia-based deployment at [recipe commit `d8a7439`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/commit/d8a7439e8c5aac8645c423ccd24189690359ab69) in the measurement record, not an assertion that Mia's current upstream was deployed. This is an end-to-end API comparison of two working configurations; checkpoint-variant, topology, drafter, clock and precision parity are not implied. An exact matched V4.1 run is not part of this table. No native rebuild of the public overlay is inferred from these receipts.

## Short-workload checks

The same client and four/eight offered-request harness ran in this order: V4.2 C4 → Mia C4 → V4.2 C4 repeat → V4.2 C8 → Mia C8 offered. The concurrent cells use 300 generated tokens/stream. There were 28/28 successful concurrent streams, and the separate native request totals, including warmups, matched the intended counts. Aggregate speed divides generated tokens by elapsed workload time, not by the sum of per-stream times. Individual stream speed measures generation after the first token. TTFT is client-observed time to first token. Do not divide aggregate throughput by concurrency and call the result per-stream decode speed.

| Short concurrent cell | Streams completed | Aggregate tok/s | Per-stream decode range, tok/s | Mean TTFT |
|---|---:|---:|---:|---:|
| V4.2 C4 | 4 | 92.9 | 23.8–28.1 | 450 ms |
| Mia C4 | 4 | 77.0 | 19.6–30.0 | 458 ms |
| V4.2 C4 repeat | 4 | 93.1 | 23.5–28.2 | 450 ms |
| V4.2 C8 | 8 | 118.4 | 15.2–18.0 | 836 ms |
| Mia C8 offered | 8 | 85.1 | 16.0–34.8 | 7,357 ms |

Mia's C8 row has only four native lanes; the last requests queue. Its higher maximum per-stream speed is not inconsistent with lower aggregate throughput and longer mean TTFT. The C1 probes run before each concurrent group. They show the spread of observations, not five independent randomized trials or confidence bounds:

| C1 probe | V4.2 decode tok/s | Mia decode tok/s | V4.2 TTFT | Mia TTFT |
|---|---:|---:|---:|---:|
| Prose | 63.7–63.9 | 39.3–39.4 | 152–153 ms | 151–153 ms |
| Code | 69.3–69.4 | 66.0–66.1 | 183 ms | 184–202 ms |
| JSON | 80.5–80.7 | 71.9–72.1 | 212–215 ms | 212–213 ms |

## Coding-agent speed workload

1,024 generated tokens per stream, same workload and client, temperature 0.2. Prompt token counts differed by tokenizer/serving template: 138 for V4.2 and 132 for Mia. All 16 streams completed; there is no Mia C8 coding cell. These are throughput/latency measurements, not a correctness ranking.

| Cell | Aggregate tok/s | Per-stream decode range, tok/s | Mean TTFT | Mean E2E |
|---|---:|---:|---:|---:|
| V4.2 C4 | 111.63 | 28.15–29.49 | 0.849 s | 36.207 s |
| Mia C4 | 96.32 | 24.26–26.47 | 0.885 s | 41.514 s |
| V4.2 C8 | 132.06 | 17.05–20.20 | 1.535 s | 58.605 s |

C4 coding throughput was 15.9% higher and mean E2E was 12.8% lower on V4.2 in this measurement window. Per-request TTFT penalties at C8 cannot be interpreted as a four-to-eight isolated engine speedup. Power and clock settings were not established from these run receipts, so no stock-power, efficiency or matched-power claim is made.

## Exact-fit and exclusions

Single cold stream: 419,488 input tokens plus 512 output tokens, no cache hit, exact retrieval and full stream termination; TTFT 313.911 s, E2E 318.984 s. Cap-plus-one was refused before inference. Startup attached 6,568 BF16 host pages and 2,463 MiB of device keys; 1,814 MiB was the minimum across the recorded startup free-VRAM samples. This is neither a full-occupancy result nor a claim that no unsampled lower point occurred. Eight concurrently offered short workloads do not prove eight simultaneously occupied 420K histories; occupied cancellation and exact refill were not measured on this deployment.

Only aggregate physical context, timing and throughput are included. Private fixtures, prompts, response bodies, evaluation artifacts and internal deployment configuration are excluded.
