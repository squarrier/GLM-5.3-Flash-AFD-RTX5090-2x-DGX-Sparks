# V4.2 deployment benchmark methodology

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
