# Benchmarks

Everything below was measured on the hardware this recipe targets. Each table says which configuration it measured.
**As shipped** is `.env.example` ([the configuration](#what-is-measured-and-how)); its own numbers come from one boot
of it on 2026-10-05 ([the release configuration](#the-release-configuration)). [The levers](#the-levers-one-at-a-time)
compare each lever this update adds with the same configuration without it, and
[v2.0's first staging](#v20s-first-staging) is the configuration before them, kept for the record.

## What is measured, and how

- **Hardware.** One RTX 5090 (32 GB, 400 W limit, clocks not capped) as the attention node in an x86_64 host; two
  DGX Sparks (GB10, 128 GB each) as the expert nodes; ConnectX-7 RoCE v2 between them, 200 Gb/s links (4X HDR) at
  MTU 9000.
  Mia's recipe ran on its own two DGX Sparks (TP2), with the SM clock cap of 2,200 MHz her recipe sets.
- **Software.** TensorFold v0.6.5 plus `patches/` at this repository's commit; the checkpoint at revision `078455ff`
  (its safetensors are byte-identical to `76c0b517`, the revision `.env.example` pins) and the drafter at `bf582e4e`.
- **Configurations.**
  - *as shipped*: eight streams with a 64-row verify window and a 6.5 GiB cache budget, 20 kept prompts, two
    exchanges in flight to each Spark, her EXL3 prompt kernel on the Sparks, the pool's room rule, shortest-first prompt
    order, her draft policy at `fnc5:0.2`, prompt chunk pairs, her attention-side prompt kernels, and every other
    switch of `.env.example`;
  - *the levers*: each table adds one lever to the configuration before it, at four streams, and names its baseline;
  - *v2.0 staging*: v2.0's first staging: prompt chunk pairs on, Mia's draft policy off, 32 kept prompts at 8 GiB,
    four streams, one link to each Spark;
  - *pairs on, policy on* and *pairs off, policy on*: two boots of one session, for the prompt-processing set, the
    cold 100K prompt and the decode check of pairs;
  - *pairs off, policy on*: the first full arena grid and the spark-bench decode run, before patch 0022.

  Prompt pairs change only the order in which prompt chunks are computed, and the draft policy only how many rows a
  decode round verifies; neither changes a reply ([Quality and correctness](#quality-and-correctness)).
- **Decode.** spark-bench's OpenAI-API benchmark script (`vllm-bench.py`): three single-stream prompts (C1: prose,
  code, JSON; 300-token replies) and four concurrent streams (C4); median of three runs; thinking at the server
  default.
- **The Spark Arena grid.** llama-benchy 0.4.0 with the Spark Arena v2 settings: 28 cells across context depth and
  concurrency, a 2,048-token prompt after the context, 128 generated tokens, prefix caching on, three runs, thinking
  off.
- **Coding-agent load.** A headless coding-agent benchmark at temperature 0.2: four concurrent requests generating
  1,024 tokens each, and one request generating 4,096; sustained aggregate tok/s, median of three runs.
- **Mixed load.** Two or five clients sending prompts of 2,048, 8,192, 32,768 and 100,000 tokens (24 and 40 requests);
  each request's time to the first token, by prompt size.
- **Agentic quality.** AEON-30: 30 agentic tasks, one at a time, thinking on.
- **Prompt processing.** Fresh 8K, 31K and 62K prompts (nothing cached), and a cold 100K-token prompt's time to
  first token. Where a table says so, each prompt size was warmed up once first.
- **Stability.** Mixed load on the shipped configuration at up to eight requests for 1.58 hours, with probe rounds and
  a decode check at the start, middle and end ([below](#stability)). Failure drills were not run for this release.

Every run of a comparison went through the same runner, with the endpoint otherwise idle; a run that saw any other
client's request was discarded and repeated.

## The release configuration

One boot of `.env.example` on 2026-10-05: spark-bench's decode script (median of three runs), each fresh or cold
prompt after a warm-up of the same size, a needle alone, the coding-agent load (median of three runs), three Spark
Arena cells and AEON-30.

| Metric | As shipped | Before patches 0028-0030: four streams, no shortest-first order | Mia's recipe v1.5, 2x DGX Spark |
| --- | ---: | ---: | ---: |
| Decode, C1 prose / code / JSON (tok/s) | 77.4 / 79.0 / 90.4 | 76.3 / 84.7 / 86.6 | 58.5 / 65.7 / 66.6 |
| Decode, C4 aggregate (tok/s) | 111.6 | 112.8 | 101.5 |
| spark-bench's ~2.6K-token prompt (tok/s) | 1,779 | 1,714 | 1,569 |
| Cold ~2.6K, time to first token | 1.449 s | 1.487 s | — |
| Fresh 8K / 31K / 62K (tok/s) | 2,599 / 2,618 / 2,762 | 2,284 / 2,335 / 2,419 | — |
| Cold 100K, time to first token | 36.8 s | 41.6 s | — |
| A needle in ~195K tokens, alone | found: 194,886 tokens, prefill 74.3 s | — | — |
| Coding-agent load, 4 x 1,024 / 1 x 4,096 (tok/s) | 229.3 / 64.4 | 243.3 / — | — |
| KV pool (tokens, all requests) | 727,040 | 966,656 | 1,740,800 |
| Spark Arena | three cells: [below](#three-arena-cells-as-shipped) | four cells at eight streams: [below](#eight-streams-patch-0030) | [v1.4's grid](#spark-arena-grid-28-cells) |
| AEON-30 (idle) | 23 of 30, 83.1 tok/s | — | 23 of 30, 66.3 tok/s |

The middle column is the shipped configuration without shortest-first order and at four streams with an 8 GiB cache,
on the head before patches 0028-0030 (their switches unset); those patches change no reply with their switches unset.

- **Her attention-side prompt kernels change the replies**, and decode moves with the texts. Against the same
  configuration without them, measured the same morning, one-stream code decodes 6.2% slower (its reply takes 93
  verify rounds per 300 tokens instead of 84) and JSON 4.5% faster (72 instead of 82); prose takes 101 in both.
- **The coding-agent load is slower than before these levers:** 229.3 tok/s at 4 x 1,024 against 243.3 (−5.8%), and
  64.4 tok/s for one 4,096-token request against 74.6 on v2.0 staging's configuration with the same draft policy
  (−13.7%, [the draft policy](#the-draft-policy)). No boot in between ran the 4,096-token request, so this release
  cannot say which lever cost it.
- Mia's KV pool is 2.4 times this one: her two Sparks hold the cache, where this recipe's 5090 does.

### Three arena cells, as shipped

llama-benchy 0.4.0 with the Spark Arena v2 settings (a 2,048-token prompt after the context, 128 generated tokens,
prefix caching on, three runs, thinking off), on the same boot. Each entry is prompt / generation t/s and the e2e time
to the first token, then generation per request and the cache hits. Mia's recipe v1.4 ran on its own two Sparks.

| Cell | As shipped | Mia's recipe v1.4 | Prompt / gen / first token |
| --- | --- | --- | ---: |
| 32,768 x 10 | 848.5 / 45.5, 12.5 s; 11.5 a request; 48.4% | 657.6 / 39.3, 15.8 s; 16.7 a request; 48.4% | 1.29x / 1.16x / 0.79x |
| 65,535 x 10 | 833.9 / 46.1, 13.4 s; 12.0 a request; 49.2% | 624.6 / 38.7, 16.9 s; 18.2 a request; 49.2% | 1.33x / 1.19x / 0.79x |
| 100,000 x 5 | 1,047.6 / 50.3, 6.4 s; 18.6 a request; 49.5% | 678.5 / 42.6, 8.1 s; 19.4 a request; 49.5% | 1.54x / 1.18x / 0.79x |

- All three cells ran clean. 65,535 x 10 and 100,000 x 5 resumed every inference request (49.2% and 49.5% are the
  ideal), as they did at four streams with the room rule.
- At ten clients each request decodes slower than hers (0.69x and 0.66x): eight requests decode together here, four
  in her main lane. The aggregate rates and the first token lead.
- The full 28-cell grid was not run on the shipped configuration. Against v2.0 staging's cells: 65,535 x 10 went from
  59.7 / 3.5 t/s and 151.7 s to 833.9 / 46.1 t/s and 13.4 s, 100,000 x 5 from 236.0 / 16.1 and 83.1 s to 1,047.6 /
  50.3 and 6.4 s, and 32,768 x 10 from 578 / 35.5 and 18.8 s to 848.5 / 45.5 and 12.5 s.

## The levers, one at a time

### Her EXL3 prompt kernel (patch 0024)

Two boots of one session, without and with `TF_GLM_EXL3_PROMPT=1`: four streams, 20 kept prompts at 8 GiB, her draft
policy at `fnc5:0.2`, prompt pairs on, one link to each Spark.

| Metric | Without | With | Change |
| --- | ---: | ---: | ---: |
| Expert kernel, per half and MoE layer at 2,048 rows (ms) | 28.41 | **16.49** | x1.72 |
| Cold 100K, time to first token (s) | 76.01 | **52.74** | −30.6% |
| Cold ~2.6K, time to first token (s) | 2.230 | **1.656** | −25.7% |
| Fresh 8K / 31K / 62K (tok/s) | 1,244 / 1,296 / 1,320 | **1,850 / 1,879 / 1,909** | +48.7 / +45.0 / +44.6% |
| spark-bench decode, geometric mean | | | +1.2% |
| Coding-agent load, 4 x 1,024 | | | +2.1% |
| KL against the serial KDA kernel's reference: mean / p99 / top-1 | 0.002967 / 0.05025 / 98.82% | 0.002808 / 0.04700 / 99.14% | closer |

- The kernel's outputs equal Mia's own kernel's byte for byte on both GB10s (0 of 128 and 0 of 64 cases differ).
- Replies change with the prompt bits: 2 of 5 greedy replies and 20 of 33 concurrent cases equal those without it.
  Within a boot the replies are deterministic.
- A needle in 194,798 tokens was found, prefilled in 103.5 s (1,882 tok/s); before this update, 185.2 s for 194,888.
- Decode moves only through the new texts: each spark-bench cell follows its replies' verify rounds.

### The pool's room rule and twenty kept prompts (patch 0025, `CACHE_ENTRIES=20`)

The arena's two worst-miss cells. Each entry is inference requests that resumed a kept context and the cell's cache
hits, then prompt / generation t/s and the e2e time to the first token.

| Cell | v2.0 staging: 32 kept prompts | 20 kept prompts | 20 kept, the room rule, her prompt kernel | Mia's recipe v1.4 |
| --- | --- | --- | --- | --- |
| 100,000 x 5 | 9 of 15, 29.7%; 236.0 / 16.1, 83.1 s | 15 of 15, 49.5%; 614.3 / 39.9, 9.3 s | **15 of 15, 49.5%; 754.8 / 47.8, 7.4 s** | 678.5 / 42.6, 8.1 s |
| 65,535 x 10 | 12 of 30, 19.7%; 59.7 / 3.5, 151.7 s | not run | **30 of 30, 49.2%; 672.3 / 40.6, 15.8 s** | 624.6 / 38.7, 16.9 s |

- 49.5% and 49.2% are the ideal: every inference request resumed (plus 5 and 10 in llama-benchy's warm-up rounds).
- Twenty kept prompts: each entry reserves about 183 MiB of the cache budget, so 20 entries at 8 GiB leave a pool of
  966,656 tokens against 626,688 at 32. Fresh-boot numbers did not move (spark-bench +0.4%, coding −0.1%, cold
  prompts within 1%). The measured fallback, 10 GiB with 32 entries (946,176 tokens), also resumed 100,000 x 5 (621.3
  / 41.4 t/s, 9.3 s).
- The room rule moved 319 extents in 41 packs and evicted 74 kept prompts, none that a waiting request wanted. The
  kept count reached the 20-entry cap in 65,535 x 10's third run without costing a resume. Cold prompts are unchanged
  with it (100K −0.3%, 2.6K −1.0%).
- Twenty kept prompts alone had already resumed 100,000 x 5, and no boot ran the right-hand cells without the room
  rule, so that column shows the whole stack holding the ideal, not the room rule's own gain.

### Two exchanges in flight (patch 0027)

Two boots of one session, with one link pair and with two exchanges in flight (`MCDMA_INFLIGHT=2`); the idle times
come from timed boots of the same two settings.

| Metric | One link pair | Two in flight | Change |
| --- | ---: | ---: | ---: |
| Expert idle per exchange, cold 100K (ms) | 7.33 | **2.03** | |
| Expert busy, share of a long prompt's MoE time | 69% | **89%** | |
| Fresh 8K / 31K / 62K (tok/s) | 1,823 / 1,892 / 1,922 | **2,284 / 2,335 / 2,419** | +25.3 / +23.4 / +25.9% |
| Cold 100K, time to first token (s) | 52.08 | **41.64** | −20.0% |
| Cold ~2.6K, time to first token (s) | 1.637 | **1.487** | −9.2% |
| spark-bench's ~2.6K-token prompt (tok/s) | 1,545 | **1,714** | +10.9% |
| Decode, C1 prose / code / JSON (tok/s) | 77.3 / 86.1 / 88.0 | 76.3 / 84.7 / 86.6 | −1.3 / −1.6 / −1.6% |
| Decode, C4 aggregate (tok/s) | 113.4 | 112.8 | −0.5% |
| Coding-agent load, 4 x 1,024 (tok/s) | 245.4 | 243.3 | −0.9% |

- Replies are bit-identical: greedy 5 of 5, and 22 concurrent requests' 33 cases equal. The wire check found 0
  mismatches in 256 checked forwards (4 of them in prompt chunk pairs, 81 layer slices).
- Idle at a cold 2.6K and 31K prompt: 1.94 ms (3.22 and 6.68 ms with one link pair).
- Against Mia's recipe on its two Sparks: 1.09x her spark-bench prompt rate (1,714 against 1,569 tok/s).

### Her attention-side prompt kernels (patch 0028)

Three boots at four streams, each prompt size warmed up once first: without them; with `TF_GLM_SEQ_ROWS=256` and
`TF_GLM_PROMPT_DSA=1`, the two that keep the bits; and with all five.

| Metric | Without | The two same-bits switches | All five | All five vs without |
| --- | ---: | ---: | ---: | ---: |
| Cold ~2.6K, time to first token (s) | 1.526 | 1.483 | **1.404** | −8.0% |
| Fresh 8K / 31K / 62K (tok/s) | 2,337 / 2,375 / 2,451 | 2,434 / 2,368 / 2,436 | **2,644 / 2,696 / 2,770** | +13.1 / +13.5 / +13.0% |
| Cold 100K, time to first token (s) | 41.39 | 41.72 | **36.62** | −11.5% |
| Replies | | bit-identical (greedy 5 of 5, 33 of 33 cases) | changed (prompt arithmetic) | |
| KL against the serial KDA kernel's reference: mean / p99 / top-1 | 0.00281 / 0.0470 / 99.14% | | 0.00309 / 0.0493 / 98.87% | mean over the 0.003 gate |
| AEON-30 | | | 23 of 30 (as shipped) | on at 21 of 30 or more: on |

- Against the configuration without them, the five's KL is 0.00282 / 0.0474 / 99.30%.
- Decode moves with the replies: spark-bench one stream prose +1.7%, code −5.6%, JSON +4.7%, four streams −0.4%.
- On the RTX 5090 the kernels give her tree's digests on the same seeded inputs, 11 of 11.
- As shipped (eight streams, shortest-first order), the reference log-probs are byte-identical to this table's
  all-five boot's, so the KL is the same.

### Shortest-first prompt order (patch 0029)

Two boots at four streams, without and with `TF_GLM_PREFILL_ORDER=sjf`, under the mixed load. Each entry is the time
to the first token in seconds, mean / p90 (number of requests).

| Clients | Prompt size | Without | With |
| --- | --- | ---: | ---: |
| 2 | all (24): median 28.32 -> **14.59** | 28.28 / 49.57 | 24.37 / 57.10 |
| 2 | 2,048 (5) | 10.08 / 41.86 | **3.72 / 8.65** |
| 2 | 8,192 (8) | 21.48 / 47.58 | 10.77 / 38.67 |
| 2 | 32,768 (5) | 30.43 / 50.37 | 26.31 / 42.35 |
| 2 | 100,000 (6) | 50.70 / 78.52 | 58.12 / 77.92 |
| 5 | all (40): median 85.17 -> **77.09** | 96.25 / 182.17 | 95.04 / 178.28 |
| 5 | 2,048 (9) | 52.57 / 85.38 | 92.08 / 178.28 |
| 5 | 8,192 (12) | 84.42 / 104.91 | 66.91 / 99.58 |
| 5 | 32,768 (3) | 92.64 / 113.14 | 77.29 / 87.00 |
| 5 | 100,000 (16) | 130.37 / 222.23 | 121.13 / 184.61 |

- From two clients short prompts no longer wait out a long fill; the longest prompts wait a little longer.
- From five clients most waits are past the 30 s aging bound, beyond which prompts go by arrival, and the 2,048-token
  prompts waited longer while every longer size came sooner.
- A second boot without it measured 27.89 s and 91.41 s mean at two and five clients.
- Replies are bit-identical with it: greedy 5 of 5, and 22 concurrent requests' 33 cases equal.

### Eight streams (patch 0030)

Four streams at 8 GiB against eight streams with the 64-row window at 6.5 GiB. Each entry is prompt / generation t/s,
then the e2e time to the first token.

| Cell | Four streams | Eight streams | Prompt / gen / first token | Mia's recipe v1.4 |
| --- | --- | --- | ---: | --- |
| 32,768 x 5 | 782.8 / 45.9, 6.5 s | **1,051.9 / 49.1, 6.2 s** | 1.34x / 1.07x / 0.94x | 697.6 / 41.2, 7.2 s |
| 32,768 x 10 | 689.0 / 41.6, 15.0 s | **787.2 / 44.5, 13.2 s** | 1.14x / 1.07x / 0.88x | 657.6 / 39.3, 15.8 s |
| 65,535 x 5 | 793.1 / 45.6, 6.7 s | **1,016.2 / 47.8, 6.5 s** | 1.28x / 1.05x / 0.97x | 721.3 / 42.9, 7.6 s |
| 65,535 x 10 | 688.6 / 40.6, 15.3 s | **768.9 / 43.8, 13.8 s** | 1.12x / 1.08x / 0.90x | 624.6 / 38.7, 16.9 s |

- The ten-client cells' first token: 0.892x (geometric mean).
- spark-bench single stream unchanged (prose 1.001x, code 1.002x, JSON 0.994x). Every concurrent reply equals its
  solo reply with up to eight decoding together (22 of 22).
- The 5090's peak at eight streams, against its memory gate of 27,136 MiB: 27,600 MiB at 7.5 GiB, 27,370 MiB at 7 GiB
  with the 64-row window (both over), 27,114 MiB at 7 GiB with the 32-row window, and **26,926 MiB** at 6.5 GiB with
  the 64-row window, as shipped. Its pool is 727,040 tokens (966,656 at four streams and 8 GiB).

### The draft policy

`fnc5:0.2` against the policy unset, on two boots of v2.0 staging's configuration:

| Metric | Unset | `fnc5:0.2` | Change |
| --- | ---: | ---: | ---: |
| C1 prose / code / JSON (tok/s) | 71.7 / 82.2 / 76.0 | **77.2 / 84.3 / 80.9** | +7.7 / +2.6 / +6.4% |
| C2 aggregate (tok/s) | 86.5 | **95.3** | +10.2% |
| C4 aggregate (tok/s) | 105.9 | **114.8** | +8.4% |
| Coding-agent load, 4 x 1,024 (tok/s) | 239.9 | 238.7 | −0.5% |

- spark-bench's geometric mean (C1 x3, C2, C4): +7.0%. Replies bit-identical; the prompt set within ±0.8%.
- One 4,096-token coding request: 74.6 tok/s with `fnc5:0.2` (three runs, 74.55-74.60) against 75.4 with the policy
  unset on an earlier boot (−1.1%).
- `fcost7:noisy` measured +9.9% on spark-bench but −5.2% on the coding load, and was not kept.

**Her lane's `fnc7:0.3` on this split.** Mia's lane runs her noise-aware DFlash2 draft policy at
`TF_GLM_DFLASH_POLICY=fnc7:0.3` (patch 0019). On the coding-agent load, with prompt pairs and the chunked KDA kernel on
in both arms (median of three runs, tok/s):

| Shape | Policy off | `fnc7:0.3` (her lane) | On vs off |
| --- | ---: | ---: | ---: |
| 4 concurrent requests, 1,024 tokens each | **241.6** | 222.4 | −7.9% |
| 1 request, 4,096 tokens | **75.4** | 71.4 | −5.3% |

- Runs: off 234.5 / 242.3 / 241.6 and 75.4 / 75.4 / 75.4; on 216.4 / 222.9 / 222.4 and 71.5 / 71.4 / 71.4. The first
  token took the same time in both arms.
- Her policy was tuned on replies of the serial KDA kernel. The chunked kernel's replies differ slightly, and with her
  policy on the drafter drafts fewer rows a round on them (the `TF_GLM_KDA_CHUNKED` note in
  [v2.0 staging's decode](#decode-spark-bench-median-of-3-toks)).
- To run her lane's policy, put `TF_GLM_DFLASH_POLICY=fnc7:0.3` in `ATTN_ENV` in place of `fnc5:0.2`. It changes how
  many rows a round verifies, never the replies.

### Measured and off (patch 0026)

| Switch | Measured with | Result |
| --- | --- | --- |
| `TF_GLM_PREFILL_LANES=4` | four exchanges in flight | out of memory on the first ~8K cold prompt, at 31,800 MiB: the 5090's memory rose 1,280 MiB a step, six steps, one 1.25 GiB working segment of the sparse attention per later lane (prompt pairs add three) |
| `TF_GLM_WIRE_FP8=1` | two exchanges in flight | KL against the serial KDA kernel's reference 0.00454 / 0.0772 / 98.39% (fails the gate of 0.003 / 0.05 / 98.5%); greedy 2 of 5 equal the bf16 wire's, as a lossy codec's must; fresh 8K / 31K / 62K 2,212 / 2,318 / 2,435 tok/s (−3.2 / −0.7 / +0.7%), cold 100K 41.60 s (−0.1%), cold ~2.6K 1.529 s (+2.8%): the four rates' geometric mean −0.8% |

## v2.0's first staging

The tables below are this version's first staging, before the levers above. *v2.0 staging* in them is that
configuration: prompt chunk pairs on, Mia's draft policy off, 32 kept prompts at 8 GiB, four streams, one link to each
Spark.

### Decode (spark-bench, median of 3, tok/s)

| Metric | v2.0 staging | This recipe, pairs off, policy on | Mia's recipe v1.5, 2x DGX Spark | Stock TensorFold 0.6.5 TP2, two Sparks |
| --- | ---: | ---: | ---: | ---: |
| C1 prose | 72.0 | **78.9** | 58.5 | 32.5 |
| C1 code | 82.1 | **87.9** | 65.7 | 38.0 |
| C1 JSON | 76.2 | **81.2** | 66.6 | 35.7 |
| C4 aggregate | 106.2 | **118.7** | 101.5 | 35.2 |

- With her policy on, against Mia's recipe: 1.35x / 1.34x / 1.22x single-stream and 1.17x at four streams. Her v1.5
  and v1.4 measure the same on this bench (58.4 / 65.9 / 66.5 and 58.4 / 65.4 / 66.2 in one clean session).
- Without her policy, v2.0 staging decoded 6-11% below the policy-on run; the draft policy setting above
  (`fnc5:0.2`) recovers most of that.
- Prompt pairs leave decode unchanged. On two boots of one session (policy on in both), pairs on against off measured
  78.6 / 87.6 / 81.3 against 78.6 / 88.1 / 81.0 single-stream and 117.8 against 117.3 at four streams (+0.0%, −0.6%,
  +0.4%, +0.4%).
- Stock TensorFold 0.6.5 decodes one request at a time (its C4 aggregate equals one stream), and on two Sparks it
  admits at most 131,072 tokens a request at its default memory reserve; this recipe serves 262,144.
- `TF_GLM_KDA_CHUNKED` costs some decode speed through the replies, not the kernels (decode keeps the serial KDA
  kernel). In one session on the same build, with her policy on, on vs off: 78.8 / 87.7 / 81.3 vs 79.8 / 85.5 / 95.2
  single-stream and 116.5 vs 122.1 at four streams. The chunked kernel changes the replies a little (the JSON prompt's
  most), and the drafter then drafts fewer rows a round on them (2.72 vs 2.94).

### Prompt processing

| Prompt | This recipe, pairs on | This recipe, pairs off | Mia's recipe v1.5 | Stock 0.6.5 TP2 |
| --- | ---: | ---: | ---: | ---: |
| spark-bench's ~2.6K-token prompt (tok/s) | **1,147** | 975 | 1,569 | 507 |
| fresh 8K / 31K / 62K (tok/s) | **1,298 / 1,315 / 1,330** | 1,053 / 1,063 / 1,066 | — | — |
| cold 100K, time to first token | **75.4 s** | 93.9 s | — | — (admits 131,072) |
| the arena's 100K context load, two streams (tok/s) | **1,206** | 1,002 | 1,611 | — |

The first three rows come from the two boots of one session, with her policy on in both; the last row from the arena
runs (v2.0 staging, and the first grid). Prompt processing was the recipe's weak side, even with pairs: about 0.73x
Mia's two Sparks on spark-bench's prompt and 0.75x on the arena's 100K load. Her Sparks run the routed experts' prompt
windows on both GPUs with her dedicated EXL3 prompt kernel, which this update ports
([above](#her-exl3-prompt-kernel-patch-0024)).

### Prompt chunk pairs (patch 0022)

With no stream decoding, a prompt fills in 2,048-row chunks, and pairs put two of them through the layers together:
one chunk's attention runs on the 5090 while the Sparks compute the other's experts. On one timed boot:

- A lone 2,048-row chunk's MoE exchange takes 36.5 ms at the median and 40.3 ms at p90, and the Sparks need about
  28 ms of it to compute the experts.
- In a pair, the exchange window is 0.975x the lone one at the median and 0.977x at p90. The 5090's work inside it,
  the other chunk's attention, takes at most 21.9 ms at p90, so it fits inside the experts' compute time on at least
  90% of the paired exchanges.
- Prompts fill faster: fresh 8K, 31K and 62K prompts by 23.3%, 23.7% and 24.8%, spark-bench's ~2.6K prompt by 17.6%,
  and a cold 100,001-token prompt's first token comes in 75.4 s instead of 93.9 s.
- The 5090's peak memory rose from 22.51 GiB to 25.21 GiB over the prompt set and the cold 100K prompt.

### Spark Arena grid (28 cells)

This recipe's cells next to Mia's recipe on its own two Sparks: her v1.4 grid, and the six cells re-measured on her
v1.5. The cells from 16,384 up were measured on v2.0 staging; the cells below, in the first grid, with prompt pairs off
and her draft policy on. Each entry is prompt t/s / generation t/s, then the e2e time to the first token of the
2,048-token prompt after the context. Cache hits are the share of prompt tokens served from kept prompt states. Prompt
rates that llama-benchy cannot measure (at one stream the first token arrives within its latency estimate) show as —.

| Context x streams | Measured | This recipe: prompt / gen t/s, first token | Cache hits | Mia v1.4 | Mia v1.5 | Gen, ours / v1.4 |
| --- | --- | --- | ---: | --- | --- | ---: |
| 0 x 1 | pairs off, policy on | — / 59.0, 2.1 s | 0.0% | — / 42.1, 1.4 s | — / 38.0, 1.3 s | 1.40x |
| 0 x 2 | pairs off, policy on | 998 / 68.7, 4.0 s | 0.0% | 1,620 / 53.8, 2.4 s | — | 1.28x |
| 0 x 5 | pairs off, policy on | 668 / 44.5, 8.3 s | 0.0% | 882 / 43.5, 5.7 s | — | 1.02x |
| 0 x 10 | pairs off, policy on | 632 / 39.6, 17.1 s | 0.0% | 726 / 42.9, 13.7 s | 754 / 43.1, 13.5 s | 0.92x |
| 4,096 x 1 | pairs off, policy on | — / 57.5, 2.2 s | 39.3% | not run | — | — |
| 4,096 x 2 | pairs off, policy on | 892 / 65.3, 4.3 s | 39.4% | 1,422 / 53.3, 2.6 s | 1,393 / 51.4, 2.6 s | 1.22x |
| 4,096 x 5 | pairs off, policy on | 648 / 41.2, 9.2 s | 39.4% | 892 / 45.3, 6.3 s | — | 0.91x |
| 4,096 x 10 | pairs off, policy on | 588 / 37.4, 18.6 s | 39.4% | 699 / 40.3, 14.6 s | — | 0.93x |
| 8,192 x 1 | pairs off, policy on | — / 59.7, 2.3 s | 44.1% | — / 41.5, 1.4 s | — | 1.44x |
| 8,192 x 2 | pairs off, policy on | 886 / 64.6, 4.3 s | 44.1% | 1,401 / 48.7, 2.6 s | — | 1.33x |
| 8,192 x 5 | pairs off, policy on | 645 / 43.2, 9.2 s | 44.1% | 757 / 44.5, 6.6 s | 772 / 43.7, 6.7 s | 0.97x |
| 8,192 x 10 | pairs off, policy on | 582 / 37.0, 18.8 s | 44.1% | 682 / 40.1, 14.8 s | — | 0.92x |
| 16,384 x 1 | v2.0 staging | — / 62.4, 2.0 s | 46.9% | — / 43.6, 1.4 s | — | 1.43x |
| 16,384 x 2 | v2.0 staging | 1,055 / 60.7, 3.6 s | 46.9% | 1,377 / 51.7, 2.7 s | — | 1.17x |
| 16,384 x 5 | v2.0 staging | 611 / 39.4, 8.9 s | 46.9% | 737 / 42.1, 6.8 s | — | 0.93x |
| 16,384 x 10 | v2.0 staging | 576 / 35.3, 18.2 s | 46.9% | 683 / 39.3, 15.0 s | — | 0.90x |
| 32,768 x 1 | v2.0 staging | — / 54.6, 2.1 s | 48.4% | — / 57.4, 1.5 s | — / 38.4, 1.5 s | 0.95x |
| 32,768 x 2 | v2.0 staging | 1,038 / 58.5, 3.7 s | 48.4% | 1,297 / 50.3, 2.8 s | — | 1.16x |
| 32,768 x 5 | v2.0 staging | 642 / 40.7, 8.9 s | 48.4% | 698 / 41.2, 7.2 s | — | 0.99x |
| 32,768 x 10 | v2.0 staging | 578 / 35.5, 18.8 s | 48.4% | 658 / 39.3, 15.8 s | — | 0.90x |
| 65,535 x 1 | v2.0 staging | — / 65.1, 2.1 s | 49.2% | — / 40.6, 1.6 s | — | 1.60x |
| 65,535 x 2 | v2.0 staging | 1,022 / 62.1, 3.7 s | 49.2% | not clean | — | — |
| 65,535 x 5 | v2.0 staging | 619 / 39.7, 9.1 s | 49.2% | 721 / 42.9, 7.6 s | — | 0.92x |
| 65,535 x 10 | v2.0 staging | 59.7 / 3.5, 151.7 s | 19.7% | 625 / 38.7, 16.9 s | — | 0.09x |
| 100,000 x 1 | v2.0 staging | — / 51.9, 2.2 s | 49.5% | — / 48.0, 1.7 s | — | 1.08x |
| 100,000 x 2 | v2.0 staging | 1,023 / 61.0, 3.8 s | 49.5% | 1,118 / 49.6, 3.4 s | 1,137 / 52.7, 3.3 s | 1.23x |
| 100,000 x 5 | v2.0 staging | 236 / 16.1, 83.1 s | 29.7% | 678 / 42.6, 8.1 s | — | 0.38x |
| 100,000 x 10 | not run | — | — | not run | — | — |

- **27 of 28 cells measured, every attempt clean** (no other client's request inside or between them): the 12 below
  16,384 from the first grid, and 15 of the 16 from 16,384 up on v2.0 staging. 100,000 x 10 was not run.
- **Generation** over the 24 cells both grids had before 65,535 x 10: 1.05x Mia's v1.4 (geometric mean). By stream
  count: 0.95-1.60x at one stream, 1.16-1.33x at two, 0.91-1.02x at five apart from 100,000 x 5 (0.38x), and
  0.90-0.93x at ten. Beyond four streams a request waits for a slot.
- **First token** over the same cells: 1.49x Mia's time (geometric mean), and 0.76x her prompt rate where both sides
  report one. That is the prompt-processing gap above.
- **Cache hits equal Mia's in every cell both grids ran before 65,535 x 10, except 100,000 x 5** (39.4-49.5% at
  depth): on the 64-token prompt grid, a conversation's kept states sit where hers do. At 65,535 x 10 they fell to
  19.7%.
- **The 5090's KV pool limited the deepest cells.** One pool of 626,688 tokens held every request's cache and the kept
  prompt states; Mia's two Sparks hold about 1.74-1.80M tokens. At 100,000 x 5, kept states were evicted and prompts
  filled again: the first token took 83.1 s against her 8.1 s. At 65,535 x 10, whose contexts alone exceed that pool,
  generation fell to 3.5 tok/s and the first token to 151.7 s, against her 38.7 tok/s and 16.9 s. Twenty kept prompts
  and the room rule remove both ([above](#the-pools-room-rule-and-twenty-kept-prompts-patch-0025-cache_entries20)).

### Prompt chunk pairs in the arena (16,384 and up)

The cells from 16,384 up on v2.0 staging, against the first grid's (pairs off, policy on). The two runs differ in the
draft policy as well as in pairs.

| Context x streams | v2.0 staging: prompt / gen t/s, first token | Pairs off, policy on | Prompt, on / off | Gen, on / off | First token, on / off | Cache hits, on / off |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 16,384 x 1 | — / 62.4, 2.0 s | — / 58.9, 2.3 s | — | 1.06x | 0.89x | 46.9% / 46.9% |
| 16,384 x 2 | 1,055 / 60.7, 3.6 s | 886 / 66.7, 4.3 s | 1.19x | 0.91x | 0.83x | 46.9% / 46.9% |
| 16,384 x 5 | 611 / 39.4, 8.9 s | 605 / 42.3, 9.4 s | 1.01x | 0.93x | 0.94x | 46.9% / 46.9% |
| 16,384 x 10 | 576 / 35.3, 18.2 s | 590 / 36.3, 18.3 s | 0.98x | 0.97x | 0.99x | 46.9% / 46.9% |
| 32,768 x 1 | — / 54.6, 2.1 s | — / 59.7, 2.3 s | — | 0.91x | 0.90x | 48.4% / 48.4% |
| 32,768 x 2 | 1,038 / 58.5, 3.7 s | 831 / 61.4, 4.5 s | 1.25x | 0.95x | 0.81x | 48.4% / 48.4% |
| 32,768 x 5 | 642 / 40.7, 8.9 s | 596 / 41.0, 9.6 s | 1.08x | 0.99x | 0.93x | 48.4% / 48.4% |
| 32,768 x 10 | 578 / 35.5, 18.8 s | 588 / 37.5, 18.8 s | 0.98x | 0.95x | 1.00x | 48.4% / 48.4% |
| 65,535 x 1 | — / 65.1, 2.1 s | — / 57.6, 2.3 s | — | 1.13x | 0.90x | 49.2% / 49.2% |
| 65,535 x 2 | 1,022 / 62.1, 3.7 s | 863 / 63.4, 4.5 s | 1.18x | 0.98x | 0.83x | 49.2% / 49.2% |
| 65,535 x 5 | 619 / 39.7, 9.1 s | 595 / 41.4, 9.8 s | 1.04x | 0.96x | 0.93x | 49.2% / 49.2% |
| 65,535 x 10 | 59.7 / 3.5, 151.7 s | 49 / 3.1, 194.6 s | 1.22x | 1.13x | 0.78x | 19.7% / 18.0% |
| 100,000 x 1 | — / 51.9, 2.2 s | — / 57.4, 2.3 s | — | 0.90x | 0.95x | 49.5% / 49.5% |
| 100,000 x 2 | 1,023 / 61.0, 3.8 s | 856 / 67.6, 4.5 s | 1.20x | 0.90x | 0.83x | 49.5% / 49.5% |
| 100,000 x 5 | 236 / 16.1, 83.1 s | 412 / 28.9, 30.4 s | 0.57x | 0.56x | 2.74x | 29.7% / 42.9% |
| 100,000 x 10 | not run | not run | — | — | — | — |

- Over the 14 cells clean in both before 65,535 x 10 (geometric means): the prompt rate 1.05x, the first token 0.975x
  the time, generation 0.93x, and the wall time 0.92x (7,162 s against 7,779 s).
- Pairs run only while no stream decodes, so the two-stream cells gain most: prompt 1.18-1.25x, first token
  0.81-0.83x.
- Generation is about even at one stream (0.90-1.13x) and 0.90-0.99x at two to ten streams. Pairs alone left decode
  unchanged on spark-bench (above), and the policy differs between the runs, so this comparison cannot say which of the
  two lowered generation at two streams and more.
- At 100,000 x 5 the pool limit bit harder: cache hits 29.7% against 42.9%, and the first token 83.1 s against 30.4 s.

## Quality and correctness

| Check | Result |
| --- | --- |
| Greedy replies identical across runs; drafted = undrafted; each concurrent reply equals its solo reply | yes, on v2.0 staging: the greedy set equal across two runs (5 of 5), drafted = plain, and 22 of 22 concurrent requests equal their solo replies; all of it equal to the pairs-off, policy-on boot's (33 of 33 recorded cases). As shipped: deterministic within the boot (greedy x3 one reply, seeded runs equal, drafted = plain), 22 of 22 concurrent requests equal their solo replies with eight decoding together, 64 of 64 concurrent-bench replies equal, and the greedy set equal across two runs (5 of 5); its replies differ from v2.0 staging's, because her attention-side prompt kernels change the prompt arithmetic |
| Prompt chunk pairs on against off | identical: the greedy set (5 of 5), the 22 concurrent requests' 33 recorded cases, and the prompt cache's replies and cached counts; on CPU, 42 of 42 identity cases |
| The new kernels against the kernels they replace (every switch except `TF_GLM_KDA_CHUNKED`) | byte-identical: 1,323 of 1,323 checks on the 5090, 233 of 233 on each Spark; greedy 5 of 5 and 33 of 33 concurrent rows equal the replies without the switches |
| Mia's chunked KDA kernel, this tree vs her recipe's | byte-identical outputs and states in all 36 cases checked (two head counts, six row counts, three positions), on the 5090 and on a GB10 |
| `TF_GLM_KDA_CHUNKED=1` against the serial kernel (log-probs, 1,866 positions) | mean KL 0.002967, p99 0.05025, top-1 agreement 98.821%: noise-like (the KL sits where the reference itself is unsure, and top-1 changes only at near-ties) |
| 4-bit dense weights and FP8 cache rows byte-identical to Mia's recipe from the same checkpoint | yes: 93 of 93 dense-weight digests and 26 of 26 FP8 cache arrays, each on the 5090 and on a GB10 |
| Wire check (`AFD_CHECK=1`) mismatches | 0 in 256 checked forwards: 4 of them in prompt chunk pairs, 81 of them layer slices (pairs and the chunked KDA kernel on); the same with two exchanges in flight |
| Long context: a needle in ~195K tokens (one request) | found: 194,888 prompt tokens, prefill 185.2 s, before this update; 194,798 tokens in 103.5 s with her EXL3 prompt kernel; as shipped, 194,886 tokens in 74.3 s |
| Long context: a needle at ~240K tokens under load | found: 239,873 tokens with eight requests running, prefill 113.8 s, 154.5 s in all |
| AEON-30, idle (measured without `TF_GLM_KDA_CHUNKED`) | 23 of 30 at 87.2 tok/s; Mia's recipe v1.5: 23 of 30 at 66.3 tok/s |
| AEON-30, as shipped (idle, one task at a time) | 23 of 30 at 83.1 tok/s, 1,498 s; Mia's recipe v1.5: 23 of 30 at 66.3 tok/s, 2,168 s |

## Stability

The shipped configuration, on the same boot as [its numbers](#the-release-configuration): a 1.58-hour run of mixed
load at up to eight requests at once (agentic, chat, coding and tool-calling requests, and long prompts of 16K to
100K tokens). The load paused, 0.22 hours in all, for the quiet parts of the probe rounds (every 20 minutes) and for
the decode checks at the start, middle and end, so 81.6 minutes were under load.

| Item | Result |
| --- | --- |
| Time under load; requests; 5xx responses; restarts | 81.6 minutes; 1,169 requests, each answered 200; 0; 0 |
| Probe rounds | 8 rounds, 42 of 42 checked replies correct; the `tool_choice: "none"` probe, information only, passed 0 of 14 (the model still writes call markup as text: [limits](DESIGN.md#limits)) |
| Decode drift, spark-bench at the start / middle / end (tok/s) | prose 77.4 / 77.1 / 77.2, code 79.2 / 79.6 / 79.4, JSON 91.4 / 90.9 / 90.6, C4 111.9 / 112.2 / 111.9; the largest change −0.9% |
| Memory creep (GiB/h) | attention host +0.23 (its GPU 0.0); the two Sparks −0.02 and −0.07 |
| A 100K-token prompt sent while seven other requests decode | first token after 117.6 s at the median and 155.4 s at p95 (alone: 36.8 s) |
| Failure drills: restart one expert node, the attention node or the MCDMA daemons; cold start of the whole stack | not run for this release |

## Memory

| Node | Measured |
| --- | --- |
| Attention node: startup estimate at 262,144 | 17.92 GiB within the card's 28.80 GiB budget at eight streams with the 64-row window (16.27 GiB at four); one KV pool of 727,040 tokens (966,656 at four streams and 8 GiB) |
| Attention node: peak while serving | as shipped, 24,884 MiB over its whole boot (three arena cells and the mixed load included); 26,926 MiB at eight streams (the boot's whole run, four arena cells included). v2.0 staging: 25.36 GiB over its arena grid |
| Each Spark: available memory with the experts loaded | 28.8 and 29.5 GiB on v2.0 staging; as shipped, the lowest over the whole boot 28.44 and 29.21 GiB (28.66 and 29.49 GiB during the mixed load) |

## Comparisons, and what they are not

The Mia columns are her own recipe on its own pair of Sparks, with the same checkpoint and drafter: someone choosing
today has her recipe or this one. Her latest release is v1.5; the arena grid shows her v1.4 grid in full and her v1.5
where a cell was re-measured, and on spark-bench the two releases measure the same. Her lane runs her draft policy at
`fnc7:0.3`, which this version runs at `fnc5:0.2` ([above](#the-draft-policy)). The stock column is unmodified
TensorFold 0.6.5 at its largest admitted window on the same two Sparks this recipe uses for its experts. Each column
names its harness; a number measured on a different harness or checkpoint is labelled as such, never mixed in. The
two systems are different topologies: this one adds an RTX 5090, so it costs more than two Sparks alone.

The README also quotes [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0)'s
numbers from its own README (2026-09-30): glm53f-afd with Local Inference Lab's TR3 4-bpw experts, measured with its
spark-bench-style harness on 1,024-token prose, code and JSON prompts. They were not re-measured on this version's
harness.
