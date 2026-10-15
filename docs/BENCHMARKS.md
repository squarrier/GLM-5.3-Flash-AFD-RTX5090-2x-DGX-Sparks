# Benchmark method and qualification limits — v3.0

Final measurements below bind the exact 1fab9035 shipping source and shared/batched/ping-pong profile. A CPU build or fake test is not a served-model result. All original failures remain visible, including the qualified timing flake, legacy parser FALSE and CPU report finalizer OOM.

## Final same-window comparison

The final window uses this three-node candidate and Mia's current two-Spark reference. Three comparison blocks run after warm-up, mid-soak and before close:

1. Pause the soak's own load, without changing either lane's selected configuration.
2. Run this lane, the reference lane, then this lane, on the same recorded 300-token C1/C4 harness and prompts.
3. Record exact engine, recipe, image, checkpoint, drafter, precision, KV, context and serving flags for both arms. Check reference currency before measuring.
4. Record request overlap/contamination and the disposition of any retries. Retain contaminated raw results, but do not include them as clean takes.
5. Resume soak load; exclude comparison-block time from the active soak duration.

The final report must show each block, sample counts, aggregation method, per-request rates beside aggregates, TTFT, errors and any configuration mismatch. No older reference figure is a substitute for a new same-window cell. A warmed/cached request and a fresh request are not interchangeable.

### Rate definitions

- Aggregate end-to-end tokens/wall is the group's output tokens divided by its measured group wall time.
- Group share is aggregate divided by concurrency. It is not an independently measured stream rate.
- Request-wall mean averages each completed request's tokens divided by its own wall time.
- Decode mean excludes TTFT and the first token, following the recorded harness definition.
- TTFT is reported separately, with its units, sampling and averaging stated.

Multiplying mean per-request rates by concurrency need not reproduce aggregate throughput because streams finish at different times. Do not compare aggregate throughput to a single stream's rate or conceal queue time by relabelling it decode time.

## Historical native lab window: 2026-10-13

The README's Shared OFF / Shared ON table comes from the 04:24–07:07 EDT lab gate window on native source `6b0d285ad79e05038f6283534a5f21a32ddcbde0`. Four users, the shared state pool, interleaved prefill, grammar, sampling and built-in MTP were selected. Shared ON additionally selected shared forwarding and drafted shared verification.

C1 drafted aggregate was 72.24 OFF / 71.97 ON tok/s; C2 undrafted aggregate 59.36 / 66.50; C4 drafted aggregate 61.91 / 79.70. C4 group shares were 15.48 / 19.92, request-wall means 15.64 / 22.62, decode means 16.03 / 23.56, and mean TTFT 0.469 / 0.489 seconds. The ON/OFF C4 ratio was 1.287x on that lab harness. These rounded values are not final same-window reference results.

The original gate was FAIL: tool probes 9/11. Both unconstrained literal-closer copy attempts lacked the required served premise; they were not waived. Other recorded evidence included AEON-30 23/30 with case-for-case reference outcomes, structured users 36/36 and seeded users 8/8. Faster Shared ON was not adopted by that gate; the recorded shipping selection was Shared OFF. A fixed whole/streamed literal-closer boundary proof is required in the final qualification.

Lab hardware: RTX 5090 at 400 W (stock 575 W); Sparks at 2,200 MHz (stock 3,003 MHz), boots before 2026-10-10 10:25. A 2,100 MHz next-boot cap was configured on 2026-10-10. Do not relabel the historical results with that future cap. Final settings and boot dates must be reported with the final run.

## Soak and lifecycle coverage

The final mixed-load soak requires at least four hours of active load, excluding comparison blocks, with its own correctness, drift, memory-creep, loaded floors, thermal, long-context, AEON-under-load and orderly-close receipts. A passed short gate or idle AEON run does not establish these results.

The 2026-10-13 separate two-cycle lifecycle result covers Shared OFF only. It did not qualify the later Shared ON candidate. The separate 2026-10-15 shipping-head test now qualifies Shared ON/batched MTP/ping-pong, 2/2 cycles with 6/6 clean native exits and 27/27 physical checks; the older OFF-only result is not substituted.

No private service controller, watchdog, reboot policy or recovery script is distributed in this source preview. Their prior tests are not runtime guarantees for the public role launcher. The launcher provides local foreground print/check/run wiring, not an approved deployment transition.

## Older published versions

[v2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0) preserves the previously published Python TensorFold recipe. [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) preserves the glm53f-afd recipe. Their own README and benchmark pages retain the original method, checkpoint, licence, power settings and limitations.

Their numbers in the current README are historical, not remeasured controls. In particular, the published four-stream aggregate rates 111.6 / 117.6 tok/s use different workloads/configurations from the native candidate's lab table. The earlier 8/16-slot capacities and larger pools remain reasons to retain the old tags; they are not equivalent concurrency promises for this four-user candidate.

## Release scope

This is a source-only distribution. Soak, restart and follow-up measurements are actual original hardware receipts; release staging itself performs no host operation. No generated CUDA object/libdevice byte, weight or image is distributed. A binary release needs its own exact toolkit legal review. Public release is separate from production/default adoption.

## Comparison receipts

The final C1/C4 cells use the pre-registered before-close block, this lane's after-reference arm versus the two-Spark reference, medians of three clean repetitions. All three blocks and both bracketing arms must remain visible in the filled release; no fastest-block selection or aggregate/4 substitute for an actual stream rate is allowed. The README's directional statements use the filled figures; older-version columns remain explicitly historical.

The historical column is labelled "published v2.15 lab numbers, not the same window". Its sources are [v2.15 README:77–83](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L77-L83), [README:133](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L133) and [BENCHMARKS:78–85](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L78-L85). The often-cited [BENCHMARKS:216](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L216) and [BENCHMARKS:221](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L221) describe the older v2.1 same-window fresh-prompt and needle rows, not v2.15 measurements; do not relabel them.

The separate fresh-prompt, cold-TTFT and C2/C8 figures use their own 2026-10-15 same-window receipts. The v2.15 coding-agent rows generated 1,024 tokens, whereas the current comparison harness requests 300. The ~2.6K prompt's 2,420 tok/s is prompt processing, not the measured cold first-token time in seconds. No matching v3.0 tok/s value is invented for that unplanned cell.

## Requests without drafts

Schema-constrained JSON, required/named tool calls and temperature > 0 do not draft on this users path. Required/named tools set `Request.call`, structured requests set `Request.structure`, and [lane_host.zig](../engine/zig/src/core/lane_host.zig) excludes these from `loneRequest`; [glm_cuda_users.zig](../engine/zig/src/native/glm_cuda_users.zig) additionally restricts drafts to greedy, unstructured requests within reach. [cuda_afd_users_dsv.zig:78–86](../engine/zig/src/families/glm/cuda_afd_users_dsv.zig#L78-L86) rejects a drafted masked/sampled window. JSON-looking text requested by a prompt is not schema enforcement.

The 2026-10-13 pre-soak Part1 receipt `part1-verdict.json`, JSON C4, records 49.11885524789362 total tok/s (median of three groups), with actual request-wall rates 12.257049118771334–13.039506483844704 tok/s. Its client sends a JSON instruction, `draft:false`, temperature 0 and no `response_format`; it therefore does not establish schema-constrained C4 throughput. The later schema-drafting lab's 2026-10-14 receipt states schema8 0/152 requests per arm, zero schema-ON boots. No measured schema C4 number is asserted here: schema-constrained C4 throughput is NOT MEASURED.

That lab stopped at its inherited M-off C1 gate: JSON 34.742519423 → 35.875435427 tok/s, ratio 1.0326089190587613 (+3.260891906%), above the unchanged +3% upper band. Original run/audit exit 1 remains FAIL even though that rate increased. Schema equality/gain, dedicated prose C4 and schema-ON measurement were not run; this was not a measured schema-output failure. schema-drafting candidate stays out/default off. 16-physical-row decode and eviction changes stay out/default off too. No post-v3 lever is admitted by these statements.

## Rows, context and shipped profile

The exported source and manifest bind the actually soaked `1fab9035bb24bb3d647103c7d4285bb230a9a52e`; `.env.example` selects shared verify, shared drafted verify, batched MTP and ping-pong ON. Packaging changes only private identifiers/comments and neutral test addresses; no model arithmetic or kernel body is changed.

Current [cuda_afd_users_dsv.zig:35–40](../engine/zig/src/families/glm/cuda_afd_users_dsv.zig#L35-L40) defines eight total physical rows, four rows per user window (one pending token plus up to three drafts). Its `total` refuses `WindowRows` or `Rows` beyond those bounds. The lane planner bounds each shared round; pending work remains for another round. Four user slots do not become eight because eight HTTP clients submit requests. [glm_cuda_users.zig:151–190](../engine/zig/src/native/glm_cuda_users.zig#L151-L190) checks context/pool/ownership and returns `NoFreeSlot` after all four slots are owned; the lane host's admission keeps waiting requests queued.

The selected 1fab source's ping-pong path retains that eight-row physical micro-cap but can plan 16 logical rows as two ≤8-row micros; that is not the omitted 16-physical-row lever. Do not describe 16 logical rows as 16 physical rows or eight concurrently decoding users. That selected path is included; the 16-physical-row extension remains absent/default off.

`.env.example` sets context 262,144 and requested pool 720,896; the source allocates 720,888 tokens. The draft reach is 2,051 positions; beyond it, a request continues undrafted. No vision encoder/API path is selected by this recipe. Toolchain pins are in `.env.example:5–17`, including the exact Zig archive SHA256 and CUDA/AOT image digests. Build locally; libdevice and all generated objects are external local-build inputs, not this source distribution.

## Restart coverage

The original 2026-10-13 M-off service receipt measured cold start 137.495 s, graceful restart 149.226 s and final stop 12.025 s, two cycles, loaded audits 6/6, startup bindings 2/2, native exits 6/6 at 0, physical checks 27/27, outer exit 0 and observer exit 0. Its shared-forward/shared-drafts switches were OFF and source was 6b0d285a. These counters are not M-on coverage. PASS 2/2 on the same shipping head/M-on profile; cold 197.586 s, graceful restart 145.404 s, final stop 12.144 s; PASS: both aliases and 3/3 canaries per cycle; original 900 s startup grace; loaded 6/6, startup bindings 2/2, native exits 6/6 = 0, physical 27/27; outer/observer exits 0; all owners absent. These are service-lifecycle timings, not speed gains.

## Gate dispositions

| Gate / receipt | Original result, retained | Result the desk used | Scope |
|---|---|---|---|
| 2026-10-15 P1-R2 legacy Part1 parser | FALSE; 27,388 one-row rounds; R2/multi-row 0 | On the same original log, the qualified physical-micro-aware parser gives PASS; 37,750 rounds total, 10,362 pipeline/multi-row rounds; widths 1:27,388, 2:4,430, 3:3,511, 4:2,421; zero problems | Desk 2026-10-15 12:40 disposition: use the qualified parser and retain legacy FALSE; no re-soak. Log SHA256 `a11883a1e861620e6a11f11c33696316a1fcd7431d2c16e3333b6c60157b8b2c`; parser SHA256 `b7049e4ed9dabef534c20488b86f53caa8befee1ab79cf4d80dc2abe321ba742`. Log-declaration proof, not numerical/speed proof or final soak acceptance. |
| Exact 1fab source, broad candidate Debug CPU suite | FAIL, rc1; 1,253/1,261 pass, 7 skipped, 1 after.fast failure, expected 3 found 2 | Same head, the failing test alone 3/3 Debug + 3/3 ReleaseSafe pass, under the recorded baseline-flake rule | Broad row stays FAIL, not PASS. Candidate ReleaseSafe 1,254/1,261 pass, 7 skipped, rc0; paired control Debug and ReleaseSafe each 1,220/1,227 pass, 7 skipped, rc0. Isolated passes qualify only the known timing signature; no unlisted failure is waived. |

These are two disclosed interpretations used by existing gates, not new hardware tests run for this restage. Final soak, service and public-release acceptance remain separate. The older v2.15 probe failure is also retained in the README's previous-version section and at [its tag](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L226-L231).

## All comparison blocks and actual streams

| Block / arm | C1 prose/code/JSON decode tok/s | C1 TTFT ms (same order) | C4 aggregate tok/s | Actual C4 decode range tok/s | Median mean C4 TTFT ms |
|---|---|---|---:|---|---:|
| after-warmup / ours_before | 74.3/71.7/79.1 | 163/196/224 | 94.6 | 24.3–27.1 | 558 |
| after-warmup / reference | 59.4/67.0/67.9 | 146/177/204 | 100.8 | 25.7–30.4 | 435 |
| after-warmup / ours_after | 74.5/71.9/79.3 | 162/193/222 | 94.9 | 23.7–26.9 | 555 |
| midpoint / ours_before | 74.2/71.8/79.1 | 164/193/222 | 94.3 | 24.0–27.1 | 554 |
| midpoint / reference | 59.3/67.1/67.8 | 146/177/204 | 102.6 | 25.8–30.7 | 345 |
| midpoint / ours_after | 74.6/71.8/79.1 | 162/197/226 | 94.8 | 24.0–28.5 | 569 |
| before-close / ours_before | 74.2/71.7/79.2 | 165/192/223 | 95.0 | 23.9–27.0 | 556 |
| before-close / reference | 59.5/67.0/67.8 | 148/179/206 | 100.5 | 25.8–30.4 | 385 |
| before-close / ours_after | 74.5/71.6/79.3 | 162/193/225 | 94.7 | 24.3–26.3 | 577 |

All three blocks CLEAN: 54/54 cells, 378 selected requests; 0 errors, overlap or retries, all original attempts retained. Both bracketing arms are disclosed. Complete producer-drain/resume pauses excluded 852.464/1116.258/875.630 s, not just benchmark-body duration. The unchanged stock production-safe client alone supplied reference traffic. The separate reference-only u4j comparison was symmetrically excluded; the original undrafted Part1 u4j gates did run.

- after_warmup / ours_before: decode [[24.3, 24.3, 27.1, 24.8], [25.7, 24.5, 25.8, 25.7], [25.6, 25.1, 25.3, 25.5]]; actual request-wall range 23.585–25.359 tok/s; TTFT ms [179, 414, 941, 635, 182, 417, 693, 941, 180, 690, 441, 939].
- after_warmup / reference: decode [[25.9, 29.7, 30.4, 26.3], [26.2, 29.4, 30.2, 25.9], [25.7, 29.6, 30.3, 26.1]]; actual request-wall range 25.084–29.297 tok/s; TTFT ms [163, 535, 534, 534, 332, 331, 332, 332, 165, 526, 525, 525].
- after_warmup / ours_after: decode [[23.7, 25.6, 24.3, 25.1], [24.7, 24.5, 25.6, 25.2], [24.2, 24.5, 26.9, 25.3]]; actual request-wall range 23.474–24.855 tok/s; TTFT ms [180, 938, 441, 662, 432, 210, 963, 654, 182, 424, 948, 647].
- midpoint / ours_before: decode [[25.7, 25.2, 25.2, 25.7], [25.1, 24.3, 26.0, 24.8], [24.0, 24.8, 25.3, 27.1]]; actual request-wall range 23.474–25.685 tok/s; TTFT ms [180, 415, 691, 939, 183, 423, 958, 651, 181, 416, 956, 636].
- midpoint / reference: decode [[26.1, 29.1, 30.3, 25.8], [26.7, 29.8, 30.6, 26.3], [26.8, 30.0, 30.7, 26.4]]; actual request-wall range 25.274–29.791 tok/s; TTFT ms [275, 274, 559, 272, 335, 334, 333, 333, 348, 347, 346, 347].
- midpoint / ours_after: decode [[25.3, 24.5, 25.8, 25.3], [28.5, 26.0, 24.0, 26.0], [24.7, 24.4, 25.6, 25.1]]; actual request-wall range 23.274–28.143 tok/s; TTFT ms [182, 417, 942, 638, 180, 703, 440, 953, 438, 211, 968, 659].
- before_close / ours_before: decode [[26.5, 24.7, 25.6, 26.4], [26.4, 26.3, 25.6, 25.7], [23.9, 26.0, 27.0, 25.5]]; actual request-wall range 23.678–26.155 tok/s; TTFT ms [181, 424, 960, 657, 180, 637, 950, 388, 181, 691, 443, 937].
- before_close / reference: decode [[25.8, 29.7, 30.4, 26.2], [26.2, 29.3, 30.0, 26.4], [26.1, 29.3, 29.9, 26.3]]; actual request-wall range 25.105–29.183 tok/s; TTFT ms [166, 520, 519, 519, 314, 313, 314, 581, 319, 320, 318, 584].
- before_close / ours_after: decode [[24.4, 24.4, 25.5, 26.3], [25.8, 24.3, 25.0, 25.7], [25.7, 24.5, 25.8, 25.7]]; actual request-wall range 23.678–25.359 tok/s; TTFT ms [434, 210, 710, 960, 433, 209, 707, 959, 181, 418, 697, 943].

## Prompt and concurrency follow-up

Separate same-window follow-up, 15:20–15:45 EDT, on the same shipping head/profile. Prompt figures are medians of both rounds, concurrency totals medians of all three takes. Fresh = endpoint prompt tokens / completed one-token wall, not pure kernel prefill; cold = salted/fresh prompt, thinking off, not process-cold. All 32 items/80 completions were CLEAN; no retries or cache hits in selected prompt requests. The reference nodes were monitored only; one reached 92.5 °C, so this is not constant-temperature hardware isolation. Reference clock/boot readback was not measured.

| Metric | v3.0 | Two-Spark reference |
|---|---:|---:|
| Fresh 8K, tok/s | 3111.49 | 1935.58 |
| Fresh 31K, tok/s | 3198.99 | 1996.25 |
| Fresh 62K, tok/s | 3180.87 | 1998.65 |
| Cold 100K TTFT, s | 32.47 | 50.45 |
| Cold ~2.6K TTFT, s | 1.26 | 1.59 |
| C2 total, tok/s | 87.40 | 80.20 |
| C8 offered total, tok/s | 92.10 | 101.00 |
| C2 actual stream decode range, tok/s | 45.0–47.7 | 41.4–46.8 |
| C2 actual stream TTFT range, s | 0.190–0.446 | 0.261–0.277 |
| C8 actual stream decode range, tok/s | 17.3–39.5 | 23.3–30.6 |
| C8 actual stream TTFT range, s | 0.191–18.470 | 0.340–13.499 |

C8 is eight offered HTTP requests on the unchanged four-user service, not eight resident decode slots. Queueing remains in the first-token times: some v3.0 requests waited up to 18.470 s. Fixed concurrency prompts are not a cache-neutral/fresh-prefill claim. The two-Spark reference needs one fewer host and no RTX 5090, supports vision and retains its own serving/cache policy. [All takes, stream rates and TTFT](#prompt-and-concurrency-follow-up).


Every selected concurrency take follows (all actual decode values, no total/C):

- ours/C2: totals [87.4, 87.8, 85.9]; stream decode [47.7, 45.0, 46.7, 46.7, 46.9, 45.6]; first-token seconds [0.446, 0.22, 0.19, 0.424, 0.19, 0.426].
- ours/C8: totals [93.4, 92.1, 92.0]; stream decode [24.4, 25.0, 21.1, 26.3, 24.3, 28.7, 26.3, 27.0, 26.4, 22.6, 27.3, 24.5, 17.3, 29.3, 25.9, 39.5, 25.4, 22.5, 25.9, 23.4, 23.1, 21.5, 28.3, 33.5]; first-token seconds [0.686, 0.963, 0.249, 0.461, 12.022, 13.163, 13.419, 14.63, 11.368, 0.488, 0.245, 0.707, 0.984, 13.968, 13.096, 18.47, 0.191, 0.426, 0.957, 0.645, 12.73, 12.186, 13.61, 13.907].
- reference/C2: totals [80.5, 80.2, 80.1]; stream decode [41.7, 46.8, 41.4, 46.5, 41.4, 46.6]; first-token seconds [0.277, 0.277, 0.261, 0.261, 0.266, 0.266].
- reference/C8: totals [101.0, 101.1, 101.0]; stream decode [25.0, 29.3, 30.6, 23.3, 25.9, 29.2, 27.1, 26.4, 23.7, 29.0, 30.4, 25.9, 25.8, 24.3, 24.9, 28.5, 23.6, 28.9, 30.3, 24.8, 25.8, 24.7, 28.6, 27.4]; first-token seconds [10.398, 0.348, 0.349, 0.347, 0.345, 13.499, 12.14, 10.832, 0.341, 0.342, 0.341, 12.148, 0.34, 10.385, 10.907, 13.233, 0.346, 0.345, 0.344, 10.943, 0.342, 10.42, 13.292, 12.215].

## Preserved failures and effective soak evaluation

The 2026-10-13 original soak failed before sustained dispatch on incomplete comparison heat telemetry. It is not a passing take or a source for the current comparison. The final 2026-10-15 soak passed 9,657/9,657 checked requests over 4.050086 active hours, with 105 non-gating observations and 208 expected HTTP 400 negative controls. All mandatory final gates pass, with P1-R2 evaluated only under the disclosed 12:40 parser disposition above. Its original automatic CPU finalizer then OOM-killed at 2 GiB, exit 143, after clean native/outer/physical close. Authorized file-only reporting recovery at 12 GiB returned 1 solely for legacy R2; the desk-used physical-micro parser resolved that predicate. The original auto exit is still FALSE, not zero. No second soak, close or GPU retry was run.

The service reporting collector initially compared a canonical JSON digest with a pretty-file SHA and recorded FALSE; the corrected file-only collector uses the original canonical rule. The actual two-cycle runtime was not repeated or changed. CPU report recovery and collector corrections are not model/hardware passes.

## Reference identity and unresolved runtime readback

HTTP readback identifies TensorFold, four decode slots, 1,796,096 pool tokens and a 262,144 context. Exact running engine/checkpoint/drafter revision, dense precision and KV dtype were not emitted by the saved HTTP identity; they remain NOT MEASURED, not inferred from recipe currency. The recipe currency check changed README only (`b5808ac6` to `3fd60dfc`), while actual source/image/default pins that HTTP did not expose remain unknown. No SSH inspection or reference reconfiguration was performed by this release fill-in. This is a configuration-disclosure limit, not proof of precision/checkpoint parity.
