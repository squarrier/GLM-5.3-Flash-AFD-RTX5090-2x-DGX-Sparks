# GLM-5.3-Flash AFD V4.2 — source-only release

**Provided as-is, without support. The measurements describe the tested deployment, not a rebuild of this source overlay.**

## Authors

[Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) also authored the [DFlash2 selector bounds patch 0014](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/blob/91a7ba6105ccbc961e34b337ef80a55e93a44040/patches/engine/0014-glm-dflash2-the-selector-skips-candidate-ids-outside.patch). This port moves the invalid-anchor normalization earlier and retains her authorship. [Inco AI](https://inco.ai) authored the external [DFlash2 checkpoint](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a); [Z Lab](https://github.com/z-lab) authored [DFlash](https://github.com/z-lab/dflash), whose original extraction revision remains unknown.

- [Ash Hart (@ashhart)](https://github.com/ashhart) authored [TensorFold v1.3.0](https://github.com/ashhart/TensorFold/tree/d3d0bf110d57161edfbd20234a84833f14548b57) and [MCDMA](https://github.com/ashhart/MCDMA). [Russell Marr (@russellmm)](https://github.com/russellmm) authored [pinned host KV](https://github.com/russellmm/TensorFold/commit/adc001b4bdf9fcdde38647ae5a46fbafd3e829f3); [Bizuayeu (@Bizuayeu)](https://github.com/Bizuayeu) authored the [MTP pooled-key fix](https://github.com/Bizuayeu/TensorFold/commit/b0e1cdcbab3b6ebc183129f9b26c9887ed22ff3e).
- [Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) authored the [Spark recipe and RoCE adaptations](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/80be4d6882837d8e0840827ae2aab6124fd48da3), [EXL3 quantization](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) and [attribution guide](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution). [Z.ai](https://huggingface.co/zai-org) authored the [base model and MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda).
- [Hugh Madden (@hughmadden)](https://github.com/hughmadden) authored the [attention/expert split and expert prompt kernels](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6). [T.J. Purtell (@tpurtell)](https://github.com/tpurtell) authored the [expert channel split](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark/tree/dc6d9b8e1600e001cb1d4228bd911f4df8091f99) and [CuteAFD](https://github.com/tpurtell/cuteafd/tree/b4818a73371788aa3b9dd54023c017646d109f01).
- [Scott Quarrier (@squarrier)](https://github.com/squarrier) authored the recipe packaging and directed the program; an AI assistant authored the native integration and packaging changes for Scott. See [full credits](#license-and-credits), [AI-assisted development](#ai-assisted-development), and the [historical V4.0.1 source manifest](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/9e1590000788254ee4f827524781b0a06421cb84/manifests/engine-source.json).

<details>
<summary>More authors and source revisions</summary>

[taussoe / @taussoe](https://github.com/taussoe) authored [KDA work atfb985b86](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c). [Turboderp / @turboderp](https://github.com/turboderp) authored [ExLlamaV3/EXL3 atf45b8451](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b). [Z.ai](https://huggingface.co/zai-org) authored [GLM-5.3-Flash and built-in MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda). [NVIDIA](https://github.com/NVIDIA) authored CUDA/libdevice; no compiled CUDA object is supplied here. [Scott Quarrier / @squarrier](https://github.com/squarrier) directed the recipe packaging; GPT-6.1 Sol via Hermes/openai-codex, an earlier local AI agent and Anthropic Opus 5.5 retain their distinct roles below. Exact per-file and previous-version authors remain in [CREDITS.md](CREDITS.md), [CREDIT-REGISTER.json](CREDIT-REGISTER.json) and every preserved notice.

[Raymond Lucke / @rwl4](https://github.com/rwl4) authored the [literal tool-closer cases at029a852f](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be). [Albert Lee / @bertholomus](https://github.com/bertholomus) authored [group-count/place and MMA work-list kernels](https://github.com/bertholomus/TensorFold/commit/90b68e063a0b8bba59aaf13340ef16619e751195). [SvangenStudios / @SvangenStudios](https://github.com/SvangenStudios) authored the [structured response-channel work in PR31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31), with the recorded Mia adapter at80be4d6 distinguished from original authorship. [Alfonso Fortunato / @devopsfortunato](https://github.com/devopsfortunato) authored [the malformed GLM argument-boundary fix](https://github.com/ashhart/TensorFold/pull/2831); its exact original merge SHA is not invented. Their earlier-candidate roles are retained, not assertions that each old patch was recopied onto the native base.

</details>

## Measured speed — V4.2 against the live Mia deployment

Same-window API tests on the repaired V4.2 TensorFold deployment with Q4 DFlash2 versus the Mia recipe deployment at [`d8a7439`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/commit/d8a7439e8c5aac8645c423ccd24189690359ab69). These compare whole deployments on different hardware/topologies and settings, not the speedup of an individual kernel or drafter. They do not represent Mia's latest upstream or an A/B against public V4.1. Aggregate rates use all output tokens divided by workload wall time; the per-stream ranges use post-first-token decode speed.

| Workload | Offered | V4.2 aggregate / per-stream, tok/s | Mia aggregate / per-stream, tok/s | Mean TTFT, V4.2 / Mia |
|---|---:|---:|---:|---:|
| Short concurrent, 300 output tokens/stream | 4 | **92.9–93.1** / 23.5–28.2 | **77.0** / 19.6–30.0 | 0.450 / 0.458 s |
| Short concurrent, 300 output tokens/stream | 8 | **118.4** / 15.2–18.0 | **85.1** / 16.0–34.8 | 0.836 / 7.357 s |
| Coding workload, 1,024 output tokens/stream | 4 | **111.63** / 28.15–29.49 | **96.32** / 24.26–26.47 | 0.849 / 0.885 s |
| Coding workload, 1,024 output tokens/stream | 8 | **132.06** / 17.05–20.20 | Not measured | 1.535 s / — |

At four offered, V4.2 increased short-workload aggregate throughput by **20.6–20.9%**; Mia still recorded the fastest individual short stream. V4.2's coding C4 mean end-to-end latency was **36.207 s** versus **41.514 s**. At eight offered, the Mia endpoint had only four native lanes, so its TTFT includes queueing and this is not an eight-resident comparison. The single-stream code rows were **69.3–69.4 tok/s** for V4.2 and **66.0–66.1 tok/s** for Mia (post-first-token). See [BENCHMARKS.md](BENCHMARKS.md) for the other C1 rows, methodology and limits. Earlier release results below are historical and are not denominators for a V4.2 speedup.

## KV and context — measured on the repaired deployment

**420,000 prompt-plus-output tokens verified for one stream**: a cold 419,488-token prompt retrieved its exact test value and generated all 512 requested tokens. The 420,001-token control was rejected before inference. Startup verified eight available lanes, 6,568 attached pages of BF16 history (41,871 MiB pinned host RAM) and 2,463 MiB of pooled keys on the attention GPU. The lowest sampled startup free VRAM was 1,814 MiB; this is not a bound on unsampled or fully occupied phases. **Eight simultaneously occupied 420,000-token windows and cancellation/refill at that occupancy have not been hardware-qualified.** This cap is a tested deployment setting, not a new default for the source overlay.

## Source scope and bounded reconstruction (no build)

This release is an affected-path overlay against official TensorFold v1.3.0 commit `d3d0bf110d57161edfbd20234a84833f14548b57`. It reconstructs **143 source paths plus four grant/notice aliases**, not a complete engine checkout, consumer build or deployment. It retains the prior public hybrid implementation, adds Mia's selector guard and the DFlash2 admission port, and includes the selected source's occupancy/lifecycle, prepaid-ring, bounded host-reader, early-parser and startup-observation changes. It also includes the allocation repair and coupled tests: the planner reserves one extra row before alignment, without changing the 32-row verification width or adding a required page to the 420K pool.

Only source text, reconstruction metadata and retained notices are supplied. No model weights, quantized drafters, compiled CUDA assets, executables, private runtime configuration, credentials or benchmark payloads are included. Users must separately obtain the complete pinned upstream tree, preserve its component notices, and build with their own CUDA toolkit. This source package does not supply the toolchain or certify redistribution rights for its outputs.

Use Python 3.9 or later and Git on Linux in an isolated directory:

```sh
python3 -B source/reconstruct.py --fetch --cache ./input-cache --output ./overlay --receipt ./reconstruction.json
```

The output must not exist. Optional `--fetch` reads only immutable public GitHub URLs in the lock; later replay can use the verified cache offline. Every resulting path, mode, size, Git blob and SHA-256 is checked. This command reconstructs the affected files only. It does not run a compiler, source test, GPU, model, service, or benchmark. Public cache labels and comment/diagnostic privacy edits are explicitly distinct from the privately built source bytes; there is no claim of a new native build of this exported tree.

## Third-party terms and unresolved boundaries

The root Apache-2.0 LICENSE matches the pinned TensorFold source and Mia selector patch. Older MIT source and nested terms remain applicable; a root grant does not relicense third-party material. Preserve [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [CREDITS.md](CREDITS.md), [source/upstream/](source/upstream/) and [source/grants/](source/grants/). The original hybrid GLM/oMLX adaptation and shared DFlash interfaces retain their existing component notices and unknown original extraction revisions. The selected DFlash source file is now part of the overlay; the earlier V4.1 statement that this overlay omitted its implementation does not describe V4.2.

The external Inco AI DFlash2 checkpoint at `bf582e4eacc1810f76656d1811693ff6c6737d2a` declares **CC BY-NC-ND 4.0**, research/evaluation, in its pinned model card; its standalone LICENSE URL returns 404. Historical engine notices that describe drafter weights as Apache-2.0 do not override that card. This package supplies no drafter weights and no instructions to redistribute or publish modified/quantized weights. The Q4 derivative's redistribution permission has not been established. Use the built-in MTP path instead where the external drafter's terms are not acceptable; that alternative's performance must be measured separately, not relabeled DFlash2 results.

The original source revision and extraction path for the inherited GLM/oMLX adaptation remain unresolved. The overlay retains the pinned TensorFold terms and the nested oMLX Apache-2.0 and mlx-vlm/Apple MIT notices that apply to the identified source. The historical oMLX license snapshot is not the original import pin. This does not establish redistribution rights for the full consumer-fetched engine, external checkpoints or build outputs.

## Provenance exceptions

- `github.com/ashhart/cortheon`: original source revision unknown.
- `github.com/descriptinc/descript-audio-codec`: original source revision unknown.
- `github.com/fla-org/flash-linear-attention`: original source revision unknown.
- `github.com/ggml-org/ggml`: original source revision unknown.
- `github.com/jundot/omlx`: original source revision unknown.
- `github.com/z-lab/dflash`: original source revision unknown.

Cortheon, DAC, FLA and ggml are historical or separately obtained upstream inputs; their direct implementations are not supplied by this overlay. The two GLM-related uses are described above. No unknown historical revision is inferred from a newer public snapshot.

## License and credits

All previous author names, source grants and historical credits remain in the retained credit files. [Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) authored the selector guard cited above and the [attribution guide](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution) retained byte-identically as [AGENTS.md](AGENTS.md). See [NOTICE](NOTICE) for this revision's modification statement. The root LICENSE remains the verbatim Apache-2.0 text. No source author's work is reassigned to the packaging agent.

## AI-assisted development

Scott Quarrier directed the program. GPT-6.1 Sol / OpenAI Codex via Hermes prepared the V4.2 port, offline source selection and packaging draft; this session verified custody and reconstruction, not historical per-call authorship. Earlier local-agent and Anthropic Opus 5.5 roles remain in CREDITS and inherited notices. This source reconstruction is not a hardware qualification, benchmark result or model-quality claim.

## Security

This package starts no listener or privileged container. A separate deployment must keep model APIs private/firewalled, use authentication where appropriate, isolate unauthenticated rank/RDMA ports, and review CUDA/container privileges and least-privilege operator access. Do not port-forward an unauthenticated model or rank endpoint. No live host settings or service changes are supplied here.

## Historical measurements — unchanged from public V4.1

### Historical decode speed — not v1.3.0 results

Median aggregate output tokens/s, with measured per-stream ranges in parentheses. Both older runs used one RTX 5090 and two DGX Sparks, the same short prompts, greedy 300-token generation, built-in MTP at confidence 0.50, and three measured repetitions per cell.

| Prompt | Concurrent requests | V4 / v1.2.8 | V4.1 lab / v1.2.9 |
|---|---:|---:|---:|
| Code | 1 | 63.75 (63.14–64.07) | 63.61 (63.52–64.56) |
| Code | 4 | 126.47 (31.59–32.79) | 126.22 (31.54–32.68) |
| Code | 8 | 207.26 (25.89–27.73) | 207.00 (25.87–27.72) |
| Chat | 1 | 54.83 (54.14–55.33) | 55.58 (53.59–56.07) |
| Chat | 4 | 106.04 (26.48–27.77) | 105.86 (26.46–27.73) |
| Chat | 8 | 173.01 (21.62–23.66) | 172.75 (21.59–23.62) |

These were separate lab runs, not an alternating same-window comparison. The rates were similar and do not establish a version speedup or long-context speed. The recorded RTX 5090 power limit was 400 W; GPU clock caps and Spark power limits were not established for these runs. The eight-window capacity result in the [V4.0.1 release](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v4.0.1) is also historical.

## Support

None; provided as-is. This is a source overlay, not a validated consumer build or a turnkey deployment.
