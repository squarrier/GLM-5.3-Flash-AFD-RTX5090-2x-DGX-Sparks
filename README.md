# GLM-5.3-Flash AFD V4.2 — TensorFold v1.3.3

**One RTX 5090 for attention + two DGX Sparks for experts, with BF16 KV history in host RAM.**
Short-workload throughput: **94.9 tok/s at four offered requests, 122.2 tok/s at eight**. Exact **420K prompt-plus-output** verified for one cold stream; eight full 420K histories are **not** qualified.

**Provided as-is, without support. Source-only overlay; measurements describe the tested deployment, not a new build of this public package.**

## Authors

[Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) also authored the [DFlash2 selector bounds patch 0014](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/blob/91a7ba6105ccbc961e34b337ef80a55e93a44040/patches/engine/0014-glm-dflash2-the-selector-skips-candidate-ids-outside.patch). This port moves the invalid-anchor normalization earlier and retains her authorship. [Inco AI](https://inco.ai) authored the external [DFlash2 checkpoint](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a); [Z Lab](https://github.com/z-lab) authored [DFlash](https://github.com/z-lab/dflash), whose original extraction revision remains unknown.

- [Ash Hart (@ashhart)](https://github.com/ashhart) authored [TensorFold v1.3.3](https://github.com/ashhart/TensorFold/tree/1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc) and [MCDMA](https://github.com/ashhart/MCDMA). [Russell Marr (@russellmm)](https://github.com/russellmm) authored [pinned host KV](https://github.com/russellmm/TensorFold/commit/adc001b4bdf9fcdde38647ae5a46fbafd3e829f3); [Bizuayeu (@Bizuayeu)](https://github.com/Bizuayeu) authored the [MTP pooled-key fix](https://github.com/Bizuayeu/TensorFold/commit/b0e1cdcbab3b6ebc183129f9b26c9887ed22ff3e).
- [Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) authored the [Spark recipe and RoCE adaptations](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/80be4d6882837d8e0840827ae2aab6124fd48da3), [EXL3 quantization](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) and [attribution guide](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution). [Z.ai](https://huggingface.co/zai-org) authored the [base model and MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda).
- [Hugh Madden (@hughmadden)](https://github.com/hughmadden) authored the [attention/expert split and expert prompt kernels](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6). [T.J. Purtell (@tpurtell)](https://github.com/tpurtell) authored the [expert channel split](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark/tree/dc6d9b8e1600e001cb1d4228bd911f4df8091f99) and [CuteAFD](https://github.com/tpurtell/cuteafd/tree/b4818a73371788aa3b9dd54023c017646d109f01).
- [Scott Quarrier (@squarrier)](https://github.com/squarrier) authored the recipe packaging and directed the program; an AI assistant authored the native integration and packaging changes for Scott. See [full credits](#license-and-credits), [AI-assisted development](#ai-assisted-development), and the [historical V4.0.1 source manifest](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/9e1590000788254ee4f827524781b0a06421cb84/manifests/engine-source.json).

<details>
<summary>More authors and source revisions</summary>

[taussoe / @taussoe](https://github.com/taussoe) authored [KDA work atfb985b86](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c). [Turboderp / @turboderp](https://github.com/turboderp) authored [ExLlamaV3/EXL3 atf45b8451](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b). [Z.ai](https://huggingface.co/zai-org) authored [GLM-5.3-Flash and built-in MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda). [NVIDIA](https://github.com/NVIDIA) authored CUDA/libdevice; no compiled CUDA object is supplied here. [Scott Quarrier / @squarrier](https://github.com/squarrier) directed the recipe packaging; GPT-6.1 Sol via Hermes/openai-codex, an earlier local AI agent and Anthropic Opus 5.5 retain their distinct roles below. Exact per-file and previous-version authors remain in [CREDITS.md](CREDITS.md), [CREDIT-REGISTER.json](CREDIT-REGISTER.json) and every preserved notice.

[Raymond Lucke / @rwl4](https://github.com/rwl4) authored the [literal tool-closer cases at029a852f](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be). [Albert Lee / @bertholomus](https://github.com/bertholomus) authored [group-count/place and MMA work-list kernels](https://github.com/bertholomus/TensorFold/commit/90b68e063a0b8bba59aaf13340ef16619e751195). [SvangenStudios / @SvangenStudios](https://github.com/SvangenStudios) authored the [structured response-channel work in PR31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31), with the recorded Mia adapter at80be4d6 distinguished from original authorship. [Alfonso Fortunato / @devopsfortunato](https://github.com/devopsfortunato) authored [the malformed GLM argument-boundary fix](https://github.com/ashhart/TensorFold/pull/2831); its exact original merge SHA is not invented. Their earlier-candidate roles are retained, not assertions that each old patch was recopied onto the native base.

</details>

## Concurrency and speed

Adding an RTX 5090 moves attention to a separate GPU while the two Sparks serve experts. The native hybrid exchange overlaps expert work; pinned host RAM holds the main BF16 KV history while pooled keys remain on the attention GPU. This is a heterogeneous system, not three identical tensor-parallel ranks. The measured benefit belongs to the complete configuration: no matched causal test isolates the card, extra RAM, engine version or drafter.

Same short harness, 300 generated tokens per stream, Q4 DFlash2, same RTX 5090 + two Sparks. The v1.3.3 runs are later than the v1.3.0 runs, not alternating A/B trials:

| Workload / metric | V4.2 / v1.3.3 | Prior V4.2 / v1.3.0 | Observed change |
|---|---:|---:|---:|
| C1 prose decode, tok/s | **65.5** | 63.9 | +2.5% |
| C1 code decode, tok/s | **70.9** | 69.4 | +2.2% |
| C1 JSON decode, tok/s | **81.9** | 80.6 | +1.6% |
| C4 mixed aggregate, tok/s | **94.9** | 93.0 | +2.0% |
| C8 mixed aggregate, tok/s | **122.2** | 118.4 | +3.2% |

C1 values are medians of three probes; C4 is the median of two waves, C8 one wave. Actual post-first-token per-stream ranges on v1.3.3: **24.3–28.9 tok/s at C4**, **15.7–18.6 at C8**. Aggregate throughput is output tokens divided by workload wall time; it is not an individual user's rate. Power limits and clock caps for these runs are **unknown**, so no stock-power or efficiency claim is made.

**Historical hardware comparison, not a new v1.3.3 head-to-head:** earlier same-window v1.3.0 AFD / Mia-recipe measurements were **93.0 / 77.0 tok/s at C4** and **118.4 / 85.1 at eight offered**. Mia's endpoint offered only four native lanes, so its eight-request result includes queueing. This does not isolate the benefit of adding a 5090 or RAM and is not a comparison to Mia's latest release. [Detailed results and methodology](BENCHMARKS.md) preserve the historical comparator revision and limitations.

## KV cache: RAM capacity, not eight full windows

| Verified v1.3.3 deployment property | Result |
|---|---:|
| Main history | **BF16, pinned host RAM** |
| Attached shared pool | **6,568 pages / 41,871 MiB host RAM** |
| Pooled keys on the attention GPU | **2,463 MiB** |
| Available request slots | **8 configured** |
| Exact cold C1 | **419,488 prompt + 512 output = 420,000 tokens** |
| Cold C1 TTFT / end-to-end | **314.266 / 319.209 s** |
| Eight simultaneously occupied 420K histories | **Not qualified** |

The main history uses host RAM rather than keeping all BF16 history on the 5090. The device still needs pooled keys, gather buffers, state, weights and workspace; host bytes minus key bytes is **not** a measured net VRAM saving. There is no host-KV on/off hardware control or measured causal speedup. Cap-plus-one was rejected before inference. A shared pool and eight configured slots are not eight independently backed full-size sessions.

A separate v1.3.3 context/concurrency matrix qualified **20 of 27 planned cells**: all short and ~13.6K-prompt cells, plus ~82.5K prose at C1/C4. Long-prose C8 completed only one of eight streams before a controller-local storage guard stopped the run; it is unqualified, **not a GPU OOM result**. Long code/JSON were not run. These are throughput/completion observations, not answer-quality results. [Matrix scope and latency](BENCHMARKS.md#extended-context-matrix--partial) matter for long-prompt use.

## Source scope and bounded reconstruction (no build)

This successor targets official [TensorFold v1.3.3](https://github.com/ashhart/TensorFold/tree/1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc). It reconstructs **144 affected source paths plus four grant/notice aliases**, not a complete engine, consumer build or deployment. It retains the public V4.2 hybrid implementation, Mia's selector guard, the DFlash2 admission port and the allocation repair, adapted to the v1.3.3 APIs; it adds the coupled exact-context request-opening/page-planner fixture. The 32-row verification width is retained.

Only source text, reconstruction metadata and notices ship. **No weights, drafters, compiled CUDA assets, executables, private configuration, credentials or raw benchmark records.** Obtain the complete pinned upstream separately, preserve its component terms and build with your own toolkit. This is not a turnkey launcher or a claim of native binary/bit parity.

Use Python 3.9 or later and Git on Linux in an isolated directory:

```sh
python3 -B source/reconstruct.py --fetch --cache ./input-cache --output ./overlay --receipt ./reconstruction.json
mkdir -p ./scratch
python3 -B source/test_reconstruct.py --cache ./input-cache --scratch ./scratch --receipt ./packaging-tests.json
```

The output must not exist. `--fetch` reads only exact immutable public URLs from the lock; the verified cache supports offline replay. Every output path, mode, size, Git blob and SHA-256 is checked. The ten packaging tests include both normal/restrictive umasks and eight refusal controls; they execute Python and Git, not a compiler, GPU, model or server. The selected source already contains the public comment/diagnostic and cache-label normalizations; reconstruction identity does not imply a new public-tree native build. The historical `hybrid-v1.3.0` cache namespace is intentionally retained with its coupled assertions; it is not the engine version pin.

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

Scott Quarrier directed the program. GPT-6.1 Sol / OpenAI Codex via Hermes prepared the V4.2 port, v1.3.3 API adaptations, source selection and this packaging/QA. Earlier local-agent and Anthropic Opus 5.5 roles remain in CREDITS and inherited notices; historical per-call authorship is not inferred from the current session. No model-quality claim is included.

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
