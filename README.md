# GLM-5.3-Flash AFD V4.1 — TensorFold v1.3.0 source overlay

**Source-only lab release. Provided as-is, without support. Not a turnkey deployment.**

## Authors

- [Ash Hart (@ashhart)](https://github.com/ashhart) authored [TensorFold v1.3.0](https://github.com/ashhart/TensorFold/tree/d3d0bf110d57161edfbd20234a84833f14548b57) and [MCDMA](https://github.com/ashhart/MCDMA). [Russell Marr (@russellmm)](https://github.com/russellmm) authored [pinned host KV](https://github.com/russellmm/TensorFold/commit/adc001b4bdf9fcdde38647ae5a46fbafd3e829f3); [Bizuayeu (@Bizuayeu)](https://github.com/Bizuayeu) authored the [MTP pooled-key fix](https://github.com/Bizuayeu/TensorFold/commit/b0e1cdcbab3b6ebc183129f9b26c9887ed22ff3e).
- [Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) authored the [Spark recipe and RoCE adaptations](https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104/tree/80be4d6882837d8e0840827ae2aab6124fd48da3), [EXL3 quantization](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) and [attribution guide](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution). [Z.ai](https://huggingface.co/zai-org) authored the [base model and MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda).
- [Hugh Madden (@hughmadden)](https://github.com/hughmadden) authored the [attention/expert split and expert prompt kernels](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6). [T.J. Purtell (@tpurtell)](https://github.com/tpurtell) authored the [expert channel split](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark/tree/dc6d9b8e1600e001cb1d4228bd911f4df8091f99) and [CuteAFD](https://github.com/tpurtell/cuteafd/tree/b4818a73371788aa3b9dd54023c017646d109f01).
- [Scott Quarrier (@squarrier)](https://github.com/squarrier) authored the recipe packaging and directed the program; an AI assistant authored the native integration and packaging changes for Scott. See [full credits](#license-and-credits), [AI-assisted development](#ai-assisted-development), and the [historical V4.0.1 source manifest](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/9e1590000788254ee4f827524781b0a06421cb84/manifests/engine-source.json).

<details>
<summary>More authors and source revisions</summary>

[taussoe / @taussoe](https://github.com/taussoe) authored [KDA work atfb985b86](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c). [Turboderp / @turboderp](https://github.com/turboderp) authored [ExLlamaV3/EXL3 atf45b8451](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b). [Z.ai](https://huggingface.co/zai-org) authored [GLM-5.3-Flash and built-in MTP](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda). [NVIDIA](https://github.com/NVIDIA) authored CUDA/libdevice; no compiled CUDA object is supplied here. [Scott Quarrier / @squarrier](https://github.com/squarrier) directed the recipe packaging; GPT-6.1 Sol via Hermes/openai-codex, an earlier local AI agent and Anthropic Opus 5.5 retain their distinct roles below. Exact per-file and previous-version authors remain in [CREDITS.md](CREDITS.md), [CREDIT-REGISTER.json](CREDIT-REGISTER.json) and every preserved notice.

[Raymond Lucke / @rwl4](https://github.com/rwl4) authored the [literal tool-closer cases at029a852f](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be). [Albert Lee / @bertholomus](https://github.com/bertholomus) authored [group-count/place and MMA work-list kernels](https://github.com/bertholomus/TensorFold/commit/90b68e063a0b8bba59aaf13340ef16619e751195). [SvangenStudios / @SvangenStudios](https://github.com/SvangenStudios) authored the [structured response-channel work in PR31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31), with the recorded Mia adapter at80be4d6 distinguished from original authorship. [Alfonso Fortunato / @devopsfortunato](https://github.com/devopsfortunato) authored [the malformed GLM argument-boundary fix](https://github.com/ashhart/TensorFold/pull/2831); its exact original merge SHA is not invented. Their earlier-candidate roles are retained, not assertions that each old patch was recopied onto the native base.

</details>

## Source scope, speed and BF16 KV

- The source input is official TensorFold v1.3.0 at `d3d0bf110d57161edfbd20234a84833f14548b57`; this capsule reconstructs 129 implementation paths plus two exact MIT grants and Mia's Apache-2.0 license/NOTICE (133 files). It is an affected-path overlay, **not the complete engine or a consumer build**.
- Compared with the privately tested source, this overlay changes cache-epoch labels in four files and removes local profile names from added comments. The comment edits do not change executable code. No compiler, weights, model configurations, private fixtures or binary assets are included.
- The prior bundled `engine/`, launch/build/download scripts, runtime configs, old source/asset manifests and executable tests are removed from this successor. Original public ancestry remains the parent; historical notices, names and original terms are deliberately retained. Historical credit registers and license indexes describe their original version, not a newly audited grant for this overlay.

## Verified lab configuration

The v1.3.0 lab hybrid ran attention on an RTX 5090 and routed experts on two DGX Sparks over RoCE. It used built-in MTP at confidence 0.50 with a 16-row attention tile. The table describes that lab configuration, not a new device run of this source overlay.

| V4.1 / v1.3.0 verification | Result |
|---|---|
| Authenticated chat | Passed a short generation check |
| Main BF16 KV cache | **7,496 pages (47,787 MiB) pinned in the attention host's system RAM** |
| Pooled index keys | 2,811 MiB on the attention GPU, separate from the host KV history |
| Advertised per-request limit | **479,232 tokens**, prompt plus output |
| Eight simultaneously occupied full windows | Not measured on v1.3.0; neither the allocation nor the advertised limit proves this capacity |
| Decode speed, extended soak and restart qualification | Not measured on v1.3.0 |

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

## Bounded reconstruction (no build)

Use Python 3 and Git in an isolated workspace. No Docker, GPU, model, service, network configuration or privileged install is involved. The optional `--fetch` reads only immutable public GitHub file URLs from the lock; subsequent replay can omit it and use the verified cache.

```sh
python3 -B source/reconstruct.py --fetch --cache ./input-cache --output ./overlay --receipt ./reconstruction.json
```

The output must not exist. The tool verifies every input digest, checks then applies the patch, attaches the four exact grants and verifies all 133 output paths, Git blob IDs, SHA256s, byte sizes and actual modes. It refuses extra paths or a changed patch. The raw cache contains only the 53 affected official base files and four public grant files. A full consumer must separately obtain the complete exact upstream tree, preserve its notices and dependencies, and undergo the supported native build and qualification. For the prior runnable public recipe and its separate device results, see [V4.0.1](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v4.0.1).

## Inherited GLM adaptations and redistribution terms

Patch context and deleted lines are redistributed source, not merely instructions. Keep all historical authors and terms in [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [CREDITS.md](CREDITS.md), [original base notices](source/upstream/) and the four [grant aliases](source/grants/LICENSES/). The two g53 MIT grants are not replaced by the newer TensorFold root Apache-2.0 text. Mia's Apache NOTICE is retained verbatim, including its nested terms; it is not a statement that this capsule ships her image or any weights.

The native GLM patch carries expression from pinned TensorFold v1.3.0. It retains TensorFold’s Apache-2.0 terms and the component-specific notices for the oMLX/mlx-vlm chain, including the historical oMLX Apache-2.0 text and the separate mlx-vlm and Apple MIT terms. The original oMLX extraction revision and exact path selection remain unresolved; the historical license snapshot is not presented as the import pin.

Shared GLM wrappers and tests contain DFlash interface code. The V4.1 overlay does not bundle the DFlash drafter implementation or any weights. Z Lab’s MIT attribution and [license text](LICENSES/z-lab-dflash-MIT.txt) are retained; that source license does not replace any checkpoint terms. Earlier public commits remain unchanged.

The [59-row provenance ledger](manifests/provenance.jsonl) retains historical records and the PicoJSON BSD-2-Clause vendor-custody entry. The raw checker reported JOIN PASS and six repository mappings without ledger rows. The owner approved disclosure of the unknown original revisions for this release rather than guessing them; this provenance decision is not a license grant.

## Provenance exceptions

- `github.com/ashhart/cortheon`: original source revision unknown.
- `github.com/descriptinc/descript-audio-codec`: original source revision unknown.
- `github.com/fla-org/flash-linear-attention`: original source revision unknown.
- `github.com/ggml-org/ggml`: original source revision unknown.
- `github.com/jundot/omlx`: original source revision unknown.
- `github.com/z-lab/dflash`: original source revision unknown.

Cortheon, DAC, FLA and ggml are historical or separately obtained upstream inputs; their direct implementations are not supplied by this overlay. The two GLM-related uses are described above. No unknown historical revision is inferred from a newer public snapshot.

## License and credits

All inherited author names, source grants and previous-version credits remain in the retained credit files. [Mia (@MiaAI_lab)](https://github.com/MiaAI-Lab) authored the attribution guide retained byte-identically in [AGENTS.md](AGENTS.md). The root LICENSE remains verbatim; original nested MIT, Apache-2.0 and other applicable terms are not relabeled.

## AI-assisted development

Scott Quarrier directed the program. GPT-6.1 Sol via Hermes/openai-codex prepared this offline packaging and bounded reconstruction test. The prior native-integration and local execution and Anthropic Opus 5.5 roles remain as recorded in the historical README and CREDITS; a current model label does not establish historical per-call model identity. This packaging work does not re-author inherited algorithms or claim a new hardware result.

## Support

None; provided as-is. This source overlay is not a qualified deployment.
