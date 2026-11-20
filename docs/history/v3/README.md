# GLM-5.3-Flash AFD: v3.0 native Zig AFD

Attention, KV state and MTP drafting on one RTX 5090; two DGX Sparks each compute half of every routed expert, over native RoCE v2. This updates the existing recipe, not a new repository.

Provided as-is, with no support. Source-only release candidate: the exact shipping head completed one 4.050086-hour active soak and its separate two-cycle restart test; original failures and desk dispositions remain disclosed below.

> **Limitations:** four users are slower than Mia's two-Spark lane and v2.15; single-stream code/JSON are slower than v2.15; first-token latency is higher than the two-Spark lane. C4 is 94.7 vs 100.5 total tok/s; C8 is 92.1 vs 101.0. Fresh prompt processing wins against the two-Spark lane, but not against historical v2.15.
> Schema-constrained JSON, forced tool calls and temperature > 0 do not draft; the decode cap is eight physical rows and four users, with no vision and a 262,144-token request limit.
> Source only; no libdevice or compiled objects are distributed. Read [Limitations](#limitations) and the [original failures and dispositions](docs/BENCHMARKS.md#gate-dispositions) before installation.

> **Where v2.15 is better:** its published lab run delivered 114.1 tok/s across four streams, 86.1 / 91.4 tok/s for one-stream code / JSON, fresh-prompt rates of 3,374–3,707 tok/s, and eight decoding requests. v3.0's filled corresponding cells are in [Where v2.15 is better](#where-v215-is-better); those are published v2.15 lab numbers, not the same window.

## Credits — before installation

Ash Hart ([ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart)) authored [TensorFold](https://github.com/ashhart/TensorFold/tree/f8fe17d24629aedabf90bbf78279dd776e6d62e7), its native engine and MTP machinery, and [MCDMA](https://github.com/ashhart/MCDMA). This snapshot contains his engine plus credited ports, not a replacement for his authorship.

Mia ([MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)) authored the [GLM recipe, CUDA preparation and drafter integration](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/549fdc56), the [EXL3 quantization at 078455ff](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c), this README's layout, and the [credit-and-attribution AGENTS.md](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), retained unchanged. Z.ai authored [GLM-5.3-Flash and its built-in MTP layer](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda). The pinned checkpoint's published LICENSE is MIT; later quantization revisions use Apache-2.0. No weights are bundled and no external DFlash2 checkpoint is used by v3.0.

Hugh Madden / Turquoise Bay AI ([hughmadden](https://github.com/hughmadden)) authored [glm53f-afd's split and expert prompt kernels](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6), including the tiered schedule; T.J. Purtell ([tpurtell](https://github.com/tpurtell)) authored [glmrt's expert-channel split](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark/tree/dc6d9b8), [ds41rt](https://github.com/tpurtell/ds41rt/tree/3067d06) and [cuteafd](https://github.com/tpurtell/cuteafd/tree/c05d9939); Hugh Madden authored cuteafd's graph-key design. Bizuayeu ([profile](https://github.com/Bizuayeu), [@Bizuayeu](https://x.com/Bizuayeu)) authored the [shared verify-window design](https://github.com/Bizuayeu/TensorFold/commit/75131b66c39fdcf55dbd130ccaec166b65394f3d) and [GLM key repair](https://github.com/Bizuayeu/TensorFold/commit/b952862585cf1bb564c297926c0072db2b48bc71), [end-token close](https://github.com/Bizuayeu/TensorFold/commit/557011805a64639ff5730aa980f7a100ab3ee7d9) and [Responses include fixes](https://github.com/Bizuayeu/TensorFold/commit/be17a101c47c6f20ac2295ea59922ad777d02252). Raymond Lucke ([rwl4](https://github.com/rwl4)) authored the [literal tool-closer design and cases](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/029a852f); Albert Lee / bertholomus ([profile](https://github.com/bertholomus)) authored the [group-count/place](https://github.com/bertholomus/TensorFold/commit/dbe0773) and [MMA work-list kernels](https://github.com/bertholomus/TensorFold/commit/90b68e063a0b8bba59aaf13340ef16619e751195). Turboderp ([profile](https://github.com/turboderp)) authored [ExLlamaV3's EXL3 format](https://github.com/turboderp-org/exllamav3/tree/f45b8451). NVIDIA authored CUDA and libdevice. The remaining authors and exact source revisions are credited immediately below and in NOTICE and THIRD_PARTY_NOTICES.md, including explicitly separate research-only and previous-version credits.

[Hugging Face tokenizers contributors](https://github.com/huggingface) authored [tokenizers](https://github.com/huggingface/tokenizers/tree/7f1623b90b5adfb9bc327d4c3468d2f70bbce262), used by qualification/preparation; Hugging Face contributors authored the [transformers GLM semantics](https://github.com/huggingface/transformers/tree/7cd73d9df0c14b151c684b708a9f27d8d0349dfe), a reference, not a newly vendored runtime. [linux-rdma/rdma-core contributors](https://github.com/linux-rdma/rdma-core/tree/v50.0), including Topspin Communications, Intel Corporation, Cisco Systems, Inc. and PathScale, Inc., authored the verbs components used as external build/runtime inputs under their individual retained terms. [Zig contributors](https://github.com/ziglang/zig) authored the pinned 0.17.0 compiler and standard library (MIT). Their full notices and exact scope remain in NOTICE and THIRD_PARTY_NOTICES.md; no generated library/compiler is bundled.

Separately studied speculative-sampling designs are not active or adopted in this shipped lane: Yaniv Leviathan, Matan Kalman and Yossi Matias authored [speculative acceptance/residual correction](https://arxiv.org/abs/2211.17192v2); Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper authored [speculative sampling correction](https://arxiv.org/abs/2302.01318v1). Those paper/design credits remain research scope, not newly imported code. Opus 5.5 via Hermes authored retained recorder/CPU-fixture work; GPT-6.1 Sol via Hermes authored the subsequent qualification and release fill-in. Scott Quarrier ([@squarrier](https://github.com/squarrier)) authored the recipe packaging and directed the work, with the AI assistance disclosed below.

### Every ledger author, with scope kept explicit

This list preserves all recorded authors, including inherited backends, benchmark references and verification-only sources. It does not claim every referenced project is a serving dependency. Exact authored work, repositories, revisions and licences follow in the source annex and NOTICE.

- [Ash Hart](https://github.com/ashhart) (@ashhart): [TensorFold @ f8fe17d2](https://github.com/ashhart/TensorFold/tree/f8fe17d24629aedabf90bbf78279dd776e6d62e7) — native source, reimplemented design or build/runtime dependency.
- [Mia](https://github.com/MiaAI-Lab) (@MiaAI_lab): [TensorFold @ b23c10ad](https://github.com/ashhart/TensorFold/tree/b23c10ad305aa068e12b901ee8ebd8166b9aa6fd) — native source, reimplemented design or build/runtime dependency.
- [Mia](https://huggingface.co/Mia-AiLab) (@MiaAI_lab): [GLM-5.3-Flash-EXL3-4bpw-TensorFold @ 078455ff](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) — pinned runtime weights, not shipped.
- [AEON-7](https://github.com/AEON-7) (@AEON-7): [Aeon-Bench-Pod @ 8651bcb5](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) — research/benchmark reference only; no code taken for v3.0.
- [Andrey Kolesnikov](https://github.com/akol1) (@akol1): [TensorFold @ 86134886](https://github.com/ashhart/TensorFold/tree/861348862a0f108027575c6e529e527e0bfe754d) — native source, reimplemented design or build/runtime dependency.
- [Apple Inc.](https://github.com/ml-explore) (ml-explore): [mlx @ v0.32.3](https://github.com/ml-explore/mlx/tree/v0.32.3) — native source, reimplemented design or build/runtime dependency.
- [Ashish Kulkarni](https://github.com/py-pdf/pypdf): [pypdf @ 6.20.0](https://github.com/py-pdf/pypdf/tree/6.20.0) — verification-only dependency; not used or shipped by v3.0.
- [avlp12](https://github.com/avlp12): [mlx-vlm @ referenc](https://github.com/Blaizzy/mlx-vlm) — research/benchmark reference only; no code taken for v3.0.
- [BadAd84](https://github.com/BadAd84) (@BadAd84): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 90ccb492](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) — native source, reimplemented design or build/runtime dependency.
- [bertholomus](https://github.com/bertholomus) (@bertholomus): [TensorFold @ dbe0773](https://github.com/bertholomus/TensorFold/tree/dbe0773) — native source, reimplemented design or build/runtime dependency.
- [Bizuayeu](https://github.com/Bizuayeu) (@Bizuayeu): [TensorFold @ 75131b66](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) — native source, reimplemented design or build/runtime dependency.
- [BlivionIaG](https://github.com/BlivionIaG) (@BlivionIaG): [TensorFold @ d7b134cd](https://github.com/ashhart/TensorFold/tree/d7b134cdf86a02b7c91d0d137ce733c04ae53cfd) — native source, reimplemented design or build/runtime dependency.
- [BobClawblaw](https://github.com/BobClawblaw) (@BobClawblaw): [TensorFold @ 8ced2018](https://github.com/ashhart/TensorFold/tree/8ced20188b70fe301d4c9f320430b182a68d216f) — native source, reimplemented design or build/runtime dependency.
- [chaog992](https://github.com/chaog992) (@chaog992): [TensorFold @ 2193d145](https://github.com/ashhart/TensorFold/tree/2193d145397500f7b661dd4697240024e1089b68) — native source, reimplemented design or build/runtime dependency.
- [data-angel](https://github.com/data-angel) (@data-angel): [sparkglm @ PR #46](https://github.com/Enntity/sparkglm/pull/46) — research/benchmark reference only; no code taken for v3.0.
- [DLPack contributors](https://github.com/dmlc) (dmlc): [dlpack @ bbd2f4d3](https://github.com/dmlc/dlpack/tree/bbd2f4d32427e548797929af08cfe2a9cbb3cf12) — native source, reimplemented design or build/runtime dependency.
- [Enntity](https://github.com/Enntity) (@Enntity): [atlas @ ade7b012](https://github.com/Enntity/atlas/tree/ade7b012) — research/benchmark reference only; no code taken for v3.0.
- [ezoushen](https://github.com/ezoushen) (@ezoushen): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 3fd60dfc](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab) — native source, reimplemented design or build/runtime dependency.
- [fly9i](https://github.com/fly9i) (@fly9i): [spark-engine @ b09a53e](https://github.com/fly9i/spark-engine/tree/b09a53e) — research/benchmark reference only; no code taken for v3.0.
- [Hugh Madden](https://github.com/hughmadden) (@dangerm00se): [glm53f-afd @ 91db3cc6](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) — native source, reimplemented design or build/runtime dependency.
- [iegamo](https://github.com/ahinore) (@ahinore): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 2636c1cb](https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/2636c1cb3019ede4ab27191299d5ad6843344942) — research/benchmark reference only; no code taken for v3.0.
- [Jay Leaton](https://github.com/jayleaton) (@jayleaton): [TensorFold @ 4284048d](https://github.com/ashhart/TensorFold/tree/4284048d1aca2799e543a7400b3f07519befb515) — native source, reimplemented design or build/runtime dependency.
- [jundot](https://github.com/jundot) (@jundot): [omlx @ referenc](https://github.com/jundot/omlx) — research/benchmark reference only; no code taken for v3.0.
- [Jürgen Schmied](https://github.com/jschmied) (@jschmied): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 60626dc1](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) — research/benchmark reference only; no code taken for v3.0.
- [Kazuho Oku](https://github.com/kazuho) (@kazuho): [picojson @ mlc-ai/x](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa/3rdparty/picojson) — native source, reimplemented design or build/runtime dependency.
- [kky42](https://github.com/kky42) (@kky42): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ c66a6cfd](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/c66a6cfd9ae16226b30f29df877b0194ae8b3945) — native source, reimplemented design or build/runtime dependency.
- [Lazarus-931](https://github.com/Lazarus-931): [mlx-vlm @ referenc](https://github.com/Blaizzy/mlx-vlm) — research/benchmark reference only; no code taken for v3.0.
- [local-inference-lab](https://github.com/local-inference-lab) (@local-inference-lab): [b12x @ 2cc7f66a](https://github.com/local-inference-lab/b12x/tree/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88) — native source, reimplemented design or build/runtime dependency.
- [local-inference-lab contributors](https://github.com/local-inference-lab) (@local-inference-lab): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 60626dc1](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) — research/benchmark reference only; no code taken for v3.0.
- [Mathieu Fenniak](https://github.com/py-pdf/pypdf): [pypdf @ 6.20.0](https://github.com/py-pdf/pypdf/tree/6.20.0) — verification-only dependency; not used or shipped by v3.0.
- [mikolaj92](https://github.com/mikolaj92) (@mikolaj92): [TensorFold @ b3b8a391](https://github.com/ashhart/TensorFold/tree/b3b8a39119f8da02026b3f2601e02eb411be1339) — native source, reimplemented design or build/runtime dependency.
- [Prince Canuma](https://github.com/Blaizzy) (@Blaizzy): [mlx-vlm @ referenc](https://github.com/Blaizzy/mlx-vlm) — research/benchmark reference only; no code taken for v3.0.
- [PyTorch contributors](https://github.com/pytorch) (pytorch): [pytorch @ 9186a08b](https://github.com/pytorch/pytorch/tree/9186a08b2c) — native source, reimplemented design or build/runtime dependency.
- [raullenchai](https://github.com/raullenchai): [mlx-vlm @ referenc](https://github.com/Blaizzy/mlx-vlm) — research/benchmark reference only; no code taken for v3.0.
- [Raymond Lucke](https://github.com/rwl4) (@rwl4): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 029a852f](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be) — native source, reimplemented design or build/runtime dependency.
- [SparkInfer/b12x contributors](https://github.com/tpurtell/sparkinfer-glmrt): [sparkinfer-glmrt @ dc9c2619](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) — research/benchmark reference only; no code taken for v3.0.
- [Steve Witham](https://github.com/py-pdf/pypdf): [pypdf @ 6.20.0](https://github.com/py-pdf/pypdf/tree/6.20.0) — verification-only dependency; not used or shipped by v3.0.
- [stevehansen](https://github.com/stevehansen) (@stevehansen): [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold @ 90ccb492](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) — native source, reimplemented design or build/runtime dependency.
- [T.J. Purtell](https://github.com/tpurtell) (@wrldsuksgo2mars): [glm53f-afd @ 91db3cc6](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) — native source, reimplemented design or build/runtime dependency.
- [taussoe](https://github.com/taussoe) (@taussoe): [TensorFold @ fb985b86](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c) — native source, reimplemented design or build/runtime dependency.
- [tonyd2wild](https://github.com/tonyd2wild) (@tonyd2wild): [GLM-5.3-Flash-TensorFold-DGX-Spark @ 062b6ce1](https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark/tree/062b6ce13e0fa9bead00f58d1cb0cfec7bcc080b) — research/benchmark reference only; no code taken for v3.0.
- [Triton contributors](https://github.com/triton-lang) (triton-lang): [triton @ v3.7.1](https://github.com/triton-lang/triton/tree/v3.7.1) — native source, reimplemented design or build/runtime dependency.
- [Turboderp](https://github.com/turboderp) (@turboderp): [exllamav3 @ f45b8451](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b) — native source, reimplemented design or build/runtime dependency.
- [XGrammar Contributors](https://github.com/mlc-ai) (mlc-ai): [xgrammar @ 97787376](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) — native source, reimplemented design or build/runtime dependency.
- [XGrammar contributors (MLC)](https://github.com/mlc-ai/xgrammar): [xgrammar @ 97787376](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) — research/benchmark reference only; no code taken for v3.0.
- [Z Lab](https://github.com/z-lab) (z-lab): [dflash @ referenc](https://github.com/z-lab/dflash) — research/benchmark reference only; no code taken for v3.0.
- [Z.ai](https://huggingface.co/zai-org): [GLM-5.3-Flash-EXL3-4bpw-TensorFold @ 078455ff](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) — pinned runtime weights, not shipped.
- [Z.ai (Zhipu AI)](https://huggingface.co/zai-org) (zai-org): [GLM-5.3-Flash @ tokenize](https://huggingface.co/zai-org/GLM-5.3-Flash) — research/benchmark reference only; no code taken for v3.0.

### Previous-version first-screen credits, retained

The following is v2.15's first-screen credit block, verbatim. In this retained block, "here", "this version", the weights/drafter conditions and script names describe v2.15, not v3.0; its links remain as published. Current v3.0 authors are credited above.

| Author | Profiles | What they authored, as used here | Repo, commit |
|---|---|---|---|
| **Mia** (Mia's AI Lab) | [MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab) | the EXL3 4-bpw checkpoint; her TensorFold pull requests #243, #285 and #301 (patches 0002-0007, her commits); the 26 patches here that port her recipe's patches (among them her EXL3 prompt kernel, 0004; her attention-side prompt kernels, 0004, 0009 and 0028; eight streams, 0069; her v1.7.1 and v1.8 fixes, 0071-0083), the code of hers that patches 0025, 0029 and 0033 change (her 0030, 0049 and 0062), and the parts of her 0009 and 0020 that patch 0034 reimplements; [AGENTS.md](AGENTS.md); this README's layout | [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at [`cf28cc4`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/cf28cc4f8038be322cdeda220c6f1c8ace8f27d1) (v1.4), [`1576746`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) (v1.5), [`68ebd67`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/68ebd67b5326974b8004009e202268b1fa7c551d) (v1.7.1) and [`33b50fd`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/33b50fde06fd7ea604cbc6a663880068ab1e2ee4) (v1.8); [the checkpoint](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) at [`76c0b517`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/76c0b5173166d2795dd48860f45d8224817f894c) |
| **Ash Hart** | [ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart) | TensorFold, the engine, and two of TensorFold v0.6.6's commits (patches 0047 and 0048, his commits); TensorFold's MIT code that Hugh Madden's expert prompt kernels carry (the EXL3 decoder, fragment MMA and Hadamard butterfly, patch 0034); MCDMA, the RDMA wire | [TensorFold](https://github.com/ashhart/TensorFold) at [`609ca41`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f) (v0.6.5), [`cb2ebf0`](https://github.com/ashhart/TensorFold/tree/cb2ebf0540f42604e2759b2ddef497861e928248) (v0.6.6) and `bb4b4a3` (v0.3.4.1, MIT); [MCDMA](https://github.com/ashhart/MCDMA) at [`e672c14`](https://github.com/ashhart/MCDMA/tree/e672c14ff9fc7b38994caf73025cf1588b4de74e) |
| **Hugh Madden** / Turquoise Bay AI | [hughmadden](https://github.com/hughmadden), [@dangerm00se](https://x.com/dangerm00se) | glm53f-afd, the engine of v1.0: GLM-5.3-Flash with attention on an RTX 5090 and the routed experts on DGX Sparks, the split this version runs on TensorFold; his expert prompt kernels (glm53f-rank's large-M EXL3 kernels), whose code patches 0034 and 0035 carry for the Sparks' prompt chunks (on), and glm53f-rank's own schedule for its biggest windows, which patches 0049 and 0051 run on the Sparks' prompt windows of 1,536 rows and more (on); the designs patches 0026, 0027, 0032 and 0050 write for it: the exchanges kept in flight ahead of the expert ranks (on), the host RAM tier for kept prompts (on), the two- and four-lane prefill (four lanes, on), BF16 partial sums on the return wire (his BF16 return planes, on) and the FP8 wire rows (off) | [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at [`91db3cc`](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) (v1.1.0); [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) at [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79) |
| **T.J. Purtell** | [tpurtell](https://github.com/tpurtell), [@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars) | ds41rt, glmrt and cuteafd: engines that run attention on RTX GPUs and the routed experts on DGX Sparks; DS41RT's wire-row format, which patch 0026's FP8 wire rows use (off); glmrt's split of every expert by intermediate channel, which Hugh Madden's expert prompt kernels follow (patch 0034); ds41rt's host cache eviction design, in glm53f-afd's host RAM tier (patch 0032) | [ds41rt](https://github.com/tpurtell/ds41rt) at `3067d06`, [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) at `dc6d9b8`, [cuteafd](https://github.com/tpurtell/cuteafd) at `c650feb` |
| **E-Zou Shen** | [ezoushen](https://github.com/ezoushen), [@ezoushen](https://x.com/ezoushen) | her recipe patches 0071 (a resumed shared prefix copies its rows) and 0074 (the pool compacts before it evicts), contributed to her recipe; here 0036, 0038 and their tests in 0041 (off) | her recipe's pull requests [#44](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/44) and [#62](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/62), at `68ebd67` (v1.7.1) |
| **desy0305** | [desy0305](https://github.com/desy0305) | her recipe patch 0073 (waiting requests whose client left are dropped at once) and its checks; here 0037 (off) | her recipe's pull request [#51](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/51), at `68ebd67` (v1.7.1) |
| **johnwhited** | [johnwhited](https://github.com/johnwhited) | her recipe patches 0081 (capacity refusals answer 429 with Retry-After), 0082 (admission at saturation) and 0083 (a stream whose delivery fails ends at once), and the delivery-failure handling in 0073; here 0043 and 0045 (on), 0044 (off, her default) and 0037 | her recipe's pull request [#48](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/48), at `33b50fd` (v1.8) |
| **m-naoki-m** | [m-naoki-m](https://github.com/m-naoki-m), [@\_m\_naoki\_m\_](https://x.com/_m_naoki_m_) | her recipe patch 0078 (a take-over decides which kept prompts stay before it copies any); here 0042 (on) | her recipe's pull request [#71](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/71), at `33b50fd` (v1.8); also [TensorFold#421](https://github.com/ashhart/TensorFold/pull/421) |
| **Philip Mossop** | [philip-pentatonic](https://github.com/philip-pentatonic) | TensorFold v0.6.6's `--name-priority ID=background` (patch 0046, his commit) | [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445), [`dce62cf`](https://github.com/ashhart/TensorFold/commit/dce62cfa3c6752c828c4a17338d3ae585e43e65a) |
| **Jay Leaton** | [jayleaton](https://github.com/jayleaton), [@jayleaton](https://x.com/jayleaton) | code Mia adapted from his patches 0580 (the experts' decode loads, in patch 0017) and 0620 (tool calls, in patch 0020) | [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4) |
| **Z.ai** | [zai-org](https://huggingface.co/zai-org), [@Zai_org](https://x.com/Zai_org) | GLM-5.3-Flash, the model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at [`eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) |
| **turboderp** | [turboderp](https://github.com/turboderp), [@turboderp_](https://x.com/turboderp_) | ExLlamaV3, which made Mia's checkpoint in its EXL3 format | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| **IncoAI** | [incoai](https://huggingface.co/incoai) | the DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at [`bf582e4e`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) |

Also credited in [License and credits](#license-and-credits) and [NOTICE.md](NOTICE.md): meleesciony ([@meleesciony](https://github.com/meleesciony)), who diagnosed the kept-cap failure of her recipe's [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75) and proposed the fix her patch 0077 makes (at `68ebd67`, v1.7.1; here 0040, 0041); Lukas-tek-no-logic ([@Lukas-tek-no-logic](https://github.com/Lukas-tek-no-logic)), who reported the `<|assistant|>` leak her patch 0075 fixes ([issue #60](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/60), at `68ebd67`; here 0039); Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp)) for [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5), Patryk Mikołajczyk ([@mikolaj92](https://github.com/mikolaj92)) and Aditya Thyagarajan ([@aditya1503](https://github.com/aditya1503)) for TensorFold #140 and #200, fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention), and every credit of v1.0, unchanged.

> **Weights and licences (read first):** the checkpoint is [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) by **Mia's AI Lab**, a quantization licensed **Apache-2.0**, derived from Z.AI's GLM-5.3-Flash (**MIT**). The drafter, [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), is **CC BY-NC-ND 4.0: non-commercial use only, no derivatives**. This repo ships no weights: `download.sh` fetches both from their own repos. See [NOTICE.md](NOTICE.md).

> **Provided as-is, with no support.** This is a personal homelab recipe, shared in case it helps someone. Issues and PRs may go unanswered. See [Support](#support).


| I want to… | Go to |
|---|---|
| Compare measured results | [Performance](#performance) and [method](docs/BENCHMARKS.md) |
| Run the native lane | [Quick start](#quick-start) and [build/run](docs/RUN.md) |
| Understand the split | [Design](docs/DESIGN.md) |
| Lock down the hosts | [Security](#security) |
| Use the older recipes | [v2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0), [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) |

## What's new in v3.0

- Native Zig TensorFold AFD snapshot at source revision `1fab9035bb24bb3d647103c7d4285bb230a9a52e`, replacing the published v2.15 Python-engine patch recipe. Public packaging changes only identifiers, documentation, configuration and test/example addresses, not model arithmetic or CUDA kernel implementations.
- Four independent users, a shared attention-state pool, and prefill chunks interleaved with other users' decode rounds. Each request is capped at 262,144 tokens; the requested shared pool is 720,896 tokens and its allocated capacity is 720,888, not four maximum-length requests.
- The checkpoint's built-in MTP drafts, with per-user acceptance/rollback. Short draft-eligible requests fit a 2,051-position draft window; longer requests continue undrafted. No external DFlash2 weight dependency.
- Grammar-constrained output and seeded sampling with multiple users, using a separately built XGrammar library and a digest-checked trusted loader. Sampled and structured users use one-row windows rather than speculative shared windows.
- One shared forward across users and drafted shared verification are included, but the qualified shipping selection is ON; batched MTP drafts and ping-pong ON; all four flags are 1. The example explicitly selects this measured configuration; the vendored native-source defaults remain OFF. Batched MTP computes users together at each draft depth; ping-pong plans up to 16 logical rows as two physical micros of at most eight rows, not 16 physical rows.
- Tool-call boundary repair for literal `</tool_call>` inside values, malformed-key repair, end-token value close and OpenAI Responses `include` validation. The earlier unconstrained exact-copy probe missed its literal-closer premise; the final fixed whole/streamed probe passed on both boots, without a waiver.
- Native transport embedded in the engine; no separate Python process or external MCDMA daemon is launched by this recipe. Experimental fingerprinting, graphs, stripes and FP8 MTP remain disabled.

## Upstream status

On 2026-10-10 Ash Hart [closed the original proposal](https://github.com/ashhart/TensorFold/issues/214#issuecomment-6096480459) in favour of [the native CUDA GLM tracker](https://github.com/ashhart/TensorFold/issues/593). That tracker puts attention on a desktop GPU with routed experts on the Sparks at step 6, after the single-GPU GLM-5.3-Flash family and TP=2/3. v3.0 is an out-of-tree implementation of that layout on a fork, built on the GLM port and kernels credited above; it is not upstream TensorFold. The previous-version replies are retained as historical text below, not an open location decision.

## At a glance

| Item | v3.0 candidate |
|---|---|
| Attention | x86_64 Linux, RTX 5090 32 GB, dense q4 / FP8 head, native CUDA |
| Experts | Two GB10 expert halves, EXL3, aarch64 Linux |
| Wire | Native RoCE v2 exchange, isolated fabric required |
| Model | Mia's GLM-5.3-Flash EXL3 `078455ff`, pinned in `.env.example` |
| API | OpenAI-compatible chat/completions and Responses; configurable model alias |
| Context / users | 262,144 tokens/request; four users share one 720,888-token pool |
| Drafts | Built-in MTP, `auto`, confidence stop `c3:0.85` |
| Constraints / sampling | XGrammar v0.2.8; per-request mask and keyed sampling |
| Binaries / weights | Not distributed; build and prepare locally |

## Performance

### Final same-window comparison — 2026-10-15

Three comparison blocks pause the soak load: after warm-up, mid-soak and before close. Each uses the same 300-token C1/C4 harness and prompts, alternating this three-node lane → Mia's live two-Spark lane → this lane. Complete pause/drain/block/resume time is excluded from the active soak. The headline is the pre-registered before-close after-reference arm, median of three clean repetitions; all earlier brackets are published below. JSON here is plain prompted JSON, not schema enforcement. Image readback limitations are stated in the table.

| Metric | v3.0 three-node lane | Mia's two-Spark lane |
|---|---|---|
| C1 prose decode, tok/s | 74.5 | 59.5 |
| C1 code decode, tok/s | 71.6 | 67.0 |
| C1 JSON decode, tok/s | 79.3 | 67.8 |
| C1 prose TTFT, ms | 162 | 148 |
| C1 code TTFT, ms | 193 | 179 |
| C1 JSON TTFT, ms | 225 | 206 |
| C4 aggregate, tok/s | 94.7 | 100.5 |
| C4 per-request decode, tok/s | 24.3–26.3 (all 12 streams) | 25.8–30.4 (all 12 streams) |
| C4 TTFT, ms | 577 | 385 |
| Request errors / contamination | 0 request errors; CLEAN, no overlap/retries (all three blocks) | 0 request errors; CLEAN, no overlap/retries (all three blocks) |
| Engine / recipe / image pins | source pin above | Currency PASS: recipe `b5808ac6` → `3fd60dfc` changes README only; running model HTTP identity retained. HTTP readback identifies TensorFold, four decode slots, 1,796,096 pool tokens and a 262,144 context. Exact running engine/checkpoint/drafter revision, dense precision and KV dtype were not emitted by the saved HTTP identity; they remain NOT MEASURED, not inferred from recipe currency. Fresh container-image digest NOT MEASURED by HTTP; no historical digest substituted |

C1 prose/code/JSON decode is 25.21%/6.87%/16.96% higher than the reference, but C4 total is 5.77% lower and C1/C4 first-token latency is worse. These are per-metric results, not an overall speed claim. [Detailed comparison blocks and limitations](docs/BENCHMARKS.md).

### Prompt processing and two/eight offered requests — 2026-10-15

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

C8 is eight offered HTTP requests on the unchanged four-user service, not eight resident decode slots. Queueing remains in the first-token times: some v3.0 requests waited up to 18.470 s. Fixed concurrency prompts are not a cache-neutral/fresh-prefill claim. The two-Spark reference needs one fewer host and no RTX 5090, supports vision and retains its own serving/cache policy. [All takes, stream rates and TTFT](docs/BENCHMARKS.md#prompt-and-concurrency-follow-up).

### Lab numbers, 2026-10-13 — not the final soak comparison

All values below come from the recorded 04:24–07:07 EDT native-head gate window, with users 4, pool/interleave, grammar/sampling and drafts auto. “Shared ON” was faster but was not adopted then: the tool-copy gate failed. Group-share is aggregate/concurrency, not a separately measured per-request rate. Request-wall mean measures each user's tokens/wall; decode mean excludes TTFT and the first token. Do not combine those denominators.

| Cell / metric | Shared OFF | Shared ON, experimental |
|---|---:|---:|
| C1 drafted aggregate, tok/s | 72.24 | 71.97 |
| C2 undrafted aggregate, tok/s | 59.36 | 66.50 |
| C2 request-wall mean, tok/s | 29.69 | 33.30 |
| C4 drafted aggregate, tok/s | 61.91 | 79.70 |
| C4 group-share, tok/s | 15.48 | 19.92 |
| C4 request-wall mean, tok/s | 15.64 | 22.62 |
| C4 decode mean, tok/s | 16.03 | 23.56 |
| C4 mean TTFT, s | 0.469 | 0.489 |

Shared-ON C4 was 1.287× Shared-OFF in that lab harness. AEON-30 on the Shared-ON boot was 23/30, matching the recorded reference case-for-case; six empty outputs and one wrong answer remain. Structured users passed 36/36; seeded users 8/8. Tool probes were 9/11, with both unconstrained literal-copy attempts failing because the model never generated the closer. These are lab receipts, not a passed final soak, service test, same-window result against the live reference, or a claim that all clients' tool cases pass.

### Soak and restart qualification — same shipping head

| Requirement | Result |
|---|---|
| Fixed literal-closer whole/streamed premise and exactness | PASS: full tools 11/11 first after health on both boots; whole/streamed fixed probe 1,539 tokens, exact text and closer, no premise waiver |
| Adopted flags / head | `1fab9035bb24bb3d647103c7d4285bb230a9a52e`; four users, shared/shared-drafts/batched-drafts/ping-pong ON; requested/allocated pool 720,896/720,888; interleave 1; grammar/sampling ON; schema/sampled drafts OFF |
| Active mixed-load duration ≥4 h, excluding comparison blocks | PASS: 4.050086 active h; 2844.353 s full pauses excluded |
| HTTP errors / restarts / PoolFull | PASS: 9,657/9,657 checked requests; 0 HTTP 5xx, 0 restarts, 0 PoolFull. 105 non-gating observations and 208 expected HTTP 400 negative controls retained |
| Exact / tool / structured correctness | PASS: tool 1,693/1,693; fixed boundary 105/105 whole+streamed; API negative/repair controls 520/520; structured 4,444/4,444; sampled 2,965/2,965 |
| Decode drift / seeded repeatability | PASS: greedy 18 probes; seeded 17/17 equal; closing rebench 39/39 replies equal on unchanged harness |
| Memory creep ≤2 GiB/h; loaded floors | PASS: attention/expert0/expert1 creep 1.691854/-0.015130/-0.019618 GiB/active h; minimum available 166.428/28.740/28.245 GiB, above loaded 24/12/12 GiB floors. Attention 1.692 is near the 2 GiB/h limit, not zero creep |
| Peak temperatures / cooldowns | PASS: attention/expert0/expert1 peaks 71.0/87.7/88.9 °C; 0 heat holds; Sparks recorded at 2,200 MHz; unchanged 92/90 °C hold/resume bands |
| ~240K-token request under concurrent load | PASS: 239,005 actual prompt tokens; exact needle HTTP 200; four completed mixed requests overlap the send-to-response interval |
| AEON-30 under soak load | PASS: 23/30, all 30 scores and empty/wrong classes equal to the recorded reference; six empty, one wrong; temperature 0, 2,048 tokens then one 8,192-token retry |
| Owner-bound orderly close, native exits and physical vacancy | PASS: both orderly closes; six native and both outer exits 0; no holders/locks. Original automatic CPU finalizer exited 143/OOM after GPU close, not clean 0; authorized file-only reporting recovery retained |
| Two start/restart/stop cycles on the exact candidate | PASS 2/2 on the same shipping head/M-on profile; cold 197.586 s, graceful restart 145.404 s, final stop 12.144 s |
| Native/canonical aliases, watchdog, exact exits and physical readback | PASS: both aliases and 3/3 canaries per cycle; original 900 s startup grace; loaded 6/6, startup bindings 2/2, native exits 6/6 = 0, physical 27/27; outer/observer exits 0; all owners absent |

The restart cells are supplied by the separate 2026-10-15 shipping-configuration service test, not inferred from the soak. The older 2026-10-13 Shared-OFF result remains historical only. Both final tests cover the same 1fab9035 head and explicit shared/batched/ping-pong ON profile; this text is not a production cutover or recovery guarantee.

## Where v2.15 is better

These are published v2.15 lab numbers, not the same window. v2.15 used its Python TensorFold recipe, external DFlash2 and its own request lengths; the v3.0 soak uses the 300-token comparison harness. The C2/C8 cells are not the older 1,024-token coding-agent workload. No cross-day ratio is a same-window speedup, and TTFT is not prompt tok/s.

| Metric | v2.15, published lab | v3.0 three-node lane | Mia's two-Spark lane, same-window within each measurement family | v2.15 source |
|---|---|---|---|---|
| One stream prose / code / JSON, decode tok/s | 71.3 / 86.1 / 91.4 | 74.5 / 71.6 / 79.3 | 59.5 / 67.0 / 67.8 | [README.md:77](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L77) |
| Four streams, total tok/s | 114.1 | 94.7; actual stream decode 24.3–26.3 (all 12 streams) | 100.5; actual stream decode 25.8–30.4 (all 12 streams) | [README.md:78](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L78) |
| spark-bench ~2.6K-token prompt, tok/s | 2,420 | No matching v3.0 prompt-tok/s cell is planned; the separate cold first-token cell below is 1.26 s | Separate cold first-token cell 1.59 s, not tok/s | [README.md:79](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L79), [docs/BENCHMARKS.md:81](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L81) |
| Fresh 8K / 31K / 62K prompts, tok/s | 3,374 / 3,642 / 3,707 | 3111.49 / 3198.99 / 3180.87 | 1935.58 / 1996.25 / 1998.65 | [README.md:81](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L81), [docs/BENCHMARKS.md:83](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L83) |
| Cold ~2.6K prompt, first token, s | 1.058 | 1.26 | 1.59 | [README.md:80](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L80), [docs/BENCHMARKS.md:82](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L82) |
| Cold 100K prompt, first token, s | 27.1 | 32.47 | 50.45 | [README.md:82](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L82) |
| Coding agents, 4 x 1,024 output tokens, total tok/s | 255.8 | No matching 1,024-token workload measured; four-stream soak harness 94.7 | Four-stream soak harness 100.5 | [README.md:83](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L83) |
| Coding agents, 8 x 1,024 output tokens, total tok/s | 421.7 | No matching 1,024-token workload measured; separate C8 total 92.1, actual stream range 17.3–39.5 | Separate C8 total 101.0, actual stream range 23.3–30.6 | [README.md:83](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L83) |
| Two streams, total tok/s and actual stream range | 94.1 total; no per-stream range published in that table | 87.4; 45.0–47.7 | 80.2; 41.4–46.8 | [docs/BENCHMARKS.md:79](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/docs/BENCHMARKS.md#L79) |
| Requests decoding together | 8; further requests queue | 4 users; C8 clients do not create eight decode slots: 92.1, 17.3–39.5 | 101.0, 23.3–30.6 | [README.md:133](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L133) |

v2.15 also kept a 1,585,152-token pool, versus this four-user snapshot's allocated 720,888. Its watcher and four failure drills were exercised, but its soak failed its probe check: 58/60, with 6.3 s / 7.0 s streamed gaps against a 6 s limit and one 60-minute probe gap against 45 minutes. Those failures are not removed by retaining the tag. See [v2.15's README:218–243](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md#L218-L243).

The [retained v2.15 comparisons](#retained-v215-comparisons) keep its own words about where v1.0 and Mia's lane were better, including the date and version of that earlier comparison. They are historical claims, not the current same-window comparison. Run [v2.15](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.15), [v2.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0) or [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) when its measured trade-offs fit the workload; each tag keeps its licence and credits.

## Hardware and power settings

The 2026-10-13 lab window used the RTX 5090 at 400 W of its stock 575 W, and both Sparks capped at 2,200 MHz of stock 3,003 MHz. Their boots predated 2026-10-10 10:25. A 2,100 MHz next-boot cap was configured on 2026-10-10; it does not relabel these as 2,100 MHz measurements. Final-soak settings and dates: 2026-10-15 soak: RTX 5090 400 W of 575 W, boot 2026-09-27 14:50:08; expert0/expert1 Sparks 2,200 MHz of 3,003 MHz under the unchanged-boot rule, boots 2026-09-30 00:31:55 / 2026-09-25 13:30:52. Next-boot 2,100 MHz configured 2026-10-10, not applied to these runs. Spark power limits and reference-node clock/boot readback NOT MEASURED. This source recipe does not set power limits, clock caps or reboot policies.

Attention needs x86_64 Linux, 32-GB RTX 5090, the CUDA driver and a RoCE-capable NIC. Both expert halves need DGX Spark/GB10, aarch64 Linux and a RoCE NIC. The measured admission floors were 200 GiB host MemAvailable on attention and 100 GiB per Spark; loaded floors were 24/12/12 GiB. Four users plus grammar/sampling have limited attention-memory headroom; use a single GPU tenant per host and an active host-memory-floor guard. CUDA unified allocations on GB10 bypass container accounting, so container memory limits alone are not protection.

## Limitations

1. **Four users:** this lane is slower than Mia's two-Spark lane in the comparison window: 94.7 vs 100.5 total tok/s, actual stream decode 24.3–26.3 (all 12 streams) vs 25.8–30.4 (all 12 streams); v2.15 published 114.1 total tok/s in another window ([receipts](docs/BENCHMARKS.md#comparison-receipts)).
2. **One stream:** prose / code / JSON are faster than the two-Spark lane: 74.5 / 71.6 / 79.3 vs 59.5 / 67.0 / 67.8 tok/s; code / JSON are slower than v2.15's 86.1 / 91.4 in its published lab window ([receipts](docs/BENCHMARKS.md#comparison-receipts)).
3. **First token:** C1 prose / code / JSON TTFT is 162 / 193 / 225 vs 148 / 179 / 206 ms; C4 is 577 vs 385 ms, slower at both concurrencies ([receipts](docs/BENCHMARKS.md#comparison-receipts)).
4. **Prompt processing:** fresh 8K / 31K / 62K rates are 3111.49 / 3198.99 / 3180.87 vs the two-Spark lane's 1935.58 / 1996.25 / 1998.65 tok/s; v2.15 published 3,374 / 3,642 / 3,707 in another window ([receipts](docs/BENCHMARKS.md#comparison-receipts)).
5. **Requests that do not draft:** schema-constrained JSON, forced tool calls and temperature > 0 take undrafted paths; schema-constrained C4 throughput is NOT MEASURED in the retained receipts, whose 49.12 total tok/s and 12.26–13.04 per-request tok/s are JSON-prompt results without `response_format`, not a schema benchmark; the schema-drafting lever's lab run failed before its schema cells, so it stays off ([request-shape and failure receipts](docs/BENCHMARKS.md#requests-without-drafts)).
6. **Rows and admission:** the shipped source caps each physical decode forward at 8 rows and each shared user window at 4 rows; its planner carries pending users to another round rather than exceeding the cap, and an invalid internal window is refused; four users decode, with further client requests waiting for admission, not receiving extra decode slots ([source receipts](docs/BENCHMARKS.md#rows-context-and-shipped-profile)).
7. **Restart:** M-on lifecycle qualification is PASS 2/2 on the same shipping head/M-on profile; cold 197.586 s, graceful restart 145.404 s, final stop 12.144 s, with PASS: both aliases and 3/3 canaries per cycle; original 900 s startup grace; loaded 6/6, startup bindings 2/2, native exits 6/6 = 0, physical 27/27; outer/observer exits 0; all owners absent; the earlier M-off run measured cold 137.495 s, restart 149.226 s and stop 12.025 s, 2/2 cycles, 6/6 native exits 0 and 27/27 physical checks, not M-on proof ([receipt](docs/BENCHMARKS.md#restart-coverage)).
8. **Model:** text only, no vision; 262,144 tokens per request and 720,888 allocated pool tokens across four users; requests beyond the 2,051-position draft reach continue undrafted ([source receipts](docs/BENCHMARKS.md#rows-context-and-shipped-profile)).
9. **Source only:** build with the exact Zig 0.17.0 SHA256, CUDA and AOT image digests, Python-engine, XGrammar and DLPack revisions in `.env.example`; libdevice and generated objects are not redistributed ([pins](.env.example), [build method](docs/RUN.md)).
10. **Testing:** measured only on 1x RTX 5090 + 2x DGX Spark over native RoCE v2; one ≥4-hour active soak, PASS: 4.050086 active h; 2844.353 s full pauses excluded, requests/errors/restarts/PoolFull PASS: 9,657/9,657 checked requests; 0 HTTP 5xx, 0 restarts, 0 PoolFull. 105 non-gating observations and 208 expected HTTP 400 negative controls retained; this release fill-in adds no hardware measurements ([method](docs/BENCHMARKS.md#soak-and-lifecycle-coverage)).
11. **Levers left out, default off:** 16-physical-row decode (16-row physical decode), eviction changes (eviction changes) and schema/sampled drafting (schema/sampled drafting); the latter stopped at its unchanged +3% C1 upper band, before schema ON ([receipt](docs/BENCHMARKS.md#requests-without-drafts)).
12. **Dispositions:** the original P1-R2 result is FALSE (27,388 one-row rounds, zero pipeline rounds); the desk-used parser result on the same log is PASS (10,362 pipeline rounds, zero problems); the CPU candidate Debug broad test is FAIL (1 after.fast failure, 1,253/1,261 passed, 7 skipped), qualified by isolated 3/3 Debug + 3/3 ReleaseSafe passes, not relabelled broad PASS ([original and used receipts](docs/BENCHMARKS.md#gate-dispositions)).

## Quick start

Build from the pinned source and review the limitations before use. Do not deploy over an existing service without your own ownership, memory and fabric checks:

```sh
git clone https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks.git
cd GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks
cp .env.example .env
# Edit .env for your paths, interfaces, addresses and approved pins.
bash build.sh native x86_64
bash build.sh native aarch64
bash build.sh grammar
bash download.sh print      # review the exact checkpoint; no download in print mode
```

Prepare Triton sets and dense q4 with [docs/RUN.md](docs/RUN.md), then place the role's build outputs on its host. The role launcher does not copy files, install systemd units or start a remote machine. Run in each host's existing guarded foreground/service environment, experts first:

```sh
bash start.sh expert0 check
bash start.sh expert1 check
bash start.sh attention check
# After exclusive GPU ownership, guards, admission and pin checks:
bash start.sh expert0 run
bash start.sh expert1 run
bash start.sh attention run
```

`print` emits only that role's command. `check` calls the native no-GPU flag parser; it is not memory admission or a hardware health check. Run mode refuses the documentation fabric address. By default the API binds loopback; this launcher requires a service-owned 0600 key file for non-loopback exposure. Never paste a key into shell arguments or logs. Your supervisor must stop the exact attention and expert processes gracefully and verify all native exits and memory recovery; no untested automatic recovery controller is included.

## Using the API and operations

Text only. Thinking follows the native server default; client sampling and structured formats are constrained by the native implementation, not guaranteed by generating JSON-looking text. The response parser includes the credited key/end-token/literal-boundary fixes. Read the failed-probe caveat above before assuming unrestricted tool-input coverage. Use the configured model name, health endpoint and API-key headers; protect logs containing prompts and generated arguments.

See [run/build](docs/RUN.md), [design](docs/DESIGN.md) and [troubleshooting](docs/TROUBLESHOOTING.md). Build and test commands are CPU-only; none changes live hosts. The old watchdog and private lab/service controllers are not shipped. Their archived qualification does not test a new operator's service manager or public foreground deployment.

## Layout

| Path | Purpose |
|---|---|
| `engine/` | Native TensorFold snapshot: source, kernels, CPU tests and retained notices |
| `.env.example`, `scripts/lib.sh` | All deployment paths, devices, hosts, ports and pinned revisions; caller exports win |
| `build.sh`, `scripts/container-build.sh` | CPU compiler entry points, bounded containers, no GPU |
| `tools/grammar/` | Exact grammar shim and trusted loader/ownership source; library built locally |
| `tools/afd-prep/` | Credited q4 converter and Triton AOT preparation |
| `start.sh`, `scripts/afd-role.sh` | Local role argv, print/check/run only |
| `tests/test_static.sh` | Offline syntax, topology, role/refusal and source/notice checks |
| `docs/` | Architecture, methods, build/run and limitations |
| `LICENSE`, `LICENSES/`, `NOTICE`, `THIRD_PARTY_NOTICES.md` | Exact legal texts and source authorship |
| `AGENTS.md` | Mia's attribution guide, unchanged |

## Security

- Keep the HTTP API on loopback, a trusted LAN or VPN behind a firewall. Never port-forward it. An API key is not encryption; use a private authenticated proxy or tunnel for remote clients.
- The native rank/exchange protocol is unauthenticated. Use an isolated RoCE subnet/VLAN, with only the three ranks allowed to reach the fabric port; do not expose it to the Internet.
- Builds run as the caller in CPU-only, network-disabled containers, with no GPU devices or privileged mode. Runtime needs the selected CUDA/RDMA device permissions and unlimited locked memory for registration; grant only those, not blanket root access. Docker control is root-equivalent.
- `.env` is trusted shell code; keep it private. It may override paths/pins, but callers' exported variables take precedence. Never execute a downloaded configuration or log secret arguments. The launcher clears the process environment so inherited experimental flags cannot silently alter the profile.
- Root installation and service management are not automated. If you install manually, stage under a service-owned 0700 directory, never a world-writable temporary path; install secret/config files with 0600 and least-privilege sudo.
- A host-memory-floor guard and one GPU tenant are operational requirements, especially on unified-memory GB10. Do not depend on Docker cgroups, hard-kill MPS, automatically reboot, or retry after a failed stop without investigating ownership.

## AI-assisted development

| Model / harness | Recorded role |
|---|---|
| Opus 5.5 (Anthropic), via [Hermes](https://github.com/NousResearch/hermes-agent) | Desk orchestrator and native-engine worker before the 2026-10-13 06:07 EDT worker switch; planning, ports, integration, grammar/concurrency and earlier qualification. Its existing Assisted-by credits remain. |
| GPT-6.1 Sol (OpenAI Codex), via Hermes | Worker/fallback code and tests before that switch; primary workers, planner/scout and subagents after it; batched-draft/ping-pong qualification, source staging, scripts, documentation and this restage. The desk chat remained on Opus. |
| GLM-5.3-Flash (Z.ai), self-hosted via Hermes | Retained previous-version executor credit, as the published v2.15 records; also the served model. No current-v3 worker role is inferred from the model being served. |

Verification uses actual Assisted-by trailers, the source notices and the provenance ledger, not the Git author display name. The shipping-head audit found 13 earlier own-identity commits without an Assisted-by trailer; it does not prove an unrecorded model never assisted. The two recorded current-v3 model families above are preserved, including mixed Anthropic Opus 5.5/GPT work before 2026-10-13; that switch did not erase earlier GPT or Anthropic Opus 5.5 credits. They did not author the upstream work credited above. Scott Quarrier directed the work and retains deployment/publication approval. Numbers are recorded executions, never generated estimates disguised as results.

## Support: none, provided as-is

No support, warranty or commitment to answer issues or pull requests. Review the scripts and your hardware/driver/fabric differences before use. GPU memory exhaustion can leave a Spark needing physical recovery. You operate it at your own risk.

## License and credits

v3.0 remains Apache-2.0, like v2.0: that matches the pinned native TensorFold engine and the Apache-licensed ports. LICENSE is the verbatim Apache-2.0 text; inherited MIT/BSD files keep their own texts in LICENSES. v1.0 remains MIT at its tag. No repository-wide licence change from v2.0 is proposed. The pinned checkpoint's MIT text and later quantization licensing are separate from this code licence.

[NOTICE.md](NOTICE.md) retains every previous-version credit and the native authors; [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) is derived from the provenance ledger, distinguishing imported code, reimplemented designs, dependencies, weights and research-only sources. Mia's pre-existing verbatim attribution guide has no express standalone SPDX terms; no blanket MIT/Apache licence is invented for it. This retained guide is not assigned an invented blanket code licence.

No cubins, fatbins, native executables, grammar libraries, model tensors, caches or container images are shipped. They are built locally at install. NVIDIA authored libdevice, which the Triton compiler may link into generated CUDA objects; the exact original build-toolkit redistribution terms were not available in the legal audit. Source-only staging does not redistribute those object bytes or claim permission to redistribute locally generated ones. A future binary release must resolve its actual toolkit's EULA, redistributable list and required notices first.

## Retained v2.15 upstream replies (historical)

These are the previous release's quoted replies; the current status above supersedes any wording that treated the location question as open.

## Upstream

This split runs on TensorFold's Python engine. In [TensorFold#214](https://github.com/ashhart/TensorFold/issues/214)
this repo's author proposed the split for TensorFold, and Ash Hart answered on 2026-10-06
([his reply](https://github.com/ashhart/TensorFold/issues/214#issuecomment-6010581196)):

- TensorFold's Python engine is frozen ([#286](https://github.com/ashhart/TensorFold/issues/286), at 0.6.5; a 0.6.6
  followed on 2026-10-06 with one CUDA-server option, `--name-priority`). New work goes into the native Zig engine.
- Cross-machine splits now live in the Zig engine's cluster layer (`zig/src/cluster`), which places one model over
  several machines through mcdma links.
- An attention-on-one-GPU, experts-on-the-Sparks layout would be a new placement there. It waits for GLM-5.3-Flash on
  the Zig engine and for the Zig engine's CUDA serving.
- He cites this split's finding that expert compute, not the wire, sets the round time as the reference for that
  design.

He answered again on 2026-10-07 ([his second reply](https://github.com/ashhart/TensorFold/issues/214#issuecomment-6033151997)):

- The split is a real one for GLM-5.3-Flash, and it wants two commands: a serve process for attention, an expert
  process for the routed experts, and a transport between them.
- The Zig engine has no GLM engine and no remote-expert transport yet. Its single-box GLM port comes first, and the
  split follows it, on the same expert weights rather than a second quantization; the transport will not start ahead
  of the model.
- A failed expert process has to fail the request, not hang the attention side. This recipe does that: in the
  soak's drill the request in flight failed 23.3 s after an expert's process stopped, and the attention side kept
  answering ([Operations](#operations)).

This repo stays on the Python engine: TensorFold v0.6.5 plus v0.6.6's two commits and its release commit
(patches 0046-0048), with the patches above.


## Retained v2.15 comparisons

The two first-screen callouts and the "What changed since v1.0" body below are v2.15's own words, retained verbatim. "This version" means v2.15; references to Mia's lane name the 2026-10-06 v1.8 window, not the current-v3 comparison. Resolve historical section links in the [v2.15 README](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/blob/v2.15/README.md).

> **Where v1.0 is better: requests at once, four-stream decode, one-stream prose and testing; on long prompts this version is now ahead.** [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0), this repo's glm53f-afd version of 2026-09-30, serves 16 requests at once (this version decodes eight and queues the rest), decodes 117.6 tok/s at four streams against this version's 114.1 in the lab and prose at one stream 73.1 against 71.3, and passed a 4-hour soak with failure drills; this version's soak passed its four failure drills but not its probe check ([Soak and failure drills](#soak-and-failure-drills)). v1.0 prefills 3.1-3.3K tok/s on 5K-95K prompts; this version prefills 3,374 / 3,642 / 3,707 tok/s on fresh 8K / 31K / 62K prompts in the lab. Both keep a ~1.58M-token KV pool (1,585,152 here): [What changed since v1.0](#what-changed-since-v10).

> **Where Mia's two-Spark lane is better:** it needs two machines, not three, and no RTX 5090; it reads pictures (this split serves text only); and in 2.1's same-window run against her v1.8, each request generated faster at ten clients (16.0 against 12.7 tok/s: her lane decodes four at a time and queues the rest, this split eight) and her KV pool was 8% larger (1,710,080 against 1,579,008 tokens). This version has not been measured against her lane. See [Performance](#performance).

[v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) (2026-09-30) ran Hugh Madden's glm53f-afd with Local Inference Lab's TR3 4-bpw experts. Versions 2.0 to 2.15 keep the three machines and the split, and run the split on TensorFold v0.6.5 with Mia's GLM work and her checkpoint. Against v1.0's published numbers:

- **v1.0 is still better at:**
  - requests at once: 16 slots; 2.15 decodes eight together and queues the rest;
  - decode at four streams, 117.6 tok/s against 2.15's 114.1 in the lab (0.97x), and prose at one stream, 73.1
    against 71.3 (0.98x);
  - AEON-30: 25 of 30, run under its soak load; 2.15 scored 22 of 30, idle and under its soak's load alike;
  - testing: a 4-hour soak with failure drills (on glm53f-afd v1.0.0). 2.15's soak (a 4.11-hour window, 4,071
    requests, no errors) passed its four failure drills but not its probe check: a long streamed tool call
    under load paused 6.3 and 7.0 s between events, over the 6 s allowed, and one gap between probe rounds was
    60 minutes, over the 45 allowed ([Soak and failure drills](#soak-and-failure-drills)).
- **2.15 is better at** prompt processing on long prompts: 3,374 / 3,642 / 3,707 tok/s on fresh 8K / 31K / 62K
  prompts in the lab against v1.0's 3.1-3.3K tok/s on 5K-95K prompts (1.02-1.20x). One stream decodes code and JSON
  faster in the lab (86.1 and 91.4 tok/s against v1.0's 83.7 and 76.2).
  It holds the same ~1.58M-token pool (1,585,152 tokens against ~1.58M) and ships
  v1.0's watcher again. It runs Mia's checkpoint (Apache-2.0) and her serving work: her GLM tool-call fixes, the
  shared-prefix prompt cache and smooth streaming.
- v1.0's numbers come from its own README and BENCHMARKS (decode on its spark-bench-style harness, 1,024-token
  prompts) and were not re-measured here. If many requests at once matter most to you, run
  [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0).

### v2.15's License and credits, retained

This version's scripts, patches and docs are **Apache-2.0** ([LICENSE](LICENSE)), the licence of TensorFold from v0.6.0 and of Mia's recipe, whose work it carries. [v1.0](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0) stays **MIT**, as published. This version vendors no third-party code beyond the patches and ships no weights or binaries. Everything it builds or downloads keeps its own licence, listed in **[NOTICE.md](NOTICE.md)**:

- **engine:** TensorFold, Apache-2.0 at v0.6.5 and v0.6.6;
- **wire:** MCDMA, Apache-2.0;
- **MiaAI-Lab's pull requests and recipe patches:** Apache-2.0, with code she adapted from Jay Leaton's glm53-tensorfold-spark (Apache-2.0) in two of them, and her recipe's contributed patches (E-Zou Shen's, desy0305's, m-naoki-m's and johnwhited's) under her recipe's licence;
- **Hugh Madden's expert prompt kernels in patches 0034 and 0035:** glm53f-afd's code, **MIT** (Copyright (c) 2026 Turquoise Bay AI Pty Ltd), with the MIT code of TensorFold's contributors it carries; both licence texts ship in the patch;
- **designs written anew in patches 0026, 0027, 0032 and 0050:** Hugh Madden's glm53f-afd and mimo26f-afd (MIT) and T.J. Purtell's ds41rt (MIT); no code from them is copied;
- **base model:** Z.AI's GLM-5.3-Flash, MIT;
- **checkpoint:** Mia's AI Lab's EXL3 quantization, Apache-2.0;
- **DFlash2 drafter:** CC BY-NC-ND 4.0, **non-commercial**, referenced only.

All the hard parts are other people's work:

- **[TensorFold](https://github.com/ashhart/TensorFold)** and **[MCDMA](https://github.com/ashhart/MCDMA)** by Ash Hart ([@ashhart](https://github.com/ashhart), [@ashxhart](https://x.com/ashxhart)). Petrus Pennanen ([@ThinkOffApp](https://github.com/ThinkOffApp), [@petruspennanen](https://x.com/petruspennanen)) wrote MCDMA's setup notes and build fix, [MCDMA#5](https://github.com/ashhart/MCDMA/pull/5). Philip Mossop ([@philip-pentatonic](https://github.com/philip-pentatonic)) wrote TensorFold v0.6.6's `--name-priority`, [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445) (patch 0046).
- **Mia's AI Lab** ([@MiaAI-Lab](https://github.com/MiaAI-Lab), [@MiaAI_lab](https://x.com/MiaAI_lab)): the checkpoint; the [two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) whose patches are ported here and against which this version is measured; her TensorFold pull requests #243, #285 and #301. This README uses the layout of the README of her [GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks), at [`674155d`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/674155dec2f2f62bb879801b5ce2cfc759a0bebf).
- **The contributors to her recipe whose patches are ported here:** E-Zou Shen ([@ezoushen](https://github.com/ezoushen), [@ezoushen](https://x.com/ezoushen)) wrote her patches 0071 and 0074 (here 0036 and 0038, tests in 0041; her recipe's pull requests #44 and #62); desy0305 ([@desy0305](https://github.com/desy0305)) wrote her patch 0073 and its checks (here 0037; #51); johnwhited ([@johnwhited](https://github.com/johnwhited)) wrote her patches 0081, 0082 and 0083 and 0073's delivery-failure handling (here 0043-0045 and 0037; #48), with vLLM v1's admission and abort behaviour as his reference; m-naoki-m ([@m-naoki-m](https://github.com/m-naoki-m), [@\_m\_naoki\_m\_](https://x.com/_m_naoki_m_)) wrote her patch 0078 (here 0042; #71, also TensorFold#421). meleesciony ([@meleesciony](https://github.com/meleesciony)) diagnosed the failure of her [issue #75](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/75) and proposed the fix her patch 0077 makes (here 0040 and 0041); Lukas-tek-no-logic ([@Lukas-tek-no-logic](https://github.com/Lukas-tek-no-logic)) reported the `<|assistant|>` leak her patch 0075 fixes ([issue #60](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/issues/60); here 0039).
- **Jay Leaton** ([@jayleaton](https://x.com/jayleaton), [jayleaton](https://github.com/jayleaton)) wrote [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark); two of Mia's ported patches (here 0017 and 0020) carry code she adapted from its patches 0580 and 0620, at [`59e0e33`](https://github.com/jayleaton/glm53-tensorfold-spark/tree/59e0e338b8fe5b7c2c6f73e7e1e705fb6e93c2d4).
- **Hugh Madden / Turquoise Bay AI** ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)) wrote [glm53f-afd](https://github.com/hughmadden/glm53f-afd) (`91db3cc`), the engine of v1.0, which serves GLM-5.3-Flash with attention, the KV cache, drafting and sampling on an RTX 5090 and the routed experts on DGX Sparks, and [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd), its serving shell, wire and RDMA transport. This version runs that split on TensorFold. Patches 0034 and 0035 carry his code: glm53f-rank's expert prompt kernels (the plan, the large-M gate/up and down kernels, the fused epilogue and reduce of `crates/glm53f-rank/kernels/exl3_rank.cu`), which the Sparks run for every prompt chunk; their EXL3 decoder, fragment MMA and Hadamard butterfly are TensorFold's MIT code (`bb4b4a3`) as glm53f-afd carries it. Patches 0026, 0027 and 0032 write four of his designs for this tree, with no code copied: the exchanges kept in flight ahead of the expert ranks (glm53f-afd `91db3cc`, `crates/glm53f-serve/src/lib.rs`; on here), the host RAM tier for kept prompts (`crates/glm53f-coordinator/src/hostcache.rs` and `scheduler.rs`; on), the prefill in two and four lanes (mimo26f-afd [`bab9fa2`](https://github.com/hughmadden/mimo26f-afd/tree/bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79), glm53f-afd `91db3cc`; four lanes, on) and FP8 wire rows (off). Patch 0050 writes a fifth: glm53f-afd's BF16 return planes, each expert rank's sum rounded to BF16 and the planes added in FP32 (`crates/glm53f-rank/README.md`; on, for prompt windows). Patches 0049 and 0051 run glm53f-rank's own schedule for its biggest windows on the Sparks' prompt windows of 1,536 rows and more.
- **T.J. Purtell** ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell)) wrote [ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and [cuteafd](https://github.com/tpurtell/cuteafd), engines that run attention on RTX GPUs and the routed experts on DGX Sparks. This version uses no code from them. Hugh Madden's expert prompt kernels (patch 0034) split every expert by intermediate channel as glmrt does, which T.J. Purtell designed; glm53f-afd's host RAM tier, whose design patch 0032 follows, takes its host cache's eviction design from ds41rt, through mimo26f-afd. Patch 0026's FP8 wire rows (off) use ds41rt's row format, E4M3 with a UE8M0 scale per 32 values, which reached glm53f-afd through mimo26f-afd.
- **fla-org's [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)** (MIT) implements the chunkwise algorithm of gated delta networks and Kimi Delta Attention that Mia's chunked KDA kernel computes in its chunked WY / UT form; her kernel is her own code.
- **Patryk Mikołajczyk** ([@mikolaj92](https://github.com/mikolaj92)) wrote [TensorFold#140](https://github.com/ashhart/TensorFold/pull/140), which bounds GLM's DSA selection to the pools a row can see; Mia's decode patch 0043 (here in 0017) extends that bound to decode. **Aditya Thyagarajan** ([@aditya1503](https://github.com/aditya1503)) wrote [TensorFold#200](https://github.com/ashhart/TensorFold/pull/200), MTP concurrency, which #243 superseded.
- **Weights:** [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) (the base model), Mia's AI Lab (the EXL3 quantization, made with turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3)), [IncoAI](https://huggingface.co/incoai) (the DFlash2 drafter).
- **[AGENTS.md](AGENTS.md):** the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution), unchanged. Agents that edit this repo follow them.
- **v1.0's credits** (glm53f-afd and its upstreams, Local Inference Lab's weights, and everyone else v1.0 names) stay in [NOTICE.md](NOTICE.md#previous-version-v10), verbatim.
- **This recipe:** [@squarrier](https://github.com/squarrier), with the AI agents above.

## Source-credit annex — ledger-derived

### Ash Hart (@ashhart)

Profile: https://github.com/ashhart

- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/f8fe17d24629aedabf90bbf78279dd776e6d62e7) at `f8fe17d24629aedabf90bbf78279dd776e6d62e7` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart (@ashhart, X @ashxhart) authored TensorFold; its v1.0.2 commit f8fe17d is the base of this fork (native TensorFold AFD snapshot).
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1) at `ed78d6fc204d89d90b045bf033d6551e7714f3a1` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart and the TensorFold contributors authored python-0.6's EXL3 kernels and GLM's exl3.cu (ashhart/TensorFold ed78d6f), which f3949cf7 copied into zig/kernels/cuda/exl3/.
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1) at `ed78d6fc204d89d90b045bf033d6551e7714f3a1` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored python-0.6's GLM extension device code (ashhart/TensorFold ed78d6f), which local integration copied as the glm_* fatbins (dd7de994).
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/db281878ddb836fd0df510d8771ecb7e0fe47d26) at `db281878ddb836fd0df510d8771ecb7e0fe47d26` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored copies.py (ashhart/TensorFold db28187), which local integration brought back as zig/tests/cuda/copies.py.
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1) at `ed78d6fc204d89d90b045bf033d6551e7714f3a1` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored python-0.6's GLM Triton kernel sources (glue.py, attention.py, qmm.py and most of forward.py, latent.py and sparse.py; ashhart/TensorFold ed78d6f), compiled ahead of time into the Triton kernel set (ed10d2f4) and wrapped in Zig (22dc0785).
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/afa35dc0b6b3ae889ea3f26ace68c4e45cb162a7) at `afa35dc0b6b3ae889ea3f26ace68c4e45cb162a7` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored the fix that keeps a declared GLM tool's name when it has spaces (ashhart/TensorFold afa35dc0), cherry-picked as 6b433291.
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/1b0268954745802a3ee9b0d1828a51f5bdee0c1d) at `1b0268954745802a3ee9b0d1828a51f5bdee0c1d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. chaog992 authored TensorFold #594 (711bacf2): nested completion prompt ids past u32 decode as unknown ids; Ash Hart's merge 1b026895 was picked as 647941a3.
- **dependency**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/c080a18eeae970a742135ee86325115b0387a1c8) at `c080a18eeae970a742135ee86325115b0387a1c8` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored TensorFold v1.0.6's tf_grammar.cc shim (c080a18e), which wraps XGrammar in local integration's libtfgrammar.so.
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/c080a18eeae970a742135ee86325115b0387a1c8) at `c080a18eeae970a742135ee86325115b0387a1c8` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored v1.0.6's mask-before-draw (tf_grammar_mask, c080a18e), which local integration's device-masked greedy draw follows (b11df7cf).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/d7b134cdf86a02b7c91d0d137ce733c04ae53cfd) at `d7b134cdf86a02b7c91d0d137ce733c04ae53cfd` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @BlivionIaG authored TensorFold #623 (head d7b134cd): the lane core's optional prefill_step / fillStream and the host's filling queue; Ash Hart merged it as 1184cdbb. local integration ported it (9263173e, picked as cd9627dc).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b) at `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Ash Hart authored TensorFold v1.0.7 (e7c523eb): the lane host's Admit hook and its test, which local integration ported (15f6451d, picked as cf2c45c8), and #639's shared MLA cache pool, whose region rule cuda_afd_pool.zig reimplements with no code copied.
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/c080a18eeae970a742135ee86325115b0387a1c8) at `c080a18eeae970a742135ee86325115b0387a1c8` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Ash Hart authored TensorFold v1.0.6's tf_grammar.cc shim (c080a18e), which wraps XGrammar in local integration's libtfgrammar.so.
- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.
- **reference**, [https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/2636c1cb3019ede4ab27191299d5ad6843344942) at `2636c1cb3019ede4ab27191299d5ad6843344942` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. iegamo (@ahinore) authored PR164 input literal-added-token marking in this fork/commit; its commit preserves Opus 5.5 as co-author, and all Mia/TensorFold/inherited notices remain intact.
- **reference**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b) at `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Ash Hart (@ashhart/@ashxhart) and TensorFold contributors authored the configured v1.0.7 base read here to reconstruct selected shared-source patch hunks; no implementation was taken into our engine.

### Mia (@MiaAI_lab)

Profile: https://github.com/MiaAI-Lab

- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/b5808ac) at `b5808ac` (NOASSERTION). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab) authored the GLM-5.3-Flash EXL3 TensorFold lane github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold; local integration prints local integration's recorded gb10 C4 figures for its b5808ac (100.5 tok/s aggregate, 25.9-29.6 per stream) beside local integration-STREAMS for comparison; nothing was taken and gb10 is never contacted.
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/b23c10ad305aa068e12b901ee8ebd8166b9aa6fd) at `b23c10ad305aa068e12b901ee8ebd8166b9aa6fd` (MIT). Native source, reimplemented design or build/runtime dependency. Mia (MiaAI-Lab) authored the 512-row prompt chunks of latent.py, sparse.py and forward.py (ashhart/TensorFold b23c10ad), compiled into the Triton kernel set (ed10d2f4).
- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab) at `3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. ezoushen authored the compaction rule of patch 0074-glm-compact-before-evict (pull request #62, issue #61) in Mia's GLM-5.3-Flash recipe, read at 3fd60dfc; cuda_afd_pool.zig reimplements it with no code copied.
- **code-ported**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/tree/cbfc429) at `cbfc429` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia (MiaAI-Lab) authored the two-rank CUDA layer of patch 0006 (with 0002 and 0007) in Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold at cbfc429, which local integration ported (b5adb25e) and local integration's exchange reuses for its RoCE proxy (2b6703d7).
- **code-ported**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/tree/cbfc429) at `cbfc429` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored flashnext_aot.py, flashnext_triton_fixtures.py and cuda_triton.zig in Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold at cbfc429, from which tools/zig/glm_aot*.py, glm_triton_*.py and cuda_triton*.zig are adapted (ed10d2f4, 22dc0785).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold/tree/a687724) at `a687724` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored patches/0128-glm-full-expert-prefetch.patch (L2 prefetch, programmatic launch) of GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold at a687724, applied to the EXL3 expert kernel copies in zig/kernels/cuda/exl3/l2d/ (7d6ab8a2).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/549fdc562c477d477244d2df0590d6c14bda4889) at `549fdc562c477d477244d2df0590d6c14bda4889` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored NUCLEUS_UNION, patches/0034-cuda-nucleus-union.patch at recipe commit 549fdc5 of GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold, which local integration ported for the TP nucleus draw (df44f812) and local integration reuses (4ae62ad8, picked as 29769d49).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/549fdc562c477d477244d2df0590d6c14bda4889) at `549fdc562c477d477244d2df0590d6c14bda4889` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored patches/0005-glm-dense-q4.patch (the q4 dense form, quantize4_mse) at v1.10, commit 549fdc5, of her GLM-5.3-Flash recipe, which qualification_63cac091.5 B8 ported as TF_GLM_DENSE=q4 (3725fc86).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/549fdc562c477d477244d2df0590d6c14bda4889) at `549fdc562c477d477244d2df0590d6c14bda4889` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored patches/0002-glm-dense-fp8.patch (make_f8 and the FP8 matmul _fmm) at v1.10, commit 549fdc5, of her GLM-5.3-Flash recipe, which qualification_63cac091.5 B9 ported for the FP8 head (657fc439).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/549fdc562c477d477244d2df0590d6c14bda4889) at `549fdc562c477d477244d2df0590d6c14bda4889` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored patch 0002's FP8 matmul at 549fdc5, whose launches the MTP draft layer on FP8 uses (32094381, 219461a2).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) at `1576746a04983b6eded0551dbf22512ee9e95654` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored recipe patches 0001-glm-exl3-prompt-experts, 0020-glm-prompt-experts-order and 0009-glm-prefill-kernels at 1576746, adapted onto the universal kernel and carried by local integration into zig/kernels/cuda/exl3/x1/ (5ab6226b).
- **code-ported**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) at `1576746a04983b6eded0551dbf22512ee9e95654` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored patches/0004-glm-prompt-kernels.patch at 1576746, the EXL3 prompt-window expert kernel that local integration ported into zig/kernels/cuda/exl3/local integration/ (b4b853f7) as the Python TF-AFD 2.15 lane's patch 0024 ports it.
- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746a04983b6eded0551dbf22512ee9e95654) at `1576746a04983b6eded0551dbf22512ee9e95654` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia authored the one-program sum of a matmul's K splits in patch 0004-glm-prompt-kernels, which zig/kernels/cuda/exl3/x1s/ (SEQ) reimplements on X1's kernel (c898e828).
- **reference**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/a3aa89835022c55ca8e55008c37785954834e04f) at `a3aa89835022c55ca8e55008c37785954834e04f` (MIT). Research/benchmark reference only; no code taken for v3.0. Mia authored the Flash Next vision patch 0008 (a3aa8983), which upstream TensorFold's Python engine ported; no file of it is in this Zig tree.
- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold/tree/ba19a9c16a51175a90e1b135748219c4fb51fb0b) at `ba19a9c16a51175a90e1b135748219c4fb51fb0b` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Mia (@MiaAI_lab) authored tools/exact.py in GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold (ba19a9c), whose exactness checks (each reply equal to its alone reference) local integration's client implements with new code (local integration-kit 813781c2) and the users path's together = solo tests follow.
- **reference**, [https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution) at `81a6dd958181aa1d581cb24d54d30a327842e955bd42b40256e54bfc1abfa8cb` (none). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab) authored the AGENTS.md credit and attribution guide; local integration read its published page and download and preserves her credit without copying additional guide text into the shipping change or assigning it an invented licence.
- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.
- **reference**, [https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/2636c1cb3019ede4ab27191299d5ad6843344942) at `2636c1cb3019ede4ab27191299d5ad6843344942` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. iegamo (@ahinore) authored PR164 input literal-added-token marking in this fork/commit; its commit preserves Opus 5.5 as co-author, and all Mia/TensorFold/inherited notices remain intact.

### Mia (@MiaAI_lab)

Profile: https://huggingface.co/Mia-AiLab

- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights the GLM-5.3-Flash lane reads (7cecb415). Never shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration head e7c960e8). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia (@MiaAI_lab) authored the EXL3 quantization of GLM-5.3-Flash by Z.ai in Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold revision 078455ff; its exact published licence and card are MIT, with Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/41ba3c2b82eff6ef5b93debff02f47ae4681d824) at `41ba3c2b82eff6ef5b93debff02f47ae4681d824` (Apache-2.0). Other weight revision documented for licence history; not selected. Mia (@MiaAI_lab) authored the EXL3 quantization of Z.ai GLM-5.3-Flash; at revision 41ba3c2b its published terms are Apache-2.0 for the quantization, with Z.ai MIT retained for the base model in LICENSE-GLM-5.3-Flash.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a, local integration's kit). Never copied or shipped.

### AEON-7 (@AEON-7)

Profile: https://github.com/AEON-7

- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration's local integration kit runs against the lab endpoint (gate G-AEON) through Hermes's local endpoint runner on Scott's fork commit 9990ec10; nothing of it is copied into the TensorFold fork or the kit.
- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration's kit runs under local integration load (S-AEON) through the canonical aeon_one.sh wrapper on Scott's fork commit 9990ec10; nothing of it is copied into the kit.
- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration ran on boot M-on (G-AEON) against the lab URL through the local integration kit's canonical aeon_one.sh wrapper on Scott's fork commit 9990ec10; nothing of it is copied.

### Andrey Kolesnikov (@akol1)

Profile: https://github.com/akol1

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/861348862a0f108027575c6e529e527e0bfe754d) at `861348862a0f108027575c6e529e527e0bfe754d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Andrey Kolesnikov (@akol1) authored forward.py's EXL3 experts on the universal path for any bit width (ashhart/TensorFold 86134886), compiled into the Triton kernel set (ed10d2f4).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/720c409114c123ed1efa19f0725cd05acc189786) at `720c409114c123ed1efa19f0725cd05acc189786` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Andrey Kolesnikov (@akol1) authored TensorFold #650's whole-response content condition (ashhart/TensorFold 720c4091), which local integration reimplemented (56b60ccd).

### Apple Inc. (ml-explore)

Profile: https://github.com/ml-explore

- **code-ported**, [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx/tree/v0.32.3) at `v0.32.3` (MIT). Native source, reimplemented design or build/runtime dependency. Apple Inc. (ml-explore) authored MLX, whose affine quantization layout and 0.32 kernel arithmetic the inherited Metal kernels of Ash's base reproduce.
- **code-ported**, [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/tree/v0.31.3) at `v0.31.3` (MIT). Native source, reimplemented design or build/runtime dependency. Apple Inc. (ml-explore) authored mlx-lm, whose expert selection the inherited Metal routing kernels of Ash's base reproduce.

### Ashish Kulkarni

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

### avlp12

Profile: https://github.com/avlp12

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

### BadAd84 (@BadAd84)

Profile: https://github.com/BadAd84

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) at `90ccb49215d58161f9c6017e9106aeecc45fe762` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the all-route SSE extension and tests (90ccb492) in Mia's GLM-5.3-Flash recipe, after stevehansen's issue106 diagnosis, design and patch; local integration's native SSE transport reimplements them (139aa26f, picked as aa39deb2).
- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/616b840651a428d514607c3ce5524ed5bc4c206d) at `616b840651a428d514607c3ce5524ed5bc4c206d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the tool-call rejection log design of patch 0114 (PR137, 616b8406) in Mia's recipe, which local integration reimplements default-off (19db3dba).
- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/11619a191999398c6392e8079fe4dcc545cc3f1a) at `11619a191999398c6392e8079fe4dcc545cc3f1a` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. BadAd84 authored the FILL_ROWS change (PR #119, 11619a19) in Mia's recipe; local integration compared against it and took no code.
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/0c7b88683c19d14f719287268817f7024341ab75) at `0c7b88683c19d14f719287268817f7024341ab75` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #34 (the draft stop under a confidence floor) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at 0c7b8868, a design this tree reimplements with no code copied (1efdb33a).
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/a82cf8aee2d25e26c8cbde44dc54a985e54671f7) at `a82cf8aee2d25e26c8cbde44dc54a985e54671f7` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #35 (patches/0010, the nucleus draw's row reductions on 1,024 threads) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at a82cf8ae, a design this tree reimplements with no code copied (33d8ab99).
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/e6a35d22336216b1bd88ab1465139dce9ed9dc12) at `e6a35d22336216b1bd88ab1465139dce9ed9dc12` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #36 (the MTP draft layer on FP8) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at e6a35d22, a design this tree reimplements with no code copied (219461a2).

### bertholomus (@bertholomus)

Profile: https://github.com/bertholomus

- **code-copied**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/dbe0773) at `dbe0773` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus (Albert Lee, BertholomusAI) authored group_count_kernel and group_place_kernel (bertholomus/TensorFold glm-dsa-tp4 dbe0773), carried unchanged into the EXL3 expert kernels (0cb7622f).
- **code-copied**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/90b68e063a0b8bba59aaf13340ef16619e751195) at `90b68e063a0b8bba59aaf13340ef16619e751195` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus authored grouped_mma_kernel and work_list_kernel (bertholomus/TensorFold deepseek-v41-tp2 90b68e0; first in glm-dsa-tp4 2786594), copied into zig/kernels/cuda/exl3/mma/ with one marked change (fd624f2e).
- **design**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/ce20a545ff5ec805651fea28d9d3ffbd8e211f89) at `ce20a545ff5ec805651fea28d9d3ffbd8e211f89` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus authored the launch order of deepseek-v41-zig's exl3_experts.zig (bertholomus/TensorFold ce20a545), which the halves' MMA launcher follows (fd624f2e).

### Bizuayeu (@Bizuayeu)

Profile: https://github.com/Bizuayeu

- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/b952862585cf1bb564c297926c0072db2b48bc71) at `b952862585cf1bb564c297926c0072db2b48bc71` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the GLM key repair (a call whose <arg_key> the model wrote as whitespace gets it back) and its test case in github.com/Bizuayeu/TensorFold commit b9528625 (Apache-2.0); local integration's served probe m1 ports that test case's input and expected call, and the dry fake reimplements the repair's outcome.
- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/557011805a64639ff5730aa980f7a100ab3ee7d9) at `557011805a64639ff5730aa980f7a100ab3ee7d9` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the end-token close (a GLM value the end token left open is closed) and its test case in github.com/Bizuayeu/TensorFold commit 557011805 (Apache-2.0); local integration's served probe m2 ports that test case's input and expected call, and the dry fake reimplements the close's outcome.
- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/be17a101c47c6f20ac2295ea59922ad777d02252) at `be17a101c47c6f20ac2295ea59922ad777d02252` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the Responses include rule (OpenAI's eight include values accepted, any other value refused by name, a non-list refused), its two refusal messages and its test cases in github.com/Bizuayeu/TensorFold commit be17a101 (Apache-2.0); local integration's served probes m3a-m3c port those test cases, spec.py carries his values and messages, and the dry fake reimplements the rule.
- **reference**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Bizuayeu (@Bizuayeu) authored the shared verify forward design (a round's verify windows share one forward, each stream's state on its own) in github.com/Bizuayeu/TensorFold commit 75131b66 (Apache-2.0), which local integration local integration reimplements and local integration local integration builds on; local integration's kit only switches those on and gates them in the local integration.
- **reference**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/146e113f6ae3a0b2b45626af6ac627479b4e2127) at `146e113f6ae3a0b2b45626af6ac627479b4e2127` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Bizuayeu (@Bizuayeu) authored the shared verify forward design (the shared windows' mixer (cuda_shared.zig)) in github.com/Bizuayeu/TensorFold commit 146e113f (Apache-2.0), which local integration local integration reimplements and local integration local integration builds on; local integration's kit only switches those on and gates them in the local integration.
- **reference**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/d3cc052c554790c7767b126ab61c1831cd14fc7a) at `d3cc052c554790c7767b126ab61c1831cd14fc7a` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Bizuayeu (@Bizuayeu) authored the shared verify forward design (concurrent streams, each on its own sequence (a slot) on every rank) in github.com/Bizuayeu/TensorFold commit d3cc052c (Apache-2.0), which local integration local integration reimplements and local integration local integration builds on; local integration's kit only switches those on and gates them in the local integration.
- **reference**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/2bc75a69db885fd6523c6bac1ff7cee66f6b496b) at `2bc75a69db885fd6523c6bac1ff7cee66f6b496b` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Bizuayeu (@Bizuayeu) authored the shared verify forward design (a prompt pass goes a chunk at a time, the other streams' rounds between chunks) in github.com/Bizuayeu/TensorFold commit 2bc75a69 (Apache-2.0), which local integration local integration reimplements and local integration local integration builds on; local integration's kit only switches those on and gates them in the local integration.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored a round's verify windows sharing one forward, each stream's state on its own (Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66), the shared verify forward local integration reimplements with no code copied (badfae2d, 976ad633; on local integration cfc9ddd5, 27042de1).
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/146e113f6ae3a0b2b45626af6ac627479b4e2127) at `146e113f6ae3a0b2b45626af6ac627479b4e2127` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored the shared verify forward (follow-up commit) (Bizuayeu/TensorFold pr/glm53-flash-cuda 146e113f), the shared verify forward local integration reimplements with no code copied (badfae2d, 976ad633; on local integration cfc9ddd5, 27042de1).
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/d3cc052c554790c7767b126ab61c1831cd14fc7a) at `d3cc052c554790c7767b126ab61c1831cd14fc7a` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored the shared verify forward (follow-up commit) (Bizuayeu/TensorFold pr/glm53-flash-cuda d3cc052c), the shared verify forward local integration reimplements with no code copied (badfae2d, 976ad633; on local integration cfc9ddd5, 27042de1).
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/2bc75a69db885fd6523c6bac1ff7cee66f6b496b) at `2bc75a69db885fd6523c6bac1ff7cee66f6b496b` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored the shared verify forward (follow-up commit) (Bizuayeu/TensorFold pr/glm53-flash-cuda 2bc75a69), the shared verify forward local integration reimplements with no code copied (badfae2d, 976ad633; on local integration cfc9ddd5, 27042de1).
- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/b952862585cf1bb564c297926c0072db2b48bc71) at `b952862585cf1bb564c297926c0072db2b48bc71` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu authored repairGlmKeys and its tag scan, with test cases (Bizuayeu/TensorFold b9528625), ported by local integration (83a8b1cb, e293790e; on local integration a9e0b220, b794fb79).
- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/557011805a64639ff5730aa980f7a100ab3ee7d9) at `557011805a64639ff5730aa980f7a100ab3ee7d9` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu authored the end token's close of a cut GLM value (design), with test cases (Bizuayeu/TensorFold 55701180), ported by local integration (83a8b1cb, e293790e; on local integration a9e0b220, b794fb79).
- **code-ported**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/be17a101c47c6f20ac2295ea59922ad777d02252) at `be17a101c47c6f20ac2295ea59922ad777d02252` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu authored /v1/responses include handling, its tests and the frozen corpus case (Bizuayeu/TensorFold be17a101), ported by local integration (83a8b1cb, e293790e; on local integration a9e0b220, b794fb79).
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/322b4de6c40318e2ce4a263b7c6505300b557ff4) at `322b4de6c40318e2ce4a263b7c6505300b557ff4` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu authored the rank fingerprint field list of cuda_ranks.zig lines 1-31 (Bizuayeu/TensorFold 322b4de6), which local integration reimplements with no code copied (81d7a557; on local integration aa70858b).
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored the shared verify windows design (a round's verify windows share one forward, each stream's state on its own) in github.com/Bizuayeu/TensorFold commit 75131b66 (Apache-2.0); local integration local integration reimplements its drafted form on the AFD users path (each user's pending row and MTP drafts verified in one forward, accepted and rolled back per user) without his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/146e113f6ae3a0b2b45626af6ac627479b4e2127) at `146e113f6ae3a0b2b45626af6ac627479b4e2127` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored the shared windows' mixer (each window's mixer part on its own stream's state inside the one shared forward, in cuda_shared.zig) in github.com/Bizuayeu/TensorFold commit 146e113f (Apache-2.0); local integration local integration reimplements that split for drafted windows in cuda_afd_users_dsv.zig without his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/d3cc052c554790c7767b126ab61c1831cd14fc7a) at `d3cc052c554790c7767b126ab61c1831cd14fc7a` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Bizuayeu (@Bizuayeu) authored concurrent streams, each on its own sequence (state, sampling rule and MTP rows by slot on every rank), in github.com/Bizuayeu/TensorFold commit d3cc052c (Apache-2.0); local integration local integration keeps each user's MTP head, held drafts and rollback on its own slot that way, without his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Historical authorship receipt for the batched-MTP extension now included and enabled in v3.0. Bizuayeu (@Bizuayeu) authored the shared verify design in https://github.com/Bizuayeu/TensorFold at 75131b66c39fdcf55dbd130ccaec166b65394f3d (Apache-2.0), which local integration extends through local integration; his MTP drafting is per stream, this new batched-MTP extension copies none of his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the shared verify windows design in github.com/Bizuayeu/TensorFold commit 75131b66 (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own State and <=4-row window, without copying his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/146e113f6ae3a0b2b45626af6ac627479b4e2127) at `146e113f6ae3a0b2b45626af6ac627479b4e2127` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the shared windows mixer in github.com/Bizuayeu/TensorFold commit 146e113f (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own <=4-row State window, with no code of his copied.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/d3cc052c554790c7767b126ab61c1831cd14fc7a) at `d3cc052c554790c7767b126ab61c1831cd14fc7a` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the concurrent streams on their own sequence by slot in github.com/Bizuayeu/TensorFold commit d3cc052c (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own <=4-row State window, with no code of his copied.

### BlivionIaG (@BlivionIaG)

Profile: https://github.com/BlivionIaG

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/d7b134cdf86a02b7c91d0d137ce733c04ae53cfd) at `d7b134cdf86a02b7c91d0d137ce733c04ae53cfd` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @BlivionIaG authored TensorFold #623 (head d7b134cd): the lane core's optional prefill_step / fillStream and the host's filling queue; Ash Hart merged it as 1184cdbb. local integration ported it (9263173e, picked as cd9627dc).

### BobClawblaw (@BobClawblaw)

Profile: https://github.com/BobClawblaw

- **dependency**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/8ced20188b70fe301d4c9f320430b182a68d216f) at `8ced20188b70fe301d4c9f320430b182a68d216f` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @BobClawblaw authored TensorFold #596 (head 8ced2018, merged d7d5f6f1, in v1.0.6 c080a18e): libtfgrammar and the lane-core grammar that local integration's external grammar library builds.

### chaog992 (@chaog992)

Profile: https://github.com/chaog992

- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/2193d145397500f7b661dd4697240024e1089b68) at `2193d145397500f7b661dd4697240024e1089b68` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. chaog992 authored TensorFold #560 (ashhart/TensorFold 2193d145, merged 17a75f55): a request keeps its own local integration of its key label; local integration picked it as 7eab6148.
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/1b0268954745802a3ee9b0d1828a51f5bdee0c1d) at `1b0268954745802a3ee9b0d1828a51f5bdee0c1d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. chaog992 authored TensorFold #594 (711bacf2): nested completion prompt ids past u32 decode as unknown ids; Ash Hart's merge 1b026895 was picked as 647941a3.

### data-angel (@data-angel)

Profile: https://github.com/data-angel

- **reference**, [https://github.com/Enntity/sparkglm](https://github.com/Enntity/sparkglm/pull/46) at `PR #46` (unknown). Research/benchmark reference only; no code taken for v3.0. @data-angel / Enntity's sparkglm PR #46 (RC4.1) was read only by local integration; nothing was taken.

### DLPack contributors (dmlc)

Profile: https://github.com/dmlc

- **dependency**, [https://github.com/dmlc/dlpack](https://github.com/dmlc/dlpack/tree/bbd2f4d32427e548797929af08cfe2a9cbb3cf12) at `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. The DLPack contributors authored DLPack (dmlc/dlpack bbd2f4d3), XGrammar's submodule inside local integration's libtfgrammar.so.

### Enntity (@Enntity)

Profile: https://github.com/Enntity

- **reference**, [https://github.com/Enntity/atlas](https://github.com/Enntity/atlas/tree/ade7b012) at `ade7b012` (AGPL-3.0). Research/benchmark reference only; no code taken for v3.0. Atlas (AGPL-3.0) was read only by local integration; nothing was taken.

### ezoushen (@ezoushen)

Profile: https://github.com/ezoushen

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab) at `3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. ezoushen authored the compaction rule of patch 0074-glm-compact-before-evict (pull request #62, issue #61) in Mia's GLM-5.3-Flash recipe, read at 3fd60dfc; cuda_afd_pool.zig reimplements it with no code copied.

### fly9i (@fly9i)

Profile: https://github.com/fly9i

- **reference**, [https://github.com/fly9i/spark-engine](https://github.com/fly9i/spark-engine/tree/b09a53e) at `b09a53e` (MIT). Research/benchmark reference only; no code taken for v3.0. fly9i's spark-engine v0.1.1 (b09a53e) was studied and benchmarked by local integration; nothing of it is in this tree.

### Hugh Madden (@dangerm00se)

Profile: https://github.com/hughmadden

- **code-copied**, [https://github.com/hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) at `91db3cc6fe672e2724efa3464f63bd31493f63f6` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden (@dangerm00se) authored glm53f-afd's large-M expert kernels and KT3 schedule (hughmadden/glm53f-afd 91db3cc), whose device lines zig/kernels/cuda/g53/ carries as python 2.15 ports them (e5fb7183).
- **design**, [https://github.com/hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) at `91db3cc6fe672e2724efa3464f63bd31493f63f6` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden (@dangerm00se) and T.J. Purtell (@wrldsuksgo2mars) authored the AFD attention/expert split (glm53f-afd 91db3cc, as our Python TF-AFD 2.15 lane runs it), which the three-node exchange (2b6703d7), the attention role on a GPU (cf594679), prompt lanes (29b1e487) and pairs (a3143b83) reimplement with no code copied.
- **design**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/c05d9939a38e3db66767291b500c97933df6497b) at `c05d9939a38e3db66767291b500c97933df6497b` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden authored cuteafd #22 (merged as c05d9939 in T.J. Purtell's tpurtell/cuteafd): decode graphs keyed by what their launches read, which cuda_afd_graphs.zig reimplements with no code copied (1aee087e).
- **reference**, [https://github.com/tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) at `dc9c26192375a80b1de7bb28b182fc5bbdbf67c4` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Hugh Madden (@hughmadden; X @dangerm00se) authored the sequence-CTA and BF16 projection-tile contributions in SparkInfer PRs #9/#10; T.J. Purtell (@tpurtell) reviewed and merged them, with the inherited SparkInfer/b12x contributors retained.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/b1755f59b43c94d163e7f5e5ea8d823fa981c0d8) at `b1755f59b43c94d163e7f5e5ea8d823fa981c0d8` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell) authored cuteafd draft #137 integrating Hugh Madden (@hughmadden)'s KDA scheduling and rejecting repeated nonnegative recurrent slots.

### iegamo (@ahinore)

Profile: https://github.com/ahinore

- **reference**, [https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/2636c1cb3019ede4ab27191299d5ad6843344942) at `2636c1cb3019ede4ab27191299d5ad6843344942` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. iegamo (@ahinore) authored PR164 input literal-added-token marking in this fork/commit; its commit preserves Opus 5.5 as co-author, and all Mia/TensorFold/inherited notices remain intact.

### Jay Leaton (@jayleaton)

Profile: https://github.com/jayleaton

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/4284048d1aca2799e543a7400b3f07519befb515) at `4284048d1aca2799e543a7400b3f07519befb515` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton (@jayleaton) authored the incomplete-UTF8-tail bounds repair of TensorFold #634 (4284048d), adapted into zig/src/server/json_write.zig (419164dc, picked as e3c490e2).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/986c996041e25e8b4a0341a8b54524a9d2248c12) at `986c996041e25e8b4a0341a8b54524a9d2248c12` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton authored the CUDA launch-failure context design of TensorFold #635 (986c9960), adapted as the opt-in bounded launch failure context (59f1310e, picked as 3202352c).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/5fa8ad0fd9c53131c599c39f8885117d5a582fda) at `5fa8ad0fd9c53131c599c39f8885117d5a582fda` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton authored the CUDA launch-failure context design of TensorFold #635 (5fa8ad0f), adapted as the opt-in bounded launch failure context (59f1310e, picked as 3202352c).
- **reference**, [https://github.com/jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) at `unversioned reference` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Jay Leaton authored inherited parser/cancellation work and patch 0620 in jayleaton/glm53-tensorfold-spark, credited through Mia's recipe NOTICE; no file of that repo is in this tree (his TensorFold #634/#635 have their own entries).

### jundot (@jundot)

Profile: https://github.com/jundot

- **reference**, [https://github.com/jundot/omlx](https://github.com/jundot/omlx) at `unversioned reference` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. jundot's oMLX vendored the mlx-vlm GLM implementation that upstream TensorFold's Python engine follows; nothing from it is in this tree.

### Jürgen Schmied (@jschmied)

Profile: https://github.com/jschmied

- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.

### Kazuho Oku (@kazuho)

Profile: https://github.com/kazuho

- **dependency**, [https://github.com/kazuho/picojson](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa/3rdparty/picojson) at `mlc-ai/xgrammar@97787376faee5ed8466cfad57c99855e4ce2f6aa:3rdparty/picojson (tree 0d9c226c2e13bada877cd092048988070ac5d232)` (BSD-2-Clause). Native source, reimplemented design or build/runtime dependency. Kazuho Oku authored picojson (Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku), vendored in XGrammar's 3rdparty/picojson and built into local integration's libtfgrammar.so.

### kky42 (@kky42)

Profile: https://github.com/kky42

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/c66a6cfd9ae16226b30f29df877b0194ae8b3945) at `c66a6cfd9ae16226b30f29df877b0194ae8b3945` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @kky42 authored the quoted-tag examples of PR134 (c66a6cfd) in Mia's recipe, used as test cases by local integration (19db3dba).

### Lazarus-931

Profile: https://github.com/Lazarus-931

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

### local-inference-lab (@local-inference-lab)

Profile: https://github.com/local-inference-lab

- **design**, [https://github.com/local-inference-lab/b12x](https://github.com/local-inference-lab/b12x/tree/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88) at `2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. local-inference-lab authored b12x's RoCEnante one-shot all-gather protocol, which Mia's fn_roce.cu and roce_proxy.c implement in new code (ported by local integration, b5adb25e).

### local-inference-lab contributors (@local-inference-lab)

Profile: https://github.com/local-inference-lab

- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.

### Mathieu Fenniak

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

### mikolaj92 (@mikolaj92)

Profile: https://github.com/mikolaj92

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/b3b8a39119f8da02026b3f2601e02eb411be1339) at `b3b8a39119f8da02026b3f2601e02eb411be1339` (MIT). Native source, reimplemented design or build/runtime dependency. mikolaj92 (github.com/mikolaj92) authored sparse.py's skip of invisible pool tiles (ashhart/TensorFold b3b8a391), compiled into the Triton kernel set (ed10d2f4).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/f119334d79981225bc103fae0d25e9ecc477b08f) at `f119334d79981225bc103fae0d25e9ecc477b08f` (MIT). Native source, reimplemented design or build/runtime dependency. mikolaj92 authored sparse.py's radix selection bounded to the visible pools (ashhart/TensorFold f119334d), compiled into the Triton kernel set (ed10d2f4).

### Prince Canuma (@Blaizzy)

Profile: https://github.com/Blaizzy

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

### PyTorch contributors (pytorch)

Profile: https://github.com/pytorch

- **dependency**, [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/tree/9186a08b2c) at `9186a08b2c` (BSD-3-Clause). Native source, reimplemented design or build/runtime dependency. The PyTorch contributors authored PyTorch (NVIDIA's container build 2.13.0a0+9186a08b2c), which tools/zig's CPU recorders and fixtures use at build time; it is not shipped.

### raullenchai

Profile: https://github.com/raullenchai

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

### Raymond Lucke (@rwl4)

Profile: https://github.com/rwl4

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be) at `029a852f14bcf604ffe9a55bfbd6ce7f50ec57be` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Raymond Lucke (@rwl4) authored the literal-closer design and its test cases (Mia's GLM PR #158, 029a852f), which local integration reimplemented with no code copied (174f4e63, picked as edccdfec).

### SparkInfer/b12x contributors

Profile: https://github.com/tpurtell/sparkinfer-glmrt

- **reference**, [https://github.com/tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) at `dc9c26192375a80b1de7bb28b182fc5bbdbf67c4` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Hugh Madden (@hughmadden; X @dangerm00se) authored the sequence-CTA and BF16 projection-tile contributions in SparkInfer PRs #9/#10; T.J. Purtell (@tpurtell) reviewed and merged them, with the inherited SparkInfer/b12x contributors retained.

### Steve Witham

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

### stevehansen (@stevehansen)

Profile: https://github.com/stevehansen

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) at `90ccb49215d58161f9c6017e9106aeecc45fe762` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the all-route SSE extension and tests (90ccb492) in Mia's GLM-5.3-Flash recipe, after stevehansen's issue106 diagnosis, design and patch; local integration's native SSE transport reimplements them (139aa26f, picked as aa39deb2).

### T.J. Purtell (@wrldsuksgo2mars)

Profile: https://github.com/tpurtell

- **design**, [https://github.com/hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) at `91db3cc6fe672e2724efa3464f63bd31493f63f6` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden (@dangerm00se) and T.J. Purtell (@wrldsuksgo2mars) authored the AFD attention/expert split (glm53f-afd 91db3cc, as our Python TF-AFD 2.15 lane runs it), which the three-node exchange (2b6703d7), the attention role on a GPU (cf594679), prompt lanes (29b1e487) and pairs (a3143b83) reimplement with no code copied.
- **design**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/c05d9939a38e3db66767291b500c97933df6497b) at `c05d9939a38e3db66767291b500c97933df6497b` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden authored cuteafd #22 (merged as c05d9939 in T.J. Purtell's tpurtell/cuteafd): decode graphs keyed by what their launches read, which cuda_afd_graphs.zig reimplements with no code copied (1aee087e).
- **reference**, [https://github.com/tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) at `dc9c26192375a80b1de7bb28b182fc5bbdbf67c4` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Hugh Madden (@hughmadden; X @dangerm00se) authored the sequence-CTA and BF16 projection-tile contributions in SparkInfer PRs #9/#10; T.J. Purtell (@tpurtell) reviewed and merged them, with the inherited SparkInfer/b12x contributors retained.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/b1755f59b43c94d163e7f5e5ea8d823fa981c0d8) at `b1755f59b43c94d163e7f5e5ea8d823fa981c0d8` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell) authored cuteafd draft #137 integrating Hugh Madden (@hughmadden)'s KDA scheduling and rejecting repeated nonnegative recurrent slots.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/d1c1e5527deea9b96c43863979e57022f8eacc2d) at `d1c1e5527deea9b96c43863979e57022f8eacc2d` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell) authored the #131 report on row-dependent two-RTX context-split attention planning and its speculative-equality failure.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/bda0efc81ef43b9aad9fe3f265f5ed92e32a0842) at `bda0efc81ef43b9aad9fe3f265f5ed92e32a0842` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell) authored the #135 diagnostic report locating the first captured GLM attention-output divergence before the experts.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/b2c52c9a14a0c8acc4bf29fa30a0dd6c15c13040) at `b2c52c9a14a0c8acc4bf29fa30a0dd6c15c13040` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell, X @wrldsuksgo2mars) authored cuteafd and the draft-attention/head work read at https://github.com/tpurtell/cuteafd/commit/b2c52c9a14a0c8acc4bf29fa30a0dd6c15c13040; this study reads saved evidence only and ports no implementation.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/95c996632dc9de9eb673284831ea57307d7e0292) at `95c996632dc9de9eb673284831ea57307d7e0292` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell, X @wrldsuksgo2mars) authored cuteafd and the draft-attention/head work read at https://github.com/tpurtell/cuteafd/commit/95c996632dc9de9eb673284831ea57307d7e0292; this study reads saved evidence only and ports no implementation.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/bf3052e2b2495c4067d1e9c63a9084c43c5973bc) at `bf3052e2b2495c4067d1e9c63a9084c43c5973bc` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell, X @wrldsuksgo2mars) authored cuteafd and the draft-attention/head work read at https://github.com/tpurtell/cuteafd/commit/bf3052e2b2495c4067d1e9c63a9084c43c5973bc; this study reads saved evidence only and ports no implementation.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/c5a5c25a9bece0e8fb0fb0e6c633b2ed1c9b13dc) at `c5a5c25a9bece0e8fb0fb0e6c633b2ed1c9b13dc` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell, X @wrldsuksgo2mars) authored cuteafd and the draft-attention/head work read at https://github.com/tpurtell/cuteafd/commit/c5a5c25a9bece0e8fb0fb0e6c633b2ed1c9b13dc; this study reads saved evidence only and ports no implementation.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/c4fd52f6233a92f963bed940d0c27eb2d4764fda) at `c4fd52f6233a92f963bed940d0c27eb2d4764fda` (none). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell, X @wrldsuksgo2mars) authored cuteafd and the draft-attention/head work read at https://github.com/tpurtell/cuteafd/commit/c4fd52f6233a92f963bed940d0c27eb2d4764fda; this study reads saved evidence only and ports no implementation.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/d6bb0772e33e6039e70f788843c363a55aeb00d8) at `d6bb0772e33e6039e70f788843c363a55aeb00d8` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell/@wrldsuksgo2mars) authored cuteafd PR124 fixed canonical BF16 TARGET-head selection at this exact commit.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/63f425447017cf7fec2c3ad540b520f758a76718) at `63f425447017cf7fec2c3ad540b520f758a76718` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell/@wrldsuksgo2mars) authored the cuteafd PR144 diagnostic and retained the fixed-head change at this exact never-merge diagnostic commit.

### taussoe (@taussoe)

Profile: https://github.com/taussoe

- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c) at `fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe (github.com/taussoe) authored the latent MLA cache and kda.cu's three-kernel wide path (prep/step/out) in ashhart/TensorFold fb985b8, copied into zig/kernels/cuda/glm_kda.cu (dd7de994).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c) at `fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe authored latent.py's latent MLA cache (ashhart/TensorFold fb985b8), compiled into the Triton kernel set (ed10d2f4).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/d6ce24fead9d9cc165728941e3f2aabf8c994553) at `d6ce24fead9d9cc165728941e3f2aabf8c994553` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe authored sparse.py's DSA scores for any index head count (ashhart/TensorFold d6ce24fe), compiled into the Triton kernel set (ed10d2f4).

### tonyd2wild (@tonyd2wild)

Profile: https://github.com/tonyd2wild

- **reference**, [https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark/tree/062b6ce13e0fa9bead00f58d1cb0cfec7bcc080b) at `062b6ce13e0fa9bead00f58d1cb0cfec7bcc080b` (none). Research/benchmark reference only; no code taken for v3.0. tonyd2wild's GLM-5.3-Flash TensorFold repo has no licence; it was read only, and no code or design of it is in this tree.

### Triton contributors (triton-lang)

Profile: https://github.com/triton-lang

- **dependency**, [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton/tree/v3.7.1) at `v3.7.1` (MIT). Native source, reimplemented design or build/runtime dependency. The Triton contributors authored the Triton compiler (3.7.1), which builds the AOT Triton kernel set; the compiler is not shipped.

### Turboderp (@turboderp)

Profile: https://github.com/turboderp

- **design**, [https://github.com/turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b) at `f45b84512ba65bca77f7d394399d4b68821fc71b` (MIT). Native source, reimplemented design or build/runtime dependency. Turboderp authored ExLlamaV3's EXL3 format (trellis bitstream, codebooks, fragment order), which TensorFold's kernels and this tree's EXL3 reader (7cecb415) implement separately.

### XGrammar Contributors (mlc-ai)

Profile: https://github.com/mlc-ai

- **dependency**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. The XGrammar Contributors (MLC) authored XGrammar v0.2.8 (mlc-ai/xgrammar 97787376), statically built into local integration's libtfgrammar.so, which the AFD attention role loads from TF_AFD_GRAMMAR_LIB.

### XGrammar contributors (MLC)

Profile: https://github.com/mlc-ai/xgrammar

- **reference**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. The XGrammar contributors (MLC) authored XGrammar; local integration read v0.2.8 (commit 97787376, Apache-2.0) to establish that a one-string guided_choice grammar can emit GLM's non-special markup tokens and that its JSON escape covers the forced text; nothing was taken.
- **code-copied**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. The XGrammar contributors (MLC) authored XGrammar; local integration read v0.2.8 (commit 97787376, Apache-2.0) to establish that a one-string guided_choice grammar can emit GLM's non-special markup tokens and that its JSON escape covers the forced text; nothing was taken.

### Z Lab (z-lab)

Profile: https://github.com/z-lab

- **reference**, [https://github.com/z-lab/dflash](https://github.com/z-lab/dflash) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Z Lab authored dflash/model_mlx.py, which upstream TensorFold's Python engine vendors; that file is not in this tree.

### Z.ai

Profile: https://huggingface.co/zai-org

- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights the GLM-5.3-Flash lane reads (7cecb415). Never shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration head e7c960e8). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia (@MiaAI_lab) authored the EXL3 quantization of GLM-5.3-Flash by Z.ai in Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold revision 078455ff; its exact published licence and card are MIT, with Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda) at `91b69215804f2d2036a8867f6ce44848e278dcda` (MIT). Other weight revision documented for licence history; not selected. Z.ai (zai-org, https://huggingface.co/zai-org) authored the base GLM-5.3-Flash model; its exact published licence at revision 91b69215 is MIT, Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/41ba3c2b82eff6ef5b93debff02f47ae4681d824) at `41ba3c2b82eff6ef5b93debff02f47ae4681d824` (Apache-2.0). Other weight revision documented for licence history; not selected. Mia (@MiaAI_lab) authored the EXL3 quantization of Z.ai GLM-5.3-Flash; at revision 41ba3c2b its published terms are Apache-2.0 for the quantization, with Z.ai MIT retained for the base model in LICENSE-GLM-5.3-Flash.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a, local integration's kit). Never copied or shipped.

### Z.ai (Zhipu AI) (zai-org)

Profile: https://huggingface.co/zai-org

- **reference**, [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at `tokenizer.json sha256 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` (NOASSERTION). Research/benchmark reference only; no code taken for v3.0. Z.ai (zai-org) authored the GLM-5.3-Flash tokenizer; local integration read its added tokens (<tool_call> 154843 ... </arg_value> 154850, special false) to design the forced local integration probes; nothing was taken.


## Other v2.15 link targets, retained

These are original v2.15 targets, including badges and navigation. Relative targets refer to [that tag](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.15), not a claim that an older controller ships in v3.0.

- Original target: `#21-against-mias-recipe-v18`
- Original target: `#hardware`
- Original target: `#operations`
- Original target: `#upstream`
- Original target: `#what-changed-since-21`
- Original target: `#what-the-patches-are`
- Original target: `CHANGELOG.md#21-kept-prompts-in-host-ram-hugh-maddens-expert-prompt-kernels-her-v18-fixes-tensorfold-066`
- Original target: `docs/BENCHMARKS.md#21-in-the-same-window-as-mias-recipe-v18`
- Original target: `docs/BENCHMARKS.md#215s-levers-one-at-a-time`
- Original target: `docs/BENCHMARKS.md#215s-soak-and-failure-drills`
- Original target: `docs/BENCHMARKS.md#21s-levers-one-at-a-time`
- Original target: `docs/BENCHMARKS.md#the-draft-policy`
- Original target: `docs/DESIGN.md#limits`
- Original target: `docs/DESIGN.md#measured-and-off`
- Original target: `docs/DESIGN.md#the-patches`
- Original target: `docs/TROUBLESHOOTING.md#dgx-spark-gb10-memory-safety-read-this-first`
- Original target: `docs/TROUBLESHOOTING.md#serving`
- Original target: `extras/gb10-hostguard/README.md`
- Original target: `extras/watch/README.md`
- Original target: `https://github.com/ashhart/TensorFold/issues/214`
- Original target: `https://github.com/ashhart/TensorFold/issues/214#issuecomment-6010581196`
- Original target: `https://github.com/ashhart/TensorFold/issues/214#issuecomment-6033151997`
- Original target: `https://github.com/ashhart/TensorFold/issues/286`
- Original target: `https://hermes-agent.nousresearch.com`
- Original target: `https://img.shields.io/badge/2%C3%97_DGX_Spark-expert_nodes-2ea44f?style=flat-square`
- Original target: `https://img.shields.io/badge/MCDMA-RoCE_v2_RDMA-0969da?style=flat-square`
- Original target: `https://img.shields.io/badge/RTX_5090-attention_node-76b900?style=flat-square`
- Original target: `https://img.shields.io/badge/TensorFold-v0.6.5_%2B_51_patches-6f42c1?style=flat-square`
- Original target: `https://img.shields.io/badge/license-Apache--2.0-555?style=flat-square`
