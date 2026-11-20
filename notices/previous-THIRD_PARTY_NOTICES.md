# Third-party source records

Generated from the captured provenance render and its active rows. No private task names, evidence paths or history are distributed. A reference remains a reference; an unlicensed or non-permissive reference supplies no copied code. Original inherited notices follow.

## Ash Hart (@ashhart)

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

## Mia (@MiaAI_lab)

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

## Mia (@MiaAI_lab)

Profile: https://huggingface.co/Mia-AiLab

- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights the GLM-5.3-Flash lane reads (7cecb415). Never shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration head e7c960e8). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia (@MiaAI_lab) authored the EXL3 quantization of GLM-5.3-Flash by Z.ai in Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold revision 078455ff; its exact published licence and card are MIT, with Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/41ba3c2b82eff6ef5b93debff02f47ae4681d824) at `41ba3c2b82eff6ef5b93debff02f47ae4681d824` (Apache-2.0). Other weight revision documented for licence history; not selected. Mia (@MiaAI_lab) authored the EXL3 quantization of Z.ai GLM-5.3-Flash; at revision 41ba3c2b its published terms are Apache-2.0 for the quantization, with Z.ai MIT retained for the base model in LICENSE-GLM-5.3-Flash.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a, local integration's kit). Never copied or shipped.

## AEON-7 (@AEON-7)

Profile: https://github.com/AEON-7

- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration's local integration kit runs against the lab endpoint (gate G-AEON) through Hermes's local endpoint runner on Scott's fork commit 9990ec10; nothing of it is copied into the TensorFold fork or the kit.
- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration's kit runs under local integration load (S-AEON) through the canonical aeon_one.sh wrapper on Scott's fork commit 9990ec10; nothing of it is copied into the kit.
- **reference**, [https://github.com/AEON-7/Aeon-Bench-Pod](https://github.com/AEON-7/Aeon-Bench-Pod/tree/8651bcb5869d8e392fbb58d0059d73d346afeeb3) at `8651bcb5869d8e392fbb58d0059d73d346afeeb3` (MIT). Research/benchmark reference only; no code taken for v3.0. AEON-7 (@AEON-7, github.com/AEON-7) authored the AEON Bench Pod (github.com/AEON-7/Aeon-Bench-Pod at 8651bcb5, MIT), whose AEON-30 eval local integration ran on boot M-on (G-AEON) against the lab URL through the local integration kit's canonical aeon_one.sh wrapper on Scott's fork commit 9990ec10; nothing of it is copied.

## Andrey Kolesnikov (@akol1)

Profile: https://github.com/akol1

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/861348862a0f108027575c6e529e527e0bfe754d) at `861348862a0f108027575c6e529e527e0bfe754d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Andrey Kolesnikov (@akol1) authored forward.py's EXL3 experts on the universal path for any bit width (ashhart/TensorFold 86134886), compiled into the Triton kernel set (ed10d2f4).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/720c409114c123ed1efa19f0725cd05acc189786) at `720c409114c123ed1efa19f0725cd05acc189786` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Andrey Kolesnikov (@akol1) authored TensorFold #650's whole-response content condition (ashhart/TensorFold 720c4091), which local integration reimplemented (56b60ccd).

## Apple Inc. (ml-explore)

Profile: https://github.com/ml-explore

- **code-ported**, [https://github.com/ml-explore/mlx](https://github.com/ml-explore/mlx/tree/v0.32.3) at `v0.32.3` (MIT). Native source, reimplemented design or build/runtime dependency. Apple Inc. (ml-explore) authored MLX, whose affine quantization layout and 0.32 kernel arithmetic the inherited Metal kernels of Ash's base reproduce.
- **code-ported**, [https://github.com/ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm/tree/v0.31.3) at `v0.31.3` (MIT). Native source, reimplemented design or build/runtime dependency. Apple Inc. (ml-explore) authored mlx-lm, whose expert selection the inherited Metal routing kernels of Ash's base reproduce.

## Ashish Kulkarni

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

## avlp12

Profile: https://github.com/avlp12

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

## BadAd84 (@BadAd84)

Profile: https://github.com/BadAd84

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) at `90ccb49215d58161f9c6017e9106aeecc45fe762` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the all-route SSE extension and tests (90ccb492) in Mia's GLM-5.3-Flash recipe, after stevehansen's issue106 diagnosis, design and patch; local integration's native SSE transport reimplements them (139aa26f, picked as aa39deb2).
- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/616b840651a428d514607c3ce5524ed5bc4c206d) at `616b840651a428d514607c3ce5524ed5bc4c206d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the tool-call rejection log design of patch 0114 (PR137, 616b8406) in Mia's recipe, which local integration reimplements default-off (19db3dba).
- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/11619a191999398c6392e8079fe4dcc545cc3f1a) at `11619a191999398c6392e8079fe4dcc545cc3f1a` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. BadAd84 authored the FILL_ROWS change (PR #119, 11619a19) in Mia's recipe; local integration compared against it and took no code.
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/0c7b88683c19d14f719287268817f7024341ab75) at `0c7b88683c19d14f719287268817f7024341ab75` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #34 (the draft stop under a confidence floor) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at 0c7b8868, a design this tree reimplements with no code copied (1efdb33a).
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/a82cf8aee2d25e26c8cbde44dc54a985e54671f7) at `a82cf8aee2d25e26c8cbde44dc54a985e54671f7` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #35 (patches/0010, the nucleus draw's row reductions on 1,024 threads) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at a82cf8ae, a design this tree reimplements with no code copied (33d8ab99).
- **design**, [https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/e6a35d22336216b1bd88ab1465139dce9ed9dc12) at `e6a35d22336216b1bd88ab1465139dce9ed9dc12` (MIT). Native source, reimplemented design or build/runtime dependency. BadAd84 authored pull request #36 (the MTP draft layer on FP8) to Mia's Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold at e6a35d22, a design this tree reimplements with no code copied (219461a2).

## bertholomus (@bertholomus)

Profile: https://github.com/bertholomus

- **code-copied**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/dbe0773) at `dbe0773` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus (Albert Lee, BertholomusAI) authored group_count_kernel and group_place_kernel (bertholomus/TensorFold glm-dsa-tp4 dbe0773), carried unchanged into the EXL3 expert kernels (0cb7622f).
- **code-copied**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/90b68e063a0b8bba59aaf13340ef16619e751195) at `90b68e063a0b8bba59aaf13340ef16619e751195` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus authored grouped_mma_kernel and work_list_kernel (bertholomus/TensorFold deepseek-v41-tp2 90b68e0; first in glm-dsa-tp4 2786594), copied into zig/kernels/cuda/exl3/mma/ with one marked change (fd624f2e).
- **design**, [https://github.com/bertholomus/TensorFold](https://github.com/bertholomus/TensorFold/tree/ce20a545ff5ec805651fea28d9d3ffbd8e211f89) at `ce20a545ff5ec805651fea28d9d3ffbd8e211f89` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. bertholomus authored the launch order of deepseek-v41-zig's exl3_experts.zig (bertholomus/TensorFold ce20a545), which the halves' MMA launcher follows (fd624f2e).

## Bizuayeu (@Bizuayeu)

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
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the shared verify design in https://github.com/Bizuayeu/TensorFold at 75131b66c39fdcf55dbd130ccaec166b65394f3d (Apache-2.0), which local integration extends through local integration; his MTP drafting is per stream, this new batched-MTP extension copies none of his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/75131b66c39fdcf55dbd130ccaec166b65394f3d) at `75131b66c39fdcf55dbd130ccaec166b65394f3d` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the shared verify windows design in github.com/Bizuayeu/TensorFold commit 75131b66 (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own State and <=4-row window, without copying his code.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/146e113f6ae3a0b2b45626af6ac627479b4e2127) at `146e113f6ae3a0b2b45626af6ac627479b4e2127` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the shared windows mixer in github.com/Bizuayeu/TensorFold commit 146e113f (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own <=4-row State window, with no code of his copied.
- **design**, [https://github.com/Bizuayeu/TensorFold](https://github.com/Bizuayeu/TensorFold/tree/d3cc052c554790c7767b126ab61c1831cd14fc7a) at `d3cc052c554790c7767b126ab61c1831cd14fc7a` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. Bizuayeu (@Bizuayeu) authored the concurrent streams on their own sequence by slot in github.com/Bizuayeu/TensorFold commit d3cc052c (Apache-2.0); local integration carries local integration's reimplementation through to 16 total rows while retaining each stream's own <=4-row State window, with no code of his copied.

## BlivionIaG (@BlivionIaG)

Profile: https://github.com/BlivionIaG

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/d7b134cdf86a02b7c91d0d137ce733c04ae53cfd) at `d7b134cdf86a02b7c91d0d137ce733c04ae53cfd` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @BlivionIaG authored TensorFold #623 (head d7b134cd): the lane core's optional prefill_step / fillStream and the host's filling queue; Ash Hart merged it as 1184cdbb. local integration ported it (9263173e, picked as cd9627dc).

## BobClawblaw (@BobClawblaw)

Profile: https://github.com/BobClawblaw

- **dependency**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/8ced20188b70fe301d4c9f320430b182a68d216f) at `8ced20188b70fe301d4c9f320430b182a68d216f` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @BobClawblaw authored TensorFold #596 (head 8ced2018, merged d7d5f6f1, in v1.0.6 c080a18e): libtfgrammar and the lane-core grammar that local integration's external grammar library builds.

## chaog992 (@chaog992)

Profile: https://github.com/chaog992

- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/2193d145397500f7b661dd4697240024e1089b68) at `2193d145397500f7b661dd4697240024e1089b68` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. chaog992 authored TensorFold #560 (ashhart/TensorFold 2193d145, merged 17a75f55): a request keeps its own local integration of its key label; local integration picked it as 7eab6148.
- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/1b0268954745802a3ee9b0d1828a51f5bdee0c1d) at `1b0268954745802a3ee9b0d1828a51f5bdee0c1d` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. chaog992 authored TensorFold #594 (711bacf2): nested completion prompt ids past u32 decode as unknown ids; Ash Hart's merge 1b026895 was picked as 647941a3.

## data-angel (@data-angel)

Profile: https://github.com/data-angel

- **reference**, [https://github.com/Enntity/sparkglm](https://github.com/Enntity/sparkglm/pull/46) at `PR #46` (unknown). Research/benchmark reference only; no code taken for v3.0. @data-angel / Enntity's sparkglm PR #46 (RC4.1) was read only by local integration; nothing was taken.

## DLPack contributors (dmlc)

Profile: https://github.com/dmlc

- **dependency**, [https://github.com/dmlc/dlpack](https://github.com/dmlc/dlpack/tree/bbd2f4d32427e548797929af08cfe2a9cbb3cf12) at `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. The DLPack contributors authored DLPack (dmlc/dlpack bbd2f4d3), XGrammar's submodule inside local integration's libtfgrammar.so.

## Enntity (@Enntity)

Profile: https://github.com/Enntity

- **reference**, [https://github.com/Enntity/atlas](https://github.com/Enntity/atlas/tree/ade7b012) at `ade7b012` (AGPL-3.0). Research/benchmark reference only; no code taken for v3.0. Atlas (AGPL-3.0) was read only by local integration; nothing was taken.

## ezoushen (@ezoushen)

Profile: https://github.com/ezoushen

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab) at `3fd60dfc836fab0f1ed0fd4f0afaf2a1d62b2aab` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. ezoushen authored the compaction rule of patch 0074-glm-compact-before-evict (pull request #62, issue #61) in Mia's GLM-5.3-Flash recipe, read at 3fd60dfc; cuda_afd_pool.zig reimplements it with no code copied.

## fly9i (@fly9i)

Profile: https://github.com/fly9i

- **reference**, [https://github.com/fly9i/spark-engine](https://github.com/fly9i/spark-engine/tree/b09a53e) at `b09a53e` (MIT). Research/benchmark reference only; no code taken for v3.0. fly9i's spark-engine v0.1.1 (b09a53e) was studied and benchmarked by local integration; nothing of it is in this tree.

## Hugh Madden (@dangerm00se)

Profile: https://github.com/hughmadden

- **code-copied**, [https://github.com/hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) at `91db3cc6fe672e2724efa3464f63bd31493f63f6` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden (@dangerm00se) authored glm53f-afd's large-M expert kernels and KT3 schedule (hughmadden/glm53f-afd 91db3cc), whose device lines zig/kernels/cuda/g53/ carries as python 2.15 ports them (e5fb7183).
- **design**, [https://github.com/hughmadden/glm53f-afd](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6) at `91db3cc6fe672e2724efa3464f63bd31493f63f6` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden (@dangerm00se) and T.J. Purtell (@wrldsuksgo2mars) authored the AFD attention/expert split (glm53f-afd 91db3cc, as our Python TF-AFD 2.15 lane runs it), which the three-node exchange (2b6703d7), the attention role on a GPU (cf594679), prompt lanes (29b1e487) and pairs (a3143b83) reimplement with no code copied.
- **design**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/c05d9939a38e3db66767291b500c97933df6497b) at `c05d9939a38e3db66767291b500c97933df6497b` (MIT). Native source, reimplemented design or build/runtime dependency. Hugh Madden authored cuteafd #22 (merged as c05d9939 in T.J. Purtell's tpurtell/cuteafd): decode graphs keyed by what their launches read, which cuda_afd_graphs.zig reimplements with no code copied (1aee087e).
- **reference**, [https://github.com/tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) at `dc9c26192375a80b1de7bb28b182fc5bbdbf67c4` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Hugh Madden (@hughmadden; X @dangerm00se) authored the sequence-CTA and BF16 projection-tile contributions in SparkInfer PRs #9/#10; T.J. Purtell (@tpurtell) reviewed and merged them, with the inherited SparkInfer/b12x contributors retained.
- **reference**, [https://github.com/tpurtell/cuteafd](https://github.com/tpurtell/cuteafd/tree/b1755f59b43c94d163e7f5e5ea8d823fa981c0d8) at `b1755f59b43c94d163e7f5e5ea8d823fa981c0d8` (MIT). Research/benchmark reference only; no code taken for v3.0. T.J. Purtell (@tpurtell) authored cuteafd draft #137 integrating Hugh Madden (@hughmadden)'s KDA scheduling and rejecting repeated nonnegative recurrent slots.

## iegamo (@ahinore)

Profile: https://github.com/ahinore

- **reference**, [https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/ahinore/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/2636c1cb3019ede4ab27191299d5ad6843344942) at `2636c1cb3019ede4ab27191299d5ad6843344942` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. iegamo (@ahinore) authored PR164 input literal-added-token marking in this fork/commit; its commit preserves Opus 5.5 as co-author, and all Mia/TensorFold/inherited notices remain intact.

## Jay Leaton (@jayleaton)

Profile: https://github.com/jayleaton

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/4284048d1aca2799e543a7400b3f07519befb515) at `4284048d1aca2799e543a7400b3f07519befb515` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton (@jayleaton) authored the incomplete-UTF8-tail bounds repair of TensorFold #634 (4284048d), adapted into zig/src/server/json_write.zig (419164dc, picked as e3c490e2).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/986c996041e25e8b4a0341a8b54524a9d2248c12) at `986c996041e25e8b4a0341a8b54524a9d2248c12` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton authored the CUDA launch-failure context design of TensorFold #635 (986c9960), adapted as the opt-in bounded launch failure context (59f1310e, picked as 3202352c).
- **design**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/5fa8ad0fd9c53131c599c39f8885117d5a582fda) at `5fa8ad0fd9c53131c599c39f8885117d5a582fda` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Jay Leaton authored the CUDA launch-failure context design of TensorFold #635 (5fa8ad0f), adapted as the opt-in bounded launch failure context (59f1310e, picked as 3202352c).
- **reference**, [https://github.com/jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) at `unversioned reference` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Jay Leaton authored inherited parser/cancellation work and patch 0620 in jayleaton/glm53-tensorfold-spark, credited through Mia's recipe NOTICE; no file of that repo is in this tree (his TensorFold #634/#635 have their own entries).

## jundot (@jundot)

Profile: https://github.com/jundot

- **reference**, [https://github.com/jundot/omlx](https://github.com/jundot/omlx) at `unversioned reference` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. jundot's oMLX vendored the mlx-vlm GLM implementation that upstream TensorFold's Python engine follows; nothing from it is in this tree.

## Jürgen Schmied (@jschmied)

Profile: https://github.com/jschmied

- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.

## Kazuho Oku (@kazuho)

Profile: https://github.com/kazuho

- **dependency**, [https://github.com/kazuho/picojson](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa/3rdparty/picojson) at `mlc-ai/xgrammar@97787376faee5ed8466cfad57c99855e4ce2f6aa:3rdparty/picojson (tree 0d9c226c2e13bada877cd092048988070ac5d232)` (BSD-2-Clause). Native source, reimplemented design or build/runtime dependency. Kazuho Oku authored picojson (Copyright 2009-2010 Cybozu Labs, Inc. and 2011-2014 Kazuho Oku), vendored in XGrammar's 3rdparty/picojson and built into local integration's libtfgrammar.so.

## kky42 (@kky42)

Profile: https://github.com/kky42

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/c66a6cfd9ae16226b30f29df877b0194ae8b3945) at `c66a6cfd9ae16226b30f29df877b0194ae8b3945` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. @kky42 authored the quoted-tag examples of PR134 (c66a6cfd) in Mia's recipe, used as test cases by local integration (19db3dba).

## Lazarus-931

Profile: https://github.com/Lazarus-931

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

## local-inference-lab (@local-inference-lab)

Profile: https://github.com/local-inference-lab

- **design**, [https://github.com/local-inference-lab/b12x](https://github.com/local-inference-lab/b12x/tree/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88) at `2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. local-inference-lab authored b12x's RoCEnante one-shot all-gather protocol, which Mia's fn_roce.cu and roce_proxy.c implement in new code (ported by local integration, b5adb25e).

## local-inference-lab contributors (@local-inference-lab)

Profile: https://github.com/local-inference-lab

- **reference**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/60626dc1b2505375854dc94c06718f7ae67ebda9) at `60626dc1b2505375854dc94c06718f7ae67ebda9` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Mia (@MiaAI_lab/MiaAI-Lab) authored the native GLM EXL3 recipe/port in this pinned repository; Ash Hart and TensorFold contributors authored the engine/arithmetic, Jürgen Schmied authored native CUDA serving/registry, and local-inference-lab authored b12x/RoCEnante.

## Mathieu Fenniak

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

## mikolaj92 (@mikolaj92)

Profile: https://github.com/mikolaj92

- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/b3b8a39119f8da02026b3f2601e02eb411be1339) at `b3b8a39119f8da02026b3f2601e02eb411be1339` (MIT). Native source, reimplemented design or build/runtime dependency. mikolaj92 (github.com/mikolaj92) authored sparse.py's skip of invisible pool tiles (ashhart/TensorFold b3b8a391), compiled into the Triton kernel set (ed10d2f4).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/f119334d79981225bc103fae0d25e9ecc477b08f) at `f119334d79981225bc103fae0d25e9ecc477b08f` (MIT). Native source, reimplemented design or build/runtime dependency. mikolaj92 authored sparse.py's radix selection bounded to the visible pools (ashhart/TensorFold f119334d), compiled into the Triton kernel set (ed10d2f4).

## Prince Canuma (@Blaizzy)

Profile: https://github.com/Blaizzy

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

## PyTorch contributors (pytorch)

Profile: https://github.com/pytorch

- **dependency**, [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/tree/9186a08b2c) at `9186a08b2c` (BSD-3-Clause). Native source, reimplemented design or build/runtime dependency. The PyTorch contributors authored PyTorch (NVIDIA's container build 2.13.0a0+9186a08b2c), which tools/zig's CPU recorders and fixtures use at build time; it is not shipped.

## raullenchai

Profile: https://github.com/raullenchai

- **reference**, [https://github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Prince Canuma's mlx-vlm and its PRs #2030 (Lazarus-931), #2105/#2107 (avlp12) and #2245 (raullenchai) are credited by upstream TensorFold for its Python/Metal GLM engine; none of that Python code is in this tree.

## Raymond Lucke (@rwl4)

Profile: https://github.com/rwl4

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/029a852f14bcf604ffe9a55bfbd6ce7f50ec57be) at `029a852f14bcf604ffe9a55bfbd6ce7f50ec57be` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. Raymond Lucke (@rwl4) authored the literal-closer design and its test cases (Mia's GLM PR #158, 029a852f), which local integration reimplemented with no code copied (174f4e63, picked as edccdfec).

## SparkInfer/b12x contributors

Profile: https://github.com/tpurtell/sparkinfer-glmrt

- **reference**, [https://github.com/tpurtell/sparkinfer-glmrt](https://github.com/tpurtell/sparkinfer-glmrt/tree/dc9c26192375a80b1de7bb28b182fc5bbdbf67c4) at `dc9c26192375a80b1de7bb28b182fc5bbdbf67c4` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. Hugh Madden (@hughmadden; X @dangerm00se) authored the sequence-CTA and BF16 projection-tile contributions in SparkInfer PRs #9/#10; T.J. Purtell (@tpurtell) reviewed and merged them, with the inherited SparkInfer/b12x contributors retained.

## Steve Witham

Profile: https://github.com/py-pdf/pypdf

- **dependency**, [https://github.com/py-pdf/pypdf](https://github.com/py-pdf/pypdf/tree/6.20.0) at `6.20.0` (BSD-3-Clause). Verification-only dependency; not used or shipped by v3.0. Mathieu Fenniak, Ashish Kulkarni and Steve Witham are the copyright authors named by pypdf's BSD-3-Clause LICENSE; local integration uses installed pypdf 6.20.0 (github.com/py-pdf/pypdf) only to read back the own-code PDF artifact and verify its six pages and complete text. No pypdf code is copied into the TensorFold engine, kit or package.

## stevehansen (@stevehansen)

Profile: https://github.com/stevehansen

- **design**, [https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/90ccb49215d58161f9c6017e9106aeecc45fe762) at `90ccb49215d58161f9c6017e9106aeecc45fe762` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. BadAd84 authored the all-route SSE extension and tests (90ccb492) in Mia's GLM-5.3-Flash recipe, after stevehansen's issue106 diagnosis, design and patch; local integration's native SSE transport reimplements them (139aa26f, picked as aa39deb2).

## T.J. Purtell (@wrldsuksgo2mars)

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

## taussoe (@taussoe)

Profile: https://github.com/taussoe

- **code-copied**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c) at `fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe (github.com/taussoe) authored the latent MLA cache and kda.cu's three-kernel wide path (prep/step/out) in ashhart/TensorFold fb985b8, copied into zig/kernels/cuda/glm_kda.cu (dd7de994).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c) at `fb985b8649c8b42c2ac2aba1fd7c6f905bc8413c` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe authored latent.py's latent MLA cache (ashhart/TensorFold fb985b8), compiled into the Triton kernel set (ed10d2f4).
- **code-ported**, [https://github.com/ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/d6ce24fead9d9cc165728941e3f2aabf8c994553) at `d6ce24fead9d9cc165728941e3f2aabf8c994553` (MIT). Native source, reimplemented design or build/runtime dependency. taussoe authored sparse.py's DSA scores for any index head count (ashhart/TensorFold d6ce24fe), compiled into the Triton kernel set (ed10d2f4).

## tonyd2wild (@tonyd2wild)

Profile: https://github.com/tonyd2wild

- **reference**, [https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark/tree/062b6ce13e0fa9bead00f58d1cb0cfec7bcc080b) at `062b6ce13e0fa9bead00f58d1cb0cfec7bcc080b` (none). Research/benchmark reference only; no code taken for v3.0. tonyd2wild's GLM-5.3-Flash TensorFold repo has no licence; it was read only, and no code or design of it is in this tree.

## Triton contributors (triton-lang)

Profile: https://github.com/triton-lang

- **dependency**, [https://github.com/triton-lang/triton](https://github.com/triton-lang/triton/tree/v3.7.1) at `v3.7.1` (MIT). Native source, reimplemented design or build/runtime dependency. The Triton contributors authored the Triton compiler (3.7.1), which builds the AOT Triton kernel set; the compiler is not shipped.

## Turboderp (@turboderp)

Profile: https://github.com/turboderp

- **design**, [https://github.com/turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/f45b84512ba65bca77f7d394399d4b68821fc71b) at `f45b84512ba65bca77f7d394399d4b68821fc71b` (MIT). Native source, reimplemented design or build/runtime dependency. Turboderp authored ExLlamaV3's EXL3 format (trellis bitstream, codebooks, fragment order), which TensorFold's kernels and this tree's EXL3 reader (7cecb415) implement separately.

## XGrammar Contributors (mlc-ai)

Profile: https://github.com/mlc-ai

- **dependency**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Native source, reimplemented design or build/runtime dependency. The XGrammar Contributors (MLC) authored XGrammar v0.2.8 (mlc-ai/xgrammar 97787376), statically built into local integration's libtfgrammar.so, which the AFD attention role loads from TF_AFD_GRAMMAR_LIB.

## XGrammar contributors (MLC)

Profile: https://github.com/mlc-ai/xgrammar

- **reference**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Research/benchmark reference only; no code taken for v3.0. The XGrammar contributors (MLC) authored XGrammar; local integration read v0.2.8 (commit 97787376, Apache-2.0) to establish that a one-string guided_choice grammar can emit GLM's non-special markup tokens and that its JSON escape covers the forced text; nothing was taken.
- **code-copied**, [https://github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar/tree/97787376faee5ed8466cfad57c99855e4ce2f6aa) at `97787376faee5ed8466cfad57c99855e4ce2f6aa` (Apache-2.0). Related study/test-package or later candidate; not newly adopted in v3.0. The XGrammar contributors (MLC) authored XGrammar; local integration read v0.2.8 (commit 97787376, Apache-2.0) to establish that a one-string guided_choice grammar can emit GLM's non-special markup tokens and that its JSON escape covers the forced text; nothing was taken.

## Z Lab (z-lab)

Profile: https://github.com/z-lab

- **reference**, [https://github.com/z-lab/dflash](https://github.com/z-lab/dflash) at `unversioned reference` (MIT). Research/benchmark reference only; no code taken for v3.0. Z Lab authored dflash/model_mlx.py, which upstream TensorFold's Python engine vendors; that file is not in this tree.

## Z.ai

Profile: https://huggingface.co/zai-org

- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights the GLM-5.3-Flash lane reads (7cecb415). Never shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration head e7c960e8). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a). Never copied or shipped.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia (@MiaAI_lab) authored the EXL3 quantization of GLM-5.3-Flash by Z.ai in Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold revision 078455ff; its exact published licence and card are MIT, with Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/91b69215804f2d2036a8867f6ce44848e278dcda) at `91b69215804f2d2036a8867f6ce44848e278dcda` (MIT). Other weight revision documented for licence history; not selected. Z.ai (zai-org, https://huggingface.co/zai-org) authored the base GLM-5.3-Flash model; its exact published licence at revision 91b69215 is MIT, Copyright (c) 2026 Z.AI Co., Ltd.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/41ba3c2b82eff6ef5b93debff02f47ae4681d824) at `41ba3c2b82eff6ef5b93debff02f47ae4681d824` (Apache-2.0). Other weight revision documented for licence history; not selected. Mia (@MiaAI_lab) authored the EXL3 quantization of Z.ai GLM-5.3-Flash; at revision 41ba3c2b its published terms are Apache-2.0 for the quantization, with Z.ai MIT retained for the base model in LICENSE-GLM-5.3-Flash.
- **weights**, [https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold/tree/078455ffe6472f9a52fbc1139f58b9db2881b25c) at `078455ffe6472f9a52fbc1139f58b9db2881b25c` (MIT). Pinned runtime weights, not shipped. Mia's AI Lab (@MiaAI_lab) made the EXL3 quantization Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold (revision 078455ff), with exllamav3, of Z.ai's GLM-5.3-Flash: the weights every local integration lab boot loads on attention-host/expert1-host/expert0-host (local integration's final head 6b0d285a, local integration's kit). Never copied or shipped.

## Z.ai (Zhipu AI) (zai-org)

Profile: https://huggingface.co/zai-org

- **reference**, [https://huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at `tokenizer.json sha256 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` (NOASSERTION). Research/benchmark reference only; no code taken for v3.0. Z.ai (zai-org) authored the GLM-5.3-Flash tokenizer; local integration read its added tokens (<tool_call> 154843 ... </arg_value> 154850, special false) to design the forced local integration probes; nothing was taken.

## Original native third-party notices

# Third-party notices

TensorFold uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` and `cuda/ngram.py` translate
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) are written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## CUDA

CUDA backends use [PyTorch](https://github.com/pytorch/pytorch), BSD-3-Clause, and
[Triton](https://github.com/triton-lang/triton), MIT, supplied by NVIDIA's container rather than bundled.
Dense Qwen implements mlx-lm's model math. CUDA DFlash2 implementations port z-lab's architecture under
the MIT License, Copyright © 2026 Z Lab.

Flash Next implements transformers' model math under the attribution above. Its new DeltaNet kernel
follows flash-linear-attention's numerics, MIT; its NCCL wrapper follows vLLM's stream convention,
Apache-2.0, without copying either implementation.

GLM's CUDA engine implements transformers' `models/glm5_next/modular_glm5_next.py` math, Apache-2.0,
without including that source. Its draft inputs and thinking-off rendering follow the public GLM recipe
from Mia-AiLab without including recipe code.

GLM EXL3 (`families/glm5_next/cuda/exl3.py`, `exl3.cu`, `exl3_mm.py`), the shared EXL3 module
(`src/tensorfold/cuda/exl3/`) and the EXL3 loaders of Qwen3.8-27B and Qwen3.8 Flash Next
(`families/qwen3_5/cuda/exl3_load.py`, `families/qwen4_exp/cuda/exl3.py`) read
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 format: its trellis layout and bitstream, its
"3inst", "mcg" and "mul1" codebooks, its half-integer bit widths and its tensor-core fragment order. Flash
Next's packs also carry ExLlamaV3's n-gram row codec, read as its `ngram_dequant` reads it. ExLlamaV3 uses the
MIT License, Copyright © 2025 Turboderp. TensorFold's decoders and kernels are separate implementations,
checked bit for bit against ExLlamaV3's dequantization. The Zig engine's CUDA kernels `zig/kernels/cuda/exl3/` and
`zig/kernels/cuda/glm_exl3.cu` are copies of that device code from python-0.6 (`zig/tests/cuda/exl3_copies.py`
checks them), and `zig/src/families/glm/cuda_exl3*.zig` launch them and read the same format.
`zig/kernels/cuda/exl3/l2d/` are the same copies with MiaAI-Lab's patch 0128 (L2 prefetch and programmatic dependent
launch in the grouped expert kernel) applied: `patches/0128-glm-full-expert-prefetch.patch` of
[GLM-5.3 EXL3 on 3x DGX Spark with TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold)
at commit a687724, Apache License 2.0, Copyright 2026 Mia's AI Lab (MiaAI-Lab), carried as
`zig/tests/cuda/exl3_l2d.patch` (its header says what was finished by hand). They run only with `TF_GLM_EXL3_L2D` or
`TF_GLM_EXL3_PDL` set.
`zig/kernels/cuda/exl3/x1/` are the same copies with TensorFold 2.15's prompt switches applied (`zig/tests/cuda/exl3_x1.patch`,
the device code of python-0.6.5 `896e05a`, which TensorFold 2.15 runs unchanged): the n-first launch order, the register
prefetch, `prompt_order_kernel`'s item order and one rotated input a row for layers whose experts share one gate/up `suh`,
adapted onto the universal kernel from MiaAI-Lab's
[GLM-5.3-Flash EXL3 2x DGX Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/1576746)
patches 0001-glm-exl3-prompt-experts, 0020-glm-prompt-experts-order and the EXL3 part of 0009-glm-prefill-kernels,
Apache License 2.0, Copyright 2026 MiaAI-Lab; `zig/src/families/glm/cuda_exl3_experts.zig` launches them as python's
`routed` does (its `order_slots` and `shared_suh` test ported unchanged). They run only with `TF_GLM_EXL3_X1=1`.

`zig/kernels/cuda/g53/` (`rank_math.cuh`, `rank_kernels.cuh`, `rank.cu`) holds expert prompt kernels from glm53f-afd,
written by Hugh Madden (@dangerm00se, [github.com/hughmadden](https://github.com/hughmadden)),
[glm53f-afd @ 91db3cc](https://github.com/hughmadden/glm53f-afd/tree/91db3cc6fe672e2724efa3464f63bd31493f63f6), as the
Python TF-AFD 2.15 lane ports them (`families/glm5_next/cuda/g53_rank.cu`, its patch 0034): every device line of that
file unchanged, split at function boundaries (`rank_math.cuh`'s header says how), so the images are the Python lane's
SASS. `crates/glm53f-rank/kernels/exl3_rank.cu`'s plan, its large-M gate/up and down kernels (`gateup_big`, `down_big`),
its fused epilogue and reduce, its unfused epilogue and reduce kernels and its persistent launch are MIT License,
Copyright (c) 2026 Turquoise Bay AI Pty Ltd ([the license text](LICENSES/glm53f-afd-MIT.txt)). That file's EXL3 trellis
decoder (`mcg2`, `decode_tile`), fragment MMA (`mma16816`) and Hadamard butterfly (`fwht128`) come from TensorFold by
Ash Hart (@ashxhart, [github.com/ashhart](https://github.com/ashhart)), TensorFold @ bb4b4a3, MIT License, Copyright (c)
2026 TensorFold contributors, kept with glm53f-afd's `LICENSE.tensorfold`
([the license text](LICENSES/glm53f-afd-LICENSE.tensorfold.txt)). Its TP split of every expert by intermediate channel
follows glmrt by T.J. Purtell (@wrldsuksgo2mars, [github.com/tpurtell](https://github.com/tpurtell)), with no code
copied. Its once-a-row rotation (`rotrows_kernel`) and group order (`order_kernel`) were reimplemented for glm53f-afd,
with no code copied, from MiaAI-Lab's GLM-5.3-Flash EXL3 2x DGX Spark recipe patches 0009 and 0020 (Apache License 2.0,
Copyright 2026 MiaAI-Lab). `zig/src/families/glm/cuda_g53.zig` is the Python lane's host side in Zig: g53_rank.cpp's
checks, g53_rank.cu's launcher (`g53_ffn_launch`, `launch_persistent`) and g53_experts.py's configuration, including
the large-M schedule of glm53f-rank (KT3) that the Python lane runs on windows of 1,536 rows and more (its patches
0049 and 0051). They run only with `TF_GLM_HALVES_KERNEL=g53kt3r`: every window of a prompt forward on an expert half
then takes them, and a prompt row's routed-expert sum has glm53f-rank's arithmetic (outputs change).

Flash Next's optional int8 and int4 KV caches (`families/qwen4_exp/cuda/kvcache.py`) follow the cache quantization scheme of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) `-cq 8` and `-cq 4` (MIT License, Copyright (c) 2025 Turboderp, text below): groups of 32, one fp16 absmax scale per group, the group rotated by a 32-point Hadamard, midpoint-grid codes, `compand_a == 0`. 8-bit stores each code as a signed int8 (`q - 128`). 4-bit stores two unsigned codes per byte, low nibble first (the same bits as ExLlamaV3's little-endian packing, a uint8 tensor rather than their uint32 words). Their dequantizer folds another `1/sqrt(32)` into the scale and applies the unnormalized butterfly on the way out; this cache applies the normalized H32 to the query and to the merged output instead, and leaves the stored codes rotated. Scales match their quantizer bit for bit. Reconstructed values agree within fp16/bf16 rounding (under 0.01 on random groups), not bit for bit. The quantizer and the attention dequant are written for TensorFold and checked against an independent reference of that arithmetic.

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer TensorFold drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

TensorFold ships no model weights. The `z-lab/Qwen3.8-27B-DFlash2` model card states Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Flash Next CUDA image integration

The multimodal rotary and image-feature integration is adapted from MiaAI-Lab's
[Flash Next vision patch 0008](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/a3aa89835022c55ca8e55008c37785954834e04f/patches/0008-flash-next-vision.patch),
MIT License, Copyright (c) 2026 MiaAI-Lab. The license is included in `LICENSES/MiaAI-Lab-MIT.txt`.
The port preserves the v0.5 CUDA execution APIs and adds an offline EXL3 vision adapter.

## Zig engine: third-party code in this tree

NOTICE names the author, repository and commit of each item below. The licence texts are in `LICENSES/`.

- `zig/src/cuda/{roce,comm,link,watch}.zig`, `zig/src/cuda/roce_proxy.c`, `roce_proxy_stub.c`, `zig/kernels/cuda/fn_roce.cu`,
  `tools/zig/glm_aot.py`, `glm_aot_rules.py`, `glm_triton_fixtures.py`, `glm_triton_cases.py` and
  `zig/src/families/glm/cuda_triton.zig` / `cuda_triton_test.zig` are ported from code written by Mia (MiaAI-Lab, X
  @MiaAI_lab) in [Qwen3.8-Flash-Next on 2x DGX Spark with TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold)
  at commit cbfc429, Apache License 2.0, Copyright 2026 MiaAI-Lab. Each file's header says what was changed.
- The FP8 head and q4 dense paths port Mia's patches 0002-glm-dense-fp8 and 0005-glm-dense-q4, and the TP nucleus draw
  her patch 0034, of [GLM-5.3-Flash EXL3 on 2x DGX Spark with TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  at v1.10, commit 549fdc5, Apache License 2.0, Copyright 2026 MiaAI-Lab.
- `zig/kernels/cuda/exl3/mma/` and the parallel grouping kernels in `zig/kernels/cuda/exl3/experts.cu` and
  `exl3/x1/experts.cu` (`zig/tests/cuda/exl3_pgroup.patch`) are copied from code written by bertholomus (Albert Lee,
  BertholomusAI) in [his TensorFold fork](https://github.com/bertholomus/TensorFold) at commits 90b68e0 (branch
  deepseek-v41-tp2) and dbe0773 (branch glm-dsa-tp4), Apache License 2.0.
- `zig/kernels/cuda/glm_kda.cu` and the other `glm_*` kernels are copies of TensorFold python-0.6's device code (ed78d6f)
  by Ash Hart and the TensorFold contributors, including taussoe's three-kernel `kda.cu` wide path (fb985b8): TensorFold's
  Apache License 2.0 from 0.6.0, and the MIT License (`LICENSES/MIT.txt`) for code written before 0.6.0, such as that path.

### GLM EXL3 prompt kernel

`zig/kernels/cuda/exl3/k4/experts_prompt.cu` is the EXL3 prompt-window expert kernel of Mia's (MiaAI-Lab, X @MiaAI_lab)
`patches/0004-glm-prompt-kernels.patch` in
[GLM-5.3-Flash EXL3 on 2x DGX Spark with TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
at commit 1576746, with her patches 0009 and 0020, as the Python TF-AFD 2.15 lane's patch 0024 ports it. Apache
License 2.0, Copyright 2026 MiaAI-Lab. The file's header lists the lines kept and what was changed.

## Grammar library loaded at run time

With `TF_AFD_GRAMMAR_SERVE=1` the AFD attention role loads `libtfgrammar.so`, at its first structured request, from the
absolute path in `TF_AFD_GRAMMAR_LIB` after checking it against `TF_AFD_GRAMMAR_LIB_SHA256`. The library is not part of
this tree or of its binaries; it is deployed beside them with its own licence files. It is a static build of
[XGrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (commit 97787376), Apache License 2.0, Copyright (c) 2024 by
XGrammar Contributors (MLC; its NOTICE is in `LICENSES/xgrammar-NOTICE.txt`), with [DLPack](https://github.com/dmlc/dlpack)
(commit bbd2f4d3) by the DLPack contributors, Apache License 2.0, and picojson, Copyright 2009-2010 Cybozu Labs, Inc.
and Copyright 2011-2014 Kazuho Oku, BSD 2-Clause License (`LICENSES/picojson-BSD-2-Clause.txt`), as XGrammar vendors
it in `3rdparty/picojson`. It wraps them in TensorFold v1.0.6's `tf_grammar.cc` shim (commit c080a18e, Ash Hart, with
@BobClawblaw's #596).

## Weights the GLM-5.3-Flash lane serves

TensorFold ships no model weights. The GLM-5.3-Flash lane reads a local copy of
[Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold)
at revision 078455ffe6472f9a52fbc1139f58b9db2881b25c: Mia's AI Lab's (MiaAI-Lab, X @MiaAI_lab) EXL3 quantization,
made with exllamav3 by Turboderp, of GLM-5.3-Flash by Z.ai ([zai-org](https://huggingface.co/zai-org)). At that
revision the checkpoint's LICENSE is Z.ai's MIT License; from revision 41ba3c2b the quantization is under the Apache
License 2.0 and Z.ai's MIT License is kept for the base model. The draft head is the checkpoint's own MTP layer.

## Source-only CPU/client proof and transitive authorship

XGrammar Contributors (MLC, https://github.com/mlc-ai) authored
https://github.com/mlc-ai/xgrammar at 97787376faee5ed8466cfad57c99855e4ce2f6aa,
Apache-2.0. Ash Hart (@ashhart, X @ashxhart, https://github.com/ashhart)
authored https://github.com/ashhart/TensorFold at
c080a18eeae970a742135ee86325115b0387a1c8, Apache-2.0, including libtfgrammar's
C ABI. The existing ABI and XGrammar Token(id) are exercised by an original
ctypes caller in a CPU/client construction proof. That proof is not a new
engine dependency or binary/library redistribution. Existing runtime/source
pins, inherited notices and external-library boundaries are unchanged.

Hugging Face tokenizers contributors (https://github.com/huggingface) authored
https://github.com/huggingface/tokenizers at
7f1623b90b5adfb9bc327d4c3468d2f70bbce262 (v0.23.1), Apache-2.0. The proof uses
the installed encoder unmodified; no tokenizer code, binary or assets are
newly shipped and no checkpoint or tokenizer asset pin changes.

Hugh Madden / Turquoise Bay AI (@dangerm00se, https://github.com/hughmadden)
authored https://github.com/hughmadden/mimo26f-afd at
bab9fa2f2fc1e22ae67b56fbc1c209278f6a9d79 (MIT). Its plan_parallel is adapted
through glm53f-afd 91db3cc6fe672e2724efa3464f63bd31493f63f6 into g53's route
planner, not merely read. The existing glm53f-afd-MIT.txt retains the same
Copyright (c) 2026 Turquoise Bay AI Pty Ltd and permission.
T.J. Purtell (@tpurtell, @wrldsuksgo2mars, https://github.com/tpurtell)
authored the intermediate-channel split in
https://github.com/tpurtell/glmrt-5.3-1rtx-4spark at
dc6d9b8e1600e001cb1d4228bd911f4df8091f99 (MIT): design retained through
glm53f-afd, without glmrt code copied. His ds41rt
(https://github.com/tpurtell/ds41rt at 3067d0684a0b572d88a17d38ca3f7d10599619b4,
MIT) keeps the AFD/wire design credit; no historical kernel body is newly ported.

linux-rdma/rdma-core contributors (https://github.com/linux-rdma) authored
https://github.com/linux-rdma/rdma-core. Its recorded Ubuntu
50.0-2ubuntu0.2 arm64 components were historical CPU link inputs, not a running
host pin. Native RoCE uses host-provided verbs headers/libibverbs as actual
build/runtime dependencies. The historical verbs.h retains Topspin
Communications, Intel Corporation, Cisco Systems, Inc. and PathScale, Inc.
and its GPL-2.0 or OpenIB.org BSD choice. No library/header binary is shipped;
no blanket component licence or unrecorded host revision is asserted.

Scott Quarrier (@squarrier, https://github.com/squarrier) authored the retained
packaging/docs of https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks
at 1bff49e94deb717bd7711480188539d315af89e6 (v2.0 Apache-2.0; v1.0 MIT).
Every original credit remains; no old runtime or weight adoption is implied.
This original text-only addendum: GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes.

The retained historical rdma-core verbs-header copyright authors are:
Topspin Communications.
Intel Corporation.
Cisco Systems, Inc.
PathScale, Inc.
These are the original header's authors, not a new grant for other components.
This additional original authorship text: GPT-6.1 Sol (OpenAI Codex), via Hermes.

## Shipping-source additions, retained authorship

The following source notes describe when their CPU changes were authored; their historical hardware-NOT-RUN statements are not the later shipping soak verdict. Earlier notices remain intact.

ping_pong_qualified private attribution reconciliation (text only; no new runtime imports).
Z.AI (@Zai_org, https://huggingface.co/zai-org) authored GLM-5.3-Flash at 91b69215804f2d2036a8867f6ce44848e278dcda (MIT). Mia (@MiaAI_lab) authored its EXL3 quantization in https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold at 078455ffe6472f9a52fbc1139f58b9db2881b25c (MIT); later 41ba3c2b Apache-2.0 terms do not change the served pin.
XGrammar Contributors (MLC, https://github.com/mlc-ai/xgrammar) authored the retained matcher/rollback API and fixed-w1 grammar at 97787376faee5ed8466cfad57c99855e4ce2f6aa (Apache-2.0). Hugging Face tokenizers contributors (https://github.com/huggingface/tokenizers) authored the unchanged tokenizer at 7f1623b90b5adfb9bc327d4c3468d2f70bbce262 (Apache-2.0). Hugging Face authored the referenced transformers GLM semantics at 7cd73d9df0c14b151c684b708a9f27d8d0349dfe (Apache-2.0).
linux-rdma/rdma-core contributors, Topspin Communications, Intel Corporation, Cisco Systems, Inc., and PathScale, Inc. authored the retained verbs/header components of https://github.com/linux-rdma/rdma-core v50.0 (selected BSD-2-Clause/OpenIB.org terms; individual-file alternatives retained). Scott Quarrier (@squarrier, https://github.com/squarrier) authored packaging/documentation of GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks at 1bff49e94deb717bd7711480188539d315af89e6 (Apache-2.0).
Zig contributors (https://github.com/ziglang/zig) authored the pinned 0.17.0 compiler and standard library (MIT). Opus 5.5 via Hermes authored the retained shared_verify recorder and CPU runner; GPT-6.1 Sol (OpenAI Codex), via Hermes, authored this qualification and private kit adaptation.
Shared-ledger cross-card design disclosures, NOT code imported here: Yaniv Leviathan, Matan Kalman and Yossi Matias authored speculative acceptance/residual correction (https://arxiv.org/abs/2211.17192v2, CC-BY-4.0). Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper authored speculative sampling correction (https://arxiv.org/abs/2302.01318v1, CC-BY-4.0). No sampled helper or instrumentation branch is included in ping_pong_qualified.
Every inherited NOTICE paragraph and licence below is preserved verbatim.

ping_pong_fixture local CPU causal fixture work (2026-10-14): per-Harness preallocated
scalar tracing and per-proxy initial-doorbell-cursor acknowledgement. The
observer never reads the watcher's non-atomic Rule and is not its step input.
Ash Hart (@ashhart / @ashxhart, https://github.com/ashhart) authored TensorFold
at https://github.com/ashhart/TensorFold commit
f8fe17d24629aedabf90bbf78279dd776e6d62e7 (Apache-2.0); the inherited exchange,
Mia, Bizuayeu, Hugh Madden, tpurtell, Turboderp, NVIDIA and XGrammar notices
remain unchanged. This fixture instrumentation and acknowledgement repair:
GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes. No product
watcher, CUDA kernel, wire or mathematical change; historical failures remain
FAIL and broad/final acceptance is not authorized by this observation.

ping_pong local CPU change (2026-10-13), default OFF:
TF_AFD_USERS_PING_PONG=1 needs shared drafts, TF_AFD_PAIRS=1 and
TF_AFD_INFLIGHT=2. Two independent user micro-batches run on one CUDA stream
in the existing mode-7 pair protocol, each at <=8 rows, with independent
State, acceptance and rollback; the halves' source/wire definitions do not
change. Bizuayeu (@Bizuayeu, https://github.com/Bizuayeu) authored the
shared-window design at https://github.com/Bizuayeu/TensorFold commit
75131b66c39fdcf55dbd130ccaec166b65394f3d (Apache-2.0), used here through
shared_verify. Hugh Madden (@dangerm00se, https://github.com/hughmadden) authored
the inherited AFD exchange design at https://github.com/hughmadden/glm53f-afd
commit 91db3cc6fe672e2724efa3464f63bd31493f63f6 (MIT), used through qualification_c533375f's
scheduler. Ash Hart (@ashhart / @ashxhart) authored TensorFold, its lane
core and MCDMA; Mia (@MiaAI_lab) authored the GLM EXL3 recipe and MTP
drafter; T.J. Purtell (tpurtell, @wrldsuksgo2mars, cuteafd, MIT) authored
the disaggregated inference implementation; Turboderp authored EXL3.
No outside code copied. GPT-6.1 Sol (OpenAI Codex; planning and execution),
via Hermes, authored this change. Hardware NOT RUN; no adoption. All
inherited notices remain verbatim.

batched_drafts local CPU extension (2026-10-13) on 6b0d285a: default-off
TF_AFD_USERS_BATCHED_DRAFTS=1 batches active users' MTP rows into one forward
per draft depth, with bounded eight-row absorb waves, private MTP hidden per
slot and each user's own backlog/cache/DSA state. Requires
TF_AFD_USERS_SHARED_DRAFTS=1; verification, acceptance and backbone rollback
are unchanged. Bizuayeu (@Bizuayeu, https://github.com/Bizuayeu) authored the
shared verify design this extends through shared_verify, in
https://github.com/Bizuayeu/TensorFold at 75131b66c39fdcf55dbd130ccaec166b65394f3d
(Apache-2.0); his drafting is per stream, and this batched-MTP extension copies
none of his code. Ash Hart (@ashhart / @ashxhart) authored TensorFold and MTP;
Mia (@MiaAI_lab) authored the EXL3 weights and MTP drafter; tpurtell
(@wrldsuksgo2mars, cuteafd, MIT) and Hugh Madden (@dangerm00se) authored the
AFD split; Turboderp authored EXL3. Earlier user_drafts/shared_verify work: Opus 5.5
(Anthropic), via Hermes. batched_drafts initial receipt analysis/worktree: Opus
5.5 (Anthropic), via Hermes; this implementation and qualification: GPT-6.1
Sol (OpenAI Codex), via Hermes. CPU only; hardware NOT RUN; no adoption or
public/production authority. Every earlier notice and licence is retained.

Scope note for historical ledger entries: the entry describing the new batched-MTP extension is an authorship-phase receipt for code now included/enabled at 1fab9035. Entries describing the separate 16-physical-row extension remain excluded. Their source authors and original entry text are retained.
