# Third-party notices

ping_pong_qualified retained author disclosures (all earlier notices remain verbatim).
Bizuayeu (@Bizuayeu, https://github.com/Bizuayeu/TensorFold) authored the shared verify design at 75131b66c39fdcf55dbd130ccaec166b65394f3d (Apache-2.0), used through shared_verify; no additional source imported here.
Z.AI (@Zai_org, https://huggingface.co/zai-org) authored GLM-5.3-Flash (91b69215804f2d2036a8867f6ce44848e278dcda, MIT). Mia (@MiaAI_lab) authored EXL3 quantization at https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@078455ffe6472f9a52fbc1139f58b9db2881b25c (MIT; served pin unchanged).
XGrammar Contributors (https://github.com/mlc-ai/xgrammar@97787376faee5ed8466cfad57c99855e4ce2f6aa) authored the matcher/rollback API and fixed-w1 grammar (Apache-2.0). Hugging Face tokenizers contributors (https://github.com/huggingface/tokenizers@7f1623b90b5adfb9bc327d4c3468d2f70bbce262) authored the unchanged tokenizer (Apache-2.0); Hugging Face authored the referenced transformers GLM semantics at 7cd73d9df0c14b151c684b708a9f27d8d0349dfe (Apache-2.0).
linux-rdma/rdma-core contributors, Topspin Communications, Intel Corporation, Cisco Systems, Inc., and PathScale, Inc. authored the retained verbs/header components (https://github.com/linux-rdma/rdma-core v50.0, selected BSD-2-Clause/OpenIB.org terms; individual-file alternatives retained). Scott Quarrier (@squarrier, https://github.com/squarrier) authored packaging/documentation at GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks@1bff49e94deb717bd7711480188539d315af89e6 (Apache-2.0).
Zig contributors authored Zig 0.17.0 (https://github.com/ziglang/zig, MIT). Claude Opus 5.5 via Hermes authored the retained shared_verify recorder and CPU runner. GPT-6.1 Sol (OpenAI Codex), via Hermes, authored this qualification and private kit adaptation.
Cross-card shared-ledger design references only, not new code: Yaniv Leviathan, Matan Kalman and Yossi Matias authored speculative acceptance/residual correction (https://arxiv.org/abs/2211.17192v2, CC-BY-4.0); Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper authored speculative sampling correction (https://arxiv.org/abs/2302.01318v1, CC-BY-4.0). No sampled helper is imported here.

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
