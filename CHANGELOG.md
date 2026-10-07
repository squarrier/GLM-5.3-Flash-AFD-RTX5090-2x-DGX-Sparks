# Changelog

## 2.1: kept prompts in host RAM, Hugh Madden's expert prompt kernels, her v1.8 fixes, TensorFold 0.6.6
- **Same engine, machines, checkpoint and drafter as 2.0**, which stays at the tag
  [`v2.0`](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v2.0). TensorFold v0.6.5
  (`609ca41`) plus 48 patches: 2.0's 30, unchanged byte for byte, and 18 new ones (0031-0048).
- **New patches, on in `.env.example`:**
  - kept prompts' states and evicted prompts' rows in pinned host RAM on the attention node (0032,
    `TF_GLM_KEPT_HOST=1`, `TF_GLM_HOST_CACHE_GIB=24`), written to the design of glm53f-afd's host RAM tier by Hugh
    Madden ([@dangerm00se](https://x.com/dangerm00se)): the KV pool grows from 727,040 to 1,579,008 tokens at
    `CACHE_GIB=8.5` (2.0: 6.5), and the attention host pins 33.35 GiB of RAM for it;
  - prompt chunks filled as pairs while streams decode (0033, `TF_GLM_FILL_PAIRS=1`), this recipe's change to her
    recipe patch 0062's sliced fills;
  - Hugh Madden's expert prompt kernels from glm53f-afd (glm53f-rank's large-M EXL3 kernels, MIT) for the Sparks'
    prompt chunks, on every prompt window from one row (0034, 0035, `TF_GLM_EXPERT_KERNEL=g53` in `EXPERT_ENV`), with
    parts reimplemented from her recipe patches 0009 and 0020; the prebuild covers its new CUDA extension;
  - her v1.8 fixes: a take-over decides which kept prompts stay before it copies any (0042, her 0078 by m-naoki-m),
    capacity refusals answer 429 with `Retry-After: 5` (0043, her 0081 by johnwhited) and a stream whose delivery
    fails ends at once (0045, her 0083 by johnwhited): `TF_GLM_DECIDE_THEN_COPY=1`, `TF_GLM_CAPACITY_STATUS=1`,
    `TF_GLM_DELIVERY_ABORT=1`, as her lane applies them;
  - TensorFold v0.6.6's three commits with their authorship (0046-0048: `--name-priority ID=background` by Philip
    Mossop, [TensorFold#445](https://github.com/ashhart/TensorFold/pull/445); its local-path fix and the 0.6.6 release
    by Ash Hart). This recipe sets no `--name-priority`.
- **New patches, shipped off** (measured: [docs/DESIGN.md](docs/DESIGN.md#measured-and-off)): her 4,096-row prompt
  chunks (0031, her 0004 and 0008: on GB10 the expert nodes' EXL3 grouping kernel cannot launch a prompt window over
  2,812 rows, so it must stay off on this split); her v1.7.1
  kept-prompt patches 0071 and 0074 by E-Zou Shen, 0073 by desy0305 (with johnwhited's delivery-failure handling),
  0075 (reported by Lukas-tek-no-logic) and 0077 (diagnosed by meleesciony), with their tests (0036-0041: the
  ten-client arena cell's prompt rate 8.1% lower); admission at saturation (0044, her 0082 by johnwhited), unset as in
  her lane.
- **Measured** in the lab on the final head, each lever against the configuration without it, with the determinism,
  greedy-bits and memory gates on every boot; and the combined configuration on one boot (2026-10-06): fresh prompts
  2,890 / 2,992 / 3,145 tok/s at 8K / 31K / 62K (2.0: 2,599 / 2,618 / 2,762), a cold 100K prompt's first token in
  32.3 s (2.0: 36.8 s), the arena's 65,535 x 10 cell at 1,147 / 63.9 t/s and 7.8 s to the first token (2.0: 833.9 /
  46.1 and 13.4 s), one-stream decode 75.0 / 80.8 / 83.1 tok/s (2.0: 77.4 / 79.0 / 90.4), AEON-30 23 of 30. Not run
  for this release: a soak, failure drills and the full arena grid.
- **Against Mia's recipe v1.8** (`33b50fd`) on its own two Sparks, in the same window on 2026-10-06 (15:57-19:04 EDT),
  with the same harnesses: one-stream decode 75.7 / 81.5 / 83.2 tok/s against her 58.4 / 65.4 / 66.7 (1.25-1.30x);
  four streams 110.6 against 100.7 (1.10x); fresh 8K / 31K / 62K prompts 2,946 / 3,030 / 3,138 tok/s against 1,875 /
  1,897 / 1,866 (1.57-1.68x); a cold 100K prompt's first token in 32.4 s against 55.4 s (0.58x); the coding-agent load
  236.2 / 71.3 against 185.8 / 52.6 tok/s (1.27x / 1.36x); the arena's 65,535 x 10 cell at 1,198 / 64.2 t/s and 7.7 s
  to the first token, against 618.4 / 37.9 t/s and 17.3 s. Her lane leads on generation per request at ten clients
  (16.0 against 12.7 tok/s) and on pool size (1,710,080 against 1,579,008 tokens).
- **Upstream:** Ash Hart answered [TensorFold#214](https://github.com/ashhart/TensorFold/issues/214) on 2026-10-06:
  TensorFold's Python engine is frozen ([#286](https://github.com/ashhart/TensorFold/issues/286)), cross-machine
  splits now live in the Zig engine's cluster layer, and this layout would be a new placement there once GLM-5.3-Flash
  and CUDA serving reach the Zig engine. This repo stays on the Python engine ([README](README.md#upstream)).
- **Credits** follow Mia's attribution rules ([AGENTS.md](AGENTS.md)), with 2.0 as the before: every 2.0 name is kept,
  and the authors of 2.1's patches are added on the README's first screen with profile, repo and commit: E-Zou Shen,
  desy0305, johnwhited, m-naoki-m and Philip Mossop, with meleesciony and Lukas-tek-no-logic under the table; Hugh
  Madden's and T.J. Purtell's rows name the new code and designs. `tools/patch_credits.py` writes the credit notes of
  the new kinds (her parts reimplemented, TensorFold's own commits), and `tests/test_static.sh` checks them.
- **Not in 2.1:** the watcher (`extras/watch`); it follows with a soak's results.

## 2.0: TensorFold with Mia's GLM work
- **The engine and the weights change.** v1.0 ran glm53f-afd v1.1.0 (`91db3cc`) by Hugh Madden / Turquoise Bay AI
  ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)), with the routed experts as
  *GLM-5.3-Flash TR3 4bpw* by **Local Inference Lab, Inc.** (published by Brandon M. Music, revision `5ab363a8`,
  LicenseRef-LIL-Attribution-1.0 / ShapleyMCG 1.0, attribution required), Z.ai's official FP8 non-expert tensors
  (`eb9eb208`) and IncoAI's DFlash2 drafter (`bf582e4e`). Version 2.0 runs [TensorFold](https://github.com/ashhart/TensorFold)
  v0.6.5 (`609ca41`) by Ash Hart plus 30 patches, with [MCDMA](https://github.com/ashhart/MCDMA) (`e672c14`) for the
  wire, and serves Mia's AI Lab's checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` (`76c0b517`, safetensors
  identical to `078455ff`) on all three hosts; the drafter is unchanged. v1.0 stays at the tag
  [`v1.0`](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/v1.0), and its credits stay in
  [NOTICE.md](NOTICE.md#previous-version-v10), verbatim. v1.0 is still better at prompt processing on long prompts
  (3.1-3.3K against 2,599-2,762 tok/s), holds a bigger KV pool (~1.58M against 727,040 tokens) and decodes four
  streams faster (117.6 against 111.6 tok/s): [what changed since v1.0](README.md#what-changed-since-v10).
- **Licence.** This version's own files are Apache-2.0 ([LICENSE](LICENSE)), the licence of TensorFold from v0.6.0 and
  of Mia's recipe. v1.0 stays MIT.
- **The patches:** the attention/expert split over MCDMA (0001, 0008, 0011, and a GPU test fix, 0021); MiaAI-Lab's
  TensorFold pull requests #243, #285 and #301 as her commits (0002-0007); ports of her recipe patches: 4-bit dense
  weights and the FP8 KV cache (0009, 0010), shared-prefix prompt states (0012), the expert prompt kernels' launch
  order (0013), grouped prompt fills (0014), smooth streaming (0015), layer-sliced prompt fills over a slice-aware wire
  (0016), her decode kernels (0017, with a test follow-up in 0018), her noise-aware DFlash2 draft policy (0019), her GLM
  tool-call fixes and kept reasoning (0020), her chunked KDA prompt kernel with the 64-token prompt grid and prompt
  replay (0023), her EXL3 prompt kernel for the routed experts (0024), her attention-side prompt kernels (0028) and up
  to eight concurrent streams (0030); this recipe's changes to her ported code: the pool's room rule (0025, her 0030)
  and shortest-first prompt order with aging (0029, her 0049); and this recipe's own: prompt chunk pairs (0022), and
  several MoE exchanges in flight to each expert node (0027) and prefill lanes with FP8 wire rows (0026), written to
  Hugh Madden's designs, the FP8 rows in T.J. Purtell's ds41rt row format.
- **Configuration:** `.env.example` turns on her main lane's switches, with measured differences: prompt chunk pairs
  on (`TF_GLM_PREFILL_PAIRS=1`); her draft policy at `fnc5:0.2` (her lane's `fnc7:0.3` measured 7.9% and 5.3% slower
  on a coding-agent load on this split); two exchanges in flight to each Spark (`MCDMA_INFLIGHT=2`: two link daemons a
  Spark, the second on `MCDMA_CTRL_PORT` + 1); her EXL3 prompt kernel on the Sparks (`TF_GLM_EXL3_PROMPT=1`); the room
  rule (`TF_GLM_CACHE_ROOM=1`) and shortest-first order (`TF_GLM_PREFILL_ORDER=sjf`). Defaults: 262,144-token context,
  eight streams with a 64-row verify window (`TF_GLM_MULTI_WINDOW=64`), `DENSE=q4`, `KV=fp8`, a 6.5 GiB prompt cache
  with 20 entries (a 727,040-token pool). Prefill lanes and FP8 wire rows ship off: measured, one ran out of memory on
  the 5090 and the other failed the KL gate. Her attention-side prompt kernels are on (AEON-30: 23 of 30). The prebuild
  covers the two new CUDA extensions and the experts extension's new prompt kernel source.
- **Scripts:** `MCDMA_INFLIGHT` (1-3) runs that many listen daemons on each Spark (`x0`, `x0-1`, ...) and one connect
  daemon with every link as a peer; `./start.sh up` waits for every link, `./stop.sh` sends SHUTDOWN to every link's
  daemon, and `./start.sh logs` tails each. `PARALLEL` takes 1-8.
- **Kept from v1.0:** `AGENTS.md` (byte-identical) and `extras/gb10-hostguard` (its code unchanged). **Dropped** (at
  `v1.0`): glm53f-afd's systemd units, wrappers and preflight (`site/`), `scripts/install.sh`, the watcher
  (`extras/watch/`), the CPU-only Rust build image (`docker/Dockerfile.build`) and glm53f-afd's patch 0001.
- **Credits** follow Mia's attribution rules ([AGENTS.md](AGENTS.md)): every author whose work this version uses is
  named on the README's first screen with profile, repo and commit, in plain authorship wording, and every name and
  link of v1.0 is kept.
- **Measured:** each lever against the configuration without it (her EXL3 prompt kernel, the exchanges in flight, the
  room rule with twenty kept prompts, eight streams, shortest-first order, the draft policy, her attention-side prompt
  kernels, and the two that stay off), v2.0's first staging (decode, prompt processing with and without prompt pairs,
  27 of the 28 Spark Arena cells), the identity and determinism checks, a ~195K-token needle and AEON-30 idle; and the
  shipped configuration on one boot: decode, prompt processing, the ~195K-token needle, three arena cells, AEON-30 (23
  of 30) and 81.6 minutes of mixed load at up to eight requests with no error or restart
  ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)). Not run for this release: the full arena grid as shipped, and failure
  drills.

## 2026-10-04: credits
- Credits follow Mia's attribution rules ([AGENTS.md](AGENTS.md)): every author is named on the README's first screen, with profile, repo and commit, in plain authorship wording. No name or link was removed.
- Added `AGENTS.md`, the credit and attribution rules by Mia ([@MiaAI_lab](https://x.com/MiaAI_lab), [MiaAI-Lab](https://github.com/MiaAI-Lab)), unchanged from [mia-ai.net](https://mia-ai.net/lab/downloads/agents-md-credit-and-attribution).

## 2026-09-30: first public release
- Recipe for glm53f-afd v1.1.0 (`91db3cc`) on 1× RTX 5090 plus 2× DGX Spark: four TP4 expert ranks, two per Spark under CUDA MPS 100, RoCE v2 over ConnectX-7.
- `build.sh`, `download.sh`, `scripts/install.sh`, and `start.sh` / `stop.sh` (ordered controller with `recover`).
- Preflight: memory floor, single GPU tenant, fabric, stale MPS, optional sha256 and power-cap pins.
- Patch 0001: `--served-model-name` (repeatable), API layer only.
- extras: `glm53f-afd-watch` keep-alive and telemetry with `glm53f-afd-report`; `gb10-hostguard`.
- Measured: 73.1 / 83.7 / 76.2 tok/s single stream, 117.6 at C4. KL identical to upstream 4-Spark.
