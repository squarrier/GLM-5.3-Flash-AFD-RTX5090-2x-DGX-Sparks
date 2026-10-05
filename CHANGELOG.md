# Changelog

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
