# Source-bearing overlay

`implementation.patch` includes added, context and removed source. `source-lock.json` describes all 148 outputs (144 source paths and four grant/notice aliases), with exact immutable GitHub inputs, blob IDs and executable modes. The source paths include the capacity-planner correction and its coupled test changes. Reconstruction checks the pinned grant/notice alias bytes; `upstream/` retains the official base grants and notices.

This is an affected-file reconstruction only, not the full TensorFold tree, build, licence clearance or runtime test. Cache-epoch labels changed in the consumer and three assertions together; local aliases were removed from added comments. The overlay packaging tests do not compile Zig or qualify hardware. See the root README for authors and limits.

After fetching the pinned input cache, run the ten bounded offline packaging tests (two positive reconstructions under umask 022/077 and eight refusal controls) with outputs in your own scratch directory:

```sh
python3 -B source/test_reconstruct.py --cache ./input-cache --scratch ./scratch --receipt ./packaging-tests.json
```

Create `scratch` first. These tests invoke only Python and Git; they do not compile or execute the reconstructed native engine.

## Historical oMLX notice

Copyright 2025 oMLX contributors. The [historical Apache-2.0 license](../LICENSES/previous/oMLX-Apache-2.0-historical.txt) is retained verbatim from [oMLX c520d7e6124cf6255868e589c6beb75cf4cee8f9](https://github.com/jundot/omlx/tree/c520d7e6124cf6255868e589c6beb75cf4cee8f9). This is a verified historical snapshot, not the original import pin or a determination that every adaptation is covered. The separate mlx-vlm MIT terms and Apple acknowledgments remain in the pinned upstream notices. The overlay includes changes to the native DFlash source and shared interfaces. External checkpoint and drafter weights are not bundled; see the root README.

## TensorFold v1.3.3 successor

The immutable base is official v1.3.3 at `1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`. All 144 implementation outputs are exact selected sealed-port bytes; the port already contains prior public comment/diagnostic and cache-label normalizations. The `hybrid-v1.3.0` cost/cache epoch intentionally remains with its matching assertions and is not the engine pin. The coupled exact-context fixture is included. This reproducibility claim is source/mode identity only, not binary/bit parity, consumer compilation, or a new hardware test. The former v1.3.0 upstream third-party notice is preserved in `v42-notices/`; current upstream terms are under `upstream/`.
