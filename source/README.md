# Source-bearing overlay

`implementation.patch` includes added, context and removed source. `source-lock.json` describes all 147 outputs (143 source paths and four grant/notice aliases), with exact immutable GitHub inputs, blob IDs and executable modes. The source paths include the capacity-planner correction and its coupled test changes. Reconstruction checks the pinned grant/notice alias bytes; `upstream/` retains the official base grants and notices.

This is an affected-file reconstruction only, not the full TensorFold tree, build, licence clearance or runtime test. Cache-epoch labels changed in the consumer and three assertions together; local aliases were removed from added comments. The overlay packaging tests do not compile Zig or qualify hardware. See the root README for authors and limits.

After fetching the pinned input cache, run the nine bounded offline packaging tests (one positive reconstruction and eight refusal controls) with outputs in your own scratch directory:

```sh
python3 -B source/test_reconstruct.py --cache ./input-cache --scratch ./scratch --receipt ./packaging-tests.json
```

Create `scratch` first. These tests invoke only Python and Git; they do not compile or execute the reconstructed native engine.

## Historical oMLX notice

Copyright 2025 oMLX contributors. The [historical Apache-2.0 license](../LICENSES/previous/oMLX-Apache-2.0-historical.txt) is retained verbatim from [oMLX c520d7e6124cf6255868e589c6beb75cf4cee8f9](https://github.com/jundot/omlx/tree/c520d7e6124cf6255868e589c6beb75cf4cee8f9). This is a verified historical snapshot, not the original import pin or a determination that every adaptation is covered. The separate mlx-vlm MIT terms and Apple acknowledgments remain in the pinned upstream notices. The overlay includes changes to the native DFlash source and shared interfaces. External checkpoint and drafter weights are not bundled; see the root README.
