# Changelog

## V4.0 — source-only lab release

- TensorFold v1.2.8 native hybrid: RTX 5090 attention, two DGX Sparks for routed experts, native RoCE, and pinned system-RAM KV on the attention host.
- Lab profile: 16-bit dense layers/head, BF16 expert inputs and host KV, ordered FP32 weighted returns, 16-row tiles and confidence 0.50 built-in MTP. No external drafter is selected.
- The lab held eight near-capacity windows of 479,232 prompt-plus-output tokens. No comparative performance or quality results are included. The original lifecycle failure remains a limitation.
- Build locally from source under the retained terms. No weights or built CUDA assets ship. See the [previous V3 release](docs/history/v3/CHANGELOG.md) for earlier changes.
