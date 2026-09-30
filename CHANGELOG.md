# Changelog

## 2026-09-30: first public release
- Recipe for glm53f-afd v1.1.0 (`91db3cc`) on 1× RTX 5090 plus 2× DGX Spark: four TP4 expert ranks, two per Spark under CUDA MPS 100, RoCE v2 over ConnectX-7.
- `build.sh`, `download.sh`, `scripts/install.sh`, and `start.sh` / `stop.sh` (ordered controller with `recover`).
- Preflight: memory floor, single GPU tenant, fabric, stale MPS, optional sha256 and power-cap pins.
- Patch 0001: `--served-model-name` (repeatable), API layer only.
- extras: `glm53f-afd-watch` keep-alive and telemetry with `glm53f-afd-report`; `gb10-hostguard`.
- Measured: 73.1 / 83.7 / 76.2 tok/s single stream, 117.6 at C4. KL identical to upstream 4-Spark.
