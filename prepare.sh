#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
python3 -B "$REPO_ROOT/scripts/verify-source.py"
[ "${1:-}" = sets ] || { printf 'usage: prepare.sh sets\n' >&2; exit 2; }
[ "$(git -C "$PYTHON_ENGINE_DIR" rev-parse HEAD)" = "$PYTHON_ENGINE_REVISION" ]
mkdir -p "$BUILD_ROOT/sets"
timeout "$BUILD_TIMEOUT" docker run --pull=never --rm --runtime=runc --network none \
 --memory "$BUILD_MEMORY" --memory-swap "$BUILD_MEMORY" --cpus "$BUILD_JOBS" \
 --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/out \
 -e PYTHONDONTWRITEBYTECODE=1 -e "BUILD_JOBS=$BUILD_JOBS" \
 -e PYTHONPATH=/tensorfold/src:/source/tools/afd-prep -e TRITON_CACHE_DIR=/out/triton \
 -v "$PYTHON_ENGINE_DIR:/tensorfold:ro" -v "$REPO_ROOT:/source:ro" -v "$BUILD_ROOT/sets:/out" \
 --entrypoint python3 "$AOT_IMAGE" -B /source/tools/afd-prep/build_sets.py
