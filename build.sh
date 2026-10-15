#!/usr/bin/env bash
# Build device code on CPU; no GPU, SSH, install or model load.
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
python3 -B "$REPO_ROOT/scripts/verify-source.py"
mode=${1:-native}; arch=${2:-x86_64}
case "$mode:$arch" in native:x86_64|native:aarch64|test:x86_64|grammar:x86_64) ;; *) printf 'usage: build.sh native x86_64|aarch64, test, grammar\n' >&2; exit 2;; esac
[ -x "$ZIG_DIR/zig" ] || { printf 'Set ZIG_DIR to the pinned toolchain\n' >&2; exit 2; }
printf '%s  %s\n' "$ZIG_SHA256" "$ZIG_DIR/zig" | sha256sum -c -
args=()
if [ "$mode" = native ]; then
  [ -f "$RDMA_VERBS_HEADERS/verbs.h" ] || { printf 'Install rdma-core development headers; a stub is not a serving build\n' >&2; exit 2; }
  args+=(-v "$RDMA_VERBS_HEADERS:/usr/include/infiniband:ro" -v "$RDMA_CORE_HEADERS:/usr/include/rdma:ro")
fi
if [ "$mode" = grammar ]; then
  [ "$(git -C "$XGRAMMAR_DIR" rev-parse HEAD)" = "$XGRAMMAR_REVISION" ]
  [ "$(git -C "$XGRAMMAR_DIR/3rdparty/dlpack" rev-parse HEAD)" = "$DLPACK_REVISION" ]
  args+=(-v "$XGRAMMAR_DIR:/xgrammar:ro")
fi
cache=$BUILD_ROOT/cache/$mode-$arch; out=$BUILD_ROOT/out/$mode-$arch
mkdir -p "$cache" "$out"
timeout "$BUILD_TIMEOUT" docker run --pull=never --rm --runtime=runc --network none \
  --memory "$BUILD_MEMORY" --memory-swap "$BUILD_MEMORY" --cpus "$BUILD_JOBS" \
  --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/cache \
  -e ZIG_GLOBAL_CACHE_DIR=/cache/global -e ZIG_LOCAL_CACHE_DIR=/cache/local \
  -e "NVCC=$NVCC" -e "BUILD_JOBS=$BUILD_JOBS" \
  --tmpfs "/work:rw,exec,size=1g,uid=$(id -u),gid=$(id -g),mode=0700" \
  -v "$REPO_ROOT:/source:ro" -v "$ZIG_DIR:/zig:ro" -v "$cache:/cache" -v "$out:/out" \
  "${args[@]}" --entrypoint /bin/bash "$CUDA_IMAGE" /source/scripts/container-build.sh "$mode" "$arch"
