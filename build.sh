#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
ZIG_BIN=${ZIG_BIN:-zig}
BUILD_DIR=${BUILD_DIR:-"$ROOT/.build"}
ASSET_DIR=${ASSET_DIR:-"$BUILD_DIR/assets"}
RDMA_HEADERS=${RDMA_HEADERS:-/usr/include}
CACHE_DIR=${CACHE_DIR:-"$BUILD_DIR/cache"}
mode=${1:-cpu-test}
mkdir -p -- "$BUILD_DIR" "$CACHE_DIR"
BUILD_DIR=$(cd -- "$BUILD_DIR" && pwd)
CACHE_DIR=$(cd -- "$CACHE_DIR" && pwd)
common=(--build-file "$ROOT/engine/build.zig" --cache-dir "$CACHE_DIR" -Doptimize=small -Dstrip=true -j1 --summary all)
case "$mode" in
 cpu-test)
  exec "$ZIG_BIN" build "${common[@]}" test-glm-524-tile-host test-glm-524-gather-host test-glm-lane-costs-host ;;
 assets)
  mkdir -p -- "$ASSET_DIR"
  "$ZIG_BIN" build "${common[@]}" fatbins -Dnvcc="${NVCC:-/usr/local/cuda/bin/nvcc}" -Dsm=120,121 -Dnative-fp-decode=false --prefix "$BUILD_DIR/kernel-install"
  cp -- "$BUILD_DIR"/kernel-install/fatbin/*.fatbin "$ASSET_DIR/"
  "${NVCC:-/usr/local/cuda/bin/nvcc}" -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_90,code=compute_90 -o "$ASSET_DIR/tp_mailbox.fatbin" "$ROOT/engine/zig/src/fabric/tp/mailbox.cu" ;;
 attention|expert|expert-host)
  test -d "$ASSET_DIR" || { printf '%s\n' 'Compile assets first, or set ASSET_DIR to your verified assets.' >&2; exit 2; }
  target=x86_64-linux-gnu.2.28;step=native
  if [[ "$mode" = expert ]]; then target=aarch64-linux-gnu.2.28;step=glm-expert; fi
  if [[ "$mode" = expert-host ]]; then step=glm-expert; fi
  exec "$ZIG_BIN" build "${common[@]}" "$step" -Dtarget="$target" -Dfatbins="$ASSET_DIR" -Dsm=120,121 -Dnative-fp-decode=false -Dhybrid-headers="$RDMA_HEADERS" --prefix "$BUILD_DIR/install" ;;
 *) printf '%s\n' 'usage: build.sh {cpu-test|assets|attention|expert|expert-host}' >&2;exit 2 ;;
esac
