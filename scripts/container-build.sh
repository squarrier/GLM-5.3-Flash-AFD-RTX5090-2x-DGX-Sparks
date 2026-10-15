#!/usr/bin/env bash
# CPU-only entry; tested compiler flags preserved from the recorded builds.
set -euo pipefail
mode=$1; arch=$2
cp -a /source/engine /work/engine
chmod -R u+w /work/engine
cd /work/engine
case "$mode" in
 test) exec /zig/zig build test -j1 --summary all ;;
 native)
  sm=120; abi=(-Dgrammar_abi=/source/tools/grammar/src/root.zig)
  if [ "$arch" = aarch64 ]; then sm=121; abi=(); fi
  exec /zig/zig build tf-glm-afd fatbins -Dnvcc="$NVCC" -Dsm="$sm" -Doptimize=safe -Dtarget="$arch-linux-gnu.2.28" "${abi[@]}" --prefix /out -j"$BUILD_JOBS" --summary all ;;
 grammar)
  cp -a /source/tools/grammar /work/grammar
  cd /work/grammar
  exec /zig/zig build grammar -Dxgrammar=/xgrammar -Dtarget=x86_64-linux-gnu.2.28 --prefix /out -j"$BUILD_JOBS" --summary all ;;
 *) exit 2 ;;
esac
