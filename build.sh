#!/bin/bash
# build.sh — build glm53f-afd (pinned UPSTREAM_REF + patches/) from source, natively:
#   coordinator: glm53f-serve for sm_120 on the x86 RTX host (CUDA 12.9 build image)
#   ranks:       glm53f-rank  for sm_121 on each GB10 Spark  (CUDA 13.0 build image)
# The build containers get no GPU. Binaries land in $GLM_HOME/bin on each host;
# ./scripts/install.sh copies them to /opt/glm53f-afd/bin.
#   ./build.sh [all|coord|ranks]
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"

build_host() {  # host base-image image-tag cuda-arch cargo-package
  local h=$1 base=$2 img=$3 arch=$4 pkg=$5
  log "$h: fetching $UPSTREAM_REPO @ $UPSTREAM_REF"
  rsh "$h" "set -e; mkdir -p $GLM_HOME/src $GLM_HOME/bin; cd $GLM_HOME/src
    [ -d glm53f-afd/.git ] || git clone -q $UPSTREAM_REPO glm53f-afd
    cd glm53f-afd; git fetch -q origin; git checkout -q -f $UPSTREAM_REF; git clean -qfdx -e 'target-*' -e .cargo"
  rstage "$h"
  rcp "$h" "$ROOT"/patches/*.patch "$ROOT/docker/Dockerfile.build"
  log "$h: applying patches, building $pkg for $arch (first build ~10-20 min)"
  RSH_TIMEOUT=3600 rsh "$h" "set -e; cd $GLM_HOME/src/glm53f-afd
    for p in $STAGE_DIR/0*.patch; do git apply --whitespace=nowarn \"\$p\"; done
    docker image inspect $img >/dev/null 2>&1 || docker build -q -t $img --build-arg BASE=$base -f $STAGE_DIR/Dockerfile.build $STAGE_DIR
    docker run --rm -v \"\$PWD\":/src -w /src -e CARGO_HOME=/src/.cargo -e CARGO_TARGET_DIR=/src/target-$arch \
      -e GLM53F_CUDA_ARCH=$arch $img \
      bash -c 'set -e; cargo build --release -p $pkg --features cuda,rdma 2>&1 | tail -3; cargo test --release -p $pkg 2>&1 | grep -E \"^test result|FAILED\" | tail -5'
    install -m755 target-$arch/release/$pkg $GLM_HOME/bin/$pkg
    sha256sum $GLM_HOME/bin/$pkg"
}

what=${1:-all}
case $what in
  all|coord|ranks) ;;
  *) echo "usage: $0 [all|coord|ranks]"; exit 2 ;;
esac
if [ "$what" != ranks ]; then build_host coord "$COORD_BUILD_BASE" "$COORD_IMAGE" sm_120 glm53f-serve; fi
if [ "$what" != coord ]; then
  for s in "${SPARKS[@]}"; do build_host "$s" "$RANK_BUILD_BASE" "$RANK_IMAGE" sm_121 glm53f-rank; done
fi
log "build done. Next: ./download.sh, then ./scripts/install.sh"
