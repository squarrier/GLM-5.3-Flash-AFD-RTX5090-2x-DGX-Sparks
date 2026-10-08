#!/bin/bash
# build.sh — the patched TensorFold tree, the serving images, MCDMA's link daemon and the CUDA extension caches.
#   tree    on this controller: TensorFold at TF_TAG (checked against TF_COMMIT), then patches/ in order with
#           `git am` (authors and messages kept) -> $BUILD_DIR/tensorfold
#   sync    that tree to $AFD_HOME/tree on all three hosts (git archive; the containers mount it read-only)
#   images  $IMAGE on each host from docker/Dockerfile (x86_64 on the attention host, arm64 on the Sparks)
#   mcdma   MCDMA at MCDMA_COMMIT, built natively on each host -> $AFD_HOME/mcdma/{mcdma-rpcd,libmcdma-rpc.so}
#   ext     the CUDA extensions prebuilt with NO model loaded, on all three hosts in parallel (refuses while
#           the stack's containers run: two GPU tenants on a GB10 can hang it)
# Needs git on this controller; docker, git, make, a C compiler and libibverbs-dev on the hosts.
#   ./build.sh [all|tree|sync|images|mcdma|ext]      (all = every step, in that order)
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"

build_tree() {
  local t=$BUILD_DIR/tensorfold p n=0
  mkdir -p "$BUILD_DIR"
  [ -d "$t/.git" ] || git clone -q "$TF_REPO_URL" "$t"
  git -C "$t" am --abort >/dev/null 2>&1 || true
  git -C "$t" fetch -q --tags origin
  [ "$(git -C "$t" rev-parse "$TF_TAG^{commit}")" = "$TF_COMMIT" ] || die "$TF_TAG is not $TF_COMMIT upstream: check the pin"
  git -C "$t" checkout -q -f --detach "$TF_COMMIT"
  git -C "$t" clean -qfdx
  for p in "$ROOT"/patches/*.patch; do
    git -C "$t" apply --check "$p" || die "does not apply on $TF_TAG with the patches before it: $(basename "$p")"
    git -C "$t" -c user.name=glm-afd-build -c user.email=build@localhost am -q "$p" || die "git am failed: $(basename "$p")"
    n=$((n + 1))
  done
  log "tree: $TF_TAG + $n patches = $(git -C "$t" rev-parse --short HEAD) in $t (tree id $TREE_ID)"
}

sync_tree() {  # the tree by commit, as the containers will mount it
  local t=$BUILD_DIR/tensorfold h sha
  sha=$(git -C "$t" rev-parse HEAD 2>/dev/null) || die "no tree in $t: ./build.sh tree first"
  for h in "${NODES[@]}"; do
    git -C "$t" archive HEAD | RSH_TIMEOUT=600 rsh "$h" "rm -rf $AFD_HOME/tree && mkdir -p $AFD_HOME/tree $AFD_HOME/logs $AFD_HOME/mcdma && tar x -C $AFD_HOME/tree && echo $sha > $AFD_HOME/tree/.commit" \
      || die "sync to $h"
    log "sync: $h:$AFD_HOME/tree = ${sha:0:12}"
  done
}

build_images() {
  local h
  for h in "${NODES[@]}"; do
    rsh "$h" "test -f $AFD_HOME/tree/pyproject.toml && mkdir -p $AFD_HOME/docker" || die "$h: no tree at $AFD_HOME/tree (./build.sh sync)"
    rcp "$h" "$AFD_HOME/docker" "$ROOT/docker/Dockerfile"
    log "$h: building $IMAGE from $BASE_IMAGE (the first build pulls the base image)"
    RSH_TIMEOUT=7200 rsh "$h" "docker build -q -t $IMAGE --build-arg BASE_IMAGE=$BASE_IMAGE -f $AFD_HOME/docker/Dockerfile $AFD_HOME/tree" >/dev/null \
      || die "$h: image build failed"
    log "$h: $IMAGE $(rsh "$h" "docker image inspect -f '{{.Id}}' $IMAGE" | cut -c1-19)"
  done
}

build_mcdma() {  # rpc/'s Makefile has -Werror; GCC 11-13 stop on a false format-truncation warning at
  local h        # rpcd_connect.c (ashhart/MCDMA#5 has the fix), so that one warning stays a warning here
  for h in "${NODES[@]}"; do
    RSH_TIMEOUT=1800 rsh "$h" "set -e; mkdir -p $AFD_HOME/src $AFD_HOME/mcdma; cd $AFD_HOME/src
      [ -d MCDMA/.git ] || git clone -q $MCDMA_REPO_URL MCDMA
      cd MCDMA; git fetch -q origin; git checkout -q -f --detach $MCDMA_COMMIT
      make -s -C rpc CFLAGS='-std=c11 -O2 -Wall -Wextra -Werror -Wno-error=format-truncation' >/dev/null
      install -m755 build/rpc/mcdma-rpcd $AFD_HOME/mcdma/
      install -m644 build/rpc/libmcdma-rpc.so $AFD_HOME/mcdma/
      cd $AFD_HOME/mcdma && sha256sum mcdma-rpcd libmcdma-rpc.so" | sed "s#^#$h #" || die "$h: MCDMA build failed"
  done
}

prebuild_ext() {  # no-model prebuild into the tree's own extension dir, stale build locks removed first
  local h st t rc=0 started=()
  for h in "${NODES[@]}"; do
    [ -z "$(rsh "$h" "docker ps -q --filter label=glm-afd=1")" ] || die "$h: the stack's containers are running: ./stop.sh first"
  done
  for h in "${NODES[@]}"; do
    rsh "$h" "test -f $AFD_HOME/tree/src/tensorfold/families/glm5_next/cuda/afd.py" || { log "$h: tree not synced at $AFD_HOME/tree"; rc=1; continue; }
    drm "$h" "glm-afd-prep-$h" || { rc=1; continue; }      # an earlier prebuild left running: SIGTERM, never SIGKILL
    rsh "$h" "mkdir -p $EXT_DIR $AFD_HOME/prep" && rcp "$h" "$AFD_HOME/prep" "$ROOT/scripts/prebuild_ext.py" \
      || { log "$h: cannot stage prebuild_ext.py"; rc=1; continue; }
    if drun "$h" "glm-afd-prep-$h" --gpus all --network none -v "$EXT_DIR:/ext" -e TORCH_EXTENSIONS_DIR=/ext \
        -e TRITON_CACHE_DIR=/ext/triton -e MAX_JOBS=4 -v "$AFD_HOME/tree:/tf:ro" -e PYTHONPATH=/tf/src -v "$AFD_HOME/prep:/p:ro" \
        "$IMAGE" bash -c "find /ext -maxdepth 2 \( -name lock -o -name .ninja_lock \) -print -delete; exec python /p/prebuild_ext.py"
    then started+=("$h"); else log "$h: the prebuild container did not start"; rc=1; fi
  done
  for h in "${started[@]}"; do
    t=0
    while :; do
      st=$(rsh "$h" "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' glm-afd-prep-$h" 2>/dev/null || true)
      case "$st" in
        "exited 0") log "$h: prebuild OK ($EXT_DIR)"; break ;;
        exited*|dead*|"") log "$h: prebuild FAILED (${st:-container gone})"; rc=1; break ;;
      esac
      [ $t -ge 1800 ] && { log "$h: prebuild still running after 1800 s"; rc=1; break; }
      sleep 10; t=$((t + 10))
    done
    rsh "$h" "docker logs glm-afd-prep-$h 2>&1 | grep -E '^(OK|FAIL) |prebuild done' | cut -c1-200" || true
  done
  for h in "${NODES[@]}"; do drm "$h" "glm-afd-prep-$h" || rc=1; done   # SIGTERM to one still running, then removed
  return $rc
}

what=${1:-all}
case $what in
  all) build_tree; sync_tree; build_images; build_mcdma; prebuild_ext ;;
  tree) build_tree ;;
  sync) sync_tree ;;
  images) build_images ;;
  mcdma) build_mcdma ;;
  ext) prebuild_ext ;;
  *) echo "usage: $0 [all|tree|sync|images|mcdma|ext]"; exit 2 ;;
esac
log "build $what done.$([ "$what" = all ] && echo ' Next: ./download.sh, then ./start.sh up')"
