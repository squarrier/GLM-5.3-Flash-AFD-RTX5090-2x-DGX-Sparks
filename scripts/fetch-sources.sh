#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/lib.sh"
fetch() {
  local url=$1 pin=$2 dir=$3
  [[ $pin =~ ^[0-9a-f]{40}$ ]] || { printf 'Exact 40-character source pin required\n' >&2; exit 2; }
  if [ -d "$dir/.git" ]; then [ "$(git -C "$dir" rev-parse HEAD)" = "$pin" ]; return; fi
  [ ! -e "$dir" ] || { printf 'Refuse to overwrite an existing source directory\n' >&2; exit 2; }
  mkdir -p "$dir"
  git -C "$dir" init -q
  timeout 180 git -C "$dir" fetch --depth=1 "$url" "$pin"
  git -C "$dir" checkout --detach FETCH_HEAD
  [ "$(git -C "$dir" rev-parse HEAD)" = "$pin" ]
}
fetch "$PYTHON_ENGINE_REPO" "$PYTHON_ENGINE_REVISION" "$PYTHON_ENGINE_DIR"
fetch "$XGRAMMAR_REPO" "$XGRAMMAR_REVISION" "$XGRAMMAR_DIR"
timeout 180 git -C "$XGRAMMAR_DIR" submodule update --init --depth=1 3rdparty/dlpack
[ "$(git -C "$XGRAMMAR_DIR/3rdparty/dlpack" rev-parse HEAD)" = "$DLPACK_REVISION" ]
