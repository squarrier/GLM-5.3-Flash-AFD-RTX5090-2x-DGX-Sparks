#!/bin/bash
# download.sh — the weights, from their own Hugging Face repos (this repository ships none):
#   all three hosts: the checkpoint MODEL_REPO@MODEL_REV into MODEL_DIR (~176 GB; skipped when already there,
#                    e.g. on a shared read-only mount)
#   attention host:  the DFlash2 drafter DRAFTER_REPO@DRAFTER_REV into DRAFTER_DIR (CC BY-NC-ND 4.0:
#                    non-commercial use only, no derivatives; leave DRAFTER_DIR empty to skip it)
# Needs `hf` (pip install -U huggingface_hub) on each host.
#   ./download.sh [all|attn|x0|x1]
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
what=${1:-all}
case $what in all|attn|x0|x1) ;; *) echo "usage: $0 [all|attn|x0|x1]"; exit 2 ;; esac

for h in "${NODES[@]}"; do
  [ "$what" = all ] || [ "$what" = "$h" ] || continue
  log "$h: $MODEL_REPO@${MODEL_REV:0:12} -> $MODEL_DIR"
  RSH_TIMEOUT=36000 rsh "$h" "set -e
    if ! { test -s $MODEL_DIR/config.json && test -s $MODEL_DIR/model.safetensors.index.json; }; then
      hf download $MODEL_REPO --revision $MODEL_REV --local-dir $MODEL_DIR >/dev/null
    fi
    test -s $MODEL_DIR/config.json && test -s $MODEL_DIR/model.safetensors.index.json
    echo \"$h: \$(ls $MODEL_DIR/*.safetensors | wc -l) safetensors files, \$(du -sh $MODEL_DIR | cut -f1)\"" \
    || die "$h: checkpoint download failed"
  if [ "$h" = attn ] && [ -n "${DRAFTER_DIR:-}" ]; then
    log "attn: $DRAFTER_REPO@${DRAFTER_REV:0:12} -> $DRAFTER_DIR (CC BY-NC-ND 4.0)"
    RSH_TIMEOUT=7200 rsh attn "set -e
      test -s $DRAFTER_DIR/config.json || hf download $DRAFTER_REPO --revision $DRAFTER_REV --local-dir $DRAFTER_DIR >/dev/null
      test -s $DRAFTER_DIR/config.json" || die "attn: drafter download failed"
  fi
done
log "download done. Next: ./start.sh up"
