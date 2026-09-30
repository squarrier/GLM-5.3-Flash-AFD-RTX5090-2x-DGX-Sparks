#!/bin/bash
# download.sh — fetch the weights and cut the rank images.
#   coordinator host: official FP8 non-expert tensors (~15.5 GB) + the DFlash2 drafter (~2.3 GB)
#   SPARK1: the EXL3 4bpw checkpoint (~164 GB) unless EXL3_CHECKPOINT already exists,
#           then `glm53f-rank slice` cuts all four TP4 rank directories (CPU only, ~2-5 min)
#           and copies r2,r3 to SPARK2 over the fabric; every rank dir is verified.
# Needs `hf` (huggingface_hub CLI) on SPARK1 and the coordinator, and a built
# glm53f-rank on SPARK1 ($GLM_HOME/bin, from ./build.sh).
#   ./download.sh [all|coord|ranks]
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
what=${1:-all}

if [ "$what" != ranks ]; then
  log "coord: FP8 non-expert tensors from $BASE_REPO@$BASE_REV (upstream scripts/fetch_tensors.py)"
  RSH_TIMEOUT=7200 rsh coord "set -e; cd $GLM_HOME/src/glm53f-afd
    python3 scripts/fetch_tensors.py --repo $BASE_REPO --revision $BASE_REV --out $GLM_HOME/coordinator --select nonexpert
    hf download $DRAFTER_REPO --revision $DRAFTER_REV --local-dir $GLM_HOME/drafter >/dev/null
    sha256sum $GLM_HOME/drafter/model.safetensors"
fi

if [ "$what" != coord ]; then
  log "spark1: EXL3 checkpoint $EXL3_REPO@$EXL3_REV -> $EXL3_CHECKPOINT"
  RSH_TIMEOUT=36000 rsh spark1 "set -e
    [ -f $EXL3_CHECKPOINT/model.safetensors.index.json ] || hf download $EXL3_REPO --revision $EXL3_REV --local-dir $EXL3_CHECKPOINT >/dev/null
    mkdir -p $GLM_HOME/ranks
    for r in 0 1 2 3; do
      [ -f $GLM_HOME/ranks/r\$r/manifest.txt ] || $GLM_HOME/bin/glm53f-rank slice --checkpoint $EXL3_CHECKPOINT --rank \$r --out $GLM_HOME/ranks/r\$r --source $EXL3_REPO@$EXL3_REV &
    done; wait
    for r in 0 1 2 3; do $GLM_HOME/bin/glm53f-rank verify --rank \$r --dir $GLM_HOME/ranks/r\$r | tail -1; done"
  log "spark1 -> spark2: copying r2,r3 (2 x 38.4 GB) over the fabric"
  RSH_TIMEOUT=7200 rsh spark1 "set -e; ssh -o BatchMode=yes $SSH_USER@$SPARK2_IP_A mkdir -p $GLM_HOME/ranks
    rsync -a $GLM_HOME/ranks/r2 $GLM_HOME/ranks/r3 $SSH_USER@$SPARK2_IP_A:$GLM_HOME/ranks/"
  RSH_TIMEOUT=600 rsh spark2 "for r in 2 3; do $GLM_HOME/bin/glm53f-rank verify --rank \$r --dir $GLM_HOME/ranks/r\$r | tail -1; done"
  log "optional: r2,r3 on spark1 are no longer needed there (rm -r $GLM_HOME/ranks/r2 $GLM_HOME/ranks/r3 frees 77 GB)"
fi
log "download done. Next: ./scripts/install.sh"
