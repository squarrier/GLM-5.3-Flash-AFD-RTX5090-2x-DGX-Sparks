#!/bin/bash
# scripts/install.sh — install the units, wrappers, binaries and per-host env files.
# Idempotent. Writes only /opt/glm53f-afd, /etc/glm53f-afd and two systemd units per host;
# enables nothing (./start.sh orders the lane).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

common_env() {
  cat <<EOF
GLM_HOME=$GLM_HOME
RUN_UID=$RUN_UID
RUN_GID=$RUN_GID
REQUIRE_GUARD=$REQUIRE_GUARD
EOF
}

for h in coord "${SPARKS[@]}"; do
  if [ $h = coord ]; then
    unit=glm53f-afd-coord.service; runner=glm53f-afd-coord-run; bin=glm53f-serve; envname=coord
    env=$(common_env; cat <<EOF
COORD_IMAGE=$COORD_IMAGE
API_LISTEN=$API_LISTEN
RANKS=$RANKS
MAX_CONTEXT=$MAX_CONTEXT
SLOTS=$SLOTS
SERVED_NAMES=${SERVED_NAMES:-}
API_KEY_FILE=${API_KEY_FILE:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
COORD_FABRIC_IPS=$COORD_FABRIC_IPS
MIN_FREE_GIB=$MIN_FREE_GIB_COORD
POWER_LIMIT_W=${POWER_LIMIT_W:-}
SERVE_SHA256=${SERVE_SHA256:-}
DRAFTER_SHA256=${DRAFTER_SHA256:-}
EOF
)
  else
    unit=glm53f-afd-ranks.service; runner=glm53f-afd-ranks-run; bin=glm53f-rank; envname=ranks
    env=$(common_env; cat <<EOF
RANK_IMAGE=$RANK_IMAGE
RANK_A=${RANK_A[$h]}
IP_A=${IP_A[$h]}
RANK_B=${RANK_B[$h]}
IP_B=${IP_B[$h]}
PEERS=$PEERS
RANK_PORT=$RANK_PORT
MIN_FREE_GIB=$MIN_FREE_GIB_SPARK
RANK_BIN_SHA256=${RANK_BIN_SHA256:-}
EOF
)
  fi
  log "$h: installing $unit"
  printf '%s\n' "$env" > "$ROOT/.install-$envname.env"
  rstage $h
  rcp $h "$ROOT/site/systemd/$unit" "$ROOT/site/$runner" "$ROOT/site/glm53f-afd-preflight"
  scp -q "${SSHO[@]}" "$ROOT/.install-$envname.env" "$SSH_USER@${HOST[$h]}:$STAGE_DIR/$envname.env"
  rm -f "$ROOT/.install-$envname.env"
  rsh $h "set -e; sudo -n install -d -m755 /opt/glm53f-afd/bin /etc/glm53f-afd
    sudo -n install -m755 $GLM_HOME/bin/$bin $STAGE_DIR/$runner $STAGE_DIR/glm53f-afd-preflight /opt/glm53f-afd/bin/
    sudo -n install -m600 $STAGE_DIR/$envname.env /etc/glm53f-afd/$envname.env
    sudo -n install -m644 $STAGE_DIR/$unit /etc/systemd/system/$unit
    sudo -n systemctl daemon-reload
    sudo -n /opt/glm53f-afd/bin/glm53f-afd-preflight $envname
    rm -rf $STAGE_DIR"
done
log "installed. Start with ./start.sh"
