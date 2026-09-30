#!/bin/bash
# Shared helpers for every script: load .env, derive topology, ssh wrappers.
# shellcheck disable=SC2034
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ ! -f "$ROOT/.env" ]; then
  echo "no .env: cp .env.example .env and edit it" >&2; exit 2
fi
# caller exports win over .env
_saved=$(export -p)
set -a; . "$ROOT/.env"; set +a
eval "$_saved"

: "${SSH_USER:?}" "${COORD_HOST:?}" "${SPARK1_HOST:?}" "${SPARK2_HOST:?}" "${GLM_HOME:?}"
RUN_UID=${RUN_UID:-1000}; RUN_GID=${RUN_GID:-1000}
RANK_PORT=${RANK_PORT:-8600}; PEER_PORT=${PEER_PORT:-8601}
API_LISTEN=${API_LISTEN:-0.0.0.0:8000}
API_PORT=${API_LISTEN##*:}
MAX_CONTEXT=${MAX_CONTEXT:-262144}; SLOTS=${SLOTS:-16}
MIN_FREE_GIB_SPARK=${MIN_FREE_GIB_SPARK:-100}; MIN_FREE_GIB_COORD=${MIN_FREE_GIB_COORD:-60}
REQUIRE_GUARD=${REQUIRE_GUARD:-0}

declare -A HOST=([coord]=$COORD_HOST [spark1]=$SPARK1_HOST [spark2]=$SPARK2_HOST)
declare -A RANK_A=([spark1]=0 [spark2]=2) RANK_B=([spark1]=1 [spark2]=3)
declare -A IP_A=([spark1]=$SPARK1_IP_A [spark2]=$SPARK2_IP_A) IP_B=([spark1]=$SPARK1_IP_B [spark2]=$SPARK2_IP_B)
SPARKS=(spark1 spark2)
# rank r listens on its rail address; order r0..r3
RANKS="$SPARK1_IP_A:$RANK_PORT,$SPARK1_IP_B:$RANK_PORT,$SPARK2_IP_A:$RANK_PORT,$SPARK2_IP_B:$RANK_PORT"
PEERS="$SPARK1_IP_A:$PEER_PORT,$SPARK1_IP_B:$PEER_PORT,$SPARK2_IP_A:$PEER_PORT,$SPARK2_IP_B:$PEER_PORT"
COORD_API_HOST=${COORD_API_HOST:-$COORD_HOST}
BASE_URL="http://$COORD_API_HOST:$API_PORT"
FIRST_NAME=${SERVED_NAMES%%,*}; FIRST_NAME=${FIRST_NAME:-glm-5.3-flash}

SSHO=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
log() { echo "$(date +%T) glm53f-afd: $*"; }
die() { echo "$(date +%T) glm53f-afd: ERROR: $*" >&2; exit 1; }
rsh() { local h=$1; shift; timeout "${RSH_TIMEOUT:-60}" ssh "${SSHO[@]}" "$SSH_USER@${HOST[$h]}" "$@"; }
# Upload files into a private (0700) staging dir on the host, never world-writable /tmp:
# files later installed by root must not be swappable by another local user.
STAGE_DIR=$GLM_HOME/.stage
rstage() { rsh $1 "rm -rf $STAGE_DIR && mkdir -p $GLM_HOME && install -d -m700 $STAGE_DIR"; }
rcp() { local h=$1; shift; scp -q "${SSHO[@]}" "$@" "$SSH_USER@${HOST[$h]}:$STAGE_DIR/"; }
