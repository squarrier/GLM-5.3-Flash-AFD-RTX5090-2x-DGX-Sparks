#!/bin/bash
# Shared helpers for every script: load .env (caller exports win), derive the topology, ssh and docker wrappers.
# shellcheck disable=SC2034
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ ! -f "$ROOT/.env" ]; then
  echo "no .env: cp .env.example .env and edit it" >&2; exit 2
fi
# caller exports win over .env
_saved=$(export -p)
set -a; . "$ROOT/.env"; set +a
eval "$_saved"

: "${SSH_USER:?}" "${ATTN_HOST:?}" "${EXPERT0_HOST:?}" "${EXPERT1_HOST:?}" "${AFD_HOME:?}" "${MODEL_DIR:?}"
: "${ATTN_FABRIC_IP:?}" "${EXPERT0_FABRIC_IP:?}" "${EXPERT1_FABRIC_IP:?}" "${TF_COMMIT:?}" "${MCDMA_COMMIT:?}"
ATTN_RDMA_DEV=${ATTN_RDMA_DEV:-mlx5_0}; EXPERT_RDMA_DEV=${EXPERT_RDMA_DEV:-rocep1s0f0}
RDMA_GID_INDEX=${RDMA_GID_INDEX:-3}; RDMA_MTU=${RDMA_MTU:-4096}
MCDMA_LINK0=${MCDMA_LINK0:-x0}; MCDMA_LINK1=${MCDMA_LINK1:-x1}; MCDMA_CTRL_PORT=${MCDMA_CTRL_PORT:-18820}
MCDMA_REQ_MIB=${MCDMA_REQ_MIB:-20}; MCDMA_REP_MIB=${MCDMA_REP_MIB:-36}
# MoE exchanges in flight per expert node (TF_GLM_MCDMA_INFLIGHT): one MCDMA link pair each. Link j of an expert node
# is NAME (j = 0), then NAME-1, NAME-2, with control port MCDMA_CTRL_PORT + j; one connect daemon takes up to six peers.
MCDMA_INFLIGHT=${MCDMA_INFLIGHT:-1}
[[ "$MCDMA_INFLIGHT" =~ ^[1-3]$ ]] || { echo "MCDMA_INFLIGHT must be 1, 2 or 3 (got '$MCDMA_INFLIGHT')" >&2; exit 2; }
API_BIND=${API_BIND:-0.0.0.0}; API_PORT=${API_PORT:-8000}; MASTER_PORT=${MASTER_PORT:-29551}
MODEL_NAME=${MODEL_NAME:-GLM-5.3-Flash-EXL3}; MODEL_ALIAS=${MODEL_ALIAS:-glm-5.3-flash}
CONTEXT=${CONTEXT:-262144}; MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-2}
AFD_CHECK=${AFD_CHECK:-0}; AFD_CHECK_FORWARDS=${AFD_CHECK_FORWARDS:-256}; AFD_TIME=${AFD_TIME:-0}
MIN_FREE_GIB_EXPERT=${MIN_FREE_GIB_EXPERT:-100}; HEALTH_WAIT=${HEALTH_WAIT:-2400}; QUIET_MAX=${QUIET_MAX:-1200}
IMAGE=${IMAGE:-glm-afd-tensorfold:v0.6.5}; BUILD_DIR=${BUILD_DIR:-./build}
case "$BUILD_DIR" in /*) ;; *) BUILD_DIR=$ROOT/${BUILD_DIR#./} ;; esac

declare -A HOST=([attn]=$ATTN_HOST [x0]=$EXPERT0_HOST [x1]=$EXPERT1_HOST)
declare -A FABRIC_IP=([attn]=$ATTN_FABRIC_IP [x0]=$EXPERT0_FABRIC_IP [x1]=$EXPERT1_FABRIC_IP)
declare -A RDMA_DEV=([attn]=$ATTN_RDMA_DEV [x0]=$EXPERT_RDMA_DEV [x1]=$EXPERT_RDMA_DEV)
declare -A LINK=([x0]=$MCDMA_LINK0 [x1]=$MCDMA_LINK1)
declare -A EXPERT_RANK=([x0]=0 [x1]=1)
declare -A MODE=([attn]=${MCDMA_MODE_ATTN:-auto} [x0]=${MCDMA_MODE_EXPERT0:-auto} [x1]=${MCDMA_MODE_EXPERT1:-auto})
declare -A CONTAINER=([attn]=glm-afd-attn [x0]=glm-afd-x0 [x1]=glm-afd-x1)
declare -A NODE_LOG=([attn]=attn [x0]=expert0 [x1]=expert1)
NODES=(attn x0 x1)
EXPERTS=(x0 x1)
BASE_URL="http://$ATTN_HOST:$API_PORT"
# One CUDA extension cache per code tree: a cache shared between trees recompiles inside `serve`, with the weights
# resident (see docs/TROUBLESHOOTING.md). The tree is the pinned commit plus patches/.
TREE_ID=$( { echo "$TF_COMMIT"; cat "$ROOT"/patches/*.patch; } | sha256sum | cut -c1-12)
EXT_DIR=$AFD_HOME/ext-$TREE_ID

SSHO=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
log() { echo "$(date +%T) glm-afd: $*"; }
die() { echo "$(date +%T) glm-afd: ERROR: $*" >&2; exit 1; }
rsh() { local h=$1; shift; timeout "${RSH_TIMEOUT:-60}" ssh "${SSHO[@]}" "$SSH_USER@${HOST[$h]}" "$@"; }
rcp() { local h=$1 dst=$2; shift 2; scp -q "${SSHO[@]}" "$@" "$SSH_USER@${HOST[$h]}:$dst/"; }

# Generic TF_GLM_* env for the attention / expert containers; each value is checked here, so nothing else reaches
# docker's -e.
ENV_RE='^TF_GLM_[A-Z0-9_]+=[A-Za-z0-9._:-]{1,32}$'
envpairs() {   # KNOB VALUE -> "-e K=V ..." on stdout, or a refusal
  local -a pairs; local p out=""
  read -r -a pairs <<< "$2"                # split on blanks, no glob expansion
  for p in "${pairs[@]}"; do
    [[ "$p" =~ $ENV_RE ]] || { echo "$1: '$p' is not a TF_GLM_*=value pair ($ENV_RE)" >&2; return 1; }
    out="$out -e $p"
  done
  printf '%s' "$out"
}

# Containers: detached, never restarted by docker, first in line for the OOM killer (so a runaway container goes
# before sshd does), labelled so ./stop.sh finds every one of them.
drun() {  # NODE NAME docker-run-args...
  local h=$1 name=$2; shift 2
  rsh "$h" "docker run -d --restart=no --oom-score-adj=1000 --label glm-afd=1 --name $name $(printf '%q ' "$@")" >/dev/null
}
dstop() {  # NODE: remove the stack's containers, then wait up to 60 s for the GPU to have no holders
  local h=$1 i
  rsh "$h" 'ids=$(docker ps -aq --filter label=glm-afd=1); [ -z "$ids" ] || docker rm -f $ids >/dev/null'
  for i in $(seq 30); do
    [ -z "$(rsh "$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null')" ] && return 0
    sleep 2
  done
  log "WARN: $h: the GPU still has holders 60 s after the containers were removed"
}

wait_mem() {  # GB10 hands unified memory back a few s after a tenant exits: start only once it is back
  local h=$1 a t=0
  [ "$h" = attn ] && return 0
  while :; do
    a=$(rsh "$h" "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo 0)
    [ "${a:-0}" -ge "$MIN_FREE_GIB_EXPERT" ] && return 0
    [ $t -ge 180 ] && die "$h MemAvailable ${a} GiB < $MIN_FREE_GIB_EXPERT GiB after 180 s: memory not returned, not starting"
    sleep 5; t=$((t + 5))
  done
}

status_connect() {  # the connect daemon's STATUS (one PEER line per link)
  rsh attn "python3 -c \"import socket,time; s=socket.socket(socket.AF_UNIX); s.connect('$AFD_HOME/mcdma/connect.sock'); s.sendall(b'STATUS\\\\n'); time.sleep(0.5); print(s.recv(65536).decode())\"" 2>/dev/null
}
lname() { if [ "$2" = 0 ]; then echo "$1"; else echo "$1-$2"; fi; }   # link j of an expert node: NAME, NAME-1, ...
links() {  # every link name, in the connect daemon's peer order: x0 x1, then x0-1 x1-1, ...
  local j
  for j in $(seq 0 $((MCDMA_INFLIGHT - 1))); do echo "$(lname "$MCDMA_LINK0" "$j") $(lname "$MCDMA_LINK1" "$j")"; done
}
links_up() {  # how many of the expected links the connect daemon reports up
  local s n c=0
  s=$(status_connect || true)
  for n in $(links); do grep -q "^PEER $n up " <<< "$s" && c=$((c + 1)); done
  echo $c
}

stop_daemons() {  # NODE SOCKET...: SHUTDOWN to each, then SIGTERM; never SIGKILL (MCDMA: it would skip the queue-pair teardown)
  local h=$1 sock t=0; shift
  for sock in "$@"; do
    rsh "$h" "test -S $sock && python3 -c \"import socket; s=socket.socket(socket.AF_UNIX); s.connect('$sock'); s.sendall(b'SHUTDOWN\\\\n')\"" 2>/dev/null || true
  done
  while rsh "$h" "pgrep -x mcdma-rpcd > /dev/null"; do
    sleep 1; t=$((t + 1))
    [ $t = 10 ] && rsh "$h" "pkill -TERM -x mcdma-rpcd"
    [ $t -ge 25 ] && { log "WARN: mcdma-rpcd still running on $h after SHUTDOWN+TERM; leaving it (no SIGKILL)"; return 1; }
  done
  return 0
}
