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
# is NAME (j = 0), then NAME-1, NAME-2, NAME-3, with control port MCDMA_CTRL_PORT + j. One connect daemon takes up to
# six peers (MCDMA), so links 0-2 of both nodes go to `connect` and, at 4, link 3 of both to a second one, `connect2`.
MCDMA_INFLIGHT=${MCDMA_INFLIGHT:-1}
[[ "$MCDMA_INFLIGHT" =~ ^[1-4]$ ]] || { echo "MCDMA_INFLIGHT must be 1, 2, 3 or 4 (got '$MCDMA_INFLIGHT')" >&2; exit 2; }
API_BIND=${API_BIND:-0.0.0.0}; API_PORT=${API_PORT:-8000}; MASTER_PORT=${MASTER_PORT:-29551}
MODEL_NAME=${MODEL_NAME:-GLM-5.3-Flash-EXL3}; MODEL_ALIAS=${MODEL_ALIAS:-glm-5.3-flash}
CONTEXT=${CONTEXT:-262144}; MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-2}
AFD_CHECK=${AFD_CHECK:-0}; AFD_CHECK_FORWARDS=${AFD_CHECK_FORWARDS:-256}; AFD_TIME=${AFD_TIME:-0}
MIN_FREE_GIB_EXPERT=${MIN_FREE_GIB_EXPERT:-100}; HEALTH_WAIT=${HEALTH_WAIT:-2400}; QUIET_MAX=${QUIET_MAX:-1200}
STOP_WAIT=${STOP_WAIT:-60}; MEM_WAIT=${MEM_WAIT:-180}
[[ "$STOP_WAIT" =~ ^[1-9][0-9]{0,3}$ ]] || { echo "STOP_WAIT must be seconds, 1-9999 (got '$STOP_WAIT')" >&2; exit 2; }
[[ "$MEM_WAIT" =~ ^[0-9]{1,4}$ ]] || { echo "MEM_WAIT must be seconds, 0-9999 (got '$MEM_WAIT')" >&2; exit 2; }
# On the controller: the control lock and ./stop.sh's stop marker (extras/watch leaves a stopped stack alone)
CTL_STATE=${CTL_STATE:-$HOME/.local/state/glm-afd}
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
# before sshd does), labelled so ./stop.sh finds every one of them. --init makes docker's tini PID 1, so a SIGTERM
# reaches the TF process (as PID 1 itself the expert node, which installs no SIGTERM handler, would never see it).
drun() {  # NODE NAME docker-run-args...
  local h=$1 name=$2; shift 2
  rsh "$h" "docker run -d --init --restart=no --oom-score-adj=1000 --label glm-afd=1 --name $name $(printf '%q ' "$@")" >/dev/null
}
# SIGTERM only, never SIGKILL (a GB10 can strand memory or hang when a GPU process is killed hard): the stack's running
# containers on a node get one `docker kill --signal TERM`, then up to STOP_WAIT s to exit; the exited ones are removed.
# A container still running after that is left running and named (return 1): look at it before anything else.
dstop() {  # NODE
  local h=$1 ids left t=0 i
  ids=$(rsh "$h" 'docker ps -q --filter label=glm-afd=1') || { log "WARN: $h: no answer over SSH: nothing stopped there"; return 1; }
  ids=${ids//$'\n'/ }
  if [ -n "${ids// /}" ]; then
    rsh "$h" "docker kill --signal TERM $ids >/dev/null" || log "WARN: $h: docker kill --signal TERM failed"
    while [ -n "$(rsh "$h" 'docker ps -q --filter label=glm-afd=1' || true)" ] && [ $t -lt "$STOP_WAIT" ]; do
      sleep 2; t=$((t + 2))
    done
  fi
  left=$(rsh "$h" "docker ps --filter label=glm-afd=1 --format '{{.Names}}'" || echo "(no answer)")
  left=${left//$'\n'/ }
  rsh "$h" 'ids=$(docker ps -aq --filter label=glm-afd=1 --filter status=exited --filter status=created --filter status=dead); [ -z "$ids" ] || docker rm $ids >/dev/null' || true
  if [ -n "${left// /}" ]; then
    log "WARN: $h: $left still running ${STOP_WAIT} s after SIGTERM: left running, not killed (docs/TROUBLESHOOTING.md)"
    return 1
  fi
  for i in $(seq 30); do
    [ -z "$(rsh "$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null' || true)" ] && return 0
    sleep 2
  done
  log "WARN: $h: the GPU still has holders 60 s after the containers exited"
}
drm() {  # NODE NAME: remove one of the stack's containers; one SIGTERM first if it still runs, never SIGKILL
  local h=$1 c=$2 t=0
  if _drm_running "$h" "$c"; then
    rsh "$h" "docker kill --signal TERM $c >/dev/null" || true
    while _drm_running "$h" "$c"; do
      [ $t -ge "$STOP_WAIT" ] && { log "WARN: $h: $c still running ${STOP_WAIT} s after SIGTERM: left running, not killed"; return 1; }
      sleep 2; t=$((t + 2))
    done
  fi
  rsh "$h" "docker rm $c >/dev/null 2>&1" || true
}
_drm_running() { [ "$(rsh "$1" "docker inspect -f '{{.State.Running}}' $2 2>/dev/null" || true)" = true ]; }

wait_mem() {  # NODE [soft]: GB10 hands unified memory back a few s after a tenant exits: start only once it is back
  local h=$1 a t=0
  [ "$h" = attn ] && return 0
  while :; do
    a=$(rsh "$h" "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo 0)
    [ "${a:-0}" -ge "$MIN_FREE_GIB_EXPERT" ] && return 0
    if [ $t -ge "$MEM_WAIT" ]; then
      [ "${2:-}" = soft ] && { log "$h MemAvailable ${a} GiB < $MIN_FREE_GIB_EXPERT GiB after $MEM_WAIT s: memory not returned"; return 1; }
      die "$h MemAvailable ${a} GiB < $MIN_FREE_GIB_EXPERT GiB after $MEM_WAIT s: memory not returned, not starting"
    fi
    sleep 5; t=$((t + 5))
  done
}

ctl_lock() {  # one start.sh up/recover/probe or stop.sh at a time on this controller (rc 75 when another one runs)
  [ "${GLM_AFD_LOCKED:-0}" = 1 ] && return 0      # a stop.sh that start.sh runs after a failure holds its lock
  mkdir -p "$CTL_STATE" || die "cannot create $CTL_STATE"
  exec 9>"$CTL_STATE/ctl.lock"
  flock -n 9 || { echo "$(date +%T) glm-afd: another start.sh up/recover/probe or stop.sh is running ($CTL_STATE/ctl.lock)" >&2; exit 75; }
  export GLM_AFD_LOCKED=1
}

connects() {  # the connect daemons on the attention host: links 0-2 on `connect`, link 3 on `connect2` (six peers a daemon)
  echo connect
  [ "$MCDMA_INFLIGHT" -gt 3 ] && echo connect2
  return 0
}
cpeers() {  # CONNECT-DAEMON: its link numbers j (each one link to each expert node)
  local j
  for j in $(seq 0 $((MCDMA_INFLIGHT - 1))); do
    if [ "$1" = connect ] && [ "$j" -lt 3 ]; then echo "$j"; elif [ "$1" = connect2 ] && [ "$j" -ge 3 ]; then echo "$j"; fi
  done
}
status_connect() {  # the connect daemons' STATUS (one PEER line per link)
  local c
  for c in $(connects); do
    rsh attn "python3 -c \"import socket,time; s=socket.socket(socket.AF_UNIX); s.connect('$AFD_HOME/mcdma/$c.sock'); s.sendall(b'STATUS\\\\n'); time.sleep(0.5); print(s.recv(65536).decode())\"" 2>/dev/null || return 1
  done
}
lname() { if [ "$2" = 0 ]; then echo "$1"; else echo "$1-$2"; fi; }   # link j of an expert node: NAME, NAME-1, ...
links() {  # every link name, in the connect daemons' peer order: x0 x1, then x0-1 x1-1, ...
  local j
  for j in $(seq 0 $((MCDMA_INFLIGHT - 1))); do echo "$(lname "$MCDMA_LINK0" "$j") $(lname "$MCDMA_LINK1" "$j")"; done
}
links_up() {  # how many of the expected links the connect daemons report up
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
stop_all_daemons() {  # the connect daemons first, then every link's listen daemon, at any MCDMA_INFLIGHT (1-4) it ran with
  local h rc=0
  stop_daemons attn "$AFD_HOME/mcdma/connect.sock" "$AFD_HOME/mcdma/connect2.sock" || rc=1
  for h in "${EXPERTS[@]}"; do
    stop_daemons "$h" "$AFD_HOME/mcdma/${LINK[$h]}.sock" "$AFD_HOME/mcdma/${LINK[$h]}-1.sock" "$AFD_HOME/mcdma/${LINK[$h]}-2.sock" \
      "$AFD_HOME/mcdma/${LINK[$h]}-3.sock" || rc=1
  done
  return $rc
}
