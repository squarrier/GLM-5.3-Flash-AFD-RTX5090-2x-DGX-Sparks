#!/bin/bash
# start.sh — bring the stack up and check it; also a read-only check, status, logs and a smoke reply.
#   up      check; the CUDA extensions prebuilt with no model loaded (./build.sh ext); the MCDMA link daemons
#           (listen ends on the Sparks, MCDMA_INFLIGHT links each, then the connect end on the attention host, every
#           link up); the attention node, then the two expert nodes; the health wait (fails fast when a container
#           exits or the start goes quiet); the startup lines (the MCDMA wire must be named); one real reply. Any
#           failure after the daemons start takes the whole stack down again (./stop.sh).
#   check   read-only preflight: image, tree, MCDMA build, checkpoint and drafter, RDMA ports, no GPU tenant,
#           no daemons or containers left over
#   probe   (optional, no model) TensorFold's MCDMA stream-memory-op probe on each GPU, one host at a time:
#           STREAMOPS PASS means the exchange can be GPU-driven there (MCDMA_MODE_* auto picks it by itself)
#   status  containers, /health, the MCDMA links
#   logs    the tails of the three node logs and the daemons' logs
#   smoke   one real completion, thinking off
#   ./start.sh [up|check|probe|status|logs|smoke]
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"

PARALLEL=${PARALLEL:-}; DENSE=${DENSE:-}; KV=${KV:-}; MAX_TOKENS=${MAX_TOKENS:-}
CACHE_GIB=${CACHE_GIB:-}; CACHE_ENTRIES=${CACHE_ENTRIES:-}
case "$PARALLEL" in ""|[1-8]) ;; *) die "PARALLEL must be empty or 1..8 (got '$PARALLEL')" ;; esac
case "$DENSE" in ""|bf16|q4) ;; *) die "DENSE must be empty, bf16 or q4 (got '$DENSE')" ;; esac
case "$KV" in ""|bf16|fp8) ;; *) die "KV must be empty, bf16 or fp8 (got '$KV')" ;; esac
[[ -z "$MAX_TOKENS" || "$MAX_TOKENS" =~ ^[1-9][0-9]{0,6}$ ]] || die "MAX_TOKENS must be empty or a token count (got '$MAX_TOKENS')"
[[ -z "$CACHE_GIB" || "$CACHE_GIB" =~ ^[0-9]{1,3}(\.[0-9]{1,2})?$ ]] || die "CACHE_GIB must be empty or GiB like 8 or 7.5 (got '$CACHE_GIB')"
[[ -z "$CACHE_ENTRIES" || "$CACHE_ENTRIES" =~ ^[1-9][0-9]{0,3}$ ]] || die "CACHE_ENTRIES must be empty or a count (got '$CACHE_ENTRIES')"
ATTN_ENV_E=$(envpairs ATTN_ENV "${ATTN_ENV:-}") || exit 2
EXPERT_ENV_E=$(envpairs EXPERT_ENV "${EXPERT_ENV:-}") || exit 2
case " ${ATTN_ENV:-} " in *" TF_GLM_MCDMA_INFLIGHT="*) die "set MCDMA_INFLIGHT, not TF_GLM_MCDMA_INFLIGHT in ATTN_ENV: the link daemons follow it" ;; esac
PAR_ARG=""; [ -n "$PARALLEL" ] && PAR_ARG="--parallel $PARALLEL"
MAXTOK_ARG=""; [ -n "$MAX_TOKENS" ] && MAXTOK_ARG="--max-tokens $MAX_TOKENS"

fail() {  # after the daemons start: the logs, then the whole stack down, then stop
  cmd_logs || true
  log "taking the stack down after the failure"
  "$ROOT/stop.sh" all >/dev/null 2>&1 || true
  die "$*"
}

cmd_check() {
  local h bad=0
  for h in "${NODES[@]}"; do
    rsh "$h" "docker image inspect $IMAGE >/dev/null 2>&1" || { log "$h: no image $IMAGE (./build.sh images)"; bad=1; }
    rsh "$h" "test -f $AFD_HOME/tree/src/tensorfold/families/glm5_next/cuda/afd.py" || { log "$h: no patched tree at $AFD_HOME/tree (./build.sh sync)"; bad=1; }
    rsh "$h" "test -x $AFD_HOME/mcdma/mcdma-rpcd && test -s $AFD_HOME/mcdma/libmcdma-rpc.so" || { log "$h: no MCDMA build (./build.sh mcdma)"; bad=1; }
    rsh "$h" "test -s $MODEL_DIR/config.json && test -s $MODEL_DIR/model.safetensors.index.json" || { log "$h: checkpoint $MODEL_DIR not readable (./download.sh)"; bad=1; }
    rsh "$h" "ibv_devinfo -d ${RDMA_DEV[$h]} | grep -q PORT_ACTIVE" || { log "$h: RDMA device ${RDMA_DEV[$h]} is not PORT_ACTIVE"; bad=1; }
    if rsh "$h" "pgrep -x mcdma-rpcd" >/dev/null; then log "$h: an mcdma-rpcd already runs (./stop.sh)"; bad=1; fi
    [ -z "$(rsh "$h" 'docker ps -aq --filter label=glm-afd=1')" ] || { log "$h: the stack's containers exist (./stop.sh)"; bad=1; }
    [ -z "$(rsh "$h" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null')" ] \
      || { log "$h: another process holds the GPU: one GPU tenant per host"; bad=1; }
    log "$h: MemAvailable $(rsh "$h" "awk '/MemAvailable/{printf \"%d\", \$2/1048576}' /proc/meminfo") GiB; tree $(rsh "$h" "cut -c1-12 $AFD_HOME/tree/.commit 2>/dev/null" || echo -); mailboxes left: $(rsh "$h" 'ls /dev/shm/mcdma-rpc.* 2>/dev/null | tr "\n" " "' || true)"
  done
  if [ -n "${DRAFTER_DIR:-}" ]; then
    rsh attn "test -s $DRAFTER_DIR/config.json" || { log "attn: drafter $DRAFTER_DIR not readable (./download.sh attn)"; bad=1; }
  fi
  [ $bad = 0 ] || die "check failed: nothing started"
  log "check OK (tree id $TREE_ID, extensions in $EXT_DIR)"
}

daemons_up() {
  local h n t j peers=""
  for h in "${EXPERTS[@]}"; do
    for j in $(seq 0 $((MCDMA_INFLIGHT - 1))); do   # one listen daemon per link: link j's control port is MCDMA_CTRL_PORT + j
      n=$(lname "${LINK[$h]}" "$j")
      rsh "$h" "cd $AFD_HOME/mcdma || exit 1; MCDMA_RPCD_SOCKET=$AFD_HOME/mcdma/$n.sock nohup ./mcdma-rpcd listen $n ${RDMA_DEV[$h]} $RDMA_GID_INDEX $RDMA_MTU ${FABRIC_IP[$h]}:$((MCDMA_CTRL_PORT + j)) $MCDMA_REQ_MIB $MCDMA_REP_MIB > listen-$n.log 2>&1 < /dev/null &"
      t=0; until rsh "$h" "test -S $AFD_HOME/mcdma/$n.sock"; do
        sleep 1; t=$((t + 1)); [ $t -ge 20 ] && { rsh "$h" "tail -5 $AFD_HOME/mcdma/listen-$n.log" || true; fail "listen daemon $n on $h: no socket"; }
      done
      log "listen daemon $n up on $h ($MCDMA_REQ_MIB+$MCDMA_REP_MIB MiB, control ${FABRIC_IP[$h]}:$((MCDMA_CTRL_PORT + j)))"
    done
  done
  for j in $(seq 0 $((MCDMA_INFLIGHT - 1))); do   # link j to each expert node, all on one connect daemon (six peers at most)
    peers="$peers $(lname "$MCDMA_LINK0" "$j"),$EXPERT0_FABRIC_IP,$((MCDMA_CTRL_PORT + j)),$ATTN_RDMA_DEV,$RDMA_GID_INDEX,$RDMA_MTU,$MCDMA_REQ_MIB,$MCDMA_REP_MIB"
    peers="$peers $(lname "$MCDMA_LINK1" "$j"),$EXPERT1_FABRIC_IP,$((MCDMA_CTRL_PORT + j)),$ATTN_RDMA_DEV,$RDMA_GID_INDEX,$RDMA_MTU,$MCDMA_REQ_MIB,$MCDMA_REP_MIB"
  done
  rsh attn "cd $AFD_HOME/mcdma || exit 1; MCDMA_RPCD_SOCKET=$AFD_HOME/mcdma/connect.sock nohup ./mcdma-rpcd connect$peers > connect.log 2>&1 < /dev/null &"
  t=0
  until [ "$(links_up)" = $((2 * MCDMA_INFLIGHT)) ]; do
    sleep 2; t=$((t + 2))
    [ $t -ge 60 ] && { status_connect || true; rsh attn "tail -8 $AFD_HOME/mcdma/connect.log" || true; fail "the $((2 * MCDMA_INFLIGHT)) links not up 60 s after the connect daemon started"; }
  done
  log "MCDMA: $((2 * MCDMA_INFLIGHT)) links up ($(links | tr '\n' ' '); $MCDMA_INFLIGHT to each expert half: $MCDMA_LINK0... to half 0, $MCDMA_LINK1... to half 1)"
}

boot() {
  local h common attn ex extra n drafter_arg=""
  # the attention opens every mailbox first thing: the daemons must be up
  for n in $(links); do rsh attn "test -e /dev/shm/mcdma-rpc.$n" || fail "no MCDMA mailbox $n on the attention host"; done
  for h in "${EXPERTS[@]}"; do
    for n in $(links); do
      case $n in "${LINK[$h]}"|"${LINK[$h]}"-[0-9]) rsh "$h" "test -S $AFD_HOME/mcdma/$n.sock" || fail "the listen daemon's socket $n is missing on $h" ;; esac
    done
  done
  common=(--gpus all --network host --ipc=host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
    -v "$AFD_HOME:/afd" -v "$AFD_HOME/tree:/tf:ro" -v "$MODEL_DIR:$MODEL_DIR:ro"
    -v "$EXT_DIR:/ext" -e TORCH_EXTENSIONS_DIR=/ext -e TRITON_CACHE_DIR=/ext/triton
    -e PYTHONPATH=/tf/src -e "TF_AFD_CHECK=$AFD_CHECK" -e "TF_AFD_CHECK_FORWARDS=$AFD_CHECK_FORWARDS" -e TF_AFD_EAGER=1
    -e "TF_AFD_TIME=$AFD_TIME" -v "$AFD_HOME/mcdma:/mcdma" -e TF_AFD_TRANSPORT=mcdma -e TF_AFD_MCDMA_LIB=/mcdma/libmcdma-rpc.so)
  attn=("${common[@]}" -e "TF_AFD_MCDMA_LINKS=$MCDMA_LINK0,$MCDMA_LINK1" -e "TF_AFD_MCDMA_MODE=${MODE[attn]}"
    -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB")
  [ -n "$DENSE" ] && attn+=(-e "TF_GLM_DENSE=$DENSE")
  [ -n "$KV" ] && attn+=(-e "TF_GLM_KV=$KV")
  [ -n "$CACHE_GIB" ] && attn+=(-e "TF_GLM_CACHE_GIB=$CACHE_GIB")
  [ -n "$CACHE_ENTRIES" ] && attn+=(-e "TF_GLM_CACHE_ENTRIES=$CACHE_ENTRIES")
  [ "$MCDMA_INFLIGHT" != 1 ] && attn+=(-e "TF_GLM_MCDMA_INFLIGHT=$MCDMA_INFLIGHT")   # the experts take it from the handshake
  read -r -a extra <<< "$ATTN_ENV_E"; attn+=("${extra[@]}")
  if [ -n "${DRAFTER_DIR:-}" ]; then attn+=(-v "$DRAFTER_DIR:$DRAFTER_DIR:ro"); drafter_arg="--drafter $DRAFTER_DIR"; fi
  log "up: $MODEL_DIR, ${DRAFTER_DIR:-no drafter}, context $CONTEXT${PARALLEL:+, parallel $PARALLEL}${DENSE:+, dense $DENSE}${KV:+, kv $KV}, $MCDMA_INFLIGHT in flight${ATTN_ENV:+, attention env: $ATTN_ENV}"
  # the attention first: it hosts the rendezvous (rank 0) the experts join; the experts wait out the settings
  drun attn "${CONTAINER[attn]}" "${attn[@]}" "$IMAGE" bash -c "exec python -m tensorfold serve $MODEL_DIR --experts remote \
    --master $ATTN_FABRIC_IP --master-port $MASTER_PORT $drafter_arg --context $CONTEXT $PAR_ARG $MAXTOK_ARG \
    --host $API_BIND --port $API_PORT --name $MODEL_NAME --alias $MODEL_ALIAS > /afd/logs/attn.log 2>&1" || fail "the attention container did not start"
  sleep 3
  for h in "${EXPERTS[@]}"; do
    ex=("${common[@]}" -e "TF_AFD_MCDMA_LINK=${LINK[$h]}" -e "MCDMA_RPCD_SOCKET=/mcdma/${LINK[$h]}.sock" -e "TF_AFD_MCDMA_MODE=${MODE[$h]}")
    read -r -a extra <<< "$EXPERT_ENV_E"; ex+=("${extra[@]}")
    drun "$h" "${CONTAINER[$h]}" "${ex[@]}" "$IMAGE" bash -c "exec python -m tensorfold experts $MODEL_DIR \
      --rank ${EXPERT_RANK[$h]} --master $ATTN_FABRIC_IP --master-port $MASTER_PORT > /afd/logs/${NODE_LOG[$h]}.log 2>&1" \
      || fail "the expert container on $h did not start"
  done
}

logsize() {
  local t=0 n h
  for h in "${NODES[@]}"; do
    n=$(rsh "$h" "stat -c %s $AFD_HOME/logs/${NODE_LOG[$h]}.log 2>/dev/null || echo 0" 2>/dev/null || echo 0)
    t=$((t + ${n:-0}))
  done
  echo $t
}

wait_health() {  # fail fast when a container exits, or when no log grows and no health comes for QUIET_MAX s
  local t=0 last=-1 quiet=0 size dead h
  while :; do
    if rsh attn "curl -fsS -m 4 http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then log "healthy after ${t}s"; break; fi
    dead=""
    for h in "${NODES[@]}"; do
      [ -n "$(rsh "$h" "docker ps -aq --filter name=^${CONTAINER[$h]}\$ --filter status=exited" 2>/dev/null)" ] && dead="$dead $h"
    done
    [ -n "$dead" ] && fail "container exited on$dead"
    size=$(logsize)
    if [ "$size" = "$last" ]; then quiet=$((quiet + 15)); else quiet=0; last=$size; fi
    [ $quiet -ge "$QUIET_MAX" ] && fail "no new log bytes for ${quiet}s and no health: a wedged start"
    [ $t -ge "$HEALTH_WAIT" ] && fail "no /health after ${t}s"
    sleep 15; t=$((t + 15))
  done
}

startup_lines() {
  local out h
  out=$(rsh attn "grep -E '^\[tensorfold\] (CUDA rank 0 startup|serving|afd|drafter)' $AFD_HOME/logs/attn.log | head -40" || true)
  echo "$out"
  for h in "${EXPERTS[@]}"; do rsh "$h" "grep -E 'afd expert|afd mcdma' $AFD_HOME/logs/${NODE_LOG[$h]}.log | head -10" || true; done
  grep -q "afd transport mcdma" <<< "$out" || fail "healthy, but no 'afd transport mcdma' line: not the MCDMA wire"
}

cmd_smoke() {
  local body out
  body=$(printf '{"model":"%s","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' "$MODEL_NAME")
  out=$(curl -sS -m 300 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body") || die "smoke: no reply from $BASE_URL"
  grep -q '"content"' <<< "$out" || die "smoke: unexpected reply: ${out:0:300}"
  log "smoke OK: ${out:0:200}"
}

cmd_up() {
  cmd_check
  "$ROOT/build.sh" ext || die "the no-model prebuild failed: nothing started"
  for h in "${EXPERTS[@]}"; do wait_mem "$h"; done   # a GB10 returns memory a few s after a tenant exits
  daemons_up
  boot
  wait_health
  startup_lines
  cmd_smoke || fail "the smoke reply failed"
  log "up: $BASE_URL/v1 serves $MODEL_NAME (alias $MODEL_ALIAS)"
}

cmd_probe() {  # one host at a time, no model, nothing else running
  local h
  for h in "${NODES[@]}"; do
    wait_mem "$h"
    rsh "$h" "docker rm -f glm-afd-probe >/dev/null 2>&1; mkdir -p $AFD_HOME/logs" || true
    drun "$h" glm-afd-probe --gpus all --network none --ipc=host --ulimit memlock=-1 --cap-add IPC_LOCK \
      -v "$AFD_HOME:/afd" -v "$AFD_HOME/tree:/tf:ro" -v "$AFD_HOME/mcdma:/mcdma" -e PYTHONPATH=/tf/src \
      -e TF_AFD_MCDMA_LIB=/mcdma/libmcdma-rpc.so "$IMAGE" python /tf/tools/mcdma_streamops_probe.py --out /afd/logs/probe.json \
      || die "$h: the probe container did not start"
    RSH_TIMEOUT=660 rsh "$h" "timeout 600 docker wait glm-afd-probe >/dev/null; docker logs glm-afd-probe 2>&1 | grep -E '^STREAMOPS' || echo 'no verdict line'; docker rm -f glm-afd-probe >/dev/null" \
      | sed "s#^#$h: #"
  done
}

cmd_status() {
  local h
  for h in "${NODES[@]}"; do
    echo "== $h: $(rsh "$h" "docker ps -a --filter label=glm-afd=1 --format '{{.Names}} {{.Status}}' | tr '\n' ' '" 2>/dev/null) daemons=$(rsh "$h" 'pgrep -xc mcdma-rpcd' 2>/dev/null || echo 0)"
  done
  echo "== health: $(rsh attn "curl -fsS -m 5 http://127.0.0.1:$API_PORT/health 2>/dev/null" || echo no-health)"
  echo "== MCDMA links:"; status_connect || echo "(no connect daemon)"
}

cmd_logs() {
  local h n
  rsh attn "tail -40 $AFD_HOME/logs/attn.log" 2>/dev/null || true
  for h in "${EXPERTS[@]}"; do rsh "$h" "tail -20 $AFD_HOME/logs/${NODE_LOG[$h]}.log" 2>/dev/null || true; done
  echo "== daemons"
  rsh attn "tail -8 $AFD_HOME/mcdma/connect.log" 2>/dev/null || true
  for h in "${EXPERTS[@]}"; do
    for n in $(links); do
      case $n in "${LINK[$h]}"|"${LINK[$h]}"-[0-9]) rsh "$h" "tail -8 $AFD_HOME/mcdma/listen-$n.log" 2>/dev/null || true ;; esac
    done
  done
}

case ${1:-up} in
  up) cmd_up ;; check) cmd_check ;; probe) cmd_probe ;; status) cmd_status ;; logs) cmd_logs ;; smoke) cmd_smoke ;;
  *) echo "usage: $0 [up|check|probe|status|logs|smoke]"; exit 2 ;;
esac
