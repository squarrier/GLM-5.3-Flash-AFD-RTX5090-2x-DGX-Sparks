#!/bin/bash
# start.sh — controller for the GLM-5.3-Flash AFD lane (1× RTX 5090 coordinator + 2× DGX Spark).
# Runs anywhere with SSH to the three hosts. Orders the lane: ranks (both Sparks) ->
# 4 ranks listening -> coordinator -> /health 200 -> one real completion.
#
#   ./start.sh            same as `up`
#   ./start.sh up         start whatever is not running (idempotent)
#   ./start.sh restart    down + up
#   ./start.sh status     units, MemAvailable, MPS, /health, log tails
#   ./start.sh recover    minimal recovery: restart the broken Spark tenant(s), then the coordinator
#   ./start.sh probe      one real completion (exit 0/1)
#   ./stop.sh             stop everything (graceful MPS shutdown)
set -u
. "$(dirname "$0")/scripts/lib.sh"
LOCK=${XDG_RUNTIME_DIR:-/tmp}/glm53f-afd-ctl.lock
exec 9>"$LOCK"; flock -n 9 || die "another start/stop/recover is running ($LOCK)"

health_code() { curl -s -o /dev/null -m 5 -w '%{http_code}' "$BASE_URL/health" 2>/dev/null; }
avail_gib() { rsh $1 "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo"; }
unit_active() { [ "$(rsh $1 "systemctl is-active $2" 2>/dev/null)" = active ]; }
listening_count() { rsh $1 "grep -h '^listening' $GLM_HOME/logs/ranks-latest/rank*.log 2>/dev/null | wc -l" 2>/dev/null || echo 0; }
mps_count() { rsh $1 "pgrep -fc ^nvidia-cuda-mps-server" 2>/dev/null || echo 0; }
broken_rank_log() { rsh $1 "grep -lE 'serve layer [0-9]+ failed|CUDA error' $GLM_HOME/logs/ranks-latest/rank*.log 2>/dev/null | wc -l" 2>/dev/null || echo 0; }
ranks_ok() {
  unit_active $1 glm53f-afd-ranks && [ "$(listening_count $1)" -ge 2 ] \
    && [ "$(mps_count $1)" -ge 1 ] && [ "$(broken_rank_log $1)" = 0 ]
}
blamed_sparks() {  # Sparks named in a latched 503 reason ("rank N")
  curl -s -m 5 "$BASE_URL/health" 2>/dev/null | grep -oE 'rank [0-3]' | awk '{print ($2<2)?"spark1":"spark2"}' | sort -u
}
probe() {  # /health can stay 200 for ~2 min with a dead rank; a real completion cannot
  local auth=(); [ -n "${API_KEY:-}" ] && auth=(-H "Authorization: Bearer $API_KEY")
  curl -s -m 90 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' "${auth[@]}" \
    -d "{\"model\":\"$FIRST_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: PONG-7731\"}],\"max_tokens\":256,\"temperature\":0,\"reasoning_effort\":\"none\"}" \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);sys.exit(0 if "PONG-7731" in (d["choices"][0]["message"].get("content") or "") else 1)' 2>/dev/null
}
wait_free() {  # GB10 hands unified memory back a few seconds after a tenant exits
  local a
  for _ in $(seq 1 30); do
    a=$(avail_gib $1); awk -v a="$a" -v m="$MIN_FREE_GIB_SPARK" 'BEGIN{exit !(a>=m)}' && return 0; sleep 3
  done
  log "$1 MemAvailable $a GiB < $MIN_FREE_GIB_SPARK GiB after 90 s"; return 1
}
unit_log() { rsh $1 "journalctl -u $2 -n 8 --no-pager" 2>&1 | tail -8; }
start_spark() { rsh $1 "sudo -n systemctl start glm53f-afd-ranks" || { unit_log $1 glm53f-afd-ranks >&2; die "$1: rank start failed"; }; }
stop_spark() { rsh $1 "sudo -n systemctl stop glm53f-afd-ranks" 2>/dev/null; }
stop_coord() { rsh coord "sudo -n systemctl stop glm53f-afd-coord" 2>/dev/null; }
start_coord() { rsh coord "sudo -n systemctl start glm53f-afd-coord" || { unit_log coord glm53f-afd-coord >&2; die "coordinator start failed"; }; }
wait_ranks() {
  local t0; t0=$(date +%s)
  while :; do
    local ok=1
    for h in "$@"; do
      unit_active $h glm53f-afd-ranks || { rsh $h "tail -5 $GLM_HOME/logs/ranks-latest/wrapper.log; tail -3 $GLM_HOME/logs/ranks-latest/rank*.log" >&2; die "$h rank tenant exited during start"; }
      [ "$(listening_count $h)" -ge 2 ] || ok=0
    done
    [ $ok = 1 ] && { log "ranks listening on $* after $(( $(date +%s)-t0 )) s"; return 0; }
    [ $(( $(date +%s)-t0 )) -gt 360 ] && die "ranks not listening after 360 s on $*"
    sleep 5
  done
}
wait_health() {
  local t0; t0=$(date +%s)
  while :; do
    [ "$(health_code)" = 200 ] && { log "/health 200 after $(( $(date +%s)-t0 )) s"; return 0; }
    unit_active coord glm53f-afd-coord || { rsh coord "tail -8 $GLM_HOME/logs/coord-latest.log" >&2; die "coordinator exited during start"; }
    [ $(( $(date +%s)-t0 )) -gt 300 ] && die "/health not 200 after 300 s"
    sleep 3
  done
}

cmd_up() {
  local t0 need=(); t0=$(date +%s)
  if [ "$(health_code)" = 200 ] && ranks_ok spark1 && ranks_ok spark2 && unit_active coord glm53f-afd-coord && probe; then
    log "already up: $BASE_URL/v1"; return 0
  fi
  for h in "${SPARKS[@]}"; do
    if ranks_ok $h; then log "$h: ranks already up"; continue; fi
    unit_active $h glm53f-afd-ranks && { log "$h: rank tenant unhealthy -> stopping"; stop_spark $h; }
    wait_free $h || die "$h is below the ${MIN_FREE_GIB_SPARK} GiB floor (another GPU process, or stranded GB10 memory that needs a reboot)"
    need+=($h)
  done
  [ ${#need[@]} -gt 0 ] && unit_active coord glm53f-afd-coord && { log "ranks restarting -> stopping the coordinator first"; stop_coord; }
  for h in "${need[@]}"; do log "$h: starting ranks ${RANK_A[$h]},${RANK_B[$h]}"; start_spark $h; done
  wait_ranks "${SPARKS[@]}"
  if ! unit_active coord glm53f-afd-coord; then log "coord: starting"; start_coord
  elif [ "$(health_code)" != 200 ]; then stop_coord; start_coord; fi
  wait_health
  probe || die "the first completion failed (see ./start.sh status)"
  log "UP in $(( $(date +%s)-t0 )) s: $BASE_URL/v1  model=$FIRST_NAME"
}
cmd_down() {
  stop_coord; stop_spark spark2; stop_spark spark1
  local bad=0
  for h in "${SPARKS[@]}" coord; do
    left=$(rsh $h "docker ps -q --filter label=glm53f-afd; pgrep -f ^nvidia-cuda-mps-server" 2>/dev/null)
    [ -n "$left" ] && { log "$h NOT clean: [$left]"; bad=1; }
  done
  [ $bad = 0 ] && log "DOWN" || die "down incomplete"
}
cmd_status() {
  for h in "${SPARKS[@]}" coord; do
    echo "== $h (${HOST[$h]})"
    rsh $h "awk '/MemAvailable/{printf \"  MemAvailable %.1f GiB\n\", \$2/1048576}' /proc/meminfo
      for u in glm53f-afd-ranks glm53f-afd-coord gb10-hostguard; do s=\$(systemctl is-active \$u 2>/dev/null); [ \"\$s\" = inactive ] || [ -z \"\$s\" ] || echo \"  \$u=\$s\"; done
      nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'
      pgrep -a nvidia-cuda-mps | sed 's/^/  mps /'" 2>/dev/null || echo "  SSH FAILED"
  done
  echo "== $BASE_URL/health: $(curl -s -m 5 -w ' http=%{http_code}' "$BASE_URL/health")"
  echo "== coordinator log"; rsh coord "tail -3 $GLM_HOME/logs/coord-latest.log" 2>/dev/null | cut -c1-220
}
# The coordinator connects to the ranks only at start and latches /health 503 after
# ANY failed exchange, so recovery always restarts it; a Spark whose daemon or MPS
# server died is restarted as a whole first. rc 3 = a Spark stayed below the memory
# floor after a graceful stop (stranded GB10 memory; needs a reboot).
cmd_recover() {
  local t0 bad=() hc; t0=$(date +%s)
  for h in "${SPARKS[@]}"; do ranks_ok $h || bad+=($h); done
  hc=$(health_code)
  [ "$hc" = 503 ] && for h in $(blamed_sparks); do [[ " ${bad[*]} " == *" $h "* ]] || bad+=($h); done
  log "recover: /health=$hc bad_sparks=[${bad[*]}]"
  if [ ${#bad[@]} = 0 ] && [ "$hc" = 200 ] && probe; then log "healthy, nothing to do"; return 0; fi
  stop_coord
  for h in "${bad[@]}"; do
    stop_spark $h
    wait_free $h || { log "$h stranded below the floor after a graceful stop"; return 3; }
    start_spark $h
  done
  [ ${#bad[@]} -gt 0 ] && wait_ranks "${bad[@]}"
  start_coord; wait_health
  probe || { log "probe still failing after recovery"; return 4; }
  log "RECOVERED in $(( $(date +%s)-t0 )) s (restarted: ${bad[*]:-} coordinator)"
}

case "${1:-up}" in
  up) cmd_up;; down|stop) cmd_down;; restart) cmd_down; cmd_up;;
  status) cmd_status;; recover) cmd_recover;;
  probe) probe && echo "probe OK" || { echo "probe FAIL"; exit 1; };;
  *) echo "usage: $0 {up|restart|status|recover|probe}  (./stop.sh to stop)"; exit 2;;
esac
