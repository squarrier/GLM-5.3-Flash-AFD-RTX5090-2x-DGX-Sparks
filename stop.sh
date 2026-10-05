#!/bin/bash
# stop.sh — take the stack down, in MCDMA's order: the containers (the attention node first), then the connect
# daemon on the attention host, then the listen daemons on the Sparks. Daemons get SHUTDOWN, then SIGTERM after
# 10 s, and are never SIGKILLed (that would skip their queue-pair teardown).
#   ./stop.sh [all|tf]      tf = the containers only; the MCDMA links stay up for the next ./start.sh up
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
what=${1:-all}
case $what in all|tf) ;; *) echo "usage: $0 [all|tf]"; exit 2 ;; esac

for h in "${NODES[@]}"; do dstop "$h"; done
log "containers removed on the attention host and both Sparks"
[ "$what" = tf ] && exit 0
rc=0
stop_daemons attn "$AFD_HOME/mcdma/connect.sock" || rc=1
# every link's listen daemon, at any MCDMA_INFLIGHT the stack was started with (1-3): NAME, NAME-1, NAME-2
for h in "${EXPERTS[@]}"; do
  stop_daemons "$h" "$AFD_HOME/mcdma/${LINK[$h]}.sock" "$AFD_HOME/mcdma/${LINK[$h]}-1.sock" "$AFD_HOME/mcdma/${LINK[$h]}-2.sock" || rc=1
done
for h in "${NODES[@]}"; do
  log "$h after stop: daemons=$(rsh "$h" 'pgrep -xc mcdma-rpcd' 2>/dev/null || echo 0) containers=$(rsh "$h" 'docker ps -aq --filter label=glm-afd=1 | wc -l' 2>/dev/null)"
done
exit $rc
