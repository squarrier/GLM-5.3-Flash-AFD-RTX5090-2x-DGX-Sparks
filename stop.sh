#!/bin/bash
# stop.sh — take the stack down, in MCDMA's order: the containers (the attention node first), then the connect
# daemons on the attention host, then the listen daemons on the Sparks. SIGTERM only, never SIGKILL: each running
# container gets one SIGTERM and STOP_WAIT s (60) to exit. A container still running after that is left running and
# named, and the daemons under it stay up (exit 1). Daemons get SHUTDOWN, then SIGTERM after 10 s (a SIGKILL would
# skip their queue-pair teardown).
# A stop by hand also leaves a stop marker on this controller, so extras/watch leaves the stack alone until
# ./start.sh up.
#   ./stop.sh [all|tf]      tf = the containers only; the MCDMA links stay up (./start.sh recover starts the
#                           containers on them again)
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
what=${1:-all}
case $what in all|tf) ;; *) echo "usage: $0 [all|tf]"; exit 2 ;; esac
ctl_lock
if [ "${GLM_AFD_INTERNAL:-0}" != 1 ]; then      # by hand, not start.sh's own stop after a failure
  echo "$(date -Iseconds) ./stop.sh $what" > "$CTL_STATE/stopped"
fi

left=""
for h in "${NODES[@]}"; do dstop "$h" || left="$left $h"; done
if [ -n "$left" ]; then
  log "containers still running on$left after SIGTERM: nothing was killed, and the MCDMA daemons stay up under them"
  exit 1
fi
log "containers removed on the attention host and both Sparks"
[ "$what" = tf ] && exit 0
rc=0
stop_all_daemons || rc=1
for h in "${NODES[@]}"; do
  log "$h after stop: daemons=$(rsh "$h" 'pgrep -xc mcdma-rpcd' 2>/dev/null || echo 0) containers=$(rsh "$h" 'docker ps -aq --filter label=glm-afd=1 | wc -l' 2>/dev/null)"
done
exit $rc
