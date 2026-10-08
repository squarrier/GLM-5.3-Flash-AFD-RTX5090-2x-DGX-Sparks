#!/bin/bash
# Dry run, no hardware: start.sh up, stop.sh, start.sh recover, build.sh ext/mcdma and download.sh against stand-in
# ssh/scp/curl that record every remote command instead of running it, and keep a little state (which containers and
# MCDMA daemons "run" on which host) so stop and recover can be checked. Checks the command lines the scripts generate
# (the docker run flags, the serve/experts lines, the MCDMA daemons' lines), the order of the steps, SIGTERM only (no
# docker rm -f of a running container, no SIGKILL), and what recover restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/repo" "$tmp/bin" "$tmp/state"
tar cf - --exclude=.git --exclude=.env . | tar xf - -C "$tmp/repo"
cp .env.example "$tmp/repo/.env"
LOG=$tmp/remote.log; RUNS=$tmp/docker-run.log; ST=$tmp/state; : > "$LOG"; : > "$RUNS"

cat > "$tmp/bin/ssh" <<'EOF'
#!/bin/bash
# stand-in ssh: the last argument is the remote command; record it, keep the containers' and daemons' state in
# $DRY_STATE (HOST.c.NAME = a running container, HOST.d.NAME = a running mcdma-rpcd), answer what the scripts ask
host=""; for a in "$@"; do case $a in *@*) host=${a#*@} ;; esac; done
cmd=${!#}
S=$DRY_STATE
printf '%s\t%s\n' "$host" "$cmd" >> "$DRY_LOG"
ctrs() { for f in "$S/$host".c.*; do [ -e "$f" ] && echo "${f##*.c.}"; done; return 0; }
case $cmd in
  *"tar x"*) cat > /dev/null ;;
  "docker run -d "*) eval "set -- $cmd"; { printf '%s' "$host"; printf '\t%s' "$@"; echo; } >> "$DRY_RUNS"
    n=""; prev=""; for a in "$@"; do [ "$prev" = --name ] && n=$a; prev=$a; done; touch "$S/$host.c.$n"; echo 0123456789ab ;;
  "docker ps -q --filter label=glm-afd=1") ctrs ;;
  "docker ps --filter label=glm-afd=1 --format '{{.Names}}'") ctrs ;;
  "docker kill --signal TERM "*)
    [ -e "$S/stubborn" ] && exit 0                     # a container that ignores its SIGTERM
    for c in ${cmd#docker kill --signal TERM }; do case $c in \>*|/dev/null) ;; *) rm -f "$S/$host.c.$c" ;; esac; done ;;
  *"docker ps -a --filter name=^"*"--format '{{.State}}'"*)
    c=${cmd#*name=^}; c=${c%%\$*}; [ -e "$S/$host.c.$c" ] && echo running ;;
  *"docker inspect -f '{{.State.Running}}'"*) c=${cmd##*\}\}\' }; c=${c%% *}; [ -e "$S/$host.c.$c" ] && echo true || echo false ;;
  *"{{.State.Status}} {{.State.ExitCode}}"*) c=${cmd##* }; rm -f "$S/$host.c.$c"; echo "exited 0" ;;
  *MemAvailable*) cat "$S/mem" 2>/dev/null || echo 120 ;;
  *"pgrep -xc"*) n=0; for f in "$S/$host".d.*; do [ -e "$f" ] && n=$((n + 1)); done; echo $n ;;
  *"pgrep -x mcdma-rpcd"*) compgen -G "$S/$host.d.*" > /dev/null || exit 1 ;;
  *"pkill -TERM -x mcdma-rpcd"*) rm -f "$S/$host".d.* ;;
  *"nohup ./mcdma-rpcd listen "*) n=${cmd#*mcdma-rpcd listen }; touch "$S/$host.d.${n%% *}" ;;
  *"nohup ./mcdma-rpcd connect"*) s=${cmd#*MCDMA_RPCD_SOCKET=}; s=${s%% *}; touch "$S/$host.d.$(basename "$s" .sock)" ;;
  *SHUTDOWN*) s=${cmd#test -S }; s=${s%% *}; n=$(basename "$s" .sock); [ -e "$S/$host.d.$n" ] || exit 1; rm -f "$S/$host.d.$n" ;;
  *STATUS*)   # each connect daemon reports its own peers: links 0-2 on connect, link 3 on connect2
    c=connect; links="x0 x1 x0-1 x1-1 x0-2 x1-2"
    case $cmd in *connect2.sock*) c=connect2; links="x0-3 x1-3" ;; esac
    [ -e "$S/$host.d.$c" ] || exit 1
    for n in $links; do
      case $n in x0*) e=10.0.0.11 ;; *) e=10.0.0.12 ;; esac
      [ -e "$S/$e.d.$n" ] && echo "PEER $n up $e"
    done ;;
  "test -S "*) s=${cmd#test -S }; n=$(basename "$s" .sock); [ -e "$S/$host.d.$n" ] || exit 1 ;;
  "test -e /dev/shm/mcdma-rpc."*) n=${cmd#test -e /dev/shm/mcdma-rpc.}
    case $n in *-3) [ -e "$S/$host.d.connect2" ] || exit 1 ;; *) [ -e "$S/$host.d.connect" ] || exit 1 ;; esac ;;
  *"curl -fsS -m 4 http://127.0.0.1"*|*"curl -fsS -m 5 http://127.0.0.1"*)
    { [ -e "$S/$host.c.glm-afd-attn" ] && [ ! -e "$S/unhealthy" ]; } || exit 7; echo '{"ok": true}' ;;
  *"docker image inspect -f"*) echo "sha256:0123456789abcdef0123" ;;
  *"grep -E '^\[tensorfold\]"*) echo "[tensorfold] afd transport mcdma, gpu-driven: link x0 -> expert 1, x1 -> expert 2" ;;
  *"stat -c %s"*) echo 100 ;;
  *"sha256sum mcdma-rpcd"*) echo "0000 mcdma-rpcd" ;;
esac
exit 0
EOF
cat > "$tmp/bin/scp" <<'EOF'
#!/bin/bash
printf 'scp\t%s\n' "$*" >> "$DRY_LOG"
EOF
cat > "$tmp/bin/curl" <<'EOF'
#!/bin/bash
printf 'curl\t%s\n' "$*" >> "$DRY_LOG"
[ -e "$DRY_STATE/smoke_fail" ] && exit 7
echo '{"choices":[{"message":{"role":"assistant","content":"OK"}}]}'
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/scp" "$tmp/bin/curl"
CTL=$tmp/ctl
run() { env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" DRY_STATE="$ST" CTL_STATE="$CTL" \
  STOP_WAIT="${STOP_WAIT:-4}" MEM_WAIT="${MEM_WAIT:-180}" ${CALLER:+GLM_AFD_CALLER=$CALLER} bash "$@"; }

fail=0
check() {  # label, file, fixed string that must appear
  grep -qF -- "$3" "$2" || { echo "dry run FAIL ($1): missing: $3"; fail=1; }
}
nokill() {  # label: no SIGKILL in the remote log (docker rm -f, kill -9/-KILL, docker stop)
  if grep -qE 'docker rm -f|kill -9|-KILL|signal KILL|docker stop' "$LOG"; then echo "dry run FAIL ($1): a SIGKILL"; fail=1; fi
}
run "$tmp/repo/start.sh" up > "$tmp/up.out" 2>&1 || { cat "$tmp/up.out"; echo "dry run FAIL: start.sh up"; exit 1; }
check up "$tmp/up.out" "check OK"
check up "$tmp/up.out" "MCDMA: 8 links up (x0 x1 x0-1 x1-1 x0-2 x1-2 x0-3 x1-3"
check up "$tmp/up.out" "smoke OK"
# the prebuild: no model, no network, the tree's own extension cache, stale locks removed
check prebuild "$RUNS" $'\t--network\tnone\t-v\t/srv/glm-afd/ext-'
check prebuild "$RUNS" 'exec python /p/prebuild_ext.py'
# the daemons (MCDMA_INFLIGHT=4): four listen ends on each Spark, link j on control port 18820 + j, then on the
# attention host one connect end with links 0-2 of both nodes (six peers, MCDMA's most) and a second with link 3
for j in 0 1 2 3; do
  for h in 0 1; do
    n=x$h; [ $j = 0 ] || n=x$h-$j
    check daemons "$LOG" "MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/$n.sock nohup ./mcdma-rpcd listen $n rocep1s0f0 3 4096 10.10.1.1$((h + 1)):1882$j 20 36"
  done
done
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/connect.sock nohup ./mcdma-rpcd connect x0,10.10.1.11,18820,mlx5_0,3,4096,20,36 x1,10.10.1.12,18820,mlx5_0,3,4096,20,36 x0-1,10.10.1.11,18821,mlx5_0,3,4096,20,36 x1-1,10.10.1.12,18821,mlx5_0,3,4096,20,36 x0-2,10.10.1.11,18822,mlx5_0,3,4096,20,36 x1-2,10.10.1.12,18822,mlx5_0,3,4096,20,36 > connect.log'
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/connect2.sock nohup ./mcdma-rpcd connect x0-3,10.10.1.11,18823,mlx5_0,3,4096,20,36 x1-3,10.10.1.12,18823,mlx5_0,3,4096,20,36 > connect2.log'
[ "$(grep -cF 'mcdma-rpcd listen' "$LOG")" = 8 ] || { echo "dry run FAIL (daemons): want 8 listen daemons"; fail=1; }
[ "$(grep -cF 'mcdma-rpcd connect' "$LOG")" = 2 ] || { echo "dry run FAIL (daemons): want 2 connect daemons"; fail=1; }
check mailboxes "$LOG" 'test -e /dev/shm/mcdma-rpc.x1-3'
# the attention node
a=$(grep -P '^10\.0\.0\.10\t' "$RUNS" | grep -F 'tensorfold serve' || true)
for s in $'--init\t--restart=no\t--oom-score-adj=1000' $'--name\tglm-afd-attn' $'--gpus\tall\t--network\thost\t--ipc=host\t--device\t/dev/infiniband\t--ulimit\tmemlock=-1\t--cap-add\tIPC_LOCK' \
    $'-e\tTF_AFD_TRANSPORT=mcdma' $'-e\tTF_AFD_MCDMA_LINKS=x0,x1' $'-e\tTF_AFD_MCDMA_MODE=auto' $'-e\tTF_AFD_EAGER=1' $'-e\tTENSORFOLD_MEMORY_RESERVE_GIB=2' \
    $'-e\tTF_GLM_DENSE=q4' $'-e\tTF_GLM_KV=fp8' $'-e\tTF_GLM_CACHE_GIB=8.5' $'-e\tTF_GLM_CACHE_ENTRIES=20' $'-e\tTF_GLM_MCDMA_INFLIGHT=4' \
    $'-e\tTF_GLM_SHARED_PREFIX=1' $'-e\tTF_GLM_STREAM_SMOOTH_MS=400' \
    $'-e\tTF_GLM_TOOL_CALLS=1' $'-e\tTF_GLM_KDA_CHUNKED=1' $'-e\tTF_GLM_PREFILL_PAIRS=1' $'-e\tTF_GLM_DFLASH_POLICY=fnc5:0.2' \
    $'-e\tTF_GLM_CACHE_ROOM=1' $'-e\tTF_GLM_PREFILL_ORDER=sjf' \
    $'-e\tTF_GLM_KEPT_HOST=1' $'-e\tTF_GLM_HOST_CACHE_GIB=24' $'-e\tTF_GLM_FILL_PAIRS=1' \
    $'-e\tTF_GLM_DECIDE_THEN_COPY=1' $'-e\tTF_GLM_CAPACITY_STATUS=1' $'-e\tTF_GLM_DELIVERY_ABORT=1' \
    $'-e\tTF_GLM_MULTI_WINDOW=32\t-e\tTF_GLM_PREFILL_LANES=4\t-e\tTF_GLM_PARTIALS_BF16=1' \
    $'-v\t/srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold:/srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold:ro' \
    'tensorfold serve /srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold --experts remote' '--master 10.10.1.10 --master-port 29551' \
    '--drafter /srv/models/GLM-5.3-Flash-DFlash2 --context 262144 --parallel 8 --max-tokens 32768' \
    '--host 0.0.0.0 --port 8000 --name GLM-5.3-Flash-EXL3 --alias glm-5.3-flash > /afd/logs/attn.log 2>&1'; do
  case "$a" in *"$s"*) ;; *) echo "dry run FAIL (attention): missing: $s"; fail=1 ;; esac
done
case "$a" in *TF_GLM_EXL3_*|*TF_GLM_EXPERT_KERNEL*) echo "dry run FAIL (attention): an expert-node switch"; fail=1 ;; esac
case "$a" in *fnc7:0.3*|*TF_GLM_MULTI_WINDOW=64*|*TF_GLM_WIRE_FP8*|*TF_GLM_PREFILL_ROWS*|*TF_GLM_SHARED_PREFIX_COPY*|*TF_GLM_QUEUED_CANCEL*|*TF_GLM_COMPACT_BEFORE_EVICT*|*TF_GLM_ASSISTANT_ENDS*|*TF_GLM_CAP_SHARED_RECENCY*|*TF_GLM_MAX_QUEUED*) echo "dry run FAIL (attention): a switch .env.example leaves off"; fail=1 ;; esac
# the expert nodes: their own link and socket, their decode and prompt kernel switches, the BF16 switch the attention
# node has, the schedule knobs with their tier; no attention-only settings
for h in 0 1; do
  x=$(grep -P "^10\.0\.0\.1$((h + 1))\t" "$RUNS" | grep -F 'tensorfold experts' || true)
  for s in $'--init\t--restart=no' $'--name\tglm-afd-x'$h $'-e\tTF_AFD_MCDMA_LINK=x'$h $'-e\tMCDMA_RPCD_SOCKET=/mcdma/x'$h'.sock' \
      $'-e\tTF_GLM_EXL3_DEC=1' $'-e\tTF_GLM_EXL3_LOADS=nc' $'-e\tTF_GLM_EXL3_PROMPT=1' $'-e\tTF_GLM_EXPERT_KERNEL=g53' \
      $'-e\tTF_GLM_PARTIALS_BF16=1\t-e\tTF_GLM_EXPERT_KERNEL_MT=4\t-e\tTF_GLM_EXPERT_KERNEL_GW=16\t-e\tTF_GLM_EXPERT_KERNEL_NT=4\t-e\tTF_GLM_EXPERT_KERNEL_L2=1\t-e\tTF_GLM_EXPERT_KERNEL_TIER_ROWS=1536' \
      "tensorfold experts /srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold" "--rank $h --master 10.10.1.10 --master-port 29551" "/afd/logs/expert$h.log"; do
    case "$x" in *"$s"*) ;; *) echo "dry run FAIL (expert $h): missing: $s"; fail=1 ;; esac
  done
  case "$x" in *TF_GLM_DENSE*|*TF_GLM_KV*|*--drafter*|*TF_GLM_KDA_CHUNKED*|*TF_GLM_DFLASH_POLICY*|*TF_GLM_PREFILL_PAIRS*|*TF_GLM_PREFILL_LANES*|*TF_GLM_MCDMA_INFLIGHT*|*TF_GLM_CACHE_ROOM*|*TF_GLM_PREFILL_ORDER*|*TF_GLM_MULTI_WINDOW*|*TF_GLM_KEPT_HOST*|*TF_GLM_HOST_CACHE_GIB*|*TF_GLM_FILL_PAIRS*|*TF_GLM_DECIDE_THEN_COPY*|*TF_GLM_CAPACITY_STATUS*|*TF_GLM_DELIVERY_ABORT*) echo "dry run FAIL (expert $h): attention-only settings"; fail=1 ;; esac
done
# order: listen daemons, connect daemon, attention container, expert containers
first_serve=$(grep -nF -- '--name glm-afd-attn' "$LOG" | head -1 | cut -d: -f1 || true)
last_daemon=$(grep -nF 'mcdma-rpcd connect' "$LOG" | tail -1 | cut -d: -f1 || true)
first_expert=$(grep -nF -- '--name glm-afd-x0' "$LOG" | head -1 | cut -d: -f1 || true)
last_listen=$(grep -nF 'mcdma-rpcd listen' "$LOG" | tail -1 | cut -d: -f1 || true)
[ -n "$first_serve" ] && [ -n "$last_daemon" ] && [ -n "$first_expert" ] && [ -n "$last_listen" ] \
  && [ "$last_listen" -lt "$last_daemon" ] && [ "$last_daemon" -lt "$first_serve" ] && [ "$first_serve" -lt "$first_expert" ] \
  || { echo "dry run FAIL: order (listen $last_listen, connect $last_daemon, attention $first_serve, expert $first_expert)"; fail=1; }
nokill up

# stop: SIGTERM to the containers (the attention host first), then SHUTDOWN to the connect daemon, then the listen ones;
# a stop marker for extras/watch
: > "$LOG"
run "$tmp/repo/stop.sh" > "$tmp/stop.out" 2>&1 || { cat "$tmp/stop.out"; echo "dry run FAIL: stop.sh"; fail=1; }
t=$(grep -nF 'docker kill --signal TERM' "$LOG" | head -1 | cut -d: -f1 || true)
ta=$(grep -nP '^10\.0\.0\.10\tdocker kill --signal TERM' "$LOG" | head -1 | cut -d: -f1 || true)
tx=$(grep -nP '^10\.0\.0\.1[12]\tdocker kill --signal TERM' "$LOG" | head -1 | cut -d: -f1 || true)
c=$(grep -nF 'connect.sock' "$LOG" | head -1 | cut -d: -f1 || true); l=$(grep -nF 'x0.sock' "$LOG" | head -1 | cut -d: -f1 || true)
[ -n "$t" ] && [ -n "$c" ] && [ -n "$l" ] && [ "$ta" = "$t" ] && [ "$ta" -lt "$tx" ] && [ "$t" -lt "$c" ] && [ "$c" -lt "$l" ] \
  || { echo "dry run FAIL: stop order (TERM $t, attention $ta, experts $tx, connect $c, listen $l)"; fail=1; }
for s in x0.sock x0-1.sock x0-2.sock x0-3.sock x1.sock x1-1.sock x1-2.sock x1-3.sock connect2.sock; do grep -F "/srv/glm-afd/mcdma/$s" "$LOG" | grep -qF SHUTDOWN || { echo "dry run FAIL: no SHUTDOWN to $s"; fail=1; }; done
nokill stop
[ -s "$CTL/stopped" ] || { echo "dry run FAIL: stop.sh left no stop marker"; fail=1; }
if compgen -G "$ST/*.[cd].*" > /dev/null; then echo "dry run FAIL: stop.sh left $(cd "$ST" && echo *.[cd].*)"; fail=1; fi

# two links a Spark (MCDMA_INFLIGHT=2, 2.1's setting, a caller export): one connect daemon with four peers, no second
: > "$LOG"; : > "$RUNS"
env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" DRY_STATE="$ST" CTL_STATE="$CTL" MCDMA_INFLIGHT=2 bash "$tmp/repo/start.sh" up > "$tmp/up2.out" 2>&1 \
  || { cat "$tmp/up2.out"; echo "dry run FAIL: start.sh up at MCDMA_INFLIGHT=2"; exit 1; }
check up2 "$tmp/up2.out" "MCDMA: 4 links up (x0 x1 x0-1 x1-1"
check up2 "$LOG" './mcdma-rpcd connect x0,10.10.1.11,18820,mlx5_0,3,4096,20,36 x1,10.10.1.12,18820,mlx5_0,3,4096,20,36 x0-1,10.10.1.11,18821,mlx5_0,3,4096,20,36 x1-1,10.10.1.12,18821,mlx5_0,3,4096,20,36 > connect.log'
[ "$(grep -cF 'mcdma-rpcd listen' "$LOG")" = 4 ] || { echo "dry run FAIL (up2): want 4 listen daemons"; fail=1; }
grep -qF 'connect2' "$LOG" && { echo "dry run FAIL (up2): a second connect daemon at two links"; fail=1; }
check up2 "$RUNS" $'-e\tTF_GLM_MCDMA_INFLIGHT=2'
run "$tmp/repo/stop.sh" > /dev/null 2>&1 || { echo "dry run FAIL: stop.sh after up2"; fail=1; }

# one link a Spark (MCDMA_INFLIGHT=1, a caller export): v2.0's daemon lines, no in-flight switch on the attention node
: > "$LOG"; : > "$RUNS"
env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" DRY_STATE="$ST" CTL_STATE="$CTL" MCDMA_INFLIGHT=1 bash "$tmp/repo/start.sh" up > "$tmp/up1.out" 2>&1 \
  || { cat "$tmp/up1.out"; echo "dry run FAIL: start.sh up at MCDMA_INFLIGHT=1"; exit 1; }
check up1 "$tmp/up1.out" "MCDMA: 2 links up (x0 x1"
check up1 "$LOG" './mcdma-rpcd connect x0,10.10.1.11,18820,mlx5_0,3,4096,20,36 x1,10.10.1.12,18820,mlx5_0,3,4096,20,36 > connect.log'
[ "$(grep -cF 'mcdma-rpcd listen' "$LOG")" = 2 ] || { echo "dry run FAIL (up1): want 2 listen daemons"; fail=1; }
grep -qF 'TF_GLM_MCDMA_INFLIGHT' "$RUNS" && { echo "dry run FAIL (up1): TF_GLM_MCDMA_INFLIGHT at one link"; fail=1; }
[ -e "$CTL/stopped" ] && { echo "dry run FAIL: start.sh up left the stop marker"; fail=1; }
run "$tmp/repo/stop.sh" > /dev/null 2>&1 || { echo "dry run FAIL: stop.sh after up1"; fail=1; }
# a bad MCDMA_INFLIGHT, a bad STOP_WAIT, an in-flight switch in ATTN_ENV and a BF16 switch on one side only are refused
# before anything runs
for bad in 'MCDMA_INFLIGHT=5' 'MCDMA_INFLIGHT=x' 'ATTN_ENV=TF_GLM_MCDMA_INFLIGHT=2' 'STOP_WAIT=0' 'STOP_WAIT=x' \
    'EXPERT_ENV=TF_GLM_EXPERT_KERNEL=g53' 'ATTN_ENV=TF_GLM_PARTIALS_BF16=0'; do
  : > "$LOG"
  if env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" DRY_STATE="$ST" CTL_STATE="$CTL" "$bad" bash "$tmp/repo/start.sh" up > /dev/null 2>&1; then
    echo "dry run FAIL: start.sh up accepted $bad"; fail=1
  fi
  [ -s "$LOG" ] && { echo "dry run FAIL: $bad ran a remote command before the refusal"; fail=1; }
done

# recover (what extras/watch runs after two failed checks)
up() { : > "$LOG"; run "$tmp/repo/start.sh" up > "$tmp/r-up.out" 2>&1 || { cat "$tmp/r-up.out"; echo "dry run FAIL: up before a recover case"; exit 1; }; : > "$LOG"; }
recover() { local rc=0; run "$tmp/repo/start.sh" recover > "$tmp/recover.out" 2>&1 || rc=$?; echo $rc; }
up
[ "$(recover)" = 0 ] && grep -qF "recover: healthy, nothing to do" "$tmp/recover.out" && ! grep -qE 'docker (kill|run)|mcdma-rpcd (listen|connect)' "$LOG" \
  || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, healthy): it acted"; fail=1; }
# an expert node gone (its container exited): the two others get SIGTERM, the daemons stay, all three start again
rm -f "$ST/10.0.0.12.c.glm-afd-x1"; : > "$LOG"
[ "$(recover)" = 0 ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, expert gone): rc"; fail=1; }
check recover-x1 "$tmp/recover.out" "RECOVERED in"
check recover-x1 "$tmp/recover.out" "(restarted: containers)"
check recover-x1 "$LOG" $'10.0.0.10\tdocker kill --signal TERM glm-afd-attn'
check recover-x1 "$LOG" $'10.0.0.11\tdocker kill --signal TERM glm-afd-x0'
grep -qE 'SHUTDOWN|mcdma-rpcd (listen|connect)' "$LOG" && { echo "dry run FAIL (recover, expert gone): the daemons were touched"; fail=1; }
[ "$(grep -c 'docker run -d .*--name glm-afd-\(attn\|x0\|x1\) ' "$LOG")" = 3 ] || { echo "dry run FAIL (recover, expert gone): want 3 containers started"; fail=1; }
nokill recover-x1
# a listen daemon gone: containers and daemons restarted, in MCDMA's order
rm -f "$ST/10.0.0.11.d.x0-1"; : > "$LOG"
[ "$(recover)" = 0 ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, daemon gone): rc"; fail=1; }
check recover-d "$tmp/recover.out" "(restarted: containers, MCDMA daemons)"
k=$(grep -nF 'docker kill --signal TERM' "$LOG" | tail -1 | cut -d: -f1 || true); sd=$(grep -nF 'SHUTDOWN' "$LOG" | head -1 | cut -d: -f1 || true)
li=$(grep -nF 'mcdma-rpcd listen' "$LOG" | head -1 | cut -d: -f1 || true); bt=$(grep -nF -- '--name glm-afd-attn' "$LOG" | head -1 | cut -d: -f1 || true)
[ -n "$k" ] && [ -n "$sd" ] && [ -n "$li" ] && [ -n "$bt" ] && [ "$k" -lt "$sd" ] && [ "$sd" -lt "$li" ] && [ "$li" -lt "$bt" ] \
  || { echo "dry run FAIL (recover, daemon gone): order (TERM $k, SHUTDOWN $sd, listen $li, attention $bt)"; fail=1; }
nokill recover-d
# the second connect daemon gone (link 3 down): containers and daemons restarted, both connect daemons up again
rm -f "$ST/10.0.0.10.d.connect2"; : > "$LOG"
[ "$(recover)" = 0 ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, connect2 gone): rc"; fail=1; }
check recover-c2 "$tmp/recover.out" "links 6/8 up"
check recover-c2 "$tmp/recover.out" "(restarted: containers, MCDMA daemons)"
check recover-c2 "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/connect2.sock nohup ./mcdma-rpcd connect x0-3,'
[ -e "$ST/10.0.0.10.d.connect2" ] || { echo "dry run FAIL (recover, connect2 gone): not started again"; fail=1; }
nokill recover-c2
# a container that ignores SIGTERM: rc 3, nothing killed, the daemons left alone, nothing started
touch "$ST/unhealthy" "$ST/stubborn"; : > "$LOG"
[ "$(STOP_WAIT=2 recover)" = 3 ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, stubborn): want rc 3"; fail=1; }
check recover-stubborn "$tmp/recover.out" "still runs after SIGTERM: not killing it"
grep -qE 'SHUTDOWN|docker run -d' "$LOG" && { echo "dry run FAIL (recover, stubborn): it went on"; fail=1; }
nokill recover-stubborn
rm -f "$ST/stubborn" "$ST/unhealthy"
# a Spark whose memory does not come back: rc 3 (latch the watcher: a reboot is the owner's call)
: > "$LOG"; touch "$ST/unhealthy"; echo 20 > "$ST/mem"
rc=$(STOP_WAIT=2 MEM_WAIT=0 recover)
grep -qF "stranded GB10 memory" "$tmp/recover.out" && [ "$rc" = 3 ] || { tail -3 "$tmp/recover.out"; echo "dry run FAIL (recover, memory): want rc 3 (got '$rc')"; fail=1; }
grep -qE 'docker run -d' "$LOG" && { echo "dry run FAIL (recover, memory): it started something"; fail=1; }
rm -f "$ST/mem" "$ST/unhealthy"
# the watcher's call: a stack stopped by hand stays down (rc 2, nothing done); the lock (rc 75)
run "$tmp/repo/stop.sh" > /dev/null 2>&1 || true
: > "$LOG"
[ "$(CALLER=watch recover)" = 2 ] && [ ! -s "$LOG" ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover after stop.sh, watch): want rc 2 and nothing done"; fail=1; }
mkdir -p "$CTL"; flock "$CTL/ctl.lock" sleep 10 & lk=$!; sleep 0.3
[ "$(recover)" = 75 ] || { cat "$tmp/recover.out"; echo "dry run FAIL (recover, lock held): want rc 75"; fail=1; }
kill $lk 2>/dev/null || true

: > "$LOG"
run "$tmp/repo/build.sh" mcdma > "$tmp/mcdma.out" 2>&1 || { cat "$tmp/mcdma.out"; echo "dry run FAIL: build.sh mcdma"; fail=1; }
check mcdma "$LOG" "git checkout -q -f --detach e672c14ff9fc7b38994caf73025cf1588b4de74e"
check mcdma "$LOG" "make -s -C rpc CFLAGS='-std=c11 -O2 -Wall -Wextra -Werror -Wno-error=format-truncation'"
: > "$LOG"
run "$tmp/repo/download.sh" > "$tmp/dl.out" 2>&1 || { cat "$tmp/dl.out"; echo "dry run FAIL: download.sh"; fail=1; }
check download "$LOG" "hf download Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold --revision 76c0b5173166d2795dd48860f45d8224817f894c"
check download "$LOG" "hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a"
[ "$(grep -cF 'hf download incoai' "$LOG")" = 1 ] || { echo "dry run FAIL: the drafter must go to the attention host only"; fail=1; }
[ $fail = 0 ] && echo "dry run: OK (up at 4, 2 and 1 in flight, refusals, stop, recover x7, build mcdma, download)" || exit 1
