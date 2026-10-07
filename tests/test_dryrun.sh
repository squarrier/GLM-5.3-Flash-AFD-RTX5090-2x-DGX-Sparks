#!/bin/bash
# Dry run, no hardware: start.sh up, stop.sh, build.sh ext/mcdma and download.sh against stand-in ssh/scp/curl that
# record every remote command instead of running it. Checks the command lines the scripts generate (the docker run
# flags, the serve/experts lines, the MCDMA daemons' lines) and the order of the steps.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/repo" "$tmp/bin"
tar cf - --exclude=.git --exclude=.env . | tar xf - -C "$tmp/repo"
cp .env.example "$tmp/repo/.env"
LOG=$tmp/remote.log; RUNS=$tmp/docker-run.log; : > "$LOG"; : > "$RUNS"

cat > "$tmp/bin/ssh" <<'EOF'
#!/bin/bash
# stand-in ssh: the last argument is the remote command; record it, answer what the scripts ask
host=""; for a in "$@"; do case $a in *@*) host=${a#*@} ;; esac; done
cmd=${!#}
printf '%s\t%s\n' "$host" "$cmd" >> "$DRY_LOG"
case $cmd in
  *"tar x"*) cat > /dev/null ;;
  "docker run -d "*) eval "set -- $cmd"; { printf '%s' "$host"; printf '\t%s' "$@"; echo; } >> "$DRY_RUNS"; echo 0123456789ab ;;
  *MemAvailable*) echo 120 ;;
  *"pgrep -xc"*) echo 0 ;;
  *"pgrep -x mcdma-rpcd"*) exit 1 ;;
  *STATUS*) printf 'PEER x0 up 10.10.1.11\nPEER x1 up 10.10.1.12\nPEER x0-1 up 10.10.1.11\nPEER x1-1 up 10.10.1.12\n' ;;
  *"{{.State.Status}} {{.State.ExitCode}}"*) echo "exited 0" ;;
  *"docker image inspect -f"*) echo "sha256:0123456789abcdef0123" ;;
  *"curl -fsS -m 4 http://127.0.0.1"*) echo '{"status":"ok"}' ;;
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
echo '{"choices":[{"message":{"role":"assistant","content":"OK"}}]}'
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/scp" "$tmp/bin/curl"
run() { env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" bash "$@"; }

fail=0
check() {  # label, file, fixed string that must appear
  grep -qF -- "$3" "$2" || { echo "dry run FAIL ($1): missing: $3"; fail=1; }
}
run "$tmp/repo/start.sh" up > "$tmp/up.out" 2>&1 || { cat "$tmp/up.out"; echo "dry run FAIL: start.sh up"; exit 1; }
check up "$tmp/up.out" "check OK"
check up "$tmp/up.out" "MCDMA: 4 links up (x0 x1 x0-1 x1-1"
check up "$tmp/up.out" "smoke OK"
# the prebuild: no model, no network, the tree's own extension cache, stale locks removed
check prebuild "$RUNS" $'\t--network\tnone\t-v\t/srv/glm-afd/ext-'
check prebuild "$RUNS" 'exec python /p/prebuild_ext.py'
# the daemons (MCDMA_INFLIGHT=2): two listen ends on each Spark, link j on control port 18820 + j, then one connect
# end on the attention host with the four peers
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/x0.sock nohup ./mcdma-rpcd listen x0 rocep1s0f0 3 4096 10.10.1.11:18820 20 36'
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/x0-1.sock nohup ./mcdma-rpcd listen x0-1 rocep1s0f0 3 4096 10.10.1.11:18821 20 36'
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/x1.sock nohup ./mcdma-rpcd listen x1 rocep1s0f0 3 4096 10.10.1.12:18820 20 36'
check daemons "$LOG" 'MCDMA_RPCD_SOCKET=/srv/glm-afd/mcdma/x1-1.sock nohup ./mcdma-rpcd listen x1-1 rocep1s0f0 3 4096 10.10.1.12:18821 20 36'
check daemons "$LOG" './mcdma-rpcd connect x0,10.10.1.11,18820,mlx5_0,3,4096,20,36 x1,10.10.1.12,18820,mlx5_0,3,4096,20,36 x0-1,10.10.1.11,18821,mlx5_0,3,4096,20,36 x1-1,10.10.1.12,18821,mlx5_0,3,4096,20,36 > connect.log'
[ "$(grep -cF 'mcdma-rpcd listen' "$LOG")" = 4 ] || { echo "dry run FAIL (daemons): want 4 listen daemons"; fail=1; }
check mailboxes "$LOG" 'test -e /dev/shm/mcdma-rpc.x1-1'
# the attention node
a=$(grep -P '^10\.0\.0\.10\t' "$RUNS" | grep -F 'tensorfold serve' || true)
for s in $'--restart=no\t--oom-score-adj=1000' $'--name\tglm-afd-attn' $'--gpus\tall\t--network\thost\t--ipc=host\t--device\t/dev/infiniband\t--ulimit\tmemlock=-1\t--cap-add\tIPC_LOCK' \
    $'-e\tTF_AFD_TRANSPORT=mcdma' $'-e\tTF_AFD_MCDMA_LINKS=x0,x1' $'-e\tTF_AFD_MCDMA_MODE=auto' $'-e\tTF_AFD_EAGER=1' $'-e\tTENSORFOLD_MEMORY_RESERVE_GIB=2' \
    $'-e\tTF_GLM_DENSE=q4' $'-e\tTF_GLM_KV=fp8' $'-e\tTF_GLM_CACHE_GIB=8.5' $'-e\tTF_GLM_CACHE_ENTRIES=20' $'-e\tTF_GLM_MCDMA_INFLIGHT=2' \
    $'-e\tTF_GLM_SHARED_PREFIX=1' $'-e\tTF_GLM_STREAM_SMOOTH_MS=400' \
    $'-e\tTF_GLM_TOOL_CALLS=1' $'-e\tTF_GLM_KDA_CHUNKED=1' $'-e\tTF_GLM_PREFILL_PAIRS=1' $'-e\tTF_GLM_DFLASH_POLICY=fnc5:0.2' \
    $'-e\tTF_GLM_CACHE_ROOM=1' $'-e\tTF_GLM_PREFILL_ORDER=sjf' $'-e\tTF_GLM_MULTI_WINDOW=64' \
    $'-e\tTF_GLM_KEPT_HOST=1' $'-e\tTF_GLM_HOST_CACHE_GIB=24' $'-e\tTF_GLM_FILL_PAIRS=1' \
    $'-e\tTF_GLM_DECIDE_THEN_COPY=1' $'-e\tTF_GLM_CAPACITY_STATUS=1' $'-e\tTF_GLM_DELIVERY_ABORT=1' \
    $'-v\t/srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold:/srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold:ro' \
    'tensorfold serve /srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold --experts remote' '--master 10.10.1.10 --master-port 29551' \
    '--drafter /srv/models/GLM-5.3-Flash-DFlash2 --context 262144 --parallel 8 --max-tokens 32768' \
    '--host 0.0.0.0 --port 8000 --name GLM-5.3-Flash-EXL3 --alias glm-5.3-flash > /afd/logs/attn.log 2>&1'; do
  case "$a" in *"$s"*) ;; *) echo "dry run FAIL (attention): missing: $s"; fail=1 ;; esac
done
case "$a" in *TF_GLM_EXL3_*|*TF_GLM_EXPERT_KERNEL*) echo "dry run FAIL (attention): an expert-node switch"; fail=1 ;; esac
case "$a" in *fnc7:0.3*|*TF_GLM_PREFILL_LANES*|*TF_GLM_WIRE_FP8*|*TF_GLM_PREFILL_ROWS*|*TF_GLM_SHARED_PREFIX_COPY*|*TF_GLM_QUEUED_CANCEL*|*TF_GLM_COMPACT_BEFORE_EVICT*|*TF_GLM_ASSISTANT_ENDS*|*TF_GLM_CAP_SHARED_RECENCY*|*TF_GLM_MAX_QUEUED*) echo "dry run FAIL (attention): a switch .env.example leaves off"; fail=1 ;; esac
# the expert nodes: their own link and socket, their decode and prompt kernel switches, no attention-only settings
for h in 0 1; do
  x=$(grep -P "^10\.0\.0\.1$((h + 1))\t" "$RUNS" | grep -F 'tensorfold experts' || true)
  for s in $'--name\tglm-afd-x'$h $'-e\tTF_AFD_MCDMA_LINK=x'$h $'-e\tMCDMA_RPCD_SOCKET=/mcdma/x'$h'.sock' \
      $'-e\tTF_GLM_EXL3_DEC=1' $'-e\tTF_GLM_EXL3_LOADS=nc' $'-e\tTF_GLM_EXL3_PROMPT=1' $'-e\tTF_GLM_EXPERT_KERNEL=g53' \
      "tensorfold experts /srv/models/GLM-5.3-Flash-EXL3-4bpw-TensorFold" "--rank $h --master 10.10.1.10 --master-port 29551" "/afd/logs/expert$h.log"; do
    case "$x" in *"$s"*) ;; *) echo "dry run FAIL (expert $h): missing: $s"; fail=1 ;; esac
  done
  case "$x" in *TF_GLM_DENSE*|*TF_GLM_KV*|*--drafter*|*TF_GLM_KDA_CHUNKED*|*TF_GLM_DFLASH_POLICY*|*TF_GLM_PREFILL_PAIRS*|*TF_GLM_MCDMA_INFLIGHT*|*TF_GLM_CACHE_ROOM*|*TF_GLM_PREFILL_ORDER*|*TF_GLM_MULTI_WINDOW*|*TF_GLM_KEPT_HOST*|*TF_GLM_HOST_CACHE_GIB*|*TF_GLM_FILL_PAIRS*|*TF_GLM_DECIDE_THEN_COPY*|*TF_GLM_CAPACITY_STATUS*|*TF_GLM_DELIVERY_ABORT*) echo "dry run FAIL (expert $h): attention-only settings"; fail=1 ;; esac
done
# order: listen daemons, connect daemon, attention container, expert containers
first_serve=$(grep -nF -- '--name glm-afd-attn' "$LOG" | head -1 | cut -d: -f1 || true)
last_daemon=$(grep -nF 'mcdma-rpcd connect' "$LOG" | tail -1 | cut -d: -f1 || true)
first_expert=$(grep -nF -- '--name glm-afd-x0' "$LOG" | head -1 | cut -d: -f1 || true)
last_listen=$(grep -nF 'mcdma-rpcd listen' "$LOG" | tail -1 | cut -d: -f1 || true)
[ -n "$first_serve" ] && [ -n "$last_daemon" ] && [ -n "$first_expert" ] && [ -n "$last_listen" ] \
  && [ "$last_listen" -lt "$last_daemon" ] && [ "$last_daemon" -lt "$first_serve" ] && [ "$first_serve" -lt "$first_expert" ] \
  || { echo "dry run FAIL: order (listen $last_listen, connect $last_daemon, attention $first_serve, expert $first_expert)"; fail=1; }

: > "$LOG"
run "$tmp/repo/stop.sh" > "$tmp/stop.out" 2>&1 || { cat "$tmp/stop.out"; echo "dry run FAIL: stop.sh"; fail=1; }
c=$(grep -nF 'connect.sock' "$LOG" | head -1 | cut -d: -f1 || true); l=$(grep -nF 'x0.sock' "$LOG" | head -1 | cut -d: -f1 || true)
r=$(grep -nF 'docker rm -f' "$LOG" | head -1 | cut -d: -f1 || true)
[ -n "$r" ] && [ -n "$c" ] && [ -n "$l" ] && [ "$r" -lt "$c" ] && [ "$c" -lt "$l" ] || { echo "dry run FAIL: stop order"; fail=1; }
for s in x0.sock x0-1.sock x1.sock x1-1.sock; do grep -F "/srv/glm-afd/mcdma/$s" "$LOG" | grep -qF SHUTDOWN || { echo "dry run FAIL: no SHUTDOWN to $s"; fail=1; }; done
grep -qE 'kill -9|KILL' "$LOG" && { echo "dry run FAIL: a SIGKILL"; fail=1; }

# one link a Spark (MCDMA_INFLIGHT=1, a caller export): v2.0's daemon lines, no in-flight switch on the attention node
: > "$LOG"; : > "$RUNS"
env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" MCDMA_INFLIGHT=1 bash "$tmp/repo/start.sh" up > "$tmp/up1.out" 2>&1 \
  || { cat "$tmp/up1.out"; echo "dry run FAIL: start.sh up at MCDMA_INFLIGHT=1"; exit 1; }
check up1 "$tmp/up1.out" "MCDMA: 2 links up (x0 x1"
check up1 "$LOG" './mcdma-rpcd connect x0,10.10.1.11,18820,mlx5_0,3,4096,20,36 x1,10.10.1.12,18820,mlx5_0,3,4096,20,36 > connect.log'
[ "$(grep -cF 'mcdma-rpcd listen' "$LOG")" = 2 ] || { echo "dry run FAIL (up1): want 2 listen daemons"; fail=1; }
grep -qF 'TF_GLM_MCDMA_INFLIGHT' "$RUNS" && { echo "dry run FAIL (up1): TF_GLM_MCDMA_INFLIGHT at one link"; fail=1; }
# a bad MCDMA_INFLIGHT and an in-flight switch in ATTN_ENV are refused before anything runs
for bad in 'MCDMA_INFLIGHT=4' 'MCDMA_INFLIGHT=x' 'ATTN_ENV=TF_GLM_MCDMA_INFLIGHT=2'; do
  : > "$LOG"
  if env -i PATH="$tmp/bin:$PATH" HOME="$HOME" DRY_LOG="$LOG" DRY_RUNS="$RUNS" "$bad" bash "$tmp/repo/start.sh" up > /dev/null 2>&1; then
    echo "dry run FAIL: start.sh up accepted $bad"; fail=1
  fi
  [ -s "$LOG" ] && { echo "dry run FAIL: $bad ran a remote command before the refusal"; fail=1; }
done

: > "$LOG"
run "$tmp/repo/build.sh" mcdma > "$tmp/mcdma.out" 2>&1 || { cat "$tmp/mcdma.out"; echo "dry run FAIL: build.sh mcdma"; fail=1; }
check mcdma "$LOG" "git checkout -q -f --detach e672c14ff9fc7b38994caf73025cf1588b4de74e"
check mcdma "$LOG" "make -s -C rpc CFLAGS='-std=c11 -O2 -Wall -Wextra -Werror -Wno-error=format-truncation'"
: > "$LOG"
run "$tmp/repo/download.sh" > "$tmp/dl.out" 2>&1 || { cat "$tmp/dl.out"; echo "dry run FAIL: download.sh"; fail=1; }
check download "$LOG" "hf download Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold --revision 76c0b5173166d2795dd48860f45d8224817f894c"
check download "$LOG" "hf download incoai/GLM-5.3-Flash-DFlash2 --revision bf582e4eacc1810f76656d1811693ff6c6737d2a"
[ "$(grep -cF 'hf download incoai' "$LOG")" = 1 ] || { echo "dry run FAIL: the drafter must go to the attention host only"; fail=1; }
[ $fail = 0 ] && echo "dry run: OK (up at 2 and 1 in flight, refusals, stop, build mcdma, download)" || exit 1
