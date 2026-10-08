#!/bin/bash
# extras/watch, no hardware: glm53f-afd-watch against a small stand-in server on 127.0.0.1 (a random port) with a
# stand-in recover command and a stand-in ssh. Checks: a healthy run is silent and records the checks and the hosts;
# a stack that answers /health but fails its replies is recovered after two failed runs (GLM_AFD_CALLER=watch); the
# hold-offs (./stop.sh's stop marker, maintenance mode, the control lock); the circuit breaker; recover rc 3 latches
# at once; glm53f-afd-report rolls the records up.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
srv=""
cleanup() { [ -n "$srv" ] && kill "$srv" 2>/dev/null; rm -rf "$tmp"; }
trap cleanup EXIT
mkdir -p "$tmp/bin"
cat > "$tmp/server.py" <<'EOF'
import json, re, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
state = {"degraded": False, "tokens": 0}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True, "busy": False, "requests_running": 0,
                                   "completion_tokens_total": state["tokens"], "iteration_s": 0})
        if self.path == "/v1/models":
            return self.send(200, {"object": "list", "data": [{"id": "GLM-5.3-Flash-EXL3"}, {"id": "glm-5.3-flash"}]})
        self.send(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/ctl":
            state.update(body); return self.send(200, {"ok": True})
        if state["degraded"]:
            return self.send(503, {"error": {"message": "the AFD expert nodes are gone; restart the serving stack"}})
        m = re.search(r"exactly: (\S+)", json.dumps(body.get("messages")))
        text = m.group(1).rstrip('"}]') if m else "OK"
        state["tokens"] += 3
        chunk = json.dumps({"choices": [{"delta": {"content": text}}]})
        self.send(200, f"data: {chunk}\n\ndata: [DONE]\n\n".encode(), "text/event-stream")
srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
EOF
python3 "$tmp/server.py" > "$tmp/port" & srv=$!
for _ in $(seq 50); do [ -s "$tmp/port" ] && break; sleep 0.1; done
PORT=$(cat "$tmp/port"); BASE=http://127.0.0.1:$PORT
ctl() { curl -s -m 5 -X POST "$BASE/ctl" -H 'Content-Type: application/json' -d "$1" > /dev/null; }

cat > "$tmp/bin/ssh" <<'EOF'
#!/bin/bash
cmd=${!#}
case $cmd in
  *MemAvailable*pgrep*)   # one kernel line 100 s ago: inside a first run's window (180 s back), never a second time
    s=${cmd#*--since @}; s=${s%% *}; t=$(date +%s); k=0; [ "$s" -lt $((t - 100)) ] && k=1
    printf 'mem 110.5 0.0\npsi 0.00\ngpu 35, 1024, 120.5, 55, 2100\nctr glm-afd-x0 running\nrpcd 2\nkern %s %s\n' "$k" "$t" ;;
  *connect2.sock*STATUS*) printf 'PEER x0-3 up g\nPEER x1-3 up h\n' ;;     # link 3 of both: the second connect daemon
  *STATUS*) printf 'PEER x0 up a\nPEER x1 up b\nPEER x0-1 up c\nPEER x1-1 up d\nPEER x0-2 up e\nPEER x1-2 up f\n' ;;
  *"stat -c %i"*) echo "42 10" ;;
  *"sed -n"*) echo "[tensorfold] a quiet line" ;;
esac
EOF
cat > "$tmp/bin/recover" <<EOF
#!/bin/bash
echo "caller=\${GLM_AFD_CALLER:-none}" >> "$tmp/recover.log"
[ -s "$tmp/recover_rc" ] && exit "\$(cat "$tmp/recover_rc")"
curl -s -m 5 -X POST "$BASE/ctl" -H 'Content-Type: application/json' -d '{"degraded": false}' > /dev/null
echo "RECOVERED in 1 s (stand-in)"
EOF
chmod +x "$tmp/bin/ssh" "$tmp/bin/recover"
cp .env.example "$tmp/env"
fail=0
W() {  # state dir, then extra env: one watcher run
  local st=$1; shift
  env -i PATH="$tmp/bin:$PATH" HOME="$HOME" WATCH_ENV_FILE="$tmp/env" WATCH_BASE="$BASE" CTL_STATE="$st" \
    WATCH_RECOVER="$tmp/bin/recover" WATCH_BENCH=0 WATCH_PROBE_TIMEOUT=10 "$@" python3 extras/watch/glm53f-afd-watch
}
rec() { tail -1 "$1"/watch/telemetry/*.jsonl; }
has() { python3 -c "import json,sys; r=json.loads(sys.argv[1]); sys.exit(0 if eval(sys.argv[2]) else 1)" "$1" "$2"; }

st=$tmp/a; out=$(W "$st"); r=$(rec "$st")
# .env.example's MCDMA_INFLIGHT=4: eight links, six on the connect daemon and two on the second one, both read
[ -z "$out" ] && has "$r" "r['healthy'] and r['models_ok'] and r['probe']['ok'] and r['x0']['avail_gib'] == 110.5 and r['links_up'] == r['links_want'] == 8" \
  || { echo "watch FAIL (healthy): '$out' $r"; fail=1; }
st2=$tmp/a2; out=$(W "$st2" MCDMA_INFLIGHT=2); r=$(rec "$st2")   # two in flight: the first connect daemon only
has "$r" "r['links_want'] == 4 and r['links_up'] == 6" || { echo "watch FAIL (two in flight reads connect.sock only): $r"; fail=1; }
out=$(W "$st"); r2=$(rec "$st")
has "$r" "r['x0']['kern_events'] == 1 and r['x0']['kern_mark'] > 0" && has "$r2" "r['x0']['kern_events'] == 0" \
  && [ "$(wc -l < "$st/watch/events.log")" = 3 ] \
  && STATE_DIR=$st/watch python3 extras/watch/glm53f-afd-report | grep -E '^\| x0 \|.*\| 1 \|$' > /dev/null \
  || { echo "watch FAIL (a kernel line is counted once, from the previous run's mark): $r2"; fail=1; }
st=$tmp/b; ctl '{"degraded": true}'
o1=$(W "$st"); r1=$(rec "$st"); o2=$(W "$st"); r2=$(rec "$st")
[ -z "$o1" ] && has "$r1" "r['fails'] == 1 and 'recover' not in r" || { echo "watch FAIL (first failed run acted): '$o1'"; fail=1; }
has "$r2" "r['recover']['rc'] == 0" && grep -q "recovered in" <<< "$o2" && grep -q "caller=watch" "$tmp/recover.log" \
  || { echo "watch FAIL (recover after two failed runs): '$o2' $r2"; fail=1; }
: > "$tmp/recover.log"; ctl '{"degraded": true}'
st=$tmp/c; mkdir -p "$st"; echo "./stop.sh all" > "$st/stopped"
o=$(W "$st"; W "$st"; W "$st"); r=$(rec "$st")
[ -z "$o" ] && [ ! -s "$tmp/recover.log" ] && has "$r" "r['hold'].startswith('stopped by hand')" || { echo "watch FAIL (stop marker): '$o' $r"; fail=1; }
st=$tmp/d; mkdir -p "$st/watch"; echo maintenance > "$st/watch/mode"
o=$(W "$st"; W "$st"); [ -z "$o" ] && [ ! -s "$tmp/recover.log" ] || { echo "watch FAIL (maintenance): '$o'"; fail=1; }
st=$tmp/e; mkdir -p "$st"; flock "$st/ctl.lock" sleep 20 & lk=$!; sleep 0.3
o=$(W "$st"; W "$st"); r=$(rec "$st"); kill $lk 2>/dev/null || true
[ -z "$o" ] && [ ! -s "$tmp/recover.log" ] && has "$r" "r['hold'].startswith('a start.sh')" || { echo "watch FAIL (control lock): '$o' $r"; fail=1; }
st=$tmp/f; mkdir -p "$st/watch"; date +%s > "$st/watch/recoveries"
o=$(W "$st" WATCH_MAX_RECOVERIES=1; W "$st" WATCH_MAX_RECOVERIES=1)
[ -e "$st/watch/latched" ] && grep -q "circuit breaker LATCHED" <<< "$o" && [ ! -s "$tmp/recover.log" ] || { echo "watch FAIL (circuit breaker): '$o'"; fail=1; }
st=$tmp/g; echo 3 > "$tmp/recover_rc"
o=$(W "$st"; W "$st")
[ -e "$st/watch/latched" ] && grep -q "needs you" <<< "$o" || { echo "watch FAIL (rc 3 latches): '$o'"; fail=1; }
: > "$tmp/recover_rc"; ctl '{"degraded": false}'
STATE_DIR=$tmp/b/watch python3 extras/watch/glm53f-afd-report > "$tmp/report.md"
grep -q "Availability" "$tmp/report.md" && grep -q "rc 0 in" "$tmp/report.md" || { echo "watch FAIL (report)"; cat "$tmp/report.md"; fail=1; }
[ $fail = 0 ] && echo "watch: OK (healthy, kernel lines once, recover after 2 failed runs, stop marker, maintenance, lock, breaker, rc 3, report)" || exit 1
