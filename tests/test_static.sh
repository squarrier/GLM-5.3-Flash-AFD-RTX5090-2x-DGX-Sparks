#!/bin/bash
# Static checks, no hardware needed: bash syntax, shellcheck (if installed), python compile,
# unit-file syntax, and that the topology helper derives the expected rank/peer lists.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
for f in build.sh download.sh start.sh stop.sh scripts/*.sh site/glm53f-afd-* extras/gb10-hostguard/install.sh; do
  bash -n "$f" || { echo "bash -n FAIL $f"; fail=1; }
done
if command -v shellcheck >/dev/null; then
  shellcheck -S error build.sh download.sh start.sh stop.sh scripts/*.sh site/glm53f-afd-* extras/gb10-hostguard/install.sh || fail=1
fi
python3 -m py_compile extras/watch/glm53f-afd-watch extras/watch/glm53f-afd-report extras/gb10-hostguard/gb10-hostguard.py || fail=1
for u in site/systemd/*.service; do grep -q '^ExecStart=' "$u" && grep -q '^Restart=no' "$u" || { echo "unit FAIL $u"; fail=1; }; done
tmp=$(mktemp -d); cp .env.example "$tmp/.env"; mkdir -p "$tmp/scripts"; cp scripts/lib.sh "$tmp/scripts/"
out=$(env -i PATH="$PATH" HOME="$HOME" bash -c ". '$tmp/scripts/lib.sh'; echo \"\$RANKS|\$PEERS|\$FIRST_NAME|\$BASE_URL|\${RANK_A[spark2]}\${RANK_B[spark2]}\"")
rm -rf "$tmp"
exp="10.10.1.11:8600,10.10.2.11:8600,10.10.1.12:8600,10.10.2.12:8600|10.10.1.11:8601,10.10.2.11:8601,10.10.1.12:8601,10.10.2.12:8601|glm-5.3-flash|http://10.0.0.10:8000|23"
[ "$out" = "$exp" ] || { echo "topology FAIL: $out"; fail=1; }
[ $fail = 0 ] && echo "static checks: OK" || exit 1
