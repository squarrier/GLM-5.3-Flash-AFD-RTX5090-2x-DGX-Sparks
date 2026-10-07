#!/bin/bash
# Static checks, no hardware needed: bash syntax, shellcheck (if installed), Python syntax, the topology lib.sh
# derives from .env.example, the TF_GLM_* env check, the patch series' shape and credits, the scrub, and the files
# kept byte-identical (AGENTS.md, the host guard's code).
# With TF_REPO=<a TensorFold clone that has the pinned tag>, also applies patches/ in order with `git apply --check`.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
scripts=(build.sh download.sh start.sh stop.sh scripts/lib.sh tools/export_patches.sh tests/test_static.sh tests/test_dryrun.sh
         extras/gb10-hostguard/install.sh)
for f in "${scripts[@]}"; do bash -n "$f" || { echo "bash -n FAIL $f"; fail=1; }; done
if command -v shellcheck >/dev/null; then shellcheck -S error "${scripts[@]}" || fail=1; fi
python3 -c 'import sys; [compile(open(f).read(), f, "exec") for f in sys.argv[1:]]' scripts/prebuild_ext.py tools/patch_credits.py \
  extras/gb10-hostguard/gb10-hostguard.py || { echo "python syntax FAIL"; fail=1; }

# topology and the env check, from the example values only (no exported variable may leak in)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/scripts" "$tmp/patches"; cp .env.example "$tmp/.env"; cp scripts/lib.sh "$tmp/scripts/"; cp patches/*.patch "$tmp/patches/"
out=$(env -i PATH="$PATH" HOME="$HOME" bash -c ". '$tmp/scripts/lib.sh'
  echo \"\${HOST[attn]} \${HOST[x0]} \${HOST[x1]}|\${FABRIC_IP[x1]}|\${LINK[x0]},\${LINK[x1]}|\${EXPERT_RANK[x1]}|\${RDMA_DEV[attn]} \${RDMA_DEV[x0]}|\$BASE_URL|\${#TREE_ID}|\${EXT_DIR%-*}|\${CONTAINER[x1]}|\$MCDMA_INFLIGHT|\$(links | tr '\n' ' ')\"")
exp="10.0.0.10 10.0.0.11 10.0.0.12|10.10.1.12|x0,x1|1|mlx5_0 rocep1s0f0|http://10.0.0.10:8000|12|/srv/glm-afd/ext|glm-afd-x1|2|x0 x1 x0-1 x1-1 "
[ "$out" = "$exp" ] || { echo "topology FAIL: $out"; fail=1; }
env_ok=$(env -i PATH="$PATH" HOME="$HOME" bash -c ". '$tmp/scripts/lib.sh'; envpairs ATTN_ENV \"\$ATTN_ENV\"")
case "$env_ok" in *"-e TF_GLM_SHARED_PREFIX=1"*"-e TF_GLM_STREAM_SMOOTH_MS=400"*"-e TF_GLM_KDA_CHUNKED=1"*"-e TF_GLM_PREFILL_PAIRS=1"*"-e TF_GLM_DFLASH_POLICY=fnc5:0.2"*"-e TF_GLM_CACHE_ROOM=1"*"-e TF_GLM_KEPT_HOST=1"*"-e TF_GLM_HOST_CACHE_GIB=24"*"-e TF_GLM_FILL_PAIRS=1"*"-e TF_GLM_DECIDE_THEN_COPY=1"*"-e TF_GLM_CAPACITY_STATUS=1"*"-e TF_GLM_DELIVERY_ABORT=1"*) ;; *) echo "envpairs FAIL: $env_ok"; fail=1 ;; esac
xenv=$(env -i PATH="$PATH" HOME="$HOME" bash -c ". '$tmp/scripts/lib.sh'; envpairs EXPERT_ENV \"\$EXPERT_ENV\"")
case "$xenv" in *"-e TF_GLM_EXL3_DEC=1"*"-e TF_GLM_EXL3_LOADS=nc"*"-e TF_GLM_EXL3_PROMPT=1"*"-e TF_GLM_EXPERT_KERNEL=g53"*) ;; *) echo "envpairs FAIL (experts): $xenv"; fail=1 ;; esac
for bad in 'TF_GLM_X=1;id' 'FOO=1' 'TF_GLM_X=$(id)' 'TF_GLM_X=1 PYTHONPATH=/x'; do
  if env -i PATH="$PATH" HOME="$HOME" bash -c ". '$tmp/scripts/lib.sh'; envpairs T '$bad'" >/dev/null 2>&1; then
    echo "envpairs FAIL: accepted '$bad'"; fail=1
  fi
done

# the patch series: numbered 0001..N with no gap; mail headers; credits on her commits, on the ports, on the changes to
# her ported code and on TensorFold's own later commits; Hugh Madden and T.J. Purtell named in the patches that bring
# glm53f-afd's designs and code
n=0
for p in patches/*.patch; do
  n=$((n + 1)); b=$(basename "$p")
  [ "${b:0:4}" = "$(printf '%04d' $n)" ] || { echo "patch numbering FAIL at $b"; fail=1; }
  grep -q '^From: ' "$p" && grep -q '^Subject: ' "$p" || { echo "patch headers FAIL $b"; fail=1; }
  subj=$(awk '/^Subject: /{s=$0; while ((getline l) > 0 && l ~ /^ /) s = s l; print s; exit}' "$p")   # unfolded
  if grep -q '^From: MiaAI-Lab <MiaAI-Lab@users.noreply.github.com>' "$p"; then
    grep -q "^Credit: MiaAI-Lab's pull request ashhart/TensorFold#" "$p" || { echo "PR credit FAIL $b"; fail=1; }
  elif grep -q "^Credit: TensorFold's own commit [0-9a-f]\{40\} (v" "$p"; then
    grep -q '^From: Scott Quarrier ' "$p" && { echo "upstream author FAIL $b"; fail=1; }
  elif [[ "$subj" == *"MiaAI-Lab recipe patch"* ]] || grep -q "^Credit: this changes MiaAI-Lab's code" "$p"; then
    grep -qi '^Co-authored-by: MiaAI-Lab <MiaAI-Lab@users.noreply.github.com>' "$p" || { echo "Co-authored-by FAIL $b"; fail=1; }
    grep -qE "^Credit: (ported from MiaAI-Lab's|this changes MiaAI-Lab's code from her|parts reimplemented from MiaAI-Lab's) GLM-5.3-Flash EXL3 2x DGX Sparks recipe" "$p" \
      || { echo "port credit FAIL $b"; fail=1; }
  else
    grep -q '^From: Scott Quarrier <squarrier@users.noreply.github.com>' "$p" || { echo "author FAIL $b"; fail=1; }
  fi
  if [[ "$subj" == *"glm53f-afd's"* ]]; then
    grep -qF 'Hugh Madden (@dangerm00se, github.com/hughmadden) authored' "$p" || { echo "Hugh Madden credit FAIL $b"; fail=1; }
  fi
  if [[ "$subj" == *"TF_GLM_WIRE_FP8"* ]]; then
    grep -qF 'T.J. Purtell (@wrldsuksgo2mars' "$p" || { echo "T.J. Purtell credit FAIL $b"; fail=1; }
  fi
done
[ $n -gt 0 ] || { echo "no patches"; fail=1; }

# RELEASE=1: the publish gate. No placeholder marker may be left (patches/ is TensorFold code, where "[pending" is a
# Python expression, so it is not searched).
if [ "${RELEASE:-0}" = 1 ] && grep -rnF '[pending' --exclude-dir=.git --exclude-dir=patches --exclude=test_static.sh . ; then
  echo "RELEASE FAIL: placeholder markers left"; fail=1
fi

# the scrub: no private-range addresses outside upstream code (examples use 10.0.0.x and 10.10.x.x)
if grep -rnIE '192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.' --exclude-dir=.git --exclude-dir=patches --exclude=test_static.sh . ; then echo "scrub FAIL"; fail=1; fi
head -3 LICENSE | grep -q 'Apache License' || { echo "LICENSE FAIL"; fail=1; }

# files kept byte-identical: Mia's AGENTS.md (her credit rules), and v1.0's host guard code
while read -r sum f; do
  [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sum" ] || { echo "kept file changed FAIL $f"; fail=1; }
done <<'EOF'
81a6dd958181aa1d581cb24d54d30a327842e955bd42b40256e54bfc1abfa8cb AGENTS.md
09c6d5fcdcfc2cef51c6e7b52b325da9cae6b2e602846ac20d9975ffa7216425 extras/gb10-hostguard/gb10-hostguard.py
38f20b9ea216e80108d659dc862dc889c5b3d5dbd334ecd796332c251e81eeb0 extras/gb10-hostguard/install.sh
EOF

if [ -n "${TF_REPO:-}" ]; then   # the series on the pinned tag, in order
  tag=$(sed -n 's/^TF_TAG=//p' .env.example); want=$(sed -n 's/^TF_COMMIT=//p' .env.example)
  [ "$(git -C "$TF_REPO" rev-parse "$tag^{commit}")" = "$want" ] || { echo "TF_REPO: $tag is not $want"; fail=1; }
  wt="$tmp/tree"; git -C "$TF_REPO" worktree add -q --detach "$wt" "$want"
  for p in patches/*.patch; do
    git -C "$wt" apply --check "$PWD/$p" || { echo "git apply --check FAIL $(basename "$p")"; fail=1; break; }
    git -C "$wt" -c user.name=t -c user.email=t@localhost am -q "$PWD/$p" 2>/dev/null || { echo "git am FAIL $(basename "$p")"; fail=1; break; }
  done
  git -C "$TF_REPO" worktree remove --force "$wt"
  [ $fail = 0 ] && echo "patches: all $n apply in order on $tag"
fi
[ $fail = 0 ] && echo "static checks: OK ($n patches)" || exit 1
