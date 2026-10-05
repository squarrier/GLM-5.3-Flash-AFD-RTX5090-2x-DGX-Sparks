#!/usr/bin/env bash
# tools/export_patches.sh HEAD — export TensorFold commits BASE..HEAD into patches/ and prove they apply.
#
#   TF_REPO      a TensorFold git clone holding BASE and HEAD (default: the current directory)
#   BASE         the upstream release the series applies to (default: v0.6.5)
#   BASE_COMMIT  BASE's full sha, checked (default: v0.6.5's)
#   OUT_DIR      where the patches go (default: this repo's patches/); its *.patch files are replaced
#   PUBLIC_IDENT the author our own commits must carry
#
# Rules, each checked (the run stops on a violation):
#   - lab-only commits are dropped: a subject starting "lab:", or a message saying "lab branch only";
#   - merge commits are refused;
#   - "(cherry picked from commit ...)" lines are dropped; nothing else in a message changes;
#   - MiaAI-Lab's commits keep her authorship and must be her TensorFold pull-request commits
#     (tools/credits.tsv); every other commit must carry PUBLIC_IDENT; ports of her recipe patches must keep
#     "Co-authored-by: MiaAI-Lab". Each patch gets its credit lines in the notes (tools/patch_credits.py);
#   - in a temporary worktree at BASE, each patch passes `git apply --check` with the ones before it applied
#     (`git am`), and keeps its author and message;
#   - the rebuilt tree equals HEAD's tree, or, when lab-only commits were dropped, HEAD's tree once those
#     commits are re-applied on top.
# Prints a table (number, kind, source commit, author, subject) on stdout; keep it with the release evidence.
set -euo pipefail
HEAD_REF=${1:?usage: tools/export_patches.sh HEAD}
HERE=$(cd "$(dirname "$0")" && pwd)
TF_REPO=${TF_REPO:-.}
BASE=${BASE:-v0.6.5}
BASE_COMMIT=${BASE_COMMIT:-609ca419abecebdc5a059498a613680bd3aa847f}
OUT_DIR=${OUT_DIR:-$HERE/../patches}
PUBLIC_IDENT=${PUBLIC_IDENT:-Scott Quarrier <squarrier@users.noreply.github.com>}
G() { git -C "$TF_REPO" "$@"; }

[ "$(G rev-parse "$BASE^{commit}")" = "$BASE_COMMIT" ] || { echo "export: $BASE is not $BASE_COMMIT in $TF_REPO" >&2; exit 1; }
HEAD_SHA=$(G rev-parse "$HEAD_REF^{commit}")
G merge-base --is-ancestor "$BASE_COMMIT" "$HEAD_SHA" || { echo "export: $HEAD_REF does not contain $BASE" >&2; exit 1; }
[ -z "$(G rev-list --merges "$BASE_COMMIT..$HEAD_SHA")" ] || { echo "export: merge commits in $BASE..$HEAD_REF" >&2; exit 1; }

work=$(mktemp -d "${TMPDIR:-/tmp}/tf-export.XXXXXX")
cleanup() { G worktree remove --force "$work/tree" >/dev/null 2>&1 || true; rm -rf "$work"; }
trap cleanup EXIT
mkdir -p "$work/out" "$OUT_DIR"

keep=() lab=()
for c in $(G rev-list --reverse "$BASE_COMMIT..$HEAD_SHA"); do
  if G log -1 --format=%s "$c" | grep -qE '^lab:' || G log -1 --format=%B "$c" | grep -qi 'lab branch only'; then
    lab+=("$c")
  else
    keep+=("$c")
  fi
done
[ ${#keep[@]} -gt 0 ] || { echo "export: no commits to export" >&2; exit 1; }

n=0
for c in "${keep[@]}"; do
  n=$((n + 1))
  f=$(G format-patch --zero-commit --no-signature --start-number "$n" -1 "$c" -o "$work/out")
  [ -s "$f" ] || { echo "export: format-patch wrote nothing for $c" >&2; exit 1; }
  G show "$c" -- THIRD_PARTY_NOTICES.md | grep -E '^\+' > "$work/added" || true
  line=$(python3 "$HERE/patch_credits.py" "$f" "$HERE/credits.tsv" "$PUBLIC_IDENT" "$work/added")
  printf '%04d\t%s\t%s\n' "$n" "$(G rev-parse --short "$c")" "$line" >> "$work/table"
done

# prove the series: apply in order on BASE, each kept author and message, then the tree
G worktree add -q --detach "$work/tree" "$BASE_COMMIT"
T() { git -C "$work/tree" -c user.name=export -c user.email=export@localhost "$@"; }
i=0
for p in "$work"/out/*.patch; do
  c=${keep[$i]}; i=$((i + 1))
  T apply --check "$p" || { echo "export: git apply --check FAILED: $(basename "$p")" >&2; exit 1; }
  T am -q "$p" || { echo "export: git am FAILED: $(basename "$p")" >&2; exit 1; }
  [ "$(T log -1 --format='%an <%ae>')" = "$(G log -1 --format='%an <%ae>' "$c")" ] || { echo "export: author changed: $(basename "$p")" >&2; exit 1; }
  want=$(G log -1 --format=%B "$c" | grep -vE '^\(cherry picked from commit [0-9a-f]{40}\)$' | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')
  got=$(T log -1 --format=%B | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')
  [ "$want" = "$got" ] || { echo "export: message changed beyond the cherry-pick lines: $(basename "$p")" >&2; exit 1; }
done
rebuilt=$(T rev-parse 'HEAD^{tree}')
if [ ${#lab[@]} -gt 0 ]; then
  for c in "${lab[@]}"; do T cherry-pick --allow-empty "$c" >/dev/null || { echo "export: lab commit $c does not re-apply" >&2; exit 1; }; done
fi
[ "$(T rev-parse 'HEAD^{tree}')" = "$(G rev-parse "$HEAD_SHA^{tree}")" ] \
  || { echo "export: the rebuilt tree differs from $HEAD_REF's" >&2; exit 1; }

find "$OUT_DIR" -maxdepth 1 -name '*.patch' -delete
cp "$work"/out/*.patch "$OUT_DIR/"
echo "# export of $HEAD_REF ($HEAD_SHA) on $BASE ($BASE_COMMIT): ${#keep[@]} patches, ${#lab[@]} lab-only commit(s) dropped"
printf 'no\tsource\tkind\tauthor\tsubject\n'
cat "$work/table"
for c in "${lab[@]}"; do printf -- '-\t%s\tdropped (lab only)\t-\t%s\n' "$(G rev-parse --short "$c")" "$(G log -1 --format=%s "$c")"; done
echo "# every patch passed git apply --check in order on $BASE; authors and messages kept; rebuilt tree $rebuilt = $HEAD_REF$([ ${#lab[@]} -gt 0 ] && echo ' minus the dropped lab commits')"
