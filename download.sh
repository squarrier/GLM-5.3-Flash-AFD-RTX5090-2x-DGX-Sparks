#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"
# Run once on shared storage; no duplicate full checkpoint copies are needed.
if [ "${1:-}" = print ]; then printf 'hf download %q --revision %q --local-dir %q\n' "$MODEL_REPO" "$MODEL_REVISION" "$MODEL"; exit 0; fi
exec hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL"
