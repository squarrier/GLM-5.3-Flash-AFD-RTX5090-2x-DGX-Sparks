#!/usr/bin/env bash
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
exec python3 -B "$R/tests/static_checks.py"
