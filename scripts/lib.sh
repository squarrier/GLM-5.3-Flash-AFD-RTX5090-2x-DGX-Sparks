#!/usr/bin/env bash
# Apache-2.0. GPT-6.1 Sol via Hermes; trusted local configuration.
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
load_env() {
  local entry key value
  local -A caller=()
  while IFS= read -r -d '' entry; do
    key=${entry%%=*}; value=${entry#*=}
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && caller[$key]=$value
  done < <(env -0)
  set -a
  . "$REPO_ROOT/.env.example"
  [ ! -f "$REPO_ROOT/.env" ] || . "$REPO_ROOT/.env"
  set +a
  for key in "${!caller[@]}"; do export "$key=${caller[$key]}"; done
  export REPO_ROOT
}
load_env
