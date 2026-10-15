#!/usr/bin/env bash
# Native-role argv preserved from the measured launcher. Apache-2.0.
# Ash Hart authored TensorFold; Mia authored the GLM recipe; GPT-6.1 Sol via Hermes packaged it.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
role=${1:?usage: start.sh attention|expert0|expert1 print|check|run}
mode=${2:-print}
case "$mode" in print|check|run) ;; *) printf 'Invalid mode\n' >&2; exit 2;; esac
base=(env -i "PATH=$PATH" "HOME=${HOME:-$RUNTIME_ROOT}" "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}")
for key in TF_AFD_MTP TF_AFD_PAIRS TF_AFD_LANES TF_GLM_HALVES_KERNEL TF_AFD_SETUP_MS TF_GLM_EXL3_SEQ; do base+=("$key=${!key}"); done
case "$role" in
 attention)
  BIN=$ATTENTION_BIN; TRITON=$ATTENTION_TRITON; DEV=$ATTENTION_DEV
  base+=("TF_GLM_DENSE=q4:$Q4" "TF_GLM_F8=$TF_GLM_F8" "TF_GLM_DRAFT_AUTO=$TF_GLM_DRAFT_AUTO" "TF_AFD_GRAMMAR_LIB=$GRAMMAR_LIB" "TF_AFD_GRAMMAR_LIB_SHA256=$GRAMMAR_LIB_SHA256")
  for key in TF_AFD_USERS TF_AFD_POOL_TOKENS TF_AFD_USERS_INTERLEAVE TF_AFD_GRAMMAR_MASK TF_AFD_GRAMMAR_SERVE TF_GLM_SAMPLING TF_AFD_USERS_SAMPLING TF_AFD_USERS_DRAFTS TF_AFD_USERS_SHARED TF_AFD_USERS_SHARED_DRAFTS TF_AFD_USERS_BATCHED_DRAFTS TF_AFD_USERS_PING_PONG; do base+=("$key=${!key}"); done
  args=(attn serve --model "$MODEL" --gid "$GID" --master "$FABRIC_ADDRESS:$FABRIC_PORT" --dev "$DEV" --triton "$TRITON" -- --host "$API_HOST" --port "$API_PORT" --name "$SERVED" --context "$CONTEXT" --temperature 0 --drafts "$DRAFTS")
  if [ -n "$API_KEY_FILE" ]; then args+=(--api-key-file "$API_KEY_FILE"); fi ;;
 expert0|expert1)
  BIN=$EXPERT_BIN; TRITON=$EXPERT_TRITON; DEV=$EXPERT_DEV
  half=0; [ "$role" != expert1 ] || half=1
  for key in TF_GLM_EXL3_L2D TF_GLM_EXL3_PDL TF_GLM_EXL3_X1 TF_GLM_EXL3_PGROUP TF_GLM_EXL3_MMA; do base+=("$key=${!key}"); done
  args=(expert --half "$half" --model "$MODEL" --triton "$TRITON" --master "$FABRIC_ADDRESS:$FABRIC_PORT" --dev "$DEV" --gid "$GID") ;;
 *) printf 'Invalid role\n' >&2; exit 2;;
esac
if [ "$mode" = print ]; then printf '%q ' "${base[@]}" "$BIN" "${args[@]}"; printf '\n'; exit 0; fi
if [ "$mode" = check ]; then exec "${base[@]}" "$BIN" check-flags "${args[@]}"; fi
case "$FABRIC_ADDRESS" in 192.0.2.*|198.51.100.*|203.0.113.*|*.invalid) printf 'Replace the documentation fabric address before run\n' >&2; exit 2;; esac
if [ "$role" = attention ] && [ "$API_HOST" != 127.0.0.1 ] && [ -z "$API_KEY_FILE" ]; then
 printf 'Non-loopback API requires API_KEY_FILE in this launcher; also firewall it\n' >&2; exit 2
fi
if [ "$role" = attention ] && [ -n "$API_KEY_FILE" ]; then
 if [ ! -f "$API_KEY_FILE" ] || [ ! -O "$API_KEY_FILE" ] || [ "$(stat -c %a -- "$API_KEY_FILE")" != 600 ]; then
  printf 'API_KEY_FILE must be a service-owned regular file with mode 0600\n' >&2; exit 2
 fi
fi
# Host guards, exclusive GPU ownership and memory admission are operator prerequisites.
exec "${base[@]}" "$BIN" "${args[@]}"
