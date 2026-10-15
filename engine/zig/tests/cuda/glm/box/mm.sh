#!/bin/bash
# GLM-5.3-Flash's affine kernel copies on one GPU: python-0.6's qmm and experts wrappers record their calls and buffers (mm_oracle.py), the Zig build's glm_qmm_group, glm_qmm_prefill, glm_experts and glm_experts_prefill fatbins replay them byte for byte. Usage: TF_GLM=<work dir holding src/ (this tree), py06/ (python-0.6, git archive), toolchain/zig> bash -u mm.sh RUN, the GPU lock held (kda.sh's layout).
set -u
RUN="${1:?run id}"
TF="${TF_GLM:?set TF_GLM}"
OUT="${TF:?}/runs/${RUN:?}"
SRC="${TF:?}/src"
PY="${TF:?}/py06"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
JOURNAL="${TF_ZIG_JOURNAL:-/dev/null}"
WHO="${TF_ZIG_WHO:-zig-glm}"
TREE=$(cat "${SRC:?}/TREE" 2>/dev/null || echo unknown)
mkdir -p "${OUT:?}" "${TF:?}/cache/zig-local" "${TF:?}/cache/zig-global"
rc=0

journal() { printf '{"utc": "%s", "event": "%s", "who": "%s", "run": "glm-mm-%s", "tree": "%s"%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$WHO" "$RUN" "$TREE" "${2:-}" >> "$JOURNAL"; }
step() { echo "=== $(date -u +%H:%M:%S) $*"; }
container() {  # name, then docker run arguments; the container is ours by name and label, removed on exit
  local name="tf-zig-glm-mm-${RUN}-$1"; shift
  timeout 1800 docker run --rm --name "$name" --label "tensorfold.zig=glm-mm-${RUN}" "$@"
}
cleanup() { for c in $(docker ps -q --filter "label=tensorfold.zig=glm-mm-${RUN}"); do docker stop --time 10 "$c" >/dev/null; done; }
trap 'cleanup; journal END ", \"rc\": 143"; exit 143' TERM INT

# one GPU tenant: a resident model or another container on a GB10 shares its unified memory
apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader || true)
if [ -n "$(docker ps -q)" ] || [ -n "$apps" ]; then
  echo "PREFLIGHT-FAIL: containers or compute apps present"; docker ps; echo "$apps"; exit 3
fi
journal START ", \"cmd\": \"zig/tests/cuda/glm/box/mm.sh\""
step "kernel copies match their python-0.6 sources"
python3 -B "${SRC:?}/zig/tests/cuda/copies.py" --tree "${PY:?}" glm_qmm_group.cu glm_qmm_prefill.cu glm_experts.cu \
  glm_experts_prefill.cu || rc=1
grep -E 'MemAvailable' /proc/meminfo

step "oracle: python-0.6's qmm and experts wrappers in ${IMAGE}"
rm -rf "${OUT:?}/torch_ext" "${OUT:?}/fixtures"
container oracle --gpus all --ipc host --network none --memory 32g --memory-swap 32g --read-only \
  --tmpfs /tmp:size=4g -e HOME=/tmp -v "${PY:?}:/py06:ro" -v "${SRC:?}:/tensorfold:ro" -v "${OUT:?}:/out" \
  -e PYTHONPATH=/py06/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e TORCH_EXTENSIONS_DIR=/out/torch_ext -e CUDA_CACHE_PATH=/out/cuda_cache -e TRITON_CACHE_DIR=/out/triton \
  -w /tensorfold "$IMAGE" python -B /tensorfold/zig/tests/cuda/glm/mm_oracle.py /out/fixtures \
  > "${OUT:?}/oracle.log" 2>&1 || { rc=1; echo "oracle failed"; tail -30 "${OUT:?}/oracle.log"; }

step "zig build with the image's nvcc (the GPU's own SASS target)"
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
container build --network none --memory 16g -v "${TF:?}:${TF:?}" -w "${SRC:?}" \
  -e ZIG_LOCAL_CACHE_DIR="${TF:?}/cache/zig-local" -e ZIG_GLOBAL_CACHE_DIR="${TF:?}/cache/zig-global" \
  --entrypoint "${TF:?}/toolchain/zig" "$IMAGE" build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm="${SM:?}" -Doptimize=safe \
  --prefix "${OUT:?}/zig-out" -j8 glm-mm-fatbins glm-mm > "${OUT:?}/build.log" 2>&1 || { rc=1; echo "build failed"; tail -40 "${OUT:?}/build.log"; }

step "SASS: the extensions' objects against our fatbins"
container sass --network none -v "${OUT:?}:/out" -v "${SRC:?}:/tensorfold:ro" --entrypoint bash "$IMAGE" -c '
  rc=0
  for pair in tensorfold_qmm_v5/qmm_group:glm_qmm_group tensorfold_qmm_v5/qmm_prefill:glm_qmm_prefill \
              tensorfold_experts_v7/experts:glm_experts tensorfold_experts_v7/experts_prefill:glm_experts_prefill; do
    py=${pair%%:*}; zig=${pair##*:}
    cuobjdump -sass /out/torch_ext/$py.cuda.o > /out/sass-python-$zig.txt
    cuobjdump -sass /out/zig-out/fatbin/$zig.fatbin > /out/sass-zig-$zig.txt
    echo "--- $zig"; python -B /tensorfold/zig/tests/cuda/nemotron/sass_compare.py /out/sass-python-$zig.txt /out/sass-zig-$zig.txt || rc=1
  done
  exit $rc' > "${OUT:?}/sass.log" 2>&1 || rc=1
cat "${OUT:?}/sass.log"

T="${OUT:?}/zig-out/bin/tf-glm-mm"
F="${OUT:?}/fixtures"
step "GPU tests on the host (no Python, no PyTorch)"
for t in "symbols" "replay $F/mm_kda_proj $F/mm_kda_fb $F/mm_kda_o $F/mm_mlp_down" \
         "replay $F/experts_decode $F/experts_prompt"; do
  echo "--- $t"
  # shellcheck disable=SC2086
  timeout 600 "$T" $t || { rc=1; echo "rc=$? for $t"; }
done 2>&1 | tee "${OUT:?}/tests.log"
grep -q '^FAIL\|^rc=' "${OUT:?}/tests.log" && rc=1

cleanup
step "done rc=$rc"
journal END ", \"rc\": $rc"
exit $rc
