# Build and run the native lane

This is a source-only distribution of the native TensorFold AFD engine. Read README's credits, qualification and limitations, hardware and Security sections first. Nothing in this guide changes a live service automatically.

## Configuration and prerequisites

Copy `.env.example` to `.env`. It centralizes engine/checkpoint/source pins, build images, toolchain paths, storage directories, hosts, devices, GID, API and fabric ports, and the candidate's serving flags. The loader reads example defaults, then your trusted `.env`, then restores caller exports. Review `print` output before any action.

Install the exact Zig toolchain (`ZIG_VERSION` and `ZIG_SHA256`), Docker, git and the host's rdma-core development headers. Obtain the pinned `CUDA_IMAGE` and `AOT_IMAGE` through your approved image workflow and read their NVIDIA terms. Builds use `--pull=never`: they will not download a missing image. The CUDA compiler image is CPU-only; no `--gpus`, privileged mode or network access is passed. Run heavy builds under a user-systemd scope with `MemoryMax=12G` and ensure at least 8 GiB of host MemAvailable. Keep outputs outside the source tree or in its ignored `.build` directory.

The native engine source is vendored in `engine/` from `ENGINE_REVISION`. Public identifiers and neutral test addresses differ from the original private source labels; runtime math, flags and kernels do not. It is not a claim of binary-SHA equality to the recorded deployment. An installed build needs its own hash manifests and hardware validation.

## Native executables and CUDA images

```sh
bash build.sh native x86_64
bash build.sh native aarch64
bash build.sh test
```

The serving build preserves `-Doptimize=safe`, GNU 2.28 target triples and the architecture's 120/121 CUDA targets. Attention includes the trusted grammar-loader root. Native builds refuse absent verbs headers rather than quietly publishing a RoCE stub. `test` is CPU-only and uses a writable disposable source copy.

Outputs are under `$BUILD_ROOT/out/native-<arch>/`: `bin/tf-glm-afd`, the installed CUDA images and their relative directories. Keep the complete output tree together. No generated ELF, cubin, fatbin, cache or library is committed to the public tree.

## Grammar library

Fetch XGrammar at `XGRAMMAR_REVISION` and its DLPack submodule at `DLPACK_REVISION`, not moving main. The build checks both Git heads. Do not pull cpptrace or googletest: they are excluded from the exact source list, as are Python TVM-FFI sources.

```sh
bash scripts/fetch-sources.sh
bash build.sh grammar
```

This compiles the TensorFold v1.0.6 C ABI shim over XGrammar v0.2.8, with the original C++17/exception/include/source flags. The library belongs only beside attention's native executable. Copy the complete generated library with the third-party licence texts. Hash your output and set `GRAMMAR_LIB_SHA256` after reviewing its trusted owner/mode/ELF/symbol checks. The example digest is the measured library's digest, not a guarantee that a path-dependent rebuild has identical bytes.

## Weights and world-1 q4 preparation

`download.sh print` shows the exact checkpoint operation without networking. `download.sh` fetches only Mia's pinned `MODEL_REVISION`, never a moving revision or external DFlash2 drafter. Run the download once on shared storage; all three roles can read it. The weights are large and require your own storage/download approval. This packaging run downloaded no weights.

The attention role additionally needs world-1 dense q4 triplets, including MTP. The converter executes Mia's actual q4 functions from its unchanged, SHA-checked recipe patch; no alternative quantization algorithm is substituted. In the pinned AOT image, mount the Python engine source at `/tensorfold`, this repo at `/source`, the checkpoint read-only at `/model` and an empty output at `/q4`, with `PYTHONPATH=/tensorfold/src`. Run the saved converter:

```sh
python3 -B /source/tools/afd-prep/glm_q4_dense.py make /model /q4 --rank 0 --world 1
```

Do not pass `--no-mtp`, and do not use a TP2 dense artifact for world-1 attention. Set `Q4` to the output directory, retain its manifest and keep it read-only at serve time. FP8 head tensors are read through the pinned EXL3 checkpoint's native loader. Real-weight conversion was not rerun by release packaging.

## Triton AOT sets

`fetch-sources.sh` pins the separate Python TensorFold source used by the AOT compiler. It also fetches grammar source; no weights or images. `prepare.sh sets` compiles the source specs for sm_120a and sm_121a and adds the FP8 head's launch rows only to set120. It refuses failed compiles, duplicate launch keys and unsupported scratch, and produces fresh SHA256SUMS. Whole-cubin captured SHA checks include path-sensitive debug tables, so they are removed only from a temporary spec copy; the checked-in specs and native loader remain unchanged.

```sh
bash prepare.sh sets
```

Set120 has 487 launch rows, set121 478 for this non-FP8-MTP profile. Place set120 on attention and set121 on both expert hosts. Source packaging ships neither set. Rebuilt sets have their own manifest/digests and need hardware qualification before claiming the archived speed or numerical results. A build does not authorize redistribution of libdevice-derived object code.

## Role launch and operational safety

The launcher is local and does not automate service installation, remote copying or recovery. On each host, set that role's `ATTENTION_BIN`/`EXPERT_BIN`, `MODEL`, `Q4`, `ATTENTION_TRITON`/`EXPERT_TRITON`, device, GID and master fabric address. Print and run the native flag check first. Match the completed soak's shared-forward selection, not the historical faster arm.

Before every load: exclusive GPU ownership, no protected deployment lease, an active host-memory-floor guard, measured available memory, read-only pinned weights/sets and a healthy isolated fabric. GB10 unified allocations bypass Docker accounting. Use your existing guarded service manager with locked-memory permissions; experts first, then attention. A failed expert must fail the request, not leave attention waiting indefinitely. If any start fails, stop the exact owned ranks gracefully once and investigate; do not add automatic retries or SIGKILL.

Default API exposure is loopback. For a non-loopback bind, set a service-owned 0600 `API_KEY_FILE`, firewall the API, and supply bearer headers securely from a private client. Never log key contents. Rank fabric traffic remains unauthenticated regardless of the HTTP key.

## Stop/restart

The public role launcher keeps a foreground process: SIGTERM that exact process or stop its exact supervisor unit. Verify the attention and both expert exits and host memory recovery before a new start. There is no broad process-name kill, fleet service map, auto-reboot or unqualified recovery watcher in this source-only distribution. The completed release's service-test results must establish two actual cycles on its exact candidate before publication.
