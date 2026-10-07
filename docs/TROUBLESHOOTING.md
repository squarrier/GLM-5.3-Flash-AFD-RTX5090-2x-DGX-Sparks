# Troubleshooting

## DGX Spark (GB10) memory safety: read this first

A GB10's GPU and CPU share one pool of memory. Two things follow, and both can hang a Spark so hard that only a
power cycle brings it back (ping still answers; SSH does not):

- **CUDA allocations are not charged to a container's memory cgroup.** A container memory limit does not protect
  the host. Keep **one GPU tenant per Spark**: nothing else may use the GPU while an expert node runs.
  `./start.sh check` refuses to start when another process holds a GPU, and every container here starts with
  `--oom-score-adj=1000`, so the kernel's OOM killer takes it before sshd.
- **Memory comes back a few seconds after a GPU process exits.** Starting the next one too early stacks both.
  `./start.sh up` waits until each Spark shows `MIN_FREE_GIB_EXPERT` (100 GiB) available, for up to 180 s.

Also:

- **Never compile CUDA extensions with the weights resident.** Building them inside `serve`, after about 90 GiB of
  weights were loaded, drove a 128 GB host down to 5.7 GiB available. `./build.sh ext` prebuilds them with no model
  loaded, into a cache per code tree (`AFD_HOME/ext-<tree id>`), and `./start.sh up` runs it before every start (a
  ~30 s no-op when current). Sharing one cache between two trees makes every switch recompile inside `serve`.
- **A process killed mid-compile leaves torch's build lock behind**, and the next process to load that extension
  waits on it forever. The prebuild removes stale `lock` / `.ninja_lock` files first; do the same by hand if you
  build elsewhere.
- **Consider a host-side guard** on each Spark: a small daemon that watches available memory and stops GPU
  containers before the host runs out, plus the kernel's watchdog and panic-on-lockup so a hung box reboots itself.
  [`extras/gb10-hostguard`](../extras/gb10-hostguard/README.md) is one (from v1.0); `./start.sh` does not install it.

## Start-up

- **`./start.sh check` fails.** It names what is missing: the image (`./build.sh images`), the tree
  (`./build.sh sync`), the MCDMA build (`./build.sh mcdma`), the checkpoint or drafter (`./download.sh`), an RDMA
  port that is not `PORT_ACTIVE`, or leftovers from an earlier run (`./stop.sh`).
- **Both links not up.** The listen daemons must be up before the connect daemon starts (`./start.sh up` does
  that). Check the fabric addresses, the RDMA device names (`ibv_devices`), the GID index (`show_gids`; RoCE v2 is
  usually 3) and the MTU, then `./start.sh logs` for the daemons' lines.
- **It hangs at the handshake.** The attention node must start first; the experts join its rendezvous on
  `MASTER_PORT` at `ATTN_FABRIC_IP`. The attention node then waits for both experts' loaded marks.
- **An expert exits at startup naming two values.** Its settings disagree with the attention node's: a different
  checkpoint, wire version or link map. Rebuild and resync the same tree on all three hosts (`./build.sh sync`).
- **Healthy, but no `afd transport mcdma` line.** Not the MCDMA wire: `./start.sh up` stops the stack. Check the
  attention log's first lines.
- **CPU-driven when GPU-driven was expected.** Run `./start.sh probe`: it names the missing device attribute or the
  failing stream memory operation on each host.
- **Out of memory at 262,144 on the 5090.** Keep `MEMORY_RESERVE_GIB=2`, or lower `CONTEXT`.

## Serving

- **`afd: ...` in the attention log, and HTTP 503s.** An expert died, its heartbeat went stale, or a link failed.
  The line names the cause. Restart the whole stack: `./stop.sh && ./start.sh up`.
- **MCDMA link loss.** A stopped or replaced daemon, or a hung expert, fails the forward; the watchdog calls a link
  dead after 30 s of an unanswered request. Restart the whole stack, as above.
- **Four long requests at once pause and replay.** The streams share one cache pool (`CACHE_GIB`); when it is full
  the youngest stream goes back to the queue and replays later, exactly but slower.
- **Wire audit.** `AFD_CHECK=1 CONTEXT=98304 ./start.sh up` hashes both sides' wire tensors for
  `AFD_CHECK_FORWARDS` forwards and prints `afd-check: window complete, N forwards, M mismatches total`.
- **Replies differ slightly from an earlier build's.** `TF_GLM_KDA_CHUNKED=1` (on in `.env.example`) runs prompts
  through Mia's chunked KDA kernel, whose arithmetic is close to the serial kernel's but not the same bits, and
  `TF_GLM_EXL3_PROMPT=1` (on in `EXPERT_ENV`) gives the experts' prompt rows her prompt kernel's bits. Replies stay
  deterministic; remove a pair to get the earlier kernel's replies back bit for bit.
- **Turning one feature off.** Every `TF_GLM_*` pair in `ATTN_ENV` and `EXPERT_ENV` is its own switch: remove the
  pair, then `./stop.sh && ./start.sh up`. Mia's draft policy runs at `fnc5:0.2` in `.env.example`; put
  `TF_GLM_DFLASH_POLICY=fnc7:0.3` in its place to run her lane's policy.
- **Out of memory on the attention node after raising `PARALLEL` or `CACHE_GIB`.** Past four streams the extra
  streams' working memory comes out of the 5090's cache budget: at eight streams `CACHE_GIB=8.5` with the kept states
  in host RAM (`TF_GLM_KEPT_HOST=1`) is the measured setting, and 6.5 without them (7.5 GiB, and 7 GiB with the 64-row
  window, went over the memory gate). 2.0 measured four streams at `PARALLEL=4`, `CACHE_GIB=8` without
  `TF_GLM_MULTI_WINDOW` and without the host tier.
- **The attention host runs out of RAM, or the start fails pinning host memory.** The host RAM tier pins 33.35 GiB as
  shipped (`TF_GLM_HOST_CACHE_GIB=24` plus the kept states). Lower `TF_GLM_HOST_CACHE_GIB`, or remove it and
  `TF_GLM_KEPT_HOST` together with `CACHE_GIB=6.5`, on a smaller host.
- **`./start.sh up` stops at "the N links not up".** With `MCDMA_INFLIGHT=2` each Spark runs two listen daemons, on
  `MCDMA_CTRL_PORT` and the port after it (`x0`, `x0-1`; `x1`, `x1-1`); let both ports through the fabric firewall.
  `./start.sh status` lists every link, and `./start.sh logs` tails each daemon's log.
- **Long first tokens at many deep contexts.** Deep contexts that together outgrow the 5090's KV pool (1,579,008 tokens
  at eight streams) evict kept prompt states, and prompts fill again unless the evicted prompt parked its rows in host
  RAM (`TF_GLM_HOST_CACHE_GIB`). Fewer concurrent deep contexts, or shorter ones, avoid it.
- **A request whose client disconnected keeps a place in the queue.** It waits for a lane before it is dropped (100-127
  s in the lab). Her `TF_GLM_QUEUED_CANCEL=1` (patch 0037, off as shipped) drops it at once
  ([measured and off](DESIGN.md#measured-and-off)).
- **Tool-call markup as text with `tool_choice: "none"`.** The server offers the model no tools, but the model can
  still write call markup into its text. That is TensorFold 0.6.5's behaviour ([Limits](DESIGN.md#limits)).

## Building

- **MCDMA's rpc build stops on `-Werror=format-truncation`** (`rpcd_connect.c`) with GCC 11-13. It is a false
  positive; `./build.sh mcdma` keeps that one warning a warning. The open
  [ashhart/MCDMA#5](https://github.com/ashhart/MCDMA/pull/5) by @ThinkOffApp fixes the build and documents the setup.
- **A patch does not apply.** `patches/` is for TensorFold `TF_TAG` at exactly `TF_COMMIT`; `./build.sh tree` checks
  both before applying anything.
- **A switch's CUDA extension compiles inside `serve`.** `scripts/prebuild_ext.py` builds every extension the
  switches can load, including `TF_GLM_HC_DEC`'s and `TF_GLM_KDA_CHUNKED`'s. If you add a patch with a new extension,
  add it there too, or it compiles with the weights resident (see the memory-safety section above).

## Stopping

- `./stop.sh` removes the containers, then stops the connect daemon, then the listen daemons: `SHUTDOWN` on the
  daemon's socket, `SIGTERM` after 10 s, and never `SIGKILL`, which would skip the queue-pair teardown. A daemon still
  running after 25 s is reported and left alone.
- `./stop.sh tf` removes the containers only and leaves the links up for the next start.
- Mailboxes left in `/dev/shm` (`mcdma-rpc.*`) after a crash are listed by `./start.sh check`; remove them only when
  no daemon runs.
