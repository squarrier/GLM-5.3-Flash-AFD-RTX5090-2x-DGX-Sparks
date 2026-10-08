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
- **Links not up.** The listen daemons must be up before the connect daemons start (`./start.sh up` does
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

- **`afd: ...` in the attention log, and requests failing.** An expert died, its heartbeat went stale, or a link
  failed.
  The line names the cause. Restart the whole stack: `./start.sh recover` (or `./stop.sh && ./start.sh up`);
  [extras/watch](../extras/watch/README.md) does it by itself after two failed checks.
- **An expert node's process dies** (the case [TensorFold#214](https://github.com/ashhart/TensorFold/issues/214)'s
  second answer names: a failed expert process must fail the request, not hang the attention side). As the soak's
  drill measured it (the expert's process stopped with `SIGTERM`): the streamed request in flight got an error event
  ("the AFD expert nodes are gone ...") and `[DONE]` 23.3 s after the fault, when the attention node's 20 s
  heartbeat watchdog fired and its log printed `afd: expert rank ... silent for 20s`. Requests sent in between
  waited for that moment, then got HTTP 429 with `Retry-After: 5` (`TF_GLM_CAPACITY_STATUS=1`; 503 without it);
  every later request got HTTP 500 ("the two ranks are out of step after an error; restart both") in about 5 ms.
  `/health` kept answering 200, so don't route traffic on it alone. The watcher's `recover` had the stack serving
  again 309 s after the fault.
- **MCDMA link loss.** A stopped or replaced daemon, or a hung expert, fails the forward; the watchdog calls a link
  dead after 30 s of an unanswered request. Restart the whole stack, as above.
- **Four long requests at once pause and replay.** The streams share one cache pool (`CACHE_GIB`); when it is full
  the youngest stream goes back to the queue and replays later, exactly but slower.
- **Wire audit.** `AFD_CHECK=1 CONTEXT=98304 ./start.sh up` hashes both sides' wire tensors for
  `AFD_CHECK_FORWARDS` forwards and prints `afd-check: window complete, N forwards, M mismatches total`.
- **Replies differ slightly from an earlier build's.** `TF_GLM_KDA_CHUNKED=1` (on in `.env.example`) runs prompts
  through Mia's chunked KDA kernel, whose arithmetic is close to the serial kernel's but not the same bits;
  `TF_GLM_EXPERT_KERNEL=g53` (on in `EXPERT_ENV`) gives the experts' prompt rows Hugh Madden's kernels' bits; and
  from 2.15 `TF_GLM_PARTIALS_BF16=1` (on in both lines) rounds each Spark's sum for a prompt window to BF16. Replies
  stay deterministic; remove a pair to get the earlier replies back bit for bit (the BF16 switch from both lines:
  2.1's replies).
- **An expert node refuses to start: `TF_GLM_PARTIALS_BF16` differs.** The attention node and both expert nodes must
  agree on it; `./start.sh` refuses `ATTN_ENV` and `EXPERT_ENV` lines that disagree before anything starts.
- **Turning one feature off.** Every `TF_GLM_*` pair in `ATTN_ENV` and `EXPERT_ENV` is its own switch: remove the
  pair, then `./stop.sh && ./start.sh up`. Mia's draft policy runs at `fnc5:0.2` in `.env.example`; put
  `TF_GLM_DFLASH_POLICY=fnc7:0.3` in its place to run her lane's policy.
- **Out of memory on the attention node after raising `PARALLEL` or `CACHE_GIB`.** Past four streams the extra
  streams' working memory comes out of the 5090's cache budget: at eight streams `CACHE_GIB=8.5` with the kept states
  in host RAM (`TF_GLM_KEPT_HOST=1`) is the measured setting, and 6.5 without them (7.5 GiB, and 7 GiB with the 64-row
  window, went over the memory gate; 2.15's 32-row window leaves the pool 6,144 tokens more). 2.0 measured four
  streams at `PARALLEL=4`, `CACHE_GIB=8` without
  `TF_GLM_MULTI_WINDOW` and without the host tier.
- **The attention host runs out of RAM, or the start fails pinning host memory.** The host RAM tier pins 33.35 GiB as
  shipped (`TF_GLM_HOST_CACHE_GIB=24` plus the kept states). Lower `TF_GLM_HOST_CACHE_GIB`, or remove it and
  `TF_GLM_KEPT_HOST` together with `CACHE_GIB=6.5`, on a smaller host.
- **`./start.sh up` stops at "the N links not up".** With `MCDMA_INFLIGHT=4` each Spark runs four listen daemons, on
  `MCDMA_CTRL_PORT` and the three ports after it (`x0`, `x0-1`, `x0-2`, `x0-3`; `x1`, ...); let all four through the
  fabric firewall. The attention host runs two connect daemons, `connect` (links 0-2) and `connect2` (link 3 of both
  Sparks), each with its own log (`connect.log`, `connect2.log` under `AFD_HOME/mcdma`). `./start.sh status` lists
  every link, and `./start.sh logs` tails each daemon's log. With fewer ports open, set `MCDMA_INFLIGHT=2` (2.1's
  setting) and `TF_GLM_PREFILL_LANES=2` or remove it.
- **Long first tokens at many deep contexts.** Deep contexts that together outgrow the 5090's KV pool (1,585,152 tokens
  at eight streams) evict kept prompt states, and prompts fill again unless the evicted prompt parked its rows in host
  RAM (`TF_GLM_HOST_CACHE_GIB`). Fewer concurrent deep contexts, or shorter ones, avoid it.
- **A request whose client disconnected keeps a place in the queue.** It waits for a lane before it is dropped (100-127
  s in the lab). Her `TF_GLM_QUEUED_CANCEL=1` (patch 0037, off as shipped) drops it at once, at a cost in the
  ten-client arena cell's prompt rate ([measured and off](DESIGN.md#measured-for-215-and-left-off)).
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

- `./stop.sh` stops the containers with `SIGTERM` (the attention node first) and gives each up to `STOP_WAIT` (60 s)
  to exit, then stops the connect daemons, then the listen daemons: `SHUTDOWN` on the daemon's socket, `SIGTERM` after
  10 s, and never `SIGKILL`, which would skip the queue-pair teardown. A daemon still running after 25 s is reported and
  left alone.
- **A container still running after its `SIGTERM`.** `./stop.sh` (and `./start.sh recover`) name it, leave it running
  and leave the MCDMA daemons up under it; nothing escalates to `SIGKILL`, because a GPU process killed hard on a GB10
  can strand its memory or hang the box. Look at its log (`./start.sh logs`) and `nvidia-smi` on that host first; if
  it is stuck in the GPU, a reboot of that host is the clean way out.
- The containers run under `docker --init`, so a `SIGTERM` reaches the TensorFold process (the expert node installs no
  handler for it, and as a container's PID 1 it would never see one).
- `./stop.sh tf` stops the containers only and leaves the links up; `./start.sh recover` starts the containers on them
  again.
- A stop by hand leaves a stop marker (`~/.local/state/glm-afd/stopped`, `CTL_STATE`) so that
  [extras/watch](../extras/watch/README.md) leaves the stack down; `./start.sh up` clears it.
- Mailboxes left in `/dev/shm` (`mcdma-rpc.*`) after a crash are listed by `./start.sh check`; remove them only when
  no daemon runs.

## The watcher (extras/watch)

- **It latched.** Three recoveries in six hours (`WATCH_MAX_RECOVERIES`, `WATCH_WINDOW_H`), or one recover that needed
  you (rc 3: a container still ran after `SIGTERM`, or a Spark's memory did not come back). It keeps recording but
  does not act. Find the cause (`./start.sh status`, `./start.sh logs`, `glm53f-afd-report`), fix it, then
  `rm ~/.local/state/glm-afd/watch/latched`.
- **It did nothing while the stack was down.** It holds off while the stop marker is there (a `./stop.sh` by hand),
  in maintenance mode, while another `start.sh up`/`recover`/`probe` or `stop.sh` runs, and for 10 minutes after the
  controller boots. Its records say which (`"hold"` in the telemetry).
- **A stranded Spark** (recover rc 3, "memory did not come back"): the expert node exited but `MemAvailable` stayed
  below `MIN_FREE_GIB_EXPERT` for `MEM_WAIT` (180 s). Nothing is started on it. Rebooting that Spark is the fix; the
  watcher never reboots anything.
