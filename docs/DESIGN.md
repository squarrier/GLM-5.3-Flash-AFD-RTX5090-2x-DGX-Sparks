# Design

This recipe serves GLM-5.3-Flash with [TensorFold](https://github.com/ashhart/TensorFold) split across three
machines: one RTX 5090 runs everything except the routed experts, and two DGX Sparks hold the routed experts.
Every MoE layer's exchange between them goes over [MCDMA](https://github.com/ashhart/MCDMA)'s RDMA link daemons.
TensorFold is pinned at v0.6.5; `patches/` adds the split and MiaAI-Lab's GLM work on top of it.

## The three nodes

| Node | Machine | Command (inside the container) | Holds |
| --- | --- | --- | --- |
| Attention (rank 0) | x86_64 + RTX 5090 32 GB | `tensorfold serve CKPT --experts remote --master <attention-ip> --drafter DRAFTER ...` | the non-expert weights, the KV cache, the DFlash2 drafter, sampling, the HTTP API |
| Expert half 0 (rank 1) | DGX Spark (GB10) | `tensorfold experts CKPT --rank 0 --master <attention-ip>` | one half of every MoE layer's routed experts |
| Expert half 1 (rank 2) | DGX Spark (GB10) | `tensorfold experts CKPT --rank 1 --master <attention-ip>` | the other half |

`CKPT` is the same EXL3 checkpoint on all three machines. Only the attention node takes `--drafter`; the expert
nodes take the window and chunking from it. The attention node starts first: it hosts the rendezvous and publishes
the settings the experts must match, and an expert whose checkpoint, wire or link map disagrees exits with a message
naming both values. `./start.sh up` follows that order.

## One MoE layer

1. The attention node runs the layer's attention and router.
2. It sends the layer's normed rows, the router's picks and the routing weights to both expert nodes, as one packed
   request into each node's mailbox.
3. While the experts work, it runs the layer's shared expert.
4. Each expert node runs its half of the routed experts and writes its fp32 partial sum back as one reply.
5. The attention node adds the three (its shared expert and the two partials) and goes on.

All of this is eager: the attention node captures no CUDA graphs across the wire, and the expert nodes refuse any
forward mode but eager and prompt chunks.

## The MCDMA wire

- **Daemons.** An `mcdma-rpcd` link daemon runs natively on each host. The attention host runs the connect end of
  every link; each Spark runs the listen end of its own links, one daemon a link: `MCDMA_INFLIGHT` links to each Spark
  (2 in `.env.example`, named `x0`, `x0-1` and `x1`, `x1-1`; link j listens on `MCDMA_CTRL_PORT` + j). `./start.sh up`
  starts the listen ends, then the connect end, and waits until the connect daemon reports every link up. `./stop.sh`
  stops them in the reverse order.
- **Mailboxes.** The daemons create the mailboxes under `/dev/shm`; the containers reach them through `--ipc=host`
  and load `libmcdma-rpc.so` from the MCDMA build (`TF_AFD_MCDMA_LIB`). A request half of 20 MiB and a reply half
  of 36 MiB fit a 2,048-row prompt chunk (a 16.9 MB request and a 32 MiB reply).
- **GPU-driven exchange.** Where the device supports CUDA stream memory operations, the serving stream itself writes
  and waits on the mailbox words (`cuStreamWriteValue64` / `cuStreamWaitValue64`): no host sync per layer. Otherwise
  the exchange is CPU-driven. `TF_AFD_MCDMA_MODE=auto` (the default) picks GPU-driven when the startup probe passes
  and says so loudly when it falls back; `./start.sh probe` runs the same checks without a model. The attention
  node's startup line names the result: `afd transport mcdma, gpu-driven` or `cpu-driven`.
- **Header.** Each request starts with 64 int32 words a MoE position: a magic number, the forward's sequence number,
  the row count, the MoE position, the mode, the MoE layer count, and for a layer slice the slice's length.
- **Layer slices.** For sliced prompt fills (patch 0016) the wire carries a slice of a prompt chunk's layers as its
  own mode, `MODE_SLICE`. The settings handshake carries a wire version (2), so a node with the older wire is refused
  at startup instead of misreading a slice.
- **Prompt chunk pairs.** With `TF_GLM_PREFILL_PAIRS=1` (patch 0022) the attention node opens two prompt chunks'
  forwards at once, `MODE_PAIR`, and each expert serves the two windows' layers in arrival order. The settings
  handshake carries `"pairs"` only with the switch on, so an expert without the mode refuses the settings at startup.
- **Exchanges in flight.** With `TF_GLM_MCDMA_INFLIGHT=N` (patch 0027; `MCDMA_INFLIGHT` in `.env` sets it and starts
  the daemons to match) each expert node has N links. The attention node puts the k-th request of a run on link
  k mod N and collects the replies in request order; the expert lands and computes the requests in that order on its
  one compute stream, so its kernels and bits are those of one link. With pairs on, the next chunk's exchange goes out
  as soon as its attention has run, while the other chunk's is still unanswered, so the Sparks find the next request
  waiting when they reply. The handshake carries `"inflight": N`, which an expert of an older tree refuses.
- **Failure.** A request left unanswered for `TF_AFD_MCDMA_TIMEOUT` seconds (30) or an expert whose heartbeat goes
  stale (5 s beats, 20 s) fails the in-flight requests with HTTP 503, and the attention node logs `afd: ...`. Restart
  the whole stack, not one node.
- **Audit.** `AFD_CHECK=1` hashes both sides' wire tensors every layer for `AFD_CHECK_FORWARDS` forwards and prints
  the mismatch count; `AFD_TIME=1` prints per-layer exchange times.

The transport sits behind one small interface (`ExpertTransport` on the attention node, `ExpertLink` on each expert
node), so another transport could implement it. The split loads MCDMA at runtime only; none of its code is copied.

## Mia's main lane

The configuration follows MiaAI-Lab's own GLM-5.3-Flash recipe for two DGX Sparks, so that this recipe and hers
serve the same weights the same way: her checkpoint, the DFlash2 drafter, 4-bit dense weights, the FP8 KV cache, her
decode and prompt kernels, and concurrent requests over one cache pool (four in her main lane, eight here). The
attention node takes most of these; the expert nodes take her expert kernels. The measured differences from her main
lane follow the table ([deviations](#deviations-from-her-main-lane)); the levers this update adds are in
[the switches](#the-switches).

| `.env` | Switch | What |
| --- | --- | --- |
| `PARALLEL=8` | `--parallel 8`, `TF_GLM_MULTI_WINDOW=64` | up to eight requests decode together (her recipe patch 0069; her main lane serves four): one cache pool, batched verify windows of up to 64 rows, DFlash2 drafts for every stream; each reply equals the one its request gets alone ([eight streams](#eight-streams)) |
| `DENSE=q4` | `TF_GLM_DENSE=q4` | the dense weights in MSE-searched 4-bit groups of 64, the head in FP8 (lossy against BF16, as in her recipe) |
| `KV=fp8` | `TF_GLM_KV=fp8` | the DSA latent and indexer caches as e4m3 rows with a power-of-two scale each |
| `CACHE_GIB=6.5`, `CACHE_ENTRIES=20` | `TF_GLM_CACHE_GIB`, `TF_GLM_CACHE_ENTRIES` | kept prompt states for the prompt cache, and the KV pool from the rest of the budget: 727,040 tokens at eight streams ([twenty kept prompts](#twenty-kept-prompts)) |
| `ATTN_ENV` | `TF_GLM_SHARED_PREFIX=1` | also keep states where a prompt stops sharing a kept one and at the end of its system block, so a new turn on the same system prompt resumes it |
| `ATTN_ENV` | `TF_GLM_MULTI_PREFILL=1` | prompts that arrive together share prompt chunks |
| `ATTN_ENV` | `TF_GLM_FILL_BUDGET_MS=200`, `TF_GLM_FILL_DRAFTS=1` | a prompt that arrives while others decode fills a few layers at a time, with drafted decode rounds between the slices |
| `ATTN_ENV` | `TF_GLM_STREAM_SMOOTH=1`, `TF_GLM_STREAM_SMOOTH_MS=400` | streamed tokens are paced from a short playout buffer |
| `ATTN_ENV` | `TF_GLM_Q4_TILE=table`, `TF_GLM_HC_DEC=1`, `TF_GLM_LATENT_STAGES=3`, `TF_GLM_KDA_WIDE_ROWS=1`, `TF_GLM_DECODE_SELECT=1` | her decode-window kernels on the attention node: per-shape 4-bit matmul tiles, the hyper-connection dots, the latent programs' pipeline depth, the wide KDA chain, and decode selection over the pools a row can see |
| `EXPERT_ENV` | `TF_GLM_EXL3_DEC=1`, `TF_GLM_EXL3_FUSE=1`, `TF_GLM_EXL3_XROW=1`, `TF_GLM_EXL3_LOADS=nc` | her decode kernel for the routed experts, adapted onto TensorFold's universal EXL3 kernel: fused down projection, one rotated input a row, 16-byte trellis loads one step ahead |
| `EXPERT_ENV` | `TF_GLM_EXL3_PROMPT=1` | her EXL3 prompt kernel for the routed experts' prompt chunks (patch 0024; [below](#her-exl3-prompt-kernel-on-the-sparks)) |
| `ATTN_ENV` | `TF_GLM_DFLASH_POLICY=fnc5:0.2` | her noise-aware DFlash2 draft policy: up to 5 drafts while the product of noise-aware confidences holds 0.2. It changes how many rows a round verifies, never the replies. Her lane runs it at `fnc7:0.3` ([deviations](#deviations-from-her-main-lane)) |
| `ATTN_ENV` | `TF_GLM_TOOL_CALLS=1`, `TF_GLM_KEEP_REASONING=1024` | her GLM tool-call fixes: calls streamed whole in their place, calls at the end of the think block, agent histories that render; a tool-calling reply's reasoning kept for clients that drop it |
| `ATTN_ENV` | `TF_GLM_KDA_CHUNKED=1` | her chunked KDA prompt kernel (32-row sub-chunks, chunked WY form); prompt chunks and kept prompt states sit on a 64-token grid, and an identical prompt replays its kept end state |
| `ATTN_ENV` | `TF_GLM_PREFILL_PAIRS=1` | this recipe's prompt chunk pairs (patch 0022): while no stream decodes, two consecutive prompt chunks go through the layers together, one chunk's attention on the 5090 while the Sparks compute the other's experts; 144 MiB of buffers for the second chunk |
| `ATTN_ENV` | `TF_GLM_CACHE_ROOM=1` | this recipe's room rule for her pool code (patch 0025; [below](#the-pools-room-rule)) |
| `ATTN_ENV` | `TF_GLM_PREFILL_ORDER=sjf` | this recipe's shortest-first order for her grouped prompt chunk (patch 0029; [below](#shortest-first-prompt-order)) |
| `MCDMA_INFLIGHT=2` | `TF_GLM_MCDMA_INFLIGHT=2` | two MoE exchanges in flight to each Spark, each on its own link (patch 0027; [below](#two-exchanges-in-flight)) |
| `ATTN_ENV` | `TF_GLM_LATENT_MMA=1`, `TF_GLM_HC_MMA=1`, `TF_GLM_SEQ_ROWS=256`, `TF_GLM_SPARSE_ONEPASS=1`, `TF_GLM_PROMPT_DSA=1` | her attention-side prompt kernels (patch 0028; [below](#her-attention-side-prompt-kernels)) |
| (default) | `TF_GLM_EXL3_NFIRST`, `_PF`, `_ORDER`, `_ROWROT` | the expert nodes' prompt-window launch order, register prefetch and per-row rotation, on by default for windows of 64 rows or more; `=0` turns one off |

Each switch is off in TensorFold unless set, and most change no reply. On this hardware, with v2.0's switches on (the
rows from `TF_GLM_SHARED_PREFIX` to `TF_GLM_KEEP_REASONING`, without `TF_GLM_EXL3_PROMPT` and the draft policy), the
greedy replies (5 of 5) and every row of the concurrent-stream check (33 of 33) are bit-identical to the same lane's
replies without them, and each new kernel's output is byte-identical to the kernel it replaces (1,323 checks on the
5090, 233 on each Spark). This update's draft policy setting, room rule, shortest-first order and second exchange in
flight change no reply either; each section below gives its check. Three change the prompt arithmetic, so the replies:

- `TF_GLM_KDA_CHUNKED=1`: its kernel is byte-identical to Mia's (36 of 36 cases: two head counts, six row counts, three
  positions), and against the serial kernel's log-probs over 1,866 positions it gives a mean KL of 0.002967, a p99 of
  0.05025 and 98.821% top-1 agreement, the signature of reduction-order noise rather than a bias. With it on, replies
  stay deterministic and drafted replies equal undrafted ones; unset, the serial kernel's replies come back bit for bit.
- `TF_GLM_EXL3_PROMPT=1`: the expert nodes' prompt rows take her prompt kernel's bits
  ([below](#her-exl3-prompt-kernel-on-the-sparks)).
- three of her five attention-side prompt switches, `TF_GLM_LATENT_MMA`, `TF_GLM_HC_MMA` and `TF_GLM_SPARSE_ONEPASS`
  ([below](#her-attention-side-prompt-kernels)).

`TF_GLM_PREFILL_PAIRS=1` changes only the order of two chunks' kernels, never a chunk's rows or arithmetic: with it on,
the greedy replies (5 of 5), the 22 concurrent requests' 33 recorded cases, and the prompt cache's replies and cached
counts equal pairs off, bit for bit, on top of `TF_GLM_KDA_CHUNKED=1`.

### Deviations from her main lane

- **Her DFlash2 draft policy runs at `fnc5:0.2`, not her lane's `fnc7:0.3`.** On this split, with the chunked KDA
  kernel and prompt pairs on, a coding-agent load ran 7.9% slower with `fnc7:0.3` at four 1,024-token requests (222.4
  against 241.6 tok/s) and 5.3% slower at one 4,096-token request (71.4 against 75.4 tok/s). Her policy was tuned on
  replies of the serial KDA kernel; the chunked kernel's replies differ slightly, and the drafter drafts fewer rows a
  round on them. `fnc5:0.2`, the same noise-aware rule with up to five drafts and a 0.2 threshold, keeps most of the
  policy's decode gain without that cost ([the draft policy setting](#the-draft-policy-setting)). Put
  `TF_GLM_DFLASH_POLICY=fnc7:0.3` in its place to run her lane's.
- **Eight streams, where her main lane serves four** (`PARALLEL=8`, her recipe patch 0069, with the 64-row verify
  window her recipe sets past four requests). On the 5090 the extra streams' working memory comes out of the cache
  budget, so the KV pool is 727,040 tokens instead of 966,656 ([eight streams](#eight-streams)).
- **Prompt chunk pairs are on** (`TF_GLM_PREFILL_PAIRS=1`, this recipe's patch 0022). Her two-Spark recipe has no
  equivalent: its prompt overlap works across its two TP ranks, which this split does not have.
- **Two MoE exchanges are in flight to each Spark** (`MCDMA_INFLIGHT=2`, patch 0027). Her lane has no wire between
  attention and experts; on this split the second exchange keeps the Sparks busy ([below](#two-exchanges-in-flight)).
- **Two changes to her code's behaviour** (this recipe's, each behind its own switch): the pool's room rule
  (`TF_GLM_CACHE_ROOM=1`, patch 0025) changes which kept prompts her pool code (her recipe patch 0030) evicts, and
  shortest-first prompt order (`TF_GLM_PREFILL_ORDER=sjf`, patch 0029) changes the order in which her grouped prompt
  chunk (her recipe patch 0049) fills waiting prompts. Neither changes a reply.
- **Her five attention-side prompt switches are on, as in her lane** (patch 0028). They change the prompt arithmetic
  and their reference KL is just over the gate, so they were to ship on only if AEON-30 held 21 of 30 with them; it
  held 23 ([below](#her-attention-side-prompt-kernels)).

## The switches

Each switch below is a lever this update adds: whose work it is, how it is set, what it does and what it measured.
Each was measured on the hardware this recipe targets against the same configuration without it, in the order they
were added; [BENCHMARKS](BENCHMARKS.md#the-levers-one-at-a-time) names each baseline. The release configuration's own
numbers come from one boot with all of them on: [BENCHMARKS](BENCHMARKS.md#the-release-configuration).

### Her EXL3 prompt kernel on the Sparks

- **Whose:** MiaAI-Lab's: her recipe patch 0004 (`glm-prompt-kernels`), with 0009's shared-row inputs and 0020's item
  order, at her recipe's v1.5 commit `1576746`. Her kernels are carried byte for byte; this recipe's part is the data
  movement onto TensorFold's per-expert weight tables, and a plan kernel that lays TensorFold's grouping out as her
  plan's items.
- **Switch:** `TF_GLM_EXL3_PROMPT=1` in `EXPERT_ENV` (patch 0024; `TF_GLM_EXL3_PASS`, `_ORDER` and `_ROWROT` keep her
  defaults).
- **What:** a prompt chunk's routed experts run through her dedicated prompt kernel: per pass of up to 64 members of
  one expert, each output is one fp32 chain over ascending k tiles (no split K), and the gate/up and down epilogues are
  fused. A row's bits depend on its own values only, so any chunking gives the same prompt state. Decode windows keep
  the grouped kernel.
- **Measured:** the kernel, per half and MoE layer at 2,048 rows, **28.41 -> 16.49 ms** (x1.72). A cold 100K-token
  prompt's first token **76.01 -> 52.74 s (−30.6%)**, a cold ~2.6K prompt's **2.230 -> 1.656 s (−25.7%)**; fresh 8K /
  31K / 62K prompts **1,244 / 1,296 / 1,320 -> 1,850 / 1,879 / 1,909 tok/s (+48.7 / +45.0 / +44.6%)**. Decode
  unchanged (spark-bench +1.2% geometric mean, coding +2.1%: the replies' texts changed, and each cell follows its
  verify rounds).
- **Replies:** prompt rows take her kernel's bits, so replies change. Against the serial KDA kernel's reference
  log-probs (1,866 positions) the mean KL is **0.002808**, p99 0.04700, top-1 **99.14%**: closer than without it
  (0.002967, 0.05025, 98.82%). On both GB10s this kernel's outputs equal her own kernel's byte for byte (0 of 128 and 0
  of 64 cases differ).

### The pool's room rule

- **Whose:** this recipe's change to MiaAI-Lab's `_room` and `_grow` (her recipe patch 0030, unchanged at v1.5), which
  keep the shared cache pool.
- **Switch:** `TF_GLM_CACHE_ROOM=1` in `ATTN_ENV` (patch 0025).
- **What:** on the 5090 the pool is one card's. Her rule evicts kept prompts, least recently used first, until a gap
  fits, and moves the extents together only once nothing is left to evict. In the arena's long cells that evicted the
  contexts queued requests were about to resume while the free rows already covered the need, and each such miss
  refilled its whole context and evicted the next. With the switch, room comes from the free rows first (the extents
  move together), and kept prompts are evicted only for rows the pool lacks, those no waiting request resumes first.
  Replies, prompt chunks and kept points do not change; only which kept prompts survive, and where the extents sit.
- **Measured** (with her prompt kernel and twenty kept prompts at four streams, against v2.0's first staging): the
  arena's two worst-miss cells resumed every inference request: 100,000 x 5 **15 of 15** (cache hits 49.5%, the ideal;
  29.7% before) and 65,535 x 10 **30 of 30** (49.2%, the ideal; 19.7% before). Their first tokens came in **7.4 s and
  15.8 s** (83.1 s and 151.7 s before), with generation at 47.8 and 40.6 tok/s (16.1 and 3.5 before), ahead of Mia's
  recipe v1.4 on its two Sparks in both cells (8.1 s and 16.9 s; 42.6 and 38.7 tok/s). The rule moved 319 extents and
  evicted 74 kept prompts, none that a waiting request wanted. Twenty kept prompts alone had already resumed 100,000 x
  5 at this pool, so these cells show the whole stack holding the ideal, not the room rule's gain alone. Cold prompts
  are unchanged (100K −0.3%, 2.6K −1.0%).

### Twenty kept prompts

- **Whose:** TensorFold's setting (`TF_GLM_CACHE_ENTRIES`); the value is this recipe's.
- **Switch:** `CACHE_ENTRIES=20` in `.env` (32 before).
- **What:** each kept entry reserves about 183 MiB of `TF_GLM_CACHE_GIB` (a KDA state and a DFlash2 window), and the
  KV pool gets the rest. At 8 GiB and four streams, 32 entries leave a pool of 626,688 tokens, 20 leave **966,656**,
  and 48 would shrink it to 286,720.
- **Measured** (at 8 GiB and four streams): on 100,000 x 5 every inference request resumed (20 of 20 with
  llama-benchy's warm-up round; cache hits 49.5%): first token **83.1 -> 9.3 s**, generation 16.1 -> 39.9 tok/s.
  Fresh-boot numbers did not move (spark-bench +0.4%, coding −0.1%, cold prompts within 1%). A cell keeps about two
  states a client, so ten-client cells can reach the cap: 65,535 x 10 reached it once (with the room rule) without
  losing a resume. The measured fallback is `CACHE_GIB=10` with 32 entries (a pool of 946,176 tokens; also 20 of 20 on
  100,000 x 5).

### Two exchanges in flight

- **Whose:** Hugh Madden ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden))
  authored the exchanges kept in flight ahead of the expert ranks in
  [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at `91db3cc` (`crates/glm53f-serve/src/lib.rs`, lines 74-75
  and 153-165: a prefill pass of N lanes keeps N exchanges in flight over RDMA, as many as the ranks queue). Ash Hart
  ([@ashxhart](https://x.com/ashxhart), [ashhart](https://github.com/ashhart)) authored
  [MCDMA](https://github.com/ashhart/MCDMA), whose protocol carries one request per link, so exchanges in flight take
  more links. Patch 0027 writes the queued exchanges for MCDMA and this tree's prompt chunks; no code is copied.
- **Switch:** `MCDMA_INFLIGHT=2` in `.env`: the scripts start two listen daemons on each Spark and pass
  `TF_GLM_MCDMA_INFLIGHT=2` to the attention node (patch 0027; 1 = one link to each Spark).
- **What:** [the wire](#the-mcdma-wire). The Sparks find the next request waiting when they reply, instead of idling
  while a reply crosses the wire and the next request comes back.
- **Measured** (against the same configuration with one link pair): each Spark's idle time per exchange **7.33 ->
  2.03 ms** at a cold 100K prompt (1.94 ms at 2.6K and 31K), so the experts are busy 89% of a long prompt's MoE time,
  from 69%. Fresh 8K / 31K / 62K prompts **1,823 / 1,892 / 1,922 -> 2,284 / 2,335 / 2,419 tok/s (+25.3 / +23.4 /
  +25.9%)**; a cold 100K prompt's first token **52.08 -> 41.64 s (−20.0%)**; a cold ~2.6K prompt's **1.637 -> 1.487 s
  (−9.2%)**. Replies are bit-identical (greedy 5 of 5; 22 concurrent requests, 33 of 33 cases), and the wire check
  found 0 mismatches in 256 checked forwards. Decode is 0.5-1.6% lower in every cell (spark-bench −1.1% geometric
  mean, coding −0.9%).

### Her attention-side prompt kernels

- **Whose:** MiaAI-Lab's: her recipe patches 0004 (`glm-prompt-kernels`), 0009 (`glm-prefill-kernels`) and 0028, with
  0038's FP8 cache rows, at her recipe's v1.5 commit `1576746`. The kernels are hers verbatim: on the same seeded
  inputs their outputs give her tree's digests, 11 of 11 on the RTX 5090, and this repo's CPU tests check the same
  digests under the Triton interpreter. On the RTX 5090 the port's GPU check held 12 of its 13 comparisons: the
  hyper-connection dots on the tensor cores (`TF_GLM_HC_MMA`) differ from the stock path by up to 0.0039 in the
  output, over the check's 0.001 bound; that switch changes the arithmetic by design.
- **Switches** (patch 0028, each off unless set): `TF_GLM_LATENT_MMA=1` (a prompt chunk's absorb and expand on the
  tensor cores), `TF_GLM_HC_MMA=1` (the hyper-connection mixing dots on the tensor cores), `TF_GLM_SEQ_ROWS=256` (from
  256 rows, the BF16 / FP8 matmuls run every K slice in one program), `TF_GLM_SPARSE_ONEPASS=1` (a prompt chunk's
  sparse rows in one online softmax a row) and `TF_GLM_PROMPT_DSA=1` (a prompt chunk's DSA selection through her prompt
  path; this port's name for her unconditional path). `TF_GLM_SEQ_ROWS` and `TF_GLM_PROMPT_DSA` give the same bits
  (greedy 5 of 5 and 33 of 33 cases equal without them, on hardware); the other three change the prompt arithmetic,
  so the replies.
- **Measured** (all five, at four streams, against the configuration without them, each after a warm-up of every
  prompt size): a cold ~2.6K prompt's first token **1.526 -> 1.404 s (−8.0%)**, fresh 8K / 31K / 62K prompts **2,337 /
  2,375 / 2,451 -> 2,644 / 2,696 / 2,770 tok/s (+13.1 / +13.5 / +13.0%)**, a cold 100K prompt's first token **41.39 ->
  36.62 s (−11.5%)**. Decode moves with the replies: spark-bench one stream prose +1.7%, code −5.6%, JSON +4.7%, four
  streams −0.4%. Against the serial KDA kernel's reference log-probs the mean KL is **0.00309**, over the 0.003 gate
  (p99 0.0493 and top-1 98.87% pass; without them 0.00281, 0.0470 and 99.14%).
- **Rule:** they ship on if AEON-30 (30 agentic tasks) scores 21 of 30 or more with them. AEON-30 scored 23 of 30
  with them on the shipped configuration, so they are on in `.env.example`; remove the five from `ATTN_ENV` to run
  without them.
- **As shipped** (eight streams), against the same configuration without them, measured the same morning: fresh 8K /
  31K / 62K prompts +14.5 / +12.0 / +13.0%, a cold 100K prompt's first token 11.3% sooner (41.49 -> 36.79 s), and in
  the ten-client arena cells prompt rate +6.6 to +6.7% and the first token 4.3-6.8% sooner. One-stream decode: prose
  +2.0%, code −6.2%, JSON +4.5%, as the new texts take more or fewer verify rounds.

### Shortest-first prompt order

- **Whose:** this recipe's change to MiaAI-Lab's grouped prompt chunk (her recipe patch 0049, which fills the prompts
  with the fewest rows left first).
- **Switch:** `TF_GLM_PREFILL_ORDER=sjf` in `ATTN_ENV`, with `TF_GLM_PREFILL_AGE_MS` (default 30,000) (patch 0029).
- **What:** waiting prompts fill fewest uncached tokens first, so a short prompt that arrives while a 100K prompt fills
  takes the next chunk instead of waiting out the whole long fill; a prompt that has waited the aging bound is
  overtaken by no later arrival. The order changes when each stream's rows fill, never which rows or how they are cut:
  on hardware the greedy replies (5 of 5) and the 22 concurrent requests' 33 cases equal those without it.
- **Measured** (at four streams, a mixed load of 2,048- to 100,000-token prompts, against the same configuration
  without it): from two clients (24 requests) the median first token **28.32 -> 14.59 s** and the mean 28.28 -> 24.37 s;
  the 2,048-token prompts' mean **10.08 -> 3.72 s** (p90 41.86 -> 8.65 s), the 100,000-token prompts' 50.70 -> 58.12 s.
  From five clients (40 requests) the median 85.17 -> 77.09 s and the mean 96.25 -> 95.04 s, but there the 2,048-token
  prompts waited longer (mean 52.57 -> 92.08 s) while every longer size came sooner: at that load most waits are past
  the 30 s aging bound, beyond which prompts go by arrival.

### Eight streams

- **Whose:** MiaAI-Lab's: her recipe patch 0069 (`glm-eight-streams`) at her recipe's v1.5 commit `1576746`, ported,
  with the 64-row verify window her recipe sets past four requests (`TF_GLM_MULTI_WINDOW` defaults to 32 here).
- **Switch:** `PARALLEL=8` in `.env` with `TF_GLM_MULTI_WINDOW=64` in `ATTN_ENV` and `CACHE_GIB=6.5` (patch 0030;
  `PARALLEL` 1-8, the window 16-64 rows).
- **What:** up to eight requests decode together. At four or fewer streams with the window unset, every table, buffer
  and kernel call is the one before. On the 5090 the extra streams' working memory comes out of the cache budget: at
  eight streams the card went over its memory gate (27,136 MiB) with a 7.5 GiB budget (27,600 MiB) and with 7 GiB and
  the 64-row window (27,370 MiB), so the budget is 6.5 GiB and the KV pool 727,040 tokens instead of 966,656.
- **Measured** (against four streams at 8 GiB, the arena's 32K and 64K cells at five and ten clients): the first token
  **15.0 -> 13.2 s** at 32,768 x 10 and **15.3 -> 13.8 s** at 65,535 x 10 (−10.8% geometric mean), 6.5 -> 6.2 s and 6.7
  -> 6.5 s at five clients; prompt rates 1.12-1.34x and generation 1.05-1.08x in all four cells (65,535 x 10: 768.9 /
  43.8 tok/s against 688.6 / 40.6), each ahead of Mia's recipe v1.4 on its two Sparks (624.6 / 38.7 tok/s, 16.9 s).
  Single-stream decode unchanged (−0.6% to +0.2%); every concurrent reply equals its solo reply with up to eight
  decoding together (22 of 22); the 5090 peaked at 26,926 MiB.

### The draft policy setting

- **Whose:** MiaAI-Lab's noise-aware DFlash2 draft policy (her recipe patches 0018 and 0021, patch 0019 here); the
  `fnc5:0.2` setting is this recipe's.
- **Switch:** `TF_GLM_DFLASH_POLICY=fnc5:0.2` in `ATTN_ENV`.
- **Measured** (against the policy unset): spark-bench decode **+7.0%** (geometric mean; one stream +7.7 / +2.6 /
  +6.4% prose / code / JSON, two streams +10.2%, four +8.4%); the coding-agent load −0.5% at four 1,024-token requests
  and −1.1% at one 4,096-token request (74.6 against 75.4 tok/s); replies bit-identical, since the policy changes only
  how many drafts a round verifies. The prompt set is untouched (within ±0.8%). Her lane's `fnc7:0.3`:
  [deviations](#deviations-from-her-main-lane).

## Measured and off

These ship in the patches, off. Each was measured on this hardware and not adopted.

### Prefill lanes and FP8 wire rows

- **Whose:** Hugh Madden authored both in [glm53f-afd](https://github.com/hughmadden/glm53f-afd) at `91db3cc`: prefill
  in four lanes of 2,048 rows (`crates/glm53f-forward/src/forward.rs`, `run_lanes`), which generalizes the two-lane
  prefill of whole prompt chunks in his [mimo26f-afd](https://github.com/hughmadden/mimo26f-afd) at `bab9fa2`, and FP8
  E4M3 wire rows with a UE8M0 scale per 32 values (`crates/glm53f-coordinator/kernels/wire.cu`). That row format is
  DS41RT's, by T.J. Purtell ([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell),
  [ds41rt](https://github.com/tpurtell/ds41rt)), carried through mimo26f-afd. Patch 0026 writes both for TensorFold
  and MCDMA; no code is copied.
- **Switches:** `TF_GLM_PREFILL_LANES=N` (1-4; unset follows prompt pairs: 2 with them on, else 1) and
  `TF_GLM_WIRE_FP8=1` (patch 0026).
- **Why off:**
  - Four lanes with four exchanges in flight ran the 5090 out of memory on the first ~8K cold prompt, at 31,800 MiB:
    each later lane adds one more 1.25 GiB working segment of the sparse attention, not just its 193 MiB of buffers.
  - FP8 wire rows (4,224 bytes a row at hidden 4,096, against bf16's 8,192) fail the KL gate against the serial KDA
    kernel's reference log-probs (mean 0.00454, p99 0.0772, top-1 98.39%; the gate is 0.003, 0.05 and 98.5%), and
    they bought no prefill speed (the four prompt rates' geometric mean −0.8%): the wire's bytes are not what limits
    prompts here.

## The patches

`build.sh` applies `patches/` to TensorFold v0.6.5 in order with `git am`, so the built tree's `git log` keeps each
patch's author, message and credits.

| No. | What | On by default here | Source |
| --- | --- | --- | --- |
| 0001 | the attention/expert split over MCDMA (`--experts remote`, `tensorfold experts`) | yes | this recipe |
| 0002 | segmented KDA and DSA kernels for multi-stream verify windows | with `--parallel` | MiaAI-Lab, [TensorFold#243](https://github.com/ashhart/TensorFold/pull/243) |
| 0003 | DFlash2 for several streams | with `--parallel` | MiaAI-Lab, #243 |
| 0004 | `--parallel N`: one cache pool, batched verify rounds | with `--parallel` | MiaAI-Lab, #243 |
| 0005 | a tool call the end token left open is sent when it parses whole; a missing `<arg_key>` is put back | yes | MiaAI-Lab, [#285](https://github.com/ashhart/TensorFold/pull/285) |
| 0006 | sealed rank messages, an exiting watchdog, `/health`'s `iteration_s` | with `--parallel` | MiaAI-Lab, #243 |
| 0007 | a stopped request ends on every rank within a round | yes | MiaAI-Lab, [#301](https://github.com/ashhart/TensorFold/pull/301) |
| 0008 | `--parallel` and #301's stop vote on the attention node (world 1) | with `--parallel` | this recipe |
| 0009 | `TF_GLM_DENSE=q4` on the attention node | `DENSE=q4` | port of MiaAI-Lab's recipe patches 0002, 0005, 0013, 0028 |
| 0010 | `TF_GLM_KV=fp8` on the attention node | `KV=fp8` | port of her patch 0038 and 0041's ring span |
| 0011 | the split's page in TensorFold's docs (`docs/recipes/glm-5.3-flash-afd.md`) | — | this recipe |
| 0012 | shared-prefix prompt states, `TF_GLM_SHARED_PREFIX` | `ATTN_ENV` | port of her patches 0015 and 0063 |
| 0013 | prompt launch order, register prefetch and per-row rotation for the routed experts | yes | port of her patches 0001, 0020, 0009 |
| 0014 | waiting prompts share a prompt chunk, `TF_GLM_MULTI_PREFILL` | `ATTN_ENV` | port of her patch 0049 |
| 0015 | smooth streaming from a playout buffer, `TF_GLM_STREAM_SMOOTH` | `ATTN_ENV` | port of her patch 0061 |
| 0016 | layer-sliced prompt fills, `TF_GLM_FILL_BUDGET_MS`, over a slice-aware wire | `ATTN_ENV` | port of her patch 0062; the slice wire is this recipe's |
| 0017 | her decode-window kernels on the attention node and her decode kernel for the routed experts (`TF_GLM_Q4_TILE`, `TF_GLM_HC_DEC`, `TF_GLM_LATENT_STAGES`, `TF_GLM_KDA_WIDE_ROWS`, `TF_GLM_DECODE_SELECT`; `TF_GLM_EXL3_DEC`, `_FUSE`, `_XROW`, `_LOADS`) | `ATTN_ENV`, `EXPERT_ENV` | port of her patches 0016, 0019, 0031, 0043, 0047 (0047's loads after Jay Leaton's patch 0580) |
| 0018 | a test follow-up to 0017: the prompt-switch test clears the decode switches first | — | port, as 0017 |
| 0019 | noise-aware DFlash2 draft policies, `TF_GLM_DFLASH_POLICY` | `ATTN_ENV` (`fnc5:0.2`; [deviations](#deviations-from-her-main-lane)) | port of her patches 0018, 0021 |
| 0020 | GLM tool calls streamed whole in their place, calls at the end of the think block, agent histories that render, kept reasoning (`TF_GLM_TOOL_CALLS`, `TF_GLM_KEEP_REASONING`) | `ATTN_ENV` | port of her patches 0036, 0051 and 0056's text part (parts of 0036 after Jay Leaton's patch 0620) |
| 0021 | a GPU test passes the FP8 KV kernel's newer arguments (test only) | — | this recipe |
| 0022 | prompt chunk pairs over the wire (`MODE_PAIR`), `TF_GLM_PREFILL_PAIRS` | `ATTN_ENV` | this recipe |
| 0023 | the chunked KDA prompt kernel, a 64-token prompt grid and prompt replay, `TF_GLM_KDA_CHUNKED` | `ATTN_ENV` | port of her patches 0012, 0014, 0039, 0008, 0042 |
| 0024 | her EXL3 prompt kernel for the routed experts' prompt chunks, `TF_GLM_EXL3_PROMPT` | `EXPERT_ENV` | port of her patches 0004, 0009, 0020 |
| 0025 | the pool's room rule, `TF_GLM_CACHE_ROOM` | `ATTN_ENV` | this recipe's change to her patch 0030's pool code |
| 0026 | prefill lanes, `TF_GLM_PREFILL_LANES`, and FP8 wire rows, `TF_GLM_WIRE_FP8` | no ([measured and off](#measured-and-off)) | this recipe's code; Hugh Madden authored the designs in glm53f-afd and mimo26f-afd, and the FP8 row format is T.J. Purtell's (ds41rt) |
| 0027 | several MoE exchanges in flight to each expert node, `TF_GLM_MCDMA_INFLIGHT` | `MCDMA_INFLIGHT=2` | this recipe's code; Hugh Madden authored the design in glm53f-afd |
| 0028 | her attention-side prompt kernels behind five switches | `ATTN_ENV` | port of her patches 0004, 0009, 0028 |
| 0029 | shortest-first prompt order with aging, `TF_GLM_PREFILL_ORDER` | `ATTN_ENV` | this recipe's change to her patch 0049's grouped chunk |
| 0030 | up to eight concurrent streams, `TF_GLM_MULTI_WINDOW` | `PARALLEL=8`, `ATTN_ENV` | port of her patch 0069 |

Her pull-request commits are her commits, with her authorship. Each port names her recipe patches in its subject,
keeps `Co-authored-by: MiaAI-Lab`, and carries a credit header (the recipe repository, the commit it was ported
from, Apache-2.0) in the patch notes; TensorFold's `THIRD_PARTY_NOTICES.md` in the built tree names each port too.
The two changes of this recipe's to her ported code (0025, 0029) keep `Co-authored-by: MiaAI-Lab` as well, and their
notes name her patch and recipe commit and say the change is this recipe's. Patches 0026 and 0027 name Hugh Madden's
files and commits, and T.J. Purtell's row format, in their messages and in `THIRD_PARTY_NOTICES.md`.
`tools/export_patches.sh` regenerates `patches/` from a TensorFold branch and checks all of this. The release
configuration was measured on the same code plus one commit of exchange and round timers, which were switched off in
every measured run and are not shipped.

## What runs where on the hosts

- `AFD_HOME/tree`: the patched TensorFold tree, mounted read-only at `/tf` with `PYTHONPATH=/tf/src`. The image is
  NVIDIA's PyTorch container with the same tree installed for its Python dependencies, so a tree update needs a
  `./build.sh sync`, not an image rebuild.
- `AFD_HOME/ext-<tree id>`: the CUDA extension cache, one per tree (pinned commit plus patches). `./build.sh ext`
  fills it with no model loaded; the serving containers mount it.
- `AFD_HOME/mcdma`: `mcdma-rpcd`, `libmcdma-rpc.so`, the daemons' sockets and logs.
- `AFD_HOME/logs`: `attn.log`, `expert0.log`, `expert1.log`.
- The checkpoint at `MODEL_DIR`, the same path on every host, mounted read-only; the drafter on the attention host.

## Where the time goes

Decode is bound by the expert side: per forward, the Sparks' expert compute plus the exchange takes longer than the
attention node's work. Prompt processing is bound by the experts too: in one profile of a 2,642-token prompt on an
earlier build, about 87% of the prefill time was the experts and the hop to them, while the 5090 sat idle. The
prompt-side ports (0012-0016, 0023) attack that: kept shared prefixes, Mia's expert prompt kernels, grouped and sliced
fills, the chunked KDA kernel. KDA is a small part of a prompt chunk on the 5090: the chunked kernel takes about half
the serial kernel's time (0.52-0.57x at 594-2,048 rows), which saves about 18 ms of a 1,024-row chunk's ~985 ms
(about 1.9%).

Prompt chunk pairs (patch 0022) put the idle 5090 to work: while the Sparks compute one chunk's experts, the 5090 runs
the next chunk's attention. A prompt alone fills in 2,048-row chunks. At that size the Sparks compute a layer's experts
in about 28 ms and a serial exchange takes 36.5 ms at the median (40.3 ms at p90); in a pair the exchange window is
0.975x the serial one at p50 and 0.977x at p90, and the 5090's work inside that window, the other chunk's attention,
takes at most 21.9 ms at p90, inside the experts' compute time. Fresh 8K-62K prompts fill 23-25% faster (1,298-1,330
against 1,053-1,066 tok/s), and a cold 100K-token prompt's first token comes in 75.4 s instead of 93.9 s. The Sparks'
expert compute is now the floor: a paired exchange still waits about 27.5 ms for its reply at the median.

With pairs on, a 2,048-row chunk's exchange window took 35.95 ms per MoE layer: the Sparks' expert kernels 28.4 ms
(79%) and the wire and handoff 7.6 ms (21%), and the 42 exchanges were 96% of a cold prompt. Her prompt kernel (patch
0024) cut the kernels to 16.5 ms, which left the wire and handoff at 29-31% of every window; two exchanges in flight
(patch 0027) hide most of that, so each Spark idles 2.03 ms per exchange instead of 7.33 ms. Her attention-side
prompt kernels (patch 0028) shorten the 5090's own part: a cold 100K prompt's first token 11.5% sooner.

## Limits

- The `glm5_next` CUDA family and EXL3 checkpoints only; `tensorfold experts` refuses anything else.
- Eager execution only across the wire.
- Up to eight requests at once (`PARALLEL`, 1-8); without `--parallel`, one at a time. Past four streams the 5090's
  extra working memory comes out of the cache budget (`CACHE_GIB=6.5` at eight).
- 262,144 tokens a request is the tested window; the startup estimate admits what fits. On a 32 GB card it needs
  `TENSORFOLD_MEMORY_RESERVE_GIB=2` (the default reserve leaves the estimate about 1.5 GiB short).
- The MTP head is not carried over the wire: the attention node does not load it and refuses `TF_GLM_MTP=1`.
  DFlash2 drafts instead.
- `AFD_CHECK=1` clones each layer's wire tensors (~3.4 GB at a 2,048-row chunk), so on a 32 GB card it fits only at a
  window of 98,304 or less.
- More than eight requests at once queue for a slot. In the release configuration's arena cells (ten clients on 32K
  and 64K contexts, five on 100K) it led Mia's recipe v1.4 in prompt rate (1.29-1.54x), generation (1.16-1.19x) and
  first token (0.79x the time), while at ten clients each request decoded slower (0.66-0.69x): eight decode together
  here, four in her main lane ([BENCHMARKS](BENCHMARKS.md#three-arena-cells-as-shipped)).
- One KV pool of 727,040 tokens at eight streams (966,656 at four with 8 GiB) holds every request's cache and the kept
  prompt states; Mia's two Sparks hold about 1.74-1.80M tokens. Deep contexts that outgrow it evict kept states and
  fill prompts again. At four streams with the room rule, the arena's 5 x 100K and 10 x 65K cells resumed every
  request ([the pool's room rule](#the-pools-room-rule)); as shipped, at eight streams, they did too (cache hits 49.5%
  and 49.2%, the ideal).
- `MCDMA_INFLIGHT` is 1-3 in the scripts: one connect daemon on the attention host holds six peers, two links a
  setting. Patch 0027 takes up to 4, which needs a second connect daemon.
- Prompt chunk pairs run only while no stream decodes; a prompt that arrives while others decode fills in layer slices,
  unpaired.
- With `tool_choice: "none"` the server offers the model no tools, but the model can still write tool-call markup as
  plain text. That is TensorFold 0.6.5's behaviour, unchanged by patch 0020.
- Not ported from Mia's recipe:
  - copy drafts and their 16-row verify windows (her patches 0007, 0013, 0032);
  - the parts that only make sense on two TP ranks: the hyper-connection split, the prefill overlap across ranks, her
    RoCE all-gather and L2 prefetch (the split has no all-gather; the experts talk to the attention node, not to each
    other). Prompt chunk pairs (patch 0022) are this recipe's own overlap for the split;
  - FP8 dense weights;
  - vision (0003, 0050, 0054, 0056's image path), left out by choice: this recipe serves text only.

## Who designed the split

Hugh Madden ([@dangerm00se](https://x.com/dangerm00se), [hughmadden](https://github.com/hughmadden)) wrote
[glm53f-afd](https://github.com/hughmadden/glm53f-afd) (`91db3cc`), which serves GLM-5.3-Flash with attention on an RTX
5090 and the routed experts on DGX Sparks, the engine of this repo's v1.0, and
[mimo26f-afd](https://github.com/hughmadden/mimo26f-afd), its serving shell, wire and RDMA transport. T.J. Purtell
([@wrldsuksgo2mars](https://x.com/wrldsuksgo2mars), [tpurtell](https://github.com/tpurtell)) wrote
[ds41rt](https://github.com/tpurtell/ds41rt), [glmrt](https://github.com/tpurtell/glmrt-5.3-1rtx-4spark) and
[cuteafd](https://github.com/tpurtell/cuteafd), engines that run attention on RTX GPUs and the routed experts on DGX
Sparks. This recipe runs that split on TensorFold over MCDMA and uses no code from their engines. Three of Hugh
Madden's designs are written here for this tree: the exchanges kept in flight ahead of the expert ranks (patch 0027,
on), and the prefill lanes and FP8 wire rows (patch 0026, off), whose row format is T.J. Purtell's DS41RT format.
Everything else is credited in [NOTICE.md](../NOTICE.md).
