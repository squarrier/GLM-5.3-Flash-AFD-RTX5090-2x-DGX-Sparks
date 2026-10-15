# concurrent_users independent users — CPU design, hardware equality UNKNOWN

Ash Hart (@ashhart/@ashxhart) authored TensorFold/MCDMA; Mia (MiaAI-Lab/@MiaAI_lab) authored inherited EXL3 recipes/exact.py; Hugh Madden, tpurtell, turboderp, bertholomus/Albert Lee, BadAd84, chaog992/NVIDIA and all inherited NOTICE/licence authors retain credit. GPT-6.1 Sol (OpenAI Codex; planning and execution), via Hermes, authored this local opt-in implementation and tests.

## Execution and ownership

TF_AFD_USERS accepts only unset, 1, 2 or 4, without whitespace coercion. Unset/1 branches to the original served path. No Users constructor or device buffer is made there; original CUDA/AFD layers, native one-user backend, prompt lane implementation, kernels and licences remain byte-identical to PRODFIX1 0987b353f71290224460d7f89bc4e9033d3af477. Separate default-false lane and engine-info fields fence the new behaviour.

Opted-in mode uses one LaneHost execution thread, one Fwd/stream, one Engine/model/weights/prompt scratch and one exchange provider. The original Engine State is slot zero; C-1 additional State descriptors/buffers are made, not C engines. Each backend slot maps one Stream owner, monotonic generation and first token. The Stream owns pending token, emitted length, EOS/cancellation. Each device State independently owns pos/parity, recurrent state, conv/cache/DSA storage. Stale generation releases do nothing; generation exhaustion refuses. Reset occurs at next generation prefill, never another owner's retirement.

Prefill is serial on the execution thread. For a nonzero slot its State descriptor swaps into e.st only during the existing prefill call, then swaps back under defer. All existing within-user PAIRS/LANES prompt code derives State views per call and still uses one user's State. No request's tokens become consecutive rows of another user's State. Requests arriving while decode executes wait in LaneHost; cancelled queued requests are never prefilled.

Decode validates all windows before any device call, sorts by physical slot, then runs layers outermost and independent users innermost. Each attention front uses R1 and its State. Saved x/normed/xs/post/comb/logits persist per user across other fronts. Dense layers stay R1. Each MoE layer gathers normed/xs into stable R=C buffers, runs the existing route/request/take exactly once, and scatters the three original expert0/expert1/shared planes for each row into R1 hcPost. The sum order and per-user residual/hyperconnection operation are unchanged. No transport, kernel, dtype or weight format is added.

After all tails the stream settles/checks before sampling. Each sample and commit has an alive check; cancellation suppresses sample/commit and the core suppresses publication after verify. A second settle completes position/conv writes before scratch reuse. Observed exchange/settle/sample failure is sticky across every owner; later admission/verify refuse before device calls. Failed device rounds are not ordinary-reusable. Normal EOS/length/cancellation retirement releases only the right generation and permits reuse.

## Fixed addresses and graphs

All selected device allocations are assigned once by Users.attach and remain fixed for the slot lifetime, including across retirement/reuse. State descriptor swapping changes which State a serial prefill references; it does NOT relocate selected device storage. Graphs are OFF and graph opt-in with users>1 refuses. No #615/NCCL splitting/typed relocation is implemented. qualification_d15badf9 can SKIP premise-absent for this design; it is not a lab prerequisite.

## Memory admission

Users.bytes walks exactly the buffers Users.attach places, aligning every device allocation to 256 bytes. It partitions C-1 persistent States and extra row/gather/scatter storage. Shared prompt/engine/exchange/pair/lane scratch is not copied; the existing sl.forecastAt accounts for it and weights, with its original 2GiB margin. The opt-in admit adds Users.bytes.total before toDevice loads weights; insufficient free memory refuses, without shrinking context. C4 uses the same map and admission rule.

Recorded full-model MTP-loaded metadata fixture at window262144/capacity262208, max_rows2048:

| Mode | Extra persistent State bytes | Extra row/workspace bytes | Extra total bytes |
| C2 | 5364978176 | 769024 | 5365747200 |
| C4 | 16094934528 | 1488896 | 16096423424 |

These are projections, not device measurements. Actual base weight form, free memory and total fit must be recorded in the future lab. C4 is not authorized to reduce the full262144 cap or use another host to fit.

## Mode limits and evidence

Opt-in multi-user decode is greedy, drafts-off and graphs-off only. Explicit requested positive/nonfinite temperature or draft:true refuses, and structured requests are refused by existing server feature checks (call_gates false). Native draft/sampled/cache-reuse windows refuse before prefill/round. Lone original MTP remains available only with users unset/1. Startup chooses temperature0 rather than inheriting model generation defaults; explicit CLI positive temperature refuses. Full context above262144 refuses in opted-in planning.

Integer fake fixtures exercise changed backend/core and the exact production scheduling function: C2/C4 together=independent serial; reordered arrival, unequal position/parity/DSA boundary/length, EOS, active/queued cancellation, slot reuse and failure fencing. Real CUDA recorder tests execute real Users front/gather/exchange/scatter/tail/sample/commit against recorded Fwd, comparing every attention-front op/grid/operand/address/scalar to serial and asserting actual R2/R4 headers/counts; failures go through a real CPU exchange Node's sticky failure state.

CPU launch/operand identity is not numerical equality of R=C router/shared/expert GEMMs. Hardware equality, real token/text equality, speed, TTFT, GPU memory and recovery are UNKNOWN/NOT RUN. No adoption, production change, public publication or additional graph/guard prerequisite follows from CPU passes.
