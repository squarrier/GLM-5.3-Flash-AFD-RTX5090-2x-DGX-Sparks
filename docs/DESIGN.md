# Architecture and constraints

Ash Hart authored the TensorFold native engine and MCDMA. This source snapshot runs native attention/expert disaggregation: one RTX 5090 owns world-1 attention, KV state, sampling and built-in MTP; two GB10 ranks each own half of every routed expert by intermediate channel. T.J. Purtell authored that expert split; Hugh Madden authored glm53f-afd's AFD design and prompt kernels. The CUDA transport here is native RoCE v2, not a separate external MCDMA-daemon deployment.

Mia authored the EXL3 lane, quantization, q4/FP8 preparation and kernel/drafter recipe work. The checkpoint's built-in MTP layer is Z.ai's model work. The served checkpoint is pinned, not chosen from moving main. Attention uses q4 dense weights and an FP8 head; no external DFlash2 checkpoint is loaded. Expert prompt computation selects Hugh's g53kt3r kernel. Other inherited kernel variants remain source, not simultaneous claimed speed gains.

## Four independent users

`TF_AFD_USERS=4` selects four user slots. The 262,144-token per-request cap does not reserve four complete contexts. All slots share a 720,896-token requested pool, with 720,888 tokens allocated from 2,816 blocks. Admission must reject or queue a request that cannot fit; pool size, model cap and active concurrency are different quantities.

Prefill work is sliced between decode rounds; historical long-fill tests still measured about 2.77-second decode gaps. Interleaving is not zero-latency service. Request-local state, sampling keys, grammar matchers, held drafts, rollback and retirement remain independent. Structured/sampled windows keep one row when speculative drafting cannot meet their constraints.

## Shared forward and drafted verification

Bizuayeu authored the shared-window design at TensorFold fork commits 75131b66, 146e113f and d3cc052c. The agents reimplemented it for this heterogeneous lane without copying his shared-window code. His tool/Responses fixes and tests are separately ported code and keep their authorship notices.

`TF_AFD_USERS_SHARED=1` computes a shared forward for user rows. `TF_AFD_USERS_SHARED_DRAFTS=1` shares drafted verification across up to eight total rows while maintaining each stream's state and acceptance/rollback. Default engine switches remain OFF. The example explicitly selects the qualified shared/shared-drafts ON profile, with batched MTP drafts and ping-pong ON. Each user retains independent state, acceptance and rollback. Ping-pong supports 16 logical rows as two physical micros of at most eight; the 16-physical-row extension is not included.

## Constraints and sampling

The attention build imports the trusted grammar-loader/ownership sources and loads a separately built XGrammar v0.2.8 C ABI library. The opened library inode's owner, mode, parent permissions, SHA256, ELF architecture and required symbols are checked before dynamic loading. Matchers are request-local. Masks apply before the sampling draw; a disallowed token has no mass.

These protections are not a sandbox for arbitrary clients. Preserve native schema/size checks and isolate the API. The literal tool-call closer repair treats a closer inside an open argument value as data. The historical unconstrained copy probe never generated its closer and therefore did not prove that served boundary. A constrained fixed probe, on the shipping head, is the final release gate; no premise waiver is claimed here.

## What is not inherited

No public binary redistribution, no production adapter, no private guard/controller, no external drafter, no vLLM code, no benchmark-client source, no release auto-push and no recovery watcher. Previous versions retain their own dependencies/credits at their tags and in NOTICE. Source presence is not evidence that an unused inherited backend is a qualified serving path.
