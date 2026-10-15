# Troubleshooting and safety

- Missing compiler image: build uses `--pull=never`. Obtain the pinned image under its own terms; do not substitute a moving tag.
- Missing verbs headers: install rdma-core development headers and set the two header paths in `.env`. A stub binary is not a serving build.
- Grammar refuses: check the attention build included its loader, the library is the pinned ABI, all twelve required symbols exist, ownership/parent permissions are safe, and its actual digest matches `GRAMMAR_LIB_SHA256`. Do not weaken the loader or reuse a different ABI's digest.
- Model load/admission fails: confirm world-1 dense q4 with MTP and the EXL3 checkpoint pin, then check pool size and real GPU/host-memory headroom. Do not raise limits just because four slots exist. Four maximum contexts do not fit the shared pool.
- GPU tenant already present: do not launch. GB10 uses unified memory outside container memory accounting; a second resident model can hang a Spark. Use an active host-memory-floor/single-tenant guard.
- Peer fails: stop the exact owned ranks gracefully, verify native exits and restored memory, and investigate the first failure. Never broad-kill by executable name, hard-kill MPS, clear someone else's lease or retry a failed stop blindly.
- Tool-copy probe fails: distinguish tokens the model did not generate from tokens the parser cut. The final literal-boundary gate must force its premise and fail on a truncated/altered closer. An ordinary copy request's failure is not silently excluded or changed to PASS.
- Rates differ from the table: the historical lab's short-request aggregate, mean request-wall rate and decode-excluding-TTFT rate are different denominators. The final reference comparison uses its stated same harness and window; older tags used different checkpoints/drafters and are not controlled baselines.

There is no automatic reboot, production cutover or recovery controller in this source-only preview. If a host hangs or retains a GPU owner, escalate to its operator; physical recovery is not an acceptable routine deployment step.
