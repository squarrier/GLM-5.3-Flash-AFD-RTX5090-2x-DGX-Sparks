"""qualification_e4ad04cd: exl3_check.py's K4 case table (TF_EXL3_CHECK_K4=1). Python 2.15's tree (--src at src@0849f3f, with
TF_EXL3_CHECK_EXPERTS=tensorfold_exl3_experts_v4) runs layer L's routed experts of TP2 rank R through MiaAI-Lab's EXL3
prompt kernel as its halves call it for a prompt window: exl3_generic.routed with ``prompt`` and TF_GLM_EXL3_PROMPT=1 is
experts.routed(x, pick, None, ex, s, None, R, limit, act_mode=ACT_BF16, prompt=True, prompt_pass, order, rowrot) on a
layer whose shared suh is recorded (exl3_generic.share_suh, GLM's loader). The rows are 2.15's BENCHMARKS §Quality list
up to 2,049 (its 4,096 dropped: group()'s R x 9 slots x 4 B of shared memory, 147,456 B, is over the GB10's and the
5090's opt-in 101,376 B); then pass 128, her order off and rowrot off. Each case's y [P, D] fp32 and xd [P, I] fp16
are tf-exl3-check's reference for TF_GLM_HALVES_KERNEL=k4. The kernel is Mia's (MiaAI-Lab, @MiaAI_lab, recipe patches
0004 / 0009 / 0020, Apache-2.0), ported to this grouping in python 2.15's patch 0024.
"""

from __future__ import annotations

from exl3_check import SLOTS, normal, picks, scratch_bytes

# kind -> (rows, pass, order, rowrot); the order is tf-exl3-check's (cuda_exl3_check.zig's k4_rows)
K4 = {"routed-k4": ((1, 2, 7, 16, 33, 64, 65, 128, 594, 2048, 2049), 64, True, True),
      "routed-k4-p128": ((7, 594, 2048), 128, True, True),
      "routed-k4-noorder": ((65, 2048), 64, False, True),
      "routed-k4-norowrot": ((65, 2048), 64, True, False)}


def k4_cases(run, m, ex, generic, exl3_generic) -> None:
    from exl3_check_cases import f32bits, timed

    torch, dev, E, D = run.torch, run.dev, ex.count, ex.dims
    if not hasattr(generic, "prompt_kernel_ready"):
        raise SystemExit("TF_EXL3_CHECK_K4=1: --src is not python 2.15's tree (no experts.prompt_kernel_ready)")
    ex = exl3_generic.share_suh(ex)
    if not generic.prompt_kernel_ready(ex, SLOTS):
        raise SystemExit("TF_EXL3_CHECK_K4=1: the layer is outside the prompt kernel (prompt_kernel_ready false)")
    m["shared_suh"] = ex.shared_suh is not None
    for kind, (rows, pass_, order, rowrot) in K4.items():
        for R in rows:
            cid = f"{kind}-{R}"
            x, pick = normal(run, cid, R, D, torch.bfloat16), picks(run, cid, R, E)
            s = generic.Scratch(ex, R, SLOTS, device=dev)             # zeros, members -1: as a fresh window finds it
            items, counts, obuf = generic._prompt_buffers(s, E)        # made here so a dry run lists them too
            xi, pi = x.to(dev), pick.to(dev)
            ms = timed(run, lambda: generic.routed(xi, pi, None, ex, s, None, R, m["limit"], act_mode=generic.ACT_BF16,
                                                   prompt=True, prompt_pass=pass_, order=order, rowrot=rowrot))
            P = R * SLOTS
            call = {"R": R, "rows": R, "slots": SLOTS, "act": "bf16", "limit": m["limit"],
                    "limit_f32_bits": f32bits(m["limit"]), "x_dtype": "bf16", "weighted": False, "hot": False,
                    "group": True, "cfg_gu": list(s.cfg_gu), "cfg_d": list(s.cfg_d), "ms": ms,
                    "prompt": {"pass": pass_, "order": order, "rowrot": rowrot,
                               "shx": rowrot and ex.shared_suh is not None,
                               "max_items": generic.prompt_items(R, SLOTS, E, pass_)},
                    "python": "experts.routed(x, pick, None, ex, s, None, R, limit, act_mode=ACT_BF16, prompt=True, "
                              "prompt_pass, order, rowrot)"}
            run.case(cid, kind, call, [run.inp(cid, "x", x), run.inp(cid, "pick", pick)],
                     [run.outp(cid, "y", s.y[:P]), run.outp(cid, "xd", s.xd[:P])],
                     scratch_bytes(xg=s.xg, xu=s.xu, xd=s.xd, y=s.y, ids=s.ids, count=s.count, members=s.members_buf,
                                   items=items, counts=counts, order=obuf))
            del s, xi, pi
        run.mark(f"{kind}: {len(rows)} cases")
