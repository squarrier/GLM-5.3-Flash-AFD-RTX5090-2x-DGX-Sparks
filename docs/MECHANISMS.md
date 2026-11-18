# How the hybrid lane works

TensorFold v1.2.8 supplies the native server, scheduler, model loading, CUDA attention, KV pages, sampling, and MTP. V4 ports the V3 device exchange and adapts those interfaces to one attention role and two expert roles. It does not split attention across the Sparks or place the active KV cache on the wire.

| Stage | What runs | Source entry points |
|---|---|---|
| Route and pack | The RTX 5090 runs attention, computes routes, and packs the live BF16 rows and FP32 route weights. | `cuda_moe.zig`, `cuda_hybrid_v3.zig` |
| Exchange and overlap | Post both expert halves before collecting either response. Local shared-expert compute can run while the remote halves are in flight; the receiver takes results in FIFO order. | `cuda_hybrid_forward.zig`, `cuda_hybrid_v3.zig`, `cuda_forward_pair.zig` |
| Expert return | Each Spark computes its channel half and returns a compact weighted FP32 plane. The attention side combines the two halves and the shared contribution in the specified order. | `cuda_hybrid_expert.zig`, `cuda_g53.zig`, `rank_math.cuh` |
| Keep active history | Pinned workstation RAM holds BF16 latent KV pages. The GPU retains the pooled index keys and gathers the selected rows. Bounded scratch uses a 16-row attention tile; page and verification capacity are priced separately. | `cuda_kv_host.zig`, `cuda_dsa_block.zig`, `cuda_latent.zig`, `cuda_lane_costs.zig` |
| Prefill and decode | Independent prompt workspaces can overlap where the prior layer and same-layer KDA, convolution and KV dependencies allow. Native MTP keeps the selected verification wave; the row-independent Q4 head is evaluated in smaller tiles. | `cuda_prompt_lanes.zig`, `cuda_forward_pair.zig`, `cuda_hybrid_users.zig`, `cuda_hybrid_mtp.zig` |
| Close or retain | Ownership lasts through the device consumer and posted operations. Partial submission or unsettled work retains ownership rather than freeing buffers early. | `cuda_hybrid_v3.zig`, `cuda_host_close.zig`, `roce_close.zig` |

The 5090 keeps 7,496 attached pages in the selected lab configuration; each independent session has a 479,232-token prompt-plus-output budget. This configuration was exercised with eight independent near-capacity histories, including cancellation and page reuse. The public build wrappers are CPU-checked, not a separately qualified portable deployment. The lab’s original post-run lifecycle check failed; later clean closure does not establish restart or unattended recovery.

## Implementation attribution

- Ash Hart and the TensorFold contributors authored the [native engine at v1.2.8](https://github.com/ashhart/TensorFold/tree/09a9eb33176df8c3f56c86fceb9af94c3ac1618f). Russell Marr authored [pinned host KV](https://github.com/ashhart/TensorFold/pull/1795) and Bizuayeu authored the [MTP pooled-key correction](https://github.com/ashhart/TensorFold/pull/2173). These are inherited native features, not V4 inventions.
- The [public V3 source](https://github.com/squarrier/GLM-5.3-Flash-AFD-RTX5090-2x-DGX-Sparks/tree/cd50fe4d4094a4b1e6d9dc93209feac8fc2257ef) supplies the device exchange and the `Layer` descriptor layout reused in `cuda_v3_g53_types.zig`. Hugh Madden authored the expert prompt kernels and schedule; their math also carries Ash Hart's MIT decoder, Mia's rotation/order design and T.J. Purtell's channel-split design. `rank_math.cuh` changes V3's separate add/multiply expression to explicit FP32 fused multiply-add. It is not merely a change in data movement, or a claim of numeric identity to V3.
- T.J. Purtell's [CuteAFD source](https://github.com/tpurtell/cuteafd/tree/b4818a73371788aa3b9dd54023c017646d109f01) provides credited examples of sequencing, bounded assembly and buffer lifetime. The corresponding native Zig paths use the V3/TensorFold device protocol, not the Rust host-staged FP8/BF16 path. `cuda_hybrid_schema.zig` reads TensorFold checkpoint tensor metadata; it does not implement the separate CuteAFD chat argument parser.

See [Credits](../CREDITS.md) and [NOTICE](../NOTICE) for author, source, revision and license details. This architecture description does not establish that any particular mechanism accounts for a measured speedup.
