//! GLM-5.3-Flash on CUDA in Zig: one TP rank's loader (the MLX checkpoint's rows, O_DIRECT and io_uring) and, from
//! qualification_qualification_44ba554e.5, the forward in python-0.6's launches (the model, the weight views, the buffers, the layer forward, its trace;
//! from B2 the decode step: stage, compute, greedy sampling, commit, and the MTP head; from B3 prompt chunks, the
//! greedy Engine and two ranks in lockstep).

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const split = @import("cuda_split.zig");
pub const index = @import("cuda_index.zig");
pub const read = @import("cuda_read.zig");
pub const memory = @import("cuda_memory.zig");
pub const digest = @import("cuda_digest.zig");
pub const weights = @import("cuda_weights.zig");
pub const synth = @import("cuda_synth.zig");
pub const model = @import("cuda_model.zig");
pub const views = @import("cuda_views.zig");
pub const buffers = @import("cuda_buffers.zig");
pub const trace = @import("cuda_trace.zig");
pub const fwd = @import("cuda_fwd.zig");
pub const qmm = @import("cuda_qmm.zig");
pub const layer = @import("cuda_layer.zig");
pub const moe = @import("cuda_moe.zig");
pub const experts = @import("cuda_experts.zig");
pub const dsa = @import("cuda_dsa.zig");
pub const dsa_kernels = @import("cuda_dsa_kernels.zig");
pub const compute = @import("cuda_compute.zig");
pub const sample = @import("cuda_sample.zig");
pub const nucleus = @import("cuda_nucleus.zig");
pub const nucleus_draw = @import("nucleus_draw.zig");
pub const mtp = @import("cuda_mtp.zig");
pub const tmp = @import("cuda_tmp.zig");
pub const prompt = @import("cuda_prompt.zig");
pub const engine = @import("cuda_engine.zig");
pub const serve = @import("cuda_serve.zig");
pub const grammar_tests_rec = @import("cuda_grammar_tests_rec.zig"); // test-only EXL3 AFD recordings, not a device adapter
pub const serve_rec = @import("cuda_serve_rec.zig"); // qualification_qualification_6c130c7f: recording ranks and trace comparisons for the serve checks
pub const decode = @import("cuda_decode.zig");
pub const ranks = @import("cuda_ranks.zig");
pub const dump = @import("cuda_dump.zig");
pub const kda_kernels = @import("cuda_kernels.zig");
pub const pack = @import("cuda_pack.zig");
pub const pack_ref = @import("cuda_pack_ref.zig");
pub const exl3 = @import("cuda_exl3.zig");
pub const afd_attn = @import("cuda_afd_attn.zig");
pub const afd_load = @import("cuda_afd_load.zig");
pub const afd_expert = @import("cuda_afd_expert.zig");
pub const afd_audit = @import("cuda_afd_audit.zig");
pub const afd_engine = @import("cuda_afd_engine.zig");
pub const afd_pairs = @import("cuda_afd_pairs.zig"); // qualification_0f991257: prompt chunk pairs on the AFD attention node (TF_AFD_PAIRS=1)
pub const afd_pairs_rec = @import("cuda_afd_pairs_rec.zig"); // qualification_0f991257: a paired prefill's record, chunk by chunk (its gates)
pub const afd_lanes = @import("cuda_afd_lanes.zig"); // qualification_9e49592e: prompt chunk lanes 3-4 on the AFD attention node (TF_AFD_LANES=N)
pub const afd_lanes_mtp = @import("cuda_afd_lanes_mtp.zig"); // qualification_47b9bf3a: the MTP absorb as each lane's extra layer
pub const afd_mtp = @import("cuda_afd_mtp.zig"); // qualification_21e21bc0: the MTP head under AFD (TF_AFD_MTP=1)
pub const afd_drafts = @import("cuda_afd_drafts.zig"); // qualification_21e21bc0: qualification_qualification_86ea5170's drafted decode loops on the AFD attention node
pub const afd_users_round = @import("cuda_afd_users_round.zig");
pub const afd_users = @import("cuda_afd_users.zig");
pub const afd_pool = @import("cuda_afd_pool.zig"); // shared_pool: the users' caches from one pool (TF_AFD_POOL_TOKENS)
pub const afd_users_drafts = @import("cuda_afd_users_drafts.zig"); // user_drafts: a lone user's MTP drafts (TF_AFD_USERS_DRAFTS)
pub const afd_users_shared = @import("cuda_afd_users_shared.zig"); // shared_forward: one forward over every row (TF_AFD_USERS_SHARED)
pub const afd_users_bd = @import("cuda_afd_users_bd.zig"); // batched_drafts: batched MTP draft depths
pub const afd_users_dsv = @import("cuda_afd_users_dsv.zig"); // shared_verify: drafted shared verify (TF_AFD_USERS_SHARED_DRAFTS)
pub const afd_users_pp = @import("cuda_afd_users_pp.zig"); // ping_pong: two independent user micro-batches, default off
pub const afd_users_pp_device = @import("cuda_afd_users_pp_device.zig");
pub const afd_serve = @import("cuda_afd_serve.zig"); // qualification_4be15bbf: the served stream's calls over afd_engine.Attn
pub const afd_sampling = @import("cuda_afd_sampling.zig");
pub const halves_kernel = @import("cuda_halves_kernel.zig"); // qualification_f504f9df: TF_GLM_HALVES_KERNEL, agreed by all three roles
pub const afd_graphs = @import("cuda_afd_graphs.zig"); // qualification_f16009a8: CUDA graphs for the AFD decode forwards (TF_AFD_GRAPHS=1)

test {
    _ = config;
    _ = split;
    _ = index;
    _ = read;
    _ = memory;
    _ = digest;
    _ = exl3;
    _ = &digest.fromDevice; // the device half compiles on any host; it runs on a GPU
    _ = &weights.load;
    _ = @import("cuda_load_test.zig");
    _ = @import("cuda_split_test.zig");
    _ = model;
    _ = views;
    _ = buffers;
    _ = trace;
    _ = qmm;
    _ = &fwd.Fwd.launching; // the launch path compiles on any host; it runs on a GPU
    _ = &layer.layerForward;
    _ = @import("cuda_layer_test.zig");
    _ = @import("cuda_afd_attn_test.zig"); // qualification_9051d0c1: the AFD attention role (the seam; the provider)
    _ = @import("cuda_afd_trace_test.zig"); // qualification_9051d0c1, qualification_0fa2bee9: G-trace-A and G-trace-A full (python 2.15 at world 1)
    _ = &afd_attn.Afd.attach; // the device half compiles on any host; it runs on a GPU
    _ = &afd_attn.Afd.detach;
    _ = afd_load; // qualification_9051d0c1: the attention node's world-1 load (its plan, its placement)
    _ = afd_audit; // qualification_0fa2bee9: TF_AFD_CHECK in Zig (the wire audit's hashes and window)
    _ = @import("cuda_afd_engine_test.zig"); // qualification_0fa2bee9: the attention engine at world 1 (begin, check, fail; chunks)
    _ = @import("cuda_afd_pairs_test.zig"); // qualification_0f991257: prompt chunk pairs (the split, the order, the copies, refusals)
    _ = @import("cuda_afd_flight_test.zig"); // qualification_c533375f: the pairs with two requests in flight (TF_AFD_INFLIGHT=2)
    _ = @import("cuda_afd_drafts_test.zig"); // qualification_21e21bc0: MTP drafts on the AFD attention node (slices, the loops, failure)
    _ = @import("cuda_afd_graphs_test.zig"); // qualification_f16009a8: the graphs by key (G-trace, the eager fallback, eviction)
    _ = afd_users_round;
    _ = afd_users_shared; // shared_forward: the row view and the stages
    _ = afd_users_dsv; // shared_verify: the switch, the window rows and view
    _ = afd_users;
    _ = afd_pool;
    _ = afd_sampling;
    _ = @import("cuda_afd_sampling_test.zig");
    _ = afd_serve; // qualification_4be15bbf: its served calls (tf-glm-afd attn serve; G-trace and G-dead-half: zig/tests/cuda/afd_serve_fail.zig)
    _ = &afd_attn.Afd.audit; // the audit's pinned ring compiles on any host; it runs on a GPU
    _ = &afd_pairs.Pairs.attach; // qualification_0f991257: B's copies and B's ring compile on any host too; they run on a GPU
    _ = &afd_pairs.Pairs.audit;
    _ = &afd_pairs.Pairs.detach;
    _ = afd_lanes; // qualification_9e49592e: TF_AFD_LANES and the lanes' turn
    _ = @import("cuda_afd_lanes_test.zig"); // qualification_9e49592e: the lanes (G-trace-lanes, the order, the copies, refusals)
    _ = @import("cuda_afd_lanes_mtp_test.zig"); // qualification_47b9bf3a: the MTP absorb as each lane's extra layer
    _ = &afd_lanes.Lanes.attach; // qualification_9e49592e: lanes 1..N-1's copies and rings compile on any host too; they run on a GPU
    _ = &afd_lanes.Lanes.audit;
    _ = &afd_lanes.Lanes.detach;
    _ = compute;
    _ = sample;
    _ = mtp;
    _ = &compute.forward; // the launch paths compile on any host; they run on a GPU
    _ = &mtp.forward;
    _ = &sample.Kernels.load;
    _ = &sample.prepare;
    _ = @import("cuda_compute_test.zig");
    _ = nucleus_draw; // qualification_45c10bcf: the TP nucleus draw with Mia's union cover
    _ = &nucleus.nucleusRows; // the launch path compiles on any host; it runs on a GPU
    _ = &nucleus.Kernels.load;
    _ = @import("cuda_nucleus_test.zig");
    _ = tmp;
    _ = engine;
    _ = &engine.Engine.prefill; // the launch paths compile on any host; they run on a GPU
    _ = &engine.Engine.mtpDecode;
    _ = &engine.Engine.serialDecode;
    _ = &engine.Engine.autoDecode;
    _ = &engine.Device.attach;
    _ = @import("cuda_prompt_test.zig");
    _ = serve;
    _ = @import("cuda_serve_test.zig"); // qualification_qualification_e91d6b91: a served request's launches are prefill's and serial_decode's
    _ = decode;
    _ = @import("cuda_decode_test.zig"); // qualification_qualification_44ba554e.7: the decode loops with MTP drafts against python-0.6's
    _ = @import("cuda_decode_sampled_test.zig"); // qualification_qualification_44ba554e.7 day 2: sampled drafted == sampled serial (qualification_45c10bcf's keyed draw)
    _ = ranks;
    _ = pack;
    _ = pack_ref;
    _ = afd_expert;
    _ = halves_kernel;
}
