//! qualification_47b9bf3a: the MTP absorb as each prompt lane's extra layer (TF_AFD_LANES=N with TF_AFD_MTP=1 on a request whose prompt
//! absorbs into the MTP head; evidence/qualification_0f991257/WIRE.md §13.11). A lane group's window then carries count = positions
//! (word 6 = word 5): the MTP layer is its last MoE position, run in the wire's order like the backbone's (lane 0's
//! MTP request, lane 1's, ...), so the absorbs ride the group's exchanges in flight instead of running one by one after
//! it. The group's last lane, when it is the prompt's last chunk (its next tokens one fewer than its rows), keeps
//! count = layers and absorbs after the group as before. Each chunk's launches stay the solo chunk's: the stage and
//! front (cuda_afd_mtp.forwardFront), the exchange at position `layers`, `back`, then the MTP length (setMtpLen on the
//! State, after the group in chunk order with the commits). Lanes 1..N-1 get the MTP residual (mx, which crosses the
//! exchange) and their MTP position (mtp_pos_dev, filled with the lane's MTP length at its open, as pos_dev).
//! The MTP head is Mia's (MiaAI-Lab, 078455ff's MTP layer); the lanes are Hugh Madden's (@dangerm00se, glm53f-afd,
//! MIT); TensorFold, its GLM engine and MCDMA are Ash Hart's (@ashxhart).
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const layer = @import("cuda_layer.zig");
const cc = @import("cuda_compute.zig");
const fwd = @import("cuda_fwd.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const amtp = @import("cuda_afd_mtp.zig");
const pairs = @import("cuda_afd_pairs.zig");

const Fwd = fwd.Fwd;
const T = trace.T;

/// Whether the chunk at `start` of `R` rows (of a prompt of `len` ids) absorbs inside its lane: the node absorbs into
/// the MTP head (Attn.absorb, with the provider's MTP position), the group has 3+ lanes (a pair's group absorbs after
/// it, as the pairs'), and its next tokens are R (the prompt's last chunk has R - 1: it absorbs after the group).
pub fn inLane(x: *const ae.Attn, lanes: usize, len: usize, start: usize, R: usize) bool {
    return x.absorb and x.a.mtp and lanes > 2 and start + R < len;
}

/// A lane's copies for the attention node's provider `a` (the forecast's too): the flight's, and with its `mtp` the
/// MTP layer's (the MTP residual, the lane's MTP position).
pub fn copies(pb: *const bufs.Buffers, st: *const bufs.State, a: *const afd.Afd) pairs.Copies {
    var c = pairs.Copies.init(pb, st, a);
    if (a.mtp) {
        c.mx = pairs.Copies.fresh(pb.mx);
        c.mtp_pos_dev = pairs.Copies.fresh(st.mtp_pos_dev);
    }
    return c;
}

/// The lane's MTP layer to its exchange, over its State view `st` and Buffers `b` (its R next tokens, its final-normed
/// rows as the hidden; at its MTP length): Attn.absorbChunk's stage and front. Gives the options its `back` takes.
pub fn front(f: *Fwd, x: *ae.Attn, st: *const bufs.State, b: *bufs.Buffers, ids: []const u32, start: usize, R: usize) !layer.Opt {
    const next = ids[start + 1 ..][0..R];
    return amtp.forwardFront(f, x.e.w, st, b, next, cc.rows(b.fnormed, 0, R), x.e.chip, &x.e.prompt);
}

/// The lane's MTP layer after its take: the planes summed into its residual, the final norm and the draft head on the
/// last row (the logits only draft: nothing reads them).
pub fn back(f: *Fwd, x: *ae.Attn, b: *const bufs.Buffers, R: usize, planes: fwd.Planes, opt: layer.Opt) !void {
    _ = try amtp.back(f, x.e.w, b, R, planes, true, opt.chip);
}

/// After the group, chunk by chunk with the commits: an in-lane chunk's MTP length on the State (the solo absorb's last
/// launch), else its absorb after the group (Attn.absorbChunk: the prompt's last chunk, or every chunk of a pair's group
/// or of a request that does not absorb).
pub fn settle(f: *Fwd, x: *ae.Attn, absorbed: bool, b: *bufs.Buffers, ids: []const u32, start: usize, R: usize) !void {
    if (!absorbed) return x.absorbChunk(f, b, ids, start, R);
    const st = &x.e.st;
    try cc.setMtpLen(f, st, st.mtp_len + R); // decode._absorb_rows
}

/// Lane k > 0's MTP position at its open (a group launch, as its pos_dev's): the MTP length the lanes before it reach.
pub fn position(f: *Fwd, st: *const bufs.State) !void {
    try cc.fill(f, T.whole(st.mtp_pos_dev), @intCast(st.mtp_len));
}
