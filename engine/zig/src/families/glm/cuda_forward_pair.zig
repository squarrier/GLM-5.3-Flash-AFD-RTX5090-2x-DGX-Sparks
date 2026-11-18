// Modified for V4 by Scott Quarrier with Helios: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! A paired prompt step's pass (TF_GLM_PROMPT_PAIR): two chunks site by site, each one's exchanges under the other's.
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const fw = @import("cuda_forward.zig");
const pieces = @import("cuda_forward_pieces.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Site = fw.Forward.Site;
const room = fw.Forward.room;
const share = fw.Forward.share;

/// Two consecutive prompt chunks of one sequence (`b` a view of `a`'s at its end), each on `pieced`'s kernels and bits.
pub fn pairBody(a: *const Forward, b: *const Forward, ra: usize, rb: usize) !void {
    const two = [_]*const Forward{ a, b };
    const rows = [_]usize{ ra, rb };
    var cuts: [2][fw.max_pieces][2]usize = undefined;
    var n: [2]usize = undefined;
    for (two, rows, 0..) |f, R, i| {
        if (f.pieces == null or f.prof != null or f.taps != null) return error.BadPair;
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            if (r.segs.len > 0 or !r.b.prefill or R == 0 or R > r.b.rows) return error.BadPair;
            try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        }
        if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        n[i] = fw.ranges(R, f.pieces.?.count, &cuts[i]);
        if (n[i] == 1 and !f.splits(R)) return error.BadPair; // run would take this chunk whole
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    }
    const layers = a.ranks[0].w.layers;
    var li: usize = 0;
    var di: usize = 0;
    for (layers, 0..) |l, j| {
        // b's mixer follows a's on the main stream, so a's caches, KDA state and conv rows of this layer are in
        for (two, rows, 0..) |f, R, i| {
            if (j > 0) try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try exchangeGlue(f, f.pieces.?, j, .mixer, cuts[i][0..n[i]]);
        }
        for (two, rows, 0..) |f, R, i| {
            try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try exchangeGlue(f, f.pieces.?, j, .ffn, cuts[i][0..n[i]]);
        }
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    for (two, rows, 0..) |f, R, i| {
        try awaitShared(f, f.pieces.?, n[i]);
        try f.finish(R);
    }
}

/// A paired site: each piece filled on the main stream, then on the side stream its exchange, glue and (split) sharing.
fn exchangeGlue(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
    var g = f.*; // the glue's launches go on the side stream
    g.stream = p.side;
    for (cut, 0..) |piece, q| {
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
        try p.filled[q].record(f.stream);
        try p.side.wait(p.filled[q]);
        try pieces.partials(f, p, lo, n, split);
        if (split) {
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try g.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
            }
            try pieces.shareGlued(f, p, lo, n, last);
        } else for (f.ranks) |r| try g.glueRows(r, j, site, lo, piece[1], room(r, lo));
        try p.shared[q].record(p.side);
    }
}

/// Before a paired chunk's next site: its last site's pieces glued and shared on the side stream.
fn awaitShared(f: *const Forward, p: *const Pieces, count: usize) !void {
    for (p.shared[0..count]) |e| try f.stream.wait(e);
}

// Helios: V3 layer-major all-post/oldest-take port on current-native kernel/state interfaces.
// Hugh Madden authored the original lane scheduler. TensorFold kernels/state remain Ash Hart's.
pub const LaneStage = enum { prepare, mixer, post, take, finish };
pub const Testing = if (@import("builtin").is_test) struct {
    pub var lane_step: ?*const fn (LaneStage, usize, usize) anyerror!void = null;
    pub var ping_step: ?*const fn (LaneStage, usize, usize) anyerror!void = null;
} else struct {};
fn pingRecorded(stage: LaneStage, j: usize, i: usize) !bool {
    if (comptime @import("builtin").is_test) if (Testing.ping_step) |hook| {
        try hook(stage, j, i);
        return true;
    };
    return false;
}
/// Direct V3 two-credit FIFO ordering on current native shared-window kernels.
/// Each group owns its crossing buffers; states/acceptance stay native per segment.
/// No host draw/commit/retirement until both groups have drained every layer.
pub fn pingPong(two: *const [2]*const Forward, rows: [2]usize) !void {
    const shared = @import("cuda_shared.zig");
    for (two, rows, 0..) |f, R, i| {
        if (try pingRecorded(.prepare, 0, i)) continue;
        if (f.ranks.len != 1 or f.prof != null or f.taps != null) return error.BadPair;
        const r = f.ranks[0];
        const c = &r.w.config;
        if (r.b.prefill or r.segs.len == 0 or R == 0 or R > r.b.rows or r.w.role != .attention) return error.BadPair;
        try shared.begin(r, R, f.stream);
        try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    }
    var li: usize = 0;
    var di: usize = 0;
    const layers = two[0].ranks[0].w.layers;
    for (layers, 0..) |l, j| {
        for (two, rows, 0..) |f, R, i| {
            if (try pingRecorded(.mixer, j, i)) continue;
            const r = f.ranks[0];
            const layer = r.w.layers[j];
            try shared.mixer(f, r, layer, li, di, R, r.b.gathered);
            if (f.prefetch()) |pf| try pf.fork(f.stream, j, .f);
            try f.resolve(R);
            try f.post(r, R);
            try f.pre(r, layer.ffn_hc, layer.post_norm, R);
        }
        const dense = switch (l.ffn) { .dense => true, else => false };
        for (two, rows, 0..) |f, R, i| {
            if (!(try pingRecorded(.post, j, i))) {
                try f.ffn(f.ranks[0], f.ranks[0].w.layers[j], R, null);
                if (f.prefetch()) |pf| try pf.fork(f.stream, j, .n);
            }
            // Native twins share dense.act. Consume it on the main stream before
            // the next body; remote MoE planes retain BOTH-post/FIFO-take overlap.
            if (dense and !(try pingRecorded(.take, j, i))) {
                const r = f.ranks[0];
                try f.fillRows(r, r.w.layers[j], .ffn, 0, R, r.b.gathered);
                try f.resolve(R);
                try f.glueRows(r, j, .ffn, 0, R, r.b.gathered);
            }
        }
        if (!dense) for (two, rows, 0..) |f, R, i| {
            if (try pingRecorded(.take, j, i)) continue;
            const r = f.ranks[0];
            try f.fillRows(r, r.w.layers[j], .ffn, 0, R, r.b.gathered);
            try f.resolve(R);
            try f.glueRows(r, j, .ffn, 0, R, r.b.gathered);
        };
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    for (two, rows, 0..) |f, R, i| {
        if (try pingRecorded(.finish, layers.len, i)) continue;
        if (f.prefetch()) |pf| try pf.join(f.stream);
        try f.finish(R);
    }
}
fn recorded(stage: LaneStage, j: usize, i: usize) !bool {
    if (comptime @import("builtin").is_test) if (Testing.lane_step) |hook| {
        try hook(stage, j, i);
        return true;
    };
    return false;
}
/// Each lane owns crossing/work buffers. All mixers follow prior lane KDA/conv/KV
/// publication on the main stream; all routed posts precede FIFO takes of this layer.
pub fn lanesBody(lanes: []const *const Forward, rows: []const usize) !void {
    if (lanes.len < 2 or lanes.len > 4 or rows.len != lanes.len) return error.BadPair;
    var cuts: [4][fw.max_pieces][2]usize = undefined;
    var n: [4]usize = undefined;
    for (lanes, rows, 0..) |f, R, i| {
        if (try recorded(.prepare, 0, i)) continue;
        if (f.pieces == null or f.prof != null or f.taps != null) return error.BadPair;
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            if (r.segs.len > 0 or !r.b.prefill or R == 0 or R > r.b.rows) return error.BadPair;
            try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        }
        if (f.images) |im| for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        n[i] = fw.ranges(R, f.pieces.?.count, &cuts[i]);
        if (n[i] == 1 and !f.splits(R)) return error.BadPair;
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    }
    const layers = lanes[0].ranks[0].w.layers;
    var li: usize = 0;
    var di: usize = 0;
    for (layers, 0..) |l, j| {
        for (lanes, rows, 0..) |f, R, i| {
            if (try recorded(.mixer, j, i)) continue;
            if (j > 0) try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try exchangeGlue(f, f.pieces.?, j, .mixer, cuts[i][0..n[i]]);
        }
        const dense = switch (l.ffn) { .dense => true, else => false };
        for (lanes, rows, 0..) |f, R, i| {
            if (!(try recorded(.post, j, i))) {
                try awaitShared(f, f.pieces.?, n[i]);
                for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            }
            // exchangeGlue fills on the main stream; only its exchange/glue is
            // on the side stream. Match native pairBody's shared dense lifetime.
            if (dense and !(try recorded(.take, j, i))) try exchangeGlue(f, f.pieces.?, j, .ffn, cuts[i][0..n[i]]);
        }
        if (!dense) for (lanes, 0..) |f, i| {
            if (try recorded(.take, j, i)) continue;
            try exchangeGlue(f, f.pieces.?, j, .ffn, cuts[i][0..n[i]]);
        };
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    for (lanes, rows, 0..) |f, R, i| {
        if (try recorded(.finish, layers.len, i)) continue;
        try awaitShared(f, f.pieces.?, n[i]);
        try f.finish(R);
    }
}
