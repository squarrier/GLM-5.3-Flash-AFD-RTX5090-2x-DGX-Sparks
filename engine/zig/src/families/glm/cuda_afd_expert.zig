//! The AFD expert role (qualification_064ac7c1 lane E): one Spark's TP2 half of every MoE layer's routed experts, and python TF-AFD 2.15's
//! per-forward loop over native RoCE exchange (src@0849f3f glm5_next/cuda/experts_node.py, afd_mcdma.py McdmaExpertLink).
//! - E1 the half: cuda_exl3_split's rank h of 2, each MoE layer's expert group only (python load_half's RankReader(h)),
//!   and check-weights' list of its buffers by python's dataclass paths (qualification_2b1188e0's weights.json names).
//! - E2 the buffers: python ExpertBuffers (decode and prompt scratch, `routed`) and McdmaExpertLink's landing.
//! - E4 one CPU wait a forward for its first request, and python next_forward's header checks.
//! - E5-E9 every layer on the stream: take, the half's experts, the shared slot zeroed, the combine, the reply; then
//!   the forward's one sync (afd_mcdma.py:455-497, experts_node.py:150-172).
//! The split's design is Hugh Madden's (@dangerm00se) and tpurtell's (@wrldsuksgo2mars); TensorFold and MCDMA are Ash
//! Hart's (@ashxhart); the EXL3 checkpoint 078455ff and patch 0006 under native RoCE exchange are Mia's (@MiaAI_lab).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const xs = @import("cuda_exl3_split.zig");
const ex3 = @import("cuda_exl3_experts.zig");
const k3 = @import("cuda_exl3_kernels.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const tri = @import("cuda_triton.zig");
const clk = @import("cuda_afd_clock.zig");
/// qualification_f16009a8: a decode forward's positions as one CUDA graph (TF_AFD_GRAPHS=1), entered from `forwards`
pub const graphs = @import("cuda_afd_graphs.zig");
/// qualification_c2d4ee2e: the halves' side of the prefill pairs (TF_AFD_PAIRS=1), entered from `forwards`
pub const pairs = @import("cuda_afd_halves.zig");
/// qualification_b3fba1da: SEQ's prompt scratch z (TF_GLM_EXL3_SEQ=1): one plane a projection where every position takes it
pub const seqz = @import("cuda_afd_seq.zig");
/// qualification_3e1f1105: TF_GLM_EXL3_PGROUP's counts block (bertholomus's parallel grouping on the 64+-row windows)
pub const pgroup = @import("cuda_afd_pgroup.zig");
/// qualification_8724e161: TF_GLM_EXL3_MMA's work list block (bertholomus's grouped_mma on X1's 64+-row windows)
pub const mma = @import("cuda_afd_mma.zig");
/// qualification_1b9bf461: glm53f-afd's large-M expert kernels on the prompt windows (TF_GLM_HALVES_KERNEL=g53kt3r)
pub const g53 = @import("cuda_g53.zig");
/// K4 on the halves (qualification_e4ad04cd): its block, readiness and launches
pub const afdk4 = @import("cuda_afd_k4.zig");
/// qualification_c863b626: the routed dispatch in one place: g53kt3r, then k4, then X1 (SEQ, PGROUP, MMA) or today's
pub const rt = @import("cuda_afd_route.zig");
const xchg = cuda.xchg;
const proto = xchg.proto;
const Buf = views.Buf;
const T = trace.T;
const Val = trace.Val;

/// The Triton wrappers (glue.select's _topk and glue.combine's _combine are the expert's), for the command's selftest.
pub const Tri = tri.Tri;

/// python ExpertBuffers' `rows` (a decode window's most) and `prefill_rows` (a prompt chunk's)
pub const decode_rows = 8;
pub const prefill_rows = 2048;
/// python's expert nodes: AFD ranks 1 and 2, TP2 halves 0 and 1
pub const halves = 2;

// -- E1: the half (python load_half) ------------------------------------------------------------------------------

/// TP2 half h of the routed experts: the split's plan of rank h, and of it each MoE layer's expert group (trellis,
/// suh, svh of gate, up and down for every expert, and the kernels' pointer and width tables); no other group is read.
pub const Half = struct {
    gpa: std.mem.Allocator,
    plan: xs.Plan,
    h: u32,
    layers: []const u32, // the MoE layers in order (python node.layers): position p is layer layers[p]
    groups: []const u32, // each position's expert group in the plan
    copies: []const split.Copy, // the reads that fill those groups, in the plan's order

    pub fn deinit(x: *Half) void {
        x.gpa.free(x.layers);
        x.gpa.free(x.groups);
        x.gpa.free(x.copies);
        x.plan.deinit();
    }

    /// The groups' bytes on the device (the tables and the alignment between tensors included).
    pub fn bytes(x: *const Half) u64 {
        var n: u64 = 0;
        for (x.groups) |g| n += x.plan.groups.items[g];
        return n;
    }

    /// The checkpoint bytes the half keeps (role_bytes.py's count: the tensors as split.py cuts them).
    pub fn kept(x: *const Half) u64 {
        var n: u64 = 0;
        for (x.copies) |c| n += c.kept();
        return n;
    }

    pub fn experts(x: *const Half, pos: usize) xs.Experts {
        return x.plan.experts[x.layers[pos]].?;
    }

    /// Position `pos`'s experts bound for the kernels, its group at device address `base`.
    pub fn bind(x: *const Half, pos: usize, base: u64) ex3.Layer {
        return x.plan.bind(x.layers[pos], base).?;
    }

    /// Position `pos`'s pointer and width tables (36 E bytes, Plan.tables) for its group at `base`: where they go in
    /// the group, and their bytes into `out`.
    pub fn tables(x: *const Half, pos: usize, base: u64, out: []u8) struct { off: u64, bytes: []u8 } {
        const e = x.experts(pos);
        const t = out[0 .. 36 * @as(usize, e.count)];
        x.plan.tables(x.layers[pos], base, t);
        return .{ .off = e.tables.off, .bytes = t };
    }
};

/// python load_half(model_dir, h, layers) over the backbone's MoE layers that `c.run` loads (the MTP layer is not
/// served: python 2.15 runs AFD with MTP 0), as rank h of 2 by split.py's rules.
pub fn half(gpa: std.mem.Allocator, c: *const cfg.Config, names: *const split.Names, h: u32) !Half {
    return halfWith(gpa, c, names, h, false);
}

/// `half`, and with `mtp` (qualification_21e21bc0, TF_AFD_MTP=1) the MTP layer's routed experts too: the split's experts[c.layers], one
/// more position after the backbone's, on the same expert path (Ash's rule: the same expert weights).
pub fn halfWith(gpa: std.mem.Allocator, c: *const cfg.Config, names: *const split.Names, h: u32, mtp: bool) !Half {
    if (h >= halves) return error.BadHalf;
    var p = try xs.plan(gpa, c, names, h, halves);
    errdefer p.deinit();
    var ls: std.ArrayList(u32) = .empty;
    defer ls.deinit(gpa);
    var gs: std.ArrayList(u32) = .empty;
    defer gs.deinit(gpa);
    for (0..@min(c.run, c.layers) + @intFromBool(mtp)) |k| {
        const i = if (k < @min(c.run, c.layers)) k else c.layers; // the MTP layer last
        const e = p.experts[i] orelse if (i == c.layers) return error.NoMtpLayer else continue;
        for ([_]xs.Proj{ e.gate, e.up, e.down }) |pr| {
            if (pr.trellis.g != e.tables.g or pr.suh.g != e.tables.g or pr.svh.g != e.tables.g) return error.SplitGroups;
        }
        try ls.append(gpa, @intCast(i));
        try gs.append(gpa, e.tables.g);
    }
    if (ls.items.len == @intFromBool(mtp)) return error.NoMoeLayers;
    var cs: std.ArrayList(split.Copy) = .empty;
    defer cs.deinit(gpa);
    for (p.copies.items) |cp| if (std.mem.indexOfScalar(u32, gs.items, cp.g) != null) try cs.append(gpa, cp);
    const layers = try gpa.dupe(u32, ls.items);
    errdefer gpa.free(layers);
    const groups = try gpa.dupe(u32, gs.items);
    errdefer gpa.free(groups);
    return .{ .gpa = gpa, .plan = p, .h = h, .layers = layers, .groups = groups, .copies = try gpa.dupe(split.Copy, cs.items) };
}

/// Exl3RoutedExperts' fields in qualification_2b1188e0's weights.json order: the address tables (no digest: they differ per process),
/// the widths, the scales, then keep[j], each expert's trellis (gate's, then up's, then down's).
pub const Field = enum { gate_ptr, up_ptr, down_ptr, gate_k2, up_k2, down_k2, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d, keep };

/// One buffer of a layer: torch's dtype name and shape, and its bytes in the layer's group.
pub const Entry = struct { layer: u32, field: Field, j: u32 = 0, dtype: []const u8, shape: [3]u64, dims: u8, off: u64, bytes: u64 };

/// Every buffer of position `pos`, in python's walk order (qualification_2b1188e0 weights.json: 3 + 3 + 6 + 3 E a layer).
pub fn entries(gpa: std.mem.Allocator, x: *const Half, pos: usize, out: *std.ArrayList(Entry)) !void {
    const i = x.layers[pos];
    const e = x.experts(pos);
    const E: u64 = e.count;
    for ([_]Field{ .gate_ptr, .up_ptr, .down_ptr, .gate_k2, .up_k2, .down_k2 }, 0..) |f, j| {
        const w: u64 = if (j < 3) 8 else 4;
        const off = e.tables.off + if (j < 3) 8 * E * j else 24 * E + 4 * E * (j - 3);
        try out.append(gpa, .{ .layer = i, .field = f, .dtype = if (j < 3) "int64" else "int32", .shape = .{ E, 0, 0 }, .dims = 1, .off = off, .bytes = w * E });
    }
    const scales = [_]struct { Field, split.Ref, u32 }{ .{ .suh_g, e.gate.suh, e.gate.k }, .{ .suh_u, e.up.suh, e.up.k }, .{ .svh_g, e.gate.svh, e.gate.n }, .{ .svh_u, e.up.svh, e.up.n }, .{ .suh_d, e.down.suh, e.down.k }, .{ .svh_d, e.down.svh, e.down.n } };
    for (scales) |s| try out.append(gpa, .{ .layer = i, .field = s[0], .dtype = "float16", .shape = .{ E, s[2], 0 }, .dims = 2, .off = s[1].off, .bytes = 2 * E * s[2] });
    var j: u32 = 0;
    for ([_]xs.Proj{ e.gate, e.up, e.down }) |pr| for (0..e.count) |k| {
        const w: u64 = 8 * @as(u64, pr.k2s[k]);
        try out.append(gpa, .{ .layer = i, .field = .keep, .j = j, .dtype = "int16", .shape = .{ pr.k / 16, pr.n / 16, w }, .dims = 3, .off = pr.trellis.off + pr.offsets[k], .bytes = @as(u64, pr.k / 16) * (pr.n / 16) * w * 2 });
        j += 1;
    };
}

/// The entry's python path, as qualification_2b1188e0's weights.json writes it.
pub fn path(e: Entry, buf: []u8) []const u8 {
    if (e.field == .keep) return std.fmt.bufPrint(buf, "w.layers[{d}].moe.experts.keep[{d}]", .{ e.layer, e.j }) catch unreachable;
    return std.fmt.bufPrint(buf, "w.layers[{d}].moe.experts.{t}", .{ e.layer, e.field }) catch unreachable;
}

pub fn isTable(e: Entry) bool {
    return e.field == .gate_ptr or e.field == .up_ptr or e.field == .down_ptr;
}

/// sha256 of an entry's bytes; none for an address table (its addresses are the process's).
pub fn digest(e: Entry, data: []const u8) !?[32]u8 {
    if (isTable(e)) return null;
    if (data.len != e.bytes) return error.EntryBytes;
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(data, &sum, .{});
    return sum;
}

/// One JSON line an entry, qualification_2b1188e0's table entry: path, shape, dtype, sha256 of `data` (its own bytes; null for an
/// address table, or for a list without bytes), bytes.
pub fn writeEntry(w: *std.Io.Writer, e: Entry, data: ?[]const u8) !void {
    return writeDigest(w, e, if (data) |d| try digest(e, d) else null);
}

/// writeEntry with the digest given (`digest`'s, or null).
pub fn writeDigest(w: *std.Io.Writer, e: Entry, sum: ?[32]u8) !void {
    var buf: [96]u8 = undefined;
    try w.print("{{\"path\": \"{s}\", \"shape\": [", .{path(e, &buf)});
    for (e.shape[0..e.dims], 0..) |d, k| try w.print("{s}{d}", .{ if (k > 0) ", " else "", d });
    try w.print("], \"dtype\": \"{s}\", \"sha256\": ", .{e.dtype});
    if (sum) |s| try w.print("\"{s}\"", .{&std.fmt.bytesToHex(s, .lower)}) else try w.writeAll("null");
    try w.print("{s}, \"bytes\": {d}}}\n", .{ if (isTable(e)) ", \"address_table\": true" else "", e.bytes });
}

// -- E2: buffers (python ExpertBuffers, McdmaExpertLink.landing) -----------------------------------------------------

/// cuda/exl3/experts.Scratch for `rows` rows of `slots` slots, its tile settings default_config's (python 2.15's `order`,
/// its prompt switches' list, is X1Bufs': placed only with TF_GLM_EXL3_X1=1).
pub const Scratch = struct {
    xg: Buf,
    xu: Buf,
    xd: Buf,
    z: Buf,
    y: Buf,
    ids: Buf,
    count: Buf,
    members_buf: Buf,
    rows: usize,
    slots: usize,
    gu: ex3.Tile,
    dn: ex3.Tile,
    /// qualification_3e1f1105: TF_GLM_EXL3_PGROUP's counts (int32 [E], cuda_afd_pgroup.zig's block); an address, never placed here; 0: off
    pg_counts: u64 = 0,
    /// qualification_8724e161: TF_GLM_EXL3_MMA's work list (cuda_afd_mma.zig's block); an address, never placed here; 0: off
    mma_work: u64 = 0,

    pub fn init(rows: usize, slots: usize, dims: usize, width: usize, experts: usize) !Scratch {
        const gu = try ex3.defaultConfig(@intCast(dims), @intCast(width), true);
        const dn = try ex3.defaultConfig(@intCast(width), @intCast(dims), false);
        const P = rows * slots;
        const maxu = @min(P, experts);
        return .{
            .xg = Buf.of(.f16, &.{ P, dims }), .xu = Buf.of(.f16, &.{ P, dims }), .xd = Buf.of(.f16, &.{ P, width }),
            .z = Buf.of(.f32, &.{@max(2 * gu.sk * width, dn.sk * dims) * P}), .y = Buf.of(.f32, &.{ P, dims }),
            .ids = Buf.of(.i32, &.{maxu}), .count = Buf.of(.i32, &.{1}), .members_buf = Buf.of(.i32, &.{maxu * rows}),
            .rows = rows, .slots = slots, .gu = gu, .dn = dn,
        };
    }

    /// experts.Scratch as qualification_qualification_44ba554e-exl3's routed takes it (python's buffers at their device addresses).
    pub fn bound(s: *const Scratch, experts: u32) ex3.Scratch {
        return .{ .xg = s.xg.ptr, .xu = s.xu.ptr, .xd = s.xd.ptr, .z = s.z.ptr, .y = s.y.ptr, .ids = s.ids.ptr, .count = s.count.ptr, .members = s.members_buf.ptr, .rows = @intCast(s.rows), .slots = @intCast(s.slots), .experts = experts, .cfg_gu = s.gu, .cfg_d = s.dn, .counts = s.pg_counts, .work = s.mma_work };
    }
};

/// python ExpertBuffers field for field (normed, pick and wts are the wire slots python allocates; the landing's views
/// replace them in every compute), at the half's shapes: D, slots = top_k + 1, the width a half (moe_width / 2).
pub const Bufs = struct {
    normed: Buf,
    pick: Buf,
    wts: Buf,
    routed: Buf,
    decode: Scratch,
    pref: Scratch,

    pub fn init(d: usize, slots: usize, width: usize, experts: usize, rows: usize, top: usize) !Bufs {
        if (rows < 1 or rows > top) return error.Rows;
        return .{ .normed = Buf.of(.bf16, &.{ top, d }), .pick = Buf.of(.i32, &.{ top, slots }), .wts = Buf.of(.f32, &.{ top, slots }), .routed = Buf.of(.f32, &.{ top, d }), .decode = try Scratch.init(rows, slots, d, width, experts), .pref = try Scratch.init(top, slots, d, width, experts) };
    }

    /// python ExpertBuffers.scratch(R): the decode scratch up to its rows, else the prompt's.
    pub fn scratch(b: *const Bufs, R: usize) *const Scratch {
        return if (R <= b.decode.rows) &b.decode else &b.pref;
    }
};

/// McdmaExpertLink's landing for the widest request. The take copies a message whole, header slot first, so ours is
/// python's landing (the bytes past the header, which python's copy skips) and the 256 bytes before it.
pub const Landing = struct {
    buf: Buf, // u8 [wire bytes of req(top)]

    pub fn init(l: proto.Layout, top: usize) Landing {
        return .{ .buf = Buf.of(.u8, &.{proto.wireBytes(l.req(top))}) };
    }

    /// python's `link.landing`: u8 [req(top) - HDR_BYTES], HDR_BYTES into ours.
    pub fn python(x: Landing, l: proto.Layout, top: usize) Buf {
        var b = Buf.of(.u8, &.{l.req(top) - proto.HDR_BYTES});
        b.ptr = x.buf.ptr + proto.HDR_BYTES;
        return b;
    }
};

/// The node's device block: the buffers then the landing (cuda_buffers.Arena's rule); returns its bytes.
pub fn place(b: *Bufs, landing: *Landing, base: u64) !usize {
    var a: bufs.Arena = .{ .base = base };
    try views.walk(b, views.Path.init("b"), &a, bufs.Arena.visit);
    try views.walk(landing, views.Path.init("link"), &a, bufs.Arena.visit);
    return a.used;
}

/// X1 (qualification_ae31a4aa, TF_GLM_EXL3_X1=1): python 2.15's buffers that serve only its prompt switches, in a block of their own so
/// that with the switch off nothing is placed and every address is today's: each scratch's order list (python
/// Scratch.order, int32 [order_slots(rows, slots, E)]) and each position's shared suh (python's `shared_suh` clone,
/// fp16 [D], written only where the position's gate and up suh are one vector).
pub const X1Bufs = struct {
    decode_order: Buf,
    pref_order: Buf,
    shared_suh: Buf, // fp16 [positions, D]

    pub fn init(b: *const Bufs, experts: usize, positions: usize) X1Bufs {
        return .{
            .decode_order = Buf.of(.i32, &.{ex3.orderSlots(@intCast(b.decode.rows), @intCast(b.decode.slots), @intCast(experts))}),
            .pref_order = Buf.of(.i32, &.{ex3.orderSlots(@intCast(b.pref.rows), @intCast(b.pref.slots), @intCast(experts))}),
            .shared_suh = Buf.of(.f16, &.{ positions, b.normed.shape[1] }),
        };
    }

    /// Scratch `s`'s order list (python's s.order).
    pub fn order(x: *const X1Bufs, b: *const Bufs, s: *const Scratch) Buf {
        return if (s == &b.decode) x.decode_order else x.pref_order;
    }

    /// Position `pos`'s shared suh row.
    pub fn suh(x: *const X1Bufs, pos: usize) u64 {
        return x.shared_suh.ptr + pos * x.shared_suh.shape[1] * 2;
    }
};

/// X1's block (an allocation of its own): its bytes, the buffers' addresses from `base`.
pub fn placeX1(x: *X1Bufs, base: u64) !usize {
    var a: bufs.Arena = .{ .base = base };
    try views.walk(x, views.Path.init("x1"), &a, bufs.Arena.visit);
    return a.used;
}

/// python's `shared_suh` test (cuda/exl3/experts.py:93-102, Mia's 0009) on host copies of a layer's gate and up suh,
/// fp16 [E, D] read as int16: every expert's gate row equal to the first, and up equal to gate, bit for bit (-0 is not
/// +0).
pub fn sharedSuh(g: []const u16, u: []const u16, D: usize) bool {
    if (g.len != u.len or D == 0 or g.len == 0 or g.len % D != 0) return false;
    if (!std.mem.eql(u16, g, u)) return false;
    var e: usize = D;
    while (e < g.len) : (e += D) if (!std.mem.eql(u16, g[e..][0..D], g[0..D])) return false;
    return true;
}

/// A request's sections where the take landed them (proto.Layout's offsets from the landing): normed bf16 [R, D] at
/// 256, pick int32 [R, S] at 256 + 2 D R, wts fp32 [R, S] after its pad; and the bytes the take copies.
pub const Views = struct { normed: u64, pick: u64, wts: u64, bytes: usize };

pub fn landed(l: proto.Layout, landing: u64, rows: usize) Views {
    const o = l.offsets(rows);
    return .{ .normed = landing + o.normed, .pick = landing + o.pick, .wts = landing + o.wts, .bytes = proto.wireBytes(o.end) };
}

// -- E4: a forward's first header (afd_mcdma.py next_forward) -------------------------------------------------------

/// A forward as its first request opens it: R for every layer, the mode, and its MoE positions [first, first + count);
/// `slot` a prompt chunk pair's window (qualification_c2d4ee2e, TF_AFD_PAIRS=1: 1 for B), 0 for every other forward.
pub const Forward = struct { fseq: i32, rows: usize, mode: proto.Mode, first: usize, count: usize, slot: u32 = 0 };

pub const Breach = error{ WrongLayer, WrongLayerCount, WrongRows, WrongSlice, WrongMode, WrongPair, WrongPartner, WrongPairEnd };

/// python next_forward's checks (afd_mcdma.py:463-473, 490-491) on a forward's first header; `count_word` is its word
/// 6 (a slice's count). The magic is Node.waitRequest's; the Zig wire has no length word, so the request's bytes follow
/// from its rows. Pairs and lanes are not agreed here, so their modes are refused as python refuses them then.
pub fn first(h: proto.Header, count_word: i32, layers: usize, max_rows: usize) Breach!Forward {
    const slice = h.mode == .slice;
    if (h.mode != .eager and h.mode != .chunk and !slice) return error.WrongMode;
    if (h.layers < 0 or @as(usize, @intCast(h.layers)) != layers) return error.WrongLayerCount;
    if (h.rows < 1 or @as(usize, @intCast(h.rows)) > max_rows) return error.WrongRows;
    if (!slice and h.layer != 0) return error.WrongLayer;
    const a: i64 = h.layer;
    const n: i64 = if (slice) count_word else @intCast(layers);
    if (a < 0 or n < 1 or a + n > layers) return error.WrongSlice;
    return .{ .fseq = h.fseq, .rows = @intCast(h.rows), .mode = h.mode, .first = @intCast(a), .count = @intCast(n) };
}

/// Word 6 of request `seq`'s header where it landed (a slice's count; zero otherwise).
pub fn countWord(n: *const xchg.Node, seq: u32) i32 {
    return std.mem.readInt(i32, n.received(n.src(0), seq, proto.HDR_BYTES)[24..28], .little);
}

// -- E5-E9: the per-forward loop (python forwards / _window) -------------------------------------------------------

/// How the loop ends: a stop frame (exit 0), or the process exits 3 (the attention lost, a breach, a take that gave
/// up, a --fault exit) or 1 (an error of its own).
pub const End = union(enum) {
    stopped,
    lost: []const u8,
    breach: Breach,
    failed: xchg.Failure,
    die, // --fault exit: the process ends at once, closing nothing (tf-xchg's _exit 3)
    err: anyerror,

    pub fn code(e: End) u8 {
        return switch (e) {
            .stopped => 0,
            .err => 1,
            else => cuda.watch.lost_exit_code,
        };
    }
};

pub const Stats = struct { turns: u64 = 0, forwards: u64 = 0, layers: u64 = 0, rows: u64 = 0 };

/// The lab's injected faults (tf-xchg's --fault; qualification_955b7191's M3 dead and silent experts): at request `at` (counted from 1
/// as the attention sends them) the expert stops answering, its process, link and heartbeat alive (`silent`), or dies
/// once that request has landed (`exit`); every request before it is answered.
pub const Fault = struct { what: enum { silent, exit }, at: u32 };

pub const Opts = struct { layers: usize, max_rows: usize, fault: ?Fault = null, pairs: bool = false, lanes: u8 = 0 };

/// python forwards over the expert's end of the wire. Per forward one CPU turn (Node.waitRequest of its first
/// request, then python's header checks), then each of its MoE positions' take, compute and post on `e`'s stream, then
/// `e.sync`: the forward's one wait (it returns error.AttentionLost when the attention goes meanwhile). R is the first
/// header's for every layer. Requests are counted in seq order from 1, as the attention sends them.
pub fn forwards(e: anytype, n: *xchg.Node, o: Opts, st: *Stats) End {
    var seq: u32 = 1;
    while (true) {
        const h = switch (n.waitRequest(seq)) {
            .request => |h| h,
            .stopped => return .stopped,
            .lost => |why| return .{ .lost = why },
        };
        st.turns += 1;
        if (o.pairs and h.mode == .pair) { // qualification_c2d4ee2e: TF_AFD_PAIRS=1's two windows, in the wire's order (cuda_afd_halves.zig)
            if (pairs.pair(e, n, o, st, &seq, h)) |end| return end;
            continue;
        }
        if (o.lanes > 0 and h.mode == .lanes) { // qualification_9e49592e: TF_AFD_LANES=N's 3..N windows, in lane order (cuda_afd_halves.zig)
            if (pairs.lanes(e, n, o, st, &seq, h)) |end| return end;
            continue;
        }
        const f = first(h, countWord(n, seq), o.layers, o.max_rows) catch |err| return .{ .breach = err };
        // a fault in this forward: its requests before it only
        const run = if (o.fault) |x| @min(f.count, @as(usize, x.at -% seq)) else f.count;
        graphs.layers(e, n, f, run) catch |err| return .{ .err = err }; // qualification_f16009a8: one graph with TF_AFD_GRAPHS=1
        e.sync(n) catch |err| return if (err == error.AttentionLost) .{ .lost = whyText(lost(n) orelse .dead) } else .{ .err = err };
        if (n.failure()) |x| return .{ .failed = x }; // a take gave up or saw a breach: nothing was posted after it
        st.layers += run;
        st.rows += f.rows * run;
        if (run < f.count) return fault(n, o.fault.?, seq +% @as(u32, @intCast(run)));
        st.forwards += 1;
        seq +%= @intCast(f.count); // fn_xchg's sequence wraps as a u32, as lab.expertLoop counts it
    }
}

/// A --fault at request `req`: `exit` once it has landed and the replies before it are out (the attention fails on it:
/// dead), `silent` answering nothing until the attention stops it or is lost (it fails on it: silent).
pub fn fault(n: *xchg.Node, x: Fault, req: u32) End {
    if (x.what == .exit) return switch (n.waitRequest(req)) {
        .request => carried(n),
        .stopped => .stopped,
        .lost => |why| .{ .lost = why },
    };
    return switch (n.idle()) {
        .lost => |why| .{ .lost = why },
        else => .stopped,
    };
}

/// `exit`'s end once this node's proxy has carried every reply the loop posted (its count at the doorbell): with two or
/// more in flight (qualification_c533375f, qualification_9e49592e) the faulty request can land before the reply ahead of it is out. Bounded by the give-up.
fn carried(n: *const xchg.Node) End {
    const p = n.proxies[0] orelse return .die;
    const t0 = n.nowNs();
    while (@as(u32, @truncate(p.ops())) != n.doorbell() and p.failure() == null and n.nowNs() -% t0 < n.cfg.give_up_ns) {
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
        _ = std.c.nanosleep(&ts, null);
    }
    return .die;
}

const rdhup: i16 = 0x2000; // POLLRDHUP, as node.zig polls the control link

/// The attention lost while a forward runs on the GPU: this node's proxy failed, its heartbeat could not be sent, or
/// its control link closed or erred (nothing is read: a stop frame stays for the next waitRequest).
pub fn lost(n: *const xchg.Node) ?proto.Why {
    if (n.proxies[0]) |p| if (p.failure() != null) return .proxy;
    if (n.beat_lost.load(.acquire)) return .beats;
    const l = n.links[0] orelse return .dead;
    var fds = [_]std.posix.pollfd{.{ .fd = l.fd, .events = rdhup, .revents = 0 }};
    const got = std.posix.poll(&fds, 0) catch return .dead;
    if (got != 0 and fds[0].revents & (rdhup | std.posix.POLL.HUP | std.posix.POLL.ERR | std.posix.POLL.NVAL) != 0) return .dead;
    return null;
}

pub fn whyText(w: proto.Why) []const u8 {
    return switch (w) {
        .proxy => "this node's proxy failed",
        .beats => "the heartbeat could not be sent",
        else => "the attention closed the control link (it exited, or the link is down)",
    };
}

/// The log's one line (PLAN §4 item 5): how the loop ended, Node.failure()'s words when the exchange failed.
pub fn describe(e: End, h: u32, st: Stats, buf: []u8) []const u8 {
    return switch (e) {
        .stopped => std.fmt.bufPrint(buf, "afd expert {d}: stopped by the attention after {d} forwards, {d} layers", .{ h, st.forwards, st.layers }),
        .lost => |why| std.fmt.bufPrint(buf, "afd expert {d}: lost the attention ({s}) after {d} forwards, {d} layers", .{ h, why, st.forwards, st.layers }),
        .breach => |b| std.fmt.bufPrint(buf, "afd expert {d}: protocol breach ({t}) after {d} forwards, {d} layers", .{ h, b, st.forwards, st.layers }),
        .failed => |f| std.fmt.bufPrint(buf, "afd: failed {t} peer {d} seq {d} gave_up {d} breach {d} (expert {d} after {d} forwards, {d} layers)", .{ f.why, f.peer, f.seq, f.gave_up, f.breach, h, st.forwards, st.layers }),
        .die => std.fmt.bufPrint(buf, "afd expert {d}: --fault exit at request {d}: exiting 3 at once after {d} forwards, {d} layers", .{ h, st.layers + 1, st.forwards, st.layers }),
        .err => |x| std.fmt.bufPrint(buf, "afd expert {d}: {t} after {d} forwards, {d} layers", .{ h, x, st.forwards, st.layers }),
    } catch "afd expert: (line too long)";
}

// -- E6-E8: the half's experts over a landed request (python ExpertNode.compute) -------------------------------------

fn at(ptr: u64, dtype: views.DType, shape: []const usize) T {
    var b = Buf.of(dtype, shape);
    b.ptr = ptr;
    return T.whole(b);
}

fn int(x: usize) Val {
    return .{ .int = @intCast(x) };
}

fn whole(b: Buf) Val {
    return .{ .t = T.whole(b) };
}

/// python 2.15's exl3_experts.grouped switches as its decode windows pass them (nfirst, order, max_items, prefetch,
/// xrow): all off. A prompt window turns them on (exl3_generic.switches at 64 rows and more): recordX1.
const switches_off = [_]Val{ .{ .bool = false }, .none, .{ .int = 0 }, .{ .bool = false }, .{ .bool = false } };

/// python-0.6's 22 grouped arguments, then 2.15's five switches.
fn grouped(a: [22]Val) [27]Val {
    return a ++ switches_off;
}

/// X1's buffers as a record names them (python's): the window's scratch order list, the layer's shared suh (null: none).
pub const X1Rec = struct { order: Buf, shared_suh: ?Buf };

/// python ExpertNode.compute(layer, R, x) recorded (glm-forward-trace/1): exl3_generic.routed's six extension calls on
/// the landing's views and the R-row scratch (python 2.15's 27-argument grouped, its switches off), the shared slot's
/// zero_ (a torch op: no event), and glue.combine of the scratch's y with the landed wts into `routed`.
pub fn record(t: *trace.Trace, e: *const views.Exl3Experts, b: *const Bufs, v: Views, R: usize, limit: f32) !void {
    return recordX1(t, e, b, v, R, limit, .{}, null);
}

/// `record` with X1 (TF_GLM_EXL3_X1): python 2.15's routed with its prompt switches (cuda/exl3/experts.py:296-332) at
/// 64+ rows where prompt_ready holds: the items' order (exl3_experts.order, unless NFIRST puts its slots past 65,535),
/// one rotation a row (exl3_experts.rot_rows) where the layer has a shared suh, and the grouped calls' five switch
/// arguments as python passes them (rowrot on gate/up only). Elsewhere, or with `x1` null: `record`'s calls.
pub fn recordX1(t: *trace.Trace, e: *const views.Exl3Experts, b: *const Bufs, v: Views, R: usize, limit: f32, p: ex3.Prompt, x1: ?X1Rec) !void {
    const s = b.scratch(R);
    if (R == 0 or R > s.rows) return error.Rows;
    const slots = s.slots;
    const D = e.dims;
    const I = e.width;
    const E = e.count;
    const P = R * slots;
    const maxu = @min(P, E);
    const x = at(v.normed, .bf16, &.{ R, D });
    const pick: Val = .{ .t = at(v.pick, .i32, &.{ R, slots }) };
    const ids: Val = .{ .t = T.of(s.ids, 0, &.{maxu}) };
    const members: Val = .{ .t = T.of(s.members_buf, 0, &.{ maxu, R }) };
    const ready = x1 != null and R >= ex3.prompt_rows and ex3.promptTiles(@enumFromInt(e.cb), e.k2_gu, e.k2_d, s.gu, s.dn);
    const xrow = ready and p.xrow and x1.?.shared_suh != null;
    const on = ready and (p.nfirst or p.pa or p.order or xrow);
    const nfirst = on and p.nfirst;
    const pa = on and p.pa;
    try t.call(.ext, "exl3_experts.group", &.{ pick, ids, whole(s.count), members, int(R), int(slots), int(E) });
    var cap: usize = 0;
    var ord: Val = .none;
    if (on and p.order) {
        cap = ex3.orderSlots(@intCast(R), @intCast(slots), @intCast(E));
        if (!(nfirst and cap > 65535)) {
            ord = whole(x1.?.order);
            try t.call(.ext, "exl3_experts.order", &.{ ids, whole(s.count), members, ord, int(cap), int(E) });
        }
    }
    if (xrow)
        try t.call(.ext, "exl3_experts.rot_rows", &.{ .{ .t = x }, int(x.stride[0]), whole(x1.?.shared_suh.?), whole(s.xg), int(R), int(D) })
    else
        try t.call(.ext, "exl3_experts.rot_in", &.{ .{ .t = x }, int(x.stride[0]), pick, whole(e.suh_g), whole(e.suh_u), whole(s.xg), whole(s.xu), int(R), int(D), int(slots), int(E) });
    const gu = [22]Val{
        whole(s.xg), if (xrow) whole(s.xg) else whole(s.xu), whole(e.gate_ptr), whole(e.up_ptr), whole(e.gate_k2), whole(e.up_k2), ids, whole(s.count), members, whole(s.z),
        int(2), int(D), int(I), int(P), int(s.gu.sk), int(slots), int(e.cb), int(s.gu.nt), int(s.gu.warps), int(s.gu.pf), int(e.k2_gu[0]), int(e.k2_gu[1]),
    };
    try t.call(.ext, "exl3_experts.grouped", &(gu ++ [_]Val{ .{ .bool = nfirst }, ord, int(cap), .{ .bool = pa }, .{ .bool = xrow } }));
    try t.call(.ext, "exl3_experts.gateup_epilogue", &.{ whole(s.z), pick, whole(e.svh_g), whole(e.svh_u), whole(e.suh_d), whole(s.xd), int(R), int(P), int(I), int(s.gu.sk), int(slots), int(E), Val.float(limit), int(0) }); // ACT_BF16
    const dn = [22]Val{
        whole(s.xd), whole(s.xd), whole(e.down_ptr), whole(e.down_ptr), whole(e.down_k2), whole(e.down_k2), ids, whole(s.count), members, whole(s.z),
        int(1), int(I), int(D), int(P), int(s.dn.sk), int(slots), int(e.cb), int(s.dn.nt), int(s.dn.warps), int(s.dn.pf), int(e.k2_d[0]), int(e.k2_d[1]),
    };
    try t.call(.ext, "exl3_experts.grouped", &(dn ++ [_]Val{ .{ .bool = nfirst }, ord, int(cap), .{ .bool = pa }, .{ .bool = false } }));
    try t.call(.ext, "exl3_experts.down_epilogue", &.{ whole(s.z), pick, whole(e.svh_d), whole(s.y), int(R), int(P), int(D), int(s.dn.sk), int(slots), int(E) });
    const tr: tri.Tri = .{ .set = null, .rec = &t.tri_rec };
    try tr.combine(s.y.ptr, true, v.wts, b.routed.ptr, R, D, slots);
    try t.flush();
}

/// exl3_generic.routed's launches (qualification_qualification_44ba554e-exl3's experts.routed: GLM's bf16 SwiGLU, no weights, y per slot) over the
/// landing's views into scratch `s`: the call cuda_moe.exl3Launches makes (A2), on the expert's own buffers.
pub fn launches(ls: *k3.Launches, v: Views, l: ex3.Layer, s: *const Scratch, R: usize, limit: f32, sh: ex3.Shared) !void {
    return launchesSched(ls, v, l, s, R, limit, sh, .{});
}

/// `launches` with Mia's 0128 switches (TF_GLM_EXL3_L2D, TF_GLM_EXL3_PDL; cuda_moe.exl3LaunchesSched's): off, the same.
pub fn launchesSched(ls: *k3.Launches, v: Views, l: ex3.Layer, s: *const Scratch, R: usize, limit: f32, sh: ex3.Shared, sd: ex3.Sched) !void {
    if (R == 0 or R > s.rows) return error.Rows;
    try ex3.routedSched(ls, v.normed, l.dims, true, v.pick, null, l, s.bound(l.count), 0, @intCast(R), limit, .bf16, true, sh, sd);
}

/// `launchesSched` with X1 (TF_GLM_EXL3_X1, python 2.15's prompt switches on windows of 64+ rows; ex3.routedX1): `ord`
/// the scratch's order list, `shared_suh` the position's (null: the layer has none, rowrot stays off for it).
pub fn launchesX1(ls: *k3.Launches, v: Views, l: ex3.Layer, s: *const Scratch, R: usize, limit: f32, sh: ex3.Shared, sd: ex3.Sched, p: ex3.Prompt, ord: u64, shared_suh: ?u64) !void {
    if (R == 0 or R > s.rows) return error.Rows;
    try ex3.routedX1(ls, v.normed, l.dims, true, v.pick, null, l, s.bound(l.count), 0, @intCast(R), limit, .bf16, true, sh, sd, p, ord, shared_suh);
}

/// The shared slot of y [rows, slots, D] fp32 for R rows as one 2-D memset takes it (python ey(R)[:, top_k].zero_()):
/// the first row's slot, the row pitch in bytes, the width in 32-bit words, the height.
pub const Slot2D = struct { dst: u64, pitch: usize, width: usize, height: usize };

/// qualification_1b9bf461: g53's large buffers, python's Scratch.big over the prompt scratch (ExpertBuffers.pref): its z, xd, y (the
/// down sums) and xg (the rotated rows); nothing else uses them while a g53 call runs.
pub fn g53Big(b: *const Bufs) g53.Big {
    const s = &b.pref;
    return .{ .z = s.z.ptr, .xd = s.xd.ptr, .zd = s.y.ptr, .xh = s.xg.ptr, .z_bytes = s.z.bytes(), .xd_bytes = s.xd.bytes(), .zd_bytes = s.y.bytes(), .xh_bytes = s.xg.bytes() };
}

pub fn sharedSlot(s: *const Scratch, top_k: usize, R: usize) Slot2D {
    const d = s.y.shape[1];
    return .{ .dst = s.y.ptr + top_k * d * 4, .pitch = s.slots * d * 4, .width = d, .height = R };
}

/// cuMemsetD2D32Async, looked up in the driver the runtime opened (zig/src/cuda/abi.zig does not bind it).
pub const Memset2D = *const fn (cuda.abi.DevicePtr, usize, c_uint, usize, usize, cuda.abi.Stream) callconv(.c) cuda.abi.Result;

pub fn memset2D(d: *const cuda.Driver) !Memset2D {
    var lib = d.lib;
    return lib.lookup(Memset2D, "cuMemsetD2D32Async") orelse error.MissingSymbol;
}

/// The expert's device side of one layer, enqueued on one stream: python _window's land, compute and reply, GPU-driven
/// (the take waits on the GPU, bounded and sticky; a post after a failure sends nothing).
pub const Device = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    xg: ?*const xchg.Gpu = null, // fn_xchg (none for a selftest: no network)
    set: *const k3.Set, // qualification_qualification_44ba554e-exl3's six images (and exl3_l2d's four when `sched` is on)
    tri: tri.Tri, // the Triton set (_combine) on `stream`
    sh: ex3.Shared,
    sched: ex3.Sched = .{}, // Mia's 0128 (TF_GLM_EXL3_L2D/_PDL, read once by the tool); off: today's launches
    /// X1 (qualification_ae31a4aa, TF_GLM_EXL3_X1=1, read once by the tool): python 2.15's prompt switches on windows of 64+ rows, with
    /// X1's block (`x1b`: the order lists, the shared suh rows) and each position's shared-suh test; off: today's
    x1: ex3.Prompt = .{},
    x1b: ?*const X1Bufs = null,
    xrow: []const bool = &.{},
    zero: Memset2D,
    x: *const Half,
    base: []const u64, // each position's group address
    b: *const Bufs,
    landing: u64,
    l: proto.Layout,
    limit: f32,
    top_k: usize,
    /// TF_AFD_TIME=1 (qualification_e5a7fc3b): each position's work timed on the GPU, a forward's sum read after its sync (null: off)
    clock: ?*clk.Clock = null,
    timed: ?bool = null,
    /// qualification_1b9bf461 (TF_GLM_HALVES_KERNEL=g53kt3r, read once by the tool): python 2.15's g53_experts.Node; null: today's path
    g53_node: ?*const g53.Node = null,
    /// K4 (qualification_e4ad04cd, TF_GLM_HALVES_KERNEL=k4): prompt windows' routed experts through Mia's prompt kernel (null: off)
    k4: ?*const afdk4.On = null,
    /// qualification_f16009a8 (TF_AFD_GRAPHS=1, never with TF_AFD_TIME=1): decode windows and MTP slices replayed by key; null: eager
    graphs: ?*graphs.Graphs = null,

    pub fn take(g: *Device, n: *const xchg.Node, R: usize) !void {
        try g.xg.?.take(n, &.{.{ .ptr = g.landing, .bytes = proto.wireBytes(g.l.req(R)) }}, g.stream);
    }

    /// The half's experts for MoE position `pos` over the landed request, the shared slot zeroed, the combine.
    pub fn compute(g: *Device, pos: usize, f: Forward) !void {
        if (g.timed == null) g.timed = clk.fromEnv(); // looked at once: the clock made on the first compute
        if (g.timed.? and g.clock == null) g.clock = try clk.Clock.create(std.heap.smp_allocator, g.d, g.x.h, g.x.layers.len);
        if (g.clock) |c| try c.begin(clk.at(f.slot, g.x.layers.len, pos), g.stream);
        const r = rt.route(g.g53_node, g.k4, g.x1, pos, f); // qualification_c863b626: g53kt3r, then k4, then X1's or today's
        if (r == .g53) return g.g53Window(g.g53_node.?, pos, f); // qualification_1b9bf461
        const v = landed(g.l, g.landing, f.rows);
        const s = g.b.scratch(f.rows);
        var ls: k3.Launches = .{};
        const l = g.x.bind(pos, g.base[pos]);
        if (r == .k4) {
            try afdk4.launches(&ls, v, l, s, f.rows, g.limit, g.sh, g.k4.?, g.b, pos);
        } else if (r == .x1) {
            const xb = g.x1b.?;
            try launchesX1(&ls, v, l, s, f.rows, g.limit, g.sh, g.sched, g.x1, xb.order(g.b, s).ptr, if (g.xrow[pos]) xb.suh(pos) else null);
        } else try launchesSched(&ls, v, l, s, f.rows, g.limit, g.sh, g.sched);
        try g.set.run(g.stream, &ls);
        const z = sharedSlot(s, g.top_k, f.rows);
        try g.d.check(g.zero(z.dst, z.pitch, 0, z.width, z.height, g.stream.handle), "cuMemsetD2D32Async");
        try g.tri.combine(s.y.ptr, true, v.wts, g.b.routed.ptr, f.rows, g.l.d, g.l.s);
        if (g.clock) |c| try c.end(clk.at(f.slot, g.x.layers.len, pos), .{ .first = f.first, .count = f.count, .rows = f.rows, .slot = f.slot }, g.stream);
    }

    /// qualification_1b9bf461: a prompt window through glm53f-afd's kernels (python's g53_experts.Node.compute): the landed rows and
    /// routes (slots 0..7 read) into `routed` [R, D] fp32 directly, the shared slot not read, no combine.
    fn g53Window(g: *Device, k: *const g53.Node, pos: usize, f: Forward) !void {
        const v = landed(g.l, g.landing, f.rows);
        var ls: g53.Launches = .{};
        try k.launches(&ls, g.x.bind(pos, g.base[pos]), pos, v.normed, @intCast(g.l.d), v.pick, v.wts, @intCast(g.l.s), g.b.routed.ptr, @intCast(f.rows), g.limit, g53Big(g.b));
        try k.set.run(g.d, g.stream, &ls);
        if (g.clock) |c| try c.end(clk.at(f.slot, g.x.layers.len, pos), .{ .first = f.first, .count = f.count, .rows = f.rows, .slot = f.slot }, g.stream);
    }

    pub fn post(g: *Device, n: *const xchg.Node, R: usize) !void {
        try g.xg.?.post(n, g.b.routed.ptr, g.l.rep(R, false), g.stream);
    }

    /// The forward's one wait; the attention's loss meanwhile ends it at once (the process exits 3).
    pub fn sync(g: *Device, n: *const xchg.Node) !void {
        while (!(try g.stream.done())) {
            if (lost(n) != null) return error.AttentionLost;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000 };
            _ = std.c.nanosleep(&ts, null);
        }
        if (g.clock) |c| try c.fold();
    }
};

test {
    _ = @import("cuda_afd_expert_test.zig");
    _ = clk;
    _ = pairs;
    _ = seqz;
    _ = pgroup;
    _ = mma;
    _ = afdk4;
    _ = rt;
    // the device side compiles on any host; it runs on a GPU
    _ = &Device.take;
    _ = &Device.compute;
    _ = &Device.post;
    _ = &Device.sync;
    _ = &Device.g53Window;
    _ = &memset2D;
    _ = g53;
}
