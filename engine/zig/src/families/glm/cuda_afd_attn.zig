//! python 2.15's AFD attention node half of a MoE layer (src@0849f3f glm5_next/cuda: forward.moe_route + moe_send,
//! afd.AfdWire.begin/exchange/collect, afd_mcdma.McdmaTransport.header/send/recv) as cuda_fwd.Moe's provider: the
//! router and top-k select; the request [header | normed | pick | pad | wts] packed on the device (send's torch.cat)
//! and posted to both expert nodes over native RoCE exchange (qualification_fe808d66); the shared expert while it flies; both expert halves'
//! partials taken into afd_recv's planes 0 and 1 and the shared expert's rows into plane 2, which hc_post sums.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const qmm = @import("cuda_qmm.zig");
const tri = @import("cuda_triton.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const Buf = views.Buf;
const T = trace.T;

/// mcdma.HDR_INTS: a header row's int32 words (afd_mcdma's header table is [layers, HDR_INTS])
pub const hdr_ints = proto.HDR_BYTES / 4;

/// The window kinds this provider sends (afd.MODE_EAGER, MODE_CHUNK); slices, pairs and lanes are python's later modes.
pub const Mode = proto.Mode;

/// The exchange this node runs: native RoCE exchange's attention Node, and its two launches: on the forward's stream (`gpu`), or
/// a CPU test's stand-in (`host`: the forward recorded, the exchange run for real, as G-fail and the audit need).
pub const Wire = struct { node: *xchg.Node, gpu: ?*const xchg.Gpu = null, host: ?Host = null };

/// A CPU test's exchange: `post` gets layer `pos`'s header row (python's) for R rows, `take` both replies of R rows.
pub const Host = struct {
    ctx: *anyopaque,
    post: *const fn (ctx: *anyopaque, node: *xchg.Node, hdr: *const [proto.HDR_BYTES]u8, pos: usize, R: usize) anyerror!void,
    take: *const fn (ctx: *anyopaque, node: *xchg.Node, pos: usize, R: usize) anyerror!void,
};

/// TF_AFD_CHECK's pinned ring on a GPU: each MoE layer's request (as posted) and both replies (as taken), copied down
/// on the stream for the step's sync (the audit hashes them there); forwards of more than `rows` rows are not checked.
/// A CPU test's exchange fills a ring of host memory itself.
pub const Ring = struct {
    rows: usize,
    bytes: []u8,
    req: usize,
    rep: usize,
    owned: ?cuda.HostBuffer = null,

    /// A ring for forwards of up to `rows` rows over `bytes` (null: none yet; `total` gives the size it needs).
    pub fn of(l: proto.Layout, rows: usize, bytes: ?[]u8) Ring {
        const req = proto.wireBytes(l.req(rows));
        const rep = 4 * rows * l.d;
        return .{ .rows = rows, .bytes = bytes orelse &.{}, .req = req, .rep = rep };
    }

    pub fn total(r: Ring, layers: usize) usize {
        return @max(layers, 1) * (r.req + 2 * r.rep);
    }

    /// Layer `pos`'s request, plane-0 reply and plane-1 reply.
    pub fn at(r: *const Ring, pos: usize) [3][]u8 {
        const each = r.req + 2 * r.rep;
        const b = r.bytes[pos * each ..][0..each];
        return .{ b[0..r.req], b[r.req..][0..r.rep], b[r.req + r.rep ..][0..r.rep] };
    }
};

/// afd_mcdma's buffers: the request (req_dev), the header table on the device and pinned (hdr_tab, hdr_tab_cpu).
pub const WireBufs = struct { req: Buf, hdr: Buf, hdr_host: Buf };

/// tf-glm-afd attn bench's profile split of a MoE layer (qualification_e5a7fc3b): `post` before the request's post (the router done),
/// `sent` once the shared expert is queued after it, `taken` once both halves' replies are in; the profile drains the
/// stream at each (shares, not a speed). Null (every other run): the layer's launches unchanged.
pub const Mark = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, what: At) anyerror!void };
pub const At = enum { post, sent, taken };

/// One request section: `bytes` from `src` (null: zeros) at `dst` bytes into the request.
pub const Piece = struct { src: ?u64, dst: usize, bytes: usize };

/// qualification_0f991257 (TF_AFD_PAIRS=1, evidence/qualification_0f991257/WIRE.md §3): a paired window's header words 7-9, its slot (0 A, 1 B), the other
/// window's rows and the pair's first seq (A's).
pub const Pair = struct { slot: u1, other: usize, first: u32 };

/// qualification_9e49592e (TF_AFD_LANES=N, WIRE.md §13): a lane window's header words 7-10, its lane (0..n-1), the next lane's rows (the
/// last lane's: lane 0's), the group's first seq (lane 0's) and its lanes n (3..N).
pub const Lane = struct { slot: u8, next: usize, first: u32, n: u8 };

/// Layer i's position among the MoE layers (afd.moe_layers: the wire's layer word), null for a dense layer.
pub fn moePos(m: *const md.Model, i: usize) ?usize {
    if (i >= m.layers or m.mlp_kinds[i] != .moe) return null;
    var n: usize = 0;
    for (m.mlp_kinds[0..i]) |k| n += @intFromBool(k == .moe);
    return n;
}

/// The model's MoE layers (the settings' `layers`, every forward's exchanges).
pub fn moeLayers(m: *const md.Model) usize {
    var n: usize = 0;
    for (m.mlp_kinds[0..m.layers]) |k| n += @intFromBool(k == .moe);
    return n;
}

/// afd_mcdma.send's torch.cat for R rows: [header row | normed[:R] | pick[:R] | zeros to 256 | wts[:R]], then zeros to
/// the wire's 16-byte unit (MCDMA sends req(R) bytes; xchg posts whole 16-byte units, so the tail is defined).
pub fn pack(l: proto.Layout, R: usize, hdr_row: u64, normed: u64, pick: u64, wts: u64) [6]Piece {
    const o = l.offsets(R);
    const sel = 4 * l.s * R;
    return .{
        .{ .src = hdr_row, .dst = 0, .bytes = proto.HDR_BYTES },
        .{ .src = normed, .dst = o.normed, .bytes = o.pick - o.normed },
        .{ .src = pick, .dst = o.pick, .bytes = sel },
        .{ .src = null, .dst = o.pick + sel, .bytes = o.wts - o.pick - sel },
        .{ .src = wts, .dst = o.wts, .bytes = sel },
        .{ .src = null, .dst = o.end, .bytes = proto.wireBytes(o.end) - o.end },
    };
}

/// TF_GLM_EXL3_SEQ on the attention node (qualification_969df694), read once before anything loads for its settings' seq (nodeConfig):
/// unset, empty or 0 off; 1 on; anything else refused, as the halves read it (their X1 rule is their own: the
/// attention runs no expert and launches nothing different).
pub fn seqFromEnv() error{BadExl3Seq}!bool {
    const v = std.mem.span(std.c.getenv("TF_GLM_EXL3_SEQ") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    std.debug.print("INFO attn: TF_GLM_EXL3_SEQ takes 0 or 1 (or unset), not \"{s}\"\n", .{v});
    return error.BadExl3Seq;
}

pub const Afd = struct {
    m: *const md.Model,
    layout: proto.Layout,
    /// the widest window (Buffers' rows: decode 8, or the prompt chunk's)
    rows: usize,
    /// the MoE layers: each forward's exchanges and the header table's rows
    layers: usize,
    /// qualification_21e21bc0 (TF_AFD_MTP=1): the halves serve the MTP layer's experts as position `layers` too; every forward is then a
    /// slice of the `layers + 1` positions (the backbone's [0, layers), or the MTP head's [layers, 1))
    mtp: bool = false,
    /// this forward's positions [first, first + count) (python's slice: header words 3 and 6)
    first: usize = 0,
    count: usize = 0,
    /// python Buffers.afd_recv (forward.py 134-136): [3 rows D] fp32, the planes hc_post sums
    afd_recv: Buf,
    w: WireBufs,
    wire: ?Wire = null,
    dev: ?cuda.DeviceBuffer = null,
    host: ?cuda.HostBuffer = null,
    /// afd.seq: the forward's number (header word 1)
    seq: u32 = 0,
    /// afd.rows: this forward's rows (header word 2)
    cur: usize = 0,
    mode: Mode = .eager,
    /// afd.layer: the next exchange's MoE position
    pos: usize = 0,
    /// afd.dead: a forward stopped between its exchanges (python's die); every later begin refuses
    dead: bool = false,
    /// TF_AFD_CHECK on a GPU (`audit`): each layer's request and replies copied down for the step's sync
    ring: ?Ring = null,
    /// the bench's profile split (qualification_e5a7fc3b): null unless a profiled step runs
    mark: ?Mark = null,
    /// qualification_0f991257: this window's pair words (cuda_afd_pairs.zig sets them for a pair's two windows); null for every other
    pair: ?Pair = null,
    /// qualification_0f991257: TF_AFD_PAIRS=1 on this node, read once by the role: its settings carry pairs (nodeConfig), so both halves
    /// must agree at the handshake; false leaves the settings text as it was
    paired: bool = false,
    /// qualification_c533375f: TF_AFD_INFLIGHT on this node (1, or 2 with `paired`), read once by the role: 2 puts inflight=2 in the
    /// settings (nodeConfig), and a pair then keeps two requests in flight a link (cuda_afd_pairs.zig)
    inflight: u8 = 1,
    /// qualification_969df694: TF_GLM_EXL3_SEQ on this node (seqFromEnv), read once by the role: true puts seq=1 in the settings
    /// (nodeConfig), so both halves must run SEQ too; the attention's launches are the same either way
    exl3_seq: bool = false,
    /// qualification_f504f9df: TF_GLM_HALVES_KERNEL's settings value (cuda_halves_kernel.zig), read once by the role: its settings carry
    /// it (nodeConfig), so both halves must run the same prompt-window kernel; empty leaves the settings text as it was
    halves: []const u8 = "",
    /// qualification_17c70e37: TF_AFD_STRIPES on this node, read once by the role: its settings carry stripes (nodeConfig), so both halves
    /// must agree at the handshake; off (k 0) leaves the settings text and the take as they were
    stripes: proto.Stripes = .{},
    /// qualification_9e49592e: this window's lane words (cuda_afd_lanes.zig sets them for a lane group's windows); null for every other
    lane: ?Lane = null,
    /// qualification_9e49592e: TF_AFD_LANES on this node (0, or 3..4 with `paired`; `inflight` is then the same N), read once by the role:
    /// it puts lanes=N in the settings (nodeConfig), so both halves must agree it, and the region takes N slots a source
    lanes: u8 = 0,

    /// The provider for a world-1 attention rank of `m` (python 2.15's AFD reads EXL3: BF16 dense, the shared expert).
    pub fn init(m: *const md.Model, rows: usize) !Afd {
        return initWith(m, rows, false);
    }

    /// `init`, with `mtp` (qualification_21e21bc0) the MTP layer's experts served as the last position: its header row, and slices.
    pub fn initWith(m: *const md.Model, rows: usize, mtp: bool) !Afd {
        if (m.world != 1) return error.NotWorld1;
        if (m.quant != .exl3) return error.NotExl3;
        if (mtp and m.mtp_layers == 0) return error.NoMtp;
        const D = m.hidden;
        const layout: proto.Layout = .{ .d = D, .s = m.top_k + 1 };
        const n = moeLayers(m);
        const rows_hdr = n + @intFromBool(mtp);
        return .{
            .m = m,
            .layout = layout,
            .rows = rows,
            .layers = n,
            .mtp = mtp,
            .afd_recv = Buf.of(.f32, &.{3 * rows * D}),
            .w = .{
                .req = Buf.of(.u8, &.{proto.wireBytes(layout.req(rows))}),
                .hdr = Buf.of(.i32, &.{ @max(rows_hdr, 1), hdr_ints }),
                .hdr_host = .{ .dtype = .i32, .dims = 2, .shape = .{ @max(rows_hdr, 1), hdr_ints, 1, 1, 1 }, .host = true },
            },
        };
    }

    /// The positions the expert halves serve: the MoE layers, and the MTP layer's last with `mtp`.
    pub fn positions(a: *const Afd) usize {
        return a.layers + @intFromBool(a.mtp);
    }

    /// cuda_fwd's provider over this exchange.
    pub fn provider(a: *Afd) fwd.Moe {
        return .{ .ctx = a, .block = block };
    }

    /// The attention node's exchange config (tf-glm-afd attn on a GPU, and G-wire): this layout, the chunk's rows and
    /// one port a device, as the expert's serve agrees them (afd_expert.zig's settings); the watcher's silence, the
    /// GPU's give-up, and whether the watcher runs.
    pub fn nodeConfig(a: *const Afd, max_rows: usize, ports: usize, silent_s: f64, give_up_s: f64, watcher: bool) xchg.Config {
        return .{
            .settings = .{ .layout = a.layout, .max_rows = max_rows, .ports = ports, .pairs = a.paired, .inflight = a.inflight, .seq = a.exl3_seq, .halves = a.halves, .stripes = a.stripes.k, .stripe_min = a.stripes.min, .lanes = a.lanes },
            .rules = .{ .silent_ns = @intFromFloat(silent_s * std.time.ns_per_s) },
            .give_up_ns = @intFromFloat(give_up_s * std.time.ns_per_s),
            .run_watcher = watcher,
        };
    }

    /// The device block (afd_recv, the request, the header table, zeroed) and the pinned header rows, on `wire`.
    pub fn attach(a: *Afd, d: *const cuda.Driver, wire: Wire) !void {
        if (a.dev != null) return error.Attached;
        var ar: bufs.Arena = .{};
        for ([_]*Buf{ &a.afd_recv, &a.w.req, &a.w.hdr }) |b| try bufs.Arena.visit(&ar, "", b);
        var dev = try cuda.DeviceBuffer.alloc(d, ar.used);
        errdefer dev.free();
        try dev.fill8(0, null);
        ar = .{ .base = dev.ptr };
        for ([_]*Buf{ &a.afd_recv, &a.w.req, &a.w.hdr }) |b| try bufs.Arena.visit(&ar, "", b);
        const host = try cuda.HostBuffer.alloc(d, a.w.hdr_host.bytes());
        @memset(host.bytes, 0);
        a.w.hdr_host.ptr = @intFromPtr(host.bytes.ptr);
        a.dev = dev;
        a.host = host;
        a.wire = wire;
    }

    pub fn detach(a: *Afd) void {
        if (a.ring) |*r| if (r.owned) |*h| h.free();
        if (a.host) |*h| h.free();
        if (a.dev) |*d| d.free();
        a.ring = null;
        a.host = null;
        a.dev = null;
        a.wire = null;
    }

    /// TF_AFD_CHECK on a GPU: a pinned ring for forwards of up to `rows` rows (a layer's request and two replies each).
    pub fn audit(a: *Afd, d: *const cuda.Driver, rows: usize) !void {
        if (a.ring != null) return error.Attached;
        var r = Ring.of(a.layout, rows, null);
        const host = try cuda.HostBuffer.alloc(d, r.total(a.positions()));
        r.bytes = host.bytes;
        r.owned = host;
        a.ring = r;
    }

    /// Layer `pos`'s header row of this forward (afd_mcdma.header's row `pos`); with `mtp` a slice's (word 6 its count);
    /// a paired window's (qualification_0f991257) is mode 7 with its words 7-9; a lane window's (qualification_9e49592e) mode 9 with its words 7-10.
    pub fn headerRow(a: *const Afd, pos: usize, out: *[proto.HDR_BYTES]u8) void {
        var h: proto.Header = .{ .fseq = @intCast(a.seq & 0x7FFFFFFF), .rows = @intCast(a.cur), .layer = @intCast(pos), .mode = if (a.mtp) .slice else a.mode, .layers = @intCast(a.positions()) };
        if (a.pair) |p| {
            h.mode = .pair;
            h.extra = .{ p.slot, @intCast(p.other), @intCast(p.first & 0x7FFFFFFF), 0 };
        }
        if (a.lane) |l| {
            h.mode = .lanes;
            h.extra = .{ l.slot, @intCast(l.next), @intCast(l.first & 0x7FFFFFFF), l.n };
        }
        h.write(out);
        if (a.mtp) std.mem.writeInt(i32, out[24..28], @intCast(a.count), .little); // python's slice count (word 6)
    }

    /// afd_mcdma.header's rows for this forward into `out` ([positions, 64] int32): magic, seq, rows, the MoE position,
    /// mode, layers; words 6-63 zero (slices' and pairs' words are not sent here; qualification_21e21bc0's slices send word 6).
    pub fn headers(a: *const Afd, out: []u8) void {
        for (0..a.positions()) |i| a.headerRow(i, out[i * proto.HDR_BYTES ..][0..proto.HDR_BYTES]);
    }

    /// afd.begin + afd_mcdma.header (afd.py 372-411, afd_mcdma.py 167-183), before the forward's kernels: refused once
    /// the exchange failed or this node died (python's dead), and a forward stopped between its exchanges dies here
    /// (python's die: the node's failure is recorded); then seq + 1 and every MoE layer's header row in one blocking copy.
    pub fn begin(a: *Afd, R: usize, mode: Mode, s: ?cuda.Stream) !void {
        return a.beginAt(0, a.layers, R, mode, s);
    }

    /// qualification_21e21bc0: the MTP head's forward, the halves' last position alone (TF_AFD_MTP=1's slice [layers, 1)).
    pub fn beginMtp(a: *Afd, R: usize, mode: Mode, s: ?cuda.Stream) !void {
        if (!a.mtp) return error.NoMtp;
        return a.beginAt(a.layers, 1, R, mode, s);
    }

    /// `begin` for positions [first, first + count): the backbone's, or with `mtp` the MTP head's.
    fn beginAt(a: *Afd, first: usize, count: usize, R: usize, mode: Mode, s: ?cuda.Stream) !void {
        if (a.wire) |w| try w.node.check();
        if (a.dead) return error.ExpertFailed;
        if (a.pos != a.first and a.pos != a.first + a.count) {
            a.dead = true;
            if (a.wire) |w| w.node.fail(.{ .why = .stopped, .peer = 0, .seq = w.node.doorbell() });
            return error.OutOfStep;
        }
        if (R == 0 or R > a.rows) return error.Rows;
        if (mode != .eager and mode != .chunk) return error.NotPorted;
        a.seq +%= 1;
        a.cur = R;
        a.mode = mode;
        a.first = first;
        a.count = count;
        a.pos = first;
        const h = a.host orelse return;
        const st = s orelse return error.NoStream;
        const n = a.positions() * proto.HDR_BYTES;
        a.headers(h.bytes[0..n]);
        try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(a.w.hdr.ptr, h.bytes.ptr, n, st.handle), "cuMemcpyHtoDAsync");
        try st.synchronize(); // python's hdr_tab.copy_ is blocking: the next forward's rewrite must not race it
    }

    /// forward.moe_block's AFD branch (forward.py 758-765): moe_route, moe_send (exchange, then the shared expert into
    /// b.sy while the request flies), afd.collect; hc_post sums the three planes, the expert halves first. qualification_0f991257 split it
    /// into its three steps (`route`, `request`, `take`) so a pair can run the other chunk between them; the launches
    /// are the same.
    fn block(ctx: *anyopaque, f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: layer.Opt) anyerror!fwd.Planes {
        const a: *Afd = @ptrCast(@alignCast(ctx));
        try a.route(f, l, b, R);
        try a.request(f, l, b, R, opt);
        return a.take(f, b, R);
    }

    /// moe_route: the checks (before any launch), then the router and the top-k select into b.pick and b.wts.
    pub fn route(a: *Afd, f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize) !void {
        const m = f.m;
        const x = l.moe orelse return error.NotMoe;
        _ = x.shared orelse return error.NoSharedExpert; // python's AFD branch needs the local shared expert (EXL3)
        if (R == 0 or R > b.rows or R != a.cur) return error.Rows;
        const at = (if (a.mtp and l.index == m.layers) a.layers else moePos(m, l.index)) orelse return error.NotMoe; // qualification_21e21bc0: the MTP layer last
        if (a.pos >= a.first + a.count or at != a.pos) return error.OutOfStep;
        if (b.sgu == null or b.sact == null or b.sxs == null or b.sy == null) return error.NoExl3Scratch;
        const normed = layer.rowsOf(b.normed, R);
        const ne = x.router.shape[0];
        if (b.gath.numel() < tri.router_ks * R * ne) return error.RouterScratch;
        try f.tri.router(normed.ptr, normed.stride[0], x.router.ptr, b.mlog.ptr, ne, b.gath.ptr, R, x.router.shape[1], ne);
        try f.tri.select(b.mlog.ptr, x.bias.ptr, b.pick.ptr, b.wts.ptr, R, m.experts, m.top_k, m.routed_scale, m.norm_topk);
        if (a.mark) |k| try k.at(k.ctx, .post);
    }

    /// moe_send: the request posted to both halves, then the shared expert into b.sy while it flies.
    pub fn request(a: *Afd, f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !void {
        const m = f.m;
        const s = (l.moe orelse return error.NotMoe).shared orelse return error.NoSharedExpert;
        const sgu = b.sgu orelse return error.NoExl3Scratch;
        const sact = b.sact orelse return error.NoExl3Scratch;
        const sxs = b.sxs orelse return error.NoExl3Scratch;
        const sy = b.sy orelse return error.NoExl3Scratch;
        const normed = layer.rowsOf(b.normed, R);
        try a.send(f, b, R);
        a.pos += 1;
        try qmm.mm(f, b.prefill, normed, s.gu, layer.rowsOf(b.xs, R), layer.rowsOf(sgu, R), false, b.sk, opt.chip);
        try f.tri.swiglu(sgu.ptr, sact.ptr, sxs.ptr, m.limit, R, s.width);
        try qmm.mm(f, b.prefill, layer.rowsOf(sact, R), s.down, layer.rowsOf(sxs, R), layer.rowsOf(sy, R), true, b.sk, opt.chip);
        if (a.mark) |k| try k.at(k.ctx, .sent);
    }

    /// afd.collect: both halves' replies and the shared expert's rows into afd_recv's three planes, for hc_post.
    pub fn take(a: *Afd, f: *fwd.Fwd, b: *const bufs.Buffers, R: usize) !fwd.Planes {
        try a.collect(f, b.sy orelse return error.NoExl3Scratch, R);
        if (a.mark) |k| try k.at(k.ctx, .taken);
        return .{ .g = a.afd_recv.ptr, .world = 3 };
    }

    /// afd.exchange → afd_mcdma.send (timing aside): the request packed on the stream, then posted to both experts; with
    /// the audit's ring, copied down too (python's check clones). Recording: one `afd.exchange` event naming what the
    /// request carries, then a CPU test's exchange if it has one.
    fn send(a: *Afd, f: *fwd.Fwd, b: *const bufs.Buffers, R: usize) !void {
        if (f.rec) |t| {
            try t.call(.ext, "afd.exchange", &.{ .{ .t = layer.rowsOf(b.normed, R) }, .{ .t = layer.rowsOf(b.pick, R) }, .{ .t = layer.rowsOf(b.wts, R) }, .{ .int = @intCast(R) } });
            const w = a.wire orelse return;
            const h = w.host orelse return;
            var row: [proto.HDR_BYTES]u8 = undefined;
            a.headerRow(a.pos, &row);
            return h.post(h.ctx, w.node, &row, a.pos, R);
        }
        const g = f.gpu orelse return error.NoGpu;
        const w = a.wire orelse return error.NoWire;
        const xg = w.gpu orelse return error.NoWire;
        const st = g.stream;
        const req = a.w.req.ptr;
        for (pack(a.layout, R, a.w.hdr.ptr + a.pos * proto.HDR_BYTES, b.normed.ptr, b.pick.ptr, b.wts.ptr)) |p| {
            if (p.bytes == 0) continue;
            if (p.src) |src| {
                try st.d.check(st.d.api.cuMemcpyDtoDAsync_v2(req + p.dst, src, p.bytes, st.handle), "cuMemcpyDtoDAsync");
            } else try st.d.check(st.d.api.cuMemsetD8Async(req + p.dst, 0, p.bytes, st.handle), "cuMemsetD8Async");
        }
        const bytes = proto.wireBytes(a.layout.req(R));
        try xg.post(w.node, req, bytes, st);
        if (a.ring) |*r| if (R <= r.rows) try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(r.at(a.pos)[0].ptr, req, bytes, st.handle), "cuMemcpyDtoHAsync");
    }

    /// afd.collect → afd_mcdma.recv (TF_GLM_PARTIALS_BF16 aside): expert 0's and expert 1's partials (python's ranks 1
    /// and 2) into planes 0 and 1, then b.sy[:R] into plane 2 (a torch copy); [3, R, D] at afd_recv for hc_post; with
    /// the audit's ring, both replies copied down. Recording: python's event names the three planes [R, D] (4·R·D
    /// bytes apart), b.sy[:R] and R; then a CPU test's exchange if it has one.
    fn collect(a: *Afd, f: *fwd.Fwd, sy: Buf, R: usize) !void {
        const D = a.layout.d;
        const n = R * D;
        if (f.rec) |t| {
            try t.call(.ext, "afd.collect", &.{ .{ .t = T.of(a.afd_recv, 0, &.{ R, D }) }, .{ .t = T.of(a.afd_recv, 4 * n, &.{ R, D }) }, .{ .t = T.of(a.afd_recv, 8 * n, &.{ R, D }) }, .{ .t = layer.rowsOf(sy, R) }, .{ .int = @intCast(R) } });
            const w = a.wire orelse return;
            const h = w.host orelse return;
            return h.take(h.ctx, w.node, a.pos - 1, R);
        }
        const g = f.gpu orelse return error.NoGpu;
        const w = a.wire orelse return error.NoWire;
        const xg = w.gpu orelse return error.NoWire;
        const st = g.stream;
        const recv = a.afd_recv.ptr;
        try xg.take(w.node, &[_]xchg.Out{ .{ .ptr = recv, .bytes = 4 * n }, .{ .ptr = recv + 4 * n, .bytes = 4 * n } }, st);
        if (a.ring) |*r| if (R <= r.rows) {
            const s = r.at(a.pos - 1);
            try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(s[1].ptr, recv, 4 * n, st.handle), "cuMemcpyDtoHAsync");
            try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(s[2].ptr, recv + 4 * n, 4 * n, st.handle), "cuMemcpyDtoHAsync");
        };
        try st.d.check(st.d.api.cuMemcpyDtoDAsync_v2(recv + 8 * n, sy.ptr, 4 * n, st.handle), "cuMemcpyDtoDAsync");
    }
};
