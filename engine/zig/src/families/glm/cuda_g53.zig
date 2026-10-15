//! glm53f-afd's large-M expert kernels for a TP2 half's prompt windows (TF_GLM_HALVES_KERNEL=g53kt3r, qualification_1b9bf461):
//! python TF-AFD 2.15's g53_experts.py (default_cfg, tuned_cfg with KT3R's tier, max_groups) and the host side of its
//! g53_rank.cu / g53_rank.cpp (g53_ffn_launch's launches, launch_persistent's grid, the binding's checks) over the image
//! g53_rank (zig/kernels/cuda/g53/: 2.15's device code unchanged). The kernels are Hugh Madden's (@dangerm00se,
//! github.com/hughmadden; glm53f-afd @ 91db3cc, MIT), KT3's schedule is glm53f-rank's, and the port onto TensorFold's
//! resident half is python 2.15's (patches 0034, 0035, 0049, 0051). Bits: outputs class against X1 (glm53f-rank's
//! arithmetic: one K split each, its fused epilogue, the 8 slots summed in slot order); every field but sk, skd and
//! swiglu keeps every bit, so KT3R's schedule and default_cfg give a row the same bits in any window.

const std = @import("std");
const cuda = @import("cuda");
const ex3 = @import("cuda_exl3_experts.zig");

pub const H: u32 = 4096; // model width: gate/up K, down N
pub const WID: u32 = 1024; // the half's width (two ranks)
pub const E: u32 = 288; // routed experts a layer
pub const TOPK: u32 = 8;
pub const MAX_ROWS: u32 = 4096; // glm53f-rank's row cap
const CB = WID / 512; // 512-column blocks of a gate/up slice
const NB = WID / 128; // 128-wide Hadamard blocks of the rank width

/// glm53f-rank's g53r_cfg fields for its large-M kernels (g53_rank.h); only sk, skd and swiglu change a bit.
pub const Cfg = struct { mt: u32, sk: u32, skd: u32, nt: u32, gw: u32, l2: u32, fuse: u32, discard: u32, swiglu: u32, rot: u32, gord: u32 };

/// default_cfg(rows): up to 2,048 rows {mt 2, nt 2, gw 8, l2 2}, above {mt 4, nt 4, gw 16, l2 1}; sk 1, skd 1, fused
/// with discard, BF16 SwiGLU everywhere; rot 2 (rowrot, as served) and gord 1 (order off, as served).
pub fn defaultCfg(rows: u32) error{Rows}!Cfg {
    if (rows < 1 or rows > MAX_ROWS) return error.Rows;
    if (rows <= 2048) return .{ .mt = 2, .sk = 1, .skd = 1, .nt = 2, .gw = 8, .l2 = 2, .fuse = 2, .discard = 2, .swiglu = 0, .rot = 2, .gord = 1 };
    return .{ .mt = 4, .sk = 1, .skd = 1, .nt = 4, .gw = 16, .l2 = 1, .fuse = 2, .discard = 2, .swiglu = 0, .rot = 2, .gord = 1 };
}

/// KT3R as 2.15 ships it (.env.example: TF_GLM_EXPERT_KERNEL_MT=4 _GW=16 _NT=4 _L2=1 _TIER_ROWS=1536): KT3's schedule on
/// windows of 1,536 rows and more, default_cfg's below (same-bits fields only).
pub const kt3_tier: u32 = 1536;

/// python's tuned_cfg(rows, fields=KT3, tier=1536) with `kt3r`, else default_cfg(rows).
pub fn tunedCfg(rows: u32, kt3r: bool) error{Rows}!Cfg {
    var c = try defaultCfg(rows);
    if (kt3r and rows >= kt3_tier) {
        c.mt = 4;
        c.gw = 16;
        c.nt = 4;
        c.l2 = 1;
    }
    return c;
}

/// The values g53_rank.cpp accepts.
pub fn cfgOk(c: Cfg) bool {
    const one2 = struct {
        fn f(v: u32) bool {
            return v == 1 or v == 2;
        }
    }.f;
    return (c.mt == 2 or c.mt == 4) and (c.nt == 1 or c.nt == 2 or c.nt == 4) and (c.gw == 8 or c.gw == 16) and c.sk >= 1 and 32 % c.sk == 0 and c.skd >= 1 and c.skd <= 4 and 32 % c.skd == 0 and one2(c.l2) and one2(c.fuse) and one2(c.discard) and c.swiglu <= 1 and one2(c.rot) and one2(c.gord);
}

/// glm53f-rank's bound on a call's groups: routes / (16 mt) full groups plus one partial group a distinct expert.
pub fn maxGroups(rows: u32, mt: u32) u32 {
    const routes = rows * TOPK;
    return @min(routes, routes / (16 * mt) + @min(routes, E));
}

/// The kernels' arguments (g53_rank.cu's Args, passed by value: 272 bytes, every field at the C++ offset).
pub const Args = extern struct {
    gate_t: u64,
    up_t: u64,
    down_t: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    x: u64,
    xs: c_int,
    pick: u64,
    wts: u64,
    ps: c_int,
    ws: c_int,
    groups: u64,
    counts: u64,
    pair_row: u64,
    pair_route: u64,
    inverse: u64,
    z: u64,
    xd: u64,
    zd: u64,
    out: u64,
    gcnt: u64,
    rcnt: u64,
    work: u64,
    fault: u64,
    P: c_int,
    rows: c_int,
    sk: c_int,
    skd: c_int,
    bf16: c_int,
    fused: c_int,
    discard: c_int,
    nch: c_int,
    l2hint: c_int,
    limit: f32,
    xh: u64,
    gorder: u64,
};

/// The plan's buffers (python's Scratch for `rows` rows: int32 groups 4 G, meta 4, pair_row, pair_route, inverse P, gcnt
/// G, rcnt 32 rows, gorder G, with G = max_groups(rows) at 32-row groups); the counters zero between calls.
pub const Plan = struct {
    groups: u64,
    meta: u64,
    pair_row: u64,
    pair_route: u64,
    inverse: u64,
    gcnt: u64,
    rcnt: u64,
    gorder: u64,
    rows: u32,

    /// Each buffer's bytes in order, each placed at a 256-byte boundary (`place`).
    pub fn sizes(rows: u32) [8]usize {
        const P: usize = @as(usize, rows) * TOPK;
        const G: usize = maxGroups(rows, 2);
        return .{ 16 * G, 16, 4 * P, 4 * P, 4 * P, 4 * G, 4 * 32 * @as(usize, rows), 4 * G };
    }

    pub fn bytes(rows: u32) usize {
        var n: usize = 0;
        for (sizes(rows)) |s| n = std.mem.alignForward(usize, n, 256) + s;
        return n;
    }

    /// The buffers of a block of `bytes(rows)` at `base` (256-byte aligned); zero the block once (the counters).
    pub fn place(rows: u32, base: u64) Plan {
        var at: [8]u64 = undefined;
        var n: u64 = 0;
        for (sizes(rows), &at) |s, *a| {
            n = std.mem.alignForward(u64, n, 256);
            a.* = base + n;
            n += s;
        }
        return .{ .groups = at[0], .meta = at[1], .pair_row = at[2], .pair_route = at[3], .inverse = at[4], .gcnt = at[5], .rcnt = at[6], .gorder = at[7], .rows = rows };
    }
};

/// The large buffers and their bytes (python's Scratch.big: views of the half's prompt scratch's z, xd, y, xg).
pub const Big = struct { z: u64, xd: u64, zd: u64, xh: u64, z_bytes: usize, xd_bytes: usize, zd_bytes: usize, xh_bytes: usize };

/// The bytes a call of R rows at `c` needs in each (g53_rank.cpp's checks).
pub fn bigNeeds(R: u32, c: Cfg) struct { z: usize, xd: usize, zd: usize, xh: usize } {
    const P: usize = @as(usize, R) * TOPK;
    return .{ .z = 4 * 2 * c.sk * P * WID, .xd = 2 * P * WID, .zd = 4 * c.skd * P * H, .xh = if (c.rot == 2) 2 * @as(usize, R) * H else 0 };
}

/// One call: the layer (TensorFold's resident half), its shared gate/up suh (null: none), BF16 rows x [rows][xs], the
/// routes pick / wts [rows][ps / ws] (slots 0..7 read), the FP32 output [rows][4096], and the buffers.
pub const Call = struct { layer: ex3.Layer, shared_suh: ?u64, x: u64, xs: u32, pick: u64, ps: u32, wts: u64, ws: u32, out: u64, rows: u32, limit: f32, plan: Plan, big: Big };

/// A launch: `items` > 0 makes it persistent (launch_persistent: the grid is min(items, SMs x its occupancy), at least 1).
pub const Launch = struct { symbol: [:0]const u8, cfg: cuda.Config, args: cuda.Args, items: u32 = 0 };

/// A call's launches in order, after clearing `meta`'s 4 words (16 bytes: groups, fault, the two work counters).
pub const Launches = struct {
    meta: u64 = 0,
    items: [7]Launch = undefined,
    len: usize = 0,

    pub fn push(l: *Launches, x: Launch) void {
        std.debug.assert(l.len < l.items.len);
        l.items[l.len] = x;
        l.len += 1;
    }

    pub fn slice(l: *const Launches) []const Launch {
        return l.items[0..l.len];
    }
};

/// Mangled names as cuobjdump -symbols lists them (g53_kernels.json).
pub const sym = struct {
    pub const plan = "_ZN3g5311plan_kernelEPKiiPKfiiiPNS_5GroupEPiS6_S6_S6_PjPf";
    pub const order = "_ZN3g5312order_kernelEPKNS_5GroupEPKiPi";
    pub const rotrows = "_ZN3g5314rotrows_kernelENS_4ArgsEPK6__halfPS1_";
    pub const epilogue = "_ZN3g5315gateup_epilogueENS_4ArgsE";
    pub const reduce = "_ZN3g5313reduce_kernelENS_4ArgsE";

    /// gateup_big<MT, NW, XH>.
    pub fn gateup(mt: u32, gw: u32, xh: bool) [:0]const u8 {
        for (gateups) |g| if (g.mt == mt and g.gw == gw and g.xh == xh) return g.name;
        return "";
    }

    /// down_big<MT, NT>.
    pub fn down(mt: u32, nt: u32) [:0]const u8 {
        for (downs) |g| if (g.mt == mt and g.nt == nt) return g.name;
        return "";
    }
};

pub const Gateup = struct { mt: u32, gw: u32, xh: bool, name: [:0]const u8 };
pub const Down = struct { mt: u32, nt: u32, name: [:0]const u8 };

/// rank.cu's instances: the persistent kernels g53_ffn_launch picks from.
pub const gateups = blk: {
    var out: [8]Gateup = undefined;
    var n = 0;
    for ([_]bool{ false, true }) |xh| for ([_]u32{ 8, 16 }) |gw| for ([_]u32{ 2, 4 }) |mt| {
        out[n] = .{ .mt = mt, .gw = gw, .xh = xh, .name = std.fmt.comptimePrint("_ZN3g5310gateup_bigILi{d}ELi{d}ELb{d}EEEvNS_4ArgsE", .{ mt, gw, @intFromBool(xh) }) };
        n += 1;
    };
    break :blk out;
};
pub const downs = blk: {
    var out: [6]Down = undefined;
    var n = 0;
    for ([_]u32{ 2, 4 }) |mt| for ([_]u32{ 1, 2, 4 }) |nt| {
        out[n] = .{ .mt = mt, .nt = nt, .name = std.fmt.comptimePrint("_ZN3g538down_bigILi{d}ELi{d}EEEvNS_4ArgsE", .{ mt, nt }) };
        n += 1;
    };
    break :blk out;
};

fn aligned(p: u64, n: u64) bool {
    return p % n == 0;
}

fn int(x: anytype) c_int {
    return @intCast(x);
}

/// g53_rank.cpp's checks: the rows and the configuration, the layer's shape and alignments, the rows' stride, the
/// buffers' sizes and alignments.
pub fn check(c: Call, cfg: Cfg) error{Invalid}!void {
    if (c.rows < 1 or c.rows > MAX_ROWS or !cfgOk(cfg) or !std.math.isFinite(c.limit)) return error.Invalid;
    const l = c.layer;
    if (l.count != E or l.dims != H or l.width != WID) return error.Invalid;
    for ([_]u64{ l.suh_g, l.suh_u, l.svh_g, l.svh_u, l.suh_d, l.svh_d }) |p| if (!aligned(p, 8)) return error.Invalid;
    if (cfg.rot == 2) if (c.shared_suh) |s| {
        if (!aligned(s, 8)) return error.Invalid;
    } else return error.Invalid;
    if (c.xs < H or c.xs % 4 != 0 or !aligned(c.x, 8) or c.ps < TOPK or c.ws < TOPK or !aligned(c.out, 16)) return error.Invalid;
    const need = bigNeeds(c.rows, cfg);
    const b = c.big;
    if (b.z_bytes < need.z or b.xd_bytes < need.xd or b.zd_bytes < need.zd or b.xh_bytes < need.xh) return error.Invalid;
    for ([_]u64{ b.z, b.xd, b.zd }) |p| if (!aligned(p, 128)) return error.Invalid;
    if (need.xh > 0 and !aligned(b.xh, 128)) return error.Invalid;
    if (c.plan.rows < c.rows or !aligned(c.plan.groups, 16)) return error.Invalid; // groups, gcnt, gorder, pairs and rcnt fit
}

/// g53_ffn_launch: the meta words cleared, the plan (and the group order with gord 2), the row rotation (rot 2 and a
/// shared suh), gate/up (its epilogue fused, or the epilogue kernel with fuse 1), down (its reduce fused, or the reduce
/// kernel with fuse 1). No host sync. A layer without a shared suh runs rot 1 (python's routed: the same bits).
pub fn ffn(out: *Launches, c: Call, cfg_in: Cfg) error{Invalid}!void {
    var cfg = cfg_in;
    if (c.shared_suh == null) cfg.rot = 1;
    try check(c, cfg);
    const routes = c.rows * TOPK;
    const gr = 16 * cfg.mt;
    const nch = 32 / cfg.nt; // output chunks of the down kernel (128 nt columns)
    const max_groups = maxGroups(c.rows, cfg.mt);
    const fused = cfg.fuse == 2;
    const discard = fused and cfg.discard == 2;
    const rowrot = cfg.rot == 2;
    const l = c.layer;
    const p = c.plan;
    const a: Args = .{
        .gate_t = l.gate_ptr, .up_t = l.up_ptr, .down_t = l.down_ptr, .suh_g = l.suh_g, .suh_u = l.suh_u, .svh_g = l.svh_g, .svh_u = l.svh_u, .suh_d = l.suh_d, .svh_d = l.svh_d,
        .x = c.x, .xs = int(c.xs), .pick = c.pick, .wts = c.wts, .ps = int(c.ps), .ws = int(c.ws),
        .groups = p.groups, .counts = p.meta, .pair_row = p.pair_row, .pair_route = p.pair_route, .inverse = p.inverse,
        .z = c.big.z, .xd = c.big.xd, .zd = c.big.zd, .out = c.out, .gcnt = p.gcnt, .rcnt = p.rcnt, .work = p.meta + 8, .fault = p.meta + 4,
        .P = int(routes), .rows = int(c.rows), .sk = int(cfg.sk), .skd = int(cfg.skd), .bf16 = if (cfg.swiglu == 1) 0 else 1, .fused = @intFromBool(fused), .discard = @intFromBool(discard), .nch = int(nch), .l2hint = @intFromBool(cfg.l2 == 2), .limit = c.limit,
        .xh = if (rowrot) c.big.xh else 0, .gorder = if (cfg.gord == 2) p.gorder else 0,
    };
    out.meta = p.meta;
    var pa: cuda.Args = .{};
    pa.add(c.pick);
    pa.add(int(c.ps));
    pa.add(c.wts);
    pa.add(int(c.ws));
    pa.add(int(c.rows));
    pa.add(int(gr));
    for ([_]u64{ p.groups, p.meta, p.pair_row, p.pair_route, p.inverse, a.fault, c.out }) |v| pa.add(v);
    out.push(.{ .symbol = sym.plan, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 } }, .args = pa });
    if (a.gorder != 0) {
        var oa: cuda.Args = .{};
        for ([_]u64{ p.groups, p.meta, p.gorder }) |v| oa.add(v);
        out.push(.{ .symbol = sym.order, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 } }, .args = oa });
    }
    var one: cuda.Args = .{};
    one.add(a);
    const items = 2 * CB * max_groups * cfg.sk;
    if (rowrot) {
        var ra: cuda.Args = .{};
        ra.add(a);
        ra.add(c.shared_suh.?);
        ra.add(c.big.xh);
        out.push(.{ .symbol = sym.rotrows, .cfg = .{ .grid = .{ .x = (c.rows * 32 + 7) / 8 }, .block = .{ .x = 256 } }, .args = ra });
    }
    out.push(.{ .symbol = sym.gateup(cfg.mt, cfg.gw, rowrot), .cfg = .{ .grid = .{}, .block = .{ .x = cfg.gw * 32 } }, .args = one, .items = items });
    if (!fused) out.push(.{ .symbol = sym.epilogue, .cfg = .{ .grid = .{ .x = routes * NB / 4 }, .block = .{ .x = 128 } }, .args = one });
    out.push(.{ .symbol = sym.down(cfg.mt, cfg.nt), .cfg = .{ .grid = .{}, .block = .{ .x = 256 } }, .args = one, .items = max_groups * nch * cfg.skd });
    if (!fused) out.push(.{ .symbol = sym.reduce, .cfg = .{ .grid = .{ .x = c.rows * 8 }, .block = .{ .x = 128 } }, .args = one });
}

/// The image on a context: its 14 persistent kernels' occupancy (launch_persistent's, at least 1) and the SM count.
pub const Set = struct {
    module: cuda.Module,
    sms: u32,
    occ: [gateups.len + downs.len]u32,

    pub fn load(d: *const cuda.Driver, image: []const u8, sms: u32) !Set {
        var s: Set = .{ .module = try cuda.Module.load(d, image), .sms = sms, .occ = undefined };
        errdefer s.module.unload();
        for (gateups, 0..) |g, i| s.occ[i] = @max(1, try (try s.module.function(g.name)).occupancy(g.gw * 32, 0));
        for (downs, 0..) |g, i| s.occ[gateups.len + i] = @max(1, try (try s.module.function(g.name)).occupancy(256, 0));
        return s;
    }

    pub fn unload(s: *Set) void {
        s.module.unload();
    }

    /// launch_persistent's grid for `symbol`: SMs x occupancy, at most `items`, at least 1.
    pub fn grid(s: *const Set, symbol: []const u8, items: u32) u32 {
        return persistentGrid(s.sms, s.occupancy(symbol), items);
    }

    fn occupancy(s: *const Set, symbol: []const u8) u32 {
        for (gateups, 0..) |g, i| if (std.mem.eql(u8, g.name, symbol)) return s.occ[i];
        for (downs, 0..) |g, i| if (std.mem.eql(u8, g.name, symbol)) return s.occ[gateups.len + i];
        return 1;
    }

    /// The meta words cleared (cuMemsetD32Async, as cudaMemsetAsync there), then each launch on `stream` in order.
    pub fn run(s: *const Set, d: *const cuda.Driver, stream: cuda.Stream, ls: *const Launches) !void {
        try d.check(d.api.cuMemsetD32Async(ls.meta, 0, 4, stream.handle), "cuMemsetD32Async");
        for (ls.slice()) |l| {
            var x = l;
            if (x.items > 0) x.cfg.grid = .{ .x = s.grid(x.symbol, x.items) };
            try cuda.launch.launch(try s.module.function(x.symbol), x.cfg, stream, &x.args);
        }
    }
};

/// launch_persistent's block count.
pub fn persistentGrid(sms: u32, occ: u32, items: u32) u32 {
    return @max(1, @min(sms * @max(occ, 1), items));
}

/// python's layer_ready: why a layer cannot take the kernels, or null: 4-bit mcg trellises for gate, up and down, 288
/// experts of width 4,096, this TP2 half's 1,024 channels, every trellis 4-byte aligned (`tables`: Half.tables' bytes,
/// the 3 E trellis addresses first) and the six scale vectors 8-byte aligned.
pub fn layerReady(l: ex3.Layer, tables: []const u8) ?[]const u8 {
    if (l.cb != .mcg) return "codebook is not mcg";
    if (l.k2_gu[0] != 8 or l.k2_gu[1] != 8 or l.k2_d[0] != 8 or l.k2_d[1] != 8) return "not 4 bits a weight";
    if (l.count != E or l.dims != H or l.width != WID) return "not 288 experts x 4096 -> 1024";
    if (tables.len < 24 * E) return "no trellis table";
    for (0..3 * E) |i| if (std.mem.readInt(u64, tables[i * 8 ..][0..8], .little) % 4 != 0) return "a trellis is not 4-byte aligned";
    for ([_]u64{ l.suh_g, l.suh_u, l.svh_g, l.svh_u, l.suh_d, l.svh_d }) |p| if (p % 8 != 0) return "a scale vector is not 8-byte aligned";
    return null;
}

/// python's g53_experts.Node on a half (TF_GLM_HALVES_KERNEL=g53kt3r): the image, the plan's buffers (zeroed once),
/// each MoE position's scope (layerReady) and shared gate/up suh row (python's share_suh at load; null: rot 1 there),
/// and KT3R's tier. It `takes` a prompt window (any mode but eager) of 1..max_rows rows on a position in scope.
pub const Node = struct {
    set: *const Set,
    plan: Plan,
    kt3r: bool = true,
    max_rows: u32,
    ready: []const bool,
    suh: []const ?u64,

    pub fn takes(n: *const Node, pos: usize, rows: usize, prompt: bool) bool {
        return prompt and rows >= 1 and rows <= n.max_rows and pos < n.ready.len and n.ready[pos];
    }

    /// Node.compute's call: `rows` BF16 rows x (row stride xs), routes pick / wts [rows][S], out fp32 [rows][4096].
    pub fn launches(n: *const Node, ls: *Launches, l: ex3.Layer, pos: usize, x: u64, xs: u32, pick: u64, wts: u64, S: u32, out: u64, rows: u32, limit: f32, big: Big) error{ Invalid, Rows }!void {
        const c: Call = .{ .layer = l, .shared_suh = n.suh[pos], .x = x, .xs = xs, .pick = pick, .ps = S, .wts = wts, .ws = S, .out = out, .rows = rows, .limit = limit, .plan = n.plan, .big = big };
        try ffn(ls, c, try tunedCfg(rows, n.kt3r));
    }
};

test {
    _ = @import("cuda_g53_test.zig");
}
