//! qualification_qualification_7c75d777's launch-trace check (qualification_qualification_7c75d777 PLAN §4): each fixture is python-0.6's own forward (ed78d6f) run on a CPU by
//! tools/zig/glm_forward_fixtures.py over a checkpoint's headers (MLX 4-bit; EXL3, whose dense weights are BF16): its
//! Config, every buffer of a TP2 rank with dtype and shape, and the launches of layer_forward for decode windows of
//! the KDA layers 0-2, of layer 3's MLP half (the MoE) and of the MTP layer's moe_block. The Zig side builds the same
//! rank from config.json, places every buffer at a stand-in address, runs the forward in recording mode and must give
//! the same buffers and the same launches, argument for argument.
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");
const moe = @import("cuda_moe.zig");
const dsa = @import("cuda_dsa.zig");

const testing = std.testing;
const J = std.json.Value;
pub const fixture_mlx = @embedFile("fixtures_cuda_forward.json");
pub const fixture_exl3 = @embedFile("fixtures_cuda_forward_exl3.json");

pub fn get(o: J, key: []const u8) J {
    return o.object.get(key) orelse std.debug.panic("fixture: no key {s}", .{key});
}

pub fn int(v: J) i64 {
    return v.integer;
}

fn f32bits(x: f32) i64 {
    return @as(u32, @bitCast(x));
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// The rank as the fixture's run built it: Model (checked against python's Config), weights, buffers, state, space;
/// with `mtp_buffers`, the MTP layer (MoE views at w.mtp.layer.moe, A3's DSA arm at w.mtp.layer.dsa; Weights.mtp itself
/// is lane B's).
pub const Rank = struct {
    m: md.Model,
    w: views.Weights,
    b: bufs.Buffers,
    st: bufs.State,
    space: trace.Space,
    mtp: ?views.Layer = null,

    pub fn init(gpa: std.mem.Allocator, fx: J) !Rank {
        var m = try md.Model.parse(gpa, get(fx, "config_json").string, @intCast(int(get(fx, "world"))), @intCast(int(get(fx, "rank"))));
        try expectConfig(&m, get(fx, "config"));
        try m.cut(@intCast(int(get(fx, "layers"))));
        const rows: usize = @intCast(int(get(fx, "max_rows")));
        const capacity: usize = @intCast(int(get(fx, "capacity")));
        var r: Rank = .{ .m = m, .w = try views.rank(gpa, &m), .b = undefined, .st = undefined, .space = .{ .gpa = gpa } };
        errdefer r.w.deinit(gpa);
        r.b = try bufs.Buffers.init(&m, rows, capacity, false);
        r.st = try bufs.State.init(gpa, &m, capacity, rows, false);
        errdefer {
            r.st.deinit(gpa);
            r.space.deinit();
        }
        try views.walk(&r.w, views.Path.init("w"), &r.space, placePython);
        try views.walk(&r.b, views.Path.init("b"), &r.space, trace.placeAll);
        try views.walk(&r.st, views.Path.init("st"), &r.space, trace.placeAll);
        if (fx.object.get("mtp_buffers")) |mb| { // weights.py layer(n, plain=True): DSA + MoE, no hyper-connections
            r.mtp = try views.mtpLayer(&m);
            try views.walk(&r.mtp.?.moe.?, views.Path.init("w.mtp.layer.moe"), &r.space, trace.placeAll);
            if (mb.object.get("w.mtp.layer.dsa.q_norm") != null) { // the DSA arm's weights (fixtures with A3's cases)
                try views.walk(&r.mtp.?.dsa.?, views.Path.init("w.mtp.layer.dsa"), &r.space, placePython);
            } else r.mtp.?.dsa = null;
        }
        return r;
    }

    pub fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.w.deinit(gpa);
        r.st.deinit(gpa);
        r.space.deinit();
    }
};

/// views.walk's visitor placing a weight under python's name: latent.AbsorbQ4's wkw .. wvb are cuda_views'
/// Absorb.q4 kw .. vb (cuda_pack.bind maps them alike). Public for lane B's step traces past layer 2 (qualification_qualification_973313f6 NOTES).
pub fn placePython(s: *trace.Space, path: []const u8, b: *views.Buf) anyerror!void {
    const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse return s.place(path, b);
    const last = path[dot + 1 ..];
    const q4 = for ([_][]const u8{ "kw", "ks", "kb", "vw", "vs", "vb" }) |x| {
        if (eql(x, last)) break true;
    } else false;
    if (!q4 or !std.mem.endsWith(u8, path[0..dot], ".absorb")) return s.place(path, b);
    var tmp: [200]u8 = undefined;
    try s.place(try std.fmt.bufPrint(&tmp, "{s}.w{s}", .{ path[0..dot], last }), b);
}

/// python-0.6's Config.read of the same config.json, field for field (floats as fp32 bits).
pub fn expectConfig(m: *const md.Model, c: J) !void {
    const ints = .{ "hidden", "layers", "vocab", "heads", "q_lora", "kv_lora", "qk_dim", "v_dim", "lin_heads", "lin_dim", "conv", "experts", "top_k", "moe_width", "shared_width", "dense_width", "streams", "hc_iters", "index_heads", "index_dim", "index_topk", "kpool", "mtp_layers", "group_size", "bits" };
    inline for (ints) |name| {
        errdefer std.debug.print("config {s}\n", .{name});
        try testing.expectEqual(int(get(c, name)), @as(i64, @intCast(@field(m, name))));
    }
    const floats = .{ "eps", "lower", "routed_scale", "hc_eps", "limit" };
    inline for (floats) |name| {
        errdefer std.debug.print("config {s}\n", .{name});
        try testing.expectEqual(int(get(c, name)), f32bits(@field(m, name)));
    }
    try testing.expectEqual(get(c, "norm_topk").bool, m.norm_topk);
    try testing.expectEqualStrings(get(c, "quant").string, @tagName(m.quant));
    try testing.expectEqual(int(get(c, "dense_limit")), @as(i64, @intCast(m.denseLimit())));
    const kinds = get(c, "kinds").array.items;
    for (kinds, 0..) |k, i| try testing.expectEqualStrings(k.string, @tagName(m.kinds[i]));
    for (get(c, "mlp_kinds").array.items, 0..) |k, i| try testing.expectEqualStrings(k.string, @tagName(m.mlp_kinds[i]));
    const eos = get(c, "eos").array.items;
    try testing.expectEqual(eos.len, m.eos_n);
    for (eos, 0..) |e, i| try testing.expectEqual(int(e), @as(i64, m.eos[i]));
}

/// Every python buffer is ours with its dtype, shape and bytes, and we have no other: the main load's (`mtp` false,
/// our w.mtp.* views aside) or the MTP load's MoE (`mtp` true: mtp_buffers against our w.mtp.* views).
pub fn expectBuffers(space: *const trace.Space, want: J, mtp: bool) !void {
    var it = want.object.iterator();
    var n: usize = 0;
    while (it.next()) |kv| {
        errdefer std.debug.print("buffer {s}\n", .{kv.key_ptr.*});
        const e = entry(space, kv.key_ptr.*) orelse return error.MissingBuffer;
        try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
        try testing.expect(get(kv.value_ptr.*, "owner").bool); // python's tensor is its whole storage, as on a GPU
        const shape = get(kv.value_ptr.*, "shape").array.items;
        try testing.expectEqual(shape.len, e.buf.dims);
        for (shape, 0..) |s, i| try testing.expectEqual(int(s), @as(i64, @intCast(e.buf.shape[i])));
        try testing.expectEqual(int(get(kv.value_ptr.*, "nbytes")), @as(i64, @intCast(e.buf.bytes())));
        n += 1;
    }
    var ours: usize = 0;
    for (space.entries.items) |e| if (std.mem.startsWith(u8, e.path, "w.mtp.") == mtp) {
        ours += 1;
        if (want.object.get(e.path) == null) {
            std.debug.print("buffer {s}: not python's\n", .{e.path});
            return error.ExtraBuffer;
        }
    };
    try testing.expectEqual(n, ours);
}

fn entry(space: *const trace.Space, path: []const u8) ?trace.Space.Entry {
    for (space.entries.items) |e| if (eql(e.path, path)) return e;
    return null;
}

/// A case's check: python's buffer names at our addresses (the rank's Space, then the case's own State's), and its
/// temporaries (the case's `tmp`: glue.router's K-slice partials, select_tokens' and sparse attention's, each a
/// torch.empty or a torch op's result in python) each bound to one (buffer, base) of ours for the whole case.
pub const Check = struct {
    space: *const trace.Space,
    extra: ?*const trace.Space = null,
    tmp: ?J = null,
    bound: [24]?Bound = @splat(null),

    const Bound = struct { name: []const u8, path: []const u8, base: i64 };
    const Named = struct { path: []const u8, off: u64, len: u64 };

    fn name(c: *const Check, addr: u64) ?Named {
        for ([_]?*const trace.Space{ c.space, c.extra }) |sp| {
            const s = sp orelse continue;
            for (s.entries.items) |e| if (addr >= e.base and addr < e.base + @max(e.len, 1)) return .{ .path = e.path, .off = addr - e.base, .len = e.len };
        }
        return null;
    }

    fn at(c: *Check, addr: u64, buf: []const u8, off: i64) !void {
        const got = c.name(addr) orelse {
            std.debug.print("address {x}: no buffer (python: {s}+{d})\n", .{ addr, buf, off });
            return error.UnknownAddress;
        };
        errdefer std.debug.print("at {s}+{d}, python {s}+{d}\n", .{ got.path, got.off, buf, off });
        if (c.isTemporary(buf)) return c.temporary(got, buf, off);
        try testing.expectEqualStrings(buf, got.path);
        try testing.expectEqual(off, @as(i64, @intCast(got.off)));
    }

    /// A name the case's `tmp` lists (other "tmp." names, e.g. qualification_qualification_3a5e8e9a's sampler scratch, are buffers like any other).
    fn isTemporary(c: *const Check, name_: []const u8) bool {
        const t = c.tmp orelse return false;
        return t.object.get(name_) != null;
    }

    /// The same buffer and base at every use in the case, with python's bytes from the base inside it.
    fn temporary(c: *Check, got: Named, name_: []const u8, off: i64) !void {
        const info = (c.tmp orelse return error.UnknownTemporary).object.get(name_) orelse return error.UnknownTemporary;
        const ours: i64 = @intCast(got.off);
        for (&c.bound) |*s| {
            if (s.*) |b| {
                if (!eql(b.name, name_)) continue;
                try testing.expectEqualStrings(b.path, got.path);
                return testing.expectEqual(b.base, ours - off);
            }
            try testing.expect(ours - off >= 0 and ours - off + int(get(info, "nbytes")) <= @as(i64, @intCast(got.len)));
            s.* = .{ .name = name_, .path = got.path, .base = ours - off };
            return;
        }
        return error.TooManyTemporaries;
    }

    /// Each temporary the case named is bound.
    fn done(c: *const Check) !void {
        const t = c.tmp orelse return;
        var it = t.object.iterator();
        while (it.next()) |kv| {
            for (c.bound) |s| {
                if (s) |b| if (eql(b.name, kv.key_ptr.*)) break;
            } else return error.UnboundTemporary;
        }
    }
};

fn expectVal(c: *Check, want: J, got: trace.Val) !void {
    const o = want.object;
    if (o.get("t")) |t| {
        const x = switch (got) {
            .t => |x| x,
            else => return error.NotATensor,
        };
        try c.at(x.ptr, get(t, "buf").string, int(get(t, "off")));
        try testing.expectEqualStrings(get(t, "dtype").string, @tagName(x.dtype));
        const shape = get(t, "shape").array.items;
        const stride = get(t, "stride").array.items;
        try testing.expectEqual(shape.len, x.dims);
        for (shape, stride, 0..) |s, d, i| {
            try testing.expectEqual(int(s), @as(i64, @intCast(x.shape[i])));
            try testing.expectEqual(int(d), @as(i64, @intCast(x.stride[i])));
        }
    } else if (o.get("list")) |l| {
        const xs = switch (got) {
            .list => |xs| xs,
            else => return error.NotAList,
        };
        try testing.expectEqual(l.array.items.len, xs.len);
        for (l.array.items, xs) |w, x| try expectVal(c, w, x);
    } else if (o.get("int")) |v| {
        try testing.expectEqual(int(v), got.int);
    } else if (o.get("f32")) |v| {
        try testing.expectEqual(int(v), @as(i64, got.f32));
    } else if (o.get("bool")) |v| {
        try testing.expectEqual(v.bool, got.bool);
    } else if (o.get("none") != null) {
        try testing.expect(got == .none);
    } else return error.UnknownValue;
}

fn expectTriton(c: *Check, w: J, e: trace.Event) !void {
    for (get(w, "grid").array.items, e.grid) |g, x| try testing.expectEqual(int(g), @as(i64, x));
    const args = get(w, "args").array.items;
    try testing.expectEqual(args.len, e.args.len);
    for (args) |a| {
        const name = get(a, "name").string;
        errdefer std.debug.print("argument {s}\n", .{name});
        const mine = for (e.args) |x| {
            if (eql(x.name, name)) break x;
        } else return error.MissingArgument;
        switch (mine.value) {
            .ptr => |p| {
                try testing.expectEqualStrings(get(a, "type").string, p.ty);
                try c.at(p.addr, get(a, "buf").string, int(get(a, "off")));
                if (c.isTemporary(get(a, "buf").string)) try expectAlone(c, e, mine.name, p.addr);
            },
            .i32 => |x| try testing.expectEqual(int(get(a, "int")), @as(i64, x)),
            .f32 => |x| try testing.expectEqual(int(get(a, "f32")), f32bits(x)),
            .u64 => return error.UnexpectedWord,
        }
    }
    var live: usize = 0;
    var it = get(w, "consts").object.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* == .null) continue;
        live += 1;
        errdefer std.debug.print("constexpr {s}\n", .{kv.key_ptr.*});
        const mine = for (e.consts) |x| {
            if (eql(x.name, kv.key_ptr.*)) break x;
        } else return error.MissingConst;
        if (kv.value_ptr.object.get("int")) |x| try testing.expectEqual(int(x), mine.int.?);
        if (kv.value_ptr.object.get("f32")) |x| try testing.expectEqual(int(x), f32bits(mine.f32.?));
    }
    try testing.expectEqual(live, e.consts.len);
}

/// A temporary's buffer of ours is no other argument of the launch (python's is a fresh allocation).
fn expectAlone(c: *Check, e: trace.Event, name: []const u8, addr: u64) !void {
    const mine = c.name(addr).?.path;
    for (e.args) |x| switch (x.value) {
        .ptr => |p| if (!eql(x.name, name)) {
            if (c.name(p.addr)) |o| if (eql(o.path, mine)) return error.TemporaryAliased;
        },
        else => {},
    };
}

/// The trace equals python's event for event (a case without temporaries; qualification_qualification_3a5e8e9a's step test calls it).
pub fn expectEvents(space: *const trace.Space, want: []const J, got: []const trace.Event) !void {
    return expectEventsTmp(space, want, got, null);
}

/// expectEvents with a case's temporaries (its "tmp": python's torch.empty in lane A's wrappers, named per case).
pub fn expectEventsTmp(space: *const trace.Space, want: []const J, got: []const trace.Event, tmp: ?J) !void {
    var c: Check = .{ .space = space, .tmp = tmp };
    return expectCase(&c, want, got);
}

/// The trace equals python's event for event: op, kernel or call, then every argument.
pub fn expectCase(c: *Check, want: []const J, got: []const trace.Event) !void {
    for (want, 0..) |w, i| {
        errdefer std.debug.print("event {d}: python {s} {s}\n", .{ i, get(w, "op").string, get(w, "fn").string });
        if (i >= got.len) return error.MissingEvent;
        const e = got[i];
        try testing.expectEqualStrings(get(w, "op").string, @tagName(e.op));
        try testing.expectEqualStrings(get(w, "fn").string, e.name);
        if (e.op == .triton) {
            try expectTriton(c, w, e);
            continue;
        }
        const vals = get(w, "args").array.items;
        try testing.expectEqual(vals.len, e.vals.len);
        for (vals, e.vals, 0..) |v, x, j| {
            errdefer std.debug.print("argument {d}\n", .{j});
            try expectVal(c, v, x);
        }
    }
    try testing.expectEqual(want.len, got.len);
    try c.done();
}

/// forward.layer_forward's MLP half (forward.py 437-442), as cuda_layer.layerForward runs it: hc_pre with ffn_hc and
/// post_norm, the dense MLP or the MoE, hc_post.
pub fn mlpHalf(f: *fwd.Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize) !void {
    const m = f.m;
    const h = l.ffn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const g = if (l.mlp != null) try layer.mlpBlock(f, l, b, R, .{}) else try moe.block(f, l, b, R, .{});
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
}

/// forward.layer_forward's attention half (forward.py 423-436) on a DSA layer, as cuda_layer.layerForward runs it:
/// hc_pre with attn_hc and in_norm, the DSA block, hc_post.
fn attnHalf(f: *fwd.Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: layer.Opt) !void {
    const m = f.m;
    const h = l.attn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const g = try dsa.block(f, l, st, b, R, opt);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
}

/// A State the DSA cases ran on beside the rank's (python's long-context State, the MTP load's two), in its own Space
/// from the rank's last address on; its buffers are python's under `prefix` ("st." or "long.st.") of `want`. When
/// `want` lists pb.* (decode.Engine's pbuf), the prompt-chunk cases' Buffers in the same Space.
pub const Extra = struct {
    st: bufs.State,
    pb: ?bufs.Buffers = null,
    space: trace.Space,

    pub fn init(gpa: std.mem.Allocator, r: *const Rank, m: *const md.Model, want: J, prefix: []const u8, long: bool) !Extra {
        var key: [64]u8 = undefined;
        const kc = get(want, try std.fmt.bufPrint(&key, "{s}kc.0", .{prefix}));
        const cap: usize = @intCast(int(get(kc, "shape").array.items[0]));
        var x: Extra = .{ .st = try bufs.State.init(gpa, m, cap, r.b.rows, long), .space = .{ .gpa = gpa, .next = r.space.next } };
        errdefer x.deinit(gpa);
        try views.walk(&x.st, views.Path.init("st"), &x.space, trace.placeAll);
        if (want.object.get("pb.x")) |px| {
            x.pb = try bufs.Buffers.init(m, @intCast(int(get(px, "shape").array.items[0])), cap, true);
            try views.walk(&x.pb.?, views.Path.init("pb"), &x.space, trace.placeAll);
        }
        var it = want.object.iterator();
        var n: usize = 0;
        while (it.next()) |kv| {
            const prompt = std.mem.startsWith(u8, kv.key_ptr.*, "pb.");
            if (!prompt and !std.mem.startsWith(u8, kv.key_ptr.*, prefix)) continue;
            errdefer std.debug.print("state buffer {s}\n", .{kv.key_ptr.*});
            const path = if (prompt) kv.key_ptr.* else try std.fmt.bufPrint(&key, "st.{s}", .{kv.key_ptr.*[prefix.len..]});
            const e = entry(&x.space, path) orelse return error.MissingBuffer;
            try testing.expectEqualStrings(get(kv.value_ptr.*, "dtype").string, @tagName(e.buf.dtype));
            const shape = get(kv.value_ptr.*, "shape").array.items;
            try testing.expectEqual(shape.len, e.buf.dims);
            for (shape, 0..) |s, i| try testing.expectEqual(int(s), @as(i64, @intCast(e.buf.shape[i])));
            try testing.expectEqual(int(get(kv.value_ptr.*, "nbytes")), @as(i64, @intCast(e.buf.bytes())));
            n += 1;
        }
        try testing.expectEqual(n, x.space.entries.items.len); // and no State (or prompt Buffers) buffer python lacks
        return x;
    }

    pub fn deinit(x: *Extra, gpa: std.mem.Allocator) void {
        x.st.deinit(gpa);
        x.space.deinit();
    }
};

fn pick(x: *?Extra) !*Extra {
    return if (x.*) |*e| e else error.NoState;
}

/// Cases checked, by kind: whole layers (KDA 0-2, DSA 3), MLP halves (layer 3's MoE), the MTP layer's moe_block,
/// layer 3's attention half (long contexts off and on), the MTP layer's dsa_block, layer 3 on a prompt chunk.
const Counts = struct { layer: usize = 0, mlp: usize = 0, moe: usize = 0, attn: usize = 0, dsa: usize = 0, prompt: usize = 0 };

/// One fixture: python's Config, every buffer, and each case's launches event for event.
fn expectTrace(text: []const u8) !Counts {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    var r = try Rank.init(gpa, fx);
    defer r.deinit(gpa);
    try expectBuffers(&r.space, get(fx, "buffers"), false);
    if (fx.object.get("mtp_buffers")) |mb| try expectBuffers(&r.space, mb, true);
    var long: ?Extra = null; // python's long-context State (indexer caches) for layer 3's sparse cases
    defer if (long) |*x| x.deinit(gpa);
    if (fx.object.get("long_state")) |ls| long = try Extra.init(gpa, &r, &r.m, ls, "st.", true);
    var mm = r.m; // python's MTP load keeps the MTP layer: its cache and its indexer cache (st.index[-1])
    mm.mtp_layers = @intCast(int(get(get(fx, "config"), "mtp_layers")));
    var mtp: ?Extra = null;
    defer if (mtp) |*x| x.deinit(gpa);
    var mtp_long: ?Extra = null;
    defer if (mtp_long) |*x| x.deinit(gpa);
    if (fx.object.get("mtp_state")) |ms| {
        mtp = try Extra.init(gpa, &r, &mm, ms, "st.", false);
        mtp_long = try Extra.init(gpa, &r, &mm, ms, "long.st.", true);
    }
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var it = get(fx, "cases").object.iterator();
    var n: Counts = .{};
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        const rows: usize = @intCast(int(get(c, "rows")));
        const block = if (c.object.get("block")) |v| v.string else "layer";
        const is_long = if (c.object.get("long")) |v| v.bool else false;
        const x: ?*Extra = if (eql(block, "dsa")) try pick(if (is_long) &mtp_long else &mtp) else if (is_long) try pick(&long) else null;
        const st = if (x) |e| &e.st else &r.st;
        st.pos = @intCast(int(get(c, "pos")));
        st.parity = @intCast(int(get(c, "parity")));
        const opt: layer.Opt = .{ .nch = @intCast(int(get(c, "nch"))), .host_pos = st.pos };
        var f = fwd.Fwd.recording(&r.m, &t);
        if (eql(block, "moe")) { // the MTP layer's moe_block alone (mtp.py)
            try testing.expectEqualStrings("mtp", get(c, "layer").string);
            _ = try moe.block(&f, &(r.mtp orelse return error.NoMtpLayer), &r.b, rows, .{});
            n.moe += 1;
        } else if (eql(block, "dsa")) { // the MTP layer's dsa_block alone (mtp.py 44-45: host_pos is mtp_len)
            try testing.expectEqualStrings("mtp", get(c, "layer").string);
            try testing.expectEqual(int(get(c, "capacity")), @as(i64, @intCast(st.capacity)));
            st.mtp_len = st.pos;
            _ = try dsa.block(&f, &(r.mtp orelse return error.NoMtpLayer), st, &r.b, rows, opt);
            n.dsa += 1;
        } else {
            const l = &r.w.layers[@intCast(int(get(c, "layer")))];
            if (eql(block, "mlp")) {
                try mlpHalf(&f, l, &r.b, rows);
                n.mlp += 1;
            } else if (eql(block, "attn")) {
                try testing.expectEqual(int(get(c, "capacity")), @as(i64, @intCast(st.capacity)));
                try attnHalf(&f, l, st, &r.b, rows, opt);
                n.attn += 1;
            } else if (c.object.get("prefill")) |pr| { // a prompt chunk on python's pbuf (decode.prefill)
                const xe = x orelse return error.NoState;
                const pb = if (xe.pb) |*p| p else return error.NoPromptBuffers;
                try testing.expectEqual(int(pr), @as(i64, @intCast(pb.rows)));
                try testing.expectEqual(int(get(c, "capacity")), @as(i64, @intCast(st.capacity)));
                try layer.layerForward(&f, l, st, pb, rows, opt);
                n.prompt += 1;
            } else {
                try layer.layerForward(&f, l, st, &r.b, rows, opt);
                n.layer += 1;
            }
        }
        try f.flush();
        var ck: Check = .{ .space = &r.space, .extra = if (x) |e| &e.space else null, .tmp = c.object.get("tmp") };
        try expectCase(&ck, get(c, "events").array.items, t.events.items);
    }
    return n;
}

/// Each fixture's cases: KDA layers 0-2 (12), layer 3 whole at (1, 0) and (4, 37), layer 3's MoE half (4) and the MTP
/// MoE (4); layer 3's attention at R = 1, 4, 8 from positions 0 and 37, with long contexts also from 2,048 (R 4, 8)
/// and 2,560 (17); the MTP layer's attention at (1, 0), (4, 37), long contexts (1, 37), (4, 2,048), (8, 2,560) (5);
/// layer 3 whole on a prompt chunk of 64 and of 512 rows from 0 (2).
const all_cases: Counts = .{ .layer = 14, .mlp = 4, .moe = 4, .attn = 17, .dsa = 5, .prompt = 2 };

test "MLX 4-bit: python-0.6's Config, buffers and launches for a TP2 rank: layers 0-3, the DSA attention, the MTP layer" {
    try testing.expectEqual(all_cases, try expectTrace(fixture_mlx));
}

test "EXL3 (BF16 dense, EXL3 routed experts): the same cases, absorb as BF16 blocks, the shared expert as BF16" {
    try testing.expectEqual(all_cases, try expectTrace(fixture_exl3));
}

test "the DSA block's limits; a launch path without the matmul kernels says so" {
    const gpa = testing.allocator;
    var m = try md.Model.from(&@import("config.zig").Config{}, .{}, 2, 0);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var st = try bufs.State.init(gpa, &m, 2560, 8, false);
    defer st.deinit(gpa);
    const b = try bufs.Buffers.init(&m, 8, 2560, false);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = fwd.Fwd.recording(&m, &t);
    var per_head = w.layers[3]; // TF_GLM_LATENT=0 (per-head caches, no absorb): not ported
    per_head.dsa.?.absorb = null;
    try testing.expectError(error.NotPorted, dsa.block(&f, &per_head, &st, &b, 1, .{ .host_pos = 0 }));
    try testing.expectError(error.NotDsa, dsa.block(&f, &w.layers[0], &st, &b, 1, .{}));
    var bare_st = st; // a State without the MTP cache: the MTP layer's arm says so
    bare_st.mtp_kc = null;
    try testing.expectError(error.NoMtpCache, dsa.block(&f, &w.mtp.?.layer, &bare_st, &b, 1, .{}));
    const pb = try bufs.Buffers.init(&m, 2048, 6144, true); // prompt chunks past 512 rows: dense blocks at pos_dev + r0
    _ = try dsa.block(&f, &w.layers[3], &st, &pb, 600, .{ .host_pos = 0 });
    var long_st = try bufs.State.init(gpa, &m, 6144, 8, true); // a 2,048-row chunk's sparse rows: 640 MiB of partials
    defer long_st.deinit(gpa); // (python's torch.empty) past b.lat_s, in a dk.Scratch the engine reserves
    try testing.expectError(error.NoDsaScratch, dsa.block(&f, &w.layers[3], &long_st, &pb, 2048, .{ .host_pos = 2048 }));
    try testing.expectError(error.Rows, layer.layerForward(&f, &w.layers[3], &st, &b, 9, .{}));
    try testing.expectError(error.Rows, layer.layerForward(&f, &w.layers[0], &st, &b, 9, .{}));
    try testing.expectError(error.Rows, moe.block(&f, &w.layers[3], &b, 9, .{}));
    try testing.expectError(error.NotMoe, moe.block(&f, &w.layers[0], &b, 1, .{}));
    var bare: fwd.Fwd = .{ .m = &m, .tri = .{ .set = null } }; // neither a recorder nor a GPU: the matmul stops first
    try testing.expectError(error.NoGpu, layer.mlpBlock(&bare, &w.layers[0], &b, 1, .{}));
    try testing.expectError(error.NoGpu, dsa.block(&bare, &w.layers[3], &st, &b, 1, .{ .host_pos = 0 }));
}

/// The bytes python's buffers under `prefix` take, each at a 256-byte boundary (pinned host buffers out).
fn pythonBytes(want: J, prefix: []const u8) usize {
    var n: usize = 0;
    var it = want.object.iterator();
    while (it.next()) |kv| {
        if (!std.mem.startsWith(u8, kv.key_ptr.*, prefix) or eql(kv.key_ptr.*, "b.ids_host")) continue;
        n += std.mem.alignForward(usize, @intCast(int(get(kv.value_ptr.*, "nbytes"))), 256);
    }
    return n;
}

test "one arena each holds python's Buffers and State, 256-aligned and disjoint (both formats)" {
    const gpa = testing.allocator;
    for ([_][]const u8{ fixture_mlx, fixture_exl3 }) |text| {
        const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
        defer parsed.deinit();
        var r = try Rank.init(gpa, parsed.value);
        defer r.deinit(gpa);
        var b = r.b;
        var a: bufs.Arena = .{ .base = 1 << 40 };
        try views.walk(&b, views.Path.init("b"), &a, bufs.Arena.visit);
        try testing.expectEqual(pythonBytes(get(parsed.value, "buffers"), "b."), a.used);
        var st = try bufs.State.init(gpa, &r.m, r.st.capacity, r.b.rows, false);
        defer st.deinit(gpa);
        var s: bufs.Arena = .{ .base = 1 << 41 };
        try views.walk(&st, views.Path.init("st"), &s, bufs.Arena.visit);
        try testing.expectEqual(pythonBytes(get(parsed.value, "buffers"), "st."), s.used);
        try testing.expect(b.normed.ptr % 256 == 0 and b.ids_host.ptr == r.b.ids_host.ptr and st.rec.ptr % 256 == 0); // host buffer untouched
        try testing.expect(b.x.ptr + b.x.bytes() <= b.normed.ptr or b.normed.ptr + b.normed.bytes() <= b.x.ptr);
    }
}

test {
    _ = @import("cuda_moe_test.zig"); // the MoE's launch half: python's calls as the kernels' launches (both formats)
    _ = @import("cuda_moe.zig"); // A4: sharedSlot's 2D copy against its per-row copies
    _ = @import("cuda_dsa_kernels.zig"); // select_tokens' tail: python's torch ops on drawn pools as glm_select's arithmetic
    _ = @import("cuda_dsa_test.zig"); // A4: the DSA arms on prompt chunks past 512 rows (fixtures_cuda_dsa*.json)
}
