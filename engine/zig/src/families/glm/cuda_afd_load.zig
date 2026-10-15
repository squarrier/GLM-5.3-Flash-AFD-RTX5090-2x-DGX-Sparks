//! python 2.15's attention node load (engine.py 677-678: load(world=1, experts=False); 502: no MTP under AFD): Mia's
//! EXL3 checkpoint whole on one rank but the routed experts, which the expert nodes hold. The plan is cuda_exl3_split's
//! for rank 0 of 1 with every MoE layer's expert group emptied; the placement skips the routed experts' views. `load`
//! adds lane B's EXL3 pack pass over it (cuda_pack_exl3.plan, the routed experts' buffers left out, the draft head
//! kept), and `bind` lane B's bind with its UnboundView guard (qualification_qualification_50384483) over every view but the routed experts'.
//! qualification_21e21bc0: `loadWith(.., true)` (TF_AFD_MTP=1) plans the MTP layer too, its routed experts the halves' like the others.
const std = @import("std");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const xs = @import("cuda_exl3_split.zig");
const views = @import("cuda_views.zig");
const md = @import("cuda_model.zig");
const pack = @import("cuda_pack.zig");
const read = @import("cuda_read.zig");
const cuda = @import("cuda");

/// The attention node's plan: rank 0 of 1, no MTP layer, every expert group empty (its copies gone, 0 bytes, unbound).
pub fn plan(gpa: std.mem.Allocator, c: *const cfg.Config, names: *const split.Names) !xs.Plan {
    var mc = c.*;
    mc.mtp = 0;
    var p = try xs.plan(gpa, &mc, names, 0, 1);
    errdefer p.deinit();
    try dropExperts(&p);
    return p;
}

/// A world-1 split's expert groups emptied in place (the other groups and their offsets as they were).
fn dropExperts(p: *xs.Plan) !void {
    var drop: [cfg.max_layers]?u32 = @splat(null);
    for (&p.experts, &drop) |*e, *g| if (e.*) |x| {
        g.* = x.gate.trellis.g;
        for ([_]split.Ref{ x.gate.suh, x.gate.svh, x.up.trellis, x.up.suh, x.up.svh, x.down.trellis, x.down.suh, x.down.svh, x.tables }) |r| if (r.g != g.*.?) return error.ExpertsSpanGroups;
        e.* = null;
    };
    var kept: usize = 0;
    for (p.copies.items) |cp| {
        if (dropped(&drop, cp.g)) continue;
        p.copies.items[kept] = cp;
        kept += 1;
    }
    p.copies.shrinkRetainingCapacity(kept);
    for (drop) |g| if (g) |x| {
        p.groups.items[x] = 0;
    };
    var it = p.tensors.valueIterator();
    while (it.next()) |r| if (dropped(&drop, r.g)) return error.TensorInExpertGroup;
}

fn dropped(drop: []const ?u32, g: u32) bool {
    for (drop) |d| if (d == g) return true;
    return false;
}

/// views.walk's visitor `inner` for every buffer but the routed experts' (python's experts=False).
pub fn skipExperts(comptime Ctx: type, comptime inner: fn (Ctx, []const u8, *views.Buf) anyerror!void) fn (Ctx, []const u8, *views.Buf) anyerror!void {
    return struct {
        fn visit(ctx: Ctx, path: []const u8, b: *views.Buf) anyerror!void {
            if (isExpert(path)) return;
            return inner(ctx, path, b);
        }
    }.visit;
}

pub fn isExpert(path: []const u8) bool {
    return std.mem.indexOf(u8, path, ".moe.experts.") != null;
}

/// The attention node's world-1 load: what it reads (`x`, `plan`'s) and what it makes (`pk`: lane B's EXL3 pack pass,
/// cuda_pack_exl3.plan, over the same split, the routed experts' buffers left out): the embedding, every layer's
/// attention, router, bias and shared expert, the norm, the head, and the draft head (quantize4 of the head, which
/// python builds with MTP off too; qualification_d3c2a5a7 read it at address 0 while it was unbound).
pub const Load = struct {
    x: xs.Plan,
    pk: pack.Plan,

    pub fn deinit(l: *Load) void {
        l.pk.deinit();
        l.x.deinit();
    }

    /// The packed buffers' device bytes (a group's in one allocation, each buffer on a 256-byte boundary).
    pub fn deviceBytes(l: *const Load) u64 {
        var n: u64 = 0;
        for (0..l.pk.groups) |g| n += l.pk.homeBytes(@intCast(g));
        return n;
    }

    /// --dry's stand-in addresses, as tf-glm-forward's --dry places a rank: each group's packed buffers from their own
    /// base, 1 MiB apart, from `base` up; the next free address. Nothing stays in a stored group (only routed experts do).
    pub fn stand(l: *Load, base: u64) !u64 {
        var at = base;
        for (0..l.pk.groups) |gi| {
            const g: u32 = @intCast(gi);
            if (l.pk.keeps(g)) return error.KeptInStoredGroup;
            l.pk.place(g, at);
            at += std.mem.alignForward(u64, l.pk.homeBytes(g), 1 << 20) + (1 << 20);
        }
        return at;
    }
};

/// The load for model `m` (world 1, no MTP: python's attention role) cut as `c` is (c.run layers): one split, packed
/// (the routed experts' items, the only ones whose ops read the split itself, left out), then its expert groups emptied.
pub fn load(gpa: std.mem.Allocator, m: *const md.Model, c: *const cfg.Config, names: *const split.Names) !Load {
    return loadWith(gpa, m, c, names, false);
}

/// `load`, and with `mtp` (qualification_21e21bc0, TF_AFD_MTP=1) the MTP layer as tf-glm-forward loads it (`Weights.mtp`: its attention,
/// router, shared expert, norms and eh_proj), its routed experts left to the halves like every other layer's.
pub fn loadWith(gpa: std.mem.Allocator, m: *const md.Model, c: *const cfg.Config, names: *const split.Names, mtp: bool) !Load {
    if (m.world != 1 or (m.mtp_layers > 0) != mtp or m.layers != c.run) return error.NotTheAttentionNode;
    var mc = c.*;
    if (!mtp) mc.mtp = 0 else if (mc.mtp == 0) return error.NoMtpLayer;
    var x = try xs.plan(gpa, &mc, names, 0, 1);
    errdefer x.deinit();
    if (mtp and x.experts[c.layers] == null) return error.NoMtpLayer; // the checkpoint stores no MTP layer
    var pk = try pack.exl3.plan(gpa, m, &mc, &x);
    errdefer pk.deinit();
    var kept: usize = 0;
    for (pk.items.items) |it| if (!isExpert(it.path)) {
        pk.items.items[kept] = it;
        kept += 1;
    };
    pk.items.shrinkRetainingCapacity(kept);
    try dropExperts(&x);
    return .{ .x = x, .pk = pk };
}

/// The routed experts' views marked off this device (python's experts=False: the expert nodes hold them, and the
/// provider sends the rows instead), so lane B's bind and its guard take every other view.
pub fn markExperts(w: *views.Weights) !void {
    try views.walk(w, views.Path.init("w"), {}, markOne);
}

/// What `dense` did: the q4 triplets read, the bytes the FP8 head saves (null: the head BF16), and the MTP layer's
/// projections on FP8 (mtp_fp8's TF_GLM_MTP_FP8=1; null: as the q4 form holds them).
pub const Dense = struct { q4: pack.q4.Summary, f8_saved: ?u64, mtp8: ?pack.mtp8.Summary = null };

/// Mia's dense-q4 form (qualification_qualification_5322571c's TF_GLM_DENSE=q4:DIR) and its FP8 head (qualification_qualification_ebed70b7's TF_GLM_F8=head) over the attention
/// role's load (qualification_e5a7fc3b): the triplets of rank 0 of 1 (glm_q4_dense.py make --world 1) appended to `ix`'s shards, each
/// eligible projection held 4-bit, as tf-glm-forward overlays a TP2 rank; bind's adapt then takes the views. Null with
/// the switch off: the load as it was. mtp_fp8: with TF_GLM_MTP_FP8=1 and the MTP layer loaded, its projections are made
/// FP8 first, so the overlay leaves them and q4-mtp.safetensors is not read.
pub fn dense(gpa: std.mem.Allocator, ld: *Load, ix: *index.Index, m: *const md.Model) !?Dense {
    const f8 = try pack.f8.fromEnv();
    const mtp = m.mtp_layers > 0;
    const m8 = try pack.mtp8.checkEnv(mtp); // mtp_fp8: 1 needs the MTP layer loaded (TF_AFD_MTP=1), whatever the verb
    const dir = (try pack.q4.fromEnv()) orelse return if (f8) error.F8IsForDenseQ4 else if (m8) error.MtpFp8IsForDenseQ4 else null;
    const s8 = if (m8) try pack.mtp8.apply(&ld.pk) else null;
    // qualification_21e21bc0: with the MTP layer loaded (TF_AFD_MTP=1) its triplets too (q4-mtp.safetensors), as tf-glm-forward's overlay
    const s = try pack.q4.overlay(gpa, &ld.pk, &ld.x.groups, &ld.x.copies, ix, m.layers, mtp and s8 == null, 0, 1, dir);
    return .{ .q4 = s, .f8_saved = if (f8) try pack.f8.apply(&ld.pk) else null, .mtp8 = s8 };
}

/// qualification_21e21bc0: the checkpoint bytes of the MTP layer the attention node reads (`...layers.<layers>.`, the routed experts
/// excepted), summed over the index's names; qualification_25364d42's attention_bytes plus these is the TF_AFD_MTP=1 load's guard.
pub fn mtpBytes(c: *const cfg.Config, names: *const split.Names) !u64 {
    var buf: [32]u8 = undefined;
    const needle = try std.fmt.bufPrint(&buf, ".layers.{d}.", .{c.layers});
    var n: u64 = 0;
    var it = names.iterator();
    while (it.next()) |e| if (std.mem.indexOf(u8, e.key_ptr.*, needle) != null and std.mem.indexOf(u8, e.key_ptr.*, ".mlp.experts.") == null) {
        n += e.value_ptr.len;
    };
    return n;
}

fn markOne(_: void, path: []const u8, b: *views.Buf) anyerror!void {
    if (isExpert(path)) b.host = true;
}

/// lane B's bind (cuda_pack.bind) for the attention node: the routed experts' views marked, then every other view at
/// its packed buffer by python's path, and none left at address 0 (qualification_qualification_50384483's UnboundView). Returns the views bound.
pub fn bind(w: *views.Weights, pk: *const pack.Plan) !usize {
    try markExperts(w);
    return pack.bind(w, pk);
}

/// The load onto a GPU, tf-glm-forward's read-and-pack loop (glm_forward.zig's Run) over the attention's plan: each
/// stored group read into device memory on `stream`, its packed buffers made there by glm_pack's kernels, then freed
/// after its last reader; the packed groups kept (the weights the forward reads, `bytes` of them).
pub const OnDevice = struct {
    gpa: std.mem.Allocator,
    groups: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,
    read_ns: u64 = 0,
    pack_ns: u64 = 0,
    /// the stored bytes on the device at once, at most
    peak: u64 = 0,

    pub fn deinit(o: *OnDevice) void {
        for (o.groups.items) |*b| b.free();
        o.groups.deinit(o.gpa);
    }
};

const Sink = struct {
    stored: []?cuda.DeviceBuffer,
    stream: cuda.Stream,

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.stored[g].?.uploadAsync(@intCast(dst), bytes, s.stream.handle);
    }
    fn flush(ctx: *anyopaque) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.stream.synchronize();
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `ld` read from `shards` and packed on `d` (glm_pack's kernels `k` on `stream`), its buffers placed there.
pub fn toDevice(gpa: std.mem.Allocator, m: *const md.Model, d: *const cuda.Driver, stream: cuda.Stream, k: *const pack.Kernels, ld: *Load, shards: []const [:0]const u8, ro: read.Options) !OnDevice {
    const groups: u32 = @intCast(ld.x.groups.items.len);
    if (groups != ld.pk.groups) return error.OtherSplit;
    const stored = try gpa.alloc(?cuda.DeviceBuffer, groups);
    @memset(stored, null);
    defer {
        for (stored) |*x| if (x.*) |*b| b.free();
        gpa.free(stored);
    }
    const addrs = try gpa.alloc(u64, groups);
    defer gpa.free(addrs);
    var out: OnDevice = .{ .gpa = gpa };
    errdefer out.deinit();
    var f = pack.hostFwd(m);
    f.gpu = pack.packOnly(stream);
    var sink: Sink = .{ .stored = stored, .stream = stream };
    for (0..groups) |gi| {
        const g: u32 = @intCast(gi);
        var needed = false; // an emptied expert group is never read
        for (ld.pk.items.items) |it| if (it.home >= g and pack.readsGroup(it, g)) {
            needed = true;
        };
        if (needed) {
            var b = try cuda.DeviceBuffer.alloc(d, @intCast(@max(ld.x.groups.items[g], 256)));
            stored[g] = b;
            try b.fill8(0, null); // the pads between tensors read as zeros, as tf-glm-forward leaves them
            var mine: std.ArrayList(split.Copy) = .empty;
            defer mine.deinit(gpa);
            for (ld.x.copies.items) |cp| if (cp.g == g) try mine.append(gpa, cp);
            var s = try read.schedule(gpa, mine.items, ro);
            defer s.deinit();
            out.read_ns += (try read.run(gpa, shards, &s, .{ .ctx = &sink, .put = Sink.put, .flush = Sink.flush }, ro)).ns;
        }
        var live: u64 = 0;
        for (stored) |x| if (x) |y| {
            live += y.len;
        };
        out.peak = @max(out.peak, live);
        const n = ld.pk.homeBytes(g);
        if (n > 0) {
            var o = try cuda.DeviceBuffer.alloc(d, @intCast(n));
            errdefer o.free();
            try o.fill8(0, null);
            try out.groups.append(gpa, o);
            out.bytes += n;
            ld.pk.place(g, o.ptr);
        }
        for (addrs, stored) |*ad, x| ad.* = if (x) |y| y.ptr else 0;
        const t0 = nowNs();
        for (ld.pk.items.items) |*it| if (it.home == g) try pack.make(&f, .{ .gpu = k }, it, addrs);
        try stream.synchronize();
        out.pack_ns += nowNs() - t0;
        for (stored, 0..) |*x, sg| if (x.* != null and ld.pk.last(@intCast(sg)) <= g) {
            x.*.?.free();
            x.* = null;
        };
    }
    return out;
}

/// qualification_25364d42's logs/role-bytes.txt: EXL3 078455ff's attention node at world 1, main layers 0-44 and outside, checkpoint bytes.
pub const attention_bytes: u64 = 17_842_600_184;

const testing = std.testing;
const trace = @import("cuda_trace.zig");
const bufs = @import("cuda_buffers.zig");
const index = @import("cuda_index.zig");
const lt = @import("cuda_layer_test.zig");

fn layerOf(name: []const u8) ?usize {
    const p = "model.language_model.layers.";
    if (!std.mem.startsWith(u8, name, p)) return null;
    const rest = name[p.len..];
    return std.fmt.parseInt(usize, rest[0 .. std.mem.indexOfScalar(u8, rest, '.') orelse return null], 10) catch null;
}

/// fixtures_cuda_exl3_split.json's headers (layers 0-3 and 45, experts 0-3) as an index's names, the config cut to
/// them, and what the attention node reads of them: every tensor but the routed experts and the MTP layer.
const Fx = struct { names: split.Names = .empty, c: cfg.Config = .{}, want: u64 = 0, tensors: u32 = 0 };

fn fixtureSplit(a: std.mem.Allocator) !Fx {
    const fx = (try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("fixtures_cuda_exl3_split.json"), .{})).object;
    var x: Fx = .{};
    var it = fx.get("tensors").?.object.iterator();
    x.c.run = @intCast(fx.get("run").?.integer);
    x.c.experts = @intCast(fx.get("experts").?.integer);
    while (it.next()) |e| {
        const t = e.value_ptr.array.items;
        var s: split.Src = .{ .shard = @intCast(t[0].integer), .off = @intCast(t[1].integer), .len = @intCast(t[2].integer), .dtype = .bf16, .rank = @intCast(t[4].array.items.len) };
        for (t[4].array.items, 0..) |d, i| s.shape[i] = @intCast(d.integer);
        const dt = t[3].string;
        s.dtype = if (std.mem.eql(u8, dt, "F32")) .f32 else if (std.mem.eql(u8, dt, "F16")) .f16 else if (std.mem.eql(u8, dt, "I16")) .i16 else if (std.mem.eql(u8, dt, "I32")) .i32 else if (std.mem.eql(u8, dt, "U32")) .u32 else .bf16;
        try x.names.put(a, e.key_ptr.*, s);
        const l = layerOf(e.key_ptr.*);
        if (std.mem.indexOf(u8, e.key_ptr.*, ".mlp.experts.") == null and (l == null or l.? < x.c.run)) {
            x.want += s.len;
            x.tensors += 1;
        }
    }
    return x;
}

test "the attention node's plan on 078455ff's headers (layers 0-3 and 45, experts 0-3): every tensor but the routed experts and the MTP layer, whole" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var fx = try fixtureSplit(arena.allocator());
    const c = fx.c;
    const want = fx.want;
    const tensors = fx.tensors;
    var p = try plan(testing.allocator, &c, &fx.names);
    defer p.deinit();
    for (p.experts) |e| try testing.expect(e == null);
    try testing.expectEqual(want, p.sourceBytes()); // read whole: rank 0 of 1
    try testing.expectEqual(@as(usize, tensors), p.tensors.count());
    var landed: u64 = 0;
    for (p.copies.items) |cp| landed += cp.kept();
    try testing.expectEqual(want, landed);
    try testing.expectEqual(@as(usize, 2 + c.run), p.groups.items.len); // the head's, layers 0-3, layer 3's emptied experts
    try testing.expect(p.bytes() >= want and p.bytes() - want < @as(u64, tensors) * split.alignment); // the groups' pads only
}

const draft_paths = [_][]const u8{ "w.draft_head.weight", "w.draft_head.scales", "w.draft_head.biases" };

test "G-load-A: the attention node's load (layers 0-3, experts 0-3): lane B's EXL3 pack pass without the routed experts, bound at stand-ins, the draft head too, no view at 0" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const gpa = testing.allocator;
    var fx = try fixtureSplit(arena.allocator());
    var m = try md.Model.from(&fx.c, .{ .quant = .exl3 }, 1, 0);
    m.mtp_layers = 0;
    var ld = try load(gpa, &m, &fx.c, &fx.names);
    defer ld.deinit();
    try testing.expectEqual(fx.want, ld.x.sourceBytes());
    for (ld.pk.items.items) |it| try testing.expect(!isExpert(it.path));
    _ = try ld.stand(1 << 36);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    const n = try bind(&w, &ld.pk); // a load without the draft head's buffers stops here: UnpackedView, by name
    try testing.expectEqual(ld.pk.items.items.len, n); // a view for every packed buffer and a buffer for every view
    try testing.expectEqual(@as(usize, 0), (try pack.Unbound.of(&w)).count);
    for (draft_paths) |p| try testing.expect(ld.pk.find(p).?.x.?.quant.src.g == ld.pk.find("w.head.weight").?.op.copy.g);
    const d = w.draft_head.?;
    for ([_]views.Buf{ d.weight, d.scales, d.biases }, draft_paths) |b, p| try testing.expect(b.ptr != 0 and b.ptr == ld.pk.find(p).?.buf.ptr);
    try testing.expect(w.layers[3].moe.?.experts.exl3.gate_ptr.host and !w.layers[3].moe.?.router.host);
    // the draft head left at 0 (qualification_d3c2a5a7's crash): qualification_qualification_50384483's guard names it (a load without its buffers fails bind with
    // UnpackedView, logged: the G-load-A mutant, evidence/qualification_cd073cc0/scripts/mutant-load.sh)
    w.draft_head.?.scales.ptr = 0;
    const u = try pack.Unbound.of(&w);
    try testing.expectEqual(@as(usize, 1), u.count);
    try testing.expectEqualStrings("w.draft_head.scales", u.name());
}

test "the attention node's placement: python 2.15's attention weights exactly (fixtures_cuda_afd_attn.json), the routed experts skipped" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures_cuda_afd_attn.json"), .{});
    defer parsed.deinit();
    const fx = parsed.value.object;
    var m = try md.Model.parse(gpa, fx.get("config_json").?.string, 1, 0);
    try m.cut(@intCast(fx.get("layers").?.integer));
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var space: trace.Space = .{ .gpa = gpa };
    defer space.deinit();
    try views.walk(&w, views.Path.init("w"), &space, skipExperts(*trace.Space, lt.placePython));
    const want = fx.get("buffers").?.object;
    var n: usize = 0;
    var bi = want.iterator();
    while (bi.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, "w.")) {
        n += 1;
    };
    try testing.expectEqual(n, space.entries.items.len);
    for (space.entries.items) |e| {
        errdefer std.debug.print("weight {s}\n", .{e.path});
        try testing.expect(!isExpert(e.path));
        const b = want.get(e.path) orelse return error.NotPythons;
        try testing.expectEqual(b.object.get("nbytes").?.integer, @as(i64, @intCast(e.buf.bytes())));
        try testing.expectEqualStrings(b.object.get("dtype").?.string, @tagName(e.buf.dtype));
    }
}

fn readAll(gpa: std.mem.Allocator, path: [:0]const u8) ![]u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.ReadFailed;
    const out = try gpa.alloc(u8, @intCast(end));
    errdefer gpa.free(out);
    var done: usize = 0;
    while (done < out.len) {
        const got = std.c.pread(fd, out[done..].ptr, out.len - done, @intCast(done));
        if (got <= 0) return error.ReadFailed;
        done += @intCast(got);
    }
    return out;
}

/// Sums the bytes of the buffers under `prefix`.
const Sum = struct {
    prefix: []const u8,
    bytes: u64 = 0,

    fn visit(s: *Sum, path: []const u8, b: *views.Buf) anyerror!void {
        if (std.mem.startsWith(u8, path, s.prefix)) s.bytes += b.bytes();
    }
};

test "the attention node's plan on 078455ff's every header (TF_GLM_EXL3_HEADERS: the checkpoint or its header mirror, else skipped): qualification_25364d42's 16.62 GiB" {
    const gpa = testing.allocator;
    const dir = std.mem.span(std.c.getenv("TF_GLM_EXL3_HEADERS") orelse return error.SkipZigTest);
    var ix = try index.open(gpa, dir);
    defer ix.deinit();
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
    defer gpa.free(path);
    const text = try readAll(gpa, path);
    defer gpa.free(text);
    const c = try cfg.parse(gpa, text);
    var p = try plan(gpa, &c, &ix.names);
    defer p.deinit();
    var m = try md.Model.parse(gpa, text, 1, 0);
    var w = try views.rank(gpa, &m);
    defer w.deinit(gpa);
    var draft: Sum = .{ .prefix = "w.draft_head" };
    try views.walk(&w, views.Path.init("w"), &draft, Sum.visit);
    var all: bufs.Arena = .{};
    try views.walk(&w, views.Path.init("w"), &all, skipExperts(*bufs.Arena, bufs.Arena.visit));
    std.debug.print("qualification_9051d0c1 load: 078455ff attention node plan: {d} B read ({d} copies), {d} B in {d} groups; draft_head {d} B; world-1 views without the routed experts {d} B\n", .{ p.sourceBytes(), p.copies.items.len, p.bytes(), p.groups.items.len, draft.bytes, all.used });
    try testing.expectEqual(attention_bytes, p.sourceBytes());
    try testing.expect(draft.bytes > 0);
}
