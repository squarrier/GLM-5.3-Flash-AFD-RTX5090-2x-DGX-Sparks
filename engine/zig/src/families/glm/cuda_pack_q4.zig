//! The dense-q4 form of an EXL3 rank (qualification_qualification_5322571c; Mia's lane, recipe patch 0005 on 0002, TF_GLM_DENSE=q4): the BF16
//! projections python's weights.load makes with q4() and stack() are held as quantize4_mse's MLX affine 4-bit triplets
//! (make_dense_q4 / stack_dense_q4), read from the files glm_q4_dense.py made for this rank and packed as MLX's dense
//! tensors are (cuda_pack's words and major ops: no new kernel, no new layout). kv_k, kv_v and the head stay BF16:
//! the lane holds them FP8, which Zig CUDA cannot multiply yet. Off (the plan untouched) unless TF_GLM_DENSE=q4:DIR.
const std = @import("std");
const core = @import("core");
const split = @import("cuda_split.zig");
const index = @import("cuda_index.zig");
const views = @import("cuda_views.zig");
const pack = @import("cuda_pack.zig");

const st = core.safetensors;

/// The endings of weights.py's q4() and stack() projections (python's attribute paths); dense() ones are not here.
const leaves = [_][]const u8{ ".kda.proj", ".kda.fb", ".kda.gb", ".kda.o", ".dsa.proj", ".dsa.q_b", ".dsa.o", ".index.kw", ".index.qb", ".mlp.gu", ".mlp.down", ".shared.gu", ".shared.down", "w.mtp.eh" };

/// Whether a pack item (`<base>.weight`, bf16 [n, k] with k % 64 == 0, make_dense_q4's test) is held 4-bit.
pub fn eligible(it: pack.Item) bool {
    const p = it.path;
    if (!std.mem.endsWith(u8, p, ".weight") or it.buf.dtype != .bf16 or it.buf.dims != 2 or it.buf.shape[1] % 64 != 0) return false;
    for (leaves) |l| if (std.mem.endsWith(u8, p[0 .. p.len - ".weight".len], l)) return true;
    return false;
}

/// TF_GLM_DENSE's value: empty or bf16 (the checkpoint's BF16, as before) gives null; q4:DIR the triplets' directory.
pub fn parse(v: []const u8) !?[]const u8 {
    if (v.len == 0 or std.mem.eql(u8, v, "bf16")) return null;
    if (std.mem.startsWith(u8, v, "q4:") and v.len > 3) return v[3..];
    return error.BadDenseForm;
}

pub fn fromEnv() !?[]const u8 {
    const v = std.mem.span(std.c.getenv("TF_GLM_DENSE") orelse return null);
    return parse(v) catch |e| {
        std.log.err("TF_GLM_DENSE is bf16 or q4:DIR (glm_q4_dense.py's triplets for this rank), not {s}", .{v});
        return e;
    };
}

/// One triplet file's header: its tensors (offsets from the data region), the data region's start, its length, and
/// the rank and world its metadata names.
pub const File = struct { tensors: st.Header, data: u64, size: u64 = 0, rank: ?u32 = null, world: ?u32 = null };

pub const Summary = struct { triplets: usize = 0, files: usize = 0, bytes: u64 = 0 };

/// Group `g`'s bytes: each tensor 256-aligned after the last.
fn take(groups: *std.ArrayList(u64), g: u32, len: u64) u64 {
    const at = groups.items[g];
    groups.items[g] = std.mem.alignForward(u64, at + len, split.alignment);
    return at;
}

/// Puts file `f` (shard `shard`) in a new stored group and replaces each eligible item it holds a triplet for with
/// make_q4's three buffers in its place: words [n, k/8] u32 and scales, biases [n, k/64] bf16, packed as q4() packs
/// MLX's. Returns the triplets used (a file's other ones belong to layers beyond the cut, which no item asks for).
pub fn apply(gpa: std.mem.Allocator, pk: *pack.Plan, groups: *std.ArrayList(u64), copies: *std.ArrayList(split.Copy), f: *const File, shard: u32) !usize {
    const a = pk.arena.allocator();
    const g: u32 = @intCast(groups.items.len);
    try groups.append(gpa, 0);
    var out: std.ArrayList(pack.Item) = .empty;
    var used: usize = 0;
    for (pk.items.items) |it| {
        if (!eligible(it)) {
            try out.append(a, it);
            continue;
        }
        const base = it.path[0 .. it.path.len - ".weight".len];
        const n = it.buf.shape[0];
        const k = it.buf.shape[1];
        var es: [3]st.Entry = undefined;
        const found = for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, j| {
            var name: [256]u8 = undefined;
            es[j] = f.tensors.get(try std.fmt.bufPrint(&name, "{s}.{s}", .{ base, leaf })) orelse break false;
            const e = es[j];
            const want: [2]usize = if (j == 0) .{ n, k / 8 } else .{ n, k / 64 };
            if (e.dtype != (if (j == 0) st.DType.u32 else st.DType.bf16) or e.rank != 2 or e.shape[0] != want[0] or e.shape[1] != want[1]) {
                std.log.err("glm q4: {s}.{s} is {t} {any}, not the [{d}, {d}] the plan's {s} needs", .{ base, leaf, e.dtype, e.shape[0..e.rank], want[0], want[1], it.path });
                return error.TripletShape;
            }
        } else true;
        if (!found) {
            try out.append(a, it);
            continue;
        }
        var refs: [3]split.Ref = undefined;
        for (es, &refs) |e, *r| {
            r.* = .{ .g = g, .off = take(groups, g, e.end - e.begin) };
            try copies.append(gpa, .{ .shard = shard, .off = f.data + e.begin, .len = e.end - e.begin, .g = g, .dst = r.off });
        }
        const q = views.Q4.of(n, k);
        const ops = [3]pack.Op{ .{ .words = .{ .src = refs[0], .n = n, .k = k, .rows = .{} } }, .{ .major = .{ .src = refs[1], .n = n, .k = k, .rows = .{} } }, .{ .major = .{ .src = refs[2], .n = n, .k = k, .rows = .{} } } };
        for ([_]views.Buf{ q.weight, q.scales, q.biases }, [_][]const u8{ "weight", "scales", "biases" }, ops) |buf, leaf, op|
            try out.append(a, .{ .path = try std.fmt.allocPrint(a, "{s}.{s}", .{ base, leaf }), .buf = buf, .op = op, .home = g });
        used += 1;
    }
    pk.items = out;
    pk.groups = groups.items.len;
    return used;
}

/// A triplet file's header, read as cuda_index reads a shard's.
pub fn open(a: std.mem.Allocator, path: [:0]const u8) !File {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) {
        std.log.err("glm q4: cannot open {s}", .{path});
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    var head: [8]u8 = undefined;
    if (end < 8 or std.c.pread(fd, &head, 8, 0) != 8) return error.BadSafetensors;
    const hlen = std.mem.readInt(u64, &head, .little);
    if (hlen > @as(u64, @intCast(end)) - 8 or hlen > 1 << 26) return error.BadSafetensors;
    const json = try a.alloc(u8, @intCast(hlen));
    if (std.c.pread(fd, json.ptr, json.len, 8) != @as(isize, @intCast(hlen))) return error.ShortRead;
    var f: File = .{ .tensors = try st.parseHeaderPrefix(a, json, @intCast(@as(u64, @intCast(end)) - 8 - hlen), "w."), .data = 8 + hlen, .size = @intCast(end) };
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    if (doc.object.get("__metadata__")) |m| if (m == .object) {
        if (m.object.get("rank")) |r| if (r == .string) {
            f.rank = std.fmt.parseInt(u32, r.string, 10) catch null;
        };
        if (m.object.get("world")) |w| if (w == .string) {
            f.world = std.fmt.parseInt(u32, w.string, 10) catch null;
        };
    };
    return f;
}

/// The dense-q4 form over an EXL3 plan: `dir`'s q4-L000.safetensors .. (one a backbone layer of the cut) and
/// q4-mtp.safetensors (the MTP layer, when loaded) appended to the index's shards, each a new stored group. Every
/// eligible item must get its triplet, and every file must name this rank and world.
pub fn overlay(gpa: std.mem.Allocator, pk: *pack.Plan, groups: *std.ArrayList(u64), copies: *std.ArrayList(split.Copy), ix: *index.Index, layers: usize, mtp: bool, rank: u32, world: u32, dir: []const u8) !Summary {
    const a = ix.arena.allocator();
    const n = layers + @intFromBool(mtp);
    const shards = try a.alloc([:0]const u8, ix.shards.len + n);
    const sizes = try a.alloc(u64, ix.sizes.len + n);
    @memcpy(shards[0..ix.shards.len], ix.shards);
    @memcpy(sizes[0..ix.sizes.len], ix.sizes);
    var s: Summary = .{};
    for (0..n) |i| {
        const path = if (i < layers) try std.fmt.allocPrintSentinel(a, "{s}/q4-L{d:0>3}.safetensors", .{ dir, i }, 0) else try std.fmt.allocPrintSentinel(a, "{s}/q4-mtp.safetensors", .{dir}, 0);
        const f = try open(a, path);
        if ((f.rank orelse rank + 1) != rank or (f.world orelse world + 1) != world) {
            std.log.err("glm q4: {s} is rank {?d} of {?d}, not {d} of {d}", .{ path, f.rank, f.world, rank, world });
            return error.TripletRank;
        }
        const si: u32 = @intCast(ix.shards.len + i);
        shards[si] = path;
        sizes[si] = f.size;
        s.triplets += try apply(gpa, pk, groups, copies, &f, si);
        s.bytes += groups.items[groups.items.len - 1];
        s.files += 1;
    }
    ix.shards = shards;
    ix.sizes = sizes;
    for (pk.items.items) |it| if (eligible(it)) {
        std.log.err("glm q4: {s} has no triplet in {s}", .{ it.path, dir });
        return error.TripletMissing;
    };
    return s;
}

/// The views a forward reads in the dense-q4 form: each Linear held B16 whose path the plan packs as q4() packs
/// (`<path>.scales` is a packed buffer) becomes that Q4. Without such buffers (MLX, and EXL3 with the switch off)
/// nothing changes. Returns how many changed.
pub fn adapt(w: *views.Weights, p: *const pack.Plan) !usize {
    var ctx: Adapter = .{ .p = p };
    try walk(w, views.Path.init("w"), &ctx);
    return ctx.changed;
}

const Adapter = struct {
    p: *const pack.Plan,
    changed: usize = 0,

    fn visit(x: *Adapter, path: []const u8, l: *views.Linear) !void {
        const b = switch (l.*) {
            .b16 => |b| b,
            .q4, .f8 => return,
        };
        var tmp: [224]u8 = undefined;
        var name: [232]u8 = undefined;
        const py = try pack.pythonPath(&tmp, path);
        const it = x.p.find(try std.fmt.bufPrint(&name, "{s}.scales", .{py})) orelse return;
        if (it.buf.dtype != .bf16 or it.buf.dims != 2) return;
        l.* = .{ .q4 = views.Q4.of(b.n, b.k) };
        x.changed += 1;
    }
};

/// views.walk's traversal, stopping at each Linear.
fn walk(v: anytype, path: views.Path, ctx: *Adapter) anyerror!void {
    const T = @typeInfo(@TypeOf(v)).pointer.child;
    if (T == views.Linear) return ctx.visit(path.str(), v);
    if (T == views.Buf) return;
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| {
            var p = path;
            p.push(name);
            try walk(&@field(v, name), p, ctx);
        },
        .optional => if (v.*) |*x| try walk(x, path, ctx),
        .@"union" => switch (v.*) {
            inline else => |*x| try walk(x, path, ctx),
        },
        .array => for (v, 0..) |*x, i| {
            var p = path;
            p.pushIndex(i);
            try walk(x, p, ctx);
        },
        .pointer => |q| if (q.size == .slice) for (v.*, 0..) |*x, i| {
            var p = path;
            p.pushIndex(i);
            try walk(x, p, ctx);
        },
        else => {},
    }
}

test "TF_GLM_DENSE: unset and bf16 are off, q4:DIR names the triplets, anything else is refused" {
    try std.testing.expectEqual(@as(?[]const u8, null), try parse(""));
    try std.testing.expectEqual(@as(?[]const u8, null), try parse("bf16"));
    try std.testing.expectEqualStrings("/q/r0", (try parse("q4:/q/r0")).?);
    try std.testing.expectError(error.BadDenseForm, parse("q4"));
    try std.testing.expectError(error.BadDenseForm, parse("fp8"));
}

test "q4 dense: an eligible B16 item becomes make_q4's three buffers over a new stored group; kv_k and the head stay" {
    var p: pack.Plan = .{ .arena = .init(std.testing.allocator), .groups = 1 };
    defer p.deinit();
    const a = p.arena.allocator();
    const kept = [_][]const u8{ "w.layers[3].dsa.kv_k.weight", "w.head.weight", "w.layers[0].kda.norm" };
    for ([_][]const u8{ "w.layers[0].kda.proj.weight", kept[0], kept[1], "w.layers[0].kda.o.weight", kept[2] }) |path|
        try p.items.append(a, .{ .path = path, .buf = views.Buf.of(.bf16, &.{ 256, 128 }), .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
    p.items.items[4].buf = views.Buf.of(.bf16, &.{128});
    var groups: std.ArrayList(u64) = .empty;
    defer groups.deinit(std.testing.allocator);
    try groups.append(std.testing.allocator, 4096);
    var copies: std.ArrayList(split.Copy) = .empty;
    defer copies.deinit(std.testing.allocator);
    var f: File = .{ .tensors = .empty, .data = 512 };
    var at: usize = 0;
    for ([_][]const u8{ "weight", "scales", "biases" }, [_]st.DType{ .u32, .bf16, .bf16 }, [_]usize{ 16, 2, 2 }) |leaf, dt, k| {
        const e: st.Entry = .{ .dtype = dt, .rank = 2, .shape = .{ 256, k, 1, 1 }, .begin = at, .end = at + 256 * k * dt.size() };
        try f.tensors.put(a, try std.fmt.allocPrint(a, "w.layers[0].kda.proj.{s}", .{leaf}), e);
        at = e.end;
    }
    try std.testing.expectEqual(@as(usize, 1), try apply(std.testing.allocator, &p, &groups, &copies, &f, 7));
    const names = [_][]const u8{ "w.layers[0].kda.proj.weight", "w.layers[0].kda.proj.scales", "w.layers[0].kda.proj.biases", kept[0], kept[1], "w.layers[0].kda.o.weight", kept[2] };
    try std.testing.expectEqual(names.len, p.items.items.len);
    for (names, p.items.items) |n, it| try std.testing.expectEqualStrings(n, it.path);
    const q = views.Q4.of(256, 128);
    try std.testing.expectEqual(q.weight.shape, p.items.items[0].buf.shape);
    try std.testing.expectEqual(@as(u32, 1), p.items.items[0].home);
    try std.testing.expect(p.items.items[0].op == .words and p.items.items[1].op == .major and p.items.items[2].op == .major);
    try std.testing.expectEqual(@as(usize, 2), groups.items.len);
    try std.testing.expectEqual(@as(usize, 3), copies.items.len);
    try std.testing.expectEqual(split.Copy{ .shard = 7, .off = 512 + 256 * 16 * 4, .len = 256 * 2 * 2, .g = 1, .dst = 256 * 16 * 4 }, copies.items[1]);
    // kda.o has no triplet in this file: it stays BF16 here, and overlay's last check refuses such a plan
    try std.testing.expect(eligible(p.items.items[5]) and !eligible(p.items.items[3]) and !eligible(p.items.items[4]));
}

test "adapt: a B16 view becomes Q4 where the plan packs it as q4() does, and bind then binds all three" {
    var p: pack.Plan = .{ .arena = .init(std.testing.allocator), .groups = 1 };
    defer p.deinit();
    const a = p.arena.allocator();
    var none: [0]views.Layer = .{};
    var w: views.Weights = .{ .embed = .{ .b16 = views.Buf.of(.bf16, &.{ 256, 128 }) }, .layers = &none, .norm = views.Buf.of(.bf16, &.{128}), .head = .{ .b16 = views.B16.of(256, 128) }, .mtp = null };
    const q = views.Q4.of(256, 128);
    for ([_][]const u8{ "w.embed", "w.norm", "w.head.weight", "w.head.scales", "w.head.biases" }, [_]views.Buf{ w.embed.b16, w.norm, q.weight, q.scales, q.biases }) |path, buf|
        try p.items.append(a, .{ .path = path, .buf = buf, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
    p.place(0, 1 << 36);
    try std.testing.expectEqual(@as(usize, 5), try pack.bind(&w, &p));
    try std.testing.expect(w.head == .q4 and w.head.q4.scales.ptr == p.find("w.head.scales").?.buf.ptr);
    try std.testing.expectEqual(@as(usize, 0), try adapt(&w, &p)); // already Q4: nothing more to change
}
