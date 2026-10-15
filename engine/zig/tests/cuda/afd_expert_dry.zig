//! tf-glm-afd expert --check-weights FILE --dry (qualification_e5a7fc3b, the full-depth dry): half H's load on a CPU over the
//! checkpoint's headers (qualification_765330ce's header mirror on the CPU build host). The split's plan of rank H (ae.half, as serve loads it),
//! its expert groups placed at stand-in addresses 1 MiB apart, every position's entries (python's walk: the tables,
//! the scales, each expert's trellis) checked inside their group, and each position bound (Half.bind: no pointer at 0,
//! none outside its group). FORECAST lines give what a Spark holds: the groups, the compute block at the run's rows
//! and the exchange's pinned region. Nothing is read; FILE gets each entry's path, shape, dtype and bytes (sha256
//! null). Exit 0, or 1 with the first entry or pointer out of place.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const expert = @import("afd_expert.zig");

const ae = glm.afd_expert;
const proto = cuda.xchg.proto;
const Allocator = std.mem.Allocator;

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

fn mib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 20);
}

/// FILE's bytes (check-weights' list).
fn saveFile(path: []const u8, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    if (path.len + 1 > buf.len) return error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const fd = std.c.open(buf[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.NoOutput;
    defer _ = std.c.close(fd);
    if (bytes.len > 0 and std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.ShortWrite;
}

/// Every u64 of a bound layer (its 12 pointers) inside [base, base + bytes), none at 0; the first field that is not.
pub fn outside(l: anytype, base: u64, bytes: u64) ?[]const u8 {
    const info = @typeInfo(@TypeOf(l)).@"struct";
    inline for (info.field_names, info.field_types) |name, ty| {
        if (ty == u64) {
            const p = @field(l, name);
            if (p == 0 or p < base or p >= base + bytes) return name;
        }
    }
    return null;
}

pub fn run(gpa: Allocator, a: expert.Args) !u8 {
    var c = try expert.config(gpa, a.model);
    if (a.layers) |n| c.run = @min(n, c.layers);
    var ix = try glm.index.open(gpa, a.model);
    defer ix.deinit();
    const mtp = try glm.afd_mtp.fromEnv(); // qualification_21e21bc0: TF_AFD_MTP=1, the MTP layer's experts the last position
    var x = try ae.halfWith(gpa, &c, &ix.names, a.half, mtp);
    defer x.deinit();
    var list: std.ArrayList(ae.Entry) = .empty;
    defer list.deinit(gpa);
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    var at: u64 = 1 << 36;
    var largest: u64 = 0;
    var count: usize = 0;
    var tables: u64 = 0;
    var entry_bytes: u64 = 0;
    var k4_ready: usize = 0;
    for (x.groups, 0..) |g, pos| {
        const gb = x.plan.groups.items[g];
        largest = @max(largest, gb);
        list.clearRetainingCapacity();
        try ae.entries(gpa, &x, pos, &list);
        for (list.items) |e| {
            if (e.bytes == 0 or e.off + e.bytes > gb) {
                var pb: [96]u8 = undefined;
                std.debug.print("FAIL expert {d} --dry: {s} at {d} + {d} B is outside its group of {d} B\n", .{ a.half, ae.path(e, &pb), e.off, e.bytes, gb });
                return 1;
            }
            if (ae.isTable(e)) tables += e.bytes;
            entry_bytes += e.bytes;
            try ae.writeDigest(&aw.writer, e, null);
            count += 1;
        }
        if (outside(x.bind(pos, at), at, gb)) |field| {
            std.debug.print("FAIL expert {d} --dry: position {d} (layer {d}) binds {s} outside its group\n", .{ a.half, pos, x.layers[pos], field });
            return 1;
        }
        if (a.halves == .k4 and ae.afdk4.ready(&x, pos, at, c.topk + 1)) k4_ready += 1; // qualification_e4ad04cd
        at += std.mem.alignForward(u64, gb, 1 << 20) + (1 << 20);
    }
    try saveFile(a.check_weights.?, aw.written());
    const l: proto.Layout = .{ .d = c.hidden, .s = c.topk + 1 };
    var b = try ae.Bufs.init(l.d, l.s, c.moe_inter / 2, c.experts, a.rows, a.prefill_rows);
    var landing = ae.Landing.init(l, a.prefill_rows);
    const block = try ae.place(&b, &landing, 0);
    var nd: usize = 0;
    var it = std.mem.tokenizeScalar(u8, a.devs, ',');
    while (it.next()) |_| nd += 1;
    const settings: proto.Settings = .{ .layout = l, .max_rows = a.prefill_rows, .ports = @max(nd, 1), .inflight = a.inflight, .lanes = a.lanes }; // qualification_9e49592e: the lanes' region
    const region: u64 = @intCast(settings.region().total()); // qualification_9e49592e: N slots a source with TF_AFD_LANES=N
    const groups = x.bytes();
    std.debug.print("INFO expert {d} --dry: {d} MoE layers ({d}-{d}) of {d}{s}, {d} entries bound at stand-ins (12 pointers a layer inside its group, none at 0) into {s}\n", .{ a.half, x.layers.len, x.layers[0], x.layers[x.layers.len - 1], c.run, if (mtp) " and the MTP layer's experts last (TF_AFD_MTP=1)" else "", count, a.check_weights.? });
    std.debug.print("FORECAST expert {d}: groups {d:.3} GiB ({d} B; the entries {d:.3} GiB, the tables {d:.2} MiB; the checkpoint bytes kept {d:.3} GiB), the largest group {d:.1} MiB\n", .{ a.half, gib(groups), groups, gib(entry_bytes), mib(tables), gib(x.kept()), mib(largest) });
    std.debug.print("FORECAST expert {d}: compute block {d:.1} MiB ({d} decode rows, {d} prompt rows); the exchange's pinned region {d:.1} MiB ({d} port{s}); device total {d:.3} GiB; the CUDA context, the EXL3 and Triton modules come on top (measured, not planned here)\n", .{ a.half, mib(block), a.rows, a.prefill_rows, mib(region), settings.ports, if (settings.ports == 1) "" else "s", gib(groups + block + region) });
    if (a.halves == .k4) { // qualification_e4ad04cd: python's prompt_kernel_ready at each position, the stand-ins 1 MiB aligned
        var kb = ae.afdk4.Bufs.init(&b, c.experts);
        std.debug.print("FORECAST expert {d}: K4: {d} of {d} positions take it (4-bit mcg, D in 256s, the width in 128s, 16-byte trellises); its block {d:.1} KiB beside X1's\n", .{ a.half, k4_ready, x.layers.len, @as(f64, @floatFromInt(try ae.afdk4.place(&kb, 0))) / 1024 });
    }
    std.debug.print("AFD OK expert {d} check-weights --dry: {d} entries, {d} B in groups\n", .{ a.half, count, groups });
    return 0;
}

test "the dry check's field test: every pointer inside the group, none at 0" {
    const L = struct { a: u64, b: u64, n: u32 };
    try std.testing.expectEqual(@as(?[]const u8, null), outside(L{ .a = 100, .b = 150, .n = 0 }, 100, 64));
    try std.testing.expectEqualStrings("b", outside(L{ .a = 100, .b = 164, .n = 0 }, 100, 64).?);
    try std.testing.expectEqualStrings("a", outside(L{ .a = 0, .b = 120, .n = 0 }, 0, 64).?);
    const d = try expert.parse(&.{ "--half", "1", "--model", "/m", "--check-weights", "/tmp/w.jsonl", "--dry", "--layers", "8" });
    try std.testing.expect(d.dry and d.layers.? == 8 and d.half == 1);
    try std.testing.expectError(error.DryIsCheckWeights, expert.parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--dry" }));
    try std.testing.expectError(error.DryIsCheckWeights, expert.parse(&.{ "--dry", "--half", "0", "--model", "/m", "--check-weights", "f", "--ref", "r.json" }));
    try std.testing.expectError(error.NoFile, run(std.testing.allocator, d));
}
