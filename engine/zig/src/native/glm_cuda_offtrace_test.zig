//! qualification_qualification_44ba554e.8 day 3's identity gate for drafts off (qualification_qualification_279525c9): one recording rank of each of the two (the prompt fixture's
//! model, no MTP head, no peer) serves the pair test's scenario through the lane core over cuda_serve.Serve (a request
//! to its length, one cancelled after 3 rounds, a second one), and its launch trace is digested (sha256 over each
//! launch's op, name, grid, arguments with addresses by buffer name, constexprs and values). The digests are qualification_0c0ab476's
//! binaries' (qualification_qualification_e91d6b91-serve 841b3f1, where this file ran unchanged: evidence/qualification_qualification_279525c9/logs/offtrace-841b3f1.log), so with
//! drafts off the served launches are qualification_0c0ab476's.
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const gl = @import("glm_cuda_lanes.zig");
const rec = glm.serve_rec;
const trace = glm.trace;
const Serve = glm.serve.Serve;

const testing = std.testing;
const gpa = testing.allocator;

/// qualification_qualification_e91d6b91-serve 841b3f1's digests of the scenario's launches, rank 0 and rank 1 (885 launches each there).
const want = [2]?[]const u8{ "ffc12135de1ec141f0797bff669e915eb1084f83ad0921b1e84ab785dd118877", "a4b860a09475414b62fee0ebd4727655cf68e8e97ae9b9dfa08f7f9a0c4e1114" };

const Sha = std.crypto.hash.sha2.Sha256;

fn put(h: *Sha, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch unreachable;
    h.update(s);
}

fn addr(h: *Sha, space: *const trace.Space, x: u64) void {
    if (space.name(x)) |p| put(h, "@{s}+{d}", .{ p.path, p.off }) else put(h, "#{d}", .{x});
}

fn val(h: *Sha, space: *const trace.Space, v: trace.Val) void {
    switch (v) {
        .t => |t| {
            addr(h, space, t.ptr);
            put(h, " {t} {d} {any} {any};", .{ t.dtype, t.dims, t.shape, t.stride });
        },
        .list => |l| {
            put(h, "[{d}", .{l.len});
            for (l) |u| val(h, space, u);
            h.update("]");
        },
        .int => |x| put(h, "i{d};", .{x}),
        .f32 => |x| put(h, "f{d};", .{x}),
        .bool => |x| put(h, "b{};", .{x}),
        .none => h.update("none;"),
    }
}

/// The trace's digest, addresses by buffer name (two ranks' or two trees' traces compare).
pub fn digest(space: *const trace.Space, events: []const trace.Event) [64]u8 {
    var h = Sha.init(.{});
    for (events) |e| {
        put(&h, "{t} {s} {any}|", .{ e.op, e.name, e.grid });
        for (e.args) |a| {
            put(&h, "{s}=", .{a.name});
            switch (a.value) {
                .ptr => |p| {
                    put(&h, "{s}", .{p.ty});
                    addr(&h, space, p.addr);
                },
                .i32 => |x| put(&h, "i32 {d}", .{x}),
                .f32 => |x| put(&h, "f32 {x}", .{@as(u32, @bitCast(x))}),
                .u64 => |x| put(&h, "u64 {d}", .{x}),
            }
            h.update(",");
        }
        for (e.consts) |c| {
            put(&h, "{s}:", .{c.name});
            if (c.int) |x| put(&h, "{d}", .{x});
            if (c.f32) |x| put(&h, "f{x}", .{@as(u32, @bitCast(x))});
            h.update(",");
        }
        for (e.vals) |v| val(&h, space, v);
        h.update("\n");
    }
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}

fn ids(comptime n: usize, base: u32, stride: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*p, i| p.* = base + stride * @as(u32, @intCast(i));
    return out;
}

fn request(b: *gl.Lanes(Serve), prompt: []const u32, max_new: u32, cancel: ?usize) !usize {
    var cfg = try lanes.Config.init(gpa, b.facts(), 1, 0);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer core.deinit();
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = max_new, .eos = &.{154_820}, .drafts = false });
    defer s.deinit(gpa);
    try core.addStream(&s);
    var rounds: usize = 0;
    while (core.activeCount() > 0) : (rounds += 1) {
        if (cancel) |n| if (rounds == n) {
            core.discard(&s);
            break;
        };
        try core.step();
    }
    return s.emitted().len;
}

test "drafts off: the served launches of the pair test's scenario are qualification_qualification_e91d6b91-serve 841b3f1's (qualification_0c0ab476's binaries), by digest" {
    for (0..2) |rank| {
        const r = try rec.Rank.init(gpa, rank, null);
        defer r.deinit(gpa);
        var f = r.recording();
        var s: Serve = .{ .e = r.e, .f = &f };
        var b: gl.Lanes(Serve) = .{ .steps = &s, .capacity = r.e.st.capacity, .reach = r.e.st.capacity };
        const first = ids(37, 1000, 7);
        const cancelled = ids(20, 3000, 11);
        const second = ids(9, 5000, 3);
        try testing.expectEqual(@as(usize, 5), try request(&b, &first, 5, null));
        _ = try request(&b, &cancelled, 40, 3);
        try testing.expectEqual(@as(usize, 3), try request(&b, &second, 3, null));
        try f.flush();
        const d = digest(&r.space, r.t.events.items);
        std.debug.print("OFFTRACE rank {d}: {d} launches, sha256 {s}\n", .{ rank, r.t.events.items.len, &d });
        if (want[rank]) |w| try testing.expectEqualStrings(w, &d);
    }
}
