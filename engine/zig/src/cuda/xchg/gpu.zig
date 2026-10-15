//! fn_xchg.cu's launches for a node (qualification_12f4270a): `post` one message from device memory, `take` this node's incoming messages into device memory, and the lab's stand-in expert half.
//! The sequences live in a small device buffer, so the launches replay in a CUDA graph; every wait is bounded (Config.give_up_ns) and none traps.
const std = @import("std");
const abi = @import("../abi.zig");
const Driver = @import("../driver.zig").Driver;
const memory = @import("../memory.zig");
const module = @import("../module.zig");
const launch = @import("../launch.zig");
const Stream = @import("../stream.zig").Stream;
const kernels = @import("../kernels.zig");
const proto = @import("proto.zig");
const Node = @import("node.zig").Node;

pub const symbols = [4][:0]const u8{ "tf_xchg_post", "tf_xchg_take", "tf_xchg_lab_experts", "tf_xchg_take_striped" };

/// A message's destination in device memory and its bytes (a multiple of 16).
pub const Out = struct { ptr: abi.DevicePtr, bytes: usize };

/// Grid blocks for `bytes` of 16-byte units: about four units a thread of 512, at most 16 blocks.
pub fn blocksFor(bytes: usize) u32 {
    return @intCast(std.math.clamp(bytes / 16 / 2048, 1, 16));
}

/// qualification_17c70e37: the striped take's grid for these messages under `s`, or null when none of them goes in stripes (decode
/// windows: tf_xchg_take as before): `parts` blocks a source's stripe, about 64 blocks in all.
pub fn stripedGrid(s: proto.Settings, outs: []const Out) ?struct { parts: u32, blocks: u32 } {
    var nmax: u32 = 1;
    for (outs) |o| nmax = @max(nmax, proto.stripesOf(s.stripes, s.stripe_min, o.bytes));
    if (nmax < 2) return null;
    const pieces = nmax * @as(u32, @intCast(outs.len));
    const parts = @max(1, 64 / pieces);
    return .{ .parts = parts, .blocks = pieces * parts };
}

/// The checks a launch's arguments pass before they reach the GPU (shared with the CPU build's launcher).
pub fn checkPost(n: *const Node, in: u64, bytes: usize) !void {
    if (bytes == 0 or bytes % 16 != 0 or bytes > n.reg.slot or in % 16 != 0) return error.Invalid;
}

pub fn checkTake(n: *const Node, outs: []const Out) !void {
    const want: usize = if (n.role == .attn) proto.EXPERTS else 1;
    if (outs.len != want) return error.Invalid;
    for (outs) |o| if (o.bytes == 0 or o.bytes % 16 != 0 or o.bytes > n.reg.slot or o.ptr % 16 != 0) return error.Invalid;
}

pub const Gpu = struct {
    d: *const Driver,
    mod: module.Module,
    fns: [4]module.Function,
    state: memory.DeviceBuffer,

    pub fn init(d: *const Driver) !Gpu {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.fn_xchg);
        errdefer mod.unload();
        var fns: [4]module.Function = undefined;
        for (symbols, &fns) |name, *f| f.* = try mod.function(name);
        var state = try memory.DeviceBuffer.alloc(d, 16);
        errdefer state.free();
        try state.fill32(0, null);
        // the zeroes land before any stream's first launch (a legacy-stream fill may not have: qualification_6607f886's race class)
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        return .{ .d = d, .mod = mod, .fns = fns, .state = state };
    }

    pub fn deinit(g: *Gpu) void {
        g.state.free();
        g.mod.unload();
    }

    /// One message (`bytes` at `in`) into this node's send slot, then the doorbell its proxies poll; once the exchange has failed, neither (nothing more is sent).
    pub fn post(g: *const Gpu, n: *const Node, in: abi.DevicePtr, bytes: usize, s: Stream) !void {
        try checkPost(n, in, bytes);
        const blocks = blocksFor(bytes);
        var a: launch.Args = .{};
        a.add(in);
        a.add(@as(u32, @intCast(bytes)));
        a.add(n.mem.region_dev);
        a.add(n.reg.send(0));
        a.add(n.reg.ctrl());
        a.add(n.reg.slot);
        a.add(g.state.ptr);
        a.add(n.mem.status_dev);
        a.add(@as(i32, @intCast(blocks)));
        a.add(@as(u32, n.reg.slots)); // qualification_9e49592e: the region's slots a source (2, or TF_AFD_LANES=N's N)
        try launch.launch(g.fns[0], .{ .grid = .{ .x = blocks }, .block = .{ .x = 512 } }, s, &a);
    }

    /// This node's incoming messages (the attention: expert 0's and 1's replies; an expert: the request) into `outs`, after bounded waits; nothing is copied once the exchange has failed.
    /// qualification_17c70e37: with the settings' stripes, a message that came in stripes is copied stripe by stripe as each lands (tf_xchg_take_striped).
    pub fn take(g: *const Gpu, n: *const Node, outs: []const Out, s: Stream) !void {
        try checkTake(n, outs);
        if (stripedGrid(n.cfg.settings, outs)) |sg| return g.takeStriped(n, outs, sg.parts, sg.blocks, s);
        var total: usize = 0;
        for (outs) |o| total += o.bytes;
        const blocks = blocksFor(total);
        var a: launch.Args = .{};
        a.add(@as(i32, @intCast(outs.len)));
        a.add(n.src(0));
        a.add(n.src(1));
        a.add(outs[0].ptr);
        a.add(if (outs.len > 1) outs[1].ptr else outs[0].ptr);
        a.add(@as(u32, @intCast(outs[0].bytes)));
        a.add(@as(u32, @intCast(if (outs.len > 1) outs[1].bytes else 0)));
        a.add(n.mem.region_dev);
        a.add(n.reg.recv(0, 0));
        a.add(n.reg.flagBase());
        a.add(n.reg.slot);
        a.add(@as(i32, @intCast(n.cfg.settings.ports)));
        a.add(g.state.ptr);
        a.add(n.mem.status_dev);
        a.add(n.cfg.give_up_ns);
        a.add(@as(i32, @intCast(blocks)));
        a.add(@as(u32, n.reg.slots)); // qualification_9e49592e: the region's slots a source
        try launch.launch(g.fns[1], .{ .grid = .{ .x = blocks }, .block = .{ .x = 512 } }, s, &a);
    }

    /// qualification_17c70e37: tf_xchg_take_striped over `blocks` blocks, `parts` a source's stripe.
    fn takeStriped(g: *const Gpu, n: *const Node, outs: []const Out, parts: u32, blocks: u32, s: Stream) !void {
        var a: launch.Args = .{};
        a.add(@as(i32, @intCast(outs.len)));
        a.add(n.src(0));
        a.add(n.src(1));
        a.add(outs[0].ptr);
        a.add(if (outs.len > 1) outs[1].ptr else outs[0].ptr);
        a.add(@as(u32, @intCast(outs[0].bytes)));
        a.add(@as(u32, @intCast(if (outs.len > 1) outs[1].bytes else 0)));
        a.add(n.mem.region_dev);
        a.add(n.reg.recv(0, 0));
        a.add(n.reg.flagBase());
        a.add(n.reg.slot);
        a.add(@as(i32, @intCast(n.cfg.settings.ports)));
        a.add(n.cfg.settings.stripes);
        a.add(@as(u64, n.cfg.settings.stripe_min));
        a.add(@as(i32, @intCast(parts)));
        a.add(g.state.ptr);
        a.add(n.mem.status_dev);
        a.add(n.cfg.give_up_ns);
        a.add(@as(i32, @intCast(blocks)));
        a.add(@as(u32, n.reg.slots)); // qualification_7a4aba6d: the region's slots a source, as post's and take's (qualification_9e49592e)
        try launch.launch(g.fns[3], .{ .grid = .{ .x = blocks }, .block = .{ .x = 512 } }, s, &a);
    }

    /// The lab's stand-in expert half over a request in device memory: experts [lo, hi) of each row's picks, fp32 or bf16 partials into `out`.
    pub fn labExperts(g: *const Gpu, req: abi.DevicePtr, l: proto.Layout, rows: usize, lo: i32, hi: i32, bf16_out: bool, out: abi.DevicePtr, s: Stream) !void {
        const o = l.offsets(rows);
        var a: launch.Args = .{};
        a.add(req + o.normed);
        a.add(req + o.pick);
        a.add(req + o.wts);
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(l.d)));
        a.add(@as(i32, @intCast(l.s)));
        a.add(lo);
        a.add(hi);
        a.add(@as(i32, @intFromBool(bf16_out)));
        const blocks: u32 = @intCast(std.math.clamp(rows * l.d / 256, 1, 1024));
        try launch.launch(g.fns[2], .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, s, &a);
    }
};

test "launch shapes: one block for decode messages, at most 16 for a prompt chunk's" {
    try std.testing.expectEqual(@as(u32, 1), blocksFor(16));
    try std.testing.expectEqual(@as(u32, 1), blocksFor(8752));
    try std.testing.expectEqual(@as(u32, 8), blocksFor(256 << 10));
    try std.testing.expectEqual(@as(u32, 16), blocksFor(64 << 20));
}

test "qualification_17c70e37: the striped take's grid: none off or for decode messages; 64 blocks for a chunk's two replies or one request in 8 stripes" {
    const off: proto.Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var on = off;
    on.stripes = 8;
    const rep = 4 * 4096 * 1024;
    const two = [2]Out{ .{ .ptr = 0, .bytes = rep }, .{ .ptr = 0, .bytes = rep } };
    try std.testing.expect(stripedGrid(off, &two) == null);
    const g = stripedGrid(on, &two).?;
    try std.testing.expectEqual([2]u32{ 4, 64 }, [2]u32{ g.parts, g.blocks });
    const req = [1]Out{.{ .ptr = 0, .bytes = 8462592 }};
    const r = stripedGrid(on, &req).?;
    try std.testing.expectEqual([2]u32{ 8, 64 }, [2]u32{ r.parts, r.blocks });
    const decode = [2]Out{ .{ .ptr = 0, .bytes = 16384 }, .{ .ptr = 0, .bytes = 16384 } };
    try std.testing.expect(stripedGrid(on, &decode) == null); // tf_xchg_take, as before
    on.stripes = 16;
    const s16 = stripedGrid(on, &two).?;
    try std.testing.expectEqual([2]u32{ 2, 64 }, [2]u32{ s16.parts, s16.blocks });
}
