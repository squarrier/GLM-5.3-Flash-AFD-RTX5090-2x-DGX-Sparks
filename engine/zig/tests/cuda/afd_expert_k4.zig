//! K4 on an expert half's GPU (qualification_e4ad04cd, TF_GLM_HALVES_KERNEL=k4). `opts`: python 2.15's prompt_switches, read once by
//! `command` after the switch (a bad TF_GLM_EXL3_PASS, _ORDER or _ROWROT refuses the run before anything loads; one
//! INFO line, python's). `Gpu.open`, after X1's block (opened for K4 when X1 is off: rowrot reads its shared suh rows;
//! X1's own switches stay as set): K4's image, its block (python's prompt_bufs, items and counts 0, order -1, as
//! python makes them) and each position's prompt_kernel_ready, with one INFO line.
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const ae = glm.afd_expert;
const afdk4 = ae.afdk4;
const k4 = glm.exl3.k4;

pub fn opts(half: u32) !k4.Opts {
    const o = k4.Opts.fromEnv() catch |e| {
        std.debug.print("INFO afd expert {d}: with TF_GLM_HALVES_KERNEL=k4, TF_GLM_EXL3_PASS takes 64 or 128 and TF_GLM_EXL3_ORDER and _ROWROT 0 or 1\n", .{half});
        return e;
    };
    std.debug.print("INFO afd expert {d}: K4: prompt windows' routed experts through MiaAI-Lab's prompt kernel (recipe patch 0004; pass {d}, order {s}, rowrot {s})\n", .{ half, o.pass, onOff(o.order), onOff(o.xrow) });
    return o;
}

fn onOff(b: bool) []const u8 {
    return if (b) "on" else "off";
}

pub const Gpu = struct {
    buf: ?cuda.DeviceBuffer = null,
    b: afdk4.Bufs = undefined,
    ready: []bool = &.{},
    on: afdk4.On = undefined,

    /// K4's image, block and readiness for the half `x` (its groups at `base`, its buffers `b`), X1's block `x1b`.
    pub fn open(k: *Gpu, gpa: std.mem.Allocator, d: *const cuda.Driver, set: *glm.exl3.kernels.Set, x: *const ae.Half, b: *const ae.Bufs, base: []const u64, experts: usize, o: k4.Opts, x1b: *const ae.X1Bufs, xrow: []const bool, half: u32) !void {
        errdefer k.close(gpa);
        try set.loadK4(d);
        k.b = afdk4.Bufs.init(b, experts);
        const n = try afdk4.place(&k.b, 0);
        k.buf = try cuda.DeviceBuffer.alloc(d, n);
        _ = try afdk4.place(&k.b, k.buf.?.ptr);
        const init = try gpa.alloc(u8, n); // python: items and counts torch.zeros, order torch.full(-1)
        defer gpa.free(init);
        @memset(init, 0);
        for ([_]@TypeOf(k.b.decode_order){ k.b.decode_order, k.b.pref_order }) |ord| @memset(init[ord.ptr - k.buf.?.ptr ..][0..ord.bytes()], 0xff);
        try k.buf.?.upload(0, init);
        k.ready = try gpa.alloc(bool, x.layers.len);
        var taken: usize = 0;
        for (k.ready, 0..) |*r, pos| {
            r.* = afdk4.ready(x, pos, base[pos], b.decode.slots);
            taken += @intFromBool(r.*);
        }
        k.on = .{ .opts = o, .bufs = &k.b, .ready = k.ready, .x1b = x1b, .xrow = xrow };
        std.debug.print("INFO afd expert {d}: K4: {d} of {d} positions take it (prompt_kernel_ready); plan items {d} and {d} slots; block {d} KiB\n", .{ half, taken, k.ready.len, k.b.decode_order.shape[0], k.b.pref_order.shape[0], n >> 10 });
    }

    pub fn close(k: *Gpu, gpa: std.mem.Allocator) void {
        if (k.buf) |*d| d.free();
        k.buf = null;
        if (k.ready.len > 0) gpa.free(k.ready);
        k.ready = &.{};
    }

    /// The Device's K4 (null: off).
    pub fn device(k: *const Gpu) ?*const afdk4.On {
        return if (k.buf != null) &k.on else null;
    }
};

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_e4ad04cd: K4's switches in the expert: python's defaults unset; a bad TF_GLM_EXL3_PASS, _ORDER or _ROWROT refuses the run" {
    const names = [_][:0]const u8{ "TF_GLM_EXL3_PASS", "TF_GLM_EXL3_ORDER", "TF_GLM_EXL3_ROWROT" };
    defer for (names) |n| {
        _ = unsetenv(n);
    };
    for (names) |n| _ = unsetenv(n);
    try std.testing.expectEqual(k4.Opts{}, try opts(0));
    _ = setenv("TF_GLM_EXL3_PASS", "128", 1);
    _ = setenv("TF_GLM_EXL3_ORDER", "0", 1);
    try std.testing.expectEqual(k4.Opts{ .pass = 128, .order = false, .xrow = true }, try opts(1));
    _ = setenv("TF_GLM_EXL3_PASS", "96", 1);
    try std.testing.expectError(error.BadExl3Prompt, opts(1));
    _ = setenv("TF_GLM_EXL3_PASS", "64", 1);
    _ = setenv("TF_GLM_EXL3_ROWROT", "on", 1);
    try std.testing.expectError(error.BadExl3Prompt, opts(0));
}
