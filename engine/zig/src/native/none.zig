// Modified for V4 by Scott Quarrier with AI assistance: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! No native backend: the engine-less build the server's tests and parity suite run (zig/tests/server/build.sh).
const std = @import("std");
const api = @import("engine_api");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{};
pub const families: []const api.Family = &.{};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

pub fn open(a: Allocator, _: Allocator, _: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (comptime @import("builtin").is_test) if (StopTest.open) |hook| return hook(o);
    problem.* = try std.fmt.allocPrint(a, "the native engine has no backend for {s} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6", .{o.model_type});
    return null;
}

// Explicit CPU native-open fence; release none backend remains unchanged.
pub const StopTest=if(@import("builtin").is_test)struct{pub var open:?*const fn(api.Open)anyerror!?api.Opened=null;}else struct{};
