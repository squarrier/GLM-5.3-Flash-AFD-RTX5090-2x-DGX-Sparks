//! TF_GLM_HALVES_KERNEL (qualification_f504f9df; Scott, 2026-10-09: "Both, then pick the faster one that passes the quality check"):
//! the expert halves' kernel for prompt windows, output-changing, default off. Unset, empty or 0 keeps today's path
//! (X1 or the plain launches); `k4` is Mia's EXL3 prompt kernel (qualification_e4ad04cd), `g53kt3r` Hugh Madden's glm53f-rank kernels
//! with KT3R's schedule (qualification_1b9bf461). All three roles read it once before anything loads and carry it in the exchange's
//! settings (` halves=<value>`), so a node with another value fails the handshake's agreement (SettingsDiffer).
//! Which windows take it is python 2.15's rule (experts_node.py:224): every window of a forward whose mode is not
//! eager (chunk, slice, pair, lanes), at any row count.

const std = @import("std");

pub const env_name = "TF_GLM_HALVES_KERNEL";

pub const Kernel = enum {
    x1,
    k4,
    g53kt3r,

    /// The settings' value: empty for today's path (the text stays as it was), else the switch's value.
    pub fn setting(k: Kernel) []const u8 {
        return switch (k) {
            .x1 => "",
            else => @tagName(k),
        };
    }

    pub fn on(k: Kernel) bool {
        return k != .x1;
    }
};

/// Unset, empty or `0`: today's path; `k4` or `g53kt3r`; anything else is refused (BadHalvesKernel).
pub fn parse(v: ?[]const u8) error{BadHalvesKernel}!Kernel {
    const s = v orelse return .x1;
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return .x1;
    if (std.mem.eql(u8, s, "k4")) return .k4;
    if (std.mem.eql(u8, s, "g53kt3r")) return .g53kt3r;
    return error.BadHalvesKernel;
}

/// The switch from the environment, read once by each role before anything loads.
pub fn fromEnv() error{BadHalvesKernel}!Kernel {
    return parse(if (std.c.getenv(env_name)) |p| std.mem.span(p) else null);
}

test "TF_GLM_HALVES_KERNEL: unset, empty and 0 are today's path; k4 and g53kt3r; anything else is refused" {
    try std.testing.expectEqual(Kernel.x1, try parse(null));
    try std.testing.expectEqual(Kernel.x1, try parse(""));
    try std.testing.expectEqual(Kernel.x1, try parse("0"));
    try std.testing.expectEqual(Kernel.k4, try parse("k4"));
    try std.testing.expectEqual(Kernel.g53kt3r, try parse("g53kt3r"));
    for ([_][]const u8{ "1", "x1", "K4", "k4 ", " k4", "g53", "g53+kt3r", "kt3r", "on", "true", "00" }) |v| try std.testing.expectError(error.BadHalvesKernel, parse(v));
    try std.testing.expectEqualStrings("", Kernel.x1.setting());
    try std.testing.expectEqualStrings("k4", Kernel.k4.setting());
    try std.testing.expectEqualStrings("g53kt3r", Kernel.g53kt3r.setting());
    try std.testing.expect(!Kernel.x1.on() and Kernel.k4.on() and Kernel.g53kt3r.on());
}
