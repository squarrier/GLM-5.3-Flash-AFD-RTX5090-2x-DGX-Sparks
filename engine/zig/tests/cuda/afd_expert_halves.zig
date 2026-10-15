//! qualification_f504f9df: TF_GLM_HALVES_KERNEL on an expert half (cuda_halves_kernel.zig), read once by `command` beside X1's switches:
//! a bad value, or one whose kernel this tree does not build, refuses the run before anything loads; set, one INFO
//! line. serve's settings carry it, so the attention and the other half must have the same value (SettingsDiffer).
const std = @import("std");
const glm = @import("glm_cuda");

pub fn halvesKernel(half: u32) !glm.halves_kernel.Kernel {
    const k = glm.halves_kernel.fromEnv() catch |e| {
        std.debug.print("INFO afd expert {d}: TF_GLM_HALVES_KERNEL takes k4 or g53kt3r (or 0, empty, unset)\n", .{half});
        return e;
    };
    if (!built(k)) {
        std.debug.print("INFO afd expert {d}: TF_GLM_HALVES_KERNEL={s}: this tree does not build that kernel\n", .{ half, k.setting() });
        return error.HalvesKernelNotBuilt;
    }
    if (k.on()) std.debug.print("INFO afd expert {d}: TF_GLM_HALVES_KERNEL={s}: every window of a prompt forward (chunk, slice, pair) takes it; the settings carry halves={s}\n", .{ half, k.setting(), k.setting() });
    return k;
}

/// The prompt-window kernels this tree's halves run, each card's own line.
pub fn built(k: glm.halves_kernel.Kernel) bool {
    return switch (k) {
        .x1 => true,
        // qualification_e4ad04cd: Mia's EXL3 prompt kernel
        .k4 => true,
        // qualification_1b9bf461: glm53f-rank's kernels with KT3R's schedule
        .g53kt3r => true,
    };
}

/// qualification_c863b626: with a halves kernel set, X1's SEQ, PGROUP and MMA (read beside it) do not apply to the windows it takes
/// (cuda_afd_route.zig: it takes them before X1's launches); they stay on X1's windows. The run is not refused (the
/// lane's line keeps SEQ set); one INFO line says so when any of the three is set, else null.
pub fn overX1(buf: []u8, half: u32, k: glm.halves_kernel.Kernel, p: glm.exl3.experts.Prompt) ?[]const u8 {
    if (!k.on() or !(p.seq or p.pgroup or p.mma)) return null;
    const seq: []const u8 = if (p.seq) " TF_GLM_EXL3_SEQ" else "";
    const pg: []const u8 = if (p.pgroup) " TF_GLM_EXL3_PGROUP" else "";
    const mma: []const u8 = if (p.mma) " TF_GLM_EXL3_MMA" else "";
    return std.fmt.bufPrint(buf, "INFO afd expert {d}: TF_GLM_HALVES_KERNEL={s} with{s}{s}{s}: those apply to none of the windows {s} takes (every window of a prompt forward at a position it serves); they stay on X1's windows (eager ones, other positions)", .{ half, k.setting(), seq, pg, mma, k.setting() }) catch null;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "qualification_f504f9df: TF_GLM_HALVES_KERNEL in the expert: unset or 0 is today's path; a bad value, or one this tree does not build, refuses the run" {
    defer _ = unsetenv("TF_GLM_HALVES_KERNEL");
    _ = unsetenv("TF_GLM_HALVES_KERNEL");
    try std.testing.expectEqual(glm.halves_kernel.Kernel.x1, try halvesKernel(0));
    _ = setenv("TF_GLM_HALVES_KERNEL", "0", 1);
    try std.testing.expectEqual(glm.halves_kernel.Kernel.x1, try halvesKernel(0));
    _ = setenv("TF_GLM_HALVES_KERNEL", "g53", 1);
    try std.testing.expectError(error.BadHalvesKernel, halvesKernel(1));
    for ([_]glm.halves_kernel.Kernel{ .k4, .g53kt3r }) |k| {
        _ = setenv("TF_GLM_HALVES_KERNEL", @tagName(k), 1);
        if (built(k)) try std.testing.expectEqual(k, try halvesKernel(1)) else try std.testing.expectError(error.HalvesKernelNotBuilt, halvesKernel(1));
    }
}

test "qualification_c863b626: both values built (no HalvesKernelNotBuilt); with a halves kernel and X1's SEQ, PGROUP or MMA, one INFO line, not a refusal" {
    defer _ = unsetenv("TF_GLM_HALVES_KERNEL");
    for ([_][:0]const u8{ "k4", "g53kt3r" }) |v| {
        _ = setenv("TF_GLM_HALVES_KERNEL", v, 1);
        const k = try halvesKernel(0);
        try std.testing.expect(built(k) and k.on());
        try std.testing.expectEqualStrings(v, k.setting());
    }
    for ([_][:0]const u8{ "K4", "g53", "1", "on", "k4,g53kt3r" }) |v| {
        _ = setenv("TF_GLM_HALVES_KERNEL", v, 1);
        try std.testing.expectError(error.BadHalvesKernel, halvesKernel(0));
    }
    var b: [512]u8 = undefined;
    const P = glm.exl3.experts.Prompt;
    const x1 = try P.parse("1", .{ null, null, null, null });
    var fs = x1; // qualification_969df694's f-s line: X1 + SEQ
    fs.seq = true;
    var fm = fs; // + PGROUP + MMA (qualification_946d4b77's seq form)
    fm.pgroup = true;
    fm.mma = true;
    try std.testing.expect(overX1(&b, 0, .x1, fm) == null); // unset: nothing to say
    try std.testing.expect(overX1(&b, 0, .k4, x1) == null); // X1 alone: K4 takes its windows, nothing else to say
    try std.testing.expect(overX1(&b, 1, .g53kt3r, .{}) == null);
    try std.testing.expectEqualStrings("INFO afd expert 1: TF_GLM_HALVES_KERNEL=g53kt3r with TF_GLM_EXL3_SEQ: those apply to none of the windows g53kt3r takes (every window of a prompt forward at a position it serves); they stay on X1's windows (eager ones, other positions)", overX1(&b, 1, .g53kt3r, fs).?);
    try std.testing.expectEqualStrings("INFO afd expert 0: TF_GLM_HALVES_KERNEL=k4 with TF_GLM_EXL3_SEQ TF_GLM_EXL3_PGROUP TF_GLM_EXL3_MMA: those apply to none of the windows k4 takes (every window of a prompt forward at a position it serves); they stay on X1's windows (eager ones, other positions)", overX1(&b, 0, .k4, fm).?);
}
