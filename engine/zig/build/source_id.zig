//! rank_fingerprint: `-Dsource_id=<commit>`, the source identity a build stamps into the CUDA runtime's kernel_options
//! (cuda.fingerprint.source_id, TF_GLM_FINGERPRINT's `src`). Unset by default: bins and kits build as before (`unset`).
const std = @import("std");

pub const name = "source_id";
pub const unset = "unset";

/// Declared once (build.zig); a value other than 1-64 of [0-9A-Za-z._-] stops the build with the rule.
pub fn declare(b: *std.Build) void {
    const v = b.option([]const u8, name, "the source commit TF_GLM_FINGERPRINT's identity carries as src (default unset)") orelse return;
    if (!valid(v)) std.debug.panic("-Dsource_id takes 1-64 of [0-9A-Za-z._-], not \"{s}\"", .{v});
}

pub fn valid(v: []const u8) bool {
    if (v.len == 0 or v.len > 64) return false;
    for (v) |c| if (!(std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-')) return false;
    return true;
}

/// What a runtime builder writes as `source_id` (`unset` without the flag).
pub fn get(b: *std.Build) []const u8 {
    const v = b.user_input_options.get(name) orelse return unset;
    return switch (v) {
        .scalar => |s| s,
        else => unset,
    };
}
