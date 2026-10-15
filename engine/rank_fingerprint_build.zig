//! rank_fingerprint: TF_GLM_FINGERPRINT's CPU tests alone (zig/build/rank_fingerprint_tests.zig's `standalone`), prose_filter_c06f-build.zig's form:
//! `zig build --build-file rank_fingerprint_build.zig focus|off [-Dsafe=true]`. `off` runs on the base tree too.
const std = @import("std");

pub fn build(b: *std.Build) void {
    @import("zig/build/rank_fingerprint_tests.zig").standalone(b);
}
