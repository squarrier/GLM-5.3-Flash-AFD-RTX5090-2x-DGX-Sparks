//! Helios: an unquiescent hybrid owner is process-lifetime quarantine, not a normal nullable refusal.
//! Returning through std.process.Init's SafeAllocator would reclaim retained watcher/frame owners.
const std = @import("std");
/// Do not free user allocators, drivers, registered buffers or accepted-partial
/// owners on this path. Preserve them through final process exit; the external
/// owner must still prove actual native/physical/lease closure. This is never a
/// semantic-clean result or an authority to retry/release a running generation.
pub fn fail(problem: []const u8) noreturn {
    std.debug.print("error: hybrid retained process-lifetime quarantine: {s}; exiting1 before allocator teardown\n", .{problem});
    std.process.exit(1);
}
