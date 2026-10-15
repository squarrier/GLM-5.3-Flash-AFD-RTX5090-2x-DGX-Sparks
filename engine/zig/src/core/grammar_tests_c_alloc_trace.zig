//! GPT-6.1 Sol via Hermes: nonallocating fixture receipt of the first induced allocation boundary.
const std = @import("std");
pub const Stage = enum { submit_job, submit_queue, admitted_append, compiler_rules, stream_init, core_add, first_mask, round, telemetry };
pub const Trace = struct {
    failing: *std.testing.FailingAllocator,
    stage: Stage = .submit_job,
    failed_stage: ?Stage = null,
    submit_allocs: usize = 0,
    pub fn allocator(t: *Trace) std.mem.Allocator {
        return .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    pub fn set(t: *Trace, stage: Stage) void { t.stage = stage; }
    fn self(p: *anyopaque) *Trace { return @ptrCast(@alignCast(p)); }
    fn alloc(p: *anyopaque, n: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const t = self(p);
        const stage: Stage = if (t.stage == .submit_job) (if (t.submit_allocs == 0) .submit_job else .submit_queue) else t.stage;
        if (t.stage == .submit_job) t.submit_allocs += 1;
        const result = t.failing.allocator().rawAlloc(n, alignment, ra);
        if (result == null and t.failed_stage == null) t.failed_stage = stage;
        return result;
    }
    fn resize(p: *anyopaque, b: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) bool {
        return self(p).failing.allocator().rawResize(b, alignment, n, ra);
    }
    fn remap(p: *anyopaque, b: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        return self(p).failing.allocator().rawRemap(b, alignment, n, ra);
    }
    fn free(p: *anyopaque, b: []u8, alignment: std.mem.Alignment, ra: usize) void {
        self(p).failing.allocator().rawFree(b, alignment, ra);
    }
};
