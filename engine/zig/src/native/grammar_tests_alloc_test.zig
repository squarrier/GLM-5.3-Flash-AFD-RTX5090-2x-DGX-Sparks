//! Exhaustive socket-free TP request allocation-failure cleanup sweep.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const a = std.testing.allocator;
    const r = try glm.grammar_tests_rec.tp(a, 0); defer r.deinit(a);
    var values = try data.Data.init(r.m.vocab, r.m.world); defer values.deinit(); values.set();
    r.e.smp.masked_cpu = &values.cpu; r.t.picker = values.picker();
    var f = r.recording();
    var serve = glm.serve.Serve{ .e = r.e, .f = &f, .absorb = true };
    var world = glm.ranks.Pair{ .gpa = a, .io = std.testing.io, .rank = 0 };
    var armed = pair.Armed(glm.serve.Serve){ .steps = &serve, .pair = &world };
    var adapter = gl.Lanes(@TypeOf(armed)){ .steps = &armed, .capacity = 2560, .reach = 2560 };
    var h = Host{ .gpa = a, .gpu = null, .family = &adapter, .release = noop, .inner = adapter.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
    const b = wrap(&h);
    var cfg = try lanes.Config.init(a, .{ .mtp = true }, 1, 8); defer cfg.deinit(a);
    var failed: usize = 0;
    var completed = false;
    for (0..512) |index| {
        var allocator = std.testing.FailingAllocator.init(a, .{ .fail_index = index });
        adapter.frame_allocator = allocator.allocator(); values.cpu.gpa = allocator.allocator();
        var rules = data.Fake{}; rules.init(r.m.vocab);
        const result = data.allocationTransaction(allocator.allocator(), b, &cfg, &rules);
        rules.rules.close();
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        try std.testing.expect(adapter.owner == null and !serve.open and !serve.deferred);
        try std.testing.expectEqual(@as(usize, 0), r.e.st.mtp_len);
        data.witness("MaskLeaseRuntimeViolation.N6-retained-round-arena", allocator.allocated_bytes, allocator.freed_bytes);
        try std.testing.expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        if (result) |_| { try std.testing.expect(!allocator.has_induced_failure); completed = true; break; }
        else |err| { try std.testing.expectEqual(error.OutOfMemory, err); try std.testing.expect(allocator.has_induced_failure); failed += 1; }
    }
    try std.testing.expect(completed and failed > 0);
    adapter.frame_allocator = a; values.cpu.gpa = a;
    var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(a, &cfg, b, clock.clock()); defer core.deinit();
    var plain = try lanes.Stream.init(a, .{ .id = "plain-after-oom-sweep", .prompt = &.{ 3, 4 }, .max_new = 1, .drafts = false }); defer plain.deinit(a);
    try core.addStream(&plain); try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
    try std.testing.expect(adapter.owner == null and !serve.open);
    std.debug.print("grammar_tests TP allocation sweep: {d} injected OOMs, closed once, prefix faithful, ordinary reuse OK\n", .{failed});
}
fn noop(_: *anyopaque) void {}
