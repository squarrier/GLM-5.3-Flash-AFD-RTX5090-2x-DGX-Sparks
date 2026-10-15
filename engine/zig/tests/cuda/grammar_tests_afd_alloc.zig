//! Exhaustive socket-free Bound -> Serve -> Attn request failure cleanup.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_lanes");
const data = @import("grammar_tests_data");
pub fn run(comptime Bound: type) !void {
    const a = std.testing.allocator;
    var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
    var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
    rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker();
    var f = glm.fwd.Fwd.recording(&rec.m, &rec.t);
    rec.at.absorb = true;
    var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f };
    var bound = Bound{ .s = &serve };
    var adapter = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560 };
    var cfg = try lanes.Config.init(a, .{ .mtp = true }, 1, 8); defer cfg.deinit(a);
    var failed: usize = 0;
    var completed = false;
    for (0..512) |index| {
        var allocator = std.testing.FailingAllocator.init(a, .{ .fail_index = index });
        adapter.frame_allocator = allocator.allocator(); values.cpu.gpa = allocator.allocator();
        var rules = data.Fake{}; rules.init(rec.m.vocab);
        const result = data.allocationTransaction(allocator.allocator(), adapter.backend(), &cfg, &rules);
        rules.rules.close();
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        try std.testing.expect(adapter.owner == null and !serve.open and rec.at.deferred_logits == null and rec.at.absorb);
        try std.testing.expectEqual(@as(usize, 0), rec.e.st.mtp_len);
        try std.testing.expectEqual(allocator.allocated_bytes, allocator.freed_bytes);
        if (result) |_| { try std.testing.expect(!allocator.has_induced_failure); completed = true; break; }
        else |err| { try std.testing.expectEqual(error.OutOfMemory, err); try std.testing.expect(allocator.has_induced_failure); failed += 1; }
    }
    try std.testing.expect(completed and failed > 0);
    adapter.frame_allocator = a; values.cpu.gpa = a;
    var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(a, &cfg, adapter.backend(), clock.clock()); defer core.deinit();
    var plain = try lanes.Stream.init(a, .{ .id = "plain-after-oom-sweep", .prompt = &.{ 3, 4 }, .max_new = 1, .drafts = false }); defer plain.deinit(a);
    try core.addStream(&plain); try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
    try std.testing.expect(adapter.owner == null and !serve.open);
    std.debug.print("grammar_tests AFD allocation sweep: {d} injected OOMs, closed once, prefix faithful, ordinary reuse OK\n", .{failed});
}
