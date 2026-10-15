//! grammar_tests_c5 actual TP committed compute/clone faults; CPU Fwd and independent peer only.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const Driver = @import("grammar_tests_c_tp_test.zig").Driver;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const a = std.testing.allocator;
const Tap = struct {
    trace: *glm.trace.Trace, hits: usize = 0, cutoff: usize = 0,
    fn put(ptr: *anyopaque, name: []const u8, _: glm.trace.T, _: usize, _: usize, _: ?[]const u8) !void {
        const t: *Tap = @ptrCast(@alignCast(ptr));
        if (std.mem.eql(u8, name, "hidden")) { try t.trace.flush(); t.hits += 1; t.cutoff = t.trace.events.items.len; return error.UnknownComputeFault; }
    }
};
fn check(comptime id: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s} expected=true observed=false\n", .{id}); return error.CommittedFaultRuntimeViolation; }
}
pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const Case = enum { reset, compute, clone, peer_clone, step_compute, step_commit, positive };
    for (std.meta.tags(Case)) |case| {
        const second = case == .step_compute or case == .step_commit;
        const r0 = try glm.grammar_tests_rec.tp(a, 0); defer r0.deinit(a);
        const r1 = try glm.grammar_tests_rec.tp(a, 1); defer r1.deinit(a);
        var f0 = r0.recording(); var f1 = r1.recording();
        var v0 = try data.Data.init(r0.m.vocab, 2); defer v0.deinit(); v0.set();
        var v1 = try data.Data.init(r1.m.vocab, 2); defer v1.deinit(); v1.set();
        r0.e.smp.masked_cpu = &v0.cpu; r1.e.smp.masked_cpu = &v1.cpu;
        r0.t.picker = v0.picker(); r1.t.picker = v1.picker();
        var s0 = glm.serve.Serve{ .e = r0.e, .f = &f0, .absorb = true };
        var s1 = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
        var sync = try Sync.init(a, 1, 2, 61); defer sync.deinit(); sync.install();
        var driver = Driver{ .serve = &s1, .session = &sync.session };
        var world = glm.ranks.Pair{ .gpa = a, .io = std.testing.io, .rank = 0 };
        var armed = pair.Armed(glm.serve.Serve){ .steps = &s0, .pair = &world };
        var backend = gl.Lanes(@TypeOf(armed)){ .steps = &armed, .capacity = 2560, .reach = 2560, .frame_allocator = a,
            .structured_session = &sync.session, .mirror = .{ .ctx = &driver, .send = Driver.legacy,
                .structured_driver_cpu = .{ .open = Driver.open, .operation = Driver.operation, .terminate = Driver.terminate } } };
        var host = Host{ .gpa = a, .gpu = null, .family = &backend, .release = struct { fn noop(_: *anyopaque) void {} }.noop,
            .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
        const actual = wrap(&host);
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, actual, clock.clock()); defer core.deinit();
        var rules = data.Fake{}; rules.init(r0.m.vocab);
        var stream = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = &.{3,4}, .max_new = 5, .rules = &rules.rules }); defer stream.deinit(a);
        if (case == .positive) {
            try core.addStream(&stream); while (!stream.finished) try core.step();
            try std.testing.expectEqualSlices(u32, &.{7, 8}, stream.emitted());
            try check("FreshSessionPositiveRuntimeViolation", sync.session.status == .ok and driver.releases == 1 and driver.outcome == null);
            var plain = try lanes.Stream.init(a, .{ .id = "fresh-plain", .prompt = &.{3,4}, .max_new = 1 }); defer plain.deinit(a);
            try actual.prefill(&plain); try std.testing.expectEqual(@as(u32, 2), try actual.read(try actual.first(&plain, 2))); actual.release(&plain);
            std.debug.print("CELL grammar_tests_c5 TP fresh_session=PASS actual_ordinary=PASS\n", .{});
            continue;
        }
        if (second) try core.addStream(&stream);
        var tap = Tap{ .trace = &r0.t };
        var fault = glm.trace.Trace.Fault{ .name = if (case == .reset) "torch.zero_" else if (case == .step_commit) "torch.fill_" else "torch.clone" };
        if (case == .compute or case == .step_compute) r0.e.tap = .{ .ctx = &tap, .tensor = Tap.put }
        else if (case == .peer_clone) r1.t.test_fault = &fault else r0.t.test_fault = &fault;
        var outcome: ?anyerror = null;
        if (second) {
            const mask = try rules.rules.next(a); defer a.free(mask);
            var sampled: [1]u32 = undefined;
            var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
            const windows = [_]lanes.backend.Window{.{ .stream = &stream, .pending = stream.pending.?, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3}, .masks = mask }};
            actual.verify(&windows, &out) catch |err| { outcome = err; };
        } else actual.prefill(&stream) catch |err| { outcome = err; };
        // Snapshot BEFORE release: cleanup or the two-second timeout cannot stand in for immediate fail-stop.
        const immediate = outcome != null and outcome.? == error.ControlPoison and sync.session.status == .poison and sync.session.pending == .poison and !r0.e.structured_logits_valid;
        const hit = if (case == .compute or case == .step_compute) tap.hits else fault.hits;
        lanes.structured.release(&core, &stream);
        try check("ImmediateCommittedPoisonWakeRuntimeViolation", immediate);
        try check("ActualFaultSeamRuntimeViolation", hit == 1 and (driver.outcome == null or driver.outcome.? != error.PeerWaitTimeout));
        const failed_trace = if (case == .peer_clone) &r1.t else &r0.t;
        try failed_trace.flush();
        try check("NoLaterCommitOrPickRuntimeViolation", failed_trace.events.items.len == (if (case == .compute or case == .step_compute) tap.cutoff else fault.cutoff));
        try check("BackendReleaseOrderRuntimeViolation", driver.releases == 1 and driver.thread == null and backend.owner == null);
        try check("DeferredLogitLeaseRuntimeViolation", !s0.open and !s1.open and !r0.e.structured_logits_valid and !r1.e.structured_logits_valid);
        try check("UnknownFaultOutputRuntimeViolation", stream.emitted().len == @intFromBool(second));
        const stages0 = data.count(&r0.t, "staged.record"); const stages1 = data.count(&r1.t, "staged.record");
        var next = try lanes.Stream.init(a, .{ .id = "same-poison", .prompt = &.{3,4}, .max_new = 1 }); defer next.deinit(a);
        try std.testing.expectError(error.ControlPoison, actual.prefill(&next));
        try check("PoisonedBackendNoWorkRuntimeViolation", data.count(&r0.t, "staged.record") == stages0 and data.count(&r1.t, "staged.record") == stages1 and driver.legacy_calls == 0);
        std.debug.print("CELL grammar_tests_c5 TP fault={s} immediate_poison=PASS peers=2 exactly_once=PASS same_backend=REFUSED hardware=NOT_RUN\n", .{@tagName(case)});
    }
}
