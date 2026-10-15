//! grammar_tests_c3 actual Host/Armed/Serve sampler-fault fail-stop; CPU doubles only.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const Driver = @import("grammar_tests_c_tp_test.zig").Driver;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const a = std.testing.allocator;
fn check(comptime id: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s} expected=true observed=false\n", .{id}); return error.UnknownFaultFailStopRuntimeViolation; }
}
pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    inline for ([_]bool{false, true}) |second| {
        const r0 = try glm.grammar_tests_rec.tp(a, 0); defer r0.deinit(a);
        const r1 = try glm.grammar_tests_rec.tp(a, 1); defer r1.deinit(a);
        var f0 = r0.recording(); var f1 = r1.recording();
        var v0 = try data.Data.init(r0.m.vocab, 2); defer v0.deinit(); v0.set();
        var v1 = try data.Data.init(r1.m.vocab, 2); defer v1.deinit(); v1.set();
        r0.e.smp.masked_cpu = &v0.cpu; r1.e.smp.masked_cpu = &v1.cpu;
        r0.t.picker = v0.picker(); r1.t.picker = v1.picker();
        var s0 = glm.serve.Serve{ .e = r0.e, .f = &f0, .absorb = true };
        var s1 = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
        var sync = try Sync.init(a, 23, 2, 61); defer sync.deinit(); sync.install();
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
        var stream = try lanes.Stream.init(a, .{ .id = "unknown-fault", .prompt = &.{3,4}, .max_new = 5, .rules = &rules.rules }); defer stream.deinit(a);
        if (second) try core.addStream(&stream);
        v0.cpu.fail = true; // Actual local rank0 GatherFailed while rank1 completes its independent sampler.
        var outcome: ?anyerror = null;
        if (second) { _ = core.step() catch |err| blk: { outcome = err; break :blk false; }; }
        else core.addStream(&stream) catch |err| { outcome = err; };
        std.debug.print("FAULT TP phase={s} outcome={s} session={t}\n", .{if (second) "second" else "first", if (outcome) |err| @errorName(err) else "none", sync.session.status});
        try check("UnknownFaultFailStopRuntimeViolation", outcome != null and outcome.? == error.ControlPoison and sync.session.status == .poison);
        try check("BackendReleaseOrderRuntimeViolation", driver.releases == 1 and driver.thread == null and backend.owner == null);
        try check("DeferredLogitLeaseRuntimeViolation", !s0.open and !s1.open and !r0.e.structured_logits_valid and !r1.e.structured_logits_valid);
        try check("UnknownFaultOutputRuntimeViolation", stream.emitted().len == @intFromBool(second) and v0.cpu.picks == @intFromBool(second));
        const stages0 = data.count(&r0.t, "staged.record"); const stages1 = data.count(&r1.t, "staged.record");
        var next = try lanes.Stream.init(a, .{ .id = "same-poisoned-backend", .prompt = &.{3,4}, .max_new = 1 }); defer next.deinit(a);
        var refused: ?anyerror = null;
        actual.prefill(&next) catch |err| { refused = err; };
        if (refused == null) actual.release(&next);
        try check("PoisonedBackendReuseRuntimeViolation", refused != null and refused.? == error.ControlPoison);
        try check("PoisonedBackendNoWorkRuntimeViolation", data.count(&r0.t, "staged.record") == stages0 and data.count(&r1.t, "staged.record") == stages1 and driver.legacy_calls == 0);
        std.debug.print("CELL grammar_tests_c3 TP sampler_fault={s} actual_chain=Host_Armed_Serve peers=2 poison=PASS same_backend_plain=REFUSED no_new_stage=PASS hardware=NOT_RUN\n", .{if (second) "second" else "first"});
    }
}
