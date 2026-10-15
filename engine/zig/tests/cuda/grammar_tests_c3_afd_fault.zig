//! grammar_tests_c3 actual Bound/Serve/Attn sampler-fault fail-stop; expert sidecars MOCK.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Trio = @import("grammar_tests_c_afd_chunks.zig").Trio;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const a = std.testing.allocator;
fn check(comptime id: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s} expected=true observed=false\n", .{id}); return error.UnknownFaultFailStopRuntimeViolation; }
}
pub fn run(comptime Bound: type) !void {
    inline for ([_]bool{false, true}) |second| {
        var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
        var f = rec.recording();
        var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
        rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker(); rec.at.absorb = true;
        var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f };
        var bound = Bound{ .s = &serve };
        var sync = try Sync.init(a, 29, 3, 71); defer sync.deinit(); sync.install();
        const mask = try a.alloc(u32, (rec.m.vocab+31)/32); defer a.free(mask);
        var trio = Trio{ .sync = &sync, .inner = sync.session.hooks.?, .mask = mask }; trio.install();
        var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a, .structured_session = &sync.session,
            .mirror = .{ .ctx = &trio, .structured_driver_cpu = .{ .open = Trio.driverOpen, .operation = Trio.operation, .terminate = Trio.terminate } } };
        const actual = backend.backend();
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, actual, clock.clock()); defer core.deinit();
        var rules = data.Fake{}; rules.init(rec.m.vocab);
        var stream = try lanes.Stream.init(a, .{ .id = "unknown-fault", .prompt = &.{3,4}, .max_new = 5, .rules = &rules.rules }); defer stream.deinit(a);
        if (second) try core.addStream(&stream);
        values.cpu.fail = true;
        var outcome: ?anyerror = null;
        if (second) { _ = core.step() catch |err| blk: { outcome = err; break :blk false; }; }
        else core.addStream(&stream) catch |err| { outcome = err; };
        std.debug.print("FAULT AFD phase={s} outcome={s} session={t}\n", .{if (second) "second" else "first", if (outcome) |err| @errorName(err) else "none", sync.session.status});
        try check("UnknownFaultFailStopRuntimeViolation", outcome != null and outcome.? == error.ControlPoison and sync.session.status == .poison);
        try check("BackendReleaseOrderRuntimeViolation", trio.releases == 1 and backend.owner == null and trio.threads[0] == null and trio.threads[1] == null);
        try check("DeferredLogitLeaseRuntimeViolation", !serve.open and serve.len == 0 and bound.len == 0 and rec.at.deferred_logits == null and rec.at.structured_control == null);
        try check("UnknownFaultOutputRuntimeViolation", stream.emitted().len == @intFromBool(second) and values.cpu.picks == @intFromBool(second));
        try check("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
        const stages = data.count(&rec.t, "staged.record"); const issued = data.count(&rec.t, "afd.exchange");
        var next = try lanes.Stream.init(a, .{ .id = "same-poisoned-backend", .prompt = &.{3,4}, .max_new = 1 }); defer next.deinit(a);
        var refused: ?anyerror = null;
        actual.prefill(&next) catch |err| { refused = err; };
        if (refused == null) actual.release(&next);
        try check("PoisonedBackendReuseRuntimeViolation", refused != null and refused.? == error.ControlPoison);
        try check("PoisonedBackendNoWorkRuntimeViolation", data.count(&rec.t, "staged.record") == stages and data.count(&rec.t, "afd.exchange") == issued);
        std.debug.print("CELL grammar_tests_c3 AFD sampler_fault={s} actual_chain=Bound_Serve_Attn roles=3 sidecars=MOCK poison=PASS same_backend_plain=REFUSED no_new_provider=PASS hardware=NOT_RUN\n", .{if (second) "second" else "first"});
    }
}
