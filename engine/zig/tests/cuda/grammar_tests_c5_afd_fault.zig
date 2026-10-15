//! grammar_tests_c5 actual Bound/Serve/Attn compute/clone/provider faults; expert sidecars MOCK.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Trio = @import("grammar_tests_c_afd_chunks.zig").Trio;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const a = std.testing.allocator;
const Tap = struct {
    trace: *glm.trace.Trace, fail: bool = true, hits: usize = 0, cutoff: usize = 0,
    fn put(ptr: *anyopaque, name: []const u8, _: glm.trace.T, _: usize, _: usize, _: ?[]const u8) !void {
        const t: *Tap = @ptrCast(@alignCast(ptr));
        if (t.fail and std.mem.eql(u8, name, "hidden")) { try t.trace.flush(); t.hits += 1; t.cutoff = t.trace.events.items.len; return error.UnknownComputeFault; }
    }
};
const Cancel = struct {
    trace: *glm.trace.Trace,
    fn check(ptr: *anyopaque) bool { const x: *Cancel = @ptrCast(@alignCast(ptr)); return data.count(x.trace, "afd.exchange") >= 1; }
};
fn check(comptime id: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s} expected=true observed=false\n", .{id}); return error.CommittedFaultRuntimeViolation; }
}
pub fn run(comptime Bound: type) !void {
    const Case = enum { reset, compute, clone, step_compute, step_commit, tapped_provider, pair_provider, pair_drain, pair_mismatch, lanes3_provider, lanes3_drain, lanes4_provider, positive };
    for (std.meta.tags(Case)) |case| {
        const second = case == .step_compute or case == .step_commit;
        const paired = case == .pair_provider or case == .pair_drain or case == .pair_mismatch;
        const configured: u8 = if (case == .lanes3_provider or case == .lanes3_drain) 3 else if (case == .lanes4_provider) 4 else 0;
        const drain = case == .pair_drain or case == .lanes3_drain;
        var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
        var f = rec.recording();
        rec.provider.paired = paired or configured > 0; rec.provider.inflight = if (configured > 0) configured else 2; rec.provider.lanes = configured;
        var pair = glm.afd_pairs.Pairs.init(&rec.at); pair.agreed = true;
        var group: glm.afd_lanes.Lanes = undefined;
        if (paired) { try pair.walk(&rec.space, glm.trace.placeAll); rec.at.pairs = &pair; }
        if (configured > 0) { group = try glm.afd_lanes.Lanes.init(&rec.at); group.agreed = true; try group.walk(&rec.space, glm.trace.placeAll); rec.at.lanes = &group; }
        var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
        rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker(); rec.at.absorb = true;
        var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f }; var bound = Bound{ .s = &serve };
        var sync = try Sync.init(a, 1, 3, 71); defer sync.deinit(); sync.install();
        const masks = try a.alloc(u32, (rec.m.vocab+31)/32); defer a.free(masks);
        var trio = Trio{ .sync = &sync, .inner = sync.session.hooks.?, .mask = masks, .trace = &rec.t, .mismatch_take = case == .pair_mismatch }; trio.install();
        var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a, .structured_session = &sync.session,
            .mirror = .{ .ctx = &trio, .structured_driver_cpu = .{ .open = Trio.driverOpen, .operation = Trio.operation, .terminate = Trio.terminate } } };
        const actual = backend.backend();
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{}; var core = lanes.Engine.init(a, &cfg, actual, clock.clock()); defer core.deinit();
        var rules = data.Fake{}; rules.init(rec.m.vocab);
        var ids: [256]u32 = @splat(3); var cancel = Cancel{ .trace = &rec.t };
        const length: usize = if (configured > 0) @as(usize, configured)*64 else if (paired) 128 else 2;
        var stream = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = ids[0..length], .max_new = 5, .rules = &rules.rules,
            .cancel_check = if (drain) .{ .ptr = &cancel, .check = Cancel.check } else null }); defer stream.deinit(a);
        if (case == .positive) {
            try core.addStream(&stream); while (!stream.finished) try core.step();
            try std.testing.expectEqualSlices(u32, &.{7, 8}, stream.emitted());
            try check("FreshSessionPositiveRuntimeViolation", sync.session.status == .ok and trio.releases == 1);
            var plain = try lanes.Stream.init(a, .{ .id = "fresh-plain", .prompt = &.{3,4}, .max_new = 1 }); defer plain.deinit(a);
            try core.addStream(&plain); while (!plain.finished) try core.step(); try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
            std.debug.print("CELL grammar_tests_c5 AFD fresh_session=PASS actual_ordinary=PASS\n", .{}); continue;
        }
        if (second) try core.addStream(&stream);
        var tap = Tap{ .trace = &rec.t, .fail = case != .tapped_provider };
        var fault = glm.trace.Trace.Fault{ .name = if (case == .pair_mismatch) "NEVER" else if (case == .reset) "torch.zero_" else if (case == .step_commit) "torch.fill_" else if (drain) "afd.collect" else if (case == .tapped_provider or paired or configured > 0) "afd.exchange" else "torch.clone", .skip = @intFromBool((paired or configured > 0) and !drain) };
        if (case == .compute or case == .step_compute or case == .tapped_provider) rec.e.tap = .{ .ctx = &tap, .tensor = Tap.put };
        if (case != .compute and case != .step_compute) rec.t.test_fault = &fault;
        const consumed_before = rec.provider.seq;
        var outcome: ?anyerror = null;
        if (second) {
            const mask = try rules.rules.next(a); defer a.free(mask);
            var sampled: [1]u32 = undefined; var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
            const windows = [_]lanes.backend.Window{.{ .stream = &stream, .pending = stream.pending.?, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{3}, .masks = mask }};
            actual.verify(&windows, &out) catch |err| { outcome = err; };
        } else actual.prefill(&stream) catch |err| { outcome = err; };
        const immediate = outcome != null and outcome.? == error.ControlPoison and sync.session.status == .poison and sync.session.pending == .poison and rec.at.deferred_logits == null;
        const consumed_after = rec.provider.seq;
        const outstanding_before_release = sync.session.outstanding[0];
        lanes.structured.release(&core, &stream);
        try check("ImmediateCommittedPoisonWakeRuntimeViolation", immediate);
        try check("ActualFaultSeamRuntimeViolation", (if (case == .compute or case == .step_compute) tap.hits else fault.hits) == @intFromBool(case != .pair_mismatch));
        try rec.t.flush();
        if (case != .pair_mismatch) try check("NoLaterCommitOrPickRuntimeViolation", rec.t.events.items.len == (if (case == .compute or case == .step_compute) tap.cutoff else fault.cutoff));
        if (case == .pair_mismatch) try check("MismatchedProviderTakeRefusedRuntimeViolation", data.count(&rec.t, "afd.collect") == 0);
        try check("BackendReleaseOrderRuntimeViolation", trio.releases == 1 and backend.owner == null and trio.threads[0] == null and trio.threads[1] == null);
        for (trio.workers) |worker| try check("ImmediatePeerWakeRuntimeViolation", (worker.outcome == null or worker.outcome.? != error.PeerWaitTimeout));
        try check("DeferredLogitLeaseRuntimeViolation", !serve.open and serve.len == 0 and bound.len == 0 and rec.at.deferred_logits == null and rec.at.structured_control == null);
        try check("UnknownFaultOutputRuntimeViolation", stream.emitted().len == @intFromBool(second));
        try check("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
        try check("ConsumedProviderSequenceRuntimeViolation", consumed_after >= consumed_before and rec.provider.seq == consumed_after);
        if (paired or configured > 0) {
            try check("OutstandingProviderFaultRuntimeViolation", data.count(&rec.t, "afd.exchange") >= 1 and outstanding_before_release >= 1 and consumed_after > consumed_before);
            if (drain) try check("FailedDrainNotRecoveredRuntimeViolation", data.count(&rec.t, "afd.collect") == 1);
        }
        const stages = data.count(&rec.t, "staged.record"); const issued = data.count(&rec.t, "afd.exchange");
        var next = try lanes.Stream.init(a, .{ .id = "same-poison", .prompt = &.{3,4}, .max_new = 1 }); defer next.deinit(a);
        try std.testing.expectError(error.ControlPoison, actual.prefill(&next));
        try check("PoisonedBackendNoWorkRuntimeViolation", data.count(&rec.t, "staged.record") == stages and data.count(&rec.t, "afd.exchange") == issued);
        std.debug.print("CELL grammar_tests_c5 AFD fault={s} configured={d} immediate_poison=PASS roles=3 sidecars=MOCK exactly_once=PASS same_backend=REFUSED hardware=NOT_RUN\n", .{@tagName(case),configured});
    }
}
