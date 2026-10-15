//! GPT-6.1 Sol via Hermes: actual configured scheduler finish/fallback variants; expert peers MOCK.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const Trio = @import("grammar_tests_c_afd_chunks.zig").Trio;
const c = lanes.structured_control;
const a = std.testing.allocator;
const Tap = struct {
    calls: usize = 0,
    fn tensor(ptr: *anyopaque, _: []const u8, _: glm.trace.T, _: usize, _: usize, _: ?[]const u8) !void {
        const t: *Tap = @ptrCast(@alignCast(ptr)); t.calls += 1;
    }
};
fn check(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.AfdFinishRuntimeViolation; }
}
pub fn run(comptime Bound: type) !void {
    const Scheduler = enum { pair1, pair2, lanes3, lanes4 };
    const Case = enum { normal, one, tapped, tapped_cancel, finish_cancel, first_cancel, step_cancel, select_cancel, empty };
    inline for (std.meta.tags(Scheduler)) |scheduler| inline for (std.meta.tags(Case)) |case| {
        var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
        var f = rec.recording();
        const n: u8 = switch (scheduler) { .pair1 => 1, .pair2 => 2, .lanes3 => 3, .lanes4 => 4 };
        rec.provider.paired = true; rec.provider.inflight = n;
        var pairs = glm.afd_pairs.Pairs.init(&rec.at); pairs.agreed = true;
        var group: glm.afd_lanes.Lanes = undefined;
        if (scheduler == .pair1 or scheduler == .pair2) {
            try pairs.walk(&rec.space, glm.trace.placeAll); rec.at.pairs = &pairs;
        } else {
            rec.provider.lanes = n; group = try glm.afd_lanes.Lanes.init(&rec.at); group.agreed = true;
            try group.walk(&rec.space, glm.trace.placeAll); rec.at.lanes = &group;
        }
        var tap = Tap{};
        const tapped = case == .tapped or case == .tapped_cancel;
        if (tapped) rec.e.tap = .{ .ctx = &tap, .tensor = Tap.tensor };
        var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
        rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker(); rec.at.absorb = true;
        var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f }; var bound = Bound{ .s = &serve };
        var sync = try Sync.init(a, 17, 3, 41); defer sync.deinit(); sync.install();
        switch (case) {
            .tapped_cancel => { sync.cancel_kind = .prompt_chunk; sync.cancel_chunk = 1; },
            .finish_cancel => { sync.cancel_kind = .group_action; sync.cancel_action = .finish; },
            .first_cancel => sync.cancel_kind = .first,
            .step_cancel => { sync.cancel_kind = .step; sync.cancel_action = .none; },
            .select_cancel => { sync.cancel_kind = .step; sync.cancel_action = .select; },
            else => {},
        }
        const masks = try a.alloc(u32, (rec.m.vocab+31)/32); defer a.free(masks);
        var trio = Trio{ .sync = &sync, .inner = sync.session.hooks.?, .mask = masks, .trace = &rec.t }; trio.install();
        var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a,
            .structured_session = &sync.session, .mirror = .{ .ctx = &trio, .structured_driver_cpu = .{
            .open = Trio.driverOpen, .operation = Trio.operation, .terminate = Trio.terminate } } };
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock()); defer core.deinit();
        var rules = data.Fake{}; rules.init(rec.m.vocab); rules.empty = case == .empty;
        var ids: [257]u32 = @splat(3);
        const length: usize = if (case == .one) 3 else if (scheduler == .lanes4) 257 else 193;
        var s = try lanes.Stream.init(a, .{ .id = "finish", .prompt = ids[0..length], .max_new = 3, .rules = &rules.rules }); defer s.deinit(a);
        const pre_cancel = case == .tapped_cancel or case == .finish_cancel or case == .first_cancel;
        if (pre_cancel) try std.testing.expectError(error.Cancelled, core.addStream(&s))
        else if (case == .empty) try std.testing.expectError(error.EmptyMask, core.addStream(&s))
        else {
            try core.addStream(&s);
            if (case == .step_cancel or case == .select_cancel) try std.testing.expectError(error.Cancelled, core.step())
            else while (!s.finished) try core.step();
        }
        if (pre_cancel or case == .empty) try std.testing.expectEqual(@as(usize, 0), s.emitted().len)
        else if (case == .step_cancel or case == .select_cancel) try std.testing.expectEqualSlices(u32, &.{7}, s.emitted())
        else try std.testing.expectEqualSlices(u32, &.{7,8}, s.emitted());
        const dispatched = if (scheduler == .pair1 or scheduler == .pair2) pairs.runs else group.runs;
        try check("ActualFallbackDispatchRuntimeViolation", if (tapped or case == .one) dispatched == 0 else dispatched > 0);
        try check("ActualTappedFallbackRuntimeViolation", if (tapped) tap.calls > 0 else tap.calls == 0);
        if (case == .tapped_cancel) try check("NoWorkAfterAgreedCancelRuntimeViolation", data.count(&rec.t, "staged.record") == 1);
        try check("OutstandingActionDrainRuntimeViolation", data.count(&rec.t, "afd.exchange") == data.count(&rec.t, "afd.collect"));
        for (sync.session.outstanding) |count| try std.testing.expectEqual(@as(u32, 0), count);
        try check("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
        try check("DeferredLogitLeaseRuntimeViolation", rec.at.deferred_logits == null and rec.at.structured_control == null and !serve.open and bound.len == 0 and backend.owner == null);
        try std.testing.expect(!rec.provider.dead and rec.provider.pair == null and rec.provider.lane == null);
        try std.testing.expectEqual(c.Phase.closed, sync.session.phase);
        for (sync.close_calls) |count| try std.testing.expectEqual(@as(u32, 1), count);
        try trio.noWorkAfterCutoff();
        rec.e.tap = null;
        var plain = try lanes.Stream.init(a, .{ .id = "plain-after-finish", .prompt = &.{3,4}, .max_new = 1 }); defer plain.deinit(a);
        try core.addStream(&plain); while (!plain.finished) try core.step();
        try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
        try std.testing.expect(rec.e.st.mtp_len > 0);
        std.debug.print("CELL AFD finish scheduler={s} case={s} prompt={d} dispatch={d} tap_calls={d} strict_cutoff=PASS no_absorb=PASS plain_next=PASS expert_sidecars=MOCK\n", .{@tagName(scheduler),@tagName(case),length,dispatched,tap.calls});
    };
}
