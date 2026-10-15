//! GPT-6.1 Sol via Hermes: actual Bound/Serve/Attn/Pairs constructors; expert peers MOCK.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const Trio = @import("grammar_tests_c_afd_chunks.zig").Trio;
const c = lanes.structured_control;
const a = std.testing.allocator;
const Cancel = struct {
    trace: *glm.trace.Trace,
    limit: usize,
    fn check(ptr: *anyopaque) bool {
        const x: *Cancel = @ptrCast(@alignCast(ptr));
        return data.count(x.trace, "afd.exchange") >= x.limit;
    }
};
fn witness(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.PairChunkAgreementRuntimeViolation; }
}
pub fn run(comptime Bound: type) !void { try runMode(Bound, null, false); }
pub fn cutoffWitness(comptime Bound: type) !void { try runMode(Bound, .stage, true); }
pub fn boundaries(comptime Bound: type) !void {
    inline for ([_]c.Action{.stage,.advance,.route,.request,.take,.post,.tail,.input_commit}) |action| try runMode(Bound, action, false);
}
fn runMode(comptime Bound: type, comptime cancel_action: ?c.Action, comptime inject_after_cutoff: bool) !void {
    inline for ([_]u8{1, 2}) |inflight| {
        inline for ([_]usize{128, 129, 255, 256}) |length| {
            inline for ([_]usize{0, 1, inflight}, 0..) |cancel_after, index| {
                if ((inflight == 1 and index == 2) or (cancel_action != null and cancel_after != 0)) continue;
                var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
                var f = rec.recording();
                rec.provider.paired = true; rec.provider.inflight = inflight;
                var pairs = glm.afd_pairs.Pairs.init(&rec.at); pairs.agreed = true;
                try pairs.walk(&rec.space, glm.trace.placeAll); rec.at.pairs = &pairs;
                var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
                rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker(); rec.at.absorb = true;
                var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f }; var bound = Bound{ .s = &serve };
                var sync = try Sync.init(a, 7, 3, 41); defer sync.deinit(); sync.cancel_kind = if (cancel_action != null) .group_action else null; sync.cancel_action = cancel_action; sync.install();
                const masks = try a.alloc(u32, (rec.m.vocab + 31) / 32); defer a.free(masks);
                var trio = Trio{ .sync = &sync, .inner = sync.session.hooks.?, .mask = masks, .trace = &rec.t }; trio.install();
                var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a,
                    .structured_session = &sync.session, .mirror = .{ .ctx = &trio, .structured_driver_cpu = .{
                    .open = Trio.driverOpen, .operation = Trio.operation, .terminate = Trio.terminate } } };
                var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
                var clock = lanes.fake.FixedClock{};
                var core = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock()); defer core.deinit();
                var rules = data.Fake{}; rules.init(rec.m.vocab);
                var ids: [256]u32 = @splat(3); var cancel = Cancel{ .trace = &rec.t, .limit = cancel_after };
                var s = try lanes.Stream.init(a, .{ .id = "pair", .prompt = ids[0..length], .max_new = 2, .rules = &rules.rules,
                    .cancel_check = if (cancel_after == 0) null else .{ .ptr = &cancel, .check = Cancel.check } }); defer s.deinit(a);
                var outcome: ?anyerror = null;
                core.addStream(&s) catch |err| { outcome = err; };
                if (outcome) |err| if (err == error.IncompleteMaskedAdapter) {
                    std.debug.print("PairChunkAgreementRuntimeViolation: actual Pairs constructor remains untapped\n", .{});
                    return error.PairChunkAgreementRuntimeViolation;
                };
                if (cancel_action != null) {
                    try std.testing.expectEqual(error.Cancelled, outcome.?);
                    try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
                    try std.testing.expect(trio.cutoff != null);
                } else if (cancel_after > 0) {
                    try witness("PairChunkAgreementRuntimeViolation", outcome != null);
                    try std.testing.expectEqualStrings("Cancelled", @errorName(outcome.?));
                    try witness("NoWorkAfterAgreedCancelRuntimeViolation", s.emitted().len == 0);
                    try witness("PairChunkAgreementRuntimeViolation", data.count(&rec.t, "staged.record") == cancel_after);
                    try witness("ExpertPermitLedgerRuntimeViolation", data.count(&rec.t, "afd.exchange") == cancel_after);
                } else {
                    if (outcome) |err| return err;
                    while (!s.finished) try core.step();
                    try std.testing.expectEqualSlices(u32, &.{7, 8}, s.emitted());
                    try std.testing.expect(pairs.runs > 0);
                }
                try witness("OutstandingActionDrainRuntimeViolation", data.count(&rec.t, "afd.exchange") == data.count(&rec.t, "afd.collect"));
                for (sync.session.outstanding) |count| try std.testing.expectEqual(@as(u32, 0), count);
                try witness("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
                try std.testing.expect(!rec.provider.dead and rec.provider.pair == null);
                try std.testing.expectEqual(rec.provider.first+rec.provider.count, rec.provider.pos);
                try std.testing.expect(rec.at.structured_control == null and backend.owner == null);
                try std.testing.expectEqual(c.Phase.closed, sync.session.phase);
                for (sync.close_calls) |count| try std.testing.expectEqual(@as(u32, 1), count);
                try trio.noWorkAfterCutoff();
                if (inject_after_cutoff) {
                    try std.testing.expect(trio.cutoff != null);
                    try std.testing.expectEqual(@as(usize, 0), rec.t.tri_rec.n);
                    const before = rec.t.events.items.len;
                    // Real recording call, buffered after the flushed frontier; never a fabricated Event.
                    try f.tri.embed(rec.e.pbuf.ids.ptr, .{ .b16 = rec.w.embed.b16.ptr }, rec.e.pbuf.x.ptr, 1, rec.m.hidden, rec.m.streams);
                    try std.testing.expect(rec.t.tri_rec.n > 0);
                    try std.testing.expectError(error.ExpertPermitPostCancelWorkRuntimeViolation, trio.noWorkAfterCutoff());
                    try std.testing.expect(rec.t.events.items.len > before);
                    std.debug.print("CELL C08 strict_cutoff inflight={d} prompt={d} injected_buffered_embed=DETECTED permitted_frontier_flushed=PASS\n", .{inflight,length});
                }
                var plain = try lanes.Stream.init(a, .{ .id = "plain-after-pair", .prompt = &.{3, 4}, .max_new = 1 }); defer plain.deinit(a);
                try core.addStream(&plain); while (!plain.finished) try core.step();
                try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
                std.debug.print("CELL AFD pairs inflight={d} prompt={d} cancel_requests={d} cancel_action={s} actual_run=PASS expert_sidecars=MOCK drain=PASS plain_next=PASS\n", .{inflight,length,cancel_after,if (cancel_action) |action| @tagName(action) else "none"});
            }
        }
    }
}
