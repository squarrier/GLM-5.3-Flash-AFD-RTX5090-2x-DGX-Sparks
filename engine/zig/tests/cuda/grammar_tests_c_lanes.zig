//! GPT-6.1 Sol via Hermes: actual configured Lanes3/4; N2 is an unchanged constructor/settings refusal.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const Trio = @import("grammar_tests_c_afd_chunks.zig").Trio;
const c = lanes.structured_control;
const a = std.testing.allocator;
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
const Cancel = struct {
    trace: *glm.trace.Trace,
    limit: usize,
    fn check(ptr: *anyopaque) bool {
        const x: *Cancel = @ptrCast(@alignCast(ptr));
        return data.count(x.trace, "afd.exchange") >= x.limit;
    }
};
fn witness(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.LaneChunkAgreementRuntimeViolation; }
}
pub fn refusal() !void {
    var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
    rec.provider.paired = true; rec.provider.lanes = 2; rec.provider.inflight = 2;
    try std.testing.expectError(error.BadAfdLanes, glm.afd_lanes.Lanes.init(&rec.at));
    _ = unsetenv("TF_AFD_INFLIGHT");
    _ = setenv("TF_AFD_LANES", "2", 1); defer _ = unsetenv("TF_AFD_LANES");
    try std.testing.expectError(error.BadAfdLanes, glm.afd_lanes.fromEnv(true));
    try std.testing.expectEqual(@as(usize, 0), rec.t.events.items.len);
    std.debug.print("CELL AFD configured_lanes=2 real_constructor=BadAfdLanes real_settings=BadAfdLanes launches=0 PASS\n", .{});
}
pub fn run(comptime Bound: type) !void { try runMode(Bound, null); }
pub fn boundaries(comptime Bound: type) !void {
    inline for ([_]c.Action{.stage,.advance,.route,.request,.take,.post,.tail,.input_commit}) |action| try runMode(Bound, action);
}
fn runMode(comptime Bound: type, comptime cancel_action: ?c.Action) !void {
    inline for ([_]u8{3, 4}) |n| {
        const width: usize = n;
        inline for ([_]usize{width*64, width*64+1, (width+2)*64, 2*width*64, width*64-1}) |length| {
            inline for ([_]usize{0, 1, n}) |cancel_after| {
                if (cancel_action != null and cancel_after != 0) continue;
                var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
                var f = rec.recording();
                rec.provider.paired = true; rec.provider.inflight = n; rec.provider.lanes = n;
                var group = try glm.afd_lanes.Lanes.init(&rec.at); group.agreed = true;
                try group.walk(&rec.space, glm.trace.placeAll); rec.at.lanes = &group;
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
                var ids: [512]u32 = @splat(3); var cancel = Cancel{ .trace = &rec.t, .limit = cancel_after };
                var s = try lanes.Stream.init(a, .{ .id = "group", .prompt = ids[0..length], .max_new = 2, .rules = &rules.rules,
                    .cancel_check = if (cancel_after == 0) null else .{ .ptr = &cancel, .check = Cancel.check } }); defer s.deinit(a);
                var outcome: ?anyerror = null;
                core.addStream(&s) catch |err| { outcome = err; };
                if (outcome) |err| if (err == error.IncompleteMaskedAdapter) {
                    std.debug.print("LaneChunkAgreementRuntimeViolation: actual Lanes constructor remains untapped\n", .{});
                    return error.LaneChunkAgreementRuntimeViolation;
                };
                if (cancel_action != null) {
                    try std.testing.expectEqual(error.Cancelled, outcome.?);
                    try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
                    try std.testing.expect(trio.cutoff != null);
                } else if (cancel_after > 0) {
                    try witness("LaneChunkAgreementRuntimeViolation", outcome != null);
                    try std.testing.expectEqualStrings("Cancelled", @errorName(outcome.?));
                    try witness("NoWorkAfterAgreedCancelRuntimeViolation", s.emitted().len == 0);
                    try witness("LaneChunkAgreementRuntimeViolation", data.count(&rec.t, "staged.record") == cancel_after);
                    try witness("ExpertPermitLedgerRuntimeViolation", data.count(&rec.t, "afd.exchange") == cancel_after);
                } else {
                    if (outcome) |err| return err;
                    while (!s.finished) try core.step();
                    try std.testing.expectEqualSlices(u32, &.{7, 8}, s.emitted());
                    try std.testing.expect(group.runs > 0);
                }
                try witness("OutstandingActionDrainRuntimeViolation", data.count(&rec.t, "afd.exchange") == data.count(&rec.t, "afd.collect"));
                for (sync.session.outstanding) |count| try std.testing.expectEqual(@as(u32, 0), count);
                try witness("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
                try std.testing.expect(!rec.provider.dead and rec.provider.pair == null and rec.provider.lane == null);
                try std.testing.expectEqual(rec.provider.first+rec.provider.count, rec.provider.pos);
                try std.testing.expect(rec.at.structured_control == null and backend.owner == null);
                try std.testing.expectEqual(c.Phase.closed, sync.session.phase);
                for (sync.close_calls) |count| try std.testing.expectEqual(@as(u32, 1), count);
                try trio.noWorkAfterCutoff();
                var plain = try lanes.Stream.init(a, .{ .id = "plain-after-group", .prompt = &.{3, 4}, .max_new = 1 }); defer plain.deinit(a);
                try core.addStream(&plain); while (!plain.finished) try core.step();
                try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
                std.debug.print("CELL AFD lanes configured={d} prompt={d} cancel_requests={d} cancel_action={s} actual_run=PASS expert_sidecars=MOCK drain=PASS plain_next=PASS\n", .{n,length,cancel_after,if (cancel_action) |action| @tagName(action) else "none"});
            }
        }
    }
}
