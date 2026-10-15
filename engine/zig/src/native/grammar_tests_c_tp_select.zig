//! C05 actual two-rank masked selection, before Stream's defensive accept/output.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const Driver = @import("grammar_tests_c_tp_test.zig").Driver;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const a = std.testing.allocator;

pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    if (comptime !@hasField(glm.sample.mask_cpu.Cpu, "endpoint")) {
        std.debug.print("CandidateValidityRuntimeViolation: cooperative local-rank picker is absent\n", .{});
        return error.CandidateValidityRuntimeViolation;
    } else {
        const Case = enum { first_second, rank0_empty, both_empty, all_allowed, inactive, first_cancel, step_cancel, step_select_cancel };
        inline for (std.meta.tags(Case)) |case| {
            const r0 = try glm.grammar_tests_rec.tp(a, 0); defer r0.deinit(a);
            const r1 = try glm.grammar_tests_rec.tp(a, 1); defer r1.deinit(a);
            var f0 = r0.recording(); var f1 = r1.recording();
            var values0 = try data.Data.init(r0.m.vocab, 2); defer values0.deinit();
            var values1 = try data.Data.init(r1.m.vocab, 2); defer values1.deinit();
            if (case == .rank0_empty) {
                // Independently owned maps put both allowed ids on rank1; no invalid rank0 sentinel may win.
                for ([_]*data.Data{&values0, &values1}) |v| {
                    const col7 = std.mem.indexOfScalar(i64, v.ids, 7).?;
                    const col8 = std.mem.indexOfScalar(i64, v.ids, 8).?;
                    std.mem.swap(i64, &v.ids[col7], &v.ids[v.ids.len / 2 + 7]);
                    std.mem.swap(i64, &v.ids[col8], &v.ids[v.ids.len / 2 + 8]);
                }
            }
            values0.set(); values1.set();
            if (case == .inactive) { values0.value(19, 200); values1.value(19, 200); }
            r0.e.smp.masked_cpu = &values0.cpu; r1.e.smp.masked_cpu = &values1.cpu;
            r0.t.picker = values0.picker(); r1.t.picker = values1.picker();
            var serve0 = glm.serve.Serve{ .e = r0.e, .f = &f0, .absorb = true };
            var serve1 = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
            var sync = try Sync.init(a, 1, 2, 0); defer sync.deinit(); sync.install();
            if (case == .first_cancel) sync.cancel_kind = .first;
            if (case == .step_cancel or case == .step_select_cancel) sync.cancel_kind = .step;
            if (case == .step_select_cancel) sync.cancel_action = .select;
            var driver = Driver{ .serve = &serve1, .session = &sync.session };
            var world = glm.ranks.Pair{ .gpa = a, .io = std.testing.io, .rank = 0 };
            var armed = pair.Armed(glm.serve.Serve){ .steps = &serve0, .pair = &world };
            var backend = gl.Lanes(@TypeOf(armed)){ .steps = &armed, .capacity = 2560, .reach = 2560, .frame_allocator = a,
                .structured_session = &sync.session, .mirror = .{ .ctx = &driver, .send = Driver.legacy,
                    .structured_driver_cpu = .{ .open = Driver.open, .operation = Driver.operation, .terminate = Driver.terminate } } };
            var h = Host{ .gpa = a, .gpu = null, .family = &backend, .release = struct { fn noop(_: *anyopaque) void {} }.noop,
                .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
            const actual = wrap(&h);
            var rules = data.Fake{}; rules.init(r0.m.vocab);
            rules.empty = case == .both_empty; rules.all = case == .all_allowed;
            if (case == .inactive) { rules.rules.active = false; rules.rules.after = 19; }
            var stream = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = &.{3, 4}, .max_new = 5, .rules = &rules.rules }); defer stream.deinit(a);
            if (case == .first_second or case == .rank0_empty) {
                try data.selection(actual, &stream);
                try std.testing.expectEqualSlices(u32, &.{7, 8}, stream.emitted());
            } else {
                var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
                var clock = lanes.fake.FixedClock{};
                var core = lanes.Engine.init(a, &cfg, actual, clock.clock()); defer core.deinit();
                if (case == .both_empty) try std.testing.expectError(error.EmptyMask, core.addStream(&stream))
                else if (case == .first_cancel) try std.testing.expectError(error.Cancelled, core.addStream(&stream))
                else {
                    try core.addStream(&stream);
                    if (case == .step_cancel or case == .step_select_cancel) try std.testing.expectError(error.Cancelled, core.step())
                    else while (!stream.finished) try core.step();
                }
                if (case == .both_empty or case == .first_cancel) try std.testing.expectEqual(@as(usize, 0), stream.emitted().len)
                else if (case == .step_cancel or case == .step_select_cancel) try std.testing.expectEqualSlices(u32, &.{7}, stream.emitted())
                else if (case == .all_allowed) try std.testing.expectEqualSlices(u32, &.{2, 2}, stream.emitted())
                else try std.testing.expectEqualSlices(u32, &.{19, 7, 8}, stream.emitted());
            }
            try std.testing.expect(backend.owner == null and !serve0.open and !serve1.open);
            try std.testing.expect(r0.e.structured_control == null and r1.e.structured_control == null);
            try std.testing.expect(!r0.e.structured_logits_valid and !r1.e.structured_logits_valid);
            try std.testing.expectEqual(lanes.structured_control.Phase.closed, sync.session.phase);
            try std.testing.expectEqual(sync.before_calls[0], sync.before_calls[1]);
            try std.testing.expectEqual(sync.after_calls[0], sync.after_calls[1]);
            try std.testing.expectEqual(@as(usize, 1), driver.releases);
            if (case == .step_cancel or case == .step_select_cancel) {
                const stages: usize = if (case == .step_select_cancel) 2 else 1;
                try std.testing.expectEqual(stages, data.count(&r0.t, "staged.record"));
                try std.testing.expectEqual(stages, data.count(&r1.t, "staged.record"));
                try std.testing.expectEqual(@as(usize, 1), values0.cpu.picks);
                try std.testing.expectEqual(@as(usize, 1), values1.cpu.picks);
            }
            var plain = try lanes.Stream.init(a, .{ .id = "plain-after-selection", .prompt = &.{3, 4}, .max_new = 1 }); defer plain.deinit(a);
            try actual.prefill(&plain);
            const expected: u32 = if (case == .inactive) 19 else 2;
            try std.testing.expectEqual(expected, try actual.read(try actual.first(&plain, 2)));
            actual.release(&plain);
            try std.testing.expect(serve0.absorb and serve1.absorb);
            std.debug.print("CELL TP selection={s} winner_before_accept=PASS common_close=PASS plain_next=PASS\n", .{@tagName(case)});
        }
    }
}
