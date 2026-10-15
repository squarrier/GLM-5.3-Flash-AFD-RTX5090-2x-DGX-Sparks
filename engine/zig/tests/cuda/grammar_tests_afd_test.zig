//! Actual Bound -> AFD Serve -> Attn CPU requests (no socket/device exchange).
const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const lanes = @import("lanes");
const gl = @import("glm_lanes");
const data = @import("grammar_tests_data");
const a = std.testing.allocator;
pub fn run(comptime Bound: type) !void {
    const Cases = enum { postclose, request, eos, window_shape, same_width, empty, shape, fill, gather, bind, inactive, forced, length, cancel, selection, stop, prefill_cancel, fill_second, missing_hook, zero, reuse };
    inline for (std.meta.tags(Cases)) |case| {
        var rec: glm.grammar_tests_rec.Rec = undefined;
        try rec.init(a);
        defer rec.deinit(a);
        var f = glm.fwd.Fwd.recording(&rec.m, &rec.t);
        var values = try data.Data.init(rec.m.vocab, 1);
        defer values.deinit();
        values.set();
        rec.e.smp.masked_cpu = &values.cpu;
        rec.t.picker = values.picker();
        rec.at.absorb = true; // structured prefill must override AND restore this
        var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f };
        var bound = Bound{ .s = &serve };
        var driver: cuda.Driver = undefined;
        driver.api.cuCtxSetCurrent = notReady;
        var context = cuda.Context{ .d = &driver, .device = 0, .handle = null };
        if (case == .bind) bound.ctx = &context;
        var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a };
        var cfg = try lanes.Config.init(a, .{ .mtp = true }, 1, 8);
        defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock());
        defer core.deinit();
        var rules = data.Fake{};
        rules.init(rec.m.vocab);
        if (case == .empty) rules.empty = true;
        if (case == .shape) rules.rules.width -= 33;
        if (case == .same_width) rules.rules.width -= 1;
        if (case == .fill) rules.fail_fill = true;
        if (case == .gather) values.cpu.fail = true;
        if (case == .postclose) { rules.rules.active = false; values.value(19, 200); rules.rules.after = 19; }
        if (case == .inactive) { rules.rules.active = false; values.value(91, 200); rules.rules.after = 91; }
        if (case == .forced) { rules.rules.active = false; values.value(91, 200); rules.rules.after = 19; }
        var cancel: std.atomic.Value(bool) = .init(false);
        var poll = data.PollCancel{};
        var hooks = core.backend.vtable.*;
        if (case == .missing_hook) { hooks.first_masked = null; core.backend.vtable = &hooks; }
        if (case == .fill_second) rules.fail_fill_at = 2;
        var s = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = &.{ 3, 4 }, .max_new = if (case == .zero) 0 else if (case == .length) 1 else 5, .eos = if (case == .eos) &.{7} else &.{}, .rules = &rules.rules, .stop_check = if (case == .stop) .{ .ptr = &poll, .check = data.stopFirst } else null, .cancel_check = if (case == .prefill_cancel) .{ .ptr = &poll, .check = data.PollCancel.check } else .{ .ptr = &cancel, .check = cancelled }, .think_budget = if (case == .forced) 2 else 0, .think_close = if (case == .forced) &.{ 18, 19, 20 } else &.{}, .think_end = if (case == .forced) 19 else -1 });
        defer s.deinit(a);
        const expected: ?anyerror = switch (case) { .empty => error.EmptyMask, .shape, .same_width => error.MaskShape, .fill => error.MaskFillFailed, .gather => error.GatherFailed, .bind => error.NotReady, .prefill_cancel => error.Cancelled, .missing_hook => error.IncompleteMaskedAdapter, else => null };
        if (expected) |err| {
            try std.testing.expectError(err, core.addStream(&s));
            try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
            try std.testing.expect(backend.owner == null and !serve.open and rec.at.deferred_logits == null);
        } else {
            if (case == .postclose) try data.postCloseSelection(core.backend, &s) else if (case == .selection) try data.selection(core.backend, &s) else try core.addStream(&s);
            if (case == .window_shape) {
                const calls = rec.t.events.items.len;
                try data.malformedWindows(core.backend, &s);
                try std.testing.expectEqual(calls, rec.t.events.items.len);
            }
            if (case == .inactive) try std.testing.expectEqual(@as(usize, 0), rules.accepted);
            if (case == .cancel) cancel.store(true, .release);
            if (case == .fill_second) {
                try std.testing.expectError(error.MaskFillFailed, core.step());
                try std.testing.expect(s.finished and s.reason == .@"error" and core.live.items.len == 0);
            } else while (!s.finished) try core.step();
            switch (case) {
                .postclose => try std.testing.expectEqualSlices(u32, &.{ 19, 7, 8 }, s.emitted()),
                .inactive => try std.testing.expectEqualSlices(u32, &.{ 91, 7, 8 }, s.emitted()),
                .forced => {
                    try std.testing.expectEqualSlices(u32, &.{ 91, 18, 19, 7, 8 }, s.emitted());
                    try std.testing.expectEqual(@as(usize, 2), rules.accepted);
                    try std.testing.expect(!rules.rules.poisoned);
                },
                .zero => try std.testing.expectEqual(@as(usize, 0), s.emitted().len),
                .length, .cancel, .stop, .fill_second => try std.testing.expectEqualSlices(u32, &.{7}, s.emitted()),
                else => try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, s.emitted()),
            }
            try std.testing.expect(!s.drafts and s.proposer == null);
            try std.testing.expectEqual(@as(usize, if (case == .inactive or case == .postclose) 1 else if (case == .forced) 3 else 0), data.count(&rec.t, "torch.topk"));
            data.witness("AbsorbRestoreRuntimeViolation.N6-structured-absorb", 0, rec.e.st.mtp_len);
            try std.testing.expectEqual(@as(usize, 0), rec.e.st.mtp_len);
            try std.testing.expectEqual(@as(usize, 0), data.count(&rec.t, "raw_mtp"));
            try std.testing.expect(backend.owner == null and !serve.open and rec.at.deferred_logits == null);
            // Five layers contain two MoE seams: exactly one prompt + one forward per emitted continuation.
            const forwards = if (case == .zero) @as(usize, 0) else if (case == .cancel or case == .length or case == .stop or case == .fill_second) @as(usize, 1) else if (case == .inactive or case == .postclose) @as(usize, 3) else if (case == .forced) @as(usize, 5) else @as(usize, 2);
            try std.testing.expectEqual(forwards * 2, data.count(&rec.t, "afd.exchange"));
            try std.testing.expectEqual(forwards * 2, data.count(&rec.t, "afd.collect"));
            if (case == .reuse) {
                var second = data.Fake{};
                second.init(rec.m.vocab);
                var fresh = try lanes.Stream.init(a, .{ .id = "fresh", .prompt = &.{ 5, 6 }, .max_new = 5, .eos = &.{}, .rules = &second.rules });
                defer fresh.deinit(a);
                try core.addStream(&fresh);
                while (!fresh.finished) try core.step();
                try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, fresh.emitted());
            }
        }
        if (case == .bind) {
            try std.testing.expectEqual(@as(usize, 0), rec.t.events.items.len);
            try std.testing.expectEqual(@as(usize, 0), bound.prefills);
            bound.ctx = null;
        }
        if (case == .prefill_cancel) {
            try std.testing.expectEqual(@as(usize, 2), poll.calls);
            try std.testing.expectEqual(@as(usize, 0), values.cpu.picks);
        }
        if (case == .missing_hook or case == .zero) try std.testing.expectEqual(@as(usize, 0), rec.t.events.items.len);
        core.backend = backend.backend();
        data.witness("AbsorbRestoreRuntimeViolation.N6-absorb-restore", 1, @intFromBool(rec.at.absorb));
        try std.testing.expect(rec.at.absorb);
        rules.rules.close();
        values.set();
        var plain = try lanes.Stream.init(a, .{ .id = "plain-after-structured", .prompt = &.{ 3, 4 }, .max_new = 3, .drafts = true });
        defer plain.deinit(a);
        try core.addStream(&plain);
        try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
        try std.testing.expect(rec.e.st.mtp_len > 0);
        while (!plain.finished) try core.step();
        try std.testing.expectEqualSlices(u32, &.{ 2, 2, 2 }, plain.emitted());
        // This existing GLM lane adapter has no draft hook; the requested plain draft policy stays unsupported.
        try std.testing.expectEqual(@as(u64, 0), plain.drafted);
        try std.testing.expect(backend.owner == null and !serve.open and rec.at.deferred_logits == null);
        rules.rules.close();
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        std.debug.print("grammar_tests AFD {s}: exchanges={d} mask_picks={d} accepts={d} closes={d}\n", .{ @tagName(case), data.count(&rec.t, "afd.exchange"), values.cpu.picks, rules.accepted, rules.closes });
    }
}
fn cancelled(ptr: *anyopaque) bool {
    const flag: *std.atomic.Value(bool) = @ptrCast(@alignCast(ptr));
    return flag.load(.acquire);
}
fn notReady(_: cuda.abi.Context) callconv(.c) cuda.abi.Result { return cuda.abi.error_not_ready; }
