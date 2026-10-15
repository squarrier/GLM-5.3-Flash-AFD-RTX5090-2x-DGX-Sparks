//! Actual Host -> Armed -> Serve CPU requests with synchronous rank-one replay.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const a = std.testing.allocator;
const Peer = struct {
    serve: *glm.serve.Serve,
    selected: ?u32 = null,
    frames: usize = 0,
    seq: u64 = 0,
    epoch: u64 = 0,
    empty: usize = 0,
    releases: usize = 0,
    fail: bool = false,
    fn chosen(p: ?*anyopaque, token: u32) !void {
        const x: *Peer = @ptrCast(@alignCast(p.?));
        if (x.selected == null or x.selected.? != token) return error.PeerChoiceMismatch;
    }
    fn legacy(p: ?*anyopaque, op: gl.Op, words: []const u32) !void {
        const x: *Peer = @ptrCast(@alignCast(p.?));
        if (op == .release) x.releases += 1;
        try gl.replay(glm.serve.Serve, x.serve, op, words);
    }
    fn framed(p: ?*anyopaque, words: []const u32) !void {
        const x: *Peer = @ptrCast(@alignCast(p.?));
        if (x.fail) return error.PeerFailed;
        const frame = try gl.mask_frame.Frame.decode(words);
        if (frame.kind == .begin) {
            if (frame.epoch <= x.epoch or frame.sequence != 0) return error.StaleFrame;
            x.epoch = frame.epoch;
            x.seq = 0;
        }
        if (frame.epoch != x.epoch or frame.sequence != x.seq) return error.StaleFrame;
        x.seq += 1;
        x.frames += 1;
        x.selected = null;
        x.selected = gl.mask_frame.select(glm.serve.Serve, x.serve, words) catch |err| {
            if (err != error.EmptyMask and err != error.GatherFailed) return err;
            if (err == error.EmptyMask) x.empty += 1;
            return; // CPU peer records the failure; rank zero must execute the same validity gather
        };
    }
};
fn cancelled(ptr: *anyopaque) bool {
    const flag: *std.atomic.Value(bool) = @ptrCast(@alignCast(ptr));
    return flag.load(.acquire);
}
pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const Cases = enum { postclose, request, eos, window_shape, same_width, empty, shape, duplicate, gather, peer, inactive, forced, length, cancel, selection, stop, prefill_cancel, fill_second, missing_hook, zero, reuse };
    inline for (std.meta.tags(Cases)) |case| {
        const r = try glm.grammar_tests_rec.tp(a, 0);
        defer r.deinit(a);
        const r1 = try glm.grammar_tests_rec.tp(a, 1);
        defer r1.deinit(a);
        var f = r.recording();
        var f1 = r1.recording();
        var values = try data.Data.init(r.m.vocab, r.m.world);
        defer values.deinit();
        values.set();
        r.e.smp.masked_cpu = &values.cpu;
        r.t.picker = values.picker();
        r1.t.picker = values.picker();
        r1.e.smp.masked_cpu = &values.cpu;
        var serve = glm.serve.Serve{ .e = r.e, .f = &f, .absorb = true };
        var peer_serve = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
        var peer = Peer{ .serve = &peer_serve };
        var world = glm.ranks.Pair{ .gpa = a, .io = std.testing.io, .rank = 0 };
        var armed = pair.Armed(glm.serve.Serve){ .steps = &serve, .pair = &world };
        var backend = gl.Lanes(@TypeOf(armed)){ .steps = &armed, .capacity = 2560, .reach = 2560, .frame_allocator = a, .mirror = .{ .ctx = &peer, .send = Peer.legacy, .structured_cpu = Peer.framed, .chosen_cpu = Peer.chosen } };
        var h = Host{ .gpa = a, .gpu = null, .family = &backend, .release = struct { fn noop(_: *anyopaque) void {} }.noop, .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
        var cfg = try lanes.Config.init(a, .{ .mtp = true }, 1, 8);
        defer cfg.deinit(a);
        var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, wrap(&h), clock.clock());
        defer core.deinit();
        var rules = data.Fake{};
        rules.init(r.m.vocab);
        var cancel: std.atomic.Value(bool) = .init(false);
        var poll = data.PollCancel{};
        var hooks = core.backend.vtable.*;
        if (case == .missing_hook) { hooks.first_masked = null; core.backend.vtable = &hooks; }
        if (case == .fill_second) rules.fail_fill_at = 2;
        if (case == .empty) rules.empty = true;
        if (case == .shape) rules.rules.width -= 33;
        if (case == .same_width) rules.rules.width -= 1;
        if (case == .duplicate) values.ids[1] = values.ids[0];
        if (case == .gather) values.cpu.fail = true;
        if (case == .peer) peer.fail = true;
        if (case == .postclose) { rules.rules.active = false; values.value(19, 200); rules.rules.after = 19; }
        if (case == .inactive) { rules.rules.active = false; values.value(91, 200); rules.rules.after = 91; }
        if (case == .forced) { rules.rules.active = false; values.value(91, 200); rules.rules.after = 19; }
        var s = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = &.{ 3, 4 }, .max_new = if (case == .zero) 0 else if (case == .length) 1 else 5, .eos = if (case == .eos) &.{7} else &.{}, .drafts = true, .rules = &rules.rules, .stop_check = if (case == .stop) .{ .ptr = &poll, .check = data.stopFirst } else null, .cancel_check = if (case == .prefill_cancel) .{ .ptr = &poll, .check = data.PollCancel.check } else .{ .ptr = &cancel, .check = cancelled }, .think_budget = if (case == .forced) 2 else 0, .think_close = if (case == .forced) &.{ 18, 19, 20 } else &.{}, .think_end = if (case == .forced) 19 else -1 });
        defer s.deinit(a);
        const expected: ?anyerror = switch (case) { .empty => error.EmptyMask, .shape, .same_width => error.MaskShape, .duplicate => error.MaskId, .gather => error.GatherFailed, .peer => error.PeerFailed, .prefill_cancel => error.Cancelled, .missing_hook => error.IncompleteMaskedAdapter, else => null };
        if (expected) |err| {
            try std.testing.expectError(err, core.addStream(&s));
            try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
            try std.testing.expect(backend.owner == null and !serve.open and !peer_serve.open);
            if (case == .empty) try std.testing.expectEqual(@as(usize, 1), peer.empty);
        } else {
            if (case == .postclose) try data.postCloseSelection(core.backend, &s) else if (case == .selection) try data.selection(core.backend, &s) else try core.addStream(&s);
            if (case == .window_shape) {
                const calls = r.t.events.items.len;
                const frames = peer.frames;
                try data.malformedWindows(core.backend, &s);
                try std.testing.expectEqual(calls, r.t.events.items.len);
                try std.testing.expectEqual(frames, peer.frames);
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
            try std.testing.expectEqual(@as(usize, 0), data.count(&r.t, "pick"));
            try std.testing.expectEqual(@as(usize, 0), r.e.st.mtp_len);
            try std.testing.expectEqual(@as(usize, 0), data.count(&r.t, "raw_mtp"));
            try std.testing.expect(backend.owner == null and !serve.open and !peer_serve.open);
            if (case == .reuse) {
                var second = data.Fake{};
                second.init(r.m.vocab);
                var fresh = try lanes.Stream.init(a, .{ .id = "fresh", .prompt = &.{ 5, 6 }, .max_new = 5, .eos = &.{}, .rules = &second.rules });
                defer fresh.deinit(a);
                try core.addStream(&fresh);
                while (!fresh.finished) try core.step();
                try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, fresh.emitted());
                try std.testing.expectEqual(@as(u64, 2), backend.epoch);
            }
        }
        if (case == .prefill_cancel) {
            try std.testing.expectEqual(@as(usize, 2), poll.calls);
            try std.testing.expectEqual(@as(usize, 0), values.cpu.picks);
        }
        if (case == .missing_hook or case == .zero) {
            try std.testing.expectEqual(@as(usize, 0), peer.frames);
            try std.testing.expectEqual(@as(usize, 0), r.t.events.items.len);
        }
        core.backend = wrap(&h);
        rules.rules.close();
        values.set();
        peer.fail = false;
        var plain = try lanes.Stream.init(a, .{ .id = "plain-after-structured", .prompt = &.{ 3, 4 }, .max_new = 3, .drafts = true });
        defer plain.deinit(a);
        try core.addStream(&plain);
        try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
        try std.testing.expect(r.e.st.mtp_len > 0);
        while (!plain.finished) try core.step();
        try std.testing.expectEqualSlices(u32, &.{ 2, 2, 2 }, plain.emitted());
        // This existing GLM lane adapter has no draft hook; the requested plain draft policy stays unsupported.
        try std.testing.expectEqual(@as(u64, 0), plain.drafted);
        try std.testing.expect(backend.owner == null and !serve.open and !peer_serve.open);
        rules.rules.close();
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        std.debug.print("grammar_tests TP {s}: frames={d} mask_picks={d} accepts={d} closes={d}\n", .{ @tagName(case), peer.frames, values.cpu.picks, rules.accepted, rules.closes });
    }
}
