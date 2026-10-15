//! grammar_mask CPU tests: the real masked adapter through the actual Lanes -> Bound -> Serve -> Attn -> sampler chain
//! (recorded; the device work is MOCK, the matcher FAKE; refusals, staging, resolution and cleanup are product code).
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const fx = @import("grammar_mask_fixture.zig");
const md = glm.sample.mask_dev;
const a = std.testing.allocator;

fn witness(comptime id: []const u8, expected: usize, observed: usize) void {
    std.debug.print("WITNESS {s} expected={d} observed={d}\n", .{ id, expected, observed });
}

/// The unmasked winner (2) is forbidden at the first and the second pick: the allowed winners 7 and 8 are chosen
/// before topk truncates (the forbidden 7 at the second pick outranks 8 and still loses), each staged first.
pub fn forbidden(comptime Bound: type) !void {
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    const picks = [_][]const fx.Pair{ &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 80 }, .{ .token = 9, .value = 60 } }, &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 90 }, .{ .token = 8, .value = 70 } } };
    var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var script = fx.Script{ .steps = &.{ &.{ 7, 9 }, &.{8} } };
    script.init(c.rec.m.vocab);
    try std.testing.expect(c.back.backend().vtable.first_masked != null);
    var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_forbidden", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &script.rules });
    defer s.deinit(a);
    try c.drive(&s);
    // the independent oracle over the same rows and the script's own masks
    const words = try a.alloc(u32, c.dev.words);
    defer a.free(words);
    const row = try a.alloc(f32, c.rec.m.vocab);
    defer a.free(row);
    for (picks, 0..) |p, i| {
        @memset(row, -1000);
        for (p) |x| row[x.token] = x.value;
        script.expected(i, words);
        const want = fx.oracle(row, words).?;
        witness("ForbiddenWinnerRuntimeViolation.grammar_mask_pick", want, s.emitted()[i]);
        if (s.emitted()[i] != want) return error.ForbiddenWinnerRuntimeViolation;
        try std.testing.expectEqual(std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(words)), sched.staged_hashes[i]);
    }
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, s.emitted());
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, sched.staged_counts[0..2]);
    // actual forwarding: one deferred prompt pass, the masked first and the masked step, no plain draw
    try std.testing.expectEqual(@as(u64, 1), c.bound.prefills);
    try std.testing.expectEqual(@as(u64, 2), c.dev.stats.masked);
    try std.testing.expectEqual(@as(u64, 2), c.dev.stats.staged);
    try std.testing.expectEqual(@as(usize, 2), try fx.maskedOrder(&c.rec.t));
    try std.testing.expectEqual(@as(usize, 2), fx.count(&c.rec.t, "torch.topk"));
    try std.testing.expectEqual(@as(usize, 4), fx.count(&c.rec.t, "afd.exchange")); // the prompt and one step forward
    try std.testing.expectEqual(@as(usize, 2), script.accepted);
    try c.clean();
    // the next request is plain and unchanged: no staging, the plain draw
    const plain_picks = [_][]const fx.Pair{ &.{.{ .token = 2, .value = 100 }}, &.{.{ .token = 5, .value = 100 }} };
    var ps = try fx.Sched.init(c.rec.m.vocab, &plain_picks);
    defer ps.deinit();
    c.rec.t.picker = ps.picker();
    const staged = c.dev.stats.staged;
    var plain = try lanes.Stream.init(a, .{ .id = "grammar_mask_plain-next", .prompt = &.{ 3, 4 }, .max_new = 2 });
    defer plain.deinit(a);
    try c.drive(&plain);
    try std.testing.expectEqualSlices(u32, &.{ 2, 5 }, plain.emitted());
    try std.testing.expectEqual(staged, c.dev.stats.staged);
    try c.clean();
    std.debug.print("grammar_mask forbidden: emitted {any}, staged {d}, masked {d}, plain next {any}\n", .{ s.emitted(), c.dev.stats.staged, c.dev.stats.masked, plain.emitted() });
}

/// Plain with the adapter installed equals plain without it; an all-allowed structured request equals plain: every
/// event, operand and address of the actual chain (the plan sees all-allowed and draws with sample.greedy).
pub fn identity(comptime Bound: type) !void {
    const picks = [_][]const fx.Pair{ &.{.{ .token = 2, .value = 100 }}, &.{.{ .token = 9, .value = 50 }}, &.{.{ .token = 4, .value = 7 }} };
    var runs: [3]fx.Chain(Bound) = .{ .{}, .{}, .{} };
    var scheds: [3]fx.Sched = undefined;
    var emitted: [3][3]u32 = undefined;
    inline for (0..3) |k| {
        try runs[k].init(k != 0);
        scheds[k] = try fx.Sched.init(runs[k].rec.m.vocab, &picks);
        runs[k].rec.t.picker = scheds[k].picker();
        runs[k].dev.emulate = scheds[k].emulator();
    }
    defer {
        inline for (0..3) |k| {
            scheds[k].deinit();
            runs[k].deinit();
        }
    }
    var script = fx.Script{ .all = true };
    script.init(runs[2].rec.m.vocab);
    inline for (0..3) |k| {
        var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_identity", .prompt = &.{ 3, 4 }, .max_new = 3, .rules = if (k == 2) &script.rules else null });
        defer s.deinit(a);
        try runs[k].drive(&s);
        @memcpy(&emitted[k], s.emitted());
        try runs[k].clean();
    }
    try fx.eqTraces(&runs[0].rec.t, &runs[1].rec.t);
    try fx.eqTraces(&runs[0].rec.t, &runs[2].rec.t);
    try std.testing.expectEqualSlices(u32, &.{ 2, 9, 4 }, &emitted[0]);
    try std.testing.expectEqualSlices(u32, &emitted[0], &emitted[1]);
    try std.testing.expectEqualSlices(u32, &emitted[0], &emitted[2]);
    try std.testing.expectEqual(@as(u64, 3), runs[2].dev.stats.identity);
    try std.testing.expectEqual(@as(u64, 0), runs[2].dev.stats.staged + runs[1].dev.stats.staged);
    std.debug.print("grammar_mask identity: plain without/with the adapter and all-allowed structured: {d} events each, equal\n", .{runs[0].rec.t.events.items.len});
}

const Case = enum { empty_first, empty_step, fill_step, cancel };

/// Empty masks refuse before any launch (the step's before its forward); a fill error and a cancel end the request;
/// each leaves no owner, deferred logits or poison, and a structured then a plain request follow correctly.
pub fn cleanup(comptime Bound: type) !void {
    inline for (std.meta.tags(Case)) |case| {
        var c: fx.Chain(Bound) = .{};
        try c.init(true);
        defer c.deinit();
        const picks = [_][]const fx.Pair{ &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 80 } }, &.{ .{ .token = 2, .value = 100 }, .{ .token = 8, .value = 70 } } };
        var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
        defer sched.deinit();
        c.rec.t.picker = sched.picker();
        c.dev.emulate = sched.emulator();
        var script = fx.Script{ .steps = switch (case) {
            .empty_first => &.{&.{}},
            .empty_step => &.{ &.{7}, &.{} },
            .fill_step, .cancel => &.{ &.{7}, &.{8} },
        }, .fail_fill_at = if (case == .fill_step) 2 else null };
        script.init(c.rec.m.vocab);
        var flag: std.atomic.Value(bool) = .init(false);
        var s = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &script.rules, .cancel_check = .{ .ptr = &flag, .check = cancelled } });
        defer s.deinit(a);
        switch (case) {
            .empty_first => try std.testing.expectError(error.EmptyMask, c.core.addStream(&s)),
            .empty_step, .fill_step => {
                try c.core.addStream(&s);
                try std.testing.expectError(if (case == .empty_step) error.EmptyMask else error.MaskFillFailed, c.core.step());
            },
            .cancel => {
                try c.core.addStream(&s);
                flag.store(true, .release);
                while (!s.finished) try c.core.step();
                try std.testing.expect(s.reason == .cancelled);
            },
        }
        try std.testing.expect(s.finished);
        try std.testing.expectEqualSlices(u32, if (case == .empty_first) &.{} else &.{7}, s.emitted());
        // no staging for a refused mask; the step's forward never ran after an empty or failed mask
        const masked: u64 = if (case == .empty_first) 0 else 1;
        try std.testing.expectEqual(masked, c.dev.stats.staged);
        try std.testing.expectEqual(@as(usize, 2), fx.count(&c.rec.t, "afd.exchange"));
        try std.testing.expectEqual(@as(u64, if (case == .empty_first or case == .empty_step) 1 else 0), c.dev.stats.refused);
        try std.testing.expect(!c.dev.poisoned);
        try c.clean();
        // a valid structured request, then a plain one, on the same chain
        const next_picks = [_][]const fx.Pair{ &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 80 } }, &.{ .{ .token = 2, .value = 100 }, .{ .token = 8, .value = 70 } }, &.{.{ .token = 2, .value = 100 }} };
        var ns = try fx.Sched.init(c.rec.m.vocab, &next_picks);
        defer ns.deinit();
        c.rec.t.picker = ns.picker();
        c.dev.emulate = ns.emulator();
        var fresh = fx.Script{ .steps = &.{ &.{7}, &.{8} } };
        fresh.init(c.rec.m.vocab);
        var t = try lanes.Stream.init(a, .{ .id = "grammar_mask_next-structured", .prompt = &.{ 5, 6 }, .max_new = 5, .rules = &fresh.rules });
        defer t.deinit(a);
        try c.drive(&t);
        try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, t.emitted());
        var p = try lanes.Stream.init(a, .{ .id = "grammar_mask_next-plain", .prompt = &.{ 5, 6 }, .max_new = 1 });
        defer p.deinit(a);
        try c.drive(&p);
        try std.testing.expectEqualSlices(u32, &.{2}, p.emitted());
        try c.clean();
        std.debug.print("grammar_mask cleanup {s}: emitted {any}, staged {d}, refused {d}, next structured {any}, next plain {any}\n", .{ @tagName(case), s.emitted(), c.dev.stats.staged, c.dev.stats.refused, t.emitted(), p.emitted() });
    }
}

fn cancelled(ptr: *anyopaque) bool {
    const flag: *std.atomic.Value(bool) = @ptrCast(@alignCast(ptr));
    return flag.load(.acquire);
}

/// A device-side failure after staging (the kernel's launch) poisons the adapter: the request fails, every later
/// structured request refuses before any launch, and plain requests are untouched.
pub fn poison(comptime Bound: type) !void {
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    const picks = [_][]const fx.Pair{ &.{.{ .token = 2, .value = 100 }}, &.{.{ .token = 4, .value = 100 }} };
    var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var fault: glm.trace.Trace.Fault = .{ .name = "grammar.mask" };
    c.rec.t.test_fault = &fault;
    var script = fx.Script{ .steps = &.{ &.{7}, &.{8} } };
    script.init(c.rec.m.vocab);
    var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_poison", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &script.rules });
    defer s.deinit(a);
    try std.testing.expectError(error.RecordedPartialFault, c.core.addStream(&s));
    c.rec.t.test_fault = null;
    try std.testing.expect(c.dev.poisoned and fault.hits == 1 and s.emitted().len == 0);
    try c.clean();
    const before = c.rec.t.events.items.len;
    var again = fx.Script{ .steps = &.{&.{7}} };
    again.init(c.rec.m.vocab);
    var r = try lanes.Stream.init(a, .{ .id = "grammar_mask_after-poison", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &again.rules });
    defer r.deinit(a);
    try std.testing.expectError(error.ControlPoison, c.core.addStream(&r));
    try std.testing.expectEqual(before, c.rec.t.events.items.len);
    try c.clean();
    var p = try lanes.Stream.init(a, .{ .id = "grammar_mask_plain-after-poison", .prompt = &.{ 3, 4 }, .max_new = 2 });
    defer p.deinit(a);
    try c.drive(&p);
    try std.testing.expectEqualSlices(u32, &.{ 2, 4 }, p.emitted());
    try c.clean();
    std.debug.print("grammar_mask poison: device fault -> poisoned, structured refused with 0 new events, plain {any}\n", .{p.emitted()});
}

/// Thinking: rules inactive until the generated marker 19; the marker's own pick is the plain draw and never matcher
/// input; the NEXT pick is masked.
pub fn marker(comptime Bound: type) !void {
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    const picks = [_][]const fx.Pair{ &.{ .{ .token = 19, .value = 200 }, .{ .token = 2, .value = 100 } }, &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 80 } }, &.{ .{ .token = 19, .value = 300 }, .{ .token = 8, .value = 70 } } };
    var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var script = fx.Script{ .steps = &.{ &.{7}, &.{8} } };
    script.init(c.rec.m.vocab);
    script.rules.active = false;
    script.rules.after = 19;
    var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_marker", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &script.rules });
    defer s.deinit(a);
    try c.drive(&s);
    try std.testing.expectEqualSlices(u32, &.{ 19, 7, 8 }, s.emitted());
    try std.testing.expectEqual(@as(usize, 2), script.accepted);
    try std.testing.expectEqual(@as(u64, 1), c.dev.stats.plain);
    try std.testing.expectEqual(@as(u64, 2), c.dev.stats.masked);
    try std.testing.expectEqual(@as(usize, 2), try fx.maskedOrder(&c.rec.t));
    try c.clean();
    std.debug.print("grammar_mask marker: emitted {any}: the marker drawn plain, the next two masked\n", .{s.emitted()});
}

/// The adapter refuses outside its scope: a TP rank (world 2), two adapters at once, a peer without structured
/// transport; malformed masks never pass the plan.
pub fn scope(comptime Bound: type) !void {
    const tp = try glm.grammar_tests_rec.tp(a, 0);
    defer tp.deinit(a);
    try std.testing.expectError(error.MaskedWorld, md.Dev.recorded(a, &tp.m, &tp.space));
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    var tf = glm.fwd.Fwd.recording(&tp.m, &tp.t);
    try std.testing.expectError(error.MaskedWorld, c.dev.ready(&tf));
    var cpu: glm.sample.mask_cpu.Cpu = .{ .gpa = a, .width = c.rec.m.vocab, .ranks = &.{} };
    c.rec.e.smp.masked_cpu = &cpu;
    try std.testing.expectError(error.IncompleteMaskedAdapter, glm.sample.maskedReady(&c.f, &c.rec.e.smp));
    c.rec.e.smp.masked_cpu = null;
    const gl = @import("glm_lanes");
    const peer: gl.Mirror = .{ .send = sendNothing };
    try std.testing.expectError(error.IncompleteMaskedAdapter, peer.tellStructured(&.{1}));
    const alone: gl.Mirror = .{};
    try alone.tellStructured(&.{1});
    const W = c.dev.words;
    const mask = try a.alloc(u32, W + 1);
    defer a.free(mask);
    @memset(mask, 0);
    try std.testing.expectError(error.MaskShape, md.plan(c.rec.m.vocab, mask)); // W + 1 words
    try std.testing.expectError(error.EmptyMask, md.plan(c.rec.m.vocab, mask[0..W]));
    try std.testing.expectError(error.MaskShape, md.plan(100, &.{ 1, 0, 0, 1 << 4 })); // a bit past width 100
    const p = try md.plan(100, &.{ 0, 1 << 3, 0, 1 << 3 });
    try std.testing.expect(p.count == 2 and p.first == 35 and !p.all);
    try std.testing.expect((try md.plan(64, &.{ 0xffff_ffff, 0xffff_ffff })).all);
    try std.testing.expectError(error.MaskInconsistent, md.resolve(p, 100, &.{ 0, 1 << 3, 0, 1 << 3 }, 3, @bitCast(@as(f32, 1.0))));
    try std.testing.expectError(error.BadTokenId, md.resolve(p, 100, &.{ 0, 1 << 3, 0, 1 << 3 }, 100, md.neg_inf));
    try std.testing.expectEqual(@as(u32, 35), try md.resolve(p, 100, &.{ 0, 1 << 3, 0, 1 << 3 }, 3, md.neg_inf));
    try c.dev.precheck(&.{}); // an inactive (thinking) step is unconstrained
    try std.testing.expectError(error.EmptyMask, c.dev.precheck(mask[0..W]));
    std.debug.print("grammar_mask scope: TP world 2, two adapters and a peer without transport refuse; plan/resolve shapes\n", .{});
}

fn sendNothing(_: ?*anyopaque, _: @import("glm_lanes").Op, _: []const u32) anyerror!void {}

/// The real-library Rules adapter (lanes.grammar_owned) over FAKE library objects of grammar_abi's tfg_owned shape:
/// width and marker refusals free the matcher; the release order (compile, matcher, the compiled grammar freed) comes
/// before any fill; matcher errors poison the Rules; Rules.close frees the matcher exactly once; through the actual
/// chain the forbidden winners are removed exactly as with the script and the matcher is freed at the Stream's deinit.
pub fn owned(comptime Bound: type) !void {
    const go = lanes.grammar_owned;
    const O = go.Owned(fx.FakeMatcher);
    var err: [512]u8 = undefined;
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    const V: u32 = @intCast(c.rec.m.vocab);
    // refusals: a compiler of another vocabulary, a marker outside the width, a kind outside 0..5, no matcher
    {
        var script = fx.Script{ .steps = &.{&.{7}} };
        script.init(V);
        var lib = fx.FakeLib{ .compiler = .{ .vocab = V + 1 }, .script = &script };
        var o: O = undefined;
        try std.testing.expectError(error.WidthMismatch, go.request(fx.FakeMatcher, &o, &lib, 0, "", &err, V, null));
        try std.testing.expectEqualStrings("cmgx", lib.log.seq());
        lib.compiler.vocab = V;
        lib.log = .{};
        try std.testing.expectError(error.WidthMismatch, go.request(fx.FakeMatcher, &o, &lib, 0, "", &err, V, V));
        try std.testing.expectEqualStrings("cmgx", lib.log.seq());
        lib.log = .{};
        try std.testing.expectError(error.KindRange, go.request(fx.FakeMatcher, &o, &lib, 6, "", &err, V, null));
        try std.testing.expectEqualStrings("", lib.log.seq());
        lib.fail_matcher = true;
        try std.testing.expectError(error.NoMatcher, go.request(fx.FakeMatcher, &o, &lib, 0, "", &err, V, null));
        try std.testing.expectEqualStrings("cmg", lib.log.seq());
    }
    // matcher errors poison the Rules; Rules.close frees once (a second close is a no-op); a free error is kept
    inline for (.{ fx.Fail.accept, fx.Fail.terminated, fx.Fail.free }) |fail| {
        var script = fx.Script{ .steps = &.{ &.{7}, &.{8} } };
        script.init(V);
        var lib = fx.FakeLib{ .compiler = .{ .vocab = V }, .script = &script, .fail = fail };
        var o: O = undefined;
        const r = try go.request(fx.FakeMatcher, &o, &lib, 1, "{}", &err, V, null);
        const mask = try r.next(a);
        a.free(mask);
        switch (fail) {
            .accept => {
                try std.testing.expectError(error.Poisoned, r.follow(7));
                try std.testing.expect(r.poisoned);
                try std.testing.expectError(error.PoisonedRules, r.next(a));
            },
            .terminated => {
                try std.testing.expect(try r.follow(7));
                try std.testing.expect(!r.terminated() and r.poisoned);
                try std.testing.expectError(error.PoisonedRules, r.next(a));
            },
            else => {},
        }
        r.close();
        r.close();
        try std.testing.expectEqual(@as(usize, 1), lib.log.frees);
        try std.testing.expect(o.freed and (o.free_error != null) == (fail == .free));
    }
    // the actual chain: the adapter's Rules in place of the script's, the same forbidden winners removed
    const picks = [_][]const fx.Pair{ &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 80 }, .{ .token = 9, .value = 60 } }, &.{ .{ .token = 2, .value = 100 }, .{ .token = 7, .value = 90 }, .{ .token = 8, .value = 70 } } };
    var sched = try fx.Sched.init(V, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var script = fx.Script{ .steps = &.{ &.{ 7, 9 }, &.{8} } };
    script.init(V);
    var lib = fx.FakeLib{ .compiler = .{ .vocab = V }, .script = &script };
    var o: O = undefined;
    const r = try go.request(fx.FakeMatcher, &o, &lib, 0, "", &err, V, null);
    var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_owned", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = r });
    try c.drive(&s);
    try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, s.emitted());
    try std.testing.expectEqual(@as(usize, 0), lib.log.frees);
    s.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), lib.log.frees);
    try std.testing.expect(std.mem.startsWith(u8, lib.log.seq(), "cmgf") and std.mem.endsWith(u8, lib.log.seq(), "x"));
    try std.testing.expectEqual(@as(u64, 2), c.dev.stats.masked);
    try c.clean();
    std.debug.print("grammar_mask owned: FAKE tfg-shaped matcher via lanes.grammar_owned: calls {s}, emitted 7 8, freed once\n", .{lib.log.seq()});
}

/// The device gate (TF_AFD_GRAMMAR_GATE) on the recorded chain with the MOCK device: every case of every row equals
/// the host oracle and the counts add up; a device that ignores the mask fails the gate (the adapter poisons).
pub fn gateCpu(comptime Bound: type) !void {
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    const V = c.rec.m.vocab;
    var gr = try fx.GateRow.init(V);
    defer gr.deinit();
    c.rec.t.picker = gr.picker();
    c.dev.emulate = gr.emulator();
    var lb = glm.views.Buf.of(.bf16, &.{ 1, V });
    try c.rec.space.place("gate.logits", &lb);
    const logits = glm.trace.T.whole(lb);
    const rows = 6;
    const g = try md.gate(&c.dev, &c.f, &c.rec.e.smp, a, logits, gr.sink(), rows, 0x4752_4132);
    try std.testing.expect(g.rows == rows and g.identity == rows and g.forbidden == rows and g.single == rows and g.all_inf == rows);
    try std.testing.expectEqual(@as(u64, 3 * rows), c.dev.stats.masked);
    try std.testing.expectEqual(@as(u64, 3 * rows), c.dev.stats.staged);
    try std.testing.expectEqual(@as(u64, rows), c.dev.stats.identity);
    try std.testing.expectEqual(@as(usize, 3 * rows), try fx.maskedOrder(&c.rec.t));
    std.debug.print("grammar_mask gate (MOCK device): {d} rows, identity {d}, forbidden {d}, single {d}, all-inf {d}; masked {d}\n", .{ g.rows, g.identity, g.forbidden, g.single, g.all_inf, c.dev.stats.masked });
    gr.honest = false;
    try std.testing.expectError(error.MaskInconsistent, md.gate(&c.dev, &c.f, &c.rec.e.smp, a, logits, gr.sink(), 1, 7));
    try std.testing.expect(c.dev.poisoned);
    std.debug.print("grammar_mask gate: a device ignoring the mask fails it (MaskInconsistent, adapter poisoned)\n", .{});
}

/// The host resolution over the kernel's CPU reference and topk's order equals the independent oracle on
/// adversarial rows: NaN, +-inf, signed zeros, ties, eligible all -inf, forbidden winners, single and empty masks.
pub fn exact() !void {
    var prng = std.Random.DefaultPrng.init(0x6752_4132_4146_4400);
    const r = prng.random();
    const specials = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -0.0, 0.0, 1.0, -1.0, 2.0, 2.0, 0.5 };
    var cases: usize = 0;
    var empties: usize = 0;
    var inf_resolved: usize = 0;
    for ([_]usize{ 96, 100, 33 }) |width| {
        const W = (width + 31) / 32;
        var row: [128]f32 = undefined;
        var masked: [128]f32 = undefined;
        var mask: [4]u32 = undefined;
        for (0..6000) |n| {
            const density = r.uintLessThan(u32, 4);
            for (mask[0..W]) |*w| w.* = switch (density) {
                0 => 0,
                1 => @as(u32, 1) << r.int(u5),
                2 => r.int(u32),
                else => ~@as(u32, 0) ^ (@as(u32, 1) << r.int(u5)),
            };
            const tail: u5 = @intCast(width % 32);
            if (tail != 0) mask[W - 1] &= (@as(u32, 1) << tail) - 1;
            for (row[0..width]) |*x| x.* = specials[r.uintLessThan(usize, specials.len)];
            switch (n % 3) { // 0: as drawn; 1: every eligible column -inf, forbidden ones keep NaN/+inf/...; 2: all -inf
                0 => {},
                1 => for (row[0..width], 0..) |*x, t| if (((mask[t / 32] >> @as(u5, @intCast(t % 32))) & 1) != 0) {
                    x.* = -std.math.inf(f32);
                },
                else => @memset(row[0..width], -std.math.inf(f32)),
            }
            const want = fx.oracle(row[0..width], mask[0..W]);
            const p = md.plan(width, mask[0..W]) catch |err| {
                try std.testing.expectEqual(error.EmptyMask, err);
                try std.testing.expect(want == null);
                empties += 1;
                continue;
            };
            @memcpy(masked[0..width], row[0..width]);
            md.maskRow(masked[0..width], mask[0..W]);
            var order: [1]usize = undefined;
            glm.sample.topkOrder(masked[0..width], 1, &order);
            const got = try md.resolve(p, width, mask[0..W], @intCast(order[0]), @bitCast(masked[order[0]]));
            if (got != order[0]) inf_resolved += 1;
            if (got != want.?) {
                std.debug.print("ExactSelectionRuntimeViolation width {d}: got {d} want {d}\n", .{ width, got, want.? });
                return error.ExactSelectionRuntimeViolation;
            }
            cases += 1;
        }
    }
    try std.testing.expect(empties > 0 and inf_resolved > 0);
    std.debug.print("grammar_mask exact: {d} rows equal to the oracle ({d} by the all -inf rule), {d} empty masks refused\n", .{ cases, inf_resolved, empties });
}

/// grammar_join_2 startup scope (users), narrowed by structured_users: TF_AFD_GRAMMAR_MASK=1 with TF_AFD_USERS 2 or 4 now goes on (each
/// users row under its own stream's mask), whatever TF_GLM_SAMPLING says; the values come through the role's own
/// parsers, as afd_attn_gpu.zig's run reads them.
pub fn startupUsers() !void {
    for ([_][]const u8{ "2", "4" }) |users| for ([_]?[]const u8{ null, "0", "1" }) |sampling| {
        try std.testing.expect(md.scopeRefusal(true, try glm.afd_users_round.parse(users), try glm.afd_sampling.parse(sampling)) == null);
    };
    std.debug.print("structured_users startup: TF_AFD_GRAMMAR_MASK=1 with TF_AFD_USERS 2 and 4 goes on (6 cases)\n", .{});
}

/// grammar_join_2 startup scope (sampling), narrowed by structured_users: TF_AFD_GRAMMAR_MASK=1 with TF_GLM_SAMPLING=1 and one user now
/// goes on (the mask before the nucleus draw).
pub fn startupSampling() !void {
    for ([_]?[]const u8{ null, "1" }) |users| {
        try std.testing.expect(md.scopeRefusal(true, try glm.afd_users_round.parse(users), try glm.afd_sampling.parse("1")) == null);
    }
    std.debug.print("structured_users startup: TF_AFD_GRAMMAR_MASK=1 with TF_GLM_SAMPLING=1 goes on (2 cases)\n", .{});
}

/// grammar_join_2 startup scope (unset): without the mask nothing is refused for any users/sampling value (no change), and
/// the mask alone, one user greedy, goes on as grammar_mask built it.
pub fn startupUnset() !void {
    var n: usize = 0;
    for ([_]?[]const u8{ null, "1", "2", "4" }) |users| for ([_]?[]const u8{ null, "", "0", "1" }) |sampling| {
        try std.testing.expect(md.scopeRefusal(false, try glm.afd_users_round.parse(users), try glm.afd_sampling.parse(sampling)) == null);
        n += 1;
    };
    for ([_]?[]const u8{ null, "1" }) |users| for ([_]?[]const u8{ null, "", "0" }) |sampling| {
        try std.testing.expect(md.scopeRefusal(true, try glm.afd_users_round.parse(users), try glm.afd_sampling.parse(sampling)) == null);
        n += 1;
    };
    std.debug.print("grammar_join_2 startup: {d} unset/allowed cases go on unrefused\n", .{n});
}

/// grammar_join_3 startup scope (served), narrowed by structured_users: TF_AFD_GRAMMAR_SERVE=1 with TF_AFD_USERS 2 or 4, or with
/// TF_GLM_SAMPLING=1, serves with the mask and never starts without it (the served switch's own refusal), in
/// afd_attn_gpu.zig's run() order.
pub fn startupServe() !void {
    const g2 = @import("afd_attn_grammar.zig");
    const Outcome = union(enum) { plain, serves, refused: []const u8 };
    const run = struct {
        fn order(masked: bool, pair: bool, serve: ?[]const u8, users: ?[]const u8, sampling: ?[]const u8) !Outcome {
            const n = try glm.afd_users_round.parse(users);
            const lib: ?[]const u8 = if (pair) "/opt/grammar_abi/libtfgrammar.so" else null;
            const digest: ?[]const u8 = if (pair) "82b651017ba77bf137e9abfebc179c94a344a4088515c958a61ee85e78af6ad4" else null;
            const grammar = g2.parse(lib, digest, serve, masked) catch |e| return .{ .refused = @errorName(e) };
            if (md.scopeRefusal(masked, n, masked and try glm.afd_sampling.parse(sampling))) |line| return .{ .refused = line };
            return if (grammar != null) .serves else .plain;
        }
    }.order;
    var without_mask: usize = 0;
    var users_served: usize = 0;
    var sampled_served: usize = 0;
    for ([_]bool{ false, true }) |masked| for ([_]bool{ false, true }) |pair| {
        for ([_][]const u8{ "2", "4" }) |users| for ([_]?[]const u8{ null, "0", "1" }) |sampling| {
            const o = try run(masked, pair, "1", users, sampling);
            if (masked) {
                try std.testing.expect(o == .serves);
                users_served += 1;
            } else {
                try std.testing.expect(o == .refused);
                try std.testing.expectEqualStrings("AfdGrammarServeWithoutMask", o.refused);
                without_mask += 1;
            }
        };
        for ([_]?[]const u8{ null, "1" }) |users| {
            const o = try run(masked, pair, "1", users, "1");
            if (masked) {
                try std.testing.expect(o == .serves);
                sampled_served += 1;
            } else {
                try std.testing.expect(o == .refused);
                try std.testing.expectEqualStrings("AfdGrammarServeWithoutMask", o.refused);
                without_mask += 1;
            }
        }
    };
    var plain: usize = 0;
    for ([_]?[]const u8{ null, "0" }) |serve| for ([_]?[]const u8{ null, "1", "2", "4" }) |users| for ([_]?[]const u8{ null, "", "0", "1" }) |sampling| {
        try std.testing.expect(try run(false, false, serve, users, sampling) == .plain);
        plain += 1;
    };
    var served: usize = 0;
    for ([_]bool{ false, true }) |pair| for ([_]?[]const u8{ null, "1" }) |users| for ([_]?[]const u8{ null, "", "0" }) |sampling| {
        try std.testing.expect(try run(true, pair, "1", users, sampling) == .serves);
        served += 1;
    };
    try std.testing.expectEqual(@as(usize, 32), without_mask + users_served + sampled_served);
    try std.testing.expectEqual(@as(usize, 16), without_mask);
    std.debug.print("structured_users startup: TF_AFD_GRAMMAR_SERVE=1 with TF_AFD_USERS 2/4 or TF_GLM_SAMPLING=1: {d} cases without the mask refused (AfdGrammarServeWithoutMask), with the mask {d} users and {d} sampled cases serve; SERVE unset/0, mask unset: {d} cases unrefused; SERVE=1 MASK=1 one user greedy: {d} cases serve\n", .{ without_mask, users_served, sampled_served, plain, served });
}
