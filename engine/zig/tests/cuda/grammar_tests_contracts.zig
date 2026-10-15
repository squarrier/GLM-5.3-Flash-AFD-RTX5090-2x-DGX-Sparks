//! grammar_tests CPU operand and ownership gates, independent expected token literals.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const a = std.testing.allocator;
test { _ = @import("grammar_tests_r_window_test.zig"); }
test { _ = @import("grammar_tests_c10_c13_test.zig"); }
fn bit(mask: []u32, token: usize) void { mask[token / 32] |= @as(u32, 1) << @as(u5, @intCast(token % 32)); }

test "grammar_tests N2 N3 pretruncation mapped ids empty rank finite tie NaN and minus infinity" {
    const r = try glm.serve_rec.Rank.init(a, 0, null);
    defer r.deinit(a);
    var f = r.recording();
    var d = try data.Data.init(r.m.vocab, r.m.world);
    defer d.deinit();
    const mask = try a.alloc(u32, try glm.sample.mask_cpu.words(r.m.vocab));
    defer a.free(mask);
    const Cases = enum { mapped, other_rank, both_empty, all, ties, nan_finite, infinities };
    inline for (std.meta.tags(Cases)) |case| {
        @memset(mask, 0);
        d.set();
        var expected: u32 = 7;
        switch (case) {
            .mapped => bit(mask, 7),
            .other_rank => {
                expected = @intCast(r.m.vl() + 1);
                bit(mask, expected);
                d.value(expected, 90);
            },
            .both_empty => {},
            .all => { @memset(mask, std.math.maxInt(u32)); expected = 2; },
            .ties => { bit(mask, 7); bit(mask, 8); d.value(8, 80); },
            .nan_finite => { expected = @intCast(r.m.vl() + 1); bit(mask, 7); bit(mask, expected); d.value(7, std.math.nan(f32)); d.value(expected, 70); },
            .infinities => { bit(mask, 7); bit(mask, 8); d.value(7, -std.math.inf(f32)); d.value(8, -std.math.inf(f32)); },
        }
        r.t.reset();
        if (case == .both_empty) try std.testing.expectError(error.EmptyMask, glm.sample.mask_cpu.pick(&d.cpu, &f, mask)) else try std.testing.expectEqual(expected, try glm.sample.mask_cpu.pick(&d.cpu, &f, mask));
        try std.testing.expectEqual(r.m.world, data.count(&r.t, "grammar_tests.cpu.masked_topk"));
        try std.testing.expectEqual(@as(usize, 1), data.count(&r.t, "grammar_tests.cpu.validity_gather"));
        for (r.t.events.items, 0..) |e, i| if (std.mem.eql(u8, e.name, "grammar_tests.cpu.masked_topk")) {
            try std.testing.expect(i != 0 and std.mem.eql(u8, r.t.events.items[i - 1].name, "grammar_tests.cpu.eligibility"));
        };
        std.debug.print("grammar_tests primitive {s}: expected={d} topk_ranks={d} gather=1\n", .{ @tagName(case), expected, r.m.world });
    }
}
test "grammar_tests N1 absent CPU adapter and N2 malformed sampler dimensions refuse before operations" {
    const r = try glm.serve_rec.Rank.init(a, 0, null);
    defer r.deinit(a);
    var f = r.recording();
    try std.testing.expectError(error.IncompleteMaskedAdapter, glm.sample.maskedReady(&f, &r.e.smp));
    var d = try data.Data.init(r.m.vocab, r.m.world);
    defer d.deinit();
    d.set();
    r.e.smp.masked_cpu = &d.cpu;
    var out: [1]u32 = undefined;
    const wrong = glm.trace.T.of(r.e.buf.logits, 0, &.{ 2, r.m.vl() });
    try std.testing.expectError(error.MaskShape, glm.sample.greedyMasked(&f, &r.e.smp, wrong, &out, &.{}));
    const one = glm.trace.T.of(r.e.buf.logits, 0, &.{ 1, r.m.vl() });
    try std.testing.expectError(error.MaskShape, glm.sample.greedyMasked(&f, &r.e.smp, one, &out, &.{1}));
    var strided = one; strided.stride[1] = 2;
    try std.testing.expectError(error.MaskShape, glm.sample.greedyMasked(&f, &r.e.smp, strided, &out, &.{}));
    var model = r.m; model.world = 0;
    var zero = f; zero.m = &model;
    const ranks = d.cpu.ranks; d.cpu.ranks = &.{};
    try std.testing.expectError(error.MaskShape, glm.sample.maskedReady(&zero, &r.e.smp));
    d.cpu.ranks = ranks;
    try std.testing.expectEqual(@as(usize, 0), r.t.events.items.len);
}
test "grammar_tests N6 transferred Rules closes exactly once on Stream allocation failure" {
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var rules = data.Fake{};
    rules.init(32);
    try std.testing.expectError(error.OutOfMemory, lanes.Stream.init(failing.allocator(), .{ .id = "failed", .prompt = &.{1}, .max_new = 2, .rules = &rules.rules }));
    data.witness("CompiledOwnerTransferRuntimeViolation.N6-init-owner-leak", 1, rules.closes);
    try std.testing.expectEqual(@as(usize, 1), rules.closes);
    rules.rules.close();
    try std.testing.expectEqual(@as(usize, 1), rules.closes);
}
test "grammar_tests N6 mask allocation failure and fill failure do not leak returned storage" {
    var rules = data.Fake{};
    rules.init(32);
    defer rules.rules.close();
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, rules.rules.next(failing.allocator()));
    try std.testing.expectEqual(@as(usize, 0), rules.fills);
    rules.fail_fill = true;
    try std.testing.expectError(error.MaskFillFailed, rules.rules.next(a));
    rules.fail_fill = false;
    const mask = try rules.rules.next(a);
    defer a.free(mask);
    try std.testing.expectEqual(@as(u32, 1 << 7), mask[0]);
}
test "grammar_tests N2 all allowed first and step equal plain recorded calls operands and commits TP and AFD" {
    const r = try glm.serve_rec.Rank.init(a, 0, null);
    defer r.deinit(a);
    const p = try glm.serve_rec.Rank.init(a, 0, null);
    defer p.deinit(a);
    var d = try data.Data.init(r.m.vocab, r.m.world);
    defer d.deinit();
    d.set();
    r.e.smp.masked_cpu = &d.cpu;
    r.t.picker = d.picker(); p.t.picker = d.picker();
    var f = r.recording(); var pf = p.recording();
    var s = glm.serve.Serve{ .e = r.e, .f = &f };
    var plain = glm.serve.Serve{ .e = p.e, .f = &pf };
    const mask = try a.alloc(u32, try glm.sample.mask_cpu.words(r.m.vocab));
    defer a.free(mask);
    @memset(mask, std.math.maxInt(u32));
    try s.prefillToLogits(&.{ 3, 4 });
    try std.testing.expectEqual(@as(u32, 2), try s.pickFirst(mask));
    try std.testing.expectEqual(@as(u32, 2), try plain.prefill(&.{ 3, 4 }));
    try std.testing.expectEqual(try plain.step(2), try s.stepMasked(2, mask));
    try glm.serve_rec.expectSame(&r.space, r.t.events.items, &p.space, p.t.events.items);
    try std.testing.expectEqual(p.e.st.pos, r.e.st.pos);
    var ar: glm.grammar_tests_rec.Rec = undefined; try ar.init(a); defer ar.deinit(a);
    var ap: glm.grammar_tests_rec.Rec = undefined; try ap.init(a); defer ap.deinit(a);
    var ad = try data.Data.init(ar.m.vocab, 1); defer ad.deinit(); ad.set();
    ar.e.smp.masked_cpu = &ad.cpu;
    ar.t.picker = ad.picker(); ap.t.picker = ad.picker();
    var af = glm.fwd.Fwd.recording(&ar.m, &ar.t); var afp = glm.fwd.Fwd.recording(&ap.m, &ap.t);
    var as = glm.afd_serve.Serve{ .at = &ar.at, .f = &af };
    var aps = glm.afd_serve.Serve{ .at = &ap.at, .f = &afp };
    try as.prefillToLogits(&.{ 3, 4 });
    try std.testing.expectEqual(try aps.prefill(&.{ 3, 4 }), try as.pickFirst(mask));
    try std.testing.expectEqual(try aps.step(2), try as.stepMasked(2, mask));
    try glm.serve_rec.expectSame(&ar.space, ar.t.events.items, &ap.space, ap.t.events.items);
    try std.testing.expectEqual(ap.e.st.pos, ar.e.st.pos);
}

test "grammar_tests N7 fake shared tree prunes rejected terminated rows and remaps masks positions" {
    var rules = data.Fake{};
    rules.init(32);
    defer rules.rules.close();
    var plan = try lanes.grammar_window.build(a, &rules.rules, &.{ 7, 9, 8 }, &.{ -1, 0, 0, 1 }, &.{ 10, 11, 11, 12 });
    defer plan.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, plan.rows.items);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0 }, plan.parents.items);
    data.witness("RowMaskRemapRuntimeViolation.N7-row-position-remap", 11, plan.positions.items[1]);
    try std.testing.expectEqualSlices(u64, &.{ 10, 11 }, plan.positions.items);
    try std.testing.expectEqualSlices(u32, &.{ 1 << 7, 1 << 8 }, plan.masks.items);
    try std.testing.expectEqual(@as(usize, 0), rules.accepted);
    try std.testing.expect(!rules.rules.poisoned);
}
test "grammar_tests N7 recursive error rolls back successful child or poisons on rollback failure" {
    inline for (.{ false, true }) |fail_rollback| {
        var rules = data.Fake{ .fail_fill_at = 2, .fail_rollback = fail_rollback };
        rules.init(32);
        defer rules.rules.close();
        const expected = if (fail_rollback) error.RollbackFailed else error.MaskFillFailed;
        try std.testing.expectError(expected, lanes.grammar_window.build(a, &rules.rules, &.{7}, &.{ -1, 0 }, &.{ 10, 11 }));
        try std.testing.expectEqual(fail_rollback, rules.rules.poisoned);
        data.witness("RollbackBalanceRuntimeViolation.N7-recursive-rollback", if (fail_rollback) 1 else 0, rules.accepted);
        try std.testing.expectEqual(@as(usize, if (fail_rollback) 1 else 0), rules.accepted);
    }
}
test "grammar_tests N2 defensive rejected token never crosses emitted prefix" {
    var rules = data.Fake{}; rules.init(32);
    var s = try lanes.Stream.init(a, .{ .id = "reject", .prompt = &.{3, 4}, .max_new = 5, .rules = &rules.rules });
    defer s.deinit(a);
    const result = s.commit(a, &.{2});
    const rejected = if (result) |_| false else |err| err == error.RejectedToken;
    data.witness("RejectedTokenRuntimeViolation.N2-posthoc-rejection", 1, @intFromBool(rejected));
    try std.testing.expectError(error.RejectedToken, result);
    try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
    try std.testing.expectEqual(@as(usize, 0), rules.accepted);
    try std.testing.expect(rules.rules.poisoned);
}
test "grammar_tests N8 missing and ambiguous activation markers fail before attachment" {
    const cases = [_][]const u32{ &.{ 18, 20 }, &.{ 18, 19, 19, 20 } };
    inline for (cases, 0..) |tokens, i| {
        var rules = data.Fake{}; rules.init(32); rules.rules.active = false; rules.rules.after = 19;
        try std.testing.expectError(if (i == 0) error.MissingActivationMarker else error.AmbiguousActivationMarker,
            lanes.Stream.init(a, .{ .id = "invalid-marker", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &rules.rules, .think_close = tokens, .think_end = 19 }));
        try std.testing.expectEqual(@as(usize, 0), rules.accepted);
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        rules.rules.close();
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
    }
}

