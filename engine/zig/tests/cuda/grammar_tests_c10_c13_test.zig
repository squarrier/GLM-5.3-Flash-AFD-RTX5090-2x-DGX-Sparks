//! grammar_tests_c10 C13 FAKE-matcher activation/termination fixtures (counted fake Rules, not real grammar or GLM heads).
//! Every negative prints a named witness immediately before its assertion, so ReleaseSafe names the failure too.
const std = @import("std");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const a = std.testing.allocator;

test "grammar_tests_c10 C13 prompt marker never activates and prompt tokens never reach the matcher" {
    var marked = data.Fake{};
    marked.init(64);
    marked.rules.active = false;
    marked.rules.after = 19;
    // The prompt carries the activation marker; only generated output may activate the grammar.
    var s = try lanes.Stream.init(a, .{ .id = "prompt-marker", .prompt = &.{ 7, 19, 8 }, .max_new = 4, .rules = &marked.rules });
    defer s.deinit(a);
    data.witness("PromptMarkerActivationRuntimeViolation.C13-prompt-marker", 0, @intFromBool(marked.rules.active));
    var open = data.Fake{};
    open.init(64);
    // An active matcher would accept these prompt tokens; prompt tokens are never matcher input.
    var p = try lanes.Stream.init(a, .{ .id = "prompt-input", .prompt = &.{ 7, 8 }, .max_new = 4, .rules = &open.rules });
    defer p.deinit(a);
    data.witness("PromptAcceptedRuntimeViolation.C13-accept-prompt", 0, open.accepted + @intFromBool(open.rules.poisoned));
    try std.testing.expect(!marked.rules.active and marked.accepted == 0 and !marked.rules.poisoned);
    try std.testing.expect(open.accepted == 0 and !open.rules.poisoned and !open.rules.terminated());
    try std.testing.expectEqual(@as(usize, 0), s.emitted().len + p.emitted().len);
}

test "grammar_tests_c10 C13 generated marker activates the NEXT selection and is not matcher input" {
    var rules = data.Fake{};
    rules.init(64);
    rules.rules.active = false;
    rules.rules.after = 19;
    var s = try lanes.Stream.init(a, .{ .id = "generated-marker", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &rules.rules });
    defer s.deinit(a);
    _ = try s.commit(a, &.{91});
    // The selection that produces the marker is still unmasked.
    const before = try rules.rules.next(a);
    defer a.free(before);
    data.witness("GeneratedMarkerNextRuntimeViolation.C13-unmasked-at-marker", 0, before.len + @intFromBool(rules.rules.active));
    const landed = s.commit(a, &.{19}) catch 0;
    data.witness("GeneratedMarkerNextRuntimeViolation.C13-marker-input", 1, @intFromBool(landed == 1 and rules.rules.active and !rules.rules.poisoned and rules.accepted == 0));
    try std.testing.expect(before.len == 0 and landed == 1 and rules.rules.active and !rules.rules.poisoned and rules.accepted == 0);
    // The token after the marker is the first grammar-masked selection.
    const mask = try rules.rules.next(a);
    defer a.free(mask);
    data.witness("GeneratedMarkerNextRuntimeViolation.C13-next-masked", 1 << 7, mask[0]);
    try std.testing.expectEqual(@as(u32, 1 << 7), mask[0]);
    try std.testing.expectEqual(@as(usize, 1), try s.commit(a, &.{7}));
    try std.testing.expectEqual(@as(usize, 1), rules.accepted);
    try std.testing.expectEqualSlices(u32, &.{ 91, 19, 7 }, s.emitted());
}

test "grammar_tests_c10 C13 ambiguous missing and absent activation markers refuse before attachment" {
    // A thinking close without an activation id refuses; it never activates the grammar on the first thinking token.
    inline for (.{
        .{ "ambiguous-marker", @as(?u32, 19), &[_]u32{ 18, 19, 19, 20 }, error.AmbiguousActivationMarker },
        .{ "missing-marker", @as(?u32, 19), &[_]u32{ 18, 20 }, error.MissingActivationMarker },
        .{ "missing-activation-id", @as(?u32, null), &[_]u32{ 18, 19, 20 }, error.MissingActivationMarker },
    }) |case| {
        var rules = data.Fake{};
        rules.init(64);
        rules.rules.active = case[1] == null;
        rules.rules.after = case[1];
        const result = lanes.Stream.init(a, .{ .id = "invalid-marker", .prompt = &.{ 3, 4 }, .max_new = 5, .rules = &rules.rules, .think_close = case[2], .think_end = 19 });
        const refused = if (result) |stream| blk: {
            var owned = stream;
            owned.deinit(a);
            break :blk false;
        } else |err| err == case[3];
        data.witness("ActivationMarkerRefusalRuntimeViolation.C13-" ++ case[0], 1, @intFromBool(refused));
        try std.testing.expect(refused);
        try std.testing.expectEqual(@as(usize, 1), rules.closes);
        try std.testing.expectEqual(@as(usize, 0), rules.accepted);
    }
}

test "grammar_tests_c10 C13 forced close excludes the trail after the activation marker" {
    var rules = data.Fake{};
    rules.init(64);
    rules.rules.active = false;
    rules.rules.after = 19;
    var s = try lanes.Stream.init(a, .{ .id = "forced-trail", .prompt = &.{ 3, 4 }, .max_new = 6, .rules = &rules.rules, .think_budget = 2, .think_close = &.{ 18, 19, 20 }, .think_end = 19 });
    defer s.deinit(a);
    data.witness("ForcedTrailRuntimeViolation.C13-forced-close-length", 2, s.think_close.len);
    _ = try s.commit(a, &.{91});
    try std.testing.expect(s.cutsNext());
    const first = try s.startClose(a);
    data.witness("ForcedTrailRuntimeViolation.C13-forced-trail", 1, s.force.items.len);
    _ = try s.commit(a, &.{first});
    const marker = s.popForce() orelse 0;
    _ = try s.commit(a, &.{marker});
    // After the forced marker the grammar selects the next token; nothing after the marker is forced.
    const trail = s.popForce();
    data.witness("ForcedTrailRuntimeViolation.C13-trail-after-marker", 0, @intFromBool(trail != null));
    try std.testing.expectEqualSlices(u32, &.{ 18, 19 }, s.think_close);
    try std.testing.expect(first == 18 and marker == 19 and trail == null);
    try std.testing.expect(rules.rules.active and !rules.rules.poisoned and rules.accepted == 0);
    try std.testing.expectEqualSlices(u32, &.{ 91, 18, 19 }, s.emitted());
}

test "grammar_tests_c10 C13 grammar termination ends a request whether or not it ignores EOS" {
    // server/chat.zig maps ignore_eos to an empty Request.eos; grammar termination must not depend on that list.
    inline for (.{ .{ "C13-ignored-eos", &[_]u32{} }, .{ "C13-request-eos", &[_]u32{9} } }) |case| {
        var rules = data.Fake{};
        rules.init(64);
        var s = try lanes.Stream.init(a, .{ .id = "termination", .prompt = &.{ 3, 4 }, .max_new = 5, .eos = case[1], .rules = &rules.rules });
        defer s.deinit(a);
        _ = try s.commit(a, &.{7});
        const early = s.finished;
        _ = try s.commit(a, &.{8});
        data.witness("GrammarTerminationRuntimeViolation." ++ case[0], 1, @intFromBool(!early and s.finished and s.reason == .stop and rules.rules.terminated()));
        try std.testing.expect(!early and s.finished and s.reason == .stop and rules.rules.terminated());
        try std.testing.expectEqualSlices(u32, &.{ 7, 8 }, s.emitted());
    }
}
