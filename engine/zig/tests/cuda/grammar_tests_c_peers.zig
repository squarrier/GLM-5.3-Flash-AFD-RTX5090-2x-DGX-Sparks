//! grammar_tests_c CPU protocol fixtures; never device or real wire qualification.
const std = @import("std");
const lanes = @import("lanes");
const a = std.testing.allocator;
const control = lanes.structured_control;

fn commitAll(s: *control.Session, descriptor: control.Descriptor) !void {
    var d = descriptor;
    d.op_seq = s.op_seq;
    for (0..s.peer_count) |peer| {
        d.peer = @intCast(peer);
        d.outstanding = s.outstanding[peer];
        _ = try s.ready(d);
    }
}
fn completeAll(s: *control.Session, completion: control.Completion) !void {
    for (0..s.peer_count) |peer| _ = try s.after(try s.permit(@intCast(peer)), completion);
}
fn ledgerWitness(comptime label: []const u8, outcome: ?anyerror) !void {
    if (outcome == null or !std.mem.eql(u8, @errorName(outcome.?), "ControlPoison")) {
        std.debug.print("ExpertPermitLedgerRuntimeViolation: {s}\n", .{label});
        return error.ExpertPermitLedgerRuntimeViolation;
    }
}

test "grammar_tests_c C08 ledger non-request completion cannot issue work" {
    const c = control;
    var s = try c.Session.init(a, 7, 3, 41); defer s.deinit();
    try commitAll(&s, .{ .kind = .reset, .epoch = 7, .peer_count = 3 });
    var outcome: ?anyerror = null;
    _ = s.after(try s.permit(0), .{ .issued = 1 }) catch |err| blk: { outcome = err; break :blk null; };
    try ledgerWitness("non-request issued", outcome);
}
test "grammar_tests_c C08 ledger request completion requires exactly one logical permit" {
    const c = control;
    inline for ([_]u32{0, 2}) |issued| {
        var s = try c.Session.init(a, 7, 3, 41); defer s.deinit();
        try commitAll(&s, .{ .kind = .group_action, .action = .request, .epoch = 7, .peer_count = 3 });
        var outcome: ?anyerror = null;
        _ = s.after(try s.permit(0), .{ .issued = issued }) catch |err| blk: { outcome = err; break :blk null; };
        try ledgerWitness("request delta", outcome);
    }
}
test "grammar_tests_c C08 ledger independently rejects wrong FIFO receipt identity" {
    const c = control;
    inline for (0..8) |field| {
        var s = try c.Session.init(a, 7, 3, 41); defer s.deinit();
        var d = c.Descriptor{ .kind = .group_action, .action = .request, .epoch = 7, .peer_count = 3,
            .group_index = 2, .slot = 1, .chunk_index = 5, .layer = 9,
            .position = 320, .provider_position = 1, .transport_seq = 17, .rows = 64 };
        try commitAll(&s, d); try completeAll(&s, .{ .issued = 1 });
        d.action = .take;
        switch (field) {
            0 => d.group_index += 1, 1 => d.slot = 0, 2 => d.chunk_index += 1,
            3 => d.layer += 1, 4 => d.position += 1, 5 => d.provider_position += 1,
            6 => d.transport_seq += 1, 7 => d.rows -= 1, else => unreachable,
        }
        var outcome: ?anyerror = null;
        commitAll(&s, d) catch |err| { outcome = err; };
        try ledgerWitness("take identity", outcome);
    }
}
test "grammar_tests_c C08 ledger quiesce drains FIFO only and forbids later work" {
    const c = control;
    var s = try c.Session.init(a, 7, 3, 41); defer s.deinit();
    var d = c.Descriptor{ .kind = .group_action, .action = .request, .epoch = 7, .peer_count = 3,
        .transport_seq = 17, .rows = 64 };
    try commitAll(&s, d); try completeAll(&s, .{ .issued = 1 });
    d.slot = 1; d.chunk_index = 1; d.transport_seq = 18; d.position = 64;
    try commitAll(&s, d); try completeAll(&s, .{ .issued = 1 });
    try std.testing.expectEqual(c.Status.ok, s.quiesce(.cancelled));
    d.action = .drain;
    var outcome: ?anyerror = null;
    commitAll(&s, d) catch |err| { outcome = err; };
    try ledgerWitness("drain out of order", outcome);
}
test { _ = @import("grammar_tests_c_rendezvous.zig"); }

fn implemented() !void {
    if (comptime !@hasDecl(lanes, "structured_control")) {
        std.debug.print("PeerAgreementRuntimeViolation: structured chunk Session is absent\n", .{});
        return error.PeerAgreementRuntimeViolation;
    }
}

test "grammar_tests_c C01 absent seam is a runtime failure, never compiler detection" {
    try implemented();
}

test "grammar_tests_c C01 exact v2 header and owned full payload" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        const d = c.Descriptor{ .phase = .prepare, .kind = .step, .peer = 1, .peer_count = 2,
            .epoch = 0x1122334455667788, .op_seq = 0x8877665544332211, .chunk_index = 3,
            .group_index = 4, .slot = 5, .position = 0x1020304050607080, .rows = 1,
            .total_prompt_rows = 0x123456789abcdef0, .vocab_width = 65, .active = true,
            .tokens = &.{7}, .mask = &.{0x80, 0x01020304, 1}, .action_index = 6,
            .action = .select, .layer = 8, .provider_position = 9, .transport_seq = 10 };
        const words = try d.encode(a); defer a.free(words);
        const expected = [_]u32{0x47524332, 2, 1, 6, 1, 2, 0x55667788, 0x11223344,
            0x44332211, 0x88776655, 3, 4, 5, 0x50607080, 0x10203040, 1,
            0x9abcdef0, 0x12345678, 65, 3, 1, 3, 0, 6, 11, 8, 9, 1, 10, 0,
            7, 0x80, 0x01020304, 1};
        try std.testing.expectEqualSlices(u32, &expected, words);
        const bytes = try c.toBytes(a, words); defer a.free(bytes);
        try std.testing.expectEqualSlices(u8, &.{0x32, 0x43, 0x52, 0x47}, bytes[0..4]);
        const decoded_words = try c.fromBytes(a, bytes); defer a.free(decoded_words);
        try std.testing.expectEqualSlices(u32, words, decoded_words);
        const parsed = try c.Descriptor.decode(words);
        try std.testing.expectEqual(d.epoch, parsed.epoch);
        try std.testing.expectEqualSlices(u32, d.mask, parsed.mask);
    } else try implemented();
}

test "grammar_tests_c C01 malformed length signed ids and unused mask bits before allocation" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        const d = c.Descriptor{ .kind = .first, .peer_count = 2, .epoch = 1, .rows = 1,
            .vocab_width = 33, .active = true, .mask = &.{0x80, 1} };
        const w = try d.encode(a); defer a.free(w);
        inline for ([_]usize{0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 27}) |i| {
            const copy = try a.dupe(u32, w); defer a.free(copy);
            copy[i] = std.math.maxInt(u32);
            try std.testing.expectError(error.ControlShape, c.Descriptor.decode(copy));
        }
        try std.testing.expectError(error.ControlShape, c.Descriptor.decode(w[0..w.len-1]));
        const bad_tail = try a.dupe(u32, w); defer a.free(bad_tail);
        bad_tail[bad_tail.len-1] = 2;
        try std.testing.expectError(error.ControlShape, c.Descriptor.decode(bad_tail));
        try std.testing.expectError(error.ControlShape, (c.Descriptor{ .kind = .step,
            .peer_count = 2, .epoch = 1, .rows = 1, .vocab_width = 33,
            .tokens = &.{0xffffffff} }).encode(a));
        try std.testing.expectError(error.ControlShape, c.fromBytes(a, &.{1, 2, 3}));
    } else try implemented();
}

test "grammar_tests_c C01 both READY before any permit and DONE before next sequence" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        var s = try c.Session.init(a, 9, 2, 40); defer s.deinit();
        var d = c.Descriptor{ .kind = .prompt_chunk, .epoch = 9, .op_seq = 40,
            .peer_count = 2, .rows = 2, .total_prompt_rows = 6, .tokens = &.{3, 4} };
        try std.testing.expectEqual(@as(?c.Permit, null), try s.ready(d));
        try std.testing.expectError(error.NotCommitted, s.permit(0));
        d.peer = 1;
        _ = try s.ready(d);
        const p0 = try s.permit(0); const p1 = try s.permit(1);
        try std.testing.expectEqual(@as(u64, 40), p0.op_seq);
        try std.testing.expectEqual(@as(?c.Completion, null), try s.after(p1, .{}));
        var next = d; next.op_seq += 1;
        try std.testing.expectError(error.OperationInProgress, s.ready(next));
        _ = try s.after(p0, .{});
        try std.testing.expectEqual(@as(u64, 41), s.op_seq);
        try std.testing.expectError(error.NotCommitted, s.permit(0));
    } else try implemented();
}

test "grammar_tests_c C01 full-mask epoch sequence and position mismatch poison everyone" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        inline for (0..4) |variant| {
            var s = try c.Session.init(a, 1, 2, 0); defer s.deinit();
            var d = c.Descriptor{ .kind = .first, .epoch = 1, .peer_count = 2,
                .rows = 1, .vocab_width = 65, .active = true, .mask = &.{0x80, 0, 1} };
            _ = try s.ready(d); d.peer = 1;
            switch (variant) {
                0 => d.epoch = 2, 1 => d.op_seq = 1, 2 => d.position = 1,
                3 => d.mask = &.{0x80, 0x80000000, 1}, else => unreachable,
            }
            try std.testing.expectError(error.ControlPoison, s.ready(d));
            try std.testing.expectEqual(c.Status.poison, s.status);
            try std.testing.expectError(error.ControlPoison, s.permit(0));
            try std.testing.expectError(error.ControlPoison, s.ready(d));
        }
    } else try implemented();
}

test "grammar_tests_c C01 owned envelope does not borrow a caller mask" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        var s = try c.Session.init(a, 1, 2, 0); defer s.deinit();
        var mask = [_]u32{0x80, 1};
        var d = c.Descriptor{ .kind = .first, .epoch = 1, .peer_count = 2,
            .rows = 1, .vocab_width = 33, .active = true, .mask = &mask };
        _ = try s.ready(d); mask[0] = 0;
        d.peer = 1; d.mask = &.{0x80, 1};
        _ = try s.ready(d);
        _ = try s.after(try s.permit(0), .{});
        _ = try s.after(try s.permit(1), .{});
    } else try implemented();
}

test "grammar_tests_c C01 quiesce drains both peer ledgers before common cancel" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        var s = try c.Session.init(a, 1, 3, 0); defer s.deinit();
        var d = c.Descriptor{ .kind = .group_action, .epoch = 1, .peer_count = 3,
            .action = .request };
        for (0..3) |peer| { d.peer = @intCast(peer); _ = try s.ready(d); }
        for (0..3) |peer| _ = try s.after(try s.permit(@intCast(peer)), .{ .issued = 1 });
        try std.testing.expectEqual(c.Status.ok, s.quiesce(.cancelled));
        try std.testing.expectEqual(c.Phase.quiesce, s.phase);
        d.op_seq = 1; d.action = .stage; d.outstanding = 1;
        d.rows = 1; d.tokens = &.{3};
        try std.testing.expectError(error.IssueGateClosed, s.ready(d));
        d.action = .take; d.rows = 0; d.tokens = &.{};
        for (0..3) |peer| { d.peer = @intCast(peer); _ = try s.ready(d); }
        for (0..3) |peer| _ = try s.after(try s.permit(@intCast(peer)), .{ .taken = 1 });
        try std.testing.expectEqual(c.Status.cancelled, s.status);
        try std.testing.expectEqual(c.Phase.abort, s.phase);
        try std.testing.expectError(error.Cancelled, s.ready(d));
        for (0..3) |peer| try s.closePeer(@intCast(peer));
        try std.testing.expectEqual(c.Phase.closed, s.phase);
    } else try implemented();
}

test "grammar_tests_c C01 cancel latched after COMMIT does not return one peer early" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        var s = try c.Session.init(a, 1, 2, 0); defer s.deinit();
        var d = c.Descriptor{ .kind = .reset, .epoch = 1, .peer_count = 2 };
        _ = try s.ready(d); d.peer = 1; _ = try s.ready(d);
        const p0 = try s.permit(0); const p1 = try s.permit(1);
        try std.testing.expectEqual(c.Status.ok, s.quiesce(.cancelled));
        _ = try s.permit(1);
        _ = try s.after(p0, .{});
        try std.testing.expectEqual(c.Status.ok, s.status);
        _ = try s.after(p1, .{});
        try std.testing.expectEqual(c.Status.cancelled, s.status);
    } else try implemented();
}

test "grammar_tests_c C01 local empty is valid contribution and both empty is common semantic outcome" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        inline for ([_]bool{false, true}) |both_empty| {
            var s = try c.Session.init(a, 1, 2, 0); defer s.deinit();
            var d = c.Descriptor{ .kind = .first, .epoch = 1, .peer_count = 2,
                .rows = 1, .vocab_width = 64, .active = true, .mask = &.{0, 1}, .action = .select };
            _ = try s.ready(d); d.peer = 1; _ = try s.ready(d);
            _ = try s.after(try s.permit(0), .{});
            const result = (try s.after(try s.permit(1), .{ .candidate = .{
                .valid = !both_empty, .value_f32_bits = @bitCast(@as(f32, 80)), .global_id = 32 } })).?;
            try std.testing.expectEqual(if (both_empty) c.Status.empty_mask else c.Status.ok, result.status);
            if (!both_empty) try std.testing.expectEqual(@as(u32, 32), result.candidate.global_id);
        }
    } else try implemented();
}

test "grammar_tests_c C01 malformed completion and unknown partial action poison" {
    if (comptime @hasDecl(lanes, "structured_control")) {
        const c = lanes.structured_control;
        var s = try c.Session.init(a, 1, 2, 0); defer s.deinit();
        var d = c.Descriptor{ .kind = .reset, .epoch = 1, .peer_count = 2 };
        _ = try s.ready(d); d.peer = 1; _ = try s.ready(d);
        try std.testing.expectError(error.ControlPoison, s.after(try s.permit(0), .{ .taken = 1 }));
        try std.testing.expectError(error.ControlPoison, s.permit(1));
    } else try implemented();
}
