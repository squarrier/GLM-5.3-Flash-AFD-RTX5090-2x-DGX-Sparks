//! qualification_47b9bf3a's checks of the MTP absorb as each prompt lane's extra layer (TF_AFD_LANES=N with TF_AFD_MTP=1,
//! evidence/qualification_0f991257/WIRE.md §13.11) on the AFD attention node on a CPU, over cuda_afd_lanes_test.zig's LRec (qualification_0f991257's
//! recorder and stand-in exchange over a region of N slots a source): word 6 = word 5 for an in-lane chunk, the MTP
//! request at the last position in the wire's order, the prompt's last chunk one short and absorbed after the group;
//! each chunk's launches equal the solo prefill's (G-trace-lanes, at 8,000 tokens too); groups back to back and a
//! pair's group; a short Turn; the refusals at the MTP layer (a group stopped at its request, a half failing at its
//! take).
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("cuda_fwd.zig");
const lanes = @import("cuda_afd_lanes.zig");
const pt = @import("cuda_afd_pairs_test.zig");
const lt = @import("cuda_afd_lanes_test.zig");

const xchg = cuda.xchg;
const proto = xchg.proto;
const testing = std.testing;
const gpa = testing.allocator;
const Rec = pt.Rec;
const LRec = lt.LRec;
const Log = lt.Log;
const run = lt.run;
const runWith = lt.runWith;
const lh = lt.lh;

test "qualification_47b9bf3a lanes with the MTP head (TF_AFD_MTP=1): 145 tokens (64 + 64 + 17): chunks 0 and 1 absorb as their lanes' extra layer (word 6 = 3: the MTP position 2 in the wire's order), chunk 2 (the prompt's last, 16 next tokens) after the group as a slice; a request that does not absorb has no MTP position (word 6 = 2) and no absorb; each chunk's launches the solo's" {
    {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(145);
        try run(&x, &y, 3, true, 2560, 64, &prompt, 3, 3, &log); // fills: pos_dev for lanes 1 and 2, mtp_pos_dev for lane 1
        defer x.deinit();
        defer y.deinit();
        const want = [_]proto.Header{
            lh(1, 64, 0, .lanes, 3, .{ 0, 64, 1, 3 }), lh(2, 64, 0, .lanes, 3, .{ 1, 17, 1, 3 }), lh(3, 17, 0, .lanes, 3, .{ 2, 64, 1, 3 }),
            lh(1, 64, 1, .lanes, 3, .{ 0, 64, 1, 3 }), lh(2, 64, 1, .lanes, 3, .{ 1, 17, 1, 3 }), lh(3, 17, 1, .lanes, 3, .{ 2, 64, 1, 3 }),
            lh(1, 64, 2, .lanes, 3, .{ 0, 64, 1, 3 }), lh(2, 64, 2, .lanes, 3, .{ 1, 17, 1, 3 }), lh(4, 16, 2, .slice, 3, .{ 0, 0, 0, 0 }),
        };
        try testing.expectEqual(want.len, y.r.s.posts);
        for (want, y.r.s.hdrs[0..want.len]) |w, g| try testing.expectEqual(w, g);
        try testing.expectEqualSlices(i32, &.{ 3, 3, 2, 3, 3, 2, 3, 3, 1 }, y.r.s.counts[0..want.len]);
        try testing.expectEqualSlices(usize, &.{ 0, 0, 0, 1, 2, 3, 4, 5, 8 }, y.r.s.before[0..want.len]);
        try testing.expectEqual(@as(usize, 144), y.r.e.st.mtp_len);
        try testing.expectEqual(@as(u64, 4), y.r.at.forwards); // three lanes and one absorb (solo: three chunks, three absorbs)
        try testing.expectEqual(@as(u64, 6), x.at.forwards);
    }
    { // drafts off: the lanes as qualification_9e49592e's, word 6 the backbone's count
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(145);
        try runWith(&x, &y, 3, true, 2560, 64, &prompt, 3, 2, &log, false);
        defer x.deinit();
        defer y.deinit();
        try testing.expectEqual(@as(usize, 6), y.r.s.posts);
        try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2, 2, 2 }, y.r.s.counts[0..6]);
        try testing.expectEqual(lh(3, 17, 1, .lanes, 3, .{ 2, 64, 1, 3 }), y.r.s.hdrs[5]);
        try testing.expectEqual(@as(usize, 0), y.r.e.st.mtp_len);
    }
}

test "qualification_47b9bf3a G-trace-lanes at 8,000 tokens in 2,048-row chunks (capacity 8,256, the long context past the dense limit) with TF_AFD_MTP's absorbs: four lanes (2,048 x 3 + 1,856, one group: lanes 0-2 absorb as their extra layer, lane 3 after the group) and three (a group of three, all absorbing in-lane, then 1,856 alone): each chunk's launches equal the solo prefill's; the wire's words" {
    const prompt = try gpa.alloc(u32, 8000);
    defer gpa.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast(1000 + (7 * i) % 5000);
    for ([_]u8{ 4, 3 }) |n| {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        try run(&x, &y, n, true, 8256, 2048, prompt, 4, if (n == 4) 3 + 2 else 2 + 2, &log); // pos_dev fills, then mtp_pos_dev's (lanes 1..N-1 absorbing in-lane)
        defer x.deinit();
        defer y.deinit();
        const rows = [_]i32{ 2048, 2048, 2048, 1856 };
        var i: usize = 0;
        for (0..3) |p| for (0..n) |k| {
            if (p == 2 and k == 3) continue; // lane 3 is the prompt's last chunk: one position short
            const next = if (k + 1 == n) rows[0] else rows[k + 1];
            try testing.expectEqual(lh(@intCast(k + 1), rows[k], @intCast(p), .lanes, 3, .{ @intCast(k), next, 1, n }), y.r.s.hdrs[i]);
            try testing.expectEqual(@as(i32, if (k == 3) 2 else 3), y.r.s.counts[i]);
            i += 1;
        };
        if (n == 3) for (0..2) |p| { // chunk 3 alone, its backbone as a slice [0, 2)
            try testing.expectEqual(lh(4, 1856, @intCast(p), .slice, 3, .{ 0, 0, 0, 0 }), y.r.s.hdrs[i]);
            i += 1;
        };
        try testing.expectEqual(lh(5, 1855, 2, .slice, 3, .{ 0, 0, 0, 0 }), y.r.s.hdrs[i]); // chunk 3's absorb (N = 3: after its own forward, fseq 4)
        try testing.expectEqual(i + 1, y.r.s.posts);
        try testing.expectEqual(@as(usize, 8000), y.r.e.st.pos);
        try testing.expectEqual(@as(usize, 7999), y.r.e.st.mtp_len);
        try testing.expectEqual(@as(usize, n), y.r.s.most);
        try testing.expectEqual(@as(u64, 1), y.l.runs);
        std.debug.print("G-trace-lanes N={d} (attention, 8,000 tokens in 2,048-row chunks, MTP absorbs in-lane): {d} launches solo, {d} in lanes; 4 chunks equal; {d} requests, at most {d} in flight; G-cross clean\n", .{ n, x.t.events.items.len, y.r.t.events.items.len, y.r.s.posts, y.r.s.most });
    }
}

test "qualification_47b9bf3a with the MTP head, 512 tokens in 64-row chunks, four lanes: two groups back to back (the absorbs in-lane, so nothing between them): the second's request 3 waits for its first take; chunk 7's absorb after the second; and 350 tokens (a group of four absorbing in-lane, then a pair's group of 64 + 30 whose absorbs run after it): launches chunk by chunk the solo's" {
    {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(512);
        try run(&x, &y, 4, true, 2560, 64, &prompt, 8, 3 + 3 + 3 + 2, &log);
        defer x.deinit();
        defer y.deinit();
        try testing.expectEqual(@as(usize, 12 + 11 + 1), y.r.s.posts);
        for (0..12) |q| try testing.expectEqual(lh(@intCast(q % 4 + 1), 64, @intCast(q / 4), .lanes, 3, .{ @intCast(q % 4), 64, 1, 4 }), y.r.s.hdrs[q]);
        for (0..11) |q| try testing.expectEqual(lh(@intCast(q % 4 + 5), 64, @intCast(q / 4), .lanes, 3, .{ @intCast(q % 4), 64, 5, 4 }), y.r.s.hdrs[12 + q]);
        try testing.expectEqual(@as(i32, 2), y.r.s.counts[15]); // group 2's lane 3: one position short
        try testing.expectEqual(lh(9, 63, 2, .slice, 3, .{ 0, 0, 0, 0 }), y.r.s.hdrs[23]);
        try testing.expectEqual(@as(usize, 12), y.r.s.before[12]); // group 2's first: after group 1's last take
        try testing.expectEqual(@as(usize, 13), y.r.s.before[15]); // its request 3 (lane 3's first) after its first take
        try testing.expectEqual(@as(u64, 2), y.l.runs);
        try testing.expectEqual(@as(usize, 511), y.r.e.st.mtp_len);
    }
    {
        var x: Rec = undefined;
        var y: LRec = undefined;
        var log: Log = .{ .t = &y.r.t };
        defer log.deinit();
        const prompt = pt.ids(350);
        try run(&x, &y, 4, true, 2560, 64, &prompt, 6, 3 + 3 + 1, &log);
        defer x.deinit();
        defer y.deinit();
        try testing.expectEqual(@as(usize, 12 + 4 + 2), y.r.s.posts);
        const pair = [_]proto.Header{ lh(5, 64, 0, .pair, 3, .{ 0, 30, 5, 0 }), lh(6, 30, 0, .pair, 3, .{ 1, 64, 5, 0 }), lh(5, 64, 1, .pair, 3, .{ 0, 30, 5, 0 }), lh(6, 30, 1, .pair, 3, .{ 1, 64, 5, 0 }), lh(7, 64, 2, .slice, 3, .{ 0, 0, 0, 0 }), lh(8, 29, 2, .slice, 3, .{ 0, 0, 0, 0 }) };
        for (pair, y.r.s.hdrs[12..18]) |w, g| try testing.expectEqual(w, g);
        try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 2, 1, 1 }, y.r.s.counts[12..18]);
        try testing.expectEqual(@as(u64, 2), y.l.runs);
    }
}

test "qualification_47b9bf3a lanes refused with the MTP layer in-lane (145 tokens, three lanes; lanes 0 and 1 absorb in-lane): a group stopped at lane 0's MTP request (its post's error, the 7th) dies at the next begin, nothing more sent; a half failing at lane 0's MTP take (the 7th, all 8 requests sent) fails the prefill before its first token, later calls refused at once" {
    const prompt = pt.ids(145);
    { // stopped at the MTP layer's request: the next begin dies (stopped), every later call refuses, nothing more sent
        var y: LRec = undefined;
        try y.init(3, true, 2560, 64);
        defer y.deinit();
        try y.start();
        y.r.s.err_post = 7;
        var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
        try testing.expectError(error.HostPost, y.r.at.prefill(&f, &prompt));
        try testing.expectEqual(@as(usize, 6), y.r.s.posts); // both backbone positions of the three lanes
        try testing.expectEqual(@as(usize, 4), y.r.s.takes); // through lane 0's position 1
        try testing.expectError(error.OutOfStep, y.r.at.step(&f, 7));
        try testing.expectEqual(xchg.proto.Why.stopped, y.r.at.failure().?.why);
        y.r.t.reset();
        try testing.expectError(error.ExpertFailed, y.r.at.step(&f, 7));
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.r.t.events.items.len);
        try testing.expectEqual(@as(usize, 6), y.r.s.posts);
    }
    { // a half fails at lane 0's MTP take: the prefill fails at once, before its first token
        var y: LRec = undefined;
        try y.init(3, true, 2560, 64);
        defer y.deinit();
        try y.start();
        y.r.s.fail_take = 7;
        var f = fwd.Fwd.recording(&y.r.m, &y.r.t);
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        for (y.r.t.events.items) |e| try testing.expect(!std.mem.eql(u8, e.name, "torch.topk"));
        try testing.expectEqual(xchg.Failure{ .why = .silent, .peer = 1, .seq = 8, .gave_up = 0, .breach = 0 }, y.r.at.failure().?);
        try testing.expectEqual(@as(usize, 8), y.r.s.posts);
        y.r.t.reset();
        try testing.expectError(error.ExpertFailed, y.r.at.prefill(&f, &prompt));
        try f.flush();
        try testing.expectEqual(@as(usize, 0), y.r.t.events.items.len);
        try testing.expectEqual(@as(usize, 8), y.r.s.posts);
    }
}

test "qualification_47b9bf3a: a short turn (the MTP layer each lane's extra position, the last lane one short: the prompt's last chunk absorbs after the group): n x count - 1 requests in the wire's order, the last lane's last position never sent, done after lane n-2's MTP take" {
    var t: lanes.Turn = .{ .n = 3, .count = 3, .most = 3, .slots = 3, .short = 1 };
    try std.testing.expectEqual(@as(usize, 8), t.total());
    for (0..2) |p| for (0..3) |k| {
        if (t.next - t.taken == 3) try t.take(t.oldest().?, t.taken / 3);
        try t.request(k, p);
    };
    try t.take(0, 1);
    try t.request(0, 2); // lane 0's MTP request
    try t.take(1, 1);
    try t.request(1, 2); // lane 1's
    try t.take(2, 1);
    try std.testing.expect(!t.may()); // lane 2 is short: nothing more to send
    try std.testing.expectError(error.OutOfStep, t.request(2, 2));
    try std.testing.expect(!t.done());
    try t.take(0, 2);
    try std.testing.expectEqual(@as(?usize, 1), t.oldest());
    try t.take(1, 2);
    try std.testing.expect(t.done());
    try std.testing.expectEqual(@as(?usize, null), t.oldest());
}
