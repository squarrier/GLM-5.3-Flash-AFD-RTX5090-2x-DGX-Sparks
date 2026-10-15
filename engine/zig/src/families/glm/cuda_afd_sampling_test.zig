//! sampling_tests top-k then top-p, with qualification_45c10bcf's strict union cover and keyed draw.
const std = @import("std");
const nd = @import("nucleus_draw.zig");

test "sampling_tests top-k: truncate then renormalize top-p, clamped k and tie fallback" {
    const t = std.testing;
    // Four logits with softmax masses 8,4,2,1; top_k=2 renormalizes to total 12.
    var words = [_]u64{ 0, 15, @bitCast(@as(f64, 3)), @bitCast(@as(f64, 2)), @bitCast(@as(f64, 1)), @bitCast(@as(f64, 0)), 0, 1, 2, 3, 8, 4, 2, 1 };
    const p: nd.Packet = .{ .words = &words, .world = 1, .rows = 1, .count = 4 };
    const sh = nd.Shards.uniform(1, 4);
    var work: [4]nd.Cand = undefined;
    for (0..100) |seed| {
        const s: nd.Sampling = .{ .seed = seed, .top_k = 2, .top_p = 0.6 };
        const got = (try nd.drawRow(p, sh, 0, 3, s, &work)).?;
        try t.expectEqual(@as(u32, 0), got.token);
        const all: nd.Sampling = .{ .seed = seed, .top_k = 999, .top_p = 0.9 };
        const no_k: nd.Sampling = .{ .seed = seed, .top_k = 0, .top_p = 0.9 };
        try t.expectEqual((try nd.drawRow(p, sh, 0, 3, no_k, &work)).?.token, (try nd.drawRow(p, sh, 0, 3, all, &work)).?.token);
        const k_one: nd.Sampling = .{ .seed = seed, .top_k = 1, .top_p = 1 };
        try t.expectEqual(@as(u32, 0), (try nd.drawRow(p, sh, 0, 3, k_one, &work)).?.token);
    }
    var partial_words = [_]u64{ 0, 15, @bitCast(@as(f64, 3)), @bitCast(@as(f64, 2)), 0, 1, 8, 4 };
    const partial: nd.Packet = .{ .words = &partial_words, .world = 1, .rows = 1, .count = 2 };
    try t.expect((try nd.drawRow(partial, sh, 0, 3, .{ .seed = 1, .top_k = 2, .top_p = 1 }, &work)) == null);
    try t.expectEqual(@as(u32, 0), (try nd.drawRow(partial, sh, 0, 3, .{ .seed = 1, .top_k = 1, .top_p = 1 }, &work)).?.token);
    partial_words[3] = partial_words[2];
    try t.expect((try nd.drawRow(partial, sh, 0, 3, .{ .seed = 1, .top_k = 1, .top_p = 1 }, &work)) == null);
}
