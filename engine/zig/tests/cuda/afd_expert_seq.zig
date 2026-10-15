//! qualification_969df694: SEQ's field in the exchange settings (TF_GLM_EXL3_SEQ=1, qualification_b3fba1da's SEQ in the halves' X1 prompt kernel) over
//! G-loop's harness (afd_expert_loop.zig: three nodes in one process). Every node with seq=1 agrees at the handshake
//! and serves as without it (SEQ changes no wire byte); a node with it against nodes without, either way, disagrees
//! (SettingsDiffer) before any request, alone and in qualification_e01f8858's FLIGHT form (pairs=1 inflight=2).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const loop = @import("afd_expert_loop.zig");
const ae = glm.afd_expert;
const xchg = cuda.xchg;
const Harness = loop.Harness;

const gpa = std.testing.allocator;
const testing = std.testing;
const layers = loop.layers;

test "qualification_969df694: TF_GLM_EXL3_SEQ=1 on all three nodes: the settings agree with seq=1 last (alone, and after pairs=1 inflight=2), and each forward's replies are exact, as without it" {
    const rows = [_]usize{ 32, 1, 7 };
    for ([_]bool{ false, true }) |flight| {
        var h: Harness = .{ .io = std.testing.io, .seq = true, .pairs = flight, .inflight = if (flight) 2 else 1 };
        try h.start(20 * std.time.ns_per_s);
        defer h.free();
        var wrong: u64 = 0;
        for (rows, 1..) |R, f| wrong += try h.forward(@intCast(f), R, null, layers);
        loop.settle();
        const failed = h.attn().failure();
        var tb: [256]u8 = undefined;
        const text = try h.attn().cfg.settings.text(&tb);
        h.stop();
        try testing.expectEqual(@as(u64, 0), wrong);
        try testing.expectEqual(@as(?xchg.Failure, null), failed);
        try testing.expect(std.mem.endsWith(u8, text, if (flight) " ports=1 pairs=1 inflight=2 seq=1" else " ports=1 seq=1"));
        for (h.ends, h.stats) |e, s| {
            try testing.expectEqual(ae.End.stopped, e);
            try testing.expectEqual(ae.Stats{ .turns = rows.len, .forwards = rows.len, .layers = rows.len * layers, .rows = layers * (32 + 1 + 7) }, s);
        }
    }
}

test "qualification_969df694 refused: SEQ on one side only (the attention with seq=1 and halves without, and the reverse; alone and with pairs=1 inflight=2): the texts disagree at the handshake (SettingsDiffer) before any request" {
    for ([_]bool{ false, true }) |flight| {
        for ([_][2]bool{ .{ true, false }, .{ false, true } }) |ab| {
            var h: Harness = .{ .io = std.testing.io, .seq = ab[1], .attn_seq = ab[0], .pairs = flight, .inflight = if (flight) 2 else 1 };
            const got = h.start(20 * std.time.ns_per_s);
            if (got) |_| { // the texts agreed (a node without seq=1 in its text): everything down, then the test fails
                h.free();
                return error.TestExpectedSettingsDiffer;
            } else |_| {}
            // the attention refused at the first hello: close what is up (links, then the listener), then the halves' ends
            for (h.nodes[0].links) |l| if (l) |x| x.close();
            h.nodes[0].links = .{ null, null };
            h.server.close();
            for (h.threads) |t| if (t) |x| x.join();
            for (1..3) |i| h.nodes[i].deinit();
            h.nodes[0].deinit();
            h.bufs.deinit(h.engine(), gpa);
            for (h.regions) |r| gpa.free(r);
            for (h.status) |s| gpa.free(s);
            try testing.expectError(error.SettingsDiffer, got);
            var differ: usize = 0;
            for (&h.experts, h.ends) |*x, e| {
                try testing.expect(e == .err); // no forward ran: the role ends in AFD FAILED <why>, exit 1
                differ += @intFromBool(e.err == error.SettingsDiffer);
                try testing.expectEqual(@as(usize, 0), x.seen.items.len);
                x.deinit();
            }
            try testing.expect(differ >= 1); // the one the attention answered saw the texts differ; the other found it gone
        }
    }
}
