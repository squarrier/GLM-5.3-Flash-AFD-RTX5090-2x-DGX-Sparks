//! Drafts in the served stream (qualification_qualification_44ba554e.8 day 3): `serve --drafts 0|N|fN|a|a:LOW:HIGH|cN:P|auto`, qualification_qualification_e0f4f003's policy kinds
//! (tf-glm-forward's glm_drafts.parse; 0, the default, is off). With drafts on, the open loads the MTP layer and the
//! prompt pass absorbs into it (decode.prefill(mtp=True)); then the lane host's driver for a lone greedy stream decodes
//! it by qualification_qualification_86ea5170's loop for the policy (cuda_decode.mtpWith or autoWith: the draft, the verify window, the keep), not a
//! rewrite of it. The loop's hook sends each operation's words to rank 1 before rank 0 runs it (the Metal GLM's rule,
//! Ash's families/glm/mirror.zig), arms the watchdog around it, and lands each round's kept tokens in the stream, whose
//! own rules (eos, stop strings, max_tokens) or a cancel end the decode after that round. Rank 1 runs the same loop on
//! the `decode` frame, checks each of its operations' words against rank 0's frame before running it, and ends where
//! rank 0's `release` stands. A request whose prompt and max_tokens pass `reach` (the dense limit by default, until a
//! lab card proves drafted windows past it; TF_GLM_DRAFT_REACH) is handed to the lane core after its prompt pass, and
//! decodes in qualification_qualification_44ba554e.8's one-row rounds, as with drafts off.
const std = @import("std");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const tool = @import("glm_tool");
const gl = @import("glm_cuda_lanes.zig");
const gp = @import("glm_cuda_pair.zig");
const decode = glm.decode;
const Pair = glm.ranks.Pair;
const Allocator = std.mem.Allocator;

pub const Drafts = tool.drafts.Drafts;

/// `--drafts`'s policy (null or 0: off); a malformed one is refused in words (null, `problem` set).
pub fn parse(a: Allocator, spec: ?[]const u8, problem: *[]const u8) !?Drafts {
    const s = spec orelse return Drafts{};
    return tool.drafts.parse(s) catch |e| {
        problem.* = if (e == error.BadDraftAuto) try std.fmt.allocPrint(a, "--drafts {s}: TF_GLM_DRAFT_AUTO is cN:P[:F] (0 < P < 1, 0 <= F < 1) or unset", .{s}) else try std.fmt.allocPrint(a, "--drafts {s}: 0 (off), N or fN (N fixed), a or a:LOW:HIGH (adaptive, 1-3), cN:P[:F] (N under a confidence floor P, the first draft's F) or auto (python's greedy c3:0.35)", .{s});
        return null;
    };
}

/// What the open does with drafts, in the start lines' words: "off", or the policy as given.
pub fn describe(d: Drafts) []const u8 {
    return if (d.on()) d.spec else "off";
}

fn opOf(op: decode.Hook.Op) gl.Op {
    return switch (op) {
        .draft => .draft,
        .window => .window,
        .keep => .keep,
    };
}

/// The decoded counters into the stream's (the server's `done` line and /metrics), and rank 0's line for the lab.
fn noted(s: *lanes.Stream, d: Drafts, r: *const decode.Decoded) void {
    s.rounds += r.rounds;
    s.drafted += r.drafted;
    s.accepted += r.accepted;
    std.debug.print("INFO rank 0: drafts {s}: {d} tokens in {d} verify forwards ({d:.3} tokens a verify forward), drafted {d}, accepted {d} ({d:.4}); depths {any}; keeps {any}; {d:.1} ms\n", .{ d.spec, r.tokens.len -| 1, r.rounds, r.perForward(), r.drafted, r.accepted, r.acceptance(), r.depths, r.keeps, @as(f64, @floatFromInt(r.ns)) / 1e6 });
}

/// Rank 0's driver for a lone drafted stream over `Steps` (gp.Armed of a Serve), run by the lane host while the
/// stream is alone (native/cuda.zig's LoneRun with this as its context).
pub fn Lone(comptime Steps: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        back: *gl.Lanes(Steps),
        d: Drafts,
        /// prompt plus max_tokens a drafted stream may reach; past it, the lane core's one-row rounds
        reach: usize,

        /// Prefill `s` (absorbing), then decode it by the policy's loop until it finishes (false), or hand it to the
        /// lane core right after the prompt pass when it passes `reach` (true).
        pub fn run(l: *Self, s: *lanes.Stream, ctx: *anyopaque, committed: *const fn (*anyopaque) void) !bool {
            const be = l.back.backend();
            _ = try be.opening(l.gpa, s); // the prompt pass and its first token, committed; begin told (Lanes.prefill)
            committed(ctx);
            if (!s.finished and s.prompt_len + s.max_new > l.reach) {
                s.cache_len = l.back.steps.len; // the lane core's adopt: the cache settled, `pending` the last token
                s.pending = s.context.items[s.context.items.len - 1];
                return true;
            }
            defer be.release(s);
            if (s.finished) return false;
            const count: usize = 1 + (s.max_new - s.emitted().len); // the pending token, then what is left
            const out = try l.gpa.alloc(u32, count);
            defer l.gpa.free(out);
            const a = l.back.steps;
            var h: Hooked = .{ .gpa = l.gpa, .s = s, .ctx = ctx, .committed = committed, .mirror = l.back.mirror, .pair = a.pair };
            const hook: decode.Hook = .{ .ctx = &h, .before = Hooked.before, .kept = Hooked.kept };
            const pending = s.context.items[s.context.items.len - 1];
            try l.back.mirror.tell(.decode, &.{ pending, @intCast(count) });
            const r = tool.drafts.runWith(a.steps.e, a.steps.f, l.d, pending, count, out, null, &hook) catch |e| {
                a.pair.disarm();
                return outOfStep(a.pair, e); // rank 1 heard the decode: a failed operation leaves the ranks apart
            };
            a.pair.disarm();
            a.steps.len = a.steps.e.st.pos;
            a.len = a.steps.len;
            noted(s, l.d, &r);
            if (h.failed) |e| return e;
            if (!s.finished and s.isCancelled()) return error.Cancelled;
            return false;
        }
    };
}

fn outOfStep(pair: *Pair, e: anyerror) anyerror {
    if (pair.watch) |*w| {
        std.debug.print("FAIL rank {d}: a drafted operation failed after rank 1 heard its words ({s}): the ranks are out of step\n", .{ pair.rank, @errorName(e) });
        w.fail("this rank's drafted operation failed after its words were sent");
    }
    return e;
}

/// Rank 0's hook: each operation's words to rank 1, then the watchdog armed while it runs; each round's kept tokens
/// landed in the stream, which may end the decode.
const Hooked = struct {
    gpa: Allocator,
    s: *lanes.Stream,
    ctx: *anyopaque,
    committed: *const fn (*anyopaque) void,
    mirror: gl.Mirror,
    pair: *Pair,
    failed: ?anyerror = null,

    fn before(p: *anyopaque, op: decode.Hook.Op, words: []const u32) anyerror!void {
        const h: *Hooked = @ptrCast(@alignCast(p));
        h.pair.disarm();
        try h.mirror.tell(opOf(op), words);
        h.pair.arm(0);
    }

    fn kept(p: *anyopaque, tokens: []const u32, _: usize, _: usize) anyerror!bool {
        const h: *Hooked = @ptrCast(@alignCast(p));
        h.pair.disarm();
        _ = h.s.commit(h.gpa, tokens) catch |e| { // no frame is out: rank 1 ends on the release that follows
            h.failed = e;
            return true;
        };
        h.committed(h.ctx);
        return h.s.finished or h.s.isCancelled();
    }
};

/// Rank 1's side of a `decode` frame (pending token, count): the same loop on `serve`, each operation's words checked
/// against rank 0's next frame before it runs (armed on the watchdog); rank 0's `release` in their place ends it.
pub fn follow(comptime S: type, serve: *S, pair: *Pair, gpa: Allocator, d: Drafts, words: []const u32, done: *gp.Followed) !void {
    if (words.len != 2 or !d.on()) return error.MirrorOutOfStep;
    const out = try gpa.alloc(u32, words[1]);
    defer gpa.free(out);
    var h: Following = .{ .gpa = gpa, .pair = pair, .done = done };
    const hook: decode.Hook = .{ .ctx = &h, .before = Following.before, .kept = Following.kept };
    done.decodes += 1;
    _ = tool.drafts.runWith(serve.e, serve.f, d, words[0], words[1], out, null, &hook) catch |e| {
        pair.disarm();
        if (e != error.Released) return e;
        serve.release(); // rank 0 ended the stream after a round: its release stands where the next operation would
        done.releases += 1;
        return;
    };
    pair.disarm();
    serve.len = serve.e.st.pos;
}

const Following = struct {
    gpa: Allocator,
    pair: *Pair,
    done: *gp.Followed,

    fn before(p: *anyopaque, op: decode.Hook.Op, words: []const u32) anyerror!void {
        const h: *Following = @ptrCast(@alignCast(p));
        h.pair.disarm();
        const f = h.pair.link.?.recv(h.gpa) catch |e| return h.pair.lost(e);
        defer f.deinit(h.gpa);
        if (f.tag != .round) {
            std.debug.print("FAIL rank 1: an unexpected frame ({d}) inside a drafted decode\n", .{@intFromEnum(f.tag)});
            return error.UnexpectedFrame;
        }
        const r = try gp.decode(h.gpa, f.bytes);
        defer h.gpa.free(r.words);
        h.done.rounds += 1;
        if (r.op == .release) return error.Released;
        if (r.op != opOf(op) or !std.mem.eql(u32, r.words, words)) {
            std.debug.print("FAIL rank 1: rank 0's {t} {any} against this rank's {t} {any}: the ranks are out of step\n", .{ r.op, r.words, opOf(op), words });
            return error.MirrorOutOfStep;
        }
        switch (op) {
            .draft => h.done.drafts += 1,
            .window => h.done.windows += 1,
            .keep => h.done.keeps += 1,
        }
        h.pair.arm(0);
    }

    fn kept(p: *anyopaque, _: []const u32, _: usize, _: usize) anyerror!bool {
        const h: *Following = @ptrCast(@alignCast(p));
        h.pair.disarm();
        return false; // rank 0 decides where the stream ends
    }
};

/// The load's warm-up with drafts on, on both ranks in lockstep with no frames: the 96 fixed ids (absorbed), then
/// gl.warm_tokens tokens by the policy's loop; its tokens equal the drafts-off warm-up's (greedy).
pub fn warmUp(comptime S: type, serve: *S, pair: *Pair, d: Drafts, out: *[gl.warm_tokens]u32) !decode.Decoded {
    var ids: [gl.warm_prompt]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    pair.arm(0);
    defer pair.disarm();
    const first = try serve.prefill(&ids);
    const r = try tool.drafts.runWith(serve.e, serve.f, d, first, gl.warm_tokens, out, null, null);
    serve.len = serve.e.st.pos;
    serve.release();
    return r;
}

/// TF_GLM_DRAFT_REACH: the positions a drafted stream may reach (null: the dense limit; never past `reach`).
pub fn reachOf(v: ?usize, dense_limit: usize, reach: usize) usize {
    return @min(v orelse dense_limit, reach);
}

test "--drafts: off by default and at 0, qualification_qualification_e0f4f003's kinds, a bad one refused in words; the reach" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expect(!(try parse(a, null, &problem)).?.on());
    try std.testing.expect(!(try parse(a, "0", &problem)).?.on());
    try std.testing.expectEqualStrings("off", describe((try parse(a, "0", &problem)).?));
    const auto = (try parse(a, "auto", &problem)).?;
    try std.testing.expect(auto.on() and auto.kind == .auto);
    try std.testing.expectEqualStrings("auto", describe(auto));
    const c3 = (try parse(a, "c3:0.35", &problem)).?;
    try std.testing.expect(c3.policy.fixed and c3.policy.most == 3 and c3.policy.confidence == 0.35);
    try std.testing.expectEqual(@as(usize, 7), (try parse(a, "f9", &problem)).?.policy.most); // MAX_ROWS - 1
    try std.testing.expectEqual(@as(?Drafts, null), try parse(a, "lots", &problem));
    try std.testing.expect(std.mem.indexOf(u8, problem, "--drafts lots: 0 (off)") != null);
    try std.testing.expectEqual(@as(usize, 2051), reachOf(null, 2051, 4096));
    try std.testing.expectEqual(@as(usize, 4096), reachOf(9999, 2051, 4096));
    try std.testing.expectEqual(@as(usize, 1000), reachOf(1000, 2051, 4096));
    try std.testing.expectEqual(@as(usize, 2051), reachOf(null, 2051, 2051 + 64));
}
