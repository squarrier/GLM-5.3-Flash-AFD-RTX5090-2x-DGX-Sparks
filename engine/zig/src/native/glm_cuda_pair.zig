//! The CUDA GLM's two ranks while they serve (qualification_qualification_44ba554e.8 day 2). Rank 0's mirror is bound to the pair's control link: each
//! operation's words go as one `round` frame before rank 0 runs it (the Metal GLM's rule, Ash's families/glm/mirror.zig).
//! Rank 1's follower replays the frames (glm_cuda_lanes.replay) under the watchdog until rank 0's `stop`, then acks.
//! A lost peer is the watchdog's exit 3 on either rank (Ash on #214: a failed peer fails the request, never a hang).
//! The link and the watchdog are Mia's patch 0006 (MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold cbfc429,
//! Apache-2.0), in cuda/link.zig and cuda/watch.zig.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const gl = @import("glm_cuda_lanes.zig");
const gd = @import("glm_cuda_drafts.zig");
const Allocator = std.mem.Allocator;
const Pair = glm.ranks.Pair;
const Watch = cuda.watch.Watch;

/// A round frame's bytes: the op, then its words, u32 little endian.
pub fn encode(gpa: Allocator, op: gl.Op, words: []const u32) ![]u8 {
    const b = try gpa.alloc(u8, 4 * (1 + words.len));
    std.mem.writeInt(u32, b[0..4], @intFromEnum(op), .little);
    for (words, 0..) |w, i| std.mem.writeInt(u32, b[4 + 4 * i ..][0..4], w, .little);
    return b;
}

/// A round frame's op and words (the words into `gpa`'s memory, the caller's to free).
pub fn decode(gpa: Allocator, bytes: []const u8) !struct { op: gl.Op, words: []u32 } {
    if (bytes.len < 4 or bytes.len % 4 != 0) return error.MirrorFrame;
    const op = std.enums.fromInt(gl.Op, std.mem.readInt(u32, bytes[0..4], .little)) orelse return error.MirrorFrame;
    const words = try gpa.alloc(u32, bytes.len / 4 - 1);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, bytes[4 + 4 * i ..][0..4], .little);
    return .{ .op = op, .words = words };
}

/// Rank 0's side of the mirror: each operation's words as a `round` frame on the pair's link; a failed send means
/// rank 1 is gone (the watchdog's exit 3 when it has one).
pub const Sender = struct {
    gpa: Allocator,
    pair: *Pair,
    sent: u64 = 0,

    pub fn mirror(s: *Sender) gl.Mirror {
        return .{ .ctx = s, .send = send };
    }

    fn send(ctx: ?*anyopaque, op: gl.Op, words: []const u32) anyerror!void {
        const s: *Sender = @ptrCast(@alignCast(ctx.?));
        const bytes = try encode(s.gpa, op, words);
        defer s.gpa.free(bytes);
        s.pair.link.?.send(.round, bytes) catch |e| return s.pair.lost(e);
        s.sent += 1;
    }
};

/// Rank 0's operations over `Steps`, each armed on the watchdog (TF_TP_TIMEOUT_S) while it runs: the calls the
/// lane backend makes after rank 1 heard their words. A failure there leaves the ranks out of step, so it is the
/// pair's end (the watchdog's exit 3), never a request that half ran on one rank.
pub fn Armed(comptime Steps: type) type {
    return struct {
        const Self = @This();
        pub const masked_supported = @hasDecl(Steps, "checkMasked") and @hasDecl(Steps, "prefillToLogits") and @hasDecl(Steps, "pickFirst") and @hasDecl(Steps, "stepMasked");
        steps: *Steps,
        pair: *Pair,
        /// the rows the sequence holds (the lane backend's position checks), as `steps` counts them
        len: usize = 0,

        pub fn prefill(a: *Self, ids: []const u32) !u32 {
            a.pair.arm(0);
            defer a.disarm();
            return a.steps.prefill(ids) catch |e| a.outOfStep(e);
        }

        pub fn step(a: *Self, token: u32) !u32 {
            a.pair.arm(0);
            defer a.disarm();
            return a.steps.step(token) catch |e| a.outOfStep(e);
        }

        pub fn installStructured(a: *Self, endpoint: @import("lanes").structured_control.Endpoint) !void {
            if (comptime !@hasDecl(Steps, "installStructured") or !@hasDecl(Steps, "clearStructured")) return error.IncompleteMaskedAdapter;
            try a.steps.installStructured(endpoint);
        }
        pub fn clearStructured(a: *Self) void {
            if (comptime @hasDecl(Steps, "clearStructured")) a.steps.clearStructured();
            a.len = a.steps.len;
        }
        pub fn maskedWidth(a: *Self) usize { return a.steps.maskedWidth(); }
        pub fn maskedWords(a: *Self) usize {
            return a.steps.maskedWords();
        }
        pub fn checkMasked(a: *Self) !void {
            try a.steps.checkMasked();
        }
        pub fn prefillToLogits(a: *Self, ids: []const u32) !void {
            try a.checkMasked();
            a.pair.arm(0);
            defer a.disarm();
            try a.steps.prefillToLogits(ids);
        }
        pub fn pickFirst(a: *Self, mask: []const u32) !u32 {
            a.pair.arm(0);
            defer a.disarm();
            return a.steps.pickFirst(mask);
        }
        pub fn stepMasked(a: *Self, token: u32, mask: []const u32) !u32 {
            a.pair.arm(0);
            defer a.disarm();
            return a.steps.stepMasked(token, mask);
        }

        pub fn release(a: *Self) void {
            a.steps.release();
            a.len = a.steps.len;
        }

        fn disarm(a: *Self) void {
            a.pair.disarm();
            a.len = a.steps.len;
        }

        fn outOfStep(a: *Self, e: anyerror) anyerror {
            if (a.pair.watch) |*w| {
                std.debug.print("FAIL rank {d}: an operation failed after rank 1 heard its words ({s}): the ranks are out of step\n", .{ a.pair.rank, @errorName(e) });
                w.fail("this rank's operation failed after its words were sent");
            }
            return e;
        }
    };
}

/// What rank 1's follower did, for its last line (with drafts on: the decodes and their operations).
pub const Followed = struct { rounds: u64 = 0, begins: u64 = 0, steps: u64 = 0, releases: u64 = 0, decodes: u64 = 0, drafts: u64 = 0, windows: u64 = 0, keeps: u64 = 0 };

/// Rank 1: replay rank 0's rounds on `steps` until rank 0's `stop` (acked), each armed on the watchdog; a frame of any
/// other kind, or a replay that fails, ends the follower with an error (and the watchdog's exit 3 on a real pair). A
/// `decode` frame runs the drafted loop for `d` (glm_cuda_drafts.follow), which reads its own operations' frames.
pub fn follow(comptime Steps: type, steps: *Steps, pair: *Pair, gpa: Allocator, done: *Followed, d: gd.Drafts) !void {
    const l = pair.link.?;
    while (true) {
        const f = l.recv(gpa) catch |e| return pair.lost(e);
        defer f.deinit(gpa);
        switch (f.tag) {
            .round => {
                const r = try decode(gpa, f.bytes);
                defer gpa.free(r.words);
                done.rounds += 1;
                if (r.op == .decode) {
                    gd.follow(Steps, steps, pair, gpa, d, r.words, done) catch |e| {
                        std.debug.print("FAIL rank 1: following rank 0's drafted decode failed ({s}): the ranks are out of step\n", .{@errorName(e)});
                        if (pair.watch) |*w| w.fail("rank 1's drafted decode failed");
                        return e;
                    };
                    continue;
                }
                pair.arm(0);
                defer pair.disarm();
                gl.replay(Steps, steps, r.op, r.words) catch |e| {
                    std.debug.print("FAIL rank 1: replaying rank 0's {t} failed ({s}): the ranks are out of step\n", .{ r.op, @errorName(e) });
                    if (pair.watch) |*w| w.fail("rank 1's replay failed");
                    return e;
                };
                switch (r.op) {
                    .begin => done.begins += 1,
                    .step => done.steps += 1,
                    .release => done.releases += 1,
                    else => {},
                }
            },
            .stop => {
                if (pair.watch) |*w| w.quiet();
                l.send(.ack, "") catch {};
                return;
            },
            else => {
                std.debug.print("FAIL rank 1: an unexpected frame ({d}) from rank 0\n", .{@intFromEnum(f.tag)});
                return error.UnexpectedFrame;
            },
        }
    }
}

/// Rank 0's clean end of the pair: rank 1 told `stop`, its ack awaited, then the pair closed (Pair.close).
pub fn stop(pair: *Pair) void {
    pair.close();
}

/// Rank 1's end after `follow` returned (its ack sent): the watchdog quieted, the communicator and the link closed.
pub fn closeFollower(pair: *Pair) void {
    if (pair.watch) |*w| {
        w.quiet();
        w.deinit();
    }
    if (pair.comm) |*c| c.deinit();
    if (pair.link) |x| x.close();
    if (pair.server) |x| x.close();
    pair.* = .{ .gpa = pair.gpa, .io = pair.io, .rank = pair.rank };
}

/// Two Links joined in memory (a UNIX socket pair): the CPU tests' pair link.
pub fn memoryLinks() ![2]cuda.link.Link {
    var fds: [2]i32 = undefined;
    const rc = std.os.linux.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM | std.posix.SOCK.CLOEXEC, 0, &fds);
    if (std.posix.errno(rc) != .SUCCESS) return error.SocketPairFailed;
    return .{ .{ .fd = fds[0] }, .{ .fd = fds[1] } };
}
