//! Bounded in-memory fixture synchronization; no socket, engine, or transport implementation.
const std = @import("std");
const c = @import("lanes").structured_control;
const io = std.testing.io;

pub const Rendezvous = struct {
    session: c.Session,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    opened: [3]bool = @splat(false),
    returned: u32 = 0,
    common: c.Completion = .{},
    cancel_ctx: ?*anyopaque = null,
    cancel_check: ?*const fn (*anyopaque) bool = null,
    cancel_kind: ?c.Kind = null,
    cancel_chunk: ?u32 = null,
    cancel_action: ?c.Action = null,
    before_calls: [3]u32 = @splat(0),
    after_calls: [3]u32 = @splat(0),
    close_calls: [3]u32 = @splat(0),

    pub fn init(gpa: std.mem.Allocator, epoch: u64, peers: u32, seq: u64) !Rendezvous {
        return .{ .session = try c.Session.init(gpa, epoch, peers, seq) };
    }
    pub fn install(r: *Rendezvous) void {
        r.session.hooks = .{ .ctx = r, .open = open, .before = before, .after = after, .quiesce = quiesce, .close = close };
    }
    pub fn deinit(r: *Rendezvous) void { r.session.deinit(); }
    fn from(p: *anyopaque) *Rendezvous { return @ptrCast(@alignCast(p)); }
    fn deadline() std.Io.Timeout {
        return .{ .deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(2), .clock = .awake }) };
    }
    fn wait(r: *Rendezvous, timeout: std.Io.Timeout) !void {
        r.wake.waitTimeout(io, &r.mutex, timeout) catch {
            const err = r.session.poison();
            r.wake.broadcast(io);
            return err;
        };
    }
    fn open(p: *anyopaque, peer: u32) !void {
        const r = from(p); r.mutex.lockUncancelable(io); defer r.mutex.unlock(io);
        if (peer >= r.session.peer_count or r.opened[peer]) return error.ControlShape;
        r.opened[peer] = true;
        r.wake.broadcast(io);
    }
    fn before(p: *anyopaque, descriptor: c.Descriptor) !c.Permit {
        const r = from(p);
        const cancelled = descriptor.peer == 0 and
            ((r.cancel_check != null and r.cancel_check.?(r.cancel_ctx.?)) or
             (r.session.cancel_check != null and r.session.cancel_check.?(r.session.cancel_ctx.?)));
        r.mutex.lockUncancelable(io); defer r.mutex.unlock(io);
        const timeout = deadline();
        while (r.returned != 0 and r.returned < r.session.peer_count and r.session.status == .ok) try r.wait(timeout);
        if (r.returned == r.session.peer_count) r.returned = 0;
        var d = descriptor;
        d.op_seq = r.session.op_seq;
        if (d.peer >= r.session.peer_count or !r.opened[d.peer]) return r.session.poison();
        r.before_calls[d.peer] += 1;
        if (cancelled or (r.cancel_kind == d.kind and (r.cancel_chunk == null or r.cancel_chunk == d.chunk_index) and (r.cancel_action == null or r.cancel_action == d.action))) {
            _ = r.session.quiesce(.cancelled);
        }
        const result = r.session.ready(d) catch |err| { r.wake.broadcast(io); return err; };
        if (result != null) r.wake.broadcast(io);
        while (!r.session.committed and r.session.status == .ok) try r.wait(timeout);
        return r.session.permit(d.peer);
    }
    fn after(p: *anyopaque, permit: c.Permit, completion: c.Completion) !c.Completion {
        const r = from(p);
        const cancelled = permit.peer == 0 and
            ((r.cancel_check != null and r.cancel_check.?(r.cancel_ctx.?)) or
             (r.session.cancel_check != null and r.session.cancel_check.?(r.session.cancel_ctx.?)));
        r.mutex.lockUncancelable(io); defer r.mutex.unlock(io);
        if (cancelled) _ = r.session.quiesce(.cancelled);
        const timeout = deadline();
        r.after_calls[permit.peer] += 1;
        const result = r.session.after(permit, completion) catch |err| { r.wake.broadcast(io); return err; };
        if (result) |v| { r.common = v; r.wake.broadcast(io); }
        while (r.session.committed and r.session.status == .ok) try r.wait(timeout);
        if (r.session.status == .poison) return error.ControlPoison;
        const common = r.common;
        r.returned += 1;
        r.wake.broadcast(io);
        return common;
    }
    fn quiesce(p: *anyopaque, _: u32, reason: c.Status) !c.Status {
        const r = from(p); r.mutex.lockUncancelable(io); defer r.mutex.unlock(io);
        const result = r.session.quiesce(reason); r.wake.broadcast(io); return result;
    }
    fn close(p: *anyopaque, peer: u32) void {
        const r = from(p); r.mutex.lockUncancelable(io); defer r.mutex.unlock(io);
        const timeout = deadline();
        r.close_calls[peer] += 1;
        r.session.closePeer(peer) catch { const err = r.session.poison(); _ = @errorName(err); r.wake.broadcast(io); return; };
        r.wake.broadcast(io);
        while (r.session.phase != .closed and r.session.status != .poison) r.wait(timeout) catch return;
    }
};

const Worker = struct {
    endpoint: c.Endpoint,
    count: u32 = 0,
    outcome: ?anyerror = null,
    fn run(w: *Worker) void { w.transaction() catch |err| { w.outcome = err; }; }
    fn transaction(w: *Worker) !void {
        try w.endpoint.open();
        defer w.endpoint.close();
        for (0..3) |chunk| {
            const d = c.Descriptor{ .kind = .prompt_chunk, .epoch = 1, .peer_count = 2,
                .rows = 2, .tokens = &.{3, 4}, .chunk_index = @intCast(chunk), .position = chunk*2, .total_prompt_rows = 6 };
            const permit = try w.endpoint.before(d);
            w.count += 1;
            const result = try w.endpoint.after(permit, .{ .logits_valid = true });
            if (result.status == .cancelled) return error.Cancelled;
        }
    }
};

test "grammar_tests_c C01 two independent workers READY COMMIT DONE close and chunk cancellation" {
    inline for ([_]?u32{null, 0, 1, 2}) |cancel_chunk| {
        var r = try Rendezvous.init(std.testing.allocator, 1, 2, 0); defer r.deinit();
        r.cancel_kind = if (cancel_chunk == null) null else .prompt_chunk; r.cancel_chunk = cancel_chunk; r.install();
        var workers = [_]Worker{.{ .endpoint = try r.session.endpoint(0) }, .{ .endpoint = try r.session.endpoint(1) }};
        const thread = try std.Thread.spawn(.{}, Worker.run, .{&workers[1]});
        Worker.run(&workers[0]); thread.join();
        for (workers) |w| {
            try std.testing.expectEqual(cancel_chunk orelse 3, w.count);
            if (cancel_chunk == null) {
                try std.testing.expect(w.outcome == null);
            } else {
                try std.testing.expect(w.outcome != null);
                try std.testing.expectEqualStrings("Cancelled", @errorName(w.outcome.?));
            }
        }
        try std.testing.expectEqual(c.Phase.closed, r.session.phase);
        try std.testing.expectEqual(r.before_calls[0], r.before_calls[1]);
        try std.testing.expectEqual(r.after_calls[0], r.after_calls[1]);
        try std.testing.expectEqual(@as(u32, 1), r.close_calls[0]);
        try std.testing.expectEqual(@as(u32, 1), r.close_calls[1]);
    }
}
