//! grammar_tests_c7 actual native Host.serve -> LaneHost -> Host.backend -> gl.Lanes(Armed(Serve)) structured session with two
//! recording TP ranks and a request-owned FAKE compiler: caller-owned prompt/text/eos/chunks/think_close slices, and one
//! targeted peer READY frame allocation failure after transfer. Claude Opus 5.5 via Hermes; CPU fixtures, not CUDA.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const Driver = @import("grammar_tests_c_tp_test.zig").Driver;
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const c = lanes.structured_control;
const seam = api.TestHostTransitions;
const a = std.testing.allocator;
const io = std.testing.io;
const Armed = pair.Armed(glm.serve.Serve);
const Backend = gl.Lanes(Armed);
const text = "grammar_tests_c7-borrowed";

fn now() i96 { return std.Io.Clock.awake.now(io).toNanoseconds(); }
fn me() u64 { return @as(u64, @intCast(std.Thread.getCurrentId())) + 1; }
fn check(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.HostPeerFrameRuntimeViolation; }
}

/// The Session ledger's allocator: fails exactly one allocation, the next one made on the thread that armed it.
const FrameFail = struct {
    armed: std.atomic.Value(u64) = .init(0),
    fired: std.atomic.Value(u32) = .init(0),
    fired_ns: i96 = 0,
    fn allocator(x: *FrameFail) std.mem.Allocator {
        return .{ .ptr = x, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(p: *anyopaque, n: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const x: *FrameFail = @ptrCast(@alignCast(p));
        const id = me();
        if (x.armed.load(.acquire) == id and x.armed.cmpxchgStrong(id, 0, .acq_rel, .acquire) == null) {
            x.fired_ns = now();
            _ = x.fired.fetchAdd(1, .acq_rel);
            return null;
        }
        return a.rawAlloc(n, alignment, ra);
    }
    fn resize(_: *anyopaque, b: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) bool { return a.rawResize(b, alignment, n, ra); }
    fn remap(_: *anyopaque, b: []u8, alignment: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 { return a.rawRemap(b, alignment, n, ra); }
    fn free(_: *anyopaque, b: []u8, alignment: std.mem.Alignment, ra: usize) void { a.rawFree(b, alignment, ra); }
};

/// Caller-owned request: every slice the Stream borrows stays alive through finished consumption, callback return,
/// the LaneHost join and the peer join; release refuses any earlier attempt.
const Lease = struct {
    prompt: []u32, text: []u8, eos: []u32, chunks: []u32, think_close: []u32,
    request: api.Request = undefined,
    finished: bool = false,
    returned: std.atomic.Value(bool) = .init(false),
    host_joined: bool = false,
    peers_joined: bool = false,
    refusals: usize = 0,
    fn init() !Lease {
        const prompt = try a.dupe(u32, &.{ 3, 4 }); errdefer a.free(prompt);
        const t = try a.dupe(u8, text); errdefer a.free(t);
        const eos = try a.dupe(u32, &.{ 99, 98 }); errdefer a.free(eos);
        const chunks = try a.dupe(u32, &.{1}); errdefer a.free(chunks);
        const think_close = try a.dupe(u32, &.{ 18, 19, 20 });
        return .{ .prompt = prompt, .text = t, .eos = eos, .chunks = chunks, .think_close = think_close,
            .request = .{ .prompt = prompt, .max_tokens = 5, .eos = eos, .chunks = chunks, .drafts = true, .think_budget = 2,
                .think_close = think_close, .think_end = 19, .structure = .{ .kind = .json_schema, .text = t, .after = 19 } } };
    }
    fn intact(l: *const Lease) bool {
        return std.mem.eql(u32, l.prompt, &.{ 3, 4 }) and std.mem.eql(u8, l.text, text) and std.mem.eql(u32, l.eos, &.{ 99, 98 }) and
            std.mem.eql(u32, l.chunks, &.{1}) and std.mem.eql(u32, l.think_close, &.{ 18, 19, 20 });
    }
    fn release(l: *Lease) !void {
        if (!l.finished or !l.returned.load(.acquire) or !l.host_joined or !l.peers_joined) { l.refusals += 1; return error.OutstandingRequest; }
        l.deinit();
    }
    fn deinit(l: *Lease) void {
        if (l.prompt.len != 0) a.free(l.prompt);
        if (l.text.len != 0) a.free(l.text);
        if (l.eos.len != 0) a.free(l.eos);
        if (l.chunks.len != 0) a.free(l.chunks);
        if (l.think_close.len != 0) a.free(l.think_close);
        l.prompt = &.{}; l.text = &.{}; l.eos = &.{}; l.chunks = &.{}; l.think_close = &.{};
    }
};

/// Request-owned FAKE compiler: reads kind/text/after, returns a separately allocated Rules owner; outlives everything.
const Compiler = struct {
    lease: *Lease, width: usize, backend: *Backend, driver: *Driver,
    live: bool = true, compiled: usize = 0, outstanding: usize = 0, closes: usize = 0,
    close_after_release: bool = false, early_close_refusals: usize = 0, violations: usize = 0,
    fn compile(k: *Compiler, r: *const api.Request) anyerror!*lanes.grammar.Rules {
        const s = r.structure.?;
        if (!k.live or !k.lease.intact() or r != &k.lease.request or s.kind != .json_schema or !std.mem.eql(u8, s.text, text) or s.compiler != null) k.violations += 1;
        const o = try a.create(Owned);
        o.* = .{ .k = k };
        o.rules = .{ .ptr = o, .width = k.width, .after = s.after, .active = s.after == null, .vtable = &Owned.vtable };
        k.compiled += 1; k.outstanding += 1;
        return &o.rules;
    }
    fn close(k: *Compiler) !void {
        if (k.outstanding != 0 or k.lease.prompt.len != 0) { k.early_close_refusals += 1; return error.OutstandingRequest; }
        k.live = false;
    }
};
const Owned = struct {
    k: *Compiler, rules: lanes.grammar.Rules = undefined, accepted: usize = 0,
    const vtable: lanes.grammar.Rules.VTable = .{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close };
    fn from(p: *anyopaque) *Owned { return @ptrCast(@alignCast(p)); }
    fn live(o: *Owned) void { if (!o.k.live or !o.k.lease.intact()) o.k.violations += 1; }
    fn fill(p: *anyopaque, words: []u32) anyerror!void {
        const o = from(p); o.live();
        @memset(words, 0);
        if (o.accepted < 2) { const t: u32 = 7 + @as(u32, @intCast(o.accepted)); words[t / 32] |= @as(u32, 1) << @as(u5, @intCast(t % 32)); }
    }
    fn accept(p: *anyopaque, token: u32) anyerror!bool {
        const o = from(p); o.live();
        if (o.accepted >= 2 or token != 7 + @as(u32, @intCast(o.accepted))) return false;
        o.accepted += 1; return true;
    }
    fn rollback(_: *anyopaque, _: usize) anyerror!void { return error.UnusedRollback; }
    fn terminated(p: *anyopaque) bool { return from(p).accepted >= 2; }
    fn close(p: *anyopaque) void {
        const o = from(p); const k = o.k; o.live();
        k.closes += 1; k.outstanding -= 1;
        k.close_after_release = k.backend.owner == null and k.driver.thread == null and k.driver.releases == 1;
        a.destroy(o);
    }
};

/// The structured request's sink and transition observer; acknowledgement is set only as the finished callback returns.
const Ctx = struct {
    lease: *Lease, compiler: *Compiler,
    seen: [32]seam.Transition = undefined, n_seen: usize = 0,
    transferred: std.atomic.Value(bool) = .init(false),
    tokens: [8]u32 = undefined, n: usize = 0,
    reason: ?api.Reason = null, message: [64]u8 = undefined, message_len: usize = 0,
    finished_seen: bool = false, violations: usize = 0,
    ack: std.atomic.Value(bool) = .init(false),
    fn from(p: *anyopaque) *Ctx { return @ptrCast(@alignCast(p)); }
    fn compile(p: *anyopaque, r: *const api.Request) anyerror!*lanes.grammar.Rules { return from(p).compiler.compile(r); }
    fn observe(p: *anyopaque, _: api.Id, t: seam.Transition) void {
        const x = from(p);
        if (x.n_seen < x.seen.len) { x.seen[x.n_seen] = t; x.n_seen += 1; }
        if (!x.lease.intact() or (x.lease.returned.load(.acquire) and t != .destroyed)) x.violations += 1;
        if (t == .after_transfer) x.transferred.store(true, .release);
        if (t == .after_event_return and x.finished_seen) { x.lease.returned.store(true, .release); x.ack.store(true, .release); }
    }
    fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x = from(p);
        if (!x.lease.intact() or x.finished_seen) x.violations += 1;
        switch (ev.*) {
            .tokens => |t| { const k = @min(t.len, x.tokens.len - x.n); @memcpy(x.tokens[x.n..][0..k], t[0..k]); x.n += k; },
            .finished => |d| {
                x.reason = d.reason; x.message_len = @min(d.message.len, x.message.len);
                @memcpy(x.message[0..x.message_len], d.message[0..x.message_len]);
                x.finished_seen = true; x.lease.finished = true;
            },
            else => {},
        }
    }
    fn wait(x: *Ctx) !void { return waitFor(&x.ack); }
    fn saw(x: *const Ctx, t: seam.Transition) bool { return std.mem.indexOfScalar(seam.Transition, x.seen[0..x.n_seen], t) != null; }
};
/// An ordinary request on the same LaneHost and backend afterwards.
const Plain = struct {
    tokens: [8]u32 = undefined, n: usize = 0, reason: ?api.Reason = null, message: [64]u8 = undefined, message_len: usize = 0,
    ack: std.atomic.Value(bool) = .init(false),
    fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const x: *Plain = @ptrCast(@alignCast(p));
        switch (ev.*) {
            .tokens => |t| { const k = @min(t.len, x.tokens.len - x.n); @memcpy(x.tokens[x.n..][0..k], t[0..k]); x.n += k; },
            .finished => |d| {
                x.reason = d.reason; x.message_len = @min(d.message.len, x.message.len);
                @memcpy(x.message[0..x.message_len], d.message[0..x.message_len]); x.ack.store(true, .release);
            },
            else => {},
        }
    }
};
fn waitFor(flag: *std.atomic.Value(bool)) !void {
    const deadline = now() + 5 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (now() > deadline) return error.HostFixtureTimeout;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
}

/// Session hooks over the Rendezvous: arms one peer's READY frame and snapshots the Session the moment it fails.
const Hooked = struct {
    sync: *Sync, inner: c.Hooks, fail: *FrameFail, ctx: *Ctx,
    peer: ?u32 = null, kind: c.Kind = .reset,
    local_error: ?anyerror = null, status_at_failure: ?c.Status = null, transferred_at_failure: bool = false, fail_ns: i96 = 0,
    follower_error: ?anyerror = null, follower_ns: i96 = 0, after_finished: usize = 0,
    fn from(p: *anyopaque) *Hooked { return @ptrCast(@alignCast(p)); }
    fn install(h: *Hooked) void {
        h.sync.session.hooks = .{ .ctx = h, .open = open, .before = before, .after = after, .quiesce = quiesce, .close = close };
    }
    fn late(h: *Hooked) void { if (h.ctx.lease.returned.load(.acquire)) h.after_finished += 1; }
    fn open(p: *anyopaque, peer: u32) anyerror!void { const h = from(p); h.late(); return h.inner.open(h.inner.ctx, peer); }
    fn before(p: *anyopaque, d: c.Descriptor) anyerror!c.Permit {
        const h = from(p); h.late();
        const target = h.peer != null and h.peer.? == d.peer and d.kind == h.kind and h.fail.fired.load(.acquire) == 0;
        if (target) h.fail.armed.store(me(), .release);
        defer if (target) h.fail.armed.store(0, .release);
        return h.inner.before(h.inner.ctx, d) catch |err| {
            h.sync.mutex.lockUncancelable(io); defer h.sync.mutex.unlock(io);
            if (target and h.fail.fired.load(.acquire) == 1) {
                h.local_error = err; h.status_at_failure = h.sync.session.status; h.fail_ns = now();
                h.transferred_at_failure = h.ctx.transferred.load(.acquire);
            } else if (h.follower_error == null) { h.follower_error = err; h.follower_ns = now(); }
            return err;
        };
    }
    fn after(p: *anyopaque, permit: c.Permit, done: c.Completion) anyerror!c.Completion { const h = from(p); h.late(); return h.inner.after(h.inner.ctx, permit, done); }
    fn quiesce(p: *anyopaque, peer: u32, why: c.Status) anyerror!c.Status { const h = from(p); h.late(); return h.inner.quiesce(h.inner.ctx, peer, why); }
    fn close(p: *anyopaque, peer: u32) void { const h = from(p); h.late(); h.inner.close(h.inner.ctx, peer); }
};

pub fn run(comptime Host: type, comptime serve: anytype) !void {
    const Case = enum { borrowed, rank0_reset_frame, rank1_reset_frame, rank0_step_frame };
    var red: usize = 0;
    for (std.meta.tags(Case)) |case| {
        var lease = try Lease.init(); defer lease.deinit();
        const r0 = try glm.grammar_tests_rec.tp(a, 0); defer r0.deinit(a);
        const r1 = try glm.grammar_tests_rec.tp(a, 1); defer r1.deinit(a);
        var f0 = r0.recording(); var f1 = r1.recording();
        var v0 = try data.Data.init(r0.m.vocab, 2); defer v0.deinit(); v0.set(); v0.value(91, 200);
        var v1 = try data.Data.init(r1.m.vocab, 2); defer v1.deinit(); v1.set(); v1.value(91, 200);
        r0.e.smp.masked_cpu = &v0.cpu; r1.e.smp.masked_cpu = &v1.cpu;
        r0.t.picker = v0.picker(); r1.t.picker = v1.picker();
        var s0 = glm.serve.Serve{ .e = r0.e, .f = &f0, .absorb = true };
        var s1 = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
        var fail = FrameFail{};
        var sync = try Sync.init(fail.allocator(), 1, 2, 71); defer sync.deinit(); sync.install();
        var driver = Driver{ .serve = &s1, .session = &sync.session };
        var world = glm.ranks.Pair{ .gpa = a, .io = io, .rank = 0 };
        var armed = Armed{ .steps = &s0, .pair = &world };
        var backend = Backend{ .steps = &armed, .capacity = 2560, .reach = 2560, .frame_allocator = a, .structured_session = &sync.session,
            .mirror = .{ .ctx = &driver, .send = Driver.legacy, .structured_driver_cpu = .{ .open = Driver.open, .operation = Driver.operation, .terminate = Driver.terminate } } };
        var compiler = Compiler{ .lease = &lease, .width = r0.m.vocab, .backend = &backend, .driver = &driver };
        var ctx = Ctx{ .lease = &lease, .compiler = &compiler };
        var hooked = Hooked{ .sync = &sync, .inner = sync.session.hooks.?, .fail = &fail, .ctx = &ctx,
            .peer = switch (case) { .borrowed => null, .rank0_reset_frame, .rank0_step_frame => 0, .rank1_reset_frame => 1 },
            .kind = if (case == .rank0_step_frame) .step else .reset };
        hooked.install();
        var h = Host{ .gpa = a, .gpu = null, .family = &backend, .release = struct { fn noop(_: *anyopaque) void {} }.noop,
            .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined,
            .test_session = &sync.session };
        // The actual GLM rank-0 constructor: backend facts, one row, one lane, the host thread started.
        try serve(&h, io, backend.facts(), 1, .{ .lanes = 1 }, null);
        var joined = false;
        defer if (!joined) { h.host.stop(); h.core.deinit(); h.cfg.deinit(a); };
        h.host.test_attachment = .{ .ctx = &ctx, .compile = Ctx.compile, .observe = Ctx.observe };
        const engine = h.host.engine();
        const t0 = now();
        try engine.submit(1, &lease.request, .{ .ctx = &ctx, .event = Ctx.event });
        try std.testing.expectError(error.OutstandingRequest, lease.release());
        try ctx.wait();
        const finish_ms = @divTrunc(now() - t0, std.time.ns_per_ms);
        // Finished and returned, but the host and peers have not been joined yet: still borrowed.
        try std.testing.expectError(error.OutstandingRequest, lease.release());
        try std.testing.expectError(error.OutstandingRequest, compiler.close());
        const failed = case != .borrowed;
        if (failed) {
            try check("PeerFrameTargetRuntimeViolation", fail.fired.load(.acquire) == 1 and hooked.local_error != null and hooked.transferred_at_failure);
            // Fail-stop at the failing READY itself: cleanup or the follower's two-second deadline cannot stand in for it.
            const immediate = hooked.status_at_failure != null and hooked.status_at_failure.? == .poison;
            // Measured from the failing allocation itself; a follower that never reached READY did not wait.
            const prompt = (hooked.follower_error == null or hooked.follower_ns - fail.fired_ns < std.time.ns_per_s) and
                (driver.outcome == null or driver.outcome.? != error.PeerWaitTimeout);
            if (!immediate) std.debug.print("ImmediatePeerFramePoisonRuntimeViolation case={s} status_at_failure={s} local_error={s}\n", .{
                @tagName(case), if (hooked.status_at_failure) |s| @tagName(s) else "none", if (hooked.local_error) |e| @errorName(e) else "none" });
            if (!prompt) std.debug.print("FollowerLeftWaitingRuntimeViolation case={s} follower_error={s} follower_wait_ms={d}\n", .{
                @tagName(case), if (hooked.follower_error) |e| @errorName(e) else "none", @divTrunc(hooked.follower_ns - fail.fired_ns, std.time.ns_per_ms) });
            red += @as(usize, @intFromBool(!immediate)) + @intFromBool(!prompt);
        }
        var plain_request = api.Request{ .prompt = &.{ 3, 4 }, .max_tokens = 1, .drafts = false };
        var plain = Plain{};
        try engine.submit(2, &plain_request, .{ .ctx = &plain, .event = Plain.event });
        try waitFor(&plain.ack);
        h.host.stop(); joined = true;
        const host_thread_gone = h.host.thread == null and h.host.closing;
        const core_empty = h.core.live.items.len == 0;
        h.core.deinit(); h.cfg.deinit(a);
        lease.host_joined = host_thread_gone and core_empty;
        lease.peers_joined = driver.thread == null and driver.releases == 1 and driver.opens == 1 and sync.session.phase == .closed;
        try lease.release();
        try compiler.close();
        try check("RequestLeaseRuntimeViolation", lease.refusals == 2 and compiler.early_close_refusals == 1 and lease.prompt.len == 0 and !compiler.live);
        try check("BorrowedSliceRuntimeViolation", ctx.violations == 0 and compiler.violations == 0 and hooked.after_finished == 0);
        try check("CompiledOwnerTransferRuntimeViolation", compiler.compiled == 1 and compiler.closes == 1 and compiler.outstanding == 0 and compiler.close_after_release);
        try check("TransitionOrderRuntimeViolation", ctx.saw(.dequeued) and ctx.saw(.compiled) and ctx.saw(.before_transfer) and ctx.saw(.after_transfer) and
            ctx.saw(.before_backend) and ctx.saw(.before_finished) and ctx.saw(.after_event_return) and ctx.saw(.destroyed));
        std.debug.print("STATE grammar_tests_c7 TP case={s} owner_null={} session_null={} s0_open={} s1_open={} armed_len={d} s0_len={d} s1_len={d} poisoned={}\n", .{
            @tagName(case), backend.owner == null, backend.structured_session == null, s0.open, s1.open, armed.len, s0.len, s1.len, backend.structured_poisoned });
        // The ordinary request ran after the structured release: only its own two-token prefill may remain; a refused one leaves none.
        const plain_len: usize = if (case == .borrowed) plain_request.prompt.len else 0;
        try check("BackendReleaseOrderRuntimeViolation", backend.owner == null and backend.structured_session == null and !s0.open and !s1.open and
            armed.len == plain_len and s0.len == plain_len and s1.len == plain_len);
        try check("DeferredLogitLeaseRuntimeViolation", r0.e.structured_control == null and r1.e.structured_control == null and !r0.e.structured_logits_valid and !r1.e.structured_logits_valid);
        try check("PeerJoinRuntimeViolation", sync.close_calls[0] == 1 and sync.close_calls[1] == 1 and sync.session.cancel_ctx == null and sync.session.cancel_check == null);
        const message = ctx.message[0..ctx.message_len];
        if (failed) {
            try check("FailedRequestRuntimeViolation", ctx.reason != null and ctx.reason.? == .failed and sync.session.status == .poison and backend.structured_poisoned);
            try check("FollowerOutcomeRuntimeViolation", driver.outcome == null or driver.outcome.? != error.PeerWaitTimeout);
            const delivered: []const u32 = if (case == .rank0_step_frame) &.{91} else &.{};
            try std.testing.expectEqualSlices(u32, delivered, ctx.tokens[0..ctx.n]);
            // Poison is fail-stop: the same backend refuses the next ordinary request before any rank-1 work.
            try check("PoisonedBackendRefusalRuntimeViolation", plain.reason != null and plain.reason.? == .failed and plain.n == 0 and
                std.mem.eql(u8, plain.message[0..plain.message_len], "ControlPoison") and driver.legacy_calls == 0);
        } else {
            try check("BorrowedCompletionRuntimeViolation", ctx.reason != null and ctx.reason.? == .stop and sync.session.status == .ok and
                !backend.structured_poisoned and driver.outcome == null and fail.fired.load(.acquire) == 0 and hooked.follower_error == null);
            try std.testing.expectEqualSlices(u32, &.{ 91, 18, 19, 7, 8 }, ctx.tokens[0..ctx.n]);
            try check("SameHostOrdinaryRuntimeViolation", plain.reason != null and plain.reason.? == .length and plain.n == 1 and plain.tokens[0] == 91 and driver.legacy_calls == 2);
        }
        std.debug.print("CELL grammar_tests_c7 TP case={s} actual_Host_serve=PASS reason={s} message={s} tokens={d} frame_fired={d} status_at_failure={s} follower_wait_ms={d} finish_ms={d} peers_joined=PASS lease_refusals={d} early_compiler_close=REFUSED close_after_release=PASS plain_next={s} compiler=FAKE peers=RECORDING_CPU\n", .{
            @tagName(case), @tagName(ctx.reason.?), message, ctx.n, fail.fired.load(.acquire),
            if (hooked.status_at_failure) |s| @tagName(s) else "none",
            if (failed and hooked.follower_error != null) @divTrunc(hooked.follower_ns - fail.fired_ns, std.time.ns_per_ms) else 0, finish_ms, lease.refusals,
            if (failed) "REFUSED" else "PASS" });
    }
    if (red != 0) return error.ImmediatePeerFramePoisonRuntimeViolation;
}
