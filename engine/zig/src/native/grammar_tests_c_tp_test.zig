//! Actual TP constructor and two independent Engine/Fwd workers, CPU only.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_cuda_lanes.zig");
const pair = @import("glm_cuda_pair.zig");
const data = @import("grammar_tests_data");
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const c = lanes.structured_control;
const a = std.testing.allocator;
const io = std.testing.io;

pub const Driver = struct {
    serve: *glm.serve.Serve,
    session: *c.Session,
    prompt: []u32 = &.{},
    thread: ?std.Thread = null,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    stopping: bool = false,
    outcome: ?anyerror = null,
    position: usize = 0,
    opens: usize = 0,
    releases: usize = 0,
    legacy_calls: usize = 0,
    command: ?c.Kind = null,
    input: []u32 = &.{},
    mask: []u32 = &.{},
    fn from(p: ?*anyopaque) *Driver { return @ptrCast(@alignCast(p.?)); }
    pub fn open(p: ?*anyopaque, ids: []const u32, endpoint: c.Endpoint) !void {
        const d = from(p);
        if (d.thread != null or endpoint.session != d.session) return error.PeerAgreementRuntimeViolation;
        d.prompt = try a.dupe(u32, ids);
        errdefer { a.free(d.prompt); d.prompt = &.{}; }
        d.thread = try std.Thread.spawn(.{}, worker, .{d});
        d.opens += 1;
    }
    pub fn operation(p: ?*anyopaque, kind: c.Kind, _: u64, tokens: []const u32, mask: []const u32) !void {
        const d = from(p);
        if (kind != .first and kind != .step) return error.ControlShape;
        const input = try a.dupe(u32, tokens); errdefer a.free(input);
        const owned_mask = try a.dupe(u32, mask); errdefer a.free(owned_mask);
        d.mutex.lockUncancelable(io); defer d.mutex.unlock(io);
        const timeout = deadline();
        while (d.command != null) try d.wake.waitTimeout(io, &d.mutex, timeout);
        if (d.stopping or d.thread == null) return error.SessionClosed;
        d.input = input; d.mask = owned_mask; d.command = kind;
        d.wake.broadcast(io);
    }
    fn deadline() std.Io.Timeout {
        return .{ .deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(2), .clock = .awake }) };
    }
    fn worker(d: *Driver) void {
        d.serve.installStructured(d.session.endpoint(1) catch unreachable) catch |err| { d.outcome = err; return; };
        d.serve.prefillToLogits(d.prompt) catch |err| { d.outcome = err; };
        d.position = d.serve.e.st.pos;
        while (true) {
            d.mutex.lockUncancelable(io);
            const timeout = deadline();
            while (!d.stopping and d.command == null) d.wake.waitTimeout(io, &d.mutex, timeout) catch {
                d.outcome = error.PeerWaitTimeout;
                d.stopping = true;
            };
            if (d.stopping and d.command == null) {
                d.mutex.unlock(io);
                break;
            }
            const kind = d.command.?;
            d.mutex.unlock(io);
            switch (kind) {
                .first => { if (d.serve.pickFirst(d.mask)) |_| {} else |err| { d.outcome = err; } },
                .step => { if (d.serve.stepMasked(d.input[0], d.mask)) |_| {} else |err| { d.outcome = err; } },
                else => unreachable,
            }
            d.mutex.lockUncancelable(io);
            a.free(d.input); a.free(d.mask); d.input = &.{}; d.mask = &.{}; d.command = null;
            d.wake.broadcast(io); d.mutex.unlock(io);
        }
        (d.session.endpoint(1) catch unreachable).close();
        d.serve.clearStructured();
        d.serve.release();
    }
    pub fn terminate(p: ?*anyopaque, _: c.Status) void {
        const d = from(p);
        if (d.thread) |thread| {
            d.mutex.lockUncancelable(io); d.stopping = true; d.wake.broadcast(io); d.mutex.unlock(io);
            (d.session.endpoint(0) catch unreachable).close();
            thread.join(); d.thread = null;
            a.free(d.prompt); d.prompt = &.{};
            d.releases += 1;
        }
    }
    pub fn legacy(p: ?*anyopaque, op: gl.Op, words: []const u32) !void {
        const d = from(p); d.legacy_calls += 1;
        try gl.replay(glm.serve.Serve, d.serve, op, words);
    }
};

fn witness(name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.NoWorkAfterAgreedCancelRuntimeViolation; }
}
pub fn run(comptime Host: type, comptime wrap: fn (*Host) lanes.backend.Backend) !void {
    const Armed = pair.Armed(glm.serve.Serve);
    const Backend = gl.Lanes(Armed);
    if (comptime !@hasField(Backend, "structured_session") or !@hasDecl(glm.serve.Serve, "installStructured")) {
        std.debug.print("MissingWrapperRuntimeViolation: actual TP coordinated constructor seam is absent\n", .{});
        return error.MissingWrapperRuntimeViolation;
    } else {
        inline for ([_]usize{129, 132, 192}) |length| {
            inline for ([_]?u32{null, 0, 1, 2}) |cancel_chunk| {
                const r0 = try glm.grammar_tests_rec.tp(a, 0); defer r0.deinit(a);
                const r1 = try glm.grammar_tests_rec.tp(a, 1); defer r1.deinit(a);
                var f0 = r0.recording(); var f1 = r1.recording();
                var values0 = try data.Data.init(r0.m.vocab, 2); defer values0.deinit(); values0.set();
                var values1 = try data.Data.init(r1.m.vocab, 2); defer values1.deinit(); values1.set();
                r0.e.smp.masked_cpu = &values0.cpu; r1.e.smp.masked_cpu = &values1.cpu;
                r0.t.picker = values0.picker(); r1.t.picker = values1.picker();
                var serve0 = glm.serve.Serve{ .e = r0.e, .f = &f0, .absorb = true };
                var serve1 = glm.serve.Serve{ .e = r1.e, .f = &f1, .absorb = true };
                var sync = try Sync.init(a, 7, 2, 41); defer sync.deinit(); sync.install();
                sync.cancel_kind = if (cancel_chunk != null) .prompt_chunk else null; sync.cancel_chunk = cancel_chunk;
                var driver = Driver{ .serve = &serve1, .session = &sync.session };
                var world = glm.ranks.Pair{ .gpa = a, .io = io, .rank = 0 };
                var armed = Armed{ .steps = &serve0, .pair = &world };
                var backend = Backend{ .steps = &armed, .capacity = 2560, .reach = 2560, .frame_allocator = a,
                    .structured_session = &sync.session,
                    .mirror = .{ .ctx = &driver, .send = Driver.legacy,
                        .structured_driver_cpu = .{ .open = Driver.open, .operation = Driver.operation, .terminate = Driver.terminate } } };
                var h = Host{ .gpa = a, .gpu = null, .family = &backend, .release = struct { fn noop(_: *anyopaque) void {} }.noop,
                    .inner = backend.backend(), .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined };
                const actual = wrap(&h);
                var rules = data.Fake{}; rules.init(r0.m.vocab);
                const prompt = try a.alloc(u32, length); defer a.free(prompt); @memset(prompt, 3);
                var stream = try lanes.Stream.init(a, .{ .id = "grammar_tests_c-tp-chunks", .prompt = prompt, .max_new = 2, .rules = &rules.rules });
                defer stream.deinit(a);
                var error0: ?anyerror = null;
                actual.prefill(&stream) catch |err| { error0 = err; };
                if (error0 != null) {
                    var cleanup_cfg = try lanes.Config.init(a, .{}, 1, 8); defer cleanup_cfg.deinit(a);
                    var cleanup_clock = lanes.fake.FixedClock{};
                    var cleanup_core = lanes.Engine.init(a, &cleanup_cfg, actual, cleanup_clock.clock()); defer cleanup_core.deinit();
                    lanes.structured.release(&cleanup_core, &stream);
                }
                if (cancel_chunk == null) {
                    try witness("EpochSequenceRuntimeViolation", error0 == null);
                    try std.testing.expectEqual(length, r0.e.st.pos);
                    try std.testing.expect(serve0.deferred and backend.first_pending);
                    actual.release(&stream);
                } else {
                    try witness("PeerAgreementRuntimeViolation", error0 != null);
                    try std.testing.expectEqualStrings("Cancelled", @errorName(error0.?));
                }
                const chunks: usize = cancel_chunk orelse 3;
                try witness("NoWorkAfterAgreedCancelRuntimeViolation", data.count(&r0.t, "staged.record") == chunks);
                try witness("NoWorkAfterAgreedCancelRuntimeViolation", data.count(&r1.t, "staged.record") == chunks);
                try witness("PeerAgreementRuntimeViolation", sync.before_calls[0] == sync.before_calls[1] and sync.after_calls[0] == sync.after_calls[1]);
                if (cancel_chunk != null) {
                    try std.testing.expect(driver.outcome != null);
                    try std.testing.expectEqualStrings("Cancelled", @errorName(driver.outcome.?));
                } else try std.testing.expect(driver.outcome == null);
                try std.testing.expectEqual(chunks * 64, if (cancel_chunk != null) driver.position else chunks * 64);
                try std.testing.expectEqual(@as(usize, 0), driver.legacy_calls);
                try std.testing.expectEqual(@as(usize, 1), driver.opens);
                try std.testing.expectEqual(@as(usize, 1), driver.releases);
                try std.testing.expectEqual(c.Phase.closed, sync.session.phase);
                try witness("DeferredLogitLeaseRuntimeViolation", !r0.e.structured_logits_valid and !r1.e.structured_logits_valid);
                try std.testing.expectEqual(@as(usize, 0), r0.e.st.pos); try std.testing.expectEqual(@as(usize, 0), r1.e.st.pos);
                try std.testing.expect(backend.owner == null and !serve0.open and !serve1.open and armed.len == 0);
                try std.testing.expect(backend.structured_session == null and r0.e.structured_control == null and r1.e.structured_control == null);
                try std.testing.expectEqual(@as(usize, 0), data.count(&r0.t, "pick") + data.count(&r1.t, "pick"));
                try std.testing.expectEqual(@as(usize, 0), values0.cpu.picks + values1.cpu.picks);
                try std.testing.expectEqual(@as(usize, 0), stream.emitted().len);
                // Genuine ordinary constructor reuse, same Engines/Serve/Host/legacy driver, without the closed Session.
                var plain = try lanes.Stream.init(a, .{ .id = "grammar_tests_c-plain-next", .prompt = &.{3, 4}, .max_new = 1 }); defer plain.deinit(a);
                try actual.prefill(&plain);
                try std.testing.expectEqual(@as(u32, 2), try actual.read(try actual.first(&plain, 2)));
                actual.release(&plain);
                try std.testing.expectEqual(@as(usize, 2), driver.legacy_calls);
                try std.testing.expect(serve0.absorb and serve1.absorb);
                std.debug.print("CELL TP prompt={d} cancel_chunk={?d} stages={d}/{d} plain_next=PASS\n", .{length, cancel_chunk, chunks, chunks});
            }
        }
    }
}
