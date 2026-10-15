//! GPT-6.1 Sol via Hermes: actual Bound/Serve/Attn CPU fixture; expert sidecars are MOCK.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const data = @import("grammar_tests_data");
const gl = @import("glm_lanes");
const Sync = @import("grammar_tests_c_sync").Rendezvous;
const c = lanes.structured_control;
const a = std.testing.allocator;
const io = std.testing.io;
const OwnerCancel = struct {
    trace: *glm.trace.Trace,
    fn check(ptr: *anyopaque) bool {
        const owner: *OwnerCancel = @ptrCast(@alignCast(ptr));
        return data.count(owner.trace, "staged.record") >= 1;
    }
};

pub const Trio = struct {
    sync: *Sync,
    inner: c.Hooks,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    workers: [2]Worker = undefined,
    threads: [2]?std.Thread = @splat(null),
    stopping: bool = false,
    serial: u32 = 0,
    refused: [3]u32 = @splat(0),
    refused_departed: [3]u32 = @splat(0),
    mismatch_take: bool = false,
    descriptor: ?c.Descriptor = null,
    tokens: [256]u32 = undefined,
    mask: []u32,
    releases: usize = 0,
    commands: usize = 0,
    trace: ?*glm.trace.Trace = null,
    cutoff: ?usize = null,
    action: c.Action = .none,
    action_kind: c.Kind = .reset,
    issued_before: usize = 0,
    taken_before: usize = 0,
    const Worker = struct { trio: *Trio, peer: u32, outcome: ?anyerror = null, calls: usize = 0 };
    fn from(p: ?*anyopaque) *Trio { return @ptrCast(@alignCast(p.?)); }
    pub fn install(t: *Trio) void {
        t.sync.session.hooks = .{ .ctx = t, .open = openEndpoint, .before = before, .after = after, .quiesce = quiesce, .close = close };
    }
    fn openEndpoint(p: *anyopaque, peer: u32) !void { const t = from(p); try t.inner.open(t.inner.ctx, peer); }
    fn before(p: *anyopaque, descriptor: c.Descriptor) !c.Permit {
        const t = from(p);
        if (descriptor.peer == 0) {
            if (t.trace) |trace| {
                try trace.flush(); // Include buffered Triton launches at this real scheduler boundary.
                t.action = descriptor.action; t.action_kind = descriptor.kind;
                t.issued_before = data.count(trace, "afd.exchange");
                t.taken_before = data.count(trace, "afd.collect");
            }
            if (descriptor.tokens.len > t.tokens.len or descriptor.mask.len > t.mask.len) return error.ControlShape;
            t.mutex.lockUncancelable(io);
            @memcpy(t.tokens[0..descriptor.tokens.len], descriptor.tokens);
            @memcpy(t.mask[0..descriptor.mask.len], descriptor.mask);
            var d = descriptor;
            d.tokens = t.tokens[0..descriptor.tokens.len]; d.mask = t.mask[0..descriptor.mask.len];
            t.descriptor = d; t.serial += 1; t.wake.broadcast(io); t.mutex.unlock(io);
        }
        var agreed = descriptor;
        if (t.mismatch_take and descriptor.peer == 1 and descriptor.action == .take) agreed.transport_seq +%= 1;
        return t.inner.before(t.inner.ctx, agreed) catch |err| {
            if (err == error.ControlPoison) return err;
            if (descriptor.peer == 0 and t.cutoff == null) if (t.trace) |trace| {
                t.cutoff = trace.events.items.len;
                std.debug.print(" CUTOFF_BEFORE {t}/{t} {s} events={d}\n", .{descriptor.kind,descriptor.action,@errorName(err),trace.events.items.len});
            };
            t.mutex.lockUncancelable(io); defer t.mutex.unlock(io);
            const serial = t.serial; const timeout = deadline();
            t.refused[descriptor.peer] = serial; t.wake.broadcast(io);
            while (t.refused[0] != serial or t.refused[1] != serial or t.refused[2] != serial)
                t.wake.waitTimeout(io, &t.mutex, timeout) catch return error.PeerWaitTimeout;
            t.refused_departed[descriptor.peer] = serial; t.wake.broadcast(io);
            if (descriptor.peer == 0) while (t.refused_departed[1] != serial or t.refused_departed[2] != serial)
                t.wake.waitTimeout(io, &t.mutex, timeout) catch return error.PeerWaitTimeout;
            return err;
        };
    }
    fn after(p: *anyopaque, permit: c.Permit, completion: c.Completion) !c.Completion {
        const t = from(p);
        if (permit.peer == 0) if (t.trace) |trace| {
            try trace.flush(); // Every permitted operation lands before any after-cutoff snapshot.
        };
        if (permit.peer == 0 and t.action_kind == .group_action) if (t.trace) |trace| {
            const issued = data.count(trace, "afd.exchange") - t.issued_before;
            const taken = data.count(trace, "afd.collect") - t.taken_before;
            if (issued != completion.issued or taken != completion.taken) {
                _ = t.inner.quiesce(t.inner.ctx, permit.peer, .poison) catch {};
                std.debug.print("ExpertPermitActualTraceRuntimeViolation action={t} actual={d}/{d} reported={d}/{d}\n", .{t.action,issued,taken,completion.issued,completion.taken});
                return error.ExpertPermitActualTraceRuntimeViolation;
            }
        };
        const result = try t.inner.after(t.inner.ctx, permit, completion);
        if (permit.peer == 0 and result.status != .ok and t.cutoff == null) if (t.trace) |trace| {
            t.cutoff = trace.events.items.len;
            std.debug.print(" CUTOFF_AFTER {t}/{t} {t} events={d}\n", .{t.action_kind,t.action,result.status,trace.events.items.len});
        };
        return result;
    }
    pub fn noWorkAfterCutoff(t: *const Trio) !void {
        if (t.cutoff) |offset| if (t.trace) |trace| {
            try trace.flush();
            for (trace.events.items[offset..]) |event| if (!std.mem.eql(u8, event.name, "afd.collect")) {
                std.debug.print("ExpertPermitPostCancelWorkRuntimeViolation: {s} cutoff={d} last_action={t}/{t}\n", .{event.name,offset,t.action_kind,t.action});
                for (trace.events.items[offset..]) |ev| std.debug.print(" POST_CANCEL {s}\n", .{ev.name});
                return error.ExpertPermitPostCancelWorkRuntimeViolation;
            };
        };
    }
    fn quiesce(p: *anyopaque, peer: u32, why: c.Status) !c.Status { const t = from(p); return t.inner.quiesce(t.inner.ctx, peer, why); }
    fn close(p: *anyopaque, peer: u32) void { const t = from(p); t.inner.close(t.inner.ctx, peer); }
    fn deadline() std.Io.Timeout { return .{ .deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(2), .clock = .awake }) }; }
    fn worker(w: *Worker) void {
        const t = w.trio;
        const endpoint = t.sync.session.endpoint(w.peer) catch unreachable;
        endpoint.open() catch |err| { w.outcome = err; return; };
        var seen: u32 = 0;
        while (true) {
            t.mutex.lockUncancelable(io);
            const timeout = deadline();
            while (!t.stopping and seen == t.serial) t.wake.waitTimeout(io, &t.mutex, timeout) catch {
                w.outcome = error.PeerWaitTimeout; t.stopping = true;
            };
            if (t.stopping and seen == t.serial) { t.mutex.unlock(io); break; }
            var d = t.descriptor.?; d.peer = w.peer; seen = t.serial; t.mutex.unlock(io);
            const permit = endpoint.before(d) catch |err| {
                if (err == error.IssueGateClosed) continue;
                w.outcome = err; break;
            };
            w.calls += 1;
            const done = c.Completion{ .logits_valid = true, .issued = if (d.action == .request) 1 else 0, .taken = if (d.action == .take or d.action == .drain) 1 else 0 };
            const result = endpoint.after(permit, done) catch |err| { w.outcome = err; break; };
            if (result.status != .ok) break;
        }
        t.mutex.lockUncancelable(io);
        const timeout = deadline();
        while (!t.stopping) t.wake.waitTimeout(io, &t.mutex, timeout) catch { w.outcome = error.PeerWaitTimeout; break; };
        t.mutex.unlock(io);
        endpoint.close();
    }
    pub fn driverOpen(p: ?*anyopaque, _: []const u32, endpoint: c.Endpoint) !void {
        const t = from(p); if (endpoint.session != &t.sync.session) return error.PeerAgreementRuntimeViolation;
        for (&t.workers, 0..) |*w, i| {
            w.* = .{ .trio = t, .peer = @intCast(i+1) };
            t.threads[i] = std.Thread.spawn(.{}, worker, .{w}) catch |err| {
                _ = endpoint.quiesce(.poison) catch {};
                t.mutex.lockUncancelable(io); t.stopping = true; t.wake.broadcast(io); t.mutex.unlock(io);
                for (&t.threads) |*thread| if (thread.*) |th| { th.join(); thread.* = null; };
                return err;
            };
        }
    }
    pub fn operation(p: ?*anyopaque, _: c.Kind, _: u64, _: []const u32, _: []const u32) !void { from(p).commands += 1; }
    pub fn terminate(p: ?*anyopaque, _: c.Status) void {
        const t = from(p);
        t.mutex.lockUncancelable(io); t.stopping = true; t.wake.broadcast(io); t.mutex.unlock(io);
        (t.sync.session.endpoint(0) catch unreachable).close();
        for (&t.threads) |*thread| if (thread.*) |th| { th.join(); thread.* = null; };
        t.releases += 1;
    }
};
fn witness(name: []const u8, good: bool) !void { if (!good) { std.debug.print("{s}\n", .{name}); return error.AfdChunkAgreementRuntimeViolation; } }
pub fn run(comptime Bound: type) !void {
    if (comptime !@hasDecl(Bound, "installStructured") or !@hasField(glm.afd_engine.Attn, "structured_control")) {
        std.debug.print("MissingAfdWrapperRuntimeViolation: actual Bound/Serve/Attn forwarding absent\n", .{});
        return error.MissingAfdWrapperRuntimeViolation;
    } else {
        const Cases = enum { one, three, chunk0, chunk1, chunk2, finish_cancel, first_cancel, step_cancel, select_cancel, empty, owner_cancel };
        inline for (std.meta.tags(Cases)) |case| {
            var rec: glm.grammar_tests_rec.Rec = undefined; try rec.init(a); defer rec.deinit(a);
            var f = rec.recording();
            var values = try data.Data.init(rec.m.vocab, 1); defer values.deinit(); values.set();
            rec.e.smp.masked_cpu = &values.cpu; rec.t.picker = values.picker(); rec.at.absorb = true;
            var serve = glm.afd_serve.Serve{ .at = &rec.at, .f = &f };
            var bound = Bound{ .s = &serve };
            var sync = try Sync.init(a, 1, 3, 0); defer sync.deinit(); sync.install();
            const mask_storage = try a.alloc(u32, (rec.m.vocab + 31) / 32); defer a.free(mask_storage);
            var trio = Trio{ .sync = &sync, .inner = sync.session.hooks.?, .mask = mask_storage }; trio.install();
            if (case == .chunk0 or case == .chunk1 or case == .chunk2) {
                sync.cancel_kind = .prompt_chunk;
                sync.cancel_chunk = if (case == .chunk0) 0 else if (case == .chunk1) 1 else 2;
            }
            if (case == .finish_cancel) { sync.cancel_kind = .group_action; sync.cancel_action = .finish; }
            if (case == .first_cancel) sync.cancel_kind = .first;
            if (case == .step_cancel or case == .select_cancel) sync.cancel_kind = .step;
            if (case == .select_cancel) sync.cancel_action = .select;
            var backend = gl.Lanes(Bound){ .steps = &bound, .capacity = 2560, .reach = 2560, .frame_allocator = a, .structured_session = &sync.session,
                .mirror = .{ .ctx = &trio, .structured_driver_cpu = .{ .open = Trio.driverOpen, .operation = Trio.operation, .terminate = Trio.terminate } } };
            var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
            var clock = lanes.fake.FixedClock{};
            var core = lanes.Engine.init(a, &cfg, backend.backend(), clock.clock()); defer core.deinit();
            var rules = data.Fake{}; rules.init(rec.m.vocab); rules.empty = case == .empty;
            var ids: [129]u32 = @splat(3);
            const prompt = ids[0..if (case == .one) 3 else ids.len];
            var owner_cancel = OwnerCancel{ .trace = &rec.t };
            var s = try lanes.Stream.init(a, .{ .id = @tagName(case), .prompt = prompt, .max_new = 5, .rules = &rules.rules,
                .cancel_check = if (case == .owner_cancel) .{ .ptr = &owner_cancel, .check = OwnerCancel.check } else null }); defer s.deinit(a);
            const before_cancel = case == .chunk0 or case == .chunk1 or case == .chunk2 or case == .finish_cancel or case == .first_cancel or case == .owner_cancel;
            if (before_cancel) {
                var outcome: ?anyerror = null;
                core.addStream(&s) catch |err| { outcome = err; };
                try witness("BorrowedOwnerCancelRuntimeViolation", outcome != null);
                try std.testing.expectEqualStrings("Cancelled", @errorName(outcome.?));
            }
            else if (case == .empty) try std.testing.expectError(error.EmptyMask, core.addStream(&s))
            else {
                try core.addStream(&s);
                if (case == .step_cancel or case == .select_cancel) try std.testing.expectError(error.Cancelled, core.step())
                else while (!s.finished) try core.step();
            }
            if (before_cancel or case == .empty) try std.testing.expectEqual(@as(usize, 0), s.emitted().len)
            else if (case == .step_cancel or case == .select_cancel) try std.testing.expectEqualSlices(u32, &.{7}, s.emitted())
            else try std.testing.expectEqualSlices(u32, &.{7, 8}, s.emitted());
            const chunks: usize = if (case == .one) 1 else if (case == .chunk0) 0 else if (case == .chunk1 or case == .owner_cancel) 1 else if (case == .chunk2) 2 else 3;
            const steps: usize = if (case == .one or case == .three or case == .select_cancel) 1 else 0;
            try witness("NoWorkAfterAgreedCancelRuntimeViolation", data.count(&rec.t, "staged.record") == chunks+steps);
            try witness("AfdChunkAgreementRuntimeViolation", data.count(&rec.t, "afd.exchange") == (chunks+steps)*2);
            try std.testing.expectEqual(data.count(&rec.t, "afd.exchange"), data.count(&rec.t, "afd.collect"));
            try witness("AbsorbRestoreRuntimeViolation", rec.at.absorb and rec.e.st.mtp_len == 0);
            try witness("DeferredLogitLeaseRuntimeViolation", !serve.open and serve.len == 0 and bound.len == 0 and rec.at.deferred_logits == null and rec.at.structured_control == null);
            try std.testing.expect(backend.owner == null and backend.structured_session == null);
            try witness("BorrowedOwnerCancelRuntimeViolation", sync.session.cancel_ctx == null and sync.session.cancel_check == null);
            try std.testing.expectEqual(c.Phase.closed, sync.session.phase);
            for (0..3) |peer| { try std.testing.expectEqual(sync.after_calls[0], sync.after_calls[peer]); try std.testing.expectEqual(@as(u32, 1), sync.close_calls[peer]); }
            try std.testing.expectEqual(@as(usize, 1), trio.releases);
            for (trio.workers) |w| if (w.outcome) |err| try std.testing.expectEqualStrings("Cancelled", @errorName(err));
            var plain = try lanes.Stream.init(a, .{ .id = "plain-after-afd", .prompt = &.{3, 4}, .max_new = 1 }); defer plain.deinit(a);
            try core.addStream(&plain); while (!plain.finished) try core.step();
            try std.testing.expectEqualSlices(u32, &.{2}, plain.emitted());
            try std.testing.expect(rec.e.st.mtp_len > 0);
            std.debug.print("CELL AFD chunk={s} roles=3 real_attn=PASS expert_sidecars=MOCK no_absorb=PASS plain_next=PASS\n", .{@tagName(case)});
        }
    }
}
