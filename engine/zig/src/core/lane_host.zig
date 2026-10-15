//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const builtin = @import("builtin");
const test_host = @import("grammar_tests_c_host_fixture.zig");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    filling: std.ArrayList(*Job) = .empty, // prefill_interleave (#623): admitted jobs whose prompt is still going in, oldest first
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    max_queued: ?u32 = null, // foreground requests past the lanes that may wait, then Busy (Mia's 0082); null: no cap
    test_attachment: if (builtin.is_test) ?test_host.Attachment else void = if (builtin.is_test) null else {},
    /// grammar_server: the role's Rules for a structured request (the AFD attention role's, TF_AFD_GRAMMAR_SERVE=1).
    structured: ?Structured = null,
    /// prefill_interleave: with a backend that fills a step a call, `rounds` rounds of the live streams, then `steps` prompt steps.
    fill_policy: FillPolicy = .{},
    fill_rounds: u32 = std.math.maxInt(u32), // rounds since the last fill turn (a new prompt's first step runs at once)

    pub const FillPolicy = struct { rounds: u8 = 1, steps: u8 = 1 };

    pub const Structured = struct {
        ctx: *anyopaque,
        /// A fresh request-owned Rules from the request's structure (its compiler), closed by the stream's end.
        rules: *const fn (ctx: *anyopaque, request: *const Request) anyerror!*lanes.grammar.Rules,
    };

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        test_cancelled: if (builtin.is_test) bool else void = if (builtin.is_test) false else {},
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        fill_began: bool = false, // prefill_interleave: its first prompt step ran
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| _ = store.keep(job.request.prompt, at, s, job.request.chunks);
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            if (job.stream.reuse_failed) store.resumed(e, job.request.prompt, false) else if (job.stream.cached == e.at) store.resumed(e, job.request.prompt, true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join() else if (builtin.is_test) {
            if (h.test_attachment != null) test_host.drain(h, true, .cancelled, "");
        }
        h.thread = null;
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.filling.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn, .admit = admitFn } };
    }

    fn admitFn(ctx: *anyopaque, background: bool) bool {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        return background or !h.full();
    }

    /// Foreground requests in the system at the cap (the lanes plus `max_queued`), read under the mutex.
    fn full(h: *const LaneHost) bool {
        const cap = h.max_queued orelse return false;
        var n = h.admitted.items.len;
        for (h.queued.items) |job| n += @intFromBool(!job.request.background);
        return n >= h.info_.lanes + cap;
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (!request.background and h.full()) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (builtin.is_test) if (h.structured == null and test_host.markCancelled(h, id)) { h.wake.signal(h.io); return; };
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
        };
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
        if (builtin.is_test) test_host.observe(job.host, job, .after_event_return);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or (builtin.is_test and job.test_cancelled) or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        if (builtin.is_test) if (job.request.structure != null and h.structured == null and cancelled(job)) return;
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, "");
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        if (builtin.is_test) if (job.request.structure != null and h.structured == null) return test_host.finish(h, job, reason, message);
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        if (builtin.is_test) if (h.structured == null) test_host.drain(h, false, .cancelled, "");
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                h.unfill(job);
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        if (h.info_.admit) |a| { // shared_pool (upstream v1.0.7 lane_host.zig:277-283): the next request waits for room (always there when idle)
            const q = h.queued.items[0].request;
            // four_user_profile join (shared_pool's admission with prefill_interleave's filling queue): a filling prompt runs too, and the backend
            // reserves its region at its first step, so the next request waits until every filling one has stepped once
            for (h.filling.items) |j| if (!j.fill_began) {
                h.unlock();
                return false;
            };
            if ((h.core.activeCount() > 0 or h.filling.items.len > 0) and !a.fits(a.ctx, @intCast(q.prompt.len), q.max_tokens)) {
                h.unlock();
                return false;
            }
        }
        const job = h.queued.orderedRemove(0);
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        const r = job.request;
        if (builtin.is_test) if (r.structure != null and h.structured == null) return test_host.admit(h, job, cancelled, drop, cancel, prefilled, deliver, remove);
        // grammar_server: a structured request's Rules from the role that owns the compiler (release lane_host.zig:321-324,
        // 380-383), taken before anything else is held; a host without one refuses it with the name grammar_tests_c's fixture
        // uses (grammar_tests_c_host_fixture.zig:25), so it is never drawn plain.
        var rules: ?*lanes.grammar.Rules = null;
        if (r.structure != null) {
            const s = h.structured orelse return h.drop(job, "IncompleteMaskedAdapter");
            rules = s.rules(s.ctx, r) catch |e| return h.drop(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
        }
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (h.cache) |store| if (store.lookup(h.gpa, r.prompt, r.history_len, r.shared_prefixes, r.chunks)) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch {
            if (rules) |x| x.close();
            return h.drop(job, "the drafter could not start");
        };
        // user_drafts: a driver that drafts alone (core_undrafted) leaves the lane core's streams undrafted; the requests it
        // may draft say so (lone_drafts). Without such a driver both are as before. shared_verify: a driver whose lane core
        // drafts (core_drafts) has the core draft exactly those requests.
        const undrafted = if (h.lone) |l| l.core_undrafted else false;
        const core_drafts = if (h.lone) |l| l.core_drafts else false;
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = if (core_drafts) r.drafts and h.loneRequest(r) else r.drafts and !undrafted,
            .lone_drafts = undrafted and h.loneRequest(r),
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .chunks = r.chunks,
            .reuse = reuse,
            .rules = rules, // consumed, also by its own failure
        }) catch |e| {
            job.proposer.deinit();
            return h.drop(job, if (e == error.OutOfMemory) "out of memory" else @errorName(e));
        };
        job.started = true;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        if (h.loneFits(job)) return h.runLone(job, began);
        if (h.core.fills() and job.stream.rules == null) { // prefill_interleave: the prompt goes in a step a turn (fillTurn)
            h.filling.append(h.gpa, job) catch return h.drop(job, "out of memory");
            return true;
        }
        h.core.addStream(&job.stream) catch |e| return if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    /// prefill_interleave (#623): the oldest filling job's next prompt step; the live streams' rounds run between steps.
    fn fillOne(h: *LaneHost) void {
        if (h.filling.items.len == 0) return;
        const job = h.filling.items[0];
        const first = !job.fill_began;
        job.fill_began = true;
        const done = h.core.fillStream(&job.stream, first) catch |e| {
            _ = if (e == error.Cancelled) h.cancel(job) else h.drop(job, h.words(e));
            return;
        };
        if (!done) return;
        _ = h.filling.orderedRemove(0);
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    /// prefill_interleave: a fill turn (`steps` prompt steps) once the live streams had `rounds` rounds, at once when none is live.
    fn fillTurn(h: *LaneHost) void {
        if (h.filling.items.len == 0) return;
        if (h.core.activeCount() > 0 and h.fill_rounds < h.fill_policy.rounds) return;
        h.fill_rounds = 0;
        for (0..h.fill_policy.steps) |_| {
            if (h.filling.items.len == 0) return;
            h.fillOne();
        }
    }

    fn unfill(h: *LaneHost, job: *Job) void {
        for (h.filling.items, 0..) |j, i| if (j == job) {
            _ = h.filling.orderedRemove(i);
            return;
        };
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        if (h.cache) |store| store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        if (!h.loneRequest(job.request)) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0;
    }

    /// The request a lone driver may take (sampling only when it drives sampled streams).
    fn loneRequest(h: *const LaneHost, r: *const Request) bool {
        const lone = h.lone orelse return false;
        return !((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null);
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        job.delivered = 0;
        job.began = began;
        return h.driveLone(job, h.lone.?.run);
    }

    /// user_drafts: the one stream the lane core decodes, nothing else admitted, queued, filling or cancelling, back to the
    /// backend's lone driver when it can take it (its drafted rounds again); an arrival hands it back as at admission.
    fn resumeLone(h: *LaneHost) bool {
        const lone = h.lone orelse return false;
        const take = lone.resume_lone orelse return false;
        const can = lone.resumable orelse return false;
        h.lock();
        const one = h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.filling.items.len == 0 and !h.closing and h.core.activeCount() == 1;
        const job: ?*Job = if (one) h.admitted.items[0] else null;
        h.unlock();
        const j = job orelse return false;
        if (!j.started or j.stream.finished or j.stream.inflight != null or !h.loneRequest(j.request) or !can(lone.ctx, &j.stream)) return false;
        h.core.detach(&j.stream);
        j.prefill_sent = true; // its prefilled event went out with its prompt pass
        return h.driveLone(j, take);
    }

    fn driveLone(h: *LaneHost, job: *Job, drive: *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool) bool {
        h.lone_job = job;
        const began = job.began;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing;
            }
        };
        const paused = drive(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.unfill(job);
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        h.lock();
        defer h.unlock();
        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            while (h.admitOne()) {}
            h.fillTurn();
            h.noteLive();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.filling.items.len > 0) { // prefill_interleave: nothing decodes: the next step at once
                    h.unlock();
                    continue;
                }
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            if (h.resumeLone()) continue; // user_drafts: a stream left alone goes back to its backend's drafted rounds
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            h.fill_rounds +|= 1;
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        if (builtin.is_test) if (h.structured == null) test_host.drain(h, true, .failed, message);
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.filling.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discardAs(&job.stream, .@"error"); // released as failed, as finish reports it
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        if (builtin.is_test) if (h.structured == null) test_host.drain(h, true, .cancelled, "");
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

test { _ = @import("lane_host_serve_test.zig"); }

test { _ = @import("lane_host_test.zig"); }

test { _ = @import("lane_host_fill_test.zig"); } // prefill_interleave: a prompt a step a round through the real host

test "grammar_tests_r actual lone guard excludes structured with eligible warmed driver and plain reuse" {
    try @import("grammar_tests_r_host_test.zig").run(LaneHost, LaneHost.Job, LaneHost.loneFits, LaneHost.runLone);
}
