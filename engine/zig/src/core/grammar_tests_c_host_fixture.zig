//! GPT-6.1 Sol via Hermes: test-only fake attachment; Structure.Compiler remains opaque.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
pub const Transition = enum { dequeued, compiled, before_transfer, after_transfer, before_backend, after_event_return, before_finished, destroyed };
pub const Attachment = struct {
    ctx: *anyopaque,
    compile: *const fn (*anyopaque, *const api.Request) anyerror!*lanes.grammar.Rules,
    observe: ?*const fn (*anyopaque, api.Id, Transition) void = null,
};
pub fn observe(h: anytype, job: anytype, step: Transition) void {
    if (job.request.structure == null) return;
    const attachment = h.test_attachment orelse return;
    if (attachment.observe) |callback| callback(attachment.ctx, job.id, step);
}
pub fn markCancelled(h: anytype, id: api.Id) bool {
    var found = false;
    for (h.queued.items) |job| if (job.id == id and job.request.structure != null) { job.test_cancelled = true; found = true; };
    for (h.admitted.items) |job| if (job.id == id and job.request.structure != null) { job.test_cancelled = true; found = true; };
    return found;
}
pub fn admit(h: anytype, job: anytype, cancelled: anytype, drop: anytype, cancel: anytype, prefilled: anytype, deliver: anytype, remove: anytype) bool {
    observe(h, job, .dequeued);
    if (cancelled(job)) return cancel(h, job);
    const attachment = h.test_attachment orelse return drop(h, job, "IncompleteMaskedAdapter");
    var pending_rules: ?*lanes.grammar.Rules = attachment.compile(attachment.ctx, job.request) catch |err| return drop(h, job, @errorName(err));
    if (cancelled(job)) { pending_rules.?.close(); pending_rules = null; return cancel(h, job); }
    observe(h, job, .compiled);
    observe(h, job, .before_transfer);
    if (cancelled(job)) { pending_rules.?.close(); pending_rules = null; return cancel(h, job); }
    const r = job.request;
    const init_rules = pending_rules.?;
    pending_rules = null; // Stream.init consumes on call, including its own allocation failure.
    job.stream = lanes.Stream.init(h.gpa, .{ .id = "structured-fixture", .prompt = r.prompt, .max_new = r.max_tokens,
        .eos = r.eos, .sampling = r.sampling, .drafts = false, .proposer = null, .rules = init_rules,
        .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
        .cancel_check = .{ .ptr = job, .check = cancelled }, .think_budget = r.think_budget,
        .think_close = r.think_close, .think_end = if (r.think_end) |t| t else -1,
        .loop_guard = r.loop_guard, .chunks = r.chunks }) catch |err| return drop(h, job, @errorName(err));
    job.started = true;
    observe(h, job, .after_transfer);
    if (cancelled(job)) return cancel(h, job);
    job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
    observe(h, job, .before_backend);
    h.core.addStream(&job.stream) catch |err| return if (err == error.Cancelled) cancel(h, job) else drop(h, job, @errorName(err));
    prefilled(h, job, job.began);
    if (deliver(h, job)) remove(h, job);
    return true;
}
pub fn finish(h: anytype, job: anytype, reason: api.Reason, message: []const u8) void {
    observe(h, job, .before_finished);
    h.gpa.free(job.marks);
    const s = &job.stream;
    const stats: api.Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted,
        .min_rows = s.min_rows, .loop_period = s.loop_period,
        .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done-job.began))))) / 1e9 else null } else .{};
    if (job.started) {
        std.debug.assert(s.backend_lease != .entered);
        s.deinit(h.gpa); // Matcher close precedes finished acknowledgment in this test-owned branch.
    }
    const attachment = h.test_attachment; const id = job.id;
    const event = api.Event{ .finished = .{ .reason = reason, .stats = stats, .message = message } };
    job.sink.event(job.sink.ctx, id, &event);
    // No request or request slices are read after the finished callback returns.
    if (attachment) |at| if (at.observe) |callback| callback(at.ctx, id, .after_event_return);
    h.gpa.destroy(job);
    if (attachment) |at| if (at.observe) |callback| callback(at.ctx, id, .destroyed);
}
pub fn drain(h: anytype, all: bool, reason: api.Reason, message: []const u8) void {
    while (true) {
        h.mutex.lockUncancelable(h.io);
        var removed: ?@TypeOf(h.queued.items[0]) = null;
        for (h.queued.items, 0..) |job, i| if (job.request.structure != null and (all or h.closing or job.test_cancelled)) {
            removed = h.queued.orderedRemove(i); break;
        };
        if (removed == null) for (h.admitted.items, 0..) |job, i| if (job.request.structure != null and (all or h.closing or job.test_cancelled)) {
            removed = h.admitted.orderedRemove(i); break;
        };
        h.mutex.unlock(h.io);
        const job = removed orelse break;
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        finish(h, job, reason, message);
    }
}
