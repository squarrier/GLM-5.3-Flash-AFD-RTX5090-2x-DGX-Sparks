//! GPT-6.1 Sol via Hermes: real private Generation cancellation against actual LaneHost, socket-free.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const chat = @import("chat.zig");
const Server = @import("server.zig").Server;
const a = std.testing.allocator;
const io = std.testing.io;
const HostFixture = api.TestHostFixture;
const seam = api.TestHostTransitions;
fn check(good: bool) !void {
    if (!good) { std.debug.print("DeliveryCancelRuntimeViolation\n", .{}); return error.DeliveryCancelRuntimeViolation; }
}
pub fn run(comptime Access: type, comptime disconnected: bool) !void { try runInternal(Access, disconnected, false); }
pub fn runLeased(comptime Access: type, comptime disconnected: bool) !void { try runInternal(Access, disconnected, true); }
fn runInternal(comptime Access: type, comptime disconnected: bool, comptime leased: bool) !void {
    const Context = struct {
        f: HostFixture.Fixture,
        box: *Access.Box,
        lease: ?*api.TestRequestLease.Lease = null,
        barrier: std.atomic.Value(bool) = .init(false),
        timed_out: bool = false,
        cutoff_forwards: usize = 0,
        pub fn self(p: *anyopaque) *@This() { return @ptrCast(@alignCast(p)); }
        pub fn compile(p: *anyopaque, r: *const api.Request) !*lanes.grammar.Rules { return HostFixture.Fixture.compile(&self(p).f, r); }
        pub fn event(p: *anyopaque, id: api.Id, ev: *const api.Event) void {
            const c = self(p);
            HostFixture.Fixture.event(&c.f, id, ev);
            Access.event(c.box, id, ev);
        }
        pub fn observe(p: *anyopaque, id: api.Id, transition: seam.Transition) void {
            const c = self(p);
            HostFixture.Fixture.observe(&c.f, id, transition);
            if (transition == .after_event_return and c.f.finished_seen) {
                if (c.lease) |lease| lease.borrowed.store(false, .release);
            }
            if (transition != .after_event_return or c.f.finished_seen) return;
            // The prefilled mailbox event already returned; the consumer cancels outside its callback.
            if (c.f.target.owner) |s| {
                c.barrier.store(true, .release);
                const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
                while (!s.isCancelled()) {
                    if (std.Io.Clock.awake.now(io).toNanoseconds() > deadline) { c.timed_out = true; return; }
                    std.Io.sleep(io, .fromMilliseconds(1), .awake) catch { c.timed_out = true; return; };
                }
                c.cutoff_forwards = c.f.target.forwards;
            }
        }
    };
    var lease: ?api.TestRequestLease.Lease = if (leased) try api.TestRequestLease.Lease.init() else null;
    defer if (lease) |l| { if (l.prompt.len != 0) a.free(l.prompt); if (l.text.len != 0) a.free(l.text); };
    var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
    var target = HostFixture.Target{}; var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock()); defer core.deinit();
    var host = api.LaneHost.init(a, io, &core, .{ .lanes = 1 });
    var stopped = false; defer if (!stopped) host.stop();
    var box = Access.Box{ .gpa = a, .io = io }; defer Access.deinitBox(&box);
    const request = if (leased) lease.?.request else api.Request{ .prompt = &.{3,4}, .max_tokens = 3, .drafts = true,
        .structure = .{ .kind = .json_schema, .text = "lease-canary" } };
    var ctx = Context{ .f = .{ .host = &host, .target = &target, .request = &request }, .box = &box,
        .lease = if (lease) |*l| l else null }; ctx.f.init();
    host.test_attachment = .{ .ctx = &ctx, .compile = Context.compile, .observe = Context.observe };
    var text = @import("structured_refusal_test.zig").ByteText{};
    const srv = try Server.init(a, io, host.engine(), text.text(), .{ .served_name = "fixture", .model_ids = &.{"fixture"}, .keep_warm_s = 0, .enable_thinking = false, .use_drafts = false }, null);
    defer srv.deinit();
    var generation = Access.Gen{ .srv = srv, .a = a, .box = &box, .id = 1, .reply_id = "fixture",
        .sink = null, .thinking = false, .stops = .{ .strings = &.{} }, .ignore_eos = false, .max_tokens = 3, .tools = &.{} };
    try host.engine().submit(1, &request, .{ .ctx = &ctx, .event = Context.event });
    try host.start();
    const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!ctx.barrier.load(.acquire)) {
        if (std.Io.Clock.awake.now(io).toNanoseconds() > deadline) return error.HostDeliveryBarrierTimeout;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
    if (lease) |*l| {
        try std.testing.expectError(error.OutstandingRequest, l.closeCompiler());
        try std.testing.expectError(error.OutstandingRequest, l.release());
    }
    const Closed = struct {
        called: usize = 0,
        pub fn call(p: *anyopaque, _: @import("json.zig").Value) error{Closed}!void { const c: *@This() = @ptrCast(@alignCast(p)); c.called += 1; return error.Closed; }
    };
    var closed = Closed{};
    const result = if (disconnected) Access.gone(&generation) else Access.emit(&generation, .{ .ctx = &closed, .call = Closed.call });
    try std.testing.expectError(error.Cancelled, result);
    try check(generation.engine_done);
    try ctx.f.wait();
    try check(!ctx.timed_out and box.finished and box.reason == .cancelled);
    try check(ctx.cutoff_forwards == target.forwards and target.entered == 1 and target.released == 1 and target.owner == null);
    try check(ctx.f.n == 0 and box.tokens.items.len == 0);
    const accepted_at_ack = ctx.f.accepted;
    const plain = api.Request{ .prompt = &.{3,4}, .max_tokens = 1, .drafts = false };
    var next = HostFixture.Fixture{ .host = &host, .target = &target, .request = &plain }; next.init();
    try host.engine().submit(2, &plain, .{ .ctx = &next, .event = HostFixture.Fixture.event });
    try next.wait();
    host.stop(); stopped = true;
    try check(ctx.f.accepted == accepted_at_ack and next.compiled == 0);
    try check(target.entered == 2 and target.released == 2 and target.owner == null);
    try std.testing.expectEqualSlices(u32, &.{2}, next.tokens[0..next.n]);
    try check(ctx.f.finished_seen and ctx.f.destroyed == 1 and ctx.f.closes == 1 and ctx.f.close_after_release and ctx.f.owned_rules == null);
    try check(ctx.f.n == 0 and box.tokens.items.len == 0 and core.live.items.len == 0);
    try check(closed.called == (if (disconnected) @as(usize,0) else 1));
    if (lease) |*l| {
        try check(!l.borrowed.load(.acquire) and l.compiler_live and l.early_close_refusals == 1);
        try l.closeCompiler(); try l.release();
        std.debug.print("CELL grammar_tests_c3 DELIVERY case={s} early_compiler_close=REFUSED actual_Generation_cancel_drain=PASS owned_request_release_after_ack=PASS compiler=FAKE\n", .{if (disconnected) "gone" else "Closed"});
    }
    std.debug.print("CELL DELIVERY case={s} actual_Generation_cancel_drain=PASS actual_LaneHost=PASS same_host_plain_next=PASS post_ack_old_matcher_accepts=0 post_cutoff_tokens=0 lease_release=1 matcher_close=1 compiler=FAKE\n", .{if (disconnected) "gone" else "Closed"});
}
