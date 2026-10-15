//! grammar_tests_c3 owned request slices and fake compiler borrow until callback-return acknowledgement.
const std = @import("std");
const api = @import("engine_api.zig");
const HostFixture = @import("grammar_tests_c_host_test.zig");
const lanes = @import("lanes");
const a = std.testing.allocator;
const io = std.testing.io;
pub const Lease = struct {
    prompt: []u32,
    text: []u8,
    request: api.Request,
    borrowed: std.atomic.Value(bool) = .init(true),
    compiler_live: bool = true,
    early_close_refusals: usize = 0,
    pub fn init() !Lease {
        const prompt = try a.dupe(u32, &.{3,4}); errdefer a.free(prompt);
        const text = try a.dupe(u8, "lease-canary");
        return .{ .prompt = prompt, .text = text, .request = .{ .prompt = prompt, .max_tokens = 3, .drafts = true,
            .structure = .{ .kind = .json_schema, .text = text } } };
    }
    pub fn closeCompiler(l: *Lease) !void {
        if (l.borrowed.load(.acquire)) { l.early_close_refusals += 1; return error.OutstandingRequest; }
        l.compiler_live = false;
    }
    pub fn release(l: *Lease) !void {
        if (l.borrowed.load(.acquire)) return error.OutstandingRequest;
        a.free(l.prompt); a.free(l.text);
        l.prompt = &.{}; l.text = &.{};
    }
};
fn check(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s} expected=true observed=false\n", .{name}); return error.RequestLeaseRuntimeViolation; }
}
pub fn run() !void {
    const Case = enum { never_started_stop, queued_cancel, normal, before_transfer_cancel, round_failure };
    inline for (std.meta.tags(Case)) |case| {
        var lease = try Lease.init();
        // Red-source failure still frees caller-owned slices; an undrained Job is reported by Zig's leak detector.
        defer { if (lease.prompt.len != 0) a.free(lease.prompt); if (lease.text.len != 0) a.free(lease.text); }
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var target = HostFixture.Target{ .fail_verify = case == .round_failure }; var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock()); defer core.deinit();
        var host = api.LaneHost.init(a, io, &core, .{ .lanes = 1 });
        const Context = struct {
            lease: *Lease,
            fixture: HostFixture.Fixture,
            ack: std.atomic.Value(bool) = .init(false),
            before_transfer_cancel: bool,
            fn self(p: *anyopaque) *@This() { return @ptrCast(@alignCast(p)); }
            fn compile(p: *anyopaque, request: *const api.Request) !*lanes.grammar.Rules {
                const c = self(p); std.debug.assert(c.lease.compiler_live and c.lease.borrowed.load(.acquire));
                return HostFixture.Fixture.compile(&c.fixture, request);
            }
            fn event(p: *anyopaque, id: api.Id, ev: *const api.Event) void {
                const c = self(p); std.debug.assert(c.lease.compiler_live and c.lease.borrowed.load(.acquire));
                HostFixture.Fixture.event(&c.fixture, id, ev);
            }
            fn observe(p: *anyopaque, id: api.Id, transition: @import("grammar_tests_c_host_fixture.zig").Transition) void {
                const c = self(p);
                HostFixture.Fixture.observe(&c.fixture, id, transition);
                if (transition == .before_transfer) {
                    c.lease.closeCompiler() catch |err| { std.debug.assert(err == error.OutstandingRequest); };
                    if (c.before_transfer_cancel) c.fixture.host.engine().cancel(id);
                }
                if (transition == .after_event_return and c.fixture.finished_seen) {
                    c.lease.borrowed.store(false, .release);
                    c.ack.store(true, .release);
                }
            }
        };
        var context = Context{ .lease = &lease, .fixture = .{ .host = &host, .target = &target, .request = &lease.request },
            .before_transfer_cancel = case == .before_transfer_cancel }; context.fixture.init();
        host.test_attachment = .{ .ctx = &context, .compile = Context.compile, .observe = Context.observe };
        try host.engine().submit(1, &lease.request, .{ .ctx = &context, .event = Context.event });
        try std.testing.expectError(error.OutstandingRequest, lease.closeCompiler());
        try std.testing.expectError(error.OutstandingRequest, lease.release());
        if (case == .queued_cancel) host.engine().cancel(1);
        if (case != .never_started_stop) {
            try host.start();
            context.fixture.wait() catch |err| { host.stop(); return err; };
        }
        host.stop();
        try check("QuiescentShutdownRuntimeViolation", context.fixture.finished_seen and context.fixture.destroyed == 1 and context.ack.load(.acquire));
        try check("RequestLeaseRuntimeViolation", !lease.borrowed.load(.acquire) and lease.compiler_live and lease.early_close_refusals >= 1);
        try check("BackendReleaseOrderRuntimeViolation", target.owner == null and target.entered == target.released and context.fixture.closes == context.fixture.made and context.fixture.owned_rules == null);
        try check("QuiescentShutdownRuntimeViolation", core.live.items.len == 0 and host.thread == null);
        if (case == .never_started_stop or case == .queued_cancel) try check("CompiledOwnerTransferRuntimeViolation", context.fixture.compiled == 0 and target.forwards == 0);
        try lease.closeCompiler(); try lease.release();
        try check("RequestLeaseRuntimeViolation", !lease.compiler_live and lease.prompt.len == 0 and lease.text.len == 0);
        std.debug.print("CELL grammar_tests_c3 HOST case={s} actual_submit_stop=PASS owned_prompt_text=PASS early_compiler_close=REFUSED callback_return_ack=PASS jobs_rules_lease=0 compiler=FAKE\n", .{@tagName(case)});
    }
}
