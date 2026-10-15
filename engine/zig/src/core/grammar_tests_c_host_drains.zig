//! GPT-6.1 Sol via Hermes: actual host drains under allocation exhaustion; separately owned fake compiler Rules.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const base = @import("grammar_tests_c_host_test.zig");
const seam = @import("grammar_tests_c_host_fixture.zig");
const a = std.testing.allocator;
const io = std.testing.io;
fn check(good: bool) !void {
    if (!good) { std.debug.print("QuiescentShutdownRuntimeViolation\n", .{}); return error.QuiescentShutdownRuntimeViolation; }
}
pub fn drains() !void {
    const Case = enum { cancel, stop, failAll };
    inline for (std.meta.tags(Case)) |case| {
        const Context = struct {
            selected: Case,
            f: base.Fixture,
            failing: *std.testing.FailingAllocator,
            armed: std.atomic.Value(bool) = .init(false),
            timed_out: bool = false,
            pub fn self(p: *anyopaque) *@This() { return @ptrCast(@alignCast(p)); }
            pub fn compile(p: *anyopaque, r: *const api.Request) !*lanes.grammar.Rules { return base.Fixture.compile(&self(p).f, r); }
            pub fn event(p: *anyopaque, id: api.Id, ev: *const api.Event) void { base.Fixture.event(&self(p).f,id,ev); }
            pub fn observe(p: *anyopaque, id: api.Id, transition: seam.Transition) void {
                const c = self(p); base.Fixture.observe(&c.f,id,transition);
                if (transition != .after_event_return or c.f.finished_seen or c.armed.load(.acquire)) return;
                c.failing.fail_index = c.failing.alloc_index;
                if (c.selected == .cancel) c.f.host.engine().cancel(id);
                c.armed.store(true, .release);
                if (c.selected == .stop) {
                    const s = c.f.target.owner.?;
                    const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
                    while (!s.isCancelled()) {
                        if (std.Io.Clock.awake.now(io).toNanoseconds() > deadline) { c.timed_out = true; return; }
                        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch { c.timed_out = true; return; };
                    }
                }
            }
        };
        var failing = std.testing.FailingAllocator.init(a, .{});
        const gpa = failing.allocator();
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var target = base.Target{ .fail_verify = case == .failAll }; var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        var host = api.LaneHost.init(gpa, io, &core, .{ .lanes = 1 });
        const request = api.Request{ .prompt = &.{3,4}, .max_tokens = 3, .drafts = true,
            .structure = .{ .kind = .json_schema, .text = "lease-canary" } };
        var c = Context{ .selected = case, .f = .{ .gpa = gpa, .host = &host, .target = &target, .request = &request }, .failing = &failing }; c.f.init();
        host.test_attachment = .{ .ctx = &c, .compile = Context.compile, .observe = Context.observe };
        try host.engine().submit(1,&request,.{ .ctx = &c, .event = Context.event });
        try host.start();
        if (case == .stop) {
            const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5 * std.time.ns_per_s;
            while (!c.armed.load(.acquire)) {
                if (std.Io.Clock.awake.now(io).toNanoseconds() > deadline) { host.stop(); core.deinit(); return error.HostDrainBarrierTimeout; }
                try std.Io.sleep(io,.fromMilliseconds(1),.awake);
            }
        } else c.f.wait() catch |err| { host.stop(); core.deinit(); return err; };
        host.stop();
        const core_empty = core.live.items.len == 0; core.deinit();
        try check(c.armed.load(.acquire) and !c.timed_out and failing.has_induced_failure);
        try check(core_empty and host.thread == null and c.f.finished_seen and c.f.destroyed == 1);
        try check(target.owner == null and target.entered == 1 and target.released == 1 and c.f.closes == 1 and c.f.close_after_release and c.f.owned_rules == null);
        try check(failing.allocated_bytes == failing.freed_bytes);
        try std.testing.expectEqual(if (case == .failAll) api.Reason.failed else api.Reason.cancelled,c.f.reason);
        try std.testing.expectEqual(@as(usize,if (case == .failAll) 1 else 0),c.f.n);
        std.debug.print("CELL HOST exhausted_drain={s} induced=true actual_submit_admit=PASS one_destroy_one_release_one_close=PASS no_jobs_no_rules_no_lease=PASS compiler=FAKE\n", .{@tagName(case)});
    }
}
pub fn kindsAndReuse() !void {
    const Kind = @TypeOf((api.Structure{ .kind = .json }).kind);
    inline for (std.meta.tags(Kind)) |kind| {
        var cfg = try lanes.Config.init(a,.{},1,8); defer cfg.deinit(a);
        var target = base.Target{}; var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a,&cfg,target.backend(),clock.clock()); defer core.deinit();
        var host = api.LaneHost.init(a,io,&core,.{ .lanes = 1 });
        const request = api.Request{ .prompt = &.{3,4}, .max_tokens = 3, .drafts = true,
            .structure = .{ .kind = kind, .text = "lease-canary" } };
        var f = base.Fixture{ .host = &host, .target = &target, .request = &request }; f.init();
        host.test_attachment = .{ .ctx = &f, .compile = base.Fixture.compile, .observe = struct {
            pub fn call(p: *anyopaque,id: api.Id,step: seam.Transition) void { base.Fixture.observe(p,id,step); }
        }.call };
        try host.engine().submit(1,&request,.{ .ctx = &f, .event = base.Fixture.event });
        try host.start();
        f.wait() catch |err| { host.stop(); return err; };
        const plain = api.Request{ .prompt = &.{3,4}, .max_tokens = 1, .drafts = false };
        var next = base.Fixture{ .host = &host, .target = &target, .request = &plain }; next.init();
        host.engine().submit(2,&plain,.{ .ctx = &next, .event = base.Fixture.event }) catch |err| { host.stop(); return err; };
        next.wait() catch |err| { host.stop(); return err; };
        host.stop();
        try check(f.compiled == 1 and f.made == 1 and f.closes == 1 and f.destroyed == 1 and f.close_after_release and f.owned_rules == null);
        try check(next.compiled == 0 and target.entered == 2 and target.released == 2 and target.owner == null and core.live.items.len == 0);
        try std.testing.expectEqualSlices(u32,&.{7,8},f.tokens[0..f.n]);
        try std.testing.expectEqualSlices(u32,&.{2},next.tokens[0..next.n]);
        std.debug.print("CELL HOST kind={s} descriptor_compile=FAKE owned_Rules=PASS same_host_plain_next=PASS\n", .{@tagName(kind)});
    }
}
