//! GPT-6.1 Sol via Hermes: actual LaneHost submit/admit/compile/finish; fake backend and opaque compiler owner.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const Host = @import("lane_host.zig").LaneHost;
const a = std.testing.allocator;
const io = std.testing.io;
const be = lanes.backend;
const AllocTrace = @import("grammar_tests_c_alloc_trace.zig").Trace;
pub const Target = struct {
    allocation_trace: ?*AllocTrace = null,
    owner: ?*lanes.Stream = null,
    entered: usize = 0,
    released: usize = 0,
    forwards: usize = 0,
    effective_no_draft: bool = false,
    fail_verify: bool = false,
    pub fn self(p: *anyopaque) *Target { return @ptrCast(@alignCast(p)); }
    pub fn backend(t: *Target) be.Backend { return .{ .ptr = t, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = masked,
        .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } }; }
    pub fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        const t = self(p); if (s.isCancelled()) return error.Cancelled;
        t.owner = s; t.entered += 1; t.forwards += 1;
        t.effective_no_draft = !s.drafts and s.proposer == null;
        if (comptime @hasField(lanes.Stream, "backend_lease")) if (s.rules != null) { s.backend_lease = .entered; };
        if (t.allocation_trace) |trace| trace.set(.first_mask);
    }
    pub fn first(_: *anyopaque, _: *lanes.Stream, _: u64) !u64 { return 2; }
    pub fn masked(_: *anyopaque, _: *lanes.Stream, _: u64, mask: []const u32) !u64 {
        if (mask.len != 1 or mask[0] != 1 << 7) return error.MaskShape; return 7;
    }
    pub fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) !u64 { return error.Unsupported; }
    pub fn read(p: *anyopaque, token: u64) !u32 {
        if (self(p).allocation_trace) |trace| trace.set(.core_add);
        return @intCast(token);
    }
    pub fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const t = self(p); t.forwards += 1;
        if (t.fail_verify) return error.FakeVerifyFailed;
        for (windows, out) |w, v| {
            if (w.stream.rules != null and (w.masks.len != 1 or w.masks[0] != 1 << 8)) return error.MaskShape;
            @memset(v.sampled, if (w.stream.rules != null) 8 else 2);
        }
    }
    pub fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) !void {}
    pub fn draft(_: *anyopaque, _: []const be.DraftRequest) !void { return error.DraftsUnsupported; }
    pub fn release(p: *anyopaque, s: *lanes.Stream) void {
        const t = self(p); t.released += 1;
        std.debug.assert(t.owner == s); t.owner = null;
    }
};
pub const Fixture = struct {
    gpa: std.mem.Allocator = a,
    made: usize = 0,
    allocation_trace: ?*AllocTrace = null,
    oom: bool = false,
    admission_oom: bool = false,
    host: *Host = undefined,
    target: *Target = undefined,
    rules: lanes.grammar.Rules = undefined,
    owned_rules: ?*lanes.grammar.Rules = null,
    request: *const api.Request = undefined,
    canary: bool = true,
    compiler_live: bool = true,
    compiled: usize = 0,
    accepted: usize = 0,
    closes: usize = 0,
    close_after_release: bool = true,
    cancel_at: ?[]const u8 = null,
    fail_compile: bool = false,
    fail_fill: bool = false,
    cancel_seen: bool = false,
    effective_no_draft: bool = false,
    finished_seen: bool = false,
    destroyed: usize = 0,
    ack: std.atomic.Value(bool) = .init(false),
    reason: api.Reason = .failed,
    tokens: [8]u32 = undefined,
    n: usize = 0,
    pub fn self(p: *anyopaque) *Fixture { return @ptrCast(@alignCast(p)); }
    pub fn init(f: *Fixture) void {
        f.rules = .{ .ptr = f, .width = 32, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    pub fn compile(p: *anyopaque, r: *const api.Request) !*lanes.grammar.Rules {
        const f = self(p); std.debug.assert(f.canary and f.compiler_live and r == f.request);
        std.debug.assert(std.mem.eql(u8, r.structure.?.text, "lease-canary") and r.structure.?.compiler == null);
        f.compiled += 1; if (f.fail_compile) return error.FakeCompileFailed;
        f.rules.after = r.structure.?.after;
        if (f.allocation_trace) |trace| trace.set(.compiler_rules);
        const owned = try f.gpa.create(lanes.grammar.Rules); owned.* = f.rules; f.owned_rules = owned; f.made += 1;
        return owned;
    }
    pub fn observe(p: *anyopaque, _: api.Id, transition: anytype) void {
        const f = self(p); const name = @tagName(transition);
        if (!std.mem.eql(u8, name, "destroyed")) std.debug.assert(f.canary and f.compiler_live);
        if (f.allocation_trace) |trace| {
            if (std.mem.eql(u8, name, "before_transfer")) trace.set(.stream_init);
            if (std.mem.eql(u8, name, "after_transfer")) trace.set(.core_add);
            if (std.mem.eql(u8, name, "after_event_return") and !f.finished_seen) trace.set(.round);
        }
        if (std.mem.eql(u8, name, "after_transfer")) f.effective_no_draft = true;
        if (f.cancel_at) |at| if (std.mem.eql(u8, at, name) and !f.cancel_seen) {
            f.cancel_seen = true; f.host.engine().cancel(1);
        };
        if (std.mem.eql(u8, name, "destroyed")) f.destroyed += 1;
        if (std.mem.eql(u8, name, "after_event_return") and f.finished_seen) f.ack.store(true, .release);
    }
    pub fn fill(p: *anyopaque, words: []u32) !void {
        const f = self(p); std.debug.assert(f.canary and f.compiler_live);
        if (f.fail_fill) return error.FakeFillFailed;
        @memset(words, 0); words[0] = @as(u32, 1) << (if (f.accepted == 0) @as(u5,7) else 8);
    }
    pub fn accept(p: *anyopaque, token: u32) !bool {
        const f = self(p); std.debug.assert(f.canary and f.compiler_live);
        if (token != (if (f.accepted == 0) @as(u32,7) else 8)) return false;
        f.accepted += 1; return true;
    }
    pub fn rollback(_: *anyopaque, _: usize) !void { return error.UnusedRollback; }
    pub fn terminated(p: *anyopaque) bool { return self(p).accepted >= 2; }
    pub fn close(p: *anyopaque) void {
        const f = self(p); std.debug.assert(f.canary and f.compiler_live);
        f.closes += 1; f.close_after_release = f.target.owner == null and f.target.entered == f.target.released;
        f.gpa.destroy(f.owned_rules.?); f.owned_rules = null;
    }
    pub fn event(p: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const f = self(p); std.debug.assert(f.canary and f.compiler_live);
        switch (ev.*) {
            .tokens => |tokens| { @memcpy(f.tokens[f.n..][0..tokens.len], tokens); f.n += tokens.len; },
            .finished => |done| {
                f.reason = done.reason; f.finished_seen = true; f.oom = std.mem.eql(u8,done.message,"OutOfMemory");
                f.admission_oom = std.mem.eql(u8, done.message, "out of memory");
                if (f.request.structure == null or comptime !@hasField(Host,"test_attachment")) f.ack.store(true,.release);
            },
            else => {},
        }
    }
    pub fn wait(f: *Fixture) !void {
        const deadline = std.Io.Clock.awake.now(io).toNanoseconds() + 5*std.time.ns_per_s;
        while (!f.ack.load(.acquire)) {
            if (std.Io.Clock.awake.now(io).toNanoseconds() > deadline) return error.HostFixtureTimeout;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
    }
};
fn check(comptime name: []const u8, good: bool) !void {
    if (!good) { std.debug.print("{s}\n", .{name}); return error.CompiledOwnerTransferRuntimeViolation; }
}
pub fn allocations() !void {
    inline for ([_]bool{false,true}) |cancel_after_transfer| {
        var failures: usize = 0;
        var submit_failures: usize = 0;
        var admission_failures: usize = 0;
        for (0..128) |index| {
            var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = index });
            var trace = AllocTrace{ .failing = &failing };
            const result = allocationTransaction(trace.allocator(), cancel_after_transfer, &trace);
            try check("QuiescentShutdownRuntimeViolation", failing.allocated_bytes == failing.freed_bytes);
            if (result) |_| {
                if (!failing.has_induced_failure) break;
                // A swallowed observation allocation is not a propagated OOM gate.
                std.debug.print("CELL HOST allocation index={d} stage={s} cancel_after_transfer={} result=completed_with_induced_failure cleanup=PASS\n", .{index,@tagName(trace.failed_stage.?),cancel_after_transfer});
            } else |err| {
                try std.testing.expect(failing.has_induced_failure);
                const stage = trace.failed_stage orelse return error.MissingAllocationBoundary;
                switch (stage) {
                    .submit_job, .submit_queue => {
                        try std.testing.expectEqualStrings("Busy", @errorName(err));
                        submit_failures += 1;
                    },
                    else => {
                        try std.testing.expectEqualStrings("OutOfMemory", @errorName(err));
                        admission_failures += 1;
                    },
                }
                failures += 1;
                std.debug.print("CELL HOST allocation index={d} stage={s} cancel_after_transfer={} error={s} induced=true no_jobs_no_rules_no_lease=PASS\n", .{index,@tagName(stage),cancel_after_transfer,@errorName(err)});
            }
            if (index == 127) return error.AllocationSweepIncomplete;
        }
        try std.testing.expect(submit_failures == 2 and admission_failures > 0);
        std.debug.print("CELL HOST allocations cancel_after_transfer={} exhausted_indices={d} Busy_boundaries={d} OOM_boundaries={d} compiler=FAKE\n", .{cancel_after_transfer,failures,submit_failures,admission_failures});
    }
}
fn allocationTransaction(gpa: std.mem.Allocator, cancel_after_transfer: bool, trace: *AllocTrace) !void {
    var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
    var target = Target{ .allocation_trace = trace }; var clock = lanes.fake.FixedClock{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock()); defer core.deinit();
    var host = Host.init(gpa, io, &core, .{ .lanes = 1 });
    var stopped = false; defer if (!stopped) host.stop();
    const r = api.Request{ .prompt = &.{3,4}, .max_tokens = 3, .drafts = true,
        .structure = .{ .kind = .json_schema, .text = "lease-canary" } };
    var f = Fixture{ .gpa = gpa, .allocation_trace = trace, .host = &host, .target = &target, .request = &r,
        .cancel_at = if (cancel_after_transfer) "after_transfer" else null }; f.init();
    const seam = @import("grammar_tests_c_host_fixture.zig");
    host.test_attachment = .{ .ctx = &f, .compile = Fixture.compile, .observe = struct {
        fn call(p: *anyopaque, id: api.Id, step: seam.Transition) void { Fixture.observe(p,id,step); }
    }.call };
    host.engine().submit(1, &r, .{ .ctx = &f, .event = Fixture.event }) catch |err| {
        try check("CompiledOwnerTransferRuntimeViolation", f.compiled == 0 and f.made == 0 and f.closes == 0 and !f.finished_seen);
        try check("BackendReleaseOrderRuntimeViolation", target.owner == null and target.entered == 0 and target.released == 0 and target.forwards == 0);
        try check("QuiescentShutdownRuntimeViolation", host.queued.items.len == 0 and host.admitted.items.len == 0 and core.live.items.len == 0);
        return err;
    };
    trace.set(.admitted_append);
    try host.start();
    try f.wait(); host.stop(); stopped = true;
    try check("BackendReleaseOrderRuntimeViolation", target.owner == null and target.entered == target.released and f.close_after_release);
    try check("CompiledOwnerTransferRuntimeViolation", f.owned_rules == null and f.closes == f.made);
    // Host.stop deinitializes its arrays: their fields are undefined, not observable empty lists.
    try check("QuiescentShutdownRuntimeViolation", core.live.items.len == 0 and host.thread == null and host.closing and f.destroyed == 1 and f.finished_seen);
    if (f.admission_oom) {
        try check("CompiledOwnerTransferRuntimeViolation", trace.failed_stage == .admitted_append and f.compiled == 0 and target.entered == 0);
        return error.OutOfMemory;
    }
    if (f.oom) {
        try check("CompiledOwnerTransferRuntimeViolation", trace.failed_stage != .submit_job and trace.failed_stage != .submit_queue and trace.failed_stage != .admitted_append);
        return error.OutOfMemory;
    }
    try std.testing.expect(f.reason != .failed);
}

pub fn run() !void {
    const Case = enum { normal, zero, queued_cancel, dequeued_cancel, compiled_cancel, transfer_cancel, entered_cancel, compile_error, fill_error };
    inline for (std.meta.tags(Case)) |case| {
        var cfg = try lanes.Config.init(a, .{}, 1, 8); defer cfg.deinit(a);
        var target = Target{}; var clock = lanes.fake.FixedClock{};
        var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock()); defer core.deinit();
        var host = Host.init(a, io, &core, .{ .lanes = 1 });
        const r = api.Request{ .prompt = &.{3,4}, .max_tokens = if (case == .zero) 0 else 3,
            .drafts = true, .structure = .{ .kind = .json_schema, .text = "lease-canary" } };
        var f = Fixture{ .host = &host, .target = &target, .request = &r,
            .fail_compile = case == .compile_error, .fail_fill = case == .fill_error,
            .cancel_at = switch (case) { .dequeued_cancel => "dequeued", .compiled_cancel => "compiled", .transfer_cancel => "after_transfer", .entered_cancel => "after_event_return", else => null } }; f.init();
        if (comptime @hasField(Host, "test_attachment")) {
            const seam = @import("grammar_tests_c_host_fixture.zig");
            host.test_attachment = .{ .ctx = &f, .compile = Fixture.compile, .observe = struct {
                pub fn call(p: *anyopaque, id: api.Id, step: seam.Transition) void { Fixture.observe(p,id,step); }
            }.call };
        }
        try host.engine().submit(1, &r, .{ .ctx = &f, .event = Fixture.event });
        if (case == .queued_cancel) host.engine().cancel(1);
        try host.start();
        f.wait() catch |err| { host.stop(); return err; };
        host.stop();
        const compiled: usize = if (case == .queued_cancel or case == .dequeued_cancel) 0 else 1;
        try check("CompiledOwnerTransferRuntimeViolation", f.compiled == compiled);
        try check("CompiledOwnerTransferRuntimeViolation", f.closes == (if (case == .compile_error) @as(usize,0) else compiled));
        try check("BackendReleaseOrderRuntimeViolation", target.entered == target.released and target.owner == null and f.close_after_release);
        try check("RequestLeaseRuntimeViolation", f.canary and f.compiler_live and f.finished_seen);
        if (case == .normal) {
            try check("CompiledOwnerTransferRuntimeViolation", target.effective_no_draft);
            try std.testing.expectEqualSlices(u32, &.{7,8}, f.tokens[0..f.n]);
        } else try std.testing.expectEqual(@as(usize,0), f.n);
        f.compiler_live = false; f.canary = false;
        std.debug.print("CELL HOST case={s} actual_submit_admit=PASS compile={d} close={d} entered={d} released={d} request_ack=PASS compiler=FAKE\n", .{@tagName(case),f.compiled,f.closes,target.entered,target.released});
    }
}
