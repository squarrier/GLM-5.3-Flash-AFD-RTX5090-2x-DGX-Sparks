//! Narrow private-boundary prototype, not full N01-N12 acceptance.
const std = @import("std");
const fixture = @import("qualification_59878042_route_test.zig");
const Observation = @import("sse_test_observe.zig").Test;
const Probe = struct {
    tags: [1024]std.meta.Tag(Observation.Event) = undefined,
    n: usize = 0,
    starts: usize = 0,
    stops: usize = 0,
    sinks: usize = 0,
    generations: usize = 0,
    mutex_checks: usize = 0,
    box_begin: usize = 0,
    box_end: usize = 0,
    enabled: bool,
    fn record(ctx: *anyopaque, event: Observation.Event) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        std.debug.assert(p.n < p.tags.len);
        p.tags[p.n] = std.meta.activeTag(event); p.n += 1;
        switch (event) {
            .session => |s| {
                std.debug.assert(p.enabled);
                if (s.phase == .start) { p.starts += 1; std.debug.assert(s.active and s.attached); }
                if (s.phase == .stop) { p.stops += 1; std.debug.assert(!s.active and !s.attached); }
            },
            .sink => |s| { p.sinks += 1; std.debug.assert(s.beat_present == p.enabled and s.idle_present == p.enabled); },
            .generate_enter => |s| { p.generations += 1; std.debug.assert(s.sink_present and s.idle_present == p.enabled); },
            .idle => |s| {
                std.debug.assert(!s.finished_snapshot and s.idle_present == p.enabled);
                std.debug.assert(s.mutex.tryLock()); s.mutex.unlock(s.io); p.mutex_checks += 1;
            },
            .box_deinit => |s| { std.debug.assert(s.finished); if (s.phase == .begin) p.box_begin += 1 else p.box_end += 1; },
            .drain_end => |s| std.debug.assert(s.finished),
            else => {},
        }
    }
    fn at(p: *Probe, tag: std.meta.Tag(Observation.Event)) ?usize {
        return std.mem.indexOfScalar(std.meta.Tag(Observation.Event), p.tags[0..p.n], tag);
    }
};
test "qualification_aee48b93 direct Run idle-null Session attachment and private Mailbox teardown prototype" {
    var cells: usize = 0; var off: usize = 0; var mutex: usize = 0;
    for (fixture.paths) |path| for ([_]fixture.Mode{ .success, .cancelled, .closed_open, .finished_backlog, .busy, .context }) |mode| for ([_]i64{ 0, 0, std.time.ns_per_s }, 0..) |period, setting| {
        var p: Probe = .{ .enabled = period > 0 };
        var hook: Observation.Hook = .{ .ctx = &p, .record = Probe.record };
        const saved = Observation.active; Observation.active = &hook;
        defer Observation.active = saved;
        const bytes = try fixture.run(path, mode, true, period, if (setting == 0) "g-unset" else if (setting == 1) "g-zero" else "g-on"); defer std.testing.allocator.free(bytes);
        try std.testing.expect(p.at(.run_enter) != null and p.at(.run_exit).? == p.n - 1);
        const generated = mode == .success or mode == .cancelled or mode == .finished_backlog;
        try std.testing.expectEqual(@as(usize, if (generated) 1 else 0), p.generations);
        try std.testing.expectEqual(p.generations, p.sinks);
        try std.testing.expectEqual(p.generations, p.box_begin);
        try std.testing.expectEqual(p.box_begin, p.box_end);
        try std.testing.expectEqual(@as(usize, if (generated and period > 0) 1 else 0), p.starts);
        try std.testing.expectEqual(p.starts, p.stops);
        if (generated) {
            try std.testing.expect(p.at(.box_deinit).? < p.at(.request_end).?);
            try std.testing.expect(p.at(.request_end).? < p.at(.generated).?);
            if (mode == .cancelled) try std.testing.expect(p.at(.drain_end).? < p.at(.box_deinit).?);
        }
        cells += 1; if (period == 0 and generated) off += 1; mutex += p.mutex_checks;
    };
    std.debug.print("qualification_ca75a6a8 route_cells={d} direct_off_generations={d} actual_mailbox_mutex_checks={d}\n", .{ cells, off, mutex });
}
