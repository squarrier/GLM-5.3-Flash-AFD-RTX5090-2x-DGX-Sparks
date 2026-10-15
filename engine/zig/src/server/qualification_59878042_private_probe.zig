//! Request-owned scalar/private boundary ledger. GPT-6.1 Sol via Hermes.
const std = @import("std");
const O = @import("sse_test_observe.zig").Test;
pub const Probe = struct {
    tags: [4096]std.meta.Tag(O.Event) = undefined,
    n: usize = 0,
    starts: usize = 0,
    stops: usize = 0,
    session_events: usize = 0,
    generates: usize = 0,
    sinks: usize = 0,
    boxes: usize = 0,
    box_ends: usize = 0,
    mutex_checks: usize = 0,
    drain: bool = false,
    enabled: bool,
    streamed: bool,
    gpa: ?*Ledger = null,
    upstream: ?*O.Hook = null,
    pub fn record(ctx: *anyopaque, event: O.Event) void {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        std.debug.assert(p.n < p.tags.len);
        p.tags[p.n] = std.meta.activeTag(event); p.n += 1;
        switch (event) {
            .session => |s| {
                std.debug.assert(p.enabled and p.streamed);
                p.session_events += 1;
                if (s.phase == .start) { p.starts += 1; std.debug.assert(s.active and s.attached); }
                if (s.phase == .stop) { p.stops += 1; std.debug.assert(!s.active and !s.attached); }
            },
            .sink => |s| { p.sinks += 1; std.debug.assert(s.beat_present == p.enabled and s.idle_present == p.enabled); },
            .generate_enter => |s| {
                p.generates += 1;
                std.debug.assert(s.sink_present == p.streamed);
                std.debug.assert(s.idle_present == (p.enabled and p.streamed));
                if (p.gpa) |g| g.armed = true;
            },
            .idle => |s| {
                std.debug.assert(!s.finished_snapshot and !p.drain);
                std.debug.assert(s.idle_present == (p.enabled and p.streamed));
                std.debug.assert(s.mutex.tryLock()); s.mutex.unlock(s.io);
                p.mutex_checks += 1;
            },
            .drain_begin => { std.debug.assert(!p.drain); p.drain = true; },
            .drain_end => |s| { std.debug.assert(p.drain and s.finished); p.drain = false; },
            .box_deinit => |s| {
                if (s.phase == .begin) { p.boxes += 1; if (p.gpa) |g| { std.debug.assert(s.finished); g.in_deinit = true; } }
                else { p.box_ends += 1; if (p.gpa) |g| { std.debug.assert(g.live == 0 and g.allocated == g.freed); g.in_deinit = false; g.armed = false; } }
            },
            else => {},
        }
        O.send(p.upstream, event);
    }
    pub fn at(p: *const Probe, tag: std.meta.Tag(O.Event)) ?usize { return std.mem.indexOfScalar(std.meta.Tag(O.Event), p.tags[0..p.n], tag); }
    pub fn check(p: *const Probe) !void {
        try std.testing.expectEqual(p.starts, p.stops);
        try std.testing.expectEqual(p.boxes, p.box_ends);
        if (!p.enabled or !p.streamed) try std.testing.expectEqual(@as(usize, 0), p.session_events);
        if (p.at(.run_enter) != null) try std.testing.expect(p.at(.run_exit).? == p.n - 1);
        if (p.at(.request_end)) |end| {
            if (p.at(.box_deinit)) |box| try std.testing.expect(box < end);
            if (p.at(.generated)) |gen| try std.testing.expect(end < gen);
        }
        if (p.at(.drain_end)) |end| try std.testing.expect(end < p.at(.box_deinit).?);
        try std.testing.expect(!p.drain);
    }
};
/// Actual srv.gpa allocations, fixed ledger independent of the request arena.
/// Engine/request operations are sequential at the held barriers; no concurrent reader.
pub const Ledger = struct {
    const Block = struct { ptr: usize, len: usize, live: bool, frames: [12]usize, nframes: usize };
    const Freed = struct { ptr: usize, len: usize, ra: usize, frames: [12]usize, nframes: usize, alloc_frames: [12]usize, alloc_nframes: usize };
    backing: std.mem.Allocator,
    armed: bool = false,
    in_deinit: bool = false,
    blocks: [256]Block = undefined,
    n: usize = 0,
    live: usize = 0,
    allocated: usize = 0,
    freed: usize = 0,
    teardown_frees: usize = 0,
    frees: [256]Freed = undefined,
    pub fn printFrees(l: *const Ledger, case_id: usize) void {
        for (l.frees[0..l.teardown_frees], 0..) |f, index| {
            std.debug.print("qualification_498b8a41 case={d} order={d} address={x} bytes={d} ret_addr={x} frames=", .{case_id, index, f.ptr, f.len, f.ra});
            for (f.frames[0..f.nframes]) |frame| std.debug.print("{x},", .{frame});
            std.debug.print("\nM144I-HELD-SYMBOL-BEGIN case={d} order={d}\n", .{case_id, index});
            var frames = f.frames;
            std.debug.dumpStackTrace(&.{ .return_addresses = frames[0..f.nframes], .skipped = .unknown });
            std.debug.print("qualification_31cc33e5 case={d} order={d}\n", .{case_id, index});
            std.debug.print("qualification_f684547b case={d} order={d} address={x}\n", .{case_id, index, f.ptr});
            var alloc_frames = f.alloc_frames;
            std.debug.dumpStackTrace(&.{ .return_addresses = alloc_frames[0..f.alloc_nframes], .skipped = .unknown });
            std.debug.print("qualification_23d5e1eb case={d} order={d}\n", .{case_id, index});
        }
    }
    pub fn allocator(l: *Ledger) std.mem.Allocator { return .{ .ptr = l, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } }; }
    fn self(ctx: *anyopaque) *Ledger { return @ptrCast(@alignCast(ctx)); }
    fn add(l: *Ledger, ptr: usize, len: usize) void {
        if (!l.armed) return;
        std.debug.assert(l.n < l.blocks.len);
        l.blocks[l.n] = .{ .ptr = ptr, .len = len, .live = true, .frames = undefined, .nframes = 0 };
        l.blocks[l.n].nframes = std.debug.captureCurrentStackTrace(.{}, &l.blocks[l.n].frames).return_addresses.len;
        l.n += 1; l.live += 1; l.allocated += len;
    }
    fn block(l: *Ledger, ptr: usize) *Block {
        for (l.blocks[0..l.n]) |*b| if (b.ptr == ptr and b.live) return b;
        @panic("unowned/double mailbox free");
    }
    fn alloc(ctx: *anyopaque, len: usize, align_: std.mem.Alignment, ra: usize) ?[*]u8 {
        const l = self(ctx); const p = l.backing.rawAlloc(len, align_, ra) orelse return null; l.add(@intFromPtr(p), len); return p;
    }
    fn resize(ctx: *anyopaque, old: []u8, align_: std.mem.Alignment, len: usize, ra: usize) bool {
        const l = self(ctx); if (!l.backing.rawResize(old, align_, len, ra)) return false;
        if (l.armed) { const b = l.block(@intFromPtr(old.ptr)); l.allocated = l.allocated - b.len + len; b.len = len; } return true;
    }
    fn remap(ctx: *anyopaque, old: []u8, align_: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const l = self(ctx); const p = l.backing.rawRemap(old, align_, len, ra) orelse return null;
        if (l.armed) { const b = l.block(@intFromPtr(old.ptr)); l.allocated = l.allocated - b.len + len; b.ptr = @intFromPtr(p); b.len = len; } return p;
    }
    fn free(ctx: *anyopaque, bytes: []u8, align_: std.mem.Alignment, ra: usize) void {
        const l = self(ctx);
        if (l.armed) {
            const b = l.block(@intFromPtr(bytes.ptr)); std.debug.assert(b.len == bytes.len);
            b.live = false; l.live -= 1; l.freed += bytes.len;
            if (l.in_deinit) {
                std.debug.assert(l.teardown_frees < l.frees.len);
                const f = &l.frees[l.teardown_frees];
                f.* = .{ .ptr = @intFromPtr(bytes.ptr), .len = bytes.len, .ra = ra, .frames = undefined, .nframes = 0, .alloc_frames = b.frames, .alloc_nframes = b.nframes };
                f.nframes = std.debug.captureCurrentStackTrace(.{}, &f.frames).return_addresses.len;
                l.teardown_frees += 1;
            }
        }
        l.backing.rawFree(bytes, align_, ra);
    }
};
