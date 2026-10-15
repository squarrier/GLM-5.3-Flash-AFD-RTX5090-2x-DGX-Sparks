//! grammar_tests_c CPU-only scheduler agreements; Ash Hart/Hugh Madden's ordinary Pairs order stays authoritative.
const std = @import("std");
const c = @import("lanes").structured_control;

pub const Gate = struct {
    permit: ?c.Permit = null,
    descriptor: ?c.Descriptor = null,
    index: u32 = 0,
    pub fn finish(g: *Gate, endpoint: ?c.Endpoint) !void {
        const p = endpoint orelse return;
        const permit = g.permit orelse return;
        const d = g.descriptor.?;
        g.permit = null;
        const done = try p.after(permit, .{ .logits_valid = true,
            .issued = @intFromBool(d.action == .request), .taken = @intFromBool(d.action == .take or d.action == .drain) });
        try status(done.status);
    }
    pub fn before(g: *Gate, endpoint: ?c.Endpoint, descriptor: c.Descriptor) !void {
        const p = endpoint orelse return;
        try g.finish(endpoint);
        var d = descriptor;
        d.action_index = g.index;
        d.outstanding = p.session.outstanding[p.peer];
        g.permit = try p.before(d);
        g.descriptor = d;
        g.index += 1;
    }
};
fn status(value: c.Status) !void {
    switch (value) {
        .ok => {}, .cancelled => return error.Cancelled, .empty_mask => return error.EmptyMask,
        .request_error => return error.StructuredRequestError, .poison => return error.ControlPoison,
    }
}
pub fn reset(x: anytype, f: anytype, ids: []const u32) !void {
    const p = x.structured_control orelse return;
    try p.open();
    const permit = try p.before(.{ .kind = .reset, .epoch = p.session.epoch,
        .peer_count = p.session.peer_count, .total_prompt_rows = ids.len });
    @import("cuda_compute.zig").reset(f, &x.e.st) catch |err| return x.committedFault(err);
    try status((try p.after(permit, .{})).status);
}
fn groupIndex(r: anytype) u32 {
    const count: usize = if (@hasField(@TypeOf(r.*), "l")) r.l.n else 2;
    return @intCast(r.start / (count*r.x.e.prefill_rows));
}
pub fn constituent(r: anytype, slot: u32, start: usize, rows: usize) !void {
    const p = r.x.structured_control orelse return;
    const permit = try p.before(.{ .kind = .prompt_chunk, .epoch = p.session.epoch,
        .peer_count = p.session.peer_count, .group_index = groupIndex(r),
        .slot = slot, .chunk_index = @intCast(start / r.x.e.prefill_rows), .position = start,
        .rows = @intCast(rows), .tokens = r.ids[start..][0..rows], .total_prompt_rows = r.ids.len });
    try status((try p.after(permit, .{})).status);
}
pub fn pair(r: anytype, ra: usize, rb: usize) !@import("cuda_trace.zig").T {
    const endpoint = r.x.structured_control;
    if (endpoint == null) return r.pairRun(ra, rb);
    try constituent(r, 0, r.start, ra);
    try constituent(r, 1, r.start+ra, rb);
    const p = endpoint.?;
    const permit = try p.before(.{ .kind = .group, .epoch = p.session.epoch,
        .peer_count = p.session.peer_count, .group_index = groupIndex(r),
        .position = r.start, .rows = @intCast(ra+rb), .total_prompt_rows = r.ids.len });
    try status((try p.after(permit, .{})).status);
    const result = r.pairRun(ra, rb) catch |err| return abortPair(r, err);
    r.gate.finish(endpoint) catch |err| return abortPair(r, err);
    return result;
}
pub fn action(r: anytype, who: anytype, step: anytype, pos: usize) !void {
    const p = r.x.structured_control orelse return;
    const k = if (who == .b or who == .pair) &r.cb else &r.ca;
    const op: c.Action = switch (step) {
        .stage, .position => .stage, .front => .advance, .route => .route, .request => .request,
        .take => .take, .post => .post, .tail => .tail, .absorb => .absorb, .commit => .input_commit,
        .first => .finish,
    };
    try r.gate.before(p, .{ .kind = .group_action, .epoch = p.session.epoch, .peer_count = p.session.peer_count,
        .group_index = groupIndex(r), .slot = k.w.pair.slot,
        .chunk_index = @intCast(k.start / r.x.e.prefill_rows), .position = k.start,
        .rows = @intCast(k.R), .tokens = if (op == .stage) r.ids[k.start..][0..k.R] else &.{},
        .layer = @intCast(if (op == .advance) pos else if (k.li > 0) k.li-1 else 0),
        .provider_position = @intCast(if (op == .request or op == .take) pos else k.w.pos),
        .transport_seq = k.w.seq, .total_prompt_rows = r.ids.len, .action = op });
}
pub fn group(r: anytype, rows: []const usize) !@import("cuda_trace.zig").T {
    const endpoint = r.x.structured_control;
    if (endpoint == null) return r.groupRun(rows);
    var start = r.start;
    var total: usize = 0;
    for (rows, 0..) |n, slot| {
        try constituent(r, @intCast(slot), start, n);
        start += n; total += n;
    }
    const p = endpoint.?;
    const permit = try p.before(.{ .kind = .group, .epoch = p.session.epoch,
        .peer_count = p.session.peer_count, .group_index = groupIndex(r),
        .position = r.start, .rows = @intCast(total), .total_prompt_rows = r.ids.len });
    try status((try p.after(permit, .{})).status);
    const result = r.groupRun(rows) catch |err| return abortGroup(r, err);
    r.gate.finish(endpoint) catch |err| return abortGroup(r, err);
    return result;
}
pub fn laneAction(r: anytype, slot: u8, step: anytype, pos: usize) !void {
    const p = r.x.structured_control orelse return;
    const k = &r.ch[slot];
    const op: c.Action = switch (step) {
        .stage, .position => .stage, .front => .advance, .route => .route, .request => .request,
        .take => .take, .post => .post, .tail => .tail, .absorb => .absorb, .commit => .input_commit,
        .first => .finish,
    };
    try r.gate.before(p, .{ .kind = .group_action, .epoch = p.session.epoch, .peer_count = p.session.peer_count,
        .group_index = groupIndex(r), .slot = k.k, .chunk_index = @intCast(k.start / r.x.e.prefill_rows),
        .position = k.start, .rows = @intCast(k.R),
        .tokens = if (op == .stage) r.ids[k.start..][0..k.R] else &.{},
        .layer = @intCast(if (op == .advance) pos else if (k.li > 0) k.li-1 else 0),
        .provider_position = @intCast(if (op == .request or op == .take) pos else k.w.pos),
        .transport_seq = k.w.seq, .total_prompt_rows = r.ids.len, .action = op });
}
fn abortGroup(r: anytype, original: anyerror) anyerror {
    const p = r.x.structured_control.?;
    if (r.gate.permit != null or p.session.pending == .ok or p.session.pending == .poison) {
        _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison;
    }
    while (r.turn.oldest()) |slot| {
        r.take(&r.ch[slot]) catch { _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison; };
        r.gate.finish(p) catch |err| {
            if (err != error.Cancelled and err != error.EmptyMask and err != error.StructuredRequestError) {
                _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison;
            }
        };
    }
    for (p.session.outstanding[0..p.session.peer_count]) |count| if (count != 0) {
        _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison;
    };
    var end = r.ch[r.n-1].w;
    end.pos = end.first+end.count;
    r.restore(end);
    r.begun = false; r.l.last = null; r.x.pending = false;
    status(p.session.status) catch |err| return err;
    return original;
}
fn abortPair(r: anytype, original: anyerror) anyerror {
    const p = r.x.structured_control.?;
    if (r.gate.permit != null or p.session.pending == .ok or p.session.pending == .poison) {
        _ = p.quiesce(.poison) catch {};
        r.stopOrdinary();
        return error.ControlPoison;
    }
    while (r.turn.taken < r.turn.next) {
        const k = if (r.turn.taken % 2 == 0) &r.ca else &r.cb;
        r.take(k) catch { _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison; };
        r.gate.finish(p) catch |err| {
            if (err != error.Cancelled and err != error.EmptyMask and err != error.StructuredRequestError) {
                _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison;
            }
        };
    }
    if (p.session.outstanding[0] != 0 or p.session.outstanding[1] != 0 or p.session.outstanding[2] != 0) {
        _ = p.quiesce(.poison) catch {}; r.stopOrdinary(); return error.ControlPoison;
    }
    var end = r.cb.w;
    end.pos = end.first+end.count;
    r.restore(end);
    r.begun = false; r.p.last = null; r.x.pending = false;
    status(p.session.status) catch |err| return err;
    return original;
}
