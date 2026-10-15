//! shared_verify recorded real Users (TF_AFD_USERS_SHARED_DRAFTS=1; no CUDA numerics): a drafted shared round's windows through
//! the users' device half (cuda_afd_users_dsv.zig) on the recorder. Each window's per-sequence launches, draws and
//! settle exact against its stream's twin running that window alone; the row-by-row launches once over every row; one
//! exchange of all rows a MoE layer; two mis-split rounds the check must catch (RED); one-row windows' State launches =
//! shared_forward's shared round's; a tracked slot's MTP cache in step through draft requests, a drafted shared round with a
//! refused draft, settles and a lone resume. Checks after shared_forward's shared_forward_test.zig, the MTP rec after user_drafts's
//! user_drafts_test.zig (fixture scaffolding follows Ash Hart's cuda_afd_engine_test.zig; inherited NOTICE credits apply).
//! Design: @Bizuayeu's shared verify windows (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66,
//! Apache-2.0; reimplemented, not copied). Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const dsv = glm.afd_users_dsv;
const schedule = glm.afd_users_round;
const Row = schedule.Row;
const Key = schedule.Key;
const Window = dsv.Window;
const Space = glm.trace.Space;
const Event = glm.trace.Event;
const Val = glm.trace.Val;
const t = std.testing;
const gpa = t.allocator;
const io = t.io;
const eq = t.expectEqual;
const proto = cuda.xchg.proto;

/// The fixture's layers (m.cut(5)) and the slots' first positions (dense, the dense limit's last row, dense, sparse).
const L = 5;
const base = [_]usize{ 511, 2047, 513, 2048 };

/// Real Users over the recording Fwd and a fake exchange, with an MTP layer (user_drafts's MRec) and `n` slots;
/// `how`: shared_verify's switch (drafts + useSharedDrafts), shared_forward's (useShared), or shared_verify's from a prompt pass (positions 0).
pub const Rec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: glm.model.Model,
    w: glm.views.Weights,
    e: *glm.engine.Engine,
    a: glm.afd_attn.Afd,
    at: glm.afd_engine.Attn,
    space: Space,
    trace: glm.trace.Trace,
    f: glm.fwd.Fwd,
    users: glm.afd_users.Users,
    region: []align(4096) u8,
    status: []u32,
    node: cuda.xchg.Node,
    posts: usize = 0,
    hdrs: [512][2]i32 = undefined,
    headers: [512]proto.Header = undefined,

    const How = enum { drafted, shared, prompt };

    pub fn init(r: *Rec, n: u8, how: How) !void {
        return r.initLanes(n, how, 0);
    }

    pub fn initLanes(r: *Rec, n: u8, how: How, prompt_lanes: u8) !void {
        r.posts = 0;
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        errdefer r.parsed.deinit();
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        const mtp = r.m.mtp_layers;
        try r.m.cut(5);
        r.m.mtp_layers = if (mtp > 0) mtp else 1;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, 2560, 8, 64, how != .prompt);
        r.a = try glm.afd_attn.Afd.initWith(&r.m, 64, true);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.trace = glm.trace.Trace.init(gpa);
        r.a.lanes = prompt_lanes;
        r.a.paired = prompt_lanes != 0;
        r.a.inflight = if (prompt_lanes == 0) 1 else prompt_lanes;
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = if (prompt_lanes == 0) 8 else 64, .ports = 1, .pairs = prompt_lanes != 0, .lanes = prompt_lanes, .inflight = r.a.inflight };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(settings.region().total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        r.f = glm.fwd.Fwd.recording(&r.m, &r.trace);
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
        r.users = try glm.afd_users.Users.init(gpa, &r.at, &r.f, n);
        switch (how) {
            .drafted, .prompt => {
                try r.users.useDrafts(.{ .depth = .{ .most = 3, .fixed = true } }, glm.afd_users_drafts.rows);
                try r.users.useSharedDrafts();
            },
            .shared => try r.users.useShared(),
        }
        try r.users.walk(&r.space, glm.trace.placeAll); // the backlogs, then the stages: after every other buffer
        if (how == .prompt) return;
        for (0..n) |i| {
            r.users.generation[i] = 1;
            const st = stateOfSlot(r, i);
            st.pos = base[i];
            st.parity = @intCast(i % 2);
        }
    }

    pub fn deinit(r: *Rec) void {
        r.users.deinit(gpa);
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.trace.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }

    fn post(p: *anyopaque, _: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const r: *Rec = @ptrCast(@alignCast(p));
        const hdr = try proto.Header.read(h);
        if (hdr.layer != @as(i32, @intCast(pos)) or hdr.rows != @as(i32, @intCast(n))) return error.HeaderNotPos;
        if (r.posts < r.hdrs.len) {
            r.hdrs[r.posts] = .{ hdr.layer, hdr.rows };
            r.headers[r.posts] = hdr;
        }
        r.posts += 1;
    }

    fn take(_: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {}
};

fn stateOfSlot(r: *Rec, slot: usize) *glm.buffers.State {
    return if (slot == 0) &r.e.st else &r.users.states[slot - 1];
}

fn moeLayers(u: *glm.afd_users.Users) usize {
    var n: usize = 0;
    for (0..u.layerCount()) |li| n += @intFromBool(u.isMoe(li));
    return n;
}

const Range = struct {
    a: usize = 0,
    b: usize = 0,
    fn of(x: Range, ev: []const Event) []const Event {
        return ev[x.a..x.b];
    }
};

/// dsv.Dev under dsv.run with each call's launches marked; `mis` makes the two mis-split rounds the check must catch
/// (RED): a window's per-sequence part handed the next window's key (another stream's State), or the next row's rows.
const Watch = struct {
    d: dsv.Dev,
    trace: *glm.trace.Trace,
    mis: enum { none, key, rows } = .none,
    wins: []const Window = &.{},
    next: usize = 0,
    r_open: Range = .{},
    r_in: [L]Range = @splat(.{}),
    r_seq: [4][L]Range = @splat(@splat(.{})),
    r_out: [L]Range = @splat(.{}),
    r_tail: Range = .{},
    r_draw: [4]Range = @splat(.{}),
    r_settle: [4]Range = @splat(.{}),

    fn mark(w: *Watch) !usize {
        try w.d.u.f.flush();
        return w.trace.events.items.len;
    }
    fn index(w: *const Watch, slot: u8) usize {
        for (w.wins, 0..) |x, i| if (x.row.key.slot == slot) return i;
        unreachable;
    }
    pub fn alive(w: *Watch) !void {
        try w.d.alive();
    }
    pub fn layerCount(w: *const Watch) usize {
        return w.d.layerCount();
    }
    pub fn settle(w: *Watch) !void {
        try w.d.settle();
    }
    pub fn dsvBegin(w: *Watch, wins: []const Window, R: usize) !void {
        w.wins = wins;
        try w.d.dsvBegin(wins, R);
    }
    pub fn dsvOpen(w: *Watch, wins: []const Window, R: usize) !void {
        w.r_open.a = try w.mark();
        try w.d.dsvOpen(wins, R);
        w.r_open.b = try w.mark();
    }
    pub fn dsvIn(w: *Watch, wins: []const Window, R: usize, li: usize) !void {
        w.r_in[li].a = try w.mark();
        try w.d.dsvIn(wins, R, li);
        w.r_in[li].b = try w.mark();
        w.next = 0;
    }
    pub fn dsvSeq(w: *Watch, win: Window, r0: usize, li: usize) !void {
        const i = w.next;
        w.next += 1;
        w.r_seq[i][li].a = try w.mark();
        switch (w.mis) {
            .none => try w.d.dsvSeq(win, r0, li),
            .key => {
                var x = win;
                x.row.key = w.wins[(i + 1) % w.wins.len].row.key;
                try w.d.dsvSeq(x, r0, li);
            },
            .rows => try w.d.dsvSeq(win, r0 + 1, li),
        }
        w.r_seq[i][li].b = try w.mark();
    }
    pub fn dsvOut(w: *Watch, R: usize, li: usize) !void {
        w.r_out[li].a = try w.mark();
        try w.d.dsvOut(R, li);
        w.r_out[li].b = try w.mark();
    }
    pub fn dsvTail(w: *Watch, wins: []const Window, R: usize) !void {
        w.r_tail.a = try w.mark();
        try w.d.dsvTail(wins, R);
        w.r_tail.b = try w.mark();
    }
    pub fn dsvDraw(w: *Watch, win: Window, r0: usize, out: []u32) !void {
        const i = w.index(win.row.key.slot);
        w.r_draw[i].a = try w.mark();
        try w.d.dsvDraw(win, r0, out);
        w.r_draw[i].b = try w.mark();
    }
    pub fn dsvCommit(w: *Watch, row: Row, rows: usize, keep: usize) !void {
        const i = w.index(row.key.slot);
        w.r_settle[i].a = try w.mark();
        try w.d.dsvCommit(row, rows, keep);
        w.r_settle[i].b = try w.mark();
    }

    /// One drafted shared round through dsv.run, its trace from empty.
    fn round(w: *Watch, wins: []const Window, out: []u32) !void {
        try w.d.u.f.flush();
        w.trace.reset();
        try dsv.run(Watch, w, wins, out);
        try w.d.u.f.flush();
    }

    /// Window i's settle (dsv.settle: the commit of `keep` of its rows), marked.
    fn settleOf(w: *Watch, i: usize, keep: usize) !void {
        const win = w.wins[i];
        w.r_settle[i].a = try w.mark();
        try dsv.settle(w.d.u, win.row.key, win.rows(), keep);
        w.r_settle[i].b = try w.mark();
    }
};

// -- shared_forward's launch checks, for windows (rows r0 .. r0 + w of a together round against rows 0 .. w alone) ------------

const row_paths = [_][]const u8{ "b.lat", "b.ikr", "b.igr", "b.qi", "b.ka", "b.kg", "b.lat_s.qa", "b.lat_s.ol", "b.logits", "users.shared_proj", "users.shared_out" };

/// A Space naming rows [r0, r0 + w) of each row buffer as that buffer's rows from 0, then everything `s` names, a
/// temporary by its kind (shared_forward's rowNames at w rows).
fn winNames(s: *const Space, r0: usize, w: usize) !Space {
    var out: Space = .{ .gpa = gpa };
    errdefer out.deinit();
    for (s.entries.items) |e| for (row_paths) |p| if (std.mem.eql(u8, e.path, p)) {
        const rb = e.buf.inner() * e.buf.dtype.size();
        try out.entries.append(gpa, .{ .path = try gpa.dupe(u8, e.path), .base = e.base + r0 * rb, .len = w * rb, .buf = e.buf });
    };
    for (s.entries.items) |e| try out.add(tmpKind(e.path), e.buf);
    return out;
}

fn tmpKind(path: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, path, "tmp.")) return path;
    const Kind = struct {
        var buf: [192]u8 = undefined;
    };
    var len: usize = 0;
    var it = std.mem.splitScalar(u8, path, '.');
    while (it.next()) |seg| {
        if (seg.len > 0 and for (seg) |c| {
            if (!std.ascii.isDigit(c)) break false;
        } else true) continue;
        if (len > 0) {
            Kind.buf[len] = '.';
            len += 1;
        }
        @memcpy(Kind.buf[len..][0..seg.len], seg);
        len += seg.len;
    }
    return Kind.buf[0..len];
}

fn sameAddr(sa: *const Space, x: u64, sb: *const Space, y: u64) !void {
    const p = sa.name(x);
    const q = sb.name(y);
    try eq(p == null, q == null);
    if (p == null) return eq(x, y);
    try t.expectEqualStrings(p.?.path, q.?.path);
    try eq(p.?.off, q.?.off);
}

/// sameAddr for a row-by-row launch at `rows` (R against w): the AFD planes sit R·D apart, so their offsets scale.
fn sameRowAddr(sa: *const Space, x: u64, sb: *const Space, y: u64, rows: [2]usize) !void {
    if (sa.name(x)) |p| if (sb.name(y)) |q| if (std.mem.eql(u8, p.path, "b.afd_recv") and std.mem.eql(u8, q.path, "b.afd_recv"))
        return eq(p.off * rows[1], q.off * rows[0]);
    try sameAddr(sa, x, sb, y);
}

fn rowVal(sa: *const Space, u: Val, sb: *const Space, v: Val, rows: [2]usize) !void {
    try eq(std.meta.activeTag(u), std.meta.activeTag(v));
    switch (u) {
        .t => |x| {
            try sameRowAddr(sa, x.ptr, sb, v.t.ptr, rows);
            try eq(x.dtype, v.t.dtype);
            try eq(x.dims, v.t.dims);
            try eq(x.stride, v.t.stride);
            try t.expectEqualSlices(usize, x.shape[1..], v.t.shape[1..]);
        },
        .list => |l| {
            try eq(l.len, v.list.len);
            for (l, v.list) |p, q| try rowVal(sa, p, sb, q, rows);
        },
        .int => |x| if (x != rows[0] or v.int != rows[1]) try eq(x, v.int),
        else => try t.expectEqualDeep(u, v),
    }
}

/// The same row-by-row launches in the same order on the same buffers, the same constexprs (the same compiled variant)
/// and scalars beside the row count (R against w). Returns the launches compared.
fn expectRowOps(sa: *const Space, a: []const Event, sb: *const Space, b: []const Event, rows: [2]usize) !usize {
    if (a.len != b.len) {
        for (a, 0..) |x, k| std.debug.print("shared_verify together op {d}: {s}\n", .{ k, x.name });
        for (b, 0..) |x, k| std.debug.print("shared_verify alone op {d}: {s}\n", .{ k, x.name });
    }
    try eq(a.len, b.len);
    for (a, b, 0..) |x, y, k| {
        errdefer std.debug.print("shared_verify row op {d}: {s} against {s}\n", .{ k, x.name, y.name });
        try eq(x.op, y.op);
        try t.expectEqualStrings(x.name, y.name);
        try t.expectEqualDeep(x.consts, y.consts);
        try eq(x.args.len, y.args.len);
        for (x.args, y.args) |u, v| switch (u.value) {
            .ptr => |p| try sameRowAddr(sa, p.addr, sb, v.value.ptr.addr, rows),
            else => {},
        };
        try eq(x.vals.len, y.vals.len);
        for (x.vals, y.vals) |u, v| try rowVal(sa, u, sb, v, rows);
    }
    return a.len;
}

const Tally = struct { seq: usize = 0, row: usize = 0, draws: usize = 0, settles: usize = 0, extra: usize = 0 };

/// The together round's index-query projection (dsaIn's `sparse`: some window attends sparse) over every row, which a
/// dense window alone skips: the launches after the twin's in-half, each on b.qi.
fn sparseExtra(sa: *const Space, a: []const Event, from: usize) bool {
    if (a.len <= from) return false;
    for (a[from..]) |x| {
        var hit = false;
        for (x.args) |arg| switch (arg.value) {
            .ptr => |p| if (sa.name(p.addr)) |nm| {
                if (std.mem.eql(u8, nm.path, "b.qi")) hit = true;
            },
            else => {},
        };
        for (x.vals) |v| switch (v) {
            .t => |y| if (sa.name(y.ptr)) |nm| {
                if (std.mem.eql(u8, nm.path, "b.qi")) hit = true;
            },
            else => {},
        };
        if (!hit) return false;
    }
    return true;
}

/// One round's check: window i of `a`'s together round against twins[i]'s round of window i alone: every per-sequence
/// launch, draw and settle exact (State by name and offset, rows r0 .. r0 + w mapped to 0 .. w, grids, constexprs),
/// every row-by-row launch the twin's once (R against w).
fn checkRound(a: *Rec, wa: *const Watch, twins: []const *Rec, wt: []const Watch, tally: *Tally) !void {
    const ea = a.trace.events.items;
    const R = try dsv.total(wa.wins);
    var r0: usize = 0;
    for (twins, wt, 0..) |b, *w, i| {
        defer r0 += wa.wins[i].rows();
        const width = wa.wins[i].rows();
        const eb = b.trace.events.items;
        var na = try winNames(&a.space, r0, width);
        defer na.deinit();
        var nb = try winNames(&b.space, 0, width);
        defer nb.deinit();
        const rows = [2]usize{ R, width };
        for (0..L) |li| {
            errdefer std.debug.print("shared_verify window {d} ({d} rows at row {d}) layer {d}\n", .{ i, width, r0, li });
            try glm.serve_rec.expectSame(&na, wa.r_seq[i][li].of(ea), &nb, w.r_seq[0][li].of(eb));
            tally.seq += wa.r_seq[i][li].b - wa.r_seq[i][li].a;
            const ra = wa.r_in[li].of(ea);
            const rb = w.r_in[li].of(eb);
            const cut = if (ra.len > rb.len and sparseExtra(&a.space, ra, rb.len)) rb.len else ra.len;
            tally.extra += ra.len - cut;
            tally.row += try expectRowOps(&a.space, ra[0..cut], &b.space, rb, rows);
            tally.row += try expectRowOps(&a.space, wa.r_out[li].of(ea), &b.space, w.r_out[li].of(eb), rows);
        }
        tally.row += try expectRowOps(&a.space, wa.r_tail.of(ea), &b.space, w.r_tail.of(eb), rows);
        try glm.serve_rec.expectSame(&na, wa.r_draw[i].of(ea), &nb, w.r_draw[0].of(eb));
        tally.draws += wa.r_draw[i].b - wa.r_draw[i].a;
        try eq(wa.r_open.b - wa.r_open.a, w.r_open.b - w.r_open.a);
        for (wa.r_open.of(ea), w.r_open.of(eb)) |x, y| try t.expectEqualStrings(x.name, y.name);
    }
}

/// Each window's settle in `a` and its twin, with `keeps`: the commit's launches exact (State by name and offset).
fn checkSettles(a: *Rec, wa: *Watch, twins: []const *Rec, wt: []Watch, keeps: []const usize, tally: *Tally) !void {
    for (twins, wt, keeps, 0..) |b, *w, keep, i| {
        if (wa.wins[i].rows() == 1) { // committed in the round
            try glm.serve_rec.expectSame(&a.space, wa.r_settle[i].of(a.trace.events.items), &b.space, w.r_settle[0].of(b.trace.events.items));
            continue;
        }
        const la = a.trace.events.items.len;
        try wa.settleOf(i, keep);
        const lb = b.trace.events.items.len;
        try w.settleOf(0, keep);
        try glm.serve_rec.expectSame(&a.space, a.trace.events.items[la..], &b.space, b.trace.events.items[lb..]);
        tally.settles += a.trace.events.items.len - la;
    }
}

const Shape = struct { widths: []const usize, keeps: []const usize };

/// `n` streams' drafted windows together against each window alone, `shapes.len` rounds; every window's settle too.
fn together(n: u8, shapes: []const Shape) !void {
    var a: Rec = undefined;
    try a.init(n, .drafted);
    defer a.deinit();
    var twins: [4]Rec = undefined;
    var made: usize = 0;
    defer for (twins[0..made]) |*x| x.deinit();
    for (twins[0..n]) |*x| {
        try x.init(n, .drafted);
        made += 1;
    }
    var tp: [4]*Rec = undefined;
    for (twins[0..n], tp[0..n]) |*x, *p| p.* = x;
    const moe = moeLayers(&a.users);
    var tally: Tally = .{};
    var drafts: [4][dsv.max_drafts]u32 = undefined;
    for (&drafts, 0..) |*d, i| for (d, 0..) |*x, j| {
        x.* = @intCast(20 + 4 * i + j);
    };
    for (shapes, 0..) |shape, k| {
        var wins: [4]Window = undefined;
        for (0..n) |i| {
            const key: Key = .{ .slot = @intCast(i), .generation = 1 };
            wins[i] = .{ .row = .{ .key = key, .token = @intCast(10 + i + k), .position = a.users.position(key) }, .drafts = drafts[i][0 .. shape.widths[i] - 1] };
        }
        const R = try dsv.total(wins[0..n]);
        var wa: Watch = .{ .d = .{ .u = &a.users }, .trace = &a.trace };
        var wt: [4]Watch = undefined;
        for (twins[0..n], wt[0..n]) |*b, *w| w.* = .{ .d = .{ .u = &b.users }, .trace = &b.trace };
        var out: [dsv.max_rows]u32 = undefined;
        const posts = a.posts;
        try wa.round(wins[0..n], out[0..R]);
        try eq(moe, a.posts - posts); // one exchange a MoE layer, all R rows in it
        for (a.hdrs[posts..a.posts]) |h| try eq(@as(i32, @intCast(R)), h[1]);
        for (0..n) |i| {
            var o: [dsv.max_width]u32 = undefined;
            const tposts = twins[i].posts;
            try wt[i].round(wins[i .. i + 1], o[0..wins[i].rows()]);
            for (twins[i].hdrs[tposts..twins[i].posts]) |h| try eq(@as(i32, @intCast(wins[i].rows())), h[1]);
        }
        try checkRound(&a, &wa, tp[0..n], wt[0..n], &tally);
        try checkSettles(&a, &wa, tp[0..n], wt[0..n], shape.keeps, &tally);
        for (0..n) |i| {
            const key: Key = wins[i].row.key;
            try eq(wins[i].row.position + shape.keeps[i], a.users.position(key));
            try eq(twins[i].users.position(key), a.users.position(key));
            try eq(stateOfSlot(&twins[i], i).parity, stateOfSlot(&a, i).parity);
        }
    }
    std.debug.print("shared_verify recorded C{d}: {d} drafted shared rounds; each window against its stream's twin alone: {d} per-sequence launches exact, {d} draw launches exact, {d} settle launches exact (rolled-back windows replayed on their own State), {d} row-by-row launch pairs equal (once a round, constexprs equal, R against w; {d} index-query launches a sparse window adds over every row); one exchange a MoE layer of all rows; no hardware numerics\n", .{ n, shapes.len, tally.seq, tally.draws, tally.settles, tally.row, tally.extra });
}

test "shared_verify recorded C2 (windows 4 + 3, then 2 + 4): each window exact against its stream alone, settles with a rollback, row ops once, one exchange of all rows" {
    try together(2, &.{ .{ .widths = &.{ 4, 3 }, .keeps = &.{ 4, 2 } }, .{ .widths = &.{ 2, 4 }, .keeps = &.{ 1, 3 } } });
}

test "shared_verify recorded C4 (windows 4 + 2 + 1 + 1 = 8 rows; dense, across the dense limit, dense, sparse), then 1 + 4 + 2 + 1" {
    try together(4, &.{ .{ .widths = &.{ 4, 2, 1, 1 }, .keeps = &.{ 3, 2, 1, 1 } }, .{ .widths = &.{ 1, 4, 2, 1 }, .keeps = &.{ 1, 4, 1, 1 } } });
}

fn misSplit(how: @FieldType(Watch, "mis")) !void {
    var a: Rec = undefined;
    try a.init(2, .drafted);
    defer a.deinit();
    var b: [2]Rec = undefined;
    try b[0].init(2, .drafted);
    defer b[0].deinit();
    try b[1].init(2, .drafted);
    defer b[1].deinit();
    const d = [_]u32{ 21, 22, 23 };
    const wins = [_]Window{ .{ .row = .{ .key = .{ .slot = 0, .generation = 1 }, .token = 10, .position = base[0] }, .drafts = &d }, .{ .row = .{ .key = .{ .slot = 1, .generation = 1 }, .token = 11, .position = base[1] }, .drafts = d[0..2] } };
    var wa: Watch = .{ .d = .{ .u = &a.users }, .trace = &a.trace, .mis = how };
    var wt: [2]Watch = undefined;
    for (&b, &wt) |*x, *w| w.* = .{ .d = .{ .u = &x.users }, .trace = &x.trace };
    var out: [7]u32 = undefined;
    try wa.round(&wins, &out);
    for (0..2) |i| try wt[i].round(wins[i .. i + 1], out[0..wins[i].rows()]);
    var tally: Tally = .{};
    const tp = [_]*Rec{ &b[0], &b[1] };
    std.debug.print("shared_verify RED mis-split ({t}): the check below must fail\n", .{how});
    if (checkRound(&a, &wa, &tp, &wt, &tally)) |_| return error.MisSplitNotCaught else |e| std.debug.print("shared_verify RED mis-split ({t}) caught: {t}\n", .{ how, e });
}

test "shared_verify RED: a window's per-sequence part on another stream's State fails the twin check" {
    try misSplit(.key);
}

test "shared_verify RED: a window's per-sequence part on the next row's rows fails the twin check" {
    try misSplit(.rows);
}

// -- one-row windows against shared_forward's shared round ------------------------------------------------------------------

/// State buffers (slot 0's "st.", another slot's "users.stateN."), but not st.proj (written per row by the q4, copied
/// from the stage when shared) and not the KDA chain's output scratch.
fn deepState(s: *const Space, addr: u64) bool {
    const x = s.name(addr) orelse return false;
    if (!std.mem.startsWith(u8, x.path, "st.") and !std.mem.startsWith(u8, x.path, "users.state")) return false;
    return !std.mem.endsWith(u8, x.path, ".scratch_set.out") and !std.mem.endsWith(u8, x.path, ".proj");
}

fn valTouches(s: *const Space, v: Val) bool {
    return switch (v) {
        .t => |x| deepState(s, x.ptr),
        .list => |l| for (l) |u| {
            if (valTouches(s, u)) break true;
        } else false,
        else => false,
    };
}

fn perSequence(s: *const Space, e: Event) bool {
    for (e.args) |a| switch (a.value) {
        .ptr => |p| if (deepState(s, p.addr)) return true,
        else => {},
    };
    for (e.vals) |v| if (valTouches(s, v)) return true;
    return false;
}

/// Every State-touching launch of `a` and `b`, in order: the same kernel, grid and constexprs, each operand the same
/// State buffer by name and offset. Returns the launches compared.
fn expectStateOps(sa: *const Space, a: []const Event, sb: *const Space, b: []const Event) !usize {
    var ia: usize = 0;
    var ib: usize = 0;
    var n: usize = 0;
    while (true) : ({
        ia += 1;
        ib += 1;
        n += 1;
    }) {
        while (ia < a.len and !perSequence(sa, a[ia])) ia += 1;
        while (ib < b.len and !perSequence(sb, b[ib])) ib += 1;
        if (ia == a.len or ib == b.len) break;
        const x = a[ia];
        const y = b[ib];
        errdefer std.debug.print("shared_verify State launch {d}: {s} against {s}\n", .{ n, x.name, y.name });
        try eq(x.op, y.op);
        try t.expectEqualStrings(x.name, y.name);
        try eq(x.grid, y.grid);
        try t.expectEqualDeep(x.consts, y.consts);
        try eq(x.args.len, y.args.len);
        for (x.args, y.args) |u, v| switch (u.value) {
            .ptr => |p| if (deepState(sa, p.addr) or deepState(sb, v.value.ptr.addr)) try sameAddr(sa, p.addr, sb, v.value.ptr.addr),
            else => {},
        };
    }
    try eq(a.len, ia);
    try eq(b.len, ib);
    return n;
}

test "shared_verify one-row windows: the State launches and positions = shared_forward's shared round's, C2 and C4, 2 rounds" {
    var compared: usize = 0;
    for ([_]u8{ 2, 4 }) |n| {
        var a: Rec = undefined;
        try a.init(n, .drafted);
        defer a.deinit();
        var b: Rec = undefined;
        try b.init(n, .shared);
        defer b.deinit();
        for (0..2) |k| {
            var wins: [4]Window = undefined;
            var rows: [4]Row = undefined;
            for (0..n) |i| {
                const key: Key = .{ .slot = @intCast(i), .generation = 1 };
                rows[i] = .{ .key = key, .token = @intCast(10 + i + k), .position = a.users.position(key) };
                wins[i] = .{ .row = rows[i] };
            }
            var oa: [4]u32 = undefined;
            var ob: [4]u32 = undefined;
            try a.f.flush();
            a.trace.reset();
            try b.f.flush();
            b.trace.reset();
            try a.users.roundDrafted(wins[0..n], oa[0..n]);
            try b.users.round(rows[0..n], ob[0..n]);
            try a.f.flush();
            try b.f.flush();
            errdefer std.debug.print("shared_verify one-row C{d} round {d}\n", .{ n, k });
            compared += try expectStateOps(&a.space, a.trace.events.items, &b.space, b.trace.events.items);
            for (0..n) |i| {
                try eq(b.users.position(rows[i].key), a.users.position(rows[i].key));
                try eq(base[i] + k + 1, a.users.position(rows[i].key));
                try eq(stateOfSlot(&b, i).parity, stateOfSlot(&a, i).parity);
            }
        }
    }
    std.debug.print("shared_verify one-row windows = shared_forward's shared round: {d} State launches exact over C2 and C4, 2 rounds each\n", .{compared});
}

// -- a tracked slot's MTP cache through draft requests, a drafted shared round, settles and a lone resume ------------

/// Every draw answered with token 5 at probability 1 (user_drafts's Picks).
const Picks = struct {
    n: usize = 0,
    fn pick(ctx: *anyopaque, logits: glm.trace.T, at: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        _ = .{ logits, at };
        const p: *Picks = @ptrCast(@alignCast(ctx));
        @memset(out, 5);
        if (probs) |x| @memset(x, 1.0);
        p.n += 1;
    }
};

const UHook = glm.afd_users.Users.DraftHook;

const Never = struct {
    fn kept(_: *anyopaque, _: []const u32, _: usize, _: usize) anyerror!bool {
        return false;
    }
};

/// The rows the slot's MTP head lacks: its State's rows less what its cache absorbed (held draft rows aside).
fn lacking(st: anytype) usize {
    return st.pos - (st.mtp_len - st.mtp_drafted);
}

test "shared_verify recorded MTP: draft requests, a drafted shared round with a refused draft, settles and a lone resume keep each tracked slot's MTP cache in step" {
    var r: Rec = undefined;
    try r.init(2, .prompt);
    defer r.deinit();
    var picks: Picks = .{};
    r.trace.picker = .{ .ctx = &picks, .pick = Picks.pick };
    const u = &r.users;
    var ids: [10]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    const k0: Key = .{ .slot = 0, .generation = 1 };
    const k1: Key = .{ .slot = 1, .generation = 1 };
    const st0 = &r.e.st;
    const st1 = &u.states[0];
    u.track(0, .seed);
    u.track(1, .seed);
    const p0 = try u.prefill(k0, &ids);
    const p1 = try u.prefill(k1, &ids);
    for ([_]*glm.buffers.State{ st0, st1 }, [_]Key{ k0, k1 }) |st, k| {
        try eq(@as(usize, 10), st.pos);
        try eq(@as(usize, 9), st.mtp_len); // the prompt absorbed but its last row
        try eq(@as(usize, 1), u.backlogLen(k)); // that row and the first token: the start pair
    }
    // the unverified requests: each start pair absorbed, the drafts chained after it
    var d0: [3]u32 = undefined;
    var d1: [3]u32 = undefined;
    try u.draftDrafted(k0, &.{p0}, false, st0.pos + 1, &d0);
    try u.draftDrafted(k1, &.{p1}, false, st1.pos + 1, d1[0..2]);
    try t.expectEqualSlices(u32, &.{ 5, 5, 5 }, &d0);
    try eq(@as(usize, 0), lacking(st0));
    try eq(@as(usize, 0), lacking(st1));
    try eq(@as(usize, 2), st0.mtp_drafted); // the drafts the chain fed back (its last draft is never absorbed)
    try eq(@as(usize, 1), st1.mtp_drafted);
    // a stale request refused before any launch: the start pair already in head
    const posts = r.posts;
    try t.expectError(error.DraftOutOfStep, u.draftDrafted(k0, &.{p0}, false, st0.pos + 1, &d0));
    try eq(posts, r.posts);
    // one drafted shared round of 4 + 3 rows: slot 0's drafts all drawn again, slot 1's second refused
    d1[1] = 7;
    const wins = [_]Window{ .{ .row = .{ .key = k0, .token = p0, .position = st0.pos }, .drafts = &d0 }, .{ .row = .{ .key = k1, .token = p1, .position = st1.pos }, .drafts = d1[0..2] } };
    var out: [7]u32 = undefined;
    const before = r.posts;
    try u.roundDrafted(&wins, &out);
    try eq(moeLayers(u), r.posts - before);
    for (r.hdrs[before..r.posts]) |h| try eq(@as(i32, 7), h[1]);
    try t.expectEqualSlices(u32, &.{ 5, 5, 5, 5, 5, 5, 5 }, &out);
    try eq(@as(usize, 10), st0.pos); // a wider window's rows wait for its settle
    try u.settleDrafted(k0, 4, 4);
    try u.settleDrafted(k1, 3, 2); // its third row rolled back on its own State
    try eq(@as(usize, 14), st0.pos);
    try eq(@as(usize, 12), st1.pos);
    try t.expectError(error.KeepOutOfStep, u.settleDrafted(k1, 3, 2)); // settled once
    // the verified requests: the kept rows absorbed with their next tokens, the new start pair in head
    try u.draftDrafted(k0, out[0..4], true, st0.pos + 1, &d0);
    try u.draftDrafted(k1, out[4..6], true, st1.pos + 1, &d1);
    try eq(@as(usize, 0), lacking(st0));
    try eq(@as(usize, 0), lacking(st1));
    try eq(@as(usize, 1), u.backlogLen(k0));
    try eq(@as(usize, 1), u.backlogLen(k1));
    const c = u.dsvs.?.counts;
    try eq(@as(u64, 4), c.draft_requests);
    try eq(@as(u64, 1 + 1 + 4 + 2), c.absorbed);
    try eq(@as(u64, 1), c.replays);
    // a lone resume on slot 1: its start pair un-absorbed (with its chained drafts), user_drafts's loop from there
    const hook: UHook = .{ .ctx = &picks, .kept = Never.kept };
    var lone: [5]u32 = undefined;
    const ran = try u.drafted(k1, out[5], 5, &lone, &hook);
    try eq(@as(usize, 5), ran.tokens);
    try eq(@as(u64, 1), u.dsvs.?.counts.unabsorbed);
    try eq(@as(usize, 16), st1.pos);
    try u.carry(k1, lone[1..5]); // its last round's kept rows carried, as at a hand-over: the head lacks exactly those
    try eq(lacking(st1), u.backlogLen(k1));
    u.release(k0);
    u.release(k1);
    try eq(@as(usize, 0), u.backlogLen(k1));
    std.debug.print("shared_verify recorded MTP: 2 tracked slots; 4 draft requests ({d} pairs absorbed), a 7-row drafted shared round (one exchange of 7 rows a MoE layer), 1 rolled back, rows lacking 0 after each request; a lone resume un-absorbed its start pair and ran 5 tokens\n", .{c.absorbed});
}
