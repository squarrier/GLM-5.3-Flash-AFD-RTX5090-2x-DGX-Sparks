//! shared_forward (local integration; tool_api_fixes § 3 L1, stage 1): TF_AFD_USERS_SHARED=1's CPU tests (DESIGN.md § 7). The switch through
//! plan(); the shared round recorded through the real users backend (afd_users_device_test.zig's Rec: real Users,
//! recording Fwd, fake exchange) against each stream alone, launch for launch, with two mis-split rounds that must fail
//! the same check; the shared round's State launches against today's per-row path; the real round scheduler over a
//! numeric stand-in; cancel and failure; the stages' bytes; default off. CPU only: no hardware numerics.
//! Design after @Bizuayeu's shared verify windows (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66,
//! Apache-2.0; reimplemented, not copied); fixture scaffolding follows Ash Hart's cuda_afd_engine_test.zig (inherited
//! NOTICE credits apply). Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
//! draft_shared (local integration): the switch's and the bytes' refusal asserts now assert the composition with user_drafts's drafts.
const std = @import("std");
const glm = @import("glm_cuda");
const dev = @import("afd_users_device_test.zig");
const schedule = glm.afd_users_round;
const Row = schedule.Row;
const Key = schedule.Key;
const Space = glm.trace.Space;
const Event = glm.trace.Event;
const Val = glm.trace.Val;
const t = std.testing;
const gpa = t.allocator;
const eq = t.expectEqual;

/// dev.Rec's layers (its m.cut(5)) and its slots' first positions (dense, the dense limit's last row, dense, sparse).
const L = 5;
const base = [_]usize{ 511, 2047, 513, 2048 };

/// Slot `slot`'s row in round `k` (every round commits one token a row).
fn rowAt(slot: u8, k: usize) Row {
    return .{ .key = .{ .slot = slot, .generation = 1 }, .token = @intCast(10 + slot + k), .position = base[slot] + k };
}

/// dev.Rec with the switch on: the stages made before any round and placed after every other buffer (Users.walk's
/// order, as attach lays them out).
fn shareOn(r: *dev.Rec) !void {
    try r.users.useShared();
    const s = &r.users.stages.?;
    try r.space.place("users.shared_proj", &s.proj);
    try r.space.place("users.shared_out", &s.out);
}

fn stateOfSlot(r: *dev.Rec, slot: usize) *glm.buffers.State {
    return if (slot == 0) &r.e.st else &r.users.states[slot - 1];
}

const Range = struct {
    a: usize = 0,
    b: usize = 0,
    fn of(x: Range, ev: []const Event) []const Event {
        return ev[x.a..x.b];
    }
};

/// The real device methods under the real scheduler (schedule.run), each call's launches marked; `mis` makes the
/// two mis-split rounds the check must catch (RED): a row's per-sequence part handed the next row's key (another
/// stream's State), or run on the next row's buffers.
const Watch = struct {
    u: *glm.afd_users.Users,
    trace: *glm.trace.Trace,
    rs: []const Row = &.{},
    mis: enum { none, key, row } = .none,
    r_open: Range = .{},
    r_in: [L]Range = @splat(.{}),
    r_seq: [4][L]Range = @splat(@splat(.{})),
    r_out: [L]Range = @splat(.{}),
    r_tail: Range = .{},
    r_sample: [4]Range = @splat(.{}),
    r_commit: [4]Range = @splat(.{}),

    fn mark(w: *Watch) !usize {
        try w.u.f.flush();
        return w.trace.events.items.len;
    }
    pub fn alive(w: *Watch) !void {
        try w.u.alive();
    }
    pub fn begin(w: *Watch, rs: []const Row) !void {
        w.rs = rs;
        try w.u.begin(rs);
    }
    pub fn layerCount(w: *Watch) usize {
        return w.u.layerCount();
    }
    pub fn isMoe(w: *Watch, li: usize) bool {
        return w.u.isMoe(li);
    }
    // today's per-row forward (the switch off): delegated
    pub fn open(w: *Watch, r: Row, i: usize) !void {
        try w.u.open(r, i);
    }
    pub fn front(w: *Watch, r: Row, i: usize, li: usize) !void {
        try w.u.front(r, i, li);
    }
    pub fn gather(w: *Watch, r: Row, i: usize) !void {
        try w.u.gather(r, i);
    }
    pub fn exchange(w: *Watch, li: usize, n: usize) !void {
        try w.u.exchange(li, n);
    }
    pub fn scatter(w: *Watch, r: Row, i: usize, n: usize) !void {
        try w.u.scatter(r, i, n);
    }
    pub fn tail(w: *Watch, r: Row, i: usize) !void {
        try w.u.tail(r, i);
    }
    // shared_forward's forward
    pub fn shared(w: *Watch) bool {
        return w.u.shared();
    }
    pub fn sharedOpen(w: *Watch, rs: []const Row) !void {
        w.r_open.a = try w.mark();
        try w.u.sharedOpen(rs);
        w.r_open.b = try w.mark();
    }
    pub fn sharedIn(w: *Watch, rs: []const Row, li: usize) !void {
        w.r_in[li].a = try w.mark();
        try w.u.sharedIn(rs, li);
        w.r_in[li].b = try w.mark();
    }
    pub fn sharedSeq(w: *Watch, row: Row, i: usize, li: usize) !void {
        const n = w.rs.len;
        w.r_seq[row.key.slot][li].a = try w.mark();
        switch (w.mis) {
            .none => try w.u.sharedSeq(row, i, li),
            .key => try w.u.sharedSeq(w.rs[(i + 1) % n], i, li),
            .row => try w.u.sharedSeq(row, (i + 1) % n, li),
        }
        w.r_seq[row.key.slot][li].b = try w.mark();
    }
    pub fn sharedOut(w: *Watch, rs: []const Row, li: usize) !void {
        w.r_out[li].a = try w.mark();
        try w.u.sharedOut(rs, li);
        w.r_out[li].b = try w.mark();
    }
    pub fn sharedTail(w: *Watch, rs: []const Row) !void {
        w.r_tail.a = try w.mark();
        try w.u.sharedTail(rs);
        w.r_tail.b = try w.mark();
    }
    pub fn settle(w: *Watch) !void {
        try w.u.settle();
    }
    pub fn sample(w: *Watch, r: Row, i: usize) !u32 {
        w.r_sample[r.key.slot].a = try w.mark();
        const token = try w.u.sample(r, i);
        w.r_sample[r.key.slot].b = try w.mark();
        return token;
    }
    pub fn commit(w: *Watch, r: Row, i: usize) !void {
        w.r_commit[r.key.slot].a = try w.mark();
        try w.u.commit(r, i);
        w.r_commit[r.key.slot].b = try w.mark();
    }

    /// One round through the real scheduler, its trace from empty.
    fn round(w: *Watch, rs: []const Row) !void {
        try w.u.f.flush();
        w.trace.reset();
        var out: [4]u32 = undefined;
        try schedule.run(Watch, w, rs, out[0..rs.len]);
        try w.u.f.flush();
        w.u.rounds += 1;
    }
};

// -- T2: a together round, stream by stream, against each stream alone ------------------------------------------------

/// The row buffers a per-sequence launch (or a draw) reads or writes, at row i of the Engine's decode Buffers or of the
/// stages (cuda_afd_users_shared.zig rowView's, kdaSeq's, Users.logitsOf's).
const row_paths = [_][]const u8{ "b.lat", "b.ikr", "b.igr", "b.qi", "b.ka", "b.kg", "b.lat_s.qa", "b.lat_s.ol", "b.logits", "users.shared_proj", "users.shared_out" };

/// A Space naming row `i` of each row buffer as that buffer's row 0, then everything `s` names (afd_pool_test.zig's
/// regionNames method): row i of a together round compared with its stream's solo twin's row 0. A temporary
/// ("tmp.float.1": python's torch.empty, named in the order a round makes them) by its kind ("tmp.float"): a draw
/// makes the same temporaries whether it is a round's first or its fourth.
fn rowNames(s: *const Space, i: usize) !Space {
    var out: Space = .{ .gpa = gpa };
    errdefer out.deinit();
    for (s.entries.items) |e| for (row_paths) |p| if (std.mem.eql(u8, e.path, p)) {
        const rb = e.buf.inner() * e.buf.dtype.size();
        try out.entries.append(gpa, .{ .path = try gpa.dupe(u8, e.path), .base = e.base + i * rb, .len = rb, .buf = e.buf });
    };
    for (s.entries.items) |e| try out.add(tmpKind(e.path), e.buf);
    return out;
}

/// A temporary's kind: its name without the counters (cuda_tmp.zig's "tmp.<op>.<n>[.<field>]" as "tmp.<op>[.<field>]").
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

/// Two addresses by buffer name and offset.
fn sameAddr(sa: *const Space, x: u64, sb: *const Space, y: u64) !void {
    const p = sa.name(x);
    const q = sb.name(y);
    try eq(p == null, q == null);
    if (p == null) return eq(x, y);
    try t.expectEqualStrings(p.?.path, q.?.path);
    try eq(p.?.off, q.?.off);
}

/// sameAddr for a row-by-row launch at `rows` (n against 1): the AFD planes (b.afd_recv, [3, R, D] as python lays the
/// two halves' replies and the shared expert's) sit R·D apart, so their offsets scale with the rows.
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
            try t.expectEqualSlices(usize, x.shape[1..], v.t.shape[1..]); // rows: n against 1
        },
        .list => |l| {
            try eq(l.len, v.list.len);
            for (l, v.list) |p, q| try rowVal(sa, p, sb, q, rows);
        },
        .int => |x| if (x != rows[0] or v.int != rows[1]) try eq(x, v.int), // else a row count (afd.exchange's R)
        else => try t.expectEqualDeep(u, v),
    }
}

/// A together round's row-by-row launches against a solo twin's: the same launches in the same order, each on the same
/// buffers at the same offsets (rows from 0), with the same constexprs (so the same compiled variant) and the same
/// scalars beside the row count; only the rows (`rows`: n against 1) and what scales with them (grids, a tensor's rows,
/// a launch's row argument) differ. Returns the launches compared.
fn expectRowOps(sa: *const Space, a: []const Event, sb: *const Space, b: []const Event, rows: [2]usize) !usize {
    try eq(a.len, b.len);
    for (a, b, 0..) |x, y, k| {
        errdefer std.debug.print("shared_forward row op {d}: {s} against {s}\n", .{ k, x.name, y.name });
        try eq(x.op, y.op);
        try t.expectEqualStrings(x.name, y.name);
        try t.expectEqualDeep(x.consts, y.consts);
        try eq(x.args.len, y.args.len);
        for (x.args, y.args) |u, v| {
            try t.expectEqualStrings(u.name, v.name);
            switch (u.value) {
                .ptr => |p| try sameRowAddr(sa, p.addr, sb, v.value.ptr.addr, rows),
                else => {}, // a row count, or a stride of one
            }
        }
        try eq(x.vals.len, y.vals.len);
        for (x.vals, y.vals) |u, v| try rowVal(sa, u, sb, v, rows);
    }
    return a.len;
}

const Tally = struct { seq: usize = 0, row: usize = 0, draws: usize = 0 };

/// T2's check of one round: row i of `a`'s together round against twins[i]'s round of row i alone: every per-sequence
/// launch, draw and commit exact (State by name and offset, rows mapped row i to row 0, grids, constexprs, scalars),
/// every row-by-row launch the twin's once (expectRowOps).
fn checkRound(a: *dev.Rec, wa: *const Watch, twins: []const *dev.Rec, wt: []const Watch, tally: *Tally) !void {
    const ea = a.trace.events.items;
    const rows = [2]usize{ twins.len, 1 };
    for (twins, wt, 0..) |b, *w, i| {
        const eb = b.trace.events.items;
        var na = try rowNames(&a.space, i);
        defer na.deinit();
        var nb = try rowNames(&b.space, 0);
        defer nb.deinit();
        for (0..L) |li| {
            errdefer std.debug.print("shared_forward row {d} layer {d}\n", .{ i, li });
            try glm.serve_rec.expectSame(&na, wa.r_seq[i][li].of(ea), &nb, w.r_seq[i][li].of(eb));
            tally.seq += wa.r_seq[i][li].b - wa.r_seq[i][li].a;
            tally.row += try expectRowOps(&a.space, wa.r_in[li].of(ea), &b.space, w.r_in[li].of(eb), rows);
            tally.row += try expectRowOps(&a.space, wa.r_out[li].of(ea), &b.space, w.r_out[li].of(eb), rows);
        }
        tally.row += try expectRowOps(&a.space, wa.r_tail.of(ea), &b.space, w.r_tail.of(eb), rows);
        try glm.serve_rec.expectSame(&na, wa.r_sample[i].of(ea), &nb, w.r_sample[i].of(eb));
        try glm.serve_rec.expectSame(&na, wa.r_commit[i].of(ea), &nb, w.r_commit[i].of(eb));
        tally.draws += wa.r_sample[i].b - wa.r_sample[i].a + wa.r_commit[i].b - wa.r_commit[i].a;
        // the open's ids: n tokens at once against one; the same launches
        try eq(wa.r_open.b - wa.r_open.a, w.r_open.b - w.r_open.a);
        for (wa.r_open.of(ea), w.r_open.of(eb)) |x, y| try t.expectEqualStrings(x.name, y.name);
    }
}

/// Recs for a together round of `n` streams and each stream's solo twin, every one with the switch on.
const Set = struct {
    a: dev.Rec = undefined,
    twins: [4]dev.Rec = undefined,
    made: usize = 0,
    n: u8 = 0,
    fn init(s: *Set, n: u8) !void {
        s.n = n;
        s.made = 0;
        try s.a.initRows(n, 64, 8);
        errdefer s.a.deinit();
        try shareOn(&s.a);
        errdefer for (s.twins[0..s.made]) |*x| x.deinit();
        for (0..n) |_| {
            try s.twins[s.made].initRows(n, 64, 8);
            s.made += 1;
            try shareOn(&s.twins[s.made - 1]);
        }
    }
    fn deinit(s: *Set) void {
        for (s.twins[0..s.made]) |*x| x.deinit();
        s.a.deinit();
    }
    fn ptrs(s: *Set) [4]*dev.Rec {
        var out: [4]*dev.Rec = undefined;
        for (0..s.n) |i| out[i] = &s.twins[i];
        return out;
    }
};

fn moeLayers(u: *const glm.afd_users.Users) usize {
    var n: usize = 0;
    for (0..u.layerCount()) |li| n += @intFromBool(u.isMoe(li));
    return n;
}

fn together(n: u8, rounds: usize) !void {
    var s: Set = .{};
    try s.init(n);
    defer s.deinit();
    // today's per-row path at the same rows, for the launch count only
    var p: dev.Rec = undefined;
    try p.initRows(n, 64, 8);
    defer p.deinit();
    var wa: Watch = .{ .u = &s.a.users, .trace = &s.a.trace };
    var wt: [4]Watch = undefined;
    for (s.twins[0..n], wt[0..n]) |*b, *w| w.* = .{ .u = &b.users, .trace = &b.trace };
    var wp: Watch = .{ .u = &p.users, .trace = &p.trace };
    const moe = moeLayers(&s.a.users);
    var tally: Tally = .{};
    var shared_launches: usize = 0;
    var row_launches: usize = 0;
    for (0..rounds) |k| {
        var rs: [4]Row = undefined;
        for (0..n) |i| rs[i] = rowAt(@intCast(i), k);
        try wa.round(rs[0..n]);
        shared_launches = s.a.trace.events.items.len;
        try wp.round(rs[0..n]);
        row_launches = p.trace.events.items.len;
        for (0..n) |i| try wt[i].round(rs[i .. i + 1]);
        const tp = s.ptrs();
        try checkRound(&s.a, &wa, tp[0..n], wt[0..n], &tally);
        for (0..n) |i| {
            const key: Key = rs[i].key;
            try eq(base[i] + k + 1, s.a.users.position(key));
            try eq(s.twins[i].users.position(key), s.a.users.position(key));
            try eq(p.users.position(key), s.a.users.position(key));
            try eq(stateOfSlot(&s.twins[i], i).parity, stateOfSlot(&s.a, i).parity);
            try eq(stateOfSlot(&p, i).parity, stateOfSlot(&s.a, i).parity);
        }
        // one exchange a MoE layer, all n rows in it (the per-row path's own count)
        try eq((k + 1) * moe, s.a.posts);
        try eq(p.posts, s.a.posts);
        for (s.a.headers[k * moe .. (k + 1) * moe], 0..) |h, li| {
            try eq(@as(i32, n), h.rows);
            try eq(p.headers[k * moe + li].layer, h.layer);
        }
        for (s.twins[0..n]) |*b| for (b.headers[k * moe .. (k + 1) * moe]) |h| try eq(@as(i32, 1), h.rows);
    }
    std.debug.print("shared_forward recorded C{d}: {d} rounds; each stream against its solo twin: {d} per-sequence launches exact, {d} draw/commit launches exact, {d} row-by-row launch pairs equal (once a round, constexprs equal, R n against 1); {d} launches a shared round against {d} per-row; exchanges {d}, rows={d}; no hardware numerics\n", .{ n, rounds, tally.seq, tally.draws, tally.row, shared_launches, row_launches, s.a.posts, n });
}

test "shared_forward recorded C2: one forward over both rows, each stream's per-sequence launches its solo twin's, row ops once, 3 rounds" {
    try together(2, 3);
}

test "shared_forward recorded C4: one forward over four rows (dense, the dense limit, sparse), each stream exact against alone, 3 rounds" {
    try together(4, 3);
}

fn misSplit(how: @FieldType(Watch, "mis")) !void {
    var s: Set = .{};
    try s.init(2);
    defer s.deinit();
    var wa: Watch = .{ .u = &s.a.users, .trace = &s.a.trace, .mis = how };
    var wt: [2]Watch = undefined;
    for (s.twins[0..2], &wt) |*b, *w| w.* = .{ .u = &b.users, .trace = &b.trace };
    const rs = [_]Row{ rowAt(0, 0), rowAt(1, 0) };
    try wa.round(&rs);
    for (0..2) |i| try wt[i].round(rs[i .. i + 1]);
    var tally: Tally = .{};
    const tp = s.ptrs();
    std.debug.print("shared_forward RED mis-split ({t}): the check below must fail\n", .{how});
    if (checkRound(&s.a, &wa, tp[0..2], &wt, &tally)) |_| return error.MisSplitNotCaught else |e| std.debug.print("shared_forward RED mis-split ({t}) caught: {t}\n", .{ how, e });
}

test "shared_forward RED: a row's per-sequence part on another stream's State fails the solo-twin check" {
    try misSplit(.key);
}

test "shared_forward RED: a row's per-sequence part on another row's buffers fails the solo-twin check" {
    try misSplit(.row);
}

/// dev.Rec with shared_pool's pool (afd_pool_test.zig's Pooled): slot 0's caches detached as afd_attn_gpu does before the
/// Engine's attach, the extra States' by usePool, the pool placed after everything else, every slot's region reserved
/// at the Engine's capacity (11 blocks each), then the switch on.
const Pooled = struct {
    r: dev.Rec = undefined,
    pool: glm.afd_pool.Pool = undefined,
    fn init(x: *Pooled, n: u8) !void {
        const P = glm.afd_pool;
        try x.r.initRows(n, 64, 8);
        errdefer x.r.deinit();
        P.detachTokens(&x.r.e.st);
        x.pool = try P.Pool.init(gpa, &x.r.m, 44 * P.block, x.r.e.buf.rows, true, x.r.e.st.capacity);
        errdefer x.pool.deinit(gpa);
        try x.pool.walk(&x.r.space, glm.trace.placeAll);
        try x.r.users.usePool(&x.pool);
        for (0..n) |i| try x.r.users.reserve(@intCast(i), 2560, 0);
        try shareOn(&x.r);
    }
    fn deinit(x: *Pooled) void {
        x.pool.deinit(gpa);
        x.r.deinit();
    }
};

test "shared_forward recorded C4 with shared_pool's pool: each stream's per-sequence launches on its own region exact against alone, row ops once, 2 rounds" {
    var a: Pooled = .{};
    try a.init(4);
    defer a.deinit();
    var twins: [4]Pooled = @splat(.{});
    var made: usize = 0;
    defer for (twins[0..made]) |*x| x.deinit();
    for (&twins) |*x| {
        try x.init(4);
        made += 1;
    }
    var wa: Watch = .{ .u = &a.r.users, .trace = &a.r.trace };
    var wt: [4]Watch = undefined;
    var tp: [4]*dev.Rec = undefined;
    for (&twins, &wt, &tp) |*x, *w, *p| {
        w.* = .{ .u = &x.r.users, .trace = &x.r.trace };
        p.* = &x.r;
    }
    var tally: Tally = .{};
    var pooled: usize = 0;
    for (0..2) |k| {
        var rs: [4]Row = undefined;
        for (0..4) |i| rs[i] = rowAt(@intCast(i), k);
        try wa.round(&rs);
        for (0..4) |i| try wt[i].round(rs[i .. i + 1]);
        try checkRound(&a.r, &wa, &tp, &wt, &tally);
        for (a.r.trace.events.items) |ev| for (ev.args) |arg| switch (arg.value) {
            .ptr => |q| if (a.r.space.name(q.addr)) |nm| if (std.mem.startsWith(u8, nm.path, "pool.")) {
                pooled += 1;
            },
            else => {},
        };
        for (0..4) |i| try eq(base[i] + k + 1, a.r.users.position(rs[i].key));
    }
    try t.expect(pooled > 0);
    std.debug.print("shared_forward recorded C4 + pool: 2 rounds; per stream against its solo twin {d} per-sequence launches exact on its own region ({d} pool operands), {d} draw/commit launches exact\n", .{ tally.seq, pooled, tally.draws });
}

// -- T3: the shared round's State launches against today's per-row path -------------------------------------------------

/// A State buffer (slot 0's "st.", another slot's "users.stateN."), but not the KDA chain's output scratch
/// (scratch_set.out: the shared round's chain writes its stage there instead).
fn stateOf(s: *const Space, addr: u64) ?[]const u8 {
    const x = s.name(addr) orelse return null;
    if (!std.mem.startsWith(u8, x.path, "st.") and !std.mem.startsWith(u8, x.path, "users.state")) return null;
    if (std.mem.endsWith(u8, x.path, ".scratch_set.out")) return null;
    return x.path;
}

/// State beyond the KDA in-projection rows (st.proj: written by the q4 per row, copied from the stage when shared).
fn deepState(s: *const Space, addr: u64) bool {
    const p = stateOf(s, addr) orelse return false;
    return !std.mem.endsWith(u8, p, ".proj");
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

/// A launch reading or writing a stream's State (its per-sequence part, or its commit).
fn perSequence(s: *const Space, e: Event) bool {
    for (e.args) |a| switch (a.value) {
        .ptr => |p| if (deepState(s, p.addr)) return true,
        else => {},
    };
    for (e.vals) |v| if (valTouches(s, v)) return true;
    return false;
}

fn sameState(sa: *const Space, x: u64, sb: *const Space, y: u64) !void {
    const p = stateOf(sa, x);
    const q = stateOf(sb, y);
    try eq(p == null, q == null);
    if (p == null) return; // rows and scratch: today's Saved rows against the Engine's rows (by design)
    try sameAddr(sa, x, sb, y);
}

fn stateVal(sa: *const Space, u: Val, sb: *const Space, v: Val) !void {
    try eq(std.meta.activeTag(u), std.meta.activeTag(v));
    switch (u) {
        .t => |x| {
            try sameState(sa, x.ptr, sb, v.t.ptr);
            if (stateOf(sa, x.ptr) == null) return;
            try eq(x.dtype, v.t.dtype);
            try eq(x.dims, v.t.dims);
            try eq(x.shape, v.t.shape);
            try eq(x.stride, v.t.stride);
        },
        .list => |l| {
            try eq(l.len, v.list.len);
            for (l, v.list) |p, q| try stateVal(sa, p, sb, q);
        },
        else => try t.expectEqualDeep(u, v),
    }
}

/// Every launch of `a` and `b` that touches a State (perSequence), in order: the same kernel, grid, constexprs and
/// scalars, the same State operands by name, offset, shape and strides. Returns the launches compared.
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
        errdefer std.debug.print("shared_forward State launch {d}: {s} against {s}\n", .{ n, x.name, y.name });
        try eq(x.op, y.op);
        try t.expectEqualStrings(x.name, y.name);
        try eq(x.grid, y.grid);
        try t.expectEqualDeep(x.consts, y.consts);
        try eq(x.args.len, y.args.len);
        for (x.args, y.args) |u, v| {
            try t.expectEqualStrings(u.name, v.name);
            switch (u.value) {
                .ptr => |p| try sameState(sa, p.addr, sb, v.value.ptr.addr),
                else => try t.expectEqualDeep(u.value, v.value),
            }
        }
        try eq(x.vals.len, y.vals.len);
        for (x.vals, y.vals) |u, v| try stateVal(sa, u, sb, v);
    }
    try eq(a.len, ia);
    try eq(b.len, ib);
    return n;
}

test "shared_forward one-row shared round's State launches equal today's per-row path's, every slot at C2 and C4, 3 rounds" {
    var compared: usize = 0;
    var cases: usize = 0;
    for ([_]u8{ 2, 4 }) |n| for (0..n) |slot| {
        var a: dev.Rec = undefined;
        try a.initRows(n, 64, 8);
        defer a.deinit();
        try shareOn(&a);
        var b: dev.Rec = undefined;
        try b.initRows(n, 64, 8);
        defer b.deinit();
        var wa: Watch = .{ .u = &a.users, .trace = &a.trace };
        var wb: Watch = .{ .u = &b.users, .trace = &b.trace };
        for (0..3) |k| {
            const rs = [_]Row{rowAt(@intCast(slot), k)};
            try wa.round(&rs);
            try wb.round(&rs);
            errdefer std.debug.print("shared_forward C{d} slot {d} round {d}\n", .{ n, slot, k });
            compared += try expectStateOps(&a.space, a.trace.events.items, &b.space, b.trace.events.items);
            try eq(b.users.position(rs[0].key), a.users.position(rs[0].key));
            try eq(base[slot] + k + 1, a.users.position(rs[0].key));
            cases += 1;
        }
    };
    std.debug.print("shared_forward one-row shared = per-row: {d} rounds, {d} State launches exact (kernel, grid, constexprs, scalars, State operands)\n", .{ cases, compared });
}

// -- T4: cancel and failure in the shared round (through Users.round, the production entry) -------------------------

fn sharedFailure(take_no: usize) !void {
    var r: dev.Rec = undefined;
    try r.initRows(2, 64, 8);
    defer r.deinit();
    try shareOn(&r);
    const rs = [_]Row{ rowAt(0, 0), rowAt(1, 0) };
    var out: [2]u32 = undefined;
    r.fail_take = take_no;
    try t.expectError(error.ExpertFailed, r.users.round(&rs, &out));
    try r.f.flush();
    for (r.trace.events.items) |ev| try t.expect(!std.mem.eql(u8, ev.name, "torch.topk") and !std.mem.eql(u8, ev.name, "torch.cpu") and !std.mem.eql(u8, ev.name, "torch.fill_"));
    for (rs) |row| try eq(row.position, r.users.position(row.key));
    const posts = r.posts;
    r.trace.reset();
    try t.expectError(error.ExpertFailed, r.users.round(&rs, &out));
    try t.expectError(error.ExpertFailed, r.users.prefill(.{ .slot = 0, .generation = 2 }, &.{ 1, 2 }));
    try r.f.flush();
    try eq(@as(usize, 0), r.trace.events.items.len);
    try eq(posts, r.posts);
}

test "shared_forward shared round transport failure at the first MoE layer: sticky, no draw or commit, positions unchanged" {
    try sharedFailure(1);
}

test "shared_forward shared round transport failure at the last MoE layer: sticky, no draw or commit, positions unchanged" {
    try sharedFailure(2);
}

test "shared_forward shared round cancelled row: not drawn or committed, its peer commits, the round settles" {
    var r: dev.Rec = undefined;
    try r.initRows(2, 64, 8);
    defer r.deinit();
    try shareOn(&r);
    var log: dev.OwnLog = .{};
    log.attach(&r);
    var rs = [_]Row{ rowAt(0, 0), rowAt(1, 0) };
    var flag = false;
    const Check = struct {
        fn yes(ctx: ?*anyopaque) bool {
            const value: *bool = @ptrCast(@alignCast(ctx.?));
            return value.*;
        }
    };
    rs[0].cancel_ctx = &flag;
    rs[0].cancel_check = Check.yes;
    r.cancel_in_take = &flag;
    var out: [2]u32 = undefined;
    try r.users.round(&rs, &out);
    try eq(@as(usize, 6), log.n);
    try eq(.cancel_before_sample, log.values[0].event);
    try eq(.cancel_before_commit, log.values[1].event);
    try eq(@as(u8, 1), log.values[2].slot);
    try eq(.commit_before, log.values[2].event);
    try eq(.commit_after, log.values[3].event);
    try eq(.round_settled, log.values[4].event);
    try eq(.round_settled, log.values[5].event);
    try eq(rs[0].position, r.users.position(rs[0].key));
    try eq(rs[1].position + 1, r.users.position(rs[1].key));
    try eq(@as(usize, 2), r.posts); // the round ran its two exchanges, both rows in each
}

// -- T6: the real round scheduler's shared branch over a numeric stand-in ----------------------------------------------

/// A device whose row ops are pure per row, whose per-sequence ops read each slot's own state (position, parity, value),
/// and whose MoE mixes nothing across rows: the per-row forward (concurrent_users's) and the shared one must agree, and every
/// stream together must equal itself alone. `mis`: the per-sequence part reads the next row's state (RED).
const Stand = struct {
    const St = struct { position: usize = 0, parity: u1 = 0, value: u32 = 0 };
    const Call = struct { kind: enum { begin, shared_open, shared_in, shared_seq, shared_out, shared_tail, open, front, sample, commit }, slot: u8 = 255, layer: usize = 0, rows: usize = 0 };
    on: bool,
    mis: bool = false,
    states: [4]St = @splat(.{}),
    hidden: [4]u32 = @splat(0), // the shared forward's, by row
    own: [4]u32 = @splat(0), // the per-row forward's, by slot
    gathered: [4]u32 = @splat(0),
    result: [4]u32 = @splat(0),
    rs: []const Row = &.{},
    log: std.ArrayList(Call) = .empty,
    fail_layer: ?usize = null,
    cancel: ?*bool = null,
    cancel_layer: ?usize = null,
    failed: bool = false,

    fn init(on: bool) Stand {
        var s: Stand = .{ .on = on };
        for (&s.states, 0..) |*st, i| st.* = .{ .position = base[i], .parity = @intCast(i % 2), .value = @intCast(1000 + 7 * i) };
        return s;
    }
    fn deinit(s: *Stand) void {
        s.log.deinit(gpa);
    }
    fn note(s: *Stand, c: Call) !void {
        try s.log.append(gpa, c);
    }
    fn rowOp(x: u32, li: usize) u32 {
        return x *% @as(u32, @intCast(li + 3)) +% 1;
    }
    fn seqOp(x: u32, st: St, li: usize) u32 {
        return x +% st.value *% @as(u32, @intCast(li + 1)) +% st.parity +% @as(u32, @intCast(st.position / 512 + st.position / 4));
    }
    fn expert(x: u32) u32 {
        return (x *% 3 +% x *% 5) +% x *% 7;
    }
    pub fn alive(s: *Stand) !void {
        if (s.failed) return error.ExpertFailed;
    }
    pub fn begin(s: *Stand, rows: []const Row) !void {
        s.rs = rows;
        for (rows) |row| try eq(row.position, s.states[row.key.slot].position);
        try s.note(.{ .kind = .begin, .rows = rows.len });
    }
    pub fn layerCount(_: *Stand) usize {
        return 6;
    }
    pub fn isMoe(_: *Stand, li: usize) bool {
        return li >= 2;
    }
    // the per-row forward
    pub fn open(s: *Stand, row: Row, _: usize) !void {
        s.own[row.key.slot] = row.token +% @as(u32, @intCast(row.position));
        try s.note(.{ .kind = .open, .slot = row.key.slot });
    }
    pub fn front(s: *Stand, row: Row, _: usize, li: usize) !void {
        const k = row.key.slot;
        s.own[k] = seqOp(rowOp(s.own[k], li), s.states[k], li);
        try s.note(.{ .kind = .front, .slot = k, .layer = li });
    }
    pub fn gather(s: *Stand, row: Row, i: usize) !void {
        s.gathered[i] = s.own[row.key.slot];
    }
    pub fn exchange(s: *Stand, li: usize, n: usize) !void {
        try s.moe(li, s.gathered[0..n]);
    }
    pub fn scatter(s: *Stand, row: Row, i: usize, _: usize) !void {
        s.own[row.key.slot] +%= s.result[i];
    }
    pub fn tail(_: *Stand, _: Row, _: usize) !void {}
    fn moe(s: *Stand, li: usize, x: []const u32) !void {
        if (s.cancel_layer == li) if (s.cancel) |flag| {
            flag.* = true;
        };
        if (s.fail_layer == li) {
            s.failed = true;
            return error.ExpertFailed;
        }
        for (x, s.result[0..x.len]) |v, *out| out.* = expert(v);
    }
    // shared_forward's forward
    pub fn shared(s: *Stand) bool {
        return s.on;
    }
    pub fn sharedOpen(s: *Stand, rows: []const Row) !void {
        for (rows, 0..) |row, i| s.hidden[i] = row.token +% @as(u32, @intCast(row.position));
        try s.note(.{ .kind = .shared_open, .rows = rows.len });
    }
    pub fn sharedIn(s: *Stand, rows: []const Row, li: usize) !void {
        for (s.hidden[0..rows.len]) |*h| h.* = rowOp(h.*, li);
        try s.note(.{ .kind = .shared_in, .layer = li, .rows = rows.len });
    }
    pub fn sharedSeq(s: *Stand, row: Row, i: usize, li: usize) !void {
        const k = if (s.mis) s.rs[(i + 1) % s.rs.len].key.slot else row.key.slot;
        s.hidden[i] = seqOp(s.hidden[i], s.states[k], li);
        try s.note(.{ .kind = .shared_seq, .slot = row.key.slot, .layer = li, .rows = 1 });
    }
    pub fn sharedOut(s: *Stand, rows: []const Row, li: usize) !void {
        if (s.isMoe(li)) {
            try s.moe(li, s.hidden[0..rows.len]);
            for (s.hidden[0..rows.len], s.result[0..rows.len]) |*h, r| h.* +%= r;
        }
        try s.note(.{ .kind = .shared_out, .layer = li, .rows = rows.len });
    }
    pub fn sharedTail(s: *Stand, rows: []const Row) !void {
        try s.note(.{ .kind = .shared_tail, .rows = rows.len });
    }
    pub fn settle(s: *Stand) !void {
        try s.alive();
    }
    fn value(s: *const Stand, row: Row, i: usize) u32 {
        return if (s.on) s.hidden[i] else s.own[row.key.slot];
    }
    pub fn sample(s: *Stand, row: Row, i: usize) !u32 {
        try s.note(.{ .kind = .sample, .slot = row.key.slot });
        return s.value(row, i) % 997;
    }
    pub fn commit(s: *Stand, row: Row, i: usize) !void {
        const st = &s.states[row.key.slot];
        st.value = s.value(row, i);
        st.position += 1;
        st.parity = 1 - st.parity;
        try s.note(.{ .kind = .commit, .slot = row.key.slot });
    }
};

/// `rounds` rounds of slots [0, n) together on `s`; each round's tokens appended to `tokens` (row order).
fn standRounds(s: *Stand, n: usize, rounds: usize, tokens: *std.ArrayList(u32)) !void {
    for (0..rounds) |k| {
        var rs: [4]Row = undefined;
        for (0..n) |i| rs[i] = .{ .key = .{ .slot = @intCast(i), .generation = 1 }, .token = @intCast(10 + i + k), .position = base[i] + k };
        var out: [4]u32 = undefined;
        try schedule.run(Stand, s, rs[0..n], out[0..n]);
        try tokens.appendSlice(gpa, out[0..n]);
    }
}

/// Slot `slot` alone for `rounds` rounds on `s` (its tokens appended).
fn standAlone(s: *Stand, slot: u8, rounds: usize, tokens: *std.ArrayList(u32)) !void {
    for (0..rounds) |k| {
        const rs = [_]Row{.{ .key = .{ .slot = slot, .generation = 1 }, .token = @intCast(10 + slot + k), .position = base[slot] + k }};
        var out: [1]u32 = undefined;
        try schedule.run(Stand, s, &rs, &out);
        try tokens.append(gpa, out[0]);
    }
}

test "shared_forward real round scheduler, numeric stand-in: shared = per-row = each stream alone at token and State level, C2 and C4; the shared call order" {
    for ([_]usize{ 2, 4 }) |n| {
        var sh = Stand.init(true);
        defer sh.deinit();
        var pr = Stand.init(false);
        defer pr.deinit();
        var a: std.ArrayList(u32) = .empty;
        defer a.deinit(gpa);
        var b: std.ArrayList(u32) = .empty;
        defer b.deinit(gpa);
        try standRounds(&sh, n, 4, &a);
        try standRounds(&pr, n, 4, &b);
        try t.expectEqualSlices(u32, b.items, a.items);
        for (0..n) |i| try eq(pr.states[i], sh.states[i]);
        for (0..n) |slot| {
            var solo = Stand.init(true);
            defer solo.deinit();
            var c: std.ArrayList(u32) = .empty;
            defer c.deinit(gpa);
            try standAlone(&solo, @intCast(slot), 4, &c);
            for (0..4) |k| try eq(a.items[k * n + slot], c.items[k]);
            try eq(sh.states[slot], solo.states[slot]);
        }
        // the shared round's calls, in order: begin, open, per layer in / seq a row / out, tail, a draw a row, a commit a row
        const per = 1 + 1 + 6 * (2 + n) + 1 + 2 * n;
        try eq(4 * per, sh.log.items.len);
        const calls = sh.log.items[0..per];
        try eq(.begin, calls[0].kind);
        try eq(.shared_open, calls[1].kind);
        var at: usize = 2;
        for (0..6) |li| {
            try eq(Stand.Call{ .kind = .shared_in, .layer = li, .rows = n }, calls[at]);
            for (0..n) |i| try eq(Stand.Call{ .kind = .shared_seq, .slot = @intCast(i), .layer = li, .rows = 1 }, calls[at + 1 + i]);
            try eq(Stand.Call{ .kind = .shared_out, .layer = li, .rows = n }, calls[at + 1 + n]);
            at += 2 + n;
        }
        try eq(Stand.Call{ .kind = .shared_tail, .rows = n }, calls[at]);
        for (0..n) |i| try eq(Stand.Call{ .kind = .sample, .slot = @intCast(i) }, calls[at + 1 + i]);
        for (0..n) |i| try eq(Stand.Call{ .kind = .commit, .slot = @intCast(i) }, calls[at + 1 + n + i]);
        for (sh.log.items) |c| try t.expect(c.kind != .open and c.kind != .front); // never the per-row forward
        for (pr.log.items) |c| try t.expect(c.kind != .shared_open and c.kind != .shared_in and c.kind != .shared_seq);
        // RED: the per-sequence part on the next row's state: tokens move
        var red = Stand.init(true);
        defer red.deinit();
        red.mis = true;
        var r: std.ArrayList(u32) = .empty;
        defer r.deinit(gpa);
        try standRounds(&red, n, 4, &r);
        try t.expect(!std.mem.eql(u32, a.items, r.items));
        std.debug.print("shared_forward scheduler stand-in C{d}: 4 rounds shared = per-row = alone ({d} tokens); mis-split RED tokens differ\n", .{ n, a.items.len });
    }
}

test "shared_forward real round scheduler, numeric stand-in: a failure in the shared forward draws and commits nothing; a cancelled row is skipped, its peers commit" {
    var s = Stand.init(true);
    defer s.deinit();
    s.fail_layer = 3;
    const before = s.states;
    const rs = [_]Row{ .{ .key = .{ .slot = 0, .generation = 1 }, .token = 1, .position = base[0] }, .{ .key = .{ .slot = 1, .generation = 1 }, .token = 2, .position = base[1] } };
    var out: [2]u32 = undefined;
    try t.expectError(error.ExpertFailed, schedule.run(Stand, &s, &rs, &out));
    for (s.log.items) |c| try t.expect(c.kind != .sample and c.kind != .commit and c.kind != .shared_tail);
    try eq(before, s.states);
    try t.expectError(error.ExpertFailed, schedule.run(Stand, &s, &rs, &out)); // sticky
    var c = Stand.init(true);
    defer c.deinit();
    var flag = false;
    c.cancel = &flag;
    c.cancel_layer = 2;
    const Check = struct {
        fn yes(ctx: ?*anyopaque) bool {
            const value: *bool = @ptrCast(@alignCast(ctx.?));
            return value.*;
        }
    };
    var rc = rs;
    rc[1].cancel_ctx = &flag;
    rc[1].cancel_check = Check.yes;
    try schedule.run(Stand, &c, &rc, &out);
    try eq(base[0] + 1, c.states[0].position);
    try eq(base[1], c.states[1].position);
    try eq(@as(u32, 0), out[1]);
    for (c.log.items) |x| try t.expect(!((x.kind == .sample or x.kind == .commit) and x.slot == 1));
}

// -- T7: bytes -----------------------------------------------------------------------------------------------------------

test "shared_forward bytes: the stages after every other buffer, + their bytes on, unchanged off; the served config's figures" {
    const served = @import("afd_attn_users.zig");
    for ([_]u8{ 2, 4 }) |n| {
        var r: dev.Rec = undefined;
        try r.initRows(n, 64, 8);
        defer r.deinit();
        const off = try r.users.bytes();
        try r.users.useShared();
        const on = try r.users.bytes();
        const add = served.sharedBytes(&r.m, n, true);
        try eq(off.states, on.states);
        try eq(off.rows + add, on.rows);
        try eq(off.total + add, on.total);
        try eq(@as(u64, 0), served.sharedBytes(&r.m, n, false));
        try t.expectError(error.NoMtp, r.users.useDrafts(.{}, 8)); // draft_shared: drafts compose with the stages (this rig has no MTP head)
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    defer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    m.mtp_layers = 1;
    const s4 = served.sharedBytes(&m, 4, true);
    const s2 = served.sharedBytes(&m, 2, true);
    std.debug.print("shared_forward bytes served config world=1: kda width {d}, kda heads {d}; USERS=4 stages {d} B, USERS=2 {d} B\n", .{ m.kdaWidth(), m.ll(), s4, s2 });
    try eq(@as(u64, 264_704), s4);
    try eq(@as(u64, 132_352), s2);
    var q = try served.extra(gpa, &m, 262208, 2048, 4);
    const was = q;
    served.withShared(&q, &m, 4, false);
    try eq(was.total, q.total);
    served.withShared(&q, &m, 4, true);
    try eq(was.total + s4, q.total);
    try eq(was.rows + s4, q.rows);
    try eq(was.states, q.states);
}

// -- T8: default off -----------------------------------------------------------------------------------------------------

fn names(s: *const Space, ev: []const Event, part: []const u8) bool {
    for (ev) |e| {
        for (e.args) |a| switch (a.value) {
            .ptr => |p| if (s.name(p.addr)) |x| if (std.mem.indexOf(u8, x.path, part) != null) return true,
            else => {},
        };
        for (e.vals) |v| if (valNames(s, v, part)) return true;
    }
    return false;
}

fn valNames(s: *const Space, v: Val, part: []const u8) bool {
    return switch (v) {
        .t => |x| if (s.name(x.ptr)) |y| std.mem.indexOf(u8, y.path, part) != null else false,
        .list => |l| for (l) |u| {
            if (valNames(s, u, part)) break true;
        } else false,
        else => false,
    };
}

test "shared_forward default off: no stage, no new buffer, today's per-row round; on: the stages, no Saved row" {
    var r: dev.Rec = undefined;
    try r.initRows(2, 64, 8);
    defer r.deinit();
    try t.expect(!r.users.shared());
    try t.expect(r.users.stages == null);
    for (r.space.entries.items) |e| try t.expect(std.mem.indexOf(u8, e.path, "users.shared") == null);
    const rs = [_]Row{ rowAt(0, 0), rowAt(1, 0) };
    var out: [2]u32 = undefined;
    try r.users.round(&rs, &out);
    try r.f.flush();
    try t.expect(names(&r.space, r.trace.events.items, "users.row"));
    try t.expect(!names(&r.space, r.trace.events.items, "users.shared"));
    var s: dev.Rec = undefined;
    try s.initRows(2, 64, 8);
    defer s.deinit();
    try shareOn(&s);
    try s.users.round(&rs, &out);
    try s.f.flush();
    try t.expect(!names(&s.space, s.trace.events.items, "users.row"));
    try t.expect(names(&s.space, s.trace.events.items, "users.shared_proj"));
    try t.expect(names(&s.space, s.trace.events.items, "users.shared_out"));
    try t.expect(!names(&s.space, s.trace.events.items, "users.normed")); // no gather copies
    try t.expect(!names(&s.space, s.trace.events.items, "users.planes")); // no scatter copies
    try eq(r.posts, s.posts);
}

// -- T1: the switch through plan() ------------------------------------------------------------------------------------

const envcall = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};
const harness = @import("env_harness.zig"); // FJPANIC: std's startup environment kept whole

/// An environment variable's value restored at the end of a test (four_user_profile's Saved).
const SavedEnv = struct {
    name: [*:0]const u8,
    old: ?[:0]u8,
    fn take(name: [*:0]const u8) !SavedEnv {
        try harness.detach(); // FJPANIC: libc's environ off std's startup array before the test unsets anything
        return .{ .name = name, .old = if (std.c.getenv(name)) |p| try gpa.dupeSentinel(u8, std.mem.span(p), 0) else null };
    }
    fn put(s: SavedEnv) void {
        if (s.old) |v| {
            _ = envcall.setenv(s.name, v, 1);
            gpa.free(v);
        } else _ = envcall.unsetenv(s.name);
    }
};

test "shared_forward env harness (FJPANIC): SavedEnv moves libc's environ off std's startup array first, so unsetting a name the process started with leaves std's startup environment whole; put() restores the name" {
    var buf: [256]u8 = undefined;
    const name = harness.startupName(&buf) orelse return error.SkipZigTest; // no startup environment: nothing to shift
    {
        const s = try SavedEnv.take(name.ptr);
        defer s.put();
        try t.expect(harness.detached()); // checked before anything is unset, so a failure leaves std's slice whole
        _ = envcall.unsetenv(name.ptr);
        try t.expect(std.c.getenv(name.ptr) == null);
        try t.expect(harness.startupWhole());
    }
    try t.expect(std.c.getenv(name.ptr) != null);
}

test "shared_forward switch through plan(): unset and 0 off, 1 on at TF_AFD_USERS 2 and 4, needs users, composes with TF_AFD_USERS_DRAFTS=1 (draft_shared), anything else refused; the others' plans unchanged" {
    const env = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_MTP", "TF_GLM_SAMPLING", "TF_AFD_USERS_SAMPLING", "TF_AFD_USERS_DRAFTS", "TF_AFD_USERS_SHARED" };
    var saved: [env.len]SavedEnv = undefined;
    for (env, &saved) |n, *s| s.* = try SavedEnv.take(n);
    defer for (saved) |s| s.put();
    for (env) |n| _ = envcall.unsetenv(n);
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const one = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--temperature", "0" });
    const p2 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const p4 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    const auto4 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    // one user: unset off; on refused (it needs users)
    try eq(false, (try served.plan(a, one, cfg)).users_shared);
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
    try t.expectError(error.UsersSharedNeedsUsers, served.plan(a, one, cfg));
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "0", 1);
    try eq(false, (try served.plan(a, one, cfg)).users_shared);
    // TF_AFD_USERS 2 and 4: unset and 0 off, 1 on
    inline for (.{ .{ "2", p2, 2 }, .{ "4", p4, 4 } }) |c| {
        _ = envcall.setenv("TF_AFD_USERS", c[0], 1);
        _ = envcall.unsetenv("TF_AFD_USERS_SHARED");
        try eq(false, (try served.plan(a, c[1], cfg)).users_shared);
        _ = envcall.setenv("TF_AFD_USERS_SHARED", "0", 1);
        try eq(false, (try served.plan(a, c[1], cfg)).users_shared);
        _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
        const p = try served.plan(a, c[1], cfg);
        try eq(true, p.users_shared);
        try eq(@as(u8, c[2]), p.users);
        try eq(false, p.users_drafts);
        try eq(false, p.sampled);
        for ([_][*:0]const u8{ "", "2", "00", "01", "true", "on", " 1", "1 ", "yes", "shared" }) |v| {
            _ = envcall.setenv("TF_AFD_USERS_SHARED", v, 1);
            try t.expectError(error.BadAfdUsersShared, served.plan(a, c[1], cfg));
        }
    }
    // with four_user_profile's pool and interleave, and user_sampling's sampled users: on beside them, their plans unchanged
    _ = envcall.setenv("TF_AFD_USERS", "4", 1);
    _ = envcall.setenv("TF_AFD_POOL_TOKENS", "720896", 1);
    _ = envcall.setenv("TF_AFD_USERS_INTERLEAVE", "1", 1);
    _ = envcall.setenv("TF_GLM_SAMPLING", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_SAMPLING", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
    const j = try served.plan(a, p4, cfg);
    try eq(true, j.users_shared);
    try eq(true, j.sampled);
    try eq(@as(usize, 720896), j.pool);
    try t.expect(j.interleave != null);
    _ = envcall.unsetenv("TF_AFD_USERS_SHARED");
    const k = try served.plan(a, p4, cfg);
    try eq(false, k.users_shared);
    try eq(j.sampled, k.sampled);
    try eq(j.pool, k.pool);
    // user_drafts's drafts: draft_shared composes them (both on: one plan with both); 0 or unset, user_drafts's plan as before
    _ = envcall.setenv("TF_AFD_MTP", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_DRAFTS", "1", 1);
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "1", 1);
    const both = try served.plan(a, auto4, cfg);
    try eq(true, both.users_drafts);
    try eq(true, both.users_shared);
    _ = envcall.setenv("TF_AFD_USERS_SHARED", "0", 1);
    const d = try served.plan(a, auto4, cfg);
    try eq(true, d.users_drafts);
    try eq(false, d.users_shared);
    _ = envcall.unsetenv("TF_AFD_USERS_SHARED");
    try eq(true, (try served.plan(a, auto4, cfg)).users_drafts);
}
