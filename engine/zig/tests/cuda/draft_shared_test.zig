//! draft_shared (local integration): user_drafts's lone drafts (TF_AFD_USERS_DRAFTS=1, mode A) and shared_forward's shared forward
//! (TF_AFD_USERS_SHARED=1, stage 1) together, CPU tests (step test-draft_shared, filter "draft_shared"). A turn with exactly one
//! active stream that may draft runs user_drafts's drafted lone loop; every lane-core round runs shared_forward's one undrafted
//! forward over every row; each tracked slot's MTP backlog grows from its own row of the shared head.
//! - plan() and the users backend take both switches; bytes and forecasts carry both.
//! - The backlog, directly: a shadow of the recorded real `Users` tags every final-norm row with its slot and follows the
//!   copies into the backlogs: through shared rounds each tracked slot carries exactly its own rows and draws, = the
//!   per-row path's (user_drafts alone) = its solo twin's; the prompt pass -> shared -> drafted -> shared -> resume life.
//! - Cancel and failure in a shared round and in a lone drafted loop.
//! - Through the real LaneHost, lane core, users backend and lone driver over a stand-in whose rounds go through the
//!   real round scheduler (its shared or per-row forward): lone -> shared -> lone and shared -> lone -> shared mid-reply
//!   at USERS=2 and 4 with pool + interleave = the solo undrafted replies; user_sampling's sampled rule and structured_users's masks.
//! No CUDA numerics; hardware equality is the lab's. The shared forward is @Bizuayeu's shared verify windows
//! (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66, Apache-2.0; reimplemented by shared_forward, not copied).
//! Ash Hart authored TensorFold, its lane core and the lone driver's yield hand-over; user_drafts, shared_forward, user_sampling, structured_users,
//! four_user_profile and joined_profile are Claude Opus 5.5 via Hermes, as their files say (the rigs here follow user_drafts_test.zig's,
//! shared_forward_test.zig's and joined_profile_test.zig's); this file: Claude Opus 5.5 (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const users = @import("glm_users");
const cuda = @import("cuda");
const P = glm.afd_pool;
const scheduling = glm.afd_users_round;
const Key = scheduling.Key;
const Row = scheduling.Row;
const Space = glm.trace.Space;
const Event = glm.trace.Event;
const T = glm.trace.T;
const proto = cuda.xchg.proto;
const t = std.testing;
const gpa = t.allocator;
const io = t.io;
const eq = t.expectEqual;

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

const switch_env = [_][*:0]const u8{ "TF_AFD_USERS", "TF_AFD_POOL_TOKENS", "TF_AFD_USERS_INTERLEAVE", "TF_AFD_MTP", "TF_GLM_SAMPLING", "TF_AFD_USERS_SAMPLING", "TF_AFD_USERS_DRAFTS", "TF_AFD_USERS_SHARED" };

const Env = struct {
    saved: [switch_env.len]SavedEnv = undefined,
    fn take(e: *Env) !void {
        for (switch_env, &e.saved) |n, *s| s.* = try SavedEnv.take(n);
        for (switch_env) |n| _ = envcall.unsetenv(n);
    }
    fn put(e: *Env) void {
        for (e.saved) |s| s.put();
    }
};

fn set(name: [*:0]const u8, value: ?[*:0]const u8) void {
    if (value) |v| _ = envcall.setenv(name, v, 1) else _ = envcall.unsetenv(name);
}

// ---- 1. the switches through plan() ------------------------------------------------------------------------------------

test "draft_shared env harness (FJPANIC): SavedEnv moves libc's environ off std's startup array first, so unsetting a name the process started with leaves std's startup environment whole; put() restores the name" {
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

test "draft_shared switches through plan(): TF_AFD_USERS_SHARED=1 and TF_AFD_USERS_DRAFTS=1 together at TF_AFD_USERS 2 and 4 (pool + interleave + user_sampling too); each alone and both unset as before; every other refusal its own switch's" {
    var env: Env = .{};
    try env.take();
    defer env.put();
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = "{\"max_position_embeddings\":1048576}";
    const one = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--temperature", "0" });
    const auto2 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const auto4 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "auto", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    const off2 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "2", "--temperature", "0" });
    const off4 = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", "0", "--context", "262144", "--parallel", "4", "--temperature", "0" });
    set("TF_AFD_MTP", "1");
    inline for (.{ .{ "2", auto2, off2, 2 }, .{ "4", auto4, off4, 4 } }) |c| {
        set("TF_AFD_USERS", c[0]);
        // both unset: today's plan (a drafts policy refused in users mode, as before)
        set("TF_AFD_USERS_DRAFTS", null);
        set("TF_AFD_USERS_SHARED", null);
        const p0 = try served.plan(a, c[2], cfg);
        try eq(false, p0.users_drafts);
        try eq(false, p0.users_shared);
        try t.expectError(error.UsersDraftsUnsupported, served.plan(a, c[1], cfg));
        // both on: one plan with both (draft_shared)
        set("TF_AFD_USERS_DRAFTS", "1");
        set("TF_AFD_USERS_SHARED", "1");
        const p = try served.plan(a, c[1], cfg);
        try eq(true, p.users_drafts);
        try eq(true, p.users_shared);
        try t.expect(p.drafts.on());
        try eq(@as(u8, c[3]), p.users);
        // each alone: its own plan
        set("TF_AFD_USERS_SHARED", "0");
        const d = try served.plan(a, c[1], cfg);
        try eq(true, d.users_drafts);
        try eq(false, d.users_shared);
        set("TF_AFD_USERS_SHARED", "1");
        set("TF_AFD_USERS_DRAFTS", null);
        const s = try served.plan(a, c[2], cfg);
        try eq(false, s.users_drafts);
        try eq(true, s.users_shared);
        try t.expectError(error.UsersDraftsUnsupported, served.plan(a, c[1], cfg)); // the drafts switch unset
        // every other refusal is still its own switch's, with the other switch on
        set("TF_AFD_USERS_DRAFTS", "1");
        try t.expectError(error.UsersDraftsNeedPolicy, served.plan(a, c[2], cfg));
        set("TF_AFD_USERS_SHARED", "yes");
        try t.expectError(error.BadAfdUsersShared, served.plan(a, c[1], cfg));
        set("TF_AFD_USERS_SHARED", "1");
        set("TF_AFD_USERS_DRAFTS", "on");
        try t.expectError(error.BadAfdUsersDrafts, served.plan(a, c[1], cfg));
    }
    // boot M's form: USERS=4 + four_user_profile's pool and interleave + user_sampling's sampled users + both switches
    set("TF_AFD_USERS", "4");
    set("TF_AFD_POOL_TOKENS", "720896");
    set("TF_AFD_USERS_INTERLEAVE", "1");
    set("TF_GLM_SAMPLING", "1");
    set("TF_AFD_USERS_SAMPLING", "1");
    set("TF_AFD_USERS_DRAFTS", "1");
    set("TF_AFD_USERS_SHARED", "1");
    const m = try served.plan(a, auto4, cfg);
    try eq(true, m.users_drafts);
    try eq(true, m.users_shared);
    try eq(true, m.sampled);
    try eq(@as(usize, 720896), m.pool);
    try t.expect(m.interleave != null);
    // one user: each switch still needs users
    set("TF_AFD_USERS", null);
    set("TF_AFD_POOL_TOKENS", null);
    set("TF_AFD_USERS_INTERLEAVE", null);
    set("TF_AFD_USERS_SAMPLING", null);
    try t.expectError(error.UsersDraftsNeedsUsers, served.plan(a, one, cfg));
    set("TF_AFD_USERS_DRAFTS", null);
    try t.expectError(error.UsersSharedNeedsUsers, served.plan(a, one, cfg));
    std.debug.print("draft_shared plan: both switches together at USERS 2 and 4 (and with pool + interleave + user_sampling); each alone its own plan; refusals each switch's own\n", .{});
}

// ---- the recorded real users backend (no numerics) ---------------------------------------------------------------

/// The attention node recorded with its MTP head (user_drafts_test.zig's MRec: the tests' fixture cut to 5 layers, the
/// MTP layer kept; TF_AFD_MTP's provider), each exchange's header kept as (layer, rows).
const MRec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: glm.model.Model,
    w: glm.views.Weights,
    e: *glm.engine.Engine,
    a: glm.afd_attn.Afd,
    at: glm.afd_engine.Attn,
    space: glm.trace.Space,
    tr: glm.trace.Trace,
    f: glm.fwd.Fwd,
    region: []align(4096) u8,
    status: []u32,
    node: cuda.xchg.Node,
    posts: usize = 0,

    fn init(r: *MRec) !void {
        r.posts = 0;
        r.parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
        r.m = try glm.model.Model.parse(gpa, r.parsed.value.object.get("config_json").?.string, 1, 0);
        const mtp = r.m.mtp_layers;
        try r.m.cut(5);
        r.m.mtp_layers = if (mtp > 0) mtp else 1;
        r.w = try glm.views.rank(gpa, &r.m);
        r.e = try glm.engine.Engine.create(gpa, &r.m, &r.w, 2560, 8, 64, false);
        r.a = try glm.afd_attn.Afd.initWith(&r.m, 64, true);
        r.space = .{ .gpa = gpa };
        try glm.views.walk(&r.w, glm.views.Path.init("w"), &r.space, glm.trace.placeAll);
        try r.e.walk(&r.space, glm.trace.placeAll);
        try r.e.walkTmp(&r.space, glm.trace.placeAll);
        r.e.tmp.space = &r.space;
        try r.space.place("b.afd_recv", &r.a.afd_recv);
        r.tr = glm.trace.Trace.init(gpa);
        const settings: proto.Settings = .{ .layout = r.a.layout, .max_rows = 8, .ports = 1 };
        r.region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast((proto.Region{ .slot = settings.slotBytes() }).total()));
        @memset(r.region, 0);
        r.status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(r.status, 0);
        r.node = try cuda.xchg.Node.init(gpa, io, .attn, .{ .settings = settings, .warn_failures = true }, .{ .region = r.region, .region_dev = @intFromPtr(r.region.ptr), .status = r.status, .status_dev = @intFromPtr(r.status.ptr) });
        r.a.wire = .{ .node = &r.node, .host = .{ .ctx = r, .post = post, .take = take } };
        r.f = glm.fwd.Fwd.recording(&r.m, &r.tr);
        r.at = try glm.afd_engine.Attn.init(r.e, &r.a);
    }

    fn deinit(r: *MRec) void {
        r.node.deinit();
        gpa.free(r.region);
        gpa.free(r.status);
        r.tr.deinit();
        r.space.deinit();
        r.e.destroy(gpa);
        r.w.deinit(gpa);
        r.parsed.deinit();
    }

    fn post(p: *anyopaque, _: *cuda.xchg.Node, h: *const [proto.HDR_BYTES]u8, pos: usize, n: usize) !void {
        const r: *MRec = @ptrCast(@alignCast(p));
        const hdr = try proto.Header.read(h);
        if (hdr.layer != @as(i32, @intCast(pos)) or hdr.rows != @as(i32, @intCast(n))) return error.HeaderNotPos;
        r.posts += 1;
    }

    fn take(_: *anyopaque, _: *cuda.xchg.Node, _: usize, _: usize) !void {}
};

/// Draws by the slot whose logits row is read: in a round, 300 + 7·slot + the round's index (a shared round's row i of
/// the decode logits is the round's row i; the per-row path's are each slot's own saved row); in a prompt pass or a
/// lone drafted loop, 5 at probability 1 (each draft drawn again by the verify: all kept, as user_drafts_test's Picks).
const SlotPicks = struct {
    space: *const Space,
    rows: []const Row = &.{},
    lone: bool = true,
    round: usize = 0,

    fn token(slot: u8, k: usize) u32 {
        return 300 + 7 * @as(u32, slot) + @as(u32, @intCast(k));
    }

    fn pick(ctx: *anyopaque, logits: T, _: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const p: *SlotPicks = @ptrCast(@alignCast(ctx));
        if (probs) |x| @memset(x, 1.0);
        if (p.lone) {
            @memset(out, 5);
            return;
        }
        const nm = p.space.name(logits.ptr) orelse return error.UnnamedLogits;
        const slot: u8 = if (std.mem.eql(u8, nm.path, "b.logits"))
            p.rows[@intCast(nm.off / (logits.shape[1] * logits.dtype.size()))].key.slot
        else if (std.mem.startsWith(u8, nm.path, "users.row") and nm.path.len > 9)
            nm.path[9] - '0'
        else
            return error.UnknownLogits;
        @memset(out, token(slot, p.round));
    }
};

/// A final-normed hidden row by its owner: the slot whose row it is and that slot's count of such rows so far.
const Tag = struct { slot: u8, n: u32 };

/// Whose rows the next final-norm launches write: one slot's (a prompt pass, a drafted loop), a shared round's (one
/// launch, row j the round's row j) or the per-row path's (launch k the round's row k, one row each).
const Phase = union(enum) { one: u8, shared: []const Row, rows: []const Row };

/// The recorded trace's hidden rows: every row the final norm (`_rmsnorm` with W = w.norm, the main model's only use of
/// it) writes gets its owner's tag; `torch.copy_` and `torch.clone` of hidden-wide bf16 rows carry the tags (the
/// backlogs, last_hidden). So a backlog row's tag says whose final-normed row it holds.
const Shadow = struct {
    space: *const Space,
    d: usize,
    map: std.AutoHashMap(u64, Tag),
    made: [4]u32 = @splat(0),
    phase: Phase = .{ .one = 255 },
    launches: usize = 0,
    unexpected: usize = 0,
    /// each slot's row's tag in the round just fed
    own: [4]?Tag = @splat(null),

    fn feed(s: *Shadow, events: []const Event) !void {
        for (events) |e| switch (e.op) {
            .triton => if (std.mem.eql(u8, e.name, "_rmsnorm")) try s.norm(e),
            .ext => if (e.vals.len >= 2) {
                if (std.mem.eql(u8, e.name, "torch.copy_")) try s.copy(e.vals[0], e.vals[1]);
                if (std.mem.eql(u8, e.name, "torch.clone")) try s.copy(e.vals[1], e.vals[0]);
            },
            else => {},
        };
    }

    fn norm(s: *Shadow, e: Event) !void {
        var w: u64 = 0;
        var o: u64 = 0;
        var stride: usize = 0;
        for (e.args) |a| {
            if (std.mem.eql(u8, a.name, "W")) w = a.value.ptr.addr;
            if (std.mem.eql(u8, a.name, "OUT")) o = a.value.ptr.addr;
            if (std.mem.eql(u8, a.name, "o_stride")) stride = switch (a.value) {
                .i32 => |v| @intCast(v),
                .u64 => |v| @intCast(v),
                else => return error.ShadowStride,
            };
        }
        const wn = s.space.name(w) orelse return;
        if (!std.mem.eql(u8, wn.path, "w.norm")) return;
        for (0..e.grid[0]) |j| {
            const slot: u8 = switch (s.phase) {
                .one => |x| x,
                .shared => |rs| if (s.launches == 0 and j < rs.len) rs[j].key.slot else 255,
                .rows => |rs| if (e.grid[0] == 1 and s.launches < rs.len) rs[s.launches].key.slot else 255,
            };
            var tag: Tag = .{ .slot = slot, .n = 0 };
            if (slot < 4) {
                s.made[slot] += 1;
                tag.n = s.made[slot];
                if (s.phase != .one) s.own[slot] = tag;
            } else s.unexpected += 1;
            try s.map.put(o + j * stride * 2, tag);
        }
        s.launches += 1;
    }

    fn copy(s: *Shadow, dv: glm.trace.Val, sv: glm.trace.Val) !void {
        const dst = switch (dv) {
            .t => |x| x,
            else => return,
        };
        const src = switch (sv) {
            .t => |x| x,
            else => return,
        };
        if (dst.dims != 2 or src.dims != 2 or dst.shape[1] != s.d or src.shape[1] != s.d or dst.dtype != .bf16) return;
        for (0..dst.shape[0]) |j| {
            const to = dst.ptr + j * dst.stride[0] * 2;
            if (s.map.get(src.ptr + j * src.stride[0] * 2)) |tag| try s.map.put(to, tag) else _ = s.map.remove(to);
        }
    }
};

const Mode = enum { shared, rows };

/// One recorded users backend with user_drafts's drafts (and shared_forward's stages when `shared`, shared_pool's pool when `pool`),
/// the slot-aware picker and the shadow; every call's launches fed to the shadow under its phase.
const Rig = struct {
    r: MRec = undefined,
    u: glm.afd_users.Users = undefined,
    pool: ?P.Pool = null,
    picks: SlotPicks = undefined,
    shadow: Shadow = undefined,
    n: u8 = 0,

    const Opt = struct { mode: Mode, drafts: bool = true, pool: bool = false, cap: usize = glm.afd_users_drafts.rows };

    fn init(x: *Rig, n: u8, o: Opt) !void {
        x.n = n;
        x.pool = null;
        try x.r.init();
        errdefer x.r.deinit();
        x.u = try glm.afd_users.Users.init(gpa, &x.r.at, &x.r.f, n);
        errdefer x.u.deinit(gpa);
        if (o.drafts) try x.u.useDrafts(.{ .depth = .{ .most = 3, .fixed = true } }, o.cap);
        if (o.mode == .shared) try x.u.useShared();
        if (o.pool) {
            P.detachTokens(&x.r.e.st);
            x.pool = try P.Pool.init(gpa, &x.r.m, 44 * P.block, x.r.e.buf.rows, x.r.e.st.index != null, x.r.e.st.capacity);
            try x.pool.?.walk(&x.r.space, glm.trace.placeAll);
            try x.u.usePool(&x.pool.?);
        }
        try x.u.walk(&x.r.space, glm.trace.placeAll);
        x.picks = .{ .space = &x.r.space };
        x.r.tr.picker = .{ .ctx = &x.picks, .pick = SlotPicks.pick };
        x.shadow = .{ .space = &x.r.space, .d = x.r.m.hidden, .map = .init(gpa) };
    }

    fn deinit(x: *Rig) void {
        x.shadow.map.deinit();
        x.u.deinit(gpa);
        if (x.pool) |*p| p.deinit(gpa);
        x.r.deinit();
    }

    /// The launches recorded so far, fed to the shadow under the current phase; the trace emptied.
    fn sync(x: *Rig) !void {
        try x.r.f.flush();
        try x.shadow.feed(x.r.tr.events.items);
        x.r.tr.reset();
    }

    fn state(x: *Rig, slot: u8) *glm.buffers.State {
        return if (slot == 0) &x.r.e.st else &x.u.states[slot - 1];
    }

    /// A claim (shared_pool's region first) and the slot's prompt pass under `how` (user_drafts's tracking).
    fn prefill(x: *Rig, key: Key, ids: []const u32, max_new: usize, how: glm.afd_users_drafts.Track) !u32 {
        try x.sync();
        x.u.track(key.slot, how);
        if (x.pool != null) try x.u.reserve(key.slot, ids.len, max_new);
        x.picks.lone = true;
        x.shadow.phase = .{ .one = key.slot };
        const first = try x.u.prefill(key, ids);
        try x.sync();
        return first;
    }

    /// One users round (the production entry, Users.round), the round's index for the picker.
    fn round(x: *Rig, rows: []const Row, out: []u32, k: usize) !void {
        try x.sync();
        x.picks.lone = false;
        x.picks.rows = rows;
        x.picks.round = k;
        x.shadow.phase = if (x.u.shared()) .{ .shared = rows } else .{ .rows = rows };
        x.shadow.launches = 0;
        x.shadow.own = @splat(null);
        defer x.sync() catch {};
        try x.u.round(rows, out);
    }

    fn drafted(x: *Rig, key: Key, pending: u32, count: usize, out: []u32, hook: *const glm.afd_users.Users.DraftHook) !glm.afd_users.Users.DraftRan {
        try x.sync();
        x.picks.lone = true;
        x.shadow.phase = .{ .one = key.slot };
        defer x.sync() catch {};
        return x.u.drafted(key, pending, count, out, hook);
    }

    /// The slot's backlog: each pair's hidden-row tag (254: a row the shadow never saw written) and token.
    fn backlog(x: *Rig, slot: u8, tags: []Tag, toks: []u32) usize {
        const b = &x.u.drafts.?.backlog[slot];
        for (0..b.n) |r| {
            tags[r] = x.shadow.map.get(b.hidden.ptr + r * x.r.m.hidden * 2) orelse .{ .slot = 254, .n = 0 };
            toks[r] = b.tokens[r];
        }
        return b.n;
    }
};

/// The lone driver's hook in small: after `yield_after` rounds the loop yields with the round's kept rows carried
/// (user_drafts_test.zig's Yield); `stop_after` ends the loop without a carry (a cancel's form).
const Yield = struct {
    u: *glm.afd_users.Users,
    key: Key,
    yield_after: usize = 1,
    stop_after: usize = std.math.maxInt(usize),
    rounds: usize = 0,
    fn kept(p: *anyopaque, tokens: []const u32, _: usize, keep: usize) anyerror!bool {
        const y: *Yield = @ptrCast(@alignCast(p));
        y.rounds += 1;
        if (y.rounds >= y.stop_after) return true;
        if (y.rounds < y.yield_after or tokens.len != keep) return false;
        try y.u.carry(y.key, tokens);
        return true;
    }
    fn hook(y: *Yield) glm.afd_users.Users.DraftHook {
        return .{ .ctx = y, .kept = kept };
    }
};

/// The rows the slot's MTP head lacks: its State's rows less what its cache absorbed (held draft rows aside).
fn lacking(st: anytype) usize {
    return st.pos - (st.mtp_len - st.mtp_drafted);
}

fn keyOf(slot: usize) Key {
    return .{ .slot = @intCast(slot), .generation = 1 };
}

fn promptFor(slot: usize, buf: *[16]u32) []const u32 {
    const len = 10 + slot;
    for (buf[0..len], 0..) |*v, i| v.* = @intCast(1000 + 10 * slot + i);
    return buf[0..len];
}

// ---- 2. the users backend takes both ----------------------------------------------------------------------------

test "draft_shared the users backend takes user_drafts's drafts and shared_forward's stages together, in either order; bytes = off + backlogs + stages = the users forecast with both" {
    const served = @import("afd_attn_users.zig");
    var cases: usize = 0;
    for ([_]u8{ 2, 4 }) |n| for ([_]bool{ false, true }) |shared_first| {
        var r: MRec = undefined;
        try r.init();
        defer r.deinit();
        var u = try glm.afd_users.Users.init(gpa, &r.at, &r.f, n);
        defer u.deinit(gpa);
        const off = try u.bytes();
        if (shared_first) {
            try u.useShared();
            try u.useDrafts(.{}, glm.afd_users_drafts.rows);
        } else {
            try u.useDrafts(.{}, glm.afd_users_drafts.rows);
            try u.useShared();
        }
        try t.expect(u.shared());
        try t.expect(u.drafts != null);
        const on = try u.bytes();
        const add = served.draftBytes(&r.m, n, true) + served.sharedBytes(&r.m, n, true);
        try eq(off.states, on.states);
        try eq(off.rows + add, on.rows);
        try eq(off.total + add, on.total);
        var x = off;
        served.withDrafts(&x, &r.m, n, true);
        served.withShared(&x, &r.m, n, true);
        try eq(on.rows, x.rows);
        try eq(on.total, x.total);
        cases += 1;
    };
    std.debug.print("draft_shared bytes: both on = off + backlogs + stages = the forecast with both, {d}/4 (USERS 2 and 4, either order)\n", .{cases});
}

// ---- 3. the backlog, directly ------------------------------------------------------------------------------------

/// What a run leaves: per slot its backlog's tags and tokens, and its own round rows' tags and draws.
const Seen = struct {
    tags: [4][64]Tag = undefined,
    toks: [4][64]u32 = undefined,
    len: [4]usize = @splat(0),
    own: [4][16]Tag = undefined,
    drawn: [4][16]u32 = undefined,
};

/// Every slot in [0, n) (or `only` alone) claims and runs a tracked prompt pass, then `rounds` rounds of all of them
/// together (or of `only` alone), in physical slot order (the users backend's order).
fn backlogRun(n: u8, mode: Mode, only: ?u8, pool: bool, rounds: usize) !Seen {
    var g: Rig = .{};
    try g.init(n, .{ .mode = mode, .pool = pool });
    defer g.deinit();
    var seen: Seen = .{};
    var pending: [4]u32 = undefined;
    var buf: [16]u32 = undefined;
    for (0..n) |s| {
        if (only) |o| if (o != s) continue;
        pending[s] = try g.prefill(keyOf(s), promptFor(s, &buf), 64, .seed);
    }
    for (0..rounds) |k| {
        var rows: [4]Row = undefined;
        var m: usize = 0;
        for (0..n) |s| {
            if (only) |o| if (o != s) continue;
            rows[m] = .{ .key = keyOf(s), .token = pending[s], .position = g.u.position(keyOf(s)) };
            m += 1;
        }
        var out: [4]u32 = undefined;
        try g.round(rows[0..m], out[0..m], k);
        for (rows[0..m], out[0..m]) |row, tok| {
            const s = row.key.slot;
            seen.own[s][k] = g.shadow.own[s] orelse return error.NoFinalRow;
            seen.drawn[s][k] = tok;
            try eq(SlotPicks.token(s, k), tok);
            pending[s] = tok;
        }
    }
    for (0..n) |s| seen.len[s] = g.backlog(@intCast(s), &seen.tags[s], &seen.toks[s]);
    try eq(@as(usize, 0), g.shadow.unexpected);
    return seen;
}

/// Each tracked slot's backlog after `rounds` rounds: 1 + rounds pairs, its last `rounds` exactly its own rows and
/// draws (first row and last alike), = its solo twin's (the slot alone, same path).
fn expectOwn(a: *const Seen, solo: *const Seen, s: usize, rounds: usize) !void {
    errdefer std.debug.print("draft_shared backlog slot {d}: tags {any}\n", .{ s, a.tags[s][0..a.len[s]] });
    try eq(1 + rounds, a.len[s]);
    for (0..rounds) |k| {
        try eq(@as(u8, @intCast(s)), a.tags[s][1 + k].slot);
        try eq(a.own[s][k], a.tags[s][1 + k]);
        try eq(a.drawn[s][k], a.toks[s][1 + k]);
    }
    try eq(@as(u8, @intCast(s)), a.tags[s][0].slot); // the seed: its prompt pass's last row
    try t.expectEqualSlices(Tag, solo.tags[s][0..solo.len[s]], a.tags[s][0..a.len[s]]);
    try t.expectEqualSlices(u32, solo.toks[s][0..solo.len[s]], a.toks[s][0..a.len[s]]);
}

test "draft_shared backlog, the per-row path (user_drafts alone): every tracked slot carries exactly its own final-normed rows and draws, the round's first row as its last, = its solo twin's; USERS=2 and 4, the pool too" {
    var checked: usize = 0;
    for ([_]u8{ 2, 4 }) |n| for ([_]bool{ false, true }) |pool| {
        if (n == 2 and pool) continue;
        const rows = try backlogRun(n, .rows, null, pool, 5);
        for (0..n) |s| {
            const solo = try backlogRun(n, .rows, @intCast(s), pool, 5);
            errdefer std.debug.print("draft_shared per-row C{d} pool={} slot {d}\n", .{ n, pool, s });
            try expectOwn(&rows, &solo, s, 5);
            checked += 1;
        }
    };
    std.debug.print("draft_shared backlog per-row path: {d}/10 tracked slots carry exactly their own rows and draws (5 rounds, = solo)\n", .{checked});
}

test "draft_shared backlog through shared rounds (both switches on): every tracked slot carries exactly its own row of the shared head and its draw, = the per-row path's = its solo twin's; USERS=2 and 4, the pool too" {
    var checked: usize = 0;
    for ([_]u8{ 2, 4 }) |n| for ([_]bool{ false, true }) |pool| {
        if (n == 2 and pool) continue;
        const shared = try backlogRun(n, .shared, null, pool, 5);
        const rows = try backlogRun(n, .rows, null, pool, 5);
        for (0..n) |s| {
            const solo = try backlogRun(n, .shared, @intCast(s), pool, 5);
            errdefer std.debug.print("draft_shared shared C{d} pool={} slot {d}\n", .{ n, pool, s });
            try expectOwn(&shared, &solo, s, 5);
            try t.expectEqualSlices(Tag, rows.tags[s][0..rows.len[s]], shared.tags[s][0..shared.len[s]]);
            try t.expectEqualSlices(u32, rows.toks[s][0..rows.len[s]], shared.toks[s][0..shared.len[s]]);
            checked += 1;
        }
    };
    std.debug.print("draft_shared backlog shared rounds: {d}/10 tracked slots = their own rows and draws = the per-row path's = solo (5 rounds)\n", .{checked});
}

// ---- 4. a tracked slot's life at the device: shared -> lone -> shared -> lone, and lone -> shared -> lone -----------

const Life = struct {
    tags: [64]Tag = undefined,
    toks: [64]u32 = undefined,
    len: usize = 0,
    pos: usize = 0,
    flushes: u64 = 0,
    yielded: usize = 0,
};

/// user_drafts's recorded life of a tracked slot, at USERS=n (+ pool) with shared_forward on (`mode` shared) or off: its prompt
/// pass (`how` seed: shared rounds first; lone: a drafted loop at once), three rounds with every other slot, a drafted
/// loop that yields after its first round (its kept rows carried), five rounds (an 8-pair backlog flushes), then the
/// resume, a drafted loop to its count. Every step: the backlog's length = the rows its MTP head lacks, every pair the
/// slot's own row (the shadow), the rounds' pairs its own rows and draws.
fn life(n: u8, mode: Mode, pool: bool, tracked: u8, how: glm.afd_users_drafts.Track) !Life {
    var g: Rig = .{};
    try g.init(n, .{ .mode = mode, .pool = pool, .cap = 8 });
    defer g.deinit();
    var buf: [16]u32 = undefined;
    var pending: [4]u32 = undefined;
    for (0..n) |s| pending[s] = try g.prefill(keyOf(s), promptFor(s, &buf), 400, if (s == tracked) how else .off);
    const kt = keyOf(tracked);
    const st = g.state(tracked);
    var tags: [64]Tag = undefined;
    var toks: [64]u32 = undefined;
    var k: usize = 0;
    var out: [9]u32 = undefined;
    var yielded: usize = 0;
    if (how == .lone) { // lone -> shared: the drafted loop from the opening, yielding after its first round
        try t.expect(!g.u.carried(kt));
        var y: Yield = .{ .u = &g.u, .key = kt };
        const hk = y.hook();
        const ran = try g.drafted(kt, pending[tracked], 9, &out, &hk);
        try eq(@as(u64, 1), ran.rounds);
        pending[tracked] = out[ran.tokens - 1];
        try t.expect(g.u.carried(kt));
        try eq(lacking(st), g.u.backlogLen(kt));
        yielded += 1;
    }
    const phases = [_]usize{ 3, 5 };
    for (phases, 0..) |count, ph| {
        for (0..count) |_| {
            var rows: [4]Row = undefined;
            for (0..n) |s| rows[s] = .{ .key = keyOf(s), .token = pending[s], .position = g.u.position(keyOf(s)) };
            var o: [4]u32 = undefined;
            try g.round(rows[0..n], o[0..n], k);
            k += 1;
            for (0..n) |s| pending[s] = o[s];
            try eq(lacking(st), g.u.backlogLen(kt));
            const len = g.backlog(tracked, &tags, &toks);
            try t.expect(len >= 1);
            for (tags[0..len]) |tag| try eq(tracked, tag.slot); // every pair the slot's own row, never a peer's
            try eq(g.shadow.own[tracked].?, tags[len - 1]); // this round's pair: its own row and draw
            try eq(o[tracked], toks[len - 1]);
        }
        if (ph == 0) { // -> lone: a drafted loop that yields after one round (an arrival), its kept rows carried
            var y: Yield = .{ .u = &g.u, .key = kt };
            const hk = y.hook();
            const ran = try g.drafted(kt, pending[tracked], 9, &out, &hk);
            try eq(@as(u64, 1), ran.rounds);
            try eq(@as(u64, 3), ran.accepted);
            pending[tracked] = out[ran.tokens - 1];
            try eq(lacking(st), g.u.backlogLen(kt));
            try t.expect(g.u.carried(kt));
            yielded += 1;
        }
    }
    try t.expect(g.u.drafts.?.backlog[tracked].flushes >= 1); // a full 8-pair backlog absorbed in the shared phase
    var res: Life = .{ .yielded = yielded, .flushes = g.u.drafts.?.backlog[tracked].flushes };
    res.len = g.backlog(tracked, &res.tags, &res.toks);
    // the resume: the carried pairs absorbed (all but the last), the loop to its count from the pending token
    var none: Yield = .{ .u = &g.u, .key = kt, .yield_after = std.math.maxInt(usize) };
    const hk2 = none.hook();
    const before = st.pos;
    var out2: [5]u32 = undefined;
    const ran2 = try g.drafted(kt, pending[tracked], 5, &out2, &hk2);
    try eq(@as(usize, 5), ran2.tokens);
    try eq(before + 4, st.pos);
    try eq(@as(usize, 0), g.shadow.unexpected);
    res.pos = st.pos;
    return res;
}

test "draft_shared a tracked slot's life at the device with both switches on: shared -> lone -> shared -> lone and lone -> shared -> lone, its backlog in step and every pair its own row, = user_drafts alone's; USERS=2 and 4 + pool, the tracked slot first and last" {
    var lives: usize = 0;
    for ([_]u8{ 2, 4 }) |n| for ([_]glm.afd_users_drafts.Track{ .seed, .lone }) |how| for ([_]u8{ 0, n - 1 }) |tracked| {
        const pool = n == 4;
        errdefer std.debug.print("draft_shared life C{d} pool={} {t} slot {d}\n", .{ n, pool, how, tracked });
        const a = try life(n, .shared, pool, tracked, how);
        const b = try life(n, .rows, pool, tracked, how);
        try t.expectEqualSlices(Tag, b.tags[0..b.len], a.tags[0..a.len]);
        try t.expectEqualSlices(u32, b.toks[0..b.len], a.toks[0..a.len]);
        try eq(b.pos, a.pos);
        try eq(b.flushes, a.flushes);
        try eq(b.yielded, a.yielded);
        lives += 1;
    };
    std.debug.print("draft_shared device life: {d}/8 (USERS 2 and 4+pool, seed and lone openings, tracked first and last): shared rounds = user_drafts alone's rows, pairs and positions\n", .{lives});
}

// ---- 5. cancel and failure at the device -----------------------------------------------------------------------------

fn yes(ctx: ?*anyopaque) bool {
    const v: *bool = @ptrCast(@alignCast(ctx.?));
    return v.*;
}

test "draft_shared cancel and failure in a shared round with tracked rows: a cancelled row adds no pair and keeps its position, its peer's pair is its own; a failed round adds nothing, is sticky and refuses a drafted loop before any launch; release clears" {
    var g: Rig = .{};
    try g.init(4, .{ .mode = .shared, .pool = true });
    defer g.deinit();
    var buf: [16]u32 = undefined;
    var pending: [4]u32 = undefined;
    for (0..4) |s| pending[s] = try g.prefill(keyOf(s), promptFor(s, &buf), 64, .seed);
    var flag = true;
    var rows: [4]Row = undefined;
    for (0..4) |s| rows[s] = .{ .key = keyOf(s), .token = pending[s], .position = g.u.position(keyOf(s)) };
    rows[0].cancel_ctx = &flag;
    rows[0].cancel_check = yes;
    var out: [4]u32 = undefined;
    try g.round(&rows, &out, 0);
    try eq(rows[0].position, g.u.position(keyOf(0)));
    try eq(@as(usize, 1), g.u.backlogLen(keyOf(0)));
    var tags: [64]Tag = undefined;
    var toks: [64]u32 = undefined;
    for (1..4) |s| {
        try eq(rows[s].position + 1, g.u.position(keyOf(s)));
        const len = g.backlog(@intCast(s), &tags, &toks);
        try eq(@as(usize, 2), len);
        try eq(g.shadow.own[s].?, tags[1]);
        try eq(out[s], toks[1]);
        pending[s] = out[s];
    }
    // a transport failure at the second MoE layer of the next shared round: nothing drawn or counted, sticky
    var fault: glm.trace.Trace.Fault = .{ .name = "afd.exchange", .skip = 1 };
    g.r.tr.test_fault = &fault;
    for (1..4) |s| rows[s] = .{ .key = keyOf(s), .token = pending[s], .position = g.u.position(keyOf(s)) };
    try t.expect(if (g.round(rows[1..4], out[1..4], 1)) |_| false else |_| true);
    g.r.tr.test_fault = null;
    for (1..4) |s| {
        try eq(rows[s].position, g.u.position(keyOf(s)));
        try eq(@as(usize, 2), g.u.backlogLen(keyOf(s)));
    }
    try g.sync();
    var none: Yield = .{ .u = &g.u, .key = keyOf(1), .yield_after = std.math.maxInt(usize) };
    const hk = none.hook();
    var o: [5]u32 = undefined;
    g.picks.lone = true;
    try t.expect(if (g.u.drafted(keyOf(1), pending[1], 5, &o, &hk)) |_| false else |_| true);
    try g.r.f.flush();
    try eq(@as(usize, 0), g.r.tr.events.items.len); // refused before any launch
    try t.expect(if (g.u.round(rows[1..4], out[1..4])) |_| false else |_| true);
    try g.r.f.flush();
    try eq(@as(usize, 0), g.r.tr.events.items.len);
    for (0..4) |s| g.u.release(keyOf(s));
    for (0..4) |s| {
        try eq(@as(usize, 0), g.u.backlogLen(keyOf(s)));
        try t.expect(!g.u.carried(keyOf(s)));
    }
    try eq(g.pool.?.blocks, g.pool.?.freeBlocks());
    std.debug.print("draft_shared device cancel/failure in a shared round: cancelled row no pair, peers own pairs 3/3; failed round sticky, nothing counted, drafted and a round refused before any launch, release clean, pool back\n", .{});
}

test "draft_shared cancel and failure inside a lone drafted loop with both switches on: a stopped loop releases cleanly and its peers' next shared round adds their own pairs; a failed loop leaves no carried state, the failure is sticky, release clears" {
    var g: Rig = .{};
    try g.init(4, .{ .mode = .shared, .pool = true });
    defer g.deinit();
    var buf: [16]u32 = undefined;
    var pending: [4]u32 = undefined;
    for (0..4) |s| pending[s] = try g.prefill(keyOf(s), promptFor(s, &buf), 64, .seed);
    // slot 3's lone loop is cancelled after its first round (the hook ends it without a carry), then released
    var stop: Yield = .{ .u = &g.u, .key = keyOf(3), .stop_after = 1 };
    const hs = stop.hook();
    var out: [9]u32 = undefined;
    const ran = try g.drafted(keyOf(3), pending[3], 9, &out, &hs);
    try eq(@as(u64, 1), ran.rounds);
    g.u.release(keyOf(3));
    try eq(@as(usize, 0), g.u.backlogLen(keyOf(3)));
    try t.expect(!g.u.carried(keyOf(3)));
    // its peers' next shared round: each adds exactly its own row and draw
    var rows: [3]Row = undefined;
    for (0..3) |s| rows[s] = .{ .key = keyOf(s), .token = pending[s], .position = g.u.position(keyOf(s)) };
    var o3: [3]u32 = undefined;
    try g.round(&rows, &o3, 0);
    var tags: [64]Tag = undefined;
    var toks: [64]u32 = undefined;
    for (0..3) |s| {
        const len = g.backlog(@intCast(s), &tags, &toks);
        try eq(@as(usize, 2), len);
        try eq(g.shadow.own[s].?, tags[1]);
        try eq(o3[s], toks[1]);
        pending[s] = o3[s];
    }
    // slot 2's lone loop fails at an exchange inside its first verify: no carried state is left, the failure sticky
    var fault: glm.trace.Trace.Fault = .{ .name = "afd.exchange", .skip = 0 };
    g.r.tr.test_fault = &fault;
    var none: Yield = .{ .u = &g.u, .key = keyOf(2), .yield_after = std.math.maxInt(usize) };
    const hn = none.hook();
    try t.expect(if (g.drafted(keyOf(2), pending[2], 9, &out, &hn)) |_| false else |_| true);
    g.r.tr.test_fault = null;
    try t.expect(fault.hits >= 1);
    try t.expect(!g.u.carried(keyOf(2)));
    try eq(@as(usize, 0), g.u.backlogLen(keyOf(2)));
    try t.expect(if (g.u.round(rows[0..2], o3[0..2])) |_| false else |_| true); // sticky: no peer round after it
    for (0..3) |s| g.u.release(keyOf(s));
    for (0..4) |s| try eq(@as(usize, 0), g.u.backlogLen(keyOf(s)));
    try eq(g.pool.?.blocks, g.pool.?.freeBlocks());
    std.debug.print("draft_shared device cancel/failure in a lone drafted loop: stopped loop released clean, peers' shared round own pairs 3/3; failed loop no carried state ({d} fault hit), sticky, release clean, pool back\n", .{fault.hits});
}

// ---- 6. through the real lane host: the stand-in's rounds go through the real round scheduler -------------------------

/// The served capacity at the 262,144 window (concurrent_users's), the backend's ContextFull bound.
const cap: usize = 262208;
/// The stand-in's mask width (a schema row's draws); lanes.fake.next's plain draws stay below 97.
const width: usize = 2048;
const words = (width + 31) / 32;

/// FAKE matcher (structured_users's Script, joined_profile's copy): step i allows `steps[i]`; terminated once every step is accepted.
const Script = struct {
    rules: lanes.grammar.Rules = undefined,
    steps: []const []const u32,
    accepted: usize = 0,
    closes: usize = 0,
    fn init(x: *Script, w: usize) void {
        x.rules = .{ .ptr = x, .width = w, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn of(p: *anyopaque) *Script {
        return @ptrCast(@alignCast(p));
    }
    fn fill(p: *anyopaque, out: []u32) !void {
        const x = of(p);
        @memset(out, 0);
        if (x.accepted < x.steps.len) for (x.steps[x.accepted]) |tok| {
            out[tok / 32] |= @as(u32, 1) << @as(u5, @intCast(tok % 32));
        };
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const x = of(p);
        if (x.accepted >= x.steps.len or std.mem.indexOfScalar(u32, x.steps[x.accepted], token) == null) return false;
        x.accepted += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) !void {
        const x = of(p);
        if (n > x.accepted) return error.BadRollback;
        x.accepted -= n;
    }
    fn terminated(p: *anyopaque) bool {
        const x = of(p);
        return x.accepted >= x.steps.len;
    }
    fn close(p: *anyopaque) void {
        of(p).closes += 1;
    }
};

/// The lowest id a mask allows (the stand-in's masked draw, as structured_users's recorded adapter's emulator).
fn lowest(mask: []const u32) u32 {
    for (mask, 0..) |w, i| if (w != 0) return @intCast(i * 32 + @ctz(w));
    return 0;
}

/// Holds the engine thread inside its next drafted round or shared round once armed, until opened.
const Gate = struct {
    armed: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(bool) = .init(false),
    open: std.atomic.Value(bool) = .init(false),
    fn at(g: *Gate) void {
        if (!g.armed.load(.acquire)) return;
        g.armed.store(false, .release);
        g.entered.store(true, .release);
        for (0..10_000) |_| {
            if (g.open.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    fn waitEntered(g: *Gate) !void {
        for (0..10_000) |_| {
            if (g.entered.load(.acquire)) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }
};

const Track = enum { off, seed, lone };

/// A slot: its generation, its cache (history), a pass under way (prefill_interleave), its MTP head (user_drafts's stand-in: the
/// rows its cache absorbed, the pairs it owes ending at `last`, `head` for a lone opening) and its request's sampling.
const Slot = struct {
    generation: u64 = 0,
    history: std.ArrayList(u32) = .empty,
    step: ?usize = null,
    track: Track = .off,
    mtp: usize = 0,
    owed: usize = 0,
    last: u32 = 0,
    head: bool = false,
    sampling: ?scheduling.Sampling = null,
};

/// The draw after `history` (its length the drawn token's position): greedy, or keyed by the request's seed.
fn draw(history: []const u32, s: ?scheduling.Sampling) u32 {
    const keyed: ?lanes.Sampling = if (s) |x| .{ .seed = x.seed, .temperature = 1.0 } else null;
    return lanes.fake.next(history, keyed, history.len);
}

fn sameSampling(a: ?scheduling.Sampling, b: ?scheduling.Sampling) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.?.seed == b.?.seed and a.?.temperature == b.?.temperature and a.?.top_k == b.?.top_k and a.?.top_p == b.?.top_p and a.?.min_p == b.?.min_p;
}

/// The stand-in device the real round scheduler (cuda_afd_users_round.run) drives: row ops pure per row, each layer's
/// per-sequence op on the row's own slot (a fingerprint of its history), the MoE per row. Both forwards (the per-row
/// one and shared_forward's shared one) must reach `want`, the row's value by the reference order; a row that does not draws
/// a token no reference has (`mis` puts a shared row's per-sequence part on the next row's slot: the RED form).
const Dev = struct {
    s: *Steps,
    rs: []const Row = &.{},
    own: [4]u64 = @splat(0),
    hidden: [4]u64 = @splat(0),
    gathered: [4]u64 = @splat(0),
    result: [4]u64 = @splat(0),
    committed: [4]bool = @splat(false),
    scratch: std.ArrayList(u32) = .empty,

    const layers = 4;
    const poison: u32 = 4000;

    fn rowOp(x: u64, li: usize) u64 {
        return x *% @as(u64, li + 3) +% 1;
    }
    fn seqOp(x: u64, print: u64, li: usize) u64 {
        return x +% print *% @as(u64, li + 1);
    }
    fn expert(x: u64) u64 {
        return x *% 15 +% 7;
    }
    fn fp(d: *Dev, slot: u8) u64 {
        return std.hash.Wyhash.hash(slot, std.mem.sliceAsBytes(d.s.slots[slot].history.items));
    }
    fn want(d: *Dev, row: Row) u64 {
        var v: u64 = row.token;
        for (0..layers) |li| {
            v = seqOp(rowOp(v, li), d.fp(row.key.slot), li);
            if (isMoe(d, li)) v +%= expert(v);
        }
        return v;
    }
    fn fail(d: *Dev, li: usize) !void {
        if (li != 1) return;
        if (d.s.round_gate) |g| if (d.rs.len >= 2) g.at(); // a round with two or more rows (a shared one with B's)
        if (d.s.fail_now) {
            d.s.failed = true;
            return error.InjectedExchangeFailure;
        }
    }
    pub fn alive(d: *Dev) !void {
        if (d.s.failed) return error.InjectedExchangeFailure;
    }
    pub fn begin(d: *Dev, rows: []const Row) !void {
        d.rs = rows;
    }
    pub fn layerCount(_: *Dev) usize {
        return layers;
    }
    pub fn isMoe(_: *Dev, li: usize) bool {
        return li >= 1;
    }
    // the per-row forward (concurrent_users's)
    pub fn open(d: *Dev, row: Row, _: usize) !void {
        d.own[row.key.slot] = row.token;
    }
    pub fn front(d: *Dev, row: Row, _: usize, li: usize) !void {
        const k = row.key.slot;
        d.own[k] = seqOp(rowOp(d.own[k], li), d.fp(k), li);
    }
    pub fn gather(d: *Dev, row: Row, i: usize) !void {
        d.gathered[i] = d.own[row.key.slot];
    }
    pub fn exchange(d: *Dev, li: usize, n: usize) !void {
        try d.fail(li);
        for (d.gathered[0..n], d.result[0..n]) |x, *o| o.* = expert(x);
    }
    pub fn scatter(d: *Dev, row: Row, i: usize, _: usize) !void {
        d.own[row.key.slot] +%= d.result[i];
    }
    pub fn tail(_: *Dev, _: Row, _: usize) !void {}
    // shared_forward's shared forward
    pub fn shared(d: *Dev) bool {
        return d.s.shared;
    }
    pub fn sharedOpen(d: *Dev, rows: []const Row) !void {
        for (rows, 0..) |row, i| d.hidden[i] = row.token;
    }
    pub fn sharedIn(d: *Dev, rows: []const Row, li: usize) !void {
        for (d.hidden[0..rows.len]) |*h| h.* = rowOp(h.*, li);
    }
    pub fn sharedSeq(d: *Dev, row: Row, i: usize, li: usize) !void {
        const k = if (d.s.mis) d.rs[(i + 1) % d.rs.len].key.slot else row.key.slot;
        d.hidden[i] = seqOp(d.hidden[i], d.fp(k), li);
    }
    pub fn sharedOut(d: *Dev, rows: []const Row, li: usize) !void {
        if (!d.isMoe(li)) return;
        try d.fail(li);
        for (d.hidden[0..rows.len]) |*h| h.* +%= expert(h.*);
    }
    pub fn sharedTail(_: *Dev, _: []const Row) !void {}
    pub fn settle(d: *Dev) !void {
        try d.alive();
    }
    pub fn sample(d: *Dev, row: Row, i: usize) !u32 {
        const v = if (d.s.shared) d.hidden[i] else d.own[row.key.slot];
        if (v != d.want(row)) return poison;
        if (row.mask.len != 0) return lowest(row.mask);
        d.scratch.clearRetainingCapacity();
        try d.scratch.appendSlice(gpa, d.s.slots[row.key.slot].history.items);
        try d.scratch.append(gpa, row.token);
        return draw(d.scratch.items, row.sampling);
    }
    pub fn commit(d: *Dev, row: Row, _: usize) !void {
        try d.s.slots[row.key.slot].history.append(gpa, row.token);
        d.committed[row.key.slot] = true;
    }
};

/// joined_profile's Steps (four_user_profile's pool, prefill_interleave's steps, structured_users's masks, user_drafts's MTP head, user_sampling's sampling), its rounds
/// through the real scheduler over `Dev` (shared_forward's forward when `shared`), its backlog the family's rule (a tracked
/// slot's pair counted at its commit, a full backlog absorbed after the round).
const Steps = struct {
    slots: [4]Slot = @splat(.{}),
    chunks: usize = 3,
    pool: ?*P.Pool = null,
    draft_gate: ?*Gate = null,
    round_gate: ?*Gate = null,
    shared: bool = false,
    mis: bool = false,
    depth: usize = 3,
    backlog: usize = 8,
    fail_round: ?usize = null,
    fail_drafted: ?usize = null,
    fail_now: bool = false,
    failed: bool = false,
    tracks: usize = 0,
    drafted_rounds: usize = 0,
    loops: usize = 0,
    resumed_loops: usize = 0,
    carries: usize = 0,
    flushes: usize = 0,
    stale: usize = 0,
    rounds: usize = 0,
    shared_rounds: usize = 0,
    row_rounds: usize = 0,
    max_rows: usize = 0,
    passes: usize = 0,
    masked_draws: usize = 0,
    sampled_draws: usize = 0,
    configured_sampled: usize = 0,
    sampled_shared: usize = 0,
    cancelled_rows: usize = 0,

    pub const DraftHook = struct {
        ctx: *anyopaque,
        kept: *const fn (ctx: *anyopaque, tokens: []const u32, drafts: usize, keep: usize) anyerror!bool,
    };
    pub const DraftRan = struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, tokens: usize = 0, ns: u64 = 0 };

    fn deinit(s: *Steps) void {
        for (&s.slots) |*sl| sl.history.deinit(gpa);
    }

    pub fn reserve(s: *Steps, slot: u8, prompt_len: usize, max_new: usize) !void {
        const p = s.pool orelse return;
        var moves: [P.max_slots]P.Move = undefined;
        _ = try p.reserve(slot, P.need(prompt_len, max_new), &moves);
    }

    pub fn configureSampling(s: *Steps, slot: u8, sampling: ?scheduling.Sampling) void {
        s.slots[slot].sampling = sampling;
        if (sampling != null) s.configured_sampled += 1;
    }

    fn claim(s: *Steps, key: Key, ids: []const u32) !void {
        if (s.failed) return error.InjectedExchangeFailure;
        const sl = &s.slots[key.slot];
        if (key.generation <= sl.generation) return error.StaleGeneration;
        sl.generation = key.generation;
        sl.history.clearRetainingCapacity();
        try sl.history.appendSlice(gpa, ids);
    }

    fn passEnd(s: *Steps, slot: u8, first: u32) !void {
        const sl = &s.slots[slot];
        if (sl.track == .off) return;
        if (sl.sampling != null) return error.SampledTracked; // user_drafts's rule: a sampled stream never drafts
        sl.mtp = sl.history.items.len - 1;
        sl.owed = 1;
        sl.last = first;
        sl.head = sl.track == .lone;
    }

    pub fn prefill(s: *Steps, key: Key, ids: []const u32) !u32 {
        try s.claim(key, ids);
        const sl = &s.slots[key.slot];
        const first = draw(sl.history.items, sl.sampling);
        try s.passEnd(key.slot, first);
        return first;
    }

    pub fn prefillDeferred(s: *Steps, key: Key, ids: []const u32) !void {
        try s.claim(key, ids);
    }

    pub fn firstMasked(s: *Steps, key: Key, mask: []const u32) !u32 {
        const sl = &s.slots[key.slot];
        if (mask.len == 0) return draw(sl.history.items, sl.sampling);
        s.masked_draws += 1;
        return lowest(mask);
    }

    pub fn checkMasked(_: *Steps) !void {}

    pub fn maskedWidth(_: *const Steps) usize {
        return width;
    }

    pub fn maskedWords(_: *const Steps) usize {
        return words;
    }

    pub fn maskCheck(_: *Steps, mask: []const u32) !void {
        if (mask.len != words) return error.MaskShape;
        for (mask) |w| if (w != 0) return;
        return error.EmptyMask;
    }

    pub fn prefillStep(s: *Steps, key: Key, ids: []const u32) !?u32 {
        if (s.failed) return error.InjectedExchangeFailure;
        const sl = &s.slots[key.slot];
        if (sl.step == null) {
            if (key.generation <= sl.generation) return error.StaleGeneration;
            sl.generation = key.generation;
            sl.history.clearRetainingCapacity();
            sl.step = 0;
            s.passes += 1;
        } else if (key.generation != sl.generation) return error.StaleGeneration;
        const k = sl.step.?;
        sl.step = k + 1;
        if (k + 1 < s.chunks) return null;
        sl.step = null;
        try sl.history.appendSlice(gpa, ids);
        const first = draw(sl.history.items, sl.sampling);
        try s.passEnd(key.slot, first);
        return first;
    }

    pub fn position(s: *Steps, key: Key) usize {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        return s.slots[key.slot].history.items.len;
    }

    pub fn release(s: *Steps, key: Key) void {
        std.debug.assert(s.slots[key.slot].generation == key.generation);
        const sl = &s.slots[key.slot];
        sl.step = null;
        sl.track = .off;
        sl.owed = 0;
        sl.head = false;
        sl.sampling = null;
        if (s.pool) |p| _ = p.release(key.slot);
    }

    pub fn round(s: *Steps, rows: []const Row, out: []u32) !void {
        if (s.failed) return error.InjectedExchangeFailure;
        for (rows) |row| {
            const sl = &s.slots[row.key.slot];
            if (sl.generation != row.key.generation) return error.StaleGeneration;
            if (sl.history.items.len != row.position) return error.PositionMismatch;
            if (sl.step != null) return error.FillingInRound;
            if (!sameSampling(row.sampling, sl.sampling)) return error.RowSamplingNotTheSlots;
            if (row.sampling != null and sl.track != .off) return error.SampledTracked;
        }
        s.rounds += 1;
        s.max_rows = @max(s.max_rows, rows.len);
        if (s.fail_round) |k| s.fail_now = s.rounds == k;
        var d: Dev = .{ .s = s };
        defer d.scratch.deinit(gpa);
        try scheduling.run(Dev, &d, rows, out); // the real scheduler: shared_forward's forward when shared, else the per-row one
        if (s.shared) s.shared_rounds += 1 else s.row_rounds += 1;
        for (rows, out) |row, tok| {
            const sl = &s.slots[row.key.slot];
            if (!d.committed[row.key.slot]) {
                s.cancelled_rows += 1;
                continue;
            }
            if (row.mask.len != 0) s.masked_draws += 1;
            if (row.sampling != null) {
                s.sampled_draws += 1;
                if (rows.len > 1) s.sampled_shared += 1;
            }
            if (sl.track == .off) continue;
            if (sl.head) { // a lone opening that never drafted: untracked from here (the family's `grown`)
                sl.track = .off;
                sl.owed = 0;
                sl.head = false;
                continue;
            }
            if (sl.last != row.token) return error.BacklogOutOfStep;
            sl.owed += 1;
            sl.last = tok;
        }
        for (rows) |row| { // the family's flushFull
            const sl = &s.slots[row.key.slot];
            if (sl.track == .off or sl.owed < s.backlog) continue;
            sl.mtp += sl.owed - 1;
            sl.owed = 1;
            s.flushes += 1;
        }
    }

    pub fn track(s: *Steps, slot: u8, how: Track) void {
        const sl = &s.slots[slot];
        sl.track = how;
        sl.owed = 0;
        sl.head = false;
        if (how != .off) s.tracks += 1;
    }

    pub fn carried(s: *Steps, key: Key) bool {
        const sl = &s.slots[key.slot];
        return sl.generation == key.generation and sl.track != .off and sl.owed >= 1 and !sl.head;
    }

    pub fn carry(s: *Steps, key: Key, kept: []const u32) !void {
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (kept.len != sl.owed or kept[kept.len - 1] != sl.last) return error.CarryOutOfStep;
        s.carries += 1;
    }

    /// mtpWith's shape (user_drafts's stand-in), refused for a sampled slot or a stale carried state.
    pub fn drafted(s: *Steps, key: Key, pending: u32, count: usize, out: []u32, hook: *const DraftHook) !DraftRan {
        if (s.failed) return error.InjectedExchangeFailure;
        const sl = &s.slots[key.slot];
        if (sl.generation != key.generation) return error.StaleGeneration;
        if (sl.sampling != null) return error.SampledDrafted;
        if (sl.track == .off) return error.NotTracked;
        if (count == 0 or count > out.len) return error.Count;
        if (sl.owed == 0 or sl.last != pending or sl.mtp + sl.owed != sl.history.items.len) {
            s.stale += 1;
            return error.StaleMtp;
        }
        s.loops += 1;
        if (!sl.head) s.resumed_loops += 1;
        sl.mtp += sl.owed - 1;
        sl.owed = 1;
        sl.head = false;
        var r: DraftRan = .{};
        out[0] = pending;
        var n: usize = 1;
        var cur = pending;
        var h: std.ArrayList(u32) = .empty;
        defer h.deinit(gpa);
        while (n < count) {
            if (s.draft_gate) |g| g.at();
            s.drafted_rounds += 1;
            if (s.fail_drafted) |k| if (s.drafted_rounds >= k) {
                s.failed = true;
                return error.InjectedExchangeFailure;
            };
            sl.mtp += sl.owed;
            sl.owed = 0;
            const depth = @min(s.depth, count - n);
            var drafts: [8]u32 = undefined;
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, cur);
            for (0..depth) |j| {
                const right = lanes.fake.next(h.items, null, h.items.len);
                drafts[j] = if (h.items.len % 5 == 0) (right + 1) % lanes.fake.vocab else right;
                try h.append(gpa, drafts[j]);
            }
            var sampled: [9]u32 = undefined;
            h.clearRetainingCapacity();
            try h.appendSlice(gpa, sl.history.items);
            try h.append(gpa, cur);
            var keep: usize = 1;
            for (0..depth + 1) |j| {
                sampled[j] = lanes.fake.next(h.items, null, h.items.len);
                if (j == depth or sampled[j] != drafts[j]) break;
                keep += 1;
                try h.append(gpa, drafts[j]);
            }
            try sl.history.append(gpa, cur);
            try sl.history.appendSlice(gpa, drafts[0 .. keep - 1]);
            sl.owed = keep;
            sl.last = sampled[keep - 1];
            r.rounds += 1;
            r.drafted += depth;
            r.accepted += keep - 1;
            const take = @min(keep, count - n);
            @memcpy(out[n .. n + take], sampled[0..take]);
            n += take;
            cur = sampled[keep - 1];
            if (try hook.kept(hook.ctx, sampled[0..take], depth, keep)) break;
        }
        r.tokens = n;
        return r;
    }
};

const Back = users.Users(Steps);

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    message: [96]u8 = @splat(0),

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |x| b.tokens.appendSlice(gpa, x) catch {},
            .finished => |f| {
                b.done = f.reason;
                const n = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..n], f.message[0..n]);
            },
            else => {},
        }
    }

    fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }

    fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
    }

    fn wait(b: *Box) !api.Reason {
        for (0..30_000) |_| {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WaitTimeout;
    }

    fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
};

/// The reference reply: each token the draw of the prompt and every token before it, greedy (the undrafted single-user
/// reply) or keyed by `sampling`'s seed (boot S's seeded reply).
fn expectRef(prompt: []const u32, sampling: ?lanes.Sampling, got: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (got) |x| {
        try eq(lanes.fake.next(history.items, sampling, history.items.len), x);
        try history.append(gpa, x);
    }
}

/// Each switch as plan() decides it from the environment (TF_AFD_USERS=n, TF_AFD_MTP=1, the two switches), so the
/// lane-level tests run what the server would boot: on the base head both on is refused (UsersSharedUndrafted).
const Opts = struct { shared: bool = true, drafts: bool = true, sampled: bool = true, interleave: bool = true, pool: ?*P.Pool = null, draft_reach: usize = cap };

fn planned(n: u8, shared: bool, drafts: bool) !Opts {
    var env: Env = .{};
    try env.take();
    defer env.put();
    const served = @import("afd_attn_serve.zig");
    const cli = @import("tf_glm_afd.zig");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    set("TF_AFD_USERS", if (n == 2) "2" else "4");
    set("TF_AFD_MTP", "1");
    if (drafts) set("TF_AFD_USERS_DRAFTS", "1");
    if (shared) set("TF_AFD_USERS_SHARED", "1");
    const par: [:0]const u8 = if (n == 2) "2" else "4";
    const f = try cli.parse(a, .attn, &.{ "serve", "--dry", "--model", "/not-read", "--", "--drafts", if (drafts) "auto" else "0", "--context", "262144", "--parallel", par, "--temperature", "0" });
    const p = try served.plan(a, f, "{\"max_position_embeddings\":1048576}");
    return .{ .shared = p.users_shared, .drafts = p.users_drafts };
}

/// The users backend over Steps at TF_AFD_USERS=n, the lane core and the real LaneHost as afd_attn_users.run installs
/// them (shared_pool's admission with a pool, prefill_interleave's interleave 1:1, structured_users's structured hook, user_sampling's switch, the lone
/// driver with user_drafts's switch); shared_forward's switch picks the stand-in's forward.
const Served = struct {
    steps: Steps = .{},
    back: Back = undefined,
    cfg: lanes.Config = undefined,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    host: api.LaneHost = undefined,
    gate: Gate = .{},
    scripts: []Script = &.{},
    given: usize = 0,
    up: bool = false,

    fn init(x: *Served, n: u8, o: Opts, scripts: []Script) !void {
        x.steps.pool = o.pool;
        x.steps.shared = o.shared;
        x.scripts = scripts;
        x.back = .{ .steps = &x.steps, .count = n, .capacity = cap, .reach = cap, .interleave = o.interleave, .sampled = o.sampled, .drafts = o.drafts, .draft_reach = o.draft_reach };
        x.cfg = try lanes.Config.init(gpa, x.back.facts(), 1, 0);
        x.core = lanes.Engine.init(gpa, &x.cfg, x.back.backend(), x.clock.clock());
        const admit: ?api.Admit = if (o.pool) |p| .{ .ctx = p, .fits = P.Pool.admitFits } else null;
        x.host = api.LaneHost.init(gpa, io, &x.core, .{ .lanes = n, .context_window = 262144, .drafts = if (o.drafts) "auto" else "off", .greedy_users = true, .users_sampled = o.sampled, .users_drafts = o.drafts, .structures = true, .admit = admit });
        x.host.structured = .{ .ctx = x, .rules = rules };
        if (o.drafts) x.host.lone = x.back.lone(gpa);
        if (o.interleave) x.host.fill_policy = .{ .rounds = 1, .steps = 1 };
    }

    fn start(x: *Served) !void {
        try x.host.start();
        x.up = true;
    }

    fn stop(x: *Served) void {
        if (x.up) x.host.stop();
        x.up = false;
    }

    fn deinit(x: *Served) void {
        x.stop();
        x.core.deinit();
        x.cfg.deinit(gpa);
        x.steps.deinit();
    }

    fn owners(x: *const Served) usize {
        var k: usize = 0;
        for (x.back.slots[0..x.back.count]) |sl| k += @intFromBool(sl.owner != null);
        return k;
    }

    fn rules(ctx: *anyopaque, _: *const api.Request) anyerror!*lanes.grammar.Rules {
        const x: *Served = @ptrCast(@alignCast(ctx));
        if (x.given >= x.scripts.len) return error.NoScript;
        const s = &x.scripts[x.given];
        x.given += 1;
        s.init(width);
        return &s.rules;
    }
};

fn promptOf(comptime n: usize, base: u32) [n]u32 {
    var v: [n]u32 = undefined;
    for (&v, 0..) |*x, i| x.* = base + @as(u32, @intCast(i % 23));
    return v;
}

const pa = promptOf(40, 3);
const pb = promptOf(300, 8);
const pc = promptOf(120, 2);
const pd = promptOf(50, 5);
const json: api.Structure = .{ .kind = .json };
const schema = [_][]const u32{ &.{ 600, 11 }, &.{12}, &.{ 13, 900 }, &.{14}, &.{ 15, 700 }, &.{16}, &.{17}, &.{ 18, 1000 }, &.{19}, &.{20}, &.{ 21, 650 }, &.{22} };
const schema_reply = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 };
const s7: lanes.Sampling = .{ .seed = 7, .temperature = 0.8, .top_k = 0, .top_p = 0.9 };
const s23: lanes.Sampling = .{ .seed = 23, .temperature = 0.6, .top_k = 40, .top_p = 1.0 };

fn fixtureModel() !struct { parsed: std.json.Parsed(std.json.Value), m: glm.model.Model } {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("afd_attn_fixture"), .{});
    errdefer parsed.deinit();
    var m = try glm.model.Model.parse(gpa, parsed.value.object.get("config_json").?.string, 1, 0);
    try m.cut(5);
    m.mtp_layers = 0;
    return .{ .parsed = parsed, .m = m };
}

/// Enough pool blocks for `rs` all at once (four_user_profile's sizing: exactly them).
fn blocksOf(rs: []const api.Request) usize {
    var blocks: usize = 0;
    for (rs) |r| blocks += P.blocksFor(P.need(r.prompt.len, r.max_tokens));
    return blocks;
}

const Reply = struct {
    buf: [256]u32 = undefined,
    n: usize = 0,
    reason: ?api.Reason = null,
    fn got(r: *const Reply) []const u32 {
        return r.buf[0..r.n];
    }
};

const Run = struct {
    out: [4]Reply = @splat(.{}),
    loops: usize = 0,
    resumed_loops: usize = 0,
    drafted_rounds: usize = 0,
    stale: usize = 0,
    rounds: usize = 0,
    shared_rounds: usize = 0,
    row_rounds: usize = 0,
    max_rows: usize = 0,
    passes: usize = 0,
    sampled_draws: usize = 0,
    sampled_shared: usize = 0,
    configured_sampled: usize = 0,
    masked_draws: usize = 0,
    closes: usize = 0,
    runs: u64 = 0,
    resumes: u64 = 0,
    yields: u64 = 0,
    pool_back: bool = false,
    owners: usize = 0,
    failed: bool = false,
};

/// A fresh TF_AFD_USERS=n served path with the pool on (exactly `rs`'s regions): `which` submitted before the host
/// starts (arriving together; one index: its solo run).
fn serve(n: u8, o: Opts, which: []const usize, rs: []const api.Request) !Run {
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    var pool = try P.Pool.init(gpa, &fx.m, blocksOf(rs) * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var boxes: [4]Box = @splat(.{});
    defer for (&boxes) |*b| b.deinit();
    var scripts = [_]Script{.{ .steps = &schema }};
    var x: Served = .{};
    var opts = o;
    opts.pool = &pool;
    try x.init(n, opts, &scripts);
    defer x.deinit();
    const e = x.host.engine();
    for (which) |i| try e.submit(@intCast(i + 1), &rs[i], boxes[i].sink());
    try x.start();
    var run: Run = .{};
    for (which) |i| {
        run.out[i].reason = try boxes[i].wait();
        const got = boxes[i].tokens.items;
        @memcpy(run.out[i].buf[0..got.len], got);
        run.out[i].n = got.len;
    }
    x.stop();
    run.loops = x.steps.loops;
    run.resumed_loops = x.steps.resumed_loops;
    run.drafted_rounds = x.steps.drafted_rounds;
    run.stale = x.steps.stale;
    run.rounds = x.steps.rounds;
    run.shared_rounds = x.steps.shared_rounds;
    run.row_rounds = x.steps.row_rounds;
    run.max_rows = x.steps.max_rows;
    run.passes = x.steps.passes;
    run.sampled_draws = x.steps.sampled_draws;
    run.sampled_shared = x.steps.sampled_shared;
    run.configured_sampled = x.steps.configured_sampled;
    run.masked_draws = x.steps.masked_draws;
    run.closes = scripts[0].closes;
    run.runs = x.back.counts.runs;
    run.resumes = x.back.counts.resumes;
    run.yields = x.back.counts.yields;
    run.pool_back = pool.freeBlocks() == pool.blocks;
    run.owners = x.owners();
    run.failed = x.back.failed != null;
    return run;
}

test "draft_shared lone -> shared -> lone mid-reply at USERS=2 and 4 with pool + interleave, both switches on: both replies = their solo undrafted replies; the shared rounds ran shared_forward's forward, the lone ones user_drafts's loop" {
    for ([_]u8{ 2, 4 }) |n| {
        const o = try planned(n, true, true);
        try t.expect(o.shared and o.drafts);
        const fx = try fixtureModel();
        defer fx.parsed.deinit();
        const ra: api.Request = .{ .prompt = &pa, .max_tokens = 160, .drafts = true };
        const rb: api.Request = .{ .prompt = &pc, .max_tokens = 40, .drafts = true };
        var pool = try P.Pool.init(gpa, &fx.m, blocksOf(&.{ ra, rb }) * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var ba: Box = .{};
        defer ba.deinit();
        var bb: Box = .{};
        defer bb.deinit();
        var x: Served = .{};
        var opts = o;
        opts.pool = &pool;
        try x.init(n, opts, &.{});
        defer x.deinit();
        // A drafts alone (held in its first drafted round), B arrives, A yields and both share rounds, B ends and A
        // drafts again
        x.steps.draft_gate = &x.gate;
        x.gate.armed.store(true, .release);
        const e = x.host.engine();
        try e.submit(1, &ra, ba.sink());
        try x.start();
        try x.gate.waitEntered();
        try e.submit(2, &rb, bb.sink());
        x.gate.open.store(true, .release);
        _ = try bb.wait();
        _ = try ba.wait();
        x.stop();
        try expectRef(&pa, null, ba.tokens.items);
        try eq(@as(usize, 160), ba.tokens.items.len);
        try expectRef(&pc, null, bb.tokens.items);
        try eq(@as(usize, 40), bb.tokens.items.len);
        try t.expect(x.back.counts.yields >= 1); // A handed over at B's arrival
        try t.expect(x.steps.carries >= 1);
        try t.expect(x.steps.shared_rounds > 0); // the lane core's rounds: one forward over both rows
        try eq(@as(usize, 0), x.steps.row_rounds); // never the per-row forward
        try eq(@as(usize, 2), x.steps.max_rows);
        try t.expect(x.back.counts.resumes >= 1); // B's end: A back to its drafted loop
        try t.expect(x.steps.resumed_loops >= 1);
        try eq(@as(usize, 0), x.steps.stale);
        try eq(@as(usize, 0), x.owners());
        try eq(pool.blocks, pool.freeBlocks());
        try eq(@as(?anyerror, null), x.back.failed);
        std.debug.print("draft_shared lone->shared->lone C{d}: A 160 = solo, B 40 = solo; yields {d} resumes {d}; shared rounds {d}, per-row 0; drafted rounds {d}\n", .{ n, x.back.counts.yields, x.back.counts.resumes, x.steps.shared_rounds, x.steps.drafted_rounds });
    }
}

test "draft_shared shared -> lone -> shared mid-reply at USERS=2 and 4 with pool + interleave, both switches on: A starts beside B, drafts alone at B's end, shares again at C's arrival, drafts again at C's end; every reply = its solo undrafted reply" {
    for ([_]u8{ 2, 4 }) |n| {
        const o = try planned(n, true, true);
        const fx = try fixtureModel();
        defer fx.parsed.deinit();
        const ra: api.Request = .{ .prompt = &pa, .max_tokens = 200, .drafts = true };
        const rb: api.Request = .{ .prompt = &pd, .max_tokens = 12, .drafts = true };
        const rc: api.Request = .{ .prompt = &pc, .max_tokens = 30, .drafts = true };
        var pool = try P.Pool.init(gpa, &fx.m, blocksOf(&.{ ra, rb, rc }) * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var boxes: [3]Box = @splat(.{});
        defer for (&boxes) |*b| b.deinit();
        var x: Served = .{};
        var opts = o;
        opts.pool = &pool;
        try x.init(n, opts, &.{});
        defer x.deinit();
        x.steps.draft_gate = &x.gate;
        x.gate.armed.store(true, .release);
        const e = x.host.engine();
        try e.submit(1, &ra, boxes[0].sink());
        try e.submit(2, &rb, boxes[1].sink());
        try x.start(); // A and B together: shared rounds; B ends; A drafts alone (held in that drafted round)
        try x.gate.waitEntered();
        const shared_before = x.steps.shared_rounds;
        try e.submit(3, &rc, boxes[2].sink()); // C arrives: A yields, A and C share rounds; C ends: A drafts again
        x.gate.open.store(true, .release);
        _ = try boxes[2].wait();
        _ = try boxes[0].wait();
        _ = try boxes[1].wait();
        x.stop();
        try expectRef(&pa, null, boxes[0].tokens.items);
        try eq(@as(usize, 200), boxes[0].tokens.items.len);
        try expectRef(&pd, null, boxes[1].tokens.items);
        try expectRef(&pc, null, boxes[2].tokens.items);
        try t.expect(shared_before > 0); // A began in shared rounds beside B
        try t.expect(x.steps.shared_rounds > shared_before); // and shared again beside C
        try eq(@as(usize, 0), x.steps.row_rounds);
        try t.expect(x.back.counts.resumes >= 2); // lone at B's end and at C's end
        try t.expect(x.back.counts.yields >= 1);
        try t.expect(x.steps.resumed_loops >= 2);
        try eq(@as(usize, 0), x.steps.stale);
        try eq(@as(usize, 0), x.owners());
        try eq(pool.blocks, pool.freeBlocks());
        std.debug.print("draft_shared shared->lone->shared C{d}: A 200 = solo, B, C = solo; shared rounds {d} (before C {d}), resumes {d}, yields {d}\n", .{ n, x.steps.shared_rounds, shared_before, x.back.counts.resumes, x.back.counts.yields });
    }
}

/// The four requests of boot M's mix with both switches on (joined_profile's): a greedy reply that drafts alone, a seeded
/// sampled reply, a sampled schema reply, a greedy reply.
const four = [_]api.Request{
    .{ .prompt = &pa, .max_tokens = 120, .drafts = true },
    .{ .prompt = &pb, .max_tokens = 40, .drafts = true, .sampling = s7 },
    .{ .prompt = &pd, .max_tokens = 20, .drafts = true, .sampling = s23, .structure = json },
    .{ .prompt = &pc, .max_tokens = 30, .drafts = true },
};

test "draft_shared USERS=4 boot M's mix with both switches on (+ pool, interleave, user_sampling, structured_users): together = each alone; the schema reply valid under its masks; sampled rows user_sampling's keyed draw, never drafted; shared rounds shared_forward's forward" {
    const o = try planned(4, true, true);
    var solo: [4]Run = undefined;
    for (0..4) |i| solo[i] = try serve(4, o, &[_]usize{i}, &four);
    const all = [_]usize{ 0, 1, 2, 3 };
    const together = try serve(4, o, &all, &four);
    var serial_o = o;
    serial_o.interleave = false;
    const serial = try serve(4, serial_o, &all, &four);
    for (0..4) |i| {
        try t.expectEqualSlices(u32, solo[i].out[i].got(), together.out[i].got());
        try t.expectEqualSlices(u32, together.out[i].got(), serial.out[i].got());
        try eq(solo[i].out[i].reason.?, together.out[i].reason.?);
    }
    try expectRef(&pa, null, together.out[0].got());
    try eq(@as(usize, 120), together.out[0].n);
    try expectRef(&pb, s7, together.out[1].got());
    try eq(@as(usize, 40), together.out[1].n);
    try t.expectEqualSlices(u32, &schema_reply, together.out[2].got());
    try eq(api.Reason.stop, together.out[2].reason.?);
    try eq(@as(usize, 1), together.closes);
    try expectRef(&pc, null, together.out[3].got());
    try t.expect(together.max_rows >= 2);
    try t.expect(together.shared_rounds > 0);
    try eq(@as(usize, 0), together.row_rounds + serial.row_rounds);
    try t.expect(together.sampled_shared > 0);
    try t.expect(together.masked_draws > 0);
    try eq(@as(usize, 2), together.configured_sampled);
    try t.expect(together.resumes >= 1);
    try eq(@as(usize, 0), together.stale);
    try t.expect(together.pool_back and serial.pool_back);
    try eq(@as(usize, 0), together.owners);
    try t.expect(!together.failed and !serial.failed);
    // alone, a sampled request is never drafted: no lone run, every round one row (shared_forward's forward at one row)
    try eq(@as(u64, 0), solo[1].runs);
    try eq(@as(usize, 0), solo[1].loops);
    try eq(@as(usize, 1), solo[1].max_rows);
    try t.expect(solo[1].shared_rounds > 0);
    try eq(@as(u64, 0), solo[2].runs);
    std.debug.print("draft_shared boot M mix C4: together = solo 4/4, interleaved = serial 4/4, schema valid, sampled shared rows {d}, masked draws {d}, shared rounds {d}, per-row 0\n", .{ together.sampled_shared, together.masked_draws, together.shared_rounds });
}

test "draft_shared each switch alone and both unset give the same bytes at USERS=2 and 4 (pool + interleave + structured): shared_forward alone shares undrafted, user_drafts alone drafts beside the per-row forward, both compose; drafted = undrafted" {
    // with the drafts switch on the requests may draft (the lone driver's); off, a draft:true request is refused in
    // users mode (as before), so those settings run the same requests with drafts false
    const greedy = [_]api.Request{
        .{ .prompt = &pa, .max_tokens = 120, .drafts = true },
        .{ .prompt = &pb, .max_tokens = 40, .drafts = true },
        .{ .prompt = &pd, .max_tokens = 20, .drafts = true, .structure = json },
        .{ .prompt = &pc, .max_tokens = 30, .drafts = true },
    };
    var undrafted = greedy;
    for (&undrafted) |*r| r.drafts = false;
    var sets: usize = 0;
    for ([_]u8{ 2, 4 }) |n| {
        const which: []const usize = if (n == 2) &.{ 0, 2 } else &.{ 0, 1, 2, 3 };
        var none = try planned(n, false, false);
        none.sampled = false;
        var shared_only = try planned(n, true, false);
        shared_only.sampled = false;
        var drafts_only = try planned(n, false, true);
        drafts_only.sampled = false;
        var both = try planned(n, true, true);
        both.sampled = false;
        try t.expect(!none.shared and !none.drafts and shared_only.shared and !shared_only.drafts);
        try t.expect(!drafts_only.shared and drafts_only.drafts and both.shared and both.drafts);
        const r0 = try serve(n, none, which, &undrafted);
        const r1 = try serve(n, shared_only, which, &undrafted);
        const r2 = try serve(n, drafts_only, which, &greedy);
        const r3 = try serve(n, both, which, &greedy);
        for (which) |i| {
            try t.expectEqualSlices(u32, r0.out[i].got(), r1.out[i].got());
            try t.expectEqualSlices(u32, r0.out[i].got(), r2.out[i].got());
            try t.expectEqualSlices(u32, r0.out[i].got(), r3.out[i].got());
        }
        try expectRef(&pa, null, r3.out[0].got());
        try t.expectEqualSlices(u32, &schema_reply, r3.out[2].got());
        try eq(@as(usize, 0), r0.loops + r1.loops); // drafts off: never a drafted loop
        try t.expect(r2.loops >= 1 and r3.loops >= 1); // drafts on: the lone stream drafted
        try eq(@as(usize, 0), r0.shared_rounds + r2.shared_rounds); // shared off: the per-row forward
        try eq(@as(usize, 0), r1.row_rounds + r3.row_rounds); // shared on: shared_forward's forward, every round
        try t.expect(r1.shared_rounds > 0 and r3.shared_rounds > 0 and r0.row_rounds > 0 and r2.row_rounds > 0);
        sets += 1;
    }
    std.debug.print("draft_shared each switch alone = both = unset: {d}/2 user counts, 4 switch settings each, every reply byte-equal\n", .{sets});
}

test "draft_shared cancel and failure through the lane host with both switches on: a stream cancelled inside a shared round ends cancelled, its peer goes on and drafts alone = solo; a failure in a shared round or in a lone drafted round fails every owner, releases every slot and region, refuses later requests" {
    const o = try planned(4, true, true);
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 160, .drafts = true };
    const rb: api.Request = .{ .prompt = &pc, .max_tokens = 120, .drafts = true };
    { // a cancel inside a shared round: B ends cancelled, A goes on (alone again: drafted) = solo
        errdefer std.debug.print("draft_shared lane cancel/failure: the cancel section failed\n", .{});
        var pool = try P.Pool.init(gpa, &fx.m, blocksOf(&.{ ra, rb }) * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var ba: Box = .{};
        defer ba.deinit();
        var bb: Box = .{};
        defer bb.deinit();
        var x: Served = .{};
        var opts = o;
        opts.pool = &pool;
        try x.init(4, opts, &.{});
        defer x.deinit();
        var rg: Gate = .{};
        x.steps.round_gate = &rg;
        // armed before the start: the first round with both rows is held (polling B's tokens instead raced: in
        // ReleaseSafe B finished between two 1 ms polls)
        rg.armed.store(true, .release);
        const e = x.host.engine();
        try e.submit(1, &ra, ba.sink());
        try e.submit(2, &rb, bb.sink());
        try x.start();
        const at_arm = bb.count();
        rg.waitEntered() catch |err| {
            std.debug.print("draft_shared lane cancel: no round entered the gate (B had {d} tokens at the start, {d} now)\n", .{ at_arm, bb.count() });
            return err;
        };
        e.cancel(2);
        rg.open.store(true, .release);
        try eq(api.Reason.cancelled, try bb.wait());
        try eq(api.Reason.length, try ba.wait());
        x.stop();
        try expectRef(&pa, null, ba.tokens.items);
        try eq(@as(usize, 160), ba.tokens.items.len);
        try expectRef(&pc, null, bb.tokens.items); // what B sent before the cancel is its solo reply's start
        try t.expect(bb.tokens.items.len < 120);
        try t.expect(x.back.counts.resumes >= 1); // A alone again: back to its drafted loop
        try eq(@as(usize, 0), x.owners());
        try eq(pool.blocks, pool.freeBlocks());
        try eq(@as(usize, 0), x.steps.stale);
        try eq(@as(usize, 0), x.steps.row_rounds);
    }
    inline for (.{ "round", "drafted" }) |where| { // a failure in a shared round, then one in a lone drafted round
        errdefer std.debug.print("draft_shared lane cancel/failure: the {s} failure section failed\n", .{where});
        var pool = try P.Pool.init(gpa, &fx.m, blocksOf(&.{ ra, rb }) * P.block, 8, true, cap);
        defer pool.deinit(gpa);
        var ba: Box = .{};
        defer ba.deinit();
        var bb: Box = .{};
        defer bb.deinit();
        var bc: Box = .{};
        defer bc.deinit();
        var x: Served = .{};
        var opts = o;
        opts.pool = &pool;
        try x.init(4, opts, &.{});
        defer x.deinit();
        if (comptime std.mem.eql(u8, where, "round")) x.steps.fail_round = 3 else x.steps.fail_drafted = 2;
        const e = x.host.engine();
        try e.submit(1, &ra, ba.sink());
        if (comptime std.mem.eql(u8, where, "round")) try e.submit(2, &rb, bb.sink());
        try x.start();
        try eq(api.Reason.failed, try ba.wait());
        try t.expect(std.mem.indexOf(u8, &ba.message, "InjectedExchangeFailure") != null);
        if (comptime std.mem.eql(u8, where, "round")) try eq(api.Reason.failed, try bb.wait());
        try expectRef(&pa, null, ba.tokens.items);
        try eq(@as(?anyerror, error.InjectedExchangeFailure), x.back.failed);
        try eq(@as(usize, 0), x.owners());
        const rc: api.Request = .{ .prompt = &pd, .max_tokens = 10, .drafts = true };
        try e.submit(3, &rc, bc.sink());
        try eq(api.Reason.failed, try bc.wait()); // refused before any device call
        x.stop();
        try eq(@as(usize, 0), bc.tokens.items.len);
        try eq(pool.blocks, pool.freeBlocks());
    }
    std.debug.print("draft_shared lane cancel/failure: cancel inside a shared round clean (peer = solo), failure in a shared round and in a lone drafted round: every owner failed, slots and regions released, later request refused\n", .{});
}

test "draft_shared RED form of the stand-in: a shared round whose per-sequence part reads the next row's slot moves the bytes (the identity checks above can fail)" {
    const o = try planned(4, true, true);
    const fx = try fixtureModel();
    defer fx.parsed.deinit();
    const ra: api.Request = .{ .prompt = &pa, .max_tokens = 30, .drafts = false };
    const rb: api.Request = .{ .prompt = &pc, .max_tokens = 30, .drafts = false };
    var pool = try P.Pool.init(gpa, &fx.m, blocksOf(&.{ ra, rb }) * P.block, 8, true, cap);
    defer pool.deinit(gpa);
    var ba: Box = .{};
    defer ba.deinit();
    var bb: Box = .{};
    defer bb.deinit();
    var x: Served = .{};
    var opts = o;
    opts.pool = &pool;
    try x.init(4, opts, &.{});
    defer x.deinit();
    x.steps.mis = true;
    const e = x.host.engine();
    try e.submit(1, &ra, ba.sink());
    try e.submit(2, &rb, bb.sink());
    try x.start();
    _ = try ba.wait();
    _ = try bb.wait();
    x.stop();
    try t.expect(std.mem.indexOfScalar(u32, ba.tokens.items, Dev.poison) != null or std.mem.indexOfScalar(u32, bb.tokens.items, Dev.poison) != null);
    std.debug.print("draft_shared RED stand-in: a mis-split shared round moves the bytes (caught)\n", .{});
}
