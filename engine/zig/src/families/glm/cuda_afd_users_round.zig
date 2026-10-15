//! concurrent_users: one R=C MoE request per layer, attention R1 per independent user; reusable CPU/device scheduling.
//! shared_forward (TF_AFD_USERS_SHARED=1): the round as one forward over every row, after @Bizuayeu's shared verify windows
//! (github.com/Bizuayeu/TensorFold pr/glm53-flash-cuda 75131b66, Apache-2.0; reimplemented, not copied).
//! draft_shared: TF_AFD_USERS_SHARED composes with user_drafts's TF_AFD_USERS_DRAFTS=1 (sharedFor); draft_shared's edits: Claude Opus 5.5
//! (Anthropic; planning and execution), via Hermes.
const std = @import("std");
const nd = @import("nucleus_draw.zig");

pub const Key = struct { slot: u8, generation: u64 };
/// user_sampling: qualification_45c10bcf's draw parameters (nucleus_draw's, which the Engine's `sampling` is)
pub const Sampling = nd.Sampling;
pub const Row = struct {
    key: Key,
    token: u32,
    position: usize,
    cancel_ctx: ?*anyopaque = null,
    cancel_check: ?*const fn (?*anyopaque) bool = null,
    mask: []const u32 = &.{}, // structured_users: a structured row's grammar mask for its draw (empty: the plain draw)
    /// user_sampling (TF_AFD_USERS_SAMPLING=1): the row's own request's sampling for its draw (qualification_45c10bcf's keyed draw at the drawn
    /// token's position, after the row's mask); null: today's greedy draw
    sampling: ?nd.Sampling = null,

    pub fn cancelled(r: Row) bool {
        return if (r.cancel_check) |check| check(r.cancel_ctx) else false;
    }
};

pub const Observation = struct {
    event: enum { prefill_before, prefill_after, retire, cancel_retired, failure_retired, cancel_before_sample, cancel_before_commit, commit_before, commit_after, round_settled, failure_before_commit, failure_during_commit, failure_after_commit },
    slot: u8,
    generation: u64,
    state: u8,
    position: usize,
    parity: u1,
    round: u64,
    commits: usize,
};

fn observe(comptime Device: type, d: *Device, event: @FieldType(Observation, "event"), row: Row, commits: usize) void {
    if (@hasDecl(Device, "observeRound")) d.observeRound(event, row, commits);
}

/// This is the actual device scheduling order, not a test-only imitation of it.
pub fn run(comptime Device: type, d: *Device, rows: []const Row, out: []u32) !void {
    if (rows.len == 0 or rows.len > 4 or rows.len != out.len) return error.Rows;
    var commits: usize = 0;
    var attempts: usize = 0;
    errdefer for (rows) |row| observe(Device, d, if (attempts > commits) .failure_during_commit else if (commits == 0) .failure_before_commit else .failure_after_commit, row, commits);
    try d.alive();
    try d.begin(rows);
    // shared_forward (TF_AFD_USERS_SHARED=1): one forward over every row; unset, each row's own forward as before (a device
    // without `shared`, concurrent_users's stand-ins among them, only ever has the per-row forward)
    if (comptime @hasDecl(Device, "shared")) {
        if (d.shared()) try forwardShared(Device, d, rows) else try forwardRows(Device, d, rows);
    } else try forwardRows(Device, d, rows);
    // No logits are read and no user State committed until ALL rows' launches synchronize/check successfully.
    try d.settle();
    for (rows, out, 0..) |row, *token, i| {
        token.* = 0;
        if (row.cancelled()) {
            observe(Device, d, .cancel_before_sample, row, commits);
            continue;
        }
        try d.alive();
        token.* = try d.sample(row, i);
    }
    try d.alive();
    for (rows, 0..) |row, i| {
        if (row.cancelled()) {
            observe(Device, d, .cancel_before_commit, row, commits);
            continue; // retirement is deferred to the owner/core after the round
        }
        try d.alive();
        attempts += 1;
        try d.commit(row, i);
        commits += 1;
    }
    try d.settle(); // queued position/conv writes finish before scratch reuse or retirement
    for (rows) |row| observe(Device, d, .round_settled, row, commits);
}

/// The per-row forward (concurrent_users's): each row's attention side alone at R = 1, one R = C MoE exchange a layer.
fn forwardRows(comptime Device: type, d: *Device, rows: []const Row) !void {
    for (rows, 0..) |row, i| {
        try d.alive();
        try d.open(row, i);
    }
    for (0..d.layerCount()) |li| {
        for (rows, 0..) |row, i| {
            try d.alive();
            try d.front(row, i, li);
        }
        if (d.isMoe(li)) {
            for (rows, 0..) |row, i| {
                try d.alive();
                try d.gather(row, i);
            }
            try d.alive();
            try d.exchange(li, rows.len);
            for (rows, 0..) |row, i| {
                try d.alive();
                try d.scatter(row, i, rows.len);
            }
        }
    }
    for (rows, 0..) |row, i| {
        try d.alive();
        try d.tail(row, i);
    }
}

/// shared_forward: one forward over every row (@Bizuayeu's shared windows, 75131b66, one row a user): each layer's row-by-row
/// half once over all rows, then each row's per-sequence half (its KDA chain or DSA attention) on its own State, then
/// the block's out-projection and the FFN (the MoE's one R = C exchange) over all rows; the head over all rows.
fn forwardShared(comptime Device: type, d: *Device, rows: []const Row) !void {
    try d.alive();
    try d.sharedOpen(rows);
    for (0..d.layerCount()) |li| {
        try d.alive();
        try d.sharedIn(rows, li);
        for (rows, 0..) |row, i| {
            try d.alive();
            try d.sharedSeq(row, i, li);
        }
        try d.alive();
        try d.sharedOut(rows, li);
    }
    try d.alive();
    try d.sharedTail(rows);
}

/// Exact spelling only: invalid values are refused before any model/device allocation.
pub fn parse(value: ?[]const u8) error{BadAfdUsers}!u8 {
    const v = value orelse return 1;
    for ([_][]const u8{ "1", "2", "4" }, [_]u8{ 1, 2, 4 }) |name, n| if (std.mem.eql(u8, v, name)) return n;
    return error.BadAfdUsers;
}

pub fn fromEnv() error{BadAfdUsers}!u8 {
    return parse(if (std.c.getenv("TF_AFD_USERS")) |v| std.mem.span(v) else null);
}

/// prefill_interleave: TF_AFD_USERS_INTERLEAVE's policy: `rounds` decode rounds of the live users, then `steps` prompt steps.
pub const Interleave = struct { rounds: u8, steps: u8 };

/// prefill_interleave: unset or 0 off (null); 1 is 1:1; R:C with R and C in 1..64 in plain digits; anything else refused.
pub fn parseInterleave(value: ?[]const u8) error{BadAfdUsersInterleave}!?Interleave {
    const v = value orelse return null;
    if (std.mem.eql(u8, v, "0")) return null;
    if (std.mem.eql(u8, v, "1")) return .{ .rounds = 1, .steps = 1 };
    const colon = std.mem.indexOfScalar(u8, v, ':') orelse return error.BadAfdUsersInterleave;
    return .{ .rounds = try policyCount(v[0..colon]), .steps = try policyCount(v[colon + 1 ..]) };
}

fn policyCount(text: []const u8) error{BadAfdUsersInterleave}!u8 {
    if (text.len == 0 or text.len > 2 or text[0] == '0') return error.BadAfdUsersInterleave;
    for (text) |c| if (c < '0' or c > '9') return error.BadAfdUsersInterleave;
    const n = std.fmt.parseInt(u8, text, 10) catch return error.BadAfdUsersInterleave;
    return if (n > 64) error.BadAfdUsersInterleave else n;
}

/// prefill_interleave: the switch, read once before any load; on needs TF_AFD_USERS 2 or 4.
pub fn interleaveFromEnv(users: u8) error{ BadAfdUsersInterleave, UsersInterleaveNeedsUsers }!?Interleave {
    const p = try parseInterleave(if (std.c.getenv("TF_AFD_USERS_INTERLEAVE")) |v| std.mem.span(v) else null);
    if (p != null and users < 2) return error.UsersInterleaveNeedsUsers;
    return p;
}

/// user_sampling: TF_AFD_USERS_SAMPLING in TF_GLM_SAMPLING's parse form: unset, empty or 0 off; 1 on; anything else refused.
pub fn parseSampling(value: ?[]const u8) error{BadAfdUsersSampling}!bool {
    const v = value orelse return false;
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadAfdUsersSampling;
}

/// user_sampling: the switch's value against the boot: on needs TF_AFD_USERS 2 or 4 and TF_GLM_SAMPLING=1 (`glm_sampling`).
pub fn samplingFor(value: ?[]const u8, users: u8, glm_sampling: bool) error{ BadAfdUsersSampling, UsersSamplingNeedsUsers, UsersSamplingNeedsGlmSampling }!bool {
    const on = try parseSampling(value);
    if (on and users < 2) return error.UsersSamplingNeedsUsers;
    if (on and !glm_sampling) return error.UsersSamplingNeedsGlmSampling;
    return on;
}

test "user_sampling users sampling is default off, explicit one only, and needs users and TF_GLM_SAMPLING" {
    const t = std.testing;
    try t.expect(!try parseSampling(null));
    try t.expect(!try parseSampling(""));
    try t.expect(!try parseSampling("0"));
    try t.expect(try parseSampling("1"));
    for ([_][]const u8{ "2", "on", "true", " 1", "1 ", "01", "-1", "yes" }) |v| try t.expectError(error.BadAfdUsersSampling, parseSampling(v));
    for ([_]u8{ 2, 4 }) |u| {
        try t.expect(try samplingFor("1", u, true));
        try t.expect(!try samplingFor(null, u, true));
        try t.expect(!try samplingFor("0", u, false));
        try t.expectError(error.UsersSamplingNeedsGlmSampling, samplingFor("1", u, false));
    }
    try t.expectError(error.UsersSamplingNeedsUsers, samplingFor("1", 1, true));
    try t.expectError(error.UsersSamplingNeedsUsers, samplingFor("1", 1, false));
    try t.expect(!try samplingFor(null, 1, false));
    try t.expectError(error.BadAfdUsersSampling, samplingFor("2", 2, true));
}

/// user_drafts: TF_AFD_USERS_DRAFTS: unset or 0 off; 1 drafts a lone stream (mode A); anything else refused.
pub fn parseDrafts(value: ?[]const u8) error{BadAfdUsersDrafts}!bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadAfdUsersDrafts;
}

/// user_drafts: the switch, read once before any load; on needs TF_AFD_USERS 2 or 4.
pub fn draftsFromEnv(users: u8) error{ BadAfdUsersDrafts, UsersDraftsNeedsUsers }!bool {
    const on = try parseDrafts(if (std.c.getenv("TF_AFD_USERS_DRAFTS")) |v| std.mem.span(v) else null);
    if (on and users < 2) return error.UsersDraftsNeedsUsers;
    return on;
}

test "user_drafts users drafts switch accepts only unset, 0 and 1" {
    const t = std.testing;
    try t.expectEqual(false, try parseDrafts(null));
    try t.expectEqual(false, try parseDrafts("0"));
    try t.expectEqual(true, try parseDrafts("1"));
    for ([_][]const u8{ "", "2", "00", "01", "true", "on", " 1", "1 ", "lone", "-1", "1:1" }) |v|
        try t.expectError(error.BadAfdUsersDrafts, parseDrafts(v));
}

/// shared_forward: TF_AFD_USERS_SHARED: unset or 0 off; 1 runs each users round as one forward over every row (stage 1,
/// undrafted rounds); anything else refused.
pub fn parseShared(value: ?[]const u8) error{BadAfdUsersShared}!bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadAfdUsersShared;
}

/// shared_forward: the switch's value against the boot: on needs TF_AFD_USERS 2 or 4. draft_shared: it composes with user_drafts's
/// TF_AFD_USERS_DRAFTS=1 (a lone stream drafts by user_drafts's loop, every round of the lane core is the shared undrafted
/// forward), so the drafts no longer refuse it.
pub fn sharedFor(value: ?[]const u8, users: u8) error{ BadAfdUsersShared, UsersSharedNeedsUsers }!bool {
    const on = try parseShared(value);
    if (on and users < 2) return error.UsersSharedNeedsUsers;
    return on;
}

test "shared_forward users shared switch: unset and 0 off, 1 on, needs users 2 or 4, anything else refused (draft_shared: user_drafts's drafts compose)" {
    const t = std.testing;
    try t.expectEqual(false, try parseShared(null));
    try t.expectEqual(false, try parseShared("0"));
    try t.expectEqual(true, try parseShared("1"));
    for ([_][]const u8{ "", "2", "00", "01", "true", "on", " 1", "1 ", "shared", "-1", "1:1", "yes" }) |v|
        try t.expectError(error.BadAfdUsersShared, parseShared(v));
    for ([_]u8{ 2, 4 }) |u| {
        try t.expect(try sharedFor("1", u));
        try t.expect(!try sharedFor(null, u));
        try t.expect(!try sharedFor("0", u));
    }
    try t.expectError(error.UsersSharedNeedsUsers, sharedFor("1", 1));
    try t.expect(!try sharedFor(null, 1));
    try t.expectError(error.BadAfdUsersShared, sharedFor("2", 4));
    std.debug.print("shared_forward switch parse: unset/0 off, 1 on at users 2 and 4, 12 spellings refused, users 1 refused (drafts compose: draft_shared)\n", .{});
}

test "prefill_interleave interleave policy accepts only unset, 0, 1 and canonical R:C in 1..64" {
    const t = std.testing;
    try t.expectEqual(@as(?Interleave, null), try parseInterleave(null));
    try t.expectEqual(@as(?Interleave, null), try parseInterleave("0"));
    try t.expectEqual(@as(?Interleave, .{ .rounds = 1, .steps = 1 }), try parseInterleave("1"));
    try t.expectEqual(@as(?Interleave, .{ .rounds = 1, .steps = 1 }), try parseInterleave("1:1"));
    try t.expectEqual(@as(?Interleave, .{ .rounds = 8, .steps = 1 }), try parseInterleave("8:1"));
    try t.expectEqual(@as(?Interleave, .{ .rounds = 64, .steps = 64 }), try parseInterleave("64:64"));
    for ([_][]const u8{ "", "00", "2", "true", "on", " 1", "1 ", "1:", ":1", "0:1", "1:0", "01:1", "1:01", "65:1", "1:65", "1:1:1", "-1:1", "1:+1", "1.0", "100:1" }) |v|
        try t.expectError(error.BadAfdUsersInterleave, parseInterleave(v));
}

test "concurrent_users users opt-in accepts only unset, 1, 2, 4" {
    try std.testing.expectEqual(@as(u8, 1), try parse(null));
    for ([_][]const u8{ "1", "2", "4" }, [_]u8{ 1, 2, 4 }) |v, n| try std.testing.expectEqual(n, try parse(v));
    for ([_][]const u8{ "", "0", "3", "8", "02", "true", " 2", "2 ", "-1" }) |v| try std.testing.expectError(error.BadAfdUsers, parse(v));
}
