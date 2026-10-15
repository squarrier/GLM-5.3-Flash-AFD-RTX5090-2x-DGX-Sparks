//! Version-2 CPU-only structured envelopes and request ledger; no transport or engine calls.
const std = @import("std");
pub const magic: u32 = 0x47524332;
pub const header_words = 30;
pub const Phase = enum(u32) { prepare = 1, ready = 2, commit = 3, done = 4, quiesce = 5, abort = 6, closed = 7 };
pub const Kind = enum(u32) { reset = 1, prompt_chunk = 2, group = 3, group_action = 4, first = 5, step = 6, end = 7 };
pub const Status = enum(u32) { ok = 0, cancelled = 1, empty_mask = 2, request_error = 3, poison = 4 };
pub const Action = enum(u32) { none = 0, stage = 1, advance = 2, route = 3, request = 4, take = 5, post = 6, tail = 7, absorb = 8, input_commit = 9, finish = 10, select = 11, drain = 12 };

pub const Descriptor = struct {
    phase: Phase = .prepare,
    kind: Kind,
    peer: u32 = 0,
    peer_count: u32,
    epoch: u64,
    op_seq: u64 = 0,
    chunk_index: u32 = 0,
    group_index: u32 = 0,
    slot: u32 = 0,
    position: u64 = 0,
    rows: u32 = 0,
    total_prompt_rows: u64 = 0,
    vocab_width: u32 = 0,
    tokens: []const u32 = &.{},
    mask: []const u32 = &.{},
    status: Status = .ok,
    action_index: u32 = 0,
    action: Action = .none,
    layer: u32 = 0,
    provider_position: u32 = 0,
    active: bool = false,
    transport_seq: u32 = 0,
    outstanding: u32 = 0,

    pub fn validate(d: Descriptor) !void {
        if (d.peer_count < 2 or d.peer_count > 3 or d.peer >= d.peer_count or d.epoch == 0) return error.ControlShape;
        if (d.tokens.len > std.math.maxInt(u32) or d.mask.len > std.math.maxInt(u32) or d.vocab_width > std.math.maxInt(i32)) return error.ControlShape;
        for (d.tokens) |t| if (t > std.math.maxInt(i32)) return error.ControlShape;
        const staged = d.kind == .prompt_chunk or (d.kind == .group_action and d.action == .stage);
        if (staged) {
            if (d.rows == 0 or d.tokens.len != d.rows or d.mask.len != 0) return error.ControlShape;
        } else if (d.kind == .step) {
            if (d.rows != 1 or d.tokens.len != 1) return error.ControlShape;
        } else if (d.kind == .first) {
            if (d.rows != 1 or d.tokens.len != 0) return error.ControlShape;
        } else if (d.tokens.len != 0 or d.mask.len != 0) return error.ControlShape;
        const selected = d.kind == .first or d.kind == .step;
        const W = (@as(u64, d.vocab_width) + 31) / 32;
        if (selected and d.active) {
            if (W == 0 or d.mask.len != W) return error.ControlShape;
            const tail: u5 = @intCast(d.vocab_width % 32);
            if (tail != 0 and d.mask[d.mask.len - 1] >> tail != 0) return error.ControlShape;
        } else if (d.mask.len != 0) return error.ControlShape;
        if (d.kind != .group_action and d.action != .none and d.action != .select) return error.ControlShape;
    }

    pub fn encode(d: Descriptor, gpa: std.mem.Allocator) ![]u32 {
        try d.validate();
        const payload = std.math.add(usize, d.tokens.len, d.mask.len) catch return error.ControlShape;
        const n = std.math.add(usize, header_words, payload) catch return error.ControlShape;
        const w = try gpa.alloc(u32, n);
        w[0..header_words].* = .{magic, 2, @intFromEnum(d.phase), @intFromEnum(d.kind), d.peer, d.peer_count,
            @truncate(d.epoch), @truncate(d.epoch >> 32), @truncate(d.op_seq), @truncate(d.op_seq >> 32),
            d.chunk_index, d.group_index, d.slot, @truncate(d.position), @truncate(d.position >> 32), d.rows,
            @truncate(d.total_prompt_rows), @truncate(d.total_prompt_rows >> 32), d.vocab_width,
            @intCast((@as(u64, d.vocab_width) + 31) / 32), @intCast(d.tokens.len), @intCast(d.mask.len),
            @intFromEnum(d.status), d.action_index, @intFromEnum(d.action), d.layer, d.provider_position,
            @intFromBool(d.active), d.transport_seq, d.outstanding};
        @memcpy(w[header_words..][0..d.tokens.len], d.tokens);
        @memcpy(w[header_words + d.tokens.len..], d.mask);
        return w;
    }

    pub fn decode(w: []const u32) !Descriptor {
        if (w.len < header_words or w[0] != magic or w[1] != 2 or w[27] > 1) return error.ControlShape;
        const nt: usize = w[20]; const nm: usize = w[21];
        const payload = std.math.add(usize, nt, nm) catch return error.ControlShape;
        const n = std.math.add(usize, header_words, payload) catch return error.ControlShape;
        if (n != w.len or w[19] != (@as(u64, w[18]) + 31) / 32) return error.ControlShape;
        const d = Descriptor{
            .phase = std.enums.fromInt(Phase, w[2]) orelse return error.ControlShape,
            .kind = std.enums.fromInt(Kind, w[3]) orelse return error.ControlShape,
            .peer = w[4], .peer_count = w[5], .epoch = wide(w[6], w[7]), .op_seq = wide(w[8], w[9]),
            .chunk_index = w[10], .group_index = w[11], .slot = w[12], .position = wide(w[13], w[14]),
            .rows = w[15], .total_prompt_rows = wide(w[16], w[17]), .vocab_width = w[18],
            .tokens = w[header_words..][0..nt], .mask = w[header_words + nt..],
            .status = std.enums.fromInt(Status, w[22]) orelse return error.ControlShape,
            .action_index = w[23], .action = std.enums.fromInt(Action, w[24]) orelse return error.ControlShape,
            .layer = w[25], .provider_position = w[26], .active = w[27] == 1,
            .transport_seq = w[28], .outstanding = w[29],
        };
        try d.validate();
        return d;
    }
};
fn wide(lo: u32, hi: u32) u64 { return @as(u64, lo) | (@as(u64, hi) << 32); }
pub fn toBytes(gpa: std.mem.Allocator, words: []const u32) ![]u8 {
    const n = std.math.mul(usize, words.len, 4) catch return error.ControlShape;
    const bytes = try gpa.alloc(u8, n);
    for (words, 0..) |w, i| std.mem.writeInt(u32, bytes[i*4..][0..4], w, .little);
    return bytes;
}
pub fn fromBytes(gpa: std.mem.Allocator, bytes: []const u8) ![]u32 {
    if (bytes.len < 4*header_words or bytes.len % 4 != 0) return error.ControlShape;
    var head: [header_words]u32 = undefined;
    for (&head, 0..) |*w, i| w.* = std.mem.readInt(u32, bytes[i*4..][0..4], .little);
    const payload = std.math.add(usize, head[20], head[21]) catch return error.ControlShape;
    const n = std.math.add(usize, header_words, payload) catch return error.ControlShape;
    if (n != bytes.len/4 or head[0] != magic or head[1] != 2) return error.ControlShape;
    const words = try gpa.alloc(u32, n);
    errdefer gpa.free(words);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, bytes[i*4..][0..4], .little);
    _ = try Descriptor.decode(words);
    return words;
}

pub const Candidate = struct { valid: bool = false, value_f32_bits: u32 = 0, global_id: u32 = 0 };
pub const Completion = struct { status: Status = .ok, issued: u32 = 0, taken: u32 = 0, logits_valid: bool = false, candidate: Candidate = .{} };
pub const Permit = struct { epoch: u64, op_seq: u64, peer: u32, kind: Kind, action_index: u32 };
pub const Hooks = struct {
    ctx: *anyopaque,
    open: *const fn (*anyopaque, u32) anyerror!void,
    before: *const fn (*anyopaque, Descriptor) anyerror!Permit,
    after: *const fn (*anyopaque, Permit, Completion) anyerror!Completion,
    quiesce: *const fn (*anyopaque, u32, Status) anyerror!Status,
    close: *const fn (*anyopaque, u32) void,
};
pub const Endpoint = struct {
    session: *Session,
    peer: u32,
    pub fn open(e: Endpoint) !void { const h = e.session.hooks orelse return error.IncompleteMaskedAdapter; try h.open(h.ctx, e.peer); }
    pub fn before(e: Endpoint, descriptor: Descriptor) !Permit {
        const h = e.session.hooks orelse return error.IncompleteMaskedAdapter;
        var d = descriptor; d.peer = e.peer; d.peer_count = e.session.peer_count;
        d.epoch = e.session.epoch;
        return h.before(h.ctx, d);
    }
    pub fn after(e: Endpoint, p: Permit, done: Completion) !Completion {
        const h = e.session.hooks orelse return error.IncompleteMaskedAdapter;
        return h.after(h.ctx, p, done);
    }
    pub fn quiesce(e: Endpoint, reason: Status) !Status {
        const h = e.session.hooks orelse return error.IncompleteMaskedAdapter;
        return h.quiesce(h.ctx, e.peer, reason);
    }
    pub fn close(e: Endpoint) void { if (e.session.hooks) |h| h.close(h.ctx, e.peer); }
};

/// Independently owned request identity, in permitted take order; no borrowed frame slices.
pub const Receipt = struct {
    epoch: u64, op_seq: u64, group: u32, slot: u32, chunk: u32, layer: u32,
    position: u64, provider_position: u32, transport_seq: u32, rows: u32,
    fn of(d: Descriptor) Receipt {
        return .{ .epoch = d.epoch, .op_seq = d.op_seq, .group = d.group_index, .slot = d.slot,
            .chunk = d.chunk_index, .layer = d.layer, .position = d.position,
            .provider_position = d.provider_position, .transport_seq = d.transport_seq, .rows = d.rows };
    }
    fn matches(r: Receipt, d: Descriptor) bool {
        var actual = of(d);
        actual.op_seq = r.op_seq;
        return std.meta.eql(r, actual);
    }
};
pub const ledger_capacity = 4;

/// The fixture callback serializes access under its mutex; this object never starts workers.
pub const Session = struct {
    gpa: std.mem.Allocator,
    epoch: u64,
    op_seq: u64,
    peer_count: u32,
    phase: Phase = .prepare,
    status: Status = .ok,
    hooks: ?Hooks = null,
    cancel_ctx: ?*anyopaque = null,
    cancel_check: ?*const fn (*anyopaque) bool = null,
    frames: [3]?[]u32 = @splat(null),
    completions: [3]?Completion = @splat(null),
    outstanding: [3]u32 = @splat(0),
    receipts: [3][ledger_capacity]Receipt = undefined,
    receipt_head: [3]u32 = @splat(0),
    closed: [3]bool = @splat(false),
    committed: bool = false,
    pending: Status = .ok,
    action_count: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, epoch: u64, peers: u32, op_seq: u64) !Session {
        if (epoch == 0 or peers < 2 or peers > 3 or op_seq == std.math.maxInt(u64)) return error.ControlShape;
        return .{ .gpa = gpa, .epoch = epoch, .op_seq = op_seq, .peer_count = peers };
    }
    pub fn deinit(s: *Session) void { s.clearFrames(); }
    pub fn endpoint(s: *Session, peer: u32) !Endpoint {
        if (peer >= s.peer_count) return error.ControlShape;
        return .{ .session = s, .peer = peer };
    }
    fn terminal(s: *const Session) !void {
        switch (s.status) {
            .ok => {}, .cancelled => return error.Cancelled, .empty_mask => return error.EmptyMask,
            .request_error => return error.StructuredRequestError, .poison => return error.ControlPoison,
        }
        if (s.phase == .closed) return error.SessionClosed;
    }
    pub fn poison(s: *Session) error{ControlPoison} {
        s.status = .poison; s.pending = .poison; s.phase = .abort;
        return error.ControlPoison;
    }
    pub fn ready(s: *Session, d: Descriptor) !?Permit {
        try s.terminal();
        if (s.committed) return error.OperationInProgress;
        try d.validate();
        if (d.peer_count != s.peer_count or d.epoch != s.epoch or d.op_seq != s.op_seq or
            (d.phase != .prepare and d.phase != .ready) or d.outstanding != s.outstanding[d.peer]) return s.poison();
        if (s.pending != .ok and d.action != .take and d.action != .drain and d.kind != .end) return error.IssueGateClosed;
        if (d.kind == .group_action) switch (d.action) {
            .request => if (s.outstanding[d.peer] >= ledger_capacity) return s.poison(),
            .take, .drain => {
                if (s.outstanding[d.peer] == 0) return s.poison();
                if (!s.receipts[d.peer][s.receipt_head[d.peer]].matches(d)) return s.poison();
            },
            else => {},
        };
        if (s.frames[d.peer] != null or s.closed[d.peer]) return s.poison();
        // A READY frame this peer cannot record is fail-stop for every peer (grammar_tests_c7): poison before returning, so peers
        // already waiting for the commit wake to ControlPoison instead of a transport deadline.
        const words = d.encode(s.gpa) catch return s.poison();
        errdefer s.gpa.free(words);
        for (s.frames[0..s.peer_count]) |other| if (other) |w| {
            if (w.len != words.len) return s.poison();
            for (w, words, 0..) |left, right, i| if (i != 4 and left != right) return s.poison();
        };
        s.frames[d.peer] = words;
        for (s.frames[0..s.peer_count]) |f| if (f == null) return null;
        if (s.action_count >= 4096) return s.poison();
        s.action_count += 1;
        s.committed = true; s.phase = .commit;
        return try s.permit(d.peer);
    }
    pub fn permit(s: *const Session, peer: u32) !Permit {
        try s.terminal();
        if (peer >= s.peer_count or !s.committed) return error.NotCommitted;
        const d = try Descriptor.decode(s.frames[peer].?);
        return .{ .epoch = s.epoch, .op_seq = s.op_seq, .peer = peer, .kind = d.kind, .action_index = d.action_index };
    }
    pub fn after(s: *Session, p: Permit, done: Completion) !?Completion {
        try s.terminal();
        if (!s.committed or p.peer >= s.peer_count or s.completions[p.peer] != null) return s.poison();
        const expected = try s.permit(p.peer);
        if (!std.meta.eql(expected, p) or done.status == .poison) return s.poison();
        const descriptor = try Descriptor.decode(s.frames[p.peer].?);
        const request = descriptor.kind == .group_action and descriptor.action == .request;
        const take = descriptor.kind == .group_action and (descriptor.action == .take or descriptor.action == .drain);
        if (done.issued != @intFromBool(request) or done.taken != @intFromBool(take)) return s.poison();
        const issued = std.math.add(u32, s.outstanding[p.peer], done.issued) catch return s.poison();
        if (done.taken > issued) return s.poison();
        if (request) {
            if (issued > ledger_capacity) return s.poison();
            const tail = (s.receipt_head[p.peer] + s.outstanding[p.peer]) % ledger_capacity;
            s.receipts[p.peer][tail] = Receipt.of(descriptor);
        }
        if (take) s.receipt_head[p.peer] = (s.receipt_head[p.peer] + 1) % ledger_capacity;
        s.outstanding[p.peer] = issued - done.taken;
        if (done.candidate.valid and (done.candidate.global_id > std.math.maxInt(i32) or
            !std.math.isFinite(@as(f32, @bitCast(done.candidate.value_f32_bits))))) return s.poison();
        s.completions[p.peer] = done;
        for (s.completions[0..s.peer_count]) |v| if (v == null) return null;
        var result: Completion = .{ .logits_valid = true };
        for (s.completions[0..s.peer_count]) |v| {
            const x = v.?;
            if (x.status == .cancelled) s.pending = .cancelled else if (x.status == .request_error and s.pending != .cancelled) s.pending = .request_error;
            result.logits_valid = result.logits_valid and x.logits_valid;
            const candidate = x.candidate;
            if (candidate.valid and (!result.candidate.valid or better(candidate, result.candidate))) result.candidate = candidate;
        }
        const d = try Descriptor.decode(s.frames[0].?);
        if ((d.kind == .first or d.kind == .step) and d.action == .select and !result.candidate.valid and s.pending == .ok) s.pending = .empty_mask;
        if (s.op_seq == std.math.maxInt(u64)) return s.poison();
        s.op_seq += 1; s.committed = false;
        s.clearFrames();
        s.phase = if (s.pending == .ok) .done else .quiesce;
        s.latchIfDrained();
        result.status = s.status;
        return result;
    }
    pub fn quiesce(s: *Session, reason: Status) Status {
        if (reason == .poison) { s.status = .poison; s.pending = .poison; s.phase = .abort; return s.status; }
        if (s.status != .ok) return s.status;
        if (reason == .cancelled or s.pending == .ok) s.pending = reason;
        if (!s.committed) s.phase = .quiesce;
        s.latchIfDrained();
        return s.status;
    }
    fn latchIfDrained(s: *Session) void {
        if (s.committed or s.pending == .ok) return;
        for (s.outstanding[0..s.peer_count]) |n| if (n != 0) return;
        s.status = s.pending; s.phase = .abort;
    }
    pub fn closePeer(s: *Session, peer: u32) !void {
        if (peer >= s.peer_count) return error.ControlShape;
        if (s.committed or s.outstanding[peer] != 0) return error.OutstandingActions;
        s.closed[peer] = true;
        for (s.closed[0..s.peer_count]) |closed| if (!closed) return;
        s.clearFrames(); s.phase = .closed;
    }
    fn clearFrames(s: *Session) void {
        for (&s.frames) |*f| { if (f.*) |w| s.gpa.free(w); f.* = null; }
        s.completions = @splat(null);
    }
};
fn better(x: Candidate, y: Candidate) bool {
    const left: f32 = @bitCast(x.value_f32_bits); const right: f32 = @bitCast(y.value_f32_bits);
    return left > right or (left == right and x.global_id < y.global_id);
}
