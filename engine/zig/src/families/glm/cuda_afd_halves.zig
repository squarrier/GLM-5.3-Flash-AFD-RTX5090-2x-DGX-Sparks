//! The expert halves' side of the 3-node prefill pairs (qualification_c2d4ee2e; TF_AFD_PAIRS=1 on every node; evidence/qualification_0f991257/WIRE.md):
//! two consecutive prompt chunks A and B of one request, their MoE positions alternating on the wire (A_0, B_0, A_1,
//! B_1, ...), one request in flight a link. python 2.15's afd_mcdma next_forward (a pair's first request) and
//! next_partner (its second window's), and experts_node._pair's order (src@0849f3f glm5_next/cuda), on the Zig loop
//! (cuda_afd_expert.forwards): a CPU turn for each window's first request, every later request taken on the stream,
//! one sync a pair, then B's last header checked where it landed. Each window is computed with its own rows, as a solo
//! chunk is. The pairs are IMP5's design (ours); the split is Hugh Madden's (@dangerm00se, glm53f-afd) and tpurtell's
//! (@wrldsuksgo2mars); TensorFold and MCDMA are Ash Hart's (@ashxhart).
const std = @import("std");
const cuda = @import("cuda");
const ae = @import("cuda_afd_expert.zig");
const xchg = cuda.xchg;
const proto = xchg.proto;

/// TF_AFD_PAIRS (WIRE.md §1): 1 serves pairs (the settings carry pairs=1); unset or 0 leaves the loop as it was.
pub fn parse(v: ?[]const u8) error{BadAfdPairs}!bool {
    const s = v orelse return false;
    if (std.mem.eql(u8, s, "1")) return true;
    if (std.mem.eql(u8, s, "0")) return false;
    return error.BadAfdPairs;
}

/// TF_AFD_PAIRS read once by the role, before anything loads; a bad value refuses the run in words.
pub fn fromEnv() error{BadAfdPairs}!bool {
    const v: ?[]const u8 = if (std.c.getenv("TF_AFD_PAIRS")) |p| std.mem.span(p) else null;
    return parse(v) catch |e| {
        std.debug.print("INFO afd: TF_AFD_PAIRS takes 1 or 0 (or unset), not \"{s}\"\n", .{v.?});
        return e;
    };
}

/// TF_AFD_INFLIGHT (qualification_c533375f, WIRE.md §12): 2 lets the attention keep two requests in flight a link inside a pair, and
/// needs TF_AFD_PAIRS=1 (`pairs`); unset or 1 is one in flight. The halves' loop is the same either way (each request
/// taken in seq order on the one stream, its reply in its own send slot); the settings carry inflight=2, so a node
/// without the switch disagrees at the handshake.
pub fn parseInflight(v: ?[]const u8, pairs: bool) error{BadAfdInflight}!u8 {
    const s = v orelse return 1;
    if (std.mem.eql(u8, s, "1")) return 1;
    if (std.mem.eql(u8, s, "2") and pairs) return 2;
    return error.BadAfdInflight;
}

/// TF_AFD_INFLIGHT read once by the role, after TF_AFD_PAIRS and before anything loads; a bad value refuses the run.
pub fn inflightFromEnv(pairs: bool) error{BadAfdInflight}!u8 {
    const v: ?[]const u8 = if (std.c.getenv("TF_AFD_INFLIGHT")) |p| std.mem.span(p) else null;
    return parseInflight(v, pairs) catch |e| {
        if (std.mem.eql(u8, v.?, "2")) std.debug.print("INFO afd: TF_AFD_INFLIGHT=2 needs TF_AFD_PAIRS=1 on every node\n", .{}) else std.debug.print("INFO afd: TF_AFD_INFLIGHT takes 1 or 2 (or unset), not \"{s}\"\n", .{v.?});
        return e;
    };
}

/// Header words 7-9 of a pair's request (proto.Header.extra): its window's slot, the other window's rows, A's fseq.
pub const slot_word = 0;
pub const other_word = 1;
pub const first_word = 2;

/// A pair as its first request (A_0) opens it: window A's forward, B's rows, and the words every request carries.
pub const Pair = struct {
    a: ae.Forward,
    b_rows: usize,
    layers: i32, // word 5: the positions the halves serve
    count_word: i32, // word 6: 0, or the windows' count with TF_AFD_MTP=1

    /// The header window `slot` sends at position `pos` (WIRE.md §3), its fseq given.
    pub fn header(p: Pair, slot: u32, fseq: i32, pos: usize) proto.Header {
        const rows = if (slot == 0) p.a.rows else p.b_rows;
        const other = if (slot == 0) p.b_rows else p.a.rows;
        return .{ .fseq = fseq, .rows = @intCast(rows), .layer = @intCast(pos), .mode = .pair, .layers = p.layers, .extra = .{ @intCast(slot), @intCast(other), p.a.fseq, 0 } };
    }

    /// B's forward number: A's + 1, as the header's 31 bits carry it.
    pub fn bSeq(p: Pair) i32 {
        return @intCast((@as(u32, @bitCast(p.a.fseq)) +% 1) & 0x7FFFFFFF);
    }

    /// The requests a pair takes on each link: both windows' positions.
    pub fn requests(p: Pair) u32 {
        return @intCast(2 * p.a.count);
    }
};

/// python next_forward's checks on a pair's first request (A_0), with WIRE.md §6's words: position 0, the positions,
/// rows 1..max_rows, word 6 none or a count within them, slot 0, the other window's rows 1..max_rows, word 9 = word 1.
pub fn open(h: proto.Header, count_word: i32, layers: usize, max_rows: usize) ae.Breach!Pair {
    if (h.mode != .pair) return error.WrongMode;
    if (h.layers < 0 or @as(usize, @intCast(h.layers)) != layers) return error.WrongLayerCount;
    if (h.rows < 1 or @as(usize, @intCast(h.rows)) > max_rows) return error.WrongRows;
    if (h.layer != 0) return error.WrongLayer;
    const other: i64 = h.extra[other_word];
    if (count_word < 0 or count_word > h.layers or h.extra[slot_word] != 0 or h.extra[first_word] != h.fseq or other < 1 or other > max_rows or h.extra[3] != 0) return error.WrongPair;
    const n: usize = if (count_word == 0) layers else @intCast(count_word);
    return .{ .a = .{ .fseq = h.fseq, .rows = @intCast(h.rows), .mode = .pair, .first = 0, .count = n }, .b_rows = @intCast(other), .layers = h.layers, .count_word = count_word };
}

/// python next_partner: B_0's words 0-10 are what A_0 implies (B's fseq A's + 1, rows A's word 8, position 0, slot 1,
/// word 8 A's rows, word 9 A's fseq) and its word 6 A's; then window B's forward.
pub fn partner(p: Pair, h: proto.Header, count_word: i32) ae.Breach!ae.Forward {
    if (!std.meta.eql(h, p.header(1, p.bSeq(), 0)) or count_word != p.count_word) return error.WrongPartner;
    return .{ .fseq = h.fseq, .rows = p.b_rows, .mode = .pair, .first = 0, .count = p.a.count, .slot = 1 };
}

/// After the pair's sync: B's last request, still in its receive slot (until seq + 2), must be B's at position n - 1.
pub fn tail(p: Pair, h: proto.Header, count_word: i32) ae.Breach!void {
    if (!std.meta.eql(h, p.header(1, p.bSeq(), p.a.count - 1)) or count_word != p.count_word) return error.WrongPairEnd;
}

/// Request `seq`'s header where it landed (the host's view of the receive slot), and its word 6.
fn landedHeader(n: *const xchg.Node, seq: u32) ae.Breach!struct { proto.Header, i32 } {
    const h = proto.Header.read(n.received(n.src(0), seq, proto.HDR_BYTES)[0..proto.HDR_BYTES]) catch return error.WrongPairEnd;
    return .{ h, ae.countWord(n, seq) };
}

/// One pair, from its first request (A_0 at `seq.*`, header `h`), as python's _pair serves it: A_0 enqueued, B_0
/// awaited on the host and checked, then A_p and B_p in turn for every position, each window's rows its own; the
/// pair's one sync; B's last header checked. Null when it was served (`seq.*` then past it), else how the loop ends.
/// A --fault inside it answers the requests before the faulty one only, counted as the attention sends them.
pub fn pair(e: anytype, n: *xchg.Node, o: ae.Opts, st: *ae.Stats, seq: *u32, h: proto.Header) ?ae.End {
    const s0 = seq.*;
    const p = open(h, ae.countWord(n, s0), o.layers, o.max_rows) catch |err| return .{ .breach = err };
    const total = p.requests();
    const run: u32 = if (o.fault) |x| @min(total, x.at -% s0) else total;
    var b: ae.Forward = undefined;
    var rows: usize = 0;
    var k: u32 = 0;
    while (k < run) : (k += 1) {
        if (k == 1) { // B_0 lands once A_0's replies are taken: the host waits while A_0 runs on the stream
            const hb = switch (n.waitRequest(s0 +% 1)) {
                .request => |x| x,
                .stopped => return .stopped,
                .lost => |why| return .{ .lost = why },
            };
            st.turns += 1;
            b = partner(p, hb, ae.countWord(n, s0 +% 1)) catch |err| return .{ .breach = err };
        }
        const f = if (k % 2 == 0) p.a else b;
        e.take(n, f.rows) catch |err| return .{ .err = err };
        e.compute(k / 2, f) catch |err| return .{ .err = err };
        e.post(n, f.rows) catch |err| return .{ .err = err };
        rows += f.rows;
    }
    e.sync(n) catch |err| return if (err == error.AttentionLost) .{ .lost = ae.whyText(ae.lost(n) orelse .dead) } else .{ .err = err };
    if (n.failure()) |x| return .{ .failed = x }; // a take gave up or saw a breach: nothing was posted after it
    st.layers += run;
    st.rows += rows;
    if (run < total) return ae.fault(n, o.fault.?, s0 +% run);
    const last = landedHeader(n, s0 +% (total - 1)) catch |err| return .{ .breach = err };
    tail(p, last[0], last[1]) catch |err| return .{ .breach = err };
    st.forwards += 2;
    seq.* = s0 +% total;
    return null;
}

// -- qualification_9e49592e: prompt chunk lanes (TF_AFD_LANES=N, N = 3 or 4; evidence/qualification_0f991257/WIRE.md §13) ---------------------------------

/// TF_AFD_LANES (qualification_9e49592e): N = 3 or 4 serves lane groups of up to N windows (Mode.lanes) with N requests in flight a link
/// (the settings carry inflight=N lanes=N, the region N slots a source); it needs TF_AFD_PAIRS=1 (`pairs`), and
/// TF_AFD_INFLIGHT unset (`inflight_set`: the lanes set it); unset or 0 is off.
pub fn parseLanes(v: ?[]const u8, pairs: bool, inflight_set: bool) error{BadAfdLanes}!u8 {
    const s = v orelse return 0;
    if (std.mem.eql(u8, s, "0")) return 0;
    const n: u8 = if (std.mem.eql(u8, s, "3")) 3 else if (std.mem.eql(u8, s, "4")) 4 else return error.BadAfdLanes;
    if (!pairs or inflight_set) return error.BadAfdLanes;
    return n;
}

/// TF_AFD_LANES read once by the role, after TF_AFD_PAIRS and TF_AFD_INFLIGHT and before anything loads; a bad value
/// refuses the run in words.
pub fn lanesFromEnv(pairs: bool) error{BadAfdLanes}!u8 {
    const v: ?[]const u8 = if (std.c.getenv("TF_AFD_LANES")) |p| std.mem.span(p) else null;
    const inflight_set = std.c.getenv("TF_AFD_INFLIGHT") != null;
    return parseLanes(v, pairs, inflight_set) catch |e| {
        if (!std.mem.eql(u8, v.?, "3") and !std.mem.eql(u8, v.?, "4")) std.debug.print("INFO afd: TF_AFD_LANES takes 3, 4 or 0 (or unset), not \"{s}\"\n", .{v.?}) else if (!pairs) std.debug.print("INFO afd: TF_AFD_LANES={s} needs TF_AFD_PAIRS=1 on every node\n", .{v.?}) else std.debug.print("INFO afd: TF_AFD_LANES={s} keeps {s} requests in flight a link: leave TF_AFD_INFLIGHT unset\n", .{ v.?, v.? });
        return e;
    };
}

/// A lanes header's word 10 (proto.Header.extra[3]): the group's lanes.
pub const count_word_lanes = 3;

/// A lane group as its first request (lane 0's, at position 0) opens it: lane 0's forward, the group's lanes, each
/// lane's rows as the lane before announced them (word 8: the next lane's rows), and the words every request carries;
/// `short` (qualification_47b9bf3a) 1 once the last lane's first request said it has one position fewer.
pub const Lanes = struct {
    first: ae.Forward,
    n: u32,
    rows: [proto.LANES_MAX]usize = @splat(0),
    layers: i32,
    count_word: i32,
    short: u32 = 0,

    /// The header lane `k` sends at position `pos` (WIRE.md §13): its fseq first + k, its rows, word 8 the next lane's.
    pub fn header(g: Lanes, k: u32, pos: usize) proto.Header {
        const fseq: i32 = @intCast((@as(u32, @bitCast(g.first.fseq)) +% k) & 0x7FFFFFFF);
        return .{ .fseq = fseq, .rows = @intCast(g.rows[k]), .layer = @intCast(pos), .mode = .lanes, .layers = g.layers, .extra = .{ @intCast(k), @intCast(g.rows[(k + 1) % g.n]), g.first.fseq, @intCast(g.n) } };
    }

    /// The requests a group takes on each link: every lane's positions (the last lane's one fewer when short).
    pub fn requests(g: Lanes) u32 {
        return @intCast(g.n * g.first.count - g.short);
    }
};

/// python next_forward's checks on a lane group's first request (lane 0 at position 0), with WIRE.md §13's words:
/// position 0, the positions, rows 1..max_rows, word 6 none or a count within them, slot 0, the next lane's rows
/// 1..max_rows, word 9 = word 1, and 3..`most` lanes (the settings' N).
pub fn openLanes(h: proto.Header, count_word: i32, layers: usize, max_rows: usize, most: u8) ae.Breach!Lanes {
    if (h.mode != .lanes) return error.WrongMode;
    if (h.layers < 0 or @as(usize, @intCast(h.layers)) != layers) return error.WrongLayerCount;
    if (h.rows < 1 or @as(usize, @intCast(h.rows)) > max_rows) return error.WrongRows;
    if (h.layer != 0) return error.WrongLayer;
    const next: i64 = h.extra[other_word];
    const n = h.extra[count_word_lanes];
    if (count_word < 0 or count_word > h.layers or h.extra[slot_word] != 0 or h.extra[first_word] != h.fseq or next < 1 or next > max_rows or n < 3 or n > most) return error.WrongPair;
    const c: usize = if (count_word == 0) layers else @intCast(count_word);
    var g: Lanes = .{ .first = .{ .fseq = h.fseq, .rows = @intCast(h.rows), .mode = .lanes, .first = 0, .count = c }, .n = @intCast(n), .layers = h.layers, .count_word = count_word };
    g.rows[0] = @intCast(h.rows);
    g.rows[1] = @intCast(next);
    return g;
}

/// python next_partner for lane `k` (1..n-1): its first request's words are what the group implies (fseq first + k,
/// its rows the lane before's word 8, position 0, slot k, word 9 lane 0's fseq, word 10 n) with word 8 the next lane's
/// rows (the last lane's: lane 0's), and its word 6 lane 0's; then lane k's forward (its rows learned). qualification_47b9bf3a: when the
/// group carries the MTP position (lane 0's word 6 = word 5: each lane's absorb is its extra layer), the last lane may
/// be one position short (word 6 one less: the prompt's last chunk, which absorbs after the group).
pub fn lane(g: *Lanes, k: u32, h: proto.Header, count_word: i32, max_rows: usize) ae.Breach!ae.Forward {
    const next = h.extra[other_word];
    const last = k + 1 == g.n;
    if (next < 1 or next > max_rows or (last and next != g.rows[0])) return error.WrongPartner;
    if (!last) g.rows[k + 1] = @intCast(next);
    const short = last and g.count_word == g.layers and count_word == g.count_word - 1;
    if (!std.meta.eql(h, g.header(k, 0)) or (count_word != g.count_word and !short)) return error.WrongPartner;
    if (short) g.short = 1;
    return .{ .fseq = h.fseq, .rows = g.rows[k], .mode = .lanes, .first = 0, .count = g.first.count - g.short, .slot = k };
}

/// After the group's sync: the last request (lane n-1 at position count-1; qualification_47b9bf3a: with the last lane short, lane n-2
/// at the MTP position), still in its receive slot (until seq + slots), must be that lane's.
pub fn lanesTail(g: Lanes, h: proto.Header, count_word: i32) ae.Breach!void {
    if (!std.meta.eql(h, g.header(g.n - 1 - g.short, g.first.count - 1)) or count_word != g.count_word) return error.WrongPairEnd;
}

/// One lane group, from its first request (lane 0's at `seq.*`, header `h`), as python's _lanes serves it: lane k's
/// first request awaited on the host and checked when the group reaches it (k = 1..n-1), every request taken on the
/// stream in the wire's order (position by position, lanes in order), each with its lane's rows; the group's one sync;
/// the last header checked. Null when it was served (`seq.*` then past it), else how the loop ends. A --fault inside it
/// answers the requests before the faulty one only, counted as the attention sends them. qualification_47b9bf3a: a short last lane
/// (`lane`) takes one request off the group's total once its first request is in.
pub fn lanes(e: anytype, n: *xchg.Node, o: ae.Opts, st: *ae.Stats, seq: *u32, h: proto.Header) ?ae.End {
    const s0 = seq.*;
    var g = openLanes(h, ae.countWord(n, s0), o.layers, o.max_rows, o.lanes) catch |err| return .{ .breach = err };
    var total = g.requests();
    var run: u32 = if (o.fault) |x| @min(total, x.at -% s0) else total;
    var fw: [proto.LANES_MAX]ae.Forward = undefined;
    fw[0] = g.first;
    var rows: usize = 0;
    var k: u32 = 0;
    while (k < run) : (k += 1) {
        const j = k % g.n;
        if (k > 0 and k < g.n) { // lane k's first request: on the host, while the earlier lanes' run on the stream
            const hk = switch (n.waitRequest(s0 +% k)) {
                .request => |x| x,
                .stopped => return .stopped,
                .lost => |why| return .{ .lost = why },
            };
            st.turns += 1;
            fw[j] = lane(&g, j, hk, ae.countWord(n, s0 +% k), o.max_rows) catch |err| return .{ .breach = err };
            total = g.requests();
            run = @min(run, total);
        }
        const f = fw[j];
        e.take(n, f.rows) catch |err| return .{ .err = err };
        e.compute(k / g.n, f) catch |err| return .{ .err = err };
        e.post(n, f.rows) catch |err| return .{ .err = err };
        rows += f.rows;
    }
    e.sync(n) catch |err| return if (err == error.AttentionLost) .{ .lost = ae.whyText(ae.lost(n) orelse .dead) } else .{ .err = err };
    if (n.failure()) |x| return .{ .failed = x }; // a take gave up or saw a breach: nothing was posted after it
    st.layers += run;
    st.rows += rows;
    if (run < total) return ae.fault(n, o.fault.?, s0 +% run);
    const last = landedHeader(n, s0 +% (total - 1)) catch |err| return .{ .breach = err };
    lanesTail(g, last[0], last[1]) catch |err| return .{ .breach = err };
    st.forwards += g.n;
    seq.* = s0 +% total;
    return null;
}

test {
    _ = @import("cuda_afd_halves_test.zig");
}
