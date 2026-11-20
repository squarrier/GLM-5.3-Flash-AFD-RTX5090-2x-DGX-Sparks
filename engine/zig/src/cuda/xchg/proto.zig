// Modified for V4 by Scott Quarrier with AI assistance: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! The three-node exchange's wire (qualification_12f4270a): python TF-AFD 2.15's request and reply layout and header, the pair-rank map over roce_proxy.c's region, and the settings both ends agree on.
//! The layout is mcdma.py's `Layout` at src@0849f3f (glm5_next/cuda), so each section's bytes can be checked against our python AFD lane; no code is copied.
const std = @import("std");

pub const ALIGN = 256; // every request section starts 256-B aligned (python ALIGN)
pub const HDR_BYTES = 256; // the request's header slot
pub const MAGIC: i32 = 0x0AFD4D31; // "AFD M1", python mcdma.WIRE_MAGIC
pub const WIRE = 2; // python afd.WIRE
pub const VERSION = 1; // this wire's own version in the settings agreement

/// The window kinds python's header names (afd.py MODE_*).
pub const Mode = enum(i32) { eager = 1, chunk = 3, slice = 5, pair = 7, lanes = 9, _ };

// roce_proxy.c's region geometry for two ranks (tf_roce_layout)
pub const SLOTS = 2;
/// qualification_9e49592e (TF_AFD_LANES): the most prompt chunk lanes, and so the most slots a source a region carries
pub const LANES_MAX = 4;
pub const PORTS_MAX = 2;
pub const FLAG_STRIDE = 128;
pub const CTRL_BYTES = 128;
pub const EXPERTS = 2; // expert ranks
pub const MAX_SLOT: u64 = 1 << 30;

// The host's failure reasons in the status's first word (the GPU writes 1 and 2: fn_xchg.cu)
pub const Why = enum(u32) { none = 0, gave_up = 1, breach = 2, silent = 16, dead = 17, proxy = 18, beats = 19, stopped = 20, _ };

// The status words (u32, pinned): fn_xchg.cu's FAILED, GAVE_UP, BREACH, FAIL_SEQ, FAIL_SRC
pub const S_FAILED = 0;
pub const S_GAVE_UP = 1;
pub const S_BREACH = 2;
pub const S_SEQ = 3;
pub const S_SRC = 4;
pub const STATUS_WORDS = 16;

pub fn alignUp(n: usize) usize {
    return (n + ALIGN - 1) / ALIGN * ALIGN;
}

/// What a message occupies on the wire: the proxy writes 16-byte units.
pub fn wireBytes(n: usize) usize {
    return (n + 15) / 16 * 16;
}

/// One exchange's payloads for `rows` rows (python mcdma.Layout): the header slot, normed bf16 [rows, d], pick int32 [rows, s], wts fp32 [rows, s]; fp8 rows are E4M3 codes then UE8M0 scale bytes.
pub const Layout = struct {
    d: usize,
    s: usize,
    fp8: bool = false,
    bf16: bool = false,

    pub const Offsets = struct { normed: usize, pick: usize, wts: usize, end: usize };

    pub fn offsets(l: Layout, rows: usize) Offsets {
        const normed = HDR_BYTES;
        const rows_bytes = if (l.fp8) l.d * rows + alignUp(l.d / 32 * rows) else 2 * l.d * rows;
        const pick = normed + rows_bytes;
        const wts = pick + alignUp(4 * l.s * rows);
        return .{ .normed = normed, .pick = pick, .wts = wts, .end = wts + 4 * l.s * rows };
    }

    /// The request's bytes (python Layout.req), before the wire's padding to 16.
    pub fn req(l: Layout, rows: usize) usize {
        return l.offsets(rows).end;
    }

    /// A reply's bytes: fp32 [rows, d]; with bf16 partials a prompt window's is bf16 (python Layout.rep).
    pub fn rep(l: Layout, rows: usize, prompt: bool) usize {
        return @as(usize, if (l.bf16 and prompt) 2 else 4) * l.d * rows;
    }

    /// The bytes per row each way, without the header and padding.
    pub fn perRow(l: Layout) struct { req: usize, rep: usize, rep_prompt: usize } {
        const rows_bytes = if (l.fp8) l.d + l.d / 32 else 2 * l.d;
        return .{ .req = rows_bytes + 8 * l.s, .rep = 4 * l.d, .rep_prompt = (if (l.bf16) @as(usize, 2) else 4) * l.d };
    }

    /// A slot that carries every message of up to `rows` rows (4 KiB pages, as roce_proxy.c wants).
    pub fn slotBytes(l: Layout, rows: usize) u64 {
        const most = @max(wireBytes(l.req(rows)), l.rep(rows, false));
        return std.mem.alignForward(u64, most, 4096);
    }
};

/// The request header's words 0-10 (python HDR_FMT <8i, PAIR_WORDS <11i); the slot's other words are zero.
pub const Header = struct {
    fseq: i32,
    rows: i32,
    layer: i32,
    mode: Mode = .eager,
    layers: i32,
    extra: [4]i32 = @splat(0), // words 7-10: a pair's or lanes' slot, other rows, first seq, count

    pub fn write(h: Header, out: *[HDR_BYTES]u8) void {
        @memset(out, 0);
        const w = [_]i32{ MAGIC, h.fseq, h.rows, h.layer, @intFromEnum(h.mode), h.layers, 0, h.extra[0], h.extra[1], h.extra[2], h.extra[3] };
        for (w, 0..) |v, i| std.mem.writeInt(i32, out[i * 4 ..][0..4], v, .little);
    }

    pub fn read(in: *const [HDR_BYTES]u8) error{BadMagic}!Header {
        const word = struct {
            fn at(b: *const [HDR_BYTES]u8, i: usize) i32 {
                return std.mem.readInt(i32, b[i * 4 ..][0..4], .little);
            }
        }.at;
        if (word(in, 0) != MAGIC) return error.BadMagic;
        return .{ .fseq = word(in, 1), .rows = word(in, 2), .layer = word(in, 3), .mode = @enumFromInt(word(in, 4)), .layers = word(in, 5), .extra = .{ word(in, 7), word(in, 8), word(in, 9), word(in, 10) } };
    }
};

/// roce_proxy.c's region for two ranks and `slot` bytes a slot (tf_roce_layout's arithmetic): recv[src][slot], flag[src][slot][port], send[slot], ctrl.
pub const Region = struct {
    slot: u64,
    /// qualification_9e49592e: the slots a source, one a request in flight a link (SLOTS; TF_AFD_LANES=N's N: Settings.region)
    slots: u32 = SLOTS,

    pub fn recv(r: Region, src: u32, slot: u32) u64 {
        return (@as(u64, src) * r.slots + slot) * r.slot;
    }

    pub fn flagBase(r: Region) u64 {
        return 2 * @as(u64, r.slots) * r.slot;
    }

    pub fn flag(r: Region, src: u32, slot: u32, port: u32) u64 {
        return r.flagBase() + ((@as(u64, src) * r.slots + slot) * PORTS_MAX + port) * FLAG_STRIDE;
    }

    pub fn send(r: Region, slot: u32) u64 {
        return r.flagBase() + 2 * @as(u64, r.slots) * PORTS_MAX * FLAG_STRIDE + @as(u64, slot) * r.slot;
    }

    pub fn ctrl(r: Region) u64 {
        return r.send(0) + @as(u64, r.slots) * r.slot;
    }

    /// The slot message `seq` takes in this region (each source's, and the sender's send slot).
    pub fn slotOf(r: Region, seq: u32) u32 {
        return seq % r.slots;
    }

    pub fn total(r: Region) u64 {
        return r.ctrl() + CTRL_BYTES;
    }
};

/// Expert k (0 or 1) is rank k of its pair and the attention rank 1 - k: expert k's replies land in the attention's recv[k], the attention's requests in expert k's recv[1 - k].
pub fn attnRank(k: u32) u32 {
    return 1 - k;
}

/// Where a node's incoming messages land: the attention takes expert k's from src k; expert k takes the attention's from src 1 - k.
pub fn takeSrc(expert: ?u32, k: u32) u32 {
    return if (expert) |e| 1 - e else k;
}

/// What both ends must run with; the control link's agreement compares their digests.
pub const Settings = struct {
    /// Current-native session/model/settings/timeout contract; absent preserves the released V3 text.
    native_contract: ?[32]u8 = null,
    layout: Layout,
    max_rows: usize,
    ports: usize,
    /// TF_AFD_PAIRS=1 (qualification_5b1b238a): prompt chunk pairs (Mode.pair) on this link; off, the text is today's
    pairs: bool = false,
    /// TF_AFD_INFLIGHT=2 (qualification_c533375f, pairs only): two requests in flight a link inside a pair, each in its own slot of the
    /// region's SLOTS = 2; 1, the text is the pairs' (or today's). qualification_9e49592e: TF_AFD_LANES=N sets it to N
    inflight: u8 = 1,
    /// TF_GLM_EXL3_SEQ=1 (qualification_b3fba1da's SEQ in the halves' prompt kernel; qualification_969df694): on every node, the attention's from its env,
    /// so a node without it disagrees at the handshake; off, the text is the one before (no wire byte changes)
    seq: bool = false,
    /// TF_GLM_HALVES_KERNEL (qualification_f504f9df): the halves' prompt-window kernel all three roles run with; empty, the text is today's
    halves: []const u8 = "",
    /// TF_AFD_STRIPES=K (qualification_17c70e37): a message of at least K stripes of `stripe_min` bytes goes out in K stripes a port,
    /// each with its own flag, and the take copies each stripe as it lands; 0 (off): one write a port, today's text
    stripes: u32 = 0,
    stripe_min: usize = STRIPE_MIN,
    /// TF_AFD_LANES=N (qualification_9e49592e, pairs only; 3 or 4): prompt chunk lanes (Mode.lanes) of up to N windows, N requests in
    /// flight a link, each in its own slot of a region of N slots a source; 0, the text is as before
    lanes: u8 = 0,
    /// rank_fingerprint: TF_GLM_FINGERPRINT=1 on every node: the text ends ` fp=1` (after every other suffix), so a node without it
    /// disagrees here, and the identity lines are swapped after this agreement (node.zig); off, the text is as before
    fp: bool = false,

    pub fn slotBytes(s: Settings) u64 {
        return s.layout.slotBytes(s.max_rows);
    }

    /// The region's slots a source: SLOTS, or more with more requests in flight (qualification_9e49592e's lanes).
    pub fn slotCount(s: Settings) u32 {
        return @max(SLOTS, s.inflight);
    }

    /// The region both ends run (its slots of slotBytes, slotCount a source).
    pub fn region(s: Settings) Region {
        return .{ .slot = s.slotBytes(), .slots = s.slotCount() };
    }

    pub fn text(s: Settings, buf: []u8) ![]const u8 {
        var fl: [32]u8 = undefined;
        const flight = if (s.inflight > 1) try std.fmt.bufPrint(&fl, " inflight={d}", .{s.inflight}) else "";
        var st: [48]u8 = undefined;
        const stripes = if (s.stripes == 0) "" else if (s.stripe_min == STRIPE_MIN) try std.fmt.bufPrint(&st, " stripes={d}", .{s.stripes}) else try std.fmt.bufPrint(&st, " stripes={d}/{d}", .{ s.stripes, s.stripe_min });
        var ln: [16]u8 = undefined;
        const lanes = if (s.lanes > 0) try std.fmt.bufPrint(&ln, " lanes={d}", .{s.lanes}) else "";
        const base=try std.fmt.bufPrint(buf, "tf-xchg v{d} wire {d} d={d} s={d} fp8={d} bf16={d} rows={d} slot={d} ports={d}{s}{s}{s}{s}{s}{s}{s}{s}", .{ VERSION, WIRE, s.layout.d, s.layout.s, @intFromBool(s.layout.fp8), @intFromBool(s.layout.bf16), s.max_rows, s.slotBytes(), s.ports, if (s.pairs) " pairs=1" else "", flight, if (s.seq) " seq=1" else "", if (s.halves.len > 0) " halves=" else "", s.halves, stripes, lanes, if (s.fp) " fp=1" else "" });
        if(s.native_contract)|id|{const suffix=try std.fmt.bufPrint(buf[base.len..]," native5={s} post=1048576",.{std.fmt.bytesToHex(id,.lower)});return buf[0..base.len+suffix.len];}
        return base;
    }
};

// qualification_17c70e37: stripes. Stripe j of a port's share lands with the flag word stripeWord(n, j) of that port's 128-byte flag,
// written after the stripe on the same queue pair; the last stripe's is word 0, so word 0 still means the whole message.
pub const STRIPES_MAX = 16; // every stripe has a word of its own in a flag's 32
pub const STRIPE_MIN: usize = 64 << 10; // a stripe of at least 64 KiB: decode messages stay one write

/// The stripes a message of `bytes` goes out in under setting `k` (0 or 1: one, today's).
pub fn stripesOf(k: u32, min: usize, bytes: usize) u32 {
    if (k < 2) return 1;
    return if (bytes >= @as(usize, k) * min) k else 1;
}

/// Port `p`'s share of `units` 16-byte units over `ports` ports (roce_proxy.c's split): its first unit and its count.
pub fn portShare(units: u64, ports: u32, p: u32) [2]u64 {
    const base = units / ports;
    const extra = units % ports;
    const first = base * p + @min(p, extra);
    return .{ first, base + @intFromBool(p < extra) };
}

/// Stripe j of n of a share of `units` units: its units [lo, hi) from the share's first.
pub fn stripeUnits(units: u64, n: u32, j: u32) [2]u64 {
    return .{ units * j / n, units * (j + 1) / n };
}

/// The flag word stripe j of n lands with: the last stripe word 0 (today's word), stripe j < n - 1 word 1 + j.
pub fn stripeWord(n: u32, j: u32) u32 {
    return if (j + 1 == n) 0 else 1 + j;
}

/// TF_AFD_STRIPES's value: K stripes (0: off) of at least `min` bytes each
pub const Stripes = struct { k: u32 = 0, min: usize = STRIPE_MIN };

/// TF_AFD_STRIPES as the roles read it: unset or 0 off; K (2-16) in stripes of at least 64 KiB; K/MIN in stripes of at
/// least MIN bytes (a multiple of 16: the lab's smaller stripes); anything else refused.
pub fn parseStripes(v: ?[]const u8) error{BadAfdStripes}!Stripes {
    const s = v orelse return .{};
    const slash = std.mem.indexOfScalar(u8, s, '/');
    const k = std.fmt.parseInt(u32, s[0 .. slash orelse s.len], 10) catch return error.BadAfdStripes;
    if (k == 0 and slash == null) return .{};
    if (k < 2 or k > STRIPES_MAX) return error.BadAfdStripes;
    const min = if (slash) |i| std.fmt.parseInt(usize, s[i + 1 ..], 10) catch return error.BadAfdStripes else STRIPE_MIN;
    if (min < 16 or min % 16 != 0 or min > MAX_SLOT) return error.BadAfdStripes;
    return .{ .k = k, .min = min };
}

/// TF_AFD_STRIPES read once by a role (all three nodes must agree: the settings' `stripes`); a bad value refuses the run
/// before anything loads, in words.
pub fn stripesFromEnv(role: []const u8) error{BadAfdStripes}!Stripes {
    const v: ?[]const u8 = if (std.c.getenv("TF_AFD_STRIPES")) |p| std.mem.span(p) else null;
    return parseStripes(v) catch |e| {
        std.debug.print("INFO {s}: TF_AFD_STRIPES takes K (2-{d}) or K/MIN (MIN bytes, a multiple of 16), or 0 (or unset), not \"{s}\"\n", .{ role, STRIPES_MAX, v.? });
        return e;
    };
}

/// The last message whose seq a flag holds, against `seq`: whether it has landed, across the 32-bit wrap.
pub fn landed(flag: u32, seq: u32) bool {
    return flag == seq;
}

test "python 2.15's request offsets: bf16 rows at 1, 7 and 2,048 rows, fp8 rows at 1 and 7 (D 4,096, S 9)" {
    const bf: Layout = .{ .d = 4096, .s = 9 };
    try std.testing.expectEqual(Layout.Offsets{ .normed = 256, .pick = 8448, .wts = 8704, .end = 8740 }, bf.offsets(1));
    try std.testing.expectEqual(Layout.Offsets{ .normed = 256, .pick = 57600, .wts = 57856, .end = 58108 }, bf.offsets(7));
    try std.testing.expectEqual(Layout.Offsets{ .normed = 256, .pick = 16777472, .wts = 16851200, .end = 16924928 }, bf.offsets(2048));
    const f8: Layout = .{ .d = 4096, .s = 9, .fp8 = true };
    try std.testing.expectEqual(Layout.Offsets{ .normed = 256, .pick = 4608, .wts = 4864, .end = 4900 }, f8.offsets(1));
    try std.testing.expectEqual(Layout.Offsets{ .normed = 256, .pick = 29952, .wts = 30208, .end = 30460 }, f8.offsets(7));
    try std.testing.expectEqual(@as(usize, 8752), wireBytes(bf.req(1)));
}

test "replies and bytes per row: fp32 partials, bf16 for prompt windows only, and the slot for 4,096 rows" {
    const l: Layout = .{ .d = 4096, .s = 9, .bf16 = true };
    try std.testing.expectEqual(@as(usize, 16384), l.rep(1, false));
    try std.testing.expectEqual(@as(usize, 8192), l.rep(1, true));
    try std.testing.expectEqual(@as(usize, 16384), (Layout{ .d = 4096, .s = 9 }).rep(1, true));
    const pr = (Layout{ .d = 4096, .s = 9 }).perRow();
    try std.testing.expectEqual(@as(usize, 8264), pr.req);
    try std.testing.expectEqual(@as(usize, 4296), (Layout{ .d = 4096, .s = 9, .fp8 = true }).perRow().req);
    try std.testing.expectEqual(@as(usize, 2 * 8264 + 2 * 16384), 2 * pr.req + 2 * pr.rep); // 49,296 B through the attention's port a row
    try std.testing.expectEqual(@as(u64, 64 << 20), (Layout{ .d = 4096, .s = 9 }).slotBytes(4096));
    try std.testing.expectEqual(@as(u64, 16384), (Layout{ .d = 4096, .s = 9 }).slotBytes(1)); // the 16 KiB reply, the larger message
    try std.testing.expectEqual(@as(u64, 12288), (Layout{ .d = 16, .s = 9 }).slotBytes(100)); // a request of 10,896 bytes, larger than the 6,400-byte reply here, in 4 KiB pages
}

test "the header's words as python packs them, and a wrong magic refused" {
    var b: [HDR_BYTES]u8 = undefined;
    const h: Header = .{ .fseq = 77, .rows = 16, .layer = 41, .mode = .chunk, .layers = 42, .extra = .{ 1, 2, 3, 4 } };
    h.write(&b);
    try std.testing.expectEqual(@as(u32, 0x0AFD4D31), std.mem.readInt(u32, b[0..4], .little));
    try std.testing.expectEqual(@as(i32, 3), std.mem.readInt(i32, b[16..20], .little));
    try std.testing.expectEqual(@as(i32, 0), std.mem.readInt(i32, b[24..28], .little));
    try std.testing.expectEqual(@as(i32, 4), std.mem.readInt(i32, b[40..44], .little));
    try std.testing.expect(std.mem.allEqual(u8, b[44..], 0));
    try std.testing.expectEqualDeep(h, try Header.read(&b));
    b[0] ^= 1;
    try std.testing.expectError(error.BadMagic, Header.read(&b));
}

test "the region is tf_roce_layout's for two ranks, and the experts' messages never share a slot" {
    const r: Region = .{ .slot = 1 << 20 };
    try std.testing.expectEqual(@as(u64, 4 << 20), r.flagBase());
    try std.testing.expectEqual(@as(u64, (4 << 20) + 1024), r.send(0));
    try std.testing.expectEqual(@as(u64, (6 << 20) + 1024), r.ctrl());
    try std.testing.expectEqual(@as(u64, (6 << 20) + 1024 + 128), r.total());
    try std.testing.expectEqual(@as(u64, (4 << 20) + (3 * 2 + 1) * 128), r.flag(1, 1, 1));
    // the attention's pair with expert k has it at rank 1 - k: expert 0 writes src 0, expert 1 src 1, and each expert reads the attention at the other index
    for (0..2) |k| {
        const e: u32 = @intCast(k);
        try std.testing.expectEqual(e, takeSrc(null, e));
        try std.testing.expectEqual(attnRank(e), takeSrc(e, 0));
        try std.testing.expect(attnRank(e) != e);
    }
    try std.testing.expect(attnRank(0) != attnRank(1));
}

test "the settings text names everything both ends must share" {
    var buf: [256]u8 = undefined;
    const s: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 4096, .ports = 1 };
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=4096 slot=67108864 ports=1", try s.text(&buf));
}

test "qualification_5b1b238a: TF_AFD_PAIRS's settings end in pairs=1, so a node without them disagrees; off, the text is today's" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var on = off;
    on.pairs = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1", try off.text(&a));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1", try on.text(&b));
    try std.testing.expect(!std.mem.eql(u8, try off.text(&a), try on.text(&b)));
    try std.testing.expectEqual(off.slotBytes(), on.slotBytes()); // one request in flight: the region is today's
}

test "qualification_c533375f: TF_AFD_INFLIGHT=2's settings end in pairs=1 inflight=2, so a pairs-only node disagrees; the region is the same (its two slots a source carry both requests)" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const one: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1, .pairs = true };
    var two = one;
    two.inflight = 2;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1", try one.text(&a));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight=2", try two.text(&b));
    try std.testing.expectEqual(one.slotBytes(), two.slotBytes());
    // the two requests in flight (seq s and s + 1) sit in the two receive slots of their source, never in one
    const r: Region = .{ .slot = two.slotBytes() };
    for (0..4) |s| try std.testing.expect(r.recv(1, @intCast(s % SLOTS)) != r.recv(1, @intCast((s + 1) % SLOTS)));
}

test "qualification_969df694: TF_GLM_EXL3_SEQ=1's settings end in seq=1 after the other suffixes, so a node without SEQ disagrees; unset, every text is the one before; the region is the same" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var seq = off;
    seq.seq = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1", try off.text(&a));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 seq=1", try seq.text(&b));
    var flight = off; // qualification_e01f8858's FLIGHT form, then with SEQ: seq=1 last
    flight.pairs = true;
    flight.inflight = 2;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight=2", try flight.text(&a));
    flight.seq = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight=2 seq=1", try flight.text(&b));
    var pairs = off;
    pairs.pairs = true;
    pairs.seq = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 seq=1", try pairs.text(&a));
    try std.testing.expect(!std.mem.eql(u8, try off.text(&a), try seq.text(&b)));
    try std.testing.expectEqual(off.slotBytes(), flight.slotBytes()); // SEQ changes no wire byte: the region is the same
}

test "qualification_f504f9df: TF_GLM_HALVES_KERNEL's settings end in halves=<value>, so a node with another value disagrees; unset, the text is today's" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    var c: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var k4 = off;
    k4.halves = "k4";
    var g53 = off;
    g53.halves = "g53kt3r";
    g53.pairs = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1", try off.text(&a));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 halves=k4", try k4.text(&b));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 halves=g53kt3r", try g53.text(&c));
    try std.testing.expectEqual(off.slotBytes(), k4.slotBytes()); // the region is today's
}

test "qualification_c863b626: every suffix together, qualification_qualification_44ba554e-int's first: pairs=1 inflight=2 seq=1, then halves=<value>; halves unset, the text is qualification_qualification_44ba554e-int's; the region is the same" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var lane = off; // the 3-node lane's FLIGHT + SEQ (qualification_969df694's f-s)
    lane.pairs = true;
    lane.inflight = 2;
    lane.seq = true;
    const fs = "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight=2 seq=1";
    try std.testing.expectEqualStrings(fs, try lane.text(&a));
    var g = lane;
    g.halves = "g53kt3r";
    try std.testing.expectEqualStrings(fs ++ " halves=g53kt3r", try g.text(&a));
    var k = lane;
    k.halves = "k4";
    try std.testing.expectEqualStrings(fs ++ " halves=k4", try k.text(&b));
    try std.testing.expect(!std.mem.eql(u8, try g.text(&a), try k.text(&b))); // the two values disagree at the handshake
    try std.testing.expect(!std.mem.eql(u8, try lane.text(&a), try k.text(&b))); // and each with an unset node
    var h = off; // a halves value alone: the branches' text
    h.halves = "k4";
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 halves=k4", try h.text(&a));
    try std.testing.expectEqual(lane.slotBytes(), g.slotBytes());
    try std.testing.expectEqual(off.slotBytes(), k.slotBytes());
}

test "qualification_17c70e37: TF_AFD_STRIPES's settings end in stripes=K (and its minimum when not 64 KiB); off, the text is today's" {
    var a: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1", try off.text(&a));
    var on = off;
    on.stripes = 8;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 stripes=8", try on.text(&a));
    on.pairs = true;
    on.stripe_min = 4096;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 stripes=8/4096", try on.text(&a));
    try std.testing.expectEqual(off.slotBytes(), on.slotBytes()); // the region is today's
    try std.testing.expectEqual(Stripes{}, try parseStripes(null));
    try std.testing.expectEqual(Stripes{}, try parseStripes("0"));
    try std.testing.expectEqual(Stripes{ .k = 8, .min = 65536 }, try parseStripes("8"));
    try std.testing.expectEqual(Stripes{ .k = 16, .min = 65536 }, try parseStripes("16"));
    try std.testing.expectEqual(Stripes{ .k = 8, .min = 4096 }, try parseStripes("8/4096"));
    try std.testing.expectEqual(Stripes{ .k = 2, .min = 16 }, try parseStripes("2/16"));
    for ([_][]const u8{ "1", "17", "-1", "8x", "", "true", "8/100", "8/0", "0/4096", "8/", "/4096", "8/4096/2", "1/4096" }) |v| try std.testing.expectError(error.BadAfdStripes, parseStripes(v));
}

test "qualification_17c70e37: stripe geometry: prompt messages in K stripes, decode messages in one; every unit in exactly one stripe of one port; the last stripe's flag is word 0" {
    // a 1,024-row chunk's request (8,462,592 B) and reply (16,777,216 B) in 8; a decode request (8,752 B) and reply (16 KiB) in 1
    const req = wireBytes((Layout{ .d = 4096, .s = 9 }).req(1024));
    try std.testing.expectEqual(@as(usize, 8462592), req);
    try std.testing.expectEqual(@as(u32, 8), stripesOf(8, STRIPE_MIN, req));
    try std.testing.expectEqual(@as(u32, 8), stripesOf(8, STRIPE_MIN, 16777216));
    try std.testing.expectEqual(@as(u32, 1), stripesOf(8, STRIPE_MIN, 8752));
    try std.testing.expectEqual(@as(u32, 1), stripesOf(8, STRIPE_MIN, 16384));
    try std.testing.expectEqual(@as(u32, 1), stripesOf(0, STRIPE_MIN, req));
    try std.testing.expectEqual(@as(u32, 1), stripesOf(1, STRIPE_MIN, req));
    try std.testing.expectEqual(@as(u32, 1), stripesOf(8, STRIPE_MIN, 8 * STRIPE_MIN - 16)); // just under: one
    try std.testing.expectEqual(@as(u32, 8), stripesOf(8, STRIPE_MIN, 8 * STRIPE_MIN));
    for ([_]u64{ 1, 2, 3, 7, 16, 17, 4095, 528912, 1048576 }) |units| for ([_]u32{ 1, 2 }) |ports| for ([_]u32{ 1, 2, 3, 8, 16 }) |n| {
        var seen: u64 = 0;
        var next: u64 = 0;
        for (0..ports) |p| {
            const sh = portShare(units, ports, @intCast(p));
            try std.testing.expectEqual(next, sh[0]); // the shares are contiguous, in port order
            var at: u64 = 0;
            for (0..n) |j| {
                const s = stripeUnits(sh[1], n, @intCast(j));
                try std.testing.expectEqual(at, s[0]);
                try std.testing.expect(s[1] >= s[0]);
                at = s[1];
                seen += s[1] - s[0];
            }
            try std.testing.expectEqual(sh[1], at);
            next = sh[0] + sh[1];
        }
        try std.testing.expectEqual(units, seen);
        try std.testing.expectEqual(units, next);
    };
    try std.testing.expectEqual([4]u32{ 1, 2, 7, 0 }, [4]u32{ stripeWord(8, 0), stripeWord(8, 1), stripeWord(8, 6), stripeWord(8, 7) });
    try std.testing.expectEqual(@as(u32, 0), stripeWord(1, 0)); // one stripe: today's word
    try std.testing.expect(STRIPES_MAX <= FLAG_STRIDE / 4); // a word of its own for every stripe
}

test "qualification_9e49592e: TF_AFD_LANES=N's settings end in pairs=1 inflight=N lanes=N, so a node without them disagrees; the region takes N slots a source (N requests in flight, never two in one slot); unset, the region is today's" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    try std.testing.expectEqual(Region{ .slot = off.slotBytes() }, off.region()); // SLOTS, today's geometry
    var two = off;
    two.pairs = true;
    two.inflight = 2;
    try std.testing.expectEqual(Region{ .slot = two.slotBytes() }, two.region()); // qualification_c533375f's two in flight: the same two slots
    for ([_]u8{ 3, 4 }) |n| {
        var on = two;
        on.inflight = n;
        on.lanes = n;
        const want = try std.fmt.bufPrint(&a, "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight={d} lanes={d}", .{ n, n });
        try std.testing.expectEqualStrings(want, try on.text(&b));
        const r = on.region();
        try std.testing.expectEqual(@as(u32, n), r.slots);
        try std.testing.expectEqual(on.slotBytes(), r.slot);
        // tf_roce_layout's arithmetic at n slots: recv 2n slots, the flags, send n slots, ctrl
        try std.testing.expectEqual(@as(u64, 2 * n) * r.slot, r.flagBase());
        try std.testing.expectEqual(r.flagBase() + 2 * @as(u64, n) * PORTS_MAX * FLAG_STRIDE, r.send(0));
        try std.testing.expectEqual(r.send(0) + @as(u64, n) * r.slot + CTRL_BYTES, r.total());
        // n consecutive requests (seqs s .. s + n - 1) sit in n distinct receive slots and send slots; seq s + n reuses s's
        for (0..8) |s0| {
            for (0..n) |i| for (i + 1..n) |j| {
                try std.testing.expect(r.recv(1, r.slotOf(@intCast(s0 + i))) != r.recv(1, r.slotOf(@intCast(s0 + j))));
                try std.testing.expect(r.send(r.slotOf(@intCast(s0 + i))) != r.send(r.slotOf(@intCast(s0 + j))));
            };
            try std.testing.expectEqual(r.slotOf(@intCast(s0)), r.slotOf(@intCast(s0 + n)));
        }
        // the flags of every (source, slot, port) apart, inside the flag block
        try std.testing.expect(r.flag(1, n - 1, 1) + FLAG_STRIDE <= r.send(0));
        try std.testing.expect(r.recv(1, n - 1) + r.slot <= r.flagBase());
    }
}

test "rank_fingerprint: TF_GLM_FINGERPRINT=1's settings end in fp=1 after every other suffix, so a node without it disagrees; unset, every text is the one before; the region is the same" {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    const off: Settings = .{ .layout = .{ .d = 4096, .s = 9 }, .max_rows = 2048, .ports = 1 };
    var on = off;
    on.fp = true;
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1", try off.text(&a));
    try std.testing.expectEqualStrings("tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 fp=1", try on.text(&b));
    try std.testing.expect(!std.mem.eql(u8, try off.text(&a), try on.text(&b)));
    var lab = off; // every suffix (qualification_df46b359's lane: pairs, lanes=4, SEQ, g53kt3r; and stripes), then fp=1 last
    lab.pairs = true;
    lab.inflight = 4;
    lab.lanes = 4;
    lab.seq = true;
    lab.halves = "g53kt3r";
    lab.stripes = 8;
    const want = "tf-xchg v1 wire 2 d=4096 s=9 fp8=0 bf16=0 rows=2048 slot=33554432 ports=1 pairs=1 inflight=4 seq=1 halves=g53kt3r stripes=8 lanes=4";
    try std.testing.expectEqualStrings(want, try lab.text(&a));
    var lab_on = lab;
    lab_on.fp = true;
    try std.testing.expectEqualStrings(want ++ " fp=1", try lab_on.text(&b));
    try std.testing.expectEqual(lab.region(), lab_on.region());
    try std.testing.expectEqual(off.region(), on.region());
}
