//! rank_fingerprint: the AFD hello's wire, recorded. One real node (an expert, or the attention) against a scripted peer over a
//! loopback control link; every frame the real node sends is written to a transcript (who, tag, length, payload; the fake
//! proxy's blob carries an address, so only its slot word is kept), and the transcript's sha256 printed as
//! `rank_fingerprint-REC <side> frames=<n> sha256=<hex>`. Written against the base's API only (b5804353), so the same file runs on
//! the base and on rank_fingerprint: with TF_GLM_FINGERPRINT unset the two transcripts must be the same bytes. The scripted peer
//! also answers the identity frame (tag 107) when the real node's settings end ` fp=1` (identity.zig's on-runs).
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const fake = @import("fake.zig");
const link = cuda.link;
const net = std.Io.net;

const gpa = std.testing.allocator;
pub const layout: proto.Layout = .{ .d = 64, .s = 9, .bf16 = true };
pub const max_rows = 32;
const ms = std.time.ns_per_ms;

// the control link's frame tags as node.zig numbers them (the wire's values)
pub const hello = 101;
pub const ack = 102;
pub const blob = 103;
pub const ready = 104;
pub const stop = 106;
pub const ident = 107;

/// A node's config with TF_GLM_FINGERPRINT unset (base API only).
pub fn configOff(watcher: bool) xchg.Config {
    return .{ .settings = .{ .layout = layout, .max_rows = max_rows, .ports = 1 }, .beat_ns = 50 * ms, .setup_timeout_ms = 10_000, .warn_failures = true, .run_watcher = watcher };
}

/// The frames a real node sent, one line each.
pub const Transcript = struct {
    buf: [8192]u8 = undefined,
    len: usize = 0,
    frames: usize = 0,
    tags: [32]u32 = undefined,

    pub fn text(t: *const Transcript) []const u8 {
        return t.buf[0..t.len];
    }

    pub fn sha(t: *const Transcript) [64]u8 {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(t.text(), &d, .{});
        return std.fmt.bytesToHex(d, .lower);
    }

    pub fn seq(t: *const Transcript) []const u32 {
        return t.tags[0..t.frames];
    }

    /// One frame `f` the real node sent on `who`'s link: its tag, length and payload (a blob's slot word only).
    fn add(t: *Transcript, who: []const u8, f: link.Frame) !void {
        var w: std.Io.Writer = .fixed(t.buf[t.len..]);
        const tag: u32 = @backingInt(f.tag);
        try w.print("{s} tag={d} len={d}", .{ who, tag, f.bytes.len });
        if (tag == blob and f.bytes.len == 16) {
            try w.print(" slot={d}", .{std.mem.readInt(u64, f.bytes[8..16], .little)});
        } else if (tag == hello and f.bytes.len >= 4) {
            try w.print(" k={d} text=\"{s}\"", .{ std.mem.readInt(u32, f.bytes[0..4], .little), f.bytes[4..] });
        } else if (f.bytes.len > 0) try w.print(" text=\"{s}\"", .{f.bytes});
        try w.writeByte('\n');
        t.len += w.buffered().len;
        if (t.frames == t.tags.len) return error.TooManyFrames;
        t.tags[t.frames] = tag;
        t.frames += 1;
    }

    pub fn print(t: *const Transcript, side: []const u8) void {
        std.debug.print("rank_fingerprint-REC {s} frames={d} sha256={s}\n{s}", .{ side, t.frames, &t.sha(), t.text() });
    }
};

/// The scripted peer's own blob: a fake proxy's (an address it may write to, the slot it must agree).
fn peerBlob(out: *[16]u8, region: []u8, slot: u64) []const u8 {
    std.mem.writeInt(u64, out[0..8], @intFromPtr(region.ptr), .little);
    std.mem.writeInt(u64, out[8..16], slot, .little);
    return out;
}

/// The next frame, recorded.
fn take(l: link.Link, t: *Transcript, who: []const u8) !link.Frame {
    const f = try l.recv(gpa);
    errdefer f.deinit(gpa);
    try t.add(who, f);
    return f;
}

fn drop(l: link.Link, t: *Transcript, who: []const u8) !void {
    const f = try take(l, t, who);
    f.deinit(gpa);
}

/// Called on the real node after its role ended and before it is freed (identity.zig reads the node's verdict).
pub const Hook = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, n: *xchg.Node) void,
};

/// What the scripted attention relays for an expert's identity line (identity.zig's on-runs): the attention's and the
/// other expert's lines, and optionally an altered line in place of the expert's own.
pub const Others = struct {
    attention: []const u8 = "",
    expert_other: []const u8 = "",
    self_override: ?[]const u8 = null,
    hook: ?Hook = null,
};

/// Node memory for one real node (a region of the settings' size, the status words), freed by `deinit`.
const Mem = struct {
    region: []align(4096) u8,
    status: []u32,

    fn init(cfg: xchg.Config) !Mem {
        const region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(cfg.settings.region().total()));
        errdefer gpa.free(region);
        @memset(region, 0);
        const status = try gpa.alloc(u32, proto.STATUS_WORDS);
        @memset(status, 0);
        return .{ .region = region, .status = status };
    }

    fn node(m: Mem) xchg.Memory {
        return .{ .region = m.region, .region_dev = @intFromPtr(m.region.ptr), .status = m.status, .status_dev = @intFromPtr(m.status.ptr) };
    }

    fn deinit(m: Mem) void {
        gpa.free(m.region);
        gpa.free(m.status);
    }
};

/// Expert `k` (a real node over `cfg`) joins a scripted attention, which answers each step as node.zig's attend does,
/// then sends the stop frame; the expert's frames recorded. A failed join closes the expert's link (the script's next
/// read fails), and the expert's error is returned first.
pub fn expertSide(cfg: xchg.Config, k: u32, others: Others, t: *Transcript) !void {
    const m = try Mem.init(cfg);
    defer m.deinit();
    const peer = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(cfg.settings.region().total()));
    defer gpa.free(peer);
    var opener: fake.Opener = .{ .gpa = gpa };
    var n = try xchg.Node.init(gpa, std.testing.io, .{ .expert = k }, cfg, m.node());
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Expert = struct {
        fn run(node: *xchg.Node, master: net.IpAddress, o: *fake.Opener, end: *?anyerror) void {
            node.join(master, o.opener()) catch |e| {
                end.* = e;
                node.deinit(); // as the process would exit: its link closes
                return;
            };
            if (node.idle() != .stopped) end.* = error.NotStopped;
        }
    };
    var end: ?anyerror = null;
    const th = try std.Thread.spawn(.{}, Expert.run, .{ &n, try net.IpAddress.parse("127.0.0.1", server.port()), &opener, &end });
    const s = if (server.accept(10_000)) |l| blk: {
        const r = scriptAttention(l, cfg, k, others, t, peer);
        l.close();
        break :blk r;
    } else |e| e;
    th.join();
    if (others.hook) |h| h.f(h.ctx, &n);
    n.deinit();
    if (end) |e| return e;
    return s;
}

fn scriptAttention(l: link.Link, cfg: xchg.Config, k: u32, others: Others, t: *Transcript, peer: []u8) !void {
    try drop(l, t, "expert->attn"); // its hello
    var tb: [256]u8 = undefined;
    const mine = try cfg.settings.text(&tb);
    try l.send(@fromBackingInt(@intCast(ack)), mine); // the attention's text: the same settings
    if (std.mem.endsWith(u8, mine, " fp=1")) { // identity.zig's on-runs: its line, then the table back
        const id = try take(l, t, "expert->attn");
        defer id.deinit(gpa);
        var b: [2048]u8 = undefined;
        const own = others.self_override orelse id.bytes;
        const table = if (k == 0)
            try std.fmt.bufPrint(&b, "attention {s}\nexpert-0 {s}\nexpert-1 {s}\n", .{ others.attention, own, others.expert_other })
        else
            try std.fmt.bufPrint(&b, "attention {s}\nexpert-0 {s}\nexpert-1 {s}\n", .{ others.attention, others.expert_other, own });
        try l.send(@fromBackingInt(@intCast(ident)), table);
    }
    var pb: [16]u8 = undefined;
    try l.send(@fromBackingInt(@intCast(blob)), peerBlob(&pb, peer, cfg.settings.slotBytes())); // the attention's blob first
    try drop(l, t, "expert->attn"); // its blob
    try l.send(@fromBackingInt(@intCast(ready)), &.{});
    try drop(l, t, "expert->attn"); // its ready
    try l.send(@fromBackingInt(@intCast(stop)), &.{});
}

/// The attention (a real node over `cfg`) attends two scripted experts, which answer each step as node.zig's join
/// does (with `lines` as their identity lines when the settings end ` fp=1`); its frames on both links recorded, then
/// its stop frames. A failed attend closes the attention's links, and its error is returned first.
pub fn attentionSide(cfg: xchg.Config, lines: [2][]const u8, t: *Transcript) !void {
    const m = try Mem.init(cfg);
    defer m.deinit();
    var peers: [2][]align(4096) u8 = undefined;
    for (&peers) |*p| p.* = try gpa.alignedAlloc(u8, .fromByteUnits(4096), @intCast(cfg.settings.region().total()));
    defer for (peers) |p| gpa.free(p);
    var opener: fake.Opener = .{ .gpa = gpa };
    var n = try xchg.Node.init(gpa, std.testing.io, .attn, cfg, m.node());
    const server = try link.Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Attn = struct {
        fn run(node: *xchg.Node, sv: link.Server, o: *fake.Opener, end: *?anyerror) void {
            node.attend(sv, o.opener()) catch |e| {
                end.* = e;
                node.deinit(); // as the process would exit: its links close
                return;
            };
            node.stopExperts();
        }
    };
    var end: ?anyerror = null;
    const th = try std.Thread.spawn(.{}, Attn.run, .{ &n, server, &opener, &end });
    var ls: [2]?link.Link = .{ null, null };
    const s = scriptExperts(&ls, try net.IpAddress.parse("127.0.0.1", server.port()), cfg, lines, t, &peers, th);
    for (ls) |l| if (l) |x| x.close();
    n.deinit();
    if (end) |e| return e;
    return s;
}

/// The two scripted experts. `th` (the attention's thread) is joined here once both readies are sent, or on an error.
fn scriptExperts(ls: *[2]?link.Link, master: net.IpAddress, cfg: xchg.Config, lines: [2][]const u8, t: *Transcript, peers: *[2][]align(4096) u8, th: std.Thread) !void {
    var joined = false;
    defer if (!joined) {
        for (ls) |*l| if (l.*) |x| {
            x.close();
            l.* = null;
        };
        th.join();
    };
    const who = [2][]const u8{ "attn->expert0", "attn->expert1" };
    var tb: [256]u8 = undefined;
    const text = try cfg.settings.text(&tb);
    const on = std.mem.endsWith(u8, text, " fp=1");
    for (0..2) |k| { // each expert's hello, in order (the attention accepts them as they come)
        ls[k] = try link.Link.join(std.testing.io, master, 10_000);
        var hb: [4 + 256]u8 = undefined;
        std.mem.writeInt(u32, hb[0..4], @intCast(k), .little);
        @memcpy(hb[4..][0..text.len], text);
        try ls[k].?.send(@fromBackingInt(@intCast(hello)), hb[0 .. 4 + text.len]);
        try drop(ls[k].?, t, who[k]); // its ack
        if (on) try ls[k].?.send(@fromBackingInt(@intCast(ident)), lines[k]);
    }
    if (on) for (0..2) |k| try drop(ls[k].?, t, who[k]); // the table
    for (0..2) |k| { // each pair: the attention's blob first, then ours
        try drop(ls[k].?, t, who[k]);
        var pb: [16]u8 = undefined;
        try ls[k].?.send(@fromBackingInt(@intCast(blob)), peerBlob(&pb, peers[k], cfg.settings.slotBytes()));
    }
    for (0..2) |k| try drop(ls[k].?, t, who[k]); // its ready
    for (ls) |l| try l.?.send(@fromBackingInt(@intCast(ready)), &.{});
    th.join();
    joined = true;
    for (0..2) |k| try drop(ls[k].?, t, who[k]); // the stop frames
}

test "rank_fingerprint-REC: TF_GLM_FINGERPRINT unset: expert 0's frames at the hello are hello, its blob, ready (the base's sequence and bytes)" {
    var t: Transcript = .{};
    try expertSide(configOff(true), 0, .{}, &t);
    t.print("expert-side-off");
    try std.testing.expectEqualSlices(u32, &.{ hello, blob, ready }, t.seq());
    try std.testing.expect(std.mem.indexOf(u8, t.text(), "text=\"tf-xchg v1 wire 2 d=64 s=9 fp8=0 bf16=1 rows=32 slot=") != null);
    try std.testing.expect(std.mem.indexOf(u8, t.text(), " fp=1") == null);
}

test "rank_fingerprint-REC: TF_GLM_FINGERPRINT unset: the attention's frames on each link are ack, its blob, ready, then stop (the base's sequence and bytes)" {
    var t: Transcript = .{};
    try attentionSide(configOff(false), .{ "", "" }, &t);
    t.print("attention-side-off");
    try std.testing.expectEqualSlices(u32, &.{ ack, ack, blob, blob, ready, ready, stop, stop }, t.seq());
    try std.testing.expect(std.mem.indexOf(u8, t.text(), " fp=1") == null);
}
