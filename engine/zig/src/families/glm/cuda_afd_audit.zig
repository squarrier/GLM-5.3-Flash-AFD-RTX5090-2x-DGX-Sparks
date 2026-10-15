//! python 2.15's TF_AFD_CHECK (afd.py exchange/collect's clones and `_finish_check`; experts_node's check record) in
//! Zig: each MoE layer's request sections (the rows, pick, wts) and its replies hashed with sha256 on both sides, the
//! attention's against each expert's at the window's end, and python's one line for the window. Records cross hosts as
//! JSON lines (one a forward and side); a CPU test compares them in memory.
const std = @import("std");
const cuda = @import("cuda");

const proto = cuda.xchg.proto;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Digest = [32]u8;

/// One side's hashes of one MoE layer: python's `in` (the request's three sections) and `out` (the attention: both
/// replies as they landed, planes 0 and 1; an expert: its reply in out[0]).
pub const Layer = struct { in: [3]Digest, out: [2]Digest };

/// Which side a record is: the attention node, or expert k (python's peers 1 and 2).
pub const Side = enum { attn, e0, e1 };

/// A digest's first 4 bytes in hex (python prints a hexdigest's [:8]).
fn hex8(d: Digest) [8]u8 {
    return std.fmt.bytesToHex(d[0..4].*, .lower);
}

fn hex(d: Digest) [64]u8 {
    return std.fmt.bytesToHex(d, .lower);
}

fn sha(b: []const u8) Digest {
    var d: Digest = undefined;
    Sha256.hash(b, &d, .{});
    return d;
}

/// The request's sections past its header, as python hashes b.normed[:R], b.pick[:R] and b.wts[:R].
pub fn sections(l: proto.Layout, R: usize, req: []const u8) [3][]const u8 {
    const o = l.offsets(R);
    const sel = 4 * l.s * R;
    return .{ req[o.normed..o.pick], req[o.pick..][0..sel], req[o.wts..o.end] };
}

/// The attention's record of a layer: the request it sent (header first) and the two replies as they landed.
pub fn attnLayer(l: proto.Layout, R: usize, req: []const u8, rep0: []const u8, rep1: []const u8) Layer {
    const s = sections(l, R, req);
    return .{ .in = .{ sha(s[0]), sha(s[1]), sha(s[2]) }, .out = .{ sha(rep0), sha(rep1) } };
}

/// An expert's record of a layer: the request as it landed (header first) and the reply it sent.
pub fn expertLayer(l: proto.Layout, R: usize, landing: []const u8, reply: []const u8) Layer {
    const s = sections(l, R, landing);
    return .{ .in = .{ sha(s[0]), sha(s[1]), sha(s[2]) }, .out = .{ sha(reply), @splat(0) } };
}

/// A window's comparison (python's checked forwards): counts, the row counts covered, the first difference's words.
pub const Window = struct {
    forwards: u64 = 0,
    layers: u64 = 0,
    mismatches: u64 = 0,
    seen: [16]usize = undefined,
    nseen: usize = 0,
    first_buf: [160]u8 = undefined,
    first_len: usize = 0,

    fn note(w: *Window, comptime fmt: []const u8, args: anytype) void {
        w.mismatches += 1;
        if (w.first_len > 0) return;
        w.first_len = if (std.fmt.bufPrint(&w.first_buf, fmt, args)) |s| s.len else |_| w.first_buf.len;
    }

    fn rows(w: *Window, R: usize) void {
        if (std.mem.indexOfScalar(usize, w.seen[0..w.nseen], R) != null or w.nseen == w.seen.len) return;
        w.seen[w.nseen] = R;
        w.nseen += 1;
        std.mem.sort(usize, w.seen[0..w.nseen], {}, std.sort.asc(usize));
    }

    /// One forward of R rows: the attention's layers against expert k's (null: its record is missing, python's empty
    /// blob: a mismatch a layer). Each section differing is one mismatch, each reply differing one (python's count).
    pub fn forward(w: *Window, R: usize, attn: []const Layer, experts: [2]?[]const Layer) void {
        w.forwards += 1;
        w.rows(R);
        for (attn, 0..) |a, pos| {
            w.layers += 1;
            for (experts, 0..) |ex, k| {
                const theirs = ex orelse {
                    w.note("layer {d} peer {d}: no record", .{ pos, k });
                    continue;
                };
                if (pos >= theirs.len) {
                    w.note("layer {d} peer {d}: no record", .{ pos, k });
                    continue;
                }
                const t = theirs[pos];
                for (0..3) |i| if (!std.mem.eql(u8, &a.in[i], &t.in[i]))
                    w.note("layer {d} peer {d} in{d}: {s} vs {s}", .{ pos, k, i, &hex8(a.in[i]), &hex8(t.in[i]) });
                if (!std.mem.eql(u8, &a.out[k], &t.out[0]))
                    w.note("layer {d} peer {d} out: {s} vs {s}", .{ pos, k, &hex8(a.out[k]), &hex8(t.out[0]) });
            }
        }
    }

    pub fn first(w: *const Window) []const u8 {
        return w.first_buf[0..w.first_len];
    }

    /// python's `afd-check: window complete, N forwards, M mismatches total, rows checked [...]`.
    pub fn line(w: *const Window, buf: []u8) []const u8 {
        var out: std.Io.Writer = .fixed(buf);
        out.print("afd-check: window complete, {d} forwards, {d} mismatches total, rows checked [", .{ w.forwards, w.mismatches }) catch {};
        for (w.seen[0..w.nseen], 0..) |r, i| out.print("{s}{d}", .{ if (i == 0) "" else ", ", r }) catch {};
        out.print("]", .{}) catch {};
        if (w.first_len > 0) out.print("; first mismatch: {s}", .{w.first()}) catch {};
        return out.buffered();
    }
};

/// A forward's record as one JSON line: `{"side":S,"seq":N,"rows":R,"layers":[{"in":[h,h,h],"out":[h,h]},...]}`.
pub fn writeRecord(gpa: std.mem.Allocator, out: *std.ArrayList(u8), side: Side, seq: u32, R: usize, layers: []const Layer) !void {
    try out.print(gpa, "{{\"side\":\"{t}\",\"seq\":{d},\"rows\":{d},\"layers\":[", .{ side, seq, R });
    for (layers, 0..) |x, i| {
        const h = [5][64]u8{ hex(x.in[0]), hex(x.in[1]), hex(x.in[2]), hex(x.out[0]), hex(x.out[1]) };
        try out.print(gpa, "{s}{{\"in\":[\"{s}\",\"{s}\",\"{s}\"],\"out\":[\"{s}\",\"{s}\"]}}", .{ if (i == 0) "" else ",", &h[0], &h[1], &h[2], &h[3], &h[4] });
    }
    try out.appendSlice(gpa, "]}\n");
}

/// One record read back (its layers owned by `gpa`).
pub const Record = struct { side: Side, seq: u32, rows: usize, layers: []Layer };

fn digest(text: []const u8) !Digest {
    var d: Digest = undefined;
    if (text.len != 64) return error.BadDigest;
    _ = std.fmt.hexToBytes(&d, text) catch return error.BadDigest;
    return d;
}

pub fn readRecord(gpa: std.mem.Allocator, line_text: []const u8) !Record {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, line_text, .{});
    defer p.deinit();
    const o = p.value.object;
    const side = std.meta.stringToEnum(Side, o.get("side").?.string) orelse return error.BadSide;
    const ls = o.get("layers").?.array.items;
    const layers = try gpa.alloc(Layer, ls.len);
    errdefer gpa.free(layers);
    for (ls, layers) |v, *x| {
        const in = v.object.get("in").?.array.items;
        const out = v.object.get("out").?.array.items;
        if (in.len != 3 or out.len != 2) return error.BadRecord;
        for (0..3) |i| x.in[i] = try digest(in[i].string);
        for (0..2) |i| x.out[i] = try digest(out[i].string);
    }
    return .{ .side = side, .seq = @intCast(o.get("seq").?.integer), .rows = @intCast(o.get("rows").?.integer), .layers = layers };
}

test "the audit: equal records compare clean; a section, a reply or a missing record each count as python counts" {
    const l: proto.Layout = .{ .d = 8, .s = 3 };
    const R = 2;
    const gpa = std.testing.allocator;
    const req = try gpa.alloc(u8, proto.wireBytes(l.req(R)));
    defer gpa.free(req);
    for (req, 0..) |*b, i| b.* = @truncate(i * 7 + 1);
    var rep: [2][2 * 8 * 4]u8 = undefined;
    for (&rep, 0..) |*r, k| for (r, 0..) |*b, i| {
        b.* = @truncate(i + 31 * k);
    };
    const a = [_]Layer{ attnLayer(l, R, req, &rep[0], &rep[1]), attnLayer(l, R, req, &rep[1], &rep[0]) };
    const e0 = [_]Layer{ expertLayer(l, R, req, &rep[0]), expertLayer(l, R, req, &rep[1]) };
    const e1 = [_]Layer{ expertLayer(l, R, req, &rep[1]), expertLayer(l, R, req, &rep[0]) };
    var w: Window = .{};
    w.forward(R, &a, .{ &e0, &e1 });
    try std.testing.expectEqual(@as(u64, 0), w.mismatches);
    var bad = e1;
    bad[1].in[2][0] ^= 1; // wts as expert 1 saw them
    bad[0].out[0][5] ^= 1; // its first reply
    w.forward(1, &a, .{ &e0, &bad });
    w.forward(R, &a, .{ null, &e1 });
    try std.testing.expectEqual(@as(u64, 4), w.mismatches);
    try std.testing.expectEqualStrings("layer 0 peer 1 out", w.first()[0..18]);
    var buf: [256]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, w.line(&buf), "afd-check: window complete, 3 forwards, 4 mismatches total, rows checked [1, 2]"));
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try writeRecord(gpa, &text, .e1, 77, R, &bad);
    const r = try readRecord(gpa, text.items[0 .. text.items.len - 1]);
    defer gpa.free(r.layers);
    try std.testing.expectEqual(Side.e1, r.side);
    try std.testing.expectEqual(@as(u32, 77), r.seq);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&bad), std.mem.sliceAsBytes(r.layers));
}
