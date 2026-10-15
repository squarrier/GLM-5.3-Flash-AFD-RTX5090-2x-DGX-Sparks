//! rank_fingerprint (TF_GLM_FINGERPRINT=1, default off): the build and weight identity a node brings to the handshake, and the one
//! agreement every node applies to all of them: `config` and `ckpt` equal on every node; `exe` and `kernels` equal among
//! nodes of the same role and arch; `src` equal when every node has one, else one WARN naming the nodes without it.
//! The AFD hello (xchg/node.zig) and the two-Spark pair's bring-up (families/glm/cuda_ranks.zig) exchange the lines.
//! Reimplemented from @Bizuayeu's field list in 322b4de6 (github.com/Bizuayeu/TensorFold, pr/glm53-flash-cuda,
//! cuda_ranks.zig:1-31, Apache-2.0: config.json, aot.json and executable sha256s and the snapshot name, agreed before the
//! load); no code is copied.
const std = @import("std");
const builtin = @import("builtin");
const link = @import("link.zig");
const options = @import("kernel_options");

const Sha256 = std.crypto.hash.sha2.Sha256;

pub const name = "TF_GLM_FINGERPRINT";
/// The control-link frame that carries identity lines (the AFD link's own tags are 101-106, the pair's 1-6).
pub const tag: link.Tag = @fromBackingInt(@intCast(107));
pub const unset = "unset";
/// -Dsource_id (zig/build/source_id.zig) where the runtime's builder writes it into kernel_options, else `unset`.
pub const source_id: []const u8 = if (@hasDecl(options, "source_id")) options.source_id else unset;

/// TF_GLM_FINGERPRINT's value: 1 on; 0 or unset off; anything else refused.
pub fn parse(v: ?[]const u8) error{BadGlmFingerprint}!bool {
    const s = v orelse return false;
    if (std.mem.eql(u8, s, "1")) return true;
    if (std.mem.eql(u8, s, "0")) return false;
    return error.BadGlmFingerprint;
}

/// TF_GLM_FINGERPRINT read once by a role before anything loads; a bad value refuses the run with one line naming it.
pub fn fromEnv(role: []const u8) error{BadGlmFingerprint}!bool {
    const v: ?[]const u8 = if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    return parse(v) catch |e| {
        std.debug.print("INFO {s}: " ++ name ++ " takes 1 or 0 (or unset), not \"{s}\"\n", .{ role, v.? });
        return e;
    };
}

/// The fields the agreement compares (`role` and `arch` group the nodes for exe and kernels).
pub const Field = enum { config, ckpt, exe, kernels, src };
const keys = [_][]const u8{ "role", "arch", "config", "ckpt", "exe", "kernels", "src" };

/// A byte a value keeps as it is; any other is written %XX (both sides alike, so equal values stay equal).
fn plain(c: u8) bool {
    return c > ' ' and c < 0x7f and c != '=' and c != '%';
}

/// One node's identity: each value as its line carries it (escaped).
pub const Identity = struct {
    role: []const u8,
    arch: []const u8,
    config: []const u8,
    ckpt: []const u8,
    exe: []const u8,
    kernels: []const u8,
    src: []const u8,

    pub fn get(id: Identity, f: Field) []const u8 {
        return switch (f) {
            inline else => |t| @field(id, @tagName(t)),
        };
    }

    /// `role=… arch=… config=… ckpt=… exe=… kernels=… src=…`, one line.
    pub fn line(id: Identity, buf: []u8) ![]const u8 {
        var w: std.Io.Writer = .fixed(buf);
        inline for (keys, 0..) |k, i| {
            if (i > 0) try w.writeByte(' ');
            try w.writeAll(k ++ "=");
            for (@field(id, k)) |c| if (plain(c)) try w.writeByte(c) else try w.print("%{X:0>2}", .{c});
        }
        return w.buffered();
    }

    /// A line back into its fields (slices of `text`): every key once, in order, nothing else.
    pub fn parse(text: []const u8) error{BadIdentity}!Identity {
        var id: Identity = undefined;
        var it = std.mem.splitScalar(u8, text, ' ');
        inline for (keys) |k| {
            const word = it.next() orelse return error.BadIdentity;
            if (!std.mem.startsWith(u8, word, k ++ "=")) return error.BadIdentity;
            @field(id, k) = word[k.len + 1 ..];
        }
        if (it.next() != null) return error.BadIdentity;
        return id;
    }
};

/// sha256 of a file, as lower-case hex (64 KiB reads; error.FileNotFound when it cannot be opened).
pub fn sha256File(path: []const u8) ![64]u8 {
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return error.NameTooLong;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const fd = std.c.open(z[0..path.len :0], .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    var h = Sha256.init(.{});
    var buf: [64 << 10]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        h.update(buf[0..@intCast(n)]);
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

/// The kernels digest: one `name sha256` line a fatbin the node loads (and `aot.json <sha256>` for its Triton set), sorted,
/// then sha256 over the lines, each ending in a newline.
pub const Kernels = struct {
    pub const max = 64;
    const width = 64 + 1 + 64;
    lines: [max][width]u8 = undefined,
    lens: [max]u8 = undefined,
    n: usize = 0,

    pub fn add(k: *Kernels, image: []const u8, bytes: []const u8) !void {
        var sha: [32]u8 = undefined;
        Sha256.hash(bytes, &sha, .{});
        try k.addHash(image, std.fmt.bytesToHex(sha, .lower));
    }

    pub fn addHash(k: *Kernels, image: []const u8, hex: [64]u8) !void {
        if (k.n == max) return error.TooManyKernels;
        if (image.len == 0 or image.len > 64) return error.BadKernelName;
        for (image) |c| if (!plain(c)) return error.BadKernelName;
        const l = try std.fmt.bufPrint(&k.lines[k.n], "{s} {s}", .{ image, &hex });
        k.lens[k.n] = @intCast(l.len);
        k.n += 1;
    }

    /// `dir`/aot.json's line when the Triton set has one; nothing when it has none.
    pub fn addTriton(k: *Kernels, dir: []const u8) !void {
        var b: [4096]u8 = undefined;
        const p = try std.fmt.bufPrint(&b, "{s}/aot.json", .{std.mem.trimEnd(u8, dir, "/")});
        const hex = sha256File(p) catch |e| {
            if (e == error.FileNotFound) return;
            return e;
        };
        try k.addHash("aot.json", hex);
    }

    fn at(k: *const Kernels, i: u8) []const u8 {
        return k.lines[i][0..k.lens[i]];
    }

    fn before(k: *const Kernels, a: u8, b: u8) bool {
        return std.mem.order(u8, k.at(a), k.at(b)) == .lt;
    }

    pub fn digest(k: *const Kernels) [64]u8 {
        var order: [max]u8 = undefined;
        for (order[0..k.n], 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u8, order[0..k.n], k, before);
        var h = Sha256.init(.{});
        for (order[0..k.n]) |i| {
            h.update(k.at(i));
            h.update("\n");
        }
        return std.fmt.bytesToHex(h.finalResult(), .lower);
    }
};

/// This node's identity with its own storage, built once at the handshake by the role.
pub const Local = struct {
    role: []const u8,
    arch_buf: [48]u8 = undefined,
    arch_len: usize = 0,
    config: [64]u8 = undefined,
    ckpt_buf: [256]u8 = undefined,
    ckpt_len: usize = 0,
    exe: [64]u8 = undefined,
    kernels: [64]u8 = undefined,
    src: []const u8 = source_id,

    pub fn identity(l: *const Local) Identity {
        return .{ .role = l.role, .arch = l.arch_buf[0..l.arch_len], .config = &l.config, .ckpt = l.ckpt_buf[0..l.ckpt_len], .exe = &l.exe, .kernels = &l.kernels, .src = l.src };
    }
};

/// `role` on this machine with a GPU of compute capability `sm`: `<cpu>/sm_<sm>`, `model_dir`/config.json's sha256 and the
/// directory's last element, /proc/self/exe's sha256, `k`'s digest and -Dsource_id.
pub fn local(role: []const u8, sm: u32, model_dir: []const u8, k: *const Kernels) !Local {
    var l: Local = .{ .role = role };
    l.arch_len = (try std.fmt.bufPrint(&l.arch_buf, "{s}/sm_{d}", .{ @tagName(builtin.cpu.arch), sm })).len;
    var b: [4096]u8 = undefined;
    l.config = try sha256File(try std.fmt.bufPrint(&b, "{s}/config.json", .{std.mem.trimEnd(u8, model_dir, "/")}));
    const c = std.fs.path.basename(model_dir);
    if (c.len > l.ckpt_buf.len) return error.NameTooLong;
    @memcpy(l.ckpt_buf[0..c.len], c);
    l.ckpt_len = c.len;
    l.exe = try sha256File("/proc/self/exe");
    l.kernels = k.digest();
    return l;
}

/// The most nodes a judgement covers (the AFD lane's three; the pair's two).
pub const max_nodes = 3;
/// An identity line's bytes at most (a 256-byte checkpoint name escaped, with the rest).
pub const line_max = 2048;
/// A table's bytes at most: max_nodes lines of a label and an identity line.
pub const table_max = 4096;

/// A node in a table: its label (`attention`, `expert-0`, `rank-1`, ...), its line as sent, and the line's fields.
pub const Entry = struct { label: []const u8, text: []const u8, id: Identity };

/// The lines every node judges: `<label> <identity line>` a node, in order.
pub fn tableOf(buf: []u8, nodes: []const [2][]const u8) ![]const u8 {
    var w: std.Io.Writer = .fixed(buf);
    for (nodes) |x| try w.print("{s} {s}\n", .{ x[0], x[1] });
    return w.buffered();
}

pub fn parseTable(text: []const u8, out: *[max_nodes]Entry) error{BadIdentity}![]Entry {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |ln| {
        if (ln.len == 0) continue;
        if (n == max_nodes) return error.BadIdentity;
        const sp = std.mem.indexOfScalar(u8, ln, ' ') orelse return error.BadIdentity;
        out[n] = .{ .label = ln[0..sp], .text = ln[sp + 1 ..], .id = try Identity.parse(ln[sp + 1 ..]) };
        n += 1;
    }
    if (n < 2) return error.BadIdentity;
    return out[0..n];
}

/// What every node concludes from the same table: refused or not, which fields differ (each named with both nodes'
/// values), and the WARN for nodes built without -Dsource_id.
pub const Verdict = struct {
    refused: bool = false,
    fields: [5]bool = @splat(false),
    why_buf: [4096]u8 = undefined,
    why_len: usize = 0,
    warn_buf: [256]u8 = undefined,
    warn_len: usize = 0,

    pub fn why(v: *const Verdict) []const u8 {
        return v.why_buf[0..v.why_len];
    }

    pub fn warning(v: *const Verdict) ?[]const u8 {
        return if (v.warn_len == 0) null else v.warn_buf[0..v.warn_len];
    }

    pub fn differs(v: *const Verdict, f: Field) bool {
        return v.fields[@backingInt(f)];
    }

    fn refuse(v: *Verdict, comptime fmt: []const u8, args: anytype) void {
        v.refused = true;
        var w: std.Io.Writer = .fixed(v.why_buf[v.why_len..]);
        if (v.why_len > 0) w.writeAll("; ") catch {};
        w.print(fmt, args) catch {};
        v.why_len += w.buffered().len;
    }

    fn differ(v: *Verdict, f: Field, a: Entry, b: Entry) void {
        v.fields[@backingInt(f)] = true;
        if (f == .exe or f == .kernels) {
            v.refuse(name ++ ": {t} differs: {s} {s}, {s} {s} (both role {s}, arch {s})", .{ f, a.label, a.id.get(f), b.label, b.id.get(f), a.id.role, a.id.arch });
        } else v.refuse(name ++ ": {t} differs: {s} {s}, {s} {s}", .{ f, a.label, a.id.get(f), b.label, b.id.get(f) });
    }
};

/// The agreement over a table's nodes (fingerprint.zig's header): the same verdict on every node that judges it.
pub fn judge(e: []const Entry) Verdict {
    var v: Verdict = .{};
    for ([_]Field{ .config, .ckpt }) |f| for (e[1..]) |x| if (!std.mem.eql(u8, x.id.get(f), e[0].id.get(f))) {
        v.differ(f, e[0], x);
        break;
    };
    for ([_]Field{ .exe, .kernels }) |f| pairs: for (e, 0..) |a, i| for (e[i + 1 ..]) |b| {
        if (!std.mem.eql(u8, a.id.role, b.id.role) or !std.mem.eql(u8, a.id.arch, b.id.arch)) continue;
        if (!std.mem.eql(u8, a.id.get(f), b.id.get(f))) {
            v.differ(f, a, b);
            break :pairs;
        }
    };
    var w: std.Io.Writer = .fixed(&v.warn_buf);
    var unset_n: usize = 0;
    for (e) |x| if (std.mem.eql(u8, x.id.src, unset)) {
        w.print("{s}{s}", .{ if (unset_n == 0) name ++ ": src unset on " else ", ", x.label }) catch {};
        unset_n += 1;
    };
    if (unset_n > 0) {
        w.writeAll(" (built without -Dsource_id): src not compared") catch {};
        v.warn_len = w.buffered().len;
    } else for (e[1..]) |x| if (!std.mem.eql(u8, x.id.src, e[0].id.src)) {
        v.differ(.src, e[0], x);
        break;
    };
    return v;
}

/// The verdict on a table as received: a table that does not parse is refused in words; with `mine` (an expert's own
/// label and line), its line must be in the table unchanged.
pub fn verdictOf(table: []const u8, mine: ?[2][]const u8) Verdict {
    var es: [max_nodes]Entry = undefined;
    const e = parseTable(table, &es) catch {
        var v: Verdict = .{};
        v.refuse(name ++ ": an identity table that does not parse ({d} bytes)", .{table.len});
        return v;
    };
    if (mine) |m| {
        const back = for (e) |x| {
            if (std.mem.eql(u8, x.label, m[0])) break x.text;
        } else "";
        if (!std.mem.eql(u8, back, m[1])) {
            var v: Verdict = .{};
            v.refuse(name ++ ": {s}'s line came back as \"{s}\", sent \"{s}\"", .{ m[0], back, m[1] });
            return v;
        }
    }
    return judge(e);
}

const testing = std.testing;

fn sample(role: []const u8, arch: []const u8, exe: []const u8, src: []const u8) Identity {
    return .{ .role = role, .arch = arch, .config = "c0", .ckpt = "mia-exl3-4bpw-tensorfold-078455ff", .exe = exe, .kernels = exe, .src = src };
}

test "rank_fingerprint: TF_GLM_FINGERPRINT takes 1 or 0 (or unset); anything else is refused" {
    try testing.expectEqual(false, try parse(null));
    try testing.expectEqual(false, try parse("0"));
    try testing.expectEqual(true, try parse("1"));
    for ([_][]const u8{ "", "2", "on", "true", " 1", "1 ", "01" }) |v| try testing.expectError(error.BadGlmFingerprint, parse(v));
    if (!@hasDecl(options, "source_id")) try testing.expectEqualStrings(unset, source_id); // a runtime without -Dsource_id's option
    try testing.expect(source_id.len > 0);
}

test "rank_fingerprint: an identity line round-trips, in key order, values escaped" {
    var b: [512]u8 = undefined;
    const id: Identity = .{ .role = "expert", .arch = "aarch64/sm_121", .config = "ab", .ckpt = "my ckpt=1%", .exe = "e", .kernels = "k", .src = "unset" };
    const l = try id.line(&b);
    try testing.expectEqualStrings("role=expert arch=aarch64/sm_121 config=ab ckpt=my%20ckpt%3D1%25 exe=e kernels=k src=unset", l);
    const back = try Identity.parse(l);
    try testing.expectEqualStrings("my%20ckpt%3D1%25", back.ckpt);
    try testing.expectEqualStrings("aarch64/sm_121", back.arch);
    try testing.expectEqualStrings("ab", back.get(.config));
    for ([_][]const u8{ "", "role=a", "arch=x role=a config= ckpt= exe= kernels= src=", "role=a arch=b config=c ckpt=d exe=e kernels=f src=g extra=1" }) |bad| try testing.expectError(error.BadIdentity, Identity.parse(bad));
}

test "rank_fingerprint: the kernels digest is over sorted name lines (order-free), binds each name to its bytes, and takes aot.json" {
    var a: Kernels = .{};
    try a.add("glm_kda", "kda bytes");
    try a.add("fn_xchg", "xchg bytes");
    var b: Kernels = .{};
    try b.add("fn_xchg", "xchg bytes");
    try b.add("glm_kda", "kda bytes");
    try testing.expectEqualStrings(&a.digest(), &b.digest());
    var swapped: Kernels = .{};
    try swapped.add("glm_kda", "xchg bytes");
    try swapped.add("fn_xchg", "kda bytes");
    try testing.expect(!std.mem.eql(u8, &a.digest(), &swapped.digest()));
    // the digest of two lines, written out: sha256("fn_xchg <sha256(xchg bytes)>\nglm_kda <sha256(kda bytes)>\n")
    var s1: [32]u8 = undefined;
    var s2: [32]u8 = undefined;
    Sha256.hash("xchg bytes", &s1, .{});
    Sha256.hash("kda bytes", &s2, .{});
    var t: [256]u8 = undefined;
    const text = try std.fmt.bufPrint(&t, "fn_xchg {s}\nglm_kda {s}\n", .{ &std.fmt.bytesToHex(s1, .lower), &std.fmt.bytesToHex(s2, .lower) });
    var want: [32]u8 = undefined;
    Sha256.hash(text, &want, .{});
    try testing.expectEqualStrings(&std.fmt.bytesToHex(want, .lower), &a.digest());
    const n = a.n;
    try a.addTriton("/nonexistent-rank_fingerprint_dir"); // a set without aot.json: no line
    try testing.expectEqual(n, a.n);
    try testing.expectError(error.BadKernelName, a.add("two words", "x"));
}

test "rank_fingerprint: a file's sha256 (the abc vector), the checkpoint name with a trailing slash, the local identity's fields" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "abc" });
    var pb: [4096]u8 = undefined;
    const dir_len = try tmp.dir.realPath(testing.io, &pb);
    const dir = pb[0..dir_len];
    var fb: [4096]u8 = undefined;
    const hex = try sha256File(try std.fmt.bufPrint(&fb, "{s}/config.json", .{dir}));
    try testing.expectEqualStrings("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", &hex);
    try testing.expectError(error.FileNotFound, sha256File("/nonexistent-rank_fingerprint/config.json"));
    var k: Kernels = .{};
    try k.add("glm_kda", "x");
    var db: [4200]u8 = undefined;
    const l = try local("expert", 121, try std.fmt.bufPrint(&db, "{s}/", .{dir}), &k);
    const id = l.identity();
    try testing.expectEqualStrings(&hex, id.config);
    try testing.expectEqualStrings(std.fs.path.basename(dir), id.ckpt);
    try testing.expectEqualStrings(&try sha256File("/proc/self/exe"), id.exe);
    try testing.expectEqualStrings(&k.digest(), id.kernels);
    try testing.expect(std.mem.endsWith(u8, id.arch, "/sm_121"));
    try testing.expectEqualStrings(unset, id.src);
}

test "rank_fingerprint: judge: config and ckpt on every node; exe and kernels within a role and arch; src set everywhere or one WARN" {
    var b: [3][512]u8 = undefined;
    const ids = [3]Identity{ sample("attention", "x86_64/sm_120", "A", "c1"), sample("expert", "aarch64/sm_121", "E", "c1"), sample("expert", "aarch64/sm_121", "E", "c1") };
    var t: [table_max]u8 = undefined;
    const table = try tableOf(&t, &.{ .{ "attention", try ids[0].line(&b[0]) }, .{ "expert-0", try ids[1].line(&b[1]) }, .{ "expert-1", try ids[2].line(&b[2]) } });
    const ok = verdictOf(table, null);
    try testing.expect(!ok.refused); // the attention's exe and kernels differ from the experts': another role, accepted
    try testing.expect(ok.warning() == null);
    try testing.expect(!verdictOf(table, .{ "expert-1", try ids[2].line(&b[2]) }).refused);
    const altered = verdictOf(table, .{ "expert-1", "role=expert arch=aarch64/sm_121 config=c0 ckpt=x exe=E kernels=E src=c1" });
    try testing.expect(altered.refused and std.mem.indexOf(u8, altered.why(), "expert-1's line came back") != null);
    try testing.expect(verdictOf("junk", null).refused);
}

test "rank_fingerprint: the longest lines fit: a 256-byte checkpoint name that escapes to 768 bytes, three such lines in one table, a refusal naming two of them" {
    const long: [256]u8 = @splat(' ');
    var id = sample("expert", "aarch64/sm_121", "E", "c1");
    id.ckpt = &long;
    var b: [3][line_max]u8 = undefined;
    const l0 = try id.line(&b[0]);
    try testing.expect(l0.len > 768 and l0.len < line_max);
    var other = id;
    other.config = "c9";
    var t: [table_max]u8 = undefined;
    const table = try tableOf(&t, &.{ .{ "expert-0", l0 }, .{ "expert-1", try id.line(&b[1]) }, .{ "attention", try other.line(&b[2]) } });
    const v = verdictOf(table, .{ "expert-0", l0 });
    try testing.expect(v.refused and v.differs(.config));
    var es: [max_nodes]Entry = undefined;
    var want: [768]u8 = undefined;
    for (0..256) |i| @memcpy(want[3 * i ..][0..3], "%20");
    try testing.expectEqualStrings(&want, (try parseTable(table, &es))[0].id.ckpt);
}
