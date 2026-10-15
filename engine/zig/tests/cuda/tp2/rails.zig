//! This node's NCCL rails from sysfs, as Mia's launcher pairs them (nodes.sh:104-129): the link's RDMA device first, then its PCIe twin paired by name.
//! Source: MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, scripts/nodes.sh rails() and link_info() (Apache-2.0), in Zig; the subnet scan for a second cabled port is left out.
//! A DGX Spark's QSFP port reaches the GB10 over two PCIe x4 links, so one cable shows two netdevs (enp1s0f0np0, enP2p1s0f0np0): same fN/npM tail, another PCIe prefix.
const std = @import("std");

pub const max_rails = 2;

pub const Rail = struct { netdev: []const u8, rdma: []const u8, gid_index: u32, gid: [16]u8 };

pub const Rails = struct {
    rails: [max_rails]Rail = undefined,
    n: usize = 0,
    /// names and paths live here; free with deinit
    arena: std.heap.ArenaAllocator,

    pub fn deinit(r: *Rails) void {
        r.arena.deinit();
    }

    pub fn slice(r: *const Rails) []const Rail {
        return r.rails[0..r.n];
    }

    /// NCCL_IB_HCA's value: the rails' RDMA devices, the link's first.
    pub fn hcaList(r: *const Rails, buf: []u8) ![]const u8 {
        var w: std.Io.Writer = .fixed(buf);
        for (r.slice(), 0..) |x, i| try w.print("{s}{s}", .{ if (i > 0) "," else "", x.rdma });
        return w.buffered();
    }
};

pub const Error = error{ NoRdmaDevice, NoRoceV2Gid, OutOfMemory };

/// A small sysfs file's text (trimmed), or null when it does not read (an empty GID slot's type fails with EINVAL).
fn readSmall(a: std.mem.Allocator, path: []const u8) ?[]const u8 {
    const z = a.dupeSentinel(u8, path, 0) catch return null;
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [256]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return a.dupe(u8, std.mem.trim(u8, buf[0..@intCast(n)], " \n")) catch null;
}

/// The first entry of a directory, by name (`ls | head -1`), or null.
fn firstEntry(a: std.mem.Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var it = dir.iterate();
    var first: ?[]const u8 = null;
    while (it.next(io) catch null) |e| {
        if (first == null or std.mem.order(u8, e.name, first.?) == .lt) first = a.dupe(u8, e.name) catch return first;
    }
    return first;
}

/// sysfs's GID text as bytes.
fn parseGid(text: []const u8) ?[16]u8 {
    var g: [16]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, ':');
    var i: usize = 0;
    while (it.next()) |part| : (i += 1) {
        if (i == 8 or part.len != 4) return null;
        const v = std.fmt.parseInt(u16, part, 16) catch return null;
        std.mem.writeInt(u16, g[2 * i ..][0..2], v, .big);
    }
    return if (i == 8) g else null;
}

/// An IPv4-mapped GID (::ffff:a.b.c.d), the form a RoCE v2 entry of an IPv4 address takes.
pub fn ipv4Of(g: [16]u8) ?[4]u8 {
    if (!std.mem.allEqual(u8, g[0..10], 0) or g[10] != 0xff or g[11] != 0xff) return null;
    return g[12..16].*;
}

/// GID `index` of `rdma`'s port 1 when it is RoCE v2 and non-zero.
fn v2Gid(a: std.mem.Allocator, sys: []const u8, rdma: []const u8, index: u32) ?[16]u8 {
    const t = readSmall(a, std.fmt.allocPrint(a, "{s}/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ sys, rdma, index }) catch return null) orelse return null;
    if (std.mem.indexOf(u8, t, "v2") == null) return null;
    const g = parseGid(readSmall(a, std.fmt.allocPrint(a, "{s}/class/infiniband/{s}/ports/1/gids/{d}", .{ sys, rdma, index }) catch return null) orelse return null) orelse return null;
    return if (std.mem.allEqual(u8, &g, 0)) null else g;
}

/// A netdev's RDMA device when it is up and has one.
fn rdmaOf(a: std.mem.Allocator, io: std.Io, sys: []const u8, netdev: []const u8) ?[]const u8 {
    const state = readSmall(a, std.fmt.allocPrint(a, "{s}/class/net/{s}/operstate", .{ sys, netdev }) catch return null) orelse return null;
    if (!std.mem.eql(u8, state, "up")) return null;
    return firstEntry(a, io, std.fmt.allocPrint(a, "{s}/class/net/{s}/device/infiniband", .{ sys, netdev }) catch return null);
}

/// The fN/npM tail a Spark-style name ends in (the text after its last 'f'), or null for other names.
pub fn twinTail(name: []const u8) ?[]const u8 {
    const f = std.mem.lastIndexOfScalar(u8, name, 'f') orelse return null;
    const tail = name[f + 1 ..];
    const np = std.mem.indexOf(u8, tail, "np") orelse return null;
    if (np == 0 or np + 2 == tail.len) return null;
    for (tail[0..np]) |c| if (!std.ascii.isDigit(c)) return null;
    for (tail[np + 2 ..]) |c| if (!std.ascii.isDigit(c)) return null;
    return tail;
}

/// The rails of the link on `netdev` under `sys` (normally "/sys"): its device with GID `gid` (null: its lowest IPv4 RoCE v2 entry), then up twins with a v2 GID there.
pub fn find(gpa: std.mem.Allocator, io: std.Io, sys: []const u8, netdev: []const u8, gid: ?u32) Error!Rails {
    var r: Rails = .{ .arena = .init(gpa) };
    errdefer r.deinit();
    const a = r.arena.allocator();
    const own = firstEntry(a, io, try std.fmt.allocPrint(a, "{s}/class/net/{s}/device/infiniband", .{ sys, netdev })) orelse return error.NoRdmaDevice;
    const index: u32 = gid orelse blk: {
        for (0..256) |i| if (v2Gid(a, sys, own, @intCast(i))) |g| if (ipv4Of(g) != null) break :blk @intCast(i);
        return error.NoRoceV2Gid;
    };
    const own_gid = v2Gid(a, sys, own, index) orelse return error.NoRoceV2Gid;
    r.rails[0] = .{ .netdev = try a.dupe(u8, netdev), .rdma = own, .gid_index = index, .gid = own_gid };
    r.n = 1;
    const tail = twinTail(netdev) orelse return r;
    var dir = std.Io.Dir.cwd().openDir(io, try std.fmt.allocPrint(a, "{s}/class/net", .{sys}), .{ .iterate = true }) catch return r;
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (std.mem.eql(u8, e.name, netdev) or e.name.len <= tail.len + 1 or !std.mem.endsWith(u8, e.name, tail) or e.name[e.name.len - tail.len - 1] != 'f') continue;
        try names.append(a, try a.dupe(u8, e.name));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);
    for (names.items) |name| {
        if (r.n == max_rails) break;
        const rdma = rdmaOf(a, io, sys, name) orelse continue;
        if (std.mem.eql(u8, rdma, own)) continue;
        const g = v2Gid(a, sys, rdma, index) orelse continue;
        r.rails[r.n] = .{ .netdev = name, .rdma = rdma, .gid_index = index, .gid = g };
        r.n += 1;
    }
    return r;
}

test "twin tails: Spark-style names only" {
    try std.testing.expectEqualStrings("0np0", twinTail("enp1s0f0np0").?);
    try std.testing.expectEqualStrings("0np0", twinTail("enP2p1s0f0np0").?);
    try std.testing.expectEqualStrings("1np1", twinTail("enp1s0f1np1").?);
    try std.testing.expect(twinTail("eno1") == null);
    try std.testing.expect(twinTail("enp1s0f0") == null);
    try std.testing.expect(twinTail("ib0") == null);
}

test "rails: the link's device, then its PCIe twin by name; a down, foreign-port or v1-only twin is left out" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const root = try std.fmt.allocPrint(a, ".tf-rails-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer a.free(root);
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteTree(io, root) catch {};
    const v4a = "0000:0000:0000:0000:0000:ffff:c000:020e\n";
    const v4b = "0000:0000:0000:0000:0000:ffff:c633:640e\n";
    const ll = "fe80:0000:0000:0000:5aa2:e1ff:fe01:0203\n";
    const files = [_][2][]const u8{
        .{ "class/net/enp1s0f0np0/operstate", "up\n" },
        .{ "class/net/enp1s0f0np0/device/infiniband/rocep1s0f0/.keep", "" },
        .{ "class/net/enP2p1s0f0np0/operstate", "up\n" },
        .{ "class/net/enP2p1s0f0np0/device/infiniband/roceP2p1s0f0/.keep", "" },
        .{ "class/net/enp1s0f1np1/operstate", "up\n" },
        .{ "class/net/enp1s0f1np1/device/infiniband/rocep1s0f1/.keep", "" },
        .{ "class/net/eno1/operstate", "up\n" },
        .{ "class/infiniband/rocep1s0f0/ports/1/gids/0", ll },
        .{ "class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/0", "IB/RoCE v1\n" },
        .{ "class/infiniband/rocep1s0f0/ports/1/gids/1", ll },
        .{ "class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/1", "RoCE v2\n" },
        .{ "class/infiniband/rocep1s0f0/ports/1/gids/2", v4a },
        .{ "class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/2", "IB/RoCE v1\n" },
        .{ "class/infiniband/rocep1s0f0/ports/1/gids/3", v4a },
        .{ "class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3", "RoCE v2\n" },
        .{ "class/infiniband/roceP2p1s0f0/ports/1/gids/3", v4b },
        .{ "class/infiniband/roceP2p1s0f0/ports/1/gid_attrs/types/3", "RoCE v2\n" },
        .{ "class/infiniband/rocep1s0f1/ports/1/gids/3", v4b },
        .{ "class/infiniband/rocep1s0f1/ports/1/gid_attrs/types/3", "RoCE v2\n" },
    };
    for (files) |f| {
        const p = try std.fmt.allocPrint(a, "{s}/{s}", .{ root, f[0] });
        defer a.free(p);
        try cwd.createDirPath(io, std.fs.path.dirname(p).?);
        try cwd.writeFile(io, .{ .sub_path = p, .data = f[1] });
    }
    var buf: [128]u8 = undefined;
    {
        var r = try find(a, io, root, "enp1s0f0np0", null);
        defer r.deinit();
        try std.testing.expectEqual(@as(usize, 2), r.n);
        try std.testing.expectEqual(@as(u32, 3), r.rails[0].gid_index); // the IPv4 address's v2 entry, not the link-local one
        try std.testing.expectEqual([4]u8{ 192, 0, 2, 14 }, ipv4Of(r.rails[0].gid).?);
        try std.testing.expectEqualStrings("rocep1s0f0,roceP2p1s0f0", try r.hcaList(&buf));
    }
    {
        var r = try find(a, io, root, "enP2p1s0f0np0", 3); // from the twin's side: the same pair, the link's device first
        defer r.deinit();
        try std.testing.expectEqualStrings("roceP2p1s0f0,rocep1s0f0", try r.hcaList(&buf));
    }
    const down = try std.fmt.allocPrint(a, "{s}/class/net/enP2p1s0f0np0/operstate", .{root});
    defer a.free(down);
    try cwd.writeFile(io, .{ .sub_path = down, .data = "down\n" });
    {
        var r = try find(a, io, root, "enp1s0f0np0", 3);
        defer r.deinit();
        try std.testing.expectEqualStrings("rocep1s0f0", try r.hcaList(&buf));
    }
    try cwd.writeFile(io, .{ .sub_path = down, .data = "up\n" });
    const v1 = try std.fmt.allocPrint(a, "{s}/class/infiniband/roceP2p1s0f0/ports/1/gid_attrs/types/3", .{root});
    defer a.free(v1);
    try cwd.writeFile(io, .{ .sub_path = v1, .data = "IB/RoCE v1\n" });
    {
        var r = try find(a, io, root, "enp1s0f0np0", 3);
        defer r.deinit();
        try std.testing.expectEqual(@as(usize, 1), r.n);
    }
    try std.testing.expectError(error.NoRoceV2Gid, find(a, io, root, "enp1s0f0np0", 0));
    try std.testing.expectError(error.NoRdmaDevice, find(a, io, root, "eno1", null));
}
