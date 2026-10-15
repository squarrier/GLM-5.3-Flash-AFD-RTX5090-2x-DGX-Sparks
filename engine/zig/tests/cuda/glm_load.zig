//! `tf-glm-load`: GLM-5.3-Flash's TP rank from a checkpoint folder: its plan, its reads timed, its digests, or a GPU load.

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");

const usage =
    \\usage: tf-glm-load <command> DIR [options]
    \\  plan DIR     each rank's tensors from the index and headers alone: bytes kept and read, unread tensors, memory
    \\  read DIR     the rank's reads as the loader makes them (O_DIRECT, io_uring), landed in no memory: seconds and GB/s
    \\  hash DIR     the rank's groups as a load fills them, hashed on the CPU (no GPU): one sha256 per group and in all
    \\  load DIR     the rank's weights onto GPU 0 (CUDA): seconds, device bytes, MemAvailable; --hash reads them back
    \\  synth DIR    a small synthetic checkpoint in DIR (fixtures_cuda_split.py splits it with python-0.6's split.py)
    \\  exl3 DIR     Mia's EXL3 checkpoint by split.py's rules, each rank: reads and kept bytes (--spans FILE: every copy)
    \\options: --rank R (plan: both when omitted) --ranks N (2) --split python|metal (python) --layers N (all)
    \\         --gap BYTES (65536) --slot-mib N (64) --slots N (8) --no-ring --hash --shards N (synth: 3)
    \\         --threads N (groups hashed at once, each a buffer of the largest group: hash 4, load --hash 1)
    \\         --context N (262144) --prompt-cache-gib G (0) --avail-gib G (104.7) --mtp 0|1 (1)
    \\
;

const Args = struct {
    dir: [:0]const u8 = "",
    rank: ?u32 = null,
    ranks: u32 = 2,
    how: glm.split.Split = .python,
    layers: ?u32 = null,
    read: glm.read.Options = .{},
    hash: bool = false,
    threads: ?u32 = null, // on GB10 a device hash's buffers sit beside the weights in the same memory, so load --hash takes one
    shards: u32 = 3,
    context: u64 = 262144,
    kept_gib: f64 = 0, // the server's --prompt-cache-gib: kept prompt states beside the weights
    avail_gib: f64 = 104.7, // an idle GB10 Spark's MemAvailable (the lowest of our lab's samples, 104.7-106.0 GiB)
    mtp: bool = true,
    spans: ?[:0]const u8 = null, // exl3: every copy into this file
};

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{};
    if (args.len < 3) return error.Usage;
    a.dir = args[2];
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--no-ring")) {
            a.read.ring = false;
            continue;
        }
        if (std.mem.eql(u8, k, "--hash")) {
            a.hash = true;
            continue;
        }
        if (i + 1 >= args.len) return error.Usage;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, k, "--rank")) a.rank = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--spans")) a.spans = v else if (std.mem.eql(u8, k, "--ranks")) a.ranks = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--split")) a.how = glm.split.Split.parse(v) orelse return error.Usage else if (std.mem.eql(u8, k, "--layers")) a.layers = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--gap")) a.read.gap = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--slot-mib")) a.read.slot_bytes = (try std.fmt.parseInt(usize, v, 10)) << 20 else if (std.mem.eql(u8, k, "--slots")) a.read.slots = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--shards")) a.shards = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--threads")) a.threads = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--context")) a.context = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--prompt-cache-gib")) a.kept_gib = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--avail-gib")) a.avail_gib = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, k, "--mtp")) a.mtp = !std.mem.eql(u8, v, "0") else return error.Usage;
    }
    return a;
}

fn config(gpa: std.mem.Allocator, a: Args) !glm.Config {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const path = try std.fmt.allocPrintSentinel(arena.allocator(), "{s}/config.json", .{a.dir}, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.NoConfig;
    defer _ = std.c.close(fd);
    const buf = try arena.allocator().alloc(u8, 1 << 20);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n <= 0) return error.NoConfig;
    var c = try glm.config.parse(gpa, buf[0..@intCast(n)]);
    if (a.layers) |l| try glm.config.subset(&c, l);
    return c;
}

/// MemAvailable now, in bytes (0 where /proc/meminfo is unreadable).
fn memAvailable() u64 {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return 0;
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(n)], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        return (std.fmt.parseInt(u64, f.next() orelse "0", 10) catch 0) * 1024;
    };
    return 0;
}

fn gb(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / 1e9;
}

fn gibs(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(glm.memory.gib));
}

fn planRank(gpa: std.mem.Allocator, a: Args, base: glm.Config, ix: *glm.index.Index, rank: u32) !void {
    const c = try glm.split.rankConfig(base, rank, a.ranks);
    var p = try glm.split.plan(gpa, &c, &ix.names, a.how);
    defer p.deinit();
    var s = try glm.read.schedule(gpa, p.copies.items, a.read);
    defer s.deinit();
    var kept: u64 = 0;
    for (p.copies.items) |cp| kept += cp.kept();
    var largest: u64 = 0;
    for (p.groups.items) |g| largest = @max(largest, g);
    const out = std.debug.print;
    out("rank {d}/{d} ({t} split): {d} tensors read, {d} unread; {d} groups, largest {d:.3} GB\n", .{ rank, a.ranks, a.how, p.used.count(), ix.unread(&p.used), p.groups.items.len, gb(largest) });
    out("rank {d}/{d}: kept {d} B ({d:.3} GB, {d:.2} GiB) in groups of {d} B; source {d} B ({d:.3} GB)\n", .{ rank, a.ranks, kept, gb(kept), gibs(kept), p.bytes(), p.sourceBytes(), gb(p.sourceBytes()) });
    out("rank {d}/{d}: {d} copies in {d} reads (gap {d} B, slot {d} MiB): {d} B asked ({d:.3} GB), {d} B in 4 KiB blocks\n", .{ rank, a.ranks, s.copies.len, s.reads.len, a.read.gap, a.read.slot_bytes >> 20, s.bytes(), gb(s.bytes()), s.spanBytes() });
    const total: u64 = 127600732 * 1024; // a GB10 Spark's MemTotal (127,600,732 kB in the lab's meminfo)
    const m = glm.memory.plan(&base, .{
        .weights = p.bytes(),
        .staging = @as(u64, a.read.slots) * a.read.slot_bytes,
        .window = a.context,
        .ranks = a.ranks,
        .mtp = a.mtp,
        .kept = @intFromFloat(a.kept_gib * @as(f64, @floatFromInt(glm.memory.gib))),
        .total = total,
    });
    const avail: u64 = @intFromFloat(a.avail_gib * @as(f64, @floatFromInt(glm.memory.gib)));
    out("rank {d}/{d} memory at {d} tokens (GiB): weights {d:.2}, caches {d:.2} fixed + {d:.2} growing, kept {d:.2}; serving {d:.2}, startup {d:.2}\n", .{ rank, a.ranks, a.context, gibs(m.weights), gibs(m.fixed), gibs(m.cache), gibs(m.kept), gibs(m.serving), gibs(m.startup) });
    out("rank {d}/{d} memory: + {d:.2} context = peak {d:.2} GiB; MemAvailable {d:.1} -> {d:.2} GiB left (engine reserve {d:.2})\n", .{ rank, a.ranks, gibs(m.context), gibs(m.peak), a.avail_gib, @as(f64, @floatFromInt(m.left(avail))) / @as(f64, @floatFromInt(glm.memory.gib)), gibs(m.reserve) });
}

fn printDigests(what: []const u8, rank: u32, ds: []const glm.digest.Digest) void {
    for (ds) |d| std.debug.print("{s} rank {d} group {d}: {d} B sha256 {s}\n", .{ what, rank, d.g, d.bytes, &std.fmt.bytesToHex(d.sha, .lower) });
    std.debug.print("{s} rank {d}: {d} groups, all {s}\n", .{ what, rank, ds.len, &std.fmt.bytesToHex(glm.digest.combined(ds), .lower) });
}

fn readRank(gpa: std.mem.Allocator, a: Args, base: glm.Config, ix: *glm.index.Index, rank: u32, mode: enum { read, hash, load }) !void {
    const c = try glm.split.rankConfig(base, rank, a.ranks);
    const before = memAvailable();
    if (mode != .load) {
        var p = try glm.split.plan(gpa, &c, &ix.names, a.how);
        defer p.deinit();
        if (mode == .hash) {
            const ds = try glm.digest.fromFiles(gpa, ix.shards, &p, a.read, a.threads orelse 4);
            defer gpa.free(ds);
            return printDigests("files", rank, ds);
        }
        var s = try glm.read.schedule(gpa, p.copies.items, a.read);
        defer s.deinit();
        var sink: glm.read.NullSink = .{};
        const st = try glm.read.run(gpa, ix.shards, &s, sink.sink(), a.read);
        std.debug.print("read rank {d}/{d}: {d} reads, {d} B in {d:.2} s = {d:.3} GB/s (ring {}, direct {}, retried {d}); kept {d} B; MemAvailable {d:.2} -> {d:.2} GiB\n", .{ rank, a.ranks, st.reads, st.bytes, @as(f64, @floatFromInt(st.ns)) / 1e9, st.gbPerS(), st.ring, st.direct, st.retried, st.kept, gibs(before), gibs(memAvailable()) });
        return;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var w = try glm.weights.load(gpa, &driver, a.dir, &c, a.how, a.read);
    defer w.deinit();
    const u = cuda.usage(false);
    std.debug.print("load rank {d}/{d} ({t} split): {d} reads, {d} B in {d:.2} s = {d:.3} GB/s (ring {}, direct {}, retried {d}); device {d:.2} GiB (peak {d:.2}); MemAvailable {d:.2} -> {d:.2} GiB; unread {d}\n", .{ rank, a.ranks, a.how, w.stats.reads, w.stats.bytes, @as(f64, @floatFromInt(w.stats.ns)) / 1e9, w.stats.gbPerS(), w.stats.ring, w.stats.direct, w.stats.retried, gibs(u.device), gibs(u.peak), gibs(before), gibs(memAvailable()), w.unread });
    if (a.hash) {
        const ds = try glm.digest.fromDevice(gpa, w.groups, w.plan.groups.items, a.threads orelse 1);
        defer gpa.free(ds);
        printDigests("device", rank, ds);
    }
}

/// The synthetic checkpoint (cuda_synth.tiny) in `dir`, and a list of its files for fixtures_cuda_split.py.
fn synthesize(gpa: std.mem.Allocator, a: Args) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const c = glm.synth.tiny();
    _ = std.c.mkdir(a.dir.ptr, 0o755); // one level, as prompt_imprint makes its own; an existing folder is fine
    const names = try glm.synth.write(arena.allocator(), a.dir, &c, a.shards);
    std.debug.print("synth {s}: {d} shards, {d} language-model tensors and one vision tensor\n", .{ a.dir, a.shards, names.count() });
}

const Place = struct { shard: u32, off: u64, len: u64, name: []const u8 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn byPlace(_: void, x: Place, y: Place) bool {
    return if (x.shard != y.shard) x.shard < y.shard else x.off < y.off;
}

/// The tensor holding a copy's first byte.
fn owner(places: []const Place, shard: u32, off: u64) []const u8 {
    var lo: usize = 0;
    var hi: usize = places.len;
    while (hi - lo > 1) {
        const mid = (lo + hi) / 2;
        if (byPlace({}, .{ .shard = shard, .off = off, .len = 0, .name = "" }, places[mid])) hi = mid else lo = mid;
    }
    const p = places[lo];
    return if (p.shard == shard and off >= p.off and off < p.off + p.len) p.name else "?";
}

/// Mia's EXL3 checkpoint by python-0.6 split.py's rules, each rank: its reads and kept bytes, and with --spans FILE
/// every copy as `rank shard offset length row at width name`, sorted (exl3_split_fixtures.py --spans writes Python's).
fn exl3Plan(gpa: std.mem.Allocator, a: Args, base: glm.Config, ix: *glm.index.Index) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var places: std.ArrayList(Place) = .empty;
    var it = ix.names.iterator();
    while (it.next()) |e| try places.append(ar, .{ .shard = e.value_ptr.shard, .off = e.value_ptr.off, .len = e.value_ptr.len, .name = e.key_ptr.* });
    std.mem.sort(Place, places.items, {}, byPlace);
    var lines: std.ArrayList([]const u8) = .empty;
    for (0..a.ranks) |r| {
        if (a.rank) |only| if (only != r) continue;
        const t0 = nowNs();
        var p = try glm.exl3.split.plan(gpa, &base, &ix.names, @intCast(r), a.ranks);
        defer p.deinit();
        const ns = nowNs() - t0;
        var kept: u64 = 0;
        for (p.copies.items) |cp| kept += cp.kept();
        var moe: u32 = 0;
        var first: ?usize = null;
        for (p.experts, 0..) |e, i| if (e != null) {
            moe += 1;
            if (first == null) first = i;
        };
        const e = p.experts[first.?].?;
        std.debug.print("exl3 rank {d}/{d}: planned in {d:.2} s: {d} tensors read, {d} markers checked by name, {d} unread; {d} groups, {d} copies; read {d} B ({d:.3} GB), kept {d} B ({d:.3} GB, {d:.2} GiB), groups {d} B\n", .{ r, a.ranks, @as(f64, @floatFromInt(ns)) / 1e9, p.used.count() - p.markers, p.markers, ix.unread(&p.used), p.groups.items.len, p.copies.items.len, p.sourceBytes(), gb(p.sourceBytes()), kept, gb(kept), gibs(kept), p.bytes() });
        std.debug.print("exl3 rank {d}/{d}: {d} MoE layers of {d} experts, codebook {t}, gate {d}x{d} k2 {any}, up {d}x{d} k2 {any}, down {d}x{d} k2 {any}\n", .{ r, a.ranks, moe, e.count, e.cb, e.gate.k, e.gate.n, e.gate.k2, e.up.k, e.up.n, e.up.k2, e.down.k, e.down.n, e.down.k2 });
        for (p.copies.items) |cp| try lines.append(ar, try std.fmt.allocPrint(ar, "{d} {d} {d} {d} {d} {d} {d} {s}\n", .{ r, cp.shard, cp.off, cp.len, cp.row, cp.at, cp.width, owner(places.items, cp.shard, cp.off) }));
    }
    const path = a.spans orelse return;
    const Key = struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            var xs = std.mem.tokenizeScalar(u8, x, ' ');
            var ys = std.mem.tokenizeScalar(u8, y, ' ');
            for (0..7) |_| {
                const u = std.fmt.parseInt(u64, xs.next().?, 10) catch 0;
                const v = std.fmt.parseInt(u64, ys.next().?, 10) catch 0;
                if (u != v) return u < v;
            }
            return std.mem.order(u8, x, y) == .lt;
        }
    };
    std.mem.sort([]const u8, lines.items, {}, Key.lt);
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    for (lines.items) |l| if (std.c.write(fd, l.ptr, l.len) != l.len) return error.CannotWrite;
    std.debug.print("exl3: {d} copies -> {s}\n", .{ lines.items.len, path });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const a = parse(argv) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const cmd = argv[1];
    if (std.mem.eql(u8, cmd, "synth")) {
        try synthesize(gpa, a);
        return 0;
    }
    const base = try config(gpa, a);
    var ix = try glm.index.open(gpa, a.dir);
    defer ix.deinit();
    std.debug.print("{s}: {d} shards, {d} language-model tensors, {d} vision ({d} B, never read), {d} stray; {d} of {d} layers{s}\n", .{ a.dir, ix.shards.len, ix.names.count(), ix.others, ix.other_bytes, ix.strays, base.run, base.layers, if (base.mtp > 0) " and the MTP layer" else "" });
    if (std.mem.eql(u8, cmd, "plan")) {
        if (a.rank) |r| try planRank(gpa, a, base, &ix, r) else for (0..a.ranks) |r| try planRank(gpa, a, base, &ix, @intCast(r));
    } else if (std.mem.eql(u8, cmd, "exl3")) {
        try exl3Plan(gpa, a, base, &ix);
    } else if (std.mem.eql(u8, cmd, "read")) {
        try readRank(gpa, a, base, &ix, a.rank orelse return 2, .read);
    } else if (std.mem.eql(u8, cmd, "hash")) {
        try readRank(gpa, a, base, &ix, a.rank orelse return 2, .hash);
    } else if (std.mem.eql(u8, cmd, "load")) {
        try readRank(gpa, a, base, &ix, a.rank orelse return 2, .load);
    } else {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    return 0;
}
