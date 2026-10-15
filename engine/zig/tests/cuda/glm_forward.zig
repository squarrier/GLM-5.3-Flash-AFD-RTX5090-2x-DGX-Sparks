//! `tf-glm-forward`: GLM-5.3-Flash's Zig forward as a tool (qualification_qualification_7c75d777 PLAN §5: B1's `check-weights`, B3's run verbs). check-weights
//! loads a TP rank (qualification_qualification_d914aa63's split and reader, one stored group at a time), packs it into python-0.6's layouts
//! (glm/cuda_pack.zig: glm_pack.cu on GPU 0, or emulated on the CPU with --host), and digests every packed buffer by
//! python's path, against a weights.json of qualification_6607f886's (`table`: path -> shape, dtype, sha256, bytes); --dry compares the
//! plan's paths, shapes, dtypes and bytes from the checkpoint's index and headers alone (a header-only mirror serves).
//! B3's run verbs, `first`, `teacher` and `dump`, are glm_run.zig's (and glm_run_gpu.zig's on a GPU).

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const options = @import("glm_forward_options");
const run_verbs = @import("glm_run.zig");
/// bench's --drafts policies and loop (qualification_qualification_44ba554e.7): the serve open reads them for `serve --drafts` (qualification_qualification_44ba554e.8 day 3)
pub const drafts = @import("glm_drafts.zig");

const pack = glm.pack;
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;

const usage =
    \\usage: tf-glm-forward check-weights DIR --rank R [options]
    \\  DIR's rank R of --ranks (2), the first --layers N backbone layers (all) and the MTP layer, packed into python-0.6's
    \\  layouts and digested by python's paths; --ref weights.json compares each (qualification_6607f886's qualification_qualification_5ab55e74 run: --layers 8)
    \\options: --dry (paths, shapes, dtypes and bytes from the index and headers; no reads) --host (CPU: glm_pack.cu emulated)
    \\         --only S[,S...] (paths containing one of them) --out FILE (the table as weights.json lays it out)
    \\         --trace FILE (with --dry: the pack pass's glm_pack.* events at stand-in addresses, and the split's copies)
    \\         --fatbin FILE (glm_pack.fatbin when the binary embeds none) --keep 0|1 (GPU: hold the packed rank, 1)
    \\         --gap BYTES --slot-mib N --slots N --no-ring (the reader's, as tf-glm-load)
    \\
;

pub const Args = struct {
    dir: [:0]const u8 = "",
    rank: ?u32 = null,
    ranks: u32 = 2,
    layers: ?u32 = null,
    ref: ?[]const u8 = null,
    out: ?[]const u8 = null,
    trace: ?[]const u8 = null,
    only: ?[]const u8 = null,
    fatbin: ?[]const u8 = null,
    dry: bool = false,
    host: bool = false,
    keep: bool = true,
    /// false: python's without_mtp (TF_GLM_MTP=0, its drafts-off load): the MTP layer neither read nor packed
    mtp_layer: bool = true,
    read: glm.read.Options = .{},
};

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{};
    if (args.len < 3 or !std.mem.eql(u8, args[1], "check-weights")) return error.Usage;
    a.dir = args[2];
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (std.mem.eql(u8, k, "--dry")) a.dry = true else if (std.mem.eql(u8, k, "--host")) a.host = true else if (std.mem.eql(u8, k, "--no-ring")) a.read.ring = false else {
            if (i + 1 >= args.len) return error.Usage;
            const v = args[i + 1];
            i += 1;
            if (std.mem.eql(u8, k, "--rank")) a.rank = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--ranks")) a.ranks = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--layers")) a.layers = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, k, "--ref")) a.ref = v else if (std.mem.eql(u8, k, "--out")) a.out = v else if (std.mem.eql(u8, k, "--trace")) a.trace = v else if (std.mem.eql(u8, k, "--only")) a.only = v else if (std.mem.eql(u8, k, "--fatbin")) a.fatbin = v else if (std.mem.eql(u8, k, "--keep")) a.keep = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, k, "--gap")) a.read.gap = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, k, "--slot-mib")) a.read.slot_bytes = (try std.fmt.parseInt(usize, v, 10)) << 20 else if (std.mem.eql(u8, k, "--slots")) a.read.slots = try std.fmt.parseInt(u32, v, 10) else return error.Usage;
        }
    }
    if (a.rank == null or a.rank.? >= a.ranks) return error.Usage;
    return a;
}

/// A file's bytes (config.json, weights.json, a fatbin), at most `most`.
pub fn slurp(a: std.mem.Allocator, path: []const u8, most: usize) ![]u8 {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or end > most) return error.FileTooBig;
    const out = try a.alloc(u8, @intCast(end));
    var done: usize = 0;
    while (done < out.len) {
        const n = std.c.pread(fd, out.ptr + done, out.len - done, @intCast(done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
    return out;
}

/// MemAvailable now, in GiB (0 where /proc/meminfo is unreadable).
pub fn memAvailableGib() f64 {
    var buf: [4096]u8 = undefined;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return 0;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return 0;
    var it = std.mem.tokenizeScalar(u8, buf[0..@intCast(n)], '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeScalar(u8, line["MemAvailable:".len..], ' ');
        return @as(f64, @floatFromInt(std.fmt.parseInt(u64, f.next() orelse "0", 10) catch 0)) / (1024.0 * 1024.0);
    };
    return 0;
}

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1024.0 * 1024.0 * 1024.0);
}

/// A packed buffer's digest, as weights.json's table row.
pub const Row = struct { sha: [32]u8 = undefined, done: bool = false };

/// qualification_765330ce's EXL3 split (cuda_exl3_split.zig) where the tree has it (qualification_qualification_44ba554e-int; not qualification_qualification_44ba554e-fwd-b, whose EXL3 ranks stop here).
const has_exl3 = @hasDecl(glm, "exl3");
const Exl3Split = if (has_exl3) glm.exl3.split.Plan else void;

/// The stored groups of a rank: qualification_qualification_d914aa63's MLX split, or qualification_765330ce's EXL3 split (on the heap: the pack's table writes read it).
pub const Split = union(enum) {
    mlx: glm.split.Plan,
    exl3: *Exl3Split,

    pub fn groups(s: *const Split) []const u64 {
        return switch (s.*) {
            .mlx => |*p| p.groups.items,
            .exl3 => |p| if (comptime has_exl3) p.groups.items else unreachable,
        };
    }
    pub fn copies(s: *const Split) []const glm.split.Copy {
        return switch (s.*) {
            .mlx => |*p| p.copies.items,
            .exl3 => |p| if (comptime has_exl3) p.copies.items else unreachable,
        };
    }
    pub fn bytes(s: *const Split) u64 {
        var n: u64 = 0;
        for (s.groups()) |g| n += g;
        return n;
    }
    pub fn deinit(s: *Split, gpa: std.mem.Allocator) void {
        switch (s.*) {
            .mlx => |*p| p.deinit(),
            .exl3 => |p| if (comptime has_exl3) {
                p.deinit();
                gpa.destroy(p);
            },
        }
    }
};

/// The rank's plans: the split (stored groups) and the pack (python's buffers), for the model of `base` cut to `layers`.
pub const Rank = struct {
    m: glm.model.Model,
    c: glm.Config,
    split: Split,
    pk: pack.Plan,
    unread: usize,

    pub fn deinit(r: *Rank, gpa: std.mem.Allocator) void {
        r.pk.deinit();
        r.split.deinit(gpa);
    }
    /// Whether the split holds the MTP layer (python's Weights.mtp).
    pub fn mtp(r: *const Rank) bool {
        return r.m.mtp_layers > 0;
    }
};

pub fn plans(gpa: std.mem.Allocator, a: Args, ix: *glm.index.Index) !Rank {
    var name: [4096]u8 = undefined;
    const text = try slurp(gpa, try std.fmt.bufPrint(&name, "{s}/config.json", .{a.dir}), 1 << 20);
    defer gpa.free(text);
    var base = try glm.config.parse(gpa, text);
    if (a.layers) |n| try glm.config.subset(&base, n); // qualification_6607f886's 8-layer copy: layers 0..n-1 and the checkpoint's MTP layer
    if (!a.mtp_layer) base.mtp = 0; // python's without_mtp: neither split plans the MTP layer
    const extra = try glm.model.Extra.parse(gpa, text);
    var m = try glm.model.Model.from(&base, extra, a.ranks, a.rank.?);
    if (extra.quant == .exl3) return exl3Plans(gpa, a, ix, m, base);
    if ((try pack.q4.fromEnv()) != null) return error.DenseQ4IsForExl3; // python's TF_GLM_DENSE is EXL3's too
    if (try pack.f8.fromEnv()) return error.F8IsForDenseQ4; // and TF_GLM_F8 the dense-q4 form's
    const c = try glm.split.rankConfig(base, a.rank.?, a.ranks);
    var sp = try glm.split.plan(gpa, &c, &ix.names, .python);
    errdefer sp.deinit();
    if (sp.mtp == null) m.mtp_layers = 0; // python's Weights.mtp only when the split loaded the MTP layer
    if (!a.mtp_layer) try leftOut(gpa, ix, &sp.used, base.layers);
    const unread = ix.unread(&sp.used);
    if (unread != 0 and c.run == c.layers) return error.UnreadCheckpointTensors;
    sp.used.clearAndFree(gpa);
    var r: Rank = .{ .m = m, .c = c, .split = .{ .mlx = sp }, .pk = undefined, .unread = unread };
    r.pk = try pack.plan(gpa, &r.m, &r.c, &r.split.mlx);
    return r;
}

/// python's without_mtp: the MTP layer's tensors (`...layers.<num_hidden_layers>.`) left out of the load on purpose,
/// so they do not count as unread.
fn leftOut(gpa: std.mem.Allocator, ix: *glm.index.Index, used: *std.StringHashMapUnmanaged(void), layers: u32) !void {
    var buf: [32]u8 = undefined;
    const needle = try std.fmt.bufPrint(&buf, ".layers.{d}.", .{layers});
    var it = ix.names.keyIterator();
    while (it.next()) |k| if (std.mem.indexOf(u8, k.*, needle) != null) try used.put(gpa, k.*, {});
}

/// An EXL3 rank: qualification_765330ce's split of the unsplit config (as tf-glm-load exl3), then pack.exl3's plan over it.
fn exl3Plans(gpa: std.mem.Allocator, a: Args, ix: *glm.index.Index, m0: glm.model.Model, base: glm.Config) !Rank {
    if (comptime has_exl3) {
        var m = m0;
        const x = try gpa.create(Exl3Split);
        errdefer gpa.destroy(x);
        x.* = try glm.exl3.split.plan(gpa, &base, &ix.names, a.rank.?, a.ranks);
        errdefer x.deinit();
        if (x.experts[base.layers] == null) m.mtp_layers = 0; // the split loads the MTP layer only when it is stored
        if (!a.mtp_layer) try leftOut(gpa, ix, &x.used, base.layers);
        const unread = ix.unread(&x.used);
        if (unread != 0 and base.run == base.layers) return error.UnreadCheckpointTensors;
        var r: Rank = .{ .m = m, .c = base, .split = .{ .exl3 = x }, .pk = undefined, .unread = unread };
        r.pk = try pack.exl3.plan(gpa, &r.m, &r.c, x);
        errdefer r.pk.deinit();
        const f8_head = try pack.f8.fromEnv(); // TF_GLM_F8=head: the lane's FP8 head (patch 0002), with DENSE=q4 only
        if (try pack.q4.fromEnv()) |dir| { // TF_GLM_DENSE=q4:DIR, Mia's q4 dense (cuda_pack_q4.zig)
            const s = try pack.q4.overlay(gpa, &r.pk, &x.groups, &x.copies, ix, r.m.layers, r.m.mtp_layers > 0, a.rank.?, a.ranks, dir);
            if (f8_head) {
                const saved = pack.f8.apply(&r.pk) catch |e| {
                    std.log.err("glm f8: lm_head is not a BF16 [n, 128 k] copy of its stored rows ({t})", .{e});
                    return e;
                };
                std.log.info("dense q4: {d} projections from {d} files in {s} ({d:.3} GiB stored); lm_head FP8 (make_f8 on the GPU, {d:.3} GiB less), kv_b_proj BF16 as the lane's (TF_GLM_KVB unset)", .{ s.triplets, s.files, dir, gib(s.bytes), gib(saved) });
            } else std.log.info("dense q4: {d} projections from {d} files in {s} ({d:.3} GiB stored); lm_head and kv_b_proj BF16 (the lane runs FP8 there)", .{ s.triplets, s.files, dir, gib(s.bytes) });
        } else if (f8_head) {
            std.log.err("TF_GLM_F8=head is the dense-q4 form's, as the lane's: set TF_GLM_DENSE=q4:DIR too", .{});
            return error.F8IsForDenseQ4;
        }
        return r;
    } else return error.Exl3SplitNotInThisTree;
}

fn wanted(a: Args, path: []const u8) bool {
    const only = a.only orelse return true;
    var it = std.mem.tokenizeScalar(u8, only, ',');
    while (it.next()) |s| if (std.mem.indexOf(u8, path, s) != null) return true;
    return false;
}

/// Each packed buffer's group, in order: the stored groups read one at a time, packed, freed after their last reader.
pub const Run = struct {
    gpa: std.mem.Allocator,
    a: Args,
    ix: *glm.index.Index,
    r: *Rank,
    rows: []Row,
    driver: ?*const cuda.Driver = null,
    stream: ?cuda.Stream = null,
    kernels: ?*const pack.Kernels = null,
    stored: []?Mem,
    packed_groups: std.ArrayList(Mem) = .empty,
    peak_stored: u64 = 0,
    read_ns: u64 = 0,
    pack_ns: u64 = 0,

    pub const Mem = union(enum) { host: []align(256) u8, dev: cuda.DeviceBuffer };

    fn addr(x: Mem) u64 {
        return switch (x) {
            .host => |h| @intFromPtr(h.ptr),
            .dev => |d| d.ptr,
        };
    }
    fn alloc(run: *Run, bytes: u64) !Mem {
        if (run.driver) |d| {
            var b = try cuda.DeviceBuffer.alloc(d, @intCast(@max(bytes, 256)));
            try b.fill8(0, null); // the padding between tensors reads as zeros, as cuda_weights.load leaves it
            return .{ .dev = b };
        }
        const h = try run.gpa.alignedAlloc(u8, .fromByteUnits(256), @intCast(@max(bytes, 256)));
        @memset(h, 0);
        return .{ .host = h };
    }
    pub fn free(run: *Run, x: *Mem) void {
        switch (x.*) {
            .host => |h| run.gpa.free(h),
            .dev => |*d| d.free(),
        }
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const run: *Run = @ptrCast(@alignCast(ctx));
        switch (run.stored[g].?) {
            .host => |h| @memcpy(h[@intCast(dst)..][0..bytes.len], bytes),
            .dev => |d| try d.uploadAsync(@intCast(dst), bytes, run.stream.?.handle),
        }
    }
    fn flush(ctx: *anyopaque) anyerror!void {
        const run: *Run = @ptrCast(@alignCast(ctx));
        if (run.stream) |s| try s.synchronize();
    }

    fn emu(kern: pack.Kern, grid: [3]u32, block: [3]u32, args: *cuda.Args) anyerror!void {
        const rc = tf_glm_pack_emu(@backingInt(kern), &grid, &block, args.pointers().?, &args.sizes, @intCast(args.count));
        if (rc != 0) return error.EmulatorRefused;
    }

    /// Stored group `g` read from the checkpoint.
    fn load(run: *Run, g: u32) !void {
        run.stored[g] = try run.alloc(run.r.split.groups()[g]);
        var mine: std.ArrayList(glm.split.Copy) = .empty;
        defer mine.deinit(run.gpa);
        for (run.r.split.copies()) |cp| if (cp.g == g) try mine.append(run.gpa, cp);
        var s = try glm.read.schedule(run.gpa, mine.items, run.a.read);
        defer s.deinit();
        const st = try glm.read.run(run.gpa, run.ix.shards, &s, .{ .ctx = run, .put = put, .flush = flush }, run.a.read);
        run.read_ns += st.ns;
        run.r.pk.placeKept(g, addr(run.stored[g].?)); // EXL3's experts stay in their stored group
    }

    pub fn all(run: *Run) !void {
        const pk = &run.r.pk;
        const groups: u32 = @intCast(run.r.split.groups().len);
        const addrs = try run.gpa.alloc(u64, groups);
        defer run.gpa.free(addrs);
        var f = pack.hostFwd(&run.r.m);
        if (run.stream) |s| f.gpu = pack.packOnly(s);
        const engine: pack.Engine = if (run.kernels) |k| .{ .gpu = k } else .{ .host = emu };
        for (0..groups) |gi| {
            const g: u32 = @intCast(gi);
            var needed = false; // a group no wanted buffer reads is never read
            for (pk.items.items) |it| if (wanted(run.a, it.path) and it.home >= g and pack.readsGroup(it, g)) {
                needed = true;
            };
            if (needed) try run.load(g);
            var live: u64 = 0;
            for (run.stored) |x| if (x) |y| {
                live += switch (y) {
                    .host => |h| h.len,
                    .dev => |d| d.len,
                };
            };
            run.peak_stored = @max(run.peak_stored, live);
            const out = try run.alloc(pk.homeBytes(g));
            try run.packed_groups.append(run.gpa, out);
            pk.place(g, addr(out));
            for (addrs, run.stored) |*ad, x| ad.* = if (x) |y| addr(y) else 0;
            const t0 = nowNs();
            for (pk.items.items) |*it| if (it.home == g and wanted(run.a, it.path)) try pack.make(&f, engine, it, addrs);
            if (run.stream) |s| try s.synchronize();
            run.pack_ns += nowNs() - t0;
            try run.digest(g);
            for (run.stored, 0..) |*x, s| if (x.* != null and pk.last(@intCast(s)) <= g) {
                run.free(&x.*.?);
                x.* = null;
            };
            if (run.driver == null or !run.a.keep) {
                run.free(&run.packed_groups.items[run.packed_groups.items.len - 1]);
                run.packed_groups.items.len -= 1;
            }
        }
    }

    /// Group `g`'s packed buffers hashed (downloaded first on the GPU).
    fn digest(run: *Run, g: u32) !void {
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(run.gpa);
        for (run.r.pk.items.items, run.rows) |it, *row| if (it.home == g and wanted(run.a, it.path)) {
            const n = it.buf.bytes();
            const bytes: []const u8 = if (run.driver) |d| blk: {
                try tmp.resize(run.gpa, n);
                try d.check(d.api.cuMemcpyDtoH_v2(@ptrCast(tmp.items.ptr), it.buf.ptr, n), "cuMemcpyDtoH");
                break :blk tmp.items;
            } else @as([*]const u8, @ptrFromInt(it.buf.ptr))[0..n];
            Sha256.hash(bytes, &row.sha, .{});
            row.done = true;
        };
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

extern fn tf_glm_pack_emu(kernel: c_int, grid: *const [3]c_uint, block: *const [3]c_uint, params: [*]?*anyopaque, sizes: [*]const u16, count: c_uint) c_int;

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// The plan's table against weights.json's: every path equal (shape, dtype, bytes; sha256 unless --dry), listed if not.
fn compare(gpa: std.mem.Allocator, a: Args, r: *const Rank, rows: []const Row, ref_text: []const u8) !bool {
    const parsed = try std.json.parseFromSlice(J, gpa, ref_text, .{});
    defer parsed.deinit();
    const table = parsed.value.object.get("table").?.object;
    const out = std.debug.print;
    var equal: usize = 0;
    var checked: usize = 0;
    var bad: usize = 0;
    var tables: usize = 0;
    var bytes: u64 = 0;
    for (r.pk.items.items, rows) |it, row| {
        if (!wanted(a, it.path)) continue;
        checked += 1;
        bytes += it.buf.bytes();
        const w = table.get(it.path) orelse {
            out("ONLY-ZIG {s} {t} {any}\n", .{ it.path, it.buf.dtype, it.buf.shape[0..it.buf.dims] });
            bad += 1;
            continue;
        };
        const shape = w.object.get("shape").?.array.items;
        var same = shape.len == it.buf.dims and std.mem.eql(u8, w.object.get("dtype").?.string, pack.dtypeName(it.buf.dtype)) and
            @as(u64, @intCast(w.object.get("bytes").?.integer)) == it.buf.bytes();
        for (shape, 0..) |s, i| same = same and i < it.buf.dims and @as(usize, @intCast(s.integer)) == it.buf.shape[i];
        if (!same) {
            out("SHAPE {s}: zig {t} {any}, python {s} {f}\n", .{ it.path, it.buf.dtype, it.buf.shape[0..it.buf.dims], w.object.get("dtype").?.string, std.json.fmt(w.object.get("shape").?, .{}) });
            bad += 1;
            continue;
        }
        if (a.dry) {
            equal += 1;
            continue;
        }
        const hex = std.fmt.bytesToHex(row.sha, .lower);
        if (w.object.get("sha256").? == .null) { // python's address tables (`*_ptr`): our addresses, never its
            tables += 1;
            continue;
        }
        if (row.done and std.mem.eql(u8, &hex, w.object.get("sha256").?.string)) equal += 1 else {
            out("SHA {s}: zig {s}, python {s}\n", .{ it.path, &hex, w.object.get("sha256").?.string });
            bad += 1;
        }
    }
    var python_only: usize = 0;
    var skipped: usize = 0;
    var it = table.iterator();
    while (it.next()) |e| if (wanted(a, e.key_ptr.*) and r.pk.find(e.key_ptr.*) == null) {
        if (std.mem.startsWith(u8, e.key_ptr.*, "w.draft_head.")) { // qualification_qualification_7c75d777 ANSWER.md pass 6 item 5
            out("SKIP {s} (EXL3's quantize4(head): allocated for parity, filled in qualification_qualification_44ba554e.7)\n", .{e.key_ptr.*});
            skipped += 1;
            continue;
        }
        out("ONLY-PYTHON {s} (not packed)\n", .{e.key_ptr.*});
        python_only += 1;
    };
    var addr_paths: usize = 0;
    var keeps: usize = 0;
    for (r.pk.items.items) |x| if (wanted(a, x.path)) {
        addr_paths += @intFromBool(pack.addressTable(x));
        keeps += @intFromBool(std.mem.indexOf(u8, x.path, ".keep[") != null);
    };
    out("check-weights{s} rank {d}/{d}: {d} of {d} paths equal ({s}), {d} different, {d} python's alone, {d} skipped; {d} B ({d:.2} GiB) packed\n", .{ if (a.dry) " --dry" else "", a.rank.?, a.ranks, equal, checked, if (a.dry) "shape, dtype, bytes" else "sha256", bad, python_only, skipped, bytes, gib(bytes) });
    out("check-weights{s} rank {d}/{d}: python's table {d} paths; ours {d}: {d} address tables (not digested by python: {d} of them skipped here), {d} expert trellises (keep[i]), {d} other\n", .{ if (a.dry) " --dry" else "", a.rank.?, a.ranks, table.count(), checked, addr_paths, tables, keeps, checked - addr_paths - keeps });
    return bad == 0 and python_only == 0;
}

/// The packed table as weights.json lays it out (path -> shape, dtype, sha256, bytes; tensors, bytes, digest).
fn writeTable(gpa: std.mem.Allocator, a: Args, r: *const Rank, rows: []const Row, path: []const u8) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    var h: Sha256 = .init(.{});
    var n: usize = 0;
    var total: u64 = 0;
    try buf.appendSlice(gpa, "{\"table\": {");
    for (r.pk.items.items, rows) |it, row| if (wanted(a, it.path) and (row.done or a.dry)) {
        const hex = std.fmt.bytesToHex(row.sha, .lower);
        try buf.print(gpa, "{s}\n\"{s}\": {{\"shape\": [", .{ if (n == 0) "" else ",", it.path });
        for (it.buf.shape[0..it.buf.dims], 0..) |s, i| try buf.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", s });
        try buf.print(gpa, "], \"dtype\": \"{s}\", \"sha256\": {s}{s}{s}{s}, \"bytes\": {d}}}", .{ pack.dtypeName(it.buf.dtype), if (pack.addressTable(it)) "null" else "\"", if (a.dry or pack.addressTable(it)) "" else &hex, if (pack.addressTable(it)) "" else "\"", if (pack.addressTable(it)) ", \"address_table\": true" else "", it.buf.bytes() });
        h.update(it.path);
        h.update(" ");
        h.update(&hex);
        h.update("\n");
        n += 1;
        total += it.buf.bytes();
    };
    try buf.print(gpa, "}},\n\"tensors\": {d}, \"bytes\": {d}, \"digest\": \"{s}\"}}\n", .{ n, total, &std.fmt.bytesToHex(h.finalResult(), .lower) });
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, buf.items.ptr, buf.items.len) != @as(isize, @intCast(buf.items.len))) return error.CannotWrite;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1 and run_verbs.isVerb(argv[1])) return run_verbs.main(init, argv);
    const a = parse(argv) catch {
        std.debug.print("{s}{s}", .{ usage, run_verbs.usage });
        return 2;
    };
    var ix = try glm.index.open(gpa, a.dir);
    defer ix.deinit();
    var r = try plans(gpa, a, &ix);
    defer r.deinit(gpa);
    const out = std.debug.print;
    out("{s}: rank {d}/{d}, {d} of {d} layers{s}; {d} stored groups ({d:.2} GiB), {d} packed buffers ({d:.2} GiB); {d} checkpoint tensors unread\n", .{ a.dir, a.rank.?, a.ranks, r.c.run, r.c.layers, if (r.mtp()) " and the MTP layer" else "", r.split.groups().len, gib(r.split.bytes()), r.pk.items.items.len, gib(r.pk.bytes()), r.unread });
    const rows = try gpa.alloc(Row, r.pk.items.items.len);
    defer gpa.free(rows);
    @memset(rows, .{});
    {
        // cuda_views' rank (the forward's handles) bound by path to the packed buffers, placed at stand-in addresses
        // (EXL3's experts in their stored groups'): the seam's dtypes and shapes checked
        var base: u64 = 1 << 36;
        for (r.split.groups(), 0..) |bytes, g| {
            r.pk.place(@intCast(g), base);
            base += std.mem.alignForward(u64, r.pk.homeBytes(@intCast(g)), 1 << 20) + (1 << 20);
            if (!r.pk.keeps(@intCast(g))) continue;
            r.pk.placeKept(@intCast(g), base);
            base += std.mem.alignForward(u64, bytes, 1 << 20) + (1 << 20);
        }
        var views = try glm.views.rank(gpa, &r.m);
        defer views.deinit(gpa);
        out("views: {d} of the forward's buffers bound to packed ones by python's path\n", .{try pack.bind(&views, &r.pk)});
    }
    if (a.trace) |p| {
        if (!a.dry) return error.TraceWantsDry;
        out("trace: {d} glm_pack events of {d} packed buffers -> {s}\n", .{ try @import("glm_pack_trace.zig").write(gpa, &r, ix.shards, p), r.pk.items.items.len, p });
    }
    var ok = true;
    if (!a.dry) {
        var run: Run = .{ .gpa = gpa, .a = a, .ix = &ix, .r = &r, .rows = rows, .stored = try gpa.alloc(?Run.Mem, r.split.groups().len) };
        defer gpa.free(run.stored);
        @memset(run.stored, null);
        var driver: cuda.Driver = undefined;
        var ctx: cuda.Context = undefined;
        var kernels: pack.Kernels = undefined;
        var image: []const u8 = if (options.glm_pack) &Blob("glm_pack_fatbin").bytes else &.{};
        if (!a.host) {
            if (a.fatbin) |p| image = try slurp(gpa, p, 64 << 20);
            if (image.len == 0) {
                out("no glm_pack fatbin: build with -Dnvcc or -Dfatbins, or give --fatbin (or --host)\n", .{});
                return 2;
            }
            driver = try cuda.Driver.open();
            ctx = try cuda.Context.init(&driver, 0);
            kernels = try pack.Kernels.load(&driver, image);
            run.driver = &driver;
            run.stream = try cuda.Stream.init(&driver, true);
            run.kernels = &kernels;
        }
        const before = memAvailableGib();
        const t0 = nowNs();
        try run.all();
        const secs = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
        const dev = if (run.driver != null) cuda.usage(false) else cuda.Usage{ .device = 0, .host = 0, .peak = 0 };
        out("check-weights {s} rank {d}/{d}: {d:.1} s (reads {d:.1} s, packing {d:.1} s); stored groups at most {d:.2} GiB at once; device {d:.2} GiB (peak {d:.2}); MemAvailable {d:.2} -> {d:.2} GiB\n", .{ if (a.host) "--host" else "GPU", a.rank.?, a.ranks, secs, @as(f64, @floatFromInt(run.read_ns)) / 1e9, @as(f64, @floatFromInt(run.pack_ns)) / 1e9, gib(run.peak_stored), gib(dev.device), gib(dev.peak), before, memAvailableGib() });
        for (run.packed_groups.items) |*x| run.free(x);
        run.packed_groups.deinit(gpa);
        for (run.stored) |*x| if (x.*) |*y| run.free(y);
        if (run.stream) |*s| s.deinit();
        if (run.kernels) |_| kernels.deinit();
        if (run.driver) |_| {
            ctx.deinit();
            driver.close();
        }
    }
    if (a.out) |p| try writeTable(gpa, a, &r, rows, p);
    if (a.ref) |p| {
        const text = try slurp(gpa, p, 64 << 20);
        defer gpa.free(text);
        ok = try compare(gpa, a, &r, rows, text);
    }
    return if (ok) 0 else 1;
}
