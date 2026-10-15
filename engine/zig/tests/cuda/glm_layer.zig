//! `tf-glm-layer mlp`: G-layer-1 on the MLP halves of the MoE layers, one TP rank on one GPU (qualification_qualification_736508ad; qualification_qualification_7c75d777 PLAN §4).
//! qualification_6607f886's oracle (python-0.6.6's TP2 session on the 8-layer copy, evidence/qualification_6607f886/qualification_qualification_5ab55e74/) holds 48 teacher steps a prompt
//! (decode, one row). Each step has every layer's stream after it (L<l>.x) and its attention's and MLP's gathered fp32
//! outputs (L<l>.kda|dsa, L<l>.mlp|moe: both ranks' partials). Each case runs the Zig layer on the dump's inputs alone:
//! - hc_pre with attn_hc on L<l-1>.x, then hc_post with the dumped attention output (no attention runs: A3's DSA);
//! - python's MLP half: hc_pre with ffn_hc, cuda_moe.partial into b.part, the gather (the peer's partial from the
//!   dump), hc_post.
//! Gate: b.part = L<l>.moe[rank] and b.x = L<l>.x, byte for byte. --dry runs the same on a CPU in recording mode over a
//! header mirror: plans, views bound, buffers, every oracle file's sha256, each case's launch sequence.
//! `tf-glm-layer dsa` (qualification_qualification_973313f6, glm_layer_dsa.zig): G-layer-1 on the DSA layers, the short prompt's chunk and steps.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const images = @import("glm_mm_images");

const pack = glm.pack;
const views = glm.views;
const bufs = glm.buffers;
const fwd = glm.fwd;
const trace = glm.trace;
const Mm = glm.experts.mm.Mm;
const Kda = std.meta.Child(@FieldType(fwd.Kernels, "kda"));
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;
const out = std.debug.print;

const usage =
    \\usage: tf-glm-layer mlp|dsa DIR --oracle RANKDIR --triton SETDIR [options]
    \\  mlp       the MoE layers' MLP halves on the teacher steps (qualification_qualification_736508ad); dsa: the DSA layers whole, the prompt's
    \\            chunk then its steps in order, the caches rebuilt as python wrote them (qualification_qualification_973313f6; the short prompt)
    \\  DIR       the MLX checkpoint (tensorfold-mlx-4bit-mtp-d2b17f52; its header mirror with --dry)
    \\  --oracle  qualification_6607f886's rank directory (qualification_qualification_5ab55e74/run-20261007-231527/rank0): teacher/<prompt>/steps.json and its dumps
    \\  --triton  GLM's Triton set (aot.json + cubins/: ZK4's flow6/set121); not read with --dry
    \\options: --rank R (0) --ranks N (2) --layers N (8: qualification_6607f886's copy) --from L (3) --to L (7) --prompts P[,P] (mlp:
    \\         short,long; dsa: short) --steps N (48) --fatbin FILE (glm_pack.fatbin) --dry (CPU, recording)
    \\         --save DIR (our bytes of a FAIL) --gap BYTES --slot-mib N --slots N --no-ring (the reader's)
    \\
;

pub const Mode = enum { mlp, dsa };

pub const Args = struct {
    mode: Mode = .mlp,
    dir: [:0]const u8 = "",
    oracle: []const u8 = "",
    triton: ?[]const u8 = null,
    rank: u32 = 0,
    ranks: u32 = 2,
    layers: u32 = 8,
    from: usize = 3,
    to: usize = 7,
    prompts: []const u8 = "short,long",
    steps: usize = 48,
    fatbin: ?[]const u8 = null,
    save: ?[]const u8 = null,
    dry: bool = false,
    read: glm.read.Options = .{},
};

pub fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{};
    if (args.len < 3) return error.Usage;
    a.mode = std.meta.stringToEnum(Mode, args[1]) orelse return error.Usage;
    if (a.mode == .dsa) a.prompts = "short"; // the long prompt's chunks keep 32 of 2,048 rows: no caches to rebuild
    a.dir = args[2];
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const k = args[i];
        if (eql(k, "--dry")) a.dry = true else if (eql(k, "--no-ring")) a.read.ring = false else {
            if (i + 1 >= args.len) return error.Usage;
            const v = args[i + 1];
            i += 1;
            const n = std.fmt.parseInt;
            if (eql(k, "--oracle")) a.oracle = v else if (eql(k, "--triton")) a.triton = v else if (eql(k, "--rank")) a.rank = try n(u32, v, 10) else if (eql(k, "--ranks")) a.ranks = try n(u32, v, 10) else if (eql(k, "--layers")) a.layers = try n(u32, v, 10) else if (eql(k, "--from")) a.from = try n(usize, v, 10) else if (eql(k, "--to")) a.to = try n(usize, v, 10) else if (eql(k, "--prompts")) a.prompts = v else if (eql(k, "--steps")) a.steps = try n(usize, v, 10) else if (eql(k, "--fatbin")) a.fatbin = v else if (eql(k, "--save")) a.save = v else if (eql(k, "--gap")) a.read.gap = try n(u64, v, 10) else if (eql(k, "--slot-mib")) a.read.slot_bytes = (try n(usize, v, 10)) << 20 else if (eql(k, "--slots")) a.read.slots = try n(u32, v, 10) else return error.Usage;
        }
    }
    if (a.oracle.len == 0 or a.rank >= a.ranks or a.from < 1 or a.to < a.from or a.to >= a.layers) return error.Usage;
    if (!a.dry and a.triton == null) return error.Usage;
    return a;
}

pub fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn get(o: J, key: []const u8) !J {
    if (o != .object) return error.BadOracle;
    return o.object.get(key) orelse error.BadOracle;
}

/// A file's bytes, at most `most`.
pub fn slurp(a: std.mem.Allocator, path: []const u8, most: usize) ![]u8 {
    const z = try a.dupeSentinel(u8, path, 0);
    defer a.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) {
        out("cannot open {s}\n", .{path});
        return error.FileNotFound;
    }
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or end > most) return error.FileTooBig;
    const buf = try a.alloc(u8, @intCast(end));
    var done: usize = 0;
    while (done < buf.len) {
        const n = std.c.pread(fd, buf.ptr + done, buf.len - done, @intCast(done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
    return buf;
}

pub fn spit(path: []const u8, bytes: []const u8) !void {
    var z: [4096]u8 = undefined;
    if (path.len >= z.len) return error.NameTooLong;
    @memcpy(z[0..path.len], path);
    z[path.len] = 0;
    const fd = std.c.open(z[0..path.len :0], .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.CannotWrite;
    defer _ = std.c.close(fd);
    if (std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.CannotWrite;
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
        const kib = std.fmt.parseInt(u64, f.next() orelse return 0, 10) catch return 0;
        return @as(f64, @floatFromInt(kib)) / (1024 * 1024);
    };
    return 0;
}

pub fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

// ---- the oracle: qualification_6607f886's teacher steps -------------------------------------------------------------------------

/// One dump: its file (from the rank directory), the full tensor's sha256, and its bytes.
pub const Dump = struct { file: []const u8, sha: []const u8, bytes: usize };

/// One case: a prompt's step and a layer, with the four dumps it reads.
const Case = struct { prompt: []const u8, k: usize, layer: usize, before: Dump, attn: Dump, mlp: Dump, after: Dump };

pub const max_layers = 64;

/// steps.json's entry as a dump; the saved rows must be the whole tensor (a teacher step's one row is).
pub fn dumpOf(e: J) !Dump {
    const saved = try get(e, "saved");
    const shape = (try get(e, "shape")).array.items;
    const kept = (try get(saved, "shape")).array.items;
    if (shape.len != kept.len) return error.PartialDump;
    const dt = (try get(e, "dtype")).string;
    var n: usize = if (eql(dt, "float32")) 4 else if (eql(dt, "bfloat16")) 2 else return error.BadOracle;
    for (shape, kept) |s, k| {
        if (s.integer != k.integer) return error.PartialDump;
        n *= @intCast(s.integer);
    }
    return .{ .file = (try get(saved, "file")).string, .sha = (try get(e, "sha256")).string, .bytes = n };
}

/// Every case of the prompts' first `steps` steps for layers [from, to], each file's sha256 checked against steps.json.
fn cases(gpa: std.mem.Allocator, arena: std.mem.Allocator, a: Args) ![]Case {
    var list: std.ArrayList(Case) = .empty;
    var files: usize = 0;
    var bytes: usize = 0;
    var it = std.mem.tokenizeScalar(u8, a.prompts, ',');
    while (it.next()) |prompt| {
        const path = try std.fmt.allocPrint(arena, "{s}/teacher/{s}/steps.json", .{ a.oracle, prompt });
        const text = try slurp(arena, path, 64 << 20);
        const parsed = try std.json.parseFromSliceLeaky(J, arena, text, .{});
        const steps = (try get(parsed, "steps")).array.items;
        if (steps.len < a.steps) return error.TooFewSteps;
        for (steps[0..a.steps]) |s| {
            var x: [max_layers]?Dump = @splat(null);
            var attn: [max_layers]?Dump = @splat(null);
            var mlp: [max_layers]?Dump = @splat(null);
            for ((try get(s, "layers")).array.items) |e| {
                const name = (try get(e, "name")).string; // L03.dsa, L03.moe, L03.x; also hidden, fnormed
                if (name.len < 5 or name[0] != 'L' or name[3] != '.' or !std.ascii.isDigit(name[1]) or !std.ascii.isDigit(name[2])) continue;
                const l = try std.fmt.parseInt(usize, name[1..3], 10);
                if (l >= max_layers) return error.BadOracle;
                const kind = name[4..];
                const d = try dumpOf(e);
                if (eql(kind, "x")) x[l] = d else if (eql(kind, "kda") or eql(kind, "dsa")) attn[l] = d else if (eql(kind, "mlp") or eql(kind, "moe")) mlp[l] = d else return error.BadOracle;
            }
            const k: usize = @intCast((try get(s, "k")).integer);
            for (a.from..a.to + 1) |l| try list.append(gpa, .{
                .prompt = prompt,
                .k = k,
                .layer = l,
                .before = x[l - 1] orelse return error.MissingDump,
                .attn = attn[l] orelse return error.MissingDump,
                .mlp = mlp[l] orelse return error.MissingDump,
                .after = x[l] orelse return error.MissingDump,
            });
        }
    }
    // each file once: its bytes and sha256 against steps.json's
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    for (list.items) |c| for ([_]Dump{ c.before, c.attn, c.mlp, c.after }) |d| {
        if ((try seen.getOrPut(gpa, d.file)).found_existing) continue;
        const b = try readDump(arena, a, d);
        var sha: [32]u8 = undefined;
        Sha256.hash(b, &sha, .{});
        const hex = std.fmt.bytesToHex(sha, .lower);
        if (!eql(&hex, d.sha)) {
            out("ORACLE {s}: sha256 {s}, steps.json {s}\n", .{ d.file, &hex, d.sha });
            return error.OracleDigest;
        }
        files += 1;
        bytes += b.len;
        arena.free(b);
    };
    out("oracle {s}: {d} cases ({s}; steps 0-{d}; layers {d}-{d}); {d} files, {d} B, every sha256 equal to steps.json\n", .{ a.oracle, list.items.len, a.prompts, a.steps - 1, a.from, a.to, files, bytes });
    return list.toOwnedSlice(gpa);
}

pub fn readDump(arena: std.mem.Allocator, a: Args, d: Dump) ![]u8 {
    const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ a.oracle, d.file });
    defer arena.free(path);
    const b = try slurp(arena, path, d.bytes);
    if (b.len != d.bytes) return error.DumpSize;
    return b;
}

// ---- the rank: plans, pack pass, views, buffers ---------------------------------------------------------------

/// The rank's plans (tf-glm-forward check-weights' `plans`): the split and the pack for the model cut to `layers`.
pub const Rank = struct {
    m: glm.model.Model,
    c: glm.Config,
    split: glm.split.Plan,
    pk: pack.Plan,
};

pub fn plans(gpa: std.mem.Allocator, a: Args, ix: *glm.index.Index) !Rank {
    const path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{a.dir});
    defer gpa.free(path);
    const text = try slurp(gpa, path, 1 << 20);
    defer gpa.free(text);
    var base = try glm.config.parse(gpa, text);
    try glm.config.subset(&base, a.layers); // qualification_6607f886's 8-layer copy: layers 0..n-1 and the checkpoint's MTP layer
    var m = try glm.model.Model.from(&base, try glm.model.Extra.parse(gpa, text), a.ranks, a.rank);
    const c = try glm.split.rankConfig(base, a.rank, a.ranks);
    var sp = try glm.split.plan(gpa, &c, &ix.names, .python);
    errdefer sp.deinit();
    if (sp.mtp == null) m.mtp_layers = 0; // python's Weights.mtp only when the split loaded the MTP layer
    sp.used.clearAndFree(gpa);
    var r: Rank = .{ .m = m, .c = c, .split = sp, .pk = undefined };
    r.pk = try pack.plan(gpa, &r.m, &r.c, &r.split);
    return r;
}

/// The pack pass on the GPU (check-weights' Run with --keep 1): each stored group read, the packed buffers whose home
/// it is made, stored groups freed after their last reader; the packed groups stay on the device for the views.
pub const Packer = struct {
    gpa: std.mem.Allocator,
    a: Args,
    ix: *glm.index.Index,
    r: *Rank,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    kernels: *const pack.Kernels,
    stored: []?cuda.DeviceBuffer,
    groups: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn alloc(p: *Packer, bytes: u64) !cuda.DeviceBuffer {
        var b = try cuda.DeviceBuffer.alloc(p.d, @intCast(@max(bytes, 256)));
        // the padding between tensors reads as zeros, as check-weights leaves it; on the work stream, so the uploads
        // and pack kernels after it are ordered behind it (a non-blocking stream does not wait for the legacy one)
        try b.fill8(0, p.stream.handle);
        return b;
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const p: *Packer = @ptrCast(@alignCast(ctx));
        try p.stored[g].?.uploadAsync(@intCast(dst), bytes, p.stream.handle);
    }

    fn flush(ctx: *anyopaque) anyerror!void {
        const p: *Packer = @ptrCast(@alignCast(ctx));
        try p.stream.synchronize();
    }

    fn load(p: *Packer, g: u32) !void {
        p.stored[g] = try p.alloc(p.r.split.groups.items[g]);
        var mine: std.ArrayList(glm.split.Copy) = .empty;
        defer mine.deinit(p.gpa);
        for (p.r.split.copies.items) |cp| if (cp.g == g) try mine.append(p.gpa, cp);
        var s = try glm.read.schedule(p.gpa, mine.items, p.a.read);
        defer s.deinit();
        _ = try glm.read.run(p.gpa, p.ix.shards, &s, .{ .ctx = p, .put = put, .flush = flush }, p.a.read);
    }

    pub fn all(p: *Packer) !void {
        const pk = &p.r.pk;
        const n: u32 = @intCast(p.r.split.groups.items.len);
        const addrs = try p.gpa.alloc(u64, n);
        defer p.gpa.free(addrs);
        var f = pack.hostFwd(&p.r.m);
        f.gpu = pack.packOnly(p.stream);
        for (0..n) |gi| {
            const g: u32 = @intCast(gi);
            var needed = false;
            for (pk.items.items) |it| if (it.home >= g and readsGroup(it, g)) {
                needed = true;
            };
            if (needed) try p.load(g);
            const o = try p.alloc(pk.homeBytes(g));
            try p.groups.append(p.gpa, o);
            pk.place(g, o.ptr);
            for (addrs, p.stored) |*ad, x| ad.* = if (x) |y| y.ptr else 0;
            for (pk.items.items) |*it| if (it.home == g) try pack.make(&f, .{ .gpu = p.kernels }, it, addrs);
            try p.stream.synchronize();
            for (p.stored, 0..) |*x, s| if (x.* != null and pk.last(@intCast(s)) <= g) {
                x.*.?.free();
                x.* = null;
            };
        }
    }
};

fn readsGroup(it: pack.Item, g: u32) bool {
    return switch (it.op) {
        .copy => |r| r.g == g,
        inline .words, .major, .rows => |x| x.src.g == g,
        .experts => |x| for (x.routed[0..x.mm], x.shared[0..x.mm]) |r, s| {
            if (r.w.g == g or s.w.g == g) break true;
        } else false,
    };
}

// ---- one case ---------------------------------------------------------------------------------------------------

/// The GPU side of a case's data movement (null on --dry: nothing moves).
pub const Dev = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,

    pub fn put(v: Dev, dst: u64, bytes: []const u8) !void {
        try v.d.check(v.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, v.stream.handle), "cuMemcpyHtoDAsync");
    }

    pub fn get(v: Dev, src: u64, buf: []u8) !void {
        try v.stream.synchronize();
        try v.d.check(v.d.api.cuMemcpyDtoHAsync_v2(buf.ptr, src, buf.len, v.stream.handle), "cuMemcpyDtoHAsync");
        try v.stream.synchronize();
    }

    pub fn copy(v: Dev, dst: u64, src: u64, n: usize) !void {
        try v.d.check(v.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, v.stream.handle), "cuMemcpyDtoDAsync");
    }
};

/// What --save keeps of a FAIL beside b.part and b.x: the stream after the attention's hc_post (b.x before the MLP
/// half) and the MoE's intermediates of the row (b.normed, b.mlog, b.pick, b.wts, b.ey), for a CPU card's bisect.
const Probe = struct {
    mid: []u8,
    normed: []u8,
    mlog: []u8,
    pick: []u8,
    wts: []u8,
    ey: []u8,

    const names = [_][]const u8{ "mid", "normed", "mlog", "pick", "wts", "ey" };

    fn init(gpa: std.mem.Allocator, b: *const bufs.Buffers) !Probe {
        var p: Probe = undefined;
        inline for (names, [_]views.Buf{ b.x, b.normed, b.mlog, b.pick, b.wts, b.ey }) |n, buf| @field(p, n) = try gpa.alloc(u8, buf.bytes() / b.rows);
        return p;
    }

    fn deinit(p: *Probe, gpa: std.mem.Allocator) void {
        inline for (names) |n| gpa.free(@field(p, n));
    }

    fn save(p: *const Probe, arena: std.mem.Allocator, dir: []const u8, c: Case) !void {
        inline for (names) |n| try spit(try std.fmt.allocPrint(arena, "{s}/{s}-s{d:0>2}-L{d:0>2}.{s}.bin", .{ dir, c.prompt, c.k, c.layer, n }), @field(p, n));
    }
};

/// forward.layer_forward with the attention's gathered output given (`attn`), then its MLP half as python runs it,
/// with the gather's peer partial given (`mlp`'s other ranks): rank `a.rank` of `m.world`, one row.
fn mlpHalf(f: *fwd.Fwd, dev: ?Dev, probe: ?*Probe, l: *const views.Layer, b: *const bufs.Buffers, a: Args, before: []const u8, attn: []const u8, mlp: []const u8) !void {
    const m = f.m;
    const R = 1;
    const D = m.hidden;
    const ha = l.attn_hc orelse return error.PlainLayer;
    const hf = l.ffn_hc orelse return error.PlainLayer;
    const x = l.moe orelse return error.NotMoe;
    if (dev) |v| try v.put(b.x.ptr, before);
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = ha.@"fn".ptr, .base = ha.base.ptr, .scale = ha.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, D, m.streams);
    if (dev) |v| try v.put(b.gath.ptr, attn); // [world, R, D] fp32: gather's layout
    try f.tri.hcPost(b.x.ptr, b.x.ptr, b.gath.ptr, b.post.ptr, b.comb.ptr, m.world, R, D, m.streams);
    if (dev) |v| if (probe) |p| try v.get(b.x.ptr, p.mid);
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = hf.@"fn".ptr, .base = hf.base.ptr, .scale = hf.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, D, m.streams);
    try glm.moe.partial(f, x, b, R, .{});
    if (dev) |v| if (probe) |p| inline for (.{ "normed", "mlog", "pick", "wts", "ey" }) |n| try v.get(@field(b, n).ptr, @field(p, n));
    if (dev) |v| { // the gather: ours at rank, every other rank's partial from the dump
        for (0..m.world) |r| {
            const at = b.gath.ptr + r * R * D * 4;
            if (r == a.rank) try v.copy(at, b.part.ptr, R * D * 4) else try v.put(at, mlp[r * R * D * 4 ..][0 .. R * D * 4]);
        }
    }
    try f.tri.hcPost(b.x.ptr, b.x.ptr, b.gath.ptr, b.post.ptr, b.comb.ptr, m.world, R, D, m.streams);
}

/// Our bytes against the dump's: equal, or the first differing element and how many differ.
fn same(what: []const u8, c: Case, ours: []const u8, want: []const u8, elem: usize) bool {
    if (std.mem.eql(u8, ours, want)) return true;
    var first: ?usize = null;
    var n: usize = 0;
    var i: usize = 0;
    while (i < ours.len) : (i += elem) if (!std.mem.eql(u8, ours[i..][0..elem], want[i..][0..elem])) {
        if (first == null) first = i / elem;
        n += 1;
    };
    const at = first.? * elem;
    const v = struct {
        fn of(b: []const u8, e: usize) f32 {
            if (e == 4) return @bitCast(std.mem.readInt(u32, b[0..4], .little));
            return @bitCast(@as(u32, std.mem.readInt(u16, b[0..2], .little)) << 16);
        }
    }.of;
    out("FAIL {s} s{d:0>2} L{d:0>2} {s}: {d} of {d} elements differ; first [{d}] zig {e} python {e}\n", .{ c.prompt, c.k, c.layer, what, n, ours.len / elem, first.?, v(ours[at..], elem), v(want[at..], elem) });
    return false;
}

// ---- main -------------------------------------------------------------------------------------------------------

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const a = parse(argv) catch {
        out("{s}", .{usage});
        return 2;
    };
    if (a.mode == .dsa) return @import("glm_layer_dsa.zig").run(init, gpa, arena, a);
    const all = try cases(gpa, arena, a);
    defer gpa.free(all);
    var ix = try glm.index.open(gpa, a.dir);
    defer ix.deinit();
    var r = try plans(gpa, a, &ix);
    defer r.split.deinit();
    defer r.pk.deinit();
    out("{s}: rank {d}/{d}, {d} layers{s}; {d} stored groups ({d:.2} GiB), {d} packed buffers ({d:.2} GiB)\n", .{ a.dir, a.rank, a.ranks, r.c.run, if (r.split.mtp != null) " and the MTP layer" else "", r.split.groups.items.len, gib(r.split.bytes()), r.pk.items.items.len, gib(r.pk.bytes()) });
    if (r.m.quant != .mlx) {
        out("tf-glm-layer: the MLX checkpoint only (EXL3's rank load is not in the pack pass yet)\n", .{});
        return 2;
    }
    for (a.from..a.to + 1) |l| if (r.m.mlp_kinds[l] != .moe) {
        out("layer {d} is not a MoE layer (G-layer-1's MLP halves here are the MoE's)\n", .{l});
        return 2;
    };

    var driver: cuda.Driver = undefined;
    var ctx: cuda.Context = undefined;
    var stream: cuda.Stream = undefined;
    var pk_kernels: pack.Kernels = undefined;
    var packer: ?Packer = null;
    var set: cuda.aot.Set = undefined;
    var mmk: Mm = undefined;
    var no_kda: Kda = undefined; // the MLP half launches no KDA kernel
    const before = memAvailableGib();
    if (a.dry) { // stand-in addresses, as check-weights --dry places them
        var base: u64 = 1 << 36;
        for (0..r.split.groups.items.len) |g| {
            r.pk.place(@intCast(g), base);
            base += std.mem.alignForward(u64, r.pk.homeBytes(@intCast(g)), 1 << 20) + (1 << 20);
        }
    } else {
        const image = slurp(gpa, a.fatbin orelse {
            out("tf-glm-layer: give --fatbin glm_pack.fatbin (the pack pass's kernels)\n", .{});
            return 2;
        }, 64 << 20) catch |e| {
            out("tf-glm-layer: cannot read the glm_pack fatbin: {t}\n", .{e});
            return 2;
        };
        defer gpa.free(image);
        driver = try cuda.Driver.open();
        ctx = try cuda.Context.init(&driver, 0);
        stream = try cuda.Stream.init(&driver, true);
        pk_kernels = try pack.Kernels.load(&driver, image);
        const t0 = nowNs();
        packer = .{ .gpa = gpa, .a = a, .ix = &ix, .r = &r, .d = &driver, .stream = stream, .kernels = &pk_kernels, .stored = try gpa.alloc(?cuda.DeviceBuffer, r.split.groups.items.len) };
        @memset(packer.?.stored, null);
        try packer.?.all();
        out("packed rank {d} on the GPU in {d:.1} s; MemAvailable {d:.2} -> {d:.2} GiB\n", .{ a.rank, @as(f64, @floatFromInt(nowNs() - t0)) / 1e9, before, memAvailableGib() });
        set = try cuda.aot.Set.load(gpa, init.io, &driver, ctx.device, a.triton.?);
        mmk = try Mm.load(&ctx, .{ .qmm_group = images.glm_qmm_group, .qmm_prefill = images.glm_qmm_prefill, .experts = images.glm_experts, .experts_prefill = images.glm_experts_prefill });
        out("kernels: the Triton set {s}, GLM's four affine images; chip {d}.{d}, {d} SMs; experts resident a SM {d}, {d}\n", .{ a.triton.?, mmk.chip.major, mmk.chip.minor, mmk.chip.sms, mmk.per_sm[0], mmk.per_sm[1] });
    }
    defer if (packer) |*p| {
        for (p.groups.items) |*x| x.free();
        p.groups.deinit(gpa);
        for (p.stored) |*x| if (x.*) |*y| y.free();
        gpa.free(p.stored);
        mmk.unload();
        set.deinit();
        pk_kernels.deinit();
        stream.deinit();
        ctx.deinit();
        driver.close();
    };
    var w = try views.rank(gpa, &r.m);
    defer w.deinit(gpa);
    out("views: {d} of the forward's buffers bound to packed ones by python's path\n", .{try pack.bind(&w, &r.pk)});

    // python's decode Buffers (8 rows, capacity 2,560), one device arena
    var b = try bufs.Buffers.init(&r.m, 8, 2560, false);
    var sizing: bufs.Arena = .{};
    try views.walk(&b, views.Path.init("b"), &sizing, bufs.Arena.visit);
    var arena_dev: ?cuda.DeviceBuffer = null;
    defer if (arena_dev) |*x| x.free();
    var place: bufs.Arena = .{ .base = 1 << 44 };
    if (!a.dry) {
        arena_dev = try cuda.DeviceBuffer.alloc(&driver, sizing.used);
        try arena_dev.?.fill8(0, stream.handle); // on the work stream: the kernels after it are ordered behind it
        place.base = arena_dev.?.ptr;
    }
    b = try bufs.Buffers.init(&r.m, 8, 2560, false);
    try views.walk(&b, views.Path.init("b"), &place, bufs.Arena.visit);
    out("buffers: {d} B in one arena\n", .{sizing.used});

    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = if (a.dry) fwd.Fwd.recording(&r.m, &t) else fwd.Fwd.launching(&r.m, .{ .stream = stream, .set = &set, .kernels = .{ .kda = &no_kda, .mm = &mmk } });
    const dev: ?Dev = if (a.dry) null else .{ .d = &driver, .stream = stream };
    const D = r.m.hidden;
    const xs = try gpa.alloc(u8, r.m.streams * D * 2);
    defer gpa.free(xs);
    const part = try gpa.alloc(u8, D * 4);
    defer gpa.free(part);
    var probe = try Probe.init(gpa, &b);
    defer probe.deinit(gpa);
    var equal: usize = 0;
    var per: [max_layers][2]usize = @splat(.{ 0, 0 });
    var events: ?usize = null;
    const t0 = nowNs();
    for (all) |c| {
        const in_x = try readDump(arena, a, c.before);
        const in_attn = try readDump(arena, a, c.attn);
        const want_mlp = try readDump(arena, a, c.mlp);
        const want_x = try readDump(arena, a, c.after);
        t.reset();
        mlpHalf(&f, dev, &probe, &w.layers[c.layer], &b, a, in_x, in_attn, want_mlp) catch |e| {
            out("FAIL {s} s{d:0>2} L{d:0>2}: {t}\n", .{ c.prompt, c.k, c.layer, e });
            return 1;
        };
        per[c.layer][1] += 1;
        if (dev) |v| {
            try v.get(b.part.ptr, part);
            try v.get(b.x.ptr, xs);
            const ok_part = same("b.part", c, part, want_mlp[a.rank * D * 4 ..][0 .. D * 4], 4);
            const ok_x = same("b.x", c, xs, want_x, 2);
            if (ok_part and ok_x) {
                equal += 1;
                per[c.layer][0] += 1;
            } else if (a.save) |dir| {
                try spit(try std.fmt.allocPrint(arena, "{s}/{s}-s{d:0>2}-L{d:0>2}.part.bin", .{ dir, c.prompt, c.k, c.layer }), part);
                try spit(try std.fmt.allocPrint(arena, "{s}/{s}-s{d:0>2}-L{d:0>2}.x.bin", .{ dir, c.prompt, c.k, c.layer }), xs);
                try probe.save(arena, dir, c);
            }
        } else {
            try f.flush();
            const n = t.events.items.len;
            if (events) |e| {
                if (e != n) return error.LaunchesDiffer;
            } else {
                events = n;
                out("dry: each case's launches ({d}):", .{n});
                for (t.events.items) |e| out(" {t}:{s}", .{ e.op, e.name });
                out("\n", .{});
            }
            equal += 1;
            per[c.layer][0] += 1;
        }
        for ([_][]u8{ in_x, in_attn, want_mlp, want_x }) |x| arena.free(x);
    }
    const secs = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    for (a.from..a.to + 1) |l| out("L{d:0>2}: {d} of {d} cases {s}\n", .{ l, per[l][0], per[l][1], if (a.dry) "recorded" else "equal" });
    if (a.dry) {
        out("G-layer-1 --dry rank {d}/{d}: {d} cases recorded, the same {d} launches each, in {d:.1} s\n", .{ a.rank, a.ranks, all.len, events orelse 0, secs });
        return 0;
    }
    out("G-layer-1 MLP halves rank {d}/{d}: {d} of {d} cases equal (b.part and b.x, byte for byte) in {d:.1} s; MemAvailable {d:.2} GiB\n", .{ a.rank, a.ranks, equal, all.len, secs, memAvailableGib() });
    return if (equal == all.len) 0 else 1;
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "usage: an oracle, MoE layers past the first, and the Triton set on a GPU" {
    _ = &main; // the whole tool compiles on any host; it runs on a GPU (or --dry on a CPU)
    try std.testing.expectError(error.Usage, parse(&.{ "tf-glm-layer", "mlp", "/ckpt" }));
    try std.testing.expectError(error.Usage, parse(&.{ "tf-glm-layer", "mlp", "/ckpt", "--oracle", "/o" }));
    try std.testing.expectError(error.Usage, parse(&.{ "tf-glm-layer", "mlp", "/ckpt", "--oracle", "/o", "--dry", "--from", "0" }));
    try std.testing.expectError(error.Usage, parse(&.{ "tf-glm-layer", "kda", "/ckpt", "--oracle", "/o", "--dry" }));
    const a = try parse(&.{ "tf-glm-layer", "mlp", "/ckpt", "--oracle", "/o", "--triton", "/set121", "--to", "5" });
    try std.testing.expectEqual(@as(usize, 3), a.from);
    try std.testing.expectEqual(@as(usize, 5), a.to);
    try std.testing.expect(!a.dry and a.rank == 0 and a.ranks == 2 and a.layers == 8);
    _ = @import("glm_layer_dsa.zig");
}
