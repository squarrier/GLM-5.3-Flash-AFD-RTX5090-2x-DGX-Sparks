//! `tf-glm-layer dsa`: G-layer-1 on the DSA layers (3 and 7 of qualification_6607f886's 8-layer copy), one TP rank on one GPU (qualification_qualification_973313f6; qualification_qualification_7c75d777
//! PLAN §4). qualification_6607f886's oracle keeps every row of the short prompt's one chunk (64 rows) and of its 48 teacher steps (one
//! row each), so a DSA layer's caches can be rebuilt in python's order, a chunk then each step at its position:
//! - hc_pre with attn_hc on L<l-1>.x's rows, A3's dsa.partial (the latent write, the indexer update, absorb, the dense
//!   chunks, expand, out_proj) on python's Buffers (prompt chunks: 2,048 rows; decode: 8), the gather (ours, the peer's
//!   dumped half), hc_post; then the MLP half (hc_pre with ffn_hc, A2's moe.partial, the gather, hc_post);
//! - the State is python's at qualification_6607f886's --context: capacity 6,144 with the indexer's caches (written, read past the dense
//!   limit only), one per DSA layer, zeroed at the start and only ever written by the cases.
//! Gate: the attention partial = L<l>.dsa[rank], the MoE partial = L<l>.moe[rank], b.x = L<l>.x, byte for byte. qualification_6607f886's
//! long prompt keeps 16 of each 2,048-row chunk, so its caches cannot be rebuilt from qualification_6607f886; qualification_qualification_ea348a94's capture kit keeps
//! L02.x, L06.x and layers 3 and 7 whole on it, and then its chunks (2,048 rows at 0 and 2,048, 904 at 4,096: A4's
//! dense and select blocks, a dk.Scratch for the sparse partials) and its 48 sparse steps run here too. Dumps kept in
//! part are left out; a case that needs one stops, said. The dumps hold no MTP rows.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const images = @import("glm_mm_images");
const gl = @import("glm_layer.zig");

const pack = glm.pack;
const views = glm.views;
const bufs = glm.buffers;
const fwd = glm.fwd;
const trace = glm.trace;
const Mm = glm.experts.mm.Mm;
const Kda = std.meta.Child(@FieldType(fwd.Kernels, "kda"));
const Sel = glm.dsa_kernels.Kernels;
const Sha256 = std.crypto.hash.sha2.Sha256;
const J = std.json.Value;
const out = std.debug.print;

/// python's Engine at qualification_6607f886's settings: prompt chunks of 2,048 rows, decode windows of 8, --context 6,144
const chunk_rows = 2048;
const window_rows = 8;
const capacity = 6144;

/// One case: a prompt's chunk (`step` null) or teacher step at one DSA layer, its position and rows, its four dumps.
const Case = struct { prompt: []const u8, step: ?usize, pos: usize, rows: usize, layer: usize, before: gl.Dump, attn: gl.Dump, mlp: gl.Dump, after: gl.Dump };

/// A chunk's or a step's dumps by layer.
const Rows = struct { pos: usize, rows: usize, k: ?usize = null, x: [gl.max_layers]?gl.Dump = @splat(null), attn: [gl.max_layers]?gl.Dump = @splat(null), mlp: [gl.max_layers]?gl.Dump = @splat(null) };

/// A dump into its layer's slot; false for a dump kept in part (qualification_6607f886's long prompt), left out.
fn addEntry(r: *Rows, e: J) !bool {
    const name = (try gl.get(e, "name")).string; // L03.dsa, L03.moe, L03.x; also hidden, fnormed
    if (name.len < 5 or name[0] != 'L' or name[3] != '.' or !std.ascii.isDigit(name[1]) or !std.ascii.isDigit(name[2])) return true;
    const l = try std.fmt.parseInt(usize, name[1..3], 10);
    if (l >= gl.max_layers) return error.BadOracle;
    const d = gl.dumpOf(e) catch |err| return if (err == error.PartialDump) false else err;
    const kind = name[4..];
    if (gl.eql(kind, "x")) r.x[l] = d else if (gl.eql(kind, "kda") or gl.eql(kind, "dsa")) r.attn[l] = d else if (gl.eql(kind, "mlp") or gl.eql(kind, "moe")) r.mlp[l] = d else return error.BadOracle;
    return true;
}

/// A case's dump, or a stop that names it (absent, or kept in part: the long prompt wants qualification_qualification_ea348a94's capture).
fn need(d: ?gl.Dump, prompt: []const u8, pos: usize, l: usize, what: []const u8) !gl.Dump {
    if (d) |x| return x;
    out("ORACLE {s} from {d}: L{d:0>2}.{s} absent or kept in part (the long prompt: qualification_qualification_ea348a94's capture kit keeps L02.x, L06.x and layers 3 and 7 whole)\n", .{ prompt, pos, l, what });
    return error.MissingDump;
}

fn json(arena: std.mem.Allocator, a: gl.Args, comptime fmt: []const u8, prompt: []const u8) !J {
    return std.json.parseFromSliceLeaky(J, arena, try gl.slurp(arena, try std.fmt.allocPrint(arena, fmt, .{ a.oracle, prompt }), 64 << 20), .{});
}

/// The cases of each prompt in python's order (its chunks, then its first `steps` teacher steps; every position once,
/// none skipped) at the DSA layers of [from, to], each file's sha256 checked against the oracle's json.
fn cases(gpa: std.mem.Allocator, arena: std.mem.Allocator, a: gl.Args, m: *const glm.model.Model) ![]Case {
    var list: std.ArrayList(Case) = .empty;
    var it = std.mem.tokenizeScalar(u8, a.prompts, ',');
    while (it.next()) |prompt| {
        var rows: std.ArrayList(Rows) = .empty;
        var partial: usize = 0;
        for ((try gl.get(try json(arena, a, "{s}/prefill/{s}/layers.json", prompt), "entries")).array.items) |e| {
            const pos: usize = @intCast((try gl.get(e, "pos0")).integer);
            const n: usize = @intCast((try gl.get(e, "rows")).integer);
            if (rows.items.len == 0 or rows.items[rows.items.len - 1].pos != pos) try rows.append(arena, .{ .pos = pos, .rows = n });
            if (rows.items[rows.items.len - 1].rows != n) return error.BadOracle;
            if (!try addEntry(&rows.items[rows.items.len - 1], e)) partial += 1;
        }
        const steps = (try gl.get(try json(arena, a, "{s}/teacher/{s}/steps.json", prompt), "steps")).array.items;
        if (steps.len < a.steps) return error.TooFewSteps;
        for (steps[0..a.steps]) |s| {
            try rows.append(arena, .{ .pos = @intCast((try gl.get(s, "position")).integer), .rows = 1, .k = @intCast((try gl.get(s, "k")).integer) });
            for ((try gl.get(s, "layers")).array.items) |e| if (!try addEntry(&rows.items[rows.items.len - 1], e)) {
                partial += 1;
            };
        }
        if (partial > 0) out("oracle {s}: {d} dumps kept in part, left out\n", .{ prompt, partial });
        var next: usize = 0;
        for (rows.items) |r| {
            if (r.pos != next) {
                out("ORACLE {s}: rows from {d} follow {d}: a gap the caches cannot cross\n", .{ prompt, r.pos, next });
                return error.OracleGap;
            }
            next = r.pos + r.rows;
            for (a.from..a.to + 1) |l| if (m.kinds[l] == .dsa) try list.append(gpa, .{
                .prompt = prompt,
                .step = r.k,
                .pos = r.pos,
                .rows = r.rows,
                .layer = l,
                .before = try need(r.x[l - 1], prompt, r.pos, l - 1, "x"),
                .attn = try need(r.attn[l], prompt, r.pos, l, "dsa"),
                .mlp = try need(r.mlp[l], prompt, r.pos, l, "moe"),
                .after = try need(r.x[l], prompt, r.pos, l, "x"),
            });
        }
    }
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var files: usize = 0;
    var bytes: usize = 0;
    for (list.items) |c| for ([_]gl.Dump{ c.before, c.attn, c.mlp, c.after }) |d| {
        if ((try seen.getOrPut(gpa, d.file)).found_existing) continue;
        const b = try gl.readDump(arena, a, d);
        var sha: [32]u8 = undefined;
        Sha256.hash(b, &sha, .{});
        if (!gl.eql(&std.fmt.bytesToHex(sha, .lower), d.sha)) {
            out("ORACLE {s}: sha256 differs from the json's {s}\n", .{ d.file, d.sha });
            return error.OracleDigest;
        }
        files += 1;
        bytes += b.len;
        arena.free(b);
    };
    out("oracle {s}: {d} cases ({s}: chunks, then steps 0-{d}; DSA layers of {d}-{d}); {d} files, {d} B, every sha256 equal to the json's\n", .{ a.oracle, list.items.len, a.prompts, a.steps - 1, a.from, a.to, files, bytes });
    return list.toOwnedSlice(gpa);
}

/// `x`'s buffers in one zeroed device block (null and stand-in addresses from `base` with --dry).
fn placeAll(x: anytype, root: []const u8, d: ?*const cuda.Driver, stream: cuda.Stream, base: u64) !?cuda.DeviceBuffer {
    var sizing: bufs.Arena = .{};
    try views.walk(x, views.Path.init(root), &sizing, bufs.Arena.visit);
    var place: bufs.Arena = .{ .base = base };
    var blk: ?cuda.DeviceBuffer = null;
    if (d) |drv| {
        blk = try cuda.DeviceBuffer.alloc(drv, sizing.used);
        try blk.?.fill8(0, stream.handle); // on the work stream: the kernels after it are ordered behind it
        place.base = blk.?.ptr;
    }
    try views.walk(x, views.Path.init(root), &place, bufs.Arena.visit);
    out("{s}: {d} B in one block\n", .{ root, sizing.used });
    return blk;
}

/// forward.gather on one rank: ours (b.part) at our rank, every other rank's partial from the dump ([world, R, D] fp32).
fn gather(dev: ?gl.Dev, b: *const bufs.Buffers, a: gl.Args, R: usize, D: usize, dump: []const u8, world: usize) !void {
    const v = dev orelse return;
    for (0..world) |r| {
        const at = b.gath.ptr + r * R * D * 4;
        if (r == a.rank) try v.copy(at, b.part.ptr, R * D * 4) else try v.put(at, dump[r * R * D * 4 ..][0 .. R * D * 4]);
    }
}

/// Our attention and MoE partials and stream of a case (on a GPU).
const Got = struct { attn: []u8, mlp: []u8, x: []u8 };

/// Layer l of a case as python's layer_forward runs it on one rank, at the case's position (st.set_pos first); `x`
/// holds a 2,048-row chunk's sparse partials (A4).
fn layerCase(f: *fwd.Fwd, dev: ?gl.Dev, l: *const views.Layer, st: *bufs.State, b: *const bufs.Buffers, x: *const glm.dsa_kernels.Scratch, a: gl.Args, c: Case, in_x: []const u8, attn: []const u8, mlp: []const u8, got: Got) !void {
    const m = f.m;
    const R = c.rows;
    const D = m.hidden;
    const ha = l.attn_hc orelse return error.PlainLayer;
    const hf = l.ffn_hc orelse return error.PlainLayer;
    const moe = l.moe orelse return error.NotMoe;
    const pos: i32 = @intCast(c.pos);
    st.pos = c.pos;
    if (dev) |v| {
        try v.put(b.x.ptr, in_x);
        try v.put(st.pos_dev.ptr, std.mem.asBytes(&pos));
    }
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = ha.@"fn".ptr, .base = ha.base.ptr, .scale = ha.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, D, m.streams);
    try glm.dsa.partialWith(f, l, st, b, R, .{ .nch = (c.pos + R + 511) / 512, .host_pos = c.pos }, x);
    if (dev) |v| try v.get(b.part.ptr, got.attn);
    try gather(dev, b, a, R, D, attn, m.world);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, b.gath.ptr, b.post.ptr, b.comb.ptr, m.world, R, D, m.streams);
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = hf.@"fn".ptr, .base = hf.base.ptr, .scale = hf.scale.ptr }, l.post_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, D, m.streams);
    try glm.moe.partial(f, moe, b, R, .{});
    if (dev) |v| try v.get(b.part.ptr, got.mlp);
    try gather(dev, b, a, R, D, mlp, m.world);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, b.gath.ptr, b.post.ptr, b.comb.ptr, m.world, R, D, m.streams);
    if (dev) |v| try v.get(b.x.ptr, got.x);
}

fn tag(c: Case, buf: []u8) []const u8 {
    return if (c.step) |k| std.fmt.bufPrint(buf, "s{d:0>2}", .{k}) catch unreachable else std.fmt.bufPrint(buf, "chunk@{d}", .{c.pos}) catch unreachable;
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
    var buf: [32]u8 = undefined;
    out("FAIL {s} {s} L{d:0>2} {s}: {d} of {d} elements differ; first [{d}] zig {e} python {e}\n", .{ c.prompt, tag(c, &buf), c.layer, what, n, ours.len / elem, first.?, v(ours[at..], elem), v(want[at..], elem) });
    return false;
}

pub fn run(init: std.process.Init, gpa: std.mem.Allocator, arena: std.mem.Allocator, a: gl.Args) !u8 {
    var ix = try glm.index.open(gpa, a.dir);
    defer ix.deinit();
    var r = try gl.plans(gpa, a, &ix);
    defer r.split.deinit();
    defer r.pk.deinit();
    out("{s}: rank {d}/{d}, {d} layers{s}; {d} stored groups ({d:.2} GiB), {d} packed buffers ({d:.2} GiB)\n", .{ a.dir, a.rank, a.ranks, r.c.run, if (r.split.mtp != null) " and the MTP layer" else "", r.split.groups.items.len, gl.gib(r.split.bytes()), r.pk.items.items.len, gl.gib(r.pk.bytes()) });
    if (r.m.quant != .mlx) {
        out("tf-glm-layer: the MLX checkpoint only (EXL3's rank load is not in the pack pass yet)\n", .{});
        return 2;
    }
    for (a.from..a.to + 1) |l| if (r.m.kinds[l] == .dsa and r.m.mlp_kinds[l] != .moe) {
        out("layer {d}: a DSA layer without the MoE (qualification_6607f886's copy has none)\n", .{l});
        return 2;
    };
    const all = try cases(gpa, arena, a, &r.m);
    defer gpa.free(all);
    if (all.len == 0) {
        out("no DSA layer in {d}-{d}\n", .{ a.from, a.to });
        return 2;
    }

    var driver: cuda.Driver = undefined;
    var ctx: cuda.Context = undefined;
    var stream: cuda.Stream = undefined;
    var pk_kernels: pack.Kernels = undefined;
    var packer: ?gl.Packer = null;
    var set: cuda.aot.Set = undefined;
    var mmk: Mm = undefined;
    var sel: Sel = undefined;
    var no_kda: Kda = undefined; // the DSA and MoE layers launch no KDA kernel
    const before = gl.memAvailableGib();
    if (a.dry) { // stand-in addresses, as check-weights --dry places them
        var base: u64 = 1 << 36;
        for (0..r.split.groups.items.len) |g| {
            r.pk.place(@intCast(g), base);
            base += std.mem.alignForward(u64, r.pk.homeBytes(@intCast(g)), 1 << 20) + (1 << 20);
        }
    } else {
        const image = gl.slurp(gpa, a.fatbin orelse {
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
        const t0 = gl.nowNs();
        packer = .{ .gpa = gpa, .a = a, .ix = &ix, .r = &r, .d = &driver, .stream = stream, .kernels = &pk_kernels, .stored = try gpa.alloc(?cuda.DeviceBuffer, r.split.groups.items.len) };
        @memset(packer.?.stored, null);
        try packer.?.all();
        out("packed rank {d} on the GPU in {d:.1} s; MemAvailable {d:.2} -> {d:.2} GiB\n", .{ a.rank, @as(f64, @floatFromInt(gl.nowNs() - t0)) / 1e9, before, gl.memAvailableGib() });
        set = try cuda.aot.Set.load(gpa, init.io, &driver, ctx.device, a.triton.?);
        mmk = try Mm.load(&ctx, .{ .qmm_group = images.glm_qmm_group, .qmm_prefill = images.glm_qmm_prefill, .experts = images.glm_experts, .experts_prefill = images.glm_experts_prefill });
        sel = try Sel.load(&driver, images.glm_select);
        out("kernels: the Triton set {s}, GLM's four affine images and glm_select; chip {d}.{d}, {d} SMs\n", .{ a.triton.?, mmk.chip.major, mmk.chip.minor, mmk.chip.sms });
    }
    defer if (packer) |*p| {
        for (p.groups.items) |*x| x.free();
        p.groups.deinit(gpa);
        for (p.stored) |*x| if (x.*) |*y| y.free();
        gpa.free(p.stored);
        sel.unload();
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

    // python's prefill and decode Buffers and its long-context State, each one zeroed device block
    const drv: ?*const cuda.Driver = if (a.dry) null else &driver;
    var pb = try bufs.Buffers.init(&r.m, chunk_rows, capacity, true);
    var pb_dev = try placeAll(&pb, "b", drv, stream, 1 << 44);
    defer if (pb_dev) |*x| x.free();
    var scratch = glm.dsa_kernels.Scratch.init(r.m.hl(), pb.lat_s.qa.shape[2], chunk_rows); // A4: sparse partials past pb.lat_s
    var scratch_dev = try placeAll(&scratch, "dsa", drv, stream, 1 << 47);
    defer if (scratch_dev) |*x| x.free();
    var db = try bufs.Buffers.init(&r.m, window_rows, capacity, false);
    var db_dev = try placeAll(&db, "b", drv, stream, 1 << 45);
    defer if (db_dev) |*x| x.free();
    var st = try bufs.State.init(gpa, &r.m, capacity, window_rows, true);
    defer st.deinit(gpa);
    var st_dev = try placeAll(&st, "st", drv, stream, 1 << 46);
    defer if (st_dev) |*x| x.free();

    var t = trace.Trace.init(gpa);
    defer t.deinit();
    var f = if (a.dry) fwd.Fwd.recording(&r.m, &t) else fwd.Fwd.launching(&r.m, .{ .stream = stream, .set = &set, .kernels = .{ .kda = &no_kda, .mm = &mmk, .dsa = &sel } });
    const dev: ?gl.Dev = if (a.dry) null else .{ .d = &driver, .stream = stream };
    const D = r.m.hidden;
    var equal: usize = 0;
    var per: [gl.max_layers][4]usize = @splat(.{ 0, 0, 0, 0 }); // chunks equal, chunks, steps equal, steps
    var lists: std.ArrayList(struct { key: u64, kind: usize, n: usize, cases: usize }) = .empty; // --dry: each launch list
    const t0 = gl.nowNs();
    for (all) |c| {
        const in_x = try gl.readDump(arena, a, c.before);
        const in_attn = try gl.readDump(arena, a, c.attn);
        const in_mlp = try gl.readDump(arena, a, c.mlp);
        const want_x = try gl.readDump(arena, a, c.after);
        const b = if (c.step == null) &pb else &db;
        const R = c.rows;
        const got: Got = .{ .attn = try arena.alloc(u8, R * D * 4), .mlp = try arena.alloc(u8, R * D * 4), .x = try arena.alloc(u8, R * r.m.streams * D * 2) };
        const kind: usize = if (c.step == null) 0 else 2;
        var buf: [32]u8 = undefined;
        t.reset();
        if (R > b.rows or want_x.len != got.x.len or in_attn.len != r.m.world * got.attn.len) return error.CaseShape;
        layerCase(&f, dev, &w.layers[c.layer], &st, b, &scratch, a, c, in_x, in_attn, in_mlp, got) catch |e| {
            out("FAIL {s} {s} L{d:0>2}: {t}\n", .{ c.prompt, tag(c, &buf), c.layer, e });
            return 1;
        };
        per[c.layer][kind + 1] += 1;
        if (dev != null) {
            const rk = a.rank * R * D * 4;
            const ok_attn = same("attention partial", c, got.attn, in_attn[rk..][0 .. R * D * 4], 4);
            const ok_mlp = same("MoE partial", c, got.mlp, in_mlp[rk..][0 .. R * D * 4], 4);
            const ok_x = same("b.x", c, got.x, want_x, 2);
            if (ok_attn and ok_mlp and ok_x) {
                equal += 1;
                per[c.layer][kind] += 1;
            } else if (a.save) |dir| {
                for ([_][]const u8{ "attn", "moe", "x" }, [_][]u8{ got.attn, got.mlp, got.x }) |n, bytes| try gl.spit(try std.fmt.allocPrint(arena, "{s}/{s}-{s}-L{d:0>2}.{s}.bin", .{ dir, c.prompt, tag(c, &buf), c.layer, n }), bytes);
            }
        } else {
            try f.flush();
            const n = t.events.items.len;
            var h = std.hash.Wyhash.init(kind);
            for (t.events.items) |e| h.update(e.name);
            const key = h.final();
            for (lists.items) |*s| {
                if (s.key == key) {
                    s.cases += 1;
                    break;
                }
            } else { // a launch list not seen yet (the long prompt's chunks differ: dense, both, sparse)
                try lists.append(arena, .{ .key = key, .kind = kind / 2, .n = n, .cases = 1 });
                out("dry: {s} {s} L{d:0>2}, a {s}'s launches ({d}):", .{ c.prompt, tag(c, &buf), c.layer, if (kind == 0) "chunk" else "step", n });
                for (t.events.items) |e| out(" {t}:{s}", .{ e.op, e.name });
                out("\n", .{});
            }
            equal += 1;
            per[c.layer][kind] += 1;
        }
        for ([_][]u8{ in_x, in_attn, in_mlp, want_x, got.attn, got.mlp, got.x }) |x| arena.free(x);
    }
    const secs = @as(f64, @floatFromInt(gl.nowNs() - t0)) / 1e9;
    const verb = if (a.dry) "recorded" else "equal";
    for (a.from..a.to + 1) |l| if (per[l][1] + per[l][3] > 0) out("L{d:0>2}: chunks {d} of {d} {s}, steps {d} of {d} {s}\n", .{ l, per[l][0], per[l][1], verb, per[l][2], per[l][3], verb });
    if (a.dry) {
        out("G-layer-1 DSA --dry rank {d}/{d}: {d} cases recorded in {d:.1} s; launch lists:", .{ a.rank, a.ranks, all.len, secs });
        for (lists.items) |s| out(" {s} {d} x{d},", .{ if (s.kind == 0) "chunk" else "step", s.n, s.cases });
        out("\n", .{});
        return 0;
    }
    out("G-layer-1 DSA layers rank {d}/{d}: {d} of {d} cases equal (attention and MoE partials and b.x, byte for byte) in {d:.1} s; MemAvailable {d:.2} GiB\n", .{ a.rank, a.ranks, equal, all.len, secs, gl.memAvailableGib() });
    return if (equal == all.len) 0 else 1;
}

test "the DSA mode compiles on any host (it runs on a GPU, or --dry on a CPU); its arguments" {
    _ = &run;
    try std.testing.expectEqual(@as(usize, 6144), capacity);
    try std.testing.expectError(error.Usage, gl.parse(&.{ "tf-glm-layer", "kda", "/ckpt", "--oracle", "/o", "--dry" }));
    try std.testing.expectEqualStrings("short,long", (try gl.parse(&.{ "tf-glm-layer", "mlp", "/ckpt", "--oracle", "/o", "--dry" })).prompts);
    const d = try gl.parse(&.{ "tf-glm-layer", "dsa", "/ckpt", "--oracle", "/o", "--dry", "--rank", "1" });
    try std.testing.expect(d.mode == .dsa and d.dry and d.rank == 1 and d.from == 3 and d.to == 7);
    try std.testing.expectEqualStrings("short", d.prompts);
}
