//! The pack pass (qualification_qualification_7c75d777 PLAN §3): a TP rank's stored MLX bytes (cuda_weights.zig's groups) in python-0.6's load-time
//! layouts (weights.py `load`: kernels/qmm.py `pack`, kv_b's rows by head, latent.AbsorbQ4, cuda/experts.py `pack`),
//! each buffer named by python's attribute path as qualification_6607f886's weights.json names it. `plan` lists the buffers from the
//! split; `group` makes those of one stored group with glm_pack.cu (GPU), records the launches, or runs them on host
//! memory (a CPU emulator); a stored group is freed once its last reader has run (`last`).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const md = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const split = @import("cuda_split.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const xo = @import("cuda_pack_x.zig");

/// The EXL3 rank's plan over qualification_765330ce's split (cuda_pack_exl3.zig), where the tree has that split (qualification_qualification_44ba554e-int); an empty
/// namespace without it (qualification_qualification_44ba554e-fwd-b), so this file compiles in both.
pub const exl3 = if (@hasDecl(@import("cuda.zig"), "exl3")) @import("cuda_pack_exl3.zig") else struct {};
pub const x_ops = xo;
/// Mia's q4 dense for an EXL3 rank (TF_GLM_DENSE=q4:DIR): the overlay of its plan, and the views bind adapts.
pub const q4 = @import("cuda_pack_q4.zig");
/// Its FP8 head (TF_GLM_F8=head, patch 0002's make_f8): the plan's head as e4m3 bytes and block scales.
pub const f8 = @import("cuda_pack_f8.zig");
/// mtp_fp8: the MTP layer's projections on the same FP8 (TF_GLM_MTP_FP8=1, BadAd84's lever), and the views bind adapts.
pub const mtp8 = @import("cuda_pack_mtp8.zig");

const Ref = split.Ref;
const Buf = views.Buf;

/// Rows in python's order: output row c is stored row first + (c / block) * stride + c % block (block 0: first + c).
pub const Rows = struct {
    first: usize = 0,
    block: usize = 0,
    stride: usize = 0,

    /// The kernels' (rb, rs) for `n` output rows.
    pub fn map(r: Rows, n: usize) [2]usize {
        return if (r.block == 0) .{ @max(n, 1), 0 } else .{ r.block, r.stride };
    }
    /// The stored row of output row c.
    pub fn at(r: Rows, c: usize) usize {
        const m = r.map(c + 1);
        return r.first + (c / m[0]) * m[1] + c % m[0];
    }
};

/// An MLX triplet's place: words [n, k/8] u32, scales and biases [n, k/64] bf16, rows contiguous.
pub const Trip = struct { w: Ref, s: Ref, b: Ref };

pub const Op = union(enum) {
    copy: Ref, // the stored bytes as they are (embedding, norms, mixes, conv, fp32 vectors, router)
    words: struct { src: Ref, n: usize, k: usize, rows: Rows }, // kernels/qmm.py pack: the tiled words
    major: struct { src: Ref, n: usize, k: usize, rows: Rows }, // its scales or biases, [k/64, npad]
    rows: struct { src: Ref, n: usize, bytes: usize, rows: Rows }, // rows gathered as stored (AbsorbQ4)
    experts: struct { routed: [2]Trip, shared: [2]Trip, mm: usize, e: usize, n: usize, k: usize }, // experts.make's up or down
};

pub const Item = struct {
    path: []const u8, // python's: w.layers[3].dsa.kv_k.weight
    buf: Buf, // dtype and shape as python leaves it; ptr once placed
    op: Op,
    home: u32, // the last stored group it reads: it is made once that group is loaded
    x: ?xo.Op = null, // EXL3's op (cuda_pack_x.zig) when set; `op` is then the stored ref it reads first
};

/// A rank's packed buffers in the split's group order.
pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    items: std.ArrayList(Item) = .empty,
    groups: usize, // the split's stored groups

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
        p.* = undefined;
    }
    /// Bytes of every packed buffer.
    pub fn bytes(p: *const Plan) u64 {
        var n: u64 = 0;
        for (p.items.items) |it| n += it.buf.bytes();
        return n;
    }
    /// The packed bytes made with stored group `g` (one allocation, each buffer on a 256-byte boundary); EXL3's bytes
    /// left in place are the stored group's.
    pub fn homeBytes(p: *const Plan, g: u32) u64 {
        var n: u64 = 0;
        for (p.items.items) |it| if (it.home == g and kept(it) == null) {
            n = std.mem.alignForward(u64, n + it.buf.bytes(), split.alignment);
        };
        return n;
    }
    /// Places group `g`'s buffers from `base` (a device or host address).
    pub fn place(p: *Plan, g: u32, base: u64) void {
        var n: u64 = 0;
        for (p.items.items) |*it| if (it.home == g and kept(it.*) == null) {
            it.buf.ptr = base + n;
            n = std.mem.alignForward(u64, n + it.buf.bytes(), split.alignment);
        };
    }
    /// Points the buffers left in stored group `g` (EXL3's experts) at it, `base` being the group's address.
    pub fn placeKept(p: *Plan, g: u32, base: u64) void {
        for (p.items.items) |*it| if (kept(it.*)) |r| if (r.g == g) {
            it.buf.ptr = base + r.off;
        };
    }
    /// Whether a buffer is left in stored group `g` (which is then never freed).
    pub fn keeps(p: *const Plan, g: u32) bool {
        for (p.items.items) |it| if (kept(it)) |r| if (r.g == g) return true;
        return false;
    }
    /// The last group whose packing reads stored group `g`: `g` can be freed once that group is packed (never when
    /// a buffer is left in it).
    pub fn last(p: *const Plan, g: u32) u32 {
        var out: u32 = g;
        for (p.items.items) |it| if (readsGroup(it, g)) {
            if (kept(it)) |r| if (r.g == g) return std.math.maxInt(u32);
            out = @max(out, it.home);
        };
        return out;
    }
    pub fn find(p: *const Plan, path: []const u8) ?*const Item {
        for (p.items.items) |*it| if (std.mem.eql(u8, it.path, path)) return it;
        return null;
    }
};

fn kept(it: Item) ?Ref {
    return if (it.x) |x| xo.inPlace(x) else null;
}

/// An EXL3 layer's pointer table (`*_ptr`): device addresses, which python's weights.json leaves undigested.
pub fn addressTable(it: Item) bool {
    return it.x != null and it.buf.dtype == .i64 and std.mem.endsWith(u8, it.path, "_ptr");
}

/// Whether making `it` reads stored group `g`.
pub fn readsGroup(it: Item, g: u32) bool {
    if (it.x) |x| return xo.reads(x, g);
    return switch (it.op) {
        .copy => |r| r.g == g,
        inline .words, .major, .rows => |x| x.src.g == g,
        .experts => |x| for (x.routed[0..x.mm], x.shared[0..x.mm]) |r, s| {
            if (r.w.g == g or s.w.g == g) break true;
        } else false,
    };
}

fn at(r: Ref, bytes: usize) Ref {
    return .{ .g = r.g, .off = r.off + bytes };
}

const Builder = struct {
    a: std.mem.Allocator,
    p: *Plan,
    m: *const md.Model,

    fn add(b: *Builder, path: []const u8, leaf: []const u8, buf: Buf, op: Op) !void {
        const full = if (leaf.len == 0) try b.a.dupe(u8, path) else try std.fmt.allocPrint(b.a, "{s}.{s}", .{ path, leaf });
        const home: u32 = switch (op) {
            .copy => |r| r.g,
            inline .words, .major, .rows => |x| x.src.g,
            .experts => |x| @max(x.routed[0].w.g, x.shared[0].w.g),
        };
        try b.p.items.append(b.a, .{ .path = full, .buf = buf, .op = op, .home = home });
    }

    fn copy(b: *Builder, path: []const u8, leaf: []const u8, src: Ref, dtype: views.DType, shape: []const usize) !void {
        try b.add(path, leaf, Buf.of(dtype, shape), .{ .copy = src });
    }

    /// qmm.make_q4 of `n` rows of a stored triplet (`rows` picks them): weight, scales, biases.
    fn q4(b: *Builder, path: []const u8, src: split.Q4, rows: Rows, n: usize) !void {
        const k: usize = src.k;
        const q = views.Q4.of(n, k);
        try b.add(path, "weight", q.weight, .{ .words = .{ .src = src.w, .n = n, .k = k, .rows = rows } });
        try b.add(path, "scales", q.scales, .{ .major = .{ .src = src.s, .n = n, .k = k, .rows = rows } });
        try b.add(path, "biases", q.biases, .{ .major = .{ .src = src.b, .n = n, .k = k, .rows = rows } });
    }

    fn sub(b: *Builder, path: []const u8, leaf: []const u8) ![]const u8 {
        return std.fmt.allocPrint(b.a, "{s}.{s}", .{ path, leaf });
    }

    fn layer(b: *Builder, path: []const u8, l: *const split.Layer) !void {
        const m = b.m;
        const D = m.hidden;
        if (l.hc) |hc| for (hc, [_][]const u8{ "attn_hc", "ffn_hc" }) |h, site| {
            const p = try b.sub(path, site);
            try b.copy(p, "fn", h.fnp, .bf16, &.{ 24, m.streams * D });
            try b.copy(p, "base", h.base, .f32, &.{24});
            try b.copy(p, "scale", h.scale, .f32, &.{3});
        };
        try b.copy(path, "in_norm", l.in_norm, .bf16, &.{D});
        try b.copy(path, "post_norm", l.post_norm, .bf16, &.{D});
        switch (l.attn) {
            .kda => |x| try b.kda(try b.sub(path, "kda"), &x),
            .mla => |x| try b.dsa(try b.sub(path, "dsa"), &x),
        }
        switch (l.mlp) {
            .dense => |x| {
                const p = try b.sub(path, "mlp");
                try b.q4(try b.sub(p, "gu"), x.gate_up, .{}, 2 * m.dl());
                try b.q4(try b.sub(p, "down"), x.down, .{}, D);
            },
            .moe => |x| try b.moe(try b.sub(path, "moe"), &x),
        }
    }

    /// KDAW: proj = stack(q, k, v, f_a, g_a, b) as the split stacked it; conv already [q | k | v][channel][tap].
    fn kda(b: *Builder, p: []const u8, x: *const split.Kda) !void {
        const m = b.m;
        const W = m.ll() * m.lin_dim;
        try b.q4(try b.sub(p, "proj"), x.in_proj, .{}, m.kdaWidth());
        try b.q4(try b.sub(p, "fb"), x.f_b, .{}, W);
        try b.q4(try b.sub(p, "gb"), x.g_b, .{}, W);
        try b.copy(p, "conv", x.conv, .bf16, &.{ 3 * W, m.conv });
        try b.copy(p, "a_log", x.a_log, .f32, &.{m.ll()});
        try b.copy(p, "dt_bias", x.dt_bias, .f32, &.{W});
        try b.copy(p, "norm", x.o_norm, .bf16, &.{m.lin_dim});
        try b.q4(try b.sub(p, "o"), x.o_proj, .{}, m.hidden);
    }

    /// DSAW with IndexW and latent.AbsorbQ4: the split's x_proj is [q_a | kv_a | indexer.wk | weights_proj] and
    /// qr_proj [q_b | indexer.wq_b]; kv_b's rows are the rank's heads, each qk_dim key rows then v_dim value rows.
    fn dsa(b: *Builder, p: []const u8, x: *const split.Mla) !void {
        const m = b.m;
        const H = m.hl();
        const per = m.qk_dim + m.v_dim;
        const keys: Rows = .{ .block = m.qk_dim, .stride = per };
        const vals: Rows = .{ .first = m.qk_dim, .block = m.v_dim, .stride = per };
        try b.q4(try b.sub(p, "proj"), x.x_proj, .{}, m.q_lora + m.kv_lora);
        try b.copy(p, "q_norm", x.q_norm, .bf16, &.{m.q_lora});
        try b.copy(p, "kv_norm", x.kv_norm, .bf16, &.{m.kv_lora});
        try b.q4(try b.sub(p, "q_b"), x.qr_proj, .{}, H * m.qk_dim);
        try b.q4(try b.sub(p, "kv_k"), x.kv_b, keys, H * m.qk_dim);
        try b.q4(try b.sub(p, "kv_v"), x.kv_b, vals, H * m.v_dim);
        try b.q4(try b.sub(p, "o"), x.o_proj, .{}, m.hidden);
        const ix = try b.sub(p, "index");
        try b.q4(try b.sub(ix, "kw"), x.x_proj, .{ .first = m.q_lora + m.kv_lora }, m.index_dim + m.index_heads);
        try b.q4(try b.sub(ix, "qb"), x.qr_proj, .{ .first = H * m.qk_dim }, m.index_heads * m.index_dim);
        try b.copy(ix, "ln_w", x.k_norm_w, .bf16, &.{m.index_dim});
        try b.copy(ix, "ln_b", x.k_norm_b, .bf16, &.{m.index_dim});
        try b.copy(ix, "gate", x.igate, .bf16, &.{ m.index_dim, m.hidden });
        try b.copy(ix, "ape", x.ape, .bf16, &.{ m.kpool, m.index_dim });
        // latent.AbsorbQ4 (ENABLED, group 64): kv_b's key and value rows as stored, [H, rows, ...] per head
        const ab = try b.sub(p, "absorb");
        const k8 = m.kv_lora / 8;
        const kg = m.kv_lora / 64;
        for ([_][]const u8{ "k", "v" }, [_]Rows{ keys, vals }, [_]usize{ m.qk_dim, m.v_dim }) |kv, r, d| {
            const n = H * d;
            try b.add(ab, try std.fmt.allocPrint(b.a, "w{s}w", .{kv}), Buf.of(.i32, &.{ H, d, k8 }), .{ .rows = .{ .src = x.kv_b.w, .n = n, .bytes = k8 * 4, .rows = r } });
            try b.add(ab, try std.fmt.allocPrint(b.a, "w{s}s", .{kv}), Buf.of(.bf16, &.{ H, d, kg }), .{ .rows = .{ .src = x.kv_b.s, .n = n, .bytes = kg * 2, .rows = r } });
            try b.add(ab, try std.fmt.allocPrint(b.a, "w{s}b", .{kv}), Buf.of(.bf16, &.{ H, d, kg }), .{ .rows = .{ .src = x.kv_b.b, .n = n, .bytes = kg * 2, .rows = r } });
        }
    }

    /// MoEW (MLX): the router, its bias, and experts.make([gate, up], down) with the shared expert as expert E; the
    /// rows and inputs are the split's (TP2 by rows: every expert's half; the shared expert's halves stacked gate, up).
    fn moe(b: *Builder, p: []const u8, x: *const split.Moe) !void {
        const m = b.m;
        const e: usize = x.experts;
        const n: usize = x.gate.n; // an expert's gate (and up) rows
        const k: usize = x.gate.k;
        if (x.sh_gate_up.n != 2 * n or x.down.k != n or x.sh_down.k != n) return error.SharedExpertShape;
        try b.copy(p, "router", x.router, .bf16, &.{ m.experts, m.hidden });
        try b.copy(p, "bias", x.bias, .f32, &.{m.experts});
        const sg = x.sh_gate_up;
        const gate: Trip = .{ .w = sg.w, .s = sg.s, .b = sg.b };
        const up: Trip = .{ .w = at(sg.w, split.Q4.wBytes(n, k)), .s = at(sg.s, split.Q4.sBytes(n, k)), .b = at(sg.b, split.Q4.sBytes(n, k)) };
        const ex = try b.sub(p, "experts");
        try b.add(ex, "up", Buf.of(.i32, &.{ e + 1, n / 32, k / 64, 2, 288 }), .{ .experts = .{
            .routed = .{ trip(x.gate), trip(x.up) },
            .shared = .{ gate, up },
            .mm = 2,
            .e = e,
            .n = n,
            .k = k,
        } });
        const dn: usize = x.down.n;
        const sd = trip(x.sh_down);
        try b.add(ex, "down", Buf.of(.i32, &.{ e + 1, dn / 32, n / 64, 1, 288 }), .{ .experts = .{
            .routed = .{ trip(x.down), trip(x.down) },
            .shared = .{ sd, sd },
            .mm = 1,
            .e = e,
            .n = dn,
            .k = n,
        } });
    }
};

fn trip(q: split.Q4) Trip {
    return .{ .w = q.w, .s = q.s, .b = q.b };
}

/// Every buffer python-0.6's weights.load leaves for this rank (MLX 4-bit): the embedding, `m.layers` layers, the norm,
/// the head's rows and, when the split loaded it, the MTP head (`p.layers[c.layers]`, python's `Weights.mtp`).
pub fn plan(gpa: std.mem.Allocator, m: *const md.Model, c: *const cfg.Config, p: *const split.Plan) !Plan {
    if (m.quant == .exl3) return error.Exl3PlansOverItsSplit; // EXL3: exl3.plan over qualification_765330ce's split (cuda_pack_exl3.zig)
    if (m.group_size != 64) return error.NotGroup64;
    var out: Plan = .{ .arena = .init(gpa), .groups = p.groups.items.len };
    errdefer out.deinit();
    var b: Builder = .{ .a = out.arena.allocator(), .p = &out, .m = m };
    const V = m.vocab;
    const D = m.hidden;
    try b.copy("w.embed[0]", "", p.embed.w, .i32, &.{ V, D / 8 });
    try b.copy("w.embed[1]", "", p.embed.s, .bf16, &.{ V, D / 64 });
    try b.copy("w.embed[2]", "", p.embed.b, .bf16, &.{ V, D / 64 });
    for (0..m.layers) |i| try b.layer(try std.fmt.allocPrint(b.a, "w.layers[{d}]", .{i}), &p.layers[i]);
    try b.copy("w.norm", "", p.norm, .bf16, &.{D});
    try b.q4("w.head", p.head, .{}, m.vl());
    if (p.mtp) |x| if (m.mtp_layers > 0) {
        try b.copy("w.mtp", "enorm", x.enorm, .bf16, &.{D});
        try b.copy("w.mtp", "hnorm", x.hnorm, .bf16, &.{D});
        try b.q4("w.mtp.eh", x.eh_proj, .{}, D);
        try b.copy("w.mtp", "norm", x.norm, .bf16, &.{D});
        try b.layer("w.mtp.layer", &p.layers[c.layers]);
    };
    return out;
}

/// glm_pack.cu's kernels, numbered as zig/tests/cuda/glm_pack_emu.cpp runs them (4 is python's pack_kernel<2> there).
pub const Kern = enum(u8) { q4 = 0, major = 1, rows = 2, experts = 3, quant4 = 5, quant4s = 6, f8 = 7, f8s = 8 };

fn kname(k: Kern) [:0]const u8 {
    return switch (k) {
        inline else => |x| "tf_glm_pack_" ++ @tagName(x), // tf_glm_pack_q4 .. tf_glm_pack_f8s, glm_pack.cu's names
    };
}

/// The glm_pack fatbin, loaded.
pub const Kernels = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    fns: std.EnumArray(Kern, cuda.Function),

    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        var k: Kernels = .{ .d = d, .module = try cuda.Module.load(d, image), .fns = undefined };
        errdefer k.module.unload();
        for (std.enums.values(Kern)) |x| k.fns.set(x, try k.module.function(kname(x)));
        return k;
    }
    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

/// A CPU run's launches over host addresses (the emulator).
pub const Host = *const fn (kern: Kern, grid: [3]u32, block: [3]u32, args: *cuda.Args) anyerror!void;

/// What runs the steps when the Fwd neither records nor holds a GPU: on the GPU through `f.gpu`, `gpu` is required.
pub const Engine = union(enum) { gpu: *const Kernels, host: Host };

/// The rank's Gpu for the pack pass alone (check-weights): its launches take the stream only; the Triton set and the
/// forward's kernels are never read here and stay unset.
pub fn packOnly(stream: cuda.Stream) fwd.Gpu {
    return .{ .stream = stream, .set = undefined, .kernels = undefined };
}

/// The Fwd of a CPU run: neither recording nor on a GPU, so `make` hands each launch to the Engine's host emulator.
pub fn hostFwd(m: *const md.Model) fwd.Fwd {
    return .{ .m = m, .tri = .{ .set = null } };
}

/// A launch recorded (`glm_pack.<kernel>`: grid, block, then each argument's value), run on `f.gpu`'s stream, or
/// handed to the host emulator.
pub fn launch(f: *fwd.Fwd, e: Engine, kern: Kern, grid: [3]u32, block: u32, args: *cuda.Args) !void {
    if (f.rec) |t| {
        const dims = [_]trace.Val{ .{ .int = grid[0] }, .{ .int = grid[1] }, .{ .int = grid[2] } };
        var vals: [16]trace.Val = undefined;
        vals[0] = .{ .list = &dims };
        vals[1] = .{ .int = block };
        for (0..args.count) |i| vals[2 + i] = .{ .int = @bitCast(args.integer(i).?) };
        var name: [32]u8 = undefined;
        return t.call(.ext, try std.fmt.bufPrint(&name, "glm_pack.{t}", .{kern}), vals[0 .. 2 + args.count]);
    }
    if (f.gpu) |g| {
        const k = switch (e) {
            .gpu => |k| k,
            .host => return error.NoKernels,
        };
        return cuda.launch.launch(k.fns.get(kern), .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, g.stream, args);
    }
    switch (e) {
        .host => |h| try h(kern, grid, .{ block, 1, 1 }, args),
        .gpu => return error.NoGpu,
    }
}

/// Stored bytes as they are: cuMemcpyDtoDAsync on the GPU, a memcpy on host addresses.
pub fn copyBytes(f: *fwd.Fwd, e: Engine, dst: u64, src: u64, n: u64) !void {
    if (n == 0) return;
    if (f.rec) |t| return t.call(.ext, "glm_pack.copy", &.{ .{ .int = @bitCast(dst) }, .{ .int = @bitCast(src) }, .{ .int = @intCast(n) } });
    if (f.gpu) |g| {
        const d = switch (e) {
            .gpu => |k| k.d,
            .host => return error.NoKernels,
        };
        return d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, n, g.stream.handle), "cuMemcpyDtoDAsync");
    }
    const to: [*]u8 = @ptrFromInt(dst);
    const from: [*]const u8 = @ptrFromInt(src);
    @memcpy(to[0..n], from[0..n]);
}

fn blocks(total: u64) [3]u32 {
    return .{ @intCast((total + 255) / 256), 1, 1 };
}

fn int(x: usize) c_int {
    return @intCast(x);
}

/// One item's launches (or copy): `stored[g]` is stored group g's base address, the item's own `buf.ptr` its output.
pub fn make(f: *fwd.Fwd, e: Engine, it: *const Item, stored: []const u64) !void {
    if (it.x) |x| return xo.make(f, e, it, x, stored);
    const out = it.buf.ptr;
    switch (it.op) {
        .copy => |r| try copyBytes(f, e, out, stored[r.g] + r.off, it.buf.bytes()),
        .words => |x| {
            const k8 = x.k / 8;
            const total: u64 = it.buf.numel();
            const rm = x.rows.map(x.n);
            var a: cuda.Args = .{};
            a.add(stored[x.src.g] + x.src.off + x.rows.first * k8 * 4);
            for ([_]usize{ x.n, k8, x.k / 64, rm[0], rm[1] }) |v| a.add(int(v));
            a.add(out);
            a.add(@as(c_longlong, @intCast(total)));
            try launch(f, e, .q4, blocks(total), 256, &a);
        },
        .major => |x| {
            const kg = x.k / 64;
            const npad = (x.n + 127) / 128 * 128;
            const rm = x.rows.map(x.n);
            var a: cuda.Args = .{};
            a.add(stored[x.src.g] + x.src.off + x.rows.first * kg * 2);
            for ([_]usize{ x.n, kg, npad, rm[0], rm[1] }) |v| a.add(int(v));
            a.add(out);
            try launch(f, e, .major, blocks(@as(u64, kg) * npad), 256, &a);
        },
        .rows => |x| {
            const rm = x.rows.map(x.n);
            var a: cuda.Args = .{};
            a.add(stored[x.src.g] + x.src.off + x.rows.first * x.bytes);
            for ([_]usize{ rm[0], rm[1], x.bytes }) |v| a.add(int(v));
            a.add(@as(c_longlong, @intCast(x.n)));
            a.add(out);
            try launch(f, e, .rows, blocks(@as(u64, x.n) * x.bytes), 256, &a);
        },
        .experts => |x| {
            const nb = x.n / 32;
            const kg = x.k / 64;
            const shared_at: u64 = @as(u64, x.e) * nb * kg * x.mm * 288 * 4; // expert E: the shared expert
            for (0..x.mm) |m| for ([_]Trip{ x.routed[m], x.shared[m] }, [_]usize{ x.e, 1 }, [_]u64{ 0, shared_at }) |t, count, off| {
                var a: cuda.Args = .{};
                for ([_]Ref{ t.w, t.s, t.b }) |r| a.add(stored[r.g] + r.off);
                a.add(out + off);
                for ([_]usize{ x.n, x.k / 8, kg, nb, m, x.mm }) |v| a.add(int(v));
                try launch(f, e, .experts, .{ @intCast(kg), @intCast(nb), @intCast(count) }, 288, &a);
            };
        },
    }
}

/// Every buffer made with stored group `g` (placed first): python's tensors that need only groups up to `g`.
pub fn group(f: *fwd.Fwd, e: Engine, p: *const Plan, g: u32, stored: []const u64) !void {
    for (p.items.items) |*it| if (it.home == g) try make(f, e, it, stored);
}

/// python's dtype names, as weights.json gives them.
pub fn dtypeName(d: views.DType) []const u8 {
    return switch (d) {
        .bf16 => "bfloat16",
        .f32 => "float32",
        .f16 => "float16",
        .i32 => "int32",
        .i64 => "int64",
        .i16 => "int16",
        .u8 => "uint8",
        .f8e4m3 => "float8_e4m3fn",
    };
}

/// A cuda_views.walk path (w.layers.3.kda.proj.weight, w.embed.0) in python's form (w.layers[3]..., w.embed[0]).
pub fn pythonPath(out: []u8, walk_path: []const u8) ![]const u8 {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, walk_path, '.');
    var first = true;
    while (it.next()) |part| {
        const index = part.len > 0 and for (part) |ch| {
            if (!std.ascii.isDigit(ch)) break false;
        } else true;
        const s = if (index) try std.fmt.bufPrint(out[n..], "[{s}]", .{part}) else try std.fmt.bufPrint(out[n..], "{s}{s}", .{ if (first) "" else ".", part });
        n += s.len;
        first = false;
    }
    return out[0..n];
}

/// cuda_views' buffers pointed at the packed ones of the same path, EXL3's draft_head too (cuda_pack_exl3's quant4 of
/// the head); a view with no packed buffer is an error, and so is one left at address 0 after (UnboundView, the first
/// named). Returns how many views were bound.
pub fn bind(w: *views.Weights, p: *const Plan) !usize {
    _ = try q4.adapt(w, p); // the dense-q4 form: its B16 projections become the Q4 views the plan packs
    _ = f8.adapt(w, p); // and its head the F8 view, when the plan makes it (TF_GLM_F8=head)
    _ = mtp8.adapt(w, p); // and the MTP layer's projections, when the plan makes them FP8 (TF_GLM_MTP_FP8=1)
    var ctx: Binder = .{ .p = p };
    try views.walk(w, views.Path.init("w"), &ctx, Binder.visit);
    const u = try Unbound.of(w);
    if (u.count > 0) {
        std.log.err("glm pack: {d} views unbound after bind, the first {s}: the forward would read them at address 0", .{ u.count, u.name() });
        return error.UnboundView;
    }
    return ctx.bound;
}

/// The views a forward reads (every buffer but a host one) still at address 0: how many, and the first one's path.
pub const Unbound = struct {
    count: usize = 0,
    first: [224]u8 = undefined,
    len: usize = 0,

    pub fn of(w: *views.Weights) !Unbound {
        var u: Unbound = .{};
        try views.walk(w, views.Path.init("w"), &u, visit);
        return u;
    }
    pub fn name(u: *const Unbound) []const u8 {
        return u.first[0..u.len];
    }
    fn visit(u: *Unbound, path: []const u8, buf: *Buf) anyerror!void {
        if (buf.host or buf.ptr != 0) return;
        if (u.count == 0) u.len = (try pythonPath(&u.first, path)).len;
        u.count += 1;
    }
};

const Binder = struct {
    p: *const Plan,
    bound: usize = 0,

    fn visit(b: *Binder, path: []const u8, buf: *Buf) anyerror!void {
        // a host tensor (Exl3RoutedExperts.trellis_bytes, for GB/s alone): python's weights.json lists the device's
        // tensors, nothing in the forward reads it, and no packed buffer holds it
        if (buf.host) return;
        var tmp: [224]u8 = undefined;
        var py = try pythonPath(&tmp, path);
        var named: [232]u8 = undefined;
        const dot = std.mem.lastIndexOfScalar(u8, py, '.').?;
        const leaf = py[dot + 1 ..];
        if (std.mem.endsWith(u8, py[0..dot], ".absorb") and (leaf[0] == 'k' or leaf[0] == 'v')) // AbsorbQ4 kw .. vb are wkw .. wvb (AbsorbW's wk, wv as they are)
            py = try std.fmt.bufPrint(&named, "{s}.w{s}", .{ py[0..dot], leaf });
        const it = b.p.find(py) orelse {
            std.log.err("glm pack: no packed buffer for the view {s}", .{py});
            return error.UnpackedView;
        };
        if (it.buf.dtype != buf.dtype or it.buf.dims != buf.dims or !std.mem.eql(usize, &it.buf.shape, &buf.shape)) {
            std.log.err("glm pack: {s} is packed {t} {any}, viewed {t} {any}", .{ py, it.buf.dtype, it.buf.shape[0..it.buf.dims], buf.dtype, buf.shape[0..buf.dims] });
            return error.ViewMismatch;
        }
        buf.ptr = it.buf.ptr;
        b.bound += 1;
    }
};

test "python's paths from the views' walk, and the row maps" {
    _ = q4; // cuda_pack_q4.zig's tests
    _ = f8; // cuda_pack_f8.zig's
    _ = mtp8; // cuda_pack_mtp8.zig's
    var tmp: [96]u8 = undefined;
    try std.testing.expectEqualStrings("w.layers[3].kda.proj.weight", try pythonPath(&tmp, "w.layers.3.kda.proj.weight"));
    try std.testing.expectEqualStrings("w.embed[0]", try pythonPath(&tmp, "w.embed.0"));
    const vals: Rows = .{ .first = 256, .block = 256, .stride = 512 }; // kv_b's value rows: head h's at 512 h + 256
    try std.testing.expectEqual(@as(usize, 256), vals.at(0));
    try std.testing.expectEqual(@as(usize, 511), vals.at(255));
    try std.testing.expectEqual(@as(usize, 768), vals.at(256));
    try std.testing.expectEqual(@as(usize, 1023), vals.at(511));
    try std.testing.expectEqual(@as(usize, 7), (Rows{ .first = 7 }).at(0));
}

test "bind points EXL3's draft head at its packed quant4 buffers (qualification_d3c2a5a7: it read address 0)" {
    var p: Plan = .{ .arena = .init(std.testing.allocator), .groups = 1 };
    defer p.deinit();
    const q = views.Q4.of(256, 128);
    var none: [0]views.Layer = .{};
    var w: views.Weights = .{ .embed = .{ .b16 = Buf.of(.bf16, &.{ 256, 128 }) }, .layers = &none, .norm = Buf.of(.bf16, &.{128}), .head = .{ .b16 = views.B16.of(256, 128) }, .draft_head = q };
    const paths = [_][]const u8{ "w.embed", "w.norm", "w.head.weight", "w.draft_head.weight", "w.draft_head.scales", "w.draft_head.biases" };
    for (paths, [_]Buf{ w.embed.b16, w.norm, w.head.b16.weight, q.weight, q.scales, q.biases }) |path, buf|
        try p.items.append(p.arena.allocator(), .{ .path = path, .buf = buf, .op = .{ .copy = .{ .g = 0, .off = 0 } }, .home = 0 });
    p.place(0, 1 << 36);
    try std.testing.expectEqual(@as(usize, 6), try bind(&w, &p));
    const d = w.draft_head.?;
    for ([_]Buf{ d.weight, d.scales, d.biases }, paths[3..]) |b, path| try std.testing.expect(b.ptr != 0 and b.ptr == p.find(path).?.buf.ptr);
}

test "a view the forward reads still at address 0 is found and named; a host buffer is not (qualification_d3c2a5a7 §7 item 2)" {
    var none: [0]views.Layer = .{};
    var w: views.Weights = .{ .embed = .{ .b16 = Buf.of(.bf16, &.{ 256, 128 }) }, .layers = &none, .norm = Buf.of(.bf16, &.{128}), .head = .{ .b16 = views.B16.of(256, 128) }, .draft_head = views.Q4.of(256, 128) };
    for ([_]*Buf{ &w.embed.b16, &w.norm, &w.head.b16.weight, &w.draft_head.?.weight }, 1..) |b, i| b.ptr = i << 36;
    const u = try Unbound.of(&w); // the draft head's scales and biases left at 0, as bind's old skip left all three
    try std.testing.expectEqual(@as(usize, 2), u.count);
    try std.testing.expectEqualStrings("w.draft_head.scales", u.name());
    w.draft_head.?.scales.ptr = 5 << 36;
    w.draft_head.?.biases.host = true; // pinned host memory, never a kernel argument
    try std.testing.expectEqual(@as(usize, 0), (try Unbound.of(&w)).count);
}
