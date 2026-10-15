//! GLM-5.3-Flash's affine kernels: glm_qmm_group, glm_qmm_prefill, glm_experts and glm_experts_prefill (copies of
//! python-0.6's tensorfold_qmm_v5 and tensorfold_experts_v7 device code with GLM's instances, zig/tests/cuda/copies.py)
//! and each C++ wrapper's launches: qmm.cpp qmm_group and qmm_prefill, experts.cpp plan, run and prefill.
//! Sources: the kernels and wrappers are Ash Hart's (github.com/ashhart), python-0.6 ed78d6f.

const std = @import("std");
const cuda = @import("cuda");

/// Mangled names in the four images (cuobjdump -symbols of their sm_121 and sm_120 fatbins).
pub const sym = struct {
    pub const group_bf16 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const group_f32 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const group7_bf16 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi8ELi128ELi1ELi4ELi4ELb0ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const group7_f32 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi8ELi128ELi1ELi4ELi4ELb1ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const group8_bf16 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi16ELi128ELi1ELi8ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const group8_f32 = "_ZN16tf_glm_qmm_group12group_kernelILi64ELi16ELi128ELi1ELi8ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
    pub const prefill_bf16 = "_ZN18tf_glm_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const prefill_f32 = "_ZN18tf_glm_qmm_prefill14prefill_kernelILi64ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii";
    pub const gate_up = "_ZN14tf_glm_experts13expert_kernelILi64ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const down = "_ZN14tf_glm_experts13expert_kernelILi64ELi1ELi0ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const plan = "_ZN14tf_glm_experts11plan_kernelEPKiiiiPiS2_S2_";
    pub const plan_rank = "_ZN14tf_glm_experts9plan_rankEPKiiiPiS2_";
    pub const plan_offsets = "_ZN14tf_glm_experts12plan_offsetsEiiiPiS0_S0_";
    pub const plan_scatter = "_ZN14tf_glm_experts12plan_scatterEPKiiiS1_S1_Pi";
    pub const pre_gate_up = "_ZN22tf_glm_experts_prefill14prefill_kernelILi64ELi2ELi2ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
    pub const pre_down = "_ZN22tf_glm_experts_prefill14prefill_kernelILi64ELi1ELi3ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4iiPKiS8_S8_Pvif";
};

/// Every kernel of the four images: its image (0 qmm_group, 1 qmm_prefill, 2 experts, 3 experts_prefill) and symbol.
pub const Kernel = enum(u8) {
    group_bf16,
    group_f32,
    group7_bf16,
    group7_f32,
    group8_bf16,
    group8_f32,
    prefill_bf16,
    prefill_f32,
    gate_up,
    down,
    plan,
    plan_rank,
    plan_offsets,
    plan_scatter,
    pre_gate_up,
    pre_down,

    pub const count = @typeInfo(Kernel).@"enum".field_names.len;

    pub fn image(k: Kernel) usize {
        return switch (k) {
            .group_bf16, .group_f32, .group7_bf16, .group7_f32, .group8_bf16, .group8_f32 => 0,
            .prefill_bf16, .prefill_f32 => 1,
            .gate_up, .down, .plan, .plan_rank, .plan_offsets, .plan_scatter => 2,
            .pre_gate_up, .pre_down => 3,
        };
    }

    pub fn symbol(k: Kernel) [:0]const u8 {
        return switch (k) {
            inline else => |t| @field(sym, @tagName(t)),
        };
    }

    /// The dynamic shared bytes its wrapper raises the kernel's limit to before the first launch (0: never raised).
    pub fn raised(k: Kernel) u32 {
        return switch (k) {
            .group_bf16, .group_f32 => group_smem,
            .group7_bf16, .group7_f32 => tiles[7].?.smem(),
            .group8_bf16, .group8_f32 => tiles[8].?.smem(),
            .prefill_bf16, .prefill_f32 => prefill_smem,
            .pre_gate_up => preSmem(2),
            .pre_down => preSmem(1),
            else => 0,
        };
    }
};

/// qmm_frag.cuh LaneTile<64, 16, 64, 1, 4, 8>::SMEM: eight stages of 16 rows (or the K-slice partials, if larger).
pub const group_smem: u32 = tiles[2].?.smem();
/// qmm_prefill.cu Tile<64, 128, 128, 2, 4, 3>::SMEM.
pub const prefill_smem: u32 = 3 * (128 * 64 * 2 + 128 * 64 / 2 + 2 * 128 * 2);

/// A tile of qmm_group.cu's dispatch: launch<64, BM, BN, WM, WN, STAGES, F32, SWAP> and its two kernels (bf16, fp32 out).
pub const Tile = struct {
    bm: u32,
    bn: u32,
    wm: u32,
    wn: u32,
    stages: u32,
    bf16: Kernel,
    f32: Kernel,

    /// qmm_frag.cuh LaneTile<64, BM, BN, WM, WN, STAGES>::SMEM: the stages, or the K-slice partials if larger.
    pub fn smem(t: Tile) u32 {
        const stage = t.bm * 64 * 2 + t.bn * 64 / 2 + 2 * t.bn * 2 + t.bm * 4;
        const partials = (t.bm / t.wm / 16) * (t.bn / t.wn / 8) * 4 * (t.wm * t.wn * 32) * 4;
        return @max(t.stages * stage, partials);
    }
};

/// The tiles glm_qmm_group builds, by dispatch's number (python's `case`): 2 a GB10's to 16 rows; 7 (rows <= 8, swapped
/// 8-row tiles) and 8 (rows 9-16) other sm_12x chips' (the RTX 5090: qualification_18588626); 1, 3-6 and 9-12 are not built.
pub const tiles = blk: {
    var t: [13]?Tile = @splat(null);
    t[2] = .{ .bm = 16, .bn = 64, .wm = 1, .wn = 4, .stages = 8, .bf16 = .group_bf16, .f32 = .group_f32 };
    t[7] = .{ .bm = 8, .bn = 128, .wm = 1, .wn = 4, .stages = 4, .bf16 = .group7_bf16, .f32 = .group7_f32 };
    t[8] = .{ .bm = 16, .bn = 128, .wm = 1, .wn = 8, .stages = 4, .bf16 = .group8_bf16, .f32 = .group8_f32 };
    break :blk t;
};

/// experts_prefill.cu Pre<64, M, 2, 2, 4>: as many stages as fit 48 KiB (4, 3, else 2) of 64 rows and 4 weight blocks.
pub fn preSmem(m: u32) u32 {
    const block = 32 * (4 * 64 / 128) + 4 * (4 / 2); // Geo<64>::BLOCK, uint4
    const bytes = (64 * 8 + 4 * m * block) * 16;
    const stages: u32 = if (bytes * 4 <= 49152) 4 else if (bytes * 3 <= 49152) 3 else 2;
    return stages * bytes;
}

/// A launch as the wrapper makes it: the kernel, its configuration and its packed arguments.
pub const Launch = struct { kernel: Kernel, cfg: cuda.Config, args: cuda.Args };

pub const Error = error{ Invalid, NotBuilt };

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The device as qmm_group_cuda and experts' launch read it (getCurrentDeviceProperties): a GB10 is 12.1, 48 SMs.
pub const Chip = struct { major: u32 = 12, minor: u32 = 1, sms: u32 = 48 };

/// qmm_group.cu's Part and Parts, passed by value as a __grid_constant__: up to four projections of one input.
pub const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
pub const Parts = extern struct { p: [4]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

/// The widest argument a launch passes (qmm_group's Parts, by value).
pub const max_arg_bytes = @sizeOf(Parts);

/// One projection as qmm_group takes it: words, scales and biases (rows `npad` apart, the wrapper's scales.stride(0)),
/// its (m, n) output, and its K split.
pub const Proj = struct { w: u64, scales: u64, biases: u64, out: u64, n: usize, npad: usize, sk: usize };

/// qmm.cpp qmm_group(x, xs, ws, scales, biases, outs, ns, sks, f32, tile, pdl): x (m, k) bf16 with rows `x_stride`
/// elements apart, xs its (m, k / 64) group sums; `tile` 0 picks by rows and chip, `pdl` -1 by chip.
pub const Group = struct { x: u64, xs: u64, m: usize, k: usize, x_stride: usize, parts: []const Proj, f32_out: bool, tile: i64 = 0, pdl: i64 = -1 };

/// qmm_group_cuda's dispatch and launch: tile 2 (rows <= 16 on a GB10), and tiles 7 (rows <= 8) and 8 (rows 9-16) on
/// other sm_12x chips (the RTX 5090) are built into glm_qmm_group; any other tile is NotBuilt.
pub fn groupLaunch(g: Group, chip: Chip) Error!Launch {
    if (g.parts.len < 1 or g.parts.len > 4 or g.m < 1 or g.k % 64 != 0 or g.x_stride < g.k) return error.Invalid;
    if (g.x % 16 != 0 or (g.m > 1 and g.x_stride % 8 != 0)) return error.Invalid;
    var c: usize = 1;
    for (g.parts) |p| {
        if ((p.sk != 1 and p.sk != 2 and p.sk != 4 and p.sk != 8) or (g.k / 64) % p.sk != 0) return error.Invalid;
        if (p.npad < (p.n + 127) / 128 * 128) return error.Invalid;
        c = @max(c, p.sk);
    }
    const gb10 = chip.major == 12 and chip.minor == 1;
    var tile = g.tile;
    if (tile == 0 and gb10) tile = if (g.m <= 16) 2 else if (g.m <= 32) 3 else if (g.m <= 64) 4 else 5;
    if (tile == 0) tile = if (g.m <= 8) 7 else if (g.m <= 16) 8 else if (g.m <= 32) 9 else if (g.m <= 64) 4 else 5;
    if (tile < 1 or tile > 12) return error.Invalid;
    const t = tiles[@intCast(tile)] orelse return error.NotBuilt; // launch<64, BM, BN, WM, WN, STAGES, F32, SWAP>
    const rows_t = (g.m + t.bm - 1) / t.bm;
    var parts = std.mem.zeroes(Parts);
    parts.count = int(g.parts.len);
    var clusters: usize = 0;
    for (g.parts, 0..) |p, i| {
        const n_tiles = (p.n + t.bn - 1) / t.bn;
        parts.p[i] = .{ .w = p.w, .scales = p.scales, .biases = p.biases, .out = p.out, .n = int(p.n), .npad = int(p.npad), .sk = int(p.sk), .tiles = int(n_tiles), .first = int(clusters) };
        clusters += rows_t * ((n_tiles + c / p.sk - 1) / (c / p.sk));
    }
    var a: cuda.Args = .{};
    a.add(g.x);
    a.add(g.xs);
    a.add(parts);
    for ([_]usize{ g.m, g.k, if (g.m == 1) g.k else g.x_stride, rows_t, c }) |v| a.add(int(v));
    return .{
        .kernel = if (g.f32_out) t.f32 else t.bf16,
        .cfg = .{ .grid = .{ .x = u(clusters * c) }, .block = .{ .x = t.wm * t.wn * 32 }, .shared = t.smem(), .cluster = if (c > 1) .{ .x = u(c) } else null, .pdl = if (g.pdl < 0) gb10 else g.pdl > 0 },
        .args = a,
    };
}

/// qmm.cpp qmm_prefill(x, w, scales, biases, out, n, gs, f32, tile): x (m, k) bf16 with rows `x_stride` apart, scales
/// and biases (k / gs, npad) contiguous; only gs 64 at tile 0 is built into glm_qmm_prefill.
pub const Prefill = struct { x: u64, w: u64, scales: u64, biases: u64, out: u64, m: usize, k: usize, x_stride: usize, n: usize, npad: usize, gs: usize = 64, f32_out: bool, tile: i64 = 0 };

/// qmm.cpp qmm_group for one weight [words, scales, biases] from python's tensors (any type with ptr, shape and stride:
/// trace.T, calls.Ten); the scales' row stride is the padded width, the x row stride python's.
pub fn groupCall(chip: Chip, x: anytype, xs: anytype, w: anytype, out: anytype, n: usize, sk: usize, f32_out: bool, tile: i64) Error!Launch {
    const p = [_]Proj{.{ .w = w[0].ptr, .scales = w[1].ptr, .biases = w[2].ptr, .out = out.ptr, .n = n, .npad = w[1].stride[0], .sk = sk }};
    return groupLaunch(.{ .x = x.ptr, .xs = xs.ptr, .m = x.shape[0], .k = x.shape[1], .x_stride = x.stride[0], .parts = &p, .f32_out = f32_out, .tile = tile }, chip);
}

/// qmm.cpp qmm_prefill from python's tensors, groups of 64 (the padded width is the scales' second dimension).
pub fn prefillCall(x: anytype, w: anytype, out: anytype, n: usize, f32_out: bool, tile: i64) Error!Launch {
    return prefillLaunch(.{ .x = x.ptr, .w = w[0].ptr, .scales = w[1].ptr, .biases = w[2].ptr, .out = out.ptr, .m = x.shape[0], .k = x.shape[1], .x_stride = x.stride[0], .n = n, .npad = w[1].shape[1], .f32_out = f32_out, .tile = tile });
}

/// qmm_prefill_cuda's dispatch and launch<64, 128, 128, 2, 4, 3, F32>: 128-row tiles, column-tile groups kept near 12 MB of L2.
pub fn prefillLaunch(p: Prefill) Error!Launch {
    if ((p.gs != 32 and p.gs != 64) or p.tile < 0 or p.tile > 11 or p.m < 1 or p.k % p.gs != 0 or p.x_stride < p.k) return error.Invalid;
    if (p.x % 16 != 0 or (p.m > 1 and p.x_stride % 8 != 0) or p.npad != (p.n + 127) / 128 * 128) return error.Invalid;
    if (p.gs != 64 or p.tile != 0) return error.NotBuilt;
    const rows_t = (p.m + 127) / 128;
    const band = (12 << 20) / (128 * p.k * 2);
    var a: cuda.Args = .{};
    for ([_]u64{ p.x, p.w, p.scales, p.biases, p.out }) |v| a.add(v);
    for ([_]usize{ p.m, p.n, p.k, p.npad, if (p.m == 1) p.k else p.x_stride, @max(1, @min(rows_t, band)) }) |v| a.add(int(v));
    return .{
        .kernel = if (p.f32_out) .prefill_f32 else .prefill_bf16,
        .cfg = .{ .grid = .{ .x = u(rows_t * ((p.n + 127) / 128)) }, .block = .{ .x = 256 }, .shared = prefill_smem },
        .args = a,
    };
}

/// experts.Plan's scratch: members (a pair each), items [*, 3], counts [2], and the wide path's rank and hist.
pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64 };

/// experts.cu's limits: EMAX experts, PSMALL pairs for the one-block plan, PLAN_THREADS a block.
pub const plan_experts_max = 1024;
pub const plan_small = 1024;

pub const PlanLaunches = struct { n: usize = 0, l: [3]Launch = undefined };

/// experts.cpp plan(picks, pairs, experts, tile, ...): plan_kernel alone up to 1,024 pairs, else rank, offsets, scatter.
pub fn planLaunches(picks: u64, pairs: usize, experts: usize, tile: usize, p: Plan) Error!PlanLaunches {
    if ((tile != 16 and tile != 64) or experts > plan_experts_max) return error.Invalid;
    var out: PlanLaunches = .{};
    if (pairs <= plan_small) {
        var a: cuda.Args = .{};
        a.add(picks);
        for ([_]usize{ pairs, experts, tile }) |v| a.add(int(v));
        for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
        out.l[0] = .{ .kernel = .plan, .cfg = .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, .args = a };
        out.n = 1;
        return out;
    }
    const nblk = (pairs + 1023) / 1024;
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(int(pairs));
    a.add(int(experts));
    a.add(p.rank);
    a.add(p.hist);
    out.l[0] = .{ .kernel = .plan_rank, .cfg = .{ .grid = .{ .x = u(nblk) }, .block = .{ .x = 1024 } }, .args = a };
    var b: cuda.Args = .{};
    for ([_]usize{ nblk, experts, tile }) |v| b.add(int(v));
    for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
    out.l[1] = .{ .kernel = .plan_offsets, .cfg = .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, .args = b };
    var c: cuda.Args = .{};
    c.add(picks);
    c.add(int(pairs));
    c.add(int(experts));
    for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
    out.l[2] = .{ .kernel = .plan_scatter, .cfg = .{ .grid = .{ .x = u((pairs + 255) / 256) }, .block = .{ .x = 256 } }, .args = c };
    out.n = 3;
    return out;
}

/// experts.cpp run and prefill's arguments: x rows `x_stride` apart (a token's when slots > 0, else a pair's), the
/// weight blocks, kg groups and nb 32-column blocks, the plan, out rows of n columns, the SwiGLU limit, and the bound
/// (max_units for run, max_items for prefill).
pub const Experts = struct { gs: usize = 64, epi: u8, x: u64, x_stride: usize, slots: usize, w: u64, kg: usize, nb: usize, items: u64, counts: u64, members: u64, out: u64, n: usize, limit: f64, bound: usize };

fn expertArgs(e: Experts) Error!cuda.Args {
    if (e.n != e.nb * 32 or e.x_stride % 8 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(e.x);
    for ([_]usize{ e.x_stride, e.slots }) |v| a.add(int(v));
    a.add(e.w);
    for ([_]usize{ e.kg, e.nb }) |v| a.add(int(v));
    for ([_]u64{ e.items, e.counts, e.members, e.out }) |v| a.add(v);
    a.add(int(e.n));
    a.add(@as(f32, @floatCast(e.limit)));
    return a;
}

/// experts_run_cuda: expert_kernel<64, M, EPI, 2, 4> (gate and up with SwiGLU, or down to fp32) on 4 warps a block,
/// min(ceil(bound / 4), per_sm * sms) blocks; null when there is no block (python launches nothing).
pub fn runLaunch(e: Experts, per_sm: usize, sms: usize) Error!?Launch {
    const kernel: Kernel = if (e.gs == 64 and e.epi == 2) .gate_up else if (e.gs == 64 and e.epi == 0) .down else return error.NotBuilt;
    const a = try expertArgs(e);
    const grid = @min((e.bound + 3) / 4, @max(per_sm, 1) * sms);
    if (grid < 1) return null;
    return .{ .kernel = kernel, .cfg = .{ .grid = .{ .x = u(grid) }, .block = .{ .x = 128 } }, .args = a };
}

/// experts_prefill_cuda: prefill_kernel<64, M, EPI, 2, 2, 4> (gate and up with SwiGLU, or down to bf16), 8 warps on 64
/// pairs by 128 columns: bound * ceil(nb / 4) blocks; null when there is none.
pub fn prefillExpertsLaunch(e: Experts) Error!?Launch {
    const kernel: Kernel = if (e.gs == 64 and e.epi == 2) .pre_gate_up else if (e.gs == 64 and e.epi == 3) .pre_down else return error.NotBuilt;
    const a = try expertArgs(e);
    const grid = e.bound * ((e.nb + 3) / 4);
    if (grid < 1) return null;
    return .{ .kernel = kernel, .cfg = .{ .grid = .{ .x = u(grid) }, .block = .{ .x = 256 }, .shared = kernel.raised() }, .args = a };
}

/// The four images, as zig/build/glm_fwd_kernels.zig embeds them (glm_mm_images).
pub const Images = struct { qmm_group: []const u8, qmm_prefill: []const u8, experts: []const u8, experts_prefill: []const u8 };

/// python-0.6's recorded calls as these launches (the CPU fixture test and tf-glm-mm's GPU replay share it).
pub const calls = @import("cuda_kernels_mm_calls.zig");

/// The four images loaded on the current context: one function a kernel, raised limits set as the wrappers set them,
/// and the decode experts' resident blocks an SM (the wrapper's cudaOccupancyMaxActiveBlocksPerMultiprocessor).
pub const Mm = struct {
    mods: [4]cuda.Module,
    fns: [Kernel.count]cuda.Function,
    chip: Chip,
    per_sm: [2]usize, // gate_up, down

    pub fn load(ctx: *const cuda.Context, im: Images) !Mm {
        const bytes = [4][]const u8{ im.qmm_group, im.qmm_prefill, im.experts, im.experts_prefill };
        for (bytes) |b| if (b.len == 0) return error.NotFound; // a host build embeds no images
        var m: Mm = undefined;
        var loaded: usize = 0;
        errdefer for (m.mods[0..loaded]) |*x| x.unload();
        for (bytes, 0..) |b, i| {
            m.mods[i] = try cuda.Module.load(ctx.d, b);
            loaded += 1;
        }
        for (std.enums.values(Kernel)) |k| {
            m.fns[@intFromEnum(k)] = try m.mods[k.image()].function(k.symbol());
            if (k.raised() > 0) try m.fns[@intFromEnum(k)].allowDynamicShared(k.raised());
        }
        m.chip = .{ .major = @intCast(try ctx.attribute(.compute_capability_major)), .minor = @intCast(try ctx.attribute(.compute_capability_minor)), .sms = @intCast(try ctx.attribute(.multiprocessor_count)) };
        m.per_sm = .{ @max(1, try m.fns[@intFromEnum(Kernel.gate_up)].occupancy(128, 0)), @max(1, try m.fns[@intFromEnum(Kernel.down)].occupancy(128, 0)) };
        return m;
    }

    pub fn unload(m: *Mm) void {
        for (&m.mods) |*x| x.unload();
    }

    pub fn run(m: *const Mm, s: cuda.Stream, l: Launch) cuda.Error!void {
        var x = l;
        try cuda.launch.launch(m.fns[@intFromEnum(l.kernel)], x.cfg, s, &x.args);
    }

    /// qmm.cpp qmm_group.
    pub fn group(m: *const Mm, s: cuda.Stream, g: Group) !void {
        try m.run(s, try groupLaunch(g, m.chip));
    }

    /// qmm.cpp qmm_prefill.
    pub fn prefill(m: *const Mm, s: cuda.Stream, p: Prefill) !void {
        try m.run(s, try prefillLaunch(p));
    }

    /// experts.cpp plan.
    pub fn plan(m: *const Mm, s: cuda.Stream, picks: u64, pairs: usize, n_experts: usize, tile: usize, p: Plan) !void {
        const ls = try planLaunches(picks, pairs, n_experts, tile, p);
        for (ls.l[0..ls.n]) |l| try m.run(s, l);
    }

    /// experts.cpp run (decode): epilogue 2 gate and up, 0 down.
    pub fn experts(m: *const Mm, s: cuda.Stream, e: Experts) !void {
        const l = try runLaunch(e, m.per_sm[if (e.epi == 2) 0 else 1], m.chip.sms) orelse return;
        try m.run(s, l);
    }

    /// experts.cpp prefill (prompt): epilogue 2 gate and up, 3 down.
    pub fn expertsPrefill(m: *const Mm, s: cuda.Stream, e: Experts) !void {
        const l = try prefillExpertsLaunch(e) orelse return;
        try m.run(s, l);
    }
};

test "the raised limits follow the wrappers' templates" {
    _ = @import("cuda_kernels_mm_test.zig"); // python-0.6's launches for GLM's calls (fixtures_cuda_mm.json)
    try std.testing.expectEqual(@as(u32, 35328), group_smem);
    try std.testing.expectEqual(@as(u32, 22656), Kernel.group7_bf16.raised()); // LaneTile<64, 8, 128, 1, 4, 4>
    try std.testing.expectEqual(@as(u32, 26880), Kernel.group8_f32.raised()); // LaneTile<64, 16, 128, 1, 8, 4>
    try std.testing.expectEqual(@as(u32, 62976), prefill_smem);
    try std.testing.expectEqual(@as(u32, 38400), preSmem(1));
    try std.testing.expectEqual(@as(u32, 34816), preSmem(2));
    for (std.enums.values(Kernel)) |k| try std.testing.expect(std.mem.startsWith(u8, k.symbol(), "_ZN") and std.mem.indexOf(u8, k.symbol(), "tf_glm_") != null);
}

test "dispatch: tile 2 on a GB10 up to 16 rows; tiles 7 and 8 on other sm_12x chips up to 16 rows; wider windows are not built" {
    const p = [_]Proj{.{ .w = 0x1000, .scales = 0x2000, .biases = 0x3000, .out = 0x4000, .n = 12576, .npad = 12672, .sk = 4 }};
    const g: Group = .{ .x = 0x10000, .xs = 0x20000, .m = 8, .k = 4096, .x_stride = 4096, .parts = &p, .f32_out = false };
    const l = try groupLaunch(g, .{});
    try std.testing.expectEqual(Kernel.group_bf16, l.kernel);
    try std.testing.expectEqual(@as(u32, 197 * 4), l.cfg.grid.x);
    try std.testing.expectEqual(@as(u32, 4), l.cfg.cluster.?.x);
    try std.testing.expect(l.cfg.pdl);
    var wide = g;
    wide.m = 17;
    try std.testing.expectError(error.NotBuilt, groupLaunch(wide, .{}));
    // the RTX 5090 (12.0, 170 SMs): tile 7 to 8 rows (one 8-row tile, 99 column tiles of 128), no PDL by chip
    const rtx: Chip = .{ .major = 12, .minor = 0, .sms = 170 };
    const l7 = try groupLaunch(g, rtx);
    try std.testing.expectEqual(Kernel.group7_bf16, l7.kernel);
    try std.testing.expectEqual(@as(u32, 99 * 4), l7.cfg.grid.x);
    try std.testing.expectEqual(@as(u32, 128), l7.cfg.block.x);
    try std.testing.expectEqual(@as(u32, 22656), l7.cfg.shared);
    try std.testing.expectEqual(@as(u32, 4), l7.cfg.cluster.?.x);
    try std.testing.expect(!l7.cfg.pdl);
    var nine = g;
    nine.m = 9;
    nine.f32_out = true;
    const l8 = try groupLaunch(nine, rtx); // tile 8: rows 9-16, 8 warps
    try std.testing.expectEqual(Kernel.group8_f32, l8.kernel);
    try std.testing.expectEqual(@as(u32, 99 * 4), l8.cfg.grid.x);
    try std.testing.expectEqual(@as(u32, 256), l8.cfg.block.x);
    try std.testing.expectEqual(@as(u32, 26880), l8.cfg.shared);
    try std.testing.expectError(error.NotBuilt, groupLaunch(wide, rtx)); // tile 9: 17-32 rows by the kernel's pick
    wide.tile = 12; // qmm_tiles.group_tile's pick at 17-32 rows from 96 SMs
    try std.testing.expectError(error.NotBuilt, groupLaunch(wide, rtx));
    var forced = g;
    forced.tile = 7; // a forced tile is honoured on any chip, as python's dispatch does
    try std.testing.expectEqual(Kernel.group7_bf16, (try groupLaunch(forced, .{})).kernel);
    var odd = g;
    odd.x_stride = 4100;
    try std.testing.expectError(error.Invalid, groupLaunch(odd, .{}));
}
