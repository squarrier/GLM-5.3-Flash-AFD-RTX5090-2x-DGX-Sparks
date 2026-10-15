//! The EXL3 fatbins (zig/kernels/cuda/exl3/*.cu and glm_exl3.cu, copies of python-0.6's device code) as launches: each
//! kernel's mangled symbol, the image that holds it, and a launch's configuration and packed arguments. `Set` loads the
//! images on a context and launches by symbol.

const std = @import("std");
const cuda = @import("cuda");
const images = cuda.kernels;

/// EXL3 codebooks (ExLlamaV3): CB 0 3inst, 1 mcg, 2 mul1.
pub const Codebook = enum(u8) { @"3inst" = 0, mcg = 1, mul1 = 2 };

/// The fatbins, one a Python translation unit: experts.cu, experts_cb0-2.cu (grouped_kernel, dequant_kernel),
/// linear.cu, and GLM's exl3.cu; then exl3/l2d/'s experts.cu and experts_cb0-2.cu (Mia's 0128 applied); then exl3/x1/'s
/// experts.cu and experts_cb1.cu (X1: python 2.15's prompt switches, exl3_x1.patch); then exl3/x1s/grouped_cb1.cu (SEQ);
/// then exl3/mma/grouped_cb1.cu (qualification_8724e161: bertholomus's grouped_mma and work list); then exl3/k4/experts_prompt.cu
/// (K4: python 2.15's copy of MiaAI-Lab's EXL3 prompt kernel, qualification_e4ad04cd).
pub const Image = enum { exl3_experts, exl3_grouped_cb0, exl3_grouped_cb1, exl3_grouped_cb2, exl3_linear, glm_exl3, exl3_l2d_experts, exl3_l2d_grouped_cb0, exl3_l2d_grouped_cb1, exl3_l2d_grouped_cb2, exl3_x1_experts, exl3_x1_grouped_cb1, exl3_x1s_grouped_cb1, exl3_mma_cb1, exl3_k4 };
pub const images_all = [_]Image{ .exl3_experts, .exl3_grouped_cb0, .exl3_grouped_cb1, .exl3_grouped_cb2, .exl3_linear, .glm_exl3 };
/// 0128's images: loaded only when a switch asks for its kernels (cuda_exl3_experts.Sched).
pub const images_l2d = [_]Image{ .exl3_l2d_experts, .exl3_l2d_grouped_cb0, .exl3_l2d_grouped_cb1, .exl3_l2d_grouped_cb2 };
/// X1's images: loaded only with TF_GLM_EXL3_X1=1 (cuda_exl3_experts.Prompt).
pub const images_x1 = [_]Image{ .exl3_x1_experts, .exl3_x1_grouped_cb1 };
/// SEQ's image (qualification_b3fba1da): loaded only with TF_GLM_EXL3_SEQ=1 beside X1's (cuda_exl3_experts.Prompt.seq).
pub const images_x1s = [_]Image{.exl3_x1s_grouped_cb1};
/// qualification_8724e161's image: loaded only with TF_GLM_EXL3_MMA=1 beside X1's (cuda_exl3_experts.Prompt.mma).
pub const images_mma = [_]Image{.exl3_mma_cb1};
/// K4's image (qualification_e4ad04cd): loaded only with TF_GLM_HALVES_KERNEL=k4, beside X1's (cuda_exl3_k4.zig).
pub const images_k4 = [_]Image{.exl3_k4};
const n_images = @typeInfo(Image).@"enum".field_names.len;

/// A launch as the wrapper makes it; `max_dynamic`: the cudaFuncAttributeMaxDynamicSharedMemorySize it sets first.
pub const Launch = struct { image: Image, symbol: [:0]const u8, cfg: cuda.Config, args: cuda.Args, max_dynamic: ?u32 = null };

/// The launches of one wrapper call, in order (a routed layer makes at most seven, eight with qualification_3e1f1105's PGROUP; qualification_8724e161's MMA
/// path eight, nine with SEQ's gate/up and its order list).
pub const Launches = struct {
    items: [10]Launch = undefined,
    len: usize = 0,

    pub fn push(l: *Launches, x: Launch) void {
        std.debug.assert(l.len < l.items.len);
        l.items[l.len] = x;
        l.len += 1;
    }

    pub fn slice(l: *const Launches) []const Launch {
        return l.items[0..l.len];
    }
};

/// Mangled names as cuobjdump -symbols lists them for the sm_121 and sm_120 fatbins (zig/tests/cuda/glm/exl3_kernels.json).
pub const sym = struct {
    pub const group = "_ZN15tf_exl3_experts12group_kernelEPKiPiS2_S2_iiii";
    pub const rot_in_bf16 = "_ZN15tf_exl3_experts13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    pub const rot_in_f16 = "_ZN15tf_exl3_experts13rot_in_kernelI6__halfEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    pub const gateup_epilogue = "_ZN15tf_exl3_experts22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi";
    pub const down_epilogue = "_ZN15tf_exl3_experts20down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    pub const combine = "_ZN15tf_exl3_experts14combine_kernelEPKfS1_Pfii";
    pub const down_combine = "_ZN15tf_exl3_experts19down_combine_kernelEPKfPKiPK6__halfPfS1_S7_iiiii";
    /// qualification_3e1f1105 (exl3_pgroup.patch, bertholomus's dbe0773): the parallel grouping, in exl3/experts.cu and exl3/x1/experts.cu
    pub const group_count = "_ZN15tf_exl3_experts18group_count_kernelEPKiPii";
    pub const group_place = "_ZN15tf_exl3_experts18group_place_kernelEPKiS1_PiS2_S2_iiii";
    pub const linear_rot_in = "_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i";
    pub const glm_rot_in = "_ZN11tf_glm_exl313rot_in_kernelEPK13__nv_bfloat16iPKiPK6__halfS7_PS5_S8_ii";
    pub const glm_gateup_epilogue = "_ZN11tf_glm_exl322gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiif";
    pub const glm_down_epilogue = "_ZN11tf_glm_exl320down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    /// X1 (exl3/x1/experts.cu): prompt_order_kernel, and rot_in_kernel<T, ROWS> (ROWS: exl3x_rot_rows_cuda's)
    pub const x1_order = "_ZN15tf_exl3_experts19prompt_order_kernelEPKiS1_S1_Piiii";
    pub const x1_rot_rows_bf16 = "_ZN15tf_exl3_experts13rot_in_kernelI13__nv_bfloat16Lb1EEEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    pub const x1_rot_rows_f16 = "_ZN15tf_exl3_experts13rot_in_kernelI6__halfLb1EEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    pub const x1_rot_in_bf16 = "_ZN15tf_exl3_experts13rot_in_kernelI13__nv_bfloat16Lb0EEEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    pub const x1_rot_in_f16 = "_ZN15tf_exl3_experts13rot_in_kernelI6__halfLb0EEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    /// K4 (exl3/k4/experts_prompt.cu, qualification_e4ad04cd): the plan, the items' order, gate/up and down by pass (64, 128) and SHX
    pub const k4_plan = "_ZN10tf_exl3_k418prompt_plan_kernelEPKiS1_S1_iiiiPiS2_";
    pub const k4_order = "_ZN10tf_exl3_k419prompt_order_kernelEPKiS1_Piii";

    /// K4's prompt_gateup_kernel<PM, NW, SHX>: PM 64 (8 warps) or 128 (16), SHX the members' shared rows.
    pub fn k4Gateup(pm: u32, shx: bool) [:0]const u8 {
        return switch (pm) {
            64 => if (shx) k4_gateup_names[1] else k4_gateup_names[0],
            128 => if (shx) k4_gateup_names[3] else k4_gateup_names[2],
            else => "",
        };
    }

    /// K4's prompt_down_kernel<PM, NW>.
    pub fn k4Down(pm: u32) [:0]const u8 {
        return switch (pm) {
            64 => k4_down_names[0],
            128 => k4_down_names[1],
            else => "",
        };
    }

    /// X1's tf_exl3x::grouped_kernel<1, 8, 4, 1, 8, 8, NFIRST, ORDER, PA, XROW> by python's selector (8 4 2 1), 1..15.
    pub fn groupedX1(sel: u32) [:0]const u8 {
        return if (sel >= 1 and sel <= 15) grouped_x1_names[sel - 1] else "";
    }

    /// SEQ's tf_exl3x::grouped_seq_kernel<1, 8, 4, 1, 8, 8, NFIRST, ORDER, PA, XROW> (exl3/x1s/), X1's selector, 1..15.
    pub fn groupedX1Seq(sel: u32) [:0]const u8 {
        return if (sel >= 1 and sel <= 15) grouped_x1s_names[sel - 1] else "";
    }

    /// qualification_8724e161 (exl3/mma/, bertholomus's 90b68e0): work_list_kernel(counts, uids, ucount, maxu, work0, rows0, n0, work1,
    /// rows1, n1).
    pub const work_list = "_ZN15tf_exl3_experts16work_list_kernelEPKiS1_S1_iPiiiS2_ii";

    /// qualification_8724e161's tf_exl3x::grouped_mma_kernel<1, 8, 8, FOLD, 4, XROW> (exl3/mma/): (true, false), (true, true) and
    /// (false, false) are built; else "".
    pub fn groupedMma(fold: bool, xrow: bool) [:0]const u8 {
        if (xrow and !fold) return "";
        return grouped_mma_names[@as(usize, @intFromBool(fold)) + @intFromBool(xrow)];
    }

    /// tf_exl3x::grouped_kernel<CB, NT, W, PF, LO, HI>.
    pub fn grouped(cb: Codebook, nt: u32, w: u32, pf: u32, lo: u32, hi: u32) [:0]const u8 {
        return pick(&grouped_names, .{ @backingInt(cb), nt, w, pf, lo, hi });
    }

    /// 0128's tf_exl3x::grouped_kernel<CB, NT, W, PF, LO, HI> (its l2d argument last), in the exl3_l2d images.
    pub fn groupedL2d(cb: Codebook, nt: u32, w: u32, pf: u32, lo: u32, hi: u32) [:0]const u8 {
        return pick(&grouped_l2d_names, .{ @backingInt(cb), nt, w, pf, lo, hi });
    }

    /// tf_exl3x::dequant_kernel<CB, K2>.
    pub fn dequant(cb: Codebook, k2: u32) [:0]const u8 {
        return pick(&dequant_names, .{ @backingInt(cb), k2 });
    }

    /// tf_exl3_linear::linear_kernel<K2, CB, WK>.
    pub fn linear(k2: u32, cb: Codebook, wk: u32) [:0]const u8 {
        return pick(&linear_names, .{ k2, @backingInt(cb), wk });
    }

    /// tf_exl3_linear::unpack_kernel<K2, CB>.
    pub fn unpack(k2: u32, cb: Codebook) [:0]const u8 {
        return pick(&unpack_names, .{ k2, @backingInt(cb) });
    }

    /// tf_glm_exl3::grouped_kernel<NT, W>.
    pub fn glmGrouped(nt: u32, w: u32) [:0]const u8 {
        return pick(&glm_grouped_names, .{ nt, w });
    }
};

/// Every instance each launch site compiles, as exl3_copies.py reads them from python-0.6's host code.
pub const grouped_tiles = [_][3]u32{ .{ 8, 4, 1 }, .{ 8, 4, 2 }, .{ 4, 4, 2 } };
/// 0128's grouped_launch adds the (4, 4, 1) setting (exl3_l2d.patch).
pub const grouped_tiles_l2d = [_][3]u32{ .{ 8, 4, 1 }, .{ 8, 4, 2 }, .{ 4, 4, 1 }, .{ 4, 4, 2 } };
pub const grouped_ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } };
pub const linear_widths = [_][2]u32{ .{ 2, 0 }, .{ 4, 0 }, .{ 6, 0 }, .{ 8, 0 }, .{ 10, 0 }, .{ 12, 0 }, .{ 14, 0 }, .{ 16, 0 }, .{ 2, 1 }, .{ 4, 1 }, .{ 6, 1 }, .{ 8, 1 }, .{ 10, 1 }, .{ 12, 1 }, .{ 14, 1 }, .{ 16, 1 }, .{ 2, 2 }, .{ 4, 2 }, .{ 6, 2 }, .{ 8, 2 }, .{ 10, 2 }, .{ 12, 2 }, .{ 14, 2 }, .{ 16, 2 }, .{ 3, 2 }, .{ 5, 2 }, .{ 7, 2 } };
pub const linear_warps = [_]u32{ 2, 4, 8 };
pub const glm_tiles = [_][2]u32{ .{ 8, 4 }, .{ 4, 4 }, .{ 4, 8 }, .{ 2, 4 }, .{ 2, 8 } };

fn Named(comptime n: usize) type {
    return struct { key: [n]u32, name: [:0]const u8 };
}

fn pick(table: anytype, key: anytype) [:0]const u8 {
    for (table) |e| {
        var same = true;
        inline for (key, 0..) |v, i| same = same and e.key[i] == v;
        if (same) return e.name;
    }
    return "";
}

const grouped_names = blk: {
    var out: [3 * grouped_tiles.len * grouped_ranges.len]Named(6) = undefined;
    var n = 0;
    for (0..3) |cb| for (grouped_tiles) |t| for (grouped_ranges) |r| {
        out[n] = .{ .key = .{ cb, t[0], t[1], t[2], r[0], r[1] }, .name = std.fmt.comptimePrint("_ZN8tf_exl3x14grouped_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii", .{ cb, t[0], t[1], t[2], r[0], r[1] }) };
        n += 1;
    };
    break :blk out;
};

const grouped_l2d_names = blk: {
    var out: [3 * grouped_tiles_l2d.len * grouped_ranges.len]Named(6) = undefined;
    var n = 0;
    for (0..3) |cb| for (grouped_tiles_l2d) |t| for (grouped_ranges) |r| {
        out[n] = .{ .key = .{ cb, t[0], t[1], t[2], r[0], r[1] }, .name = std.fmt.comptimePrint("_ZN8tf_exl3x14grouped_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiiii", .{ cb, t[0], t[1], t[2], r[0], r[1] }) };
        n += 1;
    };
    break :blk out;
};

const dequant_names = blk: {
    var out: [3 * 15]Named(2) = undefined;
    for (0..3) |cb| for (2..17) |k2| {
        out[cb * 15 + k2 - 2] = .{ .key = .{ cb, k2 }, .name = std.fmt.comptimePrint("_ZN8tf_exl3x14dequant_kernelILi{d}ELi{d}EEEvPKjP6__halfii", .{ cb, k2 }) };
    };
    break :blk out;
};

const linear_names = blk: {
    var out: [linear_widths.len * linear_warps.len]Named(3) = undefined;
    for (linear_widths, 0..) |p, i| for (linear_warps, 0..) |w, j| {
        out[i * linear_warps.len + j] = .{ .key = .{ p[0], p[1], w }, .name = std.fmt.comptimePrint("_ZN14tf_exl3_linear13linear_kernelILi{d}ELi{d}ELi{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii", .{ p[0], p[1], w }) };
    };
    break :blk out;
};

const unpack_names = blk: {
    var out: [linear_widths.len]Named(2) = undefined;
    for (linear_widths, 0..) |p, i| out[i] = .{ .key = .{ p[0], p[1] }, .name = std.fmt.comptimePrint("_ZN14tf_exl3_linear13unpack_kernelILi{d}ELi{d}EEEvPKjP6__halfill", .{ p[0], p[1] }) };
    break :blk out;
};

const glm_grouped_names = blk: {
    var out: [glm_tiles.len]Named(2) = undefined;
    for (glm_tiles, 0..) |t, i| out[i] = .{ .key = .{ t[0], t[1] }, .name = std.fmt.comptimePrint("_ZN11tf_glm_exl314grouped_kernelILi{d}ELi{d}EEEvPK6__halfS3_PKjS5_PKiS7_S7_Pfiiiii", .{ t[0], t[1] }) };
    break :blk out;
};

/// prompt_grouped_launch<1>'s instances in selector order (exl3/x1/grouped_cb1.cu), the order slots their last argument.
const grouped_x1_names = blk: {
    var out: [15][:0]const u8 = undefined;
    for (1..16) |s| out[s - 1] = std.fmt.comptimePrint("_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8ELb{d}ELb{d}ELb{d}ELb{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiS7_", .{ s >> 3 & 1, s >> 2 & 1, s >> 1 & 1, s & 1 });
    break :blk out;
};

/// SEQ's instances (exl3/x1s/grouped_cb1.cu), X1's arguments: the same selector order as X1's.
const grouped_x1s_names = blk: {
    var out: [15][:0]const u8 = undefined;
    for (1..16) |s| out[s - 1] = std.fmt.comptimePrint("_ZN8tf_exl3x18grouped_seq_kernelILi1ELi8ELi4ELi1ELi8ELi8ELb{d}ELb{d}ELb{d}ELb{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiS7_", .{ s >> 3 & 1, s >> 2 & 1, s >> 1 & 1, s & 1 });
    break :blk out;
};

/// qualification_8724e161's instances (exl3/mma/grouped_cb1.cu), (FOLD, XROW) = (false, false), (true, false), (true, true).
const grouped_mma_names = blk: {
    var out: [3][:0]const u8 = undefined;
    for ([_][2]u32{ .{ 0, 0 }, .{ 1, 0 }, .{ 1, 1 } }, 0..) |fx, i| out[i] = std.fmt.comptimePrint("_ZN8tf_exl3x18grouped_mma_kernelILi1ELi8ELi8ELb{d}ELi4ELb{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_PfiiiiiiiiS7_", .{ fx[0], fx[1] });
    break :blk out;
};

/// K4's instances (exl3/k4/experts_prompt.cu's explicit instantiations): gate/up <64, 8, SHX 0/1>, <128, 16, 0/1>; down.
const k4_gateup_names = [4][:0]const u8{
    "_ZN10tf_exl3_k420prompt_gateup_kernelILi64ELi8ELb0EEEvPK6__halfS3_PKlS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_",
    "_ZN10tf_exl3_k420prompt_gateup_kernelILi64ELi8ELb1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_",
    "_ZN10tf_exl3_k420prompt_gateup_kernelILi128ELi16ELb0EEEvPK6__halfS3_PKlS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_",
    "_ZN10tf_exl3_k420prompt_gateup_kernelILi128ELi16ELb1EEEvPK6__halfS3_PKlS5_PKiS7_S7_S3_S3_S3_PS1_iiifiS7_",
};
const k4_down_names = [2][:0]const u8{
    "_ZN10tf_exl3_k418prompt_down_kernelILi64ELi8EEEvPK6__halfPKlPKiS7_S7_S3_PfiiiiS7_",
    "_ZN10tf_exl3_k418prompt_down_kernelILi128ELi16EEEvPK6__halfPKlPKiS7_S7_S3_PfiiiiS7_",
};

/// Every symbol the bindings can launch, by image (the tests check the cubins hold exactly these).
pub fn all(image: Image, out: *std.ArrayList([:0]const u8), gpa: std.mem.Allocator) !void {
    switch (image) {
        .exl3_experts, .exl3_l2d_experts => {
            for ([_][:0]const u8{ sym.group, sym.rot_in_bf16, sym.rot_in_f16, sym.gateup_epilogue, sym.down_epilogue, sym.combine, sym.down_combine }) |s| try out.append(gpa, s);
            if (image == .exl3_experts) for ([_][:0]const u8{ sym.group_count, sym.group_place }) |s| try out.append(gpa, s);
        },
        .exl3_grouped_cb0, .exl3_grouped_cb1, .exl3_grouped_cb2 => {
            const cb: u32 = @backingInt(image) - @backingInt(Image.exl3_grouped_cb0);
            for (grouped_names) |e| if (e.key[0] == cb) try out.append(gpa, e.name);
            for (dequant_names) |e| if (e.key[0] == cb) try out.append(gpa, e.name);
        },
        .exl3_l2d_grouped_cb0, .exl3_l2d_grouped_cb1, .exl3_l2d_grouped_cb2 => {
            const cb: u32 = @backingInt(image) - @backingInt(Image.exl3_l2d_grouped_cb0);
            for (grouped_l2d_names) |e| if (e.key[0] == cb) try out.append(gpa, e.name);
            for (dequant_names) |e| if (e.key[0] == cb) try out.append(gpa, e.name);
        },
        .exl3_linear => {
            try out.append(gpa, sym.linear_rot_in);
            for (linear_names) |e| try out.append(gpa, e.name);
            for (unpack_names) |e| try out.append(gpa, e.name);
        },
        .glm_exl3 => {
            for (glm_grouped_names) |e| try out.append(gpa, e.name);
            for ([_][:0]const u8{ sym.glm_rot_in, sym.glm_gateup_epilogue, sym.glm_down_epilogue }) |s| try out.append(gpa, s);
        },
        .exl3_x1_experts => for ([_][:0]const u8{ sym.group, sym.x1_rot_in_bf16, sym.x1_rot_rows_bf16, sym.x1_rot_in_f16, sym.x1_rot_rows_f16, sym.combine, sym.down_combine, sym.x1_order, sym.down_epilogue, sym.gateup_epilogue, sym.group_count, sym.group_place }) |s| try out.append(gpa, s),
        .exl3_x1_grouped_cb1 => for (grouped_x1_names) |s| try out.append(gpa, s),
        .exl3_x1s_grouped_cb1 => for (grouped_x1s_names) |s| try out.append(gpa, s),
        .exl3_mma_cb1 => {
            try out.append(gpa, sym.work_list);
            for (grouped_mma_names) |s| try out.append(gpa, s);
        },
        .exl3_k4 => {
            for ([_][:0]const u8{ sym.k4_plan, sym.k4_order }) |s| try out.append(gpa, s);
            for (k4_gateup_names) |s| try out.append(gpa, s);
            for (k4_down_names) |s| try out.append(gpa, s);
        },
    }
}

/// The image's bytes embedded at build time (empty in a host-only build).
pub fn bytes(image: Image) []const u8 {
    return switch (image) {
        inline else => |t| @field(images, @tagName(t)),
    };
}

/// The six images loaded on the current context (and 0128's four on request); functions are resolved on first launch.
pub const Set = struct {
    modules: [n_images]?cuda.Module = @splat(null),

    pub fn load(d: *const cuda.Driver) cuda.Error!Set {
        if (!images.available) return error.NotFound; // a host-only build embeds no images
        var s: Set = .{};
        errdefer s.unload();
        inline for (images_all) |img| s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        return s;
    }

    /// 0128's four images beside the six (cuda_exl3_experts.Sched.on()).
    pub fn loadL2d(s: *Set, d: *const cuda.Driver) cuda.Error!void {
        inline for (images_l2d) |img| if (s.modules[@backingInt(img)] == null) {
            s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        };
    }

    /// X1's two images beside the six (TF_GLM_EXL3_X1=1: cuda_exl3_experts.Prompt.on()).
    pub fn loadX1(s: *Set, d: *const cuda.Driver) cuda.Error!void {
        inline for (images_x1) |img| if (s.modules[@backingInt(img)] == null) {
            s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        };
    }

    /// SEQ's image beside X1's (TF_GLM_EXL3_SEQ=1: cuda_exl3_experts.Prompt.seq).
    pub fn loadX1s(s: *Set, d: *const cuda.Driver) cuda.Error!void {
        inline for (images_x1s) |img| if (s.modules[@backingInt(img)] == null) {
            s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        };
    }

    /// qualification_8724e161's image beside X1's (TF_GLM_EXL3_MMA=1: cuda_exl3_experts.Prompt.mma).
    pub fn loadMma(s: *Set, d: *const cuda.Driver) cuda.Error!void {
        inline for (images_mma) |img| if (s.modules[@backingInt(img)] == null) {
            s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        };
    }

    /// K4's image beside X1's (TF_GLM_HALVES_KERNEL=k4: cuda_exl3_k4.zig).
    pub fn loadK4(s: *Set, d: *const cuda.Driver) cuda.Error!void {
        inline for (images_k4) |img| if (s.modules[@backingInt(img)] == null) {
            s.modules[@backingInt(img)] = try cuda.Module.load(d, bytes(img));
        };
    }

    pub fn unload(s: *Set) void {
        for (&s.modules) |*m| if (m.*) |*x| {
            x.unload();
            m.* = null;
        };
    }

    pub fn function(s: Set, l: Launch) cuda.Error!cuda.Function {
        const m = s.modules[@backingInt(l.image)] orelse return error.NotFound; // 0128's images when not loaded
        return m.function(l.symbol);
    }

    /// Each launch on `stream` in order, raising the dynamic shared-memory limit first where the wrapper does.
    pub fn run(s: Set, stream: cuda.Stream, ls: *const Launches) cuda.Error!void {
        for (ls.slice()) |l| {
            const f = try s.function(l);
            if (l.max_dynamic) |b| try f.setAttribute(.max_dynamic_shared_size_bytes, @intCast(b));
            var x = l;
            try cuda.launch.launch(f, x.cfg, stream, &x.args);
        }
    }
};

test "symbol tables cover every instance and carry their namespaces" {
    try std.testing.expectEqual(@as(usize, 27), grouped_names.len);
    try std.testing.expectEqual(@as(usize, 36), grouped_l2d_names.len);
    try std.testing.expectEqual(@as(usize, 81), linear_names.len);
    for (grouped_names) |e| try std.testing.expect(std.mem.startsWith(u8, e.name, "_ZN8tf_exl3x14grouped_kernel"));
    try std.testing.expectEqualStrings("_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii", sym.grouped(.mcg, 8, 4, 1, 8, 8));
    try std.testing.expectEqualStrings("_ZN8tf_exl3x14grouped_kernelILi1ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiiii", sym.groupedL2d(.mcg, 8, 4, 1, 8, 8));
    try std.testing.expectEqualStrings("", sym.grouped(.mcg, 4, 4, 1, 8, 8));
    try std.testing.expect(sym.groupedL2d(.mcg, 4, 4, 1, 8, 8).len > 0);
    try std.testing.expectEqualStrings("", sym.grouped(.mcg, 2, 8, 1, 8, 8));
    try std.testing.expectEqualStrings("_ZN11tf_glm_exl314grouped_kernelILi8ELi4EEEvPK6__halfS3_PKjS5_PKiS7_S7_Pfiiiii", sym.glmGrouped(8, 4));
}
