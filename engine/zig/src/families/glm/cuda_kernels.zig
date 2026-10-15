//! GLM-5.3-Flash's CUDA extension kernels: the glm_* fatbins (copies of python-0.6's extension device code, SASS-equal,
//! zig/tests/cuda/copies.py) with each Python wrapper's launch: grid, block, stream and arguments.
//!
//! Sources: python-0.6's device code by Ash Hart (github.com/ashhart; chain, replays: 3283e59, b41a6b6, 34bae79) and
//! taussoe (github.com/taussoe; the three-kernel wide path prep/step/out: fb985b8); see each fatbin's copy header.
//!   glm_kda   families/glm5_next/cuda/kda.cu   kda_chain_cuda, kda_chain_wide_cuda, kda_replay_cuda,
//!                                              kda_replay_layers_cuda (tensorfold_glm_kda_v2; kda.py's chain picks one)

const std = @import("std");
const cuda = @import("cuda");

/// Key and value dims a head (kda.cu's DK = DV); a head's state is 128 x 128 fp32.
pub const dv = 128;
/// step_kernel's one instantiation: warps a block and rows staged a pass (kda_chain_wide_cuda's WARPS, TR).
pub const step_warps = 4;
pub const step_rows = 16;
/// kda.py's WIDE_ROWS: windows of this many rows or more take the three-kernel path (the same bits).
pub const wide_rows = 64;

/// Mangled names in the glm_kda image (cuobjdump -symbols of its sm_121 and sm_120 fatbins).
pub const sym = struct {
    pub const chain = "_ZN10tf_glm_kda12chain_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiPS0_PfS6_S5_S6_S6_";
    pub const replay = "_ZN10tf_glm_kda13replay_kernelEiPKfS1_PK13__nv_bfloat16S1_S1_iPf";
    pub const replay_layers = "_ZN10tf_glm_kda20replay_layers_kernelEiPKfmS1_PK13__nv_bfloat16S1_S1_mmiPf";
    pub const prep = "_ZN10tf_glm_kda11prep_kernelEiPK13__nv_bfloat16iiS2_iS2_S2_PKfS4_fPfS5_PS0_S5_S5_";
    pub const step = "_ZN10tf_glm_kda11step_kernelILi4ELi16EEEviPKfS2_S2_PK13__nv_bfloat16S2_S2_iPS3_Pf";
    pub const out = "_ZN10tf_glm_kda10out_kernelEiPK13__nv_bfloat16S2_iS2_fPS0_";
    /// Every kernel the image exports; tests resolve each one so a wrong name cannot hide behind an unused path.
    pub const all = [_][:0]const u8{ chain, replay, replay_layers, prep, step, out };
};

/// One layer's window as kda.chain takes it (strides in elements): p rows hold q | k | v (3 * heads * 128) with beta at
/// b_off, a and g are the bf16 decay and gate rows; 0 is an undefined tensor, as the wrapper's ptr<T>() gives.
pub const Chain = struct {
    heads: usize,
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv_state: u64,
    conv_w: u64,
    state_in: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    rows: usize,
    out: u64,
    state_out: u64,
    k_save: u64,
    v_save: u64,
    g_save: u64,
    b_save: u64,
};

/// kda.replay_layers' inputs: every layer's state after the first `rows` saved rows (strides in elements).
pub const ReplayLayers = struct {
    state_in: u64,
    state_stride: usize,
    k_save: u64,
    v_save: u64,
    g_save: u64,
    b_save: u64,
    kv_stride: usize,
    b_stride: usize,
    layers: usize,
    heads: usize,
    rows: usize,
    state_out: u64,
};

/// A kernel launch as the wrapper makes it: its configuration and its packed arguments.
pub const Launch = struct { cfg: cuda.Config, args: cuda.Args };

fn int(x: usize) c_int {
    return @intCast(x);
}

fn dims(x: usize, y: usize) cuda.Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y) };
}

/// chain_kernel<<<H, 1024>>>: the whole window, one block a head.
pub fn chainLaunch(c: Chain) Launch {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    a.add(c.p);
    a.add(int(c.p_stride));
    a.add(int(c.b_off));
    a.add(c.a);
    a.add(int(c.a_stride));
    a.add(c.g);
    a.add(int(c.g_stride));
    for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
    a.add(c.eps);
    a.add(c.lower);
    a.add(int(c.rows));
    for ([_]u64{ c.out, c.state_out, c.k_save, c.v_save, c.g_save, c.b_save }) |v| a.add(v);
    return .{ .cfg = .{ .grid = dims(c.heads, 1), .block = .{ .x = 1024 } }, .args = a };
}

/// prep_kernel<<<(rows, H), 512>>>: conv, decay, beta and the q/k norms of every row; q into q_tmp.
pub fn prepLaunch(c: Chain, q_tmp: u64) Launch {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    a.add(c.p);
    a.add(int(c.p_stride));
    a.add(int(c.b_off));
    a.add(c.a);
    a.add(int(c.a_stride));
    for ([_]u64{ c.conv_state, c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
    a.add(c.lower);
    for ([_]u64{ q_tmp, c.k_save, c.v_save, c.g_save, c.b_save }) |v| a.add(v);
    return .{ .cfg = .{ .grid = dims(c.rows, c.heads), .block = .{ .x = 512 } }, .args = a };
}

/// step_kernel<4, 16><<<(H, 128 / 4 / 4), 128>>>: the delta rule alone, row by row; the read-out into y_tmp.
pub fn stepLaunch(c: Chain, q_tmp: u64, y_tmp: u64) Launch {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    for ([_]u64{ c.state_in, q_tmp, c.k_save, c.v_save, c.g_save, c.b_save }) |v| a.add(v);
    a.add(int(c.rows));
    a.add(y_tmp);
    a.add(c.state_out);
    return .{ .cfg = .{ .grid = dims(c.heads, dv / 4 / step_warps), .block = .{ .x = step_warps * 32 } }, .args = a };
}

/// out_kernel<<<(rows, H), 128>>>: the gated RMSNorm of the read-out.
pub fn outLaunch(c: Chain, y_tmp: u64) Launch {
    var a: cuda.Args = .{};
    a.add(int(c.heads));
    a.add(y_tmp);
    a.add(c.g);
    a.add(int(c.g_stride));
    a.add(c.norm_w);
    a.add(c.eps);
    a.add(c.out);
    return .{ .cfg = .{ .grid = dims(c.rows, c.heads), .block = .{ .x = dv } }, .args = a };
}

/// replay_kernel<<<H, 1024>>>: one layer's state after the first `rows` saved rows of its window.
pub fn replayLaunch(heads: usize, state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, rows: usize, state_out: u64) Launch {
    var a: cuda.Args = .{};
    a.add(int(heads));
    for ([_]u64{ state_in, k_save, v_save, g_save, b_save }) |v| a.add(v);
    a.add(int(rows));
    a.add(state_out);
    return .{ .cfg = .{ .grid = dims(heads, 1), .block = .{ .x = 1024 } }, .args = a };
}

/// replay_layers_kernel<<<layers * H, 1024>>>: every layer at once, block (layer, head).
pub fn replayLayersLaunch(r: ReplayLayers) Launch {
    var a: cuda.Args = .{};
    a.add(int(r.heads));
    a.add(r.state_in);
    a.add(@as(usize, r.state_stride));
    for ([_]u64{ r.k_save, r.v_save, r.g_save, r.b_save }) |v| a.add(v);
    a.add(@as(usize, r.kv_stride));
    a.add(@as(usize, r.b_stride));
    a.add(int(r.rows));
    a.add(r.state_out);
    return .{ .cfg = .{ .grid = dims(r.layers * r.heads, 1), .block = .{ .x = 1024 } }, .args = a };
}

/// The glm_kda image loaded on the current context, one function a kernel.
pub const Kda = struct {
    module: cuda.Module,
    chain_fn: cuda.Function,
    replay_fn: cuda.Function,
    replay_layers_fn: cuda.Function,
    prep_fn: cuda.Function,
    step_fn: cuda.Function,
    out_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) cuda.Error!Kda {
        if (cuda.kernels.glm_kda.len == 0) return error.NotFound; // a host-only build embeds no images
        var m = try cuda.Module.load(d, cuda.kernels.glm_kda);
        errdefer m.unload();
        return .{
            .module = m,
            .chain_fn = try m.function(sym.chain),
            .replay_fn = try m.function(sym.replay),
            .replay_layers_fn = try m.function(sym.replay_layers),
            .prep_fn = try m.function(sym.prep),
            .step_fn = try m.function(sym.step),
            .out_fn = try m.function(sym.out),
        };
    }

    pub fn unload(k: *Kda) void {
        k.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, l: Launch) cuda.Error!void {
        var x = l;
        try cuda.launch.launch(f, x.cfg, s, &x.args);
    }

    /// kda.chain: windows under wide_rows rows in one kernel, longer ones in prep, step and out (q_tmp, y_tmp scratch).
    pub fn chain(k: Kda, s: cuda.Stream, c: Chain, q_tmp: u64, y_tmp: u64) cuda.Error!void {
        if (c.rows < wide_rows) return k.chainOne(s, c);
        return k.chainWide(s, c, q_tmp, y_tmp);
    }

    /// kda_chain_cuda (the wrapper's TORCH_CHECK: at least one row).
    pub fn chainOne(k: Kda, s: cuda.Stream, c: Chain) cuda.Error!void {
        if (c.rows < 1 or c.heads < 1) return error.Invalid;
        return go(k.chain_fn, s, chainLaunch(c));
    }

    /// kda_chain_wide_cuda: q_tmp holds rows * heads * 128 fp32, y_tmp as many bf16; same bits as chainOne.
    pub fn chainWide(k: Kda, s: cuda.Stream, c: Chain, q_tmp: u64, y_tmp: u64) cuda.Error!void {
        if (c.rows < 1 or c.heads < 1 or q_tmp == 0 or y_tmp == 0) return error.Invalid;
        try go(k.prep_fn, s, prepLaunch(c, q_tmp));
        try go(k.step_fn, s, stepLaunch(c, q_tmp, y_tmp));
        try go(k.out_fn, s, outLaunch(c, y_tmp));
    }

    /// kda_replay_cuda.
    pub fn replay(k: Kda, s: cuda.Stream, heads: usize, state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, rows: usize, state_out: u64) cuda.Error!void {
        return go(k.replay_fn, s, replayLaunch(heads, state_in, k_save, v_save, g_save, b_save, rows, state_out));
    }

    /// kda_replay_layers_cuda.
    pub fn replayLayers(k: Kda, s: cuda.Stream, r: ReplayLayers) cuda.Error!void {
        return go(k.replay_layers_fn, s, replayLayersLaunch(r));
    }
};

/// One layer's KDAScratch in bytes for `rows` rows of `heads` heads: out and v bf16, k and g fp32 a channel, beta fp32 a head.
pub fn scratchBytes(rows: usize, heads: usize) struct { out: usize, k: usize, v: usize, g: usize, b: usize } {
    const ch = rows * heads * dv;
    return .{ .out = 2 * ch, .k = 4 * ch, .v = 2 * ch, .g = 4 * ch, .b = 4 * rows * heads };
}

/// kda.py's _wide_scratch for `rows` rows: the normalized q (fp32) and the read-out (bf16), shared by every layer.
pub fn wideScratchBytes(rows: usize, heads: usize) struct { q: usize, y: usize } {
    return .{ .q = 4 * rows * heads * dv, .y = 2 * rows * heads * dv };
}

/// Each kernel's parameters as offset:size pairs, from EIATTR_KPARAM_INFO of the sm_121 and sm_120 cubins (cuobjdump -elf).
const kparams = struct {
    const chain = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 4 }, .{ 20, 4 }, .{ 24, 8 }, .{ 32, 4 }, .{ 40, 8 }, .{ 48, 4 }, .{ 56, 8 }, .{ 64, 8 }, .{ 72, 8 }, .{ 80, 8 }, .{ 88, 8 }, .{ 96, 8 }, .{ 104, 4 }, .{ 108, 4 }, .{ 112, 4 }, .{ 120, 8 }, .{ 128, 8 }, .{ 136, 8 }, .{ 144, 8 }, .{ 152, 8 }, .{ 160, 8 } };
    const prep = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 4 }, .{ 20, 4 }, .{ 24, 8 }, .{ 32, 4 }, .{ 40, 8 }, .{ 48, 8 }, .{ 56, 8 }, .{ 64, 8 }, .{ 72, 4 }, .{ 80, 8 }, .{ 88, 8 }, .{ 96, 8 }, .{ 104, 8 }, .{ 112, 8 } };
    const step = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 8 }, .{ 24, 8 }, .{ 32, 8 }, .{ 40, 8 }, .{ 48, 8 }, .{ 56, 4 }, .{ 64, 8 }, .{ 72, 8 } };
    const out = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 8 }, .{ 24, 4 }, .{ 32, 8 }, .{ 40, 4 }, .{ 48, 8 } };
    const replay = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 8 }, .{ 24, 8 }, .{ 32, 8 }, .{ 40, 8 }, .{ 48, 4 }, .{ 56, 8 } };
    const replay_layers = [_][2]u16{ .{ 0, 4 }, .{ 8, 8 }, .{ 16, 8 }, .{ 24, 8 }, .{ 32, 8 }, .{ 40, 8 }, .{ 48, 8 }, .{ 56, 8 }, .{ 64, 8 }, .{ 72, 4 }, .{ 80, 8 } };
};

fn expectLayout(want: []const [2]u16, l: Launch) !void {
    try std.testing.expectEqual(want.len, l.args.count);
    for (want, 0..) |p, i| {
        try std.testing.expectEqual(p[0], l.args.offsets[i]);
        try std.testing.expectEqual(p[1], l.args.sizes[i]);
    }
}

fn sample(heads: usize, rows: usize) Chain {
    return .{
        .heads = heads, .p = 0x1000, .p_stride = 3 * heads * dv + 2 * heads, .b_off = 3 * heads * dv, .a = 0x2000,
        .a_stride = heads * dv, .g = 0x3000, .g_stride = heads * dv, .conv_state = 0x4000, .conv_w = 0x5000,
        .state_in = 0x6000, .a_log = 0x7000, .dt_bias = 0x8000, .norm_w = 0x9000, .eps = 1e-5, .lower = -5.0,
        .rows = rows, .out = 0xa000, .state_out = 0xb000, .k_save = 0xc000, .v_save = 0xd000, .g_save = 0xe000,
        .b_save = 0xf000,
    };
}

test "kda arguments sit where the cubins' parameter tables put them" {
    const c = sample(64, 16);
    try expectLayout(&kparams.chain, chainLaunch(c));
    try expectLayout(&kparams.prep, prepLaunch(c, 0x10000));
    try expectLayout(&kparams.step, stepLaunch(c, 0x10000, 0x11000));
    try expectLayout(&kparams.out, outLaunch(c, 0x11000));
    try expectLayout(&kparams.replay, replayLaunch(64, 1, 2, 3, 4, 5, 16, 6));
    const r: ReplayLayers = .{ .state_in = 1, .state_stride = 64 * dv * dv, .k_save = 2, .v_save = 3, .g_save = 4, .b_save = 5, .kv_stride = 16 * 64 * dv, .b_stride = 16 * 64, .layers = 34, .heads = 64, .rows = 9, .state_out = 6 };
    try expectLayout(&kparams.replay_layers, replayLayersLaunch(r));
}

test "kda grids follow kda.cu's wrappers" {
    const c = sample(32, 80);
    const ch = chainLaunch(c).cfg;
    try std.testing.expectEqual(@as(u32, 32), ch.grid.x);
    try std.testing.expectEqual(@as(u32, 1024), ch.block.x);
    const p = prepLaunch(c, 0).cfg;
    try std.testing.expectEqual(@as(u32, 80), p.grid.x);
    try std.testing.expectEqual(@as(u32, 32), p.grid.y);
    try std.testing.expectEqual(@as(u32, 512), p.block.x);
    const st = stepLaunch(c, 0, 0).cfg;
    try std.testing.expectEqual(@as(u32, 32), st.grid.x);
    try std.testing.expectEqual(@as(u32, 8), st.grid.y);
    try std.testing.expectEqual(@as(u32, 128), st.block.x);
    const o = outLaunch(c, 0).cfg;
    try std.testing.expectEqual(@as(u32, 80), o.grid.x);
    try std.testing.expectEqual(@as(u32, 128), o.block.x);
    const r: ReplayLayers = .{ .state_in = 1, .state_stride = 0, .k_save = 2, .v_save = 3, .g_save = 4, .b_save = 5, .kv_stride = 0, .b_stride = 0, .layers = 34, .heads = 32, .rows = 3, .state_out = 6 };
    try std.testing.expectEqual(@as(u32, 34 * 32), replayLayersLaunch(r).cfg.grid.x);
    for ([_]cuda.Config{ ch, p, st, o }) |cfg| try cfg.validate();
}

test "kda values reach their slots" {
    const c = sample(64, 5);
    var l = chainLaunch(c);
    try std.testing.expectEqual(@as(?u64, 64), l.args.integer(0));
    try std.testing.expectEqual(@as(?u64, 3 * 64 * dv), l.args.integer(3));
    try std.testing.expectEqual(@as(?u64, @as(u32, @bitCast(@as(f32, -5.0)))), l.args.integer(15));
    try std.testing.expectEqual(@as(?u64, 5), l.args.integer(16));
    try std.testing.expectEqual(@as(?u64, 0xf000), l.args.integer(22));
    const b = scratchBytes(64, 32);
    try std.testing.expectEqual(@as(usize, 64 * 32 * 128 * 4), b.k);
    try std.testing.expectEqual(@as(usize, 64 * 32 * 4), b.b);
    try std.testing.expectEqual(@as(usize, 80 * 32 * 128 * 2), wideScratchBytes(80, 32).y);
}

test "kda symbols carry the copy's namespace" {
    for (sym.all) |name| try std.testing.expect(std.mem.startsWith(u8, name, "_ZN10tf_glm_kda"));
}
