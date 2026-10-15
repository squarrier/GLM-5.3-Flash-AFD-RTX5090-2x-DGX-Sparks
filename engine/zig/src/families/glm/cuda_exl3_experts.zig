//! TensorFold's EXL3 routed experts on CUDA (python-0.6 cuda/exl3/experts.py and experts.cu): any codebook, a width per
//! expert, one grouped launch a projection. Each function is a C++ wrapper's launch (grid, block, shared bytes, arguments)
//! for the exl3_experts and exl3_grouped_cb* fatbins (SASS-equal copies, zig/tests/cuda/exl3_copies.py); `routed` is the
//! Python wrapper's sequence. Format and kernels after ExLlamaV3 (turboderp, MIT); the module by Ash Hart and the
//! TensorFold contributors.

const std = @import("std");
const cuda = @import("cuda");
const k = @import("cuda_exl3_kernels.zig");
/// qualification_8724e161: bertholomus's grouped_mma on X1's prompt windows (TF_GLM_EXL3_MMA=1, exl3/mma/)
pub const mma = @import("cuda_exl3_mma.zig");

const Launch = k.Launch;
const Launches = k.Launches;

pub const Codebook = k.Codebook;

/// SwiGLU's arithmetic in the epilogue: GLM's bf16 roundings (0) or fp32 (1).
pub const Act = enum(c_int) { bf16 = 0, f32 = 1 };

/// (n tiles a block, warps, K splits, tiles in flight).
pub const Tile = struct { nt: u32, warps: u32, sk: u32, pf: u32 };

/// GLM's settings, whose arithmetic order the module keeps bit for bit (experts.py GLM_GATEUP, GLM_DOWN).
pub const glm_gateup: Tile = .{ .nt = 8, .warps = 4, .sk = 4, .pf = 1 };
pub const glm_down: Tile = .{ .nt = 8, .warps = 4, .sk = 1, .pf = 1 };

/// Mia's 0128 (exl3/l2d/'s kernels, the same bits): `l2d` k steps of L2 prefetch ahead, `pdl` gate/up, its epilogue and
/// down as programmatic dependent launches; `x` takes 0128's kernels at l2d 0 without PDL too (a bench arm). Off: today's.
pub const Sched = struct {
    l2d: u32 = 0,
    pdl: bool = false,
    x: bool = false,

    pub fn on(s: Sched) bool {
        return s.x or s.l2d > 0 or s.pdl;
    }

    /// TF_GLM_EXL3_L2D (0..64, default 0) and TF_GLM_EXL3_PDL (0 or 1, default 0); anything else is refused.
    pub fn fromEnv() error{BadExl3Sched}!Sched {
        return parse(env("TF_GLM_EXL3_L2D"), env("TF_GLM_EXL3_PDL"));
    }

    pub fn parse(l2d: ?[]const u8, pdl: ?[]const u8) error{BadExl3Sched}!Sched {
        var s: Sched = .{};
        if (l2d) |v| if (v.len > 0) {
            s.l2d = std.fmt.parseInt(u32, v, 10) catch return error.BadExl3Sched;
            if (s.l2d > 64) return error.BadExl3Sched;
        };
        if (pdl) |v| if (v.len > 0) {
            if (!std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "1")) return error.BadExl3Sched;
            s.pdl = v[0] == '1';
        };
        return s;
    }

    fn env(name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    }
};

/// experts.py default_config: the first candidate whose splits divide the K -> N projection (the shape's alone).
pub fn defaultConfig(K: u32, N: u32, gateup: bool) error{NoTile}!Tile {
    const cands = [_]Tile{ if (gateup) glm_gateup else glm_down, .{ .nt = 8, .warps = 4, .sk = 2, .pf = 1 }, .{ .nt = 8, .warps = 4, .sk = 1, .pf = 1 }, .{ .nt = 4, .warps = 4, .sk = 2, .pf = 2 }, .{ .nt = 4, .warps = 4, .sk = 1, .pf = 2 } };
    for (cands) |c| if (K % (16 * c.sk * c.warps) == 0 and N % (16 * c.nt) == 0) return c;
    return error.NoTile;
}

/// One layer's routed experts as experts.prepare builds them (Exl3RoutedExperts): device addresses and widths.
pub const Layer = struct {
    gate_ptr: u64, // int64 [E]: each expert's gate trellis
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]: half-bits a value
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    count: u32, // E
    dims: u32, // D
    width: u32, // I, this rank's share of the expert width
    cb: Codebook,
    k2_gu: [2]u32, // (min, max) over gate and up
    k2_d: [2]u32,
};

/// experts.Scratch: buffers for up to `rows` rows of `slots` slots, and the tile settings fixed at its making.
pub const Scratch = struct {
    xg: u64, // fp16 [P, D]
    xu: u64,
    xd: u64, // fp16 [P, I]
    z: u64, // fp32: max(2 sk_gu I, sk_d D) * P
    y: u64, // fp32 [P, D]
    ids: u64, // int32 [min(P, E)]
    count: u64, // int32 [1]
    members: u64, // int32 [min(P, E) * rows]
    rows: u32,
    slots: u32,
    experts: u32,
    cfg_gu: Tile,
    cfg_d: Tile,
    /// qualification_3e1f1105 (TF_GLM_EXL3_PGROUP=1): int32 [E], group_count's output where the role placed it; 0: none, `group` alone
    counts: u64 = 0,
    /// qualification_8724e161 (TF_GLM_EXL3_MMA=1): int32 [mma.workInts(rows, slots, E)], the work list where the role placed it; 0: none,
    /// X1's grouped launches
    work: u64 = 0,

    /// Byte sizes of each buffer, as Scratch.__init__ allocates them.
    pub fn bytes(rows: u32, slots: u32, ex: Layer, cfg_gu: Tile, cfg_d: Tile) struct { xg: usize, xd: usize, z: usize, y: usize, ids: usize, members: usize } {
        const P: usize = @as(usize, rows) * slots;
        const maxu = @min(P, ex.count);
        return .{ .xg = 2 * P * ex.dims, .xd = 2 * P * ex.width, .z = 4 * P * @max(2 * cfg_gu.sk * ex.width, cfg_d.sk * ex.dims), .y = 4 * P * ex.dims, .ids = 4 * maxu, .members = 4 * maxu * rows };
    }
};

/// cudaFuncGetAttributes of group_kernel and the device's opt-in limit, as exl3x_group_cuda reads them: the cubin's static
/// shared bytes (sharedSizeBytes), the dynamic bytes allowed so far (48 KiB less the static until raised) and
/// sharedMemPerBlockOptin.
pub const Shared = struct {
    static: u32 = group_static,
    max_dynamic: u32 = 48 * 1024 - group_static,
    optin: u32,
};

/// group_kernel's static shared bytes in the sm_121 and sm_120 cubins (exl3_kernels.json); the launch check itself
/// counts only GROUP_THREADS / 32 ints of it, as the wrapper's constexpr does.
pub const group_static = 1152;
const group_counted = 1024 / 32 * 4;

fn int(x: anytype) c_int {
    return @intCast(x);
}

/// exl3x_group_cuda: group_kernel<<<1, 1024, R * slots * 4>>>, with the opt-in shared memory past 48 KiB.
pub fn group(out: *Launches, pick: u64, uids: u64, ucount: u64, members: u64, R: u32, slots: u32, E: u32, maxm: u32, sh: Shared) error{Invalid}!void {
    if (E > 1024 * 4 or slots > 32) return error.Invalid;
    const smem: u32 = R * slots * 4;
    var attr: ?u32 = null;
    if (smem + group_counted > 48 * 1024) {
        const limit = sh.optin - sh.static;
        if (smem > limit) return error.Invalid;
        if (smem > sh.max_dynamic) attr = limit;
    }
    var a: cuda.Args = .{};
    for ([_]u64{ pick, uids, ucount, members }) |v| a.add(v);
    for ([_]u32{ R, slots, E, maxm }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.group, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 }, .shared = smem }, .args = a, .max_dynamic = attr });
}

/// group_count_kernel's and group_place_kernel's block (dbe0773's PLACE_THREADS).
pub const place_threads = 256;

/// bertholomus's parallel grouping (qualification_3e1f1105, TF_GLM_EXL3_PGROUP; glm-dsa-tp4 dbe0773's exl3x_group_count_cuda and
/// exl3x_group_place_cuda): group_count_kernel<<<E, 256>>>(pick, counts, R slots), then group_place_kernel<<<E, 256>>>
/// (pick, counts, uids, ucount, members, R slots, slots, E, maxm). `group`'s lists byte for byte (ascending ids, each
/// one's members in pick order as row * 32 + slot, -1 to maxm, ucount), at any R: nothing staged in shared memory.
pub fn groupPar(out: *Launches, pick: u64, counts: u64, uids: u64, ucount: u64, members: u64, R: u32, slots: u32, E: u32, maxm: u32) error{Invalid}!void {
    if (slots > 32 or E == 0 or counts == 0 or @as(u64, R) * slots > std.math.maxInt(c_int)) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(pick);
    a.add(counts);
    a.add(int(R * slots));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.group_count, .cfg = .{ .grid = .{ .x = E }, .block = .{ .x = place_threads } }, .args = a });
    var b: cuda.Args = .{};
    for ([_]u64{ pick, counts, uids, ucount, members }) |v| b.add(v);
    for ([_]u32{ R * slots, slots, E, maxm }) |v| b.add(int(v));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.group_place, .cfg = .{ .grid = .{ .x = E }, .block = .{ .x = place_threads } }, .args = b });
}

/// A window's grouping: `groupPar` where the scratch carries its counts (the role's TF_GLM_EXL3_PGROUP=1) and the
/// window has `prompt_rows` rows or more, else `group`; both write the same uids, ucount and members (maxm = R).
fn grouping(out: *Launches, pick: u64, s: Scratch, R: u32, E: u32, sh: Shared) error{Invalid}!void {
    if (s.counts != 0 and R >= prompt_rows) return groupPar(out, pick, s.counts, s.ids, s.count, s.members, R, s.slots, E, R);
    return group(out, pick, s.ids, s.count, s.members, R, s.slots, E, R, sh);
}

/// exl3x_rot_in_cuda: rot_in_kernel<bf16 or fp16><<<(rows * slots, K / 128, 2), 32>>>.
pub fn rotIn(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: u32, K: u32, slots: u32, E: u32) error{Invalid}!void {
    if (K % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    for ([_]u64{ pick, suh0, suh1, out0, out1 }) |v| a.add(v);
    for ([_]u32{ K, slots, E }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_experts, .symbol = if (x_bf16) k.sym.rot_in_bf16 else k.sym.rot_in_f16, .cfg = .{ .grid = .{ .x = rows * slots, .y = K / 128, .z = 2 }, .block = .{ .x = 32 } }, .args = a });
}

/// What exl3x_grouped_cuda reads beside its tensors (GroupedArgs).
pub const Grouped = struct {
    x0: u64,
    x1: u64,
    tp0: u64,
    tp1: u64,
    k2_0: u64,
    k2_1: u64,
    uids: u64,
    ucount: u64,
    members: u64,
    z: u64,
    mats: u32,
    K: u32,
    N: u32,
    P: u32,
    SK: u32,
    slots: u32,
    maxm: u32, // members.size(1)
    nexp_max: u32, // uids.size(0)
    cb: Codebook,
    nt: u32,
    warps: u32,
    pf: u32,
    lo: u32,
    hi: u32,
    sched: Sched = .{},
};

/// grouped_launch<CB>'s K2 range for an instance: 8..8 (4 bits), 2..10 or 2..16 (TF_RANGES).
pub fn k2Range(lo: u32, hi: u32) [2]u32 {
    if (lo == 8 and hi == 8) return .{ 8, 8 };
    if (lo >= 2 and hi <= 10) return .{ 2, 10 };
    return .{ 2, 16 };
}

/// exl3x_grouped_cuda: grouped_kernel<CB, NT, W, PF, LO, HI><<<(experts, N / (16 nt), mats SK MT), W * 32>>>; with
/// g.sched on, 0128's: the grid (N / (16 nt), mats SK MT, experts), l2d last, launch_ex's PDL attribute when pdl.
pub fn grouped(out: *Launches, g: Grouped) error{Invalid}!void {
    if (g.K % (16 * g.SK * g.warps) != 0 or g.N % (16 * g.nt) != 0) return error.Invalid;
    const x = g.sched.on();
    const tile_ok = (g.nt == 8 and g.warps == 4 and (g.pf == 1 or g.pf == 2)) or (g.nt == 4 and g.warps == 4 and (g.pf == 2 or (x and g.pf == 1)));
    if (!tile_ok) return error.Invalid;
    const range = k2Range(g.lo, g.hi);
    const MT = (g.maxm + 15) / 16;
    var a: cuda.Args = .{};
    for ([_]u64{ g.x0, g.x1, g.tp0, g.tp1, g.k2_0, g.k2_1, g.uids, g.ucount, g.members, g.z }) |v| a.add(v);
    for ([_]u32{ g.K, g.N, g.P, g.SK, g.maxm, g.slots }) |v| a.add(int(v));
    if (x) {
        a.add(int(g.sched.l2d));
        const image: k.Image = switch (g.cb) {
            .@"3inst" => .exl3_l2d_grouped_cb0,
            .mcg => .exl3_l2d_grouped_cb1,
            .mul1 => .exl3_l2d_grouped_cb2,
        };
        return out.push(.{ .image = image, .symbol = k.sym.groupedL2d(g.cb, g.nt, g.warps, g.pf, range[0], range[1]), .cfg = .{ .grid = .{ .x = g.N / (16 * g.nt), .y = g.mats * g.SK * MT, .z = g.nexp_max }, .block = .{ .x = g.warps * 32 }, .pdl = g.sched.pdl }, .args = a });
    }
    const image: k.Image = switch (g.cb) {
        .@"3inst" => .exl3_grouped_cb0,
        .mcg => .exl3_grouped_cb1,
        .mul1 => .exl3_grouped_cb2,
    };
    out.push(.{ .image = image, .symbol = k.sym.grouped(g.cb, g.nt, g.warps, g.pf, range[0], range[1]), .cfg = .{ .grid = .{ .x = g.nexp_max, .y = g.N / (16 * g.nt), .z = g.mats * g.SK * MT }, .block = .{ .x = g.warps * 32 } }, .args = a });
}

/// exl3x_gateup_epilogue_cuda: <<<(rows * slots, N / 128), 32>>>.
pub fn gateupEpilogue(out: *Launches, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: u32, P: u32, N: u32, SK: u32, slots: u32, E: u32, limit: f32, act: Act) error{Invalid}!void {
    return gateupEpilogueSched(out, z, pick, svh_g, svh_u, suh_d, xd, rows, P, N, SK, slots, E, limit, act, .{});
}

/// gateupEpilogue with `s` on: 0128's kernel (exl3_l2d_experts), and launch_ex's PDL attribute when s.pdl.
pub fn gateupEpilogueSched(out: *Launches, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: u32, P: u32, N: u32, SK: u32, slots: u32, E: u32, limit: f32, act: Act, s: Sched) error{Invalid}!void {
    if (N % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ z, pick, svh_g, svh_u, suh_d, xd }) |v| a.add(v);
    for ([_]u32{ P, N, SK, E }) |v| a.add(int(v));
    a.add(limit);
    a.add(@backingInt(act));
    out.push(.{ .image = if (s.on()) .exl3_l2d_experts else .exl3_experts, .symbol = k.sym.gateup_epilogue, .cfg = .{ .grid = .{ .x = rows * slots, .y = N / 128 }, .block = .{ .x = 32 }, .pdl = s.on() and s.pdl }, .args = a });
}

/// exl3x_down_epilogue_cuda: <<<(rows * slots, D / 128), 32>>>.
pub fn downEpilogue(out: *Launches, z: u64, pick: u64, svh_d: u64, y: u64, rows: u32, P: u32, D: u32, SK: u32, slots: u32, E: u32) error{Invalid}!void {
    if (D % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ z, pick, svh_d, y }) |v| a.add(v);
    for ([_]u32{ P, D, SK, E }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.down_epilogue, .cfg = .{ .grid = .{ .x = rows * slots, .y = D / 128 }, .block = .{ .x = 32 } }, .args = a });
}

/// exl3x_combine_cuda: <<<(rows, ceil(D / 256)), 256>>>.
pub fn combine(out: *Launches, y: u64, wts: u64, o: u64, rows: u32, D: u32, slots: u32) void {
    var a: cuda.Args = .{};
    for ([_]u64{ y, wts, o }) |v| a.add(v);
    a.add(int(D));
    a.add(int(slots));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.combine, .cfg = .{ .grid = .{ .x = rows, .y = (D + 255) / 256 }, .block = .{ .x = 256 } }, .args = a });
}

/// exl3x_down_combine_cuda: <<<(rows, D / 128), 32 * slots>>>.
pub fn downCombine(out: *Launches, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, o: u64, rows: u32, P: u32, D: u32, SK: u32, slots: u32, E: u32) error{Invalid}!void {
    if (slots > 32 or D % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ z, pick, svh_d, y, wts, o }) |v| a.add(v);
    for ([_]u32{ P, D, SK, E, slots }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_experts, .symbol = k.sym.down_combine, .cfg = .{ .grid = .{ .x = rows, .y = D / 128 }, .block = .{ .x = 32 * slots } }, .args = a });
}

/// exl3x_dequant_cuda: dequant_kernel<CB, K2><<<(K / 16, N / 16), 32>>> (tests; W_q [K, N] fp16 of one matrix).
pub fn dequant(out: *Launches, t: u64, o: u64, K: u32, N: u32, k2: u32, cb: Codebook) error{Invalid}!void {
    if (k2 < 2 or k2 > 16) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(t);
    a.add(o);
    a.add(int(K));
    a.add(int(N));
    const image: k.Image = switch (cb) {
        .@"3inst" => .exl3_grouped_cb0,
        .mcg => .exl3_grouped_cb1,
        .mul1 => .exl3_grouped_cb2,
    };
    out.push(.{ .image = image, .symbol = k.sym.dequant(cb, k2), .cfg = .{ .grid = .{ .x = K / 16, .y = N / 16 }, .block = .{ .x = 32 } }, .args = a });
}

/// experts.routed for R rows: group, rotate, gate+up, epilogue, down, then the down epilogue (wts null) or the fused
/// down epilogue and combine into `o`. No host sync; picks >= E are skipped (the caller's shared-expert slot).
pub fn routed(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, wts: ?u64, ex: Layer, s: Scratch, o: u64, R: u32, limit: f32, act: Act, do_group: bool, sh: Shared) error{Invalid}!void {
    return routedSched(out, x, x_stride, x_bf16, pick, wts, ex, s, o, R, limit, act, do_group, sh, .{});
}

/// `routed` with 0128's switches: gate+up, its epilogue and down through exl3/l2d/'s kernels when `sd` is on.
pub fn routedSched(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, wts: ?u64, ex: Layer, s: Scratch, o: u64, R: u32, limit: f32, act: Act, do_group: bool, sh: Shared, sd: Sched) error{Invalid}!void {
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const slots = s.slots;
    const P = R * slots;
    if (R > s.rows) return error.Invalid;
    const maxu = @min(R * slots, s.experts); // Scratch.window(R): ids[:maxu], members [maxu, R]
    if (do_group) try grouping(out, pick, s, R, E, sh);
    try rotIn(out, x, x_stride, x_bf16, pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E);
    var g: Grouped = .{ .x0 = s.xg, .x1 = s.xu, .tp0 = ex.gate_ptr, .tp1 = ex.up_ptr, .k2_0 = ex.gate_k2, .k2_1 = ex.up_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 2, .K = D, .N = I, .P = P, .SK = s.cfg_gu.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_gu.nt, .warps = s.cfg_gu.warps, .pf = s.cfg_gu.pf, .lo = ex.k2_gu[0], .hi = ex.k2_gu[1], .sched = sd };
    try grouped(out, g);
    try gateupEpilogueSched(out, s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, s.cfg_gu.sk, slots, E, limit, act, sd);
    g = .{ .x0 = s.xd, .x1 = s.xd, .tp0 = ex.down_ptr, .tp1 = ex.down_ptr, .k2_0 = ex.down_k2, .k2_1 = ex.down_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 1, .K = I, .N = D, .P = P, .SK = s.cfg_d.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_d.nt, .warps = s.cfg_d.warps, .pf = s.cfg_d.pf, .lo = ex.k2_d[0], .hi = ex.k2_d[1], .sched = sd };
    try grouped(out, g);
    if (wts) |w| return downCombine(out, s.z, pick, ex.svh_d, s.y, w, o, R, P, D, s.cfg_d.sk, slots, E);
    try downEpilogue(out, s.z, pick, ex.svh_d, s.y, R, P, D, s.cfg_d.sk, slots, E);
}

/// X1, python 2.15's prompt switches (cuda/exl3/experts.py routed's nfirst, prefetch, order, rowrot; IMP2's adaptation of
/// Mia's 0001, 0020 and 0009 onto the universal kernel): which block runs which item and where it loads from, never a
/// bit. GLM's windows of `prompt_rows` rows and more take them (exl3_generic.switches). Off: today's launches.
pub const Prompt = struct {
    nfirst: bool = false,
    pa: bool = false,
    order: bool = false,
    xrow: bool = false,
    /// qualification_3e1f1105 (TF_GLM_EXL3_PGROUP=1): bertholomus's parallel grouping on windows of `prompt_rows` rows and more, with X1 or
    /// without; the role marks it by placing the scratch's `counts` (routedSched, routedX1 take groupPar there)
    pgroup: bool = false,
    /// SEQ (qualification_b3fba1da; Mia's TF_GLM_EXL3_SEQ): gate/up's K splits summed in one program, one plane, the epilogue at SK 1
    seq: bool = false,
    /// MMA (qualification_8724e161, qualification_7af419c1's qualification_babb0a2d; bertholomus's grouped_mma): the windows' grouped launches through exl3/mma/ (64 member
    /// rows a program, the splits folded; gate/up stays SEQ's kernel with `seq`), on PGROUP's lists (qualification_3e1f1105)
    mma: bool = false,

    pub fn on(p: Prompt) bool {
        return p.nfirst or p.pa or p.order or p.xrow;
    }

    /// TF_GLM_EXL3_X1 (0 or 1, default 0); with it 1, python's TF_GLM_EXL3_NFIRST, _PF, _ORDER, _ROWROT (on unless 0);
    /// then TF_GLM_EXL3_PGROUP (0 or 1, default 0; qualification_3e1f1105), read whether X1 is set or not (`parsePGroup`); then
    /// TF_GLM_EXL3_SEQ (0 or 1, default 0), refused without X1 in effect (`parseSeq`); then TF_GLM_EXL3_MMA (0 or 1,
    /// default 0; qualification_8724e161), refused without X1 in effect and PGROUP (`parseMma`).
    pub fn fromEnv() error{ BadExl3X1, BadExl3PGroup, BadExl3Seq, BadExl3Mma }!Prompt {
        var p: Prompt = .{}; // X1 unset: none of its four is read
        if (try bit(env("TF_GLM_EXL3_X1"), false)) p = try parse("1", .{ env("TF_GLM_EXL3_NFIRST"), env("TF_GLM_EXL3_PF"), env("TF_GLM_EXL3_ORDER"), env("TF_GLM_EXL3_ROWROT") });
        p = try parseSeq(try parsePGroup(p, env("TF_GLM_EXL3_PGROUP")), env("TF_GLM_EXL3_SEQ"));
        return parseMma(p, env("TF_GLM_EXL3_MMA"), p.pgroup);
    }

    /// TF_GLM_EXL3_SEQ on top of `p`: unset, empty or 0 off; 1 on, but only with X1 in effect (`p.on()`); else refused.
    pub fn parseSeq(p: Prompt, v: ?[]const u8) error{BadExl3Seq}!Prompt {
        var q = p;
        q.seq = bit(v, false) catch return error.BadExl3Seq;
        if (q.seq and !p.on()) return error.BadExl3Seq; // SEQ is a form of X1's gate/up launch
        return q;
    }

    /// TF_GLM_EXL3_MMA (qualification_8724e161) on top of `p`: unset, empty or 0 off; 1 on, but only with X1 in effect and PGROUP's
    /// lists (`pgroup`: TF_GLM_EXL3_PGROUP=1, qualification_3e1f1105; the work list needs its counts); else refused.
    pub fn parseMma(p: Prompt, v: ?[]const u8, pgroup: bool) error{BadExl3Mma}!Prompt {
        var q = p;
        q.mma = bit(v, false) catch return error.BadExl3Mma;
        if (q.mma and !(p.on() and pgroup)) return error.BadExl3Mma; // MMA replaces X1's grouped launches
        return q;
    }

    /// TF_GLM_EXL3_PGROUP on top of `p`: unset, empty or 0 off; 1 on; anything else refused.
    pub fn parsePGroup(p: Prompt, v: ?[]const u8) error{BadExl3PGroup}!Prompt {
        var q = p;
        q.pgroup = bit(v, false) catch return error.BadExl3PGroup;
        return q;
    }

    pub fn parse(x1: ?[]const u8, each: [4]?[]const u8) error{BadExl3X1}!Prompt {
        if (!try bit(x1, false)) return .{};
        return .{ .nfirst = try bit(each[0], true), .pa = try bit(each[1], true), .order = try bit(each[2], true), .xrow = try bit(each[3], true) };
    }

    fn bit(v: ?[]const u8, default: bool) error{BadExl3X1}!bool {
        const s = v orelse return default;
        if (s.len == 0) return default;
        if (std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "1")) return s[0] == '1';
        return error.BadExl3X1;
    }

    fn env(name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    }
};

/// exl3_generic.PROMPT_ROWS: windows of this many rows and more take the prompt switches.
pub const prompt_rows = 64;

/// experts.order_slots: an upper bound of R rows' (expert, 16-member tile) items, the order list's launch slots.
pub fn orderSlots(R: u32, slots: u32, E: u32) u32 {
    const maxu = @min(R * slots, E);
    return @max(1, @min(maxu * ((R + 15) / 16), (R * slots + 15 * maxu) / 16));
}

/// experts.prompt_ready: 4-bit mcg experts at GLM's (8, 4, *, 1) tiles, the switches' scope.
pub fn promptReady(ex: Layer, s: Scratch) bool {
    return promptTiles(ex.cb, ex.k2_gu, ex.k2_d, s.cfg_gu, s.cfg_d);
}

/// prompt_ready on its parts: codebook mcg, k2 (8, 8) for gate/up and down, both tiles (8 n tiles, 4 warps, *, 1).
pub fn promptTiles(cb: Codebook, k2_gu: [2]u32, k2_d: [2]u32, gu: Tile, dn: Tile) bool {
    const t = struct {
        fn ok(c: Tile) bool {
            return c.nt == 8 and c.warps == 4 and c.pf == 1;
        }
    };
    return cb == .mcg and k2_gu[0] == 8 and k2_gu[1] == 8 and k2_d[0] == 8 and k2_d[1] == 8 and t.ok(gu) and t.ok(dn);
}

/// exl3x_order_cuda: prompt_order_kernel<<<1, 1024>>>(uids, ucount, members, order, maxm, cap, E).
pub fn order(out: *Launches, uids: u64, ucount: u64, members: u64, ord: u64, maxm: u32, cap: u32, E: u32) void {
    var a: cuda.Args = .{};
    for ([_]u64{ uids, ucount, members, ord }) |v| a.add(v);
    for ([_]u32{ maxm, cap, E }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_x1_experts, .symbol = k.sym.x1_order, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 } }, .args = a });
}

/// exl3x_rot_rows_cuda: rot_in_kernel<bf16 or fp16, true><<<(rows, K / 128, 1), 32>>>(x, stride, null, suh, suh, o, o, K, 1, 1).
pub fn rotRows(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, suh: u64, o: u64, rows: u32, K: u32) error{Invalid}!void {
    if (K % 128 != 0) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    for ([_]u64{ 0, suh, suh, o, o }) |v| a.add(v);
    for ([_]u32{ K, 1, 1 }) |v| a.add(int(v));
    out.push(.{ .image = .exl3_x1_experts, .symbol = if (x_bf16) k.sym.x1_rot_rows_bf16 else k.sym.x1_rot_rows_f16, .cfg = .{ .grid = .{ .x = rows, .y = K / 128, .z = 1 }, .block = .{ .x = 32 } }, .args = a });
}

/// grouped_launch with the switches (prompt_grouped_launch<1>): grouped_kernel<1, 8, 4, 1, 8, 8, NFIRST, ORDER, PA, XROW>,
/// grid (N / 128, items, z) with NFIRST else (items, N / 128, z), items the order slots (ORDER: `ord` set) or the
/// experts, z = mats SK (ORDER) or mats SK MT; the order list its 17th argument. No switch: today's `grouped`. With
/// p.seq (qualification_b3fba1da), exl3/x1s/'s grouped_seq_kernel of the same selector: z loses SK, each program sums the SK splits in
/// today's order and writes one plane (Z [mats, P, N]), so the epilogue that follows takes SK 1.
pub fn groupedX1(out: *Launches, g: Grouped, p: Prompt, ord: u64, cap: u32) error{Invalid}!void {
    const sel: u32 = (if (p.nfirst) @as(u32, 8) else 0) | (if (ord != 0) @as(u32, 4) else 0) | (if (p.pa) @as(u32, 2) else 0) | (if (p.xrow) @as(u32, 1) else 0);
    if (sel == 0) return grouped(out, g);
    if (g.cb != .mcg or g.nt != 8 or g.warps != 4 or g.pf != 1 or g.lo != 8 or g.hi != 8) return error.Invalid;
    if (g.K % (16 * g.SK * g.warps) != 0 or g.N % (16 * g.nt) != 0) return error.Invalid;
    const MT = (g.maxm + 15) / 16;
    const items = if (ord != 0) cap else g.nexp_max;
    const nbs = g.N / (16 * g.nt);
    const sk_z = if (p.seq) 1 else g.SK; // SEQ: the splits loop inside the program
    const z = if (ord != 0) g.mats * sk_z else g.mats * sk_z * MT;
    const gx = if (p.nfirst) nbs else items;
    const gy = if (p.nfirst) items else nbs;
    if (gy > 65535 or z > 65535) return error.Invalid;
    var a: cuda.Args = .{};
    for ([_]u64{ g.x0, g.x1, g.tp0, g.tp1, g.k2_0, g.k2_1, g.uids, g.ucount, g.members, g.z }) |v| a.add(v);
    for ([_]u32{ g.K, g.N, g.P, g.SK, g.maxm, g.slots }) |v| a.add(int(v));
    a.add(ord);
    if (p.seq) return out.push(.{ .image = .exl3_x1s_grouped_cb1, .symbol = k.sym.groupedX1Seq(sel), .cfg = .{ .grid = .{ .x = gx, .y = gy, .z = z }, .block = .{ .x = 4 * 32 } }, .args = a });
    out.push(.{ .image = .exl3_x1_grouped_cb1, .symbol = k.sym.groupedX1(sel), .cfg = .{ .grid = .{ .x = gx, .y = gy, .z = z }, .block = .{ .x = 4 * 32 } }, .args = a });
}

/// experts.routed with the switches (experts.py:296-332): group, the items' order (ORDER, unless NFIRST would put its
/// slots past 65,535 on grid y), one rotation a row through the layer's shared suh (XROW, with `shared_suh`; else
/// rot_in), gate+up and down with the switches (XROW on gate+up only), today's epilogues. Under `prompt_rows` rows,
/// outside `promptReady`, or with no switch in effect: `routedSched`'s launches with `sd` (0128's when on). `ord`: an
/// int32 list of orderSlots(s.rows, ...) slots (python's Scratch.order). p.seq (qualification_b3fba1da): gate+up as SEQ, its epilogue at
/// SK 1 (each output's fp32 chain is today's: an RN sum from +0 is never -0, so the epilogue's 0.f + it is it). p.mma
/// (qualification_8724e161): after PGROUP's grouping, mma.routed (bertholomus's grouped_mma over a work list) where the scratch carries
/// the counts and the work list and the layer's shapes take it (mma.ready); else X1's launches below.
pub fn routedX1(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, wts: ?u64, ex: Layer, s: Scratch, o: u64, R: u32, limit: f32, act: Act, do_group: bool, sh: Shared, sd: Sched, p: Prompt, ord: u64, shared_suh: ?u64) error{Invalid}!void {
    const xrow = p.xrow and shared_suh != null; // python: rowrot = rowrot and ex.shared_suh is not None
    const any = p.nfirst or p.pa or p.order or xrow;
    if (!any or R < prompt_rows or !promptReady(ex, s)) return routedSched(out, x, x_stride, x_bf16, pick, wts, ex, s, o, R, limit, act, do_group, sh, sd);
    const D = ex.dims;
    const I = ex.width;
    const E = ex.count;
    const slots = s.slots;
    const P = R * slots;
    if (R > s.rows) return error.Invalid;
    const maxu = @min(R * slots, s.experts);
    if (do_group) try grouping(out, pick, s, R, E, sh);
    if (p.mma and do_group and s.counts != 0 and s.work != 0 and mma.ready(ex, s)) return mma.routed(out, x, x_stride, x_bf16, pick, wts, ex, s, o, R, limit, act, p, ord, shared_suh, s.counts, s.work); // qualification_8724e161: groupPar wrote the counts
    var cap: u32 = 0;
    var obuf: u64 = 0;
    if (p.order) {
        cap = orderSlots(R, slots, E);
        if (!(p.nfirst and cap > 65535)) {
            if (ord == 0) return error.Invalid;
            obuf = ord;
            order(out, s.ids, s.count, s.members, obuf, R, cap, E);
        }
    }
    if (xrow) try rotRows(out, x, x_stride, x_bf16, shared_suh.?, s.xg, R, D) else try rotIn(out, x, x_stride, x_bf16, pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E);
    var g: Grouped = .{ .x0 = s.xg, .x1 = if (xrow) s.xg else s.xu, .tp0 = ex.gate_ptr, .tp1 = ex.up_ptr, .k2_0 = ex.gate_k2, .k2_1 = ex.up_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 2, .K = D, .N = I, .P = P, .SK = s.cfg_gu.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_gu.nt, .warps = s.cfg_gu.warps, .pf = s.cfg_gu.pf, .lo = ex.k2_gu[0], .hi = ex.k2_gu[1] };
    try groupedX1(out, g, .{ .nfirst = p.nfirst, .pa = p.pa, .xrow = xrow, .seq = p.seq }, obuf, cap);
    try gateupEpilogue(out, s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, if (p.seq) 1 else s.cfg_gu.sk, slots, E, limit, act);
    g = .{ .x0 = s.xd, .x1 = s.xd, .tp0 = ex.down_ptr, .tp1 = ex.down_ptr, .k2_0 = ex.down_k2, .k2_1 = ex.down_k2, .uids = s.ids, .ucount = s.count, .members = s.members, .z = s.z, .mats = 1, .K = I, .N = D, .P = P, .SK = s.cfg_d.sk, .slots = slots, .maxm = R, .nexp_max = maxu, .cb = ex.cb, .nt = s.cfg_d.nt, .warps = s.cfg_d.warps, .pf = s.cfg_d.pf, .lo = ex.k2_d[0], .hi = ex.k2_d[1] };
    try groupedX1(out, g, .{ .nfirst = p.nfirst, .pa = p.pa }, obuf, cap);
    if (wts) |w| return downCombine(out, s.z, pick, ex.svh_d, s.y, w, o, R, P, D, s.cfg_d.sk, slots, E);
    try downEpilogue(out, s.z, pick, ex.svh_d, s.y, R, P, D, s.cfg_d.sk, slots, E);
}

test {
    _ = @import("cuda_exl3_x1_test.zig");
    _ = @import("cuda_exl3_seq_test.zig");
    _ = mma;
}

test "default tiles: GLM's at a TP2 rank's shapes, fallbacks where they divide" {
    try std.testing.expectEqual(glm_gateup, try defaultConfig(4096, 1024, true));
    try std.testing.expectEqual(glm_down, try defaultConfig(1024, 4096, false));
    try std.testing.expectEqual(glm_gateup, try defaultConfig(4096, 2048, true));
    try std.testing.expectEqual(Tile{ .nt = 4, .warps = 4, .sk = 1, .pf = 2 }, try defaultConfig(64, 64, true));
    try std.testing.expectError(error.NoTile, defaultConfig(48, 64, true));
}

test "K2 ranges pick the instance grouped_launch compiles" {
    try std.testing.expectEqual([2]u32{ 8, 8 }, k2Range(8, 8));
    try std.testing.expectEqual([2]u32{ 2, 10 }, k2Range(4, 10));
    try std.testing.expectEqual([2]u32{ 2, 16 }, k2Range(8, 12));
}

test "0128's switches: unset, 0 and empty are off; 0..64 and 0/1 parse; anything else is refused" {
    try std.testing.expect(!(try Sched.parse(null, null)).on());
    try std.testing.expect(!(try Sched.parse("0", "0")).on());
    try std.testing.expect(!(try Sched.parse("", "")).on());
    try std.testing.expectEqual(Sched{ .l2d = 4, .pdl = true }, try Sched.parse("4", "1"));
    try std.testing.expectEqual(Sched{ .l2d = 64 }, try Sched.parse("64", null));
    try std.testing.expect((try Sched.parse(null, "1")).on());
    for ([_][2]?[]const u8{ .{ "65", null }, .{ "-1", null }, .{ "4x", null }, .{ null, "2" }, .{ null, "yes" }, .{ " 4", null } }) |c| try std.testing.expectError(error.BadExl3Sched, Sched.parse(c[0], c[1]));
}

fn glmRouted(ls: *Launches, R: u32, sd: Sched) !void {
    const l: Layer = .{ .gate_ptr = 0x1000, .up_ptr = 0x2000, .down_ptr = 0x3000, .gate_k2 = 0x4000, .up_k2 = 0x5000, .down_k2 = 0x6000, .suh_g = 0x7000, .suh_u = 0x8000, .svh_g = 0x9000, .svh_u = 0xa000, .suh_d = 0xb000, .svh_d = 0xc000, .count = 256, .dims = 4096, .width = 1024, .cb = .mcg, .k2_gu = .{ 8, 8 }, .k2_d = .{ 8, 8 } };
    const s: Scratch = .{ .xg = 0x10000, .xu = 0x20000, .xd = 0x30000, .z = 0x40000, .y = 0x50000, .ids = 0x60000, .count = 0x70000, .members = 0x80000, .rows = 64, .slots = 9, .experts = 256, .cfg_gu = glm_gateup, .cfg_d = glm_down };
    try routedSched(ls, 0xd000, 4096, true, 0xe000, null, l, s, 0, R, 7.0, .bf16, true, .{ .optin = 101376 }, sd);
}

fn argBytes(l: Launch) []const u8 {
    return l.args.storage[0..l.args.used];
}

test "with a switch on, gate+up, its epilogue and down take 0128's kernels, grid and l2d; the rest is today's launch" {
    for ([_]u32{ 1, 2, 4, 8 }) |R| for ([_]Sched{ .{ .l2d = 4 }, .{ .l2d = 2, .pdl = true }, .{ .pdl = true }, .{ .x = true }, .{ .l2d = 8 } }) |sd| {
        var off: Launches = .{};
        var on: Launches = .{};
        try glmRouted(&off, R, .{});
        try glmRouted(&on, R, sd);
        try std.testing.expectEqual(@as(usize, 6), off.len);
        try std.testing.expectEqual(off.len, on.len);
        for (off.slice(), on.slice(), 0..) |a, b, i| {
            try std.testing.expect(!a.cfg.pdl);
            if (i < 2 or i == 5) { // group, rot_in, the down epilogue: unchanged
                try std.testing.expectEqual(a.image, b.image);
                try std.testing.expectEqualStrings(a.symbol, b.symbol);
                try std.testing.expectEqual(a.cfg, b.cfg);
                try std.testing.expectEqualSlices(u8, argBytes(a), argBytes(b));
                continue;
            }
            try std.testing.expectEqual(sd.pdl, b.cfg.pdl);
            try std.testing.expectEqual(a.cfg.block, b.cfg.block);
            if (i == 3) { // the epilogue: 0128's kernel, the same grid and arguments
                try std.testing.expectEqual(k.Image.exl3_experts, a.image);
                try std.testing.expectEqual(k.Image.exl3_l2d_experts, b.image);
                try std.testing.expectEqualStrings(a.symbol, b.symbol);
                try std.testing.expectEqual(a.cfg.grid, b.cfg.grid);
                try std.testing.expectEqualSlices(u8, argBytes(a), argBytes(b));
                continue;
            }
            try std.testing.expectEqual(k.Image.exl3_grouped_cb1, a.image); // gate+up and down: grid (y, z, x), l2d last
            try std.testing.expectEqual(k.Image.exl3_l2d_grouped_cb1, b.image);
            try std.testing.expectEqualStrings(a.symbol, b.symbol[0 .. b.symbol.len - 1]);
            try std.testing.expectEqual(@as(u8, 'i'), b.symbol[b.symbol.len - 1]);
            try std.testing.expectEqual([3]u32{ a.cfg.grid.y, a.cfg.grid.z, a.cfg.grid.x }, [3]u32{ b.cfg.grid.x, b.cfg.grid.y, b.cfg.grid.z });
            try std.testing.expectEqual(a.args.count + 1, b.args.count);
            try std.testing.expectEqualSlices(u8, argBytes(a), b.args.storage[0..a.args.used]);
            try std.testing.expectEqual(@as(?u64, sd.l2d), b.args.integer(a.args.count));
        }
    };
}
