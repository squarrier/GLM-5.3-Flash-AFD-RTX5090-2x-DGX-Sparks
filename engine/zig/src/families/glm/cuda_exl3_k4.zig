//! K4 (qualification_e4ad04cd; TF_GLM_HALVES_KERNEL=k4): MiaAI-Lab's EXL3 prompt kernel for a prompt window's routed experts (her
//! GLM-5.3-Flash recipe patches 0004 / 0009 / 0020, Apache-2.0), as python 2.15 ports it in patch 0024
//! (src/tensorfold/cuda/exl3/experts_prompt.cu; exl3/k4/'s copy, its SASS equal to python's own build). Each function
//! is exl3x_prompt_cuda's launch (grid, block, shared bytes, arguments); `routed` is experts.py's _prompt_routed. Every
//! output is one fp32 chain of mma over ascending k tiles, no split K and no partials in memory: her lane's prompt
//! bits, not X1's split-K sums, so the outputs change against the DEFAULT by design (the KL gate, not G-same).

const std = @import("std");
const cuda = @import("cuda");
const k = @import("cuda_exl3_kernels.zig");
const ex3 = @import("cuda_exl3_experts.zig");

const Launches = k.Launches;

/// experts.py PROMPT_PASSES: the members a pass takes (her TF_GLM_EXL3_PASS), 64 (her default; 8 warps, two blocks an
/// SM) or 128 (16 warps, one), the same bits.
pub const passes = [_]u32{ 64, 128 };

/// experts_prompt.cu's tile constants: PBK k values a stage, PSTAGES stages, PXS a stage row's halves, PWORDS a
/// stage's trellis words, and the epilogues' row strides GU_ES / DN_ES (fp32).
const PBK = 32;
const PSTAGES = 3;
const PXS = PBK + 8;
const PWORDS = 16 * (PBK / 16) * 32;
const GU_ES = 128 + 4;
const DN_ES = 256 + 4;

/// prompt_smem<PM, XM>(): the larger of the gate/up stages (XM copies of the member rows), the down stages, the gate/up
/// epilogue's half and the down epilogue's half, in bytes. prompt_launch takes XM 2 (SMEM) for gate/up and down, XM 1
/// (SMEM_SHX) for gate/up with shared rows.
pub fn smem(pm: u32, xm: u32) u32 {
    const gu = PSTAGES * (xm * pm * PXS + 2 * PWORDS) * 2;
    const ep = 2 * (pm / 2) * GU_ES * 4;
    const dn = (pm / 2) * DN_ES * 4;
    return @max(gu, @max(ep, dn));
}

/// A pass's warps (prompt_launch<64, 8> and <128, 16>).
pub fn warps(pm: u32) u32 {
    return pm / 8;
}

/// experts.py prompt_items: the launch slots (her plan's max_items, the kernels' grid y) for R rows: an item a used
/// expert (the shared one too), plus one a `pass` pairs past its first.
pub fn promptItems(R: u32, slots: u32, E: u32, pass: u32) u32 {
    return @min(R * slots, E + 1) + (R * slots) / pass;
}

/// experts.py prompt_kernel_ready: 4-bit mcg experts (one 32-bit word a lane a tile), widths in the kernels' blocks (D:
/// 256 columns, the expert width: 128), trellises on 16-byte boundaries (`aligned`: the cp.async stages), 1 to 32
/// slots (group()'s member codes).
pub fn ready(ex: ex3.Layer, slots: u32, aligned: bool) bool {
    return ex.cb == .mcg and ex.k2_gu[0] == 8 and ex.k2_gu[1] == 8 and ex.k2_d[0] == 8 and ex.k2_d[1] == 8 and ex.dims % 256 == 0 and ex.width % 128 == 0 and slots >= 1 and slots <= 32 and aligned;
}

fn int(x: anytype) c_int {
    return @intCast(x);
}

/// prompt_plan_kernel<<<1, 1024>>>(uids, ucount, members, maxm, pass, cap, E, items, counts): group()'s grouping as her
/// plan's items (expert, first member, count), `cap` the launch slots (it traps past them: never).
pub fn plan(out: *Launches, uids: u64, ucount: u64, members: u64, maxm: u32, pass: u32, cap: u32, E: u32, items: u64, counts: u64) void {
    var a: cuda.Args = .{};
    for ([_]u64{ uids, ucount, members }) |v| a.add(v);
    for ([_]u32{ maxm, pass, cap, E }) |v| a.add(int(v));
    a.add(items);
    a.add(counts);
    out.push(.{ .image = .exl3_k4, .symbol = k.sym.k4_plan, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 } }, .args = a });
}

/// prompt_order_kernel<<<1, 1024>>>(items, counts, order, cap, E): her item order (ORDER), which block runs which item.
pub fn order(out: *Launches, items: u64, counts: u64, ord: u64, cap: u32, E: u32) void {
    var a: cuda.Args = .{};
    for ([_]u64{ items, counts, ord }) |v| a.add(v);
    a.add(int(cap));
    a.add(int(E));
    out.push(.{ .image = .exl3_k4, .symbol = k.sym.k4_order, .cfg = .{ .grid = .{}, .block = .{ .x = 1024 } }, .args = a });
}

/// What prompt_launch passes the gate/up kernel beside the plan.
pub const GateUp = struct {
    xg: u64, // fp16 [P, D] (SHX: [rows, D], one row a token)
    xu: u64,
    tg: u64, // int64 [E]: each expert's gate trellis
    tu: u64,
    items: u64,
    counts: u64,
    members: u64,
    svh_g: u64, // fp16 [E, N]
    svh_u: u64,
    suh_d: u64,
    xd: u64, // fp16 [P, N]
    D: u32,
    N: u32,
    E: u32,
    limit: f32,
    slots: u32,
    ord: u64, // 0: plan order
    cap: u32, // grid y
    pass: u32,
    shx: bool,
};

/// prompt_gateup_kernel<PM, NW, SHX><<<(N / 128, cap), NW * 32, SMEM or SMEM_SHX>>>, the attribute raised to it first.
pub fn gateup(out: *Launches, g: GateUp) error{Invalid}!void {
    if (g.pass != 64 and g.pass != 128) return error.Invalid;
    if (g.D % 256 != 0 or g.N % 128 != 0 or g.D % PBK != 0 or g.N % PBK != 0) return error.Invalid;
    if (g.shx and g.xg != g.xu) return error.Invalid; // shared rows take one Xh for gate and up
    if (g.cap == 0 or g.cap > 65535) return error.Invalid;
    const sm = smem(g.pass, if (g.shx) 1 else 2);
    var a: cuda.Args = .{};
    for ([_]u64{ g.xg, g.xu, g.tg, g.tu, g.items, g.counts, g.members, g.svh_g, g.svh_u, g.suh_d, g.xd }) |v| a.add(v);
    for ([_]u32{ g.D, g.N, g.E }) |v| a.add(int(v));
    a.add(g.limit);
    a.add(int(g.slots));
    a.add(g.ord);
    out.push(.{ .image = .exl3_k4, .symbol = k.sym.k4Gateup(g.pass, g.shx), .cfg = .{ .grid = .{ .x = g.N / 128, .y = g.cap }, .block = .{ .x = warps(g.pass) * 32 }, .shared = sm }, .args = a, .max_dynamic = sm });
}

/// What prompt_launch passes the down kernel beside the plan.
pub const Down = struct {
    xd: u64, // fp16 [P, N]
    td: u64, // int64 [E]: each expert's down trellis
    items: u64,
    counts: u64,
    members: u64,
    svh_d: u64, // fp16 [E, D]
    y: u64, // fp32 [P, D]
    N: u32,
    D: u32,
    E: u32,
    slots: u32,
    ord: u64,
    cap: u32,
    pass: u32,
};

/// prompt_down_kernel<PM, NW><<<(D / 256, cap), NW * 32, SMEM>>>(xd, td, items, counts, members, svh_d, y, N, D, E,
/// slots, order), the attribute raised to SMEM first: y's rows of every member, the down epilogue fused.
pub fn down(out: *Launches, d: Down) error{Invalid}!void {
    if (d.pass != 64 and d.pass != 128) return error.Invalid;
    if (d.D % 256 != 0 or d.N % 128 != 0 or d.D % PBK != 0 or d.N % PBK != 0) return error.Invalid;
    if (d.cap == 0 or d.cap > 65535) return error.Invalid;
    const sm = smem(d.pass, 2);
    var a: cuda.Args = .{};
    for ([_]u64{ d.xd, d.td, d.items, d.counts, d.members, d.svh_d, d.y }) |v| a.add(v);
    for ([_]u32{ d.N, d.D, d.E, d.slots }) |v| a.add(int(v));
    a.add(d.ord);
    out.push(.{ .image = .exl3_k4, .symbol = k.sym.k4Down(d.pass), .cfg = .{ .grid = .{ .x = d.D / 256, .y = d.cap }, .block = .{ .x = warps(d.pass) * 32 }, .shared = sm }, .args = a, .max_dynamic = sm });
}

/// python 2.15's prompt_switches (exl3_generic.py:64-81) as K4 takes them, read once with the switch:
/// TF_GLM_EXL3_PASS (64 or 128; unset or empty 64, spaces stripped as python strips them) and Mia's TF_GLM_EXL3_ORDER
/// and TF_GLM_EXL3_ROWROT (on unless 0; X1's reading of the same names: 0 or 1, unset or empty on, else refused).
pub const Opts = struct {
    pass: u32 = 64,
    order: bool = true,
    xrow: bool = true,

    pub fn parse(pass: ?[]const u8, order_: ?[]const u8, rowrot: ?[]const u8) error{BadExl3Prompt}!Opts {
        return .{ .pass = try parsePass(pass), .order = try bit(order_), .xrow = try bit(rowrot) };
    }

    /// TF_GLM_EXL3_PASS: unset or blank 64; `64` or `128` (python's strip()); anything else refused.
    pub fn parsePass(v: ?[]const u8) error{BadExl3Prompt}!u32 {
        const s = std.mem.trim(u8, v orelse return 64, " \t\r\n\x0b\x0c");
        if (s.len == 0 or std.mem.eql(u8, s, "64")) return 64;
        if (std.mem.eql(u8, s, "128")) return 128;
        return error.BadExl3Prompt;
    }

    fn bit(v: ?[]const u8) error{BadExl3Prompt}!bool {
        const s = v orelse return true;
        if (s.len == 0 or std.mem.eql(u8, s, "1")) return true;
        if (std.mem.eql(u8, s, "0")) return false;
        return error.BadExl3Prompt;
    }

    pub fn fromEnv() error{BadExl3Prompt}!Opts {
        return parse(env("TF_GLM_EXL3_PASS"), env("TF_GLM_EXL3_ORDER"), env("TF_GLM_EXL3_ROWROT"));
    }

    fn env(name: [:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    }
};

/// The plan's buffers (python's Scratch.prompt_bufs: items [3 cap], counts [2], order [cap] int32), `cap` the slots
/// they hold: promptItems(rows, slots, E, 64) for the scratch's widest window at the smaller pass.
pub const Bufs = struct { items: u64, counts: u64, order: u64, cap: u32 };

/// experts.routed with `prompt` (experts.py:296-300) through _prompt_routed (376-405): group, one rotation a row through
/// the layer's shared suh (SHX: `o.xrow` and a shared suh) or rot_in, the plan, her item order (`o.order`), gate/up
/// writing xd and down writing y [P, D] fp32 of every routed pair; no z partials, no epilogue launches (the caller's
/// combine, as python's node does). `aligned`: every trellis on a 16-byte boundary (prompt_kernel_ready's last test);
/// outside `ready` the caller keeps its own path.
pub fn routed(out: *Launches, x: u64, x_stride: u32, x_bf16: bool, pick: u64, ex: ex3.Layer, s: ex3.Scratch, R: u32, limit: f32, do_group: bool, sh: ex3.Shared, o: Opts, b: Bufs, shared_suh: ?u64) error{Invalid}!void {
    const D = ex.dims;
    const N = ex.width;
    const E = ex.count;
    const slots = s.slots;
    if (R == 0 or R > s.rows) return error.Invalid;
    const cap = promptItems(R, slots, E, o.pass);
    if (cap > b.cap) return error.Invalid; // python: the buffers hold the widest window's items at pass 64
    const shx = o.xrow and shared_suh != null; // python: shx = rowrot and ex.shared_suh is not None
    if (do_group) try ex3.group(out, pick, s.ids, s.count, s.members, R, slots, E, R, sh);
    if (shx) try ex3.rotRows(out, x, x_stride, x_bf16, shared_suh.?, s.xg, R, D) else try ex3.rotIn(out, x, x_stride, x_bf16, pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E);
    plan(out, s.ids, s.count, s.members, R, o.pass, cap, E, b.items, b.counts);
    if (o.order) order(out, b.items, b.counts, b.order, cap, E);
    const ord: u64 = if (o.order) b.order else 0;
    const xu = if (shx) s.xg else s.xu;
    try gateup(out, .{ .xg = s.xg, .xu = xu, .tg = ex.gate_ptr, .tu = ex.up_ptr, .items = b.items, .counts = b.counts, .members = s.members, .svh_g = ex.svh_g, .svh_u = ex.svh_u, .suh_d = ex.suh_d, .xd = s.xd, .D = D, .N = N, .E = E, .limit = limit, .slots = slots, .ord = ord, .cap = cap, .pass = o.pass, .shx = shx });
    try down(out, .{ .xd = s.xd, .td = ex.down_ptr, .items = b.items, .counts = b.counts, .members = s.members, .svh_d = ex.svh_d, .y = s.y, .N = N, .D = D, .E = E, .slots = slots, .ord = ord, .cap = cap, .pass = o.pass });
}

test {
    _ = @import("cuda_exl3_k4_test.zig");
}
