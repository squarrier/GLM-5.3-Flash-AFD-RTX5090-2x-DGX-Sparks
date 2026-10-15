//! The forward's context (qualification_qualification_7c75d777 PLAN §4-§5): the model, the Triton wrappers, and every extension call and collective
//! python-0.6 makes, either recorded into a trace (a CPU test: nothing runs) or launched on the rank's stream.
//! Each call takes python's arguments as python passes them (trace.Val), so the record is the call python made; the
//! launch path turns it into the kernel's own launch (cuda_kernels.zig for KDA; lane A for the matmuls).
const std = @import("std");
const cuda = @import("cuda");
const md = @import("cuda_model.zig");
const trace = @import("cuda_trace.zig");
const tri = @import("cuda_triton.zig");
const kern = @import("cuda_kernels.zig");
const mm = @import("cuda_kernels_mm.zig");
const exl3 = @import("cuda_exl3.zig");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const layer = @import("cuda_layer.zig");

const T = trace.T;
const Val = trace.Val;

/// An FFN block's fp32 partials for hc_post: `world` planes [world, R, D] at `g`, summed plane 0 first.
pub const Planes = struct { g: u64, world: usize };

/// A MoE layer's FFN block in place of cuda_moe.block (python 2.15's `w.afd`, the AFD attention node's exchange).
pub const Moe = struct {
    ctx: *anyopaque,
    block: *const fn (ctx: *anyopaque, f: *Fwd, l: *const views.Layer, b: *const bufs.Buffers, R: usize, opt: layer.Opt) anyerror!Planes,
};

/// The extension kernels a rank launches: KDA's (qualification_qualification_44ba554e-kernels), the affine matmuls' and experts' (qualification_qualification_abcce4b2, glm_fwd_kernels),
/// and EXL3's routed experts (qualification_qualification_44ba554e-exl3's six images, with the device's opt-in shared bytes a block; EXL3 ranks only);
/// DSA's select_tokens tail (qualification_qualification_973313f6, glm_select; long contexts only).
pub const Kernels = struct {
    kda: *const kern.Kda,
    mm: *const mm.Mm,
    exl3: ?*const exl3.kernels.Set = null,
    exl3_optin: u32 = 0,
    /// Mia's 0128 in the EXL3 experts (TF_GLM_EXL3_L2D, TF_GLM_EXL3_PDL); off: today's kernels and launches
    exl3_sched: exl3.experts.Sched = .{},
    dsa: ?*const @import("cuda_dsa_kernels.zig").Kernels = null,
};

/// The rank's GPU: its stream, the Triton set, the extension kernels, and the communicator (null on one rank).
pub const Gpu = struct {
    stream: cuda.Stream,
    set: *const cuda.aot.Set,
    kernels: Kernels,
    comm: ?*const cuda.comm.Comm = null,
    /// kda.py's _wide_scratch (q fp32, y bf16) for windows of kern.wide_rows rows and more
    q_tmp: u64 = 0,
    y_tmp: u64 = 0,
};

pub const Fwd = struct {
    m: *const md.Model,
    tri: tri.Tri,
    rec: ?*trace.Trace = null,
    gpu: ?Gpu = null,
    /// null: cuda_moe.block (one box); set: the provider's block (the AFD attention node)
    moe: ?Moe = null,

    /// Records every launch into `t`; nothing runs.
    pub fn recording(m: *const md.Model, t: *trace.Trace) Fwd {
        return .{ .m = m, .tri = .{ .set = null, .rec = &t.tri_rec }, .rec = t };
    }

    /// Launches on `g`'s stream.
    pub fn launching(m: *const md.Model, g: Gpu) Fwd {
        return .{ .m = m, .tri = .{ .set = g.set, .s = g.stream }, .gpu = g };
    }

    /// Moves recorded Triton launches into the trace (call once a step ends).
    pub fn flush(f: *Fwd) !void {
        if (f.rec) |t| try t.flush();
    }

    /// kernels/qmm.py `_ext().qmm_group(x, xs, [w], [s], [b], [out], [n], [sk], f32, tile, early)` with one weight.
    pub fn qmmGroup(f: *Fwd, x: T, xs: T, w: [3]T, out: T, n: usize, sk: usize, f32_out: bool, tile: usize) !void {
        if (f.rec) |t| return t.call(.ext, "qmm.qmm_group", &.{
            .{ .t = x },                   .{ .t = xs },
            .{ .list = &.{.{ .t = w[0] }} }, .{ .list = &.{.{ .t = w[1] }} },
            .{ .list = &.{.{ .t = w[2] }} }, .{ .list = &.{.{ .t = out }} },
            .{ .list = &.{.{ .int = @intCast(n) }} }, .{ .list = &.{.{ .int = @intCast(sk) }} },
            .{ .bool = f32_out },          .{ .int = @intCast(tile) },
            .{ .int = -1 },
        });
        const g = f.gpu orelse return error.NoGpu;
        try g.kernels.mm.run(g.stream, try qmmGroupLaunch(g.kernels.mm.chip, x, xs, w, out, n, sk, f32_out, tile));
    }

    /// kernels/qmm.py `_ext().qmm_prefill(x, w, s, b, out, n, gs, f32, tile)` (prompt chunks).
    pub fn qmmPrefill(f: *Fwd, x: T, w: [3]T, out: T, n: usize, f32_out: bool, tile: usize) !void {
        if (f.rec) |t| return t.call(.ext, "qmm.qmm_prefill", &.{
            .{ .t = x }, .{ .t = w[0] }, .{ .t = w[1] }, .{ .t = w[2] }, .{ .t = out }, .{ .int = @intCast(n) },
            .{ .int = 64 }, .{ .bool = f32_out }, .{ .int = @intCast(tile) },
        });
        const g = f.gpu orelse return error.NoGpu;
        try g.kernels.mm.run(g.stream, try qmmPrefillLaunch(x, w, out, n, f32_out, tile));
    }

    /// kda.py `chain`: `_ext().chain(...)` under kern.wide_rows rows, `chain_wide(..., q_tmp, y_tmp)` from there.
    pub fn kdaChain(f: *Fwd, a: Chain) !void {
        const wide = a.rows >= kern.wide_rows;
        if (f.rec) |t| {
            const head = [_]Val{
                .{ .t = a.p },         .{ .int = @intCast(a.p.stride[0]) }, .{ .int = @intCast(a.b_off) },
                .{ .t = a.a },         .{ .int = @intCast(a.a.stride[0]) }, .{ .t = a.g },
                .{ .int = @intCast(a.g.stride[0]) }, .{ .t = a.conv_state }, .{ .t = a.conv_w },
                .{ .t = a.state_in },  .{ .t = a.a_log },                  .{ .t = a.dt_bias },
                .{ .t = a.norm_w },    Val.float(a.eps),                   Val.float(a.lower),
                .{ .int = @intCast(a.rows) }, .{ .t = a.out },             .{ .t = a.state_out },
                .{ .t = a.k_save },    .{ .t = a.v_save },                 .{ .t = a.g_save },
                .{ .t = a.b_save },
            };
            if (!wide) return t.call(.ext, "kda.chain", &head);
            return t.call(.ext, "kda.chain_wide", &(head ++ [_]Val{ .{ .t = a.q_tmp.? }, .{ .t = a.y_tmp.? } }));
        }
        const g = f.gpu orelse return error.NoGpu;
        const c: kern.Chain = .{
            .heads = a.a_log.shape[0], .p = a.p.ptr, .p_stride = a.p.stride[0], .b_off = a.b_off, .a = a.a.ptr,
            .a_stride = a.a.stride[0], .g = a.g.ptr, .g_stride = a.g.stride[0], .conv_state = a.conv_state.ptr,
            .conv_w = a.conv_w.ptr, .state_in = a.state_in.ptr, .a_log = a.a_log.ptr, .dt_bias = a.dt_bias.ptr,
            .norm_w = a.norm_w.ptr, .eps = a.eps, .lower = a.lower, .rows = a.rows, .out = a.out.ptr,
            .state_out = a.state_out.ptr, .k_save = a.k_save.ptr, .v_save = a.v_save.ptr, .g_save = a.g_save.ptr,
            .b_save = a.b_save.ptr,
        };
        try g.kernels.kda.chain(g.stream, c, if (a.q_tmp) |q| q.ptr else g.q_tmp, if (a.y_tmp) |y| y.ptr else g.y_tmp);
    }

    /// `w.comm.all_gather(send, recv)`: fp32 [rows * width] from every rank into [world, rows, width], rank order.
    pub fn allGather(f: *Fwd, send: T, recv: T) !void {
        if (f.rec) |t| return t.call(.comm, "all_gather", &.{ .{ .t = send }, .{ .t = recv } });
        const g = f.gpu orelse return error.NoGpu;
        const c = g.comm orelse return error.NoComm;
        try c.allGather(send.ptr, recv.ptr, send.shape[0], .f32, g.stream.handle);
    }
};

/// qmm.cpp qmm_group's launch for one weight [words, scales, biases] as python passes it (scales' row stride is npad).
pub fn qmmGroupLaunch(chip: mm.Chip, x: T, xs: T, w: [3]T, out: T, n: usize, sk: usize, f32_out: bool, tile: usize) mm.Error!mm.Launch {
    return mm.groupCall(chip, x, xs, w, out, n, sk, f32_out, @intCast(tile));
}

/// qmm.cpp qmm_prefill's launch (groups of 64; scales and biases (k / 64, npad)).
pub fn qmmPrefillLaunch(x: T, w: [3]T, out: T, n: usize, f32_out: bool, tile: usize) mm.Error!mm.Launch {
    return mm.prefillCall(x, w, out, n, f32_out, @intCast(tile));
}

/// kda.chain's arguments as python passes them (tensors with their shapes and strides).
pub const Chain = struct {
    p: T,
    b_off: usize,
    a: T,
    g: T,
    conv_state: T,
    conv_w: T,
    state_in: T,
    a_log: T,
    dt_bias: T,
    norm_w: T,
    eps: f32,
    lower: f32,
    rows: usize,
    out: T,
    state_out: T,
    k_save: T,
    v_save: T,
    g_save: T,
    b_save: T,
    q_tmp: ?T = null,
    y_tmp: ?T = null,
};
