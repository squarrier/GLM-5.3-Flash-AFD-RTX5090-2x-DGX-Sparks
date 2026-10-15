//! qualification_1b9bf461: TF_GLM_HALVES_KERNEL=g53kt3r on an expert half (afd_expert.zig's Gpu opens it beside X1): the g53_rank image
//! and the state python 2.15's g53_experts.Node holds, so cuda_afd_expert.Device runs every prompt window through
//! glm53f-afd's large-M expert kernels (Hugh Madden, glm53f-afd @ 91db3cc, MIT; KT3R's schedule from 1,536 rows).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const ae = glm.afd_expert;
const k = ae.g53;

pub const G53 = struct {
    gpa: std.mem.Allocator,
    set: k.Set,
    buf: cuda.DeviceBuffer,
    ready: []bool,
    suh: []?u64,
    node: k.Node,

    /// The image (its 14 persistent kernels' occupancy at this SM count); its block, an allocation of its own so the
    /// node's block and every address in it are today's: the plan's buffers for min(4,096, prompt rows) rows, zeroed once
    /// (python's torch.zeros), then a shared suh row a position; each position's scope (python's layer_ready) and
    /// shared-suh test (python's share_suh at load, experts.py:93-102, on host copies of its gate and up suh), the row
    /// copied in where it passes. The groups must be uploaded (a synchronized stream).
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, x: *const ae.Half, base: []const u64, groups: []const cuda.DeviceBuffer, b: *const ae.Bufs, experts: usize, prefill_rows: usize, half: u32) !*G53 {
        const s = try gpa.create(G53);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        const sms: u32 = @intCast(try ctx.attribute(.multiprocessor_count));
        s.set = try k.Set.load(d, cuda.kernels.g53_rank, sms);
        errdefer s.set.unload();
        const P = x.layers.len;
        const rows: u32 = @intCast(@min(k.MAX_ROWS, prefill_rows));
        const D = b.normed.shape[1];
        const plan = std.mem.alignForward(usize, k.Plan.bytes(rows), 256);
        s.buf = try cuda.DeviceBuffer.alloc(d, plan + P * D * 2);
        errdefer s.buf.free();
        try s.buf.fill8(0, null);
        s.ready = try gpa.alloc(bool, P);
        errdefer gpa.free(s.ready);
        s.suh = try gpa.alloc(?u64, P);
        errdefer gpa.free(s.suh);
        const host = try gpa.alloc(u16, 2 * experts * D);
        defer gpa.free(host);
        const tb = try gpa.alloc(u8, 36 * experts);
        defer gpa.free(tb);
        var ready: usize = 0;
        var shared: usize = 0;
        for (s.ready, s.suh, 0..) |*ok, *suh, pos| {
            const l = x.bind(pos, base[pos]);
            const why = k.layerReady(l, x.tables(pos, base[pos], tb).bytes);
            ok.* = why == null;
            suh.* = null;
            if (why) |w| std.debug.print("INFO afd expert {d}: g53kt3r: position {d} (layer {d}) stays on today's path: {s}\n", .{ half, pos, x.layers[pos], w });
            if (!ok.*) continue;
            ready += 1;
            const n = @as(usize, l.count) * l.dims;
            if (l.dims != D or 2 * n > host.len) return error.BadG53Shape;
            const grp = groups[pos];
            try grp.download(l.suh_g - grp.ptr, std.mem.sliceAsBytes(host[0..n]));
            try grp.download(l.suh_u - grp.ptr, std.mem.sliceAsBytes(host[n .. 2 * n]));
            if (!ae.sharedSuh(host[0..n], host[n .. 2 * n], D)) continue;
            const at = plan + pos * D * 2;
            try s.buf.copyFrom(at, l.suh_g, 2 * D, null);
            suh.* = s.buf.ptr + at;
            shared += 1;
        }
        s.node = .{ .set = &s.set, .plan = k.Plan.place(rows, s.buf.ptr), .kt3r = true, .max_rows = rows, .ready = s.ready, .suh = s.suh };
        std.debug.print("INFO afd expert {d}: TF_GLM_HALVES_KERNEL=g53kt3r: prompt windows of 1-{d} rows through glm53f-afd's large-M expert kernels (Hugh Madden; outputs class), KT3R's schedule on windows of {d}+ rows; {d} of {d} MoE positions in scope, {d} share one gate/up suh (rowrot); {d} SMs; block {d} KiB\n", .{ half, rows, k.kt3_tier, ready, P, shared, sms, s.buf.len >> 10 });
        return s;
    }

    pub fn close(s: *G53) void {
        s.buf.free();
        s.gpa.free(s.ready);
        s.gpa.free(s.suh);
        s.set.unload();
        s.gpa.destroy(s);
    }
};

test {
    _ = &G53.open; // runs on a GPU (afd_expert --selftest / serve with the switch set)
    _ = &G53.close;
}
