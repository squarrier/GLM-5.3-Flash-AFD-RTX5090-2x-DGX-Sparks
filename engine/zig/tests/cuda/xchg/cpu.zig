//! fn_xchg.cu's kernels on CPU threads (xchg_emu.cpp) behind the GPU launcher's checks: the tests' engine and tf-xchg's no-GPU mode, with host addresses in place of device ones.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;

extern fn tf_xchg_emu_post(blocks: c_uint, threads: c_uint, in: *const anyopaque, bytes: c_uint, region: [*]u8, send_off: u64, ctrl_off: u64, slot_bytes: u64, st: *anyopaque, status: [*]const u32, slots: c_uint) void;
extern fn tf_xchg_emu_take(blocks: c_uint, threads: c_uint, nsrc: c_int, src0: c_uint, src1: c_uint, out0: *anyopaque, out1: *anyopaque, bytes0: c_uint, bytes1: c_uint, region: [*]const u8, recv_off: u64, flag_off: u64, slot_bytes: u64, ports: c_int, st: *anyopaque, status: [*]u32, give_up_ns: u64, slots: c_uint) void;
extern fn tf_xchg_emu_take_striped(blocks: c_uint, threads: c_uint, nsrc: c_int, src0: c_uint, src1: c_uint, out0: *anyopaque, out1: *anyopaque, bytes0: c_uint, bytes1: c_uint, region: [*]const u8, recv_off: u64, flag_off: u64, slot_bytes: u64, ports: c_int, stripes: c_uint, stripe_min: u64, parts: c_int, st: *anyopaque, status: [*]u32, give_up_ns: u64, slots: c_uint) void;
extern fn tf_xchg_emu_lab_experts(blocks: c_uint, threads: c_uint, x: [*]const u16, pick: [*]const i32, wts: [*]const f32, out: *anyopaque, rows: c_int, d: c_int, slots: c_int, lo: c_int, hi: c_int, bf16_out: c_int) void;

pub const Cpu = struct {
    /// fn_xchg's State: post_seq, take_seq, stage, tail
    state: [4]u32 = @splat(0),
    blocks: u32 = 1,
    threads: u32 = 8,
    /// qualification_17c70e37: the striped take's units a stripe, and how many takes went striped
    parts: u32 = 2,
    striped: u64 = 0,

    /// One message from `in` into the node's send slot, then the doorbell (neither once the exchange has failed).
    pub fn post(c: *Cpu, n: *const xchg.Node, in: []const u8) !void {
        try xchg.gpu.checkPost(n, @intFromPtr(in.ptr), in.len);
        tf_xchg_emu_post(c.blocks, c.threads, in.ptr, @intCast(in.len), n.mem.region.ptr, n.reg.send(0), n.reg.ctrl(), n.reg.slot, &c.state, n.mem.status.ptr, n.reg.slots);
    }

    /// The node's incoming messages into `outs` after the bounded waits (this thread waits, as the GPU's stream would).
    /// qualification_17c70e37: messages in stripes (the settings' stripes) through the striped take, `parts` units a stripe.
    pub fn take(c: *Cpu, n: *const xchg.Node, outs: []const []u8) !void {
        var o: [xchg.proto.EXPERTS]xchg.Out = undefined;
        for (outs, 0..) |x, i| o[i] = .{ .ptr = @intFromPtr(x.ptr), .bytes = x.len };
        try xchg.gpu.checkTake(n, o[0..outs.len]);
        const second = if (outs.len > 1) outs[1] else outs[0];
        if (xchg.gpu.stripedGrid(n.cfg.settings, o[0..outs.len]) != null) {
            const s = n.cfg.settings;
            c.striped += 1;
            return tf_xchg_emu_take_striped(c.blocks, c.threads, @intCast(outs.len), n.src(0), n.src(1), outs[0].ptr, second.ptr, @intCast(outs[0].len), if (outs.len > 1) @intCast(outs[1].len) else 0, n.mem.region.ptr, n.reg.recv(0, 0), n.reg.flagBase(), n.reg.slot, @intCast(s.ports), s.stripes, s.stripe_min, @intCast(c.parts), &c.state, n.mem.status.ptr, n.cfg.give_up_ns, n.reg.slots);
        }
        tf_xchg_emu_take(c.blocks, c.threads, @intCast(outs.len), n.src(0), n.src(1), outs[0].ptr, second.ptr, @intCast(outs[0].len), if (outs.len > 1) @intCast(outs[1].len) else 0, n.mem.region.ptr, n.reg.recv(0, 0), n.reg.flagBase(), n.reg.slot, @intCast(n.cfg.settings.ports), &c.state, n.mem.status.ptr, n.cfg.give_up_ns, n.reg.slots);
    }

    /// The lab's stand-in expert half over a request in host memory.
    pub fn labExperts(c: *Cpu, req: []const u8, l: xchg.proto.Layout, rows: usize, lo: i32, hi: i32, bf16_out: bool, out: []u8) void {
        const o = l.offsets(rows);
        tf_xchg_emu_lab_experts(c.blocks, c.threads, @ptrCast(@alignCast(req.ptr + o.normed)), @ptrCast(@alignCast(req.ptr + o.pick)), @ptrCast(@alignCast(req.ptr + o.wts)), out.ptr, @intCast(rows), @intCast(l.d), @intCast(l.s), lo, hi, @intFromBool(bf16_out));
    }
};
