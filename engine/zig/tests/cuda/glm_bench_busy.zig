//! bench's GPU-busy profile (qualification_358eae44): how much of a decode step the GPU spends running kernels. During `busy=N`'s steps a
//! timing event is recorded on the issuing stream before and after every kernel launch and every async copy or set
//! (the rank's Driver table's entries are swapped for the wrappers below, and put back after); nothing waits on them.
//! Once the step's stream has drained, a step's GPU-busy time is the sum of its pairs' elapsed times: against the
//! step's wall time it gives the host-bound share. NCCL launches its all-gathers inside NCCL: not covered here (bench's
//! gathers=N times them alone).
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;

/// event pairs a step can hold; ops past it are counted in `over` and not timed
pub const max_ops = 4096;

const Params = ?[*]?*anyopaque;
const Ptr = ?*anyopaque;
const CPtr = ?*const anyopaque;

/// One step's figures from the pairs.
pub const Step = struct { busy_ms: f64, span_ms: f64, ops: usize, over: usize };

pub const Hook = struct {
    d: *cuda.Driver = undefined,
    real: abi.Api = undefined,
    ev: [2 * max_ops]abi.Event = @splat(null),
    made: usize = 0,
    n: usize = 0,
    over: usize = 0,

    /// The pairs made, then `d`'s launch, copy and set entries swapped for the wrappers (the one hook at a time).
    pub fn install(h: *Hook, d: *const cuda.Driver) !void {
        if (cur != null) return error.BusyHookInUse;
        h.d = @constCast(d); // the rank's Driver is the bring-up's own var: its table is writable memory
        h.real = d.api;
        for (&h.ev) |*e| {
            try d.check(d.api.cuEventCreate(e, 0), "cuEventCreate");
            h.made += 1;
        }
        cur = h;
        const api = &h.d.api;
        api.cuLaunchKernel = &launchKernel;
        api.cuLaunchKernelEx = &launchKernelEx;
        api.cuMemcpyHtoDAsync_v2 = &copyHtoD;
        api.cuMemcpyDtoHAsync_v2 = &copyDtoH;
        api.cuMemcpyDtoDAsync_v2 = &copyDtoD;
        api.cuMemcpy2DAsync_v2 = &copy2D;
        api.cuMemsetD8Async = &set8;
        api.cuMemsetD32Async = &set32;
    }

    /// The table as it was, and the pairs destroyed.
    pub fn remove(h: *Hook) void {
        if (cur == h) {
            h.d.api = h.real;
            cur = null;
        }
        for (h.ev[0..h.made]) |e| _ = h.real.cuEventDestroy_v2(e);
        h.made = 0;
    }

    fn pre(h: *Hook, s: abi.Stream) void {
        if (h.n < max_ops) _ = h.real.cuEventRecord(h.ev[2 * h.n], s);
    }

    fn post(h: *Hook, s: abi.Stream) void {
        if (h.n < max_ops) {
            _ = h.real.cuEventRecord(h.ev[2 * h.n + 1], s);
            h.n += 1;
        } else h.over += 1;
    }

    /// After the step's stream has drained: the pairs' summed elapsed time, the span from the first op's start to the
    /// last op's end, and the ops; then the pairs are free for the next step.
    pub fn take(h: *Hook) !Step {
        defer {
            h.n = 0;
            h.over = 0;
        }
        var busy: f64 = 0;
        for (0..h.n) |k| busy += try h.elapsed(2 * k, 2 * k + 1);
        const span = if (h.n > 0) try h.elapsed(0, 2 * h.n - 1) else 0;
        return .{ .busy_ms = busy, .span_ms = span, .ops = h.n, .over = h.over };
    }

    fn elapsed(h: *Hook, a: usize, b: usize) !f64 {
        var ms: f32 = 0;
        try h.d.check(h.real.cuEventElapsedTime(&ms, h.ev[a], h.ev[b]), "cuEventElapsedTime");
        return ms;
    }
};

/// the installed hook (null outside busy's steps)
var cur: ?*Hook = null;

fn launchKernel(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sh: c_uint, s: abi.Stream, p: Params, x: Params) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, p, x);
}

fn launchKernelEx(c: *const abi.LaunchConfig, f: abi.Function, p: Params, x: Params) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(c.stream);
    defer h.post(c.stream);
    return h.real.cuLaunchKernelEx(c, f, p, x);
}

fn copyHtoD(dst: abi.DevicePtr, src: CPtr, n: usize, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemcpyHtoDAsync_v2(dst, src, n, s);
}

fn copyDtoH(dst: Ptr, src: abi.DevicePtr, n: usize, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemcpyDtoHAsync_v2(dst, src, n, s);
}

fn copyDtoD(dst: abi.DevicePtr, src: abi.DevicePtr, n: usize, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemcpyDtoDAsync_v2(dst, src, n, s);
}

fn copy2D(p: *const abi.Memcpy2D, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemcpy2DAsync_v2(p, s);
}

fn set8(dst: abi.DevicePtr, v: u8, n: usize, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemsetD8Async(dst, v, n, s);
}

fn set32(dst: abi.DevicePtr, v: c_uint, n: usize, s: abi.Stream) callconv(.c) abi.Result {
    const h = cur.?;
    h.pre(s);
    defer h.post(s);
    return h.real.cuMemsetD32Async(dst, v, n, s);
}
