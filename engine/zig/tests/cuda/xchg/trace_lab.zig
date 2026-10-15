//! The lab's side of the exchange (tf-xchg and the CPU tests): requests whose values change every forward, the stand-in expert half's reference, the attention's checked and timed loops, the expert's loop, on a GPU or the CPU build.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const Cpu = @import("trace_cpu.zig").Cpu;
comptime { if (!@import("builtin").is_test) @compileError("ping_pong_fixture trace_lab is test-only"); }
const Allocator = std.mem.Allocator;

fn mix(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// A normal bf16 of either sign in [2^-7, 2^4): the rows' values, a function of the forward, row and column.
fn rowBits(fseq: i32, r: usize, d: usize) u16 {
    const h = mix(@as(u64, @as(u32, @bitCast(fseq))) << 40 ^ @as(u64, r) << 20 ^ d);
    return @intCast(((h >> 16) & 1) << 15 | (120 + h % 11) << 7 | ((h >> 8) & 0x7f));
}

/// The lab's request for header `h` into `out` (python 2.15's layout): rows, 8 distinct picks of `experts` plus the shared slot (id `experts`, weight 1), weights j/128.
pub fn fillRequest(l: proto.Layout, experts: u32, h: proto.Header, out: []u8) void {
    const rows: usize = @intCast(h.rows);
    const o = l.offsets(rows);
    @memset(out[0..proto.wireBytes(o.end)], 0);
    h.write(out[0..proto.HDR_BYTES]);
    const x = std.mem.bytesAsSlice(u16, out[o.normed..][0 .. 2 * l.d * rows]);
    for (0..rows) |r| for (0..l.d) |c| {
        x[r * l.d + c] = rowBits(h.fseq, r, c);
    };
    const pick = std.mem.bytesAsSlice(i32, out[o.pick..][0 .. 4 * l.s * rows]);
    const wts = std.mem.bytesAsSlice(f32, out[o.wts..][0 .. 4 * l.s * rows]);
    for (0..rows) |r| {
        const base = mix(@as(u64, @as(u32, @bitCast(h.fseq))) * 7919 + r);
        for (0..l.s) |j| {
            const last = j + 1 == l.s;
            pick[r * l.s + j] = if (last) @intCast(experts) else @intCast((base % experts + j * 37) % experts);
            wts[r * l.s + j] = if (last) 1.0 else @as(f32, @floatFromInt(1 + mix(base + j) % 127)) / 128.0;
        }
    }
}

/// The stand-in expert half fn_xchg.cu's lab kernel computes, on the CPU: experts [lo, hi) of each row's picks, fp32 in slot order, bf16 (nearest even) when asked.
pub fn expertRef(l: proto.Layout, req: []const u8, rows: usize, lo: i32, hi: i32, bf16_out: bool, out: []u8) void {
    const o = l.offsets(rows);
    const x = std.mem.bytesAsSlice(u16, req[o.normed..][0 .. 2 * l.d * rows]);
    const pick = std.mem.bytesAsSlice(i32, req[o.pick..][0 .. 4 * l.s * rows]);
    const wts = std.mem.bytesAsSlice(f32, req[o.wts..][0 .. 4 * l.s * rows]);
    for (0..rows) |r| for (0..l.d) |c| {
        const i = r * l.d + c;
        const xv: f32 = @bitCast(@as(u32, x[i]) << 16);
        var acc: f32 = 0;
        for (0..l.s) |j| {
            const e = pick[r * l.s + j];
            if (e >= lo and e < hi) acc = acc + wts[r * l.s + j] * (xv + @as(f32, @floatFromInt(e & 7)));
        }
        if (bf16_out) {
            const u: u32 = @bitCast(acc);
            std.mem.bytesAsSlice(u16, out)[i] = @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        } else std.mem.bytesAsSlice(f32, out)[i] = acc;
    };
}

/// Values (4 or 2 bytes) that differ.
pub fn wrongValues(a: []const u8, b: []const u8, width: usize) u64 {
    var n: u64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += width) n += @intFromBool(!std.mem.eql(u8, a[i..][0..width], b[i..][0..width]));
    return n;
}

/// Expert k's half of `experts`.
pub fn half(experts: u32, k: u32) struct { lo: i32, hi: i32 } {
    return .{ .lo = @intCast(k * experts / 2), .hi = @intCast((k + 1) * experts / 2) };
}

pub const GpuEngine = struct {
    d: *const cuda.Driver,
    g: xchg.Gpu,
    s: cuda.Stream,
    e0: cuda.Event,
    e1: cuda.Event,
};

/// A buffer the engine's kernels use: host bytes, and on a GPU their device copy.
pub const Buf = struct {
    host: []align(16) u8,
    dev: ?cuda.DeviceBuffer = null,

    pub fn ptr(b: Buf) u64 {
        return if (b.dev) |d| d.ptr else @intFromPtr(b.host.ptr);
    }
};

pub const Engine = union(enum) {
    gpu: *GpuEngine,
    cpu: *Cpu,

    pub fn alloc(e: Engine, gpa: Allocator, len: usize) !Buf {
        const host = try gpa.alignedAlloc(u8, .fromByteUnits(16), len);
        @memset(host, 0);
        return .{ .host = host, .dev = if (e == .gpu) try cuda.DeviceBuffer.alloc(e.gpu.d, len) else null };
    }

    pub fn free(_: Engine, gpa: Allocator, b: *Buf) void {
        if (b.dev) |*d| d.free();
        gpa.free(b.host);
    }

    /// The host bytes to the device, on the kernels' stream.
    pub fn upload(e: Engine, b: Buf, len: usize) !void {
        if (e == .gpu) try b.dev.?.uploadAsync(0, b.host[0..len], e.gpu.s.handle);
    }

    /// The device bytes back to the host, then a stream sync.
    pub fn download(e: Engine, b: Buf, len: usize) !void {
        if (e != .gpu) return;
        try b.dev.?.downloadAsync(0, b.host[0..len], e.gpu.s.handle);
        try e.gpu.s.synchronize();
    }

    pub fn post(e: Engine, n: *const xchg.Node, b: Buf, bytes: usize) !void {
        switch (e) {
            .gpu => |g| try g.g.post(n, b.ptr(), bytes, g.s),
            .cpu => |c| try c.post(n, b.host[0..bytes]),
        }
    }

    pub fn take(e: Engine, n: *const xchg.Node, bufs: []const Buf, bytes: []const usize) !void {
        switch (e) {
            .gpu => |g| {
                var o: [proto.EXPERTS]xchg.Out = undefined;
                for (bufs, bytes, 0..) |b, len, i| o[i] = .{ .ptr = b.ptr(), .bytes = len };
                try g.g.take(n, o[0..bufs.len], g.s);
            },
            .cpu => |c| {
                var o: [proto.EXPERTS][]u8 = undefined;
                for (bufs, bytes, 0..) |b, len, i| o[i] = b.host[0..len];
                try c.take(n, o[0..bufs.len]);
            },
        }
    }

    pub fn labExperts(e: Engine, req: Buf, l: proto.Layout, rows: usize, lo: i32, hi: i32, bf16_out: bool, out: Buf) !void {
        switch (e) {
            .gpu => |g| try g.g.labExperts(req.ptr(), l, rows, lo, hi, bf16_out, out.ptr(), g.s),
            .cpu => |c| c.labExperts(req.host, l, rows, lo, hi, bf16_out, out.host),
        }
    }

    pub fn sync(e: Engine) !void {
        if (e == .gpu) try e.gpu.s.synchronize();
    }
};

fn nowNs(io: std.Io) u64 {
    return @intCast(std.Io.Timestamp.now(io, .awake).toNanoseconds());
}

pub const AttnOpts = struct {
    layout: proto.Layout,
    experts: u32 = 288,
    layers: u32 = 42,
    rows: []const usize,
    repeats: usize = 3,
    /// rows from this many on are a prompt chunk (mode 3): bf16 replies when the layout says so
    prompt_from: usize = 64,
    verbose: bool = true,
};

pub const AttnReport = struct { exchanges: u64 = 0, wrong: u64 = 0, failed: ?xchg.Failure = null, fail_ns: u64 = 0, fast_after: u64 = 0, slow_after: u64 = 0 };

/// Buffers for the attention: the request and both replies at `max_rows`.
pub const AttnBufs = struct {
    req: Buf,
    rep: [proto.EXPERTS]Buf,
    ref: []align(16) u8,

    pub fn init(e: Engine, gpa: Allocator, l: proto.Layout, max_rows: usize) !AttnBufs {
        const rb = l.rep(max_rows, false);
        return .{ .req = try e.alloc(gpa, proto.wireBytes(l.req(max_rows))), .rep = .{ try e.alloc(gpa, rb), try e.alloc(gpa, rb) }, .ref = try gpa.alignedAlloc(u8, .fromByteUnits(16), rb) };
    }

    pub fn deinit(b: *AttnBufs, e: Engine, gpa: Allocator) void {
        e.free(gpa, &b.req);
        for (&b.rep) |*r| e.free(gpa, r);
        gpa.free(b.ref);
    }
};

/// One exchange of `rows` with forward seq `fseq`: request out, both replies back; with `check`, each reply against the reference (wrong values counted).
pub fn exchange(e: Engine, n: *xchg.Node, b: *AttnBufs, o: AttnOpts, fseq: i32, rows: usize, check: bool) !u64 {
    const l = o.layout;
    const prompt = rows >= o.prompt_from;
    const h: proto.Header = .{ .fseq = fseq, .rows = @intCast(rows), .layer = @mod(fseq, @as(i32, @intCast(o.layers))), .mode = if (prompt) .chunk else .eager, .layers = @intCast(o.layers) };
    const wire = proto.wireBytes(l.req(rows));
    if (comptime @import("builtin").is_test) if (e == .cpu) {
        e.cpu.observer.fseq = fseq; e.cpu.observer.layer = h.layer; e.cpu.observer.rows = h.rows;
        e.cpu.observer.seq = e.cpu.state[0] +% 1; e.cpu.observer.slot = n.reg.slotOf(e.cpu.observer.seq);
        e.cpu.observe(10, n);
    };
    fillRequest(l, o.experts, h, b.req.host);
    if (e == .cpu) e.cpu.observe(11, n);
    try e.upload(b.req, wire);
    if (e == .cpu) e.cpu.observe(12, n);
    const rb = l.rep(rows, prompt);
    try e.post(n, b.req, wire);
    if (e == .cpu) e.cpu.observe(13, n);
    try e.take(n, &b.rep, &.{ rb, rb });
    if (e == .cpu) e.cpu.observe(14, n);
    try e.sync();
    if (e == .cpu) e.cpu.observe(15, n);
    if (!check or n.failure() != null) return 0;
    var wrong: u64 = 0;
    for (0..proto.EXPERTS) |k| {
        try e.download(b.rep[k], rb);
        const hh = half(o.experts, @intCast(k));
        expertRef(l, b.req.host, rows, hh.lo, hh.hi, l.bf16 and prompt, b.ref);
        wrong += wrongValues(b.rep[k].host[0..rb], b.ref[0..rb], if (l.bf16 and prompt) 2 else 4);
    }
    return wrong;
}

/// The checked sweep: every row count `repeats` times, values new each time; stops at the first failure.
pub fn sweep(e: Engine, n: *xchg.Node, b: *AttnBufs, o: AttnOpts, fseq: *i32) !AttnReport {
    var rep: AttnReport = .{};
    for (o.rows) |rows| {
        var wrong: u64 = 0;
        for (0..o.repeats) |_| {
            fseq.* += 1;
            wrong += try exchange(e, n, b, o, fseq.*, rows, true);
            rep.exchanges += 1;
            if (n.failure()) |f| {
                rep.failed = f;
                return rep;
            }
        }
        rep.wrong += wrong;
        if (o.verbose) std.debug.print("{s} exact rows {d}: {d} exchanges, {d} wrong values\n", .{ if (wrong == 0) "PASS" else "FAIL", rows, o.repeats, wrong });
    }
    return rep;
}

pub const Timing = struct { single_p50_us: f64, single_p99_us: f64, layer_p50_us: f64, layer_min_us: f64 };

fn percentile(v: []f64, p: f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[@min(v.len - 1, @as(usize, @intFromFloat(p * @as(f64, @floatFromInt(v.len)))))];
}

/// Round trips at `rows`: `iters` single exchanges (post, take, sync each) and `forwards` forwards of `layers` exchanges back to back; GPU events, else the wall clock.
pub fn time(gpa: Allocator, io: std.Io, e: Engine, n: *xchg.Node, b: *AttnBufs, o: AttnOpts, rows: usize, iters: usize, forwards: usize, fseq: *i32) !Timing {
    const l = o.layout;
    const prompt = rows >= o.prompt_from;
    fseq.* += 1;
    const h: proto.Header = .{ .fseq = fseq.*, .rows = @intCast(rows), .layer = 0, .mode = if (prompt) .chunk else .eager, .layers = @intCast(o.layers) };
    const wire = proto.wireBytes(l.req(rows));
    fillRequest(l, o.experts, h, b.req.host);
    try e.upload(b.req, wire);
    const rb = l.rep(rows, prompt);
    const single = try gpa.alloc(f64, iters);
    defer gpa.free(single);
    const layer = try gpa.alloc(f64, forwards);
    defer gpa.free(layer);
    for (0..iters + 10) |i| {
        const t0 = nowNs(io);
        if (e == .gpu) try e.gpu.e0.record(e.gpu.s);
        try e.post(n, b.req, wire);
        try e.take(n, &b.rep, &.{ rb, rb });
        if (e == .gpu) try e.gpu.e1.record(e.gpu.s);
        try e.sync();
        if (n.failure() != null) return error.ExpertFailed;
        const us = if (e == .gpu) @as(f64, try cuda.Event.elapsedMs(e.gpu.e0, e.gpu.e1)) * 1000.0 else @as(f64, @floatFromInt(nowNs(io) - t0)) / 1000.0;
        if (i >= 10) single[i - 10] = us;
    }
    for (layer) |*v| {
        const t0 = nowNs(io);
        if (e == .gpu) try e.gpu.e0.record(e.gpu.s);
        for (0..o.layers) |_| {
            try e.post(n, b.req, wire);
            try e.take(n, &b.rep, &.{ rb, rb });
        }
        if (e == .gpu) try e.gpu.e1.record(e.gpu.s);
        try e.sync();
        if (n.failure() != null) return error.ExpertFailed;
        const us = if (e == .gpu) @as(f64, try cuda.Event.elapsedMs(e.gpu.e0, e.gpu.e1)) * 1000.0 else @as(f64, @floatFromInt(nowNs(io) - t0)) / 1000.0;
        v.* = us / @as(f64, @floatFromInt(o.layers));
    }
    const p50 = percentile(single, 0.5);
    const p99 = percentile(single, 0.99);
    const lmin = std.mem.min(f64, layer);
    return .{ .single_p50_us = p50, .single_p99_us = p99, .layer_p50_us = percentile(layer, 0.5), .layer_min_us = lmin };
}

/// After a failure: `count` more exchanges, each of which must return at once (under `fast_ns`) with the exchange still failed.
pub fn afterFailure(io: std.Io, e: Engine, n: *xchg.Node, b: *AttnBufs, o: AttnOpts, count: usize, fast_ns: u64, fseq: *i32) !struct { fast: u64, slow: u64 } {
    var fast: u64 = 0;
    var slow: u64 = 0;
    for (0..count) |_| {
        fseq.* += 1;
        const t0 = nowNs(io);
        if (e == .cpu) e.cpu.observe(30, n);
        _ = try exchange(e, n, b, o, fseq.*, 1, false);
        const t1 = nowNs(io);
        const quick = t1 - t0 < fast_ns and n.failure() != null;
        if (quick) fast += 1 else slow += 1;
        if (comptime @import("builtin").is_test) if (e == .cpu) {
            e.cpu.observe(31, n);
            e.cpu.observer.put(32, .{t0, t1, fast_ns, @intFromBool(n.cfg.run_watcher), n.proxies[0].?.ops(), n.proxies[1].?.ops(), n.cfg.give_up_ns, @intFromBool(quick)});
        };
    }
    return .{ .fast = fast, .slow = slow };
}

pub const ExpertOpts = struct {
    layout: proto.Layout,
    experts: u32 = 288,
    max_rows: usize,
    check: bool = true,
    /// silent: from this request on, never reply (the process, link and heartbeat live on)
    silent_at: ?u32 = null,
    /// exit: at this request the process dies (tf-xchg _exits 3; tests close the node)
    exit_at: ?u32 = null,
};

pub const ExpertEnd = struct { how: enum { stopped, lost, die }, why: []const u8 = "", requests: u64 = 0, rows: u64 = 0, wrong: u64 = 0 };

/// Expert k's loop: each request's header from its slot, take it, the stand-in half, post the reply; with `check`, the request against the lab's generator (after the reply is posted).
pub fn expertLoop(gpa: Allocator, e: Engine, n: *xchg.Node, o: ExpertOpts) !ExpertEnd {
    const l = o.layout;
    const k = n.role.expert;
    const hh = half(o.experts, k);
    var req = try e.alloc(gpa, proto.wireBytes(l.req(o.max_rows)));
    defer e.free(gpa, &req);
    var out = try e.alloc(gpa, l.rep(o.max_rows, false));
    defer e.free(gpa, &out);
    const expect = try gpa.alignedAlloc(u8, .fromByteUnits(16), proto.wireBytes(l.req(o.max_rows)));
    defer gpa.free(expect);
    var end: ExpertEnd = .{ .how = .stopped };
    var seq: u32 = 1;
    while (true) : (seq +%= 1) {
        if (e == .cpu) e.cpu.observe(52, n);
        const h = switch (n.waitRequest(seq)) {
            .request => |h| h,
            .stopped => break,
            .lost => |why| {
                end.how = .lost;
                end.why = why;
                break;
            },
        };
        if (comptime @import("builtin").is_test) if (e == .cpu) {
            e.cpu.observer.seq = seq; e.cpu.observer.slot = n.reg.slotOf(seq);
            e.cpu.observer.fseq = h.fseq; e.cpu.observer.layer = h.layer; e.cpu.observer.rows = h.rows;
            e.cpu.observe(53, n);
        };
        if (o.exit_at != null and o.exit_at.? == seq) {
            end.how = .die;
            break;
        }
        if (o.silent_at != null and o.silent_at.? == seq) {
            switch (n.idle()) {
                .lost => |why| {
                    end.how = .lost;
                    end.why = why;
                },
                else => {},
            }
            break;
        }
        const rows: usize = @intCast(h.rows);
        if (h.rows < 1 or rows > o.max_rows) {
            end.how = .lost;
            end.why = "a request with more rows than this expert's slots";
            break;
        }
        const wire = proto.wireBytes(l.req(rows));
        const prompt = h.mode != .eager;
        try e.take(n, &.{req}, &.{wire});
        try e.labExperts(req, l, rows, hh.lo, hh.hi, l.bf16 and prompt, out);
        try e.post(n, out, l.rep(rows, prompt));
        end.requests += 1;
        end.rows += rows;
        if (o.check) {
            fillRequest(l, o.experts, h, expect);
            end.wrong += wrongValues(n.received(n.src(0), seq, wire), expect[0..wire], 2);
        }
    }
    try e.sync();
    return end;
}
