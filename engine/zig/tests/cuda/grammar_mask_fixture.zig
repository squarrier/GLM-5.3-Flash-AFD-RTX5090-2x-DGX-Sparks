//! grammar_mask CPU tests: the actual Bound -> Serve -> Attn -> sampler chain with the real masked adapter (cuda_mask_dev)
//! in its recorded mode. The device work is MOCK (the kernel's CPU reference, maskRow, and topkOrder read the STAGED
//! words); the matcher is a FAKE script; the host resolution, staging order, refusals and cleanup are the product code.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const gl = @import("glm_lanes");
const md = glm.sample.mask_dev;
const a = std.testing.allocator;

/// FAKE matcher: step i (accepted tokens so far) allows `steps[i]`; `all`: every token, never terminated.
pub const Script = struct {
    rules: lanes.grammar.Rules = undefined,
    steps: []const []const u32 = &.{},
    all: bool = false,
    accepted: usize = 0,
    fills: usize = 0,
    closes: usize = 0,
    fail_fill_at: ?usize = null,
    pub fn init(x: *Script, width: usize) void {
        x.rules = .{ .ptr = x, .width = width, .vtable = &.{ .fill = fill, .accept = accept, .rollback = rollback, .terminated = terminated, .close = close } };
    }
    fn of(p: *anyopaque) *Script {
        return @ptrCast(@alignCast(p));
    }
    pub fn expected(x: *const Script, i: usize, words: []u32) void {
        @memset(words, if (x.all) std.math.maxInt(u32) else 0);
        if (!x.all and i < x.steps.len) for (x.steps[i]) |t| {
            words[t / 32] |= @as(u32, 1) << @as(u5, @intCast(t % 32));
        };
    }
    fn fill(p: *anyopaque, words: []u32) !void {
        const x = of(p);
        x.fills += 1;
        if (x.fail_fill_at == x.fills) return error.MaskFillFailed;
        x.expected(x.accepted, words);
    }
    fn accept(p: *anyopaque, token: u32) !bool {
        const x = of(p);
        if (!x.all and (x.accepted >= x.steps.len or std.mem.indexOfScalar(u32, x.steps[x.accepted], token) == null)) return false;
        x.accepted += 1;
        return true;
    }
    fn rollback(p: *anyopaque, n: usize) !void {
        const x = of(p);
        if (n > x.accepted) return error.BadRollback;
        x.accepted -= n;
    }
    fn terminated(p: *anyopaque) bool {
        const x = of(p);
        return !x.all and x.accepted >= x.steps.len;
    }
    fn close(p: *anyopaque) void {
        of(p).closes += 1;
    }
};

pub const Pair = struct { token: u32, value: f32 };

/// FAKE library objects of grammar_abi's tfg_owned shape (Compiler.compile -> Compiled.matcher/free -> Matcher), no
/// library: the matcher answers from a Script and every call is logged in order ('c' compile, 'm' matcher, 'g' the
/// compiled grammar freed, 'f' fill, 'a' accept, 'r' rollback, 't' terminated, 'x' the matcher freed).
pub const Log = struct {
    order: [64]u8 = @splat('.'),
    n: usize = 0,
    frees: usize = 0,
    compiled_freed: usize = 0,
    fn add(l: *Log, c: u8) void {
        if (l.n < l.order.len) l.order[l.n] = c;
        l.n += 1;
    }
    pub fn seq(l: *const Log) []const u8 {
        return l.order[0..@min(l.n, l.order.len)];
    }
};
pub const Fail = enum { none, fill, accept, rollback, terminated, free };
pub const FakeCompiler = struct { vocab: u32 };
pub const FakeMatcher = struct {
    c: *const FakeCompiler,
    script: *Script,
    log: *Log,
    fail: Fail = .none,
    pub fn fill(m: *FakeMatcher, words: []u32) !void {
        m.log.add('f');
        if (m.fail == .fill) return error.Grammar;
        if (words.len != (m.c.vocab + 31) / 32) return error.WidthMismatch;
        return Script.fill(m.script, words);
    }
    pub fn accept(m: *FakeMatcher, token: u32) !bool {
        m.log.add('a');
        if (m.fail == .accept) return error.Poisoned;
        return Script.accept(m.script, token);
    }
    pub fn rollback(m: *FakeMatcher, n: usize) !void {
        m.log.add('r');
        if (m.fail == .rollback) return error.Poisoned;
        return Script.rollback(m.script, n);
    }
    pub fn terminated(m: *FakeMatcher) !bool {
        m.log.add('t');
        if (m.fail == .terminated) return error.Poisoned;
        return Script.terminated(m.script);
    }
    pub fn free(m: *FakeMatcher) !void {
        m.log.add('x');
        m.log.frees += 1;
        if (m.fail == .free) return error.AlreadyFreed;
    }
};
pub const FakeCompiled = struct {
    lib: *FakeLib,
    pub fn matcher(g: *FakeCompiled) !FakeMatcher {
        g.lib.log.add('m');
        if (g.lib.fail_matcher) return error.NoMatcher;
        return .{ .c = &g.lib.compiler, .script = g.lib.script, .log = &g.lib.log, .fail = g.lib.fail };
    }
    pub fn free(g: *FakeCompiled) !void {
        g.lib.log.add('g');
        g.lib.log.compiled_freed += 1;
    }
};
pub const FakeLib = struct {
    compiler: FakeCompiler,
    script: *Script,
    log: Log = .{},
    fail: Fail = .none,
    fail_matcher: bool = false,
    pub fn compile(l: *FakeLib, kind: u32, text: []const u8, err: *[512]u8) !FakeCompiled {
        _ = text;
        err[0] = 0;
        if (kind > 5) return error.KindRange;
        l.log.add('c');
        return .{ .lib = l };
    }
};

/// The logits of each pick in order (a row of -1000 with the listed values), read by the plain picker and the MOCK
/// device alike; each pick records what the device saw staged.
pub const Sched = struct {
    width: usize,
    picks: []const []const Pair,
    calls: usize = 0,
    row: []f32,
    staged_counts: [16]usize = @splat(0),
    staged_hashes: [16]u64 = @splat(0),
    device_calls: usize = 0,
    pub fn init(width: usize, picks: []const []const Pair) !Sched {
        return .{ .width = width, .picks = picks, .row = try a.alloc(f32, width) };
    }
    pub fn deinit(s: *Sched) void {
        a.free(s.row);
    }
    fn next(s: *Sched) ![]f32 {
        if (s.calls >= s.picks.len) return error.ScheduleExhausted;
        @memset(s.row, -1000);
        for (s.picks[s.calls]) |p| s.row[p.token] = p.value;
        s.calls += 1;
        return s.row;
    }
    pub fn picker(s: *Sched) glm.trace.Picker {
        return .{ .ctx = s, .pick = plain };
    }
    fn plain(ptr: *anyopaque, _: glm.trace.T, _: ?[]const u64, out: []u32, _: ?[]f64) !void {
        const s: *Sched = @ptrCast(@alignCast(ptr));
        const row = try s.next();
        var order: [1]usize = undefined;
        glm.sample.topkOrder(row, 1, &order);
        @memset(out, @intCast(order[0]));
    }
    pub fn emulator(s: *Sched) md.Emulator {
        return .{ .ctx = s, .candidate = device };
    }
    /// MOCK device: Ash's grammar_mask (its CPU reference) over the STAGED words, then topk k = 1.
    fn device(ptr: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const s: *Sched = @ptrCast(@alignCast(ptr));
        const row = try s.next();
        const staged = x.host[0..x.words];
        var bits: usize = 0;
        for (staged) |w| bits += @popCount(w);
        if (s.device_calls < s.staged_counts.len) {
            s.staged_counts[s.device_calls] = bits;
            s.staged_hashes[s.device_calls] = std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(staged));
        }
        s.device_calls += 1;
        md.maskRow(row, staged);
        var order: [1]usize = undefined;
        glm.sample.topkOrder(row, 1, &order);
        return .{ .id = @intCast(order[0]), .bits = @bitCast(row[order[0]]) };
    }
};

/// The device gate's MOCK row on CPU: the sink keeps the widened row; the plain picker and the MOCK device (the kernel's
/// CPU reference over the STAGED words, then topk's order) both read it. `honest = false`: a device ignoring the mask.
pub const GateRow = struct {
    row: []f32,
    tmp: []f32,
    honest: bool = true,
    pub fn init(width: usize) !GateRow {
        const row = try a.alloc(f32, width);
        errdefer a.free(row);
        return .{ .row = row, .tmp = try a.alloc(f32, width) };
    }
    pub fn deinit(g: *GateRow) void {
        a.free(g.row);
        a.free(g.tmp);
    }
    fn of(p: *anyopaque) *GateRow {
        return @ptrCast(@alignCast(p));
    }
    pub fn sink(g: *GateRow) md.Sink {
        return .{ .ctx = g, .put = put };
    }
    fn put(p: *anyopaque, bits: []const u16) !void {
        const g = of(p);
        for (bits, g.row) |b, *x| x.* = @bitCast(@as(u32, b) << 16);
    }
    pub fn picker(g: *GateRow) glm.trace.Picker {
        return .{ .ctx = g, .pick = plain };
    }
    fn plain(p: *anyopaque, _: glm.trace.T, _: ?[]const u64, out: []u32, _: ?[]f64) !void {
        var order: [1]usize = undefined;
        glm.sample.topkOrder(of(p).row, 1, &order);
        @memset(out, @intCast(order[0]));
    }
    pub fn emulator(g: *GateRow) md.Emulator {
        return .{ .ctx = g, .candidate = device };
    }
    fn device(p: *anyopaque, x: *const md.Dev, _: glm.trace.T) !md.Candidate {
        const g = of(p);
        @memcpy(g.tmp, g.row);
        if (g.honest) md.maskRow(g.tmp, x.host[0..x.words]);
        var order: [1]usize = undefined;
        glm.sample.topkOrder(g.tmp, 1, &order);
        return .{ .id = @intCast(order[0]), .bits = @bitCast(g.tmp[order[0]]) };
    }
};

/// One recorded attention chain with the adapter placed (installed only with `install`).
pub fn Chain(comptime Bound: type) type {
    return struct {
        const Self = @This();
        rec: glm.grammar_tests_rec.Rec = undefined,
        f: glm.fwd.Fwd = undefined,
        dev: md.Dev = undefined,
        serve: glm.afd_serve.Serve = undefined,
        bound: Bound = undefined,
        back: gl.Lanes(Bound) = undefined,
        cfg: lanes.Config = undefined,
        clock: lanes.fake.FixedClock = .{},
        core: lanes.Engine = undefined,
        pub fn init(c: *Self, install: bool) !void {
            try c.rec.init(a);
            errdefer c.rec.deinit(a);
            c.f = glm.fwd.Fwd.recording(&c.rec.m, &c.rec.t);
            c.dev = try md.Dev.recorded(a, &c.rec.m, &c.rec.space);
            errdefer c.dev.close();
            if (install) c.rec.e.smp.masked_dev = &c.dev;
            c.serve = .{ .at = &c.rec.at, .f = &c.f };
            c.bound = .{ .s = &c.serve };
            c.back = .{ .steps = &c.bound, .capacity = 2560, .reach = 2560, .frame_allocator = a };
            c.cfg = try lanes.Config.init(a, .{ .mtp = true }, 1, 8);
            errdefer c.cfg.deinit(a);
            c.core = lanes.Engine.init(a, &c.cfg, c.back.backend(), c.clock.clock());
        }
        pub fn deinit(c: *Self) void {
            c.core.deinit();
            c.cfg.deinit(a);
            c.dev.close();
            c.rec.deinit(a);
        }
        pub fn drive(c: *Self, s: *lanes.Stream) !void {
            try c.core.addStream(s);
            while (!s.finished) try c.core.step();
        }
        pub fn clean(c: *Self) !void {
            try std.testing.expect(c.back.owner == null and !c.serve.open and c.rec.at.deferred_logits == null and !c.back.first_pending);
        }
    };
}

pub fn count(t: *glm.trace.Trace, name: []const u8) usize {
    var n: usize = 0;
    for (t.events.items) |e| if (std.mem.eql(u8, e.name, name)) {
        n += 1;
    };
    return n;
}

/// Each masked pick's events, in order, around its grammar.mask: the staging fence first, the kernel before topk.
pub const masked_order = [_][]const u8{ "grammar.staged.synchronize", "grammar.host.write", "torch.copy_", "grammar.staged.record", "torch.float", "torch.fill_", "grammar.mask", "torch.topk", "torch.add", "torch.to", "grammar.invalid.cpu", "torch.cpu", "torch.cpu", "grammar.candidate" };

pub fn maskedOrder(t: *glm.trace.Trace) !usize {
    var n: usize = 0;
    for (t.events.items, 0..) |e, i| if (std.mem.eql(u8, e.name, "grammar.mask")) {
        const at = i - 6;
        for (masked_order, 0..) |name, k| {
            if (!std.mem.eql(u8, t.events.items[at + k].name, name)) {
                std.debug.print("MaskedOrderRuntimeViolation: event {d} is {s}, expected {s}\n", .{ at + k, t.events.items[at + k].name, name });
                return error.MaskedOrderRuntimeViolation;
            }
        }
        n += 1;
    };
    return n;
}

/// The independent oracle: the lowest eligible id whose topk key (NaN on top, -0 below +0) is the largest.
pub fn oracle(row: []const f32, mask: []const u32) ?u32 {
    var best: ?u32 = null;
    for (row, 0..) |v, t| {
        if ((mask[t / 32] & (@as(u32, 1) << @as(u5, @intCast(t % 32)))) == 0) continue;
        if (best == null or key(v) > key(row[best.?])) best = @intCast(t);
    }
    return best;
}

fn key(v: f32) u32 {
    const bits: u32 = @bitCast(v);
    if (bits & 0x7fff_ffff > 0x7f80_0000) return std.math.maxInt(u32);
    return if (bits & 0x8000_0000 != 0) ~bits else bits | 0x8000_0000;
}

pub fn eqTraces(x: *glm.trace.Trace, y: *glm.trace.Trace) !void {
    try x.flush();
    try y.flush();
    try std.testing.expectEqual(x.events.items.len, y.events.items.len);
    for (x.events.items, y.events.items) |p, q| try std.testing.expectEqualDeep(p, q);
}
