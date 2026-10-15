//! grammar_mask: structured output's device mask for the AFD attention role's greedy draw (world 1, one row, k = 1).
//! After the exact bf16->f32 widening, Ash Hart's torch_ops/grammar_mask.cu (cd7ebe98, unchanged) writes -inf at every
//! column whose bit is clear; the sampler's own topk/add/narrow/host rows then choose, and the host resolves validity
//! from the words it staged: an eligible winner stands; a winner outside the mask is legal only when every eligible
//! column is -inf (topk's tie order then names the lowest eligible column); an empty mask never launches. The staging
//! (pinned words, device words, a staged event) is owned here, outside the Engine's block, so plain buffers, launches
//! and operands are unchanged. Mask-before-draw follows Ash Hart's v1.0.6 tf_grammar_mask (c080a18e, sample.cu).
//! structured_users: the same staged words and kernel also serve each users row (TF_AFD_USERS, one row at a time) and the sampled
//! draw (Dev.sampled: the widened row masked before nucleus_rows' temperature and cuts).
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const sample = @import("cuda_sample.zig");
const nucleus = @import("cuda_nucleus.zig");
const cc = @import("cuda_compute.zig");
const views = @import("cuda_views.zig");
const trace = @import("cuda_trace.zig");
const md = @import("cuda_model.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;

const T = trace.T;
const Buf = views.Buf;

/// torch_ops/grammar_mask.cu's kernel (the image torch_grammar_mask).
pub const symbol = "tf_grammar_mask_kernel";
/// -inf as the kernel writes it into an f32 row
pub const neg_inf: u32 = 0xff80_0000;
const threads = 256;

/// A mask's facts from its words alone, before any launch: eligible count, lowest eligible id, every id eligible.
pub const Plan = struct { count: usize, first: u32, all: bool };

/// W = ceil(width / 32) words, no bit past the width; an empty mask is the reply's dead end (refused, no launch).
pub fn plan(width: usize, mask: []const u32) !Plan {
    const W = try sample.mask_cpu.words(width);
    if (mask.len != W) return error.MaskShape;
    const tail: u5 = @intCast(width % 32);
    if (tail != 0 and mask[W - 1] >> tail != 0) return error.MaskShape; // bits past the width never name a token
    var count: usize = 0;
    var first: ?u32 = null;
    for (mask, 0..) |word, i| {
        if (word != 0 and first == null) first = @intCast(i * 32 + @ctz(word));
        count += @popCount(word);
    }
    if (count == 0) return error.EmptyMask;
    return .{ .count = count, .first = first.?, .all = count == width };
}

/// The host's choice from topk's candidate (`id`, its value's bits) under the staged words: eligible, or every
/// eligible column -inf (the lowest eligible wins topk's tie order); anything else is a device/host disagreement.
pub fn resolve(p: Plan, width: usize, mask: []const u32, id: u32, value_bits: u32) !u32 {
    if (id >= width) return error.BadTokenId;
    if (sample.mask_cpu.eligible(mask, id)) return id;
    if (value_bits == neg_inf) return p.first;
    return error.MaskInconsistent;
}

/// The kernel's effect on one widened row (its CPU reference): a clear bit's column becomes -inf, others keep their bits.
pub fn maskRow(row: []f32, mask: []const u32) void {
    for (row, 0..) |*x, t| if (!sample.mask_cpu.eligible(mask, @intCast(t))) {
        x.* = @bitCast(neg_inf);
    };
}

/// topk's candidate as the host reads it back.
pub const Candidate = struct { id: u32, bits: u32 };

/// TF_AFD_GRAMMAR_MASK: 1 loads the adapter on the attention role (unset or 0: nothing loads, the plain path as today).
pub fn fromEnv() error{BadAfdGrammarMask}!bool {
    const v = std.mem.span(std.c.getenv("TF_AFD_GRAMMAR_MASK") orelse return false);
    if (std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    std.debug.print("INFO afd: TF_AFD_GRAMMAR_MASK takes 1 or 0 (or unset), not \"{s}\"\n", .{v});
    return error.BadAfdGrammarMask;
}

/// TF_AFD_GRAMMAR_GATE: the device gate's row count (1..4096) before serving, only with TF_AFD_GRAMMAR_MASK=1.
pub fn gateFromEnv(masked: bool) error{BadAfdGrammarGate}!?usize {
    const v = std.mem.span(std.c.getenv("TF_AFD_GRAMMAR_GATE") orelse return null);
    const n = std.fmt.parseInt(usize, v, 10) catch 0;
    if (!masked or n == 0 or n > 4096) {
        std.debug.print("INFO afd: TF_AFD_GRAMMAR_GATE takes 1..4096 rows and needs TF_AFD_GRAMMAR_MASK=1, not \"{s}\"\n", .{v});
        return error.BadAfdGrammarGate;
    }
    return n;
}

/// grammar_join_2's startup scope, narrowed by structured_users (Claude Opus 5.5 via Hermes): the masked adapter now draws each users
/// row under its own stream's mask (TF_AFD_USERS 2 or 4) and the sampled draw with the mask before the nucleus
/// (TF_GLM_SAMPLING=1), so neither is refused any more; the arguments stay for the role's one call site. Null: go.
pub fn scopeRefusal(masked: bool, users: u8, sampling: bool) ?[]const u8 {
    _ = masked;
    _ = users;
    _ = sampling;
    return null;
}

/// structured_users: a sampled draw's token under the staged words: eligible, or the device and the host disagree (never emitted).
pub fn drawn(width: usize, mask: []const u32, id: u32) !u32 {
    if (id >= width) return error.BadTokenId;
    if (!sample.mask_cpu.eligible(mask, id)) return error.MaskInconsistent;
    return id;
}

/// The independent host oracle of a masked greedy draw: among eligible columns the largest topk key (any NaN on top,
/// -0 below +0), a tie to the lowest column; null when nothing is eligible.
pub fn oracle(row: []const f32, mask: []const u32) ?u32 {
    var best: ?u32 = null;
    for (row, 0..) |v, t| {
        if (!sample.mask_cpu.eligible(mask, @intCast(t))) continue;
        if (best == null or key(v) > key(row[best.?])) best = @intCast(t);
    }
    return best;
}

fn key(v: f32) u32 {
    const b: u32 = @bitCast(v);
    if (b & 0x7fff_ffff > 0x7f80_0000) return std.math.maxInt(u32);
    return if (b & 0x8000_0000 != 0) ~b else b | 0x8000_0000;
}

/// One gate row's destination before its draws: the GPU's logits buffer, or a CPU test's MOCK row.
pub const Sink = struct { ctx: *anyopaque, put: *const fn (ctx: *anyopaque, bits: []const u16) anyerror!void };
pub const Gate = struct { rows: usize = 0, identity: usize = 0, forbidden: usize = 0, single: usize = 0, all_inf: usize = 0 };

/// grammar_mask's device gate (TF_AFD_GRAMMAR_GATE): `rows` seeded bf16 rows [1, V] through `logits` and the masked greedy
/// draw, each under four masks against the host oracle: all allowed (equal to the plain draw, which equals the host's
/// topk reference), the plain winner forbidden among a random eligible half, one eligible column, and eligible columns
/// all -inf (the lowest eligible). The first difference ends the gate with an error: the role never serves past it.
pub fn gate(x: *Dev, f: *Fwd, smp: *sample.Sampler, gpa: std.mem.Allocator, logits: T, sink: Sink, rows: usize, seed: u64) !Gate {
    const V = x.width;
    const bits = try gpa.alloc(u16, V);
    defer gpa.free(bits);
    const row = try gpa.alloc(f32, V);
    defer gpa.free(row);
    const mask = try gpa.alloc(u32, x.words);
    defer gpa.free(mask);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const specials = [_]u16{ 0x7fc0, 0xffc0, 0x7f80, 0xff80, 0x8000, 0x0000 }; // NaN, -NaN, +inf, -inf, -0, +0
    var g: Gate = .{};
    var out: [1]u32 = undefined;
    for (0..rows) |_| {
        for (bits) |*b| b.* = @truncate(@as(u32, @bitCast(r.float(f32) * 64 - 32)) >> 16);
        for (0..4) |_| bits[r.uintLessThan(usize, V)] = specials[r.uintLessThan(usize, specials.len)];
        widen(bits, row);
        try sink.put(sink.ctx, bits);
        try landed(f);
        // all allowed: the plain draw, itself the host's topk winner
        try sample.greedy(f, smp, logits, &out);
        const plain = out[0];
        var order: [1]usize = undefined;
        sample.topkOrder(row, 1, &order);
        try check("plain", plain, @intCast(order[0]));
        @memset(mask, 0xffff_ffff);
        clearTail(mask, V);
        try x.greedy(f, smp, logits, &out, mask);
        try check("identity", out[0], plain);
        g.identity += 1;
        // the plain winner forbidden among a random half
        for (mask) |*w| w.* = r.int(u32);
        clearTail(mask, V);
        mask[plain / 32] &= ~(@as(u32, 1) << @as(u5, @intCast(plain % 32)));
        if (std.mem.allEqual(u32, mask, 0)) set(mask, (plain + 1) % @as(u32, @intCast(V)));
        try x.greedy(f, smp, logits, &out, mask);
        try check("forbidden", out[0], oracle(row, mask).?);
        if (out[0] == plain) return error.GateForbiddenWinner;
        g.forbidden += 1;
        // one eligible column
        @memset(mask, 0);
        const e: u32 = @intCast((plain + 1 + r.uintLessThan(usize, V - 1)) % V);
        set(mask, e);
        try x.greedy(f, smp, logits, &out, mask);
        try check("single", out[0], e);
        g.single += 1;
        // eligible columns all -inf: every column -inf after the mask, the lowest eligible wins
        @memset(mask, 0);
        for (0..8) |_| {
            const t: u32 = @intCast(r.uintLessThan(usize, V));
            set(mask, t);
            bits[t] = 0xff80;
        }
        widen(bits, row);
        try sink.put(sink.ctx, bits);
        try landed(f);
        try x.greedy(f, smp, logits, &out, mask);
        try check("all-inf", out[0], oracle(row, mask).?);
        g.all_inf += 1;
        g.rows += 1;
    }
    return g;
}

/// grammar_mask_2: a sink's row has landed before the next draw (a pageable cuMemcpyHtoD can return before its DMA lands).
fn landed(f: *Fwd) !void {
    if (f.rec) |t| return t.call(.ext, "torch.cuda.synchronize", &.{});
    try sample.deviceSync(f);
}

fn check(comptime case: []const u8, got: u32, want: u32) !void {
    if (got == want) return;
    std.debug.print("grammar_mask gate FAIL {s}: drew {d}, the host oracle {d}\n", .{ case, got, want });
    return error.GateMismatch;
}

fn widen(bits: []const u16, row: []f32) void {
    for (bits, row) |b, *x| x.* = @bitCast(@as(u32, b) << 16);
}

fn set(mask: []u32, t: u32) void {
    mask[t / 32] |= @as(u32, 1) << @as(u5, @intCast(t % 32));
}

fn clearTail(mask: []u32, width: usize) void {
    const t: u5 = @intCast(width % 32);
    if (t != 0) mask[mask.len - 1] &= (@as(u32, 1) << t) - 1;
}

/// A CPU test's emulation of the device work (MOCK: the kernel's CPU reference and topkOrder), reading the staged words.
pub const Emulator = struct { ctx: *anyopaque, candidate: *const fn (ctx: *anyopaque, x: *const Dev, logits: T) anyerror!Candidate };
pub const Stats = struct { staged: u64 = 0, masked: u64 = 0, identity: u64 = 0, plain: u64 = 0, resolved_inf: u64 = 0, refused: u64 = 0, sampled: u64 = 0 };

pub const Dev = struct {
    width: usize,
    words: usize,
    /// the staged words (pinned on a GPU) and, after them, the invalid word read back
    host: []u32,
    host_rows: Buf,
    /// device words, the row index list {0} and the kernel's invalid word
    bits: Buf,
    rows: Buf,
    invalid: Buf,
    staged: ?cuda.Event = null,
    func: ?cuda.Function = null,
    module: ?cuda.Module = null,
    pinned: ?cuda.HostBuffer = null,
    block: ?cuda.DeviceBuffer = null,
    gpa: ?std.mem.Allocator = null,
    poisoned: bool = false,
    stats: Stats = .{},
    emulate: if (builtin.is_test) ?Emulator else void = if (builtin.is_test) null else {},

    /// On the AFD attention GPU (world 1): the kernel from `image`, the pinned and device words, the staged event.
    pub fn open(d: *const cuda.Driver, m: *const md.Model, image: []const u8) !Dev {
        if (m.world != 1 or m.rank != 0 or m.vl() != m.vocab) return error.MaskedWorld;
        const W = try sample.mask_cpu.words(m.vocab);
        if (m.vocab > 0xffff_ffff) return error.MaskShape;
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        const func = try module.function(symbol);
        var pinned = try cuda.HostBuffer.alloc(d, (W + 1) * 4);
        errdefer pinned.free();
        const at_rows = std.mem.alignForward(usize, W * 4, 256);
        var block = try cuda.DeviceBuffer.alloc(d, at_rows + 256);
        errdefer block.free();
        var staged = try cuda.Event.init(d, false);
        errdefer staged.deinit();
        try block.upload(at_rows, std.mem.asBytes(&@as(i64, 0))); // the one masked row is row 0
        const host = pinned.slice(u32)[0 .. W + 1];
        var x = layout(m.vocab, W, host);
        x.host_rows.ptr = @intFromPtr(host.ptr);
        x.bits.ptr = block.ptr;
        x.rows.ptr = block.ptr + at_rows;
        x.invalid.ptr = block.ptr + at_rows + 64;
        x.staged = staged;
        x.func = func;
        x.module = module;
        x.pinned = pinned;
        x.block = block;
        return x;
    }

    /// A CPU test's adapter over a recorder: host words from `gpa`, device buffers placed in the trace's Space.
    pub fn recorded(gpa: std.mem.Allocator, m: *const md.Model, space: *trace.Space) !Dev {
        if (!builtin.is_test) return error.IncompleteMaskedAdapter;
        if (m.world != 1 or m.rank != 0 or m.vl() != m.vocab) return error.MaskedWorld;
        const W = try sample.mask_cpu.words(m.vocab);
        const host = try gpa.alloc(u32, W + 1);
        errdefer gpa.free(host);
        @memset(host, 0);
        var x = layout(m.vocab, W, host);
        try space.place("tmp.grammar.host", &x.host_rows); // a named operand; the words themselves are in `host`
        try space.place("tmp.grammar.bits", &x.bits);
        try space.place("tmp.grammar.rows", &x.rows);
        try space.place("tmp.grammar.invalid", &x.invalid);
        x.gpa = gpa;
        return x;
    }

    fn layout(width: usize, W: usize, host: []u32) Dev {
        var h = Buf.of(.i32, &.{W});
        h.host = true;
        return .{ .width = width, .words = W, .host = host, .host_rows = h, .bits = Buf.of(.i32, &.{W}), .rows = Buf.of(.i64, &.{1}), .invalid = Buf.of(.i32, &.{1}) };
    }

    /// The last staged copy has left the pinned words before they are freed.
    pub fn close(x: *Dev) void {
        if (x.staged) |*e| {
            e.synchronize() catch {};
            e.deinit();
        }
        if (x.block) |*b| b.free();
        if (x.pinned) |*p| p.free();
        if (x.module) |*m| m.unload();
        if (x.gpa) |a| a.free(x.host);
        x.* = undefined;
    }

    /// Usable for `f`: not poisoned, world 1 over this width, a GPU's resources or (tests only) a recorder.
    pub fn ready(x: *const Dev, f: *const Fwd) !void {
        if (x.poisoned) return error.ControlPoison;
        if (f.m.world != 1 or f.m.rank != 0 or f.m.vl() != x.width or f.m.vocab != x.width) return error.MaskedWorld;
        if (f.gpu != null) {
            if (x.func == null or x.staged == null or x.block == null or x.pinned == null) return error.IncompleteMaskedAdapter;
        } else if (f.rec == null or !builtin.is_test or x.gpa == null) return error.IncompleteMaskedAdapter;
    }

    /// Before a masked step's forward: an empty or malformed mask is refused with no launch.
    pub fn precheck(x: *Dev, mask: []const u32) !void {
        if (x.poisoned) return error.ControlPoison;
        if (mask.len == 0) return;
        _ = plan(x.width, mask) catch |err| {
            x.stats.refused += 1;
            return err;
        };
    }

    /// The greedy draw of `logits` [1, V] among `mask`'s tokens (empty: unconstrained). No mask, or every token
    /// allowed, is exactly the plain draw (sample.greedy: same launches and operands).
    pub fn greedy(x: *Dev, f: *Fwd, smp: *sample.Sampler, logits: T, out: []u32, mask: []const u32) !void {
        try x.ready(f);
        const V = x.width;
        if (logits.dims != 2 or logits.shape[0] != 1 or logits.shape[1] != V or logits.stride[0] != V or logits.stride[1] != 1 or out.len != 1) return error.MaskShape;
        if (mask.len == 0) {
            x.stats.plain += 1;
            return sample.greedy(f, smp, logits, out);
        }
        const p = plan(V, mask) catch |err| {
            x.stats.refused += 1;
            return err;
        };
        if (p.all) {
            x.stats.identity += 1;
            return sample.greedy(f, smp, logits, out);
        }
        out[0] = x.select(f, smp, logits, mask, p) catch |err| {
            x.poisoned = true; // past the plan only device work can fail: every later masked draw refuses
            return err;
        };
        x.stats.masked += 1;
    }

    /// structured_users: the sampled draw (nucleus_rows, TF_GLM_SAMPLING=1) of `logits` [1, V] at `position` among `mask`'s tokens:
    /// the staged words' grammar_mask on the widened row before the temperature, the maxima, the masses and the
    /// candidates (a clear bit's column is -inf: no mass, so outside every nucleus and never drawn), then the drawn token
    /// checked eligible on the host. No mask, or every token allowed, is exactly nucleus_rows (same launches/operands).
    /// The nucleus's own refusals (a row without mass) end the request; only a device/host disagreement poisons.
    pub fn sampled(x: *Dev, f: *Fwd, smp: *sample.Sampler, n: *const nucleus.Nucleus, logits: T, position: u64, s: nucleus.Sampling, out: []u32, stats: *nucleus.Stats, mask: []const u32) !void {
        try x.ready(f);
        const V = x.width;
        if (logits.dims != 2 or logits.shape[0] != 1 or logits.shape[1] != V or logits.stride[0] != V or logits.stride[1] != 1 or out.len != 1) return error.MaskShape;
        const positions = [1]u64{position};
        if (mask.len == 0) {
            x.stats.plain += 1;
            return nucleus.nucleusRows(f, smp, n, logits, &positions, s, out, null, stats);
        }
        const p = plan(V, mask) catch |err| {
            x.stats.refused += 1;
            return err;
        };
        if (p.all) {
            x.stats.identity += 1;
            return nucleus.nucleusRows(f, smp, n, logits, &positions, s, out, null, stats);
        }
        var w: Widen = .{ .x = x, .mask = mask };
        try nucleus.nucleusRowsWith(f, smp, n, logits, &positions, s, out, null, stats, .{ .ctx = &w, .apply = Widen.apply });
        if (f.rec == null and x.host[x.words] != 0) { // the kernel's invalid word, landed with the packet's copy
            x.poisoned = true;
            return error.MaskRowInvalid;
        }
        out[0] = drawn(V, mask, out[0]) catch |err| {
            x.poisoned = true;
            return err;
        };
        x.stats.sampled += 1;
    }

    /// structured_users: nucleus_rows' widened row [1, V] masked by `mask` (staged, the kernel, its invalid word queued to the host).
    const Widen = struct {
        x: *Dev,
        mask: []const u32,
        fn apply(ctx: *anyopaque, f: *Fwd, row: T) anyerror!void {
            const w: *Widen = @ptrCast(@alignCast(ctx));
            try w.x.stage(f, w.mask);
            try w.x.eligible(f, row);
            try w.x.invalidToHost(f);
        }
    };

    fn select(x: *Dev, f: *Fwd, smp: *sample.Sampler, logits: T, mask: []const u32, p: Plan) !u32 {
        const s = &smp.s;
        const V = x.width;
        if (V != s.float.shape[1] or s.topk.values.shape[1] < 1) return error.SampleShape;
        try x.stage(f, mask);
        try sample.made(smp, "float", &s.float);
        const row = T.of(s.float, 0, &.{ 1, V });
        try sample.toFloat(f, smp, logits, row);
        try x.eligible(f, row);
        if (smp.tmp) |t| try t.madeFields("topk", &.{ "values", "indices" }, &.{ &s.topk.values, &s.topk.indices });
        const vals = T.of(s.topk.values, 0, &.{ 1, 1 });
        try sample.topk(f, smp, row, 1, vals, T.of(s.topk.indices, 0, &.{ 1, 1 }));
        try sample.made(smp, "add", &s.add);
        try sample.add(f, smp, T.of(s.topk.indices, 0, &.{ 1, 1 }), f.m.rank * f.m.vl(), T.of(s.add, 0, &.{ 1, 1 }));
        try sample.made(smp, "to", &s.to);
        const ids = T.of(s.to, 0, &.{ 1, 1 });
        try sample.narrow(f, smp, T.of(s.add, 0, &.{ 1, 1 }), ids);
        if (s.cpu.ptr == 0) return error.NoHostRows;
        try x.invalidToHost(f); // queued before hostRows' copy and stream sync
        const host: [*]u8 = @ptrFromInt(s.cpu.ptr);
        try sample.hostRows(f, smp, vals, ids, host);
        const c = try x.candidate(f, smp, vals, ids, host, logits);
        const id = try resolve(p, V, mask, c.id, c.bits);
        if (id != c.id) x.stats.resolved_inf += 1;
        return id;
    }

    fn candidate(x: *Dev, f: *Fwd, smp: *sample.Sampler, vals: T, ids: T, host: [*]u8, logits: T) !Candidate {
        if (f.rec) |t| {
            if (comptime !builtin.is_test) return error.NoRecordedCandidate;
            const e = x.emulate orelse return error.NoRecordedCandidate;
            const c = try e.candidate(e.ctx, x, logits);
            try t.call(.ext, "grammar.candidate", &.{ .{ .int = c.id }, .{ .int = c.bits } });
            return c;
        }
        const words: [*]const u32 = @ptrCast(@alignCast(host));
        if (!sample.idsInShards(1, x.width, words, 1, 1)) { // Mia's 0092, as greedyCut: once more after the device idles
            try sample.deviceSync(f);
            try sample.hostRows(f, smp, vals, ids, host);
            if (!sample.idsInShards(1, x.width, words, 1, 1)) return error.BadTokenId;
        }
        if (x.host[x.words] != 0) return error.MaskRowInvalid;
        return .{ .id = @intCast(@as(i32, @bitCast(words[1]))), .bits = words[0] };
    }

    /// cc.stageIds' rule for the words: the last copy left the pinned rows, the words written, copied, the event recorded.
    fn stage(x: *Dev, f: *Fwd, mask: []const u32) !void {
        if (f.rec) |t| try t.call(.ext, "grammar.staged.synchronize", &.{}) else try x.staged.?.synchronize();
        @memcpy(x.host[0..x.words], mask);
        if (f.rec) |t| try t.call(.ext, "grammar.host.write", &.{ .{ .t = T.whole(x.host_rows) }, .{ .int = @intCast(std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(mask)) >> 1) } });
        try cc.copyHtoD(f, T.whole(x.bits), T.whole(x.host_rows));
        if (f.rec) |t| try t.call(.ext, "grammar.staged.record", &.{}) else try x.staged.?.record((f.gpu orelse return error.NoGpu).stream);
        x.stats.staged += 1;
    }

    /// grammar_mask over the widened row: one row (index 0), V columns, f32 elements; its invalid word cleared first.
    fn eligible(x: *Dev, f: *Fwd, row: T) !void {
        try cc.fill(f, T.whole(x.invalid), 0);
        const V = sample.w64(x.width);
        const packed_bytes = sample.w64(x.words * 4);
        if (f.rec) |t| return t.call(.ext, "grammar.mask", &.{ .{ .t = row }, .{ .t = T.whole(x.bits) }, .{ .t = T.whole(x.rows) }, .{ .int = 1 }, .{ .int = 1 }, .{ .int = @intCast(V) }, .{ .int = @intCast(packed_bytes) }, .{ .int = 0 }, .{ .int = 4 }, .{ .t = T.whole(x.invalid) } });
        try sample.go(f, x.func.?, .{ 1, 1 }, threads, .{ row.ptr, x.bits.ptr, x.rows.ptr, sample.w64(1), sample.w64(1), V, packed_bytes, sample.w64(0), @as(u32, 4), x.invalid.ptr });
    }

    fn invalidToHost(x: *Dev, f: *Fwd) !void {
        if (f.rec) |t| return t.call(.ext, "grammar.invalid.cpu", &.{.{ .t = T.whole(x.invalid) }});
        const g = f.gpu orelse return error.NoGpu;
        const d = g.stream.d;
        try d.check(d.api.cuMemcpyDtoHAsync_v2(@ptrCast(&x.host[x.words]), x.invalid.ptr, 4, g.stream.handle), "cuMemcpyDtoHAsync");
    }
};
