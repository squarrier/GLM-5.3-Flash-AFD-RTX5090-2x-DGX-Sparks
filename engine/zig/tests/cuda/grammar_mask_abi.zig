//! grammar_mask real-ABI tests: grammar_abi's published libtfgrammar (the release v1.0.6 c080a18e shim over xgrammar v0.2.8),
//! loaded only by grammar_abi's trusted loader with its MANIFEST digest, owned through grammar_abi's tfg_owned, bound to a
//! request by lanes.grammar_owned and driven through the actual Lanes -> Bound -> Serve -> Attn -> sampler chain with
//! the real masked adapter. REAL: the library, xgrammar's masks over the real GLM-5.3 tokenizer, handle lifetimes.
//! MOCK: the device work (grammar_mask's CPU reference and topk's order over the STAGED words) and the forward (CPU
//! recording). Nothing here falls back: a missing library, digest or tokenizer fails the step.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const abi = @import("grammar_abi");
const fx = @import("grammar_mask_fixture.zig");
const Bound = @import("afd_serve_lanes.zig").Bound;
const a = std.testing.allocator;
const go = lanes.grammar_owned;
const O = go.Owned(abi.owned.Matcher);

/// GLM-5.3's tokenizer facts as grammar_abi's fixtures record them (fixtures/meta.json).
const vocab: u32 = 154880;
const stops = [_]u32{ 154820, 154827, 154829 };
const think_end: u32 = 154842; // "</think>", an added token
const lbrace: u32 = 90; // "{"
const rbrace: u32 = 92; // "}"
const pair: u32 = 6257; // "{}"
const hash: u32 = 2; // "#": never legal outside a JSON string

fn env(name: [:0]const u8) ![]const u8 {
    const v = std.c.getenv(name) orelse {
        std.debug.print("grammar_mask-ABI: {s} is unset; the real-library step never falls back\n", .{name});
        return error.MissingRealAbi;
    };
    return std.mem.span(v);
}

const Lib = struct {
    loader: abi.trusted.Loader = .{},
    counts: abi.owned.Counts = .{},
    tokenizer: []u8 = &.{},
    compiler: *abi.owned.Compiler = undefined,
    err: [512]u8 = undefined,

    fn open(l: *Lib) !void {
        const path = try env("TF_grammar_mask_LIB");
        const digest = try env("TF_grammar_mask_LIB_SHA256");
        const f = l.loader.get(path, digest) orelse {
            std.debug.print("grammar_mask-ABI: grammar_abi's loader refused {s}: {s} {s}\n", .{ path, if (l.loader.report.failure) |x| @tagName(x) else "?", l.loader.report.detail() });
            return error.LibraryRefused;
        };
        l.tokenizer = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try env("TF_grammar_mask_TOKENIZER"), a, .limited(64 << 20));
        errdefer a.free(l.tokenizer);
        l.compiler = abi.owned.Compiler.open(a, f, &l.counts, .{}, l.tokenizer, vocab, &stops, &l.err) catch |e| {
            std.debug.print("grammar_mask-ABI: tfg_open refused: {s} {s}\n", .{ @errorName(e), std.mem.sliceTo(&l.err, 0) });
            return e;
        };
        std.debug.print("grammar_mask-ABI library: {s} sha256 {s} loaded by grammar_abi's trusted loader; compiler words {d}\n", .{ path, digest, l.compiler.words });
    }
    fn close(l: *Lib) !void {
        try l.compiler.close(a);
        a.free(l.tokenizer);
    }
    /// An independent matcher of the same grammar: its masks are the oracle's.
    fn twin(l: *Lib, kind: u32, text: []const u8) !abi.owned.Matcher {
        var g = try l.compiler.compile(kind, text, &l.err);
        defer g.free() catch {};
        return g.matcher();
    }
};

fn bit(words: []const u32, t: u32) bool {
    return (words[t / 32] >> @as(u5, @intCast(t % 32))) & 1 != 0;
}

/// One structured request (JSON object, kind 0) through the chain with real Rules: each pick's unmasked winner "#"
/// is forbidden by the library's mask, the masked winners equal the oracle over the twin's masks, the staged words are
/// the library's, the request ends on the stop token the grammar allows, and lifetimes close in order.
fn run(marker: ?u32) !void {
    var lib: Lib = .{};
    try lib.open();
    var c: fx.Chain(Bound) = .{};
    try c.init(true);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, vocab), c.rec.m.vocab);
    const lead: []const fx.Pair = &.{ .{ .token = think_end, .value = 200 }, .{ .token = hash, .value = 100 } };
    const body = [_][]const fx.Pair{
        &.{ .{ .token = hash, .value = 100 }, .{ .token = lbrace, .value = 80 }, .{ .token = pair, .value = 60 } },
        &.{ .{ .token = hash, .value = 100 }, .{ .token = pair, .value = 90 }, .{ .token = rbrace, .value = 70 } },
        &.{ .{ .token = hash, .value = 100 }, .{ .token = rbrace, .value = 90 }, .{ .token = stops[0], .value = 50 } },
    };
    var picks: [4][]const fx.Pair = undefined;
    const n: usize = if (marker != null) 4 else 3;
    if (marker != null) picks[0] = lead;
    @memcpy(picks[n - 3 .. n], &body);
    var sched = try fx.Sched.init(vocab, picks[0..n]);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var o: O = undefined;
    const r = try go.request(abi.owned.Matcher, &o, lib.compiler, 0, "", &lib.err, vocab, marker);
    var s = try lanes.Stream.init(a, .{ .id = "grammar_mask_abi", .prompt = &.{ 3, 4 }, .max_new = 8, .rules = r });
    var live = true;
    defer if (live) s.deinit(a);
    try c.drive(&s);
    const want: []const u32 = if (marker != null) &.{ think_end, lbrace, rbrace, stops[0] } else &.{ lbrace, rbrace, stops[0] };
    try std.testing.expectEqualSlices(u32, want, s.emitted());
    // the oracle: an independent matcher fed the emitted tokens (never the marker) gives each masked pick's words
    var t = try lib.twin(0, "");
    const words = try a.alloc(u32, lib.compiler.words);
    defer a.free(words);
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    for (body, 0..) |p, i| {
        try t.fill(words);
        @memset(row, -1000);
        for (p) |x| row[x.token] = x.value;
        const expect = fx.oracle(row, words).?;
        if (bit(words, hash)) return error.WinnerNotForbidden;
        std.debug.print("WITNESS ForbiddenWinnerRuntimeViolation.grammar_mask-ABI-pick expected={d} observed={d}\n", .{ expect, s.emitted()[n - 3 + i] });
        if (expect != s.emitted()[n - 3 + i]) return error.ForbiddenWinnerRuntimeViolation;
        try std.testing.expectEqual(std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(words)), sched.staged_hashes[i]);
        var allowed: usize = 0;
        for (words) |w| allowed += @popCount(w);
        std.debug.print("grammar_mask-ABI pick {d}: {d} of {d} tokens allowed by the library, \"#\" forbidden, drew {d}\n", .{ i, allowed, vocab, expect });
        try std.testing.expect(try t.accept(expect));
    }
    try std.testing.expect(try t.terminated());
    try std.testing.expectEqual(@as(u64, 3), c.dev.stats.masked);
    try std.testing.expectEqual(@as(u64, if (marker != null) 1 else 0), c.dev.stats.plain);
    try std.testing.expectEqual(@as(usize, 3), try fx.maskedOrder(&c.rec.t));
    // lifetimes: the compiler cannot close while the request's or the twin's matcher lives
    try std.testing.expectError(error.Busy, lib.compiler.close(a));
    s.deinit(a);
    live = false;
    try std.testing.expect(o.freed and o.free_error == null);
    try t.free();
    try c.clean();
    // the next request is plain and unchanged on the same chain
    const plain_picks = [_][]const fx.Pair{&.{.{ .token = hash, .value = 100 }}};
    var ps = try fx.Sched.init(vocab, &plain_picks);
    defer ps.deinit();
    c.rec.t.picker = ps.picker();
    var p = try lanes.Stream.init(a, .{ .id = "grammar_mask_abi-plain", .prompt = &.{ 3, 4 }, .max_new = 1 });
    defer p.deinit(a);
    try c.drive(&p);
    try std.testing.expectEqualSlices(u32, &.{hash}, p.emitted());
    const counts = lib.counts;
    try lib.close();
    try std.testing.expectEqual(@as(u32, 0), lib.counts.live());
    std.debug.print("grammar_mask-ABI run marker={any}: emitted {any}; plain next {any}; grammar_abi counts before close: foreign_calls {d} matchers {d}/{d} compiled {d}/{d} poisoned {d}; live after close {d}\n", .{ marker, want, p.emitted(), counts.foreign_calls, counts.matchers_made, counts.matchers_freed, counts.compiled_made, counts.compiled_freed, counts.poisoned, lib.counts.live() });
}

test "grammar_mask-ABI real library masks through the actual chain: forbidden winners removed, stop allowed, lifetimes" {
    try run(null);
}

test "grammar_mask-ABI real library: the generated </think> activates the NEXT pick and never reaches the matcher" {
    try run(think_end);
}

test "grammar_mask-ABI real library: a compiler of another width is refused and its matcher freed" {
    var lib: Lib = .{};
    try lib.open();
    var o: O = undefined;
    try std.testing.expectError(error.WidthMismatch, go.request(abi.owned.Matcher, &o, lib.compiler, 0, "", &lib.err, vocab + 1, null));
    try std.testing.expect(o.freed and o.free_error == null);
    try std.testing.expectError(error.KindRange, go.request(abi.owned.Matcher, &o, lib.compiler, 6, "", &lib.err, vocab, null));
    try lib.close();
    try std.testing.expectEqual(@as(u32, 0), lib.counts.live());
}
