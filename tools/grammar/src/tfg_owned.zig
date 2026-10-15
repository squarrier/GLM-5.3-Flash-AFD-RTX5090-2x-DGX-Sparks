//! grammar_abi ownership and admission over the release libtfgrammar ABI: every bound is checked before the foreign
//! call it protects, every handle is counted and freed once, a compiler with live children cannot close, and a
//! matcher whose rollback/drafts failed is poisoned (freed, never reused). Claude Opus 5.5 via Hermes (local integration).
const std = @import("std");
const trusted = @import("tfg_trusted.zig");
const Fns = trusted.Fns;

pub const Limits = struct {
    max_vocab: u32 = 4194304,
    max_stops: usize = 64,
    max_tokenizer_bytes: usize = 256 << 20, // the release server's tokenizer.json read limit
    max_grammar_bytes: usize = 1 << 20,
    max_live_matchers: u32 = 64,
};

pub const Counts = struct {
    foreign_calls: u64 = 0,
    refused: u64 = 0,
    compilers_opened: u32 = 0,
    compilers_closed: u32 = 0,
    compiled_made: u32 = 0,
    compiled_freed: u32 = 0,
    matchers_made: u32 = 0,
    matchers_freed: u32 = 0,
    poisoned: u32 = 0,

    pub fn live(c: Counts) u32 {
        return (c.compilers_opened - c.compilers_closed) + (c.compiled_made - c.compiled_freed) + (c.matchers_made - c.matchers_freed);
    }
};

pub const Refusal = error{ VocabBounds, StopBounds, TokenizerTooLarge, GrammarTooLarge, KindRange, TooManyMatchers, Busy, Closed, Poisoned, TokenRange, WidthMismatch, AlreadyFreed, GrammarSetup, Grammar, NoMatcher, ProbeMismatch, OutOfMemory };

pub const Compiler = struct {
    f: *const Fns,
    h: ?*anyopaque,
    vocab: u32,
    words: usize,
    limits: Limits,
    counts: *Counts,
    live_compiled: u32 = 0,
    live_matchers: u32 = 0,
    mutex: std.atomic.Mutex = .unlocked,

    /// tfg_open after the admission bounds; the words probe must equal ceil(vocab/32) or the compiler is closed.
    pub fn open(gpa: std.mem.Allocator, f: *const Fns, counts: *Counts, limits: Limits, tokenizer_json: []const u8, vocab: u32, stops: []const u32, err: *[512]u8) Refusal!*Compiler {
        err[0] = 0;
        errdefer counts.refused += 1;
        if (vocab == 0 or vocab > limits.max_vocab) return error.VocabBounds;
        if (stops.len == 0 or stops.len > limits.max_stops) return error.StopBounds;
        for (stops) |s| if (s >= vocab) return error.StopBounds;
        if (tokenizer_json.len > limits.max_tokenizer_bytes) return error.TokenizerTooLarge;
        const c = try gpa.create(Compiler);
        errdefer gpa.destroy(c);
        counts.foreign_calls += 1;
        const h = f.open(tokenizer_json.ptr, tokenizer_json.len, @intCast(vocab), @ptrCast(stops.ptr), @intCast(stops.len), err, err.len) orelse return error.GrammarSetup;
        counts.compilers_opened += 1;
        counts.foreign_calls += 1;
        const w = f.words(h);
        if (w < 0 or @as(usize, @intCast(w)) != (@as(usize, vocab) + 31) / 32) {
            counts.foreign_calls += 1;
            f.close(h);
            counts.compilers_closed += 1;
            return error.ProbeMismatch;
        }
        c.* = .{ .f = f, .h = h, .vocab = vocab, .words = @intCast(w), .limits = limits, .counts = counts };
        return c;
    }

    /// Refused while any compiled grammar or matcher made from it is live (shutdown waits for its requests).
    pub fn close(c: *Compiler, gpa: std.mem.Allocator) Refusal!void {
        while (!c.mutex.tryLock()) std.atomic.spinLoopHint();
        const h = c.h orelse {
            c.mutex.unlock();
            c.counts.refused += 1;
            return error.Closed;
        };
        if (c.live_compiled + c.live_matchers > 0) {
            c.mutex.unlock();
            c.counts.refused += 1;
            return error.Busy;
        }
        c.counts.foreign_calls += 1;
        c.f.close(h);
        c.h = null;
        c.counts.compilers_closed += 1;
        c.mutex.unlock();
        gpa.destroy(c);
    }

    /// tfg_compile for kind 0..5 (the shim compiles any other kind as EBNF, so the range is checked here).
    pub fn compile(c: *Compiler, kind: u32, text: []const u8, err: *[512]u8) Refusal!Compiled {
        err[0] = 0;
        while (!c.mutex.tryLock()) std.atomic.spinLoopHint();
        defer c.mutex.unlock();
        const h = c.h orelse return c.refuse(error.Closed);
        if (kind > 5) return c.refuse(error.KindRange);
        if (text.len > c.limits.max_grammar_bytes) return c.refuse(error.GrammarTooLarge);
        c.counts.foreign_calls += 1;
        const g = c.f.compile(h, @intCast(kind), text.ptr, text.len, err, err.len) orelse return error.Grammar;
        c.counts.compiled_made += 1;
        c.live_compiled += 1;
        return .{ .c = c, .h = g };
    }

    fn refuse(c: *Compiler, e: Refusal) Refusal {
        c.counts.refused += 1;
        return e;
    }
};

pub const Compiled = struct {
    c: *Compiler,
    h: ?*anyopaque,

    pub fn free(g: *Compiled) Refusal!void {
        const h = g.h orelse return g.c.refuse(error.AlreadyFreed);
        g.c.counts.foreign_calls += 1;
        g.c.f.free(h);
        g.h = null;
        g.c.counts.compiled_freed += 1;
        g.c.live_compiled -= 1;
    }

    /// A fresh matcher; it does not borrow this compiled grammar (the release frees the grammar right after).
    pub fn matcher(g: *Compiled) Refusal!Matcher {
        const h = g.h orelse return g.c.refuse(error.AlreadyFreed);
        if (g.c.live_matchers >= g.c.limits.max_live_matchers) return g.c.refuse(error.TooManyMatchers);
        g.c.counts.foreign_calls += 1;
        const m = g.c.f.matcher(h) orelse return error.NoMatcher;
        g.c.counts.matchers_made += 1;
        g.c.live_matchers += 1;
        return .{ .c = g.c, .h = m };
    }
};

pub const Matcher = struct {
    c: *Compiler,
    h: ?*anyopaque,
    poisoned: bool = false,

    fn handle(m: *Matcher) Refusal!*anyopaque {
        const h = m.h orelse return m.c.refuse(error.AlreadyFreed);
        if (m.poisoned) return m.c.refuse(error.Poisoned);
        return h;
    }

    /// true taken, false rejected; an id outside the vocabulary is refused before the call.
    pub fn accept(m: *Matcher, token: u32) Refusal!bool {
        const h = try m.handle();
        if (token >= m.c.vocab) return m.c.refuse(error.TokenRange);
        m.c.counts.foreign_calls += 1;
        return switch (m.c.f.accept(h, @intCast(token))) {
            1 => true,
            0 => false,
            else => m.poison(),
        };
    }

    pub fn rollback(m: *Matcher, n: usize) Refusal!void {
        const h = try m.handle();
        if (n == 0) return;
        m.c.counts.foreign_calls += 1;
        if (m.c.f.rollback(h, @intCast(n)) != 0) return m.poison();
    }

    pub fn fill(m: *Matcher, words: []u32) Refusal!void {
        const h = try m.handle();
        if (words.len != m.c.words) return m.c.refuse(error.WidthMismatch);
        m.c.counts.foreign_calls += 1;
        if (m.c.f.fill(h, @ptrCast(words.ptr), @intCast(words.len)) != 0) return error.Grammar;
    }

    pub fn drafts(m: *Matcher, out: []u32) Refusal!usize {
        const h = try m.handle();
        m.c.counts.foreign_calls += 1;
        const n = m.c.f.drafts(h, @ptrCast(out.ptr), @intCast(out.len));
        if (n < 0) return m.poison();
        return @intCast(n);
    }

    pub fn terminated(m: *Matcher) Refusal!bool {
        const h = try m.handle();
        m.c.counts.foreign_calls += 1;
        return m.c.f.terminated(h) == 1;
    }

    pub fn free(m: *Matcher) Refusal!void {
        const h = m.h orelse return m.c.refuse(error.AlreadyFreed);
        m.c.counts.foreign_calls += 1;
        m.c.f.matcher_free(h);
        m.h = null;
        m.c.counts.matchers_freed += 1;
        m.c.live_matchers -= 1;
    }

    fn poison(m: *Matcher) Refusal {
        m.poisoned = true;
        m.c.counts.poisoned += 1;
        return error.Poisoned;
    }
};
