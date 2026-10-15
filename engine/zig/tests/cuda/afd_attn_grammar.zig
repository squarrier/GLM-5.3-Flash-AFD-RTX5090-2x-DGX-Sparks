//! grammar_server (Claude Opus 5.5 via Hermes): the AFD attention role's served structured output, default off, as the
//! planner's evidence/grammar_mask/ANSWER.md (P3pass135) items 1-5 decide:
//! 1. TF_AFD_GRAMMAR_LIB (absolute path) and TF_AFD_GRAMMAR_LIB_SHA256 (the MANIFEST digest), both or neither, read once
//!    at the role's open (afd_attn_gpu.zig), feed grammar_abi's trusted loader unchanged; the loader runs only for a
//!    structured request (ABI-CONTRACT 7.8), so a relative path or a wrong digest is its named, sticky refusal.
//! 2. The x86_64 artifact of evidence/grammar_abi/artifacts on the attention host only; experts never read any of this.
//! 3. One compiler per attention-role server, opened lazily by the server's first structured request (release
//!    server.zig:204-235: the checkpoint's tokenizer.json, config.json's vocab_size, the eos ids) with 7.9's words
//!    probe (tfg_owned.Compiler.open); a failed open is sticky (the server keeps its problem). Matchers stay
//!    request-owned (lanes.grammar_owned); the role closes the compiler after its lane host stopped, and grammar_abi's
//!    ownership refuses that (Busy) while any matcher lives. The tokenizer's sha256 is recorded, never judged (H05 is
//!    grammar_tests_c11's).
//! 4. response_format -> (kind, text) is the server's, unchanged from the release (grammar.zig:56-67).
//! 5. TF_AFD_GRAMMAR_SERVE=1 (needs TF_AFD_GRAMMAR_MASK=1) is the only switch that sets Info.structures.
//! Every call into the compiler and its matchers holds this role's lock: grammar_abi's counts are plain integers, and
//! the server's threads check grammars while the lane host's thread fills masks.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const abi = @import("grammar_abi");

/// grammar_abi's loader and ownership exist only in a build given -Dgram2abi (its read-only view root).
pub const has_abi = @hasDecl(abi, "owned") and @hasDecl(abi, "trusted");

/// The library pair as read at the role's open (null: unset).
pub const Env = struct { lib: ?[]const u8 = null, digest: ?[]const u8 = null };
pub const Error = error{ BadAfdGrammarServe, AfdGrammarLibUnpaired, AfdGrammarServeWithoutMask };

/// The three switches: the pair first (one without the other refuses), then SERVE (1 or 0), which needs MASK; null
/// unless SERVE=1. Nothing is loaded or opened here.
pub fn parse(lib: ?[]const u8, digest: ?[]const u8, serve: ?[]const u8, masked: bool) Error!?Env {
    if ((lib == null) != (digest == null)) {
        std.debug.print("INFO afd: TF_AFD_GRAMMAR_LIB and TF_AFD_GRAMMAR_LIB_SHA256 go together: one is set without the other\n", .{});
        return error.AfdGrammarLibUnpaired;
    }
    const on = if (serve) |v| blk: {
        if (std.mem.eql(u8, v, "1")) break :blk true;
        if (std.mem.eql(u8, v, "0")) break :blk false;
        std.debug.print("INFO afd: TF_AFD_GRAMMAR_SERVE takes 1 or 0 (or unset), not \"{s}\"\n", .{v});
        return error.BadAfdGrammarServe;
    } else false;
    if (!on) return null;
    if (!masked) {
        std.debug.print("INFO afd: TF_AFD_GRAMMAR_SERVE=1 needs TF_AFD_GRAMMAR_MASK=1 (the masked adapter that enforces the grammar)\n", .{});
        return error.AfdGrammarServeWithoutMask;
    }
    return .{ .lib = lib, .digest = digest };
}

/// TF_AFD_GRAMMAR_LIB, TF_AFD_GRAMMAR_LIB_SHA256 and TF_AFD_GRAMMAR_SERVE, read once before anything loads.
pub fn fromEnv(masked: bool) Error!?Env {
    return parse(get("TF_AFD_GRAMMAR_LIB"), get("TF_AFD_GRAMMAR_LIB_SHA256"), get("TF_AFD_GRAMMAR_SERVE"), masked);
}

fn get(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The loader the role feeds: grammar_abi's process-wide `trusted.global` in the tool (sticky for the process).
pub const Loader = if (has_abi) *abi.trusted.Loader else void;

/// The attention role's grammar source (the server's lazy open and check) and its lane host's Rules.
pub const Served = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    env: Env,
    /// The model's logits width: a request's matcher must be over exactly this vocabulary.
    width: usize,
    loader: Loader,
    counts: if (has_abi) abi.owned.Counts else void = if (has_abi) .{} else {},
    compiler: if (has_abi) ?*abi.owned.Compiler else ?*anyopaque = null,
    lock: std.Io.Mutex = .init,
    /// Opens asked for: the server asks once (its first structured request), then keeps the compiler or the problem.
    opens: u32 = 0,
    tokenizer_sha256: ?[64]u8 = null,
    problem_buf: [400]u8 = undefined,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, env: Env, width: usize, loader: Loader) Served {
        return .{ .gpa = gpa, .io = io, .env = env, .width = width, .loader = loader };
    }

    pub fn source(s: *Served) api.Structure.Source {
        return .{ .ctx = s, .open = open, .check = check };
    }

    pub fn structured(s: *Served) api.LaneHost.Structured {
        return .{ .ctx = s, .rules = rules };
    }

    /// After the lane host stopped: frees the compiler, refused (Busy, the compiler kept) while a matcher lives.
    pub fn close(s: *Served) !void {
        if (comptime has_abi) {
            s.lock.lockUncancelable(s.io);
            defer s.lock.unlock(s.io);
            const c = s.compiler orelse return;
            try c.close(s.gpa);
            s.compiler = null;
        }
    }

    fn of(ctx: *anyopaque) *Served {
        return @ptrCast(@alignCast(ctx));
    }

    fn open(ctx: *anyopaque, tokenizer_json: []const u8, vocab: u32, stops: []const u32, problem: *[]const u8) ?*api.Structure.Compiler {
        const s = of(ctx);
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        s.opens += 1;
        if (comptime !has_abi) {
            problem.* = "the server cannot load libtfgrammar: " ++ abi.absent;
            return null;
        } else {
            const lib = s.env.lib orelse {
                problem.* = "the server cannot load libtfgrammar: TF_AFD_GRAMMAR_LIB and TF_AFD_GRAMMAR_LIB_SHA256 are unset";
                return null;
            };
            const fns = s.loader.get(lib, s.env.digest.?) orelse {
                const r = &s.loader.report;
                problem.* = std.fmt.bufPrint(&s.problem_buf, "the server cannot load libtfgrammar: {s} ({s})", .{ if (r.failure) |f| @tagName(f) else "refused", r.detail() }) catch "the server cannot load libtfgrammar";
                return null;
            };
            var err: [512]u8 = undefined;
            err[0] = 0;
            const c = abi.owned.Compiler.open(s.gpa, fns, &s.counts, .{}, tokenizer_json, vocab, stops, &err) catch |e| {
                const words = std.mem.sliceTo(&err, 0);
                problem.* = std.fmt.bufPrint(&s.problem_buf, "libtfgrammar refused the checkpoint's tokenizer: {s}{s}{s}", .{ @errorName(e), if (words.len > 0) ": " else "", words }) catch "libtfgrammar refused the checkpoint's tokenizer";
                return null;
            };
            s.compiler = c;
            var sum: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(tokenizer_json, &sum, .{});
            s.tokenizer_sha256 = std.fmt.bytesToHex(sum, .lower);
            std.debug.print("INFO attn: grammar_server structured output: libtfgrammar {s} (sha256 {s}, grammar_abi's trusted loader) over tokenizer.json sha256 {s}, vocab {d}, {d} words a row; the vocabulary's identity is grammar_tests_c11's (H05)\n", .{ lib, s.env.digest.?, &s.tokenizer_sha256.?, vocab, c.words });
            return @ptrCast(c);
        }
    }

    fn check(ctx: *anyopaque, opaque_compiler: *api.Structure.Compiler, kind: u32, text: []const u8, err: *[512]u8) bool {
        if (comptime !has_abi) {
            unreachable; // no compiler exists without the loader
        } else {
            const s = of(ctx);
            const c: *abi.owned.Compiler = @ptrCast(@alignCast(opaque_compiler));
            s.lock.lockUncancelable(s.io);
            defer s.lock.unlock(s.io);
            var g = c.compile(kind, text, err) catch |e| {
                if (err[0] == 0) { // grammar_abi's own refusals (KindRange, GrammarTooLarge...) carry no library words
                    const name = @errorName(e);
                    const n = @min(name.len, err.len - 1);
                    @memcpy(err[0..n], name[0..n]);
                    err[n] = 0;
                }
                return false;
            };
            g.free() catch {};
            return true;
        }
    }

    fn rules(ctx: *anyopaque, r: *const api.Request) anyerror!*lanes.grammar.Rules {
        const s = of(ctx);
        const st = r.structure orelse return error.NotStructured;
        const opaque_compiler = st.compiler orelse return error.NoGrammarCompiler;
        if (comptime !has_abi) {
            return error.NoGrammarCompiler;
        } else {
            const c: *abi.owned.Compiler = @ptrCast(@alignCast(opaque_compiler));
            var err: [512]u8 = undefined;
            err[0] = 0;
            return lanes.grammar_owned.create(LockedMatcher, s.gpa, LockedCompiler{ .c = c, .s = s }, @backingInt(st.kind), st.text, &err, s.width, st.after) catch |e| {
                std.debug.print("INFO attn: grammar_server: a reply's grammar could not start: {s} {s}\n", .{ @errorName(e), std.mem.sliceTo(&err, 0) });
                return e;
            };
        }
    }

    /// grammar_abi's compiler, compiled grammar and matcher in grammar_owned's shapes, each call under the role's lock.
    const LockedCompiler = struct {
        c: *abi.owned.Compiler,
        s: *Served,
        pub fn compile(x: LockedCompiler, kind: u32, text: []const u8, err: *[512]u8) !LockedCompiled {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return .{ .g = try x.c.compile(kind, text, err), .s = x.s };
        }
    };
    const LockedCompiled = struct {
        g: abi.owned.Compiled,
        s: *Served,
        pub fn matcher(x: *LockedCompiled) !LockedMatcher {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return .{ .m = try x.g.matcher(), .s = x.s, .c = x.g.c };
        }
        pub fn free(x: *LockedCompiled) !void {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.g.free();
        }
    };
    const LockedMatcher = struct {
        m: abi.owned.Matcher,
        s: *Served,
        c: *abi.owned.Compiler,
        pub fn accept(x: *LockedMatcher, token: u32) !bool {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.m.accept(token);
        }
        pub fn rollback(x: *LockedMatcher, n: usize) !void {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.m.rollback(n);
        }
        pub fn fill(x: *LockedMatcher, words: []u32) !void {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.m.fill(words);
        }
        pub fn terminated(x: *LockedMatcher) !bool {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.m.terminated();
        }
        pub fn free(x: *LockedMatcher) !void {
            x.s.lock.lockUncancelable(x.s.io);
            defer x.s.lock.unlock(x.s.io);
            return x.m.free();
        }
    };
};
