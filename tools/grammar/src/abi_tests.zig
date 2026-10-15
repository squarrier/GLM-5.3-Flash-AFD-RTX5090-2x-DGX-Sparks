//! grammar_abi real-library ABI tests (Claude Opus 5.5 via Hermes, local integration). Every result line is JSON on stdout;
//! the exit code is 0 only when every check passed. Suites: loader (trusted-loader refusals and the real load),
//! real (the real library over the GLM tokenizer and frozen fixtures/cases.txt), admission (bounds and ownership).
//! usage: grammar_abi_tests <suite> key=value ...   (scripts/run_tests.py writes the arguments)
const std = @import("std");
const linux = std.os.linux;
const trusted = @import("tfg_trusted.zig");
const owned = @import("tfg_owned.zig");

var passed: u32 = 0;
var failed: u32 = 0;
var line_buf: [16384]u8 = undefined;

fn write(s: []const u8) void {
    var off: usize = 0;
    while (off < s.len) {
        const n = linux.write(1, s[off..].ptr, s.len - off);
        if (trusted.errnoOf(n) != null or n == 0) return;
        off += n;
    }
}

fn line(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&line_buf, fmt ++ "\n", args) catch return write("{\"error\":\"line too long\"}\n");
    write(s);
}

fn esc(buf: []u8, s: []const u8) []const u8 {
    var n: usize = 0;
    for (s) |ch| {
        if (n + 6 >= buf.len) break;
        switch (ch) {
            '"', '\\' => {
                buf[n] = '\\';
                buf[n + 1] = ch;
                n += 2;
            },
            0...0x1f, 0x7f...0xff => {
                const d = "0123456789abcdef";
                @memcpy(buf[n..][0..4], "\\u00");
                buf[n + 4] = d[ch >> 4];
                buf[n + 5] = d[ch & 15];
                n += 6;
            },
            else => {
                buf[n] = ch;
                n += 1;
            },
        }
    }
    return buf[0..n];
}

fn check(suite: []const u8, name: []const u8, ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (ok) passed += 1 else failed += 1;
    var dbuf: [3000]u8 = undefined;
    const d = std.fmt.bufPrint(&dbuf, fmt, args) catch "detail too long";
    var ebuf: [9000]u8 = undefined;
    line("{{\"suite\":\"{s}\",\"test\":\"{s}\",\"pass\":{},\"detail\":\"{s}\"}}", .{ suite, name, ok, esc(&ebuf, d) });
}

fn note(suite: []const u8, name: []const u8, comptime fmt: []const u8, args: anytype) void {
    var dbuf: [3000]u8 = undefined;
    const d = std.fmt.bufPrint(&dbuf, fmt, args) catch "detail too long";
    var ebuf: [9000]u8 = undefined;
    line("{{\"suite\":\"{s}\",\"observe\":\"{s}\",\"detail\":\"{s}\"}}", .{ suite, name, esc(&ebuf, d) });
}

// ---- small OS helpers (raw Linux calls; no std.Io needed) -------------------------------------------------------

fn zpath(buf: *[4096]u8, p: []const u8) [*:0]const u8 {
    @memcpy(buf[0..p.len], p);
    buf[p.len] = 0;
    return @ptrCast(buf);
}

fn readFile(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    var zb: [4096]u8 = undefined;
    const rc = linux.open(zpath(&zb, path), .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (trusted.errnoOf(rc) != null) return error.Open;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var st: linux.Statx = undefined;
    if (trusted.errnoOf(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != null) return error.Stat;
    const out = try gpa.alloc(u8, @intCast(st.size));
    var got: usize = 0;
    while (got < out.len) {
        const n = linux.read(fd, out[got..].ptr, out.len - got);
        if (trusted.errnoOf(n) != null or n == 0) return error.Read;
        got += n;
    }
    return out;
}

/// /proc files report size 0: read until EOF into `buf`.
fn readProc(path: [*:0]const u8, buf: []u8) []const u8 {
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (trusted.errnoOf(rc) != null) return buf[0..0];
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var got: usize = 0;
    while (got < buf.len) {
        const n = linux.read(fd, buf[got..].ptr, buf.len - got);
        if (trusted.errnoOf(n) != null or n == 0) break;
        got += n;
    }
    return buf[0..got];
}

var proc_buf: [4 << 20]u8 = undefined;

fn statusKb(key: []const u8) u64 {
    const s = readProc("/proc/self/status", &proc_buf);
    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |l| if (std.mem.startsWith(u8, l, key)) {
        var t = std.mem.tokenizeAny(u8, l[key.len..], " \t");
        return std.fmt.parseInt(u64, t.next() orelse "0", 10) catch 0;
    };
    return 0;
}

fn mapped(needle: []const u8) bool {
    const s = readProc("/proc/self/maps", &proc_buf);
    return std.mem.indexOf(u8, s, needle) != null;
}

fn openFds() u32 {
    var n: u32 = 0;
    var fd: i32 = 0;
    while (fd < 4096) : (fd += 1) {
        if (trusted.errnoOf(linux.fcntl(fd, linux.F.GETFD, 0)) == null) n += 1;
    }
    return n;
}

fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn ms(t0: u64) u64 {
    return (nowNs() - t0) / 1_000_000;
}

// ---- arguments ----------------------------------------------------------------------------------------------------

const Cfg = struct {
    keys: [48][]const u8 = undefined,
    vals: [48][]const u8 = undefined,
    n: usize = 0,

    fn get(c: *const Cfg, k: []const u8) ?[]const u8 {
        for (c.keys[0..c.n], c.vals[0..c.n]) |key, v| if (std.mem.eql(u8, key, k)) return v;
        return null;
    }

    fn need(c: *const Cfg, k: []const u8) []const u8 {
        return c.get(k) orelse {
            line("{{\"error\":\"missing argument {s}\"}}", .{k});
            std.process.exit(2);
        };
    }
};

// ---- fixtures ------------------------------------------------------------------------------------------------------

const Rej = union(enum) { none, eos, at: usize };

const Case = struct {
    id: []const u8,
    kind: u32,
    valid: bool,
    grammar: []const u8,
    doc: []const u8,
    bytes: []const u32,
    merged: []const u32,
    rej_bytes: Rej,
    rej_merged: Rej,
};

const Fixtures = struct {
    vocab: u32 = 0,
    tok_size: u32 = 0,
    stops: []const u32 = &.{},
    hole_lo: u32 = 0,
    hole_hi: u32 = 0,
    cases: std.ArrayList(Case) = .empty,
    probe_ids: std.ArrayList([]const u8) = .empty,
    probe_steps: std.ArrayList([]const u32) = .empty,
    watch_id: []const u8 = "",
    watch_step: u32 = 0,
    watch: []const u32 = &.{},
    draft_kind: u32 = 0,
    draft_grammar: []const u8 = "",
    draft_text: []const u8 = "",

    fn find(fx: *const Fixtures, id: []const u8) ?*const Case {
        for (fx.cases.items) |*c| if (std.mem.eql(u8, c.id, id)) return c;
        return null;
    }
};

fn hex(a: std.mem.Allocator, s: []const u8) ![]u8 {
    if (std.mem.eql(u8, s, "-")) return a.alloc(u8, 0);
    const out = try a.alloc(u8, s.len / 2);
    for (out, 0..) |*b, i| b.* = try std.fmt.parseInt(u8, s[2 * i ..][0..2], 16);
    return out;
}

fn ids(a: std.mem.Allocator, s: []const u8) ![]u32 {
    var list: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, s, ',');
    while (it.next()) |x| try list.append(a, try std.fmt.parseInt(u32, x, 10));
    return list.items;
}

fn rej(s: []const u8) !Rej {
    if (std.mem.eql(u8, s, "-")) return .none;
    if (std.mem.eql(u8, s, "eos")) return .eos;
    return .{ .at = try std.fmt.parseInt(usize, s, 10) };
}

fn loadFixtures(a: std.mem.Allocator, text: []const u8) !Fixtures {
    var fx: Fixtures = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        var f = std.mem.tokenizeScalar(u8, l, ' ');
        const tag = f.next() orelse continue;
        if (std.mem.eql(u8, tag, "VOCAB")) fx.vocab = try std.fmt.parseInt(u32, f.next().?, 10);
        if (std.mem.eql(u8, tag, "TOKSIZE")) fx.tok_size = try std.fmt.parseInt(u32, f.next().?, 10);
        if (std.mem.eql(u8, tag, "STOPS")) fx.stops = try ids(a, f.next().?);
        if (std.mem.eql(u8, tag, "HOLES")) {
            var r = std.mem.splitScalar(u8, f.next().?, '-');
            fx.hole_lo = try std.fmt.parseInt(u32, r.next().?, 10);
            fx.hole_hi = try std.fmt.parseInt(u32, r.next().?, 10);
        }
        if (std.mem.eql(u8, tag, "CASE")) {
            const id = f.next().?;
            const kind = try std.fmt.parseInt(u32, f.next().?, 10);
            const expect = f.next().?;
            _ = f.next(); // reject byte offset (same as the byte-token index)
            try fx.cases.append(a, .{
                .id = id,
                .kind = kind,
                .valid = std.mem.eql(u8, expect, "valid"),
                .grammar = try hex(a, f.next().?),
                .doc = try hex(a, f.next().?),
                .bytes = try ids(a, f.next().?),
                .merged = try ids(a, f.next().?),
                .rej_bytes = try rej(f.next().?),
                .rej_merged = try rej(f.next().?),
            });
        }
        if (std.mem.eql(u8, tag, "PROBE")) {
            try fx.probe_ids.append(a, f.next().?);
            try fx.probe_steps.append(a, try ids(a, f.next().?));
        }
        if (std.mem.eql(u8, tag, "WATCH")) {
            fx.watch_id = f.next().?;
            fx.watch_step = try std.fmt.parseInt(u32, f.next().?, 10);
            fx.watch = try ids(a, f.next().?);
        }
        if (std.mem.eql(u8, tag, "DRAFT")) {
            _ = f.next();
            fx.draft_kind = try std.fmt.parseInt(u32, f.next().?, 10);
            fx.draft_grammar = try hex(a, f.next().?);
            fx.draft_text = try hex(a, f.next().?);
        }
    }
    return fx;
}

fn bit(words: []const u32, t: u32) bool {
    return (words[t / 32] >> @intCast(t % 32)) & 1 == 1;
}

fn popcount(words: []const u32) u64 {
    var n: u64 = 0;
    for (words) |w| n += @popCount(w);
    return n;
}

fn maskHash(words: []const u32) [16]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(words), &d, .{});
    return trusted.hexLower(d)[0..16].*;
}

// ---- suite: loader ------------------------------------------------------------------------------------------------

fn loaderCase(cfg: *const Cfg, name: []const u8, path: []const u8, digest: []const u8, want: ?trusted.Failure, extra_sym: []const u8) void {
    const fds0 = openFds();
    var l: trusted.Loader = .{};
    const got = l.get(path, digest);
    const r = &l.report;
    const fds1 = openFds();
    var ok = if (want) |w| got == null and r.failure == w else got != null;
    // a missing symbol is found in .dynsym before dlopen: none of the library's code ran, nothing to unload
    if (extra_sym.len > 0) ok = ok and std.mem.eql(u8, r.symbol, extra_sym) and !r.dlopened and !r.dlclosed and !r.pinned and !r.elf_symbols;
    ok = ok and fds0 == fds1;
    // a refused library must not stay mapped (incomplete-load cleanup); a refused-before-dlopen one never was
    const still = if (want != null) mapped(path) else true;
    if (want != null) ok = ok and !still;
    // the refusal is sticky: a second request does no work even with the real library
    if (want != null) {
        const again = l.get(cfg.need("lib"), cfg.need("lib_sha"));
        ok = ok and again == null and l.attempts == 1 and l.refused_after == 1;
    }
    check("loader", name, ok, "failure={s} errno={d} symbol={s} machine={d} hashed={} elf_symbols={} dlopened={} dlclosed={} pinned={} mapped_after={} fds_before={d} fds_after={d} attempts={d} detail={s}", .{
        if (r.failure) |f| @tagName(f) else "none", r.errno, r.symbol, r.machine, r.hashed, r.elf_symbols, r.dlopened, r.dlclosed, r.pinned, still, fds0, fds1, l.attempts, r.detail(),
    });
}

/// The "plain request" stand-in: work that never touches the grammar library still produces its known answer.
fn plainWork() [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("plain text request: no structure", &d, .{});
    return trusted.hexLower(d);
}

fn suiteLoader(cfg: *const Cfg) void {
    const lib = cfg.need("lib");
    const sha = cfg.need("lib_sha");
    const plain0 = plainWork();
    loaderCase(cfg, "L01-bare-name", "libtfgrammar.so", sha, .relative_path, "");
    loaderCase(cfg, "L02-relative", "lib/libtfgrammar.so", sha, .relative_path, "");
    loaderCase(cfg, "L03-dotdot", cfg.need("dotdot"), sha, .relative_path, "");
    loaderCase(cfg, "L04-absent", cfg.need("absent"), sha, .open_failed, "");
    loaderCase(cfg, "L05-symlink", cfg.need("symlink"), sha, .open_failed, "");
    loaderCase(cfg, "L06-world-writable-file", cfg.need("wfile"), sha, .unsafe_mode, "");
    loaderCase(cfg, "L07-group-writable-parent", cfg.need("gdir_file"), sha, .unsafe_parent, "");
    loaderCase(cfg, "L08-digest-mismatch", lib, cfg.need("wrong_sha"), .digest_mismatch, "");
    loaderCase(cfg, "L09-bad-digest-argument", lib, "xyz", .bad_digest_arg, "");
    loaderCase(cfg, "L10-not-elf", cfg.need("notelf"), cfg.need("notelf_sha"), .not_elf, "");
    if (cfg.get("arm")) |arm| loaderCase(cfg, "L11-wrong-arch-aarch64-real-library", arm, cfg.need("arm_sha"), .wrong_arch, "") else note("loader", "L11-wrong-arch", "aarch64 library not supplied", .{});
    loaderCase(cfg, "L12-abort-stub-missing-tfg_drafts", cfg.need("stub"), cfg.need("stub_sha"), .missing_symbol, "tfg_drafts");
    if (cfg.get("uera")) |u| loaderCase(cfg, "L13-u-era-real-library-missing-tfg_drafts", u, cfg.need("uera_sha"), .missing_symbol, "tfg_drafts") else note("loader", "L13-u-era", "u-era library not supplied", .{});
    // plain text still served after every refusal
    const plain1 = plainWork();
    check("loader", "L14-plain-path-after-refusals", std.mem.eql(u8, &plain0, &plain1), "plain work digest {s} unchanged; process alive; no library function was called (the abort stub would have aborted)", .{plain1[0..16]});
    // the real library loads, all 12 symbols, and the outcome is cached
    var l: trusted.Loader = .{};
    const t0 = nowNs();
    const f = l.get(lib, sha);
    const again = l.get(lib, sha);
    check("loader", "L15-real-library-loads", f != null and again == f and l.attempts == 1 and l.handle != null and l.report.elf_symbols and mapped(lib), "attempts={d} sha256={s} machine={d} elf_symbols={} load_ms={d} mapped={}", .{ l.attempts, l.report.sha256[0..16], l.report.machine, l.report.elf_symbols, ms(t0), mapped(lib) });
    if (f) |fns| {
        var all = true;
        inline for (@typeInfo(trusted.Fns).@"struct".field_names) |n| all = all and @intFromPtr(@field(fns.*, n)) != 0;
        check("loader", "L16-twelve-symbols", all, "12 release symbols resolved, tfg_drafts included", .{});
    }
}

// ---- suite: real --------------------------------------------------------------------------------------------------

const Ctx = struct {
    gpa: std.mem.Allocator,
    f: *const trusted.Fns,
    counts: owned.Counts = .{},
    comp: *owned.Compiler = undefined,
    fx: Fixtures,
    words: []u32 = &.{},
    words2: []u32 = &.{},

    fn holesClear(c: *Ctx, w: []const u32) bool {
        var t = c.fx.hole_lo;
        while (t <= c.fx.hole_hi) : (t += 1) if (bit(w, t)) return false;
        return true;
    }

    fn stopsAre(c: *Ctx, w: []const u32, want: bool) bool {
        for (c.fx.stops) |s| if (bit(w, s) != want) return false;
        return true;
    }
};

/// One case along one spelling: masks at every step against the frozen expectation, then the stop and termination.
fn walkCase(ctx: *Ctx, c: *const Case, toks: []const u32, r: Rej, tag: []const u8) !void {
    var err: [512]u8 = undefined;
    var g = ctx.comp.compile(c.kind, c.grammar, &err) catch |e| {
        var nb: [64]u8 = undefined;
        check("real", try std.fmt.bufPrint(&nb, "{s}/{s}", .{ c.id, tag }), false, "compile failed: {s} {s}", .{ @errorName(e), std.mem.sliceTo(&err, 0) });
        return;
    };
    var m = try g.matcher();
    try g.free(); // the release frees the compiled grammar as soon as the matcher exists
    defer m.free() catch {};
    const stop_at: usize = switch (r) {
        .at => |i| i,
        else => toks.len,
    };
    var ok = true;
    var first_bad: ?usize = null;
    var why: []const u8 = "";
    var rejected_at: ?usize = null;
    var unchanged_after_reject = true;
    var stops_at_reject: ?bool = null;
    for (toks, 0..) |t, i| {
        try m.fill(ctx.words);
        const want = i < stop_at;
        if (bit(ctx.words, t) != want) {
            ok = false;
            if (first_bad == null) {
                first_bad = i;
                why = "mask bit of the next token differs from the expectation";
            }
        }
        // before the rejection step every prefix is incomplete (no stop allowed); at it, the prefix may be complete
        // (k3-long: "yes"), so the stop bits there are only observed (x86-run-2 asserted them: a test-logic error)
        if (i == stop_at) stops_at_reject = ctx.stopsAre(ctx.words, true);
        if ((i < stop_at and !ctx.stopsAre(ctx.words, false)) or !ctx.holesClear(ctx.words)) {
            ok = false;
            if (first_bad == null) {
                first_bad = i;
                why = "a stop bit before the end, or a hole bit";
            }
        }
        const took = try m.accept(t);
        if (took != want) {
            ok = false;
            if (first_bad == null) {
                first_bad = i;
                why = "accept differs from the expectation";
            }
        }
        if (!took) {
            rejected_at = i;
            try m.fill(ctx.words2);
            unchanged_after_reject = std.mem.eql(u32, ctx.words, ctx.words2);
            break;
        }
    }
    var term = false;
    var eos_taken = false;
    var stops_at_end = false;
    switch (r) {
        .at => |i| ok = ok and rejected_at != null and rejected_at.? == i and unchanged_after_reject,
        .none, .eos => {
            try m.fill(ctx.words);
            stops_at_end = ctx.stopsAre(ctx.words, r == .none);
            eos_taken = try m.accept(ctx.fx.stops[0]);
            term = try m.terminated();
            ok = ok and rejected_at == null and stops_at_end and eos_taken == (r == .none) and term == (r == .none) and ctx.holesClear(ctx.words);
        },
    }
    var nb: [64]u8 = undefined;
    var rb: [32]u8 = undefined;
    const rexp: []const u8 = switch (r) {
        .none => "none",
        .eos => "eos",
        .at => |i| std.fmt.bufPrint(&rb, "{d}", .{i}) catch "?",
    };
    check("real", try std.fmt.bufPrint(&nb, "{s}/{s}", .{ c.id, tag }), ok, "kind={d} expect={s} tokens={d} rejected_at={?d} expected_reject={s} unchanged_after_reject={} stops_allowed_at_reject_step={?} stops_ok_at_end={} eos_taken={} terminated={} first_bad={?d} why={s}", .{
        c.kind, if (c.valid) "valid" else "invalid", toks.len, rejected_at, rexp, unchanged_after_reject, stops_at_reject, stops_at_end, eos_taken, term, first_bad, why,
    });
}

fn failedWith(r: anytype, want: anyerror) bool {
    if (r) |_| return false else |e| return e == want;
}

/// Exhaustive agreement: at each step, mask bit t == (tfg_accept(t) == 1) for every id, accepted ids rolled back.
fn probe(ctx: *Ctx, c: *const Case, steps: []const u32) !void {
    var err: [512]u8 = undefined;
    var g = try ctx.comp.compile(c.kind, c.grammar, &err);
    var m = try g.matcher();
    try g.free();
    defer m.free() catch {};
    var at: usize = 0;
    for (steps) |s| {
        while (at < s) : (at += 1) _ = try m.accept(c.bytes[at]);
        try m.fill(ctx.words);
        const t0 = nowNs();
        var mismatches: u32 = 0;
        var first: ?u32 = null;
        var accepted: u64 = 0;
        var t: u32 = 0;
        while (t < ctx.fx.vocab) : (t += 1) {
            const took = try m.accept(t);
            if (took) {
                accepted += 1;
                try m.rollback(1);
            }
            if (took != bit(ctx.words, t)) {
                mismatches += 1;
                if (first == null) first = t;
            }
        }
        try m.fill(ctx.words2);
        const restored = std.mem.eql(u32, ctx.words, ctx.words2);
        var nb: [64]u8 = undefined;
        check("real", try std.fmt.bufPrint(&nb, "probe/{s}@{d}", .{ c.id, s }), mismatches == 0 and restored and ctx.holesClear(ctx.words), "ids={d} allowed_bits={d} accepted={d} mismatches={d} first_mismatch={?d} state_restored_after_{d}_accept_rollback={} mask={s} ms={d}", .{
            ctx.fx.vocab, popcount(ctx.words), accepted, mismatches, first, accepted, restored, maskHash(ctx.words), ms(t0),
        });
    }
}

fn suiteReal(gpa: std.mem.Allocator, cfg: *const Cfg) !void {
    var loader: trusted.Loader = .{};
    const f = loader.get(cfg.need("lib"), cfg.need("lib_sha")) orelse {
        check("real", "R00-load", false, "loader refused: {s}", .{loader.report.detail()});
        return;
    };
    const fx = try loadFixtures(gpa, try readFile(gpa, cfg.need("cases")));
    var ctx: Ctx = .{ .gpa = gpa, .f = f, .fx = fx };
    const tok = try readFile(gpa, cfg.need("tokenizer"));
    var err: [512]u8 = undefined;
    const rss0 = statusKb("VmRSS:");
    const t0 = nowNs();
    ctx.comp = owned.Compiler.open(gpa, f, &ctx.counts, .{}, tok, fx.vocab, fx.stops, &err) catch |e| {
        check("real", "R01-open", false, "{s}: {s}", .{ @errorName(e), std.mem.sliceTo(&err, 0) });
        return;
    };
    const open_ms = ms(t0);
    check("real", "R01-open-glm-tokenizer", ctx.comp.words == (fx.vocab + 31) / 32, "vocab={d} words={d} (ceil(V/32)={d}) tokenizer_bytes={d} stops={any} open_ms={d} rss_kb_before={d} rss_kb_after={d}", .{
        fx.vocab, ctx.comp.words, (fx.vocab + 31) / 32, tok.len, fx.stops, open_ms, rss0, statusKb("VmRSS:"),
    });
    ctx.words = try gpa.alloc(u32, ctx.comp.words);
    ctx.words2 = try gpa.alloc(u32, ctx.comp.words);

    // every frozen case along both spellings
    for (fx.cases.items) |*c| {
        try walkCase(&ctx, c, c.bytes, c.rej_bytes, "bytes");
        try walkCase(&ctx, c, c.merged, c.rej_merged, "merged");
    }
    // separately frozen supplements (fixtures/supplement-k0.txt, supplement-k5.txt), same tokenizer
    if (cfg.get("cases2")) |list| {
        var paths = std.mem.tokenizeScalar(u8, list, ',');
        while (paths.next()) |p2| {
            const fx2 = try loadFixtures(gpa, try readFile(gpa, p2));
            for (fx2.cases.items) |*c| {
                try walkCase(&ctx, c, c.bytes, c.rej_bytes, "bytes");
                try walkCase(&ctx, c, c.merged, c.rej_merged, "merged");
            }
        }
    }
    // exhaustive fill/accept agreement
    for (fx.probe_ids.items, fx.probe_steps.items) |id, steps| try probe(&ctx, fx.find(id).?, steps);

    // added-token literal-text watch (observation with the source-derived expectation: allowed inside a string)
    {
        const c = fx.find(fx.watch_id).?;
        var g = try ctx.comp.compile(c.kind, c.grammar, &err);
        var m = try g.matcher();
        try g.free();
        for (c.bytes[0..fx.watch_step]) |t| _ = try m.accept(t);
        try m.fill(ctx.words);
        var allowed: u32 = 0;
        var list: [256]u8 = undefined;
        var n: usize = 0;
        for (fx.watch) |t| {
            const a = bit(ctx.words, t);
            if (a) allowed += 1;
            const s = std.fmt.bufPrint(list[n..], "{d}:{},", .{ t, a }) catch "";
            n += s.len;
        }
        check("real", "R02-added-tokens-are-literal-text-in-strings", allowed == fx.watch.len, "inside a JSON string, {d}/{d} watched added tokens are allowed as text (source expectation: all; GLM hazard, see RESULTS) {s}", .{ allowed, fx.watch.len, list[0..n] });
        try m.free();
    }

    // rollback restores every earlier state; termination and rollback out of it
    {
        const c = fx.find("k1-basic").?;
        var g = try ctx.comp.compile(c.kind, c.grammar, &err);
        var m = try g.matcher();
        try g.free();
        const n = c.bytes.len;
        const saved = try gpa.alloc(u32, (n + 1) * ctx.comp.words);
        for (c.bytes, 0..) |t, i| {
            try m.fill(saved[i * ctx.comp.words ..][0..ctx.comp.words]);
            _ = try m.accept(t);
        }
        try m.fill(saved[n * ctx.comp.words ..][0..ctx.comp.words]);
        var ok = true;
        var at: usize = n;
        for ([_]usize{ 1, 3, 10, n - 14 }) |k| {
            try m.rollback(k);
            at -= k;
            try m.fill(ctx.words);
            ok = ok and std.mem.eql(u32, ctx.words, saved[at * ctx.comp.words ..][0..ctx.comp.words]);
        }
        check("real", "R03-rollback-restores-masks", ok and at == 0 and !(try m.terminated()), "rolled back 1,3,10,{d} of {d} tokens; every mask equal to the one recorded at that step", .{ n - 14, n });
        // forward again, then the stop token, then out of termination
        for (c.bytes) |t| _ = try m.accept(t);
        const eos = try m.accept(fx.stops[1]);
        const term = try m.terminated();
        const after = try m.accept(c.bytes[0]);
        const fill_rc = ctx.f.fill(m.h.?, @ptrCast(ctx.words.ptr), @intCast(ctx.words.len));
        const pop_after = popcount(ctx.words);
        try m.rollback(1);
        const term2 = try m.terminated();
        try m.fill(ctx.words2);
        const back = std.mem.eql(u32, ctx.words2, saved[n * ctx.comp.words ..][0..ctx.comp.words]);
        check("real", "R04-termination-and-rollback-out", eos and term and !after and !term2 and back, "stop {d} taken={} terminated={} accept_after_termination={} fill_after_termination_rc={d} popcount={d} rollback(1)->terminated={} mask_back_to_pre_stop={}", .{ fx.stops[1], eos, term, after, fill_rc, pop_after, term2, back });
        try m.free();
    }

    // raw ABI error convention on a live matcher
    {
        const c = fx.find("k1-basic").?;
        var g = try ctx.comp.compile(c.kind, c.grammar, &err);
        var m = try g.matcher();
        try g.free();
        const h = m.h.?;
        const big = try gpa.alloc(u32, ctx.comp.words + 1);
        const narrow = f.fill(h, @ptrCast(big.ptr), @intCast(ctx.comp.words - 1));
        const wide = f.fill(h, @ptrCast(big.ptr), @intCast(ctx.comp.words + 1));
        const exact = f.fill(h, @ptrCast(big.ptr), @intCast(ctx.comp.words));
        const neg = f.accept(h, -1);
        const over = f.accept(h, @intCast(fx.vocab));
        const hole = f.accept(h, @intCast(fx.hole_lo));
        const rb0 = f.rollback(h, 0);
        const rbneg = f.rollback(h, -3);
        _ = try m.accept(c.bytes[0]);
        const rbover = f.rollback(h, 5);
        const term = f.terminated(h);
        check("real", "R05-error-convention", narrow == -1 and wide == -1 and exact == 0 and neg == 0 and over == 0 and hole == 0 and rb0 == 0 and rbneg == 0 and rbover == -1, "fill(words-1)={d} fill(words+1)={d} fill(words)={d} accept(-1)={d} accept(V)={d} accept(hole)={d} rollback(0)={d} rollback(-3)={d} rollback(5 of 1)={d} terminated={d}", .{ narrow, wide, exact, neg, over, hole, rb0, rbneg, rbover, term });
        try m.free();
    }

    // grammar drafts (tfg_drafts): ids reported for independent spelling check; the matcher must not move
    {
        var g = try ctx.comp.compile(fx.draft_kind, fx.draft_grammar, &err);
        var m = try g.matcher();
        try g.free();
        try m.fill(ctx.words);
        var out: [16]u32 = undefined;
        const n = try m.drafts(&out);
        try m.fill(ctx.words2);
        const still = std.mem.eql(u32, ctx.words, ctx.words2) and !(try m.terminated());
        var taken: usize = 0;
        for (out[0..n]) |t| {
            if (!try m.accept(t)) break;
            taken += 1;
        }
        var list: [256]u8 = undefined;
        var k: usize = 0;
        for (out[0..n]) |t| {
            const s = std.fmt.bufPrint(list[k..], "{d},", .{t}) catch "";
            k += s.len;
        }
        check("real", "R06-drafts-leave-matcher-in-place", n > 0 and still and taken == n, "drafts={d} ids={s} matcher_unchanged={} all_accepted_afterwards={} (spelling checked by run_tests.py)", .{ n, list[0..k], still, taken == n });
        line("{{\"suite\":\"real\",\"drafts_ids\":\"{s}\"}}", .{list[0..k]});
        try m.free();
    }

    // invalid grammars refuse with a message, and the same compiler then serves a valid one
    {
        const bad = [_]struct { k: u32, t: []const u8 }{
            .{ .k = 1, .t = "{" },
            .{ .k = 1, .t = "{\"type\": \"nonsense\"}" },
            .{ .k = 1, .t = "{\"$ref\": \"#/nowhere\"}" },
            .{ .k = 2, .t = "(" },
            .{ .k = 2, .t = "[z-a]" },
            .{ .k = 3, .t = "[1," },
            .{ .k = 3, .t = "{\"a\": 1}" },
            .{ .k = 4, .t = "root ::= (" },
            .{ .k = 4, .t = "other ::= \"a\"" },
            .{ .k = 5, .t = "{}" },
        };
        for (bad, 0..) |b, i| {
            const before = ctx.counts.compiled_made;
            const r = ctx.comp.compile(b.k, b.t, &err);
            const msg = std.mem.sliceTo(&err, 0);
            var nb: [64]u8 = undefined;
            if (r) |ok_g| {
                var g2 = ok_g;
                try g2.free();
                check("real", try std.fmt.bufPrint(&nb, "R07-invalid-{d}-kind{d}", .{ i, b.k }), false, "compiled although invalid: {s}", .{b.t});
            } else |e| check("real", try std.fmt.bufPrint(&nb, "R07-invalid-{d}-kind{d}", .{ i, b.k }), e == error.Grammar and msg.len > 0 and ctx.counts.compiled_made == before, "refused {s}: {s}", .{ b.t, msg });
        }
        const c = fx.find("k1-basic").?;
        try walkCase(&ctx, c, c.bytes, c.rej_bytes, "after-invalid");
    }

    // tfg_open refusals: malformed tokenizer.json texts give NULL with a message, nothing leaks
    {
        const texts = [_][]const u8{ "not json", "{}", "{\"model\": {}}", "{\"model\": {\"vocab\": {\"a\": \"x\"}}}", "[1, 2]" };
        for (texts, 0..) |t, i| {
            const before = ctx.counts.compilers_opened;
            const r = owned.Compiler.open(gpa, f, &ctx.counts, .{}, t, 100, &.{0}, &err);
            const msg = std.mem.sliceTo(&err, 0);
            var nb: [64]u8 = undefined;
            if (r) |cc| {
                cc.close(gpa) catch {};
                check("real", try std.fmt.bufPrint(&nb, "R08-bad-tokenizer-{d}", .{i}), false, "opened although malformed: {s}", .{t});
            } else |e| check("real", try std.fmt.bufPrint(&nb, "R08-bad-tokenizer-{d}", .{i}), e == error.GrammarSetup and msg.len > 0 and ctx.counts.compilers_opened == before, "refused {s}: {s}", .{ t, msg });
        }
    }

    // the shim's kind quirk, recorded on the raw ABI: an unknown kind compiles as EBNF
    {
        const raw = f.compile(ctx.comp.h.?, 6, "root ::= \"x\"", 12, &err, err.len);
        const compiled = raw != null;
        if (raw) |g| f.free(g);
        note("real", "R09-kind-6-raw", "raw tfg_compile(kind=6, EBNF text) returned {s} (source: default branch = EBNF); the grammar_abi wrapper refuses kind>5 before the call", .{if (compiled) "a grammar" else "NULL"});
    }

    // steady-state memory: compile/matcher/walk cycles, then compiler open/close cycles
    {
        const c = fx.find("k1-basic").?;
        const r0 = statusKb("VmRSS:");
        var i: usize = 0;
        while (i < 50) : (i += 1) {
            var g = try ctx.comp.compile(c.kind, c.grammar, &err);
            var m = try g.matcher();
            try g.free();
            for (c.bytes) |t| _ = try m.accept(t);
            _ = try m.accept(fx.stops[0]);
            try m.free();
        }
        const r1 = statusKb("VmRSS:");
        check("real", "R10-cycles-compile-matcher-walk", r0 > 0 and r1 <= r0 + 32 * 1024, "50 cycles: rss_kb {d} -> {d} (bound +32 MiB; a zero reading fails)", .{ r0, r1 });
        const live_before_close = ctx.counts.live();
        try ctx.comp.close(gpa);
        const r2 = statusKb("VmRSS:");
        var rs: [3]u64 = undefined;
        var opens: [3]u64 = undefined;
        for (&rs, &opens) |*r, *o| {
            const t1 = nowNs();
            const cc = try owned.Compiler.open(gpa, f, &ctx.counts, .{}, tok, fx.vocab, fx.stops, &err);
            o.* = ms(t1);
            var g = try cc.compile(1, c.grammar, &err);
            try g.free();
            try cc.close(gpa);
            r.* = statusKb("VmRSS:");
        }
        check("real", "R11-cycles-open-close", rs[0] > 0 and rs[2] <= rs[1] + 16 * 1024 and ctx.counts.live() == 0 and live_before_close == 1, "live_before_close={d} rss_kb after first close {d}; after each open/compile/close cycle {any}; open_ms {any}; peak VmHWM_kb {d}", .{ live_before_close, r2, rs, opens, statusKb("VmHWM:") });
    }
    const k = ctx.counts;
    check("real", "R12-handle-counts", k.live() == 0 and k.compilers_opened == k.compilers_closed and k.compiled_made == k.compiled_freed and k.matchers_made == k.matchers_freed, "compilers {d}/{d} compiled {d}/{d} matchers {d}/{d} poisoned {d} refused {d} foreign_calls {d}", .{ k.compilers_opened, k.compilers_closed, k.compiled_made, k.compiled_freed, k.matchers_made, k.matchers_freed, k.poisoned, k.refused, k.foreign_calls });
}

// ---- suite: admission ---------------------------------------------------------------------------------------------

fn suiteAdmission(gpa: std.mem.Allocator, cfg: *const Cfg) !void {
    var loader: trusted.Loader = .{};
    const f = loader.get(cfg.need("lib"), cfg.need("lib_sha")) orelse {
        check("admission", "A00-load", false, "loader refused: {s}", .{loader.report.detail()});
        return;
    };
    const fx = try loadFixtures(gpa, try readFile(gpa, cfg.need("cases")));
    const tok = try readFile(gpa, cfg.need("tokenizer"));
    var counts: owned.Counts = .{};
    var err: [512]u8 = undefined;
    const S = "admission";
    // open-time bounds: refused before any foreign call
    const bounds = [_]struct { name: []const u8, vocab: u32, stops: []const u32, lim: owned.Limits, want: owned.Refusal }{
        .{ .name = "A01-vocab-zero", .vocab = 0, .stops = fx.stops, .lim = .{}, .want = error.VocabBounds },
        .{ .name = "A02-vocab-over-bound", .vocab = 4194305, .stops = fx.stops, .lim = .{}, .want = error.VocabBounds },
        .{ .name = "A03-no-stops", .vocab = fx.vocab, .stops = &.{}, .lim = .{}, .want = error.StopBounds },
        .{ .name = "A04-stop-out-of-range", .vocab = fx.vocab, .stops = &.{fx.vocab}, .lim = .{}, .want = error.StopBounds },
        .{ .name = "A05-tokenizer-over-bound", .vocab = fx.vocab, .stops = fx.stops, .lim = .{ .max_tokenizer_bytes = 1 << 20 }, .want = error.TokenizerTooLarge },
    };
    for (bounds) |b| {
        const before = counts.foreign_calls;
        const r = owned.Compiler.open(gpa, f, &counts, b.lim, tok, b.vocab, b.stops, &err);
        const ok = if (r) |cc| blk: {
            cc.close(gpa) catch {};
            break :blk false;
        } else |e| e == b.want and counts.foreign_calls == before;
        check(S, b.name, ok, "refused before tfg_open: foreign calls {d} -> {d}", .{ before, counts.foreign_calls });
    }
    var many: [65]u32 = undefined;
    for (&many, 0..) |*s, i| s.* = @intCast(i);
    {
        const before = counts.foreign_calls;
        const r = owned.Compiler.open(gpa, f, &counts, .{}, tok, fx.vocab, &many, &err);
        const ok = if (r) |cc| blk: {
            cc.close(gpa) catch {};
            break :blk false;
        } else |e| e == error.StopBounds and counts.foreign_calls == before;
        check(S, "A06-too-many-stops", ok, "65 stops refused before tfg_open", .{});
    }
    const comp = try owned.Compiler.open(gpa, f, &counts, .{ .max_grammar_bytes = 4096 }, tok, fx.vocab, fx.stops, &err);
    check(S, "A07-open-probe", comp.words == (fx.vocab + 31) / 32, "tfg_words {d} == ceil({d}/32)", .{ comp.words, fx.vocab });
    const c1 = fx.find("k1-basic").?;
    {
        const before = counts.foreign_calls;
        const big = try gpa.alloc(u8, 4097);
        @memset(big, ' ');
        const r1 = comp.compile(1, big, &err);
        const r2 = comp.compile(6, "root ::= \"x\"", &err);
        check(S, "A08-grammar-size-and-kind-range", failedWith(r1, error.GrammarTooLarge) and failedWith(r2, error.KindRange) and counts.foreign_calls == before, "4097-byte grammar and kind 6 refused before tfg_compile", .{});
    }
    // live matcher bound
    {
        var g = try comp.compile(1, c1.grammar, &err);
        var ms_: [65]owned.Matcher = undefined;
        var made: usize = 0;
        while (made < 64) : (made += 1) ms_[made] = try g.matcher();
        const before = counts.foreign_calls;
        const over = g.matcher();
        const refused = if (over) |_| false else |e| e == error.TooManyMatchers;
        const no_call = counts.foreign_calls == before;
        try ms_[0].free();
        ms_[0] = try g.matcher();
        check(S, "A09-live-matcher-bound", refused and no_call, "64 live matchers made; the 65th refused without a foreign call; after one free a new one is made", .{});
        // close while children live is refused; the compiler stays usable
        const busy = comp.close(gpa);
        const still = try comp.compile(1, c1.grammar, &err);
        var still_g = still;
        try still_g.free();
        check(S, "A10-close-refused-while-live", failedWith(busy, error.Busy), "close with 1 compiled + 64 matchers live -> Busy; compiler still compiles", .{});
        // a matcher outlives its compiled grammar
        try g.free();
        var walk_ok = true;
        for (c1.bytes) |t| walk_ok = walk_ok and try ms_[5].accept(t);
        walk_ok = walk_ok and try ms_[5].accept(fx.stops[0]) and try ms_[5].terminated();
        check(S, "A11-matcher-outlives-compiled", walk_ok, "compiled grammar freed first; a live matcher still walks k1-basic to termination", .{});
        // double free refused without a foreign call
        const before2 = counts.foreign_calls;
        const df = g.free();
        try ms_[1].free();
        const before3 = counts.foreign_calls;
        const dm = ms_[1].free();
        check(S, "A12-double-free-refused", failedWith(df, error.AlreadyFreed) and failedWith(dm, error.AlreadyFreed) and counts.foreign_calls == before3 and before3 == before2 + 1, "second free of a compiled grammar and of a matcher refused before the foreign call", .{});
        // token range and width refused before the call
        const before4 = counts.foreign_calls;
        const tr = ms_[2].accept(fx.vocab);
        const narrow = try gpa.alloc(u32, comp.words - 1);
        const wm = ms_[2].fill(narrow);
        check(S, "A13-token-and-width-refused", failedWith(tr, error.TokenRange) and failedWith(wm, error.WidthMismatch) and counts.foreign_calls == before4, "token id {d} and a {d}-word buffer refused before the call", .{ fx.vocab, comp.words - 1 });
        // a failed rollback poisons the matcher; it is then only freed
        const p = ms_[3].rollback(1000);
        const before5 = counts.foreign_calls;
        const after = ms_[3].accept(c1.bytes[0]);
        check(S, "A14-poisoned-matcher", failedWith(p, error.Poisoned) and failedWith(after, error.Poisoned) and counts.foreign_calls == before5 and counts.poisoned == 1, "rollback(1000) on a fresh matcher -> -1 -> poisoned; later use refused without a call", .{});
        for (ms_[0..64], 0..) |*m, i| if (i != 1) try m.free();
    }
    try comp.close(gpa);
    check(S, "A15-balanced", counts.live() == 0 and counts.compilers_opened == counts.compilers_closed and counts.compiled_made == counts.compiled_freed and counts.matchers_made == counts.matchers_freed, "compilers {d}/{d} compiled {d}/{d} matchers {d}/{d} poisoned {d} refused {d} foreign_calls {d}", .{ counts.compilers_opened, counts.compilers_closed, counts.compiled_made, counts.compiled_freed, counts.matchers_made, counts.matchers_freed, counts.poisoned, counts.refused, counts.foreign_calls });
}

// ---- suite: memory (measurements with stated bounds) --------------------------------------------------------------

fn suiteMemory(gpa: std.mem.Allocator, cfg: *const Cfg) !void {
    const S = "memory";
    var loader: trusted.Loader = .{};
    const f = loader.get(cfg.need("lib"), cfg.need("lib_sha")) orelse {
        check(S, "M00-load", false, "loader refused: {s}", .{loader.report.detail()});
        return;
    };
    const fx = try loadFixtures(gpa, try readFile(gpa, cfg.need("cases")));
    const tok = try readFile(gpa, cfg.need("tokenizer"));
    var counts: owned.Counts = .{};
    var err: [512]u8 = undefined;
    // M01: compiler open/compile/close cycles; the release opens one compiler per server, so this bounds restarts
    var rss: [12]u64 = undefined;
    var open_ms: [12]u64 = undefined;
    for (&rss, &open_ms) |*r, *o| {
        const t0 = nowNs();
        const cc = try owned.Compiler.open(gpa, f, &counts, .{}, tok, fx.vocab, fx.stops, &err);
        o.* = ms(t0);
        var g = try cc.compile(0, "", &err);
        try g.free();
        try cc.close(gpa);
        r.* = statusKb("VmRSS:");
    }
    const growth = rss[11] -| rss[3];
    check(S, "M01-open-close-12-cycles", rss[0] > 0 and growth <= 8 * 1024, "rss_kb after each cycle {any}; growth over the last 8 cycles {d} kB (bound 8192 kB); open_ms {any}", .{ rss, growth, open_ms });
    // M02: 300 distinct schemas (xgrammar's compile cache, 256 MiB budget in the shim), then the same 300 again
    const cc = try owned.Compiler.open(gpa, f, &counts, .{}, tok, fx.vocab, fx.stops, &err);
    const r0 = statusKb("VmRSS:");
    var buf: [256]u8 = undefined;
    var first_ns: u64 = 0;
    var first_max: u64 = 0;
    var again_ns: u64 = 0;
    var ok = true;
    for (0..2) |round| {
        var i: usize = 0;
        while (i < 300) : (i += 1) {
            const schema = try std.fmt.bufPrint(&buf, "{{\"type\":\"object\",\"properties\":{{\"k{d}\":{{\"type\":\"integer\"}}}},\"required\":[\"k{d}\"]}}", .{ i, i });
            const t0 = nowNs();
            var g = cc.compile(1, schema, &err) catch {
                ok = false;
                continue;
            };
            const dt = nowNs() - t0;
            try g.free();
            if (round == 0) {
                first_ns += dt;
                first_max = @max(first_max, dt);
            } else again_ns += dt;
        }
    }
    const r1 = statusKb("VmRSS:");
    try cc.close(gpa);
    const r2 = statusKb("VmRSS:");
    check(S, "M02-300-distinct-schemas-twice", ok and counts.live() == 0, "first compile mean {d} us (max {d} us); same schema again mean {d} us; rss_kb before {d}, after 600 compiles {d}, after close {d}", .{ first_ns / 300 / 1000, first_max / 1000, again_ns / 300 / 1000, r0, r1, r2 });
    note(S, "M03-peak", "VmHWM_kb {d} for this process (tokenizer text {d} bytes held by the test, V={d})", .{ statusKb("VmHWM:"), tok.len, fx.vocab });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.c_allocator;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        line("{{\"error\":\"usage: grammar_abi_tests <loader|real|admission> key=value...\"}}", .{});
        return 2;
    }
    var cfg: Cfg = .{};
    for (argv[2..]) |a| {
        const eq = std.mem.indexOfScalar(u8, a, '=') orelse continue;
        if (cfg.n == cfg.keys.len) break;
        cfg.keys[cfg.n] = a[0..eq];
        cfg.vals[cfg.n] = a[eq + 1 ..];
        cfg.n += 1;
    }
    const suite = argv[1];
    const t0 = nowNs();
    if (std.mem.eql(u8, suite, "loader")) suiteLoader(&cfg) else if (std.mem.eql(u8, suite, "real")) try suiteReal(gpa, &cfg) else if (std.mem.eql(u8, suite, "admission")) try suiteAdmission(gpa, &cfg) else if (std.mem.eql(u8, suite, "memory")) try suiteMemory(gpa, &cfg) else {
        line("{{\"error\":\"unknown suite {s}\"}}", .{suite});
        return 2;
    }
    line("{{\"summary\":\"{s}\",\"passed\":{d},\"failed\":{d},\"ms\":{d}}}", .{ suite, passed, failed, ms(t0) });
    return if (failed == 0) 0 else 1;
}
