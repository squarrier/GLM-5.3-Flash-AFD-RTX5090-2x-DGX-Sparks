//! grammar_server real-library tests (Claude Opus 5.5 via Hermes): grammar_abi's published x86_64 libtfgrammar (the release
//! v1.0.6 c080a18e shim over xgrammar v0.2.8), loaded only by grammar_abi's trusted loader from TF_grammar_server_LIB with its
//! MANIFEST digest TF_grammar_server_LIB_SHA256 (the operator's TF_AFD_GRAMMAR_LIB pair), through the attention role's own
//! grammar source (afd_attn_grammar.zig): opened lazily by Ash's server's first structured request from the fixture
//! checkpoint TF_grammar_server_MODEL (grammar_abi's GLM-5.3 tokenizer.json and the hub config.json; NOT the served checkpoint:
//! vocabulary identity is grammar_tests_c11's H05), then request-owned matchers through api.LaneHost and the actual AFD
//! Lanes(Bound) -> Serve -> Attn -> sampler chain with the real masked adapter. REAL: the loader, the library, its masks
//! over the real tokenizer, handle lifetimes, the server's HTTP replies. MOCK: the device work (grammar_mask's CPU
//! reference and topk over the STAGED words) and the forward (CPU recording). Nothing falls back.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const server = @import("server");
const abi = @import("grammar_abi");
const fx = @import("grammar_mask_fixture.zig");
const Bound = @import("afd_serve_lanes.zig").Bound;
const role = @import("afd_attn_grammar.zig");
const h = @import("grammar_server_http.zig");
const bench = @import("grammar_server_bench.zig");
const testing = std.testing;
const gpa = testing.allocator;

const vocab: u32 = 154880;
const eos: u32 = 154820;
const lbrace: u32 = 90; // "{"
const rbrace: u32 = 92; // "}"
const pair: u32 = 6257; // "{}"
const hash: u32 = 2; // "#": never legal outside a JSON string
const tokenizer_sha256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d";

fn env(name: [:0]const u8) ![]const u8 {
    const v = std.c.getenv(name) orelse {
        std.debug.print("grammar_server-ABI: {s} is unset; the real-library step never falls back\n", .{name});
        return error.MissingRealAbi;
    };
    return std.mem.span(v);
}

fn bit(words: []const u32, t: u32) bool {
    return (words[t / 32] >> @as(u5, @intCast(t % 32))) & 1 != 0;
}

test "grammar_server-ABI loader refusals through the role: relative path, digest mismatch and an unset pair are named, sticky, no foreign call" {
    try testing.expect(role.has_abi);
    const lib = try env("TF_grammar_server_LIB");
    const digest = try env("TF_grammar_server_LIB_SHA256");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env("TF_grammar_server_MODEL");
    const Case = struct { name: []const u8, env: role.Env, failure: ?abi.trusted.Failure, message: []const u8 };
    const zeros: [64]u8 = @splat('0');
    const cases = [_]Case{
        .{ .name = "relative", .env = .{ .lib = "grammar_abi_lib/libtfgrammar.so", .digest = digest }, .failure = .relative_path, .message = "structured output: the server cannot load libtfgrammar: relative_path (not an absolute normalized path: grammar_abi_lib/libtfgrammar.so)" },
        .{ .name = "digest", .env = .{ .lib = lib, .digest = &zeros }, .failure = .digest_mismatch, .message = "" },
        .{ .name = "unset", .env = .{}, .failure = null, .message = "structured output: the server cannot load libtfgrammar: TF_AFD_GRAMMAR_LIB and TF_AFD_GRAMMAR_LIB_SHA256 are unset" },
    };
    for (cases) |k| {
        var loader: abi.trusted.Loader = .{};
        var served = role.Served.init(gpa, testing.io, k.env, vocab, &loader);
        defer served.close() catch {};
        var b: bench.Bench = .{};
        try b.start(true, dir, served.source());
        defer b.stop();
        try testing.expectEqual(@as(u32, 0), loader.attempts); // nothing loads before a structured request
        try testing.expectEqualStrings("ok", try h.content(a, try b.chat(a, "")));
        try testing.expectEqual(@as(u32, 0), loader.attempts);
        var first: []const u8 = "";
        for (0..2) |i| {
            const raw = try b.chat(a, ",\"response_format\":{\"type\":\"json_object\"}");
            try testing.expectEqual(@as(u16, 400), h.status(raw));
            const m = try h.message(a, raw);
            if (k.message.len > 0) try testing.expectEqualStrings(k.message, m);
            if (i == 0) first = m else try testing.expectEqualStrings(first, m);
        }
        try testing.expectEqualStrings("ok", try h.content(a, try b.chat(a, "")));
        try testing.expectEqual(@as(u32, 1), served.opens); // sticky: the server never asks again
        try testing.expectEqual(@as(u32, if (k.failure != null) 1 else 0), loader.attempts);
        try testing.expectEqual(k.failure, loader.report.failure);
        try testing.expect(!loader.report.dlopened);
        try testing.expectEqual(@as(u64, 0), served.counts.foreign_calls);
        try testing.expectEqual(@as(usize, 0), b.fake.structured);
        if (k.failure == .digest_mismatch) {
            try testing.expect(loader.report.hashed);
            try testing.expect(std.mem.indexOf(u8, first, "digest_mismatch") != null);
        }
        std.debug.print("grammar_server-ABI refusal {s}: \"{s}\"; loader attempts {d}, dlopened {}, foreign calls {d}\n", .{ k.name, first, loader.attempts, loader.report.dlopened, served.counts.foreign_calls });
    }
}

/// The served stack over the real library: the actual chain (adapter installed), the lane host with the role's Rules,
/// Ash's server over GLM-5.3's text (the fixture checkpoint) with the role's grammar source, on a loopback port.
const Stack = struct {
    arena: std.heap.ArenaAllocator = undefined,
    loader: abi.trusted.Loader = .{},
    served: role.Served = undefined,
    c: fx.Chain(Bound) = .{},
    sched: fx.Sched = undefined,
    host: api.LaneHost = undefined,
    hf: *server.hf_text.HfText = undefined,
    glm: server.glm_text.GlmText = undefined,
    srv: *server.Server = undefined,
    http: h.Http = .{},

    fn start(k: *Stack, picks: []const []const fx.Pair) !void {
        k.arena = .init(gpa);
        const a = k.arena.allocator();
        const dir = try env("TF_grammar_server_MODEL");
        var problem: []const u8 = "";
        k.hf = try server.hf_text.HfText.load(gpa, testing.io, dir, a, &problem);
        k.glm = .{ .inner = k.hf.text(), .clear = false };
        try k.c.init(true);
        k.sched = try fx.Sched.init(vocab, picks);
        k.c.rec.t.picker = k.sched.picker();
        k.c.dev.emulate = k.sched.emulator();
        k.served = role.Served.init(gpa, testing.io, .{ .lib = try env("TF_grammar_server_LIB"), .digest = try env("TF_grammar_server_LIB_SHA256") }, k.c.rec.m.vocab, &k.loader);
        k.host = api.LaneHost.init(gpa, testing.io, &k.c.core, .{ .name = "glm-afd-attn", .lanes = 1, .structures = true });
        k.host.structured = k.served.structured();
        try k.host.start();
        k.srv = try server.Server.init(gpa, testing.io, k.host.engine(), k.glm.text(), .{ .served_name = "glm-5.3-flash", .model_ids = &.{"glm-5.3-flash"}, .model_dir = dir, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0, .grammars = k.served.source() }, null);
        try k.http.start(k.srv);
    }

    /// The role's order: the server, then the lane host (every stream ended, every matcher freed), then the compiler.
    fn stop(k: *Stack) !void {
        k.http.finish();
        k.srv.deinit();
        k.host.stop();
        const live = k.served.counts.matchers_made - k.served.counts.matchers_freed;
        try testing.expectEqual(@as(u32, 0), live);
        try k.served.close();
        try testing.expectEqual(@as(u32, 0), k.served.counts.live());
        try k.c.clean();
        k.sched.deinit();
        k.c.deinit();
        k.hf.deinit();
        k.arena.deinit();
    }

    /// An independent matcher of the same grammar on the role's compiler (the engine idle): the oracle's masks.
    fn twin(k: *Stack, kind: u32, text: []const u8) !abi.owned.Matcher {
        var err: [512]u8 = undefined;
        var g = try k.served.compiler.?.compile(kind, text, &err);
        defer g.free() catch {};
        return g.matcher();
    }

    /// Each masked pick equals the oracle over the twin's words (the library's mask), which forbid "#" and hash-equal
    /// what the device saw staged; the twin then accepts the drawn token and ends terminated.
    fn verify(k: *Stack, kind: u32, text: []const u8, picks: []const []const fx.Pair, drawn: []const u32) !void {
        var t = try k.twin(kind, text);
        defer t.free() catch {};
        const words = try gpa.alloc(u32, k.served.compiler.?.words);
        defer gpa.free(words);
        const row = try gpa.alloc(f32, vocab);
        defer gpa.free(row);
        for (picks, drawn, 0..) |p, d, i| {
            try t.fill(words);
            @memset(row, -1000);
            for (p) |x| row[x.token] = x.value;
            const expect = fx.oracle(row, words).?;
            if (bit(words, hash)) return error.WinnerNotForbidden;
            std.debug.print("WITNESS ForbiddenWinnerRuntimeViolation.grammar_server-ABI-pick expected={d} observed={d}\n", .{ expect, d });
            if (expect != d) return error.ForbiddenWinnerRuntimeViolation;
            try testing.expectEqual(std.hash.Fnv1a_64.hash(std.mem.sliceAsBytes(words)), k.sched.staged_hashes[i]);
            var allowed: usize = 0;
            for (words) |w| allowed += @popCount(w);
            std.debug.print("grammar_server-ABI kind {d} pick {d}: {d} of {d} tokens allowed by the library, \"#\" forbidden, drew {d}\n", .{ kind, i, allowed, vocab, d });
            try testing.expect(try t.accept(d));
        }
        try testing.expect(try t.terminated());
    }
};

test "grammar_server-ABI served json_object (kind 0): lazy open by the first structured request, masked draws equal the library's oracle, plain unchanged" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const body = [_][]const fx.Pair{
        &.{ .{ .token = hash, .value = 100 }, .{ .token = lbrace, .value = 80 }, .{ .token = pair, .value = 60 } },
        &.{ .{ .token = hash, .value = 100 }, .{ .token = pair, .value = 90 }, .{ .token = rbrace, .value = 70 } },
        &.{ .{ .token = hash, .value = 100 }, .{ .token = rbrace, .value = 90 }, .{ .token = eos, .value = 50 } },
    };
    const plain = [_]fx.Pair{.{ .token = hash, .value = 100 }};
    const picks = [_][]const fx.Pair{ &plain, body[0], body[1], body[2], &plain };
    var k: Stack = .{};
    try k.start(&picks);
    var up = true;
    defer if (up) k.stop() catch {};
    // plain first: nothing loads
    const p0 = try k.http.post(a, "/v1/chat/completions", try h.chat(a, 1, false, ""));
    try testing.expectEqualStrings("#", try h.content(a, p0));
    try testing.expectEqual(@as(u32, 0), k.loader.attempts);
    try testing.expectEqual(@as(u32, 0), k.served.opens);
    const raw = try k.http.post(a, "/v1/chat/completions", try h.chat(a, 8, false, ",\"response_format\":{\"type\":\"json_object\"}"));
    try testing.expectEqual(@as(u16, 200), h.status(raw));
    try testing.expectEqualStrings("{}", try h.content(a, raw));
    try testing.expectEqualStrings("stop", try h.finishReason(a, raw));
    try testing.expectEqual(@as(u32, 1), k.served.opens);
    try testing.expectEqual(@as(u32, 1), k.loader.attempts);
    try testing.expectEqualStrings(tokenizer_sha256, &k.served.tokenizer_sha256.?);
    try testing.expectEqual(@as(usize, 4840), k.served.compiler.?.words);
    try testing.expectEqual(@as(u64, 3), k.c.dev.stats.masked);
    try testing.expectEqual(@as(usize, 3), try fx.maskedOrder(&k.c.rec.t));
    try k.verify(0, "", &body, &.{ lbrace, rbrace, eos });
    const p1 = try k.http.post(a, "/v1/chat/completions", try h.chat(a, 1, false, ""));
    try testing.expectEqualStrings("#", try h.content(a, p1));
    try testing.expectEqual(@as(u32, 1), k.served.opens);
    const counts = k.served.counts;
    up = false;
    try k.stop();
    std.debug.print("grammar_server-ABI json_object: reply \"{{}}\" stop; opens 1, loader attempts 1, tokenizer sha256 {s}; grammar_abi counts before close: foreign_calls {d} compiled {d}/{d} matchers {d}/{d} poisoned {d}; live after close 0\n", .{ tokenizer_sha256, counts.foreign_calls, counts.compiled_made, counts.compiled_freed, counts.matchers_made, counts.matchers_freed, counts.poisoned });
}

test "grammar_server-ABI served json_schema (kind 1): the schema's required key forbids an empty object; masked draws equal the library's oracle" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // the text after "{" as GLM-5.3's tokenizer spells it; a valid spelling of {"a":1} token by token
    var problem: []const u8 = "";
    const hf = try server.hf_text.HfText.load(gpa, testing.io, try env("TF_grammar_server_MODEL"), a, &problem);
    const rest = try hf.text().encode(a, "\"a\":1}", false);
    hf.deinit();
    var picks: std.ArrayList([]const fx.Pair) = .empty;
    // pick 1 offers "}" above the next key token: kind 0 would close the object, the schema's "required" forbids it
    try picks.append(a, try a.dupe(fx.Pair, &.{ .{ .token = hash, .value = 100 }, .{ .token = lbrace, .value = 80 } }));
    for (rest, 0..) |t, i| {
        if (i == 0) try picks.append(a, try a.dupe(fx.Pair, &.{ .{ .token = hash, .value = 100 }, .{ .token = rbrace, .value = 90 }, .{ .token = t, .value = 80 } })) else try picks.append(a, try a.dupe(fx.Pair, &.{ .{ .token = hash, .value = 100 }, .{ .token = t, .value = 80 } }));
    }
    try picks.append(a, try a.dupe(fx.Pair, &.{ .{ .token = hash, .value = 100 }, .{ .token = eos, .value = 50 } }));
    var k: Stack = .{};
    try k.start(picks.items);
    var up = true;
    defer if (up) k.stop() catch {};
    const schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"]}";
    const text = try server.json.stringify(a, (try server.json.parse(a, schema)).ok, .{ .ascii = false });
    const raw = try k.http.post(a, "/v1/chat/completions", try h.chat(a, 16, false, try std.fmt.allocPrint(a, ",\"response_format\":{{\"type\":\"json_schema\",\"json_schema\":{{\"name\":\"r\",\"schema\":{s}}}}}", .{schema})));
    try testing.expectEqual(@as(u16, 200), h.status(raw));
    try testing.expectEqualStrings("{\"a\":1}", try h.content(a, raw));
    try testing.expectEqualStrings("stop", try h.finishReason(a, raw));
    try testing.expectEqual(@as(u32, 1), k.served.opens);
    var drawn: std.ArrayList(u32) = .empty;
    try drawn.append(a, lbrace);
    try drawn.appendSlice(a, rest);
    try drawn.append(a, eos);
    try k.verify(1, text, picks.items, drawn.items);
    // the contrast: after "{", kind 0's mask allows "}" and this schema's does not
    var any = try k.twin(0, "");
    var sch = try k.twin(1, text);
    const w0 = try gpa.alloc(u32, k.served.compiler.?.words);
    defer gpa.free(w0);
    const w1 = try gpa.alloc(u32, k.served.compiler.?.words);
    defer gpa.free(w1);
    const took = (try any.accept(lbrace)) and (try sch.accept(lbrace));
    try any.fill(w0);
    try sch.fill(w1);
    try any.free();
    try sch.free();
    try testing.expect(took);
    try testing.expect(bit(w0, rbrace) and !bit(w1, rbrace));
    try testing.expectEqual(@as(u64, drawn.items.len), k.c.dev.stats.masked);
    up = false;
    try k.stop();
    std.debug.print("grammar_server-ABI json_schema: {d} masked draws {any} -> reply {{\"a\":1}} stop; \"}}\" after \"{{\" allowed by kind 0, forbidden by the schema\n", .{ drawn.items.len, drawn.items });
}

test "grammar_server-ABI ownership: one compiler per role, request-owned matchers, closed only after every matcher is freed" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try env("TF_grammar_server_MODEL");
    const tok = try std.Io.Dir.cwd().readFileAlloc(testing.io, try std.fs.path.join(a, &.{ dir, "tokenizer.json" }), a, .limited(256 << 20));
    var loader: abi.trusted.Loader = .{};
    var served = role.Served.init(gpa, testing.io, .{ .lib = try env("TF_grammar_server_LIB"), .digest = try env("TF_grammar_server_LIB_SHA256") }, vocab, &loader);
    const src = served.source();
    var problem: []const u8 = "";
    const c = src.open(src.ctx, tok, vocab, &.{ 154820, 154827, 154829 }, &problem) orelse {
        std.debug.print("grammar_server-ABI open refused: {s}\n", .{problem});
        return error.OpenRefused;
    };
    try testing.expectEqual(@as(u32, 1), served.counts.compilers_opened);
    var err: [512]u8 = undefined;
    try testing.expect(src.check(src.ctx, c, 0, "", &err));
    try testing.expect(!src.check(src.ctx, c, 6, "", &err)); // kinds past 5 refused before the library (KindRange)
    try testing.expect(!src.check(src.ctx, c, 1, "{", &err)); // xgrammar's own refusal (not JSON)
    try testing.expect(std.mem.sliceTo(&err, 0).len > 0);
    const req: api.Request = .{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .structure = .{ .kind = .json, .compiler = c } };
    const st = served.structured();
    const r1 = try st.rules(st.ctx, &req);
    const r2 = try st.rules(st.ctx, &req);
    try testing.expectEqual(@as(u32, 2), served.counts.matchers_made - served.counts.matchers_freed);
    try testing.expectError(error.Busy, served.close()); // never freed while a matcher lives
    try testing.expect(src.check(src.ctx, c, 0, "", &err)); // and still serving
    r1.close();
    try testing.expectError(error.Busy, served.close());
    r2.close();
    try served.close();
    try testing.expectEqual(@as(u32, 0), served.counts.live());
    try testing.expectEqual(@as(u32, 1), served.counts.compilers_closed);
    try served.close(); // closed: nothing left to free
    // a role whose logits width is not the compiler's vocabulary: refused, its matcher freed at once
    var wrong = role.Served.init(gpa, testing.io, .{ .lib = try env("TF_grammar_server_LIB"), .digest = try env("TF_grammar_server_LIB_SHA256") }, vocab + 1, &loader);
    const ws = wrong.source();
    const wc = ws.open(ws.ctx, tok, vocab, &.{ 154820, 154827, 154829 }, &problem) orelse return error.OpenRefused;
    const wreq: api.Request = .{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .structure = .{ .kind = .json, .compiler = wc } };
    const wst = wrong.structured();
    try testing.expectError(error.WidthMismatch, wst.rules(wst.ctx, &wreq));
    try testing.expectEqual(wrong.counts.matchers_made, wrong.counts.matchers_freed);
    try wrong.close();
    try testing.expectEqual(@as(u32, 0), wrong.counts.live());
    std.debug.print("grammar_server-ABI ownership: close refused Busy with 2 then 1 live matchers, closed after both freed (live 0); a width mismatch frees its matcher\n", .{});
}
