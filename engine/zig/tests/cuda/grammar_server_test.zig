//! grammar_server CPU tests (Claude Opus 5.5 via Hermes): served compiler ownership and response_format wiring for the AFD
//! attention role, default off (evidence/grammar_mask/ANSWER.md items 1-5). Ash's server over loopback HTTP; the grammar
//! source is a FAKE (counts opens and checks) or the role's own, built here without grammar_abi's loader (the stub a
//! build without -Dgram2abi gets); the lane host and the actual AFD Lanes(Bound) -> Serve -> Attn -> sampler chain
//! run with the real masked adapter in its recorded mode (MOCK device: grammar_mask's CPU reference over the staged
//! words) and a FAKE matcher. The REAL library cases are grammar_server_abi.zig's (test-grammar_server_abi).
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const server = @import("server");
const fx = @import("grammar_mask_fixture.zig");
const Bound = @import("afd_serve_lanes.zig").Bound;
const role = @import("afd_attn_grammar.zig");
const h = @import("grammar_server_http.zig");
const bench = @import("grammar_server_bench.zig");
const testing = std.testing;
const gpa = testing.allocator;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const digest = "82b651017ba77bf137e9abfebc179c94a344a4088515c958a61ee85e78af6ad4";
const names = [_][:0]const u8{ "TF_AFD_GRAMMAR_LIB", "TF_AFD_GRAMMAR_LIB_SHA256", "TF_AFD_GRAMMAR_SERVE" };

fn unsetAll() void {
    for (names) |n| _ = unsetenv(n);
}

test "grammar_server env: LIB and LIB_SHA256 together or not at all, SERVE=1 only with MASK=1, named refusals at open" {
    try testing.expect((try role.parse(null, null, null, false)) == null);
    try testing.expect((try role.parse(null, null, null, true)) == null);
    try testing.expectError(error.AfdGrammarLibUnpaired, role.parse("/opt/tfg/libtfgrammar.so", null, null, false));
    try testing.expectError(error.AfdGrammarLibUnpaired, role.parse(null, digest, null, true));
    try testing.expectError(error.AfdGrammarLibUnpaired, role.parse("/opt/tfg/libtfgrammar.so", null, "1", true));
    // paired, SERVE unset or 0: nothing serves and nothing loads (structures false)
    try testing.expect((try role.parse("/opt/tfg/libtfgrammar.so", digest, null, true)) == null);
    try testing.expect((try role.parse("/opt/tfg/libtfgrammar.so", digest, "0", true)) == null);
    try testing.expectError(error.BadAfdGrammarServe, role.parse(null, null, "yes", true));
    try testing.expectError(error.BadAfdGrammarServe, role.parse(null, null, "", true));
    try testing.expectError(error.AfdGrammarServeWithoutMask, role.parse("/opt/tfg/libtfgrammar.so", digest, "1", false));
    try testing.expectError(error.AfdGrammarServeWithoutMask, role.parse(null, null, "1", false));
    const on = (try role.parse("/opt/tfg/libtfgrammar.so", digest, "1", true)).?;
    try testing.expectEqualStrings("/opt/tfg/libtfgrammar.so", on.lib.?);
    try testing.expectEqualStrings(digest, on.digest.?);
    // SERVE=1 without the pair: served, and the lazy open's sticky refusal says the pair is unset (ABI step)
    const bare = (try role.parse(null, null, "1", true)).?;
    try testing.expect(bare.lib == null and bare.digest == null);
    // the role's names, read once by fromEnv (afd_attn_gpu.zig, before anything loads)
    unsetAll();
    defer unsetAll();
    try testing.expect((try role.fromEnv(false)) == null);
    _ = setenv("TF_AFD_GRAMMAR_LIB", "/opt/tfg/libtfgrammar.so", 1);
    try testing.expectError(error.AfdGrammarLibUnpaired, role.fromEnv(true));
    _ = setenv("TF_AFD_GRAMMAR_LIB_SHA256", digest, 1);
    try testing.expect((try role.fromEnv(true)) == null);
    _ = setenv("TF_AFD_GRAMMAR_SERVE", "1", 1);
    try testing.expectError(error.AfdGrammarServeWithoutMask, role.fromEnv(false));
    const env = (try role.fromEnv(true)).?;
    try testing.expectEqualStrings("/opt/tfg/libtfgrammar.so", env.lib.?);
    _ = unsetenv("TF_AFD_GRAMMAR_LIB");
    try testing.expectError(error.AfdGrammarLibUnpaired, role.fromEnv(true));
    std.debug.print("grammar_server env: pairing, SERVE value and SERVE-needs-MASK refused by name at open; unset or SERVE=0: nothing\n", .{});
}

test "grammar_server default off: structures false keeps today's refusal byte for byte and never touches a grammar source" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try bench.modelDir(a, "off", "{\"model\": \"fake\"}", bench.glm_config);
    var src: bench.Source = .{};
    var b: bench.Bench = .{};
    try b.start(false, dir, src.source()); // a source configured, but the engine does not enforce structures
    defer b.stop();
    try testing.expect(!b.srv.info.structures);
    var refusal: ?[]const u8 = null;
    for ([_][]const u8{
        ",\"response_format\":{\"type\":\"json_object\"}",
        ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}",
        ",\"guided_regex\":\"[a-z]+\"",
        ",\"structured_outputs\":{\"json\":{\"type\":\"object\"}}",
    }) |extra| {
        const raw = try b.chat(a, extra);
        try testing.expectEqual(@as(u16, 400), h.status(raw));
        try testing.expectEqualStrings(bench.unsupported, try h.message(a, raw));
        // every refusal's body is the same bytes (the wire A/B compares them with 045dd856's)
        if (refusal) |r| try testing.expectEqualStrings(r, h.body(raw)) else refusal = try a.dupe(u8, h.body(raw));
    }
    const plain = try b.chat(a, "");
    try testing.expectEqual(@as(u16, 200), h.status(plain));
    try testing.expectEqualStrings("ok", try h.content(a, plain));
    try testing.expectEqual(@as(usize, 0), src.opens);
    try testing.expectEqual(@as(usize, 0), src.checks);
    try testing.expectEqual(@as(usize, 1), b.fake.submits);
    try testing.expectEqual(@as(usize, 0), b.fake.structured);
    std.debug.print("grammar_server default off: 4 structured requests refused with today's message ({d} bytes each), plain served, source opens 0\n", .{refusal.?.len});
}

test "grammar_server lazy open and the release mapping: only the first structured request opens; json_object is kind 0, json_schema kind 1 with the schema text" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tokenizer = "{\"model\": {\"type\": \"BPE\"}, \"fake\": true}";
    const dir = try bench.modelDir(a, "lazy", tokenizer, bench.glm_config);
    var src: bench.Source = .{};
    var b: bench.Bench = .{};
    try b.start(true, dir, src.source());
    defer b.stop();
    // plain, type text, malformed and unknown types: no open (grammar.zig:56-67 decides before any compiler)
    const plain = try b.chat(a, "");
    try testing.expectEqualStrings("ok", try h.content(a, plain));
    const text = try b.chat(a, ",\"response_format\":{\"type\":\"text\"}");
    try testing.expectEqual(@as(u16, 200), h.status(text));
    const unknown = try b.chat(a, ",\"response_format\":{\"type\":\"xml\"}");
    try testing.expectEqual(@as(u16, 400), h.status(unknown));
    try testing.expectEqualStrings("response_format type must be text, json_object or json_schema, not 'xml'", try h.message(a, unknown));
    const missing = try b.chat(a, ",\"response_format\":{\"type\":\"json_schema\"}");
    try testing.expectEqualStrings("response_format json_schema needs json_schema.schema (a JSON schema object)", try h.message(a, missing));
    const not_object = try b.chat(a, ",\"response_format\":[]");
    try testing.expectEqualStrings("response_format must be an object such as {\"type\": \"json_object\"}", try h.message(a, not_object));
    try testing.expectEqual(@as(usize, 0), src.opens);
    try testing.expectEqual(@as(usize, 2), b.fake.submits);
    // the first structured request opens the compiler from tokenizer.json, text_config.vocab_size and the eos ids
    const obj = try b.chat(a, ",\"response_format\":{\"type\":\"json_object\"}");
    try testing.expectEqual(@as(u16, 200), h.status(obj));
    try testing.expectEqualStrings("ok", try h.content(a, obj));
    try testing.expectEqual(@as(usize, 1), src.opens);
    try testing.expectEqual(@as(u32, 154880), src.vocab);
    try testing.expectEqualSlices(u32, &.{ 154820, 154827, 154829 }, src.stops[0..src.nstops]);
    try testing.expectEqual(std.hash.Fnv1a_64.hash(tokenizer), src.tokenizer_fnv);
    try testing.expectEqual(@as(?u32, 0), b.fake.kind);
    try testing.expectEqualStrings("", b.fake.lastText());
    try testing.expectEqual(@as(?*api.Structure.Compiler, src.compiler()), b.fake.compiler);
    try testing.expectEqual(@as(?u32, null), b.fake.after);
    try testing.expectEqual(@as(u32, 0), src.kinds[0]);
    // json_schema: the schema text as the release's schemaText writes it (grammar.zig:22-30), compiled once and kept
    const schema = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"]}";
    const want = try server.json.stringify(a, (try server.json.parse(a, schema)).ok, .{ .ascii = false });
    const js = try b.chat(a, try std.fmt.allocPrint(a, ",\"response_format\":{{\"type\":\"json_schema\",\"json_schema\":{{\"name\":\"r\",\"schema\":{s}}}}}", .{schema}));
    try testing.expectEqual(@as(u16, 200), h.status(js));
    try testing.expectEqual(@as(usize, 1), src.opens);
    try testing.expectEqual(@as(?u32, 1), b.fake.kind);
    try testing.expectEqualStrings(want, b.fake.lastText());
    try testing.expectEqual(@as(u32, 1), src.kinds[1]);
    try testing.expectEqualStrings(want, src.textAt(1));
    // the release maps the guided_* fields and streams the same way: nothing new decided here
    const regex = try b.chat(a, ",\"guided_regex\":\"[a-z]+\"");
    try testing.expectEqual(@as(u16, 200), h.status(regex));
    try testing.expectEqual(@as(?u32, 2), b.fake.kind);
    try testing.expectEqualStrings("[a-z]+", b.fake.lastText());
    const stream = try b.http.post(a, "/v1/chat/completions", try h.chat(a, 2, true, ",\"response_format\":{\"type\":\"json_object\"}"));
    try testing.expectEqual(@as(u16, 200), h.status(stream));
    const completion = try b.http.post(a, "/v1/completions", "{\"model\":\"glm-5.3-flash\",\"prompt\":\"hi\",\"max_tokens\":2,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}");
    try testing.expectEqual(@as(u16, 200), h.status(completion));
    try testing.expectEqual(@as(usize, 1), src.opens);
    try testing.expectEqual(@as(usize, 5), src.checks);
    try testing.expectEqual(@as(usize, 5), b.fake.structured);
    std.debug.print("grammar_server lazy: 5 unstructured requests opened nothing; 5 structured ones opened once (vocab {d}, stops {any}); json_object kind 0 \"\", json_schema kind 1 {s}\n", .{ src.vocab, src.stops[0..src.nstops], want });
}

test "grammar_server sticky open failure: HTTP 400 for every structured request for the life of the server, plain unaffected" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const obj = ",\"response_format\":{\"type\":\"json_object\"}";
    const js = ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}";
    // (1) the source refuses (here as grammar_abi's loader would name a digest mismatch)
    {
        const dir = try bench.modelDir(a, "sticky", "{}", bench.glm_config);
        var src: bench.Source = .{ .fail_open = "the server cannot load libtfgrammar: digest_mismatch (FAKE)" };
        var b: bench.Bench = .{};
        try b.start(true, dir, src.source());
        defer b.stop();
        for ([_][]const u8{ obj, js, obj }) |extra| {
            const raw = try b.chat(a, extra);
            try testing.expectEqual(@as(u16, 400), h.status(raw));
            try testing.expectEqualStrings("structured output: the server cannot load libtfgrammar: digest_mismatch (FAKE)", try h.message(a, raw));
            const plain = try b.chat(a, "");
            try testing.expectEqualStrings("ok", try h.content(a, plain));
        }
        try testing.expectEqual(@as(usize, 1), src.opens);
        try testing.expectEqual(@as(usize, 0), src.checks);
        try testing.expectEqual(@as(usize, 0), b.fake.structured);
        try testing.expectEqual(@as(usize, 3), b.fake.submits);
    }
    // (2) the checkpoint has no tokenizer.json, (3) no vocab_size: the release's words, sticky, the source never asked
    for ([_]struct { []const u8, ?[]const u8, []const u8, []const u8 }{
        .{ "notok", null, bench.glm_config, "structured output: the checkpoint has no tokenizer.json" },
        .{ "novocab", "{}", "{\"text_config\": {}}", "structured output: structured output needs the checkpoint's config.json vocab_size" },
    }) |k| {
        const dir = try bench.modelDir(a, k[0], k[1], k[2]);
        var src: bench.Source = .{};
        var b: bench.Bench = .{};
        try b.start(true, dir, src.source());
        defer b.stop();
        for (0..2) |_| {
            const raw = try b.chat(a, obj);
            try testing.expectEqual(@as(u16, 400), h.status(raw));
            try testing.expectEqualStrings(k[3], try h.message(a, raw));
        }
        try testing.expectEqualStrings("ok", try h.content(a, try b.chat(a, "")));
        try testing.expectEqual(@as(usize, 0), src.opens);
    }
    // (4) a grammar the library refuses: a 400 naming the field, nothing submitted; another kind still served
    {
        const dir = try bench.modelDir(a, "refused", "{}", bench.glm_config);
        var src: bench.Source = .{ .fail_kind = 1 };
        var b: bench.Bench = .{};
        try b.start(true, dir, src.source());
        defer b.stop();
        const raw = try b.chat(a, js);
        try testing.expectEqual(@as(u16, 400), h.status(raw));
        try testing.expectEqualStrings("response_format: the grammar cannot be enforced: FAKE: this grammar is refused", try h.message(a, raw));
        try testing.expectEqual(@as(usize, 0), b.fake.submits);
        try testing.expectEqual(@as(u16, 200), h.status(try b.chat(a, obj)));
        try testing.expectEqual(@as(usize, 1), src.opens);
    }
    // (5) an engine that says structures without any source (not the role's wiring): refused, never served plain
    {
        const dir = try bench.modelDir(a, "nosource", "{}", bench.glm_config);
        var b: bench.Bench = .{};
        try b.start(true, dir, null);
        defer b.stop();
        const raw = try b.chat(a, obj);
        try testing.expectEqual(@as(u16, 400), h.status(raw));
        try testing.expectEqualStrings("structured output: the server cannot load libtfgrammar (no grammar library is configured for this engine)", try h.message(a, raw));
        try testing.expectEqual(@as(usize, 0), b.fake.structured);
    }
    std.debug.print("grammar_server sticky: a failed open refuses every later structured request (400) without reopening; plain served; a refused grammar 400s before submit\n", .{});
}

test "grammar_server stub build: SERVE=1 on a tf-glm-afd without grammar_abi's loader refuses structured requests (sticky 400), plain served" {
    try testing.expect(!role.has_abi); // this step builds the role over the stub, as a tool built without -Dgram2abi
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try bench.modelDir(a, "stub", "{}", bench.glm_config);
    var served = role.Served.init(gpa, testing.io, .{ .lib = "/opt/tfg/libtfgrammar.so", .digest = digest }, 154880, {});
    defer served.close() catch {};
    var b: bench.Bench = .{};
    try b.start(true, dir, served.source());
    defer b.stop();
    for (0..2) |_| {
        const raw = try b.chat(a, ",\"response_format\":{\"type\":\"json_object\"}");
        try testing.expectEqual(@as(u16, 400), h.status(raw));
        try testing.expectEqualStrings("structured output: the server cannot load libtfgrammar: this tf-glm-afd was built without grammar_abi's loader (-Dgram2abi)", try h.message(a, raw));
    }
    try testing.expectEqualStrings("ok", try h.content(a, try b.chat(a, "")));
    try testing.expectEqual(@as(u32, 1), served.opens);
    try testing.expectEqual(@as(usize, 0), b.fake.structured);
    const req: api.Request = .{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .structure = .{ .kind = .json } };
    const st = served.structured();
    try testing.expectError(error.NoGrammarCompiler, st.rules(st.ctx, &req));
    std.debug.print("grammar_server stub: the role without grammar_abi's loader opens once, refuses by name, serves plain\n", .{});
}

/// FAKE Rules for the lane host's served path: each structured request gets a fresh script allowing 'x' then 'y'.
const Rp = struct {
    width: usize,
    scripts: [4]fx.Script = undefined,
    made: usize = 0,
    fail: ?anyerror = null,
    kinds: [4]u32 = undefined,
    compilers: [4]?*api.Structure.Compiler = undefined,
    fn structured(p: *Rp) api.LaneHost.Structured {
        return .{ .ctx = p, .rules = rules };
    }
    fn rules(ctx: *anyopaque, r: *const api.Request) anyerror!*lanes.grammar.Rules {
        const p: *Rp = @ptrCast(@alignCast(ctx));
        if (p.fail) |e| return e;
        const x = &p.scripts[p.made];
        x.* = .{ .steps = &.{ &.{'x'}, &.{'y'} } };
        x.init(p.width);
        p.kinds[p.made] = @backingInt(r.structure.?.kind);
        p.compilers[p.made] = r.structure.?.compiler;
        p.made += 1;
        return &x.rules;
    }
};

test "grammar_server served path: response_format through the server, the lane host's role Rules and the actual chain (FAKE matcher, MOCK device)" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try bench.modelDir(a, "chain", "{}", bench.glm_config);
    var c: fx.Chain(Bound) = .{};
    try c.init(true); // TF_AFD_GRAMMAR_MASK=1: the masked adapter installed
    defer c.deinit();
    // picks: the structured request's two masked draws ('#' forbidden), the plain request's two, nothing for the refused
    const picks = [_][]const fx.Pair{
        &.{ .{ .token = '#', .value = 100 }, .{ .token = 'x', .value = 80 } },
        &.{ .{ .token = '#', .value = 100 }, .{ .token = 'y', .value = 80 } },
        &.{.{ .token = '#', .value = 100 }},
        &.{.{ .token = '#', .value = 100 }},
    };
    var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    c.dev.emulate = sched.emulator();
    var rp: Rp = .{ .width = c.rec.m.vocab };
    var host = api.LaneHost.init(gpa, testing.io, &c.core, .{ .name = "glm-afd-attn", .lanes = 1, .structures = true });
    host.structured = rp.structured();
    var host_up = true;
    defer if (host_up) host.stop();
    try host.start();
    var src: bench.Source = .{};
    var text: h.ByteText = .{};
    const srv = try server.Server.init(gpa, testing.io, host.engine(), text.text(), .{ .served_name = "glm-5.3-flash", .model_ids = &.{"glm-5.3-flash"}, .model_dir = dir, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0, .grammars = src.source() }, null);
    var srv_up = true;
    defer if (srv_up) srv.deinit();
    var http: h.Http = .{};
    try http.start(srv);
    var http_up = true;
    defer if (http_up) http.finish();
    const obj = try http.post(a, "/v1/chat/completions", try h.chat(a, 8, false, ",\"response_format\":{\"type\":\"json_object\"}"));
    try testing.expectEqual(@as(u16, 200), h.status(obj));
    try testing.expectEqualStrings("xy", try h.content(a, obj));
    try testing.expectEqualStrings("stop", try h.finishReason(a, obj));
    try testing.expectEqual(@as(usize, 1), rp.made);
    try testing.expectEqual(@as(u32, 0), rp.kinds[0]);
    try testing.expectEqual(@as(?*api.Structure.Compiler, src.compiler()), rp.compilers[0]);
    try testing.expectEqual(@as(u64, 2), c.dev.stats.masked);
    try testing.expectEqual(@as(usize, 2), sched.device_calls);
    try testing.expectEqual(@as(usize, 2), try fx.maskedOrder(&c.rec.t));
    const plain = try http.post(a, "/v1/chat/completions", try h.chat(a, 2, false, ""));
    try testing.expectEqualStrings("##", try h.content(a, plain));
    try testing.expectEqual(@as(usize, 1), rp.made);
    try testing.expectEqual(@as(u64, 2), c.dev.stats.masked);
    // the role's Rules refused (as a width mismatch would be): the reply fails, nothing is drawn plain
    rp.fail = error.WidthMismatch;
    const refused = try http.post(a, "/v1/chat/completions", try h.chat(a, 8, false, ",\"response_format\":{\"type\":\"json_object\"}"));
    try testing.expect(h.status(refused) != 200);
    try testing.expectEqualStrings("the reply's grammar could not start (xgrammar)", try h.message(a, refused));
    try testing.expectEqual(@as(usize, 4), sched.calls);
    http.finish();
    http_up = false;
    srv.deinit();
    srv_up = false;
    host.stop();
    host_up = false;
    // the structured request's matcher was closed once, by its stream's end
    try testing.expectEqual(@as(usize, 1), rp.scripts[0].closes);
    try c.clean();
    std.debug.print("grammar_server served path (FAKE matcher, MOCK device): json_object -> role Rules -> masked x y ('#' forbidden twice), plain ## after, a refused Rules fails the reply; matcher closed once\n", .{});
}

test "grammar_server lane host without the role's Rules: a structured request is refused, never drawn plain" {
    var c: fx.Chain(Bound) = .{};
    try c.init(false);
    defer c.deinit();
    var host = api.LaneHost.init(gpa, testing.io, &c.core, .{ .lanes = 1 });
    defer host.stop();
    try host.start();
    const Sink = struct {
        done: std.atomic.Value(bool) = .init(false),
        tokens: usize = 0,
        reason: api.Reason = .stop,
        msg: [128]u8 = undefined,
        len: usize = 0,
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |t| s.tokens += t.len,
                .finished => |f| {
                    s.reason = f.reason;
                    s.len = @min(f.message.len, s.msg.len);
                    @memcpy(s.msg[0..s.len], f.message[0..s.len]);
                    s.done.store(true, .release);
                },
                else => {},
            }
        }
    };
    var sink: Sink = .{};
    const req: api.Request = .{ .prompt = &.{ 3, 4 }, .max_tokens = 2, .drafts = false, .structure = .{ .kind = .json } };
    try host.engine().submit(1, &req, .{ .ctx = &sink, .event = Sink.event });
    var waited: usize = 0;
    while (!sink.done.load(.acquire) and waited < 500) : (waited += 1) try std.Io.sleep(testing.io, .fromMilliseconds(10), .awake);
    try testing.expect(sink.done.load(.acquire));
    try testing.expectEqual(api.Reason.failed, sink.reason);
    try testing.expectEqualStrings("IncompleteMaskedAdapter", sink.msg[0..sink.len]);
    try testing.expectEqual(@as(usize, 0), sink.tokens);
    std.debug.print("grammar_server lane host without Rules: structured request failed IncompleteMaskedAdapter, 0 tokens\n", .{});
}
