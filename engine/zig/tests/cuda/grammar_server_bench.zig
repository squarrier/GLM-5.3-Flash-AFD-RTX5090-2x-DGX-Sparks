//! grammar_server CPU bench (Claude Opus 5.5 via Hermes): Ash's server over loopback HTTP with a FAKE engine that records
//! what reaches it, a FAKE grammar source (api.Structure.Source) that counts opens and checks, and small checkpoint
//! directories under /tmp (the source tree is read-only in the build container). The server's code is the product's.
const std = @import("std");
const api = @import("engine_api");
const server = @import("server");
const h = @import("grammar_server_http.zig");
const testing = std.testing;
const gpa = testing.allocator;

/// FAKE engine: records each submitted request's structure, answers `reply` then stop.
pub const Fake = struct {
    structures: bool = false,
    submits: usize = 0,
    structured: usize = 0,
    kind: ?u32 = null,
    text: [512]u8 = undefined,
    text_len: usize = 0,
    compiler: ?*api.Structure.Compiler = null,
    after: ?u32 = null,
    reply: []const u32 = &.{ 'o', 'k' },

    pub fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn of(ctx: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return .{ .name = "fake", .structures = of(ctx).structures, .context_window = 64 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const f = of(ctx);
        f.submits += 1;
        if (r.structure) |s| {
            f.structured += 1;
            f.kind = @backingInt(s.kind);
            f.text_len = @min(s.text.len, f.text.len);
            @memcpy(f.text[0..f.text_len], s.text[0..f.text_len]);
            f.compiler = s.compiler;
            f.after = s.after;
        } else f.kind = null;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = f.reply });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    pub fn lastText(f: *const Fake) []const u8 {
        return f.text[0..f.text_len];
    }
};

/// FAKE grammar source: open hands out `&token` (or fails with `fail_open`), check passes unless `fail_kind` matches.
pub const Source = struct {
    opens: usize = 0,
    checks: usize = 0,
    fail_open: ?[]const u8 = null,
    fail_kind: ?u32 = null,
    vocab: u32 = 0,
    stops: [8]u32 = undefined,
    nstops: usize = 0,
    tokenizer_fnv: u64 = 0,
    kinds: [16]u32 = undefined,
    texts: [16][256]u8 = undefined,
    lens: [16]usize = undefined,
    token: u8 = 0,

    pub fn source(s: *Source) api.Structure.Source {
        return .{ .ctx = s, .open = open, .check = check };
    }
    pub fn compiler(s: *Source) *api.Structure.Compiler {
        return @ptrCast(&s.token);
    }
    fn of(ctx: *anyopaque) *Source {
        return @ptrCast(@alignCast(ctx));
    }
    fn open(ctx: *anyopaque, tokenizer_json: []const u8, vocab: u32, stops: []const u32, problem: *[]const u8) ?*api.Structure.Compiler {
        const s = of(ctx);
        s.opens += 1;
        s.vocab = vocab;
        s.nstops = @min(stops.len, s.stops.len);
        @memcpy(s.stops[0..s.nstops], stops[0..s.nstops]);
        s.tokenizer_fnv = std.hash.Fnv1a_64.hash(tokenizer_json);
        if (s.fail_open) |p| {
            problem.* = p;
            return null;
        }
        return s.compiler();
    }
    fn check(ctx: *anyopaque, c: *api.Structure.Compiler, kind: u32, grammar_text: []const u8, err: *[512]u8) bool {
        const s = of(ctx);
        std.debug.assert(c == s.compiler());
        if (s.checks < s.kinds.len) {
            s.kinds[s.checks] = kind;
            s.lens[s.checks] = @min(grammar_text.len, 256);
            @memcpy(s.texts[s.checks][0..s.lens[s.checks]], grammar_text[0..s.lens[s.checks]]);
        }
        s.checks += 1;
        if (s.fail_kind == kind) {
            const m = "FAKE: this grammar is refused";
            @memcpy(err[0..m.len], m);
            err[m.len] = 0;
            return false;
        }
        return true;
    }
    pub fn textAt(s: *const Source, i: usize) []const u8 {
        return s.texts[i][0..s.lens[i]];
    }
};

/// A checkpoint directory under /tmp with the given tokenizer.json and config.json (null: absent).
pub fn modelDir(a: std.mem.Allocator, name: []const u8, tokenizer: ?[]const u8, config: ?[]const u8) ![]const u8 {
    const dir = try std.fmt.allocPrint(a, "/tmp/grammar_server-{s}", .{name});
    try std.Io.Dir.cwd().createDirPath(testing.io, dir);
    if (tokenizer) |t| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "tokenizer.json" }), .data = t });
    if (config) |c| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = c });
    return dir;
}

/// GLM-5.3's config shape: text_config's vocab_size first (the top-level one is a decoy the release reads second).
pub const glm_config = "{\"vocab_size\": 1, \"text_config\": {\"vocab_size\": 154880}}";

/// The server over the FAKE engine on a loopback port: `structures` as the engine reports it, `source` as configured.
pub const Bench = struct {
    fake: Fake = .{},
    text: h.ByteText = .{ .eos = &.{ 154820, 154827, 154829 } },
    srv: *server.Server = undefined,
    http: h.Http = .{},

    pub fn start(b: *Bench, structures: bool, dir: []const u8, source: ?api.Structure.Source) !void {
        b.fake.structures = structures;
        b.srv = try server.Server.init(gpa, testing.io, b.fake.engine(), b.text.text(), .{ .served_name = "glm-5.3-flash", .model_ids = &.{"glm-5.3-flash"}, .model_dir = dir, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0, .grammars = source }, null);
        errdefer b.srv.deinit();
        try b.http.start(b.srv);
    }
    pub fn stop(b: *Bench) void {
        b.http.finish();
        b.srv.deinit();
    }
    pub fn chat(b: *Bench, a: std.mem.Allocator, extra: []const u8) ![]u8 {
        return b.http.post(a, "/v1/chat/completions", try h.chat(a, 2, false, extra));
    }
};

pub const unsupported = "structured output (response_format and the guided_* fields) is not supported by this engine yet";
