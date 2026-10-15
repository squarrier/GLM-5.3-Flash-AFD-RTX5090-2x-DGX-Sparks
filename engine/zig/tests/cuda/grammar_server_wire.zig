//! grammar_server-WIRE (Claude Opus 5.5 via Hermes): default-off identity on the served chain. With every TF_AFD_GRAMMAR_*
//! unset the attention role builds no grammar source, no lane-host attachment and no adapter (structures false), so
//! this harness serves exactly that: Ash's server over loopback HTTP -> api.LaneHost -> the lane core -> the actual
//! AFD Lanes(Bound) -> Serve -> Attn -> sampler chain (CPU recording, MOCK picks; grammar_mask's fixture). It writes each
//! raw reply to $grammar_server_WIRE_OUT and prints the recorded trace's digest. It uses only API the pristine 045dd856 has:
//! scripts/wire.sh compiles this same file on the grammar_server candidate and on the baseline (an overlay copy adding only
//! this test and its step) and compares the bytes, the volatile fields (Date, created, ids, timings, lengths) masked.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const server = @import("server");
const fx = @import("grammar_mask_fixture.zig");
const Bound = @import("afd_serve_lanes.zig").Bound;
const h = @import("grammar_server_http.zig");
const testing = std.testing;
const gpa = testing.allocator;

const Case = struct { name: []const u8, path: []const u8, body: ?[]const u8, status: u16 };

fn cases(a: std.mem.Allocator) ![]const Case {
    const rf = struct {
        fn of(x: std.mem.Allocator, stream: bool, extra: []const u8) ![]const u8 {
            return h.chat(x, 3, stream, extra);
        }
    }.of;
    var out: std.ArrayList(Case) = .empty;
    try out.append(a, .{ .name = "health", .path = "/health", .body = null, .status = 200 });
    try out.append(a, .{ .name = "models", .path = "/v1/models", .body = null, .status = 200 });
    try out.append(a, .{ .name = "chat", .path = "/v1/chat/completions", .body = try rf(a, false, ""), .status = 200 });
    try out.append(a, .{ .name = "chat-stream", .path = "/v1/chat/completions", .body = try rf(a, true, ""), .status = 200 });
    try out.append(a, .{ .name = "completion", .path = "/v1/completions", .body = "{\"model\":\"glm-5.3-flash\",\"prompt\":\"hi\",\"max_tokens\":3,\"temperature\":0}", .status = 200 });
    try out.append(a, .{ .name = "json-object", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"response_format\":{\"type\":\"json_object\"}"), .status = 400 });
    try out.append(a, .{ .name = "json-object-stream", .path = "/v1/chat/completions", .body = try rf(a, true, ",\"response_format\":{\"type\":\"json_object\"}"), .status = 400 });
    try out.append(a, .{ .name = "json-schema", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"r\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"]}}}"), .status = 400 });
    try out.append(a, .{ .name = "completion-json-object", .path = "/v1/completions", .body = "{\"model\":\"glm-5.3-flash\",\"prompt\":\"hi\",\"max_tokens\":3,\"temperature\":0,\"response_format\":{\"type\":\"json_object\"}}", .status = 400 });
    try out.append(a, .{ .name = "guided-regex", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"guided_regex\":\"[a-z]+\""), .status = 400 });
    try out.append(a, .{ .name = "type-text", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"response_format\":{\"type\":\"text\"}"), .status = 200 });
    try out.append(a, .{ .name = "type-unknown", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"response_format\":{\"type\":\"xml\"}"), .status = 400 });
    try out.append(a, .{ .name = "schema-missing", .path = "/v1/chat/completions", .body = try rf(a, false, ",\"response_format\":{\"type\":\"json_schema\"}"), .status = 400 });
    try out.append(a, .{ .name = "chat-after", .path = "/v1/chat/completions", .body = try rf(a, false, ""), .status = 200 });
    return out.toOwnedSlice(a);
}

test "grammar_server-WIRE default off: raw replies and the recorded trace of the served chain with every TF_AFD_GRAMMAR_* unset" {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var c: fx.Chain(Bound) = .{};
    try c.init(false); // TF_AFD_GRAMMAR_MASK unset: no adapter
    defer c.deinit();
    var picks: [96][]const fx.Pair = undefined;
    for (&picks, 0..) |*p, i| p.* = try a.dupe(fx.Pair, &.{.{ .token = @intCast('a' + i % 26), .value = 100 }});
    var sched = try fx.Sched.init(c.rec.m.vocab, &picks);
    defer sched.deinit();
    c.rec.t.picker = sched.picker();
    var host = api.LaneHost.init(gpa, testing.io, &c.core, .{ .name = "glm-afd-attn", .lanes = 1 });
    var host_up = true;
    defer if (host_up) host.stop();
    try host.start();
    var text: h.ByteText = .{};
    const srv = try server.Server.init(gpa, testing.io, host.engine(), text.text(), .{ .served_name = "glm-5.3-flash", .model_ids = &.{"glm-5.3-flash"}, .enable_thinking = false, .use_drafts = false, .keep_warm_s = 0 }, null);
    var srv_up = true;
    defer if (srv_up) srv.deinit();
    try testing.expect(!srv.info.structures);
    var http: h.Http = .{};
    try http.start(srv);
    var http_up = true;
    defer if (http_up) http.finish();
    const out_dir: ?[]const u8 = if (std.c.getenv("grammar_server_WIRE_OUT")) |p| std.mem.span(p) else null;
    if (out_dir) |d| try std.Io.Dir.cwd().createDirPath(testing.io, d);
    for (try cases(a), 0..) |k, i| {
        const raw = if (k.body) |b| try http.post(a, k.path, b) else try http.get(a, k.path);
        std.debug.print("grammar_server-WIRE {d:0>2}-{s} status={d} bytes={d}\n", .{ i, k.name, h.status(raw), raw.len });
        try testing.expectEqual(k.status, h.status(raw));
        if (out_dir) |d| {
            const name = try std.fmt.allocPrint(a, "{s}/{d:0>2}-{s}.http", .{ d, i, k.name });
            try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = name, .data = raw });
        }
    }
    http.finish();
    http_up = false;
    srv.deinit();
    srv_up = false;
    host.stop();
    host_up = false;
    try c.rec.t.flush();
    var fnv = std.hash.Fnv1a_64.init();
    for (c.rec.t.events.items) |e| {
        fnv.update(e.name);
        fnv.update(std.mem.asBytes(&e.grid));
        const op: u32 = @backingInt(e.op);
        fnv.update(std.mem.asBytes(&op));
    }
    std.debug.print("grammar_server-WIRE trace events={d} fnv={x:0>16} picks={d} device_calls={d}\n", .{ c.rec.t.events.items.len, fnv.final(), sched.calls, sched.device_calls });
    try testing.expectEqual(@as(usize, 0), sched.device_calls);
}
