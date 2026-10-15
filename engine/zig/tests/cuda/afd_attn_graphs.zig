//! qualification_f16009a8's role checks of TF_AFD_GRAPHS on the attention node (tf-glm-afd attn --dry, layers 0-4): a bench of greedy and
//! drafted decodes (f3, auto) with the profile's marked steps through the graphs by key, recorded eagerly and every
//! later forward of a key compared with its first: the outcome, forwards, launches and exchanges are unset's; anything
//! but 1, 0 or unset refused before anything loads.
const std = @import("std");
const attn = @import("afd_attn.zig");
const ab = @import("afd_attn_bench.zig");

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// The process's own `name` copied into `buf` (a gate's run may set it for every test), to put it back after.
fn saved(name: [*:0]const u8, buf: *[32]u8) ?[:0]const u8 {
    const v = std.mem.span(std.c.getenv(name) orelse return null);
    if (v.len >= buf.len) return null;
    @memcpy(buf[0..v.len], v);
    buf[v.len] = 0;
    return buf[0..v.len :0];
}

test "qualification_f16009a8: attn bench --dry with TF_AFD_MTP=1 and TF_AFD_GRAPHS=1 (layers 0-4, drafts=f3+auto, profile=2): outcome, forwards, launches and exchanges unset's, every key's replays its first's; TF_AFD_GRAPHS=2 refused" {
    const testing = std.testing;
    const J = std.json.Value;
    const cli = @import("tf_glm_afd.zig");
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try std.process.currentPathAlloc(testing.io, a);
    const out = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    var pj: std.ArrayList(u8) = .empty;
    try pj.appendSlice(a, "{\"prompts\": {\"a\": [");
    for (0..37) |i| try pj.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", 100 + i });
    try pj.appendSlice(a, "], \"b\": [");
    for (0..40) |i| try pj.print(a, "{s}{d}", .{ if (i == 0) "" else ", ", 200 + i });
    try pj.appendSlice(a, "]}}");
    const pfile = try std.fmt.allocPrint(a, "{s}/p.json", .{out});
    try attn.save(pfile, pj.items);
    const spec = "decode=a:12,prefill=b,runs=2,warmup=1,skip=1,drafts=f3+auto,profile=2";
    var pb: [4096]u8 = undefined;
    var t = try attn.tokens(testing.allocator, try ab.decodePrompt(&pb, pfile, spec), null);
    defer t.deinit(testing.allocator);
    const root = try std.json.parseFromSliceLeaky(J, a, @embedFile("afd_attn_fixture"), .{});
    const config = root.object.get("config_json").?.string;
    const f: cli.Flags = .{ .role = .attn, .verb = .bench, .model = "-", .layers = 5, .out = out, .dry = true, .prompt = pfile, .bench = spec };
    var kb: [32]u8 = undefined; // the process's own TF_AFD_GRAPHS (a gate's run sets it for every test), put back after
    const prior = saved("TF_AFD_GRAPHS", &kb);
    defer _ = if (prior) |p| setenv("TF_AFD_GRAPHS", p, 1) else unsetenv("TF_AFD_GRAPHS");
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    _ = unsetenv("TF_AFD_GRAPHS");
    const off = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(off.outcome.ok);
    _ = setenv("TF_AFD_GRAPHS", "1", 1);
    const on = try attn.dry(testing.allocator, f, &t, config);
    defer testing.allocator.free(on.outcome.ok);
    try testing.expectEqualStrings(off.outcome.ok, on.outcome.ok);
    inline for (.{ "forwards", "launches", "exchanges", "collects", "records" }) |name| try testing.expectEqual(@field(off, name), @field(on, name));
    try testing.expectEqual(@as(usize, 0), off.graphs.captures + off.graphs.replays);
    try testing.expect(on.graphs.captures > 0 and on.graphs.replays > on.graphs.captures);
    try testing.expect(on.graphs.captures + on.graphs.replays < on.forwards); // prompt chunks and the profile's steps eager
    std.debug.print("G-trace attn --dry bench (MTP, f3+auto, profile 2): {d} forwards, {d} through the graphs ({d} keys, {d} replays each its key's first's), {d} launches as unset's\n", .{ on.forwards, on.graphs.captures + on.graphs.replays, on.graphs.captures, on.graphs.replays, on.launches });
    _ = setenv("TF_AFD_GRAPHS", "2", 1);
    try testing.expectError(error.BadAfdGraphs, attn.dry(testing.allocator, f, &t, config));
}
