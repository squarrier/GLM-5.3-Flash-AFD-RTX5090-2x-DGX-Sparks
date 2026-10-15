//! qualification_qualification_ea348a94's launch trace (qualification_qualification_7c75d777 PLAN §4): the DSA arms on prompt chunks past 512 rows, as python-0.6's decode.prefill runs
//! them (tools/zig/glm_forward_fixtures.py --dsa-prompt: fixtures_cuda_dsa.json, MLX 4-bit; *_exl3.json, EXL3): layer 3
//! whole and the MTP layer's dsa_block on the Engine's pbuf with long contexts. Dense blocks start at pos_dev + r0 and
//! select blocks at pos_dev + a (torch adds), the blocks' pools are joined by torch.cat, and a 2,048-row chunk's sparse
//! partials take dk.Scratch (python's torch.empty). The rank, its buffers and the check are cuda_layer_test.zig's.
const std = @import("std");
const views = @import("cuda_views.zig");
const bufs = @import("cuda_buffers.zig");
const trace = @import("cuda_trace.zig");
const fwd = @import("cuda_fwd.zig");
const layer = @import("cuda_layer.zig");
const dsa = @import("cuda_dsa.zig");
const dk = @import("cuda_dsa_kernels.zig");
const lt = @import("cuda_layer_test.zig");

const testing = std.testing;
const J = std.json.Value;
const get = lt.get;
const int = lt.int;
const fixture_mlx = @embedFile("fixtures_cuda_dsa.json");
const fixture_exl3 = @embedFile("fixtures_cuda_dsa_exl3.json");

/// forward.layer_forward of a DSA + MoE layer (cuda_layer.layerForward's steps), its DSA block with `x`.
fn wholeLayer(f: *fwd.Fwd, l: *const views.Layer, st: *const bufs.State, b: *const bufs.Buffers, R: usize, opt: layer.Opt, x: *const dk.Scratch) !void {
    const m = f.m;
    const h = l.attn_hc orelse return error.PlainLayer;
    try f.tri.hcPre(b.x.ptr, .{ .fn_w = h.@"fn".ptr, .base = h.base.ptr, .scale = h.scale.ptr }, l.in_norm.ptr, .{ .out = b.normed.ptr, .xs = b.xs.ptr, .post = b.post.ptr, .comb = b.comb.ptr, .part = b.hcpart.ptr }, m.eps, m.hc_eps, m.hc_iters, R, m.hidden, m.streams);
    const g = try dsa.blockWith(f, l, st, b, R, opt, x);
    try f.tri.hcPost(b.x.ptr, b.x.ptr, g, b.post.ptr, b.comb.ptr, m.world, R, m.hidden, m.streams);
    try lt.mlpHalf(f, l, b, R);
}

/// A Scratch for the Extra's pbuf rows, placed in its space (stand-in addresses, named dsa.po, dsa.pm, dsa.pl).
fn scratch(m: *const @import("cuda_model.zig").Model, x: *lt.Extra) !dk.Scratch {
    const pb = x.pb orelse return error.NoPromptBuffers;
    var s = dk.Scratch.init(m.hl(), pb.lat_s.qa.shape[2], pb.rows);
    try views.walk(&s, views.Path.init("dsa"), &x.space, trace.placeAll);
    return s;
}

/// Cases checked: layer 3 whole on prompt chunks, the MTP layer's dsa_block on prompt chunks.
const Counts = struct { layer: usize = 0, mtp: usize = 0, adds: usize = 0, cats: usize = 0, scratch: usize = 0 };

fn expectPrompt(text: []const u8) !Counts {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(J, gpa, text, .{});
    defer parsed.deinit();
    const fx = parsed.value;
    try testing.expectEqualStrings("glm-forward-trace/1", get(fx, "schema").string);
    var r = try lt.Rank.init(gpa, fx);
    defer r.deinit(gpa);
    try lt.expectBuffers(&r.space, get(fx, "buffers"), false);
    try lt.expectBuffers(&r.space, get(fx, "mtp_buffers"), true);
    var long = try lt.Extra.init(gpa, &r, &r.m, get(fx, "long_state"), "st.", true);
    defer long.deinit(gpa);
    var mm = r.m; // the MTP load keeps the MTP layer: its cache and the indexer's last
    mm.mtp_layers = @intCast(int(get(get(fx, "config"), "mtp_layers")));
    var mtp = try lt.Extra.init(gpa, &r, &mm, get(fx, "mtp_state"), "st.", true);
    defer mtp.deinit(gpa);
    var t = trace.Trace.init(gpa);
    defer t.deinit();
    const scratches = [2]dk.Scratch{ try scratch(&r.m, &long), try scratch(&r.m, &mtp) };
    var n: Counts = .{};
    var it = get(fx, "cases").object.iterator();
    while (it.next()) |kv| {
        const c = kv.value_ptr.*;
        errdefer std.debug.print("case {s}\n", .{kv.key_ptr.*});
        t.reset();
        const is_mtp = c.object.get("block") != null; // "dsa": mtp.py's dsa_block alone, host_pos its mtp_len
        const x = if (is_mtp) &mtp else &long;
        const s = &scratches[@intFromBool(is_mtp)];
        const pb = if (x.pb) |*p| p else return error.NoPromptBuffers;
        try testing.expectEqual(int(get(c, "prefill")), @as(i64, @intCast(pb.rows)));
        try testing.expectEqual(int(get(c, "capacity")), @as(i64, @intCast(x.st.capacity)));
        try testing.expect(get(c, "long").bool);
        x.st.pos = @intCast(int(get(c, "pos")));
        x.st.parity = @intCast(int(get(c, "parity")));
        const rows: usize = @intCast(int(get(c, "rows")));
        const opt: layer.Opt = .{ .nch = @intCast(int(get(c, "nch"))), .host_pos = x.st.pos };
        var f = fwd.Fwd.recording(&r.m, &t);
        if (is_mtp) {
            try testing.expectEqualStrings("mtp", get(c, "layer").string);
            x.st.mtp_len = x.st.pos;
            _ = try dsa.blockWith(&f, &(r.mtp orelse return error.NoMtpLayer), &x.st, pb, rows, opt, s);
            n.mtp += 1;
        } else {
            try wholeLayer(&f, &r.w.layers[@intCast(int(get(c, "layer")))], &x.st, pb, rows, opt, s);
            n.layer += 1;
        }
        try f.flush();
        var ck: lt.Check = .{ .space = &r.space, .extra = &x.space, .tmp = c.object.get("tmp") };
        try lt.expectCase(&ck, get(c, "events").array.items, t.events.items);
        for (t.events.items) |e| {
            n.adds += @intFromBool(std.mem.eql(u8, e.name, "torch.add"));
            n.cats += @intFromBool(std.mem.eql(u8, e.name, "torch.cat"));
        }
        for (ck.bound) |b| if (b) |y| { // a temporary bound in the Scratch: the partials past b.lat_s
            n.scratch += @intFromBool(std.mem.startsWith(u8, y.path, "dsa."));
        };
    }
    return n;
}

/// Each fixture: layer 3 at (2,048, 0), (37, 2,048), (2,048, 2,048), (904, 4,096), (1,537, 0) and the MTP layer at
/// (2,048, 0), (36, 2,048), (2,048, 2,048), (903, 4,096); 23 block positions, 4 cats, the 2,048-row sparse chunks'
/// partials (po, pm, pl) in the Scratch.
const all_cases: Counts = .{ .layer = 5, .mtp = 4, .adds = 23, .cats = 4, .scratch = 6 };

test "MLX 4-bit: layer 3 and the MTP layer's DSA arm on prompt chunks past 512 rows, python-0.6's launches (A4)" {
    try testing.expectEqual(all_cases, try expectPrompt(fixture_mlx));
}

test "EXL3: the same prompt chunks, absorb as BF16 blocks and the MoE through EXL3's routed experts (A4)" {
    try testing.expectEqual(all_cases, try expectPrompt(fixture_exl3));
}
