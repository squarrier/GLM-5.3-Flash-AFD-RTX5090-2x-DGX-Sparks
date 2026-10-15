//! Drafts in tf-glm-afd attn serve (qualification_4be15bbf, the lab branch only: qualification_0256caaf's MTP head under AFD, TF_AFD_MTP=1):
//! `-- --drafts 0|N|fN|a|a:LOW:HIGH|cN:P|auto`, qualification_qualification_e0f4f003's policy kinds (glm_drafts.parse; 0 is off), as qualification_qualification_e91d6b91-serve serves
//! drafts on two Sparks (native/glm_cuda_drafts.zig). With drafts on, the prompt pass absorbs into the MTP head
//! (Attn.absorb); then the lane host's driver for a lone greedy stream decodes it by qualification_21e21bc0's loop for the policy
//! (cuda_afd_drafts.mtpWith or autoWith: the head's drafts, the verify window over both halves, the keep), whose hook
//! lands each round's kept tokens in the stream; the stream's own rules (eos, stop strings, max_tokens) or a cancel
//! end the decode after that round. A request past the drafted reach (the dense limit 2,051 unless TF_GLM_DRAFT_REACH
//! says more, qualification_qualification_e91d6b91-serve's rule) is handed to the lane core after its prompt pass and decodes in one-row rounds.
const std = @import("std");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const gd = @import("glm_drafts.zig");
const gl = @import("glm_lanes");
const sl = @import("afd_serve_lanes.zig");

const decode = glm.decode;
const ad = glm.afd_drafts;
pub const Drafts = gd.Drafts;
const Allocator = std.mem.Allocator;

/// `--drafts`'s policy (null or 0: off); on needs TF_AFD_MTP=1 (the MTP head loaded), else refused in words.
pub fn parse(spec: ?[]const u8) !gd.Drafts {
    const s = spec orelse return .{};
    const d = gd.parse(s) catch {
        std.debug.print("INFO attn serve: --drafts {s}: 0 (off), N or fN (N fixed), a or a:LOW:HIGH (adaptive), cN:P (N under a confidence floor P) or auto (python's greedy c3:0.35)\n", .{s});
        return error.ServeDrafts;
    };
    if (d.on() and !(try glm.afd_mtp.fromEnv())) {
        std.debug.print("INFO attn serve: --drafts {s} needs TF_AFD_MTP=1 (the MTP head on the attention node, its experts on both halves)\n", .{s});
        return error.DraftsNeedTfAfdMtp;
    }
    return d;
}

/// The drafted reach: TF_GLM_DRAFT_REACH (a whole number, never past `reach`), else the dense limit.
pub fn reachOf(dense_limit: usize, reach: usize) !usize {
    const v = std.c.getenv("TF_GLM_DRAFT_REACH") orelse return @min(dense_limit, reach);
    const n = std.fmt.parseInt(usize, std.mem.trim(u8, std.mem.span(v), " "), 10) catch return error.DraftReach;
    return @min(n, reach);
}

/// The policy's loop on the attention node, each round's kept tokens to `hook` (null: the tool's own run).
pub fn decodeWith(at: *glm.afd_engine.Attn, f: *glm.fwd.Fwd, d: gd.Drafts, pending: u32, count: usize, out: []u32, hook: ?*const decode.Hook) !decode.Decoded {
    var policy = d.policy;
    return switch (d.kind) {
        .serial => error.DraftsOff,
        .mtp => ad.mtpWith(at, f, pending, count, out, &policy, false, null, hook),
        .auto => blk: {
            policy = decode.DepthPolicy.auto(at.e.sampling == null or at.e.sampling.?.temperature <= 0.0);
            break :blk ad.autoWith(at, f, pending, count, out, &policy, false, null, hook);
        },
    };
}

/// The hook: a round's kept tokens into the stream, then the host told; the stream's end or a cancel ends the decode.
const Hooked = struct {
    gpa: Allocator,
    s: *lanes.Stream,
    hooks: api.LoneHooks,
    failed: ?anyerror = null,

    fn before(_: *anyopaque, _: decode.Hook.Op, _: []const u32) anyerror!void {} // one node: no rank 1 to tell

    fn kept(p: *anyopaque, tokens: []const u32, _: usize, _: usize) anyerror!bool {
        const h: *Hooked = @ptrCast(@alignCast(p));
        _ = h.s.commit(h.gpa, tokens) catch |e| {
            h.failed = e;
            return true;
        };
        h.hooks.committed(h.hooks.ctx);
        return h.s.finished or h.s.isCancelled();
    }
};

/// The lane host's driver for a lone drafted stream over the served calls (api.Lone; native/glm_cuda_drafts.Lone's
/// shape without rank 1).
pub const Lone = struct {
    gpa: Allocator,
    back: *sl.Backend,
    d: gd.Drafts,
    /// prompt plus max_tokens a drafted stream may reach; past it, the lane core's one-row rounds
    reach: usize,

    pub fn driver(l: *Lone) api.Lone {
        return .{ .ctx = l, .run = run };
    }

    fn run(p: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *Lone = @ptrCast(@alignCast(p));
        const be = l.back.backend();
        _ = try be.opening(l.gpa, s); // the prompt pass (absorbing) and its first token, committed (Lanes.prefill)
        hooks.committed(hooks.ctx);
        const b = l.back.steps;
        if (!s.finished and s.prompt_len + s.max_new > l.reach) {
            s.cache_len = b.len; // the lane core's adopt: the cache settled, `pending` the last token
            s.pending = s.context.items[s.context.items.len - 1];
            return true;
        }
        defer be.release(s);
        if (s.finished) return false;
        const count: usize = 1 + (s.max_new - s.emitted().len); // the pending token, then what is left
        const out = try l.gpa.alloc(u32, count);
        defer l.gpa.free(out);
        var h: Hooked = .{ .gpa = l.gpa, .s = s, .hooks = hooks };
        const hook: decode.Hook = .{ .ctx = &h, .before = Hooked.before, .kept = Hooked.kept };
        const r = try decodeWith(b.s.at, b.s.f, l.d, s.context.items[s.context.items.len - 1], count, out, &hook); // the context is bound: opening's prefill bound it on this thread
        b.s.len = b.s.at.e.st.pos;
        b.len = b.s.len;
        s.rounds += r.rounds;
        s.drafted += r.drafted;
        s.accepted += r.accepted;
        std.debug.print("INFO attn serve: drafts {s}: {d} tokens in {d} verify forwards ({d:.3} a verify forward), drafted {d}, accepted {d} ({d:.4}); {d:.1} ms\n", .{ l.d.spec, r.tokens.len -| 1, r.rounds, r.perForward(), r.drafted, r.accepted, r.acceptance(), @as(f64, @floatFromInt(r.ns)) / 1e6 });
        if (h.failed) |e| return e;
        if (!s.finished and s.isCancelled()) return error.Cancelled;
        return false;
    }
};

/// The warm-up with drafts on: 96 fixed ids absorbed, then 16 tokens by the policy's loop (qualification_qualification_e91d6b91-serve's warm-up shape).
pub fn warmUp(b: *sl.Bound, d: gd.Drafts, out: *[gl.warm_tokens]u32) !decode.Decoded {
    var ids: [gl.warm_prompt]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + i);
    const first = try b.prefill(&ids);
    defer b.release();
    const r = try decodeWith(b.s.at, b.s.f, d, first, gl.warm_tokens, out, null);
    b.s.len = b.s.at.e.st.pos;
    b.len = b.s.len;
    return r;
}

// -- G-draft-serve (qualification_4be15bbf lab): the served drafted stream on a recorded node --------------------------------------------

const testing = std.testing;
const trace = glm.trace;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// qualification_7877904d's scripted answers (afd_cover_qmm.zig's Script): greedy picks token 0; a draft's draw 0.9, every third token 1.
const Script = struct {
    drafts: usize = 0,
    e: ?*glm.engine.Engine = null,
    sampled_rows: usize = 0,
    position: u64 = 0,
    fn pick(ctx: *anyopaque, _: trace.T, positions: ?[]const u64, out: []u32, probs: ?[]f64) anyerror!void {
        const s: *Script = @ptrCast(@alignCast(ctx));
        if (positions) |ps| {
            if (s.e.?.sampling) |p| if (p.temperature > 0) {
                for (out, ps) |*o, pos| o.* = @intFromBool(glm.nucleus_draw.uniform(p.seed, pos, 0) > 0.5);
                s.sampled_rows += out.len;
                s.position = ps[0];
                return;
            };
        }
        @memset(out, 0);
        const p = probs orelse return;
        for (out, 0..) |*o, i| {
            s.drafts += 1;
            if (s.drafts % 3 == 0) o.* = 1;
            if (i < p.len) p[i] = 0.9;
        }
    }
};

/// attn --dry's node (afd_attn.dry): the fixture's model cut to 5 layers with its MTP head (TF_AFD_MTP=1), B3's Engine
/// and the provider at stand-ins, the forward recorded with the scripted answers.
const Node = struct {
    m: glm.model.Model = undefined,
    w: glm.views.Weights = undefined,
    space: trace.Space = .{ .gpa = testing.allocator },
    e: *glm.engine.Engine = undefined,
    a: glm.afd_attn.Afd = undefined,
    at: glm.afd_engine.Attn = undefined,
    tr: trace.Trace = undefined,
    script: Script = .{},
    f: glm.fwd.Fwd = undefined,

    fn start(n: *Node, config: []const u8) !void {
        return n.startWithSampling(config, false);
    }

    fn startWithSampling(n: *Node, config: []const u8, sampled: bool) !void {
        const gpa = testing.allocator;
        n.m = try @import("afd_attn_drafts.zig").model(gpa, config, 5);
        n.w = try glm.views.rank(gpa, &n.m);
        try glm.views.walk(&n.w, glm.views.Path.init("w"), &n.space, glm.afd_load.skipExperts(*trace.Space, trace.placeAll));
        n.e = try glm.engine.Engine.create(gpa, &n.m, &n.w, 2560, 8, 2048, false);
        if (sampled) {
            try n.e.useSampling(gpa);
            n.e.sampled_top_k = true;
        }
        n.script.e = n.e;
        try n.e.walk(&n.space, trace.placeAll);
        try n.e.walkTmp(&n.space, trace.placeAll);
        n.e.tmp.space = &n.space;
        n.a = try glm.afd_attn.Afd.initWith(&n.m, 2048, n.m.mtp_layers > 0);
        try n.space.place("b.afd_recv", &n.a.afd_recv);
        n.at = try glm.afd_engine.Attn.init(n.e, &n.a);
        n.tr = trace.Trace.init(gpa);
        n.tr.picker = .{ .ctx = &n.script, .pick = Script.pick };
        n.f = glm.fwd.Fwd.recording(&n.m, &n.tr);
    }

    fn free(n: *Node) void {
        n.tr.deinit();
        n.e.destroy(testing.allocator);
        n.space.deinit();
        n.w.deinit(testing.allocator);
    }

    fn digest(n: *Node) !u64 {
        try n.f.flush();
        var h = std.hash.Wyhash.init(0);
        for (n.tr.events.items) |ev| {
            h.update(ev.name);
            h.update(std.mem.asBytes(&ev.grid));
            for (ev.args) |x| {
                h.update(x.name);
                switch (x.value) {
                    .ptr => |p| h.update(std.mem.asBytes(&p.addr)),
                    .i32 => |v| h.update(std.mem.asBytes(&v)),
                    .f32 => |v| h.update(std.mem.asBytes(&v)),
                    .u64 => |v| h.update(std.mem.asBytes(&v)),
                }
            }
        }
        return h.final();
    }
};

/// One served request on `n` through the lane host (drafts `d`, or the lane core's one-row rounds when off).
fn served(n: *Node, d: Drafts, prompt: []const u32, count: u32, out: []u32) !usize {
    return servedWith(n, d, prompt, count, out, null, false);
}

fn servedWith(n: *Node, d: Drafts, prompt: []const u32, count: u32, out: []u32, p: ?api.Sampling, enabled: bool) !usize {
    var s: glm.afd_serve.Serve = .{ .at = &n.at, .f = &n.f };
    var b: sl.Bound = .{ .s = &s, .drafts = d.on(), .sampled = enabled };
    var back: sl.Backend = .{ .steps = &b, .capacity = 2560, .reach = 2051 };
    var words: sl.Words = .{ .node = null };
    var lone: Lone = .{ .gpa = testing.allocator, .back = &back, .d = d, .reach = 2051 };
    n.at.absorb = d.on();
    var ln: sl.Lanes = .{ .gpa = testing.allocator };
    try ln.startWith(testing.io, &back, .{ .lanes = 1, .context_window = 2496, .drafts = d.spec }, &words, if (d.on()) lone.driver() else null);
    defer ln.stop();
    const Got = struct {
        out: []u32,
        n: usize = 0,
        reason: ?api.Reason = null,
        done: std.atomic.Value(bool) = .init(false),
        fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
            const g: *@This() = @ptrCast(@alignCast(ctx));
            switch (ev.*) {
                .tokens => |t| for (t) |x| {
                    if (g.n < g.out.len) g.out[g.n] = x;
                    g.n += 1;
                },
                .finished => |fin| {
                    g.reason = fin.reason;
                    g.done.store(true, .release);
                },
                else => {},
            }
        }
    };
    var got: Got = .{ .out = out };
    try ln.host.engine().submit(1, &.{ .prompt = prompt, .max_tokens = count, .drafts = d.on(), .sampling = p }, .{ .ctx = &got, .event = Got.event });
    var waited: usize = 0;
    while (!got.done.load(.acquire)) : (waited += 1) {
        if (waited > 2000) return error.Hung;
        const ts: std.c.timespec = .{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }
    if (got.reason != .length) return error.NotServed;
    return got.n;
}

test "G-draft-serve (qualification_4be15bbf lab): a served drafted request (f3, auto) makes the tool's own drafted run's launches and tokens; its tokens are the drafts-off served request's (G-draft-equal)" {
    const root = try std.json.parseFromSlice(std.json.Value, testing.allocator, @embedFile("afd_attn_fixture"), .{});
    defer root.deinit();
    const config = root.value.object.get("config_json").?.string;
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    var prompt: [37]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(100 + i);
    const count = 12;
    var off_out: [count]u32 = undefined;
    var off: Node = .{};
    try off.start(config);
    defer off.free();
    try testing.expectEqual(@as(usize, count), try served(&off, .{}, &prompt, count, &off_out)); // drafts off: one-row rounds
    for ([_][]const u8{ "f3", "auto" }) |spec| {
        const d = try parse(spec);
        // the tool's own run (attn bench's drafted decode: the prompt absorbed, then the policy's loop), one Fwd
        var tool: Node = .{};
        try tool.start(config);
        defer tool.free();
        tool.at.absorb = true;
        var tool_out: [count]u32 = undefined;
        const r = try decodeWith(&tool.at, &tool.f, d, try tool.at.prefill(&tool.f, &prompt), count, &tool_out, null);
        const tool_d = try tool.digest();
        // the served drafted request
        var srv: Node = .{};
        try srv.start(config);
        defer srv.free();
        var srv_out: [count]u32 = undefined;
        try testing.expectEqual(@as(usize, count), try served(&srv, d, &prompt, count, &srv_out));
        const srv_d = try srv.digest();
        std.debug.print("G-draft-serve {s}: the tool {d} launches {x:0>16}, served {d} launches {x:0>16}; rounds {d}, drafted {d}, accepted {d}; tokens {any} (drafts off {any})\n", .{ spec, tool.tr.events.items.len, tool_d, srv.tr.events.items.len, srv_d, r.rounds, r.drafted, r.accepted, srv_out, off_out });
        try testing.expect(r.drafted > 0 and r.accepted > 0 and (d.kind != .auto or r.accepted < r.drafted)); // auto: the script refuses some
        try testing.expectEqual(tool.tr.events.items.len, srv.tr.events.items.len);
        try testing.expectEqual(tool_d, srv_d);
        try testing.expectEqualSlices(u32, &tool_out, &srv_out);
        try testing.expectEqualSlices(u32, &off_out, &srv_out); // G-draft-equal
    }
}

test "sampling_tests real AFD lane: sampling fields, seeded repeat, greedy reset and sampled drafts disabled" {
    const root = try std.json.parseFromSlice(std.json.Value, testing.allocator, @embedFile("afd_attn_fixture"), .{});
    defer root.deinit();
    const config = root.value.object.get("config_json").?.string;
    _ = setenv("TF_AFD_MTP", "1", 1);
    defer _ = unsetenv("TF_AFD_MTP");
    const prompt = [_]u32{ 100, 101, 102 };
    const count = 12;
    var node: Node = .{};
    try node.startWithSampling(config, true);
    defer node.free();
    const d = try parse("auto");
    const p: api.Sampling = .{ .seed = 9876, .temperature = 0.7, .top_p = 0.9, .top_k = 20, .min_p = 0.05 };
    var first: [count]u32 = undefined;
    var again: [count]u32 = undefined;
    try testing.expectEqual(@as(usize, count), try servedWith(&node, d, &prompt, count, &first, p, true));
    try testing.expect(!node.at.absorb);
    try testing.expectEqual(@as(usize, count), node.script.sampled_rows);
    try testing.expectEqual(@as(u64, prompt.len + count - 1), node.script.position);
    try testing.expectEqual(@as(usize, 0), node.script.drafts);
    const np = node.e.sampling.?;
    try testing.expectEqual(p.seed, np.seed);
    try testing.expectEqual(p.temperature, np.temperature);
    try testing.expectEqual(p.top_p, np.top_p);
    try testing.expectEqual(p.top_k, np.top_k);
    try testing.expectEqual(p.min_p, np.min_p);
    try testing.expectEqual(@as(usize, count), try servedWith(&node, d, &prompt, count, &again, p, true));
    try testing.expectEqualSlices(u32, &first, &again);
    const sampled_rows = node.script.sampled_rows;
    var greedy: [count]u32 = undefined;
    var zero: [count]u32 = undefined;
    try testing.expectEqual(@as(usize, count), try servedWith(&node, .{}, &prompt, count, &greedy, null, true));
    try testing.expect(node.e.sampling == null);
    try testing.expectEqual(@as(usize, count), try servedWith(&node, .{}, &prompt, count, &zero, .{ .seed = 1, .temperature = 0 }, true));
    try testing.expectEqualSlices(u32, &greedy, &zero);
    try testing.expectEqual(sampled_rows, node.script.sampled_rows);
    try testing.expectEqual(@as(usize, count), try servedWith(&node, d, &prompt, count, &again, null, true));
    try testing.expect(node.at.absorb);
    try testing.expect(node.script.drafts > 0);
    try testing.expectEqualSlices(u32, &greedy, &again);
    try testing.expectError(error.NotServed, servedWith(&node, .{}, &prompt, count, &again, p, false));
    std.debug.print("sampling_tests lane: fixed seed repeat, fields forwarded, {d} sampled rows; zero/omitted greedy; MTP drafts/absorb off for sampled, off-switch refuses\n", .{sampled_rows});
}
