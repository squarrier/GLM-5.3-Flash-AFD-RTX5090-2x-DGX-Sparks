//! tf-exl3-check's K4 half (qualification_e4ad04cd): TF_GLM_HALVES_KERNEL=k4 takes K4's table (cuda_exl3_check_k4.zig) against
//! exl3_check.py's K4 manifest (TF_EXL3_CHECK_K4=1 on python 2.15's tree): the layer's shared suh as python's
//! share_suh tests it (the row uploaded), prompt_kernel_ready, each case's routed launches through K4
//! (cuda_exl3_k4.routed) and its y and xd against python 2.15's bytes. --bench N: each case's K4 launches and today's
//! routed launches (routedSched, python-0.6's kernels) on the same inputs N times, alternating run by run, then K4's
//! gate/up and down alone; BENCH and BENCH-AB lines, and the outputs compared again after.

const std = @import("std");
const glm = @import("glm_cuda");
const cuda = @import("cuda");
const tool = @import("glm_exl3_check.zig");

const x3 = glm.exl3;
const chk = x3.check;
const ck = x3.check_k4;
const k4 = x3.k4;
const ex = x3.experts;
const Obj = std.json.ObjectMap;
const Tool = tool.Tool;

/// TF_GLM_HALVES_KERNEL as the tool takes it: unset, empty or 0 today's table; k4 K4's; g53kt3r is not this tool's.
pub fn mode() !bool {
    return switch (try glm.halves_kernel.fromEnv()) {
        .x1 => false,
        .k4 => true,
        .g53kt3r => error.HalvesKernelNotHere,
    };
}

pub const State = struct { shared: bool = false, suh: ?u64 = null, suh_buf: ?cuda.DeviceBuffer = null, ready: bool = false };

/// From the group's host bytes: python's share_suh test on the layer's gate and up suh (its first row uploaded, or a
/// --dry stand-in), and prompt_kernel_ready on the bound layer (its trellises' alignment at the group's address).
pub fn setup(t: *Tool, group: []const u8) !void {
    const st = &t.k4.?;
    const D: usize = t.e.gate.k;
    const n = @as(usize, t.e.count) * D;
    const g = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[t.e.gate.suh.off..][0 .. 2 * n]));
    const u = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, group[t.e.up.suh.off..][0 .. 2 * n]));
    const host = try t.gpa.alloc(u16, 2 * n);
    defer t.gpa.free(host);
    for (g, 0..) |v, i| host[i] = v;
    for (u, 0..) |v, i| host[n + i] = v;
    st.shared = glm.afd_expert.sharedSuh(host[0..n], host[n..], D);
    if (st.shared) {
        if (t.dev) |d| {
            st.suh_buf = try cuda.DeviceBuffer.alloc(d.d, 2 * D);
            try st.suh_buf.?.upload(0, std.mem.sliceAsBytes(host[0..D]));
            st.suh = st.suh_buf.?.ptr;
        } else {
            st.suh = t.next;
            t.next += std.mem.alignForward(u64, 2 * D, 256) + 256;
        }
    }
    st.ready = k4.ready(t.bound, chk.slots, glm.afd_expert.afdk4.aligned(t.e, t.base));
    t.out("INFO tf-exl3-check: K4 (TF_GLM_HALVES_KERNEL=k4): MiaAI-Lab's EXL3 prompt kernel against python 2.15's; the layer's gate and up suh {s} (rowrot's rows {s}); prompt_kernel_ready {}", .{ if (st.shared) "are one vector" else "differ", if (st.shared) "on" else "off: rot_in", st.ready });
}

pub fn cases(t: *Tool) !void {
    const st = t.k4.?;
    const want = try ck.table(t.gpa);
    defer t.gpa.free(want);
    const got = t.man.get("cases").?.array.items;
    var lst: std.Io.Writer.Allocating = .init(t.gpa);
    defer lst.deinit();
    try ck.head(&lst.writer, t.layer, t.rank, t.facts, .{ tool.trellisBytes(t.e.gate), tool.trellisBytes(t.e.up), tool.trellisBytes(t.e.down) }, st.shared);
    const lf = t.man.get("layer_facts").?.object;
    if (tool.int(lf, "shared_suh") != @intFromBool(st.shared)) t.fail("K4: python's share_suh gives {d}, the group's bytes {d}", .{ tool.int(lf, "shared_suh"), @intFromBool(st.shared) });
    if (!st.ready) t.fail("K4: layer {d} is outside the prompt kernel here (prompt_kernel_ready false)", .{t.layer});
    const nk = @typeInfo(ck.Kind).@"enum".field_names.len;
    var pass: [nk]u32 = @splat(0);
    var total: [nk]u32 = @splat(0);
    if (got.len != want.len) t.fail("cases: the manifest has {d}, K4's table {d} (not a TF_EXL3_CHECK_K4=1 manifest?)", .{ got.len, want.len });
    for (want, 0..) |c, i| {
        var idb: [96]u8 = undefined;
        const id = c.id(&idb);
        total[@backingInt(c.kind)] += 1;
        const s = ck.spec(c, t.facts);
        try chk.line(&lst.writer, id, c.kind, s);
        if (i >= got.len or !std.mem.eql(u8, tool.str(got[i].object, "id"), id)) {
            t.fail("{t} {s}: not the manifest's case {d}", .{ c.kind, id, i });
            continue;
        }
        if (!st.ready or !agree(t, got[i].object, c, s, id)) continue;
        if (try one(t, got[i].object, c, s, id)) pass[@backingInt(c.kind)] += 1;
    }
    if (t.a.listing) |path| try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = lst.written() });
    for (0..nk) |k| if (total[k] > 0) {
        const kind: ck.Kind = @fromBackingInt(@intCast(k));
        const ok = pass[k] == total[k];
        if (t.a.dry) {
            t.out("{s} dry {t}: {d} of {d} cases agree with the manifest (shapes, dtypes, scratch bytes, inputs' sha256, K4's switches); launches built", .{ if (ok) "PASS" else "FAIL", kind, pass[k], total[k] });
        } else t.out("{s} {t}: {d} of {d} cases byte-equal to python 2.15's K4 (y and xd)", .{ if (ok) "PASS" else "FAIL", kind, pass[k], total[k] });
    };
    t.out("RESULT {s}: {d} cases, {d} launches (K4)", .{ if (t.a.dry) "dry" else "check", want.len, t.launches });
}

/// The manifest's case against K4's table and the bindings: kind, rows, limit, the call, K4's switches and its items
/// bound (python's call.prompt), the arrays and the scratch bytes.
fn agree(t: *Tool, mc: Obj, c: ck.Case, s: chk.Spec, id: []const u8) bool {
    const call = mc.get("call").?.object;
    var why: ?[]const u8 = null;
    if (!std.mem.eql(u8, tool.str(mc, "kind"), @tagName(c.kind))) why = "kind";
    if (tool.int(call, "R") != c.rows or tool.int(call, "rows") != c.rows) why = "rows";
    if (tool.int(call, "limit_f32_bits") != @as(u32, @bitCast(s.limit))) why = "limit";
    if (!std.mem.eql(u8, tool.str(call, "act"), "bf16") or !std.mem.eql(u8, tool.str(call, "x_dtype"), "bf16") or tool.int(call, "weighted") != 0) why = "call";
    if (call.get("prompt")) |pv| {
        const p = pv.object;
        const shx = c.opts.xrow and t.k4.?.shared;
        if (tool.int(p, "pass") != c.opts.pass or tool.int(p, "order") != @intFromBool(c.opts.order) or tool.int(p, "rowrot") != @intFromBool(c.opts.xrow) or tool.int(p, "shx") != @intFromBool(shx)) why = "K4's switches";
        if (tool.int(p, "max_items") != k4.promptItems(c.rows, chk.slots, t.bound.count, c.opts.pass)) why = "K4's items bound";
    } else why = "no K4 call";
    const ins = mc.get("inputs").?;
    const outs = mc.get("outputs").?;
    if (ins.array.items.len != s.n_in or outs.array.items.len != s.n_out) why = "array count";
    for (s.inputs()) |a| why = why orelse if (tool.findArr(ins, a.name)) |m| tool.sameArr(m, a) else "inputs";
    for (s.outputs()) |a| why = why orelse if (tool.findArr(outs, a.name)) |m| tool.sameArr(m, a) else "outputs";
    const sc = mc.get("scratch").?.object;
    if (sc.count() != s.n_scratch) why = why orelse "scratch count";
    for (s.scratches()) |x| if (tool.int(sc, x.name) != x.bytes) {
        why = why orelse "scratch bytes";
    };
    if (why) |r| {
        t.fail("{t} {s}: the manifest's {s} differs from K4's table and the Zig bindings", .{ c.kind, id, r });
        return false;
    }
    return true;
}

/// One case: inputs uploaded and checked, outputs zeroed, members and the order list -1 (python's), the rest zero;
/// K4's launches built and, on a GPU, run on one stream after a context sync; y and xd against python 2.15's.
fn one(t: *Tool, mc: Obj, c: ck.Case, s: chk.Spec, id: []const u8) !bool {
    var b: tool.Bufs = .{};
    defer tool.release(t, &b);
    for (s.inputs()) |a| {
        const m = tool.findArr(mc.get("inputs").?, a.name).?;
        const host = try tool.refFile(t, tool.str(m, "file"), 1 << 31);
        if (host.len != a.bytes() or !std.mem.eql(u8, &tool.hex(host), tool.str(m, "sha256"))) {
            t.gpa.free(host);
            t.fail("{t} {s}: input {s} is not the bytes the manifest lists", .{ c.kind, id, a.name });
            return false;
        }
        try tool.place(t, &b, a.name, a.bytes(), .upload, host);
    }
    for (s.outputs()) |a| try tool.place(t, &b, a.name, a.bytes(), .zero, null);
    for (s.scratches()) |x| try tool.place(t, &b, x.name, x.bytes, if (std.mem.eql(u8, x.name, "members") or std.mem.eql(u8, x.name, "order")) .minus1 else .zero, null);
    var ls: x3.kernels.Launches = .{};
    build(t, &ls, &b, c, s) catch |e| {
        t.fail("{t} {s}: the bindings refused the case ({t})", .{ c.kind, id, e });
        return false;
    };
    t.launches += ls.len;
    const d = t.dev orelse return true;
    try d.ctx.synchronize();
    try d.set.run(d.stream, &ls);
    try d.stream.synchronize();
    var ok = true;
    for (s.outputs()) |a| ok = try tool.compare(t, mc, b.get(a.name).?, a, c.kind, id) and ok;
    if (t.a.bench > 0) {
        try bench(t, d, &ls, &b, c);
        try d.set.run(d.stream, &ls); // K4 once more: the same bytes after the bench
        try d.stream.synchronize();
        for (s.outputs()) |a| ok = try tool.compare(t, mc, b.get(a.name).?, a, c.kind, id) and ok;
    }
    return ok;
}

fn scratchOf(t: *Tool, b: *const tool.Bufs, c: ck.Case) !ex.Scratch {
    const tl = try t.facts.tiles();
    return .{ .xg = b.at("xg"), .xu = b.at("xu"), .xd = b.at("xd"), .z = b.at("z"), .y = b.at("y"), .ids = b.at("ids"), .count = b.at("count"), .members = b.at("members"), .rows = c.rows, .slots = chk.slots, .experts = t.bound.count, .cfg_gu = tl[0], .cfg_d = tl[1] };
}

fn shared(t: *Tool) ex.Shared {
    return if (t.dev) |d| d.sh else .{ .optin = 101376 };
}

fn build(t: *Tool, ls: *x3.kernels.Launches, b: *const tool.Bufs, c: ck.Case, s: chk.Spec) !void {
    const kb: k4.Bufs = .{ .items = b.at("items"), .counts = b.at("counts"), .order = b.at("order"), .cap = @intCast(s.scratchOf("order") / 4) };
    try k4.routed(ls, b.at("x"), t.bound.dims, true, b.at("pick"), t.bound, try scratchOf(t, b, c), c.rows, s.limit, true, shared(t), c.opts, kb, if (c.opts.xrow) t.k4.?.suh else null);
}

/// --bench N: K4's launches and today's routed launches (routedSched at the tool's 0128 settings, its z placed here)
/// alternating, N runs each between events; then K4's gate/up and down alone. Medians in us.
fn bench(t: *Tool, d: *tool.Dev, ls: *const x3.kernels.Launches, b: *tool.Bufs, c: ck.Case) !void {
    const n = t.a.bench;
    const tl = try t.facts.tiles();
    try tool.place(t, b, "z", ex.Scratch.bytes(c.rows, chk.slots, t.bound, tl[0], tl[1]).z, .zero, null);
    var today: x3.kernels.Launches = .{};
    try ex.routedSched(&today, b.at("x"), t.bound.dims, true, b.at("pick"), null, t.bound, try scratchOf(t, b, c), 0, c.rows, t.facts.limit, .bf16, true, shared(t), t.sched);
    const ev = try t.gpa.alloc(cuda.Event, 4 * n);
    defer t.gpa.free(ev);
    var made: usize = 0;
    defer for (ev[0..made]) |*e| e.deinit();
    while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(d.d, true);
    const us = try t.gpa.alloc(f64, 2 * n);
    defer t.gpa.free(us);
    for (0..n) |i| { // K4, then today's, run by run
        for ([_]*const x3.kernels.Launches{ ls, &today }, 0..) |arm, a| {
            try ev[4 * i + 2 * a].record(d.stream);
            try d.set.run(d.stream, arm);
            try ev[4 * i + 2 * a + 1].record(d.stream);
        }
    }
    try ev[4 * n - 1].synchronize();
    for (0..n) |i| for (0..2) |a| {
        us[a * n + i] = @as(f64, try cuda.Event.elapsedMs(ev[4 * i + 2 * a], ev[4 * i + 2 * a + 1])) * 1000;
    };
    var med: [2]f64 = undefined;
    for (0..2) |a| {
        const x = us[a * n ..][0..n];
        std.mem.sort(f64, x, {}, std.sort.asc(f64));
        med[a] = x[n / 2];
        t.out("BENCH {t} r{d} {s}: median {d:.1} us, p10 {d:.1}, p90 {d:.1} ({d} runs, alternating)", .{ c.kind, c.rows, if (a == 0) "k4-routed" else "today-routed", x[n / 2], x[n / 10], x[n - 1 - n / 10], n });
    }
    t.out("BENCH-AB {t} r{d} routed: K4 / today {d:.3} ({d:.1} / {d:.1} us)", .{ c.kind, c.rows, med[0] / med[1], med[0], med[1] });
    for ([_][]const u8{ "_ZN10tf_exl3_k420prompt_gateup", "_ZN10tf_exl3_k418prompt_down" }, [_][]const u8{ "k4-gate_up", "k4-down" }) |prefix, part| {
        var alone: x3.kernels.Launches = .{};
        for (ls.slice()) |l| if (std.mem.startsWith(u8, l.symbol, prefix)) alone.push(l);
        if (alone.len != 1) return error.NoK4Launch;
        for (0..n) |i| {
            try ev[2 * i].record(d.stream);
            try d.set.run(d.stream, &alone);
            try ev[2 * i + 1].record(d.stream);
        }
        try ev[2 * n - 1].synchronize();
        const x = us[0..n];
        for (x, 0..) |*v, i| v.* = @as(f64, try cuda.Event.elapsedMs(ev[2 * i], ev[2 * i + 1])) * 1000;
        std.mem.sort(f64, x, {}, std.sort.asc(f64));
        t.out("BENCH {t} r{d} {s}: median {d:.1} us, p10 {d:.1}, p90 {d:.1} ({d} runs)", .{ c.kind, c.rows, part, x[n / 2], x[n / 10], x[n - 1 - n / 10], n });
    }
}
