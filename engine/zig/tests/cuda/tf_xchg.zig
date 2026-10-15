//! tf-xchg (qualification_12f4270a): the per-MoE-layer exchange between one attention GPU and two expert halves over 0006's RoCE proxy, for the three-host lab (attention-host, expert1-host, expert0-host) or one host's CPU.
//! attn: a checked sweep over row counts with values new every forward, round trips (single and 42-layer forwards), or a failure test (--expect); expert K: the stand-in half, with --fault silent:N or exit:N.
//! Exit codes: 0 every check passed, 1 a check failed, 2 usage or setup, 3 an expert lost the attention.
const std = @import("std");
const cuda = @import("cuda");
const xchg = cuda.xchg;
const proto = xchg.proto;
const lab = @import("xchg/lab.zig");
const Cpu = @import("xchg/cpu.zig").Cpu;
const net = std.Io.net;
const Allocator = std.mem.Allocator;

const usage =
    \\usage: tf-xchg attn MASTER PORT [flags]     the attention rank (rank 0): listens on MASTER:PORT
    \\       tf-xchg expert K MASTER PORT [flags] expert half K (0 or 1): joins the attention
    \\       tf-xchg layout [flags]               python 2.15's wire for the layout flags
    \\flags: --cpu (no GPU: the kernels' CPU build)  --dev mlx5_0[,TWIN]  --gid 3  --max-rows 4096
    \\       --rows 1,2,4,...  --repeats 3  --check 1  --time ITERS,FORWARDS  --after 3
    \\       --expect silent:K|dead:K|gave-up:K|proxy:K|breach:K  --silent-s 10  --give-up-s 20  --no-watcher
    \\       --fault silent:N|exit:N (expert)  --d 4096 --s 9 --experts 288 --layers 42 --bf16-partials --prompt-from 64
    \\
;

const default_rows = [_]usize{ 1, 2, 3, 4, 7, 8, 16, 32, 64, 65, 128, 256, 512, 1000, 1024, 2048, 4096 };

pub const Flags = struct {
    cpu: bool = false,
    devs: []const u8 = "mlx5_0",
    gid: c_int = 3,
    max_rows: usize = 4096,
    rows: []const usize = &default_rows,
    repeats: usize = 3,
    check: bool = true,
    time: ?[2]usize = null,
    after: usize = 3,
    expect: ?Spec = null,
    fault: ?Spec = null,
    silent_s: f64 = 10,
    give_up_s: f64 = 20,
    watcher: bool = true,
    d: usize = 4096,
    s: usize = 9,
    experts: u32 = 288,
    layers: u32 = 42,
    bf16: bool = false,
    prompt_from: usize = 64,

    pub fn layout(f: Flags) proto.Layout {
        return .{ .d = f.d, .s = f.s, .bf16 = f.bf16 };
    }
};

/// `what:N`: a failure kind and an expert (--expect) or a request (--fault).
pub const Spec = struct { what: []const u8, n: u32 };

pub fn parseSpec(text: []const u8) !Spec {
    const i = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadSpec;
    return .{ .what = text[0..i], .n = try std.fmt.parseInt(u32, text[i + 1 ..], 10) };
}

pub fn parseRows(gpa: Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(usize, t, 10));
    if (list.items.len == 0) return error.BadRows;
    return list.toOwnedSlice(gpa);
}

pub fn parseFlags(gpa: Allocator, args: []const [:0]const u8) !Flags {
    var f: Flags = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--cpu")) f.cpu = true else if (std.mem.eql(u8, a, "--no-watcher")) f.watcher = false else if (std.mem.eql(u8, a, "--bf16-partials")) f.bf16 = true else {
            i += 1;
            if (i >= args.len) return error.MissingValue;
            const v = args[i];
            if (std.mem.eql(u8, a, "--dev")) f.devs = v else if (std.mem.eql(u8, a, "--gid")) f.gid = try std.fmt.parseInt(c_int, v, 10) else if (std.mem.eql(u8, a, "--max-rows")) f.max_rows = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--rows")) f.rows = try parseRows(gpa, v) else if (std.mem.eql(u8, a, "--repeats")) f.repeats = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--check")) f.check = !std.mem.eql(u8, v, "0") else if (std.mem.eql(u8, a, "--time")) {
                const c = std.mem.indexOfScalar(u8, v, ',') orelse return error.BadTime;
                f.time = .{ try std.fmt.parseInt(usize, v[0..c], 10), try std.fmt.parseInt(usize, v[c + 1 ..], 10) };
            } else if (std.mem.eql(u8, a, "--after")) f.after = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--expect")) f.expect = try parseSpec(v) else if (std.mem.eql(u8, a, "--fault")) f.fault = try parseSpec(v) else if (std.mem.eql(u8, a, "--silent-s")) f.silent_s = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, a, "--give-up-s")) f.give_up_s = try std.fmt.parseFloat(f64, v) else if (std.mem.eql(u8, a, "--d")) f.d = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--s")) f.s = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--experts")) f.experts = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, a, "--layers")) f.layers = try std.fmt.parseInt(u32, v, 10) else if (std.mem.eql(u8, a, "--prompt-from")) f.prompt_from = try std.fmt.parseInt(usize, v, 10) else return error.UnknownFlag;
        }
    }
    for (f.rows) |r| if (r < 1 or r > f.max_rows) {
        if (f.rows.ptr != &default_rows) return error.BadRows;
        // the default sweep, cut at --max-rows
        var n: usize = 0;
        while (n < default_rows.len and default_rows[n] <= f.max_rows) n += 1;
        f.rows = default_rows[0..n];
        break;
    };
    if (f.d % 128 != 0 or f.s < 2 or f.experts < f.s or f.experts % 2 != 0) return error.BadLayout;
    return f;
}

fn bad() u8 {
    std.debug.print("{s}", .{usage});
    return 2;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return bad();
    const cmd: []const u8 = args[1];
    var rest: []const [:0]const u8 = args[2..];
    var role: xchg.Role = .attn;
    if (std.mem.eql(u8, cmd, "layout")) {
        const f = parseFlags(init.arena.allocator(), rest) catch return bad();
        return layoutCmd(f);
    }
    if (std.mem.eql(u8, cmd, "expert")) {
        if (rest.len < 1) return bad();
        role = .{ .expert = std.fmt.parseInt(u32, rest[0], 10) catch return bad() };
        if (role.expert > 1) return bad();
        rest = rest[1..];
    } else if (!std.mem.eql(u8, cmd, "attn")) return bad();
    if (rest.len < 2) return bad();
    const port = std.fmt.parseInt(u16, rest[1], 10) catch return bad();
    const master = net.IpAddress.parse(rest[0], port) catch {
        std.debug.print("MASTER is an IPv4 or IPv6 address, not {s}\n", .{rest[0]});
        return 2;
    };
    const f = parseFlags(init.arena.allocator(), rest[2..]) catch |e| {
        std.debug.print("bad flags: {t}\n", .{e});
        return bad();
    };
    return run(gpa, init.io, role, master, f) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ if (role == .attn) "attn" else "expert", e });
        return 2;
    };
}

fn layoutCmd(f: Flags) u8 {
    const l = f.layout();
    const pr = l.perRow();
    std.debug.print("INFO python 2.15 wire, d {d} s {d}: request {d} B a row (+256 B header, sections 256-B aligned), reply {d} B a row ({d} in a prompt window{s}); through the attention's port {d} B a row a layer\n", .{ l.d, l.s, pr.req, pr.rep, pr.rep_prompt, if (l.bf16) ", --bf16-partials" else "", 2 * pr.req + 2 * pr.rep });
    for (f.rows) |r| {
        const o = l.offsets(r);
        std.debug.print("rows {d}: normed {d} pick {d} wts {d} end {d} wire {d}; reply {d}; slot {d}\n", .{ r, o.normed, o.pick, o.wts, o.end, proto.wireBytes(o.end), l.rep(r, false), l.slotBytes(r) });
    }
    return 0;
}

fn run(gpa: Allocator, io: std.Io, role: xchg.Role, master: net.IpAddress, f: Flags) !u8 {
    // qualification_17c70e37: TF_AFD_STRIPES as the AFD roles read it (all three tf-xchg processes must have the same: SettingsDiffer)
    const st = try proto.stripesFromEnv("tf-xchg");
    const settings: proto.Settings = .{ .layout = f.layout(), .max_rows = f.max_rows, .ports = std.mem.count(u8, f.devs, ",") + 1, .stripes = st.k, .stripe_min = st.min };
    const cfg: xchg.Config = .{ .settings = settings, .rules = .{ .silent_ns = @intFromFloat(f.silent_s * 1e9) }, .give_up_ns = @intFromFloat(f.give_up_s * 1e9), .run_watcher = f.watcher };
    const total: usize = @intCast((proto.Region{ .slot = settings.slotBytes() }).total());
    var devs: [proto.PORTS_MAX][]const u8 = undefined;
    var it = std.mem.tokenizeScalar(u8, f.devs, ',');
    var nd: usize = 0;
    while (it.next()) |d| : (nd += 1) {
        if (nd >= proto.PORTS_MAX) return error.TooManyPorts;
        devs[nd] = d;
    }
    var roce: xchg.Roce = .{ .ports = devs[0..nd], .gid = f.gid };
    var text: [256]u8 = undefined;
    std.debug.print("INFO {s}: {s}; {s}; region {d} MiB, dev {s}, gid {d}\n", .{ if (role == .attn) "attn" else "expert", try settings.text(&text), if (f.cpu) "CPU kernels" else "GPU", total >> 20, f.devs, f.gid });
    if (f.cpu) {
        const region = try gpa.alignedAlloc(u8, .fromByteUnits(4096), total);
        defer gpa.free(region);
        @memset(region, 0);
        const status = try gpa.alloc(u32, proto.STATUS_WORDS);
        defer gpa.free(status);
        @memset(status, 0);
        var c: Cpu = .{ .threads = 16 };
        return roleMain(gpa, io, role, master, f, cfg, .{ .region = region, .region_dev = @intFromPtr(region.ptr), .status = status, .status_dev = @intFromPtr(status.ptr) }, .{ .cpu = &c }, roce.opener());
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name: [256]u8 = undefined;
    std.debug.print("INFO GPU {s} sm_{d}, driver {d}\n", .{ try ctx.name(&name), try ctx.capability(), try driver.version() });
    var g: lab.GpuEngine = .{ .d = &driver, .g = try xchg.Gpu.init(&driver), .s = try cuda.Stream.init(&driver, true), .e0 = try cuda.Event.init(&driver, true), .e1 = try cuda.Event.init(&driver, true) };
    defer {
        g.e1.deinit();
        g.e0.deinit();
        g.s.deinit();
        g.g.deinit();
    }
    var region = try cuda.HostBuffer.allocMapped(&driver, total);
    defer region.free();
    @memset(region.bytes, 0);
    var status = try cuda.HostBuffer.allocMapped(&driver, 4096);
    defer status.free();
    @memset(status.bytes, 0);
    const mem: xchg.Memory = .{ .region = region.bytes, .region_dev = try region.device(), .status = status.slice(u32)[0..proto.STATUS_WORDS], .status_dev = try status.device() };
    return roleMain(gpa, io, role, master, f, cfg, mem, .{ .gpu = &g }, roce.opener());
}

fn roleMain(gpa: Allocator, io: std.Io, role: xchg.Role, master: net.IpAddress, f: Flags, cfg: xchg.Config, mem: xchg.Memory, e: lab.Engine, opener: xchg.Opener) !u8 {
    var n = try xchg.Node.init(gpa, io, role, cfg, mem);
    defer n.deinit();
    if (role == .expert) {
        cuda.watch.installStop();
        try n.join(master, opener);
        std.debug.print("INFO expert {d}: joined, pair up\n", .{role.expert});
        const o: lab.ExpertOpts = .{ .layout = f.layout(), .experts = f.experts, .max_rows = f.max_rows, .check = f.check, .silent_at = if (f.fault != null and std.mem.eql(u8, f.fault.?.what, "silent")) f.fault.?.n else null, .exit_at = if (f.fault != null and std.mem.eql(u8, f.fault.?.what, "exit")) f.fault.?.n else null };
        const end = try lab.expertLoop(gpa, e, &n, o);
        std.debug.print("INFO expert {d}: {d} requests, {d} rows, {d} wrong values in requests ({t}{s}{s})\n", .{ role.expert, end.requests, end.rows, end.wrong, end.how, if (end.why.len > 0) ": " else "", end.why });
        switch (end.how) {
            .die => {
                std.debug.print("INFO expert {d}: exiting at request {d} (--fault exit)\n", .{ role.expert, o.exit_at.? });
                std.c._exit(cuda.watch.lost_exit_code);
            },
            .lost => {
                std.debug.print("LOST expert {d}: {s}\n", .{ role.expert, end.why });
                return cuda.watch.lost_exit_code;
            },
            .stopped => {
                std.debug.print("{s} expert {d}: requests exact\n", .{ if (end.wrong == 0) "PASS" else "FAIL", role.expert });
                return if (end.wrong == 0) 0 else 1;
            },
        }
    }
    const server = try cuda.link.Server.open(master);
    defer server.close();
    try n.attend(server, opener);
    defer n.stopExperts();
    std.debug.print("INFO attn: both experts up\n", .{});
    var bufs = try lab.AttnBufs.init(e, gpa, f.layout(), f.max_rows);
    defer bufs.deinit(e, gpa);
    const o: lab.AttnOpts = .{ .layout = f.layout(), .experts = f.experts, .layers = f.layers, .rows = f.rows, .repeats = f.repeats, .prompt_from = f.prompt_from };
    var fseq: i32 = 0;
    if (f.expect) |want| return expectFailure(io, e, &n, &bufs, o, want, f.after, &fseq);
    var ok = true;
    if (f.check) {
        const rep = try lab.sweep(e, &n, &bufs, o, &fseq);
        if (rep.failed) |x| {
            std.debug.print("FAIL attn: the exchange failed: expert {d} {t} at request {d}\n", .{ x.peer, x.why, x.seq });
            return 1;
        }
        ok = rep.wrong == 0;
        std.debug.print("{s} attn: {d} exchanges over {d} row counts, {d} wrong values\n", .{ if (ok) "PASS" else "FAIL", rep.exchanges, f.rows.len, rep.wrong });
    }
    if (f.time) |t| for (f.rows) |rows| {
        const tm = try lab.time(gpa, io, e, &n, &bufs, o, rows, t[0], t[1], &fseq);
        const l = f.layout();
        std.debug.print("TIME rows {d}: single p50 {d:.1} us p99 {d:.1} us; per layer p50 {d:.1} us (min {d:.1}) over {d}-layer forwards; request {d} B, replies 2 x {d} B\n", .{ rows, tm.single_p50_us, tm.single_p99_us, tm.layer_p50_us, tm.layer_min_us, f.layers, proto.wireBytes(l.req(rows)), l.rep(rows, rows >= f.prompt_from) });
    };
    return if (ok) 0 else 1;
}

/// Exchanges of one row until the exchange fails; it must fail as `want` says, and the `after` exchanges that follow must each return at once.
fn expectFailure(io: std.Io, e: lab.Engine, n: *xchg.Node, b: *lab.AttnBufs, o: lab.AttnOpts, want: Spec, after: usize, fseq: *i32) !u8 {
    var done: u64 = 0;
    var fail_ns: u64 = 0;
    while (n.failure() == null and done < 10_000_000) : (done += 1) {
        fseq.* += 1;
        const t0 = n.nowNs();
        const wrong = try lab.exchange(e, n, b, o, fseq.*, 1, done % 1000 == 0);
        if (wrong != 0) std.debug.print("FAIL attn: {d} wrong values at request {d}\n", .{ wrong, fseq.* });
        fail_ns = n.nowNs() - t0;
    }
    const x = n.failure() orelse {
        std.debug.print("FAIL attn: no failure in {d} exchanges\n", .{done});
        return 1;
    };
    const rung = n.doorbell();
    const fast = try lab.afterFailure(io, e, n, b, o, after, 100 * std.time.ns_per_ms, fseq);
    const kept = n.doorbell() == rung; // nothing was sent after the failure
    var why_buf: [32]u8 = undefined;
    const why = try std.fmt.bufPrint(&why_buf, "{t}", .{x.why});
    const match = std.mem.eql(u8, why, if (std.mem.eql(u8, want.what, "gave-up")) "gave_up" else want.what) and x.peer == want.n;
    const ok = match and fast.fast == after and kept;
    std.debug.print("{s} attn: expert {d} {s} at request {d}, {d:.3} s after its post (gave up {d}, breach {d}); {d} of {d} later requests failed at once, {s}; the process goes on (wanted {s}:{d})\n", .{ if (ok) "PASS" else "FAIL", x.peer, why, x.seq, @as(f64, @floatFromInt(fail_ns)) / 1e9, x.gave_up, x.breach, fast.fast, after, if (kept) "none sent" else "SOME SENT", want.what, want.n });
    return if (ok) 0 else 1;
}

test "flags: rows, specs, and the layout and size checks" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const f = try parseFlags(arena.allocator(), &.{ "--rows", "1,7,4096", "--expect", "silent:1", "--time", "200,20", "--bf16-partials", "--no-watcher" });
    try std.testing.expectEqualSlices(usize, &.{ 1, 7, 4096 }, f.rows);
    try std.testing.expectEqualStrings("silent", f.expect.?.what);
    try std.testing.expectEqual(@as(u32, 1), f.expect.?.n);
    try std.testing.expectEqual([2]usize{ 200, 20 }, f.time.?);
    try std.testing.expect(f.layout().bf16 and !f.watcher);
    try std.testing.expectError(error.BadRows, parseFlags(arena.allocator(), &.{ "--rows", "8192" }));
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 4, 7, 8, 16, 32, 64 }, (try parseFlags(arena.allocator(), &.{ "--max-rows", "64" })).rows);
    try std.testing.expectError(error.BadLayout, parseFlags(arena.allocator(), &.{ "--d", "100" }));
    try std.testing.expectError(error.BadSpec, parseSpec("silent"));
    try std.testing.expectError(error.UnknownFlag, parseFlags(arena.allocator(), &.{ "--nope", "1" }));
}

test "the tool's GPU, link and failure paths compile on the host" {
    _ = &run;
    _ = &roleMain;
    _ = &expectFailure;
    _ = &main;
}
