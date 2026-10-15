//! tf-glm-afd expert (qualification_cec3c04c; qualification_955b7191's tool hands `expert ...` to `command`): the AFD expert node, python 2.15's
//! experts_node.serve.
//! - serve: half H of the routed experts read from the checkpoint (EXL3 078455ff, the librarian's path, no local copy)
//!   onto the GPU group by group, python ExpertBuffers, then the exchange joined as expert H (native RoCE exchange: 0006's RoCE
//!   proxy) and the per-forward loop until the attention stops it. --layers N serves a cut model's MoE layers (those
//!   below N, as the attention's --layers N); --fault silent:N|exit:N stops answering, or dies, at request N (the lab).
//! - --selftest SEED: requests generated from the seed and computed locally (no network), each reply's sha256 (LZ2a;
//!   qualification_955b7191's twin.py follows `Gen`'s rule).
//! - --check-weights FILE: every buffer of the half by python's dataclass paths with its sha256; --ref compares them
//!   with qualification_2b1188e0's rank-H weights.json (LZ2a).
//! Each run ends with tf-glm-afd's line, `AFD OK ...` or `AFD FAILED <why> peer <k> seq <n>`, and its exit code: 0
//! stopped by the attention, or a selftest or check done; 1 usage, an error of its own, or a check that differs; 3 the
//! attention lost (its link's EOF, keepalive or proxy loss), a protocol breach, a take that gave up, or a --fault exit.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const ae = glm.afd_expert;
/// qualification_e5a7fc3b: --check-weights FILE --dry
const dry = @import("afd_expert_dry.zig");
/// qualification_21e21bc0: check-weights' reference and tally (qualification_2b1188e0's weights.json), moved there unchanged
const afd_ref = @import("afd_expert_ref.zig");
pub const Ref = afd_ref.Ref;
pub const Tally = afd_ref.Tally;
/// qualification_f504f9df: TF_GLM_HALVES_KERNEL on a half (read in `command`, carried in the settings)
const hsw = @import("afd_expert_halves.zig");
/// qualification_1b9bf461: TF_GLM_HALVES_KERNEL=g53kt3r's image and state on a half (python's g53_experts.Node)
const g53x = @import("afd_expert_g53.zig");
const k4g = @import("afd_expert_k4.zig"); // qualification_e4ad04cd
/// rank_fingerprint: each AFD role's identity line (TF_GLM_FINGERPRINT=1)
const afd_fp = @import("afd_fp.zig");
const xchg = cuda.xchg;
const proto = xchg.proto;
const net = std.Io.net;
const Allocator = std.mem.Allocator;

pub const usage =
    \\usage: tf-glm-afd expert --half H --model DIR --triton DIR --master IP:PORT [flags]   serve half H (0 or 1)
    \\       tf-glm-afd expert --half H --model DIR --triton DIR --selftest SEED              no network: reply digests
    \\       tf-glm-afd expert --half H --model DIR --check-weights FILE [--ref weights.json]  the half's digests
    \\       tf-glm-afd expert --half H --model DIR --check-weights FILE --dry   a CPU: the half's plan, bytes, no reads
    \\flags: --layers N (a cut model: its MoE layers below N)  --dev mlx5_0[,TWIN]  --gid 3  --give-up-s 20  --rows 8
    \\       --prefill-rows 2048  --device 0  --fault silent:N|exit:N (serve: at request N stop answering, or exit 3)
    \\
;

pub const Args = struct {
    half: u32 = 2, // none until --half
    model: []const u8 = "",
    triton: ?[]const u8 = null,
    master: ?[]const u8 = null,
    selftest: ?u64 = null,
    check_weights: ?[]const u8 = null,
    ref: ?[]const u8 = null,
    layers: ?u32 = null,
    fault: ?ae.Fault = null,
    devs: []const u8 = "mlx5_0",
    gid: c_int = 3,
    give_up_s: f64 = 20,
    rows: usize = ae.decode_rows,
    prefill_rows: usize = ae.prefill_rows,
    device: c_int = 0,
    /// --check-weights on a CPU (qualification_e5a7fc3b): the half's plan at stand-ins, nothing read (afd_expert_dry.zig)
    dry: bool = false,
    /// Mia's 0128 in the half's EXL3 experts (qualification_96d80965): TF_GLM_EXL3_L2D / TF_GLM_EXL3_PDL, set by `command` (not a flag)
    sched: glm.exl3.experts.Sched = .{},
    /// X1, python 2.15's prompt switches (qualification_ae31a4aa): TF_GLM_EXL3_X1 (and its four), set by `command` (not a flag)
    x1: glm.exl3.experts.Prompt = .{},
    /// qualification_f504f9df: TF_GLM_HALVES_KERNEL, the prompt windows' kernel (the settings carry it), set by `command` (not a flag)
    halves: glm.halves_kernel.Kernel = .x1,
    /// qualification_e4ad04cd: K4's switches (python 2.15's prompt_switches), read by `command` with TF_GLM_HALVES_KERNEL=k4
    k4: glm.exl3.k4.Opts = .{},
    /// qualification_1fee6d78: TF_AFD_SETUP_MS, the handshake's window in ms (null: Config's 120,000), set by `command` (not a flag)
    setup_ms: ?u32 = null,
    /// qualification_c2d4ee2e: the 3-node prefill pairs (TF_AFD_PAIRS=1): the settings carry pairs=1, the loop serves mode-7 pairs
    pairs: bool = false,
    /// qualification_c533375f: TF_AFD_INFLIGHT (2 with the pairs): the settings carry inflight=2; the loop is the pairs' as it is
    inflight: u8 = 1,
    /// qualification_17c70e37: TF_AFD_STRIPES (K or K/MIN, or off): the settings carry stripes=K, each large message goes and is taken in K stripes
    stripes: proto.Stripes = .{},
    /// qualification_9e49592e: TF_AFD_LANES=N (3 or 4, with the pairs): the settings carry inflight=N lanes=N (a region of N slots a
    /// source); the loop serves mode-9 lane groups too
    lanes: u8 = 0,
    /// rank_fingerprint: TF_GLM_FINGERPRINT=1: the settings carry fp=1 and the identities are agreed at the hello, set by `command`
    fp: bool = false,
};

pub fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{};
    var i: usize = 0;
    while (i < args.len) {
        if (eql(args[i], "--dry")) { // qualification_e5a7fc3b: check-weights on a CPU (afd_expert_dry.zig), the one flag without a value
            a.dry = true;
            i += 1;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        const k = args[i];
        const v = args[i + 1];
        i += 2;
        const int = std.fmt.parseInt;
        if (eql(k, "--half")) a.half = try int(u32, v, 10) else if (eql(k, "--model")) a.model = v else if (eql(k, "--triton")) a.triton = v else if (eql(k, "--master")) a.master = v else if (eql(k, "--selftest")) a.selftest = try int(u64, v, 10) else if (eql(k, "--check-weights")) a.check_weights = v else if (eql(k, "--ref")) a.ref = v else if (eql(k, "--layers")) a.layers = try int(u32, v, 10) else if (eql(k, "--fault")) a.fault = try fault(v) else if (eql(k, "--dev")) a.devs = v else if (eql(k, "--gid")) a.gid = try int(c_int, v, 10) else if (eql(k, "--give-up-s")) a.give_up_s = try std.fmt.parseFloat(f64, v) else if (eql(k, "--rows")) a.rows = try int(usize, v, 10) else if (eql(k, "--prefill-rows")) a.prefill_rows = try int(usize, v, 10) else if (eql(k, "--device")) a.device = try int(c_int, v, 10) else return error.UnknownFlag;
    }
    if (a.half >= ae.halves or a.model.len == 0) return error.MissingFlag;
    if (a.rows < 1 or a.rows > a.prefill_rows or a.give_up_s <= 0 or (a.layers != null and a.layers.? == 0)) return error.BadValue;
    const modes = @as(u8, @intFromBool(a.master != null)) + @intFromBool(a.selftest != null) + @intFromBool(a.check_weights != null);
    if (modes != 1) return error.OneMode; // --master, --selftest or --check-weights
    if (a.check_weights == null and a.triton == null) return error.MissingFlag;
    if (a.fault != null and a.master == null) return error.FaultNeedsServe;
    if (a.ref != null and a.check_weights == null) return error.RefNeedsCheck;
    if (a.dry and (a.check_weights == null or a.ref != null)) return error.DryIsCheckWeights; // no bytes read: no ref
    if (a.master) |m| _ = try master(m);
    return a;
}

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// IP:PORT (an IPv6 address in brackets).
pub fn master(text: []const u8) !net.IpAddress {
    const c = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.BadMaster;
    const host = std.mem.trim(u8, text[0..c], "[]");
    return net.IpAddress.parse(host, try std.fmt.parseInt(u16, text[c + 1 ..], 10));
}

/// silent:N or exit:N, N from 1 (tf-xchg's --fault).
pub fn fault(text: []const u8) !ae.Fault {
    const c = std.mem.indexOfScalar(u8, text, ':') orelse return error.BadFault;
    const what = std.meta.stringToEnum(@FieldType(ae.Fault, "what"), text[0..c]) orelse return error.BadFault;
    const at = std.fmt.parseInt(u32, text[c + 1 ..], 10) catch return error.BadFault;
    if (at == 0) return error.BadFault;
    return .{ .what = what, .at = at };
}

/// The `expert` command: its flags, then serve, the selftest or the check; tf-glm-afd's last line; the exit code.
pub fn command(gpa: Allocator, io: std.Io, args: []const [:0]const u8) u8 {
    var a = parse(args) catch |e| {
        std.debug.print("{s}", .{usage});
        return done(1, "AFD FAILED {t} peer - seq 0", .{e});
    };
    a.sched = sched(a.half) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // before anything loads
    a.x1 = x1(a.half) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e});
    a.halves = hsw.halvesKernel(a.half) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_f504f9df
    var ob: [512]u8 = undefined; // qualification_c863b626: SEQ, PGROUP and MMA beside a halves kernel: one INFO line, not a refusal
    if (hsw.overX1(&ob, a.half, a.halves, a.x1)) |ln| std.debug.print("{s}\n", .{ln});
    if (a.halves == .k4) a.k4 = k4g.opts(a.half) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_e4ad04cd
    var wb: [24]u8 = undefined; // qualification_1fee6d78: TF_AFD_SETUP_MS, the handshake's window, read before anything loads (BadAfdSetupMs)
    a.setup_ms = xchg.setup.fromEnv(std.fmt.bufPrint(&wb, "afd expert {d}", .{a.half}) catch "afd expert") catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e});
    a.pairs = ae.pairs.fromEnv() catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_c2d4ee2e: TF_AFD_PAIRS
    a.inflight = ae.pairs.inflightFromEnv(a.pairs) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_c533375f
    a.stripes = proto.stripesFromEnv("afd expert") catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_17c70e37
    a.lanes = ae.pairs.lanesFromEnv(a.pairs) catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // qualification_9e49592e
    if (a.lanes > 0) a.inflight = a.lanes; // N lanes keep N requests in flight a link
    a.fp = cuda.fingerprint.fromEnv("afd expert") catch |e| return done(1, "AFD FAILED {t} peer - seq 0", .{e}); // rank_fingerprint
    const run = if (a.check_weights != null) (if (a.dry) dry.run(gpa, a) else checkWeights(gpa, io, a)) else if (a.selftest != null) selftest(gpa, io, a) else serve(gpa, io, a);
    return run catch |e| {
        std.debug.print("INFO afd expert {d}: {t}\n", .{ a.half, e });
        return done(1, "AFD FAILED {t} peer - seq 0", .{e});
    };
}

/// Mia's 0128 in the half's EXL3 experts (TF_GLM_EXL3_L2D 0..64, TF_GLM_EXL3_PDL 0|1), read once, as tf-glm-forward
/// reads them (qualification_6052b66d): a bad value refuses the run; on, one INFO line, and the exl3_l2d images load beside today's six.
pub fn sched(half: u32) !glm.exl3.experts.Sched {
    const s = glm.exl3.experts.Sched.fromEnv() catch |e| {
        std.debug.print("INFO afd expert {d}: TF_GLM_EXL3_L2D takes 0..64 and TF_GLM_EXL3_PDL 0 or 1\n", .{half});
        return e;
    };
    if (s.on()) std.debug.print("INFO afd expert {d}: EXL3 experts: Mia's 0128 kernels (exl3_l2d), L2 prefetch {d} steps ahead, programmatic launch {s}\n", .{ half, s.l2d, if (s.pdl) "on" else "off" });
    return s;
}

/// X1, python 2.15's prompt switches on the half's windows of 64+ rows (TF_GLM_EXL3_X1 0|1; with it 1, python's
/// TF_GLM_EXL3_NFIRST, _PF, _ORDER, _ROWROT 0|1, each on unless 0), read once beside 0128's: a bad value refuses the
/// run; on, one INFO line, and the exl3_x1 images, the order lists and the shared suh rows load beside today's.
pub fn x1(half: u32) !glm.exl3.experts.Prompt {
    const p = glm.exl3.experts.Prompt.fromEnv() catch |e| {
        std.debug.print("INFO afd expert {d}: TF_GLM_EXL3_X1 takes 0 or 1, and with it 1 TF_GLM_EXL3_NFIRST, _PF, _ORDER and _ROWROT 0 or 1; TF_GLM_EXL3_PGROUP 0 or 1; TF_GLM_EXL3_SEQ 0 or 1, and 1 only with X1 in effect; TF_GLM_EXL3_MMA 0 or 1, and 1 only with X1 in effect and PGROUP\n", .{half});
        return e;
    };
    if (p.on()) std.debug.print("INFO afd expert {d}: EXL3 experts: X1, python 2.15's prompt switches (exl3_x1) on windows of {d}+ rows: nfirst {d}, prefetch {d}, order {d}, rowrot {d}\n", .{ half, glm.exl3.experts.prompt_rows, @intFromBool(p.nfirst), @intFromBool(p.pa), @intFromBool(p.order), @intFromBool(p.xrow) });
    if (p.pgroup) {
        var b: [256]u8 = undefined;
        std.debug.print("{s}\n", .{ae.pgroup.line(&b, half)});
    }
    if (p.seq) std.debug.print("INFO afd expert {d}: EXL3 experts: SEQ (Mia's TF_GLM_EXL3_SEQ, exl3_x1s): gate/up's K splits summed in one program on X1's windows, its epilogue at SK 1\n", .{half});
    if (p.mma) {
        var b: [320]u8 = undefined;
        std.debug.print("{s}\n", .{ae.mma.line(&b, half, p)});
    }
    return p;
}

/// The run's last line (tf-glm-afd's: `AFD OK ...` or `AFD FAILED <why> peer <k> seq <n>`) and its exit code.
fn done(code: u8, comptime fmt: []const u8, args: anytype) u8 {
    std.debug.print(fmt ++ "\n", args);
    return code;
}

/// The expert's GPU: the half's groups loaded and their tables written, python's buffers and the landing in one block,
/// qualification_qualification_44ba554e-exl3's images and the Triton set (none for the check), on one stream.
const Gpu = struct {
    gpa: Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    c: glm.Config,
    ix: glm.index.Index,
    x: ae.Half,
    groups: []cuda.DeviceBuffer,
    base: []u64,
    block: ?cuda.DeviceBuffer = null,
    b: ae.Bufs = undefined,
    landing: ae.Landing = undefined,
    exl3: ?glm.exl3.kernels.Set = null,
    tset: ?cuda.aot.Set = null,
    sh: glm.exl3.experts.Shared = .{ .optin = 0 },
    sched: glm.exl3.experts.Sched = .{},
    x1: glm.exl3.experts.Prompt = .{},
    x1buf: ?cuda.DeviceBuffer = null,
    x1b: ae.X1Bufs = undefined,
    xrow: []bool = &.{},
    zero: ?ae.Memset2D = null,
    pgbuf: ?cuda.DeviceBuffer = null, // qualification_3e1f1105: TF_GLM_EXL3_PGROUP's counts block (cuda_afd_pgroup.zig)
    mmabuf: ?cuda.DeviceBuffer = null, // qualification_8724e161: TF_GLM_EXL3_MMA's work list block (cuda_afd_mma.zig)
    g53: ?*g53x.G53 = null, // qualification_1b9bf461: only with TF_GLM_HALVES_KERNEL=g53kt3r
    k4: k4g.Gpu = .{}, // qualification_e4ad04cd: K4's image, block and readiness (TF_GLM_HALVES_KERNEL=k4)

    fn open(gpa: Allocator, io: std.Io, a: Args, compute: bool) !*Gpu {
        const g = try gpa.create(Gpu);
        errdefer gpa.destroy(g);
        g.gpa = gpa;
        g.c = try config(gpa, a.model);
        if (a.layers) |n| g.c.run = @min(n, g.c.layers); // a cut model: python's GLM_LAYERS
        g.driver = try cuda.Driver.open();
        errdefer g.driver.close();
        g.ctx = try cuda.Context.init(&g.driver, a.device);
        errdefer g.ctx.deinit();
        var name: [256]u8 = undefined;
        std.debug.print("INFO afd expert {d}: GPU {s} sm_{d}\n", .{ a.half, try g.ctx.name(&name), try g.ctx.capability() });
        g.stream = try cuda.Stream.init(&g.driver, true);
        errdefer g.stream.deinit();
        g.ix = try glm.index.open(gpa, a.model);
        errdefer g.ix.deinit();
        g.x = try ae.halfWith(gpa, &g.c, &g.ix.names, a.half, try glm.afd_mtp.fromEnv()); // qualification_21e21bc0: TF_AFD_MTP=1 adds the MTP layer
        errdefer g.x.deinit();
        g.groups = try load(gpa, &g.driver, &g.ix, &g.x, g.stream);
        errdefer {
            for (g.groups) |*d| d.free();
            gpa.free(g.groups);
        }
        g.base = try gpa.alloc(u64, g.groups.len);
        errdefer gpa.free(g.base);
        for (g.groups, g.base) |d, *p| p.* = d.ptr;
        g.block = null;
        g.exl3 = null;
        g.tset = null;
        g.zero = null;
        g.x1 = .{};
        g.x1buf = null;
        g.xrow = &.{};
        g.pgbuf = null;
        g.mmabuf = null;
        g.g53 = null;
        g.k4 = .{};
        if (!compute) return g;
        const l = layout(&g.c);
        g.b = try ae.Bufs.init(l.d, l.s, g.c.moe_inter / 2, g.c.experts, a.rows, a.prefill_rows);
        _ = try ae.seqz.apply(&g.b, &g.x, g.base, a.x1, a.half); // qualification_b3fba1da: SEQ's one-plane z where every position takes it
        g.landing = ae.Landing.init(l, a.prefill_rows);
        g.block = try cuda.DeviceBuffer.alloc(&g.driver, try ae.place(&g.b, &g.landing, 0));
        errdefer g.block.?.free();
        _ = try ae.place(&g.b, &g.landing, g.block.?.ptr);
        g.exl3 = try glm.exl3.kernels.Set.load(&g.driver);
        errdefer g.exl3.?.unload();
        g.sched = a.sched;
        if (a.sched.on()) try g.exl3.?.loadL2d(&g.driver); // 0128's four images, only with a switch on
        errdefer g.closeX1();
        if (a.x1.on()) try g.openX1(a); // X1's two images and block, only with TF_GLM_EXL3_X1=1
        if (a.x1.pgroup) { // qualification_3e1f1105: the counts block, an allocation of its own, only with TF_GLM_EXL3_PGROUP=1
            g.pgbuf = try cuda.DeviceBuffer.alloc(&g.driver, ae.pgroup.bytes(g.c.experts));
            ae.pgroup.apply(&g.b, g.pgbuf.?.ptr);
        }
        if (a.x1.mma) { // qualification_8724e161: the work list block and exl3_mma, only with TF_GLM_EXL3_MMA=1 (refused without PGROUP)
            g.mmabuf = try cuda.DeviceBuffer.alloc(&g.driver, ae.mma.bytes(&g.b, g.c.experts));
            ae.mma.apply(&g.b, g.mmabuf.?.ptr);
            try g.exl3.?.loadMma(&g.driver);
        }
        errdefer if (g.g53) |s| s.close();
        if (a.halves == .g53kt3r) { // qualification_1b9bf461: g53_rank and its block, only with TF_GLM_HALVES_KERNEL=g53kt3r
            try g.stream.synchronize(); // the groups' uploads
            g.g53 = try g53x.G53.open(gpa, &g.driver, &g.ctx, &g.x, g.base, g.groups, &g.b, g.c.experts, a.prefill_rows, a.half);
        }
        g.tset = try cuda.aot.Set.load(gpa, io, &g.driver, g.ctx.device, a.triton.?);
        errdefer g.tset.?.deinit();
        g.sh = .{ .optin = @intCast(try g.ctx.attribute(.max_shared_memory_per_block_optin)) };
        g.zero = try ae.memset2D(&g.driver);
        if (a.halves == .k4) try g.openK4(a); // qualification_e4ad04cd: K4's image and block, X1's block for its rowrot
        std.debug.print("INFO afd expert {d}: buffers {d} MiB (decode {d} rows, prompt {d}), opt-in shared {d} B\n", .{ a.half, g.block.?.len >> 20, a.rows, a.prefill_rows, g.sh.optin });
        return g;
    }

    fn close(g: *Gpu) void {
        g.k4.close(g.gpa);
        g.closeX1();
        if (g.g53) |s| s.close();
        if (g.tset) |*s| s.deinit();
        if (g.exl3) |*s| s.unload();
        if (g.block) |*d| d.free();
        g.gpa.free(g.base);
        for (g.groups) |*d| d.free();
        g.gpa.free(g.groups);
        g.x.deinit();
        g.ix.deinit();
        g.stream.deinit();
        g.ctx.deinit();
        g.driver.close();
        g.gpa.destroy(g);
    }

    /// X1 (TF_GLM_EXL3_X1=1): its two images; its block (the order lists and a shared suh row a position, an allocation
    /// of its own, so the node's block and every address in it are today's); and each position's shared-suh test,
    /// python's (experts.py:93-102) on host copies of its gate and up suh, the row copied in where it passes.
    fn openX1(g: *Gpu, a: Args) !void {
        try g.exl3.?.loadX1(&g.driver);
        if (a.x1.seq) try g.exl3.?.loadX1s(&g.driver); // qualification_b3fba1da: SEQ's image, only with TF_GLM_EXL3_SEQ=1
        const P = g.x.layers.len;
        g.x1b = ae.X1Bufs.init(&g.b, g.c.experts, P);
        g.x1buf = try cuda.DeviceBuffer.alloc(&g.driver, try ae.placeX1(&g.x1b, 0));
        _ = try ae.placeX1(&g.x1b, g.x1buf.?.ptr);
        g.xrow = try g.gpa.alloc(bool, P);
        try g.stream.synchronize(); // the groups' uploads
        const D = g.b.normed.shape[1];
        const host = try g.gpa.alloc(u16, 2 * @as(usize, g.c.experts) * D);
        defer g.gpa.free(host);
        var shared: usize = 0;
        for (g.xrow, 0..) |*ok, pos| {
            const l = g.x.bind(pos, g.base[pos]);
            const n = @as(usize, l.count) * l.dims;
            if (l.dims != D or 2 * n > host.len) return error.BadX1Shape;
            const grp = g.groups[pos];
            try grp.download(l.suh_g - grp.ptr, std.mem.sliceAsBytes(host[0..n]));
            try grp.download(l.suh_u - grp.ptr, std.mem.sliceAsBytes(host[n .. 2 * n]));
            ok.* = ae.sharedSuh(host[0..n], host[n .. 2 * n], D);
            if (!ok.*) continue;
            try g.x1buf.?.copyFrom(g.x1b.suh(pos) - g.x1buf.?.ptr, l.suh_g, 2 * D, null);
            shared += 1;
        }
        std.debug.print("INFO afd expert {d}: X1: {d} of {d} positions share one gate/up suh (rowrot); order lists {d} and {d} slots; block {d} KiB\n", .{ a.half, shared, P, g.x1b.decode_order.shape[0], g.x1b.pref_order.shape[0], g.x1buf.?.len >> 10 });
        g.x1 = a.x1;
    }

    fn closeX1(g: *Gpu) void {
        if (g.pgbuf) |*d| d.free(); // qualification_3e1f1105's block, freed beside X1's
        g.pgbuf = null;
        if (g.mmabuf) |*d| d.free(); // qualification_8724e161's block
        g.mmabuf = null;
        if (g.x1buf) |*d| d.free();
        g.x1buf = null;
        if (g.xrow.len > 0) g.gpa.free(g.xrow);
        g.xrow = &.{};
        g.x1 = .{};
    }

    /// K4 (qualification_e4ad04cd, TF_GLM_HALVES_KERNEL=k4): X1's block first when X1 is off (its shared suh rows, K4's rowrot; X1's
    /// switches stay off), then K4's image, block and readiness (afd_expert_k4.zig).
    fn openK4(g: *Gpu, a: Args) !void {
        if (g.x1buf == null) try g.openX1(a);
        try g.k4.open(g.gpa, &g.driver, &g.exl3.?, &g.x, &g.b, g.base, g.c.experts, a.k4, &g.x1b, g.xrow, a.half);
    }

    /// The loop's device side (fn_xchg's `xg`, or none for the selftest).
    fn device(g: *Gpu, xg: ?*const xchg.Gpu) ae.Device {
        const l = layout(&g.c);
        return .{ .d = &g.driver, .stream = g.stream, .xg = xg, .set = &g.exl3.?, .tri = .{ .set = &g.tset.?, .s = g.stream }, .sh = g.sh, .sched = g.sched, .x1 = g.x1, .x1b = if (g.x1buf != null) &g.x1b else null, .xrow = g.xrow, .zero = g.zero.?, .x = &g.x, .base = g.base, .b = &g.b, .landing = g.landing.buf.ptr, .l = l, .limit = g.c.swiglu_limit, .top_k = g.c.topk, .g53_node = if (g.g53) |s| &s.node else null, .k4 = g.k4.device() };
    }

    /// A device address inside the buffers' block, as an offset for its uploads and downloads.
    fn off(g: *const Gpu, ptr: u64) usize {
        return @intCast(ptr - g.block.?.ptr);
    }
};

/// python's wire for GLM: bf16 rows of D, top_k + 1 slots (the shared expert's last), fp32 replies.
fn layout(c: *const glm.Config) proto.Layout {
    return .{ .d = c.hidden, .s = c.topk + 1 };
}

pub fn config(gpa: Allocator, dir: []const u8) !glm.Config {
    const p = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
    defer gpa.free(p);
    const text = try readFile(gpa, p);
    defer gpa.free(text);
    return glm.config.parse(gpa, text);
}

/// A whole file (config.json; the ref, qualification_2b1188e0's weights.json, is about 1 MB).
pub fn readFile(gpa: Allocator, path: []const u8) ![]u8 {
    const p = try std.fmt.allocPrintSentinel(gpa, "{s}", .{path}, 0);
    defer gpa.free(p);
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return error.NoFile;
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    while (true) {
        try out.ensureUnusedCapacity(gpa, 1 << 20);
        const room = out.unusedCapacitySlice();
        const n = std.c.read(fd, room.ptr, room.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        out.items.len += @intCast(n);
        if (out.items.len > 256 << 20) return error.FileTooBig;
    }
    return out.toOwnedSlice(gpa);
}

/// The reader's copies into the half's groups: plan group g lands in position at[g]'s buffer, on one stream.
const Sink = struct {
    groups: []cuda.DeviceBuffer,
    at: []const u32,
    stream: cuda.Stream,

    fn sink(s: *Sink) glm.read.Sink {
        return .{ .ctx = s, .put = put, .flush = flush };
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.groups[s.at[g]].uploadAsync(@intCast(dst), bytes, s.stream.handle);
    }

    fn flush(ctx: *anyopaque) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.stream.synchronize();
    }
};

/// python load_half: every group allocated (zeroed, as cuda_weights.load does), the half's reads through the loader,
/// then each layer's pointer and width tables at their place.
fn load(gpa: Allocator, d: *const cuda.Driver, ix: *const glm.index.Index, x: *const ae.Half, stream: cuda.Stream) ![]cuda.DeviceBuffer {
    const groups = try gpa.alloc(cuda.DeviceBuffer, x.groups.len);
    errdefer gpa.free(groups);
    var made: usize = 0;
    errdefer for (groups[0..made]) |*g| g.free();
    while (made < groups.len) : (made += 1) {
        groups[made] = try cuda.DeviceBuffer.alloc(d, @intCast(x.plan.groups.items[x.groups[made]]));
        try groups[made].fill8(0, null);
    }
    const at = try gpa.alloc(u32, x.plan.groups.items.len);
    defer gpa.free(at);
    @memset(at, std.math.maxInt(u32));
    for (x.groups, 0..) |g, pos| at[g] = @intCast(pos);
    const o: glm.read.Options = .{};
    var s = try glm.read.schedule(gpa, x.copies, o);
    defer s.deinit();
    var sink: Sink = .{ .groups = groups, .at = at, .stream = stream };
    const st = try glm.read.run(gpa, ix.shards, &s, sink.sink(), o);
    const tab = try gpa.alloc(u8, 36 * @as(usize, x.experts(0).count));
    defer gpa.free(tab);
    for (groups, 0..) |g, pos| {
        const t = x.tables(pos, g.ptr, tab);
        try g.upload(@intCast(t.off), t.bytes);
    }
    std.debug.print("INFO afd expert {d}: {d} MoE layers ({d}-{d}), {d} B kept ({d:.2} GiB) in {d} B of groups, {d} reads at {d:.2} GB/s\n", .{ x.h, x.layers.len, x.layers[0], x.layers[x.layers.len - 1], x.kept(), @as(f64, @floatFromInt(x.kept())) / (1 << 30), x.bytes(), st.reads, st.gbPerS() });
    return groups;
}

/// serve: the half loaded first (python's order: the attention waits on its accept), then joined as expert H, then the
/// per-forward loop until a stop frame, the attention's loss, a breach, a failed take or the --fault.
pub fn serve(gpa: Allocator, io: std.Io, a: Args) !u8 {
    const graphed = try ae.graphs.fromEnv(); // qualification_f16009a8: TF_AFD_GRAPHS, read once before anything loads (BadAfdGraphs)
    const to = try master(a.master.?);
    const g = try Gpu.open(gpa, io, a, true);
    defer g.close();
    const l = layout(&g.c);
    var devs: [proto.PORTS_MAX][]const u8 = undefined;
    var it = std.mem.tokenizeScalar(u8, a.devs, ',');
    var nd: usize = 0;
    while (it.next()) |d| : (nd += 1) {
        if (nd >= proto.PORTS_MAX) return error.TooManyPorts;
        devs[nd] = d;
    }
    const settings: proto.Settings = .{ .layout = l, .max_rows = a.prefill_rows, .ports = nd, .pairs = a.pairs, .inflight = a.inflight, .seq = a.x1.seq, .halves = a.halves.setting(), .stripes = a.stripes.k, .stripe_min = a.stripes.min, .lanes = a.lanes, .fp = a.fp }; // qualification_969df694: seq=1 with SEQ; qualification_17c70e37: stripes=K[/MIN] with TF_AFD_STRIPES; qualification_9e49592e: lanes=N; rank_fingerprint: fp=1
    const total: usize = @intCast(settings.region().total()); // qualification_9e49592e: N slots a source with TF_AFD_LANES=N
    var region = try cuda.HostBuffer.allocMapped(&g.driver, total);
    defer region.free();
    @memset(region.bytes, 0);
    var status = try cuda.HostBuffer.allocMapped(&g.driver, 4096);
    defer status.free();
    @memset(status.bytes, 0);
    const mem: xchg.Memory = .{ .region = region.bytes, .region_dev = try region.device(), .status = status.slice(u32)[0..proto.STATUS_WORDS], .status_dev = try status.device() };
    var xg = try xchg.Gpu.init(&g.driver);
    defer xg.deinit();
    var ib: [cuda.fingerprint.line_max]u8 = undefined; // rank_fingerprint: this half's identity line, from what it loaded (afd_fp.zig); unset, none
    const ident = if (a.fp) try afd_fp.expert(&ib, &g.exl3.?, g.g53 != null, a.triton.?, a.model, try g.ctx.capability(), a.half) else null;
    var n = try xchg.Node.init(gpa, io, .{ .expert = a.half }, xchg.setup.with(.{ .settings = settings, .give_up_ns = @intFromFloat(a.give_up_s * 1e9), .identity = ident }, a.setup_ms), mem); // qualification_1fee6d78: unset, Config's 120,000 ms
    defer n.deinit();
    var roce: xchg.Roce = .{ .ports = devs[0..nd], .gid = a.gid };
    var text: [256]u8 = undefined;
    std.debug.print("INFO afd expert {d}: {s}; {d} MoE layers a forward; joining the attention\n", .{ a.half, try settings.text(&text), g.x.layers.len });
    cuda.watch.installStop();
    try n.join(to, roce.opener());
    var dev = g.device(&xg);
    // qualification_f16009a8: TF_AFD_GRAPHS=1, a decode window's or an MTP slice's positions replayed as one CUDA graph by (rows, slice)
    var cu: ae.graphs.Cuda = .{ .stream = dev.stream };
    var gname: [32]u8 = undefined;
    var gr = ae.graphs.Graphs.init(gpa, cu.ops(), std.fmt.bufPrint(&gname, "expert {d}", .{a.half}) catch "expert");
    defer gr.deinit();
    if (graphed) {
        dev.graphs = &gr;
        std.debug.print("INFO afd expert {d}: TF_AFD_GRAPHS=1: decode windows and MTP slices of up to {d} rows replayed from CUDA graphs by (rows, slice); prompt chunks and TF_AFD_TIME's timed forwards eager\n", .{ a.half, ae.graphs.max_rows });
    }
    var st: ae.Stats = .{};
    const end = ae.forwards(&dev, &n, .{ .layers = g.x.layers.len, .max_rows = a.prefill_rows, .fault = a.fault, .pairs = a.pairs, .lanes = a.lanes }, &st);
    if (dev.clock) |c| { // qualification_e5a7fc3b: TF_AFD_TIME=1's last TIME line, over every forward
        c.report();
        c.destroy();
    }
    std.debug.print("INFO {s}\n", .{ae.describe(end, a.half, st, &text)});
    if (graphed) {
        var gb: [320]u8 = undefined;
        std.debug.print("INFO afd expert {d}: {s}\n", .{ a.half, gr.line(&gb) });
    }
    const at = st.layers + 1; // the request it was at
    return switch (end) {
        .stopped => done(0, "AFD OK expert {d}: {d} forwards, {d} layers, {d} rows", .{ a.half, st.forwards, st.layers, st.rows }),
        .failed => |x| done(3, "AFD FAILED {t} peer {d} seq {d}", .{ x.why, x.peer, x.seq }),
        .lost => done(3, "AFD FAILED {t} peer 0 seq {d}", .{ ae.lost(&n) orelse .dead, at }),
        .breach => done(3, "AFD FAILED breach peer 0 seq {d}", .{at}),
        .die => {
            _ = done(3, "AFD FAILED fault peer - seq {d}", .{at});
            std.c._exit(cuda.watch.lost_exit_code); // the lab's dead expert: nothing closed, no goodbye
        },
        .err => |x| done(1, "AFD FAILED {t} peer - seq {d}", .{ x, at }),
    };
}

/// The selftest's requests, from the seed alone (qualification_955b7191's python twin follows the same rule): SplitMix64's finalizer as
/// tf-xchg's lab uses it; per (layer, R) one key for the rows and one for the router logits.
pub const Gen = struct {
    pub fn mix(x: u64) u64 {
        var z = x +% 0x9E3779B97F4A7C15;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    /// `what` 1: the rows, 2: the logits.
    pub fn key(seed: u64, what: u64, layer: u64, rows: u64) u64 {
        return mix(mix(seed) ^ (what << 48 | layer << 32 | rows));
    }

    /// Row r, column c: a normal bf16 of either sign in [2^-7, 2^4).
    pub fn rowBits(k: u64, r: u64, c: u64) u16 {
        const h = mix(k ^ (r << 32 | c));
        return @intCast(((h >> 16) & 1) << 15 | (120 + h % 11) << 7 | ((h >> 8) & 0x7f));
    }

    /// Row r's logit of expert e: (h >> 40) / 2^21 - 4, exact in fp32, in [-4, 4).
    pub fn logit(k: u64, r: u64, e: u64) f32 {
        const h = mix(k ^ (r << 32 | e));
        return @as(f32, @floatFromInt(h >> 40)) / 2097152.0 - 4.0;
    }
};

/// The selftest's windows: decode 1-8, a prompt chunk below and at the widest.
pub const selftest_rows = [_]usize{ 1, 2, 4, 8, 64, 2048 };

/// --selftest SEED: for each R of selftest_rows and each MoE layer, the seeded rows and logits uploaded, glue.select's
/// _topk (zero bias, routed_scale, normalized) into the landing's pick and wts, the half's compute, and the sha256 of
/// routed[:R] (fp32 [R, D]); then one digest over them all. No network.
pub fn selftest(gpa: Allocator, io: std.Io, a: Args) !u8 {
    const g = try Gpu.open(gpa, io, a, true);
    defer g.close();
    const l = layout(&g.c);
    const E: usize = g.c.experts;
    const top = a.prefill_rows;
    var logits = try cuda.DeviceBuffer.alloc(&g.driver, top * E * 4);
    defer logits.free();
    var bias = try cuda.DeviceBuffer.alloc(&g.driver, E * 4);
    defer bias.free();
    try bias.fill8(0, null);
    const rows = try gpa.alloc(u16, top * l.d);
    defer gpa.free(rows);
    const lg = try gpa.alloc(f32, top * E);
    defer gpa.free(lg);
    const out = try gpa.alloc(u8, top * l.d * 4);
    defer gpa.free(out);
    var dev = g.device(null);
    var all = std.crypto.hash.sha2.Sha256.init(.{});
    var n: usize = 0;
    for (selftest_rows) |R| {
        if (R > top) continue;
        for (g.x.layers, 0..) |layer, pos| {
            const kr = Gen.key(a.selftest.?, 1, layer, R);
            const kl = Gen.key(a.selftest.?, 2, layer, R);
            for (0..R) |r| for (0..l.d) |c| {
                rows[r * l.d + c] = Gen.rowBits(kr, r, c);
            };
            for (0..R) |r| for (0..E) |e| {
                lg[r * E + e] = Gen.logit(kl, r, e);
            };
            const v = ae.landed(l, g.landing.buf.ptr, R);
            try g.block.?.upload(g.off(v.normed), std.mem.sliceAsBytes(rows[0 .. R * l.d]));
            try logits.upload(0, std.mem.sliceAsBytes(lg[0 .. R * E]));
            try dev.tri.select(logits.ptr, bias.ptr, v.pick, v.wts, R, E, g.c.topk, g.c.routed_scale, true);
            try dev.compute(pos, .{ .fseq = 0, .rows = R, .mode = if (R > a.rows) .chunk else .eager, .first = pos, .count = 1 });
            try g.stream.synchronize();
            try g.block.?.download(g.off(g.b.routed.ptr), out[0 .. R * l.d * 4]);
            var sum: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(out[0 .. R * l.d * 4], &sum, .{});
            all.update(&sum);
            n += 1;
            std.debug.print("SELFTEST half {d} layer {d} rows {d} sha256 {s}\n", .{ a.half, layer, R, &std.fmt.bytesToHex(sum, .lower) });
        }
    }
    const total = std.fmt.bytesToHex(all.finalResult(), .lower);
    std.debug.print("SELFTEST half {d}: {d} replies, seed {d}, all {s}\n", .{ a.half, n, a.selftest.?, &total });
    return done(0, "AFD OK expert {d} selftest: {d} replies, seed {d}, all {s}", .{ a.half, n, a.selftest.?, &total });
}

/// --check-weights FILE: the half's buffers in qualification_2b1188e0's weights.json order, one JSON line each (check-weights' paths and
/// shapes; sha256 of the bytes on the device; none for the address tables); with --ref, each against qualification_2b1188e0's.
pub fn checkWeights(gpa: Allocator, io: std.Io, a: Args) !u8 {
    var ref: ?Ref = null;
    defer if (ref) |*r| r.deinit();
    if (a.ref) |p| {
        const text = try readFile(gpa, p);
        defer gpa.free(text);
        ref = try Ref.init(gpa, text);
    }
    const g = try Gpu.open(gpa, io, a, false);
    defer g.close();
    const p = try std.fmt.allocPrintSentinel(gpa, "{s}", .{a.check_weights.?}, 0);
    defer gpa.free(p);
    const fd = std.c.open(p, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .CLOEXEC = true }, @as(c_uint, 0o644));
    if (fd < 0) return error.NoOutput;
    defer _ = std.c.close(fd);
    var list: std.ArrayList(ae.Entry) = .empty;
    defer list.deinit(gpa);
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    var host: []u8 = &.{};
    defer gpa.free(host);
    var count: usize = 0;
    var t: Tally = .{};
    for (g.groups, 0..) |d, pos| {
        list.clearRetainingCapacity();
        try ae.entries(gpa, &g.x, pos, &list);
        aw.clearRetainingCapacity();
        for (list.items) |e| {
            if (host.len < e.bytes) host = try gpa.realloc(host, @intCast(e.bytes));
            const data = host[0..@intCast(e.bytes)];
            if (!ae.isTable(e)) try d.download(@intCast(e.off), data);
            const sum = try ae.digest(e, data);
            try ae.writeDigest(&aw.writer, e, sum);
            if (ref) |*r| t.add(r, e, r.check(e, sum, true));
            count += 1;
        }
        const w = aw.written();
        if (std.c.write(fd, w.ptr, w.len) != @as(isize, @intCast(w.len))) return error.ShortWrite;
    }
    std.debug.print("CHECK-WEIGHTS half {d}: {d} buffers of {d} MoE layers into {s}\n", .{ a.half, count, g.x.layers.len, a.check_weights.? });
    const r = if (ref) |*x| x else return done(0, "AFD OK expert {d} check-weights: {d} buffers", .{ a.half, count });
    t.finish(r, g.x.layers);
    std.debug.print("CHECK-WEIGHTS half {d} against {s}: {d} equal, {d} differ, {d} not in it, {d} it has that we lack, {d} in layers it does not cover; first wrong {s}\n", .{ a.half, a.ref.?, t.equal, t.differ, t.extra, t.lacking, t.outside, t.first[0..t.first_len] });
    if (!t.ok()) return done(1, "AFD FAILED check-weights peer - seq {d}", .{t.differ + t.extra + t.lacking});
    return done(0, "AFD OK expert {d} check-weights: {d} buffers, {d} equal to the ref", .{ a.half, count, t.equal });
}

test "the expert command's flags: one mode, the half, the model; IP:PORT; --layers, --fault, --ref; usage errors" {
    const ok = try parse(&.{ "--half", "1", "--model", "/m", "--triton", "/t", "--master", "192.0.2.20:29600", "--dev", "mlx5_0,mlx5_1" });
    try std.testing.expectEqual(@as(u32, 1), ok.half);
    try std.testing.expectEqualStrings("mlx5_0,mlx5_1", ok.devs);
    try std.testing.expectEqual(@as(f64, 20), ok.give_up_s);
    try std.testing.expectEqual(@as(usize, 2048), ok.prefill_rows);
    try std.testing.expectEqual(@as(?u32, null), ok.layers);
    const st = try parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "7", "--layers", "8" });
    try std.testing.expectEqual(@as(?u64, 7), st.selftest);
    try std.testing.expectEqual(@as(?u32, 8), st.layers);
    const cw = try parse(&.{ "--half", "0", "--model", "/m", "--check-weights", "/tmp/w.jsonl", "--ref", "/z/weights-rank0.json", "--layers", "8" });
    try std.testing.expectEqualStrings("/z/weights-rank0.json", cw.ref.?);
    const dead = try parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--master", "1.2.3.4:5", "--fault", "exit:200" });
    try std.testing.expectEqual(ae.Fault{ .what = .exit, .at = 200 }, dead.fault.?);
    try std.testing.expectEqual(ae.Fault{ .what = .silent, .at = 1 }, try fault("silent:1"));
    const ex = std.testing.expectError;
    try ex(error.BadFault, fault("silent:0"));
    try ex(error.BadFault, fault("hang:3"));
    try ex(error.BadFault, fault("exit"));
    try ex(error.FaultNeedsServe, parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--fault", "exit:2" }));
    try ex(error.RefNeedsCheck, parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--ref", "w.json" }));
    try ex(error.BadValue, parse(&.{ "--half", "0", "--model", "/m", "--check-weights", "f", "--layers", "0" }));
    try ex(error.OneMode, parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t" }));
    try ex(error.OneMode, parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--master", "1.2.3.4:5" }));
    try ex(error.MissingFlag, parse(&.{ "--half", "2", "--model", "/m", "--triton", "/t", "--selftest", "1" }));
    try ex(error.MissingFlag, parse(&.{ "--half", "0", "--model", "/m", "--selftest", "1" }));
    try ex(error.MissingValue, parse(&.{ "--half", "0", "--model" }));
    try ex(error.UnknownFlag, parse(&.{ "--halves", "0" }));
    try ex(error.BadValue, parse(&.{ "--half", "0", "--model", "/m", "--triton", "/t", "--selftest", "1", "--rows", "0" }));
    try std.testing.expectEqual(@as(u16, 29600), (try master("192.0.2.20:29600")).getPort());
    try ex(error.BadMaster, master("192.0.2.20"));
}

test "the selftest's generator: pins for the python twin (evidence/qualification_cec3c04c/scripts/selftest_gen.py prints the same)" {
    try std.testing.expectEqual(@as(u64, 0xE220A8397B1DCDAF), Gen.mix(0)); // SplitMix64's first output for seed 0
    const k = Gen.key(1, 1, 3, 1);
    const kl = Gen.key(1, 2, 3, 1);
    try std.testing.expectEqual(@as(u64, 0xdbff181846beba6e), k);
    try std.testing.expectEqual([3]u16{ 0xc117, 0xc088, 0xbda9 }, [3]u16{ Gen.rowBits(k, 0, 0), Gen.rowBits(k, 0, 1), Gen.rowBits(k, 0, 4095) });
    try std.testing.expectEqual([2]u32{ 0x3fd94e44, 0xc065bb2e }, [2]u32{ @bitCast(Gen.logit(kl, 0, 0)), @bitCast(Gen.logit(kl, 0, 287)) });
    for (0..4096) |c| {
        const b = Gen.rowBits(k, 0, c);
        const e = (b >> 7) & 0xff;
        try std.testing.expect(e >= 120 and e <= 130); // normal, |x| in [2^-7, 2^4)
    }
    for (0..288) |e| {
        const x = Gen.logit(Gen.key(1, 2, 3, 1), 0, e);
        try std.testing.expect(x >= -4 and x < 4);
    }
}

test {
    _ = afd_ref; // afd_expert_ref.zig's test runs with this file's
    _ = hsw; // qualification_f504f9df: afd_expert_halves.zig's test too
    _ = g53x; // qualification_1b9bf461: afd_expert_g53.zig's
    _ = k4g; // qualification_e4ad04cd: afd_expert_k4.zig's
    _ = afd_fp; // rank_fingerprint: afd_fp.zig's
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Mia's 0128 switches in the expert (qualification_96d80965): unset is off; 4 and 1 are on; 65 or 2 refuse the run" {
    defer _ = unsetenv("TF_GLM_EXL3_L2D");
    defer _ = unsetenv("TF_GLM_EXL3_PDL");
    try std.testing.expect(!(try sched(0)).on());
    _ = setenv("TF_GLM_EXL3_L2D", "4", 1);
    _ = setenv("TF_GLM_EXL3_PDL", "1", 1);
    try std.testing.expectEqual(glm.exl3.experts.Sched{ .l2d = 4, .pdl = true }, try sched(1));
    _ = setenv("TF_GLM_EXL3_L2D", "65", 1);
    try std.testing.expectError(error.BadExl3Sched, sched(1));
    _ = setenv("TF_GLM_EXL3_L2D", "0", 1);
    _ = setenv("TF_GLM_EXL3_PDL", "2", 1);
    try std.testing.expectError(error.BadExl3Sched, sched(0));
}

test "X1 in the expert (qualification_ae31a4aa): TF_GLM_EXL3_X1 unset or 0 reads nothing; 1 turns python's four on, each =0 off; a bad value refuses the run" {
    const names = [_][:0]const u8{ "TF_GLM_EXL3_X1", "TF_GLM_EXL3_NFIRST", "TF_GLM_EXL3_PF", "TF_GLM_EXL3_ORDER", "TF_GLM_EXL3_ROWROT" };
    defer for (names) |n| {
        _ = unsetenv(n);
    };
    for (names) |n| _ = unsetenv(n);
    try std.testing.expect(!(try x1(0)).on());
    _ = setenv("TF_GLM_EXL3_ORDER", "yes", 1); // not read while X1 is unset
    try std.testing.expect(!(try x1(0)).on());
    _ = setenv("TF_GLM_EXL3_X1", "0", 1);
    try std.testing.expect(!(try x1(0)).on());
    _ = setenv("TF_GLM_EXL3_X1", "1", 1);
    try std.testing.expectError(error.BadExl3X1, x1(1));
    _ = setenv("TF_GLM_EXL3_ORDER", "0", 1);
    try std.testing.expectEqual(glm.exl3.experts.Prompt{ .nfirst = true, .pa = true, .order = false, .xrow = true }, try x1(1));
    _ = unsetenv("TF_GLM_EXL3_ORDER");
    try std.testing.expectEqual(glm.exl3.experts.Prompt{ .nfirst = true, .pa = true, .order = true, .xrow = true }, try x1(0));
    _ = setenv("TF_GLM_EXL3_X1", "2", 1);
    try std.testing.expectError(error.BadExl3X1, x1(0));
}
