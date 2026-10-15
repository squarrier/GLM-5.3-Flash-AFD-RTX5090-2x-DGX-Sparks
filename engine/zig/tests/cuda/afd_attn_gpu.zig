//! tf-glm-afd attn first|teacher on a GPU (qualification_cd073cc0; afd_attn.zig hands the verbs here unless --dry): python 2.15's
//! attention role at world 1 in its bring-up order. The listening socket first (the experts' join retries), then:
//! - the load: qualification_9051d0c1's world-1 plan through lane B's EXL3 pack pass (glm.afd_load.load: the routed experts left out, the
//!   draft head's quant4 packed), read and packed on the device (afd_load.toDevice) and bound with qualification_qualification_50384483's guard;
//! - B3's Engine on its device block (the prompt chunks' DSA Scratch among it: qualification_qualification_0c5fe618's), the Triton set from --triton,
//!   fwd.Kernels (kda, mm, dsa: the attention's dense weights are BF16, so no EXL3 image) and the sampler's torch ops;
//! - the exchange: native RoCE exchange's attention Node over RoCE (--dev, --gid), both experts attended, Afd.attach on the
//!   forward's stream with fn_xchg's launches, and with --audit the pinned ring (Afd.audit);
//! - the verb through qualification_0fa2bee9's engine (afd_attn.zig's first and teacher), then the experts' stop frame.
//! `drySet` is the same fwd.Kernels at stand-in addresses (--dry), refused where this path refuses.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const options = @import("glm_afd_options");
const cli = @import("tf_glm_afd.zig");
const attn = @import("afd_attn.zig");
const bench = @import("afd_attn_bench.zig");
const drafted = @import("afd_attn_drafts.zig"); // qualification_21e21bc0: TF_AFD_MTP's model and its INFO line
const serve = @import("afd_attn_serve.zig"); // qualification_4be15bbf: attn serve
const files = @import("afd_expert.zig");
const afd_fp = @import("afd_fp.zig"); // rank_fingerprint: the identity line (TF_GLM_FINGERPRINT=1)

const xchg = cuda.xchg;
const proto = xchg.proto;
const Allocator = std.mem.Allocator;

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// The images glm_afd.zig's tool embeds with -Dnvcc or -Dfatbins (a host build embeds none).
const img = struct {
    const pack: []const u8 = if (options.glm_pack) &Blob("glm_pack_fatbin").bytes else &.{};
    const kda: []const u8 = if (options.run_images) &Blob("glm_kda_fatbin").bytes else &.{};
    const pointwise: []const u8 = if (options.run_images) &Blob("torch_pointwise_fatbin").bytes else &.{};
    const topk: []const u8 = if (options.run_images) &Blob("torch_topk_fatbin").bytes else &.{};
    const indexing: []const u8 = if (options.run_images) &Blob("torch_indexing_fatbin").bytes else &.{};
    const movement: []const u8 = if (options.run_images) &Blob("torch_movement_fatbin").bytes else &.{};
    const qmm_group: []const u8 = if (options.mm_images) &Blob("glm_qmm_group_fatbin").bytes else &.{};
    const qmm_prefill: []const u8 = if (options.mm_images) &Blob("glm_qmm_prefill_fatbin").bytes else &.{};
    const experts: []const u8 = if (options.mm_images) &Blob("glm_experts_fatbin").bytes else &.{};
    const experts_prefill: []const u8 = if (options.mm_images) &Blob("glm_experts_prefill_fatbin").bytes else &.{};
    const select: []const u8 = if (options.dsa_image) &Blob("glm_select_fatbin").bytes else &.{};
    const nucleus: []const u8 = if (options.nucleus_images) &Blob("torch_nucleus_fatbin").bytes else &.{};
    const argmax_f64: []const u8 = if (options.nucleus_images) &Blob("torch_argmax_f64_fatbin").bytes else &.{};
    /// grammar_mask: the masked adapter's image (TF_AFD_GRAMMAR_MASK=1 only; never part of `images`)
    const grammar_mask: []const u8 = if (options.grammar_image) &Blob("torch_grammar_mask_fatbin").bytes else &.{};
};

/// Whether this build embeds every image the GPU path loads (a host build: its GPU verbs refuse, its --dry records).
pub const images = options.glm_pack and options.run_images and options.mm_images and options.dsa_image;

const Kda = glm.kda_kernels.Kda;
const Mm = @typeInfo(@FieldType(glm.fwd.Kernels, "mm")).pointer.child;
const Dsa = @typeInfo(@typeInfo(@FieldType(glm.fwd.Kernels, "dsa")).optional.child).pointer.child;

/// The extension kernels the attention's forward launches: KDA's, lane A's Mm (fwd.Kernels requires it; the BF16
/// matmuls are Triton's) and DSA select; no EXL3 set (the routed experts are the expert nodes').
fn kernels(kda: *const Kda, mm: *const Mm, dsa: *const Dsa) glm.fwd.Kernels {
    return .{ .kda = kda, .mm = mm, .dsa = dsa };
}

/// Stand-ins for what `run` loads, so --dry builds its set without a driver; nothing dereferences them.
const stand = struct {
    var kda: Kda = undefined;
    var mm: Mm = undefined;
    var dsa: Dsa = undefined;
};

/// The fwd.Kernels `run` passes the forward, at stand-ins (--dry): refused where `run` refuses (no images).
pub fn drySet() !glm.fwd.Kernels {
    if (!images) return error.NoKernelImages;
    return kernels(&stand.kda, &stand.mm, &stand.dsa);
}

/// The set's fields, for the log.
pub fn describe(k: glm.fwd.Kernels) []const u8 {
    if (k.exl3 != null) return "kda, mm, exl3, dsa";
    return if (k.dsa != null) "kda, mm, dsa" else "kda, mm";
}

fn kdaLoad(d: *const cuda.Driver) !Kda {
    var m = try cuda.Module.load(d, img.kda);
    errdefer m.unload();
    const s = glm.kda_kernels.sym;
    return .{ .module = m, .chain_fn = try m.function(s.chain), .replay_fn = try m.function(s.replay), .replay_layers_fn = try m.function(s.replay_layers), .prep_fn = try m.function(s.prep), .step_fn = try m.function(s.step), .out_fn = try m.function(s.out) };
}

/// --dev's RDMA devices (one a port).
pub fn ports(devs: []const u8, out: *[proto.PORTS_MAX][]const u8) ![]const []const u8 {
    var it = std.mem.tokenizeScalar(u8, devs, ',');
    var n: usize = 0;
    while (it.next()) |d| : (n += 1) {
        if (n >= proto.PORTS_MAX) return error.TooManyPorts;
        out[n] = d;
    }
    if (n == 0) return error.NoDevice;
    return out[0..n];
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// grammar_mask's device gate: one synthetic bf16 row into its logits buffer (its DMA may still land: gate() fences it).
fn gateUpload(ctx: *anyopaque, bits: []const u16) anyerror!void {
    const lb: *cuda.DeviceBuffer = @ptrCast(@alignCast(ctx));
    try lb.upload(0, std.mem.sliceAsBytes(bits));
}

fn gib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / (1 << 30);
}

/// attn serve has no prompt of its own: attn.Run's tokens, unread.
var no_ids = [_]u32{};
const no_tokens: attn.Tokens = .{ .name = "serve", .prompt = &no_ids };

/// `attn first|teacher|bench|serve` on device 0, listening on --master for both experts (serve: no tokens of its own).
pub fn run(gpa: Allocator, io: std.Io, f: cli.Flags, t: ?*const attn.Tokens) !cli.Outcome {
    if (!images) return .{ .usage = "no-kernel-images (a host build: tf-glm-afd -Dnvcc or -Dfatbins)" };
    const users = try glm.afd_users_round.fromEnv(); // invalid counts refused before sockets/model allocation
    if (users > 1 and f.verb != .serve) return error.UsersServeOnly;
    const sampled_on = f.verb == .serve and try glm.afd_sampling.fromEnv();
    if (sampled_on and !options.nucleus_images) return error.NoNucleusImages;
    const paired = try glm.afd_pairs.fromEnv(); // qualification_0f991257: TF_AFD_PAIRS, read once before anything loads (BadAfdPairs)
    const inflight = try glm.afd_pairs.inflightFromEnv(paired); // qualification_c533375f: TF_AFD_INFLIGHT, after it (BadAfdInflight)
    const seq = try glm.afd_attn.seqFromEnv(); // qualification_969df694: TF_GLM_EXL3_SEQ for the settings' seq=1, after them (BadExl3Seq)
    const halves = try attn.halvesKernel(); // qualification_f504f9df: TF_GLM_HALVES_KERNEL, read once before anything loads (BadHalvesKernel)
    const stripes = try proto.stripesFromEnv("attn"); // qualification_17c70e37: TF_AFD_STRIPES, read once before anything loads (BadAfdStripes)
    const graphed = try glm.afd_graphs.fromEnv(); // qualification_f16009a8: TF_AFD_GRAPHS, read once before anything loads (BadAfdGraphs)
    const lanes_n = try glm.afd_lanes.fromEnv(paired); // qualification_9e49592e: TF_AFD_LANES, after both (BadAfdLanes)
    if (users > 1 and (graphed or f.max_rows < users)) return error.UsersModeUnsupported;
    const masked = try glm.sample.mask_dev.fromEnv(); // grammar_mask: TF_AFD_GRAMMAR_MASK, read once before anything loads
    const gate_rows = try glm.sample.mask_dev.gateFromEnv(masked); // grammar_mask: TF_AFD_GRAMMAR_GATE, likewise
    const grammar = try @import("afd_attn_grammar.zig").fromEnv(masked); // grammar_server: TF_AFD_GRAMMAR_LIB/_LIB_SHA256 (both or neither) and TF_AFD_GRAMMAR_SERVE (needs MASK), likewise
    if (masked and img.grammar_mask.len == 0) return .{ .usage = "TF_AFD_GRAMMAR_MASK=1 without the torch_grammar_mask image (tf-glm-afd -Dnvcc, or -Dfatbins holding it)" };
    // grammar_join_2: the mask is single-user greedy: refuse TF_AFD_USERS>1 or TF_GLM_SAMPLING=1 (read only with the mask) before anything loads
    if (glm.sample.mask_dev.scopeRefusal(masked, users, masked and try glm.afd_sampling.fromEnv())) |line| return .{ .usage = line };
    const listen = f.master orelse return error.NoMaster;
    const triton = f.triton orelse return .{ .usage = "NoTritonDir" };
    var pb: [proto.PORTS_MAX][]const u8 = undefined;
    const devs = try ports(f.devs, &pb);
    const setup_ms = try xchg.setup.fromEnv("attn"); // qualification_1fee6d78: TF_AFD_SETUP_MS, each expert's accept window, read before anything loads (BadAfdSetupMs)
    const fp_on = try cuda.fingerprint.fromEnv("attn"); // rank_fingerprint: TF_GLM_FINGERPRINT, read before anything loads (BadGlmFingerprint)
    const server = try cuda.link.Server.open(listen); // before the load: an expert's join finds it listening
    defer server.close();
    const t_start = nowNs(); // qualification_e5a7fc3b: the load time, from here to both experts attended
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    // the model, the plan and the load
    const cpath = try std.fmt.allocPrint(gpa, "{s}/config.json", .{f.model});
    defer gpa.free(cpath);
    const text = try files.readFile(gpa, cpath);
    defer gpa.free(text);
    var m = try drafted.model(gpa, text, f.layers); // engine.py:499-502: no MTP under AFD, but with TF_AFD_MTP=1 (qualification_21e21bc0)
    var c = try files.config(gpa, f.model);
    c.run = @intCast(m.layers);
    var ix = try glm.index.open(gpa, f.model);
    defer ix.deinit();
    var ld = try glm.afd_load.loadWith(gpa, &m, &c, &ix.names, m.mtp_layers > 0);
    defer ld.deinit();
    const mtp_bytes = if (m.mtp_layers > 0) try glm.afd_load.mtpBytes(&c, &ix.names) else 0;
    if (c.run == c.layers and ld.x.sourceBytes() != glm.afd_load.attention_bytes + mtp_bytes) return .{ .usage = "plan-bytes-not-qualification_99ec72b7" };
    attn.denseLine(try glm.afd_load.dense(gpa, &ld, &ix, &m), "attn"); // qualification_e5a7fc3b: TF_GLM_DENSE=q4:DIR, TF_GLM_F8=head
    // qualification_4be15bbf: attn serve's window sets the capacity, and its forecast on this GPU is admitted before the load
    var sa = std.heap.ArenaAllocator.init(gpa);
    defer sa.deinit();
    const sp = if (f.verb == .serve) try serve.plan(sa.allocator(), f, text) else null;
    const capacity = if (sp) |p| p.capacity else f.capacity;
    const fc = if (sp) |p| try serve.admit(gpa, &m, ld.deviceBytes(), p, f.max_rows, &ctx, lanes_n) orelse return .{ .usage = "serve-past-the-gpu" } else null;
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const own = if (f.fatbin) |p| try files.readFile(gpa, p) else null;
    defer if (own) |x| gpa.free(x);
    var pk = try glm.pack.Kernels.load(&driver, own orelse img.pack);
    defer pk.deinit();
    const t0 = nowNs();
    var dev = try glm.afd_load.toDevice(gpa, &m, &driver, stream, &pk, &ld, ix.shards, .{});
    defer dev.deinit();
    var w = try glm.views.rank(gpa, &m);
    defer w.deinit(gpa);
    const bound = try glm.afd_load.bind(&w, &ld.pk); // UnboundView: no view the forward reads at address 0
    std.debug.print("INFO attn: the world-1 load: {d} B read, {d} views bound, {d} B packed on the device ({d:.2} GiB; stored groups at most {d:.2} GiB at once) in {d} ms (reads {d} ms, packing {d} ms)\n", .{ ld.x.sourceBytes(), bound, dev.bytes, gib(dev.bytes), gib(dev.peak), (nowNs() - t0) / std.time.ns_per_ms, dev.read_ns / std.time.ns_per_ms, dev.pack_ns / std.time.ns_per_ms });
    // B3's Engine on the device, the kernels, the sampler
    const e = try glm.engine.Engine.create(gpa, &m, &w, capacity, 8, f.max_rows, capacity > m.denseLimit());
    defer e.destroy(gpa);
    if (sp) |p| if (p.pool > 0) glm.afd_pool.detachTokens(&e.st); // shared_pool: slot 0's caches come from the pool's regions
    if (sampled_on) {
        try e.useSampling(gpa);
        e.sampled_top_k = true;
    }
    var edev = try glm.engine.Device.attach(e, &driver);
    defer edev.detach();
    const eb = try e.deviceBytes();
    std.debug.print("INFO attn: the Engine's device block {d:.2} GiB (the prompt Buffers {d:.2} GiB, the DSA Scratch {d:.2} GiB)\n", .{ gib(eb.total), gib(eb.pbuf), gib(eb.dsa) });
    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, triton);
    defer set.deinit();
    var kda = try kdaLoad(&driver);
    defer kda.unload();
    var mm = try Mm.load(&ctx, .{ .qmm_group = img.qmm_group, .qmm_prefill = img.qmm_prefill, .experts = img.experts, .experts_prefill = img.experts_prefill });
    defer mm.unload();
    var dsa = try Dsa.load(&driver, img.select);
    defer dsa.unload();
    e.chip = .{ .major = @intCast(try ctx.attribute(.compute_capability_major)), .minor = @intCast(try ctx.attribute(.compute_capability_minor)), .sms = @intCast(try ctx.attribute(.multiprocessor_count)) };
    var sk = try glm.sample.Kernels.loadImages(&driver, .{ img.pointwise, img.topk, img.indexing, img.movement });
    defer sk.unload();
    var nk: ?glm.nucleus.Kernels = if (sampled_on) try glm.nucleus.Kernels.loadImages(&driver, &sk, .{ img.nucleus, img.argmax_f64 }) else null;
    defer if (nk) |*k| k.unload();
    if (nk) |*k| e.nuc.?.k = k;
    const wide = f.max_rows * m.ll() * 128; // kda._wide_scratch's q (fp32) and y (bf16)
    var q = try cuda.DeviceBuffer.alloc(&driver, wide * 4);
    defer q.free();
    var y = try cuda.DeviceBuffer.alloc(&driver, wide * 2);
    defer y.free();
    const ks = kernels(&kda, &mm, &dsa);
    var fw = glm.fwd.Fwd.launching(&m, .{ .stream = stream, .set = &set, .kernels = ks, .q_tmp = q.ptr, .y_tmp = y.ptr });
    e.smp.k = &sk;
    try glm.sample.prepare(gpa, &fw, &e.smp);
    // grammar_mask: the masked greedy adapter, its own pinned/device words outside the Engine's block (unset: none)
    var gm: ?glm.sample.mask_dev.Dev = null;
    defer if (gm) |*x| x.close();
    if (masked) {
        gm = try glm.sample.mask_dev.Dev.open(&driver, &m, img.grammar_mask);
        e.smp.masked_dev = &gm.?;
        if (grammar) |g| {
            std.debug.print("INFO attn: TF_AFD_GRAMMAR_MASK=1: the masked greedy adapter is loaded (grammar_mask before topk, {d} words a row); TF_AFD_GRAMMAR_SERVE=1: structured output is served through it (structures true); libtfgrammar loads on the first structured request ({s})\n", .{ gm.?.words, g.lib orelse "TF_AFD_GRAMMAR_LIB unset: structured requests are refused" });
        } else std.debug.print("INFO attn: TF_AFD_GRAMMAR_MASK=1: the masked greedy adapter is loaded (grammar_mask before topk, {d} words a row); structured output stays refused to clients (structures false)\n", .{gm.?.words});
        // grammar_mask's internal device gate: seeded synthetic rows on this GPU against the host oracle, before serving
        if (gate_rows) |rows| {
            var lb = try cuda.DeviceBuffer.alloc(&driver, m.vocab * 2);
            defer lb.free();
            var b = glm.views.Buf.of(.bf16, &.{ 1, m.vocab });
            b.ptr = lb.ptr;
            const g = try glm.sample.mask_dev.gate(&gm.?, &fw, &e.smp, gpa, glm.trace.T.whole(b), .{ .ctx = &lb, .put = gateUpload }, rows, 0x4752_4132);
            std.debug.print("INFO attn: grammar_mask gate PASS on this GPU: {d} rows, identity {d}, forbidden {d}, single {d}, all-inf {d} equal the host oracle; staged {d}, masked {d}\n", .{ g.rows, g.identity, g.forbidden, g.single, g.all_inf, gm.?.stats.staged, gm.?.stats.masked });
        }
    }
    std.debug.print("INFO attn: fwd.Kernels {s}; Triton set {s}\n", .{ describe(ks), triton });
    // the exchange: the node, both experts attended, the provider's buffers on the device
    var a = try glm.afd_attn.Afd.initWith(&m, f.max_rows, m.mtp_layers > 0);
    a.paired = paired; // qualification_0f991257: the settings carry pairs, so both halves must have TF_AFD_PAIRS=1 too (SettingsDiffer)
    a.inflight = inflight; // qualification_c533375f: and inflight=2 with TF_AFD_INFLIGHT=2 (both halves must have it too)
    a.exl3_seq = seq; // qualification_969df694: and seq=1 with TF_GLM_EXL3_SEQ=1 (both halves must run SEQ too)
    if (seq) std.debug.print("INFO attn: TF_GLM_EXL3_SEQ=1: the settings carry seq=1, so both halves must run SEQ (the attention's launches are unchanged)\n", .{});
    a.halves = halves.setting(); // qualification_f504f9df: the settings carry the halves' kernel, so both halves must run it too (SettingsDiffer)
    a.stripes = stripes; // qualification_17c70e37: the settings carry stripes, so both halves must have the same TF_AFD_STRIPES (SettingsDiffer)
    if (lanes_n > 0) { // qualification_9e49592e: inflight=N lanes=N, a region of N slots a source (both halves must have TF_AFD_LANES=N too)
        a.inflight = lanes_n;
        a.lanes = lanes_n;
    }
    drafted.info(&a, "load");
    var cfg = a.nodeConfig(f.max_rows, devs.len, f.silent_s, f.give_up_s, f.watcher);
    var ib: [cuda.fingerprint.line_max]u8 = undefined; // rank_fingerprint: fp=1 (both halves must have TF_GLM_FINGERPRINT=1 too) and this node's line
    if (fp_on) {
        var lb: [16]afd_fp.Image = undefined;
        const loaded = afd_fp.attnImages(&lb, .{ .pack = own orelse img.pack, .kda = img.kda, .qmm_group = img.qmm_group, .qmm_prefill = img.qmm_prefill, .experts = img.experts, .experts_prefill = img.experts_prefill, .select = img.select, .pointwise = img.pointwise, .topk = img.topk, .indexing = img.indexing, .movement = img.movement, .nucleus = img.nucleus, .argmax_f64 = img.argmax_f64, .grammar_mask = img.grammar_mask }, sampled_on, masked);
        cfg.settings.fp = true;
        cfg.identity = try afd_fp.attn(&ib, loaded, triton, f.model, try ctx.capability());
    }
    if (stripes.k != 0) std.debug.print("INFO attn: TF_AFD_STRIPES={d}/{d}: messages of {d} KiB or more go in {d} stripes, each taken as it lands\n", .{ stripes.k, stripes.min, stripes.k * stripes.min / 1024, stripes.k });
    const total: usize = @intCast(cfg.settings.region().total()); // qualification_9e49592e: N slots a source with TF_AFD_LANES=N
    var region = try cuda.HostBuffer.allocMapped(&driver, total);
    defer region.free();
    @memset(region.bytes, 0);
    var status = try cuda.HostBuffer.allocMapped(&driver, 4096);
    defer status.free();
    @memset(status.bytes, 0);
    const mem: xchg.Memory = .{ .region = region.bytes, .region_dev = try region.device(), .status = status.slice(u32)[0..proto.STATUS_WORDS], .status_dev = try status.device() };
    var xg = try xchg.Gpu.init(&driver);
    defer xg.deinit();
    var node = try xchg.Node.init(gpa, io, .attn, xchg.setup.with(cfg, setup_ms), mem); // qualification_1fee6d78: unset, cfg's 120,000 ms
    defer node.deinit();
    var roce: xchg.Roce = .{ .ports = devs, .gid = f.gid };
    var tb: [256]u8 = undefined;
    std.debug.print("INFO attn: {s}; {d} MoE layers a forward; attending both experts on {f}\n", .{ try cfg.settings.text(&tb), a.layers, listen });
    try node.attend(server, roce.opener());
    defer node.stopExperts();
    try a.attach(&driver, .{ .node = &node, .gpu = &xg });
    defer a.detach();
    const load_ms = @as(f64, @floatFromInt(nowNs() - t_start)) / 1e6;
    std.debug.print("TIME attn: load: {d:.1} ms from the listening socket to both experts attended and the exchange attached (the weights' reads {d} ms, packing {d} ms)\n", .{ load_ms, dev.read_ns / std.time.ns_per_ms, dev.pack_ns / std.time.ns_per_ms });
    var at = try glm.afd_engine.Attn.init(e, &a);
    var pp: glm.afd_pairs.Pairs = undefined; // qualification_0f991257: B's copies on the device, B's audit ring (TF_AFD_PAIRS=1 only)
    defer if (at.pairs) |p| p.detach();
    if (paired and lanes_n == 0) {
        pp = glm.afd_pairs.Pairs.init(&at);
        try pp.attach(&driver);
        pp.agreed = cfg.settings.pairs; // node.attend passed: both halves' settings carry pairs too
        at.pairs = &pp;
        std.debug.print("INFO attn: TF_AFD_PAIRS=1: a prompt of 2+ chunks prefills in pairs (B's copies {d:.3} GiB on the device, not in serve's forecast: within its headroom)\n", .{gib(try pp.bytes())});
        if (inflight > 1) std.debug.print("INFO attn: TF_AFD_INFLIGHT=2: two requests in flight a link inside a pair (B's shared-expert rows {d:.3} GiB, among B's copies)\n", .{gib(if (pp.c.sy) |s| s.bytes() else 0)});
    }
    // qualification_f16009a8: TF_AFD_GRAPHS=1, decode windows and MTP steps replayed from CUDA graphs by key on the forward's stream
    var cu: glm.afd_graphs.Cuda = .{ .stream = stream };
    var gr = glm.afd_graphs.Graphs.init(gpa, cu.ops(), "attn");
    defer gr.deinit();
    if (graphed) {
        at.graphs = &gr;
        std.debug.print("INFO attn: TF_AFD_GRAPHS=1: decode windows of up to {d} rows and MTP steps replayed from CUDA graphs by key (at most {d} kept; a failed capture runs eagerly)\n", .{ @min(e.buf.rows, glm.afd_graphs.max_rows), gr.max });
    }
    var ln: glm.afd_lanes.Lanes = undefined; // qualification_9e49592e: lanes 1..N-1's copies on the device and their rings (TF_AFD_LANES=N only)
    defer if (at.lanes) |l| l.detach();
    if (lanes_n > 0) {
        ln = try glm.afd_lanes.Lanes.init(&at);
        try ln.attach(&driver); // all of it at the open, before the first prompt (python's IF4 OOM was a per-call alloc)
        ln.agreed = cfg.settings.lanes == lanes_n; // node.attend passed: both halves' settings carry lanes=N too
        at.lanes = &ln;
        std.debug.print("INFO attn: TF_AFD_LANES={d}: a prompt of 2+ chunks prefills in lanes of up to {d}, {d} requests in flight a link (lanes 1-{d}'s copies {d:.3} GiB on the device, reserved at the open; in serve's forecast)\n", .{ lanes_n, lanes_n, lanes_n, lanes_n - 1, gib(try ln.bytes()) });
    }
    var recs: attn.Records = .{ .gpa = gpa };
    defer recs.deinit();
    const layer_recs = try gpa.alloc(glm.afd_audit.Layer, a.positions());
    defer gpa.free(layer_recs);
    if (f.audit) {
        try a.audit(&driver, e.buf.rows);
        if (at.pairs) |p| try p.audit(&driver, &a, e.buf.rows);
        if (at.lanes) |l| try l.audit(&driver, &a, e.buf.rows);
        at.check = .{ .ctx = &recs, .forward = attn.Records.forward, .layers = layer_recs };
    }
    const out = f.out orelse ".";
    const r: attn.Run = .{ .gpa = gpa, .at = &at, .f = &fw, .stream = stream, .out = out, .t = t orelse &no_tokens };
    const res: anyerror!usize = switch (f.verb) {
        .first => if (attn.first(r)) |x| @as(usize, x) else |err| err,
        .teacher => attn.teacher(r, f.steps),
        .bench => bench.run(r, f.bench, f.prompt.?, load_ms),
        .serve => serve.run(gpa, io, f, .{ .at = &at, .f = &fw, .ctx = &ctx, .node = &node, .plan = sp.?, .load_ms = load_ms, .forecast = fc, .grammar = grammar }),
        else => error.BadVerb,
    };
    stream.synchronize() catch {};
    if (graphed) {
        var gb: [320]u8 = undefined;
        std.debug.print("INFO attn: {s}\n", .{gr.line(&gb)});
    }
    if (f.audit) try attn.writeRecords(&recs, out);
    return attn.outcome(gpa, &at, res, @tagName(f.verb));
}

/// `attn check-weights` on a GPU: the plan's groups read and packed on device 0 (afd_load.toDevice) and bound, as the
/// first and teacher verbs load them; the device bytes.
pub fn checkWeights(gpa: Allocator, io: std.Io, f: cli.Flags, m: *const glm.model.Model, c: *const glm.Config, ix: *glm.index.Index) !cli.Outcome {
    _ = io;
    if (!images) return .{ .usage = "no-kernel-images (a host build: tf-glm-afd -Dnvcc or -Dfatbins)" };
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var ld = try glm.afd_load.loadWith(gpa, m, c, &ix.names, m.mtp_layers > 0);
    defer ld.deinit();
    attn.denseLine(try glm.afd_load.dense(gpa, &ld, ix, m), "check-weights (GPU)");
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const own = if (f.fatbin) |p| try files.readFile(gpa, p) else null;
    defer if (own) |x| gpa.free(x);
    var pk = try glm.pack.Kernels.load(&driver, own orelse img.pack);
    defer pk.deinit();
    var dev = try glm.afd_load.toDevice(gpa, m, &driver, stream, &pk, &ld, ix.shards, .{});
    defer dev.deinit();
    var w = try glm.views.rank(gpa, m);
    defer w.deinit(gpa);
    const bound = try glm.afd_load.bind(&w, &ld.pk);
    std.debug.print("INFO attn check-weights: read and packed on the GPU: {d} B on the device ({d:.2} GiB), {d} views bound; reads {d} ms, packing {d} ms\n", .{ dev.bytes, gib(dev.bytes), bound, dev.read_ns / std.time.ns_per_ms, dev.pack_ns / std.time.ns_per_ms });
    return .{ .ok = std.fmt.allocPrint(gpa, "check-weights {d} B on the device, {d} views bound", .{ dev.bytes, bound }) catch "check-weights" };
}

test "the GPU path's set at stand-ins is refused in a host build; --dev's ports" {
    if (!images) try std.testing.expectError(error.NoKernelImages, drySet()) else try std.testing.expectEqualStrings("kda, mm, dsa", describe(try drySet()));
    var pb: [proto.PORTS_MAX][]const u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), (try ports("mlx5_0,mlx5_1", &pb)).len);
    try std.testing.expectError(error.NoDevice, ports("", &pb));
    _ = &run;
    _ = &checkWeights;
}
