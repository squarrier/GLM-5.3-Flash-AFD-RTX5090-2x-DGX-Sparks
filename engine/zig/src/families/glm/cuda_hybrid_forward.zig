//! Portable native hybrid role ownership. Site controllers, hardware admission and HTTP stay outside this module.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const tp = @import("tp");
const cfg = @import("cuda_hybrid_config.zig");
const tr = @import("cuda_hybrid_transport.zig");
const exl3 = @import("cuda_exl3.zig");
const expert = @import("cuda_hybrid_expert.zig");
const load = @import("cuda_hybrid_load.zig");
const ranks = @import("cuda_ranks.zig");
const model = @import("config.zig");
const Engine = @import("cuda_engine.zig").Engine;
pub const Receipt = tp.roce.Roce.CloseReceipt;
const TestHooks = struct {
    boot: ?*const fn (cfg.Config, u1, u32, u64) anyerror!tp.Bootstrap = null,
    roce: ?*const fn (?*const cuda.Driver, *tp.Bootstrap, []const u8, tp.roce.Options, *?*tp.roce.Roce) anyerror!*tp.roce.Roce = null,
    expert: ?*const fn (*ExpertOwner, std.Io, model.Config, []const u8, u64) anyerror!cuda.grouped.Router = null,
};
// No runtime knob, exported hook or fixture implementation exists in a serving executable.
var test_hooks: if (@import("builtin").is_test) TestHooks else void = if (@import("builtin").is_test) .{} else {};
fn openBootstrap(c: cfg.Config, half: u1, rank: u32, until: u64, requested: ?*const std.atomic.Value(bool)) !tp.Bootstrap {
    try stopBefore(requested, until);
    if (comptime @import("builtin").is_test) if (test_hooks.boot) |f| {
        var b = try f(c, half, rank, until);
        b.cancel_requested = requested;
        return b;
    };
    return tp.Bootstrap.openBeforeControlled(.{ .world = 2, .rank = rank, .master = c.master[half], .port = c.ports[half], .link_port = c.ports[half] + 7, .backend = .roce, .roce_fallback = .@"error", .connect_timeout_ns = @as(u64, c.setup_ms) * std.time.ns_per_ms }, until, requested);
}
fn openRoce(d: *const cuda.Driver, b: *tp.Bootstrap, image: []const u8, o: tp.roce.Options, custody: *?*tp.roce.Roce) !*tp.roce.Roce {
    if (comptime @import("builtin").is_test) if (test_hooks.roce) |f| return f(d, b, image, o, custody);
    return tp.roce.Roce.initOwned(d, b, image, o, custody);
}
fn deadline(ms: u32) u64 {
    return tp.sock.nowNs() +| (@as(u64, ms) * std.time.ns_per_ms);
}
fn before(until: u64) !void {
    if (tp.sock.nowNs() >= until) return error.Timeout;
}
pub fn modelConfig(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config) !model.Config {
    return modelConfigControlled(a, io, dir, c, null);
}
pub fn modelConfigControlled(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, requested: ?*const std.atomic.Value(bool)) !model.Config {
    if (requested) |flag| if (flag.load(.acquire)) return error.Stopped;
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24));
    defer a.free(text);
    var why: core.modelopt.Why = .{};
    try exl3.admit(a, text, &why);
    const shape = try model.parse(a, text);
    if (!shape.exl3 or shape.hidden != 4096 or shape.topk != 8 or shape.experts != 288 or shape.moe_inter != 2048 or (c.mtp and shape.mtp == 0) or c.settings.layers > shape.layers) return error.BadExpertGeometry;
    _ = try @import("cuda_hybrid_preload.zig").read(a, io, dir, c);
    if (requested) |flag| if (flag.load(.acquire)) return error.Stopped;
    const actual = try ranks.artifactControlled(a, io, dir, requested);
    const identity = try c.identity(a);
    if (!std.mem.eql(u8, &actual, &identity.model)) return error.BadHybridModel;
    return shape;
}

/// Historical native195 FrameIO formula, retained for its old fixture callers.
/// Selected direct-V3 admission uses expertExtraBytesFor below, not this geometry.
pub fn expertExtraBytes() struct { device: u64, host: u64 } {
    return .{ .device = expert.Worker.storageBytes() + 2 * tr.wire.batch_bytes + 16, .host = tr.wire.request_bytes + tr.wire.reply_bytes + 2 * tr.wire.post_bytes + 2 * tr.wire.batch_bytes + tp.roce.Roce.regionBytes(tr.wire.post_bytes) };
}
/// Actual owning arenas and mapped registered backing, not views or old fixed128
/// staging. CUDA module/verbs-private/heap costs remain explicitly unknown.
pub fn expertExtraBytesFor(c: cfg.Config) struct { device: u64, host: u64 } {
    return .{ .device = v3.ExpertDevice.storageBytes(maxRows(c), true), .host = Exchange.storageBytes(c) };
}
/// Each pair owns its Bootstrap, native Roce and pinned/device frames. It never owns the model's stream.
pub const Peer = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    boot: ?tp.Bootstrap = null,
    roce: ?*tp.roce.Roce = null,
    frame: ?tr.NativePair = null,
    proof: ?cuda.Event = null,
    closed: bool = false,
    retained: bool = false,
    cancel_requested: ?*const std.atomic.Value(bool) = null,
    pub fn start(p: *Peer, a: std.mem.Allocator, io: std.Io, c: cfg.Config, half: u1, rank: u32, until: u64) !void {
        try stopBefore(p.cancel_requested, until);
        p.proof = try cuda.Event.init(p.d, false);
        p.boot = try openBootstrap(c, half, rank, until, p.cancel_requested);
        const identity = try c.identity(a);
        var h = tr.wire.Header{ .session = identity.session, .sequence = 1, .half = half, .live_rows = 1, .call_rows = 1, .model_contract_sha256 = identity.model, .settings_sha256 = identity.settings };
        h.bind();
        var contract: [256]u8 = undefined;
        h.encode(&contract);
        try p.boot.?.agree(&contract, "native hybrid WIRE4 contract");
        const image = try std.Io.Dir.cwd().readFileAlloc(io, c.roce_image, a, .limited(64 << 20));
        defer a.free(image);
        const spec_type = @typeInfo(@TypeOf(@as(tp.roce.Options, .{}).hcas)).pointer.child;
        var specs = [_]spec_type{.{}};
        if (c.verbs_device.len > specs[0].name.len) return error.BadHybridContract;
        @memcpy(specs[0].name[0..c.verbs_device.len], c.verbs_device);
        specs[0].name_len = @intCast(c.verbs_device.len);
        try stopBefore(p.cancel_requested, until);
        {
            const memory_scope = @import("cuda_hybrid_ledger.zig").scopeOwner(.transport);
            defer memory_scope.end();
            p.roce = try openRoce(p.d, &p.boot.?, image, .{ .wire = .verbs, .hcas = &specs, .traffic_class = c.traffic_class, .max_bytes = tr.wire.post_bytes, .timeout_ns = identity.transaction_timeout_ns }, &p.roce);
        }
        p.frame = try tr.NativePair.init(p.d, p.roce.?, p.stream);
        p.frame.?.control = p.boot.?.plan_fds[1 - rank];
        try p.boot.?.agree("ready", "native hybrid frame storage ready");
        try stopBefore(p.cancel_requested, until);
    }
    fn keep(p: *Peer) void {
        p.retained = true;
        if (p.roce) |r| r.abort();
    }
    pub fn bytes(p: *const Peer) usize {
        var n: usize = 0;
        if (p.roce) |r| n += r.region.len;
        if (p.frame) |f| n += f.frames.len + f.host.bytes.len;
        return n;
    }
    pub fn closeChecked(p: *Peer, until: u64) Receipt {
        if (p.closed) return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
        if (p.retained) return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .failed };
        if (tp.sock.nowNs() >= until) {
            p.keep();
            return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .deadline };
        }
        if (p.frame) |f| if (f.failed != null) {
            p.keep();
            return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .failed };
        };
        if (p.proof) |ev| ev.record(p.stream) catch {
            p.keep();
            return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .gpu };
        };
        if (p.roce) |r| {
            const events: []const cuda.Event = if (p.proof) |*ev| @as([*]const cuda.Event, @ptrCast(ev))[0..1] else &.{};
            const receipt = r.closeChecked(until, events);
            if (!receipt.quiet) {
                p.retained = true;
                return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = receipt.reason };
            }
            p.roce = null;
        }
        if (p.roce == null) if (p.proof) |ev| while (true) {
            if (tp.sock.nowNs() >= until) {
                p.keep();
                return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .deadline };
            }
            const rc = ev.d.api.cuEventQuery(ev.handle);
            if (rc == cuda.abi.success) break;
            if (rc != cuda.abi.error_not_ready) {
                p.keep();
                return .{ .quiet = false, .retained_bytes = p.bytes(), .reason = .gpu };
            }
            std.atomic.spinLoopHint();
        };
        if (p.frame) |*f| {
            f.releaseStorage();
            p.frame = null;
        }
        if (p.proof) |*ev| {
            ev.deinit();
            p.proof = null;
        }
        if (p.boot) |*b| {
            b.close();
            p.boot = null;
        }
        p.closed = true;
        return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
    }
};

// AI-assisted integration: sole current-native Session now owns the released V3 mapped/device exchange.
const v3 = @import("cuda_hybrid_v3.zig");
const exchange_api = cuda.xchg;
pub const V3Testing = if (@import("builtin").is_test) struct {
    pub const Expert = ExpertOwner;
    pub var wire: ?*const fn (*Exchange, u64) anyerror!void = null;
} else struct {};
pub fn maxRows(c: cfg.Config) u32 {
    return if (c.settings.solo) 4096 else 2048;
}
pub const Exchange = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    c: cfg.Config,
    node: ?exchange_api.Node = null,
    gpu: ?exchange_api.Gpu = null,
    region: ?cuda.HostBuffer = null,
    status: ?cuda.HostBuffer = null,
    proof: ?cuda.Event = null,
    stream: cuda.Stream,
    closed: bool = false,
    retained: bool = false,
    pub fn storageBytes(c: cfg.Config) u64 {
        return (exchange_api.proto.Region{ .slot = (exchange_api.proto.Layout{ .d = 4096, .s = 9 }).slotBytes(maxRows(c)), .slots = 4 }).total() + exchange_api.proto.STATUS_WORDS * 4;
    }
    fn open(x: *Exchange, io: std.Io, until: u64, requested: ?*const std.atomic.Value(bool)) !void {
        try stopBefore(requested, until);
        const scope = @import("cuda_hybrid_ledger.zig").scopeOwner(.transport);
        defer scope.end();
        const l = exchange_api.proto.Layout{ .d = 4096, .s = 9 };
        const id = try x.c.identity(x.gpa);
        var contract = std.crypto.hash.sha2.Sha256.init(.{});
        contract.update(std.mem.asBytes(&id.session));
        contract.update(&id.model);
        contract.update(&id.settings);
        const settings = exchange_api.proto.Settings{ .layout = l, .max_rows = maxRows(x.c), .ports = 1, .pairs = true, .inflight = 4, .lanes = 4, .halves = "g53kt3r", .native_contract = contract.finalResult() };
        x.region = try cuda.HostBuffer.allocMapped(x.d, @intCast(settings.region().total()));
        @memset(x.region.?.bytes, 0);
        x.status = try cuda.HostBuffer.allocMapped(x.d, exchange_api.proto.STATUS_WORDS * 4);
        @memset(x.status.?.bytes, 0);
        x.proof = try cuda.Event.init(x.d, false);
        x.node = try exchange_api.Node.init(x.gpa, io, if (x.c.role == .attention) .attn else .{ .expert = try x.c.half() }, .{ .settings = settings, .give_up_ns = @as(u64, x.c.transaction_ms) * std.time.ns_per_ms, .setup_timeout_ms = x.c.setup_ms, .rules = .{ .silent_ns = @as(u64, x.c.transaction_ms) * std.time.ns_per_ms } }, .{ .region = x.region.?.bytes, .region_dev = try x.region.?.device(), .status = x.status.?.slice(u32), .status_dev = try x.status.?.device() });
        if (comptime @import("builtin").is_test) if (V3Testing.wire) |f| {
            try f(x, until);
            return;
        };
        x.gpu = try exchange_api.Gpu.init(x.d);
        // Reuse current native GID discovery: indices move on reboot, never hard-code one.
        var verbs = try tp.Verbs.open(null);
        defer verbs.close();
        var ports: [32]tp.roce_verbs.Port = undefined;
        const count = tp.roce_verbs.scanPorts(&verbs, &ports);
        var gid: ?u8 = null;
        for (ports[0..count]) |p| if (std.mem.eql(u8, p.spec.name[0..p.spec.name_len], x.c.verbs_device) and p.active and p.ethernet) {
            gid = p.spec.gid;
            break;
        };
        var roce = exchange_api.Roce{ .ports = &.{x.c.verbs_device}, .gid = gid orelse return error.NoHca, .traffic_class = x.c.traffic_class };
        try stopBefore(requested, until);
        const address = try std.Io.net.IpAddress.parse(x.c.master[0], x.c.ports[0]);
        if (x.c.role == .attention) {
            const server = try exchange_api.Server.open(address);
            defer server.close();
            try x.node.?.attend(server, roce.opener());
        } else try x.node.?.join(address, roce.opener());
        try stopBefore(requested, until);
    }
    fn bytes(x: *const Exchange) usize {
        return (if (x.region) |b| b.bytes.len else @as(usize, 0)) + (if (x.status) |b| b.bytes.len else @as(usize, 0)) + (if (x.gpu) |g| g.state.len else @as(usize, 0));
    }
    fn closeChecked(x: *Exchange, until: u64) Receipt {
        if (x.closed) return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
        if (x.retained) return .{ .quiet = false, .retained_bytes = x.bytes(), .reason = .failed };
        if (x.proof) |ev| {
            ev.record(x.stream) catch {
                return x.keep(.gpu);
            };
            while (true) {
                if (tp.sock.nowNs() >= until) return x.keep(.deadline);
                const rc = x.d.api.cuEventQuery(ev.handle);
                if (tp.sock.nowNs() >= until) return x.keep(.deadline);
                if (rc == cuda.abi.success) break;
                if (rc != cuda.abi.error_not_ready) return x.keep(.gpu);
                std.atomic.spinLoopHint();
            }
        }
        if (x.node) |*n| {
            n.closeChecked(until) catch {
                return x.keep(.failed);
            };
            x.node = null;
        }
        if (x.gpu) |*g| {
            g.deinit();
            x.gpu = null;
        }
        if (x.region) |*b| {
            if (!b.freeChecked()) return x.keep(.failed);
            x.region = null;
        }
        if (x.status) |*b| {
            if (!b.freeChecked()) return x.keep(.failed);
            x.status = null;
        }
        if (x.proof) |*ev| {
            ev.deinit();
            x.proof = null;
        }
        x.closed = true;
        return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
    }
    fn keep(x: *Exchange, reason: @TypeOf(@as(Receipt, undefined).reason)) Receipt {
        x.retained = true;
        return .{ .quiet = false, .retained_bytes = x.bytes(), .reason = reason };
    }
};
pub const Session = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    e: *Engine,
    exchange: Exchange,
    adapter: ?v3.Adapter = null,
    request: ?cuda.DeviceBuffer = null,
    header: ?cuda.HostBuffer = null,
    done: ?cuda.Event = null,
    ready: bool = false,
    closed: bool = false,
    retained: bool = false,
    cancel_requested: ?*const std.atomic.Value(bool) = null,
    close_receipts: [2]?Receipt = .{ null, null },
    pub fn create(a: std.mem.Allocator, e: *Engine, c: cfg.Config) !*Session {
        const s = try a.create(Session);
        s.* = .{ .gpa = a, .c = c, .e = e, .exchange = .{ .gpa = a, .d = e.d, .c = c, .stream = e.stream } };
        return s;
    }
    pub fn open(s: *Session, io: std.Io, until: u64) !void {
        try stopBefore(s.cancel_requested, until);
        const id = try s.c.identity(s.gpa);
        if (!std.mem.eql(u8, &s.e.checkpoint_digest, &id.model)) return error.BadHybridModel;
        try s.exchange.open(io, until, s.cancel_requested);
        const scope = @import("cuda_hybrid_ledger.zig").scopeOwner(.transport);
        defer scope.end();
        s.request = try cuda.DeviceBuffer.alloc(s.e.d, @intCast((exchange_api.proto.Layout{ .d = 4096, .s = 9 }).slotBytes(maxRows(s.c))));
        // One complete layer-major four-lane forward's immutable host headers.
        s.header = try cuda.HostBuffer.alloc(s.e.d, 4 * 256 * exchange_api.proto.HDR_BYTES);
        s.done = try cuda.Event.init(s.e.d, false);
        s.adapter = .{ .node = &s.exchange.node.?, .gpu = &s.exchange.gpu.?, .request = s.request.?, .header = s.header.?, .done = s.done.?, .session = id.session, .capacity = maxRows(s.c) };
        try stopBefore(s.cancel_requested, until);
        try s.e.installRouted(s.adapter.?.provider());
        s.ready = true;
    }
    fn keep(s: *Session) Receipt {
        s.retained = true;
        if (s.adapter) |*a| a.failed = error.HybridQuarantined;
        return .{ .quiet = false, .retained_bytes = s.exchange.bytes() + (if (s.request) |b| b.len else @as(usize, 0)) + (if (s.header) |b| b.bytes.len else @as(usize, 0)), .reason = .failed };
    }
    pub fn closeChecked(s: *Session) Receipt {
        if (s.closed) return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
        if (s.retained) return s.keep();
        const until = deadline(s.c.close_ms);
        if (s.adapter) |*a| {
            const p = a.provider();
            if ((p.settle(p.ctx, until) catch return s.keep()) != .quiet) return s.keep();
        }
        if (s.ready) if (s.exchange.node) |*n| n.stopExperts();
        const receipt = s.exchange.closeChecked(until);
        s.close_receipts = .{ receipt, receipt };
        if (!receipt.quiet) return s.keep();
        s.e.installRouted(null) catch return s.keep();
        if (s.adapter) |*a| a.releaseProofs();
        s.adapter = null;
        if (s.request) |*b| {
            b.free();
            s.request = null;
        }
        if (s.header) |*b| {
            if (!b.freeChecked()) return s.keep();
            s.header = null;
        }
        if (s.done) |*ev| {
            ev.deinit();
            s.done = null;
        }
        s.closed = true;
        return .{ .quiet = true, .retained_bytes = 0, .reason = .closed };
    }
    pub fn destroy(s: *Session) void {
        std.debug.assert(s.closed and !s.retained);
        const a = s.gpa;
        a.destroy(s);
    }
};

const ExpertOwner = struct {
    gpa: std.mem.Allocator,
    c: cfg.Config,
    d: *const cuda.Driver,
    stream: ?cuda.Stream = null,
    completion: ?cuda.Event = null,
    kernels: ?exl3.Kernels = null,
    plans: ?cuda.Module = null,
    weights: ?load.ExpertWeights = null,
    worker: ?v3.ExpertDevice = null,
    exchange: ?Exchange = null,
    prompt: ?v3.g53.Set = null,
    combine_module: ?cuda.Module = null,
    combine: ?cuda.Function = null,
    stopping: ?*const std.atomic.Value(bool) = null,
    lifecycle: ?*cuda.memory_receipt.Lifecycle = null,
    failed: bool = false,
    retained: bool = false,
    next: ?*ExpertOwner = null,
    fn init(o: *ExpertOwner, io: std.Io, shape: model.Config, dir: []const u8, until: u64) !void {
        try stopBefore(o.stopping, until);
        o.stream = try cuda.Stream.init(o.d, false);
        o.completion = try cuda.Event.init(o.d, false);
        const router = if (comptime @import("builtin").is_test) blk: {
            if (test_hooks.expert) |f| break :blk try f(o, io, shape, dir, until);
            break :blk try o.loadNative(io, shape, dir, until);
        } else try o.loadNative(io, shape, dir, until);
        try stopBefore(o.stopping, until);
        o.worker = try v3.ExpertDevice.init(o.d, &o.weights.?, &o.kernels.?, router, o.combine orelse return error.MissingV3Combine, o.stream.?, maxRows(o.c), if (o.prompt) |*s| s else null);
        o.worker.?.fallback.stopping = o.stopping;
        o.exchange = .{ .gpa = o.gpa, .d = o.d, .c = o.c, .stream = o.stream.? };
        try o.exchange.?.open(io, until, o.stopping);
        try stopBefore(o.stopping, until);
        if (o.lifecycle) |l| l.ready();
    }
    fn loadNative(o: *ExpertOwner, io: std.Io, shape: model.Config, dir: []const u8, until: u64) !cuda.grouped.Router {
        const control = load.Control{ .stopping = o.stopping, .until = until };
        try control.check();
        o.plans = try cuda.Module.load(o.d, cuda.kernels.experts);
        o.kernels = try exl3.Kernels.load(o.d);
        const router = try cuda.grouped.Router.resolve(o.plans.?);
        const count = if (o.c.settings.layers > 0) o.c.settings.layers else shape.layers;
        const layers = try o.gpa.alloc(u32, count);
        defer o.gpa.free(layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        var ck = try core.Checkpoint.openModelPrefix(o.gpa, io, dir, "model.language_model.");
        defer core.checkpoint.cache.close(&ck);
        o.weights = try load.loadExpertLayersControlled(o.gpa, o.d, o.stream.?, &ck, shape, try o.c.half(), layers, o.c.mtp, control);
        try control.check();
        var device: cuda.abi.Device = 0;
        try o.d.check(o.d.api.cuCtxGetDevice(&device), "V3 expert current device");
        var sms: c_int = 0;
        try o.d.check(o.d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "V3 expert SM count");
        if (sms <= 0) return error.BadExpertGeometry;
        o.prompt = try v3.g53.Set.load(o.d, cuda.kernels.g53_rank, @intCast(sms));
        o.combine_module = try cuda.Module.load(o.d, cuda.kernels.glm_glue);
        o.combine = try o.combine_module.?.function("tf_glm_combine");
        return router;
    }

    /// Device TAKE and native weighted producer execute on one stream. No host payload copy,
    /// per-layer synchronization or reply-wide head contraction is introduced.
    pub fn turn(o: *ExpertOwner, h: exchange_api.proto.Header, sequence: u32) !void {
        const w = &o.worker.?;
        const x = &o.exchange.?;
        const n = &x.node.?;
        const s = o.stream.?;
        if (h.fseq <= 0 or @as(u32, @intCast(h.fseq)) != sequence or h.rows <= 0 or @as(u32, @intCast(h.rows)) > w.capacity or h.layers != 1 or (h.mode != .eager and h.mode != .chunk)) return error.BadHybridCall;
        try stopBefore(o.stopping, deadline(o.c.transaction_ms));
        try n.check();
        const bytes = exchange_api.proto.wireBytes(n.cfg.settings.layout.req(@intCast(h.rows)));
        try x.gpu.?.take(n, &.{.{ .ptr = w.landing.ptr, .bytes = bytes }}, s);
        try w.compute(h);
        try x.gpu.?.post(n, w.output.ptr, n.cfg.settings.layout.rep(@intCast(h.rows), false), s);
        if (o.completion) |ev| try ev.record(s);
        try n.check();
    }
    fn loop(o: *ExpertOwner, stop: *const std.atomic.Value(bool)) !void {
        var sequence: u32 = 0;
        const n = &o.exchange.?.node.?;
        while (!stop.load(.acquire)) {
            const next = try std.math.add(u32, sequence, 1);
            switch (n.waitRequestControlled(next, stop)) {
                .stopped => return,
                .lost => |why| {
                    std.log.err("V3 expert request lost: {s}", .{why});
                    return error.ExpertFailed;
                },
                .request => |h| {
                    try o.turn(h, next);
                    sequence = next;
                },
            }
        }
    }
    pub fn close(o: *ExpertOwner) bool {
        if (o.lifecycle) |l| l.cleanup();
        if (o.retained) return false;
        const until = deadline(o.c.close_ms);
        if (o.failed or (o.worker != null and o.worker.?.failed != null)) {
            o.retained = true;
            return false;
        }
        if (o.worker) |*w| if (!(w.settle(until) catch {
            o.retained = true;
            return false;
        })) {
            o.retained = true;
            return false;
        };
        if (o.completion) |ev| {
            ev.record(o.stream.?) catch {
                o.retained = true;
                return false;
            };
            while (true) {
                if (tp.sock.nowNs() >= until) {
                    o.retained = true;
                    return false;
                }
                const rc = o.d.api.cuEventQuery(ev.handle);
                if (tp.sock.nowNs() >= until) {
                    o.retained = true;
                    return false;
                }
                if (rc == cuda.abi.success) break;
                if (rc != cuda.abi.error_not_ready) {
                    o.retained = true;
                    return false;
                }
                std.atomic.spinLoopHint();
            }
            while (!(cuda.copy_source.streamDone(o.d, o.stream.?.handle) catch {
                o.retained = true;
                return false;
            })) {
                if (tp.sock.nowNs() >= until) {
                    o.retained = true;
                    return false;
                }
                std.atomic.spinLoopHint();
            }
            if (tp.sock.nowNs() >= until) {
                o.retained = true;
                return false;
            }
        }
        if (o.exchange) |*x| if (!x.closeChecked(until).quiet) {
            o.retained = true;
            return false;
        };
        if (o.worker) |*w| w.releaseStorage();
        o.worker = null;
        if (o.weights) |*w| w.deinit();
        o.weights = null;
        if (cuda.memory_owner.snapshot()) |owned| if (owned.sticky_failure or owned.live_device != 0 or owned.live_pinned != 0 or owned.retained_device != 0 or owned.retained_pinned != 0) {
            o.retained = true;
            return false;
        };
        if (o.prompt) |*s| s.unload();
        if (o.combine_module) |*m| m.unload();
        if (o.kernels) |*k| k.unload();
        if (o.plans) |*m| m.unload();
        if (o.completion) |*ev| ev.deinit();
        if (o.stream) |s| o.d.check(o.d.api.cuStreamDestroy_v2(s.handle), "V3 expert quiet stream destroy") catch {
            o.retained = true;
            return false;
        };
        return true;
    }
};

test "I3 native Peer startup classifies the constructor device state as transport" {
    const t = std.testing;
    const C = @import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const X = struct {
        var fds: [2]std.c.fd_t = undefined;
        fn event(p: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
            p.* = null;
            return 0;
        }
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
        fn boot(_: cfg.Config, _: u1, rank: u32, until: u64) !tp.Bootstrap {
            var b = tp.Bootstrap{ .rank = rank, .world = 2, .timeout_ns = std.time.ns_per_s, .absolute_deadline = until };
            b.fds[1 - rank] = fds[rank];
            return b;
        }
        fn other(contract: *const [256]u8, until: u64, err: *?anyerror) void {
            var b = boot(undefined, 0, 1, until) catch unreachable;
            defer b.close();
            b.agree(contract, "native hybrid WIRE3 contract") catch |e| {
                err.* = e;
                return;
            };
            b.agree("ready", "native hybrid frame storage ready") catch |e| {
                err.* = e;
            };
        }
        fn roce(d: ?*const cuda.Driver, _: *tp.Bootstrap, _: []const u8, _: tp.roce.Options, custody: *?*tp.roce.Roce) !*tp.roce.Roce {
            const R = tp.roce.Roce;
            const n = R.regionBytes(tr.wire.post_bytes);
            const mem = try std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            @memset(mem, 0);
            const r = try std.heap.page_allocator.create(R);
            r.* = .{ .me = 0, .slot_bytes = tr.wire.post_bytes, .timeout_ns = std.time.ns_per_s, .region = mem, .mapping = mem, .cpu = null, .d = d, .wire = .{ .shm = .{ .local = mem, .peer = mem, .layout = .{ .send = 8192, .recv = 8192 + 2 * tr.wire.post_bytes, .flags = 4096, .slot = tr.wire.post_bytes, .len = n } } }, .state = try cuda.DeviceBuffer.alloc(d.?, 16) };
            custody.* = r;
            return r;
        }
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "roce.fatbin", .data = "x" });
    const image = try tmp.dir.realPathFileAlloc(t.io, "roce.fatbin", t.allocator);
    defer t.allocator.free(image);
    const c = @import("cuda_hybrid_owner_fixture.zig").config(image);
    const id = try c.identity(t.allocator);
    var h = tr.wire.Header{ .session = id.session, .sequence = 1, .half = 0, .live_rows = 1, .call_rows = 1, .model_contract_sha256 = id.model, .settings_sha256 = id.settings };
    h.bind();
    var contract: [256]u8 = undefined;
    h.encode(&contract);
    try t.expectEqual(@as(c_int, 0), std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &X.fds));
    const until = deadline(1000);
    var peer_error: ?anyerror = null;
    const thread = try std.Thread.spawn(.{}, X.other, .{ &contract, until, &peer_error });
    defer thread.join();
    var l = @import("cuda_hybrid_ledger.zig").Ledger.init(.attention, 929);
    try l.install();
    defer l.uninstall();
    var d: cuda.Driver = undefined;
    const e = try C.engine(&d);
    defer e.deinit();
    d.api.cuEventCreate = X.event;
    d.api.cuEventRecord = X.record;
    d.api.cuEventQuery = X.query;
    test_hooks = .{ .boot = X.boot, .roce = X.roce };
    defer test_hooks = .{};
    var p = Peer{ .d = &d, .stream = e.stream };
    try p.start(t.allocator, t.io, c, 0, 0, until);
    const device = p.frame.?.frames.len + 16;
    const status = l.status();
    const receipt = p.closeChecked(until);
    l.finish(receipt.quiet, false);
    try t.expect(receipt.quiet and l.status().clean);
    try t.expectEqual(device, status.owners[@backingInt(@import("cuda_hybrid_memory.zig").Owner.transport)].device);
    try t.expectEqual(@as(u64, 0), status.unclassified_buffers);
}

test "I3 native expert already cancelled startup dispatches no source or CUDA" {
    const t = std.testing;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("/fixture/unopened");
    c.role = .expert0;
    var ctx: cuda.Context = undefined;
    var stopping = std.atomic.Value(bool).init(true);
    try t.expectError(error.Cancelled, runExpert(t.allocator, t.io, &ctx, "/fixture/NEVER_OPENED", c, .{}, &stopping, deadline(1000), .{ .room = std.math.maxInt(u64), .unified = true }));
}

test "I3 native expert admission refuses missing foreign closed or sticky ownership before source and CUDA" {
    const t = std.testing;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("/fixture/unopened");
    c.role = .expert1;
    var ctx: cuda.Context = undefined;
    var stop = std.atomic.Value(bool).init(false);
    try t.expectError(error.HybridMemoryRefused, runExpert(t.allocator, t.io, &ctx, "/fixture/NEVER_OPENED", c, .{}, &stop, deadline(1000), .{ .room = std.math.maxInt(u64), .unified = true }));
    for (0..4) |which| {
        var l = @import("cuda_hybrid_ledger.zig").Ledger.init(if (which == 0) .attention else .expert1, if (which == 1) 2 else 1);
        l.closed = which == 2;
        l.sticky_failure = which == 3;
        try l.install();
        defer l.uninstall();
        try t.expectError(error.HybridMemoryRefused, runExpert(t.allocator, t.io, &ctx, "/fixture/NEVER_OPENED", c, .{}, &stop, deadline(1000), .{ .room = std.math.maxInt(u64), .unified = true }));
    }
}

test "I3 native expert cancellation during load admits no later workspace or peer" {
    const t = std.testing;
    const F = @import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const X = struct {
        var flag: ?*std.atomic.Value(bool) = null;
        var boots: usize = 0;
        fn event(p: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
            p.* = null;
            return 0;
        }
        fn stream(p: *cuda.abi.Stream, _: c_uint) callconv(.c) c_int {
            p.* = null;
            return 0;
        }
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
        fn ready(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn boot(_: cfg.Config, _: u1, _: u32, _: u64) !tp.Bootstrap {
            boots += 1;
            return error.CpuUnexpectedBootstrap;
        }
        fn loadFixture(own: *ExpertOwner, _: std.Io, shape: model.Config, _: []const u8, _: u64) !cuda.grouped.Router {
            own.kernels = undefined;
            own.kernels.?.mod = .{ .d = own.d, .handle = null };
            own.weights = .{ .gpa = own.gpa, .config = shape, .half = 0, .cut = .{ .lo = 0, .hi = 1024 } };
            flag.?.store(true, .release);
            return undefined;
        }
    };
    var stopping = std.atomic.Value(bool).init(false);
    X.flag = &stopping;
    X.boots = 0;
    defer X.flag = null;
    var l = @import("cuda_hybrid_ledger.zig").Ledger.init(.expert0, 930);
    try l.install();
    defer l.uninstall();
    var d: cuda.Driver = undefined;
    const e = try F.engine(&d);
    defer e.deinit();
    d.api.cuEventCreate = X.event;
    d.api.cuStreamCreate = X.stream;
    d.api.cuEventRecord = X.record;
    d.api.cuEventQuery = X.query;
    d.api.cuStreamQuery = X.ready;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("/fixture/unopened");
    c.role = .expert0;
    var o = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d, .stopping = &stopping };
    test_hooks = .{ .boot = X.boot, .expert = X.loadFixture };
    defer test_hooks = .{};
    const got = o.init(t.io, .{ .exl3 = true }, "/fixture/unopened", deadline(1000));
    const status = l.status();
    const quiet = o.close();
    l.finish(quiet, true);
    try t.expectError(error.Cancelled, got);
    try t.expectEqual(@as(usize, 0), X.boots);
    try t.expectEqual(@as(u64, 0), status.owners[@backingInt(@import("cuda_hybrid_memory.zig").Owner.workspace)].device);
    try t.expect(quiet and l.status().clean and l.status().cancelled);
}

test "I3 native expert startup classifies worker and pinned transport before bootstrap" {
    const t = std.testing;
    const F = @import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const X = struct {
        fn event(p: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
            p.* = null;
            return 0;
        }
        fn stream(p: *cuda.abi.Stream, _: c_uint) callconv(.c) c_int {
            p.* = null;
            return 0;
        }
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
        fn ready(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn boot(_: cfg.Config, _: u1, _: u32, _: u64) !tp.Bootstrap {
            return error.CpuBeforeBootstrap;
        }
    };
    var l = @import("cuda_hybrid_ledger.zig").Ledger.init(.expert0, 924);
    try l.install();
    defer l.uninstall();
    var d: cuda.Driver = undefined;
    const e = try F.engine(&d);
    defer e.deinit();
    d.api.cuEventCreate = X.event;
    d.api.cuStreamCreate = X.stream;
    d.api.cuEventRecord = X.record;
    d.api.cuEventQuery = X.query;
    d.api.cuStreamQuery = X.ready;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("/fixture/unopened");
    c.role = .expert0;
    var o = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d };
    const T = struct {
        fn loadFixture(own: *ExpertOwner, _: std.Io, shape: model.Config, _: []const u8, _: u64) !cuda.grouped.Router {
            own.kernels = undefined;
            own.kernels.?.mod = .{ .d = own.d, .handle = null };
            own.weights = .{ .gpa = own.gpa, .config = shape, .half = 0, .cut = .{ .lo = 0, .hi = 1024 } };
            return undefined;
        }
    };
    test_hooks = .{ .boot = X.boot, .expert = T.loadFixture };
    defer test_hooks = .{};
    const got = o.init(t.io, .{ .exl3 = true }, "/fixture/unopened", deadline(1000));
    const status = l.status();
    const quiet = o.close();
    l.finish(quiet, false);
    try t.expectError(error.CpuBeforeBootstrap, got);
    try t.expect(quiet and l.status().clean);
    const O = @import("cuda_hybrid_memory.zig").Owner;
    try t.expect(status.owners[@backingInt(O.workspace)].device > 0);
    try t.expect(@hasDecl(expert.Worker, "storageBytes"));
    if (comptime @hasDecl(expert.Worker, "storageBytes")) try t.expectEqual(@as(u64, expert.Worker.storageBytes()), status.owners[@backingInt(O.workspace)].device);
    try t.expectEqual(@as(u64, tr.wire.request_bytes + tr.wire.reply_bytes), status.owners[@backingInt(O.transport)].pinned);
    try t.expectEqual(@as(u64, 0), status.unclassified_buffers);
}

test "I3 native expert close retains CUDA parents after an actual failed buffer release" {
    const t = std.testing;
    const F = @import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const X = struct {
        var drops: usize = 0;
        fn drop(_: cuda.abi.Module) callconv(.c) c_int {
            drops += 1;
            return 0;
        }
        fn stream(_: cuda.abi.Stream) callconv(.c) c_int {
            drops += 1;
            return 0;
        }
    };
    var l = @import("cuda_hybrid_ledger.zig").Ledger.init(.expert0, 923);
    try l.install();
    defer l.uninstall();
    var d: cuda.Driver = undefined;
    const e = try F.engine(&d);
    defer e.deinit();
    d.api.cuModuleUnload = X.drop;
    d.api.cuStreamDestroy_v2 = X.stream;
    X.drops = 0;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("/fixture/unopened");
    c.role = .expert0;
    var o = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d, .stream = .{ .d = &d, .handle = null }, .plans = .{ .d = &d, .handle = null }, .input = try cuda.HostBuffer.alloc(&d, 16) };
    F.fail_free = @intFromPtr(o.input.?.bytes.ptr);
    const quiet = o.close();
    const retained = l.status();
    F.fail_free = 0;
    o.input.?.free();
    o.input = null;
    l.finish(true, false);
    try t.expect(!quiet and o.retained);
    try t.expectEqual(@as(u64, 16), retained.retained.pinned);
    try t.expectEqual(@as(usize, 0), X.drops);
}

test "hybrid owner: partial expert startup cannot release a stream whose completion is unproven" {
    const t = std.testing;
    const F = struct {
        var releases: usize = 0;
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            return cuda.abi.error_not_ready;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn destroy(_: cuda.abi.Stream) callconv(.c) c_int {
            releases += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuStreamSynchronize = F.sync;
    d.api.cuEventRecord = F.record;
    d.api.cuEventQuery = F.query;
    d.api.cuStreamDestroy_v2 = F.destroy;
    F.releases = 0;
    const c = cfg.Config{ .revision = 3, .role = .expert0, .session = "00000000000000000000000000000001", .model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", .master = .{ "127.0.0.1", "127.0.0.1" }, .ports = .{ 29550, 29551 }, .traffic_class = 104, .verbs_device = "fixture", .roce_image = "fixture", .setup_ms = 30000, .transaction_ms = 120000, .close_ms = 1, .served_context = 262144, .resident_users = 8, .mtp = false, .settings = .{} };
    var own = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d, .stream = .{ .d = &d, .handle = null } };
    if (comptime @hasField(ExpertOwner, "completion")) own.completion = .{ .d = &d, .handle = null };
    try t.expect(!own.close());
    try t.expect(own.retained);
    try t.expectEqual(@as(usize, 0), F.releases);
}

test "hybrid owner: late successful expert completion cannot extend the enclosing close bound" {
    const t = std.testing;
    const F = struct {
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            tp.sock.sleepNs(5 * std.time.ns_per_ms);
            return 0;
        }
        fn ready(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn drop(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuEventRecord = F.record;
    d.api.cuEventQuery = F.query;
    d.api.cuStreamQuery = F.ready;
    d.api.cuStreamDestroy_v2 = F.ready;
    d.api.cuEventDestroy_v2 = F.drop;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("fixture");
    c.close_ms = 1;
    var own = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d, .stream = .{ .d = &d, .handle = null }, .completion = .{ .d = &d, .handle = null } };
    try t.expect(!own.close());
    try t.expect(own.retained);
}

test "hybrid owner: late stream query cannot turn expert close into success" {
    const t = std.testing;
    const F = struct {
        fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }
        fn query(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
        fn ready(_: cuda.abi.Stream) callconv(.c) c_int {
            tp.sock.sleepNs(5 * std.time.ns_per_ms);
            return 0;
        }
        fn drop(_: cuda.abi.Event) callconv(.c) c_int {
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuEventRecord = F.record;
    d.api.cuEventQuery = F.query;
    d.api.cuStreamQuery = F.ready;
    d.api.cuStreamDestroy_v2 = F.ready;
    d.api.cuEventDestroy_v2 = F.drop;
    var c = @import("cuda_hybrid_owner_fixture.zig").config("fixture");
    c.close_ms = 1;
    var own = ExpertOwner{ .gpa = t.allocator, .c = c, .d = &d, .stream = .{ .d = &d, .handle = null }, .completion = .{ .d = &d, .handle = null } };
    try t.expect(!own.close());
    try t.expect(own.retained);
}

test "hybrid owner: actual attention Session and both ExpertOwners compose startup to acknowledged close" {
    try composedOwnerTrial(false);
}
test "I3 native expert load and typed close emit four ordered lifecycle phases" {
    try composedOwnerTrial(true);
}
fn composedOwnerTrial(with_receipts: bool) !void {
    const t = std.testing;
    const F = @import("cuda_hybrid_owner_fixture.zig");
    try F.prepare();
    const P = struct {
        var phases: [5]cuda.memory_receipt.Phase = undefined;
        var count: usize = 0;
        var cleanup_blocks: usize = 0;
        fn sink(r: cuda.memory_receipt.Receipt) void {
            phases[count] = r.phase;
            count += 1;
            if (r.phase == .cleanup_begin) cleanup_blocks = F.blocks.load(.acquire);
        }
    };
    P.count = 0;
    P.cleanup_blocks = 0;
    var lifecycle = cuda.memory_receipt.Lifecycle{ .rank = 0 };
    if (with_receipts) {
        cuda.memory_receipt.Test.sink = P.sink;
        cuda.memory_receipt.emit(0, .load_begin, false);
    }
    defer if (with_receipts) {
        cuda.memory_receipt.Test.sink = null;
    };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "roce.fatbin", .data = "x" });
    const image = try tmp.dir.realPathFileAlloc(t.io, "roce.fatbin", t.allocator);
    defer t.allocator.free(image);
    test_hooks = .{ .boot = F.bootstrap, .roce = F.roce, .expert = F.expert(ExpertOwner) };
    defer test_hooks = .{};
    const c = F.config(image);
    const identity = try c.identity(t.allocator);
    var e: Engine = undefined;
    e.d = &F.d;
    e.stream = try cuda.Stream.init(&F.d, false);
    defer _ = F.d.api.cuStreamDestroy_v2(e.stream.handle);
    e.checkpoint_digest = identity.model;
    e.w.role = .attention;
    e.bs = null;
    e.pair = null;
    e.l2pf = null;
    e.routed = null;
    e.pieces.side = e.stream;
    var hs: [3]@import("cuda_routed.zig").Buffer = undefined;
    for (&hs) |*h| h.* = .{ .capacity = 32, .dims = 4096, .halves = 1, .epilogue = 2, .weights = 3, .shared_weights = 4 };
    e.bp.moe.hybrid = &hs[0];
    e.bd.moe.hybrid = &hs[1];
    e.bm.moe.hybrid = &hs[2];
    const WorkerThread = struct {
        fn run(o: *ExpertOwner, err: *?anyerror, stop: *const std.atomic.Value(bool), until: u64) void {
            o.init(t.io, .{ .exl3 = true }, "/unopened-fixture", until) catch |why| {
                err.* = why;
                _ = o.close();
                return;
            };
            o.loop(stop) catch |why| {
                err.* = why;
                o.failed = true;
                _ = o.close();
                return;
            };
            if (!o.close()) err.* = error.HybridRetained;
        }
    };
    var owners: [2]ExpertOwner = undefined;
    var errors = [_]?anyerror{ null, null };
    var stop = std.atomic.Value(bool).init(false);
    const until = deadline(c.setup_ms);
    for (&owners, 0..) |*own, half| {
        var selected = c;
        selected.role = if (half == 0) .expert0 else .expert1;
        own.* = .{ .gpa = t.allocator, .c = selected, .d = &F.d, .lifecycle = if (with_receipts and half == 0) &lifecycle else null };
    }
    const thread0 = try std.Thread.spawn(.{}, WorkerThread.run, .{ &owners[0], &errors[0], &stop, until });
    const thread1 = try std.Thread.spawn(.{}, WorkerThread.run, .{ &owners[1], &errors[1], &stop, until });
    const session = try Session.create(t.allocator, &e, c);
    try session.open(t.io, until);
    try t.expect(session.ready and e.routed != null);
    for (&hs) |h| try t.expect(h.provider != null);
    const receipt = session.closeChecked();
    thread0.join();
    thread1.join();
    try t.expect(receipt.quiet);
    session.destroy();
    for (errors) |err| if (err) |why| return why;
    try t.expect(e.routed == null);
    for (&hs) |h| try t.expect(h.provider == null);
    try t.expectEqual(@as(usize, 0), F.blocks.load(.acquire));
    try t.expectEqual(@as(usize, 0), F.events.load(.acquire));
    try t.expectEqual(@as(usize, 0), F.unbounded.load(.acquire));
    if (with_receipts) {
        lifecycle.cleanup();
        cuda.memory_receipt.emit(0, .cleanup_end, false);
        try t.expectEqualSlices(cuda.memory_receipt.Phase, &.{ .load_begin, .load_end, .cleanup_begin, .cleanup_end }, P.phases[0..P.count]);
        try t.expect(P.cleanup_blocks > 0);
    }
}

var retained_experts: ?*ExpertOwner = null;
var retained_mutex: std.atomic.Mutex = .unlocked;
fn keepExpert(o: *ExpertOwner) void {
    while (!retained_mutex.tryLock()) std.Thread.yield() catch {};
    defer retained_mutex.unlock();
    o.next = retained_experts;
    retained_experts = o;
}
fn expertAdmitted(a: std.mem.Allocator, c: cfg.Config) !void {
    const identity = try c.identity(a);
    const owned = cuda.memory_owner.snapshot() orelse return error.HybridMemoryRefused;
    if (!std.mem.eql(u8, owned.role, @tagName(c.role)) or owned.incarnation != identity.session or owned.closed or owned.sticky_failure) return error.HybridMemoryRefused;
}
pub const Admission = struct { room: u64, unified: bool, lifecycle: ?*cuda.memory_receipt.Lifecycle = null };
pub fn runExpert(a: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, c: cfg.Config, shape: model.Config, stop: *const std.atomic.Value(bool), until: u64, admission: Admission) !void {
    _ = try c.half();
    if (stop.load(.acquire)) return error.Cancelled;
    try expertAdmitted(a, c);
    const memory_plan = try @import("cuda_hybrid_preload.zig").read(a, io, dir, c);
    // Compare the selected role's known owners with the SAME native pool/reserve/cap;
    // explicit private-driver/verbs/checkpoint heap unknowns remain unknown, not zero.
    const extra = expertExtraBytesFor(c);
    const known_peak = try std.math.add(u64, memory_plan.load_peak, try std.math.add(u64, extra.device, if (admission.unified) extra.host else 0));
    if (known_peak > admission.room) return error.HybridMemoryBudget;
    const o = try a.create(ExpertOwner);
    o.* = .{ .gpa = a, .c = c, .d = ctx.d, .stopping = stop, .lifecycle = admission.lifecycle };
    o.init(io, shape, dir, until) catch |err| {
        if (o.close()) a.destroy(o) else {
            keepExpert(o);
            return error.HybridRetained;
        }
        return err;
    };
    o.loop(stop) catch |err| {
        o.failed = true;
        _ = o.close();
        keepExpert(o);
        std.log.err("hybrid expert failed; retained ownership: {s}", .{@errorName(err)});
        return error.HybridRetained;
    };
    if (!o.close()) {
        keepExpert(o);
        return error.HybridRetained;
    }
    a.destroy(o);
}

// AI-assisted integration: real Peer/Session entry cancellation before an event or bootstrap admission.
pub const StopTest = if (@import("builtin").is_test) struct {
    pub fn peerEntry(session_case: bool) !void {
        const t = std.testing;
        const P = struct {
            var events: usize = 0;
            fn event(_: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
                events += 1;
                return cuda.abi.error_not_ready;
            }
        };
        P.events = 0;
        var d: cuda.Driver = undefined;
        d.api.cuEventCreate = P.event;
        var requested = std.atomic.Value(bool).init(true);
        const c = @import("cuda_hybrid_owner_fixture.zig").config("/CPU/NEVER_OPENED");
        const until = tp.sock.nowNs() + std.time.ns_per_s;
        if (session_case) {
            var e: Engine = undefined;
            @memset(std.mem.asBytes(&e), 0);
            e.d = &d;
            e.stream = .{ .d = &d, .handle = null };
            e.checkpoint_digest = @splat(0xAA);
            var session = Session{ .gpa = t.allocator, .c = c, .e = &e, .peers = .{ .{ .d = &d, .stream = e.stream }, .{ .d = &d, .stream = e.stream } } };
            defer {
                const receipt = session.closeChecked();
                std.debug.assert(receipt.quiet);
            }
            if (comptime @hasField(Session, "cancel_requested")) session.cancel_requested = &requested;
            try t.expectError(error.Cancelled, session.open(t.io, until));
        } else {
            var peer = Peer{ .d = &d, .stream = .{ .d = &d, .handle = null } };
            defer {
                const receipt = peer.closeChecked(until);
                std.debug.assert(receipt.quiet);
            }
            if (comptime @hasField(Peer, "cancel_requested")) peer.cancel_requested = &requested;
            try t.expectError(error.Cancelled, peer.start(t.allocator, t.io, c, 0, 0, until));
        }
        try t.expectEqual(@as(usize, 0), P.events);
    }
} else struct {};

fn stopBefore(requested: ?*const std.atomic.Value(bool), until: u64) !void {
    if (requested) |flag| if (flag.load(.acquire)) return error.Cancelled;
    try before(until);
}
