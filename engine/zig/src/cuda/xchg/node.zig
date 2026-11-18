// Modified for V4 by Scott Quarrier with Helios: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! One node of the three-node exchange (qualification_12f4270a): the attention rank (both experts' pairs over one region) or an expert rank (one pair), the control-link handshake, the attention's watcher, the experts' heartbeat, and the status a request checks.
//! Each pair is roce_proxy.c (Mia's patch 0006, unchanged) behind `Proxy`, or a test's stand-in; the GPU half is fn_xchg.cu (`Gpu`), or its CPU build in tests and tf-xchg's no-GPU mode.
const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Allocator = std.mem.Allocator;
const proto = @import("proto.zig");
const watcher = @import("watcher.zig");
const link = @import("../link.zig");
const watch = @import("../watch.zig");
const fingerprint = @import("../fingerprint.zig");

const log = std.log.scoped(.xchg);

// the control link's frames (link.zig's tag set is open)
pub const tag_hello: link.Tag = @enumFromInt(101);
pub const tag_ack: link.Tag = @enumFromInt(102);
pub const tag_blob: link.Tag = @enumFromInt(103);
pub const tag_ready: link.Tag = @enumFromInt(104);
pub const tag_beat: link.Tag = @enumFromInt(105);
pub const tag_stop: link.Tag = @enumFromInt(106);

const rdhup: i16 = 0x2000; // POLLRDHUP: the peer closed its end

/// A pair's transport: it posts this node's send slot to the peer each time the node's doorbell rings (roce_proxy.c's contract).
pub const Proxy = struct {
    ctx: *anyopaque,
    vt: *const VTable,

    pub const VTable = struct {
        blob: *const fn (ctx: *anyopaque, out: []u8) anyerror!usize,
        connect: *const fn (ctx: *anyopaque, peer: []const u8) anyerror!void,
        start: *const fn (ctx: *anyopaque) anyerror!void,
        failure: *const fn (ctx: *anyopaque) ?[]const u8,
        ops: *const fn (ctx: *anyopaque) u64,
        close: *const fn (ctx: *anyopaque) void,
        /// Helios: typed CQ/thread/accepted-doorbell proof; null cannot release a live runtime owner.
        quiesce: ?*const fn (ctx: *anyopaque, until: u64) anyerror!void = null,
        /// qualification_17c70e37: the settings' stripes, before the start (null: a proxy without stripes)
        stripes: ?*const fn (ctx: *anyopaque, k: u32, min: u64) anyerror!void = null,
    };

    pub fn blob(p: Proxy, out: []u8) !usize {
        return p.vt.blob(p.ctx, out);
    }
    pub fn connect(p: Proxy, peer: []const u8) !void {
        return p.vt.connect(p.ctx, peer);
    }
    pub fn start(p: Proxy) !void {
        return p.vt.start(p.ctx);
    }
    pub fn setStripes(p: Proxy, k: u32, min: u64) !void {
        const f = p.vt.stripes orelse return error.NoStripes;
        return f(p.ctx, k, min);
    }
    pub fn failure(p: Proxy) ?[]const u8 {
        return p.vt.failure(p.ctx);
    }
    pub fn ops(p: Proxy) u64 {
        return p.vt.ops(p.ctx);
    }
    pub fn close(p: Proxy) void {
        p.vt.close(p.ctx);
    }
};

/// How a node opens a pair's proxy: over `region` as rank `rank` of the pair, in geometry `reg` (its slots' bytes, and
/// qualification_9e49592e: how many a source).
pub const Opener = struct {
    ctx: ?*anyopaque = null,
    open: *const fn (ctx: ?*anyopaque, rank: u32, region: []u8, reg: proto.Region) anyerror!Proxy,
};

const RoceC = opaque {};
extern fn tf_roce_blob_bytes() u64;
extern fn tf_roce_open_slots(world: c_int, rank: c_int, names: [*]const [*:0]const u8, ports: c_int, gid_index: c_int, region: *anyopaque, region_bytes: u64, slot_bytes: u64, slots: u32, err: [*]u8, err_len: u64) ?*RoceC;
extern fn tf_roce_blob_of(r: *RoceC, out: *anyopaque) c_int;
extern fn tf_roce_connect(r: *RoceC, peer: *const anyopaque) c_int;
extern fn tf_roce_start(r: *RoceC) c_int;
extern fn tf_roce_set_stripes(r: *RoceC, stripes: u32, stripe_min: u64) c_int;
extern fn tf_roce_set_traffic_class(r: *RoceC, traffic_class: u8) c_int;
extern fn tf_roce_failed(r: *RoceC) c_int;
extern fn tf_roce_error(r: *RoceC) [*:0]const u8;
extern fn tf_roce_ops(r: *RoceC) u64;
extern fn tf_roce_quiesce(r: *RoceC, until: u64) c_int;
extern fn tf_roce_close(r: *RoceC) void;

/// roce_proxy.c over the named RDMA devices (one a port, the peer's in the same order) with one GID index; two instances may share a region.
pub const Roce = struct {
    ports: []const []const u8,
    gid: c_int,
    traffic_class: u8 = 0,

    pub fn opener(r: *Roce) Opener {
        return .{ .ctx = r, .open = open };
    }

    fn open(ctx: ?*anyopaque, rank: u32, region: []u8, reg: proto.Region) anyerror!Proxy {
        const self: *Roce = @ptrCast(@alignCast(ctx.?));
        if (self.ports.len < 1 or self.ports.len > proto.PORTS_MAX) return error.Invalid;
        var names: [proto.PORTS_MAX][256:0]u8 = undefined;
        var ptrs: [proto.PORTS_MAX][*:0]const u8 = undefined;
        for (self.ports, 0..) |p, i| {
            if (p.len >= 256) return error.Invalid;
            @memcpy(names[i][0..p.len], p);
            names[i][p.len] = 0;
            ptrs[i] = &names[i];
        }
        var err: [512]u8 = @splat(0);
        const r = tf_roce_open_slots(2, @intCast(rank), &ptrs, @intCast(self.ports.len), self.gid, region.ptr, region.len, reg.slot, reg.slots, &err, err.len) orelse {
            log.err("RoCE: {s}", .{std.mem.sliceTo(&err, 0)});
            return error.RoceUnavailable;
        };
        if(tf_roce_set_traffic_class(r,self.traffic_class)!=0){tf_roce_close(r);return error.RoceUnavailable;}
        return .{ .ctx = r, .vt = &vt };
    }

    const vt: Proxy.VTable = .{ .blob = blobOf, .connect = connectTo, .start = startIt, .failure = failureOf, .ops = opsOf, .close = closeIt, .stripes = stripesOf, .quiesce = quiesceIt };

    fn quiesceIt(ctx: *anyopaque, until: u64) anyerror!void {
        return switch (tf_roce_quiesce(@ptrCast(ctx), until)) {
            0 => {}, 2 => error.Timeout, else => error.HybridQuarantined,
        };
    }

    fn stripesOf(ctx: *anyopaque, k: u32, min: u64) anyerror!void {
        if (tf_roce_set_stripes(@ptrCast(ctx), k, min) != 0) {
            log.err("RoCE stripes: {s}", .{std.mem.span(tf_roce_error(@ptrCast(ctx)))});
            return error.RoceUnavailable;
        }
    }

    fn blobOf(ctx: *anyopaque, out: []u8) anyerror!usize {
        const n: usize = @intCast(tf_roce_blob_bytes());
        if (out.len < n or tf_roce_blob_of(@ptrCast(ctx), out.ptr) != 0) return error.Invalid;
        return n;
    }
    fn connectTo(ctx: *anyopaque, peer: []const u8) anyerror!void {
        if (peer.len != tf_roce_blob_bytes()) return error.BadRoceBlob;
        if (tf_roce_connect(@ptrCast(ctx), peer.ptr) != 0) {
            log.err("RoCE connect: {s}", .{std.mem.span(tf_roce_error(@ptrCast(ctx)))});
            return error.RoceUnavailable;
        }
    }
    fn startIt(ctx: *anyopaque) anyerror!void {
        if (tf_roce_start(@ptrCast(ctx)) != 0) {
            log.err("RoCE start: {s}", .{std.mem.span(tf_roce_error(@ptrCast(ctx)))});
            return error.RoceUnavailable;
        }
    }
    fn failureOf(ctx: *anyopaque) ?[]const u8 {
        if (tf_roce_failed(@ptrCast(ctx)) == 0) return null;
        return std.mem.span(tf_roce_error(@ptrCast(ctx)));
    }
    fn opsOf(ctx: *anyopaque) u64 {
        return tf_roce_ops(@ptrCast(ctx));
    }
    fn closeIt(ctx: *anyopaque) void {
        tf_roce_close(@ptrCast(ctx));
    }
};

pub const Role = union(enum) { attn, expert: u32 };

pub const Config = struct {
    settings: proto.Settings,
    rules: watcher.Rules = .{},
    /// a GPU wait gives up after this long (counted; the request fails): the backstop behind the watcher's `silent_ns`
    give_up_ns: u64 = 20 * std.time.ns_per_s,
    /// an expert's heartbeat period
    beat_ns: u64 = std.time.ns_per_s,
    /// how long the handshake waits for the other nodes
    setup_timeout_ms: u32 = 120_000,
    /// the attention's watcher; off only to test the GPU's give-up alone
    run_watcher: bool = true,
    /// failures log as warnings (tests that fail an expert on purpose), else as errors
    warn_failures: bool = false,
    /// rank_fingerprint: this node's identity line (fingerprint.Identity.line), required with `settings.fp`; unused without it
    identity: ?[]const u8 = null,
};

/// The host memory a node runs on: the region every pair's proxy registers, and the status words; each with the address kernels use.
pub const Memory = struct { region: []u8, region_dev: u64, status: []u32, status_dev: u64 };

/// What failed, from the status words.
pub const Failure = struct { why: proto.Why, peer: u32, seq: u32, gave_up: u32, breach: u32 };

/// What an expert's wait for its next request ends with.
pub const Wait = union(enum) { request: proto.Header, stopped, lost: []const u8 };

pub const Node = struct {
    gpa: Allocator,
    io: std.Io,
    role: Role,
    cfg: Config,
    reg: proto.Region,
    mem: Memory,
    proxies: [proto.EXPERTS]?Proxy = .{ null, null },
    links: [proto.EXPERTS]?link.Link = .{ null, null },
    beats: [proto.EXPERTS]std.atomic.Value(u64) = .{ .init(0), .init(0) },
    stopping: std.atomic.Value(bool) = .init(false),
    /// an expert's heartbeat could not be sent: the attention is gone
    beat_lost: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    rule: watcher.Rule = .{},
    /// rank_fingerprint (settings.fp): what this node concluded from the three identity lines (null without the switch)
    fp_verdict: ?fingerprint.Verdict = null,

    /// `mem.region` must hold `cfg.settings.region().total()` bytes and `mem.status` STATUS_WORDS, all zero; the node must stay at its address.
    pub fn init(gpa: Allocator, io: std.Io, role: Role, cfg: Config, mem: Memory) !Node {
        const reg = cfg.settings.region();
        if (reg.slot > proto.MAX_SLOT or reg.slots > proto.LANES_MAX or mem.region.len < reg.total() or mem.status.len < proto.STATUS_WORDS) return error.Invalid;
        if (role == .expert and role.expert >= proto.EXPERTS) return error.Invalid;
        if (cfg.settings.ports < 1 or cfg.settings.ports > proto.PORTS_MAX) return error.Invalid;
        if (cfg.settings.fp and cfg.identity == null) return error.Invalid; // rank_fingerprint: the switch needs this node's line
        const st = cfg.settings.stripes;
        if (st == 1 or st > proto.STRIPES_MAX or (st >= 2 and (cfg.settings.stripe_min < 16 or cfg.settings.stripe_min % 16 != 0))) return error.Invalid;
        return .{ .gpa = gpa, .io = io, .role = role, .cfg = cfg, .reg = reg, .mem = mem };
    }

    pub fn nowNs(n: *const Node) u64 {
        return @intCast(std.Io.Timestamp.now(n.io, .awake).toNanoseconds());
    }

    fn word(n: *const Node, off: u64) *u32 {
        return @ptrCast(@alignCast(n.mem.region.ptr + off));
    }

    /// The last message this node's GPU rang (ctrl[0]).
    pub fn doorbell(n: *const Node) u32 {
        return @atomicLoad(u32, n.word(n.reg.ctrl()), .acquire);
    }

    /// Where messages from `k` land here (the attention: expert k; an expert: the attention).
    pub fn src(n: *const Node, k: u32) u32 {
        return proto.takeSrc(if (n.role == .expert) n.role.expert else null, k);
    }

    /// Whether message `seq` from source `s` has landed on every port.
    pub fn landedFrom(n: *const Node, s: u32, seq: u32) bool {
        for (0..n.cfg.settings.ports) |p| {
            if (@atomicLoad(u32, n.word(n.reg.flag(s, n.reg.slotOf(seq), @intCast(p))), .acquire) != seq) return false;
        }
        return true;
    }

    /// A message's bytes in this node's receive slot (host view; stable until message seq + slots from the same source).
    pub fn received(n: *const Node, s: u32, seq: u32, bytes: usize) []const u8 {
        const off: usize = @intCast(n.reg.recv(s, n.reg.slotOf(seq)));
        return n.mem.region[off..][0..bytes];
    }

    fn statusWord(n: *const Node, i: usize) u32 {
        return @atomicLoad(u32, &n.mem.status[i], .acquire);
    }

    /// The first failure and the GPU's counts, or null while the exchange is alive.
    pub fn failure(n: *const Node) ?Failure {
        const why = n.statusWord(proto.S_FAILED);
        if (why == 0) return null;
        return .{ .why = @enumFromInt(why), .peer = n.statusWord(proto.S_SRC), .seq = n.statusWord(proto.S_SEQ), .gave_up = n.statusWord(proto.S_GAVE_UP), .breach = n.statusWord(proto.S_BREACH) };
    }

    /// After a step's sync: the step's requests fail if the exchange did (the process goes on).
    pub fn check(n: *const Node) error{ExpertFailed}!void {
        if (n.failure() != null) return error.ExpertFailed;
    }

    /// Records a failure once (the first wins); every GPU wait reads it and returns, and every later one returns at once.
    pub fn fail(n: *Node, v: watcher.Verdict) void {
        if (@cmpxchgStrong(u32, &n.mem.status[proto.S_FAILED], 0, @intFromEnum(v.why), .seq_cst, .seq_cst) == null) {
            @atomicStore(u32, &n.mem.status[proto.S_SEQ], v.seq, .release);
            @atomicStore(u32, &n.mem.status[proto.S_SRC], v.peer, .release);
            n.report("exchange failed: expert {d} {t} at request {d}; this request fails, the process goes on", .{ v.peer, v.why, v.seq });
        }
    }

    fn report(n: *const Node, comptime fmt: []const u8, args: anytype) void {
        if (n.cfg.warn_failures) log.warn(fmt, args) else log.err(fmt, args);
    }

    fn settingsText(n: *const Node, buf: []u8) ![]const u8 {
        return n.cfg.settings.text(buf);
    }

    /// The attention rank: accepts both experts on `server`, agrees the settings, opens both pairs over the one region, connects and starts them, then runs the watcher.
    pub fn attend(n: *Node, server: link.Server, opener: Opener) !void {
        if (n.role != .attn) return error.Invalid;
        var buf: [512]u8 = undefined;
        const mine = try n.settingsText(&buf);
        for (0..proto.EXPERTS) |_| {
            const l = try server.accept(@intCast(n.cfg.setup_timeout_ms));
            errdefer l.close();
            const hello = try l.expect(n.gpa, tag_hello);
            defer n.gpa.free(hello);
            if (hello.len < 4) return error.BadHello;
            const k = std.mem.readInt(u32, hello[0..4], .little);
            if (k >= proto.EXPERTS or n.links[k] != null) return error.BadHello;
            try l.send(tag_ack, mine);
            if (!std.mem.eql(u8, hello[4..], mine)) {
                n.report("expert {d} runs \"{s}\", the attention \"{s}\"", .{ k, hello[4..], mine });
                return error.SettingsDiffer;
            }
            watch.keepalive(l.fd);
            n.links[k] = l;
        }
        if (n.cfg.settings.fp) try n.identities(); // rank_fingerprint: TF_GLM_FINGERPRINT=1 only (both hellos agreed)
        for (0..proto.EXPERTS) |k| try n.pairUp(@intCast(k), opener, proto.attnRank(@intCast(k)), true);
        for (n.proxies) |p| try p.?.start();
        for (n.links) |l| try l.?.send(tag_ready, &.{});
        for (n.links) |l| n.gpa.free(try l.?.expect(n.gpa, tag_ready));
        const now = n.nowNs();
        for (&n.beats) |*b| b.store(now, .release);
        n.rule = .{ .rules = n.cfg.rules };
        if (n.cfg.run_watcher) n.thread = try std.Thread.spawn(.{}, watchLoop, .{n});
    }

    /// Expert k: joins the attention at `master`, agrees, opens its pair as rank k, connects and starts it, then beats.
    pub fn join(n: *Node, master: net.IpAddress, opener: Opener) !void {
        if (n.role != .expert) return error.Invalid;
        const l = try link.Link.join(n.io, master, n.cfg.setup_timeout_ms);
        n.links[0] = l;
        watch.keepalive(l.fd);
        var buf: [4 + 512]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], n.role.expert, .little);
        const mine = try n.settingsText(buf[4..]);
        try l.send(tag_hello, buf[0 .. 4 + mine.len]);
        const theirs = try l.expect(n.gpa, tag_ack);
        defer n.gpa.free(theirs);
        if (!std.mem.eql(u8, theirs, mine)) {
            n.report("the attention runs \"{s}\", expert {d} \"{s}\"", .{ theirs, n.role.expert, mine });
            return error.SettingsDiffer;
        }
        if (n.cfg.settings.fp) try n.expertIdentity(l); // rank_fingerprint: TF_GLM_FINGERPRINT=1 only (the settings agreed)
        try n.pairUp(0, opener, n.role.expert, false);
        try n.proxies[0].?.start();
        n.gpa.free(try l.expect(n.gpa, tag_ready));
        try l.send(tag_ready, &.{});
        n.thread = try std.Thread.spawn(.{}, beatLoop, .{n});
    }

    /// rank_fingerprint (settings.fp), the attention: each expert's identity line (sent right after its hello agreed), then the table
    /// of all three to both experts, then the judgement every node makes of that table (fingerprint.judge).
    fn identities(n: *Node) !void {
        var lines: [proto.EXPERTS][]u8 = undefined;
        var got: usize = 0;
        defer for (lines[0..got]) |x| n.gpa.free(x);
        while (got < proto.EXPERTS) : (got += 1) lines[got] = try n.links[got].?.expect(n.gpa, fingerprint.tag);
        var tb: [fingerprint.table_max]u8 = undefined;
        const table = try fingerprint.tableOf(&tb, &.{ .{ "attention", n.cfg.identity.? }, .{ "expert-0", lines[0] }, .{ "expert-1", lines[1] } });
        for (n.links) |l| try l.?.send(fingerprint.tag, table);
        return n.judged(fingerprint.verdictOf(table, null));
    }

    /// rank_fingerprint (settings.fp), expert k: its identity line to the attention, then the table back (its own line unchanged in
    /// it) and the same judgement.
    fn expertIdentity(n: *Node, l: link.Link) !void {
        const mine = n.cfg.identity.?;
        try l.send(fingerprint.tag, mine);
        const table = try l.expect(n.gpa, fingerprint.tag);
        defer n.gpa.free(table);
        var lb: [16]u8 = undefined;
        return n.judged(fingerprint.verdictOf(table, .{ try std.fmt.bufPrint(&lb, "expert-{d}", .{n.role.expert}), mine }));
    }

    /// The verdict kept (fp_verdict) and logged: the WARN for nodes built without -Dsource_id; a mismatch is today's
    /// SettingsDiffer, its words naming the field and both values.
    fn judged(n: *Node, v: fingerprint.Verdict) !void {
        n.fp_verdict = v;
        if (n.fp_verdict.?.warning()) |w| log.warn("{s}", .{w});
        if (!v.refused) return;
        n.report("{s}", .{n.fp_verdict.?.why()});
        return error.SettingsDiffer;
    }

    /// One pair: open its proxy, swap blobs over link k (the attention's first), connect. qualification_17c70e37: the settings' stripes
    /// go to the proxy before anything is sent (both ends agreed them).
    fn pairUp(n: *Node, k: u32, opener: Opener, rank: u32, first: bool) !void {
        const p = try opener.open(opener.ctx, rank, n.mem.region, n.reg);
        n.proxies[k] = p;
        if (n.cfg.settings.stripes >= 2) try p.setStripes(n.cfg.settings.stripes, n.cfg.settings.stripe_min);
        var buf: [512]u8 = undefined;
        const len = try p.blob(&buf);
        const l = n.links[k].?;
        if (first) try l.send(tag_blob, buf[0..len]);
        const peer = try l.expect(n.gpa, tag_blob);
        defer n.gpa.free(peer);
        if (!first) try l.send(tag_blob, buf[0..len]);
        try p.connect(peer);
    }

    /// The attention's watcher: every 200 us it reads the doorbell, both experts' reply flags, heartbeats, link state and proxies, and records the rule's verdict.
    fn watchLoop(n: *Node) void {
        while (!n.stopping.load(.acquire)) {
            if (n.statusWord(proto.S_FAILED) != 0) return; // a GPU wait gave up or saw a breach: already failed
            const now = n.nowNs();
            const posted = n.doorbell();
            var seen: [proto.EXPERTS]watcher.Seen = undefined;
            for (0..proto.EXPERTS) |k| {
                seen[k] = .{ .landed = posted == 0 or n.landedFrom(n.src(@intCast(k)), posted) };
                n.drain(@intCast(k), &seen[k], now);
                seen[k].beat_ns = n.beats[k].load(.acquire);
                if (n.proxies[k].?.failure()) |why| {
                    seen[k].proxy_failed = true;
                    n.report("expert {d}'s pair: {s}", .{ k, why });
                }
            }
            if (n.rule.step(now, posted, seen)) |v| return n.fail(v);
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    /// Link k's pending frames (heartbeats), and whether it closed or erred.
    fn drain(n: *Node, k: u32, s: *watcher.Seen, now: u64) void {
        const l = n.links[k].?;
        while (true) {
            var fds = [_]posix.pollfd{.{ .fd = l.fd, .events = posix.POLL.IN | rdhup, .revents = 0 }};
            const got = posix.poll(&fds, 0) catch return s.markDead();
            if (got == 0) return;
            if (fds[0].revents & posix.POLL.IN != 0) {
                const f = l.recv(n.gpa) catch return s.markDead();
                defer f.deinit(n.gpa);
                if (f.tag == tag_beat) n.beats[k].store(now, .release);
                continue;
            }
            if (fds[0].revents & (rdhup | posix.POLL.HUP | posix.POLL.ERR | posix.POLL.NVAL) != 0) return s.markDead();
            return;
        }
    }

    /// An expert's heartbeat to the attention, every `beat_ns`, until it stops or the link fails.
    fn beatLoop(n: *Node) void {
        const step_ns: u64 = 10 * std.time.ns_per_ms;
        while (!n.stopping.load(.acquire)) {
            n.links[0].?.send(tag_beat, &.{}) catch {
                n.beat_lost.store(true, .release);
                return;
            };
            var waited: u64 = 0;
            while (waited < n.cfg.beat_ns and !n.stopping.load(.acquire)) : (waited += step_ns) {
                const ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(step_ns) };
                _ = std.c.nanosleep(&ts, null);
            }
        }
    }

    /// Expert: request `seq`'s header once its flag lands on every port; a stop frame, a closed link, a failed proxy, a lost heartbeat or a flag past `seq` end the wait instead.
    pub fn waitRequest(n: *Node, seq: u32) Wait {return n.waitRequestControlled(seq,null);}

    /// Current-native TERM token: an idle expert must not wait for new model traffic to close.
    pub fn waitRequestControlled(n: *Node, seq: u32, requested:?*const std.atomic.Value(bool)) Wait {
        const s = n.src(0);
        const flag = n.word(n.reg.flag(s, n.reg.slotOf(seq), 0));
        var spins: u64 = 0;
        while (true) {
            if(n.stopping.load(.acquire))return .stopped;
            if(requested)|flag_requested|if(flag_requested.load(.acquire))return .stopped;
            const v = @atomicLoad(u32, flag, .acquire);
            if (v == seq and n.landedFrom(s, seq)) {
                const hdr = n.received(s, seq, proto.HDR_BYTES);
                return .{ .request = proto.Header.read(hdr[0..proto.HDR_BYTES]) catch return .{ .lost = "a request without the AFD magic" } };
            }
            if (@as(i32, @bitCast(v -% seq)) > 0) return .{ .lost = "a request ahead of its sequence (a protocol breach)" };
            spins += 1;
            if (spins % 1024 == 0) {
                if (n.linkEvent()) |w| return w;
                if (n.proxies[0].?.failure()) |why| return .{ .lost = why };
                if (n.beat_lost.load(.acquire)) return .{ .lost = "the heartbeat could not be sent" };
            }
            if (spins > 1 << 22) {
                const ts: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
                _ = std.c.nanosleep(&ts, null);
            } else std.atomic.spinLoopHint();
        }
    }

    /// Expert: no more requests are taken (a silent expert in the lab); waits only for a stop frame or a loss.
    pub fn idle(n: *Node) Wait {
        while (true) {
            if (n.linkEvent()) |w| return w;
            if (n.proxies[0].?.failure()) |why| return .{ .lost = why };
            if (n.beat_lost.load(.acquire)) return .{ .lost = "the heartbeat could not be sent" };
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 1_000_000 };
            _ = std.c.nanosleep(&ts, null);
        }
    }

    /// Expert: a stop frame or the attention's link closing, if either has happened.
    fn linkEvent(n: *Node) ?Wait {
        const l = n.links[0].?;
        var fds = [_]posix.pollfd{.{ .fd = l.fd, .events = posix.POLL.IN | rdhup, .revents = 0 }};
        const got = posix.poll(&fds, 0) catch return .{ .lost = "poll failed on the control link" };
        if (got == 0) return null;
        if (fds[0].revents & posix.POLL.IN != 0) {
            const f = l.recv(n.gpa) catch return .{ .lost = "the attention closed the control link (it exited, or the link is down)" };
            defer f.deinit(n.gpa);
            if (f.tag == tag_stop) return .stopped;
            return null;
        }
        return .{ .lost = "the control link erred" };
    }

    /// The attention's clean end: each expert gets a stop frame.
    pub fn stopExperts(n: *Node) void {
        if (n.role != .attn) return;
        for (n.links) |l| if (l) |x| x.send(tag_stop, &.{}) catch {};
    }

    /// Helios: GPU consumers must already be proved complete. Failed/unknown/partial
    /// proxies retain every handle and the registered frame; no deinit on refusal.
    pub fn closeChecked(n: *Node, until: u64) !void {
        try n.check();
        for (n.proxies) |p| if (p) |x| {
            const settle = x.vt.quiesce orelse return error.UnprovenProxyQuiescence;
            try settle(x.ctx, until);
        };
        n.deinit();
    }

    /// Stops the node's thread, then closes its proxies and links (the attention's experts see the close).
    pub fn deinit(n: *Node) void {
        n.stopping.store(true, .release);
        if (n.thread) |t| t.join();
        n.thread = null;
        for (&n.proxies) |*p| if (p.*) |x| {
            x.close();
            p.* = null;
        };
        for (&n.links) |*l| if (l.*) |x| {
            x.close();
            l.* = null;
        };
    }
};

test {
    _ = proto;
    _ = watcher;
}
