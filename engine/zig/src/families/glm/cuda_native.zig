// Modified for V4 by Scott Quarrier with AI assistance: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
//! GLM-5.3-Flash NVFP4 native CUDA at TP over processes (TF_GLM_RANK, NCCL_COMM_ID): rank 0 serves, others follow.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const images_mod = @import("cuda_images.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const dflash = @import("dflash.zig");
const dflash_load = @import("dflash_load.zig");
const pages_mod = @import("cuda_pages.zig");
const share_mod = @import("cuda_share.zig");
const kp = @import("core").kv_prompts;
const kv = @import("core").kv_pool;
const hybrid_config=@import("cuda_hybrid_config.zig");
const hybrid_forward=@import("cuda_hybrid_forward.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{ "modelopt-nvfp4", "modelopt-mixed", "exl3" };
// cc-defer: a fixed window; the Python family sizes it from free memory (capacity.admit), port that when serving needs
pub const default_context: i64 = 8192;
pub const ceiling_switch = @import("cuda_env.zig").ceiling_switch;
pub const raisedCeiling = @import("cuda_env.zig").raisedCeiling;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fw.prefill_rows;
/// Chunk rows change prompt arithmetic: states keep and resume only where a cold pass cuts (cuda_prompt_cuts).
pub const snap_grid: u32 = prompt_rows;
/// cuda_prompt_cuts cuts a cold pass at the request's chunk starts too, so the server may send its message starts.
pub const request_starts = true;
pub const request_starts_switch = "TF_GLM_REQUEST_STARTS";
/// Starts closer than 256 tokens merge, as in the measured pair receipts (#2404); a 1-row minimum needs its own run.
pub const request_min_chunk: u32 = 256;
/// TF_GLM_PROMPT_TAIL=1: a prompt that names no history keeps its last chunk start, as the KV pages' store does.
pub const snap_tail = "TF_GLM_PROMPT_TAIL";
/// `explain` words this family's open refusals too (where to set the rank and the id).
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `slide` stays false: the server refuses --slide for a family without a learner (no `slide` here).
pub const Options = struct {
    /// Process-owned sticky startup cancellation; null leaves non-server callers unchanged.
    cancel_requested: ?*const std.atomic.Value(bool) = null,
    context: usize,
    kv8: bool = false,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?images_mod.Limits = null, // --vision: rank 0's tower, every rank's image rows
    drafter: ?[]const u8 = null, // --drafter: DFlash2's folder (the same on every rank; needs drafts on)
    drafter_bits: u8 = 4, // --drafter-bits: 4 (default) or 0 (bf16)
    mtp_confidence: ?f64 = null, // --mtp-confidence: rank 0's lanes stop an MTP chain where its chance falls under it
    learn_conversations: bool = false, // --learn-conversations: refused with TF_GLM_REQUEST_STARTS=1 (cuda_env)
};
/// --drafter: DFlash2 (incoai/GLM-5.3-Flash-DFlash2, Apache-2.0 weights) drafts instead of the MTP head.
pub const takes_drafter = true;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    deinit_checked:?*const fn(*anyopaque)bool=null,
    close_gate: ?*const fn(*anyopaque) bool=null,
    draft_kind: ?[]const u8 = null,
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
    peer_room: ?Peer = null, // rank 0: the server admits streams and kept states by this when it is smaller
    vision: ?*images_mod.Images = null, // rank 0's tower for the server's image offer
    layout: ?kv.Layout = null, // every stream's memory as one pool of KV pages (null: a window each, by TF_KV_PAGES)
    engine_paged: bool = false, // with a layout, the engine's own sequence takes pages too: the plan sets a window aside
    host_snaps: bool = false, // TF_GLM_HOST_CACHE: kept states in pinned host memory, --prompt-cache-gib of it
};

pub fn benchmark(p: *anyopaque) struct { ranks: ?u32, group: ?[]const u8 } {
    const own: *Owned = @ptrCast(@alignCast(p));
    return .{ .ranks = @intCast(own.group.world), .group = own.benchmark_group };
}

/// The pool behind every stream: the engine's own sequence and `streams` more on `pages` KV pages, on every rank.
pub fn attach(p: *anyopaque, streams: u32, pages: usize) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    const l = own.layout orelse return error.NoPages;
    const kept = @import("cuda_kv_host.zig").pagesFor(@import("cuda_state.zig").pageBytes(&own.e.w), own.e.capacity, pages);
    try own.lanes.x.pool(streams + 1, kept, l.pageBytes());
    own.e.pages = try pages_mod.Pages.init(own.gpa, own.e, l, streams + 1, kept, true);
}

/// TF_KV_SHARE: the plan's `snaps` slots on each rank, a state each at the window's last mark (the layout's slot).
pub fn share_states(p: *anyopaque, snaps: usize, parallel: u32, planned: ?kv.Budget) !void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.share == 0) return;
    const l = own.layout orelse return error.NoPages;
    const slot = l.snap_bytes;
    const host = try snapshots.pages.onHost();
    const n = if (host) own.share else snaps; // TF_GLM_HOST_CACHE: in host memory, outside the device plan
    own.e.share_pages = n > 0; // no slot, no store: rank 0 counts its idle pages as the followers do
    if (n > 0) try readySnapshots(own, n * slot);
    const want = share_mod.advice(n, parallel, l, if (host) null else planned) orelse return;
    const why = "TF_KV_SHARE={d} keeps {d} states in the prompt cache's slots, under 4 a GLM stream at --parallel {d}: live streams evict each other's before a resume; ";
    if (want.gib) |gib| std.log.warn(why ++ "{d} take TF_KV_SHARE={d} and --prompt-cache-gib {d}", .{ own.share, n, parallel, want.states, want.states, gib }) else std.log.warn(why ++ "the room holds {d} at any --prompt-cache-gib", .{ own.share, n, parallel, want.most });
}

/// Rank 0, its pool attached: choose the expert form and publish MTP's window prices, timed or kept.
pub fn priceWindows(p: *anyopaque, io: std.Io, ctx: *const cuda.Context, dir: []const u8, facts: lanes.Model) !lanes.Model {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.lanes.costs = try @import("cuda_lane_costs.zig").price(own.gpa, io, ctx, dir, own.lanes.x) orelse return facts;
    return own.lanes.factsWithCopies(own.copies);
}

/// Whether a request's caches fit now (the lane host's admission, on its engine thread).
pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return own.lanes.fits(prompt, max_tokens);
}

/// The page pool the load ledger watches: rank 0's, the one that takes and gives pages for every rank.
pub fn kvPool(p: *anyopaque) ?kv.Source {
    const own: *Owned = @ptrCast(@alignCast(p));
    const pages = own.e.pages orelse return null; // a window a stream
    if (!pages.leader) return null;
    return .{ .pool = &pages.pool, .page_tokens = @import("cuda_state.zig").page_rows };
}

/// This rank's place: TF_GLM_RANK of TF_GLM_WORLD (both, or neither for a refusal that names them).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_GLM_RANK") orelse return error.NoRank;
    const w = std.c.getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w), 10) catch return error.NoRank };
    return checked(p);
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Error-propagating role metadata before generic whole-checkpoint admission. Null is unselected only.
pub fn preloadWeightBytes(a:std.mem.Allocator,io:std.Io,o:anytype) !?u64 {
    const z=std.c.getenv("TF_GLM_HYBRID_CONFIG") orelse return null;
    const text=try std.Io.Dir.cwd().readFileAlloc(io,std.mem.span(z),a,.limited(65536));defer a.free(text);
    var selected=try hybrid_config.parse(a,text);defer selected.deinit();
    const p=try place();var problem:[]const u8="";var s=try ranks.Settings.read(&ranks.environment,&problem);
    s.kv8=o.kv8;s.pages=try startup.paging(startup.pagesText(),p.world,s.kv8);
    try selected.value.attention(a,p.world,@intCast(o.context orelse default_context),o.drafts,s);
    const plan=try @import("cuda_hybrid_preload.zig").read(a,io,o.dir,selected.value);
    return plan.load_peak;
}

/// Bytes this rank's weights take at most: the checkpoint (TF_GLM_LAYERS: the tensors read) times its largest share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64, drafts: bool) u64 {
    const p = place() catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    const whole = @import("cuda_thin_bytes.zig").count(a, io, dir, total, ranks.environment("TF_GLM_LAYERS"), c.layers, drafts and c.mtp > 0);
    return @intCast(@as(u128, whole) * f[0] / f[1]);
}

/// Slots for `context` plus a verify window, a multiple of 64 so the sparse pool keeps the AOT set.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + fw.window_rows, 64);
}

/// A request's last verify window: the plan's window holds that many rows past the context, as `capacity` does.
pub const window_rows: u32 = fw.window_rows;

const snapshots = @import("cuda_snapshots.zig");

const Owned = struct {
    memory:?*@import("cuda_hybrid_ledger.zig").Ledger=null,
    gpa: std.mem.Allocator,
    hybrid: ?*hybrid_forward.Session=null,
    hybrid_config: ?std.json.Parsed(hybrid_config.Config)=null,
    quarantined: std.atomic.Value(bool)=.init(false),linked: bool=false,retained_next: ?*Owned=null,
    benchmark_group: ?[]const u8,
    progress: cuda.startup_watch.Watcher,
    group: cuda.tp_group.Group, // the ranks' communicator; two ranks add RoCE writes for the main stream's small gathers
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
    layout: ?kv.Layout,
    share: usize, // TF_KV_SHARE: states the shared store over the pool's pages keeps (0: the host store)
    copies: bool, // TF_GLM_COPIES=1: verified root and context copies
};

const env = @import("sys").env;

/// TF_ROCE_MAX_KB in KiB, bounded by the RDMA scatter/gather length before any allocation.
pub fn roceBytes(text: ?[]const u8) !usize {
    const v = text orelse return 4096 << 10;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadRoceCap;
    if (n == 0 or n > std.math.maxInt(u32) / 1024) return error.BadRoceCap;
    return std.math.mul(usize, n, 1024) catch return error.BadRoceCap;
}

const startup = @import("cuda_start.zig");
const paging = startup.paging;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (o.cancel_requested) |flag| if (flag.load(.acquire)) return error.Stopped;
    _ = kernels; // our own kernels: a captured set is not used
    var selected: ?std.json.Parsed(hybrid_config.Config)=null;
    var transferred=false;defer if(!transferred) if(selected) |*s| s.deinit();
    if(std.c.getenv("TF_GLM_HYBRID_CONFIG")) |z| {
        const text=try std.Io.Dir.cwd().readFileAlloc(io,std.mem.span(z),gpa,.limited(65536));defer gpa.free(text);
        selected=try hybrid_config.parse(gpa,text);
        if(selected.?.value.role!=.attention or o.vision!=null or o.drafter!=null) return error.BadHybridRole;
    }
    const hybrid_until=if(selected) |s| @import("tp").sock.nowNs()+|(@as(u64,s.value.setup_ms)*std.time.ns_per_ms) else 0;
    try @import("cuda_env.zig").startup();
    try @import("cuda_env.zig").requestStartsAlone(o.learn_conversations); // before any rank opens
    const p = try place();
    const roce_bytes = try roceBytes(if (std.c.getenv("TF_ROCE_MAX_KB")) |v| std.mem.span(v) else null);
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    var set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    try @import("cuda_roce_multi.zig").validate(p.world, p.rank, set.roce_multi);
    set.kv8 = o.kv8;
    if (share_mod.moved(if (std.c.getenv("TF_GLM_KV_SHARE")) |v| std.mem.span(v) else "")) return error.GlmKvShareMoved; // before any rank opens
    const share_snaps = try kp.sharing();
    if (share_mod.tooMany(share_snaps)) {
        std.log.err("TF_KV_SHARE={d} asks more kept states than the {d} it keeps: set {d} or fewer", .{ share_snaps, kv.most_snaps, kv.most_snaps });
        return error.GlmShareTooMany;
    }
    const loaded=startup.open(io, p, startup.pagesText(), set, cuda.tp_watch, StartArgs{.hybrid=selected,.hybrid_until=hybrid_until, .gpa = gpa, .io = io, .ctx = ctx, .dir = dir, .o = o, .p = p, .roce_bytes = roce_bytes, .share = share_snaps }, openRank, startup.log) catch |err| {if(err==error.HybridRetained) transferred=true;return err;};
    transferred=true;return loaded;
}

const StartArgs = struct {
    hybrid: ?std.json.Parsed(hybrid_config.Config),hybrid_until: u64,
    gpa: std.mem.Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    dir: []const u8,
    o: Options,
    p: Place,
    roce_bytes: usize,
    share: usize,
};
const MemoryTest=if(@import("builtin").is_test)struct{pub var engine_options:?*const fn(engine.Options)anyerror!void=null;pub var after:?*const fn(*Owned) anyerror!void=null;}else struct{};
test "I3 native owned close: selected Loaded has checked release result for its generic CUDA parent" {try std.testing.expect(@hasField(Loaded,"deinit_checked"));}
test "I3 native owned close: sticky accounting refusal closes backend admission even without a session" {
    var l=@import("cuda_hybrid_ledger.zig").Ledger.init(.attention,555);l.sticky_failure=true;l.overflow=true;var own:Owned=undefined;@memset(std.mem.asBytes(&own),0);own.memory=&l;
    const old=retained_hybrids;defer retained_hybrids=old;try std.testing.expect(!hybridGate(&own));try std.testing.expect(own.quarantined.load(.acquire));
}
var memory_generation=std.atomic.Value(u64).init(0);

/// After liveness and page settings: NCCL, the engine and the link, then a follower's loop or rank zero's lanes.
fn openRank(a: StartArgs, set: ranks.Settings) !Loaded {
    var retained=false;return openRankOwned(a,set,&retained) catch |err|{return if(retained)error.HybridRetained else err;};
}
fn memoryRefused(ctx:*anyopaque)void{const own:*Owned=@ptrCast(@alignCast(ctx));own.quarantined.store(true,.release);keepHybrid(own);}
fn openRankOwned(a:StartArgs,set:ranks.Settings,retained:*bool)!Loaded {
    const memory_scope=@import("cuda_hybrid_ledger.zig").scopeOwner(.workspace);defer memory_scope.end();
    const gpa = a.gpa;
    const io = a.io;
    const ctx = a.ctx;
    const dir = a.dir;
    const o = a.o;
    const p = a.p;
    const roce_bytes = a.roce_bytes;
    const copies = try cuda.priority_copies.read("TF_GLM_COPIES", o.drafts);
    var load_cache = @import("core").checkpoint.cache.Load.begin(io, p.world > 1);
    defer load_cache.deinit();
    // TF_KV_SHARE keeps prompts as the pool's pages: refused before the weights load, every rank told (startup)
    if (a.share > 0 and !set.pages) return error.GlmShareNeedsPages;
    if (@import("cuda_kv_host.zig").onHost() and !set.pages) return error.GlmKvHostNeedsPages;
    if(a.hybrid==null)cuda.memory_receipt.emit(@intCast(p.rank), .load_begin, true); // selected hybrid binds its observer first
    const own = try gpa.create(Owned);
    own.hybrid=null;own.hybrid_config=a.hybrid;own.quarantined=.init(false);own.linked=false;own.retained_next=null;own.memory=null;
    errdefer {retained.*=own.quarantined.load(.acquire);if(!own.quarantined.load(.acquire))gpa.destroy(own);}
    own.gpa = gpa;
    own.benchmark_group = null;
    own.share = a.share;
    own.copies = copies;
    if(a.hybrid)|selected|{
        const l=try gpa.create(@import("cuda_hybrid_ledger.zig").Ledger);errdefer gpa.destroy(l);
        var before=memory_generation.load(.acquire);const incarnation=while(true){const next=try std.math.add(u64,before,1);if(memory_generation.cmpxchgWeak(before,next,.acq_rel,.acquire))|new|{before=new;}else break next;};
        l.*=@import("cuda_hybrid_ledger.zig").Ledger.init(switch(selected.value.role){.attention=>.attention,.expert0=>.expert0,.expert1=>.expert1},incarnation);
        l.refusal_ctx=own;l.refuse=memoryRefused;
        try l.install();own.memory=l;
    }
    errdefer if(own.memory)|l|{
        cuda.memory_receipt.emit(@intCast(p.rank),.cleanup_begin,false);
        cuda.copy_source.dropOwned(l);
        l.finish(!own.quarantined.load(.acquire),false);const final=cuda.memory_owner.snapshot();cuda.memory_receipt.emitCaptured(@intCast(p.rank),.cleanup_end,false,final);
        if(l.status().clean){l.uninstall();gpa.destroy(l);own.memory=null;}else{own.quarantined.store(true,.release);keepHybrid(own);}
    };
    if(own.memory!=null)cuda.memory_receipt.emit(@intCast(p.rank),.load_begin,true);
    if(comptime @import("builtin").is_test)if(MemoryTest.after)|after|try after(own);
    const cap = capacity(o.context);
    const engine_options:engine.Options=.{.cancel_requested=o.cancel_requested,.setup_until=if(a.hybrid!=null)a.hybrid_until else null,.hybrid=if(own.hybrid_config)|*selected|&selected.value else null,.capacity=cap,.mtp=o.drafts,.settings=set,.vision=o.vision,.drafter=o.drafter!=null,.owner=&own.group};
    if(comptime @import("builtin").is_test)if(MemoryTest.engine_options)|hook|try hook(engine_options);
    // every connection registered at init, while memory is free (lazily, a full GB10's ibv_reg_mr can refuse it)
    if (std.c.getenv("NCCL_RUNTIME_CONNECT") == null) _ = env.setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    // NCCL's cuMem host buffers wore out GB10's driver pool under load (NV_ERR_NO_MEMORY; next ranks failed to start)
    if (std.c.getenv("NCCL_CUMEM_HOST_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_HOST_ENABLE", "0", 0);
    if (std.c.getenv("NCCL_CUMEM_ENABLE") == null) _ = env.setenv("NCCL_CUMEM_ENABLE", "0", 0);
    // NCCL_COMM_ID holds NCCL's bootstrap to its subnet past NCCL_SOCKET_IFNAME: a follower naming one lets it pick
    if (p.rank > 0 and std.c.getenv("NCCL_SOCKET_IFNAME") != null) _ = env.unsetenv("NCCL_COMM_ID");
    var diag = cuda.p2p_report.begin(p.rank, p.world);
    defer diag.end();
    own.progress = .init(io, p.rank, p.world, diag.capture);
    own.progress.start();
    errdefer if(!own.quarantined.load(.acquire)) own.progress.end();
    own.progress.enter(.nccl_init);
    var lib = try cuda.nccl.Library.open();
    errdefer if(!own.quarantined.load(.acquire)) lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (p.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // the root at NCCL_COMM_ID
        cuda.tp_watch.sendId(&id.internal);
    } else id.internal = try cuda.tp_watch.takeId(io); // rank 0's, over this rank's line
    const comm = try cuda.startup_log.init(io, &lib, id, p.rank, p.world);
    own.group = .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world };
    errdefer if(!own.quarantined.load(.acquire)) own.group.destroyComm(); // guarded: after an abort it destroys nothing
    cuda.p2p_report.beforePeers(&diag, .{ .lib = &own.group.lib, .comm = comm, .later = true });
    own.progress.enter(.peer_warmup);
    if (p.world > 1) try @import("cuda_warm_peers.zig").warm(io, ctx.d, &own.group, .of(set.split, set.ring), &diag);
    cuda.p2p_report.afterInit(&diag, .{ .lib = &own.group.lib, .comm = comm, .d = ctx.d, .later = true });

    own.progress.enter(.weights);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.group.lib, own.group.comm, p.rank, p.world, engine_options);
    errdefer if(!own.quarantined.load(.acquire)) own.e.deinit();
    if (o.vision) |lim| { // before the rooms: the tower's bytes count against rank 0's
        var problem: []const u8 = "";
        const ql: cuda.qlinear.Linear = .{ .lane = &own.e.m.lane, .nvfp4_prompt = &own.e.m.prompt }; // the tower's MXFP8, NVFP4
        own.e.images = (try images_mod.Images.init(gpa, io, ctx.d, own.e.stream, dir, lim, fw.prefill_rows + 1, p.rank == 0, ql, &problem)) orelse {
            std.log.err("{s}", .{problem});
            return error.NoVisionTower;
        };
    }
    return @import("cuda_open_owner.zig").finish(own.e, OpenArgs{
        .own = own,
        .io = io,
        .ctx = ctx,
        .o = o,
        .p = p,
        .cap = cap,
        .roce_bytes = roce_bytes,
        .load_cache = &load_cache,.hybrid_until=a.hybrid_until,
    }, finishRank);
}

const OpenArgs = struct {
    hybrid_until: u64,
    load_cache: *@import("core").checkpoint.cache.Load,
    own: *Owned,
    io: std.Io,
    ctx: *const cuda.Context,
    o: Options,
    p: Place,
    cap: usize,
    roce_bytes: usize,
};

fn finishRank(a: OpenArgs) !Loaded {
    const own = a.own;
    const set = own.e.settings;
    const gpa = own.gpa;
    const io = a.io;
    const ctx = a.ctx;
    const o = a.o;
    const p = a.p;
    const cap = a.cap;
    const roce_bytes = a.roce_bytes;
    if (o.drafter) |dd| { // before the rooms: its weights and scratch count against every rank's
        if (!o.drafts) return error.DrafterNeedsDrafts;
        own.e.drafter = try dflash_load.load(gpa, io, own.e, dd, p.rank, p.world, o.drafter_bits);
    }
    // the group's own RoCE path (every rank or none), unless TF_GLM_ROCE_MULTI=1 lends fabric/tp's wire
    if (ranks.ownRoce(set, p.world)) try own.group.openRoce(gpa, io, ctx.d, own.e.stream, roce_bytes);
    errdefer if (own.group.fast) |f| f.deinit();
    try @import("cuda_roce_multi.zig").open(gpa, io, ctx.d, &own.group, set.roce_multi); // three ranks or more, likewise
    errdefer if (own.group.multi) |m| m.deinit(m.ctx);
    own.e.installGroup(&own.group);
    own.link = try follow.Link.init(gpa, ctx.d, &own.group.lib, own.group.comm, own.e.stream, try linkCapacity(cap, o.vision));
    errdefer if(!own.quarantined.load(.acquire)) own.link.deinit();
    own.e.snap_setup = .{ .ptr = own, .run = readySnapshots };
    try own.e.prepareHostGather(); // engine owns the native singleton before room/admission
    if (p.world > 1) try a.load_cache.finishAfter(.{ io, own.e.checkpoint_digest }, cuda.tp_watch.ready) else a.load_cache.finish();
    // Rank 0 admits by the smallest room of the ranks; its sizes are the largest (the first ranks take the remainder).
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try ranks.gather(u64, ctx.d, &own.group.lib, own.group.comm, own.e.stream, p.rank, mine, rooms);
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    own.benchmark_group = try @import("core").benchmark_group.cuda(gpa, io, ctx, &own.group, own.e.stream);
    errdefer if (own.benchmark_group) |b| gpa.free(b);
    // every rank loaded and agreed (cuda_ranks) before the first step; the digest names what they agreed on
    if(own.memory==null)cuda.memory_receipt.emit(@intCast(p.rank), .load_end, false);
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, frees what it owns, then leaves the process
        startup_phase(own, .following);
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            cuda.tp_watch.fail(@errorName(err)); // rank 0 and the others stop with it
            std.c._exit(1); // the stream may hold kernels waiting on a gone rank: a normal exit would join them
        };
        cuda.tp_watch.ended();
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        cuda.memory_receipt.emit(@intCast(p.rank), .cleanup_begin, false);
        release(own);

        std.process.exit(0);
    }
    // Rank 0 alone reads these: TF_GLM_PROMPT_STEPS=0 passes prompts whole, TF_GLM_SHARED=0 runs windows alone.
    const steps = try @import("sys").environment.value(bool, "TF_GLM_PROMPT_STEPS", ranks.environment("TF_GLM_PROMPT_STEPS"), true);
    const shared = try @import("sys").environment.value(bool, "TF_GLM_SHARED", ranks.environment("TF_GLM_SHARED"), true);
    own.link.asleep = cuda.tp_watch.open(); // the other ranks begin asleep (follow): the first step wakes them
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link }, .whole = !steps, .alone = !shared, .mtp_confidence = o.mtp_confidence };
    const extra: u64 = (if (o.drafter != null) dflash.ring_bytes else 0) + if (own.e.images) |v| v.streamBytes() else @as(usize, 0);
    own.layout = if (set.pages) try pages_mod.layout(gpa, own.e, extra) else null;
    if (own.share > 0) if (own.layout) |*l| { // TF_KV_SHARE: the plan counts each kept state's slot
        own.e.share_pages = true;
        l.snap_bytes = pages_mod.shareSlot(own.e);
    };
    const host_snaps=try snapshots.pages.onHost();
    if(own.hybrid_config) |selected| {
        if(!memoryGood(own))return error.HybridRetained;
        const session=try hybrid_forward.Session.create(gpa,own.e,selected.value);own.hybrid=session;
        session.cancel_requested=o.cancel_requested;
        session.open(io,a.hybrid_until) catch |err| {
            const receipt=session.closeChecked();
            if(!receipt.quiet) {own.quarantined.store(true,.release);keepHybrid(own);return error.HybridRetained;}
            session.destroy();own.hybrid=null;return err;
        };
    }
    if(!memoryGood(own))return error.HybridRetained;
    // load_end follows all selected transport/session allocations and final accounting admission.
    if(own.memory!=null)cuda.memory_receipt.emit(@intCast(p.rank),.load_end,false);
    return .{
        .close_gate=if(own.memory!=null or own.hybrid!=null) hybridGate else null,
        .draft_kind = if (own.e.drafter != null) "DFlash2" else null,
        .backend = own.lanes.backend(),
        .facts = own.lanes.factsWithCopies(own.copies),
        .rows = glm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes() + extra, // each stream past the first makes its own sequence on every rank
        .free_streams = 1, // the first is the sequence the engine made
        .ctx = own,
        .deinit = close,
        .deinit_checked=if(own.memory!=null)closeChecked else null,
        .target = own.e,
        .peer_room = peer,
        .vision = own.e.images,
        .layout = own.layout,
        .engine_paged = set.pages,
        .host_snaps = host_snaps,
    };
}

// Each rank keeps a copy under rank 0's id, in the room rank 0 admitted by (the smallest over the ranks).

/// The rounded pages for a committed prefix, including the head's possible verify window.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes(at);
}

/// The first legal chunk mark, with the same page rounding and possible head window used by cache admission.
pub fn snap_min_bytes(ptr: *anyopaque) u64 {
    return snap_bytes(ptr, snap_grid);
}

/// Reserve every spare page inside the admitted cache budget before requests can enter either rank.
pub fn snap_ready(ptr: *engine.Engine, budget: u64) !u64 {
    if (ptr.snap_setup) |s| try s.run(s.ptr, budget) else try ptr.prepareSnapshots(budget);
    return 0; // the store accounts both held and spare pages through its vtable
}

fn readySnapshots(ptr: *anyopaque, budget: u64) !void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    const x: follow.Leader = .{ .e = own.e, .link = &own.link }; // its steps wake the ranks asleep on their lines
    try x.snapshots(budget);
    try x.idle(); // no stream yet: they sleep again until the first
}

pub fn snap_charged(ptr: *anyopaque, saved: *anyopaque) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    return e.snaps.get(k.id).?.lease.ids.items.len * snapshots.pages.page;
}

pub fn snap_spare(ptr: *anyopaque) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return if (e.snap_pages) |*p| p.spare() else 0;
}

pub fn snap_reuses(ptr: *anyopaque, at: u32) bool {
    return snap_spare(ptr) >= snap_bytes(ptr, at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): copy the state there under a new id, with its cuts below.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at, .below = @import("cuda_prompt_cuts.zig").below(e.cur.cuts.items, at) };
    return k;
}

/// TF_KV_SHARE's resume waits in `owner` for the pass's start on every rank; any other would leave the ranks behind.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (share_mod.of(e) == null) return error.RestoredInPromptPass;
    const o: *share_mod.Owner = @ptrCast(@alignCast(owner orelse return error.NoGlmOwner));
    o.pending = @ptrCast(@alignCast(saved));
}

/// TF_KV_SHARE: the pool the shared store keeps prompts in, rank 0's pages (null: a window a stream).
pub fn share_pool(ptr: *anyopaque) ?*kv.Pool {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const p = e.pages orelse return null;
    return &p.pool;
}

/// TF_KV_SHARE: the store the prompt passes keep and resume through, on rank 0's pages (null: none).
pub fn share_attach(ptr: *anyopaque, prompts: ?kp.Prompts) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pages) |p| p.prompts = prompts;
}

/// TF_KV_SHARE: rows past the store's pages are each rank's snapshots, paid by the cache's room: it makes no pages.
pub const share_kept_ppm: u32 = 0;

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    _=closeChecked(ptr);
}
fn closeChecked(ptr:*anyopaque)bool{
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if(!hybridGate(ptr)) return false;
    const rank: u32 = @intCast(own.group.rank);
    cuda.memory_receipt.emit(rank, .cleanup_begin, false);
    cuda.tp_watch.ended(); // the other ranks leave on the end step, so their lines close on purpose from here
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    const released=releaseChecked(own);

    return released;
}

/// The generic host reports calibration and readiness without changing either operation.
pub fn startup_phase(ptr: *anyopaque, phase: cuda.startup_watch.Phase) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.progress.enter(phase);
    if (phase == .ready or phase == .following or phase == .failed) own.progress.end();
}

var retained_hybrids: ?*Owned=null;
var hybrid_mutex: std.atomic.Mutex=.unlocked;
fn keepHybrid(own: *Owned) void {
    while(!hybrid_mutex.tryLock()) std.Thread.yield() catch {};defer hybrid_mutex.unlock();
    if(own.linked) return;own.linked=true;own.retained_next=retained_hybrids;retained_hybrids=own;
}
fn memoryGood(own:*Owned)bool{
    if(own.quarantined.load(.acquire))return false;if(own.memory)|l|if(l.status().sticky_failure){memoryRefused(own);return false;};return true;
}
fn hybridGate(ptr: *anyopaque) bool {
    const own: *Owned=@ptrCast(@alignCast(ptr));if(!memoryGood(own))return false;
    if(own.hybrid) |session| if(!session.closeChecked().quiet) {own.quarantined.store(true,.release);keepHybrid(own);return false;};
    return memoryGood(own);
}

/// What a rank owns, freed by rank 0's close and by a follower before it leaves.
fn release(own:*Owned)void{_=releaseChecked(own);}
fn releaseChecked(own: *Owned) bool {
    if(own.quarantined.load(.acquire))return false;
    const rank:u32=@intCast(own.group.rank);var final:?cuda.memory_owner.Ownership=null;
    own.progress.end();
    own.link.deinit();
    if (own.e.images) |v| v.deinit();
    if(own.hybrid) |s| s.destroy();
    own.e.deinit();
    if(own.memory)|l|{
        cuda.copy_source.dropOwned(l);
        l.finish(true,false);
        final=cuda.memory_owner.snapshot();
        if(!l.status().clean){cuda.memory_receipt.emitCaptured(rank,.cleanup_end,false,final);own.quarantined.store(true,.release);keepHybrid(own);return false;}
        l.uninstall();own.gpa.destroy(l);own.memory=null;
    }
    own.group.close();
    if(own.hybrid_config) |*selected| selected.deinit();
    if (own.benchmark_group) |b| own.gpa.free(b);
    own.gpa.destroy(own);
    cuda.memory_receipt.emitCaptured(rank,.cleanup_end,false,final);
    return true;
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadRoceCap => "TF_ROCE_MAX_KB: give an integer from 1 through 4194303 KiB, or unset it for 4096 KiB",
        error.NoRank => "GLM-5.3-Flash runs at TP over hosts: set TF_GLM_RANK and TF_GLM_WORLD on every host (WORLD 1: one GPU, with TF_GLM_LAYERS for a thin model)",
        error.NoCommId => "GLM-5.3-Flash's ranks meet over NCCL: with TF_GLM_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        error.Fp8OnPages => "--kv-cache fp8 keeps a window per stream: unset TF_KV_PAGES, or set it to auto or off",
        error.BadKvShare => "TF_KV_SHARE: give how many states the prompt store over the KV pages keeps, or 0 or unset for off",
        error.GlmKvShareMoved => "TF_GLM_KV_SHARE is now TF_KV_SHARE, the prompt store every family on the KV pool shares: set that instead",
        error.GlmRequestStartsWithSharedKv => "TF_GLM_REQUEST_STARTS=1 cannot be used with TF_KV_SHARE above 0: unset request starts or disable shared KV",
        error.GlmRequestStartsWithLearnConversations => "TF_GLM_REQUEST_STARTS=1 cannot be used with --learn-conversations: unset request starts or serve without it",
        error.GlmShareNeedsPages => "TF_KV_SHARE keeps prompts as KV pool pages: set TF_KV_PAGES=on as well",
        error.GlmShareTooMany => std.fmt.comptimePrint("TF_KV_SHARE keeps at most {d} states: set {d} or fewer", .{ kv.most_snaps, kv.most_snaps }),
        error.BadCopyPolicy => "TF_GLM_COPIES: expected 0 or 1 for verified root/context copies",
        error.InvalidEnvironment => "a TF_GLM_* setting is malformed: the error line before this one names it, its value and the accepted form",
        error.GlmKvHostNeedsPages => "TF_GLM_KV_HOST keeps the KV pool's pages in host memory: set TF_KV_PAGES=on (and --kv-cache bf16) as well",
        error.TooManyRanks => std.fmt.comptimePrint(
            "GLM-5.3-Flash on CUDA runs on at most {d} ranks: set TF_GLM_WORLD to {d} or fewer",
            .{ ranks.max_world, ranks.max_world },
        ),
        error.NotModelOpt, error.UnsupportedQuant => "GLM-5.3-Flash on CUDA reads ModelOpt NVFP4 and MIXED_PRECISION checkpoints (the log names what config.json has)",
        error.UnreadableFormat => "a projection is stored in a format GLM-5.3-Flash on CUDA does not read (the log names its module, recipe and tensors)",
        else => null,
    };
}

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expect(explain(null, error.TooManyRanks) != null);
}

test "a checkpoint refused for its format, at the entry or by the loader, is explained" {
    for ([_]anyerror{ error.NotModelOpt, error.UnsupportedQuant, error.UnreadableFormat }) |e| try std.testing.expect(explain(null, e) != null);
}

test "TF_KV_SHARE past the slots a startup plan holds is refused, and the refusal is explained" {
    try std.testing.expect(share_mod.tooMany(kv.most_snaps + 1) and explain(null, error.GlmShareTooMany) != null);
}

test "TF_KV_PAGES: auto pages from two ranks, off and 0 give windows, all and other values page like on" {
    try std.testing.expect(!try paging(null, 1, false));
    try std.testing.expect(!try paging("auto", 1, false));
    for ([_]usize{ 2, 3, 16 }) |w| {
        try std.testing.expect(try paging(null, w, false));
        try std.testing.expect(try paging("auto", w, false));
    }
    for ([_]usize{ 1, 2, 3 }) |w| {
        try std.testing.expect(try paging("on", w, false));
        try std.testing.expect(try paging("1", w, false));
        try std.testing.expect(!try paging("off", w, false));
        try std.testing.expect(!try paging("0", w, false));
        try std.testing.expect(try paging("all", w, false));
    }
}

test "TF_KV_PAGES with --kv-cache fp8: auto and off keep windows at every rank count, on and all are refused by name" {
    for ([_]usize{ 1, 2, 3, 16 }) |w| {
        try std.testing.expect(!try paging(null, w, true));
        try std.testing.expect(!try paging("auto", w, true));
        try std.testing.expect(!try paging("off", w, true));
        try std.testing.expect(!try paging("0", w, true));
        for ([_][]const u8{ "on", "1", "all" }) |v| try std.testing.expectError(error.Fp8OnPages, paging(v, w, true));
    }
    try std.testing.expect(explain(null, error.Fp8OnPages) != null);
    try std.testing.expect(explain(null, error.GlmKvHostNeedsPages) != null);
}

test "the plan's window, the context and a verify window, takes the pages of a stream's table at that capacity" {
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = @import("cuda_state.zig").page_rows, .snap_bytes = 0 };
    for ([_]usize{ 8192, 8191, 245_760, 524_288 - 33, 524_288 - 32, 524_288 - 31, 1_048_576, 2_097_152, 2_097_152 + 1 }) |context| {
        try std.testing.expectEqual(pages_mod.tableLen(capacity(context)), l.pages(context + window_rows));
    }
}

// Enough words for every offered frame pair and image, even with a small text context.
pub fn linkCapacity(cap: usize, media: ?images_mod.Limits) !usize {
    const lim = media orelse return cap;
    const frames: usize = (@as(usize, lim.max_frames) + 1) / 2;
    const pairs = std.math.mul(usize, frames, lim.max_videos) catch return error.TooManyImages;
    const spans = std.math.add(usize, pairs, lim.max_images) catch return error.TooManyImages;
    const words = std.math.mul(usize, spans, 3) catch return error.TooManyImages;
    return @max(cap, words);
}
test "a small text context still sizes the link for every offered media span" {
    try std.testing.expectEqual(@as(usize, 792), try linkCapacity(64, .{ .image_tokens = 2048, .max_images = 8, .video_tokens = 4096, .max_videos = 2, .max_frames = 256 }));
}
test "I3 native owned close: actual engine release failure retains family receipt and CUDA parent gate" {try ownedReleaseTrial(false);}
test "I3 native owned close: actual cleanup emits final typed ownership before observer destruction" {try ownedReleaseTrial(true);}
fn ownedReleaseTrial(observe:bool)!void{
    const t=std.testing;try t.expect(@hasField(Owned,"memory"));
    const Capture=struct{var final:?cuda.memory_receipt.Receipt=null;fn record(r:cuda.memory_receipt.Receipt)void{if(r.phase==.cleanup_end)final=r;}};
    if(observe)cuda.memory_receipt.Test.sink=Capture.record;defer cuda.memory_receipt.Test.sink=null;
    if(comptime @hasField(Owned,"memory")){
        const F=@import("cuda_hybrid_memory_fixture_cpu.zig").F;const Ledger=@import("cuda_hybrid_ledger.zig").Ledger;
        try t.expectEqual(@as(c_int,0),env.setenv("TF_GLM_KV_HOST","1",1));defer _=env.unsetenv("TF_GLM_KV_HOST");
        for([_]bool{false,true})|fail|{
            Capture.final=null;
            F.live=0;F.fail_free=0;var d:cuda.Driver=undefined;const e=try F.engine(&d);
            const l=try t.allocator.create(Ledger);l.*=Ledger.init(.attention,88);try l.install();try e.prepareHostGather();
            var saved=(@import("cuda_kv_host.zig").active.?).slot_of;if(fail)F.fail_free=saved.ptr;
            const own=try t.allocator.create(Owned);@memset(std.mem.asBytes(own),0);own.gpa=t.allocator;own.memory=l;own.e=e;own.link.gpa=t.allocator;own.link.buf=.{.d=&d,.ptr=0,.len=0,.borrowed=true};own.link.ids=&.{};
            own.group=.{.rank=0,.world=1,.comm=null,.lib=undefined,.aborted=.init(true)};
            own.group.lib.lib=try @import("sys").DynLib.open("libc.so.6");
            const old_retained=retained_hybrids;const ok=releaseChecked(own);
            try t.expectEqual(!fail,ok);
            if(fail){
                try t.expect(own.quarantined.load(.acquire) and !hybridGate(own));const status=l.status();try t.expect(status.retained.device>0 and !status.clean);
                // Dispose only CPU fixture backing from its exact captured original buffer. No production recovery path.
                F.fail_free=0;saved.free();try t.expectEqual(@as(usize,0),F.live);l.uninstall();t.allocator.destroy(l);own.group.lib.close();retained_hybrids=old_retained;t.allocator.destroy(own);
            }else try t.expectEqual(@as(usize,0),F.live);
            if(observe){try t.expect(Capture.final!=null);const receipt=Capture.final.?.ownership.?;try t.expect(receipt.closed);try t.expectEqual(!fail,receipt.clean);try t.expectEqual(@as(u128,88),receipt.incarnation);}
        }
    }
}
test "I3 native owned close: startup installs observer before owned CUDA and unwinds actual failure" {try startupReceiptTrial(false,false);}
test "I3 native owned close: failed startup binds load and terminal ownership receipts" {try startupReceiptTrial(true,false);}
test "I3 native owned close: attention wrapper scopes non-engine runtime buffers" {try startupReceiptTrial(false,true);}
fn startupReceiptTrial(observe:bool,workspace:bool)!void{
    const t=std.testing;const F=@import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const P=struct{var count:usize=0;var receipts:[3]cuda.memory_receipt.Receipt=undefined;fn sink(r:cuda.memory_receipt.Receipt)void{receipts[count]=r;count+=1;}};
    P.count=0;if(observe)cuda.memory_receipt.Test.sink=P.sink;defer if(observe){cuda.memory_receipt.Test.sink=null;};
    const C=struct{var classified=false;fn after(own:*Owned)!void{
        try t.expect(own.memory!=null);
        var d:cuda.Driver=undefined;const e=try F.engine(&d);defer e.deinit();var b=try cuda.DeviceBuffer.alloc(&d,16);defer b.free();
        try t.expectEqual(@as(u64,16),own.memory.?.status().live.device);if(classified)try t.expectEqual(@as(u64,16),own.memory.?.status().owners[@intFromEnum(@import("cuda_hybrid_memory.zig").Owner.workspace)].device);return error.FixtureAfterMemory;
    }};
    var config_= @import("cuda_hybrid_owner_fixture.zig").config("/fixture/NEVER_OPENED");config_.settings=.{};
    const text=try std.json.Stringify.valueAlloc(t.allocator,config_,.{});defer t.allocator.free(text);var parsed=try hybrid_config.parse(t.allocator,text);defer parsed.deinit();
    C.classified=workspace;MemoryTest.after=C.after;defer MemoryTest.after=null;var ctx:cuda.Context=undefined;
    try t.expectError(error.FixtureAfterMemory,openRank(.{.hybrid=parsed,.hybrid_until=1,.gpa=t.allocator,.io=t.io,.ctx=&ctx,.dir="/fixture/NEVER_LOADED",.o=.{.context=262144,.drafts=true},.p=.{.rank=0,.world=1},.roce_bytes=1048576,.share=0},.{}));
    var next=@import("cuda_hybrid_ledger.zig").Ledger.init(.attention,99);try next.install();next.uninstall();
    if(observe){
        try t.expectEqual(@as(usize,3),P.count);try t.expectEqual(cuda.memory_receipt.Phase.load_begin,P.receipts[0].phase);try t.expectEqual(cuda.memory_receipt.Phase.cleanup_begin,P.receipts[1].phase);try t.expectEqual(cuda.memory_receipt.Phase.cleanup_end,P.receipts[2].phase);
        const id=P.receipts[0].ownership.?.incarnation;for(P.receipts)|r|{try t.expect(r.ownership!=null);try t.expectEqual(id,r.ownership.?.incarnation);try t.expectEqualStrings("attention",r.ownership.?.role);}try t.expect(!P.receipts[0].ownership.?.closed);try t.expect(P.receipts[2].ownership.?.closed and P.receipts[2].ownership.?.clean);
    }
}
test "I3 native owned close: failed startup free immediately quarantines and propagates retained ownership" {
    const t=std.testing;const F=@import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const C=struct{var d:cuda.Driver=undefined;var held:?*Owned=null;var immediate=false;var saved:cuda.DeviceBuffer=undefined;
        fn after(own:*Owned)!void{held=own;const e=try F.engine(&d);defer e.deinit();saved=try cuda.DeviceBuffer.alloc(&d,16);F.fail_free=saved.ptr;saved.free();immediate=own.quarantined.load(.acquire);return error.FixtureAfterMemory;}
    };
    const old_retained=retained_hybrids;C.held=null;C.immediate=false;
    defer if(retained_hybrids!=old_retained){const own=C.held.?;F.fail_free=0;C.saved.free();own.memory.?.uninstall();t.allocator.destroy(own.memory.?);retained_hybrids=old_retained;t.allocator.destroy(own);};
    var config_= @import("cuda_hybrid_owner_fixture.zig").config("/fixture/NEVER_OPENED");config_.settings=.{};
    const text=try std.json.Stringify.valueAlloc(t.allocator,config_,.{});defer t.allocator.free(text);var parsed=try hybrid_config.parse(t.allocator,text);defer parsed.deinit();
    MemoryTest.after=C.after;defer MemoryTest.after=null;var ctx:cuda.Context=undefined;
    try t.expectError(error.HybridRetained,openRank(.{.hybrid=parsed,.hybrid_until=1,.gpa=t.allocator,.io=t.io,.ctx=&ctx,.dir="/fixture/NEVER_LOADED",.o=.{.context=262144,.drafts=true},.p=.{.rank=0,.world=1},.roce_bytes=1048576,.share=0},.{}));
    try t.expect(C.immediate);try t.expect(C.held.?.memory.?.status().retained.device>0);
}

test "I3 native async close: completed upload spares retire before final owned balance" {
    const t=std.testing;const F=@import("cuda_hybrid_memory_fixture_cpu.zig").F;const Ledger=@import("cuda_hybrid_ledger.zig").Ledger;
    const C=struct{
        fn create(p:*cuda.abi.Event,_:c_uint)callconv(.c)c_int{p.*=@ptrFromInt(78);return 0;}
        fn record(_:cuda.abi.Event,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
        fn event(_:cuda.abi.Event)callconv(.c)c_int{return 0;}
        fn copy(_:u64,_:?*const anyopaque,_:usize,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
    };
    F.live=0;F.fail_free=0;var d:cuda.Driver=undefined;const e=try F.engine(&d);
    d.api.cuEventCreate=C.create;d.api.cuEventRecord=C.record;d.api.cuEventQuery=C.event;d.api.cuEventSynchronize=C.event;d.api.cuMemcpyHtoDAsync_v2=C.copy;
    const l=try t.allocator.create(Ledger);l.*=Ledger.init(.attention,943);try l.install();
    cuda.copy_source.setCurrent(@ptrFromInt(943));defer cuda.copy_source.setCurrent(null);
    cuda.copy_source.setSparesForTest(.{.count=4,.bytes=65536});defer cuda.copy_source.setSparesForTest(.{.count=0,.bytes=0});
    const b:cuda.DeviceBuffer=.{.d=&d,.ptr=0xCC000,.len=16,.borrowed=true};try b.uploadAsync(0,&.{1,2,3,4},@ptrFromInt(7));try cuda.copy_source.synchronizeStream(&d,@ptrFromInt(7));
    try t.expectEqual(@as(usize,1),cuda.copy_source.spareCount());
    const own=try t.allocator.create(Owned);@memset(std.mem.asBytes(own),0);own.gpa=t.allocator;own.memory=l;own.e=e;own.link.gpa=t.allocator;own.link.buf=.{.d=&d,.ptr=0,.len=0,.borrowed=true};own.link.ids=&.{};
    own.group=.{.rank=0,.world=1,.comm=null,.lib=undefined,.aborted=.init(true)};own.group.lib.lib=try @import("sys").DynLib.open("libc.so.6");
    const previous=retained_hybrids;const ok=releaseChecked(own);
    if(!ok){cuda.copy_source.dropSpares(&d,@ptrFromInt(943));l.uninstall();t.allocator.destroy(l);own.group.lib.close();retained_hybrids=previous;t.allocator.destroy(own);}
    try t.expect(ok);try t.expectEqual(@as(usize,0),F.live);try t.expectEqual(@as(usize,0),cuda.copy_source.spareCount());
}

test "I3 native async startup: clean failure disposes completed sources before ledger finalization" {
    const t=std.testing;const F=@import("cuda_hybrid_memory_fixture_cpu.zig").F;
    const C=struct{var d:cuda.Driver=undefined;var own:?*Owned=null;
        fn create(p:*cuda.abi.Event,_:c_uint)callconv(.c)c_int{p.*=@ptrFromInt(79);return 0;}
        fn record(_:cuda.abi.Event,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
        fn event(_:cuda.abi.Event)callconv(.c)c_int{return 0;}
        fn copy(_:u64,_:?*const anyopaque,_:usize,_:cuda.abi.Stream)callconv(.c)c_int{return 0;}
        fn after(o:*Owned)!void{
            own=o;const e=try F.engine(&d);defer e.deinit();d.api.cuEventCreate=create;d.api.cuEventRecord=record;d.api.cuEventQuery=event;d.api.cuEventSynchronize=event;d.api.cuMemcpyHtoDAsync_v2=copy;
            const b:cuda.DeviceBuffer=.{.d=&d,.ptr=0xCC000,.len=16,.borrowed=true};try b.uploadAsync(0,&.{1,2},@ptrFromInt(7));try cuda.copy_source.synchronizeStream(&d,@ptrFromInt(7));return error.FixtureAfterMemory;
        }
    };
    F.live=0;F.fail_free=0;cuda.copy_source.setCurrent(@ptrFromInt(945));defer cuda.copy_source.setCurrent(null);cuda.copy_source.setSparesForTest(.{.count=4,.bytes=65536});defer cuda.copy_source.setSparesForTest(.{.count=0,.bytes=0});
    var config_=@import("cuda_hybrid_owner_fixture.zig").config("/fixture/NEVER_OPENED");config_.settings=.{};const text=try std.json.Stringify.valueAlloc(t.allocator,config_,.{});defer t.allocator.free(text);var parsed=try hybrid_config.parse(t.allocator,text);defer parsed.deinit();
    const previous=retained_hybrids;MemoryTest.after=C.after;defer MemoryTest.after=null;var ctx:cuda.Context=undefined;
    const result=openRank(.{.hybrid=parsed,.hybrid_until=1,.gpa=t.allocator,.io=t.io,.ctx=&ctx,.dir="/fixture/NEVER_LOADED",.o=.{.context=262144,.drafts=true},.p=.{.rank=0,.world=1},.roce_bytes=1048576,.share=0},.{});
    if(retained_hybrids!=previous){const own=C.own.?;cuda.copy_source.dropSpares(&C.d,@ptrFromInt(945));own.memory.?.uninstall();t.allocator.destroy(own.memory.?);retained_hybrids=previous;t.allocator.destroy(own);}
    try t.expectError(error.FixtureAfterMemory,result);try t.expectEqual(@as(usize,0),F.live);
}

// AI-assisted integration: the exact options object later passed to the actual Engine.init, before NCCL/device access.
pub const StopTest=if(@import("builtin").is_test)struct {
    pub fn nativeEngineOptions()!void {
        const t=std.testing;
        const Probe=struct {var requested=std.atomic.Value(bool).init(false);var calls:usize=0;
            fn capture(o:engine.Options)!void {calls+=1;if(o.cancel_requested!=&requested)return error.MissingCancellationToken;
                if(comptime @hasField(engine.Options,"setup_until")){if(o.setup_until!=123456789)return error.MissingStartupBound;}else return error.MissingStartupBound;
                return error.CpuEngineBoundary;}
        };
        var config_= @import("cuda_hybrid_owner_fixture.zig").config("/CPU/NEVER_OPENED");config_.settings=.{};
        const text=try std.json.Stringify.valueAlloc(t.allocator,config_,.{});defer t.allocator.free(text);var parsed=try hybrid_config.parse(t.allocator,text);defer parsed.deinit();
        Probe.calls=0;Probe.requested.store(false,.release);MemoryTest.engine_options=Probe.capture;defer MemoryTest.engine_options=null;var ctx:cuda.Context=undefined;
        try t.expectError(error.CpuEngineBoundary,openRank(.{.hybrid=parsed,.hybrid_until=123456789,.gpa=t.allocator,.io=t.io,.ctx=&ctx,.dir="/CPU/NEVER_LOADED",.o=.{.context=262144,.drafts=true,.cancel_requested=&Probe.requested},.p=.{.rank=0,.world=1},.roce_bytes=1048576,.share=0},.{}));
        try t.expectEqual(@as(usize,1),Probe.calls);try t.expect(!Probe.requested.load(.acquire));
        var next=@import("cuda_hybrid_ledger.zig").Ledger.init(.attention,606);try next.install();next.uninstall();
    }
}else struct{};
