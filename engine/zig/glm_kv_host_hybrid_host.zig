//! I2 actual native DSA/MTP host-KV consumers. Recording driver only; no CUDA math/device qualification.
//! Native host-KV: russellmm/PR1795; native MTP stride: Bizuayeu/PR2173.
//! Private I2 regression/adapter: Scott Quarrier with AI assistance (OpenAI gpt-6.1-sol).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;

/// Recording-only seam for the finite524K workspace tests; no CUDA arithmetic or device access.
pub fn tile524(rows:usize,part_rows:usize,at:usize) !struct {calls:usize,copies:usize,tokens:u64,counts:u64,qa:u64,base_qa:u64,base_tokens:u64,base_counts:u64} {
    var f:Fixture=undefined;f.init();defer glm.kv_host.active=null;
    glm.kv_host.active.?.cap=32*2051;
    try glm.dsa_block.attend(f.k,f.stream(),f.l,f.c,f.s,0,rows,0x70100000,at,9,part_rows);
    return .{.calls=R.attention,.copies=R.sparse_copies,.tokens=R.sparse_tokens,.counts=R.sparse_counts,.qa=R.qa,.base_qa=f.s.qa,.base_tokens=f.s.tokens,.base_counts=f.s.counts};
}

const R = struct {
    var direct_reads: usize = 0;
    var visible: usize = 0;
    var sparse_copies: usize = 0;
    var attention: usize = 0;
    var cache: u64 = 0;
    var pt: u64 = 0;
    var ps: i64 = 0;
    var at: u64 = 0;
    var qa: u64 = 0;
    var gathered_at: u64 = 0;
    var gathered_rows: i32 = 0;
    var gathered_pt: u64 = 0;
    var gathered_page_rows: i64 = 0;
    var over_flag: [4]i32 align(16) = @splat(0);
    const host_lc: u64 = 0x10000000;
    const device_slots: u64 = 0x20000000;
    const chunks: usize = 56; // glue.Kernels field order, filled via field names below
    var chunk16: usize = 0;
    var chunk32: usize = 0;
    var pool_id: usize = 0;
    var scores_id: usize = 0;
    var pool_ps: i64 = 0;
    var score_ps: i64 = 0;
    var replays: usize = 0;
    var replay_rows: i32 = 0;
    var conv_id: usize = 0;
    var conv_keep: i64 = 0;
    var sparse_at: u64 = 0;
    var sparse_table: u64 = 0;
    var sparse_page_rows: i64 = 0;
    var sparse_rows: i32 = 0;
    var sparse_tokens: u64 = 0;
    var sparse_counts: u64 = 0;
    var add_delta: i32 = 0;

    fn arg(comptime T: type, args: ?[*]?*anyopaque, n: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[n].?))).*;
    }
    fn reset() void {
        direct_reads = 0;
        replays = 0;
        replay_rows = 0;
        conv_keep = 0;
        visible = 0;
        sparse_copies = 0;
        attention = 0;
    }
    fn lookup(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) c_int {
        const s = std.mem.span(name);
        const names = [_][]const u8{ "tf_glm_add_i32", "tf_glm_kvh_rows", "tf_glm_kvh_mark", "tf_glm_kvh_prev_mark", "tf_glm_kvh_copy_cached", "tf_glm_kvh_remap", "tf_glm_kvh_unmark", "tf_glm_kvh_keep", "tf_glm_dsa_tokens", "tf_glm_kvh_copy", "tf_glm_kvh_reset" };
        for (names, 0..) |n, i| if (std.mem.eql(u8, s, n)) {
            out.* = @ptrFromInt(1000 + i);
            return 0;
        };
        return 1;
    }
    var invalidations: usize = 0;
    fn clear(_: u64, _: u32, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        if (n == 16) invalidations += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        if (id == chunk16 or id == chunk32) {
            attention += 1;
            qa = arg(u64, args, 0);
            cache = arg(u64, args, 1);
            at = arg(u64, args, 2);
            pt = arg(u64, args, 13);
            ps = arg(i64, args, 14);
            if (cache == host_lc) direct_reads += 1;
        }
        if (id == 1000) add_delta = arg(i32, args, 1);
        if (id == 1002) {
            sparse_tokens = arg(u64, args, 0);
            sparse_counts = arg(u64, args, 1);
        }
        if (id == 1003) {
            sparse_table = arg(u64, args, 3);
            sparse_page_rows = arg(i64, args, 4);
            sparse_at = arg(u64, args, 5);
            sparse_rows = arg(i32, args, 6);
        }
        if (id == 1001) {
            visible += 1;
            gathered_pt = arg(u64, args, 1);
            gathered_page_rows = arg(i64, args, 2);
            gathered_at = arg(u64, args, 3);
            gathered_rows = arg(i32, args, 4);
        }
        if (id == 990) {
            replays += 1;
            replay_rows = arg(i32, args, 9);
        }
        if (id == conv_id) conv_keep = arg(i64, args, 2);
        if (id == pool_id) pool_ps = arg(i64, args, 7);
        if (id == scores_id) score_ps = arg(i64, args, 13);
        if (id == 1004 or id == 1009) sparse_copies += 1;
        return 0;
    }
};
const Fixture = struct {
    d: cuda.Driver,
    glue: glm.glue.Kernels,
    k: glm.dsa_block.Kernels,
    l: glm.dsa_block.Layer,
    c: glm.dsa_block.Caches,
    s: glm.dsa_block.Scratch,
    fn init(f: *Fixture) void {
        R.reset();
        f.d = undefined;
        f.d.api.cuLaunchKernel = R.launch;
        f.d.api.cuModuleGetFunction = R.lookup;
        f.d.api.cuMemsetD32Async = R.clear;
        inline for (@typeInfo(glm.glue.Kernels).@"struct".field_names, 0..) |name, i| {
            @field(f.glue, name) = .{ .d = &f.d, .handle = @ptrFromInt(i + 1) };
            if (comptime std.mem.eql(u8, name, "chunks_16")) R.chunk16 = i + 1;
            if (comptime std.mem.eql(u8, name, "chunks_32")) R.chunk32 = i + 1;
            if (comptime std.mem.eql(u8, name, "pool_keys")) R.pool_id = i + 1;
            if (comptime std.mem.eql(u8, name, "scores")) R.scores_id = i + 1;
            if (comptime std.mem.eql(u8, name, "conv_shift")) R.conv_id = i + 1;
        }
        f.k = .{ .set = .{ .glue = &f.glue }, .ops = .{ .d = &f.d, .handle = null } };
        f.l = .{ .proj = .{ .bf16 = 0x30000000 }, .q_norm = 0x30010000, .kv_norm = 0x30020000, .q_b = .{ .bf16 = 0x30030000 }, .wk = 0x30040000, .wv = 0x30050000, .o = .{ .bf16 = 0x30060000 }, .index = .{ .kw = .{ .bf16 = 0x30070000 }, .gate = 0x30080000, .ln_w = 0x30090000, .ln_b = 0x300a0000, .ape = 0x300b0000, .qb = .{ .bf16 = 0x300c0000 }, .heads = 32, .d = 128 }, .heads = 16, .qk = 256, .dv = 256, .lw = 512, .q_lora = 1536, .dims = 4096, .eps = 1e-6, .dense_limit = 2048 };
        f.c = .{ .lc = R.host_lc, .ik = 0x40000000, .ig = 0x40010000, .pk = 0x40020000, .pk_rows = 4098, .pt = 0x40030000, .ps = 512 * 1024 * 3, .pks = 512 * 64 * 3 };
        f.s = .{ .dp = 0x50000000, .qr = 0x50010000, .xs_qr = 0x50020000, .lat = 0x50030000, .xs_lat = 0x50040000, .q = 0x50050000, .ikr = 0x50060000, .igr = 0x50070000, .router_part = 0x50080000, .qa = 0x50090000, .ol = 0x500a0000, .vn = 0x500b0000, .qi = 0x500c0000, .partials = .{ .po = 0x500d0000, .pm = 0x500e0000, .pl = 0x500f0000 }, .dummy = 0x50100000, .select = .{ .scores = 0x50110000, .ls = 0x50120000, .li = 0x50130000, .pools = 0x50140000, .at = 0x50150000 }, .tokens = 0x50160000, .counts = 0x50170000, .mm_part = 0x50180000, .at = 0x50190000 };
        var g: glm.kv_host.Gather = undefined;
        inline for (.{ "slot_of", "list", "slots", "n", "prev_of", "prev_rows", "prev_n", "cache" }, 0..) |name, i| {
            @field(g, name) = .{ .d = &f.d, .ptr = if (comptime std.mem.eql(u8, name, "slots")) R.device_slots else 0x60000000 + i * 0x10000, .len = 16 };
        }
        g.cap = 8192;
        g.row_bytes = 1024;
        g.tags = @splat(0);
        g.over_dev = 0x60100000;
        R.over_flag = @splat(0);
        g.over = .{ .d = &f.d, .bytes = std.mem.sliceAsBytes(&R.over_flag) };
        glm.kv_host.active = g;
    }
    fn stream(f: *Fixture) cuda.Stream {
        return .{ .d = &f.d, .handle = null };
    }
    fn last(f: *Fixture, rows: usize, host_pos: usize) !void {
        try glm.dsa_block.blockLastRow(f.k, f.stream(), f.l, f.c, f.s, 0x70000000, 4096, rows, 0x70100000, host_pos, 8, 0x70200000);
    }
};

extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
extern "c" fn unsetenv([*:0]const u8) c_int;
const HostEnv = struct {
    prior: [2]?[:0]u8,
    const names = [_][:0]const u8{ "TF_GLM_KV_HOST", "TF_GLM_KV_HOST_SHARED" };
    fn init() !HostEnv {
        var e: HostEnv = .{ .prior = .{ null, null } };
        for (names, 0..) |name, i| if (std.c.getenv(name)) |p| {
            e.prior[i] = try t.allocator.dupeSentinel(u8, std.mem.span(p), 0);
        };
        try t.expectEqual(@as(c_int, 0), setenv(names[0], "1", 1));
        try t.expectEqual(@as(c_int, 0), setenv(names[1], "0", 1));
        return e;
    }
    fn deinit(e: *HostEnv) void {
        for (names, e.prior) |name, p| if (p) |v| {
            _ = setenv(name, v, 1);
            t.allocator.free(v);
        } else {
            _ = unsetenv(name);
        };
    }
};
const PageFixture = struct {
    var next: u64 = 0x80000000;
    var hosts: usize = 0;
    var host_frees: usize = 0;
    var pinned: [8 * 1024 * 1024]u8 align(16) = undefined; // CPU-only stand-in, never pinned/mapped
    var fail_map = true;
    d: cuda.Driver,
    e: glm.engine.Engine,
    layers: [2]glm.weights.Layer,
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += std.mem.alignForward(u64, n + 256, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn put(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn host(out: *?*anyopaque, n: usize, flags: c_uint) callconv(.c) c_int {
        if (n > pinned.len or flags != cuda.abi.host_alloc_portable | cuda.abi.host_alloc_devicemap) return 1;
        hosts += 1;
        out.* = &pinned;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) c_int {
        host_frees += 1;
        return 0;
    }
    fn map(out: *u64, _: ?*anyopaque, _: c_uint) callconv(.c) c_int {
        if (fail_map) return cuda.abi.error_not_ready;
        out.* = R.host_lc;
        return 0;
    }
    fn init(f: *PageFixture) !void {
        hosts = 0;
        host_frees = 0;
        fail_map = true;
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = alloc;
        f.d.api.cuMemFree_v2 = free;
        f.d.api.cuMemsetD8_v2 = zero;
        f.d.api.cuStreamSynchronize = sync;
        f.d.api.cuMemsetD32Async = R.clear;
        f.d.api.cuMemcpyHtoD_v2 = put;
        f.d.api.cuMemHostAlloc = host;
        f.d.api.cuMemFreeHost = hostFree;
        f.d.api.cuMemHostGetDevicePointer_v2 = map;
        f.e = undefined;
        f.e.gpa = t.allocator;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = 4096;
        if (glm.kv_host.active) |*g| {
            g.cap = f.e.capacity; g.row_bytes = 32; // explicit small page-fixture geometry, not the 512-latent DSA fixture
            inline for (.{ "slot_of", "list", "slots", "n", "prev_of", "prev_rows", "prev_n", "cache" }) |name| @field(g.*,name).d = &f.d;
        }
        f.e.settings = .{};
        f.e.seqs = .empty;
        f.e.pages = null;
        f.e.owns_host_gather = glm.kv_host.active != null; // this fixture deliberately installs its same-driver native Gather
        f.e.routed = null;
        f.e.images = null;
        f.e.w = undefined;
        f.e.w.config = .{ .layers = 2, .kv_lora = 16, .i_dim = 8, .hidden = 64 };
        f.e.w.share = try glm.shard.Share.of(&f.e.w.config, 0, 1);
        f.e.share = f.e.w.share;
        f.layers[0].mixer = .{ .mla = undefined };
        f.layers[1].mixer = .{ .kda = undefined };
        f.e.w.layers = &f.layers;
        try glm.pages.ownSequence(&f.e);
    }
    fn deinit(f: *PageFixture) void {
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            t.allocator.destroy(s);
        };
        f.e.seqs.deinit(t.allocator);
    }
};

fn commitCase(keep: usize, invalid: bool) !void {
    var f: Fixture = undefined;
    f.init();
    defer glm.kv_host.active = null;
    var parity = [_]u1{ 0, 0 };
    var rings = [_]u64{ 0x90000000, 0x90010000 };
    var st: glm.forward.State = undefined;
    st.arena = .{ .buf = .{ .d = &f.d, .ptr = 0, .len = 0 } };
    st.capacity = 4096;
    st.pos = 100;
    st.pending = 8;
    st.prompt = false;
    st.cur = &parity;
    st.kda_layers = 2;
    st.heads = 1;
    st.rec = 0x91000000;
    st.rec_layer = 65536;
    st.conv = 0x92000000;
    st.conv_c = 128;
    st.proj = 0x93000000;
    st.proj_layer = 8192;
    st.proj_row = 256;
    st.save_k = 0x94000000;
    st.save_v = 0x95000000;
    st.save_g = 0x96000000;
    st.save_b = 0x97000000;
    st.pos_dev = 0x98000000;
    st.mtp_pos_dev = 0x98010000;
    st.mtp_len = 108;
    st.caches = &.{};
    st.rings = &rings;
    st.ring_row = 1;
    st.base = 0;
    st.mtp_base = 0;
    st.mtp_caches = f.c;
    var b: glm.forward.Buffers = undefined;
    b.prefill = false;
    b.moe.hybrid = null; // Exact I1 adds a provider read to commit; this fixture selects native-local.
    var m: glm.forward.Modules = undefined;
    m.glue = f.glue;
    m.kda.replay_layers = .{ .d = &f.d, .handle = @ptrFromInt(990) };
    m.index = null;
    m.tiled = false;
    m.packs = null;
    m.copy_rows = f.glue.conv_shift;
    var ranks = [_]glm.forward.Rank{.{ .w = undefined, .st = &st, .b = &b }};
    const forward: glm.forward.Forward = .{ .d = &f.d, .m = &m, .stream = f.stream(), .ranks = &ranks, .exchange = undefined };
    if (invalid) {
        try t.expectError(error.BadCommit, forward.commit(keep));
        try t.expectEqual(@as(usize, 100), st.pos);
        try t.expectEqual(@as(usize, 8), st.pending);
        try t.expectEqual(@as(usize, 0), R.replays);
    } else {
        try forward.commit(keep);
        try t.expectEqual(@as(usize, 100) + keep, st.pos);
        try t.expectEqual(@as(usize, 0), st.pending);
        try t.expectEqualSlices(u1, &.{ 1, 1 }, st.cur);
        try t.expectEqual(@as(usize, @intFromBool(keep < 8)), R.replays);
        if (keep < 8) try t.expectEqual(@as(i32, @intCast(keep)), R.replay_rows);
        try t.expectEqual(@as(i64, @intCast(keep)), R.conv_keep);
        try st.restoreHost(f.stream(), 100, 100, &.{ 0, 0 }, .{ 0, 0 });
        try t.expectEqual(@as(usize, 100), st.pos);
        try t.expectEqual(@as(usize, 100), st.mtp_len);
        try t.expectEqualSlices(u1, &.{ 0, 0 }, st.cur);
        try t.expectEqual(f.c.lc, st.mtp_caches.lc);
        try t.expectEqual(f.c.pks, st.mtp_caches.pks);
    }
}

test "I2 host KV: full accepted native window commits KDA and convolution" {
    try commitCase(8, false);
}
test "I2 host KV: partial accept replays KDA before rollback restoration" {
    try commitCase(3, false);
    try commitCase(1, false);
}
test "I2 host KV: rejected invalid commits leave native state unchanged" {
    try commitCase(0, true);
    try commitCase(9, true);
}

pub fn main() !void {
    var f: Fixture = undefined;
    f.init();
    defer glm.kv_host.active = null;
    var e: glm.engine.Engine = undefined;
    e.group = null;
    e.routed = null;
    R.over_flag[0] = 1;
    try t.expectError(error.GlmKvHostSlots, e.checkGathers());
    try t.expectEqual(@as(i32, 0), R.over_flag[0]);
    try e.checkGathers();
    std.debug.print("I2_OVERFLOW_NAMED_RESET_OK\n", .{});
}

test "I2 host KV: native page owner transfer invalidates cache before reuse" {
    var recorded: Fixture = undefined;
    recorded.init();
    defer glm.kv_host.active = null;
    var env = try HostEnv.init();
    defer env.deinit();
    var f: PageFixture = undefined;
    try f.init();
    defer f.deinit();
    PageFixture.fail_map = false;
    const layout = try glm.pages.layout(t.allocator, &f.e, 0);
    const p = try glm.pages.Pages.init(t.allocator, &f.e, layout, 2, 8, true);
    defer p.deinit();
    try t.expect(p.arena == null and p.host != null and p.pk_arena != null);
    try t.expectEqual(@as(usize, 512), layout.page_tokens);
    const cache = f.e.seqs.items[0].?.st.caches[0];
    try t.expectEqual(R.host_lc, cache.lc);
    try t.expect(cache.pk != cache.lc and cache.pks != 0 and cache.pks != cache.ps);
    R.invalidations = 0;
    const ids = try p.open(0, 2, 7);
    const first = ids[0];
    try t.expectEqual(@as(usize, 1), R.invalidations);
    p.close(0);
    try t.expectEqual(@as(usize, 2), R.invalidations);
    const next = try p.open(1, 2, 7);
    try t.expectEqual(first, next[0]);
    try t.expectEqual(@as(usize, 3), R.invalidations);
    try t.expectEqual(@as(usize, 0), f.e.seqs.items[0].?.st.pages.len);
    try t.expect(f.e.seqs.items[1].?.st.pages.len > 0);
    p.close(1);
    try t.expect(p.pool.idle());
}

test "I2 host KV: native scalar and shared segment reads use device slots" {
    var f: Fixture = undefined;
    f.init();
    defer glm.kv_host.active = null;
    for ([_]usize{ 1, 4, 17 }) |rows| for ([_]usize{ 31, 10000 }) |pos| {
        R.reset();
        try glm.dsa_block.attend(f.k, f.stream(), f.l, f.c, f.s, 3, rows, 0x70100000, pos, 8, 512);
        try t.expectEqual(@as(usize, 0), R.direct_reads);
        try t.expectEqual(R.device_slots, R.cache);
        try t.expectEqual(f.s.qa + 3 * f.l.heads * f.l.lw * 2, R.qa);
        try t.expectEqual(@as(u64, 0), R.pt);
    };
}

test "I2 host KV: native host-page map failure frees allocated host pages" {
    var env = try HostEnv.init();
    defer env.deinit();
    try t.expectEqual(@as(c_int, 0), setenv("TF_GLM_KV_HOST_SHARED", "1", 1));
    var isolated = try HostEnv.init();
    defer isolated.deinit();
    try t.expect(!glm.kv_host.shared());
    var f: PageFixture = undefined;
    try f.init();
    defer f.deinit();
    const before = cuda.usage(false).host;
    const layout = try glm.pages.layout(t.allocator, &f.e, 0);
    try t.expectError(error.NotReady, glm.pages.Pages.init(t.allocator, &f.e, layout, 2, 8, true));
    try t.expectEqual(@as(usize, 1), PageFixture.hosts);
    try t.expectEqual(PageFixture.hosts, PageFixture.host_frees);
    try t.expectEqual(before, cuda.usage(false).host);
}

test "I2 host KV: scalar and shared attend refuse missing gather" {
    var f: Fixture = undefined;
    f.init();
    glm.kv_host.active = null;
    try t.expectError(error.GlmKvHostUnavailable, glm.dsa_block.attend(f.k, f.stream(), f.l, f.c, f.s, 3, 2, 0x70100000, 10000, 8, 512));
    try t.expectEqual(@as(usize, 0), R.attention);
}

test "I2 host KV: missing gather refuses before native MTP launch" {
    var f: Fixture = undefined;
    f.init();
    glm.kv_host.active = null;
    try t.expectError(error.GlmKvHostUnavailable, f.last(7, 31));
    try t.expectEqual(@as(usize, 0), R.attention);
}

test "I2 host KV: actual MTP sparse last-row consumes selected device rows" {
    var f: Fixture = undefined;
    f.init();
    defer glm.kv_host.active = null;
    try f.last(7, 10000);
    try t.expectEqual(@as(usize, 0), R.direct_reads);
    try t.expectEqual(@as(usize, 1), R.sparse_copies);
    try t.expectEqual(R.device_slots, R.cache);
    try t.expectEqual(@as(u64, 0), R.pt);
    try t.expectEqual(@as(i64, 0), R.ps);
    try t.expectEqual(f.s.qa + 6 * f.l.heads * f.l.lw * 2, R.qa);
    try t.expectEqual(@as(i64, @intCast(f.c.pks)), R.pool_ps);
    try t.expectEqual(@as(i64, @intCast(f.c.pks)), R.score_ps);
    try t.expectEqual(f.s.at, R.sparse_at);
    try t.expectEqual(f.c.pt, R.sparse_table);
    try t.expectEqual(@as(i64, @intCast(f.c.ps / (f.l.lw * 2))), R.sparse_page_rows);
    try t.expectEqual(@as(i32, 1), R.sparse_rows);
    try t.expectEqual(f.s.tokens, R.sparse_tokens);
    try t.expectEqual(f.s.counts, R.sparse_counts);
    try t.expectEqual(@as(i32, 6), R.add_delta);
}

test "I2 host KV: actual MTP dense last-row consumes gathered device rows" {
    var f: Fixture = undefined;
    f.init();
    defer glm.kv_host.active = null;
    try f.last(7, 31);
    try t.expectEqual(@as(usize, 0), R.direct_reads);
    try t.expectEqual(@as(usize, 1), R.visible);
    try t.expectEqual(R.device_slots, R.cache);
    try t.expectEqual(@as(u64, 0), R.pt);
    try t.expectEqual(@as(i64, 0), R.ps);
    try t.expectEqual(f.s.qa + 6 * f.l.heads * f.l.lw * 2, R.qa);
    try t.expectEqual(f.s.at, R.at);
    try t.expectEqual(f.s.at, R.gathered_at);
    try t.expectEqual(@as(i32, 1), R.gathered_rows);
    try t.expectEqual(f.c.pt, R.gathered_pt);
    try t.expectEqual(@as(i64, @intCast(f.c.ps / (f.l.lw * 2))), R.gathered_page_rows);
}

// Actual Engine.deinit and native Gather allocation/free; CUDA calls allocate ordinary CPU memory only.
const Lifetime = struct {
    var live: usize = 0;
    var drains: usize = 0;
    var attempts: usize = 0;
    var fail_at: usize = std.math.maxInt(usize);
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        attempts+=1;if(attempts==fail_at) return cuda.abi.error_not_ready;
        const p=std.c.malloc(@max(n,1)) orelse return 2;
        out.*=@intFromPtr(p);live+=1;return 0;
    }
    fn free(p: u64) callconv(.c) c_int {if(p!=0){std.c.free(@ptrFromInt(p));live-=1;}return 0;}
    fn host(out: *?*anyopaque,n: usize,_: c_uint) callconv(.c) c_int {var p: u64=0;const rc=alloc(&p,n);out.*=@ptrFromInt(p);return rc;}
    fn hfree(p: ?*anyopaque) callconv(.c) c_int {return free(@intFromPtr(p));}
    fn map(out: *u64,p: ?*anyopaque,_: c_uint) callconv(.c) c_int {out.*=@intFromPtr(p);return 0;}
    fn clear(p: u64,v: u32,n: usize) callconv(.c) c_int {@memset(@as([*]u32,@ptrFromInt(p))[0..n],v);return 0;}
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {drains+=1;return 0;}
    fn stream(_: cuda.abi.Stream) callconv(.c) c_int {return 0;}
    fn module(_: cuda.abi.Module) callconv(.c) c_int {return 0;}
    fn event(_: cuda.abi.Event) callconv(.c) c_int {return 0;}
    fn bind(comptime T: type,p: *T,d: *cuda.Driver) void {
        if(T==*const cuda.Driver){p.*=d;return;}
        if(T==std.mem.Allocator){p.*=t.allocator;return;}
        switch(@typeInfo(T)){
            .@"struct"=>if(comptime @typeInfo(T).@"struct".layout!=.@"packed") {inline for(@typeInfo(T).@"struct".field_names) |name| bind(@TypeOf(@field(p.*,name)),&@field(p.*,name),d);},
            .array=>for(p) |*v| bind(@TypeOf(v.*),v,d),
            else=>{},
        }
    }
    fn init(d: *cuda.Driver) !*glm.engine.Engine {
        d.*=undefined;d.api.cuMemAlloc_v2=alloc;d.api.cuMemFree_v2=free;d.api.cuMemHostAlloc=host;d.api.cuMemFreeHost=hfree;d.api.cuMemHostGetDevicePointer_v2=map;d.api.cuMemsetD32_v2=clear;
        d.api.cuStreamSynchronize=sync;d.api.cuStreamDestroy_v2=stream;d.api.cuModuleUnload=module;d.api.cuEventDestroy_v2=event;
        const e=try t.allocator.create(glm.engine.Engine);@memset(std.mem.asBytes(e),0);bind(glm.engine.Engine,e,d);
        e.backend=try t.allocator.create(@typeInfo(@TypeOf(e.backend)).pointer.child);
        e.capacity=512;e.w.config=.{.layers=0,.kv_lora=16,.i_dim=8,.hidden=64};
        return e;
    }
    fn prepare(e: *glm.engine.Engine) !void {
        return e.prepareHostGather();
    }
};

test "I2 host KV: actual engine teardown releases its native singleton gather" {
    var env=try HostEnv.init();defer env.deinit();
    try t.expect(glm.kv_host.active==null);Lifetime.live=0;Lifetime.drains=0;
    var d: cuda.Driver=undefined;const e=try Lifetime.init(&d);
    defer if(glm.kv_host.active) |*g| {g.deinit();glm.kv_host.active=null;};
    try Lifetime.prepare(e);try t.expect(glm.kv_host.active!=null);try t.expectEqual(@as(usize,9),Lifetime.live);
    e.deinit();
    try t.expect(glm.kv_host.active==null);
    try t.expectEqual(@as(usize,0),Lifetime.live);
    try t.expect(Lifetime.drains>0);
}

test "I2 host KV: same owner reuses gather while foreign engine refuses without mutation" {
    var env=try HostEnv.init();defer env.deinit();
    try t.expect(glm.kv_host.active==null);Lifetime.attempts=0;Lifetime.fail_at=std.math.maxInt(usize);
    var d: cuda.Driver=undefined;const first=try Lifetime.init(&d);const other=try Lifetime.init(&d);
    try Lifetime.prepare(first);const slots=glm.kv_host.active.?.slots.ptr;const attempts=Lifetime.attempts;
    try Lifetime.prepare(first);try t.expectEqual(attempts,Lifetime.attempts);
    try t.expectError(error.GlmKvHostBusy,Lifetime.prepare(other));try t.expect(!other.owns_host_gather);
    other.deinit();try t.expectEqual(slots,glm.kv_host.active.?.slots.ptr);
    try t.expectEqual(@as(c_int,0),setenv("TF_GLM_KV_HOST","0",1)); // ownership must not depend on a later env read
    first.deinit();try t.expect(glm.kv_host.active==null);try t.expectEqual(@as(usize,0),Lifetime.live);
}

test "I2 host KV: real gather construction failures unwind before publishing an owner" {
    var env=try HostEnv.init();defer env.deinit();
    const before=cuda.usage(false);
    for(1..10) |fail| {
        Lifetime.attempts=0;Lifetime.fail_at=fail;
        var d: cuda.Driver=undefined;const e=try Lifetime.init(&d);
        try t.expectError(error.NotReady,Lifetime.prepare(e));
        try t.expect(!e.owns_host_gather and glm.kv_host.active==null);
        try t.expectEqual(@as(usize,0),Lifetime.live);e.deinit();
    }
    Lifetime.fail_at=std.math.maxInt(usize);
    try t.expectEqual(before.device,cuda.usage(false).device);try t.expectEqual(before.host,cuda.usage(false).host);
}

test "I2 host KV: post-image startup failure reaches actual engine gather cleanup" {
    var env=try HostEnv.init();defer env.deinit();
    const C=struct {
        fn after(e: *glm.engine.Engine) !void {try e.prepareHostGather();return error.FixtureRoomRefusal;}
        fn open(d: *cuda.Driver) !void {
            const e=try Lifetime.init(d);errdefer e.deinit();
            return glm.open_owner.finish(e,e,after); // unchanged native open owner, followed by real Engine destruction
        }
    };
    var d: cuda.Driver=undefined;try t.expectError(error.FixtureRoomRefusal,C.open(&d));
    try t.expect(glm.kv_host.active==null);try t.expectEqual(@as(usize,0),Lifetime.live);
}

test "I2 host KV: nonquiet native Session close retains host gather until owner is proven quiet" {
    var env=try HostEnv.init();defer env.deinit();
    var d: cuda.Driver=undefined;const e=try Lifetime.init(&d);try Lifetime.prepare(e);const held=Lifetime.live;
    const F=struct {fn record(_: cuda.abi.Event,_: cuda.abi.Stream) callconv(.c) c_int {return 0;}fn query(_: cuda.abi.Event) callconv(.c) c_int {return cuda.abi.error_not_ready;}};
    d.api.cuEventRecord=F.record;d.api.cuEventQuery=F.query;
    const cfg: glm.moe.hybridConfig.Config=.{.revision=3,.role=.attention,.session="00000000000000000000000000000001",.model_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",.master=.{"127.0.0.1","127.0.0.1"},.ports=.{29550,29551},.traffic_class=104,.verbs_device="fixture",.roce_image="fixture",.setup_ms=3000,.transaction_ms=3000,.close_ms=1,.served_context=262144,.resident_users=8,.mtp=true,.settings=.{}};
    const session=try glm.moe.hybridForward.Session.create(t.allocator,e,cfg);
    session.peers[0].proof=.{.d=&d,.handle=null};
    try t.expect(!session.closeChecked().quiet);try t.expect(session.retained);
    try t.expect(!session.closeChecked().quiet);try t.expect(e.owns_host_gather and glm.kv_host.active!=null);try t.expectEqual(held,Lifetime.live);
    // CPU fixture has no actual queued work. Dispose synthetic retained Session explicitly; production sticky state is unchanged.
    t.allocator.destroy(session);e.deinit();try t.expect(glm.kv_host.active==null);try t.expectEqual(@as(usize,0),Lifetime.live);
}
