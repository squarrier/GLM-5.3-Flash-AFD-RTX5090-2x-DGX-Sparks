//! shared_pool: one pool of the attention State's capacity-sized caches shared by the users' slots (TF_AFD_POOL_TOKENS,
//! default off). Each live slot holds one contiguous region of 256-token blocks (Ash Hart's TensorFold #639 layout,
//! glm/pool.zig at v1.0.7: one MLA cache shared by a Mac's streams, a contiguous region each), placed first fit; when
//! the free blocks cover a request but not in one run, the regions slide down in address order, each moved once, a
//! copy of the same bytes on the forward's stream (Mia's GLM recipe patch 0074-glm-compact-before-evict, by ezoushen).
//! A region's view is a State whose capacity-sized buffers point into the pool with State.init(capacity)'s shapes: the
//! kernels address `base + position x row` (no block table), so no kernel, fatbin or launch shape changes.
//! Ash Hart authored TensorFold/MCDMA; Mia (@MiaAI_lab) authored the recipe; inherited NOTICE credits apply.
const std = @import("std");
const cuda = @import("cuda");
const bufs = @import("cuda_buffers.zig");
const views = @import("cuda_views.zig");
const md = @import("cuda_model.zig");
const cc = @import("cuda_compute.zig");
const Fwd = @import("cuda_fwd.zig").Fwd;
const Buf = views.Buf;

/// Tokens a block holds: a multiple of the indexer's pooling width (4), as upstream's Pool.block.
pub const block: usize = 256;
/// A region of n blocks holds n * 256 - 8 tokens, so its view's capacity / 4 + 2 pooled keys end at its end.
pub const slack: usize = 8;
/// Slots a pool serves (the users path's most: cuda_afd_users_round.run takes at most 4 rows).
pub const max_slots = 4;

/// The tokens a region of `n` blocks holds.
pub fn tokensOf(n: usize) usize {
    return n * block - slack;
}

/// The blocks a region of `tokens` takes.
pub fn blocksFor(tokens: usize) usize {
    return (tokens + slack + block - 1) / block;
}

/// The cache rows a request reserves at admission: its prompt, its reply and one row past them (a greedy stream
/// writes prompt + max_new - 1 rows at most: its last token is sampled, never written).
pub fn need(prompt: usize, max_new: usize) usize {
    return prompt + max_new + 1;
}

/// TF_AFD_POOL_TOKENS's exact spelling: unset or "0" is off; else a decimal multiple of 256 without a leading zero.
pub fn parse(value: ?[]const u8) error{BadAfdPool}!usize {
    const v = value orelse return 0;
    if (std.mem.eql(u8, v, "0")) return 0;
    if (v.len == 0 or v.len > 10 or v[0] == '0') return error.BadAfdPool;
    for (v) |ch| if (ch < '0' or ch > '9') return error.BadAfdPool;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadAfdPool;
    if (n % block != 0 or n / block > std.math.maxInt(u32)) return error.BadAfdPool;
    return n;
}

/// TF_AFD_POOL_TOKENS from the environment, read once before anything loads.
pub fn fromEnv() error{BadAfdPool}!usize {
    return parse(if (std.c.getenv("TF_AFD_POOL_TOKENS")) |v| std.mem.span(v) else null);
}

/// A slot's span of the pool, in blocks.
pub const Region = struct { start: u32, blocks: u32 };
/// One region slid down by compaction: its slot, its first block before and after, its length.
pub const Move = struct { slot: u8, from: u32, to: u32, blocks: u32 };
pub const Reserved = struct { region: Region, moved: usize };

/// `b` cut to `rows` rows from row `first` (the same dtype and row shape).
fn rowsOf(b: Buf, first: usize, rows: usize) Buf {
    var v = b;
    v.ptr = b.ptr + first * b.inner() * b.dtype.size();
    v.shape[0] = rows;
    return v;
}

/// A State's capacity-sized buffers (the latent caches, the MTP latent, the indexer's keys, gates and pooled keys) set
/// to no rows, so a device block sizes nothing for them; a pool region's view replaces them before any launch.
pub fn detachTokens(st: *bufs.State) void {
    for (st.kc) |*c| c.shape[0] = 0;
    if (st.mtp_kc) |*c| c.shape[0] = 0;
    if (st.index) |idx| for (idx) |*ic| {
        for (ic) |*c| c.shape[0] = 0;
    };
}

/// Visits a State's capacity-sized buffers as `prefix`.kc.N, .mtp_kc, .index.N.M.
pub fn walkTokens(st: *bufs.State, prefix: []const u8, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
    var name: [64]u8 = undefined;
    for (st.kc, 0..) |*c, i| try visit(ctx, try std.fmt.bufPrint(&name, "{s}.kc.{d}", .{ prefix, i }), c);
    if (st.mtp_kc) |*c| try visit(ctx, try std.fmt.bufPrint(&name, "{s}.mtp_kc", .{prefix}), c);
    if (st.index) |idx| for (idx, 0..) |*ic, i| for (ic, 0..) |*c, j| try visit(ctx, try std.fmt.bufPrint(&name, "{s}.index.{d}.{d}", .{ prefix, i, j }), c);
}

const Sizer = struct {
    used: u64 = 0,
    fn visit(s: *Sizer, _: []const u8, b: *Buf) anyerror!void {
        if (!b.host) s.used = std.mem.alignForward(u64, s.used + b.bytes(), 256);
    }
};

/// The device bytes a State's capacity-sized buffers take (each at a 256-byte boundary, as bufs.Arena places them).
pub fn tokenBytes(st: *bufs.State) !u64 {
    var s: Sizer = .{};
    try walkTokens(st, "st", &s, Sizer.visit);
    return s.used;
}

pub const Pool = struct {
    blocks: u32,
    /// The engine's capacity: a view never exceeds it (the DSA scratch is sized for it; a request never needs more).
    cap_max: usize,
    /// One State's capacity-sized buffers at tokensOf(blocks): the pool's caches (its other buffers never allocated).
    base: bufs.State,
    regions: [max_slots]?Region = @splat(null),
    dev: ?cuda.DeviceBuffer = null,
    reserves: u64 = 0,
    releases: u64 = 0,
    reserved_blocks: u64 = 0,
    released_blocks: u64 = 0,
    compactions: u64 = 0,
    moves: u64 = 0,

    /// A pool of `n_tokens` (a multiple of 256) for the model's States of `rows` decode rows at most `cap_max` each.
    pub fn init(gpa: std.mem.Allocator, m: *const md.Model, n_tokens: usize, rows: usize, long_context: bool, cap_max: usize) !Pool {
        if (n_tokens == 0 or n_tokens % block != 0 or n_tokens / block > std.math.maxInt(u32)) return error.BadAfdPool;
        const n: u32 = @intCast(n_tokens / block);
        return .{ .blocks = n, .cap_max = cap_max, .base = try bufs.State.init(gpa, m, tokensOf(n), rows, long_context) };
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        if (p.dev) |*d| d.free();
        p.base.deinit(gpa);
        p.* = undefined;
    }

    /// The pool's buffers as pool.kc.N, pool.mtp_kc, pool.index.N.M (a Space or an arena).
    pub fn walk(p: *Pool, ctx: anytype, comptime visit: fn (@TypeOf(ctx), []const u8, *Buf) anyerror!void) !void {
        try walkTokens(&p.base, "pool", ctx, visit);
    }

    pub fn bytes(p: *Pool) !u64 {
        var s: Sizer = .{};
        try p.walk(&s, Sizer.visit);
        return s.used;
    }

    /// One device block for the pool, zeroed (after the Engine's and the users' blocks).
    pub fn attach(p: *Pool, d: *const cuda.Driver) !void {
        if (p.dev != null) return error.Attached;
        var dev = try cuda.DeviceBuffer.alloc(d, @intCast(try p.bytes()));
        errdefer dev.free();
        try dev.fill8(0, null);
        var arena: bufs.Arena = .{ .base = dev.ptr };
        try p.walk(&arena, bufs.Arena.visit);
        p.dev = dev;
    }

    pub fn tokens(p: *const Pool) usize {
        return tokensOf(p.blocks);
    }

    pub fn freeBlocks(p: *const Pool) u32 {
        var used: u32 = 0;
        for (p.regions) |r| if (r) |x| {
            used += x.blocks;
        };
        return p.blocks - used;
    }

    /// Whether a stream of `rows` can take its region beside the live ones now (compaction gathers free blocks); true
    /// when it can never fit, so its prefill refuses it in words instead of it waiting forever.
    pub fn fits(p: *const Pool, rows: usize) bool {
        if (rows > p.tokens()) return true;
        return blocksFor(rows) <= p.freeBlocks();
    }

    /// The lane host's admission check (api.Admit.fits, upstream lane_host.zig:277-283): a request's prompt and reply
    /// beside the running streams' regions.
    pub fn admitFits(ctx: *anyopaque, prompt: u32, max_tokens: u32) bool {
        const p: *const Pool = @ptrCast(@alignCast(ctx));
        return p.fits(need(prompt, max_tokens));
    }

    /// The first run of `n` free blocks (upstream's first fit), or null.
    fn findRun(p: *const Pool, n: u32) ?u32 {
        var at: u32 = 0;
        while (at + n <= p.blocks) {
            var next: ?u32 = null; // the end of the last region overlapping [at, at + n)
            for (p.regions) |r| if (r) |x| if (x.start < at + n and at < x.start + x.blocks) {
                next = @max(next orelse 0, x.start + x.blocks);
            };
            if (next) |end| at = end else return at;
        }
        return null;
    }

    /// Every region slid down to the lowest blocks in address order, each moved once at most; `moves` gets the moves in
    /// the order they must run (a move's destination is free, or its own source, once the moves before it ran).
    fn compact(p: *Pool, moves: *[max_slots]Move) usize {
        var order: [max_slots]u8 = undefined;
        var k: usize = 0;
        for (p.regions, 0..) |r, i| if (r != null) {
            order[k] = @intCast(i);
            k += 1;
        };
        for (1..k) |i| {
            var j = i;
            while (j > 0 and p.regions[order[j]].?.start < p.regions[order[j - 1]].?.start) : (j -= 1) std.mem.swap(u8, &order[j], &order[j - 1]);
        }
        var at: u32 = 0;
        var n: usize = 0;
        for (order[0..k]) |i| {
            const r = &p.regions[i].?;
            if (r.start != at) {
                moves[n] = .{ .slot = i, .from = r.start, .to = at, .blocks = r.blocks };
                n += 1;
                r.start = at;
            }
            at += r.blocks;
        }
        p.compactions += 1;
        p.moves += n;
        return n;
    }

    /// Slot `slot`'s region for `rows` cache rows: first fit, else after compaction (the caller runs `moves` first, in
    /// order, before any launch reads a moved region). A full pool refuses before anything changes.
    pub fn reserve(p: *Pool, slot: u8, rows: usize, moves: *[max_slots]Move) !Reserved {
        if (slot >= max_slots or p.regions[slot] != null) return error.SlotOutOfStep;
        if (rows > p.tokens()) return error.ContextFull;
        const n: u32 = @intCast(blocksFor(rows));
        var moved: usize = 0;
        const start = p.findRun(n) orelse blk: {
            if (p.freeBlocks() < n) return error.PoolFull;
            moved = p.compact(moves);
            // compacted, the live regions fill [0, used) and the free blocks (>= n) are one run at the end
            break :blk p.findRun(n).?;
        };
        p.regions[slot] = .{ .start = start, .blocks = n };
        p.reserves += 1;
        p.reserved_blocks += n;
        return .{ .region = p.regions[slot].?, .moved = moved };
    }

    /// Slot `slot`'s region back to the pool; its blocks (0 when it held none).
    pub fn release(p: *Pool, slot: u8) u32 {
        if (slot >= max_slots) return 0;
        const r = p.regions[slot] orelse return 0;
        p.regions[slot] = null;
        p.releases += 1;
        p.released_blocks += r.blocks;
        return r.blocks;
    }

    /// The capacity a region's view has: its tokens, at most the engine's.
    pub fn capacityOf(p: *const Pool, r: Region) usize {
        return @min(tokensOf(r.blocks), p.cap_max);
    }

    /// `st`'s capacity-sized buffers on region `r`: State.init(capacityOf(r))'s shapes from the region's first rows
    /// (the pooled keys from its first block's quarter), inside the region; `st.capacity` that capacity.
    pub fn view(p: *const Pool, r: Region, st: *bufs.State) void {
        const t0 = @as(usize, r.start) * block;
        const cap = p.capacityOf(r);
        for (st.kc, p.base.kc) |*c, b| c.* = rowsOf(b, t0, cap);
        if (st.mtp_kc) |*c| c.* = rowsOf(p.base.mtp_kc.?, t0, cap);
        if (st.index) |idx| for (idx, p.base.index.?) |*ic, b| {
            ic[0] = rowsOf(b[0], t0, cap);
            ic[1] = rowsOf(b[1], t0, cap);
            ic[2] = rowsOf(b[2], t0 / 4, cap / 4 + 2);
        };
        st.capacity = cap;
    }

    /// Move `mv`'s rows on the forward's stream: every cache's region rows (latent rows: tokensOf(blocks); pooled keys:
    /// 64 a block) to its new first block, in pieces of the shift, front first, so no piece reads rows an earlier
    /// piece wrote and no copy's source and destination overlap.
    pub fn copyMove(p: *Pool, f: *Fwd, mv: Move) !void {
        if (mv.to >= mv.from or mv.from + mv.blocks > p.blocks) return error.MoveOutOfStep;
        for (p.base.kc) |b| try moveRows(f, b, block, tokensOf(mv.blocks), mv);
        if (p.base.mtp_kc) |b| try moveRows(f, b, block, tokensOf(mv.blocks), mv);
        if (p.base.index) |idx| for (idx) |ic| {
            try moveRows(f, ic[0], block, tokensOf(mv.blocks), mv);
            try moveRows(f, ic[1], block, tokensOf(mv.blocks), mv);
            try moveRows(f, ic[2], block / 4, @as(usize, mv.blocks) * (block / 4), mv);
        };
    }

    fn moveRows(f: *Fwd, b: Buf, per: usize, total: usize, mv: Move) !void {
        const step = @as(usize, mv.from - mv.to) * per;
        var off: usize = 0;
        while (off < total) : (off += step) {
            const n = @min(step, total - off);
            try cc.copyDtoD(f, cc.rows(b, @as(usize, mv.to) * per + off, n), cc.rows(b, @as(usize, mv.from) * per + off, n));
        }
    }
};

test {
    _ = &Pool.reserve;
    _ = &Pool.copyMove;
    _ = &Pool.attach;
}
