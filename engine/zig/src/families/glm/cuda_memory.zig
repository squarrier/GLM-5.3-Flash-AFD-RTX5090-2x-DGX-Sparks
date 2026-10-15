//! A GLM-5.3-Flash TP rank's memory at a context window: the loader's weights plus the Python 0.6.6 engine's cache geometry.
const std = @import("std");
const cfg = @import("config.zig");

pub const gib: u64 = 1 << 30;
/// A decode window's most rows (a pending token and up to seven drafts), the cache's reserve past the window.
pub const max_rows: u64 = 8;
/// Cache slots while DSA attention stays dense (contexts up to 2,051 tokens).
pub const dense_capacity: u64 = 2560;
/// A prompt chunk's rows.
pub const prefill_rows: u64 = 2048;
const prompt_att_rows: u64 = 512; // a dense latent attention call's prompt rows (forward.PROMPT_ATT_ROWS)
const select_rows: u64 = 512; // prompt rows whose pool scores are held at once (sparse.SELECT_ROWS)

/// geometry.mla_geometry(latent=True) for a rank of `ranks` (`c` unsplit): bytes that do not grow with the window.
pub fn fixedBytes(c: *const cfg.Config, ranks: u64, mtp: bool) u64 {
    const linear: u64 = c.countKind(.kda);
    const heads: u64 = c.mla_heads / ranks;
    const lh: u64 = c.kda_heads / ranks;
    const ld: u64 = c.kda_dim;
    const conv: u64 = c.conv;
    const kd: u64 = c.nope;
    const vd: u64 = c.v_dim;
    const rows: u64 = 64;
    const d: u64 = c.hidden;
    var fixed = linear * (4 * lh * ld * ld * 4 + 3 * (conv - 1) * 3 * lh * ld * 2); // KDA states and conv tails
    fixed += linear * rows * (3 * lh * ld + 2 * ld + lh) * 2;
    fixed += linear * rows * lh * (12 * ld + 4);
    const slots: u64 = c.topk + 1;
    const width: u64 = c.moe_inter / ranks;
    var extent = d * c.hc + c.vocab / ranks + slots * (d + width) + heads * (2 * kd + vd);
    extent += @as(u64, c.q_lora) * 2 + @as(u64, c.kv_lora) * 2;
    extent += @as(u64, c.dense_inter) * 3 / ranks + @as(u64, c.i_heads) * c.i_dim;
    fixed += (if (mtp) @as(u64, 2) else 1) * (16 * rows * extent * 4 + 8 * rows * 16384 * 4); // decode windows' rows
    fixed += prefill_rows * 5 * (extent - c.vocab / ranks); // a prompt chunk's buffers
    return fixed;
}

/// Bytes that grow with the cache slots: the latent cache, the indexer's keys, a prompt chunk's selection and partials.
pub fn cacheBytes(c: *const cfg.Config, ranks: u64, mtp: bool, slots: u64) u64 {
    const heads: u64 = c.mla_heads / ranks;
    const lw: u64 = c.kv_lora;
    const topk: u64 = c.i_topk;
    const count: u64 = c.countKind(.mla) + @intFromBool(mtp); // the backbone's DSA layers, and MTP's
    var scratch = @as(u64, @min(prefill_rows, select_rows)) * 4 * ((slots + 3) / 4) + prefill_rows * 16 * (topk + 3);
    scratch += (topk + 515) / 512 * prefill_rows * heads * (lw + 2) * 4;
    const dense: u64 = @as(u64, @min(slots, dense_capacity)) + prefill_rows;
    scratch += (dense + 511) / 512 * @as(u64, @min(prefill_rows, prompt_att_rows)) * heads * (lw + 2) * 4 + 4 * prefill_rows * heads * lw;
    var cache = count * slots * lw * 2; // the 512-wide latent, bf16, a token and layer
    cache += count * (2 * slots + slots / 4 + 2) * @as(u64, c.i_dim) * 2; // the indexer's keys and pooled keys
    return cache + scratch;
}

/// Cache slots a window takes: the window and a decode window's reserve, at least the dense capacity.
pub fn slotsFor(window: u64) u64 {
    return @max(dense_capacity, window + max_rows);
}

/// One rank's memory plan on a unified-memory GPU (GB10: the host's RAM), in bytes.
pub const Plan = struct {
    weights: u64, // the loader's groups
    staging: u64, // load-time extras beside them (read slots; a layer's pack scratch)
    fixed: u64,
    cache: u64,
    kept: u64, // kept prompt states (the server's --prompt-cache-gib; its default, what 70 % of RAM leaves past the server, is none here)
    serving: u64, // weights + fixed + cache + kept
    startup: u64, // weights + staging
    reserve: u64, // the Python engine's admission reserve: max(4 GiB, a tenth of memory)
    context: u64, // CUDA context, NCCL, graphs and cuBLASLt outside the estimate (assumed; the lab measures it)
    peak: u64, // max(serving, startup) + context

    /// MemAvailable left at the peak, from `available` before the engine starts.
    pub fn left(p: Plan, available: u64) i64 {
        return @as(i64, @intCast(available)) - @as(i64, @intCast(p.peak));
    }
};

pub const Inputs = struct {
    weights: u64,
    staging: u64,
    window: u64,
    ranks: u64 = 2,
    mtp: bool = true,
    kept: u64 = 0,
    total: u64, // MemTotal
    context: u64 = 5 * gib,
};

pub fn plan(c: *const cfg.Config, in: Inputs) Plan {
    const fixed = fixedBytes(c, in.ranks, in.mtp);
    const cache = cacheBytes(c, in.ranks, in.mtp, slotsFor(in.window));
    const serving = in.weights + fixed + cache + in.kept;
    const startup = in.weights + in.staging;
    return .{
        .weights = in.weights,
        .staging = in.staging,
        .fixed = fixed,
        .cache = cache,
        .kept = in.kept,
        .serving = serving,
        .startup = startup,
        .reserve = @max(4 * gib, in.total / 10),
        .context = in.context,
        .peak = @max(serving, startup) + in.context,
    };
}

test "the cache geometry equals python-0.6 geometry.mla_geometry on GLM-5.3-Flash at TP2, byte for byte" {
    const c: cfg.Config = .{};
    // geometry.mla_geometry(text, 2, 8, minimum_slots=2560, latent=True, mtp=True).needed(window), python-0.6 ed78d6f
    const want = [_][2]u64{ .{ 2051, 4480022528 }, .{ 32768, 5075640320 }, .{ 131072, 7013408768 }, .{ 262144, 9597100032 }, .{ 487495, 14039217152 } };
    for (want) |w| try std.testing.expectEqual(w[1], fixedBytes(&c, 2, true) + cacheBytes(&c, 2, true, slotsFor(w[0])));
    const off = [_][2]u64{ .{ 2051, 3659609600 }, .{ 32768, 4206881792 }, .{ 131072, 5987363840 }, .{ 262144, 8361339904 } };
    for (off) |w| try std.testing.expectEqual(w[1], fixedBytes(&c, 2, false) + cacheBytes(&c, 2, false, slotsFor(w[0])));
}

test "a plan's peak is the larger of serving and startup, plus the context outside the estimate" {
    const c: cfg.Config = .{};
    const p = plan(&c, .{ .weights = 90 * gib, .staging = 1 * gib, .window = 262144, .kept = 3 * gib, .total = 127600732 * 1024 });
    try std.testing.expectEqual(p.serving + 5 * gib, p.peak);
    try std.testing.expectEqual(@as(u64, 127600732 * 1024 / 10), p.reserve);
    try std.testing.expect(p.left(110 * gib) < p.left(120 * gib));
}
