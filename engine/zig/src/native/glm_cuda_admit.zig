//! The CUDA GLM's rank-aware admission (qualification_qualification_44ba554e.8 day 2): a rank's device memory from its split plan alone, not the
//! checkpoint folder's total (one Spark holds one TP rank, about half the weights). The forecast is tf-glm-forward's
//! (qualification_e84ee1bf, glm_bench.zig's `forecast`): the packed weights, the load's peak (each stored group read, packed and freed as
//! the loader does), the Engine's one block (Device.attach) and the KDA wide scratch; plus qualification_e84ee1bf's measured overheads on
//! the Sparks (the CUDA context, modules, NCCL and comm.zig's buffers): +2.4 GiB over the load's peak, +4.2 GiB over
//! the run's device memory. The rank's MemAvailable less the need must stay at or above the floor (14 GiB, the lab's
//! line), else the open refuses with the numbers.
const std = @import("std");
const glm = @import("glm_cuda");
const tool = @import("glm_tool");

const gib: f64 = 1 << 30;

/// qualification_e84ee1bf §2's measured overheads (GiB): the load phase over its device peak, the run phase over its device memory.
pub const load_over_gib: f64 = 2.4;
pub const run_over_gib: f64 = 4.2;
/// The lowest MemAvailable a rank may forecast (GiB): the lab's 14 GiB line (TF_GLM_FLOOR_GIB overrides it).
pub const floor_gib: f64 = 14;

/// A rank's device bytes, from its plans and its Engine's shapes (nothing placed on a GPU).
pub const Forecast = struct {
    rank: u32,
    /// the packed weights (python's buffers), and their buffer count
    packed_bytes: u64,
    buffers: usize,
    /// the load's peak: the homes placed so far plus the stored groups still read, at most
    load_peak: u64,
    /// after the load: the homes and the stored groups kept (EXL3's experts stay in theirs)
    after_load: u64,
    /// the Engine's one block at the served capacity and prefill rows (its DSA scratch included)
    engine: u64,
    /// the forward's KDA wide scratch (q fp32, y bf16) for prefill_rows rows
    kda: u64,
    capacity: usize,
    prefill_rows: usize,

    /// The run's device memory: after the load, the Engine's block and the KDA scratch.
    pub fn run(f: Forecast) u64 {
        return f.after_load + f.engine + f.kda;
    }

    /// The need in GiB: the larger of the load's peak and the run's device memory, each with its measured overhead.
    pub fn need(f: Forecast) f64 {
        return @max(toGib(f.load_peak) + load_over_gib, toGib(f.run()) + run_over_gib);
    }
};

fn toGib(n: u64) f64 {
    return @as(f64, @floatFromInt(n)) / gib;
}

/// glm_bench.forecast's arithmetic for rank `rank`'s plan `r` and its Engine `e` (sized, not attached).
pub fn forecast(r: *tool.Rank, e: *glm.engine.Engine, rank: u32, prefill_rows: usize, gpa: std.mem.Allocator) !Forecast {
    const pk = &r.pk;
    const groups = r.split.groups();
    const live = try gpa.alloc(bool, groups.len);
    defer gpa.free(live);
    @memset(live, false);
    var stored: u64 = 0;
    var homes: u64 = 0;
    var peak: u64 = 0;
    for (0..groups.len) |gi| {
        const g: u32 = @intCast(gi);
        var needed = false;
        for (pk.items.items) |it| if (it.home >= g and glm.pack.readsGroup(it, g)) {
            needed = true;
        };
        if (needed) {
            live[gi] = true;
            stored += @max(groups[gi], 256);
        }
        homes += @max(pk.homeBytes(g), 256);
        peak = @max(peak, homes + stored);
        for (live, 0..) |*l, s| if (l.* and pk.last(@intCast(s)) <= g) {
            l.* = false;
            stored -= @max(groups[s], 256);
        };
    }
    const block = try e.deviceBytes();
    return .{
        .rank = rank,
        .packed_bytes = pk.bytes(),
        .buffers = pk.items.items.len,
        .load_peak = peak,
        .after_load = homes + stored,
        .engine = block.total,
        .kda = @as(u64, prefill_rows) * e.m.ll() * 128 * 6, // glm_run_gpu's q (fp32) and y (bf16)
        .capacity = e.st.capacity,
        .prefill_rows = prefill_rows,
    };
}

/// The forecast in one line (the open's log, and a refusal's words).
pub fn describe(f: Forecast, available_gib: f64, floor: f64, buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "rank {d}: packed weights {d:.3} GiB ({d} buffers); load peak {d:.3} + {d:.1} GiB; run {d:.3} GiB (after the load {d:.3}, the Engine's block {d:.3} at capacity {d} and prefill rows {d}, the KDA scratch {d:.3}) + {d:.1} GiB; need {d:.1} GiB; MemAvailable {d:.1} GiB leaves {d:.1} GiB, floor {d:.1} GiB", .{
        f.rank,              toGib(f.packed_bytes), f.buffers, toGib(f.load_peak), load_over_gib, toGib(f.run()), toGib(f.after_load), toGib(f.engine), f.capacity, f.prefill_rows, toGib(f.kda),
        run_over_gib,        f.need(),              available_gib, available_gib - f.need(), floor,
    });
}

/// Whether the rank fits: MemAvailable less the need at or above the floor.
pub fn fits(f: Forecast, available_gib: f64, floor: f64) bool {
    return available_gib - f.need() >= floor;
}

/// The words for a refused rank (null: it fits).
pub fn refusal(a: std.mem.Allocator, f: Forecast, available_gib: f64, floor: f64) !?[]const u8 {
    if (fits(f, available_gib, floor)) return null;
    var buf: [1024]u8 = undefined;
    return try std.fmt.allocPrint(a, "the CUDA GLM-5.3-Flash rank does not fit this Spark's memory: {s}; serve a smaller --context or TF_GLM_PREFILL_ROWS, free memory, or lower TF_GLM_FLOOR_GIB", .{try describe(f, available_gib, floor, &buf)});
}

/// TF_GLM_FLOOR_GIB, else the lab's 14 GiB line; null when it names no number of GiB from 0 to 64.
pub fn floorFrom(text: ?[]const u8) ?f64 {
    const t = std.mem.trim(u8, text orelse return floor_gib, " ");
    if (t.len == 0) return floor_gib;
    const v = std.fmt.parseFloat(f64, t) catch return null;
    return if (v >= 0 and v <= 64) v else null;
}

/// This host's MemAvailable now, in GiB (tf-glm-forward's reading of /proc/meminfo).
pub fn available() f64 {
    return tool.memAvailableGib();
}

test "qualification_e84ee1bf's EXL3 form (no MTP layer, prefill rows 1,024) fits expert1-host's 104.7 GiB start at 17.8 GiB, and a smaller pool is refused with the numbers" {
    // qualification_e84ee1bf NOTES §2's EXL3 row: device load peak 80.73, run 82.71 (after the load 80.734, the Engine 1.949, KDA 0.023)
    const f: Forecast = .{ .rank = 0, .packed_bytes = 86_686_000_000, .buffers = 1481, .load_peak = 86_685_000_000, .after_load = 86_686_000_000, .engine = 2_092_700_000, .kda = 25_000_000, .capacity = 2560, .prefill_rows = 1024 };
    try std.testing.expect(@abs(toGib(f.run()) - 82.71) < 0.01);
    try std.testing.expect(@abs(f.need() - 86.91) < 0.01);
    try std.testing.expect(fits(f, 104.7, floor_gib));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(@as(?[]const u8, null), try refusal(arena.allocator(), f, 104.7, floor_gib));
    const words = (try refusal(arena.allocator(), f, 98.0, floor_gib)).?;
    for ([_][]const u8{ "does not fit", "rank 0", "need 86.9 GiB", "MemAvailable 98.0 GiB leaves 11.1 GiB, floor 14.0 GiB", "capacity 2560", "prefill rows 1024", "load peak 80.73" }) |s| {
        if (std.mem.indexOf(u8, words, s) == null) {
            std.debug.print("missing {s} in: {s}\n", .{ s, words });
            return error.Missing;
        }
    }
    try std.testing.expectEqual(@as(?f64, 14), floorFrom(null));
    try std.testing.expectEqual(@as(?f64, 12.5), floorFrom(" 12.5"));
    try std.testing.expectEqual(@as(?f64, null), floorFrom("lots"));
    try std.testing.expectEqual(@as(?f64, null), floorFrom("-1"));
}

fn bytesOf(g: f64) u64 {
    return @intFromFloat(g * gib);
}

test "drafts on (qualification_qualification_44ba554e.8 day 3): the MTP layer's 1.93 GiB a rank is in the need, and a pool that fits drafts off but not the layer is refused with the numbers" {
    // qualification_qualification_e0f4f003 NOTES §5's EXL3 form at prefill rows 512: with the MTP layer, run device 83.849 GiB (the Engine 1.172) and
    // a need of 88.05; without it, the layer's 1.93 GiB less (qualification_e84ee1bf's after-load 80.734) and the Engine's 1.170
    const kda = bytesOf(0.0117);
    const on: Forecast = .{ .rank = 1, .packed_bytes = bytesOf(82.665), .buffers = 1505, .load_peak = bytesOf(82.664), .after_load = bytesOf(82.665), .engine = bytesOf(1.172), .kda = kda, .capacity = 2560, .prefill_rows = 512 };
    var off = on;
    off.packed_bytes -= bytesOf(1.93);
    off.load_peak -= bytesOf(1.93);
    off.after_load -= bytesOf(1.93);
    off.engine = bytesOf(1.170);
    try std.testing.expect(@abs(toGib(on.run()) - 83.849) < 0.001);
    try std.testing.expect(@abs(on.need() - 88.05) < 0.01);
    try std.testing.expect(@abs(on.need() - off.need() - 1.932) < 0.001);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectEqual(@as(?[]const u8, null), try refusal(arena.allocator(), off, 101.0, floor_gib));
    const words = (try refusal(arena.allocator(), on, 101.0, floor_gib)).?;
    for ([_][]const u8{ "does not fit", "rank 1", "need 88.0 GiB", "MemAvailable 101.0 GiB leaves 13.0 GiB, floor 14.0 GiB", "prefill rows 512", "the Engine's block 1.172" }) |s| {
        if (std.mem.indexOf(u8, words, s) == null) {
            std.debug.print("missing {s} in: {s}\n", .{ s, words });
            return error.Missing;
        }
    }
    // qualification_1bae3bb5's measured need with the layer (86.35-87.01 GiB) fits expert1-host's 104.7 GiB start above the floor
    try std.testing.expect(fits(on, 104.7, floor_gib));
}
