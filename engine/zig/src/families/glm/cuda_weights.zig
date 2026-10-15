//! GLM-5.3-Flash's TP rank on the GPU: the split's groups as device buffers, filled by the reader from O_DIRECT slots.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const split = @import("cuda_split.zig");
const index = @import("cuda_index.zig");
const read = @import("cuda_read.zig");

/// One rank's weights on the device, in the checkpoint's stored layouts (the kernels' packing comes after the load).
pub const Weights = struct {
    gpa: std.mem.Allocator,
    plan: split.Plan,
    groups: []cuda.DeviceBuffer,
    stats: read.Stats,
    unread: usize, // language-model tensors left unread (a layer subset leaves some)

    /// The device address of a planned tensor.
    pub fn ptr(w: *const Weights, r: split.Ref) u64 {
        return w.groups[r.g].ptr + r.off;
    }

    pub fn deinit(w: *Weights) void {
        for (w.groups) |*g| g.free();
        w.gpa.free(w.groups);
        w.plan.deinit();
        w.* = undefined;
    }
};

/// Copies on one stream; a read's slot is reused only after its copies have run (pageable memory: the driver stages it).
const Device = struct {
    groups: []cuda.DeviceBuffer,
    stream: cuda.Stream,

    fn sink(d: *Device) read.Sink {
        return .{ .ctx = d, .put = put, .flush = flush };
    }

    fn put(ctx: *anyopaque, g: u32, dst: u64, bytes: []const u8) anyerror!void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        try d.groups[g].uploadAsync(@intCast(dst), bytes, d.stream.handle);
    }

    fn flush(ctx: *anyopaque) anyerror!void {
        const d: *Device = @ptrCast(@alignCast(ctx));
        try d.stream.synchronize();
    }
};

/// `dir`'s rank of the split config `c` in `how`'s slices onto the current context's device; every group allocated first.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, dir: []const u8, c: *const cfg.Config, how: split.Split, o: read.Options) !Weights {
    var ix = try index.open(gpa, dir);
    defer ix.deinit();
    var p = try split.plan(gpa, c, &ix.names, how);
    errdefer p.deinit();
    const unread = ix.unread(&p.used);
    if (unread != 0 and c.run == c.layers) return error.UnreadCheckpointTensors; // the whole model must leave none
    p.used.clearAndFree(gpa); // its keys live in the index, freed on return
    var s = try read.schedule(gpa, p.copies.items, o);
    defer s.deinit();
    const groups = try gpa.alloc(cuda.DeviceBuffer, p.groups.items.len);
    errdefer gpa.free(groups);
    var made: usize = 0;
    errdefer for (groups[0..made]) |*g| g.free();
    while (made < groups.len) {
        groups[made] = try cuda.DeviceBuffer.alloc(d, @intCast(p.groups.items[made]));
        made += 1;
        try groups[made - 1].fill8(0, null); // the padding between tensors reads as zeros, so device digests equal cuda_digest.fromFiles
    }
    var dev: Device = .{ .groups = groups, .stream = try cuda.Stream.init(d, true) };
    defer dev.stream.deinit();
    const stats = try read.run(gpa, ix.shards, &s, dev.sink(), o);
    return .{ .gpa = gpa, .plan = p, .groups = groups, .stats = stats, .unread = unread };
}
