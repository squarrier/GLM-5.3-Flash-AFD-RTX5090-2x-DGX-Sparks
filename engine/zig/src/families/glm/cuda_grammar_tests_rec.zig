//! grammar_tests EXL3 AFD recording fixture; CPU only, no exchange nodes or sockets.
const std = @import("std");
const model = @import("cuda_model.zig");
const views = @import("cuda_views.zig");
const eng = @import("cuda_engine.zig");
const afd = @import("cuda_afd_attn.zig");
const ae = @import("cuda_afd_engine.zig");
const trace = @import("cuda_trace.zig");
pub const Rec = struct {
    parsed: std.json.Parsed(std.json.Value),
    m: model.Model,
    w: views.Weights,
    e: *eng.Engine,
    provider: afd.Afd,
    at: ae.Attn,
    space: trace.Space,
    t: trace.Trace,
    heap: bool = false,
    pub fn init(x: *Rec, a: std.mem.Allocator) !void { return x.initRank(a, 1, 0); }
    fn initRank(x: *Rec, a: std.mem.Allocator, world: usize, rank: usize) !void {
        x.heap = false;
        if (!@import("builtin").is_test) return error.IncompleteMaskedAdapter;
        x.parsed = try std.json.parseFromSlice(std.json.Value, a, @import("cuda_layer_test.zig").fixture_exl3, .{});
        errdefer x.parsed.deinit();
        x.m = try model.Model.parse(a, x.parsed.value.object.get("config_json").?.string, world, rank);
        const mtp_layers = x.m.mtp_layers;
        try x.m.cut(5);
        x.m.mtp_layers = mtp_layers; // loaded head stays available to the subsequent plain fixture
        x.w = try views.rank(a, &x.m);
        errdefer x.w.deinit(a);
        x.e = try eng.Engine.create(a, &x.m, &x.w, 2560, 8, 64, false);
        errdefer x.e.destroy(a);
        if (world == 1) {
            x.provider = try afd.Afd.initWith(&x.m, 64, true);
            x.at = try ae.Attn.init(x.e, &x.provider);
        }
        x.space = .{ .gpa = a };
        errdefer x.space.deinit();
        try views.walk(&x.w, views.Path.init("w"), &x.space, trace.placeAll);
        try x.e.walk(&x.space, trace.placeAll);
        try x.e.walkTmp(&x.space, trace.placeAll);
        x.e.tmp.space = &x.space;
        if (world == 1) try x.space.place("b.afd_recv", &x.provider.afd_recv);
        x.t = trace.Trace.init(a);
    }
    pub fn deinit(x: *Rec, a: std.mem.Allocator) void {
        x.t.deinit();
        x.space.deinit();
        x.e.destroy(a);
        x.w.deinit(a);
        x.parsed.deinit();
        if (x.heap) a.destroy(x);
    }
    pub fn recording(x: *Rec) @import("cuda_fwd.zig").Fwd { return @import("cuda_fwd.zig").Fwd.recording(&x.m, &x.t); }
};
pub fn tp(a: std.mem.Allocator, rank: usize) !*Rec {
    const x = try a.create(Rec);
    errdefer a.destroy(x);
    try x.initRank(a, 2, rank);
    x.heap = true;
    return x;
}

