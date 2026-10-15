//! GPU checks of GLM-5.3-Flash's CUDA kernel copies: `tf-glm-kernels <command>`, PASS/FAIL/RESULT lines, exit 1 on failure.
//! The fixtures come from zig/tests/cuda/glm/kda_oracle.py (python-0.6's own kda.py on the same GPU).

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_kernels");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const usage =
    \\usage: tf-glm-kernels <command>
    \\  symbols          every glm_kda kernel resolves by its listed symbol (registers, threads, static shared memory)
    \\  kda <dir>...     the copied KDA kernels on kda_oracle.py's fixtures, byte for byte (op chain, replay, layers)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    run(gpu, args[1], args[2..]) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ args[1], e });
        return 1;
    };
    return 0;
}

fn run(gpu: Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "symbols")) return symbols(gpu);
    if (std.mem.eql(u8, cmd, "kda") and rest.len > 0) {
        for (rest) |dir| try kda(gpu, dir);
        return;
    }
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn symbols(gpu: Gpu) !void {
    var k = try glm.Kda.load(gpu.d);
    defer k.unload();
    for (glm.sym.all) |name| {
        const f = try k.module.function(name);
        std.debug.print("RESULT glm_kda {s}: {d} registers, max {d} threads, {d} static shared bytes\n", .{ name, try f.attribute(.num_regs), try f.attribute(.max_threads_per_block), try f.attribute(.shared_size_bytes) });
    }
    check.pass("symbols: the {d} glm_kda kernels resolve", .{glm.sym.all.len});
}

/// The fixture's arrays on the device, freed together.
const Arrays = struct {
    fx: Fixture,
    gpu: Gpu,
    bufs: std.StringArrayHashMapUnmanaged(cuda.DeviceBuffer) = .empty,

    fn up(self: *Arrays, name: []const u8) !u64 {
        const host = try self.fx.bytes(name);
        defer self.gpu.gpa.free(host);
        const b = try cuda.DeviceBuffer.fromHost(self.gpu.d, host);
        try self.bufs.put(self.gpu.gpa, name, b);
        return b.ptr;
    }

    /// A zeroed device buffer the size of the fixture's expected array `name` (the kernel writes it).
    fn zeros(self: *Arrays, name: []const u8) !u64 {
        const want = try self.fx.bytes(name);
        defer self.gpu.gpa.free(want);
        var b = try cuda.DeviceBuffer.alloc(self.gpu.d, want.len);
        try b.fill8(0, null);
        try self.gpu.ctx.synchronize(); // a null-stream memset is asynchronous, and the kernels' non-blocking stream does not wait for it
        try self.bufs.put(self.gpu.gpa, name, b);
        return b.ptr;
    }

    fn same(self: *Arrays, name: []const u8, what: []const u8) !usize {
        const want = try self.fx.bytes(name);
        defer self.gpu.gpa.free(want);
        const got = try check.download(self.gpu, self.bufs.get(name).?);
        defer self.gpu.gpa.free(got);
        try check.sameBytes(what, got, want);
        return got.len;
    }

    fn deinit(self: *Arrays) void {
        for (self.bufs.values()) |*b| b.free();
        self.bufs.deinit(self.gpu.gpa);
        self.fx.deinit();
    }
};

fn usize_(fx: Fixture, name: []const u8) !usize {
    return @intCast(try fx.int(name));
}

fn kda(gpu: Gpu, dir: []const u8) !void {
    var a: Arrays = .{ .fx = try Fixture.open(gpu.gpa, gpu.io, dir), .gpu = gpu };
    defer a.deinit();
    var k = try glm.Kda.load(gpu.d);
    defer k.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const op = try a.fx.string("op");
    const heads = try usize_(a.fx, "heads");
    const rows = try usize_(a.fx, "rows");
    if (std.mem.eql(u8, op, "chain")) {
        const wide = (try a.fx.int("wide")) != 0;
        const ag = try a.up("ag");
        const c: glm.Chain = .{
            .heads = heads, .p = try a.up("p"), .p_stride = try usize_(a.fx, "p_stride"), .b_off = try usize_(a.fx, "b_off"),
            .a = ag, .a_stride = try usize_(a.fx, "ag_stride"), .g = ag + 2 * try usize_(a.fx, "g_column"),
            .g_stride = try usize_(a.fx, "ag_stride"), .conv_state = try a.up("conv_state"), .conv_w = try a.up("conv_w"),
            .state_in = try a.up("state_in"), .a_log = try a.up("a_log"), .dt_bias = try a.up("dt_bias"),
            .norm_w = try a.up("norm_w"), .eps = @floatCast(try a.fx.float("eps")), .lower = @floatCast(try a.fx.float("lower")),
            .rows = rows, .out = try a.zeros("out"), .state_out = try a.zeros("state_out"), .k_save = try a.zeros("k_save"),
            .v_save = try a.zeros("v_save"), .g_save = try a.zeros("g_save"), .b_save = try a.zeros("b_save"),
        };
        const w = glm.wideScratchBytes(rows, heads);
        var q_tmp = try cuda.DeviceBuffer.alloc(gpu.d, w.q);
        defer q_tmp.free();
        var y_tmp = try cuda.DeviceBuffer.alloc(gpu.d, w.y);
        defer y_tmp.free();
        if (wide) try k.chainWide(stream, c, q_tmp.ptr, y_tmp.ptr) else try k.chainOne(stream, c);
        try stream.synchronize();
        var n: usize = 0;
        for ([_][]const u8{ "out", "state_out", "k_save", "v_save", "g_save", "b_save" }) |name| n += try a.same(name, name);
        check.pass("BITEXACT glm_kda {s} {s}: {d} heads, {d} rows, {d} bytes equal to python-0.6's kda.chain", .{ dir, if (wide) "prep+step+out" else "chain_kernel", heads, rows, n });
        // the other path on the same inputs: kda.py promises the same bits; zeroed on the kernels' stream, so the zeros land first
        for ([_][]const u8{ "out", "state_out", "k_save", "v_save", "g_save", "b_save" }) |name| try a.bufs.get(name).?.fill8(0, stream.handle);
        if (wide) try k.chainOne(stream, c) else try k.chainWide(stream, c, q_tmp.ptr, y_tmp.ptr);
        try stream.synchronize();
        for ([_][]const u8{ "out", "state_out", "k_save", "v_save", "g_save", "b_save" }) |name| _ = try a.same(name, name);
        check.pass("BITEXACT glm_kda {s}: the {s} path gives the same bytes", .{ dir, if (wide) "one-kernel" else "three-kernel" });
    } else if (std.mem.eql(u8, op, "replay")) {
        const out = try a.zeros("state_out");
        try k.replay(stream, heads, try a.up("state_in"), try a.up("k_save"), try a.up("v_save"), try a.up("g_save"), try a.up("b_save"), rows, out);
        try stream.synchronize();
        check.pass("BITEXACT glm_kda {s} replay_kernel: {d} heads, {d} rows, {d} state bytes equal to python-0.6's kda.replay", .{ dir, heads, rows, try a.same("state_out", "state_out") });
    } else if (std.mem.eql(u8, op, "layers")) {
        const layers = try usize_(a.fx, "layers");
        const window = try usize_(a.fx, "window");
        const r: glm.ReplayLayers = .{
            .state_in = try a.up("state_in"), .state_stride = heads * glm.dv * glm.dv, .k_save = try a.up("k_save"),
            .v_save = try a.up("v_save"), .g_save = try a.up("g_save"), .b_save = try a.up("b_save"),
            .kv_stride = window * heads * glm.dv, .b_stride = window * heads, .layers = layers, .heads = heads,
            .rows = rows, .state_out = try a.zeros("state_out"),
        };
        try k.replayLayers(stream, r);
        try stream.synchronize();
        check.pass("BITEXACT glm_kda {s} replay_layers_kernel: {d} layers x {d} heads, {d} rows, {d} state bytes equal to python-0.6's kda.replay_layers", .{ dir, layers, heads, rows, try a.same("state_out", "state_out") });
    } else return error.UnknownOp;
}
