//! GPU check of GLM-5.3-Flash's affine kernel copies: `tf-glm-mm <command>`, PASS/FAIL/RESULT lines, exit 1 on failure.
//! The fixtures come from zig/tests/cuda/glm/mm_oracle.py: python-0.6's own qmm and experts wrappers on the same GPU, each
//! extension call recorded with its arguments and every buffer's bytes before and after. Each call is replayed through
//! the bindings (cuda_kernels_mm_calls.zig: the launches the CPU fixture test compares with python's), every buffer compared.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_mm");
const images = @import("glm_mm_images");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const J = std.json.Value;

const usage =
    \\usage: tf-glm-mm <command>
    \\  symbols          every kernel of the four images resolves (registers, threads, static shared memory, raised limit)
    \\  replay <dir>...  mm_oracle.py's recorded calls through the bindings, every buffer byte for byte against python-0.6
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

fn imgs() glm.Images {
    return .{ .qmm_group = images.glm_qmm_group, .qmm_prefill = images.glm_qmm_prefill, .experts = images.glm_experts, .experts_prefill = images.glm_experts_prefill };
}

fn run(gpu: Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "symbols")) return symbols(gpu);
    if (std.mem.eql(u8, cmd, "replay") and rest.len > 0) {
        var m = try glm.Mm.load(gpu.ctx, imgs());
        defer m.unload();
        for (rest) |dir| try replay(gpu, &m, dir);
        return;
    }
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

fn symbols(gpu: Gpu) !void {
    var m = try glm.Mm.load(gpu.ctx, imgs());
    defer m.unload();
    for (std.enums.values(glm.Kernel)) |k| {
        const f = m.fns[@intFromEnum(k)];
        std.debug.print("RESULT {s}: {d} registers, max {d} threads, {d} static shared bytes, {d} dynamic raised\n", .{ k.symbol(), try f.attribute(.num_regs), try f.attribute(.max_threads_per_block), try f.attribute(.shared_size_bytes), k.raised() });
    }
    std.debug.print("RESULT chip {d}.{d}, {d} SMs; decode experts resident a SM: gate+up {d}, down {d}\n", .{ m.chip.major, m.chip.minor, m.chip.sms, m.per_sm[0], m.per_sm[1] });
    check.pass("symbols: the {d} kernels of glm_qmm_group, glm_qmm_prefill, glm_experts and glm_experts_prefill resolve", .{glm.Kernel.count});
}

/// Every storage python's calls touched, on the device: its bytes before the first call, its expected bytes after.
const Storages = struct {
    gpu: Gpu,
    names: std.ArrayList([]const u8) = .empty,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(s: *Storages) void {
        for (s.bufs.items) |*b| b.free();
        s.bufs.deinit(s.gpu.gpa);
        s.names.deinit(s.gpu.gpa);
    }

    /// calls.tensor's resolver: the named storage's device address plus the view's byte offset.
    pub fn addr(s: *const Storages, t: J) !u64 {
        const name = (try glm.calls.get(t, "array")).string;
        const off: u64 = @intCast((try glm.calls.get(t, "off")).integer);
        for (s.names.items, s.bufs.items) |n, b| if (std.mem.eql(u8, n, name)) {
            if (off >= b.len) return error.BadCall;
            return b.ptr + off;
        };
        return error.BadCall;
    }
};

fn replay(gpu: Gpu, m: *const glm.Mm, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const root = fx.parsed.value;
    var s: Storages = .{ .gpu = gpu };
    defer s.deinit();
    const arrays = root.object.get("arrays").?.object;
    for (arrays.keys()) |name| {
        const host = try fx.bytes(name);
        defer gpu.gpa.free(host);
        try s.bufs.append(gpu.gpa, try cuda.DeviceBuffer.fromHost(gpu.d, host));
        try s.names.append(gpu.gpa, name);
    }
    try gpu.ctx.synchronize();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var launched: usize = 0;
    const recorded = root.object.get("calls").?.array.items;
    for (recorded, 0..) |call, i| {
        const ls = glm.calls.launches(call, m.chip, m.per_sm, &s) catch |e| {
            std.debug.print("FAIL {s}: call {d} ({s}): {t}\n", .{ dir, i, (try glm.calls.get(call, "fn")).string, e });
            return e;
        };
        for (ls.l[0..ls.n]) |l| try m.run(stream, l);
        launched += ls.n;
    }
    try stream.synchronize();
    var bytes: usize = 0;
    for (arrays.keys(), s.bufs.items) |name, b| {
        const after = arrays.get(name).?.object.get("after") orelse continue; // never written: equal to its upload
        const want = try fx.readFile(after.string);
        defer gpu.gpa.free(want);
        const got = try check.download(gpu, b);
        defer gpu.gpa.free(got);
        try check.sameBytes(name, got, want);
        bytes += got.len;
    }
    check.pass("BITEXACT glm_mm {s} ({s}): {d} calls, {d} launches; {d} written bytes equal to python-0.6's", .{ dir, try fx.string("what"), recorded.len, launched, bytes });
}

test "the tool compiles against the bindings (no GPU)" {
    std.testing.refAllDecls(@This());
    try std.testing.expect(glm.Kernel.count == 16);
}
