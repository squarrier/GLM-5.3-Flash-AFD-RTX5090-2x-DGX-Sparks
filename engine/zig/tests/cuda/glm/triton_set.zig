//! GLM-5.3-Flash's Triton kernel set (kernels.json, tools/zig/glm_aot.py's spec) against every launch of the Python
//! wrappers' fixture (src/families/glm/fixtures_cuda_triton.json), by aot.zig's own rule: each launch finds exactly one
//! variant, as `aot.Set.find` will at run time on the cubins `glm_aot.py build` makes from this spec for sm_121a or
//! sm_120a. CPU only: no cubins, no GPU. cuda_triton_test.zig shows the Zig wrappers launch what the fixture holds.

const std = @import("std");
const aot = @import("cuda").aot;
const glm = @import("../../../src/families/glm/cuda_triton_test.zig");

const testing = std.testing;
const spec_text = @embedFile("kernels.json");

const Entry = struct { spec: aot.Spec, derived: bool, function: []const u8 };

/// The spec's entries as aot.json rows: the launch facts `glm_aot.py build` writes (aot_row, zig_const).
fn entries(arena: std.mem.Allocator, root: std.json.Value) ![]Entry {
    const kernels = root.object.get("kernels").?.array.items;
    const out = try arena.alloc(Entry, kernels.len);
    for (kernels, out) |k, *e| {
        const o = k.object;
        const sig = o.get("signature").?.object;
        const attrs = o.get("attrs").?.object;
        var params: std.ArrayList(aot.SpecParam) = .empty;
        for (o.get("params").?.array.items) |pv| {
            const n = pv.string;
            const ty = sig.get(n).?.string;
            if (std.mem.eql(u8, ty, "constexpr")) continue;
            const div = if (attrs.get(n)) |a| a.array.items.len > 0 else false;
            try params.append(arena, .{ .name = n, .type = ty, .div16 = div, .nospec = false });
        }
        var consts: std.json.ArrayHashMap(aot.SpecConst) = .{};
        var it = o.get("constexprs").?.object.iterator();
        while (it.next()) |kv| {
            const v = kv.value_ptr.object;
            var c: aot.SpecConst = .{};
            if (v.get("bool")) |b| {
                c.int = @intFromBool(b.bool);
            } else if (v.get("int")) |x| {
                c.int = x.integer;
            } else if (v.get("fp64_bits")) |bits| {
                const word = try std.fmt.parseInt(u64, bits.string[2..], 16);
                const x: f64 = @bitCast(word);
                c.f32 = @bitCast(@as(f32, @floatCast(x)));
            }
            try consts.map.put(arena, kv.key_ptr.*, c);
        }
        const name = o.get("name").?.string;
        e.* = .{
            .spec = .{ .@"fn" = name, .hash = "", .name = name, .num_warps = 0, .params = params.items, .consts = consts },
            .derived = o.get("rule") != null,
            .function = o.get("function").?.string,
        };
    }
    return out;
}

test "every fixture launch finds exactly one variant of the kernel set; every captured variant serves one" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, spec_text, .{});
    const set = try entries(arena, root);
    const used = try arena.alloc(usize, set.len);
    @memset(used, 0);
    var f = try glm.Fixture.load(testing.allocator);
    defer f.deinit();
    var launches: usize = 0;
    var it = f.cases().iterator();
    while (it.next()) |case| {
        for (case.value_ptr.array.items) |l| {
            const o = l.object;
            const name = o.get("fn").?.string;
            var args: [16]aot.Arg = undefined;
            var na: usize = 0;
            for (o.get("args").?.array.items) |a| {
                const ao = a.object;
                const an = ao.get("name").?.string;
                const ty = ao.get("type").?.string;
                args[na] = if (ty[0] == '*')
                    aot.ptr(an, ty, if (ao.get("align16").?.bool) 0x7f0000000000 else 0x7f0000000008)
                else if (ao.get("int")) |x|
                    aot.int(an, @intCast(x.integer))
                else
                    aot.float(an, @bitCast(@as(u32, @intCast(ao.get("f32").?.integer))));
                na += 1;
            }
            var consts: [12]aot.Const = undefined;
            var nc: usize = 0;
            var ci = o.get("consts").?.object.iterator();
            while (ci.next()) |kv| {
                if (kv.value_ptr.* == .null) continue;
                const v = kv.value_ptr.object;
                consts[nc] = if (v.get("int")) |x| aot.ci(kv.key_ptr.*, x.integer) else aot.cf(kv.key_ptr.*, @bitCast(@as(u32, @intCast(v.get("f32").?.integer))));
                nc += 1;
            }
            var hits: usize = 0;
            var at: usize = 0;
            for (set, 0..) |e, i| {
                if (!std.mem.eql(u8, e.spec.@"fn", name) or !aot.accepts(e.spec, args[0..na], consts[0..nc])) continue;
                hits += 1;
                at = i;
            }
            if (hits != 1) {
                std.debug.print("case {s}: {s} matches {d} variants\n", .{ case.key_ptr.*, name, hits });
                return error.NotOneVariant;
            }
            used[at] += 1;
            launches += 1;
        }
    }
    var captured: usize = 0;
    for (set, used) |e, n| {
        if (e.derived) continue; // derived variants serve launches past the cases (other row counts and buckets)
        captured += 1;
        if (n == 0) {
            std.debug.print("captured variant of {s} serves no fixture launch\n", .{e.function});
            return error.DeadVariant;
        }
    }
    try testing.expect(launches > 0 and captured > 0);
}
