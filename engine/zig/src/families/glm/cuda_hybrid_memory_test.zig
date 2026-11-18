// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const m = @import("cuda_hybrid_memory.zig");
const t = std.testing;
const common = [_]m.Weight{
    .{ .identity = 1, .class = .attention, .stored = .bf16, .resident = .bf16, .elements = 32 },
    .{ .identity = 2, .class = .dense, .stored = .bf16, .resident = .bf16, .elements = 64 },
    .{ .identity = 3, .class = .shared, .stored = .bf16, .resident = .bf16, .elements = 128 },
    .{ .identity = 4, .class = .router, .stored = .bf16, .resident = .f32, .elements = 8 },
    .{ .identity = 5, .class = .embedding_head, .stored = .bf16, .resident = .bf16, .elements = 256 },
};
const routed: m.Exl3 = .{ .experts = 288, .hidden = 4096, .inter = 2048, .layers = 2 };

// RED baseline is the native generic world1 checkpoint price (cuda_native.weightShare),
// NOT a native execution test. Once present, exercise only the real private function.
fn price(role: m.Role, ws: []const m.Weight, ex: m.Exl3) !m.WeightTotals {
    if (@hasDecl(m, "weightBytes")) return m.weightBytes(role, ws, ex);
    return .{ .total = 100000000000 }; // old full-world estimate cannot provide a role plan
}
test "attention prices its native full common weights without routed checkpoint" {
    const p = try price(.attention, &common, routed);
    try t.expectEqual(@as(u64, 992), p.total);
    try t.expectEqual(@as(u64, 64), p.attention);
    try t.expectEqual(@as(u64, 128), p.dense);
    try t.expectEqual(@as(u64, 256), p.shared);
    try t.expectEqual(@as(u64, 32), p.router);
    try t.expectEqual(@as(u64, 512), p.embedding_head);
    try t.expectEqual(@as(u64, 0), p.routed);
}

test "durable two half and three epilogue planes survive multi tile calls in independent twins" {
    if (@hasDecl(m, "scratchBytes")) {
        const p = try m.scratchBytes(129, true);
        try t.expectEqual(@as(u64, 8454144), p.halves);
        try t.expectEqual(@as(u64, 12681216), p.epilogue);
        // Actual native hybrid arena packs these as views; no per-view alignment.
        try t.expectEqual(@as(u64, 8256), p.routed_weights);
        try t.expectEqual(@as(u64, 1032), p.shared_weights);
        try t.expectEqual(@as(u64, 21144648), p.total);
        try t.expectEqual(p.total / 2, (try m.scratchBytes(129, false)).total);
        try t.expectError(error.BadShape, m.scratchBytes(0, false));
        try t.expectError(error.Overflow, m.scratchBytes(std.math.maxInt(u64), true));
    } else try t.expect(false);
}

test "host accounting uses full native page stride plus separate keys and exact gather allocations" {
    if (@hasDecl(m, "hostPoolBytes")) {
        const p = try m.hostPoolBytes(2, 3, 11, 512, 128, 262208);
        try t.expectEqual(@as(u64, 6684672), p.page_stride);
        try t.expectEqual(@as(u64, 13369344), p.host_pool);
        try t.expectEqual(@as(u64, 786432), p.device_keys);
        try t.expectEqual(@as(u64, 19584), p.gather_arena_rows);
        try t.expectEqual(@as(u64, 404448592), p.device_gather);
        try t.expectEqual(@as(u64, 16), p.pinned_overflow);
        try t.expectError(error.Overflow, m.hostPoolBytes(std.math.maxInt(u64), std.math.maxInt(u64), 11, 512, 128, 262208));
        try t.expectError(error.BadShape, m.hostPoolBytes(3, 2, 11, 512, 128, 262208));
    } else try t.expect(false);
}

test "wire geometry preserves separate FP32 replies and native per pair mapped backing" {
    if (@hasDecl(m, "wireGeometry")) {
        const a = try m.wireGeometry(.attention, 128);
        const e = try m.wireGeometry(.expert0, 128);
        try t.expectEqual(@as(u64, 8404992), a.native_host_regions);
        try t.expectEqual(@as(u64, 32), a.native_device_state);
        try t.expectEqual(@as(u64, 1052672), a.request_per_pair);
        try t.expectEqual(@as(u64, 16777216), a.reply_per_pair);
        try t.expectEqual(a.native_host_regions / 2, e.native_host_regions);
        try t.expectEqual(@as(u64, 1048576), a.post_ceiling);
        try t.expectError(error.BadShape, m.wireGeometry(.attention, 129));
        try t.expectError(error.BadShape, m.wireGeometry(.attention, 0));
    } else try t.expect(false);
}

test "owner ledger counts backing identities once and keeps stage peaks distinct from external pins" {
    if (@hasDecl(m, "Plan")) {
        var p: m.Plan = .{};
        const weight: m.Allocation = .{ .identity = 1, .owner = .weights, .charge = .{ .device = 100 }, .stages = 7 };
        try p.put(weight);
        try p.put(weight); // borrowed view, not another allocation
        try p.put(.{ .identity = 2, .owner = .load_staging, .charge = .{ .host = 80, .pinned = 80 }, .stages = 1 });
        try p.put(.{ .identity = 3, .owner = .call_planes, .charge = .{ .device = 40 }, .stages = 6 });
        try p.put(.{ .identity = 4, .owner = .engram_external, .charge = .{ .host = 900, .pinned = 900, .registered = 900 }, .preexisting = true });
        const s = try p.summarize(&.{ .weights, .load_staging, .call_planes, .engram_external });
        try t.expectEqual(@as(usize, 4), p.len);
        try t.expectEqual(@as(u64, 140), s.peak.device);
        try t.expectEqual(@as(u64, 980), s.peak.host);
        try t.expectEqual(@as(u64, 80), s.incremental_peak.host);
        try t.expectEqual(@as(u64, 180), s.incremental_unified_peak);
        try t.expectEqual(@as(u64, 100), s.incremental[0].device);
        try t.expectEqual(@as(u64, 140), s.incremental[1].device);
        var conflict = weight;
        conflict.owner = .workspace;
        try t.expectError(error.IdentityConflict, p.put(conflict));
        try t.expectError(error.MissingOwner, p.summarize(&.{.sequence_state}));
        try p.put(.{ .identity = 5, .owner = .sequence_state, .charge = null });
        try t.expectError(error.UnknownAllocation, p.summarize(&.{.sequence_state}));
    } else try t.expect(false);
}

test "invalid ledger entries refuse transactionally before any size is trusted" {
    var p: m.Plan = .{};
    try t.expectError(error.BadAllocation, p.put(.{ .identity = 0, .owner = .gather, .charge = .{ .device = 1 } }));
    try t.expectError(error.BadAllocation, p.put(.{ .identity = 1, .owner = .host_pages, .charge = .{ .host = 8, .pinned = 9 } }));
    try t.expectError(error.BadAllocation, p.put(.{ .identity = 1, .owner = .transport, .charge = .{ .registered = 8 } }));
    try t.expectError(error.BadAllocation, p.put(.{ .identity = 1, .owner = .workspace, .charge = .{ .device = 8 }, .stages = 0 }));
    try t.expectEqual(@as(usize, 0), p.len);
    try p.put(.{ .identity = 1, .owner = .weights, .charge = .{ .device = std.math.maxInt(u64) } });
    try p.put(.{ .identity = 2, .owner = .workspace, .charge = .{ .device = 1 } });
    try t.expectError(error.Overflow, p.summarize(&.{ .weights, .workspace }));
    var full: m.Plan = .{};
    for (0..full.items.len) |i| try full.put(.{ .identity = i + 1, .owner = .weights, .charge = .{ .device = 1 } });
    try t.expectError(error.PlanFull, full.put(.{ .identity = full.items.len + 1, .owner = .weights, .charge = .{ .device = 1 } }));
}

test "paged sequence fixed arena uses native field alignment without duplicate latent rows" {
    if (@hasDecl(m, "pagedStateBytes")) {
        const p = try m.pagedStateBytes(.{ .kda_layers = 2, .mla_layers = 1, .heads = 4, .head_dim = 128, .row_width = 1536,
            .capacity = 1024, .index_dim = 128, .hidden = 4096, .key_dim = 128, .value_dim = 128 });
        try t.expectEqual(@as(u64, 2649856), p);
        try t.expectEqual(@as(u64, 256), try m.arenaBytes(&.{}));
        try t.expectEqual(@as(u64, 768), try m.arenaBytes(&.{ 1, 257, 0 }));
        try t.expectError(error.Overflow, m.arenaBytes(&.{std.math.maxInt(u64)}));
    } else try t.expect(false);
}

test "snapshot budget counts one preallocated pool not copies per retained lease" {
    if (@hasDecl(m, "snapshotPoolBytes")) {
        const host = try m.snapshotPoolBytes(1000, 256, true);
        const dev = try m.snapshotPoolBytes(1000, 256, false);
        try t.expectEqual(@as(u64, 768), host.host);
        try t.expectEqual(host.host, host.pinned);
        try t.expectEqual(host.host, host.registered);
        try t.expectEqual(@as(u64, 0), host.device);
        try t.expectEqual(@as(u64, 768), dev.device);
        try t.expectEqual(@as(u64, 0), dev.host);
        try t.expectError(error.BadShape, m.snapshotPoolBytes(1000, 0, false));
        try t.expectError(error.SnapshotBudget, m.snapshotPoolBytes(std.math.maxInt(u64), 256, false));
    } else try t.expect(false);
}

test "fresh admission debits one allocation peak without a reusable VRAM dividend" {
    if (@hasDecl(m, "Budget")) {
        var p: m.Plan = .{};
        try p.put(.{ .identity = 1, .owner = .weights, .charge = .{ .device = 120 } });
        try p.put(.{ .identity = 2, .owner = .host_pages, .charge = .{ .host = 80, .pinned = 80, .registered = 80 } });
        const obs: m.Observation = .{ .device_free = 300, .host_available = 500, .memlock_free = 200, .observed_ns = 10, .valid_until_ns = 20, .epoch = 77 };
        const policy: m.Policy = .{ .device_floor = 100, .host_floor = 200, .unified = false };
        var budget = try m.Budget.init(obs, policy, 10);
        const receipt = try budget.reserve(&p, &.{ .weights, .host_pages }, 11);
        try t.expectEqual(@as(u64, 77), receipt.epoch);
        try t.expectEqual(@as(u64, 80), budget.device_left);
        try t.expectEqual(@as(u64, 220), budget.host_left);
        try t.expectEqual(@as(u64, 120), budget.memlock_left);
        try t.expectError(error.DeviceFloor, budget.reserve(&p, &.{ .weights, .host_pages }, 11));
        try t.expectEqual(@as(u64, 80), budget.device_left);
        try t.expectError(error.StaleObservation, budget.reserve(&p, &.{.weights}, 21));
        try t.expectError(error.StaleObservation, m.Budget.init(obs, policy, 9));
        var low = obs;
        low.host_available = 199;
        try t.expectError(error.HostFloor, m.Budget.init(low, policy, 10));
    } else try t.expect(false);
}

test "unified memory debits device and host from one physical admission pool" {
    var p: m.Plan = .{};
    try p.put(.{ .identity = 1, .owner = .weights, .charge = .{ .device = 120, .host = 80, .pinned = 80 } });
    const obs: m.Observation = .{ .device_free = 300, .host_available = 500, .memlock_free = 200, .observed_ns = 10, .valid_until_ns = 20, .epoch = 77 };
    var b = try m.Budget.init(obs, .{ .device_floor = 100, .host_floor = 200, .unified = true }, 10);
    _ = try b.reserve(&p, &.{.weights}, 11);
    var tiny: m.Plan = .{};
    try tiny.put(.{ .identity = 2, .owner = .workspace, .charge = .{ .device = 1 } });
    try t.expectError(error.UnifiedFloor, b.reserve(&tiny, &.{.workspace}, 11));
    var limited = obs;
    limited.memlock_free = 79;
    var pin_budget = try m.Budget.init(limited, .{ .device_floor = 100, .host_floor = 200, .unified = false }, 10);
    try t.expectError(error.MemlockFloor, pin_budget.reserve(&p, &.{.weights}, 11));
    try t.expectEqual(@as(u64, 200), pin_budget.device_left);
}

test "typed close retains uncertain owned backing and cannot recycle its incarnation" {
    if (@hasDecl(m, "Accounting")) {
        var p: m.Plan = .{};
        try p.put(.{ .identity = 1, .owner = .transport, .charge = .{ .host = 80, .pinned = 80, .registered = 80 } });
        try p.put(.{ .identity = 2, .owner = .call_planes, .charge = .{ .device = 40 } });
        var a = try m.Accounting.init(.attention, p, &.{ .transport, .call_planes }, 99);
        try a.acquired(1);
        try a.acquired(2);
        try t.expectError(error.AlreadyAcquired, a.acquired(1));
        a.fail();
        try t.expectError(error.IdentityMismatch, a.close(.{ .incarnation = 98, .quiet = true, .retained_ids = &.{} }));
        try t.expectEqual(@as(u64, 40), (try a.status()).live.device);
        try a.close(.{ .incarnation = 99, .quiet = false, .retained_ids = &.{} });
        const s = try a.status();
        try t.expect(s.sticky_failure);
        try t.expect(!s.clean);
        try t.expectEqual(@as(u64, 80), s.retained.host);
        try t.expectEqual(@as(u64, 40), s.retained.device);
        try t.expectError(error.RetainedAllocations, a.assertReleased());
        try t.expectError(error.Closed, a.acquired(1));
        try t.expectError(error.Closed, a.close(.{ .incarnation = 99, .quiet = true, .retained_ids = &.{} }));
    } else try t.expect(false);
}

test "quiet is not a release receipt and explicit native quarantine is sticky" {
    var p: m.Plan = .{};
    try p.put(.{ .identity = 1, .owner = .transport, .charge = .{ .host = 80, .pinned = 80 } });
    try p.put(.{ .identity = 2, .owner = .call_planes, .charge = .{ .device = 40 } });
    var a = try m.Accounting.init(.attention, p, &.{ .transport, .call_planes }, 99);
    try a.acquired(1);
    try a.acquired(2);
    try t.expectError(error.InvalidCloseReceipt, a.close(.{ .incarnation = 99, .quiet = true, .retained_ids = &.{999} }));
    try t.expectEqual(@as(u64, 40), (try a.status()).live.device);
    try t.expectError(error.InvalidCloseReceipt, a.close(.{ .incarnation = 99, .quiet = true, .retained_ids = &.{1}, .released_ids = &.{1} }));
    try t.expectError(error.InvalidCloseReceipt, a.close(.{ .incarnation = 99, .quiet = false, .retained_ids = &.{}, .released_ids = &.{2} }));
    try a.close(.{ .incarnation = 99, .quiet = true, .retained_ids = &.{1}, .released_ids = &.{2} });
    const s = try a.status();
    try t.expectEqual(@as(u64, 80), s.retained.host);
    try t.expectEqual(@as(u64, 0), s.retained.device);
    try t.expect(!s.clean);
    try t.expect(s.sticky_failure);
    var omitted = try m.Accounting.init(.attention, p, &.{ .transport, .call_planes }, 100);
    try omitted.acquired(2);
    try omitted.close(.{ .incarnation = 100, .quiet = true, .retained_ids = &.{} });
    try t.expectEqual(@as(u64, 40), (try omitted.status()).retained.device);
}

test "allocation failure at every CPU boundary unwinds before typed cancellation balance" {
    if (@hasDecl(m.Accounting, "cancel")) {
        for (0..4) |fail_at| {
            var p: m.Plan = .{};
            for (0..3) |i| try p.put(.{ .identity = i + 1, .owner = .workspace, .charge = .{ .host = 8 } });
            var a = try m.Accounting.init(.expert0, p, &.{.workspace}, 101);
            var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = fail_at });
            const allocator = failing.allocator();
            var held: [3][]u8 = undefined;
            var ids: [3]u64 = undefined;
            var made: usize = 0;
            while (made < 3) : (made += 1) {
                held[made] = allocator.alloc(u8, 8) catch {
                    a.fail();
                    break;
                };
                ids[made] = made + 1;
                try a.acquired(ids[made]);
            }
            try t.expectEqual(@as(u64, @intCast(made * 8)), (try a.status()).live.host);
            for (held[0..made]) |buf| allocator.free(buf);
            try a.cancel(.{ .incarnation = 101, .quiet = true, .retained_ids = &.{}, .released_ids = ids[0..made] });
            try a.assertReleased();
            const s = try a.status();
            try t.expect(s.cancelled);
            try t.expect(s.clean);
            try t.expectEqual(fail_at < 3, s.sticky_failure);
            try t.expectEqual(3 - made, s.not_attempted);
            try t.expectEqual(failing.allocated_bytes, failing.freed_bytes);
        }
    } else try t.expect(false);
}

test "status serialization keeps shared capacity qualification and unknown counters distinct" {
    if (@hasDecl(m, "report")) {
        var p: m.Plan = .{};
        try p.put(.{ .identity = 1, .owner = .weights, .charge = .{ .device = 1 } });
        var a = try m.Accounting.init(.attention, p, &.{.weights}, 100);
        a.fail();
        const cap: m.Capacity = .{ .logical_per_request = 524288, .served_per_request = 262144, .allocated_shared_rows = 589824, .resident_users = 8, .qualified_per_request = null };
        const r = try m.report(&a, cap, null, null, @splat(1));
        const text = try std.json.Stringify.valueAlloc(t.allocator, r, .{});
        defer t.allocator.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, text, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        try t.expect(root.get("gather").? == .null);
        try t.expect(root.get("transport").? == .null);
        try t.expect(root.get("capacity").?.object.get("qualified_per_request").? == .null);
        try t.expect(root.get("status").?.object.get("sticky_failure").?.bool);
        try t.expectEqual(@as(i64, 589824), root.get("capacity").?.object.get("allocated_shared_rows").?.integer);
        try t.expectError(error.SourceUnbound, m.report(&a, cap, null, null, @splat(0)));
        var invalid = cap;
        invalid.served_per_request = 600000;
        try t.expectError(error.BadCapacity, m.report(&a, invalid, null, null, @splat(1)));
    } else try t.expect(false);
}

test "partial pinned or registered subsets require an explicit union instead of max inference" {
    var p: m.Plan = .{};
    try t.expectError(error.BadAllocation, p.put(.{ .identity = 1, .owner = .transport, .charge = .{ .host = 100, .pinned = 60, .registered = 60 } }));
    try t.expectEqual(@as(usize, 0), p.len);
}

test "all 43 native EXL3 layers fit their 13 independent upload backings before scratch" {
    var p: m.Plan = .{};
    for (0..43 * 13) |i| try p.put(.{ .identity = i + 1, .owner = .weights, .charge = .{ .device = 16 } });
    try t.expectEqual(@as(usize, 559), p.len);
    try t.expectEqual(@as(u64, 8944), (try p.summarize(&.{.weights})).peak.device);
}

test "missing weight identity cannot silently collapse equal shaped tensors" {
    var ws = [_]m.Weight{ common[0], common[0] };
    ws[0].identity = 0;
    ws[1].identity = 0;
    try t.expectError(error.BadWeightIdentity, price(.attention, &ws, routed));
}

test "load staging can retire before later served allocations without closing accounting" {
    if (@hasDecl(m.Accounting, "released")) {
        var p: m.Plan = .{};
        try p.put(.{ .identity = 1, .owner = .load_staging, .charge = .{ .host = 16 }, .stages = 1 });
        try p.put(.{ .identity = 2, .owner = .workspace, .charge = .{ .device = 8 }, .stages = 6 });
        var a = try m.Accounting.init(.expert0, p, &.{ .load_staging, .workspace }, 102);
        try a.acquired(1);
        try t.expectError(error.NoReleaseProof, a.released(.{ .incarnation = 102, .identity = 1, .quiet = true, .owned_free_completed = false }));
        try t.expectEqual(@as(u64, 16), (try a.status()).live.host);
        try a.released(.{ .incarnation = 102, .identity = 1, .quiet = true, .owned_free_completed = true });
        try t.expectError(error.AlreadyReleased, a.released(.{ .incarnation = 102, .identity = 1, .quiet = true, .owned_free_completed = true }));
        try a.acquired(2);
        try t.expectEqual(@as(u64, 0), (try a.status()).live.host);
        try t.expectEqual(@as(u64, 8), (try a.status()).live.device);
        try a.close(.{ .incarnation = 102, .quiet = true, .retained_ids = &.{}, .released_ids = &.{2} });
        try a.assertReleased();
    } else try t.expect(false);
}

test "native descriptor bridge refuses omitted source owners or selected MTP layer before pricing" {
    const ad = @import("cuda_hybrid_descriptor_adapter.zig");
    if (@hasDecl(ad, "price")) {
        var selection: ad.Selection = .{ .role = .attention, .common = &common, .expected_common_ids = &.{ 1, 2, 3, 4, 5 },
            .routed = routed, .expected_routed_layers = &.{ 3, 45 }, .selected_routed_layers = &.{ 3, 45 }, .mtp_layer = 45, .source_binding = @splat(1) };
        try t.expectEqual(@as(u64, 992), (try ad.price(selection)).total);
        selection.common = common[0..4];
        try t.expectError(error.MissingWeight, ad.price(selection));
        selection.common = &common;
        selection.selected_routed_layers = &.{3};
        try t.expectError(error.MissingLayer, ad.price(selection));
        selection.selected_routed_layers = &.{ 3, 45 };
        selection.mtp_layer = 46;
        try t.expectError(error.MissingMtp, ad.price(selection));
        selection.mtp_layer = 45;
        selection.source_binding = @splat(0);
        try t.expectError(error.SourceUnbound, ad.price(selection));
    } else try t.expect(false);
}

test "EXL3 incompatible geometry refuses before estimating any role" {
    var bad = routed;
    bad.inter = 2047;
    try t.expectError(error.UnsupportedGeometry, price(.expert0, &common, bad));
    bad = routed;
    bad.experts = 144;
    try t.expectError(error.UnsupportedGeometry, price(.expert1, &common, bad));
    bad = routed;
    bad.layers = 0;
    try t.expectError(error.UnsupportedGeometry, price(.attention, &common, bad));
    var huge = common;
    huge[0].elements = std.math.maxInt(u64);
    try t.expectError(error.Overflow, price(.attention, &huge, routed));
}

test "unverified source dtype conversion cannot become a plausible resident byte total" {
    var bad = common;
    bad[0].stored = .u8;
    try t.expectError(error.UnsupportedConversion, price(.attention, &bad, routed));
    bad = common;
    bad[0].stored = .f32;
    try t.expectError(error.UnsupportedConversion, price(.attention, &bad, routed));
}

test "repeated weight identity is counted once and contradictory alias refuses" {
    const dup = [_]m.Weight{ common[0], common[0] };
    const p = try price(.attention, &dup, routed);
    try t.expectEqual(@as(u64, 64), p.total);
    var bad = dup;
    bad[1].elements = 64;
    try t.expectError(error.IdentityConflict, price(.attention, &bad, routed));
}

test "both expert roles price all EXL3 ids on a channel half including MTP" {
    for ([_]m.Role{ .expert0, .expert1 }) |role| {
        const p = try price(role, &common, routed);
        // Native cuda_exl3.load: three MCG4 cuts, six F16 scale vectors,
        // three U64 pointer tables and one I32 k2 table, for every expert/layer.
        try t.expectEqual(@as(u64, 3641589504), p.routed);
        try t.expectEqual(p.routed, p.total);
        try t.expectEqual(@as(u64, 0), p.shared);
        try t.expectEqual(@as(u64, 0), p.attention);
    }
}
