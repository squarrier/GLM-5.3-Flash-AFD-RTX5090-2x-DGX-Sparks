// SPDX-License-Identifier: Apache-2.0
//! Private metadata bridge; native loader hookup awaits immutable I1+I2.
const std = @import("std");
const m = @import("cuda_hybrid_memory.zig");

pub const Selection = struct {
    role: m.Role,
    common: []const m.Weight,
    expected_common_ids: []const u64, // independent selected native ownership census, not shape hashes
    routed: m.Exl3,
    expected_routed_layers: []const u32, // selected config.isMoe plus config.layers when MTP is selected
    selected_routed_layers: []const u32, // actual descriptor traversal, not just a count
    mtp_layer: ?u32,
    source_binding: [32]u8,
};
/// No whole-checkpoint fraction and no fallback to cuda_native.weightShare on error.
/// The final loader adapter must bind these inventories independently, before payload I/O.
pub fn price(s: Selection) !m.WeightTotals {
    if (std.mem.allEqual(u8, &s.source_binding, 0)) return error.SourceUnbound;
    for (s.expected_common_ids) |id| {
        if (id == 0) return error.BadWeightIdentity;
        var found = false;
        for (s.common) |w| if (w.identity == id) { found = true; };
        if (!found) return error.MissingWeight;
    }
    for (s.common) |w| {
        var found = false;
        for (s.expected_common_ids) |id| if (w.identity == id) { found = true; };
        if (!found) return error.UnexpectedWeight;
    }
    if (s.selected_routed_layers.len != s.expected_routed_layers.len or s.routed.layers != s.expected_routed_layers.len) return error.MissingLayer;
    for (s.expected_routed_layers, s.selected_routed_layers, 0..) |expected, actual, i| {
        if (expected != actual) return error.MissingLayer;
        for (s.expected_routed_layers[0..i]) |prior| if (prior == expected) return error.DuplicateLayer;
    }
    if (s.mtp_layer) |mtp| {
        var found = false;
        for (s.selected_routed_layers) |layer| if (layer == mtp) { found = true; };
        if (!found) return error.MissingMtp;
    }
    return m.weightBytes(s.role, s.common, s.routed);
}
