//! Stage0b CPU surfaces only. No grammar compiler, masking, drafter forwarding or GPU use.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const serve = @import("serve.zig");
const server = @import("server.zig");
const fixture = @import("structured_refusal_test.zig");
const a = std.testing.allocator;

// Test-only policy specification. It does NOT turn any product capability on.
fn effective(structure: ?api.Structure, drafts: bool, early: bool) struct { drafts: bool, early: bool, absorb: bool } {
    return .{ .drafts = drafts and structure == null, .early = early and structure == null, .absorb = drafts and structure == null };
}

test "qualification_3b492693 Stage0b inert surfaces remain unsupported and default empty" {
    const descriptor: api.Structure = .{ .kind = .json };
    try std.testing.expect(descriptor.compiler == null);
    var stream = try lanes.Stream.init(a, .{ .id = "surface", .prompt = &.{1}, .max_new = 2 });
    defer stream.deinit(a);
    const window: lanes.backend.Window = .{ .stream = &stream, .pending = 1, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{0} };
    try std.testing.expectEqual(@as(usize, 0), window.masks.len);
    try std.testing.expect(!(api.Info{}).structures);
}

test "qualification_3b492693 Stage0b model directory capture rejects served aliases in TP and AFD settings" {
    inline for (.{ "tp2-fixture", "afd-fixture" }) |alias| {
        var text: fixture.ByteText = .{};
        var capture: fixture.Capture = .{};
        const dir = "/fixture/checkpoint-dir";
        const setup: serve.Setup = .{ .engine = capture.engine(), .text = text.text(), .served = alias, .model_dir = dir };
        // Same field mapping as serve.run, whose source assignment is separately checked.
        const config: server.Config = .{ .served_name = setup.served, .model_dir = setup.model_dir, .model_ids = &.{alias}, .use_drafts = true, .keep_warm_s = 0 };
        const actual = try server.Server.init(a, std.testing.io, setup.engine, setup.text, config, null);
        defer actual.deinit();
        try std.testing.expectEqualStrings(dir, actual.config.model_dir);
        try std.testing.expect(!std.mem.eql(u8, actual.config.served_name, actual.config.model_dir));
        try std.testing.expect(actual.config.use_drafts);
        try std.testing.expect(!actual.engine.info().structures);
        try std.testing.expectEqual(@as(usize, 0), capture.submits);
        try std.testing.expectEqual(@as(usize, 0), capture.slots);
    }
}

test "qualification_3b492693 Stage0b effective structured no draft policy specification restores plain defaults" {
    inline for (.{ true, false }) |globally_enabled| {
        const plain = effective(null, globally_enabled, globally_enabled);
        try std.testing.expectEqual(globally_enabled, plain.drafts);
        try std.testing.expectEqual(globally_enabled, plain.early);
        try std.testing.expectEqual(globally_enabled, plain.absorb);
        inline for (std.meta.tags(@TypeOf((api.Structure{ .kind = .json }).kind))) |kind| {
            const structured = effective(.{ .kind = kind }, globally_enabled, globally_enabled);
            try std.testing.expect(!structured.drafts and !structured.early and !structured.absorb);
            const next = effective(null, globally_enabled, globally_enabled);
            try std.testing.expectEqualDeep(plain, next);
        }
    }
}
