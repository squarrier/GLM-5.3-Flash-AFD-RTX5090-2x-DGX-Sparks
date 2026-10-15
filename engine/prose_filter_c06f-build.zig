//! prose_filter_852a isolated CPU test graph, no model, sockets, native CUDA or AOT.
const std = @import("std");
pub fn build(b: *std.Build) void {
    const opt: std.builtin.OptimizeMode = if (b.option(bool, "safe", "ReleaseSafe") orelse false) .safe else .debug;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .optimize = opt });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = b.graph.host, .optimize = opt });
    const imports: []const std.Build.Module.Import = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "lanes", .module = lanes } };
    const focus = b.createModule(.{ .root_source_file = b.path("zig/src/server/prose_filter_c06f_test.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = imports });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = focus, .filters = &.{"prose_filter_852a"} }));
    run.has_side_effects = true;
    b.step("focus", "New actual route memory-sink fixtures").dependOn(&run.step);
    const regression = b.step("regression", "Inherited socket-free server/parser/rejection regression");
    const srv = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = imports });
    const sr = b.addRunArtifact(b.addTest(.{ .root_module = srv, .filters = @import("zig/build/qualification_59878042_filters.zig").filters }));
    sr.setCwd(.{ .cwd_relative = "/tmp" });
    sr.has_side_effects = true;
    regression.dependOn(&sr.step);
    for ([_][]const u8{ "zig/src/server/tool_parse_test.zig", "zig/src/server/tool_rejection_test.zig" }) |path| {
        const m = b.createModule(.{ .root_source_file = b.path(path), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = imports });
        const r = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        r.has_side_effects = true;
        regression.dependOn(&r.step);
    }
}
