//! qualification_4533b989 target-local CPU graph; generic tests/defaults unchanged.
const std = @import("std");
pub fn add(b: *std.Build, focused_path: []const u8) void {
    const safe = b.option(bool, "qualification_a295e4a6", "qualification_4533b989 socket-free tests in ReleaseSafe") orelse false;
    const opt: std.builtin.OptimizeMode = if (safe) .safe else .debug;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = b.graph.host, .optimize = opt });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = b.graph.host, .optimize = opt });
    const text: []const std.Build.Module.Import = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "lanes", .module = lanes } };
    const focused = b.createModule(.{ .root_source_file = b.path(focused_path), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = text });
    const focus_run = b.addRunArtifact(b.addTest(.{ .root_module = focused, .filters = &.{"qualification_4533b989"} }));
    focus_run.has_side_effects = true;
    b.step("test-qualification_cc01cee6", "qualification_4533b989 actual socket-free native focused tests").dependOn(&focus_run.step);
    const regression = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = text });
    const reg_step = b.step("test-qualification_fcd52b2c", "qualification_4533b989 existing socket-free server and lane/host regressions");
    const server_run = b.addRunArtifact(b.addTest(.{ .root_module = regression, .filters = @import("qualification_59878042_filters.zig").filters }));
    server_run.setCwd(.{ .cwd_relative = "/tmp" });
    server_run.has_side_effects = true; reg_step.dependOn(&server_run.step);
    const host_run = b.addRunArtifact(b.addTest(.{ .root_module = api }));
    host_run.setCwd(.{ .cwd_relative = "/tmp" });
    host_run.has_side_effects = true; reg_step.dependOn(&host_run.step);
    const lane_run = b.addRunArtifact(b.addTest(.{ .root_module = lanes }));
    lane_run.has_side_effects = true; reg_step.dependOn(&lane_run.step);
    const parser = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_parse_test.zig"), .target = b.graph.host, .optimize = opt, .link_libc = true, .imports = text });
    const parser_run = b.addRunArtifact(b.addTest(.{ .root_module = parser }));
    parser_run.has_side_effects = true; reg_step.dependOn(&parser_run.step);
}
