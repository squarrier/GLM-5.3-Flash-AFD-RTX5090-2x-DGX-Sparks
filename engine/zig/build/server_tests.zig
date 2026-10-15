//! Ash's server tests on Linux (macOS adds them in build.zig's nativeServer): the server's unit tests, the tool parser's,
//! the engine seam's; `test-golden` (the frozen parity goldens through fake_serve); `fake-serve` (fake_serve into
//! zig-out/server for parity.py); and the CUDA GLM's serve path through the server on CPU (qualification_qualification_44ba554e.8).
const std = @import("std");

pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    @import("qualification_59878042_tests.zig").add(b, "zig/src/server/qualification_59878042_sse_test.zig");
    const mod = struct {
        fn of(bb: *std.Build, path: []const u8, imports: []const std.Build.Module.Import) *std.Build.Module {
            return bb.createModule(.{ .root_source_file = bb.path(path), .target = bb.graph.host, .optimize = .debug, .link_libc = true, .imports = imports });
        }
    };
    const lanes = mod.of(b, "zig/src/core/lanes/lanes.zig", &.{});
    const api = mod.of(b, "zig/src/core/engine_api.zig", &.{.{ .name = "lanes", .module = lanes }});
    const tokenizer = mod.of(b, "zig/src/core/tokenizer/tokenizer.zig", &.{});
    const template = mod.of(b, "zig/src/core/template/template.zig", &.{});
    const text: []const std.Build.Module.Import = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } };
    // qualification_214a6e4c: finalized parser-only diagnostics, no socket or model fixture.
    for ([_]std.builtin.OptimizeMode{ .debug, .safe }) |opt| {
        const diag = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_rejection_test.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = text });
        const run = b.addRunArtifact(b.addTest(.{ .root_module = diag }));
        run.has_side_effects = true;
        b.step(if (opt == .debug) "test-tool-rejection" else "test-tool-rejection-safe", "qualification_214a6e4c native final-decision checks, CPU no sockets").dependOn(&run.step);
    }
    for ([_]std.builtin.OptimizeMode{ .debug, .safe }) |opt| {
        const probe_mod = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_wire_probe.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = text });
        const probe = b.addExecutable(.{ .name = if (opt == .debug) "tool-wire-probe" else "tool-wire-probe-safe", .root_module = probe_mod });
        b.step(if (opt == .debug) "tool-wire-probe" else "tool-wire-probe-safe", "qualification_214a6e4c real routes over pipes, CPU no sockets").dependOn(&b.addInstallArtifact(probe, .{}).step);
    }
    // literal_tool_boundary: Mia GLM #158's literal-closer cases (literal_boundary's fixtures), strict: a failed boundary check fails the step.
    const literal_boundary_options = b.addOptions();
    literal_boundary_options.addOption(bool, "strict", true);
    const literal_boundary_imports: []const std.Build.Module.Import = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "lanes", .module = lanes }, .{ .name = "literal_boundary_options", .module = literal_boundary_options.createModule() } };
    for ([_]std.builtin.OptimizeMode{ .debug, .safe }) |opt| {
        const literal_boundary = b.step(if (opt == .debug) "test-literal_boundary" else "test-literal_boundary_safe", "literal_boundary tool-call boundary fixtures (text paths and routes), strict, CPU no sockets");
        for ([_][2][]const u8{ .{ "zig/src/server/literal_boundary_units_test.zig", "literal_boundary units" }, .{ "zig/src/server/literal_boundary_routes_test.zig", "literal_boundary routes" } }) |t| {
            const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path(t[0]), .target = host, .optimize = opt, .link_libc = true, .imports = literal_boundary_imports }), .filters = b.dupeStrings(&.{t[1]}) }));
            run.has_side_effects = true;
            literal_boundary.dependOn(&run.step);
        }
    }
    // tool_response_fixes: tool_api_fixes's fixtures (@Bizuayeu's b9528625, 557011805 and be17a101 cases: GLM key repair, end-token close,
    // Responses include); every case prints an tool_api_fixes-CASE line and a failed case fails its item's test.
    for ([_]std.builtin.OptimizeMode{ .debug, .safe }) |opt| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/server/tool_api_fixes_fixtures_test.zig"), .target = host, .optimize = opt, .link_libc = true, .imports = text }) }));
        run.has_side_effects = true;
        b.step(if (opt == .debug) "test-tool_api_fixes" else "test-tool_api_fixes_safe", "tool_api_fixes fixtures (GLM key repair, end-token close, Responses include), CPU no sockets").dependOn(&run.step);
    }
    const glm_lanes = mod.of(b, "zig/src/native/glm_cuda_lanes.zig", &.{.{ .name = "lanes", .module = lanes }});
    const server = mod.of(b, "zig/src/server/root.zig", &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "lanes", .module = lanes }, .{ .name = "glm_lanes", .module = glm_lanes } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = server })).step);
    const safe_server = b.createModule(.{ .root_source_file = b.path("zig/src/server/root.zig"), .target = host, .optimize = .safe, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "lanes", .module = lanes }, .{ .name = "glm_lanes", .module = glm_lanes } } });
    const safe_run = b.addRunArtifact(b.addTest(.{ .root_module = safe_server }));
    safe_run.has_side_effects = true;
    b.step("test-server-safe", "The server's CPU tests under ReleaseSafe").dependOn(&safe_run.step);
    const safety_run = b.addRunArtifact(b.addTest(.{ .root_module = safe_server, .filters = &.{ "nested ids outside u32", "key label outlives a reload" } }));
    safety_run.has_side_effects = true;
    b.step("test-server-safety", "ReleaseSafe nested-token and live-label reload regressions, no model").dependOn(&safety_run.step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod.of(b, "zig/src/server/tool_parse_test.zig", text) })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api })).step);
    // the CUDA GLM's lane backend under Ash's server with a scripted forward (the kit's tokenizer: TF_GLM_TOKENIZER)
    const serve_test = b.addTest(.{ .root_module = mod.of(b, "zig/tests/server/glm_cuda_serve.zig", &.{ .{ .name = "server", .module = server }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "glm_lanes", .module = glm_lanes } }) });
    const serve_run = b.addRunArtifact(serve_test);
    serve_run.has_side_effects = true; // TF_GLM_TOKENIZER is not an input the cache sees
    step.dependOn(&serve_run.step);
    b.step("test-glm-serve", "The CUDA GLM's serve path through Ash's server on CPU (TF_GLM_TOKENIZER: the kit's GLM-5.3 files)").dependOn(&serve_run.step);
    // the frozen parity goldens, checked by Zig driving fake_serve
    const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = mod.of(b, "zig/tests/server/fake_serve.zig", &.{ .{ .name = "server", .module = server }, .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } }) });
    const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/server/golden_check.zig"), .target = host, .optimize = .debug, .link_libc = true }) });
    const golden_run = b.addRunArtifact(golden_check);
    golden_run.addArtifactArg(fake_serve);
    golden_run.addFileArg(b.path("zig/tests/server/golden/cases.json"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/golden"));
    golden_run.addDirectoryArg(b.path("zig/tests/server/fixtures"));
    golden_run.setCwd(b.path("."));
    b.step("test-golden", "The frozen server parity goldens checked by a Zig step driving fake_serve").dependOn(&golden_run.step);
    b.step("fake-serve", "fake_serve into zig-out/server, for zig/tests/server/parity.py").dependOn(&b.addInstallArtifact(fake_serve, .{ .dest_dir = .{ .override = .{ .custom = "server" } } }).step);
}
