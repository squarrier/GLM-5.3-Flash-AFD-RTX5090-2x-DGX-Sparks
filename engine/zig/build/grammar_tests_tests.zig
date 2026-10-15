//! grammar_tests socket-free CPU constructor tests; no kernel images or device work.
const std = @import("std");
const glm_build = @import("glm_cuda.zig");
const serve_build = @import("glm_serve.zig");

fn module(b: *std.Build, path: []const u8, mode: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path(path), .target = b.graph.host, .optimize = mode, .link_libc = true });
}

pub fn register(b: *std.Build, mode: std.builtin.OptimizeMode, draft_ids: *std.Build.Module) void {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = module(b, "zig/src/cuda/root.zig", mode);
    cuda.addOptions("kernel_options", options);
    cuda.addCSourceFile(.{ .file = b.path("zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2" } });
    cuda.addImport("stagger", module(b, "zig/src/core/stagger.zig", mode));
    const tokenizer = module(b, "zig/src/core/tokenizer/tokenizer.zig", mode);
    const core = module(b, "zig/src/core/root.zig", mode);
    core.addImport("tokenizer", tokenizer);
    const lanes = module(b, "zig/src/core/lanes/lanes.zig", mode);
    const api = module(b, "zig/src/core/engine_api.zig", mode);
    api.addImport("lanes", lanes);
    const glm = glm_build.module(b, b.graph.host, mode, cuda, core);
    glm.addImport("lanes", lanes);
    const nemotron = module(b, "zig/src/families/nemotron/cuda.zig", mode);
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    const native = module(b, "zig/src/native/cuda.zig", mode);
    native.addImport("cuda", cuda);
    native.addImport("engine_api", api);
    native.addImport("lanes", lanes);
    native.addImport("nemotron", nemotron);
    native.addImport("glm_cuda", glm);
    native.addImport("glm_tool", serve_build.tool(b, b.graph.host, mode, glm, cuda, core));
    const bound = module(b, "zig/tests/cuda/afd_serve_lanes.zig", mode);
    bound.addImport("cuda", cuda);
    bound.addImport("glm_cuda", glm);
    bound.addImport("engine_api", api);
    bound.addImport("lanes", lanes);
    const gl = module(b, "zig/src/native/glm_cuda_lanes.zig", mode);
    gl.addImport("lanes", lanes);
    bound.addImport("glm_lanes", gl);
    const data = module(b, "zig/tests/cuda/grammar_tests_data.zig", mode);
    data.addImport("glm_cuda", glm);
    data.addImport("lanes", lanes);
    native.addImport("grammar_tests_data", data);
    bound.addImport("grammar_tests_data", data);
    const sync = module(b, "zig/tests/cuda/grammar_tests_c_rendezvous.zig", mode);
    sync.addImport("lanes", lanes);
    native.addImport("grammar_tests_c_sync", sync);
    bound.addImport("grammar_tests_c_sync", sync);
    const contract = module(b, "zig/tests/cuda/grammar_tests_contracts.zig", mode);
    contract.addImport("glm_cuda", glm);
    contract.addImport("lanes", lanes);
    contract.addImport("grammar_tests_data", data);
    const peers = module(b, "zig/tests/cuda/grammar_tests_c_peers.zig", mode);
    peers.addImport("lanes", lanes);
    // Share the rendezvous file as a module instead of compiling it through two roots.
    peers.addImport("grammar_tests_c_sync", sync);
    const chat = module(b, "zig/src/server/chat.zig", mode);
    chat.addImport("engine_api", api);
    chat.addImport("lanes", lanes);
    chat.addImport("glm_lanes", gl);
    chat.addImport("tokenizer", tokenizer);
    chat.addImport("template", module(b, "zig/src/core/template/template.zig", mode));
    const delivery = b.step("test-grammar_tests_c3-delivery", "grammar_tests_c3 CPU-only combined compiler lifetime and delivery");
    const delivery_run = b.addRunArtifact(b.addTest(.{ .root_module = chat, .filters = &.{"grammar_tests_c3"} }));
    delivery_run.has_side_effects = true;
    delivery.dependOn(&delivery_run.step);
    const lifetime = b.step("test-grammar_tests_c3-lifetime", "grammar_tests_c3 CPU-only owned request compiler lifetime");
    const lifetime_run = b.addRunArtifact(b.addTest(.{ .root_module = api, .filters = &.{"grammar_tests_c3"} }));
    lifetime_run.has_side_effects = true;
    lifetime.dependOn(&lifetime_run.step);
    const committed = b.step("test-grammar_tests_c5-fault", "grammar_tests_c5 CPU-only committed compute clone provider fail-stop");
    for ([_]*std.Build.Module{native, bound}) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"grammar_tests_c5"} }));
        run.has_side_effects = true;
        committed.dependOn(&run.step);
    }
    const borrowed = b.step("test-grammar_tests_c7", "grammar_tests_c7 CPU-only actual Host/LaneHost peer-frame allocation and borrowed request slices");
    for ([_]*std.Build.Module{native, bound}) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"grammar_tests_c7"} }));
        run.has_side_effects = true;
        borrowed.dependOn(&run.step);
    }
    const faults = b.step("test-grammar_tests_c3-fault", "grammar_tests_c3 CPU-only actual constructor fail-stop contracts");
    for ([_]*std.Build.Module{native, bound}) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"grammar_tests_c3"} }));
        run.has_side_effects = true;
        faults.dependOn(&run.step);
    }
    const coordinated = b.step("test-grammar_tests_c", "grammar_tests_c CPU-only actual constructor and peer contracts");
    for ([_]*std.Build.Module{native, peers, bound, api, chat}) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"grammar_tests_c"} }));
        run.has_side_effects = true;
        coordinated.dependOn(&run.step);
    }
    const step = b.step("test-grammar_tests", "grammar_tests CPU-only real Host/Armed and Bound constructor contracts");
    for ([_]*std.Build.Module{ native, bound, contract, api }) |m| {
        const test_exe = b.addTest(.{ .root_module = m, .filters = &.{"grammar_tests"} });
        const run = b.addRunArtifact(test_exe);
        run.has_side_effects = true; // exact-head gates must execute, not report cached run steps
        step.dependOn(&run.step);
    }
    const masked = b.step("test-grammar_mask", "grammar_mask CPU-only real masked adapter through the actual AFD Bound chain (MOCK device, FAKE matcher)");
    const masked_run = b.addRunArtifact(b.addTest(.{ .root_module = bound, .filters = &.{"grammar_mask"} }));
    masked_run.has_side_effects = true;
    masked.dependOn(&masked_run.step);
    // structured_users: grammar-constrained sampling on the actual Bound chain and its host draw (users lanes: test-structured_users_users)
    const structured_users = b.step("test-structured_users", "structured_users CPU-only grammar-constrained sampling: MOCK device rows through the host nucleus draw, the recorded masked draw and the actual AFD Bound chain");
    const structured_users_run = b.addRunArtifact(b.addTest(.{ .root_module = bound, .filters = &.{"structured_users"} }));
    structured_users_run.has_side_effects = true;
    structured_users.dependOn(&structured_users_run.step);
    // grammar_mask real ABI: grammar_abi's published library through its own loader/ownership sources (read-only view root)
    const real = b.step("test-grammar_mask_abi", "grammar_mask real-ABI: grammar_abi's published libtfgrammar (TF_grammar_mask_LIB, TF_grammar_mask_LIB_SHA256, TF_grammar_mask_TOKENIZER) through the actual chain");
    const grammar_abi_root = b.option([]const u8, "grammar_abi", "grammar_mask: a root.zig re-exporting grammar_abi's tfg_trusted.zig and tfg_owned.zig (test-grammar_mask_abi)");
    if (grammar_abi_root) |root| {
        const abi = b.createModule(.{ .root_source_file = .{ .cwd_relative = root }, .target = b.graph.host, .optimize = mode, .link_libc = true });
        const m = module(b, "zig/tests/cuda/grammar_mask_abi.zig", mode);
        m.addImport("cuda", cuda);
        m.addImport("glm_cuda", glm);
        m.addImport("engine_api", api);
        m.addImport("lanes", lanes);
        m.addImport("glm_lanes", gl);
        m.addImport("grammar_tests_data", data);
        m.addImport("grammar_tests_c_sync", sync);
        m.addImport("grammar_abi", abi);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{"grammar_mask-ABI"} }));
        run.has_side_effects = true;
        real.dependOn(&run.step);
    } else real.dependOn(&b.addFail("test-grammar_mask_abi needs -Dgram2abi=<root.zig re-exporting grammar_abi's tfg_trusted.zig and tfg_owned.zig>").step);
    // grammar_server-WIRE begin: default-off identity; this block and its test file also run on the pristine baseline (wire.sh)
    {
        const wire_server = module(b, "zig/src/server/root.zig", mode);
        wire_server.addImport("engine_api", api);
        wire_server.addImport("tokenizer", tokenizer);
        wire_server.addImport("template", module(b, "zig/src/core/template/template.zig", mode));
        const wire = module(b, "zig/tests/cuda/grammar_server_wire.zig", mode);
        wire.addImport("cuda", cuda);
        wire.addImport("glm_cuda", glm);
        wire.addImport("engine_api", api);
        wire.addImport("lanes", lanes);
        wire.addImport("glm_lanes", gl);
        wire.addImport("grammar_tests_data", data);
        wire.addImport("grammar_tests_c_sync", sync);
        wire.addImport("server", wire_server);
        const wire_run = b.addRunArtifact(b.addTest(.{ .root_module = wire, .filters = &.{"grammar_server-WIRE"} }));
        wire_run.has_side_effects = true;
        b.step("test-grammar_server_wire", "grammar_server default-off identity: raw HTTP replies and the trace of the served AFD chain, every TF_AFD_GRAMMAR_* unset (grammar_server_WIRE_OUT)").dependOn(&wire_run.step);
    }
    // grammar_server-WIRE end
    // grammar_server: served compiler ownership and response_format wiring for the attention role (default off). CPU: Ash's
    // server over loopback, the lane host and the actual AFD chain (MOCK device, FAKE matcher); the role over the stub a
    // tool without -Dgram2abi gets. REAL: grammar_abi's published library through the role (-Dgram2abi, TF_grammar_server_*).
    const g2_server = module(b, "zig/src/server/root.zig", mode);
    g2_server.addImport("engine_api", api);
    g2_server.addImport("tokenizer", tokenizer);
    g2_server.addImport("template", module(b, "zig/src/core/template/template.zig", mode));
    for ([_]struct { []const u8, []const u8, []const u8, ?[]const u8 }{
        .{ "test-grammar_server", "grammar_server CPU: lazy compiler, sticky refusals, release response_format mapping, the lane host's role Rules, env refusals (MOCK device, FAKE matcher)", "zig/tests/cuda/grammar_server_test.zig", null },
        .{ "test-grammar_server_abi", "grammar_server real-ABI: grammar_abi's published libtfgrammar through the attention role's lazy compiler and the served chain (TF_grammar_server_LIB, TF_grammar_server_LIB_SHA256, TF_grammar_server_MODEL)", "zig/tests/cuda/grammar_server_abi.zig", grammar_abi_root },
    }) |k| {
        const g2_step = b.step(k[0], k[1]);
        const real_abi = std.mem.endsWith(u8, k[0], "-abi");
        if (real_abi and k[3] == null) {
            g2_step.dependOn(&b.addFail("test-grammar_server_abi needs -Dgram2abi=<root.zig re-exporting grammar_abi's tfg_trusted.zig and tfg_owned.zig>").step);
            continue;
        }
        const abi = if (k[3]) |root| b.createModule(.{ .root_source_file = .{ .cwd_relative = root }, .target = b.graph.host, .optimize = mode, .link_libc = true }) else module(b, "zig/tests/cuda/afd_attn_noabi.zig", mode);
        const m = module(b, k[2], mode);
        m.addImport("cuda", cuda);
        m.addImport("glm_cuda", glm);
        m.addImport("engine_api", api);
        m.addImport("lanes", lanes);
        m.addImport("glm_lanes", gl);
        m.addImport("grammar_tests_data", data);
        m.addImport("grammar_tests_c_sync", sync);
        m.addImport("server", g2_server);
        m.addImport("grammar_abi", abi);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m, .filters = &.{if (real_abi) "grammar_server-ABI" else "grammar_server "} }));
        run.has_side_effects = true;
        g2_step.dependOn(&run.step);
    }
}
