//! qualification_064ac7c1's kit (qualification_955b7191): `tf-glm-afd` (zig/tests/cuda/tf_glm_afd.zig, the AFD roles' lab tool) for Linux targets, outside
//! `install`, and the kit's CPU tests in `zig build test` and `zig build test-glm-afd`: the world-1 kernel set's cover
//! (G-cover-120) and the tool's flags, summary line and exit codes. build.zig calls `hostTests`, and zig/build/cuda.zig
//! calls `tool` over its runtime (as xchg's tool).
const std = @import("std");
const glm_build = @import("glm_cuda.zig");

/// The host tests' runtime module, without kernel images, as zig/build/glm_fwd_tool.zig's (the RoCE proxy where verbs is).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    const verbs = target.result.os.tag == .linux and if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// The runtime and the GLM family over it: the host tests' (`family`), or zig/build/cuda.zig's for the tool.
pub const Mods = struct { cuda: *std.Build.Module, glm: *std.Build.Module };

/// The GLM family module over its own runtime and core (zig/build/glm_cuda.zig's `module`).
fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Mods {
    const cuda = runtime(b, target, optimize);
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    return .{ .cuda = cuda, .glm = glm_build.module(b, target, optimize, cuda, core) };
}

fn module(b: *std.Build, path: []const u8, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Mods) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = mods.glm }, .{ .name = "cuda", .module = mods.cuda } },
    });
}

/// qualification_4be15bbf: attn serve's modules on the tool's root: the lane core, the engine seam (api.LaneHost), qualification_qualification_e91d6b91's lane backend
/// (native/glm_cuda_lanes.zig) and Ash's server (zig/src/server/root.zig over the family's tokenizer, as zig/build/cuda.zig's
/// native server takes it: one module per file). The HTTP side keeps its safety checks (cuda.zig's rule).
fn serveImports(b: *std.Build, root: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Mods) void {
    const of = struct {
        fn m(bb: *std.Build, path: []const u8, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode, imports: []const std.Build.Module.Import) *std.Build.Module {
            return bb.createModule(.{ .root_source_file = bb.path(path), .target = t, .optimize = o, .link_libc = true, .imports = imports });
        }
    }.m;
    const http: std.builtin.OptimizeMode = if (optimize == .debug) .debug else .ReleaseSafe;
    const tokenizer = mods.glm.import_table.get("core").?.import_table.get("tokenizer").?;
    const lanes = mods.glm.import_table.get("lanes").?;
    const api = of(b, "zig/src/core/engine_api.zig", target, optimize, &.{.{ .name = "lanes", .module = lanes }});
    const template = of(b, "zig/src/core/template/template.zig", target, http, &.{});
    const server = of(b, "zig/src/server/root.zig", target, http, &.{ .{ .name = "engine_api", .module = api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template } });
    root.addImport("lanes", lanes);
    root.addImport("engine_api", api);
    root.addImport("glm_lanes", of(b, "zig/src/native/glm_cuda_lanes.zig", target, optimize, &.{.{ .name = "lanes", .module = lanes }}));
    root.addImport("glm_users", of(b, "zig/src/native/glm_cuda_users.zig", target, optimize, &.{ .{ .name = "lanes", .module = lanes }, .{ .name = "glm_cuda", .module = mods.glm }, .{ .name = "engine_api", .module = api } })); // user_drafts: engine_api (the lone driver)
    root.addImport("server", server);
    // grammar_server: grammar_abi's loader and ownership for TF_AFD_GRAMMAR_SERVE (-Dgram2abi: its read-only view root), else a
    // stub under which the attention role refuses every structured request by name
    const grammar_abi = if (given(b, "grammar_abi")) |path| b.createModule(.{ .root_source_file = .{ .cwd_relative = path }, .target = target, .optimize = optimize, .link_libc = true }) else of(b, "zig/tests/cuda/afd_attn_noabi.zig", target, optimize, &.{});
    root.addImport("grammar_abi", grammar_abi);
}

/// grammar_tests's test modules (grammar_tests_tests.zig's `grammar_tests_data` and `grammar_tests_c_sync`) for a host test root that reaches
/// afd_serve_lanes.zig's grammar_tests/grammar_mask tests (through afd_attn_serve.zig); build wiring only, no product module.
fn grammar_tests_imports(b: *std.Build, root: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Mods) void {
    const lanes = mods.glm.import_table.get("lanes").?;
    root.addImport("grammar_tests_data", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_data.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "glm_cuda", .module = mods.glm }, .{ .name = "lanes", .module = lanes } } }));
    root.addImport("grammar_tests_c_sync", b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/grammar_tests_c_rendezvous.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} }));
}

/// afd_cover.zig, afd_cover_f8.zig (qualification_7877904d), afd_cover_f8_mtp.zig (mtp_fp8) and the tool's tests join `zig build test`;
/// `zig build test-glm-afd` runs them alone.
pub fn hostTests(b: *std.Build, step: *std.Build.Step) void {
    const host = b.graph.host;
    const mods = family(b, host, .debug);
    const cover = module(b, "zig/tests/cuda/glm/afd_cover.zig", host, .debug, mods);
    cover.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    cover.addAnonymousImport("afd_attn_full_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn_full.json") });
    const cover_f8 = module(b, "zig/tests/cuda/glm/afd_cover_f8.zig", host, .debug, mods); // qualification_7877904d: the FP8 head's set120
    cover_f8.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    const cover_mtp8 = module(b, "zig/tests/cuda/glm/afd_cover_f8_mtp.zig", host, .debug, mods); // mtp_fp8: the MTP layer on FP8
    cover_mtp8.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    // mtp_fp8: qualification_7877904d's cover helpers as a module of their own (its tests stay in cover_f8's binary, not run twice)
    const cover_f8_lib = module(b, "zig/tests/cuda/glm/afd_cover_f8.zig", host, .debug, mods);
    cover_f8_lib.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    cover_mtp8.addImport("afd_cover_f8", cover_f8_lib);
    const own = b.step("test-glm-afd", "The AFD kit's CPU tests: the world-1 kernel set's cover (G-cover-120), tf-glm-afd's flags and attn --dry");
    const opt: std.builtin.OptimizeMode = if (b.option(bool, "concurrent_users_safe", "concurrent_users focused CPU tests in ReleaseSafe") orelse false) .ReleaseSafe else .debug;
    const focused_mods = family(b, host, opt);
    const focused = module(b, "zig/tests/cuda/afd_users_test.zig", host, opt, focused_mods);
    serveImports(b, focused, host, opt, focused_mods);
    grammar_tests_imports(b, focused, host, opt, focused_mods); // grammar_tests's AFD tests, reached through afd_attn_serve.zig
    focused.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    focused.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const concurrent_users_run = b.addRunArtifact(b.addTest(.{ .root_module = focused }));
    concurrent_users_run.has_side_effects = true;
    b.step("test-concurrent_users", "Independent-user CPU identity, ownership, cancellation and failure fixtures").dependOn(&concurrent_users_run.step);
    step.dependOn(&concurrent_users_run.step);
    // structured_users: the structured users lanes alone (they also run in test-concurrent_users above); afd_users_test.zig reaches
    // afd_users_device_test.zig (and so structured_users_users_test.zig) through its concurrent_users compile test, so that one is kept
    const structured_users_users = b.addRunArtifact(b.addTest(.{ .root_module = focused, .filters = &.{ "structured_users", "concurrent_users real device and served path compile without GPU" } }));
    structured_users_users.has_side_effects = true;
    b.step("test-structured_users_users", "structured_users CPU-only structured output with TF_AFD_USERS=2: two lanes, a matcher per request, retirement").dependOn(&structured_users_users.step);
    // shared_pool: the users' shared cache pool (TF_AFD_POOL_TOKENS): CPU fixtures and recorded real users (-Dconc2-safe too)
    const shared_pool = module(b, "zig/tests/cuda/afd_pool_test.zig", host, opt, focused_mods);
    serveImports(b, shared_pool, host, opt, focused_mods);
    grammar_tests_imports(b, shared_pool, host, opt, focused_mods);
    shared_pool.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    shared_pool.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const shared_pool_run = b.addRunArtifact(b.addTest(.{ .root_module = shared_pool }));
    shared_pool_run.has_side_effects = true;
    b.step("test-shared_pool", "shared_pool shared cache pool: switch, placement, compaction, admission, release, recorded pooled users").dependOn(&shared_pool_run.step);
    step.dependOn(&shared_pool_run.step);
    // user_sampling: TF_AFD_USERS_SAMPLING's backend rules over recorded real users (the switch's claim, keyed rows, cancel and
    // failure releases with a sampled row; -Dconc2-safe too)
    const user_sampling = module(b, "zig/tests/cuda/user_sampling_users_test.zig", host, opt, focused_mods);
    serveImports(b, user_sampling, host, opt, focused_mods);
    grammar_tests_imports(b, user_sampling, host, opt, focused_mods);
    user_sampling.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    user_sampling.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const user_sampling_run = b.addRunArtifact(b.addTest(.{ .root_module = user_sampling, .filters = &.{"user_sampling"} }));
    user_sampling_run.has_side_effects = true;
    b.step("test-user_sampling", "user_sampling sampled users backend: the switch's claim, each row's keyed draw at its own positions, cancel and failure releases with a sampled row").dependOn(&user_sampling_run.step);
    step.dependOn(&user_sampling_run.step);
    // four_user_profile: the USERS=4 join's combined CPU tests (shared_pool's pool + prefill_interleave's interleave + a structured_users schema request at
    // TF_AFD_USERS=4 through the real LaneHost; -Dconc2-safe too); only its own tests (the imported files' run above)
    const four_user_profile = module(b, "zig/tests/cuda/four_user_profile_join_test.zig", host, opt, focused_mods);
    serveImports(b, four_user_profile, host, opt, focused_mods);
    grammar_tests_imports(b, four_user_profile, host, opt, focused_mods);
    four_user_profile.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    four_user_profile.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const four_user_profile_run = b.addRunArtifact(b.addTest(.{ .root_module = four_user_profile, .filters = &.{"four_user_profile"} }));
    four_user_profile_run.has_side_effects = true;
    b.step("test-four_user_profile", "four_user_profile USERS=4 join: pool + interleave + a schema request together, solo = together, interleaved = serial").dependOn(&four_user_profile_run.step);
    step.dependOn(&four_user_profile_run.step);
    // structured_entry: the served entries end to end on the recorder (afd_attn_serve.run, which hands TF_AFD_USERS>1 to
    // afd_attn_users.run) with TF_AFD_GRAMMAR_SERVE=1 over the real GLM tokenizer directory and grammar_abi's library
    // (-Dgram2abi, TF_grammar_server_LIB/_LIB_SHA256/_MODEL as test-grammar_server_abi); one process per case (serve.run's stop is
    // process-wide); outside `zig build test` (it needs the library); -Dconc2-safe too
    const g3d_step = b.step("test-structured_entry_served", "structured_entry served entries: USERS=2 and USERS=4 (pool + interleave + sampling) structured output over the real tokenizer, plain = solo; SP sampled structured (-Dgram2abi, TF_grammar_server_*)");
    if (given(b, "grammar_abi") == null) {
        g3d_step.dependOn(&b.addFail("test-structured_entry_served needs -Dgram2abi=<root.zig re-exporting grammar_abi's tfg_trusted.zig and tfg_owned.zig>").step);
    } else {
        const g3d = module(b, "zig/tests/cuda/structured_entry_served_test.zig", host, opt, focused_mods);
        serveImports(b, g3d, host, opt, focused_mods);
        grammar_tests_imports(b, g3d, host, opt, focused_mods);
        g3d.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
        g3d.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
        const g3d_exe = b.addTest(.{ .root_module = g3d, .filters = &.{"structured_entry"} });
        for ([_][]const u8{ "u2", "u4", "sp" }) |case| {
            const run = b.addRunArtifact(g3d_exe);
            run.setEnvironmentVariable("structured_entry_CASE", case);
            run.has_side_effects = true;
            g3d_step.dependOn(&run.step);
        }
    }
    // user_sampling: TF_AFD_USERS_SAMPLING's served entries, structured_entry's form (-Dgram2abi, TF_grammar_server_*): one process per case
    // (s: the single-user boot S; u2, u4: the switch on; u2off, u4off: off; fail: a device fault mid-round; refuse: the
    // startup refusals); each case's replies to $user_sampling_OUT/<case>.json when set (scripts/check_served.py)
    const user_sampling_step = b.step("test-user_sampling_served", "user_sampling served entries: sampled requests at USERS=2 and USERS=4 (pool + interleave) with TF_AFD_USERS_SAMPLING=1, seeded identity alone = together = single-user S, greedy unchanged on/off, structured + sampled, sampling_tests_stat gates per row, cut and fault mid-round, startup refusals (-Dgram2abi, TF_grammar_server_*)");
    if (given(b, "grammar_abi") == null) {
        user_sampling_step.dependOn(&b.addFail("test-user_sampling_served needs -Dgram2abi=<root.zig re-exporting grammar_abi's tfg_trusted.zig and tfg_owned.zig>").step);
    } else {
        const ms = module(b, "zig/tests/cuda/user_sampling_served_test.zig", host, opt, focused_mods);
        serveImports(b, ms, host, opt, focused_mods);
        grammar_tests_imports(b, ms, host, opt, focused_mods);
        ms.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
        ms.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
        const ms_exe = b.addTest(.{ .root_module = ms, .filters = &.{"user_sampling"} });
        for ([_][]const u8{ "refuse", "s", "u2", "u4", "u2off", "u4off", "fail" }) |case| {
            const run = b.addRunArtifact(ms_exe);
            run.setEnvironmentVariable("user_sampling_CASE", case);
            run.has_side_effects = true;
            user_sampling_step.dependOn(&run.step);
        }
    }
    // user_drafts: MTP drafts with TF_AFD_USERS >= 2 (TF_AFD_USERS_DRAFTS=1, mode A): the lone driver through the real LaneHost,
    // lane core and users backend, and the family's carried MTP state recorded (-Dconc2-safe too); only its own tests
    const user_drafts = module(b, "zig/tests/cuda/user_drafts_test.zig", host, opt, focused_mods);
    serveImports(b, user_drafts, host, opt, focused_mods);
    grammar_tests_imports(b, user_drafts, host, opt, focused_mods);
    user_drafts.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    user_drafts.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const user_drafts_run = b.addRunArtifact(b.addTest(.{ .root_module = user_drafts, .filters = &.{"user_drafts"} }));
    user_drafts_run.has_side_effects = true;
    b.step("test-user_drafts", "user_drafts MTP drafts with users: lone drafted = undrafted, lone -> shared -> lone, cancel/failure, USERS=4 pool + interleave").dependOn(&user_drafts_run.step);
    step.dependOn(&user_drafts_run.step);
    // joined_profile: the final head's combined identity at TF_AFD_USERS=4 (pool + interleave + structured + user_sampling's sampled
    // users + user_drafts's lone drafts, both switches on) through the real LaneHost, lane core and users backend
    const joined_profile = module(b, "zig/tests/cuda/joined_profile_test.zig", host, opt, focused_mods);
    serveImports(b, joined_profile, host, opt, focused_mods);
    grammar_tests_imports(b, joined_profile, host, opt, focused_mods);
    joined_profile.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    joined_profile.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const joined_profile_run = b.addRunArtifact(b.addTest(.{ .root_module = joined_profile, .filters = &.{"joined_profile"} }));
    joined_profile_run.has_side_effects = true;
    b.step("test-joined_profile", "joined_profile combined identity at USERS=4: drafted = undrafted, solo = together, interleaved = serial, structured valid, seeded sampled alone = beside = switched = boot S, each switch's own default-off").dependOn(&joined_profile_run.step);
    step.dependOn(&joined_profile_run.step);
    // shared_forward: one forward over every user's rows (TF_AFD_USERS_SHARED=1, stage 1, undrafted): the switch, the shared
    // round recorded through the real users backend against each stream alone, against today's per-row path, the real
    // scheduler over a stand-in, cancel/failure, bytes, default off; and the family module's own shared_forward tests (the
    // switch's parse, the row view); -Dconc2-safe too
    const shared_forward = module(b, "zig/tests/cuda/shared_forward_test.zig", host, opt, focused_mods);
    serveImports(b, shared_forward, host, opt, focused_mods);
    grammar_tests_imports(b, shared_forward, host, opt, focused_mods);
    shared_forward.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    shared_forward.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const shared_forward_run = b.addRunArtifact(b.addTest(.{ .root_module = shared_forward, .filters = &.{"shared_forward"} }));
    shared_forward_run.has_side_effects = true;
    const shared_forward_family = b.addRunArtifact(b.addTest(.{ .root_module = focused_mods.glm, .filters = &.{"shared_forward"} }));
    shared_forward_family.has_side_effects = true;
    const shared_forward_step = b.step("test-shared_forward", "shared_forward one forward over every user's rows: each stream exact against alone (recorded), State launches = per-row path's, scheduler stand-in, cancel/failure, bytes, default off");
    shared_forward_step.dependOn(&shared_forward_run.step);
    shared_forward_step.dependOn(&shared_forward_family.step);
    step.dependOn(&shared_forward_run.step);
    // draft_shared: user_drafts's lone drafts and shared_forward's shared forward together (both switches on): the switches through
    // plan(), Users with both, each tracked slot's backlog through shared rounds (recorded, directly), cancel/failure,
    // lone <-> shared mid-reply through the real LaneHost over the real scheduler's forwards, each switch alone
    // (draft_shared's step: Claude Opus 5.5 (Anthropic; planning and execution), via Hermes)
    const draft_shared = module(b, "zig/tests/cuda/draft_shared_test.zig", host, opt, focused_mods);
    serveImports(b, draft_shared, host, opt, focused_mods);
    grammar_tests_imports(b, draft_shared, host, opt, focused_mods);
    draft_shared.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    draft_shared.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const draft_shared_run = b.addRunArtifact(b.addTest(.{ .root_module = draft_shared, .filters = &.{"draft_shared"} }));
    draft_shared_run.has_side_effects = true;
    b.step("test-draft_shared", "draft_shared lone drafts + shared forward: plan, Users with both, backlog through shared rounds = per-row = solo, cancel/failure, lone <-> shared mid-reply = solo undrafted, each switch alone").dependOn(&draft_shared_run.step);
    step.dependOn(&draft_shared_run.step);
    // shared_verify (TF_AFD_USERS_SHARED_DRAFTS=1): the drafted shared verify through the real LaneHost, lane core and users
    // backend over a stand-in (solo = together = switch off, transitions, cancel/failure, sampled), each drafted window
    // recorded through the real Users against its stream alone, the switch through plan(), default off; and the family
    // module's own shared_verify tests (the switch's parse, window rows, the window view)
    const shared_verify = module(b, "zig/tests/cuda/shared_verify_test.zig", host, opt, focused_mods);
    serveImports(b, shared_verify, host, opt, focused_mods);
    grammar_tests_imports(b, shared_verify, host, opt, focused_mods);
    shared_verify.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    shared_verify.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const shared_verify_run = b.addRunArtifact(b.addTest(.{ .root_module = shared_verify, .filters = &.{"shared_verify"} }));
    shared_verify_run.has_side_effects = true;
    const shared_verify_family = b.addRunArtifact(b.addTest(.{ .root_module = focused_mods.glm, .filters = &.{"shared_verify"} }));
    shared_verify_family.has_side_effects = true;
    const shared_verify_step = b.step("test-shared_verify", "shared_verify drafted shared verify: each reply = solo = switch off (stand-in), each drafted window exact against alone (recorded), transitions, cancel/failure, sampled, switch, default off");
    shared_verify_step.dependOn(&shared_verify_run.step);
    shared_verify_step.dependOn(&shared_verify_family.step);
    step.dependOn(&shared_verify_run.step);
    const ping_pong = module(b, "zig/tests/cuda/ping_pong_test.zig", host, opt, focused_mods);
    serveImports(b, ping_pong, host, opt, focused_mods);
    grammar_tests_imports(b, ping_pong, host, opt, focused_mods);
    ping_pong.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    ping_pong.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const ping_pong_run = b.addRunArtifact(b.addTest(.{ .root_module = ping_pong, .filters = &.{"ping_pong"} }));
    ping_pong_run.has_side_effects = true;
    b.step("test-ping_pong", "ping_pong independent user micro-batches: caps, FIFO overlap, identity, cancel/failure, default off").dependOn(&ping_pong_run.step);
    step.dependOn(&ping_pong_run.step);
    const batched_drafts = module(b, "zig/tests/cuda/batched_drafts_test.zig", host, opt, focused_mods);
    serveImports(b, batched_drafts, host, opt, focused_mods);
    grammar_tests_imports(b, batched_drafts, host, opt, focused_mods);
    batched_drafts.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    batched_drafts.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null))));
    const batched_drafts_run = b.addRunArtifact(b.addTest(.{ .root_module = batched_drafts, .filters = &.{"batched_drafts"} }));
    batched_drafts_run.has_side_effects = true;
    b.step("test-batched_drafts", "batched_drafts batched MTP draft depths, per-stream state and output identity").dependOn(&batched_drafts_run.step);
    step.dependOn(&batched_drafts_run.step);
    const tool_tests = module(b, "zig/tests/cuda/tf_glm_afd.zig", host, .debug, mods);
    tool_tests.addAnonymousImport("afd_attn_fixture", .{ .root_source_file = b.path("zig/src/families/glm/fixtures_cuda_afd_attn.json") });
    tool_tests.addOptions("glm_afd_options", imageOptions(b, &@as([images.len]?std.Build.LazyPath, @splat(null)))); // a host build: none
    serveImports(b, tool_tests, host, .debug, mods); // qualification_4be15bbf: attn serve
    grammar_tests_imports(b, tool_tests, host, .debug, mods); // grammar_tests's AFD tests, reached through afd_attn_serve.zig
    for ([_]*std.Build.Module{ cover, cover_f8, cover_mtp8, tool_tests }) |m| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
        step.dependOn(&run.step);
        own.dependOn(&run.step);
    }
    // grammar_server's CPU roots as grammar_tests_tests.zig builds them (test-grammar_server, test-grammar_server_wire: same filters, the stub grammar_abi) also join `zig build test` and test-glm-afd; build wiring only
    const noabi = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/afd_attn_noabi.zig"), .target = host, .optimize = .debug, .link_libc = true });
    for ([_][2][]const u8{ .{ "zig/tests/cuda/grammar_server_test.zig", "grammar_server " }, .{ "zig/tests/cuda/grammar_server_wire.zig", "grammar_server-WIRE" } }) |k| {
        const g2 = module(b, k[0], host, .debug, mods);
        serveImports(b, g2, host, .debug, mods);
        grammar_tests_imports(b, g2, host, .debug, mods);
        g2.addImport("grammar_abi", noabi);
        const run = b.addRunArtifact(b.addTest(.{ .root_module = g2, .filters = &.{k[1]} }));
        run.has_side_effects = true;
        step.dependOn(&run.step);
        own.dependOn(&run.step);
    }
}

/// An image the attention role launches beyond the runtime's (qualification_cd073cc0): its name, its source under zig/kernels/cuda and
/// zig/build/glm_fwd_tool.zig's flags for it (tf-glm-forward embeds the same), and which of fwd.Kernels needs it.
const Image = struct { name: []const u8, src: []const u8, flags: []const []const u8, part: Part = .run };
/// grammar_mask: `grammar` gates only the masked adapter (TF_AFD_GRAMMAR_MASK=1), never the GPU verbs.
const Part = enum { pack, run, mm, dsa, nucleus, grammar };

const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

/// glm_pack (the load's pack pass), KDA and the sampler's torch ops, lane A's affine matmuls (fwd.Kernels.mm; the
/// attention's EXL3 dense weights are BF16, so its matmuls are Triton's) and DSA select. No EXL3 image: the routed
/// experts are the expert nodes', and the runtime embeds qualification_765330ce's six for them.
const images = [_]Image{
    .{ .name = "glm_pack", .src = "glm_pack", .flags = &.{"-O3"}, .part = .pack },
    .{ .name = "glm_kda", .src = "glm_kda", .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "glm_qmm_group", .src = "glm_qmm_group", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_qmm_prefill", .src = "glm_qmm_prefill", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_experts", .src = "glm_experts", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_experts_prefill", .src = "glm_experts_prefill", .flags = &.{"-O3"}, .part = .mm },
    .{ .name = "glm_select", .src = "glm_select", .flags = &.{"-O3"}, .part = .dsa },
    .{ .name = "torch_nucleus", .src = "torch_ops/nucleus", .flags = torch_ops, .part = .nucleus },
    .{ .name = "torch_argmax_f64", .src = "torch_ops/argmax_f64", .flags = torch_ops, .part = .nucleus },
    // grammar_mask: Ash Hart's dormant torch_ops/grammar_mask.cu (cd7ebe98, unchanged), the masked greedy draw's mask
    .{ .name = "torch_grammar_mask", .src = "torch_ops/grammar_mask", .flags = torch_ops, .part = .grammar },
};

/// torch.utils.cpp_extension's nvcc flags, as zig/build/cuda.zig passes them to every fatbin.
const torch_flags = [_][]const u8{ "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-std=c++20" };

/// A -D option zig/build/cuda.zig declares (nvcc, fatbins, sm), as the user gave it.
fn given(b: *std.Build, name: []const u8) ?[]const u8 {
    const v = b.user_input_options.get(name) orelse return null;
    return switch (v) {
        .scalar => |s| s,
        else => null,
    };
}

/// nvcc -fatbin of an image with torch's flags, its own, and one -gencode per SASS target (glm_fwd_tool.zig's rule).
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, im: Image, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(im.flags);
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ sm, sm }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{im.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{im.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{im.src})));
    return out;
}

/// The tool's `glm_afd_options`: which images it embeds (afd_attn_gpu.zig refuses the GPU verbs without them).
fn imageOptions(b: *std.Build, paths: []const ?std.Build.LazyPath) *std.Build.Step.Options {
    var has = [_]bool{ true, true, true, true, true, true }; // Part's
    for (images, paths) |im, p| has[@backingInt(im.part)] = has[@backingInt(im.part)] and p != null;
    const o = b.addOptions();
    o.addOption(bool, "glm_pack", has[@backingInt(Part.pack)]);
    o.addOption(bool, "run_images", has[@backingInt(Part.run)]);
    o.addOption(bool, "mm_images", has[@backingInt(Part.mm)]);
    o.addOption(bool, "dsa_image", has[@backingInt(Part.dsa)]);
    o.addOption(bool, "nucleus_images", has[@backingInt(Part.nucleus)]);
    o.addOption(bool, "grammar_image", has[@backingInt(Part.grammar)]);
    return o;
}

/// `zig build tf-glm-afd`: the tool for -Dtarget (not in `install`) over `mods` (zig/build/cuda.zig's runtime: its images
/// with -Dnvcc or -Dfatbins, none without), with the attention role's images above under the same options (prebuilt in
/// -Dfatbins' directory when there, else nvcc's), installed beside it in attn-fatbin/ (cuda.zig's `fatbins` step builds
/// its own glm_kda and torch ops into fatbin/, from their own nvcc runs).
pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, mods: Mods) void {
    const nvcc = given(b, "nvcc");
    const version: ?std.Build.LazyPath = if (nvcc) |n| blk: {
        const v = b.addSystemCommand(&.{ n, "--version" });
        v.has_side_effects = true;
        break :blk v.captureStdOut(.{});
    } else null;
    var paths: [images.len]?std.Build.LazyPath = @splat(null);
    for (images, &paths) |im, *p| {
        if (given(b, "fatbins")) |dir| {
            const f = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{im.name}) });
            if (std.Io.Dir.accessAbsolute(b.graph.io, f, .{})) |_| {
                p.* = .{ .cwd_relative = f };
            } else |_| {}
        }
        if (p.* == null) if (nvcc) |n| {
            p.* = fatbin(b, n, version.?, im, given(b, "sm") orelse "121");
        };
    }
    const root = module(b, "zig/tests/cuda/tf_glm_afd.zig", target, optimize, mods);
    root.addOptions("glm_afd_options", imageOptions(b, &paths));
    serveImports(b, root, target, optimize, mods); // qualification_4be15bbf: attn serve
    for (images, paths) |im, p| if (p) |path| root.addAnonymousImport(b.fmt("{s}_fatbin", .{im.name}), .{ .root_source_file = path });
    const exe = b.addExecutable(.{ .name = "tf-glm-afd", .root_module = root });
    const step = b.step("tf-glm-afd", "GLM-5.3-Flash's AFD roles over native RoCE exchange: expert (a half of the routed experts) and attn (world 1)");
    step.dependOn(&b.addInstallArtifact(exe, .{}).step);
    for (images, paths) |im, p| if (p) |path| step.dependOn(&b.addInstallFile(path, b.fmt("attn-fatbin/{s}.fatbin", .{im.name})).step);
}
