//! The CUDA GLM's serve open (zig/src/native/glm_cuda*.zig, qualification_qualification_44ba554e.8 day 2) loads a rank with tf-glm-forward's own loader (its
//! split plans and Run: each stored group read, packed and bound, as qualification_0ad3c879 and qualification_f686be52 ran it), so the native engines import
//! the tool's root as `glm_tool`. It embeds no images there: the open reads qualification_e84ee1bf's pinned fatbins at run time
//! (TF_GLM_FATBINS). The glm_pack emulator is linked because the tool's Run names it for its host mode.
const std = @import("std");

pub fn tool(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, glm: *std.Build.Module, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    const options = b.addOptions();
    inline for (.{ "glm_pack", "run_images", "mm_images", "dsa_image", "exl3_images" }) |name| options.addOption(bool, name, false);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_forward.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{ .{ .name = "glm_cuda", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } },
    });
    m.addOptions("glm_forward_options", options);
    m.addCSourceFile(.{ .file = b.path("zig/tests/cuda/glm_pack_emu.cpp"), .flags = &.{ "-std=c++20", "-O2", "-fno-exceptions" } });
    return m;
}
