//! GLM-5.3-Flash's Triton wrappers checked on a CPU without a GPU: their launches against the Python wrappers' own
//! (fixtures_cuda_triton.json), and those launches against the kernel set's spec (tests/cuda/glm/kernels.json).
test {
    _ = @import("src/families/glm/cuda_triton_test.zig");
    _ = @import("tests/cuda/glm/triton_set.zig");
    _ = @import("tests/cuda/glm/triton_f8.zig"); // qualification_qualification_ebed70b7: patch 0002's _fmm (the FP8 head)
}
