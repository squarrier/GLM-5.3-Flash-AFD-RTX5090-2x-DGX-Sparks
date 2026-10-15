//! GLM-5.3-Flash's CUDA loader checked against the Metal path's own loader, on a CPU without Metal or a GPU.
test {
    _ = @import("tests/cuda/glm_split_check.zig");
}
