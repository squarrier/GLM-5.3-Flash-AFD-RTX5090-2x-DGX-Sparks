// glm_select.cu on a CPU (qualification_qualification_973313f6), for tests without a GPU (zig/glm_select_host.zig): each launch's CUDA threads run one
// after another, a call each (glm_pack_emu.cpp's way), which is exact for a kernel whose threads share nothing.
#include <stddef.h>
#include <stdint.h>

namespace {
struct Dim {
    unsigned x, y, z;
};
Dim blockIdx, threadIdx, blockDim, gridDim;
}  // namespace

#define __global__
#include "../../kernels/cuda/glm_select.cu"

// tf_glm_select_tail over `grid` blocks of `block` threads (x fastest), as cuLaunchKernel would run it
extern "C" void tf_glm_select_emu(const unsigned* grid, const unsigned* block, const int64_t* pools, const int32_t* pos,
                                  int32_t* tokens, int32_t* counts, uint32_t rows) {
    gridDim = {grid[0], grid[1], grid[2]};
    blockDim = {block[0], block[1], block[2]};
    for (unsigned z = 0; z < grid[2]; z++)
        for (unsigned y = 0; y < grid[1]; y++)
            for (unsigned x = 0; x < grid[0]; x++)
                for (unsigned t = 0; t < block[0] * block[1] * block[2]; t++) {
                    blockIdx = {x, y, z};
                    threadIdx = {t % block[0], (t / block[0]) % block[1], t / (block[0] * block[1])};
                    tf_glm_select_tail(pools, pos, tokens, counts, rows);
                }
}

// tf_glm_pos_add over `grid` blocks of `block` threads, as cuLaunchKernel would run it (qualification_qualification_ea348a94)
extern "C" void tf_glm_pos_add_emu(const unsigned* grid, const unsigned* block, const int32_t* pos, int32_t n,
                                   int32_t* out) {
    gridDim = {grid[0], grid[1], grid[2]};
    blockDim = {block[0], block[1], block[2]};
    for (unsigned x = 0; x < grid[0]; x++)
        for (unsigned t = 0; t < block[0]; t++) {
            blockIdx = {x, 0, 0};
            threadIdx = {t, 0, 0};
            tf_glm_pos_add(pos, n, out);
        }
}
