// The kernels src/tensorfold/cuda/exl3/experts_cb1.cu with exl3_x1.patch compiles for the prompt switches (codebook 1, mcg): prompt_grouped_launch<1>'s instances, NFIRST, ORDER, PA and XROW by its selector, written by zig/tests/cuda/exl3_copies.py.
#include "experts_grouped.cuh"

#define TF_PROMPT(N, O, P, X) template __global__ void tf_exl3x::grouped_kernel<1, 8, 4, 1, 8, 8, N, O, P, X>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int, const int*);
TF_PROMPT(false, false, false, true)
TF_PROMPT(false, false, true, false)
TF_PROMPT(false, false, true, true)
TF_PROMPT(false, true, false, false)
TF_PROMPT(false, true, false, true)
TF_PROMPT(false, true, true, false)
TF_PROMPT(false, true, true, true)
TF_PROMPT(true, false, false, false)
TF_PROMPT(true, false, false, true)
TF_PROMPT(true, false, true, false)
TF_PROMPT(true, false, true, true)
TF_PROMPT(true, true, false, false)
TF_PROMPT(true, true, false, true)
TF_PROMPT(true, true, true, false)
TF_PROMPT(true, true, true, true)
