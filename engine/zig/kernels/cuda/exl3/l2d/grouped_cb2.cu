// The kernels src/tensorfold/cuda/exl3/experts_cb2.cu with exl3_l2d.patch compiles (codebook 2, mul1): grouped_launch<2>'s tile settings x K2 ranges and dequant_launch<2>'s widths, written by zig/tests/cuda/exl3_copies.py.
#include "experts_grouped.cuh"

#define TF_GROUPED(NT, W, PF, LO, HI) template __global__ void tf_exl3x::grouped_kernel<2, NT, W, PF, LO, HI>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int, int);
TF_GROUPED(8, 4, 1, 8, 8) TF_GROUPED(8, 4, 1, 2, 10) TF_GROUPED(8, 4, 1, 2, 16)
TF_GROUPED(8, 4, 2, 8, 8) TF_GROUPED(8, 4, 2, 2, 10) TF_GROUPED(8, 4, 2, 2, 16)
TF_GROUPED(4, 4, 1, 8, 8) TF_GROUPED(4, 4, 1, 2, 10) TF_GROUPED(4, 4, 1, 2, 16)
TF_GROUPED(4, 4, 2, 8, 8) TF_GROUPED(4, 4, 2, 2, 10) TF_GROUPED(4, 4, 2, 2, 16)
#define TF_DEQUANT(K2) template __global__ void tf_exl3x::dequant_kernel<2, K2>(const uint32_t*, half*, int, int);
TF_DEQUANT(2) TF_DEQUANT(3) TF_DEQUANT(4) TF_DEQUANT(5) TF_DEQUANT(6) TF_DEQUANT(7) TF_DEQUANT(8) TF_DEQUANT(9) TF_DEQUANT(10) TF_DEQUANT(11) TF_DEQUANT(12) TF_DEQUANT(13) TF_DEQUANT(14) TF_DEQUANT(15) TF_DEQUANT(16)
