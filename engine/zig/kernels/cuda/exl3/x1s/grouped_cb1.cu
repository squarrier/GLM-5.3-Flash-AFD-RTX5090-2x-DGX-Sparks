// SEQ's instances (qualification_b3fba1da): X1's prompt_grouped_launch<1> tile (1, 8, 4, 1, 8, 8) at each switch selector (NFIRST 8, ORDER 4, PA 2, XROW 1), for gate/up's K splits.
#include "experts_seq.cuh"

#define TF_SEQ(N, O, P, X) template __global__ void tf_exl3x::grouped_seq_kernel<1, 8, 4, 1, 8, 8, N, O, P, X>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int, const int*);
TF_SEQ(false, false, false, true)
TF_SEQ(false, false, true, false)
TF_SEQ(false, false, true, true)
TF_SEQ(false, true, false, false)
TF_SEQ(false, true, false, true)
TF_SEQ(false, true, true, false)
TF_SEQ(false, true, true, true)
TF_SEQ(true, false, false, false)
TF_SEQ(true, false, false, true)
TF_SEQ(true, false, true, false)
TF_SEQ(true, false, true, true)
TF_SEQ(true, true, false, false)
TF_SEQ(true, true, false, true)
TF_SEQ(true, true, true, false)
TF_SEQ(true, true, true, true)
