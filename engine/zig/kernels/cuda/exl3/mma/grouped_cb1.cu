// qualification_8724e161's instances (qualification_7af419c1's qualification_babb0a2d, bertholomus's 90b68e0): grouped_mma_kernel<CB 1 (mcg), K2 8..8, FOLD, KB 4, XROW> as the
// halves' prompt windows take it (gate/up FOLD, XROW with X1's rowrot; down FOLD only with K splits), and work_list_kernel.
#include "experts_mma.cuh"

#define TF_MMA(F, X) template __global__ void tf_exl3x::grouped_mma_kernel<1, 8, 8, F, 4, X>( \
    const half*, const half*, const int64_t*, const int64_t*, const int*, const int*, const int*, const int*, \
    const int*, float*, int, int, int, int, int, int, int, int, const int*);
TF_MMA(true, false)
TF_MMA(true, true)
TF_MMA(false, false)
