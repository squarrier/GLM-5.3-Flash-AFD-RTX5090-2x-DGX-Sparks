// CUDA's built-ins for fn_xchg.cu on the CPU (xchg_emu.cpp): thread and block indices per host thread, __syncthreads as the block's barrier, device atomics as the compiler's.
#pragma once
#include <stdint.h>
#include <string.h>
#include <time.h>

#define __global__
#define __device__
#define __forceinline__ inline
#define __launch_bounds__(n)

struct tf_emu_dim3 {
  unsigned x, y, z;
};
struct alignas(16) uint4 {
  unsigned x, y, z, w;
};

extern thread_local tf_emu_dim3 threadIdx, blockIdx, blockDim, gridDim;

void tf_emu_syncthreads();
#define __syncthreads() tf_emu_syncthreads()

inline unsigned long long tf_emu_now_ns() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
}
inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
inline unsigned atomicExch(unsigned* p, unsigned v) { return __atomic_exchange_n(p, v, __ATOMIC_SEQ_CST); }
inline void __threadfence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fmul_rn(float a, float b) { return a * b; }
