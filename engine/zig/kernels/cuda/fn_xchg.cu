// The GPU half of the three-node exchange (qualification_12f4270a): the attention rank posts a request to two expert ranks and takes both replies, each pair over its own one-shot RoCE proxy (roce_proxy.c; qualification_9e49592e: its slots a source counted at open); the host half is zig/src/cuda/xchg.
// Design: Mia's fn_roce.cu (MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, patch 0002, Apache-2.0, Copyright 2026 MiaAI-Lab) and b12x's RoCEnante protocol (local-inference-lab/b12x, Apache-2.0), written anew for a pingpong.
// Changed from fn_roce: no all-gather and no __trap; a wait is bounded, records its give-up in the pinned status, every wait after a failure returns at once, and nothing is sent after one (Ash's #214: a dead expert fails the request, not the attention process).
// Region offsets are tf_roce_layout's for world 2: recv[src][slot], flag[src][slot][port] (128 B), send[slot], ctrl (doorbell, pad, error seq, error peer, bytes[slots]); qualification_9e49592e: the slots a source come with each launch (2, or TF_AFD_LANES=N's N).
// Sequences live on the device (State), so a CUDA graph can replay the launches; host builds (zig/tests/cuda/xchg/xchg_emu.cpp) run the same code on CPU threads.
#include <stdint.h>

namespace tf_xchg {

constexpr unsigned PORTS_MAX = 2, FLAG_WORDS = 32;  // roce_proxy.c's ports and 128-byte flags
// The status words (pinned, host-visible): failed (the first failure's reason; nonzero means the exchange is dead), give-ups, breaches, the first failure's seq and source
constexpr unsigned FAILED = 0, GAVE_UP = 1, BREACH = 2, FAIL_SEQ = 3, FAIL_SRC = 4;
constexpr unsigned WHY_GAVE_UP = 1, WHY_BREACH = 2;  // the host's reasons start at 16 (xchg/proto.zig)

struct State {
  unsigned post_seq, take_seq, stage, tail;
};

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_relaxed_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_sys16(const uint4* p) {
  uint4 r;
  // no "memory" clobber: a copy loop's loads may be in flight together (the flag's acquire and the block barrier order them after the peer's write)
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void fence_sys() { __threadfence_system(); }
__device__ __forceinline__ void add_sys(unsigned* p, unsigned v) { atomicAdd_system(p, v); }
__device__ __forceinline__ unsigned cas_sys(unsigned* p, unsigned c, unsigned v) { return atomicCAS_system(p, c, v); }
#else
// host builds: the same steps with the compiler's atomics on host memory
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
__device__ __forceinline__ unsigned ld_relaxed_sys(const unsigned* p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }
__device__ __forceinline__ void st_relaxed_sys(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELAXED); }
__device__ __forceinline__ uint4 ld_sys16(const uint4* p) { return *p; }
__device__ __forceinline__ unsigned long long now_ns() { return tf_emu_now_ns(); }
__device__ __forceinline__ void fence_sys() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
__device__ __forceinline__ void add_sys(unsigned* p, unsigned v) { __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST); }
__device__ __forceinline__ unsigned cas_sys(unsigned* p, unsigned c, unsigned v) {
  __atomic_compare_exchange_n(p, &c, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return c;
}
#endif

// The first failure wins the reason, its seq and its source; every failure counts in its counter
__device__ __forceinline__ void fail(unsigned* status, unsigned why, unsigned counter, unsigned seq, unsigned src) {
  add_sys(status + counter, 1u);
  if (cas_sys(status + FAILED, 0u, why) == 0u) {
    st_relaxed_sys(status + FAIL_SEQ, seq);
    st_relaxed_sys(status + FAIL_SRC, src);
  }
  fence_sys();
}

// The flag reaching `seq`: false at once after any failure, on a flag past `seq` (the peer is ahead: a breach), when the host fails the exchange, or past `give_up_ns` (counted)
__device__ __forceinline__ bool wait_flag(const unsigned* f, unsigned seq, unsigned* status, unsigned src, unsigned long long give_up_ns) {
  if (ld_relaxed_sys(status + FAILED) != 0u) return false;  // the exchange is already dead: never spin again
  const unsigned long long t0 = now_ns();
  for (unsigned polls = 1;; ++polls) {
    const unsigned v = ld_acquire_sys(f);
    if (v == seq) return true;
    if ((int)(v - seq) > 0) {
      fail(status, WHY_BREACH, BREACH, seq, src);
      return false;
    }
    if ((polls & 63u) == 0u) {
      if (ld_relaxed_sys(status + FAILED) != 0u) return false;
      if (now_ns() - t0 > give_up_ns) {
        fail(status, WHY_GAVE_UP, GAVE_UP, seq, src);
        return false;
      }
    }
  }
}

// The last block of a launch to arrive at `counter` (each block's thread 0 once): it alone runs the launch's tail
__device__ __forceinline__ bool last_block(unsigned* counter, int blocks) {
  if (atomicAdd(counter, 1u) != (unsigned)blocks - 1u) return false;
  atomicExch(counter, 0u);
  return true;
}

// qualification_17c70e37 (TF_AFD_STRIPES): a message of `k` stripes of at least `min` bytes goes in k stripes a port (else one); stripe j's flag word (the last stripe's is word 0)
__device__ __forceinline__ unsigned stripes_of(unsigned k, unsigned long long min, unsigned bytes) {
  return k >= 2u && (unsigned long long)bytes >= (unsigned long long)k * min ? k : 1u;
}
__device__ __forceinline__ unsigned stripe_word(unsigned n, unsigned j) { return j + 1u == n ? 0u : 1u + j; }

#if defined(__CUDA_ARCH__)
__device__ __forceinline__ void nap(unsigned ns) { __nanosleep(ns); }
#else
__device__ __forceinline__ void nap(unsigned) {}
#endif

// wait_flag for a stripe's word, napping between polls (up to 2 us) so that many waiting blocks poll host memory gently
__device__ __forceinline__ bool wait_word(const unsigned* f, unsigned seq, unsigned* status, unsigned src, unsigned long long give_up_ns) {
  if (ld_relaxed_sys(status + FAILED) != 0u) return false;
  const unsigned long long t0 = now_ns();
  unsigned sleep = 64u;
  for (unsigned polls = 1;; ++polls) {
    const unsigned v = ld_acquire_sys(f);
    if (v == seq) return true;
    if ((int)(v - seq) > 0) {
      fail(status, WHY_BREACH, BREACH, seq, src);
      return false;
    }
    if ((polls & 15u) == 0u) {
      if (ld_relaxed_sys(status + FAILED) != 0u) return false;
      if (now_ns() - t0 > give_up_ns) {
        fail(status, WHY_GAVE_UP, GAVE_UP, seq, src);
        return false;
      }
    }
    nap(sleep);
    if (sleep < 2048u) sleep <<= 1;
  }
}

// units [lo, hi) of `in` into `out` by the block's threads, four loads in flight a thread (tf_xchg_take's copy)
__device__ __forceinline__ void copy_units(const uint4* in, uint4* out, unsigned long long lo, unsigned long long hi) {
  const unsigned long long stride = blockDim.x;
  for (unsigned long long i0 = lo + threadIdx.x; i0 < hi; i0 += 4u * stride) {
    uint4 v[4];
#pragma unroll
    for (unsigned u = 0; u < 4; ++u)
      if (i0 + u * stride < hi) v[u] = ld_sys16(in + i0 + u * stride);
#pragma unroll
    for (unsigned u = 0; u < 4; ++u)
      if (i0 + u * stride < hi) out[i0 + u * stride] = v[u];
  }
}

}  // namespace tf_xchg

using namespace tf_xchg;

// One message out: `in` (`bytes`, a positive multiple of 16, both 16-aligned) into send[seq % slots], a system fence, then the last block rings the proxies' doorbell (bytes, then seq)
// After a failure nothing is staged or rung (the seq still advances, as take's does): a live peer gets no message past the failing one, and no slot is rewritten under a proxy's write
extern "C" __global__ void __launch_bounds__(512) tf_xchg_post(const uint4* in, unsigned bytes, char* region, unsigned long long send_off, unsigned long long ctrl_off, unsigned long long slot_bytes, State* st, const unsigned* status, int blocks, unsigned slots) {
  const unsigned seq = *reinterpret_cast<volatile unsigned*>(&st->post_seq) + 1u;
  const unsigned slot = seq % slots;
  uint4* send = reinterpret_cast<uint4*>(region + send_off + slot * slot_bytes);
  const unsigned n = bytes / 16u;
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x, stride = (unsigned)blocks * blockDim.x;
  if (ld_relaxed_sys(status + FAILED) == 0u)
    for (unsigned i = index; i < n; i += stride) send[i] = in[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_sys();
    // every block read the old seq before it counted, so the last one may publish the new one
    if (last_block(&st->stage, blocks)) {
      fence_sys();  // a block that saw the failure skipped its stripe; ordered after its count, this read sees the failure too
      if (ld_relaxed_sys(status + FAILED) == 0u) {
        unsigned* ctrl = reinterpret_cast<unsigned*>(region + ctrl_off);
        st_relaxed_sys(ctrl + 4 + slot, bytes);
        fence_sys();
        st_relaxed_sys(ctrl, seq);
      }
      *reinterpret_cast<volatile unsigned*>(&st->post_seq) = seq;
      __threadfence();
    }
  }
}

// One message from each of `nsrc` (1 or 2) sources: every port's flag[src][seq % slots] reaching seq (bounded), then recv[src][seq % slots] into out_k (bytes_k, multiples of 16); after any failure nothing is copied
extern "C" __global__ void __launch_bounds__(512) tf_xchg_take(int nsrc, unsigned src0, unsigned src1, uint4* out0, uint4* out1, unsigned bytes0, unsigned bytes1, const char* region, unsigned long long recv_off, unsigned long long flag_off, unsigned long long slot_bytes, int ports, State* st, unsigned* status, unsigned long long give_up_ns, int blocks, unsigned slots) {
  const unsigned seq = *reinterpret_cast<volatile unsigned*>(&st->take_seq) + 1u;
  const unsigned slot = seq % slots;
  // 1. one thread a source and port waits for its flag (each written after that port's stripe, on the same queue pair)
  if ((int)threadIdx.x < nsrc * ports) {
    const unsigned src = (int)threadIdx.x < ports ? src0 : src1;
    const unsigned port = threadIdx.x % (unsigned)ports;
    const unsigned* f = reinterpret_cast<const unsigned*>(region + flag_off) + ((src * slots + slot) * PORTS_MAX + port) * FLAG_WORDS;
    wait_flag(f, seq, status, src, give_up_ns);
  }
  __syncthreads();
  // 2. every wait that failed set the status, so a clear status means every flag reached seq
  if (ld_relaxed_sys(status + FAILED) == 0u) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x, stride = (unsigned)blocks * blockDim.x;
    for (int k = 0; k < nsrc; ++k) {
      const uint4* in = reinterpret_cast<const uint4*>(region + recv_off + ((k == 0 ? src0 : src1) * slots + slot) * slot_bytes);
      uint4* out = k == 0 ? out0 : out1;
      const unsigned n = (k == 0 ? bytes0 : bytes1) / 16u;
      // four loads from the receive slot in flight a thread, then their stores
      for (unsigned i0 = index; i0 < n; i0 += 4u * stride) {
        uint4 v[4];
#pragma unroll
        for (unsigned u = 0; u < 4; ++u)
          if (i0 + u * stride < n) v[u] = ld_sys16(in + i0 + u * stride);
#pragma unroll
        for (unsigned u = 0; u < 4; ++u)
          if (i0 + u * stride < n) out[i0 + u * stride] = v[u];
      }
    }
  }
  // 3. the last block publishes the seq, also after a failure (the next take is then seq + 1, and returns at once)
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && last_block(&st->tail, blocks)) {
    __threadfence();
    *reinterpret_cast<volatile unsigned*>(&st->take_seq) = seq;
  }
}

// qualification_17c70e37 (TF_AFD_STRIPES): tf_xchg_take for messages that came in stripes (stripes_of), each stripe copied as soon as its flag word lands on every port, so the copy of the stripes in overlaps the wire of the rest; the same bytes land where tf_xchg_take puts them.
// Work unit w: stripe j = w / (nsrc * parts) of source k = (w / parts) % nsrc, part w % parts of it (stripe-major, the order they land); after any failure nothing more is copied.
extern "C" __global__ void __launch_bounds__(512) tf_xchg_take_striped(int nsrc, unsigned src0, unsigned src1, uint4* out0, uint4* out1, unsigned bytes0, unsigned bytes1, const char* region, unsigned long long recv_off, unsigned long long flag_off, unsigned long long slot_bytes, int ports, unsigned stripes, unsigned long long stripe_min, int parts, State* st, unsigned* status, unsigned long long give_up_ns, int blocks, unsigned slots) {
  const unsigned seq = *reinterpret_cast<volatile unsigned*>(&st->take_seq) + 1u;
  const unsigned slot = seq % slots;
  const unsigned n0 = stripes_of(stripes, stripe_min, bytes0), n1 = nsrc > 1 ? stripes_of(stripes, stripe_min, bytes1) : 0u;
  const unsigned nmax = n0 > n1 ? n0 : n1;
  const unsigned long long work = (unsigned long long)nmax * (unsigned)nsrc * (unsigned)parts;
  for (unsigned long long w = blockIdx.x; w < work; w += (unsigned)blocks) {
    const unsigned part = (unsigned)(w % (unsigned)parts), k = (unsigned)((w / (unsigned)parts) % (unsigned)nsrc), j = (unsigned)(w / ((unsigned)parts * (unsigned)nsrc));
    const unsigned n = k == 0 ? n0 : n1;
    if (j >= n) continue;  // the same for every thread of the block
    const unsigned src = k == 0 ? src0 : src1;
    // 1. one thread a port waits for stripe j's word (written after that port's stripe, on the same queue pair)
    if ((int)threadIdx.x < ports) {
      const unsigned* f = reinterpret_cast<const unsigned*>(region + flag_off) + ((src * slots + slot) * PORTS_MAX + threadIdx.x) * FLAG_WORDS + stripe_word(n, j);
      wait_word(f, seq, status, src, give_up_ns);
    }
    __syncthreads();
    // 2. a clear status: stripe j landed on every port; copy this unit's part of each port's stripe
    if (ld_relaxed_sys(status + FAILED) == 0u) {
      const uint4* in = reinterpret_cast<const uint4*>(region + recv_off + (src * slots + slot) * slot_bytes);
      uint4* out = k == 0 ? out0 : out1;
      const unsigned long long units = (k == 0 ? bytes0 : bytes1) / 16u, base = units / (unsigned)ports, extra = units % (unsigned)ports;
      for (unsigned p = 0; p < (unsigned)ports; ++p) {
        const unsigned long long first = base * p + (p < extra ? p : extra), share = base + (p < extra ? 1u : 0u);
        const unsigned long long lo = first + share * j / n, hi = first + share * (j + 1u) / n;
        copy_units(in, out, lo + (hi - lo) * part / (unsigned)parts, lo + (hi - lo) * (part + 1u) / (unsigned)parts);
      }
    }
    __syncthreads();  // the next unit's wait starts after this unit's copy, block-wide
  }
  // 3. as tf_xchg_take: the last block publishes the seq, also after a failure
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && last_block(&st->tail, blocks)) {
    __threadfence();
    *reinterpret_cast<volatile unsigned*>(&st->take_seq) = seq;
  }
}

// The lab tool's stand-in for an expert half (not on any serving path): out[r][d] = the sum over slots j with lo <= pick[r][j] < hi of wts[r][j] * (x[r][d] + (pick[r][j] & 7)), fp32 in slot order, no contraction; bf16_out rounds to bf16 (nearest even)
extern "C" __global__ void __launch_bounds__(256) tf_xchg_lab_experts(const unsigned short* x, const int* pick, const float* wts, void* out, int rows, int d, int slots, int lo, int hi, int bf16_out) {
  const long long n = (long long)rows * d;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
    const long long r = i / d;
    const unsigned bits = (unsigned)x[i] << 16;
    float xv;
    memcpy(&xv, &bits, 4);
    float acc = 0.0f;
    for (int j = 0; j < slots; ++j) {
      const int e = pick[r * slots + j];
      if (e >= lo && e < hi) acc = __fadd_rn(acc, __fmul_rn(wts[r * slots + j], __fadd_rn(xv, (float)(e & 7))));
    }
    if (bf16_out) {
      unsigned u;
      memcpy(&u, &acc, 4);
      reinterpret_cast<unsigned short*>(out)[i] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    } else {
      reinterpret_cast<float*>(out)[i] = acc;
    }
  }
}
