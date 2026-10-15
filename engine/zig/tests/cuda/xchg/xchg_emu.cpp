// fn_xchg.cu's kernels on the CPU, for tests and tf-xchg's no-GPU mode: each CUDA thread a host thread, one block at a time, __syncthreads a barrier of the block's live threads.
// Pointers are host addresses (the region's own, in place of its device mapping); each launch has its own barrier, so launches from several host threads may run at once.
#include "emu/cuda_emu.h"
#include "../../../kernels/cuda/fn_xchg.cu"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

thread_local tf_emu_dim3 threadIdx, blockIdx, blockDim, gridDim;

namespace {

// A block's barrier over its live threads (a thread that returned no longer counts)
struct Barrier {
  std::mutex m;
  std::condition_variable cv;
  unsigned expected = 0, arrived = 0, gen = 0;

  void sync() {
    std::unique_lock<std::mutex> l(m);
    const unsigned g = gen;
    if (++arrived == expected) {
      arrived = 0;
      gen++;
      cv.notify_all();
    } else {
      cv.wait(l, [&] { return gen != g; });
    }
  }

  void leave() {
    std::unique_lock<std::mutex> l(m);
    expected--;
    if (arrived > 0 && arrived == expected) {
      arrived = 0;
      gen++;
      cv.notify_all();
    }
  }
};

thread_local Barrier* block_barrier = nullptr;

template <class F>
void run(unsigned blocks, unsigned threads, const F& f) {
  for (unsigned b = 0; b < blocks; b++) {
    Barrier bar;
    bar.expected = threads;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; t++)
      pool.emplace_back([&, t] {
        threadIdx = {t, 0, 0};
        blockIdx = {b, 0, 0};
        blockDim = {threads, 1, 1};
        gridDim = {blocks, 1, 1};
        block_barrier = &bar;
        f();
        bar.leave();
      });
    for (auto& th : pool) th.join();
  }
}

}  // namespace

void tf_emu_syncthreads() { block_barrier->sync(); }

extern "C" void tf_xchg_emu_post(unsigned blocks, unsigned threads, const void* in, unsigned bytes, char* region, unsigned long long send_off, unsigned long long ctrl_off, unsigned long long slot_bytes, void* st, const unsigned* status, unsigned slots) {
  run(blocks, threads, [&] { tf_xchg_post(static_cast<const uint4*>(in), bytes, region, send_off, ctrl_off, slot_bytes, static_cast<State*>(st), status, (int)blocks, slots); });
}

extern "C" void tf_xchg_emu_take(unsigned blocks, unsigned threads, int nsrc, unsigned src0, unsigned src1, void* out0, void* out1, unsigned bytes0, unsigned bytes1, const char* region, unsigned long long recv_off, unsigned long long flag_off, unsigned long long slot_bytes, int ports, void* st, unsigned* status, unsigned long long give_up_ns, unsigned slots) {
  run(blocks, threads, [&] { tf_xchg_take(nsrc, src0, src1, static_cast<uint4*>(out0), static_cast<uint4*>(out1), bytes0, bytes1, region, recv_off, flag_off, slot_bytes, ports, static_cast<State*>(st), status, give_up_ns, (int)blocks, slots); });
}

// qualification_17c70e37: the striped take (each block's work units in order; a block waits only for the proxy, never for another block)
extern "C" void tf_xchg_emu_take_striped(unsigned blocks, unsigned threads, int nsrc, unsigned src0, unsigned src1, void* out0, void* out1, unsigned bytes0, unsigned bytes1, const char* region, unsigned long long recv_off, unsigned long long flag_off, unsigned long long slot_bytes, int ports, unsigned stripes, unsigned long long stripe_min, int parts, void* st, unsigned* status, unsigned long long give_up_ns, unsigned slots) {
  run(blocks, threads, [&] { tf_xchg_take_striped(nsrc, src0, src1, static_cast<uint4*>(out0), static_cast<uint4*>(out1), bytes0, bytes1, region, recv_off, flag_off, slot_bytes, ports, stripes, stripe_min, parts, static_cast<State*>(st), status, give_up_ns, (int)blocks, slots); });
}

extern "C" void tf_xchg_emu_lab_experts(unsigned blocks, unsigned threads, const unsigned short* x, const int* pick, const float* wts, void* out, int rows, int d, int slots, int lo, int hi, int bf16_out) {
  run(blocks, threads, [&] { tf_xchg_lab_experts(x, pick, wts, out, rows, d, slots, lo, hi, bf16_out); });
}
