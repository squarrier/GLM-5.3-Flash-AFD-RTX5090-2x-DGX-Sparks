// roce_proxy.c's interface where libibverbs' headers are missing at build time: the layout as there, and every open refused, so comm.zig keeps NCCL for every gather.
// Source: Mia's patch 0006, MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, zig/src/cuda/roce_proxy_stub.c (Apache-2.0, Copyright 2026 MiaAI-Lab); changed here: tf_roce_proxy_built, tf_roce_set_stripes (qualification_17c70e37); qualification_9e49592e: tf_roce_open_slots.
#include <stdint.h>
#include <stdio.h>

#define ROCE_SLOTS 2
#define ROCE_PORTS_MAX 2
#define ROCE_FLAG_STRIDE 128

typedef struct tf_roce tf_roce;

uint64_t tf_roce_blob_bytes(void) { return 72; }

int tf_roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
  if (world < 2 || world > 2 || slot_bytes == 0 || slot_bytes % 4096 != 0 || slot_bytes > (1ull << 30)) return -1;
  out[0] = 0;
  out[1] = (uint64_t)world * ROCE_SLOTS * slot_bytes;
  out[2] = out[1] + (uint64_t)world * ROCE_SLOTS * ROCE_PORTS_MAX * ROCE_FLAG_STRIDE;
  out[3] = out[2] + ROCE_SLOTS * slot_bytes;
  out[4] = out[3] + ROCE_FLAG_STRIDE;
  return 0;
}

tf_roce *tf_roce_open(int world, int rank, const char *const *names, int ports, int gid_index, void *region,
                      uint64_t region_bytes, uint64_t slot_bytes, char *err, uint64_t err_len) {
  (void)world, (void)rank, (void)names, (void)ports, (void)gid_index, (void)region, (void)region_bytes, (void)slot_bytes;
  snprintf(err, err_len, "built without libibverbs headers: no RoCE all-gather");
  return NULL;
}
// qualification_9e49592e: roce_proxy.c's open with its slots a source counted; refused here as every open
tf_roce *tf_roce_open_slots(int world, int rank, const char *const *names, int ports, int gid_index, void *region,
                            uint64_t region_bytes, uint64_t slot_bytes, uint32_t slots, char *err, uint64_t err_len) {
  (void)slots;
  return tf_roce_open(world, rank, names, ports, gid_index, region, region_bytes, slot_bytes, err, err_len);
}
int tf_roce_blob_of(tf_roce *r, void *out) { (void)r, (void)out; return -1; }
int tf_roce_connect(tf_roce *r, const void *peer) { (void)r, (void)peer; return -1; }
int tf_roce_start(tf_roce *r) { (void)r; return -1; }
int tf_roce_set_stripes(tf_roce *r, uint32_t stripes, uint64_t stripe_min) { (void)r, (void)stripes, (void)stripe_min; return -1; }
int tf_roce_proxy_built(void) { return 0; }
int tf_roce_failed(tf_roce *r) { (void)r; return 1; }
const char *tf_roce_error(tf_roce *r) { (void)r; return "no RoCE in this build"; }
uint64_t tf_roce_ops(tf_roce *r) { (void)r; return 0; }
void tf_roce_close(tf_roce *r) { (void)r; }
