// Modified for V4 by Scott Quarrier with Helios: native hybrid integration/adaptation.
// Compared with TensorFold v1.2.8 and/or the retained public V3 source; original author notices remain.
// The host half of the one-shot RoCE all-gather between two ranks (two DGX Sparks): a proxy thread that RDMA-writes each staged shard to the peer.
// Source: Mia's patch 0006, MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold @ cbfc429, patches/0006-two-ranks.patch, zig/src/cuda/roce_proxy.c (Apache-2.0, Copyright 2026 MiaAI-Lab).
// Changed here: the header comments condensed to one line each; tf_roce_proxy_built added. The proxy's code is unchanged but for qualification_17c70e37's stripes (tf_roce_set_stripes, post_striped; off unless set, and then post's one write a port as before) and qualification_9e49592e's: the slots a source a count given at open (tf_roce_open_slots, 2 to ROCE_SLOTS_MAX; tf_roce_open keeps ROCE_SLOTS).
// Direct-port change by Helios: bounded1MiB payload WRs; full actual-row message/flag ordering retained.
// Protocol: b12x's "RoCEnante" (https://github.com/local-inference-lab/b12x, b12x/comm/roce, by local-inference-lab, Apache-2.0), in new code.
// A rank stages its shard into a pinned send slot and rings a doorbell; this thread writes it into the peer's receive slot over every port, each stripe then its flag.
// The peer's kernel (zig/kernels/cuda/fn_roce.cu) waits for every port's flag, then reads the shard; zig/src/cuda/roce.zig drives both halves.
// Region (pinned host memory mapped into the GPU, registered with every port), offsets from tf_roce_layout:
//   recv[src][slot]: slots x world x slot_bytes, written by the peers' NICs
//   flag[src][slot][port]: 128 B each, a u32 sequence, written by the peers' NICs after each stripe
//   send[slot]: slots x slot_bytes, written by this rank's kernel
//   ctrl: 128 B, u32 seq (doorbell), u32 pad, u32 error seq, u32 missing peer, u32 bytes[slots], written by this rank's kernel
// libibverbs is opened at run time (dlopen); post_send and poll_cq are the header's inline wrappers over the provider's function table.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROCE_SLOTS 2
#define ROCE_SLOTS_MAX 4  // qualification_9e49592e: TF_AFD_LANES' most requests in flight a link (proto.LANES_MAX)
#define ROCE_PORTS_MAX 2
#define ROCE_WORLD_MAX 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_IB_PORT 1
#define ROCE_DEPTH 256
#define ROCE_STRIPES_MAX 16  // qualification_17c70e37: a flag's 32 words hold every stripe's

// the exported libibverbs functions this file calls (by pointer: dlopen)
struct verbs {
  void *lib;
  struct ibv_device **(*get_device_list)(int *);
  void (*free_device_list)(struct ibv_device **);
  const char *(*get_device_name)(struct ibv_device *);
  struct ibv_context *(*open_device)(struct ibv_device *);
  int (*close_device)(struct ibv_context *);
  int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
  int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
  struct ibv_pd *(*alloc_pd)(struct ibv_context *);
  int (*dealloc_pd)(struct ibv_pd *);
  struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
  int (*dereg_mr)(struct ibv_mr *);
  struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
  int (*destroy_cq)(struct ibv_cq *);
  struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
  int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
  int (*destroy_qp)(struct ibv_qp *);
  const char *(*wc_status_str)(enum ibv_wc_status);
};

// what a rank tells its peer to connect (exchanged over the control link)
typedef struct {
  uint64_t region;  // the region's host address (the NIC's address space)
  uint32_t rkey[ROCE_PORTS_MAX];
  uint32_t qpn[ROCE_PORTS_MAX];
  uint32_t mtu[ROCE_PORTS_MAX];
  uint32_t ports;
  uint32_t slot_pages;  // slot_bytes / 4096: both ranks must use the same slots
  uint8_t gid[ROCE_PORTS_MAX][16];
} tf_roce_blob;

typedef struct {
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_mr *mr;
  struct ibv_cq *cq;
  struct ibv_qp *qp;
  union ibv_gid gid;
  enum ibv_mtu mtu;
  uint32_t outstanding;
} port_t;

typedef struct tf_roce {
  struct verbs v;
  int rank, world, ports, gid_index;
  uint8_t traffic_class;
  port_t port[ROCE_PORTS_MAX];
  uint8_t *region;
  uint64_t region_bytes, slot_bytes;
  uint32_t slots;  // qualification_9e49592e: the slots a source (ROCE_SLOTS unless opened with more)
  uint64_t recv_off, flag_off, send_off, ctrl_off;
  uint64_t peer_region;
  uint32_t peer_rkey[ROCE_PORTS_MAX];
  pthread_t thread;
  int thread_started;
  atomic_int running, failed, thread_done;
  _Atomic uint64_t close_deadline;
  uint32_t posted_seq;
  _Atomic uint64_t ops;
  uint64_t writes;
  uint32_t stripes;     // qualification_17c70e37 (TF_AFD_STRIPES): a message of stripes x stripe_min bytes or more goes in stripes; 0: one write a port
  uint64_t stripe_min;
  char err[512];
} tf_roce;

static void fail(tf_roce *r, const char *what, int e) {
  snprintf(r->err, sizeof(r->err), "%s: %s", what, e ? strerror(e) : "failed");
}

uint64_t tf_roce_blob_bytes(void) { return sizeof(tf_roce_blob); }

// {recv, flag, send, ctrl, total} offsets for `world` ranks, `slots` slots a source (qualification_9e49592e) of `slot_bytes` (a multiple of 4096)
static int layout_slots(int world, uint64_t slot_bytes, uint32_t slots, uint64_t *out) {
  if (world < 2 || world > ROCE_WORLD_MAX || slot_bytes == 0 || slot_bytes % 4096 != 0 || slot_bytes > (1ull << 30) ||
      slots < ROCE_SLOTS || slots > ROCE_SLOTS_MAX)
    return -1;
  const uint64_t recv = 0;
  const uint64_t flag = recv + (uint64_t)world * slots * slot_bytes;
  const uint64_t send = flag + (uint64_t)world * slots * ROCE_PORTS_MAX * ROCE_FLAG_STRIDE;
  const uint64_t ctrl = send + (uint64_t)slots * slot_bytes;
  out[0] = recv;
  out[1] = flag;
  out[2] = send;
  out[3] = ctrl;
  out[4] = ctrl + ROCE_FLAG_STRIDE;
  return 0;
}

// {recv, flag, send, ctrl, total} offsets for `world` ranks and slots of `slot_bytes` (a multiple of 4096)
int tf_roce_layout(int world, uint64_t slot_bytes, uint64_t *out) { return layout_slots(world, slot_bytes, ROCE_SLOTS, out); }

static int load_verbs(tf_roce *r) {
  struct verbs *v = &r->v;
  v->lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!v->lib) {
    snprintf(r->err, sizeof(r->err), "dlopen libibverbs.so.1: %s", dlerror());
    return -1;
  }
#define SYM(field, name)                                                       \
  do {                                                                         \
    *(void **)&v->field = dlsym(v->lib, name);                                 \
    if (!v->field) {                                                           \
      snprintf(r->err, sizeof(r->err), "libibverbs.so.1 has no %s", name);     \
      return -1;                                                               \
    }                                                                          \
  } while (0)
  SYM(get_device_list, "ibv_get_device_list");
  SYM(free_device_list, "ibv_free_device_list");
  SYM(get_device_name, "ibv_get_device_name");
  SYM(open_device, "ibv_open_device");
  SYM(close_device, "ibv_close_device");
  SYM(query_port, "ibv_query_port");
  SYM(query_gid, "ibv_query_gid");
  SYM(alloc_pd, "ibv_alloc_pd");
  SYM(dealloc_pd, "ibv_dealloc_pd");
  SYM(reg_mr, "ibv_reg_mr");
  SYM(dereg_mr, "ibv_dereg_mr");
  SYM(create_cq, "ibv_create_cq");
  SYM(destroy_cq, "ibv_destroy_cq");
  SYM(create_qp, "ibv_create_qp");
  SYM(modify_qp, "ibv_modify_qp");
  SYM(destroy_qp, "ibv_destroy_qp");
  SYM(wc_status_str, "ibv_wc_status_str");
#undef SYM
  return 0;
}

static int open_port(tf_roce *r, int i, const char *name) {
  struct verbs *v = &r->v;
  port_t *p = &r->port[i];
  int n = 0;
  struct ibv_device **list = v->get_device_list(&n);
  if (!list) {
    fail(r, "ibv_get_device_list", errno);
    return -1;
  }
  struct ibv_device *dev = NULL;
  for (int k = 0; k < n; k++)
    if (strcmp(v->get_device_name(list[k]), name) == 0) dev = list[k];
  if (dev) p->ctx = v->open_device(dev);
  v->free_device_list(list);
  if (!dev) {
    snprintf(r->err, sizeof(r->err), "no RDMA device %s", name);
    return -1;
  }
  if (!p->ctx) {
    fail(r, "ibv_open_device", errno);
    return -1;
  }
  struct ibv_port_attr pa;
  memset(&pa, 0, sizeof(pa));
  if (v->query_port(p->ctx, ROCE_IB_PORT, &pa) != 0) {
    fail(r, "ibv_query_port", errno);
    return -1;
  }
  if (pa.state != IBV_PORT_ACTIVE) {
    snprintf(r->err, sizeof(r->err), "%s port %d is not active", name, ROCE_IB_PORT);
    return -1;
  }
  p->mtu = pa.active_mtu;
  if (v->query_gid(p->ctx, ROCE_IB_PORT, r->gid_index, &p->gid) != 0) {
    fail(r, "ibv_query_gid", errno);
    return -1;
  }
  if (!(p->pd = v->alloc_pd(p->ctx))) {
    fail(r, "ibv_alloc_pd", errno);
    return -1;
  }
  if (!(p->mr = v->reg_mr(p->pd, r->region, r->region_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE))) {
    fail(r, "ibv_reg_mr", errno);
    return -1;
  }
  if (!(p->cq = v->create_cq(p->ctx, ROCE_DEPTH, NULL, NULL, 0))) {
    fail(r, "ibv_create_cq", errno);
    return -1;
  }
  struct ibv_qp_init_attr qa;
  memset(&qa, 0, sizeof(qa));
  qa.send_cq = p->cq;
  qa.recv_cq = p->cq;
  qa.qp_type = IBV_QPT_RC;
  qa.cap.max_send_wr = ROCE_DEPTH;
  qa.cap.max_recv_wr = 1;
  qa.cap.max_send_sge = 1;
  qa.cap.max_recv_sge = 1;
  qa.cap.max_inline_data = 16;
  if (!(p->qp = v->create_qp(p->pd, &qa))) {
    fail(r, "ibv_create_qp", errno);
    return -1;
  }
  struct ibv_qp_attr a;
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_INIT;
  a.port_num = ROCE_IB_PORT;
  a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  const int rc = v->modify_qp(p->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (rc != 0) {
    fail(r, "ibv_modify_qp INIT", rc);
    return -1;
  }
  return 0;
}

void tf_roce_close(tf_roce *r);

// Opens `ports` RDMA devices (names[i]) on port 1 with GID `gid_index`, registers `region`, one RC queue pair a port; `slots` slots a source (qualification_9e49592e); NULL with the reason in `err`.
tf_roce *tf_roce_open_slots(int world, int rank, const char *const *names, int ports, int gid_index, void *region,
                            uint64_t region_bytes, uint64_t slot_bytes, uint32_t slots, char *err, uint64_t err_len) {
  uint64_t lay[5];
  if (layout_slots(world, slot_bytes, slots, lay) != 0 || lay[4] > region_bytes || rank < 0 || rank >= world ||
      ports < 1 || ports > ROCE_PORTS_MAX) {
    snprintf(err, err_len, "invalid RoCE geometry (world %d rank %d ports %d slot %llu slots %u region %llu)", world,
             rank, ports, (unsigned long long)slot_bytes, slots, (unsigned long long)region_bytes);
    return NULL;
  }
  tf_roce *r = calloc(1, sizeof(*r));
  if (!r) {
    snprintf(err, err_len, "out of memory");
    return NULL;
  }
  r->rank = rank;
  r->world = world;
  r->ports = ports;
  r->gid_index = gid_index;
  r->region = region;
  r->region_bytes = region_bytes;
  r->slot_bytes = slot_bytes;
  r->slots = slots;
  r->recv_off = lay[0];
  r->flag_off = lay[1];
  r->send_off = lay[2];
  r->ctrl_off = lay[3];
  if (load_verbs(r) != 0) goto bad;
  for (int i = 0; i < ports; i++)
    if (open_port(r, i, names[i]) != 0) goto bad;
  return r;
bad:
  snprintf(err, err_len, "%s", r->err);
  tf_roce_close(r);
  return NULL;
}

// tf_roce_open_slots at ROCE_SLOTS slots a source (Mia's patch 0006's interface).
tf_roce *tf_roce_open(int world, int rank, const char *const *names, int ports, int gid_index, void *region,
                      uint64_t region_bytes, uint64_t slot_bytes, char *err, uint64_t err_len) {
  return tf_roce_open_slots(world, rank, names, ports, gid_index, region, region_bytes, slot_bytes, ROCE_SLOTS, err,
                            err_len);
}

int tf_roce_blob_of(tf_roce *r, void *out) {
  tf_roce_blob b;
  memset(&b, 0, sizeof(b));
  b.region = (uint64_t)(uintptr_t)r->region;
  b.ports = (uint32_t)r->ports;
  b.slot_pages = (uint32_t)(r->slot_bytes / 4096);
  for (int i = 0; i < r->ports; i++) {
    b.rkey[i] = r->port[i].mr->rkey;
    b.qpn[i] = r->port[i].qp->qp_num;
    b.mtu[i] = (uint32_t)r->port[i].mtu;
    memcpy(b.gid[i], r->port[i].gid.raw, 16);
  }
  memcpy(out, &b, sizeof(b));
  return 0;
}

// The peer's blob: each port's queue pair to RTR and RTS toward the peer's queue pair on the same port index.
int tf_roce_connect(tf_roce *r, const void *peer_blob) {
  tf_roce_blob b;
  memcpy(&b, peer_blob, sizeof(b));
  if ((int)b.ports != r->ports) {
    snprintf(r->err, sizeof(r->err), "the peer has %u RoCE ports, this rank %d", b.ports, r->ports);
    return -1;
  }
  if ((uint64_t)b.slot_pages * 4096 != r->slot_bytes) {
    snprintf(r->err, sizeof(r->err), "the peer's RoCE slots are %u KiB, this rank's %llu KiB", b.slot_pages * 4,
             (unsigned long long)(r->slot_bytes >> 10));
    return -1;
  }
  r->peer_region = b.region;
  for (int i = 0; i < r->ports; i++) {
    port_t *p = &r->port[i];
    r->peer_rkey[i] = b.rkey[i];
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = (enum ibv_mtu)(b.mtu[i] < (uint32_t)p->mtu ? b.mtu[i] : (uint32_t)p->mtu);
    a.dest_qp_num = b.qpn[i];
    a.rq_psn = 0;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1;
    a.ah_attr.port_num = ROCE_IB_PORT;
    memcpy(a.ah_attr.grh.dgid.raw, b.gid[i], 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.grh.traffic_class = r->traffic_class;
    int rc = r->v.modify_qp(p->qp, &a,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
      fail(r, "ibv_modify_qp RTR", rc);
      return -1;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = 0;
    a.max_rd_atomic = 1;
    rc = r->v.modify_qp(p->qp, &a,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                            IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
      fail(r, "ibv_modify_qp RTS", rc);
      return -1;
    }
  }
  return 0;
}

// Completions of signalled flag writes (one a stripe); any error fails the proxy.
static int reap(tf_roce *r, int i) {
  struct ibv_wc wc[16];
  const int n = ibv_poll_cq(r->port[i].cq, 16, wc);
  if (n < 0) {
    fail(r, "ibv_poll_cq", errno);
    return -1;
  }
  for (int k = 0; k < n; k++) {
    if (wc[k].status != IBV_WC_SUCCESS) {
      snprintf(r->err, sizeof(r->err), "RDMA write on port %d failed: %s (vendor error 0x%x) after seq %u", i,
               r->v.wc_status_str(wc[k].status), wc[k].vendor_err, r->posted_seq);
      return -1;
    }
    r->port[i].outstanding--;
    r->writes++;
  }
  return 0;
}

// Helios direct-V3 adapter: keep actual-R slots, but every native RDMA payload write is <=1MiB.
// Each signalled chunk is counted until reaped; a partial submit never publishes the final message flag.
#define TF_AFD_MAX_POST (1u << 20)
static int post_payload(tf_roce *r, int i, uint64_t send, uint64_t recv, uint32_t bytes) {
  port_t *p = &r->port[i];
  for (uint32_t off = 0; off < bytes;) {
    while (p->outstanding >= ROCE_DEPTH / 4) {
      if (reap(r, i) != 0) return -1;
      if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
        snprintf(r->err, sizeof(r->err), "stopped with partial V3 payload on port %d", i);
        return -1;
      }
    }
    const uint32_t n = bytes - off > TF_AFD_MAX_POST ? TF_AFD_MAX_POST : bytes - off;
    struct ibv_sge sg = {.addr = send + off, .length = n, .lkey = p->mr->lkey};
    struct ibv_send_wr wr; memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)i; wr.sg_list = &sg; wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE; wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = recv + off; wr.wr.rdma.rkey = r->peer_rkey[i];
    struct ibv_send_wr *bad = NULL;
    const int rc = ibv_post_send(p->qp, &wr, &bad);
    if (rc != 0) { fail(r, "ibv_post_send partial V3 payload", rc); return -1; }
    p->outstanding++; off += n;
  }
  return 0;
}

static int post_striped(tf_roce *r, uint32_t seq, uint32_t bytes, uint32_t nst);  // qualification_17c70e37, below

// Sequence `seq`'s shard (`bytes`, a multiple of 16) into the peer's recv[rank][seq % slots], over the ports in 16-byte units: each stripe, then its flag.
static int post(tf_roce *r, uint32_t seq, uint32_t bytes) {
  if (bytes == 0 || bytes % 16 != 0 || bytes > r->slot_bytes) {
    snprintf(r->err, sizeof(r->err), "RoCE shard of %u bytes (a positive multiple of 16 up to %llu)", bytes,
             (unsigned long long)r->slot_bytes);
    return -1;
  }
  if (r->stripes >= 2 && (uint64_t)bytes >= (uint64_t)r->stripes * r->stripe_min) return post_striped(r, seq, bytes, r->stripes);
  const uint32_t slot = seq % r->slots;
  uint8_t *send = r->region + r->send_off + (uint64_t)slot * r->slot_bytes;
  const uint64_t recv = r->peer_region + r->recv_off + ((uint64_t)r->rank * r->slots + slot) * r->slot_bytes;
  const uint32_t units = bytes / 16;
  uint32_t done = 0;
  for (int i = 0; i < r->ports; i++) {
    port_t *p = &r->port[i];
    while (p->outstanding >= ROCE_DEPTH / 4) {
      if (reap(r, i) != 0) return -1;
      if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
        snprintf(r->err, sizeof(r->err), "stopped with %u writes outstanding on port %d", p->outstanding, i);
        return -1;
      }
    }
    const uint32_t share = units / r->ports + ((uint32_t)i < units % r->ports ? 1u : 0u);
    const uint64_t off = (uint64_t)done * 16;
    uint32_t value = seq;
    struct ibv_sge fs = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
    struct ibv_send_wr fw;
    memset(&fw, 0, sizeof(fw));
    fw.wr_id = (uint64_t)i;
    fw.sg_list = &fs;
    fw.num_sge = 1;
    fw.opcode = IBV_WR_RDMA_WRITE;
    fw.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    fw.wr.rdma.remote_addr = r->peer_region + r->flag_off +
                             (((uint64_t)r->rank * r->slots + slot) * ROCE_PORTS_MAX + (uint64_t)i) * ROCE_FLAG_STRIDE;
    fw.wr.rdma.rkey = r->peer_rkey[i];
    if (share && post_payload(r, i, (uint64_t)(uintptr_t)(send + off), recv + off, share * 16) != 0) return -1;
    struct ibv_send_wr *first = &fw;
    struct ibv_send_wr *bad = NULL;
    const int rc = ibv_post_send(p->qp, first, &bad);
    if (rc != 0) {
      fail(r, "ibv_post_send", rc);
      return -1;
    }
    p->outstanding++;
    done += share;
  }
  atomic_fetch_add_explicit(&r->ops, 1, memory_order_relaxed);
  for (int i = 0; i < r->ports; i++)
    if (reap(r, i) != 0) return -1;
  return 0;
}

// qualification_17c70e37: post's message in `nst` stripes a port, each stripe then its own flag word (stripe j < nst - 1: word 1 + j; the last: word 0, today's, after every other), every flag signalled.
static int post_striped(tf_roce *r, uint32_t seq, uint32_t bytes, uint32_t nst) {
  const uint32_t slot = seq % r->slots;
  uint8_t *send = r->region + r->send_off + (uint64_t)slot * r->slot_bytes;
  const uint64_t recv = r->peer_region + r->recv_off + ((uint64_t)r->rank * r->slots + slot) * r->slot_bytes;
  const uint32_t units = bytes / 16;
  uint32_t done = 0, value = seq;
  for (int i = 0; i < r->ports; i++) {
    port_t *p = &r->port[i];
    while (p->outstanding + nst > ROCE_DEPTH / 4) {
      if (reap(r, i) != 0) return -1;
      if (!atomic_load_explicit(&r->running, memory_order_relaxed)) {
        snprintf(r->err, sizeof(r->err), "stopped with %u writes outstanding on port %d", p->outstanding, i);
        return -1;
      }
    }
    const uint32_t share = units / r->ports + ((uint32_t)i < units % r->ports ? 1u : 0u);
    const uint64_t flag = r->peer_region + r->flag_off +
                          (((uint64_t)r->rank * r->slots + slot) * ROCE_PORTS_MAX + (uint64_t)i) * ROCE_FLAG_STRIDE;
    for (uint32_t j = 0; j < nst; j++) {
      const uint32_t lo = (uint32_t)((uint64_t)share * j / nst), hi = (uint32_t)((uint64_t)share * (j + 1) / nst);
      if (hi > lo && post_payload(r, i, (uint64_t)(uintptr_t)(send + (uint64_t)(done + lo) * 16), recv + (uint64_t)(done + lo) * 16, (hi - lo) * 16) != 0) return -1;
      struct ibv_sge fs = {.addr = (uint64_t)(uintptr_t)&value, .length = 4, .lkey = 0};
      struct ibv_send_wr fw; memset(&fw, 0, sizeof(fw));
      fw.wr_id = (uint64_t)i; fw.sg_list = &fs; fw.num_sge = 1;
      fw.opcode = IBV_WR_RDMA_WRITE; fw.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
      fw.wr.rdma.remote_addr = flag + 4u * (j + 1 == nst ? 0u : 1u + j);
      fw.wr.rdma.rkey = r->peer_rkey[i];
      struct ibv_send_wr *bad = NULL;
      const int rc = ibv_post_send(p->qp, &fw, &bad);
      if (rc != 0) { fail(r, "ibv_post_send V3 stripe flag", rc); return -1; }
      p->outstanding++;
    }
    done += share;
  }
  atomic_fetch_add_explicit(&r->ops, 1, memory_order_relaxed);
  for (int i = 0; i < r->ports; i++)
    if (reap(r, i) != 0) return -1;
  return 0;
}

// qualification_17c70e37: this pair's messages of stripes x stripe_min bytes or more go in stripes (2 to ROCE_STRIPES_MAX; 0 or 1: one write a port); before tf_roce_start
int tf_roce_set_traffic_class(tf_roce *r, uint8_t traffic_class) {
  if(r->thread_started)return -1;
  r->traffic_class=traffic_class;return 0;
}
int tf_roce_set_stripes(tf_roce *r, uint32_t stripes, uint64_t stripe_min) {
  if (stripes > ROCE_STRIPES_MAX || (stripes >= 2 && (stripe_min < 16 || stripe_min % 16 != 0)) || r->thread_started) {
    snprintf(r->err, sizeof(r->err), "RoCE stripes %u of %llu bytes (2-%d of a positive multiple of 16, set before the start)",
             stripes, (unsigned long long)stripe_min, ROCE_STRIPES_MAX);
    return -1;
  }
  r->stripes = stripes;
  r->stripe_min = stripe_min;
  return 0;
}

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// The proxy: poll the doorbell and post every sequence not yet posted, in order (at most `slots` pending); after an idle second it naps 50 us a poll.
static void *proxy(void *arg) {
  tf_roce *r = arg;
  volatile uint32_t *ctrl = (volatile uint32_t *)(r->region + r->ctrl_off);
  uint64_t polls = 0, last = now_ns();
  int napping = 0;
  const struct timespec nap = {0, 50000};
  while (atomic_load_explicit(&r->running, memory_order_relaxed)) {
    const uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
    if (seq == r->posted_seq) {
      if ((++polls & 4095) == 0) {
        for (int i = 0; i < r->ports; i++)
          if (reap(r, i) != 0) goto failed;
        if (!napping && now_ns() - last > 1000000000ull) napping = 1;
      }
      if (napping) nanosleep(&nap, NULL);
      continue;
    }
    napping = 0;
    last = now_ns();
    const uint32_t pending = seq - r->posted_seq;
    if (pending > r->slots) {
      snprintf(r->err, sizeof(r->err), "the doorbell skipped %u sequences (posted %u, rung %u)", pending,
               r->posted_seq, seq);
      goto failed;
    }
    for (uint32_t s = r->posted_seq + 1; s != seq + 1; s++) {
      if (post(r, s, ctrl[4 + s % r->slots]) != 0) goto failed;
      r->posted_seq = s;
    }
  }
  atomic_store_explicit(&r->thread_done, 1, memory_order_release);
  return NULL;
failed:
  atomic_store(&r->failed, 1);
  atomic_store_explicit(&r->thread_done, 1, memory_order_release);
  return NULL;
}

int tf_roce_start(tf_roce *r) {
  if (r->thread_started) return 0;
  volatile uint32_t *ctrl = (volatile uint32_t *)(r->region + r->ctrl_off);
  r->posted_seq = ctrl[0];
  atomic_store(&r->failed, 0);
  atomic_store(&r->running, 1);
  const int rc = pthread_create(&r->thread, NULL, proxy, r);
  if (rc != 0) {
    atomic_store(&r->running, 0);
    fail(r, "pthread_create", rc);
    return -1;
  }
  r->thread_started = 1;
  return 0;
}

int tf_roce_proxy_built(void) { return 1; }
int tf_roce_failed(tf_roce *r) { return atomic_load(&r->failed); }
const char *tf_roce_error(tf_roce *r) { return r->err; }
uint64_t tf_roce_ops(tf_roce *r) { return atomic_load_explicit(&r->ops, memory_order_relaxed); }

/* Helios: only a typed, deadline-bound quiet proof permits registered-memory release.
 * The CUDA consumer must be complete BEFORE this call. Failure/unknown partial writes
 * keep the proxy, QP and MR owned; stopping a thread is not a quiet proof. */
#define TF_AFD_TYPED_QUIESCENCE 1
#define TF_AFD_QUIET_OK 0
#define TF_AFD_QUIET_FAILED 1
#define TF_AFD_QUIET_DEADLINE 2
int tf_roce_quiesce(tf_roce *r, uint64_t until) {
  uint64_t unset = 0;
  atomic_compare_exchange_strong(&r->close_deadline, &unset, until);
  const uint64_t bound = atomic_load_explicit(&r->close_deadline, memory_order_acquire);
  if (until > bound) until = bound; /* close cannot renew its first deadline */
  if (now_ns() >= until) return TF_AFD_QUIET_DEADLINE;
  atomic_store_explicit(&r->running, 0, memory_order_release);
  while (r->thread_started && !atomic_load_explicit(&r->thread_done, memory_order_acquire)) {
    if (now_ns() >= until) return TF_AFD_QUIET_DEADLINE;
    const struct timespec nap = {0, 50000}; nanosleep(&nap, NULL);
  }
  if (r->thread_started) { pthread_join(r->thread, NULL); r->thread_started = 0; }
  const volatile uint32_t *ctrl = (const volatile uint32_t *)(r->region + r->ctrl_off);
  if (atomic_load_explicit(&r->failed, memory_order_acquire) ||
      __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE) != r->posted_seq) return TF_AFD_QUIET_FAILED;
  for (int i = 0; i < r->ports; i++) while (r->port[i].outstanding) {
    if (now_ns() >= until) return TF_AFD_QUIET_DEADLINE;
    if (reap(r, i) != 0) { atomic_store(&r->failed, 1); return TF_AFD_QUIET_FAILED; }
  }
  return now_ns() < until ? TF_AFD_QUIET_OK : TF_AFD_QUIET_DEADLINE;
}

void tf_roce_close(tf_roce *r) {
  if (!r) return;
  if (r->thread_started) {
    atomic_store(&r->running, 0);
    pthread_join(r->thread, NULL);
  }
  for (int i = 0; i < ROCE_PORTS_MAX; i++) {
    port_t *p = &r->port[i];
    if (p->qp) r->v.destroy_qp(p->qp);
    if (p->cq) r->v.destroy_cq(p->cq);
    if (p->mr) r->v.dereg_mr(p->mr);
    if (p->pd) r->v.dealloc_pd(p->pd);
    if (p->ctx) r->v.close_device(p->ctx);
  }
  if (r->v.lib) dlclose(r->v.lib);
  free(r);
}
