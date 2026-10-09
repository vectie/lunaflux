#ifndef LUNAFLUX_CUDA_STREAMING_INTERNAL_H
#define LUNAFLUX_CUDA_STREAMING_INTERNAL_H

#include "resource_internal.h"

enum lf_transfer_phase { LF_TRANSFER_IDLE, LF_TRANSFER_QUEUED,
  LF_TRANSFER_RECORDED, LF_TRANSFER_COMPLETE, LF_TRANSFER_POISONED };

typedef struct {
  size_t host_offset;
  size_t device_offset;
  size_t bytes;
} lf_transfer_region;

typedef struct {
  CUstream stream;
  CUevent event;
  int32_t phase;
  int32_t count;
  lf_transfer_region *regions;
} lf_transfer_lane;

typedef struct {
  lf_allocation *allocation;
  void *host;
  size_t host_bytes;
  int32_t lanes;
  int32_t max_regions;
  int32_t usable;
  lf_transfer_lane *lane;
  lf_transfer_region *regions;
  atomic_int state;
  atomic_int active_operations;
  atomic_int gate;
} lf_streaming_pool;

int32_t lf_streaming_begin(lf_streaming_pool *pool);
void lf_streaming_end(lf_streaming_pool *pool);
int lf_transfer_pending(int32_t phase);
lf_streaming_pool *lunaflux_cuda_streaming_create(
  lf_context *, lf_allocation *, int64_t, int32_t, int32_t, int32_t *);
int32_t lunaflux_cuda_streaming_close(lf_streaming_pool *);
int32_t lunaflux_cuda_streaming_enqueue(
  lf_streaming_pool *, int32_t, int64_t, int64_t, int64_t, int32_t);
int32_t lunaflux_cuda_streaming_record(lf_streaming_pool *, int32_t);
int32_t lunaflux_cuda_streaming_poll(lf_streaming_pool *, int32_t);
int32_t lunaflux_cuda_streaming_drain(lf_streaming_pool *, int32_t);
int32_t lunaflux_cuda_streaming_reset(lf_streaming_pool *, int32_t);
int32_t lunaflux_cuda_streaming_host_copy(
  lf_streaming_pool *, uint8_t *, int64_t, int32_t, int32_t, int32_t);
int32_t lunaflux_cuda_test_streaming(int32_t);
int32_t lunaflux_cuda_separate_host_memory(lf_context *, int32_t);

#endif
