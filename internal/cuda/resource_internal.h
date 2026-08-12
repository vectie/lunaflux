#ifndef LUNAFLUX_CUDA_RESOURCE_INTERNAL_H
#define LUNAFLUX_CUDA_RESOURCE_INTERNAL_H

#include "cuda_abi.h"

#include <stdatomic.h>

typedef struct lf_context {
  lf_cuda_api *api;
  CUcontext handle;
  atomic_int state;
  atomic_int children;
} lf_context;

typedef struct lf_child {
  lf_context *context;
  void *handle;
  atomic_int state;
} lf_child;

typedef struct lf_allocation {
  lf_context *context;
  CUdeviceptr handle;
  size_t size;
  atomic_int state;
} lf_allocation;

enum lf_resource_state {
  LF_RESOURCE_LIVE = 0,
  LF_RESOURCE_CLOSING = 1,
  LF_RESOURCE_CLOSED = 2
};

int32_t lf_context_current(lf_context *context);
void lf_release_context_child(lf_context *context);
int32_t lf_begin_close(atomic_int *state);
void lf_close_failed(atomic_int *state);
void lf_close_succeeded(atomic_int *state);
void lf_finalize_failure(void);

#endif
