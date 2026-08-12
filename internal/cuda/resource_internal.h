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
  atomic_int children;
} lf_child;

typedef struct lf_allocation {
  lf_context *context;
  CUdeviceptr handle;
  size_t size;
  atomic_int state;
} lf_allocation;

typedef struct lf_gemm_plan {
  lf_child *cublas;
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t right_layout;
  cublasLtMatrixLayout_t left_layout;
  cublasLtMatrixLayout_t output_layout;
  size_t left_size;
  size_t right_size;
  size_t output_size;
  size_t workspace_size;
  atomic_int state;
} lf_gemm_plan;

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

int32_t lunaflux_cuda_context_close(lf_context *context);
int32_t lunaflux_cuda_stream_close(lf_child *stream);
int32_t lunaflux_cuda_cublas_close(lf_child *cublas);
int32_t lunaflux_cuda_allocation_close(lf_allocation *allocation);
int32_t lunaflux_cuda_copy_to_device(
  lf_allocation *allocation,
  moonbit_bytes_t source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
);
moonbit_bytes_t lunaflux_cuda_copy_to_host(
  lf_allocation *allocation,
  int64_t source_offset,
  int64_t byte_count,
  int32_t *status
);
lf_gemm_plan *lunaflux_cuda_bf16_gemm_plan_create(
  lf_child *cublas,
  int64_t rows,
  int64_t inner,
  int64_t columns,
  int64_t workspace_byte_count,
  int32_t *status
);
int32_t lunaflux_cuda_bf16_gemm_plan_close(lf_gemm_plan *plan);
int32_t lunaflux_cuda_bf16_gemm_plan_run(
  lf_gemm_plan *plan,
  lf_allocation *left,
  int64_t left_offset,
  lf_allocation *right,
  int64_t right_offset,
  lf_allocation *output,
  int64_t output_offset,
  lf_allocation *workspace,
  int64_t workspace_offset,
  lf_child *stream
);

#endif
