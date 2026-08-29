#ifndef LUNAFLUX_CUDA_RESOURCE_INTERNAL_H
#define LUNAFLUX_CUDA_RESOURCE_INTERNAL_H

#include "cuda_abi.h"

#include <stdatomic.h>

#define LF_MAX_REGION_ALIGNMENT 4096U

typedef struct lf_context {
  lf_cuda_api *api;
  CUcontext handle;
  atomic_int state;
  atomic_int active_operations;
  atomic_int children;
} lf_context;

typedef struct lf_child {
  lf_context *context;
  void *handle;
  atomic_int state;
  atomic_int active_operations;
  atomic_int children;
} lf_child;

typedef struct lf_allocation {
  lf_context *context;
  CUdeviceptr handle;
  size_t size;
  atomic_int state;
  atomic_int active_operations;
} lf_allocation;

typedef struct lf_allocation_lease {
  lf_allocation *allocation;
  atomic_int state;
} lf_allocation_lease;

typedef struct lf_module {
  lf_context *context;
  CUmodule handle;
  atomic_int state;
  atomic_int active_operations;
  atomic_int children;
} lf_module;

typedef struct lf_function {
  lf_module *module;
  CUfunction handle;
  atomic_int state;
  atomic_int active_operations;
} lf_function;

typedef struct lf_ordered_executor lf_ordered_executor;

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
  atomic_int active_operations;
} lf_gemm_plan;

enum lf_resource_state {
  LF_RESOURCE_LIVE = 0,
  LF_RESOURCE_CLOSING = 1,
  LF_RESOURCE_CLOSED = 2
};

int32_t lf_context_current(lf_context *context);
int32_t lf_allocation_region_address(
  lf_allocation *allocation,
  int64_t offset,
  size_t byte_count,
  size_t alignment,
  int allow_empty,
  CUdeviceptr *address
);
void lf_release_context_child(lf_context *context);
int32_t lf_operation_begin(atomic_int *state, atomic_int *active_operations);
void lf_operation_end(atomic_int *active_operations);
int32_t lf_begin_close(atomic_int *state, atomic_int *active_operations);
void lf_close_failed(atomic_int *state);
void lf_close_succeeded(atomic_int *state);
void lf_finalize_failure(void);

int32_t lunaflux_cuda_context_close(lf_context *context);
int32_t lunaflux_cuda_stream_close(lf_child *stream);
int32_t lunaflux_cuda_cublas_close(lf_child *cublas);
int32_t lunaflux_cuda_allocation_close(lf_allocation *allocation);
lf_allocation_lease *lunaflux_cuda_allocation_lease_create(
  lf_context *context,
  lf_allocation *allocation,
  int32_t *status
);
int32_t lunaflux_cuda_allocation_lease_close(lf_allocation_lease *lease);
int32_t lunaflux_cuda_context_validate_allocation_region(
  lf_context *context,
  lf_allocation *allocation,
  int64_t offset,
  int64_t byte_count,
  int64_t alignment
);
int32_t lunaflux_cuda_module_close(lf_module *module);
int32_t lunaflux_cuda_function_close(lf_function *function);
int32_t lunaflux_cuda_function_launch(
  lf_function *function,
  lf_child *stream,
  int32_t grid_x,
  int32_t grid_y,
  int32_t grid_z,
  int32_t block_x,
  int32_t block_y,
  int32_t block_z,
  int32_t shared_memory_bytes,
  lf_allocation **allocations,
  int64_t *offsets,
  int64_t *byte_counts,
  int64_t *alignments
);
lf_ordered_executor *lunaflux_cuda_ordered_executor_create(
  lf_context *context,
  lf_child *stream,
  lf_function **functions,
  int32_t *dimensions,
  int32_t *argument_starts,
  lf_allocation **allocations,
  int64_t *offsets,
  int64_t *byte_counts,
  int64_t *alignments,
  int32_t policy,
  int32_t *status
);
int32_t lunaflux_cuda_ordered_executor_enqueue(
  lf_ordered_executor *executor,
  int32_t index
);
int32_t lunaflux_cuda_ordered_executor_launch_captured(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_mode(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_record(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_poll(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_wait(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_abort(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_reset(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_ordered_executor_close(
  lf_ordered_executor *executor
);
int32_t lunaflux_cuda_copy_to_device(
  lf_allocation *allocation,
  moonbit_bytes_t source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
);
int32_t lunaflux_cuda_context_copy_fixed_to_device(
  lf_context *context,
  lf_allocation *allocation,
  uint8_t *source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
);
int32_t lunaflux_cuda_context_copy_device_to_fixed(
  lf_context *context,
  lf_allocation *allocation,
  uint8_t *destination,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
);
int32_t lunaflux_cuda_test_fixed_transfer_boundary(
  uint8_t *source,
  int32_t cycles
);
int32_t lunaflux_cuda_test_fixed_readback_boundary(
  uint8_t *destination,
  int32_t cycles
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
