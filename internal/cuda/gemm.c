#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_COMPUTE_32F 68
#define CUDA_R_32F 0
#define CUDA_R_16BF 14
#define CUBLASLT_MATMUL_DESC_TRANSA 3
#define CUBLAS_OP_T 1
#define LF_BF16_BYTES 2U
#define LF_GEMM_ALIGNMENT 16U
#define LF_WORKSPACE_ALIGNMENT 256U

static int32_t lf_matrix_bytes(
  int64_t rows,
  int64_t columns,
  size_t *byte_count
) {
  if (rows <= 0 || columns <= 0) return LF_INVALID_ARGUMENT;
  uint64_t unsigned_rows = (uint64_t)rows;
  uint64_t unsigned_columns = (uint64_t)columns;
  if (unsigned_rows > SIZE_MAX / unsigned_columns) return LF_SIZE_OVERFLOW;
  size_t elements = (size_t)(unsigned_rows * unsigned_columns);
  if (elements > SIZE_MAX / LF_BF16_BYTES) return LF_SIZE_OVERFLOW;
  *byte_count = elements * LF_BF16_BYTES;
  return LF_OK;
}

static int32_t lf_destroy_layout(
  lf_cuda_api *api,
  cublasLtMatrixLayout_t *layout
) {
  if (*layout == NULL) return LF_OK;
  if (api->cublasLtMatrixLayoutDestroy(*layout) != CUBLAS_STATUS_SUCCESS) {
    return LF_DRIVER_FAILURE;
  }
  *layout = NULL;
  return LF_OK;
}

static int32_t lf_destroy_operation(
  lf_cuda_api *api,
  cublasLtMatmulDesc_t *operation
) {
  if (*operation == NULL) return LF_OK;
  if (api->cublasLtMatmulDescDestroy(*operation) != CUBLAS_STATUS_SUCCESS) {
    return LF_DRIVER_FAILURE;
  }
  *operation = NULL;
  return LF_OK;
}

static int32_t lf_close_gemm_plan(lf_gemm_plan *plan) {
  if (plan == NULL) return LF_INVALID_ARGUMENT;
  int32_t begin = lf_begin_close(&plan->state, &plan->active_operations);
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  int32_t result = lf_context_current(plan->cublas->context);
  if (result != LF_OK) {
    lf_close_failed(&plan->state);
    return result;
  }
  lf_cuda_api *api = plan->cublas->context->api;
  if (lf_destroy_layout(api, &plan->output_layout) != LF_OK) {
    result = LF_DRIVER_FAILURE;
  }
  if (lf_destroy_layout(api, &plan->left_layout) != LF_OK) {
    result = LF_DRIVER_FAILURE;
  }
  if (lf_destroy_layout(api, &plan->right_layout) != LF_OK) {
    result = LF_DRIVER_FAILURE;
  }
  if (lf_destroy_operation(api, &plan->operation) != LF_OK) {
    result = LF_DRIVER_FAILURE;
  }
  if (result != LF_OK) {
    lf_close_failed(&plan->state);
    return result;
  }
  atomic_fetch_sub(&plan->cublas->children, 1);
  moonbit_decref(plan->cublas);
  plan->cublas = NULL;
  plan->left_size = 0;
  plan->right_size = 0;
  plan->output_size = 0;
  plan->workspace_size = 0;
  lf_close_succeeded(&plan->state);
  return LF_OK;
}

static void lf_finalize_gemm_plan(void *object) {
  if (lf_close_gemm_plan((lf_gemm_plan *)object) != LF_OK) {
    lf_finalize_failure();
  }
}

static int32_t lf_finish_failed_create(
  lf_gemm_plan *plan,
  int32_t create_status
) {
  int32_t close_status = lf_close_gemm_plan(plan);
  return close_status == LF_OK ? create_status : close_status;
}

MOONBIT_FFI_EXPORT
lf_gemm_plan *lunaflux_cuda_bf16_gemm_plan_create(
  lf_child *cublas,
  int64_t rows,
  int64_t inner,
  int64_t columns,
  int64_t workspace_byte_count,
  int32_t *status
) {
  lf_gemm_plan *plan = (lf_gemm_plan *)moonbit_make_external_object(
    lf_finalize_gemm_plan,
    sizeof(lf_gemm_plan)
  );
  memset(plan, 0, sizeof(*plan));
  atomic_init(&plan->state, LF_RESOURCE_CLOSED);
  atomic_init(&plan->active_operations, 0);
  if (cublas == NULL ||
      atomic_load(&cublas->state) != LF_RESOURCE_LIVE) {
    *status = LF_CLOSED;
    return plan;
  }
  if (workspace_byte_count < 0 ||
      (uint64_t)workspace_byte_count > SIZE_MAX) {
    *status = workspace_byte_count < 0
      ? LF_INVALID_ARGUMENT
      : LF_SIZE_OVERFLOW;
    return plan;
  }
  *status = lf_matrix_bytes(rows, inner, &plan->left_size);
  if (*status != LF_OK) return plan;
  *status = lf_matrix_bytes(inner, columns, &plan->right_size);
  if (*status != LF_OK) return plan;
  *status = lf_matrix_bytes(rows, columns, &plan->output_size);
  if (*status != LF_OK) return plan;
  *status = lf_operation_begin(&cublas->state, &cublas->active_operations);
  if (*status != LF_OK) return plan;
  *status = lf_context_current(cublas->context);
  if (*status != LF_OK) {
    lf_operation_end(&cublas->active_operations);
    return plan;
  }

  plan->workspace_size = (size_t)workspace_byte_count;
  moonbit_incref(cublas);
  atomic_fetch_add(&cublas->children, 1);
  plan->cublas = cublas;
  atomic_store(&plan->state, LF_RESOURCE_LIVE);
  lf_cuda_api *api = cublas->context->api;
  if (api->cublasLtMatmulDescCreate(
        &plan->operation,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F
      ) != CUBLAS_STATUS_SUCCESS) {
    *status = lf_finish_failed_create(plan, LF_DRIVER_FAILURE);
    lf_operation_end(&cublas->active_operations);
    return plan;
  }
  /* Safetensors stores a projection weight as row-major [columns, inner].
   * That byte sequence is the column-major view W^T [inner, columns]. Set
   * transpose on the first cuBLASLt operand so the column-major operation is
   * C^T = W * A^T, exactly the transpose of row-major C = A * W^T. */
  const int32_t transpose = CUBLAS_OP_T;
  if (api->cublasLtMatmulDescSetAttribute(
        plan->operation,
        CUBLASLT_MATMUL_DESC_TRANSA,
        &transpose,
        sizeof(transpose)
      ) != CUBLAS_STATUS_SUCCESS) {
    *status = lf_finish_failed_create(plan, LF_DRIVER_FAILURE);
    lf_operation_end(&cublas->active_operations);
    return plan;
  }
  if (api->cublasLtMatrixLayoutCreate(
        &plan->right_layout,
        CUDA_R_16BF,
        (uint64_t)inner,
        (uint64_t)columns,
        inner
      ) != CUBLAS_STATUS_SUCCESS ||
      api->cublasLtMatrixLayoutCreate(
        &plan->left_layout,
        CUDA_R_16BF,
        (uint64_t)inner,
        (uint64_t)rows,
        inner
      ) != CUBLAS_STATUS_SUCCESS ||
      api->cublasLtMatrixLayoutCreate(
        &plan->output_layout,
        CUDA_R_16BF,
        (uint64_t)columns,
        (uint64_t)rows,
        columns
      ) != CUBLAS_STATUS_SUCCESS) {
    *status = lf_finish_failed_create(plan, LF_DRIVER_FAILURE);
    lf_operation_end(&cublas->active_operations);
    return plan;
  }
  *status = LF_OK;
  lf_operation_end(&cublas->active_operations);
  return plan;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_bf16_gemm_plan_close(lf_gemm_plan *plan) {
  return lf_close_gemm_plan(plan);
}

static int lf_ranges_overlap(
  CUdeviceptr first,
  size_t first_size,
  CUdeviceptr second,
  size_t second_size
) {
  if (first_size > UINT64_MAX - first || second_size > UINT64_MAX - second) {
    return 1;
  }
  return first < second + second_size && second < first + first_size;
}

MOONBIT_FFI_EXPORT
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
) {
  if (plan == NULL || left == NULL || right == NULL || output == NULL ||
      workspace == NULL || stream == NULL) return LF_CLOSED;
  int32_t status = lf_operation_begin(&plan->state, &plan->active_operations);
  if (status != LF_OK) return status;
  int left_acquired = 0;
  int right_acquired = 0;
  int output_acquired = 0;
  int workspace_acquired = 0;
  int stream_acquired = 0;
  status = lf_operation_begin(&left->state, &left->active_operations);
  if (status == LF_OK) left_acquired = 1;
  if (status == LF_OK) {
    status = lf_operation_begin(&right->state, &right->active_operations);
    if (status == LF_OK) right_acquired = 1;
  }
  if (status == LF_OK) {
    status = lf_operation_begin(&output->state, &output->active_operations);
    if (status == LF_OK) output_acquired = 1;
  }
  if (status == LF_OK) {
    status = lf_operation_begin(
      &workspace->state,
      &workspace->active_operations
    );
    if (status == LF_OK) workspace_acquired = 1;
  }
  if (status == LF_OK) {
    status = lf_operation_begin(&stream->state, &stream->active_operations);
    if (status == LF_OK) stream_acquired = 1;
  }
  if (status != LF_OK) goto cleanup;
  if (plan->operation == NULL || plan->right_layout == NULL ||
      plan->left_layout == NULL || plan->output_layout == NULL) {
    status = LF_CLOSED;
    goto cleanup;
  }
  lf_context *context = plan->cublas->context;
  if (left->context != context || right->context != context ||
      output->context != context || workspace->context != context ||
      stream->context != context) {
    status = LF_INVALID_ARGUMENT;
    goto cleanup;
  }
  CUdeviceptr left_address = 0;
  CUdeviceptr right_address = 0;
  CUdeviceptr output_address = 0;
  CUdeviceptr workspace_address = 0;
  CUdeviceptr resolved_workspace_address = 0;
  status = lf_allocation_region_address(
    left,
    left_offset,
    plan->left_size,
    LF_GEMM_ALIGNMENT,
    0,
    &left_address
  );
  if (status != LF_OK) goto cleanup;
  status = lf_allocation_region_address(
    right,
    right_offset,
    plan->right_size,
    LF_GEMM_ALIGNMENT,
    0,
    &right_address
  );
  if (status != LF_OK) goto cleanup;
  status = lf_allocation_region_address(
    output,
    output_offset,
    plan->output_size,
    LF_GEMM_ALIGNMENT,
    0,
    &output_address
  );
  if (status != LF_OK) goto cleanup;
  status = lf_allocation_region_address(
    workspace,
    workspace_offset,
    plan->workspace_size,
    plan->workspace_size > 0 ? LF_WORKSPACE_ALIGNMENT : 1U,
    1,
    &resolved_workspace_address
  );
  if (status != LF_OK) goto cleanup;
  if (plan->workspace_size > 0) {
    workspace_address = resolved_workspace_address;
  }
  if (lf_ranges_overlap(
        output_address,
        plan->output_size,
        left_address,
        plan->left_size
      ) ||
      lf_ranges_overlap(
        output_address,
        plan->output_size,
        right_address,
        plan->right_size
      ) ||
      (plan->workspace_size > 0 && lf_ranges_overlap(
        workspace_address,
        plan->workspace_size,
        left_address,
        plan->left_size
      )) ||
      (plan->workspace_size > 0 && lf_ranges_overlap(
        workspace_address,
        plan->workspace_size,
        right_address,
        plan->right_size
      )) ||
      (plan->workspace_size > 0 && lf_ranges_overlap(
        workspace_address,
        plan->workspace_size,
        output_address,
        plan->output_size
      ))) {
    status = LF_INVALID_ARGUMENT;
    goto cleanup;
  }
  status = lf_context_current(context);
  if (status != LF_OK) goto cleanup;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  int32_t cublas_status = context->api->cublasLtMatmul(
    plan->cublas->handle,
    plan->operation,
    &alpha,
    (const void *)(uintptr_t)right_address,
    plan->right_layout,
    (const void *)(uintptr_t)left_address,
    plan->left_layout,
    &beta,
    (const void *)(uintptr_t)output_address,
    plan->output_layout,
    (void *)(uintptr_t)output_address,
    plan->output_layout,
    NULL,
    (void *)(uintptr_t)workspace_address,
    plan->workspace_size,
    (CUstream)stream->handle
  );
  status = cublas_status == CUBLAS_STATUS_SUCCESS
    ? lf_cuda_map_result(
        context->api->cuStreamSynchronize((CUstream)stream->handle)
      )
    : LF_DRIVER_FAILURE;

cleanup:
  if (stream_acquired != 0) lf_operation_end(&stream->active_operations);
  if (workspace_acquired != 0) {
    lf_operation_end(&workspace->active_operations);
  }
  if (output_acquired != 0) lf_operation_end(&output->active_operations);
  if (right_acquired != 0) lf_operation_end(&right->active_operations);
  if (left_acquired != 0) lf_operation_end(&left->active_operations);
  lf_operation_end(&plan->active_operations);
  return status;
}
