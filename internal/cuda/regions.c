#include "resource_internal.h"

#include <stdint.h>

int32_t lf_allocation_region_address(
  lf_allocation *allocation,
  int64_t offset,
  size_t byte_count,
  size_t alignment,
  int allow_empty,
  CUdeviceptr *address
) {
  if (allocation == NULL ||
      atomic_load(&allocation->state) != LF_RESOURCE_LIVE) return LF_CLOSED;
  if (offset < 0 || (byte_count == 0 && allow_empty == 0) || alignment == 0 ||
      alignment > LF_MAX_REGION_ALIGNMENT ||
      (alignment & (alignment - 1)) != 0) {
    return LF_INVALID_ARGUMENT;
  }
  if ((uint64_t)offset > SIZE_MAX) return LF_SIZE_OVERFLOW;
  size_t start = (size_t)offset;
  if (start > allocation->size || byte_count > allocation->size - start) {
    return LF_INVALID_ARGUMENT;
  }
  if ((uint64_t)start > UINT64_MAX - allocation->handle) {
    return LF_SIZE_OVERFLOW;
  }
  CUdeviceptr resolved = allocation->handle + start;
  if (resolved % (uint64_t)alignment != 0) return LF_INVALID_ARGUMENT;
  *address = resolved;
  return LF_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_context_validate_allocation_region(
  lf_context *context,
  lf_allocation *allocation,
  int64_t offset,
  int64_t byte_count,
  int64_t alignment
) {
  if (context == NULL || allocation == NULL) return LF_CLOSED;
  int32_t status = lf_operation_begin(
    &context->state,
    &context->active_operations
  );
  if (status != LF_OK) return status;
  int allocation_acquired = 0;
  status = lf_operation_begin(
    &allocation->state,
    &allocation->active_operations
  );
  if (status == LF_OK) allocation_acquired = 1;
  CUdeviceptr address = 0;
  if (status == LF_OK) {
    if (allocation->context != context) {
      status = LF_INVALID_ARGUMENT;
    } else if (byte_count <= 0 || alignment <= 0) {
      status = LF_INVALID_ARGUMENT;
    } else if ((uint64_t)byte_count > SIZE_MAX ||
               (uint64_t)alignment > SIZE_MAX) {
      status = LF_SIZE_OVERFLOW;
    } else {
      status = lf_allocation_region_address(
        allocation,
        offset,
        (size_t)byte_count,
        (size_t)alignment,
        0,
        &address
      );
    }
  }
  if (allocation_acquired != 0) {
    lf_operation_end(&allocation->active_operations);
  }
  lf_operation_end(&context->active_operations);
  return status;
}
