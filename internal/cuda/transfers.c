#include "resource_internal.h"

#include <limits.h>

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_copy_to_device(
  lf_allocation *allocation,
  moonbit_bytes_t source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
) {
  if (allocation == NULL) return LF_CLOSED;
  int32_t acquired = lf_operation_begin(
    &allocation->state,
    &allocation->active_operations
  );
  if (acquired != LF_OK) return acquired;
  if (source_offset < 0 || destination_offset < 0 || byte_count < 0 ||
      (uint64_t)source_offset > SIZE_MAX ||
      (uint64_t)destination_offset > SIZE_MAX ||
      (uint64_t)byte_count > SIZE_MAX) {
    lf_operation_end(&allocation->active_operations);
    return LF_SIZE_OVERFLOW;
  }
  size_t source_size = (size_t)Moonbit_array_length(source);
  size_t source_start = (size_t)source_offset;
  size_t destination_start = (size_t)destination_offset;
  size_t count = (size_t)byte_count;
  if (source_start > source_size || count > source_size - source_start ||
      destination_start > allocation->size ||
      count > allocation->size - destination_start) {
    lf_operation_end(&allocation->active_operations);
    return LF_INVALID_ARGUMENT;
  }
  int32_t status = lf_context_current(allocation->context);
  if (status == LF_OK) {
    status = lf_cuda_map_result(allocation->context->api->cuMemcpyHtoD(
      allocation->handle + destination_start,
      source + source_start,
      count
    ));
  }
  lf_operation_end(&allocation->active_operations);
  return status;
}

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_cuda_copy_to_host(
  lf_allocation *allocation,
  int64_t source_offset,
  int64_t byte_count,
  int32_t *status
) {
  if (allocation == NULL) {
    *status = LF_CLOSED;
    return moonbit_make_bytes(0, 0);
  }
  *status = lf_operation_begin(
    &allocation->state,
    &allocation->active_operations
  );
  if (*status != LF_OK) return moonbit_make_bytes(0, 0);
  if (source_offset < 0 || byte_count < 0 || byte_count > INT32_MAX ||
      (uint64_t)source_offset > SIZE_MAX ||
      (uint64_t)byte_count > SIZE_MAX) {
    *status = LF_SIZE_OVERFLOW;
    lf_operation_end(&allocation->active_operations);
    return moonbit_make_bytes(0, 0);
  }
  size_t source_start = (size_t)source_offset;
  size_t count = (size_t)byte_count;
  if (source_start > allocation->size ||
      count > allocation->size - source_start) {
    *status = LF_INVALID_ARGUMENT;
    lf_operation_end(&allocation->active_operations);
    return moonbit_make_bytes(0, 0);
  }
  *status = lf_context_current(allocation->context);
  if (*status != LF_OK) {
    lf_operation_end(&allocation->active_operations);
    return moonbit_make_bytes(0, 0);
  }
  moonbit_bytes_t output = moonbit_make_bytes((int32_t)byte_count, 0);
  *status = lf_cuda_map_result(allocation->context->api->cuMemcpyDtoH(
    output,
    allocation->handle + source_start,
    count
  ));
  lf_operation_end(&allocation->active_operations);
  return output;
}
