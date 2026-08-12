#include "resource_internal.h"

#include <limits.h>
#include <stdint.h>

static int32_t lf_copy_to_device_range(
  lf_allocation *allocation,
  const uint8_t *source,
  size_t source_size,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
) {
  if (source == NULL) return LF_INVALID_ARGUMENT;
  if (source_offset < 0 || destination_offset < 0 || byte_count < 0 ||
      (uint64_t)source_offset > SIZE_MAX ||
      (uint64_t)destination_offset > SIZE_MAX ||
      (uint64_t)byte_count > SIZE_MAX) return LF_SIZE_OVERFLOW;
  size_t source_start = (size_t)source_offset;
  size_t count = (size_t)byte_count;
  if (source_start > source_size || count > source_size - source_start) {
    return LF_INVALID_ARGUMENT;
  }
  if ((uintptr_t)source_start > UINTPTR_MAX - (uintptr_t)source) {
    return LF_SIZE_OVERFLOW;
  }
  CUdeviceptr destination = 0;
  int32_t status = lf_allocation_region_address(
    allocation,
    destination_offset,
    count,
    1,
    1,
    &destination
  );
  if (status == LF_OK) status = lf_context_current(allocation->context);
  if (status == LF_OK) {
    status = lf_cuda_map_result(allocation->context->api->cuMemcpyHtoD(
      destination,
      source + source_start,
      count
    ));
  }
  return status;
}

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
  size_t source_size = source == NULL
    ? 0
    : (size_t)Moonbit_array_length(source);
  int32_t status = lf_copy_to_device_range(
    allocation,
    source,
    source_size,
    source_offset,
    destination_offset,
    byte_count
  );
  lf_operation_end(&allocation->active_operations);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_context_copy_fixed_to_device(
  lf_context *context,
  lf_allocation *allocation,
  uint8_t *source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
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
  if (status == LF_OK && allocation->context != context) {
    status = LF_INVALID_ARGUMENT;
  }
  if (status == LF_OK) {
    size_t source_size = source == NULL
      ? 0
      : (size_t)Moonbit_array_length(source);
    status = lf_copy_to_device_range(
      allocation,
      source,
      source_size,
      source_offset,
      destination_offset,
      byte_count
    );
  }
  if (allocation_acquired != 0) {
    lf_operation_end(&allocation->active_operations);
  }
  lf_operation_end(&context->active_operations);
  return status;
}

static int32_t lf_copy_from_device_range(
  lf_allocation *allocation,
  uint8_t *destination,
  size_t destination_size,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
) {
  if (destination == NULL) return LF_INVALID_ARGUMENT;
  if (source_offset < 0 || destination_offset < 0 || byte_count < 0 ||
      (uint64_t)source_offset > SIZE_MAX ||
      (uint64_t)destination_offset > SIZE_MAX ||
      (uint64_t)byte_count > SIZE_MAX) return LF_SIZE_OVERFLOW;
  size_t destination_start = (size_t)destination_offset;
  size_t count = (size_t)byte_count;
  if (destination_start > destination_size ||
      count > destination_size - destination_start) {
    return LF_INVALID_ARGUMENT;
  }
  if ((uintptr_t)destination_start > UINTPTR_MAX - (uintptr_t)destination) {
    return LF_SIZE_OVERFLOW;
  }
  CUdeviceptr source = 0;
  int32_t status = lf_allocation_region_address(
    allocation,
    source_offset,
    count,
    1,
    1,
    &source
  );
  if (status == LF_OK) status = lf_context_current(allocation->context);
  if (status == LF_OK) {
    status = lf_cuda_map_result(allocation->context->api->cuMemcpyDtoH(
      destination + destination_start,
      source,
      count
    ));
  }
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_context_copy_device_to_fixed(
  lf_context *context,
  lf_allocation *allocation,
  uint8_t *destination,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
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
  if (status == LF_OK && allocation->context != context) {
    status = LF_INVALID_ARGUMENT;
  }
  if (status == LF_OK) {
    size_t destination_size = destination == NULL
      ? 0
      : (size_t)Moonbit_array_length(destination);
    status = lf_copy_from_device_range(
      allocation,
      destination,
      destination_size,
      source_offset,
      destination_offset,
      byte_count
    );
  }
  if (allocation_acquired != 0) {
    lf_operation_end(&allocation->active_operations);
  }
  lf_operation_end(&context->active_operations);
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
