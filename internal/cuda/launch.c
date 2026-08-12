#include "resource_internal.h"

#include <limits.h>
#include <stdint.h>

#define LF_MAX_KERNEL_ARGUMENTS 32
#define LF_MAX_GRID_YZ 65535
#define LF_MAX_BLOCK_DIMENSION 1024
#define LF_MAX_THREADS_PER_BLOCK 1024
#define LF_MAX_SHARED_MEMORY_BYTES 98304
#define LF_MAX_ARGUMENT_ALIGNMENT 4096

static int32_t lf_validate_launch_dimensions(
  int32_t grid_x,
  int32_t grid_y,
  int32_t grid_z,
  int32_t block_x,
  int32_t block_y,
  int32_t block_z,
  int32_t shared_memory_bytes
) {
  if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0 ||
      grid_y > LF_MAX_GRID_YZ || grid_z > LF_MAX_GRID_YZ) {
    return LF_INVALID_ARGUMENT;
  }
  if (block_x <= 0 || block_y <= 0 || block_z <= 0 ||
      block_x > LF_MAX_BLOCK_DIMENSION ||
      block_y > LF_MAX_BLOCK_DIMENSION ||
      block_z > LF_MAX_BLOCK_DIMENSION) {
    return LF_INVALID_ARGUMENT;
  }
  uint64_t threads = (uint64_t)block_x * (uint64_t)block_y *
    (uint64_t)block_z;
  if (threads > LF_MAX_THREADS_PER_BLOCK || shared_memory_bytes < 0 ||
      shared_memory_bytes > LF_MAX_SHARED_MEMORY_BYTES) {
    return LF_INVALID_ARGUMENT;
  }
  return LF_OK;
}

static int32_t lf_argument_address(
  lf_allocation *allocation,
  int64_t offset,
  int64_t byte_count,
  int64_t alignment,
  CUdeviceptr *address
) {
  if (allocation == NULL) return LF_CLOSED;
  if (offset < 0 || byte_count <= 0 || alignment <= 0 ||
      alignment > LF_MAX_ARGUMENT_ALIGNMENT ||
      (alignment & (alignment - 1)) != 0) {
    return LF_INVALID_ARGUMENT;
  }
  if ((uint64_t)offset > SIZE_MAX || (uint64_t)byte_count > SIZE_MAX) {
    return LF_SIZE_OVERFLOW;
  }
  size_t start = (size_t)offset;
  size_t length = (size_t)byte_count;
  if (start > allocation->size || length > allocation->size - start) {
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
) {
  int32_t status = lf_validate_launch_dimensions(
    grid_x,
    grid_y,
    grid_z,
    block_x,
    block_y,
    block_z,
    shared_memory_bytes
  );
  if (status != LF_OK) return status;
  int32_t argument_count = Moonbit_array_length(allocations);
  if (argument_count <= 0 || argument_count > LF_MAX_KERNEL_ARGUMENTS ||
      Moonbit_array_length(offsets) != argument_count ||
      Moonbit_array_length(byte_counts) != argument_count ||
      Moonbit_array_length(alignments) != argument_count) {
    return LF_INVALID_ARGUMENT;
  }
  if (function == NULL || stream == NULL) return LF_CLOSED;

  status = lf_operation_begin(&function->state, &function->active_operations);
  if (status != LF_OK) return status;
  int module_acquired = 0;
  int stream_acquired = 0;
  int32_t allocation_acquired = 0;
  lf_module *module = function->module;
  if (module == NULL) {
    status = LF_CLOSED;
    goto cleanup;
  }
  status = lf_operation_begin(&module->state, &module->active_operations);
  if (status == LF_OK) module_acquired = 1;
  if (status == LF_OK) {
    status = lf_operation_begin(&stream->state, &stream->active_operations);
    if (status == LF_OK) stream_acquired = 1;
  }
  for (int32_t index = 0; status == LF_OK && index < argument_count;
       index += 1) {
    lf_allocation *allocation = allocations[index];
    if (allocation == NULL) {
      status = LF_CLOSED;
      break;
    }
    status = lf_operation_begin(
      &allocation->state,
      &allocation->active_operations
    );
    if (status == LF_OK) allocation_acquired += 1;
  }
  if (status != LF_OK) goto cleanup;
  if (function->handle == NULL || module->handle == NULL ||
      stream->handle == NULL) {
    status = LF_CLOSED;
    goto cleanup;
  }
  lf_context *context = module->context;
  if (context == NULL || stream->context != context) {
    status = LF_INVALID_ARGUMENT;
    goto cleanup;
  }

  CUdeviceptr argument_values[LF_MAX_KERNEL_ARGUMENTS];
  void *kernel_parameters[LF_MAX_KERNEL_ARGUMENTS];
  for (int32_t index = 0; index < argument_count; index += 1) {
    lf_allocation *allocation = allocations[index];
    if (allocation->context != context) {
      status = LF_INVALID_ARGUMENT;
      goto cleanup;
    }
    status = lf_argument_address(
      allocation,
      offsets[index],
      byte_counts[index],
      alignments[index],
      &argument_values[index]
    );
    if (status != LF_OK) goto cleanup;
    kernel_parameters[index] = &argument_values[index];
  }
  status = lf_context_current(context);
  if (status != LF_OK) goto cleanup;
  status = lf_cuda_map_result(context->api->cuLaunchKernel(
    function->handle,
    (uint32_t)grid_x,
    (uint32_t)grid_y,
    (uint32_t)grid_z,
    (uint32_t)block_x,
    (uint32_t)block_y,
    (uint32_t)block_z,
    (uint32_t)shared_memory_bytes,
    (CUstream)stream->handle,
    kernel_parameters,
    NULL
  ));
  if (status == LF_OK) {
    status = lf_cuda_map_result(
      context->api->cuStreamSynchronize((CUstream)stream->handle)
    );
  }

cleanup:
  for (int32_t index = allocation_acquired; index > 0; index -= 1) {
    lf_operation_end(&allocations[index - 1]->active_operations);
  }
  if (stream_acquired != 0) lf_operation_end(&stream->active_operations);
  if (module_acquired != 0) lf_operation_end(&module->active_operations);
  lf_operation_end(&function->active_operations);
  return status;
}
