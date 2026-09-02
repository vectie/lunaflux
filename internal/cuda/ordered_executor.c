#include "ordered_executor_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define LF_DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES 49152

static int32_t lf_prepare_dynamic_shared_memory(
  lf_ordered_executor *executor,
  lf_ordered_kernel *kernel
) {
  int32_t shared = kernel->dimensions[6];
  if (shared <= LF_DEFAULT_DYNAMIC_SHARED_MEMORY_BYTES) return LF_OK;
  if (executor->context->api->cuFuncSetAttribute == NULL) {
    return LF_UNSUPPORTED;
  }
  return lf_cuda_map_result(executor->context->api->cuFuncSetAttribute(
    kernel->function->handle,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    shared
  ));
}

int32_t lf_ordered_operation_begin(lf_ordered_executor *executor) {
  int32_t result = lf_operation_begin(
    &executor->state, &executor->active_operations
  );
  if (result != LF_OK) return result;
  int expected = 0;
  if (!atomic_compare_exchange_strong(
        &executor->operation_gate, &expected, 1)) {
    lf_operation_end(&executor->active_operations);
    return LF_BUSY;
  }
  return LF_OK;
}

void lf_ordered_operation_end(lf_ordered_executor *executor) {
  atomic_store(&executor->operation_gate, 0);
  lf_operation_end(&executor->active_operations);
}

static int32_t lf_validate_dimensions(const int32_t *dimensions) {
  int32_t grid_x = dimensions[0];
  int32_t grid_y = dimensions[1];
  int32_t grid_z = dimensions[2];
  int32_t block_x = dimensions[3];
  int32_t block_y = dimensions[4];
  int32_t block_z = dimensions[5];
  int32_t shared = dimensions[6];
  if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0 ||
      grid_y > 65535 || grid_z > 65535 || block_x <= 0 ||
      block_y <= 0 || block_z <= 0 || block_x > 1024 ||
      block_y > 1024 || block_z > 1024 || shared < 0 || shared > 98304) {
    return LF_INVALID_ARGUMENT;
  }
  uint64_t threads = (uint64_t)block_x * (uint64_t)block_y *
    (uint64_t)block_z;
  return threads <= 1024 ? LF_OK : LF_INVALID_ARGUMENT;
}

static void lf_release_ordered_kernel(lf_ordered_kernel *kernel) {
  if (kernel == NULL) return;
  for (int32_t index = kernel->argument_count; index > 0; index -= 1) {
    lf_allocation *allocation = kernel->allocations[index - 1];
    lf_operation_end(&allocation->active_operations);
    moonbit_decref(allocation);
  }
  if (kernel->function != NULL) {
    lf_operation_end(&kernel->function->active_operations);
    moonbit_decref(kernel->function);
  }
  free(kernel->kernel_parameters);
  free(kernel->argument_values);
  free(kernel->allocations);
  memset(kernel, 0, sizeof(*kernel));
}

static void lf_release_ordered_executor(lf_ordered_executor *executor) {
  for (int32_t index = executor->acquired_kernel_count; index > 0; index -= 1) {
    lf_release_ordered_kernel(&executor->kernels[index - 1]);
  }
  free(executor->kernels);
  executor->kernels = NULL;
  executor->acquired_kernel_count = 0;
  if (executor->stream != NULL) {
    lf_operation_end(&executor->stream->active_operations);
    moonbit_decref(executor->stream);
    executor->stream = NULL;
  }
  executor->context = NULL;
}

static int32_t lf_close_ordered_executor(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_INVALID_ARGUMENT;
  if (atomic_load(&executor->state) == LF_RESOURCE_CLOSED) return LF_OK;
  int32_t phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_ENQUEUED || phase == LF_ORDERED_RECORDED) {
    return LF_BUSY;
  }
  int32_t begin = lf_begin_close(
    &executor->state, &executor->active_operations
  );
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_ENQUEUED || phase == LF_ORDERED_RECORDED) {
    lf_close_failed(&executor->state);
    return LF_BUSY;
  }
  int32_t result = lf_context_current(executor->context);
  if (result == LF_OK && executor->event != NULL) {
    int32_t destroy = lf_cuda_map_result(
      executor->context->api->cuEventDestroy(executor->event)
    );
    if (destroy == LF_OK) {
      executor->event = NULL;
    } else {
      result = destroy;
    }
  }
  int32_t graph_result = lf_ordered_graph_destroy(executor);
  if (result == LF_OK && graph_result != LF_OK) {
    result = graph_result;
  }
  if (result != LF_OK) {
    lf_close_failed(&executor->state);
    return result;
  }
  lf_release_ordered_executor(executor);
  lf_close_succeeded(&executor->state);
  return LF_OK;
}

static void lf_finalize_ordered_executor(void *object) {
  lf_ordered_executor *executor = (lf_ordered_executor *)object;
  int32_t phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_ENQUEUED || phase == LF_ORDERED_RECORDED) {
    if (lf_context_current(executor->context) != LF_OK ||
        lf_cuda_map_result(executor->context->api->cuStreamSynchronize(
          (CUstream)executor->stream->handle
        )) != LF_OK) {
      lf_finalize_failure();
      return;
    }
    atomic_store(&executor->phase, LF_ORDERED_COMPLETE);
  }
  if (lf_close_ordered_executor(executor) != LF_OK) lf_finalize_failure();
}

static int32_t lf_prepare_ordered_kernel(
  lf_ordered_executor *executor,
  lf_ordered_kernel *kernel,
  lf_function *function,
  const int32_t *dimensions,
  lf_allocation **allocations,
  int64_t *offsets,
  int64_t *byte_counts,
  int64_t *alignments,
  int32_t argument_start,
  int32_t argument_count
) {
  int32_t result = lf_validate_dimensions(dimensions);
  if (result != LF_OK || function == NULL || argument_count <= 0 ||
      argument_count > LF_MAX_KERNEL_ARGUMENTS) return LF_INVALID_ARGUMENT;
  result = lf_operation_begin(&function->state, &function->active_operations);
  if (result != LF_OK) return result;
  moonbit_incref(function);
  kernel->function = function;
  kernel->argument_count = 0;
  memcpy(kernel->dimensions, dimensions, sizeof(kernel->dimensions));
  kernel->allocations = calloc((size_t)argument_count, sizeof(lf_allocation *));
  kernel->argument_values = calloc((size_t)argument_count, sizeof(CUdeviceptr));
  kernel->kernel_parameters = calloc((size_t)argument_count, sizeof(void *));
  if (kernel->allocations == NULL || kernel->argument_values == NULL ||
      kernel->kernel_parameters == NULL) return LF_HOST_ALLOCATION_FAILED;
  lf_module *module = function->module;
  if (module == NULL || module->context != executor->context) {
    return LF_INVALID_ARGUMENT;
  }
  for (int32_t index = 0; index < argument_count; index += 1) {
    int32_t source = argument_start + index;
    lf_allocation *allocation = allocations[source];
    if (allocation == NULL || allocation->context != executor->context ||
        byte_counts[source] <= 0 || alignments[source] <= 0 ||
        (uint64_t)byte_counts[source] > SIZE_MAX ||
        (uint64_t)alignments[source] > SIZE_MAX) return LF_INVALID_ARGUMENT;
    result = lf_operation_begin(
      &allocation->state, &allocation->active_operations
    );
    if (result != LF_OK) return result;
    moonbit_incref(allocation);
    kernel->allocations[index] = allocation;
    kernel->argument_count = index + 1;
    result = lf_allocation_region_address(
      allocation,
      offsets[source],
      (size_t)byte_counts[source],
      (size_t)alignments[source],
      0,
      &kernel->argument_values[index]
    );
    if (result != LF_OK) return result;
    kernel->kernel_parameters[index] = &kernel->argument_values[index];
  }
  kernel->argument_count = argument_count;
  return lf_prepare_dynamic_shared_memory(executor, kernel);
}

MOONBIT_FFI_EXPORT
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
) {
  lf_ordered_executor *executor =
    (lf_ordered_executor *)moonbit_make_external_object(
      lf_finalize_ordered_executor, sizeof(lf_ordered_executor)
  );
  memset(executor, 0, sizeof(*executor));
  atomic_init(&executor->state, LF_RESOURCE_CLOSED);
  atomic_init(&executor->active_operations, 0);
  atomic_init(&executor->operation_gate, 0);
  atomic_init(&executor->next_kernel, 0);
  atomic_init(&executor->phase, LF_ORDERED_IDLE);
  if (context == NULL || stream == NULL || functions == NULL ||
      dimensions == NULL || argument_starts == NULL || allocations == NULL ||
      offsets == NULL || byte_counts == NULL || alignments == NULL) {
    *status = LF_INVALID_ARGUMENT;
    return executor;
  }
  int32_t kernel_count = Moonbit_array_length(functions);
  int32_t argument_count = Moonbit_array_length(allocations);
  if (kernel_count <= 0 ||
      kernel_count > LF_MAX_ORDERED_KERNELS ||
      kernel_count > INT32_MAX / LF_DIMENSIONS_PER_KERNEL ||
      argument_count <= 0 || argument_count > LF_MAX_ORDERED_ARGUMENTS ||
      Moonbit_array_length(dimensions) !=
        kernel_count * LF_DIMENSIONS_PER_KERNEL ||
      Moonbit_array_length(argument_starts) != kernel_count + 1 ||
      Moonbit_array_length(offsets) != argument_count ||
      Moonbit_array_length(byte_counts) != argument_count ||
      Moonbit_array_length(alignments) != argument_count ||
      argument_starts[0] != 0 || argument_starts[kernel_count] != argument_count) {
    *status = LF_INVALID_ARGUMENT;
    return executor;
  }
  *status = lf_operation_begin(&stream->state, &stream->active_operations);
  if (*status != LF_OK) return executor;
  moonbit_incref(stream);
  executor->stream = stream;
  executor->context = context;
  if (stream->context != context) {
    *status = LF_INVALID_ARGUMENT;
    goto failed;
  }
  executor->kernels = calloc((size_t)kernel_count, sizeof(lf_ordered_kernel));
  if (executor->kernels == NULL) {
    *status = LF_HOST_ALLOCATION_FAILED;
    goto failed;
  }
  executor->kernel_count = kernel_count;
  for (int32_t index = 0; index < kernel_count; index += 1) {
    int32_t start = argument_starts[index];
    int32_t end = argument_starts[index + 1];
    if (start < 0 || end <= start || end > argument_count) {
      *status = LF_INVALID_ARGUMENT;
      goto failed;
    }
    executor->acquired_kernel_count = index + 1;
    *status = lf_prepare_ordered_kernel(
      executor,
      &executor->kernels[index],
      functions[index],
      &dimensions[index * LF_DIMENSIONS_PER_KERNEL],
      allocations,
      offsets,
      byte_counts,
      alignments,
      start,
      end - start
    );
    if (*status != LF_OK) goto failed;
  }
  *status = lf_context_current(context);
  CUevent event = NULL;
  if (*status == LF_OK) {
    *status = lf_cuda_map_result(context->api->cuEventCreate(&event, 0));
  }
  if (*status != LF_OK) {
    executor->event = event;
    int32_t create_status = *status;
    atomic_store(&executor->state, LF_RESOURCE_LIVE);
    int32_t cleanup = lf_close_ordered_executor(executor);
    *status = cleanup == LF_OK ? create_status : cleanup;
    return executor;
  }
  executor->event = event;
  *status = lf_ordered_graph_prepare(executor, policy);
  if (*status != LF_OK) {
    int32_t create_status = *status;
    atomic_store(&executor->state, LF_RESOURCE_LIVE);
    int32_t cleanup = lf_close_ordered_executor(executor);
    *status = cleanup == LF_OK ? create_status : cleanup;
    return executor;
  }
  atomic_store(&executor->phase, LF_ORDERED_IDLE);
  atomic_store(&executor->state, LF_RESOURCE_LIVE);
  return executor;

failed:
  lf_release_ordered_executor(executor);
  return executor;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_enqueue(
  lf_ordered_executor *executor,
  int32_t index
) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  int32_t phase = atomic_load(&executor->phase);
  if ((phase != LF_ORDERED_IDLE && phase != LF_ORDERED_ENQUEUED) ||
      executor->execution_mode != LF_ORDERED_MODE_EAGER ||
      index != atomic_load(&executor->next_kernel) || index < 0 ||
      index >= executor->kernel_count) {
    result = LF_INVALID_ARGUMENT;
    goto complete;
  }
  lf_ordered_kernel *kernel = &executor->kernels[index];
  result = lf_context_current(executor->context);
  if (result != LF_OK) goto complete;
  result = lf_cuda_map_result(executor->context->api->cuLaunchKernel(
    kernel->function->handle,
    (uint32_t)kernel->dimensions[0],
    (uint32_t)kernel->dimensions[1],
    (uint32_t)kernel->dimensions[2],
    (uint32_t)kernel->dimensions[3],
    (uint32_t)kernel->dimensions[4],
    (uint32_t)kernel->dimensions[5],
    (uint32_t)kernel->dimensions[6],
    (CUstream)executor->stream->handle,
    kernel->kernel_parameters,
    NULL
  ));
  if (result == LF_OK) {
    atomic_fetch_add(&executor->next_kernel, 1);
    atomic_store(&executor->phase, LF_ORDERED_ENQUEUED);
  }
complete:
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_record(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  if (atomic_load(&executor->phase) != LF_ORDERED_ENQUEUED ||
      atomic_load(&executor->next_kernel) != executor->kernel_count) {
    result = LF_INVALID_ARGUMENT;
    goto complete;
  }
  result = lf_context_current(executor->context);
  if (result == LF_OK) {
    result = lf_cuda_map_result(executor->context->api->cuEventRecord(
      executor->event, (CUstream)executor->stream->handle
    ));
  }
  if (result == LF_OK) {
    atomic_store(&executor->phase, LF_ORDERED_RECORDED);
  }
complete:
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_poll(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  int32_t phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_COMPLETE) {
    result = 1;
    goto complete;
  }
  if (phase != LF_ORDERED_RECORDED) {
    result = LF_INVALID_ARGUMENT;
    goto complete;
  }
  result = lf_context_current(executor->context);
  if (result != LF_OK) goto complete;
  CUresult query = executor->context->api->cuEventQuery(executor->event);
  if (query == CUDA_ERROR_NOT_READY) {
    result = 0;
    goto complete;
  }
  result = lf_cuda_map_result(query);
  if (result == LF_OK) {
    atomic_store(&executor->phase, LF_ORDERED_COMPLETE);
    result = 1;
  }
complete:
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_wait(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  int32_t phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_COMPLETE) {
    result = LF_OK;
    goto complete;
  }
  if (phase != LF_ORDERED_RECORDED) {
    result = LF_INVALID_ARGUMENT;
    goto complete;
  }
  result = lf_context_current(executor->context);
  if (result == LF_OK) {
    result = lf_cuda_map_result(
      executor->context->api->cuEventSynchronize(executor->event)
    );
  }
  if (result == LF_OK) {
    atomic_store(&executor->phase, LF_ORDERED_COMPLETE);
  }
complete:
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_abort(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  int32_t phase = atomic_load(&executor->phase);
  if (phase == LF_ORDERED_IDLE || phase == LF_ORDERED_COMPLETE) {
    atomic_store(&executor->phase, LF_ORDERED_COMPLETE);
    result = LF_OK;
    goto complete;
  }
  result = lf_context_current(executor->context);
  if (result == LF_OK) {
    result = lf_cuda_map_result(executor->context->api->cuStreamSynchronize(
      (CUstream)executor->stream->handle
    ));
  }
  if (result == LF_OK) {
    atomic_store(&executor->phase, LF_ORDERED_COMPLETE);
  }
complete:
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_reset(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  if (atomic_load(&executor->phase) != LF_ORDERED_COMPLETE) {
    result = LF_BUSY;
  } else {
    atomic_store(&executor->next_kernel, 0);
    atomic_store(&executor->phase, LF_ORDERED_IDLE);
    result = LF_OK;
  }
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_close(lf_ordered_executor *executor) {
  return lf_close_ordered_executor(executor);
}
