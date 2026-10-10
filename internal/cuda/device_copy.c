#include "resource_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct lf_copy_region {
  lf_allocation *source;
  lf_allocation *destination;
  CUdeviceptr source_address;
  CUdeviceptr destination_address;
  size_t bytes;
} lf_copy_region;

typedef struct lf_device_copy {
  lf_context *context;
  lf_child *stream;
  lf_copy_region *regions;
  int32_t count;
  CUevent event;
  int pending;
  int recorded;
  atomic_int state;
  atomic_int active_operations;
  atomic_int operation_gate;
} lf_device_copy;

static int32_t lf_copy_begin(lf_device_copy *copy) {
  if (copy == NULL) return LF_CLOSED;
  int32_t status = lf_operation_begin(&copy->state, &copy->active_operations);
  if (status != LF_OK) return status;
  int expected = 0;
  if (!atomic_compare_exchange_strong(&copy->operation_gate, &expected, 1)) {
    lf_operation_end(&copy->active_operations);
    return LF_BUSY;
  }
  return LF_OK;
}

static void lf_copy_end(lf_device_copy *copy) {
  atomic_store(&copy->operation_gate, 0);
  lf_operation_end(&copy->active_operations);
}

static void lf_copy_release(lf_device_copy *copy) {
  for (int32_t i = copy->count; i > 0; --i) {
    lf_copy_region *region = &copy->regions[i-1];
    if (region->destination != NULL) {
      lf_operation_end(&region->destination->active_operations);
      moonbit_decref(region->destination);
    }
    if (region->source != NULL) {
      lf_operation_end(&region->source->active_operations);
      moonbit_decref(region->source);
    }
  }
  free(copy->regions);
  copy->regions = NULL;
  copy->count = 0;
  if (copy->stream != NULL) {
    lf_operation_end(&copy->stream->active_operations);
    moonbit_decref(copy->stream);
    copy->stream = NULL;
  }
  copy->context = NULL;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_device_copy_close(lf_device_copy *copy) {
  if (copy == NULL) return LF_INVALID_ARGUMENT;
  if (atomic_load(&copy->state) == LF_RESOURCE_CLOSED) return LF_OK;
  int32_t status = lf_begin_close(&copy->state, &copy->active_operations);
  if (status == LF_CLOSED) return LF_OK;
  if (status != LF_OK) return status;
  if (copy->pending) {
    lf_close_failed(&copy->state);
    return LF_BUSY;
  }
  if (copy->event != NULL) {
    status = lf_context_current(copy->context);
    if (status == LF_OK) {
      status = lf_cuda_map_result(copy->context->api->cuEventDestroy(copy->event));
    }
    if (status != LF_OK) {
      lf_close_failed(&copy->state);
      return status;
    }
    copy->event = NULL;
  }
  lf_copy_release(copy);
  lf_close_succeeded(&copy->state);
  return LF_OK;
}

static void lf_copy_finalize(void *object) {
  lf_device_copy *copy = object;
  if (copy->pending) {
    int32_t status = lf_context_current(copy->context);
    if (status == LF_OK) status = lf_cuda_map_result(
      copy->context->api->cuStreamSynchronize((CUstream)copy->stream->handle));
    if (status != LF_OK) { lf_finalize_failure(); return; }
    copy->pending = 0;
  }
  if (lunaflux_cuda_device_copy_close(copy) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_device_copy *lunaflux_cuda_device_copy_create(
  lf_context *context, lf_child *stream,
  lf_allocation **sources, lf_allocation **destinations,
  int64_t *source_offsets, int64_t *destination_offsets, int64_t *sizes,
  int32_t *status
) {
  lf_device_copy *copy = moonbit_make_external_object(lf_copy_finalize, sizeof(*copy));
  memset(copy, 0, sizeof(*copy));
  atomic_init(&copy->state, LF_RESOURCE_LIVE);
  atomic_init(&copy->active_operations, 0);
  atomic_init(&copy->operation_gate, 0);
  *status = LF_INVALID_ARGUMENT;
  if (context == NULL || stream == NULL || sources == NULL || destinations == NULL ||
      source_offsets == NULL || destination_offsets == NULL || sizes == NULL) return copy;
  int32_t count = Moonbit_array_length(sizes);
  if (count <= 0 || count != Moonbit_array_length(sources) ||
      count != Moonbit_array_length(destinations) ||
      count != Moonbit_array_length(source_offsets) ||
      count != Moonbit_array_length(destination_offsets)) return copy;
  *status = lf_operation_begin(&stream->state, &stream->active_operations);
  if (*status != LF_OK) return copy;
  moonbit_incref(stream);
  copy->stream = stream;
  copy->context = stream->context;
  if (stream->context != context) { *status = LF_INVALID_ARGUMENT; return copy; }
  if (context->api->cuMemcpyDtoDAsync == NULL) { *status = LF_UNSUPPORTED; return copy; }
  if ((uint64_t)count > SIZE_MAX / sizeof(lf_copy_region)) {
    *status = LF_SIZE_OVERFLOW; return copy;
  }
  copy->regions = calloc((size_t)count, sizeof(lf_copy_region));
  if (copy->regions == NULL) { *status = LF_HOST_ALLOCATION_FAILED; return copy; }
  for (int32_t i = 0; i < count; ++i) {
    lf_copy_region *region = &copy->regions[i];
    copy->count = i+1;
    lf_allocation *pair[2] = {sources[i], destinations[i]};
    for (int j = 0; j < 2; ++j) {
      if (pair[j] == NULL) { *status = LF_CLOSED; return copy; }
      *status = lf_operation_begin(&pair[j]->state, &pair[j]->active_operations);
      if (*status != LF_OK) return copy;
      moonbit_incref(pair[j]);
      if (j == 0) region->source = pair[j]; else region->destination = pair[j];
      if (pair[j]->context != context) { *status = LF_INVALID_ARGUMENT; return copy; }
    }
    if (sizes[i] <= 0 || (uint64_t)sizes[i] > SIZE_MAX) {
      *status = LF_INVALID_ARGUMENT; return copy;
    }
    region->bytes = (size_t)sizes[i];
    *status = lf_allocation_region_address(region->source, source_offsets[i],
      region->bytes, 1, 0, &region->source_address);
    if (*status != LF_OK) return copy;
    *status = lf_allocation_region_address(region->destination, destination_offsets[i],
      region->bytes, 1, 0, &region->destination_address);
    if (*status != LF_OK) return copy;
    /* Each destination must be disjoint from every source and destination.
     * This also makes reverse-direction rollback independent of copy order. */
    if ((uint64_t)(region->bytes-1) > UINT64_MAX-region->source_address ||
        (uint64_t)(region->bytes-1) > UINT64_MAX-region->destination_address) {
      *status = LF_SIZE_OVERFLOW; return copy;
    }
    for (int32_t j = 0; j <= i; ++j) {
      lf_copy_region *previous = &copy->regions[j];
      CUdeviceptr a = region->destination_address, b = previous->source_address;
      if (a <= b ? b-a < region->bytes : a-b < previous->bytes) {
        *status = LF_INVALID_ARGUMENT; return copy;
      }
      if (j == i) continue;
      b = previous->destination_address;
      if (a <= b ? b-a < region->bytes : a-b < previous->bytes) {
        *status = LF_INVALID_ARGUMENT; return copy;
      }
      a = region->source_address;
      if (a <= b ? b-a < region->bytes : a-b < previous->bytes) {
        *status = LF_INVALID_ARGUMENT; return copy;
      }
      b = previous->source_address;
      if (a <= b ? b-a < region->bytes : a-b < previous->bytes) {
        *status = LF_INVALID_ARGUMENT; return copy;
      }
    }
  }
  *status = lf_context_current(context);
  if (*status == LF_OK) *status = lf_cuda_map_result(context->api->cuEventCreate(&copy->event, 2));
  return copy;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_device_copy_submit(lf_device_copy *copy, int32_t reverse) {
  int32_t status = lf_copy_begin(copy);
  if (status != LF_OK) return status;
  if (copy->pending) status = LF_BUSY;
  else if (copy->event == NULL || (reverse != 0 && reverse != 1)) status = LF_INVALID_ARGUMENT;
  else status = lf_context_current(copy->context);
  if (status == LF_OK) {
    copy->pending = 1;
    copy->recorded = 0;
    for (int32_t i = 0; i < copy->count; ++i) {
      lf_copy_region *r = &copy->regions[i];
      status = lf_cuda_map_result(copy->context->api->cuMemcpyDtoDAsync(
        reverse ? r->source_address : r->destination_address,
        reverse ? r->destination_address : r->source_address,
        r->bytes, (CUstream)copy->stream->handle));
      if (status != LF_OK) break;
    }
    if (status == LF_OK) {
      status = lf_cuda_map_result(copy->context->api->cuEventRecord(copy->event,
        (CUstream)copy->stream->handle));
      if (status == LF_OK) copy->recorded = 1;
    }
  }
  lf_copy_end(copy);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_device_copy_retire(lf_device_copy *copy, int32_t wait) {
  int32_t status = lf_copy_begin(copy);
  if (status != LF_OK) return status;
  if (!copy->pending) status = wait ? 1 : LF_INVALID_ARGUMENT;
  else if (!wait && !copy->recorded) status = LF_DRIVER_FAILURE;
  else {
    status = lf_context_current(copy->context);
    if (status == LF_OK) {
      CUresult result = wait
        ? copy->context->api->cuStreamSynchronize((CUstream)copy->stream->handle)
        : copy->context->api->cuEventQuery(copy->event);
      status = !wait && result == CUDA_ERROR_NOT_READY ? 0 : lf_cuda_map_result(result);
      if (result == 0) { copy->pending = 0; status = 1; }
    }
  }
  lf_copy_end(copy);
  return status;
}

#include "device_copy_probe.c"
