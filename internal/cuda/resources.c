#include "resource_internal.h"

#include <limits.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_SUCCESS 0

int32_t lf_context_current(lf_context *context) {
  if (context == NULL || atomic_load(&context->state) != LF_RESOURCE_LIVE) {
    return LF_CLOSED;
  }
  return lf_cuda_map_result(context->api->cuCtxSetCurrent(context->handle));
}

int32_t lf_operation_begin(
  atomic_int *state,
  atomic_int *active_operations
) {
  if (atomic_load(state) != LF_RESOURCE_LIVE) return LF_CLOSED;
  atomic_fetch_add(active_operations, 1);
  if (atomic_load(state) != LF_RESOURCE_LIVE) {
    atomic_fetch_sub(active_operations, 1);
    return LF_BUSY;
  }
  return LF_OK;
}

void lf_operation_end(atomic_int *active_operations) {
  atomic_fetch_sub(active_operations, 1);
}

int32_t lf_begin_close(
  atomic_int *state,
  atomic_int *active_operations
) {
  int expected = LF_RESOURCE_LIVE;
  if (atomic_compare_exchange_strong(
        state,
        &expected,
        LF_RESOURCE_CLOSING
      )) {
    if (atomic_load(active_operations) == 0) return LF_OK;
    atomic_store(state, LF_RESOURCE_LIVE);
    return LF_BUSY;
  }
  return expected == LF_RESOURCE_CLOSED ? LF_CLOSED : LF_BUSY;
}

void lf_close_failed(atomic_int *state) {
  atomic_store(state, LF_RESOURCE_LIVE);
}

void lf_close_succeeded(atomic_int *state) {
  atomic_store(state, LF_RESOURCE_CLOSED);
}

void lf_finalize_failure(void) {
  /* A finalizer cannot return ownership to MoonBit or report a retryable error.
   * Continuing would discard the only wrapper for a live native resource and
   * its retained parent. Fail closed instead of silently leaking or allowing a
   * parent context to disappear while children remain. */
  abort();
}

void lf_release_context_child(lf_context *context) {
  if (context == NULL) return;
  atomic_fetch_sub(&context->children, 1);
  moonbit_decref(context);
}

static int32_t lf_close_context(lf_context *context) {
  if (context == NULL) return LF_INVALID_ARGUMENT;
  if (atomic_load(&context->state) == LF_RESOURCE_CLOSED) return LF_OK;
  if (atomic_load(&context->children) != 0) return LF_BUSY;
  int32_t begin = lf_begin_close(
    &context->state,
    &context->active_operations
  );
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  if (atomic_load(&context->children) != 0) {
    lf_close_failed(&context->state);
    return LF_BUSY;
  }
  CUcontext handle = context->handle;
  int32_t result = handle == NULL
    ? LF_OK
    : lf_cuda_map_result(context->api->cuCtxDestroy(handle));
  if (result != LF_OK) {
    lf_close_failed(&context->state);
    return result;
  }
  context->handle = NULL;
  lf_close_succeeded(&context->state);
  return LF_OK;
}

static void lf_finalize_context(void *object) {
  if (lf_close_context((lf_context *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_context *lunaflux_cuda_context_create(int32_t ordinal, int32_t *status) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_finalize_context,
    sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  atomic_init(&context->state, LF_RESOURCE_CLOSED);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  context->api = lf_cuda_api_get();
  if (context->api->availability != LF_AVAILABLE) {
    *status = LF_UNAVAILABLE;
    return context;
  }
  if (ordinal < 0 || ordinal >= context->api->device_count) {
    *status = LF_INVALID_ARGUMENT;
    return context;
  }
  CUdevice device = 0;
  if (context->api->cuDeviceGet(&device, ordinal) != CUDA_SUCCESS ||
      context->api->cuCtxCreate(&context->handle, 0, device) != CUDA_SUCCESS) {
    context->handle = NULL;
    *status = LF_DRIVER_FAILURE;
    return context;
  }
  atomic_store(&context->state, LF_RESOURCE_LIVE);
  *status = LF_OK;
  return context;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_context_close(lf_context *context) {
  return lf_close_context(context);
}

static int32_t lf_close_stream(lf_child *stream) {
  if (stream == NULL) return LF_INVALID_ARGUMENT;
  int32_t begin = lf_begin_close(&stream->state, &stream->active_operations);
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  int32_t result = lf_context_current(stream->context);
  if (result == LF_OK && stream->handle != NULL) {
    result = lf_cuda_map_result(stream->context->api->cuStreamDestroy(
      (CUstream)stream->handle
    ));
  }
  if (result != LF_OK) {
    lf_close_failed(&stream->state);
    return result;
  }
  stream->handle = NULL;
  lf_release_context_child(stream->context);
  stream->context = NULL;
  lf_close_succeeded(&stream->state);
  return LF_OK;
}

static void lf_finalize_stream(void *object) {
  if (lf_close_stream((lf_child *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_child *lunaflux_cuda_stream_create(lf_context *context, int32_t *status) {
  lf_child *stream = (lf_child *)moonbit_make_external_object(
    lf_finalize_stream,
    sizeof(lf_child)
  );
  memset(stream, 0, sizeof(*stream));
  atomic_init(&stream->state, LF_RESOURCE_CLOSED);
  atomic_init(&stream->active_operations, 0);
  atomic_init(&stream->children, 0);
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return stream;
  *status = lf_context_current(context);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return stream;
  }
  CUstream handle = NULL;
  *status = lf_cuda_map_result(context->api->cuStreamCreate(&handle, 0));
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return stream;
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  stream->context = context;
  stream->handle = handle;
  atomic_store(&stream->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return stream;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_stream_close(lf_child *stream) {
  return lf_close_stream(stream);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_stream_synchronize(lf_child *stream) {
  if (stream == NULL) return LF_CLOSED;
  int32_t acquired = lf_operation_begin(
    &stream->state,
    &stream->active_operations
  );
  if (acquired != LF_OK) return acquired;
  int32_t current = lf_context_current(stream->context);
  if (current == LF_OK) {
    current = lf_cuda_map_result(
      stream->context->api->cuStreamSynchronize((CUstream)stream->handle)
    );
  }
  lf_operation_end(&stream->active_operations);
  return current;
}

static int32_t lf_close_event(lf_child *event) {
  if (event == NULL) return LF_INVALID_ARGUMENT;
  int32_t begin = lf_begin_close(&event->state, &event->active_operations);
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  int32_t result = lf_context_current(event->context);
  if (result == LF_OK && event->handle != NULL) {
    result = lf_cuda_map_result(event->context->api->cuEventDestroy(
      (CUevent)event->handle
    ));
  }
  if (result != LF_OK) {
    lf_close_failed(&event->state);
    return result;
  }
  event->handle = NULL;
  lf_release_context_child(event->context);
  event->context = NULL;
  lf_close_succeeded(&event->state);
  return LF_OK;
}

static void lf_finalize_event(void *object) {
  if (lf_close_event((lf_child *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_child *lunaflux_cuda_event_create(lf_context *context, int32_t *status) {
  lf_child *event = (lf_child *)moonbit_make_external_object(
    lf_finalize_event,
    sizeof(lf_child)
  );
  memset(event, 0, sizeof(*event));
  atomic_init(&event->state, LF_RESOURCE_CLOSED);
  atomic_init(&event->active_operations, 0);
  atomic_init(&event->children, 0);
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return event;
  *status = lf_context_current(context);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return event;
  }
  CUevent handle = NULL;
  *status = lf_cuda_map_result(context->api->cuEventCreate(&handle, 0));
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return event;
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  event->context = context;
  event->handle = handle;
  atomic_store(&event->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return event;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_event_close(lf_child *event) {
  return lf_close_event(event);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_event_record(lf_child *event, lf_child *stream) {
  if (event == NULL || stream == NULL) return LF_CLOSED;
  int32_t result = lf_operation_begin(&event->state, &event->active_operations);
  if (result != LF_OK) return result;
  int stream_acquired = 0;
  result = lf_operation_begin(&stream->state, &stream->active_operations);
  if (result == LF_OK) stream_acquired = 1;
  if (result == LF_OK && event->context != stream->context) {
    result = LF_INVALID_ARGUMENT;
  }
  if (result == LF_OK) result = lf_context_current(event->context);
  if (result == LF_OK) {
    result = lf_cuda_map_result(event->context->api->cuEventRecord(
      (CUevent)event->handle,
      (CUstream)stream->handle
    ));
  }
  if (stream_acquired != 0) {
    lf_operation_end(&stream->active_operations);
  }
  lf_operation_end(&event->active_operations);
  return result;
}

int32_t lunaflux_cuda_event_synchronize(lf_child *event) {
  if (event == NULL) return LF_CLOSED;
  int32_t result = lf_operation_begin(&event->state, &event->active_operations);
  if (result != LF_OK) return result;
  result = lf_context_current(event->context);
  if (result == LF_OK) {
    result = lf_cuda_map_result(
      event->context->api->cuEventSynchronize((CUevent)event->handle)
    );
  }
  lf_operation_end(&event->active_operations);
  return result;
}

MOONBIT_FFI_EXPORT
double lunaflux_cuda_event_elapsed(
  lf_child *start,
  lf_child *end,
  int32_t *status
) {
  if (start == NULL || end == NULL) {
    *status = LF_CLOSED;
    return 0.0;
  }
  *status = lf_operation_begin(&start->state, &start->active_operations);
  if (*status != LF_OK) return 0.0;
  int32_t end_acquired = lf_operation_begin(&end->state, &end->active_operations);
  if (end_acquired != LF_OK) {
    *status = end_acquired;
    lf_operation_end(&start->active_operations);
    return 0.0;
  }
  if (start->context != end->context) {
    *status = LF_INVALID_ARGUMENT;
    lf_operation_end(&end->active_operations);
    lf_operation_end(&start->active_operations);
    return 0.0;
  }
  *status = lf_context_current(start->context);
  if (*status != LF_OK) {
    lf_operation_end(&end->active_operations);
    lf_operation_end(&start->active_operations);
    return 0.0;
  }
  float millis = 0.0f;
  *status = lf_cuda_map_result(start->context->api->cuEventElapsedTime(
    &millis,
    (CUevent)start->handle,
    (CUevent)end->handle
  ));
  lf_operation_end(&end->active_operations);
  lf_operation_end(&start->active_operations);
  return (double)millis;
}

static int32_t lf_close_allocation(lf_allocation *allocation) {
  if (allocation == NULL) return LF_INVALID_ARGUMENT;
  int32_t begin = lf_begin_close(
    &allocation->state,
    &allocation->active_operations
  );
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  int32_t result = lf_context_current(allocation->context);
  if (result == LF_OK && allocation->handle != 0) {
    result = lf_cuda_map_result(
      allocation->context->api->cuMemFree(allocation->handle)
    );
  }
  if (result != LF_OK) {
    lf_close_failed(&allocation->state);
    return result;
  }
  allocation->handle = 0;
  allocation->size = 0;
  lf_release_context_child(allocation->context);
  allocation->context = NULL;
  lf_close_succeeded(&allocation->state);
  return LF_OK;
}

static void lf_finalize_allocation(void *object) {
  if (lf_close_allocation((lf_allocation *)object) != LF_OK) {
    lf_finalize_failure();
  }
}

MOONBIT_FFI_EXPORT
lf_allocation *lunaflux_cuda_allocation_create(
  lf_context *context,
  int64_t byte_count,
  int32_t *status
) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_finalize_allocation,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  atomic_init(&allocation->state, LF_RESOURCE_CLOSED);
  atomic_init(&allocation->active_operations, 0);
  if (byte_count <= 0) {
    *status = LF_INVALID_ARGUMENT;
    return allocation;
  }
  if ((uint64_t)byte_count > SIZE_MAX) {
    *status = LF_SIZE_OVERFLOW;
    return allocation;
  }
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return allocation;
  *status = lf_context_current(context);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return allocation;
  }
  CUdeviceptr handle = 0;
  *status = lf_cuda_map_result(
    context->api->cuMemAlloc(&handle, (size_t)byte_count)
  );
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return allocation;
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  allocation->context = context;
  allocation->handle = handle;
  allocation->size = (size_t)byte_count;
  atomic_store(&allocation->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return allocation;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_allocation_close(lf_allocation *allocation) {
  return lf_close_allocation(allocation);
}
