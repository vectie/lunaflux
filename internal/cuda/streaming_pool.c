#include "streaming_internal.h"
#include <stdlib.h>
#include <string.h>

int lf_transfer_pending(int32_t phase) {
  return phase == LF_TRANSFER_QUEUED || phase == LF_TRANSFER_RECORDED ||
    phase == LF_TRANSFER_POISONED;
}

int32_t lf_streaming_begin(lf_streaming_pool *pool) {
  if (pool == NULL) return LF_INVALID_ARGUMENT;
  int32_t result = lf_operation_begin(&pool->state, &pool->active_operations);
  if (result != LF_OK) return result;
  int expected = 0;
  if (!atomic_compare_exchange_strong(&pool->gate, &expected, 1)) {
    lf_operation_end(&pool->active_operations);
    return LF_BUSY;
  }
  if (!pool->usable) {
    lf_streaming_end(pool);
    return LF_CLOSED;
  }
  result = lf_context_current(pool->allocation->context);
  if (result != LF_OK) lf_streaming_end(pool);
  return result;
}

void lf_streaming_end(lf_streaming_pool *pool) {
  atomic_store(&pool->gate, 0);
  lf_operation_end(&pool->active_operations);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_close(lf_streaming_pool *pool) {
  if (pool == NULL) return LF_INVALID_ARGUMENT;
  int32_t result = lf_begin_close(&pool->state, &pool->active_operations);
  if (result == LF_CLOSED) return LF_OK;
  if (result != LF_OK) return result;
  for (int32_t i = 0; i < pool->lanes; ++i) {
    if (pool->lane != NULL && lf_transfer_pending(pool->lane[i].phase)) {
      lf_close_failed(&pool->state);
      return LF_BUSY;
    }
  }
  pool->usable = 0;
  lf_cuda_api *api = pool->allocation == NULL ? NULL : pool->allocation->context->api;
  if (api != NULL) {
    result = lf_context_current(pool->allocation->context);
    if (result != LF_OK) goto failed;
    for (int32_t i = pool->lanes; i > 0 && pool->lane != NULL; --i) {
      lf_transfer_lane *lane = &pool->lane[i - 1];
      if (lane->event != NULL) {
        result = lf_cuda_map_result(api->cuEventDestroy(lane->event));
        if (result != LF_OK) goto failed;
        lane->event = NULL;
      }
      if (lane->stream != NULL) {
        result = lf_cuda_map_result(api->cuStreamDestroy(lane->stream));
        if (result != LF_OK) goto failed;
        lane->stream = NULL;
      }
    }
    if (pool->host != NULL) {
      memset(pool->host, 0, pool->host_bytes);
      result = lf_cuda_map_result(api->cuMemFreeHost(pool->host));
      if (result != LF_OK) goto failed;
      pool->host = NULL;
    }
  }
  free(pool->regions);
  free(pool->lane);
  pool->regions = NULL;
  pool->lane = NULL;
  if (pool->allocation != NULL) {
    lf_operation_end(&pool->allocation->active_operations);
    moonbit_decref(pool->allocation);
    pool->allocation = NULL;
  }
  lf_close_succeeded(&pool->state);
  return LF_OK;
failed:
  lf_close_failed(&pool->state);
  return result;
}

static void lf_streaming_finalize(void *object) {
  lf_streaming_pool *pool = object;
  if (atomic_load(&pool->state) == LF_RESOURCE_CLOSED) return;
  for (int32_t i = 0; i < pool->lanes && pool->lane != NULL; ++i) {
    if (lf_transfer_pending(pool->lane[i].phase) &&
        lunaflux_cuda_streaming_drain(pool, i) != LF_OK) {
      lf_finalize_failure();
      return;
    }
  }
  if (lunaflux_cuda_streaming_close(pool) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_streaming_pool *lunaflux_cuda_streaming_create(
  lf_context *context, lf_allocation *allocation, int64_t host_bytes,
  int32_t lanes, int32_t max_regions, int32_t *status
) {
  lf_streaming_pool *pool = moonbit_make_external_object(
    lf_streaming_finalize, sizeof(*pool));
  memset(pool, 0, sizeof(*pool));
  atomic_init(&pool->state, LF_RESOURCE_LIVE);
  atomic_init(&pool->active_operations, 0);
  atomic_init(&pool->gate, 0);
  *status = LF_INVALID_ARGUMENT;
  if (context == NULL || allocation == NULL || host_bytes <= 0 ||
      (uint64_t)host_bytes > SIZE_MAX || lanes <= 0 || lanes > 64 ||
      max_regions <= 0 || max_regions > 4096) return pool;
  *status = lf_operation_begin(&allocation->state, &allocation->active_operations);
  if (*status != LF_OK) return pool;
  if (allocation->context != context) {
    lf_operation_end(&allocation->active_operations);
    *status = LF_INVALID_ARGUMENT;
    return pool;
  }
  moonbit_incref(allocation);
  pool->allocation = allocation;
  *status = lf_context_current(context);
  if (*status != LF_OK) return pool;
  lf_cuda_api *api = context->api;
  if (api->cuMemHostAlloc == NULL || api->cuMemFreeHost == NULL ||
      api->cuMemcpyHtoDAsync == NULL || api->cuMemcpyDtoHAsync == NULL) {
    *status = LF_UNSUPPORTED;
    return pool;
  }
  pool->lanes = lanes;
  pool->max_regions = max_regions;
  pool->host_bytes = (size_t)host_bytes;
  pool->lane = calloc((size_t)lanes, sizeof(*pool->lane));
  pool->regions = calloc((size_t)lanes * (size_t)max_regions, sizeof(*pool->regions));
  if (pool->lane == NULL || pool->regions == NULL) {
    *status = LF_HOST_ALLOCATION_FAILED;
    return pool;
  }
  *status = lf_cuda_map_result(api->cuMemHostAlloc(&pool->host, pool->host_bytes, 0));
  if (*status != LF_OK) return pool;
  if (pool->host == NULL) { *status = LF_INVALID_OUTPUT; return pool; }
  memset(pool->host, 0, pool->host_bytes);
  for (int32_t i = 0; i < lanes; ++i) {
    pool->lane[i].regions = pool->regions + (size_t)i * (size_t)max_regions;
    *status = lf_cuda_map_result(api->cuStreamCreate(&pool->lane[i].stream, 1));
    if (*status != LF_OK) return pool;
    if (pool->lane[i].stream == NULL) { *status = LF_INVALID_OUTPUT; return pool; }
    *status = lf_cuda_map_result(api->cuEventCreate(&pool->lane[i].event, 2));
    if (*status != LF_OK) return pool;
    if (pool->lane[i].event == NULL) { *status = LF_INVALID_OUTPUT; return pool; }
  }
  pool->usable = 1;
  return pool;
}
