#include "streaming_internal.h"
#include <string.h>

static int lf_overlaps(size_t a, size_t n, size_t b, size_t m) {
  return a < b + m && b < a + n;
}

static int lf_range(int64_t offset, int64_t bytes, size_t size) {
  return offset >= 0 && bytes > 0 && (uint64_t)offset <= size &&
    (uint64_t)bytes <= size - (size_t)offset;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_enqueue(
  lf_streaming_pool *pool, int32_t index, int64_t host_offset,
  int64_t device_offset, int64_t bytes, int32_t to_device
) {
  int32_t result = lf_streaming_begin(pool);
  if (result != LF_OK) return result;
  if (index < 0 || index >= pool->lanes || (to_device != 0 && to_device != 1) ||
      !lf_range(host_offset, bytes, pool->host_bytes) ||
      !lf_range(device_offset, bytes, pool->allocation->size)) {
    result = LF_INVALID_ARGUMENT;
    goto done;
  }
  lf_transfer_lane *lane = &pool->lane[index];
  if ((lane->phase != LF_TRANSFER_IDLE && lane->phase != LF_TRANSFER_QUEUED) ||
      lane->count == pool->max_regions) { result = LF_BUSY; goto done; }
  for (int32_t i = 0; i < pool->lanes; ++i) {
    lf_transfer_lane *other = &pool->lane[i];
    for (int32_t j = 0; j < other->count; ++j) {
      lf_transfer_region r = other->regions[j];
      if (lf_overlaps((size_t)host_offset, (size_t)bytes, r.host_offset, r.bytes) ||
          lf_overlaps((size_t)device_offset, (size_t)bytes, r.device_offset, r.bytes)) {
        result = LF_BUSY;
        goto done;
      }
    }
  }
  CUdeviceptr address = 0;
  result = lf_allocation_region_address(pool->allocation, device_offset,
    (size_t)bytes, 1, 0, &address);
  if (result != LF_OK) goto done;
  lane->regions[lane->count++] = (lf_transfer_region){
    (size_t)host_offset, (size_t)device_offset, (size_t)bytes};
  /* Retain ranges even if the driver reports an asynchronous prior error. */
  lane->phase = LF_TRANSFER_POISONED;
  lf_cuda_api *api = pool->allocation->context->api;
  void *host = (uint8_t *)pool->host + (size_t)host_offset;
  result = lf_cuda_map_result(to_device
    ? api->cuMemcpyHtoDAsync(address, host, (size_t)bytes, lane->stream)
    : api->cuMemcpyDtoHAsync(host, address, (size_t)bytes, lane->stream));
  if (result == LF_OK) lane->phase = LF_TRANSFER_QUEUED;
done:
  lf_streaming_end(pool);
  return result;
}

static int32_t lf_lane_begin(lf_streaming_pool *pool, int32_t index) {
  int32_t result = lf_streaming_begin(pool);
  if (result != LF_OK) return result;
  if (index < 0 || index >= pool->lanes) {
    lf_streaming_end(pool);
    return LF_INVALID_ARGUMENT;
  }
  return LF_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_record(lf_streaming_pool *pool, int32_t index) {
  int32_t result = lf_lane_begin(pool, index);
  if (result != LF_OK) return result;
  lf_transfer_lane *lane = &pool->lane[index];
  if (lane->phase != LF_TRANSFER_QUEUED) result = LF_BUSY;
  else {
    lane->phase = LF_TRANSFER_POISONED;
    result = lf_cuda_map_result(pool->allocation->context->api->cuEventRecord(
      lane->event, lane->stream));
    if (result == LF_OK) lane->phase = LF_TRANSFER_RECORDED;
  }
  lf_streaming_end(pool);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_poll(lf_streaming_pool *pool, int32_t index) {
  int32_t result = lf_lane_begin(pool, index);
  if (result != LF_OK) return result;
  lf_transfer_lane *lane = &pool->lane[index];
  if (lane->phase == LF_TRANSFER_COMPLETE) result = 1;
  else if (lane->phase != LF_TRANSFER_RECORDED) result = LF_BUSY;
  else {
    CUresult query = pool->allocation->context->api->cuEventQuery(lane->event);
    if (query == CUDA_ERROR_NOT_READY) result = 0;
    else if (query == 0) { lane->phase = LF_TRANSFER_COMPLETE; result = 1; }
    else { lane->phase = LF_TRANSFER_POISONED; result = LF_DRIVER_FAILURE; }
  }
  lf_streaming_end(pool);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_drain(lf_streaming_pool *pool, int32_t index) {
  int32_t result = lf_lane_begin(pool, index);
  if (result != LF_OK) return result;
  lf_transfer_lane *lane = &pool->lane[index];
  if (lf_transfer_pending(lane->phase)) {
    result = lf_cuda_map_result(pool->allocation->context->api->cuStreamSynchronize(
      lane->stream));
    if (result == LF_OK) lane->phase = LF_TRANSFER_COMPLETE;
  }
  lf_streaming_end(pool);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_reset(lf_streaming_pool *pool, int32_t index) {
  int32_t result = lf_lane_begin(pool, index);
  if (result != LF_OK) return result;
  lf_transfer_lane *lane = &pool->lane[index];
  if (lf_transfer_pending(lane->phase)) result = LF_BUSY;
  else { lane->phase = LF_TRANSFER_IDLE; lane->count = 0; }
  lf_streaming_end(pool);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_streaming_host_copy(
  lf_streaming_pool *pool, uint8_t *buffer, int64_t host_offset,
  int32_t offset, int32_t bytes, int32_t write
) {
  int32_t result = lf_streaming_begin(pool);
  if (result != LF_OK) return result;
  if (buffer == NULL || offset < 0 || bytes <= 0 ||
      offset > Moonbit_array_length(buffer) ||
      bytes > Moonbit_array_length(buffer) - offset ||
      (write != 0 && write != 1) || !lf_range(host_offset, bytes, pool->host_bytes)) {
    result = LF_INVALID_ARGUMENT;
    goto done;
  }
  for (int32_t i = 0; i < pool->lanes; ++i) {
    lf_transfer_lane *lane = &pool->lane[i];
    if (!lf_transfer_pending(lane->phase)) continue;
    for (int32_t j = 0; j < lane->count; ++j) {
      lf_transfer_region r = lane->regions[j];
      if (lf_overlaps((size_t)host_offset, (size_t)bytes, r.host_offset, r.bytes)) {
        result = LF_BUSY;
        goto done;
      }
    }
  }
  void *host = (uint8_t *)pool->host + (size_t)host_offset;
  if (write) memcpy(host, buffer + offset, (size_t)bytes);
  else memcpy(buffer + offset, host, (size_t)bytes);
done:
  lf_streaming_end(pool);
  return result;
}
