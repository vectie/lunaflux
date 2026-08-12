#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_TRANSFER_CONTEXT ((void *)(uintptr_t)0x110)
#define LF_TRANSFER_ADDRESS ((CUdeviceptr)0x5000)

typedef struct lf_transfer_probe_state {
  int32_t fail_mem_free_once;
  int32_t copy_to_device_calls;
  int32_t copy_to_host_calls;
  const uint8_t *expected_host_source;
} lf_transfer_probe_state;

static lf_transfer_probe_state *lf_active_transfer_probe;

static CUresult lf_transfer_context_current(CUcontext context) {
  return context == LF_TRANSFER_CONTEXT ? 0 : 1;
}

static CUresult lf_transfer_context_destroy(CUcontext context) {
  return context == LF_TRANSFER_CONTEXT ? 0 : 1;
}

static CUresult lf_transfer_mem_free(CUdeviceptr address) {
  if (lf_active_transfer_probe->fail_mem_free_once != 0) {
    lf_active_transfer_probe->fail_mem_free_once = 0;
    return 1;
  }
  return address == LF_TRANSFER_ADDRESS ? 0 : 1;
}

static CUresult lf_transfer_to_device(
  CUdeviceptr destination,
  const void *source,
  size_t byte_count
) {
  if (destination != LF_TRANSFER_ADDRESS + 3 ||
      source != lf_active_transfer_probe->expected_host_source + 5 ||
      byte_count != 8) return 1;
  lf_active_transfer_probe->copy_to_device_calls += 1;
  return 0;
}

static CUresult lf_transfer_to_host(
  void *destination,
  CUdeviceptr source,
  size_t byte_count
) {
  if (source != LF_TRANSFER_ADDRESS + 4 || byte_count != 6) return 1;
  uint8_t *bytes = (uint8_t *)destination;
  for (size_t index = 0; index < byte_count; index += 1) {
    bytes[index] = (uint8_t)(index + 10);
  }
  lf_active_transfer_probe->copy_to_host_calls += 1;
  return 0;
}

static void lf_transfer_finalizer(void *object) {
  (void)object;
}

static void lf_transfer_api(lf_cuda_api *api) {
  memset(api, 0, sizeof(*api));
  api->cuCtxSetCurrent = lf_transfer_context_current;
  api->cuCtxDestroy = lf_transfer_context_destroy;
  api->cuMemFree = lf_transfer_mem_free;
  api->cuMemcpyHtoD = lf_transfer_to_device;
  api->cuMemcpyDtoH = lf_transfer_to_host;
}

static lf_context *lf_transfer_make_context(lf_cuda_api *api) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_transfer_finalizer,
    sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  context->api = api;
  context->handle = LF_TRANSFER_CONTEXT;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  return context;
}

static lf_allocation *lf_transfer_make_allocation(
  lf_context *context,
  int retain_context
) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_transfer_finalizer,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = LF_TRANSFER_ADDRESS;
  allocation->size = 16;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  if (retain_context != 0) {
    moonbit_incref(context);
    atomic_fetch_add(&context->children, 1);
  }
  return allocation;
}

static int32_t lf_test_transfer_bounds(void) {
  lf_transfer_probe_state state;
  memset(&state, 0, sizeof(state));
  lf_active_transfer_probe = &state;
  lf_cuda_api api;
  lf_transfer_api(&api);
  lf_context *context = lf_transfer_make_context(&api);
  lf_allocation *allocation = lf_transfer_make_allocation(context, 0);
  moonbit_bytes_t source = moonbit_make_bytes(32, 0);
  for (int32_t index = 0; index < 32; index += 1) source[index] = (uint8_t)index;
  state.expected_host_source = source;
  int32_t result = 0;
  if (lunaflux_cuda_copy_to_device(allocation, source, 5, 3, 8) != LF_OK ||
      state.copy_to_device_calls != 1) result = 101;
  if (result == 0 &&
      lunaflux_cuda_copy_to_device(allocation, source, 25, 0, 8) !=
        LF_INVALID_ARGUMENT) result = 102;
  if (result == 0 &&
      lunaflux_cuda_copy_to_device(allocation, source, 0, 9, 8) !=
        LF_INVALID_ARGUMENT) result = 103;
  int32_t status = 0;
  moonbit_bytes_t output = lunaflux_cuda_copy_to_host(allocation, 4, 6, &status);
  if (result == 0 &&
      (status != LF_OK || Moonbit_array_length(output) != 6 ||
       output[0] != 10 || output[5] != 15 ||
       state.copy_to_host_calls != 1)) result = 104;
  moonbit_decref(output);
  output = lunaflux_cuda_copy_to_host(allocation, 12, 6, &status);
  if (result == 0 &&
      (status != LF_INVALID_ARGUMENT || Moonbit_array_length(output) != 0 ||
       state.copy_to_host_calls != 1)) result = 105;
  moonbit_decref(output);
  moonbit_decref(source);
  moonbit_decref(allocation);
  moonbit_decref(context);
  lf_active_transfer_probe = NULL;
  return result;
}

static int32_t lf_test_allocation_retry(void) {
  lf_transfer_probe_state state;
  memset(&state, 0, sizeof(state));
  state.fail_mem_free_once = 1;
  lf_active_transfer_probe = &state;
  lf_cuda_api api;
  lf_transfer_api(&api);
  lf_context *context = lf_transfer_make_context(&api);
  lf_allocation *allocation = lf_transfer_make_allocation(context, 1);
  int32_t result = 0;
  if (lunaflux_cuda_allocation_close(allocation) != LF_DRIVER_FAILURE ||
      atomic_load(&allocation->state) != LF_RESOURCE_LIVE ||
      allocation->handle != LF_TRANSFER_ADDRESS ||
      atomic_load(&context->children) != 1) result = 201;
  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
    result = 202;
  }
  if (result == 0 &&
      (atomic_load(&allocation->state) != LF_RESOURCE_CLOSED ||
       allocation->handle != 0 || atomic_load(&context->children) != 0)) {
    result = 203;
  }
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) result = 204;
  moonbit_decref(allocation);
  moonbit_decref(context);
  lf_active_transfer_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_transfer_boundary(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_transfer_bounds();
    if (result != 0) return result;
    result = lf_test_allocation_retry();
    if (result != 0) return result;
  }
  return LF_OK;
}
