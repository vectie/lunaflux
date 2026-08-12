#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_TRANSFER_CONTEXT ((void *)(uintptr_t)0x110)
#define LF_TRANSFER_ADDRESS ((CUdeviceptr)0x5000)

typedef struct lf_transfer_probe_state {
  int32_t fail_mem_free_once;
  int32_t copy_to_device_calls;
  int32_t copy_to_host_calls;
  int32_t check_close_interlocks;
  int32_t allocation_close_status;
  int32_t context_close_status;
  const uint8_t *expected_host_source;
  lf_context *active_context;
  lf_allocation *active_allocation;
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
  if (lf_active_transfer_probe->check_close_interlocks != 0) {
    lf_active_transfer_probe->allocation_close_status =
      lunaflux_cuda_allocation_close(
        lf_active_transfer_probe->active_allocation
      );
    lf_active_transfer_probe->context_close_status =
      lunaflux_cuda_context_close(lf_active_transfer_probe->active_context);
  }
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

static int lf_transfer_guards_released(
  lf_context *context,
  lf_allocation *allocation
) {
  return atomic_load(&context->active_operations) == 0 &&
    atomic_load(&allocation->active_operations) == 0;
}

static int32_t lf_test_fixed_transfer(
  uint8_t *source,
  int32_t source_length
) {
  lf_transfer_probe_state state;
  memset(&state, 0, sizeof(state));
  state.allocation_close_status = LF_OK;
  state.context_close_status = LF_OK;
  lf_active_transfer_probe = &state;
  lf_cuda_api api;
  lf_transfer_api(&api);
  lf_context *context = lf_transfer_make_context(&api);
  lf_context *other_context = lf_transfer_make_context(&api);
  lf_allocation *allocation = lf_transfer_make_allocation(context, 0);
  state.expected_host_source = source;
  state.check_close_interlocks = 1;
  state.active_context = context;
  state.active_allocation = allocation;
  int32_t result = 0;
  if (source_length < 16) {
    result = 301;
  } else if (lunaflux_cuda_context_copy_fixed_to_device(
               context,
               allocation,
               source,
               5,
               3,
               8
             ) != LF_OK ||
             state.copy_to_device_calls != 1 ||
             state.allocation_close_status != LF_BUSY ||
             state.context_close_status != LF_BUSY ||
             !lf_transfer_guards_released(context, allocation)) {
    result = 302;
  }
  state.check_close_interlocks = 0;
  if (result == 0 &&
      (lunaflux_cuda_context_copy_fixed_to_device(
         other_context,
         allocation,
         source,
         5,
         3,
         8
       ) != LF_INVALID_ARGUMENT ||
       state.copy_to_device_calls != 1 ||
       !lf_transfer_guards_released(other_context, allocation))) {
    result = 303;
  }
  if (result == 0 &&
      lunaflux_cuda_context_copy_fixed_to_device(
        context, allocation, source, -1, 0, 1
      ) !=
        LF_SIZE_OVERFLOW) result = 304;
  if (result == 0 &&
      lunaflux_cuda_context_copy_fixed_to_device(
        context, allocation, source, source_length - 4, 0, 8
      ) !=
        LF_INVALID_ARGUMENT) result = 305;
  if (result == 0 &&
      lunaflux_cuda_context_copy_fixed_to_device(
        context, allocation, source, 0, 12, 8
      ) !=
        LF_INVALID_ARGUMENT) result = 306;
  atomic_store(&allocation->state, LF_RESOURCE_CLOSING);
  if (result == 0 &&
      lunaflux_cuda_context_copy_fixed_to_device(
        context, allocation, source, 5, 3, 8
      ) !=
        LF_CLOSED) result = 307;
  atomic_store(&allocation->state, LF_RESOURCE_LIVE);
  atomic_store(&context->state, LF_RESOURCE_CLOSING);
  if (result == 0 &&
      lunaflux_cuda_context_copy_fixed_to_device(
        context, allocation, source, 5, 3, 8
      ) !=
        LF_CLOSED) result = 308;
  atomic_store(&context->state, LF_RESOURCE_LIVE);
  if (result == 0 && !lf_transfer_guards_released(context, allocation)) {
    result = 309;
  }
  moonbit_decref(allocation);
  moonbit_decref(other_context);
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

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_fixed_transfer_boundary(
  uint8_t *source,
  int32_t cycles
) {
  if (source == NULL || cycles < 1 || cycles > 10000) {
    return LF_INVALID_ARGUMENT;
  }
  int32_t source_length = Moonbit_array_length(source);
  int32_t reference_count = Moonbit_rc_count(Moonbit_object_header(source));
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_fixed_transfer(source, source_length);
    if (result != 0) return result;
    if (Moonbit_rc_count(Moonbit_object_header(source)) != reference_count) {
      return 310;
    }
  }
  return LF_OK;
}
