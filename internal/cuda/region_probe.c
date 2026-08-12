#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_REGION_CONTEXT ((void *)(uintptr_t)0x5100)
#define LF_REGION_OTHER_CONTEXT ((void *)(uintptr_t)0x5200)
#define LF_REGION_BASE ((CUdeviceptr)0x1003)

static void lf_region_fake_finalizer(void *object) {
  (void)object;
}

static CUresult lf_region_fake_context_current(CUcontext context) {
  return context == LF_REGION_CONTEXT || context == LF_REGION_OTHER_CONTEXT
    ? 0
    : 1;
}

static CUresult lf_region_fake_context_destroy(CUcontext context) {
  return lf_region_fake_context_current(context);
}

static CUresult lf_region_fake_mem_free(CUdeviceptr address) {
  return address == LF_REGION_BASE ? 0 : 1;
}

static lf_context *lf_region_make_context(
  lf_cuda_api *api,
  CUcontext handle
) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_region_fake_finalizer,
    sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  context->api = api;
  context->handle = handle;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  return context;
}

static lf_allocation *lf_region_make_allocation(lf_context *context) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_region_fake_finalizer,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = LF_REGION_BASE;
  allocation->size = 128;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return allocation;
}

static int lf_region_guards_released(
  lf_context *context,
  lf_allocation *allocation
) {
  return atomic_load(&context->active_operations) == 0 &&
    atomic_load(&allocation->active_operations) == 0;
}

static int32_t lf_test_region_validation(void) {
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_region_fake_context_current;
  api.cuCtxDestroy = lf_region_fake_context_destroy;
  api.cuMemFree = lf_region_fake_mem_free;
  lf_context *context = lf_region_make_context(&api, LF_REGION_CONTEXT);
  lf_context *other_context = lf_region_make_context(
    &api,
    LF_REGION_OTHER_CONTEXT
  );
  lf_allocation *allocation = lf_region_make_allocation(context);
  int32_t result = 0;

  if (lunaflux_cuda_context_validate_allocation_region(
        context,
        allocation,
        13,
        16,
        16
      ) != LF_OK || !lf_region_guards_released(context, allocation)) {
    result = 500;
  }
  if (result == 0 &&
      (lunaflux_cuda_context_validate_allocation_region(
         context,
         allocation,
         16,
         16,
         16
       ) != LF_INVALID_ARGUMENT ||
       !lf_region_guards_released(context, allocation))) result = 501;
  if (result == 0 &&
      (lunaflux_cuda_context_validate_allocation_region(
         other_context,
         allocation,
         13,
         16,
         16
       ) != LF_INVALID_ARGUMENT ||
       !lf_region_guards_released(other_context, allocation))) result = 514;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 120, 16, 1
      ) !=
        LF_INVALID_ARGUMENT) result = 502;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, -1, 16, 1
      ) !=
        LF_INVALID_ARGUMENT) result = 503;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 0, 0, 1
      ) !=
        LF_INVALID_ARGUMENT) result = 504;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 0, 1, 3
      ) !=
        LF_INVALID_ARGUMENT) result = 505;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 0, 1, 8192
      ) !=
        LF_INVALID_ARGUMENT) result = 506;
  allocation->handle = UINT64_MAX - 8;
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 16, 1, 1
      ) !=
        LF_SIZE_OVERFLOW) result = 507;
  allocation->handle = LF_REGION_BASE;

  atomic_store(&allocation->active_operations, 1);
  if (result == 0 &&
      (lunaflux_cuda_allocation_close(allocation) != LF_BUSY ||
       atomic_load(&allocation->state) != LF_RESOURCE_LIVE)) result = 508;
  atomic_store(&allocation->active_operations, 0);
  atomic_store(&context->state, LF_RESOURCE_CLOSING);
  if (result == 0 &&
      (lunaflux_cuda_context_validate_allocation_region(
         context, allocation, 13, 16, 16
       ) !=
         LF_CLOSED ||
       !lf_region_guards_released(context, allocation))) result = 509;
  atomic_store(&context->state, LF_RESOURCE_LIVE);
  atomic_store(&allocation->state, LF_RESOURCE_CLOSING);
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 13, 16, 16
      ) !=
        LF_CLOSED) result = 510;
  atomic_store(&allocation->state, LF_RESOURCE_LIVE);

  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
    result = 511;
  }
  if (result == 0 &&
      lunaflux_cuda_context_validate_allocation_region(
        context, allocation, 0, 1, 1
      ) !=
        LF_CLOSED) result = 512;
  atomic_store(&context->active_operations, 1);
  if (result == 0 &&
      (lunaflux_cuda_context_close(context) != LF_BUSY ||
       atomic_load(&context->state) != LF_RESOURCE_LIVE)) result = 516;
  atomic_store(&context->active_operations, 0);
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) {
    result = 513;
  }
  if (result == 0 && lunaflux_cuda_context_close(other_context) != LF_OK) {
    result = 515;
  }
  moonbit_decref(allocation);
  moonbit_decref(other_context);
  moonbit_decref(context);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_region_validation(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_region_validation();
    if (result != 0) return result;
  }
  return LF_OK;
}
