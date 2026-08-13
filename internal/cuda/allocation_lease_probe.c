#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_LEASE_CONTEXT ((void *)(uintptr_t)0x7100)
#define LF_LEASE_OTHER_CONTEXT ((void *)(uintptr_t)0x7200)
#define LF_LEASE_ADDRESS ((CUdeviceptr)0x7300)

static void lf_lease_fake_finalizer(void *object) {
  (void)object;
}

static CUresult lf_lease_fake_context_current(CUcontext context) {
  return context == LF_LEASE_CONTEXT || context == LF_LEASE_OTHER_CONTEXT
    ? 0
    : 1;
}

static CUresult lf_lease_fake_context_destroy(CUcontext context) {
  return lf_lease_fake_context_current(context);
}

static CUresult lf_lease_fake_mem_free(CUdeviceptr address) {
  return address == LF_LEASE_ADDRESS ? 0 : 1;
}

static lf_context *lf_lease_make_context(
  lf_cuda_api *api,
  CUcontext handle
) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_lease_fake_finalizer,
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

static lf_allocation *lf_lease_make_allocation(lf_context *context) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_lease_fake_finalizer,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = LF_LEASE_ADDRESS;
  allocation->size = 256;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return allocation;
}

static int32_t lf_test_allocation_lease(void) {
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_lease_fake_context_current;
  api.cuCtxDestroy = lf_lease_fake_context_destroy;
  api.cuMemFree = lf_lease_fake_mem_free;
  lf_context *context = lf_lease_make_context(&api, LF_LEASE_CONTEXT);
  lf_context *other = lf_lease_make_context(&api, LF_LEASE_OTHER_CONTEXT);
  lf_allocation *allocation = lf_lease_make_allocation(context);
  int32_t status = LF_OK;
  int32_t result = 0;

  lf_allocation_lease *first = lunaflux_cuda_allocation_lease_create(
    context,
    allocation,
    &status
  );
  if (status != LF_OK || atomic_load(&allocation->active_operations) != 1) {
    result = 700;
  }
  if (result == 0 &&
      (lunaflux_cuda_allocation_close(allocation) != LF_BUSY ||
       atomic_load(&allocation->state) != LF_RESOURCE_LIVE)) result = 701;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_BUSY) {
    result = 702;
  }

  lf_allocation_lease *second = lunaflux_cuda_allocation_lease_create(
    context,
    allocation,
    &status
  );
  if (result == 0 &&
      (status != LF_OK ||
       atomic_load(&allocation->active_operations) != 2)) result = 703;
  if (result == 0 &&
      (lunaflux_cuda_allocation_lease_close(first) != LF_OK ||
       lunaflux_cuda_allocation_lease_close(first) != LF_OK ||
       atomic_load(&allocation->active_operations) != 1)) result = 704;

  lf_allocation_lease *foreign = lunaflux_cuda_allocation_lease_create(
    other,
    allocation,
    &status
  );
  if (result == 0 &&
      (status != LF_INVALID_ARGUMENT ||
       atomic_load(&allocation->active_operations) != 1 ||
       atomic_load(&other->active_operations) != 0 ||
       lunaflux_cuda_allocation_lease_close(foreign) != LF_OK)) result = 705;

  atomic_store(&context->state, LF_RESOURCE_CLOSING);
  lf_allocation_lease *closing = lunaflux_cuda_allocation_lease_create(
    context,
    allocation,
    &status
  );
  if (result == 0 &&
      (status != LF_CLOSED ||
       atomic_load(&allocation->active_operations) != 1 ||
       lunaflux_cuda_allocation_lease_close(closing) != LF_OK)) result = 706;
  atomic_store(&context->state, LF_RESOURCE_LIVE);

  if (result == 0 &&
      (lunaflux_cuda_allocation_lease_close(second) != LF_OK ||
       atomic_load(&allocation->active_operations) != 0)) result = 707;
  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
    result = 708;
  }
  lf_allocation_lease *closed = lunaflux_cuda_allocation_lease_create(
    context,
    allocation,
    &status
  );
  if (result == 0 &&
      (status != LF_CLOSED ||
       lunaflux_cuda_allocation_lease_close(closed) != LF_OK)) result = 709;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) {
    result = 710;
  }
  if (result == 0 && lunaflux_cuda_context_close(other) != LF_OK) {
    result = 711;
  }

  moonbit_decref(closed);
  moonbit_decref(closing);
  moonbit_decref(foreign);
  moonbit_decref(second);
  moonbit_decref(first);
  moonbit_decref(allocation);
  moonbit_decref(other);
  moonbit_decref(context);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_allocation_lease(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_allocation_lease();
    if (result != 0) return result;
  }
  return LF_OK;
}
