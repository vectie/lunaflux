#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_FAKE_SET_CURRENT_FAILURE ((void *)(uintptr_t)2)
#define LF_FAKE_CONTEXT_DESTROY_FAILURE ((void *)(uintptr_t)3)
#define LF_FAKE_CHILD_DESTROY_FAILURE ((void *)(uintptr_t)4)
#define LF_FAKE_VALID_HANDLE ((void *)(uintptr_t)16)

int32_t lunaflux_cuda_context_close(lf_context *context);
int32_t lunaflux_cuda_stream_close(lf_child *stream);

static CUresult lf_fake_context_set_current(CUcontext context) {
  return context == LF_FAKE_SET_CURRENT_FAILURE ? 1 : 0;
}

static CUresult lf_fake_context_destroy(CUcontext context) {
  return context == LF_FAKE_CONTEXT_DESTROY_FAILURE ? 1 : 0;
}

static CUresult lf_fake_stream_destroy(CUstream stream) {
  return stream == LF_FAKE_CHILD_DESTROY_FAILURE ? 1 : 0;
}

static void lf_fake_finalizer(void *object) {
  (void)object;
}

static int32_t lf_check_child_live(
  lf_context *context,
  lf_child *child,
  void *expected_handle
) {
  return atomic_load(&child->state) == LF_RESOURCE_LIVE &&
         child->handle == expected_handle && child->context == context &&
         atomic_load(&context->children) == 1;
}

static int32_t lf_test_retry_cycle(int32_t failure_kind) {
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_fake_context_set_current;
  api.cuCtxDestroy = lf_fake_context_destroy;
  api.cuStreamDestroy = lf_fake_stream_destroy;

  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  context->api = &api;
  context->handle = LF_FAKE_VALID_HANDLE;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->children, 1);

  lf_child *child = (lf_child *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_child)
  );
  memset(child, 0, sizeof(*child));
  child->context = context;
  child->handle = failure_kind == 1
    ? LF_FAKE_CHILD_DESTROY_FAILURE
    : LF_FAKE_VALID_HANDLE;
  atomic_init(&child->state, LF_RESOURCE_LIVE);
  moonbit_incref(context);

  int32_t result = 0;
  if (lunaflux_cuda_context_close(context) != LF_BUSY) result = 10;
  if (result == 0 && failure_kind == 2) {
    context->handle = LF_FAKE_SET_CURRENT_FAILURE;
  }
  void *expected_child_handle = child->handle;
  if (result == 0 && lunaflux_cuda_stream_close(child) != LF_DRIVER_FAILURE) {
    result = 11;
  }
  if (result == 0 &&
      !lf_check_child_live(context, child, expected_child_handle)) {
    result = 12;
  }
  context->handle = LF_FAKE_VALID_HANDLE;
  child->handle = LF_FAKE_VALID_HANDLE;
  if (result == 0 && lunaflux_cuda_stream_close(child) != LF_OK) result = 13;
  if (result == 0 &&
      (atomic_load(&child->state) != LF_RESOURCE_CLOSED ||
       child->handle != NULL || child->context != NULL ||
       atomic_load(&context->children) != 0)) {
    result = 14;
  }
  if (result == 0 && lunaflux_cuda_stream_close(child) != LF_OK) result = 15;

  context->handle = LF_FAKE_CONTEXT_DESTROY_FAILURE;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_DRIVER_FAILURE) {
    result = 16;
  }
  if (result == 0 &&
      (atomic_load(&context->state) != LF_RESOURCE_LIVE ||
       context->handle != LF_FAKE_CONTEXT_DESTROY_FAILURE)) {
    result = 17;
  }
  context->handle = LF_FAKE_VALID_HANDLE;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) result = 18;
  if (result == 0 &&
      (atomic_load(&context->state) != LF_RESOURCE_CLOSED ||
       context->handle != NULL)) {
    result = 19;
  }
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) result = 20;

  moonbit_decref(child);
  moonbit_decref(context);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_retryable_lifecycle(
  int32_t cycles,
  int32_t failure_kind
) {
  if (cycles < 1 || cycles > 10000 || failure_kind < 1 || failure_kind > 2) {
    return LF_INVALID_ARGUMENT;
  }
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_retry_cycle(failure_kind);
    if (result != 0) return result;
  }
  return LF_OK;
}
