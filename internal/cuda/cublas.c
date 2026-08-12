#include "resource_internal.h"

#include <string.h>

static int32_t lf_close_cublas(lf_child *cublas) {
  if (cublas == NULL) return LF_INVALID_ARGUMENT;
  if (atomic_load(&cublas->state) == LF_RESOURCE_CLOSED) return LF_OK;
  if (atomic_load(&cublas->children) != 0) return LF_BUSY;
  int32_t begin = lf_begin_close(&cublas->state, &cublas->active_operations);
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  if (atomic_load(&cublas->children) != 0) {
    lf_close_failed(&cublas->state);
    return LF_BUSY;
  }
  int32_t result = lf_context_current(cublas->context);
  if (result == LF_OK && cublas->handle != NULL &&
      cublas->context->api->cublasLtDestroy(cublas->handle) != 0) {
    result = LF_DRIVER_FAILURE;
  }
  if (result != LF_OK) {
    lf_close_failed(&cublas->state);
    return result;
  }
  cublas->handle = NULL;
  lf_release_context_child(cublas->context);
  cublas->context = NULL;
  lf_close_succeeded(&cublas->state);
  return LF_OK;
}

static void lf_finalize_cublas(void *object) {
  if (lf_close_cublas((lf_child *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_child *lunaflux_cuda_cublas_create(lf_context *context, int32_t *status) {
  lf_child *cublas = (lf_child *)moonbit_make_external_object(
    lf_finalize_cublas,
    sizeof(lf_child)
  );
  memset(cublas, 0, sizeof(*cublas));
  atomic_init(&cublas->state, LF_RESOURCE_CLOSED);
  atomic_init(&cublas->active_operations, 0);
  atomic_init(&cublas->children, 0);
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return cublas;
  *status = lf_context_current(context);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return cublas;
  }
  if (!context->api->cublas_available) {
    *status = LF_UNSUPPORTED;
    lf_operation_end(&context->active_operations);
    return cublas;
  }
  cublasLtHandle_t handle = NULL;
  if (context->api->cublasLtCreate(&handle) != 0) {
    *status = LF_DRIVER_FAILURE;
    lf_operation_end(&context->active_operations);
    return cublas;
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  cublas->context = context;
  cublas->handle = handle;
  atomic_store(&cublas->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return cublas;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_cublas_close(lf_child *cublas) {
  return lf_close_cublas(cublas);
}
