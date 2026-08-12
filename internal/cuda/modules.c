#include "resource_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct lf_module {
  lf_context *context;
  CUmodule handle;
  atomic_int state;
  atomic_int active_operations;
  atomic_int children;
} lf_module;

typedef struct lf_function {
  lf_module *module;
  CUfunction handle;
  atomic_int state;
  atomic_int active_operations;
} lf_function;

static int32_t lf_close_module(lf_module *module) {
  if (module == NULL) return LF_INVALID_ARGUMENT;
  if (atomic_load(&module->state) == LF_RESOURCE_CLOSED) return LF_OK;
  if (atomic_load(&module->children) != 0) return LF_BUSY;
  int32_t begin = lf_begin_close(&module->state, &module->active_operations);
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  if (atomic_load(&module->children) != 0) {
    lf_close_failed(&module->state);
    return LF_BUSY;
  }
  int32_t result = lf_context_current(module->context);
  if (result == LF_OK && module->handle != NULL) {
    result = lf_cuda_map_result(
      module->context->api->cuModuleUnload(module->handle)
    );
  }
  if (result != LF_OK) {
    lf_close_failed(&module->state);
    return result;
  }
  module->handle = NULL;
  lf_release_context_child(module->context);
  module->context = NULL;
  lf_close_succeeded(&module->state);
  return LF_OK;
}

static void lf_finalize_module(void *object) {
  if (lf_close_module((lf_module *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_module *lunaflux_cuda_module_load(
  lf_context *context,
  moonbit_bytes_t image,
  int32_t *status
) {
  lf_module *module = (lf_module *)moonbit_make_external_object(
    lf_finalize_module,
    sizeof(lf_module)
  );
  memset(module, 0, sizeof(*module));
  atomic_init(&module->state, LF_RESOURCE_CLOSED);
  atomic_init(&module->active_operations, 0);
  atomic_init(&module->children, 0);
  if (Moonbit_array_length(image) == 0) {
    *status = LF_INVALID_ARGUMENT;
    return module;
  }
  *status = context == NULL
    ? LF_CLOSED
    : lf_operation_begin(&context->state, &context->active_operations);
  if (*status != LF_OK) return module;
  *status = lf_context_current(context);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return module;
  }
  size_t image_length = (size_t)Moonbit_array_length(image);
  uint8_t *terminated_image = (uint8_t *)malloc(image_length + 1);
  if (terminated_image == NULL) {
    *status = LF_HOST_ALLOCATION_FAILED;
    lf_operation_end(&context->active_operations);
    return module;
  }
  memcpy(terminated_image, image, image_length);
  terminated_image[image_length] = 0;
  CUmodule handle = NULL;
  *status = lf_cuda_map_result(
    context->api->cuModuleLoadData(&handle, terminated_image)
  );
  free(terminated_image);
  if (*status != LF_OK) {
    lf_operation_end(&context->active_operations);
    return module;
  }
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  module->context = context;
  module->handle = handle;
  atomic_store(&module->state, LF_RESOURCE_LIVE);
  lf_operation_end(&context->active_operations);
  return module;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_module_close(lf_module *module) {
  return lf_close_module(module);
}

static int32_t lf_close_function(lf_function *function) {
  if (function == NULL) return LF_INVALID_ARGUMENT;
  int32_t begin = lf_begin_close(
    &function->state,
    &function->active_operations
  );
  if (begin == LF_CLOSED) return LF_OK;
  if (begin != LF_OK) return begin;
  atomic_fetch_sub(&function->module->children, 1);
  moonbit_decref(function->module);
  function->module = NULL;
  function->handle = NULL;
  lf_close_succeeded(&function->state);
  return LF_OK;
}

static void lf_finalize_function(void *object) {
  if (lf_close_function((lf_function *)object) != LF_OK) lf_finalize_failure();
}

MOONBIT_FFI_EXPORT
lf_function *lunaflux_cuda_function_load(
  lf_module *module,
  moonbit_bytes_t name,
  int32_t *status
) {
  lf_function *function = (lf_function *)moonbit_make_external_object(
    lf_finalize_function,
    sizeof(lf_function)
  );
  memset(function, 0, sizeof(*function));
  atomic_init(&function->state, LF_RESOURCE_CLOSED);
  atomic_init(&function->active_operations, 0);
  int32_t name_length = Moonbit_array_length(name);
  if (module == NULL || atomic_load(&module->state) != LF_RESOURCE_LIVE) {
    *status = LF_CLOSED;
    return function;
  }
  if (name_length <= 0 || name_length > 128 ||
      memchr(name, 0, (size_t)name_length) != NULL) {
    *status = LF_INVALID_ARGUMENT;
    return function;
  }
  *status = lf_operation_begin(&module->state, &module->active_operations);
  if (*status != LF_OK) return function;
  *status = lf_context_current(module->context);
  if (*status != LF_OK) {
    lf_operation_end(&module->active_operations);
    return function;
  }
  char terminated_name[129];
  memcpy(terminated_name, name, (size_t)name_length);
  terminated_name[name_length] = '\0';
  CUfunction handle = NULL;
  *status = lf_cuda_map_result(module->context->api->cuModuleGetFunction(
    &handle,
    module->handle,
    terminated_name
  ));
  if (*status != LF_OK) {
    lf_operation_end(&module->active_operations);
    return function;
  }
  moonbit_incref(module);
  atomic_fetch_add(&module->children, 1);
  function->module = module;
  function->handle = handle;
  atomic_store(&function->state, LF_RESOURCE_LIVE);
  lf_operation_end(&module->active_operations);
  return function;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_function_close(lf_function *function) {
  return lf_close_function(function);
}
