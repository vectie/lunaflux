#include "resource_internal.h"

#include <stdint.h>
#include <string.h>

#define LF_FAKE_CONTEXT ((void *)(uintptr_t)0x100)
#define LF_FAKE_OTHER_CONTEXT ((void *)(uintptr_t)0x101)
#define LF_FAKE_STREAM ((void *)(uintptr_t)0x200)
#define LF_FAKE_MODULE ((void *)(uintptr_t)0x300)
#define LF_FAKE_FUNCTION ((void *)(uintptr_t)0x400)
#define LF_FAKE_FIRST ((CUdeviceptr)0x1000)
#define LF_FAKE_SECOND ((CUdeviceptr)0x2000)

typedef struct lf_launch_probe_state {
  int32_t launch_calls;
  int32_t synchronize_calls;
  int32_t module_unloads;
  int32_t close_race_failures;
  int32_t fail_launch_once;
  int32_t fail_synchronize_once;
  lf_function *function;
  lf_module *module;
  lf_child *stream;
  lf_allocation *first;
  lf_allocation *second;
} lf_launch_probe_state;

typedef struct lf_probe_arguments {
  lf_allocation **allocations;
  int64_t *offsets;
  int64_t *byte_counts;
  int64_t *alignments;
} lf_probe_arguments;

static lf_launch_probe_state *lf_active_launch_probe;

static void lf_fake_finalizer(void *object) {
  (void)object;
}

static CUresult lf_fake_context_current(CUcontext context) {
  return context == LF_FAKE_CONTEXT || context == LF_FAKE_OTHER_CONTEXT
    ? 0
    : 1;
}

static CUresult lf_fake_context_destroy(CUcontext context) {
  return lf_fake_context_current(context);
}

static CUresult lf_fake_stream_destroy(CUstream stream) {
  return stream == LF_FAKE_STREAM ? 0 : 1;
}

static CUresult lf_fake_stream_synchronize(CUstream stream) {
  if (stream != LF_FAKE_STREAM) return 1;
  lf_active_launch_probe->synchronize_calls += 1;
  if (lf_active_launch_probe->fail_synchronize_once != 0) {
    lf_active_launch_probe->fail_synchronize_once = 0;
    return 1;
  }
  return 0;
}

static CUresult lf_fake_module_unload(CUmodule module) {
  if (module != LF_FAKE_MODULE) return 1;
  lf_active_launch_probe->module_unloads += 1;
  return 0;
}

static CUresult lf_fake_mem_free(CUdeviceptr address) {
  return address == LF_FAKE_FIRST || address == LF_FAKE_SECOND ? 0 : 1;
}

static int lf_launch_interlocks_hold(void) {
  lf_launch_probe_state *state = lf_active_launch_probe;
  if (atomic_load(&state->function->active_operations) != 1 ||
      atomic_load(&state->module->active_operations) != 1 ||
      atomic_load(&state->stream->active_operations) != 1 ||
      atomic_load(&state->first->active_operations) != 2 ||
      atomic_load(&state->second->active_operations) != 1) {
    return 0;
  }
  return lunaflux_cuda_function_close(state->function) == LF_BUSY &&
    lunaflux_cuda_module_close(state->module) == LF_BUSY &&
    lunaflux_cuda_stream_close(state->stream) == LF_BUSY &&
    lunaflux_cuda_allocation_close(state->first) == LF_BUSY &&
    lunaflux_cuda_allocation_close(state->second) == LF_BUSY;
}

static CUdeviceptr lf_parameter_value(void *parameter) {
  CUdeviceptr value = 0;
  memcpy(&value, parameter, sizeof(value));
  return value;
}

static CUresult lf_fake_launch(
  CUfunction function,
  uint32_t grid_x,
  uint32_t grid_y,
  uint32_t grid_z,
  uint32_t block_x,
  uint32_t block_y,
  uint32_t block_z,
  uint32_t shared_memory_bytes,
  CUstream stream,
  void **parameters,
  void **extra
) {
  lf_launch_probe_state *state = lf_active_launch_probe;
  state->launch_calls += 1;
  if (function != LF_FAKE_FUNCTION || grid_x != 3 || grid_y != 2 ||
      grid_z != 1 || block_x != 8 || block_y != 4 || block_z != 2 ||
      shared_memory_bytes != 64 || stream != LF_FAKE_STREAM ||
      parameters == NULL || extra != NULL || parameters[0] == NULL ||
      parameters[1] == NULL || parameters[2] == NULL ||
      parameters[0] == parameters[1] || parameters[0] == parameters[2] ||
      parameters[1] == parameters[2] ||
      lf_parameter_value(parameters[0]) != LF_FAKE_FIRST + 16 ||
      lf_parameter_value(parameters[1]) != LF_FAKE_SECOND + 32 ||
      lf_parameter_value(parameters[2]) != LF_FAKE_FIRST + 64) {
    return 1;
  }
  if (!lf_launch_interlocks_hold()) {
    state->close_race_failures += 1;
    return 1;
  }
  if (state->fail_launch_once != 0) {
    state->fail_launch_once = 0;
    return 1;
  }
  return 0;
}

static lf_context *lf_make_context(lf_cuda_api *api, CUcontext handle) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_fake_finalizer,
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

static lf_module *lf_make_module(lf_context *context) {
  lf_module *module = (lf_module *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_module)
  );
  memset(module, 0, sizeof(*module));
  module->context = context;
  module->handle = LF_FAKE_MODULE;
  atomic_init(&module->state, LF_RESOURCE_LIVE);
  atomic_init(&module->active_operations, 0);
  atomic_init(&module->children, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return module;
}

static lf_function *lf_make_function(lf_module *module) {
  lf_function *function = (lf_function *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_function)
  );
  memset(function, 0, sizeof(*function));
  function->module = module;
  function->handle = LF_FAKE_FUNCTION;
  atomic_init(&function->state, LF_RESOURCE_LIVE);
  atomic_init(&function->active_operations, 0);
  moonbit_incref(module);
  atomic_fetch_add(&module->children, 1);
  return function;
}

static lf_child *lf_make_stream(lf_context *context) {
  lf_child *stream = (lf_child *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_child)
  );
  memset(stream, 0, sizeof(*stream));
  stream->context = context;
  stream->handle = LF_FAKE_STREAM;
  atomic_init(&stream->state, LF_RESOURCE_LIVE);
  atomic_init(&stream->active_operations, 0);
  atomic_init(&stream->children, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return stream;
}

static lf_allocation *lf_make_allocation(
  lf_context *context,
  CUdeviceptr handle
) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_fake_finalizer,
    sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = handle;
  allocation->size = 128;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return allocation;
}

static lf_probe_arguments lf_make_arguments(
  lf_allocation *first,
  lf_allocation *second,
  lf_allocation *third
) {
  lf_probe_arguments result;
  result.allocations = (lf_allocation **)moonbit_make_extern_ref_array_raw(3);
  result.offsets = moonbit_make_int64_array_raw(3);
  result.byte_counts = moonbit_make_int64_array_raw(3);
  result.alignments = moonbit_make_int64_array_raw(3);
  result.allocations[0] = first;
  result.allocations[1] = second;
  result.allocations[2] = third;
  result.offsets[0] = 16;
  result.offsets[1] = 32;
  result.offsets[2] = 64;
  result.byte_counts[0] = 32;
  result.byte_counts[1] = 16;
  result.byte_counts[2] = 16;
  result.alignments[0] = 16;
  result.alignments[1] = 32;
  result.alignments[2] = 64;
  return result;
}

static void lf_release_arguments(lf_probe_arguments arguments) {
  moonbit_decref(arguments.alignments);
  moonbit_decref(arguments.byte_counts);
  moonbit_decref(arguments.offsets);
  moonbit_decref(arguments.allocations);
}

static int32_t lf_run_launch(
  lf_launch_probe_state *state,
  lf_probe_arguments arguments
) {
  (void)state;
  return lunaflux_cuda_function_launch(
    lf_active_launch_probe->function,
    lf_active_launch_probe->stream,
    3,
    2,
    1,
    8,
    4,
    2,
    64,
    arguments.allocations,
    arguments.offsets,
    arguments.byte_counts,
    arguments.alignments
  );
}

static int32_t lf_test_launch_boundary(void) {
  lf_launch_probe_state state;
  memset(&state, 0, sizeof(state));
  lf_active_launch_probe = &state;
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_fake_context_current;
  api.cuCtxDestroy = lf_fake_context_destroy;
  api.cuStreamDestroy = lf_fake_stream_destroy;
  api.cuStreamSynchronize = lf_fake_stream_synchronize;
  api.cuModuleUnload = lf_fake_module_unload;
  api.cuMemFree = lf_fake_mem_free;
  api.cuLaunchKernel = lf_fake_launch;

  lf_context *context = lf_make_context(&api, LF_FAKE_CONTEXT);
  lf_context *other_context = lf_make_context(&api, LF_FAKE_OTHER_CONTEXT);
  state.module = lf_make_module(context);
  state.function = lf_make_function(state.module);
  state.stream = lf_make_stream(context);
  state.first = lf_make_allocation(context, LF_FAKE_FIRST);
  state.second = lf_make_allocation(context, LF_FAKE_SECOND);
  lf_allocation *other = lf_make_allocation(other_context, LF_FAKE_SECOND);
  lf_probe_arguments valid = lf_make_arguments(
    state.first,
    state.second,
    state.first
  );
  lf_probe_arguments mismatch = lf_make_arguments(
    state.first,
    state.second,
    other
  );

  int32_t result = 0;
  lf_allocation **too_many =
    (lf_allocation **)moonbit_make_extern_ref_array_raw(33);
  int64_t *too_many_values = moonbit_make_int64_array(33, 0);
  if (lunaflux_cuda_function_launch(
        state.function,
        state.stream,
        3,
        2,
        1,
        8,
        4,
        2,
        64,
        too_many,
        too_many_values,
        too_many_values,
        too_many_values
      ) != LF_INVALID_ARGUMENT || state.launch_calls != 0) result = 100;
  moonbit_decref(too_many_values);
  moonbit_decref(too_many);
  int64_t *short_alignments = moonbit_make_int64_array(2, 16);
  if (result == 0 &&
      (lunaflux_cuda_function_launch(
         state.function,
         state.stream,
         3,
         2,
         1,
         8,
         4,
         2,
         64,
         valid.allocations,
         valid.offsets,
         valid.byte_counts,
         short_alignments
       ) != LF_INVALID_ARGUMENT || state.launch_calls != 0)) result = 120;
  moonbit_decref(short_alignments);
  if (lunaflux_cuda_function_launch(
        state.function,
        state.stream,
        0,
        2,
        1,
        8,
        4,
        2,
        64,
        valid.allocations,
        valid.offsets,
        valid.byte_counts,
        valid.alignments
      ) != LF_INVALID_ARGUMENT || state.launch_calls != 0) result = 101;
  valid.offsets[0] = 120;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_INVALID_ARGUMENT ||
       state.launch_calls != 0)) result = 102;
  valid.offsets[0] = 17;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_INVALID_ARGUMENT ||
       state.launch_calls != 0)) result = 103;
  valid.offsets[0] = 16;
  if (result == 0 &&
      (lf_run_launch(&state, mismatch) != LF_INVALID_ARGUMENT ||
       state.launch_calls != 0)) result = 104;
  state.first->handle = LF_FAKE_FIRST + 1;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_INVALID_ARGUMENT ||
       state.launch_calls != 0)) result = 124;
  state.first->handle = LF_FAKE_FIRST;
  atomic_store(&state.first->state, LF_RESOURCE_CLOSING);
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_CLOSED ||
       state.launch_calls != 0)) result = 125;
  atomic_store(&state.first->state, LF_RESOURCE_LIVE);
  state.first->handle = UINT64_MAX - 8;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_SIZE_OVERFLOW ||
       state.launch_calls != 0)) result = 123;
  state.first->handle = LF_FAKE_FIRST;
  state.fail_launch_once = 1;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_DRIVER_FAILURE ||
       state.launch_calls != 1 || state.synchronize_calls != 0)) result = 105;
  state.fail_synchronize_once = 1;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_DRIVER_FAILURE ||
       state.launch_calls != 2 || state.synchronize_calls != 1)) result = 106;
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_OK || state.launch_calls != 3 ||
       state.synchronize_calls != 2 || state.close_race_failures != 0)) {
    result = 107;
  }
  if (result == 0 &&
      (atomic_load(&state.function->active_operations) != 0 ||
       atomic_load(&state.module->active_operations) != 0 ||
       atomic_load(&state.stream->active_operations) != 0 ||
       atomic_load(&state.first->active_operations) != 0 ||
       atomic_load(&state.second->active_operations) != 0)) result = 121;
  if (result == 0 && lunaflux_cuda_function_close(state.function) != LF_OK) {
    result = 108;
  }
  if (result == 0 &&
      (lf_run_launch(&state, valid) != LF_CLOSED || state.launch_calls != 3 ||
       state.synchronize_calls != 2)) result = 122;
  if (result == 0 && atomic_load(&state.module->children) != 0) result = 109;
  if (result == 0 && lunaflux_cuda_module_close(state.module) != LF_OK) {
    result = 110;
  }
  if (result == 0 && state.module_unloads != 1) result = 111;
  if (result == 0 && lunaflux_cuda_stream_close(state.stream) != LF_OK) {
    result = 112;
  }
  if (result == 0 && lunaflux_cuda_allocation_close(state.first) != LF_OK) {
    result = 113;
  }
  if (result == 0 && lunaflux_cuda_allocation_close(state.second) != LF_OK) {
    result = 114;
  }
  if (result == 0 && lunaflux_cuda_allocation_close(other) != LF_OK) {
    result = 115;
  }
  if (result == 0 && atomic_load(&context->children) != 0) result = 116;
  if (result == 0 && atomic_load(&other_context->children) != 0) result = 117;
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) {
    result = 118;
  }
  if (result == 0 && lunaflux_cuda_context_close(other_context) != LF_OK) {
    result = 119;
  }

  lf_release_arguments(mismatch);
  lf_release_arguments(valid);
  moonbit_decref(other);
  moonbit_decref(state.second);
  moonbit_decref(state.first);
  moonbit_decref(state.stream);
  moonbit_decref(state.function);
  moonbit_decref(state.module);
  moonbit_decref(other_context);
  moonbit_decref(context);
  lf_active_launch_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_aot_launch_boundary(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_test_launch_boundary();
    if (result != 0) return result;
  }
  return LF_OK;
}
