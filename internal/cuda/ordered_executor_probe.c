#include "ordered_executor_internal.h"

#include <stdint.h>
#include <string.h>
#if !defined(_WIN32)
#include <pthread.h>
#include <sched.h>
#endif

#define LF_FAKE_CONTEXT ((void *)(uintptr_t)0x100)
#define LF_FAKE_STREAM ((void *)(uintptr_t)0x200)
#define LF_FAKE_EVENT ((void *)(uintptr_t)0x300)
#define LF_FAKE_MODULE ((void *)(uintptr_t)0x400)
#define LF_FAKE_FUNCTION ((void *)(uintptr_t)0x500)
#define LF_FAKE_ALLOCATION ((CUdeviceptr)0x1000)

typedef struct lf_ordered_probe_state {
  int32_t launches;
  int32_t records;
  int32_t queries;
  int32_t event_synchronizes;
  int32_t synchronizes;
  int32_t event_destroys;
  int32_t fail_event_destroy_once;
  int32_t fail_event_create_with_handle;
  atomic_int block_launch;
  atomic_int launch_entered;
  atomic_int release_launch;
  atomic_int block_query;
  atomic_int query_entered;
  atomic_int release_query;
} lf_ordered_probe_state;

static lf_ordered_probe_state *lf_active_ordered_probe;

static void lf_ordered_fake_finalizer(void *object) {
  (void)object;
}

static CUresult lf_ordered_context_current(CUcontext context) {
  return context == LF_FAKE_CONTEXT ? 0 : 1;
}

static CUresult lf_ordered_context_destroy(CUcontext context) {
  return lf_ordered_context_current(context);
}

static CUresult lf_ordered_stream_destroy(CUstream stream) {
  return stream == LF_FAKE_STREAM ? 0 : 1;
}

static CUresult lf_ordered_stream_synchronize(CUstream stream) {
  if (stream != LF_FAKE_STREAM) return 1;
  lf_active_ordered_probe->synchronizes += 1;
  return 0;
}

static CUresult lf_ordered_event_create(CUevent *event, uint32_t flags) {
  if (event == NULL || flags != 0) return 1;
  *event = LF_FAKE_EVENT;
  if (lf_active_ordered_probe->fail_event_create_with_handle != 0) return 1;
  return 0;
}

static CUresult lf_ordered_event_destroy(CUevent event) {
  if (event != LF_FAKE_EVENT) return 1;
  lf_active_ordered_probe->event_destroys += 1;
  if (lf_active_ordered_probe->fail_event_destroy_once != 0) {
    lf_active_ordered_probe->fail_event_destroy_once = 0;
    return 1;
  }
  return 0;
}

static CUresult lf_ordered_event_record(CUevent event, CUstream stream) {
  if (event != LF_FAKE_EVENT || stream != LF_FAKE_STREAM) return 1;
  lf_active_ordered_probe->records += 1;
  return 0;
}

static CUresult lf_ordered_event_query(CUevent event) {
  if (event != LF_FAKE_EVENT) return 1;
  if (atomic_load(&lf_active_ordered_probe->block_query) != 0) {
    atomic_store(&lf_active_ordered_probe->query_entered, 1);
    while (atomic_load(&lf_active_ordered_probe->release_query) == 0) {
#if !defined(_WIN32)
      sched_yield();
#endif
    }
  }
  lf_active_ordered_probe->queries += 1;
  return lf_active_ordered_probe->queries == 1 ? CUDA_ERROR_NOT_READY : 0;
}

static CUresult lf_ordered_event_synchronize(CUevent event) {
  if (event != LF_FAKE_EVENT) return 1;
  lf_active_ordered_probe->event_synchronizes += 1;
  return 0;
}

static CUresult lf_ordered_module_unload(CUmodule module) {
  return module == LF_FAKE_MODULE ? 0 : 1;
}

static CUresult lf_ordered_memory_free(CUdeviceptr allocation) {
  return allocation == LF_FAKE_ALLOCATION ? 0 : 1;
}

static CUresult lf_ordered_launch(
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
  if (function != LF_FAKE_FUNCTION || grid_x != 1 || grid_y != 1 ||
      grid_z != 1 || block_x != 32 || block_y != 1 || block_z != 1 ||
      shared_memory_bytes != 0 || stream != LF_FAKE_STREAM ||
      parameters == NULL || parameters[0] == NULL || extra != NULL) return 1;
  CUdeviceptr value = 0;
  memcpy(&value, parameters[0], sizeof(value));
  if (value != LF_FAKE_ALLOCATION) return 1;
  if (atomic_load(&lf_active_ordered_probe->block_launch) != 0) {
    atomic_store(&lf_active_ordered_probe->launch_entered, 1);
    while (atomic_load(&lf_active_ordered_probe->release_launch) == 0) {
#if !defined(_WIN32)
      sched_yield();
#endif
    }
  }
  lf_active_ordered_probe->launches += 1;
  return 0;
}

static lf_context *lf_ordered_make_context(lf_cuda_api *api) {
  lf_context *context = (lf_context *)moonbit_make_external_object(
    lf_ordered_fake_finalizer, sizeof(lf_context)
  );
  memset(context, 0, sizeof(*context));
  context->api = api;
  context->handle = LF_FAKE_CONTEXT;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  return context;
}

static lf_child *lf_ordered_make_stream(lf_context *context) {
  lf_child *stream = (lf_child *)moonbit_make_external_object(
    lf_ordered_fake_finalizer, sizeof(lf_child)
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

static lf_module *lf_ordered_make_module(lf_context *context) {
  lf_module *module = (lf_module *)moonbit_make_external_object(
    lf_ordered_fake_finalizer, sizeof(lf_module)
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

static lf_function *lf_ordered_make_function(lf_module *module) {
  lf_function *function = (lf_function *)moonbit_make_external_object(
    lf_ordered_fake_finalizer, sizeof(lf_function)
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

static lf_allocation *lf_ordered_make_allocation(lf_context *context) {
  lf_allocation *allocation = (lf_allocation *)moonbit_make_external_object(
    lf_ordered_fake_finalizer, sizeof(lf_allocation)
  );
  memset(allocation, 0, sizeof(*allocation));
  allocation->context = context;
  allocation->handle = LF_FAKE_ALLOCATION;
  allocation->size = 128;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  moonbit_incref(context);
  atomic_fetch_add(&context->children, 1);
  return allocation;
}

#if !defined(_WIN32)
typedef struct lf_ordered_race_call {
  lf_ordered_executor *executor;
  int32_t result;
} lf_ordered_race_call;

static void *lf_ordered_enqueue_thread(void *opaque) {
  lf_ordered_race_call *call = (lf_ordered_race_call *)opaque;
  call->result = lunaflux_cuda_ordered_executor_enqueue(call->executor, 0);
  return NULL;
}

static void *lf_ordered_poll_thread(void *opaque) {
  lf_ordered_race_call *call = (lf_ordered_race_call *)opaque;
  call->result = lunaflux_cuda_ordered_executor_poll(call->executor);
  return NULL;
}
#endif

static int32_t lf_run_ordered_executor_probe(int mode) {
  lf_ordered_probe_state state;
  memset(&state, 0, sizeof(state));
  state.fail_event_destroy_once = mode == 2 || mode == 5;
  state.fail_event_create_with_handle = mode == 4 || mode == 5;
  atomic_init(&state.block_launch, mode == 3);
  atomic_init(&state.launch_entered, 0);
  atomic_init(&state.release_launch, 0);
  atomic_init(&state.block_query, 0);
  atomic_init(&state.query_entered, 0);
  atomic_init(&state.release_query, 0);
  lf_active_ordered_probe = &state;
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_ordered_context_current;
  api.cuCtxDestroy = lf_ordered_context_destroy;
  api.cuStreamDestroy = lf_ordered_stream_destroy;
  api.cuStreamSynchronize = lf_ordered_stream_synchronize;
  api.cuEventCreate = lf_ordered_event_create;
  api.cuEventDestroy = lf_ordered_event_destroy;
  api.cuEventRecord = lf_ordered_event_record;
  api.cuEventQuery = lf_ordered_event_query;
  api.cuEventSynchronize = lf_ordered_event_synchronize;
  api.cuModuleUnload = lf_ordered_module_unload;
  api.cuMemFree = lf_ordered_memory_free;
  api.cuLaunchKernel = lf_ordered_launch;
  lf_context *context = lf_ordered_make_context(&api);
  lf_child *stream = lf_ordered_make_stream(context);
  lf_module *module = lf_ordered_make_module(context);
  lf_function *function = lf_ordered_make_function(module);
  lf_allocation *allocation = lf_ordered_make_allocation(context);
  lf_function **functions =
    (lf_function **)moonbit_make_extern_ref_array_raw(2);
  int32_t *dimensions = moonbit_make_int32_array_raw(14);
  int32_t *starts = moonbit_make_int32_array_raw(3);
  lf_allocation **allocations =
    (lf_allocation **)moonbit_make_extern_ref_array_raw(2);
  int64_t *offsets = moonbit_make_int64_array(2, 0);
  int64_t *byte_counts = moonbit_make_int64_array(2, 16);
  int64_t *alignments = moonbit_make_int64_array(2, 16);
  functions[0] = function;
  functions[1] = function;
  starts[0] = 0;
  starts[1] = 1;
  starts[2] = 2;
  allocations[0] = allocation;
  allocations[1] = allocation;
  for (int32_t index = 0; index < 2; index += 1) {
    int32_t base = index * 7;
    dimensions[base] = 1;
    dimensions[base + 1] = 1;
    dimensions[base + 2] = 1;
    dimensions[base + 3] = 32;
    dimensions[base + 4] = 1;
    dimensions[base + 5] = 1;
    dimensions[base + 6] = 0;
  }
  int32_t status = 0;
  lf_ordered_executor *executor = lunaflux_cuda_ordered_executor_create(
    context, stream, functions, dimensions, starts, allocations, offsets,
    byte_counts, alignments, LF_ORDERED_EAGER_ONLY, &status
  );
  int32_t result = 0;
  if (mode == 4 || mode == 5) {
    if (status != LF_DRIVER_FAILURE || state.event_destroys != 1) result = 128;
    if (result == 0 && mode == 4 &&
        (atomic_load(&executor->state) != LF_RESOURCE_CLOSED ||
         atomic_load(&function->active_operations) != 0 ||
         atomic_load(&stream->active_operations) != 0 ||
         atomic_load(&allocation->active_operations) != 0)) result = 129;
    if (result == 0 && mode == 5 &&
        (atomic_load(&executor->state) != LF_RESOURCE_LIVE ||
         atomic_load(&function->active_operations) == 0 ||
         atomic_load(&stream->active_operations) == 0 ||
         atomic_load(&allocation->active_operations) == 0)) result = 130;
    if (result == 0 && mode == 5 &&
        (lunaflux_cuda_ordered_executor_close(executor) != LF_OK ||
         atomic_load(&executor->state) != LF_RESOURCE_CLOSED ||
         state.event_destroys != 2 ||
         atomic_load(&function->active_operations) != 0 ||
         atomic_load(&stream->active_operations) != 0 ||
         atomic_load(&allocation->active_operations) != 0 ||
         lunaflux_cuda_ordered_executor_close(executor) != LF_OK)) result = 131;
    if (result == 0 && lunaflux_cuda_function_close(function) != LF_OK) result = 132;
    if (result == 0 && lunaflux_cuda_module_close(module) != LF_OK) result = 133;
    if (result == 0 && lunaflux_cuda_stream_close(stream) != LF_OK) result = 134;
    if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
      result = 135;
    }
    if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) {
      result = 136;
    }
    goto cleanup;
  }
  if (status != LF_OK || executor == NULL) result = 100;
  if (result == 0 && lunaflux_cuda_function_close(function) != LF_BUSY) {
    result = 101;
  }
  if (result == 0 && lunaflux_cuda_stream_close(stream) != LF_BUSY) {
    result = 102;
  }
  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_BUSY) {
    result = 103;
  }
  if (result == 0 && lunaflux_cuda_ordered_executor_enqueue(executor, 1) !=
      LF_INVALID_ARGUMENT) result = 104;
  if (result == 0 && mode == 3) {
#if defined(_WIN32)
    if (lunaflux_cuda_ordered_executor_enqueue(executor, 0) != LF_OK) {
      result = 105;
    }
#else
    pthread_t thread;
    lf_ordered_race_call call = { executor, LF_DRIVER_FAILURE };
    int thread_started =
      pthread_create(&thread, NULL, lf_ordered_enqueue_thread, &call) == 0;
    if (!thread_started) {
      result = 117;
    }
    while (result == 0 && atomic_load(&state.launch_entered) == 0) {
      sched_yield();
    }
    if (result == 0 &&
        (lunaflux_cuda_ordered_executor_enqueue(executor, 0) != LF_BUSY ||
         lunaflux_cuda_ordered_executor_record(executor) != LF_BUSY ||
         lunaflux_cuda_ordered_executor_poll(executor) != LF_BUSY ||
         lunaflux_cuda_ordered_executor_wait(executor) != LF_BUSY ||
         lunaflux_cuda_ordered_executor_abort(executor) != LF_BUSY ||
         lunaflux_cuda_ordered_executor_reset(executor) != LF_BUSY)) {
      result = 134;
    }
    if (result == 0 && lunaflux_cuda_ordered_executor_close(executor) !=
        LF_BUSY) result = 118;
    atomic_store(&state.release_launch, 1);
    if (thread_started &&
        (pthread_join(thread, NULL) != 0 || call.result != LF_OK)) result = 119;
#endif
  } else if (result == 0 &&
      lunaflux_cuda_ordered_executor_enqueue(executor, 0) != LF_OK) {
    result = 105;
  }
  if (result == 0 && lunaflux_cuda_ordered_executor_enqueue(executor, 1) !=
      LF_OK) result = 106;
  if (mode == 1) {
    if (result == 0 && lunaflux_cuda_ordered_executor_close(executor) !=
        LF_BUSY) result = 107;
    if (result == 0 && lunaflux_cuda_ordered_executor_abort(executor) !=
        LF_OK) result = 108;
  } else {
    if (result == 0 && lunaflux_cuda_ordered_executor_record(executor) !=
        LF_OK) result = 109;
    if (result == 0 && mode == 3) {
#if defined(_WIN32)
      if (lunaflux_cuda_ordered_executor_poll(executor) != 0) result = 110;
#else
      atomic_store(&state.block_query, 1);
      pthread_t thread;
      lf_ordered_race_call call = { executor, LF_DRIVER_FAILURE };
      int thread_started =
        pthread_create(&thread, NULL, lf_ordered_poll_thread, &call) == 0;
      if (!thread_started) {
        result = 120;
      }
      while (result == 0 && atomic_load(&state.query_entered) == 0) {
        sched_yield();
      }
      if (result == 0 &&
          (lunaflux_cuda_ordered_executor_enqueue(executor, 2) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_record(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_poll(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_wait(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_abort(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_reset(executor) != LF_BUSY)) {
        result = 135;
      }
      if (result == 0 && lunaflux_cuda_ordered_executor_close(executor) !=
          LF_BUSY) result = 121;
      atomic_store(&state.release_query, 1);
      if (thread_started &&
          (pthread_join(thread, NULL) != 0 || call.result != 0)) result = 122;
      if (result == 0 && lunaflux_cuda_ordered_executor_poll(executor) != 1) {
        result = 111;
      }
#endif
    } else if (mode == 0) {
      if (result == 0 &&
          lunaflux_cuda_ordered_executor_wait(executor) != LF_OK) result = 136;
    } else {
      if (result == 0 && lunaflux_cuda_ordered_executor_poll(executor) != 0) {
        result = 110;
      }
      if (result == 0 && lunaflux_cuda_ordered_executor_poll(executor) != 1) {
        result = 111;
      }
    }
  }
  if (result == 0) {
    int32_t closed = lunaflux_cuda_ordered_executor_close(executor);
    if (mode == 2) {
      if (closed != LF_DRIVER_FAILURE ||
          atomic_load(&function->active_operations) == 0 ||
          atomic_load(&stream->active_operations) == 0 ||
          atomic_load(&allocation->active_operations) == 0 ||
          lunaflux_cuda_ordered_executor_close(executor) != LF_OK) {
        result = 112;
      }
    } else if (closed != LF_OK) {
      result = 112;
    }
  }
  int32_t expected_destroys = mode == 2 ? 2 : 1;
  if (result == 0 &&
      (state.launches != 2 || state.event_destroys != expected_destroys)) {
    result = 113;
  }
  if (result == 0 && mode == 1 && state.synchronizes != 1) result = 114;
  if (result == 0 && mode == 0 &&
      (state.records != 1 || state.queries != 0 ||
       state.event_synchronizes != 1 || state.synchronizes != 0)) {
    result = 115;
  }
  if (result == 0 && mode != 0 && mode != 1 &&
      (state.records != 1 || state.queries != 2 || state.synchronizes != 0)) {
    result = 115;
  }
  if (result == 0 && (atomic_load(&function->active_operations) != 0 ||
      atomic_load(&stream->active_operations) != 0 ||
      atomic_load(&allocation->active_operations) != 0)) result = 116;
  if (result == 0 && lunaflux_cuda_function_close(function) != LF_OK) {
    result = 123;
  }
  if (result == 0 && lunaflux_cuda_module_close(module) != LF_OK) result = 124;
  if (result == 0 && lunaflux_cuda_stream_close(stream) != LF_OK) result = 125;
  if (result == 0 && lunaflux_cuda_allocation_close(allocation) != LF_OK) {
    result = 126;
  }
  if (result == 0 && lunaflux_cuda_context_close(context) != LF_OK) {
    result = 127;
  }

cleanup:
  moonbit_decref(executor);
  moonbit_decref(alignments);
  moonbit_decref(byte_counts);
  moonbit_decref(offsets);
  moonbit_decref(allocations);
  moonbit_decref(starts);
  moonbit_decref(dimensions);
  moonbit_decref(functions);
  moonbit_decref(allocation);
  moonbit_decref(function);
  moonbit_decref(module);
  moonbit_decref(stream);
  moonbit_decref(context);
  lf_active_ordered_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_ordered_executor(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t result = lf_run_ordered_executor_probe(cycle % 6);
    if (result != 0) return result;
  }
  return LF_OK;
}
