#include "ordered_executor_internal.h"

#include <stdint.h>
#include <string.h>
#if !defined(_WIN32)
#include <pthread.h>
#include <sched.h>
#endif

#define LF_GRAPH_CONTEXT ((void *)(uintptr_t)0x6100)
#define LF_GRAPH_STREAM ((void *)(uintptr_t)0x6200)
#define LF_GRAPH_EVENT ((void *)(uintptr_t)0x6300)
#define LF_GRAPH_MODULE ((void *)(uintptr_t)0x6400)
#define LF_GRAPH_FUNCTION ((void *)(uintptr_t)0x6500)
#define LF_GRAPH ((void *)(uintptr_t)0x6600)
#define LF_GRAPH_EXEC ((void *)(uintptr_t)0x6700)
#define LF_GRAPH_ALLOCATION ((CUdeviceptr)0x6800)

typedef struct lf_graph_probe_state {
  int32_t begin_captures;
  int32_t end_captures;
  int32_t captured_launches;
  int32_t graph_instantiations;
  int32_t graph_destroys;
  int32_t graph_launches;
  int32_t graph_exec_destroys;
  int32_t event_records;
  int32_t event_waits;
  int32_t event_destroys;
  int32_t fail_event_destroy_once;
  int32_t fail_graph_destroy_once;
  int32_t fail_graph_exec_destroy_once;
  int32_t fail_graph_instantiate_with_exec;
  int32_t capturing;
  atomic_int block_graph_launch;
  atomic_int graph_launch_entered;
  atomic_int release_graph_launch;
} lf_graph_probe_state;

static lf_graph_probe_state *lf_active_graph_probe;

static void lf_graph_noop_finalizer(void *object) {
  (void)object;
}

static CUresult lf_graph_context_current(CUcontext context) {
  return context == LF_GRAPH_CONTEXT ? 0 : 1;
}

static CUresult lf_graph_stream_sync(CUstream stream) {
  return stream == LF_GRAPH_STREAM ? 0 : 1;
}

static CUresult lf_graph_event_create(CUevent *event, uint32_t flags) {
  if (event == NULL || flags != 0) return 1;
  *event = LF_GRAPH_EVENT;
  return 0;
}

static CUresult lf_graph_event_destroy(CUevent event) {
  if (event != LF_GRAPH_EVENT) return 1;
  lf_active_graph_probe->event_destroys += 1;
  if (lf_active_graph_probe->fail_event_destroy_once != 0) {
    lf_active_graph_probe->fail_event_destroy_once = 0;
    return 1;
  }
  return 0;
}

static CUresult lf_graph_event_record(CUevent event, CUstream stream) {
  if (event != LF_GRAPH_EVENT || stream != LF_GRAPH_STREAM) return 1;
  lf_active_graph_probe->event_records += 1;
  return 0;
}

static CUresult lf_graph_event_wait(CUevent event) {
  if (event != LF_GRAPH_EVENT) return 1;
  lf_active_graph_probe->event_waits += 1;
  return 0;
}

static CUresult lf_graph_begin_capture(CUstream stream, int32_t mode) {
  if (stream != LF_GRAPH_STREAM || mode != 1) return 1;
  lf_active_graph_probe->begin_captures += 1;
  lf_active_graph_probe->capturing = 1;
  return 0;
}

static CUresult lf_graph_end_capture(CUstream stream, CUgraph *graph) {
  if (stream != LF_GRAPH_STREAM || graph == NULL) return 1;
  lf_active_graph_probe->end_captures += 1;
  lf_active_graph_probe->capturing = 0;
  *graph = LF_GRAPH;
  return 0;
}

static CUresult lf_graph_instantiate(
  CUgraphExec *graph_exec,
  CUgraph graph,
  uint64_t flags
) {
  if (graph_exec == NULL || graph != LF_GRAPH || flags != 0) return 1;
  lf_active_graph_probe->graph_instantiations += 1;
  *graph_exec = LF_GRAPH_EXEC;
  if (lf_active_graph_probe->fail_graph_instantiate_with_exec != 0) return 1;
  return 0;
}

static CUresult lf_graph_destroy(CUgraph graph) {
  if (graph != LF_GRAPH) return 1;
  lf_active_graph_probe->graph_destroys += 1;
  if (lf_active_graph_probe->fail_graph_destroy_once != 0) {
    lf_active_graph_probe->fail_graph_destroy_once = 0;
    return 1;
  }
  return 0;
}

static CUresult lf_graph_exec_destroy(CUgraphExec graph_exec) {
  if (graph_exec != LF_GRAPH_EXEC) return 1;
  lf_active_graph_probe->graph_exec_destroys += 1;
  if (lf_active_graph_probe->fail_graph_exec_destroy_once != 0) {
    lf_active_graph_probe->fail_graph_exec_destroy_once = 0;
    return 1;
  }
  return 0;
}

static CUresult lf_graph_launch(CUgraphExec graph_exec, CUstream stream) {
  if (graph_exec != LF_GRAPH_EXEC || stream != LF_GRAPH_STREAM) return 1;
  if (atomic_load(&lf_active_graph_probe->block_graph_launch) != 0) {
    atomic_store(&lf_active_graph_probe->graph_launch_entered, 1);
    while (atomic_load(&lf_active_graph_probe->release_graph_launch) == 0) {
#if !defined(_WIN32)
      sched_yield();
#endif
    }
  }
  lf_active_graph_probe->graph_launches += 1;
  return 0;
}

static CUresult lf_graph_kernel_launch(
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
  if (function != LF_GRAPH_FUNCTION || grid_x != 1 || grid_y != 1 ||
      grid_z != 1 || block_x != 32 || block_y != 1 || block_z != 1 ||
      shared_memory_bytes != 0 || stream != LF_GRAPH_STREAM ||
      parameters == NULL || parameters[0] == NULL || extra != NULL) return 1;
  CUdeviceptr address = 0;
  memcpy(&address, parameters[0], sizeof(address));
  if (address != LF_GRAPH_ALLOCATION) return 1;
  if (lf_active_graph_probe->capturing != 0) {
    lf_active_graph_probe->captured_launches += 1;
  }
  return 0;
}

static void *lf_graph_object(uint32_t size) {
  void *object = moonbit_make_external_object(lf_graph_noop_finalizer, size);
  memset(object, 0, size);
  return object;
}

#if !defined(_WIN32)
typedef struct lf_graph_race_call {
  lf_ordered_executor *executor;
  int32_t result;
} lf_graph_race_call;

static void *lf_graph_launch_thread(void *opaque) {
  lf_graph_race_call *call = (lf_graph_race_call *)opaque;
  call->result =
    lunaflux_cuda_ordered_executor_launch_captured(call->executor);
  return NULL;
}
#endif

static int32_t lf_run_graph_probe(int32_t mode) {
  lf_graph_probe_state state;
  memset(&state, 0, sizeof(state));
  state.fail_event_destroy_once = mode == 7;
  state.fail_graph_destroy_once = mode == 6;
  state.fail_graph_exec_destroy_once = mode == 4 || mode == 6;
  state.fail_graph_instantiate_with_exec = mode == 6 || mode == 7;
#if defined(_WIN32)
  atomic_init(&state.block_graph_launch, 0);
#else
  atomic_init(&state.block_graph_launch, mode == 8);
#endif
  atomic_init(&state.graph_launch_entered, 0);
  atomic_init(&state.release_graph_launch, 0);
  lf_active_graph_probe = &state;
  lf_cuda_api api;
  memset(&api, 0, sizeof(api));
  api.cuCtxSetCurrent = lf_graph_context_current;
  api.cuStreamSynchronize = lf_graph_stream_sync;
  api.cuEventCreate = lf_graph_event_create;
  api.cuEventDestroy = lf_graph_event_destroy;
  api.cuEventRecord = lf_graph_event_record;
  api.cuEventSynchronize = lf_graph_event_wait;
  api.cuLaunchKernel = lf_graph_kernel_launch;
  if (mode != 2 && mode != 5) {
    api.graph_available = 1;
    api.cuStreamBeginCapture = lf_graph_begin_capture;
    api.cuStreamEndCapture = lf_graph_end_capture;
    api.cuGraphInstantiateWithFlags = lf_graph_instantiate;
    api.cuGraphDestroy = lf_graph_destroy;
    api.cuGraphExecDestroy = lf_graph_exec_destroy;
    api.cuGraphLaunch = lf_graph_launch;
  }
  lf_context *context = lf_graph_object(sizeof(lf_context));
  context->api = &api;
  context->handle = LF_GRAPH_CONTEXT;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 0);
  lf_child *stream = lf_graph_object(sizeof(lf_child));
  stream->context = context;
  stream->handle = LF_GRAPH_STREAM;
  atomic_init(&stream->state, LF_RESOURCE_LIVE);
  atomic_init(&stream->active_operations, 0);
  atomic_init(&stream->children, 0);
  lf_module *module = lf_graph_object(sizeof(lf_module));
  module->context = context;
  module->handle = LF_GRAPH_MODULE;
  atomic_init(&module->state, LF_RESOURCE_LIVE);
  atomic_init(&module->active_operations, 0);
  atomic_init(&module->children, 0);
  lf_function *function = lf_graph_object(sizeof(lf_function));
  function->module = module;
  function->handle = LF_GRAPH_FUNCTION;
  atomic_init(&function->state, LF_RESOURCE_LIVE);
  atomic_init(&function->active_operations, 0);
  lf_allocation *allocation = lf_graph_object(sizeof(lf_allocation));
  allocation->context = context;
  allocation->handle = LF_GRAPH_ALLOCATION;
  allocation->size = 64;
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  lf_function **functions =
    (lf_function **)moonbit_make_extern_ref_array_raw(1);
  int32_t *dimensions = moonbit_make_int32_array_raw(7);
  int32_t *starts = moonbit_make_int32_array_raw(2);
  lf_allocation **allocations =
    (lf_allocation **)moonbit_make_extern_ref_array_raw(1);
  int64_t *offsets = moonbit_make_int64_array(1, 0);
  int64_t *byte_counts = moonbit_make_int64_array(1, 16);
  int64_t *alignments = moonbit_make_int64_array(1, 16);
  functions[0] = function;
  dimensions[0] = 1;
  dimensions[1] = 1;
  dimensions[2] = 1;
  dimensions[3] = 32;
  dimensions[4] = 1;
  dimensions[5] = 1;
  dimensions[6] = 0;
  starts[0] = 0;
  starts[1] = 1;
  allocations[0] = allocation;
  int32_t policy = mode == 1
    ? LF_ORDERED_EAGER_ONLY
    : (mode == 2
      ? LF_ORDERED_CAPTURE_WITH_EAGER_FALLBACK
      : LF_ORDERED_CAPTURE_REQUIRED);
  int32_t status = LF_OK;
  lf_ordered_executor *executor = lunaflux_cuda_ordered_executor_create(
    context, stream, functions, dimensions, starts, allocations, offsets,
    byte_counts, alignments, policy, &status
  );
  int32_t result = 0;
  if (mode == 5) {
    if (status != LF_UNSUPPORTED || executor == NULL ||
        state.event_destroys != 1 ||
        atomic_load(&stream->active_operations) != 0 ||
        atomic_load(&function->active_operations) != 0 ||
        atomic_load(&allocation->active_operations) != 0) {
      result = 209;
    }
  } else if (mode == 6 || mode == 7) {
    if (status != LF_DRIVER_FAILURE || executor == NULL ||
        atomic_load(&executor->state) != LF_RESOURCE_LIVE ||
        atomic_load(&stream->active_operations) == 0 ||
        atomic_load(&function->active_operations) == 0 ||
        atomic_load(&allocation->active_operations) == 0) result = 213;
    if (result == 0 && mode == 6 &&
        (state.event_destroys != 1 || state.graph_destroys != 2 ||
         state.graph_exec_destroys != 1 || executor->event != NULL ||
         executor->graph != NULL || executor->graph_exec == NULL)) result = 214;
    if (result == 0 && mode == 7 &&
        (state.event_destroys != 1 || state.graph_destroys != 1 ||
         state.graph_exec_destroys != 1 || executor->event == NULL ||
         executor->graph != NULL || executor->graph_exec != NULL)) result = 215;
    if (result == 0 &&
        (lunaflux_cuda_ordered_executor_close(executor) != LF_OK ||
         atomic_load(&executor->state) != LF_RESOURCE_CLOSED ||
         atomic_load(&stream->active_operations) != 0 ||
         atomic_load(&function->active_operations) != 0 ||
         atomic_load(&allocation->active_operations) != 0 ||
         lunaflux_cuda_ordered_executor_close(executor) != LF_OK)) result = 216;
    if (result == 0 && mode == 6 &&
        (state.event_destroys != 1 || state.graph_destroys != 2 ||
         state.graph_exec_destroys != 2)) result = 217;
    if (result == 0 && mode == 7 &&
        (state.event_destroys != 2 || state.graph_destroys != 1 ||
         state.graph_exec_destroys != 1)) result = 218;
  } else if (status != LF_OK || executor == NULL) {
    result = 200;
  } else {
    int32_t expected_mode = mode == 0 || mode == 3 || mode == 4 || mode == 8
      ? LF_ORDERED_MODE_CAPTURED
      : LF_ORDERED_MODE_EAGER;
    if (lunaflux_cuda_ordered_executor_mode(executor) != expected_mode) {
      result = 201;
    }
    if (result == 0 && expected_mode == LF_ORDERED_MODE_CAPTURED) {
      if (lunaflux_cuda_ordered_executor_enqueue(executor, 0) !=
          LF_INVALID_ARGUMENT) {
        result = 202;
      }
#if !defined(_WIN32)
      if (result == 0 && mode == 8) {
        pthread_t thread;
        lf_graph_race_call call = { executor, LF_DRIVER_FAILURE };
        int started = pthread_create(
          &thread, NULL, lf_graph_launch_thread, &call
        ) == 0;
        if (!started) result = 210;
        while (result == 0 &&
          atomic_load(&state.graph_launch_entered) == 0) {
          sched_yield();
        }
        if (result == 0 &&
          (lunaflux_cuda_ordered_executor_launch_captured(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_record(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_wait(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_abort(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_reset(executor) != LF_BUSY ||
           lunaflux_cuda_ordered_executor_close(executor) != LF_BUSY)) {
          result = 211;
        }
        atomic_store(&state.release_graph_launch, 1);
        if (started &&
          (pthread_join(thread, NULL) != 0 || call.result != LF_OK)) {
          result = 212;
        }
      } else
#endif
      if (result == 0 &&
        lunaflux_cuda_ordered_executor_launch_captured(executor) != LF_OK) {
        result = 202;
      }
    } else if (result == 0 &&
      lunaflux_cuda_ordered_executor_enqueue(executor, 0) != LF_OK) {
      result = 203;
    }
    if (result == 0 &&
      (lunaflux_cuda_ordered_executor_record(executor) != LF_OK ||
       lunaflux_cuda_ordered_executor_wait(executor) != LF_OK ||
       lunaflux_cuda_ordered_executor_reset(executor) != LF_OK)) {
      result = 204;
    }
    if (result == 0) {
      int32_t close = lunaflux_cuda_ordered_executor_close(executor);
      if (mode == 4) {
        if (close != LF_DRIVER_FAILURE ||
            atomic_load(&stream->active_operations) == 0 ||
            atomic_load(&function->active_operations) == 0 ||
            atomic_load(&allocation->active_operations) == 0 ||
            lunaflux_cuda_ordered_executor_close(executor) != LF_OK) {
          result = 205;
        }
      } else if (close != LF_OK) {
        result = 206;
      }
    }
  }
  if (result == 0 && mode != 5 && mode != 6 && mode != 7 &&
      (atomic_load(&stream->active_operations) != 0 ||
       atomic_load(&function->active_operations) != 0 ||
       atomic_load(&allocation->active_operations) != 0)) {
    result = 207;
  }
  int32_t captured = mode == 0 || mode == 3 || mode == 4 || mode == 8;
  if (result == 0 && mode != 5 && mode != 6 && mode != 7 &&
      (state.begin_captures != captured ||
       state.end_captures != captured ||
       state.captured_launches != captured ||
       state.graph_instantiations != captured ||
       state.graph_destroys != captured ||
       state.graph_launches != captured ||
       state.graph_exec_destroys != (captured ? (mode == 4 ? 2 : 1) : 0) ||
       state.event_records != 1 || state.event_waits != 1 ||
       state.event_destroys != 1)) {
    result = 208;
  }
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
  lf_active_graph_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_ordered_graph(int32_t cycles) {
  if (cycles < 1 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int32_t cycle = 0; cycle < cycles; cycle += 1) {
    int32_t mode = cycle % 9;
    int32_t result = lf_run_graph_probe(mode);
    if (result != LF_OK) return result;
  }
  return LF_OK;
}
