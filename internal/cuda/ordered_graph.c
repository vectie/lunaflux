#include "ordered_executor_internal.h"

#include <stdint.h>

#define CUDA_SUCCESS 0
#define CU_STREAM_CAPTURE_MODE_THREAD_LOCAL 1

static int lf_ordered_graph_api_available(const lf_cuda_api *api) {
  return api != NULL && api->graph_available != 0 &&
    api->cuStreamBeginCapture != NULL && api->cuStreamEndCapture != NULL &&
    api->cuGraphInstantiateWithFlags != NULL && api->cuGraphDestroy != NULL &&
    api->cuGraphExecDestroy != NULL && api->cuGraphLaunch != NULL;
}

static int32_t lf_ordered_capture_launches(lf_ordered_executor *executor) {
  for (int32_t index = 0; index < executor->kernel_count; index += 1) {
    lf_ordered_kernel *kernel = &executor->kernels[index];
    int32_t launch = lf_ordered_launch_record(executor, kernel);
    if (launch != LF_OK) return launch;
  }
  return LF_OK;
}

int32_t lf_ordered_graph_prepare(
  lf_ordered_executor *executor,
  int32_t policy
) {
  if (executor == NULL ||
      (policy != LF_ORDERED_EAGER_ONLY &&
       policy != LF_ORDERED_CAPTURE_REQUIRED &&
       policy != LF_ORDERED_CAPTURE_WITH_EAGER_FALLBACK)) {
    return LF_INVALID_ARGUMENT;
  }
  executor->execution_mode = LF_ORDERED_MODE_EAGER;
  executor->graph_exec = NULL;
  if (policy == LF_ORDERED_EAGER_ONLY) return LF_OK;
  lf_cuda_api *api = executor->context->api;
  if (!lf_ordered_graph_api_available(api)) {
    return policy == LF_ORDERED_CAPTURE_WITH_EAGER_FALLBACK
      ? LF_OK
      : LF_UNSUPPORTED;
  }
  int32_t result = lf_context_current(executor->context);
  if (result != LF_OK) return result;
  CUstream stream = (CUstream)executor->stream->handle;
  result = lf_cuda_map_result(api->cuStreamBeginCapture(
    stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
  ));
  if (result != LF_OK) return result;
  int32_t launch_result = lf_ordered_capture_launches(executor);
  CUgraph graph = NULL;
  int32_t end_result = lf_cuda_map_result(api->cuStreamEndCapture(
    stream, &graph
  ));
  executor->graph = graph;
  if (launch_result != LF_OK || end_result != LF_OK || graph == NULL) {
    return launch_result != LF_OK ? launch_result :
      (end_result != LF_OK ? end_result : LF_DRIVER_FAILURE);
  }
  result = lf_cuda_map_result(api->cuGraphInstantiateWithFlags(
    &executor->graph_exec, executor->graph, 0
  ));
  int32_t destroy_result = lf_cuda_map_result(
    api->cuGraphDestroy(executor->graph)
  );
  if (destroy_result == LF_OK) executor->graph = NULL;
  if (result != LF_OK || destroy_result != LF_OK ||
      executor->graph_exec == NULL) {
    return result != LF_OK ? result :
      (destroy_result != LF_OK ? destroy_result : LF_DRIVER_FAILURE);
  }
  executor->execution_mode = LF_ORDERED_MODE_CAPTURED;
  return LF_OK;
}

int32_t lf_ordered_graph_launch(lf_ordered_executor *executor) {
  if (executor == NULL || executor->graph_exec == NULL ||
      executor->execution_mode != LF_ORDERED_MODE_CAPTURED) {
    return LF_INVALID_ARGUMENT;
  }
  int32_t result = lf_context_current(executor->context);
  if (result != LF_OK) return result;
  return lf_cuda_map_result(executor->context->api->cuGraphLaunch(
    executor->graph_exec, (CUstream)executor->stream->handle
  ));
}

int32_t lf_ordered_graph_destroy(lf_ordered_executor *executor) {
  if (executor == NULL) return LF_INVALID_ARGUMENT;
  if (executor->capture_active) {
    int32_t cancel = lf_ordered_capture_cancel(executor);
    if (cancel != LF_OK) return cancel;
  }
  if (executor->graph_exec == NULL && executor->graph == NULL) return LF_OK;
  int32_t result = lf_context_current(executor->context);
  if (result != LF_OK) return result;
  if (executor->graph_exec != NULL) {
    int32_t destroy = lf_cuda_map_result(
      executor->context->api->cuGraphExecDestroy(executor->graph_exec)
    );
    if (destroy == LF_OK) {
      executor->graph_exec = NULL;
    } else {
      result = destroy;
    }
  }
  if (executor->graph != NULL) {
    int32_t destroy = lf_cuda_map_result(
      executor->context->api->cuGraphDestroy(executor->graph)
    );
    if (destroy == LF_OK) {
      executor->graph = NULL;
    } else if (result == LF_OK) {
      result = destroy;
    }
  }
  return result;
}

/* Startup-only composition: the immutable kernel records retain operands;
 * external same-stream effects may be inserted between those records. */
MOONBIT_FFI_EXPORT int32_t lunaflux_cuda_ordered_capture_begin(lf_ordered_executor *e) {
  int32_t status = lf_ordered_operation_begin(e);
  if (status != LF_OK) return status;
  if (e->capture_active || e->execution_mode != LF_ORDERED_MODE_EAGER ||
      atomic_load(&e->phase) != LF_ORDERED_IDLE || atomic_load(&e->next_kernel) != 0) {
    status = LF_INVALID_ARGUMENT;
  } else if (!lf_ordered_graph_api_available(e->context->api)) {
    status = LF_UNSUPPORTED;
  } else if ((status = lf_context_current(e->context)) == LF_OK) {
    status = lf_cuda_map_result(e->context->api->cuStreamBeginCapture(
      (CUstream)e->stream->handle, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    if (status == LF_OK) e->capture_active = 1;
  }
  lf_ordered_operation_end(e);
  return status;
}

int32_t lf_ordered_capture_cancel(lf_ordered_executor *e) {
  if (!e->capture_active) return LF_OK;
  int32_t status = lf_context_current(e->context);
  if (status != LF_OK) return status;
  CUgraph graph = NULL;
  /* An invalidated capture returns an error but still terminates capture. */
  (void)e->context->api->cuStreamEndCapture((CUstream)e->stream->handle, &graph);
  e->capture_active = 0;
  e->graph = graph;
  atomic_store(&e->phase, LF_ORDERED_COMPLETE);
  return LF_OK;
}

MOONBIT_FFI_EXPORT int32_t lunaflux_cuda_ordered_capture_end(lf_ordered_executor *e) {
  int32_t status = lf_ordered_operation_begin(e);
  if (status != LF_OK) return status;
  if (!e->capture_active || atomic_load(&e->next_kernel) != e->kernel_count) {
    status = LF_INVALID_ARGUMENT;
  } else if ((status = lf_context_current(e->context)) == LF_OK) {
    status = lf_cuda_map_result(e->context->api->cuStreamEndCapture(
      (CUstream)e->stream->handle, &e->graph));
    e->capture_active = 0;
    atomic_store(&e->phase, LF_ORDERED_IDLE);
    atomic_store(&e->next_kernel, 0);
    if (status == LF_OK && e->graph != NULL) {
      status = lf_cuda_map_result(e->context->api->cuGraphInstantiateWithFlags(&e->graph_exec, e->graph, 0));
      if (status == LF_OK) {
        status = lf_cuda_map_result(e->context->api->cuGraphDestroy(e->graph));
        if (status == LF_OK) { e->graph = NULL; e->execution_mode = LF_ORDERED_MODE_CAPTURED; }
      }
    } else if (status == LF_OK) { status = LF_DRIVER_FAILURE; }
  }
  lf_ordered_operation_end(e);
  return status;
}

/* Drop graph references before NCCL abort: NCCL may otherwise wait for its
 * graph user objects. Keep kernel/operand/stream leases until explicit drain. */
MOONBIT_FFI_EXPORT int32_t lunaflux_cuda_ordered_capture_discard(lf_ordered_executor *e) {
  int32_t status = lf_ordered_operation_begin(e);
  if (status != LF_OK) return status;
  status = lf_ordered_graph_destroy(e);
  if (status == LF_OK) e->execution_mode = LF_ORDERED_MODE_EAGER;
  lf_ordered_operation_end(e);
  return status;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_launch_captured(
  lf_ordered_executor *executor
) {
  if (executor == NULL) return LF_CLOSED;
  int32_t result = lf_ordered_operation_begin(executor);
  if (result != LF_OK) return result;
  if (atomic_load(&executor->phase) != LF_ORDERED_IDLE ||
      atomic_load(&executor->next_kernel) != 0 ||
      executor->execution_mode != LF_ORDERED_MODE_CAPTURED) {
    result = LF_INVALID_ARGUMENT;
  } else {
    result = lf_ordered_graph_launch(executor);
    if (result == LF_OK) {
      atomic_store(&executor->next_kernel, executor->kernel_count);
      atomic_store(&executor->phase, LF_ORDERED_ENQUEUED);
    }
  }
  lf_ordered_operation_end(executor);
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_ordered_executor_mode(lf_ordered_executor *executor) {
  if (executor == NULL || atomic_load(&executor->state) != LF_RESOURCE_LIVE) {
    return LF_CLOSED;
  }
  return executor->execution_mode;
}
