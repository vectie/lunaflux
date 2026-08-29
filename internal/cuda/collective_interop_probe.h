#ifndef LUNAFLUX_DEVICE_COLLECTIVE_INTEROP_PROBE_H
#define LUNAFLUX_DEVICE_COLLECTIVE_INTEROP_PROBE_H

#include "collective_interop.h"
#include "resource_internal.h"

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

typedef struct lf_device_interop_probe {
  lf_cuda_api api;
  lf_context context;
  lf_allocation send;
  lf_allocation receive;
  lf_child queue;
  CUresult query_result;
  int32_t query_calls;
} lf_device_interop_probe;

/* MoonBit white-box tests that cross the real ownership seam must not pass
 * stack fixture addresses to code that retains owners. This wrapper keeps the
 * concrete CUDA test objects private to the CUDA probe boundary while exposing
 * only the same opaque interop tokens consumed by NCCL. */
typedef struct lf_device_interop_managed_probe {
  lf_device_interop_probe fixture;
  lf_context *context;
  lf_allocation *send;
  lf_allocation *receive;
  lf_child *queue;
} lf_device_interop_managed_probe;

static lf_device_interop_probe *lf_active_device_interop_probe = NULL;

static inline CUresult lf_device_interop_probe_set_current(CUcontext context) {
  return lf_active_device_interop_probe != NULL &&
         context == lf_active_device_interop_probe->context.handle
    ? 0
    : 1;
}

static inline CUresult lf_device_interop_probe_destroy_context(CUcontext context) {
  return lf_device_interop_probe_set_current(context);
}

static inline CUresult lf_device_interop_probe_query(CUstream queue) {
  if (lf_active_device_interop_probe == NULL ||
      queue != (CUstream)lf_active_device_interop_probe->queue.handle) {
    return 1;
  }
  lf_active_device_interop_probe->query_calls += 1;
  return lf_active_device_interop_probe->query_result;
}

static inline void lf_device_interop_probe_init_at(
  lf_device_interop_probe *probe,
  uintptr_t context_address,
  uintptr_t send_address,
  uintptr_t receive_address,
  uintptr_t queue_address,
  size_t region_bytes
) {
  memset(probe, 0, sizeof(*probe));
  probe->api.availability = LF_AVAILABLE;
  probe->api.cuCtxSetCurrent = lf_device_interop_probe_set_current;
  probe->api.cuCtxDestroy = lf_device_interop_probe_destroy_context;
  probe->api.cuStreamQuery = lf_device_interop_probe_query;
  probe->context.api = &probe->api;
  probe->context.handle = (CUcontext)context_address;
  atomic_init(&probe->context.state, LF_RESOURCE_LIVE);
  atomic_init(&probe->context.active_operations, 0);
  atomic_init(&probe->context.children, 0);
  probe->send.context = &probe->context;
  probe->send.handle = (CUdeviceptr)send_address;
  probe->send.size = region_bytes;
  atomic_init(&probe->send.state, LF_RESOURCE_LIVE);
  atomic_init(&probe->send.active_operations, 0);
  probe->receive.context = &probe->context;
  probe->receive.handle = (CUdeviceptr)receive_address;
  probe->receive.size = region_bytes;
  atomic_init(&probe->receive.state, LF_RESOURCE_LIVE);
  atomic_init(&probe->receive.active_operations, 0);
  probe->queue.context = &probe->context;
  probe->queue.handle = (void *)queue_address;
  atomic_init(&probe->queue.state, LF_RESOURCE_LIVE);
  atomic_init(&probe->queue.active_operations, 0);
  atomic_init(&probe->queue.children, 0);
  probe->query_result = 0;
  lf_active_device_interop_probe = probe;
}

static inline void lf_device_interop_probe_init(lf_device_interop_probe *probe) {
  lf_device_interop_probe_init_at(
    probe,
    0x2000U,
    0x10000U,
    0x20000U,
    0x3000U,
    4096U
  );
}

static inline void lf_device_interop_probe_owner_finalize(void *object) {
  (void)object;
}

static inline void lf_device_interop_managed_probe_init(
  lf_device_interop_managed_probe *probe
) {
  memset(probe, 0, sizeof(*probe));
  lf_device_interop_probe_init(&probe->fixture);
  probe->context = (lf_context *)moonbit_make_external_object(
    lf_device_interop_probe_owner_finalize,
    sizeof(lf_context)
  );
  probe->send = (lf_allocation *)moonbit_make_external_object(
    lf_device_interop_probe_owner_finalize,
    sizeof(lf_allocation)
  );
  probe->receive = (lf_allocation *)moonbit_make_external_object(
    lf_device_interop_probe_owner_finalize,
    sizeof(lf_allocation)
  );
  probe->queue = (lf_child *)moonbit_make_external_object(
    lf_device_interop_probe_owner_finalize,
    sizeof(lf_child)
  );
  memcpy(probe->context, &probe->fixture.context, sizeof(lf_context));
  memcpy(probe->send, &probe->fixture.send, sizeof(lf_allocation));
  memcpy(probe->receive, &probe->fixture.receive, sizeof(lf_allocation));
  memcpy(probe->queue, &probe->fixture.queue, sizeof(lf_child));
  probe->send->context = probe->context;
  probe->receive->context = probe->context;
  probe->queue->context = probe->context;
}

static inline void lf_device_interop_managed_probe_close(
  lf_device_interop_managed_probe *probe
) {
  moonbit_decref(probe->queue);
  moonbit_decref(probe->receive);
  moonbit_decref(probe->send);
  moonbit_decref(probe->context);
  memset(probe, 0, sizeof(*probe));
}

static inline lf_device_context_token *
lf_device_interop_managed_probe_context(
  lf_device_interop_managed_probe *probe
) {
  lf_active_device_interop_probe = &probe->fixture;
  return (lf_device_context_token *)(void *)probe->context;
}

static inline lf_device_region_token *lf_device_interop_managed_probe_send(
  lf_device_interop_managed_probe *probe
) {
  return (lf_device_region_token *)(void *)probe->send;
}

static inline lf_device_region_token *lf_device_interop_managed_probe_receive(
  lf_device_interop_managed_probe *probe
) {
  return (lf_device_region_token *)(void *)probe->receive;
}

static inline lf_device_queue_token *lf_device_interop_managed_probe_queue(
  lf_device_interop_managed_probe *probe
) {
  return (lf_device_queue_token *)(void *)probe->queue;
}

static inline int32_t lf_device_interop_managed_probe_query_calls(
  const lf_device_interop_managed_probe *probe
) {
  return probe->fixture.query_calls;
}

static inline lf_device_context_token *lf_device_interop_probe_context(
  lf_device_interop_probe *probe
) {
  lf_active_device_interop_probe = probe;
  return (lf_device_context_token *)(void *)&probe->context;
}

static inline lf_device_region_token *lf_device_interop_probe_send(
  lf_device_interop_probe *probe
) {
  return (lf_device_region_token *)(void *)&probe->send;
}

static inline lf_device_region_token *lf_device_interop_probe_receive(
  lf_device_interop_probe *probe
) {
  return (lf_device_region_token *)(void *)&probe->receive;
}

static inline lf_device_queue_token *lf_device_interop_probe_queue(
  lf_device_interop_probe *probe
) {
  return (lf_device_queue_token *)(void *)&probe->queue;
}

static inline lf_device_context_token *lf_device_interop_probe_capture_context(
  lf_device_interop_probe *probe
) {
  return lunaflux_device_interop_context_token(&probe->context);
}

static inline lf_device_region_token *lf_device_interop_probe_capture_region(
  lf_device_interop_probe *probe
) {
  return lunaflux_device_interop_region_token(&probe->send);
}

static inline lf_device_queue_token *lf_device_interop_probe_capture_queue(
  lf_device_interop_probe *probe
) {
  return lunaflux_device_interop_queue_token(&probe->queue);
}

static inline lf_device_region_token *lf_device_interop_probe_in_place(
  lf_device_interop_probe *probe
) {
  return lf_device_interop_probe_send(probe);
}

static inline void lf_device_interop_probe_query_result(
  lf_device_interop_probe *probe,
  int32_t result
) {
  probe->query_result = (CUresult)result;
  lf_active_device_interop_probe = probe;
}

static inline int32_t lf_device_interop_probe_query_calls(
  const lf_device_interop_probe *probe
) {
  return probe->query_calls;
}

static inline int32_t lf_device_interop_probe_context_children(
  const lf_device_interop_probe *probe
) {
  return atomic_load(&probe->context.children);
}

static inline int32_t lf_device_interop_probe_context_close(
  lf_device_interop_probe *probe
) {
  return lunaflux_cuda_context_close(&probe->context);
}

static inline void lf_device_interop_probe_close_context(
  lf_device_interop_probe *probe
) {
  atomic_store(&probe->context.state, LF_RESOURCE_CLOSED);
}

static inline int32_t lf_device_interop_probe_active_count(
  const lf_device_interop_probe *probe
) {
  return atomic_load(&probe->context.active_operations) +
    atomic_load(&probe->send.active_operations) +
    atomic_load(&probe->receive.active_operations) +
    atomic_load(&probe->queue.active_operations);
}

#endif
