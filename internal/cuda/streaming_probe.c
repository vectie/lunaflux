#include "streaming_internal.h"
#include <stdlib.h>
#include <string.h>

typedef struct { void *destination; const void *source; size_t bytes; } probe_copy;
typedef struct { probe_copy copies[16]; int count; } probe_stream;
typedef struct { probe_stream *stream; } probe_event;
typedef struct {
  int fail;
  int hosts;
  int streams;
  int events;
  int pending_once;
} probe_state;
static _Thread_local probe_state *streaming_probe;

static int fail_once(int point) {
  if (streaming_probe->fail != point) return 0;
  streaming_probe->fail = 0;
  return 1;
}
static CUresult current(CUcontext context) { return context == NULL; }
static CUresult host_new(void **p, size_t bytes, uint32_t flags) {
  if (flags != 0 || fail_once(1)) return 1;
  *p = calloc(1, bytes);
  if (*p == NULL) return 1;
  streaming_probe->hosts++;
  return fail_once(14);
}
static CUresult host_free(void *p) {
  if (fail_once(10)) return 1;
  free(p);
  streaming_probe->hosts--;
  return 0;
}
static CUresult stream_new(CUstream *s, uint32_t flags) {
  if (flags != 1 || fail_once(2)) return 1;
  *s = calloc(1, sizeof(probe_stream));
  if (*s == NULL) return 1;
  streaming_probe->streams++;
  return fail_once(12);
}
static CUresult stream_free(CUstream s) {
  if (fail_once(9)) return 1;
  if (((probe_stream *)s)->count != 0) return 1;
  free(s);
  streaming_probe->streams--;
  return 0;
}
static CUresult event_new(CUevent *e, uint32_t flags) {
  if (flags != 2 || fail_once(3)) return 1;
  *e = calloc(1, sizeof(probe_event));
  if (*e == NULL) return 1;
  streaming_probe->events++;
  return fail_once(13);
}
static CUresult event_free(CUevent e) {
  if (fail_once(8)) return 1;
  free(e);
  streaming_probe->events--;
  return 0;
}
static CUresult copy(void *destination, const void *source, size_t bytes, CUstream s) {
  probe_stream *stream = s;
  if (fail_once(4) || stream->count == 16) return 1;
  stream->copies[stream->count++] = (probe_copy){destination, source, bytes};
  return fail_once(15); /* Driver may report an error after accepting work. */
}
static CUresult h2d(CUdeviceptr d, const void *h, size_t n, CUstream s) {
  return copy((void *)(uintptr_t)d, h, n, s);
}
static CUresult d2h(void *h, CUdeviceptr d, size_t n, CUstream s) {
  return copy(h, (void *)(uintptr_t)d, n, s);
}
static CUresult synchronize(CUstream s) {
  if (fail_once(7)) return 1;
  probe_stream *stream = s;
  for (int i = 0; i < stream->count; ++i) {
    probe_copy c = stream->copies[i];
    memcpy(c.destination, c.source, c.bytes);
  }
  stream->count = 0;
  return 0;
}
static CUresult record(CUevent e, CUstream s) {
  if (fail_once(5)) return 1;
  ((probe_event *)e)->stream = s;
  return 0;
}
static CUresult query(CUevent e) {
  if (fail_once(6)) return 1;
  if (streaming_probe->pending_once) {
    streaming_probe->pending_once = 0;
    return CUDA_ERROR_NOT_READY;
  }
  return synchronize(((probe_event *)e)->stream);
}
static void no_finalize(void *p) { (void)p; }

#define CHECK(expression, code) do { if (!(expression)) { result = code; goto cleanup; } } while (0)

static int32_t run_probe(int fault) {
  probe_state state = {0};
  streaming_probe = &state;
  lf_cuda_api api = {0};
  api.cuCtxSetCurrent = current;
  api.cuMemHostAlloc = host_new;
  api.cuMemFreeHost = host_free;
  api.cuStreamCreate = stream_new;
  api.cuStreamDestroy = stream_free;
  api.cuEventCreate = event_new;
  api.cuEventDestroy = event_free;
  api.cuMemcpyHtoDAsync = h2d;
  api.cuMemcpyDtoHAsync = d2h;
  api.cuStreamSynchronize = synchronize;
  api.cuEventRecord = record;
  api.cuEventQuery = query;
  lf_context *context = moonbit_make_external_object(no_finalize, sizeof(*context));
  memset(context, 0, sizeof(*context));
  context->api = &api;
  context->handle = context;
  atomic_init(&context->state, LF_RESOURCE_LIVE);
  atomic_init(&context->active_operations, 0);
  atomic_init(&context->children, 1);
  lf_allocation *allocation = moonbit_make_external_object(no_finalize, sizeof(*allocation));
  memset(allocation, 0, sizeof(*allocation));
  uint8_t device[128];
  for (int i = 0; i < 128; ++i) device[i] = (uint8_t)i;
  allocation->context = context;
  allocation->handle = (CUdeviceptr)(uintptr_t)device;
  allocation->size = sizeof(device);
  atomic_init(&allocation->state, LF_RESOURCE_LIVE);
  atomic_init(&allocation->active_operations, 0);
  int32_t status = 0;
  int32_t result = 0;
  uint8_t *scratch = (uint8_t *)moonbit_make_int32_array_raw(32);
  memset(scratch, 0, 32);
  if ((fault >= 1 && fault <= 3) || (fault >= 12 && fault <= 14)) state.fail = fault;
  if (fault == 11) api.cuMemHostAlloc = NULL;
  lf_streaming_pool *pool = lunaflux_cuda_streaming_create(context, allocation, 128, 2, 4, &status);
  if ((fault >= 1 && fault <= 3) || (fault >= 11 && fault <= 14)) {
    CHECK(status != LF_OK, 100 + fault);
    CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, 0, 0, 8, 0) == LF_CLOSED, 120);
    if (fault == 3) {
      state.fail = 9;
      CHECK(lunaflux_cuda_streaming_close(pool) == LF_DRIVER_FAILURE, 121);
      CHECK(atomic_load(&allocation->active_operations) == 1, 122);
    }
    goto cleanup;
  }
  CHECK(status == LF_OK, 130);
  CHECK(lunaflux_cuda_allocation_close(allocation) == LF_BUSY, 131);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, -1, 0, 0, 8, 0) == LF_INVALID_ARGUMENT, 132);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, INT64_MAX, 0, 8, 0) == LF_INVALID_ARGUMENT, 133);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, 0, 127, 2, 0) == LF_INVALID_ARGUMENT, 134);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, 0, 0, 16, 0) == LF_OK, 135);
  CHECK(lunaflux_cuda_streaming_host_copy(pool, scratch, 0, 0, 8, 0) == LF_BUSY, 160);
  CHECK(lunaflux_cuda_streaming_host_copy(pool, scratch, 0, 0, 8, 1) == LF_BUSY, 161);
  CHECK(lunaflux_cuda_streaming_host_copy(pool, scratch, 64, 31, 8, 1) == LF_INVALID_ARGUMENT, 162);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 1, 8, 64, 16, 0) == LF_BUSY, 136);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 1, 64, 8, 16, 0) == LF_BUSY, 137);
  CHECK(lunaflux_cuda_streaming_close(pool) == LF_BUSY, 138);
  CHECK(lunaflux_cuda_streaming_reset(pool, 0) == LF_BUSY, 139);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 1, 64, 64, 16, 0) == LF_OK, 163);
  CHECK(lunaflux_cuda_streaming_record(pool, 1) == LF_OK, 164);
  CHECK(lunaflux_cuda_streaming_poll(pool, 1) == 1, 165);
  CHECK(memcmp((uint8_t *)pool->host + 64, device + 64, 16) == 0, 166);
  CHECK(lunaflux_cuda_streaming_reset(pool, 1) == LF_OK, 167);
  if (fault == 4 || fault == 15) {
    state.fail = fault;
    CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, 16, 16, 16, 0) == LF_DRIVER_FAILURE, 140);
  } else {
    if (fault == 5) state.fail = 5;
    status = lunaflux_cuda_streaming_record(pool, 0);
    CHECK(status == (fault == 5 ? LF_DRIVER_FAILURE : LF_OK), 141);
    if (fault != 5) {
      if (fault == 6) state.fail = 6;
      state.pending_once = fault == 0;
      if (fault == 0) CHECK(lunaflux_cuda_streaming_poll(pool, 0) == 0, 142);
      status = lunaflux_cuda_streaming_poll(pool, 0);
      CHECK(status == (fault == 6 ? LF_DRIVER_FAILURE : 1), 143);
    }
  }
  if ((fault >= 4 && fault <= 6) || fault == 15) {
    CHECK(lunaflux_cuda_streaming_reset(pool, 0) == LF_BUSY, 144);
    state.fail = 7;
    CHECK(lunaflux_cuda_streaming_drain(pool, 0) == LF_DRIVER_FAILURE, 145);
    CHECK(lunaflux_cuda_streaming_close(pool) == LF_BUSY, 146);
  }
  CHECK(lunaflux_cuda_streaming_drain(pool, 0) == LF_OK, 147);
  CHECK(memcmp(pool->host, device, 16) == 0, 148);
  if (fault == 15) CHECK(memcmp(pool->host, device, 32) == 0, 170);
  CHECK(lunaflux_cuda_streaming_host_copy(pool, scratch, 0, 0, 16, 0) == LF_OK, 168);
  CHECK(memcmp(scratch, device, 16) == 0, 169);
  CHECK(lunaflux_cuda_streaming_reset(pool, 0) == LF_OK, 149);
  CHECK(lunaflux_cuda_streaming_enqueue(pool, 1, 0, 64, 16, 1) == LF_OK, 150);
  CHECK(lunaflux_cuda_streaming_record(pool, 1) == LF_OK, 151);
  CHECK(device[64] == 64, 152); /* The fake DMA really remains pending. */
  if (fault == 7) {
    state.fail = 7;
    CHECK(lunaflux_cuda_streaming_drain(pool, 1) == LF_DRIVER_FAILURE, 171);
    CHECK(lunaflux_cuda_streaming_close(pool) == LF_BUSY, 172);
  }
  CHECK(lunaflux_cuda_streaming_poll(pool, 1) == 1, 153);
  CHECK(memcmp(device, device + 64, 16) == 0, 154);
  CHECK(lunaflux_cuda_streaming_reset(pool, 1) == LF_OK, 155);
  if (fault >= 8 && fault <= 10) {
    state.fail = fault;
    CHECK(lunaflux_cuda_streaming_close(pool) == LF_DRIVER_FAILURE, 156);
    CHECK(atomic_load(&allocation->active_operations) == 1, 157);
    CHECK(lunaflux_cuda_streaming_enqueue(pool, 0, 0, 0, 8, 0) == LF_CLOSED, 158);
  }
cleanup:
  state.fail = 0;
  if (pool->usable) {
    for (int i = 0; i < pool->lanes; ++i) {
      if (lunaflux_cuda_streaming_drain(pool, i) != LF_OK && result == 0) result = 190;
    }
  }
  if (lunaflux_cuda_streaming_close(pool) != LF_OK && result == 0) result = 191;
  if (lunaflux_cuda_streaming_close(pool) != LF_OK && result == 0) result = 192;
  if ((state.hosts || state.streams || state.events ||
      atomic_load(&allocation->active_operations) != 0) && result == 0) result = 193;
  moonbit_decref(pool);
  moonbit_decref(scratch);
  moonbit_decref(allocation);
  moonbit_decref(context);
  streaming_probe = NULL;
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_streaming(int32_t cycles) {
  if (cycles <= 0 || cycles > 10000) return LF_INVALID_ARGUMENT;
  for (int i = 0; i < cycles; ++i) {
    for (int fault = 0; fault <= 15; ++fault) {
      int32_t result = run_probe(fault);
      if (result != 0) return result;
    }
  }
  return LF_OK;
}
