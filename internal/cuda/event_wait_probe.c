#include "resource_internal.h"
#include <stdint.h>
#include <string.h>

int32_t lunaflux_cuda_stream_wait_event(lf_child *, lf_child *);
int32_t lunaflux_cuda_event_record(lf_child *, lf_child *);

static int records;
static int waits;
static int fail_wait;

static CUresult current(CUcontext context) { return context == (void *)(uintptr_t)1 ? 0 : 1; }
static CUresult record(CUevent event, CUstream stream) {
  if (event != (void *)(uintptr_t)4 || stream != (void *)(uintptr_t)2) return 1;
  records += 1;
  return 0;
}
static CUresult wait_event(CUstream stream, CUevent event, uint32_t flags) {
  if (stream != (void *)(uintptr_t)3 || event != (void *)(uintptr_t)4 || flags != 0 || records == 0) return 1;
  waits += 1;
  return fail_wait;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_cuda_test_event_wait(void) {
  lf_cuda_api api;
  lf_context context;
  lf_context foreign;
  lf_child producer, consumer, event;
  memset(&api, 0, sizeof(api));
  memset(&context, 0, sizeof(context));
  memset(&foreign, 0, sizeof(foreign));
  memset(&producer, 0, sizeof(producer));
  memset(&consumer, 0, sizeof(consumer));
  memset(&event, 0, sizeof(event));
  api.cuCtxSetCurrent = current;
  api.cuEventRecord = record;
  api.cuStreamWaitEvent = wait_event;
  context.api = &api;
  context.handle = (void *)(uintptr_t)1;
  atomic_init(&context.state, LF_RESOURCE_LIVE);
  atomic_init(&context.active_operations, 0);
  producer.context = consumer.context = event.context = &context;
  producer.handle = (void *)(uintptr_t)2;
  consumer.handle = (void *)(uintptr_t)3;
  event.handle = (void *)(uintptr_t)4;
  atomic_init(&producer.state, LF_RESOURCE_LIVE);
  atomic_init(&consumer.state, LF_RESOURCE_LIVE);
  atomic_init(&event.state, LF_RESOURCE_LIVE);
  atomic_init(&producer.active_operations, 0);
  atomic_init(&consumer.active_operations, 0);
  atomic_init(&event.active_operations, 0);
  records = waits = fail_wait = 0;
  for (int i = 0; i < 256; i += 1) {
    if (lunaflux_cuda_event_record(&event, &producer) != LF_OK) return 1;
    if (lunaflux_cuda_stream_wait_event(&consumer, &event) != LF_OK) return 2;
  }
  if (records != 256 || waits != 256) return 3;
  event.context = &foreign;
  if (lunaflux_cuda_stream_wait_event(&consumer, &event) != LF_INVALID_ARGUMENT || waits != 256) return 4;
  event.context = &context;
  fail_wait = 1;
  if (lunaflux_cuda_stream_wait_event(&consumer, &event) != LF_DRIVER_FAILURE) return 5;
  fail_wait = 0;
  atomic_store(&event.state, LF_RESOURCE_CLOSED);
  if (lunaflux_cuda_stream_wait_event(&consumer, &event) != LF_CLOSED) return 6;
  atomic_store(&event.state, LF_RESOURCE_LIVE);
  atomic_store(&consumer.state, LF_RESOURCE_CLOSED);
  if (lunaflux_cuda_stream_wait_event(&consumer, &event) != LF_CLOSED) return 7;
  if (atomic_load(&event.active_operations) != 0 || atomic_load(&consumer.active_operations) != 0 || atomic_load(&context.active_operations) != 0) return 8;
  return LF_OK;
}
