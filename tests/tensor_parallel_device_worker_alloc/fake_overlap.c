#include "moonbit.h"
#include <stdint.h>
#include <stdlib.h>

extern void *lunaflux_device_worker_fake_stream_create(void *, int32_t *);
extern int32_t lunaflux_device_worker_fake_stream_close(void *);
extern int lunaflux_tp_alloc_submission_pending_on(void *);
extern int lunaflux_tp_alloc_gpu_pending(void);

typedef struct {
  void *resource;
  void *producer;
} fake_event;

static uint64_t records;
static uint64_t waits;
static uint64_t waits_before_gpu_completion;
static int fail_close_once;

static void finalize(void *raw) {
  if (((fake_event *)raw)->resource != NULL) abort();
}

void *lunaflux_tp_alloc_event_create(void *context, int32_t *status) {
  fake_event *event = moonbit_make_external_object(finalize, sizeof(fake_event));
  event->producer = NULL;
  event->resource = lunaflux_device_worker_fake_stream_create(context, status);
  return event;
}

int32_t lunaflux_tp_alloc_event_close(fake_event *event) {
  if (event->resource == NULL) return 0;
  if (fail_close_once) { fail_close_once = 0; return -1; }
  int32_t result = lunaflux_device_worker_fake_stream_close(event->resource);
  if (result != 0) return result;
  moonbit_decref(event->resource);
  event->resource = NULL;
  return 0;
}

int32_t lunaflux_tp_alloc_event_record(fake_event *event, void *stream) {
  if (event->resource == NULL || lunaflux_tp_alloc_submission_pending_on(stream)) abort();
  event->producer = stream;
  records += 1;
  return 0;
}

int32_t lunaflux_tp_alloc_stream_wait_event(void *stream, fake_event *event) {
  if (event->resource == NULL || event->producer == NULL || event->producer == stream) abort();
  waits += 1;
  if (lunaflux_tp_alloc_gpu_pending()) waits_before_gpu_completion += 1;
  return 0;
}

void lunaflux_tp_alloc_verify_overlap(void) {
  if (records == 0 || waits == 0 || waits_before_gpu_completion == 0) abort();
}

void lunaflux_tp_alloc_fail_event_close_once(void) { fail_close_once = 1; }
