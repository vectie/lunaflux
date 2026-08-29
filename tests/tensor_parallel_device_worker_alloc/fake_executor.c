#include "moonbit.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern int32_t lunaflux_device_worker_fake_device_info(
  int32_t,
  int64_t *,
  uint8_t *
);
extern void *lunaflux_device_worker_fake_context_create(int32_t, int32_t *);
extern int32_t lunaflux_device_worker_fake_function_launch(
  void *, void *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
  int32_t, void **, int64_t *, int64_t *, int64_t *
);

typedef struct {
  void *stream;
  void **functions;
  int32_t *dimensions;
  int32_t *argument_starts;
  void **allocations;
  int64_t *offsets;
  int64_t *byte_counts;
  int64_t *alignments;
  int32_t kernel_count;
  int32_t next_kernel;
  int32_t execution_mode;
  int32_t phase;
  int32_t poll_count;
  int32_t live;
} tp_fake_executor;

enum {
  TP_EXECUTOR_IDLE = 0,
  TP_EXECUTOR_ENQUEUED = 1,
  TP_EXECUTOR_RECORDED = 2,
  TP_EXECUTOR_COMPLETE = 3,
};

static uint64_t executor_enqueues;
static uint64_t executor_records;
static uint64_t executor_pending_polls;
static uint64_t executor_complete_polls;
static uint64_t executor_resets;
static uint64_t blocking_synchronizations;
static int32_t live_executors;

static void finalize_executor(void *raw) {
  tp_fake_executor *executor = (tp_fake_executor *)raw;
  if (executor->live != 0 || executor->functions != NULL) abort();
}

int32_t lunaflux_tp_alloc_fake_device_info(
  int32_t ordinal,
  int64_t *numeric,
  uint8_t *name
) {
  if (ordinal < 0 || ordinal > 1) return -2;
  return lunaflux_device_worker_fake_device_info(0, numeric, name);
}

void *lunaflux_tp_alloc_fake_context_create(
  int32_t ordinal,
  int32_t *status
) {
  if (ordinal < 0 || ordinal > 1) {
    *status = -2;
    return lunaflux_device_worker_fake_context_create(2, status);
  }
  return lunaflux_device_worker_fake_context_create(0, status);
}

void *lunaflux_tp_alloc_fake_ordered_executor_create(
  void *context,
  void *stream,
  void **functions,
  int32_t *dimensions,
  int32_t *argument_starts,
  void **allocations,
  int64_t *offsets,
  int64_t *byte_counts,
  int64_t *alignments,
  int32_t policy,
  int32_t *status
) {
  tp_fake_executor *executor =
    (tp_fake_executor *)moonbit_make_external_object(
      finalize_executor,
      sizeof(tp_fake_executor)
    );
  memset(executor, 0, sizeof(*executor));
  if (status == NULL) return executor;
  int32_t kernels = functions == NULL ? 0 : Moonbit_array_length(functions);
  int32_t arguments = allocations == NULL ? 0 : Moonbit_array_length(allocations);
  if (context == NULL || stream == NULL || policy != 0 ||
      kernels <= 0 || arguments <= 0 ||
      dimensions == NULL || argument_starts == NULL || offsets == NULL ||
      byte_counts == NULL || alignments == NULL ||
      Moonbit_array_length(dimensions) != kernels * 7 ||
      Moonbit_array_length(argument_starts) != kernels + 1 ||
      Moonbit_array_length(offsets) != arguments ||
      Moonbit_array_length(byte_counts) != arguments ||
      Moonbit_array_length(alignments) != arguments ||
      argument_starts[0] != 0 || argument_starts[kernels] != arguments) {
    *status = -2;
    return executor;
  }
  moonbit_incref(stream);
  moonbit_incref(functions);
  moonbit_incref(dimensions);
  moonbit_incref(argument_starts);
  moonbit_incref(allocations);
  moonbit_incref(offsets);
  moonbit_incref(byte_counts);
  moonbit_incref(alignments);
  executor->stream = stream;
  executor->functions = functions;
  executor->dimensions = dimensions;
  executor->argument_starts = argument_starts;
  executor->allocations = allocations;
  executor->offsets = offsets;
  executor->byte_counts = byte_counts;
  executor->alignments = alignments;
  executor->kernel_count = kernels;
  executor->execution_mode = 0;
  executor->phase = TP_EXECUTOR_IDLE;
  executor->live = 1;
  live_executors += 1;
  *status = 0;
  return executor;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_launch_captured(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0) return -3;
  return -2;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_mode(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0) return -3;
  return executor->execution_mode;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_enqueue(
  tp_fake_executor *executor,
  int32_t index
) {
  if (executor == NULL || executor->live == 0 ||
      (executor->phase != TP_EXECUTOR_IDLE &&
       executor->phase != TP_EXECUTOR_ENQUEUED) ||
      index != executor->next_kernel || index >= executor->kernel_count) {
    return -2;
  }
  int32_t *dimensions = executor->dimensions + index * 7;
  int32_t status = lunaflux_device_worker_fake_function_launch(
    executor->functions[index],
    executor->stream,
    dimensions[0], dimensions[1], dimensions[2],
    dimensions[3], dimensions[4], dimensions[5], dimensions[6],
    executor->allocations,
    executor->offsets,
    executor->byte_counts,
    executor->alignments
  );
  if (status != 0) return status;
  executor->next_kernel += 1;
  executor->phase = TP_EXECUTOR_ENQUEUED;
  executor_enqueues += 1U;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_record(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0 ||
      executor->phase != TP_EXECUTOR_ENQUEUED ||
      executor->next_kernel != executor->kernel_count) return -2;
  executor->phase = TP_EXECUTOR_RECORDED;
  executor->poll_count = 0;
  executor_records += 1U;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_poll(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0) return -3;
  if (executor->phase == TP_EXECUTOR_COMPLETE) return 1;
  if (executor->phase != TP_EXECUTOR_RECORDED) return -2;
  if (executor->poll_count == 0) {
    executor->poll_count = 1;
    executor_pending_polls += 1U;
    return 0;
  }
  executor->phase = TP_EXECUTOR_COMPLETE;
  executor_complete_polls += 1U;
  return 1;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_abort(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0) return -3;
  blocking_synchronizations += 1U;
  executor->phase = TP_EXECUTOR_COMPLETE;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_reset(
  tp_fake_executor *executor
) {
  if (executor == NULL || executor->live == 0 ||
      executor->phase != TP_EXECUTOR_COMPLETE) return -4;
  executor->next_kernel = 0;
  executor->phase = TP_EXECUTOR_IDLE;
  executor->poll_count = 0;
  executor_resets += 1U;
  return 0;
}

int32_t lunaflux_tp_alloc_fake_ordered_executor_close(
  tp_fake_executor *executor
) {
  if (executor == NULL) return -2;
  if (executor->live == 0) return 0;
  if (executor->phase != TP_EXECUTOR_IDLE &&
      executor->phase != TP_EXECUTOR_COMPLETE) return -4;
  moonbit_decref(executor->stream);
  moonbit_decref(executor->functions);
  moonbit_decref(executor->dimensions);
  moonbit_decref(executor->argument_starts);
  moonbit_decref(executor->allocations);
  moonbit_decref(executor->offsets);
  moonbit_decref(executor->byte_counts);
  moonbit_decref(executor->alignments);
  executor->stream = NULL;
  executor->functions = NULL;
  executor->live = 0;
  live_executors -= 1;
  return 0;
}

int32_t lunaflux_tp_alloc_blocking_sync(void *resource) {
  if (resource == NULL) return -2;
  blocking_synchronizations += 1U;
  return 0;
}

void lunaflux_tp_alloc_evidence_reset(void) {
  executor_enqueues = 0U;
  executor_records = 0U;
  executor_pending_polls = 0U;
  executor_complete_polls = 0U;
  executor_resets = 0U;
  blocking_synchronizations = 0U;
}

uint64_t lunaflux_tp_alloc_executor_enqueues(void) { return executor_enqueues; }
uint64_t lunaflux_tp_alloc_executor_records(void) { return executor_records; }
uint64_t lunaflux_tp_alloc_executor_pending_polls(void) {
  return executor_pending_polls;
}
uint64_t lunaflux_tp_alloc_executor_complete_polls(void) {
  return executor_complete_polls;
}
uint64_t lunaflux_tp_alloc_executor_resets(void) { return executor_resets; }
uint64_t lunaflux_tp_alloc_blocking_synchronizations(void) {
  return blocking_synchronizations;
}
int32_t lunaflux_tp_alloc_live_executors(void) { return live_executors; }
