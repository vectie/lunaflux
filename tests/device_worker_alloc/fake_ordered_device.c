#include "moonbit.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum fake_ordered_phase {
  FAKE_ORDERED_IDLE = 0,
  FAKE_ORDERED_ENQUEUED = 1,
  FAKE_ORDERED_RECORDED = 2,
  FAKE_ORDERED_COMPLETE = 3
};

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
  int32_t argument_count;
  int32_t next_kernel;
  int32_t execution_mode;
  int32_t phase;
  int32_t live;
} fake_ordered_executor;

int32_t lunaflux_device_worker_fake_function_launch_counted(
  void *, void *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
  int32_t, void **, int64_t *, int64_t *, int64_t *, int, int
);
void lunaflux_device_worker_fake_ordered_synchronize(void);
void lunaflux_device_worker_fake_ordered_resource_opened(void);
void lunaflux_device_worker_fake_ordered_resource_closed(void);

static void fake_ordered_release(fake_ordered_executor *value) {
  free(value->alignments);
  free(value->byte_counts);
  free(value->offsets);
  free(value->allocations);
  free(value->argument_starts);
  free(value->dimensions);
  free(value->functions);
  value->alignments = NULL;
  value->byte_counts = NULL;
  value->offsets = NULL;
  value->allocations = NULL;
  value->argument_starts = NULL;
  value->dimensions = NULL;
  value->functions = NULL;
}

static void fake_ordered_finalize(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value->live != 0) abort();
  fake_ordered_release(value);
}

static void *fake_ordered_copy(const void *source, size_t byte_count) {
  void *copy = libc_malloc(byte_count);
  if (copy != NULL) memcpy(copy, source, byte_count);
  return copy;
}

void *lunaflux_device_worker_fake_ordered_executor_create(
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
  fake_ordered_executor *value =
    (fake_ordered_executor *)moonbit_make_external_object(
      fake_ordered_finalize, sizeof(fake_ordered_executor)
    );
  memset(value, 0, sizeof(*value));
  if (status == NULL) return value;
  int32_t kernels = functions == NULL ? 0 : Moonbit_array_length(functions);
  int32_t arguments = allocations == NULL ? 0 : Moonbit_array_length(allocations);
  if (context == NULL || stream == NULL || policy != 0 || kernels <= 0 ||
      arguments <= 0 || Moonbit_array_length(dimensions) != kernels * 7 ||
      Moonbit_array_length(argument_starts) != kernels + 1 ||
      Moonbit_array_length(offsets) != arguments ||
      Moonbit_array_length(byte_counts) != arguments ||
      Moonbit_array_length(alignments) != arguments ||
      argument_starts[0] != 0 || argument_starts[kernels] != arguments) {
    *status = -2;
    return value;
  }
  value->functions = fake_ordered_copy(
    functions, (size_t)kernels * sizeof(void *)
  );
  value->dimensions = fake_ordered_copy(
    dimensions, (size_t)kernels * 7U * sizeof(int32_t)
  );
  value->argument_starts = fake_ordered_copy(
    argument_starts, (size_t)(kernels + 1) * sizeof(int32_t)
  );
  value->allocations = fake_ordered_copy(
    allocations, (size_t)arguments * sizeof(void *)
  );
  value->offsets = fake_ordered_copy(
    offsets, (size_t)arguments * sizeof(int64_t)
  );
  value->byte_counts = fake_ordered_copy(
    byte_counts, (size_t)arguments * sizeof(int64_t)
  );
  value->alignments = fake_ordered_copy(
    alignments, (size_t)arguments * sizeof(int64_t)
  );
  if (value->functions == NULL || value->dimensions == NULL ||
      value->argument_starts == NULL || value->allocations == NULL ||
      value->offsets == NULL || value->byte_counts == NULL ||
      value->alignments == NULL) {
    fake_ordered_release(value);
    *status = -6;
    return value;
  }
  value->stream = stream;
  value->kernel_count = kernels;
  value->argument_count = arguments;
  value->execution_mode = 0;
  value->phase = FAKE_ORDERED_IDLE;
  value->live = 1;
  lunaflux_device_worker_fake_ordered_resource_opened();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_ordered_executor_launch_captured(
  void *raw
) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  return -2;
}

int32_t lunaflux_device_worker_fake_ordered_executor_mode(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  return value->execution_mode;
}

int32_t lunaflux_device_worker_fake_ordered_executor_enqueue(
  void *raw,
  int32_t index
) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if ((value->phase != FAKE_ORDERED_IDLE &&
       value->phase != FAKE_ORDERED_ENQUEUED) ||
      index != value->next_kernel || index < 0 ||
      index >= value->kernel_count) return -2;
  int32_t start = value->argument_starts[index];
  int32_t end = value->argument_starts[index + 1];
  int32_t *dimensions = &value->dimensions[index * 7];
  int32_t result = lunaflux_device_worker_fake_function_launch_counted(
    value->functions[index], value->stream,
    dimensions[0], dimensions[1], dimensions[2], dimensions[3], dimensions[4],
    dimensions[5], dimensions[6], &value->allocations[start],
    &value->offsets[start], &value->byte_counts[start],
    &value->alignments[start], end - start, 0
  );
  if (result == 0) {
    value->next_kernel += 1;
    value->phase = FAKE_ORDERED_ENQUEUED;
  }
  return result;
}

int32_t lunaflux_device_worker_fake_ordered_executor_record(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if (value->phase != FAKE_ORDERED_ENQUEUED ||
      value->next_kernel != value->kernel_count) return -2;
  value->phase = FAKE_ORDERED_RECORDED;
  return 0;
}

int32_t lunaflux_device_worker_fake_ordered_executor_wait(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if (value->phase == FAKE_ORDERED_COMPLETE) return 0;
  if (value->phase != FAKE_ORDERED_RECORDED) return -2;
  lunaflux_device_worker_fake_ordered_synchronize();
  value->phase = FAKE_ORDERED_COMPLETE;
  return 0;
}

int32_t lunaflux_device_worker_fake_ordered_executor_poll(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if (value->phase == FAKE_ORDERED_COMPLETE) return 1;
  if (value->phase != FAKE_ORDERED_RECORDED) return -2;
  lunaflux_device_worker_fake_ordered_synchronize();
  value->phase = FAKE_ORDERED_COMPLETE;
  return 1;
}

int32_t lunaflux_device_worker_fake_ordered_executor_abort(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if (value->phase == FAKE_ORDERED_ENQUEUED ||
      value->phase == FAKE_ORDERED_RECORDED) {
    lunaflux_device_worker_fake_ordered_synchronize();
  }
  value->phase = FAKE_ORDERED_COMPLETE;
  return 0;
}

int32_t lunaflux_device_worker_fake_ordered_executor_reset(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL || value->live == 0) return -3;
  if (value->phase != FAKE_ORDERED_COMPLETE) return -4;
  value->next_kernel = 0;
  value->phase = FAKE_ORDERED_IDLE;
  return 0;
}

int32_t lunaflux_device_worker_fake_ordered_executor_close(void *raw) {
  fake_ordered_executor *value = (fake_ordered_executor *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  if (value->phase == FAKE_ORDERED_ENQUEUED ||
      value->phase == FAKE_ORDERED_RECORDED) return -4;
  fake_ordered_release(value);
  value->live = 0;
  lunaflux_device_worker_fake_ordered_resource_closed();
  return 0;
}
