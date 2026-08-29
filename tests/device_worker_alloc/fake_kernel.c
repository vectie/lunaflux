#include "moonbit.h"

#include "fake_device_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct fake_module {
  fake_context *context;
  int live;
  int children;
} fake_module;

typedef struct {
  fake_module *module;
  int live;
  int emits_logits;
} fake_function;

static void module_finalize(void *raw) {
  fake_module *value = (fake_module *)raw;
  if (value->live != 0 || value->children != 0) abort();
}

static void function_finalize(void *raw) {
  if (((fake_function *)raw)->live != 0) abort();
}

void *lunaflux_device_worker_fake_module_load(
  void *raw_context,
  uint8_t *image,
  int32_t *status
) {
  fake_context *context = (fake_context *)raw_context;
  fake_module *value = (fake_module *)moonbit_make_external_object(
    module_finalize,
    sizeof(fake_module)
  );
  memset(value, 0, sizeof(*value));
  if (context == NULL || context->live == 0 || Moonbit_array_length(image) <= 0) {
    *status = -2;
    return value;
  }
  value->context = context;
  value->live = 1;
  context->children += 1;
  lunaflux_device_worker_fake_resource_opened_internal();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_module_close(void *raw) {
  fake_module *value = (fake_module *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  if (value->children != 0) return -4;
  value->live = 0;
  value->context->children -= 1;
  value->context = NULL;
  lunaflux_device_worker_fake_resource_closed_internal();
  return 0;
}

void *lunaflux_device_worker_fake_function_load(
  void *raw_module,
  uint8_t *name,
  int32_t *status
) {
  fake_module *module = (fake_module *)raw_module;
  fake_function *value = (fake_function *)moonbit_make_external_object(
    function_finalize,
    sizeof(fake_function)
  );
  memset(value, 0, sizeof(*value));
  if (module == NULL || module->live == 0 || Moonbit_array_length(name) <= 0) {
    *status = -2;
    return value;
  }
  value->module = module;
  value->live = 1;
  value->emits_logits =
    Moonbit_array_length(name) == 8 && memcmp(name, "kernel_9", 8U) == 0;
  module->children += 1;
  lunaflux_device_worker_fake_resource_opened_internal();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_function_close(void *raw) {
  fake_function *value = (fake_function *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  value->live = 0;
  value->module->children -= 1;
  value->module = NULL;
  lunaflux_device_worker_fake_resource_closed_internal();
  return 0;
}

int32_t lunaflux_device_worker_fake_function_launch_counted(
  void *raw_function,
  void *raw_stream,
  int32_t grid_x,
  int32_t grid_y,
  int32_t grid_z,
  int32_t block_x,
  int32_t block_y,
  int32_t block_z,
  int32_t shared,
  void **allocations,
  int64_t *offsets,
  int64_t *counts,
  int64_t *alignments,
  int length,
  int synchronize
) {
  fake_function *function = (fake_function *)raw_function;
  fake_stream *stream = (fake_stream *)raw_stream;
  if (function == NULL || stream == NULL || function->live == 0 ||
      stream->live == 0 || grid_x <= 0 || grid_y <= 0 || grid_z <= 0 ||
      block_x <= 0 || block_y <= 0 || block_z <= 0 || shared < 0 || length <= 0) {
    return -2;
  }
  for (int index = 0; index < length; ++index) {
    if (!lunaflux_device_worker_fake_valid_region_internal(
      stream->context,
      (fake_allocation *)allocations[index],
      offsets[index],
      counts[index],
      alignments[index]
    )) return -2;
  }
  if (lunaflux_device_worker_fake_fault == 1) {
    lunaflux_device_worker_fake_fault = 0;
    return -5;
  }
  if (function->emits_logits != 0) {
    int output = length - 1;
    fake_allocation *allocation = (fake_allocation *)allocations[output];
    if (counts[output] < 128L) return -2;
    uint8_t *base = allocation->storage + (size_t)offsets[output];
    for (int row = 0; row < 8; ++row) {
      int first = row % 5;
      for (int token = 0; token < 8; ++token) {
        size_t byte = (size_t)(row * 8 + token) * 2U;
        int rank = token - first;
        uint16_t bits = rank == 0 ? 0x4080U : rank == 1 ? 0x4040U :
          rank == 2 ? 0x4000U : rank == 3 ? 0x3f80U : 0U;
        base[byte] = (uint8_t)(bits & 0xffU);
        base[byte + 1U] = (uint8_t)(bits >> 8);
      }
    }
  }
  lunaflux_device_worker_fake_launch_calls += 1U;
  if (synchronize != 0) lunaflux_device_worker_fake_sync_calls += 1U;
  return 0;
}

int32_t lunaflux_device_worker_fake_function_launch(
  void *raw_function,
  void *raw_stream,
  int32_t grid_x,
  int32_t grid_y,
  int32_t grid_z,
  int32_t block_x,
  int32_t block_y,
  int32_t block_z,
  int32_t shared,
  void **allocations,
  int64_t *offsets,
  int64_t *counts,
  int64_t *alignments
) {
  int length = Moonbit_array_length(allocations);
  if (Moonbit_array_length(offsets) != length ||
      Moonbit_array_length(counts) != length ||
      Moonbit_array_length(alignments) != length) return -2;
  return lunaflux_device_worker_fake_function_launch_counted(
    raw_function, raw_stream, grid_x, grid_y, grid_z, block_x, block_y, block_z,
    shared, allocations, offsets, counts, alignments, length, 1
  );
}
