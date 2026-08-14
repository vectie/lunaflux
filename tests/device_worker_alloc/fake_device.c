#include "moonbit.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct fake_context fake_context;
typedef struct fake_allocation fake_allocation;
typedef struct fake_module fake_module;

struct fake_context {
  int live;
  int children;
};

struct fake_allocation {
  fake_context *context;
  uint8_t *storage;
  size_t size;
  int live;
  int leases;
};

typedef struct {
  fake_context *context;
  int live;
} fake_stream;

typedef struct {
  fake_allocation *allocation;
  int live;
} fake_lease;

struct fake_module {
  fake_context *context;
  int live;
  int children;
};

typedef struct {
  fake_module *module;
  int live;
} fake_function;

static uint64_t copy_calls;
static uint64_t launch_calls;
static uint64_t sync_calls;
static uint64_t readback_calls;
static uint64_t resource_create_calls;
static uint64_t resource_close_calls;
static int live_resources;
static int fault_mode;

static void resource_opened(void) {
  live_resources += 1;
  resource_create_calls += 1U;
}

static void resource_closed(void) {
  live_resources -= 1;
  resource_close_calls += 1U;
}

static void context_finalize(void *raw) {
  fake_context *value = (fake_context *)raw;
  if (value->live != 0 || value->children != 0) abort();
}

static void allocation_finalize(void *raw) {
  fake_allocation *value = (fake_allocation *)raw;
  if (value->live != 0 || value->storage != NULL || value->leases != 0) abort();
}

static void stream_finalize(void *raw) {
  if (((fake_stream *)raw)->live != 0) abort();
}

static void lease_finalize(void *raw) {
  if (((fake_lease *)raw)->live != 0) abort();
}

static void module_finalize(void *raw) {
  fake_module *value = (fake_module *)raw;
  if (value->live != 0 || value->children != 0) abort();
}

static void function_finalize(void *raw) {
  if (((fake_function *)raw)->live != 0) abort();
}

int32_t lunaflux_device_worker_fake_device_info(
  int32_t ordinal,
  int64_t *numeric,
  uint8_t *name
) {
  if (ordinal != 0 || numeric == NULL || name == NULL) return -2;
  numeric[0] = 1073741824L;
  numeric[1] = 8L;
  numeric[2] = 0L;
  numeric[3] = 1L;
  numeric[4] = 0L;
  memcpy(name, "fake", 5U);
  return 0;
}

void *lunaflux_device_worker_fake_context_create(
  int32_t ordinal,
  int32_t *status
) {
  fake_context *value = (fake_context *)moonbit_make_external_object(
    context_finalize,
    sizeof(fake_context)
  );
  memset(value, 0, sizeof(*value));
  if (ordinal != 0) {
    *status = -2;
    return value;
  }
  value->live = 1;
  resource_opened();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_context_close(void *raw) {
  fake_context *value = (fake_context *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  if (value->children != 0) return -4;
  value->live = 0;
  resource_closed();
  return 0;
}

void *lunaflux_device_worker_fake_allocation_create(
  void *raw_context,
  int64_t byte_count,
  int32_t *status
) {
  fake_context *context = (fake_context *)raw_context;
  fake_allocation *value = (fake_allocation *)moonbit_make_external_object(
    allocation_finalize,
    sizeof(fake_allocation)
  );
  memset(value, 0, sizeof(*value));
  if (context == NULL || context->live == 0 || byte_count <= 0 ||
      (uint64_t)byte_count > SIZE_MAX) {
    *status = -2;
    return value;
  }
  value->storage = (uint8_t *)libc_malloc((size_t)byte_count);
  if (value->storage == NULL) {
    *status = -6;
    return value;
  }
  memset(value->storage, 0, (size_t)byte_count);
  value->context = context;
  value->size = (size_t)byte_count;
  value->live = 1;
  context->children += 1;
  resource_opened();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_allocation_close(void *raw) {
  fake_allocation *value = (fake_allocation *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  if (value->leases != 0) return -4;
  free(value->storage);
  value->storage = NULL;
  value->live = 0;
  value->context->children -= 1;
  value->context = NULL;
  resource_closed();
  return 0;
}

void *lunaflux_device_worker_fake_stream_create(
  void *raw_context,
  int32_t *status
) {
  fake_context *context = (fake_context *)raw_context;
  fake_stream *value = (fake_stream *)moonbit_make_external_object(
    stream_finalize,
    sizeof(fake_stream)
  );
  memset(value, 0, sizeof(*value));
  if (context == NULL || context->live == 0) {
    *status = -3;
    return value;
  }
  value->context = context;
  value->live = 1;
  context->children += 1;
  resource_opened();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_stream_close(void *raw) {
  fake_stream *value = (fake_stream *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  value->live = 0;
  value->context->children -= 1;
  value->context = NULL;
  resource_closed();
  return 0;
}

void *lunaflux_device_worker_fake_lease_create(
  void *raw_context,
  void *raw_allocation,
  int32_t *status
) {
  fake_context *context = (fake_context *)raw_context;
  fake_allocation *allocation = (fake_allocation *)raw_allocation;
  fake_lease *value = (fake_lease *)moonbit_make_external_object(
    lease_finalize,
    sizeof(fake_lease)
  );
  memset(value, 0, sizeof(*value));
  if (context == NULL || allocation == NULL || context->live == 0 ||
      allocation->live == 0 || allocation->context != context) {
    *status = -3;
    return value;
  }
  value->allocation = allocation;
  value->live = 1;
  allocation->leases += 1;
  resource_opened();
  *status = 0;
  return value;
}

int32_t lunaflux_device_worker_fake_lease_close(void *raw) {
  fake_lease *value = (fake_lease *)raw;
  if (value == NULL) return -2;
  if (value->live == 0) return 0;
  value->allocation->leases -= 1;
  value->allocation = NULL;
  value->live = 0;
  resource_closed();
  return 0;
}

static int valid_region(
  fake_context *context,
  fake_allocation *allocation,
  int64_t offset,
  int64_t count,
  int64_t alignment
) {
  if (context == NULL || allocation == NULL || context->live == 0 ||
      allocation->live == 0 || allocation->context != context || offset < 0 ||
      count <= 0 || alignment <= 0) return 0;
  if ((uint64_t)offset > SIZE_MAX || (uint64_t)count > SIZE_MAX) return 0;
  size_t start = (size_t)offset;
  size_t size = (size_t)count;
  return start <= allocation->size && size <= allocation->size - start &&
    start % (size_t)alignment == 0U;
}

int32_t lunaflux_device_worker_fake_validate_region(
  void *context,
  void *allocation,
  int64_t offset,
  int64_t count,
  int64_t alignment
) {
  return valid_region(context, allocation, offset, count, alignment) ? 0 : -2;
}

static int32_t copy_into(
  fake_allocation *allocation,
  const uint8_t *source,
  size_t source_size,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t count
) {
  if (allocation == NULL || allocation->live == 0 || source == NULL ||
      source_offset < 0 || destination_offset < 0 || count < 0) return -2;
  size_t from = (size_t)source_offset;
  size_t to = (size_t)destination_offset;
  size_t size = (size_t)count;
  if (from > source_size || size > source_size - from ||
      to > allocation->size || size > allocation->size - to) return -2;
  memcpy(allocation->storage + to, source + from, size);
  copy_calls += 1U;
  return 0;
}

int32_t lunaflux_device_worker_fake_copy_to_device(
  void *allocation,
  uint8_t *source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t count
) {
  return copy_into(
    allocation,
    source,
    (size_t)Moonbit_array_length(source),
    source_offset,
    destination_offset,
    count
  );
}

int32_t lunaflux_device_worker_fake_copy_fixed_to_device(
  void *context,
  void *allocation,
  uint8_t *source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t count
) {
  fake_allocation *value = (fake_allocation *)allocation;
  if (!valid_region(context, value, destination_offset, count, 1L)) return -2;
  return copy_into(
    value,
    source,
    (size_t)Moonbit_array_length(source),
    source_offset,
    destination_offset,
    count
  );
}

int32_t lunaflux_device_worker_fake_copy_device_to_fixed(
  void *context,
  void *allocation,
  uint8_t *destination,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t count
) {
  fake_allocation *value = (fake_allocation *)allocation;
  if (fault_mode == 2) {
    fault_mode = 0;
    return -5;
  }
  if (!valid_region(context, value, source_offset, count, 1L) ||
      destination_offset < 0 || count < 0) return -2;
  size_t to = (size_t)destination_offset;
  size_t size = (size_t)count;
  if (to > (size_t)Moonbit_array_length(destination) ||
      size > (size_t)Moonbit_array_length(destination) - to) return -2;
  memcpy(destination + to, value->storage + (size_t)source_offset, size);
  if (fault_mode == 3 && size >= 2U) {
    destination[to] = 0xc0U;
    destination[to + 1U] = 0x7fU;
    fault_mode = 0;
  }
  readback_calls += 1U;
  return 0;
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
  resource_opened();
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
  resource_closed();
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
  module->children += 1;
  resource_opened();
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
  resource_closed();
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
  fake_function *function = (fake_function *)raw_function;
  fake_stream *stream = (fake_stream *)raw_stream;
  int length = Moonbit_array_length(allocations);
  if (function == NULL || stream == NULL || function->live == 0 ||
      stream->live == 0 || grid_x <= 0 || grid_y <= 0 || grid_z <= 0 ||
      block_x <= 0 || block_y <= 0 || block_z <= 0 || shared < 0 || length <= 0 ||
      Moonbit_array_length(offsets) != length ||
      Moonbit_array_length(counts) != length ||
      Moonbit_array_length(alignments) != length) return -2;
  for (int index = 0; index < length; ++index) {
    if (!valid_region(
      stream->context,
      (fake_allocation *)allocations[index],
      offsets[index],
      counts[index],
      alignments[index]
    )) return -2;
  }
  if (fault_mode == 1) {
    fault_mode = 0;
    return -5;
  }
  launch_calls += 1U;
  sync_calls += 1U;
  return 0;
}

uint64_t lunaflux_device_worker_fake_copy_count(void) { return copy_calls; }
uint64_t lunaflux_device_worker_fake_launch_count(void) { return launch_calls; }
uint64_t lunaflux_device_worker_fake_sync_count(void) { return sync_calls; }
uint64_t lunaflux_device_worker_fake_readback_count(void) { return readback_calls; }
uint64_t lunaflux_device_worker_fake_resource_create_count(void) {
  return resource_create_calls;
}
uint64_t lunaflux_device_worker_fake_resource_close_count(void) {
  return resource_close_calls;
}
int32_t lunaflux_device_worker_fake_live_children(void) { return live_resources; }
void lunaflux_device_worker_fake_set_fault(int32_t mode) { fault_mode = mode; }
int32_t lunaflux_device_worker_fake_fault_mode(void) { return fault_mode; }
