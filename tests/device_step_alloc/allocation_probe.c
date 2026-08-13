#include "moonbit.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void *moonbit_malloc_array(
  enum moonbit_block_kind kind,
  int element_size_shift,
  int32_t length
);
extern moonbit_string_t moonbit_add_string(
  moonbit_string_t first,
  moonbit_string_t second
);
extern moonbit_string_t moonbit_unsafe_bytes_sub_string(
  moonbit_bytes_t bytes,
  int32_t start,
  int32_t length
);

static uint64_t probe_direct;
static uint64_t probe_array;
static uint64_t probe_string;
static int probe_active;
static const volatile void *probe_keep_alive;
static volatile int32_t probe_seed = 7;

static void probe_count(uint64_t *counter) {
  if (probe_active != 0) ++*counter;
}

void lunaflux_device_step_alloc_probe_begin(void) {
  probe_direct = 0U;
  probe_array = 0U;
  probe_string = 0U;
  probe_active = 1;
}

uint64_t lunaflux_device_step_alloc_probe_end(void) {
  probe_active = 0;
  return probe_direct + probe_array + probe_string;
}

int32_t lunaflux_device_step_alloc_probe_check(
  uint64_t observed,
  int32_t expect_zero
) {
  int valid = expect_zero != 0 ? observed == 0U : observed > 0U;
  if (!valid) {
    (void)fprintf(
      stderr,
      "device-step allocation gate failed: expected %s, observed=%" PRIu64
      " (direct=%" PRIu64 ", array=%" PRIu64 ", string=%" PRIu64 ")\n",
      expect_zero != 0 ? "zero stage/finish allocations" : "a positive control",
      observed,
      probe_direct,
      probe_array,
      probe_string
    );
  }
  return valid ? 1 : 0;
}

void lunaflux_device_step_alloc_probe_abort(void) { abort(); }

void lunaflux_device_step_alloc_probe_keep_alive(const void *value) {
  probe_keep_alive = value;
}

int32_t lunaflux_device_step_alloc_probe_seed(void) { return probe_seed; }

void *lunaflux_device_step_probe_malloc(size_t size) {
  probe_count(&probe_direct);
  return libc_malloc(size);
}

void *lunaflux_device_step_probe_malloc_array(
  enum moonbit_block_kind kind,
  int element_size_shift,
  int32_t length
) {
  probe_count(&probe_array);
  return moonbit_malloc_array(kind, element_size_shift, length);
}

#define ARRAY_WRAPPER(name, result, arguments, invocation)       \
  result lunaflux_device_step_probe_##name arguments {           \
    probe_count(&probe_array);                                   \
    return moonbit_##name invocation;                            \
  }

ARRAY_WRAPPER(make_bytes, moonbit_bytes_t, (int32_t n, int v), (n, v))
ARRAY_WRAPPER(make_bytes_raw, moonbit_bytes_t, (int32_t n), (n))
ARRAY_WRAPPER(make_int32_array, int32_t *, (int32_t n, int32_t v), (n, v))
ARRAY_WRAPPER(make_int32_array_raw, int32_t *, (int32_t n), (n))
ARRAY_WRAPPER(make_ref_array, void **, (int32_t n, void *v), (n, v))
ARRAY_WRAPPER(make_ref_array_raw, void **, (int32_t n), (n))
ARRAY_WRAPPER(make_int64_array, int64_t *, (int32_t n, int64_t v), (n, v))
ARRAY_WRAPPER(make_int64_array_raw, int64_t *, (int32_t n), (n))
ARRAY_WRAPPER(make_double_array, double *, (int32_t n, double v), (n, v))
ARRAY_WRAPPER(make_double_array_raw, double *, (int32_t n), (n))
ARRAY_WRAPPER(make_float_array, float *, (int32_t n, float v), (n, v))
ARRAY_WRAPPER(make_float_array_raw, float *, (int32_t n), (n))
ARRAY_WRAPPER(make_extern_ref_array, void **, (int32_t n, void *v), (n, v))
ARRAY_WRAPPER(make_extern_ref_array_raw, void **, (int32_t n), (n))
ARRAY_WRAPPER(
  make_v128_array,
  moonbit_v128_storage_t *,
  (int32_t n, uint64_t low, uint64_t high),
  (n, low, high)
)
ARRAY_WRAPPER(
  make_v128_array_raw,
  moonbit_v128_storage_t *,
  (int32_t n),
  (n)
)

void *lunaflux_device_step_probe_make_scalar_valtype_array(
  int32_t length,
  size_t value_size,
  void *initial
) {
  probe_count(&probe_array);
  return moonbit_make_scalar_valtype_array(length, value_size, initial);
}

void *lunaflux_device_step_probe_make_ref_valtype_array(
  int32_t length,
  size_t value_size,
  uint32_t header,
  void *initial
) {
  probe_count(&probe_array);
  return moonbit_make_ref_valtype_array(length, value_size, header, initial);
}

void *lunaflux_device_step_probe_make_scalar_valtype_array_raw(
  int32_t length,
  size_t value_size
) {
  probe_count(&probe_array);
  return moonbit_make_scalar_valtype_array_raw(length, value_size);
}

void *lunaflux_device_step_probe_make_ref_valtype_array_raw(
  int32_t length,
  size_t value_size,
  uint32_t header
) {
  probe_count(&probe_array);
  return moonbit_make_ref_valtype_array_raw(length, value_size, header);
}

void **lunaflux_device_step_probe_make_ref_array_with_blit(
  int32_t allocate_length,
  void *value,
  void *source,
  int32_t source_offset,
  int32_t destination_offset,
  int32_t length
) {
  probe_count(&probe_array);
  return moonbit_make_ref_array_with_blit(
    allocate_length,
    value,
    source,
    source_offset,
    destination_offset,
    length
  );
}

void *lunaflux_device_step_probe_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
) {
  probe_count(&probe_direct);
  return moonbit_make_external_object(finalize, payload_size);
}

moonbit_string_t lunaflux_device_step_probe_make_string(
  int32_t size,
  uint16_t value
) {
  probe_count(&probe_string);
  return moonbit_make_string(size, value);
}

moonbit_string_t lunaflux_device_step_probe_make_string_raw(int32_t size) {
  probe_count(&probe_string);
  return moonbit_make_string_raw(size);
}

moonbit_string_t lunaflux_device_step_probe_add_string(
  moonbit_string_t first,
  moonbit_string_t second
) {
  probe_count(&probe_string);
  return moonbit_add_string(first, second);
}

moonbit_string_t lunaflux_device_step_probe_bytes_sub_string(
  moonbit_bytes_t bytes,
  int32_t start,
  int32_t length
) {
  probe_count(&probe_string);
  return moonbit_unsafe_bytes_sub_string(bytes, start, length);
}

typedef struct {
  int live;
  int children;
} fake_context;

typedef struct {
  fake_context *context;
  uint8_t *storage;
  size_t size;
  int live;
} fake_allocation;

static uint64_t fake_copy_calls;

static void fake_context_finalize(void *object) {
  fake_context *context = (fake_context *)object;
  if (context->live != 0 || context->children != 0) abort();
}

static void fake_allocation_finalize(void *object) {
  fake_allocation *allocation = (fake_allocation *)object;
  if (allocation->live != 0 || allocation->storage != NULL) abort();
}

int32_t lunaflux_device_step_fake_device_info(
  int32_t ordinal,
  int64_t *numeric,
  uint8_t *name
) {
  if (ordinal != 0 || numeric == NULL || name == NULL) return -2;
  numeric[0] = 1073741824L;
  numeric[1] = 8L;
  numeric[2] = 0L;
  numeric[3] = 1L;
  numeric[4] = 1L;
  name[0] = (uint8_t)'f';
  name[1] = (uint8_t)'a';
  name[2] = (uint8_t)'k';
  name[3] = (uint8_t)'e';
  name[4] = 0U;
  return 0;
}

void *lunaflux_device_step_fake_context_create(
  int32_t ordinal,
  int32_t *status
) {
  fake_context *context = (fake_context *)moonbit_make_external_object(
    fake_context_finalize,
    sizeof(fake_context)
  );
  memset(context, 0, sizeof(*context));
  if (ordinal != 0) {
    *status = -2;
    return context;
  }
  context->live = 1;
  *status = 0;
  return context;
}

int32_t lunaflux_device_step_fake_context_close(void *raw) {
  fake_context *context = (fake_context *)raw;
  if (context == NULL) return -2;
  if (context->live == 0) return 0;
  if (context->children != 0) return -4;
  context->live = 0;
  return 0;
}

void *lunaflux_device_step_fake_allocation_create(
  void *raw_context,
  int64_t byte_count,
  int32_t *status
) {
  fake_context *context = (fake_context *)raw_context;
  fake_allocation *allocation =
    (fake_allocation *)moonbit_make_external_object(
      fake_allocation_finalize,
      sizeof(fake_allocation)
    );
  memset(allocation, 0, sizeof(*allocation));
  if (context == NULL || context->live == 0) {
    *status = -3;
    return allocation;
  }
  if (byte_count <= 0 || (uint64_t)byte_count > SIZE_MAX) {
    *status = -2;
    return allocation;
  }
  allocation->storage = (uint8_t *)libc_malloc((size_t)byte_count);
  if (allocation->storage == NULL) {
    *status = -6;
    return allocation;
  }
  memset(allocation->storage, 0, (size_t)byte_count);
  allocation->context = context;
  allocation->size = (size_t)byte_count;
  allocation->live = 1;
  context->children += 1;
  *status = 0;
  return allocation;
}

int32_t lunaflux_device_step_fake_allocation_close(void *raw) {
  fake_allocation *allocation = (fake_allocation *)raw;
  if (allocation == NULL) return -2;
  if (allocation->live == 0) return 0;
  free(allocation->storage);
  allocation->storage = NULL;
  allocation->live = 0;
  allocation->context->children -= 1;
  allocation->context = NULL;
  return 0;
}

int32_t lunaflux_device_step_fake_validate_region(
  void *raw_context,
  void *raw_allocation,
  int64_t offset,
  int64_t byte_count,
  int64_t alignment
) {
  fake_context *context = (fake_context *)raw_context;
  fake_allocation *allocation = (fake_allocation *)raw_allocation;
  if (context == NULL || allocation == NULL || context->live == 0 ||
      allocation->live == 0) return -3;
  if (allocation->context != context || offset < 0 || byte_count <= 0 ||
      alignment <= 0 || (alignment & (alignment - 1)) != 0) return -2;
  if ((uint64_t)offset > SIZE_MAX || (uint64_t)byte_count > SIZE_MAX) return -7;
  size_t start = (size_t)offset;
  size_t count = (size_t)byte_count;
  if (start > allocation->size || count > allocation->size - start ||
      start % (size_t)alignment != 0) return -2;
  return 0;
}

int32_t lunaflux_device_step_fake_copy_fixed_to_device(
  void *raw_context,
  void *raw_allocation,
  uint8_t *source,
  int64_t source_offset,
  int64_t destination_offset,
  int64_t byte_count
) {
  fake_context *context = (fake_context *)raw_context;
  fake_allocation *allocation = (fake_allocation *)raw_allocation;
  if (context == NULL || allocation == NULL || source == NULL ||
      context->live == 0 || allocation->live == 0) return -3;
  if (allocation->context != context) return -2;
  if (source_offset < 0 || destination_offset < 0 || byte_count < 0 ||
      (uint64_t)source_offset > SIZE_MAX ||
      (uint64_t)destination_offset > SIZE_MAX ||
      (uint64_t)byte_count > SIZE_MAX) return -7;
  size_t source_size = (size_t)Moonbit_array_length(source);
  size_t source_start = (size_t)source_offset;
  size_t destination_start = (size_t)destination_offset;
  size_t count = (size_t)byte_count;
  if (source_start > source_size || count > source_size - source_start ||
      destination_start > allocation->size ||
      count > allocation->size - destination_start) return -2;
  memcpy(
    allocation->storage + destination_start,
    source + source_start,
    count
  );
  fake_copy_calls += 1U;
  return 0;
}

uint64_t lunaflux_device_step_fake_copy_count(void) { return fake_copy_calls; }
