#include "moonbit.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The counter is deliberately thread-confined, matching the scheduler owner. */

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

static uint64_t lunaflux_probe_direct_allocations;
static uint64_t lunaflux_probe_array_allocations;
static uint64_t lunaflux_probe_string_allocations;
static int lunaflux_probe_active;
static const volatile void *lunaflux_probe_keep_alive_value;
static volatile int32_t lunaflux_probe_seed_value = 7;

static void lunaflux_probe_count(uint64_t *counter) {
  if (lunaflux_probe_active != 0) {
    ++*counter;
  }
}

void lunaflux_alloc_probe_begin(void) {
  lunaflux_probe_direct_allocations = 0U;
  lunaflux_probe_array_allocations = 0U;
  lunaflux_probe_string_allocations = 0U;
  lunaflux_probe_active = 1;
}

uint64_t lunaflux_alloc_probe_end(void) {
  lunaflux_probe_active = 0;
  return lunaflux_probe_direct_allocations + lunaflux_probe_array_allocations +
    lunaflux_probe_string_allocations;
}

int32_t lunaflux_alloc_probe_check(uint64_t observed, int32_t expect_zero) {
  int valid = expect_zero != 0 ? observed == 0U : observed > 0U;
  if (!valid) {
    (void)fprintf(
      stderr,
      "allocation gate failed: expected %s, observed=%" PRIu64
      " (direct=%" PRIu64 ", array=%" PRIu64 ", string=%" PRIu64 ")\n",
      expect_zero != 0 ? "zero hot-path allocations" : "a positive control",
      observed,
      lunaflux_probe_direct_allocations,
      lunaflux_probe_array_allocations,
      lunaflux_probe_string_allocations
    );
  }
  return valid ? 1 : 0;
}

void lunaflux_alloc_probe_abort(void) {
  abort();
}

void lunaflux_alloc_probe_keep_alive(const void *value) {
  lunaflux_probe_keep_alive_value = value;
}

int32_t lunaflux_alloc_probe_seed(void) {
  return lunaflux_probe_seed_value;
}

void *lunaflux_probe_malloc(size_t size) {
  lunaflux_probe_count(&lunaflux_probe_direct_allocations);
  return libc_malloc(size);
}

void *lunaflux_probe_malloc_array(
  enum moonbit_block_kind kind,
  int element_size_shift,
  int32_t length
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_malloc_array(kind, element_size_shift, length);
}

#define LUNAFLUX_ARRAY_WRAPPER(name, result, arguments, invocation) \
  result lunaflux_probe_##name arguments {                           \
    lunaflux_probe_count(&lunaflux_probe_array_allocations);         \
    return moonbit_##name invocation;                               \
  }

LUNAFLUX_ARRAY_WRAPPER(
  make_bytes,
  moonbit_bytes_t,
  (int32_t size, int value),
  (size, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_bytes_raw,
  moonbit_bytes_t,
  (int32_t size),
  (size)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_int32_array,
  int32_t *,
  (int32_t length, int32_t value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_int32_array_raw,
  int32_t *,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_ref_array,
  void **,
  (int32_t length, void *value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_ref_array_raw,
  void **,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_int64_array,
  int64_t *,
  (int32_t length, int64_t value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_int64_array_raw,
  int64_t *,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_double_array,
  double *,
  (int32_t length, double value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_double_array_raw,
  double *,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_float_array,
  float *,
  (int32_t length, float value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_float_array_raw,
  float *,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_extern_ref_array,
  void **,
  (int32_t length, void *value),
  (length, value)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_extern_ref_array_raw,
  void **,
  (int32_t length),
  (length)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_v128_array,
  moonbit_v128_storage_t *,
  (int32_t length, uint64_t low, uint64_t high),
  (length, low, high)
)
LUNAFLUX_ARRAY_WRAPPER(
  make_v128_array_raw,
  moonbit_v128_storage_t *,
  (int32_t length),
  (length)
)

void *lunaflux_probe_make_scalar_valtype_array(
  int32_t length,
  size_t value_size,
  void *initial
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_make_scalar_valtype_array(length, value_size, initial);
}

void *lunaflux_probe_make_ref_valtype_array(
  int32_t length,
  size_t value_size,
  uint32_t header,
  void *initial
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_make_ref_valtype_array(length, value_size, header, initial);
}

void *lunaflux_probe_make_scalar_valtype_array_raw(
  int32_t length,
  size_t value_size
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_make_scalar_valtype_array_raw(length, value_size);
}

void *lunaflux_probe_make_ref_valtype_array_raw(
  int32_t length,
  size_t value_size,
  uint32_t header
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_make_ref_valtype_array_raw(length, value_size, header);
}

void **lunaflux_probe_make_ref_array_with_blit(
  int32_t allocate_length,
  void *value,
  void *source,
  int32_t source_offset,
  int32_t destination_offset,
  int32_t length
) {
  lunaflux_probe_count(&lunaflux_probe_array_allocations);
  return moonbit_make_ref_array_with_blit(
    allocate_length,
    value,
    source,
    source_offset,
    destination_offset,
    length
  );
}

void *lunaflux_probe_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
) {
  lunaflux_probe_count(&lunaflux_probe_direct_allocations);
  return moonbit_make_external_object(finalize, payload_size);
}

moonbit_string_t lunaflux_probe_make_string(int32_t size, uint16_t value) {
  lunaflux_probe_count(&lunaflux_probe_string_allocations);
  return moonbit_make_string(size, value);
}

moonbit_string_t lunaflux_probe_make_string_raw(int32_t size) {
  lunaflux_probe_count(&lunaflux_probe_string_allocations);
  return moonbit_make_string_raw(size);
}

moonbit_string_t lunaflux_probe_add_string(
  moonbit_string_t first,
  moonbit_string_t second
) {
  lunaflux_probe_count(&lunaflux_probe_string_allocations);
  return moonbit_add_string(first, second);
}

moonbit_string_t lunaflux_probe_bytes_sub_string(
  moonbit_bytes_t bytes,
  int32_t start,
  int32_t length
) {
  lunaflux_probe_count(&lunaflux_probe_string_allocations);
  return moonbit_unsafe_bytes_sub_string(bytes, start, length);
}
