#include "moonbit.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void *moonbit_malloc_array(
  enum moonbit_block_kind kind,
  int element_size_shift,
  int32_t length
);
extern moonbit_string_t moonbit_add_string(moonbit_string_t, moonbit_string_t);
extern moonbit_string_t moonbit_unsafe_bytes_sub_string(
  moonbit_bytes_t,
  int32_t,
  int32_t
);

static uint64_t direct_count;
static uint64_t array_count;
static uint64_t string_count;
static int active;
static const volatile void *keep_alive;
static volatile int32_t seed = 7;

static void count(uint64_t *value) {
  if (active != 0) ++*value;
}

void lunaflux_device_worker_alloc_probe_begin(void) {
  direct_count = 0U;
  array_count = 0U;
  string_count = 0U;
  active = 1;
}

uint64_t lunaflux_device_worker_alloc_probe_end(void) {
  active = 0;
  return direct_count + array_count + string_count;
}

uint64_t lunaflux_device_worker_alloc_probe_direct_count(void) {
  return direct_count;
}

uint64_t lunaflux_device_worker_alloc_probe_array_count(void) {
  return array_count;
}

uint64_t lunaflux_device_worker_alloc_probe_string_count(void) {
  return string_count;
}

int32_t lunaflux_device_worker_alloc_probe_check(
  uint64_t observed,
  int32_t expect_zero
) {
  int valid = expect_zero != 0 ? observed == 0U : observed > 0U;
  if (!valid) {
    (void)fprintf(
      stderr,
      "device-worker allocation gate failed: expected %s, observed=%" PRIu64
      " (direct=%" PRIu64 ", array=%" PRIu64 ", string=%" PRIu64 ")\n",
      expect_zero != 0 ? "zero execute lifecycle allocations" : "positive control",
      observed,
      direct_count,
      array_count,
      string_count
    );
  }
  return valid ? 1 : 0;
}

void lunaflux_device_worker_alloc_probe_abort(void) { abort(); }
void lunaflux_device_worker_alloc_probe_keep_alive(const void *value) {
  keep_alive = value;
}
int32_t lunaflux_device_worker_alloc_probe_seed(void) { return seed; }

void *lunaflux_device_worker_probe_malloc(size_t size) {
  count(&direct_count);
  return libc_malloc(size);
}

void *lunaflux_device_worker_probe_malloc_raw(size_t size) {
  count(&direct_count);
  return libc_malloc(size);
}

void *lunaflux_device_worker_probe_malloc_array(
  enum moonbit_block_kind kind,
  int shift,
  int32_t length
) {
  count(&array_count);
  return moonbit_malloc_array(kind, shift, length);
}

#define ARRAY_WRAPPER(name, result, arguments, invocation)       \
  result lunaflux_device_worker_probe_##name arguments {         \
    count(&array_count);                                         \
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

void *lunaflux_device_worker_probe_make_scalar_valtype_array(
  int32_t length,
  size_t size,
  void *initial
) {
  count(&array_count);
  return moonbit_make_scalar_valtype_array(length, size, initial);
}

void *lunaflux_device_worker_probe_make_ref_valtype_array(
  int32_t length,
  size_t size,
  uint32_t header,
  void *initial
) {
  count(&array_count);
  return moonbit_make_ref_valtype_array(length, size, header, initial);
}

void *lunaflux_device_worker_probe_make_scalar_valtype_array_raw(
  int32_t length,
  size_t size
) {
  count(&array_count);
  return moonbit_make_scalar_valtype_array_raw(length, size);
}

void *lunaflux_device_worker_probe_make_ref_valtype_array_raw(
  int32_t length,
  size_t size,
  uint32_t header
) {
  count(&array_count);
  return moonbit_make_ref_valtype_array_raw(length, size, header);
}

void **lunaflux_device_worker_probe_make_ref_array_with_blit(
  int32_t length,
  void *value,
  void *source,
  int32_t source_offset,
  int32_t destination_offset,
  int32_t count_value
) {
  count(&array_count);
  return moonbit_make_ref_array_with_blit(
    length,
    value,
    source,
    source_offset,
    destination_offset,
    count_value
  );
}

void *lunaflux_device_worker_probe_make_external_object(
  void (*finalize)(void *),
  uint32_t size
) {
  count(&direct_count);
  return moonbit_make_external_object(finalize, size);
}

moonbit_string_t lunaflux_device_worker_probe_make_string(
  int32_t size,
  uint16_t value
) {
  count(&string_count);
  return moonbit_make_string(size, value);
}

moonbit_string_t lunaflux_device_worker_probe_make_string_raw(int32_t size) {
  count(&string_count);
  return moonbit_make_string_raw(size);
}

moonbit_string_t lunaflux_device_worker_probe_add_string(
  moonbit_string_t left,
  moonbit_string_t right
) {
  count(&string_count);
  return moonbit_add_string(left, right);
}

moonbit_string_t lunaflux_device_worker_probe_bytes_sub_string(
  moonbit_bytes_t bytes,
  int32_t start,
  int32_t length
) {
  count(&string_count);
  return moonbit_unsafe_bytes_sub_string(bytes, start, length);
}
