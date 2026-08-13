#ifndef LUNAFLUX_DEVICE_STEP_ALLOCATION_REDIRECT_H
#define LUNAFLUX_DEVICE_STEP_ALLOCATION_REDIRECT_H

/*
 * Force-included only for the native release allocation harness. Generated
 * MoonBit allocation calls are counted, while the public device API is routed
 * through a deterministic in-process CUDA seam with real fixed-buffer copies.
 */

#include "moonbit.h"

#include <stddef.h>
#include <stdint.h>

void *lunaflux_device_step_probe_malloc(size_t size);
void *lunaflux_device_step_probe_malloc_array(
  enum moonbit_block_kind kind,
  int element_size_shift,
  int32_t length
);
moonbit_string_t lunaflux_device_step_probe_make_string(
  int32_t size,
  uint16_t value
);
moonbit_string_t lunaflux_device_step_probe_make_string_raw(int32_t size);
moonbit_bytes_t lunaflux_device_step_probe_make_bytes(int32_t size, int value);
moonbit_bytes_t lunaflux_device_step_probe_make_bytes_raw(int32_t size);
int32_t *lunaflux_device_step_probe_make_int32_array(
  int32_t length,
  int32_t value
);
int32_t *lunaflux_device_step_probe_make_int32_array_raw(int32_t length);
void **lunaflux_device_step_probe_make_ref_array(int32_t length, void *value);
void **lunaflux_device_step_probe_make_ref_array_raw(int32_t length);
int64_t *lunaflux_device_step_probe_make_int64_array(
  int32_t length,
  int64_t value
);
int64_t *lunaflux_device_step_probe_make_int64_array_raw(int32_t length);
double *lunaflux_device_step_probe_make_double_array(
  int32_t length,
  double value
);
double *lunaflux_device_step_probe_make_double_array_raw(int32_t length);
float *lunaflux_device_step_probe_make_float_array(
  int32_t length,
  float value
);
float *lunaflux_device_step_probe_make_float_array_raw(int32_t length);
void **lunaflux_device_step_probe_make_extern_ref_array(
  int32_t length,
  void *value
);
void **lunaflux_device_step_probe_make_extern_ref_array_raw(int32_t length);
moonbit_v128_storage_t *lunaflux_device_step_probe_make_v128_array(
  int32_t length,
  uint64_t low,
  uint64_t high
);
moonbit_v128_storage_t *lunaflux_device_step_probe_make_v128_array_raw(
  int32_t length
);
void *lunaflux_device_step_probe_make_scalar_valtype_array(
  int32_t length,
  size_t value_size,
  void *initial
);
void *lunaflux_device_step_probe_make_ref_valtype_array(
  int32_t length,
  size_t value_size,
  uint32_t header,
  void *initial
);
void *lunaflux_device_step_probe_make_scalar_valtype_array_raw(
  int32_t length,
  size_t value_size
);
void *lunaflux_device_step_probe_make_ref_valtype_array_raw(
  int32_t length,
  size_t value_size,
  uint32_t header
);
void **lunaflux_device_step_probe_make_ref_array_with_blit(
  int32_t allocate_length,
  void *value,
  void *source,
  int32_t source_offset,
  int32_t destination_offset,
  int32_t length
);
void *lunaflux_device_step_probe_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
);
moonbit_string_t lunaflux_device_step_probe_add_string(
  moonbit_string_t first,
  moonbit_string_t second
);
moonbit_string_t lunaflux_device_step_probe_bytes_sub_string(
  moonbit_bytes_t bytes,
  int32_t start,
  int32_t length
);

#define malloc lunaflux_device_step_probe_malloc
#define moonbit_malloc_array lunaflux_device_step_probe_malloc_array
#define moonbit_make_string lunaflux_device_step_probe_make_string
#define moonbit_make_string_raw lunaflux_device_step_probe_make_string_raw
#define moonbit_make_bytes lunaflux_device_step_probe_make_bytes
#define moonbit_make_bytes_raw lunaflux_device_step_probe_make_bytes_raw
#define moonbit_make_int32_array lunaflux_device_step_probe_make_int32_array
#define moonbit_make_int32_array_raw lunaflux_device_step_probe_make_int32_array_raw
#define moonbit_make_ref_array lunaflux_device_step_probe_make_ref_array
#define moonbit_make_ref_array_raw lunaflux_device_step_probe_make_ref_array_raw
#define moonbit_make_int64_array lunaflux_device_step_probe_make_int64_array
#define moonbit_make_int64_array_raw lunaflux_device_step_probe_make_int64_array_raw
#define moonbit_make_double_array lunaflux_device_step_probe_make_double_array
#define moonbit_make_double_array_raw lunaflux_device_step_probe_make_double_array_raw
#define moonbit_make_float_array lunaflux_device_step_probe_make_float_array
#define moonbit_make_float_array_raw lunaflux_device_step_probe_make_float_array_raw
#define moonbit_make_extern_ref_array lunaflux_device_step_probe_make_extern_ref_array
#define moonbit_make_extern_ref_array_raw lunaflux_device_step_probe_make_extern_ref_array_raw
#define moonbit_make_v128_array lunaflux_device_step_probe_make_v128_array
#define moonbit_make_v128_array_raw lunaflux_device_step_probe_make_v128_array_raw
#define moonbit_make_scalar_valtype_array lunaflux_device_step_probe_make_scalar_valtype_array
#define moonbit_make_ref_valtype_array lunaflux_device_step_probe_make_ref_valtype_array
#define moonbit_make_scalar_valtype_array_raw lunaflux_device_step_probe_make_scalar_valtype_array_raw
#define moonbit_make_ref_valtype_array_raw lunaflux_device_step_probe_make_ref_valtype_array_raw
#define moonbit_make_ref_array_with_blit lunaflux_device_step_probe_make_ref_array_with_blit
#define moonbit_make_external_object lunaflux_device_step_probe_make_external_object
#define moonbit_add_string lunaflux_device_step_probe_add_string
#define moonbit_unsafe_bytes_sub_string lunaflux_device_step_probe_bytes_sub_string

#define lunaflux_cuda_device_info lunaflux_device_step_fake_device_info
#define lunaflux_cuda_context_create lunaflux_device_step_fake_context_create
#define lunaflux_cuda_context_close lunaflux_device_step_fake_context_close
#define lunaflux_cuda_allocation_create lunaflux_device_step_fake_allocation_create
#define lunaflux_cuda_allocation_close lunaflux_device_step_fake_allocation_close
#define lunaflux_cuda_context_validate_allocation_region lunaflux_device_step_fake_validate_region
#define lunaflux_cuda_context_copy_fixed_to_device lunaflux_device_step_fake_copy_fixed_to_device

#endif
