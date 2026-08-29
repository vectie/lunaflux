#include "resource_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int32_t lunaflux_cuda_test_ordered_executor(int32_t cycles);
int32_t lunaflux_cuda_test_ordered_graph(int32_t cycles);

static void *lf_probe_object(
  uint32_t payload_size,
  uint32_t meta,
  int kind
) {
  struct moonbit_object *header = calloc(
    1, sizeof(struct moonbit_object) + payload_size + sizeof(void *)
  );
  if (header == NULL) abort();
  header->rc = (int32_t)(
    (1u << MOONBIT_RC_COUNT_SHIFT) |
    ((uint32_t)kind & MOONBIT_RC_KIND_MASK)
  );
  header->meta = meta;
  return header + 1;
}

void *moonbit_make_external_object(
  void (*finalize)(void *self),
  uint32_t payload_size
) {
  void *payload = lf_probe_object(
    payload_size,
    payload_size,
    moonbit_BLOCK_KIND_REGULAR
  );
  memcpy((uint8_t *)payload + payload_size, &finalize, sizeof(finalize));
  return payload;
}

static void *lf_probe_array(int32_t length, size_t element_size, int kind) {
  if (length < 0 || (size_t)length > SIZE_MAX / element_size) abort();
  return lf_probe_object((uint32_t)((size_t)length * element_size), length, kind);
}

int32_t *moonbit_make_int32_array_raw(int32_t length) {
  return lf_probe_array(length, sizeof(int32_t), moonbit_BLOCK_KIND_VAL_ARRAY);
}

int64_t *moonbit_make_int64_array_raw(int32_t length) {
  return lf_probe_array(length, sizeof(int64_t), moonbit_BLOCK_KIND_VAL_ARRAY);
}

int64_t *moonbit_make_int64_array(int32_t length, int64_t value) {
  int64_t *array = moonbit_make_int64_array_raw(length);
  for (int32_t index = 0; index < length; index += 1) array[index] = value;
  return array;
}

void **moonbit_make_extern_ref_array_raw(int32_t length) {
  return lf_probe_array(length, sizeof(void *), moonbit_BLOCK_KIND_REF_ARRAY);
}

void moonbit_incref(void *object) {
  if (object != NULL) {
    Moonbit_object_header(object)->rc +=
      (int32_t)(1u << MOONBIT_RC_COUNT_SHIFT);
  }
}

void moonbit_decref(void *object) {
  if (object == NULL) return;
  struct moonbit_object *header = Moonbit_object_header(object);
  header->rc -= (int32_t)(1u << MOONBIT_RC_COUNT_SHIFT);
  if (Moonbit_rc_count(header) != 0) return;
  if (Moonbit_object_kind(object) == moonbit_BLOCK_KIND_REGULAR) {
    void (*finalize)(void *) = NULL;
    memcpy(&finalize, (uint8_t *)object + header->meta, sizeof(finalize));
    if (finalize != NULL) finalize(object);
  }
  free(header);
}

int32_t lf_cuda_map_result(CUresult result) {
  return result == 0 ? LF_OK : LF_DRIVER_FAILURE;
}

lf_cuda_api *lf_cuda_api_get(void) {
  return NULL;
}

int main(void) {
  int32_t result = lunaflux_cuda_test_ordered_executor(128);
  if (result != LF_OK) {
    fprintf(stderr, "ordered executor probe failed: %d\n", result);
    return 1;
  }
  result = lunaflux_cuda_test_ordered_graph(128);
  if (result != LF_OK) {
    fprintf(stderr, "ordered graph probe failed: %d\n", result);
    return 1;
  }
  return 0;
}
