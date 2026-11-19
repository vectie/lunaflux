/* Test-only byte ABI shim. The production boundary includes real moonbit.h. */
#ifndef LUNAFLUX_RGB8_TEST_MOONBIT_H
#define LUNAFLUX_RGB8_TEST_MOONBIT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define MOONBIT_FFI_EXPORT
typedef uint8_t *moonbit_bytes_t;
static int Moonbit_array_length(moonbit_bytes_t bytes) {
  return ((int32_t *)bytes)[-1];
}
static moonbit_bytes_t moonbit_make_bytes(int32_t size, int value) {
  int32_t *prefix = (int32_t *)malloc(sizeof(int32_t) + (size_t)size);
  if (!prefix) abort();
  *prefix = size;
  moonbit_bytes_t bytes = (moonbit_bytes_t)(prefix + 1);
  memset(bytes, value, (size_t)size);
  return bytes;
}
static void release_bytes(moonbit_bytes_t bytes) { free(((int32_t *)bytes) - 1); }
#endif
