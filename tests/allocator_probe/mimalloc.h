#ifndef LUNAFLUX_TEST_ALLOCATOR_PROBE_MIMALLOC_H
#define LUNAFLUX_TEST_ALLOCATOR_PROBE_MIMALLOC_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Native toolchains may use libc or mimalloc. Keeping this optional reference
 * permits libc builds to link without changing the runtime allocator/free pair.
 * If generated code actually calls the mimalloc redirect, absence is an error,
 * never permission to substitute malloc and free through a different allocator.
 */
#if defined(__APPLE__)
extern void *mi_malloc(size_t size) __attribute__((weak_import));
#else
extern void *mi_malloc(size_t size) __attribute__((weak));
#endif

static inline void *lunaflux_test_mimalloc(size_t size) {
  if (mi_malloc == NULL) {
    (void)fprintf(stderr, "allocation probe: runtime calls missing mi_malloc\n");
    abort();
  }
  return mi_malloc(size);
}

#endif
