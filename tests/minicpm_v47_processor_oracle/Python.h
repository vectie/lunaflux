/* Offline libImaging header shim, not a Python implementation or ABI.
 * Resample.c uses only C memory, image rows, and no-op section callbacks.
 * Opaque Python declarations allow its unchanged public headers to compile. */
#include <stddef.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define HAVE_PROTOTYPES 1
#define STDC_HEADERS 1
typedef struct _object PyObject;
typedef ptrdiff_t Py_ssize_t;
