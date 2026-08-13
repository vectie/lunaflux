#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <stdint.h>
#include <time.h>

#include "monotonic_clock_status.h"

#ifndef LF_MONOTONIC_CLOCK_GETTIME
#define LF_MONOTONIC_CLOCK_GETTIME clock_gettime
#endif

static int32_t lf_monotonic_clock_read(uint64_t *output) {
  if (output == NULL) return LF_MONOTONIC_CLOCK_UNAVAILABLE;

  struct timespec value;
  if (LF_MONOTONIC_CLOCK_GETTIME(CLOCK_MONOTONIC, &value) != 0) {
    return LF_MONOTONIC_CLOCK_UNAVAILABLE;
  }
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000L) {
    return LF_MONOTONIC_CLOCK_OUT_OF_RANGE;
  }

  uint64_t seconds = (uint64_t)value.tv_sec;
  uint64_t milliseconds = (uint64_t)value.tv_nsec / 1000000U;
  if (seconds > (UINT64_MAX - milliseconds) / 1000U) {
    return LF_MONOTONIC_CLOCK_OUT_OF_RANGE;
  }
  *output = seconds * 1000U + milliseconds;
  return LF_MONOTONIC_CLOCK_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_monotonic_clock_now_millis(uint64_t *output) {
  return lf_monotonic_clock_read(output);
}
