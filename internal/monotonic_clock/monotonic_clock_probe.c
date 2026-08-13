#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <stdint.h>
#include <time.h>

static int lf_probe_mode = 0;

static int lf_probe_clock_gettime(clockid_t clock_id, struct timespec *value) {
  assert(clock_id == CLOCK_MONOTONIC);
  if (lf_probe_mode == 1) return -1;
  if (lf_probe_mode == 2) {
    value->tv_sec = (time_t)(UINT64_MAX / 1000U + 1U);
    value->tv_nsec = 0;
    return 0;
  }
  if (lf_probe_mode == 3) {
    value->tv_sec = 1;
    value->tv_nsec = 1000000000L;
    return 0;
  }
  value->tv_sec = 12;
  value->tv_nsec = 345678901L;
  return 0;
}

#define LF_MONOTONIC_CLOCK_GETTIME lf_probe_clock_gettime
#include "monotonic_clock.c"

int main(void) {
  uint64_t output = 99U;
  assert(lunaflux_monotonic_clock_now_millis(NULL) ==
         LF_MONOTONIC_CLOCK_UNAVAILABLE);

  lf_probe_mode = 0;
  assert(lunaflux_monotonic_clock_now_millis(&output) ==
         LF_MONOTONIC_CLOCK_OK);
  assert(output == 12345U);

  lf_probe_mode = 1;
  output = 99U;
  assert(lunaflux_monotonic_clock_now_millis(&output) ==
         LF_MONOTONIC_CLOCK_UNAVAILABLE);
  assert(output == 99U);

  lf_probe_mode = 2;
  assert(lunaflux_monotonic_clock_now_millis(&output) ==
         LF_MONOTONIC_CLOCK_OUT_OF_RANGE);
  assert(output == 99U);

  lf_probe_mode = 3;
  assert(lunaflux_monotonic_clock_now_millis(&output) ==
         LF_MONOTONIC_CLOCK_OUT_OF_RANGE);
  assert(output == 99U);
  return 0;
}
