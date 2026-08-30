#include <moonbit.h>

#include <stdint.h>
#include <stdio.h>

MOONBIT_FFI_EXPORT
int32_t lunaflux_native_stdout_flush(void) {
  return fflush(stdout) == 0 ? 0 : 1;
}
