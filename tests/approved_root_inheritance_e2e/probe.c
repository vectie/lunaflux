#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <fcntl.h>
#include <unistd.h>

MOONBIT_FFI_EXPORT
int32_t lunaflux_inheritance_test_open_ambient_fd(void) {
  int source = open("/dev/null", O_RDONLY);
  if (source < 0) return 0;
  if (dup2(source, 9) < 0) {
    (void)close(source);
    return 0;
  }
  if (source != 9) (void)close(source);
  return 1;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_inheritance_test_close_ambient_fd(void) {
  return close(9) == 0;
}
