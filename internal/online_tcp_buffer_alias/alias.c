#include <moonbit.h>

MOONBIT_FFI_EXPORT
moonbit_bytes_t lunaflux_online_tcp_retain_bytes_as_fixed_array(
  moonbit_bytes_t source
) {
  moonbit_incref(source);
  return source;
}
