#include <moonbit.h>
#include <stdint.h>
#include "process_handle.h"
#include "process_io.h"
#include "process_status.h"

MOONBIT_FFI_EXPORT
int64_t lunaflux_process_monotonic_now(void) {
  int64_t now = lf_process_now_millis();
  return now < 0 ? -1 : now;
}

static int32_t lf_process_try_io(
  lf_process *process,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int write_mode
) {
  if (process == NULL || process->closed || process->fd < 0) {
    return -LF_PROCESS_INVALID_STATE;
  }
  if (bytes == NULL || offset < 0 || byte_count <= 0 ||
      offset > Moonbit_array_length(bytes) - byte_count) {
    return -LF_PROCESS_FAILED;
  }
  int32_t transferred = 0;
  int32_t status = lf_process_try_fd_io(
    process->fd,
    bytes,
    offset,
    byte_count,
    write_mode,
    &transferred
  );
  return status == LF_PROCESS_OK
    ? transferred
    : (status == LF_PROCESS_PENDING ? 0 : -status);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_try_write(
  lf_process *process,
  uint8_t *source,
  int32_t offset,
  int32_t byte_count
) {
  return lf_process_try_io(process, source, offset, byte_count, 1);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_try_read(
  lf_process *process,
  uint8_t *destination,
  int32_t offset,
  int32_t byte_count
) {
  return lf_process_try_io(process, destination, offset, byte_count, 0);
}
