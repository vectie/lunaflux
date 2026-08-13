#ifndef LUNAFLUX_PROCESS_IO_H
#define LUNAFLUX_PROCESS_IO_H

#include <stdint.h>
#include <sys/types.h>

int64_t lf_process_now_millis(void);

int32_t lf_process_fd_io_exact(
  int fd,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis,
  int write_mode
);

int32_t lf_process_status_from_io_result(
  ssize_t count,
  int error_number,
  int write_mode
);

int32_t lf_process_try_fd_io(
  int fd,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int write_mode,
  int32_t *transferred
);

#endif
