#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <poll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <time.h>
#include "process_io.h"
#include "process_status.h"

int64_t lf_process_now_millis(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1;
  return (int64_t)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

static int32_t lf_process_poll_fd(
  int fd,
  short events,
  int64_t deadline
) {
  for (;;) {
    int64_t now = lf_process_now_millis();
    if (now < 0) return LF_PROCESS_FAILED;
    int64_t remaining = deadline - now;
    if (remaining <= 0) return LF_PROCESS_TIMEOUT;
    int timeout = remaining > INT32_MAX ? INT32_MAX : (int)remaining;
    struct pollfd item = {.fd = fd, .events = events, .revents = 0};
    int result = poll(&item, 1, timeout);
    if (result > 0) {
      if ((item.revents & events) != 0) return LF_PROCESS_OK;
      if ((item.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        return LF_PROCESS_CHANNEL_CLOSED;
      }
    } else if (result == 0) {
      return LF_PROCESS_TIMEOUT;
    } else if (errno != EINTR) {
      return LF_PROCESS_FAILED;
    }
  }
}

int32_t lf_process_status_from_io_result(
  ssize_t count,
  int error_number,
  int write_mode
) {
  if (count > 0) return LF_PROCESS_OK;
  if (count == 0) return LF_PROCESS_CHANNEL_CLOSED;
  if (error_number == EINTR || error_number == EAGAIN ||
      error_number == EWOULDBLOCK) {
    return LF_PROCESS_PENDING;
  }
  if (write_mode && error_number == EPIPE) {
    return LF_PROCESS_CHANNEL_CLOSED;
  }
  return LF_PROCESS_FAILED;
}

int32_t lf_process_try_fd_io(
  int fd,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int write_mode,
  int32_t *transferred
) {
  if (fd < 0 || bytes == NULL || offset < 0 || byte_count <= 0 ||
      transferred == NULL) {
    return LF_PROCESS_FAILED;
  }
  *transferred = 0;
  ssize_t count = write_mode
    ? send(
        fd,
        bytes + offset,
        (size_t)byte_count,
        MSG_NOSIGNAL | MSG_DONTWAIT
      )
    : recv(fd, bytes + offset, (size_t)byte_count, MSG_DONTWAIT);
  int32_t status = lf_process_status_from_io_result(
    count, count < 0 ? errno : 0, write_mode
  );
  if (status == LF_PROCESS_OK) *transferred = (int32_t)count;
  return status;
}

int32_t lf_process_fd_io_exact(
  int fd,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis,
  int write_mode
) {
  if (offset < 0 || byte_count < 0 || timeout_millis <= 0) {
    return LF_PROCESS_FAILED;
  }
  int64_t now = lf_process_now_millis();
  if (now < 0 || now > INT64_MAX - timeout_millis) {
    return LF_PROCESS_FAILED;
  }
  int64_t deadline = now + timeout_millis;
  int32_t cursor = 0;
  while (cursor < byte_count) {
    int32_t poll_status = lf_process_poll_fd(
      fd, write_mode ? POLLOUT : POLLIN, deadline
    );
    if (poll_status != LF_PROCESS_OK) return poll_status;
    int32_t transferred = 0;
    int32_t io_status = lf_process_try_fd_io(
      fd,
      bytes,
      offset + cursor,
      byte_count - cursor,
      write_mode,
      &transferred
    );
    if (io_status == LF_PROCESS_OK) {
      cursor += transferred;
    } else if (io_status != LF_PROCESS_PENDING) {
      return io_status;
    }
  }
  return LF_PROCESS_OK;
}
