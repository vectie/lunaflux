#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <poll.h>
#include <stdint.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include "process_status.h"

static int64_t lf_child_now_millis(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1;
  return (int64_t)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_inherited_expect_clean_eof(int32_t timeout_millis) {
  if (timeout_millis <= 0) return LF_PROCESS_FAILED;
  int64_t now = lf_child_now_millis();
  if (now < 0 || now > INT64_MAX - timeout_millis) {
    return LF_PROCESS_FAILED;
  }
  int64_t deadline = now + timeout_millis;
  for (;;) {
    now = lf_child_now_millis();
    if (now < 0) return LF_PROCESS_FAILED;
    int64_t remaining = deadline - now;
    if (remaining <= 0) return LF_PROCESS_TIMEOUT;
    int timeout = remaining > INT32_MAX ? INT32_MAX : (int)remaining;
    struct pollfd item = {.fd = 0, .events = POLLIN, .revents = 0};
    int result = poll(&item, 1, timeout);
    if (result == 0) return LF_PROCESS_TIMEOUT;
    if (result < 0) {
      if (errno == EINTR) continue;
      return LF_PROCESS_FAILED;
    }
    if ((item.revents & POLLNVAL) != 0) return LF_PROCESS_FAILED;
    if ((item.revents & (POLLIN | POLLHUP | POLLERR)) != 0) {
      uint8_t byte = 0;
      ssize_t count = recv(0, &byte, 1, 0);
      if (count == 0) return LF_PROCESS_OK;
      if (count > 0) return LF_PROCESS_FAILED;
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      return LF_PROCESS_FAILED;
    }
  }
}

MOONBIT_FFI_EXPORT
_Noreturn void lunaflux_process_exit_failure(void) {
  _exit(1);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_test_expect_clean_eof(int32_t mode) {
  if (mode < 0 || mode > 3) return LF_PROCESS_FAILED;
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    return LF_PROCESS_FAILED;
  }
  int saved_stdin = dup(0);
  if (saved_stdin < 0 || dup2(sockets[0], 0) < 0) {
    if (saved_stdin >= 0) (void)close(saved_stdin);
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    return LF_PROCESS_FAILED;
  }
  (void)close(sockets[0]);
  if (mode == 0) {
    (void)close(sockets[1]);
    sockets[1] = -1;
  } else if (mode == 1) {
    uint8_t byte = 1;
    (void)send(sockets[1], &byte, 1, 0);
    (void)close(sockets[1]);
    sockets[1] = -1;
  } else if (mode == 2) {
    uint8_t prefix[3] = {1, 0, 0};
    (void)send(sockets[1], prefix, sizeof(prefix), 0);
    (void)close(sockets[1]);
    sockets[1] = -1;
  }
  int32_t status = lunaflux_process_inherited_expect_clean_eof(
    mode == 3 ? 5 : 1000
  );
  if (sockets[1] >= 0) (void)close(sockets[1]);
  if (dup2(saved_stdin, 0) < 0) status = LF_PROCESS_FAILED;
  (void)close(saved_stdin);
  return status;
}
