#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include "process_io.h"
#include "process_status.h"

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_inherited_wait(
  int32_t write_mode,
  int32_t timeout_millis
) {
  if ((write_mode != 0 && write_mode != 1) ||
      timeout_millis <= 0 || timeout_millis > 600000) {
    return LF_PROCESS_FAILED;
  }
  int64_t now = lf_process_now_millis();
  if (now < 0 || now > INT64_MAX - timeout_millis) {
    return LF_PROCESS_FAILED;
  }
  int64_t deadline = now + timeout_millis;
  int fd = write_mode ? STDOUT_FILENO : STDIN_FILENO;
  short event = write_mode ? POLLOUT : POLLIN;
  for (;;) {
    now = lf_process_now_millis();
    if (now < 0) return LF_PROCESS_FAILED;
    int64_t remaining = deadline - now;
    if (remaining <= 0) return LF_PROCESS_TIMEOUT;
    int timeout = remaining > INT32_MAX ? INT32_MAX : (int)remaining;
    struct pollfd item = {.fd = fd, .events = event, .revents = 0};
    int result = poll(&item, 1, timeout);
    if (result == 0) return LF_PROCESS_TIMEOUT;
    if (result < 0) {
      if (errno == EINTR) continue;
      return LF_PROCESS_FAILED;
    }
    if ((item.revents & POLLNVAL) != 0) return LF_PROCESS_FAILED;
    if ((item.revents & event) != 0) return LF_PROCESS_OK;
    if (!write_mode && (item.revents & POLLHUP) != 0) {
      // EOF must wake the transactional reader, which authenticates whether
      // the close occurred at a legal frame boundary.
      return LF_PROCESS_OK;
    }
    if ((item.revents & (POLLERR | POLLHUP)) != 0) {
      return LF_PROCESS_CHANNEL_CLOSED;
    }
  }
}

static volatile sig_atomic_t lf_wait_signal_count = 0;

static void lf_wait_test_signal(int signal_number) {
  (void)signal_number;
  lf_wait_signal_count += 1;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_test_inherited_wait(int32_t mode) {
  if (mode < 0 || mode > 4) return LF_PROCESS_FAILED;
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    return LF_PROCESS_FAILED;
  }
  int target = mode == 3 ? STDOUT_FILENO : STDIN_FILENO;
  int saved = dup(target);
  if (saved < 0 || dup2(sockets[0], target) < 0) {
    if (saved >= 0) (void)close(saved);
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    return LF_PROCESS_FAILED;
  }
  (void)close(sockets[0]);
  sockets[0] = -1;
  int32_t status = LF_PROCESS_FAILED;
  struct sigaction previous_action;
  int action_installed = 0;
  if (mode == 0) {
    status = lunaflux_process_inherited_wait(0, 5);
  } else if (mode == 1) {
    (void)close(sockets[1]);
    sockets[1] = -1;
    status = lunaflux_process_inherited_wait(0, 100);
  } else if (mode == 2) {
    uint8_t byte = 1;
    if (send(sockets[1], &byte, 1, 0) == 1) {
      status = lunaflux_process_inherited_wait(0, 100);
    }
  } else if (mode == 3) {
    status = lunaflux_process_inherited_wait(1, 100);
  } else {
    struct sigaction action;
    action.sa_handler = lf_wait_test_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    lf_wait_signal_count = 0;
    if (sigaction(SIGALRM, &action, &previous_action) == 0) {
      action_installed = 1;
    }
    if (action_installed) {
      struct itimerval timer = {
        .it_interval = {.tv_sec = 0, .tv_usec = 0},
        .it_value = {.tv_sec = 0, .tv_usec = 1000},
      };
      if (setitimer(ITIMER_REAL, &timer, NULL) == 0) {
        status = lunaflux_process_inherited_wait(0, 10);
      }
      if (lf_wait_signal_count == 0 || status != LF_PROCESS_TIMEOUT) {
        status = LF_PROCESS_FAILED;
      }
    }
  }
  if (action_installed) {
    (void)sigaction(SIGALRM, &previous_action, NULL);
  }
  if (dup2(saved, target) < 0) status = LF_PROCESS_FAILED;
  (void)close(saved);
  if (sockets[1] >= 0) (void)close(sockets[1]);
  return status;
}
