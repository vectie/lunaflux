#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/wait.h>

static int wait_calls;
static int wait_mode;
static int kill_calls;
static int last_signal;
static int kill_result;
static int close_calls;
static int close_result;

static pid_t fake_waitpid(pid_t pid, int *status, int options) {
  assert(options == WNOHANG);
  wait_calls += 1;
  if (wait_mode == 0) return 0;
  if (wait_mode == 1) {
    errno = EINTR;
    return -1;
  }
  if (wait_mode == 2) {
    *status = 7 << 8;
    return pid;
  }
  if (wait_mode == 3) {
    *status = SIGKILL;
    return pid;
  }
  errno = ECHILD;
  return -1;
}

static int fake_kill(pid_t pid, int signal_number) {
  assert(pid == 47);
  kill_calls += 1;
  last_signal = signal_number;
  if (kill_result != 0) errno = EPERM;
  return kill_result;
}

static int fake_close(int fd) {
  assert(fd == 9);
  close_calls += 1;
  if (close_result != 0) errno = EINTR;
  return close_result;
}

#define LF_PROCESS_MAINT_WAITPID fake_waitpid
#define LF_PROCESS_MAINT_KILL fake_kill
#define LF_PROCESS_MAINT_CLOSE fake_close
#include "process_maintenance.c"

static lf_process live_process(void) {
  lf_process process = {
    .pid = 47,
    .fd = 9,
    .reaped = 0,
    .closed = 0,
    .exit_kind = 0,
    .exit_code = 0,
  };
  return process;
}

int main(void) {
  int64_t result;

  lf_process interrupted = live_process();
  wait_mode = 1;
  wait_calls = 0;
  assert(lunaflux_process_maintenance_try_wait(&interrupted) ==
    LF_PROCESS_PENDING);
  assert(wait_calls == 1);
  assert(!interrupted.reaped && interrupted.pid == 47);

  wait_mode = 0;
  for (int pending_poll = 0; pending_poll < 4096; pending_poll += 1) {
    assert(lunaflux_process_maintenance_try_wait(&interrupted) ==
      LF_PROCESS_PENDING);
    assert(!interrupted.reaped && interrupted.pid == 47 && interrupted.fd == 9);
  }
  assert(wait_calls == 4097);

  wait_mode = 2;
  result = lunaflux_process_maintenance_try_wait(&interrupted);
  assert((result & 0xff) == LF_PROCESS_OK);
  assert(((result >> 8) & 0xff) == 0 && ((result >> 16) & 0xff) == 7);
  assert(interrupted.reaped && interrupted.pid == -1);
  int calls_after_reap = wait_calls;
  assert((lunaflux_process_maintenance_try_wait(&interrupted) & 0xff) ==
    LF_PROCESS_OK);
  assert(wait_calls == calls_after_reap);

  kill_calls = 0;
  assert(lunaflux_process_maintenance_signal(&interrupted, 0) ==
    LF_PROCESS_INVALID_STATE);
  assert(kill_calls == 0);

  lf_process signaled = live_process();
  kill_result = -1;
  assert(lunaflux_process_maintenance_signal(&signaled, 0) ==
    LF_PROCESS_FAILED);
  assert(!signaled.reaped && signaled.pid == 47);
  kill_result = 0;
  assert(lunaflux_process_maintenance_signal(&signaled, 0) == LF_PROCESS_OK);
  assert(kill_calls == 2 && last_signal == SIGTERM);
  assert(lunaflux_process_maintenance_signal(&signaled, 1) == LF_PROCESS_OK);
  assert(kill_calls == 3 && last_signal == SIGKILL);

  lf_process foreign_reap = live_process();
  wait_mode = 4;
  wait_calls = 0;
  assert(lunaflux_process_maintenance_try_wait(&foreign_reap) ==
    LF_PROCESS_FAILED);
  assert(wait_calls == 1);
  assert(!foreign_reap.reaped && foreign_reap.pid == 47);
  wait_mode = 3;
  result = lunaflux_process_maintenance_try_wait(&foreign_reap);
  assert((result & 0xff) == LF_PROCESS_OK);
  assert(((result >> 8) & 0xff) == 1 &&
    ((result >> 16) & 0xff) == SIGKILL);

  lf_process not_reaped = live_process();
  close_calls = 0;
  assert(lunaflux_process_maintenance_close_reaped(&not_reaped) ==
    LF_PROCESS_INVALID_STATE);
  assert(close_calls == 0 && not_reaped.fd == 9);

  close_result = 0;
  assert(lunaflux_process_maintenance_close_reaped(&foreign_reap) ==
    LF_PROCESS_OK);
  assert(close_calls == 1 && foreign_reap.fd == -1 && foreign_reap.closed);
  assert(lunaflux_process_maintenance_close_reaped(&foreign_reap) ==
    LF_PROCESS_OK);
  assert(close_calls == 1);

  lf_process ambiguous_close = live_process();
  ambiguous_close.reaped = 1;
  ambiguous_close.pid = -1;
  close_result = -1;
  assert(lunaflux_process_maintenance_close_reaped(&ambiguous_close) ==
    LF_PROCESS_FAILED);
  assert(ambiguous_close.fd == -1 && ambiguous_close.closed);
  close_result = 0;
  assert(lunaflux_process_maintenance_close_reaped(&ambiguous_close) ==
    LF_PROCESS_OK);
  return 0;
}
