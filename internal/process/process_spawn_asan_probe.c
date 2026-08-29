#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int probe_kill_failure = 0;
static int probe_wait_failure = 0;
static int probe_close_evidence = 0;

static int probe_kill(pid_t pid, int signal_number) {
  if (probe_kill_failure) {
    errno = EPERM;
    return -1;
  }
  return kill(pid, signal_number);
}

static pid_t probe_waitpid(pid_t pid, int *status, int options) {
  if (probe_wait_failure) {
    errno = EIO;
    return -1;
  }
  return waitpid(pid, status, options);
}

static int probe_close(int fd) {
  int result = close(fd);
  if (probe_close_evidence) {
    errno = EIO;
    return -1;
  }
  return result;
}

#define LF_PROCESS_KILL probe_kill
#define LF_PROCESS_WAITPID probe_waitpid
#define LF_PROCESS_CLOSE probe_close

#include "process.c"

void *moonbit_make_external_object(
  void (*finalize)(void *self), uint32_t payload_size
) {
  (void)finalize;
  return calloc(1, payload_size);
}

void moonbit_incref(void *object) { (void)object; }

int main(void) {
  lf_process *invalid = lunaflux_process_prepare_child();
  assert(lunaflux_process_spawn_prepared_with_approved_executable_and_roots(
    invalid, NULL, NULL
  ) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(invalid));
  assert(lunaflux_process_close(invalid) == LF_PROCESS_OK);

  lf_process *kill_failed = lunaflux_process_prepare_child();
  kill_failed->pid = 999999;
  kill_failed->reaped = 0;
  kill_failed->closed = 0;
  probe_kill_failure = 1;
  assert(lunaflux_process_close(kill_failed) == LF_PROCESS_FAILED);
  assert(!lunaflux_process_is_closed(kill_failed));
  probe_kill_failure = 0;
  assert(lunaflux_process_close(kill_failed) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(kill_failed));

  lf_process *wait_failed = lunaflux_process_prepare_child();
  wait_failed->pid = 999998;
  wait_failed->reaped = 0;
  wait_failed->closed = 0;
  probe_wait_failure = 1;
  assert(lunaflux_process_close(wait_failed) == LF_PROCESS_FAILED);
  assert(!lunaflux_process_is_closed(wait_failed));
  probe_wait_failure = 0;
  assert(lunaflux_process_close(wait_failed) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(wait_failed));

  int pipe_fds[2];
  assert(pipe(pipe_fds) == 0);
  lf_process *close_evidence = lunaflux_process_prepare_child();
  close_evidence->fd = pipe_fds[0];
  close_evidence->closed = 0;
  probe_close_evidence = 1;
  assert(lunaflux_process_close(close_evidence) == LF_PROCESS_FAILED);
  assert(lunaflux_process_is_closed(close_evidence));
  probe_close_evidence = 0;
  assert(lunaflux_process_close(close_evidence) == LF_PROCESS_OK);
  assert(close(pipe_fds[1]) == 0);
  return 0;
}
