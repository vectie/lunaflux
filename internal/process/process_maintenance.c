#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>
#include "process_handle.h"
#include "process_status.h"

#ifndef LF_PROCESS_MAINT_WAITPID
#define LF_PROCESS_MAINT_WAITPID waitpid
#endif
#ifndef LF_PROCESS_MAINT_KILL
#define LF_PROCESS_MAINT_KILL kill
#endif
#ifndef LF_PROCESS_MAINT_CLOSE
#define LF_PROCESS_MAINT_CLOSE close
#endif

static void lf_process_maintenance_record_exit(
  lf_process *process,
  int status
) {
  process->reaped = 1;
  process->pid = -1;
  if (WIFEXITED(status)) {
    process->exit_kind = 0;
    process->exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    process->exit_kind = 1;
    process->exit_code = WTERMSIG(status);
  } else {
    process->exit_kind = 2;
    process->exit_code = 0;
  }
}

MOONBIT_FFI_EXPORT
int64_t lunaflux_process_maintenance_try_wait(lf_process *process) {
  if (process == NULL || process->closed) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (process->reaped) {
    return ((int64_t)process->exit_code << 16) |
      ((int64_t)process->exit_kind << 8);
  }
  if (process->pid <= 0) {
    return LF_PROCESS_FAILED;
  }
  int status = 0;
  pid_t owned_pid = process->pid;
  pid_t result = LF_PROCESS_MAINT_WAITPID(owned_pid, &status, WNOHANG);
  if (result == 0 || (result < 0 && errno == EINTR)) {
    return LF_PROCESS_PENDING;
  }
  if (result == owned_pid) {
    lf_process_maintenance_record_exit(process, status);
    return ((int64_t)process->exit_code << 16) |
      ((int64_t)process->exit_kind << 8);
  }
  return LF_PROCESS_FAILED;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_maintenance_signal(
  lf_process *process,
  int32_t signal_kind
) {
  if (process == NULL || process->closed || process->reaped ||
      process->pid <= 0 || (signal_kind != 0 && signal_kind != 1)) {
    return LF_PROCESS_INVALID_STATE;
  }
  int signal_number = signal_kind == 0 ? SIGTERM : SIGKILL;
  if (LF_PROCESS_MAINT_KILL(process->pid, signal_number) != 0 &&
      errno != ESRCH) {
    return LF_PROCESS_FAILED;
  }
  return LF_PROCESS_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_maintenance_close_reaped(lf_process *process) {
  if (process == NULL || !process->reaped || process->pid > 0) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (process->closed && process->fd < 0) {
    return LF_PROCESS_OK;
  }
  process->closed = 1;
  if (process->fd < 0) {
    return LF_PROCESS_OK;
  }
  int owned_fd = process->fd;
  process->fd = -1;
  if (LF_PROCESS_MAINT_CLOSE(owned_fd) != 0) {
    return LF_PROCESS_FAILED;
  }
  return LF_PROCESS_OK;
}
