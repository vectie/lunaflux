#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "../approved_fs_capability/approved_fs_capability.h"
#include "process_status.h"
#include "process_io.h"
#include "process_handle.h"
#include "process_approved_spawn.h"

/* Executable duplication is header-owned so this archive is link-closed. */

#ifndef LF_PROCESS_KILL
#define LF_PROCESS_KILL kill
#endif
#ifndef LF_PROCESS_WAITPID
#define LF_PROCESS_WAITPID waitpid
#endif
#ifndef LF_PROCESS_CLOSE
#define LF_PROCESS_CLOSE close
#endif
#ifndef LF_PROCESS_FCNTL
#define LF_PROCESS_FCNTL fcntl
#endif

static void lf_process_record_exit(lf_process *process, int status);
MOONBIT_FFI_EXPORT int32_t lunaflux_process_close(lf_process *process);

static int32_t lf_process_reap_force(lf_process *process) {
  if (process == NULL || process->reaped || process->pid <= 0) {
    return LF_PROCESS_OK;
  }
  if (LF_PROCESS_KILL(process->pid, SIGKILL) != 0 && errno != ESRCH) {
    return LF_PROCESS_FAILED;
  }
  int status = 0;
  pid_t result;
  do {
    result = LF_PROCESS_WAITPID(process->pid, &status, 0);
  } while (result < 0 && errno == EINTR);
  if (result == process->pid) {
    lf_process_record_exit(process, status);
    return LF_PROCESS_OK;
  }
  if (result < 0 && errno == ECHILD) {
    process->reaped = 1;
    process->pid = -1;
    return LF_PROCESS_FAILED;
  }
  return LF_PROCESS_FAILED;
}

static void lf_process_finalize(void *pointer) {
  lf_process *process = (lf_process *)pointer;
  if (process->fd >= 0) {
    (void)LF_PROCESS_CLOSE(process->fd);
    process->fd = -1;
  }
  (void)lf_process_reap_force(process);
  process->closed = 1;
}

static int lf_duplicate_cloexec_at_least_five(int source) {
  return fcntl(source, F_DUPFD_CLOEXEC, 5);
}

#ifndef SOCK_CLOEXEC
static int lf_set_cloexec(int fd) {
  int flags = fcntl(fd, F_GETFD, 0);
  return flags >= 0 && fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}
#endif

static lf_process *lf_process_allocate(void) {
  lf_process *process = (lf_process *)moonbit_make_external_object(
    lf_process_finalize, sizeof(lf_process)
  );
  process->pid = -1;
  process->fd = -1;
  process->reaped = 1;
  process->closed = 1;
  process->exit_kind = 0;
  process->exit_code = 0;
  return process;
}

static int32_t lf_process_spawn_into(
  lf_process *process,
  lf_approved_executable *approved_executable,
  lf_worker_approved_roots *roots
) {
  if (process == NULL || !process->closed || process->fd >= 0 ||
      !process->reaped || process->pid > 0 || approved_executable == NULL ||
      roots == NULL) {
    return LF_PROCESS_FAILED;
  }

  int approved_fd = -1;
  if (lf_approved_executable_duplicate(
        approved_executable, &approved_fd
      ) != LF_APPROVED_CAPABILITY_OK) {
    return LF_PROCESS_FAILED;
  }

  int32_t model_root = -1;
  int32_t kernel_root = -1;
  int roots_active = 0;
  if (roots != NULL) {
    if (lf_worker_roots_begin(roots, &model_root, &kernel_root) !=
        LF_APPROVED_CAPABILITY_OK) {
      if (approved_fd >= 0) (void)LF_PROCESS_CLOSE(approved_fd);
      return LF_PROCESS_FAILED;
    }
    roots_active = 1;
    struct stat model_info;
    struct stat kernel_info;
    if (fstat(model_root, &model_info) != 0 ||
        fstat(kernel_root, &kernel_info) != 0 ||
        !S_ISDIR(model_info.st_mode) || !S_ISDIR(kernel_info.st_mode)) {
      goto finish;
    }
  }

  int raw_sockets[2] = {-1, -1};
  int sockets[2] = {-1, -1};
#ifdef SOCK_CLOEXEC
  if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, raw_sockets) != 0) {
    goto finish;
  }
#else
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, raw_sockets) != 0) goto finish;
  if (!lf_set_cloexec(raw_sockets[0]) || !lf_set_cloexec(raw_sockets[1])) {
    (void)close(raw_sockets[0]);
    (void)close(raw_sockets[1]);
    goto finish;
  }
#endif
  sockets[0] = lf_duplicate_cloexec_at_least_five(raw_sockets[0]);
  sockets[1] = lf_duplicate_cloexec_at_least_five(raw_sockets[1]);
  (void)close(raw_sockets[0]);
  (void)close(raw_sockets[1]);
  if (sockets[0] < 0 || sockets[1] < 0 || sockets[0] == sockets[1]) {
    if (sockets[0] >= 0) (void)close(sockets[0]);
    if (sockets[1] >= 0) (void)close(sockets[1]);
    goto finish;
  }
  int fd_flags = LF_PROCESS_FCNTL(sockets[0], F_GETFL, 0);
  if (fd_flags < 0 ||
      LF_PROCESS_FCNTL(sockets[0], F_SETFL, fd_flags | O_NONBLOCK) != 0) {
    (void)LF_PROCESS_CLOSE(sockets[0]);
    (void)LF_PROCESS_CLOSE(sockets[1]);
    goto finish;
  }

  pid_t pid = -1;
  int spawned;
  spawned = lf_process_spawn_approved(
    approved_fd, sockets[1], model_root, kernel_root, &pid
  );
  if (approved_fd >= 0) {
    (void)close(approved_fd);
    approved_fd = -1;
  }
  (void)close(sockets[1]);
  if (roots_active) {
    lf_worker_roots_end(roots);
    roots_active = 0;
  }
  if (spawned != 0) {
    (void)close(sockets[0]);
    goto finish;
  }
  process->pid = pid;
  process->fd = sockets[0];
  process->reaped = 0;
  process->closed = 0;
  return LF_PROCESS_OK;
finish:
  if (approved_fd >= 0) (void)LF_PROCESS_CLOSE(approved_fd);
  if (roots_active) {
    lf_worker_roots_end(roots);
  }
  return LF_PROCESS_FAILED;
}

MOONBIT_FFI_EXPORT
lf_process *lunaflux_process_prepare_child(void) {
  return lf_process_allocate();
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_spawn_prepared_with_approved_executable_and_roots(
  lf_process *process,
  lf_approved_executable *executable,
  lf_worker_approved_roots *roots
) {
  if (executable == NULL || roots == NULL) return LF_PROCESS_FAILED;
  return lf_process_spawn_into(process, executable, roots);
}

static int32_t lf_process_io(
  lf_process *process,
  uint8_t *bytes,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis,
  int write_mode
) {
  if (process == NULL || process->closed || process->fd < 0) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (bytes == NULL || offset < 0 || byte_count < 0 ||
      offset > Moonbit_array_length(bytes) - byte_count) {
    return LF_PROCESS_FAILED;
  }
  return lf_process_fd_io_exact(
    process->fd, bytes, offset, byte_count, timeout_millis, write_mode
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_inherited_write_exact(
  uint8_t *source,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis
) {
  if (source == NULL || offset < 0 || byte_count < 0 ||
      offset > Moonbit_array_length(source) - byte_count) {
    return LF_PROCESS_FAILED;
  }
  return lf_process_fd_io_exact(
    1, source, offset, byte_count, timeout_millis, 1
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_inherited_read_exact(
  uint8_t *destination,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis
) {
  if (destination == NULL || offset < 0 || byte_count < 0 ||
      offset > Moonbit_array_length(destination) - byte_count) {
    return LF_PROCESS_FAILED;
  }
  return lf_process_fd_io_exact(
    0, destination, offset, byte_count, timeout_millis, 0
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_write_exact(
  lf_process *process,
  uint8_t *source,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis
) {
  return lf_process_io(
    process, source, offset, byte_count, timeout_millis, 1
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_read_exact(
  lf_process *process,
  uint8_t *destination,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis
) {
  return lf_process_io(
    process, destination, offset, byte_count, timeout_millis, 0
  );
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_shutdown_write(lf_process *process) {
  if (process == NULL || process->closed || process->fd < 0) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (shutdown(process->fd, SHUT_WR) != 0 && errno != ENOTCONN) {
    return LF_PROCESS_FAILED;
  }
  return LF_PROCESS_OK;
}

static void lf_process_record_exit(lf_process *process, int status) {
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
int32_t lunaflux_process_wait(
  lf_process *process,
  int32_t timeout_millis,
  int32_t *kind,
  int32_t *code
) {
  if (process == NULL || process->closed || timeout_millis <= 0) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (process->reaped) {
    *kind = process->exit_kind;
    *code = process->exit_code;
    return LF_PROCESS_OK;
  }
  int64_t now = lf_process_now_millis();
  if (now < 0 || now > INT64_MAX - timeout_millis) {
    return LF_PROCESS_FAILED;
  }
  int64_t deadline = now + timeout_millis;
  for (;;) {
    int status = 0;
    pid_t result = waitpid(process->pid, &status, WNOHANG);
    if (result == process->pid) {
      lf_process_record_exit(process, status);
      *kind = process->exit_kind;
      *code = process->exit_code;
      return LF_PROCESS_OK;
    }
    if (result < 0 && errno != EINTR) {
      return LF_PROCESS_FAILED;
    }
    if (lf_process_now_millis() >= deadline) {
      return LF_PROCESS_TIMEOUT;
    }
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 1000000};
    while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {
    }
  }
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_terminate(lf_process *process) {
  if (process == NULL || process->closed || process->reaped) {
    return LF_PROCESS_INVALID_STATE;
  }
  if (kill(process->pid, SIGTERM) != 0 && errno != ESRCH) {
    return LF_PROCESS_FAILED;
  }
  return LF_PROCESS_OK;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_close(lf_process *process) {
  if (process == NULL) {
    return LF_PROCESS_FAILED;
  }
  if (process->closed && process->fd < 0 && process->reaped) {
    return LF_PROCESS_OK;
  }
  process->closed = 1;
  int32_t result = LF_PROCESS_OK;
  if (process->fd >= 0) {
    (void)shutdown(process->fd, SHUT_RDWR);
    int close_status;
#if defined(__APPLE__)
    do {
      close_status = LF_PROCESS_CLOSE(process->fd);
    } while (close_status != 0 && errno == EINTR);
#else
    close_status = LF_PROCESS_CLOSE(process->fd);
#endif
    if (close_status != 0) {
      result = LF_PROCESS_FAILED;
    }
    /* close consumes our descriptor authority even when EINTR leaves the
       kernel result indeterminate; retrying could close a reused descriptor. */
    process->fd = -1;
  }
  if (lf_process_reap_force(process) != LF_PROCESS_OK) {
    result = LF_PROCESS_FAILED;
  }
  return result;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_is_closed(lf_process *process) {
  return process != NULL && process->closed && process->fd < 0 && process->reaped;
}
