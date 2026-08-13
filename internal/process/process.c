#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
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
extern char **environ;
static void lf_process_reap_force(lf_process *process) {
  if (process == NULL || process->reaped || process->pid <= 0) {
    return;
  }
  (void)kill(process->pid, SIGKILL);
  int status = 0;
  while (waitpid(process->pid, &status, 0) < 0 && errno == EINTR) {
  }
  process->reaped = 1;
}

static void lf_process_finalize(void *pointer) {
  lf_process *process = (lf_process *)pointer;
  if (process->fd >= 0) {
    (void)close(process->fd);
    process->fd = -1;
  }
  lf_process_reap_force(process);
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

static lf_process *lf_process_spawn_impl(
  moonbit_bytes_t path,
  lf_worker_approved_roots *roots,
  int32_t *status
) {
  lf_process *process = (lf_process *)moonbit_make_external_object(
    lf_process_finalize, sizeof(lf_process)
  );
  process->pid = -1;
  process->fd = -1;
  process->reaped = 1;
  process->closed = 1;
  process->exit_kind = 0;
  process->exit_code = 0;
  *status = LF_PROCESS_FAILED;

  int32_t model_root = -1;
  int32_t kernel_root = -1;
  int roots_active = 0;
  if (roots != NULL) {
    if (lf_worker_roots_begin(roots, &model_root, &kernel_root) !=
        LF_APPROVED_CAPABILITY_OK) {
      return process;
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

  int32_t path_length = Moonbit_array_length(path);
  if (path_length <= 0 || memchr(path, '\0', (size_t)path_length) != NULL) {
    goto finish;
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

  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    goto finish;
  }
  if (posix_spawnattr_init(&attributes) != 0) {
    (void)posix_spawn_file_actions_destroy(&actions);
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    goto finish;
  }
  int setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 0);
  if (setup == 0) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 1);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 2);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclose(&actions, 2);
  }
  if (setup == 0 && roots == NULL) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 3);
  }
  if (setup == 0 && roots == NULL) {
    setup = posix_spawn_file_actions_addclose(&actions, 3);
  }
  if (setup == 0 && roots == NULL) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 4);
  }
  if (setup == 0 && roots == NULL) {
    setup = posix_spawn_file_actions_addclose(&actions, 4);
  }
  if (setup == 0 && roots != NULL) {
    setup = posix_spawn_file_actions_adddup2(&actions, model_root, 3);
  }
  if (setup == 0 && roots != NULL) {
    setup = posix_spawn_file_actions_adddup2(&actions, kernel_root, 4);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[0]);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[1]);
  }
  if (setup == 0 && roots != NULL && model_root != sockets[0] &&
      model_root != sockets[1]) {
    setup = posix_spawn_file_actions_addclose(&actions, model_root);
  }
  if (setup == 0 && roots != NULL && kernel_root != sockets[0] &&
      kernel_root != sockets[1]) {
    setup = posix_spawn_file_actions_addclose(&actions, kernel_root);
  }
#if defined(__GLIBC__)
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclosefrom_np(&actions, 5);
  }
#elif !defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
#error "fixed-FD spawn requires closefrom actions or CLOEXEC-default spawn"
#endif
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  short spawn_flags = POSIX_SPAWN_CLOEXEC_DEFAULT;
  if (setup == 0) {
    setup = posix_spawnattr_setflags(&attributes, spawn_flags);
  }
#endif
  pid_t pid = -1;
  char *const argv[] = {
    roots == NULL ? (char *)path : (char *)"lunaflux-worker", NULL
  };
  char *const sanitized_environment[] = {NULL};
  char *const *spawn_environment = roots == NULL
    ? environ
    : sanitized_environment;
  int spawned = setup == 0
    ? posix_spawn(
        &pid,
        (char *)path,
        &actions,
        &attributes,
        argv,
        spawn_environment
      )
    : setup;
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  (void)close(sockets[1]);
  if (roots_active) {
    lf_worker_roots_end(roots);
    roots_active = 0;
  }
  if (spawned != 0) {
    (void)close(sockets[0]);
    goto finish;
  }
  int fd_flags = fcntl(sockets[0], F_GETFL, 0);
  if (fd_flags < 0 ||
      fcntl(sockets[0], F_SETFL, fd_flags | O_NONBLOCK) != 0) {
    (void)close(sockets[0]);
    (void)kill(pid, SIGKILL);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
    }
    goto finish;
  }
  process->pid = pid;
  process->fd = sockets[0];
  process->reaped = 0;
  process->closed = 0;
  *status = LF_PROCESS_OK;
finish:
  if (roots_active) {
    lf_worker_roots_end(roots);
  }
  return process;
}

MOONBIT_FFI_EXPORT
lf_process *lunaflux_process_spawn(moonbit_bytes_t path, int32_t *status) {
  return lf_process_spawn_impl(path, NULL, status);
}

MOONBIT_FFI_EXPORT
lf_process *lunaflux_process_spawn_with_approved_roots(
  moonbit_bytes_t path,
  lf_worker_approved_roots *roots,
  int32_t *status
) {
  return lf_process_spawn_impl(path, roots, status);
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
  if (process->closed) {
    return LF_PROCESS_OK;
  }
  process->closed = 1;
  int32_t result = LF_PROCESS_OK;
  if (process->fd >= 0) {
    (void)shutdown(process->fd, SHUT_RDWR);
    if (close(process->fd) != 0 && errno != EINTR) {
      result = LF_PROCESS_FAILED;
    }
    process->fd = -1;
  }
  lf_process_reap_force(process);
  return result;
}
