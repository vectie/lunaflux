#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

enum {
  LF_PROCESS_OK = 0,
  LF_PROCESS_INVALID_STATE = 1,
  LF_PROCESS_TIMEOUT = 2,
  LF_PROCESS_CHANNEL_CLOSED = 3,
  LF_PROCESS_FAILED = 4
};

typedef struct lf_process {
  pid_t pid;
  int fd;
  int reaped;
  int closed;
  int exit_kind;
  int exit_code;
} lf_process;

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

static int64_t lf_now_millis(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
    return -1;
  }
  return (int64_t)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

static int32_t lf_poll_fd(int fd, short events, int64_t deadline) {
  for (;;) {
    int64_t now = lf_now_millis();
    if (now < 0) {
      return LF_PROCESS_FAILED;
    }
    int64_t remaining = deadline - now;
    if (remaining <= 0) {
      return LF_PROCESS_TIMEOUT;
    }
    int timeout = remaining > INT32_MAX ? INT32_MAX : (int)remaining;
    struct pollfd item = {.fd = fd, .events = events, .revents = 0};
    int result = poll(&item, 1, timeout);
    if (result > 0) {
      if ((item.revents & events) != 0) {
        return LF_PROCESS_OK;
      }
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

MOONBIT_FFI_EXPORT
lf_process *lunaflux_process_spawn(moonbit_bytes_t path, int32_t *status) {
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

  int32_t path_length = Moonbit_array_length(path);
  if (path_length <= 0 || memchr(path, '\0', (size_t)path_length) != NULL) {
    return process;
  }
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    return process;
  }
  (void)fcntl(sockets[0], F_SETFD, FD_CLOEXEC);
  (void)fcntl(sockets[1], F_SETFD, FD_CLOEXEC);

  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    return process;
  }
  if (posix_spawnattr_init(&attributes) != 0) {
    (void)posix_spawn_file_actions_destroy(&actions);
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    return process;
  }
  int setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 0);
  if (setup == 0) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], 1);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[0]);
  }
  if (setup == 0 && sockets[1] != 0 && sockets[1] != 1) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[1]);
  }
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  short spawn_flags = POSIX_SPAWN_CLOEXEC_DEFAULT;
  if (setup == 0) {
    setup = posix_spawnattr_setflags(&attributes, spawn_flags);
  }
#endif
  pid_t pid = -1;
  char *const argv[] = {(char *)path, NULL};
  int spawned = setup == 0
    ? posix_spawn(&pid, (char *)path, &actions, &attributes, argv, environ)
    : setup;
  (void)posix_spawn_file_actions_destroy(&actions);
  (void)posix_spawnattr_destroy(&attributes);
  (void)close(sockets[1]);
  if (spawned != 0) {
    (void)close(sockets[0]);
    return process;
  }
  int fd_flags = fcntl(sockets[0], F_GETFL, 0);
  if (fd_flags < 0 ||
      fcntl(sockets[0], F_SETFL, fd_flags | O_NONBLOCK) != 0) {
    (void)close(sockets[0]);
    (void)kill(pid, SIGKILL);
    while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
    }
    return process;
  }
  process->pid = pid;
  process->fd = sockets[0];
  process->reaped = 0;
  process->closed = 0;
  *status = LF_PROCESS_OK;
  return process;
}

static int32_t lf_fd_io(
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
  int64_t now = lf_now_millis();
  if (now < 0 || now > INT64_MAX - timeout_millis) {
    return LF_PROCESS_FAILED;
  }
  int64_t deadline = now + timeout_millis;
  int32_t cursor = 0;
  while (cursor < byte_count) {
    int32_t poll_status = lf_poll_fd(
      fd, write_mode ? POLLOUT : POLLIN, deadline
    );
    if (poll_status != LF_PROCESS_OK) {
      return poll_status;
    }
    ssize_t count = write_mode
      ? send(
          fd,
          bytes + offset + cursor,
          (size_t)(byte_count - cursor),
          MSG_NOSIGNAL
        )
      : recv(
          fd,
          bytes + offset + cursor,
          (size_t)(byte_count - cursor),
          0
        );
    if (count > 0) {
      cursor += (int32_t)count;
    } else if (count == 0) {
      return LF_PROCESS_CHANNEL_CLOSED;
    } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      return errno == EPIPE ? LF_PROCESS_CHANNEL_CLOSED : LF_PROCESS_FAILED;
    }
  }
  return LF_PROCESS_OK;
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
  return lf_fd_io(
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
  return lf_fd_io(1, source, offset, byte_count, timeout_millis, 1);
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_process_inherited_read_exact(
  uint8_t *destination,
  int32_t offset,
  int32_t byte_count,
  int32_t timeout_millis
) {
  return lf_fd_io(0, destination, offset, byte_count, timeout_millis, 0);
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
  int64_t now = lf_now_millis();
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
    if (lf_now_millis() >= deadline) {
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
