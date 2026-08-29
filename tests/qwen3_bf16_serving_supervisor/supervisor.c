#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <moonbit.h>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define LF_DRAIN_FD 5
#define LF_TRIGGER_TIMEOUT_MILLIS 1800000
#define LF_DRAIN_TIMEOUT_MILLIS 60000
#define LF_EXIT_TIMEOUT_MILLIS 600000

extern char **environ;

static int64_t lf_now_millis(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return -1;
  return (int64_t)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

static char *lf_copy_argument(moonbit_bytes_t bytes, int32_t maximum) {
  if (bytes == NULL) return NULL;
  int32_t length = Moonbit_array_length(bytes);
  if (length <= 0 || length > maximum ||
      memchr(bytes, '\0', (size_t)length) != NULL) return NULL;
  char *copy = (char *)malloc((size_t)length + 1);
  if (copy == NULL) return NULL;
  memcpy(copy, bytes, (size_t)length);
  copy[length] = '\0';
  return copy;
}

static void lf_kill_and_reap(pid_t pid) {
  if (pid <= 0) return;
  (void)kill(pid, SIGKILL);
  while (waitpid(pid, NULL, 0) < 0 && errno == EINTR) {
  }
}

static int lf_wait_direction(int fd, short events, int64_t deadline) {
  for (;;) {
    int64_t now = lf_now_millis();
    if (now < 0 || now >= deadline) return -1;
    int64_t remaining = deadline - now;
    int timeout = remaining > INT32_MAX ? INT32_MAX : (int)remaining;
    struct pollfd descriptor = {.fd = fd, .events = events, .revents = 0};
    int status = poll(&descriptor, 1, timeout);
    if (status > 0 && (descriptor.revents & events) != 0) return 0;
    if (status > 0 && (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
      return -1;
    if (status == 0) return -1;
    if (status < 0 && errno != EINTR) return -1;
  }
}

static int lf_write_exact(int fd, const uint8_t *source, size_t length) {
  int64_t now = lf_now_millis();
  if (now < 0) return -1;
  int64_t deadline = now + LF_DRAIN_TIMEOUT_MILLIS;
  size_t offset = 0;
  while (offset < length) {
    if (lf_wait_direction(fd, POLLOUT, deadline) != 0) return -1;
    ssize_t count = send(fd, source + offset, length - offset, MSG_NOSIGNAL);
    if (count > 0) offset += (size_t)count;
    else if (count < 0 && errno != EINTR && errno != EAGAIN &&
             errno != EWOULDBLOCK) return -1;
  }
  return 0;
}

static int lf_read_exact(int fd, uint8_t *destination, size_t length) {
  int64_t now = lf_now_millis();
  if (now < 0) return -1;
  int64_t deadline = now + LF_DRAIN_TIMEOUT_MILLIS;
  size_t offset = 0;
  while (offset < length) {
    if (lf_wait_direction(fd, POLLIN, deadline) != 0) return -1;
    ssize_t count = recv(fd, destination + offset, length - offset, 0);
    if (count > 0) offset += (size_t)count;
    else if (count == 0) return -1;
    else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
      return -1;
  }
  return 0;
}

static int lf_trigger_ready(const char *path) {
  struct stat value;
  if (lstat(path, &value) != 0) return errno == ENOENT ? 0 : -1;
  return S_ISREG(value.st_mode) && value.st_nlink == 1 ? 1 : -1;
}

static int lf_wait_for_trigger(pid_t pid, const char *trigger, int *status) {
  int64_t now = lf_now_millis();
  if (now < 0) return -1;
  int64_t deadline = now + LF_TRIGGER_TIMEOUT_MILLIS;
  while (lf_now_millis() < deadline) {
    pid_t observed = waitpid(pid, status, WNOHANG);
    if (observed == pid) return -2;
    if (observed < 0 && errno != EINTR) return -1;
    int ready = lf_trigger_ready(trigger);
    if (ready != 0) return ready;
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 100000000};
    while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {
    }
  }
  return -1;
}

static int lf_wait_for_exit(pid_t pid, int *status) {
  int64_t now = lf_now_millis();
  if (now < 0) return -1;
  int64_t deadline = now + LF_EXIT_TIMEOUT_MILLIS;
  while (lf_now_millis() < deadline) {
    pid_t observed = waitpid(pid, status, WNOHANG);
    if (observed == pid) return 0;
    if (observed < 0 && errno != EINTR) return -1;
    struct timespec pause = {.tv_sec = 0, .tv_nsec = 100000000};
    while (nanosleep(&pause, &pause) != 0 && errno == EINTR) {
    }
  }
  return -1;
}

static pid_t lf_spawn(
  const char *executable,
  const char *launch,
  const char *stdout_path,
  const char *stderr_path,
  int *parent_socket
) {
  int output = open(stdout_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (output < 0) return -1;
  int errors = open(stderr_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (errors < 0) {
    (void)close(output);
    return -1;
  }
  int sockets[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) != 0) {
    (void)close(output);
    (void)close(errors);
    return -1;
  }
  if (fcntl(sockets[0], F_SETFD, FD_CLOEXEC) != 0 ||
      fcntl(sockets[1], F_SETFD, FD_CLOEXEC) != 0) {
    (void)close(output);
    (void)close(errors);
    (void)close(sockets[0]);
    (void)close(sockets[1]);
    return -1;
  }
  posix_spawn_file_actions_t actions;
  posix_spawnattr_t attributes;
  int setup = posix_spawn_file_actions_init(&actions);
  int actions_initialized = setup == 0;
  int attributes_initialized = 0;
  if (setup == 0) {
    setup = posix_spawnattr_init(&attributes);
    attributes_initialized = setup == 0;
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_addopen(
      &actions, 0, "/dev/null", O_RDONLY, 0
    );
  }
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, output, 1);
  if (setup == 0) setup = posix_spawn_file_actions_adddup2(&actions, errors, 2);
  if (setup == 0) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[0]);
  }
  if (setup == 0) {
    setup = posix_spawn_file_actions_adddup2(&actions, sockets[1], LF_DRAIN_FD);
  }
  if (setup == 0 && output != 1 && output != LF_DRAIN_FD) {
    setup = posix_spawn_file_actions_addclose(&actions, output);
  }
  if (setup == 0 && errors != 2 && errors != LF_DRAIN_FD) {
    setup = posix_spawn_file_actions_addclose(&actions, errors);
  }
  if (setup == 0 && sockets[1] != LF_DRAIN_FD) {
    setup = posix_spawn_file_actions_addclose(&actions, sockets[1]);
  }
  sigset_t empty_mask;
  short flags = 0;
  if (setup == 0 && sigemptyset(&empty_mask) != 0) {
    setup = errno;
  }
  if (setup == 0) {
    setup = posix_spawnattr_setsigmask(&attributes, &empty_mask);
    flags |= POSIX_SPAWN_SETSIGMASK;
  }
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
  flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
  if (setup == 0) setup = posix_spawnattr_setflags(&attributes, flags);
  pid_t pid = -1;
  char *const arguments[] = {
    (char *)executable, (char *)"run", (char *)launch, NULL,
  };
  if (setup == 0) {
    setup = posix_spawn(
      &pid, executable, &actions, &attributes, arguments, environ
    );
  }
  if (actions_initialized) (void)posix_spawn_file_actions_destroy(&actions);
  if (attributes_initialized) (void)posix_spawnattr_destroy(&attributes);
  (void)close(output);
  (void)close(errors);
  (void)close(sockets[1]);
  if (setup != 0 || pid < 0) {
    (void)close(sockets[0]);
    return -1;
  }
  *parent_socket = sockets[0];
  return pid;
}

MOONBIT_FFI_EXPORT
int32_t lunaflux_qwen3_supervise_runtime(
  moonbit_bytes_t executable_bytes,
  moonbit_bytes_t launch_bytes,
  moonbit_bytes_t stdout_bytes,
  moonbit_bytes_t stderr_bytes,
  moonbit_bytes_t trigger_bytes,
  int32_t *child_pid,
  int32_t *drain_acknowledged,
  int32_t *child_exit_code
) {
  if (child_pid == NULL || drain_acknowledged == NULL ||
      child_exit_code == NULL) return 1;
  *child_pid = -1;
  *drain_acknowledged = 0;
  *child_exit_code = -1;
  char *executable = lf_copy_argument(executable_bytes, 4096);
  char *launch = lf_copy_argument(launch_bytes, 8192);
  char *stdout_path = lf_copy_argument(stdout_bytes, 4096);
  char *stderr_path = lf_copy_argument(stderr_bytes, 4096);
  char *trigger = lf_copy_argument(trigger_bytes, 4096);
  if (executable == NULL || launch == NULL || stdout_path == NULL ||
      stderr_path == NULL || trigger == NULL || executable[0] != '/' ||
      stdout_path[0] != '/' || stderr_path[0] != '/' || trigger[0] != '/') {
    free(executable); free(launch); free(stdout_path); free(stderr_path);
    free(trigger);
    return 2;
  }
  int channel = -1;
  pid_t pid = lf_spawn(executable, launch, stdout_path, stderr_path, &channel);
  *child_pid = pid > INT32_MAX ? -1 : (int32_t)pid;
  free(executable); free(launch); free(stdout_path); free(stderr_path);
  if (pid <= 0) {
    free(trigger);
    return 3;
  }
  int status = 0;
  int trigger_status = lf_wait_for_trigger(pid, trigger, &status);
  free(trigger);
  if (trigger_status != 1) {
    (void)close(channel);
    if (trigger_status == -2 && WIFEXITED(status)) {
      *child_exit_code = WEXITSTATUS(status);
    } else if (trigger_status == -2 && WIFSIGNALED(status)) {
      *child_exit_code = -WTERMSIG(status);
    }
    if (trigger_status != -2) lf_kill_and_reap(pid);
    return trigger_status == -2 ? 4 : 5;
  }
  static const uint8_t command[8] = {'L', 'F', 'D', '1', 'D', 'R', 'N', '\n'};
  static const uint8_t expected[8] = {'L', 'F', 'D', '1', 'A', 'C', 'K', '\n'};
  uint8_t response[8] = {0};
  if (lf_write_exact(channel, command, sizeof(command)) != 0 ||
      lf_read_exact(channel, response, sizeof(response)) != 0 ||
      memcmp(response, expected, sizeof(expected)) != 0) {
    (void)close(channel);
    lf_kill_and_reap(pid);
    return 6;
  }
  *drain_acknowledged = 1;
  (void)shutdown(channel, SHUT_RDWR);
  (void)close(channel);
  if (lf_wait_for_exit(pid, &status) != 0) {
    lf_kill_and_reap(pid);
    return 7;
  }
  if (!WIFEXITED(status)) return 8;
  *child_exit_code = WEXITSTATUS(status);
  return *child_exit_code == 0 ? 0 : 9;
}
